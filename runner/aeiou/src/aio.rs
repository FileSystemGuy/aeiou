//! The `libaio` backends: the kernel AIO system calls (`io_setup`, `io_submit`,
//! `io_getevents`; what the libaio library wraps and what fio and vendors mean by "AIO") as
//! the engine of the event loop of `uring.rs`: one loop per thread, many actors per loop, one
//! AIO context per loop. The op stream and the fingerprint are what the `sync` sink produces.
//!
//! The kernel AIO interface carries `pread`, `pwrite`, `fsync`, and `fdatasync` (and a poll,
//! which is how a barrier release wakes the loop: `IOCB_CMD_POLL` on the loop's eventfd,
//! Linux 4.18). Every other op (`open`, `stat`, `close`, …) has no AIO form and is issued
//! inline with the blocking backend, which blocks the loop and every actor on it for that
//! long: what an AIO application has to do too, and the reason the interface suits a few big
//! files better than many small ones. Reads and writes go at their effective offset, as under
//! `io_uring`; requests queue as the loop advances its tasks and go to the kernel in one
//! `io_submit` per turn of the loop.
//!
//! The interface is asynchronous only for `O_DIRECT` (`libaio-direct`): a buffered read is
//! carried out inside `io_submit`, which returns when the data is in the buffer. The report
//! says how long the loops spent inside `io_submit` so that this is seen, not assumed.
//!
//! A context holds `--aio-depth` requests (`io_setup`'s `nr_events`; the sum over all
//! contexts on the host is bounded by `fs.aio-max-nr`). A full context returns `EAGAIN` from
//! `io_submit`; the loop then reaps a completion and submits again, and counts it.

use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use crate::backend::{Backend, ALIGN};
use crate::eval::Model;
use crate::run::{fill, issue_blocking, round_out, AioReport, Ring, Shared};
use crate::uring::{spread, Engine, Issued, Loop, Pool, TaskIo, EFD};
use crate::vm::{Op, OpKind};

/// `nr_events` of a loop's context unless `--aio-depth` says otherwise.
pub const DEPTH: u32 = 256;

const IOCB_CMD_PREAD: u16 = 0;
const IOCB_CMD_PWRITE: u16 = 1;
const IOCB_CMD_FSYNC: u16 = 2;
const IOCB_CMD_FDSYNC: u16 = 3;
const IOCB_CMD_POLL: u16 = 5;

/// `struct iocb` of `linux/aio_abi.h` (the little-endian field order of `aio_key` and
/// `aio_rw_flags`).
#[cfg(target_endian = "little")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Iocb {
    data: u64,
    key: u32,
    rw_flags: u32,
    opcode: u16,
    reqprio: i16,
    fildes: u32,
    buf: u64,
    nbytes: u64,
    offset: i64,
    reserved2: u64,
    flags: u32,
    resfd: u32,
}

/// `struct io_event`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IoEvent {
    data: u64,
    obj: u64,
    res: i64,
    res2: i64,
}

/// Run this host's instances over the event loops; returns what the contexts did.
pub(crate) fn run(model: &'static Model<'static>, sh: &Arc<Shared>, counts: &[(&'static str, i64)], ranges: &[(i64, i64)]) -> Result<AioReport> {
    let per = spread(sh, counts, ranges);
    if per.is_empty() {
        return Ok(AioReport::default());
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = per
            .into_iter()
            .enumerate()
            .map(|(i, insts)| {
                let sh = sh.clone();
                std::thread::Builder::new()
                    .name(format!("libaio loop {i}"))
                    .spawn_scoped(s, move || {
                        let r = AioIo::new(&sh).and_then(|io| Loop::with(sh.clone(), i, io)).and_then(|mut l| l.run(model, insts).map(|_| l.io.report.clone()));
                        if r.is_err() {
                            sh.aborted.store(true, Ordering::Relaxed);
                        }
                        r
                    })
                    .expect("spawn event loop")
            })
            .collect();
        let mut first: Option<anyhow::Error> = None;
        let mut total = AioReport::default();
        for h in handles {
            match h.join() {
                Ok(Ok(r)) => total.merge(&r),
                Ok(Err(e)) => {
                    first.get_or_insert(e);
                }
                Err(_) => {
                    first.get_or_insert(anyhow!("an event loop panicked"));
                }
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(total),
        }
    })
}

/// A loop's AIO context and what issuing needs.
pub(crate) struct AioIo {
    ctx: libc::c_ulong,
    inline: Box<dyn Backend>,
    rbuf: Ring,
    wbuf: Ring,
    pool: Pool,
    /// Requests made this turn of the loop, not yet submitted.
    queue: Vec<Iocb>,
    /// Completions reaped while making room in a full context, and requests `io_submit`
    /// refused, for the next `wait`.
    ready: Vec<(u64, i32)>,
    events: Vec<IoEvent>,
    /// Requests the kernel holds (the eventfd poll included).
    in_kernel: usize,
    in_flight: usize,
    /// The loop's eventfd, once a poll on it has been asked for.
    efd: RawFd,
    pub(crate) report: AioReport,
}

impl AioIo {
    fn new(sh: &Shared) -> Result<Self> {
        let depth = if sh.opts.aio_depth == 0 { DEPTH } else { sh.opts.aio_depth };
        let mut ctx: libc::c_ulong = 0;
        // SAFETY: `ctx` is zero as io_setup requires and outlives the call
        if unsafe { libc::syscall(libc::SYS_io_setup, depth as libc::c_long, &mut ctx as *mut libc::c_ulong) } < 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| format!("io_setup({depth}) (EAGAIN: the contexts on this host exceed fs.aio-max-nr; lower --aio-depth or --threads, or raise the sysctl)"));
        }
        let buf = sh.opts.buffer_bytes;
        Ok(AioIo {
            ctx,
            inline: sh.backend(),
            rbuf: Ring::new(buf),
            wbuf: Ring::new(buf),
            pool: Pool::new(buf),
            queue: Vec::new(),
            ready: Vec::new(),
            events: vec![IoEvent::default(); 256],
            in_kernel: 0,
            in_flight: 0,
            efd: -1,
            report: AioReport { loops: 1, depth: depth as u64, ..Default::default() },
        })
    }

    fn inline_op(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op) -> Issued {
        let at = Instant::now();
        let r = issue_blocking(&mut *self.inline, sh, &mut t.a, &mut self.rbuf, &mut self.wbuf, op);
        Issued::Done(r, at.elapsed().as_nanos() as u64)
    }

    /// `io_getevents`: wait up to `timeout` (forever when `None`) for one completion and
    /// move every completion that is there to `ready`.
    fn reap(&mut self, timeout: Option<Duration>) -> Result<()> {
        let ts = timeout.map(|t| libc::timespec { tv_sec: t.as_secs() as libc::time_t, tv_nsec: t.subsec_nanos() as libc::c_long });
        let tsp = ts.as_ref().map(|t| t as *const libc::timespec).unwrap_or(std::ptr::null());
        self.report.getevents += 1;
        // SAFETY: `events` has room for the count passed; `tsp` is null or a live timespec
        let n = unsafe { libc::syscall(libc::SYS_io_getevents, self.ctx, 1 as libc::c_long, self.events.len() as libc::c_long, self.events.as_mut_ptr(), tsp) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                return Ok(());
            }
            return Err(e).context("io_getevents");
        }
        for ev in &self.events[..n as usize] {
            self.in_kernel -= 1;
            if ev.data == EFD {
                // the poll fired: take the eventfd's count so the next poll waits for the next write
                let mut v = 0u64;
                unsafe { libc::read(self.efd, &mut v as *mut u64 as *mut libc::c_void, 8) };
                self.ready.push((EFD, 0));
            } else {
                self.in_flight -= 1;
                self.ready.push((ev.data, ev.res as i32));
            }
        }
        Ok(())
    }

    /// `io_submit` everything queued. A request the kernel refuses completes with that error.
    fn flush(&mut self) -> Result<()> {
        let mut i = 0;
        while i < self.queue.len() {
            let ptrs: Vec<*mut Iocb> = self.queue[i..].iter_mut().map(|c| c as *mut Iocb).collect();
            let at = Instant::now();
            // SAFETY: `ptrs` points at live control blocks whose buffers the tasks keep alive
            let n = unsafe { libc::syscall(libc::SYS_io_submit, self.ctx, ptrs.len() as libc::c_long, ptrs.as_ptr()) };
            self.report.submits += 1;
            self.report.submit_ns += at.elapsed().as_nanos() as u64;
            if n > 0 {
                self.report.submitted += n as u64;
                self.in_kernel += n as usize;
                i += n as usize;
                continue;
            }
            let e = std::io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EINTR) => {}
                Some(libc::EAGAIN) => {
                    // the context is full: a completion frees a slot
                    self.report.full += 1;
                    if self.in_kernel == 0 {
                        return Err(e).context("io_submit: EAGAIN with nothing in flight");
                    }
                    self.reap(None)?;
                }
                Some(code) => {
                    // the request at the head is refused (EBADF, EINVAL, …)
                    let c = self.queue[i];
                    if c.data == EFD {
                        return Err(e).context("io_submit: IOCB_CMD_POLL on the loop's eventfd (needs Linux 4.18)");
                    }
                    self.in_flight -= 1;
                    self.ready.push((c.data, -code));
                    i += 1;
                }
                None => return Err(e).context("io_submit"),
            }
        }
        self.queue.clear();
        Ok(())
    }
}

impl Drop for AioIo {
    fn drop(&mut self) {
        // SAFETY: the context is this loop's; destroying it cancels or waits out what is in flight
        unsafe { libc::syscall(libc::SYS_io_destroy, self.ctx) };
    }
}

impl Engine for AioIo {
    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn post_wake(&mut self, efd: RawFd, _buf: *mut u64) -> Result<()> {
        self.efd = efd;
        self.queue.push(Iocb { data: EFD, opcode: IOCB_CMD_POLL, fildes: efd as u32, buf: libc::POLLIN as u64, ..Default::default() });
        Ok(())
    }

    fn wait(&mut self, timeout: Duration, out: &mut Vec<(u64, i32)>) -> Result<()> {
        self.flush()?;
        let timeout = if self.ready.is_empty() { timeout } else { Duration::ZERO };
        self.reap(Some(timeout))?;
        out.append(&mut self.ready);
        Ok(())
    }

    fn issue(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op) -> Result<Issued> {
        let direct = sh.opts.backend.direct();
        t.started = Instant::now();
        let (opcode, buf, nbytes, offset) = match op.kind {
            OpKind::Read => {
                let al = ALIGN as i64;
                let (lo, len, rounded) = if direct && (op.offset % al != 0 || op.len % al != 0) {
                    let (lo, hi) = round_out(op.offset, op.len);
                    (lo, hi - lo, true)
                } else {
                    (op.offset, op.len, false)
                };
                let chunk = self.pool.get(len as usize);
                let ptr = chunk.ptr;
                t.round = rounded.then_some(lo);
                t.buf = Some(chunk);
                (IOCB_CMD_PREAD, ptr as u64, len as u64, lo)
            }
            OpKind::Write => {
                let chunk = self.pool.get(op.len as usize);
                let ptr = chunk.ptr;
                fill(&mut t.a, op, unsafe { std::slice::from_raw_parts_mut(ptr, op.len as usize) });
                t.buf = Some(chunk);
                (IOCB_CMD_PWRITE, ptr as u64, op.len as u64, op.offset)
            }
            OpKind::Fsync => (IOCB_CMD_FSYNC, 0, 0, 0),
            OpKind::Fdatasync => (IOCB_CMD_FDSYNC, 0, 0, 0),
            _ => return Ok(self.inline_op(sh, t, op)),
        };
        let f = match t.a.fd(op.path) {
            Ok(f) => f,
            Err(e) => {
                if let Some(c) = t.buf.take() {
                    self.pool.put(c);
                }
                t.round = None;
                return Ok(Issued::Done(Err(e), 0));
            }
        };
        self.queue.push(Iocb { data: t.id as u64, opcode, fildes: f.as_raw_fd() as u32, buf, nbytes, offset, ..Default::default() });
        t.fd = Some(f);
        self.in_flight += 1;
        self.report.in_flight_peak = self.report.in_flight_peak.max(self.in_flight as u64);
        Ok(Issued::Pending)
    }

    fn complete(&mut self, _sh: &Shared, t: &mut TaskIo, op: &Op, res: i32) -> std::io::Result<i64> {
        t.fd = None;
        if let Some(c) = t.buf.take() {
            self.pool.put(c);
        }
        let round = t.round.take();
        if res < 0 {
            return Err(std::io::Error::from_raw_os_error(-res));
        }
        match (op.kind, round) {
            (OpKind::Read, Some(lo)) => Ok((res as i64 - (op.offset - lo)).clamp(0, op.len)),
            (OpKind::Read | OpKind::Write, _) => Ok(res as i64),
            _ => Ok(0),
        }
    }
}
