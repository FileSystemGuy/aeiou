//! The `io_uring` backends: one event loop per thread, many actors per loop, one ring per
//! loop (`NAPKIN_MATH.md` §4.2). Each actor instance (and every sub-actor it forks) is a
//! `Task`: a resumable `Vm` parked at its current event plus its `ActorState`. The loop
//! advances runnable tasks until each one blocks (an op on the ring, a `compute` timer, a
//! channel, a barrier, a join), submits the queued SQEs with `io_uring_enter`, waits for a
//! completion or the next timer, and dispatches completions back to their tasks by
//! `user_data`. The op stream and the fingerprint are what the `sync` sink produces; only
//! the API that carries each op changes (`PROJECT_BRIEF.md` §4).
//!
//! Where an op has no opcode (`lseek`, `ioctl`, `readdir`/`getdents64`) or the kernel lacks
//! one (probed at startup; `ftruncate` needs 6.9), the loop issues it inline with the
//! blocking backend, which is what an `io_uring` application has to do too. `close` drops
//! the actor's reference as under `sync` (the descriptor closes with the last reference).
//! Every actor keeps at most one op in flight: an actor is a sequential program, and a
//! PyTorch worker blocks in `read`. Concurrency comes from the actors on the loop, so the
//! read and write buffers are a pool of aligned chunks, one per op in flight, rotated FIFO so
//! the aggregate exceeds the cache (`--buffer-mib` sets the floor).
//!
//! Every read and write goes on the ring at its effective offset, the one the VM computed
//! and the fingerprint hashes, never at the kernel's file position (`offset = -1`): on Linux
//! 6.18 a `-1` read of an `O_DIRECT` file through the ring does not advance the position
//! (a buffered one does), so the sequential `until_eof` idiom read the first megabyte over
//! and over. An `io_uring` application keeps its own positions, and so does this backend; the
//! op stream is unchanged, and under the interposition test (`PROJECT_BRIEF.md` §5) turning
//! `read` into `pread` at the known position is exactly what a shim may do.
//!
//! An instance's sub-actors run on its loop, so channels are loop-local and lock-free.
//! Barriers go through the `Coordinator`'s non-blocking half: `arrive`, then a read posted on
//! the loop's eventfd, which every release writes (`NAPKIN_MATH.md` §8.A). Kernel 5.11 or
//! later: `IORING_ENTER_EXT_ARG` carries the wait timeout.
//!
//! The loop itself (tasks, channels, timers, barriers) is generic over an `Engine`, the part
//! that carries ops to the kernel and brings completions back: the ring here, an AIO context
//! in `aio.rs`.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use io_uring::{opcode, squeue, types, IoUring, Probe};

use crate::backend::{open_flags, Backend, ALIGN};
use crate::eval::Model;
use crate::run::{fill, issue_blocking, round_out, untaken, ActorState, Ring, Shared, UringReport};
use crate::vm::{Control, Event, ForkKind, Op, OpKind, Vm};

/// `user_data` of the read posted on the loop's eventfd.
pub(crate) const EFD: u64 = u64::MAX;
/// How long a loop sleeps with nothing to wake it, so the abort flag is seen.
const IDLE: Duration = Duration::from_millis(50);

/// The first loop's ring descriptor, for the other loops to attach to under
/// `sqpoll_shared`; the error when that ring could not be built, so nobody waits for it.
type Gate = (Mutex<Option<std::result::Result<RawFd, String>>>, Condvar);

/// This host's instances `(template, actor, count)` dealt round-robin over the event loops
/// (`--threads`, or one per core, at most one per instance); an instance's sub-actors follow
/// it. Empty when the host has no instance.
pub(crate) fn spread(sh: &Shared, counts: &[(&'static str, i64)], ranges: &[(i64, i64)]) -> Vec<Vec<(&'static str, i64, i64)>> {
    let mut instances: Vec<(&'static str, i64, i64)> = Vec::new();
    for ((template, count), (lo, hi)) in counts.iter().zip(ranges) {
        for actor in *lo..*hi {
            instances.push((template, actor, *count));
        }
    }
    if instances.is_empty() {
        return Vec::new();
    }
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = if sh.opts.threads == 0 { cores } else { sh.opts.threads }.clamp(1, instances.len());
    let mut per: Vec<Vec<(&'static str, i64, i64)>> = vec![Vec::new(); threads];
    for (i, inst) in instances.into_iter().enumerate() {
        per[i % threads].push(inst);
    }
    per
}

/// The `SQPOLL` thread of a ring, as the kernel states it in the ring's `fdinfo`
/// (`SqThread:`, there from the moment `io_uring_setup` returns; the thread's name is not,
/// since the thread names itself `iou-sqp-*` only when it first runs). `None` without
/// `SQPOLL`. Rings that share a poll thread state the same one.
fn sq_thread(ring: RawFd) -> Option<i64> {
    let s = std::fs::read_to_string(format!("/proc/self/fdinfo/{ring}")).ok()?;
    let pid: i64 = s.lines().find_map(|l| l.strip_prefix("SqThread:"))?.trim().parse().ok()?;
    (pid > 0).then_some(pid)
}

/// Run this host's instances over the event loops; returns how many loop threads ran and
/// what the rings were set up with, and the distinct `SQPOLL` threads the rings stated once built.
pub(crate) fn run(model: &'static Model<'static>, sh: &Arc<Shared>, counts: &[(&'static str, i64)], ranges: &[(i64, i64)]) -> Result<(UringReport, u64)> {
    sh.opts.uring.check()?;
    let per = spread(sh, counts, ranges);
    if per.is_empty() {
        return Ok((UringReport { loops: 0, opts: sh.opts.uring.clone(), iowq_defaults: None, in_flight_peak: 0 }, 0));
    }
    let threads = per.len();
    let shared = sh.opts.uring.sqpoll_shared;
    let gate: Arc<Gate> = Arc::new((Mutex::new(None), Condvar::new()));
    std::thread::scope(|s| {
        let handles: Vec<_> = per
            .into_iter()
            .enumerate()
            .map(|(i, insts)| {
                let sh = sh.clone();
                let gate = gate.clone();
                std::thread::Builder::new()
                    .name(format!("io_uring loop {i}"))
                    .spawn_scoped(s, move || {
                        // the rings are built on their own threads (SINGLE_ISSUER binds a
                        // ring to the task that built it); under `sqpoll_shared` loops 1.. wait
                        // for loop 0's ring and attach to it
                        let attach = if shared && i > 0 {
                            let (m, cv) = &*gate;
                            let mut g = m.lock().unwrap();
                            while g.is_none() {
                                g = cv.wait(g).unwrap();
                            }
                            match g.as_ref().unwrap() {
                                Ok(fd) => Some(*fd),
                                Err(e) => {
                                    sh.aborted.store(true, Ordering::Relaxed);
                                    return Err(anyhow!("io_uring loop 0: {e}"));
                                }
                            }
                        } else {
                            None
                        };
                        let built = Loop::new(sh.clone(), i, attach);
                        if shared && i == 0 {
                            let (m, cv) = &*gate;
                            *m.lock().unwrap() = Some(built.as_ref().map(|(l, _)| l.io.ring.as_raw_fd()).map_err(|e| format!("{e:#}")));
                            cv.notify_all();
                        }
                        let r = built.and_then(|(mut l, defaults)| {
                            let sq = sq_thread(l.io.ring.as_raw_fd());
                            l.run(model, insts).map(|_| (defaults, l.io.in_flight_peak as u64, sq))
                        });
                        if r.is_err() {
                            sh.aborted.store(true, Ordering::Relaxed);
                        }
                        r
                    })
                    .expect("spawn event loop")
            })
            .collect();
        let mut first: Option<anyhow::Error> = None;
        let mut defaults = None;
        let mut in_flight_peak = 0;
        let mut sq_threads = std::collections::BTreeSet::new();
        for (i, h) in handles.into_iter().enumerate() {
            match h.join() {
                Ok(Ok((d, peak, sq))) => {
                    sq_threads.extend(sq);
                    if i == 0 {
                        defaults = d;
                    }
                    in_flight_peak = in_flight_peak.max(peak);
                }
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
            None => Ok((UringReport { loops: threads as u64, opts: sh.opts.uring.clone(), iowq_defaults: defaults, in_flight_peak }, sq_threads.len() as u64)),
        }
    })
}

// ---------------------------------------------------------------- buffers

pub(crate) struct Chunk {
    pub(crate) ptr: *mut u8,
    size: usize,
}

/// Aligned buffers, one per op in flight. Freed chunks go to the back of their size's queue
/// and are reused only once the pool holds `target` bytes, so successive I/Os land in
/// different memory the way a loader's copies do (`NAPKIN_MATH.md` §2.2).
pub(crate) struct Pool {
    free: HashMap<usize, VecDeque<Chunk>>,
    total: usize,
    target: usize,
}

impl Pool {
    pub(crate) fn new(target: usize) -> Self {
        Pool { free: HashMap::new(), total: 0, target }
    }

    pub(crate) fn get(&mut self, len: usize) -> Chunk {
        let size = (len.max(1) + ALIGN - 1) / ALIGN * ALIGN;
        if self.total >= self.target {
            if let Some(c) = self.free.get_mut(&size).and_then(|q| q.pop_front()) {
                return c;
            }
        }
        let layout = std::alloc::Layout::from_size_align(size, ALIGN).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "buffer allocation of {size} bytes failed");
        self.total += size;
        Chunk { ptr, size }
    }

    pub(crate) fn put(&mut self, c: Chunk) {
        self.free.entry(c.size).or_default().push_back(c);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for (_, q) in self.free.drain() {
            for c in q {
                let layout = std::alloc::Layout::from_size_align(c.size, ALIGN).unwrap();
                unsafe { std::alloc::dealloc(c.ptr, layout) };
            }
        }
    }
}

// ---------------------------------------------------------------- channels

/// A loop-local channel with the semantics of `run::Channel`: a `loader` bounds batches
/// started, a `channel` bounds items delivered, `take` is in order unless unordered.
struct Chan {
    capacity: i64,
    ordered: bool,
    total: Option<i64>,
    next_take: i64,
    done: BTreeSet<i64>,
    closed: bool,
    /// Tasks parked on this channel; woken (all of them) on every change.
    waiters: Vec<usize>,
}

impl Chan {
    fn new(capacity: i64, ordered: bool, total: Option<i64>) -> Self {
        Chan { capacity: capacity.max(1), ordered, total, next_take: 0, done: BTreeSet::new(), closed: false, waiters: Vec::new() }
    }

    /// `Ok(true)`: batch `b` may start; `Ok(false)`: wait; `Err`: closed.
    fn try_begin(&self, b: i64) -> Result<bool> {
        if self.closed {
            bail!("channel closed before batch {b} started");
        }
        Ok(b < self.next_take + self.capacity)
    }

    fn end(&mut self, b: i64) {
        self.done.insert(b);
    }

    fn try_put(&mut self, seq: i64) -> Result<bool> {
        if self.closed {
            bail!("put {seq} on a closed channel");
        }
        if (self.done.len() as i64) < self.capacity {
            self.done.insert(seq);
            return Ok(true);
        }
        Ok(false)
    }

    fn try_take(&mut self) -> Result<Option<i64>> {
        if let Some(total) = self.total {
            if self.next_take >= total {
                bail!("take after the last of {total} items");
            }
        }
        let got = if self.ordered {
            let want = self.next_take;
            if self.done.remove(&want) { Some(want) } else { None }
        } else {
            self.done.iter().next().copied().map(|x| {
                self.done.remove(&x);
                x
            })
        };
        if let Some(x) = got {
            self.next_take += 1;
            return Ok(Some(x));
        }
        if self.closed {
            bail!("take on a closed channel with nothing delivered");
        }
        Ok(None)
    }
}

// ---------------------------------------------------------------- tasks

#[derive(Clone, Copy)]
enum Role {
    Main,
    Parallel { parent: usize },
    /// Loader worker `w` of `workers`: builds batches `w, w + workers, …` below `batches`.
    Worker { parent: usize, chan: usize, workers: i64, batches: i64, b: i64 },
}

#[derive(Clone, Copy)]
enum ChanWait {
    Begin(i64),
    Put(i64),
    Take,
}

#[derive(Clone, Copy)]
enum Wait {
    None,
    Io,
    Timer,
    Chan { chan: usize, what: ChanWait, since: Instant },
    Barrier { scope: &'static str, generation: u64, since: Instant },
    /// A `parallel` parent: sub-actors still running.
    JoinParallel { remaining: usize },
    /// A main line that has ended its body: loader workers still running.
    JoinLoaders,
}

/// What an op in flight needs kept alive and remembered until its completion.
pub(crate) struct TaskIo {
    pub(crate) id: usize,
    pub(crate) a: ActorState,
    cpath: Option<CString>,
    cpath2: Option<CString>,
    statx: Box<libc::statx>,
    pub(crate) buf: Option<Chunk>,
    pub(crate) fd: Option<Arc<crate::backend::OpenFile>>,
    /// A direct read rounded out: the aligned start it was issued at.
    pub(crate) round: Option<i64>,
    pub(crate) started: Instant,
}

struct Task {
    vm: Vm<'static, 'static>,
    io: TaskIo,
    /// The main task of the instance this task belongs to (channels are per instance).
    inst: usize,
    role: Role,
    wait: Wait,
    /// A walk is in progress (between `start`/`start_sub` and its end).
    active: bool,
    queued: bool,
    done: bool,
    /// Main line: loader workers alive, and the loader channels (checked at the end).
    workers_live: usize,
    loaders: Vec<usize>,
}

pub(crate) enum Issued {
    /// Done inline: the result and the nanoseconds it took.
    Done(std::io::Result<i64>, u64),
    /// On the ring (or in the AIO context).
    Pending,
}

/// What carries a loop's ops to the kernel and brings their completions back.
pub(crate) trait Engine {
    /// Queue the op for the kernel, or do it inline.
    fn issue(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op) -> Result<Issued>;
    /// Interpret a completion the way the blocking path interprets a return value.
    fn complete(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op, res: i32) -> std::io::Result<i64>;
    /// Ops queued or in the kernel.
    fn in_flight(&self) -> usize;
    /// Ask for one completion with `user_data` `EFD` when the eventfd is next written; the
    /// engine consumes the eventfd's count (`buf` is the loop's, alive until then).
    fn post_wake(&mut self, efd: RawFd, buf: *mut u64) -> Result<()>;
    /// Submit what is queued, wait up to `timeout` for a completion, and append every
    /// completion that is ready as `(user_data, result)`.
    fn wait(&mut self, timeout: Duration, out: &mut Vec<(u64, i32)>) -> Result<()>;
}

/// The ring and what issuing needs; disjoint from the tasks so a parked task's op can be
/// borrowed while it is issued.
struct LoopIo {
    ring: IoUring,
    inline: Box<dyn Backend>,
    rbuf: Ring,
    wbuf: Ring,
    pool: Pool,
    supported: Vec<bool>,
    in_flight: usize,
    in_flight_peak: usize,
}

impl LoopIo {
    fn supported(&self, code: u8) -> bool {
        self.supported.get(code as usize).copied().unwrap_or(false)
    }

    fn push(&mut self, e: squeue::Entry) -> Result<()> {
        loop {
            {
                let mut sq = self.ring.submission();
                if unsafe { sq.push(&e) }.is_ok() {
                    return Ok(());
                }
            }
            self.ring.submit().context("io_uring_enter (submission queue full)")?;
        }
    }

    fn inline_op(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op) -> Issued {
        let at = Instant::now();
        let r = issue_blocking(&mut *self.inline, sh, &mut t.a, &mut self.rbuf, &mut self.wbuf, op);
        Issued::Done(r, at.elapsed().as_nanos() as u64)
    }

}

impl Engine for LoopIo {
    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn post_wake(&mut self, efd: RawFd, buf: *mut u64) -> Result<()> {
        self.push(opcode::Read::new(types::Fd(efd), buf as *mut u8, 8).build().user_data(EFD))
    }

    fn wait(&mut self, timeout: Duration, out: &mut Vec<(u64, i32)>) -> Result<()> {
        let ts = types::Timespec::new().sec(timeout.as_secs()).nsec(timeout.subsec_nanos());
        let args = types::SubmitArgs::new().timespec(&ts);
        match self.ring.submitter().submit_with_args(1, &args) {
            Ok(_) => {}
            Err(e) if matches!(e.raw_os_error(), Some(libc::ETIME) | Some(libc::EINTR) | Some(libc::EBUSY) | Some(libc::EAGAIN)) => {}
            Err(e) => return Err(e).context("io_uring_enter"),
        }
        for c in self.ring.completion() {
            if c.user_data() != EFD {
                self.in_flight -= 1;
            }
            out.push((c.user_data(), c.result()));
        }
        Ok(())
    }

    /// Put the op on the ring, or do it inline.
    fn issue(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op) -> Result<Issued> {
        let direct = sh.opts.backend.direct();
        let cwd = types::Fd(libc::AT_FDCWD);
        let fd = |t: &mut TaskIo| -> std::io::Result<types::Fd> {
            let f = t.a.fd(op.path)?;
            let raw = f.as_raw_fd();
            t.fd = Some(f);
            Ok(types::Fd(raw))
        };
        macro_rules! try_fd {
            ($t:expr) => {
                match fd($t) {
                    Ok(f) => f,
                    Err(e) => return Ok(Issued::Done(Err(e), 0)),
                }
            };
        }
        let path = |rel: &str| -> Result<CString> {
            use std::os::unix::ffi::OsStrExt;
            CString::new(sh.opts.root.join(rel).as_os_str().as_bytes()).map_err(|_| anyhow!("path contains NUL: {rel}"))
        };
        t.started = Instant::now();
        let entry: squeue::Entry = match op.kind {
            OpKind::Close | OpKind::Lseek | OpKind::Ioctl | OpKind::Readdir => return Ok(self.inline_op(sh, t, op)),
            OpKind::Open => {
                if !self.supported(opcode::OpenAt::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let mode = (op.aux >> 32) as u32;
                let mode = if mode == 0 { 0o644 } else { mode };
                let mut flags = open_flags(op.aux & 0xffff_ffff);
                if direct && flags & libc::O_DIRECTORY == 0 {
                    flags |= libc::O_DIRECT;
                }
                let c = path(op.path)?;
                let e = opcode::OpenAt::new(cwd, c.as_ptr()).flags(flags).mode(mode).build();
                t.cpath = Some(c);
                e
            }
            OpKind::Read => {
                if !self.supported(opcode::Read::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                let al = ALIGN as i64;
                let (lo, len, rounded) = if direct && (op.offset % al != 0 || op.len % al != 0) {
                    let (lo, hi) = round_out(op.offset, op.len);
                    (lo, hi - lo, true)
                } else {
                    (op.offset, op.len, false)
                };
                t.round = rounded.then_some(lo);
                let chunk = self.pool.get(len as usize);
                let e = opcode::Read::new(f, chunk.ptr, len as u32).offset(lo as u64).build();
                t.buf = Some(chunk);
                e
            }
            OpKind::Write => {
                if !self.supported(opcode::Write::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                let chunk = self.pool.get(op.len as usize);
                let buf = unsafe { std::slice::from_raw_parts_mut(chunk.ptr, op.len as usize) };
                fill(&mut t.a, op, buf);
                let e = opcode::Write::new(f, chunk.ptr, op.len as u32).offset(op.offset as u64).build();
                t.buf = Some(chunk);
                e
            }
            OpKind::Fadvise => {
                if !self.supported(opcode::Fadvise::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                opcode::Fadvise::new(f, op.len as libc::off_t, op.aux as i32).offset(op.offset as u64).build()
            }
            OpKind::Fstat => {
                if !self.supported(opcode::Statx::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                let c = CString::new("").unwrap();
                let e = opcode::Statx::new(f, c.as_ptr(), &mut *t.statx as *mut libc::statx as *mut types::statx).flags(libc::AT_EMPTY_PATH).mask(libc::STATX_SIZE).build();
                t.cpath = Some(c);
                e
            }
            OpKind::Stat => {
                if !self.supported(opcode::Statx::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let c = path(op.path)?;
                let e = opcode::Statx::new(cwd, c.as_ptr(), &mut *t.statx as *mut libc::statx as *mut types::statx).mask(libc::STATX_SIZE).build();
                t.cpath = Some(c);
                e
            }
            OpKind::Fsync | OpKind::Fdatasync => {
                if !self.supported(opcode::Fsync::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                let flags = if op.kind == OpKind::Fdatasync { types::FsyncFlags::DATASYNC } else { types::FsyncFlags::empty() };
                opcode::Fsync::new(f).flags(flags).build()
            }
            OpKind::Unlink | OpKind::Rmdir => {
                if !self.supported(opcode::UnlinkAt::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let c = path(op.path)?;
                let flags = if op.kind == OpKind::Rmdir { libc::AT_REMOVEDIR } else { 0 };
                let e = opcode::UnlinkAt::new(cwd, c.as_ptr()).flags(flags).build();
                t.cpath = Some(c);
                e
            }
            OpKind::Ftruncate => {
                if !self.supported(opcode::Ftruncate::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                opcode::Ftruncate::new(f, op.len as u64).build()
            }
            OpKind::Fallocate => {
                if !self.supported(opcode::Fallocate::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let f = try_fd!(t);
                opcode::Fallocate::new(f, op.len as u64).offset(op.offset as u64).mode(0).build()
            }
            OpKind::Mkdir => {
                if !self.supported(opcode::MkDirAt::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let c = path(op.path)?;
                let e = opcode::MkDirAt::new(cwd, c.as_ptr()).mode(op.aux as libc::mode_t).build();
                t.cpath = Some(c);
                e
            }
            OpKind::Rename => {
                if !self.supported(opcode::RenameAt::CODE) {
                    return Ok(self.inline_op(sh, t, op));
                }
                let c1 = path(op.path)?;
                let c2 = path(op.path2.unwrap_or(""))?;
                let e = opcode::RenameAt::new(cwd, c1.as_ptr(), cwd, c2.as_ptr()).build();
                t.cpath = Some(c1);
                t.cpath2 = Some(c2);
                e
            }
        };
        self.push(entry.user_data(t.id as u64))?;
        self.in_flight += 1;
        self.in_flight_peak = self.in_flight_peak.max(self.in_flight);
        Ok(Issued::Pending)
    }

    fn complete(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op, res: i32) -> std::io::Result<i64> {
        t.cpath = None;
        t.cpath2 = None;
        t.fd = None;
        if let Some(c) = t.buf.take() {
            self.pool.put(c);
        }
        let round = t.round.take();
        if res < 0 {
            return Err(std::io::Error::from_raw_os_error(-res));
        }
        match op.kind {
            OpKind::Open => {
                let fd = unsafe { OwnedFd::from_raw_fd(res) };
                t.a.opened(sh, op.path, op.aux, fd);
                Ok(0)
            }
            OpKind::Read => {
                let n = res as i64;
                match round {
                    Some(lo) => Ok((n - (op.offset - lo)).clamp(0, op.len)),
                    None => Ok(n),
                }
            }
            OpKind::Write => Ok(res as i64),
            OpKind::Fstat | OpKind::Stat => Ok(t.statx.stx_size as i64),
            OpKind::Unlink => {
                t.a.removed.push(op.path.to_string());
                Ok(0)
            }
            OpKind::Mkdir => {
                t.a.created.push((op.path.to_string(), t.a.actor));
                Ok(0)
            }
            OpKind::Rename => {
                t.a.removed.push(op.path.to_string());
                t.a.created.push((op.path2.unwrap_or("").to_string(), t.a.actor));
                Ok(0)
            }
            _ => Ok(0),
        }
    }
}

// ---------------------------------------------------------------- the loop

pub(crate) struct Loop<E: Engine> {
    sh: Arc<Shared>,
    index: usize,
    pub(crate) io: E,
    tasks: Vec<Task>,
    runnable: VecDeque<usize>,
    timers: BinaryHeap<Reverse<(Instant, usize)>>,
    chans: Vec<Chan>,
    chan_names: HashMap<(usize, String), usize>,
    barrier_waiters: Vec<usize>,
    efd: OwnedFd,
    efd_buf: Box<u64>,
    efd_posted: bool,
    live: usize,
}

impl Loop<LoopIo> {
    /// Build the loop's ring with the knobs of `RunOpts::uring` (`attach`: the ring whose
    /// `SQPOLL` thread to share) and cap its io-wq; returns the caps the kernel had before.
    fn new(sh: Arc<Shared>, index: usize, attach: Option<RawFd>) -> Result<(Self, Option<[u32; 2]>)> {
        let k = &sh.opts.uring;
        let mut b = IoUring::builder();
        b.setup_cqsize(4096);
        if let Some(idle) = k.sqpoll_idle_ms {
            b.setup_sqpoll(idle);
        }
        if let Some(fd) = attach {
            b.setup_attach_wq(fd);
        }
        if k.defer_taskrun {
            b.setup_single_issuer().setup_defer_taskrun();
        }
        if k.coop_taskrun {
            b.setup_coop_taskrun();
        }
        let ring = b.build(1024).with_context(|| format!("io_uring_setup with {} (is `kernel.io_uring_disabled` set? SQPOLL needs 5.13 unprivileged, DEFER_TASKRUN 6.1)", k.describe()))?;
        // one call reads the kernel's io-wq caps and sets ours: a 0 leaves that cap alone,
        // and the previous values come back in the array. Under SQPOLL the caps go to the
        // poll thread's io-wq, the one that runs this ring's punted ops.
        let mut caps = [k.iowq_max_workers, 0];
        let defaults = match ring.submitter().register_iowq_max_workers(&mut caps) {
            Ok(()) => Some(caps),
            Err(_) if k.iowq_max_workers == 0 => None,
            Err(e) => return Err(e).context("IORING_REGISTER_IOWQ_MAX_WORKERS (needs Linux 5.15)"),
        };
        let mut probe = Probe::new();
        ring.submitter().register_probe(&mut probe).context("io_uring probe")?;
        let supported: Vec<bool> = (0..=255u8).map(|c| probe.is_supported(c)).collect();
        let buf = sh.opts.buffer_bytes;
        let io = LoopIo { ring, inline: sh.backend(), rbuf: Ring::new(buf), wbuf: Ring::new(buf), pool: Pool::new(buf), supported, in_flight: 0, in_flight_peak: 0 };
        Ok((Loop::with(sh, index, io)?, defaults))
    }
}

impl<E: Engine> Loop<E> {
    /// A loop over `io`, with its eventfd subscribed to the coordinator's releases.
    pub(crate) fn with(sh: Arc<Shared>, index: usize, io: E) -> Result<Self> {
        let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if efd < 0 {
            return Err(std::io::Error::last_os_error()).context("eventfd");
        }
        let efd = unsafe { OwnedFd::from_raw_fd(efd) };
        sh.coord.subscribe(efd.as_raw_fd());
        Ok(Loop {
            io,
            sh,
            index,
            tasks: Vec::new(),
            runnable: VecDeque::new(),
            timers: BinaryHeap::new(),
            chans: Vec::new(),
            chan_names: HashMap::new(),
            barrier_waiters: Vec::new(),
            efd,
            efd_buf: Box::new(0),
            efd_posted: false,
            live: 0,
        })
    }

    fn add_task(&mut self, vm: Vm<'static, 'static>, a: ActorState, inst: Option<usize>, role: Role, active: bool) -> usize {
        let id = self.tasks.len();
        self.tasks.push(Task {
            vm,
            io: TaskIo { id, a, cpath: None, cpath2: None, statx: Box::new(unsafe { std::mem::zeroed() }), buf: None, fd: None, round: None, started: Instant::now() },
            inst: inst.unwrap_or(id),
            role,
            wait: Wait::None,
            active,
            queued: false,
            done: false,
            workers_live: 0,
            loaders: Vec::new(),
        });
        self.live += 1;
        self.ready(id);
        id
    }

    fn ready(&mut self, id: usize) {
        let t = &mut self.tasks[id];
        if !t.queued && !t.done {
            t.queued = true;
            self.runnable.push_back(id);
        }
    }

    fn wake(&mut self, chan: usize) {
        let waiters = std::mem::take(&mut self.chans[chan].waiters);
        for id in waiters {
            self.ready(id);
        }
    }

    fn label(&self, id: usize) -> String {
        let t = &self.tasks[id];
        let (op, ctx) = t.vm.current();
        let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
        let _ = op;
        format!("{}#{} [{}]", ctx.template, ctx.actor, idx.join(","))
    }

    pub(crate) fn run(&mut self, model: &'static Model<'static>, insts: Vec<(&'static str, i64, i64)>) -> Result<()> {
        for (template, actor, count) in insts {
            let mut vm = Vm::new(model, template, actor, count);
            vm.start(&model.ast.actors[template].body);
            let a = ActorState::new(&self.sh, template, actor, true, None, 0);
            self.add_task(vm, a, None, Role::Main, true);
        }
        let r = self.serve();
        self.sh.coord.unsubscribe(self.efd.as_raw_fd());
        r
    }

    fn serve(&mut self) -> Result<()> {
        loop {
            if self.sh.aborted.load(Ordering::Relaxed) {
                bail!("run aborted: {}", self.sh.coord.abort_reason().unwrap_or_else(|| "another actor failed".into()));
            }
            while let Some(id) = self.runnable.pop_front() {
                self.tasks[id].queued = false;
                if !self.tasks[id].done {
                    self.step(id)?;
                }
            }
            if self.live == 0 {
                return Ok(());
            }
            if !self.barrier_waiters.is_empty() && !self.efd_posted {
                self.io.post_wake(self.efd.as_raw_fd(), &mut *self.efd_buf as *mut u64)?;
                self.efd_posted = true;
            }
            if self.io.in_flight() == 0 && self.timers.is_empty() && !self.efd_posted {
                let parked: Vec<String> = (0..self.tasks.len()).filter(|i| !self.tasks[*i].done).map(|i| self.label(i)).collect();
                bail!("event loop {}: {} actor(s) wait on channels nothing will complete: {}", self.index, parked.len(), parked.join(", "));
            }
            let now = Instant::now();
            let timeout = match self.timers.peek() {
                Some(Reverse((t, _))) => t.saturating_duration_since(now).min(IDLE),
                None => IDLE,
            };
            let mut cqes: Vec<(u64, i32)> = Vec::new();
            self.io.wait(timeout, &mut cqes)?;
            for (ud, res) in cqes {
                if ud == EFD {
                    self.efd_posted = false;
                } else {
                    self.complete(ud as usize, res)?;
                }
            }
            let now = Instant::now();
            while let Some(Reverse((t, id))) = self.timers.peek() {
                if *t > now {
                    break;
                }
                let id = *id;
                self.timers.pop();
                self.tasks[id].wait = Wait::None;
                self.ready(id);
            }
            if !self.barrier_waiters.is_empty() {
                let waiters = std::mem::take(&mut self.barrier_waiters);
                for id in waiters {
                    let Wait::Barrier { scope, generation, since } = self.tasks[id].wait else { continue };
                    if self.sh.coord.released(scope, generation) {
                        let t = &mut self.tasks[id];
                        t.io.a.st.barriers += 1;
                        t.io.a.st.barrier_wait_ns += since.elapsed().as_nanos() as u64;
                        t.wait = Wait::None;
                        self.ready(id);
                    } else {
                        self.barrier_waiters.push(id);
                    }
                }
            }
        }
    }

    /// A completion for task `id`.
    fn complete(&mut self, id: usize, res: i32) -> Result<()> {
        let t = &mut self.tasks[id];
        let ns = t.io.started.elapsed().as_nanos() as u64;
        let (op, ctx) = t.vm.current();
        let r = self.io.complete(&self.sh, &mut t.io, &op, res);
        t.io.a.settle(&op, &ctx, r, ns)?;
        t.wait = Wait::None;
        self.ready(id);
        Ok(())
    }

    /// A task parked on a channel tries again. `Ok(true)`: it may go on.
    fn retry_chan(&mut self, id: usize) -> Result<bool> {
        let Wait::Chan { chan, what, since } = self.tasks[id].wait else { return Ok(true) };
        match what {
            ChanWait::Begin(b) => match self.chans[chan].try_begin(b) {
                Ok(true) => {
                    let t = &mut self.tasks[id];
                    t.vm.start_sub(b);
                    t.active = true;
                }
                Ok(false) => {
                    self.chans[chan].waiters.push(id);
                    return Ok(false);
                }
                Err(_) => {
                    // closed by the consumer: this worker is done
                    let t = &mut self.tasks[id];
                    if let Role::Worker { ref mut b, batches, .. } = t.role {
                        *b = batches;
                    }
                }
            },
            ChanWait::Put(seq) => {
                if !self.chans[chan].try_put(seq)? {
                    self.chans[chan].waiters.push(id);
                    return Ok(false);
                }
                self.tasks[id].io.a.st.puts += 1;
                self.wake(chan);
            }
            ChanWait::Take => {
                let got = self.chans[chan].try_take().with_context(|| format!("{} take", self.label(id)))?;
                if got.is_none() {
                    self.chans[chan].waiters.push(id);
                    return Ok(false);
                }
                self.tasks[id].io.a.took(since.elapsed().as_nanos() as u64);
                self.wake(chan);
            }
        }
        self.tasks[id].wait = Wait::None;
        Ok(true)
    }

    /// Advance task `id` until it blocks or ends.
    fn step(&mut self, id: usize) -> Result<()> {
        if let Wait::Chan { .. } = self.tasks[id].wait {
            if !self.retry_chan(id)? {
                return Ok(());
            }
        }
        loop {
            if !matches!(self.tasks[id].wait, Wait::None) {
                return Ok(());
            }
            if !self.tasks[id].active {
                // between walks: a worker needs its next batch; anything else has ended
                match self.tasks[id].role {
                    Role::Worker { .. } => {
                        if !self.worker_next(id)? {
                            return Ok(());
                        }
                        continue;
                    }
                    _ => return self.finished(id),
                }
            }
            let sh = self.sh.clone();
            let t = &mut self.tasks[id];
            match t.vm.next()? {
                None => {
                    t.active = false;
                    if let Role::Worker { chan, ref mut b, workers, .. } = t.role {
                        let done = *b;
                        *b += workers;
                        self.chans[chan].end(done);
                        self.wake(chan);
                    }
                }
                Some(Event::Op(op, ctx)) => {
                    t.io.a.check_align(&sh, &op)?;
                    match self.io.issue(&sh, &mut t.io, &op)? {
                        Issued::Done(r, ns) => t.io.a.settle(&op, &ctx, r, ns)?,
                        Issued::Pending => {
                            t.wait = Wait::Io;
                            return Ok(());
                        }
                    }
                }
                Some(Event::Control(c, ctx)) => {
                    let inst = t.inst;
                    match c {
                        Control::Compute { ns } => {
                            let ns = ns.max(0) as u64;
                            t.io.a.computed(ns);
                            let scaled = (ns as f64 * sh.opts.time_scale) as u64;
                            if scaled > 0 {
                                t.wait = Wait::Timer;
                                self.timers.push(Reverse((Instant::now() + Duration::from_nanos(scaled), id)));
                                return Ok(());
                            }
                        }
                        Control::Barrier { scope } => {
                            if !t.io.a.main {
                                bail!("barrier `{scope}` inside a sub-actor is not supported");
                            }
                            let since = Instant::now();
                            let generation = sh.coord.arrive(scope)?;
                            if sh.coord.released(scope, generation) {
                                t.io.a.st.barriers += 1;
                            } else {
                                t.wait = Wait::Barrier { scope, generation, since };
                                self.barrier_waiters.push(id);
                                return Ok(());
                            }
                        }
                        Control::Channel { name, capacity, ordered } => {
                            if self.chan_names.contains_key(&(inst, name.to_string())) {
                                bail!("channel `{name}` declared twice");
                            }
                            self.chans.push(Chan::new(capacity, ordered, None));
                            self.chan_names.insert((inst, name.to_string()), self.chans.len() - 1);
                        }
                        Control::Put { channel, seq } => {
                            let chan = *self.chan_names.get(&(inst, channel.to_string())).ok_or_else(|| anyhow!("put on undeclared channel `{channel}`"))?;
                            self.tasks[id].wait = Wait::Chan { chan, what: ChanWait::Put(seq), since: Instant::now() };
                            if !self.retry_chan(id)? {
                                return Ok(());
                            }
                        }
                        Control::Take { channel } => {
                            let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
                            let chan = *self.chan_names.get(&(inst, channel.to_string())).ok_or_else(|| anyhow!("{}#{} [{}] take on undeclared channel `{channel}` (a loader declares one under its name)", ctx.template, ctx.actor, idx.join(",")))?;
                            self.tasks[id].wait = Wait::Chan { chan, what: ChanWait::Take, since: Instant::now() };
                            if !self.retry_chan(id)? {
                                return Ok(());
                            }
                        }
                    }
                }
                Some(Event::Fork(kind)) => {
                    let snap = t.vm.snapshot();
                    t.vm.accept_fork();
                    let inst = t.inst;
                    match kind {
                        ForkKind::Parallel { width, .. } => {
                            let children: Vec<ActorState> = (0..width).map(|_| t.io.a.child(&sh, 0)).collect();
                            for (k, a) in children.into_iter().enumerate() {
                                let mut vm = Vm::resume(snap.clone());
                                vm.start_sub(k as i64);
                                self.add_task(vm, a, Some(inst), Role::Parallel { parent: id }, true);
                            }
                            if width > 0 {
                                self.tasks[id].wait = Wait::JoinParallel { remaining: width as usize };
                                return Ok(());
                            }
                        }
                        ForkKind::Loader { name, workers, prefetch, batches, ordered, .. } => {
                            if self.chan_names.contains_key(&(inst, name.to_string())) {
                                bail!("loader `{name}`: a channel of that name exists");
                            }
                            self.chans.push(Chan::new(workers * prefetch, ordered, Some(batches)));
                            let chan = self.chans.len() - 1;
                            self.chan_names.insert((inst, name.to_string()), chan);
                            let children: Vec<ActorState> = (0..workers).map(|_| self.tasks[id].io.a.child(&sh, 0)).collect();
                            for (w, a) in children.into_iter().enumerate() {
                                let w = w as i64;
                                let vm = Vm::resume(snap.clone());
                                self.add_task(vm, a, Some(inst), Role::Worker { parent: id, chan, workers, batches, b: w }, false);
                            }
                            let t = &mut self.tasks[id];
                            t.loaders.push(chan);
                            t.workers_live += workers as usize;
                        }
                    }
                }
            }
        }
    }

    /// A worker between batches: start the next, park for a slot, or end.
    fn worker_next(&mut self, id: usize) -> Result<bool> {
        let Role::Worker { chan, b, batches, .. } = self.tasks[id].role else { unreachable!() };
        if b >= batches {
            self.finished(id)?;
            return Ok(false);
        }
        self.tasks[id].wait = Wait::Chan { chan, what: ChanWait::Begin(b), since: Instant::now() };
        if !self.retry_chan(id)? {
            return Ok(false);
        }
        // begun, or the channel was closed under it (then `b` is `batches` and the next round ends it)
        Ok(true)
    }

    /// Task `id` has walked its whole body (a main line) or its last batch (a sub-actor).
    fn finished(&mut self, id: usize) -> Result<()> {
        match self.tasks[id].role {
            Role::Main => {
                // close this instance's channels so a worker parked for a slot ends
                let chans: Vec<usize> = self.chan_names.iter().filter(|((i, _), _)| *i == id).map(|(_, c)| *c).collect();
                for c in chans {
                    self.chans[c].closed = true;
                    self.wake(c);
                }
                if self.tasks[id].workers_live > 0 {
                    self.tasks[id].wait = Wait::JoinLoaders;
                    return Ok(());
                }
                self.finalize_main(id)
            }
            Role::Parallel { parent } => {
                self.end_child(id, parent);
                let p = &mut self.tasks[parent];
                if let Wait::JoinParallel { remaining } = p.wait {
                    if remaining <= 1 {
                        p.wait = Wait::None;
                        self.ready(parent);
                    } else {
                        p.wait = Wait::JoinParallel { remaining: remaining - 1 };
                    }
                }
                Ok(())
            }
            Role::Worker { parent, .. } => {
                self.end_child(id, parent);
                let p = &mut self.tasks[parent];
                p.workers_live -= 1;
                if p.workers_live == 0 && matches!(p.wait, Wait::JoinLoaders) {
                    self.finalize_main(parent)?;
                }
                Ok(())
            }
        }
    }

    fn end_child(&mut self, id: usize, parent: usize) {
        let t = &mut self.tasks[id];
        t.done = true;
        self.live -= 1;
        t.io.a.finish_shared(&self.sh);
        let st = std::mem::take(&mut t.io.a.st);
        self.tasks[parent].io.a.st.merge(&st);
    }

    fn finalize_main(&mut self, id: usize) -> Result<()> {
        let t = &mut self.tasks[id];
        t.done = true;
        t.wait = Wait::None;
        self.live -= 1;
        for &c in &t.loaders {
            let ch = &self.chans[c];
            if let Some(total) = ch.total {
                if ch.next_take < total {
                    let name = self.chan_names.iter().find(|(_, v)| **v == c).map(|((_, n), _)| n.clone()).unwrap_or_default();
                    return Err(untaken(&name, ch.next_take, total));
                }
            }
        }
        t.io.a.finish_shared(&self.sh);
        Ok(())
    }
}
