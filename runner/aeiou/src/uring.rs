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
//! A `trace` node (`DESIGN_REVIEW.md` §3.58, `trace.rs`) runs as one task per lane on the
//! loop that reached it, joined like a `parallel`'s sub-actors: a lane task walks its lane
//! of the file with one op in flight, and a `submit` group puts its members in flight
//! together as member tasks, so they leave in one `io_uring_enter` (or `io_submit`) as the
//! application's did. The open table and the path order of `run.rs` are loop-local here,
//! without locks, since every lane of an instance is on its loop; a lane that must wait
//! parks on its instance and is woken by every change to its tables.
//!
//! A path placed in an object store (`object.rs`) goes to the object engine rather than to
//! the engine below: the op is begun on the loop, what it sends runs on the engine's tokio
//! runtime, and the answer wakes the loop through its eventfd (`Objects`).
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

use crate::backend::{open_flags, Backend, OpenFile, ALIGN};
use crate::eval::Model;
use crate::run::{fill, issue_blocking, round_out, untaken, ActorState, Ring, Shared, UringReport};
use crate::trace::{Built, LineOp, OwnedCtx, TraceFile};
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
/// (`SqThread:`); `None` without `SQPOLL`. Rings that share a poll thread state the same one.
/// Read when the loop's work is done and its ring still open, because two things make an
/// earlier reading unreliable: the thread names itself `iou-sqp-*` only when it first runs
/// (so the thread list cannot be trusted just after setup), and kernels up to 6.8 or so
/// state the pid of the ring's creator until then. By the end of a run the thread has
/// carried every submission. The kernel fills the field under a trylock of the ring and
/// states -1 when it loses, hence the few tries.
fn sq_thread(ring: RawFd, expected: bool) -> Option<i64> {
    for _ in 0..if expected { 50 } else { 1 } {
        let s = std::fs::read_to_string(format!("/proc/self/fdinfo/{ring}")).ok()?;
        let pid: Option<i64> = s.lines().find_map(|l| l.strip_prefix("SqThread:")).and_then(|v| v.trim().parse().ok());
        match pid {
            Some(p) if p > 0 => return Some(p),
            Some(_) => std::thread::yield_now(),
            None => return None,
        }
    }
    None
}

/// Run this host's instances over the event loops; returns how many loop threads ran and
/// what the rings were set up with, and the distinct `SQPOLL` threads the rings stated.
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
                            l.run(model, insts)?;
                            let sq = sq_thread(l.io.ring.as_raw_fd(), sh.opts.uring.sqpoll_idle_ms.is_some());
                            Ok((defaults, l.io.in_flight_peak as u64, sq))
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
    /// A `parallel` parent (or a trace node, or a lane at a `submit` group): sub-actors
    /// still running.
    JoinParallel { remaining: usize },
    /// A main line that has ended its body: loader workers still running.
    JoinLoaders,
    /// A trace lane parked on a table of its instance (an open not yet published, a close
    /// with uses outstanding, path order).
    Trace,
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

/// What a task runs: an actor's VM, or a lane of a `trace`.
enum Prog {
    Vm(Vm<'static, 'static>),
    Lane(Lane),
}

struct Task {
    prog: Prog,
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
            CString::new(sh.opts.path(rel).as_os_str().as_bytes()).map_err(|_| anyhow!("path contains NUL: {rel}"))
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
                t.a.opened(sh, op.path, op.aux, OpenFile::from(fd));
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

// ---------------------------------------------------------------- the object engine

/// How an op in flight ended: a completion of the engine's, or the object engine's answer.
enum Completion {
    Kernel(i32),
    Object(crate::object::Answer),
}

/// The bridge from a loop to the object engine (`object.rs`, `DESIGN_REVIEW.md` §3.65): an
/// op whose path lives in an object store is begun on the loop (`object::start`), and what
/// it sends runs on the engine's runtime, whose worker puts the answer here and writes the
/// loop's eventfd, which the loop keeps a read posted on (as for a barrier's release) while
/// any is in flight. The loop then settles each answer on its own thread, so the actor's
/// state is never touched by a worker. A read's buffer is owned by the op while it is in
/// flight, from a FIFO of buffers that grows to `--buffer-mib` as the engine's pool does.
struct Objects {
    done: Arc<Mutex<Vec<(usize, crate::object::Answer)>>>,
    efd: Arc<OwnedFd>,
    in_flight: usize,
    free: VecDeque<Vec<u8>>,
    total: usize,
    target: usize,
}

impl Objects {
    fn new(efd: Arc<OwnedFd>, target: usize) -> Self {
        Objects { done: Arc::new(Mutex::new(Vec::new())), efd, in_flight: 0, free: VecDeque::new(), total: 0, target }
    }

    fn buffer(&mut self, len: usize) -> Vec<u8> {
        if self.total >= self.target {
            if let Some(mut v) = self.free.pop_front() {
                if v.len() < len {
                    v.resize(len, 0);
                }
                return v;
            }
        }
        self.total += len;
        vec![0u8; len]
    }

    fn issue(&mut self, sh: &Shared, t: &mut TaskIo, op: &Op, store: &Arc<crate::object::Store>, rest: String) -> Issued {
        use crate::object::{Sink, Step};
        t.started = Instant::now();
        let sink = (op.kind == OpKind::Read).then(|| Sink::Owned(self.buffer(op.len as usize), op.len as usize));
        match crate::object::start(store, rest, sh, &mut t.a, op, sink) {
            Step::Done(r) => Issued::Done(r, t.started.elapsed().as_nanos() as u64),
            Step::Send(f) => {
                let (done, efd, id) = (self.done.clone(), self.efd.clone(), t.id);
                crate::object::spawn(
                    f,
                    Box::new(move |answer| {
                        done.lock().unwrap().push((id, answer));
                        let one = 1u64;
                        // SAFETY: an eventfd takes an 8-byte count; `efd` lives while this does
                        unsafe { libc::write(efd.as_raw_fd(), &one as *const u64 as *const libc::c_void, 8) };
                    }),
                );
                self.in_flight += 1;
                Issued::Pending
            }
        }
    }

    /// The answers that have come back.
    fn take(&mut self) -> Vec<(usize, crate::object::Answer)> {
        let got = std::mem::take(&mut *self.done.lock().unwrap());
        self.in_flight -= got.len();
        got
    }

    fn settle(&mut self, t: &mut TaskIo, op: &Op, answer: crate::object::Answer) -> std::io::Result<i64> {
        let (r, buf) = crate::object::settle(&mut t.a, op, answer);
        if let Some(v) = buf {
            self.free.push_back(v);
        }
        r
    }
}

/// The engine of a loop with no POSIX API: a run of S3 names alone under `--s3 async`.
/// Every op goes to the object engine, so this carries none; it only waits for the loop's
/// eventfd (`poll(2)`), which the runtime's workers write, or for the next timer.
pub(crate) struct WaitIo {
    efd: Option<RawFd>,
}

impl Engine for WaitIo {
    fn issue(&mut self, _: &Shared, _: &mut TaskIo, op: &Op) -> Result<Issued> {
        bail!("{} {}: a POSIX op on a loop with no POSIX API (every name of the abstract is `protocol: s3`)", op.kind.name(), op.path)
    }

    fn complete(&mut self, _: &Shared, _: &mut TaskIo, op: &Op, _: i32) -> std::io::Result<i64> {
        unreachable!("{} {}: no op is in flight on a loop with no POSIX API", op.kind.name(), op.path)
    }

    fn in_flight(&self) -> usize {
        0
    }

    fn post_wake(&mut self, efd: RawFd, _: *mut u64) -> Result<()> {
        self.efd = Some(efd);
        Ok(())
    }

    fn wait(&mut self, timeout: Duration, out: &mut Vec<(u64, i32)>) -> Result<()> {
        let Some(efd) = self.efd else {
            std::thread::sleep(timeout);
            return Ok(());
        };
        let mut p = libc::pollfd { fd: efd, events: libc::POLLIN, revents: 0 };
        let ms = timeout.as_millis().clamp(0, i32::MAX as u128) as i32;
        // SAFETY: one live pollfd
        let n = unsafe { libc::poll(&mut p, 1, ms) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            return if e.raw_os_error() == Some(libc::EINTR) { Ok(()) } else { Err(e).context("poll") };
        }
        if n > 0 {
            // take the count, so the next poll waits for the next write
            let mut v = 0u64;
            unsafe { libc::read(efd, &mut v as *mut u64 as *mut libc::c_void, 8) };
            self.efd = None;
            out.push((EFD, 0));
        }
        Ok(())
    }
}

/// Run this host's instances over loops of `WaitIo`; returns how many loop threads ran.
pub(crate) fn run_s3(model: &'static Model<'static>, sh: &Arc<Shared>, counts: &[(&'static str, i64)], ranges: &[(i64, i64)]) -> Result<usize> {
    let per = spread(sh, counts, ranges);
    let loops = per.len();
    std::thread::scope(|s| {
        let handles: Vec<_> = per
            .into_iter()
            .enumerate()
            .map(|(i, insts)| {
                let sh = sh.clone();
                std::thread::Builder::new()
                    .name(format!("s3 loop {i}"))
                    .spawn_scoped(s, move || {
                        let r = Loop::with(sh.clone(), i, WaitIo { efd: None }).and_then(|mut l| l.run(model, insts));
                        if r.is_err() {
                            sh.aborted.store(true, Ordering::Relaxed);
                        }
                        r
                    })
                    .expect("spawn event loop")
            })
            .collect();
        let mut first: Option<anyhow::Error> = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
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
            None => Ok(loops),
        }
    })
}

/// Issue `op` through the object engine when its path lives in an object store, through the
/// loop's engine otherwise.
fn issue<E: Engine>(io: &mut E, objects: &mut Objects, sh: &Shared, t: &mut TaskIo, op: &Op) -> Result<Issued> {
    match sh.opts.endpoints.object(op.path) {
        Some((store, rest)) => Ok(objects.issue(sh, t, op, &store, rest)),
        None => io.issue(sh, t, op),
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
    /// The `trace` instances begun on this loop.
    traces: Vec<TraceInst>,
    efd: Arc<OwnedFd>,
    efd_buf: Box<u64>,
    efd_posted: bool,
    /// The object engine's ops in flight from this loop.
    objects: Objects,
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
        let efd = Arc::new(unsafe { OwnedFd::from_raw_fd(efd) });
        sh.coord.subscribe(efd.as_raw_fd());
        let objects = Objects::new(efd.clone(), sh.opts.buffer_bytes);
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
            traces: Vec::new(),
            efd,
            efd_buf: Box::new(0),
            efd_posted: false,
            objects,
            live: 0,
        })
    }

    fn add_task(&mut self, prog: Prog, a: ActorState, inst: Option<usize>, role: Role, active: bool) -> usize {
        let id = self.tasks.len();
        self.tasks.push(Task {
            prog,
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
        match &self.tasks[id].prog {
            Prog::Vm(vm) => {
                let (_, ctx) = vm.current();
                let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
                format!("{}#{} [{}]", ctx.template, ctx.actor, idx.join(","))
            }
            Prog::Lane(l) => l.label(),
        }
    }

    pub(crate) fn run(&mut self, model: &'static Model<'static>, insts: Vec<(&'static str, i64, i64)>) -> Result<()> {
        for (template, actor, count) in insts {
            let mut vm = Vm::new(model, template, actor, count);
            vm.start(&model.ast.actors[template].body);
            let a = ActorState::new(&self.sh, template, actor, true, None, 0);
            self.add_task(Prog::Vm(vm), a, None, Role::Main, true);
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
            if (!self.barrier_waiters.is_empty() || self.objects.in_flight > 0) && !self.efd_posted {
                self.io.post_wake(self.efd.as_raw_fd(), &mut *self.efd_buf as *mut u64)?;
                self.efd_posted = true;
            }
            if self.io.in_flight() == 0 && self.timers.is_empty() && !self.efd_posted {
                let parked: Vec<String> = (0..self.tasks.len()).filter(|i| !self.tasks[*i].done).map(|i| self.label(i)).collect();
                bail!("event loop {}: {} actor(s) wait on channels or trace order nothing will complete: {}", self.index, parked.len(), parked.join(", "));
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
                    self.complete(ud as usize, Completion::Kernel(res))?;
                }
            }
            if self.objects.in_flight > 0 {
                for (id, answer) in self.objects.take() {
                    self.complete(id, Completion::Object(answer))?;
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
    fn complete(&mut self, id: usize, c: Completion) -> Result<()> {
        let t = &mut self.tasks[id];
        let ns = t.io.started.elapsed().as_nanos() as u64;
        let (io, objects, sh) = (&mut self.io, &mut self.objects, &self.sh);
        let done = |t: &mut TaskIo, op: &Op| match c {
            Completion::Kernel(res) => io.complete(sh, t, op, res),
            Completion::Object(answer) => objects.settle(t, op, answer),
        };
        match &t.prog {
            Prog::Vm(vm) => {
                let (op, ctx) = vm.current();
                let r = done(&mut t.io, &op);
                t.io.a.settle(&op, &ctx, r, ns)?;
                t.wait = Wait::None;
                self.ready(id);
            }
            Prog::Lane(l) => {
                let r = {
                    let b = l.built();
                    done(&mut t.io, &b.op)
                };
                t.wait = Wait::None;
                match self.lane_finish(id, r, ns)? {
                    Advance::Ended => self.finished(id)?,
                    _ => self.ready(id),
                }
            }
        }
        Ok(())
    }

    /// A task parked on a channel tries again. `Ok(true)`: it may go on.
    fn retry_chan(&mut self, id: usize) -> Result<bool> {
        let Wait::Chan { chan, what, since } = self.tasks[id].wait else { return Ok(true) };
        match what {
            ChanWait::Begin(b) => match self.chans[chan].try_begin(b) {
                Ok(true) => {
                    let t = &mut self.tasks[id];
                    t.vm().start_sub(b);
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
        if matches!(self.tasks[id].prog, Prog::Lane(_)) {
            return self.step_lane(id);
        }
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
            let Prog::Vm(vm) = &mut t.prog else { unreachable!("a lane stepped as a VM") };
            match vm.next()? {
                None => {
                    t.active = false;
                    if let Role::Worker { chan, ref mut b, workers, .. } = t.role {
                        let done = *b;
                        *b += workers;
                        self.chans[chan].end(done);
                        self.wake(chan);
                    }
                }
                Some(Event::Trace(tf, ctx)) => {
                    let base = OwnedCtx::of(&ctx);
                    if self.start_trace(id, tf, base) {
                        return Ok(());
                    }
                }
                Some(Event::Op(op, ctx)) => {
                    t.io.a.check_align(&sh, &op)?;
                    match issue(&mut self.io, &mut self.objects, &sh, &mut t.io, &op)? {
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
                    let snap = vm.snapshot();
                    vm.accept_fork();
                    let inst = t.inst;
                    match kind {
                        ForkKind::Parallel { width, .. } => {
                            let children: Vec<ActorState> = (0..width).map(|_| t.io.a.child(&sh, 0)).collect();
                            for (k, a) in children.into_iter().enumerate() {
                                let mut vm = Vm::resume(snap.clone());
                                vm.start_sub(k as i64);
                                self.add_task(Prog::Vm(vm), a, Some(inst), Role::Parallel { parent: id }, true);
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
                                self.add_task(Prog::Vm(vm), a, Some(inst), Role::Worker { parent: id, chan, workers, batches, b: w }, false);
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

// ---------------------------------------------------------------- the `trace` node

impl Task {
    fn vm(&mut self) -> &mut Vm<'static, 'static> {
        match &mut self.prog {
            Prog::Vm(vm) => vm,
            Prog::Lane(_) => unreachable!("a trace lane has no VM"),
        }
    }
}

/// One `trace` instance under way on this loop: the open table and the path order of
/// `run.rs` (`OpenTable`, `PathOrder`) without their locks, and the lanes parked on them.
struct TraceInst {
    slots: Vec<Slot>,
    /// Per path the trace changes: its ops completed, its changing ops completed.
    done: Vec<(usize, usize)>,
    /// Lanes parked on a table of this instance; all woken on every change.
    waiters: Vec<usize>,
}

struct Slot {
    fd: Option<Arc<OpenFile>>,
    /// The open failed in this run (its uses, if any, cannot proceed).
    failed: bool,
    uses_left: usize,
    closes_left: usize,
}

impl TraceInst {
    fn new(tf: &TraceFile) -> Self {
        TraceInst {
            slots: tf.opens.iter().map(|o| Slot { fd: None, failed: false, uses_left: o.uses, closes_left: o.closes }).collect(),
            done: vec![(0, 0); tf.changed.len()],
            waiters: Vec::new(),
        }
    }

    /// Path order: may op `g` go?
    fn ordered(&self, tf: &TraceFile, g: usize) -> bool {
        let ok = |d: &crate::trace::Dep| {
            let (done, mdone) = self.done[d.path];
            if d.mutating { done >= d.k } else { mdone >= d.mk }
        };
        tf.deps[g].as_ref().map_or(true, ok) && tf.deps2.get(&g).map_or(true, ok)
    }

    fn completed(&mut self, tf: &TraceFile, g: usize) {
        for d in tf.deps[g].iter().chain(tf.deps2.get(&g)) {
            self.done[d.path].0 += 1;
            if d.mutating {
                self.done[d.path].1 += 1;
            }
        }
    }
}

/// What a lane task runs.
enum LaneProg {
    /// Lane `lane` of the file, its next line `tf.lanes[lane][next]`; `begun` and `end` are
    /// the gap's bookkeeping (whether a line has gone, and where the last one ended).
    Lines { lane: usize, next: usize, begun: bool, end: i64 },
    /// Member `j` of the `submit` on line `line`.
    Member { line: usize, j: usize },
}

/// Where a lane task is in its current line.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    /// Between lines: the next line's gap goes out as `compute`.
    Next,
    /// The gap has elapsed: build the op (or fan a group out).
    Build,
    /// Built; path order.
    Order,
    /// Ordered; the open (or, for a close, its uses), then the engine.
    Acquire,
    /// On the engine.
    InFlight,
    /// A group's members are running as tasks.
    Join,
}

/// The op a lane has under way: enough to build it again at completion (`TraceFile::build`
/// is a function of the line and the position it was built at).
#[derive(Clone, Copy)]
struct Cur {
    line: usize,
    member: Option<usize>,
    /// The op's ordinal in the file (the key of its path order).
    g: usize,
    oid: Option<usize>,
    /// The position a sequential read or write was built at.
    off: i64,
    /// A close: counted against its open's closes already.
    counted: bool,
}

struct Lane {
    tf: Arc<TraceFile>,
    inst: usize,
    base: Arc<OwnedCtx>,
    prog: LaneProg,
    /// The enclosing indices plus `[lane, ordinal of the line]`.
    idx: Vec<i64>,
    /// Positions per open id, for the lane's sequential reads and writes.
    pos: HashMap<usize, i64>,
    stage: Stage,
    cur: Option<Cur>,
}

impl Lane {
    fn line_op<'c>(tf: &'c TraceFile, c: &Cur) -> &'c LineOp {
        match (c.member, &tf.lines[c.line].op) {
            (Some(j), LineOp::Submit { ops }) => &ops[j],
            (None, op) => op,
            _ => unreachable!("a member of a line that is not a group"),
        }
    }

    fn built_of(&self, c: &Cur) -> Built<'_> {
        self.tf.build(Self::line_op(&self.tf, c), |_| c.off)
    }

    /// The op under way, built again.
    fn built(&self) -> Built<'_> {
        self.built_of(self.cur.as_ref().expect("a lane with an op under way"))
    }

    fn lane(&self) -> usize {
        match self.prog {
            LaneProg::Lines { lane, .. } => lane,
            LaneProg::Member { line, .. } => self.tf.lines[line].lane,
        }
    }

    fn label(&self) -> String {
        let idx: Vec<String> = self.idx.iter().map(|i| i.to_string()).collect();
        format!("{}#{} [{}] trace `{}` lane {}", self.base.template, self.base.actor, idx.join(","), self.tf.name, self.lane())
    }
}

/// How a lane's step ended.
enum Advance {
    /// On to the next stage.
    Go,
    /// Parked: a timer, an op on the engine, a join.
    Parked,
    /// Parked on a table of its instance.
    Blocked,
    /// The lane (or the member) has walked its program.
    Ended,
}

impl<E: Engine> Loop<E> {
    /// `Event::Trace` from task `id`: the instance's tables and one task per lane, which
    /// the task joins as a `parallel` parent does. `false` when the trace has no lane.
    fn start_trace(&mut self, id: usize, tf: Arc<TraceFile>, base: OwnedCtx) -> bool {
        let lanes = tf.lanes();
        if lanes == 0 {
            return false;
        }
        let inst = self.traces.len();
        self.traces.push(TraceInst::new(&tf));
        let base = Arc::new(base);
        let sh = self.sh.clone();
        let parent_inst = self.tasks[id].inst;
        let children: Vec<ActorState> = (0..lanes).map(|_| self.tasks[id].io.a.child(&sh, 0)).collect();
        for (lane, a) in children.into_iter().enumerate() {
            let mut idx = base.indices.clone();
            idx.extend_from_slice(&[lane as i64, 0]);
            let l = Lane { tf: tf.clone(), inst, base: base.clone(), prog: LaneProg::Lines { lane, next: 0, begun: false, end: 0 }, idx, pos: HashMap::new(), stage: Stage::Next, cur: None };
            self.add_task(Prog::Lane(l), a, Some(parent_inst), Role::Parallel { parent: id }, true);
        }
        self.tasks[id].wait = Wait::JoinParallel { remaining: lanes };
        true
    }

    fn lane(&mut self, id: usize) -> &mut Lane {
        match &mut self.tasks[id].prog {
            Prog::Lane(l) => l,
            Prog::Vm(_) => unreachable!("a VM task stepped as a lane"),
        }
    }

    /// Every lane parked on instance `inst` tries again.
    fn wake_trace(&mut self, inst: usize) {
        let waiters = std::mem::take(&mut self.traces[inst].waiters);
        for id in waiters {
            self.tasks[id].wait = Wait::None;
            self.ready(id);
        }
    }

    /// Advance lane task `id` until it parks or ends.
    fn step_lane(&mut self, id: usize) -> Result<()> {
        loop {
            if !matches!(self.tasks[id].wait, Wait::None) {
                return Ok(());
            }
            match self.lane_advance(id)? {
                Advance::Go => {}
                Advance::Parked => return Ok(()),
                Advance::Blocked => {
                    let inst = self.lane(id).inst;
                    self.tasks[id].wait = Wait::Trace;
                    self.traces[inst].waiters.push(id);
                    return Ok(());
                }
                Advance::Ended => return self.finished(id),
            }
        }
    }

    /// One stage of lane task `id`.
    fn lane_advance(&mut self, id: usize) -> Result<Advance> {
        let sh = self.sh.clone();
        let stage = self.lane(id).stage;
        match stage {
            Stage::Next => {
                let t = &mut self.tasks[id];
                let Prog::Lane(l) = &mut t.prog else { unreachable!() };
                match &mut l.prog {
                    LaneProg::Lines { lane, next, begun, end } => {
                        let Some(&li) = l.tf.lanes[*lane].get(*next) else { return Ok(Advance::Ended) };
                        let line = &l.tf.lines[li];
                        let gap = if *begun { line.t - *end } else { line.t };
                        *begun = true;
                        *end = line.t + line.dur;
                        let n = l.idx.len();
                        l.idx[n - 1] = *next as i64;
                        l.stage = Stage::Build;
                        // the gap before the line, as `compute` (`run.rs` `trace_lane`)
                        let ns = gap.max(0) as u64;
                        t.io.a.computed(ns);
                        let scaled = (ns as f64 * sh.opts.time_scale) as u64;
                        if scaled > 0 {
                            t.wait = Wait::Timer;
                            self.timers.push(Reverse((Instant::now() + Duration::from_nanos(scaled), id)));
                            return Ok(Advance::Parked);
                        }
                        Ok(Advance::Go)
                    }
                    LaneProg::Member { .. } => {
                        l.stage = Stage::Build;
                        Ok(Advance::Go)
                    }
                }
            }
            Stage::Build => {
                let t = &mut self.tasks[id];
                let Prog::Lane(l) = &mut t.prog else { unreachable!() };
                let (line_i, member) = match l.prog {
                    LaneProg::Lines { lane, next, .. } => (l.tf.lanes[lane][next], None),
                    LaneProg::Member { line, j } => (line, Some(j)),
                };
                let line = &l.tf.lines[line_i];
                if let (None, LineOp::Submit { ops }) = (member, &line.op) {
                    // a group: its members in flight together as member tasks, joined before
                    // the lane goes on; built at the lane's positions, which they leave alone
                    let n = ops.len();
                    l.stage = Stage::Join;
                    if n == 0 {
                        return Ok(Advance::Go);
                    }
                    let (tf, inst, base, idx, pos) = (l.tf.clone(), l.inst, l.base.clone(), l.idx.clone(), l.pos.clone());
                    let children: Vec<ActorState> = (0..n).map(|_| t.io.a.child(&sh, 0)).collect();
                    let parent_inst = t.inst;
                    for (j, a) in children.into_iter().enumerate() {
                        let m = Lane { tf: tf.clone(), inst, base: base.clone(), prog: LaneProg::Member { line: line_i, j }, idx: idx.clone(), pos: pos.clone(), stage: Stage::Next, cur: None };
                        self.add_task(Prog::Lane(m), a, Some(parent_inst), Role::Parallel { parent: id }, true);
                    }
                    self.tasks[id].wait = Wait::JoinParallel { remaining: n };
                    return Ok(Advance::Parked);
                }
                let lop = match (member, &line.op) {
                    (Some(j), LineOp::Submit { ops }) => &ops[j],
                    (None, op) => op,
                    _ => unreachable!(),
                };
                // `build` asks for the position of the op's own open id only
                let off = lop.fd().map(|fd| l.pos.get(&fd).copied().unwrap_or(0)).unwrap_or(0);
                let b = l.tf.build(lop, |_| off);
                let g = line.g0 + member.unwrap_or(0);
                t.io.a.check_align(&sh, &b.op)?;
                let (oid, pos_after) = (b.oid, b.pos_after);
                if let (Some(fd), Some(p), None) = (oid, pos_after, member) {
                    l.pos.insert(fd, p);
                }
                l.cur = Some(Cur { line: line_i, member, g, oid, off, counted: false });
                l.stage = Stage::Order;
                Ok(Advance::Go)
            }
            Stage::Order => {
                let Prog::Lane(l) = &self.tasks[id].prog else { unreachable!() };
                let g = l.cur.as_ref().expect("built").g;
                if !self.traces[l.inst].ordered(&l.tf, g) {
                    return Ok(Advance::Blocked);
                }
                self.lane(id).stage = Stage::Acquire;
                Ok(Advance::Go)
            }
            Stage::Acquire => {
                let t = &mut self.tasks[id];
                let Prog::Lane(l) = &mut t.prog else { unreachable!() };
                let mut c = l.cur.expect("built");
                let inst = &mut self.traces[l.inst];
                let b = l.built_of(&c);
                let kind = b.op.kind;
                match kind {
                    OpKind::Close => {
                        // the descriptor closes with the last close line, after every use
                        // (`OpenTable::close`); an earlier close line is just a line
                        let oid = c.oid.expect("a close has its id");
                        let s = &mut inst.slots[oid];
                        if !c.counted {
                            s.closes_left = s.closes_left.saturating_sub(1);
                            c.counted = true;
                            l.cur = Some(c);
                        }
                        if s.closes_left > 0 {
                            return self.lane_finish(id, Ok(0), 0);
                        }
                        if s.uses_left > 0 {
                            return Ok(Advance::Blocked);
                        }
                        let at = Instant::now();
                        drop(s.fd.take());
                        let ns = at.elapsed().as_nanos() as u64;
                        return self.lane_finish(id, Ok(0), ns);
                    }
                    OpKind::Open => {}
                    _ => {
                        if let Some(oid) = c.oid {
                            // the table's descriptor, lent to this task under the op's path
                            // for the engine's `fd(path)`; given back at completion
                            let s = &inst.slots[oid];
                            if s.failed {
                                bail!("{}: open #{oid} of {} failed in this run; its later uses cannot be issued", l.label(), b.op.path);
                            }
                            let Some(fd) = s.fd.clone() else { return Ok(Advance::Blocked) };
                            t.io.a.fds.own.insert(Arc::from(b.op.path), fd);
                        }
                    }
                }
                match issue(&mut self.io, &mut self.objects, &sh, &mut t.io, &b.op)? {
                    Issued::Done(r, ns) => self.lane_finish(id, r, ns),
                    Issued::Pending => {
                        l.stage = Stage::InFlight;
                        t.wait = Wait::Io;
                        Ok(Advance::Parked)
                    }
                }
            }
            Stage::InFlight => unreachable!("a lane stepped with its op on the engine"),
            Stage::Join => {
                let l = self.lane(id);
                let LaneProg::Lines { next, .. } = &mut l.prog else { unreachable!("a member is one op") };
                *next += 1;
                l.stage = Stage::Next;
                Ok(Advance::Go)
            }
        }
    }

    /// The op under way has its result: give the descriptor back to the table, keep the
    /// tables, settle the op as every op is, and move the lane on.
    fn lane_finish(&mut self, id: usize, r: std::io::Result<i64>, ns: u64) -> Result<Advance> {
        let t = &mut self.tasks[id];
        let Prog::Lane(l) = &mut t.prog else { unreachable!() };
        let c = l.cur.take().expect("an op under way");
        let inst = &mut self.traces[l.inst];
        let b = l.built_of(&c);
        let key: Arc<str> = Arc::from(b.op.path);
        match b.op.kind {
            OpKind::Open => {
                // the descriptor belongs to the table, not to this task's own files
                let oid = c.oid.expect("an open has its id");
                let fd = t.io.a.fds.own.remove(&key);
                let s = &mut inst.slots[oid];
                s.failed = fd.is_none();
                s.fd = fd;
            }
            OpKind::Close => {}
            _ => {
                if let Some(oid) = c.oid {
                    t.io.a.fds.own.remove(&key);
                    inst.slots[oid].uses_left = inst.slots[oid].uses_left.saturating_sub(1);
                }
            }
        }
        inst.completed(&l.tf, c.g);
        let ctx = l.base.ctx(&l.idx);
        t.io.a.settle(&b.op, &ctx, r, ns)?;
        let ended = match &mut l.prog {
            LaneProg::Lines { next, .. } => {
                *next += 1;
                l.stage = Stage::Next;
                false
            }
            LaneProg::Member { .. } => true,
        };
        let inst = l.inst;
        self.wake_trace(inst);
        Ok(if ended { Advance::Ended } else { Advance::Go })
    }
}
