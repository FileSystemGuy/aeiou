//! `aeiou run`: execute an abstract against a directory with a blocking backend. One OS thread
//! per actor instance, one per loader worker, one per `parallel` sub-actor (the `sync`
//! fidelity reference of `PROJECT_BRIEF.md` §5, *Backend order*: literally what a PyTorch
//! worker does). The VM of `vm.rs` walks the body; this module is its `Sink`: it issues each
//! op through the backend, times it, checks the result structurally (byte counts, expected
//! errnos, entry counts), sums the fingerprint, and implements the control nodes: `compute`
//! sleeps, `barrier` goes to the coordinator, a `loader` is `workers` threads feeding an
//! ordered channel of `workers × prefetch` slots that `take` drains in batch order, and
//! `channel`/`put`/`take` are the general form.
//!
//! Startup checks: every dataset's `.aeiou-dataset.json` matches the resolved definition
//! (`schema/README.md` §6); namespace roots are empty (or `--clean-namespaces`); barrier
//! participants are counted statically. Measurement: per-op latency histograms by kind, and
//! per `take` the stall (time blocked) and the compute issued until the next take, kept per
//! actor instance in take order so steady state can be selected after the run.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use crate::ast::{Ast, Node};
use crate::backend::{errno_name, Backend, BackendKind, ALIGN};
use crate::coord::{Coordinator, Local};
use crate::dryrun::{human_bytes, human_ns};
use crate::eval::{Config, Model, Params};
use crate::payload::{self, Filler, Manifest, NamespaceManifest, RankRecord};
use crate::vm::{actor_counts, op_hash, Control, ForkKind, Op, OpCtx, OpKind, Sink, Snapshot, Vm};

// ---------------------------------------------------------------- options

#[derive(Debug, Clone)]
pub struct RunOpts {
    /// The directory the abstract's paths are relative to.
    pub root: PathBuf,
    pub backend: BackendKind,
    /// Per-thread read and write buffer ring, bytes.
    pub buffer_bytes: usize,
    /// Compression ratio of written content.
    pub write_compress: u64,
    /// Multiplier on `compute` sleeps.
    pub time_scale: f64,
    pub clean_namespaces: bool,
    pub expect_fingerprint: Option<u64>,
    /// Refuse to start unless every dataset id is in this list (when given).
    pub expect_dataset_ids: Vec<String>,
    /// This host's index and the host count; the GPU ids this host runs are
    /// `gpu_range(gpus, ranks, rank, rank_rotate)`.
    pub rank: i64,
    pub ranks: i64,
    /// Run the GPU range of rank `(rank + rank_rotate) mod ranks`: with the same host list
    /// as the run that wrote an input namespace, a non-zero rotation makes every host read
    /// what another host wrote.
    pub rank_rotate: i64,
    /// Refuse when an input namespace was finished more than this many seconds ago.
    pub max_gap: Option<f64>,
    /// Refuse when this host would read an input object it wrote itself.
    pub require_cold: bool,
}

/// The `[lo, hi)` of instance ids host `rank` of `ranks` runs for a template of `count`
/// instances: contiguous ranges of `ceil(count / ranks)`, rotated by `rotate` hosts
/// (`NAPKIN_MATH.md` §8.A: rank is only the host index that selects a GPU id range).
pub fn gpu_range(count: i64, ranks: i64, rank: i64, rotate: i64) -> (i64, i64) {
    let ranks = ranks.max(1);
    let r = (rank + rotate).rem_euclid(ranks);
    let per = (count + ranks - 1) / ranks;
    let lo = (r * per).min(count);
    let hi = ((r + 1) * per).min(count);
    (lo, hi)
}

pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    let r = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if r == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        if let Ok(s) = std::str::from_utf8(&buf[..end]) {
            return s.to_string();
        }
    }
    std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "?".into())
}

pub fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

// ---------------------------------------------------------------- statistics

/// Log-linear latency histogram: four buckets per octave of nanoseconds.
#[derive(Debug, Clone)]
pub struct LatHist {
    pub buckets: Vec<u64>,
    pub count: u64,
    pub sum: u64,
    pub max: u64,
}

impl Default for LatHist {
    fn default() -> Self {
        LatHist { buckets: vec![0; 256], count: 0, sum: 0, max: 0 }
    }
}

impl LatHist {
    fn idx(ns: u64) -> usize {
        if ns < 2 {
            return 0;
        }
        let l = 63 - ns.leading_zeros() as usize;
        let frac = if l >= 2 { ((ns >> (l - 2)) & 3) as usize } else { 0 };
        (l * 4 + frac).min(255)
    }

    fn lower(i: usize) -> u64 {
        let l = i / 4;
        let frac = (i % 4) as u64;
        if l < 2 {
            return 1u64 << l;
        }
        (1u64 << l) + (frac << (l - 2))
    }

    pub fn add(&mut self, ns: u64) {
        self.buckets[Self::idx(ns)] += 1;
        self.count += 1;
        self.sum += ns;
        self.max = self.max.max(ns);
    }

    pub fn merge(&mut self, o: &LatHist) {
        for (a, b) in self.buckets.iter_mut().zip(&o.buckets) {
            *a += b;
        }
        self.count += o.count;
        self.sum += o.sum;
        self.max = self.max.max(o.max);
    }

    /// The lower bound of the bucket holding quantile `q` (0..1).
    pub fn quantile(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((self.count as f64) * q).ceil().max(1.0) as u64;
        let mut acc = 0;
        for (i, b) in self.buckets.iter().enumerate() {
            acc += b;
            if acc >= target {
                return Self::lower(i);
            }
        }
        self.max
    }

    pub fn mean(&self) -> u64 {
        if self.count == 0 { 0 } else { self.sum / self.count }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PhaseStats {
    pub ops: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub io_ns: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub ops: u64,
    pub counts: BTreeMap<OpKind, u64>,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub fingerprint: u64,
    pub lat: BTreeMap<OpKind, LatHist>,
    pub io_ns: u64,
    pub expected_errors: u64,
    pub compute_ns: u64,
    pub barriers: u64,
    pub barrier_wait_ns: u64,
    pub puts: u64,
    pub takes: u64,
    pub phases: BTreeMap<String, PhaseStats>,
    pub threads: u64,
    /// Opens of objects of input namespaces, and of those the ones written on this host.
    pub input_opens: u64,
    pub warm_opens: u64,
}

impl Stats {
    pub fn merge(&mut self, o: &Stats) {
        self.ops += o.ops;
        for (k, v) in &o.counts {
            *self.counts.entry(*k).or_insert(0) += v;
        }
        self.bytes_read += o.bytes_read;
        self.bytes_written += o.bytes_written;
        self.fingerprint = self.fingerprint.wrapping_add(o.fingerprint);
        for (k, h) in &o.lat {
            self.lat.entry(*k).or_default().merge(h);
        }
        self.io_ns += o.io_ns;
        self.expected_errors += o.expected_errors;
        self.compute_ns += o.compute_ns;
        self.barriers += o.barriers;
        self.barrier_wait_ns += o.barrier_wait_ns;
        self.puts += o.puts;
        self.takes += o.takes;
        for (k, p) in &o.phases {
            let e = self.phases.entry(k.clone()).or_default();
            e.ops += p.ops;
            e.bytes_read += p.bytes_read;
            e.bytes_written += p.bytes_written;
            e.io_ns += p.io_ns;
        }
        self.threads += o.threads;
        self.input_opens += o.input_opens;
        self.warm_opens += o.warm_opens;
    }
}

/// One `take` of an actor instance: how long it blocked, and the compute issued after it
/// before the next take.
#[derive(Debug, Clone, Copy, Default)]
pub struct TakeRec {
    pub stall_ns: u64,
    pub compute_ns: u64,
}

pub struct ActorRecord {
    pub template: String,
    pub actor: i64,
    pub takes: Vec<TakeRec>,
    pub elapsed: Duration,
}

// ---------------------------------------------------------------- channels

struct ChanState {
    next_take: i64,
    done: BTreeSet<i64>,
    closed: bool,
    failed: Option<String>,
}

/// A bounded channel. A `loader` bounds batches *started* (a worker takes a slot before it
/// begins a batch, as PyTorch dispatches indices); a `channel` bounds items *delivered*.
pub struct Channel {
    m: Mutex<ChanState>,
    cv: Condvar,
    capacity: i64,
    ordered: bool,
    /// Items the producer side will deliver in all (a loader's `batches`).
    total: Option<i64>,
}

const POLL: Duration = Duration::from_millis(50);

impl Channel {
    fn new(capacity: i64, ordered: bool, total: Option<i64>) -> Self {
        Channel {
            m: Mutex::new(ChanState { next_take: 0, done: BTreeSet::new(), closed: false, failed: None }),
            cv: Condvar::new(),
            capacity: capacity.max(1),
            ordered,
            total,
        }
    }

    /// Loader worker: wait for a slot to start batch `b`.
    fn begin(&self, b: i64, aborted: &AtomicBool) -> Result<()> {
        let mut st = self.m.lock().unwrap();
        loop {
            if st.closed {
                bail!("channel closed before batch {b} started");
            }
            if b < st.next_take + self.capacity {
                return Ok(());
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("run aborted");
            }
            st = self.cv.wait_timeout(st, POLL).unwrap().0;
        }
    }

    /// Loader worker: batch `b` is complete.
    fn end(&self, b: i64) {
        let mut st = self.m.lock().unwrap();
        st.done.insert(b);
        self.cv.notify_all();
    }

    /// General producer: deliver item `seq`, blocking while `capacity` items are undelivered.
    fn put(&self, seq: i64, aborted: &AtomicBool) -> Result<()> {
        let mut st = self.m.lock().unwrap();
        loop {
            if st.closed {
                bail!("put {seq} on a closed channel");
            }
            if (st.done.len() as i64) < self.capacity {
                st.done.insert(seq);
                self.cv.notify_all();
                return Ok(());
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("run aborted");
            }
            st = self.cv.wait_timeout(st, POLL).unwrap().0;
        }
    }

    fn take(&self, aborted: &AtomicBool) -> Result<i64> {
        let mut st = self.m.lock().unwrap();
        loop {
            if let Some(e) = &st.failed {
                bail!("producer failed: {e}");
            }
            if let Some(total) = self.total {
                if st.next_take >= total {
                    bail!("take after the last of {total} items");
                }
            }
            let got = if self.ordered {
                let want = st.next_take;
                if st.done.remove(&want) { Some(want) } else { None }
            } else {
                st.done.iter().next().copied().map(|x| {
                    st.done.remove(&x);
                    x
                })
            };
            if let Some(x) = got {
                st.next_take += 1;
                self.cv.notify_all();
                return Ok(x);
            }
            if st.closed {
                bail!("take on a closed channel with nothing delivered");
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("run aborted");
            }
            st = self.cv.wait_timeout(st, POLL).unwrap().0;
        }
    }

    fn fail(&self, msg: String) {
        let mut st = self.m.lock().unwrap();
        st.failed = Some(msg);
        st.closed = true;
        self.cv.notify_all();
    }

    fn close(&self) {
        let mut st = self.m.lock().unwrap();
        st.closed = true;
        self.cv.notify_all();
    }

    fn taken(&self) -> i64 {
        self.m.lock().unwrap().next_take
    }
}

// ---------------------------------------------------------------- shared state

struct Shared {
    opts: RunOpts,
    coord: Local,
    aborted: AtomicBool,
    stats: Mutex<Stats>,
    actors: Mutex<Vec<ActorRecord>>,
    /// Barrier scopes each template participates in.
    scopes: HashMap<String, Vec<String>>,
    host: String,
    /// Objects of input namespaces: path → the host that wrote it (when recorded).
    input_objects: HashMap<String, Option<String>>,
    /// Objects this run created (path, creating GPU id), for the namespace manifests, and
    /// paths it removed (unlink, the source of a rename).
    created: Mutex<Vec<(String, i64)>>,
    removed: Mutex<Vec<String>>,
}

/// Per actor instance: its channels and its loader threads.
struct Instance {
    channels: Mutex<HashMap<String, Arc<Channel>>>,
    loaders: Mutex<Vec<(String, JoinHandle<(Result<()>, Stats)>)>>,
}

type Fds = HashMap<Arc<str>, Arc<OwnedFd>>;

/// Files opened by this thread, plus what the parent had open at the fork.
struct FdTable {
    own: Fds,
    inherited: Option<Arc<FdTable>>,
    closed: HashSet<Arc<str>>,
}

impl FdTable {
    fn get(&self, path: &str) -> Option<Arc<OwnedFd>> {
        if let Some(fd) = self.own.get(path) {
            return Some(fd.clone());
        }
        if self.closed.contains(path) {
            return None;
        }
        self.inherited.as_ref().and_then(|p| p.get(path))
    }

    fn freeze(&self) -> Arc<FdTable> {
        Arc::new(FdTable { own: self.own.clone(), inherited: self.inherited.clone(), closed: self.closed.clone() })
    }
}

/// A page-aligned buffer ring: successive I/Os land in successive slices so copies are not
/// cache-hot (`NAPKIN_MATH.md` §2.2). Allocated on first use, grown to fit the largest op.
struct Ring {
    ptr: *mut u8,
    cap: usize,
    pos: usize,
    target: usize,
}

unsafe impl Send for Ring {}

impl Ring {
    fn new(target: usize) -> Self {
        Ring { ptr: std::ptr::null_mut(), cap: 0, pos: 0, target: target.max(ALIGN) }
    }

    fn ensure(&mut self, len: usize) {
        let want = if self.cap == 0 { self.target.max(len) } else { len };
        if want > self.cap {
            let cap = ((want + ALIGN - 1) / ALIGN) * ALIGN;
            let layout = std::alloc::Layout::from_size_align(cap, ALIGN).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null(), "buffer allocation of {cap} bytes failed");
            self.free();
            self.ptr = ptr;
            self.cap = cap;
            self.pos = 0;
        }
    }

    fn slice(&mut self, len: usize) -> &mut [u8] {
        self.ensure(len);
        let aligned = ((len + ALIGN - 1) / ALIGN) * ALIGN;
        if self.pos + aligned > self.cap {
            self.pos = 0;
        }
        let s = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(self.pos), len) };
        self.pos += aligned;
        s
    }

    fn free(&mut self) {
        if !self.ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.cap, ALIGN).unwrap();
            unsafe { std::alloc::dealloc(self.ptr, layout) };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        self.free();
    }
}

// ---------------------------------------------------------------- the sink

pub struct Runner {
    sh: Arc<Shared>,
    inst: Arc<Instance>,
    be: Box<dyn Backend>,
    fds: FdTable,
    rbuf: Ring,
    wbuf: Ring,
    filler: Filler,
    st: Stats,
    takes: Vec<TakeRec>,
    created: Vec<(String, i64)>,
    removed: Vec<String>,
    /// The actor instance's main thread (not a sub-actor).
    main: bool,
    template: &'static str,
    actor: i64,
    started: Instant,
}

impl Runner {
    fn new(sh: Arc<Shared>, inst: Arc<Instance>, template: &'static str, actor: i64, main: bool, inherited: Option<Arc<FdTable>>) -> Self {
        let be = sh.opts.backend.make();
        let buf = sh.opts.buffer_bytes;
        let compress = sh.opts.write_compress;
        Runner {
            sh,
            inst,
            be,
            fds: FdTable { own: HashMap::new(), inherited, closed: HashSet::new() },
            rbuf: Ring::new(buf),
            wbuf: Ring::new(buf),
            filler: Filler::new(compress),
            st: Stats { threads: 1, ..Default::default() },
            takes: Vec::new(),
            created: Vec::new(),
            removed: Vec::new(),
            main,
            template,
            actor,
            started: Instant::now(),
        }
    }

    fn child(&self) -> Runner {
        Runner::new(self.sh.clone(), self.inst.clone(), self.template, self.actor, false, Some(self.fds.freeze()))
    }

    fn full(&self, rel: &str) -> PathBuf {
        self.sh.opts.root.join(rel)
    }

    /// `sync-direct`: an unaligned write cannot be issued (it would need read-modify-write);
    /// an unaligned read is rounded out to alignment and the requested part counted (what an
    /// `O_DIRECT` shim under a buffered application has to do).
    fn check_align(&self, op: &Op) -> Result<()> {
        if op.kind == OpKind::Write && self.sh.opts.backend.direct() && (op.offset % ALIGN as i64 != 0 || op.len % ALIGN as i64 != 0) {
            bail!(
                "write {} off={} len={}: `sync-direct` needs {ALIGN}-byte alignment for writes; use `sync`",
                op.path,
                op.offset,
                op.len
            );
        }
        Ok(())
    }

    /// Issue the op; `Ok(n)` is the count it returned (bytes, entries, or 0).
    fn issue(&mut self, op: &Op) -> std::io::Result<i64> {
        match op.kind {
            OpKind::Open => {
                let mode = (op.aux >> 32) as u32;
                let mode = if mode == 0 { 0o644 } else { mode };
                let fd = self.be.open(&self.full(op.path), op.aux & 0xffff_ffff, mode)?;
                self.fds.own.insert(Arc::from(op.path), Arc::new(fd));
                if op.aux & (1 << (crate::ast::OpenFlag::CREAT as u8)) != 0 {
                    self.created.push((op.path.to_string(), self.actor));
                }
                if let Some(writer) = self.sh.input_objects.get(op.path) {
                    self.st.input_opens += 1;
                    if writer.as_deref() == Some(self.sh.host.as_str()) {
                        self.st.warm_opens += 1;
                    }
                }
                Ok(0)
            }
            OpKind::Close => {
                let key: Arc<str> = Arc::from(op.path);
                if self.fds.own.remove(&key).is_none() {
                    if self.fds.get(op.path).is_none() {
                        return Err(std::io::Error::from_raw_os_error(libc::EBADF));
                    }
                    self.fds.closed.insert(key);
                }
                Ok(0)
            }
            OpKind::Read => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                let a = ALIGN as i64;
                if self.sh.opts.backend.direct() && (op.offset % a != 0 || op.len % a != 0) {
                    // round out; O_DIRECT needs the offset, length, and buffer aligned
                    let lo = op.offset - op.offset.rem_euclid(a);
                    let hi = (op.offset + op.len + a - 1) / a * a;
                    let buf = self.rbuf.slice((hi - lo) as usize);
                    let n = self.be.read(fd.as_fd(), buf, Some(lo))? as i64;
                    let got = (n - (op.offset - lo)).clamp(0, op.len);
                    if !op.positioned {
                        // keep the file position where a plain read would have left it
                        self.be.lseek(fd.as_fd(), op.offset + got, crate::ast::Whence::SET)?;
                    }
                    return Ok(got);
                }
                let buf = self.rbuf.slice(op.len as usize);
                let off = if op.positioned { Some(op.offset) } else { None };
                self.be.read(fd.as_fd(), buf, off).map(|n| n as i64)
            }
            OpKind::Write => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                let buf = self.wbuf.slice(op.len as usize);
                let seed = payload::object_seed(op.seed, op.path);
                self.filler.fill_range(|b| payload::block_seed(seed, 0, b), op.offset as u64, buf);
                let off = if op.positioned { Some(op.offset) } else { None };
                self.be.write(fd.as_fd(), buf, off).map(|n| n as i64)
            }
            OpKind::Lseek => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                let whence = match op.aux {
                    0 => crate::ast::Whence::SET,
                    1 => crate::ast::Whence::CUR,
                    _ => crate::ast::Whence::END,
                };
                self.be.lseek(fd.as_fd(), op.offset, whence)
            }
            OpKind::Ioctl => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                let req = match op.aux {
                    0 => crate::ast::IoctlRequest::TCGETS,
                    1 => crate::ast::IoctlRequest::FIONREAD,
                    _ => crate::ast::IoctlRequest::BLKGETSIZE64,
                };
                self.be.ioctl(fd.as_fd(), req).map(|_| 0)
            }
            OpKind::Fstat => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                self.be.fstat(fd.as_fd())
            }
            OpKind::Stat => self.be.stat(&self.full(op.path)),
            OpKind::Fsync => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                self.be.fsync(fd.as_fd()).map(|_| 0)
            }
            OpKind::Fdatasync => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                self.be.fdatasync(fd.as_fd()).map(|_| 0)
            }
            OpKind::Unlink => {
                self.be.unlink(&self.full(op.path))?;
                self.removed.push(op.path.to_string());
                Ok(0)
            }
            OpKind::Ftruncate => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                self.be.ftruncate(fd.as_fd(), op.len).map(|_| 0)
            }
            OpKind::Fallocate => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                self.be.fallocate(fd.as_fd(), op.offset, op.len).map(|_| 0)
            }
            OpKind::Mkdir => {
                self.be.mkdir(&self.full(op.path), op.aux as u32)?;
                self.created.push((op.path.to_string(), self.actor));
                Ok(0)
            }
            OpKind::Rmdir => self.be.rmdir(&self.full(op.path)).map(|_| 0),
            OpKind::Rename => {
                self.be.rename(&self.full(op.path), &self.full(op.path2.unwrap_or("")))?;
                self.removed.push(op.path.to_string());
                self.created.push((op.path2.unwrap_or("").to_string(), self.actor));
                Ok(0)
            }
            OpKind::Readdir => {
                let fd = self.fds.get(op.path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
                self.be.readdir(fd.as_fd()).map(|n| n as i64)
            }
        }
    }

    fn record(&mut self, op: &Op, ctx: &OpCtx, ns: u64, transferred: u64) {
        self.st.ops += 1;
        *self.st.counts.entry(op.kind).or_insert(0) += 1;
        self.st.lat.entry(op.kind).or_default().add(ns);
        self.st.io_ns += ns;
        match op.kind {
            OpKind::Read => self.st.bytes_read += transferred,
            OpKind::Write => self.st.bytes_written += transferred,
            _ => {}
        }
        self.st.fingerprint = self.st.fingerprint.wrapping_add(op_hash(op, ctx));
        if let Some(p) = ctx.phase {
            let e = self.st.phases.entry(p.to_string()).or_default();
            e.ops += 1;
            e.io_ns += ns;
            match op.kind {
                OpKind::Read => e.bytes_read += transferred,
                OpKind::Write => e.bytes_written += transferred,
                _ => {}
            }
        }
    }

    fn spawn_sub(&self, snap: Snapshot<'static, 'static>, label: String, work: impl FnOnce(&mut Vm<'static, 'static, Runner>) -> Result<()> + Send + 'static) -> JoinHandle<(Result<()>, Stats)> {
        let child = self.child();
        std::thread::Builder::new()
            .name(label)
            .spawn(move || {
                let mut vm = Vm::resume(child, snap);
                let r = work(&mut vm);
                let mut sink = vm.into_sink();
                let r = r.and_then(|_| sink.finish());
                if r.is_err() {
                    sink.sh.aborted.store(true, Ordering::Relaxed);
                }
                (r, std::mem::take(&mut sink.st))
            })
            .expect("spawn")
    }
}

impl Sink<'static, 'static> for Runner {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> Result<()> {
        if matches!(op.kind, OpKind::Read | OpKind::Write) {
            self.check_align(op)?;
        }
        let t = Instant::now();
        let r = self.issue(op);
        let ns = t.elapsed().as_nanos() as u64;
        let where_ = || {
            let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
            format!("{}#{} [{}] {} {}", ctx.template, ctx.actor, idx.join(","), op.kind.name(), op.path)
        };
        match r {
            Ok(n) => {
                let transferred = match op.kind {
                    OpKind::Read | OpKind::Write => {
                        if n != op.bytes {
                            bail!("{}: off={} len={} expected {} bytes, got {}: the corpus does not match the dataset definition (or a short write)", where_(), op.offset, op.len, op.bytes, n);
                        }
                        n as u64
                    }
                    OpKind::Readdir => {
                        if op.bytes >= 0 && n != op.bytes {
                            bail!("{}: expected {} entries, found {}: the corpus does not match the dataset definition", where_(), op.bytes, n);
                        }
                        0
                    }
                    _ => 0,
                };
                self.record(op, ctx, ns, transferred);
                Ok(())
            }
            Err(e) => {
                let code = e.raw_os_error().unwrap_or(0);
                let name = errno_name(code);
                if op.expect.iter().any(|x| x == name) {
                    self.st.expected_errors += 1;
                    self.record(op, ctx, ns, 0);
                    Ok(())
                } else {
                    let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
                    let mut extra = String::new();
                    if matches!(op.kind, OpKind::Read | OpKind::Write) {
                        extra = format!(" off={} len={}", op.offset, op.len);
                    }
                    Err(anyhow!("{}#{} [{}] {} {}{}: {} ({})", ctx.template, ctx.actor, idx.join(","), op.kind.name(), op.path, extra, e, name))
                }
            }
        }
    }

    fn control(&mut self, c: Control<'static>, ctx: &OpCtx) -> Result<()> {
        match c {
            Control::Compute { ns } => {
                let ns = ns.max(0) as u64;
                let scaled = (ns as f64 * self.sh.opts.time_scale) as u64;
                if scaled > 0 {
                    std::thread::sleep(Duration::from_nanos(scaled));
                }
                self.st.compute_ns += ns;
                if let Some(last) = self.takes.last_mut() {
                    last.compute_ns += ns;
                }
            }
            Control::Barrier { scope } => {
                if !self.main {
                    bail!("barrier `{scope}` inside a sub-actor is not supported");
                }
                let waited = self.sh.coord.barrier(scope, &self.sh.aborted)?;
                self.st.barriers += 1;
                self.st.barrier_wait_ns += waited.as_nanos() as u64;
            }
            Control::Channel { name, capacity, ordered } => {
                let mut ch = self.inst.channels.lock().unwrap();
                if ch.contains_key(name) {
                    bail!("channel `{name}` declared twice");
                }
                ch.insert(name.to_string(), Arc::new(Channel::new(capacity, ordered, None)));
            }
            Control::Put { channel, seq } => {
                let ch = self.inst.channels.lock().unwrap().get(channel).cloned().ok_or_else(|| anyhow!("put on undeclared channel `{channel}`"))?;
                ch.put(seq, &self.sh.aborted)?;
                self.st.puts += 1;
            }
            Control::Take { channel } => {
                let ch = self.inst.channels.lock().unwrap().get(channel).cloned().ok_or_else(|| anyhow!("take on undeclared channel `{channel}` (a loader declares one under its name)"))?;
                let t = Instant::now();
                ch.take(&self.sh.aborted).with_context(|| {
                    let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
                    format!("{}#{} [{}] take `{channel}`", ctx.template, ctx.actor, idx.join(","))
                })?;
                let stall = t.elapsed().as_nanos() as u64;
                self.st.takes += 1;
                self.takes.push(TakeRec { stall_ns: stall, compute_ns: 0 });
            }
        }
        Ok(())
    }

    fn fork(&mut self, kind: &ForkKind<'static>, snapshot: &dyn Fn() -> Snapshot<'static, 'static>) -> Result<bool> {
        let snap = snapshot();
        match *kind {
            ForkKind::Parallel { index, width } => {
                let mut handles = Vec::with_capacity(width as usize);
                for k in 0..width {
                    let label = format!("{}#{} {index}={k}", self.template, self.actor);
                    handles.push(self.spawn_sub(snap.clone(), label, move |vm| vm.run_sub(k)));
                }
                let mut first_err = None;
                for h in handles {
                    let (r, st) = h.join().map_err(|_| anyhow!("a `{index}` sub-actor panicked"))?;
                    self.st.merge(&st);
                    if let Err(e) = r {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                }
                match first_err {
                    Some(e) => Err(e),
                    None => Ok(true),
                }
            }
            ForkKind::Loader { name, index, workers, prefetch, batches, ordered } => {
                let chan = Arc::new(Channel::new(workers * prefetch, ordered, Some(batches)));
                {
                    let mut ch = self.inst.channels.lock().unwrap();
                    if ch.contains_key(name) {
                        bail!("loader `{name}`: a channel of that name exists");
                    }
                    ch.insert(name.to_string(), chan.clone());
                }
                let mut handles = Vec::with_capacity(workers as usize);
                for w in 0..workers {
                    let chan = chan.clone();
                    let label = format!("{}#{} {index} worker {w}", self.template, self.actor);
                    let h = self.spawn_sub(snap.clone(), label, move |vm| {
                        let aborted = &vm.sink.sh.clone().aborted;
                        let mut b = w;
                        while b < batches {
                            if let Err(e) = chan.begin(b, aborted) {
                                // closed by the consumer: not this worker's error
                                if !chan.m.lock().unwrap().closed {
                                    chan.fail(format!("{e:#}"));
                                }
                                return Ok(());
                            }
                            if let Err(e) = vm.run_sub(b) {
                                chan.fail(format!("{e:#}"));
                                return Err(e);
                            }
                            chan.end(b);
                            b += workers;
                        }
                        Ok(())
                    });
                    handles.push((name.to_string(), h));
                }
                self.inst.loaders.lock().unwrap().extend(handles);
                Ok(true)
            }
        }
    }

    fn finish(&mut self) -> Result<()> {
        let mut result = Ok(());
        if !self.created.is_empty() {
            self.sh.created.lock().unwrap().append(&mut self.created);
        }
        if !self.removed.is_empty() {
            self.sh.removed.lock().unwrap().append(&mut self.removed);
        }
        if self.main {
            // loaders: close their channels so a worker blocked on a slot exits, then join
            let channels: Vec<(String, Arc<Channel>)> = self.inst.channels.lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            for (_, ch) in &channels {
                ch.close();
            }
            let loaders: Vec<_> = std::mem::take(&mut *self.inst.loaders.lock().unwrap());
            for (name, h) in loaders {
                match h.join() {
                    Ok((r, st)) => {
                        self.st.merge(&st);
                        if let Err(e) = r {
                            if result.is_ok() {
                                result = Err(e.context(format!("loader `{name}` worker")));
                            }
                        }
                    }
                    Err(_) => {
                        if result.is_ok() {
                            result = Err(anyhow!("loader `{name}`: a worker panicked"));
                        }
                    }
                }
            }
            for (name, ch) in &channels {
                if let Some(total) = ch.total {
                    let taken = ch.taken();
                    if taken < total && result.is_ok() {
                        result = Err(anyhow!("loader `{name}`: {taken} of {total} batches were taken; the consumer must take exactly `batches` items (`NAPKIN_MATH.md` §4.1, finite producers)"));
                    }
                }
            }
            if let Some(scopes) = self.sh.scopes.get(self.template) {
                for s in scopes {
                    self.sh.coord.leave(s);
                }
            }
            self.sh.actors.lock().unwrap().push(ActorRecord {
                template: self.template.to_string(),
                actor: self.actor,
                takes: std::mem::take(&mut self.takes),
                elapsed: self.started.elapsed(),
            });
            self.sh.stats.lock().unwrap().merge(&std::mem::take(&mut self.st));
        }
        if result.is_err() {
            self.sh.aborted.store(true, Ordering::Relaxed);
        }
        result
    }
}

// ---------------------------------------------------------------- startup checks

#[derive(Debug, Clone)]
pub struct DatasetCheck {
    pub name: String,
    pub root: String,
    pub id: String,
    pub payload: payload::PayloadSpec,
    pub files: Option<u64>,
}

/// Compare every dataset against its manifest; returns the dataset ids.
pub fn check_datasets(loaded: &crate::Loaded, cfg: &Config, root: &Path) -> Result<Vec<DatasetCheck>> {
    let mut out = Vec::new();
    for name in loaded.ast.datasets.keys() {
        let rel = payload::dataset_root(&loaded.ast, name)?;
        let dir = root.join(&rel);
        let m = Manifest::read(&dir).with_context(|| format!("dataset `{name}` at {}: run `aeiou datagen` first", dir.display()))?;
        let want = payload::resolved_dataset(&loaded.doc, name, cfg)?;
        if m.dataset != want {
            let mut lines = Vec::new();
            payload::diff(&want, &m.dataset, "", &mut lines);
            bail!(
                "dataset `{name}` at {}: the manifest does not match the resolved definition (abstract vs manifest):\n  {}",
                dir.display(),
                lines.join("\n  ")
            );
        }
        m.payload.check().with_context(|| format!("dataset `{name}` at {}", dir.display()))?;
        let files = m.provenance.get("files_written").and_then(|v| v.as_u64());
        out.push(DatasetCheck { name: name.clone(), root: rel, id: m.id(), payload: m.payload.clone(), files });
    }
    Ok(out)
}

/// What a run learned about one input namespace root at startup.
#[derive(Debug, Clone)]
pub struct NamespaceCheck {
    pub root: String,
    pub names: Vec<String>,
    pub writer_abstract: String,
    pub writer_sha256: String,
    pub writer_hosts: Vec<String>,
    /// Seconds from the writer's finish to this check.
    pub gap: f64,
    pub objects: Option<usize>,
    /// This host wrote part of the GPU range it is about to run.
    pub same_host: bool,
}

/// Every namespace declared `input` must have a manifest at its root whose resolved
/// definitions match this abstract's; reports the write-to-read gap and the host overlap.
/// Returns the checks and the writer map of every input object.
pub fn check_input_namespaces(loaded: &crate::Loaded, cfg: &Config, root: &Path, opts: &RunOpts) -> Result<(Vec<NamespaceCheck>, HashMap<String, Option<String>>)> {
    let ast = &loaded.ast;
    let mut by_root: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, n) in &ast.namespaces {
        if n.input.unwrap_or(false) {
            by_root.entry(payload::namespace_root(ast, name)?).or_default().push(name.clone());
        }
    }
    let host = hostname();
    let (lo, hi) = gpu_range(cfg.gpus, opts.ranks, opts.rank, opts.rank_rotate);
    let mut checks = Vec::new();
    let mut objects: HashMap<String, Option<String>> = HashMap::new();
    for (rel, names) in by_root {
        let dir = root.join(&rel);
        let m = NamespaceManifest::read(&dir).with_context(|| format!("input namespace(s) {} at {}: no run has written this root", names.join(", "), dir.display()))?;
        for name in &names {
            let want = payload::resolved_namespace(&loaded.doc, name, cfg)?;
            let Some(have) = m.namespaces.get(name) else {
                bail!("input namespace `{name}` at {}: the manifest (written by `{}`) has no namespace of that name; it has {}", dir.display(), m.abstract_name, m.namespaces.keys().cloned().collect::<Vec<_>>().join(", "));
            };
            if *have != want {
                let mut lines = Vec::new();
                payload::diff(&want, have, "", &mut lines);
                bail!("input namespace `{name}` at {}: this abstract's definition differs from the writer's (`{}`): \n  {}", dir.display(), m.abstract_name, lines.join("\n  "));
            }
        }
        let gap = unix_now() - m.finished;
        if let Some(max) = opts.max_gap {
            if gap > max {
                bail!("input namespace root {}: written {gap:.1} s ago, more than --max-gap {max}", dir.display());
            }
        }
        let same_host = m.ranks.iter().any(|r| r.host == host && r.gpus[0] < hi && lo < r.gpus[1]);
        if same_host && opts.require_cold {
            bail!("input namespace root {}: this host ({host}) wrote part of GPU range [{lo}, {hi}) it is about to run; use --rank-rotate or other hosts (--require-cold)", dir.display());
        }
        let mut hosts: Vec<String> = m.ranks.iter().map(|r| r.host.clone()).collect();
        hosts.sort();
        hosts.dedup();
        if let Some(list) = &m.objects {
            for (path, gpu) in list {
                objects.insert(path.clone(), m.host_of_gpu(*gpu).map(|h| h.to_string()));
            }
        }
        checks.push(NamespaceCheck {
            root: rel,
            names,
            writer_abstract: m.abstract_name.clone(),
            writer_sha256: m.ast_sha256.clone(),
            writer_hosts: hosts,
            gap,
            objects: m.objects.as_ref().map(|o| o.len()),
            same_host,
        });
    }
    Ok((checks, objects))
}

/// After a run: `.aeiou-namespace.json` at every output namespace root (one not declared
/// `input`), written last and atomically, with the objects created there.
pub fn write_namespace_manifests(loaded: &crate::Loaded, cfg: &Config, root: &Path, opts: &RunOpts, report: &Report, started: f64, finished: f64) -> Result<Vec<PathBuf>> {
    let ast = &loaded.ast;
    let mut by_root: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, n) in &ast.namespaces {
        if !n.input.unwrap_or(false) {
            by_root.entry(payload::namespace_root(ast, name)?).or_default().push(name.clone());
        }
    }
    let (lo, hi) = gpu_range(cfg.gpus, opts.ranks, opts.rank, opts.rank_rotate);
    let mut written = Vec::new();
    for (rel, names) in by_root {
        let prefix = if rel.is_empty() { String::new() } else { format!("{rel}/") };
        let mut seen = HashSet::new();
        let mut objects: Vec<(String, i64)> = Vec::new();
        for (path, gpu) in &report.created {
            if path.starts_with(&prefix) && seen.insert(path.clone()) {
                objects.push((path.clone(), *gpu));
            }
        }
        objects.sort();
        let count = objects.len() as u64;
        let mut namespaces = BTreeMap::new();
        for name in &names {
            namespaces.insert(name.clone(), payload::resolved_namespace(&loaded.doc, name, cfg)?);
        }
        let m = NamespaceManifest {
            manifest_version: payload::NAMESPACE_MANIFEST_VERSION,
            namespaces,
            abstract_name: ast.name.clone(),
            ast_sha256: loaded.sha256.clone(),
            seed: cfg.seed,
            gpus: cfg.gpus,
            params: payload::params_json(&loaded.doc, cfg, &Params::new(ast, cfg)?)?,
            ranks: vec![RankRecord { rank: opts.rank, host: hostname(), gpus: [lo, hi] }],
            started,
            finished,
            objects_created: count,
            bytes_written: report.stats.bytes_written,
            objects: if objects.len() <= payload::NAMESPACE_OBJECT_LIMIT { Some(objects) } else { None },
        };
        written.push(m.write(&root.join(&rel))?);
    }
    Ok(written)
}

/// Output namespace roots must be empty (`--clean-namespaces` empties them); input roots
/// are left as they are; dataset roots inside a namespace root are left alone. Creates the
/// output roots.
pub fn prepare_namespaces(ast: &Ast, root: &Path, clean: bool) -> Result<Vec<String>> {
    let mut dataset_roots: Vec<PathBuf> = Vec::new();
    for name in ast.datasets.keys() {
        dataset_roots.push(root.join(payload::dataset_root(ast, name)?));
    }
    let mut cleaned = Vec::new();
    let mut seen = BTreeSet::new();
    for (name, n) in &ast.namespaces {
        let rel = payload::namespace_root(ast, name)?;
        if n.input.unwrap_or(false) {
            seen.insert(rel);
            continue;
        }
        if !seen.insert(rel.clone()) {
            continue;
        }
        let dir = root.join(&rel);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating namespace root {}", dir.display()))?;
        let mut stale = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if dataset_roots.iter().any(|d| d.starts_with(&p) || p.starts_with(d)) {
                continue;
            }
            stale.push(p);
        }
        if !stale.is_empty() {
            if !clean {
                bail!(
                    "namespace `{name}` root {} is not empty ({} entries, e.g. {}); pass --clean-namespaces to empty it",
                    dir.display(),
                    stale.len(),
                    stale[0].display()
                );
            }
            for p in stale {
                if p.is_dir() {
                    std::fs::remove_dir_all(&p)?;
                } else {
                    std::fs::remove_file(&p)?;
                }
            }
            cleaned.push(rel.clone());
        }
    }
    Ok(cleaned)
}

/// Barrier scopes per template, from the bodies (outside `parallel`/`loader`).
fn barrier_scopes(ast: &Ast) -> Result<HashMap<String, Vec<String>>> {
    fn walk(nodes: &[Node], nested: bool, out: &mut BTreeSet<String>) -> Result<()> {
        for n in nodes {
            match n {
                Node::Barrier { scope } => {
                    if nested {
                        bail!("barrier `{scope}` inside a `parallel` or `loader` is not supported by `aeiou run`");
                    }
                    out.insert(scope.clone());
                }
                Node::Loop { body, .. } | Node::Phase { body, .. } => walk(body, nested, out)?,
                Node::Parallel { body, .. } | Node::Loader { body, .. } => walk(body, true, out)?,
                Node::Cond { then, otherwise, .. } => {
                    walk(then, nested, out)?;
                    if let Some(b) = otherwise {
                        walk(b, nested, out)?;
                    }
                }
                Node::Choose { arms, .. } => {
                    for a in arms {
                        walk(&a.body, nested, out)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    let mut out = HashMap::new();
    for (name, a) in &ast.actors {
        let mut s = BTreeSet::new();
        walk(&a.body, false, &mut s).with_context(|| format!("actor `{name}`"))?;
        out.insert(name.clone(), s.into_iter().collect());
    }
    Ok(out)
}

// ---------------------------------------------------------------- the run

pub struct Report {
    pub elapsed: Duration,
    pub stats: Stats,
    /// (template, instance count, [lo, hi) run on this host)
    pub templates: Vec<(String, i64, (i64, i64))>,
    pub actors: Vec<ActorRecord>,
    pub departure_releases: Vec<(String, u64)>,
    pub threads_peak: u64,
    pub created: Vec<(String, i64)>,
    pub host: String,
}

/// Execute the abstract. `model` must outlive the threads, hence `'static` (the caller leaks
/// the loaded abstract and model for the life of the process; a run is the process).
pub fn run(model: &'static Model<'static>, opts: RunOpts, input_objects: HashMap<String, Option<String>>) -> Result<Report> {
    let counts: Vec<(&'static str, i64)> = actor_counts(model)?;
    let ranges: Vec<(i64, i64)> = counts.iter().map(|(_, c)| gpu_range(*c, opts.ranks, opts.rank, opts.rank_rotate)).collect();
    let scopes = barrier_scopes(model.ast)?;
    let mut participants: BTreeMap<String, usize> = BTreeMap::new();
    for ((name, _), (lo, hi)) in counts.iter().zip(&ranges) {
        for s in scopes.get(*name).map(|v| v.as_slice()).unwrap_or(&[]) {
            *participants.entry(s.clone()).or_insert(0) += (hi - lo) as usize;
        }
    }
    let participants: Vec<(String, usize)> = participants.into_iter().collect();
    let sh = Arc::new(Shared {
        opts,
        coord: Local::new(&participants),
        aborted: AtomicBool::new(false),
        stats: Mutex::new(Stats::default()),
        actors: Mutex::new(Vec::new()),
        scopes,
        host: hostname(),
        input_objects,
        created: Mutex::new(Vec::new()),
        removed: Mutex::new(Vec::new()),
    });

    let t0 = Instant::now();
    let mut handles = Vec::new();
    for (&(template, count), &(lo, hi)) in counts.iter().zip(&ranges) {
        for actor in lo..hi {
            let sh = sh.clone();
            let inst = Arc::new(Instance { channels: Mutex::new(HashMap::new()), loaders: Mutex::new(Vec::new()) });
            let h = std::thread::Builder::new()
                .name(format!("{template}#{actor}"))
                .spawn(move || {
                    let sink = Runner::new(sh.clone(), inst, template, actor, true, None);
                    let body = &model.ast.actors[template].body;
                    let mut vm = Vm::new(model, sink, template, actor, count);
                    let r = vm.run(body);
                    if r.is_err() {
                        sh.aborted.store(true, Ordering::Relaxed);
                    }
                    r
                })
                .expect("spawn actor thread");
            handles.push((template, actor, h));
        }
    }
    let mut first_err: Option<anyhow::Error> = None;
    for (template, actor, h) in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                if first_err.is_none() {
                    first_err = Some(e.context(format!("actor `{template}` instance {actor}")));
                }
            }
            Err(_) => {
                if first_err.is_none() {
                    first_err = Some(anyhow!("actor `{template}` instance {actor} panicked"));
                }
            }
        }
    }
    let elapsed = t0.elapsed();
    if let Some(e) = first_err {
        return Err(e);
    }
    let stats = std::mem::take(&mut *sh.stats.lock().unwrap());
    let mut actors = std::mem::take(&mut *sh.actors.lock().unwrap());
    actors.sort_by(|a, b| a.template.cmp(&b.template).then(a.actor.cmp(&b.actor)));
    let removed: HashSet<String> = std::mem::take(&mut *sh.removed.lock().unwrap()).into_iter().collect();
    let mut created: Vec<(String, i64)> = std::mem::take(&mut *sh.created.lock().unwrap()).into_iter().filter(|(p, _)| !removed.contains(p)).collect();
    created.sort();
    Ok(Report {
        elapsed,
        threads_peak: stats.threads,
        stats,
        templates: counts.iter().zip(&ranges).map(|((n, c), r)| (n.to_string(), *c, *r)).collect(),
        actors,
        departure_releases: sh.coord.departure_releases(),
        created,
        host: sh.host.clone(),
    })
}

// ---------------------------------------------------------------- report

fn us(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.1}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.0}µs", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

pub fn write_report(out: &mut impl Write, r: &Report) -> std::io::Result<()> {
    let s = &r.stats;
    let secs = r.elapsed.as_secs_f64().max(1e-9);
    writeln!(out, "elapsed {:.3} s  threads {}", secs, s.threads)?;
    writeln!(
        out,
        "ops {}  read {}  written {}  ({:.0} ops/s, {}/s read, {}/s written)",
        s.ops,
        human_bytes(s.bytes_read),
        human_bytes(s.bytes_written),
        s.ops as f64 / secs,
        human_bytes((s.bytes_read as f64 / secs) as u64),
        human_bytes((s.bytes_written as f64 / secs) as u64)
    )?;
    let counts: Vec<String> = OpKind::ALL.iter().filter_map(|k| s.counts.get(k).map(|n| format!("{}={}", k.name(), n))).collect();
    writeln!(out, "by kind: {}", counts.join(" "))?;
    writeln!(out, "latency (mean/p50/p99/max):")?;
    for k in OpKind::ALL {
        if let Some(h) = s.lat.get(&k) {
            writeln!(out, "  {:<10} {:>9} {:>9} {:>9} {:>9}   n={}", k.name(), us(h.mean()), us(h.quantile(0.5)), us(h.quantile(0.99)), us(h.max), h.count)?;
        }
    }
    let mut phases: Vec<_> = s.phases.iter().collect();
    phases.sort_by(|a, b| b.1.ops.cmp(&a.1.ops).then(a.0.cmp(b.0)));
    for (name, p) in phases {
        writeln!(out, "phase {:<16} ops={:<10} read={:<12} written={:<12} io-time={}", name, p.ops, human_bytes(p.bytes_read), human_bytes(p.bytes_written), human_ns(p.io_ns as i128))?;
    }
    writeln!(
        out,
        "compute {}  io-time {}  barriers {} (waited {})  takes {}  puts {}  expected-errors {}",
        human_ns(s.compute_ns as i128),
        human_ns(s.io_ns as i128),
        s.barriers,
        human_ns(s.barrier_wait_ns as i128),
        s.takes,
        s.puts,
        s.expected_errors
    )?;
    if s.input_opens > 0 {
        writeln!(out, "input objects opened {}  of which written on this host ({}) {}", s.input_opens, r.host, s.warm_opens)?;
        if s.warm_opens > 0 {
            writeln!(out, "WARNING: {} read(s) of input objects hit the host that wrote them (page cache, not storage); run the reader on other hosts or with --rank-rotate", s.warm_opens)?;
        }
    }
    for (scope, n) in &r.departure_releases {
        writeln!(out, "WARNING: barrier `{scope}` was released {n} time(s) by instances finishing: not every instance hit it the same number of times")?;
    }
    // takes: per-step stall and busy fraction, bucketed by take ordinal
    let with_takes: Vec<&ActorRecord> = r.actors.iter().filter(|a| !a.takes.is_empty()).collect();
    if !with_takes.is_empty() {
        let n = with_takes.iter().map(|a| a.takes.len()).max().unwrap_or(0);
        let buckets = n.min(10).max(1);
        let per = (n + buckets - 1) / buckets;
        let total_stall: u64 = with_takes.iter().flat_map(|a| a.takes.iter().map(|t| t.stall_ns)).sum();
        let total_compute: u64 = with_takes.iter().flat_map(|a| a.takes.iter().map(|t| t.compute_ns)).sum();
        let total_takes: u64 = with_takes.iter().map(|a| a.takes.len() as u64).sum();
        writeln!(
            out,
            "takes: {} per instance over {} instance(s); stall/take mean {}; busy {:.3} (compute / (compute + stall))",
            n,
            with_takes.len(),
            us(if total_takes > 0 { total_stall / total_takes } else { 0 }),
            if total_compute + total_stall > 0 { total_compute as f64 / (total_compute + total_stall) as f64 } else { 0.0 }
        )?;
        writeln!(out, "  {:<12} {:>10} {:>10} {:>10} {:>8}", "steps", "stall/take", "p99", "max", "busy")?;
        for b in 0..buckets {
            let lo = b * per;
            let hi = ((b + 1) * per).min(n);
            if lo >= hi {
                break;
            }
            let mut h = LatHist::default();
            let mut compute = 0u64;
            for a in &with_takes {
                for t in a.takes.iter().take(hi).skip(lo) {
                    h.add(t.stall_ns);
                    compute += t.compute_ns;
                }
            }
            let busy = if compute + h.sum > 0 { compute as f64 / (compute + h.sum) as f64 } else { 0.0 };
            writeln!(out, "  {:<12} {:>10} {:>10} {:>10} {:>8.3}", format!("{lo}..{hi}"), us(h.mean()), us(h.quantile(0.99)), us(h.max), busy)?;
        }
    }
    writeln!(out, "fingerprint {:016x}", s.fingerprint)?;
    Ok(())
}
