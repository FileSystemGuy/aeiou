//! `aeiou run`: execute an abstract against a directory with a blocking backend. One OS thread
//! per actor instance, one per loader worker, one per `parallel` sub-actor, the last kept in a
//! pool by the actor that forks them (`Pool`) so a `parallel` inside a loop does not spawn its
//! threads afresh on every iteration (the `sync`
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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ast::{Ast, Node};
use crate::endpoint::Place;
use crate::backend::{errno_name, Backend, BackendKind, MmapConsume, MmapMode, MmapStats, OpenFile, ALIGN};
use crate::coord::{Coordinator, Local};
use crate::counters::{HostCounters, Sampler};
use crate::dryrun::{human_bytes, human_ns};
use crate::eval::{Config, Model, Params};
use crate::payload::{self, Filler, Manifest, NamespaceManifest, RankRecord};
use crate::vm::{actor_counts, drive, op_hash, Control, ForkKind, Op, OpCtx, OpKind, Sink, Snapshot, Vm};

// ---------------------------------------------------------------- options

/// The `io_uring` knobs of the A/B matrix (`NAPKIN_MATH.md` §8.5), applied to every event
/// loop's ring. None of them changes the op stream or the fingerprint.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UringOpts {
    /// Cap on the bounded io-wq workers (`IORING_REGISTER_IOWQ_MAX_WORKERS`) of each loop's
    /// io-wq (of the one shared io-wq under `sqpoll_shared`); 0 keeps the kernel's default,
    /// which the report shows either way.
    pub iowq_max_workers: u32,
    /// `IORING_SETUP_SQPOLL`: a kernel submission thread that sleeps after this many idle ms.
    pub sqpoll_idle_ms: Option<u32>,
    /// One submission thread (and one io-wq) shared by every loop, `IORING_SETUP_ATTACH_WQ`
    /// to the first loop's ring, instead of one per loop.
    pub sqpoll_shared: bool,
    /// `IORING_SETUP_SINGLE_ISSUER` with `IORING_SETUP_DEFER_TASKRUN`: completions are
    /// processed only in the loop's own `io_uring_enter`.
    pub defer_taskrun: bool,
    /// `IORING_SETUP_COOP_TASKRUN`: no interrupt of the loop thread to run completions.
    pub coop_taskrun: bool,
}

impl UringOpts {
    pub fn any(&self) -> bool {
        *self != UringOpts::default()
    }

    /// The combinations the kernel refuses, said before a ring is built.
    pub fn check(&self) -> Result<()> {
        if self.sqpoll_shared && self.sqpoll_idle_ms.is_none() {
            bail!("sqpoll_shared needs sqpoll");
        }
        if self.defer_taskrun && self.sqpoll_idle_ms.is_some() {
            bail!("defer_taskrun and sqpoll exclude each other (IORING_SETUP_DEFER_TASKRUN is not allowed with IORING_SETUP_SQPOLL)");
        }
        Ok(())
    }

    /// The knobs as words, for the run header and the report.
    pub fn describe(&self) -> String {
        let mut v = Vec::new();
        if self.iowq_max_workers > 0 {
            v.push(format!("io-wq max workers {}", self.iowq_max_workers));
        }
        if let Some(ms) = self.sqpoll_idle_ms {
            v.push(format!("sqpoll idle {ms} ms{}", if self.sqpoll_shared { " shared" } else { " per loop" }));
        }
        if self.defer_taskrun {
            v.push("single-issuer defer-taskrun".into());
        }
        if self.coop_taskrun {
            v.push("coop-taskrun".into());
        }
        if v.is_empty() {
            "defaults".into()
        } else {
            v.join(", ")
        }
    }
}

/// What the `io_uring` backends report about their rings (`Report::uring`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UringReport {
    pub loops: u64,
    pub opts: UringOpts,
    /// The io-wq caps `[bounded, unbounded]` the kernel had before the run set anything, as
    /// `IORING_REGISTER_IOWQ_MAX_WORKERS` returned them to the first loop; `None` when the
    /// kernel lacks the call (before 5.15) and nothing was asked.
    pub iowq_defaults: Option<[u32; 2]>,
    /// Most ops one loop had on its ring at once (submitted, not yet completed), counted by
    /// the loop; the largest over the loops, and over the hosts once merged. An actor has
    /// one op in flight at most, so this is how many of a loop's actors were waiting on I/O
    /// together.
    #[serde(default)]
    pub in_flight_peak: u64,
}

impl UringReport {
    pub fn describe(&self) -> String {
        let caps = match self.iowq_defaults {
            Some([b, u]) => format!("kernel io-wq caps bounded {b} unbounded {u}{}", if self.opts.sqpoll_shared { " (one io-wq for all loops)" } else { " per loop" }),
            None => "io-wq caps unknown (no IORING_REGISTER_IOWQ_MAX_WORKERS)".into(),
        };
        format!("loops {}  in-flight peak {}  {}  knobs: {}", self.loops, self.in_flight_peak, caps, self.opts.describe())
    }
}

/// What the `libaio` backends report about their AIO contexts (`Report::aio`), summed over
/// the event loops (and over the hosts once merged); the depth is per loop and the peak is
/// the largest any loop saw.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AioReport {
    pub loops: u64,
    /// `io_setup`'s `nr_events` of each loop's context (`--aio-depth`).
    pub depth: u64,
    /// Most requests one context had in flight (queued or in the kernel).
    pub in_flight_peak: u64,
    /// `io_submit` calls, the requests they carried, and the time spent inside them: a
    /// buffered read is done by the time `io_submit` returns, so for a buffered run this is
    /// the I/O time and the loop was blocked for it.
    pub submits: u64,
    pub submitted: u64,
    pub submit_ns: u64,
    pub getevents: u64,
    /// `io_submit` returned `EAGAIN` (the context was full): completions were reaped first.
    pub full: u64,
}

impl AioReport {
    pub fn merge(&mut self, o: &AioReport) {
        self.loops += o.loops;
        self.depth = self.depth.max(o.depth);
        self.in_flight_peak = self.in_flight_peak.max(o.in_flight_peak);
        self.submits += o.submits;
        self.submitted += o.submitted;
        self.submit_ns += o.submit_ns;
        self.getevents += o.getevents;
        self.full += o.full;
    }

    pub fn describe(&self) -> String {
        format!(
            "loops {}  context depth {}  in-flight peak {}  io_submit {} calls, {} requests, {} inside  io_getevents {} calls  context full {}",
            self.loops,
            self.depth,
            self.in_flight_peak,
            self.submits,
            self.submitted,
            human_ns(self.submit_ns as i128),
            self.getevents,
            self.full
        )
    }
}

/// What the `mmap` backend reports (`Report::mmap`), summed over the actors and the hosts.
/// The faults themselves are in the host counters (every backend has them).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MmapReport {
    pub mode: MmapMode,
    #[serde(default)]
    pub consume: MmapConsume,
    pub maps: u64,
    pub mapped_bytes: u64,
    /// `madvise` calls (`populate` and `willneed`).
    pub advised: u64,
    /// Pages a byte was read from (`touch`; none under `populate`, which touches nothing).
    #[serde(default)]
    pub touched_pages: u64,
    /// Bytes copied into the actors' buffers (`copy`).
    pub copied_bytes: u64,
}

impl MmapReport {
    pub fn merge(&mut self, o: &MmapReport) {
        self.maps += o.maps;
        self.mapped_bytes += o.mapped_bytes;
        self.advised += o.advised;
        self.touched_pages += o.touched_pages;
        self.copied_bytes += o.copied_bytes;
    }

    pub fn describe(&self) -> String {
        format!(
            "mode {}  consume {}  mappings {} ({})  madvise calls {}  pages touched {}  copied out {}",
            self.mode.name(),
            self.consume.name(),
            self.maps,
            human_bytes(self.mapped_bytes),
            self.advised,
            self.touched_pages,
            human_bytes(self.copied_bytes)
        )
    }
}

#[derive(Debug, Clone)]
pub struct RunOpts {
    /// The directory the abstract's paths are relative to.
    pub root: PathBuf,
    /// The datasets and namespaces placed elsewhere (`endpoint.rs`).
    pub endpoints: crate::endpoint::Endpoints,
    /// The POSIX names' API and cache mode (`--posix`, `--cache`).
    pub backend: BackendKind,
    /// How the S3 names' client waits (`--s3`): `Async` puts the run on event loops.
    pub s3: crate::backend::S3Api,
    /// Per-thread read and write buffer ring, bytes.
    pub buffer_bytes: usize,
    /// Event-loop threads for the `io_uring` and `libaio` backends (0: one per core, at most
    /// one per actor instance). The other backends run one thread per actor and ignore it.
    pub threads: usize,
    /// Compression ratio of written content.
    pub write_compress: u64,
    /// Multiplier on `compute` sleeps.
    pub time_scale: f64,
    /// The ring and io-wq knobs of the `io_uring` backends; the `sync` backends refuse them.
    pub uring: UringOpts,
    /// What the `mmap` backend does before a read's range is consumed, and how it is
    /// consumed (a byte of every page, or a copy out).
    pub mmap: MmapMode,
    pub mmap_consume: MmapConsume,
    /// `nr_events` of each `libaio` loop's context (0: `aio::DEPTH`).
    pub aio_depth: u32,
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
    /// Refuse when this host would read an input object it wrote itself, or when the
    /// residency check finds dataset pages in its page cache (`cold`).
    pub require_cold: bool,
    /// `sync` and drop the page cache, dentries, and inodes before the start gate (`cold`).
    pub drop_caches: bool,
}

impl RunOpts {
    /// The actors run on event loops (`uring.rs`) rather than a thread each: the POSIX API is
    /// event-driven, or the S3 one is (V19 and the run's checks keep the two agreeing when
    /// both protocols have names).
    pub fn event_loop(&self) -> bool {
        self.backend.event_loop() || self.s3.event_loop()
    }

    /// Where the abstract's path `rel` lives: under its endpoint, or under `--root`.
    pub fn path(&self, rel: &str) -> PathBuf {
        self.endpoints.path(&self.root, rel)
    }
}

impl Place for RunOpts {
    fn at(&self, rel: &str) -> PathBuf {
        self.path(rel)
    }
    fn object(&self, rel: &str) -> Option<(Arc<crate::object::Store>, String)> {
        self.endpoints.object(rel)
    }
}

/// The event-loop threads this host will run under an event-loop backend: `--threads`, or
/// one per core, at most one per instance (as `uring::spread` deals them).
pub fn loop_count(model: &Model<'_>, opts: &RunOpts) -> Result<u64> {
    let mut instances = 0i64;
    for (_, count) in actor_counts(model)? {
        let (lo, hi) = gpu_range(count, opts.ranks, opts.rank, opts.rank_rotate);
        instances += (hi - lo).max(0);
    }
    if instances == 0 {
        return Ok(0);
    }
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Ok(if opts.threads == 0 { cores } else { opts.threads }.clamp(1, instances as usize) as u64)
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn lower(i: usize) -> u64 {
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseStats {
    pub ops: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub io_ns: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TakeRec {
    pub stall_ns: u64,
    pub compute_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

pub(crate) struct Shared {
    pub(crate) opts: RunOpts,
    pub(crate) coord: Arc<dyn Coordinator>,
    pub(crate) aborted: Arc<AtomicBool>,
    pub(crate) stats: Mutex<Stats>,
    pub(crate) actors: Mutex<Vec<ActorRecord>>,
    /// Barrier scopes each template participates in.
    pub(crate) scopes: HashMap<String, Vec<String>>,
    pub(crate) host: String,
    /// Objects of input namespaces: path → the host that wrote it (when recorded).
    pub(crate) input_objects: HashMap<String, Option<String>>,
    /// Objects this run created (path, creating GPU id), for the namespace manifests, and
    /// paths it removed (unlink, the source of a rename).
    pub(crate) created: Mutex<Vec<(String, i64)>>,
    pub(crate) removed: Mutex<Vec<String>>,
    /// What the `mmap` backends count (zero under the others).
    pub(crate) mmap_stats: Arc<MmapStats>,
}

impl Shared {
    /// This run's blocking backend: an actor thread's own, or an event loop's inline one.
    pub(crate) fn backend(&self) -> Box<dyn Backend> {
        self.opts.backend.make(self.opts.mmap, self.opts.mmap_consume, &self.mmap_stats)
    }
}

/// Per actor instance: its channels and its loader threads.
struct Instance {
    channels: Mutex<HashMap<String, Arc<Channel>>>,
    loaders: Mutex<Vec<(String, JoinHandle<(Result<()>, Stats)>)>>,
}

type Fds = HashMap<Arc<str>, Arc<OpenFile>>;

/// Files opened by this actor, plus what the parent had open at the fork.
pub(crate) struct FdTable {
    pub(crate) own: Fds,
    inherited: Option<Arc<FdTable>>,
    closed: HashSet<Arc<str>>,
}

impl FdTable {
    pub(crate) fn get(&self, path: &str) -> Option<Arc<OpenFile>> {
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
pub(crate) struct Ring {
    pub(crate) ptr: *mut u8,
    cap: usize,
    pos: usize,
    target: usize,
}

unsafe impl Send for Ring {}

impl Ring {
    pub(crate) fn new(target: usize) -> Self {
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

    pub(crate) fn slice(&mut self, len: usize) -> &mut [u8] {
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

// ---------------------------------------------------------------- per-actor state

/// What every actor instance or sub-actor carries whichever driver runs it: its files, its
/// payload filler, its statistics, and the bookkeeping the namespace manifests need. The
/// thread-per-actor `Runner` (the blocking backends) and the event loops (`uring.rs`,
/// `aio.rs`) both build on it, so an op is checked and recorded the same way under every
/// backend.
pub(crate) struct ActorState {
    pub(crate) fds: FdTable,
    pub(crate) filler: Filler,
    pub(crate) st: Stats,
    pub(crate) takes: Vec<TakeRec>,
    pub(crate) created: Vec<(String, i64)>,
    pub(crate) removed: Vec<String>,
    /// The actor instance's main line (not a sub-actor).
    pub(crate) main: bool,
    pub(crate) template: &'static str,
    pub(crate) actor: i64,
    pub(crate) started: Instant,
}

impl ActorState {
    /// `threads` is what this actor adds to the report's OS thread count (1 when it is one).
    pub(crate) fn new(sh: &Shared, template: &'static str, actor: i64, main: bool, inherited: Option<Arc<FdTable>>, threads: u64) -> Self {
        ActorState {
            fds: FdTable { own: HashMap::new(), inherited, closed: HashSet::new() },
            filler: Filler::new(sh.opts.write_compress),
            st: Stats { threads, ..Default::default() },
            takes: Vec::new(),
            created: Vec::new(),
            removed: Vec::new(),
            main,
            template,
            actor,
            started: Instant::now(),
        }
    }

    /// A sub-actor's state: sees the files this one has open at the fork.
    pub(crate) fn child(&self, sh: &Shared, threads: u64) -> ActorState {
        ActorState::new(sh, self.template, self.actor, false, Some(self.fds.freeze()), threads)
    }

    /// An `O_DIRECT` backend: an unaligned write cannot be issued (it would need
    /// read-modify-write); an unaligned read is rounded out to alignment by the driver and
    /// the requested part counted (what an `O_DIRECT` shim under a buffered application has
    /// to do).
    pub(crate) fn check_align(&self, sh: &Shared, op: &Op) -> Result<()> {
        if op.kind == OpKind::Write && sh.opts.backend.direct() && (op.offset % ALIGN as i64 != 0 || op.len % ALIGN as i64 != 0) {
            bail!(
                "write {} off={} len={}: `{}` needs {ALIGN}-byte alignment for writes; use --cache per-open",
                op.path,
                op.offset,
                op.len,
                sh.opts.backend.describe()
            );
        }
        Ok(())
    }

    pub(crate) fn fd(&self, path: &str) -> std::io::Result<Arc<OpenFile>> {
        self.fds.get(path).ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))
    }

    /// After a successful open: own the handle, note a creation, count an input open.
    pub(crate) fn opened(&mut self, sh: &Shared, path: &str, aux: u64, file: OpenFile) {
        self.fds.own.insert(Arc::from(path), Arc::new(file));
        if aux & (1 << (crate::ast::OpenFlag::CREAT as u8)) != 0 {
            self.created.push((path.to_string(), self.actor));
        }
        if let Some(writer) = sh.input_objects.get(path) {
            self.st.input_opens += 1;
            if writer.as_deref() == Some(sh.host.as_str()) {
                self.st.warm_opens += 1;
            }
        }
    }

    /// `close`: drop this actor's reference (the descriptor closes with the last one); a file
    /// inherited from the parent is marked closed here without closing it there.
    pub(crate) fn close(&mut self, path: &str) -> std::io::Result<()> {
        let key: Arc<str> = Arc::from(path);
        if self.fds.own.remove(&key).is_none() {
            if self.fds.get(path).is_none() {
                return Err(std::io::Error::from_raw_os_error(libc::EBADF));
            }
            self.fds.closed.insert(key);
        }
        Ok(())
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

    /// The structural check of an op's result and its recording: a read must return the
    /// computed count, a write its length, `readdir` the computed entries; a failure must be
    /// in the statement's `expect` list.
    pub(crate) fn settle(&mut self, op: &Op, ctx: &OpCtx, r: std::io::Result<i64>, ns: u64) -> Result<()> {
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
                let code = crate::object::errno_of(&e).unwrap_or(0);
                let name = errno_name(code);
                if op.expect.iter().any(|x| x == name) {
                    self.st.expected_errors += 1;
                    self.record(op, ctx, ns, 0);
                    Ok(())
                } else {
                    let mut extra = String::new();
                    if matches!(op.kind, OpKind::Read | OpKind::Write) {
                        extra = format!(" off={} len={}", op.offset, op.len);
                    }
                    if code == libc::EMFILE {
                        extra.push_str(" [RLIMIT_NOFILE reached: the `limits:` lines at startup have the estimate and the limit]");
                    }
                    Err(anyhow!("{}{}: {} ({})", where_(), extra, e, name))
                }
            }
        }
    }

    /// `compute`: recorded unscaled, and against the last take.
    pub(crate) fn computed(&mut self, ns: u64) {
        self.st.compute_ns += ns;
        if let Some(last) = self.takes.last_mut() {
            last.compute_ns += ns;
        }
    }

    pub(crate) fn took(&mut self, stall_ns: u64) {
        self.st.takes += 1;
        self.takes.push(TakeRec { stall_ns, compute_ns: 0 });
    }

    /// What every actor hands to the run when it ends: created and removed paths; for a main
    /// line also its barrier departures, its take record, and its statistics (a sub-actor's
    /// statistics go to its parent).
    pub(crate) fn finish_shared(&mut self, sh: &Shared) {
        if !self.created.is_empty() {
            sh.created.lock().unwrap().append(&mut self.created);
        }
        if !self.removed.is_empty() {
            sh.removed.lock().unwrap().append(&mut self.removed);
        }
        if self.main {
            if let Some(scopes) = sh.scopes.get(self.template) {
                for s in scopes {
                    sh.coord.leave(s);
                }
            }
            sh.actors.lock().unwrap().push(ActorRecord {
                template: self.template.to_string(),
                actor: self.actor,
                takes: std::mem::take(&mut self.takes),
                elapsed: self.started.elapsed(),
            });
            sh.stats.lock().unwrap().merge(&std::mem::take(&mut self.st));
        }
    }
}

/// Issue the op through a blocking backend; `Ok(n)` is the count it returned (bytes, entries,
/// or 0). The `sync` sink's whole backend, and what the `io_uring` loop runs inline for the
/// ops the ring has no opcode for (`lseek`, `ioctl`, `readdir`) or the kernel lacks.
pub(crate) fn issue_blocking(be: &mut dyn Backend, sh: &Shared, a: &mut ActorState, rbuf: &mut Ring, wbuf: &mut Ring, op: &Op) -> std::io::Result<i64> {
    // a path in an object store is the object engine's, whatever the run's API
    if let Some((store, rest)) = sh.opts.endpoints.object(op.path) {
        return crate::object::issue(&store, rest, sh, a, rbuf, op);
    }
    let full = |rel: &str| sh.opts.path(rel);
    match op.kind {
        OpKind::Open => {
            let mode = (op.aux >> 32) as u32;
            let mode = if mode == 0 { 0o644 } else { mode };
            let fd = be.open(&full(op.path), op.aux & 0xffff_ffff, mode)?;
            a.opened(sh, op.path, op.aux, fd);
            Ok(0)
        }
        OpKind::Close => {
            a.close(op.path).map(|_| 0)
        }
        OpKind::Read => {
            let fd = a.fd(op.path)?;
            let al = ALIGN as i64;
            if sh.opts.backend.direct() && (op.offset % al != 0 || op.len % al != 0) {
                // round out; O_DIRECT needs the offset, length, and buffer aligned
                let (lo, hi) = round_out(op.offset, op.len);
                let buf = rbuf.slice((hi - lo) as usize);
                let n = be.read(&fd, buf, Some(lo))? as i64;
                let got = (n - (op.offset - lo)).clamp(0, op.len);
                if !op.positioned && !be.positional() {
                    // keep the file position where a plain read would have left it
                    be.lseek(&fd, op.offset + got, crate::ast::Whence::SET)?;
                }
                return Ok(got);
            }
            let buf = rbuf.slice(op.len as usize);
            let off = if op.positioned || be.positional() { Some(op.offset) } else { None };
            be.read(&fd, buf, off).map(|n| n as i64)
        }
        OpKind::Write => {
            let fd = a.fd(op.path)?;
            let buf = wbuf.slice(op.len as usize);
            fill(a, op, buf);
            let off = if op.positioned || be.positional() { Some(op.offset) } else { None };
            be.write(&fd, buf, off).map(|n| n as i64)
        }
        OpKind::Lseek => {
            let fd = a.fd(op.path)?;
            be.lseek(&fd, op.offset, crate::ast::Whence::from_code(op.aux))
        }
        OpKind::Ioctl => {
            let fd = a.fd(op.path)?;
            be.ioctl(&fd, crate::ast::IoctlRequest::from_code(op.aux)).map(|_| 0)
        }
        OpKind::Fadvise => {
            let fd = a.fd(op.path)?;
            be.fadvise(&fd, op.offset, op.len, crate::ast::Advice::from_code(op.aux)).map(|_| 0)
        }
        OpKind::Fstat => {
            let fd = a.fd(op.path)?;
            be.fstat(&fd)
        }
        OpKind::Stat => be.stat(&full(op.path)),
        OpKind::Fsync => {
            let fd = a.fd(op.path)?;
            be.fsync(&fd).map(|_| 0)
        }
        OpKind::Fdatasync => {
            let fd = a.fd(op.path)?;
            be.fdatasync(&fd).map(|_| 0)
        }
        OpKind::Unlink => {
            be.unlink(&full(op.path))?;
            a.removed.push(op.path.to_string());
            Ok(0)
        }
        OpKind::Ftruncate => {
            let fd = a.fd(op.path)?;
            be.ftruncate(&fd, op.len).map(|_| 0)
        }
        OpKind::Fallocate => {
            let fd = a.fd(op.path)?;
            be.fallocate(&fd, op.offset, op.len).map(|_| 0)
        }
        OpKind::Mkdir => {
            be.mkdir(&full(op.path), op.aux as u32)?;
            a.created.push((op.path.to_string(), a.actor));
            Ok(0)
        }
        OpKind::Rmdir => be.rmdir(&full(op.path)).map(|_| 0),
        OpKind::Rename => {
            be.rename(&full(op.path), &full(op.path2.unwrap_or("")))?;
            a.removed.push(op.path.to_string());
            a.created.push((op.path2.unwrap_or("").to_string(), a.actor));
            Ok(0)
        }
        OpKind::Readdir => {
            let fd = a.fd(op.path)?;
            be.readdir(&fd).map(|n| n as i64)
        }
    }
}

/// The `O_DIRECT` rounding of an unaligned `[offset, offset + len)`: aligned `[lo, hi)`.
pub(crate) fn round_out(offset: i64, len: i64) -> (i64, i64) {
    let al = ALIGN as i64;
    (offset - offset.rem_euclid(al), (offset + len + al - 1) / al * al)
}

/// A write's content: the positional payload of the object at the op's offset (§5).
pub(crate) fn fill(a: &mut ActorState, op: &Op, buf: &mut [u8]) {
    let seed = payload::object_seed(op.seed, op.path);
    a.filler.fill_range(|b| payload::block_seed(seed, 0, b), op.offset as u64, buf);
}

// ---------------------------------------------------------------- the sink

/// One `parallel` sub-actor for a pool thread: where the parent stood at the fork (one
/// snapshot and one file table for all the sub-actors of the fork) and which sub-actor of
/// the fork this is.
struct Job {
    snap: Arc<Snapshot<'static, 'static>>,
    k: i64,
    fds: Arc<FdTable>,
}

/// How the forking actor waits for a fork's sub-actors: a count the pool threads take down,
/// the last of them waking the parent once (not once per sub-actor), and the errors.
struct Join {
    remaining: AtomicUsize,
    parent: std::thread::Thread,
    errs: Mutex<Vec<(i64, anyhow::Error)>>,
}

impl Join {
    fn done(&self, err: Option<(i64, anyhow::Error)>) {
        if let Some(e) = err {
            self.errs.lock().unwrap_or_else(|e| e.into_inner()).push(e);
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.parent.unpark();
        }
    }
}

/// An actor's `parallel` sub-actor threads, kept between forks. The forking actor runs
/// sub-actor 0 of a fork on its own thread (it would otherwise sleep until the others end)
/// and sub-actor `k ≥ 1` always runs on pool thread `k − 1` (the assignment is positional;
/// there is no shared queue); the pool grows to the widest fork the actor has issued, less
/// one. A sub-actor run on the forking thread may fork in turn while this pool is busy, so
/// an actor has one pool per depth of fork it runs itself (`Runner::pools`). A thread keeps what a
/// sub-actor needs and what it produces: its backend, its two buffer rings, its VM (resumed
/// from the fork's snapshot in place), its statistics (summed over the sub-actors it ran
/// and handed to the forking actor when the pool ends, since only the sum is ever
/// reported) and, when its sub-actors fork in turn, its own pool. Per sub-actor it gets
/// only the file table (what the parent had open at that fork); what the sub-actor opens
/// itself ends with it. Threads idle in `recv` between forks and end with the actor that
/// owns the pool.
#[derive(Default)]
struct Pool {
    workers: Vec<(mpsc::Sender<Job>, JoinHandle<Stats>)>,
    join: Option<Arc<Join>>,
}

impl Pool {
    /// Grow to `threads` threads; returns how many were spawned. Called by the forking actor
    /// on its own thread, which is the one the pool's threads wake.
    fn grow(&mut self, threads: usize, sh: &Arc<Shared>, inst: &Arc<Instance>, template: &'static str, actor: i64) -> u64 {
        let join = self.join.get_or_insert_with(|| Arc::new(Join { remaining: AtomicUsize::new(0), parent: std::thread::current(), errs: Mutex::new(Vec::new()) })).clone();
        let mut spawned = 0;
        while self.workers.len() < threads {
            let (tx, rx) = mpsc::channel::<Job>();
            let (sh, inst, join) = (sh.clone(), inst.clone(), join.clone());
            let h = std::thread::Builder::new()
                .name(format!("{template}#{actor} sub {}", self.workers.len()))
                .spawn(move || {
                    let mut sink = Runner::new(sh.clone(), inst, template, actor, false, None);
                    sink.a.st.threads = 0;
                    let mut vm: Option<Vm<'static, 'static>> = None;
                    while let Ok(Job { snap, k, fds }) = rx.recv() {
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            sink.a.fds = FdTable { own: HashMap::new(), inherited: Some(fds), closed: HashSet::new() };
                            sink.a.takes.clear();
                            let vm = match &mut vm {
                                Some(vm) => {
                                    vm.resume_from(&snap);
                                    vm
                                }
                                None => vm.insert(Vm::resume((*snap).clone())),
                            };
                            drop(snap);
                            vm.start_sub(k);
                            let r = drive(vm, &mut sink);
                            sink.a.finish_shared(&sh);
                            // the sub-actor's own files end here, before the parent's join
                            sink.a.fds = FdTable { own: HashMap::new(), inherited: None, closed: HashSet::new() };
                            r
                        }));
                        let (err, dead) = match r {
                            Ok(Ok(())) => (None, false),
                            Ok(Err(e)) => (Some((k, e)), false),
                            Err(_) => (Some((k, anyhow!("a sub-actor panicked"))), true),
                        };
                        if err.is_some() {
                            sh.aborted.store(true, Ordering::Relaxed);
                        }
                        join.done(err);
                        if dead {
                            break;
                        }
                    }
                    sink.drain_pools();
                    std::mem::take(&mut sink.a.st)
                })
                .expect("spawn");
            self.workers.push((tx, h));
            spawned += 1;
        }
        spawned
    }

    /// End the threads and return what their sub-actors counted.
    fn drain(&mut self) -> Stats {
        let mut st = Stats::default();
        for (tx, h) in self.workers.drain(..) {
            drop(tx);
            if let Ok(s) = h.join() {
                st.merge(&s);
            }
        }
        st
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.drain();
    }
}

pub struct Runner {
    sh: Arc<Shared>,
    inst: Arc<Instance>,
    be: Box<dyn Backend>,
    rbuf: Ring,
    wbuf: Ring,
    a: ActorState,
    /// The sub-actor pools, by the depth of fork this thread is running: `pools[0]` for a
    /// fork of the actor itself, `pools[1]` for a fork of the sub-actor 0 it runs on its own
    /// thread, and so on. `subs` are the VMs of those sub-actors, kept between forks.
    pools: Vec<Pool>,
    subs: Vec<Option<Vm<'static, 'static>>>,
    depth: usize,
}

impl Runner {
    fn new(sh: Arc<Shared>, inst: Arc<Instance>, template: &'static str, actor: i64, main: bool, inherited: Option<Arc<FdTable>>) -> Self {
        let be = sh.backend();
        let buf = sh.opts.buffer_bytes;
        let a = ActorState::new(&sh, template, actor, main, inherited, 1);
        Runner { sh, inst, be, rbuf: Ring::new(buf), wbuf: Ring::new(buf), a, pools: Vec::new(), subs: Vec::new(), depth: 0 }
    }

    /// End the pools' threads; what their sub-actors counted is this actor's.
    fn drain_pools(&mut self) {
        for p in &mut self.pools {
            let st = p.drain();
            self.a.st.merge(&st);
        }
    }

    /// Run sub-actor 0 of a fork on this thread, as the sub-actor it is: it sees the files
    /// open at the fork and its own opens end with it, it is not the main line (no barrier,
    /// no take record), and a fork inside it uses the next pool.
    fn run_inline(&mut self, snap: &Snapshot<'static, 'static>, fds: Arc<FdTable>) -> Result<()> {
        let d = self.depth;
        if self.subs.len() <= d {
            self.subs.resize_with(d + 1, || None);
        }
        let mut vm = match self.subs[d].take() {
            Some(mut vm) => {
                vm.resume_from(snap);
                vm
            }
            None => Vm::resume(snap.clone()),
        };
        let table = FdTable { own: HashMap::new(), inherited: Some(fds), closed: HashSet::new() };
        let (fds, takes, main) = (std::mem::replace(&mut self.a.fds, table), std::mem::take(&mut self.a.takes), self.a.main);
        self.a.main = false;
        self.depth += 1;
        vm.start_sub(0);
        let r = drive(&mut vm, self);
        self.depth -= 1;
        self.a.fds = fds;
        self.a.takes = takes;
        self.a.main = main;
        self.subs[d] = Some(vm);
        r
    }

    fn child(&self) -> Runner {
        Runner::new(self.sh.clone(), self.inst.clone(), self.a.template, self.a.actor, false, Some(self.a.fds.freeze()))
    }

    fn spawn_sub(&self, snap: Snapshot<'static, 'static>, label: String, work: impl FnOnce(&mut Vm<'static, 'static>, &mut Runner) -> Result<()> + Send + 'static) -> JoinHandle<(Result<()>, Stats)> {
        let mut sink = self.child();
        std::thread::Builder::new()
            .name(label)
            .spawn(move || {
                let mut vm = Vm::resume(snap);
                let r = work(&mut vm, &mut sink);
                let r = r.and_then(|_| sink.finish());
                if r.is_err() {
                    sink.sh.aborted.store(true, Ordering::Relaxed);
                }
                (r, std::mem::take(&mut sink.a.st))
            })
            .expect("spawn")
    }
}

impl Sink<'static, 'static> for Runner {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> Result<()> {
        if self.sh.aborted.load(Ordering::Relaxed) {
            bail!("run aborted: {}", self.sh.coord.abort_reason().unwrap_or_else(|| "another actor failed".into()));
        }
        self.a.check_align(&self.sh, op)?;
        let t = Instant::now();
        let r = issue_blocking(&mut *self.be, &self.sh, &mut self.a, &mut self.rbuf, &mut self.wbuf, op);
        let ns = t.elapsed().as_nanos() as u64;
        self.a.settle(op, ctx, r, ns)
    }

    fn control(&mut self, c: Control<'static>, ctx: &OpCtx) -> Result<()> {
        match c {
            Control::Compute { ns } => {
                let ns = ns.max(0) as u64;
                let scaled = (ns as f64 * self.sh.opts.time_scale) as u64;
                if scaled > 0 {
                    std::thread::sleep(Duration::from_nanos(scaled));
                }
                self.a.computed(ns);
            }
            Control::Barrier { scope } => {
                if !self.a.main {
                    bail!("barrier `{scope}` inside a sub-actor is not supported");
                }
                let waited = self.sh.coord.barrier(scope, &self.sh.aborted)?;
                self.a.st.barriers += 1;
                self.a.st.barrier_wait_ns += waited.as_nanos() as u64;
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
                self.a.st.puts += 1;
            }
            Control::Take { channel } => {
                let ch = self.inst.channels.lock().unwrap().get(channel).cloned().ok_or_else(|| anyhow!("take on undeclared channel `{channel}` (a loader declares one under its name)"))?;
                let t = Instant::now();
                ch.take(&self.sh.aborted).with_context(|| {
                    let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
                    format!("{}#{} [{}] take `{channel}`", ctx.template, ctx.actor, idx.join(","))
                })?;
                self.a.took(t.elapsed().as_nanos() as u64);
            }
        }
        Ok(())
    }

    fn trace(&mut self, t: &Arc<crate::trace::TraceFile>, ctx: &OpCtx) -> Result<()> {
        self.trace_node(t, ctx)
    }

    fn fork(&mut self, kind: &ForkKind<'static>, snapshot: &dyn Fn() -> Snapshot<'static, 'static>) -> Result<bool> {
        let snap = snapshot();
        match *kind {
            ForkKind::Parallel { index, width } => {
                let width = width.max(0) as usize;
                if width == 0 {
                    return Ok(true);
                }
                let d = self.depth;
                if self.pools.len() <= d {
                    self.pools.resize_with(d + 1, Pool::default);
                }
                self.a.st.threads += self.pools[d].grow(width - 1, &self.sh, &self.inst, self.a.template, self.a.actor);
                let join = self.pools[d].join.clone().expect("a grown pool has its join");
                let (snap, fds) = (Arc::new(snap), self.a.fds.freeze());
                join.remaining.store(width - 1, Ordering::Release);
                for (j, (tx, _)) in self.pools[d].workers.iter().take(width - 1).enumerate() {
                    let k = j as i64 + 1;
                    if tx.send(Job { snap: snap.clone(), k, fds: fds.clone() }).is_err() {
                        join.done(Some((k, anyhow!("a `{index}` sub-actor thread has ended (an earlier sub-actor panicked)"))));
                    }
                }
                // sub-actor 0 here, while the others run; then wait for them
                let first = self.run_inline(&snap, fds);
                if first.is_err() {
                    self.sh.aborted.store(true, Ordering::Relaxed);
                }
                while join.remaining.load(Ordering::Acquire) != 0 {
                    std::thread::park();
                }
                first?;
                let mut errs = std::mem::take(&mut *join.errs.lock().unwrap_or_else(|e| e.into_inner()));
                errs.sort_by_key(|(k, _)| *k);
                match errs.into_iter().next() {
                    Some((_, e)) => Err(e),
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
                    let label = format!("{}#{} {index} worker {w}", self.a.template, self.a.actor);
                    let h = self.spawn_sub(snap.clone(), label, move |vm, sink| {
                        let aborted = &sink.sh.clone().aborted;
                        let mut b = w;
                        while b < batches {
                            if let Err(e) = chan.begin(b, aborted) {
                                // closed by the consumer: not this worker's error
                                if !chan.m.lock().unwrap().closed {
                                    chan.fail(format!("{e:#}"));
                                }
                                return Ok(());
                            }
                            vm.start_sub(b);
                            if let Err(e) = drive(vm, sink) {
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
        // the pool's threads end here; what their sub-actors counted is this actor's
        self.drain_pools();
        if self.a.main {
            // loaders: close their channels so a worker blocked on a slot exits, then join
            let channels: Vec<(String, Arc<Channel>)> = self.inst.channels.lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            for (_, ch) in &channels {
                ch.close();
            }
            let loaders: Vec<_> = std::mem::take(&mut *self.inst.loaders.lock().unwrap());
            for (name, h) in loaders {
                match h.join() {
                    Ok((r, st)) => {
                        self.a.st.merge(&st);
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
                        result = Err(untaken(name, taken, total));
                    }
                }
            }
        }
        self.a.finish_shared(&self.sh);
        if result.is_err() {
            self.sh.aborted.store(true, Ordering::Relaxed);
        }
        result
    }
}

/// A loader whose consumer did not take every batch (`NAPKIN_MATH.md` §4.1, finite producers).
pub(crate) fn untaken(name: &str, taken: i64, total: i64) -> anyhow::Error {
    anyhow!("loader `{name}`: {taken} of {total} batches were taken; the consumer must take exactly `batches` items (`NAPKIN_MATH.md` §4.1, finite producers)")
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
pub fn check_datasets(loaded: &crate::Loaded, cfg: &Config, root: &(impl Place + ?Sized)) -> Result<Vec<DatasetCheck>> {
    let mut out = Vec::new();
    for name in loaded.ast.datasets.keys() {
        let rel = payload::dataset_root(&loaded.ast, name)?;
        let dir = root.at(&rel);
        let m = match root.object(&rel) {
            Some((store, rest)) => Manifest::read_object(&store, &rest),
            None => Manifest::read(&dir),
        }
        .with_context(|| format!("dataset `{name}` at {}: run `aeiou datagen` first", dir.display()))?;
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
pub fn check_input_namespaces(loaded: &crate::Loaded, cfg: &Config, root: &(impl Place + ?Sized), opts: &RunOpts) -> Result<(Vec<NamespaceCheck>, HashMap<String, Option<String>>)> {
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
        let dir = root.at(&rel);
        let m = match root.object(&rel) {
            Some((store, rest)) => NamespaceManifest::read_object(&store, &rest),
            None => NamespaceManifest::read(&dir),
        }
        .with_context(|| format!("input namespace(s) {} at {}: no run has written this root", names.join(", "), dir.display()))?;
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
        // V15: an abstract that reads names the writer drew is the writer's run or it is nothing
        let same: Vec<&String> = names.iter().filter(|n| ast.namespaces[*n].same_run.unwrap_or(false)).collect();
        if !same.is_empty() {
            let what = format!("input namespace(s) {} at {} (`same_run`): the names are draws of the run that wrote them (`{}`)", same.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "), dir.display(), m.abstract_name);
            let mut lines = Vec::new();
            if m.seed != cfg.seed {
                lines.push(format!("--seed {} here, {} in the writer", cfg.seed, m.seed));
            }
            if m.gpus != cfg.gpus {
                lines.push(format!("--gpus {} here, {} in the writer", cfg.gpus, m.gpus));
            }
            let mine = payload::params_json(&loaded.doc, cfg, &Params::new(ast, cfg)?)?;
            if let (Some(mine), Some(theirs)) = (mine.as_object(), m.params.as_object()) {
                for (k, v) in mine {
                    if let Some(w) = theirs.get(k) {
                        if crate::canon::canonical(v) != crate::canon::canonical(w) {
                            lines.push(format!("parameter `{k}`: {} here, {} in the writer", String::from_utf8_lossy(&crate::canon::canonical(v)), String::from_utf8_lossy(&crate::canon::canonical(w))));
                        }
                    }
                }
            }
            if !lines.is_empty() {
                bail!("{what}; this run differs:\n  {}", lines.join("\n  "));
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
/// `input`), written last and atomically, with the objects created there and the rank
/// records of `report` (every host's, when it is the coordinator's merged report).
pub fn write_namespace_manifests(loaded: &crate::Loaded, cfg: &Config, root: &(impl Place + ?Sized), _opts: &RunOpts, report: &Report, started: f64, finished: f64) -> Result<Vec<PathBuf>> {
    let ast = &loaded.ast;
    let mut by_root: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, n) in &ast.namespaces {
        if !n.input.unwrap_or(false) {
            by_root.entry(payload::namespace_root(ast, name)?).or_default().push(name.clone());
        }
    }
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
            ranks: report.ranks.clone(),
            started,
            finished,
            objects_created: count,
            bytes_written: report.stats.bytes_written,
            objects: if objects.len() <= payload::NAMESPACE_OBJECT_LIMIT { Some(objects) } else { None },
        };
        written.push(match root.object(&rel) {
            Some((store, rest)) => PathBuf::from(m.write_object(&store, &rest)?),
            None => m.write(&root.at(&rel))?,
        });
    }
    Ok(written)
}

/// Output namespace roots must be empty (`--clean-namespaces` empties them); input roots
/// are left as they are; dataset roots inside a namespace root are left alone. Creates the
/// output roots; one in an object store is a prefix, which a `LIST` finds empty or a
/// `DELETE` of every key below it empties.
pub fn prepare_namespaces(ast: &Ast, root: &(impl Place + ?Sized), clean: bool) -> Result<Vec<String>> {
    let mut dataset_roots: Vec<PathBuf> = Vec::new();
    for name in ast.datasets.keys() {
        dataset_roots.push(root.at(&payload::dataset_root(ast, name)?));
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
        if let Some((store, rest)) = root.object(&rel) {
            // the keys of a dataset placed under this prefix are the dataset's
            let mut skip: Vec<String> = Vec::new();
            for d in ast.datasets.keys() {
                if let Some((s, r)) = root.object(&payload::dataset_root(ast, d)?) {
                    if Arc::ptr_eq(&s, &store) {
                        skip.push(format!("{r}/"));
                    }
                }
            }
            let stale: Vec<String> = store.keys_below(&rest)?.into_iter().filter(|k| !skip.iter().any(|d| k.starts_with(d.as_str()))).collect();
            if !stale.is_empty() {
                if !clean {
                    bail!("namespace `{name}` root {} is not empty ({} objects, e.g. {}); pass --clean-namespaces to empty it", store.show(&rest), stale.len(), store.show(&stale[0]));
                }
                store.delete_all(&stale)?;
                cleaned.push(rel.clone());
            }
            continue;
        }
        let dir = root.at(&rel);
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

/// What a run produced: one host's, or every host's after the coordinator merged them
/// (`merge_all`). It is what a host sends to the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub elapsed: Duration,
    pub stats: Stats,
    /// (template, instance count)
    pub templates: Vec<(String, i64)>,
    /// The hosts and the GPU id range each ran.
    pub ranks: Vec<RankRecord>,
    pub actors: Vec<ActorRecord>,
    pub departure_releases: Vec<(String, u64)>,
    pub threads_peak: u64,
    /// What the host did meanwhile (`counters`): tasks, io-wq workers, CPU, the mount's RPCs.
    pub counters: HostCounters,
    /// The rings' setup under the `io_uring` backends (rank 0's once merged); `None` under `sync`.
    pub uring: Option<UringReport>,
    /// The AIO contexts under the `libaio` backends, and the mappings under `mmap`.
    #[serde(default)]
    pub aio: Option<AioReport>,
    #[serde(default)]
    pub mmap: Option<MmapReport>,
    /// Each host's cold start (`cold`): the drop when asked, and with `--drop-caches` or
    /// `--require-cold` the dataset residency sample.
    /// Filled by the caller that ran `cold::start` before the gate; empty otherwise.
    #[serde(default)]
    pub cold: Vec<crate::cold::ColdStart>,
    /// Objects created (path, creating GPU id) net of this host's removals, and the paths
    /// removed, so a merge can drop what another host created and this one removed.
    pub created: Vec<(String, i64)>,
    pub removed: Vec<String>,
    /// The host, or the hosts joined with `,` once merged.
    pub host: String,
}

impl Report {
    /// Merge every host's report: stats summed (the fingerprint modulo 2^64, histograms
    /// bucket-wise), instances concatenated, created objects net of every removal, elapsed
    /// the longest host's.
    pub fn merge_all(reports: Vec<Report>) -> Report {
        let mut it = reports.into_iter();
        let Some(mut m) = it.next() else {
            return Report { elapsed: Duration::ZERO, stats: Stats::default(), templates: vec![], ranks: vec![], actors: vec![], departure_releases: vec![], threads_peak: 0, counters: HostCounters::default(), uring: None, aio: None, mmap: None, cold: vec![], created: vec![], removed: vec![], host: String::new() };
        };
        let mut hosts = vec![m.host.clone()];
        let mut departures: BTreeMap<String, u64> = m.departure_releases.drain(..).collect();
        for r in it {
            m.elapsed = m.elapsed.max(r.elapsed);
            m.stats.merge(&r.stats);
            m.ranks.extend(r.ranks);
            m.actors.extend(r.actors);
            for (k, n) in r.departure_releases {
                *departures.entry(k).or_insert(0) += n;
            }
            m.threads_peak += r.threads_peak;
            m.counters.merge(&r.counters);
            if let (Some(a), Some(b)) = (&mut m.uring, &r.uring) {
                a.in_flight_peak = a.in_flight_peak.max(b.in_flight_peak);
            }
            match (&mut m.aio, &r.aio) {
                (Some(a), Some(b)) => a.merge(b),
                (None, Some(b)) => m.aio = Some(b.clone()),
                _ => {}
            }
            match (&mut m.mmap, &r.mmap) {
                (Some(a), Some(b)) => a.merge(b),
                (None, Some(b)) => m.mmap = Some(b.clone()),
                _ => {}
            }
            m.cold.extend(r.cold);
            m.created.extend(r.created);
            m.removed.extend(r.removed);
            hosts.push(r.host);
        }
        let removed: HashSet<&String> = m.removed.iter().collect();
        m.created.retain(|(p, _)| !removed.contains(p));
        m.created.sort();
        m.created.dedup();
        m.ranks.sort_by_key(|r| r.rank);
        m.actors.sort_by(|a, b| a.template.cmp(&b.template).then(a.actor.cmp(&b.actor)));
        m.departure_releases = departures.into_iter().collect();
        hosts.dedup();
        m.host = hosts.join(",");
        m
    }
}

/// The barrier scopes this host's instances participate in, with the instance count of each:
/// what `coord::Local::new` and the coordinator's `Hello` take.
pub fn participants(model: &Model<'_>, opts: &RunOpts) -> Result<Vec<(String, usize)>> {
    let counts = actor_counts(model)?;
    let scopes = barrier_scopes(model.ast)?;
    let mut participants: BTreeMap<String, usize> = BTreeMap::new();
    for (name, c) in &counts {
        let (lo, hi) = gpu_range(*c, opts.ranks, opts.rank, opts.rank_rotate);
        for s in scopes.get(*name).map(|v| v.as_slice()).unwrap_or(&[]) {
            *participants.entry(s.clone()).or_insert(0) += (hi - lo) as usize;
        }
    }
    Ok(participants.into_iter().collect())
}

/// Execute the abstract on one host with the in-process coordinator. `model` must outlive
/// the threads, hence `'static` (the caller leaks the loaded abstract and model for the life
/// of the process; a run is the process).
pub fn run(model: &'static Model<'static>, opts: RunOpts, input_objects: HashMap<String, Option<String>>) -> Result<Report> {
    if opts.uring.any() && !opts.backend.uring() {
        bail!("io_uring knobs ({}) under --posix {}, which has no ring", opts.uring.describe(), opts.backend.api.name());
    }
    let p = participants(model, &opts)?;
    run_with(model, opts, input_objects, Arc::new(Local::new(&p)), Arc::new(AtomicBool::new(false)))
}

/// Execute the abstract with the given coordinator (`coord::Tcp` for several hosts) and
/// abort flag (which the coordinator sets when another host fails). On failure here the
/// coordinator is told to stop the other hosts.
pub fn run_with(model: &'static Model<'static>, opts: RunOpts, input_objects: HashMap<String, Option<String>>, coord: Arc<dyn Coordinator>, aborted: Arc<AtomicBool>) -> Result<Report> {
    let counts: Vec<(&'static str, i64)> = actor_counts(model)?;
    let ranges: Vec<(i64, i64)> = counts.iter().map(|(_, c)| gpu_range(*c, opts.ranks, opts.rank, opts.rank_rotate)).collect();
    let scopes = barrier_scopes(model.ast)?;
    let (glo, ghi) = gpu_range(model.cfg.gpus, opts.ranks, opts.rank, opts.rank_rotate);
    let rank_record = RankRecord { rank: opts.rank, host: hostname(), gpus: [glo, ghi] };
    let sh = Arc::new(Shared {
        opts,
        coord,
        aborted,
        stats: Mutex::new(Stats::default()),
        actors: Mutex::new(Vec::new()),
        scopes,
        host: hostname(),
        input_objects,
        created: Mutex::new(Vec::new()),
        removed: Mutex::new(Vec::new()),
        mmap_stats: Arc::new(MmapStats::default()),
    });

    crate::backend::open_files_reset();
    let sampler = Sampler::start(&sh.opts.root);
    let t0 = Instant::now();
    if sh.opts.event_loop() {
        let loops = if sh.opts.backend.uring() {
            crate::uring::run(model, &sh, &counts, &ranges).map(|(u, sqpoll)| (u.loops, Some(u), None, sqpoll))
        } else if sh.opts.backend.libaio() {
            crate::aio::run(model, &sh, &counts, &ranges).map(|a| (a.loops, None, Some(a), 0))
        } else {
            // S3 names alone under `--s3 async`: loops that wait on their eventfd, no ring
            crate::uring::run_s3(model, &sh, &counts, &ranges).map(|n| (n as _, None, None, 0))
        };
        let elapsed = t0.elapsed();
        let mut counters = sampler.finish();
        return match loops {
            Ok((n, uring, aio, sqpoll)) => {
                counters.sqpoll_threads = sqpoll;
                sh.stats.lock().unwrap().threads += n;
                assemble(sh, counts, rank_record, elapsed, counters, uring, aio)
            }
            Err(e) => {
                sh.coord.stop(&format!("{e:#}"));
                Err(e)
            }
        };
    }
    let mut handles = Vec::new();
    for (&(template, count), &(lo, hi)) in counts.iter().zip(&ranges) {
        for actor in lo..hi {
            let sh = sh.clone();
            let inst = Arc::new(Instance { channels: Mutex::new(HashMap::new()), loaders: Mutex::new(Vec::new()) });
            let h = std::thread::Builder::new()
                .name(format!("{template}#{actor}"))
                .spawn(move || {
                    let mut sink = Runner::new(sh.clone(), inst, template, actor, true, None);
                    let body = &model.ast.actors[template].body;
                    let mut vm = Vm::new(model, template, actor, count);
                    vm.start(body);
                    let r = drive(&mut vm, &mut sink).and_then(|_| sink.finish());
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
    let counters = sampler.finish();
    if let Some(e) = first_err {
        sh.coord.stop(&format!("{e:#}"));
        return Err(e);
    }
    assemble(sh, counts, rank_record, elapsed, counters, None, None)
}

/// The host's report once every actor has ended.
fn assemble(sh: Arc<Shared>, counts: Vec<(&'static str, i64)>, rank_record: RankRecord, elapsed: Duration, counters: HostCounters, uring: Option<UringReport>, aio: Option<AioReport>) -> Result<Report> {
    if sh.aborted.load(Ordering::Relaxed) {
        bail!("run aborted: {}", sh.coord.abort_reason().unwrap_or_else(|| "an actor failed".into()));
    }
    let stats = std::mem::take(&mut *sh.stats.lock().unwrap());
    let mut actors = std::mem::take(&mut *sh.actors.lock().unwrap());
    actors.sort_by(|a, b| a.template.cmp(&b.template).then(a.actor.cmp(&b.actor)));
    let mut removed: Vec<String> = std::mem::take(&mut *sh.removed.lock().unwrap());
    removed.sort();
    removed.dedup();
    let removed_set: HashSet<&String> = removed.iter().collect();
    let mut created: Vec<(String, i64)> = std::mem::take(&mut *sh.created.lock().unwrap()).into_iter().filter(|(p, _)| !removed_set.contains(p)).collect();
    created.sort();
    Ok(Report {
        elapsed,
        threads_peak: stats.threads,
        stats,
        templates: counts.iter().map(|(n, c)| (n.to_string(), *c)).collect(),
        ranks: vec![rank_record],
        actors,
        departure_releases: sh.coord.departure_releases(),
        counters,
        uring,
        aio,
        mmap: sh.opts.backend.mmap().then(|| {
            let m = &sh.mmap_stats;
            MmapReport {
                mode: sh.opts.mmap,
                consume: sh.opts.mmap_consume,
                touched_pages: m.touched_pages.load(Ordering::Relaxed),
                maps: m.maps.load(Ordering::Relaxed),
                mapped_bytes: m.mapped_bytes.load(Ordering::Relaxed),
                advised: m.advised.load(Ordering::Relaxed),
                copied_bytes: m.copied_bytes.load(Ordering::Relaxed),
            }
        }),
        cold: Vec::new(),
        created,
        removed,
        host: sh.host.clone(),
    })
}

// ---------------------------------------------------------------- report

/// The host counters (`counters`): one line for the process, one for the mount, and for an
/// NFS mount the RPCs by procedure with the mean round trip.
fn write_counters(out: &mut impl Write, c: &HostCounters) -> std::io::Result<()> {
    writeln!(
        out,
        "host: tasks peak {}  open files peak {}  io-wq workers peak {}{}  cpu user {} sys {}  maxrss {}  faults minor {} major {}",
        c.tasks_peak,
        c.open_files_peak,
        c.iowq_workers_peak,
        if c.sqpoll_threads > 0 { format!("  sqpoll threads {}", c.sqpoll_threads) } else { String::new() },
        human_ns(c.cpu_user_ns as i128),
        human_ns(c.cpu_sys_ns as i128),
        human_bytes(c.maxrss_bytes),
        c.minor_faults,
        c.major_faults
    )?;
    let Some(m) = &c.mount else { return Ok(()) };
    write!(out, "mount {} ({}, {})", m.mount_point, m.fstype, m.device)?;
    let mut opts = m.opts.as_ref().map(|o| format!("mount opts {o}\n")).unwrap_or_default();
    if !m.read_ahead_kb.is_empty() {
        let v: Vec<String> = m.read_ahead_kb.iter().map(|k| k.to_string()).collect();
        opts.push_str(&format!("mount read_ahead_kb {}\n", v.join(",")));
    }
    let Some(n) = &m.nfs else { return write!(out, "\n{opts}") };
    // the client counts buffered bytes as returned and O_DIRECT bytes as requested
    writeln!(
        out,
        ": server read {} wrote {}; buffered read {} wrote {}; O_DIRECT requested read {} wrote {}",
        human_bytes(n.server_read_bytes),
        human_bytes(n.server_write_bytes),
        human_bytes(n.normal_read_bytes),
        human_bytes(n.normal_write_bytes),
        human_bytes(n.direct_read_bytes),
        human_bytes(n.direct_write_bytes)
    )?;
    write!(out, "{opts}")?;
    let mut ops: Vec<_> = n.ops.iter().collect();
    ops.sort_by(|a, b| b.1.ops.cmp(&a.1.ops).then(a.0.cmp(b.0)));
    let total: u64 = ops.iter().map(|(_, o)| o.ops).sum();
    let cells: Vec<String> = ops
        .iter()
        .map(|(k, o)| {
            let mut cell = format!("{k}={}", o.ops);
            if o.ops > 0 {
                cell.push_str(&format!(" ({} rtt", us(o.rtt_ms * 1_000_000 / o.ops)));
                if o.timeouts > 0 || o.errors > 0 || o.transmissions != o.ops {
                    cell.push_str(&format!(", {} sent, {} timeouts, {} errors", o.transmissions, o.timeouts, o.errors));
                }
                cell.push(')');
            }
            cell
        })
        .collect();
    writeln!(out, "rpcs {total}: {}", cells.join("  "))?;
    writeln!(out, "  (the mount's counters over the run, every process on this host included)")
}

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
    if let Some(u) = &r.uring {
        writeln!(out, "io_uring: {}", u.describe())?;
    }
    if let Some(a) = &r.aio {
        writeln!(out, "libaio: {}", a.describe())?;
    }
    if let Some(m) = &r.mmap {
        writeln!(out, "mmap: {}", m.describe())?;
    }
    for c in &r.cold {
        crate::cold::write(out, c)?;
    }
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
    write_counters(out, &r.counters)?;
    if s.input_opens > 0 {
        writeln!(out, "input objects opened {}  of which written on the opening host ({}) {}", s.input_opens, r.host, s.warm_opens)?;
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

// ---------------------------------------------------------------- the `trace` node

/// The open table of one `trace` instance (`trace.rs`): one slot per open id, filled by the
/// lane that executes the `open`, read by every lane that uses it. The only waits a trace
/// run has: a use waits for its open; the last close of an id waits for its uses.
pub(crate) struct OpenTable {
    slots: Mutex<Vec<Slot>>,
    cv: Condvar,
}

struct Slot {
    fd: Option<Arc<OpenFile>>,
    /// Set when the open failed in this run (its uses, if any, cannot proceed).
    failed: bool,
    uses_left: usize,
    closes_left: usize,
}

impl OpenTable {
    fn new(tf: &crate::trace::TraceFile) -> Self {
        let slots = tf.opens.iter().map(|o| Slot { fd: None, failed: false, uses_left: o.uses, closes_left: o.closes }).collect();
        OpenTable { slots: Mutex::new(slots), cv: Condvar::new() }
    }

    fn publish(&self, oid: usize, fd: Option<Arc<OpenFile>>) {
        let mut s = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        s[oid].failed = fd.is_none();
        s[oid].fd = fd;
        drop(s);
        self.cv.notify_all();
    }

    /// Wait for open `oid` to be published; `Err` when it failed or the run was aborted.
    fn acquire(&self, oid: usize, path: &str, aborted: &AtomicBool) -> Result<Arc<OpenFile>> {
        let mut s = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(fd) = &s[oid].fd {
                return Ok(fd.clone());
            }
            if s[oid].failed {
                bail!("open #{oid} of {path} failed in this run; its later uses cannot be issued");
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("run aborted while waiting for open #{oid} of {path}");
            }
            s = self.cv.wait_timeout(s, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    fn release(&self, oid: usize) {
        let mut s = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        s[oid].uses_left = s[oid].uses_left.saturating_sub(1);
        drop(s);
        self.cv.notify_all();
    }

    /// One close line of `oid`: the descriptor closes with the last of them, after every
    /// use has completed. Returns the reference to drop (the close itself) when this is it.
    fn close(&self, oid: usize, path: &str, aborted: &AtomicBool) -> Result<Option<Arc<OpenFile>>> {
        let mut s = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        s[oid].closes_left = s[oid].closes_left.saturating_sub(1);
        if s[oid].closes_left > 0 {
            return Ok(None);
        }
        loop {
            if s[oid].uses_left == 0 {
                return Ok(s[oid].fd.take());
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("run aborted while waiting to close #{oid} of {path}");
            }
            s = self.cv.wait_timeout(s, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

/// Path order of one `trace` instance (`trace.rs`, module doc): per path the trace changes,
/// how many of its ops and how many of its changing ops have completed. A changing op at
/// index `k` among the path's ops waits for `done == k`; a reading op waits for `mdone ==
/// mk`, the changing ops before it.
pub(crate) struct PathOrder {
    done: Mutex<Vec<(usize, usize)>>,
    cv: Condvar,
}

impl PathOrder {
    fn new(tf: &crate::trace::TraceFile) -> Self {
        PathOrder { done: Mutex::new(vec![(0, 0); tf.changed.len()]), cv: Condvar::new() }
    }

    fn wait(&self, d: &crate::trace::Dep, path: &str, aborted: &AtomicBool) -> Result<()> {
        let mut s = self.done.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let (done, mdone) = s[d.path];
            if (d.mutating && done >= d.k) || (!d.mutating && mdone >= d.mk) {
                return Ok(());
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("run aborted while waiting for earlier ops on {path}");
            }
            s = self.cv.wait_timeout(s, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    fn complete(&self, d: &crate::trace::Dep) {
        let mut s = self.done.lock().unwrap_or_else(|e| e.into_inner());
        s[d.path].0 += 1;
        if d.mutating {
            s[d.path].1 += 1;
        }
        drop(s);
        self.cv.notify_all();
    }
}

/// The two tables of a `trace` instance.
pub(crate) struct TraceTables {
    opens: OpenTable,
    paths: PathOrder,
}

impl TraceTables {
    /// Path order before op `g`: wait for what it depends on.
    fn before(&self, tf: &crate::trace::TraceFile, g: usize, path: &str, aborted: &AtomicBool) -> Result<()> {
        if let Some(d) = &tf.deps[g] {
            self.paths.wait(d, path, aborted)?;
        }
        if let Some(d) = tf.deps2.get(&g) {
            self.paths.wait(d, path, aborted)?;
        }
        Ok(())
    }

    fn after(&self, tf: &crate::trace::TraceFile, g: usize) {
        if let Some(d) = &tf.deps[g] {
            self.paths.complete(d);
        }
        if let Some(d) = tf.deps2.get(&g) {
            self.paths.complete(d);
        }
    }
}

impl Runner {
    /// One op of a lane: the open table in place of the VM's handles, `issue_blocking` for
    /// the call itself (so `O_DIRECT` rounding, the payload, and the bookkeeping are the
    /// run's), `settle` for the check and the record.
    fn trace_op(&mut self, tf: &crate::trace::TraceFile, g: usize, op: &crate::trace::LineOp, tables: &TraceTables, ctx: &OpCtx, pos: &mut HashMap<usize, i64>) -> Result<()> {
        if self.sh.aborted.load(Ordering::Relaxed) {
            bail!("run aborted: {}", self.sh.coord.abort_reason().unwrap_or_else(|| "another actor failed".into()));
        }
        let b = tf.build(op, |fd| pos.get(&fd).copied().unwrap_or(0));
        if let (Some(fd), Some(p)) = (b.oid, b.pos_after) {
            pos.insert(fd, p);
        }
        let o = &b.op;
        self.a.check_align(&self.sh, o)?;
        let key: Arc<str> = Arc::from(o.path);
        let table = &tables.opens;
        tables.before(tf, g, o.path, &self.sh.aborted)?;
        let r = self.trace_issue(o, b.oid, table, ctx, &key);
        tables.after(tf, g);
        r
    }

    fn trace_issue(&mut self, o: &Op, oid: Option<usize>, table: &OpenTable, ctx: &OpCtx, key: &Arc<str>) -> Result<()> {
        let b_oid = oid;
        match o.kind {
            OpKind::Open => {
                let oid = b_oid.expect("an open has its id");
                let t = Instant::now();
                let r = issue_blocking(&mut *self.be, &self.sh, &mut self.a, &mut self.rbuf, &mut self.wbuf, o);
                let ns = t.elapsed().as_nanos() as u64;
                // the descriptor belongs to the table, not to this lane's own files
                table.publish(oid, self.a.fds.own.remove(key));
                self.a.settle(o, ctx, r, ns)
            }
            OpKind::Close => {
                let oid = b_oid.expect("a close has its id");
                let last = table.close(oid, o.path, &self.sh.aborted)?;
                let t = Instant::now();
                drop(last);
                let ns = t.elapsed().as_nanos() as u64;
                self.a.settle(o, ctx, Ok(0), ns)
            }
            _ => match b_oid {
                Some(oid) => {
                    let fd = table.acquire(oid, o.path, &self.sh.aborted)?;
                    self.a.fds.own.insert(key.clone(), fd);
                    let t = Instant::now();
                    let r = issue_blocking(&mut *self.be, &self.sh, &mut self.a, &mut self.rbuf, &mut self.wbuf, o);
                    let ns = t.elapsed().as_nanos() as u64;
                    self.a.fds.own.remove(key);
                    table.release(oid);
                    self.a.settle(o, ctx, r, ns)
                }
                None => {
                    let t = Instant::now();
                    let r = issue_blocking(&mut *self.be, &self.sh, &mut self.a, &mut self.rbuf, &mut self.wbuf, o);
                    let ns = t.elapsed().as_nanos() as u64;
                    self.a.settle(o, ctx, r, ns)
                }
            },
        }
    }

    /// A `submit` group: its members issued at once, one thread each (what the application's
    /// `io_submit` did), reaped before the lane goes on; settled in order afterwards.
    fn trace_group(&mut self, tf: &crate::trace::TraceFile, g0: usize, ops: &[crate::trace::LineOp], tables: &TraceTables, ctx: &OpCtx, pos: &mut HashMap<usize, i64>) -> Result<()> {
        let table = &tables.opens;
        let built: Vec<crate::trace::Built> = ops.iter().map(|op| tf.build(op, |fd| pos.get(&fd).copied().unwrap_or(0))).collect();
        for b in &built {
            self.a.check_align(&self.sh, &b.op)?;
        }
        let sh = self.sh.clone();
        let (template, actor) = (self.a.template, self.a.actor);
        let results: Vec<(std::io::Result<i64>, u64)> = std::thread::scope(|s| {
            let handles: Vec<_> = built
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let sh = sh.clone();
                    s.spawn(move || -> (std::io::Result<i64>, u64) {
                        let mut be = sh.backend();
                        let mut a = ActorState::new(&sh, template, actor, false, None, 0);
                        let (mut rbuf, mut wbuf) = (Ring::new(ALIGN), Ring::new(ALIGN));
                        let o = &b.op;
                        let Some(oid) = b.oid else { return (Err(std::io::Error::from_raw_os_error(libc::EINVAL)), 0) };
                        if tables.before(tf, g0 + j, o.path, &sh.aborted).is_err() {
                            return (Err(std::io::Error::from_raw_os_error(libc::ECANCELED)), 0);
                        }
                        let fd = match table.acquire(oid, o.path, &sh.aborted) {
                            Ok(fd) => fd,
                            Err(_) => return (Err(std::io::Error::from_raw_os_error(libc::EBADF)), 0),
                        };
                        a.fds.own.insert(Arc::from(o.path), fd);
                        let t = Instant::now();
                        let r = issue_blocking(&mut *be, &sh, &mut a, &mut rbuf, &mut wbuf, o);
                        let ns = t.elapsed().as_nanos() as u64;
                        table.release(oid);
                        tables.after(tf, g0 + j);
                        (r, ns)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| (Err(std::io::Error::from_raw_os_error(libc::EIO)), 0))).collect()
        });
        for (b, (r, ns)) in built.iter().zip(results) {
            self.a.settle(&b.op, ctx, r, ns)?;
        }
        Ok(())
    }

    /// One lane of a `trace`: its lines in order, the gap before each as `compute`.
    fn trace_lane(&mut self, tf: &crate::trace::TraceFile, lane: usize, tables: &TraceTables, base: &crate::trace::OwnedCtx) -> Result<()> {
        let n = base.indices.len();
        let mut idx = base.indices.clone();
        idx.extend_from_slice(&[lane as i64, 0]);
        let mut pos: HashMap<usize, i64> = HashMap::new();
        let (mut begun, mut end) = (false, 0i64);
        for (ord, &i) in tf.lanes[lane].iter().enumerate() {
            let line = &tf.lines[i];
            let gap = if begun { line.t - end } else { line.t };
            begun = true;
            end = line.t + line.dur;
            idx[n + 1] = ord as i64;
            let ctx = base.ctx(&idx);
            self.control(Control::Compute { ns: gap.max(0) }, &ctx)?;
            match &line.op {
                crate::trace::LineOp::Submit { ops } => self.trace_group(tf, line.g0, ops, tables, &ctx, &mut pos)?,
                op => self.trace_op(tf, line.g0, op, tables, &ctx, &mut pos)?,
            }
        }
        Ok(())
    }

    /// A `trace` node: one thread per lane, joined here; the lanes' statistics are this
    /// actor's, their created and removed paths the run's.
    fn trace_node(&mut self, tf: &Arc<crate::trace::TraceFile>, ctx: &OpCtx) -> Result<()> {
        let tables = TraceTables { opens: OpenTable::new(tf), paths: PathOrder::new(tf) };
        let base = crate::trace::OwnedCtx::of(ctx);
        let lanes = tf.lanes();
        if lanes == 0 {
            return Ok(());
        }
        self.a.st.threads += lanes as u64;
        let sh = self.sh.clone();
        let mut children: Vec<Runner> = (0..lanes)
            .map(|_| {
                let mut c = self.child();
                c.a.st.threads = 0; // counted above, as lanes
                c
            })
            .collect();
        let outcome: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = children
                .iter_mut()
                .enumerate()
                .map(|(lane, sink)| {
                    let (tf, tables, base, sh) = (tf.clone(), &tables, &base, sh.clone());
                    std::thread::Builder::new()
                        .name(format!("{}#{} trace lane {lane}", base.template, base.actor))
                        .spawn_scoped(s, move || {
                            let r = sink.trace_lane(&tf, lane, tables, base);
                            if r.is_err() {
                                sh.aborted.store(true, Ordering::Relaxed);
                            }
                            sink.a.finish_shared(&sh);
                            r
                        })
                        .expect("spawn")
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("a trace lane panicked")))).collect()
        });
        // the cause before the lanes that only saw the abort
        let mut errs: Vec<(usize, anyhow::Error)> = Vec::new();
        for (lane, (r, mut sink)) in outcome.into_iter().zip(children).enumerate() {
            self.a.st.merge(&std::mem::take(&mut sink.a.st));
            if let Err(e) = r {
                errs.push((lane, e));
            }
        }
        let cause = errs.iter().position(|(_, e)| !format!("{e:#}").contains("run aborted")).unwrap_or(0);
        match errs.into_iter().nth(cause) {
            Some((lane, e)) => Err(e.context(format!("trace `{}` lane {lane}", tf.name))),
            None => Ok(()),
        }
    }
}
