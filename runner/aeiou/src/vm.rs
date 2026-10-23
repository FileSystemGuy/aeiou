//! The VM: evaluates expressions positionally and walks an actor instance's body, emitting
//! ops to a `Sink`. It holds no state that the abstract does not define: loop frames, `let`
//! bindings, open-file positions, and the as-written byte sums of objects this actor created
//! at this position. A `parallel` or `loader` body is walked once per sub-actor index, in
//! index order; that order is not observable through the sink (the fingerprint is a sum).
//!
//! Definitions the runner fixes here (`schema/README.md` §5 leaves them to it):
//! - a draw at site `s` inside frames with indices `i₁ … iₙ` is `Words(position_key(seed,
//!   actor, s, [i₁ … iₙ]))`; a `choose` is the same with the statement's site;
//! - `consume` position: with the enclosing `loader` (else the innermost loop) as the batch
//!   frame, `b` is the row-major ordinal of the frames down to and including it, `j` and `B`
//!   the ordinal and the product of the frames inside it; an epoch is `N div (G·B)` batches
//!   (`drop_last`); the id is `Perm(N, key(seed, dataset, epoch))(g + G·((b mod epoch_len)·B + j))`;
//! - `x @ i` evaluates `x`'s definition with the frame of `x`'s loop set to `i`, sibling
//!   `let`s of the same body re-evaluated at that index and memoized; below the loop's `from`
//!   the enclosing `cond` takes its other arm;
//! - `until_eof` issues reads of `len` from the position until the computed size is exhausted,
//!   then one more; `readdir` is one op.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ast::Access;
use crate::ast::*;
use crate::eval::*;
use crate::pattern::FieldValue;
use crate::rng::{position_key, Words};

// ---------------------------------------------------------------- ops and sinks

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpKind {
    Open = 1,
    Close,
    Read,
    Write,
    Lseek,
    Ioctl,
    Fstat,
    Stat,
    Fsync,
    Fdatasync,
    Unlink,
    Ftruncate,
    Fallocate,
    Mkdir,
    Rmdir,
    Rename,
    Readdir,
    Fadvise,
}

impl OpKind {
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Open => "open",
            OpKind::Close => "close",
            OpKind::Read => "read",
            OpKind::Write => "write",
            OpKind::Lseek => "lseek",
            OpKind::Ioctl => "ioctl",
            OpKind::Fstat => "fstat",
            OpKind::Stat => "stat",
            OpKind::Fsync => "fsync",
            OpKind::Fdatasync => "fdatasync",
            OpKind::Unlink => "unlink",
            OpKind::Ftruncate => "ftruncate",
            OpKind::Fallocate => "fallocate",
            OpKind::Mkdir => "mkdir",
            OpKind::Rmdir => "rmdir",
            OpKind::Rename => "rename",
            OpKind::Readdir => "readdir",
            OpKind::Fadvise => "fadvise",
        }
    }

    pub const ALL: [OpKind; 18] = [
        OpKind::Open, OpKind::Close, OpKind::Read, OpKind::Write, OpKind::Lseek, OpKind::Ioctl, OpKind::Fstat,
        OpKind::Stat, OpKind::Fsync, OpKind::Fdatasync, OpKind::Unlink, OpKind::Ftruncate, OpKind::Fallocate,
        OpKind::Mkdir, OpKind::Rmdir, OpKind::Rename, OpKind::Readdir, OpKind::Fadvise,
    ];
}

/// One operation as the abstract issues it: POSIX-shaped, backend-independent.
#[derive(Debug, Clone)]
pub struct Op<'c> {
    pub kind: OpKind,
    pub path: &'c str,
    /// `rename`'s destination.
    pub path2: Option<&'c str>,
    /// Effective byte offset (reads, writes, fallocate) or the `lseek` argument.
    pub offset: i64,
    /// Requested length (reads, writes, ftruncate, fallocate).
    pub len: i64,
    /// Bytes the op is expected to transfer: the returned count of a read, `len` of a write
    /// (a `trace` node's write: its traced count, 0 when it failed, which its metrics and the
    /// position it leaves move by; it is still checked as `len` when it runs); for `readdir`,
    /// the entries expected (−1 when not computable).
    pub bytes: i64,
    /// Open flags and mode, `lseek` whence, `ioctl` request, `mkdir` mode.
    pub aux: u64,
    /// The op's `expect` list: errno names the op may fail with (empty: none).
    pub expect: &'c [String],
    /// A read or write that names its offset (`pread`/`pwrite`) rather than using the position.
    pub positioned: bool,
    /// The payload key of the target: the namespace seed for an object, the dataset seed for a
    /// dataset file, 0 otherwise. Not hashed; a backend that writes uses it to generate content.
    pub seed: u64,
    /// A read of an inventory under `--subset` (`DESIGN_REVIEW.md` §3.73 item 5): issued with
    /// length `bytes`, never past the size at the run's count, while `len` (hashed) stays what
    /// the application asks. Not hashed.
    pub capped: bool,
}

impl Op<'_> {
    /// The length a backend issues: `len`, or for a capped read the bytes it is expected to
    /// return (0 for the read that finds the subset's end).
    pub fn issue_len(&self) -> i64 {
        if self.capped { self.bytes } else { self.len }
    }
}

/// Where an op comes from: the actor instance, its position, and the statistics label.
#[derive(Debug, Clone, Copy)]
pub struct OpCtx<'c> {
    pub template: &'c str,
    pub actor: i64,
    pub indices: &'c [i64],
    pub phase: Option<&'c str>,
}

#[derive(Debug, Clone, Copy)]
pub enum Control<'a> {
    Compute { ns: i64 },
    Barrier { scope: &'a str },
    Put { channel: &'a str, seq: i64 },
    Take { channel: &'a str },
    Channel { name: &'a str, capacity: i64, ordered: bool },
}

/// A `parallel` or `loader` node: the sub-actors a sink may run concurrently.
#[derive(Debug, Clone, Copy)]
pub enum ForkKind<'a> {
    Parallel { index: &'a str, width: i64 },
    Loader { name: &'a str, index: &'a str, workers: i64, prefetch: i64, batches: i64, ordered: bool },
}

/// Where `drive` delivers what the VM walks. `dry-run` counts; `run` does the I/O. A driver
/// that multiplexes actors (the `io_uring` event loop) consumes `Vm::next` directly instead.
pub trait Sink<'m, 'a>: Sized {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> Result<()>;
    fn control(&mut self, _c: Control<'a>, _ctx: &OpCtx) -> Result<()> {
        Ok(())
    }
    /// A `parallel` or `loader`. Return `Ok(false)` to have the VM walk the sub-actors inline,
    /// in index order (the dry run). A sink that runs them itself takes a `Snapshot` from the
    /// closure, resumes one `Vm` per sub-actor from it, and calls `start_sub` per index and
    /// drives it to the end; a `parallel` is joined before returning, a `loader` at `finish`.
    fn fork(&mut self, _kind: &ForkKind<'a>, _snapshot: &dyn Fn() -> Snapshot<'m, 'a>) -> Result<bool> {
        Ok(false)
    }
    /// A `trace` node: the captured trace to run at this position (`trace.rs`). The dry run
    /// walks it in line order; the run gives it one thread per lane.
    fn trace(&mut self, t: &Arc<crate::trace::TraceFile>, _ctx: &OpCtx) -> Result<()> {
        bail!("trace `{}`: this driver does not run traces", t.name)
    }
    /// The actor instance (or sub-actor) has walked its whole body.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The per-op hash whose sum over the run is the workload fingerprint. Order-independent by
/// construction (`NAPKIN_MATH.md` §8.D); `phase`, `expect`, and results are not hashed.
pub fn op_hash(op: &Op, ctx: &OpCtx) -> u64 {
    let mut buf = Vec::with_capacity(64 + op.path.len());
    buf.push(op.kind as u8);
    buf.extend_from_slice(&(ctx.actor as u64).to_le_bytes());
    buf.extend_from_slice(&(ctx.indices.len() as u32).to_le_bytes());
    for i in ctx.indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf.extend_from_slice(&op.offset.to_le_bytes());
    buf.extend_from_slice(&op.len.to_le_bytes());
    buf.extend_from_slice(&op.aux.to_le_bytes());
    buf.extend_from_slice(op.path.as_bytes());
    if let Some(p2) = op.path2 {
        buf.push(0);
        buf.extend_from_slice(p2.as_bytes());
    }
    xxhash_rust::xxh3::xxh3_64_with_seed(&buf, 0x0f1e_6e12_a1e7_0000)
}

pub fn flag_bits(flags: &[OpenFlag]) -> u64 {
    flags.iter().fold(0u64, |acc, f| acc | (1 << (*f as u8)))
}

// ---------------------------------------------------------------- the VM

#[derive(Debug, Clone, Copy)]
struct Frame<'a> {
    name: &'a str,
    idx: i64,
    from: i64,
    step: i64,
    iters: i64,
    loader: bool,
}

#[derive(Clone)]
struct Binding<'a> {
    value: Value<'a>,
    body: &'a [Node],
    /// The `let` that bound it: of two `let`s of one name in a body, the one in scope.
    def: &'a ExprLike,
}

#[derive(Clone)]
struct ScopeLevel<'a> {
    bindings: HashMap<&'a str, Binding<'a>>,
    body: &'a [Node],
    depth: usize,
}

/// Entries `Vm::at_cache` holds before it sheds the half with the smaller indices: an `x @ i`
/// chain reaches back from the current index, so the recent entries are the ones that keep
/// its evaluation shallow (a conversation of a thousand turns recomputed from its start
/// overflowed the stack, 2026-10-02).
const AT_CACHE_MAX: usize = 1 << 16;

/// `Vm::at_cache`'s key: the definition, the index, and the indices of the enclosing loops.
type AtKey = (usize, i64, Vec<i64>);

/// The values of `x @ i` already evaluated. A full cache sheds half its entries, those with
/// the smaller indices, and of entries with the same index the ones inserted first; the
/// half is found by selection, so an insert costs O(1) amortized however the indices are
/// spread. (Entries with the index at the cut were all kept, so when most indices were
/// equal, an `at` in a short loop nested in a long one, nothing was shed: every insert then
/// sorted the whole cache, and it grew without bound; RC1 review.)
#[derive(Default)]
struct AtCache<'a> {
    map: HashMap<AtKey, (Value<'a>, u64)>,
    /// Inserts so far: an entry's order among those of its index.
    seq: u64,
}

impl<'a> AtCache<'a> {
    fn get(&self, k: &AtKey) -> Option<Value<'a>> {
        self.map.get(k).map(|(v, _)| v.clone())
    }

    fn insert(&mut self, k: AtKey, v: Value<'a>) {
        if self.map.len() >= AT_CACHE_MAX {
            let mut order: Vec<(i64, u64)> = self.map.iter().map(|(k, (_, s))| (k.1, *s)).collect();
            let half = order.len() / 2;
            let (_, &mut cut, _) = order.select_nth_unstable(half);
            self.map.retain(|k, (_, s)| (k.1, *s) >= cut);
        }
        self.seq += 1;
        self.map.insert(k, (v, self.seq));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

/// An active `x @ i` evaluation: frame `frame` reads as `idx`, sibling lets of `body` are
/// re-evaluated at that index and memoized. Names resolve as they do where the definition
/// is written: a `ref` to the nearest `let` of `body` before `pos` (the definition being
/// evaluated), else to a binding of the scopes below `level` (the scope of `body`), never to
/// one of a body nested inside it that the `at` happened to be evaluated from.
struct Shift<'a> {
    frame: usize,
    idx: i64,
    body: &'a [Node],
    level: usize,
    pos: usize,
    /// By position in `body`.
    memo: HashMap<usize, Value<'a>>,
}

#[derive(Debug, Clone)]
struct FileState {
    pos: i64,
    size: Option<i64>,
    as_written: bool,
    capped: bool,
}

/// A handle resolved to something an op can name.
#[derive(Debug, Clone)]
struct Target {
    path: Arc<str>,
    size: Option<i64>,
    as_written: bool,
    seed: u64,
    /// For a dataset directory: the entries a `readdir` is expected to return, when computable.
    entries: Option<i64>,
    /// An inventory read under `--subset`: reads are issued no further than `size`.
    capped: bool,
}

/// What a sub-actor starts from: the parent's position (frames, with the fork's frame last),
/// its visible bindings, open files, as-written sums, and the sub-actor body. Cloned once per
/// sub-actor thread by a forking sink.
#[derive(Clone)]
pub struct Snapshot<'m, 'a> {
    model: &'m Model<'a>,
    template: &'a str,
    actor: i64,
    actor_count: i64,
    frames: Vec<Frame<'a>>,
    scopes: Vec<ScopeLevel<'a>>,
    phases: Vec<Arc<str>>,
    open: HashMap<Arc<str>, FileState>,
    written: HashMap<Arc<str>, i64>,
    body: &'a [Node],
}

enum EvalError {
    BelowFrom,
    Other(anyhow::Error),
}

impl From<anyhow::Error> for EvalError {
    fn from(e: anyhow::Error) -> Self {
        EvalError::Other(e)
    }
}

type ER<T> = std::result::Result<T, EvalError>;

fn other<T>(msg: impl Into<String>) -> ER<T> {
    Err(EvalError::Other(anyhow!(msg.into())))
}

/// What the walk hands its driver next. An `Op` or a `Control` comes with the position it was
/// issued from; a `Fork` must be answered before the next call: `snapshot` and `accept_fork`
/// to run the sub-actors elsewhere, or `inline_fork` to walk them here in index order.
pub enum Event<'e, 'a> {
    Op(Op<'e>, OpCtx<'e>),
    Control(Control<'a>, OpCtx<'e>),
    Fork(ForkKind<'a>),
    /// A `trace` node at this position.
    Trace(Arc<crate::trace::TraceFile>, OpCtx<'e>),
}

/// `Event` before the borrows are taken: what `step` found.
enum Next<'a> {
    Op,
    Control(Control<'a>),
    Fork(ForkKind<'a>),
    Trace(Arc<crate::trace::TraceFile>),
}

/// The op the last `Next::Op` describes; `next` lends it out as an `Op`. A `path` of `None`
/// means the op belongs to the `IoSeq` on top of the stack and names its file.
struct Pending<'a> {
    kind: OpKind,
    path: Option<Arc<str>>,
    path2: Option<Arc<str>>,
    offset: i64,
    len: i64,
    bytes: i64,
    aux: u64,
    expect: &'a [String],
    positioned: bool,
    seed: u64,
    capped: bool,
}

/// How many ops a `read`/`write` statement still has to issue.
enum Left {
    Count(i64),
    UntilEof { size: i64, tail_done: bool },
}

/// A `read` or `write` statement in progress: one op per `step` until `left` is exhausted,
/// then the file position is updated (a positioned read or write leaves it alone).
struct IoSeq<'a> {
    kind: OpKind,
    path: Arc<str>,
    seed: u64,
    off: i64,
    len: i64,
    /// The size the statement started with (reads: computes the expected count).
    size: Option<i64>,
    positioned: bool,
    expect: &'a [String],
    as_written: bool,
    capped: bool,
    left: Left,
}

/// The walk's explicit stack: what the recursive walk kept on the thread's stack, so the walk
/// can stop between any two events and continue later (`NAPKIN_MATH.md` §4.2: an event loop
/// parks an actor at every blocking call).
enum Cont<'a> {
    /// A body being walked (its `ScopeLevel` is pushed with it and popped when it ends).
    Body { nodes: &'a [Node], i: usize },
    /// A loop whose frame is the last of `frames`: iteration `k` of `iters` is next.
    Loop { body: &'a [Node], from: i64, step: i64, iters: i64, k: i64 },
    /// A fork walked inline (`inline_fork`): a loop over the sub-actors in which every
    /// sub-actor starts from the parent's open files and as-written sums at the fork, and the
    /// parent gets them back unchanged when the last has ended, as a sub-actor's own
    /// `Snapshot` gives it under `run` (`runner/REFERENCE.md` §4, Threads).
    Fork { body: &'a [Node], iters: i64, k: i64, open: HashMap<Arc<str>, FileState>, written: HashMap<Arc<str>, i64> },
    /// Pop the phase label when the phase body ends.
    PhaseEnd,
    Io(IoSeq<'a>),
}

/// A `parallel` or `loader` the driver has not answered yet.
struct PendingFork<'a> {
    index: &'a str,
    width: i64,
    loader: bool,
    body: &'a [Node],
}

pub struct Vm<'m, 'a> {
    pub model: &'m Model<'a>,
    template: &'a str,
    actor: i64,
    actor_count: i64,
    frames: Vec<Frame<'a>>,
    scopes: Vec<ScopeLevel<'a>>,
    shifts: Vec<Shift<'a>>,
    /// Values of `x @ i` already evaluated, by definition, index, and the indices of the
    /// enclosing loops. `x @ i` is a pure function of those (and of this actor), so this
    /// stores nothing the definition does not say; it makes a chain that steps back one
    /// index per iteration (`x @ k-1`) one link per iteration instead of a walk to the
    /// start, and keeps the recursion as deep as one link, not as the loop is long.
    at_cache: AtCache<'a>,
    /// Frames visible to the expression being evaluated (shorter during a shifted evaluation).
    depth: usize,
    fields: Option<HashMap<&'a str, Value<'a>>>,
    open: HashMap<Arc<str>, FileState>,
    written: HashMap<Arc<str>, i64>,
    phases: Vec<Arc<str>>,
    indices_buf: Vec<i64>,
    /// The sub-actor body of a resumed VM.
    sub_body: Option<&'a [Node]>,
    /// Unit lengths of recently addressed container files (`GRAMMAR_OPTIONS.md` §6.3: the
    /// layout is computed once per open file and held). Bounded; never shared.
    unit_lens: HashMap<(usize, i64), Arc<Vec<i64>>>,
    /// Sizes and sample offsets of the container files this actor addressed last, most recent
    /// last (`DsMeta::file_geom`): a file's samples' sizes are drawn once while it is read, not
    /// once per op. Bounded by `GEOM_FILES` files and `GEOM_OFFSETS` offset anchors, never a
    /// table over the dataset; never shared.
    geom: Vec<((usize, i64), Arc<FileGeom>)>,
    stack: Vec<Cont<'a>>,
    pending: Pending<'a>,
    pending_fork: Option<PendingFork<'a>>,
}

/// The bounds of a VM's file geometry cache (`Vm::geom`): the files an actor interleaves
/// (a `cycle` of shards, the files of a batch), and 8 MiB of sample offsets. A file of more
/// samples than `GEOM_OFFSETS` holds an anchor every k = ceil(n / `GEOM_OFFSETS`) samples or
/// so, and an offset draws at most k-1 sizes from its anchor (`DsMeta::offset_from_geom`).
const GEOM_FILES: usize = 16;
const GEOM_OFFSETS: usize = 1 << 20;

/// The offset anchors `geom` holds.
fn geom_held(geom: &[((usize, i64), Arc<FileGeom>)]) -> usize {
    geom.iter().map(|(_, g)| g.anchors.as_ref().map_or(0, |o| o.len())).sum()
}

impl<'m, 'a: 'm> Vm<'m, 'a> {
    pub fn new(model: &'m Model<'a>, template: &'a str, actor: i64, actor_count: i64) -> Self {
        Vm {
            model,
            template,
            actor,
            actor_count,
            frames: Vec::new(),
            scopes: vec![ScopeLevel { bindings: HashMap::new(), body: &[], depth: 0 }],
            shifts: Vec::new(),
            at_cache: AtCache::default(),
            depth: 0,
            fields: None,
            open: HashMap::new(),
            written: HashMap::new(),
            phases: Vec::new(),
            indices_buf: Vec::new(),
            unit_lens: HashMap::new(),
            geom: Vec::new(),
            sub_body: None,
            stack: Vec::new(),
            pending: Pending::empty(),
            pending_fork: None,
        }
    }

    /// A sub-actor VM: the parent's state at the fork. `start_sub` begins one sub-actor.
    pub fn resume(snap: Snapshot<'m, 'a>) -> Self {
        let depth = snap.frames.len();
        Vm {
            model: snap.model,
            template: snap.template,
            actor: snap.actor,
            actor_count: snap.actor_count,
            frames: snap.frames,
            scopes: snap.scopes,
            shifts: Vec::new(),
            at_cache: AtCache::default(),
            depth,
            fields: None,
            open: snap.open,
            written: snap.written,
            phases: snap.phases,
            indices_buf: Vec::new(),
            unit_lens: HashMap::new(),
            geom: Vec::new(),
            sub_body: Some(snap.body),
            stack: Vec::new(),
            pending: Pending::empty(),
            pending_fork: None,
        }
    }

    /// `resume` into a VM that has already run a sub-actor: the same state, with this VM's
    /// allocations kept (a pool thread resumes once per fork, `run.rs`).
    pub fn resume_from(&mut self, snap: &Snapshot<'m, 'a>) {
        self.model = snap.model;
        self.template = snap.template;
        self.actor = snap.actor;
        self.actor_count = snap.actor_count;
        self.frames.clone_from(&snap.frames);
        self.scopes.truncate(snap.scopes.len());
        for (i, s) in snap.scopes.iter().enumerate() {
            match self.scopes.get_mut(i) {
                Some(mine) => {
                    mine.bindings.clone_from(&s.bindings);
                    mine.body = s.body;
                    mine.depth = s.depth;
                }
                None => self.scopes.push(s.clone()),
            }
        }
        self.shifts.clear();
        self.depth = snap.frames.len();
        self.fields = None;
        self.open.clone_from(&snap.open);
        self.written.clone_from(&snap.written);
        self.phases.clone_from(&snap.phases);
        self.indices_buf.clear();
        self.unit_lens.clear();
        self.geom.clear();
        self.sub_body = Some(snap.body);
        self.stack.clear();
        self.pending = Pending::empty();
        self.pending_fork = None;
    }

    #[allow(clippy::too_many_arguments)]
    fn snapshot_of(model: &'m Model<'a>, template: &'a str, actor: i64, actor_count: i64, frames: &[Frame<'a>], scopes: &[ScopeLevel<'a>], phases: &[Arc<str>], open: &HashMap<Arc<str>, FileState>, written: &HashMap<Arc<str>, i64>, body: &'a [Node]) -> Snapshot<'m, 'a> {
        Snapshot {
            model,
            template,
            actor,
            actor_count,
            frames: frames.to_vec(),
            scopes: scopes.to_vec(),
            phases: phases.to_vec(),
            open: open.clone(),
            written: written.clone(),
            body,
        }
    }

    /// Begin walking an actor instance's body; `next` then yields its events.
    pub fn start(&mut self, body: &'a [Node]) {
        self.push_body(body);
    }

    /// On a resumed VM: begin the sub-actor body at index `k` of the fork (ordinal, so the
    /// frame's index is `from + k·step`). Call once the previous sub-actor's walk has ended.
    pub fn start_sub(&mut self, k: i64) {
        let body = self.sub_body.expect("start_sub on a VM that was not resumed from a snapshot");
        let f = self.frames.last_mut().expect("a resumed VM has the fork frame");
        // wrapping is exact: the index fits (it is below the loop's `to`), only k·step may not
        f.idx = f.from.wrapping_add(k.wrapping_mul(f.step));
        self.push_body(body);
    }

    /// The next event, or `None` when the body started by `start` / `start_sub` has ended.
    /// Between two calls the VM holds no borrowed state, so a driver may hold the walk at any
    /// event for as long as it likes (an outstanding I/O, a channel, a barrier, a timer).
    pub fn next(&mut self) -> Result<Option<Event<'_, 'a>>> {
        let n = match self.step() {
            Ok(n) => n,
            Err(e) => return Err(self.wrap(e)),
        };
        Ok(match n {
            None => None,
            Some(Next::Op) => {
                let (op, ctx) = self.current();
                Some(Event::Op(op, ctx))
            }
            Some(Next::Control(c)) => Some(Event::Control(c, self.ctx())),
            Some(Next::Fork(k)) => Some(Event::Fork(k)),
            Some(Next::Trace(t)) => Some(Event::Trace(t, self.ctx())),
        })
    }

    /// The op of the last `Event::Op`, again: for a driver that parked the walk at it and now
    /// has its result.
    pub fn current(&self) -> (Op<'_>, OpCtx<'_>) {
        let p = &self.pending;
        let path: &str = match &p.path {
            Some(path) => path,
            None => match self.stack.last() {
                Some(Cont::Io(io)) => &io.path,
                _ => "",
            },
        };
        (Op { kind: p.kind, path, path2: p.path2.as_deref(), offset: p.offset, len: p.len, bytes: p.bytes, aux: p.aux, expect: p.expect, positioned: p.positioned, seed: p.seed, capped: p.capped }, self.ctx())
    }

    fn ctx(&self) -> OpCtx<'_> {
        OpCtx { template: self.template, actor: self.actor, indices: &self.indices_buf, phase: self.phases.last().map(|p| &**p) }
    }

    /// After an `Event::Fork`: the parent's position for the sub-actors (frames with the fork's
    /// frame last, visible bindings, phases, open files, as-written sums, the sub-actor body).
    pub fn snapshot(&self) -> Snapshot<'m, 'a> {
        let pf = self.pending_fork.as_ref().expect("snapshot without a pending fork");
        let mut fr = self.frames.clone();
        fr.push(Frame { name: pf.index, idx: 0, from: 0, step: 1, iters: pf.width, loader: pf.loader });
        Self::snapshot_of(self.model, self.template, self.actor, self.actor_count, &fr, &self.scopes, &self.phases, &self.open, &self.written, pf.body)
    }

    /// After an `Event::Fork`: the driver runs the sub-actors itself; the walk continues after
    /// the node.
    pub fn accept_fork(&mut self) {
        self.pending_fork.take().expect("accept_fork without a pending fork");
    }

    /// After an `Event::Fork`: walk the sub-actors here, in index order (the dry run). Each
    /// sub-actor sees the parent's file state at the fork, never an earlier sibling's, and the
    /// parent's is restored after the last: what `snapshot`/`resume` give the drivers of `run`.
    pub fn inline_fork(&mut self) {
        let pf = self.pending_fork.take().expect("inline_fork without a pending fork");
        self.frames.push(Frame { name: pf.index, idx: 0, from: 0, step: 1, iters: pf.width, loader: pf.loader });
        self.depth = self.frames.len();
        let (open, written) = (self.open.clone(), self.written.clone());
        self.stack.push(Cont::Fork { body: pf.body, iters: pf.width, k: 0, open, written });
    }

    fn push_body(&mut self, nodes: &'a [Node]) {
        self.scopes.push(ScopeLevel { bindings: HashMap::new(), body: nodes, depth: self.frames.len() });
        self.stack.push(Cont::Body { nodes, i: 0 });
    }

    fn push_loop(&mut self, index: &'a str, from: i64, step: i64, iters: i64, loader: bool, body: &'a [Node]) {
        self.frames.push(Frame { name: index, idx: from, from, step, iters, loader });
        self.depth = self.frames.len();
        self.stack.push(Cont::Loop { body, from, step, iters, k: 0 });
    }
    fn wrap(&self, e: EvalError) -> anyhow::Error {
        let e = match e {
            EvalError::BelowFrom => anyhow!("`at` below the loop's start with no alternative arm"),
            EvalError::Other(e) => e,
        };
        let pos: Vec<String> = self.frames.iter().map(|f| format!("{}={}", f.name, f.idx)).collect();
        e.context(format!("actor {}#{} at [{}]", self.template, self.actor, pos.join(", ")))
    }

    // ---- positions ----

    fn frame_index(&self, f: usize) -> i64 {
        for s in self.shifts.iter().rev() {
            if s.frame == f {
                return s.idx;
            }
        }
        self.frames[f].idx
    }

    /// `None` past 2^63-1 (a loop spanning more than 2^63 at an index `x @ i` names).
    fn frame_ordinal(&self, f: usize) -> Option<i64> {
        let fr = &self.frames[f];
        let i = self.frame_index(f);
        match i.checked_sub(fr.from) {
            Some(d) => Some(d.div_euclid(fr.step)),
            None => i64::try_from((i as i128 - fr.from as i128).div_euclid(fr.step as i128)).ok(),
        }
    }

    fn indices(&self, depth: usize) -> Vec<i64> {
        (0..depth).map(|f| self.frame_index(f)).collect()
    }

    fn draw_key(&self, site: u64) -> u64 {
        let idx = self.indices(self.depth);
        position_key(self.model.cfg.seed, self.actor as u64, site, &idx)
    }

    // ---- public evaluation entry points (used by the model builder) ----

    pub fn eval_int(&mut self, e: &'a Expr) -> Result<i64> {
        self.expr(e).map_err(|e| self.wrap(e))?.as_int()
    }

    pub fn resolve_distref(&mut self, d: &'a DistRef) -> Result<RDist<'a>> {
        self.distref(d).map_err(|e| self.wrap(e))
    }

    pub fn resolve_dist(&mut self, d: &'a Dist) -> Result<RDist<'a>> {
        self.dist(d).map_err(|e| self.wrap(e))
    }

    // ---- expressions ----

    fn expr(&mut self, e: &'a Expr) -> ER<Value<'a>> {
        let node = match e {
            Expr::Lit(l) => return Ok(Value::from_literal(l)),
            Expr::Node(n) => n.as_ref(),
        };
        Ok(match node {
            ExprNode::Param(name) => self.param(name)?,
            ExprNode::Index(name) => self.index(name)?,
            ExprNode::Ref(name) => self.reference(name)?,
            ExprNode::At { name, index } => {
                let i = self.expr(index)?.as_int()?;
                self.at(name, i)?
            }
            ExprNode::Actor(ActorAttr::Id) => Value::Int(self.actor),
            ExprNode::Actor(ActorAttr::Count) => Value::Int(self.actor_count),
            ExprNode::Draw(d) => {
                let rd = self.distref(&d.dist)?;
                let mut w = Words::new(self.draw_key(d.site.get()));
                sample(&rd, &mut w, None)?
            }
            ExprNode::Elem { array, index } => {
                let i = self.expr(index)?.as_int()?;
                let a = self.array(array, "elem")?;
                if i < 0 || i as usize >= a.len() {
                    return other(format!("elem: index {i} outside parameter array `{array}` of {}", a.len()));
                }
                Value::from_pvalue(&a[i as usize])
            }
            ExprNode::Len(array) => Value::Int(self.array(array, "len")?.len() as i64),
            ExprNode::Sum(array) => {
                let mut int = 0i64;
                let mut float = 0f64;
                let mut any_float = false;
                for x in self.array(array, "sum")? {
                    match Value::from_pvalue(x) {
                        Value::Int(n) => int = int.checked_add(n).ok_or_else(|| anyhow!("sum: the integers of `{array}` pass 2^63-1"))?,
                        Value::Float(f) => {
                            any_float = true;
                            float += f;
                        }
                        // rule V27 refuses a `sum` over a parameter whose elements are not numbers
                        v => return other(format!("sum: element of `{array}` is {}", v.kind())),
                    }
                }
                if any_float { Value::Float(float + int as f64) } else { Value::Int(int) }
            }
            ExprNode::Size(h) => {
                let hv = self.handle(h)?;
                Value::Int(self.size_of(&hv)?)
            }
            ExprNode::Offset(h) => {
                let hv = self.handle(h)?;
                Value::Int(self.offset_of(&hv)?)
            }
            ExprNode::UnitIndex(h) => {
                let hv = self.handle(h)?;
                match hv {
                    HVal::Sample { ds, id } => Value::Int(self.model.datasets[ds].unit_of_sample(id)?),
                    HVal::Unit { u, .. } | HVal::Column { u, .. } => Value::Int(u),
                    _ => return other("unit_index: not a sample, unit, or column handle"),
                }
            }
            ExprNode::Units(h) => {
                let hv = self.handle(h)?;
                let (ds, file) = match hv {
                    HVal::Sample { ds, id } => (ds, self.container_of(ds, id)),
                    HVal::File { ds, file } | HVal::Unit { ds, file, .. } | HVal::Column { ds, file, .. } => (ds, file),
                    _ => return other("units: not a file, sample, or unit handle"),
                };
                Value::Int(self.model.datasets[ds].units_in_file(file)?)
            }
            ExprNode::Chunks(h) => {
                let hv = self.handle(h)?;
                let (ds, file) = match hv {
                    HVal::Sample { ds, id } => (ds, self.container_of(ds, id)),
                    HVal::File { ds, file } => (ds, file),
                    _ => return other("chunks: not a file or sample handle"),
                };
                Value::Int(self.model.datasets[ds].chunks(file)?)
            }
            ExprNode::Count(name) => Value::Int(self.model.datasets[self.model.dataset(name)?].count()),
            ExprNode::Dirs(name) => match &self.model.datasets[self.model.dataset(name)?] {
                DsMeta::Files { dirs: DirScheme::Unsupported, .. } => return other(format!("dirs: dataset `{name}` has several `id` fields in its directory part")),
                DsMeta::Files { ndirs, .. } => Value::Int(*ndirs),
                _ => return other(format!("dirs: `{name}` is not a files dataset")),
            },
            ExprNode::Add(a) => self.arith(a, |x, y| x.checked_add(y), |x, y| x + y, "add")?,
            ExprNode::Sub(a) => self.arith(a, |x, y| x.checked_sub(y), |x, y| x - y, "sub")?,
            ExprNode::Mul(a) => self.arith(a, |x, y| x.checked_mul(y), |x, y| x * y, "mul")?,
            ExprNode::Div(a) => self.arith(a, |x, y| if y == 0 { None } else { x.checked_div_euclid(y) }, |x, y| (x / y).floor(), "div")?,
            ExprNode::Mod(a) => self.arith(a, |x, y| if y == 0 { None } else { x.checked_rem_euclid(y) }, |x, y| x.rem_euclid(y), "mod")?,
            ExprNode::CeilDiv(a) => self.arith(a, |x, y| if y == 0 { None } else { x.checked_neg().and_then(|n| n.checked_div_euclid(y)).and_then(i64::checked_neg) }, |x, y| (x / y).ceil(), "ceil_div")?,
            ExprNode::Min(a) => self.arith(a, |x, y| Some(x.min(y)), f64::min, "min")?,
            ExprNode::Max(a) => self.arith(a, |x, y| Some(x.max(y)), f64::max, "max")?,
            ExprNode::Neg(x) => match self.expr(x)? {
                Value::Int(n) => Value::Int(n.checked_neg().ok_or_else(|| anyhow!("neg overflow"))?),
                Value::Float(f) => Value::Float(-f),
                v => return other(format!("neg of {}", v.kind())),
            },
            ExprNode::Eq(a) => Value::Bool(self.compare(a, "eq")? == Some(std::cmp::Ordering::Equal)),
            ExprNode::Ne(a) => Value::Bool(self.compare(a, "ne")? != Some(std::cmp::Ordering::Equal)),
            ExprNode::Lt(a) => Value::Bool(self.compare(a, "lt")? == Some(std::cmp::Ordering::Less)),
            ExprNode::Le(a) => Value::Bool(matches!(self.compare(a, "le")?, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal))),
            ExprNode::Gt(a) => Value::Bool(self.compare(a, "gt")? == Some(std::cmp::Ordering::Greater)),
            ExprNode::Ge(a) => Value::Bool(matches!(self.compare(a, "ge")?, Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))),
            ExprNode::And(a) => {
                for x in a {
                    if !self.expr(x)?.as_bool()? {
                        return Ok(Value::Bool(false));
                    }
                }
                Value::Bool(true)
            }
            ExprNode::Or(a) => {
                for x in a {
                    if self.expr(x)?.as_bool()? {
                        return Ok(Value::Bool(true));
                    }
                }
                Value::Bool(false)
            }
            ExprNode::Not(x) => Value::Bool(!self.expr(x)?.as_bool()?),
            ExprNode::Cond(c) => {
                let test = self.expr(&c.test)?.as_bool()?;
                let (first, second) = if test { (&c.then, &c.otherwise) } else { (&c.otherwise, &c.then) };
                match self.exprlike(first) {
                    Err(EvalError::BelowFrom) => self.exprlike(second)?,
                    r => r?,
                }
            }
        })
    }

    fn arith(&mut self, a: &'a [Expr; 2], int: impl Fn(i64, i64) -> Option<i64>, float: impl Fn(f64, f64) -> f64, name: &str) -> ER<Value<'a>> {
        let x = self.expr(&a[0])?;
        let y = self.expr(&a[1])?;
        match (&x, &y) {
            (Value::Int(x), Value::Int(y)) => match int(*x, *y) {
                Some(v) => Ok(Value::Int(v)),
                None => other(format!("{name}: overflow or division by zero ({x}, {y})")),
            },
            (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => Ok(Value::Float(float(x.as_f64()?, y.as_f64()?))),
            _ => other(format!("{name}: operands are {} and {}", x.kind(), y.kind())),
        }
    }

    /// `None` when the values are of different kinds or one is `none` (then only `eq`/`ne` are
    /// meaningful and they compare as unequal).
    fn compare(&mut self, a: &'a [Expr; 2], name: &str) -> ER<Option<std::cmp::Ordering>> {
        let x = self.expr(&a[0])?;
        let y = self.expr(&a[1])?;
        Ok(match (&x, &y) {
            (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
            (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
            (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
            (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
            (Value::None, Value::None) => Some(std::cmp::Ordering::Equal),
            (Value::None, _) | (_, Value::None) => None,
            (Value::Handle(x), Value::Handle(y)) => {
                if x == y { Some(std::cmp::Ordering::Equal) } else { None }
            }
            _ => return other(format!("{name}: cannot compare {} with {}", x.kind(), y.kind())),
        })
    }

    fn exprlike(&mut self, v: &'a ExprLike) -> ER<Value<'a>> {
        match v {
            ExprLike::Expr(e) => self.expr(e),
            ExprLike::Dist(d) => Ok(Value::Dist(d)),
            ExprLike::Handle(h) => Ok(Value::Handle(self.handle(h)?)),
        }
    }

    /// A declared parameter's value in effect (`gpus` is not one: V2 reserves it, and both
    /// checkers refuse `{param: gpus}` as unknown; `{actor: count}` is the instance count).
    fn param(&self, name: &str) -> ER<Value<'a>> {
        let params: &'a Params = self.model.params;
        match params.values.get(name) {
            Some(v) => Ok(Value::from_pvalue(v)),
            None => other(format!("unknown param `{name}`")),
        }
    }

    /// The elements of the parameter array `op` (`elem`, `len`, `sum`) reads; rule V27 refuses
    /// one that is not an array, by its value in effect.
    fn array(&self, name: &str, op: &str) -> ER<&'a [PValue]> {
        match self.param(name)? {
            Value::Array(a) => Ok(a),
            _ => other(format!("{op}: parameter `{name}` is not an array")),
        }
    }

    fn index(&self, name: &str) -> ER<Value<'a>> {
        if let Some(fields) = &self.fields {
            if let Some(v) = fields.get(name) {
                return Ok(v.clone());
            }
        }
        for f in (0..self.depth).rev() {
            if self.frames[f].name == name {
                return Ok(Value::Int(self.frame_index(f)));
            }
        }
        other(format!("index `{name}` not in scope"))
    }

    /// The position and definition of the first `let` of `name` in `body`.
    fn find_let(body: &'a [Node], name: &str) -> Option<(usize, &'a ExprLike)> {
        body.iter().enumerate().find_map(|(p, n)| match n {
            Node::Let { name: n, value } if n == name => Some((p, value)),
            _ => None,
        })
    }

    /// The position and definition of the last `let` of `name` in `body` before position `pos`:
    /// the one a `ref` written at `pos` reads.
    fn let_before(body: &'a [Node], pos: usize, name: &str) -> Option<(usize, &'a ExprLike)> {
        body[..pos.min(body.len())].iter().enumerate().rev().find_map(|(p, n)| match n {
            Node::Let { name: n, value } if n == name => Some((p, value)),
            _ => None,
        })
    }

    fn reference(&mut self, name: &str) -> ER<Value<'a>> {
        if let Some(fields) = &self.fields {
            if let Some(v) = fields.get(name) {
                return Ok(v.clone());
            }
        }
        // inside `x @ i`: the name resolves where x's definition is written. A sibling let
        // before it takes its value at the shifted index; else a binding of a scope enclosing
        // x's body, which the shift does not move.
        let below = match self.shifts.last() {
            None => self.scopes.len(),
            Some(top) => {
                let (body, pos) = (top.body, top.pos);
                if let Some((q, def)) = Self::let_before(body, pos, name) {
                    if let Some(v) = top.memo.get(&q) {
                        return Ok(v.clone());
                    }
                    // the sibling's own names resolve where it is written
                    self.shifts.last_mut().unwrap().pos = q;
                    let v = self.exprlike(def);
                    let top = self.shifts.last_mut().unwrap();
                    top.pos = pos;
                    let v = v?;
                    top.memo.insert(q, v.clone());
                    return Ok(v);
                }
                top.level
            }
        };
        for level in self.scopes[..below].iter().rev() {
            if let Some(b) = level.bindings.get(name) {
                return Ok(b.value.clone());
            }
        }
        other(format!("binding `{name}` not in scope (a forward reference needs `at`)"))
    }

    /// `x @ i`: the value `x` takes at index `i` of its loop. `x` is the binding the checkers
    /// resolve (rule V1, V3): a `let` of the body the `at` is written in, before or after it,
    /// else the binding in scope from an enclosing body; never an outer binding of the same
    /// name that the body shadows.
    fn at(&mut self, name: &str, i: i64) -> ER<Value<'a>> {
        // the body the `at` is written in: the definition being evaluated under a shift, else the running body
        let here = match self.shifts.last() {
            Some(top) => top.level,
            None => self.scopes.len() - 1,
        };
        let here_body = match self.shifts.last() {
            Some(top) => top.body,
            None => self.scopes[here].body,
        };
        let (level, pos, def) = if let Some((p, def)) = Self::find_let(here_body, name) {
            (here, p, def)
        } else if let Some((l, b)) = self.scopes[..here].iter().enumerate().rev().find_map(|(l, s)| s.bindings.get(name).map(|b| (l, b))) {
            let p = b.body.iter().position(|n| matches!(n, Node::Let { value, .. } if std::ptr::eq(value, b.def))).expect("a binding's let is in its body");
            (l, p, b.def)
        } else {
            return other(format!("`at` names unknown binding `{name}`"));
        };
        let depth = self.scopes[level].depth;
        let body = self.scopes[level].body;
        if depth == 0 {
            return other(format!("`{name} @`: binding is outside any loop"));
        }
        let frame = depth - 1;
        if i < self.frames[frame].from {
            return Err(EvalError::BelowFrom);
        }
        // a namespace's field values are not part of the key, so nothing is kept under them
        let key = if self.fields.is_none() { Some((def as *const ExprLike as usize, i, (0..frame).map(|f| self.frame_index(f)).collect::<Vec<i64>>())) } else { None };
        if let Some(v) = key.as_ref().and_then(|k| self.at_cache.get(k)) {
            return Ok(v);
        }
        // Rule V3 says an `at` inside a loop body's own bindings goes to a strictly smaller index.
        // The checkers prove it; this is the net under a proof they got wrong: a body already
        // being evaluated at index j may only be asked for an index below j, or the recursion
        // never ends (it ended in a stack overflow).
        if let Some(s) = self.shifts.iter().rev().find(|s| s.frame == frame && std::ptr::eq(s.body, body)) {
            if i >= s.idx {
                return other(format!("`{name} @ {i}`: index not below {}, the index of its loop body already being evaluated (rule V3: an `at` on a binding of the same loop body must go to a smaller index)", s.idx));
            }
        }
        let saved_depth = self.depth;
        self.shifts.push(Shift { frame, idx: i, body, level, pos, memo: HashMap::new() });
        self.depth = depth;
        let r = self.exprlike(def);
        self.depth = saved_depth;
        self.shifts.pop();
        if let (Some(k), Ok(v)) = (key, &r) {
            self.at_cache.insert(k, v.clone());
        }
        r
    }

    // ---- distributions ----

    fn distref(&mut self, d: &'a DistRef) -> ER<RDist<'a>> {
        match d {
            DistRef::Dist(d) => self.dist(d),
            DistRef::Expr(e) => match self.expr(e)? {
                Value::Dist(d) => {
                    if let Some(r) = self.model.param_dists.get(&(d as *const Dist as usize)) {
                        return Ok(r.clone());
                    }
                    self.dist(d)
                }
                v => Ok(RDist::Const(v)),
            },
        }
    }

    fn dist(&mut self, d: &'a Dist) -> ER<RDist<'a>> {
        Ok(match d {
            Dist::Const(e) => RDist::Const(self.expr(e)?),
            Dist::Uniform { lo, hi } => RDist::Uniform { lo: self.expr(lo)?.as_int()?, hi: self.expr(hi)?.as_int()? },
            Dist::Uniform64 {} => RDist::Uniform64,
            Dist::Normal { mean, sd, min, max } => RDist::Normal {
                mean: self.expr(mean)?.as_f64()?,
                sd: self.expr(sd)?.as_f64()?,
                min: self.opt_f64(min)?,
                max: self.opt_f64(max)?,
            },
            Dist::Lognormal { median, sigma, min, max } => RDist::Lognormal {
                median: self.expr(median)?.as_f64()?,
                sigma: *sigma,
                min: self.opt_f64(min)?,
                max: self.opt_f64(max)?,
            },
            Dist::Empirical { values, weights } => {
                let mut cum = Vec::with_capacity(weights.len());
                let mut acc = 0.0;
                for w in weights {
                    acc += w;
                    cum.push(acc);
                }
                RDist::Empirical { values: values.iter().map(Value::from_literal).collect(), cum }
            }
            Dist::Zipf { s } => RDist::Zipf { s: *s },
            Dist::Hotset { fraction, weight } => RDist::Hotset { fraction: *fraction, weight: *weight },
            Dist::Mixture(arms) => {
                let mut out = Vec::with_capacity(arms.len());
                let mut total = 0.0;
                for arm in arms {
                    total += arm.weight;
                    let d = match &arm.dist {
                        None => None,
                        Some(d) => Some(self.distref(d)?),
                    };
                    out.push((arm.weight, d));
                }
                RDist::Mixture { arms: out, total }
            }
        })
    }

    fn opt_f64(&mut self, e: &'a Option<Box<Expr>>) -> ER<Option<f64>> {
        match e {
            None => Ok(None),
            Some(e) => Ok(Some(self.expr(e)?.as_f64()?)),
        }
    }

    // ---- handles ----

    fn container_of(&self, ds: usize, id: i64) -> i64 {
        match &self.model.datasets[ds] {
            DsMeta::Files { spf, .. } => id / spf,
            DsMeta::Array { .. } => 0,
        }
    }

    fn handle(&mut self, h: &'a Handle) -> ER<HVal> {
        Ok(match h {
            Handle::Ref(name) => match self.reference(name)? {
                Value::Handle(h) => h,
                v => return other(format!("`{name}` is a {}, not a handle", v.kind())),
            },
            Handle::File(f) => match (&f.dataset, &f.id, &f.of) {
                (Some(ds), Some(id), _) => {
                    let ds = self.model.dataset(ds)?;
                    let file = self.expr(id)?.as_int()?;
                    match &self.model.datasets[ds] {
                        DsMeta::Array { .. } => {
                            if file != 0 {
                                return other("an array dataset has one file, id 0");
                            }
                            HVal::ArrayFile { ds }
                        }
                        DsMeta::Files { .. } => match &f.chunk {
                            Some(k) => HVal::Chunk { ds, file, k: self.expr(k)?.as_int()? },
                            None => HVal::File { ds, file },
                        },
                    }
                }
                (_, _, Some(of)) => {
                    let inner = self.handle(of)?;
                    let (ds, file) = match inner {
                        HVal::Sample { ds, id } => (ds, self.container_of(ds, id)),
                        HVal::File { ds, file } => (ds, file),
                        HVal::Slot { ds, .. } | HVal::ArrayFile { ds } => (ds, 0),
                        HVal::Chunk { ds, file, .. } | HVal::Unit { ds, file, .. } | HVal::Column { ds, file, .. } => (ds, file),
                        other_h => return other(format!("file {{of}}: {other_h:?} has no container")),
                    };
                    match &f.chunk {
                        Some(k) => HVal::Chunk { ds, file, k: self.expr(k)?.as_int()? },
                        None => match &self.model.datasets[ds] {
                            DsMeta::Array { .. } => HVal::ArrayFile { ds },
                            _ => HVal::File { ds, file },
                        },
                    }
                }
                _ => return other("file handle needs {dataset, id} or {of}"),
            },
            Handle::Dir { dataset, id } => {
                let ds = self.model.dataset(dataset)?;
                HVal::Dir { ds, d: self.expr(id)?.as_int()? }
            }
            Handle::Inventory(dataset) => HVal::Inventory { ds: self.model.dataset(dataset)? },
            Handle::Object { namespace, fields } => {
                let ns = self.model.namespace(namespace)?;
                let meta = &self.model.namespaces[ns];
                let mut vals: HashMap<&'a str, Value<'a>> = HashMap::with_capacity(fields.len());
                for (name, e) in fields {
                    let v = self.expr(e)?;
                    match (meta.fields.get(name), &v) {
                        (Some(FieldType::Int), Value::Int(_)) | (Some(FieldType::Str), Value::Str(_)) => {}
                        // rounded as `as_int` rounds it, and refused where it does
                        (Some(FieldType::Int), Value::Float(_)) => {
                            v.as_int()?;
                        }
                        (Some(t), _) => return other(format!("object field `{name}` of `{namespace}` is {:?}, got {}", t, v.kind())),
                        (None, _) => unreachable!("object field `{name}` of `{namespace}`: the checkers refuse fields other than the namespace's"),
                    }
                    vals.insert(name.as_str(), v);
                }
                let path = meta.pattern.format(|f| match vals.get(f) {
                    Some(Value::Int(n)) => Some(FieldValue::Int(*n)),
                    Some(Value::Float(x)) => Some(FieldValue::Int(x.round() as i64)),
                    Some(Value::Str(s)) => Some(FieldValue::Str(s)),
                    _ => None,
                })?;
                let size = match meta.size {
                    None => None,
                    Some(e) => {
                        let saved = self.fields.replace(vals);
                        let r = self.expr(e);
                        self.fields = saved;
                        Some(r?.as_int()?)
                    }
                };
                HVal::Object { ns, path: Arc::from(path), size }
            }
            Handle::Unit(uh) => {
                let inner = self.handle(&uh.of)?;
                let index = match &uh.index {
                    Some(e) => Some(self.expr(e)?.as_int()?),
                    None => None,
                };
                match (inner, index) {
                    (HVal::Sample { ds, id }, None) => HVal::Unit { ds, file: self.container_of(ds, id), u: self.model.datasets[ds].unit_of_sample(id)? },
                    (HVal::Sample { ds, id }, Some(u)) => HVal::Unit { ds, file: self.container_of(ds, id), u },
                    (HVal::File { ds, file }, Some(u)) => HVal::Unit { ds, file, u },
                    (HVal::File { .. }, None) => return other("unit {of: file} needs an index"),
                    (h, _) => return other(format!("unit {{of}}: {h:?} is not a sample or file handle")),
                }
            }
            Handle::Column(ch) => {
                let inner = self.handle(&ch.of)?;
                let c = self.expr(&ch.index)?.as_int()?;
                match inner {
                    HVal::Unit { ds, file, u } => HVal::Column { ds, file, u, c },
                    HVal::Sample { ds, id } => HVal::Column { ds, file: self.container_of(ds, id), u: self.model.datasets[ds].unit_of_sample(id)?, c },
                    h => return other(format!("column {{of}}: {h:?} is not a unit or sample handle")),
                }
            }
            Handle::Consume(dataset) => {
                let ds = self.model.dataset(dataset)?;
                self.consume(ds)?
            }
            Handle::Pick(pick) => {
                let ds = self.model.dataset(&pick.dataset)?;
                let meta = &self.model.datasets[ds];
                let n = meta.draw_domain();
                if n == 0 {
                    return other(format!("pick from empty dataset `{}`", pick.dataset));
                }
                let mut w = Words::new(self.draw_key(pick.site.get()));
                let id = match &pick.dist {
                    None => w.next_range(0, n),
                    Some(d) => {
                        let rd = self.distref(d)?;
                        let perm = meta.rank_perm();
                        sample(&rd, &mut w, Some(&perm))?.as_int()?
                    }
                };
                if id < 0 || id >= n {
                    return other(format!("pick: id {id} outside dataset `{}` of {n}", pick.dataset));
                }
                match meta {
                    DsMeta::Files { access: Access::Stream, .. } => HVal::File { ds, file: id },
                    DsMeta::Files { .. } => HVal::Sample { ds, id },
                    DsMeta::Array { .. } => HVal::Slot { ds, c: id },
                }
            }
        })
    }

    /// `consume`: the next sample without replacement, by position (see the module doc).
    fn consume(&mut self, ds: usize) -> ER<HVal> {
        let depth = self.depth;
        if depth == 0 {
            return other("`consume` outside any loop");
        }
        let split = (0..depth).rev().find(|f| self.frames[*f].loader).unwrap_or(depth - 1);
        let mut b: i64 = 0;
        for f in 0..=split {
            b = b.checked_mul(self.frames[f].iters).and_then(|x| x.checked_add(self.frame_ordinal(f)?)).ok_or_else(|| anyhow!("position overflow"))?;
        }
        let mut j: i64 = 0;
        let mut batch: i64 = 1;
        for f in split + 1..depth {
            j = j.checked_mul(self.frames[f].iters).and_then(|x| x.checked_add(self.frame_ordinal(f)?)).ok_or_else(|| anyhow!("position overflow"))?;
            batch = batch.checked_mul(self.frames[f].iters).ok_or_else(|| anyhow!("batch size overflow"))?;
        }
        let meta = &self.model.datasets[ds];
        let n = meta.draw_domain();
        let g = self.actor_count;
        let per_epoch = g.checked_mul(batch).ok_or_else(|| anyhow!("gpus × batch overflow"))?;
        let epoch_len = n / per_epoch;
        if epoch_len == 0 {
            return other(format!("dataset `{}` has {n} samples, fewer than one batch across {g} actors ({batch} per batch)", meta.name()));
        }
        let epoch = b / epoch_len;
        let pos = (b % epoch_len).checked_mul(batch).and_then(|x| x.checked_add(j)).and_then(|x| x.checked_mul(g)).and_then(|x| x.checked_add(self.actor)).ok_or_else(|| anyhow!("position overflow"))?;
        let perm = meta.consume_perm(epoch)?;
        let id = perm.apply(pos as u64) as i64;
        Ok(if meta.access() == Access::Stream { HVal::File { ds, file: id } } else { HVal::Sample { ds, id } })
    }

    /// The unit lengths of a container file, computed once and held (bounded cache).
    fn unit_lens(&mut self, ds: usize, file: i64) -> ER<Arc<Vec<i64>>> {
        if let Some(v) = self.unit_lens.get(&(ds, file)) {
            return Ok(v.clone());
        }
        let v = Arc::new(self.model.datasets[ds].unit_lens(file)?);
        if self.unit_lens.len() >= 64 {
            self.unit_lens.clear();
        }
        self.unit_lens.insert((ds, file), v.clone());
        Ok(v)
    }

    /// A files dataset's file geometry, from `geom` or computed and held there; `None` when
    /// one sample fills the file (`file_size` is one draw) or `file_geom` refuses (the caller
    /// asks `DsMeta` again, for its error).
    fn file_geom(&mut self, ds: usize, file: i64) -> Option<Arc<FileGeom>> {
        self.file_geom_within(ds, file, GEOM_OFFSETS)
    }

    /// `file_geom` holding at most `max_offsets` offsets, the newest file's included (RC1
    /// review: the newest file was always kept, so a file of n samples held 8·n bytes).
    fn file_geom_within(&mut self, ds: usize, file: i64, max_offsets: usize) -> Option<Arc<FileGeom>> {
        let model = self.model;
        let meta = &model.datasets[ds];
        if !matches!(meta, DsMeta::Files { spf, .. } if *spf > 1) {
            return None;
        }
        if let Some(i) = self.geom.iter().rposition(|(k, _)| *k == (ds, file)) {
            let e = self.geom.remove(i);
            let g = e.1.clone();
            self.geom.push(e);
            return Some(g);
        }
        let g = Arc::new(meta.file_geom(file, max_offsets).ok()?);
        self.geom.push(((ds, file), g.clone()));
        while self.geom.len() > 1 && (self.geom.len() > GEOM_FILES || geom_held(&self.geom) > max_offsets) {
            self.geom.remove(0);
        }
        Some(g)
    }

    /// `DsMeta::file_size`, through `geom`.
    fn file_size(&mut self, ds: usize, file: i64) -> ER<i64> {
        match self.file_geom(ds, file) {
            Some(g) => Ok(g.size),
            None => Ok(self.model.datasets[ds].file_size(file)?),
        }
    }

    /// `DsMeta::sample_offset`, through `geom`.
    fn sample_offset(&mut self, ds: usize, id: i64) -> ER<i64> {
        self.sample_offset_within(ds, id, GEOM_OFFSETS)
    }

    /// `sample_offset` through a cache of at most `max_offsets` anchors.
    fn sample_offset_within(&mut self, ds: usize, id: i64, max_offsets: usize) -> ER<i64> {
        let model = self.model;
        let meta = &model.datasets[ds];
        if let DsMeta::Files { spf, .. } = meta {
            if let Some(g) = self.file_geom_within(ds, id / spf, max_offsets) {
                if let Some(off) = meta.offset_from_geom(&g, id) {
                    return Ok(off);
                }
            }
        }
        Ok(meta.sample_offset(id)?)
    }

    fn size_of(&mut self, h: &HVal) -> ER<i64> {
        let model = self.model;
        let ds = &model.datasets;
        Ok(match h {
            HVal::Sample { ds: d, id } => ds[*d].sample_size(*id)?,
            HVal::File { ds: d, file } => self.file_size(*d, *file)?,
            HVal::Chunk { ds: d, file, k } => ds[*d].chunk_size(*file, *k)?,
            HVal::Slot { ds: d, c } => ds[*d].sample_size(*c)?,
            HVal::ArrayFile { ds: d } => ds[*d].file_size(0)?,
            HVal::Unit { ds: d, file, u } => {
                let lens = self.unit_lens(*d, *file)?;
                match lens.get(*u as usize) {
                    Some(n) if *u >= 0 => *n,
                    _ => return other(format!("unit {u} outside file {file} ({} units)", lens.len())),
                }
            }
            HVal::Column { ds: d, file, u, c } => ds[*d].col_len(*file, *u, *c)?,
            HVal::Dir { .. } => return other("size of a directory"),
            HVal::Inventory { ds: d } => match ds[*d].inventory() {
                Some((_, size, _)) => size,
                None => return other(format!("dataset `{}` has no inventory", ds[*d].name())),
            },
            HVal::Object { size: Some(s), .. } => *s,
            HVal::Object { path, .. } => match self.written.get(path) {
                Some(n) => *n,
                None => return other(format!("size of as_written object `{path}`: no writes recorded at this position")),
            },
        })
    }

    fn offset_of(&mut self, h: &HVal) -> ER<i64> {
        Ok(match h {
            HVal::Sample { ds, id } => self.sample_offset(*ds, *id)?,
            HVal::Slot { ds, c } => self.model.datasets[*ds].sample_offset(*c)?,
            HVal::Chunk { .. } | HVal::File { .. } | HVal::ArrayFile { .. } | HVal::Inventory { .. } => 0,
            HVal::Unit { ds, file, u } => self.unit_offset(*ds, *file, *u)?,
            HVal::Column { ds, file, u, c } => match self.unit_offset(*ds, *file, *u)?.checked_add(self.model.datasets[*ds].col_offset_in_unit(*file, *u, *c)?) {
                Some(off) => off,
                None => return other(format!("dataset `{}`: the offset of column {c} of unit {u} of file {file} overflows the 64-bit integers of the contract (2^63-1)", self.model.datasets[*ds].name())),
            },
            _ => return other(format!("offset of {h:?}")),
        })
    }

    fn unit_offset(&mut self, ds: usize, file: i64, u: i64) -> ER<i64> {
        let lens = self.unit_lens(ds, file)?;
        if u < 0 || u as usize >= lens.len() {
            return other(format!("unit {u} outside file {file} ({} units)", lens.len()));
        }
        let start = self.model.datasets[ds].data_start(file)?;
        match lens[..u as usize].iter().try_fold(start, |s, n| s.checked_add(*n)) {
            Some(off) => Ok(off),
            None => other(format!("dataset `{}`: the offset of unit {u} of file {file} overflows the 64-bit integers of the contract (2^63-1)", self.model.datasets[ds].name())),
        }
    }

    fn target(&mut self, h: &'a Handle) -> ER<Target> {
        let hv = self.handle(h)?;
        let model = self.model;
        let ds = &model.datasets;
        Ok(match &hv {
            HVal::Sample { ds: d, id } => {
                let file = self.container_of(*d, *id);
                Target { path: ds[*d].file_path(file, None)?, size: Some(self.file_size(*d, file)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None, capped: false }
            }
            HVal::File { ds: d, file } | HVal::Unit { ds: d, file, .. } | HVal::Column { ds: d, file, .. } => {
                Target { path: ds[*d].file_path(*file, None)?, size: Some(self.file_size(*d, *file)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None, capped: false }
            }
            HVal::Chunk { ds: d, file, k } => Target { path: ds[*d].file_path(*file, Some(*k))?, size: Some(ds[*d].chunk_size(*file, *k)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None, capped: false },
            HVal::Slot { ds: d, .. } | HVal::ArrayFile { ds: d } => Target { path: ds[*d].file_path(0, None)?, size: Some(ds[*d].file_size(0)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None, capped: false },
            HVal::Dir { ds: d, d: dir } => Target { path: ds[*d].dir_path(*dir)?, size: None, as_written: false, seed: ds[*d].dataset_seed(), entries: ds[*d].dir_entries(*dir), capped: false },
            HVal::Inventory { ds: d } => {
                let Some((path, size, capped)) = ds[*d].inventory() else { return other(format!("dataset `{}` has no inventory", ds[*d].name())) };
                Target { path, size: Some(size), as_written: false, seed: ds[*d].dataset_seed(), entries: None, capped }
            }
            HVal::Object { ns, path, size } => Target { path: path.clone(), size: *size, as_written: size.is_none(), seed: self.model.namespaces[*ns].seed, entries: None, capped: false },
        })
    }

    // ---- statements ----

    /// Advance the walk to the next event.
    fn step(&mut self) -> ER<Option<Next<'a>>> {
        if self.pending_fork.is_some() {
            return other("the fork was not answered: call accept_fork or inline_fork before the next event");
        }
        loop {
            let Some(top) = self.stack.last_mut() else { return Ok(None) };
            match top {
                Cont::Body { nodes, i } => {
                    let nodes: &'a [Node] = nodes;
                    let at = *i;
                    if at >= nodes.len() {
                        self.stack.pop();
                        self.scopes.pop();
                        continue;
                    }
                    *i += 1;
                    if let Some(n) = self.statement(&nodes[at])? {
                        return Ok(Some(n));
                    }
                }
                Cont::Loop { body, from, step, iters, k } => {
                    if *k < *iters {
                        // wrapping is exact: the index fits (it is below `to`), only k·step may not
                        let (body, idx) = (*body, from.wrapping_add(k.wrapping_mul(*step)));
                        *k += 1;
                        self.frames.last_mut().unwrap().idx = idx;
                        self.push_body(body);
                    } else {
                        self.stack.pop();
                        self.frames.pop();
                        self.depth = self.frames.len();
                    }
                }
                Cont::Fork { body, iters, k, open, written } => {
                    if *k < *iters {
                        let (body, idx) = (*body, *k);
                        if idx > 0 {
                            self.open.clone_from(open);
                            self.written.clone_from(written);
                        }
                        *k += 1;
                        self.frames.last_mut().unwrap().idx = idx;
                        self.push_body(body);
                    } else {
                        let Some(Cont::Fork { open, written, .. }) = self.stack.pop() else { unreachable!() };
                        self.open = open;
                        self.written = written;
                        self.frames.pop();
                        self.depth = self.frames.len();
                    }
                }
                Cont::PhaseEnd => {
                    self.stack.pop();
                    self.phases.pop();
                }
                Cont::Io(_) => {
                    if let Some(n) = self.io_step()? {
                        return Ok(Some(n));
                    }
                }
            }
        }
    }

    fn statement(&mut self, node: &'a Node) -> ER<Option<Next<'a>>> {
        match node {
            Node::Let { name, value } => {
                let v = self.exprlike(value)?;
                let level = self.scopes.last_mut().unwrap();
                level.bindings.insert(name.as_str(), Binding { value: v, body: level.body, def: value });
                Ok(None)
            }
            Node::Loop { index, from, to, step, body } => {
                let from = match from {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                let to = self.expr(to)?.as_int()?;
                let step = match step {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 1,
                };
                if step <= 0 {
                    return other(format!("loop `{index}`: step {step} must be positive"));
                }
                // in 128 bits: `to - from` passes 2^63 when the bounds have opposite signs
                let iters = if to > from {
                    match i64::try_from((to as i128 - from as i128 + step as i128 - 1) / step as i128) {
                        Ok(n) => n,
                        Err(_) => return other(format!("loop `{index}`: from {from} to {to} by {step} is more than 2^63-1 iterations")),
                    }
                } else {
                    0
                };
                self.push_loop(index, from, step, iters, false, body);
                Ok(None)
            }
            Node::Parallel { index, width, body } => {
                let width = self.expr(width)?.as_int()?;
                if width < 0 {
                    return other(format!("parallel `{index}`: width {width}"));
                }
                self.pending_fork = Some(PendingFork { index, width, loader: false, body });
                Ok(Some(Next::Fork(ForkKind::Parallel { index, width })))
            }
            Node::Loader { name, index, workers, prefetch, batches, ordered, body } => {
                let workers = self.expr(workers)?.as_int()?;
                let prefetch = self.expr(prefetch)?.as_int()?;
                let batches = self.expr(batches)?.as_int()?;
                if workers < 1 || prefetch < 0 || batches < 0 {
                    return other(format!("loader: workers {workers}, prefetch {prefetch}, batches {batches}"));
                }
                self.pending_fork = Some(PendingFork { index, width: batches, loader: true, body });
                Ok(Some(Next::Fork(ForkKind::Loader { name, index, workers, prefetch, batches, ordered: ordered.unwrap_or(true) })))
            }
            Node::Channel { name, capacity, ordered } => {
                let capacity = self.expr(capacity)?.as_int()?;
                self.ctl(Control::Channel { name, capacity, ordered: ordered.unwrap_or(true) })
            }
            Node::Put { channel, seq } => {
                let seq = self.expr(seq)?.as_int()?;
                self.ctl(Control::Put { channel, seq })
            }
            Node::Take { channel } => self.ctl(Control::Take { channel }),
            Node::Barrier { scope } => self.ctl(Control::Barrier { scope }),
            Node::Compute { ns } => {
                let ns = self.expr(ns)?.as_int()?;
                self.ctl(Control::Compute { ns })
            }
            Node::Cond { test, then, otherwise } => {
                if self.expr(test)?.as_bool()? {
                    self.push_body(then);
                } else if let Some(b) = otherwise {
                    self.push_body(b);
                }
                Ok(None)
            }
            Node::Choose { arms, site } => {
                let total: f64 = arms.iter().map(|a| a.weight).sum();
                if total <= 0.0 {
                    return other("choose: total weight is zero");
                }
                let u = Words::new(self.draw_key(site.get())).next_f64() * total;
                let mut acc = 0.0;
                let mut chosen = arms.len() - 1;
                for (i, arm) in arms.iter().enumerate() {
                    acc += arm.weight;
                    if u < acc {
                        chosen = i;
                        break;
                    }
                }
                self.push_body(&arms[chosen].body);
                Ok(None)
            }
            Node::Phase { name, body } => {
                let label: Arc<str> = match self.expr(name)? {
                    Value::Str(s) => s,
                    Value::Int(n) => Arc::from(n.to_string()),
                    v => return other(format!("phase name is {}", v.kind())),
                };
                self.phases.push(label);
                self.stack.push(Cont::PhaseEnd);
                self.push_body(body);
                Ok(None)
            }
            Node::Trace { file, .. } => match self.model.traces.get(file.as_str()) {
                Some(t) => {
                    self.fill_indices();
                    Ok(Some(Next::Trace(t.clone())))
                }
                None => other(format!("trace `{file}`: not loaded (a document loaded from text has no directory to find it in)")),
            },
            _ => self.op(node),
        }
    }

    /// One op of the `read`/`write` statement on top of the stack, or finish it.
    fn io_step(&mut self) -> ER<Option<Next<'a>>> {
        let Some(Cont::Io(io)) = self.stack.last_mut() else { unreachable!("io_step without an Io on the stack") };
        let (kind, len) = (io.kind, io.len);
        let next = match &mut io.left {
            Left::Count(n) if *n > 0 => {
                *n -= 1;
                let bytes = if kind == OpKind::Read { expected(io.size, io.off, len) } else { len };
                let off = io.off;
                past_end(kind, off, len)?;
                io.off = off + bytes;
                Some((off, bytes))
            }
            Left::Count(_) => None,
            Left::UntilEof { size, tail_done } => {
                if *tail_done {
                    None
                } else if io.off < *size {
                    let bytes = (*size - io.off).min(len);
                    let off = io.off;
                    io.off += bytes;
                    Some((off, bytes))
                } else {
                    *tail_done = true;
                    Some((io.off, 0))
                }
            }
        };
        let Some((off, bytes)) = next else {
            let Some(Cont::Io(io)) = self.stack.pop() else { unreachable!() };
            if !io.positioned {
                self.open.get_mut(&io.path).unwrap().pos = io.off;
            }
            return Ok(None);
        };
        let (seed, expect, positioned, as_written, capped) = (io.seed, io.expect, io.positioned, io.as_written, io.capped);
        if kind == OpKind::Write && as_written {
            let path = io.path.clone();
            self.add_written(path, len)?;
        }
        self.set_pending(Pending { kind, path: None, path2: None, offset: off, len, bytes, aux: 0, expect, positioned, seed, capped });
        Ok(Some(Next::Op))
    }

    fn fill_indices(&mut self) {
        self.indices_buf.clear();
        self.indices_buf.extend(self.frames.iter().map(|f| f.idx));
    }

    fn ctl(&mut self, c: Control<'a>) -> ER<Option<Next<'a>>> {
        self.fill_indices();
        Ok(Some(Next::Control(c)))
    }

    fn set_pending(&mut self, p: Pending<'a>) {
        self.fill_indices();
        self.pending = p;
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, kind: OpKind, t: Target, path2: Option<Arc<str>>, offset: i64, len: i64, bytes: i64, aux: u64, expect: &'a Expect, positioned: bool) -> ER<Option<Next<'a>>> {
        let capped = t.capped && kind == OpKind::Read;
        self.set_pending(Pending { kind, path: Some(t.path), path2, offset, len, bytes, aux, expect: expect.as_deref().unwrap_or(&[]), positioned, seed: t.seed, capped });
        Ok(Some(Next::Op))
    }

    /// Bytes written to an `as_written` object, its size; checked, never wrapped (RC1 review).
    fn add_written(&mut self, path: Arc<str>, len: i64) -> ER<()> {
        let n = self.written.entry(path).or_insert(0);
        match n.checked_add(len) {
            Some(v) => *n = v,
            None => return other(format!("write: the size written, {n} + {len} bytes, passes 2^63-1")),
        }
        Ok(())
    }

    fn opened(&self, path: &Arc<str>, what: &str) -> ER<FileState> {
        match self.open.get(path) {
            Some(s) => Ok(s.clone()),
            None => other(format!("{what} on `{path}`, which this actor has not opened")),
        }
    }

    fn op(&mut self, node: &'a Node) -> ER<Option<Next<'a>>> {
        match node {
            Node::Open { file, flags, mode, expect } => {
                let t = self.target(file)?;
                let append = flags.contains(&OpenFlag::APPEND);
                let trunc = flags.contains(&OpenFlag::TRUNC);
                if trunc && t.as_written {
                    self.written.insert(t.path.clone(), 0);
                }
                let size = if t.as_written { self.written.get(&t.path).copied() } else { t.size };
                let pos = if append { size.unwrap_or(0) } else { 0 };
                self.open.insert(t.path.clone(), FileState { pos, size, as_written: t.as_written, capped: t.capped });
                let aux = flag_bits(flags) | ((mode.unwrap_or(0) as u64) << 32);
                self.emit(OpKind::Open, t, None, 0, 0, 0, aux, expect, false)
            }
            Node::Close(f) => {
                let t = self.target(&f.file)?;
                self.opened(&t.path, "close")?;
                self.open.remove(&t.path);
                self.emit(OpKind::Close, t, None, 0, 0, 0, 0, &f.expect, false)
            }
            Node::Read { file, len, offset, repeat, expect } => {
                let t = self.target(file)?;
                let len = self.expr(len)?.as_int()?;
                if len < 0 {
                    return other(format!("read: negative length {len}"));
                }
                let st = self.opened(&t.path, "read")?;
                let size = if st.as_written { self.written.get(&t.path).copied() } else { st.size };
                let positioned = match offset {
                    Some(e) => Some(self.expr(e)?.as_int()?),
                    None => None,
                };
                let off = positioned.unwrap_or(st.pos);
                past_end(OpKind::Read, off, len)?;
                if repeat.is_none() {
                    // the common case, without the sequence machinery
                    let bytes = expected(size, off, len);
                    if positioned.is_none() {
                        self.open.get_mut(&t.path).unwrap().pos = off + bytes;
                    }
                    let t = Target { capped: st.capped, ..t };
                    return self.emit(OpKind::Read, t, None, off, len, bytes, 0, expect, positioned.is_some());
                }
                let left = match repeat {
                    None => Left::Count(1),
                    Some(Repeat::Count(n)) => Left::Count(self.expr(n)?.as_int()?),
                    Some(Repeat::UntilEof(_)) => {
                        let Some(size) = size else {
                            return other(format!("until_eof on `{}`: size unknown (an as_written object needs the creating writes at this position)", t.path));
                        };
                        if len == 0 {
                            return other("until_eof with a zero-length read");
                        }
                        Left::UntilEof { size, tail_done: false }
                    }
                };
                self.stack.push(Cont::Io(IoSeq { kind: OpKind::Read, path: t.path, seed: t.seed, off, len, size, positioned: positioned.is_some(), expect: expect.as_deref().unwrap_or(&[]), as_written: st.as_written, capped: st.capped, left }));
                Ok(None)
            }
            Node::Write { file, len, offset, repeat, expect } => {
                let t = self.target(file)?;
                let len = self.expr(len)?.as_int()?;
                if len < 0 {
                    return other(format!("write: negative length {len}"));
                }
                let st = self.opened(&t.path, "write")?;
                let positioned = match offset {
                    Some(e) => Some(self.expr(e)?.as_int()?),
                    None => None,
                };
                let n = match repeat {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 1,
                };
                let off = positioned.unwrap_or(st.pos);
                past_end(OpKind::Write, off, len)?;
                if n == 1 {
                    if st.as_written {
                        self.add_written(t.path.clone(), len)?;
                    }
                    if positioned.is_none() {
                        self.open.get_mut(&t.path).unwrap().pos = off + len;
                    }
                    return self.emit(OpKind::Write, t, None, off, len, len, 0, expect, positioned.is_some());
                }
                self.stack.push(Cont::Io(IoSeq { kind: OpKind::Write, path: t.path, seed: t.seed, off, len, size: None, positioned: positioned.is_some(), expect: expect.as_deref().unwrap_or(&[]), as_written: st.as_written, capped: false, left: Left::Count(n) }));
                Ok(None)
            }
            Node::Lseek { file, offset, whence } => {
                let t = self.target(file)?;
                let off = self.expr(offset)?.as_int()?;
                let st = self.opened(&t.path, "lseek")?;
                let size = if st.as_written { self.written.get(&t.path).copied() } else { st.size };
                let pos = match whence {
                    Whence::SET => Some(off),
                    Whence::CUR => st.pos.checked_add(off),
                    Whence::END => match size {
                        Some(s) => s.checked_add(off),
                        None => return other(format!("lseek END on `{}`: size unknown", t.path)),
                    },
                };
                let Some(pos) = pos else {
                    return other(format!("lseek by {off} on `{}`: the position passes 2^63-1", t.path));
                };
                if pos < 0 {
                    return other(format!("lseek to {pos} on `{}`", t.path));
                }
                self.open.get_mut(&t.path).unwrap().pos = pos;
                self.emit(OpKind::Lseek, t, None, off, 0, 0, *whence as u64, &None, false)
            }
            Node::Ioctl { file, request, expect } => {
                let t = self.target(file)?;
                self.opened(&t.path, "ioctl")?;
                self.emit(OpKind::Ioctl, t, None, 0, 0, 0, *request as u64, expect, false)
            }
            Node::Fadvise { file, advice, offset, len, expect } => {
                let t = self.target(file)?;
                let off = match offset {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                let len = match len {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                if off < 0 || len < 0 {
                    return other(format!("fadvise: offset {off}, len {len}"));
                }
                self.opened(&t.path, "fadvise")?;
                self.emit(OpKind::Fadvise, t, None, off, len, 0, *advice as u64, expect, false)
            }
            Node::Fstat(f) | Node::Fsync(f) | Node::Fdatasync(f) => {
                let t = self.target(&f.file)?;
                let kind = match node {
                    Node::Fstat(_) => OpKind::Fstat,
                    Node::Fsync(_) => OpKind::Fsync,
                    _ => OpKind::Fdatasync,
                };
                self.opened(&t.path, kind.name())?;
                self.emit(kind, t, None, 0, 0, 0, 0, &f.expect, false)
            }
            Node::Stat(f) => {
                let t = self.target(&f.file)?;
                self.emit(OpKind::Stat, t, None, 0, 0, 0, 0, &f.expect, false)
            }
            Node::Unlink(f) => {
                let t = self.target(&f.file)?;
                self.written.remove(&t.path);
                self.emit(OpKind::Unlink, t, None, 0, 0, 0, 0, &f.expect, false)
            }
            Node::Ftruncate { file, len, expect } => {
                let t = self.target(file)?;
                let len = self.expr(len)?.as_int()?;
                self.opened(&t.path, "ftruncate")?;
                self.emit(OpKind::Ftruncate, t, None, 0, len, 0, 0, expect, false)
            }
            Node::Fallocate { file, offset, len, expect } => {
                let t = self.target(file)?;
                let off = match offset {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                let len = self.expr(len)?.as_int()?;
                self.opened(&t.path, "fallocate")?;
                self.emit(OpKind::Fallocate, t, None, off, len, 0, 0, expect, false)
            }
            Node::Mkdir { dir, mode, expect } => {
                let t = self.target(dir)?;
                self.emit(OpKind::Mkdir, t, None, 0, 0, 0, mode.unwrap_or(0o777) as u64, expect, false)
            }
            Node::Rmdir { dir, expect } => {
                let t = self.target(dir)?;
                self.emit(OpKind::Rmdir, t, None, 0, 0, 0, 0, expect, false)
            }
            Node::Rename { from, to, expect } => {
                let a = self.target(from)?;
                let b = self.target(to)?;
                if let Some(n) = self.written.remove(&a.path) {
                    self.written.insert(b.path.clone(), n);
                }
                self.emit(OpKind::Rename, a, Some(b.path), 0, 0, 0, 0, expect, false)
            }
            Node::Readdir { dir, expect, .. } => {
                let t = self.target(dir)?;
                self.opened(&t.path, "readdir")?;
                let entries = t.entries.unwrap_or(-1);
                self.emit(OpKind::Readdir, t, None, 0, 0, entries, 0, expect, false)
            }
            _ => unreachable!("control node in op()"),
        }
    }
}

impl Pending<'_> {
    fn empty() -> Self {
        Pending { kind: OpKind::Open, path: None, path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: &[], positioned: false, seed: 0, capped: false }
    }
}

/// The end of a read or write of `len` at `off`: an op wholly inside 0 to 2^63-1, as the
/// kernel's `rw_verify_area` requires, or the error, never a wrapped offset (RC1 review).
fn past_end(kind: OpKind, off: i64, len: i64) -> ER<i64> {
    match off.checked_add(len) {
        Some(end) if off >= 0 => Ok(end),
        _ => other(format!("{}: offset {off} and length {len} reach outside 0 to 2^63-1", kind.name())),
    }
}

/// Bytes a read of `len` at `off` returns against a file of known `size`.
fn expected(size: Option<i64>, off: i64, len: i64) -> i64 {
    match size {
        Some(s) => (s - off).clamp(0, len),
        None => len,
    }
}

/// Feed a VM's events to a `Sink` until the walk it has started ends. A `Fork` the sink
/// declines is walked inline. The sink's `finish` is the caller's to call.
pub fn drive<'m, 'a: 'm, S: Sink<'m, 'a>>(vm: &mut Vm<'m, 'a>, sink: &mut S) -> Result<()> {
    loop {
        match vm.next()? {
            None => return Ok(()),
            Some(Event::Op(op, ctx)) => sink.op(&op, &ctx)?,
            Some(Event::Control(c, ctx)) => sink.control(c, &ctx)?,
            Some(Event::Trace(t, ctx)) => sink.trace(&t, &ctx)?,
            Some(Event::Fork(kind)) => {
                let forked = {
                    let snap = || vm.snapshot();
                    sink.fork(&kind, &snap)?
                };
                if forked {
                    vm.accept_fork();
                } else {
                    vm.inline_fork();
                }
            }
        }
    }
}

/// Run one actor instance of `template` to completion through `sink`, `finish` included.
pub fn run_actor<'m, 'a: 'm, S: Sink<'m, 'a>>(model: &'m Model<'a>, template: &'a str, actor: i64, count: i64, mut sink: S) -> Result<S> {
    let a = model.ast.actors.get(template).ok_or_else(|| anyhow!("no actor `{template}`"))?;
    let mut vm = Vm::new(model, template, actor, count);
    vm.start(&a.body);
    drive(&mut vm, &mut sink).and_then(|_| sink.finish()).with_context(|| format!("actor `{template}` instance {actor}"))?;
    Ok(sink)
}

/// Instance counts of every actor template (`count` defaults to `gpus`).
pub fn actor_counts<'a>(model: &'a Model<'a>) -> Result<Vec<(&'a str, i64)>> {
    let mut out = Vec::new();
    for (name, a) in &model.ast.actors {
        let count = match &a.count {
            None => model.cfg.gpus,
            Some(e) => {
                let mut vm = Vm::new(model, name, 0, model.cfg.gpus.max(1));
                vm.eval_int(e).with_context(|| format!("actor `{name}` count"))?
            }
        };
        if count < 0 {
            bail!("actor `{name}`: count {count}");
        }
        out.push((name.as_str(), count));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(v: Option<Value>) -> Option<i64> {
        v.map(|v| match v {
            Value::Int(n) => n,
            v => panic!("{v:?}"),
        })
    }

    /// Keys that share their index (an `at` in a short loop nested in a long one: the index
    /// repeats, the outer loop's moves) are shed like any others: the cache stays bounded and
    /// keeps the entries inserted last.
    #[test]
    fn the_at_cache_is_bounded_when_indices_repeat() {
        let mut c = AtCache::default();
        let n = 3 * AT_CACHE_MAX as i64;
        for j in 0..n {
            c.insert((1, 0, vec![j]), Value::Int(j));
            assert!(c.len() <= AT_CACHE_MAX, "{} entries after {j}", c.len());
        }
        let (newest, oldest_kept) = (n - 1, n - AT_CACHE_MAX as i64 / 2);
        assert_eq!(int(c.get(&(1, 0, vec![newest]))), Some(newest), "the newest entry is kept");
        assert_eq!(int(c.get(&(1, 0, vec![oldest_kept]))), Some(oldest_kept), "so is one inserted half a cache before it");
        assert_eq!(int(c.get(&(1, 0, vec![0]))), None, "the first is shed");
    }

    /// Of entries with different indices, those with the larger are kept, whatever the order
    /// they were inserted in.
    #[test]
    fn the_at_cache_keeps_the_larger_indices() {
        let mut c = AtCache::default();
        let n = AT_CACHE_MAX as i64;
        for i in (0..n).rev() {
            c.insert((1, i, vec![]), Value::Int(i));
        }
        assert_eq!(c.len(), AT_CACHE_MAX);
        c.insert((1, -1, vec![]), Value::Int(-1));
        assert_eq!(c.len(), AT_CACHE_MAX / 2 + 1);
        assert_eq!(int(c.get(&(1, n - 1, vec![]))), Some(n - 1), "the first inserted, the largest index, is kept");
        assert_eq!(int(c.get(&(1, n / 2, vec![]))), Some(n / 2));
        assert_eq!(int(c.get(&(1, n / 2 - 1, vec![]))), None, "the smaller half is shed");
        assert_eq!(int(c.get(&(1, -1, vec![]))), Some(-1), "the value just inserted is kept");
    }

    /// RC1 review: the geometry cache always kept the file it had just computed, so one file of
    /// n samples (a dataset of one big shard) held a table of n offsets in every VM. A file of
    /// more samples than the bound held its size alone, and its offsets cost O(n) draws each
    /// again (RC1 review); it now holds an anchor every k samples, an offset draws at most k-1
    /// sizes from its anchor, and what the cache answers is `DsMeta`'s to the bit.
    #[test]
    fn the_geometry_cache_never_holds_more_offsets_than_its_bound() {
        use crate::eval::tests::SIZE_DRAWS;
        let draws = || SIZE_DRAWS.with(|c| c.replace(0));
        let tfrecord = r#", "format": {"class": "tfrecord", "reader": "tf.data", "framing": {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}}"#;
        // one column in units smaller and larger than k, with every header, footer, and alignment
        let units = |unit: i64| {
            format!(
                r#", "format": {{"class": "tfrecord", "reader": "tf.data", "framing": {{"unit": {unit}, "file_header": 4, "file_header_per_unit": 3, "unit_header": 9,
                  "unit_header_per_sample": 2, "unit_footer": 5, "unit_footer_per_sample": 1, "unit_align": 16, "file_footer": 7, "file_align": 64,
                  "columns": [{{"header": 3, "row_header": 4, "row_footer": 1, "row_align": 8, "align": 32, "weight": 1}}]}}}}"#
            )
        };
        let cap = 30;
        for format in [String::new(), tfrecord.to_string(), units(1), units(3), units(7), units(40)] {
            // files of 100, 100, and 20 samples: two past the bound, one under it
            let text = format!(
                r#"{{"ast": "0.13", "name": "t",
                  "datasets": {{"d": {{"layout": "compound", "pattern": "d/{{id:04}}.bin", "count": 220, "samples_per_file": 100,
                     "size": {{"lognormal": {{"median": 1000, "sigma": 0.5, "min": 1}}}}, "seed": 7, "access": "map"{format}}}}},
                  "actors": {{"gpu": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "d", "id": 0}}}}}}}}]}}}}}}"#
            );
            let loaded = crate::load_str(&text).unwrap_or_else(|e| panic!("{e:#}"));
            let cfg = Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() };
            let params = Params::new(&loaded.ast, &cfg).unwrap();
            let model = build_model(&loaded.ast, &cfg, &params).unwrap();
            let meta = &model.datasets[0];
            let mut vm = Vm::new(&model, "gpu", 0, 1);
            for file in [0, 2, 1, 2, 0, 1] {
                let g = vm.file_geom_within(0, file, cap).expect("a container file's geometry");
                assert!(geom_held(&vm.geom) <= cap, "format `{format}`: {} offsets held after file {file}", geom_held(&vm.geom));
                assert_eq!(g.size, meta.file_size(file).unwrap(), "format `{format}` file {file}");
                let n = meta.samples_in_file(file).unwrap();
                let k = g.every;
                assert!(g.anchors.as_ref().is_some_and(|a| a.len() <= cap), "format `{format}` file {file}: anchors {:?}", g.anchors.as_ref().map(|a| a.len()));
                assert!(k >= 1 && (n as usize > cap || k == 1), "format `{format}` file {file}: {n} samples, every {k}");
                for id in file * 100..file * 100 + n {
                    draws();
                    let got = meta.offset_from_geom(&g, id);
                    let d = draws();
                    assert_eq!(got, Some(meta.sample_offset(id).unwrap()), "format `{format}` sample {id}");
                    assert!(d < k as u64, "format `{format}` sample {id}: {d} size draws from an anchor every {k}");
                }
            }
            // through the VM: what it reads is DsMeta's, at most k-1 draws once its file is held
            for (max, what) in [(cap, "the test bound"), (GEOM_OFFSETS, "the production bound")] {
                let mut vm = Vm::new(&model, "gpu", 0, 1);
                let mut held = None;
                for id in (0..220).rev() {
                    let want = meta.sample_offset(id).unwrap();
                    draws();
                    assert_eq!(vm.sample_offset_within(0, id, max).ok(), Some(want), "format `{format}` sample {id} at {what}");
                    let d = draws();
                    assert!(geom_held(&vm.geom) <= max, "format `{format}` at {what}: {} anchors held", geom_held(&vm.geom));
                    let k = vm.geom.last().unwrap().1.every;
                    assert!(held != Some(id / 100) || d < k as u64, "format `{format}` sample {id} at {what}: {d} size draws, every {k}");
                    held = Some(id / 100);
                }
            }
        }
    }
}
