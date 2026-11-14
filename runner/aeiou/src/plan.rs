//! The planner of a timeline's run (`DESIGN_REVIEW.md` §3.99, the note "The run side of
//! timelines", step 2; brief item 72 (7)): the walk `dry-run` makes (`Vm::next` through
//! `tl_step`), each op turned into an owned **record**: its number in planned order, its
//! **executor lane**, its planned time, the op, and the **edges** it waits on. An executor
//! (step 3) holds the records a horizon of planned time ahead of the wall clock and issues a
//! lane's ops one at a time at `max(plan + L, the previous completion, its edges)`.
//!
//! An executor lane (`vm::TlAt::exec`) is a lane of the timeline's body (the root and each
//! sub-actor of a `parallel`), a worker lane (a `workers` declaration's lane, submission i on
//! lane i mod `lanes`), or an eviction lane (a tier's, victim i of a batch on lane i mod `lanes`,
//! the same lanes batch after batch); ops outside a timeline have one lane of their own. Lanes are
//! numbered in the order their first event is planned.
//!
//! Edges are the `trace` node's **path order** (`trace.rs`, `runner/REFERENCE.md` §13) across the
//! executor lanes, in planned order: a changing op (a creating or truncating open, a write, a
//! truncate, an allocate, a sync or close of a writable open, a rename, an unlink, a mkdir, a
//! rmdir) waits for every earlier op on its path, and a reading op (any other) for every earlier
//! changing op on it; reads never wait for reads. Since an executor lane issues its ops one after
//! another and a changing op issues only once the path's earlier ops have completed, an op's
//! edges name only the path's last change and, for a change, each other lane's latest read since:
//! never an op of its own lane, which completes first anyway. A path's entry is kept until the
//! executor says what it names has completed (`retire`), so memory follows the paths in flight.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::eval::Model;
use crate::trace::OwnedCtx;
use crate::vm::{Control, Event, Op, OpCtx, OpKind, TlAt, Vm};

/// The executor lane of ops outside a timeline.
const LINE: (u8, i64, i64) = (u8::MAX, 0, 0);

/// One op as planned, owned.
#[derive(Debug, Clone)]
pub struct Record {
    /// Its place in planned order, from 0.
    pub seq: u64,
    /// Its executor lane.
    pub lane: usize,
    /// Its planned time, ns from the timeline's start (outside a timeline, 0).
    pub plan: i64,
    pub kind: OpKind,
    pub path: Arc<str>,
    pub path2: Option<Arc<str>>,
    pub offset: i64,
    pub len: i64,
    pub bytes: i64,
    pub aux: u64,
    pub expect: Vec<String>,
    pub positioned: bool,
    pub seed: u64,
    pub capped: bool,
    pub ctx: OwnedCtx,
    /// The records (by `seq`) it waits for, ascending.
    pub edges: Vec<u64>,
}

impl Record {
    pub fn op(&self) -> Op<'_> {
        Op { kind: self.kind, path: &self.path, path2: self.path2.as_deref(), offset: self.offset, len: self.len, bytes: self.bytes, aux: self.aux, expect: &self.expect, positioned: self.positioned, seed: self.seed, capped: self.capped }
    }

    pub fn ctx(&self) -> OpCtx<'_> {
        self.ctx.ctx(&self.ctx.indices)
    }
}

/// What the planner gives, in planned order.
#[derive(Debug, Clone)]
pub enum Planned {
    Op(Record),
    /// A `compute` of `ns` on `lane`, starting at planned time `plan`.
    Compute { lane: usize, plan: i64, ns: i64 },
}

/// A path's ops not yet retired: its last change and each lane's latest read since.
#[derive(Default)]
struct PathState {
    change: Option<(usize, u64)>,
    reads: Vec<(usize, u64)>,
}

/// The planner's tables: executor lanes, the paths in flight, and which opens are writable.
#[derive(Default)]
pub struct Planner {
    seq: u64,
    lanes: HashMap<(u8, i64, i64), usize>,
    paths: HashMap<Arc<str>, PathState>,
    /// The writability of the latest open of a path on an executor lane, and on any lane.
    open_lane: HashMap<(usize, Arc<str>), bool>,
    open_any: HashMap<Arc<str>, bool>,
}

const O_WRONLY: u64 = 1 << 1;
const O_RDWR: u64 = 1 << 2;
const O_CREAT: u64 = 1 << 3;
const O_TRUNC: u64 = 1 << 4;

impl Planner {
    /// The executor lane of key `exec`, numbered when first seen.
    pub fn lane(&mut self, exec: (u8, i64, i64)) -> usize {
        let n = self.lanes.len();
        *self.lanes.entry(exec).or_insert(n)
    }

    /// The record of `op`, issued from `at` (`None`: outside a timeline), with its edges.
    pub fn record(&mut self, at: Option<TlAt>, op: &Op, ctx: &OpCtx) -> Record {
        let (exec, plan) = at.map_or((LINE, 0), |a| (a.exec, a.plan));
        let lane = self.lane(exec);
        let seq = self.seq;
        self.seq += 1;
        let path: Arc<str> = Arc::from(op.path);
        let path2: Option<Arc<str>> = op.path2.map(Arc::from);
        let changes = self.changes(lane, op, &path);
        let mut edges = self.touch(lane, seq, &path, changes);
        if let Some(p2) = &path2 {
            edges.extend(self.touch(lane, seq, p2, true));
            edges.sort_unstable();
            edges.dedup();
        }
        Record {
            seq,
            lane,
            plan,
            kind: op.kind,
            path,
            path2,
            offset: op.offset,
            len: op.len,
            bytes: op.bytes,
            aux: op.aux,
            expect: op.expect.to_vec(),
            positioned: op.positioned,
            seed: op.seed,
            capped: op.capped,
            ctx: OwnedCtx::of(ctx),
            edges,
        }
    }

    /// Whether `op` changes its path (the module doc), keeping the opens' writability.
    fn changes(&mut self, lane: usize, op: &Op, path: &Arc<str>) -> bool {
        match op.kind {
            OpKind::Open => {
                let w = op.aux & (O_WRONLY | O_RDWR) != 0;
                self.open_lane.insert((lane, path.clone()), w);
                self.open_any.insert(path.clone(), w);
                op.aux & (O_CREAT | O_TRUNC) != 0
            }
            OpKind::Close | OpKind::Fsync | OpKind::Fdatasync => {
                let key = (lane, path.clone());
                let w = self.open_lane.get(&key).or_else(|| self.open_any.get(path)).copied().unwrap_or(false);
                if op.kind == OpKind::Close {
                    self.open_lane.remove(&key);
                }
                w
            }
            OpKind::Write | OpKind::Ftruncate | OpKind::Fallocate | OpKind::Unlink | OpKind::Truncate | OpKind::Mkdir | OpKind::Rmdir | OpKind::Rename => true,
            OpKind::Read | OpKind::Lseek | OpKind::Fstat | OpKind::Readdir | OpKind::Ioctl | OpKind::Fadvise | OpKind::Stat => false,
        }
    }

    /// Path order for one path: what an op of `lane` waits for, and the path's entry after it.
    fn touch(&mut self, lane: usize, seq: u64, path: &Arc<str>, changes: bool) -> Vec<u64> {
        let st = self.paths.entry(path.clone()).or_default();
        let mut edges: Vec<u64> = Vec::new();
        if let Some((l, s)) = st.change {
            if l != lane {
                edges.push(s);
            }
        }
        if changes {
            edges.extend(st.reads.iter().filter(|(l, _)| *l != lane).map(|(_, s)| *s));
            st.change = Some((lane, seq));
            st.reads.clear();
        } else {
            match st.reads.iter_mut().find(|(l, _)| *l == lane) {
                Some(r) => r.1 = seq,
                None => st.reads.push((lane, seq)),
            }
        }
        edges.sort_unstable();
        edges
    }

    /// Forget what has completed (`done`, by `seq`): a path's entry goes once every op it
    /// names is done, so a later op on it waits for nothing.
    pub fn retire(&mut self, done: impl Fn(u64) -> bool) {
        self.paths.retain(|_, st| {
            st.reads.retain(|(_, s)| !done(*s));
            if st.change.is_some_and(|(_, s)| done(s)) {
                st.change = None;
            }
            st.change.is_some() || !st.reads.is_empty()
        });
        self.open_any.retain(|p, _| self.paths.contains_key(p));
    }

    /// The paths with ops not yet retired.
    pub fn paths(&self) -> usize {
        self.paths.len()
    }
}

/// An actor instance's plan: its walk, each op a record. A fork outside a timeline is walked
/// inline, as the dry run walks it; a `trace` node and the controls that wait across actors
/// (`barrier`, `put`, `take`, `channel`) are not planned.
pub struct Plan<'m, 'a> {
    vm: Vm<'m, 'a>,
    planner: Planner,
}

impl<'m, 'a: 'm> Plan<'m, 'a> {
    pub fn new(model: &'m Model<'a>, template: &'a str, actor: i64, count: i64) -> Plan<'m, 'a> {
        let body = &model.ast.actors[template].body;
        let mut vm = Vm::new(model, template, actor, count);
        vm.start(body);
        Plan { vm, planner: Planner::default() }
    }

    /// The next planned event, `None` once the walk has ended.
    pub fn next(&mut self) -> Result<Option<Planned>> {
        loop {
            let (ev, at) = match self.vm.next_at()? {
                Some((ev, at)) => (Some(ev), at),
                None => (None, None),
            };
            return Ok(Some(match ev {
                None => return Ok(None),
                Some(Event::Op(op, ctx)) => Planned::Op(self.planner.record(at, &op, &ctx)),
                Some(Event::Control(Control::Compute { ns }, _)) => {
                    let (exec, plan) = at.map_or((LINE, 0), |a| (a.exec, a.plan - ns));
                    Planned::Compute { lane: self.planner.lane(exec), plan, ns }
                }
                Some(Event::Control(c, _)) => bail!("a `{}` is not planned: a planned run is one actor's timeline", control_name(&c)),
                Some(Event::Trace(..)) => bail!("a `trace` node is not planned: it runs as the trace's own lanes"),
                Some(Event::Fork(_)) => {
                    self.vm.inline_fork();
                    continue;
                }
            }));
        }
    }

    /// See `Planner::retire`.
    pub fn retire(&mut self, done: impl Fn(u64) -> bool) {
        self.planner.retire(done)
    }

    pub fn paths(&self) -> usize {
        self.planner.paths()
    }
}

fn control_name(c: &Control) -> &'static str {
    match c {
        Control::Barrier { .. } => "barrier",
        Control::Put { .. } => "put",
        Control::Take { .. } => "take",
        Control::Channel { .. } => "channel",
        Control::Compute { .. } => "compute",
    }
}
