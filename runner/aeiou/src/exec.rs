//! The executor of a timeline's run (`DESIGN_REVIEW.md` §3.99, the note "The run side of
//! timelines", step 3; brief item 72 (7)): the planner's records (`plan.rs`), issued lane by
//! lane, one op in flight a lane, each at `ready = max(plan + L, the lane's previous completion,
//! its edges' completions)`, after which `L := ready − plan`: L, the lane's lag behind its plan,
//! never shrinks.
//!
//! The planner's lane events order lanes beyond path order. A fork starts its sub-lanes once the
//! parent reaches it (its `max(plan + L, previous completion)`), with the parent's L; a joined
//! parent then goes on at `max(plan + L, its sub-lanes' last completions)`. A submission starts
//! its body on its worker lane, and an eviction batch a store started its victims on their
//! eviction lanes, no earlier than the submitting or storing lane reaches it; worker and
//! eviction lanes are queues that go on from batch to batch: an op there issues at `max(plan,
//! the lane's previous completion, its gate, its edges)`, so a batch past its tick delays the
//! next by its overage and no more, and its L is reported, never carried (carried, the ops of
//! one batch, issued back to back, would add their times to every later batch's). The bytes a submission writes
//! are staged from when it is submitted until each write completes: past the workers'
//! `staging`, a would-be drop (a gate failure: the plan never drops a submitted store).
//!
//! `Exec` is the core, a state machine over a clock it is given (ns from the run's start, the
//! timeline's planned 0): `push` the planner's events in planned order, `poll` for the records
//! to issue now, `complete` each when it completes, `next_wake` for when to poll again.
//! `simulate` drives it on a simulated clock and storage (the tests, and what-if); the event
//! loops drive it on the wall clock.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use anyhow::Result;

use crate::plan::{Plan, Planned, Record};
use crate::vm::OpKind;

/// A lane event's place in a lane's queue.
enum Item {
    Op(Record),
    /// The lane reaches a fork, submission, or eviction batch of planned time `plan`: gate `gate`
    /// opens for the lanes it starts.
    Spawn { gate: usize, plan: i64 },
    /// The lanes started at gate `gate` begin no earlier than it opens; a sub-lane of a fork
    /// (`fresh`) takes its parent's L; a submission's (`workers`) stages its writes.
    Gate { gate: usize, plan: i64, fresh: bool, workers: Option<usize> },
    /// A joined parent goes on once the `kids` sub-lanes of its fork have ended, at its plan the
    /// latest of their planned ends. A parent's forks are joined one at a time: the next fork
    /// is behind this join in its queue, so its sub-lanes end after it.
    Join { plan: i64, kids: usize },
    End { plan: i64 },
    /// The lane enters span `name` (planned at `plan`), its reads drawing on the staging of
    /// workers `staging`, or leaves the one it entered last.
    SpanBegin { name: Arc<str>, staging: Option<usize>, plan: i64 },
    SpanEnd,
}

#[derive(Default)]
struct Lane {
    q: VecDeque<Item>,
    /// Its lag behind its plan, ns.
    l: i64,
    /// The op in flight.
    busy: Option<u64>,
    /// Its previous completion, or the time a gate let it start.
    last: i64,
    /// The latest completion of an edge it waited for.
    edge_t: i64,
    /// Sub-lanes of a joined fork that have ended, their latest completion, and their latest
    /// planned end, since the last join.
    kids_done: usize,
    kids_end: i64,
    kids_plan: i64,
    parent: Option<usize>,
    /// A worker lane: its workers, whose staging its writes draw on.
    workers: Option<usize>,
    /// A worker or eviction lane: a queue, its L reported, not carried.
    queue: bool,
    /// Gates in its queue not yet passed, and whether the latest passed one was a submission's
    /// (a write pushed after it was passed is staged at once).
    gates_queued: usize,
    staging_open: bool,
    timer: Option<i64>,
    ended: bool,
    /// The span it is in: name, planned start, launch, and the workers whose staging its reads
    /// draw on.
    spans: Vec<(Arc<str>, i64, i64, Option<usize>)>,
    /// It has issued an op.
    issued: bool,
}

/// The workers' staging: bytes submitted and not yet written, and the bytes of the loads in
/// flight that draw on it (a span naming it, contract 0.21); its peak, and the would-be drops.
#[derive(Debug, Default, Clone)]
pub struct Staging {
    /// The workers' name.
    pub name: String,
    pub capacity: i64,
    pub held: i64,
    pub peak: i64,
    /// (time, bytes held) each time a submission or a load took it past its capacity.
    pub drops: Vec<(i64, i64)>,
}

/// What the run measured of the executor (step 5 reports it).
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub issued: u64,
    /// Each lane's L at the end, ns, by executor lane.
    pub lag: Vec<i64>,
    /// Ops whose edges made them later than their lane alone would have been, and by how much.
    pub edge_waits: u64,
    pub edge_wait_ns: i64,
    /// Ops issued later than their `ready` (the host's lateness), and the latest.
    pub late: u64,
    pub late_max_ns: i64,
    pub staging: Vec<Staging>,
    /// The most paths the planner held (`Plan::paths`), sampled as it retires them.
    pub paths_peak: usize,
    /// Each span's (planned start, D) by name, ns, in the order they ended.
    pub spans: BTreeMap<Arc<str>, Vec<(i64, i64)>>,
    /// The final L of each lane of the timeline's body that issued an op (a session's; worker
    /// and eviction lanes' L is reported, never carried, so not here).
    pub body_lag: Vec<i64>,
    /// Each op's lateness, its issue less its `ready`: the host's, not the system under test's
    /// (`DESIGN_REVIEW.md` §3.99 point 7; a run is invalid past a p99 of `LATENESS_P99_MAX_NS`).
    pub lateness: crate::run::LatHist,
    /// When a body lane's L first passed the lag ceiling: the time, the lane, and its L.
    pub ceiling: Option<(i64, usize, i64)>,
    /// The first certain failure, and whether the timeline stopped at it.
    pub certain: Option<String>,
    pub stopped: bool,
}

/// A run's host lateness p99 past which it is invalid: 10 ms (`DESIGN_REVIEW.md` §3.99 point 7,
/// the user's choice of 2026-11-14).
pub const LATENESS_P99_MAX_NS: u64 = 10_000_000;

/// A case's constraint that is certain to fail once a window closes above it (the user's choice
/// of 2026-11-14): `result.span.SPAN.window_p99_s` (or `.max_s` when `max`) at most `limit_ns`
/// (below it when `strict`). Its `text` is the constraint's.
#[derive(Debug, Clone, PartialEq)]
pub struct Watch {
    pub span: String,
    pub max: bool,
    pub limit_ns: i64,
    pub strict: bool,
    pub text: String,
}

impl Watch {
    fn lost(&self, value: i64) -> bool {
        if self.strict {
            value >= self.limit_ns
        } else {
            value > self.limit_ns
        }
    }
}

/// What a run of a timeline is told to stop on (`--on-certain-failure`, `--lag-ceiling`, and the
/// watches from a case's constraints).
#[derive(Debug, Clone, Default)]
pub struct TimelineOpts {
    /// Stop at the first certain failure (`--on-certain-failure stop`, the default), or go on.
    pub keep_going: bool,
    pub lag_ceiling_ns: Option<i64>,
    pub watches: Vec<Watch>,
}

pub struct Exec {
    lanes: Vec<Lane>,
    /// Pushed and not completed.
    outstanding: HashSet<u64>,
    /// In flight: lane, and the bytes it holds in a staging and whose.
    flight: HashMap<u64, (usize, i64, Option<usize>)>,
    /// Writes staged at their submission, not yet issued.
    staged: HashMap<u64, i64>,
    waiters: HashMap<u64, Vec<usize>>,
    /// The time each gate opened, and the lanes it starts.
    gates: Vec<(Option<i64>, Vec<usize>)>,
    timers: BinaryHeap<Reverse<(i64, usize)>>,
    dirty: Vec<usize>,
    /// Lanes that have ended since `take_ended`.
    ended: Vec<usize>,
    /// A body lane's L past this is a certain failure (`--lag-ceiling`).
    pub ceiling_ns: Option<i64>,
    /// A constraint is watched: D is kept by window until the window closes.
    pub watching: bool,
    /// The latest planned time pushed, and whether the planner has ended: a window is closed
    /// once it is past and its spans have ended.
    pushed_to: i64,
    /// Spans pushed and not ended, by window of planned start; D by window and span name, kept
    /// until the window closes.
    open_win: HashMap<i64, usize>,
    win_d: BTreeMap<(i64, Arc<str>), Vec<i64>>,
    pub stats: Stats,
}

impl Exec {
    pub fn new() -> Self {
        Exec { lanes: Vec::new(), outstanding: HashSet::new(), flight: HashMap::new(), staged: HashMap::new(), waiters: HashMap::new(), gates: Vec::new(), timers: BinaryHeap::new(), dirty: Vec::new(), ended: Vec::new(), ceiling_ns: None, watching: false, pushed_to: 0, open_win: HashMap::new(), win_d: BTreeMap::new(), stats: Stats::default() }
    }

    fn lane(&mut self, l: usize) -> &mut Lane {
        if l >= self.lanes.len() {
            self.lanes.resize_with(l + 1, Lane::default);
        }
        &mut self.lanes[l]
    }

    /// Workers `w`'s staging, named `name`, of `capacity` bytes.
    fn staging(&mut self, w: usize, name: &str, capacity: i64) {
        if self.stats.staging.len() <= w {
            self.stats.staging.resize_with(w + 1, Staging::default);
        }
        let s = &mut self.stats.staging[w];
        s.capacity = capacity;
        if s.name.is_empty() {
            s.name = name.to_string();
        }
    }

    fn gate(&mut self, lanes: Vec<usize>) -> usize {
        self.gates.push((None, lanes));
        self.gates.len() - 1
    }

    /// The planner's next event, in planned order.
    pub fn push(&mut self, p: Planned) {
        self.pushed_to = self.pushed_to.max(planned_at(&p));
        match p {
            Planned::Op(r) => {
                let l = r.lane;
                self.outstanding.insert(r.seq);
                let lane = self.lane(l);
                let stage_now = lane.staging_open && lane.gates_queued == 0 && r.kind == OpKind::Write;
                let (seq, len, w) = (r.seq, r.len, lane.workers);
                lane.q.push_back(Item::Op(r));
                if stage_now {
                    if let Some(w) = w {
                        self.stage(w, &[(seq, len)], None);
                    }
                }
                self.dirty.push(l);
            }
            Planned::Compute { lane, .. } => {
                self.lane(lane);
            }
            Planned::Fork { parent, kids, join, plan } => {
                let g = self.gate(kids.clone());
                self.lane(parent).q.push_back(Item::Spawn { gate: g, plan });
                for &k in &kids {
                    let kid = self.lane(k);
                    kid.q.push_back(Item::Gate { gate: g, plan, fresh: true, workers: None });
                    kid.gates_queued += 1;
                    if join {
                        kid.parent = Some(parent);
                    }
                }
                if join {
                    let n = kids.len();
                    self.lane(parent).q.push_back(Item::Join { plan, kids: n });
                }
                self.dirty.push(parent);
            }
            Planned::Submit { parent, worker, workers, name, staging, plan } => {
                self.staging(workers, &name, staging);
                let g = self.gate(vec![worker]);
                self.lane(parent).q.push_back(Item::Spawn { gate: g, plan });
                let w = self.lane(worker);
                w.workers = Some(workers);
                w.queue = true;
                w.q.push_back(Item::Gate { gate: g, plan, fresh: false, workers: Some(workers) });
                w.gates_queued += 1;
                self.dirty.push(parent);
            }
            Planned::Evict { parent: Some(parent), lanes, plan } => {
                let g = self.gate(lanes.clone());
                self.lane(parent).q.push_back(Item::Spawn { gate: g, plan });
                for &k in &lanes {
                    let e = self.lane(k);
                    e.queue = true;
                    e.q.push_back(Item::Gate { gate: g, plan, fresh: false, workers: None });
                    e.gates_queued += 1;
                }
                self.dirty.push(parent);
            }
            Planned::Evict { parent: None, lanes, .. } => {
                for k in lanes {
                    self.lane(k).queue = true;
                }
            }
            Planned::End { lane, plan } => {
                self.lane(lane).q.push_back(Item::End { plan });
                self.dirty.push(lane);
            }
            Planned::SpanBegin { lane, name, staging, plan } => {
                *self.open_win.entry(plan.div_euclid(WINDOW_NS)).or_default() += 1;
                if let Some((w, wname, bytes)) = &staging {
                    self.staging(*w, wname, *bytes);
                }
                self.lane(lane).q.push_back(Item::SpanBegin { name, staging: staging.map(|s| s.0), plan });
                self.dirty.push(lane);
            }
            Planned::SpanEnd { lane, .. } => {
                self.lane(lane).q.push_back(Item::SpanEnd);
                self.dirty.push(lane);
            }
        }
    }

    /// Stage a submission's writes (`(seq, bytes)`) on workers `w` at `at` (`None`: now, which a
    /// write pushed after its gate opened is staged at; its gate's time is past).
    fn stage(&mut self, w: usize, writes: &[(u64, i64)], at: Option<i64>) {
        let bytes: i64 = writes.iter().map(|x| x.1).sum();
        for &(s, b) in writes {
            self.staged.insert(s, b);
        }
        let st = &mut self.stats.staging[w];
        st.held += bytes;
        st.peak = st.peak.max(st.held);
        if st.held > st.capacity && bytes > 0 {
            st.drops.push((at.unwrap_or(-1), st.held));
        }
    }

    /// The records to issue at `now`, each once.
    pub fn poll(&mut self, now: i64) -> Vec<Record> {
        while let Some(Reverse((t, l))) = self.timers.peek().copied() {
            if t > now {
                break;
            }
            self.timers.pop();
            if self.lanes[l].timer == Some(t) {
                self.lanes[l].timer = None;
                self.dirty.push(l);
            }
        }
        let mut out = Vec::new();
        while let Some(l) = self.dirty.pop() {
            self.advance(l, now, &mut out);
        }
        out
    }

    /// When to poll next for a lane waiting on its plan, `None` when none is.
    pub fn next_wake(&self) -> Option<i64> {
        self.timers.iter().map(|Reverse((t, _))| *t).min()
    }

    fn wake_at(&mut self, l: usize, t: i64) {
        let lane = &mut self.lanes[l];
        if lane.timer != Some(t) {
            lane.timer = Some(t);
            self.timers.push(Reverse((t, l)));
        }
    }

    /// Take lane `l` as far as it goes at `now`.
    fn advance(&mut self, l: usize, now: i64, out: &mut Vec<Record>) {
        loop {
            let lane = &mut self.lanes[l];
            if lane.busy.is_some() || lane.ended {
                return;
            }
            let Some(item) = lane.q.front() else { return };
            match item {
                Item::Op(r) => {
                    if let Some(&e) = r.edges.iter().find(|e| self.outstanding.contains(e)) {
                        let w = self.waiters.entry(e).or_default();
                        if !w.contains(&l) {
                            w.push(l);
                        }
                        return;
                    }
                    let own = (r.plan + if lane.queue { 0 } else { lane.l }).max(lane.last);
                    let ready = own.max(lane.edge_t);
                    if now < ready {
                        self.wake_at(l, ready);
                        return;
                    }
                    let Some(Item::Op(r)) = lane.q.pop_front() else { unreachable!() };
                    lane.l = lane.l.max(ready - r.plan);
                    self.stats.lateness.add((now - ready).max(0) as u64);
                    if let (false, Some(c), None) = (lane.queue, self.ceiling_ns, self.stats.ceiling) {
                        if lane.l > c {
                            self.stats.ceiling = Some((now, l, lane.l));
                        }
                    }
                    lane.busy = Some(r.seq);
                    lane.edge_t = 0;
                    if ready > own {
                        self.stats.edge_waits += 1;
                        self.stats.edge_wait_ns += ready - own;
                    }
                    if now > ready {
                        self.stats.late += 1;
                        self.stats.late_max_ns = self.stats.late_max_ns.max(now - ready);
                    }
                    self.stats.issued += 1;
                    lane.issued = true;
                    // a write's bytes were staged at its submission; a load's, a read in a span
                    // naming a staging, are from now until it completes
                    let load = if r.kind == OpKind::Read { lane.spans.last().and_then(|s| s.3) } else { None };
                    let held = match load {
                        Some(w) => {
                            let st = &mut self.stats.staging[w];
                            st.held += r.len;
                            st.peak = st.peak.max(st.held);
                            if st.held > st.capacity && r.len > 0 {
                                st.drops.push((now, st.held));
                            }
                            (r.len, Some(w))
                        }
                        None => (self.staged.remove(&r.seq).unwrap_or(0), lane.workers),
                    };
                    self.flight.insert(r.seq, (l, held.0, held.1));
                    out.push(r);
                    return;
                }
                Item::Spawn { gate, plan } => {
                    let fire = (plan + lane.l).max(lane.last);
                    if now < fire {
                        self.wake_at(l, fire);
                        return;
                    }
                    let gate = *gate;
                    lane.l = lane.l.max(fire - plan);
                    lane.q.pop_front();
                    self.gates[gate].0 = Some(fire);
                    let started = self.gates[gate].1.clone();
                    self.dirty.extend(started);
                }
                Item::Gate { gate, plan, fresh, workers } => {
                    let Some(t) = self.gates[*gate].0 else { return };
                    let (plan, fresh, workers) = (*plan, *fresh, *workers);
                    lane.q.pop_front();
                    lane.gates_queued -= 1;
                    if fresh {
                        lane.l = lane.l.max(t - plan);
                    }
                    lane.last = lane.last.max(t);
                    lane.staging_open = workers.is_some();
                    if let Some(w) = workers {
                        // the submission's writes: up to the lane's next gate
                        let writes: Vec<(u64, i64)> = lane
                            .q
                            .iter()
                            .take_while(|i| !matches!(i, Item::Gate { .. }))
                            .filter_map(|i| match i {
                                Item::Op(r) if r.kind == OpKind::Write => Some((r.seq, r.len)),
                                _ => None,
                            })
                            .collect();
                        self.stage(w, &writes, Some(t));
                    }
                }
                Item::Join { plan, kids } => {
                    if lane.kids_done < *kids {
                        return;
                    }
                    let plan = (*plan).max(lane.kids_plan);
                    let fire = (plan + lane.l).max(lane.last).max(lane.kids_end);
                    if now < fire {
                        self.wake_at(l, fire);
                        return;
                    }
                    let Some(Item::Join { kids, .. }) = lane.q.pop_front() else { unreachable!() };
                    lane.l = lane.l.max(fire - plan);
                    lane.last = lane.last.max(fire);
                    lane.kids_done -= kids;
                    lane.kids_end = 0;
                    lane.kids_plan = 0;
                }
                Item::SpanBegin { plan, .. } => {
                    // launched when the lane reaches it: its plan plus L, or its previous completion
                    let launch = (plan + if lane.queue { 0 } else { lane.l }).max(lane.last);
                    let Some(Item::SpanBegin { name, staging, plan }) = lane.q.pop_front() else { unreachable!() };
                    lane.spans.push((name, plan, launch, staging));
                }
                Item::SpanEnd => {
                    lane.q.pop_front();
                    let (name, plan, launch, _) = lane.spans.pop().expect("a span's end follows its begin");
                    let d = lane.last.max(launch) - launch;
                    let w = plan.div_euclid(WINDOW_NS);
                    if let Some(n) = self.open_win.get_mut(&w) {
                        *n -= 1;
                    }
                    if self.watching {
                        self.win_d.entry((w, name.clone())).or_default().push(d);
                    }
                    self.stats.spans.entry(name).or_default().push((plan, d));
                }
                Item::End { plan } => {
                    let plan = *plan;
                    lane.q.pop_front();
                    lane.ended = true;
                    self.ended.push(l);
                    let (parent, last) = (lane.parent, lane.last);
                    if let Some(p) = parent {
                        let p = &mut self.lanes[p];
                        p.kids_done += 1;
                        p.kids_end = p.kids_end.max(last);
                        p.kids_plan = p.kids_plan.max(plan);
                        self.dirty.push(parent.unwrap());
                    }
                    return;
                }
            }
        }
    }

    /// Record `seq` completed at `now`.
    pub fn complete(&mut self, seq: u64, now: i64) {
        self.outstanding.remove(&seq);
        if let Some((l, staged, w)) = self.flight.remove(&seq) {
            let lane = &mut self.lanes[l];
            lane.busy = None;
            lane.last = lane.last.max(now);
            if let (true, Some(w)) = (staged > 0, w) {
                self.stats.staging[w].held -= staged;
            }
            self.dirty.push(l);
        }
        if let Some(ws) = self.waiters.remove(&seq) {
            for w in ws {
                let lane = &mut self.lanes[w];
                lane.edge_t = lane.edge_t.max(now);
                self.dirty.push(w);
            }
        }
    }

    /// The lanes that have ended since the last call: a driver frees what it holds for them (a
    /// lane's task on the event loops, its thread under a blocking backend). Worker and eviction
    /// lanes never end; they go on until the timeline does.
    pub fn take_ended(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.ended)
    }

    /// Take `opts`' lag ceiling, and keep D by window when it watches a constraint.
    pub fn configure(&mut self, opts: &TimelineOpts) {
        self.ceiling_ns = opts.lag_ceiling_ns;
        self.watching = !opts.watches.is_empty();
    }

    /// The windows of planned start that have closed since the last call (the planner past their
    /// end, or ended, and their spans ended): each span name's (window, count, p99, maximum) of D.
    pub fn closed_windows(&mut self, planner_done: bool) -> Vec<(Arc<str>, i64, u64, i64, i64)> {
        let closed: Vec<(i64, Arc<str>)> = self
            .win_d
            .keys()
            .filter(|(w, _)| (planner_done || self.pushed_to >= (w + 1).saturating_mul(WINDOW_NS)) && self.open_win.get(w).copied().unwrap_or(0) == 0)
            .cloned()
            .collect();
        let mut out = Vec::new();
        for k in closed {
            let mut v = self.win_d.remove(&k).unwrap_or_default();
            v.sort_unstable();
            out.push((k.1.clone(), k.0, v.len() as u64, quantile(&v, 0.99), v.last().copied().unwrap_or(0)));
        }
        out.retain(|x| x.2 > 0);
        out
    }

    /// The first certain failure since the run began, if there is one now: a would-be drop, the
    /// lag ceiling crossed, or a watched constraint lost in a window that has closed. Recorded in
    /// the stats once.
    pub fn certain_failure(&mut self, watches: &[Watch], planner_done: bool) -> Option<String> {
        if let Some(c) = &self.stats.certain {
            return Some(c.clone());
        }
        let mut why = None;
        if let Some((at, held, st)) = self.stats.staging.iter().find_map(|s| s.drops.first().map(|d| (d.0, d.1, s))) {
            why = Some(format!("staging of workers `{}` held {held} bytes, past its {}, at {:.3} s into the timeline: a store would have been dropped", st.name, st.capacity, at.max(0) as f64 / 1e9));
        } else if let Some((at, lane, l)) = self.stats.ceiling {
            why = Some(format!("lane {lane}'s lag reached {:.3} s, past the lag ceiling of {:.3} s, at {:.3} s into the timeline", l as f64 / 1e9, self.ceiling_ns.unwrap_or(0) as f64 / 1e9, at as f64 / 1e9));
        } else if !watches.is_empty() {
            for (name, w, n, p99, max) in self.closed_windows(planner_done) {
                for x in watches.iter().filter(|x| x.span == *name) {
                    let v = if x.max { max } else { p99 };
                    if x.lost(v) && why.is_none() {
                        let what = if x.max { "a D of" } else { "a p99 of" };
                        why = Some(format!("constraint `{}` lost: span `{name}`'s window {w} (planned {} s to {} s) closed with {what} {:.3} s over {n} spans", x.text, w * (WINDOW_NS / 1_000_000_000), (w + 1) * (WINDOW_NS / 1_000_000_000), v as f64 / 1e9));
                    }
                }
            }
        }
        if why.is_some() {
            self.stats.certain = why.clone();
        }
        why
    }

    /// Whether `seq` has completed (or was never pushed): for the planner's `retire`.
    pub fn done(&self, seq: u64) -> bool {
        !self.outstanding.contains(&seq)
    }

    /// Ops in flight.
    pub fn in_flight(&self) -> usize {
        self.flight.len()
    }

    /// Nothing queued or in flight.
    pub fn idle(&self) -> bool {
        self.flight.is_empty() && self.lanes.iter().all(|l| l.q.is_empty())
    }

    /// Each lane's L, into `stats.lag`.
    pub fn finish(&mut self) -> Stats {
        self.stats.lag = self.lanes.iter().map(|l| l.l).collect();
        self.stats.body_lag = self.lanes.iter().filter(|l| !l.queue && l.issued).map(|l| l.l).collect();
        self.stats.clone()
    }

    /// A lane that cannot go on, and what it waits for: for a run that ends with work queued.
    pub fn stuck(&self) -> Option<String> {
        // a joined parent waits because a sub-lane does: name the lane at the bottom
        let waiting = |l: &&Lane| !l.q.is_empty() && l.busy.is_none();
        let (i, l) = self.lanes.iter().enumerate().find(|(_, l)| waiting(l) && !matches!(l.q.front(), Some(Item::Join { .. }))).or_else(|| self.lanes.iter().enumerate().find(|(_, l)| waiting(l)))?;
        let what = match l.q.front()? {
            Item::Op(r) => format!("op {} on `{}`, its edges {:?} of which outstanding {:?}", r.seq, r.path, r.edges, r.edges.iter().filter(|e| self.outstanding.contains(e)).collect::<Vec<_>>()),
            Item::Spawn { gate, .. } => format!("its fork, submission, or eviction batch (gate {gate})"),
            Item::Gate { gate, .. } => format!("gate {gate}, never opened"),
            Item::Join { kids, .. } => format!("its {} sub-lanes, {} ended", kids, l.kids_done),
            Item::End { .. } => "its end".into(),
            Item::SpanBegin { .. } | Item::SpanEnd => "a span's mark".into(),
        };
        let mut out = format!("lane {i} waits on {what}");
        // an outstanding edge: the lane that holds it, and what that lane waits on
        if let Some(Item::Op(r)) = l.q.front() {
            let held = r.edges.iter().filter(|e| self.outstanding.contains(e)).find_map(|e| self.lanes.iter().enumerate().find(|(_, m)| m.q.iter().any(|x| matches!(x, Item::Op(o) if o.seq == *e))).map(|(j, m)| (*e, j, m)));
            if let Some((e, j, m)) = held {
                let head = match m.q.front() {
                    Some(Item::Gate { gate, .. }) => format!("gate {gate}, never opened"),
                    Some(Item::Op(o)) => format!("op {}", o.seq),
                    _ => "a lane event".into(),
                };
                out += &format!("; op {e} is queued on lane {j}, which waits on {head}");
            }
        }
        Some(out)
    }
}

/// How far ahead of the wall clock a driver pulls the planner (`DESIGN_REVIEW.md` §3.99, the note
/// "The run side of timelines": a horizon of planned time).
pub const HORIZON_NS: i64 = 10_000_000_000;

/// The paths a driver lets the planner hold before it first retires them.
pub const RETIRE_FLOOR: usize = 1024;

/// The planned time of a planner's event.
pub fn planned_at(p: &Planned) -> i64 {
    match p {
        Planned::Op(r) => r.plan,
        Planned::Compute { plan, .. } | Planned::Fork { plan, .. } | Planned::Submit { plan, .. } | Planned::Evict { plan, .. } | Planned::End { plan, .. } => *plan,
        Planned::SpanBegin { plan, .. } | Planned::SpanEnd { plan, .. } => *plan,
    }
}

/// The window D is judged by, a span assigned to one by its planned start (`DESIGN_REVIEW.md`
/// §3.99, the note "The run side of timelines", point 3): 5 minutes.
pub const WINDOW_NS: i64 = 300_000_000_000;

/// A span name's D, summarized: count, sum, nearest-rank p50 and p99, maximum; each window's
/// (index, count, p99), by planned start; and the worst window's p99.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpanSummary {
    pub count: u64,
    pub sum: i64,
    pub p50: i64,
    pub p99: i64,
    pub max: i64,
    pub windows: Vec<(i64, u64, i64)>,
    pub window_p99: i64,
}

/// The nearest-rank quantile `q` of `sorted` (ascending); 0 when empty.
pub fn quantile(sorted: &[i64], q: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// Summarize spans given as (planned start, D).
pub fn summarize(ds: &[(i64, i64)]) -> SpanSummary {
    let mut all: Vec<i64> = ds.iter().map(|(_, d)| *d).collect();
    all.sort_unstable();
    let mut by: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for (p, d) in ds {
        by.entry(p.div_euclid(WINDOW_NS)).or_default().push(*d);
    }
    let windows: Vec<(i64, u64, i64)> = by
        .into_iter()
        .map(|(w, mut v)| {
            v.sort_unstable();
            (w, v.len() as u64, quantile(&v, 0.99))
        })
        .collect();
    SpanSummary {
        count: all.len() as u64,
        sum: all.iter().sum(),
        p50: quantile(&all, 0.5),
        p99: quantile(&all, 0.99),
        max: all.last().copied().unwrap_or(0),
        window_p99: windows.iter().map(|w| w.2).max().unwrap_or(0),
        windows,
    }
}

/// One op as `simulate` issued it: the record, and when it was issued and completed.
#[derive(Debug, Clone)]
pub struct Simulated {
    pub record: Record,
    pub issued: i64,
    pub completed: i64,
}

/// Run `plan` through an executor on a simulated clock: each op completes `latency(record,
/// issued)` ns after it is issued; the planner is pulled `horizon` ns of planned time ahead
/// of the clock. Returns the ops in the order issued, and the executor's stats.
pub fn simulate(plan: &mut Plan, horizon: i64, latency: impl FnMut(&Record, i64) -> i64) -> Result<(Vec<Simulated>, Stats)> {
    simulate_with(plan, horizon, &TimelineOpts { keep_going: true, ..Default::default() }, latency)
}

/// `simulate` told what to stop on: at the first certain failure, unless `opts.keep_going`, the
/// planner is left and no record is issued again; what is in flight completes.
pub fn simulate_with(plan: &mut Plan, horizon: i64, opts: &TimelineOpts, mut latency: impl FnMut(&Record, i64) -> i64) -> Result<(Vec<Simulated>, Stats)> {
    let mut ex = Exec::new();
    ex.configure(opts);
    let mut stopping = false;
    let mut now = 0i64;
    let mut ahead: Option<Planned> = plan.next()?;
    let mut flight: BinaryHeap<Reverse<(i64, u64)>> = BinaryHeap::new();
    let mut issued: HashMap<u64, usize> = HashMap::new();
    let mut out: Vec<Simulated> = Vec::new();
    // the planner's paths are retired when they have doubled since the last time
    let mut keep = RETIRE_FLOOR;
    let mut peak = 0;
    loop {
        while let Some(p) = ahead.take() {
            if planned_at(&p) > now.saturating_add(horizon) {
                ahead = Some(p);
                break;
            }
            ex.push(p);
            ahead = plan.next()?;
        }
        if !stopping {
            for r in ex.poll(now) {
                let done = now + latency(&r, now).max(0);
                flight.push(Reverse((done, r.seq)));
                issued.insert(r.seq, out.len());
                out.push(Simulated { record: r, issued: now, completed: -1 });
            }
            if ex.certain_failure(&opts.watches, ahead.is_none()).is_some() && !opts.keep_going {
                stopping = true;
                ex.stats.stopped = true;
                ahead = None;
            }
        }
        if (stopping || (ahead.is_none() && ex.idle())) && flight.is_empty() {
            break;
        }
        if stopping {
            // what is in flight completes; nothing more issues
            let Some(Reverse((t, seq))) = flight.pop() else { break };
            now = t.max(now);
            ex.complete(seq, now);
            let i = issued.remove(&seq).expect("an op in flight was issued");
            out[i].completed = now;
            continue;
        }
        let mut next = [flight.peek().map(|Reverse((t, _))| *t), ex.next_wake(), ahead.as_ref().map(|p| planned_at(p).saturating_sub(horizon))]
            .into_iter()
            .flatten()
            .min();
        if next.is_none() {
            anyhow::bail!("the executor stopped: {}", ex.stuck().unwrap_or_else(|| "no lane able to go on".into()));
        }
        let t = next.take().unwrap().max(now);
        now = t;
        while let Some(Reverse((t, seq))) = flight.peek().copied() {
            if t > now {
                break;
            }
            flight.pop();
            ex.complete(seq, now);
            let i = issued.remove(&seq).expect("an op in flight was issued");
            out[i].completed = now;
        }
        if plan.paths() >= keep {
            peak = peak.max(plan.paths());
            plan.retire(|s| ex.done(s));
            keep = (2 * plan.paths()).max(RETIRE_FLOOR);
        }
    }
    // the last windows close with the plan
    ex.certain_failure(&opts.watches, true);
    let mut st = ex.finish();
    st.paths_peak = peak.max(plan.paths());
    Ok((out, st))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::OwnedCtx;
    use std::sync::Arc;

    fn rec(seq: u64, lane: usize, plan: i64, kind: OpKind, edges: &[u64]) -> Planned {
        Planned::Op(Record {
            seq, lane, plan, outside: false, kind, path: Arc::from(format!("p{seq}")), path2: None, offset: 0, len: 4096, bytes: 4096, aux: 0,
            expect: vec![], positioned: true, seed: 0, capped: false,
            ctx: OwnedCtx { template: "gpu".into(), actor: 0, indices: vec![], phase: None }, edges: edges.to_vec(),
        })
    }

    /// An executor that cannot go on names the lane and what it waits for, down to a gate never
    /// opened: lane 0 waits on op 1, queued on worker lane 1 behind a submission lane 0 has not
    /// reached (a plan that could not be: the planner's edges point to earlier ops).
    #[test]
    fn a_stuck_executor_says_what_waits_on_what() {
        let mut ex = Exec::new();
        ex.push(rec(0, 0, 0, OpKind::Stat, &[1]));
        ex.push(Planned::Submit { parent: 0, worker: 1, workers: 0, name: "w".into(), staging: 1 << 20, plan: 0 });
        ex.push(rec(1, 1, 0, OpKind::Write, &[]));
        assert!(ex.poll(0).is_empty());
        assert_eq!(ex.in_flight(), 0);
        assert!(!ex.idle());
        assert_eq!(ex.stuck().unwrap(), "lane 0 waits on op 0 on `p0`, its edges [1] of which outstanding [1]; op 1 is queued on lane 1, which waits on gate 0, never opened");
    }

    /// A lane's end is said once, after its last op completed; a lane with no end (a worker
    /// lane) is never said to have ended.
    #[test]
    fn ended_lanes_are_taken_once() {
        let mut ex = Exec::new();
        ex.push(Planned::Fork { parent: 0, kids: vec![1], join: true, plan: 0 });
        ex.push(rec(0, 1, 0, OpKind::Stat, &[]));
        ex.push(Planned::End { lane: 1, plan: 0 });
        ex.push(Planned::Submit { parent: 0, worker: 2, workers: 0, name: "w".into(), staging: 1 << 20, plan: 0 });
        ex.push(rec(1, 2, 0, OpKind::Stat, &[]));
        ex.push(Planned::End { lane: 0, plan: 0 });
        let r = ex.poll(0);
        assert_eq!(r.len(), 1);
        assert!(ex.take_ended().is_empty(), "lane 1 has an op in flight");
        ex.complete(r[0].seq, 1);
        // lane 1 ends; its parent joins it, submits, and ends; the worker lane does not
        let r = ex.poll(1);
        assert_eq!(r.len(), 1);
        assert_eq!(ex.take_ended(), [1, 0]);
        assert!(ex.take_ended().is_empty());
        ex.complete(r[0].seq, 2);
        assert!(ex.poll(2).is_empty());
        assert!(ex.take_ended().is_empty());
        assert!(ex.idle());
    }

    /// A write that reaches its worker lane after its submission's gate has opened (a planner
    /// pulled behind the clock) is staged as it arrives.
    #[test]
    fn a_write_pushed_after_its_gate_opened_is_staged_at_once() {
        let mut ex = Exec::new();
        ex.push(Planned::Submit { parent: 0, worker: 1, workers: 0, name: "w".into(), staging: 4096, plan: 0 });
        assert!(ex.poll(0).is_empty());
        ex.push(rec(0, 1, 0, OpKind::Write, &[]));
        ex.push(rec(1, 1, 0, OpKind::Write, &[]));
        assert_eq!(ex.stats.staging[0].held, 8192);
        assert_eq!(ex.stats.staging[0].drops, [(-1, 8192)]);
        let r = ex.poll(0);
        assert_eq!(r.len(), 1);
        ex.complete(r[0].seq, 5);
        let r = ex.poll(5);
        ex.complete(r[0].seq, 9);
        assert_eq!(ex.stats.staging[0].held, 0);
        assert!(ex.idle());
    }
}
