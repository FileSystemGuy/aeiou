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
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

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
}

/// The workers' staging: bytes submitted and not yet written, its peak, and the would-be drops.
#[derive(Debug, Default, Clone)]
pub struct Staging {
    pub capacity: i64,
    pub held: i64,
    pub peak: i64,
    /// (time, bytes held) each time a submission took it past its capacity.
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
}

pub struct Exec {
    lanes: Vec<Lane>,
    /// Pushed and not completed.
    outstanding: HashSet<u64>,
    /// In flight: lane, and the bytes it staged.
    flight: HashMap<u64, (usize, i64)>,
    /// Writes staged at their submission, not yet issued.
    staged: HashMap<u64, i64>,
    waiters: HashMap<u64, Vec<usize>>,
    /// The time each gate opened, and the lanes it starts.
    gates: Vec<(Option<i64>, Vec<usize>)>,
    timers: BinaryHeap<Reverse<(i64, usize)>>,
    dirty: Vec<usize>,
    pub stats: Stats,
}

impl Exec {
    pub fn new() -> Self {
        Exec { lanes: Vec::new(), outstanding: HashSet::new(), flight: HashMap::new(), staged: HashMap::new(), waiters: HashMap::new(), gates: Vec::new(), timers: BinaryHeap::new(), dirty: Vec::new(), stats: Stats::default() }
    }

    fn lane(&mut self, l: usize) -> &mut Lane {
        if l >= self.lanes.len() {
            self.lanes.resize_with(l + 1, Lane::default);
        }
        &mut self.lanes[l]
    }

    fn gate(&mut self, lanes: Vec<usize>) -> usize {
        self.gates.push((None, lanes));
        self.gates.len() - 1
    }

    /// The planner's next event, in planned order.
    pub fn push(&mut self, p: Planned) {
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
            Planned::Submit { parent, worker, workers, staging, plan } => {
                if self.stats.staging.len() <= workers {
                    self.stats.staging.resize_with(workers + 1, Staging::default);
                }
                self.stats.staging[workers].capacity = staging;
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
                    let staged = self.staged.remove(&r.seq).unwrap_or(0);
                    self.flight.insert(r.seq, (l, staged));
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
                Item::End { plan } => {
                    let plan = *plan;
                    lane.q.pop_front();
                    lane.ended = true;
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
        if let Some((l, staged)) = self.flight.remove(&seq) {
            let lane = &mut self.lanes[l];
            lane.busy = None;
            lane.last = lane.last.max(now);
            if staged > 0 {
                if let Some(w) = lane.workers {
                    self.stats.staging[w].held -= staged;
                }
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

/// The paths a driver lets the planner hold before it first retires them.
pub const RETIRE_FLOOR: usize = 1024;

/// The planned time of a planner's event.
pub fn planned_at(p: &Planned) -> i64 {
    match p {
        Planned::Op(r) => r.plan,
        Planned::Compute { plan, .. } | Planned::Fork { plan, .. } | Planned::Submit { plan, .. } | Planned::Evict { plan, .. } | Planned::End { plan, .. } => *plan,
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
pub fn simulate(plan: &mut Plan, horizon: i64, mut latency: impl FnMut(&Record, i64) -> i64) -> Result<(Vec<Simulated>, Stats)> {
    let mut ex = Exec::new();
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
        for r in ex.poll(now) {
            let done = now + latency(&r, now).max(0);
            flight.push(Reverse((done, r.seq)));
            issued.insert(r.seq, out.len());
            out.push(Simulated { record: r, issued: now, completed: -1 });
        }
        if ahead.is_none() && flight.is_empty() && ex.idle() {
            break;
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
        ex.push(Planned::Submit { parent: 0, worker: 1, workers: 0, staging: 1 << 20, plan: 0 });
        ex.push(rec(1, 1, 0, OpKind::Write, &[]));
        assert!(ex.poll(0).is_empty());
        assert_eq!(ex.in_flight(), 0);
        assert!(!ex.idle());
        assert_eq!(ex.stuck().unwrap(), "lane 0 waits on op 0 on `p0`, its edges [1] of which outstanding [1]; op 1 is queued on lane 1, which waits on gate 0, never opened");
    }

    /// A write that reaches its worker lane after its submission's gate has opened (a planner
    /// pulled behind the clock) is staged as it arrives.
    #[test]
    fn a_write_pushed_after_its_gate_opened_is_staged_at_once() {
        let mut ex = Exec::new();
        ex.push(Planned::Submit { parent: 0, worker: 1, workers: 0, staging: 4096, plan: 0 });
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
