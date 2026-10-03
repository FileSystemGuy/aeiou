//! `aeiou dry-run --metrics`: the locality metrics of the abstract's op stream
//! (`GRAMMAR_OPTIONS.md` §5.4, `runner/README.md` §10). The fingerprint says which stream
//! ran; these say whether it is the right one, by comparison with the same numbers taken
//! from a trace of the real application.
//!
//! Two kinds of metric, kept apart because only one kind has an order to stand on:
//! - **order-free** (sums over the op multiset, the same for every interleaving): the
//!   read/write mix, request sizes, popularity of blocks and of objects, `parallel` fan-out;
//! - **order-dependent** (reuse distance, sequential runs, dependency depth): taken over one
//!   actor instance at a time, in the *round-robin order* of its concurrent sub-actors: each
//!   live sub-actor of a fork issues one op per turn, in index order; a `parallel` nested in a
//!   sub-actor is that sub-actor's turn; the forking line waits for a `parallel` and takes
//!   turns beside a `loader`'s workers. It is one linearization of the partial order the
//!   abstract defines, chosen because it needs no timing. Reuse between instances is not
//!   ordered at all and is reported only as popularity.
//!
//! This is an analysis pass, not the runner: it keeps one entry per distinct block touched
//! (`--metrics-sample N` keeps one block and one object in `N`, chosen by hash, and scales:
//! SHARDS, Waldspurger et al., FAST '15). Nothing here is hashed into the fingerprint.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use xxhash_rust::xxh3::xxh3_64;

use crate::dryrun::{human_bytes, DryRun};
use crate::eval::Model;
use crate::rng::mix64;
use crate::vm::{Control, Event, ForkKind, Op, OpKind, Sink, Vm};

/// The most entries (distinct sampled blocks or objects) one map may hold before the pass
/// stops and asks for `--metrics-sample`.
pub const MAX_ENTRIES: usize = 1 << 26;

#[derive(Debug, Clone, Copy)]
pub struct Opts {
    /// Block size in bytes of the reuse-distance and popularity units.
    pub block: u64,
    /// Keep one block (and one object) in `sample`, by hash; 1 is exact.
    pub sample: u64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { block: 4096, sample: 1 }
    }
}

// ---------------------------------------------------------------- histogram

/// A histogram over quarter-octave buckets: a value `v ≥ 4` lands in the bucket whose lower
/// bound keeps its top three bits, so a bound is at most 25 % below the value and powers of
/// two are exact.
#[derive(Debug, Clone, Default)]
pub struct LogHist {
    pub n: u64,
    pub sum: u128,
    pub max: u64,
    buckets: BTreeMap<u16, u64>,
}

impl LogHist {
    fn bucket(v: u64) -> u16 {
        if v < 4 {
            v as u16
        } else {
            let e = 63 - v.leading_zeros() as u16;
            (e << 2) | ((v >> (e - 2)) & 3) as u16
        }
    }

    pub fn lower(bucket: u16) -> u64 {
        if bucket < 4 {
            bucket as u64
        } else {
            (4 + (bucket & 3) as u64) << ((bucket >> 2) - 2)
        }
    }

    pub fn add(&mut self, v: u64, weight: u64) {
        self.n += weight;
        self.sum += v as u128 * weight as u128;
        self.max = self.max.max(v);
        *self.buckets.entry(Self::bucket(v)).or_insert(0) += weight;
    }

    pub fn merge(&mut self, o: &LogHist) {
        self.n += o.n;
        self.sum += o.sum;
        self.max = self.max.max(o.max);
        for (b, w) in &o.buckets {
            *self.buckets.entry(*b).or_insert(0) += w;
        }
    }

    /// The lower bound of the bucket holding the `p` quantile (0 < p ≤ 1).
    pub fn quantile(&self, p: f64) -> u64 {
        let want = ((self.n as f64) * p).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (b, w) in &self.buckets {
            seen += w;
            if seen >= want {
                return Self::lower(*b);
            }
        }
        0
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.sum as f64 / self.n as f64 }
    }

    /// `(lower bound, weight)` per bucket, ascending.
    pub fn rows(&self) -> Vec<(u64, u64)> {
        self.buckets.iter().map(|(b, w)| (Self::lower(*b), *w)).collect()
    }
}

// ---------------------------------------------------------------- the accumulated metrics

/// Keys are already hashes.
#[derive(Default, Clone)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("IdHasher hashes u64 keys only")
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type IdMap<V> = HashMap<u64, V, BuildHasherDefault<IdHasher>>;

const R: usize = 0;
const W: usize = 1;
const KINDS: [&str; 2] = ["read", "write"];

/// Sequential runs of one kind: a run is the consecutive ops of that kind by one sequential
/// context (a main line or one sub-actor) on one file, each starting where the previous ended.
#[derive(Debug, Clone, Default)]
pub struct Runs {
    /// Run lengths in bytes, one sample per run.
    pub bytes: LogHist,
    /// Run lengths in ops, one sample per run.
    pub ops: LogHist,
    /// Bytes in runs of two ops or more.
    pub multi_op_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub opts_block: u64,
    pub opts_sample: u64,
    /// Requested length per op, `[read, write]`.
    pub request: [LogHist; 2],
    pub runs: [Runs; 2],
    /// Reuse distance in bytes per block access, `[previous access][this access]`: the
    /// distinct blocks the instance touched since its last access to this one, this one
    /// included, times the block size: the smallest LRU cache in which the access hits.
    pub reuse: [[LogHist; 2]; 2],
    /// Block accesses with no earlier access by the same instance, `[read, write]`.
    pub first: [u64; 2],
    /// `parallel` width → number of forks.
    pub fan_out: BTreeMap<u64, u64>,
    /// Consecutive `parallel`s of one context with no op or control of its own between → count.
    pub depth: BTreeMap<u64, u64>,
    /// Accesses per sampled block and per sampled object (data ops), over all instances.
    blocks: IdMap<u64>,
    objects: IdMap<u64>,
    inst: Instance,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instance({} blocks)", self.last.len())
    }
}

#[derive(Clone, Copy)]
struct Slot {
    time: u32,
    kind: u8,
    count: u32,
}

/// One instance's access history: the last access time of every block it has touched and a
/// Fenwick tree with a mark at each of those times, so the marks after a time are the
/// distinct blocks touched since (Bennett and Kruskal's stack-distance algorithm).
#[derive(Clone, Default)]
struct Instance {
    last: IdMap<Slot>,
    tree: Vec<u32>,
    now: usize,
}

impl Instance {
    fn add(&mut self, i: usize, up: bool) {
        let mut i = i + 1;
        while i <= self.tree.len() {
            if up {
                self.tree[i - 1] += 1;
            } else {
                self.tree[i - 1] -= 1;
            }
            i += i & i.wrapping_neg();
        }
    }

    /// Marks at times `0..=i`.
    fn prefix(&self, i: usize) -> u64 {
        let mut i = i + 1;
        let mut s = 0u64;
        while i > 0 {
            s += self.tree[i - 1] as u64;
            i -= i & i.wrapping_neg();
        }
        s
    }

    /// Renumber the live times `0..n` and size the tree for as many again.
    fn compact(&mut self) {
        let mut live: Vec<(u32, u64)> = self.last.iter().map(|(k, s)| (s.time, *k)).collect();
        live.sort_unstable();
        let cap = (live.len() * 2).max(1024).next_power_of_two();
        self.tree.clear();
        self.tree.resize(cap, 0);
        for (t, (_, k)) in live.iter().enumerate() {
            self.last.get_mut(k).unwrap().time = t as u32;
        }
        // a full prefix of ones: tree[i-1] covers lowbit(i) positions, all marked up to n
        let n = live.len();
        for i in 1..=cap {
            let low = i & i.wrapping_neg();
            let start = i - low;
            self.tree[i - 1] = n.saturating_sub(start).min(low) as u32;
        }
        self.now = n;
    }

    /// One access to block `key`: the distinct blocks since its last access (itself included)
    /// and that access's kind, or `None` on first touch.
    fn touch(&mut self, key: u64, kind: usize) -> Option<(u64, usize)> {
        if self.now == self.tree.len() {
            self.compact();
        }
        let now = self.now;
        let live = self.last.len() as u64;
        let out = match self.last.get_mut(&key) {
            Some(s) => {
                let (t, k) = (s.time as usize, s.kind as usize);
                s.time = now as u32;
                s.kind = kind as u8;
                s.count += 1;
                let d = live - self.prefix(t) + 1;
                self.add(t, false);
                Some((d, k))
            }
            None => {
                self.last.insert(key, Slot { time: now as u32, kind: kind as u8, count: 1 });
                None
            }
        };
        self.add(now, true);
        self.now += 1;
        out
    }
}

fn block_key(path: u64, block: u64) -> u64 {
    mix64(path ^ block.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

impl Metrics {
    pub fn new(o: Opts) -> Self {
        Metrics { opts_block: o.block.max(1), opts_sample: o.sample.max(1), ..Default::default() }
    }

    /// A data op: its request size, its object, and every block it transfers.
    fn data(&mut self, kind: usize, path: u64, offset: i64, len: i64, bytes: i64) -> Result<()> {
        self.request[kind].add(len.max(0) as u64, 1);
        let n = self.opts_sample;
        if mix64(path) % n == 0 {
            *self.objects.entry(path).or_insert(0) += 1;
        }
        if bytes <= 0 || offset < 0 {
            return Ok(());
        }
        let b = self.opts_block;
        let (first, last) = (offset as u64 / b, (offset as u64 + bytes as u64 - 1) / b);
        for blk in first..=last {
            let key = block_key(path, blk);
            if key % n != 0 {
                continue;
            }
            match self.inst.touch(key, kind) {
                Some((d, prev)) => self.reuse[prev][kind].add(d.saturating_mul(n).saturating_mul(b), n),
                None => self.first[kind] += n,
            }
        }
        if self.inst.last.len() > MAX_ENTRIES {
            bail!("--metrics: more than {MAX_ENTRIES} distinct blocks in one instance; use --metrics-sample N or a larger --metrics-block");
        }
        Ok(())
    }

    fn run_done(&mut self, kind: usize, bytes: u64, ops: u64) {
        let r = &mut self.runs[kind];
        r.bytes.add(bytes, 1);
        r.ops.add(ops, 1);
        if ops >= 2 {
            r.multi_op_bytes += bytes;
        }
    }

    /// The instance has ended: its block counts join the popularity map, its history goes.
    fn end_instance(&mut self) -> Result<()> {
        for (k, s) in self.inst.last.drain() {
            *self.blocks.entry(k).or_insert(0) += s.count as u64;
        }
        self.inst.tree.clear();
        self.inst.now = 0;
        self.check_size()
    }

    fn check_size(&self) -> Result<()> {
        if self.blocks.len() > MAX_ENTRIES || self.objects.len() > MAX_ENTRIES {
            bail!("--metrics: more than {MAX_ENTRIES} distinct blocks or objects; use --metrics-sample N or a larger --metrics-block");
        }
        Ok(())
    }

    pub fn merge(&mut self, o: &Metrics) {
        for k in 0..2 {
            self.request[k].merge(&o.request[k]);
            self.runs[k].bytes.merge(&o.runs[k].bytes);
            self.runs[k].ops.merge(&o.runs[k].ops);
            self.runs[k].multi_op_bytes += o.runs[k].multi_op_bytes;
            self.first[k] += o.first[k];
            for j in 0..2 {
                self.reuse[k][j].merge(&o.reuse[k][j]);
            }
        }
        for (k, v) in &o.fan_out {
            *self.fan_out.entry(*k).or_insert(0) += v;
        }
        for (k, v) in &o.depth {
            *self.depth.entry(*k).or_insert(0) += v;
        }
        for (k, v) in &o.blocks {
            *self.blocks.entry(*k).or_insert(0) += v;
        }
        for (k, v) in &o.objects {
            *self.objects.entry(*k).or_insert(0) += v;
        }
    }

    pub fn popularity_blocks(&self) -> Popularity {
        Popularity::of(&self.blocks, self.opts_sample)
    }

    pub fn popularity_objects(&self) -> Popularity {
        Popularity::of(&self.objects, self.opts_sample)
    }
}

/// The rank–frequency curve as counts of counts: `(accesses, items with that many)`,
/// descending by accesses. `distinct` and `accesses` are scaled by the sample rate.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Popularity {
    pub distinct: u64,
    pub accesses: u64,
    pub max: u64,
    /// Share of the accesses that go to the most popular 0.1 %, 1 %, and 10 % of the items.
    pub top_0_1_pct: f64,
    pub top_1_pct: f64,
    pub top_10_pct: f64,
    pub counts: Vec<(u64, u64)>,
}

impl Popularity {
    fn of(map: &IdMap<u64>, sample: u64) -> Popularity {
        let mut cc: BTreeMap<u64, u64> = BTreeMap::new();
        for v in map.values() {
            *cc.entry(*v).or_insert(0) += 1;
        }
        let counts: Vec<(u64, u64)> = cc.into_iter().rev().collect();
        let items: u64 = counts.iter().map(|(_, n)| n).sum();
        let total: u64 = counts.iter().map(|(c, n)| c * n).sum();
        let share = |frac: f64| -> f64 {
            if total == 0 {
                return 0.0;
            }
            // the top `frac` of the items, a fractional item taken pro rata
            let mut left = items as f64 * frac;
            let mut got = 0.0;
            for (c, n) in &counts {
                let take = left.min(*n as f64);
                got += take * *c as f64;
                left -= take;
                if left <= 0.0 {
                    break;
                }
            }
            got / total as f64
        };
        Popularity {
            distinct: items * sample,
            accesses: total * sample,
            max: counts.first().map_or(0, |(c, _)| *c),
            top_0_1_pct: share(0.001),
            top_1_pct: share(0.01),
            top_10_pct: share(0.1),
            counts,
        }
    }
}

// ---------------------------------------------------------------- the round-robin walk

#[derive(Clone, Copy)]
enum Role {
    Main,
    Parallel,
    Worker { b: i64, workers: i64, batches: i64 },
}

#[derive(Clone, Copy, Default)]
struct Run {
    end: i64,
    bytes: u64,
    ops: u64,
}

/// One sequential context: a main line or a sub-actor, with the sub-actors it has forked.
struct Stream<'m, 'a> {
    vm: Vm<'m, 'a>,
    role: Role,
    /// A `parallel`'s sub-actors; while any is live this context waits. Kept for the next fork.
    par: Vec<Stream<'m, 'a>>,
    par_n: usize,
    par_at: usize,
    /// `loader` workers, which take turns beside this context.
    bg: Vec<Stream<'m, 'a>>,
    at: usize,
    own_done: bool,
    done: bool,
    /// Open runs by `(path hash, kind)`.
    runs: HashMap<(u64, usize), Run>,
    chain: u64,
}

impl<'m, 'a: 'm> Stream<'m, 'a> {
    fn new(vm: Vm<'m, 'a>, role: Role) -> Self {
        Stream { vm, role, par: Vec::new(), par_n: 0, par_at: 0, bg: Vec::new(), at: 0, own_done: false, done: false, runs: HashMap::new(), chain: 0 }
    }

    fn end_chain(&mut self, m: &mut Metrics) {
        if self.chain > 0 {
            *m.depth.entry(self.chain).or_insert(0) += 1;
            self.chain = 0;
        }
    }

    fn end_runs(&mut self, m: &mut Metrics) {
        for ((_, k), r) in self.runs.drain() {
            m.run_done(k, r.bytes, r.ops);
        }
    }

    fn record(runs: &mut HashMap<(u64, usize), Run>, m: &mut Metrics, op: &Op) -> Result<()> {
        let kind = match op.kind {
            OpKind::Read => R,
            OpKind::Write => W,
            OpKind::Close => {
                let p = xxh3_64(op.path.as_bytes());
                for k in 0..2 {
                    if let Some(r) = runs.remove(&(p, k)) {
                        m.run_done(k, r.bytes, r.ops);
                    }
                }
                return Ok(());
            }
            _ => return Ok(()),
        };
        let path = xxh3_64(op.path.as_bytes());
        let bytes = if kind == R { op.bytes } else { op.len };
        m.data(kind, path, op.offset, op.len, bytes)?;
        if bytes > 0 {
            let r = runs.entry((path, kind)).or_default();
            if r.ops > 0 && r.end != op.offset {
                m.run_done(kind, r.bytes, r.ops);
                *r = Run::default();
            }
            r.end = op.offset + bytes;
            r.bytes += bytes as u64;
            r.ops += 1;
        }
        Ok(())
    }

    /// Issue one op from this context or its sub-actors; `false` when all of it has ended.
    fn turn(&mut self, sink: &mut DryRun, m: &mut Metrics) -> Result<bool> {
        let n = self.bg.len() + 1;
        for i in 0..n {
            let slot = (self.at + i) % n;
            let issued = if slot == 0 {
                if self.own_done {
                    false
                } else if self.own_turn(sink, m)? {
                    true
                } else {
                    self.own_done = true;
                    self.end_chain(m);
                    self.end_runs(m);
                    false
                }
            } else {
                let kid = &mut self.bg[slot - 1];
                !kid.done && kid.turn(sink, m)?
            };
            if issued {
                // a loader forked during this turn has grown `bg`; the modulus is taken next turn
                self.at = slot + 1;
                return Ok(true);
            }
        }
        if self.bg.len() + 1 > n {
            // workers forked by the line's last statements: they still have their turns
            self.at = n;
            return self.turn(sink, m);
        }
        self.done = true;
        Ok(false)
    }

    fn own_turn(&mut self, sink: &mut DryRun, m: &mut Metrics) -> Result<bool> {
        let mut traced: Option<(std::sync::Arc<crate::trace::TraceFile>, crate::trace::OwnedCtx)> = None;
        loop {
            if let Some((tf, base)) = traced.take() {
                // a `trace` node: the whole file in its line order, a lane as a context; it
                // is one turn, since its order is the point
                trace_walk(&tf, &base, sink, m)?;
                self.end_chain(m);
                return Ok(true);
            }
            if self.par_n > 0 {
                let n = self.par_n;
                for i in 0..n {
                    let k = (self.par_at + i) % n;
                    let kid = &mut self.par[k];
                    if !kid.done && kid.turn(sink, m)? {
                        self.par_at = k + 1;
                        return Ok(true);
                    }
                }
                self.par_n = 0;
            }
            match self.vm.next()? {
                None => match &mut self.role {
                    Role::Worker { b, workers, batches } => {
                        *b += *workers;
                        if *b >= *batches {
                            return Ok(false);
                        }
                        let b = *b;
                        self.vm.start_sub(b);
                    }
                    _ => return Ok(false),
                },
                Some(Event::Op(op, ctx)) => {
                    sink.op(&op, &ctx)?;
                    Self::record(&mut self.runs, m, &op)?;
                    self.end_chain(m);
                    return Ok(true);
                }
                Some(Event::Control(c, ctx)) => {
                    sink.control(c, &ctx)?;
                    if !matches!(c, Control::Channel { .. }) {
                        self.end_chain(m);
                    }
                }
                Some(Event::Trace(tf, ctx)) => {
                    traced = Some((tf, crate::trace::OwnedCtx::of(&ctx)));
                }
                Some(Event::Fork(kind)) => {
                    let snap = self.vm.snapshot();
                    self.vm.accept_fork();
                    match kind {
                        ForkKind::Parallel { width, .. } => {
                            *m.fan_out.entry(width.max(0) as u64).or_insert(0) += 1;
                            self.chain += 1;
                            let width = width.max(0) as usize;
                            for k in 0..width {
                                if k < self.par.len() {
                                    let kid = &mut self.par[k];
                                    kid.vm.resume_from(&snap);
                                    kid.role = Role::Parallel;
                                    kid.par_n = 0;
                                    kid.bg.clear();
                                    kid.at = 0;
                                    kid.own_done = false;
                                    kid.done = false;
                                    kid.chain = 0;
                                } else {
                                    self.par.push(Stream::new(Vm::resume(snap.clone()), Role::Parallel));
                                }
                                self.par[k].vm.start_sub(k as i64);
                            }
                            self.par_n = width;
                            self.par_at = 0;
                        }
                        ForkKind::Loader { workers, batches, .. } => {
                            for w in 0..workers.min(batches) {
                                let mut kid = Stream::new(Vm::resume(snap.clone()), Role::Worker { b: w, workers, batches });
                                kid.vm.start_sub(w);
                                self.bg.push(kid);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A `trace` node for the metrics: the file in line order (the order the calls returned),
/// each lane a context with its own runs, a group a fan-out. No depth: a trace's chains
/// need a think-time threshold the file does not carry (`aeiou-trace --chain-gap-us`), so
/// the row is empty on both sides and not judged.
fn trace_walk(tf: &crate::trace::TraceFile, base: &crate::trace::OwnedCtx, sink: &mut DryRun, m: &mut Metrics) -> Result<()> {
    use crate::trace::Step;
    let mut runs: Vec<HashMap<(u64, usize), Run>> = (0..tf.lanes()).map(|_| HashMap::new()).collect();
    crate::trace::walk(tf, &base.indices, |step| match step {
        Step::Gap { ns, .. } => {
            sink.compute_ns += ns as i128;
            Ok(())
        }
        Step::Group { n, .. } => {
            *m.fan_out.entry(n as u64).or_insert(0) += 1;
            Ok(())
        }
        Step::Op { lane, op, indices } => {
            sink.op(&op, &base.ctx(indices))?;
            Stream::record(&mut runs[lane], m, &op)
        }
    })?;
    for r in runs.iter_mut() {
        for ((_, k), run) in r.drain() {
            m.run_done(k, run.bytes, run.ops);
        }
    }
    Ok(())
}

/// Walk one actor instance in round-robin order through `sink`, which must carry a
/// `Metrics` (`DryRun::with_metrics`); the counts and the fingerprint are those of the
/// inline walk, since both are sums.
pub fn walk<'m, 'a: 'm>(model: &'m Model<'a>, template: &'a str, actor: i64, count: i64, mut sink: DryRun) -> Result<DryRun> {
    let a = model.ast.actors.get(template).ok_or_else(|| anyhow!("no actor `{template}`"))?;
    let mut m = sink.metrics.take().ok_or_else(|| anyhow!("metrics walk without metrics"))?;
    let mut vm = Vm::new(model, template, actor, count);
    vm.start(&a.body);
    let mut main = Stream::new(vm, Role::Main);
    (|| -> Result<()> {
        while main.turn(&mut sink, &mut m)? {}
        m.end_instance()
    })()
    .with_context(|| format!("actor `{template}` instance {actor}"))?;
    sink.metrics = Some(m);
    Ok(sink)
}

// ---------------------------------------------------------------- reports

fn pct(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

fn bytes_row(h: &LogHist) -> String {
    format!(
        "p10 {}  p50 {}  p90 {}  p99 {}  max {}  mean {}",
        human_bytes(h.quantile(0.10)),
        human_bytes(h.quantile(0.50)),
        human_bytes(h.quantile(0.90)),
        human_bytes(h.quantile(0.99)),
        human_bytes(h.max),
        human_bytes(h.mean() as u64)
    )
}

fn exact_row(map: &BTreeMap<u64, u64>) -> String {
    let total: u64 = map.values().sum();
    let mut parts: Vec<String> = map.iter().take(12).map(|(k, v)| format!("{k}: {:.1}%", pct(*v, total))).collect();
    if map.len() > 12 {
        parts.push(format!("… ({} values)", map.len()));
    }
    format!("n={total}  {}", parts.join("  "))
}

pub fn write_text(out: &mut impl Write, d: &DryRun, indent: &str) -> std::io::Result<()> {
    let Some(m) = &d.metrics else { return Ok(()) };
    let s = &d.total;
    let reads = s.counts.get(&OpKind::Read).copied().unwrap_or(0);
    let writes = s.counts.get(&OpKind::Write).copied().unwrap_or(0);
    let data = reads + writes;
    writeln!(out, "{indent}metrics (block {}, sample 1 in {}, round-robin order within an instance)", human_bytes(m.opts_block), m.opts_sample)?;
    writeln!(
        out,
        "{indent}  mix          data ops {:.1}% of ops; reads {:.1}% of data ops, {:.1}% of data bytes",
        pct(data, s.ops),
        pct(reads, data),
        pct(s.bytes_read, s.bytes_read + s.bytes_written)
    )?;
    for k in 0..2 {
        if m.request[k].n > 0 {
            writeln!(out, "{indent}  request size {:<5} n={}  {}", KINDS[k], m.request[k].n, bytes_row(&m.request[k]))?;
        }
    }
    for k in 0..2 {
        let r = &m.runs[k];
        if r.bytes.n > 0 {
            let total = r.bytes.sum as u64;
            writeln!(
                out,
                "{indent}  run length   {:<5} runs={}  {}  ops/run mean {:.2} max {}  {:.1}% of bytes in runs of 2+ ops",
                KINDS[k],
                r.bytes.n,
                bytes_row(&r.bytes),
                r.ops.mean(),
                r.ops.max,
                pct(r.multi_op_bytes, total)
            )?;
        }
    }
    let accesses: u64 = m.first.iter().sum::<u64>() + m.reuse.iter().flatten().map(|h| h.n).sum::<u64>();
    if accesses > 0 {
        writeln!(
            out,
            "{indent}  reuse        block accesses {}; first touch: read {:.1}%, write {:.1}%",
            accesses,
            pct(m.first[R], accesses),
            pct(m.first[W], accesses)
        )?;
        for cur in 0..2 {
            for prev in 0..2 {
                let h = &m.reuse[prev][cur];
                if h.n > 0 {
                    writeln!(out, "{indent}    {:<5} after {:<5} {:.1}%  distance {}", KINDS[cur], KINDS[prev], pct(h.n, accesses), bytes_row(h))?;
                }
            }
        }
    }
    for (name, p) in [("blocks", m.popularity_blocks()), ("objects", m.popularity_objects())] {
        if p.distinct > 0 {
            write!(out, "{indent}  popularity   {:<7} distinct {}  accesses {}  max {}", name, p.distinct, p.accesses, p.max)?;
            // the shares of the top fractions mean nothing over a handful of items
            if p.distinct / m.opts_sample >= 100 {
                write!(out, "  top 0.1% {:.1}%  top 1% {:.1}%  top 10% {:.1}% of accesses", 100.0 * p.top_0_1_pct, 100.0 * p.top_1_pct, 100.0 * p.top_10_pct)?;
            }
            writeln!(out)?;
        }
    }
    if !m.fan_out.is_empty() {
        writeln!(out, "{indent}  fan-out      {}", exact_row(&m.fan_out))?;
        writeln!(out, "{indent}  depth        {}", exact_row(&m.depth))?;
    }
    Ok(())
}

#[derive(Serialize)]
struct HistOut {
    n: u64,
    mean: f64,
    max: u64,
    /// `[lower bound, weight]`, ascending.
    buckets: Vec<(u64, u64)>,
}

impl HistOut {
    fn of(h: &LogHist) -> Self {
        HistOut { n: h.n, mean: h.mean(), max: h.max, buckets: h.rows() }
    }
}

#[derive(Serialize)]
struct RunsOut {
    bytes: HistOut,
    ops: HistOut,
    multi_op_bytes: u64,
}

#[derive(Serialize)]
struct KindPair<T> {
    read: T,
    write: T,
}

#[derive(Serialize)]
struct ReuseOut {
    first_touch: KindPair<u64>,
    read_after_read: HistOut,
    read_after_write: HistOut,
    write_after_read: HistOut,
    write_after_write: HistOut,
}

/// One template's (or the total's) metrics as written by `--metrics-json`.
#[derive(Serialize)]
pub struct MetricsOut {
    ops: u64,
    counts: BTreeMap<&'static str, u64>,
    bytes_read: u64,
    bytes_written: u64,
    request_size: KindPair<HistOut>,
    run_length: KindPair<RunsOut>,
    reuse_distance_bytes: ReuseOut,
    popularity_blocks: Popularity,
    popularity_objects: Popularity,
    fan_out: BTreeMap<u64, u64>,
    depth: BTreeMap<u64, u64>,
}

pub fn to_json(d: &DryRun) -> Option<MetricsOut> {
    let m = d.metrics.as_ref()?;
    let runs = |k: usize| RunsOut { bytes: HistOut::of(&m.runs[k].bytes), ops: HistOut::of(&m.runs[k].ops), multi_op_bytes: m.runs[k].multi_op_bytes };
    Some(MetricsOut {
        ops: d.total.ops,
        counts: d.total.counts.iter().map(|(k, v)| (k.name(), *v)).collect(),
        bytes_read: d.total.bytes_read,
        bytes_written: d.total.bytes_written,
        request_size: KindPair { read: HistOut::of(&m.request[R]), write: HistOut::of(&m.request[W]) },
        run_length: KindPair { read: runs(R), write: runs(W) },
        reuse_distance_bytes: ReuseOut {
            first_touch: KindPair { read: m.first[R], write: m.first[W] },
            read_after_read: HistOut::of(&m.reuse[R][R]),
            read_after_write: HistOut::of(&m.reuse[W][R]),
            write_after_read: HistOut::of(&m.reuse[R][W]),
            write_after_write: HistOut::of(&m.reuse[W][W]),
        },
        popularity_blocks: m.popularity_blocks(),
        popularity_objects: m.popularity_objects(),
        fan_out: m.fan_out.clone(),
        depth: m.depth.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_keep_three_bits() {
        for v in [0u64, 1, 2, 3, 4, 5, 7, 8, 4096, 4097, 5000, 6143, 6144, 1 << 40, u64::MAX] {
            let lo = LogHist::lower(LogHist::bucket(v));
            assert!(lo <= v, "{v}");
            assert!(v < 4 || (v - lo) as f64 <= 0.25 * lo as f64, "{v} in bucket {lo}");
        }
        assert_eq!(LogHist::lower(LogHist::bucket(4096)), 4096);
        assert_eq!(LogHist::lower(LogHist::bucket(6144)), 6144);
        let mut h = LogHist::default();
        for v in 1..=100u64 {
            h.add(v * 4096, 1);
        }
        assert_eq!(h.quantile(0.5), 48 * 4096); // 50·4096 lies in [48, 56)·4096
        assert_eq!(h.max, 100 * 4096);
    }

    /// The Fenwick history agrees with the definition (distinct keys since the last access,
    /// this one included) on a sequence long enough to compact several times.
    #[test]
    fn stack_distance_matches_brute_force() {
        let mut inst = Instance::default();
        let mut history: Vec<u64> = Vec::new();
        let mut x = 12345u64;
        for i in 0..20_000u64 {
            x = mix64(x.wrapping_add(i));
            // a hot set of 16 keys, a warm set of 600, and fresh keys
            let key = match x % 10 {
                0..=3 => x >> 8 & 15,
                4..=8 => 100 + (x >> 8) % 600,
                _ => 10_000 + i,
            };
            let want = history.iter().rposition(|k| *k == key).map(|p| {
                let mut seen: Vec<u64> = history[p..].to_vec();
                seen.sort_unstable();
                seen.dedup();
                seen.len() as u64
            });
            let got = inst.touch(key, R).map(|(d, _)| d);
            assert_eq!(got, want, "access {i} key {key}");
            history.push(key);
        }
    }

    #[test]
    fn popularity_shares() {
        let mut map: IdMap<u64> = IdMap::default();
        map.insert(1, 90);
        for k in 2..=10 {
            map.insert(k, 1);
        }
        let p = Popularity::of(&map, 1);
        assert_eq!((p.distinct, p.accesses, p.max), (10, 99, 90));
        assert!((p.top_10_pct - 90.0 / 99.0).abs() < 1e-12);
        assert_eq!(p.counts, vec![(90, 1), (1, 9)]);
    }
}
