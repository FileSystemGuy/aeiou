//! `aeiou dry-run --metrics`: the locality metrics of the abstract's op stream
//! (`GRAMMAR_OPTIONS.md` §5.4, `runner/REFERENCE.md` §10). The fingerprint says which stream
//! ran; these say whether it is the right one, by comparison with the same numbers taken
//! from a trace of the real application.
//!
//! Two kinds of metric, kept apart because only one kind has an order to stand on:
//! - **order-free** (sums over the op multiset, the same for every interleaving): the
//!   read/write mix, request sizes, popularity of bytes (of blocks under `--metrics-block`) and
//!   of objects, `parallel` fan-out;
//! - **order-dependent** (reuse distance, sequential runs, dependency depth): taken over one
//!   actor instance at a time, in the *round-robin order* of its concurrent sub-actors: each
//!   live sub-actor of a fork issues one op per turn, in index order; a `parallel` nested in a
//!   sub-actor is that sub-actor's turn; the forking line waits for a `parallel` and takes
//!   turns beside a `loader`'s workers. It is one linearization of the partial order the
//!   abstract defines, chosen because it needs no timing. Reuse between instances is not
//!   ordered at all and is reported only as popularity.
//!
//! Reuse and popularity are taken over extents, byte ranges (`DESIGN_REVIEW.md` §3.79): per
//! instance the pieces it has touched, each last touched by one op, so the state grows with
//! how fragmented the accesses are, never with the bytes. The block metrics of version 1 are
//! computed only under `--metrics-block`, for the captures taken before. This is an analysis
//! pass, not the runner; nothing here is hashed into the fingerprint.

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

/// The most entries (pieces, blocks, or objects) one map may hold before the pass stops: a
/// guard on memory.
pub const MAX_ENTRIES: usize = 1 << 26;

#[derive(Debug, Clone, Copy, Default)]
pub struct Opts {
    /// The block metrics too, in blocks of this many bytes (`--metrics-block`).
    pub block: Option<u64>,
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

/// Reuse over extents (`runner/REFERENCE.md` §10), in bytes.
#[derive(Debug, Clone, Default)]
pub struct ExtentStats {
    /// Reuse distance per group of bytes touched again, weighted by its bytes,
    /// `[previous access][this access]`: the distinct bytes the instance touched after the op
    /// that last touched the group, plus the group's own.
    pub reuse: [[LogHist; 2]; 2],
    /// Bytes with no earlier access by the same instance, `[read, write]`.
    pub first: [u64; 2],
    /// Bytes read again, by whether the run that read them left anything of the extent they
    /// came from untouched (`partial`) or not (`whole`).
    pub whole: u64,
    pub partial: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub opts_block: Option<u64>,
    /// Requested length per op, `[read, write]`.
    pub request: [LogHist; 2],
    pub runs: [Runs; 2],
    pub extents: ExtentStats,
    /// Under `--metrics-block`: reuse distance in bytes per block access, `[previous
    /// access][this access]`: the distinct blocks the instance touched since its last access
    /// to this one, this one included, times the block size.
    pub reuse: [[LogHist; 2]; 2],
    /// Under `--metrics-block`: block accesses with no earlier access by the same instance.
    pub first: [u64; 2],
    /// `parallel` width → number of forks.
    pub fan_out: BTreeMap<u64, u64>,
    /// Consecutive `parallel`s of one context with no op or control of its own between → count.
    pub depth: BTreeMap<u64, u64>,
    /// Accesses per block (under `--metrics-block`) and per object (data ops), over all instances.
    blocks: IdMap<u64>,
    objects: IdMap<u64>,
    /// Accesses per byte over all instances.
    bytes: Bytes,
    inst: Instance,
    ext: Extents,
    /// The groups of the last data op, for its run.
    hits: Vec<Hit>,
    next_run: u64,
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

// ---------------------------------------------------------------- extents

/// A byte range `[start, end)` of one path, last touched by one op.
#[derive(Clone, Copy, Debug)]
struct Piece {
    end: u64,
    /// The op's time, a slot of the Fenwick tree.
    time: u32,
    kind: u8,
    /// Accesses of these bytes by the instance.
    count: u32,
    /// The run (`Run::id`) whose op it was: the extent the bytes belong to.
    run: u64,
}

/// A group of bytes an op touched again: contiguous, last touched by one earlier op.
#[derive(Clone, Copy, Debug)]
struct Hit {
    distance: u64,
    bytes: u64,
    prev: usize,
    run: u64,
}

/// One instance's history by byte range: its pieces, keyed by `(path, start)`, and a Fenwick
/// tree over op times holding the bytes last touched at each, so the bytes touched after a
/// time are the live bytes less a prefix sum (Bennett and Kruskal's algorithm, weighted).
#[derive(Clone, Default)]
struct Extents {
    pieces: BTreeMap<(u64, u64), Piece>,
    tree: Vec<u64>,
    live: u64,
    now: usize,
    /// Live bytes per run: an extent's bytes not touched since.
    runs: HashMap<u64, u64>,
    compactions: u64,
}

impl std::fmt::Debug for Extents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Extents({} pieces)", self.pieces.len())
    }
}

impl Extents {
    fn add(&mut self, i: usize, d: i64) {
        let mut i = i + 1;
        while i <= self.tree.len() {
            self.tree[i - 1] = (self.tree[i - 1] as i64 + d) as u64;
            i += i & i.wrapping_neg();
        }
    }

    /// Bytes last touched at times `0..=i`.
    fn prefix(&self, i: usize) -> u64 {
        let mut i = i + 1;
        let mut s = 0u64;
        while i > 0 {
            s += self.tree[i - 1];
            i -= i & i.wrapping_neg();
        }
        s
    }

    /// Renumber the live times `0..n` in order and size the tree for as many again.
    fn compact(&mut self) {
        let mut times: Vec<u32> = self.pieces.values().map(|p| p.time).collect();
        times.sort_unstable();
        times.dedup();
        let cap = (times.len() * 2).max(1024).next_power_of_two();
        self.tree.clear();
        self.tree.resize(cap, 0);
        let pieces = std::mem::take(&mut self.pieces);
        for (k, mut p) in pieces {
            p.time = times.binary_search(&p.time).expect("a live time") as u32;
            self.add(p.time as usize, (p.end - k.1) as i64);
            self.pieces.insert(k, p);
        }
        self.now = times.len();
        self.compactions += 1;
    }

    fn put(&mut self, path: u64, start: u64, p: Piece) {
        let n = p.end - start;
        self.add(p.time as usize, n as i64);
        *self.runs.entry(p.run).or_insert(0) += n;
        self.pieces.insert((path, start), p);
    }

    fn take(&mut self, path: u64, start: u64) -> Piece {
        let p = self.pieces.remove(&(path, start)).expect("a piece");
        let n = p.end - start;
        self.add(p.time as usize, -(n as i64));
        let left = self.runs.get_mut(&p.run).expect("a run's bytes");
        *left -= n;
        if *left == 0 {
            self.runs.remove(&p.run);
        }
        p
    }

    fn live_of(&self, run: u64) -> u64 {
        self.runs.get(&run).copied().unwrap_or(0)
    }

    /// One op of `kind` by run `run` touching `[start, end)` of `path`: the groups of bytes it
    /// touches again into `hits`, in ascending order, and its first-touch bytes.
    fn touch(&mut self, path: u64, start: u64, end: u64, kind: usize, run: u64, hits: &mut Vec<Hit>) -> u64 {
        if self.now == self.tree.len() {
            self.compact();
        }
        let now = self.now as u32;
        let mine = |count| Piece { end: 0, time: now, kind: kind as u8, count, run };
        let mut over: Vec<u64> = Vec::new();
        if let Some((&(p, s), piece)) = self.pieces.range(..(path, start)).next_back() {
            if p == path && piece.end > start {
                over.push(s);
            }
        }
        over.extend(self.pieces.range((path, start)..(path, end)).map(|(k, _)| k.1));
        let (mut at, mut first) = (start, 0);
        // the open group: (old time, kind, run, bytes, distance less its bytes), and where it ends
        let mut group: Option<(u32, usize, u64, u64, u64)> = None;
        let mut group_end = start;
        let close = |g: &mut Option<(u32, usize, u64, u64, u64)>, hits: &mut Vec<Hit>| {
            if let Some((_, prev, r, n, after)) = g.take() {
                hits.push(Hit { distance: after + n, bytes: n, prev, run: r });
            }
        };
        for s in over {
            let old = self.pieces[&(path, s)];
            let (from, to) = (s.max(start), old.end.min(end));
            if from > at {
                // bytes no op of the instance touched before
                close(&mut group, hits);
                self.put(path, at, Piece { end: from, ..mine(1) });
                self.live += from - at;
                first += from - at;
            }
            let same = matches!(group, Some((t, ..)) if t == old.time) && group_end == from;
            if !same {
                close(&mut group, hits);
                // the bytes touched after the old op, this op's own so far included (the old
                // piece is still at its time)
                let after = self.live - self.prefix(old.time as usize);
                group = Some((old.time, old.kind as usize, old.run, 0, after));
            }
            if let Some(g) = group.as_mut() {
                g.3 += to - from;
            }
            group_end = to;
            self.take(path, s);
            if s < from {
                self.put(path, s, Piece { end: from, ..old });
            }
            if old.end > to {
                self.put(path, to, Piece { end: old.end, ..old });
            }
            self.put(path, from, Piece { end: to, ..mine(old.count + 1) });
            at = to;
        }
        close(&mut group, hits);
        if end > at {
            self.put(path, at, Piece { end, ..mine(1) });
            self.live += end - at;
            first += end - at;
        }
        self.now += 1;
        first
    }
}

/// Accesses per byte, as pieces `(path, start) → (end, accesses)`.
#[derive(Clone, Default, Debug)]
struct Bytes {
    pieces: BTreeMap<(u64, u64), (u64, u64)>,
}

impl Bytes {
    /// `count` more accesses of `[start, end)` of `path`.
    fn add(&mut self, path: u64, start: u64, end: u64, count: u64) {
        let mut over: Vec<u64> = Vec::new();
        if let Some((&(p, s), &(e, _))) = self.pieces.range(..(path, start)).next_back() {
            if p == path && e > start {
                over.push(s);
            }
        }
        over.extend(self.pieces.range((path, start)..(path, end)).map(|(k, _)| k.1));
        let mut at = start;
        for s in over {
            let (e, c) = self.pieces.remove(&(path, s)).expect("a piece");
            let (from, to) = (s.max(start), e.min(end));
            if from > at {
                self.pieces.insert((path, at), (from, count));
            }
            if s < from {
                self.pieces.insert((path, s), (from, c));
            }
            if e > to {
                self.pieces.insert((path, to), (e, c));
            }
            self.pieces.insert((path, from), (to, c + count));
            at = to;
        }
        if end > at {
            self.pieces.insert((path, at), (end, count));
        }
    }

    fn merge(&mut self, o: &Bytes) {
        for (&(p, s), &(e, c)) in &o.pieces {
            self.add(p, s, e, c);
        }
    }

    fn popularity(&self) -> Popularity {
        Popularity::of(self.pieces.iter().map(|(k, (e, c))| (*c, e - k.1)))
    }
}

impl Metrics {
    pub fn new(o: Opts) -> Self {
        Metrics { opts_block: o.block.map(|b| b.max(1)), ..Default::default() }
    }

    fn new_run(&mut self) -> u64 {
        self.next_run += 1;
        self.next_run
    }

    /// A data op of run `run`: its request size, its object, the bytes it transfers (their
    /// groups left in `self.hits`), and, under `--metrics-block`, every block it transfers.
    fn data(&mut self, kind: usize, path: u64, offset: i64, len: i64, bytes: i64, run: u64) -> Result<()> {
        self.request[kind].add(len.max(0) as u64, 1);
        *self.objects.entry(path).or_insert(0) += 1;
        self.hits.clear();
        if bytes <= 0 || offset < 0 {
            return Ok(());
        }
        let (start, end) = (offset as u64, offset as u64 + bytes as u64);
        let mut hits = std::mem::take(&mut self.hits);
        self.extents.first[kind] += self.ext.touch(path, start, end, kind, run, &mut hits);
        for h in &hits {
            self.extents.reuse[h.prev][kind].add(h.distance, h.bytes);
        }
        self.hits = hits;
        if self.ext.pieces.len() > MAX_ENTRIES {
            bail!("--metrics: more than {MAX_ENTRIES} pieces in one instance (byte ranges touched, as fragmented as the accesses)");
        }
        if let Some(b) = self.opts_block {
            for blk in start / b..=(end - 1) / b {
                match self.inst.touch(block_key(path, blk), kind) {
                    Some((d, prev)) => self.reuse[prev][kind].add(d.saturating_mul(b), 1),
                    None => self.first[kind] += 1,
                }
            }
            if self.inst.last.len() > MAX_ENTRIES {
                bail!("--metrics: more than {MAX_ENTRIES} distinct blocks in one instance; give a larger --metrics-block");
            }
        }
        Ok(())
    }

    fn run_done(&mut self, kind: usize, run: &Run) {
        let r = &mut self.runs[kind];
        r.bytes.add(run.bytes, 1);
        r.ops.add(run.ops, 1);
        if run.ops >= 2 {
            r.multi_op_bytes += run.bytes;
        }
        for (x, n) in &run.again {
            if self.ext.live_of(*x) == 0 {
                self.extents.whole += n;
            } else {
                self.extents.partial += n;
            }
        }
    }

    /// The instance has ended: its counts join the popularity maps, its history goes.
    fn end_instance(&mut self) -> Result<()> {
        for (k, s) in self.inst.last.drain() {
            *self.blocks.entry(k).or_insert(0) += s.count as u64;
        }
        self.inst.tree.clear();
        self.inst.now = 0;
        let ext = std::mem::take(&mut self.ext);
        for ((p, s), piece) in &ext.pieces {
            self.bytes.add(*p, *s, piece.end, piece.count as u64);
        }
        self.check_size()
    }

    fn check_size(&self) -> Result<()> {
        if self.blocks.len() > MAX_ENTRIES || self.objects.len() > MAX_ENTRIES || self.bytes.pieces.len() > MAX_ENTRIES {
            bail!("--metrics: more than {MAX_ENTRIES} distinct byte ranges, blocks, or objects over all instances");
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
            self.extents.first[k] += o.extents.first[k];
            for j in 0..2 {
                self.reuse[k][j].merge(&o.reuse[k][j]);
                self.extents.reuse[k][j].merge(&o.extents.reuse[k][j]);
            }
        }
        self.extents.whole += o.extents.whole;
        self.extents.partial += o.extents.partial;
        self.bytes.merge(&o.bytes);
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
        Popularity::of(self.blocks.values().map(|c| (*c, 1)))
    }

    pub fn popularity_objects(&self) -> Popularity {
        Popularity::of(self.objects.values().map(|c| (*c, 1)))
    }

    pub fn popularity_bytes(&self) -> Popularity {
        self.bytes.popularity()
    }
}

/// The rank–frequency curve as counts of counts: `(accesses, items with that many)`,
/// descending by accesses; for bytes, an item is a byte.
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
    /// From `(accesses, items)` pairs, an item counted once per pair.
    fn of(items: impl Iterator<Item = (u64, u64)>) -> Popularity {
        let mut cc: BTreeMap<u64, u64> = BTreeMap::new();
        for (c, n) in items {
            *cc.entry(c).or_insert(0) += n;
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
            distinct: items,
            accesses: total,
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

#[derive(Clone, Default)]
struct Run {
    id: u64,
    end: i64,
    bytes: u64,
    ops: u64,
    /// Bytes its reads touched again, per extent (the run that last touched them).
    again: Vec<(u64, u64)>,
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
            m.run_done(k, &r);
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
                        m.run_done(k, &r);
                    }
                }
                return Ok(());
            }
            _ => return Ok(()),
        };
        let path = xxh3_64(op.path.as_bytes());
        let bytes = op.bytes;
        if bytes <= 0 {
            return m.data(kind, path, op.offset, op.len, bytes, 0);
        }
        // a run ends before the op that does not continue it
        let r = runs.entry((path, kind)).or_default();
        if r.ops > 0 && r.end != op.offset {
            m.run_done(kind, r);
            *r = Run::default();
        }
        if r.ops == 0 {
            r.id = m.new_run();
        }
        m.data(kind, path, op.offset, op.len, bytes, r.id)?;
        if kind == R {
            for h in &m.hits {
                match r.again.iter_mut().find(|(x, _)| *x == h.run) {
                    Some(e) => e.1 += h.bytes,
                    None => r.again.push((h.run, h.bytes)),
                }
            }
        }
        r.end = op.offset + bytes;
        r.bytes += bytes as u64;
        r.ops += 1;
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
            m.run_done(k, &run);
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
    match m.opts_block {
        Some(b) => writeln!(out, "{indent}metrics (extents, and blocks of {}; round-robin order within an instance)", human_bytes(b))?,
        None => writeln!(out, "{indent}metrics (extents; round-robin order within an instance)")?,
    }
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
    let e = &m.extents;
    let touched: u64 = e.first.iter().sum::<u64>() + e.reuse.iter().flatten().map(|h| h.n).sum::<u64>();
    if touched > 0 {
        writeln!(
            out,
            "{indent}  reuse        bytes touched {}; first touch: read {:.1}%, write {:.1}%",
            human_bytes(touched),
            pct(e.first[R], touched),
            pct(e.first[W], touched)
        )?;
        for cur in 0..2 {
            for prev in 0..2 {
                let h = &e.reuse[prev][cur];
                if h.n > 0 {
                    writeln!(out, "{indent}    {:<5} after {:<5} {:.1}%  distance {}", KINDS[cur], KINDS[prev], pct(h.n, touched), bytes_row(h))?;
                }
            }
        }
        if e.whole + e.partial > 0 {
            writeln!(out, "{indent}    re-read      whole extents {:.1}%, part of one {:.1}% of the bytes read again", pct(e.whole, e.whole + e.partial), pct(e.partial, e.whole + e.partial))?;
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
    for (name, p) in [("bytes", m.popularity_bytes()), ("blocks", m.popularity_blocks()), ("objects", m.popularity_objects())] {
        if p.distinct > 0 {
            write!(out, "{indent}  popularity   {:<7} distinct {}  accesses {}  max {}", name, p.distinct, p.accesses, p.max)?;
            // the shares of the top fractions mean nothing over a handful of items
            if p.distinct >= 100 {
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

#[derive(Serialize)]
struct ExtentsOut {
    first_touch: KindPair<u64>,
    read_after_read: HistOut,
    read_after_write: HistOut,
    write_after_read: HistOut,
    write_after_write: HistOut,
    whole: u64,
    partial: u64,
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
    reuse_extents: ExtentsOut,
    popularity_bytes: Popularity,
    /// Under `--metrics-block` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    reuse_distance_bytes: Option<ReuseOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    popularity_blocks: Option<Popularity>,
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
        reuse_extents: ExtentsOut {
            first_touch: KindPair { read: m.extents.first[R], write: m.extents.first[W] },
            read_after_read: HistOut::of(&m.extents.reuse[R][R]),
            read_after_write: HistOut::of(&m.extents.reuse[W][R]),
            write_after_read: HistOut::of(&m.extents.reuse[R][W]),
            write_after_write: HistOut::of(&m.extents.reuse[W][W]),
            whole: m.extents.whole,
            partial: m.extents.partial,
        },
        popularity_bytes: m.popularity_bytes(),
        reuse_distance_bytes: m.opts_block.map(|_| ReuseOut {
            first_touch: KindPair { read: m.first[R], write: m.first[W] },
            read_after_read: HistOut::of(&m.reuse[R][R]),
            read_after_write: HistOut::of(&m.reuse[W][R]),
            write_after_read: HistOut::of(&m.reuse[R][W]),
            write_after_write: HistOut::of(&m.reuse[W][W]),
        }),
        popularity_blocks: m.opts_block.map(|_| m.popularity_blocks()),
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
        let p = Popularity::of(map.values().map(|c| (*c, 1)));
        assert_eq!((p.distinct, p.accesses, p.max), (10, 99, 90));
        assert!((p.top_10_pct - 90.0 / 99.0).abs() < 1e-12);
        assert_eq!(p.counts, vec![(90, 1), (1, 9)]);
    }

    /// Byte popularity: counts of counts in bytes, a fraction of a piece taken pro rata.
    #[test]
    fn popularity_of_bytes() {
        let mut b = Bytes::default();
        b.add(1, 0, 100, 3);
        b.add(1, 50, 150, 1); // [50, 100) now 4, [100, 150) 1
        b.add(2, 0, 10, 2);
        let mut other = Bytes::default();
        other.add(1, 140, 160, 2); // [140, 150) 3, [150, 160) 2
        b.merge(&other);
        let p = b.popularity();
        assert_eq!(p.counts, vec![(4, 50), (3, 60), (2, 20), (1, 40)]);
        assert_eq!((p.distinct, p.accesses, p.max), (170, 4 * 50 + 3 * 60 + 2 * 20 + 40, 4));
        // the top 10 % of 170 bytes: 17 bytes accessed 4 times
        assert!((p.top_10_pct - 68.0 / 460.0).abs() < 1e-12);
        // a gap between pieces and a piece's right part: [100, 140) splits at 120, [205, 300) is
        // new, [300, 310) splits at 305
        b.add(1, 120, 200, 1);
        b.add(1, 300, 310, 1);
        b.add(1, 205, 305, 2);
        let p = b.popularity();
        assert_eq!(p.counts, vec![(4, 60), (3, 65), (2, 125), (1, 65)]);
        assert_eq!((p.distinct, p.accesses), (315, 750));
        assert!((p.top_10_pct - 126.0 / 750.0).abs() < 1e-12);
    }

    /// One op of the brute force: the groups `(distance, bytes, previous kind)` of the bytes it
    /// touches again (contiguous bytes last touched by one earlier op, in ascending order,
    /// each op's bytes taking its time as they are passed), and its first-touch bytes.
    fn brute(last: &mut HashMap<(u64, u64), (usize, usize)>, t: usize, path: u64, start: u64, end: u64, kind: usize) -> (Vec<(u64, u64, usize)>, u64) {
        let (mut groups, mut first) = (Vec::new(), 0);
        let mut at = start;
        while at < end {
            match last.get(&(path, at)).copied() {
                None => {
                    first += 1;
                    last.insert((path, at), (t, kind));
                    at += 1;
                }
                Some((op, prev)) => {
                    let after = last.values().filter(|(o, _)| *o > op).count() as u64;
                    let mut len = 0;
                    while at < end && last.get(&(path, at)).map(|x| x.0) == Some(op) {
                        last.insert((path, at), (t, kind));
                        at += 1;
                        len += 1;
                    }
                    groups.push((after + len, len, prev));
                }
            }
        }
        (groups, first)
    }

    /// The extents agree with the definition on random ops over three paths, overlapping at
    /// every alignment, long enough to compact; and each run's live bytes are its own.
    #[test]
    fn extent_distance_matches_brute_force() {
        let mut ext = Extents::default();
        let mut last: HashMap<(u64, u64), (usize, usize)> = HashMap::new();
        let mut owner: HashMap<(u64, u64), u64> = HashMap::new();
        let mut x = 777u64;
        let mut hits = Vec::new();
        for t in 0..3000usize {
            x = mix64(x.wrapping_add(t as u64));
            let path = x % 3;
            let start = (x >> 8) % 1500;
            let end = start + 1 + (x >> 20) % 300;
            let kind = ((x >> 40) % 3 == 0) as usize;
            let run = (x >> 44) % 50;
            let (want, want_first) = brute(&mut last, t, path, start, end, kind);
            hits.clear();
            let first = ext.touch(path, start, end, kind, run, &mut hits);
            let got: Vec<(u64, u64, usize)> = hits.iter().map(|h| (h.distance, h.bytes, h.prev)).collect();
            assert_eq!((got, first), (want, want_first), "op {t}: path {path} [{start}, {end})");
            for b in start..end {
                owner.insert((path, b), run);
            }
            if t % 97 == 0 {
                for r in 0..50 {
                    assert_eq!(ext.live_of(r), owner.values().filter(|o| **o == r).count() as u64, "op {t}: run {r}");
                }
            }
        }
        assert!(ext.compactions > 0, "the test compacts");
    }
}
