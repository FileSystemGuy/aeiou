//! `aeiou-trace digest`: the structural summary of a trace that a fit is made from
//! (`PROJECT_BRIEF.md` item 47; `DESIGN_REVIEW.md` §3.86 for what it can and cannot decide, §3.87
//! for its design, §3.88 for its sites, verdicts, and questions, §3.89 for this port and the
//! version-2 document), with a verdict per site on what the trace decides and an `open` list of
//! what it does not, ranked by how far the candidates diverge at the scale the benchmark will run.
//!
//! The digest reads the calls `aeiou-trace export` writes (the exporter's event stream, so
//! descriptor tables, `clone`, `dup`, and positions are the one resolver's), or an exported trace
//! file, and reduces them to:
//!
//! - **path templates**: every run of digits in a path is a field (`train/{a:05}/sample_{b:09}.npz`),
//!   a run of 16 or more hex digits with a letter in it a hex field (`{a:x32}`); per template, how
//!   many values each field took and their range, the op mix, bytes, request sizes, and how each
//!   open moved through its file (`whole`, `sequential`, `positioned`). No raw path unless asked.
//! - **lanes and phases**: a lane is a traced task; a unit is the calls on one open from `open` to
//!   `close` (a path call, or an `io_submit`, is a unit of one op); a unit's motif is its op
//!   sequence with repeats collapsed (`read+`, `(lseek read)+`); a phase is a run of units of one
//!   motif on one template, with the repeat counts and the think time between units.
//! - **sites** of five kinds (`set`, `keyed`, `bulk_write`, `restore`, `within`), each with its
//!   verdicts, the facts a person can give about it at scale, and the bounds the trace sets on them.
//! - **open**: every undecided verdict as a question in the application's terms, ranked.
//! - **metrics**: the `aeiou_metrics: 2` numbers over the same calls (`export::FileMetrics`).
//!
//! It streams: an open's ops are folded into its unit at its close and dropped; a site keeps one
//! record per open; paths are kept once per template; the offsets of one open are kept, as a count
//! per offset, only while that open could still be a template's one open or one reader (a
//! `within` site). The Python digest's tests are its specification (`tests/digest/`).

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::export::{self, FileMetrics, Opener, TRACE_FORMAT};
use crate::metrics::LogHist;
use crate::objectlog;
use crate::strace;
use crate::trace::Line;

pub const FORMAT: u64 = 2;

/// A fast hash for the digest's maps (rustc's FxHash): every call looks something up.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64);
    }
    fn write_u16(&mut self, v: u16) {
        self.write_u64(v as u64);
    }
    fn write_u8(&mut self, v: u8) {
        self.write_u64(v as u64);
    }
    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }
}

type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fx>>;
type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<Fx>>;
const P_REFUTE: f64 = 0.05; // a candidate whose chance of producing the evidence is below this is refuted
const MAX_PERIOD: usize = 4; // the longest cycle of ops collapsed in a motif
const WITHIN_MIN_READS: u64 = 8;
const SIMULTANEOUS: f64 = 0.25; // a save's lanes start together when their first opens fall within this share of its length
pub const UNKNOWN: &str = "unknown";

fn ln_refute() -> f64 {
    (1.0 / P_REFUTE).ln()
}

/// The options a document is made with.
pub struct Opts {
    /// Passes over a set the open questions are projected over (`--epochs`).
    pub epochs: i64,
    /// The corpus a set's files were chosen among (`--candidates`).
    pub candidates: Option<i64>,
    /// Every raw path under its template (`--paths`).
    pub paths: bool,
}

// ---------------------------------------------------------------- formatting as the Python did

/// `format(x, ".Pg")`.
pub fn fmt_g(x: f64, p: usize) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return if x.is_nan() { "nan".into() } else if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let p = p.max(1);
    let e = format!("{:.*e}", p - 1, x);
    let (m, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if exp < -4 || exp >= p as i32 {
        format!("{}e{}{:02}", strip(m), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        strip(&format!("{:.*}", (p as i32 - 1 - exp) as usize, x))
    }
}

/// `format(x, ".P%")`.
pub(crate) fn pct(x: f64, p: usize) -> String {
    format!("{:.*}%", p, x * 100.0)
}

/// A float as the Python's `str` writes it: `0.05`, `1.0`.
fn py_float(x: f64) -> String {
    let s = format!("{x}");
    if s.contains('.') || s.contains('e') || !x.is_finite() { s } else { format!("{s}.0") }
}

/// A value as the Python's f-string writes it.
pub(crate) fn py(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) if n.is_f64() => py_float(n.as_f64().unwrap()),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        other => py(other),
    }
}

pub fn bytes_text(n: i64) -> String {
    for (unit, div) in [("GiB", 1i64 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)] {
        if n >= div {
            return format!("{:.1} {unit}", n as f64 / div as f64);
        }
    }
    format!("{n} B")
}

pub fn ns_text(ns: f64) -> String {
    if ns >= 1e9 {
        format!("{:.2} s", ns / 1e9)
    } else if ns >= 1e6 {
        format!("{:.1} ms", ns / 1e6)
    } else if ns >= 1e3 {
        format!("{:.0} us", ns / 1e3)
    } else {
        format!("{ns:.0} ns")
    }
}

/// The lower bound of the bucket holding the median of a histogram as the document writes it.
fn p50(h: &Value) -> i64 {
    let half = h["n"].as_f64().unwrap_or(0.0) / 2.0;
    let mut acc = 0.0;
    for b in h["buckets"].as_array().into_iter().flatten() {
        acc += b[1].as_f64().unwrap_or(0.0);
        if acc >= half {
            return b[0].as_i64().unwrap_or(0);
        }
    }
    h["max"].as_i64().unwrap_or(0)
}

fn hist(h: &LogHist) -> Value {
    json!({"n": h.n, "mean": h.mean(), "max": h.max, "buckets": h.rows()})
}

fn add(h: &mut LogHist, v: i64) {
    h.add(v.max(0) as u64, 1);
}

// ---------------------------------------------------------------- templates

/// One varying run in a template: its kind, the widths and values it took.
struct Field {
    hex: bool,
    widths: BTreeSet<usize>,
    values: HashSet<String>,
    lo: Option<u64>,
    hi: Option<u64>,
}

impl Field {
    fn new(hex: bool) -> Field {
        Field { hex, widths: BTreeSet::new(), values: HashSet::default(), lo: None, hi: None }
    }

    fn add(&mut self, text: &str) {
        self.widths.insert(text.len());
        if !self.values.contains(text) {
            self.values.insert(text.to_string());
        }
        if !self.hex {
            if let Ok(v) = text.parse::<u64>() {
                self.lo = Some(self.lo.map_or(v, |l| l.min(v)));
                self.hi = Some(self.hi.map_or(v, |h| h.max(v)));
            }
        }
    }

    fn width(&self) -> Option<usize> {
        match (self.widths.len(), self.widths.iter().next()) {
            (1, Some(&w)) if w > 1 => Some(w),
            _ => None,
        }
    }

    fn placeholder(&self, i: usize) -> String {
        let n = field_name(i);
        match (self.hex, self.width()) {
            (true, Some(w)) => format!("{{{n}:x{w}}}"),
            (true, None) => format!("{{{n}:x}}"),
            (false, Some(w)) => format!("{{{n}:0{w}}}"),
            (false, None) => format!("{{{n}}}"),
        }
    }

    fn out(&self, i: usize) -> Map<String, Value> {
        let mut d = Map::new();
        d.insert("name".into(), json!(field_name(i)));
        d.insert("kind".into(), json!(if self.hex { "hex" } else { "int" }));
        d.insert("width".into(), json!(self.width()));
        d.insert("values".into(), json!(self.values.len()));
        if !self.hex {
            d.insert("min".into(), json!(self.lo));
            d.insert("max".into(), json!(self.hi));
        }
        d
    }
}

fn field_name(i: usize) -> String {
    if i < 26 { ((b'a' + i as u8) as char).to_string() } else { format!("f{i}") }
}

const MARK: char = '\0';

/// The template key of a path (fields marked) and its field values with their kinds (hex or
/// not): a run of 16 or more hex digits, else a run of digits (`[0-9a-f]{16,}|[0-9]+`).
pub fn split(path: &str) -> (String, Vec<(bool, &str)>) {
    let b = path.as_bytes();
    let hexd = |c: u8| c.is_ascii_digit() || (b'a'..=b'f').contains(&c);
    let (mut key, mut fields, mut i, mut from) = (String::with_capacity(path.len()), Vec::new(), 0, 0);
    while i < b.len() {
        let mut j = i;
        while j < b.len() && hexd(b[j]) {
            j += 1;
        }
        let end = if j - i >= 16 {
            j
        } else if b[i].is_ascii_digit() {
            let mut k = i;
            while k < b.len() && b[k].is_ascii_digit() {
                k += 1;
            }
            k
        } else {
            i += 1;
            continue;
        };
        let text = &path[i..end];
        key.push_str(&path[from..i]);
        key.push(MARK);
        fields.push((text.len() >= 16 && !text.bytes().all(|c| c.is_ascii_digit()), text));
        i = end;
        from = end;
    }
    key.push_str(&path[from..]);
    (key, fields)
}

/// One open of a template, as a site judges it.
#[derive(Default)]
struct Touch {
    t: i64,
    path: u32,
    lane: i64,
    first: i64,
    last: i64,
    bytes: [i64; 2],
    synced: bool,
    span: i64,
    ndata: u64,
    positioned: bool,
    closed: bool,
    /// The offsets it moved through, kept while it could be a `within` site.
    within: Option<Box<Data>>,
}

struct Template {
    id: usize,
    key: String,
    kinds: Vec<bool>,
    index: HashMap<String, u32>,
    paths: Vec<String>,
    opens: u64,
    ops: Vec<(String, u64)>,
    bytes: [i64; 2],
    size: [LogHist; 2],
    offsets: BTreeMap<&'static str, u64>,
    runs: LogHist,
    touches: Vec<Touch>,
    removes: Vec<(i64, u32)>,
    readers: u32,
    kept: Vec<usize>,
}

impl Template {
    fn count(&mut self, op: &str) {
        match self.ops.iter_mut().find(|(k, _)| k == op) {
            Some((_, n)) => *n += 1,
            None => self.ops.push((op.to_string(), 1)),
        }
    }

    fn name(&self) -> String {
        let fields = self.fields_over(0..self.paths.len() as u32);
        let mut out = String::new();
        for (i, part) in self.key.split(MARK).enumerate() {
            if i > 0 {
                out.push_str(&fields[i - 1].placeholder(i - 1));
            }
            out.push_str(part);
        }
        out
    }

    fn fields_over(&self, paths: impl IntoIterator<Item = u32>) -> Vec<Field> {
        let mut fields: Vec<Field> = self.kinds.iter().map(|&h| Field::new(h)).collect();
        for p in paths {
            for (f, (_, text)) in fields.iter_mut().zip(split(&self.paths[p as usize]).1) {
                f.add(text);
            }
        }
        fields
    }

    /// A kept `within` record is good while this touch can still be the template's one open, or its one reader.
    fn prune(&mut self) {
        let (n, readers) = (self.touches.len(), self.readers);
        let touches = &mut self.touches;
        self.kept.retain(|&i| {
            let keep = (n == 1 && i == 0) || (readers == 1 && touches[i].bytes[0] > 0);
            if !keep {
                touches[i].within = None;
            }
            keep
        });
    }

    fn out(&self, raw: bool, name: &str) -> Value {
        let fields = self.fields_over(0..self.paths.len() as u32);
        let mut ops = self.ops.clone();
        ops.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let mut d = json!({
            "id": self.id,
            "template": name,
            "paths": self.paths.len(),
            "fields": fields.iter().enumerate().map(|(i, f)| Value::Object(f.out(i))).collect::<Vec<_>>(),
            "opens": self.opens,
            "ops": ops.iter().map(|(k, n)| (k.clone(), json!(n))).collect::<Map<_, _>>(),
            "bytes_read": self.bytes[0],
            "bytes_written": self.bytes[1],
            "read_size": hist(&self.size[0]),
            "write_size": hist(&self.size[1]),
            "offsets": self.offsets.iter().map(|(k, n)| (k.to_string(), json!(n))).collect::<Map<_, _>>(),
            "runs_per_open": hist(&self.runs),
        });
        if raw {
            let mut p = self.paths.clone();
            p.sort();
            d["raw_paths"] = json!(p);
        }
        d
    }
}

// ---------------------------------------------------------------- units, motifs, phases

/// The op sequence as groups: the shortest cycle of up to `MAX_PERIOD` ops that repeats at a
/// position is one group with its count; anything else a group of one.
pub fn compress<T: PartialEq + Clone>(ops: &[T]) -> Vec<(Vec<T>, u64)> {
    let (mut out, mut i, n) = (Vec::new(), 0, ops.len());
    while i < n {
        let mut hit = None;
        for l in 1..=MAX_PERIOD {
            if i + 2 * l > n {
                break;
            }
            let unit = &ops[i..i + l];
            let mut r = 1;
            while i + (r + 1) * l <= n && &ops[i + r * l..i + (r + 1) * l] == unit {
                r += 1;
            }
            if r >= 2 {
                hit = Some((l, r));
                break;
            }
        }
        match hit {
            Some((l, r)) => {
                out.push((ops[i..i + l].to_vec(), r as u64));
                i += l * r;
            }
            None => {
                out.push((vec![ops[i].clone()], 1));
                i += 1;
            }
        }
    }
    out
}

fn plus(group: &str) -> String {
    if group.contains(' ') { format!("({group})+") } else { format!("{group}+") }
}

/// How an open moved through its file, kept as its data arrives.
#[derive(Default, Clone)]
struct Data {
    n: u64,
    first_off: i64,
    end: Option<i64>,
    runs: u64,
    cur: u64,
    run_hist: LogHist,
    counts: HashMap<i64, u64>,
    span: Option<i64>,
}

impl Data {
    fn add(&mut self, off: i64, moved: i64) {
        if self.n == 0 {
            self.first_off = off;
        }
        self.n += 1;
        if self.end == Some(off) {
            self.cur += 1;
        } else {
            if self.cur > 0 {
                self.run_hist.add(self.cur, 1);
            }
            self.cur = 1;
            self.runs += 1;
        }
        self.end = Some(off + moved);
        self.span = Some(self.span.map_or(off + moved, |s| s.max(off + moved)));
        *self.counts.entry(off).or_insert(0) += 1;
    }

    /// The class and the number of sequential runs.
    fn class(&self) -> (&'static str, u64) {
        if self.n == 0 {
            ("none", 0)
        } else if self.runs > 1 {
            ("positioned", self.runs)
        } else if self.first_off == 0 {
            ("whole", 1)
        } else {
            ("sequential", 1)
        }
    }

    fn finish(&mut self) {
        if self.cur > 0 {
            self.run_hist.add(self.cur, 1);
            self.cur = 0;
        }
    }
}

struct Open {
    lane: i64,
    slot: usize,
    tpl: usize,
    touch: usize,
    ops: Vec<u16>,
    first: i64,
    last: i64,
    bytes: [i64; 2],
    synced: bool,
    pos: i64,
    data: Data,
}

impl Open {
    fn op(&mut self, id: u16, t: i64, dur: i64) {
        self.ops.push(id);
        self.last = self.last.max(t + dur);
    }
}

/// A unit folded: its motif's groups (by id), for the phases.
struct Unit {
    first: i64,
    last: i64,
    template: i64,
    key: u32,
    groups: Vec<(u32, u64)>,
    ops: u64,
    bytes: [i64; 2],
}

struct Phase<'a> {
    key: u32,
    template: i64,
    units: u64,
    ops: u64,
    bytes: [i64; 2],
    repeats: BTreeMap<String, LogHist>,
    think: LogHist,
    first: i64,
    last: i64,
    groups: &'a [(u32, u64)],
    rich: i64,
}

// ---------------------------------------------------------------- the digester

#[derive(Default)]
struct Digester {
    templates: HashMap<String, usize>,
    by_id: Vec<Template>,
    opens: HashMap<i64, Open>,
    lanes: BTreeMap<i64, Vec<Option<Unit>>>,
    lane_ops: BTreeMap<i64, u64>,
    ops: u64,
    notes: BTreeMap<String, u64>,
    op_names: Vec<String>,
    groups: Vec<String>,
    group_ids: HashMap<Vec<u16>, u32>,
    keys: Vec<String>,
    key_ids: HashMap<String, u32>,
}

/// An exported line as the digest reads it: every field optional, as the Python read a dict.
#[derive(Deserialize, Default)]
struct Ev {
    #[serde(default)]
    lane: i64,
    #[serde(default)]
    t: i64,
    #[serde(default)]
    dur: i64,
    #[serde(default)]
    op: String,
    fd: Option<i64>,
    path: Option<String>,
    to: Option<String>,
    ret: Option<Value>,
    offset: Option<i64>,
    len: Option<i64>,
    ops: Option<Vec<Ev>>,
}

impl Ev {
    fn ret(&self) -> Option<i64> {
        self.ret.as_ref().and_then(Value::as_i64)
    }
}

impl Digester {
    fn op_id(&mut self, name: &str) -> u16 {
        // a few dozen names at most: a scan beats a hash
        if let Some(i) = self.op_names.iter().position(|n| n == name) {
            return i as u16;
        }
        self.op_names.push(name.to_string());
        (self.op_names.len() - 1) as u16
    }

    fn note(&mut self, k: &str) {
        *self.notes.entry(k.to_string()).or_insert(0) += 1;
    }

    fn template(&mut self, path: &str) -> (usize, u32) {
        let (key, values) = split(path);
        let id = match self.templates.get(&key) {
            Some(&i) => i,
            None => {
                let id = self.by_id.len();
                self.by_id.push(Template {
                    id,
                    key: key.clone(),
                    kinds: values.iter().map(|(h, _)| *h).collect(),
                    index: HashMap::default(),
                    paths: Vec::new(),
                    opens: 0,
                    ops: Vec::new(),
                    bytes: [0, 0],
                    size: [LogHist::default(), LogHist::default()],
                    offsets: BTreeMap::new(),
                    runs: LogHist::default(),
                    touches: Vec::new(),
                    removes: Vec::new(),
                    readers: 0,
                    kept: Vec::new(),
                });
                self.templates.insert(key, id);
                id
            }
        };
        let t = &mut self.by_id[id];
        let p = match t.index.get(path) {
            Some(&p) => p,
            None => {
                let p = t.paths.len() as u32;
                t.paths.push(path.to_string());
                t.index.insert(path.to_string(), p);
                p
            }
        };
        (id, p)
    }

    fn slot(&mut self, lane: i64) -> usize {
        let units = self.lanes.entry(lane).or_default();
        units.push(None);
        units.len() - 1
    }

    fn fold(&mut self, ops: &[u16]) -> (u32, Vec<(u32, u64)>) {
        let groups = compress(ops);
        let mut out = Vec::with_capacity(groups.len());
        let mut key = String::new();
        for (g, r) in groups {
            let id = match self.group_ids.get(&g) {
                Some(&i) => i,
                None => {
                    let text = g.iter().map(|&o| self.op_names[o as usize].as_str()).collect::<Vec<_>>().join(" ");
                    let i = self.groups.len() as u32;
                    self.groups.push(text);
                    self.group_ids.insert(g, i);
                    i
                }
            };
            if !key.is_empty() {
                key.push(' ');
            }
            key.push_str(&self.groups[id as usize]);
            out.push((id, r));
        }
        let k = match self.key_ids.get(&key) {
            Some(&i) => i,
            None => {
                let i = self.keys.len() as u32;
                self.keys.push(key.clone());
                self.key_ids.insert(key, i);
                i
            }
        };
        (k, out)
    }

    fn place(&mut self, lane: i64, slot: usize, u: Unit) {
        self.lanes.get_mut(&lane).unwrap()[slot] = Some(u);
    }

    fn data(&mut self, e: &Ev, t: i64, dur: i64, member: Option<i64>) {
        let oid = member.unwrap_or(e.fd.unwrap_or(-1));
        let op = e.op.as_str();
        let Some(u) = self.opens.get_mut(&oid) else {
            self.note("data_on_unknown_open");
            return;
        };
        let kind = usize::from(op != "read");
        let n = e.ret().unwrap_or(0);
        let len = e.len.unwrap_or(0);
        let off = match e.offset {
            Some(o) => o,
            None => {
                let o = u.pos;
                u.pos = o + n;
                o
            }
        };
        u.bytes[kind] += n;
        u.data.add(off, n);
        let tpl = u.tpl;
        if member.is_none() {
            let id = self.op_id(op);
            self.opens.get_mut(&oid).unwrap().op(id, t, dur); // a submit's members are the submit unit's ops, not the open's
        }
        let tp = &mut self.by_id[tpl];
        tp.count(op);
        tp.bytes[kind] += n;
        tp.size[kind].add(len.max(0) as u64, 1);
    }

    fn event(&mut self, e: &Ev) {
        let (lane, op, t, dur) = (e.lane, e.op.as_str(), e.t, e.dur);
        self.ops += 1;
        *self.lane_ops.entry(lane).or_insert(0) += 1;
        match op {
            "open" => {
                let path = e.path.as_deref().unwrap_or("");
                let (tpl, p) = self.template(path);
                let open = self.op_id("open");
                let slot = self.slot(lane);
                let tp = &mut self.by_id[tpl];
                tp.opens += 1;
                tp.count("open");
                tp.touches.push(Touch { t, path: p, lane, first: t, last: t + dur, ..Default::default() });
                let touch = tp.touches.len() - 1;
                tp.prune();
                let o = Open { lane, slot, tpl, touch, ops: vec![open], first: t, last: t + dur, bytes: [0, 0], synced: false, pos: 0, data: Data::default() };
                if let Some(old) = self.opens.insert(e.fd.unwrap_or(-1), o) {
                    self.retire(old, false);
                }
            }
            "read" | "write" => self.data(e, t, dur, None),
            "submit" => {
                let members = e.ops.as_deref().unwrap_or(&[]);
                let slot = self.slot(lane);
                let mut ops = vec![self.op_id("submit")];
                *self.notes.entry("submit_members".into()).or_insert(0) += members.len() as u64;
                let mut template = -1i64;
                for sub in members {
                    let fd = sub.fd.unwrap_or(-1);
                    let Some(m) = self.opens.get(&fd) else {
                        self.note("data_on_unknown_open");
                        continue;
                    };
                    let mt = m.tpl as i64;
                    self.data(sub, t, dur, Some(fd));
                    if template < 0 {
                        template = mt;
                    }
                }
                let member = self.op_id("member");
                ops.extend(std::iter::repeat_n(member, members.len()));
                let (key, groups) = self.fold(&ops);
                self.place(lane, slot, Unit { first: t, last: t + dur, template, key, groups, ops: ops.len() as u64, bytes: [0, 0] });
            }
            "close" => {
                let fd = e.fd.unwrap_or(-1);
                let Some(mut u) = self.opens.remove(&fd) else {
                    self.note("close_of_unknown_open");
                    return;
                };
                let id = self.op_id("close");
                u.op(id, t, dur);
                self.by_id[u.tpl].count("close");
                self.retire(u, true);
            }
            _ if e.fd.is_some() => {
                let fd = e.fd.unwrap();
                let id = self.op_id(op);
                let Some(u) = self.opens.get_mut(&fd) else {
                    self.note(&format!("{op}_on_unknown_open"));
                    return;
                };
                if op == "lseek" {
                    if let Some(r) = e.ret() {
                        u.pos = r;
                    }
                }
                if op == "fsync" || op == "fdatasync" {
                    u.synced = true;
                }
                u.op(id, t, dur);
                let tpl = u.tpl;
                self.by_id[tpl].count(op);
            }
            _ => {
                // a path op: a unit of one op
                let path = e.path.as_deref().unwrap_or("");
                let (tpl, p) = self.template(path);
                self.by_id[tpl].count(op);
                let slot = self.slot(lane);
                let id = self.op_id(op);
                let (key, groups) = self.fold(&[id]);
                self.place(lane, slot, Unit { first: t, last: t + dur, template: tpl as i64, key, groups, ops: 1, bytes: [0, 0] });
                if op == "unlink" && e.ret.as_ref().map_or(true, |r| r.as_i64() == Some(0)) {
                    self.by_id[tpl].removes.push((t, p));
                }
                if op == "rename" {
                    if let Some(to) = e.to.as_deref() {
                        let (t2, _) = self.template(to);
                        self.by_id[t2].count("renamed_to");
                    }
                }
            }
        }
    }

    /// An open's end: its unit folded, its touch filled, its offsets class counted when it closed
    /// (or the trace ended with it open), not when another open took its id.
    fn retire(&mut self, mut u: Open, counted: bool) {
        let (key, groups) = self.fold(&u.ops);
        self.place(u.lane, u.slot, Unit { first: u.first, last: u.last, template: u.tpl as i64, key, groups, ops: u.ops.len() as u64, bytes: u.bytes });
        u.data.finish();
        let (cls, runs) = u.data.class();
        let tp = &mut self.by_id[u.tpl];
        if counted {
            *tp.offsets.entry(cls).or_insert(0) += 1;
            if runs > 0 {
                tp.runs.add(runs, 1);
            }
        }
        if u.bytes[0] > 0 {
            tp.readers += 1;
        }
        let keep = tp.touches.len() == 1 || (u.bytes[0] > 0 && tp.readers == 1);
        let th = &mut tp.touches[u.touch];
        th.last = u.last;
        th.bytes = u.bytes;
        th.synced = u.synced;
        th.span = u.data.span.unwrap_or(0);
        th.ndata = u.data.n;
        th.positioned = cls == "positioned";
        th.closed = true;
        if keep {
            th.within = Some(Box::new(std::mem::take(&mut u.data)));
            tp.kept.push(u.touch);
        }
        tp.prune();
    }

    fn finish(&mut self) {
        let mut open: Vec<(i64, Open)> = self.opens.drain().collect();
        open.sort_by_key(|(_, u)| (u.tpl, u.touch));
        for (_, u) in open {
            *self.notes.entry("opens_not_closed".into()).or_insert(0) += 1;
            self.retire(u, true);
        }
    }

    fn lanes_out(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for (lane, units) in &self.lanes {
            let mut units: Vec<&Unit> = units.iter().map(|u| u.as_ref().expect("every unit folded")).collect();
            units.sort_by_key(|u| u.first);
            let mut phases: Vec<Phase> = Vec::new();
            for u in units {
                let fresh = match phases.last() {
                    Some(p) => !(p.key == u.key && p.template == u.template),
                    None => true,
                };
                if fresh {
                    phases.push(Phase { key: u.key, template: u.template, units: 0, ops: 0, bytes: [0, 0], repeats: BTreeMap::new(), think: LogHist::default(), first: u.first, last: i64::MIN, groups: &[], rich: -1 });
                }
                let p = phases.last_mut().unwrap();
                p.units += 1;
                p.ops += u.ops;
                p.bytes[0] += u.bytes[0];
                p.bytes[1] += u.bytes[1];
                let mut repeated = 0;
                for (g, r) in &u.groups {
                    if *r >= 2 {
                        repeated += 1;
                        p.repeats.entry(plus(&self.groups[*g as usize])).or_default().add(*r, 1);
                    }
                }
                if repeated > p.rich {
                    p.rich = repeated;
                    p.groups = &u.groups;
                }
                if p.last != i64::MIN && u.first > p.last {
                    add(&mut p.think, u.first - p.last);
                }
                p.first = p.first.min(u.first);
                p.last = p.last.max(u.last);
            }
            let phases: Vec<Value> = phases
                .iter()
                .map(|p| {
                    let motif: Vec<String> = p.groups.iter().map(|(g, _)| {
                        let g = &self.groups[*g as usize];
                        if p.repeats.contains_key(&plus(g)) { plus(g) } else { g.clone() }
                    }).collect();
                    json!({
                        "motif": motif.join(" "),
                        "template": p.template,
                        "units": p.units,
                        "repeats": p.repeats.iter().map(|(k, h)| (k.clone(), hist(h))).collect::<Map<_, _>>(),
                        "ops": p.ops,
                        "bytes_read": p.bytes[0],
                        "bytes_written": p.bytes[1],
                        "think": hist(&p.think),
                        "first": p.first,
                        "last": p.last,
                    })
                })
                .collect();
            out.push(json!({"lane": lane, "ops": self.lane_ops.get(lane).copied().unwrap_or(0), "phases": phases}));
        }
        out
    }
}

// ---------------------------------------------------------------- the verdicts

/// Greedy passes: a pass ends at the first path it already holds.
fn epochs_of(paths: &[u32]) -> Vec<Vec<u32>> {
    let (mut out, mut cur, mut seen) = (Vec::new(), Vec::new(), HashSet::default());
    for &p in paths {
        if seen.contains(&p) {
            out.push(std::mem::take(&mut cur));
            seen.clear();
        }
        cur.push(p);
        seen.insert(p);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The chance that `k` draws with replacement from `n` repeat nothing.
pub fn p_no_repeat(k: i64, n: i64) -> f64 {
    if n > 0 { (-((k * (k - 1)) as f64) / ((2 * n) as f64)).exp() } else { 0.0 }
}

/// The smallest corpus from which `k` draws with replacement have at least a 5 % chance of no repeat.
pub fn candidates_for_choose(k: i64) -> i64 {
    k.max(((k * (k - 1)) as f64 / (2.0 * ln_refute())).ceil() as i64)
}

/// The fewest draws with replacement from `n` that repeat with 95 % probability.
pub fn touches_to_decide(n: i64) -> i64 {
    let mut k = 2i64.max(((1.0 + (1.0 + (8 * n) as f64 * ln_refute()).sqrt()) / 2.0).floor() as i64); // the real root, never above the answer
    while p_no_repeat(k, n) > P_REFUTE {
        k += 1;
    }
    k
}

/// The passes the trace completed: the last one is partial unless it is as long as the first.
fn full_passes(epochs: &[Vec<u32>]) -> &[Vec<u32>] {
    if epochs.len() <= 1 {
        return &[];
    }
    if epochs[epochs.len() - 1].len() == epochs[0].len() { epochs } else { &epochs[..epochs.len() - 1] }
}

fn top_share(mut counts: Vec<u64>, total: f64) -> f64 {
    let top = 1.max((counts.len() as f64 / 10.0).ceil() as usize);
    counts.sort_unstable_by(|a, b| b.cmp(a));
    counts.iter().take(top).sum::<u64>() as f64 / total
}

fn selection_of(paths: &[u32], epochs: &[Vec<u32>], candidates: Option<i64>, notes: &mut BTreeMap<String, Value>, tid: usize) -> Value {
    let k = paths.len() as i64;
    let d = paths.iter().collect::<HashSet<_>>().len() as i64;
    let full = full_passes(epochs);
    let lengths: Vec<i64> = full.iter().map(|e| e.len() as i64).collect();
    let mut candidates = candidates;
    if let Some(c) = candidates {
        if c < d {
            notes.insert("candidates".into(), json!(format!("--candidates {c} is fewer than the {d} distinct paths of template {tid}; the paths seen were used")));
            candidates = Some(d);
        }
    }
    let mut out = json!({"verdict": "undecided", "consistent": [], "refuted": {}});
    if full.is_empty() {
        out["consume"] = json!({"epoch": k});
        let n_star = candidates_for_choose(k);
        out["candidates_for_choose"] = json!(n_star);
        if let Some(c) = candidates {
            let p = p_no_repeat(k, c);
            out["p_no_repeat_under_choose"] = json!(p);
            if p < P_REFUTE {
                out["verdict"] = json!("consume");
                out["consistent"] = json!(["consume"]);
                out["refuted"]["choose"] = json!(format!("no repeat in {k} touches of {c} candidates has probability {} under a draw with replacement", fmt_g(p, 2)));
                out["settles"] = json!("decided");
                return out;
            }
            let ttd = touches_to_decide(c);
            out["touches_to_decide"] = json!(ttd);
            out["consistent"] = json!(["consume", "choose"]);
            out["settles"] = json!(format!("a second pass over the set: a repeat only after all {k} decides a permutation, an earlier one a draw; or {ttd} touches of {c} candidates, after which a draw with replacement has repeated with 95 % probability"));
            return out;
        }
        out["consistent"] = json!(["consume", "choose"]);
        out["settles"] = json!(format!("a second pass over the set (a repeat decides it), or the corpus size: fewer than {n_star} candidates refutes a draw with replacement (no repeat in {k} touches would have probability below {})", py_float(P_REFUTE)));
        return out;
    }
    let l = lengths[0];
    let equal = lengths.iter().all(|&x| x == l);
    if full.len() >= 2 && equal && (d as f64) < 1.5 * l as f64 {
        out["verdict"] = json!("consume");
        out["consistent"] = json!(["consume"]);
        out["consume"] = json!({"epoch": l});
        if d > l {
            out["consume"]["seen_not_every_pass"] = json!(d - l);
        }
        out["refuted"]["choose"] = json!(format!("{} full passes of equal length {l} over {d} paths; a draw with replacement repeats after about sqrt(pi N / 2) touches and its passes have unequal lengths", full.len()));
        out["settles"] = json!("decided");
        return out;
    }
    let mut per_path: HashMap<u32, u64> = HashMap::default();
    for &p in paths {
        *per_path.entry(p).or_insert(0) += 1;
    }
    let mut counts = LogHist::default();
    for &c in per_path.values() {
        counts.add(c, 1);
    }
    let share = top_share(per_path.values().copied().collect(), k as f64);
    let mean = lengths.iter().sum::<i64>() as f64 / lengths.len() as f64;
    let implied = (2.0 * mean * mean / std::f64::consts::PI).round_ties_even() as i64;
    let choose = json!({"candidates_implied": implied, "touches_per_path": hist(&counts), "top_10pct_share": share});
    if full.len() == 1 {
        out["consume"] = json!({"epoch": l});
        out["choose"] = choose;
        out["consistent"] = json!(["consume", "choose"]);
        out["settles"] = json!(format!("a second full pass: the same length {l} again is a permutation over the set; an earlier repeat, a draw with replacement from about {implied} candidates"));
        return out;
    }
    out["verdict"] = json!("choose");
    out["consistent"] = json!(["choose"]);
    out["choose"] = choose;
    let why = if !equal {
        format!("passes of unequal length {}-{}", lengths.iter().min().unwrap(), lengths.iter().max().unwrap())
    } else {
        format!("passes of equal length {l} but {d} distinct paths, more than a set of {l}")
    };
    out["refuted"]["consume"] = json!(format!("{why}: a permutation pass over the set touches each path once"));
    out["settles"] = json!("decided");
    out
}

fn order_of(epochs: &[Vec<u32>], ranks: &HashMap<u32, i64>) -> Value {
    let (mut pairs, mut asc, mut one) = (0i64, 0i64, 0i64);
    for e in epochs {
        for w in e.windows(2) {
            pairs += 1;
            let delta = ranks[&w[1]] - ranks[&w[0]];
            asc += i64::from(delta > 0);
            one += i64::from(delta == 1);
        }
    }
    let ratio = |x: i64| if pairs > 0 { json!(x as f64 / pairs as f64) } else { Value::Null };
    let mut out = json!({"pairs": pairs, "ascending": ratio(asc), "step_of_one": ratio(one)});
    if pairs < 8 {
        out["verdict"] = json!("undecided");
        out["settles"] = json!("at least 8 consecutive touches within one pass");
        return out;
    }
    let (a, s) = (asc as f64 / pairs as f64, one as f64 / pairs as f64);
    if s >= 0.9 {
        out["verdict"] = json!("sequential");
    } else if (a - 0.5).abs() <= 3.0 * (0.25 / pairs as f64).sqrt() && s < 0.2 {
        out["verdict"] = json!("permuted");
    } else {
        out["verdict"] = json!("other");
        out["note"] = json!("neither a scan (steps of one) nor a permutation (half the pairs ascending): interleaved scans, or a sort the trace does not show");
    }
    out
}

fn size_of(touches: &[&Touch]) -> Value {
    let totals: Vec<i64> = touches.iter().map(|t| t.bytes[0] + t.bytes[1]).collect();
    if totals.iter().all(|&b| b == totals[0]) {
        return json!({"verdict": "constant", "bytes": totals[0]});
    }
    let mut by_path: HashMap<u32, HashSet<i64>> = HashMap::default();
    let mut seen: HashMap<u32, u64> = HashMap::default();
    let mut order = Vec::new();
    for (t, b) in touches.iter().zip(&totals) {
        by_path.entry(t.path).or_default().insert(*b);
        let c = seen.entry(t.path).or_insert(0);
        if *c == 0 {
            order.push(t.path);
        }
        *c += 1;
    }
    let repeated: Vec<u32> = order.into_iter().filter(|p| seen[p] > 1).collect();
    if repeated.is_empty() {
        return json!({"verdict": "undecided", "settles": "a repeat of a path: the same bytes again is a size per path, other bytes a draw per open"});
    }
    let differing = repeated.iter().filter(|p| by_path[p].len() > 1).count();
    if differing > 0 {
        return json!({"verdict": "drawn", "paths_with_differing_bytes": differing, "repeated_paths": repeated.len()});
    }
    json!({"verdict": "per-path", "repeated_paths": repeated.len()})
}

// ---------------------------------------------------------------- the site kinds (§3.88, revised 2026-10-30)
//
// A site is what the digest judges: a template, or, where a template is two phases, the opens of
// one of them (`phase` `write` or `read`). Five kinds: a `set`, opened twice or more and read (a
// sample set: selection, order, size); `keyed`, its writes and reads interleaving within the trace
// (a KV cache: keys, hits, the lag from a key's write to its reads, where the keys came from; a hex
// field in the name only supports it); `bulk_write`, opened twice or more and only written, or the
// writes of a template whose writes all come before its reads or after a restore (a checkpoint, an
// index build: its saves, every field of the name, rewrites, syncs, retention, and whether it is a
// checkpoint); `restore`, reads before a template's first write of paths it never writes, starting
// in the trace's first quarter (a load of an earlier checkpoint; later, they are a reader); `within`,
// one open read at many offsets (an index traversal: reuse, skew, runs). Each kind has its facts
// (what a person knows about the scale) and its questions. All the writes before all the reads is
// two phases, a bulk write and a reader; a trace is otherwise one phase (finding phases within it
// is brief item 51).

pub const FACT_NAMES: [&str; 7] = ["candidates", "passes", "keys", "requests", "size", "keep", "writers"]; // plus values:<field>

fn fact_ask(name: &str) -> &'static str {
    match name {
        "candidates" => "How many files of this kind does the application have at the scale it runs?",
        "passes" => "How many passes over them will the benchmark run?",
        "keys" => "How many distinct keys does the application hold at the scale it runs (its pool)?",
        "requests" => "How many requests will the benchmark serve?",
        "size" => "How large is this file at the scale the application runs, in bytes?",
        "keep" => "How many saves of this kind does the application keep (its retention policy)?",
        "writers" => "How many processes write each save at the scale the application runs (plain DDP: rank 0 alone)?",
        _ => "",
    }
}

const BULK_PASSES: &str = "How many times will the benchmark write this kind of file (checkpoints, builds)?";
const WITHIN_REQUESTS: &str = "How many queries will the benchmark run?";

fn ask(q: &str) -> &'static str {
    match q {
        "selection" => "Does the application revisit a sample within one pass over the set (a draw with replacement), or touch each exactly once per pass (a permutation)?",
        "size" => "Is the number of bytes read from a file a property of the file (its size), or chosen per access?",
        "order" => "Within a pass, does the application take the files in their sorted order, or in a shuffled one?",
        "origin" => "The trace reads keys it never wrote. Were they written by an earlier run of the application (a warm cache), or by another process the trace does not cover?",
        "reuse" => "Is a key read again after its first read (a request repeating a prefix, a neighbor found again), or is each key read once?",
        "skew" => "Are the offsets read spread evenly over the file, or do a few parts of it take most of the reads?",
        "memory" => "Does this file fit in memory at the scale the application runs, so that reads after the first hit the page cache, or does it exceed it?",
        "dependency" => "Does each read's offset depend on what the previous read returned (a graph walk, a tree descent), or are the offsets independent of one another?",
        "label" => "Is this kind of file a checkpoint (the application's state, written to survive a failure and read back only after one), or output (an index build, a dataset, results)?",
        _ => "",
    }
}

/// Every question, in the order a site's kinds ask them.
pub const QUESTIONS: [&str; 9] = ["selection", "size", "order", "origin", "reuse", "skew", "memory", "dependency", "label"];

/// The candidate answers of a question.
pub fn candidates_of(q: &str) -> Option<[&'static str; 2]> {
    Some(match q {
        "selection" => ["consume", "choose"],
        "size" => ["per-path", "drawn"],
        "order" => ["sequential", "permuted"],
        "origin" => ["earlier_run", "other_writer"],
        "reuse" => ["once", "repeated"],
        "skew" => ["uniform", "skewed"],
        "memory" => ["fits", "exceeds"],
        "dependency" => ["dependent", "independent"],
        "label" => ["checkpoint", "output"],
        _ => return None,
    })
}

fn questions_of(kind: &str) -> &'static [&'static str] {
    match kind {
        "set" => &["selection", "size", "order"],
        "keyed" => &["origin", "reuse", "size"],
        "bulk_write" => &["label", "size"],
        "restore" => &["size"],
        _ => &["skew", "memory", "dependency"],
    }
}

fn at_scale(q: &str) -> &'static str {
    match q {
        "size" => "the op multiset is the same in distribution; which file gets which bytes differs",
        "order" => "the op multiset is the same; the order within a pass differs, which read-ahead sees",
        "origin" => "a warm cache serves hits the benchmark must first write; another writer is a second actor the abstract must model",
        "reuse" => "a key read again is a hit the pool must hold; keys read once make the pool a write-once log",
        "skew" => "a skewed spread keeps a hot set in the page cache at scale; a uniform one does not",
        "memory" => "a file that fits is read from storage once; one that exceeds memory is read from storage at every query",
        "dependency" => "a dependent walk is a chain of reads each waiting on the last; independent offsets can be issued together",
        "label" => "a checkpoint is saved at an interval, its saves kept or deleted by a retention policy, and read only after a failure; output is written once per build",
        _ => "",
    }
}

const LABEL_SETTLES: &str = "not from a trace: the application's documentation or its record (a checkpoint holds its state and is read back only after a failure)";

/// A site's opens: a template's, or one phase's of a template that is two (`phase`).
struct Part {
    tpl: usize,
    kind: &'static str,
    phase: Option<&'static str>,
    touches: Vec<usize>,
    read_after: bool,
    restored: usize,
    bulk: Option<Value>,
}

fn is_within(tp: &Template, touches: &[usize]) -> bool {
    touches.len() == 1 && {
        let t = &tp.touches[touches[0]];
        t.ndata >= WITHIN_MIN_READS && t.positioned
    }
}

impl Part {
    fn new(tp: &Template, kind: &'static str, phase: Option<&'static str>, touches: Vec<usize>, read_after: bool, restored: usize) -> Part {
        let mut p = Part { tpl: tp.id, kind, phase, touches, read_after, restored, bulk: None };
        if kind == "bulk_write" {
            p.bulk = Some(bulk_of(tp, &p));
        }
        p
    }

    fn paths<'a>(&'a self, tp: &'a Template) -> impl Iterator<Item = u32> + 'a {
        self.touches.iter().map(|&i| tp.touches[i].path)
    }

    fn touches<'a>(&self, tp: &'a Template) -> Vec<&'a Touch> {
        self.touches.iter().map(|&i| &tp.touches[i]).collect()
    }
}

fn half(tp: &Template, kind: &'static str, phase: &'static str, touches: Vec<usize>, read_after: bool, restored: usize) -> Vec<Part> {
    if touches.len() >= 2 {
        return vec![Part::new(tp, kind, Some(phase), touches, read_after, restored)];
    }
    if kind != "bulk_write" && is_within(tp, &touches) {
        return vec![Part::new(tp, "within", Some(phase), touches, false, 0)];
    }
    vec![]
}

/// The sites of a template, in time order. `early` ends the trace's first quarter, within which
/// a restore's reads start (§3.88, 2026-10-31: an application restoring a checkpoint does its
/// start-up I/O, reads the model, then the parameters and optimizer state, the checkpoint).
fn parts_of(tp: &Template, early: i64) -> Vec<Part> {
    let mut touches: Vec<usize> = (0..tp.touches.len()).collect();
    touches.sort_by_key(|&i| tp.touches[i].t);
    if touches.len() < 2 {
        return if !touches.is_empty() && is_within(tp, &touches) { vec![Part::new(tp, "within", None, touches, false, 0)] } else { vec![] };
    }
    let w: Vec<usize> = touches.iter().copied().filter(|&i| tp.touches[i].bytes[1] > 0).collect();
    let r: Vec<usize> = touches.iter().copied().filter(|&i| tp.touches[i].bytes[0] > 0).collect();
    if !w.is_empty() && !r.is_empty() {
        let both = w.iter().any(|&i| tp.touches[i].bytes[0] > 0);
        let t = |i: usize| tp.touches[i].t;
        if !both && t(*w.last().unwrap()) < t(r[0]) {
            let mut out = half(tp, "bulk_write", "write", w, true, 0);
            out.extend(half(tp, "set", "read", r, false, 0));
            return out;
        }
        let rp: HashSet<u32> = r.iter().map(|&i| tp.touches[i].path).collect();
        let wp: HashSet<u32> = w.iter().map(|&i| tp.touches[i].path).collect();
        if !both && t(*r.last().unwrap()) < t(w[0]) && rp.is_disjoint(&wp) {
            if r.iter().all(|&i| tp.touches[i].first <= early) {
                let mut out = half(tp, "restore", "read", r, false, 0);
                let restored = if out.is_empty() { 0 } else { rp.len() };
                out.extend(half(tp, "bulk_write", "write", w, false, restored));
                return out;
            }
            let mut out = half(tp, "set", "read", r, false, 0);
            out.extend(half(tp, "bulk_write", "write", w, false, 0));
            return out;
        }
        return vec![Part::new(tp, "keyed", None, touches, false, 0)];
    }
    let kind = if w.is_empty() { "set" } else { "bulk_write" };
    vec![Part::new(tp, kind, None, touches, false, 0)]
}

/// The template's fields over this part's paths only.
fn fields_of(tp: &Template, part: &Part) -> Vec<Map<String, Value>> {
    tp.fields_over(part.paths(tp)).iter().enumerate().map(|(i, f)| f.out(i)).collect()
}

/// The facts a person can give about a site at scale: (name, question).
fn facts_of(part: &Part) -> Vec<(String, String)> {
    let f = |n: &str| (n.to_string(), fact_ask(n).to_string());
    match part.kind {
        "set" => vec![f("candidates"), f("passes")],
        "keyed" => vec![f("keys"), f("requests")],
        "bulk_write" => {
            let b = part.bulk.as_ref().unwrap();
            let mut out: Vec<(String, String)> = b["fields"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|o| o["kind"] == "int" && o["verdict"] != "constant")
                .map(|o| {
                    let name = o["name"].as_str().unwrap();
                    (format!("values:{name}"), format!("How many values does field {name} ({}..{} in the trace) take at the scale the application runs?", py(&o["min"]), py(&o["max"])))
                })
                .collect();
            out.push(("passes".into(), BULK_PASSES.into()));
            out.push(f("keep"));
            if b["saves"]["verdict"] == "one_writer" {
                out.push(f("writers"));
            }
            out
        }
        "within" => vec![f("size"), ("requests".into(), WITHIN_REQUESTS.into())],
        _ => vec![],
    }
}

/// The least value of a fact the trace allows, and what it shows; None when it allows any.
fn fact_bound(tp: &Template, part: &Part, fact: &str) -> Option<(i64, String)> {
    match fact {
        "candidates" | "keys" => {
            let d = part.paths(tp).collect::<HashSet<_>>().len() as i64;
            Some((d, format!("the trace touched {d} distinct {} of this template", if fact == "candidates" { "paths" } else { "keys" })))
        }
        "passes" if part.kind == "set" => {
            let paths: Vec<u32> = part.paths(tp).collect();
            let n = full_passes(&epochs_of(&paths)).len() as i64;
            Some((n, format!("the trace shows {n} full passes over the set")))
        }
        "passes" => {
            let n = part.bulk.as_ref()?["saves"]["count"].as_i64().unwrap_or(0);
            Some((n, format!("the trace shows {n} saves")))
        }
        "keep" => {
            let b = part.bulk.as_ref()?;
            let (rt, n) = (&b["retention"], b["saves"]["count"].as_i64().unwrap_or(0));
            match rt["verdict"].as_str() {
                Some("kept") => Some((n, format!("the trace kept all {n} saves"))),
                Some("keeps") => {
                    let k = rt["n"].as_i64().unwrap_or(0);
                    Some((k, format!("the trace kept {k} saves at once")))
                }
                _ => None,
            }
        }
        "size" => {
            let span = span_of(&part.touches(tp));
            Some((span, format!("the trace read up to byte {span} of this file")))
        }
        _ => {
            let name = fact.strip_prefix("values:")?;
            fields_of(tp, part).into_iter().find(|o| o["name"] == name).map(|o| {
                let v = o["values"].as_i64().unwrap_or(0);
                (v, format!("field {name} took {v} values in the trace"))
            })
        }
    }
}

/// Why the trace contradicts a fact, or None.
fn fact_check(tp: &Template, part: &Part, fact: &str, value: i64) -> Option<String> {
    let (least, why) = fact_bound(tp, part, fact)?;
    (value < least).then(|| format!("{why}; {value} is {}", if fact == "size" { "smaller" } else { "fewer" }))
}

fn span_of(touches: &[&Touch]) -> i64 {
    touches.iter().filter(|t| t.ndata > 0).map(|t| t.span).max().unwrap_or(0)
}

/// A site: its kind's verdicts, its facts with the bounds the trace sets, and its `projection`,
/// the facts at scale (`--candidates` and `--epochs` for a set, else what the owner asserted)
/// with which were asserted.
fn site_of(tp: &Template, part: &Part, facts: &BTreeMap<String, i64>, candidates: Option<i64>, epochs: i64, notes: &mut BTreeMap<String, Value>) -> Value {
    let touches = part.touches(tp);
    let paths: Vec<u32> = part.paths(tp).collect();
    let distinct = paths.iter().collect::<HashSet<_>>().len();
    let mut s = json!({"template": tp.id, "phase": part.phase, "kind": part.kind, "touches": paths.len(), "distinct": distinct});
    let hex_name = tp.kinds.iter().any(|&h| h);
    let fact = |n: &str| facts.get(n).map_or(Value::Null, |v| json!(v));
    match part.kind {
        "set" => {
            let epochs_seen = epochs_of(&paths);
            let mut sorted: Vec<u32> = paths.iter().copied().collect::<HashSet<_>>().into_iter().collect();
            sorted.sort_by(|a, b| tp.paths[*a as usize].cmp(&tp.paths[*b as usize]));
            let ranks: HashMap<u32, i64> = sorted.iter().enumerate().map(|(i, &p)| (p, i as i64)).collect();
            let c = facts.get("candidates").copied().or(candidates);
            s["epochs"] = json!(epochs_seen.iter().map(Vec::len).collect::<Vec<_>>());
            s["selection"] = selection_of(&paths, &epochs_seen, c, notes, tp.id);
            s["order"] = order_of(&epochs_seen, &ranks);
            s["size"] = size_of(&touches);
            s["projection"] = json!({"candidates": c, "passes": facts.get("passes").copied().unwrap_or(epochs)});
            if hex_name && part.phase.is_none() {
                s["note"] = json!("names with a hex field, read and never written: a keyed store written elsewhere fits it as well");
            }
        }
        "keyed" => {
            s["hex_name"] = json!(hex_name);
            for (k, v) in keyed_of(&touches).as_object().unwrap() {
                s[k] = v.clone();
            }
            s["size"] = size_of(&touches);
            s["projection"] = json!({"keys": fact("keys"), "requests": fact("requests")});
        }
        "bulk_write" => {
            for (k, v) in part.bulk.as_ref().unwrap().as_object().unwrap() {
                s[k] = v.clone();
            }
            s["size"] = size_of(&touches);
            s["projection"] = facts_of(part).iter().map(|(n, _)| (n.clone(), fact(n))).collect::<Map<_, _>>().into();
        }
        "restore" => {
            s["restore"] = json!({"files": distinct, "bytes": touches.iter().map(|t| t.bytes[0]).sum::<i64>()});
            s["size"] = size_of(&touches);
            s["projection"] = json!({});
        }
        _ => {
            for (k, v) in within_of(touches[0]).as_object().unwrap() {
                s[k] = v.clone();
            }
            s["projection"] = json!({"size": fact("size"), "requests": fact("requests")});
        }
    }
    s["projection"]["asserted"] = json!(facts.keys().collect::<Vec<_>>());
    s["facts"] = facts_of(part)
        .into_iter()
        .map(|(name, q)| {
            let mut f = json!({"name": name, "ask": q});
            if let Some((least, why)) = fact_bound(tp, part, &name).filter(|(l, _)| *l >= 2) {
                f["least"] = json!(least);
                f["why"] = json!(why);
            }
            f
        })
        .collect();
    s
}

/// Keys, hits, and the lag from a key's write to each later read.
fn keyed_of(touches: &[&Touch]) -> Value {
    let mut last_write: HashMap<u32, (i64, usize)> = HashMap::default(); // path -> (time, index) of its latest write
    let mut seen_read: HashMap<u32, u64> = HashMap::default();
    let mut written: HashMap<u32, u64> = HashMap::default();
    let (mut reads, mut writes, mut after, mut before, mut rereads) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut lag_ns, mut lag_touches) = (LogHist::default(), LogHist::default());
    for (i, u) in touches.iter().enumerate() {
        let p = u.path;
        if u.bytes[1] > 0 {
            writes += 1;
            *written.entry(p).or_insert(0) += 1;
            last_write.insert(p, (u.first, i));
        }
        if u.bytes[0] > 0 {
            reads += 1;
            if let Some(&(wt, wi)) = last_write.get(&p) {
                after += 1;
                add(&mut lag_ns, u.first - wt);
                add(&mut lag_touches, (i - wi) as i64);
            } else {
                before += 1;
            }
            let c = seen_read.entry(p).or_insert(0);
            if *c > 0 {
                rereads += 1;
            }
            *c += 1;
        }
    }
    let mut per_key = LogHist::default();
    for &c in seen_read.values() {
        per_key.add(c, 1);
    }
    let share = top_share(seen_read.values().copied().collect(), reads as f64);
    let keys_reread = seen_read.values().filter(|&&c| c > 1).count();
    let keys = touches.iter().map(|t| t.path).collect::<HashSet<_>>().len();
    let lag = json!({"verdict": if after > 0 { "write_then_read" } else { "no_read_after_write" }, "ns": hist(&lag_ns), "touches": hist(&lag_touches)});
    let origin = if before == 0 {
        json!({"verdict": "traced", "settles": "decided"})
    } else {
        json!({"verdict": "undecided", "consistent": ["earlier_run", "other_writer"], "refuted": {},
               "settles": "a trace from the application's start, or of every process that writes this kind of file: a key read before any write is then a miss or another writer's"})
    };
    let reuse = if keys_reread > 0 {
        json!({"verdict": "repeated", "keys_reread": keys_reread, "settles": "decided"})
    } else {
        json!({"verdict": "undecided", "consistent": ["once", "repeated"], "refuted": {}, "keys_reread": 0,
               "settles": "a longer trace: a request repeating a prefix, or a neighbor found again, reads a key a second time"})
    };
    json!({
        "access": {"keys": keys, "written": written.len(), "read": seen_read.len(), "reads": reads, "writes": writes,
                   "written_once_share": written.values().filter(|&&c| c == 1).count() as f64 / written.len() as f64,
                   "read_after_write": after, "read_before_write": before, "rereads": rereads,
                   "reread_share": rereads as f64 / reads as f64, "top_10pct_share": share, "reads_per_key": hist(&per_key)},
        "lag": lag, "origin": origin, "reuse": reuse,
    })
}

/// One burst of a bulk write: the opens that overlap, or follow within the burst's own length.
struct Save {
    first: i64,
    last: i64,
    lanes: Vec<(i64, i64)>, // lane -> its first open, in the order the lanes came
    paths: Vec<u32>,
}

impl Save {
    fn spread(&self) -> i64 {
        let firsts = self.lanes.iter().map(|(_, f)| *f);
        firsts.clone().max().unwrap() - firsts.min().unwrap()
    }

    fn simultaneous(&self) -> bool {
        self.lanes.len() >= 2 && self.spread() as f64 <= SIMULTANEOUS * (self.last - self.first) as f64
    }
}

/// The bursts: an open starting later after the burst than the burst has lasted begins the next.
fn saves_of(touches: &[&Touch]) -> Vec<Save> {
    let mut sorted: Vec<&&Touch> = touches.iter().collect();
    sorted.sort_by_key(|t| t.first);
    let mut out: Vec<Save> = Vec::new();
    for u in sorted {
        match out.last_mut() {
            Some(cur) if u.first - cur.last <= cur.last - cur.first => {
                cur.last = cur.last.max(u.last);
                if !cur.lanes.iter().any(|(l, _)| *l == u.lane) {
                    cur.lanes.push((u.lane, u.first));
                }
                cur.paths.push(u.path);
            }
            _ => out.push(Save { first: u.first, last: u.last, lanes: vec![(u.lane, u.first)], paths: vec![u.path] }),
        }
    }
    out
}

/// What the trace deleted of this template's saves: nothing (`kept`), only saves an already
/// stable later one replaced (`keeps` N, the most it held at once after a deletion), or a save
/// before the next was stable (`unsafe`). A path the trace never wrote is a save from before it.
fn retention_of(tp: &Template, saves: &[Save]) -> Value {
    let mut save_of: HashMap<u32, i64> = HashMap::default();
    for (i, sv) in saves.iter().enumerate() {
        for p in &sv.paths {
            save_of.entry(*p).or_insert(i as i64);
        }
    }
    let ends: Vec<i64> = saves.iter().map(|s| s.last).collect();
    let (mut deleted, mut unsafe_, mut held) = (0u64, 0u64, 0usize);
    let mut dead: HashSet<usize> = HashSet::default();
    let mut removes = tp.removes.clone();
    removes.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| tp.paths[a.1 as usize].cmp(&tp.paths[b.1 as usize])));
    for (t, p) in removes {
        let k = save_of.get(&p).copied().unwrap_or(-1);
        deleted += 1;
        if !((k + 1) as usize..saves.len()).any(|j| ends[j] <= t) {
            unsafe_ += 1;
        }
        if k >= 0 {
            dead.insert(k as usize);
        }
        held = held.max(ends.iter().enumerate().filter(|(j, e)| **e <= t && !dead.contains(j)).count());
    }
    if deleted == 0 {
        return json!({"verdict": "kept", "deleted": 0, "unsafe": 0});
    }
    if unsafe_ > 0 {
        return json!({"verdict": "unsafe", "deleted": deleted, "unsafe": unsafe_,
                      "note": "an earlier save deleted before the next was stable, which contradicts what a checkpoint is for"});
    }
    json!({"verdict": "keeps", "n": held, "deleted": deleted, "unsafe": 0})
}

/// The saves (bursts) and their cadence; every field a part of the name that varies or stays
/// over the saves; the paths written again; the syncs; the retention; and whether it is a
/// checkpoint, which the trace refutes (a rewrite, a read after the first write) but never decides.
fn bulk_of(tp: &Template, part: &Part) -> Value {
    let touches = part.touches(tp);
    let saves = saves_of(&touches);
    let mut per_path: HashMap<u32, u64> = HashMap::default();
    for p in part.paths(tp) {
        *per_path.entry(p).or_insert(0) += 1;
    }
    let rewritten = per_path.values().filter(|&&c| c > 1).count();
    let fields: Vec<Value> = fields_of(tp, part)
        .into_iter()
        .map(|mut o| {
            if o["values"].as_u64().unwrap_or(0) >= 2 {
                o.insert("verdict".into(), json!("varies"));
            } else if saves.len() >= 2 {
                o.insert("verdict".into(), json!("constant"));
            } else {
                let name = o["name"].as_str().unwrap().to_string();
                o.insert("verdict".into(), json!("undecided"));
                o.insert("settles".into(), json!(format!("a second save: a field keeping its one value over the saves is the same in every save's name; how many values it takes at scale is the fact values:{name}")));
            }
            Value::Object(o)
        })
        .collect();
    let multi: Vec<&Save> = saves.iter().filter(|s| s.lanes.len() >= 2).collect();
    let verdict = if multi.iter().any(|s| !s.simultaneous()) {
        "staggered"
    } else if multi.is_empty() {
        "one_writer"
    } else if multi.len() == saves.len() {
        "simultaneous"
    } else {
        "mixed"
    };
    let (mut lanes, mut spread, mut length, mut interval) = (LogHist::default(), LogHist::default(), LogHist::default(), LogHist::default());
    for (i, sv) in saves.iter().enumerate() {
        lanes.add(sv.lanes.len() as u64, 1);
        add(&mut spread, sv.spread());
        add(&mut length, sv.last - sv.first);
        if i > 0 {
            add(&mut interval, sv.first - saves[i - 1].first);
        }
    }
    let retention = retention_of(tp, &saves);
    let synced = touches.iter().filter(|t| t.synced).count();
    let mut rewrite = json!({"verdict": if rewritten > 0 { "overwrite" } else { "new_names" }, "rewritten_paths": rewritten});
    let (mut against, mut supports): (Vec<String>, Vec<String>) = (vec![], vec![]);
    if rewritten > 0 {
        rewrite["note"] = json!("a checkpoint is never overwritten: a rewritten path is evidence against one");
        against.push(format!("{rewritten} path(s) written again: a checkpoint is never overwritten"));
    }
    if part.read_after {
        against.push("read after its first write: a checkpoint is read only after a failure, in another trace".into());
    }
    if verdict == "simultaneous" {
        supports.push("every save starts on all its lanes together".into());
    } else if verdict == "one_writer" {
        supports.push("one writing lane, as plain DDP's rank 0".into());
    }
    if synced == touches.len() {
        supports.push("every file fsynced".into());
    }
    if part.restored > 0 {
        supports.push(format!("{} file(s) of it read before its first write and never written: a restore", part.restored));
    }
    if retention["verdict"] == "keeps" {
        supports.push(format!("earlier saves deleted only once the next was stable (keeps {})", retention["n"]));
    }
    let label = if !against.is_empty() {
        json!({"verdict": "output", "refuted": {"checkpoint": against.join("; ")}, "supports": supports, "settles": "decided"})
    } else {
        json!({"verdict": "undecided", "consistent": ["checkpoint", "output"], "refuted": {}, "supports": supports, "settles": LABEL_SETTLES})
    };
    json!({"fields": fields, "rewrite": rewrite, "sync": {"fsync_share": synced as f64 / touches.len() as f64, "opens": touches.len()},
           "saves": {"verdict": verdict, "count": saves.len(), "lanes": hist(&lanes), "start_spread_ns": hist(&spread),
                     "duration_ns": hist(&length), "interval_ns": hist(&interval)},
           "retention": retention, "label": label})
}

/// Reuse and skew of the offsets one open read, and the sequential runs.
fn within_of(t: &Touch) -> Value {
    let d0 = Data::default();
    let data = t.within.as_deref().unwrap_or(&d0);
    let (n, d) = (data.n, data.counts.len() as u64);
    let share = top_share(data.counts.values().copied().collect(), n as f64);
    // under a uniform spread the counts per offset are Poisson with mean n/d, and the top tenth of
    // them sit about 1.75 standard deviations up, so the top tenth's share is about 0.1 (1 + 1.75 / sqrt(n/d))
    let uniform = 0.1 * (1.0 + 1.75 / (n as f64 / d as f64).sqrt());
    let mut skew = json!({"top_10pct_share": share, "uniform_share": uniform});
    let undecided = |skew: &mut Value, settles: String| {
        skew["verdict"] = json!("undecided");
        skew["consistent"] = json!(["uniform", "skewed"]);
        skew["refuted"] = json!({});
        skew["settles"] = json!(settles);
    };
    if n < 50 {
        undecided(&mut skew, format!("50 or more reads ({n} seen): the share the top tenth of the offsets takes against a uniform spread's"));
    } else if share >= 1.5 * uniform {
        skew["verdict"] = json!("skewed");
        skew["settles"] = json!("decided");
    } else if share <= 1.2 * uniform {
        skew["verdict"] = json!("uniform");
        skew["settles"] = json!("decided");
    } else {
        undecided(&mut skew, format!("more reads: a top-tenth share settling within 1.2x of a uniform spread's ({uniform:.2}) is uniform, above 1.5x skewed"));
    }
    json!({
        "reads": n, "distinct_offsets": d, "reuse_share": 1.0 - d as f64 / n as f64, "span": data.span.unwrap_or(0), "runs": hist(&data.run_hist), "skew": skew,
        "memory": {"verdict": "undecided", "consistent": ["fits", "exceeds"], "refuted": {},
                   "settles": "not from a trace: the file's size at scale against the hosts' memory (the facts `size` and the host's memory)"},
        "dependency": {"verdict": "undecided", "consistent": ["dependent", "independent"], "refuted": {},
                       "settles": "not from a trace: the application's source, or a trace with the same query twice (a dependent walk repeats its offsets exactly)"},
    })
}

/// The undecided verdicts as questions, ranked by the candidates' divergence at scale: for a
/// set's selection the share of the set a draw never reads at the site's passes (its
/// `projection`, a fact the owner gave, else `--epochs`); zero, with what differs said, for the rest.
fn open_of(sites: &[Value], epochs: i64) -> Vec<Value> {
    let mut items = Vec::new();
    for s in sites {
        for q in questions_of(s["kind"].as_str().unwrap()) {
            let v = &s[*q];
            if v["verdict"] != "undecided" {
                continue;
            }
            let mut it = json!({"site": s["template"], "question": q, "candidates": candidates_of(q).unwrap(), "settles": v["settles"], "divergence": 0.0, "ask": ask(q)});
            if !s["phase"].is_null() {
                it["phase"] = s["phase"].clone();
            }
            let e = s["projection"].get("passes").and_then(Value::as_i64).filter(|&p| p != 0).unwrap_or(epochs);
            if *q == "selection" {
                let never = (-(e as f64)).exp();
                it["divergence"] = json!(never);
                it["at_scale"] = json!({
                    "epochs": e,
                    "measure": "share of the set never read; share of touches that re-read a path",
                    "consume": {"never_read_share": 0.0, "repeat_share": (e - 1) as f64 / e as f64},
                    "choose": {"never_read_share": never, "repeat_share": 1.0 - (1.0 - never) / e as f64},
                });
            } else {
                it["at_scale"] = json!({"epochs": e, "note": at_scale(q)});
            }
            items.push(it);
        }
    }
    ranked(items)
}

fn ranked(mut items: Vec<Value>) -> Vec<Value> {
    items.sort_by(|a, b| b["divergence"].as_f64().unwrap().partial_cmp(&a["divergence"].as_f64().unwrap()).unwrap());
    for (i, it) in items.iter_mut().enumerate() {
        it["rank"] = json!(i + 1);
    }
    items
}

/// What decided a verdict, said against the candidate an answer named.
fn evidence(q: &str, v: &Value, against: &str) -> String {
    match q {
        "selection" => match v["refuted"].get(against).and_then(Value::as_str) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => format!("passes {}", v.get("consume").and_then(|c| c.get("epoch")).map(py).unwrap_or_default()).trim().to_string(),
        },
        "order" => format!("ascending {:.2} of {} adjacent pairs, step of one {:.2}", v["ascending"].as_f64().unwrap_or(0.0), v["pairs"], v["step_of_one"].as_f64().unwrap_or(0.0)),
        "origin" => "every key read was written earlier in the trace".into(),
        "reuse" => format!("{} key(s) read more than once", v["keys_reread"]),
        "skew" => {
            let share = format!("the top tenth of the offsets took {:.2} of the reads, about {:.2} under a uniform spread", v["top_10pct_share"].as_f64().unwrap_or(0.0), v["uniform_share"].as_f64().unwrap_or(0.0));
            share + if v["verdict"] == "uniform" { ", uniform within noise" } else { "" }
        }
        "label" => py(&v["refuted"]["checkpoint"]),
        _ if v["verdict"] == "constant" => format!("every open moved {} B", v["bytes"]),
        _ if v["verdict"] == "per-path" => format!("{} repeated path(s) always moved the same bytes", v["repeated_paths"]),
        _ => format!("{} of {} repeated path(s) moved differing bytes", v["paths_with_differing_bytes"], v["repeated_paths"]),
    }
}

// ---------------------------------------------------------------- the analysis

/// A trace reduced to what the document is made from, kept so that answers can be applied and
/// the document remade without reading the trace again.
pub struct Analysis {
    pub header: Value,
    pub source: String,
    pub root: String,
    d: Digester,
    parts: Vec<Vec<Part>>,
    names: Vec<String>,
    metrics: Value,
}

/// Reads exported lines, the header first, into a digester and the metrics.
struct Reader {
    no: usize,
    header: Option<Value>,
    d: Digester,
    fm: FileMetrics,
    skipped: u64,
}

impl Reader {
    fn new() -> Reader {
        Reader { no: 0, header: None, d: Digester::default(), fm: FileMetrics::new(None), skipped: 0 }
    }

    fn line(&mut self, line: &str) -> Result<()> {
        self.no += 1;
        let no = self.no;
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        if self.header.is_some() {
            let ev: Ev = match serde_json::from_str(line) {
                Ok(ev) => ev,
                Err(e) => match serde_json::from_str::<Value>(line) {
                    Err(e) => bail!("line {no}: {e}"),
                    Ok(v) if !v.is_object() => bail!("line {no}: not a JSON object"),
                    Ok(_) => bail!("line {no}: {e}"),
                },
            };
            self.d.event(&ev);
            // the metrics read the line as the runner does; a line it refuses is counted, not read
            match serde_json::from_str::<Line>(line) {
                Ok(l) if self.fm.line(&l).is_ok() => {}
                _ => self.skipped += 1,
            }
            return Ok(());
        }
        let doc: Value = serde_json::from_str(line).map_err(|e| anyhow!("line {no}: {e}"))?;
        if !doc.is_object() {
            bail!("line {no}: not a JSON object");
        }
        if doc.get("aeiou_trace").and_then(Value::as_u64) != Some(TRACE_FORMAT) {
            bail!("not an aeiou_trace: {TRACE_FORMAT} file");
        }
        self.header = Some(doc);
        Ok(())
    }
}

/// Exported lines written by an export, read as they are written.
struct Sink<'a> {
    buf: Vec<u8>,
    r: &'a mut Reader,
    err: Option<anyhow::Error>,
}

impl Write for Sink<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(b);
        let mut from = 0;
        while let Some(i) = self.buf[from..].iter().position(|&c| c == b'\n') {
            let line = String::from_utf8_lossy(&self.buf[from..from + i + 1]).into_owned();
            from += i + 1;
            if self.err.is_none() {
                if let Err(e) = self.r.line(&line) {
                    self.err = Some(e);
                }
            }
        }
        self.buf.drain(..from);
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Read an exported trace file (carries its root; `root`, when given, must be it), or an strace
/// or an object-store request log (needs `root`: the directory, or the bucket and prefix), whose
/// calls the export turns into the exported lines. `first` is the input's first line.
pub fn analyze(open: &Opener, first: &str, root: Option<&str>, exclude: &[String], cwd: Option<&str>) -> Result<Analysis> {
    let mut r = Reader::new();
    let source;
    let root_out;
    if first.starts_with("{\"aeiou_trace\"") {
        strace::for_each_line(open()?, |line| r.line(line))?;
        let header = r.header.clone().unwrap_or(json!({}));
        let hroot = header.get("root").and_then(Value::as_str).unwrap_or("").to_string();
        if let Some(root) = root {
            if strace::abspath(root) != hroot {
                bail!("an exported trace file carries its root ({hroot}); --root {root} is not it");
            }
        }
        if r.d.ops == 0 {
            bail!("a trace file with no call in it");
        }
        source = "trace".to_string();
        root_out = hroot;
    } else {
        let Some(root) = root else {
            bail!("an strace or an object-store request log needs --root (the directory, or BUCKET[/PREFIX])");
        };
        let mut sink = Sink { buf: Vec::new(), r: &mut r, err: None };
        let summary = if let Some(kind) = objectlog::source_of(first) {
            let s = objectlog::export(open, root, kind, &mut sink)?;
            if s.lines == 0 {
                bail!("no request under --root {} ({} request(s) read)", root.trim_matches('/'), s.header["requests"].as_u64().unwrap_or(0));
            }
            s
        } else {
            let s = export::strace(open, &strace::abspath(root), exclude, cwd, &mut sink)?;
            if s.lines == 0 {
                let r = s.header["root"].as_str().unwrap_or("").to_string();
                bail!("{}", strace::nothing_counted(&[r], s.seen));
            }
            s
        };
        if let Some(e) = sink.err.take() {
            return Err(e);
        }
        source = summary.header.get("source").and_then(Value::as_str).unwrap_or("strace").to_string();
        root_out = summary.header.get("root").and_then(Value::as_str).unwrap_or("").to_string();
    }
    let header = r.header.take().unwrap_or(json!({}));
    let mut d = r.d;
    d.finish();
    let mut mh = header.clone();
    mh["source"] = json!(source);
    let mut metrics = r.fm.document(&mh)?;
    if r.skipped > 0 {
        metrics["notes"]["lines_not_counted"] = json!(r.skipped);
    }
    let units = d.lanes.values().flatten().flatten();
    let (start, end) = units.fold((i64::MAX, i64::MIN), |(s, e), u| (s.min(u.first), e.max(u.last)));
    let (start, end) = if start == i64::MAX { (0, 0) } else { (start, end) };
    let early = start + (end - start).div_euclid(4);
    let parts = d.by_id.iter().map(|t| parts_of(t, early)).collect();
    let names = d.by_id.iter().map(Template::name).collect();
    Ok(Analysis { header, source, root: root_out, d, parts, names, metrics })
}

impl Analysis {
    fn part(&self, tid: usize, phase: Option<&str>) -> &Part {
        self.parts[tid].iter().find(|p| p.phase == phase).unwrap()
    }

    /// The digest document, with `answers` (facts and answers to open questions, §3.88) checked
    /// against the trace and recorded under `asserted` with `by` and `at` (an answer carrying its
    /// own `by`, a record's, keeps it), or under `rejected` with why; the verdicts stay the
    /// trace's. `application` is what the owner or a record said the traced application is, with
    /// the blind spots that follow, each also a note.
    pub fn document(&self, o: &Opts, answers: &[Value], by: Option<&str>, at: Option<&str>, application: &Value) -> Value {
        let mut notes: BTreeMap<String, Value> = BTreeMap::new();
        if let Some(m) = self.header.get("notes").and_then(Value::as_object) {
            for (k, v) in m {
                notes.insert(k.clone(), v.clone());
            }
        }
        for (k, v) in &self.d.notes {
            notes.insert(k.clone(), json!(v));
        }
        for b in application.get("blind_spots").and_then(Value::as_array).into_iter().flatten() {
            notes.insert(format!("blind_spot_{}", py(&b["kind"])), b["note"].clone());
        }
        let mut asserted: Vec<Value> = Vec::new();
        let mut rejected: Vec<(usize, Value)> = Vec::new(); // (the answer's index, so rejections keep the answers' order)
        let mut facts: HashMap<(usize, Option<&str>), BTreeMap<String, i64>> = HashMap::default();
        let mut questions: Vec<(usize, Vec<&Part>)> = Vec::new();
        let reject = |rejected: &mut Vec<(usize, Value)>, i: usize, why: String| {
            let mut a = answers[i].clone();
            if !a.is_object() {
                a = json!({});
            }
            a["why"] = json!(why);
            rejected.push((i, a));
        };
        let truthy = |v: Option<&Value>| v.filter(|x| !matches!(x, Value::Null | Value::Bool(false)) && x.as_str() != Some("") && x.as_i64() != Some(0)).cloned();
        let record = |asserted: &mut Vec<Value>, part: &Part, a: &Value, kind: &str, value: Value| -> Value {
            let mut e = Map::new();
            e.insert("site".into(), json!(part.tpl));
            e.insert("template".into(), json!(self.names[part.tpl]));
            e.insert(kind.into(), a[kind].clone());
            e.insert(if kind == "fact" { "value" } else { "answer" }.into(), value);
            if let Some(p) = part.phase {
                e.insert("phase".into(), json!(p));
            }
            if let Some(n) = truthy(a.get("note")) {
                e.insert("note".into(), n);
            }
            e.insert("by".into(), truthy(a.get("by")).unwrap_or_else(|| json!(by)));
            e.insert("at".into(), truthy(a.get("at")).unwrap_or_else(|| json!(at)));
            if let Some(s) = truthy(a.get("source")) {
                e.insert("source".into(), s);
            }
            let e = Value::Object(e);
            asserted.push(e.clone());
            e
        };

        for (i, a) in answers.iter().enumerate() {
            // the sites an answer is about: its template's, or the one of the phase it names
            let tid = a.get("template").and_then(Value::as_str).and_then(|n| self.names.iter().position(|x| x == n));
            let Some(tid) = tid else {
                reject(&mut rejected, i, "no such template in the trace".into());
                continue;
            };
            let mut ps: Vec<&Part> = self.parts[tid].iter().collect();
            if ps.is_empty() {
                reject(&mut rejected, i, if a.get("fact").is_some() { "not a site".into() } else { "not a site: the template was opened once and read in one run, or never opened".into() });
                continue;
            }
            if let Some(ph) = a.get("phase") {
                let named: Vec<&Part> = ps.iter().copied().filter(|p| p.phase.map_or(false, |x| ph.as_str() == Some(x))).collect();
                if named.is_empty() {
                    let have: Vec<&str> = ps.iter().filter_map(|p| p.phase).collect();
                    reject(&mut rejected, i, format!("no such phase here: {}", if have.is_empty() { "none".into() } else { have.join(", ") }));
                    continue;
                }
                ps = named;
            }
            if let Some(fact) = a.get("fact") {
                let name = fact.as_str().unwrap_or("\0");
                let v = a.get("value");
                let having: Vec<&Part> = ps.iter().copied().filter(|p| facts_of(p).iter().any(|(n, _)| n == name)).collect();
                if having.is_empty() {
                    let names: Vec<String> = facts_of(ps[0]).into_iter().map(|(n, _)| n).collect();
                    reject(&mut rejected, i, format!("no such fact for a {} site: {}", ps[0].kind, if names.is_empty() { "none".into() } else { names.join(", ") }));
                    continue;
                }
                let value = match v.and_then(|v| if v.is_i64() || v.is_u64() { v.as_i64() } else { None }) {
                    Some(x) if x >= 1 => x,
                    _ => {
                        reject(&mut rejected, i, "not a whole number of at least 1".into());
                        continue;
                    }
                };
                if let Some(why) = having.iter().find_map(|p| fact_check(&self.d.by_id[p.tpl], p, name, value)) {
                    reject(&mut rejected, i, why);
                    continue;
                }
                for p in having {
                    facts.entry((p.tpl, p.phase)).or_default().insert(name.to_string(), value);
                    record(&mut asserted, p, a, "fact", json!(value));
                }
            } else if a.get("question").is_some() {
                questions.push((i, ps));
            } else {
                reject(&mut rejected, i, "neither a fact nor a question".into());
            }
        }

        let empty = BTreeMap::new();
        let mut sites: Vec<Value> = Vec::new();
        for (tid, ps) in self.parts.iter().enumerate() {
            for p in ps {
                let tp = &self.d.by_id[tid];
                sites.push(site_of(tp, p, facts.get(&(tid, p.phase)).unwrap_or(&empty), o.candidates, o.epochs, &mut notes));
            }
        }
        let index_of = |sites: &[Value], tid: usize, phase: Option<&str>| sites.iter().position(|s| s["template"] == json!(tid) && s["phase"].as_str() == phase).unwrap();

        for (i, ps) in questions {
            let a = &answers[i];
            let q = a["question"].as_str().unwrap_or("\0");
            let ans = a.get("answer").cloned().unwrap_or(Value::Null);
            let having: Vec<(&Part, usize)> = ps.iter().map(|p| (*p, index_of(&sites, p.tpl, p.phase))).filter(|(p, _)| questions_of(p.kind).contains(&q)).collect();
            if having.is_empty() {
                reject(&mut rejected, i, format!("no such question for a {} site: {}", ps[0].kind, questions_of(ps[0].kind).join(", ")));
                continue;
            }
            let cands = candidates_of(q).unwrap();
            let ans_s = ans.as_str().unwrap_or("\0");
            if !cands.contains(&ans_s) && ans_s != UNKNOWN {
                reject(&mut rejected, i, format!("not one of {}, {UNKNOWN}", cands.join(", ")));
                continue;
            }
            for (p, si) in having {
                let v = &sites[si][q];
                let verdict = v["verdict"].as_str().unwrap_or("").to_string();
                if verdict != "undecided" {
                    if ans_s == verdict || ans_s == UNKNOWN {
                        let n = notes.get("redundant_answers").and_then(Value::as_i64).unwrap_or(0);
                        notes.insert("redundant_answers".into(), json!(n + 1));
                    } else {
                        let why = format!("the trace decided {verdict}: {}", evidence(q, v, ans_s));
                        reject(&mut rejected, i, why);
                    }
                } else {
                    let e = record(&mut asserted, p, a, "question", ans.clone());
                    let mut mine = Map::new();
                    for k in ["answer", "by", "at", "note"] {
                        if let Some(x) = e.get(k) {
                            mine.insert(k.into(), x.clone());
                        }
                    }
                    sites[si][q]["asserted"] = Value::Object(mine);
                }
            }
        }

        let mut open_items = Vec::new();
        for mut it in open_of(&sites, o.epochs) {
            let si = index_of(&sites, it["site"].as_u64().unwrap() as usize, it.get("phase").and_then(Value::as_str));
            let answer = sites[si][it["question"].as_str().unwrap()].get("asserted").and_then(|a| a.get("answer")).cloned();
            match answer {
                None => open_items.push(it),
                Some(a) if a == UNKNOWN => {
                    it["answered"] = json!(UNKNOWN);
                    open_items.push(it);
                }
                _ => {}
            }
        }
        rejected.sort_by_key(|(i, _)| *i);
        let rejected: Vec<Value> = rejected.into_iter().map(|(_, r)| r).collect();
        if !rejected.is_empty() {
            notes.insert("rejected_answers".into(), json!(rejected.len()));
        }
        json!({
            "aeiou_digest": FORMAT,
            "source": self.source,
            "root": self.root,
            "ops": self.d.ops,
            "contexts": self.d.lane_ops.len(),
            "projection": {"epochs": o.epochs, "candidates": o.candidates},
            "notes": notes,
            "templates": self.d.by_id.iter().map(|t| t.out(o.paths, &self.names[t.id])).collect::<Vec<_>>(),
            "lanes": self.d.lanes_out(),
            "sites": sites,
            "open": ranked(open_items),
            "asserted": asserted,
            "rejected": rejected,
            "application": application,
            "metrics": self.metrics,
        })
    }

    /// The facts of a site and whether the trace contradicts a value of one (for the tests).
    pub fn fact_names(&self, tid: usize, phase: Option<&str>) -> Vec<String> {
        facts_of(self.part(tid, phase)).into_iter().map(|(n, _)| n).collect()
    }
}

// ---------------------------------------------------------------- the text

/// A site as the report and the interview name it: its template, and its phase when the template is two.
pub fn site_name(site: &Value, phase: &Value) -> String {
    match phase.as_str() {
        Some(p) => format!("{}/{p}", py(site)),
        None => py(site),
    }
}

/// One line of evidence about a site, by kind.
pub fn site_line(s: &Value) -> String {
    let f2 = |v: &Value| format!("{:.2}", v.as_f64().unwrap_or(0.0));
    match s["kind"].as_str().unwrap_or("") {
        "set" => format!("{} touches of {} paths, passes {}", s["touches"], s["distinct"], py(&s["epochs"])),
        "keyed" => {
            let a = &s["access"];
            format!("keyed: {} keys, {} written, {} reads ({} after a write, {} re-reads)", a["keys"], a["written"], a["reads"], a["read_after_write"], a["rereads"])
        }
        "bulk_write" => format!(
            "bulk write: {} files in {} save(s), {} rewritten, fsync on {}",
            s["distinct"],
            s["saves"]["count"],
            s["rewrite"]["rewritten_paths"],
            pct(s["sync"]["fsync_share"].as_f64().unwrap_or(0.0), 0)
        ),
        "restore" => format!("restore: {} files, {} read before the first write", s["restore"]["files"], bytes_text(s["restore"]["bytes"].as_i64().unwrap_or(0))),
        _ => format!(
            "within one open: {} reads at {} offsets over {}, top tenth {}",
            s["reads"],
            s["distinct_offsets"],
            bytes_text(s["span"].as_i64().unwrap_or(0)),
            f2(&s["skew"]["top_10pct_share"])
        ),
    }
}

fn asserted_text(v: &Value) -> String {
    match v.get("asserted") {
        Some(a) if a.is_object() => format!("; asserted {}", py(&a["answer"])),
        _ => String::new(),
    }
}

fn verdict_line(name: &str, v: &Value) -> String {
    let mut line = format!("    {name}: {}", py(&v["verdict"]));
    if v["verdict"] == "undecided" {
        line += &format!("; settles: {}", py(&v["settles"]));
    }
    line + &asserted_text(v)
}

fn size_line(z: &Value) -> String {
    let mut line = format!("    size: {}", py(&z["verdict"]));
    if z["verdict"] == "constant" {
        line += &format!(" {} B", z["bytes"]);
    } else if z["verdict"] == "undecided" {
        line += &format!("; settles: {}", py(&z["settles"]));
    }
    line + &asserted_text(z)
}

fn kind_lines(s: &Value) -> Vec<String> {
    let mut out = Vec::new();
    match s["kind"].as_str().unwrap_or("") {
        "keyed" => {
            let lag = &s["lag"];
            let mut line = format!("    lag: {}", py(&lag["verdict"]));
            if lag["ns"]["n"].as_u64().unwrap_or(0) > 0 {
                line += &format!(" (write to read p50 {}, {} touches between, {} read(s))", ns_text(p50(&lag["ns"]) as f64), p50(&lag["touches"]), lag["ns"]["n"]);
            }
            out.push(line);
            for q in ["origin", "reuse"] {
                out.push(verdict_line(q, &s[q]));
            }
        }
        "bulk_write" => {
            let sv = &s["saves"];
            let mut line = format!("    saves: {}, {} save(s)", py(&sv["verdict"]), sv["count"]);
            if sv["interval_ns"]["n"].as_u64().unwrap_or(0) > 0 {
                line += &format!(", interval p50 {}", ns_text(p50(&sv["interval_ns"]) as f64));
            }
            out.push(line);
            let rt = &s["retention"];
            let mut line = format!("    retention: {}", py(&rt["verdict"]));
            if rt["verdict"] == "keeps" {
                line += &format!(" {}", rt["n"]);
            }
            if rt["unsafe"].as_u64().unwrap_or(0) > 0 {
                line += &format!(" ({} file(s) deleted before the next save was stable)", rt["unsafe"]);
            }
            out.push(line);
            out.push(format!("    rewrite: {} ({} path(s) written again)", py(&s["rewrite"]["verdict"]), s["rewrite"]["rewritten_paths"]));
            for f in s["fields"].as_array().into_iter().flatten() {
                let rng = if f["kind"] == "int" { format!(" {}..{}", py(&f["min"]), py(&f["max"])) } else { String::new() };
                let n = f["values"].as_i64().unwrap_or(0);
                let mut line = format!("    field {}: {} ({} value{}{rng})", py(&f["name"]), py(&f["verdict"]), n, if n != 1 { "s" } else { "" });
                if f["verdict"] == "undecided" {
                    line += &format!("; settles: {}", py(&f["settles"]));
                }
                out.push(line);
            }
            let lb = &s["label"];
            let mut line = format!("    label: {}", py(&lb["verdict"]));
            for (k, why) in lb["refuted"].as_object().into_iter().flatten() {
                line += &format!("; {k} refuted: {}", py(why));
            }
            let supports: Vec<String> = lb["supports"].as_array().into_iter().flatten().map(py).collect();
            if !supports.is_empty() {
                line += &format!("; supports: {}", supports.join(", "));
            }
            if lb["verdict"] == "undecided" {
                line += &format!("; settles: {}", py(&lb["settles"]));
            }
            out.push(line + &asserted_text(lb));
        }
        "restore" => {}
        _ => {
            out.push(format!("    reuse: {:.2} of the reads revisit an offset; runs p50 {} max {}", s["reuse_share"].as_f64().unwrap_or(0.0), p50(&s["runs"]), s["runs"]["max"]));
            for q in ["skew", "memory", "dependency"] {
                out.push(verdict_line(q, &s[q]));
            }
        }
    }
    if s.get("size").is_some() {
        out.push(size_line(&s["size"]));
    }
    out
}

/// The order a site's projection is written in: its facts' order.
fn projection_rank(k: &str) -> (usize, String) {
    let r = match k {
        _ if k.starts_with("values:") => 0,
        "candidates" | "keys" | "size" => 1,
        "passes" | "requests" => 2,
        "keep" => 3,
        "writers" => 4,
        _ => 5,
    };
    (r, k.to_string())
}

/// The text report of a document.
pub fn render(doc: &Value) -> String {
    let mut out: Vec<String> = Vec::new();
    let arr = |v: &Value| v.as_array().cloned().unwrap_or_default();
    let what = match doc.get("trace") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => py(&doc["root"]),
    };
    let (templates, sites, open) = (arr(&doc["templates"]), arr(&doc["sites"]), arr(&doc["open"]));
    let (asserted, rejected) = (arr(&doc["asserted"]), arr(&doc["rejected"]));
    let mut head = format!(
        "digest of {what}: {} ops on {} lane(s) under {}; {} path template(s), {} site(s), {} open question(s)",
        doc["ops"],
        doc["contexts"],
        py(&doc["root"]),
        templates.len(),
        sites.len(),
        open.len()
    );
    if !asserted.is_empty() {
        head += &format!(", {} asserted", asserted.len());
    }
    if !rejected.is_empty() {
        head += &format!(", {} rejected", rejected.len());
    }
    out.push(head);
    out.push("templates:".into());
    for t in &templates {
        let mut ops: Vec<(&String, i64)> = t["ops"].as_object().into_iter().flatten().map(|(k, v)| (k, v.as_i64().unwrap_or(0))).collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let ops: Vec<String> = ops.iter().map(|(k, v)| format!("{k} {v}")).collect();
        let mut line = format!("  {:>3}  {}  {} paths, {} opens; ops {}", py(&t["id"]), py(&t["template"]), t["paths"], t["opens"], ops.join(", "));
        if t["bytes_read"].as_i64().unwrap_or(0) != 0 {
            line += &format!("; read {} (request p50 {})", bytes_text(t["bytes_read"].as_i64().unwrap_or(0)), bytes_text(p50(&t["read_size"])));
        }
        if t["bytes_written"].as_i64().unwrap_or(0) != 0 {
            line += &format!("; wrote {} (request p50 {})", bytes_text(t["bytes_written"].as_i64().unwrap_or(0)), bytes_text(p50(&t["write_size"])));
        }
        let offs: Vec<String> = t["offsets"].as_object().into_iter().flatten().map(|(k, v)| format!("{k} {v}")).collect();
        if !offs.is_empty() {
            line += &format!("; offsets {}", offs.join(", "));
        }
        out.push(line);
        for f in arr(&t["fields"]) {
            let rng = if f["kind"] == "int" { format!(" {}..{}", py(&f["min"]), py(&f["max"])) } else { String::new() };
            let width = match f["width"].as_i64() {
                Some(w) if w != 0 => format!("width {w}"),
                _ => "no fixed width".into(),
            };
            out.push(format!("       {}: {}, {width}, {} value(s){rng}", py(&f["name"]), py(&f["kind"]), f["values"]));
        }
        for p in arr(&t["raw_paths"]) {
            out.push(format!("       {}", py(&p)));
        }
    }
    out.push("lanes:".into());
    for lane in arr(&doc["lanes"]) {
        out.push(format!("  lane {}: {} ops", lane["lane"], lane["ops"]));
        for (i, p) in arr(&lane["phases"]).iter().enumerate() {
            let mut line = format!("    phase {}  {} x{} on template {}", i + 1, py(&p["motif"]), p["units"], p["template"]);
            let reps: Vec<String> = p["repeats"].as_object().into_iter().flatten().map(|(k, h)| format!("{k} p50 {} max {}", p50(h), h["max"])).collect();
            if !reps.is_empty() {
                line += &format!("; repeats {}", reps.join(", "));
            }
            if p["think"]["n"].as_u64().unwrap_or(0) > 0 {
                line += &format!("; think p50 {}", ns_text(p50(&p["think"]) as f64));
            }
            if p["bytes_read"].as_i64().unwrap_or(0) != 0 {
                line += &format!("; read {}", bytes_text(p["bytes_read"].as_i64().unwrap_or(0)));
            }
            if p["bytes_written"].as_i64().unwrap_or(0) != 0 {
                line += &format!("; wrote {}", bytes_text(p["bytes_written"].as_i64().unwrap_or(0)));
            }
            out.push(line);
        }
    }
    out.push("sites:".into());
    for s in &sites {
        let mut line = format!("  site {} (template {}): {}", site_name(&s["template"], &s["phase"]), s["template"], site_line(s));
        if let Some(pr) = s.get("projection").filter(|p| p.is_object()) {
            let asserted_facts: Vec<String> = pr["asserted"].as_array().into_iter().flatten().map(py).collect();
            if !asserted_facts.is_empty() {
                let mut kv: Vec<(&String, &Value)> = pr.as_object().unwrap().iter().filter(|(k, v)| *k != "asserted" && !v.is_null()).collect();
                kv.sort_by_key(|(k, _)| projection_rank(k));
                let kv: Vec<String> = kv.iter().map(|(k, v)| format!("{k} {}", py(v))).collect();
                line += &format!("; projected at {} (asserted: {})", kv.join(", "), asserted_facts.join(", "));
            }
        }
        out.push(line);
        if s["kind"] != "set" {
            out.extend(kind_lines(s));
            continue;
        }
        let sel = &s["selection"];
        let mut line = format!("    selection: {}", py(&sel["verdict"]));
        if sel["verdict"] == "undecided" {
            let cons: Vec<String> = sel["consistent"].as_array().into_iter().flatten().map(py).collect();
            line += &format!(" - {}", cons.join(" or "));
            if let Some(n) = sel.get("candidates_for_choose") {
                line += &format!(" (a draw needs a corpus of at least {n})");
            }
        }
        for (k, why) in sel["refuted"].as_object().into_iter().flatten() {
            line += &format!("; {k} refuted: {}", py(why));
        }
        if let Some(st) = sel.get("settles").and_then(Value::as_str).filter(|s| !s.is_empty() && *s != "decided") {
            line += &format!("; settles: {st}");
        }
        out.push(line + &asserted_text(sel));
        let o = &s["order"];
        let mut line = format!("    order: {}", py(&o["verdict"]));
        if o["pairs"].as_i64().unwrap_or(0) != 0 {
            line += &format!(" (ascending {:.2} of {} pairs, step of one {:.2})", o["ascending"].as_f64().unwrap_or(0.0), o["pairs"], o["step_of_one"].as_f64().unwrap_or(0.0));
        }
        if let Some(n) = o.get("note").and_then(Value::as_str).filter(|n| !n.is_empty()) {
            line += &format!("; {n}");
        }
        out.push(line + &asserted_text(o));
        out.push(size_line(&s["size"]));
    }
    let pr = &doc["projection"];
    let cands = match pr["candidates"].as_i64() {
        Some(c) if c != 0 => format!(", {c} candidates"),
        _ => String::new(),
    };
    out.push(format!("open questions (projected over {} pass(es){cands}):", pr["epochs"]));
    if open.is_empty() {
        out.push("  none: the trace decides every verdict".into());
    }
    for it in &open {
        let cands: Vec<String> = it["candidates"].as_array().into_iter().flatten().map(py).collect();
        let mut line = format!(
            "  {}. site {} {}: {}; divergence {:.3}",
            it["rank"],
            site_name(&it["site"], it.get("phase").unwrap_or(&Value::Null)),
            py(&it["question"]),
            cands.join(" | "),
            it["divergence"].as_f64().unwrap_or(0.0)
        );
        if let Some(a) = it.get("answered") {
            line += &format!("; answered: {}", py(a));
        }
        out.push(line);
        out.push(format!("     ask: {}", py(&it["ask"])));
        out.push(format!("     settles: {}", py(&it["settles"])));
        let sc = &it["at_scale"];
        if sc.get("choose").is_some() {
            let f3 = |v: &Value| format!("{:.3}", v.as_f64().unwrap_or(0.0));
            out.push(format!(
                "     at {} pass(es): never read {} under choose, {} under consume; re-reads {} and {} of the touches",
                sc["epochs"],
                f3(&sc["choose"]["never_read_share"]),
                f3(&sc["consume"]["never_read_share"]),
                f3(&sc["choose"]["repeat_share"]),
                f3(&sc["consume"]["repeat_share"])
            ));
        } else if let Some(n) = sc.get("note") {
            out.push(format!("     at scale: {}", py(n)));
        }
    }
    let app = &doc["application"];
    if app.is_object() && !app.as_object().unwrap().is_empty() {
        let mut line = format!("application: {}", py(&app["display"]));
        match app.get("version").and_then(Value::as_str).filter(|v| !v.is_empty()) {
            Some(v) => line += &format!(" {v}"),
            None => line += " (version unknown)",
        }
        match app.get("record").filter(|r| !r.is_null() && r.as_str() != Some("")) {
            Some(r) => {
                line += &format!("; record {}", py(r));
                if app.get("exact") != Some(&json!(true)) {
                    line += ", suggestions only";
                }
            }
            None => line += "; no record",
        }
        out.push(line);
        for (k, v) in app.get("options").and_then(Value::as_object).into_iter().flatten() {
            out.push(format!("  {k}: {}", py(v)));
        }
        for b in app.get("blind_spots").and_then(Value::as_array).into_iter().flatten() {
            out.push(format!("  blind spot ({}): {}", py(&b["kind"]), py(&b["note"])));
        }
    }
    if !asserted.is_empty() {
        out.push("asserted:".into());
        for a in &asserted {
            let (kind, value) = if a.get("fact").is_some() { ("fact", &a["value"]) } else { ("question", &a["answer"]) };
            let mut line = format!("  site {} {}: {}", site_name(&a["site"], a.get("phase").unwrap_or(&Value::Null)), py(&a[kind]), py(value));
            if let Some(n) = a.get("note").and_then(Value::as_str).filter(|n| !n.is_empty()) {
                line += &format!(" ({n})");
            }
            out.push(line + &format!("; by {}, {}", py(&a["by"]), py(&a["at"])));
        }
    }
    if !rejected.is_empty() {
        out.push("rejected:".into());
        let get = |r: &Value, k: &str| py(r.get(k).unwrap_or(&Value::Null));
        for r in &rejected {
            let what = if r.get("fact").is_some() { format!("{} {}", get(r, "fact"), get(r, "value")) } else { format!("{} {}", get(r, "question"), get(r, "answer")) };
            out.push(format!("  {} {what}: {}", get(r, "template"), get(r, "why")));
        }
    }
    out.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_splits_into_its_template_and_fields() {
        let (k, f) = split("train/00012/sample_000000345.npz");
        assert_eq!(k, "train/\0/sample_\0.npz");
        assert_eq!(f, vec![(false, "00012"), (false, "000000345")]);
        let (k, f) = split("kv/0123456789abcdef0123/x7");
        assert_eq!(k, "kv/\0/x\0");
        assert_eq!(f, vec![(true, "0123456789abcdef0123"), (false, "7")]);
        // sixteen digits and no letter is a number; a short hex run is a literal around its digits
        assert_eq!(split("1234567890123456").1, vec![(false, "1234567890123456")]);
        assert_eq!(split("ab12cd").0, "ab\0cd");
        assert_eq!(split("plain").1, vec![]);
    }

    #[test]
    fn compress_takes_the_shortest_cycle_that_repeats() {
        let g = |s: &str| compress(&s.split(' ').collect::<Vec<_>>()).into_iter().map(|(g, r)| (g.join(" "), r)).collect::<Vec<_>>();
        assert_eq!(g("open read read read close"), vec![("open".into(), 1), ("read".into(), 3), ("close".into(), 1)]);
        assert_eq!(g("open lseek read lseek read close"), vec![("open".into(), 1), ("lseek read".into(), 2), ("close".into(), 1)]);
        assert_eq!(g("a b c d e a b c d e"), vec![("a".into(), 1), ("b".into(), 1), ("c".into(), 1), ("d".into(), 1), ("e".into(), 1), ("a".into(), 1), ("b".into(), 1), ("c".into(), 1), ("d".into(), 1), ("e".into(), 1)]);
        assert_eq!(compress::<u8>(&[]), vec![]);
    }

    #[test]
    fn an_opens_offsets_class() {
        let class = |data: &[(i64, i64)]| {
            let mut d = Data::default();
            for &(o, m) in data {
                d.add(o, m);
            }
            d.class()
        };
        assert_eq!(class(&[]), ("none", 0));
        assert_eq!(class(&[(0, 4), (4, 4)]), ("whole", 1));
        assert_eq!(class(&[(8, 4), (12, 4)]), ("sequential", 1));
        assert_eq!(class(&[(0, 4), (100, 4), (104, 4)]), ("positioned", 2));
    }

    #[test]
    fn the_birthday_bounds() {
        assert_eq!(p_no_repeat(5, 0), 0.0);
        assert!((p_no_repeat(2, 1) - (-1.0f64).exp()).abs() < 1e-12);
        for n in [1, 10, 100, 1000, 12345] {
            let k = touches_to_decide(n);
            assert!(p_no_repeat(k, n) <= P_REFUTE && p_no_repeat(k - 1, n) > P_REFUTE || k == 2, "{n}");
        }
        assert_eq!(candidates_for_choose(1), 1);
        let k = 30;
        let n = candidates_for_choose(k);
        assert!(p_no_repeat(k, n) >= P_REFUTE && p_no_repeat(k, n - 1) < P_REFUTE);
    }

    #[test]
    fn the_python_formats() {
        assert_eq!(fmt_g(0.0, 2), "0");
        assert_eq!(fmt_g(0.012345, 2), "0.012");
        assert_eq!(fmt_g(1.5e-7, 2), "1.5e-07");
        assert_eq!(fmt_g(0.5, 2), "0.5");
        assert_eq!(fmt_g(123.0, 2), "1.2e+02");
        assert_eq!(fmt_g(f64::INFINITY, 2), "inf");
        assert_eq!(fmt_g(f64::NEG_INFINITY, 2), "-inf");
        assert_eq!(fmt_g(f64::NAN, 2), "nan");
        assert_eq!(pct(0.5, 0), "50%");
        assert_eq!(py_float(0.05), "0.05");
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py(&json!(null)), "None");
        assert_eq!(py(&json!(true)), "True");
        assert_eq!(py(&json!(false)), "False");
        assert_eq!(py(&json!([1, "a"])), "[1, 'a']");
        assert_eq!(bytes_text(512), "512 B");
        assert_eq!(bytes_text(1536), "1.5 KiB");
        assert_eq!(bytes_text(3 << 20), "3.0 MiB");
        assert_eq!(bytes_text(5 << 30), "5.0 GiB");
        assert_eq!(ns_text(500.0), "500 ns");
        assert_eq!(ns_text(1500.0), "2 us");
        assert_eq!(ns_text(2.5e6), "2.5 ms");
        assert_eq!(ns_text(3e9), "3.00 s");
        assert_eq!(p50(&json!({"n": 0, "max": 7, "buckets": []})), 7);
        assert_eq!(field_name(27), "f27");
    }
}
