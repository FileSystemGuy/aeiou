//! `aeiou-trace export`: the runner's trace file (`DESIGN_REVIEW.md` §3.58, `runner/REFERENCE.md`
//! §13) from an strace or an object-store log, and the metrics of such a file. JSON Lines: a
//! header, then one op per line in the order the calls returned, each on a lane (a traced task)
//! and naming its open by id (`fd`) through every `dup` and inheritance the tracer resolved.
//!
//! Two passes over the input, so the state is per open and per path, never per call (§3.89):
//! the first learns what only the end of the trace says (which opens more than one lane used,
//! whose sequential reads and writes are then written positioned since the lanes' interleaving
//! decided their position; which opens an `O_APPEND` write pinned; the earliest start, from
//! which `t` counts; the paths the trace created; what each AIO request returned), the second
//! writes the lines. The header, which carries the counts, is written first.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Write};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::metrics::{Opts as MetricsOpts, TraceMetrics};
use crate::strace::{self, for_each_line, int0, normpath, Seen, Tracer, When, R, W};
use crate::trace::{Line, LineOp, Ret};

pub const TRACE_FORMAT: u64 = 1;

/// The open flags the schema knows (`OpenFlag`); anything else strace shows is dropped and counted.
fn open_flag(f: &str) -> Option<&'static str> {
    Some(match f {
        "O_RDONLY" => "RDONLY",
        "O_WRONLY" => "WRONLY",
        "O_RDWR" => "RDWR",
        "O_CREAT" => "CREAT",
        "O_TRUNC" => "TRUNC",
        "O_EXCL" => "EXCL",
        "O_APPEND" => "APPEND",
        "O_CLOEXEC" => "CLOEXEC",
        "O_DIRECTORY" => "DIRECTORY",
        "O_DIRECT" => "DIRECT",
        "O_SYNC" => "SYNC",
        "O_DSYNC" => "DSYNC",
        "O_NOATIME" => "NOATIME",
        "O_NOFOLLOW" => "NOFOLLOW",
        _ => return None,
    })
}

const IOCTLS: [&str; 3] = ["TCGETS", "FIONREAD", "BLKGETSIZE64"];

fn advice(a: &str) -> Option<&'static str> {
    Some(match a {
        "POSIX_FADV_NORMAL" => "NORMAL",
        "POSIX_FADV_RANDOM" => "RANDOM",
        "POSIX_FADV_SEQUENTIAL" => "SEQUENTIAL",
        "POSIX_FADV_WILLNEED" => "WILLNEED",
        "POSIX_FADV_DONTNEED" => "DONTNEED",
        "POSIX_FADV_NOREUSE" => "NOREUSE",
        _ => return None,
    })
}

/// A request of a `submit` line.
#[derive(Debug, Clone)]
pub struct Member {
    pub op: &'static str,
    pub fd: usize,
    pub offset: i64,
    pub len: i64,
    pub ret: Value,
}

/// One line of the trace file.
#[derive(Debug, Clone, Default)]
pub struct Event {
    pub lane: usize,
    /// Nanoseconds from the earliest start (set in the second pass).
    pub t: i64,
    pub dur: i64,
    pub op: &'static str,
    pub fd: Option<usize>,
    pub path: Option<String>,
    pub to: Option<String>,
    pub flags: Option<Vec<&'static str>>,
    pub mode: Option<u32>,
    pub offset: Option<i64>,
    pub len: Option<i64>,
    pub whence: Option<String>,
    pub advice: Option<&'static str>,
    pub request: Option<String>,
    /// A count other than 0, or an errno's name.
    pub ret: Value,
    pub ops: Option<Vec<Member>>,
    /// The position a sequential read or write was at (the exporter's own).
    pub at: Option<i64>,
}

fn esc(s: &str, out: &mut Vec<u8>) {
    serde_json::to_writer(&mut *out, s).expect("a string serializes");
}

impl Event {
    /// The line as written, without its newline.
    pub fn write(&self, out: &mut Vec<u8>) {
        let _ = write!(out, r#"{{"lane":{},"t":{},"dur":{},"op":"{}""#, self.lane, self.t, self.dur, self.op);
        if let Some(fd) = self.fd {
            let _ = write!(out, r#","fd":{fd}"#);
        }
        if let Some(p) = &self.path {
            out.extend_from_slice(br#","path":"#);
            esc(p, out);
        }
        if let Some(p) = &self.to {
            out.extend_from_slice(br#","to":"#);
            esc(p, out);
        }
        if let Some(f) = &self.flags {
            out.extend_from_slice(br#","flags":["#);
            for (i, x) in f.iter().enumerate() {
                let _ = write!(out, r#"{}"{x}""#, if i > 0 { "," } else { "" });
            }
            out.push(b']');
        }
        if let Some(m) = self.mode {
            let _ = write!(out, r#","mode":{m}"#);
        }
        if let Some(o) = self.offset {
            let _ = write!(out, r#","offset":{o}"#);
        }
        if let Some(l) = self.len {
            let _ = write!(out, r#","len":{l}"#);
        }
        if let Some(w) = &self.whence {
            out.extend_from_slice(br#","whence":"#);
            esc(w, out);
        }
        if let Some(a) = self.advice {
            let _ = write!(out, r#","advice":"{a}""#);
        }
        if let Some(r) = &self.request {
            out.extend_from_slice(br#","request":"#);
            esc(r, out);
        }
        if let Some(ops) = &self.ops {
            out.extend_from_slice(br#","ops":["#);
            for (i, m) in ops.iter().enumerate() {
                let _ = write!(out, r#"{}{{"op":"{}","fd":{},"offset":{},"len":{},"ret":{}}}"#, if i > 0 { "," } else { "" }, m.op, m.fd, m.offset, m.len, m.ret);
            }
            out.push(b']');
        }
        if !self.ret.is_null() {
            let _ = write!(out, r#","ret":{}"#, self.ret);
        }
        out.push(b'}');
    }
}

/// What the first pass learned of an open.
#[derive(Debug, Clone, Default)]
struct Open {
    first_lane: Option<usize>,
    multi: bool,
    pinned: bool,
    creat: bool,
    /// Sequential reads and writes on it (positioned in the file when `multi` or `pinned`).
    sequential: u64,
}

/// What the second pass needs from the first.
#[derive(Debug, Clone)]
pub struct Plan {
    t0: Option<f64>,
    positioned: Vec<bool>,
    rets: HashMap<(usize, usize), Value>,
}

/// The exporter of one pass.
pub struct Exporter {
    root: String,
    plan: Option<Plan>,
    lanes: HashMap<i64, usize>,
    opens: Vec<Open>,
    t0: Option<f64>,
    notes: BTreeMap<String, Value>,
    events: usize,
    seen: HashSet<String>,
    creates: Vec<String>,
    created: HashSet<String>,
    rets: HashMap<(usize, usize), Value>,
    /// The lines of the second pass, until the driver takes them.
    pub queue: Vec<Event>,
}

/// An errno name, or the count when it is not 0 (`_ret` of the Python).
fn ret_of(ret: Value, err: Option<&str>) -> Value {
    if let Some(e) = err {
        return json!(e);
    }
    if ret.as_i64() == Some(0) {
        Value::Null
    } else {
        ret
    }
}

fn round_ns(s: f64) -> i64 {
    (s * 1e9).round_ties_even() as i64
}

impl Exporter {
    /// The first pass's exporter for paths under `root` (normalized, no trailing slash).
    pub fn new(root: &str) -> Exporter {
        Exporter {
            root: format!("{}/", root.trim_end_matches('/')),
            plan: None,
            lanes: HashMap::new(),
            opens: Vec::new(),
            t0: None,
            notes: BTreeMap::new(),
            events: 0,
            seen: HashSet::new(),
            creates: Vec::new(),
            created: HashSet::new(),
            rets: HashMap::new(),
            queue: Vec::new(),
        }
    }

    /// The second pass's exporter.
    pub fn emitting(root: &str, plan: Plan) -> Exporter {
        Exporter { plan: Some(plan), ..Exporter::new(root) }
    }

    fn first_pass(&self) -> bool {
        self.plan.is_none()
    }

    pub fn note(&mut self, k: &str, n: u64) {
        if self.first_pass() {
            let e = self.notes.entry(k.to_string()).or_insert(json!(0));
            *e = json!(e.as_u64().unwrap_or(0) + n);
        }
    }

    pub fn pin(&mut self, oid: usize) {
        if self.first_pass() {
            self.opens[oid].pinned = true;
        }
    }

    /// The earliest start so far (the final one in the second pass).
    pub fn t0(&self) -> Option<f64> {
        match &self.plan {
            Some(p) => p.t0,
            None => self.t0,
        }
    }

    fn lane(&mut self, tid: i64) -> usize {
        let n = self.lanes.len();
        *self.lanes.entry(tid).or_insert(n)
    }

    /// The path below the root (the root itself is `.`): never absolute, never empty, no `..`,
    /// what the runner accepts (`trace.rs`, `check_path`).
    pub fn rel(&self, path: &str) -> Result<String> {
        if path == self.root || format!("{path}/") == self.root {
            return Ok(".".into());
        }
        match path.strip_prefix(self.root.as_str()) {
            Some(out) if !out.is_empty() && !out.starts_with('/') && !out.split('/').any(|c| c == "..") => Ok(out.to_string()),
            _ => bail!("aeiou-trace export: internal error: {path:?} is not a normalized path under the root {:?}", self.root),
        }
    }

    fn event(&mut self, when: When, mut e: Event) {
        let (tid, issued, done) = when;
        e.lane = self.lane(tid);
        match issued {
            None => {
                self.note("lines_without_timestamps", 1);
                e.dur = 0;
            }
            Some(iss) => {
                if self.first_pass() && self.t0.is_none_or(|t0| iss < t0) {
                    self.t0 = Some(iss);
                }
                e.dur = done.map_or(0, |d| round_ns(d - iss)).max(0);
            }
        }
        let index = self.events;
        self.events += 1;
        if let Some(fd) = e.fd {
            let lane = e.lane;
            let o = &mut self.opens[fd];
            match o.first_lane {
                None => o.first_lane = Some(lane),
                Some(l) if l != lane => o.multi = true,
                _ => {}
            }
        }
        match &self.plan {
            None => self.learn(&e),
            Some(plan) => {
                e.t = match (issued, plan.t0) {
                    (Some(i), Some(t0)) => round_ns(i - t0),
                    _ => 0,
                };
                if let (Some(at), None, Some(fd)) = (e.at, e.offset, e.fd) {
                    if plan.positioned[fd] {
                        e.offset = Some(at);
                    }
                }
                if let Some(ops) = &mut e.ops {
                    for (j, m) in ops.iter_mut().enumerate() {
                        if let Some(r) = plan.rets.get(&(index, j)) {
                            m.ret = r.clone();
                        }
                    }
                }
                self.queue.push(e);
            }
        }
    }

    /// The first pass's bookkeeping of a line: the creates, in the order the lines come.
    fn learn(&mut self, e: &Event) {
        let failed = !e.ret.is_null() && !e.ret.is_number();
        let ok = e.ret.is_null();
        match e.op {
            "open" => {
                let p = e.path.clone().unwrap_or_default();
                if self.opens[e.fd.unwrap()].creat && !self.seen.contains(&p) && !self.created.contains(&p) && ok {
                    self.created.insert(p.clone());
                    self.creates.push(p.clone());
                }
                self.seen.insert(p);
            }
            "mkdir" if ok => self.create(e.path.clone().unwrap_or_default()),
            "rename" if ok => self.create(e.to.clone().unwrap_or_default()),
            "stat" if ok => {
                self.seen.insert(e.path.clone().unwrap_or_default());
            }
            "read" | "write" if e.offset.is_none() && e.at.is_some() => self.opens[e.fd.unwrap()].sequential += 1,
            _ => {}
        }
        let _ = failed;
    }

    fn create(&mut self, p: String) {
        if self.created.insert(p.clone()) {
            self.creates.push(p);
        }
    }

    pub fn open(&mut self, when: When, path: &str, flags: &str, mode: Option<&str>, err: Option<&str>) -> Result<Option<usize>> {
        let mut names: Vec<&'static str> = Vec::new();
        let mut dropped = 0;
        for f in flags.split('|') {
            match open_flag(f) {
                Some(n) => names.push(n),
                None if !f.is_empty() && f != "O_LARGEFILE" => dropped += 1,
                None => {}
            }
        }
        if dropped > 0 {
            self.note("open_flags_dropped", dropped);
        }
        if !names.iter().any(|n| matches!(*n, "RDONLY" | "WRONLY" | "RDWR")) {
            names.insert(0, "RDONLY"); // O_RDONLY is 0 and strace prints it, but be sure
        }
        let rel = self.rel(path)?;
        if (names.contains(&"CREAT") || names.contains(&"TRUNC")) && rel == "." {
            // the runner refuses a trace that changes --root itself: the open is left out, and
            // the calls on its descriptor with it
            self.note("calls_not_exported", 1);
            self.note("not_exported:open:creates or truncates the root", 1);
            return Ok(None);
        }
        let oid = self.opens.len();
        self.opens.push(Open { creat: names.contains(&"CREAT") && err.is_none(), ..Default::default() });
        let mode = mode.filter(|m| !m.is_empty() && m.bytes().all(|c| c.is_ascii_digit())).and_then(|m| u32::from_str_radix(m, 8).ok());
        self.event(when, Event { op: "open", fd: Some(oid), path: Some(rel), flags: Some(names), mode, ret: ret_of(Value::Null, err), ..Default::default() });
        Ok(Some(oid))
    }

    /// An op on an open; `fields` its own (`offset`, `whence`, `len`, `advice`, `request`).
    pub fn op(&mut self, when: When, op: &'static str, oid: usize, err: Option<&str>, ret: Value, fields: Vec<(&str, Value)>) {
        let mut e = Event { op, fd: Some(oid), ret: ret_of(ret, err), ..Default::default() };
        for (k, v) in fields {
            match k {
                "offset" => e.offset = v.as_i64(),
                "len" => e.len = v.as_i64(),
                "whence" => e.whence = v.as_str().map(str::to_string),
                "advice" => e.advice = v.as_str().and_then(advice),
                "request" => e.request = v.as_str().map(str::to_string),
                _ => {}
            }
        }
        self.event(when, e);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn data(&mut self, when: When, kind: &'static str, oid: usize, offset: i64, length: i64, positioned: bool, ret: i64, err: Option<&str>) {
        let e = Event {
            op: kind,
            fd: Some(oid),
            len: Some(length),
            ret: ret_of(json!(ret.max(0)), err),
            offset: positioned.then_some(offset),
            at: Some(offset),
            ..Default::default()
        };
        self.event(when, e);
    }

    /// An `io_submit`'s requests as one line; the handles name each member for its result.
    pub fn submit(&mut self, when: When, members: &[(&'static str, usize, i64, i64)]) -> Vec<(usize, usize)> {
        let lane = self.lane(when.0);
        let mut ops = Vec::new();
        for (op, oid, off, nbytes) in members {
            let o = &mut self.opens[*oid];
            match o.first_lane {
                None => o.first_lane = Some(lane),
                Some(l) if l != lane => o.multi = true,
                _ => {}
            }
            // `ret` is filled when the result is reaped; a result the trace never shows is
            // taken as complete, as the metrics take it
            ops.push(Member { op, fd: *oid, offset: *off, len: *nbytes, ret: json!(nbytes) });
        }
        let index = self.events;
        let n = ops.len();
        self.event(when, Event { op: "submit", ops: Some(ops), ..Default::default() });
        (0..n).map(|j| (index, j)).collect()
    }

    /// A member's result, reaped after its line.
    pub fn member_ret(&mut self, member: (usize, usize), ret: Value) {
        if self.first_pass() {
            self.rets.insert(member, ret);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fd_op(&mut self, when: When, kind: &'static str, call: &str, oid: usize, a: &[String], ret: i64, err: Option<&str>) {
        let int = |i: usize| a.get(i).and_then(|x| int0(x));
        let mut fields: Vec<(&str, Value)> = Vec::new();
        let unparsed = |s: &mut Self| {
            s.note("calls_not_exported", 1);
            s.note(&format!("not_exported:{call}:unparsed"), 1);
        };
        match kind {
            "ftruncate" => match int(1) {
                Some(l) => fields.push(("len", json!(l))),
                None => return unparsed(self),
            },
            "fallocate" => match (int(2), int(3)) {
                (Some(o), Some(l)) => {
                    fields.push(("offset", json!(o)));
                    fields.push(("len", json!(l)));
                }
                _ => return unparsed(self),
            },
            "fadvise" => {
                let Some(name) = a.get(3) else { return unparsed(self) };
                if advice(name).is_none() {
                    self.note("calls_not_exported", 1);
                    self.note(&format!("not_exported:{call}:{name}"), 1);
                    return;
                }
                match (int(1), int(2)) {
                    (Some(o), Some(l)) => {
                        fields.push(("offset", json!(o)));
                        fields.push(("len", json!(l)));
                        fields.push(("advice", json!(name)));
                    }
                    _ => return unparsed(self),
                }
            }
            "ioctl" => {
                let req = a.get(1).map(|x| x.split(' ').next().unwrap_or("").to_string()).unwrap_or_default();
                if !IOCTLS.contains(&req.as_str()) {
                    self.note("calls_not_exported", 1);
                    self.note(&format!("not_exported:ioctl:{}", if req.is_empty() { "?" } else { &req }), 1);
                    return;
                }
                fields.push(("request", json!(req)));
            }
            _ => {}
        }
        // a listing's entries are not known from the trace: not checked
        let ret = if kind == "readdir" { Value::Null } else { json!(ret) };
        self.op(when, kind, oid, err, ret, fields);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn path_op(&mut self, when: When, kind: &'static str, call: &str, path: &str, a: &[String], pi: usize, to: Option<&str>, err: Option<&str>) -> Result<()> {
        if call == "truncate" {
            self.note("calls_not_exported", 1);
            self.note("not_exported:truncate", 1);
            return Ok(());
        }
        let rel = self.rel(path)?;
        let to_rel = to.map(|t| self.rel(t)).transpose()?;
        if kind != "stat" && (rel == "." || to_rel.as_deref() == Some(".")) {
            // `mkdir` of the root (an `os.makedirs(..., exist_ok=True)`), its `rmdir`: the runner
            // refuses a trace that changes --root itself, whatever the call returned
            self.note("calls_not_exported", 1);
            self.note(&format!("not_exported:{kind}:the root itself"), 1);
            return Ok(());
        }
        let mut e = Event { op: kind, path: Some(rel), ret: ret_of(Value::Null, err), ..Default::default() };
        if kind == "rename" {
            match to_rel {
                Some(t) => e.to = Some(t),
                None => {
                    self.note("calls_not_exported", 1);
                    self.note("not_exported:rename:destination outside the root", 1);
                    return Ok(());
                }
            }
        } else if kind == "mkdir" {
            if let Some(m) = a.get(pi + 1).filter(|m| !m.is_empty() && m.bytes().all(|c| c.is_ascii_digit())) {
                e.mode = u32::from_str_radix(m, 8).ok();
            }
        }
        self.event(when, e);
        Ok(())
    }

    /// A path op with no open, from an object log (a HEAD's stat, a DELETE's unlink).
    pub fn path_event(&mut self, when: When, op: &'static str, path: &str, err: Option<&str>) -> Result<()> {
        let rel = self.rel(path)?;
        self.event(when, Event { op, path: Some(rel), ret: ret_of(Value::Null, err), ..Default::default() });
        Ok(())
    }

    /// The end of the first pass: the plan of the second, and the header.
    pub fn finish(&self, root: &str, source: &str, source_notes: &BTreeMap<String, u64>) -> (Plan, Value) {
        let mut notes = self.notes.clone();
        let shared: u64 = self.opens.iter().filter(|o| o.multi).map(|o| o.sequential).sum();
        if shared > 0 {
            notes.insert("shared_positions_resolved".into(), json!(shared));
        }
        for (k, v) in source_notes {
            notes.entry(k.clone()).or_insert(json!(v));
        }
        let plan = Plan { t0: self.t0, positioned: self.opens.iter().map(|o| o.multi || o.pinned).collect(), rets: self.rets.clone() };
        let header = json!({
            "aeiou_trace": TRACE_FORMAT,
            "source": source,
            "root": root,
            "lanes": self.lanes.len(),
            "lines": self.events,
            "opens": self.opens.len(),
            "creates": self.creates,
            "notes": notes,
        });
        (plan, header)
    }
}

/// What an export wrote.
#[derive(Debug, Clone)]
pub struct Summary {
    pub header: Value,
    /// The lines after the header; nothing is written when there are none.
    pub lines: usize,
    pub sha256: String,
    pub seen: Seen,
}

/// The source of an input, opened once per pass.
pub type Opener<'a> = dyn Fn() -> std::io::Result<Box<dyn BufRead + 'a>> + 'a;

/// Write `header` and then the lines `pass` makes, hashing what is written.
pub fn write_file(out: &mut dyn Write, header: &Value, pass: impl FnOnce(&mut dyn FnMut(Vec<Event>) -> Result<()>) -> Result<()>) -> Result<String> {
    let mut h = Sha256::new();
    let mut buf = serde_json::to_vec(header)?;
    buf.push(b'\n');
    h.update(&buf);
    out.write_all(&buf)?;
    let mut write = |events: Vec<Event>| -> Result<()> {
        buf.clear();
        for e in &events {
            e.write(&mut buf);
            buf.push(b'\n');
        }
        h.update(&buf);
        out.write_all(&buf)?;
        Ok(())
    };
    pass(&mut write)?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Export an strace: the calls on paths under `root`, written relative to it.
pub fn strace(open: &Opener, root: &str, exclude: &[String], cwd: Option<&str>, out: &mut dyn Write) -> Result<Summary> {
    let o = strace::Opts { roots: vec![root.to_string()], exclude: exclude.to_vec(), cwd: cwd.map(str::to_string), ..Default::default() };
    let root = normpath(root);
    let mut t = Tracer::new(&o, false, Some(Exporter::new(&root)));
    t.feed(open()?)?;
    t.finish()?;
    let (plan, header) = t.export.as_ref().unwrap().finish(&root, "strace", &t.notes);
    let seen = t.seen;
    let lines = header["lines"].as_u64().unwrap_or(0) as usize;
    if lines == 0 {
        return Ok(Summary { header, lines, sha256: String::new(), seen });
    }
    let sha256 = write_file(out, &header, |write| {
        let mut t = Tracer::new(&o, false, Some(Exporter::emitting(&root, plan)));
        for_each_line(open()?, |line| {
            t.seen.lines += 1;
            t.line(line);
            let q = std::mem::take(&mut t.export.as_mut().unwrap().queue);
            if q.is_empty() { Ok(()) } else { write(q) }
        })?;
        t.finish()?;
        let q = std::mem::take(&mut t.export.as_mut().unwrap().queue);
        write(q)
    })?;
    Ok(Summary { header, lines, sha256, seen })
}

// ---------------------------------------------------------------- the metrics of a trace file

/// The metrics of an exported trace file's lines, with the strace's definitions: a lane is a
/// context, the whole file one instance, positions tracked per open id as the runner tracks
/// them. What `aeiou dry-run --metrics` computes for a `trace` node must equal this.
pub struct FileMetrics {
    pub m: TraceMetrics,
    pos: HashMap<usize, i64>,
    paths: HashMap<usize, String>,
    lanes: HashSet<usize>,
}

impl FileMetrics {
    pub fn new(block: Option<u64>) -> FileMetrics {
        FileMetrics { m: TraceMetrics::new(MetricsOpts { block }), pos: HashMap::new(), paths: HashMap::new(), lanes: HashSet::new() }
    }

    fn path(&self, fd: usize) -> Result<String> {
        self.paths.get(&fd).cloned().ok_or_else(|| anyhow!("an op on open {fd}, which no line opened"))
    }

    fn data(&mut self, lane: usize, kind: usize, fd: usize, offset: Option<i64>, len: i64, ret: &Value) -> Result<()> {
        let n = ret.as_i64().unwrap_or(0);
        let off = match offset {
            Some(o) => o,
            None => {
                let p = self.pos.entry(fd).or_insert(0);
                let o = *p;
                *p = o + n;
                o
            }
        };
        let path = self.path(fd)?;
        self.m.data(kind, 0, lane as i64, &path, off, len, n)
    }

    /// One line, as the runner reads it (`trace::Line`).
    pub fn line(&mut self, l: &Line) -> Result<()> {
        let lane = l.lane;
        self.lanes.insert(lane);
        let ret = |r: &Ret| match r {
            Ret::Int(n) => json!(n),
            Ret::Err(_) => Value::Null,
        };
        match &l.op {
            LineOp::Open { fd, path, .. } => {
                self.paths.insert(*fd, path.clone());
                self.pos.insert(*fd, 0);
                self.m.op("open");
            }
            LineOp::Read { fd, offset, len, ret: r } => self.data(lane, R, *fd, *offset, *len, &ret(r))?,
            LineOp::Write { fd, offset, len, ret: r } => self.data(lane, W, *fd, *offset, *len, &ret(r))?,
            LineOp::Submit { ops } => {
                *self.m.m.fan_out.entry(ops.len() as u64).or_insert(0) += 1;
                for sub in ops {
                    match sub {
                        LineOp::Read { fd, offset, len, ret: r } => self.data(lane, R, *fd, *offset, *len, &ret(r))?,
                        LineOp::Write { fd, offset, len, ret: r } => self.data(lane, W, *fd, *offset, *len, &ret(r))?,
                        _ => bail!("a `submit` member that is not a read or a write"),
                    }
                }
            }
            LineOp::Close { fd, .. } => {
                self.m.op("close");
                let path = self.path(*fd)?;
                self.m.close(lane as i64, &path);
            }
            LineOp::Lseek { fd, ret: r, .. } => {
                self.m.op("lseek");
                if let Ret::Int(n) = r {
                    self.pos.insert(*fd, *n);
                }
            }
            LineOp::Fstat { .. } => self.m.op("fstat"),
            LineOp::Fsync { .. } => self.m.op("fsync"),
            LineOp::Fdatasync { .. } => self.m.op("fdatasync"),
            LineOp::Readdir { .. } => self.m.op("readdir"),
            LineOp::Ioctl { .. } => self.m.op("ioctl"),
            LineOp::Fadvise { .. } => self.m.op("fadvise"),
            LineOp::Ftruncate { .. } => self.m.op("ftruncate"),
            LineOp::Fallocate { .. } => self.m.op("fallocate"),
            LineOp::Stat { .. } => self.m.op("stat"),
            LineOp::Unlink { .. } => self.m.op("unlink"),
            LineOp::Mkdir { .. } => self.m.op("mkdir"),
            LineOp::Rmdir { .. } => self.m.op("rmdir"),
            LineOp::Rename { .. } => self.m.op("rename"),
        }
        Ok(())
    }

    /// The `aeiou_metrics: 2` document, the file's header giving its source, root, and notes.
    pub fn document(mut self, header: &Value) -> Result<Value> {
        self.m.finish()?;
        Ok(json!({
            "aeiou_metrics": strace::METRICS_FORMAT,
            "source": header.get("source").and_then(Value::as_str).unwrap_or("trace"),
            "roots": [header.get("root").and_then(Value::as_str).unwrap_or("")],
            "block": self.m.m.opts_block,
            "order": "completion",
            "instances": 1,
            "contexts": self.lanes.len(),
            "chain_gap_us": null,
            "notes": header.get("notes").cloned().unwrap_or(json!({})),
            "total": self.m.total(false),
        }))
    }
}

/// The metrics of an exported trace file, read as the runner reads it: a line that is not
/// JSON is refused with its number.
pub fn metrics_of_file(input: impl BufRead, block: Option<u64>) -> Result<Value> {
    let mut header: Option<Value> = None;
    let mut fm = FileMetrics::new(block);
    let mut no = 0;
    for_each_line(input, |line| {
        no += 1;
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        let doc: Value = serde_json::from_str(line).map_err(|e| anyhow!("line {no}: {e}"))?;
        if !doc.is_object() {
            bail!("line {no}: not a JSON object");
        }
        if header.is_none() {
            if doc.get("aeiou_trace").and_then(Value::as_u64) != Some(TRACE_FORMAT) {
                bail!("not an aeiou_trace: {TRACE_FORMAT} file");
            }
            header = Some(doc);
            return Ok(());
        }
        // read as the runner reads the file (`trace::TraceFile::load`), from the line's own
        // text, so a repeated key or a lone surrogate is refused as the runner refuses it
        let l: Line = serde_json::from_str(line).map_err(|e| anyhow!("line {no}: {e}"))?;
        fm.line(&l).with_context(|| format!("line {no}"))
    })?;
    // the source is the strace's: a file of it is a `trace`
    let mut h = header.unwrap_or(json!({}));
    h["source"] = json!("trace");
    fm.document(&h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_not_under_the_root_is_an_internal_error() {
        let ex = Exporter::new("/d");
        assert_eq!(ex.rel("/d").unwrap(), ".");
        assert_eq!(ex.rel("/d/a/b").unwrap(), "a/b");
        for bad in ["/e/x", "/d//x", "/d/../x", "/dx", "x"] {
            assert!(ex.rel(bad).unwrap_err().to_string().contains("internal error"), "{bad}");
        }
        assert_eq!(Exporter::new("/").rel("/a").unwrap(), "a");
    }

    #[test]
    fn a_line_writes_its_fields_in_order() {
        let e = Event {
            lane: 1,
            t: 2,
            dur: 3,
            op: "submit",
            ops: Some(vec![Member { op: "read", fd: 0, offset: 0, len: 4, ret: json!("EIO") }]),
            ..Default::default()
        };
        let mut b = Vec::new();
        e.write(&mut b);
        assert_eq!(String::from_utf8(b).unwrap(), r#"{"lane":1,"t":2,"dur":3,"op":"submit","ops":[{"op":"read","fd":0,"offset":0,"len":4,"ret":"EIO"}]}"#);
        let e = Event { op: "lseek", fd: Some(0), whence: Some("SET".into()), offset: Some(5), ret: json!(5), ..Default::default() };
        let mut b = Vec::new();
        e.write(&mut b);
        assert_eq!(String::from_utf8(b).unwrap(), r#"{"lane":0,"t":0,"dur":0,"op":"lseek","fd":0,"offset":5,"whence":"SET","ret":5}"#);
    }
}
