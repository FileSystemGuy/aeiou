//! The `trace` node (`DESIGN_REVIEW.md` §3.58): a captured trace of a real application,
//! executed literally. The file is what `aeiou-trace export` writes (`builder/aeiou/trace.py`):
//! JSON Lines, a header, then one op per line in the order the calls returned. A **lane** is
//! a traced task (a thread or a process); an op names its open by id (`fd`), so lanes share
//! descriptors the way the application's tasks did; a sequential read or write on an open
//! that several lanes used was written positioned by the exporter, so a position here is
//! lane-local. `t` and `dur` are nanoseconds from the first exported call and the call's
//! duration: the **gap** before a line is `t − (t_prev + dur_prev)` of its lane (its `t` for
//! the lane's first line) and is issued as `compute`, scaled by `--time-scale`.
//!
//! Two orders cross lanes, both from the trace: an op waits for the open it names, and the
//! last close of an open waits for its uses (the **open table**, `run.rs`); and on a path
//! the trace changes, a changing op (a creating or truncating open, a write, a truncate, an
//! allocate, a sync or close of a writable open, a rename, an unlink, a mkdir, a rmdir)
//! waits for every earlier op on that path and a reading op (an open to read, a read, a
//! stat, a listing) waits for every earlier changing op on it: **path order**, computed
//! here as a dependency per op (`Dep`) and kept by `run.rs`. Reads never wait for reads, so
//! the concurrency of readers is the trace's. Every wait is for something earlier in the
//! trace, so the lanes cannot deadlock.
//!
//! This module holds the file, builds the VM's `Op` for a line (one place, so the dry run,
//! the metrics walk, and the run agree), and does the checks before the gate. The walkers
//! are in `dryrun.rs` (line order), `metrics.rs` (line order, a lane as a context), and
//! `run.rs` (one thread per lane, an open table with the waits).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::ast::{Advice, Ast, IoctlRequest, Node, OpenFlag, Whence};
use crate::vm::{flag_bits, Op, OpCtx, OpKind};

pub const FORMAT: u64 = 1;

/// The traced result: a count (0 when absent) or an errno name, which becomes the op's `expect`.
#[derive(Debug, Clone)]
pub enum Ret {
    Int(i64),
    Err(Vec<String>),
}

impl Default for Ret {
    fn default() -> Self {
        Ret::Int(0)
    }
}

impl<'de> Deserialize<'de> for Ret {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Int(i64),
            Err(String),
        }
        Ok(match Raw::deserialize(d)? {
            Raw::Int(n) => Ret::Int(n),
            Raw::Err(s) => Ret::Err(vec![s]),
        })
    }
}

impl Ret {
    pub fn count(&self) -> i64 {
        match self {
            Ret::Int(n) => *n,
            Ret::Err(_) => 0,
        }
    }
    pub fn expect(&self) -> &[String] {
        match self {
            Ret::Int(_) => &[],
            Ret::Err(v) => v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase", deny_unknown_fields)]
pub enum LineOp {
    Open { fd: usize, path: String, flags: Vec<OpenFlag>, #[serde(default)] mode: Option<u32>, #[serde(default)] ret: Ret },
    Close { fd: usize, #[serde(default)] ret: Ret },
    Read { fd: usize, #[serde(default)] offset: Option<i64>, len: i64, #[serde(default)] ret: Ret },
    Write { fd: usize, #[serde(default)] offset: Option<i64>, len: i64, #[serde(default)] ret: Ret },
    Lseek { fd: usize, offset: i64, whence: Whence, #[serde(default)] ret: Ret },
    Fstat { fd: usize, #[serde(default)] ret: Ret },
    Fsync { fd: usize, #[serde(default)] ret: Ret },
    Fdatasync { fd: usize, #[serde(default)] ret: Ret },
    Readdir { fd: usize, #[serde(default)] ret: Ret },
    Ioctl { fd: usize, request: IoctlRequest, #[serde(default)] ret: Ret },
    Fadvise { fd: usize, #[serde(default)] offset: i64, #[serde(default)] len: i64, advice: Advice, #[serde(default)] ret: Ret },
    Ftruncate { fd: usize, len: i64, #[serde(default)] ret: Ret },
    Fallocate { fd: usize, #[serde(default)] offset: i64, len: i64, #[serde(default)] ret: Ret },
    Stat { path: String, #[serde(default)] ret: Ret },
    Unlink { path: String, #[serde(default)] ret: Ret },
    Mkdir { path: String, #[serde(default)] mode: Option<u32>, #[serde(default)] ret: Ret },
    Rmdir { path: String, #[serde(default)] ret: Ret },
    Rename { path: String, to: String, #[serde(default)] ret: Ret },
    /// An `io_submit`: positioned reads and writes issued together and reaped before the
    /// lane's next line.
    Submit { ops: Vec<LineOp> },
}

impl LineOp {
    /// The open id the op works on (`None` for a path op, an open, or a group).
    pub fn fd(&self) -> Option<usize> {
        match self {
            LineOp::Close { fd, .. } | LineOp::Read { fd, .. } | LineOp::Write { fd, .. } | LineOp::Lseek { fd, .. } | LineOp::Fstat { fd, .. } | LineOp::Fsync { fd, .. } | LineOp::Fdatasync { fd, .. } | LineOp::Readdir { fd, .. } | LineOp::Ioctl { fd, .. } | LineOp::Fadvise { fd, .. } | LineOp::Ftruncate { fd, .. } | LineOp::Fallocate { fd, .. } => Some(*fd),
            _ => None,
        }
    }
    pub fn ret(&self) -> &Ret {
        static OK: Ret = Ret::Int(0);
        match self {
            LineOp::Open { ret, .. } | LineOp::Close { ret, .. } | LineOp::Read { ret, .. } | LineOp::Write { ret, .. } | LineOp::Lseek { ret, .. } | LineOp::Fstat { ret, .. } | LineOp::Fsync { ret, .. } | LineOp::Fdatasync { ret, .. } | LineOp::Readdir { ret, .. } | LineOp::Ioctl { ret, .. } | LineOp::Fadvise { ret, .. } | LineOp::Ftruncate { ret, .. } | LineOp::Fallocate { ret, .. } | LineOp::Stat { ret, .. } | LineOp::Unlink { ret, .. } | LineOp::Mkdir { ret, .. } | LineOp::Rmdir { ret, .. } | LineOp::Rename { ret, .. } => ret,
            LineOp::Submit { .. } => &OK,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Line {
    pub lane: usize,
    #[serde(default)]
    pub t: i64,
    #[serde(default)]
    pub dur: i64,
    #[serde(flatten)]
    pub op: LineOp,
    /// The ordinal of this line's first op among all ops of the file (members counted).
    #[serde(skip)]
    pub g0: usize,
}

/// Path order for one op (`deps`, by op ordinal): `path` is the id of a path the trace
/// changes; `k` the op's index among that path's ops, `mk` the changing ops before it.
#[derive(Debug, Clone, Copy)]
pub struct Dep {
    pub path: usize,
    pub k: usize,
    pub mk: usize,
    pub mutating: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Header {
    pub aeiou_trace: u64,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub root: String,
    #[serde(default)]
    pub lanes: usize,
    #[serde(default)]
    pub lines: usize,
    #[serde(default)]
    pub opens: usize,
    #[serde(default)]
    pub creates: Vec<String>,
    #[serde(default)]
    pub notes: serde_json::Map<String, serde_json::Value>,
}

/// What the file says about one open id.
#[derive(Debug, Clone, Default)]
pub struct OpenInfo {
    pub path: String,
    /// Ops on the id other than the open and its closes (what a close waits for).
    pub uses: usize,
    /// Close lines on the id (one per descriptor the application held on it).
    pub closes: usize,
    /// Set when the open itself failed in the trace (no fd ever exists).
    pub failed: bool,
}

/// A loaded trace file.
#[derive(Debug)]
pub struct TraceFile {
    /// The node's `file`, as written in the abstract.
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
    pub header: Header,
    pub lines: Vec<Line>,
    /// Line indices per lane, in order.
    pub lanes: Vec<Vec<usize>>,
    pub opens: Vec<OpenInfo>,
    /// Per path read without being created by the trace: the end of the farthest read, so
    /// the file under `--root` must be at least that long.
    pub read_extent: HashMap<String, i64>,
    /// Paths the trace names without creating them (opens, stats, unlinks): must exist.
    pub inputs: Vec<String>,
    /// The most descriptors live at once, walking the file in line order.
    pub peak_open: u64,
    /// Ops in the file, members of a group counted each.
    pub ops: u64,
    /// Path order per op ordinal: `None` for an op on a path the trace never changes.
    pub deps: Vec<Option<Dep>>,
    /// A rename's second dependency (its destination), by op ordinal.
    pub deps2: HashMap<usize, Dep>,
    /// Every path the file names (`Dep::path` indexes it).
    pub changed: Vec<String>,
}

impl TraceFile {
    /// Load `path` and check its sha256 against the node's.
    pub fn load(name: &str, path: &Path, sha256: &str) -> Result<TraceFile> {
        let bytes = std::fs::read(path).map_err(|e| anyhow!("trace `{name}`: {}: {e}", path.display()))?;
        let got = {
            let d = Sha256::digest(&bytes);
            let mut s = String::with_capacity(64);
            for b in d {
                s.push_str(&format!("{:02x}", b));
            }
            s
        };
        if got != sha256 {
            bail!("trace `{name}`: {} has sha256 {got}, the abstract names {sha256}", path.display());
        }
        let text = std::str::from_utf8(&bytes).map_err(|e| anyhow!("trace `{name}`: not UTF-8: {e}"))?;
        let mut it = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
        let (_, first) = it.next().ok_or_else(|| anyhow!("trace `{name}`: empty file"))?;
        let header: Header = serde_json::from_str(first).map_err(|e| anyhow!("trace `{name}`: header: {e}"))?;
        if header.aeiou_trace != FORMAT {
            bail!("trace `{name}`: aeiou_trace {} is not {FORMAT}", header.aeiou_trace);
        }
        let mut lines = Vec::with_capacity(header.lines);
        for (i, l) in it {
            let line: Line = serde_json::from_str(l).map_err(|e| anyhow!("trace `{name}`: line {}: {e}", i + 1))?;
            lines.push(line);
        }
        let mut tf = TraceFile { name: name.to_string(), path: path.to_path_buf(), sha256: got, header, lines, lanes: Vec::new(), opens: Vec::new(), read_extent: HashMap::new(), inputs: Vec::new(), peak_open: 0, ops: 0, deps: Vec::new(), deps2: HashMap::new(), changed: Vec::new() };
        tf.index()?;
        tf.path_order();
        Ok(tf)
    }

    /// Lanes, open ids, uses and closes, the read extents, and the peak of open descriptors.
    fn index(&mut self) -> Result<()> {
        let name = self.name.clone();
        let mut opens: Vec<OpenInfo> = Vec::new();
        let mut lanes: Vec<Vec<usize>> = Vec::new();
        let mut pos: Vec<i64> = Vec::new();
        let mut created: HashSet<String> = HashSet::new();
        let mut inputs: Vec<String> = Vec::new();
        let mut seen_input: HashSet<String> = HashSet::new();
        let mut extent: HashMap<String, i64> = HashMap::new();
        let (mut live, mut peak, mut ops) = (0u64, 0u64, 0u64);
        let input = |p: &str, created: &HashSet<String>, inputs: &mut Vec<String>, seen: &mut HashSet<String>| {
            if !created.contains(p) && seen.insert(p.to_string()) {
                inputs.push(p.to_string());
            }
        };
        for (i, line) in self.lines.iter_mut().enumerate() {
            while lanes.len() <= line.lane {
                lanes.push(Vec::new());
            }
            lanes[line.lane].push(i);
            line.g0 = ops as usize;
            let members: Vec<&LineOp> = match &line.op {
                LineOp::Submit { ops } => ops.iter().collect(),
                op => vec![op],
            };
            for op in members {
                ops += 1;
                if let LineOp::Open { fd, path, flags, ret, .. } = op {
                    if *fd != opens.len() {
                        bail!("trace `{name}`: line {}: open id {fd} out of order (expected {})", i + 2, opens.len());
                    }
                    let failed = matches!(ret, Ret::Err(_));
                    opens.push(OpenInfo { path: path.clone(), uses: 0, closes: 0, failed });
                    pos.push(0);
                    if !failed {
                        live += 1;
                        peak = peak.max(live);
                        if flags.contains(&OpenFlag::CREAT) {
                            created.insert(path.clone());
                        } else {
                            input(path, &created, &mut inputs, &mut seen_input);
                        }
                    }
                    continue;
                }
                if let Some(fd) = op.fd() {
                    let info = opens.get_mut(fd).ok_or_else(|| anyhow!("trace `{name}`: line {}: open id {fd} before its open", i + 2))?;
                    match op {
                        LineOp::Close { .. } => {
                            info.closes += 1;
                            live = live.saturating_sub(1);
                        }
                        _ => info.uses += 1,
                    }
                    let path = info.path.clone();
                    match op {
                        LineOp::Read { offset, len, ret, .. } => {
                            let off = offset.unwrap_or(pos[fd]);
                            let n = ret.count().max(0);
                            if offset.is_none() {
                                pos[fd] = off + n;
                            }
                            if !created.contains(&path) {
                                let e = extent.entry(path).or_insert(0);
                                *e = (*e).max(off + n);
                            }
                            let _ = len;
                        }
                        LineOp::Write { offset, len, .. } => {
                            if offset.is_none() {
                                pos[fd] += *len;
                            }
                        }
                        LineOp::Lseek { ret, .. } => pos[fd] = ret.count().max(0),
                        _ => {}
                    }
                    continue;
                }
                match op {
                    LineOp::Stat { path, ret } | LineOp::Unlink { path, ret } | LineOp::Rmdir { path, ret } => {
                        if !matches!(ret, Ret::Err(_)) {
                            input(path, &created, &mut inputs, &mut seen_input);
                        }
                    }
                    LineOp::Mkdir { path, ret, .. } => {
                        if !matches!(ret, Ret::Err(_)) {
                            created.insert(path.clone());
                        }
                    }
                    LineOp::Rename { path, to, ret } => {
                        if !matches!(ret, Ret::Err(_)) {
                            input(path, &created, &mut inputs, &mut seen_input);
                            created.insert(to.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        if lanes.len() != self.header.lanes && self.header.lanes != 0 {
            bail!("trace `{name}`: {} lanes in the file, the header says {}", lanes.len(), self.header.lanes);
        }
        self.lanes = lanes;
        self.opens = opens;
        self.read_extent = extent;
        self.inputs = inputs;
        self.peak_open = peak;
        self.ops = ops;
        Ok(())
    }

    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Path order (module doc): which ops change which paths, and what each op waits for.
    fn path_order(&mut self) {
        let mut ids: HashMap<String, usize> = HashMap::new();
        let mut paths: Vec<String> = Vec::new();
        let mut intern = |p: &str| -> usize {
            if let Some(i) = ids.get(p) {
                return *i;
            }
            paths.push(p.to_string());
            ids.insert(p.to_string(), paths.len() - 1);
            paths.len() - 1
        };
        let mut writable = vec![false; self.opens.len()];
        // per op: the paths it touches and whether it changes each (a rename touches two)
        let mut touched: Vec<[Option<(usize, bool)>; 2]> = Vec::with_capacity(self.ops as usize);
        for line in &self.lines {
            let members: Vec<&LineOp> = match &line.op {
                LineOp::Submit { ops } => ops.iter().collect(),
                op => vec![op],
            };
            for op in members {
                let one = |p: &str, m: bool, intern: &mut dyn FnMut(&str) -> usize| [Some((intern(p), m)), None];
                let t = match op {
                    LineOp::Open { fd, path, flags, .. } => {
                        writable[*fd] = flags.iter().any(|f| matches!(f, OpenFlag::WRONLY | OpenFlag::RDWR));
                        one(path, flags.iter().any(|f| matches!(f, OpenFlag::CREAT | OpenFlag::TRUNC)), &mut intern)
                    }
                    LineOp::Write { fd, .. } | LineOp::Ftruncate { fd, .. } | LineOp::Fallocate { fd, .. } => one(&self.opens[*fd].path, true, &mut intern),
                    LineOp::Close { fd, .. } | LineOp::Fsync { fd, .. } | LineOp::Fdatasync { fd, .. } => one(&self.opens[*fd].path, writable[*fd], &mut intern),
                    LineOp::Read { fd, .. } | LineOp::Lseek { fd, .. } | LineOp::Fstat { fd, .. } | LineOp::Readdir { fd, .. } | LineOp::Ioctl { fd, .. } | LineOp::Fadvise { fd, .. } => one(&self.opens[*fd].path, false, &mut intern),
                    LineOp::Stat { path, .. } => one(path, false, &mut intern),
                    LineOp::Unlink { path, .. } | LineOp::Mkdir { path, .. } | LineOp::Rmdir { path, .. } => one(path, true, &mut intern),
                    LineOp::Rename { path, to, .. } => [Some((intern(path), true)), Some((intern(to), true))],
                    LineOp::Submit { .. } => [None, None],
                };
                touched.push(t);
            }
        }
        let mut is_changed = vec![false; paths.len()];
        for t in &touched {
            for (p, m) in t.iter().flatten() {
                if *m {
                    is_changed[*p] = true;
                }
            }
        }
        let mut count = vec![0usize; paths.len()];
        let mut mcount = vec![0usize; paths.len()];
        let mut deps = Vec::with_capacity(touched.len());
        let mut second = HashMap::new();
        for (g, t) in touched.iter().enumerate() {
            let mut d = [None, None];
            for (slot, (p, m)) in t.iter().flatten().enumerate() {
                if is_changed[*p] {
                    d[slot] = Some(Dep { path: *p, k: count[*p], mk: mcount[*p], mutating: *m });
                }
                count[*p] += 1;
                if *m {
                    mcount[*p] += 1;
                }
            }
            deps.push(d[0]);
            if let Some(x) = d[1] {
                second.insert(g, x);
            }
        }
        self.deps = deps;
        self.deps2 = second;
        self.changed = paths;
    }

    /// The VM's `Op` for one line op. `cur` gives an open id's current position for a
    /// sequential read or write; `pos_after` says what to set it to afterwards.
    pub fn build<'c>(&'c self, op: &'c LineOp, cur: impl Fn(usize) -> i64) -> Built<'c> {
        let path = |fd: &usize| self.opens[*fd].path.as_str();
        let mut pos_after = None;
        let mut oid = op.fd();
        let built = match op {
            LineOp::Open { fd, path: p, flags, mode, ret } => {
                oid = Some(*fd);
                let aux = flag_bits(flags) | ((mode.unwrap_or(0) as u64) << 32);
                Op { kind: OpKind::Open, path: p, path2: None, offset: 0, len: 0, bytes: 0, aux, expect: ret.expect(), positioned: false, seed: 0 }
            }
            LineOp::Close { fd, ret } => Op { kind: OpKind::Close, path: path(fd), path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Read { fd, offset, len, ret } => {
                let off = offset.unwrap_or_else(|| cur(*fd));
                let bytes = ret.count().max(0);
                if offset.is_none() {
                    pos_after = Some(off + bytes);
                }
                Op { kind: OpKind::Read, path: path(fd), path2: None, offset: off, len: *len, bytes, aux: 0, expect: ret.expect(), positioned: offset.is_some(), seed: 0 }
            }
            LineOp::Write { fd, offset, len, ret } => {
                let off = offset.unwrap_or_else(|| cur(*fd));
                if offset.is_none() {
                    pos_after = Some(off + *len);
                }
                Op { kind: OpKind::Write, path: path(fd), path2: None, offset: off, len: *len, bytes: *len, aux: 0, expect: ret.expect(), positioned: offset.is_some(), seed: 0 }
            }
            LineOp::Lseek { fd, offset, whence, ret } => {
                pos_after = Some(ret.count().max(0));
                Op { kind: OpKind::Lseek, path: path(fd), path2: None, offset: *offset, len: 0, bytes: 0, aux: *whence as u64, expect: ret.expect(), positioned: false, seed: 0 }
            }
            LineOp::Fstat { fd, ret } => Op { kind: OpKind::Fstat, path: path(fd), path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Fsync { fd, ret } => Op { kind: OpKind::Fsync, path: path(fd), path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Fdatasync { fd, ret } => Op { kind: OpKind::Fdatasync, path: path(fd), path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Readdir { fd, ret } => Op { kind: OpKind::Readdir, path: path(fd), path2: None, offset: 0, len: 0, bytes: -1, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Ioctl { fd, request, ret } => Op { kind: OpKind::Ioctl, path: path(fd), path2: None, offset: 0, len: 0, bytes: 0, aux: *request as u64, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Fadvise { fd, offset, len, advice, ret } => Op { kind: OpKind::Fadvise, path: path(fd), path2: None, offset: *offset, len: *len, bytes: 0, aux: *advice as u64, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Ftruncate { fd, len, ret } => Op { kind: OpKind::Ftruncate, path: path(fd), path2: None, offset: 0, len: *len, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Fallocate { fd, offset, len, ret } => Op { kind: OpKind::Fallocate, path: path(fd), path2: None, offset: *offset, len: *len, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Stat { path: p, ret } => Op { kind: OpKind::Stat, path: p, path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Unlink { path: p, ret } => Op { kind: OpKind::Unlink, path: p, path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Mkdir { path: p, mode, ret } => Op { kind: OpKind::Mkdir, path: p, path2: None, offset: 0, len: 0, bytes: 0, aux: mode.unwrap_or(0o777) as u64, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Rmdir { path: p, ret } => Op { kind: OpKind::Rmdir, path: p, path2: None, offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Rename { path: p, to, ret } => Op { kind: OpKind::Rename, path: p, path2: Some(to), offset: 0, len: 0, bytes: 0, aux: 0, expect: ret.expect(), positioned: false, seed: 0 },
            LineOp::Submit { .. } => unreachable!("build() on a group; build its members"),
        };
        Built { op: built, oid, pos_after }
    }
}

pub struct Built<'c> {
    pub op: Op<'c>,
    pub oid: Option<usize>,
    pub pos_after: Option<i64>,
}


/// An `OpCtx` that owns its parts: the enclosing position of a `trace` node, from which each
/// op's context is made by appending the lane and the line's ordinal in it.
#[derive(Debug, Clone)]
pub struct OwnedCtx {
    pub template: String,
    pub actor: i64,
    pub indices: Vec<i64>,
    pub phase: Option<String>,
}

impl OwnedCtx {
    pub fn of(ctx: &OpCtx) -> Self {
        OwnedCtx { template: ctx.template.to_string(), actor: ctx.actor, indices: ctx.indices.to_vec(), phase: ctx.phase.map(|p| p.to_string()) }
    }
    pub fn ctx<'c>(&'c self, indices: &'c [i64]) -> OpCtx<'c> {
        OpCtx { template: &self.template, actor: self.actor, indices, phase: self.phase.as_deref() }
    }
}

/// One step of a walk in line order: what the dry run and the metrics walk consume.
pub enum Step<'c> {
    /// The think time before a line, as `compute`.
    Gap { lane: usize, ns: i64 },
    /// A group of `n` ops issued together follows (the members come as `Op`s).
    Group { lane: usize, n: usize },
    Op { lane: usize, op: Op<'c>, indices: &'c [i64] },
}

/// Walk the file in line order (the order the calls returned), tracking positions per open
/// id, with each op's indices the enclosing ones plus `[lane, ordinal in lane]`.
pub fn walk(tf: &TraceFile, base: &[i64], mut f: impl FnMut(Step<'_>) -> Result<()>) -> Result<()> {
    let n = base.len();
    let mut idx: Vec<i64> = base.to_vec();
    idx.extend_from_slice(&[0, 0]);
    let mut pos = vec![0i64; tf.opens.len()];
    let mut ord = vec![0i64; tf.lanes()];
    let mut end = vec![0i64; tf.lanes()];
    let mut begun = vec![false; tf.lanes()];
    for line in &tf.lines {
        let lane = line.lane;
        let gap = if begun[lane] { line.t - end[lane] } else { line.t };
        begun[lane] = true;
        end[lane] = line.t + line.dur;
        f(Step::Gap { lane, ns: gap.max(0) })?;
        idx[n] = lane as i64;
        idx[n + 1] = ord[lane];
        ord[lane] += 1;
        let members: Vec<&LineOp> = match &line.op {
            LineOp::Submit { ops } => {
                f(Step::Group { lane, n: ops.len() })?;
                ops.iter().collect()
            }
            op => vec![op],
        };
        for op in members {
            let b = tf.build(op, |fd| pos[fd]);
            if let (Some(fd), Some(p)) = (b.oid, b.pos_after) {
                pos[fd] = p;
            }
            f(Step::Op { lane, op: b.op, indices: &idx })?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- the abstract's trace nodes

/// Every `trace` node of the AST: `(file, sha256, actor template)`.
pub fn refs(ast: &Ast) -> Vec<(&str, &str, &str)> {
    fn body<'a>(nodes: &'a [Node], template: &'a str, out: &mut Vec<(&'a str, &'a str, &'a str)>) {
        for n in nodes {
            match n {
                Node::Trace { file, sha256 } => out.push((file, sha256, template)),
                Node::Loop { body: b, .. } | Node::Parallel { body: b, .. } | Node::Loader { body: b, .. } | Node::Phase { body: b, .. } => body(b, template, out),
                Node::Cond { then, otherwise, .. } => {
                    body(then, template, out);
                    if let Some(o) = otherwise {
                        body(o, template, out);
                    }
                }
                Node::Choose { arms, .. } => {
                    for a in arms {
                        body(&a.body, template, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    for (name, a) in &ast.actors {
        body(&a.body, name, &mut out);
    }
    out
}

/// Load every trace file an AST names, relative to `dir` (the document's directory).
pub fn load_all(ast: &Ast, dir: &Path) -> Result<HashMap<String, Arc<TraceFile>>> {
    let mut out: HashMap<String, Arc<TraceFile>> = HashMap::new();
    for (file, sha, _) in refs(ast) {
        if let Some(t) = out.get(file) {
            if t.sha256 != sha {
                bail!("trace `{file}` is named with two sha256 values");
            }
            continue;
        }
        let path = if Path::new(file).is_absolute() { PathBuf::from(file) } else { dir.join(file) };
        out.insert(file.to_string(), Arc::new(TraceFile::load(file, &path, sha)?));
    }
    Ok(out)
}

// ---------------------------------------------------------------- before the gate

/// What the check found about one trace file's paths under `--root`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RootCheck {
    pub file: String,
    pub inputs: usize,
    pub creates: usize,
}

/// Every path the trace reads without creating must exist with at least the farthest read's
/// extent; every path it creates must not exist (after `--clean-namespaces` removed them).
pub fn check_root(tf: &TraceFile, root: &Path) -> Result<RootCheck> {
    for p in &tf.inputs {
        let full = root.join(p);
        let md = std::fs::symlink_metadata(&full).map_err(|e| anyhow!("trace `{}`: {} ({}): {e}: the application's data must be under --root", tf.name, p, full.display()))?;
        if let Some(need) = tf.read_extent.get(p) {
            if md.is_file() && (md.len() as i64) < *need {
                bail!("trace `{}`: {} is {} bytes, the trace reads to {need}", tf.name, p, md.len());
            }
        }
    }
    for p in &tf.header.creates {
        let full = root.join(p);
        if std::fs::symlink_metadata(&full).is_ok() {
            bail!("trace `{}`: {} exists, and the trace creates it: remove it or pass --clean-namespaces", tf.name, p);
        }
    }
    Ok(RootCheck { file: tf.name.clone(), inputs: tf.inputs.len(), creates: tf.header.creates.len() })
}

/// `--clean-namespaces` for a trace: remove what the trace creates, last first, nothing else.
pub fn clean(tf: &TraceFile, root: &Path) -> Result<usize> {
    let mut removed = 0;
    for p in tf.header.creates.iter().rev() {
        let full = root.join(p);
        let r = match std::fs::symlink_metadata(&full) {
            Ok(md) if md.is_dir() => std::fs::remove_dir_all(&full),
            Ok(_) => std::fs::remove_file(&full),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => Err(e),
        };
        r.with_context(|| format!("trace `{}`: removing {}", tf.name, full.display()))?;
        removed += 1;
    }
    Ok(removed)
}

/// V16: a `trace` that creates paths may not run in an actor template with more than one
/// instance (the copies would write the same files). Checked with the resolved counts.
pub fn check_counts(ast: &Ast, traces: &HashMap<String, Arc<TraceFile>>, counts: &[(&str, i64)]) -> Result<()> {
    for (file, _, template) in refs(ast) {
        let Some(tf) = traces.get(file) else { continue };
        let count = counts.iter().find(|(t, _)| *t == template).map(|(_, c)| *c).unwrap_or(0);
        if count > 1 && !tf.header.creates.is_empty() {
            bail!("V16: trace `{file}` creates {} path(s) and actor `{template}` has {count} instances; a trace that writes runs in one instance (set the actor's count to 1)", tf.header.creates.len());
        }
    }
    Ok(())
}
