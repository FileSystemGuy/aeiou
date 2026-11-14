//! Use cases, benchmark rules, and the lock (contract 0.11, `DESIGN_REVIEW.md` §3.78): what
//! `--case NAME` and `--case-file PATH` read. A lock (`schema/lock.schema.json`, written by
//! `aeiou-params lock`, never edited) holds each run's abstract by its path below aeiou's
//! `workloads/` and its `ast_sha256`, every parameter with its state and source, the identity
//! options a fragment set, and a rules file's freeze and constraints. This module finds and
//! reads a lock, turns a run of it into the parameter layer the model is built from, computes
//! the **workload id** (the canonical resolved values, without the submission's required
//! values and the storage-reported numbers), and evaluates the constraints over the runner's
//! `fact.` and `result.` names at the time each is known.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::ast::Ast;
use crate::eval::{entry, ParamSet, State};

/// The `lock_version` this runner reads.
pub const LOCK_VERSION: u64 = 1;
/// A use case's lock beside its `case.toml`.
pub const LOCK_FILE: &str = "case.lock.json";
/// The runner's identity options a lock may set (§3.78 point 3), as its `run` table names them.
pub const IDENTITY: &[&str] = &["seed", "posix", "cache", "s3", "time_scale", "rank_rotate", "write_compress"];

/// An entry of a lock: its state and, unless required, its value.
pub type Entry = (State, Option<Value>);

#[derive(Debug, Clone)]
pub struct LockRun {
    /// The abstract's name, what `--case-run` names.
    pub name: String,
    /// The abstract's path below `workloads/`.
    pub abstract_rel: String,
    pub ast_sha256: String,
    pub params: BTreeMap<String, Entry>,
    pub sources: BTreeMap<String, String>,
    pub run: BTreeMap<String, Entry>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub text: String,
    pub expr: Value,
    pub names: Vec<String>,
    pub doc: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Lock {
    pub path: String,
    /// SHA-256 of the file's bytes: the **file hash**, whether this is the exact file published.
    pub sha256: String,
    pub name: String,
    pub runs: Vec<LockRun>,
    /// The rules' freeze locked every identity option (`identity_options: locked`).
    pub identity_frozen: bool,
    pub constraints: Vec<Constraint>,
    /// The file as read, embedded in the report (§3.78 point 14).
    pub doc: Value,
}

fn entries(v: Option<&Value>, what: &str) -> Result<BTreeMap<String, Entry>> {
    let mut out = BTreeMap::new();
    if let Some(v) = v {
        for (k, e) in v.as_object().ok_or_else(|| anyhow!("`{what}` is not an object"))? {
            out.insert(k.clone(), entry(e).with_context(|| format!("{what}: {k}"))?);
        }
    }
    Ok(out)
}

impl Lock {
    pub fn load(path: &Path) -> Result<Lock> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text, &path.display().to_string()).with_context(|| format!("lock {}", path.display()))
    }

    pub fn parse(text: &str, path: &str) -> Result<Lock> {
        use sha2::Digest;
        let sha256 = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));
        let doc: Value = serde_json::from_str(text).context("not JSON")?;
        let ints = crate::canon::int_range_errors(text);
        if !ints.is_empty() {
            bail!("{}", ints.join("\n  "));
        }
        let o = doc.as_object().ok_or_else(|| anyhow!("not a JSON object"))?;
        let version = o.get("lock_version").and_then(Value::as_u64).ok_or_else(|| anyhow!("not a lock: no `lock_version` (aeiou-params lock writes one)"))?;
        if version != LOCK_VERSION {
            bail!("lock_version {version} (this runner reads {LOCK_VERSION})");
        }
        for k in o.keys() {
            if !matches!(k.as_str(), "lock_version" | "name" | "doc" | "runs" | "freeze" | "constraints" | "provenance") {
                bail!("unknown key `{k}`");
            }
        }
        let name = o.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("`name` missing"))?.to_string();
        let mut runs = Vec::new();
        for (i, r) in o.get("runs").and_then(Value::as_array).filter(|a| !a.is_empty()).ok_or_else(|| anyhow!("`runs` missing or empty"))?.iter().enumerate() {
            let s = |k: &str| r.get(k).and_then(Value::as_str).map(str::to_string).ok_or_else(|| anyhow!("runs/{i}: `{k}` missing"));
            let sources = r.get("sources").and_then(Value::as_object).map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default();
            let run = LockRun {
                name: s("name")?,
                abstract_rel: s("abstract")?,
                ast_sha256: s("ast_sha256")?,
                params: entries(r.get("params"), &format!("runs/{i}/params"))?,
                sources,
                run: entries(r.get("run"), &format!("runs/{i}/run"))?,
            };
            if let Some(k) = run.run.keys().find(|k| !IDENTITY.contains(&k.as_str())) {
                bail!("runs/{i}/run: `{k}` is not an identity option ({})", IDENTITY.join(", "));
            }
            runs.push(run);
        }
        let identity_frozen = o.get("freeze").and_then(|f| f.get("identity_options")).and_then(Value::as_str) == Some("locked");
        let mut constraints = Vec::new();
        for (i, c) in o.get("constraints").and_then(Value::as_array).into_iter().flatten().enumerate() {
            constraints.push(Constraint {
                text: c.get("text").and_then(Value::as_str).ok_or_else(|| anyhow!("constraints/{i}: `text` missing"))?.to_string(),
                expr: c.get("expr").cloned().ok_or_else(|| anyhow!("constraints/{i}: `expr` missing"))?,
                names: c.get("names").and_then(Value::as_array).map(|a| a.iter().filter_map(|n| n.as_str().map(str::to_string)).collect()).unwrap_or_default(),
                doc: c.get("doc").and_then(Value::as_str).map(str::to_string),
            });
        }
        Ok(Lock { path: path.to_string(), sha256, name, runs, identity_frozen, constraints, doc })
    }

    /// The run `--case-run` names: required when the lock has several, refused when it has one.
    pub fn run(&self, which: Option<&str>) -> Result<&LockRun> {
        let names = || self.runs.iter().map(|r| r.name.as_str()).collect::<Vec<_>>().join(", ");
        match (which, self.runs.len()) {
            (None, 1) => Ok(&self.runs[0]),
            (None, _) => crate::usage!("lock {} has {} runs ({}): name one with --case-run", self.name, self.runs.len(), names()),
            (Some(w), 1) => crate::usage!("--case-run {w}: lock {} has one run ({}); --case-run is for a lock of several", self.name, names()),
            (Some(w), _) => self.runs.iter().find(|r| r.name == w).ok_or_else(|| crate::usage::err(format!("--case-run {w}: lock {} has no run of that name ({})", self.name, names()))),
        }
    }
}

impl LockRun {
    /// The run's values as the layer under every file and `--param` of the command line: a
    /// lock's locked value is final there too. An entry whose value is a storage-reported rule
    /// keeps the abstract's source.
    pub fn param_set(&self, lock: &Lock) -> ParamSet {
        let (mut values, mut states, mut keep) = (BTreeMap::new(), BTreeMap::new(), BTreeSet::new());
        for (k, (state, v)) in &self.params {
            states.insert(k.clone(), *state);
            match v {
                Some(v) if v.get("from").is_some() => {
                    keep.insert(k.clone());
                }
                Some(v) => {
                    values.insert(k.clone(), v.clone());
                }
                None => {}
            }
        }
        ParamSet { path: format!("lock {}", lock.name), sha256: lock.sha256.clone(), abstract_name: self.name.clone(), ast_sha256: Some(self.ast_sha256.clone()), doc: None, values, states, keep }
    }
}

/// aeiou's `workloads/` directory: `--workloads DIR`, else beside the executable's `bin/` (the
/// release tarball), else the source checkout's (`runner/target/<profile>/aeiou`).
pub fn workloads_dir(given: Option<&Path>) -> Result<PathBuf> {
    if let Some(d) = given {
        if !d.is_dir() {
            crate::usage!("--workloads {}: not a directory", d.display());
        }
        return Ok(d.to_path_buf());
    }
    let exe = std::env::current_exe().context("the executable's path")?;
    let dir = exe.parent().unwrap_or(Path::new("."));
    for c in [dir.join("../workloads"), dir.join("../../../workloads")] {
        if c.is_dir() {
            return Ok(c.canonicalize().unwrap_or(c));
        }
    }
    crate::usage!("aeiou's workloads/ is not beside {} (bin/../workloads in a release): name it with --workloads DIR", exe.display())
}

/// The lock of `name` under `workloads`: a walk for every `case.lock.json`, matched by its
/// `name` (`mechanism/case`) or by its case alone; none or several refused, every match listed,
/// and with none every lock that could not be read, which may be the one named.
pub fn find(workloads: &Path, name: &str) -> Result<PathBuf> {
    let (mut found, mut unread, mut all) = (Vec::new(), Vec::new(), Vec::new());
    let mut stack = vec![workloads.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some(LOCK_FILE) {
                match Lock::load(&p) {
                    Ok(l) if crate::listing::matches_name(&l.name, name) => found.push((l.name, p)),
                    Ok(l) => all.push(l.name),
                    Err(e) => unread.push(format!("{e:#}")),
                }
            }
        }
    }
    found.sort();
    unread.sort();
    let unread = match unread.len() {
        0 => String::new(),
        n => format!("; {n} lock{} could not be read:\n  {}", if n == 1 { "" } else { "s" }, unread.join("\n  ")),
    };
    match found.len() {
        1 => Ok(found.remove(0).1),
        0 => {
            all.sort();
            crate::usage!("--case {name}: no lock of that name under {} (a case's lock is cases/<case>/{LOCK_FILE}, made by `aeiou-params lock`){}{unread}", workloads.display(), crate::listing::not_found(&all, name))
        }
        _ => crate::usage!("--case {name}: {} locks match ({}): name it as mechanism/case", found.len(), found.iter().map(|(n, p)| format!("{n} at {}", p.display())).collect::<Vec<_>>().join(", ")),
    }
}

/// The workload id (§3.78 point 12): SHA-256 of the canonical form of what the run does. For
/// the run executed, every parameter's value in effect (the command line's replacements of a
/// default included), `{"required": true}` for a value the lock leaves to the submission, and
/// the rule for a storage-reported one, with the identity options in effect; for the lock's
/// other runs, their entries as the lock holds them. Never the lock's name, docs, sources,
/// constraints, or provenance: a corrected `doc` does not split the results.
pub fn workload_id(lock: &Lock, run: &LockRun, ast: &Ast, values: &BTreeMap<String, Value>, given: &BTreeSet<String>, options: &Value) -> String {
    let mut params = Map::new();
    for (name, p) in &ast.params {
        let state = run.params.get(name).map(|e| e.0);
        let required = state == Some(State::Required) || (state.is_none() && p.required());
        let v = if required {
            json!({"required": true})
        } else if let (Some(src), false) = (&p.source, given.contains(name) || matches!(run.params.get(name), Some((_, Some(v)))  if v.get("from").is_none())) {
            json!({"from": "st_blksize", "of": src.of, "fallback": src.fallback})
        } else {
            values.get(name).cloned().unwrap_or(Value::Null)
        };
        params.insert(name.clone(), v);
    }
    let entries_json = |m: &BTreeMap<String, Entry>| -> Value {
        Value::Object(m.iter().map(|(k, (_, v))| (k.clone(), v.clone().unwrap_or_else(|| json!({"required": true})))).collect())
    };
    let mut with: Vec<Value> = lock
        .runs
        .iter()
        .filter(|r| r.name != run.name)
        .map(|r| json!({"ast_sha256": r.ast_sha256, "params": entries_json(&r.params), "run": entries_json(&r.run)}))
        .collect();
    with.sort_by_key(|v| v["ast_sha256"].as_str().unwrap_or("").to_string());
    crate::canon::sha256_hex(&json!({
        "lock_version": LOCK_VERSION,
        "run": {"ast_sha256": run.ast_sha256, "params": Value::Object(params), "options": options},
        "with": with,
    }))
}

// ---------------------------------------------------------------- facts, results, constraints

/// When a name is known (§3.78 point 8): at resolve (from the parameters and `--gpus`), at the
/// gate (on the hosts, before any I/O), or after the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum When {
    Resolve,
    Gate,
    After,
}

impl When {
    pub fn name(self) -> &'static str {
        match self {
            When::Resolve => "resolve",
            When::Gate => "the gate",
            When::After => "after the run",
        }
    }
}

/// The runner's facts and results, each defined once here and in `runner/REFERENCE.md`;
/// changing a definition is a recorded decision.
pub const NAMES: &[(&str, When, &str)] = &[
    ("fact.gpus", When::Resolve, "G, the GPU instances of the run (--gpus)"),
    ("fact.dataset_bytes", When::Resolve, "the bytes of every dataset, as datagen writes them (dry-run's projection)"),
    ("fact.min_samples_per_gpu", When::Resolve, "the least, over the consumed datasets, of floor(count / G): the distinct samples (or shards) one GPU reads in an epoch"),
    ("fact.hosts", When::Gate, "the hosts of the run (--ranks)"),
    ("fact.sum_host_mem", When::Gate, "the hosts' memory summed, each host's MemTotal or its cgroup memory.max, whichever is less"),
    ("fact.min_host_mem", When::Gate, "the least of the hosts' memory, as fact.sum_host_mem counts it"),
    ("fact.max_gpus_per_host", When::Gate, "the most GPU instances one host runs: ceil(G / hosts)"),
    ("result.elapsed_s", When::After, "the run's elapsed seconds, from the start gate to the end"),
    ("result.busy_fraction", When::After, "compute x time-scale / (compute x time-scale + stall) over every take of every instance but each instance's first"),
    ("result.min_gpu_busy_fraction", When::After, "the least, over the instances that take, of the busy fraction of that instance's takes but its first"),
];

/// A timeline span's statistics, `result.span.<span>.<stat>` (brief item 72 (7) step 5): D's
/// count, sum, nearest-rank p50 and p99, and maximum, and the worst 5-minute window's p99 by
/// planned start, in seconds.
pub const SPAN_STATS: &[(&str, &str)] = &[
    ("count", "the spans of that name that ended"),
    ("sum_s", "their D summed"),
    ("p50_s", "D's nearest-rank median"),
    ("p99_s", "D's nearest-rank 99th percentile"),
    ("max_s", "D's largest"),
    ("window_p99_s", "the largest, over the 5-minute windows of planned start, of a window's p99"),
];

/// The lag of a timeline's body lanes that issued ops, `result.lag.<stat>`: their final L, in
/// seconds (worker and eviction lanes' L is never carried, so not among them).
pub const LAG_STATS: &[(&str, &str)] = &[("p50_s", "the lanes' final L, nearest-rank median"), ("p99_s", "nearest-rank 99th percentile"), ("max_s", "largest")];

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_ascii_alphabetic() || f == '_') && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

/// The span a `result.span.<span>.<stat>` name reads, with the statistic.
fn span_name(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix("result.span.")?;
    let (span, stat) = rest.split_once('.')?;
    (is_ident(span) && SPAN_STATS.iter().any(|(s, _)| *s == stat)).then_some((span, stat))
}

pub fn when_of(name: &str) -> Option<When> {
    if let Some((_, w, _)) = NAMES.iter().find(|(n, _, _)| *n == name) {
        return Some(*w);
    }
    if span_name(name).is_some() || name.strip_prefix("result.lag.").is_some_and(|s| LAG_STATS.iter().any(|(x, _)| *x == s)) {
        return Some(When::After);
    }
    None
}

/// The constraints certain to fail once a window of planned start closes above them (the user's
/// choice of 2026-11-14): of the form `result.span.SPAN.window_p99_s <= C` (or `<`), or `.max_s`,
/// a maximum over windows, so one window above C decides it. Others are judged after the run.
pub fn watches(lock: &Lock) -> Vec<crate::exec::Watch> {
    let mut out = Vec::new();
    for c in &lock.constraints {
        let Some(o) = c.expr.as_object().filter(|o| o.len() == 1) else { continue };
        let (op, args) = o.iter().next().expect("one");
        let strict = match op.as_str() {
            "le" => false,
            "lt" => true,
            _ => continue,
        };
        let Some([l, r]) = args.as_array().map(|a| a.as_slice()).and_then(|a| <&[Value; 2]>::try_from(a).ok()) else { continue };
        let (Some(name), Some(limit)) = (l.get("result").and_then(Value::as_str), r.as_f64()) else { continue };
        let full = format!("result.{name}");
        let Some((span, stat)) = span_name(&full) else { continue };
        let max = match stat {
            "max_s" => true,
            "window_p99_s" => false,
            _ => continue,
        };
        out.push(crate::exec::Watch { span: span.to_string(), max, limit_ns: (limit * 1e9).round() as i64, strict, text: c.text.clone() });
    }
    out
}

/// Every constraint's `result.span.<span>.` names a span the abstract declares (`spans`):
/// checked when the lock is read, so a misspelled span fails before the run, not after it.
pub fn check_span_names(lock: &Lock, spans: &BTreeSet<String>) -> Result<()> {
    for c in &lock.constraints {
        for n in names_in(&c.expr) {
            if let Some((span, _)) = span_name(&n) {
                if !spans.contains(span) {
                    let have = if spans.is_empty() { "none".to_string() } else { spans.iter().cloned().collect::<Vec<_>>().join(", ") };
                    bail!("constraint `{}`: the abstract declares no span `{span}` (its spans: {have})", c.text);
                }
            }
        }
    }
    Ok(())
}

impl Constraint {
    /// The time it is checked: the latest of its names'; a name the runner does not define is
    /// refused, so a misspelled result fails at resolve, never after a long run.
    pub fn when(&self) -> Result<When> {
        let mut w = When::Resolve;
        for n in names_in(&self.expr) {
            w = w.max(when_of(&n).ok_or_else(|| {
                let span: Vec<&str> = SPAN_STATS.iter().map(|(s, _)| *s).collect();
                let lag: Vec<&str> = LAG_STATS.iter().map(|(s, _)| *s).collect();
                anyhow!("constraint `{}`: `{n}` is not a name of the runner ({}, result.span.SPAN.{{{}}}, result.lag.{{{}}})", self.text, NAMES.iter().map(|(n, _, _)| *n).collect::<Vec<_>>().join(", "), span.join(","), lag.join(","))
            })?);
        }
        Ok(w)
    }
}

fn names_in(e: &Value) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(e: &Value, out: &mut Vec<String>) {
        match e {
            Value::Object(o) => {
                for (k, v) in o {
                    match (k.as_str(), v.as_str()) {
                        ("fact" | "result", Some(n)) => out.push(format!("{k}.{n}")),
                        _ => walk(v, out),
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    walk(e, &mut out);
    out
}

/// A number of a constraint: integer arithmetic where both operands are integers (`div` and
/// `mod` Euclidean, as the abstract's), a float otherwise; a comparison is 1 or 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    I(i64),
    F(f64),
}

impl Num {
    pub fn f(self) -> f64 {
        match self {
            Num::I(n) => n as f64,
            Num::F(x) => x,
        }
    }

    fn truth(self) -> bool {
        self.f() != 0.0
    }

    pub fn show(self) -> String {
        match self {
            Num::I(n) => n.to_string(),
            Num::F(x) => format!("{x}"),
        }
    }
}

/// The value of a constraint's expression, its names read through `get`.
pub fn eval(e: &Value, get: &dyn Fn(&str) -> Option<Num>) -> Result<Num> {
    Ok(match e {
        Value::Number(n) => n.as_i64().map(Num::I).or_else(|| n.as_f64().map(Num::F)).ok_or_else(|| anyhow!("{n}: not a number"))?,
        Value::Bool(b) => Num::I(*b as i64),
        Value::Object(o) if o.len() == 1 => {
            let (k, v) = o.iter().next().expect("one key");
            let args = |n: usize| -> Result<Vec<Num>> {
                let a = v.as_array().filter(|a| a.len() == n).ok_or_else(|| anyhow!("`{k}` takes {n} operands"))?;
                a.iter().map(|x| eval(x, get)).collect()
            };
            let two = |f: fn(i64, i64) -> Option<i64>, g: fn(f64, f64) -> f64| -> Result<Num> {
                let a = args(2)?;
                Ok(match (a[0], a[1]) {
                    (Num::I(x), Num::I(y)) => Num::I(f(x, y).ok_or_else(|| anyhow!("`{k}` of {x} and {y} is not an integer of 64 bits"))?),
                    (x, y) => Num::F(g(x.f(), y.f())),
                })
            };
            // two integers compare as integers: a byte count passes 2^53, where an f64 rounds
            let cmp = |f: fn(Option<std::cmp::Ordering>) -> bool| -> Result<Num> {
                let a = args(2)?;
                let o = match (a[0], a[1]) {
                    (Num::I(x), Num::I(y)) => Some(x.cmp(&y)),
                    (x, y) => x.f().partial_cmp(&y.f()),
                };
                Ok(Num::I(f(o) as i64))
            };
            let nonzero = |op: &str| -> Result<()> {
                let a = args(2)?;
                if a[1].f() == 0.0 {
                    bail!("`{op}` by zero");
                }
                Ok(())
            };
            match k.as_str() {
                "fact" | "result" => {
                    let n = format!("{k}.{}", v.as_str().unwrap_or(""));
                    get(&n).ok_or_else(|| anyhow!("`{n}` is not known here"))?
                }
                "add" => two(i64::checked_add, |x, y| x + y)?,
                "sub" => two(i64::checked_sub, |x, y| x - y)?,
                "mul" => two(i64::checked_mul, |x, y| x * y)?,
                "div" => {
                    nonzero("div")?;
                    two(|x, y| x.checked_div_euclid(y), |x, y| (x / y).floor())?
                }
                "mod" => {
                    nonzero("mod")?;
                    two(|x, y| x.checked_rem_euclid(y), |x, y| x.rem_euclid(y))?
                }
                "min" => two(|x, y| Some(x.min(y)), f64::min)?,
                "max" => two(|x, y| Some(x.max(y)), f64::max)?,
                "neg" => match eval(v, get)? {
                    Num::I(x) => Num::I(x.checked_neg().ok_or_else(|| anyhow!("`neg` of {x}"))?),
                    Num::F(x) => Num::F(-x),
                },
                "eq" => cmp(|o| o.is_some_and(|o| o.is_eq()))?,
                "ne" => cmp(|o| !o.is_some_and(|o| o.is_eq()))?,
                "lt" => cmp(|o| o.is_some_and(|o| o.is_lt()))?,
                "le" => cmp(|o| o.is_some_and(|o| o.is_le()))?,
                "gt" => cmp(|o| o.is_some_and(|o| o.is_gt()))?,
                "ge" => cmp(|o| o.is_some_and(|o| o.is_ge()))?,
                "and" => {
                    let a = args(2)?;
                    Num::I((a[0].truth() && a[1].truth()) as i64)
                }
                "or" => {
                    let a = args(2)?;
                    Num::I((a[0].truth() || a[1].truth()) as i64)
                }
                "not" => Num::I(!eval(v, get)?.truth() as i64),
                other => bail!("`{other}` is not an operator of a constraint"),
            }
        }
        other => bail!("{other}: not an expression of a constraint"),
    })
}

/// A constraint's verdict, as the header and the report give it.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub text: String,
    pub doc: Option<String>,
    pub when: When,
    pub passed: bool,
    /// Each name it read with its number.
    pub numbers: Vec<(String, String)>,
}

impl Verdict {
    pub fn line(&self) -> String {
        format!(
            "{} {}{}  ({})",
            if self.passed { "pass" } else { "FAIL" },
            self.text,
            self.doc.as_ref().map(|d| format!("  [{d}]")).unwrap_or_default(),
            self.numbers.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ")
        )
    }

    pub fn json(&self) -> Value {
        json!({"expr": self.text, "doc": self.doc, "when": self.when.name(), "passed": self.passed, "numbers": self.numbers.iter().map(|(n, v)| (n.clone(), Value::String(v.clone()))).collect::<Map<_, _>>()})
    }
}

/// The verdicts of the constraints checked at `when`, every name read through `get`.
pub fn judge(lock: &Lock, when: When, get: &dyn Fn(&str) -> Option<Num>) -> Result<Vec<Verdict>> {
    let mut out = Vec::new();
    for c in &lock.constraints {
        if c.when()? != when {
            continue;
        }
        let v = eval(&c.expr, get).with_context(|| format!("constraint `{}`", c.text))?;
        let numbers = names_in(&c.expr).into_iter().collect::<BTreeSet<_>>().into_iter().map(|n| { let v = get(&n).map(Num::show).unwrap_or_default(); (n, v) }).collect();
        out.push(Verdict { text: c.text.clone(), doc: c.doc.clone(), when, passed: v.truth(), numbers });
    }
    Ok(out)
}

/// The facts known at resolve (`fact.gpus`, `fact.dataset_bytes`, `fact.min_samples_per_gpu`):
/// from `--gpus` and each dataset's projection, what `aeiou datagen` writes.
pub fn resolve_facts(ast: &Ast, projections: &[crate::project::Projection], gpus: i64) -> BTreeMap<String, Num> {
    let mut out = BTreeMap::new();
    out.insert("fact.gpus".into(), Num::I(gpus));
    out.insert("fact.dataset_bytes".into(), Num::I(projections.iter().map(|p| p.bytes as i64).sum()));
    let consumed = projections.iter().filter(|p| ast.datasets.get(&p.name).is_some_and(|d| !matches!(d.access, crate::ast::Access::Indexed)));
    if let Some(m) = consumed.map(|p| p.samples.div_euclid(gpus.max(1))).min() {
        out.insert("fact.min_samples_per_gpu".into(), Num::I(m));
    }
    out
}

/// The facts known at the gate, from every host's facts (`run::host_facts`, in its Hello):
/// `fact.hosts`, `fact.max_gpus_per_host`, and, when every host knows its memory,
/// `fact.sum_host_mem` and `fact.min_host_mem` (each host's `MemTotal` or cgroup `memory.max`,
/// whichever is less).
pub fn gate_facts(hosts: &[Value], gpus: i64) -> BTreeMap<String, Num> {
    let mut out = BTreeMap::new();
    let n = hosts.len().max(1) as i64;
    out.insert("fact.hosts".into(), Num::I(n));
    out.insert("fact.max_gpus_per_host".into(), Num::I((gpus + n - 1) / n));
    let mem: Option<Vec<i64>> = hosts
        .iter()
        .map(|f| {
            let total = f["memory"].as_u64()?;
            Some(f["cgroup_memory_max"].as_u64().map_or(total, |c| c.min(total)) as i64)
        })
        .collect();
    if let Some(mem) = mem.filter(|m| !m.is_empty()) {
        out.insert("fact.sum_host_mem".into(), Num::I(mem.iter().sum()));
        out.insert("fact.min_host_mem".into(), Num::I(*mem.iter().min().expect("not empty")));
    }
    out
}

/// The results known after the run, from the (merged) report: `result.elapsed_s`, and the busy
/// fractions over every take but each instance's first, which waits on the pipeline filling.
pub fn results(report: &crate::run::Report) -> BTreeMap<String, Num> {
    let mut out = BTreeMap::new();
    out.insert("result.elapsed_s".into(), Num::F(report.elapsed.as_secs_f64()));
    let busy = |takes: &[crate::run::TakeRec]| {
        let (c, s) = takes.iter().skip(1).fold((0u64, 0u64), |(c, s), t| (c + t.compute_ns, s + t.stall_ns));
        (c, s)
    };
    let per: Vec<(u64, u64)> = report.actors.iter().filter(|a| a.takes.len() > 1).map(|a| busy(&a.takes)).collect();
    if !per.is_empty() {
        let (c, s) = per.iter().fold((0, 0), |(c, s), (x, y)| (c + x, s + y));
        out.insert("result.busy_fraction".into(), Num::F(crate::report::busy(c, s, report.time_scale)));
        let least = per.iter().map(|(c, s)| crate::report::busy(*c, *s, report.time_scale)).fold(f64::INFINITY, f64::min);
        out.insert("result.min_gpu_busy_fraction".into(), Num::F(least));
    }
    // a timeline's spans and its lanes' lag (brief item 72 (7) step 5)
    let s = |ns: i64| Num::F(ns as f64 / 1e9);
    for (name, ds) in &report.stats.spans {
        let m = crate::exec::summarize(ds);
        out.insert(format!("result.span.{name}.count"), Num::I(m.count as i64));
        out.insert(format!("result.span.{name}.sum_s"), s(m.sum));
        out.insert(format!("result.span.{name}.p50_s"), s(m.p50));
        out.insert(format!("result.span.{name}.p99_s"), s(m.p99));
        out.insert(format!("result.span.{name}.max_s"), s(m.max));
        out.insert(format!("result.span.{name}.window_p99_s"), s(m.window_p99));
    }
    if !report.stats.lane_lag.is_empty() {
        let mut l = report.stats.lane_lag.clone();
        l.sort_unstable();
        out.insert("result.lag.p50_s".into(), s(crate::exec::quantile(&l, 0.5)));
        out.insert("result.lag.p99_s".into(), s(crate::exec::quantile(&l, 0.99)));
        out.insert("result.lag.max_s".into(), s(*l.last().unwrap()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span's and the lanes' lag names are known after the run, a span's for each statistic
    /// the runner defines; any other is not a name of the runner (brief item 72 (7) step 5).
    #[test]
    fn span_and_lag_names_are_results() {
        for n in ["result.span.load.count", "result.span.load.sum_s", "result.span.load.p50_s", "result.span.load.p99_s", "result.span.load.max_s", "result.span.load.window_p99_s", "result.lag.p50_s", "result.lag.p99_s", "result.lag.max_s"] {
            assert_eq!(when_of(n), Some(When::After), "{n}");
        }
        for n in ["result.span.load", "result.span.load.p98_s", "result.span..p99_s", "result.lag.mean_s", "result.span.a.b.p99_s"] {
            assert_eq!(when_of(n), None, "{n}");
        }
    }

    /// The results of a run with spans: each span name's statistics in seconds, the worst
    /// window's p99, and the body lanes' final L; none of them in a run without a timeline.
    #[test]
    fn a_runs_spans_and_lag_are_results() {
        let mut r = crate::run::Report::merge_all(vec![]);
        assert!(!results(&r).keys().any(|k| k.starts_with("result.span.") || k.starts_with("result.lag.")));
        r.stats.spans.insert("load".into(), vec![(0, 1_000_000_000), (crate::exec::WINDOW_NS, 3_000_000_000)]);
        r.stats.lane_lag = vec![0, 2_000_000_000];
        let got = results(&r);
        let f = |n: &str| match got[n] {
            Num::F(x) => x,
            Num::I(x) => x as f64,
        };
        assert_eq!(f("result.span.load.count"), 2.0);
        assert_eq!(f("result.span.load.sum_s"), 4.0);
        assert_eq!(f("result.span.load.p50_s"), 1.0);
        assert_eq!(f("result.span.load.p99_s"), 3.0);
        assert_eq!(f("result.span.load.max_s"), 3.0);
        assert_eq!(f("result.span.load.window_p99_s"), 3.0);
        assert_eq!(f("result.lag.p50_s"), 0.0);
        assert_eq!(f("result.lag.max_s"), 2.0);
    }

    /// The constraints a run watches, certain to fail once a window closes above them: a span's
    /// `window_p99_s` or `max_s` at most (or below) a number; any other form is judged after.
    #[test]
    fn window_and_max_constraints_are_watched() {
        let c = |t: &str, e: Value| Constraint { text: t.into(), expr: e, names: vec![], doc: None };
        let lock = Lock { path: String::new(), sha256: String::new(), name: "c".into(), runs: vec![], identity_frozen: false, doc: Value::Null, constraints: vec![
            c("a", json!({"le": [{"result": "span.load.window_p99_s"}, 2]})),
            c("b", json!({"lt": [{"result": "span.load.max_s"}, 0.5]})),
            c("c", json!({"le": [{"result": "span.load.p99_s"}, 2]})),
            c("d", json!({"ge": [{"result": "span.load.max_s"}, 2]})),
            c("e", json!({"le": [2, {"result": "span.load.max_s"}]})),
            c("f", json!({"le": [{"result": "elapsed_s"}, 2]})),
        ] };
        let w = watches(&lock);
        assert_eq!(w, [
            crate::exec::Watch { span: "load".into(), max: false, limit_ns: 2_000_000_000, strict: false, text: "a".into() },
            crate::exec::Watch { span: "load".into(), max: true, limit_ns: 500_000_000, strict: true, text: "b".into() },
        ]);
    }

    /// A constraint over a span the abstract does not declare is refused when the lock is read,
    /// never after a long run.
    #[test]
    fn a_constraint_over_an_undeclared_span_is_refused() {
        let c = |t: &str, n: &str| Constraint { text: t.into(), expr: json!({"le": [{"result": n}, 2]}), names: vec![format!("result.{n}")], doc: None };
        let lock = Lock { path: String::new(), sha256: String::new(), name: "c".into(), runs: vec![], identity_frozen: false,
            constraints: vec![c("result.span.load.p99_s <= 2", "span.load.p99_s"), c("result.elapsed_s <= 2", "elapsed_s")], doc: Value::Null };
        let mut spans = BTreeSet::new();
        spans.insert("load".to_string());
        check_span_names(&lock, &spans).unwrap();
        let e = check_span_names(&lock, &BTreeSet::from(["store".to_string()])).unwrap_err().to_string();
        assert!(e.contains("constraint `result.span.load.p99_s <= 2`: the abstract declares no span `load` (its spans: store)"), "{e}");
    }
}
