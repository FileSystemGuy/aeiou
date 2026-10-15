//! The semantic rules of `schema/README.md` §4 (V1–V15), ported from `schema/check.py`. The
//! runner must reject everything the reference checker rejects; the structural rules the
//! JSON Schema states and serde cannot (identifier syntax, errno syntax, ranges, uniqueness)
//! are checked here too. Messages carry the `/`-joined path of the offending node, as
//! `check.py` prints it.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::*;

pub const RESERVED_PREFIX: &str = ".aeiou";

/// The protocols an abstract has names on: `s3` when a dataset or namespace declares it,
/// `posix` when one declares it (or none), when a `trace` node runs (its files are under
/// `--root`), or when no name is `s3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Protocols {
    pub posix: bool,
    pub s3: bool,
}

pub fn protocols(ast: &Ast, traces: bool) -> Protocols {
    let declared = ast
        .datasets
        .values()
        .map(|d| d.protocol)
        .chain(ast.namespaces.values().map(|n| n.protocol))
        .map(|p| p.unwrap_or_default());
    let (mut posix, mut s3) = (traces, false);
    for p in declared {
        match p {
            Protocol::Posix => posix = true,
            Protocol::S3 => s3 = true,
        }
    }
    Protocols { posix: posix || !s3, s3 }
}

pub fn check(ast: &Ast) -> Vec<String> {
    let mut c = Checker { ast, given: None, errors: Vec::new(), ops: BTreeMap::new(), traces: false, consumes: BTreeMap::new() };
    c.run();
    c.errors
}

/// Validate and return the op-kind counts (as `check.py` prints them) when there are no errors.
pub fn check_with_ops(ast: &Ast) -> Result<BTreeMap<&'static str, usize>, Vec<String>> {
    let mut c = Checker { ast, given: None, errors: Vec::new(), ops: BTreeMap::new(), traces: false, consumes: BTreeMap::new() };
    c.run();
    if c.errors.is_empty() {
        Ok(c.ops)
    } else {
        Err(c.errors)
    }
}

/// The rules again with the parameters in effect in place of the defaults (§3.57): rule V3
/// judges a `param` by its distribution's minimum, and a `--params-file` file or `--param` may
/// replace that distribution with one whose minimum is lower. Without this an `at` offset
/// of zero is a loop in the VM. Called from `Params::new`, so every subcommand gets it.
pub fn check_given(ast: &Ast, given: &BTreeMap<String, PValue>) -> Vec<String> {
    let mut c = Checker { ast, given: Some(given), errors: Vec::new(), ops: BTreeMap::new(), traces: false, consumes: BTreeMap::new() };
    c.run();
    c.errors
}

/// The lesser of two V25 values, NaN when either is (an undefined value, refused as no finite
/// size): `f64::min` drops a NaN and Python's `min` answers by argument order, and the two
/// checkers must judge one document alike.
fn least_of(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() { f64::NAN } else { x.min(y) }
}

/// The greater of two V25 values, NaN when either is (`least_of`).
fn most_of(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() { f64::NAN } else { x.max(y) }
}

/// The least a dataset's size can be (rule V25): a number, or no bound at all.
#[derive(Clone, Copy)]
enum Least {
    Known(f64),
    Unbounded,
}

struct Checker<'a> {
    ast: &'a Ast,
    /// the parameter values in effect, when the check is on a run and not on the document
    given: Option<&'a BTreeMap<String, PValue>>,
    errors: Vec<String>,
    ops: BTreeMap<&'static str, usize>,
    /// A `trace` node was seen: its files are under `--root`, POSIX names (V19).
    traces: bool,
    /// The `consume`s inside a loop, by dataset, in actor-name then document order, each with
    /// the innermost enclosing loop whose count varies with an enclosing one, if any (V6).
    consumes: BTreeMap<&'a str, Vec<(P, Option<P>)>>,
}

#[derive(Clone)]
struct Scope<'a> {
    bindings: Vec<HashMap<&'a str, &'a ExprLike>>, // innermost last
    indices: Vec<&'a str>,
    /// indices provably ≥ 0 (rule V3): a `loop` whose `from` is absent or provably ≥ 0, every `parallel` and `loader`
    nonneg_indices: HashSet<&'a str>,
    fields: HashSet<&'a str>,
    forward: HashSet<&'a str>,
    /// The `let` names this body binds more than once: an `at` naming one is refused (V3).
    twice: HashSet<&'a str>,
    /// per level of `bindings`, the bindings there whose value varies (V6)
    varying: Vec<HashSet<&'a str>>,
    /// the innermost enclosing frame whose count varies with an enclosing one (V6)
    count_varies: Option<P>,
}

/// State shared across every scope of one actor, as `check.py` shares `channels` and `written`.
#[derive(Default)]
struct ActorState<'a> {
    channels: HashSet<&'a str>,
    written: HashSet<&'a str>,
    /// Inside a `parallel` or `loader` body: the handles opened so far in the innermost one
    /// (by their `Debug` form, as `check.py` compares them by their JSON), for V24.
    fork_opens: Option<Vec<String>>,
    /// V26: inside a `loader` body, the loader's name and the path of the body it is in.
    loader: Option<(&'a str, P)>,
    /// V26: the actor's channel declarations (`channel` and `loader`), by name: the path of the
    /// body each is in, and whether a loop encloses it.
    sites: HashMap<&'a str, Vec<(P, bool)>>,
}

impl<'a> Scope<'a> {
    fn new(fields: HashSet<&'a str>) -> Self {
        Scope { bindings: vec![HashMap::new()], indices: Vec::new(), nonneg_indices: HashSet::new(), fields, forward: HashSet::new(), twice: HashSet::new(), varying: vec![HashSet::new()], count_varies: None }
    }

    fn child(&self) -> Self {
        let mut s = self.clone();
        s.bindings.push(HashMap::new());
        s.varying.push(HashSet::new());
        s.forward = HashSet::new();
        s.twice = HashSet::new();
        s
    }

    fn with_index(&self, name: &'a str, nonneg: bool, count_varies: Option<P>) -> Self {
        let mut s = self.child();
        s.indices.push(name);
        if nonneg {
            s.nonneg_indices.insert(name);
        } else {
            s.nonneg_indices.remove(name);
        }
        if count_varies.is_some() {
            s.count_varies = count_varies;
        }
        s
    }

    fn bind(&mut self, name: &'a str, value: &'a ExprLike, varies: bool) {
        self.bindings.last_mut().unwrap().insert(name, value);
        let v = self.varying.last_mut().unwrap();
        if varies {
            v.insert(name);
        } else {
            v.remove(name);
        }
    }

    fn is_varying(&self, name: &str) -> bool {
        self.bindings.iter().zip(&self.varying).rev().find(|(b, _)| b.contains_key(name)).map_or(false, |(_, v)| v.contains(name))
    }

    fn lookup(&self, name: &str) -> Option<&'a ExprLike> {
        self.bindings.iter().rev().find_map(|m| m.get(name).copied())
    }
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.chars().next().map_or(false, |c| c.is_ascii_lowercase() || c == '_')
        && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn is_errno(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 2 && b.len() <= 16 && b[0] == b'E' && b[1..].iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

pub fn dataset_root(d: &Dataset) -> String {
    let prefix = d.pattern.split('{').next().unwrap_or("");
    prefix.rsplit_once('/').map(|(a, _)| a).unwrap_or("").to_string()
}

/// The pattern's component directly below the dataset's root: the names it can put there.
fn inventory_sibling(d: &Dataset) -> String {
    let root = dataset_root(d);
    let rest = if root.is_empty() { d.pattern.as_str() } else { d.pattern.get(root.len() + 1..).unwrap_or("") };
    rest.split('/').next().unwrap_or("").to_string()
}

/// Whether `name` matches a pattern component, each `{field}` standing for one or more of the
/// characters an integer field can format to (digits, hex digits, a sign, padding).
fn glob_component(component: &str, name: &str) -> bool {
    let mut parts: Vec<&str> = Vec::new();
    let mut rest = component;
    while let Some(open) = rest.find('{') {
        parts.push(&rest[..open]);
        rest = match rest[open..].find('}') {
            Some(close) => &rest[open + close + 1..],
            None => "",
        };
    }
    parts.push(rest);
    fn m(parts: &[&str], s: &str) -> bool {
        let (first, more) = (parts[0], &parts[1..]);
        let Some(s) = s.strip_prefix(first) else { return false };
        if more.is_empty() {
            return s.is_empty();
        }
        let run = s.bytes().take_while(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c) || *c == b'-' || *c == b' ').count();
        (1..=run).any(|i| m(more, &s[i..]))
    }
    m(&parts, name)
}

fn namespace_root(pattern: &str) -> String {
    let prefix = pattern.split('{').next().unwrap_or("");
    prefix.rsplit_once('/').map(|(a, _)| a).unwrap_or("").to_string()
}

type P = Vec<String>;

fn p(base: &P, more: &[&str]) -> P {
    let mut v = base.clone();
    v.extend(more.iter().map(|s| s.to_string()));
    v
}

/// V26: an actor's channel declarations (`channel` and `loader`), by name: the path of the body
/// each is in, and whether a loop encloses it.
fn channel_sites<'a>(nodes: &'a [Node], path: &P, in_loop: bool, out: &mut HashMap<&'a str, Vec<(P, bool)>>) {
    for (i, node) in nodes.iter().enumerate() {
        let pp = p(path, &[&i.to_string(), node.kind()]);
        match node {
            Node::Channel { name, .. } => out.entry(name.as_str()).or_default().push((path.clone(), in_loop)),
            Node::Loader { name, body, .. } => {
                out.entry(name.as_str()).or_default().push((path.clone(), in_loop));
                channel_sites(body, &p(&pp, &["body"]), in_loop, out);
            }
            Node::Loop { body, .. } => channel_sites(body, &p(&pp, &["body"]), true, out),
            Node::Parallel { body, .. } | Node::Phase { body, .. } => channel_sites(body, &p(&pp, &["body"]), in_loop, out),
            Node::Cond { then, otherwise, .. } => {
                channel_sites(then, &p(&pp, &["then"]), in_loop, out);
                if let Some(b) = otherwise {
                    channel_sites(b, &p(&pp, &["else"]), in_loop, out);
                }
            }
            Node::Choose { arms, .. } => {
                for (j, arm) in arms.iter().enumerate() {
                    channel_sites(&arm.body, &p(&pp, &["arms", &j.to_string(), "body"]), in_loop, out);
                }
            }
            _ => {}
        }
    }
}

impl<'a> Checker<'a> {
    /// V20 (the fields each layout takes) and V21 (access) on one dataset (contract 0.8,
    /// `DESIGN_REVIEW.md` §3.72).
    fn layout_fields(&mut self, d: &Dataset, path: &P) {
        let has = [("samples_per_file", d.samples_per_file.is_some()), ("chunk", d.chunk.is_some()), ("slot", d.slot.is_some()), ("format", d.format.is_some())];
        let (need, may): (&[&str], &[&str]) = match d.layout {
            Layout::Whole => (&[], &[]),
            Layout::Chunked => (&["chunk"], &[]),
            Layout::Compound => (&["samples_per_file"], &["format"]),
            Layout::Array => (&["slot"], &[]),
        };
        let l = d.layout.name();
        for (k, present) in has {
            if need.contains(&k) && !present {
                self.err(path, format!("layout `{l}` needs `{k}` (V20)"));
            } else if present && !need.contains(&k) && !may.contains(&k) {
                self.err(&p(path, &[k]), format!("layout `{l}` takes no `{k}` (V20)"));
            }
        }
        let fields = crate::pattern::Pattern::parse(&d.pattern).map(|pt| pt.field_names().iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap_or_default();
        match d.layout {
            Layout::Array if !fields.is_empty() => self.err(&p(path, &["pattern"]), "an `array` is one file: its pattern has no fields (V20)"),
            Layout::Chunked if !fields.iter().any(|f| f == "k") => self.err(&p(path, &["pattern"]), "a `chunked` dataset's pattern needs a `{k}` field (V20)"),
            Layout::Whole | Layout::Compound if fields.iter().any(|f| f == "k") => self.err(&p(path, &["pattern"]), "`{k}` names a chunk: only a `chunked` dataset has one (V20)"),
            Layout::Whole | Layout::Chunked | Layout::Compound if !fields.iter().any(|f| f == "id") => self.err(&p(path, &["pattern"]), "the pattern needs an `{id}` field (V20)"),
            _ => {}
        }
        if d.layout == Layout::Array && d.access != Access::Indexed {
            self.err(&p(path, &["access"]), "an `array` is `indexed`: its slots are looked up by position (V21)");
        }
        if d.access.consumed() {
            if let Ok(pt) = crate::pattern::Pattern::parse(&d.pattern) {
                if pt.dir_id_fields().iter().any(|f| matches!(f.op, Some((crate::pattern::FieldOp::Mod, _)))) {
                    self.err(&p(path, &["pattern"]), "a consumed dataset's directories are `{id div D}` or none, never `mod`, so a subset of it is whole directories (V21)");
                }
            }
        }
    }

    /// V22 (contract 0.9, `DESIGN_REVIEW.md` §3.73): a dataset's inventory is a file directly
    /// in its root, of a consumed dataset, named so the pattern cannot produce it.
    fn inventory(&mut self, d: &Dataset, path: &P) {
        let Some(inv) = &d.inventory else { return };
        let ip = p(path, &["inventory"]);
        if !d.access.consumed() {
            self.err(&ip, "only a consumed dataset (`map` or `stream`) has an inventory: it stands in for the walk of a corpus the job draws from (V22)");
        }
        let n = inv.name.as_str();
        if n.is_empty() || n == "." || n == ".." || n.contains('/') || n.contains('{') || n.contains('}') {
            self.err(&p(&ip, &["name"]), format!("`{n}` is not a file name: the inventory lies directly in the dataset's root, its name a constant (V22)"));
        } else if n.starts_with(RESERVED_PREFIX) {
            self.err(&p(&ip, &["name"]), format!("`{n}` begins with `{RESERVED_PREFIX}`, reserved for the manifest (V13)"));
        } else if glob_component(&inventory_sibling(d), n) {
            self.err(&p(&ip, &["name"]), format!("`{n}` is a name the pattern `{}` can produce in the dataset's root (V22)", d.pattern));
        }
    }

    fn err(&mut self, path: &P, msg: impl AsRef<str>) {
        self.errors.push(format!("{}: {}: {}", self.ast.name, path.join("/"), msg.as_ref()));
    }

    /// V26: a channel is the actor's, shared by its sub-actors, so one declared in a sub-actor
    /// would be declared by each of them, and which declaration replaced which depends on timing.
    fn declared_outside_forks(&mut self, kind: &str, pp: &P, st: &ActorState<'a>) {
        if st.fork_opens.is_some() {
            self.err(pp, format!("a `{kind}` inside a `parallel` or `loader` body: its channel is the actor's, shared by every sub-actor, so each sub-actor would declare it again over the others' and what it holds would depend on timing; declare it before the fork (V26)"));
        }
    }

    /// V26: a loader's workers run beside the actor, so a channel they put or take on that the
    /// actor declares again while they run (in a loop the loader is not declared again in, or
    /// at a second site) would be the old one or the new one as timing had it.
    fn used_by_workers(&mut self, kind: &str, channel: &str, pp: &P, st: &ActorState<'a>) {
        let (Some((lname, lbody)), Some(sites)) = (&st.loader, st.sites.get(channel)) else { return };
        if sites.len() == 1 && (!sites[0].1 || (&sites[0].0 == lbody && st.sites.get(lname).map_or(0, |s| s.len()) == 1)) {
            return;
        }
        self.err(pp, format!("a `{kind}` on channel `{channel}` inside a `loader` body: the loader's workers run beside the actor, so the channel is declared at one site, outside every loop or next to a loader declared at one site; declared again while they run, which declaration a worker reached would depend on timing (V26)"));
    }

    fn run(&mut self) {
        let ast = self.ast;
        if !is_ident(&ast.name) {
            self.err(&vec!["name".into()], "not an identifier");
        }
        let api = ast.posix.as_deref().map(|a| (a, crate::backend::Api::parse(a)));
        let cache = ast.cache.as_deref().map(|c| (c, crate::backend::Cache::parse(c)));
        if let Some((a, None)) = api {
            self.err(&vec!["posix".into()], format!("`{a}` is not one of {}", crate::backend::API_NAMES));
        }
        if let Some(s) = ast.s3.as_deref() {
            if crate::backend::S3Api::parse(s).is_none() {
                self.err(&vec!["s3".into()], format!("`{s}` is not one of {}", crate::backend::S3_NAMES));
            }
        }
        if let Some((c, None)) = cache {
            self.err(&vec!["cache".into()], format!("`{c}` is not one of {}", crate::backend::CACHE_NAMES));
        }
        if let (Some((_, Some(a))), Some((_, Some(c)))) = (api, cache) {
            if crate::backend::BackendKind::of(a, c).is_none() {
                self.err(&vec!["cache".into()], "`direct` under `mmap`: its reads are page faults on a mapping");
            }
        }
        if ast.params.contains_key("gpus") {
            self.err(&vec!["params".into()], "`gpus` is reserved and set by the runner");
        }
        if ast.actors.is_empty() {
            self.err(&vec!["actors".into()], "at least one actor");
        }
        for (pname, param) in &ast.params {
            let path = vec!["params".into(), pname.clone()];
            if !is_ident(pname) {
                self.err(&path, "not an identifier");
            }
            if let Some(u) = &param.unit {
                if !["bytes", "ns", "count", "ratio", "tokens", "none"].contains(&u.as_str()) {
                    self.err(&p(&path, &["unit"]), format!("unknown unit `{u}`"));
                }
            }
            // on a run, the value in effect: a given distribution is checked as the default is
            self.pvalue(self.given.and_then(|g| g.get(pname.as_str())).unwrap_or(&param.default), &path);
        }
        let mut roots: BTreeMap<String, &str> = BTreeMap::new();
        for (dname, d) in &ast.datasets {
            let path = vec!["datasets".into(), dname.clone()];
            if !is_ident(dname) {
                self.err(&path, "not an identifier");
            }
            let scope = Scope::new(HashSet::new());
            let mut st = ActorState::default();
            self.layout_fields(d, &path);
            self.inventory(d, &path);
            let f = d;
            self.distref(&f.size, &p(&path, &["size"]), &scope, &mut st);
            if f.layout != Layout::Array {
                self.size_not_negative(&f.size, &p(&path, &["size"]));
            }
            self.expr(&f.count, &p(&path, &["count"]), &scope, &mut st, None);
            if let Some(e) = &f.samples_per_file {
                self.expr(e, &p(&path, &["samples_per_file"]), &scope, &mut st, None);
            }
            if let Some(e) = &f.chunk {
                self.expr(e, &p(&path, &["chunk"]), &scope, &mut st, None);
            }
            if let Some(e) = &f.slot {
                self.expr(e, &p(&path, &["slot"]), &scope, &mut st, None);
            }
            if let Some(i) = &f.inventory {
                self.expr(&i.entry, &p(&path, &["inventory", "entry"]), &scope, &mut st, None);
            }
            if let Some(fm) = &f.format {
                if !is_ident(&fm.class) {
                    self.err(&p(&path, &["format", "class"]), "not an identifier");
                }
                if let Some(l) = &fm.framing {
                    let lp = p(&path, &["format", "framing"]);
                    for (k, e) in [
                        ("unit", &l.unit),
                        ("file_header", &l.file_header),
                        ("file_header_per_sample", &l.file_header_per_sample),
                        ("file_header_per_unit", &l.file_header_per_unit),
                        ("file_footer", &l.file_footer),
                        ("file_footer_per_sample", &l.file_footer_per_sample),
                        ("file_footer_per_unit", &l.file_footer_per_unit),
                        ("file_align", &l.file_align),
                        ("unit_header", &l.unit_header),
                        ("unit_header_per_sample", &l.unit_header_per_sample),
                        ("unit_footer", &l.unit_footer),
                        ("unit_footer_per_sample", &l.unit_footer_per_sample),
                        ("unit_align", &l.unit_align),
                    ] {
                        if let Some(e) = e {
                            self.expr(e, &p(&lp, &[k]), &scope, &mut st, None);
                        }
                    }
                    if let Some(cols) = &l.columns {
                        if cols.is_empty() {
                            self.err(&p(&lp, &["columns"]), "at least one column");
                        }
                        let total: f64 = cols.iter().map(|c| c.weight).sum();
                        if cols.iter().any(|c| c.weight < 0.0) {
                            self.err(&p(&lp, &["columns"]), "negative weight");
                        } else if !cols.is_empty() && (total - 1.0).abs() > 1e-9 {
                            self.err(&p(&lp, &["columns"]), format!("weights sum to {total}, not 1: the sample's bytes must land somewhere once"));
                        }
                        for (i, c) in cols.iter().enumerate() {
                            let cp = p(&lp, &["columns", &i.to_string()]);
                            for (k, e) in [("header", &c.header), ("fixed", &c.fixed), ("row_header", &c.row_header), ("row_footer", &c.row_footer), ("row_align", &c.row_align), ("align", &c.align)] {
                                if let Some(e) = e {
                                    self.expr(e, &p(&cp, &[k]), &scope, &mut st, None);
                                }
                            }
                        }
                    }
                }
            }
            if f.pattern.is_empty() {
                self.err(&p(&path, &["pattern"]), "empty pattern");
            }
            self.reserved(&f.pattern, &path);
            let root = dataset_root(d);
            if let Some(other) = roots.get(&root) {
                self.err(&path, format!("shares root `{root}/` with dataset `{other}` (V13)"));
            }
            for (other, owner) in &roots {
                let (inner, outer) = if root.len() > other.len() { (&root, other) } else { (other, &root) };
                if inner != outer && (outer.is_empty() || inner.starts_with(&format!("{outer}/"))) {
                    self.err(&path, format!("root `{root}/` and root `{other}/` of dataset `{owner}` are nested (V13)"));
                }
            }
            roots.insert(root, dname);
        }
        let mut nroots: HashMap<String, (&str, bool, crate::ast::Protocol)> = HashMap::new();
        for (nname, n) in &ast.namespaces {
            let path = vec!["namespaces".into(), nname.clone()];
            if !is_ident(nname) {
                self.err(&path, "not an identifier");
            }
            if n.fields.is_empty() {
                self.err(&p(&path, &["fields"]), "a namespace declares at least one field");
            }
            for f in n.fields.keys() {
                if !is_ident(f) {
                    self.err(&p(&path, &["fields", f]), "not an identifier");
                }
            }
            if let NsSize::Expr(e) = &n.size {
                let scope = Scope::new(n.fields.keys().map(|s| s.as_str()).collect());
                let mut st = ActorState::default();
                self.expr(e, &p(&path, &["size"]), &scope, &mut st, None);
            }
            if n.pattern.is_empty() {
                self.err(&p(&path, &["pattern"]), "empty pattern");
            }
            self.reserved(&n.pattern, &path);
            let nroot = namespace_root(&n.pattern);
            for (root, dname) in &roots {
                if !root.is_empty() && (&nroot == root || nroot.starts_with(&format!("{root}/"))) {
                    self.err(&path, format!("root `{nroot}/` lies inside dataset `{dname}`'s root `{root}/` (V13)"));
                }
            }
            // V14: namespaces sharing a root share its manifest, so they agree on `input`
            let inp = n.input.unwrap_or(false);
            let proto = n.protocol.unwrap_or_default();
            if let Some((other, oinp, oproto)) = nroots.get(&nroot) {
                if *oinp != inp {
                    self.err(&path, format!("shares root `{nroot}/` with namespace `{other}` but `input` differs (V14)"));
                }
                // V17: and a root is in one place, so they agree on `protocol`
                if *oproto != proto {
                    self.err(&path, format!("shares root `{nroot}/` with namespace `{other}` but `protocol` differs (V17)"));
                }
            } else {
                nroots.insert(nroot, (nname.as_str(), inp, proto));
            }
            // V15: `same_run` compares this run with the writer's manifest, which only an input has
            if n.same_run.unwrap_or(false) && !inp {
                self.err(&path, "`same_run` without `input`: only an input namespace has a writer to compare with (V15)".to_string());
            }
        }
        for (aname, a) in &ast.actors {
            let path = vec!["actors".into(), aname.clone()];
            if !is_ident(aname) {
                self.err(&path, "not an identifier");
            }
            let scope = Scope::new(HashSet::new());
            let mut st = ActorState::default();
            if let Some(c) = &a.count {
                self.expr(c, &p(&path, &["count"]), &scope, &mut st, None);
            }
            channel_sites(&a.body, &p(&path, &["body"]), false, &mut st.sites);
            self.body(&a.body, &p(&path, &["body"]), &scope, &mut st);
        }
        self.v6_one_consume();
        self.v19();
    }

    /// V6: a dataset is consumed in one place. A `consume` has no site: its sample is a
    /// function of the actor's number and the ordinals of its enclosing frames alone
    /// (`runner/REFERENCE.md` §2), so two `consume`s of one dataset (in one body, in sibling
    /// loops, or in two actors) meet at the same positions and return the same samples.
    /// A `consume` under a loop whose count varies with an enclosing loop repeats positions
    /// alike: the ordinal assumes fixed counts.
    fn v6_one_consume(&mut self) {
        let consumes = std::mem::take(&mut self.consumes);
        for (ds, paths) in &consumes {
            for (i, (path, varies)) in paths.iter().enumerate() {
                if let Some(lp) = varies {
                    self.err(path, format!("a `consume` of `{ds}` under a loop whose count varies with an enclosing loop ({}): its position is the row-major ordinal of the enclosing loops at fixed counts, so a count that varies repeats positions and draws the same samples, a reuse the job does not have; give the loop a fixed count (the most it can be, with a `cond`, the builder's `when`, inside to skip the rest) (V6)", lp.join("/")));
                }
                if i > 0 {
                    self.err(path, format!("a second `consume` of `{ds}` (the first is at {}): `consume` has no site, so its sample depends only on the actor and the enclosing loop indices, and two of them draw the same samples, a reuse the job does not have; consume once and use the handle in each place, or put the one `consume` in an inner loop so each draw has its own position (V6)", paths[0].0.join("/")));
                }
            }
        }
    }

    /// V6: whether an expression can take another value at another iteration of an enclosing
    /// frame: it names a loop index (or `at`), a binding that varies, or draws inside a frame
    /// (a draw is keyed on the enclosing indices). Parameters and the actor number are fixed
    /// for an actor's walk. As `check.py`'s `varies`.
    fn varies(&self, e: &Expr, scope: &Scope<'a>) -> bool {
        let n = match e {
            Expr::Lit(_) => return false,
            Expr::Node(n) => n,
        };
        match &**n {
            ExprNode::Index(_) | ExprNode::At { .. } => true,
            ExprNode::Ref(name) => scope.is_varying(name),
            ExprNode::Draw(_) => !scope.indices.is_empty(),
            ExprNode::Param(_) | ExprNode::Actor(_) | ExprNode::Len(_) | ExprNode::Sum(_) | ExprNode::Count(_) | ExprNode::Dirs(_) => false,
            ExprNode::Elem { index, .. } => self.varies(index, scope),
            ExprNode::Size(h) | ExprNode::Offset(h) | ExprNode::UnitIndex(h) | ExprNode::Units(h) | ExprNode::Chunks(h) => self.handle_varies(h, scope),
            ExprNode::Add(x) | ExprNode::Sub(x) | ExprNode::Mul(x) | ExprNode::Div(x) | ExprNode::Mod(x) | ExprNode::CeilDiv(x) | ExprNode::Min(x) | ExprNode::Max(x)
            | ExprNode::Eq(x) | ExprNode::Ne(x) | ExprNode::Lt(x) | ExprNode::Le(x) | ExprNode::Gt(x) | ExprNode::Ge(x) => x.iter().any(|e| self.varies(e, scope)),
            ExprNode::And(x) | ExprNode::Or(x) => x.iter().any(|e| self.varies(e, scope)),
            ExprNode::Neg(e) | ExprNode::Not(e) => self.varies(e, scope),
            ExprNode::Cond(c) => self.varies(&c.test, scope) || self.exprlike_varies(&c.then, scope) || self.exprlike_varies(&c.otherwise, scope),
        }
    }

    fn exprlike_varies(&self, v: &ExprLike, scope: &Scope<'a>) -> bool {
        match v {
            ExprLike::Expr(e) => self.varies(e, scope),
            ExprLike::Dist(Dist::Const(e)) => self.varies(e, scope),
            ExprLike::Dist(_) => !scope.indices.is_empty(),
            ExprLike::Handle(h) => self.handle_varies(h, scope),
        }
    }

    fn handle_varies(&self, h: &Handle, scope: &Scope<'a>) -> bool {
        let opt = |e: &Option<Expr>| e.as_ref().map_or(false, |e| self.varies(e, scope));
        match h {
            Handle::Ref(name) => scope.is_varying(name),
            Handle::File(f) => opt(&f.id) || f.of.as_ref().map_or(false, |h| self.handle_varies(h, scope)) || opt(&f.chunk),
            Handle::Dir { id, .. } => self.varies(id, scope),
            Handle::Object { fields, .. } => fields.values().any(|e| self.varies(e, scope)),
            Handle::Consume(_) | Handle::Pick(_) => !scope.indices.is_empty(),
            Handle::Inventory(_) => false,
            Handle::Unit(u) => self.handle_varies(&u.of, scope) || opt(&u.index),
            Handle::Column(c) => self.handle_varies(&c.of, scope) || self.varies(&c.index, scope),
        }
    }

    /// V19: the abstract declares an API only for a protocol it has names on, and an abstract
    /// with both declares APIs of one scheduling (event-driven or a thread per actor),
    /// defaults counted (`DESIGN_REVIEW.md` §3.65).
    fn v19(&mut self) {
        let ast = self.ast;
        let used = protocols(ast, self.traces);
        if ast.s3.is_some() && !used.s3 {
            self.err(&vec!["s3".into()], "declared, but no dataset or namespace is `protocol: s3` (V19)");
        }
        for (key, v) in [("posix", &ast.posix), ("cache", &ast.cache)] {
            if v.is_some() && !used.posix {
                self.err(&vec![key.into()], "declared, but every dataset and namespace is `protocol: s3` (V19)");
            }
        }
        if used.posix && used.s3 {
            let (Ok(b), Ok(s)) = (crate::backend::declared(ast), crate::backend::declared_s3(ast)) else { return };
            if b.event_loop() != s.event_loop() {
                self.err(
                    &vec![if ast.s3.is_some() { "s3" } else { "posix" }.into()],
                    format!("posix `{}` and s3 `{}`: one is event-driven and the other a thread per actor; an abstract with both protocols declares APIs of one kind (V19)", b.api.name(), s.name()),
                );
            }
        }
    }

    fn reserved(&mut self, pattern: &str, path: &P) {
        for comp in pattern.split('/') {
            if comp.starts_with(RESERVED_PREFIX) {
                self.err(path, format!("path component `{comp}` begins with `{RESERVED_PREFIX}`, reserved for the manifest (V13)"));
            }
        }
        self.divisors(pattern, &p(path, &["pattern"]));
    }

    /// V23 (contract 0.10, `DESIGN_REVIEW.md` §3.75): a divisor named in a pattern
    /// (`{id div files_per_directory}`) is a declared parameter whose default is an integer.
    fn divisors(&mut self, pattern: &str, path: &P) {
        let pt = match crate::pattern::Pattern::parse(pattern) {
            Ok(pt) => pt,
            Err(e) => return self.err(path, e.to_string()),
        };
        for name in pt.params() {
            match self.ast.params.get(name).map(|p| &p.default) {
                Some(PValue::Scalar(Literal::Int(_))) => {}
                Some(_) => self.err(path, format!("divisor `{name}` is a parameter whose default is not an integer (V23)")),
                None => self.err(path, format!("divisor `{name}` is not a declared parameter (V23)")),
            }
        }
    }

    fn pvalue(&mut self, v: &'a PValue, path: &P) {
        match v {
            PValue::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    self.pvalue(x, &p(path, &[&i.to_string()]));
                }
            }
            PValue::Dist(d) => {
                let scope = Scope::new(HashSet::new());
                let mut st = ActorState::default();
                self.dist(d, path, &scope, &mut st);
            }
            PValue::Scalar(_) => {}
        }
    }

    // ---- statements ----

    fn body(&mut self, nodes: &'a [Node], path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>) {
        let mut scope = scope.child();
        scope.forward = HashSet::new();
        for n in nodes {
            if let Node::Let { name, .. } = n {
                if !scope.forward.insert(name.as_str()) {
                    scope.twice.insert(name.as_str());
                }
            }
        }
        for (i, node) in nodes.iter().enumerate() {
            let kind = node.kind();
            let pp = p(path, &[&i.to_string(), kind]);
            if !node.is_control() {
                *self.ops.entry(kind).or_insert(0) += 1;
            }
            self.node(node, &pp, &mut scope, st);
        }
    }

    fn node(&mut self, node: &'a Node, pp: &P, scope: &mut Scope<'a>, st: &mut ActorState<'a>) {
        match node {
            Node::Let { name, value } => {
                if !is_ident(name) {
                    self.err(&p(pp, &["name"]), "not an identifier");
                }
                self.exprlike(value, &p(pp, &["value"]), scope, st, Some(name));
                let varies = self.exprlike_varies(value, scope);
                scope.bind(name, value, varies);
            }
            Node::Loop { index, from, to, step, body } => {
                if !is_ident(index) {
                    self.err(&p(pp, &["index"]), "not an identifier");
                }
                if let Some(e) = from {
                    self.expr(e, &p(pp, &["from"]), scope, st, None);
                }
                self.expr(to, &p(pp, &["to"]), scope, st, None);
                if let Some(e) = step {
                    self.expr(e, &p(pp, &["step"]), scope, st, None);
                }
                let nonneg = from.as_ref().map_or(true, |e| self.at_least(e, 0, scope));
                let varies = [from.as_ref(), Some(to), step.as_ref()].into_iter().flatten().any(|e| self.varies(e, scope));
                let inner = scope.with_index(index, nonneg, varies.then(|| pp.clone()));
                self.body(body, &p(pp, &["body"]), &inner, st);
            }
            Node::Parallel { index, width, body } => {
                if !is_ident(index) {
                    self.err(&p(pp, &["index"]), "not an identifier");
                }
                self.expr(width, &p(pp, &["width"]), scope, st, None);
                let inner = scope.with_index(index, true, self.varies(width, scope).then(|| pp.clone()));
                let outer = st.fork_opens.replace(Vec::new());
                self.body(body, &p(pp, &["body"]), &inner, st);
                st.fork_opens = outer;
            }
            Node::Loader { name, index, workers, prefetch, batches, body, .. } => {
                if !is_ident(name) {
                    self.err(&p(pp, &["name"]), "not an identifier");
                }
                if !is_ident(index) {
                    self.err(&p(pp, &["index"]), "not an identifier");
                }
                self.expr(workers, &p(pp, &["workers"]), scope, st, None);
                self.expr(prefetch, &p(pp, &["prefetch"]), scope, st, None);
                self.expr(batches, &p(pp, &["batches"]), scope, st, None);
                self.declared_outside_forks("loader", pp, st);
                st.channels.insert(name);
                let inner = scope.with_index(index, true, self.varies(batches, scope).then(|| pp.clone()));
                let outer = st.fork_opens.replace(Vec::new());
                let outer_loader = st.loader.replace((name.as_str(), pp[..pp.len() - 2].to_vec()));
                self.body(body, &p(pp, &["body"]), &inner, st);
                st.fork_opens = outer;
                st.loader = outer_loader;
            }
            Node::Channel { name, capacity, .. } => {
                if !is_ident(name) {
                    self.err(&p(pp, &["name"]), "not an identifier");
                }
                self.expr(capacity, &p(pp, &["capacity"]), scope, st, None);
                self.declared_outside_forks("channel", pp, st);
                st.channels.insert(name);
            }
            Node::Put { channel, seq } => {
                if !st.channels.contains(channel.as_str()) {
                    self.err(pp, format!("channel `{channel}` not declared"));
                }
                self.used_by_workers("put", channel, pp, st);
                self.expr(seq, &p(pp, &["seq"]), scope, st, None);
            }
            Node::Take { channel } => {
                if !st.channels.contains(channel.as_str()) {
                    self.err(pp, format!("channel `{channel}` not declared"));
                }
                self.used_by_workers("take", channel, pp, st);
            }
            Node::Barrier { scope: s } => {
                if s != "global" && s != "host" && !is_ident(s) {
                    self.err(&p(pp, &["scope"]), "not `global`, `host`, or an identifier");
                }
            }
            Node::Compute { ns } => self.expr(ns, &p(pp, &["ns"]), scope, st, None),
            Node::Cond { test, then, otherwise } => {
                self.expr(test, &p(pp, &["if"]), scope, st, None);
                self.body(then, &p(pp, &["then"]), scope, st);
                if let Some(b) = otherwise {
                    self.body(b, &p(pp, &["else"]), scope, st);
                }
            }
            Node::Choose { arms, .. } => {
                if arms.is_empty() {
                    self.err(&p(pp, &["arms"]), "at least one arm");
                }
                for (i, arm) in arms.iter().enumerate() {
                    if !(arm.weight >= 0.0) {
                        self.err(&p(pp, &["arms", &i.to_string(), "weight"]), "negative weight");
                    }
                    self.body(&arm.body, &p(pp, &["arms", &i.to_string(), "body"]), scope, st);
                }
            }
            Node::Phase { name, body } => {
                self.expr(name, &p(pp, &["name"]), scope, st, None);
                self.body(body, &p(pp, &["body"]), scope, st);
            }
            Node::Trace { sha256, .. } => {
                self.traces = true;
                if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                    self.err(&p(pp, &["sha256"]), "not a lowercase hex sha256");
                }
            }
            _ => self.op(node, pp, scope, st),
        }
    }

    fn op(&mut self, node: &'a Node, pp: &P, scope: &mut Scope<'a>, st: &mut ActorState<'a>) {
        let kind = node.kind();
        // handles
        let mut handles: Vec<(&str, &'a Handle)> = Vec::new();
        let mut exprs: Vec<(&str, &'a Expr)> = Vec::new();
        let mut expect: Option<&'a Expect> = None;
        match node {
            Node::Open { file, flags, mode, expect: ex } => {
                handles.push(("file", file));
                expect = Some(ex);
                if flags.is_empty() {
                    self.err(&p(pp, &["flags"]), "at least one flag");
                }
                let mut seen = HashSet::new();
                for f in flags {
                    if !seen.insert(f) {
                        self.err(&p(pp, &["flags"]), format!("duplicate flag {f:?}"));
                    }
                }
                if let Some(m) = mode {
                    if *m > 4095 {
                        self.err(&p(pp, &["mode"]), "mode > 0o7777");
                    }
                }
            }
            Node::Close(f) | Node::Fstat(f) | Node::Stat(f) | Node::Fsync(f) | Node::Fdatasync(f) | Node::Unlink(f) => {
                handles.push(("file", &f.file));
                expect = Some(&f.expect);
            }
            Node::Read { file, len, offset, repeat, expect: ex } => {
                handles.push(("file", file));
                exprs.push(("len", len));
                if let Some(e) = offset {
                    exprs.push(("offset", e));
                }
                if let Some(Repeat::Count(e)) = repeat {
                    exprs.push(("repeat", e));
                }
                expect = Some(ex);
            }
            Node::Write { file, len, offset, repeat, expect: ex } => {
                handles.push(("file", file));
                exprs.push(("len", len));
                if let Some(e) = offset {
                    exprs.push(("offset", e));
                }
                if let Some(e) = repeat {
                    exprs.push(("repeat", e));
                }
                expect = Some(ex);
            }
            Node::Lseek { file, offset, .. } => {
                handles.push(("file", file));
                exprs.push(("offset", offset));
            }
            Node::Ioctl { file, expect: ex, .. } => {
                handles.push(("file", file));
                expect = Some(ex);
            }
            Node::Fadvise { file, offset, len, expect: ex, .. } => {
                handles.push(("file", file));
                if let Some(e) = offset {
                    exprs.push(("offset", e));
                }
                if let Some(e) = len {
                    exprs.push(("len", e));
                }
                expect = Some(ex);
            }
            Node::Ftruncate { file, len, expect: ex } => {
                handles.push(("file", file));
                exprs.push(("len", len));
                expect = Some(ex);
            }
            Node::Fallocate { file, offset, len, expect: ex } => {
                handles.push(("file", file));
                if let Some(e) = offset {
                    exprs.push(("offset", e));
                }
                exprs.push(("len", len));
                expect = Some(ex);
            }
            Node::Mkdir { dir, mode, expect: ex } => {
                handles.push(("dir", dir));
                expect = Some(ex);
                if let Some(m) = mode {
                    if *m > 4095 {
                        self.err(&p(pp, &["mode"]), "mode > 0o7777");
                    }
                }
            }
            Node::Rmdir { dir, expect: ex } | Node::Readdir { dir, expect: ex, .. } => {
                handles.push(("dir", dir));
                expect = Some(ex);
            }
            Node::Rename { from, to, expect: ex } => {
                handles.push(("from", from));
                handles.push(("to", to));
                expect = Some(ex);
            }
            _ => unreachable!("control node in op()"),
        }
        if let Some(Some(list)) = expect {
            if list.is_empty() {
                self.err(&p(pp, &["expect"]), "empty expect list");
            }
            let mut seen = HashSet::new();
            for e in list {
                if !is_errno(e) {
                    self.err(&p(pp, &["expect"]), format!("`{e}` is not an errno name"));
                }
                if !seen.insert(e) {
                    self.err(&p(pp, &["expect"]), format!("duplicate errno `{e}`"));
                }
            }
        }
        for (key, h) in &handles {
            self.handle(h, &p(pp, &[key]), scope, st);
        }
        for (key, e) in &exprs {
            self.expr(e, &p(pp, &[key]), scope, st, None);
        }
        if let Node::Write { file, .. } = node {
            self.note_write(file, scope, st);
        }
        // V24: a sub-actor shares the descriptors it inherits, and with them the kernel's file
        // position, so an op at that position on one is timing-dependent
        if let Some(opened) = &mut st.fork_opens {
            let unpositioned = match node {
                Node::Open { file, .. } => {
                    opened.push(format!("{file:?}"));
                    None
                }
                Node::Read { file, offset: None, .. } => Some(("read", file)),
                Node::Write { file, offset: None, .. } => Some(("write", file)),
                Node::Lseek { file, .. } => Some(("lseek", file)),
                _ => None,
            };
            if let Some((what, file)) = unpositioned {
                if !opened.contains(&format!("{file:?}")) {
                    self.err(&p(pp, &["file"]), format!("{what} that uses the file position of a handle this `parallel` or `loader` body did not open: its sub-actors share the descriptor and its position, so the offsets would depend on timing; give an `offset`, or open the handle in the body (V24)"));
                }
            }
        }
        if let Node::Read { file, repeat: Some(Repeat::UntilEof(_)), .. } = node {
            self.check_until_eof(file, pp, scope, st);
        }
        // V12: datasets are read-only
        if matches!(node, Node::Write { .. } | Node::Ftruncate { .. } | Node::Fallocate { .. } | Node::Unlink(_) | Node::Rename { .. } | Node::Mkdir { .. } | Node::Rmdir { .. }) {
            for (key, h) in &handles {
                if let Some(ds) = self.dataset_of(h, scope) {
                    self.err(&p(pp, &[key]), format!("{kind} on dataset `{ds}`: datasets are read-only (V12)"));
                }
            }
        }
        if let Node::Open { file, flags, .. } = node {
            let writes = flags.iter().any(|f| matches!(f, OpenFlag::WRONLY | OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::TRUNC | OpenFlag::APPEND));
            if writes {
                if let Some(ds) = self.dataset_of(file, scope) {
                    self.err(&p(pp, &["flags"]), format!("open for writing on dataset `{ds}`: datasets are read-only (V12)"));
                }
                if let Some(ns) = self.input_namespace_of(file, scope) {
                    self.err(&p(pp, &["flags"]), format!("open for writing on input namespace `{ns}`: input namespaces are read-only (V14)"));
                }
            }
        }
        // V14: input namespaces are read-only
        if matches!(node, Node::Write { .. } | Node::Ftruncate { .. } | Node::Fallocate { .. } | Node::Unlink(_) | Node::Rename { .. } | Node::Mkdir { .. } | Node::Rmdir { .. }) {
            for (key, h) in &handles {
                if let Some(ns) = self.input_namespace_of(h, scope) {
                    self.err(&p(pp, &[key]), format!("{kind} on input namespace `{ns}`: input namespaces are read-only (V14)"));
                }
            }
        }
        // V18: an object of a namespace in an object store is written once, in order, by one
        // upload; the forms that would rewrite, extend, or read it in place have no object form
        let v18 = match node {
            Node::Open { file, flags, .. } => flags.iter().find(|f| matches!(f, OpenFlag::APPEND | OpenFlag::RDWR | OpenFlag::EXCL)).map(|f| (file, format!("open with {f:?}"), "flags")),
            Node::Ftruncate { file, .. } => Some((file, "ftruncate".to_string(), "file")),
            Node::Fallocate { file, .. } => Some((file, "fallocate".to_string(), "file")),
            _ => None,
        };
        if let Some((h, what, key)) = v18 {
            if let Some(ns) = self.object_store_namespace(h, scope) {
                self.err(&p(pp, &[key]), format!("{what} on namespace `{ns}`, declared `protocol: s3`: an object is written once, in order from 0, and has no {what} (V18)"));
            }
        }
    }

    /// Namespace name if the handle is (a binding to) an object of a namespace declared
    /// `protocol: s3`.
    fn object_store_namespace(&self, h: &'a Handle, scope: &Scope<'a>) -> Option<&'a str> {
        let ns = self.object_namespace(h, scope)?;
        if self.ast.namespaces.get(ns).and_then(|n| n.protocol) == Some(Protocol::S3) { Some(ns) } else { None }
    }

    /// Namespace name if the handle is (a binding to) an object of an `input` namespace.
    fn input_namespace_of(&self, h: &'a Handle, scope: &Scope<'a>) -> Option<&'a str> {
        let ns = self.object_namespace(h, scope)?;
        if self.ast.namespaces.get(ns).and_then(|n| n.input).unwrap_or(false) { Some(ns) } else { None }
    }

    /// Dataset name if the handle is (a binding to) a dataset file, dir, sample, region, or chunk.
    fn dataset_of(&self, h: &'a Handle, scope: &Scope<'a>) -> Option<&'a str> {
        match h {
            Handle::Ref(name) => match scope.lookup(name) {
                Some(ExprLike::Handle(v)) => self.dataset_of(v, scope),
                _ => None,
            },
            Handle::File(f) => match (&f.dataset, &f.of) {
                (Some(ds), _) => Some(ds.as_str()),
                (None, Some(of)) => self.dataset_of(of, scope),
                _ => None,
            },
            Handle::Dir { dataset, .. } => Some(dataset.as_str()),
            Handle::Consume(ds) | Handle::Inventory(ds) => Some(ds.as_str()),
            Handle::Pick(pk) => Some(pk.dataset.as_str()),
            Handle::Unit(u) => self.dataset_of(&u.of, scope),
            Handle::Column(c) => self.dataset_of(&c.of, scope),
            Handle::Object { .. } => None,
        }
    }

    fn object_namespace(&self, h: &'a Handle, scope: &Scope<'a>) -> Option<&'a str> {
        match h {
            Handle::Ref(name) => match scope.lookup(name) {
                Some(ExprLike::Handle(v)) => self.object_namespace(v, scope),
                _ => None,
            },
            Handle::Object { namespace, .. } => Some(namespace.as_str()),
            _ => None,
        }
    }

    fn note_write(&mut self, h: &'a Handle, scope: &Scope<'a>, st: &mut ActorState<'a>) {
        if let Handle::Ref(name) = h {
            if self.object_namespace(h, scope).is_some() {
                st.written.insert(name.as_str());
            }
        }
    }

    fn check_until_eof(&mut self, h: &'a Handle, pp: &P, scope: &Scope<'a>, st: &ActorState<'a>) {
        if let Some(ns) = self.object_namespace(h, scope) {
            if matches!(self.ast.namespaces.get(ns).map(|n| &n.size), Some(NsSize::AsWritten(_))) {
                let ok = matches!(h, Handle::Ref(name) if st.written.contains(name.as_str()));
                if !ok {
                    self.err(pp, "until_eof on an as_written object needs the handle binding the creating writes used (same position)");
                }
            }
        }
    }

    // ---- expressions ----

    fn exprlike(&mut self, v: &'a ExprLike, path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>, binding: Option<&'a str>) {
        match v {
            ExprLike::Handle(h) => self.handle(h, path, scope, st),
            ExprLike::Dist(d) => self.dist(d, path, scope, st),
            ExprLike::Expr(e) => self.expr(e, path, scope, st, binding),
        }
    }

    fn dist(&mut self, d: &'a Dist, path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>) {
        match d {
            Dist::Empirical { values, weights } => {
                if values.is_empty() {
                    self.err(path, "empirical: no values");
                }
                if values.len() != weights.len() {
                    self.err(path, "empirical: values and weights differ in length");
                }
                if weights.iter().any(|w| !(*w >= 0.0)) {
                    self.err(path, "empirical: negative weight");
                }
            }
            Dist::Mixture(arms) => {
                if arms.is_empty() {
                    self.err(path, "mixture: no arms");
                }
                for (i, arm) in arms.iter().enumerate() {
                    if !(arm.weight >= 0.0) {
                        self.err(&p(path, &[&i.to_string(), "weight"]), "negative weight");
                    }
                    if let Some(d) = &arm.dist {
                        self.distref(d, &p(path, &[&i.to_string(), "dist"]), scope, st);
                    }
                }
            }
            Dist::Const(e) => self.expr(e, path, scope, st, None),
            Dist::Uniform { lo, hi } => {
                self.expr(lo, &p(path, &["lo"]), scope, st, None);
                self.expr(hi, &p(path, &["hi"]), scope, st, None);
            }
            Dist::Uniform64 {} => {}
            Dist::Normal { mean, sd, min, max } => {
                self.expr(mean, &p(path, &["mean"]), scope, st, None);
                self.expr(sd, &p(path, &["sd"]), scope, st, None);
                if let Some(e) = min {
                    self.expr(e, &p(path, &["min"]), scope, st, None);
                }
                if let Some(e) = max {
                    self.expr(e, &p(path, &["max"]), scope, st, None);
                }
                self.bounds("normal", min.as_deref(), max.as_deref(), path);
            }
            Dist::Lognormal { median, sigma, min, max } => {
                if !(*sigma > 0.0) {
                    self.err(&p(path, &["sigma"]), "sigma must be > 0");
                }
                self.expr(median, &p(path, &["median"]), scope, st, None);
                if let Some(e) = min {
                    self.expr(e, &p(path, &["min"]), scope, st, None);
                }
                if let Some(e) = max {
                    self.expr(e, &p(path, &["max"]), scope, st, None);
                }
                self.bounds("lognormal", min.as_deref(), max.as_deref(), path);
            }
            Dist::Zipf { s } => {
                if !(*s > 0.0) {
                    self.err(&p(path, &["s"]), "s must be > 0");
                }
            }
            Dist::Hotset { fraction, weight } => {
                if !(*fraction > 0.0 && *fraction <= 1.0) {
                    self.err(&p(path, &["fraction"]), "fraction must be in (0, 1]");
                }
                if !(*weight >= 0.0 && *weight <= 1.0) {
                    self.err(&p(path, &["weight"]), "weight must be in [0, 1]");
                }
            }
        }
    }

    fn distref(&mut self, d: &'a DistRef, path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>) {
        match d {
            DistRef::Dist(d) => self.dist(d, path, scope, st),
            DistRef::Expr(e) => self.expr(e, path, scope, st, None),
        }
    }

    fn expr(&mut self, e: &'a Expr, path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>, binding: Option<&'a str>) {
        let node = match e {
            Expr::Lit(_) => return,
            Expr::Node(n) => n.as_ref(),
        };
        match node {
            ExprNode::Param(name) => {
                if !self.ast.params.contains_key(name) {
                    self.err(path, format!("unknown param `{name}`"));
                }
            }
            ExprNode::Index(name) => {
                if !scope.indices.contains(&name.as_str()) && !scope.fields.contains(name.as_str()) {
                    self.err(path, format!("index `{name}` not in scope"));
                }
            }
            ExprNode::Ref(name) => {
                if scope.lookup(name).is_none() && !scope.fields.contains(name.as_str()) {
                    self.err(path, format!("binding `{name}` not in scope"));
                }
            }
            ExprNode::At { name, index } => self.at(name, index, path, scope, st, binding),
            ExprNode::Draw(d) => self.distref(&d.dist, path, scope, st),
            ExprNode::Elem { array, index } => {
                if !self.ast.params.contains_key(array) {
                    self.err(path, format!("unknown parameter array `{array}`"));
                }
                self.expr(index, &p(path, &["index"]), scope, st, None);
            }
            ExprNode::Len(name) | ExprNode::Sum(name) => {
                if !self.ast.params.contains_key(name) {
                    self.err(path, format!("unknown parameter array `{name}`"));
                }
            }
            ExprNode::Size(h) | ExprNode::Offset(h) | ExprNode::UnitIndex(h) | ExprNode::Units(h) | ExprNode::Chunks(h) => self.handle(h, path, scope, st),
            ExprNode::Count(name) => {
                if !self.ast.datasets.contains_key(name) {
                    self.err(path, format!("unknown dataset `{name}`"));
                }
            }
            ExprNode::Dirs(name) => {
                if !matches!(self.ast.datasets.get(name), Some(d) if d.layout != Layout::Array) {
                    self.err(path, format!("`dirs` needs a dataset named by ids, not an `array`: got `{name}`"));
                }
            }
            ExprNode::Cond(c) => {
                self.expr(&c.test, &p(path, &["if"]), scope, st, None);
                self.exprlike(&c.then, &p(path, &["then"]), scope, st, binding);
                self.exprlike(&c.otherwise, &p(path, &["else"]), scope, st, binding);
            }
            ExprNode::Add(a) | ExprNode::Sub(a) | ExprNode::Mul(a) | ExprNode::Div(a) | ExprNode::Mod(a) | ExprNode::CeilDiv(a)
            | ExprNode::Min(a) | ExprNode::Max(a) | ExprNode::Eq(a) | ExprNode::Ne(a) | ExprNode::Lt(a) | ExprNode::Le(a)
            | ExprNode::Gt(a) | ExprNode::Ge(a) => {
                for (i, x) in a.iter().enumerate() {
                    self.expr(x, &p(path, &[&i.to_string()]), scope, st, None);
                }
            }
            ExprNode::And(a) | ExprNode::Or(a) => {
                if a.len() < 2 {
                    self.err(path, "and/or need at least two operands");
                }
                for (i, x) in a.iter().enumerate() {
                    self.expr(x, &p(path, &[&i.to_string()]), scope, st, None);
                }
            }
            ExprNode::Neg(x) | ExprNode::Not(x) => self.expr(x, path, scope, st, None),
            ExprNode::Actor(_) => {}
        }
    }

    fn at(&mut self, name: &'a str, idx: &'a Expr, path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>, binding: Option<&'a str>) {
        if scope.lookup(name).is_none() && !scope.forward.contains(name) {
            self.err(path, format!("`at` names unknown binding `{name}`"));
        }
        let Some(loop_index) = scope.indices.last().copied() else {
            self.err(path, "`at` outside any loop");
            return;
        };
        let ok = match idx {
            Expr::Node(n) => match n.as_ref() {
                ExprNode::Sub([a, b]) => {
                    matches!(a, Expr::Node(m) if matches!(m.as_ref(), ExprNode::Index(i) if i == loop_index)) && self.at_least(b, 1, scope)
                }
                _ => false,
            },
            _ => false,
        };
        // a binding of the same loop body: the one being defined, or any `let` of this body
        if (binding == Some(name) || scope.forward.contains(name)) && !ok {
            self.err(path, format!("`{name} @` names a binding of this loop body, so its index must be {{sub: [{{index: {loop_index}}}, e]}} with e provably >= 1"));
        }
        // of two `let`s of one name in this body, which one an `at` reads is not defined
        if scope.twice.contains(name) {
            self.err(path, format!("`{name} @` names a binding this body defines more than once, so which one it reads is not defined"));
        }
        self.expr(idx, &p(path, &["index"]), scope, st, None);
    }

    /// `e` provably ≥ `k`, `k` ∈ {0, 1} (rule V3): a literal ≥ k; a `ref`, `param`, or `draw` whose distribution has
    /// min ≥ k; `add` of a term ≥ k and a term ≥ 0. For k = 0 also any `mod` (Euclidean: in `[0, |b|)`) and an index
    /// of `scope.nonneg_indices`. Judged on the document: parameter defaults, not the values a run was given.
    fn at_least(&self, e: &'a Expr, k: i64, scope: &Scope<'a>) -> bool {
        match e {
            Expr::Lit(Literal::Int(n)) => *n >= k,
            Expr::Lit(Literal::Float(x)) => *x >= k as f64,
            Expr::Lit(_) => false,
            Expr::Node(n) => match n.as_ref() {
                ExprNode::Ref(name) => match scope.lookup(name) {
                    Some(ExprLike::Expr(v)) => self.at_least(v, k, scope),
                    Some(ExprLike::Dist(d)) => self.dist_at_least(d, k, scope),
                    _ => false,
                },
                ExprNode::Param(name) => {
                    let value = self.given.and_then(|g| g.get(name.as_str())).or_else(|| self.ast.params.get(name).map(|p| &p.default));
                    match value {
                        Some(PValue::Dist(d)) => self.dist_at_least(d, k, scope),
                        _ => false,
                    }
                }
                ExprNode::Draw(d) => self.distref_at_least(&d.dist, k, scope),
                ExprNode::Add([a, b]) => {
                    (self.at_least(a, k, scope) && self.at_least(b, 0, scope)) || (self.at_least(a, 0, scope) && self.at_least(b, k, scope))
                }
                ExprNode::Mod(_) if k == 0 => true,
                ExprNode::Index(i) if k == 0 => scope.nonneg_indices.contains(i.as_str()),
                _ => false,
            },
        }
    }

    fn distref_at_least(&self, d: &'a DistRef, k: i64, scope: &Scope<'a>) -> bool {
        match d {
            DistRef::Dist(d) => self.dist_at_least(d, k, scope),
            DistRef::Expr(e) => self.at_least(e, k, scope),
        }
    }

    fn dist_at_least(&self, d: &'a Dist, k: i64, scope: &Scope<'a>) -> bool {
        let lit_ge = |e: &Expr| matches!(e, Expr::Lit(Literal::Int(n)) if *n >= k) || matches!(e, Expr::Lit(Literal::Float(x)) if *x >= k as f64);
        match d {
            Dist::Uniform { lo, .. } => lit_ge(lo),
            // the runner clamps to `min` and then to `max`, so `min` bounds a draw only when no
            // `max` below it can win: none, or one whose value in effect is known and not below
            Dist::Normal { min: Some(m), max, .. } | Dist::Lognormal { min: Some(m), max, .. } => {
                lit_ge(m)
                    && match max {
                        None => true,
                        Some(x) => matches!((self.number(m), self.number(x)), (Some(lo), Some(hi)) if hi >= lo),
                    }
            }
            Dist::Normal { .. } | Dist::Lognormal { .. } => false,
            Dist::Empirical { values, .. } => values.iter().all(|v| match v {
                Literal::Int(n) => *n >= k,
                Literal::Float(x) => *x >= k as f64,
                _ => false,
            }),
            // a `none` arm draws no number, so at least one arm must draw one
            Dist::Mixture(arms) => {
                arms.iter().any(|a| a.dist.is_some()) && arms.iter().all(|a| a.dist.as_ref().map_or(true, |d| self.distref_at_least(d, k, scope)))
            }
            Dist::Const(e) => self.at_least(e, k, scope),
            _ => false,
        }
    }

    /// The number an expression is known to be: a literal, or a `param` whose value in effect
    /// (given, else the default) is a number. Anything else is not known before the run.
    fn number(&self, e: &Expr) -> Option<f64> {
        match self.number_lit(e)? {
            Literal::Int(n) => Some(*n as f64),
            Literal::Float(x) => Some(*x),
            _ => None,
        }
    }

    fn number_lit<'e>(&'e self, e: &'e Expr) -> Option<&'e Literal> {
        let l = match e {
            Expr::Lit(l) => l,
            Expr::Node(n) => match n.as_ref() {
                ExprNode::Param(name) => match self.given.and_then(|g| g.get(name.as_str())).or_else(|| self.ast.params.get(name).map(|p| &p.default)) {
                    Some(PValue::Scalar(l)) => l,
                    _ => return None,
                },
                _ => return None,
            },
        };
        matches!(l, Literal::Int(_) | Literal::Float(_)).then_some(l)
    }

    /// `min` above `max` on a `normal` or `lognormal`: the runner clamps to `min` and then to
    /// `max`, so every draw is `max` and the declared minimum is a lie rule V3 would believe.
    /// Compared as the clamp compares, in f64 (2^53 + 1 is not above 2^53 there); the values
    /// printed as check.py prints them (`1e+20`, `1e-05`).
    fn bounds(&mut self, kind: &str, min: Option<&Expr>, max: Option<&Expr>, path: &P) {
        let text = |l: &Literal| match l {
            Literal::Int(n) => n.to_string(),
            Literal::Float(x) => crate::canon::python_float_repr(*x),
            _ => String::new(),
        };
        let (Some(min), Some(max)) = (min, max) else {
            return;
        };
        let (Some(lo), Some(hi)) = (self.number_lit(min), self.number_lit(max)) else {
            return;
        };
        let (a, b) = (text(lo), text(hi));
        if self.number(min) > self.number(max) {
            self.err(path, format!("{kind}: min {a} is above max {b}"));
        }
    }

    /// Rule V25: a dataset's `size` (not an `array`'s, which the runner clamps to `[0, slot]`)
    /// that can be below 0 where the document shows it: a number known below 0 (rounded as
    /// the runner rounds), or a distribution unbounded below (a `normal` without `min`).
    /// Judged on the parameter values in effect; what cannot be known is left to the
    /// runner, which refuses a negative size when it draws one.
    fn size_not_negative(&mut self, size: &DistRef, path: &P) {
        let least = match size {
            DistRef::Dist(d) => self.least_dist(d),
            DistRef::Expr(e) => self.least_expr(e),
        };
        match least {
            Some(Least::Unbounded) => self.err(path, "a size below 0: the distribution is unbounded below; give it a `min` of 0 or more (V25)"),
            // an overflow (`mul` past f64) or an undefined value (`inf - inf`): no size at all
            Some(Least::Known(v)) if !v.is_finite() => self.err(path, "a size that is not a finite number (V25)"),
            // every digit of the rounded value, as check.py prints it: never saturated at i64
            Some(Least::Known(v)) if v.round() < 0.0 => self.err(path, format!("a size below 0: it can be {:.0} (V25)", v.round())),
            _ => {}
        }
    }

    /// The value an expression is known to have: a number (`number`), or `add`, `sub`, `mul`,
    /// `min`, `max`, `neg` of known values.
    fn exact(&self, e: &Expr) -> Option<f64> {
        let Expr::Node(n) = e else {
            return self.number(e);
        };
        Some(match n.as_ref() {
            ExprNode::Param(_) => return self.number(e),
            ExprNode::Add([a, b]) => self.exact(a)? + self.exact(b)?,
            ExprNode::Sub([a, b]) => self.exact(a)? - self.exact(b)?,
            ExprNode::Mul([a, b]) => self.exact(a)? * self.exact(b)?,
            ExprNode::Min([a, b]) => least_of(self.exact(a)?, self.exact(b)?),
            ExprNode::Max([a, b]) => most_of(self.exact(a)?, self.exact(b)?),
            ExprNode::Neg(x) => -self.exact(x)?,
            _ => return None,
        })
    }

    /// The least an expression can be: its known value, or a parameter's distribution's least.
    fn least_expr(&self, e: &Expr) -> Option<Least> {
        if let Expr::Node(n) = e {
            if let ExprNode::Param(name) = n.as_ref() {
                if let Some(PValue::Dist(d)) = self.given.and_then(|g| g.get(name.as_str())).or_else(|| self.ast.params.get(name).map(|p| &p.default)) {
                    return self.least_dist(d);
                }
            }
        }
        self.exact(e).map(Least::Known)
    }

    /// The least a distribution draws, when the document shows it (`None` when it does not).
    fn least_dist(&self, d: &Dist) -> Option<Least> {
        match d {
            Dist::Const(e) => self.least_expr(e),
            Dist::Uniform { lo, .. } => Some(Least::Known(self.exact(lo)?)),
            Dist::Uniform64 {} => Some(Least::Unbounded),
            Dist::Normal { mean, sd, min, max } => {
                let raw = if self.exact(sd)? == 0.0 { Least::Known(self.exact(mean)?) } else { Least::Unbounded };
                self.clamped(raw, min.as_deref(), max.as_deref())
            }
            Dist::Lognormal { median, min, max, .. } => {
                let m = self.exact(median)?;
                let raw = if m < 0.0 { Least::Unbounded } else { Least::Known(0.0) };
                self.clamped(raw, min.as_deref(), max.as_deref())
            }
            Dist::Empirical { values, .. } => {
                let mut least: Option<f64> = None;
                for v in values {
                    let x = match v {
                        Literal::Int(n) => *n as f64,
                        Literal::Float(x) => *x,
                        _ => return None,
                    };
                    least = Some(least.map_or(x, |l| l.min(x)));
                }
                least.map(Least::Known)
            }
            // the arms that can be drawn: any one not a finite number is no finite size, whatever
            // the order of the arms; else any one unbounded or below 0 is; else every one known
            Dist::Mixture(arms) => {
                let (mut least, mut unknown, mut unbounded) = (None::<f64>, false, false);
                for d in arms.iter().filter(|a| a.weight > 0.0).filter_map(|a| a.dist.as_ref()) {
                    let l = match d {
                        DistRef::Dist(d) => self.least_dist(d),
                        DistRef::Expr(e) => self.least_expr(e),
                    };
                    match l {
                        Some(Least::Unbounded) => unbounded = true,
                        Some(Least::Known(x)) if !x.is_finite() => return Some(Least::Known(f64::NAN)),
                        Some(Least::Known(x)) => least = Some(least.map_or(x, |l| l.min(x))),
                        None => unknown = true,
                    }
                }
                if unbounded {
                    return Some(Least::Unbounded);
                }
                match least {
                    Some(x) if x.round() < 0.0 || !unknown => Some(Least::Known(x)),
                    _ => None,
                }
            }
            Dist::Zipf { .. } | Dist::Hotset { .. } => None,
        }
    }

    /// A draw's least after the runner's clamp to `min` and then to `max`.
    fn clamped(&self, raw: Least, min: Option<&Expr>, max: Option<&Expr>) -> Option<Least> {
        let mut least = raw;
        if let Some(m) = min {
            let m = self.exact(m)?;
            least = Least::Known(match least {
                Least::Unbounded => m,
                Least::Known(v) => most_of(v, m),
            });
        }
        if let Some(m) = max {
            least = match (least, self.exact(m)) {
                (Least::Unbounded, _) => Least::Unbounded,
                (Least::Known(v), Some(m)) => Least::Known(least_of(v, m)),
                (Least::Known(_), None) => return None,
            };
        }
        Some(least)
    }

    fn handle(&mut self, h: &'a Handle, path: &P, scope: &Scope<'a>, st: &mut ActorState<'a>) {
        match h {
            Handle::Ref(name) => {
                if scope.lookup(name).is_none() {
                    self.err(path, format!("handle binding `{name}` not in scope"));
                }
            }
            Handle::File(f) => {
                match (&f.dataset, &f.id, &f.of) {
                    (Some(ds), Some(id), None) => {
                        if !self.ast.datasets.contains_key(ds) {
                            self.err(path, format!("unknown dataset `{ds}`"));
                        }
                        self.expr(id, &p(path, &["id"]), scope, st, None);
                    }
                    (None, None, Some(of)) => self.handle(of, &p(path, &["of"]), scope, st),
                    _ => self.err(path, "file handle needs either {dataset, id} or {of}"),
                }
                if let Some(e) = &f.chunk {
                    self.expr(e, &p(path, &["chunk"]), scope, st, None);
                }
            }
            Handle::Dir { dataset, id } => {
                if !matches!(self.ast.datasets.get(dataset), Some(d) if d.layout != Layout::Array) {
                    self.err(path, format!("`dir` needs a dataset named by ids, not an `array`: got `{dataset}`"));
                }
                self.expr(id, &p(path, &["id"]), scope, st, None);
            }
            Handle::Object { namespace, fields } => match self.ast.namespaces.get(namespace) {
                None => self.err(path, format!("unknown namespace `{namespace}`")),
                Some(ns) => {
                    let have: Vec<&String> = fields.keys().collect();
                    let want: Vec<&String> = ns.fields.keys().collect();
                    if have != want {
                        self.err(path, format!("object fields {have:?} != namespace fields {want:?}"));
                    }
                    for (f, e) in fields {
                        self.expr(e, &p(path, &["fields", f]), scope, st, None);
                    }
                }
            },
            Handle::Inventory(ds) => match self.ast.datasets.get(ds) {
                None => self.err(path, format!("unknown dataset `{ds}`")),
                Some(d) if d.inventory.is_none() => self.err(path, format!("`inventory` of `{ds}`, which declares none (V22)")),
                Some(_) => {}
            },
            Handle::Consume(ds) => {
                match self.ast.datasets.get(ds) {
                    None => self.err(path, format!("unknown dataset `{ds}`")),
                    Some(d) if !d.access.consumed() => self.err(path, format!("`consume` from `{ds}`, which is `indexed`: its items are looked up by id (`file`, `pick`), never drawn from the shuffle (V21)")),
                    Some(_) => {}
                }
                if scope.indices.is_empty() {
                    self.err(path, "`consume` outside any loop: its position needs an index");
                } else {
                    let seen = self.consumes.entry(ds.as_str()).or_default();
                    if !seen.iter().any(|(q, _)| q == path) {
                        seen.push((path.clone(), scope.count_varies.clone()));
                    }
                }
            }
            Handle::Pick(pk) => {
                if !self.ast.datasets.contains_key(&pk.dataset) {
                    self.err(path, format!("unknown dataset `{}`", pk.dataset));
                }
                if let Some(d) = &pk.dist {
                    self.distref(d, &p(path, &["dist"]), scope, st);
                }
            }
            Handle::Unit(u) => {
                self.handle(&u.of, &p(path, &["of"]), scope, st);
                if let Some(e) = &u.index {
                    self.expr(e, &p(path, &["index"]), scope, st, None);
                }
            }
            Handle::Column(c) => {
                self.handle(&c.of, &p(path, &["of"]), scope, st);
                self.expr(&c.index, &p(path, &["index"]), scope, st, None);
            }
        }
    }
}
