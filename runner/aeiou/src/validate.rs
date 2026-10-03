//! The semantic rules of `schema/README.md` §4 (V1–V15), ported from `schema/check.py`. The
//! runner must reject everything the reference checker rejects; the structural rules the
//! JSON Schema states and serde cannot (identifier syntax, errno syntax, ranges, uniqueness)
//! are checked here too. Messages carry the `/`-joined path of the offending node, as
//! `check.py` prints it.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::*;

pub const RESERVED_PREFIX: &str = ".aeiou";

pub fn check(ast: &Ast) -> Vec<String> {
    let mut c = Checker { ast, given: None, errors: Vec::new(), ops: BTreeMap::new() };
    c.run();
    c.errors
}

/// Validate and return the op-kind counts (as `check.py` prints them) when there are no errors.
pub fn check_with_ops(ast: &Ast) -> Result<BTreeMap<&'static str, usize>, Vec<String>> {
    let mut c = Checker { ast, given: None, errors: Vec::new(), ops: BTreeMap::new() };
    c.run();
    if c.errors.is_empty() {
        Ok(c.ops)
    } else {
        Err(c.errors)
    }
}

/// The rules again with the parameters in effect in place of the defaults (§3.57): rule V3
/// judges a `param` by its distribution's minimum, and a `--params` file or `--param` may
/// replace that distribution with one whose minimum is lower. Without this an `at` offset
/// of zero is a loop in the VM. Called from `Params::new`, so every subcommand gets it.
pub fn check_given(ast: &Ast, given: &BTreeMap<String, PValue>) -> Vec<String> {
    let mut c = Checker { ast, given: Some(given), errors: Vec::new(), ops: BTreeMap::new() };
    c.run();
    c.errors
}

struct Checker<'a> {
    ast: &'a Ast,
    /// the parameter values in effect, when the check is on a run and not on the document
    given: Option<&'a BTreeMap<String, PValue>>,
    errors: Vec<String>,
    ops: BTreeMap<&'static str, usize>,
}

#[derive(Clone)]
struct Scope<'a> {
    bindings: Vec<HashMap<&'a str, &'a ExprLike>>, // innermost last
    indices: Vec<&'a str>,
    /// indices provably ≥ 0 (rule V3): a `loop` whose `from` is absent or provably ≥ 0, every `parallel` and `loader`
    nonneg_indices: HashSet<&'a str>,
    fields: HashSet<&'a str>,
    forward: HashSet<&'a str>,
}

/// State shared across every scope of one actor, as `check.py` shares `channels` and `written`.
#[derive(Default)]
struct ActorState<'a> {
    channels: HashSet<&'a str>,
    written: HashSet<&'a str>,
}

impl<'a> Scope<'a> {
    fn new(fields: HashSet<&'a str>) -> Self {
        Scope { bindings: vec![HashMap::new()], indices: Vec::new(), nonneg_indices: HashSet::new(), fields, forward: HashSet::new() }
    }

    fn child(&self) -> Self {
        let mut s = self.clone();
        s.bindings.push(HashMap::new());
        s.forward = HashSet::new();
        s
    }

    fn with_index(&self, name: &'a str, nonneg: bool) -> Self {
        let mut s = self.child();
        s.indices.push(name);
        if nonneg {
            s.nonneg_indices.insert(name);
        } else {
            s.nonneg_indices.remove(name);
        }
        s
    }

    fn bind(&mut self, name: &'a str, value: &'a ExprLike) {
        self.bindings.last_mut().unwrap().insert(name, value);
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
    match d {
        Dataset::Regions(r) => r.file.rsplit_once('/').map(|(a, _)| a).unwrap_or("").to_string(),
        Dataset::Files(f) => {
            let prefix = f.pattern.split('{').next().unwrap_or("");
            prefix.rsplit_once('/').map(|(a, _)| a).unwrap_or("").to_string()
        }
    }
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

impl<'a> Checker<'a> {
    fn err(&mut self, path: &P, msg: impl AsRef<str>) {
        self.errors.push(format!("{}: {}: {}", self.ast.name, path.join("/"), msg.as_ref()));
    }

    fn run(&mut self) {
        let ast = self.ast;
        if !is_ident(&ast.name) {
            self.err(&vec!["name".into()], "not an identifier");
        }
        if let Some(b) = &ast.backend {
            if crate::backend::BackendKind::parse(b).is_none() {
                self.err(&vec!["backend".into()], format!("`{b}` is not one of {}", crate::backend::NAMES));
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
            self.pvalue(&param.default, &path);
        }
        let mut roots: BTreeMap<String, &str> = BTreeMap::new();
        for (dname, d) in &ast.datasets {
            let path = vec!["datasets".into(), dname.clone()];
            if !is_ident(dname) {
                self.err(&path, "not an identifier");
            }
            let scope = Scope::new(HashSet::new());
            let mut st = ActorState::default();
            match d {
                Dataset::Files(f) => {
                    self.distref(&f.size, &p(&path, &["size"]), &scope, &mut st);
                    self.expr(&f.count, &p(&path, &["count"]), &scope, &mut st, None);
                    if let Some(e) = &f.samples_per_file {
                        self.expr(e, &p(&path, &["samples_per_file"]), &scope, &mut st, None);
                    }
                    if let Some(e) = &f.chunk {
                        self.expr(e, &p(&path, &["chunk"]), &scope, &mut st, None);
                    }
                    if let Some(fm) = &f.format {
                        if !is_ident(&fm.class) {
                            self.err(&p(&path, &["format", "class"]), "not an identifier");
                        }
                        if let Some(l) = &fm.layout {
                            let lp = p(&path, &["format", "layout"]);
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
                }
                Dataset::Regions(r) => {
                    self.distref(&r.size, &p(&path, &["size"]), &scope, &mut st);
                    self.expr(&r.count, &p(&path, &["count"]), &scope, &mut st, None);
                    self.expr(&r.slot, &p(&path, &["slot"]), &scope, &mut st, None);
                    if r.file.is_empty() {
                        self.err(&p(&path, &["file"]), "empty file");
                    }
                    self.reserved(&r.file, &path);
                }
            }
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
        let mut nroots: HashMap<String, (&str, bool)> = HashMap::new();
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
            if let Some((other, oinp)) = nroots.get(&nroot) {
                if *oinp != inp {
                    self.err(&path, format!("shares root `{nroot}/` with namespace `{other}` but `input` differs (V14)"));
                }
            } else {
                nroots.insert(nroot, (nname.as_str(), inp));
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
            self.body(&a.body, &p(&path, &["body"]), &scope, &mut st);
        }
    }

    fn reserved(&mut self, pattern: &str, path: &P) {
        for comp in pattern.split('/') {
            if comp.starts_with(RESERVED_PREFIX) {
                self.err(path, format!("path component `{comp}` begins with `{RESERVED_PREFIX}`, reserved for the manifest (V13)"));
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
        scope.forward = nodes
            .iter()
            .filter_map(|n| match n {
                Node::Let { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
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
                scope.bind(name, value);
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
                let inner = scope.with_index(index, nonneg);
                self.body(body, &p(pp, &["body"]), &inner, st);
            }
            Node::Parallel { index, width, body } => {
                if !is_ident(index) {
                    self.err(&p(pp, &["index"]), "not an identifier");
                }
                self.expr(width, &p(pp, &["width"]), scope, st, None);
                let inner = scope.with_index(index, true);
                self.body(body, &p(pp, &["body"]), &inner, st);
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
                st.channels.insert(name);
                let inner = scope.with_index(index, true);
                self.body(body, &p(pp, &["body"]), &inner, st);
            }
            Node::Channel { name, capacity, .. } => {
                if !is_ident(name) {
                    self.err(&p(pp, &["name"]), "not an identifier");
                }
                self.expr(capacity, &p(pp, &["capacity"]), scope, st, None);
                st.channels.insert(name);
            }
            Node::Put { channel, seq } => {
                if !st.channels.contains(channel.as_str()) {
                    self.err(pp, format!("channel `{channel}` not declared"));
                }
                self.expr(seq, &p(pp, &["seq"]), scope, st, None);
            }
            Node::Take { channel } => {
                if !st.channels.contains(channel.as_str()) {
                    self.err(pp, format!("channel `{channel}` not declared"));
                }
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
        if let Node::Read { file, repeat: Some(Repeat::UntilEof(_)), .. } = node {
            self.check_until_eof(file, pp, scope, st);
        }
        // V12: datasets are read-only
        if matches!(node, Node::Write { .. } | Node::Ftruncate { .. } | Node::Fallocate { .. } | Node::Unlink(_) | Node::Rename { .. }) {
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
            Handle::Consume(ds) => Some(ds.as_str()),
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
                if !matches!(self.ast.datasets.get(name), Some(Dataset::Files(_))) {
                    self.err(path, format!("`dirs` needs a files dataset, got `{name}`"));
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
            Dist::Normal { min: Some(m), .. } | Dist::Lognormal { min: Some(m), .. } => lit_ge(m),
            Dist::Normal { .. } | Dist::Lognormal { .. } => false,
            Dist::Empirical { values, .. } => values.iter().all(|v| match v {
                Literal::Int(n) => *n >= k,
                Literal::Float(x) => *x >= k as f64,
                _ => false,
            }),
            Dist::Mixture(arms) => arms.iter().all(|a| a.dist.as_ref().map_or(true, |d| self.distref_at_least(d, k, scope))),
            Dist::Const(e) => self.at_least(e, k, scope),
            _ => false,
        }
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
                if !matches!(self.ast.datasets.get(dataset), Some(Dataset::Files(_))) {
                    self.err(path, format!("`dir` needs a files dataset, got `{dataset}`"));
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
            Handle::Consume(ds) => {
                if !self.ast.datasets.contains_key(ds) {
                    self.err(path, format!("unknown dataset `{ds}`"));
                }
                if scope.indices.is_empty() {
                    self.err(path, "`consume` outside any loop: its position needs an index");
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
