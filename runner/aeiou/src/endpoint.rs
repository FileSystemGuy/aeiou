//! Where each dataset and namespace lives (`DESIGN_REVIEW.md` §3.65): by default under
//! `--root`, or at the endpoint `--endpoint NAME=DIR` gives it. An endpoint places a name's
//! root directory (the constant directory prefix of its pattern, `schema/README.md` §6), so
//! the path `kv/sys/0001/blk_0002` of a dataset whose root is `kv/sys` is `DIR/0001/blk_0002`
//! under `--endpoint sysp=DIR`. A path falls under the longest root that is a prefix of it
//! and has an endpoint; any other path, and every path when no endpoint is given, is under
//! `--root`. Namespaces sharing a root share its place (V14, V17).
//!
//! The protocol a name declares decides what an endpoint may be: a directory for `posix`,
//! `s3://BUCKET[/PREFIX]` for `s3` (`object.rs`, built with the cargo feature `object`).
//! An s3 name has no default place, so every one needs an endpoint, and a name may not
//! fall under a root placed for the other protocol.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

use crate::object::Store;

use crate::ast::{Ast, Protocol};
use crate::payload;

/// Where an abstract's relative paths live: a bare root (every path under it), or a run's
/// root with its endpoints.
pub trait Place {
    fn at(&self, rel: &str) -> PathBuf;
    /// The store and the key below its prefix, for a path placed in an object store.
    fn object(&self, _rel: &str) -> Option<(Arc<Store>, String)> {
        None
    }
}

impl Place for Path {
    fn at(&self, rel: &str) -> PathBuf {
        self.join(rel)
    }
}

impl Place for PathBuf {
    fn at(&self, rel: &str) -> PathBuf {
        self.join(rel)
    }
}

impl Place for (&Path, &Endpoints) {
    fn at(&self, rel: &str) -> PathBuf {
        self.1.path(self.0, rel)
    }
    fn object(&self, rel: &str) -> Option<(Arc<Store>, String)> {
        self.1.object(rel)
    }
}

/// One placed root: the names it belongs to, its root relative to `--root`, and its
/// directory, or its object store (`dir` is then the URI, for messages).
#[derive(Clone)]
pub struct Placed {
    pub names: Vec<String>,
    pub root: String,
    pub dir: PathBuf,
    pub store: Option<Arc<Store>>,
}

impl std::fmt::Debug for Placed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Placed {{ names: {:?}, root: {:?}, at: {:?} }}", self.names, self.root, self.dir)
    }
}

/// The endpoints of a run, longest root first.
#[derive(Debug, Clone, Default)]
pub struct Endpoints {
    placed: Vec<Placed>,
}

/// `p` as a comparable path: absolute, `.` and `..` resolved, and its longest existing
/// ancestor as the file system resolves it (symbolic links), the rest as written.
fn resolved(p: &Path) -> PathBuf {
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut lex = PathBuf::new();
    for c in abs.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                lex.pop();
            }
            c => lex.push(c),
        }
    }
    let mut base = lex.as_path();
    let mut rest = Vec::new();
    loop {
        if let Ok(mut out) = base.canonicalize() {
            out.extend(rest.iter().rev());
            return out;
        }
        match (base.parent(), base.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                base = parent;
            }
            _ => return lex,
        }
    }
}

/// A dataset's or namespace's protocol and root, by name.
fn lookup(ast: &Ast, name: &str) -> Result<Option<(Protocol, String)>> {
    if let Some(d) = ast.datasets.get(name) {
        return Ok(Some((d.protocol.unwrap_or_default(), payload::dataset_root(ast, name)?)));
    }
    if let Some(n) = ast.namespaces.get(name) {
        return Ok(Some((n.protocol.unwrap_or_default(), payload::namespace_root(ast, name)?)));
    }
    Ok(None)
}

impl Endpoints {
    /// From the `--endpoint NAME=DIR` values, checked against the abstract.
    pub fn parse(ast: &Ast, given: &[String]) -> Result<Endpoints> {
        let mut placed: Vec<Placed> = Vec::new();
        for g in given {
            let Some((name, dir)) = g.split_once('=') else {
                crate::usage!("--endpoint {g}: expected NAME=DIR");
            };
            let Some((protocol, root)) = lookup(ast, name)? else {
                crate::usage!("--endpoint {g}: the abstract has no dataset or namespace `{name}`");
            };
            let uri = crate::object::is_uri(dir);
            match protocol {
                Protocol::Posix if uri => crate::usage!("--endpoint {g}: `{name}` is a `posix` dataset or namespace; its endpoint is a directory"),
                Protocol::S3 if !uri => crate::usage!("--endpoint {g}: `{name}` is declared `protocol: s3`; its endpoint is s3://BUCKET[/PREFIX]"),
                _ => {}
            }
            if dir.is_empty() {
                crate::usage!("--endpoint {g}: no directory");
            }
            match placed.iter_mut().find(|p| p.root == root) {
                Some(p) if p.names.iter().any(|n| n == name) => crate::usage!("--endpoint {name}: given twice"),
                Some(p) if p.dir != Path::new(dir) => crate::usage!(
                    "--endpoint {g}: `{name}` shares root `{root}/` with `{}`, placed at {}; a root is in one place",
                    p.names.join("`, `"),
                    p.dir.display()
                ),
                Some(p) => p.names.push(name.to_string()),
                None => {
                    let store = if uri { Some(Arc::new(Store::open(dir).map_err(|e| crate::usage::err(format!("--endpoint {g}: {e:#}")))?)) } else { None };
                    placed.push(Placed { names: vec![name.to_string()], root, dir: PathBuf::from(dir), store })
                }
            }
        }
        placed.sort_by(|a, b| b.root.len().cmp(&a.root.len()).then_with(|| a.root.cmp(&b.root)));
        let e = Endpoints { placed };
        // every name lands where its protocol can be reached: an object name at an object
        // endpoint (there is no default for one), a posix name never under an object's root
        let mut names: Vec<&String> = ast.datasets.keys().chain(ast.namespaces.keys()).collect();
        names.sort();
        for name in names {
            let Some((protocol, root)) = lookup(ast, name)? else { continue };
            let under = e.placement(&root);
            match (protocol, under.map(|p| p.store.is_some())) {
                (Protocol::S3, None | Some(false)) => crate::usage!(
                    "`{name}` is declared `protocol: s3` and has no object endpoint{}: give it --endpoint {name}=s3://BUCKET[/PREFIX]",
                    under.map(|p| format!(" (it falls under `{}`, placed at {})", p.names.join("`, `"), p.dir.display())).unwrap_or_default()
                ),
                (Protocol::Posix, Some(true)) => {
                    let p = under.expect("placed");
                    crate::usage!("`{name}` is a `posix` dataset or namespace under `{}`, placed in the object store {}: give it a directory, --endpoint {name}=DIR", p.names.join("`, `"), p.dir.display())
                }
                _ => {}
            }
        }
        e.datasets_apart(ast, None)?;
        Ok(e)
    }

    /// Every dataset in a place of its own: no two resolved dataset roots the same directory
    /// or bucket prefix, nor one inside the other (V13 for the abstract's roots, here for where
    /// the endpoints put them: `aeiou datagen` wants each root empty and to itself, and writes
    /// one manifest at each). `root` is `--root`, for the datasets without an endpoint; with
    /// `None` (`parse`) only the placed ones are compared. A namespace may still hold a
    /// dataset's root (bug 108), so namespaces are not compared.
    pub fn datasets_apart(&self, ast: &Ast, root: Option<&Path>) -> Result<()> {
        enum At {
            Dir(PathBuf),
            Key(Arc<Store>, String),
        }
        let mut names: Vec<&String> = ast.datasets.keys().collect();
        names.sort();
        let mut at: Vec<(&String, At, String)> = Vec::new();
        for name in names {
            let rel = payload::dataset_root(ast, name)?;
            let (place, shown) = match (self.placement(&rel), root) {
                (Some(p), _) => {
                    let rest = if p.root.is_empty() { rel.as_str() } else { rel[p.root.len()..].trim_start_matches('/') };
                    let via = if p.names.iter().any(|n| n == name) { format!("--endpoint {name}={}", p.dir.display()) } else { format!("under `{}`, --endpoint {}={}", p.names.join("`, `"), p.names[0], p.dir.display()) };
                    match &p.store {
                        Some(s) => (At::Key(s.clone(), s.full(rest)), format!("{} ({via})", s.show(rest))),
                        None => {
                            let d = if rest.is_empty() { p.dir.clone() } else { p.dir.join(rest) };
                            (At::Dir(resolved(&d)), format!("{} ({via})", d.display()))
                        }
                    }
                }
                (None, Some(r)) => {
                    let d = r.join(&rel);
                    (At::Dir(resolved(&d)), format!("{} (under --root)", d.display()))
                }
                (None, None) => continue,
            };
            at.push((name, place, shown));
        }
        let nested = |a: &str, b: &str| {
            let (a, b): (Vec<&str>, Vec<&str>) = (a.split('/').filter(|c| !c.is_empty()).collect(), b.split('/').filter(|c| !c.is_empty()).collect());
            a.starts_with(&b) || b.starts_with(&a)
        };
        for (i, (n1, a1, s1)) in at.iter().enumerate() {
            for (n2, a2, s2) in &at[i + 1..] {
                let clash = match (a1, a2) {
                    (At::Dir(x), At::Dir(y)) => x.starts_with(y) || y.starts_with(x),
                    (At::Key(s, x), At::Key(t, y)) => s.same_bucket(t) && nested(x, y),
                    _ => false,
                };
                if clash {
                    crate::usage!(
                        "datasets `{n1}` and `{n2}` would share a place: `{n1}` at {s1}, `{n2}` at {s2}; each dataset's root is a directory or prefix of its own, never the same as or inside another's (one manifest at each, V13)"
                    );
                }
            }
        }
        Ok(())
    }

    /// The placed root a path falls under: the longest that is a prefix of it.
    fn placement(&self, rel: &str) -> Option<&Placed> {
        self.placed.iter().find(|p| p.root.is_empty() || rel == p.root || rel.strip_prefix(p.root.as_str()).is_some_and(|r| r.starts_with('/')))
    }

    /// The store and the key below its prefix of the path `rel`, when it falls under a root
    /// placed in an object store.
    pub fn object(&self, rel: &str) -> Option<(Arc<Store>, String)> {
        let p = self.placement(rel)?;
        let store = p.store.as_ref()?;
        let rest = if p.root.is_empty() { rel } else { rel[p.root.len()..].trim_start_matches('/') };
        Some((store.clone(), rest.to_string()))
    }

    /// Some name is placed in an object store.
    pub fn any_object(&self) -> bool {
        self.placed.iter().any(|p| p.store.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.placed.is_empty()
    }

    pub fn placed(&self) -> &[Placed] {
        &self.placed
    }

    /// Where the path `rel` (relative to `--root`, as the abstract names it) lives.
    pub fn path(&self, root: &Path, rel: &str) -> PathBuf {
        for p in &self.placed {
            if p.root.is_empty() {
                return p.dir.join(rel);
            }
            if let Some(rest) = rel.strip_prefix(p.root.as_str()) {
                if rest.is_empty() {
                    return p.dir.clone();
                }
                if let Some(rest) = rest.strip_prefix('/') {
                    return p.dir.join(rest);
                }
            }
        }
        root.join(rel)
    }

    /// What a run prints of its endpoints: one line each, and a warning for one on another
    /// file system than `--root`, whose mount the report's counters do not cover.
    pub fn lines(&self, root: &Path) -> Vec<String> {
        use std::os::unix::fs::MetadataExt;
        let dev = |p: &Path| std::fs::metadata(p).ok().map(|m| m.dev());
        let mut out = Vec::new();
        for p in &self.placed {
            out.push(format!("endpoint {}  root {}/  at {}", p.names.join(", "), p.root, p.dir.display()));
            if p.store.is_some() {
                out.push(format!("  {} is an object store: the report's mount counters and the residency sample do not cover it", p.dir.display()));
                continue;
            }
            if dev(&p.dir).is_some() && dev(&p.dir) != dev(root) {
                out.push(format!("  WARNING: {} is on another file system than --root; the report's mount counters cover --root's mount only", p.dir.display()));
            }
        }
        out
    }

    /// The endpoints as the report records them: name, root, directory.
    pub fn json(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.placed
                .iter()
                .map(|p| match p.store {
                    Some(_) => serde_json::json!({"names": p.names, "root": p.root, "uri": p.dir.display().to_string()}),
                    None => serde_json::json!({"names": p.names, "root": p.root, "dir": p.dir.display().to_string()}),
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the object namespace `obj` when `object`.
    fn ast_with(object: bool) -> Ast {
        let obj = if object { r#", "obj": {"pattern": "o/{c}", "fields": {"c": "int"}, "size": 1, "seed": 5, "protocol": "s3"}"# } else { "" };
        crate::load_str(&format!(
            r#"{{"ast": "0.11", "name": "t",
                "datasets": {{"sysp": {{"layout": "whole", "access": "map", "pattern": "kv/sys/{{id:04}}/blk", "count": 2, "size": {{"const": 1}}, "seed": 1}},
                             "flat": {{"layout": "whole", "access": "map", "pattern": "flat/f_{{id}}", "count": 2, "size": {{"const": 1}}, "seed": 2}}}},
                "namespaces": {{"kv": {{"pattern": "kv/{{c}}.pt", "fields": {{"c": "int"}}, "size": 1, "seed": 3}},
                               "kw": {{"pattern": "kv/{{c}}.w", "fields": {{"c": "int"}}, "size": 1, "seed": 4}}{obj}}},
                "actors": {{"a": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "flat", "id": 0}}}}}}}}]}}}}}}"#
        ))
        .unwrap()
        .ast
    }

    fn ast() -> Ast {
        ast_with(false)
    }

    #[test]
    fn a_path_falls_under_the_longest_placed_root_and_otherwise_under_root() {
        let a = ast();
        let root = Path::new("/r");
        let none = Endpoints::default();
        assert_eq!(none.path(root, "kv/sys/0001/blk"), PathBuf::from("/r/kv/sys/0001/blk"));
        let e = Endpoints::parse(&a, &["kv=/n".into(), "sysp=/s".into()]).unwrap();
        assert_eq!(e.path(root, "kv/sys/0001/blk"), PathBuf::from("/s/0001/blk"));
        assert_eq!(e.path(root, "kv/sys"), PathBuf::from("/s"));
        assert_eq!(e.path(root, "kv/7.pt"), PathBuf::from("/n/7.pt"));
        assert_eq!(e.path(root, "kvx/7.pt"), PathBuf::from("/r/kvx/7.pt"));
        assert_eq!(e.path(root, "flat/f_1"), PathBuf::from("/r/flat/f_1"));
        // a nested root without its own endpoint moves with the one it is under
        let e = Endpoints::parse(&a, &["kv=/n".into()]).unwrap();
        assert_eq!(e.path(root, "kv/sys/0001/blk"), PathBuf::from("/n/sys/0001/blk"));
        let e = Endpoints::parse(&a, &["flat=/f".into()]).unwrap();
        assert_eq!(e.path(root, "flat/f_1"), PathBuf::from("/f/f_1"));
        assert_eq!(e.path(root, "kv/7.pt"), PathBuf::from("/r/kv/7.pt"));
    }

    #[test]
    fn names_sharing_a_root_share_its_place_and_an_endpoint_fits_its_protocol() {
        let a = ast_with(true);
        let err = |g: &[&str]| format!("{:#}", Endpoints::parse(&a, &g.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap_err());
        // `obj` is an object namespace: without its endpoint nothing parses
        assert!(err(&["kv=/n"]).contains("`obj` is declared `protocol: s3` and has no object endpoint"));
        assert!(err(&["kv=/n", "kw=/m"]).contains("a root is in one place"));
        assert!(err(&["kv=/n", "kv=/n"]).contains("given twice"));
        assert!(err(&["nope=/n"]).contains("no dataset or namespace `nope`"));
        assert!(err(&["kv"]).contains("expected NAME=DIR"));
        assert!(err(&["kv=s3://b/kv"]).contains("`kv` is a `posix` dataset or namespace; its endpoint is a directory"));
        assert!(err(&["obj=/o"]).contains("`obj` is declared `protocol: s3`; its endpoint is s3://BUCKET[/PREFIX]"));
    }

    #[test]
    fn two_datasets_are_never_placed_at_one_directory_or_one_inside_the_other() {
        let a = ast();
        let parse = |g: &[&str]| Endpoints::parse(&a, &g.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let err = |g: &[&str]| format!("{:#}", parse(g).unwrap_err());
        let e = err(&["sysp=/s", "flat=/s"]);
        assert!(e.contains("datasets `flat` and `sysp` would share a place: `flat` at /s (--endpoint flat=/s), `sysp` at /s (--endpoint sysp=/s)"), "{e}");
        assert!(parse(&["sysp=/s", "flat=/s"]).unwrap_err().downcast_ref::<crate::usage::UsageError>().is_some());
        assert!(err(&["sysp=/s", "flat=/s/x"]).contains("would share a place"));
        assert!(err(&["sysp=/s/./x/..", "flat=/s"]).contains("would share a place"));
        assert!(err(&["sysp=/s/x", "flat=/s"]).contains("would share a place"));
        // `sysp` moves with the namespace `kv`, to /n/sys
        let e = err(&["kv=/n", "flat=/n/sys"]);
        assert!(e.contains("`sysp` at /n/sys (under `kv`, --endpoint kv=/n)"), "{e}");
        // apart, and a namespace's place holding a dataset's, are fine
        parse(&["sysp=/s", "flat=/sx"]).unwrap();
        parse(&["sysp=/s/a", "flat=/s/b"]).unwrap();
        parse(&["kv=/n", "flat=/n/flat"]).unwrap();
        // against --root: `flat` placed where `sysp` is by default, or around it
        let r = Path::new("/r");
        let e = format!("{:#}", parse(&["flat=/r/kv"]).unwrap().datasets_apart(&a, Some(r)).unwrap_err());
        assert!(e.contains("`flat` at /r/kv (--endpoint flat=/r/kv), `sysp` at /r/kv/sys (under --root)"), "{e}");
        assert!(format!("{:#}", parse(&["sysp=/r/flat/x"]).unwrap().datasets_apart(&a, Some(r)).unwrap_err()).contains("would share a place"));
        assert!(format!("{:#}", parse(&["sysp=/r"]).unwrap().datasets_apart(&a, Some(r)).unwrap_err()).contains("would share a place"));
        parse(&["flat=/r/flat"]).unwrap().datasets_apart(&a, Some(r)).unwrap();
        parse(&["flat=/r/f2"]).unwrap().datasets_apart(&a, Some(r)).unwrap();
        Endpoints::default().datasets_apart(&a, Some(r)).unwrap();
    }

    #[cfg(feature = "object")]
    #[test]
    fn two_object_datasets_are_never_placed_at_one_prefix_or_one_inside_the_other() {
        let a = crate::load_str(
            r#"{"ast": "0.11", "name": "t",
                "datasets": {"x": {"layout": "whole", "access": "map", "pattern": "x/f_{id}", "count": 2, "size": {"const": 1}, "seed": 1, "protocol": "s3"},
                             "y": {"layout": "whole", "access": "map", "pattern": "y/f_{id}", "count": 2, "size": {"const": 1}, "seed": 2, "protocol": "s3"}},
                "actors": {"a": {"body": [{"stat": {"file": {"file": {"dataset": "x", "id": 0}}}}]}}}"#,
        )
        .unwrap()
        .ast;
        let parse = |g: &[&str]| Endpoints::parse(&a, &g.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let e = format!("{:#}", parse(&["x=s3://b/p", "y=s3://b/p"]).unwrap_err());
        assert!(e.contains("`x` at s3://b/p (--endpoint x=s3://b/p), `y` at s3://b/p (--endpoint y=s3://b/p)"), "{e}");
        assert!(format!("{:#}", parse(&["x=s3://b/p", "y=s3://b/p/q"]).unwrap_err()).contains("would share a place"));
        assert!(format!("{:#}", parse(&["x=s3://b", "y=s3://b/q"]).unwrap_err()).contains("would share a place"));
        parse(&["x=s3://b/p", "y=s3://b/pq"]).unwrap();
        parse(&["x=s3://b/p", "y=s3://c/p"]).unwrap();
    }

    #[cfg(not(feature = "object"))]
    #[test]
    fn a_build_without_the_engine_refuses_an_object_endpoint() {
        let a = ast_with(true);
        let e = format!("{:#}", Endpoints::parse(&a, &["obj=s3://b/o".into()]).unwrap_err());
        assert!(e.contains("built without the object engine"), "{e}");
    }

    #[cfg(feature = "object")]
    #[test]
    fn a_posix_name_never_falls_under_an_object_root_and_a_key_keeps_its_rest() {
        let a = crate::load_str(
            r#"{"ast": "0.11", "name": "t",
                "datasets": {"inner": {"layout": "whole", "access": "map", "pattern": "o/in/f_{id}", "count": 2, "size": {"const": 1}, "seed": 1}},
                "namespaces": {"outer": {"pattern": "o/{c}", "fields": {"c": "int"}, "size": 1, "seed": 2, "protocol": "s3"}},
                "actors": {"a": {"body": [{"stat": {"file": {"file": {"dataset": "inner", "id": 0}}}}]}}}"#,
        )
        .unwrap()
        .ast;
        let parse = |g: &[&str]| Endpoints::parse(&a, &g.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let e = format!("{:#}", parse(&["outer=s3://b/p"]).unwrap_err());
        assert!(e.contains("`inner` is a `posix` dataset or namespace under `outer`, placed in the object store s3://b/p"), "{e}");
        let e = parse(&["outer=s3://b/p", "inner=/d"]).unwrap();
        assert_eq!(e.path(Path::new("/r"), "o/in/f_1"), PathBuf::from("/d/f_1"));
        assert!(e.object("o/in/f_1").is_none());
        let (s, rest) = e.object("o/f_1").unwrap();
        assert_eq!((s.uri(), rest.as_str()), ("s3://b/p", "f_1"));
        assert_eq!(e.object("o").unwrap().1, "");
        assert!(e.object("ox/f_1").is_none());
    }
}
