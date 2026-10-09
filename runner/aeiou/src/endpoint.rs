//! Where each dataset and namespace lives (`DESIGN_REVIEW.md` §3.65): by default under
//! `--root`, or at the endpoint `--endpoint NAME=DIR` gives it. An endpoint places a name's
//! root directory (the constant directory prefix of its pattern, `schema/README.md` §6), so
//! the path `kv/sys/0001/blk_0002` of a dataset whose root is `kv/sys` is `DIR/0001/blk_0002`
//! under `--endpoint sysp=DIR`. A path falls under the longest root that is a prefix of it
//! and has an endpoint; any other path, and every path when no endpoint is given, is under
//! `--root`. Namespaces sharing a root share its place (V14, V17).
//!
//! The protocol a name declares decides what an endpoint may be: a directory for `posix`.
//! `object` is accepted by the contract and refused here until the object engine is built
//! (§3.65, step 3).

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::ast::{Ast, Dataset, Protocol};
use crate::payload;

/// Where an abstract's relative paths live: a bare root (every path under it), or a run's
/// root with its endpoints.
pub trait Place {
    fn at(&self, rel: &str) -> PathBuf;
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
}

/// One placed root: the names it belongs to, its root relative to `--root`, and its directory.
#[derive(Debug, Clone)]
pub struct Placed {
    pub names: Vec<String>,
    pub root: String,
    pub dir: PathBuf,
}

/// The endpoints of a run, longest root first.
#[derive(Debug, Clone, Default)]
pub struct Endpoints {
    placed: Vec<Placed>,
}

/// A dataset's or namespace's protocol and root, by name.
fn lookup(ast: &Ast, name: &str) -> Result<Option<(Protocol, String)>> {
    if let Some(d) = ast.datasets.get(name) {
        let p = match d {
            Dataset::Files(f) => f.protocol,
            Dataset::Regions(r) => r.protocol,
        };
        return Ok(Some((p.unwrap_or_default(), payload::dataset_root(ast, name)?)));
    }
    if let Some(n) = ast.namespaces.get(name) {
        return Ok(Some((n.protocol.unwrap_or_default(), payload::namespace_root(ast, name)?)));
    }
    Ok(None)
}

/// Refuse an abstract that declares a protocol this runner has no engine for.
pub fn check_protocols(ast: &Ast) -> Result<()> {
    for name in ast.datasets.keys().chain(ast.namespaces.keys()) {
        if let Some((Protocol::Object, _)) = lookup(ast, name)? {
            bail!("`{name}` is declared `protocol: object`, and the object engine is not built yet (DESIGN_REVIEW.md §3.65, step 3)");
        }
    }
    Ok(())
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
            if protocol != Protocol::Posix || dir.contains("://") {
                crate::usage!("--endpoint {g}: only a directory for a `posix` dataset or namespace; object endpoints come with the object engine (DESIGN_REVIEW.md §3.65, step 3)");
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
                None => placed.push(Placed { names: vec![name.to_string()], root, dir: PathBuf::from(dir) }),
            }
        }
        placed.sort_by(|a, b| b.root.len().cmp(&a.root.len()).then_with(|| a.root.cmp(&b.root)));
        Ok(Endpoints { placed })
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
                .map(|p| serde_json::json!({"names": p.names, "root": p.root, "dir": p.dir.display().to_string()}))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ast() -> Ast {
        crate::load_str(
            r#"{"ast": "0.6", "name": "t",
                "datasets": {"sysp": {"files": {"pattern": "kv/sys/{id:04}/blk", "count": 2, "size": {"const": 1}, "seed": 1}},
                             "flat": {"files": {"pattern": "flat/f_{id}", "count": 2, "size": {"const": 1}, "seed": 2}}},
                "namespaces": {"kv": {"pattern": "kv/{c}.pt", "fields": {"c": "int"}, "size": 1, "seed": 3},
                               "kw": {"pattern": "kv/{c}.w", "fields": {"c": "int"}, "size": 1, "seed": 4},
                               "obj": {"pattern": "o/{c}", "fields": {"c": "int"}, "size": 1, "seed": 5, "protocol": "object"}},
                "actors": {"a": {"body": [{"stat": {"file": {"file": {"dataset": "flat", "id": 0}}}}]}}}"#,
        )
        .unwrap()
        .ast
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
    fn names_sharing_a_root_share_its_place_and_object_waits_for_its_engine() {
        let a = ast();
        let err = |g: &[&str]| format!("{:#}", Endpoints::parse(&a, &g.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap_err());
        assert!(Endpoints::parse(&a, &["kv=/n".into(), "kw=/n".into()]).is_ok());
        assert!(err(&["kv=/n", "kw=/m"]).contains("a root is in one place"));
        assert!(err(&["kv=/n", "kv=/n"]).contains("given twice"));
        assert!(err(&["nope=/n"]).contains("no dataset or namespace `nope`"));
        assert!(err(&["kv"]).contains("expected NAME=DIR"));
        assert!(err(&["kv=s3://b/kv"]).contains("object endpoints come with the object engine"));
        assert!(err(&["obj=/o"]).contains("object endpoints come with the object engine"));
        assert!(format!("{:#}", check_protocols(&a).unwrap_err()).contains("`obj` is declared `protocol: object`"));
    }
}
