//! `aeiou cases` (DESIGN_REVIEW.md §3.80): the use cases a binary can run, from the
//! `catalog.json` that `aeiou-catalog write` generates beside `CATALOG.md` from the records
//! under `workloads/`. The runner reads that one file and never the records, so the catalog's
//! derivations (the API, writes, inputs, status, and the measured classes) have one author.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::case::{Lock, LOCK_FILE};
use crate::eval::State;

/// The catalog's index, beside `CATALOG.md` in `workloads/`.
pub const INDEX_FILE: &str = "catalog.json";
/// The `catalog_version` this runner reads (`aeiou-catalog`'s `INDEX_VERSION`).
pub const INDEX_VERSION: u64 = 1;

/// One attribute of the vocabulary and its values (a measured one's classes), in the
/// dictionary's order.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub values: Vec<String>,
}

#[derive(Debug)]
pub struct Index {
    pub path: PathBuf,
    pub attributes: Vec<Attribute>,
    /// Every case, as the index holds it, in the tree's order.
    pub cases: Vec<Value>,
}

impl Index {
    pub fn load(workloads: &Path) -> Result<Index> {
        let path = workloads.join(INDEX_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("{}: the catalog's index of the cases, which `aeiou-catalog write` makes beside CATALOG.md", path.display()))?;
        let doc: Value = serde_json::from_str(&text).with_context(|| format!("{}: not JSON", path.display()))?;
        let version = doc.get("catalog_version").and_then(Value::as_u64);
        if version != Some(INDEX_VERSION) {
            bail!("{}: catalog_version {} (this runner reads {INDEX_VERSION}); `aeiou-catalog write` of this release makes it", path.display(),
                  version.map_or("missing".to_string(), |v| v.to_string()));
        }
        let shape = || anyhow!("{}: not the catalog's index (`attributes` and `cases`, each a list)", path.display());
        let mut attributes = Vec::new();
        for a in doc.get("attributes").and_then(Value::as_array).ok_or_else(shape)? {
            let name = a.get("name").and_then(Value::as_str).ok_or_else(shape)?.to_string();
            let values = a.get("values").and_then(Value::as_array).ok_or_else(shape)?.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
            attributes.push(Attribute { name, values });
        }
        let cases = doc.get("cases").and_then(Value::as_array).ok_or_else(shape)?.clone();
        if cases.iter().any(|c| c.get("name").and_then(Value::as_str).is_none()) {
            return Err(shape());
        }
        Ok(Index { path, attributes, cases })
    }

    pub fn names(&self) -> Vec<String> {
        self.cases.iter().map(name).collect()
    }

    /// The case `given` names, as `--case` finds a lock: `mechanism/case`, or the case alone when
    /// one matches; several refused with each, none with the names near it.
    pub fn find(&self, given: &str) -> Result<&Value> {
        let found: Vec<&Value> = self.cases.iter().filter(|c| matches_name(&name(c), given)).collect();
        match found.len() {
            1 => Ok(found[0]),
            0 => crate::usage!("{given}: no case of that name{}", not_found(&self.names(), given)),
            n => crate::usage!("{given}: {n} cases match ({}): name it as mechanism/case", found.iter().map(|c| name(c)).collect::<Vec<_>>().join(", ")),
        }
    }
}

fn name(case: &Value) -> String {
    case["name"].as_str().unwrap_or_default().to_string()
}

/// Whether a lock's name (`mechanism/case`) is the one given, whole or by its case.
pub fn matches_name(full: &str, given: &str) -> bool {
    full == given || full.split_once('/').map(|(_, c)| c) == Some(given)
}

/// The names near `given`: its case contains it or is within two edits of it, or `given` (at
/// least three characters) contains it; eight at most, in their order.
pub fn near(names: &[String], given: &str) -> Vec<String> {
    let given_case = given.rsplit('/').next().unwrap_or(given);
    names
        .iter()
        .filter(|n| {
            let case = n.rsplit('/').next().unwrap_or(n);
            (given_case.len() >= 3 && (case.contains(given_case) || n.contains(given))) || edits(case, given_case) <= 2
        })
        .take(8)
        .cloned()
        .collect()
}

/// What a name that matches nothing adds to its error: the names near it, and the listing.
pub fn not_found(names: &[String], given: &str) -> String {
    let close = near(names, given);
    let close = if close.is_empty() { String::new() } else { format!("; near it: {}", close.join(", ")) };
    format!("{close}; `aeiou cases` lists every one")
}

/// Levenshtein distance, by characters.
fn edits(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let here = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != cb)).min(row[j] + 1).min(here + 1);
            prev = here;
        }
    }
    row[b.len()]
}

/// One `--where`: an attribute of the vocabulary and one of its values, kept (`=`) or dropped (`!=`).
#[derive(Debug, Clone)]
pub struct Filter {
    pub attr: String,
    pub value: String,
    pub keep: bool,
}

impl Filter {
    /// `ATTR=VALUE` or `ATTR!=VALUE`, both names of the vocabulary; anything else a usage error
    /// that says what is valid.
    pub fn parse(text: &str, index: &Index) -> Result<Filter> {
        let form = || crate::usage::err(format!("--where {text}: ATTR=VALUE keeps the cases with the value, ATTR!=VALUE drops them"));
        let (attr, value, keep) = match text.split_once("!=") {
            Some((a, v)) => (a, v, false),
            None => {
                let (a, v) = text.split_once('=').ok_or_else(form)?;
                (a, v, true)
            }
        };
        if attr.is_empty() || value.is_empty() || value.contains('=') || attr.contains('!') {
            return Err(form());
        }
        let Some(a) = index.attributes.iter().find(|a| a.name == attr) else {
            crate::usage!("--where {text}: no attribute `{attr}`; the attributes: {}", index.attributes.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", "));
        };
        if !a.values.iter().any(|v| v == value) {
            crate::usage!("--where {text}: `{value}` is not a value of {attr} ({})", a.values.join(", "));
        }
        Ok(Filter { attr: attr.into(), value: value.into(), keep })
    }

    /// A case has a value when it or its mechanism records it (one of several, for an attribute
    /// of several values) or any of its runs derives or measures it.
    pub fn holds(&self, case: &Value) -> bool {
        values(case, &self.attr).iter().any(|v| *v == self.value) == self.keep
    }
}

/// A case's values of an attribute: its own and its mechanism's, then every run's.
pub fn values(case: &Value, attr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut add = |v: Option<&Value>| match v {
        Some(Value::String(s)) => out.push(s.clone()),
        Some(Value::Array(a)) => out.extend(a.iter().filter_map(|x| x.as_str().map(str::to_string))),
        _ => {}
    };
    add(case.get("attributes").and_then(|a| a.get(attr)));
    for r in case.get("runs").and_then(Value::as_array).into_iter().flatten() {
        add(r.get("attributes").and_then(|a| a.get(attr)));
    }
    out
}

fn gpus(case: &Value) -> Vec<i64> {
    case.get("gpus").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default()
}

fn gpus_shown(g: &[i64]) -> String {
    let list = g.iter().map(i64::to_string).collect::<Vec<_>>().join(" or ");
    format!("{list} GPU{}", if g == [1] { "" } else { "s" })
}

fn str_of<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

/// The listing: a count, then each case on a line (its name, kind, GPU counts, category and
/// class, application) and what each run does on the next.
pub fn render_list(shown: &[&Value], total: usize, index: &Index) -> String {
    let mut out = format!("{} of {total} cases ({})\n", shown.len(), index.path.display());
    let width = shown.iter().map(|c| name(c).len()).max().unwrap_or(0);
    for c in shown {
        let a = &c["attributes"];
        let mut tags = vec![str_of(a, "kind").to_string(), gpus_shown(&gpus(c))];
        let place = [str_of(a, "category"), str_of(a, "class")].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("/");
        if !place.is_empty() {
            tags.push(place);
        }
        if !str_of(a, "application").is_empty() {
            tags.push(str_of(a, "application").to_string());
        }
        out += &format!("\n{:width$}  {}\n", name(c), tags.join(", "));
        let runs = c["runs"].as_array().cloned().unwrap_or_default();
        for r in &runs {
            match runs.len() {
                1 => out += &format!("    {}\n", str_of(r, "does")),
                _ => out += &format!("    {}: {}\n", str_of(r, "ast"), str_of(r, "does")),
            }
        }
    }
    out
}

/// What a lock's run fixes (`locked`), leaves to a run (`default`), and requires of a
/// submission (`required`), by name.
pub fn states(lock: &Lock, run: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let (mut locked, mut defaults, mut required) = (Vec::new(), Vec::new(), Vec::new());
    if let Some(r) = lock.runs.iter().find(|r| r.name == run) {
        for (k, (state, _)) in &r.params {
            match state {
                State::Locked => locked.push(k.clone()),
                State::Default => defaults.push(k.clone()),
                State::Required => required.push(k.clone()),
            }
        }
    }
    (locked, defaults, required)
}

/// The case's lock, read from where the index says it is.
pub fn lock_of(case: &Value, workloads: &Path) -> Result<Lock> {
    let rel = str_of(case, "lock");
    if !rel.ends_with(LOCK_FILE) {
        bail!("{}: the index names no lock for it", name(case));
    }
    Lock::load(&workloads.join(rel))
}

fn listed(names: &[String], what: &str) -> String {
    if names.is_empty() {
        "none".to_string()
    } else if what.is_empty() {
        names.join(", ")
    } else {
        format!("{} ({what})", names.join(", "))
    }
}

/// One case: where its records are, its attributes (in the dictionary's order), and per run
/// what it does, what it is, its attributes, and what its lock fixes, leaves, and requires;
/// then the commands that write its data and run it at its first GPU count.
pub fn render_one(case: &Value, lock: &Lock, index: &Index) -> String {
    let n = name(case);
    let tags = |a: &Value| {
        index
            .attributes
            .iter()
            .filter_map(|attr| match a.get(&attr.name) {
                Some(Value::String(s)) => Some(format!("{}={s}", attr.name)),
                Some(Value::Array(v)) => Some(format!("{}={}", attr.name, v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","))),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let row = |label: &str, value: &str| format!("  {label:<13}{value}\n");
    let g = gpus(case);
    let mut out = format!("case {n}\n");
    out += &row("dir", str_of(case, "dir"));
    out += &row("lock", str_of(case, "lock"));
    out += &row("description", str_of(case, "description"));
    out += &row("gpus", &g.iter().map(i64::to_string).collect::<Vec<_>>().join(", "));
    out += &row("attributes", &tags(&case["attributes"]));
    let runs = case["runs"].as_array().cloned().unwrap_or_default();
    let mut commands = Vec::new();
    for r in &runs {
        let ast = str_of(r, "ast");
        let (locked, defaults, required) = states(lock, ast);
        out += &format!("run {ast}\n");
        out += &row("does", str_of(r, "does"));
        out += &row("is", str_of(r, "is"));
        out += &row("attributes", &tags(&r["attributes"]));
        out += &row("locked", &listed(&locked, ""));
        out += &row("defaults", &listed(&defaults, "a run may give another with --param NAME=VALUE"));
        out += &row("required", &listed(&required, "a submission gives each with --param NAME=VALUE"));
        let which = if runs.len() > 1 { format!(" --case-run {ast}") } else { String::new() };
        let params: String = required.iter().map(|k| format!(" --param {k}=VALUE")).collect();
        commands.push(format!("  aeiou datagen --case {n}{which}{params} --root DIR"));
        commands.push(format!("  aeiou run --case {n}{which} --gpus {}{params} --root DIR", g.first().copied().unwrap_or(1)));
    }
    out += "commands\n";
    for c in commands {
        out += &c;
        out += "\n";
    }
    out
}

/// The vocabulary for `aeiou cases --help`: every attribute and its values, without their
/// meanings, which CATALOG.md has; or why there is none.
pub fn vocabulary_help(workloads: Result<PathBuf>) -> String {
    let index = workloads.and_then(|w| Index::load(&w));
    match index {
        Ok(index) => {
            let catalog = index.path.with_file_name("CATALOG.md");
            let mut out = format!("Attributes and their values (from {}; what each means is in {}):\n", index.path.display(), catalog.display());
            for a in &index.attributes {
                out += &format!("  {}: {}\n", a.name, a.values.join(", "));
            }
            out
        }
        Err(e) => format!("Attributes and their values: no catalog.json to read them from ({e:#}); give --workloads DIR"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(attributes: &[(&str, &[&str])]) -> Index {
        Index {
            path: PathBuf::from("w/catalog.json"),
            attributes: attributes.iter().map(|(n, v)| Attribute { name: n.to_string(), values: v.iter().map(|s| s.to_string()).collect() }).collect(),
            cases: Vec::new(),
        }
    }

    /// A lock that fixes a value and requires another: the case shows each by its state, and
    /// its commands give the required one with --param; several GPU counts are listed, the
    /// commands at the first.
    #[test]
    fn one_case_shows_each_state_and_gives_what_is_required() {
        let lock = Lock::parse(r#"{"lock_version": 1, "name": "m/c", "runs": [{"name": "a", "abstract": "x/a.ast.json", "ast_sha256": "0",
            "params": {"seed_bytes": {"locked": 4}, "steps": {"default": 10}, "step_time": {"required": true}}}]}"#, "m/c/case.lock.json").unwrap();
        assert_eq!(states(&lock, "a"), (vec!["seed_bytes".into()], vec!["steps".into()], vec!["step_time".into()]));
        assert_eq!(states(&lock, "b"), (vec![], vec![], vec![]));
        let case = serde_json::json!({"name": "m/c", "dir": "k/m/cases/c", "lock": "k/m/cases/c/case.lock.json", "description": "k/m/cases/c/DESCRIPTION.md",
            "gpus": [8, 16], "attributes": {"kind": "use", "phase": ["training", "inference"]}, "runs": [{"ast": "a", "does": "d", "is": "i", "attributes": {}}]});
        let out = render_one(&case, &lock, &index(&[("phase", &["training", "inference"]), ("kind", &["use"])]));
        for want in ["  gpus         8, 16", "  attributes   phase=training,inference kind=use", "  locked       seed_bytes",
                     "  required     step_time (a submission gives each with --param NAME=VALUE)",
                     "  aeiou run --case m/c --gpus 8 --param step_time=VALUE --root DIR"] {
            assert!(out.lines().any(|l| l == want), "no {want:?} in:\n{out}");
        }
        assert_eq!(gpus_shown(&[8, 16]), "8 or 16 GPUs");
        assert_eq!(gpus_shown(&[1]), "1 GPU");
        // a row with no application nor class: the tags it has
        let list = render_list(&[&case], 1, &index(&[]));
        assert!(list.contains("\nm/c  use, 8 or 16 GPUs\n    d\n"), "{list}");
        // the index names no lock
        let bad = serde_json::json!({"name": "m/c", "lock": "k/m/cases/c/other.json"});
        assert!(format!("{:#}", lock_of(&bad, Path::new("w")).unwrap_err()).contains("m/c: the index names no lock for it"));
    }

    /// An index that is JSON of the right version but not the index's shape is refused as such.
    #[test]
    fn an_index_of_another_shape_is_refused() {
        let dir = std::env::temp_dir().join(format!("aeiou-listing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for text in [r#"{"catalog_version": 1, "attributes": {}, "cases": []}"#, r#"{"catalog_version": 1, "attributes": [{"values": []}], "cases": []}"#,
                     r#"{"catalog_version": 1, "attributes": [], "cases": [{"dir": "x"}]}"#, r#"{"attributes": [], "cases": []}"#] {
            std::fs::write(dir.join(INDEX_FILE), text).unwrap();
            let e = format!("{:#}", Index::load(&dir).unwrap_err());
            assert!(e.contains("not the catalog's index") || e.contains("catalog_version missing"), "{text}: {e}");
        }
        std::fs::write(dir.join(INDEX_FILE), r#"{"catalog_version": 1, "attributes": [], "cases": [{"name": "m/c"}]}"#).unwrap();
        assert_eq!(Index::load(&dir).unwrap().names(), vec!["m/c".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Near names: by containment from three characters, by two edits at most, eight at most;
    /// a mechanism/case name given is matched by its case.
    #[test]
    fn near_names_are_few_and_close() {
        let names: Vec<String> = (0..12).map(|i| format!("m/case-{i}")).chain(["n/other".to_string()]).collect();
        assert_eq!(near(&names, "case").len(), 8);
        assert_eq!(near(&names, "othr"), vec!["n/other".to_string()]);
        assert_eq!(near(&names, "x/othr"), vec!["n/other".to_string()]);
        assert!(near(&names, "zz").is_empty());
        assert_eq!(edits("kitten", "sitting"), 3);
        assert_eq!(edits("", "abc"), 3);
        assert!(matches_name("m/c", "c") && matches_name("m/c", "m/c") && !matches_name("m/c", "m"));
    }

    /// A filter's forms, and what a case's values are: its own, then every run's.
    #[test]
    fn filters_parse_and_hold() {
        let idx = index(&[("kind", &["use", "example"]), ("api", &["sync", "mmap"])]);
        let case = serde_json::json!({"attributes": {"kind": "use"}, "runs": [{"attributes": {"api": "sync"}}, {"attributes": {"api": "mmap"}}]});
        assert_eq!(values(&case, "api"), vec!["sync".to_string(), "mmap".to_string()]);
        assert!(Filter::parse("api=mmap", &idx).unwrap().holds(&case));
        assert!(!Filter::parse("api!=mmap", &idx).unwrap().holds(&case));
        assert!(Filter::parse("kind!=example", &idx).unwrap().holds(&case));
        for bad in ["k!ind=use", "kind=use=x", "!=use"] {
            assert!(format!("{:#}", Filter::parse(bad, &idx).unwrap_err()).contains("ATTR=VALUE keeps"), "{bad}");
        }
    }
}
