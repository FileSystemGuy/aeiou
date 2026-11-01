//! The application records (`applications/`, `DESIGN_REVIEW.md` §3.88; in Rust since §3.89 stage
//! 3): what `aeiou-trace digest` knows about the applications whose traces it reads, as data. One
//! directory per application: `application.toml` (name, display, kind, summary, home, source,
//! path signatures, catalog mechanisms) and `<version>.toml` per version range the facts were read
//! for (`from`, optional `until`, `read`, `sources`, the `[[option]]`s that change the storage
//! workload with the blind spots a choice causes, the `[[blind_spot]]`s that hold regardless, and
//! the `[[assert]]`s the record answers for the digest). `attributes.toml` is the vocabulary;
//! `load` refuses a record outside it, every problem listed. The three rules live in the digest:
//! the trace always wins, a record's assertion ranks below the owner's answer and says so in its
//! provenance, and a version no record was read for gets the nearest lower record as suggestions
//! (`Application::resolve`). The checkout's records are built into the crate (`build.rs`,
//! `shipped`); `--applications DIR` reads another tree.
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

use crate::digest::{candidates_of, FACT_NAMES, QUESTIONS};

mod built {
    include!(concat!(env!("OUT_DIR"), "/applications.rs"));
}

pub struct Opt {
    pub name: String,
    pub question: String,
    pub choices: Vec<String>,
    pub default: Option<String>,
    pub source: String,
    /// The blind spot a choice causes, by choice.
    pub blind_spots: BTreeMap<String, String>,
}

pub struct Spot {
    pub kind: String,
    pub note: Option<String>,
    pub source: String,
}

pub enum Claim {
    Fact(String, i64),
    Question(String, String),
}

pub struct Assert {
    pub template: String,
    pub claim: Claim,
    pub note: Option<String>,
    pub source: String,
    pub confidence: String,
    /// The option and the choice it holds under.
    pub when: Option<(String, String)>,
}

pub struct Record {
    /// The application's display name, which the record's provenance names.
    pub display: String,
    pub version: String,
    key: Vec<u64>,
    pub until: Option<String>,
    until_key: Option<Vec<u64>>,
    pub read: String,
    pub sources: Vec<String>,
    pub options: Vec<Opt>,
    pub blind_spots: Vec<Spot>,
    pub asserts: Vec<Assert>,
}

impl Record {
    fn covers(&self, key: &[u64]) -> bool {
        self.key.as_slice() <= key && self.until_key.as_ref().is_none_or(|u| key < u.as_slice())
    }

    pub fn provenance(&self) -> String {
        format!("{} {} record, read {}", self.display, self.version, self.read)
    }

    /// The record's assertions as the digest's answers, for every template one of whose paths
    /// matches the assertion's glob and whose `when` the chosen options satisfy, each with the
    /// record as its `by`.
    pub fn assertions(&self, options: &Map<String, Value>, templates: &[(String, Vec<String>)]) -> Vec<Value> {
        let mut out = vec![];
        for a in &self.asserts {
            if let Some((option, choice)) = &a.when {
                if options.get(option).and_then(Value::as_str) != Some(choice) {
                    continue;
                }
            }
            for (name, paths) in templates {
                if !paths.iter().any(|p| matches(p, &a.template)) {
                    continue;
                }
                let mut e = Map::new();
                e.insert("template".into(), json!(name));
                match &a.claim {
                    Claim::Fact(f, v) => {
                        e.insert("fact".into(), json!(f));
                        e.insert("value".into(), json!(v));
                    }
                    Claim::Question(q, ans) => {
                        e.insert("question".into(), json!(q));
                        e.insert("answer".into(), json!(ans));
                    }
                }
                if let Some(n) = &a.note {
                    e.insert("note".into(), json!(n));
                }
                e.insert("by".into(), json!(format!("{} ({} confidence)", self.provenance(), a.confidence)));
                e.insert("source".into(), json!(a.source));
                out.push(Value::Object(e));
            }
        }
        out
    }
}

pub struct Application {
    pub name: String,
    pub display: String,
    pub kind: String,
    pub summary: String,
    pub home: String,
    pub source: String,
    pub signatures: Vec<String>,
    pub mechanisms: Vec<String>,
    /// By version, lowest first.
    pub records: Vec<Record>,
}

impl Application {
    /// The signature that identifies this application in `path`, or None.
    pub fn matches(&self, path: &str) -> Option<&str> {
        self.signatures.iter().find(|s| matches(path, s)).map(String::as_str)
    }

    /// The record for a version and whether it claims it: the one whose range holds the version
    /// (exact); else the nearest lower record, as suggestions (not exact); none below the lowest;
    /// the newest record, not exact, when the version is unknown or not a number.
    pub fn resolve(&self, version: Option<&str>) -> (Option<&Record>, bool) {
        let Some(key) = version.and_then(version_key) else {
            return (self.records.last(), false);
        };
        let mut lower = None;
        for r in &self.records {
            if r.covers(&key) {
                return (Some(r), true);
            }
            if r.key <= key {
                lower = Some(r);
            }
        }
        (lower, false)
    }
}

/// The records and the vocabulary's blind spots, which a blind spot's note defaults to.
pub struct Records {
    pub apps: BTreeMap<String, Application>,
    spots: Vec<(String, String)>,
}

impl Records {
    /// The applications a path signature identifies, by name, each with the first signature that matched.
    pub fn identify<'a>(&self, paths: impl IntoIterator<Item = &'a str>) -> Vec<(&Application, &str)> {
        let mut found: BTreeMap<&str, (&Application, &str)> = BTreeMap::new();
        for p in paths {
            for (name, app) in &self.apps {
                if found.contains_key(name.as_str()) {
                    continue;
                }
                if let Some(sig) = app.matches(p) {
                    found.insert(name, (app, sig));
                }
            }
        }
        found.into_values().collect()
    }

    /// The blind spots that hold for the chosen options: the record's unconditional ones, then
    /// those the choices cause, each with the vocabulary's description as `note` when the record
    /// gives none.
    pub fn blind_spots(&self, r: &Record, options: &Map<String, Value>) -> Vec<Value> {
        let described = |kind: &str| self.spots.iter().find(|(k, _)| k == kind).map_or(String::new(), |(_, d)| d.clone());
        let mut out: Vec<Value> = r.blind_spots.iter().map(|b| json!({"kind": b.kind, "note": b.note.clone().unwrap_or_else(|| described(&b.kind)), "source": b.source})).collect();
        for o in &r.options {
            let Some(choice) = options.get(&o.name).and_then(Value::as_str).filter(|c| !c.is_empty()) else { continue };
            if let Some(spot) = o.blind_spots.get(choice) {
                out.push(json!({"kind": spot, "note": described(spot), "option": o.name, "choice": choice, "source": o.source}));
            }
        }
        out
    }
}

/// A signature matches the path, or its part after any `/` (so `__*_*.distcp` matches a shard at any depth).
pub fn matches(path: &str, sig: &str) -> bool {
    crate::strace::glob(path, sig) || crate::strace::glob(path, &format!("*/{sig}"))
}

/// `3.0.3` -> [3, 0, 3]; a leading `v` and anything after the dotted numbers are dropped; None when there is no number.
pub fn version_key(v: &str) -> Option<Vec<u64>> {
    let v = v.trim();
    let v = v.strip_prefix('v').unwrap_or(v);
    let mut key = vec![];
    let mut rest = v;
    loop {
        let n = rest.bytes().take_while(u8::is_ascii_digit).count();
        if n == 0 {
            break;
        }
        key.push(rest[..n].parse().ok()?);
        rest = &rest[n..];
        match rest.strip_prefix('.') {
            Some(r) if r.starts_with(|c: char| c.is_ascii_digit()) => rest = r,
            _ => break,
        }
    }
    (!key.is_empty()).then_some(key)
}

/// The records built into the crate: the checkout's `applications/` when it was built.
pub fn shipped() -> Records {
    let file = |rel: &str| built::SHIPPED.iter().find(|(p, _)| *p == rel).map(|(_, t)| Ok(t.to_string()));
    let mut dirs: BTreeMap<String, Vec<(String, std::result::Result<String, String>)>> = BTreeMap::new();
    for (p, _) in built::SHIPPED {
        if let Some((d, f)) = p.split_once('/') {
            dirs.entry(d.to_string()).or_default().push((f.to_string(), file(p).unwrap()));
        }
    }
    let tree = Tree { attributes: file("attributes.toml").unwrap_or_else(|| Err("not built in".into())), dirs: dirs.into_iter().collect() };
    load_tree(tree).expect("the built-in application records load (tests/applications.rs)")
}

/// Every application under `root`, checked against `root/attributes.toml`; the error lists every problem, one a line.
pub fn load(root: &Path) -> Result<Records> {
    if !root.is_dir() {
        return Err(anyhow!("{}: no such directory of application records", root.display()));
    }
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| e.to_string());
    let mut entries: Vec<_> = std::fs::read_dir(root)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    entries.sort();
    let mut dirs = vec![];
    for d in entries {
        let mut files: Vec<_> = std::fs::read_dir(&d)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        let name = d.file_name().unwrap().to_string_lossy().into_owned();
        dirs.push((name, files.iter().map(|f| (f.file_name().unwrap().to_string_lossy().into_owned(), read(f))).collect()));
    }
    load_tree(Tree { attributes: read(&root.join("attributes.toml")), dirs }).map_err(|problems| anyhow!("{}", problems.join("\n")))
}

/// A tree of records as read: the vocabulary, and each directory's files by name with their text or why it was not read.
struct Tree {
    attributes: std::result::Result<String, String>,
    dirs: Vec<(String, Vec<(String, std::result::Result<String, String>)>)>,
}

#[derive(serde::Deserialize)]
struct Vocabulary {
    #[serde(default)]
    kind: BTreeMap<String, toml::Spanned<String>>,
    #[serde(default)]
    confidence: BTreeMap<String, toml::Spanned<String>>,
    #[serde(default)]
    blind_spot: BTreeMap<String, toml::Spanned<String>>,
}

/// A vocabulary's words in the order the file defines them.
fn in_order(m: BTreeMap<String, toml::Spanned<String>>) -> Vec<(String, String)> {
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by_key(|(_, s)| s.span().start);
    v.into_iter().map(|(k, s)| (k, s.into_inner())).collect()
}

/// A TOML error on one line, where it is.
fn toml_error(text: &str, e: &toml::de::Error) -> String {
    let mut s = format!("TOML: {}", e.message().trim_end());
    if let Some(span) = e.span() {
        let before = &text[..span.start.min(text.len())];
        let line = before.matches('\n').count() + 1;
        let col = before.len() - before.rfind('\n').map_or(0, |i| i + 1) + 1;
        s += &format!(" (at line {line}, column {col})");
    }
    s.replace('\n', " ")
}

fn parse<T: serde::de::DeserializeOwned>(path: &str, text: &std::result::Result<String, String>, problems: &mut Vec<String>) -> Option<T> {
    match text {
        Err(why) => {
            problems.push(format!("{path}: {why}"));
            None
        }
        Ok(t) => match toml::from_str(t) {
            Ok(v) => Some(v),
            Err(e) => {
                problems.push(format!("{path}: {}", toml_error(t, &e)));
                None
            }
        },
    }
}

/// A value as the Python's `str` wrote it in a problem: a string bare, nothing as None.
fn shown(v: Option<&toml::Value>) -> String {
    match v {
        None => "None".into(),
        Some(toml::Value::String(s)) => s.clone(),
        Some(other) => format!("{other:?}"),
    }
}

fn text_of<'a>(t: &'a toml::Table, key: &str) -> Option<&'a str> {
    t.get(key).and_then(toml::Value::as_str).filter(|s| !s.is_empty())
}

fn strings(v: Option<&toml::Value>) -> Option<Vec<String>> {
    match v {
        None => Some(vec![]),
        Some(toml::Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string)).collect(),
        _ => None,
    }
}

fn tables(t: &toml::Table, key: &str) -> Vec<toml::Table> {
    match t.get(key) {
        Some(toml::Value::Array(a)) => a.iter().map(|x| x.as_table().cloned().unwrap_or_default()).collect(),
        _ => vec![],
    }
}

fn load_tree(tree: Tree) -> std::result::Result<Records, Vec<String>> {
    let mut problems = vec![];
    let Some(vocab) = parse::<Vocabulary>("attributes.toml", &tree.attributes, &mut problems) else { return Err(problems) };
    let (kinds, conf, spots) = (in_order(vocab.kind), in_order(vocab.confidence), in_order(vocab.blind_spot));
    let words = |v: &[(String, String)]| v.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
    let (kinds, conf_words) = (words(&kinds), words(&conf));
    let spot_words = words(&spots);
    let mut apps = BTreeMap::new();
    for (dir, files) in &tree.dirs {
        let Some((_, app_text)) = files.iter().find(|(f, _)| f == "application.toml") else {
            problems.push(format!("{dir}: no application.toml"));
            continue;
        };
        let app_path = format!("{dir}/application.toml");
        let Some(a) = parse::<toml::Table>(&app_path, app_text, &mut problems) else { continue };
        let Some(mut app) = application(dir, &a, &kinds, &app_path, &mut problems) else { continue };
        for (f, text) in files {
            if f == "application.toml" {
                continue;
            }
            let path = format!("{dir}/{f}");
            let Some(stem) = f.strip_suffix(".toml") else {
                problems.push(format!("{path}: not application.toml or a <version>.toml"));
                continue;
            };
            let Some(r) = parse::<toml::Table>(&path, text, &mut problems) else { continue };
            if let Some(rec) = record(&app.display, stem, &path, &r, &spot_words, &conf_words, &mut problems) {
                app.records.push(rec);
            }
        }
        app.records.sort_by(|a, b| a.key.cmp(&b.key));
        if app.records.is_empty() {
            problems.push(format!("{dir}: no <version>.toml"));
        }
        apps.insert(app.name.clone(), app);
    }
    if problems.is_empty() {
        Ok(Records { apps, spots })
    } else {
        Err(problems)
    }
}

fn application(dir: &str, a: &toml::Table, kinds: &[String], path: &str, problems: &mut Vec<String>) -> Option<Application> {
    let mut ok = true;
    for key in ["name", "display", "kind", "summary"] {
        if text_of(a, key).is_none() {
            problems.push(format!("{path}: {key} is required"));
            ok = false;
        }
    }
    if ok && text_of(a, "name") != Some(dir) {
        problems.push(format!("{path}: name must be the directory's, {dir}"));
        ok = false;
    }
    if ok && !kinds.iter().any(|k| Some(k.as_str()) == text_of(a, "kind")) {
        problems.push(format!("{path}: kind must be one of {}", kinds.join(", ")));
        ok = false;
    }
    let mut lists = [vec![], vec![]];
    for (i, key) in ["signatures", "mechanisms"].into_iter().enumerate() {
        match strings(a.get(key)) {
            Some(v) => lists[i] = v,
            None => {
                problems.push(format!("{path}: {key} must be a list of strings"));
                ok = false;
            }
        }
    }
    if !ok {
        return None;
    }
    let [signatures, mechanisms] = lists;
    let s = |k: &str| a.get(k).and_then(toml::Value::as_str).unwrap_or("").to_string();
    Some(Application { name: s("name"), display: s("display"), kind: s("kind"), summary: s("summary"), home: s("home"), source: s("source"), signatures, mechanisms, records: vec![] })
}

fn record(display: &str, stem: &str, path: &str, r: &toml::Table, spots: &[String], conf: &[String], problems: &mut Vec<String>) -> Option<Record> {
    let mut ok = true;
    for key in ["from", "read"] {
        if text_of(r, key).is_none() {
            problems.push(format!("{path}: {key} is required"));
            ok = false;
        }
    }
    let sources = strings(r.get("sources")).filter(|s| !s.is_empty() && r.contains_key("sources"));
    if sources.is_none() {
        problems.push(format!("{path}: sources must be a non-empty list of strings"));
        ok = false;
    }
    if !ok {
        return None;
    }
    let from = text_of(r, "from").unwrap();
    let Some(key) = (from == stem).then(|| version_key(from)).flatten() else {
        problems.push(format!("{path}: from must be the file's stem, {stem}, a dotted version"));
        return None;
    };
    let until = r.get("until");
    let until_key = until.and_then(toml::Value::as_str).and_then(version_key);
    if until.is_some() && until_key.as_ref().is_none_or(|u| *u <= key) {
        problems.push(format!("{path}: until must be above from"));
        ok = false;
    }
    let is_conf = |t: &toml::Table| t.get("confidence").and_then(toml::Value::as_str).is_some_and(|c| conf.iter().any(|w| w == c));
    let conf_words = conf.join(" or ");
    let mut options = vec![];
    for (i, o) in tables(r, "option").iter().enumerate() {
        let (Some(name), Some(question)) = (o.get("name").and_then(toml::Value::as_str), o.get("question").and_then(toml::Value::as_str)) else {
            problems.push(format!("{path}: option {}: name and question are required", i + 1));
            ok = false;
            continue;
        };
        let Some(choices) = strings(o.get("choices")).filter(|c| c.len() >= 2) else {
            problems.push(format!("{path}: option {name}: choices must be a list of at least two strings"));
            ok = false;
            continue;
        };
        let default = o.get("default").map(|d| shown(Some(d)));
        if default.as_ref().is_some_and(|d| !choices.contains(d)) {
            problems.push(format!("{path}: option {name}: default must be one of its choices"));
            ok = false;
        }
        if !is_conf(o) {
            problems.push(format!("{path}: option {name}: confidence must be {conf_words}"));
            ok = false;
        }
        let mut caused = BTreeMap::new();
        for (choice, spot) in o.get("blind_spots").and_then(toml::Value::as_table).into_iter().flatten() {
            let spot = shown(Some(spot));
            if !choices.contains(choice) {
                problems.push(format!("{path}: option {name}: blind spot for a choice it does not have, {choice}"));
                ok = false;
            } else if !spots.contains(&spot) {
                problems.push(format!("{path}: option {name}: blind spot {spot} is not in applications/attributes.toml"));
                ok = false;
            }
            caused.insert(choice.clone(), spot);
        }
        let source = o.get("source").and_then(toml::Value::as_str).unwrap_or("").to_string();
        options.push(Opt { name: name.into(), question: question.into(), choices, default, source, blind_spots: caused });
    }
    let mut blind_spots = vec![];
    for (i, b) in tables(r, "blind_spot").iter().enumerate() {
        let kind = shown(b.get("kind"));
        if !b.get("kind").and_then(toml::Value::as_str).is_some_and(|k| spots.iter().any(|s| s == k)) {
            problems.push(format!("{path}: blind spot {kind} is not in applications/attributes.toml"));
            ok = false;
        }
        if !is_conf(b) {
            problems.push(format!("{path}: blind spot {}: confidence must be {conf_words}", i + 1));
            ok = false;
        }
        let note = b.get("note").and_then(toml::Value::as_str).filter(|n| !n.is_empty()).map(str::to_string);
        blind_spots.push(Spot { kind, note, source: b.get("source").and_then(toml::Value::as_str).unwrap_or("").to_string() });
    }
    let mut asserts = vec![];
    for (i, a) in tables(r, "assert").iter().enumerate() {
        let n = i + 1;
        let template = a.get("template").and_then(toml::Value::as_str);
        if template.is_none() {
            problems.push(format!("{path}: assert {n}: template is required"));
            ok = false;
        }
        let claim = if let Some(f) = a.get("fact") {
            let fact = f.as_str().filter(|f| FACT_NAMES.contains(f) || f.starts_with("values:"));
            if fact.is_none() {
                problems.push(format!("{path}: assert {n}: fact must be one of {}, or values:<field>", FACT_NAMES.join(", ")));
                ok = false;
            }
            let value = a.get("value").and_then(toml::Value::as_integer).filter(|v| *v >= 1);
            if value.is_none() {
                problems.push(format!("{path}: assert {n}: value must be a whole number of at least 1"));
                ok = false;
            }
            fact.zip(value).map(|(f, v)| Claim::Fact(f.into(), v))
        } else if let Some(q) = a.get("question") {
            match q.as_str().and_then(|q| candidates_of(q).map(|c| (q, c))) {
                None => {
                    problems.push(format!("{path}: assert {n}: question must be one of {}", QUESTIONS.join(", ")));
                    ok = false;
                    None
                }
                Some((q, cands)) => {
                    let answer = a.get("answer").and_then(toml::Value::as_str).filter(|x| cands.contains(x));
                    if answer.is_none() {
                        problems.push(format!("{path}: assert {n}: answer must be one of {}", cands.join(", ")));
                        ok = false;
                    }
                    answer.map(|x| Claim::Question(q.into(), x.into()))
                }
            }
        } else {
            problems.push(format!("{path}: assert {n}: either fact and value or question and answer"));
            ok = false;
            None
        };
        if !is_conf(a) {
            problems.push(format!("{path}: assert {n}: confidence must be {conf_words}"));
            ok = false;
        }
        let mut when = None;
        if let Some(w) = a.get("when") {
            let (opt, choice) = (w.get("option"), w.get("choice"));
            match options.iter().find(|o| Some(o.name.as_str()) == opt.and_then(toml::Value::as_str)) {
                None => {
                    problems.push(format!("{path}: assert {n}: when names no option of the record, {}", shown(opt)));
                    ok = false;
                }
                Some(o) if !choice.and_then(toml::Value::as_str).is_some_and(|c| o.choices.iter().any(|x| x == c)) => {
                    problems.push(format!("{path}: assert {n}: when names no choice of option {}, {}", o.name, shown(choice)));
                    ok = false;
                }
                Some(o) => when = Some((o.name.clone(), shown(choice))),
            }
        }
        if let (Some(template), Some(claim)) = (template, claim) {
            let note = a.get("note").and_then(toml::Value::as_str).filter(|n| !n.is_empty()).map(str::to_string);
            let source = a.get("source").and_then(toml::Value::as_str).unwrap_or("").to_string();
            let confidence = a.get("confidence").and_then(toml::Value::as_str).unwrap_or("").to_string();
            asserts.push(Assert { template: template.into(), claim, note, source, confidence, when });
        }
    }
    ok.then(|| Record {
        display: display.to_string(),
        version: from.to_string(),
        key,
        until: until.and_then(toml::Value::as_str).map(str::to_string),
        until_key,
        read: text_of(r, "read").unwrap().to_string(),
        sources: sources.unwrap(),
        options,
        blind_spots,
        asserts,
    })
}
