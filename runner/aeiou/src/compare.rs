//! `aeiou-trace compare` (`DESIGN_REVIEW.md` §3.89, stage 4; `builder/REFERENCE.md` §7): two
//! observation documents, the application's (A) and the abstract's (B), held side by side.
//! The numbers are the metrics of `runner/REFERENCE.md` §10, a metrics document's or a
//! digest's `metrics` section, compared row by row: a share's difference, a histogram's
//! cumulative distance, an exact distribution's total-variation distance, each in [0, 1].
//! `--judge` holds each row to the tolerance of its class (`TOLERANCES`, §3.55), raised by
//! what the abstract differs from itself by at other seeds (`--self`); a tolerance file
//! (`aeiou_tolerances: 1`) replaces classes, names the rows the trace cannot show, and records
//! the rows known to be outside. The rows, their wording, and the verdicts are the Python's of
//! before the port (`tests/compare/`, the golden corpus).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::digest::py;
use crate::usage;

/// The metrics documents read: version 1, a capture's from before 2026-10-22 (blocks only,
/// `DESIGN_REVIEW.md` §3.79), and version 2.
pub const METRICS_READ: [u64; 2] = [1, 2];
pub const TOLERANCES_FORMAT: u64 = 1;
const KINDS: [&str; 2] = ["read", "write"];

/// The largest distance at which a row still counts as a match, by the class of the row (the
/// longest of these prefixes of its name). A tolerance file may replace any of them;
/// `DESIGN_REVIEW.md` §3.55 has where they come from.
pub const TOLERANCES: [(&str, f64); 11] = [
    ("mix", 0.05),
    ("ops", 0.05),
    ("request size", 0.05),
    ("run length", 0.10),
    ("reuse", 0.10),
    ("reuse distance", 0.10),
    ("extent reuse", 0.10),
    ("extent reuse distance", 0.10),
    ("popularity", 0.05),
    ("fan-out", 0.10),
    ("depth", 0.10),
];

/// A popularity row is the share of accesses that go to the top fraction of the units; it
/// says something only when that fraction is at least this many units on both sides.
pub const POPULARITY_MIN_UNITS: f64 = 10.0;

/// One row: the metric, A's and B's values as shown, and their distance (`None` when both
/// sides are empty).
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub metric: String,
    pub a: String,
    pub b: String,
    pub distance: Option<f64>,
}

/// A row judged: the tolerance it was held to, the verdict (`ok`, `outside`, `unseen`, `not
/// judged`), and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Judged {
    pub row: Row,
    pub allowed: Option<f64>,
    pub verdict: &'static str,
    pub note: String,
}

// ---------------------------------------------------------------- the rows

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| v.as_f64().unwrap_or(0.0) as i64)
}

fn share(x: f64, y: f64) -> f64 {
    if y != 0.0 { x / y } else { 0.0 }
}

fn buckets(h: &Value) -> Vec<(i64, i64)> {
    h["buckets"].as_array().map(|b| b.iter().map(|p| (int(&p[0]), int(&p[1]))).collect()).unwrap_or_default()
}

/// The largest difference between two histograms' cumulative shares over the bucket bounds of
/// both (the Kolmogorov-Smirnov statistic on the bucketed values).
fn cdf_distance(a: &Value, b: &Value) -> Option<f64> {
    let (na, nb) = (int(&a["n"]), int(&b["n"]));
    if na == 0 || nb == 0 {
        return if na == nb { None } else { Some(1.0) };
    }
    let (da, db): (BTreeMap<i64, i64>, BTreeMap<i64, i64>) = (buckets(a).into_iter().collect(), buckets(b).into_iter().collect());
    let bounds: BTreeSet<i64> = da.keys().chain(db.keys()).copied().collect();
    let (mut ca, mut cb, mut worst) = (0i64, 0i64, 0.0f64);
    for bound in bounds {
        ca += da.get(&bound).copied().unwrap_or(0);
        cb += db.get(&bound).copied().unwrap_or(0);
        worst = worst.max((ca as f64 / na as f64 - cb as f64 / nb as f64).abs());
    }
    Some(worst)
}

fn quantile(h: &Value, p: f64) -> i64 {
    let want = (int(&h["n"]) as f64 * p).ceil().max(1.0);
    let mut seen = 0i64;
    for (bound, w) in buckets(h) {
        seen += w;
        if seen as f64 >= want {
            return bound;
        }
    }
    0
}

/// Total-variation distance between two exact distributions (fan-out, depth).
fn tv_distance(a: &BTreeMap<i64, i64>, b: &BTreeMap<i64, i64>) -> Option<f64> {
    let (na, nb): (i64, i64) = (a.values().sum(), b.values().sum());
    if na == 0 || nb == 0 {
        return if na == nb { None } else { Some(1.0) };
    }
    let keys: BTreeSet<i64> = a.keys().chain(b.keys()).copied().collect();
    let sum: f64 = keys.iter().map(|k| (*a.get(k).unwrap_or(&0) as f64 / na as f64 - *b.get(k).unwrap_or(&0) as f64 / nb as f64).abs()).sum();
    Some(0.5 * sum)
}

/// A size in its largest unit: `4 KiB`, `1.5 MiB`.
fn size(v: i64) -> String {
    let mut v = v as f64;
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"] {
        if v < 1024.0 || unit == "TiB" {
            return if unit == "B" || v == v.trunc() { format!("{v:.0} {unit}") } else { format!("{v:.1} {unit}") };
        }
        v /= 1024.0;
    }
    unreachable!()
}

fn has(t: &Value, key: &str) -> bool {
    t.get(key).is_some()
}

/// Rows `(metric, a, b, distance)` for two `total`s (or two actor templates' metrics).
pub fn rows(a: &Value, b: &Value) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    let pct = |rows: &mut Vec<Row>, name: String, va: f64, vb: f64| {
        rows.push(Row { metric: name, a: format!("{:.1} %", 100.0 * va), b: format!("{:.1} %", 100.0 * vb), distance: Some((va - vb).abs()) });
    };
    let count = |t: &Value, k: &str| num(&t["counts"][k]);
    let data = |t: &Value| count(t, "read") + count(t, "write");
    pct(&mut rows, "mix: data ops / all ops".into(), share(data(a), num(&a["ops"])), share(data(b), num(&b["ops"])));
    pct(&mut rows, "mix: reads / data ops".into(), share(count(a, "read"), data(a)), share(count(b, "read"), data(b)));
    let rb = |t: &Value| share(num(&t["bytes_read"]), num(&t["bytes_read"]) + num(&t["bytes_written"]));
    pct(&mut rows, "mix: read bytes / data bytes".into(), rb(a), rb(b));
    let kinds: BTreeSet<&String> = [a, b].iter().filter_map(|t| t["counts"].as_object()).flat_map(|m| m.keys()).collect();
    for kind in kinds {
        pct(&mut rows, format!("ops: {kind} / all ops"), share(count(a, kind), num(&a["ops"])), share(count(b, kind), num(&b["ops"])));
    }

    let hist = |rows: &mut Vec<Row>, name: String, ha: &Value, hb: &Value, plain: bool| {
        if int(&ha["n"]) == 0 && int(&hb["n"]) == 0 {
            return;
        }
        let show = |h: &Value| {
            if int(&h["n"]) == 0 {
                return "none".to_string();
            }
            [0.5, 0.9, 0.99].iter().map(|p| quantile(h, *p)).map(|q| if plain { q.to_string() } else { size(q) }).collect::<Vec<_>>().join(" / ")
        };
        rows.push(Row { metric: format!("{name} (p50 / p90 / p99)"), a: show(ha), b: show(hb), distance: cdf_distance(ha, hb) });
    };
    for k in KINDS {
        hist(&mut rows, format!("request size, {k}"), &a["request_size"][k], &b["request_size"][k], false);
    }
    for (k, total) in [("read", "bytes_read"), ("write", "bytes_written")] {
        let (ra, rb) = (&a["run_length"][k], &b["run_length"][k]);
        hist(&mut rows, format!("run length, {k}, bytes"), &ra["bytes"], &rb["bytes"], false);
        hist(&mut rows, format!("run length, {k}, ops"), &ra["ops"], &rb["ops"], true);
        if num(&a[total]) != 0.0 || num(&b[total]) != 0.0 {
            pct(&mut rows, format!("run length, {k}: bytes in multi-op runs"), share(num(&ra["multi_op_bytes"]), num(&a[total])), share(num(&rb["multi_op_bytes"]), num(&b[total])));
        }
    }
    // the locality sections both have (DESIGN_REVIEW.md §3.79): extents, and blocks, which a
    // capture taken before 2026-10-22 has alone
    let sections: Vec<(&str, &str, bool)> = [("reuse_extents", "extent reuse", false), ("reuse_distance_bytes", "reuse", true)]
        .into_iter()
        .filter(|(key, _, _)| has(a, key) && has(b, key))
        .collect();
    if sections.is_empty() {
        return Err(anyhow!(
            "no locality metrics in common: one document has extents only and the other blocks only (take the dry run with --metrics-block, the capture's block size)"
        ));
    }
    for (key, prefix, blocks) in sections {
        let (ra, rb) = (&a[key], &b[key]);
        for this in KINDS {
            let per = if blocks { format!("block {this}s") } else { format!("bytes {}", if this == "read" { "read" } else { "written" }) };
            let accesses = |r: &Value| num(&r["first_touch"][this]) + KINDS.iter().map(|prev| num(&r[format!("{this}_after_{prev}")]["n"])).sum::<f64>();
            if accesses(ra) != 0.0 || accesses(rb) != 0.0 {
                pct(&mut rows, format!("{prefix}: first touches / {per}"), share(num(&ra["first_touch"][this]), accesses(ra)), share(num(&rb["first_touch"][this]), accesses(rb)));
            }
            for prev in KINDS {
                let k = format!("{this}_after_{prev}");
                if int(&ra[&k]["n"]) != 0 || int(&rb[&k]["n"]) != 0 {
                    pct(&mut rows, format!("{prefix}: {this} after {prev} / {per}"), share(num(&ra[&k]["n"]), accesses(ra)), share(num(&rb[&k]["n"]), accesses(rb)));
                }
                hist(&mut rows, format!("{prefix} distance, {this} after {prev}"), &ra[&k], &rb[&k], false);
            }
        }
        let again = |r: &Value| num(&r["whole"]) + num(&r["partial"]);
        if !blocks && (again(ra) != 0.0 || again(rb) != 0.0) {
            pct(&mut rows, format!("{prefix}: partial re-reads / bytes read again"), share(num(&ra["partial"]), again(ra)), share(num(&rb["partial"]), again(rb)));
        }
    }
    let units = ["bytes", "blocks"].into_iter().filter(|u| has(a, &format!("popularity_{u}")) && has(b, &format!("popularity_{u}"))).chain(["objects"]);
    for unit in units {
        let (pa, pb) = (&a[format!("popularity_{unit}")], &b[format!("popularity_{unit}")]);
        for (top, label) in [("top_0_1_pct", "0.1 %"), ("top_1_pct", "1 %"), ("top_10_pct", "10 %")] {
            pct(&mut rows, format!("popularity, {unit}: accesses to the top {label}"), num(&pa[top]), num(&pb[top]));
        }
    }
    for key in ["fan_out", "depth"] {
        let dist = |t: &Value| -> BTreeMap<i64, i64> {
            t[key].as_object().map(|m| m.iter().filter_map(|(k, v)| Some((k.parse().ok()?, int(v)))).collect()).unwrap_or_default()
        };
        let (da, db) = (dist(a), dist(b));
        if !da.is_empty() || !db.is_empty() {
            let show = |d: &BTreeMap<i64, i64>| {
                if d.is_empty() {
                    return "none".to_string();
                }
                let n: i64 = d.values().sum();
                d.iter().map(|(k, v)| format!("{k}:{:.0}%", 100.0 * *v as f64 / n as f64)).collect::<Vec<_>>().join(" ")
            };
            rows.push(Row { metric: key.replace('_', "-"), a: show(&da), b: show(&db), distance: tv_distance(&da, &db) });
        }
    }
    Ok(rows)
}

// ---------------------------------------------------------------- the tolerances

/// The class of a row: the longest class name its metric starts with.
pub fn tolerance_class(metric: &str) -> &'static str {
    TOLERANCES.iter().filter(|(c, _)| metric.starts_with(c)).max_by_key(|(c, _)| c.len()).map(|(c, _)| *c).unwrap_or("mix")
}

/// A JSON file, or its path and what was wrong with it.
pub fn read_json(path: &str) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| anyhow!("{path}: {e}"))
}

/// A tolerance file: `{"aeiou_tolerances": 1, "tolerances": {class: x}, "unseen": [{"metric":
/// prefix, "reason": text}], "outside": [the same]}`. `unseen` names rows the trace cannot show
/// (they are not judged); `outside` records rows known to be outside, with the reason (they
/// are judged, and still outside).
pub fn load_tolerances(path: &str) -> Result<Value> {
    let spec = read_json(path)?;
    if spec.get("aeiou_tolerances").and_then(Value::as_u64) != Some(TOLERANCES_FORMAT) {
        return Err(anyhow!("{path}: not an aeiou_tolerances: {TOLERANCES_FORMAT} document"));
    }
    let known = ["aeiou_tolerances", "tolerances", "unseen", "outside", "comment"];
    let mut extra: Vec<&String> = spec.as_object().unwrap().keys().filter(|k| !known.contains(&k.as_str())).collect();
    if !extra.is_empty() {
        extra.sort();
        return Err(anyhow!("{path}: unknown key(s) {}", extra.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
    }
    let classes: Vec<&str> = TOLERANCES.iter().map(|(c, _)| *c).collect();
    let bad_class = |c: &str| anyhow!("{path}: tolerances: `{c}` must be one of {} with a value in [0, 1]", classes.join(", "));
    match spec.get("tolerances") {
        None => {}
        Some(Value::Object(m)) => {
            for (c, x) in m {
                let ok = classes.contains(&c.as_str()) && x.is_number() && (0.0..=1.0).contains(&num(x));
                if !ok {
                    return Err(bad_class(c));
                }
            }
        }
        Some(other) => return Err(bad_class(&py(other))),
    }
    for key in ["unseen", "outside"] {
        let Some(list) = spec.get(key) else { continue };
        let entry_ok = |e: &Value| {
            e.as_object().is_some_and(|o| {
                o.len() == 2 && ["metric", "reason"].iter().all(|k| o.get(*k).and_then(Value::as_str).is_some_and(|s| !s.is_empty()))
            })
        };
        if !list.as_array().is_some_and(|l| l.iter().all(entry_ok)) {
            return Err(anyhow!("{path}: {key}: each entry is {{\"metric\": prefix, \"reason\": text}}"));
        }
    }
    Ok(spec)
}

fn entries<'s>(spec: Option<&'s Value>, key: &str) -> Vec<(&'s str, &'s str)> {
    spec.and_then(|s| s.get(key)).and_then(Value::as_array).map(|l| l.iter().map(|e| (e["metric"].as_str().unwrap_or(""), e["reason"].as_str().unwrap_or(""))).collect()).unwrap_or_default()
}

/// The rows judged, and the entries of the tolerance file that matched no row or recorded as
/// outside a row that is not. `allowed` is the class tolerance plus the largest distance of
/// that row between `b` and each of `selfs` (the same abstract at other seeds: what the
/// abstract differs from itself by). `docs` are the two documents, for whether the trace was
/// read with a chain gap (no depth to judge without one).
pub fn judge(rows: &[Row], a: &Value, b: &Value, docs: &[&Value], selfs: &[Value], spec: Option<&Value>) -> Result<(Vec<Judged>, Vec<String>)> {
    let mut tol: HashMap<&str, f64> = TOLERANCES.iter().copied().collect();
    if let Some(m) = spec.and_then(|s| s.get("tolerances")).and_then(Value::as_object) {
        for (c, x) in m {
            if let Some(k) = TOLERANCES.iter().find(|(k, _)| k == c) {
                tol.insert(k.0, num(x));
            }
        }
    }
    let mut spread: HashMap<String, f64> = HashMap::new();
    for s in selfs {
        for r in crate::compare::rows(b, s)? {
            let e = spread.entry(r.metric).or_insert(0.0);
            *e = e.max(r.distance.unwrap_or(0.0));
        }
    }
    let no_depth = docs.iter().any(|d| d.get("source").and_then(Value::as_str) == Some("strace") && d.get("chain_gap_us").map_or(true, Value::is_null));
    let (unseen, outside) = (entries(spec, "unseen"), entries(spec, "outside"));
    let mut used: HashSet<(&str, usize)> = HashSet::new();
    let mut listed = |key: &'static str, list: &[(&str, &str)], name: &str| -> Option<String> {
        for (i, (metric, reason)) in list.iter().enumerate() {
            if name.starts_with(metric) {
                used.insert((key, i));
                return Some(reason.to_string());
            }
        }
        None
    };
    let mut out = Vec::new();
    for r in rows {
        let name = r.metric.as_str();
        let allowed = tol[tolerance_class(name)] + spread.get(name).copied().unwrap_or(0.0);
        let judged = |allowed: Option<f64>, verdict: &'static str, note: String| Judged { row: r.clone(), allowed, verdict, note };
        if let Some(why) = listed("unseen", &unseen, name) {
            out.push(judged(None, "unseen", why));
            continue;
        }
        let Some(dist) = r.distance else {
            out.push(judged(None, "not judged", "nothing on either side".into()));
            continue;
        };
        if name.starts_with("popularity") {
            let unit = name.split(':').next().unwrap_or("").split(", ").nth(1).unwrap_or("");
            let top = name.rsplit_once("top ").map(|(_, t)| t).unwrap_or("");
            let frac = match top {
                "0.1 %" => 0.001,
                "1 %" => 0.01,
                _ => 0.1,
            };
            let key = format!("popularity_{unit}");
            let fewest = int(&a[&key]["distinct"]).min(int(&b[&key]["distinct"]));
            if (fewest as f64) * frac < POPULARITY_MIN_UNITS {
                out.push(judged(None, "not judged", format!("the top {top} of {fewest} {unit} is fewer than {}", POPULARITY_MIN_UNITS as i64)));
                continue;
            }
        }
        if name == "depth" && no_depth {
            out.push(judged(None, "not judged", "the trace was read without --chain-gap-us".into()));
            continue;
        }
        let why = listed("outside", &outside, name);
        if dist <= allowed {
            out.push(judged(Some(allowed), "ok", if why.is_some() { "recorded as outside, now within".into() } else { String::new() }));
        } else {
            out.push(judged(Some(allowed), "outside", why.unwrap_or_default()));
        }
    }
    let mut stale: Vec<String> = Vec::new();
    for (key, list) in [("unseen", &unseen), ("outside", &outside)] {
        for (i, (metric, _)) in list.iter().enumerate() {
            if !used.contains(&(key, i)) {
                stale.push(format!("{key}: `{metric}` matches no row"));
            }
        }
    }
    stale.extend(out.iter().filter(|j| j.verdict == "ok" && !j.note.is_empty()).map(|j| format!("outside: `{}` is recorded as outside and is within", j.row.metric)));
    Ok((out, stale))
}

// ---------------------------------------------------------------- the structure

/// One thing compared on a pair of templates: what A and B have, and the verdict: `same`,
/// `different`, `not compared` (a side undecided), or `shown` (never judged); `note` says why,
/// and `assertion` when a side's value is its owner's answer rather than the trace's verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct Aspect {
    pub name: String,
    pub a: String,
    pub b: String,
    pub verdict: &'static str,
    pub note: String,
}

/// Two templates paired, how (`identical`, `--pair`, `shape`), and what was compared.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    pub a: String,
    pub b: String,
    pub by: &'static str,
    pub aspects: Vec<Aspect>,
}

/// The structure of two digests compared (`DESIGN_REVIEW.md` §3.89): the templates paired,
/// those of A with no partner (each a difference), those of B with none (listed), and what of
/// B was set aside as the runner's own I/O.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Structure {
    pub pairs: Vec<Pair>,
    pub unpaired_a: Vec<String>,
    pub unpaired_b: Vec<String>,
    pub set_aside: Vec<String>,
    pub lanes: [i64; 2],
}

impl Structure {
    pub fn differences(&self) -> usize {
        self.pairs.iter().flat_map(|p| &p.aspects).filter(|x| x.verdict == "different").count() + self.unpaired_a.len()
    }
}

fn text_of(t: &Value) -> &str {
    t["template"].as_str().unwrap_or("")
}

/// What of a template is the runner's own I/O on the run side, never the workload's: a name
/// beginning `.aeiou` (V13 keeps them out of every workload), and the stat of the root.
fn runners_own(t: &Value) -> Option<String> {
    let text = text_of(t);
    if text.split('/').any(|c| c.starts_with(".aeiou")) {
        return Some(text.to_string());
    }
    let only_stat = t["ops"].as_object().is_some_and(|m| m.keys().all(|k| k == "stat"));
    ((text.is_empty() || text == ".") && only_stat).then(|| "the root's stat".to_string())
}

fn fields_of(t: &Value) -> Vec<(String, i64)> {
    t["fields"].as_array().map(|f| f.iter().map(|f| (py(&f["kind"]), int(&f["width"]))).collect()).unwrap_or_default()
}

fn show_fields(f: &[(String, i64)]) -> String {
    if f.is_empty() {
        return "none".into();
    }
    f.iter().map(|(k, w)| if *w > 0 { format!("{k}:{w}") } else { k.clone() }).collect::<Vec<_>>().join(" ")
}

fn sites_of(doc: &Value, id: i64) -> Vec<&Value> {
    doc["sites"].as_array().map(|s| s.iter().filter(|s| int(&s["template"]) == id).collect()).unwrap_or_default()
}

/// The phases of a template across the lanes: `(lane, first, last, motif)`.
fn phases_of(doc: &Value, id: i64) -> Vec<(i64, i64, i64, String)> {
    let mut out = Vec::new();
    for l in doc["lanes"].as_array().into_iter().flatten() {
        for p in l["phases"].as_array().into_iter().flatten().filter(|p| int(&p["template"]) == id) {
            out.push((int(&l["lane"]), int(&p["first"]), int(&p["last"]), py(&p["motif"])));
        }
    }
    out
}

/// The most lanes working on the template at once: each phase on it an interval of its lane,
/// an end before a start at the same instant.
fn peak_lanes(phases: &[(i64, i64, i64, String)]) -> i64 {
    let mut events: Vec<(i64, i32, i64)> = Vec::new();
    for (lane, first, last, _) in phases {
        events.push((*first, 1, *lane));
        events.push((*last, 0, *lane));
    }
    // at one instant, starts before ends: a phase of one call has its first equal to its last
    events.sort_by_key(|(t, kind, _)| (*t, std::cmp::Reverse(*kind)));
    let mut live: HashMap<i64, i64> = HashMap::new();
    let mut peak = 0;
    for (_, kind, lane) in events {
        let n = live.entry(lane).or_insert(0);
        if kind == 1 {
            *n += 1;
        } else {
            *n -= 1;
        }
        peak = peak.max(live.values().filter(|n| **n > 0).count() as i64);
    }
    peak
}

/// A verdict as a side has it: the trace's, else its owner's answer (an assertion), else none.
fn effective(v: &Value) -> Option<(String, bool)> {
    match v["verdict"].as_str() {
        Some("undecided") | None => match v.get("asserted").and_then(|a| a["answer"].as_str()) {
            Some(a) if a != crate::digest::UNKNOWN => Some((a.to_string(), true)),
            _ => None,
        },
        Some(d) => Some((d.to_string(), false)),
    }
}

fn verdicts_of(site: &Value) -> Vec<(String, &Value)> {
    let mut out = Vec::new();
    for (k, v) in site.as_object().into_iter().flatten() {
        if v.get("verdict").is_some_and(Value::is_string) {
            out.push((k.clone(), v));
        } else if let Some(list) = v.as_array() {
            for f in list.iter().filter(|f| f.get("verdict").is_some_and(Value::is_string)) {
                out.push((format!("field {}", py(&f["name"])), f));
            }
        }
    }
    out
}

fn same(a: bool) -> &'static str {
    if a { "same" } else { "different" }
}

fn compare_sites(sa: &[&Value], sb: &[&Value], out: &mut Vec<Aspect>) {
    let mut phases: Vec<&Value> = Vec::new();
    for s in sa.iter().chain(sb) {
        if !phases.contains(&&s["phase"]) {
            phases.push(&s["phase"]);
        }
    }
    for phase in phases {
        let suffix = if phase.is_null() { String::new() } else { format!(" ({})", py(phase)) };
        let (a, b) = (sa.iter().find(|s| &s["phase"] == phase), sb.iter().find(|s| &s["phase"] == phase));
        let kind = |s: Option<&&Value>| s.map_or("none".to_string(), |s| py(&s["kind"]));
        out.push(Aspect { name: format!("site{suffix}"), a: kind(a), b: kind(b), verdict: same(kind(a) == kind(b)), note: String::new() });
        let (Some(a), Some(b)) = (a, b) else { continue };
        if a["kind"] != b["kind"] {
            continue;
        }
        let (va, vb) = (verdicts_of(a), verdicts_of(b));
        let mut names: Vec<&String> = va.iter().map(|(k, _)| k).collect();
        for (k, _) in &vb {
            if !names.contains(&k) {
                names.push(k);
            }
        }
        for name in names {
            let get = |v: &[(String, &Value)]| v.iter().find(|(k, _)| k == name).and_then(|(_, v)| effective(v));
            let (ea, eb) = (get(&va), get(&vb));
            let show = |e: &Option<(String, bool)>| match e {
                Some((v, true)) => format!("{v} (asserted)"),
                Some((v, false)) => v.clone(),
                None => "undecided".into(),
            };
            let (verdict, note) = match (&ea, &eb) {
                (Some((x, ax)), Some((y, ay))) => (same(x == y), if *ax || *ay { "assertion".to_string() } else { String::new() }),
                (None, None) => ("not compared", "both undecided".into()),
                (None, _) => ("not compared", "A undecided".into()),
                (_, None) => ("not compared", "B undecided".into()),
            };
            out.push(Aspect { name: format!("{name}{suffix}"), a: show(&ea), b: show(&eb), verdict, note });
        }
    }
}

/// The kinds of what a template's calls did to the space its files hold (its `space`, §3.92),
/// in a fixed order: what happened, never how often, which grows with the scale.
fn space_kinds(t: &Value) -> Vec<String> {
    let sp = &t["space"];
    let n = |k: &str| int(&sp[k]);
    let mut out: Vec<String> = [
        ("truncating_opens", "truncating opens"),
        ("truncating_opens_of_written", "of a path written before"),
        ("truncate", "truncate"),
        ("ftruncate", "ftruncate"),
        ("to_zero", "to length 0"),
    ]
    .iter()
    .filter(|(k, _)| n(k) > 0)
    .map(|(_, name)| name.to_string())
    .collect();
    out.extend(sp["fallocate"].as_object().into_iter().flatten().map(|(mode, _)| format!("fallocate {mode}")));
    out
}

fn compare_pair(da: &Value, ta: &Value, db: &Value, tb: &Value, by: &'static str) -> Pair {
    let mut aspects = Vec::new();
    let (fa, fb) = (fields_of(ta), fields_of(tb));
    aspects.push(Aspect { name: "fields".into(), a: show_fields(&fa), b: show_fields(&fb), verdict: same(fa == fb), note: String::new() });
    let (ia, ib) = (int(&ta["id"]), int(&tb["id"]));
    compare_sites(&sites_of(da, ia), &sites_of(db, ib), &mut aspects);
    let ops = |t: &Value| t["ops"].as_object().map(|m| m.keys().cloned().collect::<BTreeSet<String>>()).unwrap_or_default();
    let (oa, ob) = (ops(ta), ops(tb));
    let join = |s: &BTreeSet<String>, sep: &str| if s.is_empty() { "none".to_string() } else { s.iter().cloned().collect::<Vec<_>>().join(sep) };
    aspects.push(Aspect { name: "ops".into(), a: join(&oa, " "), b: join(&ob, " "), verdict: same(oa == ob), note: String::new() });
    let (sa, sb) = (space_kinds(ta), space_kinds(tb));
    if !sa.is_empty() || !sb.is_empty() {
        let show = |k: &[String]| if k.is_empty() { "none".to_string() } else { k.join(", ") };
        aspects.push(Aspect { name: "space".into(), a: show(&sa), b: show(&sb), verdict: same(sa == sb), note: String::new() });
    }
    let (pa, pb) = (phases_of(da, ia), phases_of(db, ib));
    let motifs = |p: &[(i64, i64, i64, String)]| p.iter().map(|x| x.3.clone()).collect::<BTreeSet<String>>();
    let (ma, mb) = (motifs(&pa), motifs(&pb));
    aspects.push(Aspect { name: "motifs".into(), a: join(&ma, " | "), b: join(&mb, " | "), verdict: same(ma == mb), note: String::new() });
    let (ka, kb) = (peak_lanes(&pa), peak_lanes(&pb));
    aspects.push(Aspect { name: "peak lanes".into(), a: ka.to_string(), b: kb.to_string(), verdict: same(ka == kb), note: String::new() });
    let lanes = |p: &[(i64, i64, i64, String)]| p.iter().map(|x| x.0).collect::<BTreeSet<i64>>().len();
    aspects.push(Aspect { name: "lanes".into(), a: lanes(&pa).to_string(), b: lanes(&pb).to_string(), verdict: "shown", note: String::new() });
    Pair { a: text_of(ta).to_string(), b: text_of(tb).to_string(), by, aspects }
}

/// The shape a template is paired by when its text differs: its fields' kinds and widths and
/// the kinds of its sites.
fn shape(doc: &Value, t: &Value) -> (Vec<(String, i64)>, Vec<String>) {
    let mut kinds: Vec<String> = sites_of(doc, int(&t["id"])).iter().map(|s| py(&s["kind"])).collect();
    kinds.sort();
    (fields_of(t), kinds)
}

/// The structure of digest `a` (the application's) against digest `b` (the run's): templates
/// paired by identical text, then by `pairs` (A's text, B's text), then by shape where one of
/// each side has it.
pub fn structure(a: &Value, b: &Value, pairs: &[(String, String)]) -> Result<Structure> {
    let empty = Vec::new();
    let ta: Vec<&Value> = a["templates"].as_array().unwrap_or(&empty).iter().collect();
    let mut s = Structure { lanes: [int(&a["contexts"]), int(&b["contexts"])], ..Default::default() };
    let mut tb: Vec<&Value> = Vec::new();
    for t in b["templates"].as_array().unwrap_or(&empty) {
        match runners_own(t) {
            Some(what) => s.set_aside.push(what),
            None => tb.push(t),
        }
    }
    for (x, y) in pairs {
        if !ta.iter().any(|t| text_of(t) == x) {
            return Err(anyhow!("--pair {x}={y}: A has no template `{x}`"));
        }
        if !tb.iter().any(|t| text_of(t) == y) {
            return Err(anyhow!("--pair {x}={y}: B has no template `{y}`"));
        }
    }
    let mut partner: Vec<Option<(usize, &'static str)>> = vec![None; ta.len()];
    let mut taken = vec![false; tb.len()];
    for (i, t) in ta.iter().enumerate() {
        if let Some((x, y)) = pairs.iter().find(|(x, _)| x == text_of(t)) {
            let j = tb.iter().position(|u| text_of(u) == y).unwrap();
            if taken[j] {
                return Err(anyhow!("--pair {x}={y}: B's `{y}` is paired twice"));
            }
            partner[i] = Some((j, "--pair"));
            taken[j] = true;
        }
    }
    for (i, t) in ta.iter().enumerate() {
        if partner[i].is_none() {
            if let Some(j) = (0..tb.len()).find(|&j| !taken[j] && text_of(tb[j]) == text_of(t)) {
                partner[i] = Some((j, "identical"));
                taken[j] = true;
            }
        }
    }
    for i in 0..ta.len() {
        if partner[i].is_some() {
            continue;
        }
        let want = shape(a, ta[i]);
        let mine: Vec<usize> = (0..ta.len()).filter(|&k| partner[k].is_none() && shape(a, ta[k]) == want).collect();
        let theirs: Vec<usize> = (0..tb.len()).filter(|&j| !taken[j] && shape(b, tb[j]) == want).collect();
        if mine.len() == 1 && theirs.len() == 1 {
            partner[i] = Some((theirs[0], "shape"));
            taken[theirs[0]] = true;
        }
    }
    for (i, t) in ta.iter().enumerate() {
        match partner[i] {
            Some((j, by)) => s.pairs.push(compare_pair(a, t, b, tb[j], by)),
            None => s.unpaired_a.push(text_of(t).to_string()),
        }
    }
    s.unpaired_b = (0..tb.len()).filter(|&j| !taken[j]).map(|j| text_of(tb[j]).to_string()).collect();
    Ok(s)
}

/// The structure as the comparison prints it.
pub fn structure_text(s: &Structure) -> String {
    let mut out = String::new();
    out.push_str(&format!("structure: {} pair(s) of templates; lanes: A {}, B {} (shown, not judged)\n", s.pairs.len(), s.lanes[0], s.lanes[1]));
    if !s.set_aside.is_empty() {
        out.push_str(&format!("set aside from B, the runner's own I/O: {}\n", s.set_aside.join(", ")));
    }
    let head = ("aspect".to_string(), "A".to_string(), "B".to_string());
    let all: Vec<(String, String, String)> = s.pairs.iter().flat_map(|p| p.aspects.iter().map(|x| (x.name.clone(), x.a.clone(), x.b.clone()))).chain([head]).collect();
    let w = |f: fn(&(String, String, String)) -> &String| all.iter().map(|r| f(r).chars().count()).max().unwrap_or(0);
    let (w0, w1, w2) = (w(|r| &r.0), w(|r| &r.1), w(|r| &r.2));
    for p in &s.pairs {
        if p.a == p.b {
            out.push_str(&format!("template {} ({})\n", p.a, p.by));
        } else {
            out.push_str(&format!("template {} = {} ({})\n", p.a, p.b, p.by));
        }
        out.push_str(&format!("  {:<w0$}  {:<w1$}  {:<w2$}  verdict\n", "aspect", "A", "B"));
        for x in &p.aspects {
            let v = if x.verdict == "different" { "DIFFERENT" } else { x.verdict };
            let note = if x.note.is_empty() { String::new() } else { format!(" ({})", x.note) };
            out.push_str(&format!("  {:<w0$}  {:<w1$}  {:<w2$}  {v}{note}\n", x.name, x.a, x.b));
        }
    }
    for t in &s.unpaired_a {
        out.push_str(&format!("template {t}: A only, DIFFERENT (no template of B pairs with it; --pair A=B names one)\n"));
    }
    for t in &s.unpaired_b {
        out.push_str(&format!("template {t}: B only (listed)\n"));
    }
    match s.differences() {
        0 => out.push_str("structure: same\n"),
        n => out.push_str(&format!("structure: {n} difference(s)\n")),
    }
    out
}

fn structure_json(s: &Structure) -> Value {
    json!({
        "pairs": s.pairs.iter().map(|p| json!({"a": p.a, "b": p.b, "by": p.by,
            "aspects": p.aspects.iter().map(|x| json!({"name": x.name, "a": x.a, "b": x.b, "verdict": x.verdict, "note": x.note})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "unpaired_a": s.unpaired_a,
        "unpaired_b": s.unpaired_b,
        "set_aside": s.set_aside,
        "lanes": s.lanes,
        "differences": s.differences(),
    })
}

// ---------------------------------------------------------------- the command

/// What `aeiou-trace compare` was asked.
#[derive(Debug, Clone, Default)]
pub struct Opts {
    pub a: String,
    pub b: String,
    pub template_a: Option<String>,
    pub template_b: Option<String>,
    pub max_distance: Option<f64>,
    pub judge: bool,
    pub tolerances: Option<String>,
    pub selfs: Vec<String>,
    pub only: Vec<String>,
    /// Templates paired by hand, A's text and B's.
    pub pairs: Vec<(String, String)>,
    /// The comparison as one JSON document (`aeiou_compare: 1`) instead of the text.
    pub json: bool,
}

pub const COMPARE_FORMAT: u64 = 1;

/// The numbers of a document: a metrics document's `total`, or one actor template's.
pub fn select<'d>(doc: &'d Value, template: Option<&str>) -> Result<&'d Value> {
    let Some(t) = template else { return Ok(&doc["total"]) };
    doc.get("templates").and_then(|m| m.get(t)).ok_or_else(|| {
        let names: Vec<&String> = doc.get("templates").and_then(Value::as_object).map(|m| m.keys().collect()).unwrap_or_default();
        anyhow!("no template `{t}` in the document (it has: {})", if names.is_empty() { "none".to_string() } else { names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") })
    })
}

fn is_digest(doc: &Value) -> bool {
    doc.get("aeiou_digest").and_then(Value::as_u64) == Some(crate::digest::FORMAT)
}

/// The numbers of a document: a digest's `metrics` section, or the metrics document itself.
pub fn numbers(doc: &Value) -> &Value {
    if is_digest(doc) { &doc["metrics"] } else { doc }
}

/// A document as the first lines of the comparison describe it.
pub fn describe(doc: &Value) -> String {
    if is_digest(doc) {
        let n = doc["templates"].as_array().map_or(0, Vec::len);
        let mut nums = numbers(doc).clone();
        if nums.get("trace").is_none() {
            nums["trace"] = doc.get("trace").cloned().unwrap_or(Value::Null);
        }
        return format!("digest of {}, {n} template(s); {}", py(doc.get("trace").unwrap_or(&Value::Null)), describe(&nums));
    }
    let g = |k: &str| py(doc.get(k).unwrap_or(&Value::Null));
    let roots = doc.get("roots").and_then(Value::as_array).map(|r| r.iter().map(py).collect::<Vec<_>>().join(", ")).unwrap_or_default();
    match doc.get("source").and_then(Value::as_str).unwrap_or("dry-run") {
        "strace" => format!("strace of {roots} (order: completion; {} instance(s), {} thread(s))", g("instances"), g("contexts")),
        "trace" => format!("trace file {} of {roots} (order: completion; {} instance(s), {} lane(s))", g("trace"), g("instances"), g("contexts")),
        src @ ("minio-audit" | "s3-access") => format!("{src} log of {roots} (order: completion; {} client(s))", g("contexts")),
        src => format!("{src} of {} seed {} gpus {} (order: {})", g("abstract"), g("seed"), g("gpus"), g("order")),
    }
}

/// A digest (`aeiou_digest: 2`) or a metrics document (`aeiou_metrics: 2`, or 1 from before
/// 2026-10-22, numbers only).
pub fn load(path: &str) -> Result<Value> {
    let d = read_json(path)?;
    if !is_digest(&d) && !d.get("aeiou_metrics").and_then(Value::as_u64).is_some_and(|v| METRICS_READ.contains(&v)) {
        return Err(anyhow!("{path}: not an aeiou_digest: 2 or aeiou_metrics: 2 document (or aeiou_metrics: 1, a capture's from before 2026-10-22)"));
    }
    Ok(d)
}

fn width(rows: &[Row]) -> [usize; 3] {
    let head = Row { metric: "metric".into(), a: "A".into(), b: "B".into(), distance: None };
    let all = || rows.iter().chain(std::iter::once(&head));
    [all().map(|r| r.metric.chars().count()).max().unwrap(), all().map(|r| r.a.chars().count()).max().unwrap(), all().map(|r| r.b.chars().count()).max().unwrap()]
}

/// A float as Python's `str` writes it (`0.1`, `0.0`).
fn py_float(x: f64) -> String {
    py(&json!(x))
}

fn row_json(r: &Row) -> Value {
    json!({"metric": r.metric, "a": r.a, "b": r.b, "distance": r.distance})
}

/// Run the comparison: what it prints is appended to `out`, which keeps what was printed
/// before a failure (under `json`, the document, written whole at the end); the exit status
/// (0, or 1 for a comparison that did not pass: a difference of structure, a distance past
/// `--max-distance`, a row outside its tolerance, or nothing judged).
pub fn run(o: &Opts, out: &mut String) -> Result<i32> {
    let judging = o.judge || o.tolerances.is_some() || !o.selfs.is_empty();
    if o.max_distance.is_some() && judging {
        let rule = if o.judge {
            "--judge"
        } else if o.tolerances.is_some() {
            "--tolerances (which implies --judge)"
        } else {
            "--self (which implies --judge)"
        };
        return Err(usage::err(format!("--max-distance cannot be used with {rule}: they are two rules; give one")));
    }
    let docs = [load(&o.a)?, load(&o.b)?];
    let nums = [numbers(&docs[0]), numbers(&docs[1])];
    let (a, b) = (select(nums[0], o.template_a.as_deref())?, select(nums[1], o.template_b.as_deref())?);
    // where only the blocks are in common, they are in common only at one size
    let block = |d: &Value| d.get("block").cloned().unwrap_or(Value::Null);
    if !(has(a, "reuse_extents") && has(b, "reuse_extents")) && block(nums[0]) != block(nums[1]) {
        return Err(anyhow!("block sizes differ ({} and {}): reuse distance and block popularity are not comparable", py(&block(nums[0])), py(&block(nums[1]))));
    }
    let structure = match (is_digest(&docs[0]), is_digest(&docs[1])) {
        (true, true) => Some(structure(&docs[0], &docs[1], &o.pairs)?),
        _ if !o.pairs.is_empty() => return Err(anyhow!("--pair pairs the templates of two digests, and {} a metrics document", if is_digest(&docs[0]) { "B is" } else { "A is" })),
        _ => None,
    };
    let unstructured = match (is_digest(&docs[0]), is_digest(&docs[1])) {
        (true, true) | (false, false) => None,
        (true, false) => Some("structure: not compared, B is a metrics document (numbers only)\n"),
        (false, true) => Some("structure: not compared, A is a metrics document (numbers only)\n"),
    };
    let mut rows = rows(a, b)?;
    if !o.only.is_empty() {
        rows.retain(|r| o.only.iter().any(|p| r.metric.starts_with(p.as_str())));
    }
    let mut doc = json!({
        "aeiou_compare": COMPARE_FORMAT,
        "a": {"path": o.a, "describe": describe(&docs[0])},
        "b": {"path": o.b, "describe": describe(&docs[1])},
        "structure": structure.as_ref().map(structure_json),
    });
    let mut text = String::new();
    text.push_str(&format!("A: {}: {}\n", o.a, describe(&docs[0])));
    text.push_str(&format!("B: {}: {}\n", o.b, describe(&docs[1])));
    if let Some(s) = &structure {
        text.push_str(&structure_text(s));
    }
    text.push_str(unstructured.unwrap_or(""));
    let differences = structure.as_ref().map_or(0, Structure::differences);
    if !o.json {
        out.push_str(&text);
    }
    if judging {
        return judged(o, &rows, a, b, &docs, differences, &mut doc, out);
    }
    let w = width(&rows);
    let mut text = format!("{:<w0$}  {:>w1$}  {:>w2$}  distance\n", "metric", "A", "B", w0 = w[0], w1 = w[1], w2 = w[2]);
    let mut worst = 0.0f64;
    for r in &rows {
        worst = worst.max(r.distance.unwrap_or(0.0));
        let flag = match (o.max_distance, r.distance) {
            (Some(m), Some(d)) if d > m => "  <",
            _ => "",
        };
        let d = r.distance.map_or("-".to_string(), |d| format!("{d:.3}"));
        text.push_str(&format!("{:<w0$}  {:>w1$}  {:>w2$}  {d}{flag}\n", r.metric, r.a, r.b, w0 = w[0], w1 = w[1], w2 = w[2]));
    }
    text.push_str(&format!("largest distance {worst:.3}\n"));
    let exceeds = o.max_distance.filter(|m| worst > *m);
    doc["rows"] = json!(rows.iter().map(row_json).collect::<Vec<_>>());
    doc["largest_distance"] = json!(worst);
    doc["max_distance"] = json!(o.max_distance);
    doc["verdict"] = json!(if differences > 0 || exceeds.is_some() { "different" } else { "same" });
    if o.json {
        out.push_str(&(serde_json::to_string_pretty(&doc)? + "\n"));
    } else {
        out.push_str(&text);
    }
    if let Some(m) = exceeds {
        return Err(anyhow!("exceeds --max-distance {}", py_float(m)));
    }
    if differences > 0 {
        return Err(anyhow!("the structure differs ({differences} difference(s))"));
    }
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn judged(o: &Opts, rows: &[Row], a: &Value, b: &Value, docs: &[Value; 2], differences: usize, doc: &mut Value, out: &mut String) -> Result<i32> {
    let mut selfs = Vec::new();
    let mut text = String::new();
    let mut seeds = Vec::new();
    let nb = numbers(&docs[1]);
    for path in &o.selfs {
        let d = load(path).or_else(|_| read_json(path))?;
        let d = numbers(&d).clone();
        let get = |d: &Value, k: &str| d.get(k).cloned().unwrap_or(Value::Null);
        if get(&d, "aeiou_metrics") != get(nb, "aeiou_metrics") {
            return Err(anyhow!(
                "{path}: --self is B's abstract at another seed, and its aeiou_metrics is not B's ({} and {})",
                py(&get(&d, "aeiou_metrics")),
                py(&get(nb, "aeiou_metrics"))
            ));
        }
        for key in ["source", "sha256", "gpus", "block"] {
            if get(&d, key) != get(nb, key) {
                return Err(anyhow!("{path}: --self is B's abstract at another seed, and its `{key}` is not B's ({} and {})", py(&get(&d, key)), py(&get(nb, key))));
            }
        }
        if get(&d, "seed") == get(nb, "seed") {
            return Err(anyhow!("{path}: --self has B's seed ({}); another seed is the point", py(&get(&d, "seed"))));
        }
        let line = format!("self: {path}: seed {}\n", py(&get(&d, "seed")));
        if o.json {
            text.push_str(&line);
        } else {
            out.push_str(&line);
        }
        seeds.push(json!({"path": path, "seed": get(&d, "seed")}));
        selfs.push(select(&d, o.template_b.as_deref())?.clone());
    }
    let spec = o.tolerances.as_deref().map(load_tolerances).transpose()?;
    let (judged, stale) = judge(rows, a, b, &[numbers(&docs[0]), nb], &selfs, spec.as_ref())?;
    let w = width(rows);
    text.clear();
    text.push_str(&format!("{:<w0$}  {:>w1$}  {:>w2$}  distance  allowed  verdict\n", "metric", "A", "B", w0 = w[0], w1 = w[1], w2 = w[2]));
    for j in &judged {
        let r = &j.row;
        let d = r.distance.map_or("-".to_string(), |d| format!("{d:.3}"));
        let al = j.allowed.map_or("-".to_string(), |x| format!("{x:.3}"));
        let v = if j.verdict == "outside" { "OUTSIDE" } else { j.verdict };
        let note = if j.note.is_empty() { String::new() } else { format!(" ({})", j.note) };
        text.push_str(&format!("{:<w0$}  {:>w1$}  {:>w2$}  {d:>8}  {al:>7}  {v}{note}\n", r.metric, r.a, r.b, w0 = w[0], w1 = w[1], w2 = w[2]));
    }
    let n = |v: &str| judged.iter().filter(|j| j.verdict == v).count();
    for line in &stale {
        text.push_str(&format!("note: {line}\n"));
    }
    let counts = format!("{} row(s) within tolerance, {} outside, {} unseen, {} not judged", n("ok"), n("outside"), n("unseen"), n("not judged"));
    let (verdict, status, last) = if differences > 0 {
        ("not accepted", 1, format!("not accepted: the structure differs ({differences} difference(s)); {counts}"))
    } else if n("outside") > 0 {
        ("not accepted", 1, format!("not accepted: {counts}"))
    } else if n("ok") == 0 {
        ("nothing judged", 1, format!("nothing judged: {counts}"))
    } else {
        ("accepted", 0, format!("accepted: {counts}"))
    };
    text.push_str(&last);
    text.push('\n');
    doc["selfs"] = json!(seeds);
    doc["rows"] = json!(judged.iter().map(|j| {
        let mut r = row_json(&j.row);
        r["allowed"] = json!(j.allowed);
        r["verdict"] = json!(j.verdict);
        r["note"] = json!(j.note);
        r
    }).collect::<Vec<_>>());
    doc["notes"] = json!(stale);
    doc["verdict"] = json!(verdict);
    if o.json {
        out.push_str(&(serde_json::to_string_pretty(doc)? + "\n"));
    } else {
        out.push_str(&text);
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_quantiles() {
        assert_eq!(size(0), "0 B");
        assert_eq!(size(4096), "4 KiB");
        assert_eq!(size(1536 * 1024), "1.5 MiB");
        assert_eq!(size(1 << 50), "1024 TiB");
        let h = json!({"n": 10, "buckets": [[4, 5], [8, 4], [16, 1]]});
        assert_eq!((quantile(&h, 0.5), quantile(&h, 0.9), quantile(&h, 0.99)), (4, 8, 16));
        assert_eq!(quantile(&json!({"n": 0, "buckets": []}), 0.5), 0);
    }

    #[test]
    fn distances() {
        let m = |p: &[(i64, i64)]| p.iter().copied().collect::<BTreeMap<i64, i64>>();
        assert_eq!(tv_distance(&m(&[(4, 3), (2, 1)]), &m(&[(4, 1), (2, 1)])), Some(0.25));
        assert_eq!(tv_distance(&m(&[]), &m(&[])), None);
        assert_eq!(tv_distance(&m(&[(1, 1)]), &m(&[])), Some(1.0));
        let (e, one) = (json!({"n": 0, "buckets": []}), json!({"n": 1, "buckets": [[1, 1]]}));
        assert_eq!((cdf_distance(&e, &e), cdf_distance(&e, &one)), (None, Some(1.0)));
    }

    #[test]
    fn classes_are_the_longest_prefix() {
        assert_eq!(tolerance_class("extent reuse distance, read after read (p50 / p90 / p99)"), "extent reuse distance");
        assert_eq!(tolerance_class("extent reuse: partial re-reads / bytes read again"), "extent reuse");
        assert_eq!(tolerance_class("reuse: first touches / block reads"), "reuse");
        assert_eq!(tolerance_class("ops: read / all ops"), "ops");
    }

    #[test]
    fn an_object_log_is_described_as_one() {
        // the Python described it as a dry run's, `minio-audit of None seed None gpus None`
        let d = json!({"aeiou_metrics": 2, "source": "minio-audit", "roots": ["bucket/p"], "contexts": 3});
        assert_eq!(describe(&d), "minio-audit log of bucket/p (order: completion; 3 client(s))");
    }
}
