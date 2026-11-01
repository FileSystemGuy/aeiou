//! The digest of `aeiou-trace` (`DESIGN_REVIEW.md` §3.89, stage 2) against the golden corpus in
//! `tests/digest/`: every trace the Python digest's tests fed it on 2026-10-31
//! (`builder/tests/test_digest.py`, `test_objectlog.py`, `test_applications.py`,
//! `test_trace.py`), with every set of answers and options its document was made with and what
//! it made, and every document its text report was rendered from with the text. A
//! `digest-<hash>.json` names its input, a `trace-<hash>.in` (an strace, an exported trace
//! file, or an object-store log), and holds the arguments and the document or the refusal; a
//! `render-<hash>.json` holds a document and its report. The Python digest was the
//! specification until the port; these files keep it.
//!
//! What the port adds is not in the goldens and is set aside before the comparison: the
//! version (`aeiou_digest: 2`), the `metrics` section, and each site's `facts`. One thing the
//! port does differently: a template's ops with equal counts are listed by name in the report,
//! where the Python listed them in the order they first appeared, which a JSON object read back
//! does not keep; the comparison sorts both reports' op lists so.
use std::io::BufRead;
use std::path::PathBuf;

use aeiou::digest;
use serde_json::{json, Value};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/digest")
}

fn cases(kind: &str) -> Vec<(String, Value)> {
    let mut names: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".json").map(str::to_string))
        .filter(|n| n.starts_with(&format!("{kind}-")))
        .collect();
    names.sort();
    names.into_iter().map(|n| {
        let v = serde_json::from_str(&std::fs::read_to_string(dir().join(format!("{n}.json"))).unwrap()).unwrap();
        (n, v)
    }).collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

fn first_difference(g: &Value, w: &Value) -> String {
    let (gs, ws) = (serde_json::to_string_pretty(g).unwrap(), serde_json::to_string_pretty(w).unwrap());
    let at = gs.lines().zip(ws.lines()).position(|(a, b)| a != b).unwrap_or(0);
    format!("at line {at}:\n  got  {}\n  want {}", gs.lines().nth(at).unwrap_or(""), ws.lines().nth(at).unwrap_or(""))
}

/// The document without what version 2 adds.
fn as_version_1(mut doc: Value) -> Value {
    assert_eq!(doc["aeiou_digest"], json!(2));
    assert!(doc["metrics"].is_object(), "a metrics section");
    doc["aeiou_digest"] = json!(1);
    doc.as_object_mut().unwrap().remove("metrics");
    for s in doc["sites"].as_array_mut().unwrap() {
        assert!(s["facts"].is_array(), "a site's facts");
        s.as_object_mut().unwrap().remove("facts");
    }
    doc
}

#[test]
fn every_golden_digest_is_reproduced() {
    let cases = cases("digest");
    assert!(cases.len() >= 100, "the corpus is there");
    for (name, want) in &cases {
        let text = std::fs::read_to_string(dir().join(want["input"].as_str().unwrap())).unwrap();
        let args = &want["args"];
        let first = text.lines().next().map(|l| format!("{l}\n")).unwrap_or_default();
        let opener = || -> std::io::Result<Box<dyn BufRead + '_>> { Ok(Box::new(text.as_bytes())) };
        let got = digest::analyze(&opener, &first, args["root"].as_str(), &strs(&args["exclude"]), args["cwd"].as_str());
        let an = match (got, want.get("error").and_then(Value::as_str)) {
            (Ok(an), None) => an,
            (Err(e), Some(w)) => {
                // the JSON decoders word their messages differently; the line number is the contract
                let g = format!("{e:#}");
                if w.contains("Expecting") {
                    assert_eq!(g.split(':').next(), w.split(':').next(), "{name}");
                } else {
                    assert_eq!(g, w, "{name}");
                }
                continue;
            }
            (Ok(_), Some(w)) => panic!("{name}: the golden refused it ({w}), the port did not"),
            (Err(e), None) => panic!("{name}: {e:#}"),
        };
        let d = &want["document"];
        let o = digest::Opts {
            epochs: d.get("epochs").and_then(Value::as_i64).unwrap_or(1),
            candidates: d.get("candidates").and_then(Value::as_i64),
            paths: d.get("paths").and_then(Value::as_bool).unwrap_or(false),
        };
        let answers = d.get("answers").and_then(Value::as_array).cloned().unwrap_or_default();
        let doc = an.document(&o, &answers, d.get("by").and_then(Value::as_str), d.get("at").and_then(Value::as_str), d.get("application").unwrap_or(&Value::Null));
        let got = as_version_1(doc);
        assert!(got == want["want"], "{name}: differs from the golden {}", first_difference(&got, &want["want"]));
    }
}

/// A report with each template's ops listed by count and then by name.
fn ops_by_name(text: &str) -> String {
    text.lines()
        .map(|l| {
            let Some(at) = l.find("; ops ") else { return l.to_string() };
            let rest = &l[at + 6..];
            let end = rest.find(';').unwrap_or(rest.len());
            let mut ops: Vec<(i64, &str)> = rest[..end].split(", ").map(|kv| {
                let (k, v) = kv.rsplit_once(' ').unwrap();
                (-v.parse::<i64>().unwrap(), k)
            }).collect();
            ops.sort();
            let list: Vec<String> = ops.iter().map(|(n, k)| format!("{k} {}", -n)).collect();
            format!("{}; ops {}{}", &l[..at], list.join(", "), &rest[end..])
        })
        .map(|l| l + "\n")
        .collect()
}

#[test]
fn every_golden_report_is_reproduced() {
    let cases = cases("render");
    assert!(cases.len() >= 10, "the corpus is there");
    for (name, want) in &cases {
        let got = digest::render(&want["doc"]);
        let (g, w) = (ops_by_name(&got), ops_by_name(want["want"].as_str().unwrap()));
        if g != w {
            let at = g.lines().zip(w.lines()).position(|(a, b)| a != b).unwrap_or(0);
            panic!("{name}: the report differs at line {at}:\n  got  {}\n  want {}", g.lines().nth(at).unwrap_or(""), w.lines().nth(at).unwrap_or(""));
        }
    }
}

/// The application records may assert only what the digest asks (`builder/aeiou/applications.py`
/// keeps their vocabulary, `FACTS` and `QUESTIONS`, until stage 3 moves the records here).
#[test]
fn the_application_records_vocabulary_is_the_digests() {
    let src = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../builder/aeiou/applications.py")).unwrap();
    let quoted = |text: &str| -> Vec<String> { text.split('"').skip(1).step_by(2).map(str::to_string).collect() };
    // a statement: from its name to the line that closes it
    let statement = |name: &str, close: &str| -> String {
        let at = src.find(&format!("\n{name} = ")).unwrap_or_else(|| panic!("{name} in applications.py")) + 1;
        let end = src[at..].find(close).unwrap() + close.len();
        src[at..at + end].to_string()
    };
    assert_eq!(quoted(&statement("FACTS", ")\n")), digest::FACT_NAMES.to_vec());
    let q = quoted(&statement("QUESTIONS", "}\n"));
    let want: Vec<String> = digest::QUESTIONS.iter().flat_map(|q| std::iter::once(q.to_string()).chain(digest::candidates_of(q).unwrap().into_iter().map(str::to_string))).collect();
    assert_eq!(q, want);
    assert_eq!(digest::candidates_of("label"), Some(["checkpoint", "output"]));
    assert!(digest::candidates_of("field:step").is_none() && digest::candidates_of("nothing").is_none());
}
