//! The comparison of `aeiou-trace` (`DESIGN_REVIEW.md` §3.89, stage 4) against the golden corpus
//! in `tests/compare/`: every comparison the Python's tests made on 2026-11-01
//! (`builder/tests/test_trace.py`, `test_usage.py`, and the capture pairs of `workloads/`), and
//! written cases for the rare refusals, recorded before the Python was cut. A `rows-<hash>.json`
//! holds two documents' numbers and the rows made of them (or the refusal); a
//! `judge-<hash>.json` the rows judged, with the documents, the seeds' spread, and the tolerance
//! file; a `tolerances-<hash>.json` a tolerance file's text and what it was read as; a
//! `cli-<hash>.json` a command line, the files it named (`{pN}`), and what it printed and
//! returned. The Python was the specification until the port; these files keep it.
//!
//! What differs, chosen: a file that is not JSON is refused in serde's words, not Python's (the
//! path is the contract); a bound the parser refuses (`--max-distance nan`) is worded by the
//! binary's parser, as every other flag of `aeiou-trace`; and a total-variation distance (fan-out,
//! depth) is summed over the values in order, where the Python summed over a set in its hash
//! order, so the two can differ in the last bit (distances are held to 1e-12, every string
//! exactly). And since digests are compared too, a document that is neither kind is refused
//! naming both (two goldens' messages were changed so, 2026-11-01).
use std::path::PathBuf;
use std::process::Command;

use aeiou::compare::{self, Row};

use crate::common;
use serde_json::{json, Value};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/compare")
}

fn cases(kind: &str) -> Vec<(String, Value)> {
    let mut names: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".json").map(str::to_string))
        .filter(|n| n.starts_with(&format!("{kind}-")))
        .collect();
    names.sort();
    names.into_iter().map(|n| (n.clone(), serde_json::from_str(&std::fs::read_to_string(dir().join(format!("{n}.json"))).unwrap()).unwrap())).collect()
}

fn row_of(v: &Value) -> Row {
    Row { metric: v[0].as_str().unwrap().into(), a: v[1].as_str().unwrap().into(), b: v[2].as_str().unwrap().into(), distance: v[3].as_f64() }
}

#[test]
fn every_golden_row_set_is_reproduced() {
    let cases = cases("rows");
    assert!(cases.len() >= 50, "the corpus is there");
    for (name, want) in &cases {
        match (compare::rows(&want["a"], &want["b"]), want.get("error").and_then(Value::as_str)) {
            (Ok(rows), None) => {
                let want_rows = want["rows"].as_array().unwrap();
                assert_eq!(rows.len(), want_rows.len(), "{name}");
                for (g, w) in rows.iter().zip(want_rows) {
                    let w = row_of(w);
                    assert_eq!((&g.metric, &g.a, &g.b), (&w.metric, &w.a, &w.b), "{name}");
                    match (g.distance, w.distance) {
                        (Some(x), Some(y)) => assert!((x - y).abs() < 1e-12, "{name}: {}: {x} against {y}", g.metric),
                        (x, y) => assert_eq!(x, y, "{name}: {}", g.metric),
                    }
                }
            }
            (Err(e), Some(w)) => assert_eq!(format!("{e:#}"), w, "{name}"),
            (got, w) => panic!("{name}: got {:?}, want {w:?}", got.map(|r| r.len())),
        }
    }
}

#[test]
fn every_golden_judgement_is_reproduced() {
    let cases = cases("judge");
    assert!(cases.len() >= 20, "the corpus is there");
    for (name, want) in &cases {
        let rows: Vec<Row> = want["rows"].as_array().unwrap().iter().map(row_of).collect();
        let docs: Vec<&Value> = want["docs"].as_array().unwrap().iter().collect();
        let selfs: Vec<Value> = want["selfs"].as_array().unwrap().clone();
        let spec = Some(&want["spec"]).filter(|s| !s.is_null());
        let (judged, stale) = compare::judge(&rows, &want["a"], &want["b"], &docs, &selfs, spec).unwrap();
        let got: Vec<Value> = judged.iter().map(|j| json!([j.row.metric, j.row.a, j.row.b, j.row.distance, j.allowed, j.verdict, j.note])).collect();
        assert_eq!(Value::Array(got), want["want"]["rows"], "{name}");
        assert_eq!(json!(stale), want["want"]["stale"], "{name}");
    }
}

/// Python's `json` refusals, which serde words otherwise.
fn python_json_error(s: &str) -> bool {
    ["Expecting", "Extra data", "Unterminated", "Invalid control", "Invalid \\escape"].iter().any(|p| s.contains(p))
}

#[test]
fn every_golden_tolerance_file_is_read_as_it_was() {
    let cases = cases("tolerances");
    assert!(cases.len() >= 10, "the corpus is there");
    let tmp = common::tmpdir("compare", "tolerances");
    for (name, want) in &cases {
        let path = tmp.join(format!("{name}.tolerances.json"));
        match want["text"].as_str() {
            Some(t) => std::fs::write(&path, t).unwrap(),
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
        let p = path.to_str().unwrap();
        match (compare::load_tolerances(p), want.get("error").and_then(Value::as_str)) {
            (Ok(spec), None) => assert_eq!(spec, want["spec"], "{name}"),
            (Err(e), Some(w)) => {
                let g = format!("{e:#}").replace(p, "{path}");
                if python_json_error(w) {
                    assert!(g.starts_with("{path}: "), "{name}: {g}");
                } else {
                    assert_eq!(g, w, "{name}");
                }
            }
            (got, w) => panic!("{name}: got {got:?}, want {w:?}"),
        }
    }
}

#[test]
fn every_golden_command_prints_and_returns_what_it_did() {
    let cases = cases("cli");
    assert!(cases.len() >= 50, "the corpus is there");
    for (name, want) in &cases {
        let tmp = common::tmpdir("compare", "cli");
        let mut paths: Vec<(String, String)> = Vec::new();
        for (k, content) in want["files"].as_object().unwrap() {
            let p = tmp.join(format!("{k}.json"));
            if let Some(text) = content.as_str() {
                std::fs::write(&p, text).unwrap();
            }
            paths.push((format!("{{{k}}}"), p.to_str().unwrap().to_string()));
        }
        let args: Vec<String> = want["args"].as_array().unwrap().iter().map(|a| {
            let mut a = a.as_str().unwrap().to_string();
            for (k, p) in &paths {
                a = a.replace(k, p);
            }
            a
        }).collect();
        let o = Command::new(env!("CARGO_BIN_EXE_aeiou-trace")).args(&args).output().unwrap();
        let mut so = String::from_utf8(o.stdout).unwrap();
        let mut se = String::from_utf8(o.stderr).unwrap();
        // longest first, so that p1 is never replaced inside p10
        paths.sort_by_key(|(_, p)| std::cmp::Reverse(p.len()));
        for (k, p) in &paths {
            so = so.replace(p.as_str(), k);
            se = se.replace(p.as_str(), k);
        }
        let (ws, we, wx) = (want["stdout"].as_str().unwrap(), want["stderr"].as_str().unwrap(), want["exit"].as_i64().unwrap());
        assert_eq!(o.status.code(), Some(wx as i32), "{name}: {args:?}\n{se}");
        assert_eq!(so, ws, "{name}: {args:?}");
        if we.contains("invalid non-negative number value") {
            // argparse's words; the binary's parser has its own (tests/trace_tool.rs)
            assert!(se.starts_with("aeiou-trace compare: invalid value ") && se.contains("--max-distance"), "{name}: {se}");
        } else if python_json_error(we) {
            assert_eq!(se.split(": ").take(2).collect::<Vec<_>>(), we.split(": ").take(2).collect::<Vec<_>>(), "{name}: {se}");
        } else {
            assert_eq!(se, we, "{name}: {args:?}");
        }
    }
}
