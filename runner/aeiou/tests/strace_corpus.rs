//! The trace-side tools of `aeiou-trace` (`DESIGN_REVIEW.md` §3.89) against the golden corpus in
//! `tests/strace/`: every strace, exported trace file, and object-store log the Python
//! implementation's tests fed it on 2026-10-29 (`builder/tests/test_trace.py`, `test_digest.py`,
//! `test_objectlog.py`), each `<kind>-<hash>.in` beside `<kind>-<hash>.json`, what that implementation
//! made of it: the exported header and lines, and the metrics document. The Python parser was
//! the specification until the port; these files keep it. Three of the straces are of this
//! runner (`file_per_sample`, `lmcache_local_disk`, `diskann_search`), their temporary paths
//! rewritten to `/t/...` and the checkout's to `/src/aeiou`.
use std::io::BufRead;
use std::path::PathBuf;

use aeiou::{export, objectlog, strace};
use serde_json::Value;

fn corpus() -> Vec<(String, String, Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/strace");
    let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".json").map(str::to_string)).collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let text = std::fs::read_to_string(dir.join(format!("{n}.in"))).unwrap();
            let want: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{n}.json"))).unwrap()).unwrap();
            (n, text, want)
        })
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// An exported file as `(header, lines)`; nothing is written for a trace with no line.
fn split(out: &[u8], summary: &export::Summary) -> Value {
    if out.is_empty() {
        assert_eq!(summary.lines, 0);
        return serde_json::json!({"header": summary.header, "events": []});
    }
    let mut lines = out.split(|b| *b == b'\n').filter(|l| !l.is_empty()).map(|l| serde_json::from_slice::<Value>(l).unwrap());
    let header = lines.next().unwrap();
    serde_json::json!({"header": header, "events": lines.collect::<Vec<_>>()})
}

/// `want` is the Python's: a document, or `{"error": message}`.
fn check(name: &str, what: &str, got: anyhow::Result<Value>, want: &Value) {
    match (got, want.get("error").and_then(Value::as_str)) {
        (Ok(g), None) => {
            if &g != want {
                let (gs, ws) = (serde_json::to_string_pretty(&g).unwrap(), serde_json::to_string_pretty(want).unwrap());
                let at = gs.lines().zip(ws.lines()).position(|(a, b)| a != b).unwrap_or(0);
                panic!("{name} {what}: differs from the golden at line {at}:\n  got  {}\n  want {}", gs.lines().nth(at).unwrap_or(""), ws.lines().nth(at).unwrap_or(""));
            }
        }
        (Err(e), Some(w)) => {
            // the JSON decoders word their messages differently; the line number is the contract
            let (g, w) = (format!("{e:#}"), w.to_string());
            if w.contains("Expecting") || w.contains("column") {
                assert_eq!(g.split(':').next(), w.split(':').next(), "{name} {what}");
            } else {
                assert_eq!(g, w, "{name} {what}");
            }
        }
        (Ok(_), Some(w)) => panic!("{name} {what}: the golden refused it ({w}), the port did not"),
        (Err(e), None) => panic!("{name} {what}: {e:#}"),
    }
}

#[test]
fn every_golden_case_is_reproduced() {
    let cases = corpus();
    assert!(cases.len() >= 100, "the corpus is there");
    for (name, text, want) in &cases {
        let args = &want["args"];
        let opener = || -> std::io::Result<Box<dyn BufRead + '_>> { Ok(Box::new(text.as_bytes())) };
        match want["kind"].as_str().unwrap() {
            "export" | "metrics" => {
                let o = strace::Opts {
                    roots: strs(&args["roots"]),
                    exclude: strs(&args["exclude"]),
                    cwd: args["cwd"].as_str().map(str::to_string),
                    instance_roots: args["instance_roots"].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default(),
                    chain_gap_us: args["chain_gap_us"].as_f64(),
                    block: args["block"].as_u64(),
                };
                check(name, "metrics", strace::metrics(text.as_bytes(), &o).map(|(d, _)| d), &want["metrics"]);
                if let Some(w) = want.get("export") {
                    let mut out = Vec::new();
                    let got = export::strace(&opener, &o.roots[0], &o.exclude, o.cwd.as_deref(), &mut out).map(|s| split(&out, &s));
                    check(name, "export", got, w);
                }
            }
            "metrics_file" => check(name, "metrics", export::metrics_of_file(text.as_bytes(), args["block"].as_u64()), &want["metrics"]),
            "objectlog" => {
                let (root, format) = (args["root"].as_str().unwrap(), args["format"].as_str().unwrap());
                let mut out = Vec::new();
                let got = objectlog::export(&opener, root, format, &mut out).map(|s| split(&out, &s));
                check(name, "export", got, &want["export"]);
                if let Some(w) = want.get("metrics") {
                    check(name, "metrics", objectlog::metrics(&opener, root, format, None).map(|(d, _)| d), w);
                }
            }
            k => panic!("{name}: kind {k}"),
        }
    }
}
