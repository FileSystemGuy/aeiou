//! Contract 0.24 (review suggestion 4; `DESIGN_REVIEW.md` §3.100, the note "Suggestion 4
//! built"): a `span` outside a timeline. A run reports the elapsed time of its body each time
//! it is entered, its own ops and a joined `parallel`'s, under the names a timeline's spans
//! have; the plan, the op set, and the fingerprint are what they are without it.

use serde_json::{json, Value};

fn doc(version: &str, body: Value) -> String {
    json!({"ast": version, "name": "t",
           "namespaces": {"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": 8192, "seed": 2}},
           "actors": {"gpu": {"count": 1, "body": body}}}).to_string()
}

fn obj(k: Value) -> Value {
    json!({"object": {"namespace": "o", "fields": {"k": k}}})
}

/// Three files a pass, written (one on the actor's line, two by a joined parallel) and one
/// read back; `wrap` puts each pass in a span.
fn passes(wrap: bool) -> Value {
    let file = |k: Value, flags: Value, io: &str| json!([
        {"let": {"name": "f", "value": obj(k)}},
        {"open": {"file": {"ref": "f"}, "flags": flags}}, {(io): {"file": {"ref": "f"}, "len": 8192}}, {"close": {"file": {"ref": "f"}}}]);
    let base = json!({"mul": [{"index": "i"}, 10]});
    let mut inner = file(base.clone(), json!(["WRONLY", "CREAT"]), "write").as_array().unwrap().clone();
    inner.push(json!({"parallel": {"index": "k", "width": 2, "body": file(json!({"add": [base.clone(), {"add": [{"index": "k"}, 1]}]}), json!(["WRONLY", "CREAT"]), "write")}}));
    inner.push(json!({"phase": {"name": "back", "body": file(base, json!(["RDONLY"]), "read")}}));
    let pass = if wrap { json!([{"span": {"name": "pass", "body": inner}}]) } else { json!(inner) };
    json!([{"loop": {"index": "i", "to": 4, "body": pass}}, {"compute": {"ns": 1000}}])
}

fn fingerprint(o: &std::process::Output) -> Option<String> {
    String::from_utf8_lossy(&o.stdout).lines().find(|l| l.starts_with("fingerprint ")).map(str::to_string)
}

/// A span changes nothing the dry run walks: the ops and the fingerprint are those of its body
/// written where it stands (no draw is in it: a draw's site is its place in the document).
#[test]
fn a_span_outside_a_timeline_leaves_the_plan_as_it_was() {
    let dir = crate::common::tmpdir("spans", "plan");
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let mut fps = Vec::new();
    for (name, version, wrap) in [("bare", "0.29", false), ("span", "0.29", true)] {
        let p = dir.join(format!("{name}.ast.json"));
        std::fs::write(&p, doc(version, passes(wrap))).unwrap();
        let o = std::process::Command::new(aeiou).args(["dry-run", p.to_str().unwrap(), "--gpus", "1"]).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        assert!(String::from_utf8_lossy(&o.stdout).contains("by kind: open=16 close=16 read=4 write=12"), "{}", String::from_utf8_lossy(&o.stdout));
        fps.push(fingerprint(&o).expect("a fingerprint"));
    }
    assert_eq!(fps[0], fps[1]);
}

/// `aeiou run` times each entry of the span, on the blocking backends and on the event loops:
/// four D, each from the instant the actor reached the span to its last op's completion, the
/// parallel in it joined; reported in the text and in the JSON report where a timeline's spans
/// are, and the run's ops are the plan's.
#[test]
fn a_run_reports_a_span_outside_a_timeline_on_every_backend() {
    let dir = crate::common::tmpdir("spans", "run");
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let p = dir.join("t.ast.json");
    std::fs::write(&p, doc("0.29", passes(true))).unwrap();
    let a = p.to_str().unwrap();
    let dry = std::process::Command::new(aeiou).args(["dry-run", a, "--gpus", "1"]).output().unwrap();
    assert!(dry.status.success(), "{}", String::from_utf8_lossy(&dry.stderr));
    for posix in ["sync", "io_uring", "libaio", "posix-aio", "mmap"] {
        let root = dir.join(posix);
        std::fs::create_dir_all(&root).unwrap();
        let report = dir.join(format!("{posix}.json"));
        let t0 = std::time::Instant::now();
        let r = std::process::Command::new(aeiou)
            .args(["run", a, "--gpus", "1", "--root", root.to_str().unwrap(), "--posix", posix, "--report-json", report.to_str().unwrap()])
            .output()
            .unwrap();
        let took = t0.elapsed().as_nanos() as i64;
        let out = String::from_utf8_lossy(&r.stdout);
        assert!(r.status.success(), "{posix}: {}", String::from_utf8_lossy(&r.stderr));
        assert_eq!(fingerprint(&r), fingerprint(&dry), "{posix}: the run's ops are the plan's");
        assert!(out.lines().any(|l| l.trim_start().starts_with("pass ") && l.trim_end().ends_with("n=4")), "{posix}: {out}");
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
        let s = &doc["result"]["timeline"]["spans"]["pass"];
        assert_eq!(s["count"], 4, "{posix}: {}", doc["result"]["timeline"]);
        let (sum, max, p50) = (s["sum_ns"].as_i64().unwrap(), s["max_ns"].as_i64().unwrap(), s["p50_ns"].as_i64().unwrap());
        assert!(p50 > 0 && max >= p50 && sum >= max && sum < took, "{posix}: {s}");
        // every D holds its three writes' and its read's time: the sum is no less than the I/O time of the actor's own ops
        assert_eq!(s["windows"].as_array().unwrap().len(), 1, "{posix}: one five-minute window of the run's clock");
        for k in [0, 1, 2, 30, 31, 32] {
            assert_eq!(std::fs::metadata(root.join(format!("o/{k}"))).unwrap().len(), 8192, "{posix}");
        }
    }
}

/// A span in the sub-actors of a `parallel` (a query of one search thread) is timed in each:
/// two threads of three queries are six D, under every backend.
#[test]
fn a_span_in_a_parallels_sub_actors_is_timed_in_each() {
    let dir = crate::common::tmpdir("spans", "sub");
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let query = json!([
        {"let": {"name": "f", "value": obj(json!({"add": [{"mul": [{"index": "th"}, 10]}, {"index": "q"}]}))}},
        {"span": {"name": "query", "body": [
            {"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": {"ref": "f"}, "len": 8192}}, {"close": {"file": {"ref": "f"}}}]}},
        {"compute": {"ns": 1000}}]);
    let body = json!([{"parallel": {"index": "th", "width": 2, "body": [{"loop": {"index": "q", "to": 3, "body": query}}]}}]);
    let p = dir.join("t.ast.json");
    std::fs::write(&p, doc("0.29", body)).unwrap();
    for posix in ["sync", "io_uring", "libaio", "posix-aio", "mmap"] {
        let root = dir.join(posix);
        std::fs::create_dir_all(&root).unwrap();
        let report = dir.join(format!("{posix}.json"));
        let r = std::process::Command::new(aeiou)
            .args(["run", p.to_str().unwrap(), "--gpus", "1", "--root", root.to_str().unwrap(), "--posix", posix, "--report-json", report.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(r.status.success(), "{posix}: {}", String::from_utf8_lossy(&r.stderr));
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
        let s = &doc["result"]["timeline"]["spans"]["query"];
        assert_eq!(s["count"], 6, "{posix}: {}", doc["result"]["timeline"]);
        assert!(s["p50_ns"].as_i64().unwrap() > 0, "{posix}: {s}");
    }
}

/// What the validator holds a span outside a timeline to, in check.py's words (the shared cases
/// are `refusals.json`'s): no compute and no wait on another actor in it, no staging, no span
/// in it, and the document's version.
#[test]
fn the_validator_holds_a_span_outside_a_timeline_to_its_rules() {
    let refused = |version: &str, body: Value| aeiou::validate::check(&aeiou::ast::parse(&doc(version, body)).expect("parses"));
    let stat = json!({"stat": {"file": obj(json!(0))}});
    let span = |name: &str, body: Value| json!({"span": {"name": name, "body": body}});
    assert_eq!(refused("0.29", json!([span("a", json!([stat.clone(), {"parallel": {"index": "k", "width": 2, "body": [stat.clone()]}}]))])), Vec::<String>::new());
    let why = "outside a timeline a span measures its own ops and a joined `parallel`'s, never compute or a wait on another actor (V36)";
    for (node, what) in [(json!({"compute": {"ns": 1}}), "compute"), (json!({"barrier": {"scope": "host"}}), "barrier"),
                         (json!({"loader": {"name": "q", "index": "b", "workers": 1, "prefetch": 1, "batches": 1, "body": []}}), "loader"),
                         (json!({"timeline": {"name": "tl", "body": []}}), "timeline")] {
        let e = refused("0.29", json!([span("a", json!([{"loop": {"index": "i", "to": 1, "body": [node]}}]))]));
        assert!(e.iter().any(|m| m.ends_with(&format!("a `{what}` in span `a`: {why}"))), "{what}: {e:#?}");
    }
    let e = refused("0.29", json!([{"span": {"name": "a", "staging": "w", "body": []}}]));
    assert!(e.iter().any(|m| m.ends_with("span `a`: `staging` names a timeline's workers, and no timeline encloses it (V36)")), "{e:#?}");
    let e = refused("0.29", json!([span("a", json!([{"parallel": {"index": "k", "width": 1, "body": [span("b", json!([]))]}}]))]));
    assert!(e.iter().any(|m| m.ends_with("a span inside span `a`: a lane's wait is measured once (V36)")), "{e:#?}");
    // contract 0.24 is below the floor since 2026-11-23: no version distinguishes a document with such a span
    assert_eq!(refused("0.29", json!([span("a", json!([]))])), Vec::<String>::new());
}
