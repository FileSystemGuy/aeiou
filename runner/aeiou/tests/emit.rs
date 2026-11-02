//! `aeiou dry-run --emit-trace` (`DESIGN_REVIEW.md` §3.89, stage 4; `emit.rs`): the op stream as
//! the runner's trace file. At one instance its digest's numbers are the dry run's own metrics
//! but fan-out and depth (a file has no forks to count), it loads as a `trace` node whose dry run
//! measures the same, the header names the dry run so `compare --self` takes its metrics
//! documents, and instances interleave in it.
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn workloads() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../workloads")
}

fn npy() -> (PathBuf, PathBuf) {
    let m = workloads().join("input-pipeline/map/npy-per-sample");
    (m.join("npy_per_sample.ast.json"), m.join("params/smoke.params.json"))
}

fn run(bin: &str, args: &[&str]) -> String {
    let o = Command::new(bin).args(args).env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

fn aeiou(args: &[&str]) -> String {
    run(env!("CARGO_BIN_EXE_aeiou"), args)
}

fn trace_tool(args: &[&str]) -> String {
    run(env!("CARGO_BIN_EXE_aeiou-trace"), args)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn read(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// A total without what a trace file cannot carry.
fn comparable(t: &Value) -> Value {
    let mut t = t.clone();
    let m = t.as_object_mut().unwrap();
    m.remove("fan_out");
    m.remove("depth");
    t
}

/// The dry run's metrics and its emitted file, at `gpus` and `seed`.
fn emitted(d: &Path, tag: &str, gpus: &str, seed: &str) -> (PathBuf, PathBuf) {
    let (ast, params) = npy();
    let (m, t) = (d.join(format!("{tag}.metrics.json")), d.join(format!("{tag}.jsonl")));
    let out = aeiou(&["dry-run", s(&ast), "--params-file", s(&params), "--gpus", gpus, "--seed", seed, "--metrics-json", s(&m), "--emit-trace", s(&t)]);
    assert!(out.contains(&format!("emitted trace {}: ", t.display())), "{out}");
    (m, t)
}

#[test]
fn one_instance_emits_the_dry_runs_numbers() {
    let d = crate::common::tmpdir("emit", "one");
    let (m, t) = emitted(&d, "a", "1", "1");
    let text = std::fs::read_to_string(&t).unwrap();
    let mut lines = text.lines();
    let head = lines.next().unwrap();
    assert!(head.starts_with("{\"aeiou_trace\":1,"), "the tools know a trace file by its first key: {head}");
    let h: Value = serde_json::from_str(head).unwrap();
    let dry = read(&m);
    for k in ["abstract", "sha256", "seed", "gpus", "fingerprint"] {
        assert_eq!(h[k], dry[k], "{k}");
    }
    assert_eq!(h["source"], "dry-run");
    let body: Vec<Value> = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(h["lines"].as_u64(), Some(body.len() as u64));
    assert!(body.windows(2).all(|w| w[0]["t"].as_i64() < w[1]["t"].as_i64()), "one microsecond per line, in order");
    // the lseeks say where they leave the position, as a traced lseek's result does
    assert!(body.iter().any(|l| l["op"] == "lseek" && l["ret"].as_i64().is_some_and(|r| r > 0)));
    assert_eq!(h["lanes"].as_u64(), Some(body.iter().map(|l| l["lane"].as_u64().unwrap()).max().unwrap() + 1));
    // the digest's numbers are the dry run's, its source the dry run's
    let dg = d.join("a.digest.json");
    trace_tool(&["digest", s(&t), "--no-questions", "-o", s(&dg)]);
    let digest = read(&dg);
    assert_eq!(digest["source"], "dry-run");
    assert_eq!(comparable(&digest["metrics"]["total"]), comparable(&dry["total"]));
    assert_eq!(digest["metrics"]["sha256"], dry["sha256"]);
    let out = trace_tool(&["compare", s(&dg), s(&m), "--only", "mix", "--only", "request", "--only", "extent", "--only", "run", "--only", "popularity", "--max-distance", "0"]);
    assert!(out.contains("dry-run of npy_per_sample seed 1 gpus 1 (order: round-robin)"), "{out}");
}

#[test]
fn an_emitted_file_runs_as_a_trace_node_of_the_same_numbers() {
    let d = crate::common::tmpdir("emit", "node");
    let (m, t) = emitted(&d, "a", "1", "1");
    let text = std::fs::read(&t).unwrap();
    let sha: String = <sha2::Sha256 as sha2::Digest>::digest(&text).iter().map(|b| format!("{b:02x}")).collect();
    let ast = d.join("node.ast.json");
    std::fs::write(
        &ast,
        format!(
            r#"{{"ast": "0.14", "name": "node", "doc": "the emitted file", "params": {{}}, "datasets": {{}},
                "actors": {{"app": {{"count": 1, "body": [{{"trace": {{"file": "a.jsonl", "sha256": "{sha}"}}}}]}}}}}}"#
        ),
    )
    .unwrap();
    let again = d.join("node.metrics.json");
    aeiou(&["dry-run", s(&ast), "--gpus", "1", "--metrics-json", s(&again)]);
    assert_eq!(comparable(&read(&again)["total"]), comparable(&read(&m)["total"]));
}

#[test]
fn instances_interleave_and_another_seed_is_the_digests_own_spread() {
    let d = crate::common::tmpdir("emit", "two");
    let (_, t1) = emitted(&d, "one", "1", "1");
    let (_, t) = emitted(&d, "a", "2", "1");
    let (s2, _) = emitted(&d, "b", "2", "2");
    let h = |p: &Path| -> Value { serde_json::from_str(std::fs::read_to_string(p).unwrap().lines().next().unwrap()).unwrap() };
    assert_eq!(h(&t)["lanes"].as_u64(), h(&t1)["lanes"].as_u64().map(|n| 2 * n), "every lane of each instance");
    let (da, db) = (d.join("a.digest.json"), d.join("one.digest.json"));
    trace_tool(&["digest", s(&t), "--no-questions", "-o", s(&da)]);
    trace_tool(&["digest", s(&t1), "--no-questions", "-o", s(&db)]);
    // two instances at once on the sample files: twice the peak of one
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou-trace")).args(["compare", s(&db), s(&da), "--json"]).output().unwrap();
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    let pair = doc["structure"]["pairs"].as_array().unwrap().iter().find(|p| p["a"].as_str().unwrap().contains("sample_")).unwrap().clone();
    let peak = pair["aspects"].as_array().unwrap().iter().find(|x| x["name"] == "peak lanes").unwrap();
    let (pa, pb): (i64, i64) = (peak["a"].as_str().unwrap().parse().unwrap(), peak["b"].as_str().unwrap().parse().unwrap());
    assert_eq!(pb, 2 * pa, "{peak}");
    // the abstract at another seed, as its dry run's metrics, is what B differs from itself by
    let out = trace_tool(&["compare", s(&da), s(&da), "--self", s(&s2)]);
    assert!(out.contains(&format!("self: {}: seed 2\n", s2.display())) && out.lines().last().unwrap().starts_with("accepted: "), "{out}");
}

#[test]
fn a_file_that_cannot_be_written_is_named_and_nothing_is_left_beside_it() {
    let d = crate::common::tmpdir("emit", "unwritable");
    let (ast, params) = npy();
    let to = d.join("no/such/dir/t.jsonl");
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["dry-run", s(&ast), "--params-file", s(&params), "--gpus", "1", "--emit-trace", s(&to)]).env_remove("AEIOU_CONFIG").output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains(&format!("{}: No such file or directory", to.display())), "{err}");
    // a directory the file goes in, made read-only once written: the rename refused, no part left
    let ro = d.join("ro");
    std::fs::create_dir(&ro).unwrap();
    let to = ro.join("t.jsonl");
    std::fs::write(&to, "").unwrap();
    std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o500)).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["dry-run", s(&ast), "--params-file", s(&params), "--gpus", "1", "--emit-trace", s(&to)]).env_remove("AEIOU_CONFIG").output().unwrap();
    std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    if unsafe { libc::geteuid() } != 0 {
        assert_eq!(o.status.code(), Some(1), "{}", String::from_utf8_lossy(&o.stderr));
        assert_eq!(std::fs::read_dir(&ro).unwrap().count(), 1, "only the file that was there");
    }
}
