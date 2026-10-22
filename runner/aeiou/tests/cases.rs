//! Contract 0.11 (`DESIGN_REVIEW.md` §3.78): a required parameter, which has no value until a
//! later layer gives one and declares its kind; the `scale` tag; and a value the storage
//! reports, `{"from": "st_blksize"}`, read by `stat` under `aeiou run` and given with
//! `--st-blksize` to `aeiou dry-run`.

use std::process::Command;

use aeiou::eval::{Config, Params};
use serde_json::{json, Value};

fn tmpdir(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("cases", tag)
}

/// An abstract over dataset `d` of four 8 KiB files whose one actor opens file 0 and reads it
/// in requests of `b` bytes, with `params` declared.
fn doc(params: Value) -> String {
    json!({
        "ast": "0.11", "name": "t", "params": params,
        "datasets": {"d": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4, "size": {"const": 8192}, "seed": 1}},
        "actors": {"gpu": {"body": [
            {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": 0}}}},
            {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
            {"read": {"file": {"ref": "f"}, "len": {"param": "b"}, "repeat": "until_eof"}},
            {"close": {"file": {"ref": "f"}}}]}}
    })
    .to_string()
}

fn params(text: &str, overrides: &[(&str, &str)]) -> Result<Params, String> {
    let loaded = aeiou::load_str(text).map_err(|e| format!("{e:#}"))?;
    let cfg = Config { seed: 1, gpus: 1, overrides: overrides.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![], st_blksize: Default::default() };
    Params::new(&loaded.ast, &cfg).map_err(|e| format!("{e:#}"))
}

#[test]
fn a_required_parameter_is_given_a_value_of_its_kind_or_the_run_is_refused() {
    let text = doc(json!({"b": {"kind": "int", "unit": "bytes"}, "d": {"kind": "dist"}, "xs": {"kind": "array", "items": "int"}, "k": {"default": 2, "scale": true}}));
    assert_eq!(params(&text, &[]).err().unwrap(), "required parameters with no value: b, d, xs (give each in a parameter file or with --param NAME=VALUE)");
    let all = [("b", "4096"), ("d", r#"{"const": 1}"#), ("xs", "[1, 2]")];
    let with = |k: &str, v: &str| -> Vec<(&str, String)> { all.iter().map(|(n, x)| (*n, if *n == k { v.to_string() } else { x.to_string() })).collect() };
    let run = |k: &str, v: &str| {
        let o = with(k, v);
        params(&text, &o.iter().map(|(n, x)| (*n, x.as_str())).collect::<Vec<_>>())
    };
    assert_eq!(run("b", "2.5").err().unwrap(), "--param b: the kind is an integer, the value 2.5 is not");
    assert_eq!(run("b", r#"{"const": 4}"#).err().unwrap(), "--param b: the kind is a scalar, the value is a distribution");
    assert_eq!(run("d", "3").err().unwrap(), "--param d: the kind is a distribution, the value is a scalar");
    assert_eq!(run("xs", r#"[1, {"const": 2}]"#).err().unwrap(), "--param xs[1]: the kind's elements are a scalar, this one is a distribution");
    assert_eq!(run("xs", "[1, 2.5]").err().unwrap(), "--param xs[1]: the kind is an integer, the value 2.5 is not");
    // an integral float is that integer, as for a default's kind
    let p = run("b", "4e3").unwrap();
    assert!(matches!(p.values["b"], aeiou::ast::PValue::Scalar(aeiou::ast::Literal::Int(4000))));
    let loaded = aeiou::load_str(&text).unwrap();
    assert!(loaded.ast.params["k"].scale && !loaded.ast.params["b"].scale);
    assert!(loaded.ast.params["b"].required() && !loaded.ast.params["k"].required());
}

#[test]
fn a_parameter_has_a_default_or_a_kind_never_both_nor_neither() {
    for (p, want) in [
        (json!({"b": {"default": 4, "kind": "int"}}), "a parameter with a default takes its kind from it"),
        (json!({"b": {"unit": "bytes"}}), "a parameter needs a `default` or, required, a `kind`"),
        (json!({"b": {"kind": "array"}}), "`items`, the element kind, goes with `kind: array`"),
        (json!({"b": {"kind": "int", "items": "int"}}), "`items`, the element kind, goes with `kind: array`"),
        (json!({"b": {"default": [1], "items": "int"}}), "`items` goes with `kind: array` on a required parameter"),
    ] {
        let e = aeiou::load_str(&doc(p.clone())).err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(e.contains(want), "{p}: {e}");
    }
    // `"default": null` is a default, the `none` outcome, not a missing one
    assert!(aeiou::load_str(&doc(json!({"b": {"default": 4}, "z": {"default": null}}))).is_ok());
}

/// A storage-reported value is a default: replaced by a later layer like any other, its
/// fallback where nothing is reported, and the number `--st-blksize` gives in a dry run.
#[test]
fn a_storage_reported_value_is_the_number_given_else_its_fallback() {
    let text = doc(json!({"b": {"default": {"from": "st_blksize", "of": "d", "fallback": 4096}}}));
    let loaded = aeiou::load_str(&text).unwrap();
    assert_eq!(loaded.ast.params["b"].kind, aeiou::ast::Kind::Int);
    let value = |st: &[(&str, i64)], o: &[(&str, &str)]| {
        let cfg = Config {
            seed: 1,
            gpus: 1,
            overrides: o.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            sets: vec![],
            st_blksize: st.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        };
        match Params::new(&loaded.ast, &cfg).unwrap().values["b"] {
            aeiou::ast::PValue::Scalar(aeiou::ast::Literal::Int(n)) => n,
            ref v => panic!("{v:?}"),
        }
    };
    assert_eq!(value(&[], &[]), 4096);
    assert_eq!(value(&[("d", 1 << 20)], &[]), 1 << 20);
    assert_eq!(value(&[("d", 1 << 20)], &[("b", "512")]), 512, "a value given replaces it, as any default");
    let e = aeiou::load_str(&doc(json!({"b": {"default": {"from": "st_blksize", "of": "nope", "fallback": 4096}}}))).err().unwrap();
    assert_eq!(format!("{e:#}"), "invalid abstract:\n  t: params/b/default/of: `nope` is not a declared dataset (V28)");
}

fn aeiou(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(args).env_remove("AEIOU_CONFIG").output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn line<'a>(out: &'a str, start: &str) -> &'a str {
    out.lines().find(|l| l.starts_with(start)).unwrap_or_else(|| panic!("no `{start}` line in:\n{out}"))
}

/// `aeiou dry-run --st-blksize` stands in for the storage, the header saying where the number
/// came from; `aeiou run` reads it by `stat` of the dataset's first file and reports it.
#[test]
fn the_storage_reports_st_blksize_to_a_run_and_a_dry_run_is_told_it() {
    let dir = tmpdir("st-blksize");
    let ast = dir.join("t.ast.json");
    std::fs::write(&ast, doc(json!({"b": {"default": {"from": "st_blksize", "of": "d", "fallback": 4096}}}))).unwrap();
    let a = ast.to_str().unwrap();
    let (code, fallback, err) = aeiou(&["dry-run", a, "--gpus", "1"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&fallback, "storage-reported"), "storage-reported: b=4096 (st_blksize of `d`: the fallback, none reported)");
    let (_, big, _) = aeiou(&["dry-run", a, "--gpus", "1", "--st-blksize", "8192"]);
    assert_eq!(line(&big, "storage-reported"), "storage-reported: b=8192 (st_blksize of `d`, --st-blksize)");
    assert_ne!(line(&fallback, "fingerprint"), line(&big, "fingerprint"), "two requests of 4 KiB, or one of 8 KiB");
    let (_, named, _) = aeiou(&["dry-run", a, "--gpus", "1", "--st-blksize", "d=8192"]);
    assert_eq!(line(&named, "fingerprint"), line(&big, "fingerprint"));
    for (bad, want) in [("e=8192", "--st-blksize e=8192: no storage-reported parameter reads dataset `e`"), ("0", "--st-blksize 0: expected a positive byte count, or DATASET=BYTES")] {
        let (code, _, err) = aeiou(&["dry-run", a, "--gpus", "1", "--st-blksize", bad]);
        assert_eq!(code, 2);
        assert!(err.contains(want), "{err}");
    }

    let root = dir.join("root");
    let r = root.to_str().unwrap();
    let (code, _, err) = aeiou(&["datagen", a, "--root", r]);
    assert_eq!(code, 0, "{err}");
    use std::os::unix::fs::MetadataExt;
    let st = std::fs::metadata(root.join("d/f0000")).unwrap().blksize();
    let report = dir.join("report.json");
    let (code, out, err) = aeiou(&["run", a, "--gpus", "1", "--root", r, "--report-json", report.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "storage-reported"), format!("storage-reported: b={st} (st_blksize of `d`, stat of its first file)"));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(doc["storage_reported"]["b"], json!({"from": "st_blksize", "of": "d", "value": st, "reported": true}));
    assert_eq!(doc["params"]["b"], json!(st), "the parameters in effect hold the number");
    // a dry run told the run's number has the run's fingerprint
    let (_, told, _) = aeiou(&["dry-run", a, "--gpus", "1", "--st-blksize", &st.to_string()]);
    assert_eq!(line(&told, "fingerprint").split_whitespace().nth(1), doc["verdict"]["fingerprint"].as_str());
}

// ---------------------------------------------------------------- the lock

/// An abstract whose op stream reads `k` files of dataset `d` (four of 8 KiB) in requests of
/// `b` bytes, one take of 1 ms compute after, and a lock of it in a `workloads/` tree.
struct Tree {
    _dir: crate::common::TempDir,
    workloads: std::path::PathBuf,
    lock: std::path::PathBuf,
    root: std::path::PathBuf,
}

fn toy() -> String {
    json!({
        "ast": "0.11", "name": "t",
        "params": {"b": {"default": 4096, "unit": "bytes"}, "k": {"kind": "int", "unit": "count"}, "samples": {"default": 4, "scale": true}},
        "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/f{id:04}", "count": {"param": "samples"}, "size": {"const": 8192}, "seed": 1}},
        "actors": {"gpu": {"body": [
            {"loader": {"name": "c", "index": "i", "workers": 1, "prefetch": 1, "batches": {"param": "k"}, "body": [
                {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": {"mod": [{"index": "i"}, 4]}}}}},
                {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
                {"read": {"file": {"ref": "f"}, "len": {"param": "b"}, "repeat": "until_eof"}},
                {"close": {"file": {"ref": "f"}}}]}},
            {"loop": {"index": "s", "to": {"param": "k"}, "body": [{"take": {"channel": "c"}}, {"compute": {"ns": 1000000}}]}}]}}
    })
    .to_string()
}

fn tree(params: Value, top: Value) -> Tree {
    let dir = tmpdir("lock");
    let workloads = dir.join("workloads");
    let mech = workloads.join("cat/toy");
    std::fs::create_dir_all(mech.join("cases/job")).unwrap();
    let text = toy();
    std::fs::write(mech.join("t.ast.json"), &text).unwrap();
    let sha = aeiou::load_str(&text).unwrap().sha256;
    let sources: serde_json::Map<String, Value> = params.as_object().unwrap().keys().map(|k| (k.clone(), json!("cat/toy/cases/job/job.params.json"))).collect();
    let mut lock = json!({"lock_version": 1, "name": "toy/job",
        "runs": [{"name": "t", "abstract": "cat/toy/t.ast.json", "ast_sha256": sha, "params": params, "sources": sources}],
        "provenance": {"tool": "tests/cases.rs"}});
    for (k, v) in top.as_object().unwrap() {
        if k == "run" {
            lock["runs"][0]["run"] = v.clone();
        } else {
            lock[k.as_str()] = v.clone();
        }
    }
    let path = mech.join("cases/job/case.lock.json");
    std::fs::write(&path, serde_json::to_string_pretty(&lock).unwrap()).unwrap();
    let root = dir.join("root");
    Tree { _dir: dir, workloads, lock: path, root }
}

impl Tree {
    fn args<'a>(&'a self, sub: &'a str, more: &[&'a str]) -> Vec<String> {
        let mut v = vec![sub.to_string(), "--case-file".into(), self.lock.display().to_string(), "--workloads".into(), self.workloads.display().to_string()];
        v.extend(more.iter().map(|s| s.to_string()));
        v
    }

    fn aeiou(&self, sub: &str, more: &[&str]) -> (i32, String, String) {
        let a = self.args(sub, more);
        aeiou(&a.iter().map(String::as_str).collect::<Vec<_>>())
    }
}

fn id_of(out: &str) -> String {
    line(out, "workload id ").trim_start_matches("workload id ").to_string()
}

#[test]
fn a_lock_is_run_by_its_values_states_and_sources_and_has_a_workload_id() {
    let t = tree(json!({"b": {"locked": 2048}, "k": {"default": 3}, "samples": {"required": true}}), json!({}));
    // the submission's required value: given, and the id leaves it out
    let (code, out, err) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8"]);
    assert_eq!(code, 0, "{err}");
    assert!(line(&out, "case toy/job  run t  lock ").ends_with(&format!("sha256 {}", aeiou::case::Lock::load(&t.lock).unwrap().sha256)));
    assert_eq!(line(&out, "  b = "), "  b = 2048  locked  cat/toy/cases/job/job.params.json");
    assert_eq!(line(&out, "  k = "), "  k = 3  default  cat/toy/cases/job/job.params.json");
    assert_eq!(line(&out, "  samples = "), "  samples = 8  given  --param");
    let id = id_of(&out);
    let (_, other, _) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=16"]);
    assert_eq!(id_of(&other), id, "a required value is the submission's: not in the id");
    let (_, more, _) = t.aeiou("dry-run", &["--gpus", "4", "--param", "samples=8", "--seed", "3"]);
    assert_ne!(id_of(&more), id, "the seed is an identity option");
    // a default replaced is another workload; a locked value is final; a required one missing is refused
    let (_, k4, _) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8", "--param", "k=4"]);
    assert_ne!(id_of(&k4), id);
    assert!(line(&k4, "  k = ").ends_with("given  --param"));
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8", "--param", "b=4096"]);
    assert!(code == 1 && err.contains("--param b: locked by lock toy/job"), "{err}");
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1"]);
    assert!(code == 1 && err.contains("required parameters with no value: samples"), "{err}");
    // the checks before anything else
    let (code, _, _) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8", "--expect-workload-id", &id]);
    assert_eq!(code, 0);
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8", "--expect-workload-id", &"0".repeat(64)]);
    assert!(code == 1 && err.contains(&format!("workload id {id} does not match the expected")), "{err}");
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8", "--expect-case-sha", &"0".repeat(64)]);
    assert!(code == 1 && err.contains("its file SHA-256 is"), "{err}");
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--expect-case-sha", "xyz"]);
    assert!(code == 2 && err.contains("--expect-case-sha xyz: not a SHA-256"), "{err}");
    // the abstract the lock pins, and no other
    std::fs::write(t.workloads.join("cat/toy/t.ast.json"), toy().replace("\"seed\":1", "\"seed\":2")).unwrap();
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--param", "samples=8"]);
    assert!(code == 1 && err.contains("pins cat/toy/t.ast.json at") && err.contains("the abstract changed since the lock was made"), "{err}");
}

#[test]
fn a_lock_is_found_by_its_name_and_one_run_of_several_is_named() {
    let t = tree(json!({"k": {"default": 1}}), json!({}));
    let w = t.workloads.display().to_string();
    for name in ["toy/job", "job"] {
        let (code, out, err) = aeiou(&["dry-run", "--case", name, "--workloads", &w, "--gpus", "1"]);
        assert_eq!(code, 0, "{name}: {err}");
        assert!(line(&out, "case ").starts_with("case toy/job  run t"));
    }
    let (code, _, err) = aeiou(&["dry-run", "--case", "nope", "--workloads", &w, "--gpus", "1"]);
    assert!(code == 2 && err.contains("--case nope: no lock of that name under"), "{err}");
    // a second mechanism with a case of the same name: the short name matches both
    let other = t.workloads.join("cat/other/cases/job");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("case.lock.json"), std::fs::read_to_string(&t.lock).unwrap().replace("toy/job", "other/job")).unwrap();
    let (code, _, err) = aeiou(&["dry-run", "--case", "job", "--workloads", &w, "--gpus", "1"]);
    assert!(code == 2 && err.contains("--case job: 2 locks match (other/job at") && err.contains("name it as mechanism/case"), "{err}");
    // the options of a lock without a lock; a lock and an abstract
    let (code, _, err) = aeiou(&["dry-run", "--case-run", "t", "--gpus", "1", t.workloads.join("cat/toy/t.ast.json").to_str().unwrap()]);
    assert!(code == 2 && err.contains("--case-run: options of a use case's lock, and no --case or --case-file names one"), "{err}");
    let (code, _, _) = aeiou(&["dry-run", "--case", "toy/job", t.workloads.join("cat/toy/t.ast.json").to_str().unwrap(), "--gpus", "1"]);
    assert_eq!(code, 2);
    // --case-run: for a lock of several runs only
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--case-run", "t"]);
    assert!(code == 2 && err.contains("--case-run t: lock toy/job has one run (t)"), "{err}");
    let mut two: Value = serde_json::from_str(&std::fs::read_to_string(&t.lock).unwrap()).unwrap();
    let mut second = two["runs"][0].clone();
    second["name"] = json!("u");
    two["runs"].as_array_mut().unwrap().push(second);
    std::fs::write(&t.lock, two.to_string()).unwrap();
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1"]);
    assert!(code == 2 && err.contains("lock toy/job has 2 runs (t, u): name one with --case-run"), "{err}");
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1", "--case-run", "v"]);
    assert!(code == 2 && err.contains("--case-run v: lock toy/job has no run of that name (t, u)"), "{err}");
}

#[test]
fn a_lock_sets_and_locks_the_identity_options() {
    let t = tree(json!({"k": {"default": 2}}), json!({"run": {"seed": {"default": 7}, "posix": {"locked": "sync"}}}));
    let (code, out, err) = t.aeiou("dry-run", &["--gpus", "1"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("seed 7  gpus 1"), "the lock's seed where none is given: {out}");
    let (_, s3, _) = t.aeiou("dry-run", &["--gpus", "1", "--seed", "3"]);
    assert!(s3.contains("seed 3  gpus 1"), "a default the command line replaces");
    let r = t.root.display().to_string();
    let (code, _, err) = t.aeiou("run", &["--gpus", "1", "--root", &r, "--posix", "io_uring"]);
    assert!(code == 2 && err.contains("--posix io_uring: the lock toy/job locks posix at sync"), "{err}");
    // the rules' freeze: every identity option at the lock's value, else its default
    let t = tree(json!({"k": {"default": 2}}), json!({"freeze": {"identity_options": "locked"}, "run": {"seed": {"locked": 7}}}));
    let r = t.root.display().to_string();
    for (flag, v, want) in [("--seed", "3", "--seed 3: the lock toy/job locks seed at 7"), ("--time-scale", "2", "--time-scale 2: the lock toy/job locks time_scale at 1 (its rules freeze every identity option at its default)"), ("--rank-rotate", "1", "locks rank_rotate at 0")] {
        let (code, _, err) = t.aeiou("run", &["--gpus", "1", "--root", &r, flag, v]);
        assert!(code == 2 && err.contains(want), "{flag}: {err}");
    }
    let (code, _, err) = t.aeiou("run", &["--gpus", "1", "--root", &r, "--posix", "io_uring"]);
    assert!(code == 2 && err.contains("--posix io_uring: the lock toy/job freezes every identity option, posix at the abstract's sync"), "{err}");
}

#[test]
fn a_locks_constraints_are_judged_when_their_names_are_known() {
    // the AST form aeiou-params lock writes; the runner reads the expression only
    let constraints = json!([
        {"text": "fact.min_samples_per_gpu >= 2", "expr": {"ge": [{"fact": "min_samples_per_gpu"}, 2]}, "names": ["fact.min_samples_per_gpu"], "doc": "a floor per GPU"},
        {"text": "fact.dataset_bytes <= 10 * fact.sum_host_mem", "expr": {"le": [{"fact": "dataset_bytes"}, {"mul": [10, {"fact": "sum_host_mem"}]}]}, "names": ["fact.dataset_bytes", "fact.sum_host_mem"]},
        {"text": "result.busy_fraction > 2", "expr": {"gt": [{"result": "busy_fraction"}, 2]}, "names": ["result.busy_fraction"]}]);
    let t = tree(json!({"k": {"default": 3}, "samples": {"default": 4}}), json!({"constraints": constraints}));
    let (code, out, err) = t.aeiou("dry-run", &["--gpus", "2"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "constraint at resolve"), "constraint at resolve: pass fact.min_samples_per_gpu >= 2  [a floor per GPU]  (fact.min_samples_per_gpu = 2)");
    assert_eq!(line(&out, "constraint checked at the gate"), "constraint checked at the gate: fact.dataset_bytes <= 10 * fact.sum_host_mem");
    assert!(out.contains("constraint checked at after the run: result.busy_fraction > 2"), "{out}");
    // one that fails at resolve: an invalid run, refused before anything
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "4"]);
    assert!(code == 1 && err.contains("constraint(s) not met at resolve, an invalid run, refused before any I/O: fact.min_samples_per_gpu >= 2"), "{err}");
    // the run: the gate passes on this host's memory, the run completes, the result does not pass
    let r = t.root.display().to_string();
    let (code, _, err) = t.aeiou("datagen", &["--root", &r]);
    assert_eq!(code, 0, "{err}");
    let report = t.root.join("../report.json");
    let (code, out, err) = t.aeiou("run", &["--gpus", "2", "--root", &r, "--report-json", report.to_str().unwrap()]);
    assert_eq!(code, 3, "{out}\n{err}");
    assert!(line(&out, "constraint at the gate").starts_with("constraint at the gate: pass fact.dataset_bytes <= 10 * fact.sum_host_mem  (fact.dataset_bytes = 32768, fact.sum_host_mem = "), "{out}");
    assert!(line(&out, "constraint at after the run").starts_with("constraint at after the run: FAIL result.busy_fraction > 2  (result.busy_fraction = "), "{out}");
    assert!(err.contains("did not pass: result.busy_fraction > 2"), "{err}");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(doc["verdict"]["error"], Value::Null, "a valid run that did not pass: no error");
    let verdicts = doc["constraints"].as_array().unwrap();
    assert_eq!(verdicts.iter().map(|v| (v["when"].as_str().unwrap(), v["passed"].as_bool().unwrap())).collect::<Vec<_>>(), [("resolve", true), ("the gate", true), ("after the run", false)]);
    assert_eq!(doc["case"]["name"], "toy/job");
    assert_eq!(doc["case"]["lock"]["name"], "toy/job", "the lock embedded");
    assert_eq!(doc["case"]["params"]["k"], json!({"value": 3, "state": "default", "source": "cat/toy/cases/job/job.params.json"}));
    // a gate constraint that fails: refused before any I/O, nothing read
    let mut lock: Value = serde_json::from_str(&std::fs::read_to_string(&t.lock).unwrap()).unwrap();
    lock["constraints"] = json!([{"text": "fact.sum_host_mem >= 1000000000000000000", "expr": {"ge": [{"fact": "sum_host_mem"}, 1_000_000_000_000_000_000i64]}, "names": ["fact.sum_host_mem"]}]);
    std::fs::write(&t.lock, lock.to_string()).unwrap();
    let (code, out, err) = t.aeiou("run", &["--gpus", "2", "--root", &r]);
    assert!(code == 1 && err.contains("constraint(s) not met at the gate, an invalid run, refused before any I/O: fact.sum_host_mem >= 1000000000000000000"), "{err}");
    assert!(!out.contains("\nfingerprint ") && !out.contains("start gate"), "no op was issued: {out}");
    // a name the runner does not define fails at resolve, never after a long run
    lock["constraints"] = json!([{"text": "result.min_gpu_busyness > 0.9", "expr": {"gt": [{"result": "min_gpu_busyness"}, 0.9]}, "names": ["result.min_gpu_busyness"]}]);
    std::fs::write(&t.lock, lock.to_string()).unwrap();
    let (code, _, err) = t.aeiou("dry-run", &["--gpus", "2"]);
    assert!(code == 1 && err.contains("constraint `result.min_gpu_busyness > 0.9`: `result.min_gpu_busyness` is not a name of the runner"), "{err}");
}

#[test]
fn datagen_writes_the_corpus_of_a_locks_values() {
    let t = tree(json!({"k": {"default": 1}, "samples": {"locked": 6}}), json!({}));
    let r = t.root.display().to_string();
    let (code, out, err) = t.aeiou("datagen", &["--root", &r]);
    assert_eq!(code, 0, "{err}");
    assert!(line(&out, "case toy/job  run t").contains(" sha256 "), "{out}");
    assert_eq!(std::fs::read_dir(t.root.join("d")).unwrap().filter(|e| !e.as_ref().unwrap().file_name().to_string_lossy().starts_with('.')).count(), 6);
    let (code, _, err) = t.aeiou("datagen", &["--root", &r, "--expect-workload-id", &"0".repeat(64)]);
    assert!(code == 2 && err.contains("--expect-workload-id: a run's check"), "{err}");
}

/// A gate constraint on several hosts: rank 0 judges every host's facts before the gate opens,
/// and every host is refused.
#[test]
fn a_gate_constraint_is_judged_by_rank_0_on_every_host_and_refuses_them_all() {
    let constraints = json!([{"text": "fact.hosts >= 3", "expr": {"ge": [{"fact": "hosts"}, 3]}, "names": ["fact.hosts"]}]);
    let t = tree(json!({"k": {"default": 2}}), json!({"constraints": constraints}));
    let r = t.root.display().to_string();
    let (code, _, err) = t.aeiou("datagen", &["--root", &r]);
    assert_eq!(code, 0, "{err}");
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let spawn = |rank: &str| {
        let a = t.args("run", &["--gpus", "2", "--root", &r, "--ranks", "2", "--rank", rank, "--coordinator", &format!("127.0.0.1:{port}")]);
        Command::new(env!("CARGO_BIN_EXE_aeiou")).args(&a).env_remove("AEIOU_CONFIG").stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap()
    };
    let c0 = spawn("0");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let c1 = spawn("1");
    for c in [c0, c1] {
        let o = c.wait_with_output().unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(1), "{err}");
        assert!(err.contains("fact.hosts >= 3"), "{err}");
        assert!(!String::from_utf8_lossy(&o.stdout).contains("start gate:"), "the gate never opened");
    }
}
