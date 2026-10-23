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
        "ast": "0.12", "name": "t", "params": params,
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
    for (bad, want) in [("e=8192", "--st-blksize e=8192: no storage-reported parameter reads `e`"), ("0", "--st-blksize 0: expected a positive byte count, or NAME=BYTES")] {
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

// ---------------------------------------------------------------- a namespace's st_blksize

/// An abstract over namespace `o` (`out/o{n}`) whose one actor writes object 0 in two writes
/// of `b` bytes (`input`: reads it, 8 KiB, in requests of `b` instead), with `params` declared and
/// `more` merged into the document (a dataset, to make a name ambiguous).
fn ns_doc(params: Value, input: bool, more: Value) -> String {
    let mut ns = json!({"pattern": "out/o{n}", "fields": {"n": "int"}, "size": "as_written", "seed": 7});
    let body = if input {
        // the writer's two writes of 4096 bytes: a size the reader can read to its end
        ns["input"] = json!(true);
        ns["size"] = json!(8192);
        json!([{"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"n": 0}}}}},
               {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
               {"read": {"file": {"ref": "f"}, "len": {"param": "b"}, "repeat": "until_eof"}},
               {"close": {"file": {"ref": "f"}}}])
    } else {
        json!([{"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"n": 0}}}}},
               {"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT", "TRUNC"]}},
               {"write": {"file": {"ref": "f"}, "len": {"param": "b"}}},
               {"write": {"file": {"ref": "f"}, "len": {"param": "b"}}},
               {"close": {"file": {"ref": "f"}}}])
    };
    let mut d = json!({"ast": "0.12", "name": if input { "r" } else { "w" }, "params": params, "namespaces": {"o": ns}, "actors": {"gpu": {"body": body}}});
    for (k, v) in more.as_object().unwrap() {
        d[k] = v.clone();
    }
    d.to_string()
}

fn from_o() -> Value {
    json!({"b": {"default": {"from": "st_blksize", "of": "o", "fallback": 4096}}})
}

/// Contract 0.12: the `of` of a storage-reported value may name a namespace; a name declared
/// as both a dataset and a namespace says neither, and one declared as neither is refused.
#[test]
fn a_storage_reported_value_may_name_a_namespace_never_an_ambiguous_name() {
    assert!(aeiou::load_str(&ns_doc(from_o(), false, json!({}))).is_ok());
    let both = json!({"datasets": {"o": {"layout": "whole", "access": "indexed", "pattern": "d/f{id}", "count": 1, "size": {"const": 1}, "seed": 1}}});
    let e = aeiou::load_str(&ns_doc(from_o(), false, both)).err().unwrap();
    assert!(format!("{e:#}").contains("params/b/default/of: `o` is both a dataset and a namespace (V28)"), "{e:#}");
    let e = aeiou::load_str(&doc(json!({"b": {"default": {"from": "st_blksize", "of": "nope", "fallback": 4096}}}))).err().unwrap();
    assert_eq!(format!("{e:#}"), "invalid abstract:\n  t: params/b/default/of: `nope` is not a declared dataset or namespace (V28)");
}

/// An output namespace is empty at the start, so `aeiou run` reads its `st_blksize` from a
/// probe file it creates under the namespace's root and removes before the gate (one per
/// host, named apart); another host's probe there does not make the root not empty, and is
/// left alone. A dry run is told the number by the namespace's name.
#[test]
fn an_output_namespace_reports_st_blksize_by_a_probe_file_it_removes() {
    let dir = tmpdir("ns-st-blksize");
    let ast = dir.join("w.ast.json");
    std::fs::write(&ast, ns_doc(from_o(), false, json!({}))).unwrap();
    let a = ast.to_str().unwrap();
    let (code, out, err) = aeiou(&["dry-run", a, "--gpus", "1", "--st-blksize", "o=8192"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "storage-reported"), "storage-reported: b=8192 (st_blksize of `o`, --st-blksize)");
    for (bad, want) in [("e=8192", "--st-blksize e=8192: no storage-reported parameter reads `e`"), ("x", "--st-blksize x: expected a positive byte count, or NAME=BYTES")] {
        let (code, _, err) = aeiou(&["dry-run", a, "--gpus", "1", "--st-blksize", bad]);
        assert_eq!(code, 2);
        assert!(err.contains(want), "{err}");
    }

    use std::os::unix::fs::MetadataExt;
    let root = dir.join("root");
    std::fs::create_dir_all(root.join("out")).unwrap();
    let other = root.join("out/.aeiou-blksize-otherhost-1");
    std::fs::write(&other, b"").unwrap();
    let st = std::fs::metadata(&other).unwrap().blksize();
    let report = dir.join("report.json");
    let (code, out, err) = aeiou(&["run", a, "--gpus", "1", "--root", root.to_str().unwrap(), "--report-json", report.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "storage-reported"), format!("storage-reported: b={st} (st_blksize of `o`, stat of a probe file in its root)"));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(doc["storage_reported"]["b"], json!({"from": "st_blksize", "of": "o", "value": st, "reported": true}));
    let mut names: Vec<String> = std::fs::read_dir(root.join("out")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    assert_eq!(names, [".aeiou-blksize-otherhost-1", ".aeiou-namespace.json", "o0"], "our probe is gone, the other host's left");
    assert_eq!(std::fs::metadata(root.join("out/o0")).unwrap().len(), 2 * st as u64);

    // a fresh root, which the run creates: the probe goes where the namespace will be
    let fresh = dir.join("fresh");
    let (code, out, err) = aeiou(&["run", a, "--gpus", "1", "--root", fresh.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "storage-reported"), format!("storage-reported: b={st} (st_blksize of `o`, stat of a probe file in its root)"));
}

/// An input namespace is read-only (V14), so it is never probed: its `st_blksize` is that of
/// the first object its manifest lists, or of the manifest when it lists none.
#[test]
fn an_input_namespace_reports_the_st_blksize_of_its_first_object() {
    let dir = tmpdir("in-st-blksize");
    let (w, r) = (dir.join("w.ast.json"), dir.join("r.ast.json"));
    std::fs::write(&w, ns_doc(json!({"b": {"default": 4096}}), false, json!({}))).unwrap();
    std::fs::write(&r, ns_doc(from_o(), true, json!({}))).unwrap();
    let root = dir.join("root");
    let rt = root.to_str().unwrap();
    let (code, _, err) = aeiou(&["run", w.to_str().unwrap(), "--gpus", "1", "--root", rt]);
    assert_eq!(code, 0, "{err}");
    use std::os::unix::fs::MetadataExt;
    let st = std::fs::metadata(root.join("out/o0")).unwrap().blksize();
    let (code, out, err) = aeiou(&["run", r.to_str().unwrap(), "--gpus", "1", "--root", rt]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "storage-reported"), format!("storage-reported: b={st} (st_blksize of `o`, stat of its first object)"));
    let names: Vec<String> = std::fs::read_dir(root.join("out")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    assert!(!names.iter().any(|n| n.starts_with(".aeiou-blksize")), "an input root is never written: {names:?}");
    // a manifest that lists no objects (a namespace past the listing's limit)
    let mpath = root.join("out/.aeiou-namespace.json");
    let mut m: Value = serde_json::from_str(&std::fs::read_to_string(&mpath).unwrap()).unwrap();
    m.as_object_mut().unwrap().remove("objects");
    std::fs::write(&mpath, m.to_string()).unwrap();
    let (code, out, err) = aeiou(&["run", r.to_str().unwrap(), "--gpus", "1", "--root", rt]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(line(&out, "storage-reported"), format!("storage-reported: b={st} (st_blksize of `o`, stat of its manifest)"));
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
        "ast": "0.12", "name": "t",
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
    // the step times (DESIGN_REVIEW §3.78 point 15): 2 GPUs × 3 steps of compute, 2 × 2 delays
    // (an instance's last step has none), the slowest first, each with its instance and step
    let steps = &doc["result"]["steps"];
    assert_eq!((steps["compute"]["count"].as_u64(), steps["delay"]["count"].as_u64()), (Some(6), Some(4)), "{steps}");
    assert!(steps["compute"]["p50_ns"].as_u64().unwrap() >= 500_000, "1 ms of compute slept, a bucket's lower bound: {steps}");
    let slow = steps["slowest_delays"].as_array().unwrap();
    assert_eq!(slow.len(), 4);
    assert!(slow.windows(2).all(|w| w[0]["delay_ns"].as_u64() >= w[1]["delay_ns"].as_u64()));
    assert!(slow.iter().all(|d| d["template"] == "gpu" && d["step"].as_u64().unwrap() < 2 && d["instance"].as_u64().unwrap() < 2));
    assert!(out.contains("step compute (6 steps): p50 ") && out.contains("  slowest delays: "), "{out}");
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

/// Bug A: a comparison of two integers is exact. A byte count passes 2^53 at 9 PB, where an
/// f64 no longer holds every integer, and a capacity constraint compares two such counts.
#[test]
fn a_constraint_compares_integers_as_integers() {
    use aeiou::case::{eval, Num};
    let e = |v: Value| eval(&v, &|_| None).map_err(|e| format!("{e:#}"));
    let big = 1i64 << 53;
    assert_eq!(e(json!({"eq": [big + 1, big]})), Ok(Num::I(0)));
    assert_eq!(e(json!({"ne": [big + 1, big]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"gt": [big + 1, big]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"ge": [big, big + 1]})), Ok(Num::I(0)));
    assert_eq!(e(json!({"lt": [big, big + 1]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"le": [big + 1, big]})), Ok(Num::I(0)));
    // an integer and a float compare as floats
    assert_eq!(e(json!({"lt": [1, 1.5]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"eq": [2, 2.0]})), Ok(Num::I(1)));
}

/// Bug B: a division by zero is refused, never an infinity or a NaN that passes or fails a
/// constraint silently, and the refusal says so.
#[test]
fn a_constraint_refuses_a_division_by_zero() {
    use aeiou::case::eval;
    for op in ["div", "mod"] {
        for (x, y) in [(json!(7), json!(0)), (json!(7.5), json!(0.0)), (json!(7), json!(0.0))] {
            let mut m = serde_json::Map::new();
            m.insert(op.into(), json!([x, y]));
            let err = eval(&Value::Object(m), &|_| None).map(|n| n.show()).map_err(|e| format!("{e:#}"));
            assert_eq!(err, Err(format!("`{op}` by zero")), "{op} {x} {y}");
        }
    }
}

/// Every operator of a constraint's expression, and what is not one.
#[test]
fn a_constraints_expression_takes_its_operators_and_refuses_others() {
    use aeiou::case::{eval, Num};
    let get = |n: &str| (n == "fact.x").then_some(Num::I(3));
    let e = |v: Value| eval(&v, &get).map_err(|e| format!("{e:#}"));
    assert_eq!(e(json!({"and": [1, {"not": 0}]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"and": [1, false]})), Ok(Num::I(0)));
    assert_eq!(e(json!({"or": [0, 0.5]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"or": [false, 0]})), Ok(Num::I(0)));
    assert_eq!(e(json!({"not": true})), Ok(Num::I(0)));
    assert_eq!(e(json!({"neg": {"fact": "x"}})), Ok(Num::I(-3)));
    assert_eq!(e(json!({"neg": 2.5})), Ok(Num::F(-2.5)));
    assert_eq!(e(json!({"div": [-7, 2]})), Ok(Num::I(-4)), "Euclidean, as the abstract's");
    assert_eq!(e(json!({"mod": [-7, 2]})), Ok(Num::I(1)));
    assert_eq!(e(json!({"div": [7.0, 2]})), Ok(Num::F(3.0)));
    assert_eq!(e(json!({"sub": [1, 0.5]})), Ok(Num::F(0.5)));
    assert_eq!(e(json!({"min": [{"fact": "x"}, 2]})), Ok(Num::I(2)));
    assert_eq!(e(json!({"max": [1.5, 2]})), Ok(Num::F(2.0)));
    assert_eq!(e(json!({"neg": i64::MIN})), Err(format!("`neg` of {}", i64::MIN)));
    assert_eq!(e(json!({"add": [i64::MAX, 1]})), Err(format!("`add` of {} and 1 is not an integer of 64 bits", i64::MAX)));
    assert_eq!(e(json!({"add": [1]})), Err("`add` takes 2 operands".into()));
    assert_eq!(e(json!({"pow": [1, 2]})), Err("`pow` is not an operator of a constraint".into()));
    assert_eq!(e(json!({"fact": "y"})), Err("`fact.y` is not known here".into()));
    assert_eq!(e(json!("x")), Err("\"x\": not an expression of a constraint".into()));
    assert_eq!(e(json!({"add": [1, 2], "sub": [1, 2]})), Err("{\"add\":[1,2],\"sub\":[1,2]}: not an expression of a constraint".into()));
}

/// Bug C: `--case NAME` names the locks it could not read, so a broken lock of that case is
/// reported as broken rather than as missing.
#[test]
fn a_lock_that_cannot_be_read_is_named_when_none_matches() {
    let t = tree(json!({"k": {"default": 1}}), json!({}));
    let w = t.workloads.display().to_string();
    std::fs::write(&t.lock, r#"{"lock_version": 9, "name": "toy/job"}"#).unwrap();
    let (code, _, err) = aeiou(&["dry-run", "--case", "job", "--workloads", &w, "--gpus", "1"]);
    assert!(code == 2 && err.contains("--case job: no lock of that name under"), "{err}");
    assert!(err.contains(&format!("1 lock could not be read:\n  lock {}: lock_version 9 (this runner reads 1)", t.lock.display())), "{err}");
}

/// What a lock file may hold, each refusal naming the lock.
#[test]
fn a_lock_with_an_unknown_key_or_value_is_refused() {
    let t = tree(json!({"k": {"default": 1}}), json!({}));
    let good: Value = serde_json::from_str(&std::fs::read_to_string(&t.lock).unwrap()).unwrap();
    let refused = |edit: &dyn Fn(&mut Value), text: Option<String>, want: &str| {
        let mut l = good.clone();
        edit(&mut l);
        std::fs::write(&t.lock, text.unwrap_or_else(|| l.to_string())).unwrap();
        let (code, _, err) = t.aeiou("dry-run", &["--gpus", "1"]);
        assert!(code != 0 && err.contains(&format!("lock {}: {want}", t.lock.display())), "{want}: {err}");
    };
    refused(&|l| l["extra"] = json!(1), None, "unknown key `extra`");
    refused(&|l| l["runs"][0]["run"] = json!({"gpus": {"locked": 2}}), None, "runs/0/run: `gpus` is not an identity option (");
    refused(&|_| {}, Some(good.to_string().replace("\"default\":1", "\"default\":18446744073709551616")), "");
    refused(&|l| l["lock_version"] = json!(2), None, "lock_version 2 (this runner reads 1)");
    // --workloads names a directory
    std::fs::write(&t.lock, good.to_string()).unwrap();
    let (code, _, err) = aeiou(&["dry-run", "--case-file", t.lock.to_str().unwrap(), "--gpus", "1", "--workloads", "/nonexistent/aeiou"]);
    assert!(code == 2 && err.contains("--workloads /nonexistent/aeiou: not a directory"), "{err}");
}

/// Without --workloads, aeiou's `workloads/` is the one beside its `bin/`, as a release lays
/// them out; none there is a usage error naming the option.
#[test]
fn a_lock_is_found_in_the_workloads_beside_the_executable() {
    let t = tree(json!({"k": {"default": 1}}), json!({}));
    let place = |dir: &std::path::Path| {
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        let exe = dir.join("bin/aeiou");
        if std::fs::hard_link(env!("CARGO_BIN_EXE_aeiou"), &exe).is_err() {
            std::fs::copy(env!("CARGO_BIN_EXE_aeiou"), &exe).unwrap();
        }
        exe
    };
    let exe = place(t.workloads.parent().unwrap());
    let run = |exe: &std::path::Path| {
        let o = Command::new(exe).args(["dry-run", "--case", "job", "--gpus", "1"]).env_remove("AEIOU_CONFIG").output().unwrap();
        (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (code, out, err) = run(&exe);
    assert_eq!(code, 0, "{err}");
    assert!(line(&out, "case ").starts_with("case toy/job  run t"), "{out}");
    let lone = tmpdir("lone");
    let exe = place(&lone);
    let (code, _, err) = run(&exe);
    assert!(code == 2 && err.contains(&format!("aeiou's workloads/ is not beside {}", exe.display())) && err.contains("name it with --workloads DIR"), "{err}");
}

/// A lock's storage-reported entry keeps the abstract's rule: the number the storage gives is
/// the run's, and the workload id holds the rule, not the number.
#[test]
fn a_storage_reported_value_in_a_lock_is_the_storages() {
    let t = tree(json!({"k": {"default": 1}, "b": {"default": {"from": "st_blksize", "of": "d", "fallback": 4096}}}), json!({}));
    let text = toy().replace("\"b\":{\"default\":4096", "\"b\":{\"default\":{\"fallback\":4096,\"from\":\"st_blksize\",\"of\":\"d\"}");
    assert_ne!(text, toy());
    std::fs::write(t.workloads.join("cat/toy/t.ast.json"), &text).unwrap();
    let mut lock: Value = serde_json::from_str(&std::fs::read_to_string(&t.lock).unwrap()).unwrap();
    lock["runs"][0]["ast_sha256"] = json!(aeiou::load_str(&text).unwrap().sha256);
    std::fs::write(&t.lock, lock.to_string()).unwrap();
    let (code, a, err) = t.aeiou("dry-run", &["--gpus", "1", "--st-blksize", "d=8192"]);
    assert_eq!(code, 0, "{err}");
    assert!(line(&a, "  b = ").starts_with("  b = 8192 "), "{a}");
    let (_, b, _) = t.aeiou("dry-run", &["--gpus", "1", "--st-blksize", "d=65536"]);
    assert!(line(&b, "  b = ").starts_with("  b = 65536 "), "{b}");
    assert_eq!(id_of(&a), id_of(&b), "the rule is in the id, not the storage's number");
    let (_, c, _) = t.aeiou("dry-run", &["--gpus", "1", "--param", "b=8192"]);
    assert_ne!(id_of(&c), id_of(&a), "a number given in its place is another workload");
}

/// The workload id holds the lock's other runs by their entries, a required one included.
#[test]
fn the_workload_id_holds_the_locks_other_runs() {
    let t = tree(json!({"k": {"default": 1}}), json!({}));
    let mut lock: Value = serde_json::from_str(&std::fs::read_to_string(&t.lock).unwrap()).unwrap();
    let mut second = lock["runs"][0].clone();
    second["name"] = json!("u");
    second["params"] = json!({"k": {"required": true}});
    lock["runs"].as_array_mut().unwrap().push(second);
    std::fs::write(&t.lock, lock.to_string()).unwrap();
    let (code, out, err) = t.aeiou("dry-run", &["--gpus", "1", "--case-run", "t"]);
    assert_eq!(code, 0, "{err}");
    lock["runs"][1]["params"] = json!({"k": {"default": 2}});
    std::fs::write(&t.lock, lock.to_string()).unwrap();
    let (_, other, _) = t.aeiou("dry-run", &["--gpus", "1", "--case-run", "t"]);
    assert_ne!(id_of(&other), id_of(&out), "another run of the lock changed: another workload");
}

/// The brief's Unet3D on an H100 is a use case (DESIGN_REVIEW §3.78 points 1 and 18): its lock
/// runs exactly what `npy_per_sample`'s defaults ran until 2026-10-22 (tests/golden.rs's
/// fingerprint), and the abstract alone, whose job and GPU values are now required, is refused.
#[test]
fn the_unet3d_h100_use_case_is_the_run_the_old_defaults_made() {
    let w = crate::common::workloads().display().to_string();
    let (code, out, err) = aeiou(&["dry-run", "--case", "unet3d-h100", "--workloads", &w, "--gpus", "2", "--seed", "1", "--param", "steps=5"]);
    assert_eq!(code, 0, "{err}");
    assert!(line(&out, "case ").starts_with("case npy-per-sample/unet3d-h100  run npy_per_sample"), "{out}");
    assert_eq!(line(&out, "fingerprint "), "fingerprint f755bd4187b75b78");
    assert!(line(&out, "  ops ").starts_with("  ops 50491 "), "{out}");
    let ast = crate::common::workload_ast("npy_per_sample.ast.json").display().to_string();
    let (code, _, err) = aeiou(&["dry-run", &ast, "--gpus", "1"]);
    assert!(code == 1 && err.contains("required parameters with no value: batch, prefetch, sample_mean, sample_sd, samples, step_time, steps, workers"), "{err}");
}

/// The untraced containers' cases (item 33, part 5b; DESIGN_REVIEW §3.78 points 1 and 4): each
/// lock runs exactly what its abstract's defaults ran until 2026-10-22 (the fingerprints taken at
/// cb978da, 2 GPUs, seed 1, 8 steps), and the abstract alone is refused, naming every value it
/// now requires.
#[test]
fn the_untraced_containers_cases_are_the_runs_the_old_defaults_made() {
    let w = crate::common::workloads().display().to_string();
    let stream = "batch, cycle, decode, prefetch, sample_median, samples, samples_per_file, step_time, steps";
    for (case, ast, fingerprint, ops, required) in [
        ("rgb256-rows", "hdf5_map", "32456fa46e566c10", 6144, "batch, prefetch, samples, samples_per_file, step_time, steps, workers"),
        ("imagenet-rows", "parquet_stream", "627b8aaa7cdc75ae", 56, stream),
        ("imagenet-records", "tfrecord_stream", "aa31f120e5b4cd9c", 1941, stream),
    ] {
        let (code, out, err) = aeiou(&["dry-run", "--case", case, "--workloads", &w, "--gpus", "2", "--seed", "1", "--param", "steps=8"]);
        assert_eq!(code, 0, "{case}: {err}");
        assert!(line(&out, "case ").contains(&format!("/{case}  run {ast}")), "{out}");
        assert_eq!(line(&out, "fingerprint "), format!("fingerprint {fingerprint}"), "{case}");
        assert!(line(&out, "  ops ").starts_with(&format!("  ops {ops} ")), "{case}: {out}");
        let file = crate::common::workload_ast(&format!("{ast}.ast.json")).display().to_string();
        let (code, _, err) = aeiou(&["dry-run", &file, "--gpus", "1"]);
        assert!(code == 1 && err.contains(&format!("required parameters with no value: {required}")), "{ast}: {err}");
    }
}

/// dcp_save's unsettled case (item 38; DESIGN_REVIEW §3.78 points 1 and 4): its lock runs exactly
/// what the abstract's defaults ran until 2026-10-23 (tests/golden.rs's fingerprint, 2 GPUs,
/// seed 1, 200 steps), the shard's `buf` the namespace's st_blksize, its fallback in a dry run;
/// the abstract alone is refused, naming every value it now requires.
#[test]
fn the_dcp_save_case_is_the_run_the_old_defaults_made() {
    let w = crate::common::workloads().display().to_string();
    let (code, out, err) = aeiou(&["dry-run", "--case", "four-items", "--workloads", &w, "--gpus", "2", "--seed", "1", "--param", "steps=200"]);
    assert_eq!(code, 0, "{err}");
    assert!(line(&out, "case ").contains("/four-items  run dcp_save"), "{out}");
    assert_eq!(line(&out, "fingerprint "), "fingerprint 850e8c9019b3bda8");
    assert!(line(&out, "  ops ").starts_with("  ops 130 "), "{out}");
    assert_eq!(line(&out, "storage-reported"), "storage-reported: buf=1048576 (st_blksize of `ckpt`: the fallback, none reported)  xfer=1048576 (st_blksize of `ckpt`: the fallback, none reported)");
    let file = crate::common::workload_ast("dcp_save.ast.json").display().to_string();
    let (code, _, err) = aeiou(&["dry-run", &file, "--gpus", "1"]);
    assert!(code == 1 && err.contains("required parameters with no value: ckpt_every, item_bytes, items, meta_bytes, step_time, steps"), "{err}");
}

/// file_per_sample's unsettled case (item 33, part 5c; DESIGN_REVIEW §3.78 points 1 and 4): its
/// lock runs exactly what the abstract's defaults ran until 2026-10-22 (the fingerprint taken at
/// 1663e7e, 2 GPUs, seed 1, 8 steps), and the abstract alone is refused, naming every value it
/// now requires.
#[test]
fn the_file_per_sample_case_is_the_run_the_old_defaults_made() {
    let w = crate::common::workloads().display().to_string();
    let (code, out, err) = aeiou(&["dry-run", "--case", "jpeg-50m", "--workloads", &w, "--gpus", "2", "--seed", "1", "--param", "steps=8"]);
    assert_eq!(code, 0, "{err}");
    assert!(line(&out, "case ").contains("/jpeg-50m  run file_per_sample"), "{out}");
    assert_eq!(line(&out, "fingerprint "), "fingerprint b7ed35ea7708d600");
    assert!(line(&out, "  ops ").starts_with("  ops 388716 "), "{out}");
    let file = crate::common::workload_ast("file_per_sample.ast.json").display().to_string();
    let (code, _, err) = aeiou(&["dry-run", &file, "--gpus", "1"]);
    assert!(code == 1 && err.contains("required parameters with no value: batch, prefetch, sample_median, samples, step_time, steps, workers"), "{err}");
}
