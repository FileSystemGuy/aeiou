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
