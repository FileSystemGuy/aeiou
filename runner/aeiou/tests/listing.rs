//! `aeiou cases` (item 40, DESIGN_REVIEW §3.80): the use cases a binary can run, listed from
//! the `catalog.json` that `aeiou-catalog write` generates beside `CATALOG.md`, filtered by the
//! catalog's attributes, one shown with the commands that run it; the vocabulary in `--help`;
//! a `--case` that matches nothing naming the near matches.

use std::process::Command;

use serde_json::Value;

/// The binary on `args`, the committed workloads/ named after `cases` unless `args` names one
/// (the default beside the executable is not there under `cargo llvm-cov`, as for tests/cases.rs).
fn aeiou_env(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let w = crate::common::workloads().display().to_string();
    let mut args: Vec<&str> = args.to_vec();
    if !args.contains(&"--workloads") {
        if let Some(i) = args.iter().position(|a| *a == "cases") {
            args.splice(i + 1..i + 1, ["--workloads", w.as_str()]);
        }
    }
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.args(&args).env_remove("AEIOU_CONFIG").env_remove("AEIOU_WHERE");
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn aeiou(args: &[&str]) -> (i32, String, String) {
    aeiou_env(args, &[])
}

fn index() -> Value {
    serde_json::from_slice(&std::fs::read(crate::common::workloads().join("catalog.json")).unwrap()).unwrap()
}

fn names(cases: &[&Value]) -> Vec<String> {
    cases.iter().map(|c| c["name"].as_str().unwrap().to_string()).collect()
}

/// The case's values of an attribute: its own and its mechanism's, and every run's.
fn values_of(case: &Value, attr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut add = |v: &Value| match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => out.extend(a.iter().filter_map(|x| x.as_str().map(str::to_string))),
        _ => {}
    };
    add(&case["attributes"][attr]);
    for r in case["runs"].as_array().unwrap() {
        add(&r["attributes"][attr]);
    }
    out
}

/// The names of the cases a listing printed, in its order: the lines that start with a name.
fn listed(out: &str) -> Vec<String> {
    let all = index();
    let known: Vec<String> = names(&all["cases"].as_array().unwrap().iter().collect::<Vec<_>>());
    out.lines().filter_map(|l| l.split_whitespace().next()).filter(|w| known.iter().any(|k| k == w)).map(str::to_string).collect()
}

/// A bare `aeiou cases` lists every case of the catalog, of every kind, in its order, each with
/// its kind, GPU counts, and category, and what each run does (the AST's doc); the options it
/// resolved come first, as every subcommand's.
#[test]
fn cases_lists_every_case_of_the_catalog() {
    let all = index();
    let cases: Vec<&Value> = all["cases"].as_array().unwrap().iter().collect();
    let (code, out, err) = aeiou(&["cases"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.starts_with("options (cli > env > config > default)"), "{out}");
    assert_eq!(listed(&out), names(&cases));
    assert!(out.contains(&format!("{n} of {n} cases", n = cases.len())), "{out}");
    let faiss = out.lines().find(|l| l.starts_with("faiss-ivf-ondisk/sift1m-ivf1024-pq32 ")).unwrap();
    for word in ["example", "1 GPU", "vector-search"] {
        assert!(faiss.contains(word), "{faiss}");
    }
    let doc = cases.iter().find(|c| c["name"] == "faiss-ivf-ondisk/sift1m-ivf1024-pq32").unwrap()["runs"][0]["does"].as_str().unwrap();
    assert!(out.contains(&format!("    {doc}")), "{out}");
    // a case of two runs: each run's doc under its AST's name
    let shared = cases.iter().find(|c| c["name"] == "lmcache-shared-fs/sharegpt-chat").unwrap();
    for r in shared["runs"].as_array().unwrap() {
        assert!(out.contains(&format!("    {}: {}", r["ast"].as_str().unwrap(), r["does"].as_str().unwrap())), "{out}");
    }
}

/// `--where ATTR=VALUE` keeps the cases with that value, `--where ATTR!=VALUE` drops them, every
/// one given must hold; a case has a value when it or its mechanism records it, or any of its
/// runs derives or measures it; a curated attribute of several values has each.
#[test]
fn where_keeps_and_drops_by_attribute() {
    let all = index();
    let cases: Vec<&Value> = all["cases"].as_array().unwrap().iter().collect();
    let pick = |f: &dyn Fn(&Value) -> bool| names(&cases.iter().copied().filter(|c| f(c)).collect::<Vec<_>>());
    let has = |c: &Value, a: &str, v: &str| values_of(c, a).iter().any(|x| x == v);
    for (args, want) in [
        (vec!["--where", "category=kvcache"], pick(&|c| has(c, "category", "kvcache"))),
        (vec!["--where", "kind!=example"], pick(&|c| !has(c, "kind", "example"))),
        (vec!["--where", "inputs=yes"], pick(&|c| has(c, "inputs", "yes"))),
        (vec!["--where", "domain=ai", "--where", "kind!=example", "--where", "category!=kvcache"],
         pick(&|c| has(c, "domain", "ai") && !has(c, "kind", "example") && !has(c, "category", "kvcache"))),
        (vec!["--where", "api=mmap"], pick(&|c| has(c, "api", "mmap"))),
    ] {
        assert!(!want.is_empty() && want.len() < cases.len(), "{args:?}: a filter that tells something");
        let mut line = vec!["cases"];
        line.extend(&args);
        let (code, out, err) = aeiou(&line);
        assert_eq!(code, 0, "{args:?}: {err}");
        assert_eq!(listed(&out), want, "{args:?}");
        assert!(out.contains(&format!("{} of {} cases", want.len(), cases.len())), "{out}");
    }
    // the shared store's reader reads what its writer wrote; the writer does not
    let (_, out, _) = aeiou(&["cases", "--where", "inputs=yes", "--where", "category=kvcache"]);
    assert!(listed(&out).contains(&"lmcache-shared-fs/sharegpt-chat".to_string()), "{out}");
    // nothing matches: said, not an error
    let (code, out, _) = aeiou(&["cases", "--where", "kind=use", "--where", "kind!=use"]);
    assert_eq!((code, listed(&out).len()), (0, 0), "{out}");
    assert!(out.contains(&format!("0 of {} cases", cases.len())), "{out}");
}

/// A filter outside the dictionary is a usage error naming what is valid: the attributes, or
/// the attribute's values; a filter without `=` or `!=` says the two forms.
#[test]
fn where_refuses_what_the_vocabulary_lacks() {
    let all = index();
    let (code, _, err) = aeiou(&["cases", "--where", "colour=red"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.starts_with("aeiou cases: --where colour=red: no attribute `colour`; the attributes: category, class, "), "{err}");
    assert!(err.contains("\n\nUsage: aeiou cases"), "the usage frame: {err}");
    let kind = all["attributes"].as_array().unwrap().iter().find(|a| a["name"] == "kind").unwrap();
    let kinds: Vec<&str> = kind["values"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let (code, _, err) = aeiou(&["cases", "--where", "kind!=job"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.starts_with(&format!("aeiou cases: --where kind!=job: `job` is not a value of kind ({})", kinds.join(", "))), "{err}");
    for bad in ["kind", "=example", "kind=", "kind==use"] {
        let (code, _, err) = aeiou(&["cases", "--where", bad]);
        assert_eq!(code, 2, "{bad}: {err}");
        assert!(err.contains(&format!("--where {bad}: ATTR=VALUE keeps the cases with the value, ATTR!=VALUE drops them")), "{bad}: {err}");
    }
}

/// `--json`: the options, the workloads directory, and the cases that match, each as
/// catalog.json holds it, so a script reads what the catalog says and nothing else.
#[test]
fn cases_json_is_the_catalogs_rows() {
    let all = index();
    let (code, out, err) = aeiou(&["cases", "--json", "--where", "category=vector-search"]);
    assert_eq!(code, 0, "{err}");
    let doc: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"));
    let want: Vec<&Value> = all["cases"].as_array().unwrap().iter().filter(|c| c["attributes"]["category"] == "vector-search").collect();
    assert_eq!(doc["cases"].as_array().unwrap().iter().collect::<Vec<_>>(), want);
    assert_eq!(doc["total"], all["cases"].as_array().unwrap().len());
    assert_eq!(doc["options"]["options"]["where"]["value"], serde_json::json!(["category=vector-search"]), "{}", doc["options"]);
    assert!(doc["workloads"].as_str().unwrap().ends_with("workloads"), "{}", doc["workloads"]);
}

/// `aeiou cases NAME`: one case, found as `--case` finds a lock (`mechanism/case`, or the case
/// alone when one matches): where its records are, its attributes, and per run what it does,
/// what it is, and what its lock fixes, leaves to a run, and requires of a submission; then the
/// commands that write its data and run it at its first GPU count (`--case-run` for a lock of
/// several runs).
#[test]
fn cases_name_shows_one_and_how_to_run_it() {
    let (code, out, err) = aeiou(&["cases", "sift1m-ivf1024-pq32"]);
    assert_eq!(code, 0, "{err}");
    let name = "faiss-ivf-ondisk/sift1m-ivf1024-pq32";
    for want in [
        format!("case {name}"),
        "  lock         vector-search/faiss-ivf-ondisk/cases/sift1m-ivf1024-pq32/case.lock.json".to_string(),
        "  description  vector-search/faiss-ivf-ondisk/cases/sift1m-ivf1024-pq32/DESCRIPTION.md".to_string(),
        "  gpus         1".to_string(),
        "run faiss_ivf_ondisk".to_string(),
        "  locked       none".to_string(),
        "  required     none".to_string(),
        format!("  aeiou datagen --case {name} --root DIR"),
        format!("  aeiou run --case {name} --gpus 1 --root DIR"),
    ] {
        assert!(out.lines().any(|l| l == want), "no line {want:?} in:\n{out}");
    }
    let defaults = out.lines().find(|l| l.starts_with("  defaults     ")).unwrap();
    assert!(defaults.contains("batch") && defaults.contains("nprobe") && defaults.contains("--param"), "{defaults}");
    assert!(out.contains("kind=example") && out.contains("api=mmap") && out.contains("category=vector-search"), "{out}");
    // a lock of two runs: each run's commands name it
    let (code, out, err) = aeiou(&["cases", "lmcache-shared-fs/sharegpt-chat"]);
    assert_eq!(code, 0, "{err}");
    for run in ["lmcache_shared_fs", "lmcache_shared_fs_reader"] {
        assert!(out.lines().any(|l| l == format!("run {run}")), "{out}");
        assert!(out.contains(&format!("aeiou run --case lmcache-shared-fs/sharegpt-chat --case-run {run} --gpus 1 --root DIR")), "{out}");
    }
    // the JSON form: the catalog's row
    let (_, out, _) = aeiou(&["cases", "--json", "sift1m-ivf1024-pq32"]);
    let doc: Value = serde_json::from_str(&out).unwrap();
    let row = index()["cases"].as_array().unwrap().iter().find(|c| c["name"] == name).cloned().unwrap();
    assert_eq!(doc["case"], row);
    assert_eq!(doc["lock"][0]["required"], serde_json::json!([]));
}

/// A name that matches several cases, or none, is a usage error: several listed as
/// `mechanism/case`; none with the names near it and the listing that has them all.
#[test]
fn cases_name_refuses_none_and_several() {
    let (code, _, err) = aeiou(&["cases", "chat-64-slots"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("chat-64-slots: 2 cases match (lmcache-local-disk/chat-64-slots, lmcache-shared-fs/chat-64-slots): name it as mechanism/case"), "{err}");
    let (code, _, err) = aeiou(&["cases", "sift1m-nocach"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("sift1m-nocach: no case of that name; near it: diskann-search/sift1m-nocache"), "{err}");
    assert!(err.contains("`aeiou cases` lists every one"), "{err}");
    let (code, _, err) = aeiou(&["cases", "zzzz"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("zzzz: no case of that name; `aeiou cases` lists every one"), "{err}");
    // a filter is for the list, not for one case
    let (code, _, err) = aeiou(&["cases", "sift1m-nocache", "--where", "kind=example"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("--where filters the list; NAME shows one case"), "{err}");
}

/// `--case NAME` of a run that matches no lock names the near ones and `aeiou cases`.
#[test]
fn a_case_that_matches_nothing_names_the_near_ones() {
    let w = crate::common::workloads().display().to_string();
    let (code, _, err) = aeiou(&["dry-run", "--case", "sift1m-nocach", "--workloads", &w, "--gpus", "1"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("near it: diskann-search/sift1m-nocache"), "{err}");
    assert!(err.contains("`aeiou cases` lists every one"), "{err}");
    // the help of every command that takes --case says where the names are
    for sub in ["dry-run", "datagen", "run"] {
        let (_, out, _) = aeiou(&[sub, "--help"]);
        let flat = out.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(flat.contains("`aeiou cases` lists them") && flat.contains("CATALOG.md"), "{sub}: {out}");
    }
}

/// `aeiou cases --help` gives the syntax and the vocabulary: every attribute and its values, as
/// catalog.json holds them, without their meanings, which CATALOG.md has.
#[test]
fn cases_help_gives_the_vocabulary() {
    let all = index();
    for flag in ["--help", "-h"] {
        let (code, out, err) = aeiou(&["cases", flag]);
        assert_eq!(code, 0, "{err}");
        assert!(out.contains("--where <ATTR=VALUE|ATTR!=VALUE>"), "{out}");
        assert!(out.contains("CATALOG.md"), "{out}");
        let flat = out.split_whitespace().collect::<Vec<_>>().join(" ");
        for a in all["attributes"].as_array().unwrap() {
            let name = a["name"].as_str().unwrap();
            let values: Vec<&str> = a["values"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            assert!(flat.contains(&format!("{name}: {}", values.join(", "))), "{flag}: {name}: {out}");
        }
    }
    // a tree with no catalog: the help says so, and the syntax is still there
    let empty = crate::common::tmpdir("listing", "no-catalog");
    let (code, out, _) = aeiou(&["cases", "--workloads", empty.to_str().unwrap(), "--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("--where") && out.contains("no catalog.json"), "{out}");
}

/// `where` is an option as any other: from the environment (`AEIOU_WHERE`, whitespace-separated)
/// or a config file's `[cases]` table (an array), under the command line, and the options block
/// says from where. The workloads directory is the command line's alone, as for `--case`.
#[test]
fn where_comes_from_the_environment_or_the_config_file() {
    let all = index();
    let kv = names(&all["cases"].as_array().unwrap().iter().filter(|c| c["attributes"]["category"] == "kvcache").collect::<Vec<_>>());
    let (code, out, err) = aeiou_env(&["cases"], &[("AEIOU_WHERE", "category=kvcache")]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(listed(&out), kv);
    assert!(out.lines().any(|l| l.starts_with("  where = category=kvcache") && l.ends_with("[env AEIOU_WHERE]")), "{out}");
    let dir = crate::common::tmpdir("listing", "config");
    let cfg = dir.join("aeiou.toml");
    std::fs::write(&cfg, "[cases]\nwhere = [\"category=kvcache\", \"kind!=example\"]\n").unwrap();
    let (code, out, err) = aeiou(&["--config", cfg.to_str().unwrap(), "cases"]);
    assert_eq!(code, 0, "{err}");
    let want = names(&all["cases"].as_array().unwrap().iter().filter(|c| c["attributes"]["category"] == "kvcache" && c["attributes"]["kind"] != "example").collect::<Vec<_>>());
    assert_eq!(listed(&out), want);
    // the command line's replaces the file's
    let (_, out, _) = aeiou(&["--config", cfg.to_str().unwrap(), "cases", "--where", "category=checkpointing"]);
    assert!(listed(&out).iter().all(|n| n.starts_with("dcp-")), "{out}");
    std::fs::write(&cfg, "[cases]\nworkloads = \"/tmp\"\n").unwrap();
    let (code, _, err) = aeiou(&["--config", cfg.to_str().unwrap(), "cases"]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("workloads"), "{err}");
}

/// A workloads directory without catalog.json, or with one of another version or not JSON,
/// is refused with what makes it.
#[test]
fn a_missing_or_foreign_catalog_is_refused() {
    let dir = crate::common::tmpdir("listing", "foreign");
    let d = dir.to_str().unwrap();
    let (code, _, err) = aeiou(&["cases", "--workloads", d]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains(&format!("{d}/catalog.json")) && err.contains("aeiou-catalog write"), "{err}");
    std::fs::write(dir.join("catalog.json"), r#"{"catalog_version": 2, "attributes": {}, "cases": []}"#).unwrap();
    let (code, _, err) = aeiou(&["cases", "--workloads", d]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("catalog_version 2 (this runner reads 1)"), "{err}");
    std::fs::write(dir.join("catalog.json"), "not json").unwrap();
    let (code, _, err) = aeiou(&["cases", "--workloads", d]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("catalog.json") && err.contains("not JSON"), "{err}");
}
