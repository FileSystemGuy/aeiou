//! `aeiou-trace compare` over two digests (`DESIGN_REVIEW.md` §3.89, stage 4): the structure by
//! equality (templates paired by text, by `--pair`, by shape; each pair's fields, sites and
//! verdicts, op kinds, motifs, and peak of lanes), the runner's own I/O set aside on B, the
//! numbers of the digests' `metrics` sections as a metrics document's, and `--json`. The rows
//! and the judgement are `compare_corpus.rs`'s; these are what digests add.
use std::path::Path;
use std::process::{Command, Output};

use serde_json::{json, Value};

fn tool(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aeiou-trace")).args(args).output().unwrap()
}

fn text(o: &Output) -> (String, String) {
    (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// An strace of reads of whole 4 KiB files under `/d`: per `(pid, path, seek)`, an open, an
/// fstat, a read (with an lseek and a second read when `seek`), and a close, a millisecond
/// apart.
fn strace(calls: &[(u32, &str, bool)]) -> String {
    let mut out = String::new();
    let mut t = 0u64;
    let mut line = |pid: u32, call: String| {
        t += 1000;
        out.push_str(&format!("{pid} 1700000000.{t:06} {call} <0.000005>\n"));
    };
    for (pid, path, seek) in calls {
        let p = format!("/d/{path}");
        line(*pid, format!("openat(AT_FDCWD</d>, \"{p}\", O_RDONLY|O_CLOEXEC) = 3<{p}>"));
        line(*pid, format!("fstat(3<{p}>, {{st_mode=S_IFREG|0644, st_size=4096, ...}}) = 0"));
        line(*pid, format!("read(3<{p}>, \"x\"..., 4096) = 4096"));
        if *seek {
            line(*pid, format!("lseek(3<{p}>, 0, SEEK_SET) = 0"));
            line(*pid, format!("read(3<{p}>, \"x\"..., 4096) = 4096"));
        }
        line(*pid, format!("close(3<{p}>) = 0"));
    }
    out
}

/// The digest of `calls`, written under `dir` as `name.digest.json`.
fn digest(dir: &Path, name: &str, calls: &[(u32, &str, bool)], extra: &[&str]) -> String {
    let trace = dir.join(format!("{name}.strace"));
    std::fs::write(&trace, strace(calls)).unwrap();
    let out = dir.join(format!("{name}.digest.json"));
    let mut args = vec!["digest", s(&trace), "--root", "/d", "--no-questions", "-o", s(&out)];
    args.extend_from_slice(extra);
    let o = tool(&args);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    s(&out).to_string()
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}/{i:03}.npz")).collect()
}

/// `paths` read once each by `pid`, in the order given.
fn reads<'a>(pid: u32, paths: &'a [String], seek: bool) -> Vec<(u32, &'a str, bool)> {
    paths.iter().map(|p| (pid, p.as_str(), seek)).collect()
}

/// A fixed shuffle (by a multiplicative hash of the index): an order the digest calls permuted.
fn shuffled(paths: &[String]) -> Vec<String> {
    let mut idx: Vec<usize> = (0..paths.len()).collect();
    idx.sort_by_key(|i| (*i as u64).wrapping_mul(2654435761) & 0xffff_ffff);
    idx.into_iter().map(|i| paths[i].clone()).collect()
}

fn compare(a: &str, b: &str, extra: &[&str]) -> (i32, String, String) {
    let mut args = vec!["compare", a, b];
    args.extend_from_slice(extra);
    let o = tool(&args);
    let (out, err) = text(&o);
    (o.status.code().unwrap(), out, err)
}

fn json_of(a: &str, b: &str, extra: &[&str]) -> (i32, Value) {
    let mut args = vec!["compare", a, b, "--json"];
    args.extend_from_slice(extra);
    let o = tool(&args);
    (o.status.code().unwrap(), serde_json::from_slice(&o.stdout).unwrap_or_else(|e| panic!("{e}: {}", text(&o).0)))
}

fn aspect<'d>(doc: &'d Value, pair: usize, name: &str) -> &'d Value {
    doc["structure"]["pairs"][pair]["aspects"].as_array().unwrap().iter().find(|x| x["name"] == name).unwrap_or_else(|| panic!("no aspect {name}: {doc}"))
}

#[test]
fn a_digest_against_itself_is_the_same_structure_and_numbers() {
    let d = crate::common::tmpdir("compare-tool", "same");
    let p = names("s", 28);
    let a = digest(&d, "a", &reads(100, &shuffled(&p), false), &[]);
    let (code, out, err) = compare(&a, &a, &["--judge"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.starts_with(&format!("A: {a}: digest of ")), "{out}");
    assert!(out.contains("template s/{a:03}.npz (identical)\n"), "{out}");
    assert!(out.contains("structure: same\n"), "{out}");
    assert!(out.ends_with("not judged\n") && out.lines().last().unwrap().starts_with("accepted: "), "{out}");
    let (code, doc) = json_of(&a, &a, &[]);
    assert_eq!(code, 0);
    assert_eq!(doc["aeiou_compare"], 1);
    assert_eq!(doc["verdict"], "same");
    assert_eq!(doc["structure"]["differences"], 0);
    assert_eq!(aspect(&doc, 0, "site")["a"], "set");
    assert_eq!(aspect(&doc, 0, "order")["verdict"], "same");
    assert_eq!(aspect(&doc, 0, "ops")["a"], "close fstat open read");
    assert_eq!(aspect(&doc, 0, "motifs")["a"], "open fstat read close");
    assert_eq!(aspect(&doc, 0, "peak lanes")["a"], "1");
    assert_eq!(aspect(&doc, 0, "lanes")["verdict"], "shown");
    assert!(doc["rows"].as_array().unwrap().iter().all(|r| r["distance"].as_f64().is_none_or(|x| x == 0.0)), "{doc}");
}

#[test]
fn a_verdict_motif_or_peak_that_differs_is_a_difference() {
    let d = crate::common::tmpdir("compare-tool", "differs");
    let p = names("s", 28);
    let a = digest(&d, "a", &reads(100, &shuffled(&p), false), &[]);
    // in order, so the order verdict is sequential where A's is permuted
    let b = digest(&d, "b", &reads(100, &p, false), &[]);
    let (code, out, err) = compare(&a, &b, &[]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.contains("permuted") && out.contains("sequential") && out.contains("DIFFERENT"), "{out}");
    assert!(out.contains("structure: 1 difference(s)\n"), "{out}");
    assert_eq!(err, "aeiou-trace compare: the structure differs (1 difference(s))\n");
    // an lseek and a second read per file: another motif and another op kind
    let c = digest(&d, "c", &reads(100, &shuffled(&p), true), &[]);
    let (code, doc) = json_of(&a, &c, &[]);
    assert_eq!(code, 1);
    assert_eq!(aspect(&doc, 0, "motifs")["verdict"], "different");
    assert_eq!(aspect(&doc, 0, "ops")["b"], "close fstat lseek open read");
    assert_eq!(doc["verdict"], "different");
    // two processes at once on the template where A had one
    let sh = shuffled(&p);
    let mut two = Vec::new();
    for (i, path) in sh.iter().enumerate() {
        two.push((if i % 2 == 0 { 100 } else { 200 }, path.as_str(), false));
    }
    let e = digest(&d, "e", &two, &[]);
    let (_, doc) = json_of(&a, &e, &[]);
    assert_eq!((aspect(&doc, 0, "peak lanes")["a"].as_str(), aspect(&doc, 0, "peak lanes")["b"].as_str()), (Some("1"), Some("2")));
    assert_eq!(aspect(&doc, 0, "peak lanes")["verdict"], "different");
    assert_eq!((aspect(&doc, 0, "lanes")["b"].as_str(), aspect(&doc, 0, "lanes")["verdict"].as_str()), (Some("2"), Some("shown")));
    // under --judge, the structure decides the verdict even where every row is within
    let (code, out, _) = compare(&a, &b, &["--judge"]);
    assert_eq!(code, 1);
    assert!(out.lines().last().unwrap().starts_with("not accepted: the structure differs (1 difference(s)); "), "{out}");
}

#[test]
fn templates_pair_by_text_then_by_hand_then_by_shape() {
    let d = crate::common::tmpdir("compare-tool", "pairing");
    let (sn, t) = (names("s", 28), names("t", 28));
    let a = digest(&d, "a", &reads(100, &shuffled(&sn), false), &[]);
    let b = digest(&d, "b", &reads(100, &shuffled(&t), false), &[]);
    let (code, doc) = json_of(&a, &b, &[]);
    assert_eq!(code, 0, "{doc}");
    assert_eq!(doc["structure"]["pairs"][0]["by"], "shape");
    assert_eq!((doc["structure"]["pairs"][0]["a"].as_str(), doc["structure"]["pairs"][0]["b"].as_str()), (Some("s/{a:03}.npz"), Some("t/{a:03}.npz")));
    // two templates of one shape on each side: no shape decides, so each of A's is a difference
    // and each of B's is listed, until --pair names them
    let (u, v) = (names("u", 28), names("v", 28));
    let (ss, us, ts, vs) = (shuffled(&sn), shuffled(&u), shuffled(&t), shuffled(&v));
    let mut both = reads(100, &ss, false);
    both.extend(reads(100, &us, false));
    let mut theirs = reads(100, &ts, false);
    theirs.extend(reads(100, &vs, false));
    let a2 = digest(&d, "a2", &both, &[]);
    let b2 = digest(&d, "b2", &theirs, &[]);
    let (code, out, _) = compare(&a2, &b2, &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("template s/{a:03}.npz: A only, DIFFERENT") && out.contains("template v/{a:03}.npz: B only (listed)"), "{out}");
    let (code, doc) = json_of(&a2, &b2, &["--pair", "s/{a:03}.npz=t/{a:03}.npz"]);
    assert_eq!(code, 0, "{doc}");
    let by: Vec<(&str, &str)> = doc["structure"]["pairs"].as_array().unwrap().iter().map(|p| (p["a"].as_str().unwrap(), p["by"].as_str().unwrap())).collect();
    assert_eq!(by, [("s/{a:03}.npz", "--pair"), ("u/{a:03}.npz", "shape")]);
    // a template B has and A does not is listed, never a difference
    let (code, doc) = json_of(&a, &b2, &["--pair", "s/{a:03}.npz=t/{a:03}.npz"]);
    assert_eq!(code, 0);
    assert_eq!(doc["structure"]["unpaired_b"], json!(["v/{a:03}.npz"]));
    // --pair names templates the digests have, of two digests, written A=B
    let (code, _, err) = compare(&a2, &b2, &["--pair", "x=t/{a:03}.npz"]);
    assert_eq!((code, err.as_str()), (1, "aeiou-trace compare: --pair x=t/{a:03}.npz: A has no template `x`\n"));
    let (code, _, err) = compare(&a2, &b2, &["--pair", "s/{a:03}.npz"]);
    assert_eq!(code, 2);
    assert!(err.starts_with("aeiou-trace compare: --pair s/{a:03}.npz: give A's template and B's as A=B\n\nUsage: aeiou-trace compare "), "{err}");
    let m = d.join("m.json");
    std::fs::write(&m, serde_json::to_string(&serde_json::from_str::<Value>(&std::fs::read_to_string(&a).unwrap()).unwrap()["metrics"]).unwrap()).unwrap();
    let (code, _, err) = compare(&a, s(&m), &["--pair", "s/{a:03}.npz=t/{a:03}.npz"]);
    assert_eq!((code, err.as_str()), (1, "aeiou-trace compare: --pair pairs the templates of two digests, and B is a metrics document\n"));
}

#[test]
fn a_digest_against_a_metrics_document_is_judged_on_numbers_alone() {
    let d = crate::common::tmpdir("compare-tool", "numbers");
    let p = names("s", 28);
    let a = digest(&d, "a", &reads(100, &shuffled(&p), false), &[]);
    let m = d.join("m.json");
    std::fs::write(&m, serde_json::to_string(&serde_json::from_str::<Value>(&std::fs::read_to_string(&a).unwrap()).unwrap()["metrics"]).unwrap()).unwrap();
    let (code, out, err) = compare(&a, s(&m), &[]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("structure: not compared, B is a metrics document (numbers only)\n"), "{out}");
    assert!(out.contains(&format!("B: {}: strace of /d (order: completion; 1 instance(s), 1 thread(s))\n", s(&m))), "{out}");
    let (_, out, _) = compare(s(&m), &a, &[]);
    assert!(out.contains("structure: not compared, A is a metrics document (numbers only)\n"), "{out}");
    let (_, doc) = json_of(s(&m), s(&m), &[]);
    assert_eq!(doc["structure"], Value::Null);
}

#[test]
fn an_owners_answer_is_compared_as_an_assertion() {
    let d = crate::common::tmpdir("compare-tool", "assertion");
    let p = names("s", 28);
    // one pass: the selection is undecided on A until its owner answers it
    let answers = d.join("a.answers.json");
    std::fs::write(&answers, json!({"aeiou_answers": 1, "answers": [{"template": "s/{a:03}.npz", "question": "selection", "answer": "consume", "by": "owner", "at": "2026-11-01T00:00:00+00:00"}]}).to_string()).unwrap();
    // a corpus far larger than the trace: one pass decides nothing of the selection
    let plain = digest(&d, "plain", &reads(100, &shuffled(&p), false), &["--candidates", "2000"]);
    let a = digest(&d, "a", &reads(100, &shuffled(&p), false), &["--candidates", "2000", "--answers", s(&answers)]);
    // two passes: B's decides consume from the trace
    let sh = shuffled(&p);
    let mut twice = reads(100, &sh, false);
    twice.extend(reads(100, &sh, false));
    let b = digest(&d, "b", &twice, &[]);
    let (_, doc) = json_of(&plain, &b, &[]);
    let x = aspect(&doc, 0, "selection");
    assert_eq!((x["a"].as_str(), x["b"].as_str(), x["verdict"].as_str(), x["note"].as_str()), (Some("undecided"), Some("consume"), Some("not compared"), Some("A undecided")));
    let (_, doc) = json_of(&a, &b, &[]);
    let x = aspect(&doc, 0, "selection");
    assert_eq!((x["a"].as_str(), x["verdict"].as_str(), x["note"].as_str()), (Some("consume (asserted)"), Some("same"), Some("assertion")));
}

#[test]
fn the_runners_own_io_on_b_is_set_aside() {
    let d = crate::common::tmpdir("compare-tool", "own");
    let p = names("s", 28);
    let a = digest(&d, "a", &reads(100, &shuffled(&p), false), &[]);
    let sh = shuffled(&p);
    let mut run = vec![(100, ".aeiou-dataset.json", false)];
    run.extend(reads(100, &sh, false));
    let b = digest(&d, "b", &run, &[]);
    let (code, out, _) = compare(&a, &b, &[]);
    assert!(out.contains("set aside from B, the runner's own I/O: .aeiou-dataset.json\n"), "{out}");
    assert_eq!(code, 0, "{out}");
    // A's own is the application's: never set aside
    let (code, out, _) = compare(&b, &a, &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("template .aeiou-dataset.json: A only, DIFFERENT"), "{out}");
}

/// The digest of a writer: per path, an open (`O_TRUNC` when `trunc`), a 64 KiB write, `extra`
/// calls on it, and a close, then the same again, a millisecond apart.
fn writer_digest(dir: &Path, name: &str, trunc: bool, extra: &[&str]) -> String {
    let mut out = String::new();
    let mut t = 0u64;
    let mut line = |call: String| {
        t += 1000;
        out.push_str(&format!("100 1700000000.{t:06} {call} <0.000005>\n"));
    };
    let flags = if trunc { "O_WRONLY|O_CREAT|O_TRUNC" } else { "O_WRONLY|O_CREAT" };
    for _ in 0..2 {
        for i in 0..3 {
            let p = format!("/d/ck/s{i}.bin");
            line(format!("openat(AT_FDCWD</d>, \"{p}\", {flags}, 0644) = 3<{p}>"));
            line(format!("write(3<{p}>, \"x\"..., 65536) = 65536"));
            for call in extra {
                line(call.replace("FD", &format!("3<{p}>")));
            }
            line(format!("close(3<{p}>) = 0"));
        }
    }
    let trace = dir.join(format!("{name}.strace"));
    std::fs::write(&trace, out).unwrap();
    let doc = dir.join(format!("{name}.digest.json"));
    let o = tool(&["digest", s(&trace), "--root", "/d", "--no-questions", "-o", s(&doc)]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    s(&doc).to_string()
}

/// What a template's calls did to the space its files hold (the digests' `space`, §3.92) is
/// part of its structure: the kinds of it, never the counts, which grow with the scale; an
/// abstract that writes where the application truncated is another workload. A pair that did
/// none of it on either side has no such aspect.
#[test]
fn what_frees_or_moves_space_is_part_of_the_structure() {
    let d = crate::common::tmpdir("compare-tool", "space");
    let a = writer_digest(&d, "a", true, &[]);
    let (code, doc) = json_of(&a, &a, &[]);
    assert_eq!(code, 0);
    assert_eq!(aspect(&doc, 0, "space")["a"], "truncating opens, of a path written before");
    assert_eq!(aspect(&doc, 0, "space")["verdict"], "same");
    // the same opens without O_TRUNC: the same ops and motifs, another workload
    let b = writer_digest(&d, "b", false, &[]);
    let (code, out, _) = compare(&a, &b, &[]);
    assert_eq!(code, 1, "{out}");
    let (_, doc) = json_of(&a, &b, &[]);
    assert_eq!((aspect(&doc, 0, "ops")["verdict"].as_str(), aspect(&doc, 0, "motifs")["verdict"].as_str()), (Some("same"), Some("same")));
    assert_eq!((aspect(&doc, 0, "space")["b"].as_str(), aspect(&doc, 0, "space")["verdict"].as_str()), (Some("none"), Some("different")));
    // a hole punched in each is a kind of its own
    let c = writer_digest(&d, "c", true, &["fallocate(FD, FALLOC_FL_KEEP_SIZE|FALLOC_FL_PUNCH_HOLE, 0, 4096) = 0"]);
    let (_, doc) = json_of(&a, &c, &[]);
    assert_eq!(aspect(&doc, 0, "space")["b"], "truncating opens, of a path written before, fallocate KEEP_SIZE|PUNCH_HOLE");
    assert_eq!(aspect(&doc, 0, "space")["verdict"], "different");
    // readers on both sides: no space aspect
    let p = names("s", 28);
    let r = digest(&d, "r", &reads(100, &p, false), &[]);
    let (_, doc) = json_of(&r, &r, &[]);
    assert!(doc["structure"]["pairs"][0]["aspects"].as_array().unwrap().iter().all(|x| x["name"] != "space"), "{doc}");
}
