//! The `aeiou-trace` binary (`DESIGN_REVIEW.md` §3.89): its usage errors in the suite's frame
//! (`runner/REFERENCE.md` §15), what `metrics` and `export` print and write, standard input, the
//! failures that name the file, and `compare`'s command line. What the
//! calls become is `strace_corpus.rs`'s; these are the command line's.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::Value;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_aeiou-trace")
}

fn tool(args: &[&str]) -> Output {
    Command::new(bin()).args(args).env_remove("AEIOU_PYTHON").output().unwrap()
}

fn text(o: &Output) -> (String, String) {
    (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/strace").join(name)
}

fn golden(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture(&format!("{name}.json"))).unwrap()).unwrap()
}

fn tmp(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("trace-tool", tag)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn usage_errors_list_every_missing_argument_in_the_frame() {
    let o = tool(&[]);
    assert_eq!(o.status.code(), Some(2));
    let (_, err) = text(&o);
    assert!(err.starts_with("aeiou-trace: the following required arguments were not provided:\n  <COMMAND>  one of metrics, export, compare, digest"), "{err}");
    assert!(err.contains("\n\nUsage: aeiou-trace <COMMAND> [ARGS]...\n\nFor more information, try 'aeiou-trace --help'.\n"), "{err}");

    let (_, err) = text(&tool(&["export"]));
    assert!(err.starts_with("aeiou-trace export: the following required arguments were not provided:"), "{err}");
    for want in ["<TRACE>", "--root <DIR>", "-o <FILE>"] {
        assert!(err.contains(want), "{want}: {err}");
    }
    let (_, err) = text(&tool(&["metrics"]));
    assert!(err.contains("<TRACE>") && err.contains("Usage: aeiou-trace metrics [OPTIONS] <TRACE>"), "{err}");

    // the parser's own errors, reframed
    let o = tool(&["metrics", "x", "--frob"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.starts_with("aeiou-trace metrics: unexpected argument '--frob'"), "{}", text(&o).1);
    let o = tool(&["metrics", "x", "--chain-gap-us", "nan"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("Usage: aeiou-trace metrics"), "{}", text(&o).1);
    let f = fixture("export-0070bb6a.in");
    let o = tool(&["metrics", s(&f), "--root", "/d", "--metrics-block", "0"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.starts_with("aeiou-trace metrics: --metrics-block must be at least 1"), "{}", text(&o).1);
    // an strace needs a root; an exported file does not
    let o = tool(&["metrics", s(&f)]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("--root <DIR>") && text(&o).1.contains("needed for an strace"), "{}", text(&o).1);

    let (out, _) = text(&tool(&["--version"]));
    assert_eq!(out.trim(), format!("aeiou-trace {}", env!("CARGO_PKG_VERSION")));
    let (out, _) = text(&tool(&["export", "--help"]));
    assert!(out.contains("--exclude <GLOB>") && out.contains("Usage: aeiou-trace export [OPTIONS] --root <DIR> -o <FILE> <TRACE>"), "{out}");
}

#[test]
fn metrics_print_the_document_or_write_it_and_say_so() {
    let f = fixture("export-0070bb6a.in");
    let want = golden("export-0070bb6a");
    let o = tool(&["metrics", s(&f), "--root", "/d"]);
    assert!(o.status.success(), "{:?}", text(&o));
    let mut got: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(got["trace"], s(&f));
    got.as_object_mut().unwrap().remove("trace");
    assert_eq!(got, want["metrics"]);

    let d = tmp("metrics-o");
    let out = d.join("m.json");
    let o = tool(&["metrics", s(&f), "--root", "/d/", "-o", s(&out)]);
    assert!(o.status.success(), "{:?}", text(&o));
    let (said, _) = text(&o);
    let t = &want["metrics"]["total"];
    assert!(said.starts_with(&format!("{} ops under /d: read {} bytes, wrote {} bytes; 1 thread(s); {}", t["ops"], t["bytes_read"], t["bytes_written"], out.display())), "{said}");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(doc["total"], want["metrics"]["total"]);

    // standard input, an strace
    let mut child = Command::new(bin()).args(["metrics", "-", "--root", "/d"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(&std::fs::read(&f).unwrap()).unwrap();
    let o = child.wait_with_output().unwrap();
    assert!(o.status.success());
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["total"], want["metrics"]["total"]);
    assert_eq!(doc["trace"], "-");
}

#[test]
fn export_writes_the_file_and_its_sha256_and_reads_back_as_the_strace() {
    let f = fixture("export-0070bb6a.in");
    let want = golden("export-0070bb6a");
    let d = tmp("export");
    let out = d.join("t.jsonl");
    let o = tool(&["export", s(&f), "--root", "/d", "-o", s(&out)]);
    assert!(o.status.success(), "{:?}", text(&o));
    let (said, _) = text(&o);
    let h = &want["export"]["header"];
    assert!(said.starts_with(&format!("{} line(s) on {} lane(s), {} open(s), 3 path(s) created under /d; {}\nsha256 ", h["lines"], h["lanes"], h["opens"], out.display())), "{said}");
    let bytes = std::fs::read(&out).unwrap();
    use sha2::{Digest, Sha256};
    let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    assert!(said.contains(&format!("sha256 {sha}")), "{said}");
    let lines: Vec<Value> = bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()).map(|l| serde_json::from_slice(l).unwrap()).collect();
    assert_eq!(lines[0], *h);
    assert_eq!(lines[1..], want["export"]["events"].as_array().unwrap()[..]);
    assert_eq!(std::fs::read_dir(&*d).unwrap().count(), 1, "nothing left beside the file");

    // the file's metrics are the strace's, read as a trace file
    let o = tool(&["metrics", s(&out)]);
    assert!(o.status.success(), "{:?}", text(&o));
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["source"], "trace");
    assert_eq!(doc["total"]["request_size"], want["metrics"]["total"]["request_size"]);
}

#[test]
fn nothing_counted_is_a_failure_naming_why_and_writes_nothing() {
    let f = fixture("export-0070bb6a.in");
    let d = tmp("nothing");
    let out = d.join("t.jsonl");
    let o = tool(&["export", s(&f), "--root", "/elsewhere", "-o", s(&out)]);
    assert_eq!(o.status.code(), Some(1));
    let (_, err) = text(&o);
    assert!(err.starts_with(&format!("aeiou-trace export: {}: no call under --root /elsewhere (", s(&f))), "{err}");
    assert!(err.contains("check --root against the paths the trace shows") && err.contains(&format!("{} not written", out.display())), "{err}");
    assert!(!out.exists());
    assert_eq!(std::fs::read_dir(&*d).unwrap().count(), 0);

    let o = tool(&["metrics", s(&f), "--root", "/elsewhere"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("no call under --root /elsewhere"), "{}", text(&o).1);

    // not an strace at all
    let junk = d.join("junk.txt");
    std::fs::write(&junk, "hello\nworld\n").unwrap();
    let o = tool(&["metrics", s(&junk), "--root", "/d"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("(2 line(s) read, 0 parsed as strace calls): not an strace"), "{}", text(&o).1);

    // a trace file with no line, and one that is not JSON
    let empty = d.join("e.jsonl");
    std::fs::write(&empty, "{\"aeiou_trace\":1,\"root\":\"/d\"}\n").unwrap();
    let o = tool(&["metrics", s(&empty)]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.ends_with("a trace file with no call in it\n"), "{}", text(&o).1);
    std::fs::write(&empty, "{\"aeiou_trace\":1,\"root\":\"/d\"}\nnot json\n").unwrap();
    let o = tool(&["metrics", s(&empty)]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains(": line 2: "), "{}", text(&o).1);

    // a file that is not there, and an output that cannot be written
    let o = tool(&["metrics", "/no/such/trace", "--root", "/d"]);
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(text(&o).1, "aeiou-trace metrics: /no/such/trace: No such file or directory (os error 2)\n");
    let o = tool(&["metrics", s(&f), "--root", "/d", "-o", "/no/such/dir/m.json"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.starts_with("aeiou-trace metrics: /no/such/dir/m.json: No such file or directory"), "{}", text(&o).1);
    let o = tool(&["export", s(&f), "--root", "/d", "-o", "/no/such/dir/t.jsonl"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("/no/such/dir/t.jsonl: No such file or directory"), "{}", text(&o).1);
}

#[test]
fn object_logs_take_one_bucket_root() {
    let f = fixture("objectlog-2e5ae1d9.in");
    let want = golden("objectlog-2e5ae1d9");
    let o = tool(&["metrics", s(&f)]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("--root <BUCKET[/PREFIX]>") && text(&o).1.contains("needed for an object log (minio-audit)"), "{}", text(&o).1);
    let o = tool(&["metrics", s(&f), "--root", "a", "--root", "b"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("an object log takes one --root BUCKET[/PREFIX], not 2"), "{}", text(&o).1);
    let o = tool(&["metrics", s(&f), "--root", "a-bucket/files"]);
    assert!(o.status.success(), "{:?}", text(&o));
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["total"], want["metrics"]["total"]);
    let o = tool(&["metrics", s(&f), "--root", "no-such-bucket"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("no request under --root no-such-bucket ("), "{}", text(&o).1);

    let d = tmp("objectlog");
    let out = d.join("t.jsonl");
    let o = tool(&["export", s(&f), "--root", "/a-bucket/files/", "-o", s(&out)]);
    assert!(o.status.success(), "{:?}", text(&o));
    let first: Value = serde_json::from_str(std::fs::read_to_string(&out).unwrap().lines().next().unwrap()).unwrap();
    assert_eq!(first, want["export"]["header"]);
    let o = tool(&["export", s(&f), "--root", "no-such-bucket", "-o", s(&out)]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("no request under --root no-such-bucket"), "{}", text(&o).1);
}

#[test]
fn compare_needs_no_python_and_lists_what_it_lacks() {
    // ported (DESIGN_REVIEW.md §3.89, stage 4): no interpreter is asked for
    let o = Command::new(bin()).args(["compare"]).env("AEIOU_PYTHON", "/no/such/python").output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    let err = text(&o).1;
    assert!(err.starts_with("aeiou-trace compare: the following required arguments were not provided:\n  <A>  the application's document: a digest, or a metrics document\n  <B>  the document A is compared with, the abstract's\n\nUsage: aeiou-trace compare [OPTIONS] <A> <B>"), "{err}");
    let o = tool(&["compare", "a.json", "b.json", "--max-distance", "x"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.starts_with("aeiou-trace compare: invalid value 'x' for '--max-distance <X>': not a finite number at least 0\n\nUsage: aeiou-trace compare "), "{}", text(&o).1);
    let o = tool(&["compare", "/no/such/a.json", "b.json"]);
    assert_eq!(text(&o).1, "aeiou-trace compare: /no/such/a.json: No such file or directory (os error 2)\n");
}

/// An strace of `np.load`-shaped opens, one line per call, 1 ms apart.
fn npz_strace(paths: &[&str]) -> String {
    let (mut out, mut t) = (String::new(), 1700000000.0f64);
    for p in paths {
        for line in [
            format!("100 {t:.6} openat(AT_FDCWD</d>, \"/d/{p}\", O_RDONLY|O_CLOEXEC) = 3</d/{p}> <0.000010>"),
            format!("100 {:.6} read(3</d/{p}>, \"x\"..., 4096) = 4096 <0.000100>", t + 0.001),
            format!("100 {:.6} close(3</d/{p}>) = 0 <0.000005>", t + 0.002),
        ] {
            out.push_str(&line);
            out.push('\n');
        }
        t += 0.01;
    }
    out
}

fn sha256_of(p: &Path) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(std::fs::read(p).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn digest_prints_the_document_the_report_or_writes_the_file() {
    let d = tmp("digest");
    let trace = d.join("t.strace");
    let paths: Vec<String> = (0..12).map(|i| format!("train/s_{i:03}.npz")).collect();
    let mut order: Vec<&str> = paths.iter().map(String::as_str).collect();
    let again = order.clone();
    order.extend(again);
    std::fs::write(&trace, npz_strace(&order)).unwrap();
    let o = tool(&["digest", s(&trace), "--root", "/d", "--no-questions"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["aeiou_digest"], 2);
    assert_eq!(doc["trace"], s(&trace));
    assert_eq!(doc["trace_sha256"], sha256_of(&trace));
    assert_eq!(doc["sites"][0]["selection"]["verdict"], "consume");
    assert_eq!(doc["metrics"]["aeiou_metrics"], 2);
    assert_eq!(doc["metrics"]["total"]["ops"], 72);
    assert_eq!(doc["sites"][0]["facts"][0], serde_json::json!({"name": "candidates", "ask": "How many files of this kind does the application have at the scale it runs?", "least": 12, "why": "the trace touched 12 distinct paths of this template"}));
    assert_eq!(doc["sites"][0]["facts"][1]["least"], 2);

    let (out, _) = text(&tool(&["digest", s(&trace), "--root", "/d", "--no-questions", "--text"]));
    assert!(out.starts_with(&format!("digest of {}: 72 ops on 1 lane(s) under /d; 1 path template(s), 1 site(s), 0 open question(s)\n", s(&trace))), "{out}");
    assert!(out.contains("    selection: consume; choose refuted: 2 full passes of equal length 12"), "{out}");

    let file = d.join("t.digest.json");
    let o = tool(&["digest", s(&trace), "--root", "/d", "--no-questions", "-o", s(&file), "--epochs", "3", "--candidates", "40", "--paths"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    let (out, _) = text(&o);
    assert!(out.starts_with("digest of "), "{out}");
    assert!(out.contains(&format!("\nwritten: {}\n", s(&file))), "{out}");
    assert!(out.contains("note: candidates: --candidates 40") || !out.contains("note: "), "{out}");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(doc["projection"], serde_json::json!({"epochs": 3, "candidates": 40}));
    assert_eq!(doc["templates"][0]["raw_paths"].as_array().unwrap().len(), 12);

    // an exported trace file carries its root, and digests as its strace
    let exported = d.join("t.jsonl");
    assert_eq!(tool(&["export", s(&trace), "--root", "/d", "-o", s(&exported)]).status.code(), Some(0));
    let o = tool(&["digest", s(&exported), "--no-questions"]);
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["source"], "trace");
    assert_eq!(doc["sites"][0]["selection"]["verdict"], "consume");
}

#[test]
fn digest_usage_errors_and_failures() {
    let d = tmp("digest-usage");
    let (_, err) = text(&tool(&["digest"]));
    assert!(err.starts_with("aeiou-trace digest: the following required arguments were not provided:\n  <TRACE>"), "{err}");
    assert!(err.contains("Usage: aeiou-trace digest [OPTIONS] --root <DIR> <TRACE>"), "{err}");
    let trace = d.join("t.strace");
    std::fs::write(&trace, npz_strace(&["a/1.npz", "a/2.npz"])).unwrap();
    let o = tool(&["digest", s(&trace), "--no-questions"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("--root <DIR>") && text(&o).1.contains("needed for an strace; an exported trace file carries its root"), "{}", text(&o).1);
    for (args, want) in [
        (vec!["--epochs", "0"], "--epochs must be at least 1"),
        (vec!["--candidates", "0"], "--candidates must be at least 1"),
    ] {
        let mut a = vec!["digest", s(&trace), "--root", "/d"];
        a.extend(args);
        let o = tool(&a);
        assert_eq!(o.status.code(), Some(2));
        assert!(text(&o).1.starts_with(&format!("aeiou-trace digest: {want}")), "{}", text(&o).1);
    }
    let o = tool(&["digest", "-", "--root", "/d"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("the digest reads the trace as a file, never stdin"), "{}", text(&o).1);
    let o = tool(&["digest", "/no/such/trace", "--root", "/d"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.starts_with("aeiou-trace digest: /no/such/trace: No such file or directory"), "{}", text(&o).1);
    let o = tool(&["digest", s(&trace), "--root", "/nowhere", "--no-questions"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.starts_with(&format!("aeiou-trace digest: {}: no call under --root /nowhere", s(&trace))), "{}", text(&o).1);
    // answers named and absent, with no terminal to ask on
    let missing = d.join("none.answers.json");
    let o = Command::new(bin()).args(["digest", s(&trace), "--root", "/d", "--answers", s(&missing)]).stdin(Stdio::null()).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("no such answers file, and no terminal to ask on"), "{}", text(&o).1);
    // an application the records do not have, and records that are not there
    let o = tool(&["digest", s(&trace), "--root", "/d", "--application", "nope"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.starts_with("aeiou-trace digest: --application nope: no such application record; one of diskann, faiss, lmcache, milvus, pytorch-dcp"), "{}", text(&o).1);
    let o = tool(&["digest", s(&trace), "--root", "/d", "--applications", s(&d.join("absent"))]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("absent: no such directory of application records"), "{}", text(&o).1);
}

/// A trace of the golden corpus (`tests/digest/`), copied to `to`.
fn corpus_trace(name: &str, to: &Path) {
    std::fs::copy(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/digest").join(name), to).unwrap();
}

/// `aeiou-trace` with its standard input a terminal (a pseudo-terminal, echo off) on which the
/// lines `typed` are typed, then end of input.
fn on_terminal(args: &[&str], typed: &[&str]) -> Output {
    use std::os::fd::FromRawFd;
    unsafe {
        let m = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
        assert!(m >= 0 && libc::grantpt(m) == 0 && libc::unlockpt(m) == 0);
        let slave = libc::open(libc::ptsname(m), libc::O_RDWR | libc::O_NOCTTY);
        assert!(slave >= 0);
        let mut t: libc::termios = std::mem::zeroed();
        assert_eq!(libc::tcgetattr(slave, &mut t), 0);
        t.c_lflag &= !libc::ECHO;
        assert_eq!(libc::tcsetattr(slave, libc::TCSANOW, &t), 0);
        let mut input: String = typed.iter().map(|l| format!("{l}\n")).collect();
        input.push('\x04'); // VEOF at the start of a line: end of input
        assert_eq!(libc::write(m, input.as_ptr().cast(), input.len()), input.len() as isize);
        let o = Command::new(bin()).args(args).env_remove("AEIOU_PYTHON").stdin(Stdio::from(std::os::fd::OwnedFd::from_raw_fd(slave))).output().unwrap();
        libc::close(m);
        o
    }
}

fn read_json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn digest_asks_on_a_terminal_and_writes_the_answers_beside_the_document() {
    let d = tmp("digest-ask");
    let tr = d.join("t.txt");
    corpus_trace("trace-1b35ac0c.in", &tr);
    let out = d.join("t.digest.json");
    let o = on_terminal(&["digest", s(&tr), "--root", "/d", "-o", s(&out), "--answered-by", "tester"], &["8", "7", "4", "1", "shuffle=True", "3", "", "2", "because"]);
    let (stdout, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{err}");
    let doc = read_json(&out);
    assert_eq!(doc["sites"][0]["selection"]["verdict"], "consume"); // 7 candidates refute a draw after 7 touches without a repeat
    let asked: Vec<&str> = doc["asserted"].as_array().unwrap().iter().map(|a| a.get("fact").unwrap_or(&a["question"]).as_str().unwrap()).collect();
    assert_eq!(asked, ["candidates", "passes", "size", "order"]);
    assert!(doc["asserted"].as_array().unwrap().iter().all(|a| a["by"] == "tester" && a["at"].as_str().unwrap().ends_with("+00:00")));
    assert_eq!(doc["trace_sha256"], sha256_of(&tr));
    let af = d.join("t.answers.json");
    let header = read_json(&af);
    assert_eq!((&header["trace"], &header["trace_sha256"], &header["by"]), (&Value::from(s(&tr)), &doc["trace_sha256"], &Value::from("tester")));
    assert_eq!(header["answers"].as_array().unwrap().len(), 4);
    assert!(header["application"].is_null());
    // the questions on stderr, the report on stdout, and what was kept said after it
    assert!(err.contains("How many files of this kind") && err.contains("number (Enter for don't know): "), "{err}");
    assert!(stdout.starts_with("digest of ") && stdout.contains(&format!("written: {}", s(&out))), "{stdout}");
    assert!(err.ends_with(&format!("answers: {}\nreplay without questions: aeiou-trace digest {} --root /d --answers {} -o {}\n", s(&af), s(&tr), s(&af), s(&out))), "{err}");
    // the file replays without a terminal, the same assertions
    let again = d.join("again.json");
    let o = tool(&["digest", s(&tr), "--root", "/d", "-o", s(&again), "--answers", s(&af)]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    let doc2 = read_json(&again);
    assert_eq!((&doc2["asserted"], &doc2["sites"]), (&doc["asserted"], &doc["sites"]));
}

#[test]
fn digest_asks_nothing_with_no_questions_and_keeps_answers_where_it_is_told() {
    let d = tmp("digest-ask-not");
    let tr = d.join("t.txt");
    corpus_trace("trace-1b35ac0c.in", &tr);
    let o = on_terminal(&["digest", s(&tr), "--root", "/d", "--no-questions"], &["100"]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(text(&o).1, "");
    assert_eq!(serde_json::from_slice::<Value>(&o.stdout).unwrap()["asserted"], serde_json::json!([]));
    // a terminal, no -o and no --answers: the answers are in the document only, and it says so
    let o = on_terminal(&["digest", s(&tr), "--root", "/d", "--text"], &["8", "", "", "1", "", "3", "", "3", ""]);
    let (out, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{err}");
    assert!(out.contains("asserted") && err.ends_with("answers: in the document only (give -o or --answers FILE to keep them as a file)\n"), "{err}");
    assert!(!std::fs::read_dir(&*d).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".answers.json")));
    // a terminal and an --answers path that does not exist yet: asked, then written there
    let new = d.join("new.json");
    let o = on_terminal(&["digest", s(&tr), "--root", "/d", "--answers", s(&new)], &["8", "", "", "1", "", "3", "", "3", ""]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    assert_eq!(read_json(&new)["answers"][0]["answer"], "consume");
    // nothing answered: no file, and the document as without questions
    let none = d.join("none.json");
    let o = on_terminal(&["digest", s(&tr), "--root", "/d", "--answers", s(&none)], &[]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    assert!(!none.exists());
    assert_eq!(serde_json::from_slice::<Value>(&o.stdout).unwrap()["asserted"], serde_json::json!([]));
}

#[test]
fn digest_states_the_application_and_replays_it() {
    let d = tmp("digest-app");
    let tr = d.join("t.txt");
    corpus_trace("trace-a1c52e89.in", &tr);
    let o = tool(&["digest", s(&tr), "--root", "/d", "--application", "pytorch-dcp@2.14.1"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    let app = &doc["application"];
    assert_eq!((&app["record"], &app["exact"]), (&Value::from("2.14"), &Value::from(true)));
    assert_eq!(app["options"], serde_json::json!({"single_file_per_rank": "one file per rank", "sync_files": "fsync on"}));
    assert!(app.get("app").is_none());
    let asserted: Vec<&str> = doc["asserted"].as_array().unwrap().iter().map(|a| a["question"].as_str().unwrap()).collect();
    assert_eq!(asserted, ["size"]); // the record's label is about a bulk write, and this template is read
    assert!(doc["asserted"][0]["by"].as_str().unwrap().contains("record"));
    let o = tool(&["digest", s(&tr), "--root", "/d", "--application", "pytorch-dcp@2.9"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(serde_json::from_slice::<Value>(&o.stdout).unwrap()["application"]["record"].is_null());
    // stated, with an --answers file named that is not there and no terminal to ask on
    let o = Command::new(bin()).args(["digest", s(&tr), "--root", "/d", "--application", "pytorch-dcp", "--answers", s(&d.join("x.json"))]).stdin(Stdio::null()).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("x.json: no such answers file, and no terminal to ask on"), "{}", text(&o).1);
    // on a terminal with the application stated: the options are asked, the answers file carries the application, the replay restores it
    let out = d.join("t.digest.json");
    let o = on_terminal(&["digest", s(&tr), "--root", "/d", "-o", s(&out), "--application", "pytorch-dcp@2.14.1"], &["2", "1", "", "", "3", "", "3", ""]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    let doc = read_json(&out);
    assert_eq!(doc["application"]["options"], serde_json::json!({"single_file_per_rank": "one file per item", "sync_files": "fsync on"}));
    let af = d.join("t.answers.json");
    assert_eq!(read_json(&af)["application"], doc["application"]);
    let again = d.join("again.json");
    assert_eq!(tool(&["digest", s(&tr), "--root", "/d", "-o", s(&again), "--answers", s(&af)]).status.code(), Some(0));
    let again = read_json(&again);
    assert_eq!((&again["application"], &again["asserted"]), (&doc["application"], &doc["asserted"]));
    // the records of another tree
    let o = tool(&["digest", s(&tr), "--root", "/d", "--applications", s(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../applications")), "--application", "pytorch-dcp@2.14"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    assert_eq!(serde_json::from_slice::<Value>(&o.stdout).unwrap()["application"]["record"], "2.14");
}

#[test]
fn digest_replays_an_answers_file_and_says_when_it_is_another_traces() {
    let d = tmp("digest-answers");
    let trace = d.join("t.strace");
    let paths: Vec<String> = (0..30).map(|i| format!("train/s_{i:03}.npz")).collect();
    std::fs::write(&trace, npz_strace(&paths.iter().map(String::as_str).collect::<Vec<_>>())).unwrap();
    let answers = d.join("a.answers.json");
    let header = |sha: &str| serde_json::json!({"aeiou_answers": 1, "trace": "t", "trace_sha256": sha, "by": "owner@host", "at": "2026-10-31T00:00:00+00:00", "application": null,
        "answers": [{"template": "train/s_{a:03}.npz", "fact": "candidates", "value": 31}, {"template": "train/s_{a:03}.npz", "question": "order", "answer": "permuted"}]});
    std::fs::write(&answers, header(&sha256_of(&trace)).to_string()).unwrap();
    let o = tool(&["digest", s(&trace), "--root", "/d", "--answers", s(&answers)]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["asserted"][0]["value"], 31);
    assert_eq!(doc["asserted"][0]["by"], "owner@host");
    assert_eq!(doc["sites"][0]["selection"]["verdict"], "consume");
    assert!(doc["notes"].get("answers_trace").is_none());
    std::fs::write(&answers, header("0").to_string()).unwrap();
    let doc: Value = serde_json::from_slice(&tool(&["digest", s(&trace), "--root", "/d", "--answers", s(&answers)]).stdout).unwrap();
    assert!(doc["notes"]["answers_trace"].as_str().unwrap().contains("was answered on another trace (t)"), "{}", doc["notes"]);
    for (body, want) in [("{", "a.answers.json: "), ("[]", "not an aeiou_answers: 1 file"), (r#"{"aeiou_answers": 1, "answers": [1]}"#, "answers must be a list of objects")] {
        std::fs::write(&answers, body).unwrap();
        let o = tool(&["digest", s(&trace), "--root", "/d", "--answers", s(&answers)]);
        assert_eq!(o.status.code(), Some(1));
        assert!(text(&o).1.contains(want), "{want}: {}", text(&o).1);
    }
}

/// The peak resident set of `aeiou-trace args`, in KiB, measured from a fresh process (this test
/// binary run again for `peak_of_a_child` alone): a child's `ru_maxrss` counts the peak of the
/// process it was forked from, and this one, running every test of the crate, reaches gigabytes
/// under coverage, which the child would report as its own.
fn peak_kib(args: &[&str]) -> i64 {
    let o = Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "trace_tool::peak_of_a_child", "--nocapture", "--test-threads=1"])
        .env("AEIOU_TEST_PEAK_ARGS", args.join("\x1f"))
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let n = out.split("peak_kib=").nth(1).and_then(|r| r.split_whitespace().next());
    n.unwrap_or_else(|| panic!("no peak: {out}{}", String::from_utf8_lossy(&o.stderr))).parse().unwrap()
}

#[test]
#[ignore = "a helper: run by peak_kib in a fresh process, never alone"]
fn peak_of_a_child() {
    let args = std::env::var("AEIOU_TEST_PEAK_ARGS").unwrap();
    let child = Command::new(bin()).args(args.split('\x1f')).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let mut status = 0;
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let pid = unsafe { libc::wait4(child.id() as libc::pid_t, &mut status, 0, &mut ru) };
    assert_eq!(pid, child.id() as libc::pid_t);
    assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    println!("peak_kib={}", ru.ru_maxrss);
}

#[test]
fn the_digests_memory_follows_the_opens_never_the_ops() {
    // the same 200 opens of 200 files, each reading the same 16 blocks once or 300 times over:
    // 3,200 reads against 960,000, and the peak the same within a few MiB
    let d = tmp("digest-memory");
    let write = |name: &str, rounds: usize| -> PathBuf {
        let p = d.join(name);
        let mut f = std::io::BufWriter::new(std::fs::File::create(&p).unwrap());
        writeln!(f, r#"{{"aeiou_trace":1,"root":"/d","lanes":1,"lines":0,"opens":200,"creates":[],"notes":{{}}}}"#).unwrap();
        let mut t = 0;
        for i in 0..200 {
            writeln!(f, r#"{{"lane":0,"t":{t},"dur":1,"op":"open","fd":{i},"path":"f/{i:04}.bin","flags":["RDONLY"]}}"#).unwrap();
            for _ in 0..rounds {
                for b in 0..16 {
                    t += 1;
                    writeln!(f, r#"{{"lane":0,"t":{t},"dur":1,"op":"read","fd":{i},"offset":{},"len":4096,"ret":4096}}"#, b * 4096).unwrap();
                }
            }
            t += 1;
            writeln!(f, r#"{{"lane":0,"t":{t},"dur":1,"op":"close","fd":{i},"ret":0}}"#).unwrap();
        }
        p
    };
    let (small, large) = (write("small.jsonl", 1), write("large.jsonl", 300));
    let a = peak_kib(&["digest", s(&small), "--no-questions"]);
    let b = peak_kib(&["digest", s(&large), "--no-questions"]);
    assert!(b < a + 8 * 1024, "{a} KiB for 3,200 reads, {b} KiB for 960,000");
}
