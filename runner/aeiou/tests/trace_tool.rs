//! The `aeiou-trace` binary (`DESIGN_REVIEW.md` §3.89): its usage errors in the suite's frame
//! (`runner/REFERENCE.md` §15), what `metrics` and `export` print and write, standard input, the
//! failures that name the file, and the subcommands the Python package still runs. What the
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
fn the_unported_subcommands_say_what_they_need() {
    let o = Command::new(bin()).args(["compare", "a.json", "b.json"]).env("AEIOU_PYTHON", "/no/such/python").output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.starts_with("aeiou-trace compare: compare is the Python package's until it is ported"), "{err}");
    assert!(err.contains("AEIOU_PYTHON"), "{err}");
    // with an interpreter that has the package, its own messages and exit status come through
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../builder/.venv/bin/python");
    if venv.exists() {
        let o = Command::new(bin()).args(["compare", "--max-distance", "x"]).env("AEIOU_PYTHON", &venv).output().unwrap();
        assert_eq!(o.status.code(), Some(2), "{}", String::from_utf8_lossy(&o.stderr));
        assert!(String::from_utf8_lossy(&o.stderr).starts_with("aeiou-trace compare: "), "{}", String::from_utf8_lossy(&o.stderr));
    }
}
