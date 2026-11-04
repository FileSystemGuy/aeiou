//! `aeiou-trace capture --requests shim` (`DESIGN_REVIEW.md` §3.96, phase 2 of brief item 54):
//! the library preloaded into the command records the HTTP heads of its object requests, over
//! plain HTTP through libc's socket calls (curl against MinIO, signed) and over TLS through a
//! shared libssl (curl against a local HTTPS server), into `requests.jsonl` with each response;
//! no secret of the request reaches the capture's files; the library's own calls in the trace
//! are only its start's; a full log counts what it dropped; a process the library did not reach
//! is named; the anchors are sent to each endpoint at the start and the end; the readers take
//! the capture directory, its trace for a directory's root and its requests for a bucket's.
//! MinIO is `TEST_MINIO_BIN`'s, never skipped (`tests/object.rs`); curl, openssl, and python3
//! are the host's, as the object tests' curl is.
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};

use serde_json::Value;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_aeiou-trace")
}

fn tmp(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("capture_requests", tag)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The checks a test on a shared machine waives (`tests/capture.rs`).
const WAIVE: [&str; 4] = ["--accept", "alone", "--accept", "clock"];

fn capture(args: &[&str]) -> Output {
    Command::new(bin()).arg("capture").args(args).env_remove("http_proxy").env_remove("https_proxy").output().unwrap()
}

fn record(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("capture.json")).unwrap()).unwrap()
}

fn requests(dir: &Path) -> (Value, Vec<Value>) {
    let text = std::fs::read_to_string(dir.join("requests.jsonl")).unwrap();
    let mut lines = text.lines().map(|l| serde_json::from_str::<Value>(l).unwrap());
    let head = lines.next().unwrap();
    (head, lines.collect())
}

/// A Python HTTP server of `dir` on a port of its own choosing, over TLS with `tls`'s
/// certificate and key (TLS 1.2 at most, so the anchor's 1.2 path is taken); killed when dropped.
struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn server(dir: &Path, tls: Option<(&Path, &Path)>) -> Server {
    let script = "import http.server, ssl, sys\n\
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler)\n\
if len(sys.argv) > 2:\n    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n    ctx.maximum_version = ssl.TLSVersion.TLSv1_2\n    ctx.load_cert_chain(sys.argv[1], sys.argv[2])\n    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)\n\
print(srv.server_address[1], flush=True)\n\
srv.serve_forever()\n";
    let mut c = Command::new("python3");
    c.args(["-c", script]).current_dir(dir).stdout(Stdio::piped()).stderr(Stdio::null());
    if let Some((cert, key)) = tls {
        c.args([s(cert), s(key)]);
    }
    let mut child = c.spawn().expect("python3");
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    Server { port: line.trim().parse().expect("the server's port"), child }
}

/// The trace's calls on `name`: the call's name of each line that mentions it.
fn calls_on(trace: &Path, name: &str) -> Vec<String> {
    std::fs::read_to_string(trace)
        .unwrap()
        .lines()
        .filter(|l| l.contains(name))
        .map(|l| {
            let mut t = l.split_whitespace().skip(2);
            let first = t.next().unwrap_or("");
            let call = if first == "<..." { t.next().unwrap_or("") } else { first };
            call.split('(').next().unwrap_or("").to_string()
        })
        .collect()
}

#[cfg(feature = "object")]
#[test]
fn object_requests_over_http_are_recorded_with_their_responses_and_never_a_secret() {
    let m = crate::object::minio("object_requests_over_http_are_recorded_with_their_responses_and_never_a_secret");
    let d = tmp("http");
    let data = d.join("data");
    std::fs::create_dir(&data).unwrap();
    std::fs::write(d.join("obj"), vec![7u8; 4096]).unwrap();
    assert!(crate::object::curl(&m, "/bench/obj", &["-T", s(&d.join("obj"))]).status.success());
    // the credentials in a file of curl's, so neither the command line nor the environment
    // (both in the trace and the record) holds them: only the requests' heads could
    let cfg = d.join("curl.cfg");
    std::fs::write(&cfg, format!("user = \"{}:{}\"\naws-sigv4 = \"aws:amz:us-east-1:s3\"\nsilent\nfail\n", crate::object::USER, m.secret)).unwrap();
    let out = d.join("cap");
    let script = format!("curl -K {cfg} -r 0-99 -o {part} {url}/bench/obj && curl -K {cfg} -I -o /dev/null {url}/bench/obj", cfg = s(&cfg), part = s(&data.join("part")), url = m.url);
    let endpoint = format!("{}/bench", m.url);
    let mut args = vec!["--root", s(&data), "--endpoint", &endpoint, "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    assert!(o.status.success(), "{}", err(&o));

    let rec = record(&out);
    let r = &rec["requests"];
    assert_eq!(r["source"], "shim", "{r}");
    assert!(r["library"]["path"].as_str().unwrap().ends_with("libaeiou_capture.so") && r["library"]["sha256"].as_str().unwrap().len() == 64, "{r}");
    assert_eq!((r["requests"].as_u64(), r["answered"].as_u64(), r["log"]["dropped"].as_u64(), r["log"]["torn"].as_u64()), (Some(2), Some(2), Some(0), Some(0)), "{r}");
    // sh and both curls loaded the library; every process the trace saw begin did
    assert!(r["processes"]["reached"].as_u64().unwrap() >= 3, "{r}");
    assert_eq!(r["processes"]["not_reached"], serde_json::json!([]), "{r}");
    assert!(rec["not_seen"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().starts_with("object-store requests of clients the library does not reach")), "{}", rec["not_seen"]);

    let (head, reqs) = requests(&out);
    assert_eq!((head["aeiou_requests"].as_u64(), head["requests"].as_u64()), (Some(1), Some(2)));
    let port = m.url.rsplit(':').next().unwrap();
    // by method, not position: the requests are in the order of their wall-clock starts, and a
    // host whose clock is stepped (WSL2 follows its host's) can put the second curl's first
    let get = reqs.iter().find(|r| r["method"] == "GET").unwrap_or_else(|| panic!("no GET: {reqs:?}"));
    let headr = reqs.iter().find(|r| r["method"] == "HEAD").unwrap_or_else(|| panic!("no HEAD: {reqs:?}"));
    assert_eq!((get["method"].as_str(), get["target"].as_str(), get["via"].as_str(), get["status"].as_u64()), (Some("GET"), Some("/bench/obj"), Some("socket"), Some(206)), "{get}");
    assert_eq!(get["request"]["range"], "bytes=0-99", "{get}");
    assert_eq!(get["response"]["content-range"], "bytes 0-99/4096", "{get}");
    assert_eq!(get["peer"], format!("127.0.0.1:{port}"), "{get}");
    assert!(get["end"].as_f64().unwrap() >= get["start"].as_f64().unwrap());
    assert_eq!(headr["status"].as_u64(), Some(200), "{headr}");
    assert!(headr["pid"] != get["pid"], "two curls, two processes");

    // the request's secrets are in none of the capture's own files
    for f in ["requests.jsonl", "requests.bin", "capture.json"] {
        let text = String::from_utf8_lossy(&std::fs::read(out.join(f)).unwrap()).into_owned();
        for secret in [m.secret.as_str(), "Signature=", "AWS4-HMAC-SHA256", "x-amz-content-sha256"] {
            assert!(!text.contains(secret), "{secret} in {f}");
        }
    }
    // the library's own calls in the trace: its start's, and the environment execve shows
    let calls = calls_on(&out.join("trace.txt"), "requests.bin");
    assert!(calls.iter().all(|c| ["openat", "fstat", "newfstatat", "mmap", "close", "execve"].contains(&c.as_str())), "{calls:?}");
    assert!(calls.iter().filter(|c| *c == "mmap").count() >= 3, "{calls:?}");

    // the anchors, at the start and the end, refused by the store and logged by it
    let anchors = rec["anchors"].as_array().unwrap();
    assert_eq!(anchors.len(), 2, "{anchors:?}");
    for (a, when) in anchors.iter().zip(["start", "end"]) {
        assert!(a["target"].as_str().unwrap().starts_with("/bench/aeiou-anchor-") && a["target"].as_str().unwrap().ends_with(when), "{a}");
        assert_eq!(a["status"], 403, "{a}");
        assert!(a["received"].as_f64().unwrap() >= a["sent"].as_f64().unwrap());
    }
    assert!(anchors[0]["received"].as_f64().unwrap() <= get["start"].as_f64().unwrap() && anchors[1]["sent"].as_f64().unwrap() >= headr["end"].as_f64().unwrap());

    // the readers take the directory: its requests for a bucket's root, its trace for a directory's
    let metrics = |root: &str| Command::new(bin()).args(["metrics", s(&out), "--root", root]).output().unwrap();
    let o = metrics("bench");
    assert!(o.status.success(), "{}", err(&o));
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["total"]["bytes_read"], 100, "{}", doc["total"]);
    assert_eq!(doc["notes"]["lanes"], "one per client thread (pid/tid)", "{}", doc["notes"]);
    let o = metrics(s(&data));
    assert!(o.status.success(), "{}", err(&o));
    let doc: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(doc["total"]["bytes_written"], 100, "{}", doc["total"]);
    let o = Command::new(bin()).args(["export", s(&out), "--root", "bench", "-o", s(&d.join("x.trace"))]).output().unwrap();
    assert!(o.status.success(), "{}", err(&o));
    let o = Command::new(bin()).args(["metrics", s(&out), "--root", "bench", "--root", "/x"]).output().unwrap();
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    assert!(err(&o).contains("a capture directory's roots are all directories (its trace.txt) or all one bucket (its requests.jsonl)"), "{}", err(&o));
}

#[cfg(feature = "object")]
#[test]
fn requests_over_tls_are_seen_through_a_shared_libssl_and_anchored_over_https() {
    let d = tmp("tls");
    let (cert, key) = (d.join("cert.pem"), d.join("key.pem"));
    let o = Command::new("openssl").args(["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=localhost", "-days", "1", "-keyout", s(&key), "-out", s(&cert)]).output().unwrap();
    assert!(o.status.success(), "{}", err(&o));
    let www = d.join("www");
    std::fs::create_dir(&www).unwrap();
    std::fs::write(www.join("f"), b"hello").unwrap();
    let srv = server(&www, Some((&cert, &key)));
    let url = format!("https://127.0.0.1:{}", srv.port);
    let out = d.join("cap");
    let script = format!("curl -sk -o /dev/null {url}/f");
    let mut args = vec!["--endpoint", &url, "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    assert!(o.status.success(), "{}", err(&o));
    let (_, reqs) = requests(&out);
    assert_eq!(reqs.len(), 1, "{reqs:?}");
    let q = &reqs[0];
    assert_eq!((q["via"].as_str(), q["method"].as_str(), q["target"].as_str(), q["status"].as_u64()), (Some("ssl"), Some("GET"), Some("/f"), Some(200)), "{q}");
    assert_eq!(q["response"]["content-length"], "5", "{q}");
    // the descriptor under the SSL connection, and the peer its connect named
    assert!(q["fd"].as_i64().unwrap() >= 0 && q["peer"] == format!("127.0.0.1:{}", srv.port), "{q}");
    let rec = record(&out);
    for a in rec["anchors"].as_array().unwrap() {
        assert_eq!(a["status"], 404, "{a}");
    }
}

#[test]
fn a_full_log_counts_what_it_dropped_and_a_process_the_library_missed_is_named() {
    let d = tmp("full");
    let www = d.join("www");
    std::fs::create_dir(&www).unwrap();
    let srv = server(&www, None);
    let url = format!("http://127.0.0.1:{}", srv.port);
    let out = d.join("cap");
    let script = format!("for i in $(seq 40); do curl -s -o /dev/null {url}/; done");
    let mut args = vec!["--endpoint", &url, "--requests-buffer", "8KiB", "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    assert!(o.status.success(), "{}", err(&o));
    let rec = record(&out);
    let r = &rec["requests"];
    assert_eq!(r["log"]["capacity"], 8192, "{r}");
    assert!(r["log"]["dropped"].as_u64().unwrap() > 0 && r["requests"].as_u64().unwrap() < 40, "{r}");
    // cut to what was recorded
    assert_eq!(std::fs::metadata(out.join("requests.bin")).unwrap().len(), r["log"]["used"].as_u64().unwrap());
    // the processes' own records were dropped with the rest: which the library missed is unknown
    assert!(r["processes"]["not_reached"].is_null() && r["processes"]["not_reached_unknown"].as_str().unwrap().contains("the log was full"), "{r}");
    assert!(err(&o).contains("WARNING: the request log was full: "), "{}", err(&o));
    // a plain server answers the anchor's HEAD of a name it does not have
    assert_eq!(rec["anchors"][0]["status"], 404, "{}", rec["anchors"]);

    let out = d.join("cap2");
    let script = format!("curl -s -o /dev/null {url}/; env -u LD_PRELOAD true");
    let mut args = vec!["--endpoint", &url, "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    assert!(o.status.success(), "{}", err(&o));
    let r = &record(&out)["requests"];
    let missed = r["processes"]["not_reached"].as_array().unwrap();
    assert_eq!(missed.len(), 1, "{missed:?}");
    assert!(missed[0]["exe"].as_str().unwrap().ends_with("/true"), "{missed:?}");
    assert!(err(&o).contains("WARNING: 1 process(es) began without the library"), "{}", err(&o));
    assert_eq!(r["requests"], 1, "{r}");
}

#[test]
fn requests_none_records_none_and_bad_values_are_usage_errors() {
    let d = tmp("none");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let out = d.join("cap");
    let mut args = vec!["--root", s(&root), "--endpoint", "s3://bucket/p", "--requests", "none", "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "true"]);
    let o = capture(&args);
    assert!(o.status.success(), "{}", err(&o));
    let rec = record(&out);
    assert_eq!(rec["requests"]["source"], "none", "{}", rec["requests"]);
    assert!(!out.join("requests.bin").exists() && !out.join("requests.jsonl").exists());
    assert!(rec["anchors"][0]["not_sent"].as_str().unwrap().contains("names no HTTP host"), "{}", rec["anchors"]);
    // no --endpoint: none unless asked
    let out2 = d.join("cap2");
    let mut args = vec!["--root", s(&root), "-o", s(&out2)];
    args.extend(WAIVE);
    args.extend(["--", "true"]);
    assert!(capture(&args).status.success());
    assert_eq!(record(&out2)["requests"]["source"], "none");
    assert_eq!(record(&out2)["anchors"], serde_json::json!([]));

    for (bad, want) in [(["--requests", "frob"], "invalid value 'frob' for '--requests <MODE>'"), (["--requests-buffer", "1TB"], "--requests-buffer 1TB: the unit is KiB, MiB, or GiB")] {
        let out3 = d.join("cap3");
        let mut args = vec!["--root", s(&root), "-o", s(&out3)];
        args.extend(bad);
        args.extend(["--", "true"]);
        let o = capture(&args);
        assert_eq!(o.status.code(), Some(2), "{}", err(&o));
        assert!(err(&o).contains(want), "{want}: {}", err(&o));
    }
}

#[test]
fn a_binary_without_its_library_fails_the_requests_check_unless_it_is_waived() {
    let d = tmp("nolib");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let lone: PathBuf = d.join("bin").join("aeiou-trace");
    std::fs::create_dir(d.join("bin")).unwrap();
    std::fs::copy(bin(), &lone).unwrap();
    let run = |out: &Path, extra: &[&str]| {
        Command::new(&lone).arg("capture").args(["--root", s(&root), "--endpoint", "s3://bucket", "-o", s(out)]).args(WAIVE).args(extra).args(["--", "true"]).output().unwrap()
    };
    let o = run(&d.join("cap"), &[]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("requests") && err(&o).contains("FAILED") && err(&o).contains("libaeiou_capture.so is not beside"), "{}", err(&o));
    assert!(!d.join("cap").exists());
    let o = run(&d.join("cap"), &["--accept", "requests"]);
    assert!(o.status.success(), "{}", err(&o));
    let r = &record(&d.join("cap"))["requests"];
    assert_eq!(r["source"], "none", "{r}");
    assert!(r["why"].as_str().unwrap().contains("waived"), "{r}");
}
