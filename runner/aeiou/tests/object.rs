//! The object engine (`object.rs`, the cargo feature `object`) through the binary: `aeiou
//! datagen` writes a dataset declared `protocol: s3` into a MinIO bucket, the bytes of
//! each object are the bytes the POSIX writer puts in the file of the same name (a multipart
//! upload included), and `aeiou run` reads it back to the dry run's fingerprint, every read
//! returning the computed count; a run writes a namespace into a bucket (uploads, multipart
//! ones included, `rename` as a copy and a `DELETE`, the namespace manifest as an object), its
//! objects the bytes a POSIX run writes, and a second abstract reads it back as its input. The
//! refusals need no server.
//!
//! The tests that need a server start their own MinIO from the binary `TEST_MINIO_BIN` names
//! (CI downloads the pinned release; `runner/REFERENCE.md` §16), and fail, naming it, when it
//! is unset: a Rust test has no skipped state, and one that returned early would pass with
//! nothing tested (its message hidden, as a passing test's output is).

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const USER: &str = "aeiou";
const SECRET: &str = "aeiou-secret";

fn tmpdir(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("object", tag)
}

/// A committed example with its datasets and namespaces `names` declared `protocol: s3`, as a
/// file in `dir`; when no name is left `posix`, its declared POSIX API and cache mode go too
/// (V19: an API only for a protocol with names).
fn object_variant(example: &str, names: &[&str], dir: &Path) -> PathBuf {
    let mut d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(crate::common::workload_ast(example)).unwrap()).unwrap();
    for n in names {
        if let Some(ns) = d.get_mut("namespaces").and_then(|v| v.get_mut(*n)) {
            ns["protocol"] = "s3".into();
            continue;
        }
        d["datasets"][*n]["protocol"] = "s3".into();
    }
    let all = d.get("datasets").and_then(|v| v.as_object()).map_or(0, |m| m.len()) + d.get("namespaces").and_then(|v| v.as_object()).map_or(0, |m| m.len());
    if names.len() == all {
        let top = d.as_object_mut().unwrap();
        top.remove("posix");
        top.remove("cache");
    }
    let p = dir.join(format!("{example}.object.ast.json"));
    std::fs::write(&p, serde_json::to_string(&d).unwrap()).unwrap();
    p
}

/// A MinIO server on a free port with a directory of its own, killed when dropped.
struct Minio {
    child: Child,
    url: String,
    _dir: crate::common::TempDir,
}

impl Drop for Minio {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn minio(test: &str) -> Minio {
    let Some(bin) = std::env::var_os("TEST_MINIO_BIN") else {
        panic!("{test}: TEST_MINIO_BIN is not set: it names a MinIO server binary, the pinned release .github/workflows/ci.yml downloads (runner/REFERENCE.md §16)");
    };
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let dir = tmpdir("minio");
    let child = Command::new(bin)
        .args(["server", dir.to_str().unwrap(), "--address", &format!("127.0.0.1:{port}"), "--console-address", "127.0.0.1:0", "--quiet"])
        .env("MINIO_ROOT_USER", USER)
        .env("MINIO_ROOT_PASSWORD", SECRET)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("starting MinIO");
    let m = Minio { child, url: format!("http://127.0.0.1:{port}"), _dir: dir };
    let t0 = Instant::now();
    while !curl(&m, "/bench", &["-X", "PUT"]).status.success() {
        assert!(t0.elapsed() < Duration::from_secs(30), "MinIO did not come up at {}", m.url);
        std::thread::sleep(Duration::from_millis(200));
    }
    m
}

/// A request for `path` on the server, signed with the test's credentials; fails on an HTTP error.
fn curl(m: &Minio, path: &str, args: &[&str]) -> Output {
    Command::new("curl").args(["-sf", "--aws-sigv4", "aws:amz:us-east-1:s3", "--user", &format!("{USER}:{SECRET}")]).args(args).arg(format!("{}{path}", m.url)).env_remove("http_proxy").output().unwrap()
}

/// `aeiou` with the store of `m` in its environment (none when `m` is `None`).
fn aeiou(m: Option<&Minio>, args: &[&str]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.args(args).env_remove("AEIOU_CONFIG");
    for k in ["AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_ALLOW_HTTP"] {
        c.env_remove(k);
    }
    if let Some(m) = m {
        c.env("AWS_ENDPOINT_URL", &m.url).env("AWS_ACCESS_KEY_ID", USER).env("AWS_SECRET_ACCESS_KEY", SECRET).env("AWS_REGION", "us-east-1").env("AWS_ALLOW_HTTP", "true");
    }
    c.output().unwrap()
}

fn ok(o: &Output) -> String {
    let out = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(o.status.success(), "failed:\n{out}\n{}", String::from_utf8_lossy(&o.stderr));
    out
}

fn refused(o: &Output, what: &str) {
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "expected a refusal containing {what:?}; it succeeded:\n{}", String::from_utf8_lossy(&o.stdout));
    assert!(err.contains(what), "expected {what:?} in:\n{err}");
}

fn fingerprint(out: &str) -> String {
    out.lines().find_map(|l| l.strip_prefix("fingerprint ")).map(|s| s.split_whitespace().next().unwrap().to_string()).expect("a fingerprint line")
}

#[test]
fn an_object_dataset_is_written_byte_for_byte_and_read_back_to_the_fingerprint() {
    let m = minio("an_object_dataset_is_written_byte_for_byte_and_read_back_to_the_fingerprint");
    let dir = tmpdir("large");
    let ast = object_variant("npy_per_sample", &["train"], &dir);
    let ast = ast.to_str().unwrap();
    // 20 MiB samples: three parts each (8 + 8 + 4 MiB), read in 1 MiB requests across the seams
    let params = ["--param", "samples=12", "--param", "sample_mean=20971520", "--param", "sample_sd=0", "--param", "steps=2", "--param", "batch=2", "--param", "workers=1", "--param", "step_time=0"];
    let root = dir.join("root");
    let gen = |root: &Path, endpoint: &str| {
        let mut a = vec!["datagen", ast, "--root", root.to_str().unwrap(), "--endpoint", endpoint];
        a.extend(params);
        aeiou(Some(&m), &a)
    };
    let out = ok(&gen(&root, "train=s3://bench/large"));
    assert!(out.contains("s3 engine: object_store 0.14.2 on tokio (epoll sockets)  runtime threads 2"), "{out}");
    assert!(out.contains("at s3://bench/large  id "), "{out}");
    // a dataset is written once: the prefix is not empty now
    refused(&gen(&root, "train=s3://bench/large"), "s3://bench/large is not empty");

    // the POSIX writer's files, from the same definition, are the objects' bytes
    let posix = dir.join("posix");
    let plain = crate::common::workload_ast("npy_per_sample.ast.json");
    let mut a = vec!["datagen", plain.to_str().unwrap(), "--root", posix.to_str().unwrap()];
    a.extend(params);
    ok(&aeiou(None, &a));
    for id in [0, 5, 11] {
        let rel = format!("{:05}/sample_{id:09}.npz", 0);
        let file = std::fs::read(posix.join("train").join(&rel)).unwrap();
        let obj = curl(&m, &format!("/bench/large/{rel}"), &[]);
        assert!(obj.status.success(), "GET {rel}: {}", String::from_utf8_lossy(&obj.stderr));
        assert_eq!(obj.stdout.len(), 20 << 20, "{rel}");
        assert!(obj.stdout == file, "{rel}: the object's bytes differ from the POSIX writer's");
    }
    // the manifest is an object at the prefix's root
    assert!(curl(&m, "/bench/large/.aeiou-dataset.json", &[]).status.success());

    let mut a = vec!["dry-run", ast, "--gpus", "1"];
    a.extend(params);
    let want = fingerprint(&ok(&aeiou(None, &a)));
    let mut a = vec!["run", ast, "--gpus", "1", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/large", "--expect-fingerprint", &want];
    a.extend(params);
    let out = ok(&aeiou(Some(&m), &a));
    assert!(out.contains("fingerprint matches"), "{out}");
    assert!(out.contains("s3://bench/large is an object store: the report's mount counters and the residency sample do not cover it"), "{out}");
    assert!(out.contains("s3 blocking  root") && out.contains("s3 engine: object_store") && !out.contains("cache per-open"), "{out}");
    // the async client: event loops waiting on the engine's answers, with no ring
    let mut a = vec!["run", ast, "--gpus", "1", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/large", "--expect-fingerprint", &want, "--s3", "async"];
    a.extend(params);
    let out = ok(&aeiou(Some(&m), &a));
    assert!(out.contains("fingerprint matches") && out.contains("s3 async  root"), "{out}");
    assert!(out.contains("s3 async is not the abstract's (s3 blocking)"), "{out}");
}

#[test]
fn listings_and_stats_of_prefixes_read_like_directories() {
    let m = minio("listings_and_stats_of_prefixes_read_like_directories");
    let dir = tmpdir("small");
    let ast = object_variant("file_per_sample", &["train"], &dir);
    let ast = ast.to_str().unwrap();
    // four directories of 1300 files, walked by `stat` and `readdir` at startup (the counts are
    // checked against the definition, so the manifest object is not counted), and a header read
    // of 1 MiB past every file's end: a short read, then a read of nothing
    let params = ["--param", "samples=5200", "--param", "steps=6", "--param", "workers=2", "--param", "step_time=0"];
    let root = dir.join("root");
    let mut a = vec!["datagen", ast, "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/small/t", "--object-threads", "3"];
    a.extend(params);
    assert!(ok(&aeiou(Some(&m), &a)).contains("runtime threads 3"));
    let mut a = vec!["dry-run", ast, "--gpus", "2"];
    a.extend(params);
    let want = fingerprint(&ok(&aeiou(None, &a)));
    let mut a = vec!["run", ast, "--gpus", "2", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/small/t", "--expect-fingerprint", &want];
    a.extend(params);
    let out = ok(&aeiou(Some(&m), &a));
    assert!(out.contains("fingerprint matches"), "{out}");
    assert!(out.contains("readdir=8"), "{out}");
    // the same under an event loop: stats, listings, and short reads come back to the loop
    let mut a = vec!["run", ast, "--gpus", "2", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/small/t", "--expect-fingerprint", &want, "--s3", "async", "--threads", "1"];
    a.extend(params);
    let out = ok(&aeiou(Some(&m), &a));
    assert!(out.contains("fingerprint matches") && out.contains("readdir=8"), "{out}");
    // the manifest says where it was checked, and a missing one says to write it
    let mut a = vec!["run", ast, "--gpus", "2", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/elsewhere"];
    a.extend(params);
    refused(&aeiou(Some(&m), &a), "no manifest at s3://bench/elsewhere/.aeiou-dataset.json");
}

#[test]
fn a_namespace_is_written_as_objects_and_read_back_by_another_run() {
    let m = minio("a_namespace_is_written_as_objects_and_read_back_by_another_run");
    let dir = tmpdir("kv");
    // the shared KV store: every chunk file a header and a 6 MiB chunk written under a temporary
    // name and renamed, so with 5 MiB parts each is a multipart upload of two parts and a copy;
    // the system prompts (datasets inside the namespace's root) live under the same prefix
    let objects = ["sysp", "subp", "kv"];
    let writer = object_variant("lmcache_shared_fs", &objects, &dir);
    let reader = object_variant("lmcache_shared_fs_reader", &objects, &dir);
    let (writer, reader) = (writer.to_str().unwrap(), reader.to_str().unwrap());
    let reuse = r#"reuse={"mixture": [{"weight": 0.3, "dist": null}, {"weight": 0.7, "dist": {"const": 2}}]}"#;
    let keep = r#"keep={"empirical": {"values": [0, 50, 100], "weights": [1, 1, 1]}}"#;
    let mut params = vec!["--seed", "5"];
    for p in ["sys_prompts=2", "sys_tokens=2", "chunk_bytes=6291456", "buf=65536", "concurrency=2", "warm=2", "requests=4", reuse, keep, "retain=6", "sys_local=false", "prefill_per_token=0", "decode_per_token=0"] {
        params.extend(["--param", p]);
    }
    let with = |cmd: &str, ast: &str, extra: &[&str]| {
        let mut a = vec![cmd, ast];
        if cmd != "datagen" {
            a.extend(["--gpus", "1"]);
        }
        a.extend(params.iter().copied().filter(|p| cmd != "datagen" || *p != "--seed" && *p != "5"));
        a.extend(extra);
        a.into_iter().map(String::from).collect::<Vec<_>>()
    };
    let call = |m: Option<&Minio>, a: Vec<String>| aeiou(m, &a.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let root = dir.join("root");
    let root = root.to_str().unwrap();
    let at = ["--root", root, "--endpoint", "kv=s3://bench/kvs"];
    ok(&call(Some(&m), with("datagen", writer, &at)));

    let want = fingerprint(&ok(&call(None, with("dry-run", writer, &[]))));
    let mut a = at.to_vec();
    a.extend(["--expect-fingerprint", &want, "--object-part-mib", "5"]);
    let out = ok(&call(Some(&m), with("run", writer, &a)));
    assert!(out.contains("fingerprint matches"), "{out}");
    assert!(out.contains("runtime threads 2  part 5 MiB"), "{out}");
    assert!(out.contains("namespace manifest s3://bench/kvs/.aeiou-namespace.json"), "{out}");
    // a namespace root holds what a run wrote: a second run is refused, or empties it first,
    // leaving the datasets inside it alone; this one under an event loop, whose objects the
    // POSIX run's files are compared with below
    refused(&call(Some(&m), with("run", writer, &at)), "namespace `kv` root s3://bench/kvs is not empty");
    let mut a = at.to_vec();
    a.extend(["--expect-fingerprint", &want, "--clean-namespaces", "--object-part-mib", "5", "--s3", "async"]);
    let out = ok(&call(Some(&m), with("run", writer, &a)));
    assert!(out.contains("namespace root kv/ emptied") && out.contains("fingerprint matches"), "{out}");

    // the same run on a directory: its files are the objects, name for name and byte for byte
    let posix = dir.join("posix");
    let plain = crate::common::workload_ast("lmcache_shared_fs.ast.json");
    let plain = plain.to_str().unwrap();
    ok(&call(None, with("datagen", plain, &["--root", posix.to_str().unwrap()])));
    ok(&call(None, with("run", plain, &["--root", posix.to_str().unwrap()])));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(posix.join("kv/.aeiou-namespace.json")).unwrap()).unwrap();
    let names: Vec<&str> = manifest["objects"].as_array().unwrap().iter().map(|o| o[0].as_str().unwrap()).collect();
    assert!(!names.is_empty() && names.iter().all(|n| n.ends_with(".data")), "{names:?}");
    let listed = String::from_utf8_lossy(&curl(&m, "/bench?list-type=2&prefix=kvs/", &[]).stdout).to_string();
    for n in &names {
        let rest = n.strip_prefix("kv/").unwrap();
        assert!(listed.contains(&format!("<Key>kvs/{rest}</Key>")), "{rest} not in the bucket");
    }
    assert_eq!(listed.matches(".data</Key>").count(), names.len(), "only the renamed chunks remain: {listed}");
    for n in names.iter().take(3) {
        let rest = n.strip_prefix("kv/").unwrap();
        let obj = curl(&m, &format!("/bench/kvs/{rest}"), &[]);
        assert!(obj.stdout == std::fs::read(posix.join(n)).unwrap(), "{n}: the object's bytes differ from the POSIX run's file");
    }

    // the reader: the namespace is its input, its manifest an object, read to the fingerprint
    let want = fingerprint(&ok(&call(None, with("dry-run", reader, &[]))));
    let mut a = at.to_vec();
    a.extend(["--expect-fingerprint", &want]);
    let out = ok(&call(Some(&m), with("run", reader, &a)));
    assert!(out.contains("input namespace(s) kv at kv/: written by `lmcache_shared_fs`"), "{out}");
    assert!(out.contains("fingerprint matches"), "{out}");
    a.extend(["--s3", "async"]);
    assert!(ok(&call(Some(&m), with("run", reader, &a))).contains("fingerprint matches"));

    // both protocols in one run: the system prompts in a directory, the chunks in the bucket,
    // on the ring and through the engine from the same loops
    std::fs::create_dir_all(dir.join("mixed")).unwrap();
    let mixed = object_variant("lmcache_shared_fs", &["kv"], &dir.join("mixed"));
    let mixed = mixed.to_str().unwrap();
    let mroot = dir.join("mixed/root");
    let mroot = mroot.to_str().unwrap();
    // the prompts' roots lie inside the namespace's: they are placed in directories of their own
    let (sysp, subp) = (format!("sysp={mroot}/sysp"), format!("subp={mroot}/subp"));
    let mat = ["--root", mroot, "--endpoint", "kv=s3://bench/kvmix", "--endpoint", &sysp, "--endpoint", &subp];
    ok(&call(Some(&m), with("datagen", mixed, &mat)));
    let want = fingerprint(&ok(&call(None, with("dry-run", mixed, &[]))));
    let out = ok(&call(Some(&m), with("run", mixed, &[&mat[..], &["--expect-fingerprint", &want, "--posix", "io_uring", "--s3", "async"]].concat())));
    assert!(out.contains("fingerprint matches") && out.contains("posix io_uring  cache per-open  s3 async  root"), "{out}");

    // a write the upload has not reached is refused when it comes (V18: offsets are positional,
    // so the order is known at the write), and the upload is abandoned
    let skip = dir.join("skip.ast.json");
    std::fs::write(
        &skip,
        r#"{"ast": "0.10", "name": "skip",
            "namespaces": {"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 1, "protocol": "s3"}},
            "actors": {"gpu": {"body": [{"loop": {"index": "i", "to": 1, "body": [
                {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"k": {"index": "i"}}}}}},
                {"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT", "TRUNC"]}},
                {"write": {"file": {"ref": "f"}, "len": 4096}},
                {"write": {"file": {"ref": "f"}, "len": 4096, "offset": 8192}},
                {"close": {"file": {"ref": "f"}}}]}}]}}}"#,
    )
    .unwrap();
    let o = aeiou(Some(&m), &["run", skip.to_str().unwrap(), "--gpus", "1", "--root", root, "--endpoint", "o=s3://bench/skip"]);
    refused(&o, "write o/0 at 8192: the object has 4096 bytes written; an upload is written in order from 0 (V18)");
}

#[test]
fn object_names_need_object_endpoints() {
    let dir = tmpdir("refusals");
    let ast = object_variant("file_per_sample", &["train"], &dir);
    let ast = ast.to_str().unwrap();
    let root = dir.join("root");
    let root = root.to_str().unwrap();
    let run = |extra: &[&str]| {
        let mut a = vec!["run", ast, "--gpus", "1", "--root", root, "--param", "samples=1300"];
        a.extend(extra);
        aeiou(None, &a)
    };
    refused(&run(&[]), "`train` is declared `protocol: s3` and has no object endpoint: give it --endpoint train=s3://BUCKET[/PREFIX]");
    refused(&run(&["--endpoint", "train=/somewhere"]), "`train` is declared `protocol: s3`; its endpoint is s3://BUCKET[/PREFIX]");
    // an option for a protocol without names changes nothing, and is refused (§3.65)
    refused(&run(&["--endpoint", "train=s3://bench/t", "--posix", "io_uring"]), "--posix: POSIX options, and every dataset and namespace is `protocol: s3`");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--cache", "direct", "--sqpoll", "10"]), "--cache, --sqpoll: POSIX options, and every dataset and namespace is `protocol: s3`");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--s3", "threads"]), "--s3 threads: not one of blocking, async");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--threads", "2"]), "--threads sets the event-loop threads");
    // both protocols: their APIs are of one kind
    let mixed = object_variant("lmcache_shared_fs", &["kv"], &dir);
    let at = ["run", mixed.to_str().unwrap(), "--gpus", "1", "--root", root, "--endpoint", "kv=s3://bench/x", "--endpoint", "sysp=/p/sysp", "--endpoint", "subp=/p/subp"];
    let o = aeiou(None, &[&at[..], &["--posix", "io_uring"]].concat());
    refused(&o, "--posix io_uring with --s3 blocking: one is event-driven and the other a thread per actor");
    let o = aeiou(None, &[&at[..], &["--s3", "async", "--posix", "mmap"]].concat());
    refused(&o, "--posix mmap with --s3 async");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--object-threads", "0"]), "--object-threads 0");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--object-part-mib", "4"]), "--object-part-mib 4: S3 takes parts of 5 MiB to 5 GiB");
    // a posix abstract has no use for the engine's threads or an object endpoint
    let plain = crate::common::workload_ast("file_per_sample.ast.json");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--object-threads", "4"]);
    refused(&o, "--object-threads: no dataset or namespace is placed in an object store");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--object-part-mib", "8"]);
    refused(&o, "--object-part-mib: no dataset or namespace is placed in an object store");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--endpoint", "train=s3://bench/t"]);
    refused(&o, "`train` is a `posix` dataset or namespace; its endpoint is a directory");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--s3", "async"]);
    refused(&o, "--s3: no dataset or namespace is `protocol: s3`");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--posix"]);
    refused(&o, "a value is required for '--posix <API>'");
    // an array file has no object form
    let vdb = object_variant("diskann_search", &["index"], &dir);
    // its other names stay posix under its declared `libaio`, so the S3 client is async (V19)
    let mut d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&vdb).unwrap()).unwrap();
    d["s3"] = "async".into();
    std::fs::write(&vdb, d.to_string()).unwrap();
    refused(&aeiou(None, &["datagen", vdb.to_str().unwrap(), "--root", root, "--endpoint", "index=s3://bench/vdb"]), "dataset `index`: an `array` dataset is a sparse layout written at offsets and has no object form");
    // and before the space, which a refused datagen never asks to be freed: its posix `pq`,
    // grown to 888 PiB, fits on no host
    let o = aeiou(None, &["datagen", vdb.to_str().unwrap(), "--root", root, "--endpoint", "index=s3://bench/vdb", "--param", "pq_code=1000000000"]);
    refused(&o, "dataset `index`: an `array` dataset is a sparse layout written at offsets and has no object form");
    assert!(!String::from_utf8_lossy(&o.stderr).contains("space:"), "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn a_missing_object_is_enoent_whatever_its_key_and_a_read_past_the_end_is_eof() {
    // the engine reads a ranged GET's 416 as EOF; a 404 on a key with "416" in it (here
    // `img_000000416.jpg`), whose message carries the key, is ENOENT all the same
    let m = minio("a_missing_object_is_enoent_whatever_its_key_and_a_read_past_the_end_is_eof");
    let dir = tmpdir("416");
    let ast = object_variant("file_per_sample", &["train"], &dir);
    // without its `fstat`s (a HEAD, which would find the object missing first), so that the
    // ranged GET is the first request for each sample
    fn no_fstat(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Array(a) => {
                a.retain(|n| n.get("fstat").is_none());
                a.iter_mut().for_each(no_fstat);
            }
            serde_json::Value::Object(o) => o.values_mut().for_each(no_fstat),
            _ => {}
        }
    }
    let mut d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ast).unwrap()).unwrap();
    no_fstat(&mut d);
    std::fs::write(&ast, d.to_string()).unwrap();
    let ast = ast.to_str().unwrap();
    // one epoch of all 512 samples, every one read in 1 MiB requests past its end (a short
    // read, then a read of nothing: the 416), no walk to count the files first
    let params = ["--param", "samples=512", "--param", "steps=16", "--param", "batch=32", "--param", "workers=1", "--param", "step_time=0", "--param", "walk=false"];
    let root = dir.join("root");
    let mut a = vec!["datagen", ast, "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/t416"];
    a.extend(params);
    ok(&aeiou(Some(&m), &a));
    let mut a = vec!["dry-run", ast, "--gpus", "1"];
    a.extend(params);
    let want = fingerprint(&ok(&aeiou(None, &a)));
    let mut a = vec!["run", ast, "--gpus", "1", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/t416", "--expect-fingerprint", &want];
    a.extend(params);
    for s3 in ["blocking", "async"] {
        assert!(ok(&aeiou(Some(&m), &[&a[..], &["--s3", s3]].concat())).contains("fingerprint matches"));
    }
    let gone = "/bench/t416/00000/img_000000416.jpg";
    assert!(curl(&m, gone, &["-X", "DELETE"]).status.success());
    assert!(!curl(&m, gone, &[]).status.success(), "{gone} is still there");
    for s3 in ["blocking", "async"] {
        let o = aeiou(Some(&m), &[&a[..], &["--s3", s3]].concat());
        refused(&o, "s3://bench/t416/00000/img_000000416.jpg");
        refused(&o, "not found");
        refused(&o, "(ENOENT)");
    }
}

/// A resume of an object dataset sends only the unfinished ids whose keys are missing, and
/// counts the ones it finds there as a directory resume counts the ids it writes again
/// (`runner/REFERENCE.md` §5, Resume): the corpus's totals are every file once (RC1 review).
/// Its names hold `~` and `#`, which are keys as written: the listed keys are the names the
/// resume makes, so it finds them (RC1 review, bug R28).
#[test]
fn a_resumed_object_dataset_counts_the_objects_it_finds() {
    let m = minio("a_resumed_object_dataset_counts_the_objects_it_finds");
    let dir = tmpdir("resume");
    let ast = object_variant("npy_per_sample", &["train"], &dir);
    let mut d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ast).unwrap()).unwrap();
    d["datasets"]["train"]["pattern"] = "train/{id div files_per_directory:05}/s~ample#{id:09}.npz".into();
    std::fs::write(&ast, d.to_string()).unwrap();
    let root = dir.join("root");
    let mut a = vec!["datagen", ast.to_str().unwrap(), "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/res"];
    a.extend(["--param", "samples=10", "--param", "sample_mean=1000", "--param", "sample_sd=0"]);
    ok(&aeiou(Some(&m), &a));
    let get = || -> serde_json::Value { serde_json::from_slice(&curl(&m, "/bench/res/.aeiou-dataset.json", &[]).stdout).unwrap() };
    let full = get();
    let (files, bytes) = (full["provenance"]["files_written"].as_u64().unwrap(), full["provenance"]["bytes_written"].as_u64().unwrap());
    assert_eq!(files, 10, "{full}");
    // a datagen killed before its first window: every id unfinished, every object but two sent
    let mut w = full.clone();
    w["state"] = "writing".into();
    w["progress"] = serde_json::json!({"unfinished": [[0, 10]], "files_written": 0, "bytes_written": 0, "updated": 1.0, "progress_secs": 30.0});
    let wfile = dir.join("writing.json");
    std::fs::write(&wfile, serde_json::to_vec(&w).unwrap()).unwrap();
    assert!(curl(&m, "/bench/res/.aeiou-dataset.json", &["-X", "PUT", "--data-binary", &format!("@{}", wfile.display())]).status.success());
    for id in [3, 7] {
        assert!(curl(&m, &format!("/bench/res/00000/s~ample%23{id:09}.npz"), &["-X", "DELETE"]).status.success());
    }
    let out = ok(&aeiou(Some(&m), &["datagen", "--resume", "s3://bench/res"]));
    let after = get();
    assert_eq!(after["state"], "complete", "{after}");
    assert_eq!((after["provenance"]["files_written"].as_u64(), after["provenance"]["bytes_written"].as_u64()), (Some(files), Some(bytes)), "{after}");
    // the resume's own record is what it sent: the two deleted, every sample the same size;
    // the eight found count toward the totals only
    let r = &after["provenance"]["resumes"][0];
    assert_eq!((r["files_written"].as_u64(), r["bytes_written"].as_u64()), (Some(2), Some(bytes / 5)), "{after}");
    assert_eq!((r["files_found"].as_u64(), r["bytes_found"].as_u64()), (Some(8), Some(bytes / 5 * 4)), "{after}");
    assert!(out.contains("dataset train: 2 file(s), "), "{out}");
    assert!(out.contains(", 8 already there at "), "{out}");
    for id in [3, 7] {
        assert!(curl(&m, &format!("/bench/res/00000/s~ample%23{id:09}.npz"), &[]).status.success(), "object {id} was not sent again");
    }
}

/// An object's key is its name as the application writes it, byte for byte: `~ [ ] # %` are
/// not percent-encoded, so the bucket holds the names the POSIX run's directory holds, and
/// `--clean-namespaces` deletes the keys it lists, foreign and stale ones included, so the
/// next run finds the root empty. A name no key holds as POSIX does is refused (RC1 review,
/// bug R28).
#[test]
fn object_keys_are_the_names_as_written() {
    let m = minio("object_keys_are_the_names_as_written");
    let dir = tmpdir("keys");
    let ast = |to: i64, pattern: &str, protocol: bool| {
        let mut d = serde_json::json!({"ast": "0.10", "name": "sp",
            "namespaces": {"o": {"pattern": pattern, "fields": {"k": "int"}, "size": "as_written", "seed": 1, "protocol": "s3"}},
            "actors": {"gpu": {"body": [{"loop": {"index": "i", "to": to, "body": [
                {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"k": {"index": "i"}}}}}},
                {"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT", "TRUNC"]}},
                {"write": {"file": {"ref": "f"}, "len": 4096}},
                {"close": {"file": {"ref": "f"}}},
                {"stat": {"file": {"ref": "f"}}}]}}]}}});
        if !protocol {
            d["namespaces"]["o"].as_object_mut().unwrap().remove("protocol");
        }
        let p = dir.join(format!("sp{to}{protocol}{}.ast.json", pattern.len()));
        std::fs::write(&p, d.to_string()).unwrap();
        p
    };
    let name = "o/ck~pt [{k}]#%.bin";
    let root = dir.join("root");
    let root = root.to_str().unwrap();
    let run = |ast: &Path, extra: &[&str]| aeiou(Some(&m), &[&["run", ast.to_str().unwrap(), "--gpus", "1", "--root", root, "--endpoint", "o=s3://bench/sp"][..], extra].concat());
    let listed = || String::from_utf8_lossy(&curl(&m, "/bench?list-type=2&prefix=sp/", &[]).stdout).to_string();
    let two = ast(2, name, true);
    ok(&run(&two, &[]));
    // the POSIX run's names are the keys
    let posix = dir.join("posix");
    ok(&aeiou(None, &["run", ast(2, name, false).to_str().unwrap(), "--gpus", "1", "--root", posix.to_str().unwrap()]));
    let mut files: Vec<String> = std::fs::read_dir(posix.join("o")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).filter(|n| !n.starts_with(".aeiou")).collect();
    files.sort();
    assert_eq!(files, ["ck~pt [0]#%.bin", "ck~pt [1]#%.bin"]);
    let l = listed();
    for f in &files {
        assert!(l.contains(&format!("<Key>sp/{f}</Key>")), "{f} not in the bucket: {l}");
    }
    let manifest = String::from_utf8_lossy(&curl(&m, "/bench/sp/.aeiou-namespace.json", &[]).stdout).to_string();
    assert!(manifest.contains("o/ck~pt [0]#%.bin"), "{manifest}");
    // a foreign key and a stale one: a run is refused, naming one as it is listed; with
    // --clean-namespaces it deletes both, and its own one object is what is left
    assert!(curl(&m, "/bench/sp/foreign~key", &["-X", "PUT", "--data-binary", "x"]).status.success());
    let one = ast(1, name, true);
    refused(&run(&one, &[]), "namespace `o` root s3://bench/sp is not empty (4 objects, e.g. s3://bench/sp/.aeiou-namespace.json)");
    assert!(ok(&run(&one, &["--clean-namespaces"])).contains("namespace root o/ emptied"));
    let l = listed();
    assert_eq!(l.matches("<Key>").count(), 2, "{l}");
    assert!(l.contains("<Key>sp/ck~pt [0]#%.bin</Key>") && l.contains("<Key>sp/.aeiou-namespace.json</Key>"), "{l}");
    // the next run finds the root as the last one left it
    refused(&run(&one, &[]), "namespace `o` root s3://bench/sp is not empty (2 objects");
    assert!(ok(&run(&one, &["--clean-namespaces"])).contains("namespace root o/ emptied"));
    // a `..` component no key holds as POSIX does
    let o = aeiou(Some(&m), &["run", ast(1, "o/{k}/../x", true).to_str().unwrap(), "--gpus", "1", "--root", root, "--endpoint", "o=s3://bench/dots"]);
    refused(&o, "s3://bench/dots/0/../x: no object key holds the component \"..\" (`.`, `..`, or a control character)");
}

/// Two endpoints in one bucket are one key space, though each has a store of its own: a
/// namespace emptied by `--clean-namespaces` keeps the keys of a dataset (V12) and of an
/// input namespace (V14) placed below its prefix through their own endpoints, and a `rename`
/// from one endpoint's prefix to another's is a copy between their keys, not `EXDEV`; one into
/// another bucket is (RC1 review).
#[test]
fn endpoints_in_one_bucket_share_its_keys() {
    let m = minio("endpoints_in_one_bucket_share_its_keys");
    let dir = tmpdir("onebucket");
    let root = dir.join("root");
    let root = root.to_str().unwrap();
    let write = |name: &str, more: &str, namespaces: &str, body: &str| {
        let p = dir.join(format!("{name}.ast.json"));
        let ast = format!(r#"{{"ast": "0.10", "name": "{name}", {more} "namespaces": {{{namespaces}}}, "actors": {{"gpu": {{"body": [{body}]}}}}}}"#);
        std::fs::write(&p, ast).unwrap();
        p
    };
    let ns = |name: &str, extra: &str| format!(r#""{name}": {{"pattern": "{name}/f{{k}}", "fields": {{"k": "int"}}, "size": "as_written", "seed": 1, "protocol": "s3"{extra}}}"#);
    let bind = |v: &str, n: &str| format!(r#"{{"let": {{"name": "{v}", "value": {{"object": {{"namespace": "{n}", "fields": {{"k": 0}}}}}}}}}}"#);
    let put = |v: &str| format!(r#"{{"open": {{"file": {{"ref": "{v}"}}, "flags": ["WRONLY", "CREAT", "TRUNC"]}}}}, {{"write": {{"file": {{"ref": "{v}"}}, "len": 4096}}}}, {{"close": {{"file": {{"ref": "{v}"}}}}}}"#);
    let get = |v: &str| format!(r#"{{"open": {{"file": {{"ref": "{v}"}}, "flags": ["RDONLY"]}}}}, {{"read": {{"file": {{"ref": "{v}"}}, "len": 4096}}}}, {{"close": {{"file": {{"ref": "{v}"}}}}}}"#);
    let run = |ast: &Path, extra: &[&str]| {
        let mut a = vec!["run", ast.to_str().unwrap(), "--gpus", "1", "--root", root];
        a.extend(extra);
        aeiou(Some(&m), &a)
    };
    let present = |key: &str| curl(&m, &format!("/bench/{key}"), &[]).status.success();

    // a dataset at ns/data, the namespace `o` at ns: emptied twice, the dataset stays and is read
    let dataset = r#""datasets": {"d": {"layout": "whole", "access": "indexed", "pattern": "d/f_{id}", "count": 2, "size": {"const": 4096}, "seed": 7, "protocol": "s3"}},"#;
    let read_d = r#"{"let": {"name": "g", "value": {"file": {"dataset": "d", "id": 0}}}}"#;
    let dn = write("dn", dataset, &ns("o", ""), &format!("{read_d}, {}, {}, {}", get("g"), bind("f", "o"), put("f")));
    let at = ["--endpoint", "d=s3://bench/ns/data", "--endpoint", "o=s3://bench/ns"];
    ok(&aeiou(Some(&m), &[&["datagen", dn.to_str().unwrap(), "--root", root][..], &at[..]].concat()));
    ok(&run(&dn, &at));
    for _ in 0..2 {
        let out = ok(&run(&dn, &[&at[..], &["--clean-namespaces"]].concat()));
        assert!(out.contains("namespace root o/ emptied"), "{out}");
    }
    assert!(present("ns/f0") && present("ns/data/f_0") && present("ns/data/f_1") && present("ns/data/.aeiou-dataset.json"));

    // an input namespace at in/in, the output `q` at in: the input's objects and manifest stay
    let w = write("w", "", &ns("o", ""), &format!("{}, {}", bind("f", "o"), put("f")));
    ok(&run(&w, &["--endpoint", "o=s3://bench/in/in"]));
    let r = write("r", "", &format!("{}, {}", ns("o", r#", "input": true"#), ns("q", "")), &format!("{}, {}, {}, {}", bind("f", "o"), get("f"), bind("g", "q"), put("g")));
    let at = ["--endpoint", "o=s3://bench/in/in", "--endpoint", "q=s3://bench/in"];
    ok(&run(&r, &at));
    for _ in 0..2 {
        let out = ok(&run(&r, &[&at[..], &["--clean-namespaces"]].concat()));
        assert!(out.contains("namespace root q/ emptied"), "{out}");
    }
    assert!(present("in/in/f0") && present("in/in/.aeiou-namespace.json") && present("in/f0"));

    // a rename from one endpoint's prefix to another's in the bucket, and into another bucket
    let rn = write("rn", "", &format!("{}, {}", ns("a", ""), ns("b", "")), &format!(r#"{}, {}, {}, {{"rename": {{"from": {{"ref": "f"}}, "to": {{"ref": "g"}}}}}}"#, bind("f", "a"), bind("g", "b"), put("f")));
    ok(&run(&rn, &["--endpoint", "a=s3://bench/x/tmp", "--endpoint", "b=s3://bench/x/fin"]));
    assert!(present("x/fin/f0") && !present("x/tmp/f0"));
    assert!(curl(&m, "/other", &["-X", "PUT"]).status.success());
    refused(&run(&rn, &["--endpoint", "a=s3://bench/y/tmp", "--endpoint", "b=s3://other/fin"]), "rename a/f0 to b/f0: the target is in another bucket (s3://other/fin)");
}
