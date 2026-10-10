//! The object engine (`object.rs`, the cargo feature `object`) through the binary: `aeiou
//! datagen` writes a dataset declared `protocol: object` into a MinIO bucket, the bytes of
//! each object are the bytes the POSIX writer puts in the file of the same name (a multipart
//! upload included), and `aeiou run` reads it back to the dry run's fingerprint, every read
//! returning the computed count; a run writes a namespace into a bucket (uploads, multipart
//! ones included, `rename` as a copy and a `DELETE`, the namespace manifest as an object), its
//! objects the bytes a POSIX run writes, and a second abstract reads it back as its input. The
//! refusals need no server.
//!
//! The tests that need a server start their own MinIO from the binary `TEST_MINIO_BIN` names
//! (CI downloads the pinned release; `runner/REFERENCE.md` §16), and are skipped, saying so,
//! when it is unset.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static N: AtomicUsize = AtomicUsize::new(0);

const USER: &str = "aeiou";
const SECRET: &str = "aeiou-secret";

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aeiou-object-{}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples")
}

/// A committed example with its datasets and namespaces `names` declared `protocol: object`,
/// as a file in `dir`.
fn object_variant(example: &str, names: &[&str], dir: &Path) -> PathBuf {
    let mut d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(examples().join(format!("{example}.ast.json"))).unwrap()).unwrap();
    for n in names {
        if let Some(ns) = d.get_mut("namespaces").and_then(|v| v.get_mut(*n)) {
            ns["protocol"] = "object".into();
            continue;
        }
        let ds = d["datasets"][*n].as_object_mut().unwrap();
        let (_, body) = ds.iter_mut().next().unwrap();
        body["protocol"] = "object".into();
    }
    let p = dir.join(format!("{example}.object.ast.json"));
    std::fs::write(&p, serde_json::to_string(&d).unwrap()).unwrap();
    p
}

/// A MinIO server on a free port with a directory of its own, killed when dropped.
struct Minio {
    child: Child,
    url: String,
    _dir: PathBuf,
}

impl Drop for Minio {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self._dir);
    }
}

fn minio(test: &str) -> Option<Minio> {
    let Some(bin) = std::env::var_os("TEST_MINIO_BIN") else {
        eprintln!("{test}: skipped, TEST_MINIO_BIN is not set (a MinIO server binary)");
        return None;
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
    while !curl(&m, &["-X", "PUT", &format!("{}/bench", m.url)]).status.success() {
        assert!(t0.elapsed() < Duration::from_secs(30), "MinIO did not come up at {}", m.url);
        std::thread::sleep(Duration::from_millis(200));
    }
    Some(m)
}

/// A request to the server signed with the test's credentials; fails on an HTTP error.
fn curl(m: &Minio, args: &[&str]) -> Output {
    Command::new("curl").args(["-sf", "--aws-sigv4", "aws:amz:us-east-1:s3", "--user", &format!("{USER}:{SECRET}")]).args(args).env_remove("http_proxy").output().unwrap()
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
    let Some(m) = minio("an_object_dataset_is_written_byte_for_byte_and_read_back_to_the_fingerprint") else { return };
    let dir = tmpdir("large");
    let ast = object_variant("train_large_samples", &["train"], &dir);
    let ast = ast.to_str().unwrap();
    // 20 MiB samples: three parts each (8 + 8 + 4 MiB), read in 1 MiB requests across the seams
    let params = ["--param", "files=12", "--param", "sample_mean=20971520", "--param", "sample_sd=0", "--param", "steps=2", "--param", "batch=2", "--param", "workers=1", "--param", "step_time=0"];
    let root = dir.join("root");
    let gen = |root: &Path, endpoint: &str| {
        let mut a = vec!["datagen", ast, "--root", root.to_str().unwrap(), "--endpoint", endpoint];
        a.extend(params);
        aeiou(Some(&m), &a)
    };
    let out = ok(&gen(&root, "train=s3://bench/large"));
    assert!(out.contains("object engine: object_store 0.14.2  runtime threads 2"), "{out}");
    assert!(out.contains("at s3://bench/large  id "), "{out}");
    // a dataset is written once: the prefix is not empty now
    refused(&gen(&root, "train=s3://bench/large"), "s3://bench/large is not empty");

    // the POSIX writer's files, from the same definition, are the objects' bytes
    let posix = dir.join("posix");
    let plain = examples().join("train_large_samples.ast.json");
    let mut a = vec!["datagen", plain.to_str().unwrap(), "--root", posix.to_str().unwrap()];
    a.extend(params);
    ok(&aeiou(None, &a));
    for id in [0, 5, 11] {
        let rel = format!("{:05}/sample_{id:09}.npz", 0);
        let file = std::fs::read(posix.join("train").join(&rel)).unwrap();
        let obj = curl(&m, &[&format!("{}/bench/large/{rel}", m.url)]);
        assert!(obj.status.success(), "GET {rel}: {}", String::from_utf8_lossy(&obj.stderr));
        assert_eq!(obj.stdout.len(), 20 << 20, "{rel}");
        assert!(obj.stdout == file, "{rel}: the object's bytes differ from the POSIX writer's");
    }
    // the manifest is an object at the prefix's root
    assert!(curl(&m, &[&format!("{}/bench/large/.aeiou-dataset.json", m.url)]).status.success());

    let mut a = vec!["dry-run", ast, "--gpus", "1"];
    a.extend(params);
    let want = fingerprint(&ok(&aeiou(None, &a)));
    let mut a = vec!["run", ast, "--gpus", "1", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/large", "--expect-fingerprint", &want];
    a.extend(params);
    let out = ok(&aeiou(Some(&m), &a));
    assert!(out.contains("fingerprint matches"), "{out}");
    assert!(out.contains("s3://bench/large is an object store: the report's mount counters and the residency sample do not cover it"), "{out}");
    // the run's other APIs leave the object engine's ops alone, the event loops waiting on it
    for api in ["posix-aio", "mmap", "io_uring", "libaio"] {
        let mut a = vec!["run", ast, "--gpus", "1", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/large", "--expect-fingerprint", &want, "--io-api", api];
        a.extend(params);
        assert!(ok(&aeiou(Some(&m), &a)).contains("fingerprint matches"), "{api}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn listings_and_stats_of_prefixes_read_like_directories() {
    let Some(m) = minio("listings_and_stats_of_prefixes_read_like_directories") else { return };
    let dir = tmpdir("small");
    let ast = object_variant("train_small_files", &["train"], &dir);
    let ast = ast.to_str().unwrap();
    // four directories of 1300 files, walked by `stat` and `readdir` at startup (the counts are
    // checked against the definition, so the manifest object is not counted), and a header read
    // of 1 MiB past every file's end: a short read, then a read of nothing
    let params = ["--param", "files=5200", "--param", "steps=6", "--param", "workers=2", "--param", "step_time=0"];
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
    let mut a = vec!["run", ast, "--gpus", "2", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/small/t", "--expect-fingerprint", &want, "--io-api", "io_uring", "--threads", "1"];
    a.extend(params);
    let out = ok(&aeiou(Some(&m), &a));
    assert!(out.contains("fingerprint matches") && out.contains("readdir=8"), "{out}");
    // the manifest says where it was checked, and a missing one says to write it
    let mut a = vec!["run", ast, "--gpus", "2", "--root", root.to_str().unwrap(), "--endpoint", "train=s3://bench/elsewhere"];
    a.extend(params);
    refused(&aeiou(Some(&m), &a), "no manifest at s3://bench/elsewhere/.aeiou-dataset.json");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_namespace_is_written_as_objects_and_read_back_by_another_run() {
    let Some(m) = minio("a_namespace_is_written_as_objects_and_read_back_by_another_run") else { return };
    let dir = tmpdir("kv");
    // the shared KV store: every chunk file a header and a 6 MiB chunk written under a temporary
    // name and renamed, so with 5 MiB parts each is a multipart upload of two parts and a copy;
    // the system prompts (datasets inside the namespace's root) live under the same prefix
    let objects = ["sysp", "subp", "kv"];
    let writer = object_variant("kv_cache_shared", &objects, &dir);
    let reader = object_variant("kv_cache_shared_reader", &objects, &dir);
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
    a.extend(["--expect-fingerprint", &want, "--clean-namespaces", "--object-part-mib", "5", "--io-api", "io_uring"]);
    let out = ok(&call(Some(&m), with("run", writer, &a)));
    assert!(out.contains("namespace root kv/ emptied") && out.contains("fingerprint matches"), "{out}");

    // the same run on a directory: its files are the objects, name for name and byte for byte
    let posix = dir.join("posix");
    let plain = examples().join("kv_cache_shared.ast.json");
    let plain = plain.to_str().unwrap();
    ok(&call(None, with("datagen", plain, &["--root", posix.to_str().unwrap()])));
    ok(&call(None, with("run", plain, &["--root", posix.to_str().unwrap()])));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(posix.join("kv/.aeiou-namespace.json")).unwrap()).unwrap();
    let names: Vec<&str> = manifest["objects"].as_array().unwrap().iter().map(|o| o[0].as_str().unwrap()).collect();
    assert!(!names.is_empty() && names.iter().all(|n| n.ends_with(".data")), "{names:?}");
    let listed = String::from_utf8_lossy(&curl(&m, &[&format!("{}/bench?list-type=2&prefix=kvs/", m.url)]).stdout).to_string();
    for n in &names {
        let rest = n.strip_prefix("kv/").unwrap();
        assert!(listed.contains(&format!("<Key>kvs/{rest}</Key>")), "{rest} not in the bucket");
    }
    assert_eq!(listed.matches(".data</Key>").count(), names.len(), "only the renamed chunks remain: {listed}");
    for n in names.iter().take(3) {
        let rest = n.strip_prefix("kv/").unwrap();
        let obj = curl(&m, &[&format!("{}/bench/kvs/{rest}", m.url)]);
        assert!(obj.stdout == std::fs::read(posix.join(n)).unwrap(), "{n}: the object's bytes differ from the POSIX run's file");
    }

    // the reader: the namespace is its input, its manifest an object, read to the fingerprint
    let want = fingerprint(&ok(&call(None, with("dry-run", reader, &[]))));
    let mut a = at.to_vec();
    a.extend(["--expect-fingerprint", &want]);
    let out = ok(&call(Some(&m), with("run", reader, &a)));
    assert!(out.contains("input namespace(s) kv at kv/: written by `kv_cache_shared`"), "{out}");
    assert!(out.contains("fingerprint matches"), "{out}");
    a.extend(["--io-api", "libaio"]);
    assert!(ok(&call(Some(&m), with("run", reader, &a))).contains("fingerprint matches"));

    // a write the upload has not reached is refused when it comes (V18: offsets are positional,
    // so the order is known at the write), and the upload is abandoned
    let skip = dir.join("skip.ast.json");
    std::fs::write(
        &skip,
        r#"{"ast": "0.6", "name": "skip",
            "namespaces": {"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 1, "protocol": "object"}},
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
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn object_names_need_object_endpoints() {
    let dir = tmpdir("refusals");
    let ast = object_variant("train_small_files", &["train"], &dir);
    let ast = ast.to_str().unwrap();
    let root = dir.join("root");
    let root = root.to_str().unwrap();
    let run = |extra: &[&str]| {
        let mut a = vec!["run", ast, "--gpus", "1", "--root", root, "--param", "files=1300"];
        a.extend(extra);
        aeiou(None, &a)
    };
    refused(&run(&[]), "`train` is declared `protocol: object` and has no object endpoint: give it --endpoint train=s3://BUCKET[/PREFIX]");
    refused(&run(&["--endpoint", "train=/somewhere"]), "`train` is declared `protocol: object`; its endpoint is s3://BUCKET[/PREFIX]");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--object-threads", "0"]), "--object-threads 0");
    refused(&run(&["--endpoint", "train=s3://bench/t", "--object-part-mib", "4"]), "--object-part-mib 4: S3 takes parts of 5 MiB to 5 GiB");
    // a posix abstract has no use for the engine's threads or an object endpoint
    let plain = examples().join("train_small_files.ast.json");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--object-threads", "4"]);
    refused(&o, "--object-threads: no dataset or namespace is placed in an object store");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--object-part-mib", "8"]);
    refused(&o, "--object-part-mib: no dataset or namespace is placed in an object store");
    let o = aeiou(None, &["run", plain.to_str().unwrap(), "--gpus", "1", "--root", root, "--endpoint", "train=s3://bench/t"]);
    refused(&o, "`train` is a `posix` dataset or namespace; its endpoint is a directory");
    // a regions file has no object form
    let vdb = object_variant("vdb_search_diskann", &["index"], &dir);
    refused(&aeiou(None, &["datagen", vdb.to_str().unwrap(), "--root", root, "--endpoint", "index=s3://bench/vdb"]), "dataset `index`: a `regions` dataset is a sparse layout written at offsets and has no object form");
    std::fs::remove_dir_all(&dir).unwrap();
}
