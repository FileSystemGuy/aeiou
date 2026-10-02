//! `aeiou run --report-json` (`report.rs`, `runner/README.md` §12), through the binary: the
//! document of a passing run against its text report, of a failed verdict and of a run that
//! never started, and of two ranks (rank 0 holds the merged report and its own, rank 1 its
//! own with the run's fingerprint).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

static N: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aeiou-report-{}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn aeiou(sub: &str, ast: &str, params: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.arg(sub).arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples").join(ast));
    for p in params {
        c.arg("--param").arg(p);
    }
    c
}

fn out(c: &mut Command) -> (bool, String, String) {
    let o = c.output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

const SMALL: [&str; 6] = ["files=600", "batch=4", "workers=2", "prefetch=2", "steps=12", "enumerate=true"];

#[test]
fn one_host_report_matches_the_text_and_failures_are_written() {
    let root = tmpdir("one");
    let json = root.join("report.json");
    let (ok, _, err) = out(aeiou("datagen", "train_small_files.ast.json", &SMALL).arg("--root").arg(&root).args(["--gpus", "2"]));
    assert!(ok, "{err}");
    let run = |extra: &[&str]| {
        let mut c = aeiou("run", "train_small_files.ast.json", &SMALL);
        c.arg("--root").arg(&root).args(["--gpus", "2", "--seed", "7", "--time-scale", "0"]).arg("--report-json").arg(&json).args(extra);
        out(&mut c)
    };

    let (ok, text, err) = run(&["--report-takes"]);
    assert!(ok, "{text}\n{err}");
    let d = read(&json);
    assert_eq!(d["aeiou_report"], 1);
    assert_eq!(d["abstract"]["name"], "train_small_files");
    assert!(text.contains(&format!("sha256 {}", d["abstract"]["sha256"].as_str().unwrap())));
    assert_eq!((d["seed"].as_u64(), d["gpus"].as_u64(), d["rank"].as_u64(), d["ranks"].as_u64()), (Some(7), Some(2), Some(0), Some(1)));
    assert_eq!(d["params"]["files"], 600);
    assert_eq!(d["backend"], "sync");
    assert_eq!(d["datasets"][0]["files"], 600);
    assert!(text.contains(&format!("id {}", d["datasets"][0]["id"].as_str().unwrap())));
    assert_eq!(d["scope"], "run");
    assert_eq!(d["verdict"]["ok"], true);
    assert_eq!(d["verdict"]["fingerprint_scope"], "run");
    let fp = d["verdict"]["fingerprint"].as_str().unwrap().to_string();
    assert!(text.contains(&format!("fingerprint {fp}")), "{text}");
    assert!(d["finished"].as_f64().unwrap() >= d["started"].as_f64().unwrap());

    let r = &d["result"];
    assert_eq!(r["fingerprint"], fp.as_str());
    let ops = r["ops"].as_u64().unwrap();
    assert!(text.contains(&format!("ops {ops}  read ")), "{text}");
    assert_eq!(r["by_kind"].as_object().unwrap().values().map(|v| v.as_u64().unwrap()).sum::<u64>(), ops);
    assert_eq!(r["bytes_written"], 0);
    assert_eq!(r["templates"]["gpu"], 2);
    assert_eq!(r["objects_created"], 0);
    // a histogram's buckets hold every op of its kind, in increasing bounds, and the
    // quantiles are bucket bounds in order
    for (kind, n) in r["by_kind"].as_object().unwrap() {
        let h = &r["latency"][kind];
        assert_eq!(&h["count"], n, "{kind}");
        let buckets: Vec<(u64, u64)> = h["buckets"].as_array().unwrap().iter().map(|b| (b[0].as_u64().unwrap(), b[1].as_u64().unwrap())).collect();
        assert_eq!(buckets.iter().map(|b| b.1).sum::<u64>(), n.as_u64().unwrap(), "{kind}");
        assert!(buckets.windows(2).all(|w| w[0].0 < w[1].0), "{kind}");
        let q = |k: &str| h[k].as_u64().unwrap();
        assert!(q("p50_ns") <= q("p90_ns") && q("p90_ns") <= q("p99_ns") && q("p99_ns") <= q("p999_ns") && q("p999_ns") <= q("max_ns"), "{kind}");
        assert!(buckets.iter().any(|b| b.0 == q("p99_ns")), "{kind}");
    }
    assert_eq!(r["phases"]["enumerate"]["ops"], 10);   // 2 instances, 1 directory: stat, open, fstat, readdir, close
    // the takes: 12 per instance, the sums of the instances' records
    let t = &r["take_summary"];
    assert_eq!((t["per_instance"].as_u64(), t["instances"].as_u64(), t["takes"].as_u64()), (Some(12), Some(2), Some(24)));
    assert_eq!(t["steps"].as_array().unwrap().iter().map(|s| s["takes"].as_u64().unwrap()).sum::<u64>(), 24);
    let instances = r["instances"].as_array().unwrap();
    assert_eq!(instances.len(), 2);
    let mut stall = 0;
    for i in instances {
        let recs = i["take_records"].as_array().unwrap();
        assert_eq!(recs.len(), 12);
        assert_eq!(recs.iter().map(|t| t[0].as_u64().unwrap()).sum::<u64>(), i["stall_ns"].as_u64().unwrap());
        stall += i["stall_ns"].as_u64().unwrap();
    }
    assert_eq!(t["stall_ns"].as_u64().unwrap(), stall);

    // without --report-takes the records are left out
    let (ok, _, _) = run(&[]);
    assert!(ok);
    assert!(read(&json)["result"]["instances"][0].get("take_records").is_none());

    // a failed verdict: the results are there, with the error
    let (ok, _, err) = run(&["--expect-fingerprint", "1234"]);
    assert!(!ok);
    let d = read(&json);
    assert_eq!(d["verdict"]["ok"], false);
    assert!(err.contains(d["verdict"]["error"].as_str().unwrap()), "{err}");
    assert_eq!(d["verdict"]["expected_fingerprint"], "0000000000001234");
    assert_eq!(d["verdict"]["fingerprint"], fp.as_str());
    assert_eq!(d["result"]["ops"].as_u64(), Some(ops));

    // a run that never starts replaces the earlier report: identity and error, no result
    let mut c = aeiou("run", "train_small_files.ast.json", &SMALL);
    let (ok, _, _) = out(c.arg("--root").arg(root.join("nowhere")).args(["--gpus", "2"]).arg("--report-json").arg(&json));
    assert!(!ok);
    let d = read(&json);
    assert_eq!(d["verdict"]["ok"], false);
    assert!(d["verdict"]["error"].as_str().unwrap().contains("aeiou datagen"));
    assert_eq!(d["abstract"]["name"], "train_small_files");
    assert!(d["result"].is_null() && d["scope"].is_null() && d["verdict"]["fingerprint"].is_null());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn two_ranks_merged_on_rank_0_and_own_on_rank_1() {
    let root = tmpdir("two");
    let params = ["steps=4", "ckpt_every=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "meta_bytes=65536"];
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut children = Vec::new();
    for rank in 0..2 {
        let mut c = aeiou("run", "ckpt_write_dcp.ast.json", &params);
        c.arg("--root").arg(&root).args(["--gpus", "2", "--time-scale", "0", "--ranks", "2"]).arg("--rank").arg(rank.to_string());
        c.arg("--coordinator").arg(format!("127.0.0.1:{port}")).arg("--report-json").arg(root.join(format!("report-{rank}.json")));
        children.push(c.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
        if rank == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    for c in children {
        let o = c.wait_with_output().unwrap();
        assert!(o.status.success(), "{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    }
    let (d0, d1) = (read(&root.join("report-0.json")), read(&root.join("report-1.json")));
    assert_eq!((&d0["scope"], &d1["scope"]), (&Value::from("run"), &Value::from("host")));
    assert!(d1.get("this_host").is_none());
    let u = |v: &Value| v.as_u64().unwrap();
    // the merged report is the sum of the two hosts'
    let (run, h0, h1) = (&d0["result"], &d0["this_host"], &d1["result"]);
    assert_eq!(run["ranks"].as_array().unwrap().len(), 2);
    assert_eq!(h1["ranks"][0]["gpus"], serde_json::json!([1, 2]));
    assert_eq!(u(&run["ops"]), u(&h0["ops"]) + u(&h1["ops"]));
    assert_eq!(u(&run["bytes_written"]), u(&h0["bytes_written"]) + u(&h1["bytes_written"]));
    assert_eq!(u(&run["latency"]["write"]["count"]), u(&h0["latency"]["write"]["count"]) + u(&h1["latency"]["write"]["count"]));
    assert_eq!(run["objects_created"], 2 * (1 + 2 + 1));
    assert_eq!(run["instances"].as_array().unwrap().len(), 2);
    assert_eq!(run["cold"].as_array().unwrap().len(), 2);
    let sum = u64::from_str_radix(h0["fingerprint"].as_str().unwrap(), 16).unwrap().wrapping_add(u64::from_str_radix(h1["fingerprint"].as_str().unwrap(), 16).unwrap());
    assert_eq!(run["fingerprint"], format!("{sum:016x}").as_str());
    // one verdict on every host: the run's fingerprint
    for d in [&d0, &d1] {
        assert_eq!(d["verdict"]["ok"], true);
        assert_eq!(d["verdict"]["fingerprint_scope"], "run");
        assert_eq!(d["verdict"]["fingerprint"], run["fingerprint"]);
    }
    std::fs::remove_dir_all(&root).unwrap();
}
