//! `aeiou dry-run --metrics` (`runner/REFERENCE.md` §10): the round-robin walk issues the same op
//! multiset as the inline walk, and the metrics recover the structure the abstracts put in.


use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Params};
use aeiou::metrics::{self, Opts};

fn dry(name: &str, gpus: i64, params: &[(&str, &str)], threads: usize, opts: Option<Opts>) -> dryrun::Report {
    let text = std::fs::read_to_string(crate::common::workload_ast(name)).unwrap();
    // the LMCache abstracts at the values their defaults held until 2026-10-23 (their unsettled cases chat-64-slots)
    let params = if name.starts_with("lmcache") { crate::common::lmcache(params) } else { params.to_vec() };
    let cfg = Config { seed: 1, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![], st_blksize: Default::default() };
    let loaded = aeiou::load_str(&text).expect("valid abstract");
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let model = build_model(&loaded.ast, &cfg, &params).unwrap();
    dryrun::run_with(&model, threads, None, opts).unwrap()
}

/// The golden configurations of `tests/golden.rs`.
const CASES: &[(&str, i64, &[(&str, &str)])] = &[
    ("file_per_sample", 2, &[("batch", "32"), ("workers", "8"), ("prefetch", "2"), ("step_time", "105000000"), ("samples", "50000000"), ("sample_median", "112640"), ("steps", "10")]),
    ("npy_per_sample", 2, &[("batch", "7"), ("workers", "4"), ("prefetch", "2"), ("step_time", "323000000"), ("samples", "50000"), ("sample_mean", "146800640"), ("sample_sd", "4194304"), ("steps", "5")]),
    ("dcp_save", 2, &[("steps", "200"), ("step_time", "323000000"), ("ckpt_every", "100"), ("items", "4"), ("item_bytes", "[16777216, 33554432, 16777216, 8388608]"), ("meta_bytes", "2097152")]),
    ("dcp_load", 2, &crate::common::DCP_LOAD),
    ("safetensors_load", 2, &[]),
    ("diskann_search", 1, &[("queries", "100"), ("threads", "2")]),
    ("faiss_ivf_ondisk", 1, &[("calls", "100"), ("threads", "2")]),
    ("diskann_build", 1, &[("sectors", "1000"), ("shards", "2"), ("n", "1000000"), ("shard_index_bytes", "4194304"), ("index_bytes", "8388608"), ("sample_rows", "1000")]),
    ("lmcache_local_disk", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")]),
    ("lmcache_shared_fs", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")]),
    ("lmcache_shared_fs_reader", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")]),
    ("tfrecord_stream", 2, &[("samples", "768"), ("samples_per_file", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2"), ("prefetch", "1"), ("step_time", "105000000"), ("decode", "0"), ("sample_median", "112640")]),
    ("parquet_stream", 2, &[("samples", "768"), ("samples_per_file", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2"), ("prefetch", "1"), ("step_time", "105000000"), ("decode", "0"), ("sample_median", "112640")]),
    ("hdf5_map", 2, &[("samples", "64"), ("samples_per_file", "16"), ("batch", "4"), ("workers", "2"), ("steps", "4"), ("prefetch", "2"), ("step_time", "105000000")]),
];

#[test]
fn the_round_robin_walk_issues_the_same_ops() {
    for (name, gpus, params) in CASES {
        let plain = dry(name, *gpus, params, 1, None);
        for (threads, opts) in [(1, Opts::default()), (4, Opts { block: Some(65536) })] {
            let m = dry(name, *gpus, params, threads, Some(opts));
            assert_eq!(m.total.fingerprint, plain.total.fingerprint, "{name}: fingerprint");
            assert_eq!(m.total.total.counts, plain.total.total.counts, "{name}: counts");
            assert_eq!(m.total.total.bytes_read, plain.total.total.bytes_read, "{name}: bytes read");
            assert_eq!(m.total.total.bytes_written, plain.total.total.bytes_written, "{name}: bytes written");
            assert_eq!((m.total.compute_ns, m.total.barriers, m.total.takes, m.total.puts), (plain.total.compute_ns, plain.total.barriers, plain.total.takes, plain.total.puts), "{name}: controls");
            let met = m.total.metrics.as_ref().expect("metrics");
            // every data op has a request size; every transferred block is a first touch or a reuse
            let data: u64 = [aeiou::vm::OpKind::Read, aeiou::vm::OpKind::Write].iter().map(|k| plain.total.total.counts.get(k).copied().unwrap_or(0)).sum();
            assert_eq!(met.request[0].n + met.request[1].n, data, "{name}: request sizes");
            // run lengths account for every byte transferred
            let run_bytes = (met.runs[0].bytes.sum + met.runs[1].bytes.sum) as u64;
            assert_eq!(run_bytes, plain.total.total.bytes_read + plain.total.total.bytes_written, "{name}: bytes in runs");
            // every byte transferred is a first touch or in one group of the extent reuse; every
            // byte read again is in a whole or a partial re-read
            let ext = &met.extents;
            let again: u64 = ext.reuse.iter().flatten().map(|h| h.n).sum();
            assert_eq!(ext.first[0] + ext.first[1] + again, run_bytes, "{name}: bytes in extents");
            assert_eq!(ext.whole + ext.partial, ext.reuse[0][0].n + ext.reuse[1][0].n, "{name}: re-read bytes");
            assert_eq!(met.popularity_bytes().accesses, run_bytes, "{name}: byte accesses");
            assert!(metrics::to_json(&m.total).is_some());
        }
    }
}

#[test]
fn metrics_do_not_depend_on_the_thread_count() {
    for (name, gpus, params) in CASES {
        let opts = Opts { block: Some(65536) };
        let a = serde_json::to_string(&metrics::to_json(&dry(name, *gpus, params, 1, Some(opts)).total)).unwrap();
        let b = serde_json::to_string(&metrics::to_json(&dry(name, *gpus, params, 4, Some(opts)).total)).unwrap();
        assert_eq!(a, b, "{name}");
    }
}

/// DiskANN search: the depth distribution is the `hops` parameter's plus the two rounds near
/// the entry points, the fan-out is `beam` (and 1 for the entry point's sector), no sector
/// read continues another, and the entry points show as reuse and as skew.
#[test]
fn diskann_structure_is_recovered() {
    let r = dry("diskann_search", 1, &[("queries", "2000"), ("threads", "2"), ("nodes", "20000")], 2, Some(Opts::default()));
    let m = r.total.metrics.as_ref().unwrap();
    assert_eq!(m.depth.keys().copied().collect::<Vec<_>>(), vec![1, 26, 27, 28, 29, 30, 31, 32]); // 1: the thread fork
    let chains: u64 = m.depth.iter().filter(|(k, _)| **k >= 26).map(|(_, v)| v).sum();
    assert_eq!(chains, 2 * 2000);
    let share = |d: u64| m.depth[&d] as f64 / chains as f64;
    for (d, want) in [(27, 0.135), (28, 0.537), (29, 0.269), (30, 0.047)] {
        assert!((share(d) - want).abs() < 0.03, "depth {d}: {}", share(d));
    }
    assert_eq!(m.fan_out.keys().copied().collect::<Vec<_>>(), vec![1, 2, 4]);
    assert_eq!(m.runs[0].ops.max, 2); // the two stream reads of the PQ file at load
    assert_eq!(m.runs[0].multi_op_bytes, 20000 * 5 * 32 + 8);
    // the first round of every query reads an entry point's sector and the second lands on the entry
    // points' neighbours: at 20,000 sectors both sets are one sector, which takes over 4 %
    // of the accesses, where the uniform rounds alone would give the top 1 % about 1.5 %
    let p = m.popularity_bytes();
    assert!(p.top_1_pct > 0.05, "top 1% of bytes: {}", p.top_1_pct);
    assert!(m.extents.reuse[0][0].n > 0);
}

/// Small-file training: every file is read whole, front to back, and once per epoch.
#[test]
fn small_file_training_is_sequential_and_flat() {
    let r = dry("file_per_sample", 2, &[("batch", "32"), ("workers", "8"), ("prefetch", "2"), ("step_time", "105000000"), ("samples", "50000000"), ("sample_median", "112640"), ("steps", "10")], 2, Some(Opts::default()));
    let m = r.total.metrics.as_ref().unwrap();
    let p = m.popularity_objects();
    assert_eq!(p.max, p.counts.last().unwrap().0, "every file has the same number of data ops");
    assert_eq!(m.extents.reuse.iter().flatten().map(|h| h.n).sum::<u64>(), 0, "no byte is touched twice within an epoch");
    assert!(m.fan_out.is_empty());
}

/// Extents (DESIGN_REVIEW §3.79): one file of 1 MiB read in four fills of 256 KiB, its first
/// fill read again alone, then all of it again. Every number is the definition's, by hand:
/// the alone read re-reads part of the first pass at a distance of the three fills after it
/// plus its own (1 MiB), a partial re-read; the last pass re-reads that read at 256 KiB and the
/// first pass's other three fills at 1 MiB each, both extents whole.
#[test]
fn extents_by_hand() {
    use serde_json::{json, Value};
    let ast = json!({
        "ast": "0.13", "name": "t",
        "datasets": {"d": {"layout": "whole", "access": "indexed", "pattern": "d/f{id}", "count": 1, "size": {"const": 1048576}, "seed": 1}},
        "actors": {"gpu": {"body": [
            {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": 0}}}},
            {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
            {"read": {"file": {"ref": "f"}, "len": 262144, "repeat": 4}},
            {"close": {"file": {"ref": "f"}}},
            {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
            {"read": {"file": {"ref": "f"}, "len": 262144}},
            {"close": {"file": {"ref": "f"}}},
            {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
            {"read": {"file": {"ref": "f"}, "len": 262144, "repeat": 4}},
            {"close": {"file": {"ref": "f"}}}]}}
    });
    let dir = crate::common::tmpdir("metrics", "hand");
    let (path, out) = (dir.join("t.ast.json"), dir.join("m.json"));
    std::fs::write(&path, ast.to_string()).unwrap();
    let run = |more: &[&str]| -> Value {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou"))
            .args(["dry-run", path.to_str().unwrap(), "--gpus", "1", "--metrics-json", out.to_str().unwrap()])
            .args(more)
            .env_remove("AEIOU_CONFIG")
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap()
    };
    let d = run(&[]);
    assert_eq!((&d["aeiou_metrics"], &d["block"]), (&json!(2), &Value::Null));
    let t = &d["total"];
    assert!(t.get("reuse_distance_bytes").is_none() && t.get("popularity_blocks").is_none(), "no block sections unasked: {t}");
    let r = &t["reuse_extents"];
    let (k, m) = (1u64 << 10, 1u64 << 20);
    assert_eq!(r["first_touch"], json!({"read": m, "write": 0}));
    assert_eq!(r["read_after_read"]["n"], json!(m + 256 * k));
    assert_eq!(r["read_after_read"]["buckets"], json!([[256 * k, 256 * k], [m, m]]));
    assert_eq!((&r["read_after_write"]["n"], &r["write_after_read"]["n"], &r["write_after_write"]["n"]), (&json!(0), &json!(0), &json!(0)));
    assert_eq!((&r["whole"], &r["partial"]), (&json!(m), &json!(256 * k)));
    let p = &t["popularity_bytes"];
    assert_eq!(p["counts"], json!([[3, 256 * k], [2, 768 * k]]));
    assert_eq!((&p["distinct"], &p["accesses"], &p["max"]), (&json!(m), &json!(3 * 256 * k + 2 * 768 * k), &json!(3)));
    // the block sections when asked, as version 1 had them: 256 blocks touched three times, 768 twice
    let b = run(&["--metrics-block", "4096"]);
    assert_eq!(b["block"], json!(4096));
    assert_eq!(b["total"]["popularity_blocks"]["counts"], json!([[3, 64], [2, 192]]));
    assert_eq!(b["total"]["reuse_distance_bytes"]["first_touch"]["read"], json!(256));
    assert_eq!(b["total"]["reuse_extents"], d["total"]["reuse_extents"], "the extents do not depend on the block");
    // --metrics-sample is gone
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["dry-run", path.to_str().unwrap(), "--gpus", "1", "--metrics", "--metrics-sample", "4"]).output().unwrap();
    assert_eq!(o.status.code(), Some(2), "{}", String::from_utf8_lossy(&o.stderr));
}
