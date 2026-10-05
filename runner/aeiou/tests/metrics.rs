//! `aeiou dry-run --metrics` (`runner/REFERENCE.md` §10): the round-robin walk issues the same op
//! multiset as the inline walk, and the metrics recover the structure the abstracts put in.

use std::path::PathBuf;

use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Params};
use aeiou::metrics::{self, Opts};

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples")
}

fn dry(name: &str, gpus: i64, params: &[(&str, &str)], threads: usize, opts: Option<Opts>) -> dryrun::Report {
    let text = std::fs::read_to_string(examples().join(format!("{name}.ast.json"))).unwrap();
    let cfg = Config { seed: 1, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![] };
    let loaded = aeiou::load_str(&text).expect("valid abstract");
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let model = build_model(&loaded.ast, &cfg, &params).unwrap();
    dryrun::run_with(&model, threads, None, opts).unwrap()
}

/// The golden configurations of `tests/golden.rs`.
const CASES: &[(&str, i64, &[(&str, &str)])] = &[
    ("train_small_files", 2, &[("steps", "10")]),
    ("train_large_samples", 2, &[("steps", "5")]),
    ("ckpt_write_dcp", 2, &[("steps", "200")]),
    ("ckpt_restore", 2, &[]),
    ("model_load", 2, &[]),
    ("vdb_search_diskann", 1, &[("queries", "100"), ("threads", "2")]),
    ("vdb_search_ivf", 1, &[("calls", "100"), ("threads", "2")]),
    ("vdb_build_diskann", 1, &[("sectors", "1000"), ("shards", "2"), ("n", "1000000"), ("shard_index_bytes", "4194304"), ("index_bytes", "8388608"), ("sample_rows", "1000")]),
    ("kv_cache_serving", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")]),
    ("kv_cache_shared", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")]),
    ("kv_cache_shared_reader", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")]),
    ("train_stream_tfrecord", 2, &[("samples", "768"), ("per_shard", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2")]),
    ("train_stream_parquet", 2, &[("samples", "768"), ("per_shard", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2")]),
    ("train_map_hdf5", 2, &[("samples", "64"), ("per_file", "16"), ("batch", "4"), ("workers", "2"), ("steps", "4")]),
];

#[test]
fn the_round_robin_walk_issues_the_same_ops() {
    for (name, gpus, params) in CASES {
        let plain = dry(name, *gpus, params, 1, None);
        for (threads, opts) in [(1, Opts::default()), (4, Opts { block: 65536, sample: 3 })] {
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
            assert!(metrics::to_json(&m.total).is_some());
        }
    }
}

#[test]
fn metrics_do_not_depend_on_the_thread_count() {
    for (name, gpus, params) in CASES {
        let opts = Opts { block: 65536, sample: 1 };
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
    let r = dry("vdb_search_diskann", 1, &[("queries", "2000"), ("threads", "2"), ("nodes", "20000")], 2, Some(Opts::default()));
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
    let p = m.popularity_blocks();
    assert!(p.top_1_pct > 0.05, "top 1% of blocks: {}", p.top_1_pct);
    assert!(m.reuse[0][0].n > 0);
}

/// Small-file training: every file is read whole, front to back, and once per epoch.
#[test]
fn small_file_training_is_sequential_and_flat() {
    let r = dry("train_small_files", 2, &[("steps", "10")], 2, Some(Opts::default()));
    let m = r.total.metrics.as_ref().unwrap();
    let p = m.popularity_objects();
    assert_eq!(p.max, p.counts.last().unwrap().0, "every file has the same number of data ops");
    assert_eq!(m.reuse.iter().flatten().map(|h| h.n).sum::<u64>(), 0, "no block is touched twice within an epoch");
    assert!(m.fan_out.is_empty());
}

/// Sampling scales back to the exact totals within sampling error.
#[test]
fn sampling_estimates_the_exact_counts() {
    let params: &[(&str, &str)] = &[("queries", "2000"), ("threads", "2"), ("nodes", "1000000")];
    let exact = dry("vdb_search_diskann", 1, params, 2, Some(Opts::default()));
    let sampled = dry("vdb_search_diskann", 1, params, 2, Some(Opts { block: 4096, sample: 8 }));
    let (e, s) = (exact.total.metrics.as_ref().unwrap(), sampled.total.metrics.as_ref().unwrap());
    let (pe, ps) = (e.popularity_blocks(), s.popularity_blocks());
    let close = |a: u64, b: u64| (a as f64 - b as f64).abs() / (b as f64) < 0.10;
    assert!(close(ps.accesses, pe.accesses), "accesses {} vs {}", ps.accesses, pe.accesses);
    assert!(close(ps.distinct, pe.distinct), "distinct {} vs {}", ps.distinct, pe.distinct);
    assert!(close(s.first[0], e.first[0]), "first touches {} vs {}", s.first[0], e.first[0]);
    let (qe, qs) = (e.reuse[0][0].quantile(0.5) as f64, s.reuse[0][0].quantile(0.5) as f64);
    assert!((qs / qe - 1.0).abs() < 0.5, "median reuse distance {qs} vs {qe}");
}
