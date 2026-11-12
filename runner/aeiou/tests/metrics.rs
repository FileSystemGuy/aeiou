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
    ("safetensors_load", 2, &crate::common::SAFETENSORS),
    ("diskann_search", 1, &[("n", "999999995"), ("dim", "128"), ("degree", "64"), ("pq_code", "32"), ("beam", "4"), ("hops", r#"{"empirical": {"values": [24, 25, 26, 27, 28, 29, 30], "weights": [1, 135, 536, 269, 47, 9, 2]}}"#), ("entry", r#"{"hotset": {"fraction": 2e-7, "weight": 1.0}}"#), ("near", r#"{"hotset": {"fraction": 1e-5, "weight": 1.0}}"#), ("node_cache", "0"), ("rerank", "500000"), ("queries", "100"), ("threads", "2")]),
    ("faiss_ivf_ondisk", 1, &[("threads", "2"), ("calls", "100"), ("batch", "1"), ("nprobe", "64"), ("list_pop", r#"{"zipf": {"s": 0.2}}"#), ("distance", "400000"), ("lists", "1000000"), ("list_bytes", r#"{"lognormal": {"median": 40960, "sigma": 0.4}}"#), ("code_size", "32"), ("index_bytes", "553779476"), ("index_reads", "[8192, 536862720, 8192, 122880, 8192, 16769024, 8192]")]),
    ("diskann_build", 1, &[("dim", "128"), ("degree", "64"), ("pq_code", "32"), ("shards", "2"), ("n", "1000000"), ("shard_index_bytes", "4194304"), ("index_bytes", "8388608"), ("sample_rows", "1000"), ("pq_train", "60000000000"), ("partition_time", "90000000000"), ("build_time", "1800000000000")]),
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
    let r = dry("diskann_search", 1, &crate::common::diskann_search(&[("queries", "2000"), ("threads", "2"), ("n", "99995")]), 2, Some(Opts::default()));
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
    assert_eq!(m.runs[0].multi_op_bytes, 99995 * 32 + 8);    // the PQ file: its points' codes and its header
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
        "ast": "0.15", "name": "t",
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

/// What a path holds (`FS_API_SWEEP.md` S-18, the user's choices 2026-11-03): an `O_TRUNC`
/// open and an `unlink` end a name's history, a `rename` moves it to the new name, a truncate
/// ends the bytes past its length and a punched hole the bytes in it; an object is the data,
/// not the name. One abstract's calls, as its dry run, its emitted trace file, and an strace of
/// the same calls, each to the numbers by hand (K = 4 KiB): the bytes first touched are the
/// writes to `o/1` twice (the second after its `O_TRUNC`), `o/2` once (after its `unlink`),
/// `o/3` 2K then K (past its `ftruncate`), and the read of `o/3`'s hole; the reads again what
/// was written are `o/1` once, `o/2` once (what `o/1` held, renamed), `o/3`'s second K; the one
/// write again is `o/3`'s first K. Four objects.
#[test]
fn what_a_path_holds_ends_and_moves_its_history() {
    use serde_json::{json, Value};
    let k = 4096u64;
    let obj = |v: u64| json!({"object": {"namespace": "o", "fields": {"v": v}}});
    let open = |v: u64, flags: &[&str]| json!({"open": {"file": obj(v), "flags": flags}});
    let close = |v: u64| json!({"close": {"file": obj(v)}});
    let io = |op: &str, v: u64, len: u64| json!({op: {"file": obj(v), "len": len, "offset": 0}});
    let ast = json!({
        "ast": "0.15", "name": "t",
        "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": "as_written", "seed": 3}},
        "actors": {"gpu": {"body": [
            open(1, &["WRONLY", "CREAT"]), io("write", 1, k), close(1),
            open(1, &["RDONLY"]), io("read", 1, k), close(1),
            open(1, &["WRONLY", "TRUNC"]), io("write", 1, k), close(1),
            {"rename": {"from": obj(1), "to": obj(2)}},
            open(2, &["RDONLY"]), io("read", 2, k), close(2),
            {"unlink": {"file": obj(2)}},
            open(2, &["WRONLY", "CREAT"]), io("write", 2, k), close(2),
            open(3, &["RDWR", "CREAT"]), io("write", 3, 2 * k),
            {"ftruncate": {"file": obj(3), "len": k}},
            io("write", 3, 2 * k), close(3),
            open(3, &["RDWR"]),
            {"fallocate": {"file": obj(3), "offset": 0, "len": k, "mode": ["KEEP_SIZE", "PUNCH_HOLE"]}},
            io("read", 3, 2 * k), close(3)]}}
    });
    let strace = [
        r#"openat(AT_FDCWD</d>, "/d/o/1", O_WRONLY|O_CREAT, 0644) = 3</d/o/1>"#,
        r#"pwrite64(3</d/o/1>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/o/1>) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/o/1", O_RDONLY) = 3</d/o/1>"#,
        r#"pread64(3</d/o/1>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/o/1>) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/o/1", O_WRONLY|O_TRUNC) = 3</d/o/1>"#,
        r#"pwrite64(3</d/o/1>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/o/1>) = 0"#,
        r#"rename("/d/o/1", "/d/o/2") = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/o/2", O_RDONLY) = 3</d/o/2>"#,
        r#"pread64(3</d/o/2>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/o/2>) = 0"#,
        r#"unlink("/d/o/2") = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/o/2", O_WRONLY|O_CREAT, 0644) = 3</d/o/2>"#,
        r#"pwrite64(3</d/o/2>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/o/2>) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/o/3", O_RDWR|O_CREAT, 0644) = 3</d/o/3>"#,
        r#"pwrite64(3</d/o/3>, "x"..., 8192, 0) = 8192"#,
        r#"ftruncate(3</d/o/3>, 4096) = 0"#,
        r#"pwrite64(3</d/o/3>, "x"..., 8192, 0) = 8192"#,
        r#"close(3</d/o/3>) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/o/3", O_RDWR) = 3</d/o/3>"#,
        r#"fallocate(3</d/o/3>, FALLOC_FL_KEEP_SIZE|FALLOC_FL_PUNCH_HOLE, 0, 4096) = 0"#,
        r#"pread64(3</d/o/3>, "x"..., 8192, 0) = 8192"#,
        r#"close(3</d/o/3>) = 0"#,
    ];
    let dir = crate::common::tmpdir("metrics", "holds");
    let (path, out, emitted) = (dir.join("t.ast.json"), dir.join("m.json"), dir.join("t.jsonl"));
    std::fs::write(&path, ast.to_string()).unwrap();
    let st = dir.join("t.strace");
    let lines: Vec<String> = strace.iter().enumerate().map(|(i, l)| format!("100 1700000000.{:06} {l} <0.000010>", i * 100)).collect();
    std::fs::write(&st, lines.join("\n") + "\n").unwrap();
    let ok = |o: std::process::Output| {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        o
    };
    ok(std::process::Command::new(env!("CARGO_BIN_EXE_aeiou"))
        .args(["dry-run", path.to_str().unwrap(), "--gpus", "1", "--metrics-json", out.to_str().unwrap(), "--emit-trace", emitted.to_str().unwrap()])
        .env_remove("AEIOU_CONFIG")
        .output()
        .unwrap());
    let read = |p: &std::path::Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
    let trace = |input: &std::path::Path, root: &[&str]| -> Value {
        let o = ok(std::process::Command::new(env!("CARGO_BIN_EXE_aeiou-trace")).arg("metrics").arg(input).args(root).output().unwrap());
        serde_json::from_slice(&o.stdout).unwrap()
    };
    // the emitted file as a `trace` node, its dry run's numbers
    let text = std::fs::read(&emitted).unwrap();
    let sha: String = <sha2::Sha256 as sha2::Digest>::digest(&text).iter().map(|b| format!("{b:02x}")).collect();
    let node = dir.join("node.ast.json");
    let node_ast = json!({"ast": "0.15", "name": "node", "params": {}, "datasets": {}, "actors": {"app": {"count": 1, "body": [{"trace": {"file": "t.jsonl", "sha256": sha}}]}}});
    std::fs::write(&node, node_ast.to_string()).unwrap();
    let dry = |ast: &std::path::Path, to: &std::path::Path, more: &[&str]| {
        ok(std::process::Command::new(env!("CARGO_BIN_EXE_aeiou"))
            .args(["dry-run", ast.to_str().unwrap(), "--gpus", "1", "--metrics-json", to.to_str().unwrap()])
            .args(more)
            .env_remove("AEIOU_CONFIG")
            .output()
            .unwrap());
        read(to)
    };
    let as_node = dry(&node, &dir.join("node.json"), &[]);
    let docs = [("dry run", read(&out)), ("trace node", as_node), ("trace file", trace(&emitted, &[])), ("strace", trace(&st, &["--root", "/d"]))];
    for (what, d) in &docs {
        let r = &d["total"]["reuse_extents"];
        assert_eq!(r["first_touch"], json!({"read": k, "write": 6 * k}), "{what}: {r}");
        assert_eq!((&r["read_after_write"]["n"], &r["write_after_write"]["n"]), (&json!(3 * k), &json!(k)), "{what}: {r}");
        assert_eq!((&r["read_after_read"]["n"], &r["write_after_read"]["n"]), (&json!(0), &json!(0)), "{what}: {r}");
        assert_eq!(d["total"]["popularity_objects"]["distinct"], json!(4), "{what}: {}", d["total"]["popularity_objects"]);
        for key in ["reuse_extents", "popularity_bytes", "popularity_objects"] {
            assert_eq!(d["total"][key], docs[0].1["total"][key], "{what}: {key}");
        }
    }
    // the blocks of 4 KiB: the same, the hole's block moved by the punch
    let b = dry(&path, &dir.join("b.json"), &["--metrics-block", "4096"]);
    assert_eq!(b["total"]["reuse_distance_bytes"]["first_touch"], json!({"read": 1, "write": 6}), "{}", b["total"]["reuse_distance_bytes"]);
    assert_eq!(trace(&st, &["--root", "/d", "--metrics-block", "4096"])["total"]["reuse_distance_bytes"], b["total"]["reuse_distance_bytes"]);
}

/// The calls only a trace has: a directory renamed moves the history of every path under it,
/// `RENAME_EXCHANGE` trades two names' histories, a failed `unlink` ends nothing, a rename from
/// outside the root ends the history of the name it replaces, a collapsed range ends every
/// byte from its offset on, an open holds its file after its name is unlinked, and a length
/// the trace does not show ends nothing.
#[test]
fn a_traces_renames_move_history_and_failed_calls_change_nothing() {
    use serde_json::{json, Value};
    let strace = [
        r#"openat(AT_FDCWD</d>, "/d/x/f", O_WRONLY|O_CREAT, 0644) = 3</d/x/f>"#,
        r#"pwrite64(3</d/x/f>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/x/f>) = 0"#,
        r#"rename("/d/x", "/d/y") = 0"#,
        r#"unlink("/d/y/f") = -1 EACCES (Permission denied)"#,
        r#"openat(AT_FDCWD</d>, "/d/y/f", O_RDONLY) = 3</d/y/f>"#,
        r#"pread64(3</d/y/f>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/y/f>) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/g", O_WRONLY|O_CREAT, 0644) = 3</d/g>"#,
        r#"pwrite64(3</d/g>, "x"..., 8192, 0) = 8192"#,
        r#"close(3</d/g>) = 0"#,
        r#"renameat2(AT_FDCWD</d>, "/d/g", AT_FDCWD</d>, "/d/y/f", RENAME_EXCHANGE) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/g", O_RDWR) = 3</d/g>"#,
        r#"pread64(3</d/g>, "x"..., 4096, 0) = 4096"#,
        r#"fallocate(3</d/g>, FALLOC_FL_COLLAPSE_RANGE, 1024, 1024) = 0"#,
        r#"pread64(3</d/g>, "x"..., 3072, 0) = 3072"#,
        r#"close(3</d/g>) = 0"#,
        r#"rename("/tmp/z", "/d/y/f") = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/y/f", O_RDONLY) = 3</d/y/f>"#,
        r#"pread64(3</d/y/f>, "x"..., 8192, 0) = 8192"#,
        r#"close(3</d/y/f>) = 0"#,
        r#"openat(AT_FDCWD</d>, "/d/u", O_RDWR|O_CREAT, 0644) = 3</d/u>"#,
        r#"pwrite64(3</d/u>, "x"..., 4096, 0) = 4096"#,
        r#"unlink("/d/u") = 0"#,
        r#"pread64(3</d/u>, "x"..., 4096, 0) = 4096"#,
        r#"ftruncate(3</d/u>, junk) = 0"#,
        r#"pread64(3</d/u>, "x"..., 4096, 0) = 4096"#,
        r#"close(3</d/u>) = 0"#,
    ];
    let dir = crate::common::tmpdir("metrics", "renames");
    let st = dir.join("t.strace");
    let lines: Vec<String> = strace.iter().enumerate().map(|(i, l)| format!("100 1700000000.{:06} {l} <0.000010>", i * 100)).collect();
    std::fs::write(&st, lines.join("\n") + "\n").unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou-trace")).args(["metrics", st.to_str().unwrap(), "--root", "/d"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let d: Value = serde_json::from_slice(&o.stdout).unwrap();
    let r = &d["total"]["reuse_extents"];
    // written: x/f 4K, g 8K, u 4K; read again what was written: y/f's 4K (x/f's, its directory
    // renamed), u's 4K (unlinked, read through its open); read again what was read: g's 4K
    // (y/f's, traded), g's first 1K after the collapse, u's 4K after the ftruncate of no
    // length; read first: g's 2K after the collapse, and y/f's 8K (from outside the root)
    assert_eq!(r["first_touch"], json!({"read": 2048 + 8192, "write": 4096 + 8192 + 4096}), "{r}");
    assert_eq!(r["read_after_write"]["n"], json!(4096 + 4096), "{r}");
    assert_eq!(r["read_after_read"]["n"], json!(4096 + 1024 + 4096), "{r}");
}

/// Under `--metrics-block`, a truncate ends the blocks wholly past its length, and the last
/// block the data ended inside: a file of 6000 bytes truncated to 4096 and written to 6000
/// again touches its second 4 KiB block for the first time twice, its first never.
#[test]
fn a_truncate_ends_the_block_the_data_ended_inside() {
    use serde_json::{json, Value};
    let strace = [
        r#"openat(AT_FDCWD</d>, "/d/f", O_RDWR|O_CREAT, 0644) = 3</d/f>"#,
        r#"pwrite64(3</d/f>, "x"..., 6000, 0) = 6000"#,
        r#"ftruncate(3</d/f>, 4096) = 0"#,
        r#"pwrite64(3</d/f>, "x"..., 1904, 4096) = 1904"#,
        r#"close(3</d/f>) = 0"#,
    ];
    let dir = crate::common::tmpdir("metrics", "blocks");
    let st = dir.join("t.strace");
    let lines: Vec<String> = strace.iter().enumerate().map(|(i, l)| format!("100 1700000000.{:06} {l} <0.000010>", i * 100)).collect();
    std::fs::write(&st, lines.join("\n") + "\n").unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou-trace")).args(["metrics", st.to_str().unwrap(), "--root", "/d", "--metrics-block", "4096"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let d: Value = serde_json::from_slice(&o.stdout).unwrap();
    let r = &d["total"]["reuse_distance_bytes"];
    assert_eq!(r["first_touch"], json!({"read": 0, "write": 3}), "{r}");
    assert_eq!(r["write_after_write"]["n"], json!(0), "{r}");
}
