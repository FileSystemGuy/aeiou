//! Golden tests: the Rust loader agrees with `schema/check.py` on every committed AST, the
//! fingerprints of the nine workloads at fixed small configurations are pinned (a change
//! here is a change of the runner's definitions, `runner/README.md` §2), and the positional
//! semantics behave as specified.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Params};
use aeiou::vm::{run_actor, Op, OpCtx, OpKind, Sink};

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples")
}

fn config(gpus: i64, seed: u64, params: &[(&str, &str)]) -> Config {
    Config { seed, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![] }
}

fn dry(text: &str, cfg: &Config, threads: usize) -> dryrun::Report {
    let loaded = aeiou::load_str(text).expect("valid abstract");
    let params = Params::new(&loaded.ast, cfg).unwrap();
    let model = build_model(&loaded.ast, cfg, &params).unwrap();
    dryrun::run(&model, threads, None).unwrap()
}

#[test]
fn hashes_match_check_py() {
    // From `python3 schema/check.py` on 2026-09-30 (corpus sizes became parameters the same day; re-recorded for
    // contract 0.2 and model_load's `full` split the same day; all again 2026-10-01 for contract 0.3, the
    // optional `backend`; all again 2026-10-02 for contract 0.4, the optional `same_run` of a namespace, and
    // for contract 0.5, the wider `at` rule V3, with kv_cache_serving and kv_cache_shared changed under it).
    let want = [
        ("ckpt_restore", "ed1b8905fb5f96a3892c0e895f27c172fb06fa2571568ae4d362ff344ae547d0"),
        ("ckpt_write_dcp", "f67dbd42462fd8ce7d0d0d9c92312f49cf4567a5cdd36a240d0cf446e2ed88a9"),
        ("kv_cache_serving", "5b1ffc0c77ca06a85071c62ec8004148cfc2ec5ab6fcdcf4bb5a394eb40918ae"),
        ("model_load", "df63f75a22b063b44421e87c787d29d5d7f820cd4e96b0a8797876806ca496ae"),
        ("train_large_samples", "2fd904f665b70711797d9d588a94d5d660eabc68fe9a8b6ece876ae2a079c75a"),   // 2026-10-02: the `enumerate` phase, on by default
        ("train_small_files", "3f93512e6fe335abedd6e49421980af009984df1b35b7238b17c180a206cee65"),   // 2026-10-02: `enumerate` on by default
        ("vdb_build_diskann", "e4b6967cbd5ed044c2086754254ca3e6df274c0ee0c3cc70de4e8174774c1ec1"),
        ("vdb_search_diskann", "04099554a1a9f7c734c13269d0a91452e13e9f82b623ad07059a43d0bd9d3cf4"),
        ("train_stream_tfrecord", "76148083eaa787365ba7de7238d106d54a65a4993394c4f59458a633272e6472"),
        ("train_stream_parquet", "32c54f1a6cd676d9cce333f7571f71b7f6be3de31714ad95bce604a645d9db21"),
        ("train_map_hdf5", "48f8b14822e4c02f55d948e61d48bf93cdad84e8042580d7b23769edf9bd4f63"),
        ("vdb_search_ivf", "3abb616e11a0b57ba594737d9e3dc05fe5caa591232fbf9b34903c423a440327"),
        // the shared store and its cold reader, 2026-10-02 (DESIGN_REVIEW §3.52)
        ("kv_cache_shared", "e28da93d6e8822a0c9d94d044aa0893eddcd6c386f8e697a1563c1118b41f53c"),
        ("kv_cache_shared_reader", "b5bb10a041a8c4af8990bde3bdaf22a6dea4d77f67a774aa69e5cdc4ca0eef9c"),
    ];
    for (name, sha) in want {
        let loaded = aeiou::load(&examples().join(format!("{name}.ast.json"))).unwrap();
        assert_eq!(loaded.sha256, sha, "{name}");
        // the provenance block carries the same hash
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(examples().join(format!("{name}.ast.json"))).unwrap()).unwrap();
        assert_eq!(doc["provenance"]["ast_sha256"], sha, "{name}: provenance");
    }
}

#[test]
fn golden_fingerprints() {
    // (abstract, gpus, params, fingerprint, ops) at --seed 1, recorded 2026-09-30 (kv_cache_serving
    // re-recorded the same day after the conversation-directory `mkdir` was added to the abstract;
    // vdb_build_diskann likewise when its base file moved to `base/base.fbin`; ckpt_write_dcp when its
    // readback phase went behind `readback = false`; train_small_files on 2026-10-01 after the trace of
    // the real loader added the second `lseek` per file, DESIGN_REVIEW.md §3.43; train_large_samples the
    // same day, rewritten from the trace of `np.load`, §3.44; ckpt_write_dcp from the trace of
    // `dcp.save`, §3.45, and ckpt_restore with it: its default offsets follow the writer; ckpt_restore
    // again the same day, its item loop rewritten from the trace of `dcp.load`, §3.47; model_load with the calls of
    // `safetensors.safe_open` and the small JSON files, same section; vdb_search_ivf from the trace of FAISS
    // `OnDiskInvertedLists`, §3.49: the index file, the prefetch threads, ids before codes; vdb_search_diskann
    // 2026-10-02 from the trace of DiskANN's `PQFlashIndex`, §3.50: the load, the entry rounds, 24 to 30 more; vdb_build_diskann
    // the same day from the trace of `build_disk_index`, same section: the passes over the base, the shard files;
    // kv_cache_serving the same day from the trace of vLLM with LMCache, §3.51: no lookups on storage, one flat
    // directory, whole chunks of the prompt only, reads only for what the engine lost; the three KV-cache abstracts
    // again the same day from the replay of ShareGPT, §3.56: one `reuse` distance, `keep` in place of `local`,
    // `context`, and the measured lengths as defaults; the three again the same day, §3.57: one `keep` draw per
    // round of the open conversations, `kp @ (r − (r mod d + 1))` under the wider rule V3 of contract 0.5).
    let cases: &[(&str, i64, &[(&str, &str)], u64, u64)] = &[
        // the directory walk is on by default since 2026-10-02 (DESIGN_REVIEW §3.54): five ops per class directory and
        // actor in train_small_files (38,462 directories), four per directory in train_large_samples; with
        // `enumerate=false` both are what they were
        ("train_small_files", 2, &[("steps", "10")], 0xfb3215e5607b32ae, 389740),
        ("train_small_files", 2, &[("steps", "10"), ("enumerate", "false")], 0x71628bdb4289c4c8, 5120),
        ("train_large_samples", 2, &[("steps", "5")], 0xf755bd4187b75b78, 50491),
        ("train_large_samples", 2, &[("steps", "5"), ("enumerate", "false")], 0xa2c284c6137f9639, 50451),
        ("ckpt_write_dcp", 2, &[("steps", "200")], 0x850e8c9019b3bda8, 130),
        ("ckpt_restore", 2, &[], 0x22990a88db4e6d4f, 368),
        ("model_load", 2, &[], 0x2668387c7aa4126b, 24682),
        ("vdb_search_diskann", 1, &[("queries", "100"), ("threads", "2")], 0x4b25fcd827dbfa0f, 21970),
        ("vdb_search_ivf", 1, &[("calls", "100"), ("threads", "2")], 0x42c16b5296ebaefc, 25612),
        (
            "vdb_build_diskann",
            1,
            &[("sectors", "1000"), ("shards", "2"), ("n", "1000000"), ("shard_index_bytes", "4194304"), ("index_bytes", "8388608"), ("sample_rows", "1000")],
            0xeaf5da9b000203a1,
            1229207,
        ),
        ("kv_cache_serving", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")], 0x38aa9448f8ec6918, 1596),
        // the same request stream on LMCache's fs:// backend, and the cold engine on the store it leaves (2026-10-02, §3.52)
        ("kv_cache_shared", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")], 0x8e224a705f74b4c6, 3448),
        ("kv_cache_shared_reader", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")], 0x6f1f174a6c47109f, 3017),
        // the container workloads (contract 0.2, 2026-09-30), at builder/tests/test_formats.py's configurations
        ("train_stream_tfrecord", 2, &[("samples", "768"), ("per_shard", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2")], 0xd48ec6c021d88539, 263),
        ("train_stream_parquet", 2, &[("samples", "768"), ("per_shard", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2")], 0x74a8ab1574196e74, 28),
        ("train_map_hdf5", 2, &[("samples", "64"), ("per_file", "16"), ("batch", "4"), ("workers", "2"), ("steps", "4")], 0x3dd38374803bff38, 384),
    ];
    for (name, gpus, params, fp, ops) in cases {
        let text = std::fs::read_to_string(examples().join(format!("{name}.ast.json"))).unwrap();
        let cfg = config(*gpus, 1, params);
        let r1 = dry(&text, &cfg, 1);
        assert_eq!(r1.total.fingerprint, *fp, "{name}: fingerprint {:016x}", r1.total.fingerprint);
        assert_eq!(r1.total.total.ops, *ops, "{name}: ops");
        // independent of the thread count and of the instance order
        let r4 = dry(&text, &cfg, 4);
        assert_eq!(r4.total.fingerprint, *fp, "{name}: threads");
    }
}

#[test]
fn parameter_files_apply_over_defaults_and_under_overrides() {
    use aeiou::eval::ParamSet;
    let text = std::fs::read_to_string(examples().join("train_small_files.ast.json")).unwrap();
    let set = ParamSet::load(&examples().join("params/train_small_files.smoke.params.json")).unwrap();
    assert_eq!(set.abstract_name, "train_small_files");
    // the set (files 4000, steps 50) under --param steps=10 equals the same values given as overrides
    let mut cfg = config(2, 1, &[("steps", "10")]);
    cfg.sets = vec![set.clone()];
    let by_file = dry(&text, &cfg, 1);
    let by_cli = dry(&text, &config(2, 1, &[("files", "4000"), ("steps", "10")]), 1);
    assert_eq!(by_file.total.fingerprint, by_cli.total.fingerprint);
    assert_ne!(by_file.total.fingerprint, dry(&text, &config(2, 1, &[("steps", "10")]), 1).total.fingerprint, "files=4000 changes the permutation");
    // the rules (schema/README.md §8)
    let loaded = aeiou::load_str(&text).unwrap();
    let outcome = |json: &str| -> String {
        let set = ParamSet::parse(json, "x.params.json").unwrap();
        let cfg = Config { sets: vec![set], ..config(2, 1, &[]) };
        match Params::new(&loaded.ast, &cfg) {
            Ok(_) => "accepted".into(),
            Err(e) => format!("{e:#}"),
        }
    };
    assert!(outcome(r#"{"params_version":1,"abstract":"other","params":{}}"#).contains("for abstract `other`"));
    assert!(outcome(r#"{"params_version":1,"abstract":"train_small_files","params":{"nope":1}}"#).contains("no parameter `nope`"));
    assert!(outcome(r#"{"params_version":1,"abstract":"train_small_files","params":{"steps":[1]}}"#).contains("the default is a scalar, the value is an array"));
    assert!(outcome(r#"{"params_version":1,"abstract":"train_small_files","params":{"step_time":{"const":5}}}"#).contains("the value is a distribution"));
    assert!(outcome(r#"{"params_version":1,"abstract":"train_small_files","params":{"gpus":2}}"#).contains("--gpus"));
    assert_eq!(outcome(r#"{"params_version":1,"abstract":"train_small_files","params":{"enumerate":true,"hdr_read":4096}}"#), "accepted");
    assert!(ParamSet::parse(r#"{"params_version":2,"abstract":"a","params":{}}"#, "x").is_err());
    assert!(ParamSet::parse(r#"{"params_version":1,"abstract":"a","params":{},"extra":1}"#, "x").is_err());
    assert!(ParamSet::parse(r#"{"params_version":1,"abstract":"a"}"#, "x").is_err());
    // an AST hash pins a set to one shape
    let pinned = ParamSet::parse(&format!(r#"{{"params_version":1,"abstract":"train_small_files","ast_sha256":"{}","params":{{}}}}"#, "0".repeat(64)), "x").unwrap();
    let cfg = Config { sets: vec![pinned], ..config(2, 1, &[]) };
    assert!(format!("{:#}", cfg.check_sets("train_small_files", &loaded.sha256).unwrap_err()).contains("is for AST"));
    let pinned = ParamSet::parse(&format!(r#"{{"params_version":1,"abstract":"train_small_files","ast_sha256":"{}","params":{{}}}}"#, loaded.sha256), "x").unwrap();
    assert!(Config { sets: vec![pinned], ..config(2, 1, &[]) }.check_sets("train_small_files", &loaded.sha256).is_ok());
    // --param keeps the kind too
    assert!(Params::new(&loaded.ast, &config(2, 1, &[("steps", "[1]")])).is_err());
    // the committed synthetic tensor table drives model_load
    let text = std::fs::read_to_string(examples().join("model_load.ast.json")).unwrap();
    let set = ParamSet::load(&examples().join("params/model_load.synthetic.params.json")).unwrap();
    let mut cfg = config(2, 1, &[]);
    cfg.sets = vec![set];
    let r = dry(&text, &cfg, 1);
    assert!(r.total.total.ops > 2 * 21, "every tensor of the table is read: {} ops", r.total.total.ops);
}

#[test]
fn seed_changes_drawn_workloads_only() {
    let text = std::fs::read_to_string(examples().join("train_small_files.ast.json")).unwrap();
    let a = dry(&text, &config(2, 1, &[("steps", "10")]), 2).total.fingerprint;
    let b = dry(&text, &config(2, 2, &[("steps", "10")]), 2).total.fingerprint;
    assert_ne!(a, b, "the file order depends on the seed");
    // no draw anywhere: the seed is irrelevant, the parameters and the actor ids decide everything
    let text = std::fs::read_to_string(examples().join("ckpt_restore.ast.json")).unwrap();
    let a = dry(&text, &config(2, 1, &[]), 2).total.fingerprint;
    let b = dry(&text, &config(2, 2, &[]), 2).total.fingerprint;
    assert_eq!(a, b);
    // and the workload changes with G, by design
    let c = dry(&text, &config(3, 1, &[]), 2).total.fingerprint;
    assert_ne!(a, c);
}

#[derive(Default)]
struct Collect {
    ops: Vec<(i64, Vec<i64>, OpKind, String, i64, i64, i64)>,
}

impl<'m, 'a> Sink<'m, 'a> for Collect {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> anyhow::Result<()> {
        self.ops.push((ctx.actor, ctx.indices.to_vec(), op.kind, op.path.to_string(), op.offset, op.len, op.bytes));
        Ok(())
    }
}

fn collect(text: &str, cfg: &Config, template: &str, count: i64) -> Vec<(i64, Vec<i64>, OpKind, String, i64, i64, i64)> {
    let loaded = aeiou::load_str(text).expect("valid abstract");
    let params = Params::new(&loaded.ast, cfg).unwrap();
    let model = build_model(&loaded.ast, cfg, &params).unwrap();
    let mut all = Vec::new();
    for g in 0..count {
        let sink = run_actor(&model, template, g, count, Collect::default()).unwrap();
        all.extend(sink.ops);
    }
    all
}

const CONSUME: &str = r#"{
  "ast": "0.5", "name": "consume_test",
  "params": {"batches": {"default": 24}, "batch": {"default": 4}},
  "datasets": {"d": {"files": {"pattern": "d/{id}", "count": 100, "size": {"const": 10}, "seed": 7}}},
  "actors": {"gpu": {"body": [
    {"loader": {"name": "q", "index": "b", "workers": 2, "prefetch": 1, "batches": {"param": "batches"}, "body": [
      {"loop": {"index": "j", "to": {"param": "batch"}, "body": [
        {"let": {"name": "f", "value": {"consume": "d"}}},
        {"stat": {"file": {"ref": "f"}}}
      ]}}
    ]}},
    {"loop": {"index": "s", "to": {"param": "batches"}, "body": [{"take": {"channel": "q"}}]}}
  ]}}
}"#;

#[test]
fn consume_is_a_permutation_per_epoch_across_actors() {
    // N = 100, G = 3, B = 4: an epoch is 100 div 12 = 8 batches, 96 samples, 4 never drawn.
    let ops = collect(CONSUME, &config(3, 5, &[]), "gpu", 3);
    assert_eq!(ops.len(), 3 * 24 * 4);
    let mut by_epoch: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut per_epoch_counts: BTreeMap<i64, usize> = BTreeMap::new();
    for (_, idx, _, path, _, _, _) in &ops {
        let b = idx[0];
        let id: i64 = path.strip_prefix("d/").unwrap().parse().unwrap();
        by_epoch.entry(b / 8).or_default().insert(id);
        *per_epoch_counts.entry(b / 8).or_default() += 1;
    }
    assert_eq!(by_epoch.len(), 3);
    for (epoch, ids) in &by_epoch {
        assert_eq!(per_epoch_counts[epoch], 96, "epoch {epoch}: 3 × 8 × 4 draws");
        assert_eq!(ids.len(), 96, "epoch {epoch}: every draw is a distinct sample");
    }
    assert_ne!(by_epoch[&0], by_epoch[&1], "different permutation per epoch");
    // the same GPU draws the same files whatever the number of ranks: ids are global
    let again = collect(CONSUME, &config(3, 5, &[]), "gpu", 3);
    assert_eq!(ops, again);
}

const CHAIN: &str = r#"{
  "ast": "0.5", "name": "chain_test",
  "params": {"reuse": {"default": {"mixture": [{"weight": 0.4, "dist": null}, {"weight": 0.6, "dist": {"uniform": {"lo": 1, "hi": 5}}}]}}},
  "namespaces": {"kv": {"pattern": "kv/{conv:016x}/d{d}", "fields": {"conv": "int", "d": "int"}, "size": 4096, "seed": 3}},
  "actors": {"gpu": {"count": 1, "body": [
    {"loop": {"index": "r", "to": 200, "body": [
      {"let": {"name": "d", "value": {"draw": {"param": "reuse"}}}},
      {"let": {"name": "cont", "value": {"and": [{"ne": [{"ref": "d"}, null]}, {"le": [{"ref": "d"}, {"index": "r"}]}]}}},
      {"let": {"name": "conv", "value": {"cond": {"if": {"ref": "cont"},
          "then": {"at": {"ref": "conv", "index": {"sub": [{"index": "r"}, {"ref": "d"}]}}},
          "else": {"draw": {"uniform64": {}}}}}}},
      {"let": {"name": "dd", "value": {"cond": {"if": {"ref": "cont"}, "then": {"ref": "d"}, "else": 0}}}},
      {"stat": {"file": {"object": {"namespace": "kv", "fields": {"conv": {"ref": "conv"}, "d": {"ref": "dd"}}}}}}
    ]}}
  ]}}
}"#;

#[test]
fn at_chains_reach_the_original_conversation() {
    let ops = collect(CHAIN, &config(1, 11, &[]), "gpu", 1);
    assert_eq!(ops.len(), 200);
    let mut conv: Vec<u64> = Vec::new();
    let mut d: Vec<i64> = Vec::new();
    for (_, _, _, path, _, _, _) in &ops {
        let rest = path.strip_prefix("kv/").unwrap();
        let (c, dd) = rest.split_once("/d").unwrap();
        conv.push(u64::from_str_radix(c, 16).unwrap());
        d.push(dd.parse().unwrap());
    }
    let mut continued = 0;
    for r in 0..200usize {
        if d[r] > 0 {
            continued += 1;
            assert_eq!(conv[r], conv[r - d[r] as usize], "request {r} continues request {}", r - d[r] as usize);
        }
    }
    assert!(continued > 60, "{continued} continuations");
    let fresh: BTreeSet<u64> = conv.iter().copied().collect();
    assert!(fresh.len() < 200 && fresh.len() > 40, "{} distinct conversations", fresh.len());
}

const UNTIL_EOF: &str = r#"{
  "ast": "0.5", "name": "eof_test",
  "datasets": {"d": {"files": {"pattern": "d/{id}", "count": 4, "size": {"const": 2621440}, "seed": 7}}},
  "actors": {"gpu": {"count": 1, "body": [
    {"loop": {"index": "i", "to": 1, "body": [
      {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": 2}}}},
      {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
      {"read": {"file": {"ref": "f"}, "len": 1048576, "repeat": "until_eof"}},
      {"lseek": {"file": {"ref": "f"}, "offset": -100, "whence": "END"}},
      {"read": {"file": {"ref": "f"}, "len": 1000}},
      {"read": {"file": {"ref": "f"}, "len": 4096, "offset": 8}},
      {"close": {"file": {"ref": "f"}}}
    ]}}
  ]}}
}"#;

#[test]
fn until_eof_and_positions() {
    let ops = collect(UNTIL_EOF, &config(1, 0, &[]), "gpu", 1);
    let reads: Vec<(i64, i64, i64)> = ops.iter().filter(|o| o.2 == OpKind::Read).map(|o| (o.4, o.5, o.6)).collect();
    assert_eq!(
        reads,
        vec![
            (0, 1048576, 1048576),
            (1048576, 1048576, 1048576),
            (2097152, 1048576, 524288),
            (2621440, 1048576, 0),
            (2621340, 1000, 100), // after lseek END-100: a short read at the tail
            (8, 4096, 4096),      // positioned, position unchanged
        ]
    );
    assert!(ops.iter().all(|o| o.3 == "d/2"));
}

fn rejects(text: &str, needle: &str) {
    match aeiou::load_str(text) {
        Ok(_) => panic!("accepted an abstract that should fail with `{needle}`"),
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(msg.contains(needle), "wanted `{needle}` in:\n{msg}");
        }
    }
}

#[test]
fn validator_rejects_what_check_py_rejects() {
    let base = |datasets: &str, body: &str| {
        format!(
            r#"{{"ast": "0.5", "name": "t", "params": {{"n": {{"default": 3}}}},
                "datasets": {{{datasets}}},
                "actors": {{"gpu": {{"body": [{body}]}}}}}}"#
        )
    };
    let ds = r#""d": {"files": {"pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1}}"#;
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": {"param": "gpus"}, "body": []}}"#).replace(r#""n": {"default": 3}"#, r#""gpus": {"default": 3}"#), "reserved");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"let": {"name": "f", "value": {"consume": "d"}}}, {"write": {"file": {"ref": "f"}, "len": 1}}]}}"#), "read-only (V12)");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"open": {"file": {"file": {"dataset": "d", "id": 0}}, "flags": ["WRONLY"]}}]}}"#), "read-only (V12)");
    rejects(&base(r#""d": {"files": {"pattern": ".aeiou-x/{id}", "count": 10, "size": {"const": 1}, "seed": 1}}"#, ""), ".aeiou");
    rejects(&base(ds, r#"{"let": {"name": "f", "value": {"consume": "d"}}}"#), "outside any loop");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"stat": {"file": {"ref": "nope"}}}]}}"#), "not in scope");
    rejects(&base(ds, r#"{"compute": {"ns": {"param": "missing"}}}"#), "unknown param");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 4, "body": [{"let": {"name": "x", "value": {"cond": {"if": true, "then": {"at": {"ref": "x", "index": {"index": "i"}}}, "else": 1}}}}]}}"#), "provably >= 1");
    // the wider V3 of contract 0.5 (§3.57): the offset may be an expression provably >= 1, and the rule
    // covers every binding of the same loop body, not only the one being defined
    let accepts = |text: &str| aeiou::load_str(text).map(|_| ()).unwrap_or_else(|e| panic!("rejected a valid abstract:\n{e:#}"));
    let chain = |e: &str| format!(r#"{{"loop": {{"index": "i", "to": 4, "body": [{{"let": {{"name": "d", "value": {{"draw": {{"uniform": {{"lo": 1, "hi": 5}}}}}}}}}}, {{"let": {{"name": "x", "value": {{"cond": {{"if": true, "then": {{"at": {{"ref": "x", "index": {{"sub": [{{"index": "i"}}, {e}]}}}}}}, "else": 1}}}}}}}}]}}}}"#);
    accepts(&base(ds, &chain(r#"{"add": [{"mod": [{"index": "i"}, {"ref": "d"}]}, 1]}"#))); // the per-round draw: r − (r mod d + 1)
    accepts(&base(ds, &chain(r#"{"add": [{"index": "i"}, {"ref": "d"}]}"#))); // index >= 0 (no `from`) plus a draw >= 1
    accepts(&base(ds, &chain(r#"{"add": [{"add": [{"mod": [3, 2]}, 0]}, {"param": "n"}]}"#)).replace(r#""n": {"default": 3}"#, r#""n": {"default": {"uniform": {"lo": 1, "hi": 2}}}"#));
    rejects(&base(ds, &chain(r#"{"add": [{"mod": [{"index": "i"}, {"ref": "d"}]}, 0]}"#)), "provably >= 1"); // >= 0 only
    rejects(&base(ds, &chain(r#"{"add": [{"index": "i"}, {"index": "i"}]}"#)), "provably >= 1");
    rejects(&base(ds, &chain(r#"{"mul": [{"ref": "d"}, 2]}"#)), "provably >= 1"); // `mul` is not a form the rule knows
    rejects(&base(ds, &chain(r#"{"add": [{"draw": {"uniform": {"lo": 0, "hi": 5}}}, {"draw": {"uniform": {"lo": 0, "hi": 5}}}]}"#)), "provably >= 1");
    // an index whose loop starts below zero is not provably >= 0
    rejects(&base(ds, r#"{"loop": {"index": "j", "from": -2, "to": 4, "body": [{"loop": {"index": "i", "to": 4, "body": [{"let": {"name": "x", "value": {"cond": {"if": true, "then": {"at": {"ref": "x", "index": {"sub": [{"index": "i"}, {"add": [{"index": "j"}, 1]}]}}}, "else": 1}}}}]}}]}}"#), "provably >= 1");
    // a `let` of the same body defined earlier is under the rule too (a cycle through it would not end)
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 4, "body": [{"let": {"name": "y", "value": 1}}, {"let": {"name": "x", "value": {"at": {"ref": "y", "index": {"add": [{"index": "i"}, 1]}}}}}]}}"#), "binding of this loop body");
    rejects(&base(ds, r#"{"take": {"channel": "q"}}"#), "not declared");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"let": {"name": "f", "value": {"consume": "d"}}}, {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"], "expect": ["bogus"]}}]}}"#), "not an errno");
    // two datasets sharing a root (V13)
    rejects(
        &base(
            r#""a": {"files": {"pattern": "x/{id}", "count": 1, "size": {"const": 1}, "seed": 1}}, "b": {"files": {"pattern": "x/y{id}", "count": 1, "size": {"const": 1}, "seed": 2}}"#,
            "",
        ),
        "shares root",
    );
    // or one root inside another (V13, 2026-10-01): datagen wants each root to itself
    rejects(
        &base(
            r#""a": {"files": {"pattern": "x/{id}", "count": 1, "size": {"const": 1}, "seed": 1}}, "b": {"files": {"pattern": "x/y/{id}", "count": 1, "size": {"const": 1}, "seed": 2}}"#,
            "",
        ),
        "are nested",
    );
    // input namespaces are read-only and agree per root (V14)
    let ns = |input_a: &str, input_b: &str| {
        format!(
            r#""namespaces": {{"a": {{"pattern": "n/x{{i}}", "fields": {{"i": "int"}}, "size": 1, "seed": 1{input_a}}},
                           "b": {{"pattern": "n/y{{i}}", "fields": {{"i": "int"}}, "size": 1, "seed": 1{input_b}}}}},"#
        )
    };
    let with_ns = |ns: &str, body: &str| base(ds, body).replace(r#""actors""#, &format!("{ns} \"actors\""));
    rejects(&with_ns(&ns(r#", "input": true"#, ""), ""), "`input` differs (V14)");
    rejects(&with_ns(&ns(r#", "input": true"#, r#", "input": true"#), r#"{"open": {"file": {"object": {"namespace": "a", "fields": {"i": 1}}}, "flags": ["WRONLY", "CREAT"]}}"#), "read-only (V14)");
    rejects(&with_ns(&ns(r#", "input": true"#, r#", "input": true"#), r#"{"unlink": {"file": {"object": {"namespace": "a", "fields": {"i": 1}}}}}"#), "read-only (V14)");
    rejects(&with_ns(&ns(r#", "input": true"#, r#", "input": true"#), r#"{"mkdir": {"dir": {"object": {"namespace": "b", "fields": {"i": 1}}}}}"#), "read-only (V14)");
    // structural: an unknown node kind, an unknown field
    rejects(&base(ds, r#"{"frobnicate": {}}"#), "unknown variant");
    rejects(&base(ds, r#"{"compute": {"ns": 1, "bogus": 2}}"#), "unknown field");
}

/// Rule V3 is judged on the document's defaults; a `--param` may replace a distribution with
/// one whose minimum is lower, and an `at` offset of zero would be a loop in the VM. The
/// parameters in effect are held to the same rule (§3.57), before anything is evaluated.
#[test]
fn parameters_given_are_held_to_the_at_rule() {
    let text = std::fs::read_to_string(examples().join("kv_cache_serving.ast.json")).unwrap();
    let loaded = aeiou::load_str(&text).unwrap();
    let small = [("concurrency", "1"), ("warm", "0"), ("requests", "20")];
    let with = |reuse: &str| {
        let mut p = small.to_vec();
        p.push(("reuse", reuse));
        config(1, 1, &p)
    };
    assert!(Params::new(&loaded.ast, &with(r#"{"const": 3}"#)).is_ok());
    assert!(Params::new(&loaded.ast, &with(r#"{"uniform": {"lo": 1, "hi": 4}}"#)).is_ok());
    for bad in [r#"{"const": 0}"#, r#"{"uniform": {"lo": 0, "hi": 4}}"#, r#"{"mixture": [{"weight": 0.5, "dist": null}, {"weight": 0.5, "dist": {"const": 0}}]}"#] {
        let e = Params::new(&loaded.ast, &with(bad)).err().map(|e| format!("{e:#}")).unwrap_or_else(|| panic!("accepted reuse={bad}"));
        assert!(e.contains("with the parameters given") && e.contains("provably >= 1"), "{e}");
    }
}

/// A chain that steps back one index per iteration (`x @ k-1`, the restore's buffer start)
/// is one link per iteration: the values already evaluated are kept, so a long loop neither
/// walks to its start every time nor recurses as deep as it is long.
#[test]
fn a_long_at_chain_is_one_link_per_iteration() {
    let text = r#"{
  "ast": "0.5", "name": "long_chain",
  "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": 1, "seed": 3}},
  "actors": {"gpu": {"count": 1, "body": [
    {"loop": {"index": "k", "to": 200000, "body": [
      {"let": {"name": "acc", "value": {"cond": {"if": {"eq": [{"index": "k"}, 0]}, "then": 0,
          "else": {"add": [{"at": {"ref": "acc", "index": {"sub": [{"index": "k"}, 1]}}}, {"mod": [{"index": "k"}, 3]}]}}}}},
      {"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"ref": "acc"}}}}}}
    ]}}
  ]}}
}"#;
    let ops = collect(text, &config(1, 1, &[]), "gpu", 1);
    assert_eq!(ops.len(), 200_000);
    assert_eq!(ops[0].3, "o/0");
    assert_eq!(ops[4].3, "o/4");                // 0 + 1 + 2 + 0 + 1
    assert_eq!(ops[199_999].3, "o/199999");     // 66,666 full periods of 0 + 1 + 2, then 0 + 1
}
