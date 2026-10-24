//! Golden tests: the Rust loader agrees with `schema/check.py` on every committed AST, the
//! fingerprints of the nine workloads at fixed small configurations are pinned (a change
//! here is a change of the runner's definitions, `runner/REFERENCE.md` §2), and the positional
//! semantics behave as specified.

use std::collections::{BTreeMap, BTreeSet};

use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Params};
use aeiou::vm::{run_actor, Op, OpCtx, OpKind, Sink};

fn config(gpus: i64, seed: u64, params: &[(&str, &str)]) -> Config {
    Config { seed, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![], st_blksize: Default::default() }
}

/// `config` for file_per_sample: its old defaults under `params` (required since 2026-10-22).
fn fps(gpus: i64, seed: u64, params: &[(&str, &str)]) -> Config {
    config(gpus, seed, &crate::common::file_per_sample(params))
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
    // contract 0.2 and safetensors_load's `full` split the same day; all again 2026-10-01 for contract 0.3, the
    // optional `backend`; all again 2026-10-02 for contract 0.4, the optional `same_run` of a namespace, and
    // for contract 0.5, the wider `at` rule V3, with lmcache_local_disk and lmcache_shared_fs changed under it;
    // the three KV abstracts again 2026-10-02 for `prefill_step`, `think`, `trim`, and `turns`, DESIGN_REVIEW §3.59,
    // and 2026-10-07 for `sys_per_slot` and the system prompt read only when the engine holds none of the conversation,
    // then lmcache_local_disk for `direct`, LMCache's O_DIRECT calls (its fingerprints at `direct` false unchanged), §3.63;
    // the three again 2026-10-08 for the sub-agent prefix, `kind`, `first_in`, `sys_held`, and `sub_held`, whose defaults
    // leave every fingerprint unchanged; all again 2026-10-09 for contract 0.6, `api` and `cache` in place of
    // `backend`, DESIGN_REVIEW §3.65, every fingerprint unchanged; all again the same day for contract 0.7,
    // `posix` in place of `api` and `s3` in place of the protocol `object`, every fingerprint unchanged; all again
    // 2026-10-10 when each abstract was renamed for its directory under workloads/, DESIGN_REVIEW §3.66, every
    // fingerprint unchanged; all again 2026-10-13 for contract 0.8, one dataset shape with `layout`, `access`, and
    // `format.framing`, DESIGN_REVIEW §3.72, every fingerprint unchanged; all again the same day for contract 0.9,
    // a dataset's `inventory`, DESIGN_REVIEW §3.73, with file_per_sample and npy_per_sample's `enumerate` renamed `walk`
    // and an `inventory` read added, off by default, every fingerprint unchanged; all again the same day for
    // contract 0.10, `div` by a parameter, DESIGN_REVIEW §3.75, with the standard names `samples`,
    // `samples_per_file`, and `files_per_directory`, every fingerprint unchanged; the five training
    // abstracts again 2026-10-14, `sync_every` 1 by default, DESIGN_REVIEW §3.71 point 4, no fingerprint; all again
    // 2026-10-22 for contract 0.11 and 2026-10-23 for contract 0.12, a namespace's st_blksize, and the same day
    // for contract 0.13, V3 on a required parameter, every fingerprint unchanged).
    let want = [
        // 2026-10-23: the job's and the model's values required (its unsettled case four-items), `buf` and `meta_buf`
        // the st_blksize of its input namespaces (contract 0.12), DESIGN_REVIEW §3.78
        ("dcp_load", "b7f3bf97db57763935d86586f84409a81b76fc6b6123cad13a67580bed5423d9"),
        // 2026-10-23: the job's and the GPU's values required (its unsettled case four-items), `buf` and `xfer`
        // the st_blksize of its namespace (contract 0.12), DESIGN_REVIEW §3.78
        ("dcp_save", "53013972da025aff82f5f41b9227b029b07df4132d25ac158c4ac228948503c6"),
        // 2026-10-23: the model's on its GPU, the deployment's, and the load's values required (its unsettled case
        // chat-64-slots), DESIGN_REVIEW §3.78
        ("lmcache_local_disk", "16f242b5087cb86ea7ce2bc0f97871d017861b65c7598a810b4363ab1872374e"),
        // 2026-10-23: the model's shards and tensor table and the deployment's tp required (its unsettled case
        // five-tensors), DESIGN_REVIEW §3.78
        ("safetensors_load", "3cfab3b2572475f1ee9949af25d58a40de618fcaa050259dbdd0f91c89bee1a7"),
        // 2026-10-02: the `enumerate` phase, on by default (`walk` since 2026-10-13); 2026-10-22: the job's and the
        // GPU's values required (the use case unet3d-h100), the read sizes st_blksize, DESIGN_REVIEW §3.78
        ("npy_per_sample", "54aacfd3477c5708f60dc230b381270d40d68dc2c32d4e8c3194b53b85090760"),
        // 2026-10-02: `enumerate` on by default; 2026-10-22: the job's and the GPU's values and the corpus's median
        // required (its unsettled case jpeg-50m), the read sizes st_blksize, DESIGN_REVIEW §3.78
        ("file_per_sample", "7d31eb53d7446f9da83cec42e2342f87ced07ac2c135702a4c699d9e8e8080f7"),
        // 2026-10-23: the base's shape, the build's configuration, its graphs, and its compute required, the
        // disk index's sectors derived from the graph's degree (the traced example sift1m-13-shards), DESIGN_REVIEW §3.78
        ("diskann_build", "a5118edd1ea8b33c4d51f5fd90e33506596f162a3d0f5bec13542e9ed05b46cf"),
        // 2026-10-23: the index (as diskann_build names it, its sectors derived), the search's configuration, its
        // load, and its CPU required (the traced examples sift1m-nocache and sift1m-cache), DESIGN_REVIEW §3.78
        ("diskann_search", "41b6a22231bea721f46358a119512747a2858b535e883667f9dbc777437e785a"),
        // 2026-10-15 (RC1 review): a step takes the shards whose first sample is in its batch, for
        // any batch and samples_per_file, in place of `every(samples_per_file div batch)`; the
        // fingerprints where batch divides samples_per_file unchanged; the three again 2026-10-22, the job's and the GPU's
        // values required (each mechanism's untraced case holds the old defaults), DESIGN_REVIEW §3.78
        ("tfrecord_stream", "bf10f07938e49198907ae94b4a2438f49a5e7954e1bf1e415be7f171c0500bff"),
        ("parquet_stream", "444efde72b074b8dcfff9ee4abb5810a630ff8bc5ec466b022de795d91d179d2"),
        ("hdf5_map", "9f758776089271be97f5fa1e2e94e60a90c82b0876ce776ecf4fb4363f5c5051"),
        // 2026-10-23: the index, the search's configuration, its load, and its CPU required, the prefetch threads
        // derived (the traced example sift1m-ivf1024-pq32), DESIGN_REVIEW §3.78
        ("faiss_ivf_ondisk", "9a5b8371dc75aed9b19b64c83634529f41583a294e21acbba5792f8a32a96ef2"),
        // the shared store and its cold reader, 2026-10-02 (DESIGN_REVIEW §3.52); 2026-10-23 as lmcache_local_disk,
        // and `buf` the st_blksize of the store (contract 0.12)
        ("lmcache_shared_fs", "2f895fd237fb993e88607ce7251f7b62b03d98a3514e2ad086075d21c9d621c6"),
        ("lmcache_shared_fs_reader", "18258eee5c9c56924e1f2a9ac4cfb375a47368d06b51e7b9af3b2713510a7b9d"),
    ];
    for (name, sha) in want {
        let loaded = aeiou::load(&crate::common::workload_ast(name)).unwrap();
        assert_eq!(loaded.sha256, sha, "{name}");
        // the provenance block carries the same hash
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(crate::common::workload_ast(name)).unwrap()).unwrap();
        assert_eq!(doc["provenance"]["ast_sha256"], sha, "{name}: provenance");
    }
}

#[test]
fn golden_fingerprints() {
    // (abstract, gpus, params, fingerprint, ops) at --seed 1, recorded 2026-09-30 (lmcache_local_disk
    // re-recorded the same day after the conversation-directory `mkdir` was added to the abstract;
    // diskann_build likewise when its base file moved to `base/base.fbin`; dcp_save when its
    // readback phase went behind `readback = false`; file_per_sample on 2026-10-01 after the trace of
    // the real loader added the second `lseek` per file, DESIGN_REVIEW.md §3.43; npy_per_sample the
    // same day, rewritten from the trace of `np.load`, §3.44; dcp_save from the trace of
    // `dcp.save`, §3.45, and dcp_load with it: its default offsets follow the writer; dcp_load
    // again the same day, its item loop rewritten from the trace of `dcp.load`, §3.47; safetensors_load with the calls of
    // `safetensors.safe_open` and the small JSON files, same section; faiss_ivf_ondisk from the trace of FAISS
    // `OnDiskInvertedLists`, §3.49: the index file, the prefetch threads, ids before codes; diskann_search
    // 2026-10-02 from the trace of DiskANN's `PQFlashIndex`, §3.50: the load, the entry rounds, 24 to 30 more; diskann_build
    // the same day from the trace of `build_disk_index`, same section: the passes over the base, the shard files;
    // lmcache_local_disk the same day from the trace of vLLM with LMCache, §3.51: no lookups on storage, one flat
    // directory, whole chunks of the prompt only, reads only for what the engine lost; the three KV-cache abstracts
    // again the same day from the replay of ShareGPT, §3.56: one `reuse` distance, `keep` in place of `local`,
    // `context`, and the measured lengths as defaults; the three again the same day, §3.57: one `keep` draw per
    // round of the open conversations, `kp @ (r − (r mod d + 1))` under the wider rule V3 of contract 0.5).
    let cases: &[(&str, i64, &[(&str, &str)], u64, u64)] = &[
        // the directory walk is on by default since 2026-10-02 (DESIGN_REVIEW §3.54): five ops per class directory and
        // actor in file_per_sample (38,462 directories), four per directory in npy_per_sample; with
        // `enumerate=false` both are what they were; npy_per_sample at the values its defaults held until 2026-10-22, its
        // job and GPU values required since (the use case unet3d-h100, DESIGN_REVIEW §3.78)
        // since 2026-10-22 with the defaults its job and GPU values had (required since; DESIGN_REVIEW §3.78)
        ("file_per_sample", 2, &[("batch", "32"), ("workers", "8"), ("prefetch", "2"), ("step_time", "105000000"), ("samples", "50000000"), ("sample_median", "112640"), ("steps", "10")], 0xfb3215e5607b32ae, 389740),
        ("file_per_sample", 2, &[("batch", "32"), ("workers", "8"), ("prefetch", "2"), ("step_time", "105000000"), ("samples", "50000000"), ("sample_median", "112640"), ("steps", "10"), ("walk", "false")], 0x71628bdb4289c4c8, 5120),
        ("npy_per_sample", 2, &[("batch", "7"), ("workers", "4"), ("prefetch", "2"), ("step_time", "323000000"), ("samples", "50000"), ("sample_mean", "146800640"), ("sample_sd", "4194304"), ("steps", "5")], 0xf755bd4187b75b78, 50491),
        ("npy_per_sample", 2, &[("batch", "7"), ("workers", "4"), ("prefetch", "2"), ("step_time", "323000000"), ("samples", "50000"), ("sample_mean", "146800640"), ("sample_sd", "4194304"), ("steps", "5"), ("walk", "false")], 0xa2c284c6137f9639, 50451),
        // dcp_save at the values its defaults held until 2026-10-23 (its unsettled case four-items; DESIGN_REVIEW §3.78)
        ("dcp_save", 2, &[("steps", "200"), ("step_time", "323000000"), ("ckpt_every", "100"), ("items", "4"), ("item_bytes", "[16777216, 33554432, 16777216, 8388608]"), ("meta_bytes", "2097152")], 0x850e8c9019b3bda8, 130),
        // dcp_load at the values its defaults held until 2026-10-23 (its unsettled case four-items; DESIGN_REVIEW §3.78)
        ("dcp_load", 2, &crate::common::DCP_LOAD, 0x22990a88db4e6d4f, 368),
        // safetensors_load at the values its defaults held until 2026-10-23 (its unsettled case five-tensors; DESIGN_REVIEW §3.78)
        ("safetensors_load", 2, &crate::common::SAFETENSORS, 0x2668387c7aa4126b, 24682),
        // diskann_search at the values its defaults held until 2026-10-23 (no case holds them; DESIGN_REVIEW §3.78)
        ("diskann_search", 1, &[("n", "999999995"), ("dim", "128"), ("degree", "64"), ("pq_code", "32"), ("beam", "4"), ("hops", r#"{"empirical": {"values": [24, 25, 26, 27, 28, 29, 30], "weights": [1, 135, 536, 269, 47, 9, 2]}}"#), ("entry", r#"{"hotset": {"fraction": 2e-7, "weight": 1.0}}"#), ("near", r#"{"hotset": {"fraction": 1e-5, "weight": 1.0}}"#), ("node_cache", "0"), ("rerank", "500000"), ("queries", "100"), ("threads", "2")], 0x4b25fcd827dbfa0f, 21970),
        // faiss_ivf_ondisk at the values its defaults held until 2026-10-23 (no case holds them; DESIGN_REVIEW §3.78)
        ("faiss_ivf_ondisk", 1, &[("threads", "2"), ("calls", "100"), ("batch", "1"), ("nprobe", "64"), ("list_pop", r#"{"zipf": {"s": 0.2}}"#), ("distance", "400000"), ("lists", "1000000"), ("list_bytes", r#"{"lognormal": {"median": 40960, "sigma": 0.4}}"#), ("code_size", "32"), ("index_bytes", "553779476"), ("index_reads", "[8192, 536862720, 8192, 122880, 8192, 16769024, 8192]")], 0x42c16b5296ebaefc, 25612),
        // re-recorded 2026-10-23 (§3.78): `sectors` derived from `degree` 64, 200,001 at these million points where
        // 1,000 were given, so the layout writes twelve more calls of 64 MiB and a sector
        (
            "diskann_build",
            1,
            &[("dim", "128"), ("degree", "64"), ("pq_code", "32"), ("shards", "2"), ("n", "1000000"), ("shard_index_bytes", "4194304"), ("index_bytes", "8388608"), ("sample_rows", "1000"), ("pq_train", "60000000000"), ("partition_time", "90000000000"), ("build_time", "1800000000000")],
            0x93142e1f7cf15fb4,
            1229219,
        ),
        // re-recorded 2026-10-02 (§3.59): the writes under the prefill-step loop's index, and the `trim` and `turns` draws
        // before the conversation and system-prompt picks move their sites, so a seed draws other prompts (the fitted
        // files, whose system prompts have one size, keep their counts); re-recorded 2026-10-07 (§3.63): the pick inside
        // `sys_per_slot`'s `when` moves its site again, the same way (with one prompt size the op counts are unchanged)
        ("lmcache_local_disk", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")], 0x9f7017e7011639d5, 1590),
        // the same request stream on LMCache's fs:// backend, and the cold engine on the store it leaves (2026-10-02, §3.52)
        ("lmcache_shared_fs", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")], 0xc7d83399ee6948bc, 3417),
        ("lmcache_shared_fs_reader", 1, &[("concurrency", "2"), ("warm", "50"), ("requests", "50")], 0x382dba518a89c85f, 2986),
        // the container workloads (contract 0.2, 2026-09-30), at builder/tests/test_formats.py's configurations
        // and, since 2026-10-22, the defaults their job and GPU values had (required since; DESIGN_REVIEW §3.78)
        ("tfrecord_stream", 2, &[("samples", "768"), ("samples_per_file", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2"), ("prefetch", "1"), ("step_time", "105000000"), ("decode", "0"), ("sample_median", "112640")], 0xd48ec6c021d88539, 263),
        ("parquet_stream", 2, &[("samples", "768"), ("samples_per_file", "128"), ("batch", "32"), ("steps", "8"), ("cycle", "2"), ("prefetch", "1"), ("step_time", "105000000"), ("decode", "0"), ("sample_median", "112640")], 0x74a8ab1574196e74, 28),
        ("hdf5_map", 2, &[("samples", "64"), ("samples_per_file", "16"), ("batch", "4"), ("workers", "2"), ("steps", "4"), ("prefetch", "2"), ("step_time", "105000000")], 0x3dd38374803bff38, 384),
    ];
    for (name, gpus, params, fp, ops) in cases {
        let text = std::fs::read_to_string(crate::common::workload_ast(name)).unwrap();
        // the LMCache abstracts at the values their defaults held until 2026-10-23 (their unsettled cases chat-64-slots)
        let params = if name.starts_with("lmcache") { crate::common::lmcache(params) } else { params.to_vec() };
        let cfg = config(*gpus, 1, &params);
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
    let text = std::fs::read_to_string(crate::common::workload_ast("file_per_sample.ast.json")).unwrap();
    let set = ParamSet::load(&crate::common::workloads().join("input-pipeline/map/file-per-sample/params/smoke.params.json")).unwrap();
    assert_eq!(set.abstract_name, "file_per_sample");
    // the set (files 4000, steps 50) under --param steps=10 equals the same values given as overrides
    let mut cfg = config(2, 1, &[("steps", "10")]);      // the smoke set gives every required value
    cfg.sets = vec![set.clone()];
    let by_file = dry(&text, &cfg, 1);
    let by_cli = dry(&text, &fps(2, 1, &[("samples", "4000"), ("steps", "10")]), 1);
    assert_eq!(by_file.total.fingerprint, by_cli.total.fingerprint);
    assert_ne!(by_file.total.fingerprint, dry(&text, &fps(2, 1, &[("steps", "10")]), 1).total.fingerprint, "samples=4000 changes the permutation");
    // the rules (schema/README.md §8)
    let loaded = aeiou::load_str(&text).unwrap();
    let outcome = |json: &str| -> String {
        let set = ParamSet::parse(json, "x.params.json").unwrap();
        // the job values the file does not name, required since 2026-10-22, from the command line
        let job: Vec<_> = crate::common::FILE_PER_SAMPLE.into_iter().filter(|(k, _)| !json.contains(&format!("\"{k}\""))).collect();
        let cfg = Config { sets: vec![set], ..config(2, 1, &job) };
        match Params::new(&loaded.ast, &cfg) {
            Ok(_) => "accepted".into(),
            Err(e) => format!("{e:#}"),
        }
    };
    assert!(outcome(r#"{"params_version":2,"abstract":"other","params":{}}"#).contains("for abstract `other`"));
    assert!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"nope":{"default":1}}}"#).contains("no parameter `nope`"));
    assert!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"inv_entry":{"default":[1]}}}"#).contains("the default is a scalar, the value is an array"));
    assert!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"steps":{"default":[1]}}}"#).contains("the kind is a scalar, the value is an array"));
    assert!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"step_time":{"locked":{"const":5}}}}"#).contains("the value is a distribution"));
    assert!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"gpus":{"default":2}}}"#).contains("--gpus"));
    assert_eq!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"walk":{"default":true},"hdr_read":{"locked":4096}}}"#), "accepted");
    // contract 0.11: a required entry leaves the parameter without a value; a locked one is final
    assert_eq!(outcome(r#"{"params_version":2,"abstract":"file_per_sample","params":{"steps":{"required":true},"batch":{"required":true}}}"#), "required parameters with no value: batch, steps (give each in a parameter file or with --param NAME=VALUE)");
    let two = |a: &str, b: &str, cli: &[(&str, &str)]| {
        let sets = [a, b].iter().enumerate().map(|(i, j)| ParamSet::parse(&format!(r#"{{"params_version":2,"abstract":"file_per_sample","params":{j}}}"#), &format!("{i}.params.json")).unwrap()).collect();
        let mut job: Vec<_> = crate::common::FILE_PER_SAMPLE.into_iter().filter(|(k, _)| *k != "steps").collect();
        job.extend_from_slice(cli);
        Params::new(&loaded.ast, &Config { sets, ..config(2, 1, &job) }).map(|_| ()).map_err(|e| format!("{e:#}"))
    };
    assert_eq!(two(r#"{"steps":{"locked":5}}"#, r#"{"steps":{"default":6}}"#, &[]), Err("1.params.json: steps: locked by 0.params.json".into()));
    assert_eq!(two(r#"{"steps":{"locked":5}}"#, r#"{}"#, &[("steps", "6")]), Err("--param steps: locked by 0.params.json".into()));
    assert_eq!(two(r#"{"steps":{"default":5}}"#, r#"{"steps":{"required":true}}"#, &[("steps", "6")]), Ok(()), "a later layer gives what an earlier one requires");
    for (bad, want) in [
        (r#"{"params_version":2,"abstract":"a","params":{"n":4}}"#, "params: n: not an entry"),
        (r#"{"params_version":2,"abstract":"a","params":{"n":{"required":false}}}"#, "params: n: not an entry"),
        (r#"{"params_version":2,"abstract":"a","params":{"n":{"default":1,"locked":1}}}"#, "params: n: not an entry"),
        (r#"{"params_version":2,"abstract":"a","params":{},"run":{"posix":{"locked":"sync"}}}"#, "a `run` table sets the runner's identity options"),
        (r#"{"params_version":1,"abstract":"a","params":{}}"#, "params_version 1 (this runner reads 2)"),
    ] {
        let e = ParamSet::parse(bad, "x").err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(e.contains(want), "{bad}: {e}");
    }
    assert!(ParamSet::parse(r#"{"params_version":2,"abstract":"a","params":{},"extra":1}"#, "x").is_err());
    assert!(ParamSet::parse(r#"{"params_version":2,"abstract":"a"}"#, "x").is_err());
    // an AST hash pins a set to one shape
    let pinned = ParamSet::parse(&format!(r#"{{"params_version":2,"abstract":"file_per_sample","ast_sha256":"{}","params":{{}}}}"#, "0".repeat(64)), "x").unwrap();
    let cfg = Config { sets: vec![pinned], ..config(2, 1, &[]) };
    assert!(format!("{:#}", cfg.check_sets("file_per_sample", &loaded.sha256).unwrap_err()).contains("is for AST"));
    let pinned = ParamSet::parse(&format!(r#"{{"params_version":2,"abstract":"file_per_sample","ast_sha256":"{}","params":{{}}}}"#, loaded.sha256), "x").unwrap();
    assert!(Config { sets: vec![pinned], ..config(2, 1, &[]) }.check_sets("file_per_sample", &loaded.sha256).is_ok());
    // --param keeps the kind too
    assert!(Params::new(&loaded.ast, &config(2, 1, &[("steps", "[1]")])).is_err());
    // the committed synthetic tensor table drives safetensors_load
    let text = std::fs::read_to_string(crate::common::workload_ast("safetensors_load.ast.json")).unwrap();
    let set = ParamSet::load(&crate::common::workloads().join("model-loading/safetensors-load/params/synthetic.params.json")).unwrap();
    let mut cfg = config(2, 1, &[]);
    cfg.sets = vec![set];
    let r = dry(&text, &cfg, 1);
    assert!(r.total.total.ops > 2 * 21, "every tensor of the table is read: {} ops", r.total.total.ops);
}

#[test]
fn seed_changes_drawn_workloads_only() {
    let text = std::fs::read_to_string(crate::common::workload_ast("file_per_sample.ast.json")).unwrap();
    let a = dry(&text, &fps(2, 1, &[("steps", "10")]), 2).total.fingerprint;
    let b = dry(&text, &fps(2, 2, &[("steps", "10")]), 2).total.fingerprint;
    assert_ne!(a, b, "the file order depends on the seed");
    // no draw anywhere: the seed is irrelevant, the parameters and the actor ids decide everything
    let text = std::fs::read_to_string(crate::common::workload_ast("dcp_load.ast.json")).unwrap();
    let a = dry(&text, &config(2, 1, &crate::common::DCP_LOAD), 2).total.fingerprint;
    let b = dry(&text, &config(2, 2, &crate::common::DCP_LOAD), 2).total.fingerprint;
    assert_eq!(a, b);
    // and the workload changes with G, by design
    let c = dry(&text, &config(3, 1, &crate::common::DCP_LOAD), 2).total.fingerprint;
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
  "ast": "0.13", "name": "consume_test",
  "params": {"batches": {"default": 24}, "batch": {"default": 4}},
  "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 100, "size": {"const": 10}, "seed": 7}},
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
  "ast": "0.13", "name": "chain_test",
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
  "ast": "0.13", "name": "eof_test",
  "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 4, "size": {"const": 2621440}, "seed": 7}},
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
            r#"{{"ast": "0.13", "name": "t", "params": {{"n": {{"default": 3}}}},
                "datasets": {{{datasets}}},
                "actors": {{"gpu": {{"body": [{body}]}}}}}}"#
        )
    };
    let ds = r#""d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1}"#;
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": {"param": "gpus"}, "body": []}}"#).replace(r#""n": {"default": 3}"#, r#""gpus": {"default": 3}"#), "reserved");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"let": {"name": "f", "value": {"consume": "d"}}}, {"write": {"file": {"ref": "f"}, "len": 1}}]}}"#), "read-only (V12)");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"open": {"file": {"file": {"dataset": "d", "id": 0}}, "flags": ["WRONLY"]}}]}}"#), "read-only (V12)");
    // a dataset's directories too, as the builder and V14 refuse them
    rejects(&base(ds, r#"{"rmdir": {"dir": {"dir": {"dataset": "d", "id": 0}}}}"#), "rmdir on dataset `d`: datasets are read-only (V12)");
    rejects(&base(ds, r#"{"mkdir": {"dir": {"dir": {"dataset": "d", "id": 0}}}}"#), "mkdir on dataset `d`: datasets are read-only (V12)");
    rejects(&base(r#""d": {"layout": "whole", "access": "map", "pattern": ".aeiou-x/{id}", "count": 10, "size": {"const": 1}, "seed": 1}"#, ""), ".aeiou");
    rejects(&base(ds, r#"{"let": {"name": "f", "value": {"consume": "d"}}}"#), "outside any loop");
    // contract 0.8: one dataset shape; the fields each layout takes (V20) and access (V21)
    let one = |d: &str| base(&format!(r#""d": {{{d}, "count": 10, "size": {{"const": 1}}, "seed": 1}}"#), "");
    rejects(&one(r#""pattern": "d/{id}", "layout": "whole", "access": "map", "samples_per_file": 2"#), "layout `whole` takes no `samples_per_file` (V20)");
    rejects(&one(r#""pattern": "d/{id}", "layout": "compound", "access": "map""#), "layout `compound` needs `samples_per_file` (V20)");
    rejects(&one(r#""pattern": "d/{id}", "layout": "chunked", "access": "indexed", "chunk": 4"#), "needs a `{k}` field (V20)");
    rejects(&one(r#""pattern": "d/{id}_{k}", "layout": "whole", "access": "map""#), "only a `chunked` dataset has one (V20)");
    rejects(&one(r#""pattern": "d/{id}", "layout": "array", "access": "indexed", "slot": 4"#), "its pattern has no fields (V20)");
    rejects(&one(r#""pattern": "d/one.bin", "layout": "array", "access": "map", "slot": 4"#), "an `array` is `indexed`");
    rejects(&one(r#""pattern": "d/{id mod 16}/f_{id}", "layout": "whole", "access": "map""#), "never `mod`");
    aeiou::load_str(&one(r#""pattern": "d/{id mod 16}/f_{id}", "layout": "whole", "access": "indexed""#)).unwrap();
    rejects(&one(r#""pattern": "d/{id}", "layout": "whole""#), "missing field `access`");
    rejects(&one(r#""pattern": "d/{id}", "layout": "regions", "access": "indexed""#), "unknown variant `regions`");
    // contract 0.9: a dataset's inventory (V22, DESIGN_REVIEW §3.73)
    let inv = |pattern: &str, access: &str, name: &str| one(&format!(r#""pattern": "{pattern}", "layout": "whole", "access": "{access}", "inventory": {{"name": "{name}", "entry": 8}}"#));
    rejects(&inv("d/{id}", "indexed", "list.txt"), "only a consumed dataset (`map` or `stream`) has an inventory");
    rejects(&inv("d/{id}", "map", "a/list.txt"), "is not a file name");
    rejects(&inv("d/{id}", "map", ".aeiou-list"), "begins with `.aeiou`");
    rejects(&inv("d/{id}", "map", "12"), "a name the pattern `d/{id}` can produce");
    rejects(&inv("d/f_{id:04}.jpg", "stream", "f_0012.jpg"), "can produce in the dataset's root (V22)");
    aeiou::load_str(&inv("d/f_{id:04}.jpg", "stream", "files.txt")).unwrap();
    aeiou::load_str(&inv("d/{id div 4:05}/f_{id}", "map", "inventory.txt")).unwrap();
    rejects(&base(ds, r#"{"open": {"file": {"inventory": "d"}, "flags": ["RDONLY"]}}"#), "`inventory` of `d`, which declares none (V22)");
    // contract 0.10: a divisor named in a pattern is a declared integer parameter (V23, DESIGN_REVIEW §3.75)
    aeiou::load_str(&one(r#""pattern": "d/{id div n:03}/f_{id}", "layout": "whole", "access": "map""#)).unwrap();
    rejects(&one(r#""pattern": "d/{id div width}/f_{id}", "layout": "whole", "access": "map""#), "divisor `width` is not a declared parameter (V23)");
    rejects(&one(r#""pattern": "d/{id div n}/f_{id}", "layout": "whole", "access": "map""#).replace(r#""n": {"default": 3}"#, r#""n": {"default": 2.5}"#), "whose kind is not an integer (V23)");
    rejects(&one(r#""pattern": "d/{id div n}/f_{id}", "layout": "whole", "access": "map""#).replace(r#""n": {"default": 3}"#, r#""n": {"default": {"const": 3}}"#), "whose kind is not an integer (V23)");
    rejects(&one(r#""pattern": "d/{id mod n}/f_{id}", "layout": "whole", "access": "map""#), "never `mod`");
    let with_inv = r#""d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1, "inventory": {"name": "list.txt", "entry": 8}}"#;
    rejects(&base(with_inv, r#"{"open": {"file": {"inventory": "d"}, "flags": ["WRONLY"]}}"#), "read-only (V12)");
    let indexed = r#""d": {"layout": "whole", "access": "indexed", "pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1}"#;
    rejects(&base(indexed, r#"{"loop": {"index": "i", "to": 1, "body": [{"let": {"name": "f", "value": {"consume": "d"}}}]}}"#), "which is `indexed`");
    // contract 0.7: the declared APIs (one per protocol) and cache mode, each a name, and never
    // `direct` under `mmap`; `api` (0.6) and `backend` (0.3) are gone
    let declared = |kv: &str| base(ds, "").replacen(r#""name": "t","#, &format!(r#""name": "t", {kv},"#), 1);
    rejects(&declared(r#""posix": "pread""#), "is not one of");
    rejects(&declared(r#""cache": "dontcache""#), "is not one of");
    rejects(&declared(r#""posix": "mmap", "cache": "direct""#), "page faults on a mapping");
    rejects(&declared(r#""api": "sync""#), "api");
    rejects(&declared(r#""backend": "sync""#), "backend");
    // V19: an API only for a protocol with names (the dataset `d` is posix)
    rejects(&declared(r#""s3": "async""#), "no dataset or namespace is `protocol: s3` (V19)");
    // V17: namespaces sharing a root share its place, so they declare one protocol
    let two = |pa: &str, pb: &str| {
        base(ds, "").replacen(
            r#""datasets""#,
            &format!(
                r#""namespaces": {{"a": {{"pattern": "o/{{k}}.a", "fields": {{"k": "int"}}, "size": 1, "seed": 2{pa}}},
                                  "b": {{"pattern": "o/{{k}}.b", "fields": {{"k": "int"}}, "size": 1, "seed": 3{pb}}}}}, "datasets""#
            ),
            1,
        )
    };
    rejects(&two(r#", "protocol": "s3""#, ""), "`protocol` differs (V17)");
    aeiou::load_str(&two(r#", "protocol": "s3""#, r#", "protocol": "s3""#)).unwrap();
    rejects(&two(r#", "protocol": "object""#, r#", "protocol": "object""#), "unknown variant `object`");
    // V19: posix and s3 names together take APIs of one kind, defaults counted
    let both = |kv: &str| two(r#", "protocol": "s3""#, r#", "protocol": "s3""#).replacen(r#""name": "t","#, &format!(r#""name": "t", {kv},"#), 1);
    rejects(&both(r#""posix": "io_uring""#), "one is event-driven and the other a thread per actor");
    rejects(&both(r#""s3": "async""#), "one is event-driven and the other a thread per actor");
    aeiou::load_str(&both(r#""posix": "libaio", "s3": "async""#)).unwrap();
    aeiou::load_str(&both(r#""posix": "mmap""#)).unwrap();
    // V18: an object in an object store is written once, in order from 0, by one upload
    let v18 = |protocol: &str, op: &str| {
        two(protocol, protocol).replacen(r#""body": []"#, &format!(r#""body": [{{"loop": {{"index": "i", "to": 1, "body": [{{"let": {{"name": "f", "value": {{"object": {{"namespace": "a", "fields": {{"k": {{"index": "i"}}}}}}}}}}}}, {op}]}}}}]"#), 1)
    };
    let obj = r#", "protocol": "s3""#;
    for flag in ["APPEND", "RDWR", "EXCL"] {
        rejects(&v18(obj, &format!(r#"{{"open": {{"file": {{"ref": "f"}}, "flags": ["WRONLY", "CREAT", "{flag}"]}}}}"#)), "(V18)");
    }
    rejects(&v18(obj, r#"{"ftruncate": {"file": {"ref": "f"}, "len": 0}}"#), "has no ftruncate (V18)");
    rejects(&v18(obj, r#"{"fallocate": {"file": {"ref": "f"}, "len": 4096}}"#), "has no fallocate (V18)");
    let open = r#"{"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT", "TRUNC"]}}, {"write": {"file": {"ref": "f"}, "len": 4096}}"#;
    let close = r#"{"close": {"file": {"ref": "f"}}}"#;
    aeiou::load_str(&v18(obj, &format!("{open}, {close}"))).unwrap();
    aeiou::load_str(&v18("", r#"{"open": {"file": {"ref": "f"}, "flags": ["RDWR", "CREAT"]}}"#)).unwrap();
    // an upload is begun by WRONLY alone: CREAT or TRUNC on a read-only handle would make or
    // empty an object it never writes (RC1 review, bug R30)
    for flags in [r#"["RDONLY", "CREAT"]"#, r#"["RDONLY", "TRUNC"]"#, r#"["CREAT"]"#] {
        rejects(&v18(obj, &format!(r#"{{"open": {{"file": {{"ref": "f"}}, "flags": {flags}}}}}, {close}"#)), "without WRONLY on namespace `a`, declared `protocol: s3`: an object is made only by the upload an open with WRONLY begins, and a read-only handle writes none (V18)");
    }
    aeiou::load_str(&v18("", r#"{"open": {"file": {"ref": "f"}, "flags": ["RDONLY", "CREAT"]}}"#)).unwrap();
    // the close sends the upload, so it comes later in the same body, on the same handle, with
    // no `let` rebinding a name the handle reads in between (bug R29); a posix name need not
    let unclosed = "with no close of that handle later in the same body (before it is opened again or a name it reads is bound again): an object is made by one upload that its close sends, so an upload never closed would be discarded (V18)";
    rejects(&v18(obj, open), unclosed);
    rejects(&v18(obj, &format!("{open}, {open}, {close}")), unclosed);
    aeiou::load_str(&v18("", open)).unwrap();
    rejects(&v18(obj, &format!(r#"{open}, {{"cond": {{"if": true, "then": [{close}], "else": [{close}]}}}}"#)), unclosed);
    rejects(&v18(obj, &format!(r#"{open}, {{"let": {{"name": "f", "value": {{"object": {{"namespace": "a", "fields": {{"k": 7}}}}}}}}}}, {close}"#)), unclosed);
    rejects(&v18(obj, &format!(r#"{open}, {{"close": {{"file": {{"object": {{"namespace": "a", "fields": {{"k": {{"index": "i"}}}}}}}}}}}}"#)), unclosed);
    aeiou::load_str(&v18(obj, &format!(r#"{open}, {{"let": {{"name": "n", "value": 1}}}}, {close}"#))).unwrap();
    aeiou::load_str(&v18(obj, &format!(r#"{open}, {{"loop": {{"index": "j", "to": 2, "body": [{{"write": {{"file": {{"ref": "f"}}, "len": 4096}}}}]}}}}, {close}"#))).unwrap();
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 1, "body": [{"stat": {"file": {"ref": "nope"}}}]}}"#), "not in scope");
    rejects(&base(ds, r#"{"compute": {"ns": {"param": "missing"}}}"#), "unknown param");
    rejects(&base(ds, r#"{"loop": {"index": "i", "to": 4, "body": [{"let": {"name": "x", "value": {"cond": {"if": true, "then": {"at": {"ref": "x", "index": {"index": "i"}}}, "else": 1}}}}]}}"#), "provably >= 1");
    // the wider V3 of contract 0.5 (§3.57): the offset may be an expression provably >= 1, and the rule
    // covers every binding of the same loop body, not only the one being defined
    let accepts = |text: &str| aeiou::load_str(text).map(|_| ()).unwrap_or_else(|e| panic!("rejected a valid abstract:\n{e:#}"));
    let chain = |e: &str| format!(r#"{{"loop": {{"index": "i", "to": 4, "body": [{{"let": {{"name": "d", "value": {{"draw": {{"uniform": {{"lo": 1, "hi": 5}}}}}}}}}}, {{"let": {{"name": "x", "value": {{"cond": {{"if": true, "then": {{"at": {{"ref": "x", "index": {{"sub": [{{"index": "i"}}, {e}]}}}}}}, "else": 1}}}}}}}}]}}}}"#);
    accepts(&base(ds, &chain(r#"{"add": [{"mod": [{"index": "i"}, {"ref": "d"}]}, 1]}"#))); // the per-round draw: r − (r mod d + 1)
    accepts(&base(ds, &chain(r#"{"add": [{"index": "i"}, {"ref": "d"}]}"#))); // index >= 0 (no `from`) plus a draw >= 1
    // a parameter whose value is a distribution is drawn: as an operand it is a distribution (V27)
    accepts(&base(ds, &chain(r#"{"add": [{"add": [{"mod": [3, 2]}, 0]}, {"draw": {"param": "n"}}]}"#)).replace(r#""n": {"default": 3}"#, r#""n": {"default": {"uniform": {"lo": 1, "hi": 2}}}"#));
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
            r#""a": {"layout": "whole", "access": "map", "pattern": "x/{id}", "count": 1, "size": {"const": 1}, "seed": 1}, "b": {"layout": "whole", "access": "map", "pattern": "x/y{id}", "count": 1, "size": {"const": 1}, "seed": 2}"#,
            "",
        ),
        "shares root",
    );
    // or one root inside another (V13, 2026-10-01): datagen wants each root to itself
    rejects(
        &base(
            r#""a": {"layout": "whole", "access": "map", "pattern": "x/{id}", "count": 1, "size": {"const": 1}, "seed": 1}, "b": {"layout": "whole", "access": "map", "pattern": "x/y/{id}", "count": 1, "size": {"const": 1}, "seed": 2}"#,
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
    let text = std::fs::read_to_string(crate::common::workload_ast("lmcache_local_disk.ast.json")).unwrap();
    let loaded = aeiou::load_str(&text).unwrap();
    let small = [("concurrency", "1"), ("warm", "0"), ("requests", "20")];
    let with = |reuse: &str| {
        let mut p = small.to_vec();
        p.push(("reuse", reuse));
        config(1, 1, &crate::common::lmcache(&p))
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
  "ast": "0.13", "name": "long_chain",
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

/// An `at` in a short loop nested in a long one evaluates at the same few indices under every
/// outer index; the values kept are shed all the same, at a constant cost per value. (They were
/// shed by sorting their indices, which shed nothing when the indices were equal: past 65,536
/// values every one sorted them all, 18 s for 100,000 outer indices; RC1 review.)
#[test]
fn an_at_in_a_short_loop_nested_in_a_long_one_stays_linear() {
    let text = r#"{
  "ast": "0.13", "name": "nested_chain",
  "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": 1, "seed": 3}},
  "actors": {"gpu": {"count": 1, "body": [
    {"loop": {"index": "j", "to": 200000, "body": [
      {"loop": {"index": "i", "to": 2, "body": [
        {"let": {"name": "x", "value": {"cond": {"if": {"eq": [{"index": "i"}, 0]}, "then": {"index": "j"},
            "else": {"add": [{"at": {"ref": "x", "index": {"sub": [{"index": "i"}, 1]}}}, 1]}}}}},
        {"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"ref": "x"}}}}}}
      ]}}
    ]}}
  ]}}
}"#;
    let t = std::time::Instant::now();
    let ops = collect(text, &config(1, 1, &[]), "gpu", 1);
    let took = t.elapsed();
    assert_eq!(ops.len(), 400_000);
    assert_eq!((ops[1].3.as_str(), ops[399_998].3.as_str(), ops[399_999].3.as_str()), ("o/1", "o/199999", "o/200000"));
    assert!(took < std::time::Duration::from_secs(20), "200,000 outer indices took {took:?}");
}

/// A value keeps its default's kind down to the number (schema/README.md §8; RC1 review): where
/// the default is an integer, `samples=3e3` and `3000.0` are 3000 (the manifest recorded
/// `"count": 3000.0`, so the dataset id was another one for the same files and `run --param
/// samples=3000` refused it, "count: 3000 vs 3000.0"), a divisor `1300.0` binds as 1300 (it was
/// refused "not an integer"), and a fractional value is refused (it was rounded), from
/// `--param` and a parameter file alike.
#[test]
fn an_integral_float_is_the_integer_its_default_is() {
    use aeiou::eval::ParamSet;
    let path = crate::common::workload_ast("file_per_sample.ast.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let loaded = aeiou::load_str(&text).unwrap();
    // the job values the file does not give, required since 2026-10-22
    let job: Vec<_> = crate::common::FILE_PER_SAMPLE.into_iter().filter(|(k, _)| *k != "samples").collect();
    let file = |json: &str| Config { sets: vec![ParamSet::parse(&format!(r#"{{"params_version":2,"abstract":"file_per_sample","params":{json}}}"#), "x.params.json").unwrap()], ..config(1, 0, &job) };
    let resolved = |cfg: &Config| aeiou::payload::resolved_dataset(&doc, "train", cfg).unwrap();
    let want = resolved(&fps(1, 0, &[("samples", "3000"), ("files_per_directory", "1300")]));
    assert_eq!(want["count"], serde_json::json!(3000));
    for cfg in [fps(1, 0, &[("samples", "3e3"), ("files_per_directory", "1300.0")]), fps(1, 0, &[("samples", "3000.0")]), file(r#"{"samples":{"default":3000.0},"files_per_directory":{"locked":1.3e3}}"#)] {
        assert_eq!(resolved(&cfg), want, "{:?} {:?}", cfg.overrides, cfg.sets.first().map(|s| &s.values));
        let params = Params::new(&loaded.ast, &cfg).unwrap();
        assert!(matches!(params.values["samples"], aeiou::ast::PValue::Scalar(aeiou::ast::Literal::Int(3000))), "{:?}", params.values["samples"]);
        build_model(&loaded.ast, &cfg, &params).unwrap();
    }
    let refused = |cfg: &Config| Params::new(&loaded.ast, cfg).err().map(|e| format!("{e:#}")).unwrap_or_default();
    assert_eq!(refused(&fps(1, 0, &[("samples", "20.7")])), "--param samples: the kind is an integer, the value 20.7 is not");
    assert_eq!(refused(&file(r#"{"samples":{"default":1.5}}"#)), "x.params.json: samples: the kind is an integer, the value 1.5 is not");
    assert!(aeiou::payload::resolved_dataset(&doc, "train", &fps(1, 0, &[("samples", "20.7")])).is_err());
}

/// A divisor named in a pattern (contract 0.10, DESIGN_REVIEW §3.75) is bound to the parameter's
/// value: the names, the directory count, and the resolved definition use the number, so at the
/// default the dataset id is the literal pattern's, and another width is another dataset.
#[test]
fn a_divisor_parameter_binds_the_names_the_directories_and_the_dataset_id() {
    let path = crate::common::workload_ast("file_per_sample.ast.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["datasets"]["train"]["pattern"], "train/{id div files_per_directory:05}/img_{id:09}.jpg");
    let resolved = |doc: &serde_json::Value, params: &[(&str, &str)]| aeiou::payload::resolved_dataset(doc, "train", &fps(1, 0, params)).unwrap();
    let at_default = resolved(&doc, &[("samples", "5000")]);
    assert_eq!(at_default["pattern"], "train/{id div 1300:05}/img_{id:09}.jpg");
    let mut literal = doc.clone();
    literal["datasets"]["train"]["pattern"] = "train/{id div 1300:05}/img_{id:09}.jpg".into();
    assert_eq!(at_default, resolved(&literal, &[("samples", "5000")]), "the default width resolves to the literal pattern's definition");
    let wide = resolved(&doc, &[("samples", "5000"), ("files_per_directory", "4096")]);
    assert_eq!(wide["pattern"], "train/{id div 4096:05}/img_{id:09}.jpg");
    assert_ne!(wide, at_default);

    let loaded = aeiou::load_str(&text).unwrap();
    for (width, dirs, name) in [("1300", 4, "train/00003/img_000004000.jpg"), ("4096", 2, "train/00000/img_000004000.jpg")] {
        let cfg = fps(1, 0, &[("samples", "5000"), ("files_per_directory", width)]);
        let params = Params::new(&loaded.ast, &cfg).unwrap();
        let model = build_model(&loaded.ast, &cfg, &params).unwrap();
        match &model.datasets[model.dataset("train").unwrap()] {
            aeiou::eval::DsMeta::Files { pattern, ndirs, .. } => {
                assert_eq!(*ndirs, dirs, "width {width}");
                assert_eq!(pattern.format(|f| (f == "id").then_some(aeiou::pattern::FieldValue::Int(4000))).unwrap(), name);
            }
            _ => unreachable!(),
        }
    }
    let cfg = fps(1, 0, &[("files_per_directory", "0")]);
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let e = format!("{:#}", build_model(&loaded.ast, &cfg, &params).err().unwrap());
    assert!(e.contains("`div files_per_directory` is 0: it needs a positive divisor"), "{e}");
}

/// A compound dataset's directories are counted from its files, not its samples: the pattern's
/// `id` is a shard's (DESIGN_REVIEW §3.76, found 2026-10-14). 10,240 samples of 1,024 a shard are
/// 10 shards, one directory of `{id div 10}`; it was 1,024 directories, the sample count over 10.
#[test]
fn a_compound_datasets_directories_are_counted_from_its_files() {
    let text = r#"{
      "ast": "0.13", "name": "shards_in_dirs",
      "params": {"samples": {"default": 10240}},
      "datasets": {"s": {"layout": "compound", "access": "stream", "pattern": "s/{id div 10:03}/shard-{id:05}", "count": {"param": "samples"},
                         "samples_per_file": 1024, "size": {"const": 4096}, "seed": 7}},
      "actors": {"gpu": {"count": 1, "body": [{"loop": {"index": "d", "to": {"dirs": "s"}, "body": [
        {"let": {"name": "h", "value": {"dir": {"dataset": "s", "id": {"index": "d"}}}}},
        {"stat": {"file": {"ref": "h"}}}]}}]}}
    }"#;
    let loaded = aeiou::load_str(text).unwrap();
    for (samples, dirs) in [("10240", 1), ("10241", 2), ("1024", 1), ("102400", 10)] {
        let cfg = config(1, 0, &[("samples", samples)]);
        let params = Params::new(&loaded.ast, &cfg).unwrap();
        let model = build_model(&loaded.ast, &cfg, &params).unwrap();
        let meta = &model.datasets[model.dataset("s").unwrap()];
        let ndirs = |m: &aeiou::eval::DsMeta| match m {
            aeiou::eval::DsMeta::Files { ndirs, .. } => *ndirs,
            _ => unreachable!(),
        };
        assert_eq!(ndirs(meta), dirs, "{samples} samples");
        assert_eq!(aeiou::project::project(meta, 1).unwrap().directories, dirs);
        assert_eq!(ndirs(&meta.with_count(20480)), 2, "a grown corpus counts its files too: 20 shards");
    }
}

const INVENTORY: &str = r#"{
  "ast": "0.13", "name": "inventory_test",
  "params": {"files": {"default": 100000}},
  "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id div 1000:03}/f_{id:06}", "count": {"param": "files"},
                     "size": {"const": 4096}, "seed": 7, "inventory": {"name": "inventory.txt", "entry": 24}}},
  "actors": {"gpu": {"count": 1, "body": [
    {"let": {"name": "inv", "value": {"inventory": "d"}}},
    {"open": {"file": {"ref": "inv"}, "flags": ["RDONLY"]}},
    {"read": {"file": {"ref": "inv"}, "len": 1048576, "repeat": "until_eof"}},
    {"close": {"file": {"ref": "inv"}}}
  ]}}
}"#;

#[derive(Default)]
struct Issued {
    reads: Vec<(i64, i64, i64, i64)>,
}

impl<'m, 'a> Sink<'m, 'a> for Issued {
    fn op(&mut self, op: &Op, _: &OpCtx) -> anyhow::Result<()> {
        if op.kind == OpKind::Read {
            assert_eq!(op.path, "d/inventory.txt");
            self.reads.push((op.offset, op.len, op.bytes, op.issue_len()));
        }
        Ok(())
    }
}

/// The inventory's size is `count × entry` at the run's count; under `--subset` (capped) the
/// reads are the same ops, so the same fingerprint, issued no further than that size: the tail
/// short, the read that finds the end with length 0 (DESIGN_REVIEW §3.73 item 5).
#[test]
fn an_inventory_is_count_times_entry_and_a_subset_caps_its_reads() {
    let loaded = aeiou::load_str(INVENTORY).unwrap();
    let cfg = config(1, 0, &[("files", "50000")]);
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let plain = build_model(&loaded.ast, &cfg, &params).unwrap();
    let mut capped = build_model(&loaded.ast, &cfg, &params).unwrap();
    capped.cap_inventory("d");
    let size = 50000 * 24;
    let want: Vec<(i64, i64, i64)> = vec![(0, 1048576, 1048576), (1048576, 1048576, size - 1048576), (size, 1048576, 0)];
    let a = run_actor(&plain, "gpu", 0, 1, Issued::default()).unwrap().reads;
    let b = run_actor(&capped, "gpu", 0, 1, Issued::default()).unwrap().reads;
    assert_eq!(a.iter().map(|r| (r.0, r.1, r.2)).collect::<Vec<_>>(), want);
    assert_eq!(b.iter().map(|r| (r.0, r.1, r.2)).collect::<Vec<_>>(), want, "the same ops capped");
    assert!(a.iter().all(|r| r.3 == r.1), "uncapped: issued as asked");
    assert_eq!(b.iter().map(|r| r.3).collect::<Vec<_>>(), vec![1048576, size - 1048576, 0], "capped: the tail short, the end read 0");
    assert_eq!(dryrun::run(&plain, 1, None).unwrap().total.fingerprint, dryrun::run(&capped, 1, None).unwrap().total.fingerprint);
}

/// The `x @ i - d` loop of the rule V3 review (RC1, bug 30): `d` drawn from `DIST`.
fn v3_chain(dist: &str, params: &str) -> String {
    r#"{
  "ast": "0.13", "name": "v3_chain", "params": PARAMS,
  "datasets": {"d": {"pattern": "d/f{id}", "layout": "whole", "access": "indexed", "count": 4, "size": {"const": 4096}, "seed": 1}},
  "actors": {"gpu": {"count": 1, "body": [
    {"loop": {"index": "i", "to": 4, "body": [
      {"let": {"name": "d", "value": {"draw": DIST}}},
      {"let": {"name": "x", "value": {"cond": {"if": {"gt": [{"index": "i"}, 0]},
          "then": {"add": [{"at": {"ref": "x", "index": {"sub": [{"index": "i"}, {"ref": "d"}]}}}, 1]}, "else": 0}}}},
      {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": 0}}}},
      {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
      {"read": {"file": {"ref": "f"}, "len": {"add": [{"ref": "x"}, 1]}, "offset": 0}},
      {"close": {"file": {"ref": "f"}}}
    ]}}
  ]}}
}"#
    .replace("DIST", dist)
    .replace("PARAMS", params)
}

/// The runner clamps a `normal` or `lognormal` to `min` and then to `max`, so a `max` below
/// `min` makes every draw `max`: `min` is no lower bound then, and an `at` offset drawn from
/// such a distribution is not provably >= 1 (rule V3). Nor is a mixture whose every arm is
/// `none`. Both used to pass the check and the first overflowed the VM's stack (RC1, bug 30).
#[test]
fn the_at_rule_believes_no_minimum_a_maximum_undercuts() {
    let refused = |dist: &str, want: &[&str]| {
        let e = aeiou::load_str(&v3_chain(dist, "{}")).err().map(|e| format!("{e:#}")).unwrap_or_else(|| panic!("accepted {dist}"));
        for w in want {
            assert!(e.contains(w), "{dist}: {e}");
        }
    };
    refused(r#"{"normal": {"mean": 5, "sd": 1, "min": 1, "max": 0}}"#, &["normal: min 1 is above max 0", "provably >= 1"]);
    refused(r#"{"lognormal": {"median": 5, "sigma": 1.0, "min": 2.5, "max": 1}}"#, &["lognormal: min 2.5 is above max 1", "provably >= 1"]);
    refused(r#"{"mixture": [{"weight": 1, "dist": null}]}"#, &["provably >= 1"]);
    // a `max` known and not below `min` leaves `min` the bound, as it always was
    assert!(aeiou::load_str(&v3_chain(r#"{"normal": {"mean": 5, "sd": 1, "min": 1, "max": 9}}"#, "{}")).is_ok());
    // a `max` that is a parameter: judged on the default by the check, on the value given by the run
    let text = v3_chain(r#"{"normal": {"mean": 5, "sd": 1, "min": 1, "max": {"param": "hi"}}}"#, r#"{"hi": {"default": 9}}"#);
    let loaded = aeiou::load_str(&text).expect("max 9 is above min 1");
    assert!(Params::new(&loaded.ast, &config(1, 1, &[("hi", "3")])).is_ok());
    let e = Params::new(&loaded.ast, &config(1, 1, &[("hi", "0")])).err().map(|e| format!("{e:#}")).expect("max 0 under min 1 accepted");
    assert!(e.contains("with the parameters given") && e.contains("normal: min 1 is above max 0") && e.contains("provably >= 1"), "{e}");
    // a whole distribution given for a parameter is held to the same rules as a default
    let text = v3_chain(r#"{"param": "back"}"#, r#"{"back": {"default": {"const": 1}}}"#);
    let loaded = aeiou::load_str(&text).unwrap();
    let e = Params::new(&loaded.ast, &config(1, 1, &[("back", r#"{"normal": {"mean": 5, "sd": 1, "min": 1, "max": 0}}"#)])).err().map(|e| format!("{e:#}")).expect("accepted");
    assert!(e.contains("params/back: normal: min 1 is above max 0") && e.contains("provably >= 1"), "{e}");
}

/// Under a proof the checkers got wrong, an `at` that does not go to a smaller index is an
/// error naming the binding and the index, not a recursion that ends in a stack overflow.
#[test]
fn an_at_that_does_not_decrease_is_an_error_not_a_stack_overflow() {
    let text = v3_chain(r#"{"normal": {"mean": 5, "sd": 1, "min": 1, "max": 0}}"#, "{}");
    let ast = aeiou::ast::parse(&text).unwrap(); // the checks skipped, as a wrong proof would
    aeiou::sites::annotate(&ast);
    let cfg = config(1, 1, &[]);
    let params = Params { values: BTreeMap::new() };
    let model = build_model(&ast, &cfg, &params).unwrap();
    let e = run_actor(&model, "gpu", 0, 1, Collect::default()).err().map(|e| format!("{e:#}")).expect("a non-decreasing `at` ran");
    assert!(e.contains("`x @ 1`: index not below 1") && e.contains("rule V3"), "{e}");
}

#[test]
fn a_dataset_is_consumed_in_one_place() {
    // `consume` has no site: its sample is a function of the actor and the enclosing loop
    // indices alone, so two `consume`s of one dataset return the same samples (a reuse the job
    // does not have). The second is refused with check.py's message and path (V6, RC1 review).
    let doc = |actors: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t",
                "datasets": {{"d": {{"layout": "whole", "access": "map", "pattern": "d/f{{id:04}}", "count": 16, "size": {{"const": 1}}, "seed": 1}}}},
                "actors": {{{actors}}}}}"#
        )
    };
    let x = r#"{"let": {"name": "x", "value": {"consume": "d"}}}, {"stat": {"file": {"ref": "x"}}}"#;
    let loop_j = |more: &str| format!(r#"{{"loop": {{"index": "j", "to": 4, "body": [{x}{more}]}}}}"#);
    let loop_k = r#"{"loop": {"index": "k", "to": 2, "body": [{"stat": {"file": {"consume": "d"}}}]}}"#;
    let want = |second: &str, first: &str| format!("t: {second}: a second `consume` of `d` (the first is at {first}): `consume` has no site, so its sample depends only on the actor and the enclosing loop indices, and two of them draw the same samples, a reuse the job does not have; consume once and use the handle in each place, or put the one `consume` in an inner loop so each draw has its own position (V6)");
    // two in one loop body: dry-run gave `stat d/f0006` twice at j = 0, `d/f0000` twice at j = 1, …
    let body = loop_j(r#", {"let": {"name": "y", "value": {"consume": "d"}}}, {"stat": {"file": {"ref": "y"}}}"#);
    rejects(&doc(&format!(r#""gpu": {{"body": [{body}]}}"#)), &want("actors/gpu/body/0/loop/body/2/let/value", "actors/gpu/body/0/loop/body/0/let/value"));
    // two sibling loops, a train phase then an eval phase: both start at position 0
    rejects(&doc(&format!(r#""gpu": {{"body": [{}, {loop_k}]}}"#, loop_j(""))), &want("actors/gpu/body/1/loop/body/0/stat/file", "actors/gpu/body/0/loop/body/0/let/value"));
    // two actors: actor 0 of each draws position 0
    rejects(&doc(&format!(r#""gpu": {{"body": [{}]}}, "eval": {{"body": [{loop_k}]}}"#, loop_j(""))), &want("actors/gpu/body/0/loop/body/0/let/value", "actors/eval/body/0/loop/body/0/stat/file"));
    // one `consume`, its handle used twice: accepted
    aeiou::load_str(&doc(&format!(r#""gpu": {{"body": [{}]}}"#, loop_j(r#", {"stat": {"file": {"ref": "x"}}}"#)))).unwrap();
}

/// A dataset's size below 0 used to pass both checkers and reach `aeiou datagen` cast to
/// `u64`: `{sub: [small, 100]}` at `small` 10 projected 16384 PiB (RC1 review). Rule V25
/// refuses it where the document shows it, with check.py's message and path, on the
/// parameters in effect; the runner refuses a negative draw it could not see, naming the
/// dataset and the sample, instead of a wrap.
#[test]
fn a_dataset_size_is_never_below_zero() {
    let doc = |size: &str, params: &str, layout: &str| {
        let (pattern, extra) = if layout == "array" { ("d/all", r#", "slot": 64"#) } else { ("d/f{id:04}", "") };
        format!(
            r#"{{"ast": "0.13", "name": "t", "params": {params},
                "datasets": {{"d": {{"layout": "{layout}", "access": "indexed", "pattern": "{pattern}", "count": 4, "size": {size}, "seed": 1{extra}}}}},
                "actors": {{"gpu": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "d", "id": 0}}}}}}}}]}}}}}}"#
        )
    };
    let below = |n: i64| format!("t: datasets/d/size: a size below 0: it can be {n} (V25)");
    let unbounded = "t: datasets/d/size: a size below 0: the distribution is unbounded below; give it a `min` of 0 or more (V25)";
    let small = r#"{"small": {"default": 10}}"#;
    rejects(&doc(r#"{"const": {"sub": [{"param": "small"}, 100]}}"#, small, "whole"), &below(-90));
    rejects(&doc(r#"{"sub": [{"param": "small"}, 100]}"#, small, "whole"), &below(-90));
    rejects(&doc(r#"{"const": -1}"#, "{}", "whole"), &below(-1));
    rejects(&doc(r#"{"uniform": {"lo": -5, "hi": 10}}"#, "{}", "whole"), &below(-5));
    rejects(&doc(r#"{"normal": {"mean": 4096, "sd": 100}}"#, "{}", "whole"), unbounded);
    rejects(&doc(r#"{"normal": {"mean": 4096, "sd": 100, "min": -3}}"#, "{}", "whole"), &below(-3));
    rejects(&doc(r#"{"normal": {"mean": 4096, "sd": 100, "min": 1, "max": {"neg": 7}}}"#, "{}", "whole"), &below(-7));
    rejects(&doc(r#"{"lognormal": {"median": -4096, "sigma": 0.5}}"#, "{}", "whole"), unbounded);
    rejects(&doc(r#"{"empirical": {"values": [4096, -2], "weights": [1, 1]}}"#, "{}", "whole"), &below(-2));
    rejects(&doc(r#"{"mixture": [{"weight": 1, "dist": {"const": 4096}}, {"weight": 1, "dist": {"normal": {"mean": 1, "sd": 1}}}]}"#, "{}", "whole"), unbounded);
    rejects(&doc(r#"{"param": "sz"}"#, r#"{"sz": {"default": {"normal": {"mean": 4096, "sd": 9}}}}"#, "whole"), unbounded);
    // bounded at 0 or above, or not shown by the document: accepted
    for ok in [
        r#"{"normal": {"mean": 4096, "sd": 100, "min": 0}}"#,
        r#"{"normal": {"mean": 4096, "sd": 0}}"#,
        r#"{"lognormal": {"median": 4096, "sigma": 0.5}}"#,
        r#"{"const": {"sub": [{"param": "small"}, 10]}}"#,
        r#"{"const": -0.4}"#,
        r#"{"mixture": [{"weight": 0, "dist": {"const": -1}}, {"weight": 1, "dist": {"const": 4096}}]}"#,
        r#"{"normal": {"mean": 10, "sd": 100, "min": {"div": [-100, 1]}}}"#,
    ] {
        aeiou::load_str(&doc(ok, small, "whole")).unwrap_or_else(|e| panic!("{ok}: {e:#}"));
    }
    // an array's sizes are clamped to [0, slot] by the runner, so the rule leaves them alone
    aeiou::load_str(&doc(r#"{"normal": {"mean": 32, "sd": 100}}"#, "{}", "array")).unwrap();
    // a parameter given at run time is judged again
    let text = doc(r#"{"const": {"sub": [{"param": "small"}, 100]}}"#, r#"{"small": {"default": 200}}"#, "whole");
    let loaded = aeiou::load_str(&text).unwrap();
    let e = Params::new(&loaded.ast, &config(1, 1, &[("small", "10")])).err().map(|e| format!("{e:#}")).expect("size -90 accepted");
    assert!(e.contains("with the parameters given") && e.contains(&below(-90)), "{e}");
    // what the document does not show: the draw is refused, not cast to some 16 EiB
    let text = doc(r#"{"normal": {"mean": 10, "sd": 100, "min": {"div": [-100, 1]}}}"#, "{}", "whole");
    let loaded = aeiou::load_str(&text).unwrap();
    let cfg = config(1, 1, &[]);
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let model = build_model(&loaded.ast, &cfg, &params).unwrap();
    let meta = &model.datasets[model.ds_index["d"]];
    let sizes: Vec<Result<i64, String>> = (0..4).map(|id| meta.sample_size(id).map_err(|e| format!("{e:#}"))).collect();
    assert!(sizes.iter().any(|s| s.is_err()), "no sample of 4 drew below 0: {sizes:?}");
    for (id, s) in sizes.iter().enumerate() {
        match s {
            Ok(n) => assert!(*n >= 0, "sample {id}: {n}"),
            Err(e) => assert!(e.contains(&format!("sample {id} of dataset `d` has size -")) && e.contains("below 0"), "{e}"),
        }
    }
    assert!(meta.file_size(sizes.iter().position(|s| s.is_err()).unwrap() as i64).is_err());
}

#[test]
fn integers_outside_64_bits_are_refused_as_check_py_refuses_them() {
    // The contract's integers are i64, a seed u64 (schema/README.md §1). serde_json parses an
    // integer past u64 as a float, so such a document hashed and ran differently here than in
    // Python, which keeps it exact; one between i64 and u64 became a float literal.
    let doc = |n: &str, seed: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t", "params": {{"n": {{"default": {n}}}, "a": {{"default": [1, 3]}}}},
                "datasets": {{"d": {{"layout": "whole", "access": "map", "pattern": "d/{{id}}", "count": 10, "size": {{"const": 1}}, "seed": {seed}}}}},
                "actors": {{"gpu": {{"body": [{{"compute": {{"ns": {{"add": [1, 1.5e30]}}}}}}]}}}}}}"#
        )
    };
    let range = "is outside the 64-bit signed range of the contract's integers (-2^63 to 2^63-1)";
    aeiou::load_str(&doc("9223372036854775807", "18446744073709551615")).unwrap();
    aeiou::load_str(&doc("-9223372036854775808", "0")).unwrap();
    rejects(&doc("9223372036854775808", "1"), &format!("t: params/n/default: integer 9223372036854775808 {range}"));
    rejects(&doc("18446744073709551616", "1"), &format!("t: params/n/default: integer 18446744073709551616 {range}"));
    rejects(&doc("-9223372036854775809", "1"), &format!("t: params/n/default: integer -9223372036854775809 {range}"));
    rejects(&doc("1000000000000000000000000000000000000000000", "1"), "integer 1000000000000000000000000000000000000000000 is outside");
    rejects(&doc("1", "18446744073709551616"), "t: datasets/d/seed: seed 18446744073709551616 is outside 0 to 2^64-1");
    rejects(&doc("1", "-1"), "t: datasets/d/seed: seed -1 is outside 0 to 2^64-1");
    // inside arrays and expressions, with the path to it
    rejects(&doc("1", "1").replace("[1, 3]", "[1, 2, -99999999999999999999]"), "t: params/a/default/2: integer -99999999999999999999 is outside");
    rejects(&doc("1", "1").replace("[1, 1.5e30]", "[1, 36893488147419103232]"), "t: actors/gpu/body/0/compute/ns/add/1: integer 36893488147419103232 is outside");
    // a parameter file and --param: the same rule on the values a run is given
    use aeiou::eval::ParamSet;
    let e = ParamSet::parse(r#"{"params_version":2,"abstract":"t","params":{"n":{"default":18446744073709551616}}}"#, "x").err().map(|e| format!("{e:#}")).unwrap();
    assert!(e.contains(&format!("params/n/default: integer 18446744073709551616 {range}")), "{e}");
    let loaded = aeiou::load_str(&doc("1", "1")).unwrap();
    let e = Params::new(&loaded.ast, &config(1, 1, &[("n", "9223372036854775808")])).err().map(|e| format!("{e:#}")).expect("accepted");
    assert_eq!(e, format!("--param n: integer 9223372036854775808 {range}"));
    assert!(Params::new(&loaded.ast, &config(1, 1, &[("n", "9223372036854775807")])).is_ok());
}

/// `x @ i` names the binding the checkers resolve (rule V3, "which binding"): a `let` of the
/// body the `at` is written in, before or after it, shadows an outer `let` of the same name;
/// the definition re-evaluated reads its own names where it is written. Until 2026-10-14 the
/// runner took the outer binding, at the outer loop's index, while both checkers proved V3
/// on the inner one; and a re-evaluated definition read a later sibling, or a binding of the
/// body the `at` was evaluated from, in place of the one in its scope (RC1 review).
#[test]
fn at_names_the_binding_its_body_shadows() {
    let paths = |text: &str| -> Vec<String> { collect(text, &config(1, 1, &[]), "gpu", 1).into_iter().map(|o| o.3).collect() };
    // the inner `x` counts j; `y` is the inner `x` one index back. The outer `x` is 1000·i.
    let shadow = r#"{"ast": "0.13", "name": "shadow",
 "namespaces": {"o": {"pattern": "o/{v}/{w}", "fields": {"v": "int", "w": "int"}, "size": 1, "seed": 3}},
 "actors": {"gpu": {"count": 1, "body": [
  {"loop": {"index": "i", "to": 3, "body": [
    {"let": {"name": "x", "value": {"mul": [{"index": "i"}, 1000]}}},
    {"loop": {"index": "j", "to": 4, "body": [
      {"let": {"name": "y", "value": {"cond": {"if": {"eq": [{"index": "j"}, 0]}, "then": -1, "else": {"at": {"ref": "x", "index": {"sub": [{"index": "j"}, 1]}}}}}}},
      {"let": {"name": "x", "value": {"cond": {"if": {"eq": [{"index": "j"}, 0]}, "then": 0, "else": {"add": [{"at": {"ref": "x", "index": {"sub": [{"index": "j"}, 1]}}}, 1]}}}}},
      {"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"ref": "x"}, "w": {"ref": "y"}}}}}}
    ]}}
  ]}}
 ]}}}"#;
    let want: Vec<String> = (0..3).flat_map(|_| (0..4).map(|j| format!("o/{j}/{}", if j == 0 { -1 } else { j - 1 }))).collect();
    assert_eq!(paths(shadow), want);
    // `z` reads the outer `x` (the inner one comes after it), so `z @ (j-1)` is 1000·i + j-1;
    // `k` reads the outer `c` = 7, so `k @ (i-1)` is i-1 + 7, whatever `c` the inner body binds.
    let scope = r#"{"ast": "0.13", "name": "shadow2",
 "namespaces": {"o": {"pattern": "o/{v}/{w}", "fields": {"v": "int", "w": "int"}, "size": 1, "seed": 3}},
 "actors": {"gpu": {"count": 1, "body": [
  {"let": {"name": "c", "value": 7}},
  {"loop": {"index": "i", "to": 2, "body": [
    {"let": {"name": "x", "value": {"mul": [{"index": "i"}, 1000]}}},
    {"let": {"name": "k", "value": {"add": [{"index": "i"}, {"ref": "c"}]}}},
    {"loop": {"index": "j", "to": 3, "body": [
      {"let": {"name": "z", "value": {"add": [{"ref": "x"}, {"index": "j"}]}}},
      {"let": {"name": "x", "value": {"mul": [{"index": "j"}, 10]}}},
      {"let": {"name": "c", "value": 500}},
      {"let": {"name": "w", "value": {"cond": {"if": {"eq": [{"index": "i"}, 0]}, "then": -1, "else": {"at": {"ref": "k", "index": {"sub": [{"index": "i"}, 1]}}}}}}},
      {"let": {"name": "v", "value": {"cond": {"if": {"eq": [{"index": "j"}, 0]}, "then": -1, "else": {"at": {"ref": "z", "index": {"sub": [{"index": "j"}, 1]}}}}}}},
      {"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"ref": "v"}, "w": {"ref": "w"}}}}}}
    ]}}
  ]}}
 ]}}}"#;
    let want: Vec<String> = (0..2i64)
        .flat_map(|i| (0..3i64).map(move |j| format!("o/{}/{}", if j == 0 { -1 } else { 1000 * i + j - 1 }, if i == 0 { -1 } else { i - 1 + 7 })))
        .collect();
    assert_eq!(paths(scope), want);
    // a body that binds `x` twice: which one `x @` reads is not defined, so it is refused
    let twice = r#"{"ast": "0.13", "name": "t",
 "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": 1, "seed": 3}},
 "actors": {"gpu": {"count": 1, "body": [
  {"loop": {"index": "i", "to": 3, "body": [
    {"let": {"name": "x", "value": {"index": "i"}}},
    {"let": {"name": "x", "value": {"cond": {"if": {"eq": [{"index": "i"}, 0]}, "then": 0, "else": {"at": {"ref": "x", "index": {"sub": [{"index": "i"}, 1]}}}}}}},
    {"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"ref": "x"}}}}}}
  ]}}
 ]}}}"#;
    rejects(twice, "`x @` names a binding this body defines more than once");
}

/// `consume`'s position is the row-major ordinal of its enclosing frames at fixed counts
/// (`runner/REFERENCE.md` §2), so a loop around the `consume` whose count varies with an
/// enclosing loop repeats positions: `loop i < 3 { loop j < 3 - i { stat consume d } }`
/// statted `d/f0002` at (0, 2), (1, 0), and (2, 0). Both checkers refuse it, with check.py's
/// message and path, naming the loop (V6, RC1 review).
#[test]
fn a_consume_is_under_loops_of_fixed_count() {
    let doc = |body: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t", "params": {{"n": {{"default": 3}}}},
                "datasets": {{"d": {{"layout": "whole", "access": "map", "pattern": "d/f{{id:04}}", "count": 64, "size": {{"const": 1}}, "seed": 1}}}},
                "actors": {{"gpu": {{"body": [{body}]}}}}}}"#
        )
    };
    let c = r#"{"stat": {"file": {"consume": "d"}}}"#;
    let want = |at: &str, lp: &str| format!("t: {at}: a `consume` of `d` under a loop whose count varies with an enclosing loop ({lp}): its position is the row-major ordinal of the enclosing loops at fixed counts, so a count that varies repeats positions and draws the same samples, a reuse the job does not have; give the loop a fixed count (the most it can be, with a `cond`, the builder's `when`, inside to skip the rest) (V6)");
    let l0 = "actors/gpu/body/0/loop/body/0";
    let l1 = "actors/gpu/body/0/loop/body/1";
    // the count is an expression of the outer index
    rejects(&doc(&format!(r#"{{"loop": {{"index": "i", "to": 3, "body": [{{"loop": {{"index": "j", "to": {{"sub": [3, {{"index": "i"}}]}}, "body": [{c}]}}}}]}}}}"#)), &want(&format!("{l0}/loop/body/0/stat/file"), &format!("{l0}/loop")));
    // its `from`, through a binding of the outer index
    rejects(&doc(&format!(r#"{{"loop": {{"index": "i", "to": 3, "body": [{{"let": {{"name": "m", "value": {{"index": "i"}}}}}}, {{"loop": {{"index": "j", "from": {{"ref": "m"}}, "to": 3, "body": [{c}]}}}}]}}}}"#)), &want(&format!("{l1}/loop/body/0/stat/file"), &format!("{l1}/loop")));
    // a draw inside the outer loop, keyed on its index
    rejects(&doc(&format!(r#"{{"loop": {{"index": "i", "to": 3, "body": [{{"let": {{"name": "m", "value": {{"draw": {{"uniform": {{"lo": 1, "hi": 4}}}}}}}}}}, {{"loop": {{"index": "j", "to": {{"ref": "m"}}, "body": [{c}]}}}}]}}}}"#)), &want(&format!("{l1}/loop/body/0/stat/file"), &format!("{l1}/loop")));
    // a `parallel` width, with the `consume` one loop further in
    rejects(&doc(&format!(r#"{{"loop": {{"index": "i", "to": 3, "body": [{{"parallel": {{"index": "w", "width": {{"add": [{{"index": "i"}}, 1]}}, "body": [{{"loop": {{"index": "j", "to": 2, "body": [{c}]}}}}]}}}}]}}}}"#)), &want(&format!("{l0}/parallel/body/0/loop/body/0/stat/file"), &format!("{l0}/parallel")));
    // accepted: the most it can be with a `cond` inside; a count fixed for the actor (a
    // parameter, a draw at the actor's top); a varying loop inside the `consume`'s, not around it
    for ok in [
        format!(r#"{{"loop": {{"index": "i", "to": 3, "body": [{{"loop": {{"index": "j", "to": 3, "body": [{{"cond": {{"if": {{"lt": [{{"index": "j"}}, {{"sub": [3, {{"index": "i"}}]}}]}}, "then": [{c}]}}}}]}}}}]}}}}"#),
        format!(r#"{{"let": {{"name": "m", "value": {{"draw": {{"uniform": {{"lo": 1, "hi": 4}}}}}}}}}}, {{"loop": {{"index": "i", "to": 3, "body": [{{"loop": {{"index": "j", "to": {{"add": [{{"ref": "m"}}, {{"param": "n"}}]}}, "body": [{c}]}}}}]}}}}"#),
        r#"{"loop": {"index": "i", "to": 3, "body": [{"let": {"name": "x", "value": {"consume": "d"}}}, {"loop": {"index": "j", "to": {"add": [{"index": "i"}, 1]}, "body": [{"stat": {"file": {"ref": "x"}}}]}}]}}"#.to_string(),
    ] {
        aeiou::load_str(&doc(&ok)).unwrap_or_else(|e| panic!("{ok}: {e:#}"));
    }
}

/// Integer arithmetic on what an abstract and its parameters give (loop trip counts, a file's
/// size summed from its samples, offsets, an array's length, an inventory's size) is checked:
/// past 2^63-1 the run is refused with the dataset or op named. Until 2026-10-15 the release
/// build wrapped it silently: a 16 EiB file was 0 bytes, a loop across the signs ran no
/// iteration, a write past 2^63 landed at a negative offset (RC1 review).
#[test]
fn integer_arithmetic_past_64_bits_is_refused_not_wrapped() {
    // the error of building the model, else of walking the actor; the ops when there is none
    let walk = |text: &str| -> Result<Vec<(i64, Vec<i64>, OpKind, String, i64, i64, i64)>, String> {
        let loaded = aeiou::load_str(text).expect("valid abstract");
        let cfg = config(1, 0, &[]);
        let params = Params::new(&loaded.ast, &cfg).map_err(|e| format!("{e:#}"))?;
        let model = build_model(&loaded.ast, &cfg, &params).map_err(|e| format!("{e:#}"))?;
        run_actor(&model, "gpu", 0, 1, Collect::default()).map(|s| s.ops).map_err(|e| format!("{e:#}"))
    };
    let refused = |text: &str, needle: &str| match walk(text) {
        Ok(ops) => panic!("ran where `{needle}` was wanted: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    };
    let doc = |datasets: &str, body: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t", "params": {{"min": {{"default": -9223372036854775808}}}}, "datasets": {{{datasets}}},
                "namespaces": {{"o": {{"pattern": "o/{{v}}", "fields": {{"v": "int"}}, "size": "as_written", "seed": 3}}}},
                "actors": {{"gpu": {{"count": 1, "body": [{body}]}}}}}}"#
        )
    };
    let big = 1i64 << 62;
    // four samples of 2^62 bytes are a file of 2^64: refused, not 0 bytes
    let file = r#"{"let": {"name": "h", "value": {"file": {"dataset": "d", "id": 0}}}}, {"open": {"file": {"ref": "h"}, "flags": ["RDONLY"]}}, {"read": {"file": {"ref": "h"}, "len": {"size": {"ref": "h"}}}}"#;
    let compound = format!(r#""d": {{"layout": "compound", "access": "stream", "pattern": "d/f{{id}}", "count": 4, "samples_per_file": 4, "size": {{"const": {big}}}, "seed": 1}}"#);
    refused(&doc(&compound, file), "dataset `d`: the size of file 0 overflows the 64-bit integers of the contract");
    // one such sample fits: that file is read whole
    let fits = compound.replace(r#""count": 4, "samples_per_file": 4"#, r#""count": 1, "samples_per_file": 1"#);
    assert_eq!(walk(&doc(&fits, file)).unwrap()[1].6, big);
    // an array of 4 slots of 2^62 bytes, an inventory of 2^62 entries of 4 bytes
    refused(&doc(&format!(r#""d": {{"layout": "array", "access": "indexed", "pattern": "d/all", "count": 4, "slot": {big}, "size": {{"const": 4096}}, "seed": 1}}"#), ""), "dataset `d`: its file, count 4 × slot 4611686018427387904 bytes, overflows");
    refused(
        &doc(&format!(r#""d": {{"layout": "whole", "access": "map", "pattern": "d/{{id}}", "count": {big}, "size": {{"const": 1}}, "seed": 1, "inventory": {{"name": "inv", "entry": 4}}}}"#), ""),
        "dataset `d`: its inventory, count 4611686018427387904 × entry 4 bytes, overflows",
    );
    // a loop from -(2^63-1) to 2^63-1 by 2^62 is four iterations, not none
    let stat = r#"{"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"index": "i"}}}}}}"#;
    let ops = walk(&doc("", &format!(r#"{{"loop": {{"index": "i", "from": -9223372036854775807, "to": 9223372036854775807, "step": {big}, "body": [{stat}]}}}}"#))).unwrap();
    let idx: Vec<i64> = ops.iter().map(|o| o.1[0]).collect();
    assert_eq!(idx, vec![-i64::MAX, -i64::MAX + big, 1, 1 + big]);
    // a write that reaches past 2^63-1, and an lseek past it
    let obj = r#"{"let": {"name": "h", "value": {"object": {"namespace": "o", "fields": {"v": 0}}}}}, {"open": {"file": {"ref": "h"}, "flags": ["WRONLY", "CREAT"]}}"#;
    refused(&doc("", &format!(r#"{obj}, {{"lseek": {{"file": {{"ref": "h"}}, "offset": 9223372036854775800, "whence": "SET"}}}}, {{"write": {{"file": {{"ref": "h"}}, "len": 100}}}}"#)), "write: offset 9223372036854775800 and length 100 reach outside 0 to 2^63-1");
    refused(&doc("", &format!(r#"{obj}, {{"write": {{"file": {{"ref": "h"}}, "len": 100, "offset": 9223372036854775800}}}}"#)), "write: offset 9223372036854775800 and length 100 reach outside");
    refused(&doc("", &format!(r#"{obj}, {{"write": {{"file": {{"ref": "h"}}, "len": 10}}}}, {{"lseek": {{"file": {{"ref": "h"}}, "offset": 9223372036854775800, "whence": "CUR"}}}}"#)), "lseek by 9223372036854775800 on `o/0`: the position passes 2^63-1");
    refused(&doc("", &format!(r#"{obj}, {{"write": {{"file": {{"ref": "h"}}, "len": {big}, "repeat": 3}}}}"#)), "write: offset 4611686018427387904 and length 4611686018427387904 reach outside");
    // -2^63 div -1 and its kin: an error, not a panic; a float past 2^63 is not 2^63-1
    let field = |e: &str| format!(r#"{{"stat": {{"file": {{"object": {{"namespace": "o", "fields": {{"v": {e}}}}}}}}}}}"#);
    refused(&doc("", &field(r#"{"div": [{"param": "min"}, -1]}"#)), "div: overflow");
    refused(&doc("", &field(r#"{"mod": [{"param": "min"}, -1]}"#)), "mod: overflow");
    refused(&doc("", &field(r#"{"ceil_div": [{"param": "min"}, 1]}"#)), "ceil_div: overflow");
    refused(&doc("", &field(r#"{"add": [1e19, 0]}"#)), "float 10000000000000000000 where an integer is needed is outside the 64-bit integers");
}

/// Rule V25 rounds a size as the runner rounds a draw (`f64::round`) and prints every digit of
/// the rounded value, as check.py does: check.py's `int(abs(x) + 0.5)` refused -0.49999999999999994
/// (it rounds to 0 here) and printed -(2^52 + 1) as -(2^52 + 2); this side saturated -1e20 at
/// i64::MIN. A size that is no finite number (`mul` past f64, `inf - inf`) is refused in both
/// with one message, where check.py raised a traceback and this side passed the NaN (RC1 review).
#[test]
fn a_dataset_size_is_judged_and_printed_as_check_py_judges_and_prints_it() {
    let doc = |size: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t",
                "datasets": {{"d": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": {size}, "seed": 1}}}},
                "actors": {{"gpu": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "d", "id": 0}}}}}}}}]}}}}}}"#
        )
    };
    let below = |n: &str| format!("t: datasets/d/size: a size below 0: it can be {n} (V25)");
    let infinite = "t: datasets/d/size: a size that is not a finite number (V25)";
    aeiou::load_str(&doc("-0.49999999999999994")).unwrap_or_else(|e| panic!("{e:#}"));
    rejects(&doc("-0.5"), &below("-1"));
    rejects(&doc("-4503599627370497.0"), &below("-4503599627370497"));
    rejects(&doc("-1e20"), &below("-100000000000000000000"));
    rejects(&doc(r#"{"mul": [-1e308, 10]}"#), infinite);
    rejects(&doc(r#"{"mul": [1e308, 10]}"#), infinite);
    rejects(&doc(r#"{"add": [{"mul": [1e308, 10]}, {"mul": [-1e308, 10]}]}"#), infinite);
}

/// `min` above `max` on a `normal` or `lognormal` is judged as the runner's clamp compares,
/// in f64, and its values printed as check.py prints them: this side printed Rust's Debug
/// (`1e20`, `1e-5`), check.py compared integers exactly and refused 2^53 + 1 over 2^53, which
/// are one f64 to the clamp (RC1 review). The same comparison holds where rule V3 believes a
/// `min` only under a `max` not below it.
#[test]
fn min_above_max_is_judged_and_printed_as_check_py_judges_and_prints_it() {
    let doc = |size: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t",
                "datasets": {{"d": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": {size}, "seed": 1}}}},
                "actors": {{"gpu": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "d", "id": 0}}}}}}}}]}}}}}}"#
        )
    };
    rejects(&doc(r#"{"normal": {"mean": 0, "sd": 1, "min": 1e20, "max": 1e19}}"#), "t: datasets/d/size: normal: min 1e+20 is above max 1e+19");
    rejects(&doc(r#"{"lognormal": {"median": 1, "sigma": 1.0, "min": 1e-5, "max": 1e-7}}"#), "t: datasets/d/size: lognormal: min 1e-05 is above max 1e-07");
    aeiou::load_str(&doc(r#"{"normal": {"mean": 0, "sd": 1, "min": 9007199254740993, "max": 9007199254740992}}"#)).unwrap_or_else(|e| panic!("{e:#}"));
    aeiou::load_str(&v3_chain(r#"{"normal": {"mean": 5, "sd": 1, "min": 9007199254740993, "max": 9007199254740992}}"#, "{}")).unwrap_or_else(|e| panic!("{e:#}"));
}

/// A size whose least value is not a number (`inf − inf`) is judged alike by both checkers
/// whatever the order of a `mixture`'s arms or of a `min`'s or `max`'s operands, and in the
/// clamp of a `normal`: `f64::min` dropped the NaN here (ok, or "it can be -3") where
/// check.py's `min` answered by argument order (RC1 review). An arm of `+inf` is no finite
/// size either, and outranks an arm unbounded below.
#[test]
fn a_size_that_is_not_a_number_is_refused_whatever_the_order() {
    let doc = |size: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t",
                "datasets": {{"d": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": {size}, "seed": 1}}}},
                "actors": {{"gpu": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "d", "id": 0}}}}}}}}]}}}}}}"#
        )
    };
    let nan = r#"{"add": [{"mul": [1e308, 10]}, {"mul": [-1e308, 10]}]}"#;
    let arm = |x: &str| format!(r#"{{"weight": 1, "dist": {{"const": {x}}}}}"#);
    let normal = r#"{"weight": 1, "dist": {"normal": {"mean": 0, "sd": 1}}}"#;
    let infinite = "t: datasets/d/size: a size that is not a finite number (V25)";
    for (a, b) in [(arm(nan), arm("5")), (arm(nan), arm("-3")), (arm(r#"{"mul": [1e308, 10]}"#), arm("5")), (arm(nan), normal.to_string())] {
        rejects(&doc(&format!(r#"{{"mixture": [{a}, {b}]}}"#)), infinite);
        rejects(&doc(&format!(r#"{{"mixture": [{b}, {a}]}}"#)), infinite);
    }
    for (op, x) in [("min", "5"), ("max", "-5")] {
        rejects(&doc(&format!(r#"{{"{op}": [{nan}, {x}]}}"#)), infinite);
        rejects(&doc(&format!(r#"{{"{op}": [{x}, {nan}]}}"#)), infinite);
    }
    rejects(&doc(&format!(r#"{{"normal": {{"mean": {nan}, "sd": 0, "min": 5}}}}"#)), infinite);
    rejects(&doc(&format!(r#"{{"normal": {{"mean": {nan}, "sd": 0, "max": -5}}}}"#)), infinite);
    rejects(&doc(&format!(r#"{{"normal": {{"mean": 0, "sd": 1, "min": {nan}, "max": 5}}}}"#)), infinite);
    // a weight of 0 is never drawn
    aeiou::load_str(&doc(&format!(r#"{{"mixture": [{{"weight": 0, "dist": {{"const": {nan}}}}}, {}]}}"#, arm("5")))).unwrap_or_else(|e| panic!("{e:#}"));
}

/// V26 (RC1 regression review), as `check.py` has it: a `channel` or `loader` inside a
/// `parallel` or `loader` body is refused, at any depth; one declared by the actor and used by
/// its sub-actors, or declared again in each iteration of the actor's loop, is not.
#[test]
fn a_channel_or_loader_inside_a_fork_is_refused() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"v26","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let chan = r#"{"channel":{"name":"c","capacity":2}}"#;
    let loader = |body: &str| format!(r#"{{"loader":{{"name":"q","index":"b","workers":2,"prefetch":1,"batches":2,"body":[{body}]}}}}"#);
    let par = |body: &str| format!(r#"{{"parallel":{{"index":"p","width":2,"body":[{body}]}}}}"#);
    let put_take = r#"{"put":{"channel":"c","seq":{"index":"p"}}}, {"take":{"channel":"c"}}"#;
    let take_q = r#"{"loop":{"index":"s","to":2,"body":[{"take":{"channel":"q"}}]}}"#;
    // accepted: the actor declares, its sub-actors put and take; a channel and a loader per iteration
    aeiou::load_str(&doc(&format!("{chan}, {}", par(put_take)))).unwrap();
    aeiou::load_str(&doc(&format!(r#"{{"loop":{{"index":"e","to":2,"body":[{chan}, {}, {}, {take_q}]}}}}"#, loader(r#"{"compute":{"ns":1}}"#), par(put_take)))).unwrap();
    for (body, path) in [
        (par(&format!("{chan}, {put_take}")), "parallel/body/0/channel"),
        (format!("{}, {take_q}", loader(chan)), "loader/body/0/channel"),
        (par(&format!("{}, {take_q}", loader(r#"{"compute":{"ns":1}}"#))), "parallel/body/0/loader"),
        (format!("{}, {take_q}", loader(&par(chan))), "loader/body/0/parallel/body/0/channel"),
    ] {
        let kind = path.rsplit('/').next().unwrap();
        rejects(&doc(&body), &format!("{path}: a `{kind}` inside a `parallel` or `loader` body"));
        rejects(&doc(&body), "(V26)");
    }
}

/// V26 (RC1 regression review), as `check.py` has it: a channel a loader's workers put or take
/// on is declared at one site, outside every loop or in the body the loader is in, the loader
/// declared at one site; the workers run beside the actor, so one declared again under them
/// was the old channel or the new one as timing had it.
#[test]
fn a_channel_a_loaders_workers_use_is_declared_once_or_beside_it() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"v26","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let chan = r#"{"channel":{"name":"c","capacity":2}}"#;
    let loader = |name: &str, body: &str| format!(r#"{{"loader":{{"name":"{name}","index":"b","workers":2,"prefetch":1,"batches":2,"body":[{body}]}}}}"#);
    let lput = loader("q", r#"{"put":{"channel":"c","seq":{"index":"b"}}}"#);
    let ltake = loader("q", r#"{"take":{"channel":"c"}}"#);
    let lp = |body: &str| format!(r#"{{"loop":{{"index":"e","to":2,"body":[{body}]}}}}"#);
    let compute = r#"{"compute":{"ns":1}}"#;
    // accepted: declared once, outside every loop; declared beside the loader in its loop
    aeiou::load_str(&doc(&format!("{chan}, {}", lp(&lput)))).unwrap();
    aeiou::load_str(&doc(&lp(&format!("{chan}, {lput}")))).unwrap();
    aeiou::load_str(&doc(&format!(r#"{{"cond":{{"if":1,"then":[{chan}]}}}}, {}"#, lp(&lput)))).unwrap();
    for (body, path, kind, ch) in [
        (format!("{chan}, {lput}, {}", lp(chan)), "1/loader/body/0", "put", "c"),
        (format!("{}, {lput}", lp(chan)), "1/loader/body/0", "put", "c"),
        (format!("{chan}, {lput}, {chan}"), "1/loader/body/0", "put", "c"),
        (format!("{}, {ltake}", lp(chan)), "1/loader/body/0", "take", "c"),
        (lp(&format!(r#"{chan}, {{"cond":{{"if":1,"then":[{lput}]}}}}"#)), "0/loop/body/1/cond/then/0/loader/body/0", "put", "c"),
        (format!("{}, {}", loader("q", compute), lp(&format!("{chan}, {lput}"))), "1/loop/body/1/loader/body/0", "put", "c"),
        (format!("{}, {}", lp(&loader("r", compute)), loader("q", r#"{"take":{"channel":"r"}}"#)), "1/loader/body/0", "take", "r"),
    ] {
        rejects(&doc(&body), &format!("actors/gpu/body/{path}/{kind}: a `{kind}` on channel `{ch}` inside a `loader` body"));
        rejects(&doc(&body), "would depend on timing");
        rejects(&doc(&body), "(V26)");
    }
}

/// A `take` inside a `loader` body, at any depth, is refused (V26, 2026-10-15): the loader's
/// workers run beside each other and the actor, so which worker took the item the actor put,
/// and whether the run finished, was timing. The RC1 review's R20, `channel c; loader q {take c};
/// loop {put c; take q}`, passed both checkers with `c` declared once (bug 77's rule), then
/// failed 10 in 10 runs under `sync` at `--time-scale 0` and 9 in 10 at 1, and passed under
/// io_uring at 1. A loader's body still puts (bug 77's rule).
#[test]
fn a_take_inside_a_loader_body_is_refused() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"v26","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let chan = r#"{"channel":{"name":"c","capacity":4}}"#;
    let loader = |name: &str, body: &str| format!(r#"{{"loader":{{"name":"{name}","index":"b","workers":2,"prefetch":1,"batches":8,"body":[{body}]}}}}"#);
    let take = r#"{"take":{"channel":"c"}}"#;
    let feed = r#"{"loop":{"index":"j","to":8,"body":[{"put":{"channel":"c","seq":{"index":"j"}}},{"take":{"channel":"q"}}]}}"#;
    // accepted: the loader's body puts, the actor takes
    aeiou::load_str(&doc(&format!(r#"{chan}, {}, {{"loop":{{"index":"j","to":8,"body":[{{"take":{{"channel":"q"}}}},{take}]}}}}"#, loader("q", r#"{"put":{"channel":"c","seq":{"index":"b"}}}"#)))).unwrap();
    for (body, path, ch) in [
        // R20 as the reviewer had it
        (format!(r#"{chan}, {}, {feed}"#, loader("q", &format!(r#"{{"compute":{{"ns":1000}}}}, {take}"#))), "1/loader/body/1", "c"),
        // deeper: in a loop and a cond, or a `parallel`, of the loader's body
        (format!(r#"{chan}, {}, {feed}"#, loader("q", &format!(r#"{{"loop":{{"index":"i","to":1,"body":[{{"cond":{{"if":1,"then":[{take}]}}}}]}}}}"#))), "1/loader/body/0/loop/body/0/cond/then/0", "c"),
        (format!(r#"{chan}, {}, {feed}"#, loader("q", &format!(r#"{{"parallel":{{"index":"p","width":1,"body":[{take}]}}}}"#))), "1/loader/body/0/parallel/body/0", "c"),
        // another loader's channel, declared once
        (format!(r#"{}, {}, {{"loop":{{"index":"j","to":8,"body":[{{"take":{{"channel":"q"}}}}]}}}}"#, loader("r", r#"{"compute":{"ns":1}}"#), loader("q", r#"{"take":{"channel":"r"}}"#)), "1/loader/body/0", "r"),
    ] {
        rejects(&doc(&body), &format!("actors/gpu/body/{path}/take: a `take` on channel `{ch}` inside a `loader` body: the loader's workers run beside each other and the actor, so which of them took which item, and whether the run finished, would depend on timing; take in the actor, a loader's body only puts (V26)"));
    }
}

/// A float literal is read to the nearest f64, as Python's `json` reads it: serde_json without
/// `float_roundtrip` read 4503599627370495.5, 9007199254740993.0, and 123456789.12345679 one ULP
/// off, so the AST's sha256 differed from check.py's (a parameter file pinned to check.py's hash
/// was refused by `dry-run --params-file`) and a `max` printed as 4503599627370495.0 (RC1 review).
/// The hashes are check.py's, 2026-10-15 (re-recorded for each contract version since).
#[test]
fn float_literals_are_read_as_check_py_reads_them() {
    let doc = |size: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "t", "datasets": {{"d": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": {size}, "seed": 1}}}}, "actors": {{"gpu": {{"body": [{{"stat": {{"file": {{"file": {{"dataset": "d", "id": 0}}}}}}}}]}}}}}}"#
        )
    };
    for (x, sha) in [
        ("4503599627370495.5", "ea11781d0078b2a8bb73751dd2c38ff90dbdfb0e9199c34f3db414a54a43358b"),
        ("9007199254740993.0", "bdaca37d8cb8a4d964528684e33448c0ef9e5acf348e8b357b4a89c3d624596f"),
        ("123456789.12345679", "e0c40e775e3363d9b896c34344318052734f6f04ebb76f9fb12e0d18c0be86ce"),
    ] {
        let loaded = aeiou::load_str(&doc(&format!(r#"{{"const": {x}}}"#))).unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(loaded.sha256, sha, "{x}");
    }
    rejects(
        &doc(r#"{"normal": {"mean": 0, "sd": 1, "min": 9007199254740992, "max": 4503599627370495.5}}"#),
        "t: datasets/d/size: normal: min 9007199254740992 is above max 4503599627370495.5",
    );
}

/// V27 (RC1 review, R18), as `check.py` has it: a value whose kind the document shows (a
/// literal, a parameter by its value in effect, a binding by its value) where the runner
/// refuses that kind is refused at check time, not when the run reaches it: `"count": "gpus"`
/// passed both checkers and failed `run` with "expected an integer, got str". A float or a bool
/// where an integer is needed is read as the runner reads it, and a kind the document does not
/// show (a draw, a `cond` whose arms differ) is left to the run.
#[test]
fn a_value_of_a_kind_the_runner_refuses_is_refused_at_check() {
    let doc = |count: &str, body: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "v27",
                "params": {{"s": {{"default": "x"}}, "n": {{"default": 2}}, "d": {{"default": {{"uniform": {{"lo": 1, "hi": 4}}}}}}, "a": {{"default": [1, 2]}}}},
                "datasets": {{"ds": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": 10, "seed": 1}}}},
                "namespaces": {{"o": {{"pattern": "o/{{name}}-{{k:03}}", "fields": {{"name": "str", "k": "int"}}, "size": 10, "seed": 2}}}},
                "actors": {{"gpu": {{"count": {count}, "body": [{body}]}}}}}}"#
        )
    };
    let compute = |ns: &str| format!(r#"{{"compute": {{"ns": {ns}}}}}"#);
    // accepted: what the runner reads as an integer, and what the document does not show
    for (count, body) in [
        ("1.5", compute("true")),
        (r#"{"param": "n"}"#, compute(r#"{"draw": {"param": "d"}}"#)),
        (r#"{"actor": "count"}"#, compute(r#"{"cond": {"if": true, "then": 1, "else": "x"}}"#)),
        ("1", format!(r#"{{"let": {{"name": "x", "value": 3}}}}, {}"#, compute(r#"{"add": [{"ref": "x"}, 1.5]}"#))),
        ("1", r#"{"open": {"file": {"object": {"namespace": "o", "fields": {"name": {"param": "s"}, "k": 1.5}}}, "flags": ["WRONLY", "CREAT"]}}"#.to_string()),
        // a `null` count is no count, as serde reads an absent key
        ("null", compute("1")),
    ] {
        aeiou::load_str(&doc(count, &body)).unwrap_or_else(|e| panic!("{count} {body}: {e:#}"));
    }
    for (count, body, want) in [
        (r#""gpus""#, compute("1"), "actors/gpu/count: expected an integer, got a string (V27)"),
        (r#"{"param": "s"}"#, compute("1"), "actors/gpu/count: expected an integer, got a string (V27)"),
        ("1", compute(r#"{"param": "d"}"#), "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        ("1", compute(r#"{"param": "a"}"#), "actors/gpu/body/0/compute/ns: expected an integer, got an array (V27)"),
        ("1", format!(r#"{{"let": {{"name": "x", "value": "x"}}}}, {{"let": {{"name": "y", "value": {{"ref": "x"}}}}}}, {}"#, compute(r#"{"ref": "y"}"#)),
         "actors/gpu/body/2/compute/ns: expected an integer, got a string (V27)"),
        ("1", format!(r#"{{"let": {{"name": "f", "value": {{"file": {{"dataset": "ds", "id": 0}}}}}}}}, {}"#, compute(r#"{"ref": "f"}"#)),
         "actors/gpu/body/1/compute/ns: expected an integer, got a handle (V27)"),
        ("1", compute(r#"{"add": [1, true]}"#), "actors/gpu/body/0/compute/ns/1: expected a number, got a bool (V27)"),
        ("1", compute(r#"{"neg": {"elem": {"array": "a", "index": "x"}}}"#), "actors/gpu/body/0/compute/ns/index: expected an integer, got a string (V27)"),
        ("1", compute(r#"{"cond": {"if": 0.5, "then": 1, "else": 2}}"#), "actors/gpu/body/0/compute/ns/if: expected a condition, got a float (V27)"),
        ("1", compute(r#"{"draw": {"normal": {"mean": "m", "sd": 1, "min": 0}}}"#), "actors/gpu/body/0/compute/ns/mean: expected a number, got a string (V27)"),
        ("1", r#"{"loop": {"index": "i", "to": {"cond": {"if": true, "then": "a", "else": "b"}}, "body": []}}"#.to_string(),
         "actors/gpu/body/0/loop/to: expected an integer, got a string (V27)"),
        ("1", r#"{"stat": {"file": {"pick": {"dataset": "ds", "dist": "x"}}}}"#.to_string(), "actors/gpu/body/0/stat/file/dist: expected an integer, got a string (V27)"),
        ("1", r#"{"open": {"file": {"object": {"namespace": "o", "fields": {"name": "a", "k": true}}}, "flags": ["WRONLY", "CREAT"]}}"#.to_string(),
         "actors/gpu/body/0/open/file/fields/k: expected an integer, got a bool (V27)"),
        ("1", r#"{"open": {"file": {"object": {"namespace": "o", "fields": {"name": 3, "k": 1}}}, "flags": ["WRONLY", "CREAT"]}}"#.to_string(),
         "actors/gpu/body/0/open/file/fields/name: expected a string, got an integer (V27)"),
    ] {
        rejects(&doc(count, &body), &format!("v27: {want}"));
    }
    // the parameters in effect are held to it too: a string given where the default was a number
    let text = doc(r#"{"param": "n"}"#, &compute("1"));
    let loaded = aeiou::load_str(&text).unwrap();
    assert!(Params::new(&loaded.ast, &config(1, 1, &[("n", "3")])).is_ok());
    let e = Params::new(&loaded.ast, &config(1, 1, &[("n", "abc")])).err().map(|e| format!("{e:#}")).expect("accepted n=abc");
    assert!(e.contains("with the parameters given") && e.contains("v27: actors/gpu/count: expected an integer, got a string (V27)"), "{e}");
}

/// V1 (RC1 review, R22), as `check.py` has it: a parameter whose value names, through `param`s,
/// the parameter itself is refused with the cycle, and the rules that follow a `param` to its
/// value (V3's `at_least`, V25's least size) are not run on it: `{"const": {"param": "pa"}}` as
/// `pa`'s default, read as a dataset's size, overflowed the stack of `aeiou check` (and
/// check.py's recursion). A parameter naming another one is no cycle and stays accepted, and a
/// cycle a run is given (`--param`, a parameter file) is refused by `check_given`.
#[test]
fn a_parameter_defined_through_itself_is_refused_with_its_cycle() {
    let doc = |params: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "cyc", "params": {{{params}}},
                "datasets": {{"d": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": {{"param": "pa"}}, "seed": 1}}}},
                "actors": {{"gpu": {{"body": [{{"loop": {{"index": "i", "to": 2, "body": [
                    {{"let": {{"name": "x", "value": {{"draw": {{"param": "pa"}}}}}}}},
                    {{"stat": {{"file": {{"file": {{"dataset": "d", "id": {{"at": {{"ref": "x", "index": {{"sub": [{{"index": "i"}}, {{"draw": {{"param": "pa"}}}}]}}}}}}}}}}}}}}]}}}}]}}}}}}"#
        )
    };
    rejects(&doc(r#""pa": {"default": {"const": {"param": "pa"}}}"#), "cyc: params/pa: parameter `pa` is defined through itself (pa -> pa) (V1)");
    // through a mixture's arm and a uniform's bound; a parameter that only leads to the cycle
    // (`pc`) is not on it
    let two = r#""pa": {"default": {"mixture": [{"weight": 1, "dist": {"param": "pb"}}]}}, "pb": {"default": {"uniform": {"lo": 1, "hi": {"add": [{"param": "pa"}, 1]}}}}, "pc": {"default": {"const": {"param": "pa"}}}"#;
    let e = aeiou::load_str(&doc(two)).err().map(|e| format!("{e:#}")).expect("accepted a cycle");
    assert!(e.contains("cyc: params/pa: parameter `pa` is defined through itself (pa -> pb -> pa) (V1)"), "{e}");
    assert!(e.contains("cyc: params/pb: parameter `pb` is defined through itself (pb -> pa -> pb) (V1)"), "{e}");
    assert!(!e.contains("params/pc: parameter"), "{e}");
    // a parameter naming another is no cycle
    let chain = r#""pa": {"default": {"mixture": [{"weight": 1, "dist": {"param": "pb"}}]}}, "pb": {"default": {"const": 3}}"#;
    let loaded = aeiou::load_str(&doc(chain)).unwrap_or_else(|e| panic!("{e:#}"));
    // and a value given to `pb` that names `pa` closes one: refused before the run
    let e = Params::new(&loaded.ast, &config(1, 1, &[("pb", r#"{"const": {"param": "pa"}}"#)])).err().map(|e| format!("{e:#}")).expect("accepted a given cycle");
    assert!(e.contains("with the parameters given") && e.contains("cyc: params/pb: parameter `pb` is defined through itself (pb -> pa -> pb) (V1)"), "{e}");
}

/// V27 (RC1 review, R25), as `check.py` has it: a draw yields a kind the document shows (a
/// `const` its argument's, an `empirical` of one kind of value that kind, a `mixture` or `cond`
/// whose arms yield one kind that kind, a parameter whose value is a distribution what that
/// distribution yields), judged where the drawn value is read: `pa` given `{"const": {"param":
/// "pb"}}` with `pb` a distribution, drawn as a `compute`'s `ns`, passed both checkers and failed
/// `run` with "expected an integer, got dist". What the document does not show (arms of
/// different kinds, a binding's distribution drawn again) is left to the run, and a value given
/// to a run is held to it too.
#[test]
fn a_draw_of_a_kind_the_runner_refuses_is_refused_at_check() {
    let doc = |size: &str, body: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "v27",
                "params": {{"n": {{"default": 3}}, "d": {{"default": {{"uniform": {{"lo": 1, "hi": 4}}}}}}, "cd": {{"default": {{"const": {{"param": "d"}}}}}},
                            "cn": {{"default": {{"const": {{"param": "n"}}}}}}, "cs": {{"default": {{"const": "x"}}}}}},
                "datasets": {{"ds": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": {size}, "seed": 1}}}},
                "namespaces": {{"o": {{"pattern": "o/{{name}}-{{k:03}}", "fields": {{"name": "str", "k": "int"}}, "size": 10, "seed": 2}}}},
                "actors": {{"gpu": {{"body": [{body}]}}}}}}"#
        )
    };
    let compute = |ns: &str| format!(r#"{{"compute": {{"ns": {ns}}}}}"#);
    let obj = |name: &str, k: &str| format!(r#"{{"open": {{"file": {{"object": {{"namespace": "o", "fields": {{"name": {name}, "k": {k}}}}}}}, "flags": ["WRONLY", "CREAT"]}}}}"#);
    // accepted: what the runner reads where it stands, and what the document does not show
    for (size, body) in [
        (r#"{"param": "cn"}"#, compute(r#"{"draw": {"param": "cn"}}"#)),
        ("10", compute(r#"{"draw": {"cond": {"if": true, "then": {"param": "cn"}, "else": {"uniform": {"lo": 1, "hi": 2}}}}}"#)),
        ("10", compute(r#"{"draw": {"mixture": [{"weight": 1, "dist": null}, {"weight": 1, "dist": 2}]}}"#)),
        ("10", format!(r#"{{"let": {{"name": "x", "value": {{"draw": {{"param": "cd"}}}}}}}}, {}"#, compute(r#"{"draw": {"ref": "x"}}"#))),
        ("10", obj(r#"{"draw": {"param": "cs"}}"#, r#"{"draw": {"empirical": {"values": [1.5, 2], "weights": [1, 1]}}}"#)),
    ] {
        aeiou::load_str(&doc(size, &body)).unwrap_or_else(|e| panic!("{size} {body}: {e:#}"));
    }
    for (size, body, want) in [
        ("10", compute(r#"{"draw": {"param": "cd"}}"#), "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        ("10", compute(r#"{"draw": {"const": {"param": "d"}}}"#), "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        ("10", compute(r#"{"draw": {"mixture": [{"weight": 1, "dist": {"param": "cd"}}]}}"#), "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        ("10", compute(r#"{"draw": {"param": "cs"}}"#), "actors/gpu/body/0/compute/ns: expected an integer, got a string (V27)"),
        ("10", compute(r#"{"add": [1, {"draw": {"empirical": {"values": ["a", "b"], "weights": [1, 1]}}}]}"#), "actors/gpu/body/0/compute/ns/1: expected a number, got a string (V27)"),
        ("10", r#"{"cond": {"if": {"draw": {"const": 0.5}}, "then": []}}"#.to_string(), "actors/gpu/body/0/cond/if: expected a condition, got a float (V27)"),
        ("10", format!(r#"{{"let": {{"name": "x", "value": {{"draw": {{"param": "cd"}}}}}}}}, {}"#, compute(r#"{"ref": "x"}"#)),
         "actors/gpu/body/1/compute/ns: expected an integer, got a distribution (V27)"),
        ("10", obj(r#""a""#, r#"{"draw": {"param": "cs"}}"#), "actors/gpu/body/0/open/file/fields/k: expected an integer, got a string (V27)"),
        ("10", r#"{"stat": {"file": {"pick": {"dataset": "ds", "dist": {"param": "cd"}}}}}"#.to_string(), "actors/gpu/body/0/stat/file/dist: expected an integer, got a distribution (V27)"),
        (r#"{"const": {"param": "d"}}"#, compute("1"), "datasets/ds/size: expected an integer, got a distribution (V27)"),
        (r#"{"mixture": [{"weight": 1, "dist": null}]}"#, compute("1"), "datasets/ds/size: expected an integer, got null (V27)"),
    ] {
        rejects(&doc(size, &body), &format!("v27: {want}"));
    }
    // a parameter drawn through itself is V1's, and following it does not recurse
    let cyc = doc("10", &compute("1")).replace(r#""n": {"default": 3}"#, r#""n": {"default": {"uniform": {"lo": {"draw": {"param": "n"}}, "hi": 9}}}"#);
    rejects(&cyc, "v27: params/n: parameter `n` is defined through itself (n -> n) (V1)");
    // the parameters in effect are held to it too
    let text = doc("10", &compute(r#"{"draw": {"param": "cn"}}"#));
    let loaded = aeiou::load_str(&text).unwrap();
    assert!(Params::new(&loaded.ast, &config(1, 1, &[("cn", r#"{"const": 7}"#)])).is_ok());
    let e = Params::new(&loaded.ast, &config(1, 1, &[("cn", r#"{"const": {"param": "d"}}"#)])).err().map(|e| format!("{e:#}")).expect("accepted a draw of a distribution");
    assert!(e.contains("with the parameters given") && e.contains("v27: actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"), "{e}");
}

/// V27 (RC1 review): two operands of a comparison of kinds the runner cannot compare, and an
/// `elem`, `len`, or `sum` of a parameter that is not an array (or, for `sum`, holds other
/// than numbers), were accepted and failed the run ("eq: cannot compare int with str").
#[test]
fn a_comparison_or_array_read_the_runner_refuses_is_refused_at_check() {
    let doc = |body: &str| {
        format!(
            r#"{{"ast": "0.13", "name": "v27",
                "params": {{"n": {{"default": 3}}, "s": {{"default": "x"}}, "z": {{"default": null}}, "arr": {{"default": [1, 2.5]}},
                            "sarr": {{"default": ["a", "b"]}}, "d": {{"default": {{"uniform": {{"lo": 1, "hi": 4}}}}}}}},
                "datasets": {{"ds": {{"layout": "whole", "access": "indexed", "pattern": "d/f{{id:04}}", "count": 4, "size": 10, "seed": 1}}}},
                "actors": {{"gpu": {{"body": [{{"let": {{"name": "h", "value": {{"file": {{"dataset": "ds", "id": 0}}}}}}}}, {body}]}}}}}}"#
        )
    };
    let when = |c: &str| format!(r#"{{"cond": {{"if": {c}, "then": [{{"compute": {{"ns": 1}}}}]}}}}"#);
    let compute = |ns: &str| format!(r#"{{"compute": {{"ns": {ns}}}}}"#);
    // accepted: the pairs the runner compares, and kinds the document does not show
    for body in [
        when(r#"{"eq": [1, 2.5]}"#),
        when(r#"{"lt": [{"param": "s"}, "a"]}"#),
        when(r#"{"ne": [true, {"lt": [1, 2]}]}"#),
        when(r#"{"eq": [null, {"param": "d"}]}"#),
        when(r#"{"eq": [{"ref": "h"}, {"ref": "h"}]}"#),
        when(r#"{"eq": [{"add": [1, 2]}, "a"]}"#),
        when(r#"{"eq": [{"elem": {"array": "sarr", "index": 0}}, "a"]}"#),
        compute(r#"{"add": [{"len": "sarr"}, {"sum": "arr"}]}"#),
    ] {
        aeiou::load_str(&doc(&body)).unwrap_or_else(|e| panic!("{body}: {e:#}"));
    }
    for (body, want) in [
        (when(r#"{"eq": [1, "a"]}"#), "actors/gpu/body/1/cond/if: eq: cannot compare an integer with a string (V27)"),
        (when(r#"{"ge": [{"param": "n"}, true]}"#), "actors/gpu/body/1/cond/if: ge: cannot compare an integer with a bool (V27)"),
        (when(r#"{"not": {"ne": [{"param": "d"}, {"param": "d"}]}}"#), "actors/gpu/body/1/cond/if: ne: cannot compare a distribution with a distribution (V27)"),
        (when(r#"{"lt": [{"ref": "h"}, {"elem": {"array": "sarr", "index": 1}}]}"#), "actors/gpu/body/1/cond/if: lt: cannot compare a handle with a string (V27)"),
        (when(r#"{"eq": [{"param": "arr"}, {"count": "ds"}]}"#), "actors/gpu/body/1/cond/if: eq: cannot compare an array with an integer (V27)"),
        (compute(r#"{"len": "n"}"#), "actors/gpu/body/1/compute/ns: len: parameter `n` is not an array (V27)"),
        (compute(r#"{"elem": {"array": "z", "index": 0}}"#), "actors/gpu/body/1/compute/ns: elem: parameter `z` is not an array (V27)"),
        (compute(r#"{"sum": "sarr"}"#), "actors/gpu/body/1/compute/ns: sum: an element of `sarr` is a string (V27)"),
    ] {
        rejects(&doc(&body), &format!("v27: {want}"));
    }
    // the parameters in effect are held to it too
    let text = doc(&compute(r#"{"sum": "arr"}"#));
    let loaded = aeiou::load_str(&text).unwrap();
    assert!(Params::new(&loaded.ast, &config(1, 1, &[("arr", "[4, 5]")])).is_ok());
    let e = Params::new(&loaded.ast, &config(1, 1, &[("arr", r#"[4, "a"]"#)])).err().map(|e| format!("{e:#}")).expect("accepted a sum of a string");
    assert!(e.contains("with the parameters given") && e.contains("v27: actors/gpu/body/1/compute/ns: sum: an element of `arr` is a string (V27)"), "{e}");
}
