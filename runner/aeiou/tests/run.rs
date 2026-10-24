//! `aeiou datagen` and `aeiou run` against a temporary directory on the local filesystem:
//! the corpus a dataset definition describes is written and read back exactly (every read
//! returns the computed count), the run's fingerprint equals the dry run's, the manifest
//! check refuses a changed definition, namespaces must be empty, and a loader delivers its
//! batches in order under real concurrency. The `io_uring` backends run the same abstracts
//! on the event loop and must produce the same fingerprint, counts, and bytes, and so must
//! `posix-aio`, `libaio`, and `mmap`.

use std::path::PathBuf;

use aeiou::backend::{BackendKind, MmapConsume, MmapMode};
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};

fn tmpdir(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("test", tag)
}

fn config(gpus: i64, seed: u64, params: &[(&str, &str)]) -> Config {
    Config { seed, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![], st_blksize: Default::default() }
}

/// A run is the process: the model is leaked so the actor threads can borrow it.
fn leaked_model(name: &str, cfg: Config) -> (&'static aeiou::Loaded, &'static Config, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&crate::common::workload_ast(name)).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    (loaded, cfg, model)
}

fn gen(loaded: &aeiou::Loaded, cfg: &Config, model: &Model<'_>, root: &PathBuf) {
    let params = Params::new(&loaded.ast, cfg).unwrap();
    let opts = DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 4, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false };
    let mut log = Vec::new();
    let written = datagen(loaded, cfg, &params, model, &opts, None, &mut log).unwrap();
    // the projection is what datagen writes: its files (chunks and an array's one file) and
    // bytes (the inventory's apart, which datagen does not count)
    for w in &written {
        let meta = model.datasets.iter().find(|m| m.name() == w.name).unwrap();
        let p = aeiou::project::project(meta, 3).unwrap();
        assert_eq!((p.files, p.bytes - p.inventory_bytes), (w.total_files, w.total_bytes), "dataset `{}`", w.name);
    }
}

fn opts(root: &PathBuf, backend: BackendKind) -> RunOpts {
    RunOpts {
        root: root.clone(),
        endpoints: Default::default(),
        backend,
        s3: aeiou::backend::S3Api::Blocking,
        buffer_bytes: 1 << 20,
        threads: 0,
        write_compress: 1,
        time_scale: 0.0,
        uring: Default::default(),
        mmap: Default::default(),
        mmap_consume: Default::default(),
        aio_depth: 0,
        clean_namespaces: false,
        expect_fingerprint: None,
        expect_dataset_ids: vec![],
        rank: 0,
        ranks: 1,
        rank_rotate: 0,
        max_gap: None,
        require_cold: false,
        drop_caches: false,
    }
}

fn go(model: &'static Model<'static>, o: RunOpts) -> run::Report {
    run::run(model, o, std::collections::HashMap::new()).unwrap()
}

fn dry_fingerprint(model: &Model<'_>) -> (u64, u64, u64) {
    let r = dryrun::run(model, 2, None).unwrap();
    (r.total.fingerprint, r.total.total.ops, r.total.total.bytes_read)
}

#[test]
fn small_files_training_round_trip() {
    let root = tmpdir("tsf");
    let params = [("samples", "600"), ("batch", "4"), ("workers", "2"), ("prefetch", "2"), ("steps", "12"), ("walk", "true")];
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(2, 7, &crate::common::file_per_sample(&params)));
    gen(loaded, cfg, model, &root);
    let checks = run::check_datasets(loaded, cfg, &root).unwrap();
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].files, Some(600));
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    for backend in [
        BackendKind::Sync,
        BackendKind::SyncDirect,
        BackendKind::Uring,
        BackendKind::UringDirect,
        BackendKind::PosixAio,
        BackendKind::PosixAioDirect,
        BackendKind::LibAio,
        BackendKind::LibAioDirect,
        BackendKind::Mmap,
    ] {
        let r = go(model, opts(&root, backend));
        assert_eq!(r.stats.fingerprint, fp, "{:?}: fingerprint", backend);
        assert_eq!(r.stats.ops, ops);
        assert_eq!(r.stats.bytes_read, bytes, "every read returned the computed count");
        assert_eq!(r.stats.takes, 24);
        assert_eq!(r.stats.barriers, 24, "`every(sync_every)` fires every step (`sync_every` 1) on each of the two instances");
        assert_eq!(r.stats.expected_errors, 2 * 12 * 4, "ENOTTY on every ioctl");
        assert_eq!(r.actors.len(), 2);
        assert!(r.actors.iter().all(|a| a.takes.len() == 12));
        assert!(r.departure_releases.is_empty());
    }
}

#[test]
fn manifest_mismatch_and_missing_are_refused() {
    let root = tmpdir("manifest");
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(1, 1, &crate::common::file_per_sample(&[("samples", "50")])));
    // nothing generated yet
    assert!(run::check_datasets(loaded, cfg, &root).unwrap_err().to_string().contains("datagen"));
    gen(loaded, cfg, model, &root);
    run::check_datasets(loaded, cfg, &root).unwrap();
    // a different count resolves to a different definition
    let (loaded2, cfg2, _) = leaked_model("file_per_sample", config(1, 1, &crate::common::file_per_sample(&[("samples", "51")])));
    let e = run::check_datasets(loaded2, cfg2, &root).unwrap_err();
    assert!(e.to_string().contains("count: 51 vs 50"), "{e:#}");
    // datagen refuses a non-empty root
    let (loaded3, cfg3, model3) = leaked_model("file_per_sample", config(1, 1, &crate::common::file_per_sample(&[("samples", "50")])));
    let params = Params::new(&loaded3.ast, cfg3).unwrap();
    let o = DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 1, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false };
    let mut log = Vec::new();
    assert!(datagen(loaded3, cfg3, &params, model3, &o, None, &mut log).is_err());
}

#[test]
fn namespaces_must_be_empty_and_writes_are_read_back() {
    let root = tmpdir("ckpt");
    let params = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 65536]"), ("meta_bytes", "65536"), ("readback", "true")];
    let (loaded, _cfg, model) = leaked_model("dcp_save", config(2, 3, &crate::common::dcp_save(&params)));
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    let (fp, ops, _) = dry_fingerprint(model);
    let r = go(model, opts(&root, BackendKind::Sync));
    assert_eq!(r.stats.fingerprint, fp);
    assert_eq!(r.stats.ops, ops);
    assert_eq!(r.stats.barriers, 2 * 2 * 4, "four barriers per checkpoint, two checkpoints, two ranks");
    assert!(r.stats.bytes_written > 0);
    assert_eq!(r.stats.bytes_read, r.stats.bytes_written - 2 * 65536, "read-back of the shard files (not .metadata)");
    // the objects exist with the computed sizes
    let shard = root.join("ckpt/step_000002/__1_0.distcp");
    assert_eq!(std::fs::metadata(&shard).unwrap().len(), 1048576 + 2097152 + 1048576 + 65536 + 4 * (704 + 873));
    assert!(root.join("ckpt/step_000002/.metadata").exists());
    // a second run refuses the stale namespace, and --clean-namespaces empties it
    let e = run::prepare_namespaces(&loaded.ast, &root, false).unwrap_err();
    assert!(e.to_string().contains("not empty"), "{e:#}");
    let cleaned = run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
    assert_eq!(cleaned, vec!["ckpt".to_string()]);
    assert!(!shard.exists());
}

/// The KV-cache abstracts in a few requests: a slot with two conversations open (the default's forty would start
/// nothing but new ones), and an engine that keeps none, half, or all of a returning one.
const KV_REUSE: &str = r#"{"mixture": [{"weight": 0.3, "dist": null}, {"weight": 0.7, "dist": {"const": 2}}]}"#;
const KV_KEEP: &str = r#"{"empirical": {"values": [0, 50, 100], "weights": [1, 1, 1]}}"#;

#[test]
fn kv_cache_chunked_dataset_parallel_slots_and_namespace() {
    let root = tmpdir("kv");
    let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8"), ("reuse", KV_REUSE), ("keep", KV_KEEP)];
    let (loaded, cfg, model) = leaked_model("lmcache_local_disk", config(1, 5, &crate::common::lmcache(&params)));
    gen(loaded, cfg, model, &root);
    // the chunked files exist: kv/sys/0000/blk_0000 …
    assert!(root.join("kv/sys/0000/blk_0000").exists());
    run::check_datasets(loaded, cfg, &root).unwrap();
    // the dataset root lies inside the namespace root and is left alone
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    assert!(root.join("kv/sys/.aeiou-dataset.json").exists());
    let (fp, ops, bytes) = dry_fingerprint(model);
    let r = go(model, opts(&root, BackendKind::Sync));
    assert_eq!(r.stats.fingerprint, fp);
    assert_eq!(r.stats.ops, ops);
    assert_eq!(r.stats.bytes_read, bytes);
    assert!(r.stats.threads >= 1 + 2, "the actor's thread runs slot 0 itself: two pool threads for three slots, and one more under any slot that loads two chunks at once: {}", r.stats.threads);
    assert!(r.stats.expected_errors > 0, "the ENOTTY of the terminal probe in every open (ENOENT lookups and EEXIST mkdirs until the 2026-10-02 trace)");
}

#[test]
fn endpoints_place_a_dataset_and_a_namespace_apart_from_the_root() {
    // `--endpoint` (DESIGN_REVIEW §3.65): the system prompts' dataset in one directory, the KV namespace in
    // another, the sub-agent prompts (no endpoint, root `kv/sub` inside the namespace's `kv`) moving with the
    // namespace; nothing under --root, and the fingerprint of the run under --root alone
    let base = tmpdir("kvplace");
    let (root, sys, kv) = (base.join("root"), base.join("sys"), base.join("kv"));
    std::fs::create_dir_all(&root).unwrap();
    let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8"), ("reuse", KV_REUSE), ("keep", KV_KEEP)];
    let (loaded, cfg, model) = leaked_model("lmcache_local_disk", config(1, 5, &crate::common::lmcache(&params)));
    let endpoints = aeiou::endpoint::Endpoints::parse(&loaded.ast, &[format!("sysp={}", sys.display()), format!("kv={}", kv.display())]).unwrap();
    let dparams = Params::new(&loaded.ast, cfg).unwrap();
    let dopts = DatagenOpts { root: root.clone(), endpoints: endpoints.clone(), writers: 4, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false };
    datagen(loaded, cfg, &dparams, model, &dopts, None, &mut Vec::new()).unwrap();
    assert!(sys.join("0000/blk_0000").exists() && sys.join(".aeiou-dataset.json").exists());
    assert!(kv.join("sub/.aeiou-dataset.json").exists());
    assert!(!root.join("kv").exists());
    let mut o = opts(&root, BackendKind::Sync);
    o.endpoints = endpoints;
    assert!(run::check_datasets(loaded, cfg, &root).is_err(), "nothing under --root");
    run::check_datasets(loaded, cfg, &o).unwrap();
    run::prepare_namespaces(&loaded.ast, &o, false).unwrap();
    let (fp, ops, _) = dry_fingerprint(model);
    let r = go(model, o);
    assert_eq!((r.stats.fingerprint, r.stats.ops), (fp, ops));
    let chunks = std::fs::read_dir(&kv).unwrap().filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".pt")).count();
    assert!(chunks > 0, "the namespace's chunks are written at its endpoint");
    assert!(!root.join("kv").exists());
}

/// An input namespace placed by its endpoint inside an output namespace's directory is left
/// alone when the output is emptied, as a dataset there is (V14: a run never empties an input
/// root; RC1 review).
#[test]
fn an_input_namespace_inside_an_output_root_is_not_emptied() {
    let base = tmpdir("nsinput");
    let ast = base.join("r.ast.json");
    std::fs::write(
        &ast,
        r#"{"ast": "0.13", "name": "r",
            "namespaces": {"o": {"pattern": "o/f{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 1, "input": true},
                           "q": {"pattern": "q/f{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 1}},
            "actors": {"gpu": {"body": [{"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"k": 0}}}}},
                {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}}, {"close": {"file": {"ref": "f"}}},
                {"let": {"name": "g", "value": {"object": {"namespace": "q", "fields": {"k": 0}}}}},
                {"open": {"file": {"ref": "g"}, "flags": ["WRONLY", "CREAT", "TRUNC"]}}, {"close": {"file": {"ref": "g"}}}]}}}"#,
    )
    .unwrap();
    let loaded = aeiou::load(&ast).unwrap();
    let (root, out) = (base.join("root"), base.join("out"));
    std::fs::create_dir_all(out.join("in")).unwrap();
    std::fs::write(out.join("in/f0"), b"x").unwrap();
    std::fs::write(out.join("in/.aeiou-namespace.json"), b"{}").unwrap();
    std::fs::write(out.join("f0"), b"x").unwrap();
    let endpoints = aeiou::endpoint::Endpoints::parse(&loaded.ast, &[format!("o={}", out.join("in").display()), format!("q={}", out.display())]).unwrap();
    let mut o = opts(&root, BackendKind::Sync);
    o.endpoints = endpoints;
    assert!(run::prepare_namespaces(&loaded.ast, &o, false).unwrap_err().to_string().contains("not empty (1 entries"));
    assert_eq!(run::prepare_namespaces(&loaded.ast, &o, true).unwrap(), vec!["q".to_string()]);
    assert!(!out.join("f0").exists() && out.join("in/f0").exists() && out.join("in/.aeiou-namespace.json").exists());
}

#[test]
fn kv_shared_store_filled_by_one_engine_and_read_by_a_cold_one() {
    // lmcache_shared_fs and lmcache_shared_fs_reader come from one script and place their draws at the same sites:
    // with the writer's seed, instance count and parameters the reader opens exactly the chunks the writer
    // renamed into place (two instances, eviction with re-store, system prompts loaded).
    let root = tmpdir("kvshared");
    let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("buf", "65536"), ("concurrency", "3"), ("warm", "4"),
                  ("requests", "12"), ("reuse", KV_REUSE), ("keep", KV_KEEP), ("retain", "6"), ("sys_local", "false")];
    let (wl, wcfg, wmodel) = leaked_model("lmcache_shared_fs", config(2, 5, &crate::common::lmcache(&params)));
    gen(wl, wcfg, wmodel, &root);
    run::check_datasets(wl, wcfg, &root).unwrap();
    run::prepare_namespaces(&wl.ast, &root, false).unwrap();
    let (wfp, wops, _) = dry_fingerprint(wmodel);
    let wo = opts(&root, BackendKind::Sync);
    let t0 = run::unix_now();
    let wr = go(wmodel, wo.clone());
    assert_eq!((wr.stats.fingerprint, wr.stats.ops), (wfp, wops));
    let renames = wr.stats.counts[&aeiou::vm::OpKind::Rename];
    assert!(renames > 0 && wr.stats.counts[&aeiou::vm::OpKind::Write] == 2 * renames, "a header and a chunk per stored file");
    run::write_namespace_manifests(wl, wcfg, &root, &wo, &wr, t0, run::unix_now()).unwrap();
    let m = aeiou::payload::NamespaceManifest::read(&root.join("kv")).unwrap();
    let objects = m.objects.as_ref().unwrap();
    assert!(objects.iter().all(|(p, _)| p.ends_with(".data")), "the temporary names are gone: {objects:?}");

    let (rl, rcfg, rmodel) = leaked_model("lmcache_shared_fs_reader", config(2, 5, &crate::common::lmcache(&params)));
    let ro = opts(&root, BackendKind::Sync);
    let (checks, input_objects) = run::check_input_namespaces(rl, rcfg, &root, &ro).unwrap();
    assert_eq!(checks[0].names, vec!["kv".to_string()]);
    run::prepare_namespaces(&rl.ast, &root, false).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(rmodel);
    let r = run::run(rmodel, ro, input_objects).unwrap();
    assert_eq!((r.stats.fingerprint, r.stats.ops, r.stats.bytes_read), (fp, ops, bytes));
    assert_eq!(r.stats.counts.get(&aeiou::vm::OpKind::Write), None);
    assert_eq!(r.stats.counts.get(&aeiou::vm::OpKind::Rename), None);
    assert!(r.stats.counts[&aeiou::vm::OpKind::Open] > wr.stats.counts[&aeiou::vm::OpKind::Open] - renames, "the reader loads the chunks the writer computed");
    assert!(r.stats.input_opens > 0 && r.stats.input_opens == r.stats.warm_opens, "every chunk was written on this host");
    // another seed draws other conversation ids, which the store does not have. The namespace is declared
    // `same_run` (V15, contract 0.4), so the manifest check refuses before the gate, naming what differs
    let refusal = |seed: u64, gpus: i64, p: &[(&str, &str)]| {
        let (l, c, _) = leaked_model("lmcache_shared_fs_reader", config(gpus, seed, &crate::common::lmcache(p)));
        run::check_input_namespaces(l, c, &root, &opts(&root, BackendKind::Sync)).unwrap_err().to_string()
    };
    let e = refusal(6, 2, &params);
    assert!(e.contains("same_run") && e.contains("--seed 6 here, 5 in the writer"), "{e}");
    let e = refusal(5, 1, &params);
    assert!(e.contains("--gpus 1 here, 2 in the writer"), "{e}");
    let mut other = params.to_vec();
    other[9] = ("retain", "7");
    let e = refusal(5, 2, &other);
    assert!(e.contains("parameter `retain`: 7 here, 6 in the writer") && !e.contains("--seed"), "{e}");
    // without the check the same mismatch fails late, at the first chunk the store does not have
    let (_, _, late) = leaked_model("lmcache_shared_fs_reader", config(2, 6, &crate::common::lmcache(&params)));
    let (_, inputs) = run::check_input_namespaces(rl, rcfg, &root, &opts(&root, BackendKind::Sync)).unwrap();
    assert!(run::run(late, opts(&root, BackendKind::Sync), inputs).is_err());
}

#[test]
fn array_dataset_nested_parallel_and_direct_reads() {
    let root = tmpdir("diskann");
    let params = crate::common::diskann_search(&[("n", "24995"), ("threads", "2"), ("queries", "6")]);    // 5,000 sectors
    let (loaded, cfg, model) = leaked_model("diskann_search", config(1, 9, &params));
    gen(loaded, cfg, model, &root);
    assert_eq!(std::fs::metadata(root.join("diskann/index.bin")).unwrap().len(), 5000 * 4096);
    run::check_datasets(loaded, cfg, &root).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    let r = go(model, opts(&root, BackendKind::Sync));
    assert_eq!(r.stats.fingerprint, fp);
    assert_eq!(r.stats.ops, ops);
    assert_eq!(r.stats.bytes_read, bytes);
    // the sub-actor pool: a forking thread runs sub-actor 0 itself and keeps a thread for
    // each of the others, so the instance's thread and one more are the two search threads,
    // and each of those has `beam - 1` threads kept across every hop of every query
    let beam: u64 = 4;
    assert!(ops > 2 * 6 * 2 * beam, "several hops per query: {ops} ops");
    assert_eq!(r.stats.threads, 1 + (2 - 1) + 2 * (beam - 1));
}

#[test]
fn loader_delivers_in_order_and_bounds_prefetch() {
    // 1 worker, prefetch 1: batch b+1 cannot start before batch b is taken, so with a
    // non-zero step time the stall of every take after the first is about one batch's I/O.
    let root = tmpdir("loader");
    let params = [("samples", "200"), ("batch", "2"), ("workers", "3"), ("prefetch", "1"), ("steps", "30"), ("step_time", "2000000")];
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(1, 11, &crate::common::file_per_sample(&params)));
    gen(loaded, cfg, model, &root);
    let (fp, _, _) = dry_fingerprint(model);
    let mut o = opts(&root, BackendKind::Sync);
    o.time_scale = 1.0;
    let r = go(model, o);
    assert_eq!(r.stats.fingerprint, fp, "the same ops whatever the interleaving");
    let takes = &r.actors[0].takes;
    assert_eq!(takes.len(), 30);
    assert!(takes.iter().all(|t| t.compute_ns == 2_000_000));
    assert_eq!(r.stats.threads, 1 + 3);
}

#[test]
fn write_then_restore_through_the_namespace_manifest() {
    // dcp_save leaves `.aeiou-namespace.json` at ckpt/; dcp_load declares the
    // namespaces `input`, reads what was written, measures the gap, and counts the warm opens
    // (all of them on one host).
    let root = tmpdir("restore");
    let wparams = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 1048576]"), ("meta_bytes", "65536")];
    let (wl, wcfg, wmodel) = leaked_model("dcp_save", config(2, 3, &crate::common::dcp_save(&wparams)));
    run::prepare_namespaces(&wl.ast, &root, false).unwrap();
    let wo = opts(&root, BackendKind::Sync);
    let t0 = run::unix_now();
    let wr = go(wmodel, wo.clone());
    assert_eq!(wr.stats.counts.get(&aeiou::vm::OpKind::Read), None, "readback is off by default");
    let t1 = run::unix_now();
    let written = run::write_namespace_manifests(wl, wcfg, &root, &wo, &wr, t0, t1).unwrap();
    assert_eq!(written.len(), 1);
    assert!(root.join("ckpt/.aeiou-namespace.json").exists());
    let m = aeiou::payload::NamespaceManifest::read(&root.join("ckpt")).unwrap();
    assert_eq!(m.abstract_name, "dcp_save");
    assert_eq!(m.ranks.len(), 1);
    assert_eq!(m.ranks[0].gpus, [0, 2]);
    let objects = m.objects.as_ref().unwrap();
    // two step dirs, two shards each, .metadata (renamed into place) each
    assert_eq!(objects.len(), 2 * (1 + 2 + 1), "{objects:?}");
    assert!(objects.iter().any(|(p, g)| p == "ckpt/step_000002/.metadata" && *g == 0));
    assert!(objects.iter().any(|(p, g)| p == "ckpt/step_000002/__1_0.distcp" && *g == 1));

    // the restore: same layout parameters, restore_step 2
    let rparams = [
        ("restore_step", "2"),
        ("item_bytes", "[1048576, 2097152, 1048576, 1048576]"),
        ("item_off", "[0, 1050153, 3148882, 4199035]"),   // Σ (704 + item + 873) of the writer
        ("meta_bytes", "65536"),
    ];
    let (rl, rcfg, rmodel) = leaked_model("dcp_load", config(2, 5, &crate::common::dcp_load(&rparams)));
    let mut ro = opts(&root, BackendKind::Sync);
    ro.max_gap = Some(30.0);
    let (checks, input_objects) = run::check_input_namespaces(rl, rcfg, &root, &ro).unwrap();
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].names, vec!["ckpt".to_string(), "ckpt_meta".to_string()]);
    assert!(checks[0].gap >= 0.0 && checks[0].gap < 30.0);
    assert!(checks[0].same_host, "one host wrote and reads");
    assert_eq!(input_objects.len(), objects.len());
    // the input root is neither refused nor cleaned
    run::prepare_namespaces(&rl.ast, &root, false).unwrap();
    assert!(root.join("ckpt/step_000002/__1_0.distcp").exists());
    let (fp, ops, bytes) = dry_fingerprint(rmodel);
    let r = run::run(rmodel, ro.clone(), input_objects).unwrap();
    assert_eq!(r.stats.fingerprint, fp);
    assert_eq!(r.stats.ops, ops);
    assert_eq!(r.stats.bytes_read, bytes);
    assert_eq!(r.stats.input_opens, 4, "two ranks open .metadata and a shard");
    assert_eq!(r.stats.warm_opens, 4, "all written on this host");
    // --require-cold refuses on one host; a different layout parameter is refused by the definition check
    ro.require_cold = true;
    assert!(run::check_input_namespaces(rl, rcfg, &root, &ro).unwrap_err().to_string().contains("require-cold"));
    // `size` is each abstract's own model and is not compared (the writer says as_written);
    // the names and the content seed are: a manifest with another seed is refused
    let mp = root.join("ckpt/.aeiou-namespace.json");
    let text = std::fs::read_to_string(&mp).unwrap();
    std::fs::write(&mp, text.replace(r#""seed": 1592646274"#, r#""seed": 7"#)).unwrap();
    let e = run::check_input_namespaces(rl, rcfg, &root, &opts(&root, BackendKind::Sync)).unwrap_err();
    assert!(e.to_string().contains("differs from the writer"), "{e:#}");
    std::fs::write(&mp, text).unwrap();
    // a run without the manifest is refused
    std::fs::remove_file(root.join("ckpt/.aeiou-namespace.json")).unwrap();
    assert!(run::check_input_namespaces(rl, rcfg, &root, &ro).unwrap_err().to_string().contains("no run has written"));
}

#[test]
fn gpu_ranges_rotate_by_host() {
    use run::gpu_range;
    assert_eq!(gpu_range(8, 1, 0, 0), (0, 8));
    assert_eq!(gpu_range(8, 1, 0, 5), (0, 8));
    assert_eq!(gpu_range(8, 4, 1, 0), (2, 4));
    assert_eq!(gpu_range(8, 4, 1, 1), (4, 6));
    assert_eq!(gpu_range(8, 4, 3, 1), (0, 2));
    assert_eq!(gpu_range(10, 4, 3, 0), (9, 10), "uneven split, last host short");
    assert_eq!(gpu_range(1, 4, 2, 0), (1, 1), "a single-instance template runs on rank 0 only");
    // every GPU id is run exactly once whatever the rotation
    for rotate in 0..4 {
        let mut all: Vec<i64> = (0..4).flat_map(|r| { let (lo, hi) = gpu_range(10, 4, r, rotate); lo..hi }).collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }
}

#[test]
fn io_uring_reproduces_the_sync_runs() {
    // The event loop multiplexes every actor and sub-actor of these abstracts (loaders with
    // ordered channels, `parallel` slots, nested `parallel`, barriers, namespace writes and
    // read-back) over one ring per thread; the fingerprint, op count, and bytes must equal the
    // dry run's exactly, on one loop and on several.
    for threads in [1usize, 3] {
        let root = tmpdir("uring-kv");
        let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8"), ("reuse", KV_REUSE), ("keep", KV_KEEP)];
        let (loaded, cfg, model) = leaked_model("lmcache_local_disk", config(2, 5, &crate::common::lmcache(&params)));
        gen(loaded, cfg, model, &root);
        run::check_datasets(loaded, cfg, &root).unwrap();
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        let (fp, ops, bytes) = dry_fingerprint(model);
        let mut o = opts(&root, BackendKind::Uring);
        o.threads = threads;
        let r = go(model, o);
        assert_eq!(r.stats.fingerprint, fp, "lmcache_local_disk on {threads} loop(s)");
        assert_eq!(r.stats.ops, ops);
        assert_eq!(r.stats.bytes_read, bytes);
        assert_eq!(r.stats.threads as usize, threads.min(2), "loop threads, not actors");
        assert!(r.stats.expected_errors > 0);
        // the host counters: the loops are tasks, and the root is on some mount. Not asserted:
        // an io-wq worker. An `openat` whose lookup is cached completes inline, so whether a
        // short run punts at all is the kernel's; under `cargo test` the idle workers of the
        // binary's earlier tests were what this saw (2026-10-07, CI under nextest). Nor the
        // sampler and the main thread on top of the loops: the peak is sampled, and a short
        // run on a slow host has shown 3 where loops + 2 would be 4 (CI, twice on 2026-10-12)
        let c = &r.counters;
        assert!(c.tasks_peak as usize >= threads.min(2), "{c:?}");
        assert!(c.cpu_user_ns + c.cpu_sys_ns > 0, "{c:?}");
        assert!(c.mount.as_ref().map(|m| !m.fstype.is_empty()).unwrap_or(false), "{c:?}");
    }
    {
        let root = tmpdir("uring-diskann");
        let params = crate::common::diskann_search(&[("n", "24995"), ("threads", "2"), ("queries", "6")]);    // 5,000 sectors
        let (loaded, cfg, model) = leaked_model("diskann_search", config(2, 9, &params));
        gen(loaded, cfg, model, &root);
        let (fp, ops, bytes) = dry_fingerprint(model);
        for backend in [BackendKind::Uring, BackendKind::UringDirect] {
            let r = go(model, opts(&root, backend));
            assert_eq!(r.stats.fingerprint, fp, "{backend:?}");
            assert_eq!(r.stats.ops, ops);
            assert_eq!(r.stats.bytes_read, bytes);
        }
    }
    {
        let root = tmpdir("uring-ckpt");
        let params = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 65536]"), ("meta_bytes", "65536"), ("readback", "true"),
            ("hdr", "4096"), ("trailer", "4096")];   // the traced 704 and 873 are not aligned, and a direct backend refuses such writes
        let (loaded, _cfg, model) = leaked_model("dcp_save", config(2, 3, &crate::common::dcp_save(&params)));
        let (fp, ops, _) = dry_fingerprint(model);
        for backend in [BackendKind::Uring, BackendKind::UringDirect] {
            run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
            let r = go(model, opts(&root, backend));
            assert_eq!(r.stats.fingerprint, fp, "{backend:?}");
            assert_eq!(r.stats.ops, ops);
            assert_eq!(r.stats.barriers, 2 * 2 * 4, "barriers between loops go through the coordinator's eventfd");
            assert_eq!(r.stats.bytes_read, r.stats.bytes_written - 2 * 65536);
            assert_eq!(std::fs::metadata(root.join("ckpt/step_000002/__1_0.distcp")).unwrap().len(), 1048576 + 2097152 + 1048576 + 65536 + 4 * (4096 + 4096));
        }
    }
}

#[test]
fn posix_aio_libaio_and_mmap_reproduce_the_sync_runs() {
    // The same three abstracts as the io_uring test, under the other backends: glibc AIO and
    // the mapping on one thread per actor, the kernel AIO context on the event loop. Each
    // must reproduce the dry run's fingerprint, op count, and bytes, and report its own block.
    let blocking = [BackendKind::PosixAio, BackendKind::PosixAioDirect, BackendKind::Mmap];
    let looped = [BackendKind::LibAio, BackendKind::LibAioDirect];
    {
        let root = tmpdir("more-kv");
        let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8"), ("reuse", KV_REUSE), ("keep", KV_KEEP)];
        let (loaded, cfg, model) = leaked_model("lmcache_local_disk", config(2, 5, &crate::common::lmcache(&params)));
        gen(loaded, cfg, model, &root);
        let (fp, ops, bytes) = dry_fingerprint(model);
        for backend in blocking.into_iter().chain(looped) {
            for threads in [1usize, 3] {
                run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
                let mut o = opts(&root, backend);
                o.threads = threads;
                let r = go(model, o);
                assert_eq!(r.stats.fingerprint, fp, "{backend:?} with --threads {threads}");
                assert_eq!(r.stats.ops, ops);
                assert_eq!(r.stats.bytes_read, bytes);
                assert!(r.uring.is_none());
                match &r.aio {
                    Some(a) => {
                        assert!(backend.libaio());
                        assert_eq!(r.stats.threads as usize, threads.min(2), "loop threads, not actors");
                        assert_eq!((a.loops as usize, a.depth), (threads.min(2), aeiou::aio::DEPTH as u64));
                        let rw = r.stats.counts[&aeiou::vm::OpKind::Read] + r.stats.counts.get(&aeiou::vm::OpKind::Write).copied().unwrap_or(0);
                        assert!(a.submitted >= rw, "every read and write went through io_submit: {a:?}");
                        assert!(a.in_flight_peak >= 1 && a.in_flight_peak <= a.depth, "{a:?}");
                        assert!(a.submits <= a.submitted && a.getevents > 0, "{a:?}");
                    }
                    None => assert!(!backend.libaio()),
                }
                assert_eq!(r.mmap.is_some(), backend == BackendKind::Mmap);
            }
        }
        // the smallest context that can be asked for (the kernel rounds it up to a few per
        // CPU, so it does not fill here): the op stream is the same
        run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
        let mut o = opts(&root, BackendKind::LibAioDirect);
        o.threads = 1;
        o.aio_depth = 1;
        let r = go(model, o);
        assert_eq!(r.stats.fingerprint, fp);
        assert_eq!(r.aio.as_ref().unwrap().depth, 1);
    }
    {
        let root = tmpdir("more-diskann");
        let params = crate::common::diskann_search(&[("n", "24995"), ("threads", "2"), ("queries", "6")]);    // 5,000 sectors
        let (loaded, cfg, model) = leaked_model("diskann_search", config(2, 9, &params));
        gen(loaded, cfg, model, &root);
        let (fp, ops, bytes) = dry_fingerprint(model);
        for backend in blocking.into_iter().chain(looped) {
            let r = go(model, opts(&root, backend));
            assert_eq!(r.stats.fingerprint, fp, "{backend:?}");
            assert_eq!(r.stats.ops, ops);
            assert_eq!(r.stats.bytes_read, bytes);
        }
        // the mapping under each mode and each way of consuming a range: the two advising
        // modes call madvise once per read; `touch` reads a byte of every page of every
        // read (the sector reads are 4 KiB at 4 KiB offsets: one page each; the load's stream
        // reads are 8,191 bytes of the PQ file, two pages, the rest of it, 195, and 8,191
        // bytes of the index) and copies nothing,
        // except under `populate`, which leaves nothing to touch; `copy` copies every byte
        for mode in [MmapMode::Fault, MmapMode::Populate, MmapMode::WillNeed] {
            for consume in [MmapConsume::Touch, MmapConsume::Copy] {
                let mut o = opts(&root, BackendKind::Mmap);
                o.mmap = mode;
                o.mmap_consume = consume;
                let r = go(model, o);
                assert_eq!(r.stats.fingerprint, fp, "{mode:?} {consume:?}");
                assert_eq!(r.stats.bytes_read, bytes, "{mode:?} {consume:?}");
                let m = r.mmap.as_ref().unwrap();
                assert_eq!((m.mode, m.consume), (mode, consume), "{m:?}");
                // one mapping per open, whoever reads: each of the 2 instances opens the PQ
                // file (24,995 points' codes and its header) once and the index twice, and its search threads and their beam
                // sub-actors read the index through the second mapping
                assert_eq!((m.maps, m.mapped_bytes), (6, 2 * (24995 * 32 + 8 + 2 * 5000 * 4096)), "{m:?}");
                let reads = r.stats.counts[&aeiou::vm::OpKind::Read];
                assert_eq!(m.advised, if mode == MmapMode::Fault { 0 } else { reads }, "{m:?}");
                let (touched, copied) = match (consume, mode) {
                    (MmapConsume::Copy, _) => (0, bytes),
                    (MmapConsume::Touch, MmapMode::Populate) => (0, 0),
                    (MmapConsume::Touch, _) => ((bytes - 2 * (24995 * 32 + 8 + 8191)) / 4096 + 2 * (2 + 195 + 2), 0),
                };
                assert_eq!((m.touched_pages, m.copied_bytes), (touched, copied), "{m:?}");
                assert!(r.counters.minor_faults + r.counters.major_faults > 0, "{:?}", r.counters);
            }
        }
    }
    {
        let root = tmpdir("more-ckpt");
        let params = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 65536]"), ("meta_bytes", "65536"), ("readback", "true"),
            ("hdr", "4096"), ("trailer", "4096")];   // the traced 704 and 873 are not aligned, and a direct backend refuses such writes
        let (loaded, _cfg, model) = leaked_model("dcp_save", config(2, 3, &crate::common::dcp_save(&params)));
        let (fp, ops, _) = dry_fingerprint(model);
        for backend in blocking.into_iter().chain(looped) {
            run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
            let r = go(model, opts(&root, backend));
            assert_eq!(r.stats.fingerprint, fp, "{backend:?}");
            assert_eq!(r.stats.ops, ops);
            assert_eq!(r.stats.barriers, 2 * 2 * 4, "{backend:?}: barriers (on the loop: the poll on the eventfd)");
            assert_eq!(r.stats.bytes_read, r.stats.bytes_written - 2 * 65536, "{backend:?}: written, then read back (through a mapping made after the write)");
            assert_eq!(std::fs::metadata(root.join("ckpt/step_000002/__1_0.distcp")).unwrap().len(), 1048576 + 2097152 + 1048576 + 65536 + 4 * (4096 + 4096));
        }
    }
}

#[test]
fn io_uring_loader_delivers_in_order_and_bounds_prefetch() {
    let root = tmpdir("uring-loader");
    let params = [("samples", "200"), ("batch", "2"), ("workers", "3"), ("prefetch", "1"), ("steps", "30"), ("step_time", "2000000")];
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(1, 11, &crate::common::file_per_sample(&params)));
    gen(loaded, cfg, model, &root);
    let (fp, _, _) = dry_fingerprint(model);
    let mut o = opts(&root, BackendKind::Uring);
    o.time_scale = 1.0;
    let r = go(model, o);
    assert_eq!(r.stats.fingerprint, fp, "the same ops whatever the interleaving");
    let takes = &r.actors[0].takes;
    assert_eq!(takes.len(), 30);
    assert!(takes.iter().all(|t| t.compute_ns == 2_000_000), "compute is recorded unscaled against the take");
    assert_eq!(r.stats.compute_ns, 30 * 2_000_000);
    assert_eq!(r.stats.threads, 1, "one loop for one instance");
    assert!(r.elapsed >= std::time::Duration::from_millis(60), "the timers slept: {:?}", r.elapsed);
}

/// `fsync` and `POSIX_FADV_DONTNEED` every file under `dir`: what a test can do about the
/// page cache without root.
fn evict(dir: &std::path::Path) {
    use std::os::fd::AsRawFd;
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            evict(&p);
        } else {
            let f = std::fs::File::open(&p).unwrap();
            f.sync_all().unwrap();
            assert_eq!(unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
        }
    }
}

/// Read every file under `dir` through the page cache.
fn warm(dir: &std::path::Path) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            warm(&p);
        } else {
            std::fs::read(&p).unwrap();
        }
    }
}

#[test]
fn residency_check_sees_the_page_cache_and_require_cold_refuses() {
    // The cold start (`cold.rs`): a formula picks at most 256 files per dataset and `mincore`
    // counts their resident pages. Just generated (`O_DIRECT`), no page is resident; once
    // read, every page is and `--require-cold` refuses; once evicted, none is and the start
    // passes. A plain run (neither flag) does not sample at all. A tmpfs is its own page
    // cache, so there the pages stay.
    use aeiou::cold;
    let root = tmpdir("cold");
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(1, 3, &crate::common::file_per_sample(&[("samples", "300")])));
    gen(loaded, cfg, model, &root);
    let fstype = aeiou::counters::MountSnapshot::for_path(&root).map(|m| m.fstype).unwrap_or_default();
    let r = cold::residency(model, &root).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].dataset.as_str(), r[0].files, r[0].of), ("train", 256, 300));
    assert!(r[0].pages > 256, "{r:?}");
    if fstype != "tmpfs" {
        // Datagen writes with `O_DIRECT`, so the client's cache stays cold, except that where
        // an unaligned tail is padded and truncated the truncate zeroes the end of the last
        // block through the page cache, and on some kernels that folio stays resident after
        // write-back: on Linux 6.17 (ext4, `data=writeback`, CI) the last page or two of 2
        // to 6 files in 256, never on 6.18 locally, stable over time, gone after a
        // sync-and-DONTNEED pass. At most one folio of two pages per sampled file, then.
        assert!(r[0].resident <= 2 * r[0].files, "more than a tail folio per file resident after datagen on {fstype}: {r:?}");
        if r[0].resident > 0 {
            eprintln!("note: {} tail page(s) resident after datagen on {fstype} (this kernel keeps the truncated tail folio)", r[0].resident);
        }
    }
    warm(&root);
    let r = cold::residency(model, &root).unwrap();
    assert_eq!(r[0].resident, r[0].pages, "just read: {r:?}");
    let mut o = opts(&root, BackendKind::Sync);
    let plain = cold::start(model, &o).unwrap();
    assert!(plain.residency.is_empty() && plain.dropped.is_none(), "neither flag: no drop, no sample, no opens");
    let mut text = Vec::new();
    cold::write(&mut text, &plain).unwrap();
    assert!(String::from_utf8(text).unwrap().contains("not sampled"));
    o.require_cold = true;
    let e = format!("{:#}", cold::start(model, &o).unwrap_err());
    assert!(e.contains("--require-cold") && e.contains("--drop-caches"), "{e}");

    evict(&root);
    let r2 = cold::residency(model, &root).unwrap();
    if fstype == "tmpfs" {
        assert_eq!(r2[0].resident, r2[0].pages);
    } else {
        assert_eq!(r2[0].resident, 0, "evicted on {fstype}: {r2:?}");
        let c = cold::start(model, &o).unwrap();
        assert!(c.dropped.is_none());
        let mut text = Vec::new();
        cold::write(&mut text, &c).unwrap();
        let text = String::from_utf8(text).unwrap();
        assert!(text.contains("caches not dropped") && !text.contains("not sampled") && text.contains("0 of") && !text.contains("WARNING"), "{text}");
    }
    // the mount's options are in the counters on any filesystem
    let m = aeiou::counters::MountSnapshot::for_path(&root).unwrap();
    assert!(m.opts.as_deref().map(|o| o.starts_with("rw") || o.starts_with("ro")).unwrap_or(false), "{m:?}");
}

/// The 4 KiB-block dedupe ratio of the data files under `root`: their bytes over the unique
/// blocks' (what a fixed-block deduplicating store keeps).
fn block_dedupe(root: &std::path::Path) -> f64 {
    use sha2::Digest;
    fn walk(d: &std::path::Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if !p.file_name().unwrap().to_string_lossy().starts_with(".aeiou") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(root, &mut files);
    let (mut total, mut unique) = (0usize, 0usize);
    let mut seen = std::collections::HashSet::new();
    for f in files {
        for block in std::fs::read(&f).unwrap().chunks(4096) {
            total += block.len();
            if seen.insert(sha2::Sha256::digest(block)) {
                unique += block.len();
            }
        }
    }
    total as f64 / unique as f64
}

fn datagen_into(name: &str, params: &[(&str, &str)], dedupe: u64, root: &std::path::Path) -> &'static Model<'static> {
    let (l, c, m) = leaked_model(name, config(1, 1, params));
    let p = Params::new(&l.ast, c).unwrap();
    let opts = DatagenOpts { root: root.to_path_buf(), endpoints: Default::default(), writers: 2, dedupe, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false };
    datagen(l, c, &p, m, &opts, None, &mut Vec::new()).unwrap();
    m
}

#[test]
fn dedupe_sets_files_of_near_sizes_and_does_not_depend_on_the_count() {
    // aeiou-positional/4 (DESIGN_REVIEW.md §3.77): a file of one sample shares its unit with the
    // files of its set, `D` ids of near sizes among 8 × D (ordered by the size draws, those past
    // the count too), so file `id`'s bytes are the same under any count and a set's files are
    // prefixes of its largest
    let forty = tmpdir("dd40");
    let seventeen = tmpdir("dd17");
    let m40 = datagen_into("file_per_sample", &crate::common::file_per_sample(&[("samples", "40")]), 2, &forty);
    datagen_into("file_per_sample", &crate::common::file_per_sample(&[("samples", "17")]), 2, &seventeen);
    let file = |root: &PathBuf, id: i64| std::fs::read(root.join(m40.datasets[0].file_path(id, None).unwrap().as_ref())).unwrap();
    let shares = |a: &[u8], b: &[u8]| {
        let n = a.len().min(b.len());
        a[..n] == b[..n]
    };
    for id in 0..17 {
        assert_eq!(file(&forty, id), file(&seventeen, id), "file {id} under count 40 and count 17");
    }
    // every file of the first 32 ids (whole groups) shares a prefix with exactly one other
    let mut not_consecutive = 0;
    for id in 0..32 {
        let a = file(&forty, id);
        let partners: Vec<i64> = (0..40).filter(|&j| j != id && shares(&a, &file(&forty, j))).collect();
        assert_eq!(partners.len(), 1, "file {id}: {partners:?}");
        assert_eq!(partners[0] / 16, id / 16, "file {id}: a set lies in its group of 16 ids");
        not_consecutive += (partners[0] / 2 != id / 2) as usize;
    }
    assert!(not_consecutive > 0, "the sets are by size, not `id div D`");
    let m = aeiou::payload::Manifest::read(&forty.join("train")).unwrap();
    assert_eq!(m.payload.wrapper, "aeiou-positional/4");
    assert_eq!(m.payload.dedupe, 2);
}

#[test]
fn dedupe_gives_near_the_ratio_in_4_kib_blocks() {
    // R33: `aeiou-positional/3` laid a container's samples end to end, so the framing moved the
    // shared bytes off the 4 KiB grid (TFRecord at `--dedupe 2`: 1.07), and grouped `id div D`,
    // so files of lognormal sizes kept the larger's tail (file_per_sample: 1.6)
    let whole = tmpdir("ddr-whole");
    datagen_into("file_per_sample", &crate::common::file_per_sample(&[("samples", "320")]), 2, &whole);
    let r = block_dedupe(&whole);
    assert!(r > 1.8 && r <= 2.0, "file_per_sample --dedupe 2: {r}");
    let three = tmpdir("ddr-three");
    datagen_into("file_per_sample", &crate::common::file_per_sample(&[("samples", "240")]), 3, &three);
    let r = block_dedupe(&three);
    assert!(r > 2.6 && r <= 3.0, "file_per_sample --dedupe 3: {r}");
    #[cfg(feature = "formats")]
    {
        let tfr = tmpdir("ddr-tfr");
        datagen_into("tfrecord_stream", &[("samples", "256"), ("samples_per_file", "64"), ("sample_median", "112640"), ("batch", "256"), ("cycle", "4"), ("prefetch", "1"), ("steps", "500"), ("step_time", "105000000"), ("decode", "0")], 2, &tfr);
        let r = block_dedupe(&tfr);
        // a block holding a record's framing differs, and so does the larger file's tail
        assert!(r > 1.75 && r <= 2.0, "tfrecord_stream --dedupe 2: {r}");
    }
}

/// A corpus larger than the free space at its root is refused before anything is written
/// (`DESIGN_REVIEW.md` §3.74 item 4): a million files of a TiB fit on no test host.
#[test]
fn a_corpus_larger_than_the_free_space_is_refused_before_anything_is_written() {
    let root = tmpdir("space");
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(
        aeiou::load_str(
            r#"{"ast": "0.13", "name": "t",
                "datasets": {"big": {"layout": "whole", "access": "map", "pattern": "big/{id div 1000:04}/f_{id}", "count": 1000000, "size": {"const": 1099511627776}, "seed": 1}},
                "actors": {"a": {"body": [{"stat": {"file": {"file": {"dataset": "big", "id": 0}}}}]}}}"#,
        )
        .unwrap(),
    ));
    let cfg: &'static Config = Box::leak(Box::new(config(1, 1, &[])));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let p = aeiou::project::project(&model.datasets[0], 2).unwrap();
    assert_eq!((p.samples, p.files, p.directories, p.bytes), (1_000_000, 1_000_000, 1000, 1_000_000 << 40));
    let opts = DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 1, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false };
    let Err(e) = datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()) else { panic!("written") };
    let e = format!("{e:#}");
    assert!(e.contains("space: dataset(s) `big`: 976.56 PiB to write,") && e.contains("the corpus does not fit (--ignore-space writes anyway"), "{e}");
    assert!(!root.join("big").exists());
    // the check is the lines datagen prints, and --ignore-space keeps the line
    let lines = aeiou::datagen::check_space(loaded, model, &opts, true).unwrap();
    assert!(lines.len() == 1 && lines[0].starts_with("space: IGNORED: dataset(s) `big`"), "{lines:?}");
}

/// The arrival that completes a generation releases it and returns at once: it used to wait
/// the 50 ms poll first, with no one left to wake it, so every barrier of a single-host run
/// cost 50 ms (2,000 barriers of 8 GPUs took 105 s; DESIGN_REVIEW §3.71 point 4).
#[test]
fn a_local_barrier_releases_without_waiting_out_its_poll() {
    use aeiou::coord::{Coordinator, Local};
    use std::sync::atomic::AtomicBool;
    let aborted = AtomicBool::new(false);
    let one = Local::new(&[("global".into(), 1)]);
    let t = std::time::Instant::now();
    for _ in 0..20 {
        one.barrier("global", &aborted).unwrap();
    }
    assert!(t.elapsed() < std::time::Duration::from_millis(500), "20 barriers of one participant took {:?}", t.elapsed());
    let four = std::sync::Arc::new(Local::new(&[("global".into(), 4)]));
    let aborted = std::sync::Arc::new(aborted);
    let t = std::time::Instant::now();
    let hs: Vec<_> = (0..4)
        .map(|_| {
            let (c, a) = (four.clone(), aborted.clone());
            std::thread::spawn(move || (0..100).for_each(|_| { c.barrier("global", &a).unwrap(); }))
        })
        .collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    assert!(t.elapsed() < std::time::Duration::from_secs(2), "100 barriers of four threads took {:?}", t.elapsed());
}

/// A sub-actor starts from the parent's open files and as-written sums at the fork, and
/// nothing it does moves the parent's: the dry run walked the sub-actors inline in the
/// parent's state, so a sub-actor's open and read of the parent's file moved the parent's
/// position, and the dry run's fingerprint was not the stream `run` issued (RC1 review).
/// An op at the file position of a handle the fork body did not open is refused (V24): the
/// sub-actors would share the descriptor's position, and the offsets would depend on timing.
#[test]
fn a_fork_leaves_the_parents_file_positions_as_run_does() {
    const AST: &str = r#"{"ast":"0.13","name":"fork_positions",
      "datasets":{"d":{"pattern":"d/f{id:04}","layout":"whole","access":"indexed","count":1,"size":{"const":8192},"seed":7}},
      "actors":{"gpu":{"body":[
        {"open":{"file":{"file":{"dataset":"d","id":0}},"flags":["RDONLY"]}},
        {"read":{"file":{"file":{"dataset":"d","id":0}},"len":2048}},
        {"parallel":{"index":"p","width":3,"body":[
          {"open":{"file":{"file":{"dataset":"d","id":0}},"flags":["RDONLY"]}},
          {"read":{"file":{"file":{"dataset":"d","id":0}},"len":1024,"repeat":{"add":[{"index":"p"},1]}}},
          {"read":{"file":{"file":{"dataset":"d","id":0}},"len":512,"offset":7680}},
          {"close":{"file":{"file":{"dataset":"d","id":0}}}}
        ]}},
        {"read":{"file":{"file":{"dataset":"d","id":0}},"len":1024}},
        {"close":{"file":{"file":{"dataset":"d","id":0}}}}
      ]}}}"#;
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(AST).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let root = tmpdir("fork-pos");
    gen(loaded, cfg, model, &root);
    let (fp, ops, bytes) = dry_fingerprint(model);
    // the parent's last read is at 2048, where its own read left it
    assert_eq!((ops, bytes), (1 + 1 + 3 * 3 + 6 + 1 + 1, 2048 + 6 * 1024 + 3 * 512 + 1024));
    for backend in [BackendKind::Sync, BackendKind::Uring] {
        let r = go(model, opts(&root, backend));
        assert_eq!(r.stats.fingerprint, fp, "{backend:?}: the dry run is the stream run issues");
        assert_eq!((r.stats.ops, r.stats.bytes_read), (ops, bytes), "{backend:?}");
    }
    // without the sub-actor's own open, its read would be at the inherited descriptor's position
    let shared = AST.replacen(r#"{"open":{"file":{"file":{"dataset":"d","id":0}},"flags":["RDONLY"]}},
          {"read""#, r#"{"read""#, 1);
    assert_ne!(shared, AST);
    let e = format!("{:#}", aeiou::load_str(&shared).err().expect("refused"));
    assert!(e.contains("parallel/body/0/read/file") && e.contains("(V24)"), "{e}");
}

/// A loader worker whose op cannot get its buffer ends the run with the op and the size
/// named: the allocation panicked on the worker's thread, the panic skipped setting
/// `aborted` and failing the channel, and the consumer waited on its take forever (RC1
/// review). The run is driven on a thread and given a wall-clock limit, so a hang fails.
#[test]
fn an_op_whose_buffer_cannot_be_allocated_ends_the_run_with_its_error() {
    let root = tmpdir("alloc");
    let params = [("samples", "40"), ("batch", "2"), ("workers", "2"), ("steps", "4"), ("hdr_read", "70368744177664")];
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(1, 5, &crate::common::file_per_sample(&params)));
    gen(loaded, cfg, model, &root);
    for backend in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio] {
        let o = opts(&root, backend);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(run::run(model, o, std::collections::HashMap::new()).map(|_| ()).map_err(|e| format!("{e:#}")));
        });
        let r = rx.recv_timeout(std::time::Duration::from_secs(20)).unwrap_or_else(|_| panic!("{backend:?}: the run did not end within 20 s"));
        let e = r.expect_err("the run fails");
        assert!(e.contains(" read ") && e.contains("len=70368744177664: buffer allocation of 70368744177664 bytes failed"), "{backend:?}: {e}");
    }
}

/// A failed run reports its cause, never a thread that only saw the abort, whatever order the
/// threads end and are joined in (RC1 review: about one run in ten of the test above reported
/// "take `batches`: producer failed: run aborted: another actor failed"). Each part fails one
/// thread on a read whose buffer cannot be allocated, and has another, joined or read first,
/// wait out a longer compute and then meet the abort: loader worker 1 failed the channel again
/// with the abort, over worker 0's error, before the consumer's take; sub-actor 0 of a
/// `parallel` was the fork's error ahead of sub-actor 1's; actor 0 was joined before actor 1.
#[test]
fn a_failed_run_reports_its_cause_not_the_abort_it_raised() {
    let root = tmpdir("cause");
    let huge = r#"{"open":{"file":{"file":{"dataset":"d","id":0}},"flags":["RDONLY"]}},
        {"read":{"file":{"file":{"dataset":"d","id":0}},"len":70368744177664}},
        {"close":{"file":{"file":{"dataset":"d","id":0}}}}"#;
    // the thread whose `who` is 0 fails at 100 ms, when every thread has started (a loader
    // worker that starts after the failure finds the channel closed and ends quietly); the
    // others meet the abort at 400 ms, and the consumer takes at 900 ms
    let wait = |who: &str| format!(r#"{{"compute":{{"ns":{{"add":[100000000,{{"mul":[{who},300000000]}}]}}}}}}"#);
    let bodies = [
        ("loader", 1, format!(r#"{{"loader":{{"name":"q","index":"b","workers":2,"prefetch":1,"batches":2,"body":[{}, {huge}]}}}},
            {{"compute":{{"ns":900000000}}}},
            {{"loop":{{"index":"s","to":2,"body":[{{"take":{{"channel":"q"}}}}]}}}}"#, wait(r#"{"index":"b"}"#))),
        ("parallel", 1, format!(r#"{{"parallel":{{"index":"p","width":2,"body":[{}, {huge}]}}}}"#, wait(r#"{"sub":[1,{"index":"p"}]}"#))),
        ("actors", 2, format!("{}, {huge}", wait(r#"{"sub":[1,{"actor":"id"}]}"#))),
    ];
    let mut wrong = Vec::new();
    for (what, gpus, body) in bodies {
        let ast = format!(
            r#"{{"ast":"0.13","name":"cause",
              "datasets":{{"d":{{"pattern":"d/f{{id:04}}","layout":"whole","access":"indexed","count":1,"size":{{"const":8192}},"seed":7}}}},
              "actors":{{"gpu":{{"body":[{body}]}}}}}}"#
        );
        let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(&ast).unwrap_or_else(|e| panic!("{what}: {e:#}"))));
        let cfg: &'static Config = Box::leak(Box::new(config(gpus, 0, &[])));
        let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
        let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
        let data = root.join(what);
        gen(loaded, cfg, model, &data);
        for backend in [BackendKind::Sync, BackendKind::Uring] {
            let mut o = opts(&data, backend);
            o.time_scale = 1.0;
            let e = format!("{:#}", run::run(model, o, std::collections::HashMap::new()).map(|_| ()).expect_err("the run fails"));
            if !e.contains("buffer allocation of 70368744177664 bytes failed") || e.contains("run aborted") {
                wrong.push(format!("{what}, {backend:?}: {e}"));
            }
        }
    }
    assert!(wrong.is_empty(), "the abort reported, not its cause:\n{}", wrong.join("\n"));
}

/// An `array` file is sparse (`DESIGN_REVIEW.md` §3.72): datagen writes each slot's content and
/// leaves its tail a hole, so what it writes and counts is the projection's bytes (`gen`), the
/// manifest's `bytes_written` is the same exactly, the file's length is still every slot's, its
/// allocation is about the contents where the file system has holes, each content is the bytes a
/// dense file holds at its offsets (the dataset id is unchanged), each tail reads as zeros, and a
/// run reads only contents. The lists span two 64 MiB pieces.
#[test]
fn array_datagen_writes_only_the_slots_contents() {
    use std::os::unix::fs::MetadataExt;
    let root = tmpdir("faiss-sparse");
    let params = crate::common::faiss(&[("lists", "300"), ("index_bytes", "4096"), ("calls", "20"), ("threads", "1")]);
    let (loaded, cfg, model) = leaked_model("faiss_ivf_ondisk", config(1, 9, &params));
    let meta = model.datasets.iter().find(|m| m.name() == "lists").unwrap();
    let p = aeiou::project::project(meta, 2).unwrap();
    let (slot, count) = (256u64 << 10, 300u64);
    let size = meta.file_size(0).unwrap() as u64;
    assert!(p.bytes * 3 < size, "short lists in long slots: {} of {size}", p.bytes);
    gen(loaded, cfg, model, &root);
    let m = aeiou::payload::Manifest::read(&root.join("ivf")).unwrap();
    assert_eq!(m.provenance["bytes_written"].as_u64(), Some(p.bytes), "the manifest counts the contents, as the projection does");
    let path = root.join("ivf/merged_index.ivfdata");
    let md = std::fs::metadata(&path).unwrap();
    assert_eq!(md.len(), size);
    // a file system with holes allocates the contents and at most two partial 4 KiB blocks each
    let probe = root.join("probe");
    std::fs::File::create(&probe).unwrap().set_len(1 << 20).unwrap();
    if std::fs::metadata(&probe).unwrap().blocks() == 0 {
        let allocated = md.blocks() * 512;
        assert!(allocated >= p.bytes && allocated <= p.bytes + count * 8192, "allocated {allocated}, contents {}", p.bytes);
    }
    std::fs::remove_file(&probe).unwrap();
    // the dense writer's stream: the payload of each file offset, a block at a time
    let seed = meta.dataset_seed();
    let seed_of = |blk: u64| aeiou::payload::array_block_seed(seed, blk, 1);
    let mut filler = aeiou::payload::Filler::new(1);
    let mut dense = vec![0u8; size as usize];
    for (i, b) in dense.chunks_mut(aeiou::payload::BLOCK as usize).enumerate() {
        filler.fill_range(&seed_of, i as u64 * aeiou::payload::BLOCK, b);
    }
    let file = std::fs::read(&path).unwrap();
    for c in 0..count {
        let (at, n) = ((c * slot) as usize, meta.sample_size(c as i64).unwrap() as usize);
        assert!(file[at..at + n] == dense[at..at + n], "slot {c}: its content is the dense file's");
        let end = ((c + 1) * slot).min(size) as usize;
        assert!(file[at + n..end].iter().all(|&b| b == 0), "slot {c}: its tail is a hole");
    }
    run::check_datasets(loaded, cfg, &root).unwrap();
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    let r = go(model, opts(&root, BackendKind::Mmap));
    assert_eq!((r.stats.fingerprint, r.stats.ops, r.stats.bytes_read), (fp, ops, bytes));
}

/// A test's scratch directory goes when its guard drops, on a panic's unwind as well as at the
/// end of the test: a failing test leaves nothing behind (a failed coordinator test used to
/// leave gigabytes in the temp directory, until the disk filled and every later test failed).
#[test]
fn a_test_directory_is_removed_when_its_test_panics() {
    if std::env::var_os(crate::common::KEEP_TEST_DIRS).is_some_and(|v| !v.is_empty() && v != "0") {
        return;
    }
    let mut seen = None;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let d = tmpdir("guard-panic");
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("sub/f"), b"bytes").unwrap();
        seen = Some(d.to_path_buf());
        panic!("the test fails with its directory full");
    }));
    assert!(r.is_err());
    let d = seen.unwrap();
    assert!(!d.exists(), "{} is left behind by a panicking test", d.display());
    let d = tmpdir("guard-pass");
    let p = d.to_path_buf();
    drop(d);
    assert!(!p.exists(), "{} is left behind by a passing test", p.display());
}

/// A `loader` inside a loop (one per epoch) declares its channel again in every iteration: the
/// last iteration's is retired once all its batches are taken, so the abstract `check` and the
/// dry run accept also runs, under the blocking path and the event loop, with the dry run's
/// fingerprint (RC1 review: the second declaration was refused, "a channel of that name exists").
#[test]
fn a_loader_in_a_loop_runs_once_per_iteration() {
    let root = tmpdir("loader-epochs");
    let mut ast: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(crate::common::workload_ast("file_per_sample")).unwrap()).unwrap();
    let body = ast["actors"]["gpu"]["body"].as_array_mut().unwrap();
    let at = body.iter().position(|n| n.get("loader").is_some()).unwrap();
    let epoch: Vec<serde_json::Value> = body.drain(at..).collect();
    body.push(serde_json::json!({"loop": {"index": "epoch", "to": 2, "body": epoch}}));
    let path = root.join("epochs.ast.json");
    std::fs::write(&path, serde_json::to_string(&ast).unwrap()).unwrap();
    let params = [("samples", "600"), ("batch", "4"), ("workers", "2"), ("prefetch", "2"), ("steps", "12"), ("walk", "false")];
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&path).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(2, 7, &crate::common::file_per_sample(&params))));
    let p: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, p).unwrap()));
    let data = root.join("data");
    gen(loaded, cfg, model, &data);
    run::check_datasets(loaded, cfg, &data).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    for backend in [BackendKind::Sync, BackendKind::Uring] {
        let r = go(model, opts(&data, backend));
        assert_eq!((r.stats.fingerprint, r.stats.ops, r.stats.bytes_read), (fp, ops, bytes), "{backend:?}");
        assert_eq!(r.stats.takes, 2 * 2 * 12, "two epochs of twelve batches on each of two instances");
    }
}

/// A run of consecutive `compute`s takes their sum, on the blocking path and the event loop
/// alike: each is paced from where the previous one ends, not from when its sleep returned,
/// and the timer slack is 1 ns, so the total overshoot is one sleep's, never one per compute
/// (RC1 review: with relative sleeps under the default 50 µs slack, 20,000 computes of 20 µs
/// took 1.9-2.4 s under `sync` and 0.7-0.9 s under `io_uring` for an ideal 0.4 s, so
/// emulated GPU time depended on the backend). Computes shorter than a timer's wake-up are
/// in the set: once behind its deadline the event loop must reap without sleeping, as the
/// blocking path skips the sleep (RC1 review: the loop's 0 ns `io_uring_enter` wait still
/// slept on an hrtimer, about 100 µs on the CI host's VM, so 5,000 computes of 20 µs ran
/// 0.48 s over there, and 40,000 of 2 µs about 0.1 s over here). Each is the best of a few
/// tries (`least_over`): a loaded host's late wake-ups are transient, the overshoot this test
/// guards against is in every run.
#[test]
fn consecutive_computes_take_their_sum_on_every_backend() {
    let root = tmpdir("computes");
    for (n, ns) in [(5000u64, 20_000u64), (40_000, 2000)] {
        let ast = serde_json::json!({"ast": "0.13", "name": "computes", "params": {}, "datasets": {},
            "actors": {"gpu": {"body": [{"loop": {"index": "s", "to": n, "body": [{"compute": {"ns": ns}}]}}]}}});
        let path = root.join(format!("computes-{ns}.ast.json"));
        std::fs::write(&path, serde_json::to_string(&ast).unwrap()).unwrap();
        let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&path).unwrap()));
        let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
        let p: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
        let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, p).unwrap()));
        let ideal = std::time::Duration::from_nanos(n * ns);
        for backend in [BackendKind::Sync, BackendKind::Uring] {
            // the run's setup and one late wake-up on a loaded host, never one per compute
            // (5000 × the 50 µs slack is 250 ms)
            let limit = std::time::Duration::from_millis(50);
            let over = least_over(limit, || {
                let mut o = opts(&root.join("data"), backend);
                o.time_scale = 1.0;
                let r = go(model, o);
                assert_eq!(r.stats.compute_ns, n * ns, "{backend:?}");
                assert!(r.elapsed >= ideal, "{backend:?}: {:?} is shorter than the computes' {ideal:?}", r.elapsed);
                r.elapsed - ideal
            });
            assert!(over < limit, "{backend:?}: {n} computes of {ns} ns took {over:?} over their sum {ideal:?} at the least of {TRIES} runs");
        }
    }
}

/// A run ends when its work does: the `libaio` loops' `io_destroy` (about 50 ms here, an RCU
/// grace period the kernel waits out at a context's teardown) comes after the last compute
/// and is not the workload's time (RC1 review: one compute of 400 ms took 0.453 s under
/// `libaio`, 0.400 s under `sync`, whatever the number of computes). The best of a few tries
/// (`least_over`), as the teardown is in every run and a loaded host's late wake-up is not.
#[test]
fn libaio_teardown_is_not_in_the_elapsed_time() {
    let root = tmpdir("aio-teardown");
    let ns = 100_000_000u64;
    let ast = serde_json::json!({"ast": "0.13", "name": "teardown", "params": {}, "datasets": {},
        "actors": {"gpu": {"body": [{"compute": {"ns": ns}}]}}});
    let path = root.join("teardown.ast.json");
    std::fs::write(&path, serde_json::to_string(&ast).unwrap()).unwrap();
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&path).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
    let p: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, p).unwrap()));
    let ideal = std::time::Duration::from_nanos(ns);
    for backend in [BackendKind::LibAio, BackendKind::LibAioDirect] {
        let limit = std::time::Duration::from_millis(20);
        let over = least_over(limit, || {
            let mut o = opts(&root.join("data"), backend);
            o.time_scale = 1.0;
            let r = go(model, o);
            assert!(r.elapsed >= ideal, "{backend:?}: {:?} is shorter than the compute's {ideal:?}", r.elapsed);
            r.elapsed - ideal
        });
        assert!(over < limit, "{backend:?}: a compute of {ideal:?} took {over:?} over it at the least of {TRIES} runs");
    }
}

/// Runs, at most, of each measurement of the timing tests above.
const TRIES: usize = 5;

/// The least of up to `TRIES` overshoots `measure` returns, stopping at the first under
/// `limit`. What those tests guard against (a slack paid per compute, a teardown in the
/// elapsed time) is deterministic and in every run, so it fails all of them and the limit
/// keeps its meaning; a late wake-up on a host loaded by the suites beside it is not, and one
/// clean run is enough (RC1 review: under parallel load one run in a few went 24-156 ms over
/// a limit of 20 or 50 ms, with no regression).
fn least_over(limit: std::time::Duration, mut measure: impl FnMut() -> std::time::Duration) -> std::time::Duration {
    let mut least = std::time::Duration::MAX;
    for _ in 0..TRIES {
        least = least.min(measure());
        if least < limit {
            break;
        }
    }
    least
}

/// A thread's buffer ring is grown and faulted in before an op's clock starts: the first read
/// larger than the ring is not charged its allocation and page faults (RC1 review: the first
/// 32 MiB read of a `sync` run took 3.6 to 3.7× the mean of the others, about 1.2× with the
/// ring sized beforehand). The best of three runs, so a busy host's one late wake-up does not
/// decide it; 32 MiB of data, so it does not slow the timing tests that run beside it.
#[test]
fn the_first_large_read_is_not_charged_the_rings_growth() {
    let root = tmpdir("ring-growth");
    let mb16 = 16u64 << 20;
    let ast = serde_json::json!({"ast": "0.13", "name": "ring", "params": {}, "datasets": {},
        "namespaces": {"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 1}},
        "actors": {"gpu": {"body": [
            {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"k": 0}}}}},
            {"open": {"file": {"ref": "f"}, "flags": ["RDWR", "CREAT", "TRUNC"]}},
            {"loop": {"index": "i", "to": 2, "body": [{"write": {"file": {"ref": "f"}, "len": mb16, "offset": {"mul": [{"index": "i"}, mb16]}}}]}},
            {"loop": {"index": "r", "to": 4, "body": [
                {"loop": {"index": "i", "to": 2, "body": [{"read": {"file": {"ref": "f"}, "len": mb16, "offset": {"mul": [{"index": "i"}, mb16]}}}]}}]}},
            {"close": {"file": {"ref": "f"}}}]}}});
    let path = root.join("ring.ast.json");
    std::fs::write(&path, serde_json::to_string(&ast).unwrap()).unwrap();
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&path).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
    let p: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, p).unwrap()));
    let mut ratios = vec![];
    for n in 0..3 {
        let data = root.join(format!("data{n}"));
        std::fs::create_dir_all(data.join("o")).unwrap();
        let r = go(model, opts(&data, BackendKind::Sync));
        let h = &r.stats.lat[&aeiou::vm::OpKind::Read];
        assert_eq!(h.count, 8);
        let others = (h.sum - h.max) as f64 / (h.count - 1) as f64;
        ratios.push(h.max as f64 / others);
        std::fs::remove_dir_all(&data).unwrap();
    }
    let best = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(best < 2.5, "the slowest 16 MiB read took {best:.2}× the mean of the others (runs: {ratios:.2?})");
}

/// An `io_uring` loop that ends with an error while a request of its own is still in the
/// kernel cancels that request and reaps its completion before it lets go of the ring: the loop
/// returned at once, and closing a ring only begins the kernel's teardown, so a request still
/// there could write into memory the loop had freed (the eventfd's count here, a `statx`
/// result or a read's buffer elsewhere; RC1 review). Actor 0 reads and waits at a barrier, on
/// the read posted on the loop's eventfd; actor 1, on the same loop, fails 100 ms later on a
/// read whose buffer cannot be allocated, and never arrives.
#[test]
fn a_failed_io_uring_loop_drains_its_ring_before_it_ends() {
    let ast = r#"{"ast":"0.13","name":"drain",
      "datasets":{"d":{"pattern":"d/f{id:04}","layout":"whole","access":"indexed","count":2,"size":{"const":8192},"seed":7}},
      "actors":{"gpu":{"body":[
        {"compute":{"ns":{"mul":[{"actor":"id"},100000000]}}},
        {"open":{"file":{"file":{"dataset":"d","id":{"actor":"id"}}},"flags":["RDONLY"]}},
        {"read":{"file":{"file":{"dataset":"d","id":{"actor":"id"}}},"len":{"add":[8,{"mul":[{"actor":"id"},70368744177656]}]}}},
        {"barrier":{"scope":"global"}},
        {"close":{"file":{"file":{"dataset":"d","id":{"actor":"id"}}}}}
      ]}}}"#;
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(ast).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(2, 0, &[])));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let root = tmpdir("drain");
    gen(loaded, cfg, model, &root);
    let mut o = opts(&root, BackendKind::Uring);
    o.threads = 1;
    o.time_scale = 1.0;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run::run(model, o, std::collections::HashMap::new()).map(|_| ()).map_err(|e| format!("{e:#}")));
    });
    let r = rx.recv_timeout(std::time::Duration::from_secs(20)).expect("the run ends within 20 s");
    let e = r.expect_err("the run fails");
    assert!(e.contains("gpu#1") && e.contains("buffer allocation of 70368744177664 bytes failed"), "{e}");
    assert_eq!(aeiou::uring::undrained(), 0, "a ring was closed with a request still in the kernel");
}

/// The stream abstracts take the shards whose first sample is in a step's batch, so the steps
/// take exactly the loader's `ceil_div(steps · batch, samples_per_file)` shards for any two
/// positive values: a `batch` that divides `samples_per_file`, one that does not (once a take
/// after the last shard, `every(samples_per_file div batch)` firing too often), and one above
/// it (once a `mod` by zero, `samples_per_file div batch` being 0); RC1 review.
#[cfg(feature = "formats")]
#[test]
fn stream_abstracts_take_every_shard_for_any_batch() {
    for name in ["tfrecord_stream", "parquet_stream"] {
        for (batch, steps) in [("32", "8"), ("48", "8"), ("200", "4")] {
            let root = tmpdir(&format!("stream-{name}-{batch}"));
            let params = [("samples", "768"), ("samples_per_file", "128"), ("sample_median", "20000"), ("cycle", "2"), ("prefetch", "1"), ("step_time", "105000000"), ("decode", "0"), ("batch", batch), ("steps", steps)];
            let (loaded, cfg, model) = leaked_model(name, config(1, 5, &params));
            gen(loaded, cfg, model, &root);
            run::check_datasets(loaded, cfg, &root).unwrap();
            run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
            let (fp, ops, bytes) = dry_fingerprint(model);
            let r = go(model, opts(&root, BackendKind::Sync));
            let shards = (steps.parse::<u64>().unwrap() * batch.parse::<u64>().unwrap()).div_ceil(128);
            assert_eq!(r.stats.takes, shards, "{name} batch {batch}: one take per shard dispatched");
            assert_eq!((r.stats.fingerprint, r.stats.ops, r.stats.bytes_read), (fp, ops, bytes), "{name} batch {batch}");
        }
    }
}

/// A `channel` or `loader` declared inside a `parallel` or `loader` body is refused (V26): the
/// channel is the actor's, shared by its sub-actors, and since a channel in a loop may be
/// declared again once it is empty (bug 37), each sub-actor's declaration replaced the
/// other's when it happened to be empty, so `run` failed ("declared again with 1 item(s) put
/// and not taken"), hung, or passed as timing had it (RC1 regression review). The shape that
/// means it, the actor declaring the channel in each iteration and its sub-actors putting, one
/// of them taking (a channel has one taker at a time), runs alike under both engines whatever
/// the timing.
#[test]
fn a_channel_declared_inside_a_fork_is_refused_whatever_the_timing() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"ch_fork","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let sub = |decl: &str| {
        format!(
            r#"{{"parallel":{{"index":"p","width":2,"body":[
                {{"compute":{{"ns":{{"mul":[{{"index":"p"}},50000000]}}}}}},
                {decl}
                {{"put":{{"channel":"c","seq":{{"index":"p"}}}}}},
                {{"compute":{{"ns":100000000}}}},
                {{"cond":{{"if":{{"eq":[{{"index":"p"}},0]}},"then":[{{"take":{{"channel":"c"}}}},{{"take":{{"channel":"c"}}}}]}}}}]}}}}"#
        )
    };
    let chan = r#"{"channel":{"name":"c","capacity":1}},"#;
    for bad in [doc(&sub(chan)), doc(&sub(r#"{"loader":{"name":"c","index":"b","workers":1,"prefetch":1,"batches":1,"body":[{"compute":{"ns":1000}}]}},"#))] {
        let e = format!("{:#}", aeiou::load_str(&bad).err().expect("refused"));
        assert!(e.contains("parallel/body/1/") && e.contains("inside a `parallel` or `loader` body") && e.contains("(V26)"), "{e}");
    }
    let good = doc(&format!(r#"{{"loop":{{"index":"e","to":3,"body":[{chan} {}]}}}}"#, sub("")));
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(&good).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let root = tmpdir("ch-fork");
    for backend in [BackendKind::Sync, BackendKind::Uring] {
        for scale in [0.0, 0.2] {
            let mut o = opts(&root, backend);
            o.time_scale = scale;
            let r = go(model, o);
            assert_eq!((r.stats.puts, r.stats.takes), (6, 6), "{backend:?}, time scale {scale}");
        }
    }
}

/// A channel a loader's workers put on, declared again by the actor while they ran (in a loop
/// the loader was declared outside of), was the old one or the new one as timing had it: at
/// `--time-scale 0` the run failed ("declared again with 2 item(s) put and not taken"), at 1 it
/// passed (RC1 regression review). It is refused (V26); the shapes that mean it, the channel
/// declared once or beside the loader in each iteration, run alike under both engines.
#[test]
fn a_channel_a_loaders_workers_use_is_not_declared_again_under_them() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"ch_loader","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let chan = r#"{"channel":{"name":"c","capacity":8}}"#;
    let lput = r#"{"loader":{"name":"q","index":"b","workers":2,"prefetch":2,"batches":4,"body":[
        {"compute":{"ns":{"mul":[{"index":"b"},10000000]}}},
        {"put":{"channel":"c","seq":{"index":"b"}}}]}}"#;
    let takes = r#"{"loop":{"index":"s","to":4,"body":[{"take":{"channel":"q"}},{"take":{"channel":"c"}}]}}"#;
    let bad = doc(&format!(r#"{chan}, {lput}, {{"loop":{{"index":"e","to":4,"body":[{chan}, {{"take":{{"channel":"q"}}}}, {{"take":{{"channel":"c"}}}}]}}}}"#));
    let e = format!("{:#}", aeiou::load_str(&bad).err().expect("refused"));
    assert!(e.contains("actors/gpu/body/1/loader/body/1/put: a `put` on channel `c` inside a `loader` body") && e.contains("(V26)"), "{e}");
    let root = tmpdir("ch-loader");
    for good in [
        // unordered: every iteration's batches are put on the one channel, numbered across the
        // iterations (a `seq` is put once while its channel is declared, 2026-10-15)
        doc(&format!(r#"{}, {{"loop":{{"index":"e","to":3,"body":[{}, {takes}]}}}}"#, chan.replace("8}", r#"8,"ordered":false}"#), lput.replace(r#""seq":{"index":"b"}"#, r#""seq":{"add":[{"mul":[{"index":"e"},4]},{"index":"b"}]}"#))),
        doc(&format!(r#"{{"loop":{{"index":"e","to":3,"body":[{chan}, {lput}, {takes}]}}}}"#)),
    ] {
        let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(&good).unwrap()));
        let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
        let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
        let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
        for backend in [BackendKind::Sync, BackendKind::Uring] {
            for scale in [0.0, 0.2] {
                let mut o = opts(&root, backend);
                o.time_scale = scale;
                let r = go(model, o);
                assert_eq!((r.stats.puts, r.stats.takes), (12, 24), "{backend:?}, time scale {scale}");
            }
        }
    }
}

/// An abstract given as text, leaked as a run needs it.
fn model_of(doc: &str) -> &'static Model<'static> {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(doc).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1, 0, &[])));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()))
}

/// The run, or its error, or "hung" when it has not ended in 30 s (its threads are left
/// behind; the test fails either way).
fn run_within(model: &'static Model<'static>, o: RunOpts) -> Result<run::Report, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run::run(model, o, std::collections::HashMap::new()).map_err(|e| format!("{e:#}")));
    });
    rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap_or_else(|_| Err("hung".into()))
}

/// Loader workers putting on a channel smaller than their batches in flight (the RC1 review's
/// R12: 2 workers × prefetch 2 on a channel of 2, the main line taking the loader's batch and
/// then the channel's item) hung in some runs and finished in others under `sync`, as the
/// workers happened to run: a later batch's put filled the channel while batch 0's waited, and
/// the main line waited for batch 0. The loader written out in the primitives
/// (`GRAMMAR_OPTIONS.md` §3), two producers of an ordered channel of 2, hung or failed alike.
/// A put waits for its position since 2026-10-15 (`run::admits`), so each of these finishes,
/// every time, under both engines.
#[test]
fn a_put_waits_for_its_position_so_no_run_hangs_by_timing() {
    let doc = |name: &str, body: &str| format!(r#"{{"ast":"0.13","name":"{name}","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let loader = |cap: i64, ordered: bool, epochs: i64, seq: &str| doc("r12", &format!(r#"
        {{"channel":{{"name":"c","capacity":{cap},"ordered":{ordered}}}}},
        {{"loop":{{"index":"e","to":{epochs},"body":[
          {{"loader":{{"name":"q","index":"b","workers":4,"prefetch":2,"batches":16,"body":[{{"put":{{"channel":"c","seq":{seq}}}}}]}}}},
          {{"loop":{{"index":"s","to":16,"body":[{{"take":{{"channel":"q"}}}},{{"take":{{"channel":"c"}}}}]}}}}]}}}}"#));
    let across = r#"{"add":[{"mul":[{"index":"e"},16]},{"index":"b"}]}"#;
    let cases = [
        (loader(1, false, 6, across), 96),
        (loader(1, true, 6, across), 96),
        (loader(3, false, 1, r#"{"index":"b"}"#), 16),
        (doc("desugared", r#"{"channel":{"name":"c","capacity":2,"ordered":true}},
            {"parallel":{"index":"w","width":3,"body":[{"cond":{"if":{"eq":[{"index":"w"},0]},
              "then":[{"loop":{"index":"s","to":40,"body":[{"take":{"channel":"c"}}]}}],
              "else":[{"loop":{"index":"k","to":20,"body":[{"put":{"channel":"c","seq":{"add":[{"mul":[{"index":"k"},2]},{"sub":[{"index":"w"},1]}]}}}]}}]}}]}}"#), 40),
    ];
    let root = tmpdir("ch-position");
    for (d, puts) in &cases {
        let model = model_of(d);
        for backend in [BackendKind::Sync, BackendKind::Uring] {
            for scale in [0.0, 0.0, 0.0, 0.0001] {
                let mut o = opts(&root, backend);
                o.time_scale = scale;
                let r = run_within(model, o).unwrap_or_else(|e| panic!("{backend:?}, time scale {scale}: {e}\n{d}"));
                assert_eq!(r.stats.puts, *puts, "{backend:?}, time scale {scale}");
            }
        }
    }
}

/// An ordered channel declared once whose per-epoch loader put `seq b` again each epoch (the RC1
/// review's R13) waited forever under `sync` for an item numbered past the repeats, and failed
/// under the event loop only as "actor(s) wait on channels"; a seq below 0 was taken as given.
/// Both are errors naming the put, its channel, and the seq, under both engines (2026-10-15).
/// And a wait no thread can end, every thread of the actor parked on its channels, is an error
/// under the thread runner as it was under the event loop: it had hung.
#[test]
fn a_repeated_seq_and_a_wait_nothing_ends_are_errors_not_hangs() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"r13","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let repeated = doc(r#"{"channel":{"name":"c","capacity":4}},
        {"loop":{"index":"e","to":2,"body":[
          {"loader":{"name":"q","index":"b","workers":2,"prefetch":1,"batches":4,"body":[{"put":{"channel":"c","seq":{"index":"b"}}}]}},
          {"loop":{"index":"s","to":4,"body":[{"take":{"channel":"q"}},{"take":{"channel":"c"}}]}}]}}"#);
    let below = doc(r#"{"channel":{"name":"c","capacity":1}}, {"put":{"channel":"c","seq":-1}}, {"take":{"channel":"c"}}"#);
    let (repeated, below) = (model_of(&repeated), model_of(&below));
    let root = tmpdir("ch-seq");
    for backend in [BackendKind::Sync, BackendKind::Uring] {
        for scale in [0.0, 0.0001] {
            let o = || {
                let mut o = opts(&root, backend);
                o.time_scale = scale;
                o
            };
            // which of the second epoch's seqs 0 and 1 is put first is the workers' timing
            let e = run_within(repeated, o()).err().expect("refused");
            assert!(["put 0 on `c`: seq 0", "put 1 on `c`: seq 1"].iter().any(|s| e.contains(&format!("{s} was put on this channel before"))), "{backend:?}, time scale {scale}: {e}");
            let e = run_within(below, o()).err().expect("refused");
            assert!(e.contains("put -1 on `c`: seq -1 is below 0"), "{backend:?}, time scale {scale}: {e}");
        }
    }
    // waits nothing ends: an ordered channel's item 0 never put; an unordered channel's items
    // numbered with gaps (item 2 waits for item 1, never put); a
    // sub-actor taking from a channel no one puts on, its parent waiting for it
    let stuck = [
        doc(r#"{"channel":{"name":"c","capacity":4}}, {"put":{"channel":"c","seq":1}}, {"take":{"channel":"c"}}"#),
        doc(r#"{"channel":{"name":"c","capacity":1,"ordered":false}},
            {"loader":{"name":"q","index":"b","workers":2,"prefetch":2,"batches":4,"body":[{"put":{"channel":"c","seq":{"mul":[{"index":"b"},2]}}}]}},
            {"loop":{"index":"s","to":4,"body":[{"take":{"channel":"q"}},{"take":{"channel":"c"}}]}}"#),
        doc(r#"{"channel":{"name":"c","capacity":1}}, {"parallel":{"index":"w","width":3,"body":[{"cond":{"if":{"eq":[{"index":"w"},2]},"then":[{"take":{"channel":"c"}}]}}]}}"#),
    ];
    for d in &stuck {
        let model = model_of(d);
        for backend in [BackendKind::Sync, BackendKind::Uring] {
            let e = run_within(model, opts(&root, backend)).err().expect("refused");
            let says = match backend {
                BackendKind::Sync => "every thread of gpu#0 waits on a channel that nothing will change",
                _ => "wait on channels or trace order nothing will complete",
            };
            assert!(e.contains(says), "{backend:?}: {e}\n{d}");
        }
    }
}

/// One instance of an event loop stuck on its channels while another of the same loop waits
/// at a barrier the stuck one must reach (the RC1 review's R14: gpu 1 takes from a channel no
/// one puts on, gpu 0 goes on to a global barrier) hung under io_uring and libaio with both
/// instances on one loop: the loop judged only as a whole, and the barrier's waiter kept it
/// from looking. The stuck instance is an error now whatever its loop's other instances do,
/// as it was under the thread runner (2026-10-15).
#[test]
fn an_instance_stuck_on_its_channels_is_an_error_while_a_sibling_waits_at_a_barrier() {
    let doc = r#"{"ast":"0.13","name":"r14","actors":{"gpu":{"body":[{"channel":{"name":"c","capacity":1}},
        {"cond":{"if":{"eq":[{"actor":"id"},1]},"then":[{"take":{"channel":"c"}}]}},{"barrier":{"scope":"global"}}]}}}"#;
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(doc).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(2, 0, &[])));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let root = tmpdir("stuck-sibling");
    for (backend, threads) in [(BackendKind::Sync, 0), (BackendKind::Uring, 1), (BackendKind::Uring, 2), (BackendKind::LibAio, 1), (BackendKind::LibAio, 2)] {
        let mut o = opts(&root, backend);
        o.threads = threads;
        let e = run_within(model, o).err().expect("refused");
        let says = match backend {
            BackendKind::Sync => "every thread of gpu#1 waits on a channel that nothing will change",
            _ => "1 actor(s) wait on channels or trace order nothing will complete: gpu#1 []",
        };
        assert!(e.contains(says), "{backend:?}, {threads} loop(s): {e}");
    }
}

/// A loader worker, or a `parallel` sub-actor inside one, that failed while the main line waited
/// on another channel (the RC1 review's R15) was reported under the thread runner only as
/// "take `c`: run aborted": the main line's error skipped joining the loader workers, so their
/// error, the cause, was dropped, and they could outlive the run. The main line's error now
/// still closes the actor's channels and joins its workers, and the cause wins over the abort
/// (`keep_cause`); a put or take the closing ends is an echo too. Under both engines the error
/// names the put that failed, every time, and a wait nothing ends is named as one, never as the
/// abort or a closed channel.
#[test]
fn a_loader_workers_failure_is_the_cause_whatever_the_main_line_waits_on() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"r15","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    // R6: the main line takes `c` while batch 2 puts -1 on it
    let worker = doc(r#"{"channel":{"name":"c","capacity":4,"ordered":true}},
        {"loader":{"name":"q","index":"b","workers":2,"prefetch":2,"batches":4,"body":[
          {"put":{"channel":"c","seq":{"cond":{"if":{"eq":[{"index":"b"},2]},"then":-1,"else":{"index":"b"}}}}}]}},
        {"loop":{"index":"s","to":4,"body":[{"take":{"channel":"c"}}]}},
        {"loop":{"index":"s","to":4,"body":[{"take":{"channel":"q"}}]}}"#);
    // R5: sub-actor 1 of a `parallel` in the loader's body puts -1 while sub-actor 0 computes
    let sub = doc(r#"{"channel":{"name":"c","capacity":4,"ordered":true}},
        {"loader":{"name":"q","index":"b","workers":1,"prefetch":1,"batches":2,"body":[
          {"parallel":{"index":"p","width":2,"body":[{"cond":{"if":{"eq":[{"index":"p"},1]},
            "then":[{"put":{"channel":"c","seq":-1}}],"else":[{"compute":{"ns":300000000}}]}}]}}]}},
        {"take":{"channel":"q"}},{"take":{"channel":"q"}}"#);
    // the fuzzer's shape: the workers' sub-actors put items the main line takes too late while
    // it waits on the loader, a wait nothing ends
    let takes: Vec<String> = "qqqcqqccccccqccqccccqqcccccccccccccc".chars().map(|c| format!(r#"{{"take":{{"channel":"{c}"}}}}"#)).collect();
    let stuck = doc(&format!(r#"{{"channel":{{"name":"c","capacity":6,"ordered":false}}}},
        {{"loader":{{"name":"q","index":"b","workers":4,"prefetch":2,"batches":9,"body":[
          {{"parallel":{{"index":"p","width":3,"body":[{{"compute":{{"ns":{{"mul":[{{"index":"p"}},3922]}}}}}},
            {{"put":{{"channel":"c","seq":{{"add":[{{"mul":[{{"index":"b"}},3]}},{{"index":"p"}}]}}}}}}]}}}}]}}}},
        {}"#, takes.join(",")));
    let (worker, sub, stuck) = (model_of(&worker), model_of(&sub), model_of(&stuck));
    let root = tmpdir("r15-cause");
    for backend in [BackendKind::Sync, BackendKind::Uring] {
        for scale in [0.0, 0.0, 0.0, 0.001] {
            let o = || {
                let mut o = opts(&root, backend);
                o.time_scale = scale;
                o
            };
            for model in [worker, sub] {
                let e = run_within(model, o()).err().expect("refused");
                assert!(e.contains("put -1 on `c`: seq -1 is below 0"), "{backend:?}, time scale {scale}: {e}");
            }
            let e = run_within(stuck, o()).err().expect("refused");
            let says = match backend {
                BackendKind::Sync => "every thread of gpu#0 waits on a channel that nothing will change",
                _ => "wait on channels or trace order nothing will complete",
            };
            assert!(e.contains(says) && !e.contains("run aborted") && !e.contains("closed channel"), "{backend:?}, time scale {scale}: {e}");
        }
    }
}

/// Two sub-actors of one `parallel` taking on one channel got whichever items they reached
/// first, so which of them went on, and whether the run finished, was timing (the RC1 review's
/// R16: w0 `put c 0; take d; put c 1`, w1 `take c; put d 0`, w2 `take c` failed 1 in 40 runs
/// under `sync`; with w1 and w2 swapped, 36 in 40 under `sync` and 40 in 40 under io_uring).
/// A channel has one taker at a time (2026-10-15): a second sub-actor of the same fork
/// instance taking is an error naming both, in every run, under every engine, since both
/// always take. Sub-actors of later forks (a `parallel` in a loop, a `parallel` inside the
/// taker), and the main line before and after a fork, take in turn and are not refused.
#[test]
fn two_takers_in_one_fork_are_an_error_in_every_run() {
    let doc = |body: &str| format!(r#"{{"ast":"0.13","name":"r16","actors":{{"gpu":{{"body":[{body}]}}}}}}"#);
    let take = r#"{"take":{"channel":"c"}}"#;
    let put = |seq: &str| format!(r#"{{"put":{{"channel":"c","seq":{seq}}}}}"#);
    let when = |index: &str, k: i64, then: &str, otherwise: &str| format!(r#"{{"cond":{{"if":{{"eq":[{{"index":"{index}"}},{k}]}},"then":[{then}],"else":[{otherwise}]}}}}"#);
    let par = |index: &str, width: i64, body: &str| format!(r#"{{"parallel":{{"index":"{index}","width":{width},"body":[{body}]}}}}"#);
    let r16 = |taker: i64| {
        doc(&format!(
            r#"{{"channel":{{"name":"c","capacity":2}}}}, {{"channel":{{"name":"d","capacity":1}}}}, {}"#,
            par("w", 3, &when("w", 0, &format!(r#"{}, {{"take":{{"channel":"d"}}}}, {}"#, put("0"), put("1")), &when("w", taker, &format!(r#"{take}, {{"put":{{"channel":"d","seq":0}}}}"#), take)))
        ))
    };
    let refused = [
        (r16(1), "sub-actors `w` = 1 and 2 of one `parallel` both take on this channel"),
        (r16(2), "sub-actors `w` = 1 and 2 of one `parallel` both take on this channel"),
        // nested: sub-actor 0 of each outer sub-actor's own fork takes, so the outer two collide
        (doc(&format!(r#"{{"channel":{{"name":"c","capacity":2}}}}, {}, {}, {}"#, put("0"), put("1"), par("a", 2, &par("b", 2, &when("b", 0, take, ""))))), "sub-actors `a` = 0 and 1 of one `parallel` both take on this channel"),
        // three takers: which two meet first is timing, that two do is not
        (doc(&format!(r#"{{"channel":{{"name":"c","capacity":3,"ordered":false}}}}, {}, {}, {}, {}"#, put("0"), put("1"), put("2"), par("w", 3, take))), "of one `parallel` both take on this channel: a channel has one taker at a time"),
    ];
    // taken in turn: the main line, then one sub-actor of each of two forks per iteration, then
    // a sub-actor of a fork inside a sub-actor and that sub-actor after it, then the main line
    let seq2 = |odd: i64| format!(r#"{{"add":[{{"mul":[{{"index":"e"}},2]}},{odd}]}}"#);
    let good = doc(&format!(
        r#"{{"channel":{{"name":"c","capacity":2}}}}, {}, {take},
        {{"loop":{{"index":"e","to":3,"body":[{}, {}]}}}}, {}, {}, {take}"#,
        put("0"),
        par("w", 2, &when("w", 0, take, &put(&seq2(1)))),
        par("v", 2, &when("v", 1, take, &put(&seq2(2)))),
        par("a", 2, &when("a", 0, &format!("{}, {take}", par("b", 2, &when("b", 1, take, ""))), &format!("{}, {}", put("7"), put("8")))),
        put("9"),
    ));
    let root = tmpdir("r16-takers");
    let good = model_of(&good);
    for backend in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio] {
        for scale in [0.0, 0.0, 0.0, 0.001] {
            let o = || {
                let mut o = opts(&root, backend);
                o.time_scale = scale;
                o
            };
            for (d, says) in &refused {
                let e = run_within(model_of(d), o()).err().expect("refused");
                assert!(e.contains(says) && e.contains("take `c`"), "{backend:?}, time scale {scale}: {e}\n{d}");
            }
            let r = run_within(good, o()).unwrap_or_else(|e| panic!("{backend:?}, time scale {scale}: {e}"));
            assert_eq!((r.stats.puts, r.stats.takes), (10, 10), "{backend:?}, time scale {scale}");
        }
    }
}

/// An unordered channel whose items skip a seq (the RC1 review's R17: `put c seq s+1; take c`
/// in a loop, so 0 is never put) took the run's items past the gap and kept every seq above it,
/// a record that grew without bound (80 MB at 2,000,000 items against 40 MB without the gap)
/// for a gap silently taken. A put's window starts at the lowest seq not yet put when that is
/// below the items taken (2026-10-15, `Seqs::window`), on an unordered channel as on an
/// ordered one, so the record stays within the capacity and the gap is a wait nothing ends,
/// whose error says which seq it waits for, under both engines.
#[test]
fn an_unordered_channels_missing_seq_is_a_wait_nothing_ends() {
    let d = r#"{"ast":"0.13","name":"r17","actors":{"gpu":{"body":[{"channel":{"name":"c","capacity":2,"ordered":false}},
        {"loop":{"index":"s","to":1000,"body":[{"put":{"channel":"c","seq":{"add":[{"index":"s"},1]}}},{"take":{"channel":"c"}}]}}]}}}"#;
    let model = model_of(d);
    let root = tmpdir("r17-gap");
    for backend in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio] {
        let e = run_within(model, opts(&root, backend)).err().expect("refused");
        let says = match backend {
            BackendKind::Sync => "every thread of gpu#0 waits on a channel that nothing will change (this one waits to put 2 (seq 0 is not put yet))",
            _ => "1 actor(s) wait on channels or trace order nothing will complete: gpu#0 [1] on `c` waits to put 2 (seq 0 is not put yet)",
        };
        assert!(e.contains(says), "{backend:?}: {e}");
    }
}
