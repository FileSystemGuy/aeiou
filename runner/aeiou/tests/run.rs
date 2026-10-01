//! `aeiou datagen` and `aeiou run` against a temporary directory on the local filesystem:
//! the corpus a dataset definition describes is written and read back exactly (every read
//! returns the computed count), the run's fingerprint equals the dry run's, the manifest
//! check refuses a changed definition, namespaces must be empty, and a loader delivers its
//! batches in order under real concurrency.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use aeiou::backend::BackendKind;
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};

static N: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aeiou-test-{}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples")
}

fn config(gpus: i64, seed: u64, params: &[(&str, &str)]) -> Config {
    Config { seed, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![] }
}

/// A run is the process: the model is leaked so the actor threads can borrow it.
fn leaked_model(name: &str, cfg: Config) -> (&'static aeiou::Loaded, &'static Config, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&examples().join(format!("{name}.ast.json"))).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    (loaded, cfg, model)
}

fn gen(loaded: &aeiou::Loaded, cfg: &Config, model: &Model<'_>, root: &PathBuf) {
    let params = Params::new(&loaded.ast, cfg).unwrap();
    let opts = DatagenOpts { root: root.clone(), threads: 4, dedupe: 1, compress: 1, datasets: vec![] };
    let mut log = Vec::new();
    datagen(loaded, cfg, &params, model, &opts, &mut log).unwrap();
}

fn opts(root: &PathBuf, backend: BackendKind) -> RunOpts {
    RunOpts {
        root: root.clone(),
        backend,
        buffer_bytes: 1 << 20,
        write_compress: 1,
        time_scale: 0.0,
        clean_namespaces: false,
        expect_fingerprint: None,
        expect_dataset_ids: vec![],
        rank: 0,
        ranks: 1,
        rank_rotate: 0,
        max_gap: None,
        require_cold: false,
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
    let params = [("files", "600"), ("batch", "4"), ("workers", "2"), ("prefetch", "2"), ("steps", "12"), ("enumerate", "true")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(2, 7, &params));
    gen(loaded, cfg, model, &root);
    let checks = run::check_datasets(loaded, cfg, &root).unwrap();
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].files, Some(600));
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    for backend in [BackendKind::Sync, BackendKind::SyncDirect] {
        let r = go(model, opts(&root, backend));
        assert_eq!(r.stats.fingerprint, fp, "{:?}: fingerprint", backend);
        assert_eq!(r.stats.ops, ops);
        assert_eq!(r.stats.bytes_read, bytes, "every read returned the computed count");
        assert_eq!(r.stats.takes, 24);
        assert_eq!(r.stats.barriers, 2, "`every(sync_every)` fires at step 0 on each of the two instances");
        assert_eq!(r.stats.expected_errors, 2 * 12 * 4, "ENOTTY on every ioctl");
        assert_eq!(r.actors.len(), 2);
        assert!(r.actors.iter().all(|a| a.takes.len() == 12));
        assert!(r.departure_releases.is_empty());
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn manifest_mismatch_and_missing_are_refused() {
    let root = tmpdir("manifest");
    let (loaded, cfg, model) = leaked_model("train_small_files", config(1, 1, &[("files", "50")]));
    // nothing generated yet
    assert!(run::check_datasets(loaded, cfg, &root).unwrap_err().to_string().contains("datagen"));
    gen(loaded, cfg, model, &root);
    run::check_datasets(loaded, cfg, &root).unwrap();
    // a different count resolves to a different definition
    let (loaded2, cfg2, _) = leaked_model("train_small_files", config(1, 1, &[("files", "51")]));
    let e = run::check_datasets(loaded2, cfg2, &root).unwrap_err();
    assert!(e.to_string().contains("files.count: 51 vs 50"), "{e:#}");
    // datagen refuses a non-empty root
    let (loaded3, cfg3, model3) = leaked_model("train_small_files", config(1, 1, &[("files", "50")]));
    let params = Params::new(&loaded3.ast, cfg3).unwrap();
    let o = DatagenOpts { root: root.clone(), threads: 1, dedupe: 1, compress: 1, datasets: vec![] };
    let mut log = Vec::new();
    assert!(datagen(loaded3, cfg3, &params, model3, &o, &mut log).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn namespaces_must_be_empty_and_writes_are_read_back() {
    let root = tmpdir("ckpt");
    let params = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 65536]"), ("meta_bytes", "65536"), ("readback", "true")];
    let (loaded, _cfg, model) = leaked_model("ckpt_write_dcp", config(2, 3, &params));
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
    assert_eq!(std::fs::metadata(&shard).unwrap().len(), 1048576 + 2097152 + 1048576 + 65536 + 4 * 65536);
    assert!(root.join("ckpt/step_000002/.metadata").exists());
    // a second run refuses the stale namespace, and --clean-namespaces empties it
    let e = run::prepare_namespaces(&loaded.ast, &root, false).unwrap_err();
    assert!(e.to_string().contains("not empty"), "{e:#}");
    let cleaned = run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
    assert_eq!(cleaned, vec!["ckpt".to_string()]);
    assert!(!shard.exists());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn kv_cache_chunked_dataset_parallel_slots_and_namespace() {
    let root = tmpdir("kv");
    let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8")];
    let (loaded, cfg, model) = leaked_model("kv_cache_serving", config(1, 5, &params));
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
    assert_eq!(r.stats.threads, 1 + 3, "one actor thread plus three slots");
    assert!(r.stats.expected_errors > 0, "ENOENT lookups and EEXIST mkdirs");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn regions_dataset_nested_parallel_and_direct_reads() {
    let root = tmpdir("diskann");
    let params = [("nodes", "5000"), ("threads", "2"), ("queries", "6")];
    let (loaded, cfg, model) = leaked_model("vdb_search_diskann", config(1, 9, &params));
    gen(loaded, cfg, model, &root);
    assert_eq!(std::fs::metadata(root.join("diskann/index.bin")).unwrap().len(), 5000 * 4096);
    run::check_datasets(loaded, cfg, &root).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    let r = go(model, opts(&root, BackendKind::Sync));
    assert_eq!(r.stats.fingerprint, fp);
    assert_eq!(r.stats.ops, ops);
    assert_eq!(r.stats.bytes_read, bytes);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn loader_delivers_in_order_and_bounds_prefetch() {
    // 1 worker, prefetch 1: batch b+1 cannot start before batch b is taken, so with a
    // non-zero step time the stall of every take after the first is about one batch's I/O.
    let root = tmpdir("loader");
    let params = [("files", "200"), ("batch", "2"), ("workers", "3"), ("prefetch", "1"), ("steps", "30"), ("step_time", "2000000")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(1, 11, &params));
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
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn write_then_restore_through_the_namespace_manifest() {
    // ckpt_write_dcp leaves `.aeiou-namespace.json` at ckpt/; ckpt_restore declares the
    // namespaces `input`, reads what was written, measures the gap, and counts the warm opens
    // (all of them on one host).
    let root = tmpdir("restore");
    let wparams = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 1048576]"), ("meta_bytes", "65536")];
    let (wl, wcfg, wmodel) = leaked_model("ckpt_write_dcp", config(2, 3, &wparams));
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
    assert_eq!(m.abstract_name, "ckpt_write_dcp");
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
        ("item_off", "[0, 1114112, 3276800, 4390912]"),   // Σ (item + 64 KiB tail) of the writer
        ("meta_bytes", "65536"),
    ];
    let (rl, rcfg, rmodel) = leaked_model("ckpt_restore", config(2, 5, &rparams));
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
    std::fs::remove_dir_all(&root).unwrap();
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
