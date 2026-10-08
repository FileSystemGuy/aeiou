//! `aeiou datagen` and `aeiou run` against a temporary directory on the local filesystem:
//! the corpus a dataset definition describes is written and read back exactly (every read
//! returns the computed count), the run's fingerprint equals the dry run's, the manifest
//! check refuses a changed definition, namespaces must be empty, and a loader delivers its
//! batches in order under real concurrency. The `io_uring` backends run the same abstracts
//! on the event loop and must produce the same fingerprint, counts, and bytes, and so must
//! `posix-aio`, `libaio`, and `mmap`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use aeiou::backend::{BackendKind, MmapConsume, MmapMode};
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
    let opts = DatagenOpts { root: root.clone(), threads: 4, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1 };
    let mut log = Vec::new();
    datagen(loaded, cfg, &params, model, &opts, None, &mut log).unwrap();
}

fn opts(root: &PathBuf, backend: BackendKind) -> RunOpts {
    RunOpts {
        root: root.clone(),
        backend,
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
    let params = [("files", "600"), ("batch", "4"), ("workers", "2"), ("prefetch", "2"), ("steps", "12"), ("enumerate", "true")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(2, 7, &params));
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
    let o = DatagenOpts { root: root.clone(), threads: 1, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1 };
    let mut log = Vec::new();
    assert!(datagen(loaded3, cfg3, &params, model3, &o, None, &mut log).is_err());
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
    assert_eq!(std::fs::metadata(&shard).unwrap().len(), 1048576 + 2097152 + 1048576 + 65536 + 4 * (704 + 873));
    assert!(root.join("ckpt/step_000002/.metadata").exists());
    // a second run refuses the stale namespace, and --clean-namespaces empties it
    let e = run::prepare_namespaces(&loaded.ast, &root, false).unwrap_err();
    assert!(e.to_string().contains("not empty"), "{e:#}");
    let cleaned = run::prepare_namespaces(&loaded.ast, &root, true).unwrap();
    assert_eq!(cleaned, vec!["ckpt".to_string()]);
    assert!(!shard.exists());
    std::fs::remove_dir_all(&root).unwrap();
}

/// The KV-cache abstracts in a few requests: a slot with two conversations open (the default's forty would start
/// nothing but new ones), and an engine that keeps none, half, or all of a returning one.
const KV_REUSE: &str = r#"{"mixture": [{"weight": 0.3, "dist": null}, {"weight": 0.7, "dist": {"const": 2}}]}"#;
const KV_KEEP: &str = r#"{"empirical": {"values": [0, 50, 100], "weights": [1, 1, 1]}}"#;

#[test]
fn kv_cache_chunked_dataset_parallel_slots_and_namespace() {
    let root = tmpdir("kv");
    let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8"), ("reuse", KV_REUSE), ("keep", KV_KEEP)];
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
    assert!(r.stats.threads >= 1 + 2, "the actor's thread runs slot 0 itself: two pool threads for three slots, and one more under any slot that loads two chunks at once: {}", r.stats.threads);
    assert!(r.stats.expected_errors > 0, "the ENOTTY of the terminal probe in every open (ENOENT lookups and EEXIST mkdirs until the 2026-10-02 trace)");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn kv_shared_store_filled_by_one_engine_and_read_by_a_cold_one() {
    // kv_cache_shared and kv_cache_shared_reader come from one script and place their draws at the same sites:
    // with the writer's seed, instance count and parameters the reader opens exactly the chunks the writer
    // renamed into place (two instances, eviction with re-store, system prompts loaded).
    let root = tmpdir("kvshared");
    let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("buf", "65536"), ("concurrency", "3"), ("warm", "4"),
                  ("requests", "12"), ("reuse", KV_REUSE), ("keep", KV_KEEP), ("retain", "6"), ("sys_local", "false")];
    let (wl, wcfg, wmodel) = leaked_model("kv_cache_shared", config(2, 5, &params));
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

    let (rl, rcfg, rmodel) = leaked_model("kv_cache_shared_reader", config(2, 5, &params));
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
        let (l, c, _) = leaked_model("kv_cache_shared_reader", config(gpus, seed, p));
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
    let (_, _, late) = leaked_model("kv_cache_shared_reader", config(2, 6, &params));
    let (_, inputs) = run::check_input_namespaces(rl, rcfg, &root, &opts(&root, BackendKind::Sync)).unwrap();
    assert!(run::run(late, opts(&root, BackendKind::Sync), inputs).is_err());
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
    // the sub-actor pool: a forking thread runs sub-actor 0 itself and keeps a thread for
    // each of the others, so the instance's thread and one more are the two search threads,
    // and each of those has `beam - 1` threads kept across every hop of every query
    let beam: u64 = 4;
    assert!(ops > 2 * 6 * 2 * beam, "several hops per query: {ops} ops");
    assert_eq!(r.stats.threads, 1 + (2 - 1) + 2 * (beam - 1));
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
        ("item_off", "[0, 1050153, 3148882, 4199035]"),   // Σ (704 + item + 873) of the writer
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

#[test]
fn io_uring_reproduces_the_sync_runs() {
    // The event loop multiplexes every actor and sub-actor of these abstracts (loaders with
    // ordered channels, `parallel` slots, nested `parallel`, barriers, namespace writes and
    // read-back) over one ring per thread; the fingerprint, op count, and bytes must equal the
    // dry run's exactly, on one loop and on several.
    for threads in [1usize, 3] {
        let root = tmpdir("uring-kv");
        let params = [("sys_prompts", "3"), ("sys_tokens", "6"), ("chunk_bytes", "262144"), ("concurrency", "3"), ("warm", "4"), ("requests", "8"), ("reuse", KV_REUSE), ("keep", KV_KEEP)];
        let (loaded, cfg, model) = leaked_model("kv_cache_serving", config(2, 5, &params));
        gen(loaded, cfg, model, &root);
        run::check_datasets(loaded, cfg, &root).unwrap();
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        let (fp, ops, bytes) = dry_fingerprint(model);
        let mut o = opts(&root, BackendKind::Uring);
        o.threads = threads;
        let r = go(model, o);
        assert_eq!(r.stats.fingerprint, fp, "kv_cache_serving on {threads} loop(s)");
        assert_eq!(r.stats.ops, ops);
        assert_eq!(r.stats.bytes_read, bytes);
        assert_eq!(r.stats.threads as usize, threads.min(2), "loop threads, not actors");
        assert!(r.stats.expected_errors > 0);
        // the host counters: the loops and the sampler are tasks, and the root is on some mount.
        // Not asserted: an io-wq worker. An `openat` whose lookup is cached completes inline, so
        // whether a short run punts at all is the kernel's; under `cargo test` the idle workers
        // of the binary's earlier tests were what this saw (2026-10-07, CI under nextest)
        let c = &r.counters;
        assert!(c.tasks_peak as usize >= threads.min(2) + 2, "{c:?}");
        assert!(c.cpu_user_ns + c.cpu_sys_ns > 0, "{c:?}");
        assert!(c.mount.as_ref().map(|m| !m.fstype.is_empty()).unwrap_or(false), "{c:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }
    {
        let root = tmpdir("uring-diskann");
        let params = [("nodes", "5000"), ("threads", "2"), ("queries", "6")];
        let (loaded, cfg, model) = leaked_model("vdb_search_diskann", config(2, 9, &params));
        gen(loaded, cfg, model, &root);
        let (fp, ops, bytes) = dry_fingerprint(model);
        for backend in [BackendKind::Uring, BackendKind::UringDirect] {
            let r = go(model, opts(&root, backend));
            assert_eq!(r.stats.fingerprint, fp, "{backend:?}");
            assert_eq!(r.stats.ops, ops);
            assert_eq!(r.stats.bytes_read, bytes);
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
    {
        let root = tmpdir("uring-ckpt");
        let params = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 65536]"), ("meta_bytes", "65536"), ("readback", "true"),
            ("hdr", "4096"), ("trailer", "4096")];   // the traced 704 and 873 are not aligned, and a direct backend refuses such writes
        let (loaded, _cfg, model) = leaked_model("ckpt_write_dcp", config(2, 3, &params));
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
        std::fs::remove_dir_all(&root).unwrap();
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
        let (loaded, cfg, model) = leaked_model("kv_cache_serving", config(2, 5, &params));
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
        std::fs::remove_dir_all(&root).unwrap();
    }
    {
        let root = tmpdir("more-diskann");
        let params = [("nodes", "5000"), ("threads", "2"), ("queries", "6")];
        let (loaded, cfg, model) = leaked_model("vdb_search_diskann", config(2, 9, &params));
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
                // file once and the index twice, and its search threads and their beam
                // sub-actors read the index through the second mapping
                assert_eq!((m.maps, m.mapped_bytes), (6, 2 * (5000 * 5 * 32 + 8 + 2 * 5000 * 4096)), "{m:?}");
                let reads = r.stats.counts[&aeiou::vm::OpKind::Read];
                assert_eq!(m.advised, if mode == MmapMode::Fault { 0 } else { reads }, "{m:?}");
                let (touched, copied) = match (consume, mode) {
                    (MmapConsume::Copy, _) => (0, bytes),
                    (MmapConsume::Touch, MmapMode::Populate) => (0, 0),
                    (MmapConsume::Touch, _) => ((bytes - 2 * (5000 * 5 * 32 + 8 + 8191)) / 4096 + 2 * (2 + 195 + 2), 0),
                };
                assert_eq!((m.touched_pages, m.copied_bytes), (touched, copied), "{m:?}");
                assert!(r.counters.minor_faults + r.counters.major_faults > 0, "{:?}", r.counters);
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
    {
        let root = tmpdir("more-ckpt");
        let params = [("steps", "4"), ("ckpt_every", "2"), ("item_bytes", "[1048576, 2097152, 1048576, 65536]"), ("meta_bytes", "65536"), ("readback", "true"),
            ("hdr", "4096"), ("trailer", "4096")];   // the traced 704 and 873 are not aligned, and a direct backend refuses such writes
        let (loaded, _cfg, model) = leaked_model("ckpt_write_dcp", config(2, 3, &params));
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
        std::fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn io_uring_loader_delivers_in_order_and_bounds_prefetch() {
    let root = tmpdir("uring-loader");
    let params = [("files", "200"), ("batch", "2"), ("workers", "3"), ("prefetch", "1"), ("steps", "30"), ("step_time", "2000000")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(1, 11, &params));
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
    std::fs::remove_dir_all(&root).unwrap();
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
    let (loaded, cfg, model) = leaked_model("train_small_files", config(1, 3, &[("files", "300")]));
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
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn dedupe_groups_consecutive_files_and_does_not_depend_on_the_count() {
    // aeiou-positional/2: `dedupe` consecutive files share a unit, so a prefix of the ids has
    // the full ratio and a dataset can grow; file `id`'s bytes are the same under any count
    let six = tmpdir("dd6");
    let four = tmpdir("dd4");
    let (l6, c6, m6) = leaked_model("train_small_files", config(1, 1, &[("files", "6")]));
    let (l4, c4, m4) = leaked_model("train_small_files", config(1, 1, &[("files", "4")]));
    for (l, c, m, root) in [(l6, c6, m6, &six), (l4, c4, m4, &four)] {
        let params = Params::new(&l.ast, c).unwrap();
        let opts = DatagenOpts { root: root.clone(), threads: 2, dedupe: 2, compress: 1, datasets: vec![], rank: 0, ranks: 1 };
        datagen(l, c, &params, m, &opts, None, &mut Vec::new()).unwrap();
    }
    let file = |root: &PathBuf, id: i64| std::fs::read(root.join(m6.datasets[0].file_path(id, None).unwrap().as_ref())).unwrap();
    let prefix_equal = |a: &[u8], b: &[u8]| {
        let n = a.len().min(b.len());
        a[..n] == b[..n]
    };
    assert!(prefix_equal(&file(&six, 0), &file(&six, 1)), "files 0 and 1 share a unit");
    assert!(prefix_equal(&file(&six, 2), &file(&six, 3)));
    assert!(prefix_equal(&file(&six, 4), &file(&six, 5)));
    assert!(!prefix_equal(&file(&six, 1), &file(&six, 2)), "files 1 and 2 do not");
    for id in 0..4 {
        assert_eq!(file(&six, id), file(&four, id), "file {id} under count 6 and count 4");
    }
    let m = aeiou::payload::Manifest::read(&six.join("train")).unwrap();
    assert_eq!(m.payload.wrapper, "aeiou-positional/2");
    assert_eq!(m.payload.dedupe, 2);
    std::fs::remove_dir_all(&six).unwrap();
    std::fs::remove_dir_all(&four).unwrap();
}
