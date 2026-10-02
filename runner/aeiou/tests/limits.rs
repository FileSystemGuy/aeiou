//! The limit checks before the gate (`limits.rs`, `runner/README.md` §11). A file of its own,
//! so a process of its own: the open-file peak is counted per process, and the checks raise
//! this process's soft limits.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use aeiou::backend::BackendKind;
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::limits::{self, Limits, Need};
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

/// Small-file training: a loader's workers each hold one file at a time, beside the trainer;
/// the run's counted peak is what the estimate said, and it runs on as many threads.
#[test]
fn the_estimate_matches_the_counted_peak() {
    let root = tmpdir("limits");
    let params = [("files", "300"), ("batch", "4"), ("workers", "3"), ("prefetch", "2"), ("steps", "10")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(3, 11, &params));
    gen(loaded, cfg, model, &root);
    run::check_datasets(loaded, cfg, &root).unwrap();
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    let o = opts(&root, BackendKind::Sync);
    let need = limits::estimate(model, &o).unwrap();
    assert_eq!(need, Need { open_files: 3 * 3, contexts: 3 * (1 + 3), truncated: false });
    let l = limits::check(model, &o, 0, false).unwrap();
    assert!(l.fds_needed > need.open_files && l.nofile_soft == l.nofile_hard, "{l:?}");
    assert_eq!(l.lines().len(), 2);

    let r = run::run(model, o, Default::default()).unwrap();
    assert!(r.counters.open_files_peak >= 1 && r.counters.open_files_peak <= need.open_files, "peak {} of {}", r.counters.open_files_peak, need.open_files);
    // the same under an event loop, where the threads are the loops
    let mut o = opts(&root, BackendKind::Uring);
    o.threads = 2;
    assert_eq!(run::loop_count(model, &o).unwrap(), 2);
    let r = run::run(model, o, Default::default()).unwrap();
    assert!(r.counters.open_files_peak >= 1 && r.counters.open_files_peak <= need.open_files, "io_uring peak {}", r.counters.open_files_peak);
    let _ = std::fs::remove_dir_all(&root);
}

/// DiskANN search at its default size: the budget ends inside the first search thread, the
/// estimate says so, and the fork structure still gives one open file per thread and a
/// context per beam slot (2 instances × 32 threads × beam 4).
#[test]
fn a_truncated_walk_scales_by_the_fork_width() {
    let (_, _, model) = leaked_model("vdb_search_diskann", config(2, 1, &[]));
    let need = limits::estimate(model, &opts(&PathBuf::from("/nonexistent"), BackendKind::Sync)).unwrap();
    assert_eq!(need, Need { open_files: 2 * 32, contexts: 2 * 32 * 4, truncated: true });
    // host 1 of 2 runs one of the two instances
    let mut o = opts(&PathBuf::from("/nonexistent"), BackendKind::Sync);
    (o.ranks, o.rank) = (2, 1);
    assert_eq!(limits::estimate(model, &o).unwrap().open_files, 32);
}

#[test]
fn what_does_not_fit_names_its_limit() {
    let fits = Limits { fds_needed: 100, nofile_soft: 1024, nofile_hard: 1024, threads_needed: 10, nproc_soft: u64::MAX, nproc_hard: u64::MAX, threads_max: Some(1000), maps_needed: 50, max_map_count: Some(65530), ..Default::default() };
    assert!(fits.problems().is_empty(), "{:?}", fits.problems());
    let l = Limits { fds_needed: 16_100, threads_needed: 16_010, maps_needed: 70_000, ..fits.clone() };
    let p = l.problems();
    assert!(p.iter().any(|s| s.contains("RLIMIT_NOFILE is 1024")), "{p:?}");
    assert!(p.iter().any(|s| s.contains("kernel.threads-max is 1000")), "{p:?}");
    assert!(p.iter().any(|s| s.contains("vm.max_map_count is 65530")), "{p:?}");
}
