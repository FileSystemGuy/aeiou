//! The `io_uring` ring and io-wq knobs (`RunOpts::uring`, `NAPKIN_MATH.md` §8.5) against a
//! temporary directory. A file of its own, so a process of its own: the assertions count
//! this process's `iou-wrk-*` and `iou-sqp-*` threads, which the other `io_uring` tests
//! would add to if they ran beside this one.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use aeiou::backend::BackendKind;
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts, UringOpts};

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

fn dry_fingerprint(model: &Model<'_>) -> (u64, u64, u64) {
    let r = dryrun::run(model, 2, None).unwrap();
    (r.total.fingerprint, r.total.total.ops, r.total.total.bytes_read)
}

/// The `iou-wrk-*` and `iou-sqp-*` threads of this process.
fn io_threads_left() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| std::fs::read_to_string(e.path().join("comm")).map(|c| c.starts_with("iou-")).unwrap_or(false)).count())
        .unwrap_or(0)
}

#[test]
fn io_uring_knobs_keep_the_fingerprint() {
    // Every ring knob (`NAPKIN_MATH.md` §8.5) runs the same op stream: the fingerprint, op
    // count, and bytes equal the dry run's. The io-wq cap is per loop's io-wq (one io-wq for
    // all loops under a shared SQPOLL thread), the kernel's own caps come back in the
    // report, and the counters see the SQPOLL threads. A kernel that refuses a setup flag
    // (SQPOLL needs 5.13 unprivileged, DEFER_TASKRUN 6.1) skips that row.
    let root = tmpdir("uring-knobs");
    let params = [("files", "300"), ("batch", "4"), ("workers", "3"), ("prefetch", "2"), ("steps", "10"), ("enumerate", "true")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(3, 11, &params));
    gen(loaded, cfg, model, &root);
    run::check_datasets(loaded, cfg, &root).unwrap();
    run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
    let (fp, ops, bytes) = dry_fingerprint(model);
    let loops = 3u64;
    let rows: Vec<(&str, UringOpts)> = vec![
        ("iowq 1", UringOpts { iowq_max_workers: 1, ..Default::default() }),
        ("defer+coop", UringOpts { defer_taskrun: true, coop_taskrun: true, ..Default::default() }),
        ("coop+iowq 2", UringOpts { coop_taskrun: true, iowq_max_workers: 2, ..Default::default() }),
        ("sqpoll per loop", UringOpts { sqpoll_idle_ms: Some(100), ..Default::default() }),
        ("sqpoll shared, iowq 1", UringOpts { sqpoll_idle_ms: Some(100), sqpoll_shared: true, iowq_max_workers: 1, ..Default::default() }),
    ];
    for (name, k) in rows {
        // a closed ring's SQPOLL thread and io-wq workers exit asynchronously; the rows share
        // this process, so wait for the previous row's to go before the sampler starts
        let t0 = std::time::Instant::now();
        while io_threads_left() > 0 {
            assert!(t0.elapsed() < std::time::Duration::from_secs(10), "io threads of the previous ring still alive");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let mut o = opts(&root, BackendKind::Uring);
        o.threads = loops as usize;
        o.uring = k.clone();
        let r = match run::run(model, o, std::collections::HashMap::new()) {
            Ok(r) => r,
            Err(e) if format!("{e:#}").contains("io_uring_setup") => {
                eprintln!("skipping `{name}`: {e:#}");
                continue;
            }
            Err(e) => panic!("`{name}`: {e:#}"),
        };
        assert_eq!(r.stats.fingerprint, fp, "`{name}`: fingerprint");
        assert_eq!(r.stats.ops, ops, "`{name}`");
        assert_eq!(r.stats.bytes_read, bytes, "`{name}`");
        let u = r.uring.as_ref().expect("the io_uring backends report their rings");
        assert_eq!(u.loops, loops, "`{name}`");
        assert_eq!(u.opts, k, "`{name}`");
        let c = &r.counters;
        if let Some([bounded, unbounded]) = u.iowq_defaults {
            assert!(bounded > 0 && unbounded > 0, "`{name}`: the kernel's caps before ours: {u:?}");
            let per = k.iowq_max_workers as u64;
            if per > 0 {
                let cap = if k.sqpoll_shared { per } else { per * loops };
                assert!(c.iowq_workers_peak <= cap, "`{name}`: {} io-wq workers over the cap {cap}", c.iowq_workers_peak);
            }
        } else {
            assert_eq!(k.iowq_max_workers, 0, "`{name}`: a cap without the register call cannot succeed");
        }
        match (k.sqpoll_idle_ms, k.sqpoll_shared) {
            // what each ring's fdinfo states once built, not a sample: exact however short the row is
            (Some(_), true) => assert_eq!(c.sqpoll_threads, 1, "`{name}`: {c:?}"),
            (Some(_), false) => assert_eq!(c.sqpoll_threads, loops, "`{name}`: {c:?}"),
            (None, _) => assert_eq!(c.sqpoll_threads, 0, "`{name}`: {c:?}"),
        }
        // 3 instances × (1 main line + 3 loader workers), one op in flight each at most,
        // spread over the loops; a loader worker reading is at least one
        assert!((1..=12).contains(&u.in_flight_peak), "`{name}`: {u:?}");
    }
    // the knobs belong to the rings: the `sync` backends refuse them, and the kernel's
    // exclusions are said before a ring is built
    let mut o = opts(&root, BackendKind::Sync);
    o.uring.coop_taskrun = true;
    let e = run::run(model, o, std::collections::HashMap::new()).unwrap_err().to_string();
    assert!(e.contains("no ring"), "{e}");
    let mut o = opts(&root, BackendKind::Uring);
    o.uring = UringOpts { sqpoll_idle_ms: Some(100), defer_taskrun: true, ..Default::default() };
    let e = format!("{:#}", run::run(model, o, std::collections::HashMap::new()).unwrap_err());
    assert!(e.contains("exclude each other"), "{e}");
    let mut o = opts(&root, BackendKind::Uring);
    o.uring = UringOpts { sqpoll_shared: true, ..Default::default() };
    let e = format!("{:#}", run::run(model, o, std::collections::HashMap::new()).unwrap_err());
    assert!(e.contains("needs sqpoll"), "{e}");
    std::fs::remove_dir_all(&root).unwrap();
}
