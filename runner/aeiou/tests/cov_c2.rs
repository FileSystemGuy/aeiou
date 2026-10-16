//! The I/O backends against one another (RC1 review, coverage lane C2): every POSIX op of the
//! abstract, issued through every backend, gives the dry run's fingerprint and leaves the same
//! files behind; and an op's arguments reach the kernel whole through every API.

use aeiou::backend::BackendKind;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};

const EVERY: [BackendKind; 9] = [
    BackendKind::Sync,
    BackendKind::SyncDirect,
    BackendKind::Uring,
    BackendKind::UringDirect,
    BackendKind::PosixAio,
    BackendKind::PosixAioDirect,
    BackendKind::LibAio,
    BackendKind::LibAioDirect,
    BackendKind::Mmap,
];

fn leaked(text: &str) -> (&'static aeiou::Loaded, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(text).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![] }));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    (loaded, Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap())))
}

fn opts(root: &std::path::Path, backend: BackendKind) -> RunOpts {
    RunOpts {
        root: root.to_path_buf(),
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

/// Every op kind but `ioctl` on a namespace's files and directories, twice.
const ALL_OPS: &str = r#"{
  "ast": "0.10", "name": "allops",
  "namespaces": {
    "o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": "as_written", "seed": 3},
    "dd": {"pattern": "dd/{v}", "fields": {"v": "int"}, "size": "as_written", "seed": 3}
  },
  "actors": {"gpu": {"body": [
    {"loop": {"index": "i", "to": 2, "body": [
      {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"v": {"index": "i"}}}}}},
      {"let": {"name": "g", "value": {"object": {"namespace": "o", "fields": {"v": {"add": [{"index": "i"}, 10]}}}}}},
      {"let": {"name": "d", "value": {"object": {"namespace": "dd", "fields": {"v": {"index": "i"}}}}}},
      {"open": {"file": {"ref": "f"}, "flags": ["RDWR", "CREAT"]}},
      {"write": {"file": {"ref": "f"}, "len": 65536, "offset": 0}},
      {"fsync": {"file": {"ref": "f"}}},
      {"fdatasync": {"file": {"ref": "f"}}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "NORMAL"}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "RANDOM", "offset": 0, "len": 4096}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "NOREUSE", "offset": 4096, "len": 4096}},
      {"fallocate": {"file": {"ref": "f"}, "offset": 65536, "len": 8192}},
      {"fstat": {"file": {"ref": "f"}}},
      {"read": {"file": {"ref": "f"}, "len": 8192, "offset": 0}},
      {"ftruncate": {"file": {"ref": "f"}, "len": 65536}},
      {"fstat": {"file": {"ref": "f"}}},
      {"read": {"file": {"ref": "f"}, "len": 4096, "offset": 4096}},
      {"lseek": {"file": {"ref": "f"}, "offset": 0, "whence": "END"}},
      {"close": {"file": {"ref": "f"}}},
      {"stat": {"file": {"ref": "f"}}},
      {"rename": {"from": {"ref": "f"}, "to": {"ref": "g"}}},
      {"open": {"file": {"ref": "g"}, "flags": ["RDONLY"]}},
      {"read": {"file": {"ref": "g"}, "len": 65536, "offset": 0}},
      {"close": {"file": {"ref": "g"}}},
      {"mkdir": {"dir": {"ref": "d"}, "mode": 493}},
      {"open": {"file": {"ref": "d"}, "flags": ["RDONLY", "DIRECTORY"]}},
      {"readdir": {"dir": {"ref": "d"}, "repeat": "until_end"}},
      {"close": {"file": {"ref": "d"}}},
      {"rmdir": {"dir": {"ref": "d"}}},
      {"unlink": {"file": {"ref": "g"}}}
    ]}}
  ]}}
}"#;

#[test]
fn every_backend_issues_every_posix_op_alike() {
    let (loaded, model) = leaked(ALL_OPS);
    let dry = aeiou::dryrun::run(model, 1, None).unwrap();
    for backend in EVERY {
        let root = crate::common::tmpdir("cov-c2", "allops");
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        let r = run::run(model, opts(&root, backend), Default::default()).unwrap_or_else(|e| panic!("{backend:?}: {e:#}"));
        assert_eq!((r.stats.fingerprint, r.stats.ops), (dry.total.fingerprint, dry.total.total.ops), "{backend:?}");
        assert_eq!(r.stats.bytes_read, 2 * (8192 + 4096 + 65536), "{backend:?}: every read returned its count");
        assert_eq!(r.stats.bytes_written, 2 * 65536, "{backend:?}");
        // what the ops left: the renamed objects were unlinked, the directories removed
        for ns in ["o", "dd"] {
            let left: Vec<_> = std::fs::read_dir(root.join(ns)).unwrap().map(|e| e.unwrap().file_name()).collect();
            assert!(left.is_empty(), "{backend:?}: {ns}/ holds {left:?}");
        }
    }
}

/// The descriptor ops of an open that failed (its errno expected): no descriptor, `EBADF`.
const BAD_FD: &str = r#"{
  "ast": "0.10", "name": "badfd",
  "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": 4096, "seed": 3}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"v": 0}}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDWR"], "expect": ["ENOENT"]}},
    {"read": {"file": {"ref": "f"}, "len": 4096, "offset": 0, "expect": ["EBADF"]}},
    {"write": {"file": {"ref": "f"}, "len": 4096, "offset": 0, "expect": ["EBADF"]}},
    {"fsync": {"file": {"ref": "f"}, "expect": ["EBADF"]}},
    {"fdatasync": {"file": {"ref": "f"}, "expect": ["EBADF"]}},
    {"fadvise": {"file": {"ref": "f"}, "advice": "WILLNEED", "expect": ["EBADF"]}},
    {"fstat": {"file": {"ref": "f"}, "expect": ["EBADF"]}},
    {"ftruncate": {"file": {"ref": "f"}, "len": 0, "expect": ["EBADF"]}},
    {"fallocate": {"file": {"ref": "f"}, "len": 4096, "expect": ["EBADF"]}}
  ]}}
}"#;

#[test]
fn an_op_on_an_open_that_failed_is_ebadf_on_every_backend() {
    let (loaded, model) = leaked(BAD_FD);
    let dry = aeiou::dryrun::run(model, 1, None).unwrap();
    for backend in EVERY {
        let root = crate::common::tmpdir("cov-c2", "badfd");
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        let r = run::run(model, opts(&root, backend), Default::default()).unwrap_or_else(|e| panic!("{backend:?}: {e:#}"));
        assert_eq!((r.stats.fingerprint, r.stats.expected_errors), (dry.total.fingerprint, 9), "{backend:?}");
    }
}

/// A read through a write-only open and a write through a read-only one: the kernel refuses.
const WRONG_MODE: &str = r#"{
  "ast": "0.10", "name": "wrongmode",
  "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": 4096, "seed": 3}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"v": 0}}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT"]}},
    {"write": {"file": {"ref": "f"}, "len": 4096, "offset": 0}},
    {"read": {"file": {"ref": "f"}, "len": 4096, "offset": 0, "expect": ["EBADF", "EACCES"]}},
    {"close": {"file": {"ref": "f"}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
    {"write": {"file": {"ref": "f"}, "len": 4096, "offset": 0, "expect": ["EBADF"]}},
    {"close": {"file": {"ref": "f"}}}
  ]}}
}"#;

#[test]
fn a_request_the_kernel_refuses_is_its_errno_on_every_backend() {
    // under libaio io_submit itself refuses the request; under io_uring it completes with the
    // errno; under mmap the read is a mapping, which a write-only open cannot have (EACCES)
    let (loaded, model) = leaked(WRONG_MODE);
    let dry = aeiou::dryrun::run(model, 1, None).unwrap();
    for backend in EVERY {
        let root = crate::common::tmpdir("cov-c2", "wrongmode");
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        let r = run::run(model, opts(&root, backend), Default::default()).unwrap_or_else(|e| panic!("{backend:?}: {e:#}"));
        assert_eq!((r.stats.fingerprint, r.stats.expected_errors, r.stats.bytes_read, r.stats.bytes_written), (dry.total.fingerprint, 2, 0, 4096), "{backend:?}");
    }
}

/// A file per GPU written, read back, and removed: on one loop, every instance's request at
/// once.
const WIDE: &str = r#"{
  "ast": "0.10", "name": "wide",
  "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": "as_written", "seed": 3}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"v": {"actor": "id"}}}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDWR", "CREAT"]}},
    {"write": {"file": {"ref": "f"}, "len": 4096, "offset": 0}},
    {"read": {"file": {"ref": "f"}, "len": 4096, "offset": 0}},
    {"close": {"file": {"ref": "f"}}},
    {"unlink": {"file": {"ref": "f"}}}
  ]}}
}"#;

#[test]
fn a_loop_with_more_requests_than_its_rings_hold_submits_them_all() {
    // 5000 instances on one loop: the ring's submission queue (1024) fills within one turn and
    // is submitted to make room, and the completions overflow its completion queue (4096); the
    // AIO context of the smallest depth is full, and the loop reaps to make room
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(WIDE).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(Config { seed: 1, gpus: 5000, overrides: vec![], sets: vec![] }));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let dry = aeiou::dryrun::run(model, 1, None).unwrap();
    for backend in [BackendKind::Uring, BackendKind::LibAio] {
        let root = crate::common::tmpdir("cov-c2", "wide");
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        let r = run::run(model, RunOpts { threads: 1, aio_depth: 1, ..opts(&root, backend) }, Default::default()).unwrap_or_else(|e| panic!("{backend:?}: {e:#}"));
        assert_eq!((r.stats.fingerprint, r.stats.ops), (dry.total.fingerprint, 25000), "{backend:?}");
        assert_eq!((r.stats.bytes_read, r.stats.bytes_written), (5000 * 4096, 5000 * 4096), "{backend:?}");
        assert_eq!(std::fs::read_dir(root.join("o")).unwrap().count(), 0, "{backend:?}");
        if let Some(u) = &r.uring {
            assert!(u.in_flight_peak > 4096, "{} requests were in flight at once", u.in_flight_peak);
        }
        if let Some(a) = &r.aio {
            assert!(a.full > 0 && a.in_flight_peak > 4096, "full {} in flight peak {}", a.full, a.in_flight_peak);
        }
    }
}

/// A `DONTNEED` over a range of 4 GiB and one page: every page of the file written before it.
const FADVISE_4G: &str = r#"{
  "ast": "0.10", "name": "fadv",
  "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": "as_written", "seed": 3}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"object": {"namespace": "o", "fields": {"v": 0}}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDWR", "CREAT"]}},
    {"write": {"file": {"ref": "f"}, "len": 65536, "offset": 0}},
    {"fsync": {"file": {"ref": "f"}}},
    {"fadvise": {"file": {"ref": "f"}, "advice": "DONTNEED", "offset": 0, "len": 4294971392}},
    {"close": {"file": {"ref": "f"}}}
  ]}}
}"#;

#[test]
fn an_fadvise_of_4_gib_or_more_advises_its_whole_range_on_every_backend() {
    // RC1 review: the io_uring SQE carries an fadvise's length in 32 bits, so this range was
    // advised as its low 32 bits, one page, and the file's pages stayed in the page cache
    let (loaded, model) = leaked(FADVISE_4G);
    let mut resident = Vec::new();
    for backend in EVERY {
        let root = crate::common::tmpdir("cov-c2", "fadv4g");
        run::prepare_namespaces(&loaded.ast, &root, false).unwrap();
        run::run(model, opts(&root, backend), Default::default()).unwrap_or_else(|e| panic!("{backend:?}: {e:#}"));
        resident.push((backend, aeiou::cold::file_residency(&root.join("o/0")).unwrap()));
    }
    if resident[0].1 .1 != 0 {
        // a filesystem that is its page cache (tmpfs) drops nothing for anyone
        eprintln!("skipped: DONTNEED leaves the pages of {} resident under sync too: {:?}", std::env::temp_dir().display(), resident[0]);
        return;
    }
    for (backend, (pages, left)) in resident {
        assert_eq!((pages, left), (16, 0), "{backend:?}: pages of the file left in the page cache after the DONTNEED");
    }
}

#[test]
fn a_traces_gaps_are_waited_out_on_the_event_loops_as_on_the_threads() {
    // a lane's gap before a line is `compute`: under a time scale the lane task parks on a
    // timer (never tested on the loops before: every trace test ran at time scale 0)
    let lines = r#"{"aeiou_trace":1,"source":"strace","root":"/d","lanes":2,"lines":6,"opens":1,"creates":[],"notes":{}}
{"lane":0,"t":0,"dur":0,"op":"open","fd":0,"path":"f","flags":["RDONLY"]}
{"lane":1,"t":10000000,"dur":0,"op":"stat","path":"f"}
{"lane":0,"t":30000000,"dur":0,"op":"read","fd":0,"offset":0,"len":4096,"ret":4096}
{"lane":1,"t":40000000,"dur":0,"op":"stat","path":"f"}
{"lane":0,"t":60000000,"dur":0,"op":"close","fd":0}
{"lane":1,"t":70000000,"dur":0,"op":"stat","path":"f"}
"#;
    use sha2::Digest;
    let d = crate::common::tmpdir("cov-c2", "trace-gaps");
    std::fs::write(d.join("t.jsonl"), lines).unwrap();
    let sha: String = sha2::Sha256::digest(lines.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    let ast = format!(r#"{{"ast": "0.10", "name": "gaps", "actors": {{"app": {{"count": 1, "body": [{{"trace": {{"file": "t.jsonl", "sha256": "{sha}"}}}}]}}}}}}"#);
    std::fs::write(d.join("t.ast.json"), ast).unwrap();
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&d.join("t.ast.json")).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![] }));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static mut Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    model.traces = loaded.traces.clone();
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("f"), vec![1u8; 4096]).unwrap();
    for backend in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio] {
        let r = run::run(model, RunOpts { time_scale: 1.0, ..opts(&root, backend) }, Default::default()).unwrap_or_else(|e| panic!("{backend:?}: {e:#}"));
        // each lane's gaps add up to 60 and 70 ms (lane 1's first line is 10 ms in)
        assert_eq!((r.stats.ops, r.stats.compute_ns), (6, 130_000_000), "{backend:?}");
        assert!(r.elapsed >= std::time::Duration::from_millis(70), "{backend:?}: the run took {:?}, under the trace's 70 ms", r.elapsed);
    }
}
