//! The `trace` node (`DESIGN_REVIEW.md` §3.58): a trace file is loaded against its sha256,
//! walked in line order by the dry run (lane and ordinal as the indices), run with one
//! thread per lane (the blocking backends) or one task per lane on an event loop under the
//! open table and path order, checked against `--root` before the gate, cleaned by
//! `--clean-namespaces`, and refused where the design says (V16). The fixtures are written
//! here, as `aeiou-trace export` writes them.
use std::path::{Path, PathBuf};

use aeiou::backend::BackendKind;
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};
use sha2::{Digest, Sha256};

fn tmpdir(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("trace-test", tag)
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Write `lines` as a trace file and an abstract that is one `trace` node in an actor of
/// `count` instances (`None`: the default, `gpus`); returns the abstract's path.
fn abstract_with(dir: &Path, lines: &str, count: Option<i64>) -> PathBuf {
    let text = lines.trim_start().to_string();
    std::fs::write(dir.join("t.jsonl"), &text).unwrap();
    let count = count.map(|c| format!(r#""count": {c},"#)).unwrap_or_default();
    let ast = format!(
        r#"{{"ast": "0.11", "name": "trace_test", "doc": "t", "params": {{}}, "datasets": {{}},
            "actors": {{"app": {{{count} "body": [{{"trace": {{"file": "t.jsonl", "sha256": "{}"}}}}]}}}}}}"#,
        sha(text.as_bytes())
    );
    let p = dir.join("t.ast.json");
    std::fs::write(&p, ast).unwrap();
    p
}

fn leaked(path: &Path, gpus: i64) -> (&'static aeiou::Loaded, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(path).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(Config { seed: 1, gpus, overrides: vec![], sets: vec![], st_blksize: Default::default() }));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static mut Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    model.traces = loaded.traces.clone();
    (loaded, model)
}

fn opts(root: &Path, backend: BackendKind, clean: bool) -> RunOpts {
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
        clean_namespaces: clean,
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

/// Two lanes over a shared open, a file the trace creates and another lane then reads
/// (path order), a group, a failed stat, and gaps.
const FIXTURE: &str = r#"
{"aeiou_trace":1,"source":"strace","root":"/d","lanes":2,"lines":19,"opens":4,"creates":["out/new","out/final"],"notes":{}}
{"lane":0,"t":0,"dur":1000,"op":"open","fd":0,"path":"shared","flags":["RDONLY","CLOEXEC"]}
{"lane":0,"t":2000,"dur":1000,"op":"fstat","fd":0}
{"lane":0,"t":4000,"dur":1000,"op":"read","fd":0,"offset":0,"len":4096,"ret":4096}
{"lane":1,"t":4500,"dur":1000,"op":"read","fd":0,"offset":4096,"len":4096,"ret":4096}
{"lane":1,"t":6000,"dur":1000,"op":"open","fd":1,"path":"own","flags":["RDONLY"]}
{"lane":1,"t":8000,"dur":1000,"op":"read","fd":1,"len":100,"ret":100}
{"lane":1,"t":10000,"dur":1000,"op":"read","fd":1,"len":100,"ret":100}
{"lane":1,"t":12000,"dur":1000,"op":"submit","ops":[{"op":"read","fd":1,"offset":0,"len":50,"ret":50},{"op":"read","fd":1,"offset":100,"len":50,"ret":50},{"op":"read","fd":1,"offset":200,"len":50,"ret":50}]}
{"lane":1,"t":14000,"dur":1000,"op":"close","fd":1}
{"lane":0,"t":15000,"dur":1000,"op":"open","fd":2,"path":"out/new","flags":["WRONLY","CREAT","TRUNC"],"mode":420}
{"lane":0,"t":17000,"dur":1000,"op":"write","fd":2,"len":1000,"ret":1000}
{"lane":0,"t":19000,"dur":1000,"op":"close","fd":2}
{"lane":0,"t":21000,"dur":1000,"op":"rename","path":"out/new","to":"out/final"}
{"lane":1,"t":23000,"dur":1000,"op":"open","fd":3,"path":"out/final","flags":["RDONLY"]}
{"lane":1,"t":25000,"dur":1000,"op":"read","fd":3,"len":4096,"ret":1000}
{"lane":1,"t":27000,"dur":1000,"op":"close","fd":3}
{"lane":1,"t":29000,"dur":1000,"op":"close","fd":0}
{"lane":1,"t":31000,"dur":1000,"op":"stat","path":"missing","ret":"ENOENT"}
{"lane":0,"t":33000,"dur":1000,"op":"close","fd":0}
"#;

fn populate(root: &Path) {
    std::fs::create_dir_all(root.join("out")).unwrap();
    std::fs::write(root.join("shared"), vec![b'x'; 9000]).unwrap();
    std::fs::write(root.join("own"), vec![b'y'; 300]).unwrap();
}

#[test]
fn loads_against_its_sha256_and_indexes_lanes_opens_and_dependencies() {
    let d = tmpdir("load");
    let p = abstract_with(&d, FIXTURE, Some(1));
    let loaded = aeiou::load(&p).unwrap();
    let tf = loaded.traces.get("t.jsonl").expect("the trace is loaded with the document");
    assert_eq!((tf.lanes(), tf.opens.len(), tf.lines.len(), tf.ops), (2, 4, 19, 21));
    // open 0 is used by both lanes (shared), closed twice; the group's members are uses
    assert_eq!((tf.opens[0].uses, tf.opens[0].closes), (3, 2));
    assert_eq!((tf.opens[1].uses, tf.opens[1].closes), (5, 1));
    assert_eq!(tf.header.creates, vec!["out/new", "out/final"]);
    assert_eq!(tf.inputs, vec!["shared", "own"]);
    assert_eq!(tf.read_extent["shared"], 8192);
    assert_eq!(tf.read_extent["own"], 250);
    assert_eq!(tf.peak_open, 2);
    // path order: the reader's open of out/final waits for the rename (a changing op before it)
    let g_open_final = tf.lines[14].g0;
    let dep = tf.deps[g_open_final].expect("out/final is a path the trace changes");
    assert!(!dep.mutating && dep.mk == 1, "{dep:?}");
    // reads of a never-changed path carry no dependency
    assert!(tf.deps[tf.lines[2].g0].is_none());

    // a changed file is refused by its hash
    let text = std::fs::read_to_string(d.join("t.jsonl")).unwrap() + "\n";
    std::fs::write(d.join("t.jsonl"), text).unwrap();
    let e = match aeiou::load(&p) {
        Ok(_) => panic!("a changed trace file loaded"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("sha256"), "{e}");
}

#[test]
fn dry_run_walks_in_line_order_with_lane_and_ordinal_as_indices() {
    let d = tmpdir("dry");
    let p = abstract_with(&d, FIXTURE, Some(1));
    let (_, model) = leaked(&p, 1);
    let filter = dryrun::Filter { actor: Some(0), steps: None, limit: None };
    let mut r = dryrun::run(model, 1, Some(filter)).unwrap();
    assert_eq!(r.total.total.ops, 21);
    assert_eq!(r.total.total.bytes_read, 4096 * 2 + 200 + 150 + 1000);
    assert_eq!(r.total.total.bytes_written, 1000);
    let lines = r.total.take_lines();
    assert!(lines[0].starts_with("app#0 [0,0] open shared"), "{}", lines[0]);
    assert!(lines[3].starts_with("app#0 [1,0] read shared off=4096"), "{}", lines[3]);
    // the group's members share the line's ordinal; the ordinal counts lines per lane
    assert!(lines[7].starts_with("app#0 [1,4] read own off=0 len=50") && lines[9].starts_with("app#0 [1,4] read own off=200"), "{} / {}", lines[7], lines[9]);
    assert!(lines[10].starts_with("app#0 [1,5] close own"), "{}", lines[10]);
    assert!(lines.last().unwrap().starts_with("app#0 [0,7] close shared"), "{}", lines.last().unwrap());
    // the gaps are compute
    assert!(r.total.compute_ns > 0);

    // the metrics walk sees the same ops, one group of three as a fan-out, a lane as a context
    let m = dryrun::run_with(model, 1, None, Some(aeiou::metrics::Opts::default())).unwrap();
    assert_eq!(m.total.fingerprint, r.total.fingerprint);
    let mm = m.total.metrics.as_ref().unwrap();
    assert_eq!(mm.fan_out.get(&3), Some(&1));
    assert!(mm.depth.is_empty());
}

#[test]
fn runs_with_the_dry_runs_fingerprint_under_every_backend() {
    let d = tmpdir("run");
    let p = abstract_with(&d, FIXTURE, Some(1));
    let (_, model) = leaked(&p, 1);
    let dry = dryrun::run(model, 1, None).unwrap();
    let root = d.join("root");
    populate(&root);
    // the blocking drivers (one thread per lane) and the event loops (one task per lane,
    // the group's members in flight together) must give the same run
    for be in [BackendKind::Sync, BackendKind::PosixAio, BackendKind::Mmap, BackendKind::Uring, BackendKind::LibAio] {
        // what the trace creates must be absent, so every run but the first cleans up
        for p in ["out/final", "out/new"] {
            let _ = std::fs::remove_file(root.join(p));
        }
        let rep = run::run(model, opts(&root, be, false), Default::default()).unwrap();
        assert_eq!(rep.stats.fingerprint, dry.total.fingerprint, "{}", be.describe());
        assert_eq!(rep.stats.ops, 21);
        assert_eq!(rep.stats.bytes_written, 1000);
        assert_eq!(rep.stats.expected_errors, 1);
        assert_eq!(std::fs::metadata(root.join("out/final")).unwrap().len(), 1000);
        assert!(rep.created.iter().any(|(p, _)| p == "out/final"), "{:?}", rep.created);
        if be.event_loop() {
            // the lanes are tasks on the loop, not threads
            assert_eq!(rep.stats.threads, 1, "{}", be.describe());
        }
    }
}

#[test]
fn a_read_over_4_gib_moves_what_sync_moves_under_every_backend() {
    // An SQE carries its length in 32 bits: a read of 4 GiB + 4096 cast whole went to the
    // ring as a read of 4096, and on an 8 KiB file returned 4096 where `pread` returns 8192.
    // The kernel moves at most MAX_RW_COUNT in one call whatever the API, so every backend
    // issues the length `sync` does and returns the same count. Nothing over 8 KiB is touched.
    let lines = r#"
{"aeiou_trace":1,"source":"strace","root":"/d","lanes":1,"lines":3,"opens":1,"creates":[],"notes":{}}
{"lane":0,"t":0,"dur":100,"op":"open","fd":0,"path":"big","flags":["RDONLY"]}
{"lane":0,"t":200,"dur":100,"op":"read","fd":0,"offset":0,"len":4294971392,"ret":8192}
{"lane":0,"t":400,"dur":100,"op":"close","fd":0}
"#;
    let d = tmpdir("over4g");
    let p = abstract_with(&d, lines, Some(1));
    let (_, model) = leaked(&p, 1);
    let dry = dryrun::run(model, 1, None).unwrap();
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("big"), vec![b'z'; 8192]).unwrap();
    for be in [BackendKind::Sync, BackendKind::Uring, BackendKind::UringDirect, BackendKind::LibAio] {
        let rep = run::run(model, opts(&root, be, false), Default::default()).unwrap_or_else(|e| panic!("{}: {e:#}", be.describe()));
        assert_eq!(rep.stats.fingerprint, dry.total.fingerprint, "{}", be.describe());
        assert_eq!(rep.stats.bytes_read, 8192, "{}", be.describe());
    }
}

#[test]
fn path_order_makes_a_reader_wait_for_the_writer_whatever_the_lanes_timing() {
    // lane 1 reads a file lane 0 writes much later in the trace; with a time scale the
    // reader would start first, and path order holds it until the write and close are done
    let lines = r#"
{"aeiou_trace":1,"source":"strace","root":"/d","lanes":2,"lines":7,"opens":2,"creates":["f"],"notes":{}}
{"lane":1,"t":0,"dur":100,"op":"stat","path":"other"}
{"lane":0,"t":5000000,"dur":100,"op":"open","fd":0,"path":"f","flags":["WRONLY","CREAT","TRUNC"]}
{"lane":0,"t":5000200,"dur":100,"op":"write","fd":0,"len":8192,"ret":8192}
{"lane":0,"t":5000400,"dur":100,"op":"close","fd":0}
{"lane":1,"t":5000600,"dur":100,"op":"open","fd":1,"path":"f","flags":["RDONLY"]}
{"lane":1,"t":5000800,"dur":100,"op":"read","fd":1,"len":8192,"ret":8192}
{"lane":1,"t":5001000,"dur":100,"op":"close","fd":1}
"#;
    let d = tmpdir("order");
    let p = abstract_with(&d, lines, Some(1));
    let (_, model) = leaked(&p, 1);
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("other"), b"").unwrap();
    for be in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio] {
        for _ in 0..5 {
            let _ = std::fs::remove_file(root.join("f"));
            let rep = run::run(model, opts(&root, be, false), Default::default()).unwrap();
            assert_eq!(rep.stats.ops, 7, "{}", be.describe());
            assert_eq!(rep.stats.bytes_read, 8192, "{}", be.describe());
        }
    }
}

#[test]
fn root_checks_before_the_gate_and_clean_namespaces() {
    let d = tmpdir("checks");
    let p = abstract_with(&d, FIXTURE, Some(1));
    let (loaded, _) = leaked(&p, 1);
    let tf = loaded.traces.get("t.jsonl").unwrap();
    let root = d.join("root");
    std::fs::create_dir_all(root.join("out")).unwrap();
    // inputs missing
    let e = aeiou::trace::check_root(tf, &root, true).unwrap_err().to_string();
    assert!(e.contains("shared") && e.contains("--root"), "{e}");
    populate(&root);
    // an input shorter than the trace reads
    std::fs::write(root.join("shared"), vec![b'x'; 100]).unwrap();
    let e = aeiou::trace::check_root(tf, &root, true).unwrap_err().to_string();
    assert!(e.contains("100 bytes") && e.contains("8192"), "{e}");
    populate(&root);
    let c = aeiou::trace::check_root(tf, &root, true).unwrap();
    assert_eq!((c.inputs, c.creates), (2, 2));
    // a created path present is refused; `clean` removes exactly the creates
    std::fs::write(root.join("out/final"), b"old").unwrap();
    std::fs::write(root.join("keep"), b"mine").unwrap();
    let e = aeiou::trace::check_root(tf, &root, true).unwrap_err().to_string();
    assert!(e.contains("out/final") && e.contains("--clean-namespaces"), "{e}");
    // a host other than rank 0 does not check the creates (rank 0 checks them after its clean)
    aeiou::trace::check_root(tf, &root, false).unwrap();
    assert_eq!(aeiou::trace::clean(tf, &root).unwrap(), 1);
    assert!(root.join("keep").exists() && !root.join("out/final").exists());
    aeiou::trace::check_root(tf, &root, true).unwrap();
}

#[test]
fn v16_a_writing_trace_needs_one_instance() {
    let d = tmpdir("v16");
    let p = abstract_with(&d, FIXTURE, None); // count defaults to gpus
    let (loaded, model) = leaked(&p, 2);
    let counts = aeiou::vm::actor_counts(model).unwrap();
    let e = aeiou::trace::check_counts(&loaded.ast, &loaded.traces, &counts).unwrap_err().to_string();
    assert!(e.contains("V16") && e.contains("2 instances"), "{e}");
    let (loaded, model) = leaked(&p, 1);
    let counts = aeiou::vm::actor_counts(model).unwrap();
    aeiou::trace::check_counts(&loaded.ast, &loaded.traces, &counts).unwrap();
}

#[test]
fn a_path_outside_root_is_refused_at_load_and_nothing_outside_is_touched() {
    let d = tmpdir("escape");
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(d.join("outside/victimdir")).unwrap();
    std::fs::write(d.join("outside/victim"), b"precious").unwrap();
    std::fs::write(d.join("outside/victimdir/keep"), b"precious").unwrap();
    let victim = d.join("outside/victim");
    let cases: [(&str, &str); 5] = [
        // `..` in creates: --clean-namespaces would remove it, the open would truncate it
        (r#"{"aeiou_trace":1,"lanes":1,"creates":["../outside/victimdir"]}
{"lane":0,"op":"stat","path":"x"}"#, "line 1: path `../outside/victimdir` has a `..` component"),
        // an absolute op path
        (&format!(r#"{{"aeiou_trace":1,"lanes":1,"creates":[]}}
{{"lane":0,"op":"open","fd":0,"path":"{}","flags":["WRONLY","CREAT","TRUNC"]}}"#, victim.display()), "line 2: path"),
        // `..` hidden in a group's member and a rename's destination
        (r#"{"aeiou_trace":1,"lanes":1,"creates":[]}
{"lane":0,"op":"rename","path":"a","to":"b/../../c"}"#, "line 2: path `b/../../c` has a `..` component"),
        // the root itself, changed
        (r#"{"aeiou_trace":1,"lanes":1,"creates":["."]}"#, "line 1: path `.` is --root itself"),
        (r#"{"aeiou_trace":1,"lanes":1,"creates":[]}

{"lane":0,"op":"rmdir","path":""}"#, "line 3: an empty path"),
    ];
    for (lines, want) in cases {
        let p = abstract_with(&d, lines, Some(1));
        let e = match aeiou::load(&p) {
            Ok(_) => panic!("a trace naming a path outside --root loaded: {lines}"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains(want), "{e}");
        if want == "line 2: path" {
            assert!(e.contains("is absolute"), "{e}");
        }
    }
    // the root itself may be read (the exporter writes `.` for it)
    let p = abstract_with(&d, r#"{"aeiou_trace":1,"lanes":1,"creates":[]}
{"lane":0,"op":"open","fd":0,"path":".","flags":["RDONLY","DIRECTORY"]}
{"lane":0,"op":"readdir","fd":0}
{"lane":0,"op":"close","fd":0}
{"lane":0,"op":"stat","path":"./"}"#, Some(1));
    aeiou::load(&p).unwrap();
    // and the reviewer's run, refused before anything is removed or truncated
    let p = abstract_with(&d, r#"{"aeiou_trace":1,"lanes":1,"creates":["../outside/victim","../outside/victimdir"]}
{"lane":0,"op":"open","fd":0,"path":"../outside/victim","flags":["WRONLY","CREAT","TRUNC"],"mode":420}
{"lane":0,"op":"close","fd":0}"#, Some(1));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou"))
        .args(["run", p.to_str().unwrap(), "--gpus", "1", "--root", root.to_str().unwrap(), "--clean-namespaces", "--time-scale", "0"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && err.contains("`..` component"), "{err}");
    assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
    assert_eq!(std::fs::read(d.join("outside/victimdir/keep")).unwrap(), b"precious");
}

/// Several hosts with `--clean-namespaces` and a path the trace creates left by an earlier
/// run: only rank 0 removes it, so only rank 0 may check it absent, after its clean; a check
/// on another host raced that clean and failed the run at random. Made deterministic here:
/// rank 1's view of the root keeps the stale path (another directory with the same inputs,
/// as a client whose cache still shows the removed path would), which nobody but rank 0 is
/// allowed to empty, so a rank 1 that checked the creates would always refuse.
#[test]
fn only_rank_0_checks_what_a_trace_creates_after_its_clean() {
    let d = tmpdir("ranks");
    let p = abstract_with(&d, FIXTURE, Some(1));
    let (root0, root1) = (d.join("root0"), d.join("root1"));
    for r in [&root0, &root1] {
        populate(r);
        std::fs::write(r.join("out/final"), b"old").unwrap();
    }
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut children = Vec::new();
    for (rank, root) in [(0, &root0), (1, &root1)] {
        let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("run").arg(&p).args(["--gpus", "2", "--time-scale", "0", "--clean-namespaces", "--ranks", "2"]).arg("--root").arg(root);
        c.arg("--rank").arg(rank.to_string()).arg("--coordinator").arg(format!("127.0.0.1:{port}"));
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
    }
    let outs: Vec<_> = children.into_iter().map(|c| c.wait_with_output().unwrap()).collect();
    for (rank, o) in outs.iter().enumerate() {
        assert!(o.status.success(), "rank {rank}: {}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    }
    let r1 = String::from_utf8_lossy(&outs[1].stdout);
    assert!(r1.contains("trace t.jsonl: 2 input path(s) present under --root, 2 to be created checked by rank 0"), "{r1}");
    assert!(String::from_utf8_lossy(&outs[0].stdout).contains("trace t.jsonl: 1 created path(s) removed"));
    // rank 0 cleaned and ran on its root; rank 1 touched nothing on its own
    assert_eq!(std::fs::read(root0.join("out/final")).unwrap().len(), 1000);
    assert_eq!(std::fs::read(root1.join("out/final")).unwrap(), b"old");
}

/// A line the runner cannot execute is refused when the file loads, with its line number,
/// by `check`, `dry-run`, and `run` alike: a `submit` holds one or more positioned reads and
/// writes and nothing else (a nested one panicked the dry run, an `open` in one hung the
/// run), a lane is in range (a huge one allocated without end), the header's line count is
/// never an allocation (a huge one overflowed it), lengths, times, and offsets stay in range,
/// and an open that failed in the trace has no uses.
#[test]
fn a_malformed_line_is_refused_at_load_with_its_line_number() {
    let d = tmpdir("malformed");
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("f"), vec![0u8; 9000]).unwrap();
    let h = r#"{"aeiou_trace":1,"lanes":1,"creates":[]}"#;
    let open = r#"{"lane":0,"op":"open","fd":0,"path":"f","flags":["RDONLY"]}"#;
    let cases: Vec<(String, &str)> = vec![
        (format!("{h}\n\n{}", r#"{"lane":0,"op":"submit","ops":[{"op":"submit","ops":[]}]}"#), "line 3: `submit` member 0 is a nested `submit`"),
        (format!("{h}\n{}", r#"{"lane":0,"op":"submit","ops":[]}"#), "line 2: an empty `submit`"),
        (format!("{h}\n{}", r#"{"lane":0,"op":"submit","ops":[{"op":"open","fd":0,"path":"f","flags":["RDONLY"]}]}"#), "line 2: `submit` member 0 is an op other than a read or a write"),
        (format!("{h}\n{open}\n{}", r#"{"lane":0,"op":"submit","ops":[{"op":"read","fd":0,"offset":0,"len":10},{"op":"close","fd":0}]}"#), "line 3: `submit` member 1 is an op other than"),
        (format!("{h}\n{open}\n{}", r#"{"lane":0,"op":"submit","ops":[{"op":"read","fd":0,"len":10}]}"#), "line 3: `submit` member 0 is a sequential read"),
        (format!("{h}\n{open}\n{}", r#"{"lane":0,"op":"read","fd":0,"offset":0,"len":-5}"#), "line 3: a read of length -5"),
        (format!("{h}\n{}", r#"{"lane":4000000000000,"op":"stat","path":"f"}"#), "line 2: lane 4000000000000 out of range"),
        (format!("{}\n{}", r#"{"aeiou_trace":1,"lines":4000000000000000000,"creates":[]}"#, r#"{"lane":1,"op":"stat","path":"f"}"#), "line 2: lane 1 out of range"),
        (format!("{h}\n{}", r#"{"lane":0,"t":9223372036854775807,"dur":1,"op":"stat","path":"f"}"#), "line 2: `t` 9223372036854775807 and `dur` 1"),
        (format!("{h}\n{open}\n{}", r#"{"lane":0,"op":"read","fd":0,"offset":9223372036854775000,"len":4096,"ret":4096}"#), "line 3: the read's end passes"),
        (format!("{h}\n{}\n{}", r#"{"lane":0,"op":"open","fd":0,"path":"nope","flags":["RDONLY"],"ret":"ENOENT"}"#, r#"{"lane":0,"op":"read","fd":0,"len":10}"#), "line 3: open id 0 is used, and its open failed"),
    ];
    let bin = env!("CARGO_BIN_EXE_aeiou");
    for (lines, want) in &cases {
        let p = abstract_with(&d, lines, Some(1));
        let e = match aeiou::load(&p) {
            Ok(_) => panic!("a malformed trace loaded: {lines}"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains(want), "{e}");
        let (p, root) = (p.to_str().unwrap(), root.to_str().unwrap());
        for args in [vec!["check", p], vec!["dry-run", p, "--gpus", "1"], vec!["run", p, "--gpus", "1", "--root", root, "--time-scale", "0"]] {
            let out = std::process::Command::new(bin).args(&args).output().unwrap();
            let all = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            assert_eq!(out.status.code(), Some(1), "{} on {lines}: {all}", args[0]);
            assert!(all.contains(want) && !all.contains("panicked"), "{} on {lines}: {all}", args[0]);
        }
    }
    // the header's line count is a hint: a huge one over a well-formed file loads
    let p = abstract_with(&d, r#"{"aeiou_trace":1,"lanes":1,"lines":4000000000000000000,"creates":[]}
{"lane":0,"op":"stat","path":"f"}"#, Some(1));
    aeiou::load(&p).unwrap();
}

#[test]
fn the_root_check_places_a_read_where_the_walk_does_after_a_short_write_or_a_failed_lseek() {
    // `index` (the extents `check_root` holds an input to) and `build` (the walk and the run)
    // share one position rule: a short write moves the position by what it returned, so the
    // read after it ends at 110, not 1100; a failed `lseek` leaves the position where it was
    let lines = r#"
{"aeiou_trace":1,"source":"strace","root":"/d","lanes":1,"lines":7,"opens":1,"creates":[],"notes":{}}
{"lane":0,"t":0,"dur":100,"op":"open","fd":0,"path":"f","flags":["RDWR"]}
{"lane":0,"t":200,"dur":100,"op":"write","fd":0,"len":1000,"ret":10}
{"lane":0,"t":400,"dur":100,"op":"read","fd":0,"len":100,"ret":100}
{"lane":0,"t":600,"dur":100,"op":"lseek","fd":0,"offset":-5,"whence":"SET","ret":"EINVAL"}
{"lane":0,"t":800,"dur":100,"op":"read","fd":0,"len":50,"ret":50}
{"lane":0,"t":1000,"dur":100,"op":"close","fd":0}
"#;
    let d = tmpdir("shortpos");
    let p = abstract_with(&d, lines, Some(1));
    let (loaded, model) = leaked(&p, 1);
    let filter = dryrun::Filter { actor: Some(0), steps: None, limit: None };
    let mut dry = dryrun::run(model, 1, Some(filter)).unwrap();
    let walked = dry.total.take_lines();
    assert!(walked[2].starts_with("app#0 [0,2] read f off=10 len=100"), "{}", walked[2]);
    assert!(walked[4].starts_with("app#0 [0,4] read f off=110 len=50"), "{}", walked[4]);
    let tf = loaded.traces.get("t.jsonl").unwrap();
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    // one byte short of where the application read to is refused, naming that end
    std::fs::write(root.join("f"), vec![b'f'; 159]).unwrap();
    let e = aeiou::trace::check_root(tf, &root, true).unwrap_err().to_string();
    assert!(e.contains("159 bytes") && e.contains("reads to 160"), "{e}");
    for be in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio] {
        // the run writes the call's length, so each run starts from the input's size
        std::fs::write(root.join("f"), vec![b'f'; 160]).unwrap();
        let rep = run::run(model, opts(&root, be, false), Default::default()).unwrap_or_else(|e| panic!("{}: {e:#}", be.describe()));
        assert_eq!(rep.stats.fingerprint, dry.total.fingerprint, "{}", be.describe());
        assert_eq!(rep.stats.bytes_read, 150, "{}", be.describe());
    }
}

/// Every op a trace line may hold, issued under the blocking driver and the event loops to
/// the dry run's fingerprint: the descriptor ops on a file the trace makes (sync, truncate,
/// allocate, advise, ioctl, seek, stat, a listing), a read that failed in the trace (its
/// errno the op's `expect`, its count 0), a failed `mkdir`, and the path ops that take it
/// all away again.
#[test]
fn every_op_of_a_trace_runs_to_the_dry_runs_fingerprint() {
    let lines = r#"
{"aeiou_trace":1,"lanes":1,"creates":["dir"]}
{"lane":0,"op":"mkdir","path":"dir","mode":493}
{"lane":0,"op":"mkdir","path":"in","ret":"EEXIST"}
{"lane":0,"op":"open","fd":0,"path":"dir/a","flags":["WRONLY","CREAT","TRUNC"],"mode":420}
{"lane":0,"op":"write","fd":0,"len":100,"ret":100}
{"lane":0,"op":"fsync","fd":0}
{"lane":0,"op":"fdatasync","fd":0}
{"lane":0,"op":"ftruncate","fd":0,"len":50}
{"lane":0,"op":"fallocate","fd":0,"offset":0,"len":200}
{"lane":0,"op":"fadvise","fd":0,"advice":"DONTNEED"}
{"lane":0,"op":"ioctl","fd":0,"request":"TCGETS","ret":"ENOTTY"}
{"lane":0,"op":"lseek","fd":0,"offset":0,"whence":"SET","ret":0}
{"lane":0,"op":"read","fd":0,"len":10,"ret":"EBADF"}
{"lane":0,"op":"fstat","fd":0}
{"lane":0,"op":"close","fd":0}
{"lane":0,"op":"open","fd":1,"path":"dir","flags":["RDONLY","DIRECTORY"]}
{"lane":0,"op":"readdir","fd":1}
{"lane":0,"op":"close","fd":1}
{"lane":0,"op":"rename","path":"dir/a","to":"dir/b"}
{"lane":0,"op":"stat","path":"dir/b"}
{"lane":0,"op":"unlink","path":"dir/b"}
{"lane":0,"op":"rmdir","path":"dir"}
"#;
    let d = tmpdir("every-op");
    let p = abstract_with(&d, lines, Some(1));
    let (loaded, model) = leaked(&p, 1);
    let tf = loaded.traces.get("t.jsonl").unwrap();
    // `in` is named only by the failed mkdir: no input; `dir` is made by the trace
    assert_eq!(tf.inputs, Vec::<String>::new());
    assert_eq!(tf.ops, 21);
    let dry = dryrun::run(model, 1, None).unwrap();
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("in"), b"x").unwrap();
    for be in [BackendKind::Sync, BackendKind::Uring, BackendKind::LibAio, BackendKind::PosixAio] {
        let rep = run::run(model, opts(&root, be, false), Default::default()).unwrap_or_else(|e| panic!("{}: {e:#}", be.describe()));
        assert_eq!(rep.stats.fingerprint, dry.total.fingerprint, "{}", be.describe());
        assert_eq!((rep.stats.ops, rep.stats.expected_errors, rep.stats.bytes_written), (21, 3, 100), "{}", be.describe());
        assert!(!root.join("dir").exists(), "{}", be.describe());
    }
}

/// A read that returned nothing read nothing: one past the end of its file (an HDF5 library
/// probing for its superblock at 512, 1024, 2048, … of a small file, a `pread` at a stale
/// size) says the file was no longer than its offset, never that it was at least that long,
/// so the root check holds the file to the reads that returned bytes (RC1 review: a read of 0
/// at 4096 refused the 100-byte file the application read).
#[test]
fn a_read_that_returned_nothing_asks_nothing_of_the_files_size() {
    let lines = r#"
{"aeiou_trace":1,"lanes":1,"creates":[]}
{"lane":0,"op":"open","fd":0,"path":"f","flags":["RDONLY"]}
{"lane":0,"op":"read","fd":0,"offset":0,"len":100,"ret":100}
{"lane":0,"op":"read","fd":0,"offset":4096,"len":100,"ret":0}
{"lane":0,"op":"read","fd":0,"offset":100,"len":100,"ret":0}
{"lane":0,"op":"close","fd":0}
"#;
    let d = tmpdir("eof-probe");
    let p = abstract_with(&d, lines, Some(1));
    let (loaded, model) = leaked(&p, 1);
    let tf = loaded.traces.get("t.jsonl").unwrap();
    assert_eq!(tf.read_extent["f"], 100);
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("f"), vec![b'f'; 100]).unwrap();
    aeiou::trace::check_root(tf, &root, true).unwrap();
    let dry = dryrun::run(model, 1, None).unwrap();
    let rep = run::run(model, opts(&root, BackendKind::Sync, false), Default::default()).unwrap();
    assert_eq!((rep.stats.fingerprint, rep.stats.bytes_read), (dry.total.fingerprint, 100));
    // one byte short of the read that returned bytes is still refused
    std::fs::write(root.join("f"), vec![b'f'; 99]).unwrap();
    let e = aeiou::trace::check_root(tf, &root, true).unwrap_err().to_string();
    assert!(e.contains("99 bytes, the trace reads to 100"), "{e}");
}

/// The errno a trace (or an abstract's `expect`) names is matched by its number, whatever the
/// errno: an `O_NOFOLLOW` open of a symlink fails `ELOOP` in the application and in the run
/// alike (RC1 review: an errno outside a short list of names was "E?", so it matched no
/// `expect` and failed a run that did exactly what the application did), and a name's alias
/// (`ENOTSUP` for `EOPNOTSUPP`, `EWOULDBLOCK` for `EAGAIN`) is the same errno.
#[test]
fn every_errno_a_trace_names_matches_the_one_the_run_gets() {
    let lines = r#"
{"aeiou_trace":1,"lanes":1,"creates":[]}
{"lane":0,"op":"open","fd":0,"path":"link","flags":["RDONLY","NOFOLLOW"],"ret":"ELOOP"}
{"lane":0,"op":"open","fd":1,"path":"f/x","flags":["RDONLY"],"ret":"ENOTDIR"}
{"lane":0,"op":"open","fd":2,"path":"f","flags":["RDONLY"]}
{"lane":0,"op":"lseek","fd":2,"offset":0,"whence":"SET","ret":0}
{"lane":0,"op":"close","fd":2}
"#;
    let d = tmpdir("errno");
    let p = abstract_with(&d, lines, Some(1));
    let (_, model) = leaked(&p, 1);
    let root = d.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("f"), b"x").unwrap();
    std::os::unix::fs::symlink("f", root.join("link")).unwrap();
    for be in [BackendKind::Sync, BackendKind::Uring] {
        let rep = run::run(model, opts(&root, be, false), Default::default()).unwrap_or_else(|e| panic!("{}: {e:#}", be.describe()));
        assert_eq!(rep.stats.expected_errors, 2, "{}", be.describe());
    }
    use aeiou::backend::{errno_is, errno_name};
    assert_eq!([libc::ELOOP, libc::ESPIPE, libc::EOPNOTSUPP, libc::EAGAIN].map(|c| errno_name(c).into_owned()), ["ELOOP", "ESPIPE", "EOPNOTSUPP", "EAGAIN"]);
    assert!(errno_is("ENOTSUP", libc::EOPNOTSUPP) && errno_is("EOPNOTSUPP", libc::EOPNOTSUPP) && errno_is("EWOULDBLOCK", libc::EAGAIN) && errno_is("EDEADLOCK", libc::EDEADLK));
    assert!(!errno_is("ENOENT", libc::EIO) && !errno_is("E?", 0) && !errno_is("ENOSUCH", libc::EIO));
    assert_eq!(errno_name(4095), "E4095");
}

/// A trace's header names a format this runner reads, its opens are numbered in order, its
/// lanes are as many as the header says, and one file is named with one sha256.
#[test]
fn a_trace_of_another_format_or_out_of_order_is_refused_at_load() {
    let d = tmpdir("refused");
    let stat = r#"{"lane":0,"op":"stat","path":"f"}"#;
    for (lines, want) in [
        (format!("{}\n{stat}", r#"{"aeiou_trace":2}"#), "aeiou_trace 2 is not 1"),
        (format!("{}\n{}", r#"{"aeiou_trace":1}"#, r#"{"lane":0,"op":"open","fd":1,"path":"f","flags":["RDONLY"]}"#), "line 2: open id 1 out of order (expected 0)"),
        (format!("{}\n{stat}", r#"{"aeiou_trace":1,"lanes":2}"#), "1 lanes in the file, the header says 2"),
        (format!("{}\n{}", r#"{"aeiou_trace":1}"#, r#"{"lane":0,"op":"submit","ops":[{"op":"write","fd":0,"len":1}]}"#), "`submit` member 0 is a sequential write (no `offset`)"),
        ("\n  \n".to_string(), "empty file"),
        (format!("{}\n{stat}", r#"{"aeiou_trace":1,"creates":["../up"]}"#), "line 1: path `../up` has a `..` component"),
    ] {
        let p = abstract_with(&d, &lines, Some(1));
        let e = format!("{:#}", aeiou::load(&p).err().unwrap_or_else(|| panic!("loaded: {lines}")));
        assert!(e.contains(want), "{e}");
    }
    // one file named twice with two sha256 values; and a trace inside a `choose` arm is found
    let text = format!("{}\n{stat}", r#"{"aeiou_trace":1}"#);
    std::fs::write(d.join("t.jsonl"), &text).unwrap();
    let good = sha(text.as_bytes());
    let doc = |other: &str| {
        format!(
            r#"{{"ast": "0.11", "name": "trace_test", "doc": "t", "params": {{}}, "datasets": {{}},
            "actors": {{"app": {{"count": 1, "body": [{{"choose": {{"arms": [
               {{"weight": 1, "body": [{{"trace": {{"file": "t.jsonl", "sha256": "{good}"}}}}]}},
               {{"weight": 1, "body": [{{"trace": {{"file": "t.jsonl", "sha256": "{other}"}}}}]}}]}}}}]}}}}}}"#
        )
    };
    let p = d.join("two.ast.json");
    std::fs::write(&p, doc(&good)).unwrap();
    let loaded = aeiou::load(&p).unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(aeiou::trace::refs(&loaded.ast).len(), 2);
    std::fs::write(&p, doc(&"0".repeat(64))).unwrap();
    let e = format!("{:#}", aeiou::load(&p).err().expect("two sha256 values for one file"));
    assert!(e.contains("trace `t.jsonl` is named with two sha256 values"), "{e}");
}

/// A trace's paths are issued where the run places them: one under the root of a dataset or
/// namespace given an `--endpoint` lives there, so the checks before the gate look for it
/// there too, and `--clean-namespaces` removes what the trace creates there (RC1 review: the
/// checks looked under `--root`, refusing a run whose input was where the run reads it).
#[test]
fn a_traces_paths_are_checked_where_their_endpoints_place_them() {
    let d = tmpdir("endpoint");
    let lines = r#"{"aeiou_trace":1,"lanes":1,"creates":["o/new"]}
{"lane":0,"op":"open","fd":0,"path":"d/f0","flags":["RDONLY"]}
{"lane":0,"op":"read","fd":0,"len":100,"ret":100}
{"lane":0,"op":"close","fd":0}
{"lane":0,"op":"open","fd":1,"path":"o/new","flags":["WRONLY","CREAT","TRUNC"]}
{"lane":0,"op":"write","fd":1,"len":10,"ret":10}
{"lane":0,"op":"close","fd":1}"#;
    std::fs::write(d.join("t.jsonl"), lines).unwrap();
    let ast = d.join("t.ast.json");
    std::fs::write(
        &ast,
        format!(
            r#"{{"ast": "0.11", "name": "trace_ep", "doc": "t", "params": {{}},
             "datasets": {{"d": {{"layout": "whole", "pattern": "d/f{{id}}", "count": 1, "size": {{"const": 100}}, "seed": 3, "access": "map"}}}},
             "namespaces": {{"o": {{"pattern": "o/f{{k}}", "fields": {{"k": "int"}}, "size": "as_written", "seed": 1}}}},
             "actors": {{"app": {{"count": 1, "body": [{{"trace": {{"file": "t.jsonl", "sha256": "{}"}}}}]}}}}}}"#,
            sha(lines.as_bytes())
        ),
    )
    .unwrap();
    let (root, data, out) = (d.join("root"), d.join("data"), d.join("out"));
    std::fs::create_dir_all(&root).unwrap();
    let aeiou = |args: &[&str]| {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou")).args(args).env_remove("AEIOU_CONFIG").output().unwrap();
        (o.status.success(), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
    };
    let (a, r, dd, oo) = (ast.to_str().unwrap(), root.to_str().unwrap(), format!("d={}", data.display()), format!("o={}", out.display()));
    let (ok, o) = aeiou(&["datagen", a, "--root", r, "--endpoint", &dd]);
    assert!(ok, "{o}");
    let run = |extra: &[&str]| {
        let mut args = vec!["run", a, "--gpus", "1", "--root", r, "--endpoint", &dd, "--endpoint", &oo, "--time-scale", "0"];
        args.extend(extra);
        aeiou(&args)
    };
    let (ok, o) = run(&[]);
    assert!(ok, "{o}");
    assert!(o.contains("trace t.jsonl: 1 input path(s) present under --root, 1 to be created absent"), "{o}");
    assert_eq!(std::fs::metadata(out.join("new")).unwrap().len(), 10);
    assert!(!root.join("o").exists() && !root.join("d").exists());
    // the input at its endpoint, short of where the trace reads
    std::fs::write(data.join("f0"), b"short").unwrap();
    let (ok, o) = run(&["--clean-namespaces"]);
    assert!(!ok && o.contains("trace `t.jsonl`: d/f0 is 5 bytes, the trace reads to 100"), "{o}");
    // missing at its endpoint, though a file of that name is under --root
    std::fs::remove_file(data.join("f0")).unwrap();
    std::fs::create_dir_all(root.join("d")).unwrap();
    std::fs::write(root.join("d/f0"), vec![0u8; 100]).unwrap();
    let (ok, o) = run(&["--clean-namespaces"]);
    assert!(!ok && o.contains(&format!("trace `t.jsonl`: d/f0 ({}): No such file or directory", data.join("f0").display())), "{o}");
}
