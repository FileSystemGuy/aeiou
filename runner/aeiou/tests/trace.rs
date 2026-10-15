//! The `trace` node (`DESIGN_REVIEW.md` §3.58): a trace file is loaded against its sha256,
//! walked in line order by the dry run (lane and ordinal as the indices), run with one
//! thread per lane (the blocking backends) or one task per lane on an event loop under the
//! open table and path order, checked against `--root` before the gate, cleaned by
//! `--clean-namespaces`, and refused where the design says (V16). The fixtures are written
//! here, as `aeiou-trace export` writes them.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use aeiou::backend::BackendKind;
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};
use sha2::{Digest, Sha256};

static N: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aeiou-trace-test-{}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
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
        r#"{{"ast": "0.10", "name": "trace_test", "doc": "t", "params": {{}}, "datasets": {{}},
            "actors": {{"app": {{{count} "body": [{{"trace": {{"file": "t.jsonl", "sha256": "{}"}}}}]}}}}}}"#,
        sha(text.as_bytes())
    );
    let p = dir.join("t.ast.json");
    std::fs::write(&p, ast).unwrap();
    p
}

fn leaked(path: &Path, gpus: i64) -> (&'static aeiou::Loaded, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(path).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(Config { seed: 1, gpus, overrides: vec![], sets: vec![] }));
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
    std::fs::remove_dir_all(&d).unwrap();
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
    let m = dryrun::run_with(model, 1, None, Some(aeiou::metrics::Opts { block: 4096, sample: 1 })).unwrap();
    assert_eq!(m.total.fingerprint, r.total.fingerprint);
    let mm = m.total.metrics.as_ref().unwrap();
    assert_eq!(mm.fan_out.get(&3), Some(&1));
    assert!(mm.depth.is_empty());
    std::fs::remove_dir_all(&d).unwrap();
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
    std::fs::remove_dir_all(&d).unwrap();
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
    std::fs::remove_dir_all(&d).unwrap();
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
    std::fs::remove_dir_all(&d).unwrap();
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
    std::fs::remove_dir_all(&d).unwrap();
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
    std::fs::remove_dir_all(&d).unwrap();
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
    std::fs::remove_dir_all(&d).unwrap();
}
