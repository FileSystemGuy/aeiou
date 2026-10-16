//! The TCP coordinator: barriers across hosts (with departures), the configuration check in
//! `Hello`, a multi-host run whose merged fingerprint is the dry run's, and the binary end to
//! end: `dcp_save` on two ranks, then `dcp_load` on two ranks with `--rank-rotate 1`,
//! through the namespace manifest the coordinator's merged report produced. Every "host" is a
//! thread or a process on this machine; the wire is the same.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aeiou::backend::BackendKind;
use aeiou::coord::{Coordinator, Server, Tcp};
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};

fn tmpdir(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("coord", tag)
}

fn config(gpus: i64, seed: u64, params: &[(&str, &str)]) -> Config {
    Config { seed, gpus, overrides: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![] }
}

fn leaked_model(name: &str, cfg: Config) -> (&'static aeiou::Loaded, &'static Config, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&crate::common::workload_ast(name)).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    (loaded, cfg, model)
}

fn opts(root: &PathBuf, rank: i64, ranks: i64) -> RunOpts {
    RunOpts {
        root: root.clone(),
        endpoints: Default::default(),
        backend: BackendKind::Sync,
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
        rank,
        ranks,
        rank_rotate: 0,
        max_gap: None,
        require_cold: false,
        drop_caches: false,
    }
}

fn connect(server: &Server, rank: i64, ranks: i64, config: &str, participants: &[(String, usize)]) -> anyhow::Result<Arc<Tcp>> {
    Ok(Arc::new(Tcp::connect(&server.addr.to_string(), rank, ranks, &format!("host{rank}"), &serde_json::json!({"cfg": config}), &serde_json::json!({}), participants, Arc::new(AtomicBool::new(false)))?))
}

/// The manifest at `dir` as a datagen that notes progress every 30 s leaves it `ago` seconds
/// after its last note. The live check (refused under two windows, 60 s) then decides the
/// same however long killing the ranks or starting the next process takes, and however the
/// wall clock it compares is stepped meanwhile: a killed datagen's own windows of a few ms
/// are shorter than a loaded host's process start, and than the steps of a few seconds a
/// virtual machine's clock takes under load (WSL2 stepped back 2.4 s every 32 s, 2026-10-14),
/// back past a short sleep.
fn noted(dir: &Path, ago: f64) {
    let mut m = aeiou::payload::Manifest::read(dir).unwrap();
    m.progress["progress_secs"] = serde_json::json!(30.0);
    m.progress["updated"] = serde_json::json!(aeiou::datagen::now_f64() - ago);
    m.write(dir).unwrap();
}

#[test]
fn barriers_across_hosts_with_a_departure() {
    // rank 0 has two participants of `sync`, rank 1 one. Rank 0's second participant leaves
    // after two generations; the third generation on rank 0 is completed by its departure
    // (a departure release on that host), the server releases it once rank 1 arrives.
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let scope = vec![("sync".to_string(), 2usize)];
    let scope1 = vec![("sync".to_string(), 1usize)];
    let s0 = server.addr.to_string();
    let h0 = std::thread::spawn(move || {
        let t = Arc::new(Tcp::connect(&s0, 0, 2, "host0", &serde_json::json!({"cfg": "cfg"}), &serde_json::json!({}), &scope, Arc::new(AtomicBool::new(false))).unwrap());
        let (t0, hosts) = t.ready().unwrap();
        assert!(t0 > 0.0);
        assert_eq!(hosts, vec!["host0".to_string(), "host1".to_string()]);
        // B leaves only once A is waiting at the third generation, so that the departure is
        // what completes it (a departure before A arrives would leave nothing to release)
        let a_waiting = Arc::new(AtomicBool::new(false));
        let a = {
            let t = t.clone();
            let a_waiting = a_waiting.clone();
            std::thread::spawn(move || {
                let ab = AtomicBool::new(false);
                for i in 0..3 {
                    if i == 2 {
                        a_waiting.store(true, Ordering::Relaxed);
                    }
                    t.barrier("sync", &ab).unwrap();
                }
                t.leave("sync");
            })
        };
        let b = {
            let t = t.clone();
            std::thread::spawn(move || {
                let ab = AtomicBool::new(false);
                for _ in 0..2 {
                    t.barrier("sync", &ab).unwrap();
                }
                while !a_waiting.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
                t.leave("sync");
            })
        };
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(t.departure_releases(), vec![("sync".to_string(), 1)]);
        t
    });
    let s1 = server.addr.to_string();
    let h1 = std::thread::spawn(move || {
        let t = Tcp::connect(&s1, 1, 2, "host1", &serde_json::json!({"cfg": "cfg"}), &serde_json::json!({}), &scope1, Arc::new(AtomicBool::new(false))).unwrap();
        t.ready().unwrap();
        let ab = AtomicBool::new(false);
        for _ in 0..3 {
            t.barrier("sync", &ab).unwrap();
        }
        t.leave("sync");
        assert!(t.departure_releases().is_empty());
        t
    });
    let t0 = h0.join().unwrap();
    let t1 = h1.join().unwrap();
    // no host-level departure release: both hosts arrived at every generation
    assert!(server.departure_releases().is_empty());
    // the reduction round trip with empty reports
    let empty = run::Report::merge_all(vec![]);
    t0.report(&empty).unwrap();
    t1.report(&empty).unwrap();
    let merged = server.merged().unwrap();
    assert_eq!(merged.stats.ops, 0);
    server.finish(true, 42, None);
    assert_eq!(t1.result().unwrap(), (true, 42, None));
}

#[test]
fn configuration_mismatch_is_refused_before_the_start() {
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let t0 = connect(&server, 0, 2, "cfg-a", &[]).unwrap();
    let e = connect(&server, 1, 2, "cfg-b", &[]).err().expect("refused");
    assert!(format!("{e:#}").contains("cfg \"cfg-b\" differs from rank 0's \"cfg-a\""), "{e:#}");
    // rank 0 is told, and its start gate fails
    let e = t0.ready().unwrap_err();
    assert!(format!("{e:#}").contains("differs from rank 0"), "{e:#}");
}

#[test]
fn a_failing_host_stops_the_others() {
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let scope = vec![("sync".to_string(), 1usize)];
    let t0 = connect(&server, 0, 2, "cfg", &scope).unwrap();
    let t1 = connect(&server, 1, 2, "cfg", &scope).unwrap();
    let w = std::thread::spawn(move || {
        t0.ready().unwrap();
        let ab = AtomicBool::new(false);
        // nobody else arrives: the wait ends with the other host's reason
        let e = t0.barrier("sync", &ab).unwrap_err();
        (format!("{e:#}"), t0.abort_reason())
    });
    t1.ready().unwrap();
    t1.stop("disk on fire");
    let (e, reason) = w.join().unwrap();
    assert!(e.contains("disk on fire"), "{e}");
    assert!(reason.unwrap().contains("rank 1 (host1): disk on fire"));
    // and the server reports the failure to whoever asks for the reduction
    assert!(format!("{:#}", server.merged().err().expect("failed")).contains("disk on fire"));
}

/// A connection that sends no `Hello` (a port probe, a launcher's readiness check, a client
/// of something else) is no host: it is dropped and the run goes on, and the ranks that do
/// say `Hello` are the ones counted. Here, before the two hosts connect: one that stays open
/// and silent, one that closes at once, one that speaks HTTP, and one that sends a frame that
/// is no `Hello`.
#[test]
fn a_connection_that_sends_no_hello_is_no_host() {
    use std::io::Write;
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let silent = std::net::TcpStream::connect(server.addr).unwrap();
    drop(std::net::TcpStream::connect(server.addr).unwrap());
    std::net::TcpStream::connect(server.addr).unwrap().write_all(b"GET / HTTP/1.0\r\n\r\n").unwrap();
    std::net::TcpStream::connect(server.addr).unwrap().write_all(&[&7u32.to_be_bytes()[..], b"\"Ready\""].concat()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let (tx, rx) = std::sync::mpsc::channel();
    for rank in 0..2 {
        let (tx, addr) = (tx.clone(), server.addr.to_string());
        std::thread::spawn(move || {
            let r = Tcp::connect(&addr, rank, 2, &format!("host{rank}"), &serde_json::json!({"cfg": "cfg"}), &serde_json::json!({}), &[], Arc::new(AtomicBool::new(false))).and_then(|t| t.ready().map(|_| t));
            tx.send((rank, r.map(|_| ()).map_err(|e| format!("{e:#}")))).unwrap();
        });
    }
    for _ in 0..2 {
        let (rank, r) = rx.recv_timeout(std::time::Duration::from_secs(30)).expect("both hosts pass the start gate");
        assert!(r.is_ok(), "rank {rank}: {r:?}");
    }
    drop(silent);
}

/// A raw host: its `Hello` (the wire's JSON, written by hand), then its `Welcome` read.
fn raw_host(server: &Server, rank: i64, ranks: i64, cfg: &str, welcome: bool) -> std::net::TcpStream {
    use std::io::Write;
    let mut s = std::net::TcpStream::connect(server.addr).unwrap();
    let hello = serde_json::json!({"Hello": {"rank": rank, "ranks": ranks, "host": format!("raw{rank}"), "identity": {"cfg": cfg}, "layers": {}, "participants": [], "facts": null}}).to_string();
    s.write_all(&[&(hello.len() as u32).to_be_bytes()[..], hello.as_bytes()].concat()).unwrap();
    if welcome {
        read_welcome(&mut s);
    }
    s
}

fn read_welcome(s: &mut std::net::TcpStream) {
    use std::io::Read;
    s.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    let mut hdr = [0u8; 4];
    s.read_exact(&mut hdr).unwrap();
    let mut body = vec![0u8; u32::from_be_bytes(hdr) as usize];
    s.read_exact(&mut body).unwrap();
    assert!(String::from_utf8_lossy(&body).contains("Welcome"), "{}", String::from_utf8_lossy(&body));
}

/// What has already arrived on `s`, without waiting.
fn arrived(s: &mut std::net::TcpStream) -> String {
    use std::io::Read;
    s.set_nonblocking(true).unwrap();
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    while let Ok(n) = s.read(&mut buf) {
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A refusal reaches every host before rank 0 may exit. Rank 0's process ends as soon as its
/// own client hears the `Stop`, and once the server goes out of scope, every host's `Stop`
/// must already be on its socket: the refused host's (with the reason) and every registered
/// host's, rank 0's being the first of the broadcast. Repeated, since it is a race: rank 0 is
/// refused (another host's `Hello` came first), or a third host is. This thread, and so every
/// thread it starts (the server's, the client's reader), is held to one CPU, where the client
/// waking on its `Stop` preempts the server's broadcast: without the server taking its lock as
/// it goes out of scope, the first iteration fails.
#[test]
fn a_refusal_reaches_every_host_before_rank_0_can_exit() {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(libc::sched_getcpu().max(0) as usize, &mut set);
        assert_eq!(libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set), 0);
    }
    const K: i64 = 16;
    for i in 0..50 {
        // a third host refused: rank 0 hears the broadcast first, then the K others
        let server = Server::start("127.0.0.1:0", K + 2).unwrap();
        let t0 = connect(&server, 0, K + 2, "cfg", &[]).unwrap();
        // every Hello sent before any Welcome is read, so the accept loop takes them in one go
        let mut raws: Vec<_> = (1..=K).map(|r| raw_host(&server, r, K + 2, "cfg", false)).collect();
        raws.iter_mut().for_each(read_welcome);
        let mut refused = raw_host(&server, K + 1, K + 2, "cfg-b", false);
        assert!(format!("{:#}", t0.ready().unwrap_err()).contains("differs from rank 0"));
        drop(server);
        for (r, s) in raws.iter_mut().enumerate() {
            let got = arrived(s);
            assert!(got.contains("Stop") && got.contains("differs from rank 0"), "iteration {i}: rank {} had not been told when rank 0 could exit: {got:?}", r + 1);
        }
        drop(t0);
        // the refused host hears its reason before the close
        refused.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
        let mut got = Vec::new();
        let _ = std::io::Read::read_to_end(&mut refused, &mut got);
        assert!(String::from_utf8_lossy(&got).contains("cfg \\\"cfg-b\\\" differs from rank 0's \\\"cfg\\\""), "iteration {i}: {:?}", String::from_utf8_lossy(&got));

        // rank 0 refused: another host's Hello came first
        let server = Server::start("127.0.0.1:0", 2).unwrap();
        let mut raw = raw_host(&server, 1, 2, "cfg", true);
        let e = connect(&server, 0, 2, "cfg-b", &[]).err().expect("refused");
        assert!(format!("{e:#}").contains("differs from rank 1's"), "{e:#}");
        drop(server);
        let got = arrived(&mut raw);
        assert!(got.contains("Stop") && got.contains("differs from rank 1's"), "iteration {i}: rank 1 had not been told when rank 0 could exit: {got:?}");
    }
}

#[test]
fn two_hosts_reproduce_the_dry_run_fingerprint() {
    let root = tmpdir("two");
    let params = [("samples", "600"), ("batch", "4"), ("workers", "2"), ("prefetch", "2"), ("steps", "12"), ("walk", "true")];
    let (loaded, cfg, model) = leaked_model("file_per_sample", config(4, 7, &params));
    let dparams = Params::new(&loaded.ast, cfg).unwrap();
    let mut log = Vec::new();
    datagen(loaded, cfg, &dparams, model, &DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 4, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false }, None, &mut log).unwrap();
    let dry = dryrun::run(model, 2, None).unwrap();

    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let addr = server.addr.to_string();
    let mut hs = Vec::new();
    for rank in 0..2 {
        let addr = addr.clone();
        let o = opts(&root, rank, 2);
        hs.push(std::thread::spawn(move || {
            let p = run::participants(model, &o).unwrap();
            let aborted = Arc::new(AtomicBool::new(false));
            let t = Arc::new(Tcp::connect(&addr, rank, 2, &format!("host{rank}"), &serde_json::json!({"cfg": "cfg"}), &serde_json::json!({}), &p, aborted.clone()).unwrap());
            t.ready().unwrap();
            let r = run::run_with(model, o, HashMap::new(), t.clone(), aborted).unwrap();
            t.report(&r).unwrap();
            (t, r)
        }));
    }
    let parts: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    let merged = server.merged().unwrap();
    assert_eq!(merged.stats.fingerprint, dry.total.fingerprint);
    assert_eq!(merged.stats.ops, dry.total.total.ops);
    assert_eq!(merged.stats.bytes_read, dry.total.total.bytes_read);
    assert_eq!(merged.ranks.len(), 2);
    assert_eq!(merged.ranks[0].gpus, [0, 2]);
    assert_eq!(merged.ranks[1].gpus, [2, 4]);
    assert_eq!(merged.host, run::hostname(), "both ranks ran on this machine, so one host after dedup");
    assert_eq!(merged.actors.len(), 4);
    // each host's partial sum is not the fingerprint; their sum is
    let (p0, p1) = (parts[0].1.stats.fingerprint, parts[1].1.stats.fingerprint);
    assert_ne!(p0, dry.total.fingerprint);
    assert_eq!(p0.wrapping_add(p1), dry.total.fingerprint);
    server.finish(true, merged.stats.fingerprint, None);
    for (t, _) in &parts {
        assert_eq!(t.result().unwrap().1, dry.total.fingerprint);
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn dry_fingerprint_cli(ast: &str, gpus: &str, params: &[&str]) -> String {
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.arg("dry-run").arg(crate::common::workload_ast(ast)).arg("--gpus").arg(gpus);
    for p in params {
        c.arg("--param").arg(p);
    }
    let o = c.output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let text = String::from_utf8(o.stdout).unwrap();
    text.lines().find_map(|l| l.strip_prefix("fingerprint ")).unwrap().split_whitespace().next().unwrap().to_string()
}

/// `aeiou run` on `ranks` processes. Every rank writes a `--report-json` (rank 0 the merged
/// report, every other rank its own), so a rank other than 0 is given `FILE.rank<r>`: one
/// path for all would leave whichever rank finished last.
fn run_ranks(ast: &str, root: &PathBuf, gpus: &str, params: &[&str], ranks: i64, extra: &[&str]) -> Vec<(bool, String, String)> {
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..ranks {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("run").arg(crate::common::workload_ast(ast)).arg("--gpus").arg(gpus).arg("--root").arg(root).arg("--time-scale").arg("0");
        c.arg("--ranks").arg(ranks.to_string()).arg("--rank").arg(rank.to_string()).arg("--coordinator").arg(format!("127.0.0.1:{port}"));
        for p in params {
            c.arg("--param").arg(p);
        }
        let report = extra.iter().position(|a| *a == "--report-json").map(|i| i + 1);
        c.args(extra.iter().enumerate().map(|(i, a)| if Some(i) == report && rank > 0 { format!("{a}.rank{rank}") } else { a.to_string() }));
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
        if rank == 0 {
            // give the listener a moment; the clients retry anyway
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    children.into_iter().map(|c| {
        let o = c.wait_with_output().unwrap();
        (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    }).collect()
}

/// A host that owns no GPU (4 GPUs over 3 ranks: rank 2's range is [4, 4)) only hears
/// `Release`s, a step at a time, and sends nothing; it must still heartbeat (a beat is due
/// whenever its side has sent nothing for the heartbeat time, not only when it hears nothing),
/// or the server declares it dead and the run fails on every host. The coordinator's times
/// are scaled by `AEIOU_COORD_HEARTBEAT_MS` (heartbeat 100 ms, dead after 600 ms), and the
/// run lasts several of those.
#[test]
fn a_host_that_only_receives_still_heartbeats() {
    let root = tmpdir("oneway");
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(&root).args(["--param", "samples=500"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let port = free_port();
    let t = std::time::Instant::now();
    let children: Vec<_> = (0..3)
        .map(|rank| {
            let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
            c.arg("run").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--gpus").arg("4").arg("--root").arg(&root);
            c.args(["--param", "samples=500", "--param", "batch=4", "--param", "steps=30", "--param", "step_time=100000000"]);
            c.arg("--ranks").arg("3").arg("--rank").arg(rank.to_string()).arg("--coordinator").arg(format!("127.0.0.1:{port}"));
            c.env("AEIOU_COORD_HEARTBEAT_MS", "100");
            c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap()
        })
        .collect();
    let outs: Vec<_> = children.into_iter().map(|c| c.wait_with_output().unwrap()).collect();
    let took = t.elapsed();
    for (rank, o) in outs.iter().enumerate() {
        assert!(o.status.success(), "rank {rank}:\nstdout:\n{}\nstderr:\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    }
    assert!(String::from_utf8_lossy(&outs[2].stdout).contains("gpu ids [4, 4)"), "{}", String::from_utf8_lossy(&outs[2].stdout));
    assert!(took > std::time::Duration::from_millis(1800), "the run must outlast the 600 ms dead time several times over: {took:?}");
}

#[test]
fn checkpoint_write_and_restore_on_two_ranks_end_to_end() {
    let root = tmpdir("e2e");
    let wparams = ["steps=4", "ckpt_every=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "meta_bytes=65536"];
    let wfp = dry_fingerprint_cli("dcp_save.ast.json", "2", &wparams);
    let outs = run_ranks("dcp_save.ast.json", &root, "2", &wparams, 2, &["--expect-fingerprint", &wfp]);
    for (ok, stdout, stderr) in &outs {
        assert!(*ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
        assert!(stdout.contains("fingerprint matches"), "{stdout}");
        assert!(stdout.contains("start gate: 2 host(s) ready"), "{stdout}");
    }
    assert!(outs[0].1.contains("--- all 2 hosts"), "{}", outs[0].1);
    assert!(outs[0].1.contains("namespace manifest"), "{}", outs[0].1);
    assert!(!outs[1].1.contains("namespace manifest"), "rank 1 writes no manifest:\n{}", outs[1].1);
    let m = aeiou::payload::NamespaceManifest::read(&root.join("ckpt")).unwrap();
    assert_eq!(m.ranks.len(), 2);
    assert_eq!(m.ranks[0].gpus, [0, 1]);
    assert_eq!(m.ranks[1].gpus, [1, 2]);
    let objects = m.objects.as_ref().unwrap();
    assert_eq!(objects.len(), 2 * (1 + 2 + 1), "{objects:?}");
    assert!(objects.iter().any(|(p, g)| p == "ckpt/step_000002/__1_0.distcp" && *g == 1));

    // the restore on two ranks, rotated: each process reads the other's shard (same host
    // here, so the warm-open warning is expected and --require-cold would refuse)
    let rparams = ["restore_step=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "item_off=[0, 1050153, 3148882, 4199035]", "meta_bytes=65536"];
    let rfp = dry_fingerprint_cli("dcp_load.ast.json", "2", &rparams);
    let outs = run_ranks("dcp_load.ast.json", &root, "2", &rparams, 2, &["--rank-rotate", "1", "--max-gap", "120", "--expect-fingerprint", &rfp]);
    for (ok, stdout, stderr) in &outs {
        assert!(*ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
        assert!(stdout.contains("fingerprint matches"), "{stdout}");
    }
    assert!(outs[0].1.contains("gpu ids [1, 2)"), "{}", outs[0].1);
    assert!(outs[1].1.contains("gpu ids [0, 1)"), "{}", outs[1].1);
    assert!(outs[0].1.contains("input objects opened 4"), "{}", outs[0].1);

    // a differing configuration on one rank is refused on every rank before any I/O
    let outs = run_ranks_mixed(&root);
    assert!(outs.iter().all(|(ok, _, _)| !ok));
    assert!(outs.iter().any(|(_, _, e)| e.contains("seed 2 differs from rank 0's 1") || e.contains("seed 1 differs from rank 1's 2")), "{outs:?}");
}

/// Two ranks of `dcp_load` with different seeds.
fn run_ranks_mixed(root: &PathBuf) -> Vec<(bool, String, String)> {
    let port = free_port();
    let rparams = ["restore_step=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "item_off=[0, 1050153, 3148882, 4199035]", "meta_bytes=65536"];
    let mut children = Vec::new();
    for rank in 0..2 {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("run").arg(crate::common::workload_ast("dcp_load.ast.json")).arg("--gpus").arg("2").arg("--root").arg(root).arg("--time-scale").arg("0");
        c.arg("--ranks").arg("2").arg("--rank").arg(rank.to_string()).arg("--coordinator").arg(format!("127.0.0.1:{port}")).arg("--seed").arg((rank + 1).to_string());
        for p in &rparams {
            c.arg("--param").arg(p);
        }
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
    }
    children.into_iter().map(|c| {
        let o = c.wait_with_output().unwrap();
        (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    }).collect()
}

// ---------------------------------------------------------------- datagen on several hosts

fn dopts(root: &PathBuf, rank: i64, ranks: i64) -> DatagenOpts {
    DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 3, dedupe: 1, compress: 1, datasets: vec![], rank, ranks, progress: std::time::Duration::from_millis(20), resume: None, resize: None, settle: None, sigint: false, ignore_space: false }
}

/// Every data file under `root` (the `.aeiou*` metadata left out), by relative path.
fn tree(root: &PathBuf) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut std::collections::BTreeMap<String, Vec<u8>>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if !p.file_name().unwrap().to_string_lossy().starts_with(".aeiou") {
                out.insert(p.strip_prefix(root).unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap());
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(root, root, &mut out);
    out
}

#[test]
fn datagen_on_two_ranks_writes_the_one_host_corpus() {
    // a files dataset (120 files, one directory: both ranks create in it) and a corpus with a
    // array file above one 64 MiB piece plus a one-file dataset (rank 1's slice is empty)
    for (ast, gpus, params) in [("file_per_sample", 2, vec![("samples", "120")]), ("faiss_ivf_ondisk", 1, vec![("lists", "280"), ("index_bytes", "1048576")])] {
        let one = tmpdir("dg-one");
        let two = tmpdir("dg-two");
        let (loaded, cfg, model) = leaked_model(ast, config(gpus, 7, &params));
        let dparams = Params::new(&loaded.ast, cfg).unwrap();
        let mut log = Vec::new();
        let alone = aeiou::datagen::datagen(loaded, cfg, &dparams, model, &dopts(&one, 0, 1), None, &mut log).unwrap();
        assert!(alone.iter().all(|r| r.wrote_manifest && r.owner.is_none()));

        let server = Server::start("127.0.0.1:0", 2).unwrap();
        let addr = server.addr.to_string();
        let results: Vec<Vec<aeiou::datagen::DatasetResult>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..2)
                .map(|rank| {
                    let (addr, two, server) = (addr.clone(), two.clone(), &server);
                    s.spawn(move || {
                        let t = Tcp::connect(&addr, rank, 2, &format!("host{rank}"), &serde_json::json!({"cfg": "dg"}), &serde_json::json!({}), &[], Arc::new(AtomicBool::new(false))).unwrap();
                        let hosts = aeiou::datagen::Hosts { tcp: &t, server: if rank == 0 { Some(server) } else { None } };
                        let dparams = Params::new(&loaded.ast, cfg).unwrap();
                        let mut log = Vec::new();
                        let r = aeiou::datagen::datagen(loaded, cfg, &dparams, model, &dopts(&two, rank, 2), Some(&hosts), &mut log).unwrap();
                        let text = String::from_utf8(log).unwrap();
                        assert!(text.contains("start gate: 2 host(s) ready (host0, host1)"), "{text}");
                        r
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (r0, r1) = (&results[0], &results[1]);
        assert_eq!(r0.len(), alone.len());
        for (i, a) in alone.iter().enumerate() {
            assert_eq!(r0[i].id, a.id, "{ast}: the dataset id is the definition's, not the hosts'");
            assert_eq!(r1[i].id, a.id);
            assert!(r0[i].wrote_manifest && !r1[i].wrote_manifest);
            assert_eq!(r0[i].total_files, a.files, "{ast} {}", a.name);
            assert_eq!(r0[i].total_bytes, a.bytes);
            assert_eq!(r0[i].files + r1[i].files, a.files, "{ast} {}: the slices cover the ids once", a.name);
            assert_eq!(r0[i].bytes + r1[i].bytes, a.bytes);
            let m = aeiou::payload::Manifest::read(&two.join(&a.root.strip_prefix(&one).unwrap())).unwrap();
            assert_eq!(m.provenance["ranks"], 2);
            assert_eq!(m.provenance["hosts"].as_array().unwrap().len(), 2, "{}", m.provenance);
            assert_eq!(m.provenance["hosts"][1]["rank"], 1);
            assert_eq!(m.provenance["files_written"].as_u64().unwrap(), a.files);
            assert_eq!(m.id(), a.id);
        }
        if ast == "faiss_ivf_ondisk" {
            let lists = r0.iter().position(|r| r.name == "lists").unwrap();
            assert!(r0[lists].owner.is_none() && r1[lists].owner == Some(0), "the first array dataset is rank 0's");
            assert_eq!(r1[lists].files, 0);
            let index = r0.iter().position(|r| r.name == "index").unwrap();
            assert_eq!(r0[index].files + r1[index].files, 1, "one file, written by the rank that claimed it");
            assert!(std::fs::metadata(two.join("ivf/merged_index.ivfdata")).unwrap().len() > 64 << 20, "several pieces");
        }
        assert_eq!(tree(&one), tree(&two), "{ast}: byte for byte the same corpus");
        // a run accepts the corpus two hosts wrote
        run::check_datasets(loaded, cfg, &two).unwrap();
    }
}

fn datagen_ranks(root: &PathBuf, params: &[&str], ranks: i64, extra: impl Fn(i64) -> Vec<String>) -> Vec<(bool, String, String)> {
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..ranks {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(root).arg("--writers").arg("2");
        c.arg("--ranks").arg(ranks.to_string()).arg("--rank").arg(rank.to_string()).arg("--coordinator").arg(format!("127.0.0.1:{port}"));
        for p in params {
            c.arg("--param").arg(p);
        }
        c.args(extra(rank));
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
        if rank == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    children.into_iter().map(|c| {
        let o = c.wait_with_output().unwrap();
        (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    }).collect()
}

#[test]
fn datagen_on_two_processes_end_to_end() {
    let root = tmpdir("dg-e2e");
    let outs = datagen_ranks(&root, &["samples=100"], 2, |_| vec!["--progress-secs".into(), "0.01".into()]);
    for (ok, stdout, stderr) in &outs {
        assert!(*ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
        assert!(stdout.contains("start gate: 2 host(s) ready"), "{stdout}");
    }
    // rank 0 prints the progress of both, the other rank none
    assert!(outs[0].1.contains("progress ") && outs[0].1.contains("/100 files"), "{}", outs[0].1);
    assert!(!outs[1].1.contains("progress "), "{}", outs[1].1);
    // one directory of 100 ids, cut in four for the four writers; rank 0 hands the runs out
    assert!(outs[0].1.contains("dataset train: 100 id(s) of 100, in 4 run(s) handed out by this rank in order"), "{}", outs[0].1);
    assert!(outs[1].1.contains("dataset train: 100 id(s) of 100, in runs claimed from rank 0"), "{}", outs[1].1);
    assert!(outs[0].1.contains("all 2 hosts 100 file(s)") && outs[0].1.contains("manifest written"), "{}", outs[0].1);
    assert!(outs[1].1.contains("(manifest by rank 0)"), "{}", outs[1].1);
    let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert_eq!(m.provenance["ranks"], 2);
    assert_eq!(m.provenance["files_written"], 100);
    assert_eq!((m.state.as_str(), &m.progress), ("complete", &serde_json::Value::Null));
    // a differing payload on one rank is refused on every rank before anything is written
    let root2 = tmpdir("dg-mixed");
    let outs = datagen_ranks(&root2, &["samples=100"], 2, |rank| if rank == 1 { vec!["--dedupe".into(), "2".into()] } else { vec![] });
    assert!(outs.iter().all(|(ok, _, _)| !ok), "{outs:?}");
    assert!(outs.iter().any(|(_, _, e)| e.contains("dedupe 2 differs from rank 0's 1") || e.contains("dedupe 1 differs from rank 1's 2")), "{outs:?}");
    assert!(aeiou::payload::Manifest::read(&root2.join("train")).is_err(), "no manifest");
    assert!(tree(&root2).is_empty(), "nothing written: {:?}", tree(&root2).keys().collect::<Vec<_>>());
}

/// A datagen killed in the middle, on one host or two, leaves its manifest `writing`, and
/// every id the manifest does not list as unfinished is a whole file (DESIGN_REVIEW.md
/// §3.70: each rank notes, flushes, and tells rank 0, which merges); a run refuses it.
#[test]
fn a_killed_datagen_leaves_a_manifest_that_is_true() {
    use std::io::BufRead;
    let files = 20000i64;
    let (loaded, cfg, model) = leaked_model("file_per_sample.ast.json", config(1, 1, &[("samples", "20000")]));
    let meta = &model.datasets[0];
    for ranks in [1i64, 2] {
        let root = tmpdir("dg-killed");
        let port = free_port();
        let mut children = Vec::new();
        for rank in 0..ranks {
            let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
            c.arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(&root);
            c.args(["--writers", "2", "--progress-secs", "0.02", "--param", &format!("samples={files}")]);
            if ranks > 1 {
                c.args(["--ranks", &ranks.to_string(), "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
            }
            children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).spawn().unwrap());
            if rank == 0 {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
        // rank 0's lines: the first window that has seen a file finish, then two more
        let mut lines = std::io::BufReader::new(children[0].stdout.take().unwrap()).lines();
        let mut seen = 0;
        while let Some(Ok(l)) = lines.next() {
            if l.starts_with("progress ") && !l.contains(" 0/") {
                seen += 1;
                if seen == 3 {
                    break;
                }
            }
        }
        assert_eq!(seen, 3, "{ranks} rank(s): the datagen ended before three windows");
        for c in children.iter_mut() {
            c.kill().unwrap();
            c.wait().unwrap();
        }
        let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
        assert_eq!(m.state, "writing");
        let unfinished: Vec<(i64, i64)> = m.progress["unfinished"].as_array().unwrap().iter().map(|r| (r[0].as_i64().unwrap(), r[1].as_i64().unwrap())).collect();
        let left: i64 = unfinished.iter().map(|(a, b)| b - a).sum();
        assert!(left > 0 && left < files, "{ranks} rank(s): {unfinished:?}");
        // the claim order (DESIGN_REVIEW.md §3.70): below the boundary only the band of runs in
        // flight is unfinished (a directory of 1,300 for each of the writers on every host, and
        // one claimed ahead by each writer on another host than rank 0), and nothing was
        // written at or past the mark
        let (claimed, mark) = (m.progress["claimed"].as_i64().unwrap(), m.progress["claimed_to"].as_i64().unwrap());
        assert!(0 < claimed && claimed <= mark && mark <= files, "{}", m.progress);
        let band: i64 = unfinished.iter().map(|&(a, b)| (b.min(claimed) - a).max(0)).sum();
        assert!(band <= (ranks * 2 + (ranks - 1) * 2) * 1300, "{ranks} rank(s): {band} unfinished below {claimed}: {unfinished:?}");
        assert!(unfinished.last().unwrap().1 == files && unfinished.last().unwrap().0 <= claimed, "{unfinished:?}");
        for id in mark..files {
            assert!(!root.join(meta.file_path(id, None).unwrap().to_string()).exists(), "id {id} written past the mark {mark}");
        }
        for id in 0..files {
            if unfinished.iter().any(|(a, b)| (*a..*b).contains(&id)) {
                continue;
            }
            let path = root.join(meta.file_path(id, None).unwrap().to_string());
            assert_eq!(std::fs::metadata(&path).map(|m| m.len()).ok(), Some(meta.file_size(id).unwrap() as u64), "{}: recorded as finished", path.display());
        }
        let e = run::check_datasets(loaded, cfg, &root).unwrap_err().to_string();
        assert!(e.contains("the manifest says `writing`") && e.contains(&format!("{left} file(s) not written")), "{e}");

        // the resume, on the other host count: a stale file longer than its size is cut back
        let first = root.join(meta.file_path(unfinished[0].0, None).unwrap().to_string());
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&first).unwrap();
        std::io::Write::write_all(&mut f, &vec![7u8; 300_000]).unwrap();
        drop(f);
        // a manifest younger than two of the windows its datagen recorded may still have a
        // datagen writing it
        assert_eq!(m.progress["progress_secs"].as_f64(), Some(0.02), "{}", m.progress);
        for ago in [0.0, 40.0] {
            noted(&root.join("train"), ago);
            let outs = resume_ranks(&root.join("train"), 1);
            assert!(!outs[0].0 && outs[0].2.contains("progress every 30 s, so it may still be writing; stop it, or wait until 60 s"), "{ago} s: {outs:?}");
        }
        noted(&root.join("train"), 90.0);
        let outs = resume_ranks(&root.join("train"), 3 - ranks);
        assert!(outs.iter().all(|(ok, _, _)| *ok), "{ranks} rank(s) killed: {outs:?}");
        assert!(outs[0].1.contains(&format!("{left} file(s) unfinished")), "{}", outs[0].1);
        let after = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
        assert_eq!((after.state.as_str(), after.id()), ("complete", m.id()));
        assert_eq!(after.provenance["files_written"].as_i64(), Some(files), "every file counted once");
        assert_eq!(after.provenance["resumes"][0]["ranks"].as_i64(), Some(3 - ranks));
        run::check_datasets(loaded, cfg, &root).unwrap();
        // the files written again are the payload's (the first few of every range)
        let mut filler = aeiou::payload::Filler::new(1);
        for (a, b) in &unfinished {
            for id in *a..(*b).min(a + 20) {
                let path = root.join(meta.file_path(id, None).unwrap().to_string());
                let got = std::fs::read(&path).unwrap();
                let mut want = vec![0u8; meta.file_size(id).unwrap() as usize];
                let seed = meta.dataset_seed();
                filler.fill_range(|blk| aeiou::payload::file_block_seed(seed, id as u64, blk, 1), 0, &mut want);
                assert!(got == want, "{}: not the payload", path.display());
            }
        }
    }
}

/// `aeiou datagen --resume ROOT` on `ranks` processes.
fn resume_ranks(at: &std::path::Path, ranks: i64) -> Vec<(bool, String, String)> {
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..ranks {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("datagen").arg("--resume").arg(at).args(["--writers", "2"]);
        if ranks > 1 {
            c.args(["--ranks", &ranks.to_string(), "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
        }
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
    }
    children.into_iter().map(|c| {
        let o = c.wait_with_output().unwrap();
        (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    }).collect()
}

/// `--resume` takes everything from the manifest: what it fixes is refused beside it, and a
/// complete corpus has nothing to write.
#[test]
fn a_resume_takes_nothing_the_manifest_fixes() {
    let root = tmpdir("dg-resume-usage");
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--resume"]).arg(root.join("train")).args(["--param", "samples=10", "--dedupe", "2"]).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(e.contains("--param, --dedupe given too"), "{e}");
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(&root).args(["--param", "samples=30"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--resume"]).arg(root.join("train")).output().unwrap();
    assert!(o.status.success() && String::from_utf8_lossy(&o.stdout).contains("the manifest says `complete`; nothing to write"), "{o:?}");
}

/// `aeiou datagen --grow N ROOT` or `--shrink N ROOT` (`op`) on `ranks` processes, with `extra`.
fn resize_ranks(op: &str, n: i64, at: &std::path::Path, ranks: i64, extra: &[&str]) -> Vec<(bool, String, String)> {
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..ranks {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("datagen").arg(op).arg(n.to_string()).arg(at).args(["--writers", "2"]).args(extra);
        if ranks > 1 {
            c.args(["--ranks", &ranks.to_string(), "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
        }
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
    }
    children.into_iter().map(|c| {
        let o = c.wait_with_output().unwrap();
        (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    }).collect()
}

fn datagen_one(root: &PathBuf, files: i64) {
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(root).args(["--writers", "3", "--param", &format!("samples={files}")]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

/// A grow and a shrink (DESIGN_REVIEW.md §3.70), on one host and on two, leave the corpus a
/// datagen of the new count writes: the same files, byte for byte, the directories a shrink
/// empties gone, the same id and totals, the change recorded, and `run` accepts it.
#[test]
fn grow_and_shrink_make_the_corpus_of_the_new_count() {
    // 1300 files a directory: 3000 has three, 5000 four, 2100 two
    let fresh5 = tmpdir("dg-fresh5");
    let fresh21 = tmpdir("dg-fresh21");
    datagen_one(&fresh5, 5000);
    datagen_one(&fresh21, 2100);
    let id = |r: &PathBuf| aeiou::payload::Manifest::read(&r.join("train")).unwrap();
    for ranks in [1i64, 2] {
        let root = tmpdir("dg-resize");
        datagen_one(&root, 3000);
        let was = id(&root).id();
        let outs = resize_ranks("--grow", 5000, &root.join("train"), ranks, &[]);
        assert!(outs.iter().all(|(ok, _, _)| *ok), "{ranks} rank(s): {outs:?}");
        assert!(outs[0].1.contains("count 3000 -> 5000: 2000 file(s) to write"), "{}", outs[0].1);
        assert!(tree(&root) == tree(&fresh5), "{ranks} rank(s): the grown corpus is not the 5000-file one");
        let (m, f) = (id(&root), id(&fresh5));
        assert_eq!((m.state.as_str(), m.id()), ("complete", f.id()));
        assert_eq!((&m.provenance["files_written"], &m.provenance["bytes_written"]), (&f.provenance["files_written"], &f.provenance["bytes_written"]));
        let r = &m.provenance["resizes"][0];
        assert_eq!((r["op"].as_str(), r["from"].as_i64(), r["to"].as_i64(), r["from_id"].as_str(), r["files_written"].as_i64(), r["ranks"].as_i64()), (Some("grow"), Some(3000), Some(5000), Some(was.as_str()), Some(2000), Some(ranks)));
        let (loaded, cfg, _) = leaked_model("file_per_sample.ast.json", config(1, 1, &[("samples", "5000")]));
        run::check_datasets(loaded, cfg, &root).unwrap();

        let outs = resize_ranks("--shrink", 2100, &root.join("train"), 3 - ranks, &[]);
        assert!(outs.iter().all(|(ok, _, _)| *ok), "{ranks} rank(s): {outs:?}");
        // the ids handed out, and what becomes of them (it said "2900 id(s) of 2100")
        assert!(outs[0].1.contains("dataset train: the ids from 2100 to 4999 of 5000 (those from 2100 up removed), in "), "{}", outs[0].1);
        assert!(tree(&root) == tree(&fresh21), "{ranks} rank(s): the shrunk corpus is not the 2100-file one");
        assert!(!root.join("train/00002").exists() && !root.join("train/00003").exists(), "the emptied directories are removed");
        let (m, f) = (id(&root), id(&fresh21));
        assert_eq!((m.state.as_str(), m.id()), ("complete", f.id()));
        assert_eq!((&m.provenance["files_written"], &m.provenance["bytes_written"]), (&f.provenance["files_written"], &f.provenance["bytes_written"]));
        assert_eq!(m.provenance["resizes"][1]["op"], "shrink");
        let (loaded, cfg, _) = leaked_model("file_per_sample.ast.json", config(1, 1, &[("samples", "2100")]));
        run::check_datasets(loaded, cfg, &root).unwrap();
    }
}

/// A grow killed in the middle leaves the manifest `growing`, naming the new count and the
/// old one; `run` refuses it, a second grow is refused, and `--resume` finishes it on another
/// host count to the corpus of the new count.
#[test]
fn a_killed_grow_is_finished_by_resume() {
    use std::io::BufRead;
    let (from, to) = (2000i64, 20000i64);
    let root = tmpdir("dg-grow-killed");
    datagen_one(&root, from);
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.args(["datagen", "--grow", &to.to_string()]).arg(root.join("train")).args(["--writers", "2", "--progress-secs", "0.02"]);
    let mut child = c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).spawn().unwrap();
    let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let mut seen = 0;
    while let Some(Ok(l)) = lines.next() {
        if l.starts_with("progress ") && !l.contains(&format!(" {from}/")) {
            seen += 1;
            if seen == 3 {
                break;
            }
        }
    }
    assert_eq!(seen, 3, "the grow ended before three windows");
    child.kill().unwrap();
    child.wait().unwrap();
    let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert_eq!((m.state.as_str(), m.progress["from"].as_i64(), m.dataset["count"].as_i64()), ("growing", Some(from), Some(to)));
    let (loaded, cfg, model) = leaked_model("file_per_sample.ast.json", config(1, 1, &[("samples", "20000")]));
    let e = run::check_datasets(loaded, cfg, &root).unwrap_err().to_string();
    assert!(e.contains("the manifest says `growing`") && e.contains("--resume"), "{e}");
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--grow", "30000"]).arg(root.join("train")).output().unwrap();
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("finish it first with `aeiou datagen --resume"), "{o:?}");

    noted(&root.join("train"), 90.0);
    let outs = resume_ranks(&root.join("train"), 2);
    assert!(outs.iter().all(|(ok, _, _)| *ok), "{outs:?}");
    assert!(outs[0].1.contains(&format!("(growing from count {from})")), "{}", outs[0].1);
    let after = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert_eq!((after.state.as_str(), after.id()), ("complete", m.id()));
    assert_eq!(after.provenance["files_written"].as_i64(), Some(to), "every file counted once");
    run::check_datasets(loaded, cfg, &root).unwrap();
    let meta = &model.datasets[0];
    let files = tree(&root);
    assert_eq!(files.len() as i64, to + 1, "the files and the inventory");
    assert!(inventory_is_payload(&root, meta, to), "the grown corpus's inventory");
    for id in (0..to).step_by(997) {
        let mut want = vec![0u8; meta.file_size(id).unwrap() as usize];
        let seed = meta.dataset_seed();
        aeiou::payload::Filler::new(1).fill_range(|blk| aeiou::payload::file_block_seed(seed, id as u64, blk, 1), 0, &mut want);
        assert!(files[&meta.file_path(id, None).unwrap().to_string()] == want, "id {id}: not the payload");
    }
}

/// What a grow or shrink refuses: a count that does not grow or shrink, one of zero, anything
/// the manifest fixes, two operations at once.
#[test]
fn grow_and_shrink_refuse_what_they_cannot_do() {
    let root = tmpdir("dg-resize-usage");
    datagen_one(&root, 30);
    let at = root.join("train");
    let refuse = |args: &[&str], want: &str| {
        let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").args(args).output().unwrap();
        let e = String::from_utf8_lossy(&o.stderr);
        assert!(o.status.code() == Some(2) && e.contains(want), "{args:?}: {e}");
    };
    let a = at.to_str().unwrap();
    refuse(&["--grow", "30", a], "the dataset's count is 30; a grow takes a larger one");
    refuse(&["--shrink", "31", a], "the dataset's count is 30; a shrink takes a smaller one");
    refuse(&["--shrink", "0", a], "a whole number above 0");
    refuse(&["--grow", "40", a, "--param", "samples=40"], "--grow takes the dataset, its place, the payload, and the format from the manifest at its root: --param given too");
    refuse(&["--grow", "40", a, "--resume", a], "--resume, --grow: one at a time");
    assert_eq!(tree(&root).len(), 31, "nothing changed (30 files and the inventory)");
}

/// At a dataset root on several hosts, rank 0 alone reads the manifest, and what it decides
/// reaches every host with `Welcome`: its refusal, or a complete corpus's nothing to do. Every
/// host names the same root.
#[test]
fn only_rank_0_reads_the_manifest_at_the_root() {
    let root = tmpdir("dg-rank0-manifest");
    datagen_one(&root, 30);
    let at = root.join("train");
    for (rank, (ok, _, e)) in resize_ranks("--grow", 30, &at, 2, &[]).iter().enumerate() {
        assert!(!ok && e.contains("the dataset's count is 30; a grow takes a larger one"), "rank {rank}: {e}");
    }
    for (rank, (ok, out, e)) in resume_ranks(&at, 2).iter().enumerate() {
        assert!(*ok && out.contains("the manifest says `complete`; nothing to write"), "rank {rank}: {out}{e}");
    }
    let port = free_port();
    let children: Vec<_> = [at.clone(), root.join("elsewhere")]
        .iter()
        .enumerate()
        .map(|(rank, at)| {
            let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
            c.arg("datagen").arg("--grow").arg("40").arg(at).args(["--ranks", "2", "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
            c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap()
        })
        .collect();
    for (rank, c) in children.into_iter().enumerate() {
        let o = c.wait_with_output().unwrap();
        let e = String::from_utf8_lossy(&o.stderr);
        assert!(!o.status.success() && e.contains("not the same run") && e.contains("root"), "rank {rank}: {e}");
    }
    assert_eq!(tree(&root).len(), 31, "nothing changed (30 files and the inventory)");
}

// ---------------------------------------------------------------- settle and Ctrl-C

/// Data files under `root` with their lengths (the `.aeiou*` metadata left out).
fn listing(root: &std::path::Path) -> std::collections::BTreeMap<String, u64> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut std::collections::BTreeMap<String, u64>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if !p.file_name().unwrap().to_string_lossy().starts_with(".aeiou") {
                out.insert(p.strip_prefix(root).unwrap().to_string_lossy().into_owned(), std::fs::metadata(&p).unwrap().len());
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// What a datagen of `count` writes under the dataset root: every name with its size, and the
/// inventory at that count (DESIGN_REVIEW §3.73).
fn expected(meta: &aeiou::eval::DsMeta<'_>, count: i64) -> std::collections::BTreeMap<String, u64> {
    let m = meta.with_count(count);
    let mut out: std::collections::BTreeMap<String, u64> = (0..m.files()).map(|id| (m.file_path(id, None).unwrap().to_string().trim_start_matches("train/").to_string(), m.file_size(id).unwrap() as u64)).collect();
    if let Some((path, size, _)) = m.inventory() {
        out.insert(path.trim_start_matches("train/").to_string(), size as u64);
    }
    out
}

/// The inventory under `root` holds the payload's bytes at `count` (its own label, §3.73).
fn inventory_is_payload(root: &std::path::Path, meta: &aeiou::eval::DsMeta<'_>, count: i64) -> bool {
    let (path, size, _) = meta.with_count(count).inventory().unwrap();
    let got = std::fs::read(root.join(path.to_string())).unwrap_or_default();
    let mut want = vec![0u8; size as usize];
    let seed = meta.dataset_seed();
    aeiou::payload::Filler::new(1).fill_range(|blk| aeiou::payload::inventory_block_seed(seed, blk, 1), 0, &mut want);
    got == want
}

/// File `id` under `root` holds the payload's bytes.
fn is_payload(root: &std::path::Path, meta: &aeiou::eval::DsMeta<'_>, id: i64) -> bool {
    let got = std::fs::read(root.join(meta.file_path(id, None).unwrap().to_string())).unwrap_or_default();
    let mut want = vec![0u8; meta.file_size(id).unwrap() as usize];
    let seed = meta.dataset_seed();
    aeiou::payload::Filler::new(1).fill_range(|blk| aeiou::payload::file_block_seed(seed, id as u64, blk, 1), 0, &mut want);
    got == want
}

/// `aeiou datagen ARGS` on `ranks` processes, started (rank 0 first), output piped.
fn spawn_ranks(args: &[String], ranks: i64) -> Vec<std::process::Child> {
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..ranks {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("datagen").args(args);
        if ranks > 1 {
            c.args(["--ranks", &ranks.to_string(), "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
        }
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
        if rank == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    children
}

/// Read rank 0's lines until `n` progress windows that `moved` accepts; the reader is
/// returned for the rest.
fn windows(child: &mut std::process::Child, n: usize, moved: impl Fn(&str) -> bool) -> std::io::Lines<std::io::BufReader<std::process::ChildStdout>> {
    use std::io::BufRead;
    let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let mut seen = 0;
    while let Some(Ok(l)) = lines.next() {
        if l.starts_with("progress ") && moved(&l) {
            seen += 1;
            if seen == n {
                return lines;
            }
        }
    }
    panic!("the datagen ended before {n} windows");
}

fn settle(at: &std::path::Path) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--settle"]).arg(at).args(["--writers", "2"]).output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// The settled corpus is what a datagen of its count writes: the names and sizes, the id a
/// run accepts, the totals; the ids the settle finished or wrote again, and a sample of the
/// rest, are the payload's. Returns the count.
fn check_settled(root: &PathBuf, was: &aeiou::payload::Manifest, asked: i64) -> i64 {
    let after = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert_eq!(after.state, "complete");
    let s = after.dataset["count"].as_i64().unwrap();
    let (loaded, cfg, model) = leaked_model("file_per_sample.ast.json", config(1, 1, &[("samples", &s.to_string())]));
    let meta = &model.datasets[0];
    assert!(listing(&root.join("train")) == expected(meta, s), "the settled corpus is not the {s}-file one");
    run::check_datasets(loaded, cfg, root).unwrap();
    assert_eq!(after.provenance["files_written"].as_i64(), Some(s), "{}", after.provenance);
    let inventory = meta.with_count(s).inventory().map_or(0, |(_, n, _)| n as u64);
    assert_eq!(after.provenance["bytes_written"].as_u64(), Some(expected(meta, s).values().sum::<u64>() - inventory), "the files' bytes, the inventory not counted");
    assert!(inventory_is_payload(root, meta, s), "the settled corpus's inventory");
    let st = after.provenance["settles"].as_array().unwrap().last().unwrap();
    assert_eq!((st["asked"].as_i64(), st["count"].as_i64(), st["state"].as_str()), (Some(asked), Some(s), Some(was.state.as_str())));
    let mut ids: Vec<i64> = (0..s).step_by(991).collect();
    for (a, b) in was.unfinished() {
        ids.extend((a..b.min(s)).take(30));
    }
    for id in ids {
        assert!(is_payload(root, meta, id), "id {id}: not the payload");
    }
    s
}

/// `--settle` (DESIGN_REVIEW.md §3.70) keeps what a killed datagen or grow reached: the count
/// is the boundary of the runs handed out, a directory boundary, and the corpus is a fresh
/// datagen's of that count.
#[test]
fn a_killed_datagen_or_grow_settles_at_whole_directories() {
    let files = 20000i64;
    for (op, ranks) in [("datagen", 2i64), ("grow", 1)] {
        let root = tmpdir("dg-settle");
        let from = 2000i64;
        let args: Vec<String> = match op {
            "datagen" => vec![crate::common::workload_ast("file_per_sample.ast.json").to_string_lossy().into_owned(), "--root".into(), root.to_string_lossy().into_owned(), "--param".into(), format!("samples={files}")],
            _ => {
                datagen_one(&root, from);
                vec!["--grow".into(), files.to_string(), root.join("train").to_string_lossy().into_owned()]
            }
        };
        let mut args = args;
        args.extend(["--writers", "2", "--progress-secs", "0.02"].map(String::from));
        let mut children = spawn_ranks(&args, ranks);
        let start = if op == "datagen" { " 0/".to_string() } else { format!(" {from}/") };
        let _rest = windows(&mut children[0], 2, |l| !l.contains(&start));
        for c in children.iter_mut() {
            c.kill().unwrap();
            c.wait().unwrap();
        }
        let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
        assert_ne!(m.state, "complete");
        let claimed = m.progress["claimed"].as_i64().unwrap();
        assert_eq!(m.progress["progress_secs"].as_f64(), Some(0.02), "{}", m.progress);
        for ago in [0.0, 40.0] {
            noted(&root.join("train"), ago);
            let o = settle(&root.join("train"));
            assert!(!o.0 && o.2.contains("progress every 30 s, so it may still be writing; stop it, or wait until 60 s"), "{ago} s: {o:?}");
        }
        noted(&root.join("train"), 90.0);
        let o = settle(&root.join("train"));
        assert!(o.0, "{op}: {o:?}");
        assert!(o.1.contains(&format!("of the {files} asked")), "{}", o.1);
        let s = check_settled(&root, &m, files);
        let low = if op == "grow" { from } else { 0 };
        assert_eq!(s, low.max(claimed.min(files)), "{op}: settled at the boundary {claimed}");
        assert!(s % 1300 == 0 || s == files || s == from, "{op}: {s} is not a directory boundary");
        if op == "grow" {
            let r = &m.provenance["resizes"][0];
            let after = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
            assert_eq!((r["to"].as_i64(), after.provenance["resizes"][0]["settled"].as_i64()), (Some(files), Some(s)));
        }
        // a complete corpus has nothing to settle
        let o = settle(&root.join("train"));
        assert!(o.0 && o.1.contains("nothing to settle"), "{o:?}");
    }
}

/// Settling a killed shrink finishes only the removals it had in hand: it never continues
/// the shrink, and what it keeps is a fresh datagen's corpus of the count it reached.
#[test]
fn a_killed_shrink_settles_without_removing_more() {
    let (files, to) = (20000i64, 100i64);
    let root = tmpdir("dg-settle-shrink");
    datagen_one(&root, files);
    let args: Vec<String> = vec!["--shrink".into(), to.to_string(), root.join("train").to_string_lossy().into_owned(), "--writers".into(), "1".into(), "--progress-secs".into(), "0.01".into()];
    let mut children = spawn_ranks(&args, 1);
    let _rest = windows(&mut children[0], 2, |l| !l.contains(" 0/"));
    children[0].kill().unwrap();
    children[0].wait().unwrap();
    let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert_eq!(m.state, "shrinking", "the shrink ended before it was killed");
    let there = listing(&root.join("train")).len() as i64;
    noted(&root.join("train"), 90.0);
    let o = settle(&root.join("train"));
    assert!(o.0, "{o:?}");
    let s = check_settled(&root, &m, to);
    let claimed = m.progress["claimed"].as_i64().unwrap();
    assert_eq!(s, to.max(claimed.min(files)));
    assert!(there - s <= 1300, "the settle removed {} files past the run in hand", there - s);
    let after = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert_eq!((after.provenance["resizes"][0]["op"].as_str(), after.provenance["resizes"][0]["settled"].as_i64()), (Some("shrink"), Some(s)));
}

/// Ctrl-C on any host stops every host (DESIGN_REVIEW.md §3.70): each exits 130, the
/// manifest records what is finished and that no datagen is writing, so `--settle` needs no
/// wait.
#[test]
fn ctrl_c_stops_every_host_and_leaves_a_true_manifest() {
    let files = 20000i64;
    let (_, _, model) = leaked_model("file_per_sample.ast.json", config(1, 1, &[("samples", "20000")]));
    let meta = &model.datasets[0];
    // and SIGTERM (Slurm, systemd, docker, timeout) as Ctrl-C: the host it reached exits 143
    for (ranks, at, sig) in [(2i64, 1usize, libc::SIGINT), (1, 0, libc::SIGINT), (2, 1, libc::SIGTERM)] {
        let root = tmpdir("dg-ctrl-c");
        let mut args = vec![crate::common::workload_ast("file_per_sample.ast.json").to_string_lossy().into_owned(), "--root".into(), root.to_string_lossy().into_owned(), "--param".into(), format!("samples={files}")];
        args.extend(["--writers", "2", "--progress-secs", "0.05"].map(String::from));
        let mut children = spawn_ranks(&args, ranks);
        let rest = windows(&mut children[0], 1, |l| !l.contains(" 0/"));
        unsafe { libc::kill(children[at].id() as i32, sig) };
        let tail: Vec<String> = rest.map_while(Result::ok).collect();
        for (i, c) in children.iter_mut().enumerate() {
            let st = c.wait().unwrap();
            let mut err = String::new();
            std::io::Read::read_to_string(c.stderr.as_mut().unwrap(), &mut err).unwrap();
            assert_eq!(st.code(), Some(if i == at { 128 + sig } else { 130 }), "{ranks} rank(s), signal {sig}, rank {i}: {err}");
        }
        let said = if ranks > 1 { "all 2 ranks stopped" } else { "stopped; the manifest records what is finished" };
        assert!(tail.iter().any(|l| l.starts_with(said)), "{tail:?}");
        let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
        assert!(m.state == "writing" && !m.progress["stopped"].is_null(), "{}", m.progress);
        let unfinished = m.unfinished();
        let mark = m.progress["claimed_to"].as_i64().unwrap();
        for id in 0..mark {
            if !unfinished.iter().any(|(a, b)| (*a..*b).contains(&id)) {
                let path = root.join(meta.file_path(id, None).unwrap().to_string());
                assert_eq!(std::fs::metadata(&path).map(|m| m.len()).ok(), Some(meta.file_size(id).unwrap() as u64), "{}: recorded as finished", path.display());
            }
        }
        // stopped: no wait for the windows
        let o = settle(&root.join("train"));
        assert!(o.0, "{o:?}");
        let s = check_settled(&root, &m, files);
        assert_eq!(s, m.progress["claimed"].as_i64().unwrap());
    }
}

/// A run stopped by Ctrl-C or SIGTERM (RC1 review, R47) stops promptly, says so on stderr,
/// writes its `--report-json` with the error as the verdict, and exits 128 + the signal; on
/// several hosts the others fail with the interrupted rank and the reason, not a connection
/// closed. Under the thread runner and an event loop alike.
#[test]
fn a_signal_stops_a_run_with_its_report_and_tells_the_other_hosts() {
    let root = tmpdir("run-signal");
    datagen_one(&root, 256);
    for (ranks, at, sig, posix) in [(1i64, 0usize, libc::SIGINT, "sync"), (1, 0, libc::SIGTERM, "io_uring"), (2, 1, libc::SIGINT, "sync")] {
        let port = free_port();
        let what = format!("{ranks} rank(s), signal {sig} on rank {at}, {posix}");
        let mut children: Vec<std::process::Child> = (0..ranks).map(|rank| {
            let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
            c.arg("run").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(&root).args(["--gpus", "2", "--posix", posix]);
            // 10 ms steps without end: only the signal stops it
            c.args(["--param", "samples=256", "--param", "batch=4", "--param", "steps=100000", "--param", "step_time=10000000"]);
            c.arg("--report-json").arg(root.join(format!("signal-{rank}.json")));
            if ranks > 1 {
                c.args(["--ranks", &ranks.to_string(), "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
            }
            let child = c.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped()).spawn().unwrap();
            if rank == 0 {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            child
        }).collect();
        // well into the run's steps
        std::thread::sleep(std::time::Duration::from_millis(1500));
        unsafe { libc::kill(children[at].id() as i32, sig) };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let mut outs = Vec::new();
        for c in children.iter_mut() {
            let st = loop {
                if let Some(st) = c.try_wait().unwrap() {
                    break Some(st);
                }
                if std::time::Instant::now() > deadline {
                    let _ = c.kill();
                    let _ = c.wait();
                    break None;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            };
            let mut err = String::new();
            std::io::Read::read_to_string(c.stderr.as_mut().unwrap(), &mut err).unwrap();
            outs.push((st.and_then(|s| s.code()), err));
        }
        let name = if sig == libc::SIGINT { "SIGINT" } else { "SIGTERM" };
        let reason = format!("interrupted ({name})");
        let (code, err) = &outs[at];
        assert_eq!(*code, Some(128 + sig), "{what}: {err}");
        assert_eq!(err.trim_end(), format!("aeiou run: {reason}"), "{what}");
        let d: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join(format!("signal-{at}.json"))).unwrap()).unwrap();
        assert_eq!((d["verdict"]["ok"].as_bool(), d["verdict"]["error"].as_str()), (Some(false), Some(reason.as_str())), "{what}");
        assert!(d["seed"].is_number(), "{what}: the run's header is in the report");
        if ranks > 1 {
            // rank 0 names the rank and its reason
            let (code, err) = &outs[0];
            assert_eq!(*code, Some(1), "{what}: {err}");
            assert!(err.contains("rank 1 (") && err.contains(&format!("): {reason}")) && !err.contains("connection closed"), "{what}: {err}");
            let d: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("signal-0.json")).unwrap()).unwrap();
            assert!(d["verdict"]["error"].as_str().is_some_and(|e| e.ends_with(&format!("): {reason}"))), "{what}: {}", d["verdict"]);
        }
    }
}

/// `run --subset` (DESIGN_REVIEW.md §3.70 item 6): a count below the corpus's runs the corpus's
/// first ids, rounded up to whole directories, on one host and on two, with the fingerprint of
/// a fresh corpus of the rounded count and the corpus's dataset id; and the refusals.
#[test]
fn a_subset_runs_whole_directories_of_the_corpus() {
    // 1300 files a directory: 3200 is two whole directories and 600 files
    let root = tmpdir("subset");
    datagen_one(&root, 3200);
    let corpus = aeiou::payload::Manifest::read(&root.join("train")).unwrap().id();
    let params = ["samples=2000", "steps=4"];
    let fp = dry_fingerprint_cli("file_per_sample.ast.json", "2", &["samples=2600", "steps=4"]);
    for ranks in [1i64, 2] {
        let json = root.join(format!("report-{ranks}.json"));
        let outs = run_ranks("file_per_sample.ast.json", &root, "2", &params, ranks, &["--subset", "--expect-fingerprint", &fp, "--report-json", json.to_str().unwrap()]);
        assert!(outs.iter().all(|(ok, _, _)| *ok), "{ranks} rank(s): {outs:?}");
        let text = &outs[0].1;
        assert!(text.contains("subset = train: samples = 2000 asked, rounded up to 2600 (whole directories of 1300 files), of the corpus's 3200  [cli]"), "{text}");
        assert!(text.contains("subset: a subset is the corpus's lowest ids"), "{text}");
        assert!(text.contains("params: steps=4 samples=2600"), "{text}");
        let d: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        let ds = &d["datasets"][0];
        assert_eq!(ds["id"].as_str(), Some(corpus.as_str()), "the corpus's id");
        let s = &ds["subset"];
        assert_eq!((s["param"].as_str(), s["corpus"].as_i64(), s["requested"].as_i64(), s["run"].as_i64(), s["unit"].as_i64()), (Some("samples"), Some(3200), Some(2000), Some(2600), Some(1300)));
        assert_eq!(d["verdict"]["fingerprint"].as_str().map(|f| f.trim_start_matches("0x").to_string()), Some(fp.trim_start_matches("0x").to_string()));
    }
    // the inventory in place of the walk (DESIGN_REVIEW §3.73): the whole corpus's read to its
    // end; a subset's read as a corpus of the rounded count's would be, issued no further than
    // its end (a read past it would return the corpus's bytes where 0 is expected, and fail)
    let inv = ["walk=false", "inventory=true", "steps=4"];
    let whole = [&inv[..], &["samples=3200"]].concat();
    let fp = dry_fingerprint_cli("file_per_sample.ast.json", "2", &whole);
    let outs = run_ranks("file_per_sample.ast.json", &root, "2", &whole, 1, &["--expect-fingerprint", &fp]);
    assert!(outs.iter().all(|(ok, _, _)| *ok), "the whole corpus's inventory: {outs:?}");
    let fp = dry_fingerprint_cli("file_per_sample.ast.json", "2", &[&inv[..], &["samples=2600"]].concat());
    for (ranks, posix) in [(1i64, "sync"), (2, "sync"), (1, "io_uring"), (1, "libaio")] {
        let json = root.join(format!("report-inv-{ranks}-{posix}.json"));
        let outs = run_ranks("file_per_sample.ast.json", &root, "2", &[&inv[..], &["samples=2000"]].concat(), ranks, &["--subset", "--posix", posix, "--expect-fingerprint", &fp, "--report-json", json.to_str().unwrap()]);
        assert!(outs.iter().all(|(ok, _, _)| *ok), "{ranks} rank(s), {posix}: {outs:?}");
        let d: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        assert_eq!(d["scope"].as_str(), Some("run"), "{ranks} rank(s), {posix}: the whole run's report");
        let ph = &d["result"]["phases"]["inventory"];
        // 2 GPUs: open, fstat, ioctl, lseek, close, and 2600 × 24 bytes in 1 MiB reads, the end read
        assert_eq!((ph["ops"].as_i64(), ph["bytes_read"].as_i64()), (Some(2 * (5 + 2)), Some(2 * 2600 * 24)), "{ranks} rank(s), {posix}: {ph}");
        if ranks > 1 {
            // rank 1's own report: its one GPU's read
            let d: serde_json::Value = serde_json::from_slice(&std::fs::read(json.with_extension("json.rank1")).unwrap()).unwrap();
            let ph = &d["result"]["phases"]["inventory"];
            assert_eq!((d["scope"].as_str(), ph["ops"].as_i64(), ph["bytes_read"].as_i64()), (Some("host"), Some(5 + 2), Some(2600 * 24)), "rank 1, {posix}: {ph}");
        }
    }

    let refused = |params: &[&str], extra: &[&str]| {
        let outs = run_ranks("file_per_sample.ast.json", &root, "2", params, 1, extra);
        assert!(!outs[0].0, "{outs:?}");
        outs[0].2.clone()
    };
    let e = refused(&["samples=3000", "steps=4"], &["--subset"]);
    assert!(e.contains("3000 rounds up to 3900 (whole directories of 1300 files), past the corpus's 3200; the largest subset of whole units is 2600"), "{e}");
    let e = refused(&["samples=4000", "steps=4"], &["--subset"]);
    assert!(e.contains("a subset is smaller; `aeiou datagen --grow 4000"), "{e}");
    let e = refused(&["samples=3200", "steps=4"], &["--subset"]);
    assert!(e.contains("--subset: no dataset is run smaller than its corpus"), "{e}");
    let e = refused(&["samples=2600", "steps=4"], &[]);
    assert!(e.contains("count: 2600 vs 3200") && e.contains("`--subset` runs the corpus's first ids"), "{e}");
    // another directory width is another dataset (contract 0.10): the bound pattern differs
    let e = refused(&["samples=2000", "files_per_directory=1000", "steps=4"], &["--subset"]);
    assert!(e.contains("differs from the corpus's in more than its count") && e.contains("{id div 1000:05}"), "{e}");

    // a flat corpus whose directory the abstract lists: the listing would be the whole corpus
    let mut doc: serde_json::Value = serde_json::from_slice(&std::fs::read(crate::common::workload_ast("file_per_sample.ast.json")).unwrap()).unwrap();
    doc["datasets"]["train"]["pattern"] = serde_json::json!("train/img_{id:09}.jpg");
    let flat = tmpdir("subset-flat");
    let ast = flat.join("file_per_sample.ast.json");
    std::fs::write(&ast, serde_json::to_vec(&doc).unwrap()).unwrap();
    let cli = |args: &[&str]| Command::new(env!("CARGO_BIN_EXE_aeiou")).args(args).output().unwrap();
    let (a, r) = (ast.to_str().unwrap(), flat.to_str().unwrap());
    assert!(cli(&["datagen", a, "--root", r, "--param", "samples=300"]).status.success());
    let o = cli(&["run", a, "--root", r, "--gpus", "2", "--time-scale", "0", "--param", "samples=100", "--param", "steps=4", "--subset"]);
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && e.contains("its files share one directory, and the abstract lists it"), "{e}");
    // without the listing, whole files: one sample a file, so nothing is rounded
    let o = cli(&["run", a, "--root", r, "--gpus", "2", "--time-scale", "0", "--param", "samples=100", "--param", "steps=4", "--param", "walk=false", "--subset"]);
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && e.contains("the abstract lists it"), "the refusal reads the abstract, not the parameters: {e}");

    // an `indexed` dataset is looked up by id: never a smaller corpus
    let small = ["shard_bytes=4194304", "hdr_len=[4096, 4096]", "off=[4104, 1052680, 4104, 1052680, 2101256]", "bytes=[1048576, 1048576, 1048576, 1048576, 16384]", "rows=[1, 64, 1, 64, 1]", "row_bytes=[1048576, 16384, 1048576, 16384, 16384]"];
    let st = crate::common::workload_ast("safetensors_load.ast.json");
    let mut args: Vec<String> = vec!["datagen".into(), st.display().to_string(), "--root".into(), r.into(), "--gpus".into(), "2".into()];
    args.extend(small.iter().flat_map(|p| ["--param".to_string(), p.to_string()]));
    let v: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    assert!(cli(&v).status.success());
    args[0] = "run".into();
    args.extend(["--time-scale", "0", "--param", "configs=2", "--subset"].map(String::from));
    let v: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let o = cli(&v);
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && e.contains("`indexed` (looked up by id, never consumed)"), "{e}");
}

/// A host that fails stops the others, and each of them reports that host's error and exits
/// 1: an abort from the coordinator is a failure, not a Ctrl-C (130, "stopped"). Rank 2 of
/// three writes under a dataset directory it cannot create in; rank 1 only hears of it.
#[test]
fn a_failing_datagen_host_fails_the_others_with_its_reason() {
    use std::os::unix::fs::PermissionsExt;
    let (root, elsewhere) = (tmpdir("dg-fail"), tmpdir("dg-fail-ro"));
    let ro = elsewhere.join("train");
    std::fs::create_dir(&ro).unwrap();
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..3i64 {
        let at = if rank == 2 { &*elsewhere } else { &*root };
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(at);
        c.args(["--param", "samples=4000", "--param", "files_per_directory=50", "--writers", "2"]);
        c.args(["--ranks", "3", "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")]);
        children.push(c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap());
        if rank == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    let outs: Vec<_> = children.into_iter().map(|c| c.wait_with_output().unwrap()).collect();
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).unwrap();
    for (rank, o) in outs.iter().enumerate() {
        let (stdout, stderr) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        assert_eq!(o.status.code(), Some(1), "rank {rank}:\n{stdout}\n{stderr}");
        assert!(stderr.contains("dg-fail-ro/train/0") && stderr.contains("Permission denied"), "rank {rank}: {stderr}");
        assert!(!stdout.lines().any(|l| l.starts_with("stopped") || l.starts_with("all 3 ranks stopped")), "rank {rank}: {stdout}");
        if rank != 2 {
            // the failing host is named
            assert!(stderr.contains("rank 2 ("), "rank {rank}: {stderr}");
        }
    }
    // what was finished is recorded, and that no datagen is writing
    let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
    assert!(m.state == "writing" && !m.progress["stopped"].is_null(), "{}", m.progress);
}

/// The event-loop blocks of a run on two hosts (RC1 review, bug 54): rank 0's merged report
/// counts every host's loops and takes the largest in-flight peak, never rank 0's alone; the
/// knobs and the kernel's io-wq caps are rank 0's. 3 GPUs over 2 ranks, so the hosts differ.
#[test]
fn the_merged_report_counts_every_hosts_event_loops() {
    let root = tmpdir("loops");
    datagen_one(&root, 500);
    let params = ["samples=500", "batch=4", "steps=3"];
    for (posix, block) in [("io_uring", "io_uring"), ("libaio", "libaio")] {
        let json = root.join(format!("report-{posix}.json"));
        let outs = run_ranks("file_per_sample.ast.json", &root, "3", &params, 2, &["--posix", posix, "--threads", "4", "--report-json", json.to_str().unwrap()]);
        assert!(outs.iter().all(|(ok, _, _)| *ok), "{posix}: {outs:?}");
        let d: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        let d1: serde_json::Value = serde_json::from_slice(&std::fs::read(json.with_extension("json.rank1")).unwrap()).unwrap();
        let (run, h0, h1) = (&d["result"][block], &d["this_host"][block], &d1["result"][block]);
        let n = |v: &serde_json::Value, k: &str| v[k].as_u64().unwrap_or_else(|| panic!("{posix}: no {k} in {v}"));
        assert_eq!((n(h0, "loops"), n(h1, "loops")), (2, 1), "{posix}: one loop per instance, at most 4: {h0} {h1}");
        assert_eq!(n(run, "loops"), 3, "{posix}: the run's loops are every host's: {run}");
        assert_eq!(n(run, "in_flight_peak"), n(h0, "in_flight_peak").max(n(h1, "in_flight_peak")), "{posix}: {run}");
        if posix == "io_uring" {
            assert_eq!((&run["opts"], &run["iowq_defaults"]), (&h0["opts"], &h0["iowq_defaults"]), "rank 0's knobs and caps");
        }
        let text = &outs[0].1;
        assert!(text.contains(&format!("{block}: loops 3 ")), "{posix}: the merged text report: {text}");
    }
}

/// `aeiou datagen` with `args`: (success, stdout, stderr).
fn datagen_cli(args: &[&std::ffi::OsStr]) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").args(args).output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// What `O_DIRECT` and unaligned tails are learned per dataset root (RC1 review): an
/// `--endpoint` may be another file system than `--root`. One finding per datagen made a root
/// that takes unaligned direct tails (tmpfs since Linux 6.6, NFS) pad and truncate every file
/// once an ext4 root had refused one, and an `O_DIRECT` refusal on one root sent every later
/// dataset through the page cache and recorded `direct: false` for all.
#[test]
fn the_direct_findings_are_each_roots_own() {
    struct Gone(PathBuf);
    impl Drop for Gone {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let shm = Path::new("/dev/shm");
    if !shm.is_dir() {
        eprintln!("no /dev/shm: skipped");
        return;
    }
    let root = tmpdir("dg-mode");
    let elsewhere = Gone(shm.join(format!("aeiou-coord-{}-dg-mode", std::process::id())));
    // `config` (written first) on the test's root, `model` on tmpfs; both with unaligned tails
    let a = crate::common::workload_ast("safetensors_load.ast.json");
    let ep = format!("model={}", elsewhere.0.join("model").display());
    let o = datagen_cli(&[a.as_os_str(), "--root".as_ref(), root.as_os_str(), "--endpoint".as_ref(), ep.as_ref(), "--param".as_ref(), "config_bytes=5000".as_ref(), "--param".as_ref(), "shard_bytes=10000".as_ref(), "--writers".as_ref(), "1".as_ref()]);
    assert!(o.0, "{o:?}");
    let part = |dir: &Path| aeiou::payload::Manifest::read(dir).unwrap().provenance["hosts"][0].clone();
    let (config, model) = (part(&root.join("config")), part(&elsewhere.0.join("model")));
    // what each file system does with an unaligned direct write, asked here directly
    let probe = |dir: &Path| -> Option<bool> {
        use std::os::unix::fs::{FileExt, OpenOptionsExt};
        let path = dir.join("probe");
        let f = std::fs::OpenOptions::new().write(true).create(true).custom_flags(libc::O_DIRECT).open(&path).ok()?;
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        let buf = unsafe { std::alloc::alloc_zeroed(layout) };
        let r = f.write_at(unsafe { std::slice::from_raw_parts(buf, 100) }, 0);
        unsafe { std::alloc::dealloc(buf, layout) };
        let _ = std::fs::remove_file(&path);
        Some(r.is_ok())
    };
    let (here, there) = (probe(&root), probe(&elsewhere.0));
    assert_eq!((config["direct"].as_bool(), model["direct"].as_bool()), (Some(here.is_some()), Some(there.is_some())), "{config} {model}");
    assert_eq!((config["pad_tails"].as_bool(), model["pad_tails"].as_bool()), (Some(here == Some(false)), Some(there == Some(false))), "{config} {model}");
}

/// A root that cannot be made takes back the roots this datagen made before it, so a refused
/// datagen leaves nothing (RC1 review): the earlier dataset's root stayed behind, empty.
#[test]
fn a_root_that_cannot_be_made_leaves_nothing_behind() {
    let root = tmpdir("dg-unmake");
    std::fs::write(root.join("blocker"), b"").unwrap();
    let fresh = root.join("fresh");
    // `config` is made first under --root; `model` cannot be, a file is in its way
    let a = crate::common::workload_ast("safetensors_load.ast.json");
    let ep = format!("model={}", root.join("blocker/model").display());
    let o = datagen_cli(&[a.as_os_str(), "--root".as_ref(), fresh.as_os_str(), "--endpoint".as_ref(), ep.as_ref(), "--param".as_ref(), "shard_bytes=10000".as_ref()]);
    assert!(!o.0 && o.2.contains("blocker/model: Not a directory"), "{o:?}");
    assert!(!fresh.exists(), "left behind: {:?}", listing(&root));
}

/// A root holding a corpus a datagen did not finish is refused with the ways to finish it
/// (RC1 review): a killed datagen run again without `--resume` was told only "is not empty
/// … remove it first".
#[test]
fn an_unfinished_corpus_points_at_resume_and_settle() {
    let root = tmpdir("dg-unfinished");
    datagen_one(&root, 30);
    let at = root.join("train");
    let path = at.join(aeiou::payload::MANIFEST_NAME);
    let mut m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    m["state"] = "writing".into();
    std::fs::write(&path, serde_json::to_string(&m).unwrap()).unwrap();
    let a = crate::common::workload_ast("file_per_sample.ast.json");
    let o = datagen_cli(&[a.as_os_str(), "--root".as_ref(), root.as_os_str(), "--param".as_ref(), "samples=30".as_ref()]);
    let want = format!("dataset `train`: {d} holds a corpus a datagen did not finish (its manifest is `writing`): `aeiou datagen --resume {d}` finishes it, `--settle {d}` keeps what it reached, or remove it first", d = at.display());
    assert!(!o.0 && o.2.contains(&want), "{o:?}");
    // a complete one is a dataset, read-only
    m["state"] = "complete".into();
    std::fs::write(&path, serde_json::to_string(&m).unwrap()).unwrap();
    let o = datagen_cli(&[a.as_os_str(), "--root".as_ref(), root.as_os_str(), "--param".as_ref(), "samples=30".as_ref()]);
    assert!(!o.0 && o.2.contains("is not empty (") && o.2.contains("datasets are read-only, remove it first"), "{o:?}");
}

/// A ratio that does not divide 4096 is made as the nearest a 4 KiB piece holds, and the
/// manifest says so (RC1 review): `--compress 1000` made 819.2 and recorded 1000 alone. The
/// payload, and so the dataset id, is the ratio asked, as it was.
#[test]
fn a_compress_ratio_a_piece_cannot_hold_is_recorded_as_made() {
    let a = crate::common::workload_ast("file_per_sample.ast.json");
    for (c, made) in [("3", Some(4096.0 / 1366.0)), ("1000", Some(819.2)), ("4", None), ("4096", None)] {
        let root = tmpdir("dg-compress");
        let o = datagen_cli(&[a.as_os_str(), "--root".as_ref(), root.as_os_str(), "--param".as_ref(), "samples=2".as_ref(), "--compress".as_ref(), c.as_ref()]);
        assert!(o.0, "{o:?}");
        let m = aeiou::payload::Manifest::read(&root.join("train")).unwrap();
        assert_eq!(m.payload.compress.to_string(), c);
        assert_eq!(m.provenance["compress_achieved"].as_f64(), made, "--compress {c}");
        assert_eq!(o.1.contains("so the ratio made is"), made.is_some(), "--compress {c}: {}", o.1);
    }
}

/// The progress line of an array file shows its bytes, rate, and time left while it is
/// written (RC1 review): it said "0/1 files, 0 B written, 0 B/s" until the file was whole. The
/// manifest still counts the file once whole.
#[test]
fn an_array_files_progress_is_live() {
    let root = tmpdir("dg-array-progress");
    let a = crate::common::workload_ast("faiss_ivf_ondisk.ast.json");
    let o = datagen_cli(&[a.as_os_str(), "--root".as_ref(), root.as_os_str(), "--dataset".as_ref(), "lists".as_ref(), "--param".as_ref(), "lists=4096".as_ref(), "--writers".as_ref(), "1".as_ref(), "--progress-secs".as_ref(), "0.01".as_ref()]);
    assert!(o.0, "{o:?}");
    let partial: Vec<&str> = o.1.lines().filter(|l| l.starts_with("progress ") && l.contains(" 0/1 files")).collect();
    assert!(!partial.is_empty(), "no window while the file was written: {}", o.1);
    assert!(partial.iter().any(|l| !l.contains(" 0 B written") && l.contains(" s left")), "{partial:?}");
}
