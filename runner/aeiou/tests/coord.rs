//! The TCP coordinator: barriers across hosts (with departures), the configuration check in
//! `Hello`, a multi-host run whose merged fingerprint is the dry run's, and the binary end to
//! end: `ckpt_write_dcp` on two ranks, then `ckpt_restore` on two ranks with `--rank-rotate 1`,
//! through the namespace manifest the coordinator's merged report produced. Every "host" is a
//! thread or a process on this machine; the wire is the same.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use aeiou::backend::BackendKind;
use aeiou::coord::{Coordinator, Server, Tcp};
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};

static N: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aeiou-coord-{}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed), tag));
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

fn leaked_model(name: &str, cfg: Config) -> (&'static aeiou::Loaded, &'static Config, &'static Model<'static>) {
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&examples().join(format!("{name}.ast.json"))).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    (loaded, cfg, model)
}

fn opts(root: &PathBuf, rank: i64, ranks: i64) -> RunOpts {
    RunOpts {
        root: root.clone(),
        backend: BackendKind::Sync,
        buffer_bytes: 1 << 20,
        write_compress: 1,
        time_scale: 0.0,
        clean_namespaces: false,
        expect_fingerprint: None,
        expect_dataset_ids: vec![],
        rank,
        ranks,
        rank_rotate: 0,
        max_gap: None,
        require_cold: false,
    }
}

fn connect(server: &Server, rank: i64, ranks: i64, config: &str, participants: &[(String, usize)]) -> anyhow::Result<Arc<Tcp>> {
    Ok(Arc::new(Tcp::connect(&server.addr.to_string(), rank, ranks, &format!("host{rank}"), config, participants, Arc::new(AtomicBool::new(false)))?))
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
        let t = Arc::new(Tcp::connect(&s0, 0, 2, "host0", "cfg", &scope, Arc::new(AtomicBool::new(false))).unwrap());
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
        let t = Tcp::connect(&s1, 1, 2, "host1", "cfg", &scope1, Arc::new(AtomicBool::new(false))).unwrap();
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
    assert!(format!("{e:#}").contains("differs from rank 0"), "{e:#}");
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

#[test]
fn two_hosts_reproduce_the_dry_run_fingerprint() {
    let root = tmpdir("two");
    let params = [("files", "600"), ("batch", "4"), ("workers", "2"), ("prefetch", "2"), ("steps", "12"), ("enumerate", "true")];
    let (loaded, cfg, model) = leaked_model("train_small_files", config(4, 7, &params));
    let dparams = Params::new(&loaded.ast, cfg).unwrap();
    let mut log = Vec::new();
    datagen(loaded, cfg, &dparams, model, &DatagenOpts { root: root.clone(), threads: 4, dedupe: 1, compress: 1, datasets: vec![] }, &mut log).unwrap();
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
            let t = Arc::new(Tcp::connect(&addr, rank, 2, &format!("host{rank}"), "cfg", &p, aborted.clone()).unwrap());
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
    std::fs::remove_dir_all(&root).unwrap();
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn dry_fingerprint_cli(ast: &str, gpus: &str, params: &[&str]) -> String {
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.arg("dry-run").arg(examples().join(ast)).arg("--gpus").arg(gpus);
    for p in params {
        c.arg("--param").arg(p);
    }
    let o = c.output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let text = String::from_utf8(o.stdout).unwrap();
    text.lines().find_map(|l| l.strip_prefix("fingerprint ")).unwrap().split_whitespace().next().unwrap().to_string()
}

fn run_ranks(ast: &str, root: &PathBuf, gpus: &str, params: &[&str], ranks: i64, extra: &[&str]) -> Vec<(bool, String, String)> {
    let port = free_port();
    let mut children = Vec::new();
    for rank in 0..ranks {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("run").arg(examples().join(ast)).arg("--gpus").arg(gpus).arg("--root").arg(root).arg("--time-scale").arg("0");
        c.arg("--ranks").arg(ranks.to_string()).arg("--rank").arg(rank.to_string()).arg("--coordinator").arg(format!("127.0.0.1:{port}"));
        for p in params {
            c.arg("--param").arg(p);
        }
        c.args(extra);
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

#[test]
fn checkpoint_write_and_restore_on_two_ranks_end_to_end() {
    let root = tmpdir("e2e");
    let wparams = ["steps=4", "ckpt_every=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "meta_bytes=65536"];
    let wfp = dry_fingerprint_cli("ckpt_write_dcp.ast.json", "2", &wparams);
    let outs = run_ranks("ckpt_write_dcp.ast.json", &root, "2", &wparams, 2, &["--expect-fingerprint", &wfp]);
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
    let rparams = ["restore_step=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "item_off=[0, 1114112, 3276800, 4390912]", "meta_bytes=65536"];
    let rfp = dry_fingerprint_cli("ckpt_restore.ast.json", "2", &rparams);
    let outs = run_ranks("ckpt_restore.ast.json", &root, "2", &rparams, 2, &["--rank-rotate", "1", "--max-gap", "120", "--expect-fingerprint", &rfp]);
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
    assert!(outs.iter().any(|(_, _, e)| e.contains("differs from rank")), "{outs:?}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Two ranks of `ckpt_restore` with different seeds.
fn run_ranks_mixed(root: &PathBuf) -> Vec<(bool, String, String)> {
    let port = free_port();
    let rparams = ["restore_step=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "item_off=[0, 1114112, 3276800, 4390912]", "meta_bytes=65536"];
    let mut children = Vec::new();
    for rank in 0..2 {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
        c.arg("run").arg(examples().join("ckpt_restore.ast.json")).arg("--gpus").arg("2").arg("--root").arg(root).arg("--time-scale").arg("0");
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
