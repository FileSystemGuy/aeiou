//! The coordinator's wire when a peer breaks the protocol, spoken by hand over a `TcpStream`
//! (a bad frame, a message out of place, a scope no host registered, a rank twice, a close in
//! the middle of a frame), from both sides: a host the server hears, and a server the client
//! hears. Every failure reaches every host as a `Stop` with its reason, and none hangs.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use aeiou::coord::{Coordinator, Server, Tcp};

fn put(s: &mut TcpStream, body: &str) {
    s.write_all(&[&(body.len() as u32).to_be_bytes()[..], body.as_bytes()].concat()).unwrap();
}

/// The next frame on `s` that is no heartbeat, as text; `None` when the peer closed.
fn next(s: &mut TcpStream) -> Option<String> {
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    loop {
        let mut hdr = [0u8; 4];
        s.read_exact(&mut hdr).ok()?;
        let mut body = vec![0u8; u32::from_be_bytes(hdr) as usize];
        s.read_exact(&mut body).ok()?;
        let t = String::from_utf8_lossy(&body).into_owned();
        if t != "\"Heartbeat\"" {
            return Some(t);
        }
    }
}

fn hello(rank: i64, ranks: i64, participants: serde_json::Value) -> String {
    serde_json::json!({"Hello": {"rank": rank, "ranks": ranks, "host": format!("raw{rank}"), "identity": {"cfg": "c"}, "layers": {}, "participants": participants, "facts": {"cpus": rank}}}).to_string()
}

/// A host that said `Hello` and read its `Welcome`.
fn host(server: &Server, rank: i64, ranks: i64, participants: serde_json::Value) -> TcpStream {
    let mut s = TcpStream::connect(server.addr).unwrap();
    put(&mut s, &hello(rank, ranks, participants));
    let w = next(&mut s).expect("a Welcome");
    assert!(w.contains("Welcome"), "{w}");
    s
}

/// The `Stop` a host hears, its reason.
fn stopped(s: &mut TcpStream) -> String {
    loop {
        match next(s) {
            Some(m) if m.contains("Stop") => return m,
            Some(_) => continue,
            None => panic!("closed without a Stop"),
        }
    }
}

/// Two hosts welcomed, then rank 1 sends `bad`: rank 0 hears the `Stop` naming rank 1 and
/// `what`, and the server's waits end with it.
fn breaks_the_run(bad: &dyn Fn(&mut TcpStream), what: &str) {
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let mut r0 = host(&server, 0, 2, serde_json::json!([["s", 1]]));
    let mut r1 = host(&server, 1, 2, serde_json::json!([["s", 1]]));
    server.wait_connected().unwrap();
    bad(&mut r1);
    let stop = stopped(&mut r0);
    assert!(stop.contains("rank 1 (raw1)") && stop.contains(what), "{stop}");
    let e = format!("{:#}", server.merged().unwrap_err());
    assert!(e.contains(what), "{e}");
    assert!(format!("{:#}", server.gathered().unwrap_err()).contains(what));
    assert!(format!("{:#}", server.wait_connected().unwrap_err()).contains(what));
}

#[test]
fn a_host_that_breaks_the_protocol_stops_every_host_with_its_reason() {
    // a length past the largest frame
    breaks_the_run(&|s| s.write_all(&u32::MAX.to_be_bytes()).unwrap(), "frame of 4294967295 bytes");
    // bytes that are no message
    breaks_the_run(&|s| put(s, "{\"Nonsense\": 1}"), "unknown variant");
    // a message only the server sends
    breaks_the_run(&|s| put(s, "{\"Welcome\": {}}"), "unexpected message: Welcome");
    // a barrier no host registered
    breaks_the_run(&|s| put(s, "{\"Arrive\": {\"scope\": \"nowhere\"}}"), "Arrive at `nowhere`, which no host registered");
    breaks_the_run(&|s| put(s, "{\"Leave\": {\"scope\": \"nowhere\"}}"), "Leave `nowhere`, which no host registered");
    // the connection closed in the middle of a frame
    breaks_the_run(&|s| {
        s.write_all(&100u32.to_be_bytes()).unwrap();
        s.write_all(b"{\"Ar").unwrap();
        s.shutdown(std::net::Shutdown::Both).unwrap();
    }, "connection closed");
    // a host's own failure, said
    breaks_the_run(&|s| put(s, "{\"Stop\": {\"reason\": \"disk full\"}}"), "disk full");
}

/// A host whose `Hello` the server cannot take hears why on its own socket, and so does every
/// host already welcomed: a rank twice, a rank out of range, another `--ranks`.
#[test]
fn a_rank_twice_out_of_range_or_of_another_size_is_refused_on_every_host() {
    for (rank, ranks, what) in [(0, 2, "rank 0 is out of range or already connected"), (5, 2, "rank 5 is out of range or already connected"), (-1, 2, "rank -1 is out of range"), (1, 3, "--ranks 3, the coordinator was started for 2")] {
        let server = Server::start("127.0.0.1:0", 2).unwrap();
        let mut r0 = host(&server, 0, 2, serde_json::json!([]));
        let mut bad = TcpStream::connect(server.addr).unwrap();
        put(&mut bad, &hello(rank, ranks, serde_json::json!([])));
        let own = stopped(&mut bad);
        assert!(own.contains(what), "{own}");
        let theirs = stopped(&mut r0);
        assert!(theirs.contains(what), "{theirs}");
        assert!(format!("{:#}", server.wait_connected().unwrap_err()).contains(what));
    }
}

/// The barrier across hosts, by hand: a host whose participants all left (`Leave`) no longer
/// holds a release back, and its leaving completes a generation another host was waiting at
/// (a departure release, counted on the server). `Progress`, `Claim`, and `Halt` reach rank 0's
/// side; a `Halt` goes to every host once.
#[test]
fn leave_releases_by_departure_and_a_datagens_messages_reach_rank_0() {
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let mut r0 = host(&server, 0, 2, serde_json::json!([["s", 2], ["t", 0]]));
    let mut r1 = host(&server, 1, 2, serde_json::json!([["s", 1]]));
    put(&mut r0, "\"Ready\"");
    put(&mut r1, "\"Ready\"");
    let start = next(&mut r0).unwrap();
    assert!(start.contains("Start") && start.contains("raw0") && start.contains("raw1"), "{start}");
    assert!(next(&mut r1).unwrap().contains("Start"));
    assert_eq!(server.hosts().iter().map(|(r, h, _)| (*r, h.clone())).collect::<Vec<_>>(), [(0, "raw0".to_string()), (1, "raw1".to_string())]);
    assert_eq!(server.facts()[&1]["cpus"], 1);
    // rank 0 arrives; rank 1 leaves without arriving: released, by departure
    put(&mut r0, "{\"Arrive\": {\"scope\": \"s\"}}");
    // (two connections: the arrival first)
    std::thread::sleep(Duration::from_millis(300));
    put(&mut r1, "{\"Leave\": {\"scope\": \"s\"}}");
    let rel = next(&mut r0).unwrap();
    assert!(rel.contains("Release") && rel.contains("\"generation\":1"), "{rel}");
    assert!(next(&mut r1).unwrap().contains("Release"));
    assert_eq!(server.departure_releases(), [("s".to_string(), 1)]);
    // from now on rank 0 alone completes `s`, and no departure is counted
    put(&mut r0, "{\"Arrive\": {\"scope\": \"s\"}}");
    assert!(next(&mut r0).unwrap().contains("\"generation\":2"));
    // a host whose participants left still hears every release
    assert!(next(&mut r1).unwrap().contains("\"generation\":2"));
    assert_eq!(server.departure_releases(), [("s".to_string(), 1)]);
    // a datagen's messages
    put(&mut r1, "{\"Progress\": {\"files\": 3}}");
    put(&mut r1, "{\"Claim\": {\"dataset\": \"d\", \"ticket\": 7}}");
    assert_eq!(server.next_claim(Duration::from_secs(10)), Some((1, "d".to_string(), 7)));
    assert_eq!(server.progress()[&1]["files"], 3);
    assert_eq!(server.next_claim(Duration::from_millis(10)), None);
    server.grant(1, "d", 7, Some((0, 4)));
    let g = next(&mut r1).unwrap();
    assert!(g.contains("Grant") && g.contains("\"ticket\":7") && g.contains("[0,4]"), "{g}");
    put(&mut r1, "\"Halt\"");
    put(&mut r0, "\"Halt\"");
    assert_eq!(next(&mut r0).unwrap(), "\"Halt\"");
    assert_eq!(next(&mut r1).unwrap(), "\"Halt\"");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !server.halted() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(server.halted());
    // halted, a host that goes away without its Done counts as stopped, and is no failure
    put(&mut r0, "{\"Done\": {\"files\": 1}}");
    drop(r1);
    let got = server.gathered().unwrap();
    assert_eq!(got.keys().copied().collect::<Vec<_>>(), [0]);
    // one Halt only: the second was not broadcast again
    server.finish(true, 9, None);
    let last = next(&mut r0).unwrap();
    assert!(last.contains("Result") && last.contains("\"fingerprint\":9"), "{last}");
}

/// `last_word`: rank 0 goes no further, every other host gets what it says with `Welcome`.
#[test]
fn rank_0s_last_word_reaches_a_host_waiting_for_its_welcome() {
    let server = Server::start_held("127.0.0.1:0", 2).unwrap();
    let addr = server.addr;
    let t = std::thread::spawn(move || {
        let mut s = TcpStream::connect(addr).unwrap();
        put(&mut s, &hello(1, 2, serde_json::json!([])));
        next(&mut s).unwrap()
    });
    std::thread::sleep(Duration::from_millis(100));
    server.last_word(serde_json::json!({"refused": "nothing to do"}));
    let w = t.join().unwrap();
    assert!(w.contains("Welcome") && w.contains("nothing to do"), "{w}");
}

/// `stop` before every host is welcomed: one waiting for its `Welcome` hears the reason.
#[test]
fn a_stop_before_the_welcome_reaches_a_waiting_host() {
    let server = Server::start_held("127.0.0.1:0", 2).unwrap();
    let mut s = TcpStream::connect(server.addr).unwrap();
    put(&mut s, &hello(1, 2, serde_json::json!([])));
    std::thread::sleep(Duration::from_millis(100));
    server.stop("rank 0: the manifest is unreadable");
    let m = stopped(&mut s);
    assert!(m.contains("the manifest is unreadable"), "{m}");
}

// ---------------------------------------------------------------- a server by hand

/// A coordinator by hand: it reads the client's `Hello`, then plays `script` on the socket.
fn fake_server(script: impl FnOnce(&mut TcpStream) + Send + 'static) -> (String, std::thread::JoinHandle<()>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let h = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let m = next(&mut s).unwrap();
        assert!(m.contains("Hello"), "{m}");
        script(&mut s);
    });
    (addr, h)
}

fn client(addr: &str) -> anyhow::Result<Tcp> {
    Tcp::connect(addr, 1, 2, "h1", &serde_json::json!({}), &serde_json::json!({}), &[("s".to_string(), 1)], Arc::new(AtomicBool::new(false)))
}

#[test]
fn a_coordinator_that_breaks_the_protocol_fails_the_host_with_why() {
    let stop = "{\"Stop\":{\"reason\":\"no, said the coordinator\"}}";
    let stop = [&(stop.len() as u32).to_be_bytes()[..], stop.as_bytes()].concat();
    for (send, what) in [
        (&b"\x00\x00\x00\x07\"Ready\""[..], "unexpected message from the coordinator: Ready"),
        (&b"\x00\x00\x00\x03\"Ha"[..], "EOF while parsing"),
        (&b"\xff\xff\xff\xff"[..], "frame of 4294967295 bytes"),
        (&b"\x00\x00\x00\x30{\"Welc"[..], "connection closed"),
        (&stop[..], "no, said the coordinator"),
    ] {
        let send = send.to_vec();
        let (addr, h) = fake_server(move |s| {
            s.write_all(&send).unwrap();
            s.shutdown(std::net::Shutdown::Write).unwrap();
            std::thread::sleep(Duration::from_millis(200));
        });
        let e = format!("{:#}", client(&addr).err().expect("refused"));
        assert!(e.contains("refused rank 1") && e.contains(what), "{what}: {e}");
        h.join().unwrap();
    }
}

/// The client's side of a datagen and a barrier: grants by ticket, a `Halt`, a release; then a
/// coordinator gone in the middle of the run fails what waits on it.
#[test]
fn a_hosts_grants_halt_and_barrier_come_from_its_reader() {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let (addr, h) = fake_server(move |s| {
        put(s, "{\"Welcome\": {\"doc\": {\"m\": 1}}}");
        assert!(next(s).unwrap().contains("Ready"));
        put(s, "{\"Start\": {\"t0\": 1.5, \"hosts\": [\"a\", \"b\"], \"share\": {\"h\": 2}}}");
        let c = next(s).unwrap();
        assert!(c.contains("\"Claim\"") && c.contains("\"ticket\":1"), "{c}");
        let c = next(s).unwrap();
        assert!(c.contains("\"ticket\":2"), "{c}");
        put(s, "{\"Grant\": {\"dataset\": \"d\", \"ticket\": 2, \"run\": null}}");
        put(s, "{\"Grant\": {\"dataset\": \"d\", \"ticket\": 1, \"run\": [0, 8]}}");
        assert!(next(s).unwrap().contains("Arrive"));
        put(s, "{\"Release\": {\"scope\": \"s\", \"generation\": 1}}");
        put(s, "{\"Release\": {\"scope\": \"elsewhere\", \"generation\": 1}}");
        put(s, "\"Halt\"");
        rx.recv().unwrap();
        // gone without a word
    });
    let t = client(&addr).unwrap();
    assert_eq!(t.welcome["m"], 1);
    let (t0, hosts, share) = t.ready_shared().unwrap();
    assert_eq!((t0, hosts, share["h"].as_i64()), (1.5, vec!["a".to_string(), "b".to_string()], Some(2)));
    let (a, b) = (t.claim("d").unwrap(), t.claim("d").unwrap());
    t.wait_grant(a).unwrap();
    t.wait_grant(b).unwrap();
    assert_eq!(t.grant_book("d"), (1, vec![(0, 8)]));
    assert_eq!(t.take_grant(b), Some(None));
    assert_eq!(t.take_grant(a), Some(Some((0, 8))));
    assert_eq!(t.take_grant(a), None);
    let never = AtomicBool::new(false);
    t.barrier("s", &never).unwrap();
    assert!(t.released("s", 0) && !t.released("s", 1));
    assert!(t.arrive("unknown").unwrap_err().to_string().contains("no participants registered on this host"));
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !t.halted() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(t.halted());
    tx.send(()).unwrap();
    h.join().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while t.abort_reason().is_none() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    // the coordinator gone: a barrier, the verdict, and a send all fail with why
    let e = format!("{:#}", t.barrier("s", &never).unwrap_err());
    assert!(e.contains("connection closed") || e.contains("Broken pipe") || e.contains("reset"), "{e}");
    assert!(format!("{:#}", t.result().unwrap_err()).contains("connection closed"));
    assert!(t.abort_reason().unwrap().contains("connection closed"));
    // a send: the first may still go out before the peer's reset is in
    let e = (0..100).find_map(|_| t.done(&serde_json::json!({})).err().or_else(|| {
        std::thread::sleep(Duration::from_millis(10));
        None
    }));
    assert!(format!("{:#}", e.expect("a send to a closed coordinator fails")).contains("connection closed"));
}

/// The address a host is given that is no address: said, with the address.
#[test]
fn a_coordinator_address_that_resolves_to_nothing_is_named() {
    let e = format!("{:#}", client("no-such-host.invalid:1").err().unwrap());
    assert!(e.contains("coordinator address `no-such-host.invalid:1`"), "{e}");
    let e = format!("{:#}", client("not an address").err().unwrap());
    assert!(e.contains("coordinator address `not an address`"), "{e}");
}

/// A coordinator that welcomes a host and then says nothing more (no heartbeat) is declared
/// gone after `DEAD_AFTER` of silence, and the host fails naming it, rather than waiting for
/// the gate for ever. The times are scaled down by `AEIOU_COORD_HEARTBEAT_MS` (50 ms: dead
/// after 300 ms), which only a process of its own reads.
#[test]
fn a_silent_coordinator_is_declared_gone() {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let (addr, h) = fake_server(move |s| {
        put(s, "{\"Welcome\": {}}");
        // hears the host's heartbeats, says nothing, until the host has gone
        let _ = rx.recv_timeout(Duration::from_secs(60));
        let _ = s;
    });
    let root = crate::common::tmpdir("cov_c3", "silent");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou"))
        .arg("datagen")
        .arg(crate::common::workload_ast("file_per_sample.ast.json"))
        .arg("--root")
        .arg(&*root)
        .args(crate::common::file_per_sample_args())
        .args(["--param", "samples=10", "--ranks", "2", "--rank", "1", "--coordinator", &addr])
        .env("AEIOU_COORD_HEARTBEAT_MS", "50")
        .env_remove("AEIOU_CONFIG")
        .output()
        .unwrap();
    tx.send(()).unwrap();
    // judged before the join: a host that fails before it connects leaves the server in accept
    let err = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(1), "{err}");
    assert!(err.contains(&format!("coordinator {addr}: no message for 300ms")), "{err}");
    h.join().unwrap();
}

// ---------------------------------------------------------------- datagen's refusals

/// The abstract, model, and options of a datagen of `text` at `root` (rank 0 of 1).
fn dg(text: &str, root: &std::path::Path) -> (&'static aeiou::Loaded, &'static aeiou::eval::Config, &'static aeiou::eval::Params, &'static aeiou::eval::Model<'static>, aeiou::datagen::DatagenOpts) {
    use aeiou::eval::{build_model, Config, Model, Params};
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(text).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() }));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let opts = aeiou::datagen::DatagenOpts { root: root.to_path_buf(), endpoints: Default::default(), writers: 2, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: true };
    (loaded, cfg, params, model, opts)
}

fn whole(name: &str, n: i64) -> String {
    format!(r#""{name}": {{"layout": "whole", "pattern": "{name}/f{{id}}", "count": {n}, "size": {{"const": 100}}, "seed": 3, "access": "map"}}"#)
}

fn doc(datasets: &[String]) -> String {
    format!(r#"{{"ast": "0.12", "name": "dg", "params": {{}}, "datasets": {{{}}}, "actors": {{"a": {{"body": []}}}}}}"#, datasets.join(", "))
}

/// What a datagen is refused before it writes anything: a rank outside its ranks, several
/// ranks without a coordinator, and a resume that would mix two corpora (another payload,
/// another format, more than the one dataset it finishes); a grow of an `array` and settles
/// that have nothing to decide.
#[test]
fn a_datagen_that_would_mix_corpora_or_ranks_is_refused() {
    let root = crate::common::tmpdir("cov_c3", "dg-refused");
    let text = doc(&[whole("d", 4)]);
    let (loaded, cfg, params, model, opts) = dg(&text, &root);
    let refused = |o: &aeiou::datagen::DatagenOpts, m: &aeiou::eval::Model<'_>, l: &aeiou::Loaded| format!("{:#}", aeiou::datagen::datagen(l, cfg, params, m, o, None, &mut Vec::new()).err().expect("refused"));
    let e = refused(&aeiou::datagen::DatagenOpts { rank: 2, ..dg(&text, &root).4 }, model, loaded);
    assert_eq!(e, "rank 2 of 1");
    let e = refused(&aeiou::datagen::DatagenOpts { ranks: 2, ..dg(&text, &root).4 }, model, loaded);
    assert_eq!(e, "2 ranks and no coordinator");
    // a corpus to resume: complete, then its payload or its format other than this datagen's
    aeiou::datagen::datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()).unwrap();
    let m = aeiou::payload::Manifest::read(&root.join("d")).unwrap();
    let e = refused(&aeiou::datagen::DatagenOpts { resume: Some(m.clone()), dedupe: 2, ..dg(&text, &root).4 }, model, loaded);
    assert!(e.starts_with("the corpus was written with the payload ") && e.contains("dedupe 1") && e.contains("dedupe 2") && e.ends_with("a resume would mix them"), "{e}");
    let mut other = m.clone();
    other.format = Some(serde_json::json!({"class": "tfrecord"}));
    let e = refused(&aeiou::datagen::DatagenOpts { resume: Some(other), ..dg(&text, &root).4 }, model, loaded);
    assert!(e.contains("dataset `d`: the corpus's format is {\"class\":\"tfrecord\"} and this aeiou writes none: a resume would mix them"), "{e}");
    let two = doc(&[whole("d", 4), whole("e", 4)]);
    let (l2, _, _, m2, _) = dg(&two, &root);
    let e = refused(&aeiou::datagen::DatagenOpts { resume: Some(m.clone()), ..dg(&text, &root).4 }, m2, l2);
    assert_eq!(e, "a resume finishes one dataset, not 2");
    // a grow or shrink of an `array`
    let mut arr = m.clone();
    arr.dataset["layout"] = "array".into();
    assert!(aeiou::datagen::resize_manifest(&arr, 4, 8).unwrap_err().to_string().contains("an `array` does not grow or shrink"));
    // a settle: one already decided is kept as it is; one with nothing handed out is refused
    let meta = &model.datasets[0];
    let mut w = m.clone();
    w.state = "writing".into();
    w.progress = serde_json::json!({"settle": {"asked": 4, "count": 2}, "claimed": 2});
    assert_eq!(aeiou::datagen::settle_manifest(&w, meta).unwrap().progress, w.progress);
    w.progress = serde_json::json!({"claimed": 0, "unfinished": [[0, 4]]});
    assert!(aeiou::datagen::settle_manifest(&w, meta).unwrap_err().to_string().contains("nothing was handed out, so a settled corpus would be empty"));
    w.progress = serde_json::json!({"unfinished": [[0, 4]]});
    assert!(aeiou::datagen::settle_manifest(&w, meta).unwrap_err().to_string().contains("records no claim boundary"));
}

/// An HDF5 dataset's headers come from `aeiou-header` beside the binary, on rank 0 only;
/// `Start` carries them to every other rank, which writes the same bytes at the head of its
/// files; a resume whose `aeiou-header` makes other headers is refused. Here the binary is
/// a hard link in a directory of the test's own, beside an `aeiou-header` that is a script
/// writing `file_header` bytes of one letter per row count asked.
#[cfg(feature = "formats")]
#[test]
fn rank_0s_headers_reach_every_rank_and_a_resume_must_make_the_same() {
    use std::os::unix::fs::PermissionsExt;
    let d = crate::common::tmpdir("cov_c3", "headers");
    let bin = d.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let aeiou = bin.join("aeiou");
    if std::fs::hard_link(env!("CARGO_BIN_EXE_aeiou"), &aeiou).is_err() {
        std::fs::copy(env!("CARGO_BIN_EXE_aeiou"), &aeiou).unwrap();
    }
    let tool = bin.join("aeiou-header");
    let script = |letter: char| format!("#!/bin/sh\n# the rows asked, in order: 2048 bytes of '{letter}' each\nfor a in \"$@\"; do\n  case \"$prev\" in --rows) head -c 2048 /dev/zero | tr '\\0' '{letter}';; esac\n  prev=\"$a\"\ndone\n");
    std::fs::write(&tool, script('H')).unwrap();
    std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o755)).unwrap();
    // 10 samples in files of 4: headers for 4 rows and for the short last file's 2
    let ast = d.join("h5.ast.json");
    std::fs::write(
        &ast,
        r#"{"ast": "0.12", "name": "h5", "params": {},
  "datasets": {"d": {"layout": "compound", "pattern": "h/{id:04}.h5", "count": 10, "samples_per_file": 4,
     "size": {"const": 300}, "seed": 11, "access": "stream", "format": {"class": "hdf5", "reader": "h5py", "version": "test", "framing": {"file_header": 2048, "writer": {"dataset": "records", "shape": [300], "dtype": "u1"}}}}},
  "actors": {"gpu": {"body": []}}}"#,
    )
    .unwrap();
    let root = d.join("root");
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let children: Vec<_> = (0..2)
        .map(|rank| {
            let c = std::process::Command::new(&aeiou)
                .arg("datagen")
                .arg(&ast)
                .arg("--root")
                .arg(&root)
                .args(["--writers", "1", "--ranks", "2", "--rank", &rank.to_string(), "--coordinator", &format!("127.0.0.1:{port}")])
                .env_remove("AEIOU_CONFIG")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            if rank == 0 {
                std::thread::sleep(Duration::from_millis(100));
            }
            c
        })
        .collect();
    for (rank, c) in children.into_iter().enumerate() {
        let o = c.wait_with_output().unwrap();
        let (out, err) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        assert!(o.status.success(), "rank {rank}:\n{out}\n{err}");
        if rank == 0 {
            assert!(out.contains("dataset d: 2 header(s) from "), "{out}");
        }
    }
    // every file, whichever rank wrote it, starts with the header rank 0 was given
    for f in ["h/0000.h5", "h/0001.h5", "h/0002.h5"] {
        let bytes = std::fs::read(root.join(f)).unwrap();
        assert!(bytes[..2048].iter().all(|b| *b == b'H'), "{f}");
    }
    let m = aeiou::payload::Manifest::read(&root.join("h")).unwrap();
    assert_eq!(m.provenance["headers"]["tool"], "aeiou-header");
    // the corpus as a datagen killed before its first window leaves it
    let mut w = m.clone();
    w.state = "writing".into();
    w.progress = serde_json::json!({"unfinished": [[0, 3]], "files_written": 0, "bytes_written": 0, "updated": 1.0, "progress_secs": 30.0});
    w.write(&root.join("h")).unwrap();
    let resume = || std::process::Command::new(&aeiou).args(["datagen", "--resume"]).arg(root.join("h")).env_remove("AEIOU_CONFIG").output().unwrap();
    // a resume whose aeiou-header makes other bytes would mix two corpora
    std::fs::write(&tool, script('J')).unwrap();
    let o = resume();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && err.contains("dataset `d`: aeiou-header made headers other than the corpus's"), "{}\n{err}", String::from_utf8_lossy(&o.stdout));
    // the same headers: finished, the corpus the one it was
    std::fs::write(&tool, script('H')).unwrap();
    let o = resume();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let after = aeiou::payload::Manifest::read(&root.join("h")).unwrap();
    assert_eq!((after.state.as_str(), after.id()), ("complete", m.id()));
}

/// One host: an instance that finishes leaves its barriers, and when the others have all
/// arrived its leaving releases them, counted as a departure release.
#[test]
fn a_local_departure_releases_the_instances_waiting() {
    let c = aeiou::coord::Local::new(&[("s".to_string(), 2), ("t".to_string(), 1)]);
    let g = c.arrive("s").unwrap();
    assert!(!c.released("s", g));
    c.leave("s");
    assert!(c.released("s", g));
    assert_eq!(c.departure_releases(), [("s".to_string(), 1)]);
    // from now on one arrival completes it, and no departure is counted
    let g = c.arrive("s").unwrap();
    assert!(c.released("s", g));
    assert_eq!(c.departure_releases(), [("s".to_string(), 1)]);
    // a leave with nobody waiting releases nothing; a scope no instance has is no error
    c.leave("t");
    c.leave("nowhere");
    assert_eq!(c.departure_releases(), [("s".to_string(), 1)]);
    assert!(c.arrive("nowhere").unwrap_err().to_string().contains("no participants registered"));
}

/// A host's `Hello` whose identity differs from the first host's is refused with each
/// difference, a parameter by its name; identities that are no object differ as a whole.
#[test]
fn an_identity_that_differs_is_named_parameter_by_parameter() {
    let connect = |addr: &str, rank: i64, id: &serde_json::Value| Tcp::connect(addr, rank, 2, &format!("h{rank}"), id, &serde_json::json!({}), &[], Arc::new(AtomicBool::new(false)));
    let id = |seed: i64, b: i64| serde_json::json!({"seed": seed, "params": {"a": 1, "b": b}});
    for (first, second, want) in [
        (id(1, 2), id(1, 3), "not the same run: params.b 3 differs from rank 0's 2"),
        (serde_json::json!({"seed": 1}), serde_json::json!({"seed": 2, "params": {"a": 1}}), "params {\"a\":1} differs from rank 0's null; seed 2 differs from rank 0's 1"),
        (serde_json::json!("x"), serde_json::json!("y"), "not the same run: identity differs from rank 0's"),
    ] {
        let server = Server::start("127.0.0.1:0", 2).unwrap();
        let addr = server.addr.to_string();
        let _t0 = connect(&addr, 0, &first).unwrap();
        let e = format!("{:#}", connect(&addr, 1, &second).err().unwrap());
        assert!(e.contains(want) && !e.contains("params.a"), "{e}");
    }
}

/// The merged report counts the departure releases of both levels: a host's own (its
/// instances) and the server's (a host whose participants all left), added up per scope.
#[test]
fn the_merged_report_adds_the_hosts_and_the_servers_departures() {
    let server = Server::start("127.0.0.1:0", 2).unwrap();
    let addr = server.addr.to_string();
    let parts = [("s".to_string(), 1)];
    let connect = |rank: i64| Tcp::connect(&addr, rank, 2, &format!("h{rank}"), &serde_json::json!({}), &serde_json::json!({}), &parts, Arc::new(AtomicBool::new(false))).unwrap();
    let (t0, t1) = std::thread::scope(|sc| {
        let a = sc.spawn(|| connect(0));
        let b = sc.spawn(|| connect(1));
        (a.join().unwrap(), b.join().unwrap())
    });
    std::thread::scope(|sc| {
        sc.spawn(|| t0.ready().unwrap());
        t1.ready().unwrap();
    });
    // rank 0 arrives and waits; rank 1's only participant leaves: the server releases it by
    // departure
    let g = t0.arrive("s").unwrap();
    std::thread::sleep(Duration::from_millis(300));
    t1.leave("s");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !t0.released("s", g) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(t0.released("s", g));
    assert_eq!(server.departure_releases(), [("s".to_string(), 1)]);
    let mut r = aeiou::run::Report::merge_all(vec![]);
    r.departure_releases = vec![("s".to_string(), 2), ("u".to_string(), 1)];
    t0.report(&r).unwrap();
    r.departure_releases = vec![];
    t1.report(&r).unwrap();
    let m = server.merged().unwrap();
    assert_eq!(m.departure_releases, [("s".to_string(), 3), ("u".to_string(), 1)]);
    server.finish(true, 0, None);
    assert_eq!(t1.result().unwrap(), (true, 0, None));
}
