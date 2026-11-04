//! `aeiou-trace capture` (`DESIGN_REVIEW.md` §3.96, brief item 54, phase 1): the strace source
//! starting the command, the capture directory and its record (`capture.json`), the checks as
//! the first step of every capture and their waivers by name, the dry `--check`, a signal sent
//! to the tool reaching the command, and several ranks over the coordinator: the start barrier,
//! the clock offsets, a check failing on one rank starting nothing anywhere, and rank 0's
//! `capture-set.json`. The pieces without a process (mountinfo, the comparisons) are the unit
//! tests of `capture.rs`. strace is required (`dev.sh` needs it; CI installs it): never skipped.
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_aeiou-trace")
}

fn tmp(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("capture", tag)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn text(o: &Output) -> (String, String) {
    (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// The checks a test on a shared machine waives: other tests' captures run beside it
/// (`alone`), and a test host's clock may be unsynchronized (`clock`).
const WAIVE: [&str; 4] = ["--accept", "alone", "--accept", "clock"];

fn capture(args: &[&str]) -> Output {
    Command::new(bin()).arg("capture").args(args).output().unwrap()
}

fn record(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("capture.json")).unwrap()).unwrap()
}

fn check<'a>(rec: &'a Value, name: &str) -> &'a Value {
    rec["checks"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap_or_else(|| panic!("no check {name}: {}", rec["checks"]))
}

fn sha256(p: &Path) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(std::fs::read(p).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn usage_errors_list_everything_missing_at_once() {
    let o = capture(&[]);
    assert_eq!(o.status.code(), Some(2));
    let (_, err) = text(&o);
    assert!(err.starts_with("aeiou-trace capture: the following required arguments were not provided:"), "{err}");
    for want in ["-o <DIR>", "--root <DIR>", "<COMMAND>"] {
        assert!(err.contains(want), "{want}: {err}");
    }
    assert!(err.contains("Usage: aeiou-trace capture [OPTIONS] -o <DIR> -- <COMMAND>..."), "{err}");

    // a check is waived by its name alone
    let d = tmp("usage");
    let out = d.join("c");
    let o = capture(&["--root", s(&d), "-o", s(&out), "--accept", "frob", "--", "true"]);
    assert_eq!(o.status.code(), Some(2));
    let err = text(&o).1;
    assert!(err.contains("--accept frob: no such check; one of ") && err.contains("output-outside-roots"), "{err}");
    // the eBPF source's buffer: not with strace, and a size the kernel's ring buffer can be
    let o = capture(&["--source", "strace", "--buffer", "1GiB", "--root", s(&d), "-o", s(&out), "--", "true"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("--buffer: the eBPF source's ring buffer; strace has none"), "{}", text(&o).1);
    let o = capture(&["--buffer", "3MiB", "--root", s(&d), "-o", s(&out), "--", "true"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("--buffer 3MiB: a power of two from 64KiB to 2GiB"), "{}", text(&o).1);
    // --convert takes no other option, and needs a capture made under eBPF
    let o = capture(&["--convert", s(&d), "-o", s(&out)]);
    assert_eq!(o.status.code(), Some(2), "{}", text(&o).1);
    let o = capture(&["--convert", s(&d)]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("no syscalls.bin (a capture under --source ebpf keeps one"), "{}", text(&o).1);
    // several ranks: the source is named, the coordinator given
    let o = capture(&["--ranks", "2", "--rank", "0", "--root", s(&d), "-o", s(&out), "--", "true"]);
    assert_eq!(o.status.code(), Some(2));
    let err = text(&o).1;
    assert!(err.contains("--coordinator <HOST:PORT>") && err.contains("--source <SOURCE>"), "{err}");
    // an endpoint is a URL
    let o = capture(&["--endpoint", "bucket", "-o", s(&out), "--", "true"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o).1.contains("--endpoint bucket: expected a URL, SCHEME://HOST[/PATH]"), "{}", text(&o).1);
    assert!(!out.exists());

    let help = text(&Command::new(bin()).args(["capture", "--help"]).output().unwrap()).0;
    for flag in ["--source", "--root", "--endpoint", "--cold", "--accept", "--check", "--capture-id", "--ranks", "--rank", "--coordinator", "--out"] {
        assert!(help.contains(flag), "{flag}: {help}");
    }
}

#[test]
fn a_capture_writes_the_trace_and_its_record_and_exits_with_the_command_s_status() {
    let d = tmp("one");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("a.bin"), vec![7u8; 10000]).unwrap();
    let out = d.join("cap-%h-%p");
    let script = format!("cat {}/a.bin > /dev/null; exit 3", s(&root));
    let mut args = vec!["--root", s(&root), "--endpoint", "s3://bucket/pre", "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    let (_, err) = text(&o);
    assert_eq!(o.status.code(), Some(3), "{err}");
    // %h and %p expanded: the one directory there is
    let dirs: Vec<PathBuf> = std::fs::read_dir(&*d).unwrap().map(|e| e.unwrap().path()).filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("cap-")).collect();
    assert_eq!(dirs.len(), 1, "{dirs:?}");
    let dir = &dirs[0];
    let name = dir.file_name().unwrap().to_str().unwrap();
    assert!(name.starts_with(&format!("cap-{}-", aeiou::run::hostname())) && name.rsplit('-').next().unwrap().parse::<u32>().is_ok(), "{name}");
    assert!(err.contains(&format!("written: {}", dir.display())), "{err}");

    let trace = dir.join("trace.txt");
    let t = std::fs::read_to_string(&trace).unwrap();
    assert!(t.contains("a.bin") && t.contains("execve("), "{t}");
    let rec = record(dir);
    assert_eq!(rec["aeiou_capture"], 1);
    assert_eq!(rec["tool"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(rec["source"]["name"], "strace");
    let line: Vec<&str> = rec["source"]["line"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(&line[..3], ["strace", "-f", "--seccomp-bpf"]);
    let set = line.iter().find(|a| a.starts_with("trace=")).unwrap();
    assert!(set.contains("%network") && set.contains("%file"), "{set}");
    assert!(rec["source"]["version"].as_str().unwrap().chars().next().unwrap().is_ascii_digit(), "{}", rec["source"]);
    assert_eq!(rec["command"]["argv"], serde_json::json!(["sh", "-c", script]));
    assert_eq!(rec["command"]["cwd"], std::env::current_dir().unwrap().to_str().unwrap());
    assert_eq!(rec["command"]["uid"], unsafe { libc::getuid() });
    assert_eq!(rec["exit"], serde_json::json!({"status": 3}));
    assert!(rec["interrupted"].is_null());
    let (start, end) = (rec["start"]["realtime"].as_f64().unwrap(), rec["end"]["realtime"].as_f64().unwrap());
    assert!(start > 1.7e9 && end >= start, "{start} {end}");
    assert!(rec["start"]["realtime_minus_monotonic"].as_f64().unwrap() > 1.0e9);
    assert_eq!(rec["trace"]["file"], "trace.txt");
    assert_eq!(rec["trace"]["sha256"], sha256(&trace));
    assert_eq!(rec["trace"]["bytes"], t.len());
    let r = &rec["roots"][0];
    assert_eq!(r["path"], s(&root));
    assert!(r["fs_type"].is_string() && r["mount_point"].is_string() && r["options"].is_string(), "{r}");
    assert_eq!(rec["endpoints"], serde_json::json!([{"url": "s3://bucket/pre", "scheme": "s3"}]));
    assert_eq!(rec["cache"], "not dropped");
    assert_eq!(rec["losses"]["events"], 0);
    assert!(rec["mountstats"].as_str().unwrap().contains("untraced"));
    assert!(rec["not_seen"].as_array().unwrap().iter().any(|v| v.as_str().unwrap().contains("mapped")), "{}", rec["not_seen"]);
    assert!(rec["host"]["kernel"].is_string() && rec["host"]["cpus"].is_number());
    assert_eq!(rec["waivers"], serde_json::json!(["alone", "clock"]));
    for c in ["source", "root", "output-local", "output-outside-roots", "space"] {
        assert_eq!(check(&rec, c)["ok"], true, "{c}: {}", check(&rec, c));
    }
    assert_eq!(check(&rec, "cold")["ok"], true, "not asked");
    assert!(rec["inherited_descriptors"].as_array().unwrap().iter().any(|d| d["fd"] == 0));
    assert!(rec["ranks"].is_null() && rec["capture_id"].is_null());
    assert_eq!(rec["changes_at_end"], serde_json::json!([]));

    // the readers take the capture's trace.txt as they take any strace
    let o = Command::new(bin()).args(["metrics", s(&trace), "--root", s(&root)]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o).1);
    let m: Value = serde_json::from_str(&text(&o).0).unwrap();
    assert_eq!(m["total"]["bytes_read"], 10000);
}

#[test]
fn a_failed_check_starts_nothing_and_a_waiver_by_name_lets_it_run() {
    let d = tmp("checks");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let marker = d.join("ran");
    let out = root.join("cap");
    let script = format!("touch {}", s(&marker));
    let mut args = vec!["--root", s(&root), "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    let err = text(&o).1;
    assert_eq!(o.status.code(), Some(1), "{err}");
    assert!(err.contains("output-outside-roots") && err.contains("FAILED") && err.contains("nothing was started"), "{err}");
    assert!(!marker.exists() && !out.exists());

    let mut args = vec!["--root", s(&root), "-o", s(&out), "--accept", "output-outside-roots"];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
    assert!(marker.exists());
    let rec = record(&out);
    assert_eq!(rec["waivers"], serde_json::json!(["output-outside-roots", "alone", "clock"]));
    let c = check(&rec, "output-outside-roots");
    assert_eq!((c["ok"].as_bool(), c["waived"].as_bool()), (Some(false), Some(true)), "{c}");
    assert!(text(&o).1.contains("waived"), "{}", text(&o).1);

    // -o names a directory that must be empty
    std::fs::remove_file(&marker).unwrap();
    let busy = d.join("busy");
    std::fs::create_dir(&busy).unwrap();
    std::fs::write(busy.join("x"), "").unwrap();
    let mut args = vec!["--root", s(&root), "-o", s(&busy)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", &script]);
    let o = capture(&args);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).1.contains("is not empty"), "{}", text(&o).1);
    assert!(!marker.exists());

    // --cold drops the page cache, which needs root; a capture that cannot is refused by the check
    if unsafe { libc::geteuid() } != 0 {
        let cold = d.join("cold");
        let mut args = vec!["--root", s(&root), "-o", s(&cold), "--cold"];
        args.extend(WAIVE);
        args.extend(["--", "sh", "-c", &script]);
        let o = capture(&args);
        assert_eq!(o.status.code(), Some(1));
        assert!(text(&o).1.contains("cold") && text(&o).1.contains("root"), "{}", text(&o).1);
        assert!(!marker.exists());
    }
}

#[test]
fn check_alone_runs_the_checks_and_starts_nothing() {
    let d = tmp("dry");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let out = d.join("cap");
    let mut args = vec!["--check", "--root", s(&root), "-o", s(&out)];
    args.extend(WAIVE);
    let o = capture(&args);
    let err = text(&o).1;
    assert_eq!(o.status.code(), Some(0), "{err}");
    for c in ["source", "root", "clock", "output-local", "output-outside-roots", "space", "alone", "cold"] {
        assert!(err.contains(c), "{c}: {err}");
    }
    assert!(!out.exists(), "a dry check writes nothing");
    // and fails as a capture would
    let inside = root.join("in");
    let mut args = vec!["--check", "--root", s(&root), "-o", s(&inside)];
    args.extend(WAIVE);
    let o = capture(&args);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o).1);
    assert!(text(&o).1.contains("output-outside-roots"), "{}", text(&o).1);
}

#[test]
fn a_signal_sent_to_the_tool_reaches_the_command_and_the_record_says_so() {
    let d = tmp("signal");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let out = d.join("cap");
    let ready = d.join("ready");
    let script = format!("trap 'exit 7' TERM; touch {}; while :; do sleep 0.05; done", s(&ready));
    let mut c = Command::new(bin());
    c.arg("capture").args(["--root", s(&root), "-o", s(&out)]).args(WAIVE).args(["--", "sh", "-c", &script]);
    let child = c.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let until = Instant::now() + Duration::from_secs(20);
    while !ready.exists() {
        assert!(Instant::now() < until, "the command never started");
        std::thread::sleep(Duration::from_millis(20));
    }
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let o = child.wait_with_output().unwrap();
    assert_eq!(o.status.code(), Some(7), "{}", text(&o).1);
    let rec = record(&out);
    assert_eq!(rec["interrupted"], "SIGTERM");
    assert_eq!(rec["exit"], serde_json::json!({"status": 7}));
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// `n` captures of `command(rank)` under `d/data` over the coordinator, rank r's in `out(r)`.
fn ranks(d: &Path, n: usize, out: impl Fn(usize) -> PathBuf, command: impl Fn(usize) -> String) -> Vec<Output> {
    let root = d.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let coordinator = format!("127.0.0.1:{}", free_port());
    let mut children = Vec::new();
    for r in 0..n {
        let mut c = Command::new(bin());
        c.arg("capture").args(["--source", "strace", "--root", s(&root), "-o"]).arg(out(r));
        c.args(["--ranks", &n.to_string(), "--rank", &r.to_string(), "--coordinator", &coordinator]).args(WAIVE);
        c.args(["--", "sh", "-c", &command(r)]);
        children.push(c.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
        if r == 0 {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    children.into_iter().map(|c| c.wait_with_output().unwrap()).collect()
}

#[test]
fn ranks_meet_at_a_start_barrier_measure_their_clocks_and_rank_0_writes_the_set() {
    let d = tmp("ranks");
    let outs = ranks(&d, 2, |r| d.join(format!("r{r}")), |r| format!("echo {r} > /dev/null; exit {r}"));
    for (r, o) in outs.iter().enumerate() {
        assert_eq!(o.status.code(), Some(r as i32), "rank {r}: {}", text(o).1);
    }
    let recs: Vec<Value> = (0..2).map(|r| record(&d.join(format!("r{r}")))).collect();
    let arrived = recs.iter().map(|r| r["barrier"]["arrived"].as_f64().unwrap()).fold(0.0, f64::max);
    for (r, rec) in recs.iter().enumerate() {
        assert_eq!(rec["rank"], r);
        assert_eq!(rec["ranks"], 2);
        assert!(rec["barrier"]["released"].as_f64().unwrap() >= arrived, "rank {r} released before every rank arrived");
        assert!(rec["start"]["realtime"].as_f64().unwrap() >= rec["barrier"]["released"].as_f64().unwrap());
        let off = &rec["clock"]["offset_to_rank_0"];
        assert!(off["offset_s"].as_f64().unwrap().abs() < 0.05 && off["bound_s"].as_f64().unwrap() >= 0.0, "{off}");
        for c in ["release", "same-source", "mounts", "clock-offset"] {
            assert_eq!(check(rec, c)["ok"], true, "rank {r} {c}: {}", check(rec, c));
        }
    }
    let set: Value = serde_json::from_str(&std::fs::read_to_string(d.join("r0").join("capture-set.json")).unwrap()).unwrap();
    assert_eq!(set["aeiou_capture_set"], 1);
    assert_eq!(set["ranks"], 2);
    assert_eq!(set["status"], "complete");
    let nodes = set["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 2);
    for (r, n) in nodes.iter().enumerate() {
        assert_eq!(n["rank"], r);
        assert_eq!(n["directory"], s(&d.join(format!("r{r}"))));
        assert_eq!(n["source"], "strace");
        assert_eq!(n["losses"], 0);
        assert_eq!(n["exit"], serde_json::json!({"status": r}));
        assert_eq!(n["trace_sha256"], recs[r]["trace"]["sha256"]);
        assert!(n["clock_offset_s"].is_number() && n["clock_bound_s"].is_number(), "{n}");
    }
    assert!(!d.join("r1").join("capture-set.json").exists(), "rank 0's alone");
}

#[test]
fn a_check_failing_on_one_rank_starts_no_rank_and_every_rank_names_it() {
    let d = tmp("ranks-fail");
    let marker = |r: usize| d.join(format!("ran{r}"));
    // rank 1 writes its capture under the root it captures
    let outs = ranks(
        &d,
        2,
        |r| if r == 1 { d.join("data").join("r1") } else { d.join("r0") },
        |r| format!("touch {}", s(&marker(r))),
    );
    for (r, o) in outs.iter().enumerate() {
        let err = text(o).1;
        assert_eq!(o.status.code(), Some(1), "rank {r}: {err}");
        assert!(err.contains("rank 1") && err.contains("output-outside-roots"), "rank {r}: {err}");
        assert!(!marker(r).exists(), "rank {r} started its command");
    }
}

#[test]
fn a_root_that_is_no_directory_and_a_missing_strace_fail_their_checks() {
    let d = tmp("bad-root");
    let file = d.join("file");
    std::fs::write(&file, "").unwrap();
    let none = d.join("none");
    let out = d.join("cap");
    let mut args = vec!["--check", "--root", s(&file), "--root", s(&none), "-o", s(&out)];
    args.extend(WAIVE);
    let o = capture(&args);
    let err = text(&o).1;
    assert_eq!(o.status.code(), Some(1), "{err}");
    assert!(err.contains(&format!("{}: not a directory", s(&file))) && err.contains(&format!("{}: No such file", s(&none))), "{err}");

    // no strace on the PATH: the source cannot run, and the check says so
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let o = Command::new(bin()).arg("capture").args(["--check", "--root", s(&root), "-o", s(&out)]).args(WAIVE).env("PATH", s(&d)).output().unwrap();
    let err = text(&o).1;
    assert_eq!(o.status.code(), Some(1), "{err}");
    assert!(err.contains("source") && err.contains("strace: not found on the PATH"), "{err}");
}

#[test]
fn another_capture_on_the_host_fails_alone_unless_it_is_of_the_same_capture_id() {
    let d = tmp("alone");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let ready = d.join("ready");
    let script = format!("touch {}; exec sleep 30", s(&ready));
    let mut first = Command::new(bin())
        .arg("capture")
        .args(["--root", s(&root), "-o", s(&d.join("first")), "--capture-id", "job-a"])
        .args(WAIVE)
        .args(["--", "sh", "-c", &script])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(20);
    while !ready.exists() {
        assert!(Instant::now() < until, "the first capture never started");
        std::thread::sleep(Duration::from_millis(20));
    }
    let pid = first.id().to_string();
    let dry = |id: &str| capture(&["--check", "--root", s(&root), "-o", s(&d.join("second")), "--capture-id", id, "--accept", "clock"]);
    let o = dry("job-b");
    let err = text(&o).1;
    assert_eq!(o.status.code(), Some(1), "{err}");
    let line = err.lines().find(|l| l.contains(" alone ")).unwrap();
    assert!(line.contains("FAILED") && line.contains(&pid), "{err}");
    // of the same capture id it is a sibling, never named (other tests' captures may be)
    let o = dry("job-a");
    let line = text(&o).1.lines().find(|l| l.contains(" alone ")).unwrap().to_string();
    assert!(!line.split(|c: char| !c.is_ascii_digit()).any(|w| w == pid), "{line}");
    unsafe { libc::kill(first.id() as i32, libc::SIGTERM) };
    let _ = first.wait();
}

#[test]
fn a_command_a_signal_ends_ends_the_tool_by_it_once_the_record_is_written() {
    use std::os::unix::process::ExitStatusExt;
    let d = tmp("killed");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let out = d.join("cap");
    let mut args = vec!["--root", s(&root), "-o", s(&out)];
    args.extend(WAIVE);
    args.extend(["--", "sh", "-c", "kill -KILL $$"]);
    let o = capture(&args);
    assert_eq!(o.status.signal(), Some(libc::SIGKILL), "{:?} {}", o.status, text(&o).1);
    let rec = record(&out);
    assert_eq!(rec["exit"], serde_json::json!({"signal": "SIGKILL"}));
    assert!(rec["interrupted"].is_null());
}

#[test]
fn a_dry_check_on_several_ranks_compares_them_and_makes_no_directory() {
    let d = tmp("ranks-dry");
    let root = d.join("data");
    std::fs::create_dir(&root).unwrap();
    let coordinator = format!("127.0.0.1:{}", free_port());
    let children: Vec<_> = (0..2)
        .map(|r| {
            let c = Command::new(bin())
                .arg("capture")
                .args(["--check", "--source", "strace", "--root", s(&root), "-o"])
                .arg(d.join(format!("r{r}")))
                .args(["--ranks", "2", "--rank", &r.to_string(), "--coordinator", &coordinator])
                .args(WAIVE)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            if r == 0 {
                std::thread::sleep(Duration::from_millis(100));
            }
            c
        })
        .collect();
    for (r, c) in children.into_iter().enumerate() {
        let o = c.wait_with_output().unwrap();
        let err = text(&o).1;
        assert_eq!(o.status.code(), Some(0), "rank {r}: {err}");
        for c in ["release", "same-source", "mounts", "clock-offset"] {
            assert!(err.contains(c), "rank {r} {c}: {err}");
        }
        assert!(err.contains("rank 1 ("), "rank {r} prints every node's rows: {err}");
        assert!(!d.join(format!("r{r}")).exists());
    }
}

#[test]
fn a_rank_lost_during_the_capture_leaves_rank_0_s_set_incomplete() {
    let d = tmp("ranks-lost");
    // rank 1's command kills rank 1's capture tool (its grandparent: sh, strace, the tool)
    let outs = ranks(&d, 2, |r| d.join(format!("r{r}")), |r| if r == 1 { "kill -KILL $(awk '{print $4}' /proc/$PPID/stat)".to_string() } else { "sleep 1".to_string() });
    assert_eq!(outs[0].status.code(), Some(1), "{}", text(&outs[0]).1);
    let set: Value = serde_json::from_str(&std::fs::read_to_string(d.join("r0").join("capture-set.json")).unwrap()).unwrap();
    assert!(set["status"].as_str().unwrap().starts_with("incomplete: "), "{set}");
    // rank 0's own capture is whole
    assert_eq!(record(&d.join("r0"))["exit"], serde_json::json!({"status": 0}));
}
