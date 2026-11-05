//! `aeiou-trace capture --source ebpf` (`DESIGN_REVIEW.md` §3.96 "Phase 3", brief item 54):
//! the same command captured under strace and under eBPF gives the same parsed calls (every op
//! of the export, lane by lane, and the metrics), through threads, a fork, `dup`, `chdir`, and
//! relative paths; a ring buffer too small loses events and says so loudly; the command's
//! environment is the user's although the tool is secure-exec; the probes need privilege. The
//! pieces without a kernel (the table, BTF, the loss record) are the unit tests of `ebpf.rs`,
//! the converter's of `ebpf_text.rs`.
//!
//! The probes need `CAP_BPF` and `CAP_PERFMON`: each test runs a copy of the binary given them
//! with `sudo -n setcap` (password-less in CI; locally, `sudo -v` first). Never skipped: without
//! sudo the test fails and says how; `dev.sh` leaves these tests out only when
//! `AEIOU_WITHOUT` names `ebpf`.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_aeiou-trace")
}

fn tmp(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("capture-ebpf", tag)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// A copy of the built binary with the file capabilities the probes need, made once per binary
/// (named by its SHA-256, so a build that links the same bytes again keeps the grant) on the
/// temporary directory's disk.
fn privileged() -> PathBuf {
    use sha2::Digest;
    let sha: String = sha2::Sha256::digest(std::fs::read(bin()).unwrap()).iter().take(8).map(|b| format!("{b:02x}")).collect();
    let dir = std::env::temp_dir().join(format!("aeiou-capture-ebpf-bin-{sha}"));
    let exe = dir.join("aeiou-trace");
    let granted = || {
        Command::new("getcap").arg(&exe).output().is_ok_and(|o| {
            let t = String::from_utf8_lossy(&o.stdout);
            t.contains("cap_bpf") && t.contains("cap_perfmon")
        })
    };
    if exe.exists() && granted() {
        return exe;
    }
    std::fs::create_dir_all(&dir).unwrap();
    let part = dir.join(format!("aeiou-trace.{}", std::process::id()));
    std::fs::copy(bin(), &part).unwrap();
    let o = Command::new("sudo").args(["-n", "setcap", "cap_bpf,cap_perfmon+ep"]).arg(&part).output().unwrap();
    assert!(
        o.status.success(),
        "sudo -n setcap cap_bpf,cap_perfmon+ep {}: {} (the eBPF tests need it: run `sudo -v` first, or leave them out with AEIOU_WITHOUT=ebpf)",
        part.display(),
        String::from_utf8_lossy(&o.stderr).trim()
    );
    std::fs::rename(&part, &exe).unwrap();
    exe
}

/// The checks a test on a shared machine waives (`tests/capture.rs`), and `fault-around`, which
/// fails on a host at the kernel's default under the file capabilities (no root to set it;
/// `faults_on_a_mapping_are_traced_as_runs` and `the_fault_around_check_refuses_or_passes_as_the_host_is`
/// are its tests).
const WAIVE: [&str; 6] = ["--accept", "alone", "--accept", "clock", "--accept", "fault-around"];

fn capture(exe: &Path, args: &[&str]) -> Output {
    Command::new(exe).arg("capture").args(WAIVE).args(args).output().unwrap()
}

/// The host's `fault_around_bytes`, held by a test that changes it (every capture as root sets
/// it to a page for its command, then puts it back) or whose faults depend on it, for as long as
/// the test runs: tests run in parallel, and a capture as root ending put it back under a
/// capture of the faults test's (CI, 2026-11-05). A lock on a file of the temporary directory,
/// so it holds across nextest's processes; released when dropped.
fn fault_around_lock() -> std::fs::File {
    use std::os::fd::AsRawFd;
    let f = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(std::env::temp_dir().join("aeiou-capture-ebpf-fault-around.lock")).unwrap();
    assert_eq!(unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) }, 0, "{}", std::io::Error::last_os_error());
    f
}

/// A capture as root, of the built binary with no file capabilities: POSIX AIO's uprobes need
/// `CAP_SYS_ADMIN` (DESIGN_REVIEW.md §3.98 "Phase 4 step 4", brief item 68). `sudo -n`, as the
/// grant is (password-less in CI; locally after `sudo -v` in a terminal); what it wrote under
/// `own` is given back to the user after. Its test holds `fault_around_lock`.
fn capture_as_root(args: &[&str], own: &Path) -> Output {
    let o = Command::new("sudo").args(["-n", bin(), "capture"]).args(WAIVE).args(args).output().unwrap();
    assert!(!text(&o).contains("sudo: a password is required"), "the POSIX AIO tests capture as root: run `sudo -v` in a terminal first, or leave the eBPF tests out with AEIOU_WITHOUT=ebpf");
    let id = |f: &str| String::from_utf8(Command::new("id").arg(f).output().unwrap().stdout).unwrap().trim().to_string();
    let c = Command::new("sudo").args(["-n", "chown", "-R", &format!("{}:{}", id("-u"), id("-g"))]).arg(own).output().unwrap();
    assert!(c.status.success(), "{}", text(&c));
    o
}

fn record(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("capture.json")).unwrap()).unwrap()
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

/// A program of many kinds of call under `root`: writes and reads of every family, `dup`,
/// `lseek`, `fstat`, `chdir` and relative paths after it, a rename and a listing, four
/// threads in turn, a forked child, unlink and rmdir.
const PROGRAM: &str = r#"
import os, sys, threading
d = sys.argv[1]
os.makedirs(d + "/sub", exist_ok=True)
fd = os.open(d + "/a", os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
os.write(fd, b"x" * 10000); os.pwrite(fd, b"y" * 100, 20000); os.writev(fd, [b"a" * 10, b"b" * 20]); os.fsync(fd); os.close(fd)
fd = os.open(d + "/a", os.O_RDONLY)
os.read(fd, 4096); os.pread(fd, 100, 5000); os.lseek(fd, 0, os.SEEK_SET); os.readv(fd, [bytearray(10), bytearray(30)])
fd2 = os.dup(fd); os.read(fd2, 7); os.close(fd2); os.fstat(fd); os.close(fd)
os.chdir(d)
os.rename("a", "sub/b"); os.stat("sub/b"); list(os.scandir("sub"))
def w(i):
    f = os.open("t%d" % i, os.O_CREAT | os.O_RDWR, 0o600); os.write(f, b"z" * (1000 * (i + 1))); os.pread(f, 10, 0); os.close(f)
# one after another: the metrics' distances across threads depend on how threads interleave,
# which differs from run to run under any source
for i in range(4):
    t = threading.Thread(target=w, args=(i,)); t.start(); t.join()
pid = os.fork()
if pid == 0:
    f = os.open("child", os.O_CREAT | os.O_WRONLY, 0o600); os.write(f, b"c" * 333); os.close(f); os._exit(0)
os.waitpid(pid, 0)
os.unlink("sub/b"); os.rmdir("sub")
try:
    os.open("missing", os.O_RDONLY)
except OSError:
    pass
"#;

/// The export of a capture's trace without what timing decides: its header's counts, and each
/// lane's ops in order without their times, the lanes sorted (their numbers are the order
/// threads first appeared, which the scheduler decides).
fn parsed(exe_dir: &Path, root: &Path) -> (Value, Vec<Vec<Value>>) {
    let out = exe_dir.join("export.jsonl");
    let o = Command::new(bin()).args(["export", "--root", s(root), "-o", s(&out), s(exe_dir)]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let body = std::fs::read_to_string(&out).unwrap();
    let mut lines = body.lines();
    let mut head: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    head.as_object_mut().unwrap().remove("root");
    let mut lanes: Vec<Vec<Value>> = Vec::new();
    for l in lines {
        let mut v: Value = serde_json::from_str(l).unwrap();
        let lane = v["lane"].as_u64().unwrap() as usize;
        let m = v.as_object_mut().unwrap();
        for k in ["lane", "t", "dur"] {
            m.remove(k);
        }
        if lanes.len() <= lane {
            lanes.resize(lane + 1, Vec::new());
        }
        lanes[lane].push(v);
    }
    // an open's id is the order of opens across lanes, which timing decides too
    for lane in lanes.iter_mut() {
        for op in lane.iter_mut() {
            op.as_object_mut().unwrap().remove("fd");
        }
    }
    lanes.sort_by_key(|l| serde_json::to_string(l).unwrap());
    (head, lanes)
}

fn metrics(dir: &Path, root: &Path) -> Value {
    let o = Command::new(bin()).args(["metrics", "--root", s(root), s(dir)]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let mut v: Value = serde_json::from_str(&String::from_utf8_lossy(&o.stdout)).unwrap();
    v.as_object_mut().unwrap().remove("trace");
    v
}

/// One capture of the program under `source`, its data under the same root each time.
fn run(exe: &Path, source: &str, t: &Path, root: &Path, extra: &[&str]) -> (PathBuf, Output) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root).unwrap();
    let prog = t.join("program.py");
    std::fs::write(&prog, PROGRAM).unwrap();
    let out = t.join(format!("cap-{source}"));
    let mut args = vec!["--source", source, "--root", s(root), "-o", s(&out)];
    args.extend_from_slice(extra);
    args.extend(["--", "python3", s(&prog), s(root)]);
    let o = capture(exe, &args);
    (out, o)
}

#[test]
fn ebpf_gives_the_calls_strace_gives() {
    let exe = privileged();
    let t = tmp("same");
    let root = t.join("data");
    let (st, o) = run(Path::new(bin()), "strace", &t, &root, &[]);
    assert!(o.status.success(), "{}", text(&o));
    let (eb, o) = run(&exe, "ebpf", &t, &root, &[]);
    assert!(o.status.success(), "{}", text(&o));

    let rec = record(&eb);
    assert_eq!(rec["source"]["name"], "ebpf");
    assert_eq!(rec["source"]["privilege"], "cap_bpf,cap_perfmon");
    // POSIX AIO's uprobes need CAP_SYS_ADMIN: under file capabilities it is said why, and the
    // trace's POSIX AIO is strace's (brief item 68)
    assert_eq!(rec["posix_aio"]["traced"], false, "{}", rec["posix_aio"]);
    assert!(rec["posix_aio"]["why"].as_str().unwrap().contains("CAP_SYS_ADMIN"), "{}", rec["posix_aio"]);
    assert!(rec["not_seen"].to_string().contains("which thread issued a POSIX AIO request"), "{}", rec["not_seen"]);
    assert_eq!(rec["complete"], true, "{}", rec["losses"]);
    assert_eq!(rec["losses"]["events"], 0);
    assert_eq!(rec["reader"]["file"], "syscalls.bin");
    assert!(rec["reader"]["records"].as_u64().unwrap() > 100, "{}", rec["reader"]);
    assert_eq!(rec["conversion"]["exits_without_entry"], 0, "{}", rec["conversion"]);
    assert_eq!(rec["conversion"]["descriptors_unresolved"], 0, "{}", rec["conversion"]);
    assert_eq!(rec["exit"]["status"], 0);
    assert!(eb.join("syscalls.bin").is_file() && eb.join("trace.txt").is_file());
    assert_eq!(record(&st)["complete"], true);

    let (sh, sl) = parsed(&st, &root);
    let (eh, el) = parsed(&eb, &root);
    assert_eq!(eh, sh, "the exports' headers");
    assert_eq!(el.len(), 6, "the main thread, four threads, the child: {el:#?}");
    for (a, b) in sl.iter().zip(el.iter()) {
        assert_eq!(b, a, "a lane's ops: strace {a:#?}\nebpf {b:#?}");
    }
    assert_eq!(el, sl);
    assert_eq!(metrics(&eb, &root), metrics(&st, &root));

    // the trace begins at the command's exec, as strace's does, after the header of its format
    // when io_uring's requests are traced (DESIGN_REVIEW.md §3.98)
    let first = std::fs::read_to_string(eb.join("trace.txt")).unwrap();
    let mut lines = first.lines();
    let mut first = lines.next().unwrap();
    // and libaio's completions (phase 4 step 3), and POSIX AIO (step 4)
    let paio = "aeiou_aio_read aeiou_aio_write aeiou_aio_fsync aeiou_lio_listio aeiou_aio_serve aeiou_aio_suspend aeiou_aio_return aeiou_aio_cancel aeiou_aio_refused";
    let kinds = [("uring", "aeiou_uring_sqe aeiou_uring_cqe"), ("faults", "aeiou_fault"), ("aio", "aeiou_aio_issue aeiou_aio_complete"), ("posix_aio", paio)]
        .iter()
        .filter(|(k, _)| rec[*k]["traced"] == true)
        .map(|(_, v)| *v)
        .collect::<Vec<_>>()
        .join(" ");
    if !kinds.is_empty() {
        assert!(first.starts_with("0 ") && first.contains(&format!(" --- aeiou: trace format 1, records {kinds} ---")), "{first}");
        first = lines.next().unwrap();
        if rec["faults"]["first_touch_observed"] == false {
            assert!(first.starts_with("0 ") && first.contains(" --- aeiou: faults: a first touch of a page may not fault"), "{first}");
            first = lines.next().unwrap();
        }
    }
    assert!(first.contains(" execve(\"") && first.contains("python3"), "{first}");
    // the failed open, as strace prints it
    let body = std::fs::read_to_string(eb.join("trace.txt")).unwrap();
    assert!(body.contains("\"missing\", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory) <"), "{body}");
    assert!(body.contains("+++ exited with 0 +++"));
    // execve's arguments and environment are not copied: no secret of them in the trace
    assert!(!body.contains("PATH="), "the environment reached the trace");
}

#[test]
fn a_full_buffer_loses_events_and_says_so() {
    let exe = privileged();
    let t = tmp("loss");
    let root = t.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let eb = t.join("cap");
    // cached calls as fast as a thread makes them, into the smallest ring buffer
    let burst = format!("import os\nf = os.open({:?}, os.O_CREAT | os.O_RDWR)\nfor i in range(300000): os.pread(f, 1, 0)\n", s(&root.join("f")));
    let o = capture(&exe, &["--source", "ebpf", "--buffer", "64KiB", "--root", s(&root), "-o", s(&eb), "--", "python3", "-c", &burst]);
    // the command's status, as ever
    assert!(o.status.success(), "{}", text(&o));
    let err = text(&o);
    let rec = record(&eb);
    assert_eq!(rec["complete"], false);
    let lost = rec["losses"]["events"].as_u64().unwrap();
    assert!(lost > 0, "{}", rec["losses"]);
    assert!(err.contains(&format!("THE TRACE IS INCOMPLETE: {lost} event(s) lost")), "{err}");
    assert!(err.contains("--buffer ") && err.contains("--source strace"), "{err}");
    assert!(rec["losses"]["suggested_buffer"].as_u64().unwrap() >= 8192);
    assert!(!rec["losses"]["windows"].as_array().unwrap().is_empty());
    let body = std::fs::read_to_string(eb.join("trace.txt")).unwrap();
    assert!(body.contains("--- aeiou: ") && body.contains("event(s) lost on cpu"), "no loss marker");
    // the readers still read it: the marker is a line they skip
    let o = Command::new(bin()).args(["metrics", "--root", s(&root), s(&eb)]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
}

#[test]
fn the_command_keeps_the_users_environment() {
    let exe = privileged();
    let t = tmp("env");
    let root = t.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let out = t.join("cap");
    let dump = t.join("env.txt");
    // a secure-exec binary's glibc strips LD_LIBRARY_PATH and TMPDIR from its own environment
    let o = Command::new(&exe)
        .arg("capture")
        .args(WAIVE)
        .args(["--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", "sh", "-c", &format!("env > {}", s(&dump))])
        .env("LD_LIBRARY_PATH", "/nonexistent-aeiou-test")
        .env("TMPDIR", "/tmp")
        .env("AEIOU_TEST_MARK", "kept")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let env = std::fs::read_to_string(&dump).unwrap();
    for want in ["LD_LIBRARY_PATH=/nonexistent-aeiou-test", "TMPDIR=/tmp", "AEIOU_TEST_MARK=kept"] {
        assert!(env.lines().any(|l| l == want), "{want} not in the command's environment:\n{env}");
    }
}

#[test]
fn the_command_status_and_a_missing_program() {
    let exe = privileged();
    let t = tmp("status");
    let root = t.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let o = capture(&exe, &["--source", "ebpf", "--root", s(&root), "-o", s(&t.join("a")), "--", "sh", "-c", "exit 7"]);
    assert_eq!(o.status.code(), Some(7), "{}", text(&o));
    assert_eq!(record(&t.join("a"))["exit"]["status"], 7);
    let o = capture(&exe, &["--source", "ebpf", "--root", s(&root), "-o", s(&t.join("b")), "--", "no-such-program-aeiou"]);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
    assert!(text(&o).contains("no-such-program-aeiou: not found on the PATH"), "{}", text(&o));
}

#[test]
fn without_privilege_the_source_check_says_why_and_auto_takes_strace() {
    let t = tmp("unpriv");
    let root = t.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let o = capture(Path::new(bin()), &["--source", "ebpf", "--root", s(&root), "--check", "-o", s(&t.join("x"))]);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
    assert!(text(&o).contains("cap_bpf,cap_perfmon") && text(&o).contains("--source strace runs as any user"), "{}", text(&o));
    let o = capture(Path::new(bin()), &["--root", s(&root), "--check", "-o", s(&t.join("y"))]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("strace") && text(&o).contains("auto: eBPF needs root"), "{}", text(&o));
}

#[test]
fn convert_writes_the_trace_again_from_the_records() {
    let exe = privileged();
    let t = tmp("convert");
    let root = t.join("data");
    let (eb, o) = run(&exe, "ebpf", &t, &root, &[]);
    assert!(o.status.success(), "{}", text(&o));
    let before = std::fs::read(eb.join("trace.txt")).unwrap();
    std::fs::remove_file(eb.join("trace.txt")).unwrap();
    let o = Command::new(bin()).args(["capture", "--convert", s(&eb)]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    assert_eq!(std::fs::read(eb.join("trace.txt")).unwrap(), before);
}

#[test]
fn no_secret_reaches_an_ebpf_capture() {
    crate::capture::no_secret_reaches_a_capture_file(&privileged(), "ebpf", &["--accept", "fault-around"]);
}

/// A program outside the command, writing under the same root all through the capture: its
/// calls are in neither source's trace (DESIGN_REVIEW.md §3.98, "Other programs on a node").
pub const OUTSIDER: &str = r#"
import os, sys, time
d = sys.argv[1]
os.makedirs(d, exist_ok=True)
open(d + "/outsider-ready", "w").close()
i = 0
while not os.path.exists(d + "/outsider-stop"):
    with open("%s/outsider-%d" % (d, i % 8), "w") as f:
        f.write("o" * 100)
    i += 1
    time.sleep(0.002)
"#;

/// The command of the outsider tests: it forks children that exit, writes under the root, and
/// waits long enough for the outsider to write many times.
pub const INSIDER: &str = r#"
import os, sys, time
d = sys.argv[1]
for k in range(3):
    pid = os.fork()
    if pid == 0:
        with open("%s/insider-child-%d" % (d, k), "w") as f:
            f.write("i")
        os._exit(0)
    os.waitpid(pid, 0)
for k in range(20):
    with open("%s/insider-%d" % (d, k), "w") as f:
        f.write("i" * 10)
    time.sleep(0.01)
"#;

/// Runs the outsider, then `capture` (given the root and the insider's path), then stops the
/// outsider; the capture's output, and how many files the outsider wrote meanwhile.
pub fn beside_an_outsider(t: &Path, root: &Path, capture: impl FnOnce(&Path) -> Output) -> Output {
    std::fs::create_dir_all(root).unwrap();
    let (o, i) = (t.join("outsider.py"), t.join("insider.py"));
    std::fs::write(&o, OUTSIDER).unwrap();
    std::fs::write(&i, INSIDER).unwrap();
    let mut child = Command::new("python3").arg(&o).arg(root).spawn().unwrap();
    let ready = root.join("outsider-ready");
    for _ in 0..500 {
        if ready.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(ready.exists(), "the outsider did not start");
    let out = capture(&i);
    std::fs::write(root.join("outsider-stop"), b"").unwrap();
    assert!(child.wait().unwrap().success());
    out
}

#[test]
fn a_program_outside_the_command_is_not_traced_and_no_exited_process_stays_in_the_tree() {
    let exe = privileged();
    let t = tmp("others");
    let root = t.join("data");
    let out = t.join("cap");
    let o = beside_an_outsider(&t, &root, |insider| capture(&exe, &["--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", "python3", s(insider), s(&root)]));
    assert!(o.status.success(), "{}", text(&o));
    let body = std::fs::read_to_string(out.join("trace.txt")).unwrap();
    assert!(body.contains("insider-19") && body.contains("insider-child-2"), "the command's own calls: {body}");
    assert!(!body.contains("/outsider-"), "a program outside the command reached the trace");
    let rec = record(&out);
    assert_eq!(rec["complete"], true, "{}", rec["losses"]);
    assert_eq!(rec["losses"]["untraced_processes"], 0, "{}", rec["losses"]);
    // the command and its three children have exited: none is still traced
    assert_eq!(rec["processes_traced_at_end"], 0, "{rec:#}");
}

/// The runner's own `io_uring` engine as the traced application (DESIGN_REVIEW.md §3.98, brief
/// item 54 phase 4): `aeiou run --posix io_uring` of npy_per_sample's smoke set opens, reads,
/// and closes in the ring, and its capture under eBPF reads as the same run under `--posix
/// sync` does, the same reads of the same bytes of the same files and the same `fstat`s (the
/// engine's `IORING_OP_STATX` of a descriptor), with and without a kernel
/// submission thread (SQPOLL, whose requests no call of the application's submits). What each
/// thread's order decides (runs, reuse) differs between the backends by design and is not
/// compared.
#[test]
fn io_uring_requests_are_traced_and_read_as_the_calls_they_do() {
    let exe = privileged();
    let t = tmp("uring");
    let root = t.join("data");
    let m = crate::common::workloads().join("input-pipeline/map/npy-per-sample");
    let (ast, params) = (m.join("npy_per_sample.ast.json"), m.join("params/smoke.params.json"));
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let o = Command::new(aeiou).args(["datagen", s(&ast), "--params-file", s(&params), "--root", s(&root)]).env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let dry = t.join("dry.json");
    let o = Command::new(aeiou).args(["dry-run", s(&ast), "--params-file", s(&params), "--gpus", "1", "--metrics-json", s(&dry)]).env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let dry: Value = serde_json::from_str(&std::fs::read_to_string(&dry).unwrap()).unwrap();
    // what the order of a thread's calls does not decide
    let fixed = |m: &Value| {
        let t = &m["total"];
        serde_json::json!({
            "bytes_read": t["bytes_read"], "bytes_written": t["bytes_written"],
            "read": t["counts"]["read"], "open": t["counts"]["open"], "close": t["counts"]["close"],
            "fstat": t["counts"]["fstat"], "stat": t["counts"]["stat"],
            "request_size": t["request_size"], "popularity_bytes": t["popularity_bytes"], "popularity_objects": t["popularity_objects"],
        })
    };
    let mut seen = Vec::new();
    for (tag, posix, extra) in [("sync", "sync", &[][..]), ("uring", "io_uring", &[][..]), ("sqpoll", "io_uring", &["--sqpoll", "10"][..])] {
        let out = t.join(format!("cap-{tag}"));
        let mut args = vec!["--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", aeiou, "run", s(&ast), "--params-file", s(&params), "--param", "step_time=0", "--gpus", "1", "--root", s(&root), "--posix", posix];
        args.extend_from_slice(extra);
        let o = Command::new(&exe).arg("capture").args(WAIVE).args(&args).env_remove("AEIOU_CONFIG").output().unwrap();
        assert!(o.status.success(), "{tag}: {}", text(&o));
        let rec = record(&out);
        assert_eq!(rec["complete"], true, "{tag}: {}", rec["losses"]);
        assert_eq!(rec["uring"]["traced"], true, "{tag}: {}", rec["uring"]);
        let not_seen = rec["not_seen"].to_string();
        assert!(!not_seen.contains("the submission queue's entries are not") && not_seen.contains("SQPOLL"), "{tag}: {not_seen}");
        assert_eq!(rec["conversion"]["uring_completions_unmatched"], 0, "{tag}: {}", rec["conversion"]);
        let body = std::fs::read_to_string(out.join("trace.txt")).unwrap();
        let sqes = body.lines().filter(|l| l.contains(" aeiou_uring_sqe(") && l.contains("op=IORING_OP_READ,")).count();
        if posix == "sync" {
            assert_eq!(sqes, 0, "{tag}");
        } else {
            assert!(sqes as u64 >= dry["total"]["counts"]["read"].as_u64().unwrap(), "{tag}: {sqes} reads in the ring");
            assert!(body.contains("op=IORING_OP_OPENAT, dirfd=") && body.contains(" aeiou_uring_cqe("), "{tag}");
        }
        let got = metrics(&out, &root);
        if posix != "sync" {
            // the engine's statx in the ring is noted by name, never dropped in silence
            assert!(got["notes"].as_object().unwrap().keys().all(|k| !k.starts_with("not_counted:aeiou_uring_sqe:IORING_OP_READ")), "{tag}: {}", got["notes"]);
        }
        seen.push((tag, fixed(&got)));
    }
    // the dry run counts the lengths the abstract asks for, a read past a file's end included; a
    // trace the bytes each call returned
    let (got, asked) = (seen[0].1["bytes_read"].as_u64().unwrap(), dry["total"]["bytes_read"].as_u64().unwrap());
    assert!(got > 0 && got <= asked, "the sync run read {got} of the {asked} bytes the dry run asks");
    for (tag, f) in &seen[1..] {
        assert_eq!(f, &seen[0].1, "{tag} against sync");
    }
}

/// The runner's own libaio engine as the traced application (DESIGN_REVIEW.md §3.98 "Phase 4
/// step 3"): `aeiou run --posix libaio` of npy_per_sample's smoke set, through the page cache
/// and with `--cache direct`, captured under eBPF, reads as the same run under `--posix sync`
/// does with the same cache, every request numbered after its `io_submit` and answered by its
/// completion, none unmatched or left in flight; through the page cache every completion is
/// posted inside the `io_submit` that took it (buffered kernel AIO is synchronous), and with
/// `--cache direct` some are posted after it returned.
#[test]
fn libaio_completions_are_traced_and_read_as_the_calls_they_do() {
    let exe = privileged();
    let t = tmp("aio");
    let root = t.join("data");
    let m = crate::common::workloads().join("input-pipeline/map/npy-per-sample");
    let (ast, params) = (m.join("npy_per_sample.ast.json"), m.join("params/smoke.params.json"));
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let o = Command::new(aeiou).args(["datagen", s(&ast), "--params-file", s(&params), "--root", s(&root)]).env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let fixed = |m: &Value| {
        let t = &m["total"];
        serde_json::json!({
            "bytes_read": t["bytes_read"], "bytes_written": t["bytes_written"],
            "read": t["counts"]["read"], "open": t["counts"]["open"], "close": t["counts"]["close"],
            "request_size": t["request_size"], "popularity_bytes": t["popularity_bytes"], "popularity_objects": t["popularity_objects"],
        })
    };
    for cache in ["per-open", "direct"] {
        let mut seen = Vec::new();
        for posix in ["sync", "libaio"] {
            let tag = format!("{posix}-{cache}");
            let out = t.join(format!("cap-{tag}"));
            let args = ["--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", aeiou, "run", s(&ast), "--params-file", s(&params), "--param", "step_time=0", "--gpus", "1", "--root", s(&root), "--posix", posix, "--cache", cache];
            let o = Command::new(&exe).arg("capture").args(WAIVE).args(args).env_remove("AEIOU_CONFIG").output().unwrap();
            assert!(o.status.success(), "{tag}: {}", text(&o));
            let rec = record(&out);
            assert_eq!(rec["complete"], true, "{tag}: {}", rec["losses"]);
            assert_eq!((&rec["aio"]["traced"], &rec["aio"]["completions_at"]), (&serde_json::json!(true), &serde_json::json!("aio_complete")), "{tag}: {}", rec["aio"]);
            assert!(!rec["not_seen"].to_string().contains("when a libaio request completed"), "{tag}: {}", rec["not_seen"]);
            let c = &rec["conversion"];
            assert_eq!((c["aio_completions_unmatched"].as_u64(), c["aio_in_flight_at_end"].as_u64()), (Some(0), Some(0)), "{tag}: {c}");
            let body = std::fs::read_to_string(out.join("trace.txt")).unwrap();
            let (issues, completes) = (body.lines().filter(|l| l.contains(" aeiou_aio_issue(")).count() as u64, body.lines().filter(|l| l.contains(" aeiou_aio_complete(")).count() as u64);
            let got = metrics(&out, &root);
            let notes = got["notes"].as_object().unwrap();
            assert!(notes.get("aio_results_assumed_complete").is_none(), "{tag}: {notes:?}");
            if posix == "sync" {
                assert_eq!((issues, completes), (0, 0), "{tag}");
            } else {
                assert!(issues > 0 && issues == completes && issues == c["aio_requests"].as_u64().unwrap(), "{tag}: {issues} issued, {completes} completed, {c}");
                let inside = c["aio_completed_in_submit"].as_u64().unwrap();
                assert_eq!(notes.get("aio_completed_in_submit").and_then(Value::as_u64).unwrap_or(0), inside, "{tag}: {notes:?}");
                if cache == "per-open" {
                    assert_eq!(inside, issues, "{tag}: a buffered request completes inside its io_submit");
                } else {
                    assert!(inside < issues, "{tag}: {inside} of {issues} direct requests completed inside their io_submit");
                }
                eprintln!("{tag}: {issues} requests, {inside} completed inside io_submit, never reaped by a call: {:?}", notes.get("aio_completed_never_reaped"));
            }
            seen.push((tag, fixed(&got)));
        }
        assert_eq!(seen[1].1, seen[0].1, "{} against {}", seen[1].0, seen[0].0);
    }
}

/// The runner's own POSIX AIO engine as the traced application (DESIGN_REVIEW.md §3.98 "Phase 4
/// step 4"): `aeiou run --posix posix-aio` of npy_per_sample's smoke set, captured under eBPF,
/// reads as the same run under `--posix sync` does, every request served by a glibc helper's
/// call and collected by `aio_return`, none ambiguous or left in flight, and the export's data in
/// as many lanes as the sync run's (each actor's thread its own), none in a glibc helper's.
#[test]
fn posix_aio_requests_are_traced_and_read_as_the_calls_they_do() {
    let _held = fault_around_lock();
    let t = tmp("paio");
    let root = t.join("data");
    let m = crate::common::workloads().join("input-pipeline/map/npy-per-sample");
    let (ast, params) = (m.join("npy_per_sample.ast.json"), m.join("params/smoke.params.json"));
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let o = Command::new(aeiou).args(["datagen", s(&ast), "--params-file", s(&params), "--root", s(&root)]).env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let fixed = |m: &Value| {
        let t = &m["total"];
        serde_json::json!({
            "bytes_read": t["bytes_read"], "bytes_written": t["bytes_written"],
            "read": t["counts"]["read"], "open": t["counts"]["open"], "close": t["counts"]["close"],
            "request_size": t["request_size"], "popularity_bytes": t["popularity_bytes"], "popularity_objects": t["popularity_objects"],
        })
    };
    let mut seen = Vec::new();
    let mut data_lanes = 0;
    for posix in ["sync", "posix-aio"] {
        let out = t.join(format!("cap-{posix}"));
        let args = ["--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", aeiou, "run", s(&ast), "--params-file", s(&params), "--param", "step_time=0", "--gpus", "1", "--root", s(&root), "--posix", posix];
        let o = capture_as_root(&args, &t);
        assert!(o.status.success(), "{posix}: {}", text(&o));
        let rec = record(&out);
        assert_eq!(rec["complete"], true, "{posix}: {}", rec["losses"]);
        assert_eq!(rec["posix_aio"]["traced"], true, "{posix}: {}", rec["posix_aio"]);
        assert!(rec["source"]["line"][0].as_str().unwrap().contains("(POSIX AIO)"), "{}", rec["source"]["line"]);
        assert!(!rec["not_seen"].to_string().contains("which thread issued a POSIX AIO request"), "{posix}: {}", rec["not_seen"]);
        let c = &rec["conversion"];
        let n = c["posix_aio_requests"].as_u64().unwrap();
        assert_eq!(
            (c["posix_aio_served"].as_u64(), c["posix_aio_ambiguous"].as_u64(), c["posix_aio_never_collected"].as_u64(), c["posix_aio_in_flight_at_end"].as_u64(), c["posix_aio_libraries_not_traced"].as_array().map(Vec::len)),
            (Some(n), Some(0), Some(0), Some(0), Some(0)),
            "{posix}: {c}"
        );
        let body = std::fs::read_to_string(out.join("trace.txt")).unwrap();
        let serves = body.lines().filter(|l| l.contains(" aeiou_aio_serve(")).count() as u64;
        let got = metrics(&out, &root);
        let (_, lanes) = parsed(&out, &root);
        let with_data = lanes.iter().filter(|l| l.iter().any(|op| op["op"] == "submit" || op["op"] == "read")).count();
        if posix == "sync" {
            assert_eq!((n, serves), (0, 0), "{posix}");
            data_lanes = with_data;
        } else {
            assert!(n > 0 && serves == n, "{posix}: {n} requests, {serves} served lines, {c}");
            let notes = got["notes"].as_object().unwrap();
            assert!(notes.keys().all(|k| !k.starts_with("posix_aio_")), "{posix}: {notes:?}");
            // the reads in the actors' lanes, as the sync run's are: a helper's would be a lane more
            assert_eq!(with_data, data_lanes, "{posix}: {lanes:#?}");
            eprintln!("{posix}: {n} requests served by glibc's helpers");
        }
        seen.push((posix, fixed(&got)));
    }
    assert_eq!(seen[1].1, seen[0].1, "{} against {}", seen[1].0, seen[0].0);
}

/// POSIX AIO through the C library, by ctypes: two writes waited for by `aio_suspend`, one to an
/// `O_APPEND` file at an offset the kernel ignores, an `aio_fsync(O_DSYNC)`, a `lio_listio`
/// (`LIO_WAIT`) of two reads with a null entry between them, and a read.
const PAIO: &str = r#"
import ctypes, os, struct, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)
d = sys.argv[1]
def cb(fd, op, buf, n, off):
    b = ctypes.create_string_buffer(168)
    struct.pack_into("ii", b, 0, fd, op)
    struct.pack_into("QQ", b, 16, ctypes.addressof(buf), n)
    struct.pack_into("i", b, 44, 1)
    struct.pack_into("q", b, 128, off)
    return b
def wait(c):
    lst = (ctypes.c_void_p * 1)(ctypes.addressof(c))
    while libc.aio_error(c) == 115:
        libc.aio_suspend(lst, 1, None)
    return libc.aio_return(c)
fd = os.open(d + "/f", os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
fa = os.open(d + "/g", os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_TRUNC, 0o644)
bufs = [ctypes.create_string_buffer(b"w" * 4096, 4096) for _ in range(4)]
ws = [cb(fd, 1, bufs[i], 4096, i * 4096) for i in range(2)]
for c in ws:
    assert libc.aio_write(c) == 0
for c in ws:
    assert wait(c) == 4096
a = cb(fa, 1, bufs[2], 100, 999999)
assert libc.aio_write(a) == 0 and wait(a) == 100
s = cb(fd, 0, bufs[3], 0, 0)
assert libc.aio_fsync(os.O_DSYNC, s) == 0 and wait(s) == 0
rs = [cb(fd, 0, bufs[i], 4096, i * 4096) for i in range(2)]
lst = (ctypes.c_void_p * 3)(ctypes.addressof(rs[0]), None, ctypes.addressof(rs[1]))
assert libc.lio_listio(0, lst, 3, None) == 0
for c in rs:
    assert libc.aio_return(c) == 4096
r = cb(fd, 0, bufs[0], 1000, 0)
assert libc.aio_read(r) == 0 and wait(r) == 1000
os.close(fd)
os.close(fa)
"#;

/// Every POSIX AIO function the probes take, through ctypes (`PAIO`): under eBPF each request is
/// numbered and served, collected, and counted in the issuing thread, the `lio_listio` one line
/// of its two reads, the fsync the issuer's `fdatasync`; the metrics are strace's, which counts
/// the same calls as the helpers' own.
#[test]
fn every_posix_aio_function_is_traced() {
    let _held = fault_around_lock();
    let t = tmp("paio-fns");
    let root = t.join("data");
    let prog = t.join("paio.py");
    std::fs::write(&prog, PAIO).unwrap();
    let mut got = Vec::new();
    for source in ["strace", "ebpf"] {
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let out = t.join(format!("cap-{source}"));
        let args = ["--source", source, "--root", s(&root), "-o", s(&out), "--", "python3", s(&prog), s(&root)];
        let o = if source == "ebpf" { capture_as_root(&args, &t) } else { capture(Path::new(bin()), &args) };
        assert!(o.status.success(), "{source}: {}", text(&o));
        assert_eq!(std::fs::metadata(root.join("g")).unwrap().len(), 100, "the append");
        let m = metrics(&out, &root);
        let total = &m["total"];
        assert_eq!((total["bytes_read"].as_u64(), total["bytes_written"].as_u64(), total["counts"]["fdatasync"].as_u64()), (Some(8192 + 1000), Some(8192 + 100), Some(1)), "{source}: {total}");
        got.push(total.clone());
        if source == "strace" {
            assert!(record(&out)["not_seen"].to_string().contains("which thread issued a POSIX AIO request"));
            continue;
        }
        let rec = record(&out);
        let c = &rec["conversion"];
        assert_eq!(
            (c["posix_aio_requests"].as_u64(), c["posix_aio_served"].as_u64(), c["posix_aio_never_collected"].as_u64(), c["posix_aio_in_flight_at_end"].as_u64(), c["posix_aio_ambiguous"].as_u64()),
            (Some(7), Some(7), Some(0), Some(0), Some(0)),
            "{c}"
        );
        let body = std::fs::read_to_string(out.join("trace.txt")).unwrap();
        let count = |k: &str| body.lines().filter(|l| l.contains(&format!(" {k}("))).count();
        assert_eq!((count("aeiou_aio_write"), count("aeiou_aio_fsync"), count("aeiou_lio_listio"), count("aeiou_aio_read"), count("aeiou_aio_serve"), count("aeiou_aio_return")), (3, 1, 1, 1, 7, 7), "{body}");
        let lio = body.lines().find(|l| l.contains(" aeiou_lio_listio(")).unwrap();
        assert!(lio.contains("mode=LIO_WAIT") && lio.matches("op=LIO_READ").count() == 2 && lio.contains("nent=3"), "{lio}");
        assert!(body.lines().any(|l| l.contains(" aeiou_aio_fsync(") && l.contains("op=O_DSYNC")), "{body}");
        assert!(body.lines().any(|l| l.contains(" aeiou_aio_suspend(ids=[")), "{body}");
        // the data ops in the python thread's lane alone
        let (_, lanes) = parsed(&out, &root);
        let with_data = lanes.iter().filter(|l| l.iter().any(|op| matches!(op["op"].as_str(), Some("submit" | "read" | "write" | "fdatasync")))).count();
        assert_eq!(with_data, 1, "{lanes:#?}");
    }
    assert_eq!(got[1]["bytes_read"], got[0]["bytes_read"]);
    assert_eq!(got[1]["counts"], got[0]["counts"]);
}

/// A file written a page at a time (so it is cached in folios of a page: a large write caches
/// large folios, and a fault maps the whole folio), mapped and read a page at a time, in two
/// halves 50 ms apart, then one store.
const MAPPED: &str = r#"
import mmap, os, sys, time
d, page, n = sys.argv[1], mmap.PAGESIZE, 128
fd = os.open(d + "/m", os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
for i in range(n):
    os.pwrite(fd, b"m" * page, i * page)
m = mmap.mmap(fd, n * page, mmap.MAP_SHARED, mmap.PROT_READ)
s = 0
for i in range(0, n // 2):
    s += m[i * page]
time.sleep(0.05)
for i in range(n // 2, n):
    s += m[i * page]
m.close()
w = mmap.mmap(fd, page, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
w[0] = 1
w.close()
os.close(fd)
"#;

/// Faults on a mapping (DESIGN_REVIEW.md §3.98 "Phase 4 step 2"): the file's faults in the trace
/// as runs (each half one, the pause between them past `--fault-gap`, the store its own),
/// counted by the metrics apart from the calls and not replayed by the export; every page
/// where the host's fault-around is one page (the check's verdict), fault-around's stride where
/// not; a longer `--fault-gap` makes the halves one run.
#[test]
fn faults_on_a_mapping_are_traced_as_runs() {
    let _held = fault_around_lock();
    let exe = privileged();
    let t = tmp("faults");
    let root = t.join("data");
    let prog = t.join("mapped.py");
    std::fs::write(&prog, MAPPED).unwrap();
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let mut runs_of = Vec::new();
    let mut observed_all = true;
    for (tag, gap) in [("default", None), ("long-gap", Some("200ms"))] {
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let out = t.join(format!("cap-{tag}"));
        let mut args = vec!["--source", "ebpf", "--root", s(&root), "-o", s(&out)];
        if let Some(g) = gap {
            args.extend(["--fault-gap", g]);
        }
        args.extend(["--", "python3", s(&prog), s(&root)]);
        let o = capture(&exe, &args);
        assert!(o.status.success(), "{tag}: {}", text(&o));
        let rec = record(&out);
        assert_eq!(rec["complete"], true, "{tag}: {}", rec["losses"]);
        assert_eq!(rec["faults"]["traced"], true, "{tag}: {}", rec["faults"]);
        assert_eq!(rec["faults"]["gap_ns"], if gap.is_some() { 200_000_000 } else { 1_000_000 });
        assert!(rec["source"]["line"][0].as_str().unwrap().contains("fentry,fexit:handle_mm_fault"), "{}", rec["source"]["line"]);
        let observed = rec["faults"]["first_touch_observed"] == true;
        // the check taken again at the end, where the start passed (held by the lock: no capture
        // as root changes the setting under this one)
        assert_eq!(rec["faults"]["first_touch_at_end"] == true, observed, "{tag}: {}", rec["faults"]);
        observed_all &= observed;
        let not_seen = rec["not_seen"].to_string();
        assert!(!not_seen.contains("the page faults they cause are not system calls") && not_seen.contains("after its first"), "{not_seen}");
        assert_eq!(not_seen.contains("the fault-around check was waived"), !observed, "{not_seen}");
        let body = std::fs::read_to_string(out.join("trace.txt")).unwrap();
        let file = format!("aeiou_fault(path=\"{}/m\"", std::fs::canonicalize(&root).unwrap().display());
        assert!(body.contains(&file), "{tag}: {body}");
        let m = metrics(&out, &root);
        let f = &m["faults"];
        // a warm cache: no fault read the disk
        assert_eq!((f["major"].as_u64(), f["write"].as_u64(), f["in_call"].as_u64()), (Some(0), Some(1), Some(0)), "{tag}: {f}");
        let strides: Vec<i64> = f["stride"].as_object().unwrap().keys().map(|k| k.parse().unwrap()).collect();
        if observed {
            assert_eq!(f["faults"], 129, "{tag}: every page once, and the store: {f}");
            assert_eq!(strides, [page as i64], "{tag}: {f}");
        } else {
            // the kernel mapped pages ahead (fault-around, large folios: on this box at the
            // default, one fault mapped all 128 pages): the first touch and the store at least
            let n = f["faults"].as_u64().unwrap();
            assert!((2..129).contains(&n), "{tag}: {f}");
            assert!(strides.iter().all(|&s| s > page as i64), "{tag}: {f}");
        }
        assert!(m["notes"]["faults_kernel_observed"].is_string());
        assert_eq!(m["notes"].get("faults_first_touch_unseen").is_some(), !observed, "{}", m["notes"]);
        assert_eq!(rec["conversion"]["faults_failed"], 0, "{}", rec["conversion"]);
        runs_of.push(f["runs"].as_u64().unwrap());
        // the export keeps the mappings as reads of their ranges and says what it does not replay
        let exp = t.join(format!("export-{tag}.jsonl"));
        let o = Command::new(bin()).args(["export", "--root", s(&root), "-o", s(&exp), s(&out)]).output().unwrap();
        assert!(o.status.success(), "{}", text(&o));
        let head: Value = serde_json::from_str(std::fs::read_to_string(&exp).unwrap().lines().next().unwrap()).unwrap();
        assert_eq!(head["notes"]["faults_not_replayed"], f["faults"], "{head}");
    }
    // the halves, and the store, as runs; a gap longer than the pause joins the halves (where
    // every page faults: where not, which touches fault is how the kernel cached the file that
    // time, and a run is the first touch's and the store's at least)
    if observed_all {
        assert_eq!(runs_of, [3, 2]);
    } else {
        assert!(runs_of.iter().all(|&r| r >= 2), "{runs_of:?}");
    }
}

/// `MAPPED`'s first half, then the host's fault-around made 16 pages (the program runs as root),
/// then its second half: the change another capture as root, ending, or an administrator makes.
const MAPPED_THEN_AROUND: &str = r#"
import mmap, os, sys
d, page, n = sys.argv[1], mmap.PAGESIZE, 128
fd = os.open(d + "/m", os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
for i in range(n):
    os.pwrite(fd, b"m" * page, i * page)
m = mmap.mmap(fd, n * page, mmap.MAP_SHARED, mmap.PROT_READ)
s = 0
for i in range(0, n // 2):
    s += m[i * page]
with open("/sys/kernel/debug/fault_around_bytes", "w") as f:
    f.write(str(16 * page))
for i in range(n // 2, n):
    s += m[i * page]
m.close()
os.close(fd)
"#;

/// A host's fault-around changed during a capture (DESIGN_REVIEW.md §3.98 "Phase 4 step 2", found
/// in CI 2026-11-05): the check, passed at the start, is taken again when the command ends and
/// fails, so the record, `not_seen`, the trace's line, and the metrics say a first touch may not
/// have faulted and why, and `--convert` writes the same line again from the record. As root:
/// the setting is the host's, put back as the test found it.
#[test]
fn a_fault_around_changed_during_the_capture_is_said() {
    let _held = fault_around_lock();
    const AT: &str = "/sys/kernel/debug/fault_around_bytes";
    let sudo = |args: &[&str]| {
        let o = Command::new("sudo").arg("-n").args(args).output().unwrap();
        assert!(o.status.success(), "sudo -n {args:?}: {}", text(&o));
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    };
    let before = sudo(&["cat", AT]);
    let t = tmp("around-changed");
    let root = t.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let prog = t.join("mapped.py");
    std::fs::write(&prog, MAPPED_THEN_AROUND).unwrap();
    let out = t.join("cap");
    let args = ["--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", "python3", s(&prog), s(&root)];
    let o = capture_as_root(&args, &t);
    sudo(&["sh", "-c", &format!("echo {before} > {AT}")]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("fault-around changed during the capture"), "{}", text(&o));
    let rec = record(&out);
    let f = &rec["faults"];
    assert_eq!((&f["traced"], &f["first_touch_observed"], &f["first_touch_at_end"]), (&Value::Bool(true), &Value::Bool(false), &Value::Bool(false)), "{f}");
    let not_seen = rec["not_seen"].to_string();
    assert!(not_seen.contains("changed during the capture") && !not_seen.contains("the fault-around check was waived"), "{not_seen}");
    let line = |out: &Path| std::fs::read_to_string(out.join("trace.txt")).unwrap().lines().find(|l| l.contains("--- aeiou: faults: ")).map(str::to_string);
    let l = line(&out).unwrap();
    assert!(l.contains("a first touch of a page may not fault") && l.contains("changed during the capture"), "{l}");
    assert!(metrics(&out, &root)["notes"]["faults_first_touch_unseen"].is_number());
    let o = Command::new(bin()).args(["capture", "--convert", s(&out)]).output().unwrap();
    assert!(o.status.success(), "{}", text(&o));
    assert_eq!(line(&out).unwrap(), l);
}

/// `fault-around` fails where a first touch may not fault and nothing is started, saying how
/// to set it; passes where it does.
#[test]
fn the_fault_around_check_refuses_or_passes_as_the_host_is() {
    let _held = fault_around_lock();
    let exe = privileged();
    let t = tmp("fault-around");
    let root = t.join("data");
    std::fs::create_dir_all(&root).unwrap();
    let out = t.join("cap");
    let o = Command::new(&exe).args(["capture", "--accept", "alone", "--accept", "clock", "--source", "ebpf", "--root", s(&root), "-o", s(&out), "--", "true"]).output().unwrap();
    if aeiou::capture::first_touch_faults(&t).unwrap() {
        assert!(o.status.success(), "{}", text(&o));
        assert_eq!(record(&out)["faults"]["first_touch_observed"], true);
    } else {
        assert!(!o.status.success(), "{}", text(&o));
        let msg = text(&o);
        assert!(msg.contains("fault-around") && msg.contains("/sys/kernel/debug/fault_around_bytes") && msg.contains("nothing was started"), "{msg}");
        assert!(!out.join("trace.txt").exists());
    }
}
