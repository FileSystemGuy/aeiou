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

/// The checks a test on a shared machine waives (`tests/capture.rs`).
const WAIVE: [&str; 4] = ["--accept", "alone", "--accept", "clock"];

fn capture(exe: &Path, args: &[&str]) -> Output {
    Command::new(exe).arg("capture").args(WAIVE).args(args).output().unwrap()
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

    // the trace begins at the command's exec, as strace's does
    let first = std::fs::read_to_string(eb.join("trace.txt")).unwrap();
    let first = first.lines().next().unwrap();
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
    crate::capture::no_secret_reaches_a_capture_file(&privileged(), "ebpf", &[]);
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
