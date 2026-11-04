//! `aeiou-trace capture` (`DESIGN_REVIEW.md` §3.96, brief item 54, phase 1): start a command
//! under a trace and keep, beside the trace, the record of what was captured and how. The
//! source in phase 1 is strace (`strace -f --seccomp-bpf ...`, the man page's line with
//! `%network`); eBPF is phase 3; phase 2, the HTTP-head shim and the anchors, is `requests.rs`.
//!
//! The checks are the capture's first step, every time: each node checks itself (`node_checks`),
//! rank 0 compares the nodes (`cross_checks`), and any check failing on any node, not waived by
//! name with `--accept`, starts nothing anywhere. On several nodes the capture joins the
//! runner's coordinator (`coord.rs`): `Hello` carries the node's document (its facts and its
//! checks) as its options block, rank 0 measures every rank's clock against its own
//! (`Server::clocks`), judges, and either stops every rank with the table or shares the table
//! with `Start`. Then each node makes its directory and drops its cache, and the barrier
//! `capture-start` releases every command together: none starts before every node is ready. At
//! the end each rank sends its summary with `Done`, rank 0 measures the clocks again and writes
//! `capture-set.json`.
//!
//! The capture directory (`-o DIR`, new or empty; `%h` the host, `%p` the pid): `trace.txt`,
//! the strace text every reader reads, and `capture.json` (`aeiou_capture: 1`), the record.
//! Under `--requests shim` (phase 2, the default with an `--endpoint`) the command also runs
//! with the capture's library preloaded (`requests.rs`), and the directory holds its log,
//! `requests.bin`, and the requests made of it, `requests.jsonl`; with an `--endpoint` an
//! anchor is sent to each at the start and the end, with or without the library.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::coord::{Coordinator, Server, Tcp};
use crate::run::{hostname, unix_now};

/// Every check, by the name `--accept` waives it by. The first nine each node makes of
/// itself; the last four rank 0 makes across the nodes.
pub const CHECKS: [&str; 13] = ["source", "root", "clock", "output-local", "output-outside-roots", "space", "alone", "cold", "requests", "release", "same-source", "mounts", "clock-offset"];

/// The calls strace traces: the man page's set (`aeiou-trace(1)`), and `%network`, since an
/// owner cannot know whether descriptors pass over a socket (`SCM_RIGHTS`) or an object client
/// talks to the store.
pub const TRACE_SET: &str = "trace=%file,%desc,%process,%network,sync,msync,madvise,munmap,mremap,mlock,mlock2,io_setup,io_submit,io_getevents,io_pgetevents,io_cancel,io_destroy";

/// The free space below which `space` fails: a trace of a busy job grows by hundreds of
/// megabytes a minute.
pub const MIN_FREE: u64 = 1 << 30;

/// A clock further from rank 0's than this, beyond what the measurement can tell, fails
/// `clock-offset`: a synchronized cluster is within a millisecond.
pub const MAX_OFFSET_S: f64 = 0.010;

/// Rounds of the clock exchange per rank (`Server::clocks`).
pub const CLOCK_ROUNDS: usize = 8;

/// What a capture under strace cannot see, recorded in every record.
pub const NOT_SEEN_STRACE: [&str; 4] = [
    "loads and stores to mapped memory (mmap): the page faults they cause are not system calls",
    "io_uring operations: io_uring_enter is traced, the submission queue's entries are not",
    "object-store requests: the sockets' calls are traced, the HTTP heads are not (the capture's library, --requests shim, or the store's own log)",
    "which reads the page cache served",
];

/// What a capture under eBPF cannot see beyond what strace cannot, recorded in every record.
pub const NOT_SEEN_EBPF: [&str; 4] = [
    "the lengths of vectored AIO requests (IOCB_CMD_PREADV, IOCB_CMD_PWRITEV): their iovec arrays are not copied, the requests are left out of trace.txt and counted (conversion.vectored_aio_left_out)",
    "execve's arguments and environment: not copied (the path is), so no secret in them reaches the trace",
    "the descriptors a recvmsg receives (SCM_RIGHTS), and those pipe, pipe2, and socketpair return: calls on them show no path",
    "the paths of descriptors as the kernel named them: they are resolved from the recorded opens, their directories' symbolic links as they are when the capture is converted",
];

/// The strace source's line, less `-E` and `-o`: `-v` prints arrays whole, but
/// `-e verbose=!execve,execveat` prints execve's argument list and environment as their
/// addresses (as the eBPF converter does) and `-s 0` no call's bytes, so neither a secret in
/// them nor anything a read or a write carries reaches the trace (brief item 62,
/// `DESIGN_REVIEW.md` §3.97); paths, `-yy`'s, and arrays are printed whole.
pub fn strace_line() -> Vec<String> {
    ["strace", "-f", "--seccomp-bpf", "-ttt", "-T", "-yy", "-v", "-e", "verbose=!execve,execveat", "-s", "0", "-e", TRACE_SET].iter().map(|s| s.to_string()).collect()
}

/// The words of an option or a variable whose value is taken for a secret (case ignored).
const SECRET_WORDS: [&str; 10] = ["password", "passwd", "secret", "token", "key", "credential", "auth", "signature", "session", "cookie"];
pub const REDACTED: &str = "<redacted>";

/// `name` an option's or a variable's name (`--NAME`, `NAME`; letters, digits, `_.-`), not a
/// script's text, holding a word of `SECRET_WORDS`.
fn secret_name(name: &str) -> bool {
    let bare = name.trim_start_matches('-');
    if name.len() - bare.len() > 2 || bare.is_empty() || !bare.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)) {
        return false;
    }
    let n = bare.to_ascii_lowercase();
    SECRET_WORDS.iter().any(|w| n.contains(w))
}

/// `u` with the password of its user part (`scheme://user:password@host`) replaced.
pub fn scrub_url(u: &str) -> String {
    let Some(s) = u.find("://") else { return u.to_string() };
    let rest = &u[s + 3..];
    let host_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let Some(at) = rest[..host_end].rfind('@') else { return u.to_string() };
    let Some(colon) = rest[..at].find(':') else { return u.to_string() };
    format!("{}{REDACTED}{}", &u[..s + 3 + colon + 1], &rest[at..])
}

/// The command line as a record keeps it (brief item 62): the value of a `NAME=value`, a
/// `--NAME=value`, or the argument after a `--NAME` (unless itself an option) whose name holds
/// a word of `SECRET_WORDS`, and a URL's password, replaced by `<redacted>`; and the indices
/// replaced. By name and pattern alone: a secret passed by position is not found
/// (`--argv none`).
pub fn scrub_argv(argv: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut out = Vec::with_capacity(argv.len());
    let mut at = Vec::new();
    let mut value_next = false;
    for (i, a) in argv.iter().enumerate() {
        let s = if i == 0 {
            a.clone()
        } else if value_next && !a.starts_with('-') {
            REDACTED.to_string()
        } else {
            match a.split_once('=') {
                Some((name, _)) if secret_name(name) => format!("{name}={REDACTED}"),
                _ => scrub_url(a),
            }
        };
        value_next = i > 0 && a.starts_with('-') && !a.contains('=') && secret_name(a);
        if s != *a {
            at.push(i);
        }
        out.push(s);
    }
    (out, at)
}

/// The environment a record keeps, a fixed list, never the whole environment.
pub const ENVIRONMENT: [&str; 7] = ["SLURM_JOB_ID", "SLURM_NODEID", "SLURM_PROCID", "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR"];

/// File systems a capture directory must not be on: the trace is written for every traced call,
/// and on shared storage it is load on the system under test.
pub fn network_fs(fs_type: &str) -> bool {
    const NET: [&str; 18] = ["nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "ceph", "glusterfs", "9p", "virtiofs", "panfs", "wekafs", "orangefs", "afs", "ncpfs", "pvfs2"];
    NET.contains(&fs_type) || fs_type == "fuse" || fs_type.starts_with("fuse.")
}

// ---------------------------------------------------------------- mounts

/// One line of `/proc/self/mountinfo`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mount {
    pub mount_point: String,
    pub fs_type: String,
    pub source: String,
    /// The mount's own options (field 6), then the superblock's (after the separator).
    pub options: String,
    pub super_options: String,
}

/// `\040` and its kind, as the kernel escapes a space, tab, newline, and backslash.
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 3 < b.len() && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c)) {
            out.push((b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0'));
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The mounts of a mountinfo text, in its order (a later one is mounted over an earlier one at
/// the same point). Lines that are not mountinfo's are left out.
pub fn mountinfo(text: &str) -> Vec<Mount> {
    text.lines()
        .filter_map(|l| {
            let (left, right) = l.split_once(" - ")?;
            let f: Vec<&str> = left.split(' ').collect();
            let r: Vec<&str> = right.split(' ').collect();
            if f.len() < 6 || r.len() < 3 {
                return None;
            }
            Some(Mount { mount_point: unescape(f[4]), fs_type: r[0].to_string(), source: unescape(r[1]), options: f[5].to_string(), super_options: r[2].to_string() })
        })
        .collect()
}

/// `path` is `base` or below it, by whole components.
pub fn under(path: &Path, base: &Path) -> bool {
    path.starts_with(base)
}

/// The mount `path` (absolute, resolved) is on: the longest mount point it is under, the last
/// of those when several are mounted at one point.
pub fn mount_of<'a>(mounts: &'a [Mount], path: &Path) -> Option<&'a Mount> {
    let mut best: Option<&Mount> = None;
    for m in mounts {
        if under(path, Path::new(&m.mount_point)) && best.is_none_or(|b| Path::new(&m.mount_point).components().count() >= Path::new(&b.mount_point).components().count()) {
            best = Some(m);
        }
    }
    best
}

fn mounts_now() -> Vec<Mount> {
    mountinfo(&std::fs::read_to_string("/proc/self/mountinfo").unwrap_or_default())
}

/// The options two nodes' mounts of one root are compared by: a client's own address
/// (`clientaddr`, NFS) differs from node to node by design.
pub fn comparable_options(m: &Value) -> String {
    let all = format!("{},{}", m["options"].as_str().unwrap_or(""), m["super_options"].as_str().unwrap_or(""));
    let mut v: Vec<&str> = all.split(',').filter(|o| !o.is_empty() && !o.starts_with("clientaddr=")).collect();
    v.sort();
    v.dedup();
    v.join(",")
}

/// A root's record: the path as given, resolved, and its mount (NFS's server and export split
/// out of the source).
fn root_record(path: &str, mounts: &[Mount]) -> (Value, Option<String>) {
    let resolved = match std::fs::canonicalize(path) {
        Ok(p) if p.is_dir() => p,
        Ok(p) => return (json!({"path": path, "resolved": p}), Some(format!("{path}: not a directory"))),
        Err(e) => return (json!({"path": path}), Some(format!("{path}: {e}"))),
    };
    let Some(m) = mount_of(mounts, &resolved) else {
        return (json!({"path": path, "resolved": resolved}), Some(format!("{path}: no mount in /proc/self/mountinfo")));
    };
    let mut r = json!({
        "path": path,
        "resolved": resolved,
        "mount_point": m.mount_point,
        "fs_type": m.fs_type,
        "source": m.source,
        "options": m.options,
        "super_options": m.super_options,
    });
    if m.fs_type.starts_with("nfs") {
        if let Some((server, export)) = m.source.split_once(':') {
            r["server"] = json!(server);
            r["export"] = json!(export);
        }
    }
    (r, None)
}

/// `path` made absolute and resolved as far as it exists, the rest appended lexically: where a
/// directory not yet made will be.
pub fn resolve_new(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(path) };
    let mut rest = Vec::new();
    let mut p = abs.as_path();
    loop {
        if let Ok(r) = std::fs::canonicalize(p) {
            let mut out = r;
            for c in rest.iter().rev() {
                out.push(c);
            }
            return out;
        }
        match (p.file_name(), p.parent()) {
            (Some(n), Some(parent)) => {
                rest.push(n.to_os_string());
                p = parent;
            }
            _ => return abs,
        }
    }
}

/// `path` without `.` and `..`, lexically (for a path not yet made).
fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------- the clock

/// What the kernel says of this host's clock (`adjtimex` with no modes, unprivileged): the
/// clock state, synchronized or not (`TIME_ERROR`, or `STA_UNSYNC` set), and its error bounds.
pub fn clock_state() -> Value {
    const TIME_ERROR: i32 = 5;
    const STA_UNSYNC: i32 = 0x0040;
    let mut t: libc::timex = unsafe { std::mem::zeroed() };
    let state = unsafe { libc::adjtimex(&mut t) };
    if state < 0 {
        return json!({"synchronized": null, "error": std::io::Error::last_os_error().to_string()});
    }
    json!({
        "synchronized": state != TIME_ERROR && t.status & STA_UNSYNC == 0,
        "state": state,
        "maxerror_us": t.maxerror,
        "esterror_us": t.esterror,
    })
}

/// The wall clock minus the monotonic clock, seconds: what turns a monotonic time into a wall
/// time, and how far the wall clock was stepped between two readings.
pub fn realtime_minus_monotonic() -> f64 {
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    unix_now() - (ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9)
}

// ---------------------------------------------------------------- the checks

/// One check's verdict, of one node (`node`: its rank and host, none on one host) or across
/// the nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    pub name: String,
    pub node: Option<String>,
    pub ok: bool,
    pub detail: String,
    pub waived: bool,
}

impl Check {
    pub fn new(name: &str, ok: bool, detail: impl Into<String>) -> Check {
        Check { name: name.into(), node: None, ok, detail: detail.into(), waived: false }
    }

    pub fn json(&self) -> Value {
        json!({"name": self.name, "node": self.node, "ok": self.ok, "waived": self.waived, "detail": self.detail})
    }

    pub fn of(v: &Value) -> Check {
        Check {
            name: v["name"].as_str().unwrap_or("").into(),
            node: v["node"].as_str().map(str::to_string),
            ok: v["ok"].as_bool().unwrap_or(false),
            detail: v["detail"].as_str().unwrap_or("").into(),
            waived: v["waived"].as_bool().unwrap_or(false),
        }
    }

    /// Failed and not waived: the capture cannot start.
    pub fn fails(&self) -> bool {
        !self.ok && !self.waived
    }
}

/// Mark the failed checks `accept` names as waived.
pub fn waive(checks: &mut [Check], accept: &[String]) {
    for c in checks.iter_mut() {
        if !c.ok && accept.contains(&c.name) {
            c.waived = true;
        }
    }
}

/// The table every capture prints: one row per check, the node, the verdict, and why.
pub fn table(checks: &[Check]) -> String {
    let nodes = checks.iter().any(|c| c.node.is_some());
    let wn = checks.iter().map(|c| c.node.as_deref().unwrap_or("all").len()).max().unwrap_or(0);
    let wc = checks.iter().map(|c| c.name.len()).max().unwrap_or(0);
    let mut out = String::new();
    for c in checks {
        let verdict = if c.ok {
            "ok"
        } else if c.waived {
            "waived"
        } else {
            "FAILED"
        };
        if nodes {
            out += &format!("  {:<wn$}  {:<wc$}  {:<6}  {}\n", c.node.as_deref().unwrap_or("all"), c.name, verdict, c.detail);
        } else {
            out += &format!("  {:<wc$}  {:<6}  {}\n", c.name, verdict, c.detail);
        }
    }
    out
}

/// What a node checks of itself with: the options that bear on the checks.
pub struct NodeInput<'a> {
    pub roots: &'a [String],
    pub out: &'a Path,
    pub cold: bool,
    pub capture_id: Option<&'a str>,
    pub source: &'a Source,
    /// The library asked for (`--requests shim`).
    pub requests: bool,
}

/// The source a capture traces with, and why it was chosen.
#[derive(Debug, Clone)]
pub struct Source {
    pub name: String,
    pub version: Option<String>,
    pub why: String,
    /// The source cannot run here, and why.
    pub broken: Option<String>,
}

/// The strace on the `PATH`: its version, and whether it can trace a child with the line's
/// filter here (ptrace forbidden, `kernel.yama.ptrace_scope` 3, or a seccomp filter refused).
pub fn strace_source(why: &str) -> Source {
    let version = std::process::Command::new("strace").arg("-V").output().ok().filter(|o| o.status.success()).and_then(|o| {
        let t = String::from_utf8_lossy(&o.stdout).into_owned();
        t.lines().next().and_then(|l| l.rsplit(' ').next()).map(str::to_string)
    });
    let broken = match &version {
        None => Some("strace: not found on the PATH, or `strace -V` failed".to_string()),
        Some(_) => match std::process::Command::new("strace").args(["-f", "--seccomp-bpf", "-o", "/dev/null", "--", "true"]).stdin(std::process::Stdio::null()).output() {
            Ok(o) if o.status.success() => None,
            Ok(o) => Some(format!("strace cannot trace a child here: {}", String::from_utf8_lossy(&o.stderr).trim())),
            Err(e) => Some(format!("strace: {e}")),
        },
    };
    Source { name: "strace".into(), version, why: why.into(), broken }
}

/// The eBPF source: this binary's probes, loaded and passed by the verifier with this
/// process's privilege (then unloaded), or why they cannot be.
pub fn ebpf_source(why: &str) -> Source {
    let (version, broken) = match crate::ebpf::usable() {
        Ok(how) => (Some(format!("aeiou-trace {} ({how})", env!("CARGO_PKG_VERSION"))), None),
        Err(e) => (None, Some(e)),
    };
    Source { name: "ebpf".into(), version, why: why.into(), broken }
}

/// Another capture running on this host: its pid and capture id (its `--capture-id`, else its
/// `SLURM_JOB_ID` where its environment is readable).
fn other_captures() -> Vec<(i32, Option<String>)> {
    let me = std::process::id() as i32;
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else { return out };
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<i32>().ok()) else { continue };
        if pid == me {
            continue;
        }
        let Ok(cmdline) = std::fs::read(e.path().join("cmdline")) else { continue };
        let argv: Vec<String> = cmdline.split(|&b| b == 0).map(|a| String::from_utf8_lossy(a).into_owned()).collect();
        if argv.len() < 2 || Path::new(&argv[0]).file_name().is_none_or(|n| n != "aeiou-trace") || argv[1] != "capture" {
            continue;
        }
        let dashdash = argv.iter().position(|a| a == "--").unwrap_or(argv.len());
        let own = &argv[..dashdash];
        if own.iter().any(|a| a == "--check") {
            continue;
        }
        let mut id = own.iter().position(|a| a == "--capture-id").and_then(|i| own.get(i + 1).cloned()).or_else(|| own.iter().find_map(|a| a.strip_prefix("--capture-id=").map(str::to_string)));
        if id.is_none() {
            id = std::fs::read(e.path().join("environ"))
                .ok()
                .and_then(|env| env.split(|&b| b == 0).find_map(|kv| std::str::from_utf8(kv).ok().and_then(|kv| kv.strip_prefix("SLURM_JOB_ID=")).map(str::to_string)));
        }
        out.push((pid, id));
    }
    out
}

/// Free bytes for an unprivileged user on the file system `path` (resolved, may not exist yet)
/// will be on.
fn free_bytes(path: &Path) -> Option<u64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).ok()?;
    let mut v: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut v) } == 0).then(|| v.f_bavail as u64 * v.f_frsize as u64)
}

/// The checks a node makes of itself, and the roots' records they read.
pub fn node_checks(i: &NodeInput<'_>) -> (Vec<Check>, Vec<Value>, usize) {
    let mounts = mounts_now();
    let mut checks = Vec::new();
    let src = i.source;
    checks.push(match &src.broken {
        None => Check::new("source", true, format!("{} {} ({})", src.name, src.version.as_deref().unwrap_or("?"), src.why)),
        Some(why) => Check::new("source", false, why.clone()),
    });

    let mut roots = Vec::new();
    let mut bad = Vec::new();
    for r in i.roots {
        let (rec, err) = root_record(r, &mounts);
        if let Some(e) = err {
            bad.push(e);
        }
        roots.push(rec);
    }
    checks.push(if !bad.is_empty() {
        Check::new("root", false, bad.join("; "))
    } else if roots.is_empty() {
        Check::new("root", true, "no --root (object endpoints only)")
    } else {
        Check::new("root", true, roots.iter().map(|r| format!("{} on {} {} at {}", r["path"].as_str().unwrap_or(""), r["fs_type"].as_str().unwrap_or(""), r["source"].as_str().unwrap_or(""), r["mount_point"].as_str().unwrap_or(""))).collect::<Vec<_>>().join("; "))
    });

    let clock = clock_state();
    checks.push(match clock["synchronized"].as_bool() {
        Some(true) => Check::new("clock", true, format!("synchronized, estimated error {} µs, at most {} µs", clock["esterror_us"], clock["maxerror_us"])),
        Some(false) => Check::new("clock", false, "the kernel says the clock is not synchronized (adjtimex): start chrony or ntpd, or the traces' times cannot be compared across nodes or with the store's log"),
        None => Check::new("clock", false, format!("adjtimex: {}", clock["error"].as_str().unwrap_or("failed"))),
    });

    let out = lexical(&resolve_new(i.out));
    checks.push(match mount_of(&mounts, &out) {
        Some(m) if network_fs(&m.fs_type) => Check::new("output-local", false, format!("{} is on {} ({} at {}): a trace written to shared storage is load on it; write it to a node-local disk", out.display(), m.fs_type, m.source, m.mount_point)),
        Some(m) => Check::new("output-local", true, format!("{} on {}", out.display(), m.fs_type)),
        None => Check::new("output-local", false, format!("{}: no mount in /proc/self/mountinfo", out.display())),
    });
    let inside: Vec<&str> = roots.iter().filter(|r| r["resolved"].as_str().is_some_and(|p| under(&out, Path::new(p)))).filter_map(|r| r["path"].as_str()).collect();
    checks.push(if inside.is_empty() {
        Check::new("output-outside-roots", true, format!("{} is under no --root", out.display()))
    } else {
        Check::new("output-outside-roots", false, format!("{} is under --root {}: the trace's own writes would be in the trace", out.display(), inside.join(", ")))
    });
    checks.push(match free_bytes(&out) {
        Some(n) if n >= MIN_FREE => Check::new("space", true, format!("{} free", crate::dryrun::human_bytes(n))),
        Some(n) => Check::new("space", false, format!("{} free under {}, less than {}", crate::dryrun::human_bytes(n), out.display(), crate::dryrun::human_bytes(MIN_FREE))),
        None => Check::new("space", false, format!("statvfs {}: failed", out.display())),
    });

    let others = other_captures();
    let (siblings, strangers): (Vec<_>, Vec<_>) = others.iter().partition(|(_, id)| i.capture_id.is_some() && id.as_deref() == i.capture_id);
    checks.push(if strangers.is_empty() {
        Check::new("alone", true, if siblings.is_empty() { "no other capture on this host".to_string() } else { format!("{} other capture(s) of this capture id on this host", siblings.len()) })
    } else {
        let pids: Vec<String> = strangers.iter().map(|(p, _)| p.to_string()).collect();
        Check::new("alone", false, format!("another capture runs on this host (pid {}): every probe and every tracer slows every traced call on the host", pids.join(", ")))
    });

    checks.push(if !i.cold {
        Check::new("cold", true, "not asked: the page cache is as the host left it")
    } else if unsafe { libc::geteuid() } == 0 {
        Check::new("cold", true, "sync, then 3 into /proc/sys/vm/drop_caches before the command starts")
    } else {
        Check::new("cold", false, "--cold drops the page cache, which needs root (/proc/sys/vm/drop_caches); run the capture as root, or drop the cache beforehand and waive cold")
    });
    checks.push(if !i.requests {
        Check::new("requests", true, "not asked: no library preloaded (--requests none)")
    } else {
        match crate::requests::library() {
            Ok(p) => Check::new("requests", true, format!("{} preloaded into the command", p.display())),
            Err(e) => Check::new("requests", false, e),
        }
    });
    (checks, roots, siblings.len() + 1)
}

/// What rank 0 compares across the nodes: their documents by rank (`Hello`'s options block)
/// and their clocks against its own.
pub fn cross_checks(nodes: &BTreeMap<i64, Value>, offsets: &BTreeMap<i64, (f64, f64)>) -> Vec<Check> {
    let name = |r: &i64| format!("rank {r} ({})", nodes[r]["host"].as_str().unwrap_or("?"));
    let mut checks = Vec::new();
    let differ = |field: &dyn Fn(&Value) -> String| -> Vec<String> {
        let r0 = nodes.values().next().map(field).unwrap_or_default();
        nodes.iter().filter(|(_, n)| field(n) != r0).map(|(r, n)| format!("{} has {}, rank 0 {}", name(r), field(n), r0)).collect()
    };
    let release = differ(&|n: &Value| n["release"].as_str().unwrap_or("?").to_string());
    checks.push(if release.is_empty() { Check::new("release", true, format!("aeiou-trace {} on every node", nodes.values().next().map(|n| n["release"].as_str().unwrap_or("?")).unwrap_or("?"))) } else { Check::new("release", false, release.join("; ")) });
    let source = differ(&|n: &Value| format!("{} {}, requests {}", n["source"]["name"].as_str().unwrap_or("?"), n["source"]["version"].as_str().unwrap_or("?"), n["source"]["requests"].as_str().unwrap_or("none")));
    checks.push(if source.is_empty() {
        Check::new("same-source", true, "one source and version on every node")
    } else {
        Check::new("same-source", false, format!("{}: one node under a slower source slows a synchronized job and distorts every node's trace", source.join("; ")))
    });
    let roots = nodes.values().next().and_then(|n| n["roots"].as_array()).map_or(0, Vec::len);
    let mut mounts = Vec::new();
    for i in 0..roots {
        let m = differ(&|n: &Value| {
            let r = &n["roots"][i];
            format!("{} {} {}", r["fs_type"].as_str().unwrap_or("?"), r["source"].as_str().unwrap_or("?"), comparable_options(r))
        });
        if !m.is_empty() {
            let path = nodes.values().next().map(|n| n["roots"][i]["path"].as_str().unwrap_or("?").to_string()).unwrap_or_default();
            mounts.push(format!("{path}: {}", m.join("; ")));
        }
    }
    checks.push(if mounts.is_empty() { Check::new("mounts", true, "every --root mounted alike on every node") } else { Check::new("mounts", false, mounts.join("; ")) });
    let far: Vec<String> = offsets.iter().filter(|(_, (o, b))| o.abs() - b > MAX_OFFSET_S).map(|(r, (o, b))| format!("{} is {:+.3} ms from rank 0 (±{:.3} ms)", name(r), o * 1e3, b * 1e3)).collect();
    let worst = offsets.values().map(|(o, _)| o.abs()).fold(0.0, f64::max);
    checks.push(if far.is_empty() {
        Check::new("clock-offset", true, format!("every clock within {:.3} ms of rank 0's", worst * 1e3))
    } else {
        Check::new("clock-offset", false, format!("{}; more than {} ms apart", far.join("; "), MAX_OFFSET_S * 1e3))
    });
    checks
}

// ---------------------------------------------------------------- the capture

/// The options of a capture, checked by the command line.
pub struct Opts {
    pub source: Source,
    pub roots: Vec<String>,
    pub endpoints: Vec<(String, String)>,
    pub cold: bool,
    pub accept: Vec<String>,
    pub check_only: bool,
    /// The capture's library preloaded (`--requests shim`), and its log's size.
    pub requests: bool,
    pub requests_buffer: u64,
    /// The eBPF source's ring buffer (`--buffer`).
    pub buffer: u64,
    pub capture_id: Option<String>,
    pub ranks: i64,
    pub rank: i64,
    pub coordinator: Option<String>,
    /// `-o` with `%h` and `%p` expanded.
    pub out: PathBuf,
    pub command: Vec<String>,
    /// `--argv recorded`: the command line kept, its secrets redacted; `none`: its program alone.
    pub argv_recorded: bool,
}

/// `-o`'s template: `%h` the host name, `%p` this process's id, `%%` a percent sign.
pub fn expand_out(template: &str, host: &str, pid: u32) -> std::result::Result<String, String> {
    let mut out = String::new();
    let mut it = template.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('h') => out += host,
            Some('p') => out += &pid.to_string(),
            Some('%') => out.push('%'),
            other => return Err(format!("-o {template}: %{}: only %h (the host), %p (the pid), and %% expand", other.map(String::from).unwrap_or_default())),
        }
    }
    Ok(out)
}

/// A signal's name (`SIGTERM`), or its number.
pub fn signal_name(n: i32) -> String {
    let names = [(1, "SIGHUP"), (2, "SIGINT"), (3, "SIGQUIT"), (4, "SIGILL"), (6, "SIGABRT"), (7, "SIGBUS"), (8, "SIGFPE"), (9, "SIGKILL"), (10, "SIGUSR1"), (11, "SIGSEGV"), (12, "SIGUSR2"), (13, "SIGPIPE"), (14, "SIGALRM"), (15, "SIGTERM")];
    names.iter().find(|(k, _)| *k == n).map_or_else(|| format!("signal {n}"), |(_, v)| v.to_string())
}

/// The descriptors the command inherits: this process's open ones without close-on-exec, and
/// what each is.
fn inherited_descriptors() -> Vec<Value> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc/self/fd") else { return out };
    let mut fds: Vec<i32> = dir.flatten().filter_map(|e| e.file_name().to_str().and_then(|n| n.parse().ok())).collect();
    fds.sort();
    for fd in fds {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 || flags & libc::FD_CLOEXEC != 0 {
            continue;
        }
        let target = std::fs::read_link(format!("/proc/self/fd/{fd}")).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        out.push(json!({"fd": fd, "target": target}));
    }
    out
}

fn distribution() -> Option<String> {
    let t = std::fs::read_to_string("/etc/os-release").ok()?;
    t.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=")).map(|v| v.trim_matches('"').to_string())
}

fn sha256_file(p: &Path) -> Result<(u64, String)> {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    let mut f = std::fs::File::open(p).with_context(|| p.display().to_string())?;
    let n = std::io::copy(&mut f, &mut h)?;
    Ok((n, h.finalize().iter().map(|b| format!("{b:02x}")).collect()))
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(v)? + "\n").with_context(|| tmp.display().to_string())?;
    std::fs::rename(&tmp, path).with_context(|| path.display().to_string())
}

/// The signal the tool was sent, for the waiting loop (0 when none), and whether the kernel
/// sent it (a terminal's, which the whole foreground process group got).
static SIGNO: AtomicI32 = AtomicI32::new(0);
static FROM_KERNEL: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(signo: libc::c_int, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
    let code = if info.is_null() { 0 } else { unsafe { (*info).si_code } };
    FROM_KERNEL.store(code == libc::SI_KERNEL, Ordering::SeqCst);
    SIGNO.store(signo, Ordering::SeqCst);
}

fn catch(signo: libc::c_int) {
    let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
    sa.sa_sigaction = on_signal as *const () as usize;
    sa.sa_flags = libc::SA_SIGINFO;
    unsafe {
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(signo, &sa, std::ptr::null_mut());
    }
}

/// The processes whose parent is `pid`.
fn children_of(pid: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else { return out };
    for e in dir.flatten() {
        let Some(p) = e.file_name().to_str().and_then(|n| n.parse::<i32>().ok()) else { continue };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        // pid (comm) state ppid ...: comm may hold spaces and parentheses, so after the last ')'
        let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) else { continue };
        if rest.split_whitespace().nth(1).and_then(|x| x.parse::<i32>().ok()) == Some(pid) {
            out.push(p);
        }
    }
    out
}

/// The command forked and held before its `exec` until `release` (the write end of a pipe the
/// child reads) is written to: its pid, as this process sees it, and `release`. Everything the
/// child needs is made before the fork, so it calls only `read`, `execve`, and `_exit`.
fn spawn_held(command: &[String], env_extra: &[(String, String)]) -> Result<(i32, i32)> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let prog = find_program(&command[0])?;
    let prog = CString::new(prog.as_os_str().as_bytes())?;
    let argv: Vec<CString> = command.iter().map(|a| CString::new(a.as_bytes())).collect::<std::result::Result<_, _>>()?;
    let mut envp: Vec<CString> = Vec::new();
    for (k, v) in original_env() {
        if env_extra.iter().any(|(e, _)| e.as_bytes() == k.as_bytes()) {
            continue;
        }
        let mut kv = k.as_bytes().to_vec();
        kv.push(b'=');
        kv.extend_from_slice(v.as_bytes());
        envp.push(CString::new(kv)?);
    }
    for (k, v) in env_extra {
        envp.push(CString::new(format!("{k}={v}"))?);
    }
    let mut argv_p: Vec<*const libc::c_char> = argv.iter().map(|a| a.as_ptr()).collect();
    argv_p.push(std::ptr::null());
    let mut envp_p: Vec<*const libc::c_char> = envp.iter().map(|a| a.as_ptr()).collect();
    envp_p.push(std::ptr::null());
    let mut fds = [0i32; 2];
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        bail!("pipe2: {}", std::io::Error::last_os_error());
    }
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        unsafe {
            let mut b = 0u8;
            let r = loop {
                let r = libc::read(fds[0], &mut b as *mut u8 as *mut libc::c_void, 1);
                if r >= 0 || *libc::__errno_location() != libc::EINTR {
                    break r;
                }
            };
            if r != 1 {
                libc::_exit(1);
            }
            libc::execve(prog.as_ptr(), argv_p.as_ptr(), envp_p.as_ptr());
            libc::_exit(127);
        }
    }
    unsafe { libc::close(fds[0]) };
    Ok((pid, fds[1]))
}

/// The environment this process was started with. Under file capabilities the process is
/// secure-exec, and glibc strips `LD_PRELOAD`, `LD_LIBRARY_PATH`, `TMPDIR`, and others from its
/// own copy at start, which would make the command's another run: the strings the kernel
/// placed are the user's, untouched.
fn original_env() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    use std::os::unix::ffi::OsStrExt;
    if unsafe { libc::getauxval(libc::AT_SECURE) } == 0 {
        return std::env::vars_os().collect();
    }
    // a secure-exec process is not dumpable, so its /proc/self/environ is root's to read: the
    // strings themselves, where the kernel put them (fields 50 and 51 of /proc/self/stat),
    // which glibc's removals leave in place (it moves only the pointers to them)
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let f: Vec<&str> = stat.rsplit_once(')').map_or("", |(_, r)| r).split_whitespace().collect();
    let (Some(a), Some(b)) = (f.get(47).and_then(|x| x.parse::<usize>().ok()), f.get(48).and_then(|x| x.parse::<usize>().ok())) else { return std::env::vars_os().collect() };
    if b <= a {
        return std::env::vars_os().collect();
    }
    let raw = unsafe { std::slice::from_raw_parts(a as *const u8, b - a) };
    raw.split(|&b| b == 0)
        .filter(|kv| !kv.is_empty())
        .filter_map(|kv| {
            let i = kv.iter().position(|&b| b == b'=')?;
            Some((std::ffi::OsStr::from_bytes(&kv[..i]).to_os_string(), std::ffi::OsStr::from_bytes(&kv[i + 1..]).to_os_string()))
        })
        .collect()
}

/// A command's program as `execvp` finds it: a name with a `/` as it is, else the first
/// executable file of that name on the `PATH`.
fn find_program(name: &str) -> Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    if name.contains('/') {
        return Ok(PathBuf::from(name));
    }
    let path = original_env().into_iter().find(|(k, _)| k == "PATH").map(|(_, v)| v).unwrap_or_else(|| "/usr/bin:/bin".into());
    for d in std::env::split_paths(&path) {
        let p = d.join(name);
        if std::fs::metadata(&p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0) {
            return Ok(p);
        }
    }
    bail!("{name}: not found on the PATH")
}

/// How a process ended, from `waitpid`'s status.
fn exit_of(status: i32) -> Value {
    if libc::WIFEXITED(status) {
        json!({"status": libc::WEXITSTATUS(status)})
    } else if libc::WIFSIGNALED(status) {
        json!({"signal": signal_name(libc::WTERMSIG(status))})
    } else {
        json!({"status": null})
    }
}

/// The barrier every rank's command waits at.
const START: &str = "capture-start";

/// The coordinator of a capture on several nodes: this rank's client, and rank 0's server.
struct Joined {
    tcp: Tcp,
    server: Option<Server>,
}

/// What a node sends in `Hello`: its facts, its source, its roots, its checks.
fn node_doc(o: &Opts, roots: &[Value], checks: &[Check]) -> Value {
    json!({
        "host": hostname(),
        "release": env!("CARGO_PKG_VERSION"),
        "source": {"name": o.source.name, "version": o.source.version, "requests": if o.requests { "shim" } else { "none" }},
        "roots": roots,
        "out": o.out,
        "checks": checks.iter().map(Check::json).collect::<Vec<_>>(),
    })
}

/// Print to standard error, prefixed.
fn say(msg: &str) {
    let mut e = std::io::stderr().lock();
    for l in msg.lines() {
        let _ = writeln!(e, "aeiou-trace capture: {l}");
    }
}

/// Run the capture: the checks, then (unless `--check`) the command under the source. The
/// command's exit status (or the signal that ended it, raised again here once the record is
/// written, as strace does) is the tool's.
pub fn run(o: Opts) -> Result<i32> {
    let host = hostname();
    if o.out.exists() && std::fs::read_dir(&o.out).map(|mut d| d.next().is_some()).unwrap_or(true) {
        bail!("{}: is not empty: a capture directory is new or empty", o.out.display());
    }
    let (mut checks, roots, on_host) = node_checks(&NodeInput { roots: &o.roots, out: &o.out, cold: o.cold, capture_id: o.capture_id.as_deref(), source: &o.source, requests: o.requests });
    waive(&mut checks, &o.accept);
    if unsafe { libc::geteuid() } == 0 && !o.check_only {
        say("WARNING: running as root: the command runs as root too, which is another run of an application that runs as a user (the record says so)");
    }

    // several nodes: every node's checks, and rank 0's comparison of them, before any command
    let mut joined = None;
    let mut offset = None;
    if o.ranks > 1 {
        let addr = o.coordinator.as_deref().expect("checked");
        let server = if o.rank == 0 { Some(Server::start(addr, o.ranks)?) } else { None };
        let identity = json!({"capture_id": o.capture_id, "check": o.check_only});
        for c in checks.iter_mut() {
            c.node = Some(format!("rank {} ({host})", o.rank));
        }
        let tcp = Tcp::connect(addr, o.rank, o.ranks, &host, &identity, &node_doc(&o, &roots, &checks), &[(START.to_string(), 1)], Arc::new(AtomicBool::new(false)))?;
        if let Some(s) = &server {
            let verdict = (|| -> Result<Value> {
                s.all_facts()?;
                let nodes: BTreeMap<i64, Value> = s.hosts().into_iter().map(|(r, _, doc)| (r, doc)).collect();
                let offsets = s.clocks(CLOCK_ROUNDS)?;
                let mut all: Vec<Check> = nodes.values().flat_map(|n| n["checks"].as_array().cloned().unwrap_or_default()).map(|c| Check::of(&c)).collect();
                let mut cross = cross_checks(&nodes, &offsets);
                waive(&mut cross, &o.accept);
                all.extend(cross);
                let offsets: BTreeMap<String, Value> = offsets.iter().map(|(r, (off, b))| (r.to_string(), json!({"offset_s": off, "bound_s": b}))).collect();
                Ok(json!({"checks": all.iter().map(Check::json).collect::<Vec<_>>(), "offsets": offsets}))
            })();
            let verdict = match verdict {
                Ok(v) => v,
                Err(e) => {
                    s.stop(&format!("{e:#}"));
                    return Err(e);
                }
            };
            let all: Vec<Check> = verdict["checks"].as_array().unwrap().iter().map(Check::of).collect();
            if all.iter().any(Check::fails) {
                let msg = format!("a check failed; nothing was started on any rank:\n{}", table(&all));
                s.stop(&msg);
                bail!("{msg}");
            }
            s.share(verdict);
        }
        let started = tcp.ready_shared();
        let (_, _, shared) = started.map_err(|e| anyhow!("{e:#}"))?;
        checks = shared["checks"].as_array().cloned().unwrap_or_default().iter().map(Check::of).collect();
        offset = Some(shared["offsets"][o.rank.to_string()].clone());
        joined = Some(Joined { tcp, server });
    } else if checks.iter().any(Check::fails) {
        bail!("a check failed; nothing was started:\n{}", table(&checks));
    }
    let waived: Vec<&Check> = checks.iter().filter(|c| c.waived).collect();
    say(&format!("checks:\n{}", table(&checks)));
    let waivers: Vec<String> = {
        let mut w: Vec<String> = Vec::new();
        for n in &o.accept {
            if !w.contains(n) {
                w.push(n.clone());
            }
        }
        w
    };
    if !waived.is_empty() {
        say(&format!("waived by --accept: {}", waived.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")));
    }

    if o.check_only {
        if let Some(j) = joined {
            finish(&j, &o, json!({"rank": o.rank, "host": host, "check": true}), None)?;
        }
        return Ok(0);
    }

    std::fs::create_dir_all(&o.out).with_context(|| o.out.display().to_string())?;
    let out = std::fs::canonicalize(&o.out)?;
    let trace = out.join("trace.txt");
    let inherited = inherited_descriptors();
    let cache_drop = if o.cold { Some(crate::cold::drop_caches()?) } else { None };
    let ebpf = o.source.name == "ebpf";
    let mut line: Vec<String> = if ebpf {
        let n = crate::ebpf::calls().iter().flatten().count();
        vec![format!("raw_tracepoint:sys_enter,sys_exit ({n} calls: {TRACE_SET}), raw_tracepoint:sched_process_fork,sched_process_exit; ring buffer {}", crate::ebpf::human(o.buffer))]
    } else {
        strace_line()
    };
    // the library, unless its check failed and was waived
    let requests_check = checks.iter().find(|c| c.name == "requests").cloned();
    let library = match (o.requests, crate::requests::library()) {
        (true, Ok(lib)) => Some(lib),
        _ => None,
    };
    let log = out.join(crate::requests::LOG);
    let mut env_extra: Vec<(String, String)> = Vec::new();
    if let Some(lib) = &library {
        crate::requests::create_log(&log, o.requests_buffer)?;
        // in the command's environment alone (strace's -E), the user's own preloads after it
        let preload = match original_env().into_iter().find(|(k, _)| k == "LD_PRELOAD").map(|(_, v)| v.to_string_lossy().into_owned()) {
            Some(p) if !p.is_empty() => format!("{}:{p}", lib.display()),
            _ => lib.display().to_string(),
        };
        env_extra.push(("LD_PRELOAD".into(), preload));
        env_extra.push(("AEIOU_CAPTURE_LOG".into(), log.display().to_string()));
        if !ebpf {
            for (k, v) in &env_extra {
                line.extend(["-E".to_string(), format!("{k}={v}")]);
            }
        }
    }
    if !ebpf {
        line.push("-o".into());
        line.push(trace.to_string_lossy().into_owned());
        line.push("--".into());
        line.extend(o.command.iter().cloned());
    }

    // the probes attached and every capability given up before the command exists
    let privilege = if ebpf { crate::ebpf::privilege() } else { None };
    let mut probes = if ebpf {
        let p = crate::ebpf::Probes::attach(crate::ebpf::load(o.buffer)?)?;
        if unsafe { libc::geteuid() } != 0 {
            crate::ebpf::drop_privilege()?;
        }
        Some(p)
    } else {
        None
    };

    catch(libc::SIGINT);
    catch(libc::SIGTERM);
    let arrived = unix_now();
    if let Some(j) = &joined {
        j.tcp.barrier(START, &AtomicBool::new(false))?;
    }
    let released = unix_now();
    let anchor_id = o.capture_id.clone().unwrap_or_else(|| format!("{host}-{}", std::process::id()));
    let mut anchors: Vec<Value> = o.endpoints.iter().map(|(u, _)| crate::requests::anchor(u, &format!("aeiou-anchor-{anchor_id}-start"))).collect();
    let start = json!({"realtime": unix_now(), "realtime_minus_monotonic": realtime_minus_monotonic()});
    let bin = out.join(crate::ebpf::FILE);
    let stop = Arc::new(AtomicBool::new(false));
    let (pid, mut strace_child, reader) = match probes.as_mut() {
        Some(p) => {
            // every orphan of the command is this process's to wait for, as strace -f waits
            unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) };
            let (pid, release) = spawn_held(&o.command, &env_extra)?;
            p.seed(pid as u32)?;
            let header = json!({
                "arch": crate::ebpf::ARCH,
                "kernel": crate::run::host_facts()["kernel"],
                "realtime_minus_monotonic": start["realtime_minus_monotonic"],
                "seed": pid,
                "cwd": std::env::current_dir().ok(),
                "inherited": inherited,
            });
            let reader = p.read_into(&bin, &header, stop.clone())?;
            unsafe {
                libc::write(release, b"x".as_ptr() as *const libc::c_void, 1);
                libc::close(release);
            }
            (pid, None, Some(reader))
        }
        None => {
            let child = std::process::Command::new(&line[0]).args(&line[1..]).spawn().with_context(|| "strace".to_string())?;
            (child.id() as i32, Some(child), None)
        }
    };
    let mut interrupted: Option<String> = None;
    let mut status = None;
    loop {
        let mut st = 0;
        // under eBPF every descendant, as strace -f waits for every tracee; the command's own
        // status is the one kept
        let r = unsafe { libc::waitpid(if ebpf { -1 } else { pid }, &mut st, libc::WNOHANG) };
        if r == pid {
            status = Some(st);
            if !ebpf {
                break;
            }
        }
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::ECHILD) && status.is_some() {
                break;
            }
            if e.raw_os_error() != Some(libc::EINTR) {
                if let Some(c) = strace_child.as_mut() {
                    let _ = c.wait();
                }
                bail!("waitpid: {e}");
            }
        }
        let sig = SIGNO.swap(0, Ordering::SeqCst);
        if sig != 0 {
            interrupted = Some(signal_name(sig));
            // a terminal's signal reached the command's process group already; one sent to
            // this process alone (kill, a launcher, a scheduler's time limit) is passed on
            if !FROM_KERNEL.load(Ordering::SeqCst) {
                let mut kids = if ebpf { vec![pid] } else { children_of(pid) };
                for _ in 0..100 {
                    if !kids.is_empty() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                    kids = children_of(pid);
                }
                for k in kids {
                    unsafe { libc::kill(k, sig) };
                }
            }
        }
        if r <= 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let status = status.expect("the command's status");
    let ebpf_rec = match (probes, reader) {
        (Some(p), Some(reader)) => {
            stop.store(true, Ordering::SeqCst);
            let reading = reader.join().map_err(|_| anyhow!("the reader thread panicked"))??;
            let losses = p.losses()?;
            drop(p);
            let tail = crate::ebpf_text::trailing_gaps(&losses);
            if !tail.is_empty() {
                std::fs::OpenOptions::new().append(true).open(&bin)?.write_all(&tail)?;
            }
            let stats = crate::ebpf_text::convert(&bin, &trace)?;
            let rec = crate::ebpf::losses_record(&losses, &reading, o.buffer, &stats.windows);
            Some((rec, reading, stats))
        }
        _ => None,
    };
    let end = json!({"realtime": unix_now(), "realtime_minus_monotonic": realtime_minus_monotonic()});
    let exit = exit_of(status);
    anchors.extend(o.endpoints.iter().map(|(u, _)| crate::requests::anchor(u, &format!("aeiou-anchor-{anchor_id}-end"))));
    let requests = match &library {
        Some(lib) => {
            let mut r = crate::requests::convert(&log, &out.join(crate::requests::FILE), &crate::requests::execs(&trace)?)?;
            let (_, sha) = sha256_file(lib)?;
            r["source"] = json!("shim");
            r["library"] = json!({"path": lib, "sha256": sha});
            r["visible_calls"] = json!(crate::requests::VISIBLE);
            let l = &r["log"];
            if l["dropped"].as_u64().unwrap_or(0) > 0 {
                say(&format!(
                    "WARNING: the request log was full: {} record(s) dropped ({} bytes) of a {}-byte log; capture again with a larger --requests-buffer",
                    l["dropped"], l["dropped_bytes"], l["capacity"]
                ));
            }
            let missed = r["processes"]["not_reached"].as_array().cloned().unwrap_or_default();
            if !missed.is_empty() {
                let names: Vec<String> = missed.iter().map(|p| format!("{} (pid {})", p["exe"].as_str().unwrap_or("?"), p["pid"])).collect();
                say(&format!("WARNING: {} process(es) began without the library (LD_PRELOAD dropped from its environment, or a static or setuid binary); their requests are not in {}: {}", missed.len(), crate::requests::FILE, names.join(", ")));
            }
            for p in r["processes"]["connected_without_heads"].as_array().cloned().unwrap_or_default() {
                say(&format!("note: {} (pid {}) connected to {} and the library saw no HTTP head: TLS inside the program, HTTP/2, or not HTTP", p["exe"].as_str().unwrap_or("?"), p["pid"], p["peers"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).unwrap_or_default()));
            }
            r
        }
        None if o.requests => json!({"source": "none", "why": format!("the requests check failed and was waived: {}", requests_check.map(|c| c.detail).unwrap_or_default())}),
        None => json!({"source": "none", "why": "not asked (--requests none, the default without an --endpoint)"}),
    };

    // the checks again at the end: what changed during the capture
    let mounts = mounts_now();
    let mut changes = Vec::new();
    for r in &roots {
        let Some(path) = r["path"].as_str() else { continue };
        let (now, _) = root_record(path, &mounts);
        for k in ["fs_type", "source", "options", "super_options"] {
            if now[k] != r[k] {
                changes.push(json!(format!("--root {path}: {k} was {} and is {}", r[k], now[k])));
            }
        }
    }
    let clock_end = clock_state();
    let clock_start = checks.iter().find(|c| c.name == "clock").map(|c| c.ok);
    if clock_end["synchronized"].as_bool() != clock_start {
        changes.push(json!(format!("the clock was {} and is {}", if clock_start == Some(true) { "synchronized" } else { "not synchronized" }, if clock_end["synchronized"] == true { "synchronized" } else { "not synchronized" })));
    }

    let (bytes, sha) = sha256_file(&trace)?;
    let environment: serde_json::Map<String, Value> = ENVIRONMENT.iter().filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), json!(v)))).collect();
    let mut clock = json!({"start": checks.iter().find(|c| c.name == "clock").map(|c| c.detail.clone()), "end": clock_end});
    if let Some(off) = &offset {
        clock["offset_to_rank_0"] = off.clone();
    }
    let uid = unsafe { libc::getuid() };
    let facts = crate::run::host_facts();
    let (argv, redacted) = if o.argv_recorded { scrub_argv(&o.command) } else { (o.command.iter().take(1).cloned().collect(), Vec::new()) };
    // the line as run, its command as the record keeps it
    if !ebpf {
        line.truncate(line.len() - o.command.len());
        line.extend(argv.iter().cloned());
    }
    let rec = json!({
        "aeiou_capture": 1,
        "tool": {"name": "aeiou-trace", "version": env!("CARGO_PKG_VERSION")},
        "host": {"name": host, "kernel": facts["kernel"], "distribution": distribution(), "cpus": facts["cpus"], "cpus_online": facts["cpus_online"], "memory": facts["memory"], "cgroup_memory_max": facts["cgroup_memory_max"]},
        "source": {"name": o.source.name, "version": o.source.version, "why": o.source.why, "line": line, "privilege": privilege},
        "command": {"argv": argv, "redacted": redacted, "argv_kept": if o.argv_recorded { "recorded" } else { "none" }, "cwd": std::env::current_dir().ok(), "uid": uid, "gid": unsafe { libc::getgid() }, "as_root": unsafe { libc::geteuid() } == 0},
        "environment": environment,
        "inherited_descriptors": inherited,
        "capture_id": o.capture_id,
        "rank": (o.ranks > 1).then_some(o.rank),
        "ranks": (o.ranks > 1).then_some(o.ranks),
        "coordinator": o.coordinator,
        "captures_on_host": on_host,
        "barrier": {"arrived": arrived, "released": released},
        "start": start,
        "end": end,
        "exit": exit,
        "interrupted": interrupted,
        "roots": roots,
        "endpoints": o.endpoints.iter().map(|(u, s)| json!({"url": scrub_url(u), "scheme": s})).collect::<Vec<_>>(),
        "cache": if cache_drop.is_some() { "dropped" } else { "not dropped" },
        "cache_drop": cache_drop,
        "losses": match &ebpf_rec {
            Some((l, _, _)) => l.clone(),
            None => json!({"events": 0, "note": "strace loses no call: it stops the traced thread instead, and the application is slower"}),
        },
        "complete": ebpf_rec.as_ref().is_none_or(|(l, _, _)| l["events"] == 0),
        "paths_not_copied": ebpf_rec.as_ref().map_or(0, |(_, _, s)| s.paths_not_copied),
        "reader": ebpf_rec.as_ref().map(|(_, r, _)| json!({"records": r.records, "bytes": r.bytes, "seconds": r.seconds, "peak_records_per_s": r.peak_records_per_s, "peak_bytes_per_s": r.peak_bytes_per_s, "file": crate::ebpf::FILE})),
        "conversion": ebpf_rec.as_ref().map(|(_, _, s)| s.json()),
        "not_seen": NOT_SEEN_STRACE.iter().chain(if ebpf { NOT_SEEN_EBPF.iter() } else { [].iter() }).map(|n| if library.is_some() && n.starts_with("object-store requests") { crate::requests::NOT_SEEN } else { n }).collect::<Vec<_>>(),
        "requests": requests,
        "anchors": anchors,
        "mountstats": "not taken: a traced run's wire counts are not the application's (strace -yy adds a GETATTR to each READ); take them from an untraced run",
        "clock": clock,
        "checks": checks.iter().map(Check::json).collect::<Vec<_>>(),
        "waivers": waivers,
        "changes_at_end": changes,
        "trace": {"file": "trace.txt", "bytes": bytes, "sha256": sha},
    });
    write_json(&out.join("capture.json"), &rec)?;
    for c in &changes {
        say(&format!("changed during the capture: {}", c.as_str().unwrap_or("")));
    }
    if let Some((l, _, _)) = &ebpf_rec {
        if l["events"].as_u64().unwrap_or(0) > 0 {
            say(&crate::ebpf::loss_box(l));
        }
    }
    say(&format!("written: {}", out.display()));

    if let Some(j) = &joined {
        let summary = json!({
            "rank": o.rank,
            "host": host,
            "directory": out,
            "source": o.source.name,
            "losses": rec["losses"]["events"],
            "complete": rec["complete"],
            "clock_offset_s": offset.as_ref().map(|v| v["offset_s"].clone()),
            "clock_bound_s": offset.as_ref().map(|v| v["bound_s"].clone()),
            "exit": rec["exit"],
            "interrupted": rec["interrupted"],
            "trace_sha256": rec["trace"]["sha256"],
            "requests": rec["requests"]["requests"],
            "requests_dropped": rec["requests"]["log"]["dropped"],
            "waived": checks.iter().filter(|c| c.waived).map(|c| json!({"name": c.name, "node": c.node})).collect::<Vec<_>>(),
            "changes_at_end": rec["changes_at_end"],
        });
        finish(j, &o, summary, Some(&out))?;
    }

    if libc::WIFSIGNALED(status) {
        // as strace does: the command's signal ends this process too
        let sig = libc::WTERMSIG(status);
        unsafe {
            libc::signal(sig, libc::SIG_DFL);
            libc::raise(sig);
        }
        return Ok(128 + sig);
    }
    Ok(libc::WEXITSTATUS(status))
}

/// `--convert DIR`: the capture's `trace.txt` written again from its `syscalls.bin`, and its
/// record's `trace` and `conversion` updated, with when and by which version.
pub fn convert_again(dir: &Path) -> Result<()> {
    let bin = dir.join(crate::ebpf::FILE);
    if !bin.is_file() {
        bail!("{}: no {} (a capture under --source ebpf keeps one; one under strace has nothing to convert)", dir.display(), crate::ebpf::FILE);
    }
    let rec_path = dir.join("capture.json");
    let mut rec: Value = serde_json::from_str(&std::fs::read_to_string(&rec_path).with_context(|| rec_path.display().to_string())?).with_context(|| rec_path.display().to_string())?;
    let trace = dir.join("trace.txt");
    let stats = crate::ebpf_text::convert(&bin, &trace)?;
    let (bytes, sha) = sha256_file(&trace)?;
    rec["trace"] = json!({"file": "trace.txt", "bytes": bytes, "sha256": sha});
    rec["conversion"] = stats.json();
    rec["converted_again"] = json!({"by": format!("aeiou-trace {}", env!("CARGO_PKG_VERSION")), "at": unix_now()});
    write_json(&rec_path, &rec)?;
    say(&format!("written: {} ({} calls)", trace.display(), stats.calls));
    Ok(())
}

/// The end on several nodes: this rank's summary to rank 0, which gathers every rank's,
/// measures the clocks again, and writes `capture-set.json` into its directory.
fn finish(j: &Joined, o: &Opts, summary: Value, dir: Option<&Path>) -> Result<()> {
    let sent = j.tcp.done(&summary);
    if let Some(s) = &j.server {
        // rank 0 records the set whatever became of its own connection: a rank lost during the
        // capture failed the coordinator, and the set says so
        let gathered = s.gathered();
        let clocks_end = if gathered.is_ok() { s.clocks(CLOCK_ROUNDS).ok() } else { None };
        let (nodes, status) = match &gathered {
            Ok(g) => (g.values().cloned().collect::<Vec<_>>(), "complete".to_string()),
            Err(e) => (s.done_so_far().values().cloned().collect(), format!("incomplete: {e:#}")),
        };
        if let Some(dir) = dir {
            let set = json!({
                "aeiou_capture_set": 1,
                "capture_id": o.capture_id,
                "ranks": o.ranks,
                "coordinator": o.coordinator,
                "nodes": nodes,
                "clock_end": clocks_end.map(|c| c.iter().map(|(r, (off, b))| (r.to_string(), json!({"offset_s": off, "bound_s": b}))).collect::<serde_json::Map<_, _>>()),
                "status": status,
            });
            write_json(&dir.join("capture-set.json"), &set)?;
            say(&format!("capture set: {}", dir.join("capture-set.json").display()));
        }
        s.finish(gathered.is_ok(), 0, gathered.as_ref().err().map(|e| format!("{e:#}")));
        gathered?;
    } else {
        sent?;
        j.tcp.result()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_is_found_as_execvp_finds_it_in_the_users_environment() {
        assert_eq!(find_program("/bin/sh").unwrap(), PathBuf::from("/bin/sh"));
        assert_eq!(find_program("./x").unwrap(), PathBuf::from("./x"));
        let sh = find_program("sh").unwrap();
        assert!(sh.is_absolute() && sh.ends_with("sh"), "{}", sh.display());
        assert!(find_program("no-such-program-aeiou").unwrap_err().to_string().contains("not found on the PATH"));
        // not secure-exec here: the process's own environment
        let env = original_env();
        assert!(env.iter().any(|(k, _)| k == "PATH"));
        assert_eq!(env.len(), std::env::vars_os().count());
    }

    const INFO: &str = "\
22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw
40 22 0:50 / /mnt/data rw,relatime shared:2 - nfs4 filer:/export/a rw,vers=4.2,rsize=1048576,clientaddr=10.0.0.5,addr=10.0.0.1
41 40 0:51 / /mnt/data/sub\\040dir rw - tmpfs none rw
42 22 0:52 / /mnt/data rw - nfs4 filer2:/b rw,vers=3
";

    #[test]
    fn mountinfo_is_read_with_its_escapes_and_the_last_mount_over_a_point_wins() {
        let m = mountinfo(INFO);
        assert_eq!(m.len(), 4);
        assert_eq!(m[2].mount_point, "/mnt/data/sub dir");
        assert_eq!(m[1].super_options, "rw,vers=4.2,rsize=1048576,clientaddr=10.0.0.5,addr=10.0.0.1");
        assert_eq!(mount_of(&m, Path::new("/home/x")).unwrap().fs_type, "ext4");
        assert_eq!(mount_of(&m, Path::new("/mnt/data/f")).unwrap().source, "filer2:/b");
        assert_eq!(mount_of(&m, Path::new("/mnt/data/sub dir/f")).unwrap().fs_type, "tmpfs");
        // by whole components: /mnt/database is not under /mnt/data
        assert_eq!(mount_of(&m, Path::new("/mnt/database")).unwrap().fs_type, "ext4");
        assert!(mountinfo("garbage\n").is_empty());
    }

    #[test]
    fn options_compare_without_the_client_s_own_address() {
        let a = json!({"options": "rw,relatime", "super_options": "rw,vers=4.2,clientaddr=10.0.0.5"});
        let b = json!({"options": "relatime,rw", "super_options": "rw,vers=4.2,clientaddr=10.0.0.6"});
        assert_eq!(comparable_options(&a), comparable_options(&b));
        let c = json!({"options": "rw,relatime", "super_options": "rw,vers=3"});
        assert_ne!(comparable_options(&a), comparable_options(&c));
    }

    #[test]
    fn network_file_systems_are_named() {
        for t in ["nfs", "nfs4", "lustre", "fuse.s3fs", "fuse", "cifs", "9p"] {
            assert!(network_fs(t), "{t}");
        }
        for t in ["ext4", "xfs", "tmpfs", "btrfs", "overlay"] {
            assert!(!network_fs(t), "{t}");
        }
    }

    #[test]
    fn the_output_template_expands_host_pid_and_percent() {
        assert_eq!(expand_out("/l/cap-%h-%p-%%", "n1", 42).unwrap(), "/l/cap-n1-42-%");
        assert!(expand_out("/l/%x", "n1", 1).unwrap_err().contains("%x: only %h"));
        assert!(expand_out("/l/%", "n1", 1).unwrap_err().contains("%: only %h"));
    }

    fn node(host: &str, release: &str, fs: &str, opts: &str) -> Value {
        json!({"host": host, "release": release, "source": {"name": "strace", "version": "6.8"}, "roots": [{"path": "/mnt/data", "fs_type": fs, "source": "filer:/a", "options": "rw", "super_options": opts}]})
    }

    #[test]
    fn rank_0_compares_releases_sources_mounts_and_clocks() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, node("a", "0.3.0", "nfs4", "vers=4.2,clientaddr=1"));
        nodes.insert(1, node("b", "0.3.0", "nfs4", "vers=4.2,clientaddr=2"));
        let offsets: BTreeMap<i64, (f64, f64)> = [(0, (0.0, 0.0001)), (1, (0.002, 0.0001))].into();
        let c = cross_checks(&nodes, &offsets);
        assert!(c.iter().all(|c| c.ok), "{c:?}");
        assert_eq!(c.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["release", "same-source", "mounts", "clock-offset"]);

        nodes.insert(2, node("c", "0.2.0", "nfs4", "vers=3,clientaddr=3"));
        let offsets: BTreeMap<i64, (f64, f64)> = [(0, (0.0, 0.0)), (1, (0.0, 0.0)), (2, (-0.5, 0.001))].into();
        let c = cross_checks(&nodes, &offsets);
        let by = |n: &str| c.iter().find(|c| c.name == n).unwrap();
        assert!(!by("release").ok && by("release").detail.contains("rank 2 (c) has 0.2.0, rank 0 0.3.0"), "{:?}", by("release"));
        assert!(!by("mounts").ok && by("mounts").detail.starts_with("/mnt/data: rank 2 (c)"), "{:?}", by("mounts"));
        assert!(!by("clock-offset").ok && by("clock-offset").detail.contains("rank 2 (c) is -500.000 ms"), "{:?}", by("clock-offset"));
        assert!(by("same-source").ok);
    }

    #[test]
    fn waivers_mark_failed_checks_and_the_table_says_which() {
        let mut c = vec![Check::new("space", false, "little"), Check::new("clock", false, "unsynchronized"), Check::new("root", true, "/d on ext4")];
        waive(&mut c, &["clock".into(), "root".into()]);
        assert!(c[0].fails() && !c[1].fails() && c[1].waived && !c[2].waived);
        let t = table(&c);
        assert!(t.contains("space  FAILED  little") && t.contains("clock  waived  unsynchronized") && t.contains("root   ok      /d on ext4"), "{t}");
        c[0].node = Some("rank 1 (b)".into());
        let t = table(&c);
        assert!(t.contains("rank 1 (b)  space") && t.contains("all         clock"), "{t}");
        assert_eq!(Check::of(&c[0].json()), c[0]);
    }

    #[test]
    fn a_new_directory_resolves_through_what_exists() {
        let d = std::env::temp_dir();
        let r = resolve_new(&d.join("aeiou-capture-none").join("x"));
        assert_eq!(r, std::fs::canonicalize(&d).unwrap().join("aeiou-capture-none").join("x"));
        assert_eq!(lexical(Path::new("/a/b/../c/./d")), PathBuf::from("/a/c/d"));
        assert_eq!(signal_name(15), "SIGTERM");
        assert_eq!(signal_name(40), "signal 40");
        assert_eq!(unescape("a\\040b\\134c"), "a b\\c");
    }

    #[test]
    fn the_strace_line_prints_no_argument_list_environment_or_bytes() {
        let l = strace_line();
        assert_eq!(&l[..3], ["strace", "-f", "--seccomp-bpf"]);
        // -v for whole arrays, but execve's lists as addresses and every string's bytes cut at 0
        for w in [["-v", "-e"], ["-e", "verbose=!execve,execveat"], ["-s", "0"], ["-e", TRACE_SET]] {
            assert!(l.windows(2).any(|p| p == w), "{w:?}: {l:?}");
        }
    }

    #[test]
    fn secret_values_on_a_command_line_are_redacted_and_their_places_listed() {
        let argv: Vec<String> = [
            "env", "AWS_SECRET_ACCESS_KEY=abc", "PATH=/bin", "train.py", "--password=hunter2", "--token", "t0k", "--Auth-Header", "-v", "--batch=8",
            "--data", "s3://user:pw@host/b", "DSN=postgres://u:p@db/x", "--session-id=1", "https://h/x?a=b", "--key-file", "/k",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let (got, at) = scrub_argv(&argv);
        assert_eq!(
            got,
            [
                "env", "AWS_SECRET_ACCESS_KEY=<redacted>", "PATH=/bin", "train.py", "--password=<redacted>", "--token", "<redacted>", "--Auth-Header", "-v", "--batch=8",
                "--data", "s3://user:<redacted>@host/b", "DSN=postgres://u:<redacted>@db/x", "--session-id=<redacted>", "https://h/x?a=b", "--key-file", "<redacted>",
            ]
        );
        // an option's value is the next argument unless that is an option itself
        assert_eq!(at, [1, 4, 6, 11, 12, 13, 16]);
        // the program is never taken for an option, a URL's user alone is no secret
        // nor a script's text for a name
        let script = "echo \"$TOKEN\"; a=1";
        let (got, at) = scrub_argv(&["--token=x".into(), "http://user@h/".into(), "http://h:9000/b".into(), script.into(), "---key=x".into()]);
        assert_eq!((got.as_slice(), at.as_slice()), (["--token=x".to_string(), "http://user@h/".into(), "http://h:9000/b".into(), script.into(), "---key=x".into()].as_slice(), [].as_slice()));
        assert_eq!(scrub_url("http://a:b@h:9000/p"), "http://a:<redacted>@h:9000/p");
        assert_eq!(scrub_url("s3://bucket/pre"), "s3://bucket/pre");
    }
}
