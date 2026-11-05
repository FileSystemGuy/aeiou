//! `aeiou-trace`: an `strace` of a real application read as its calls (`DESIGN_REVIEW.md` §3.42,
//! §3.89; `builder/REFERENCE.md` §7). The calls on paths under the roots are the workload: the
//! metrics count them (`metrics::TraceMetrics`, the definitions of `runner/REFERENCE.md` §10)
//! and the exporter writes them as the runner's trace file (`export.rs`). Ported from the
//! Python implementation (`builder/aeiou/trace.py` until 2026-10-29), whose behavior the golden
//! corpus under `tests/strace/` keeps.
//!
//! What differs from the dry-run side, because a trace is not an abstract:
//! - **Which calls count.** Only calls on paths under a root (the `-yy` annotation of the
//!   descriptor, or the path argument), less those an `--exclude` glob matches.
//! - **Order.** Calls are taken in the order they returned (the trace's line order).
//! - **Instance.** The whole trace is one instance unless `--instance-root PID` names process
//!   trees; reuse is taken per instance.
//! - **Sequential context.** A thread (a tid of the trace).
//! - **Offsets** of `read` and `write` come from the descriptor's tracked position (`open`,
//!   `lseek`, and the bytes each call moved); `pread`/`pwrite` carry their own.
//! - **Fan-out and depth** only where the trace shows them: the requests of one `io_submit`
//!   are a fan-out, consecutive `io_submit`s of a thread with no other counted call between
//!   and at most `--chain-gap-us` of think time a chain.
//!
//! Descriptor tables follow `clone` when the trace has it (`-e trace=…,%process`); otherwise a
//! thread that uses a descriptor it never opened joins the table of the one that did (the
//! `-yy` path decides which). Not visible to `strace`, so not counted: `io_uring` submissions
//! (an eBPF capture's `aeiou_uring_sqe` and `aeiou_uring_cqe` records are, as the calls they
//! do; DESIGN_REVIEW.md §3.98) and page faults on a mapping. The trace is read once, in a stream; the state is per
//! descriptor, per thread, and per path under the roots, never per call.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::rc::Rc;

use anyhow::Result;
use serde_json::{json, Value};

use crate::export::Exporter;
use crate::metrics::{Opts as MetricsOpts, TraceMetrics};

pub const R: usize = 0;
pub const W: usize = 1;
pub const KINDS: [&str; 2] = ["read", "write"];
pub const METRICS_FORMAT: u64 = 2;
/// The version of the capture's trace format this reader knows (DESIGN_REVIEW.md §3.98): strace's
/// text with the `aeiou_` records, given by the header line an eBPF capture begins with.
pub const TRACE_FORMAT: u64 = 1;
const UNFINISHED: &str = "<unfinished ...>";
/// The note of the events an eBPF capture lost (its `--- aeiou: N event(s) lost` lines): the
/// numbers come from an incomplete trace.
pub const EVENTS_LOST: &str = "events_lost";
/// The note of a trace whose faults may miss first touches: its capture's fault-around check
/// was waived, so the kernel mapped pages ahead of a fault and their touches did not fault.
pub const FIRST_TOUCH_UNSEEN: &str = "faults_first_touch_unseen";

/// The notes of what the capture's line keeps out of a trace and this one prints (brief item 62,
/// `DESIGN_REVIEW.md` §3.97): `execve`'s arguments and environment (`-v` without
/// `-e verbose=!execve,execveat`), and a call's bytes (without `-s 0`), either of which can hold
/// the owner's secrets. Counted by call; nothing read uses them.
pub const ARGUMENTS_PRINTED: &str = "execve_arguments_printed";
pub const DATA_PRINTED: &str = "buffer_contents_printed";
/// The calls whose bytes strace prints as a string.
const DATA_CALLS: [&str; 16] = ["read", "write", "pread64", "pwrite64", "readv", "writev", "preadv", "pwritev", "preadv2", "pwritev2", "sendto", "recvfrom", "sendmsg", "recvmsg", "sendmmsg", "recvmmsg"];
/// Why a trace that prints them is not to be sent (`trace_may_hold_secrets`).
pub const SECRETS: &str = "this trace prints execve's arguments and environment, or the bytes calls read and wrote, where a password, a token, or a key can be: keep it to yourself and send its digest, or trace again with aeiou-trace capture or the man page's line (-e verbose=!execve,execveat -s 0)";
const UNDER_CACHE: usize = 1 << 20;

/// What the metrics and the exporter are told about a trace.
#[derive(Debug, Clone, Default)]
pub struct Opts {
    pub roots: Vec<String>,
    pub exclude: Vec<String>,
    pub cwd: Option<String>,
    pub instance_roots: Vec<i64>,
    pub chain_gap_us: Option<f64>,
    pub block: Option<u64>,
}

/// The lines read, and those of them that parsed as a finished call strace printed: what the
/// tool says when nothing was counted (the wrong root, or not an strace).
#[derive(Debug, Clone, Copy, Default)]
pub struct Seen {
    pub lines: u64,
    pub calls: u64,
}

// ---------------------------------------------------------------- text

/// Whether `strace` cut the array `arg` short (`[..., ...]`): it prints only the first 32
/// elements of an array unless `-v` is given.
pub fn truncated(arg: &str) -> bool {
    arg.trim_end().ends_with("...]")
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether a `clone`/`clone3` (whose argument text is `args`) shares the descriptor table with
/// its child: `CLONE_FILES` in its `flags=` (`fork` and `vfork` copy it).
pub fn shares_files(name: &str, args: &str) -> bool {
    if name != "clone" && name != "clone3" {
        return false;
    }
    let b = args.as_bytes();
    let mut from = 0;
    while let Some(i) = args[from..].find("flags=").map(|i| i + from) {
        from = i + 1;
        if i > 0 && is_word(b[i - 1]) {
            continue;
        }
        let seg_start = i + 6;
        let seg_end = args[seg_start..].find([',', '}']).map_or(args.len(), |j| seg_start + j);
        let seg = &args[seg_start..seg_end];
        let mut k = 0;
        while let Some(j) = seg[k..].find("CLONE_FILES").map(|j| j + k) {
            let (s, e) = (seg_start + j, seg_start + j + 11);
            let before = s == 0 || !is_word(b[s - 1]);
            let after = e >= b.len() || !is_word(b[e]);
            if before && after {
                return true;
            }
            k = j + 1;
        }
    }
    false
}

/// The top-level arguments of a call: commas inside strings, brackets, and the `<path>`
/// annotation of a descriptor do not split.
pub fn split_args(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let (mut out, mut depth, mut angle, mut quote, mut start, mut i) = (Vec::new(), 0i64, 0i64, false, 0usize, 0usize);
    let n = b.len();
    while i < n {
        let c = b[i];
        if quote {
            if c == b'\\' {
                i += 1;
            } else if c == b'"' {
                quote = false;
            }
        } else if angle > 0 {
            if c == b'<' {
                angle += 1;
            } else if c == b'>' {
                angle -= 1;
            }
        } else if c == b'"' {
            quote = true;
        } else if c == b'<' && i > start && (b[i - 1].is_ascii_digit() || b[i - 1] == b'D') {
            angle = 1;
        } else if matches!(c, b'(' | b'[' | b'{') {
            depth += 1;
        } else if matches!(c, b')' | b']' | b'}') {
            depth -= 1;
        } else if c == b',' && depth == 0 {
            out.push(s[start..i].trim().to_string());
            start = i + 1;
        }
        i += 1;
    }
    let tail = s.get(start..).unwrap_or("").trim();
    if !tail.is_empty() || !out.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// `3</path>` → `(Some(3), Some("/path"))`, `3</path>(deleted)` the same; `AT_FDCWD</cwd>` →
/// `(None, Some("/cwd"))`; a
/// descriptor without an annotation → `(fd, None)`; anything else `(None, None)`.
pub fn fd_arg(tok: &str) -> (Option<i64>, Option<String>) {
    let (num, rest) = if let Some(r) = tok.strip_prefix("AT_FDCWD") {
        (None, r)
    } else {
        let digits = tok.strip_prefix('-').unwrap_or(tok);
        let end = digits.bytes().position(|c| !c.is_ascii_digit()).unwrap_or(digits.len());
        if end == 0 {
            return (None, None);
        }
        let len = tok.len() - digits.len() + end;
        match tok[..len].parse::<i64>() {
            Ok(v) => (Some(v), &tok[len..]),
            Err(_) => return (None, None),
        }
    };
    if rest.is_empty() {
        return (num, None);
    }
    // strace 6.8 prints an unlinked file's `(deleted)` after the brackets, older ones inside
    let rest = rest.strip_suffix("(deleted)").unwrap_or(rest);
    match rest.strip_prefix('<').and_then(|r| r.strip_suffix('>')) {
        Some(raw) => (num, Some(fd_path(raw))),
        None => (None, None),
    }
}

/// The path of a descriptor's `<...>` annotation, unescaped as a quoted string is: strace
/// prints it as one without the quotes, with `<` and `>` as octal so no raw angle bracket is
/// part of the path, and a `<char 1:3>` it appends to a device's is dropped.
pub fn fd_path(raw: &str) -> String {
    let mut raw = raw;
    if raw.ends_with('>') {
        if let Some(i) = raw.rfind('<') {
            raw = &raw[..i];
        }
    }
    let raw = raw.strip_suffix(" (deleted)").unwrap_or(raw);
    unescape(raw)
}

/// A quoted string argument, unescaped; `None` if `tok` is not one.
pub fn c_string(tok: &str) -> Option<String> {
    if !tok.starts_with('"') {
        return None;
    }
    let end = tok.rfind('"')?;
    if end == 0 {
        return None;
    }
    Some(unescape(&tok[1..end]))
}

/// strace's escapes (`\303\251`, `\n`, `\\`, `\"`, `\x41`) undone, the bytes read as UTF-8;
/// what cannot be undone is left as it was.
pub fn unescape(raw: &str) -> String {
    if !raw.contains('\\') {
        return raw.to_string();
    }
    let chars: Vec<char> = raw.chars().collect();
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    let hex = |s: &[char]| -> Option<u32> {
        let t: String = s.iter().collect();
        if s.iter().all(|c| c.is_ascii_hexdigit()) { u32::from_str_radix(&t, 16).ok() } else { None }
    };
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' {
            if (c as u32) > 0xFF {
                return raw.to_string();
            }
            out.push(c as u32 as u8);
            i += 1;
            continue;
        }
        let Some(&e) = chars.get(i + 1) else { return raw.to_string() };
        i += 2;
        let byte = |v: u32, out: &mut Vec<u8>| -> bool {
            if v > 0xFF {
                return false;
            }
            out.push(v as u8);
            true
        };
        match e {
            '\n' => {}
            '\\' => out.push(b'\\'),
            '\'' => out.push(b'\''),
            '"' => out.push(b'"'),
            'a' => out.push(7),
            'b' => out.push(8),
            'f' => out.push(12),
            'n' => out.push(b'\n'),
            'r' => out.push(b'\r'),
            't' => out.push(b'\t'),
            'v' => out.push(11),
            '0'..='7' => {
                let mut v = e.to_digit(8).unwrap();
                let mut k = 0;
                while k < 2 && i < chars.len() && chars[i].is_digit(8) {
                    v = v * 8 + chars[i].to_digit(8).unwrap();
                    i += 1;
                    k += 1;
                }
                if !byte(v, &mut out) {
                    return raw.to_string();
                }
            }
            'x' | 'u' | 'U' => {
                let w = match e {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                if i + w > chars.len() {
                    return raw.to_string();
                }
                let Some(v) = hex(&chars[i..i + w]) else { return raw.to_string() };
                i += w;
                if !byte(v, &mut out) {
                    return raw.to_string();
                }
            }
            'N' => return raw.to_string(),
            other => {
                out.push(b'\\');
                if (other as u32) > 0xFF {
                    return raw.to_string();
                }
                out.push(other as u32 as u8);
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `os.path.normpath` as Linux resolves names: a leading `//` is `/`.
pub fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let mut initial = if path.starts_with('/') { 1 } else { 0 };
    if path.starts_with("//") && !path.starts_with("///") {
        initial = 2;
    }
    let mut new: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".." || (initial == 0 && new.is_empty()) || new.last() == Some(&"..") {
            new.push(comp);
        } else if !new.is_empty() {
            new.pop();
        }
    }
    let mut p = new.join("/");
    if initial > 0 {
        p = format!("{}{}", "/".repeat(initial), p);
    }
    if p.is_empty() {
        p = ".".into();
    }
    if p.starts_with("//") {
        p.remove(0);
    }
    p
}

/// `os.path.join(base, path)` for one relative `path`.
pub fn join(base: &str, path: &str) -> String {
    if path.starts_with('/') || base.is_empty() {
        path.to_string()
    } else if base.ends_with('/') {
        format!("{base}{path}")
    } else {
        format!("{base}/{path}")
    }
}

/// `os.path.abspath`: `path` against the working directory, normalized.
pub fn abspath(path: &str) -> String {
    if path.starts_with('/') {
        return normpath(path);
    }
    let cwd = std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| "/".into());
    normpath(&join(&cwd, path))
}

/// `fnmatch.fnmatchcase`: `*` (crossing `/`), `?`, `[seq]`, `[!seq]`, over the whole name.
pub fn glob(name: &str, pat: &str) -> bool {
    fn class(p: &[char], at: usize, c: char) -> Option<(bool, usize)> {
        // returns (matched, index after the class), or None when `[` opens no class
        let mut i = at + 1;
        let neg = p.get(i) == Some(&'!');
        if neg {
            i += 1;
        }
        let start = i;
        let mut hit = false;
        while i < p.len() && (p[i] != ']' || i == start) {
            if i + 2 < p.len() && p[i + 1] == '-' && p[i + 2] != ']' {
                if p[i] <= c && c <= p[i + 2] {
                    hit = true;
                }
                i += 3;
            } else {
                if p[i] == c {
                    hit = true;
                }
                i += 1;
            }
        }
        if i >= p.len() {
            return None;
        }
        Some((hit != neg, i + 1))
    }
    fn go(n: &[char], p: &[char]) -> bool {
        let (mut ni, mut pi) = (0, 0);
        let mut star: Option<(usize, usize)> = None;
        while ni < n.len() {
            if pi < p.len() {
                match p[pi] {
                    '*' => {
                        star = Some((pi, ni));
                        pi += 1;
                        continue;
                    }
                    '?' => {
                        ni += 1;
                        pi += 1;
                        continue;
                    }
                    '[' => match class(p, pi, n[ni]) {
                        Some((true, next)) => {
                            ni += 1;
                            pi = next;
                            continue;
                        }
                        Some((false, _)) => {}
                        None => {
                            if n[ni] == '[' {
                                ni += 1;
                                pi += 1;
                                continue;
                            }
                        }
                    },
                    c if c == n[ni] => {
                        ni += 1;
                        pi += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            match star {
                Some((sp, sn)) => {
                    pi = sp + 1;
                    ni = sn + 1;
                    star = Some((sp, sn + 1));
                }
                None => return false,
            }
        }
        while pi < p.len() && p[pi] == '*' {
            pi += 1;
        }
        pi == p.len()
    }
    let n: Vec<char> = name.chars().collect();
    let p: Vec<char> = pat.chars().collect();
    go(&n, &p)
}

fn seconds(ts: &str) -> f64 {
    if ts.contains(':') {
        let mut it = ts.split(':');
        let h: i64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let m: i64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let s: f64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
        return (h * 3600 + m * 60) as f64 + s;
    }
    ts.parse().unwrap_or(0.0)
}

/// A line's `[pid N] ` or `N `, its timestamp, and the rest (`_PREFIX` of the Python).
fn prefix(line: &str) -> (i64, Option<&str>, &str) {
    let s = line.trim_start();
    let b = s.as_bytes();
    let ws = |i: usize| -> usize { b[i..].iter().position(|c| !c.is_ascii_whitespace()).map_or(b.len(), |k| i + k) };
    let digits = |i: usize| -> usize { b[i..].iter().position(|c| !c.is_ascii_digit()).map_or(b.len(), |k| i + k) };
    let mut tid = 0;
    let mut at = 0;
    if let Some(r) = s.strip_prefix("[pid") {
        let i = s.len() - r.len();
        let j = ws(i);
        let k = digits(j);
        if j > i && k > j && b.get(k) == Some(&b']') {
            let l = ws(k + 1);
            if l > k + 1 {
                tid = s[j..k].parse().unwrap_or(0);
                at = l;
            }
        }
    } else {
        let k = digits(0);
        if k > 0 && k < b.len() && b[k].is_ascii_whitespace() {
            tid = s[..k].parse().unwrap_or(0);
            at = ws(k);
        }
    }
    // the timestamp: `\d+\.\d+` or `\d\d:\d\d:\d\d(\.\d+)?`, then whitespace
    let k = digits(at);
    let mut end = None;
    if k > at && b.get(k) == Some(&b'.') {
        let f = digits(k + 1);
        if f > k + 1 {
            end = Some(f);
        }
    }
    if end.is_none() && b.len() >= at + 8 && k == at + 2 && b[at + 2] == b':' {
        let ok = b[at + 3].is_ascii_digit() && b[at + 4].is_ascii_digit() && b[at + 5] == b':' && b[at + 6].is_ascii_digit() && b[at + 7].is_ascii_digit();
        if ok {
            let mut e = at + 8;
            if b.get(e) == Some(&b'.') {
                let f = digits(e + 1);
                if f > e + 1 {
                    e = f;
                }
            }
            end = Some(e);
        }
    }
    if let Some(e) = end {
        if e < b.len() && b[e].is_ascii_whitespace() {
            return (tid, Some(&s[at..e]), &s[ws(e)..]);
        }
    }
    (tid, None, &s[at..])
}

/// `<... NAME resumed> rest`.
fn resumed(body: &str) -> Option<(&str, &str)> {
    let r = body.strip_prefix("<... ")?;
    let k = r.bytes().position(|c| !is_word(c))?;
    if k == 0 {
        return None;
    }
    let rest = r[k..].strip_prefix(" resumed>")?;
    Some((&r[..k], rest.trim_start()))
}

/// Where a finished call's arguments end (its last `)`) and its return value begins: the last
/// `)` followed by spaces and `= `. strace pads a short call's ` = ` out to column 40 (`-a`),
/// so `rmdir("d")  = 0` has two spaces.
fn ret_split(body: &str) -> Option<(usize, usize)> {
    let b = body.as_bytes();
    let mut end = body.len();
    while let Some(e) = body[..end].rfind("= ") {
        let mut j = e;
        while j > 0 && b[j - 1] == b' ' {
            j -= 1;
        }
        if j < e && j > 0 && b[j - 1] == b')' {
            return Some((j - 1, e + 2));
        }
        end = e;
    }
    None
}

/// The return value at the start of `s`: `-?\d+`, `0x[0-9a-f]+`, or `?`; with its length.
fn ret_token(s: &str) -> Option<(&str, usize)> {
    let b = s.as_bytes();
    if b.first() == Some(&b'?') {
        return Some(("?", 1));
    }
    if s.starts_with("0x") {
        let k = b[2..].iter().position(|c| !(c.is_ascii_digit() || (b'a'..=b'f').contains(c))).map_or(b.len(), |k| k + 2);
        if k > 2 {
            // `-?\d+` comes first in the alternation and matches the `0`
            return Some((&s[..1], 1));
        }
    }
    let neg = usize::from(b.first() == Some(&b'-'));
    let k = b[neg..].iter().position(|c| !c.is_ascii_digit()).map_or(b.len(), |k| k + neg);
    if k > neg {
        return Some((&s[..k], k));
    }
    None
}

pub(crate) fn int0(s: &str) -> Option<i64> {
    let (neg, t) = match s.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, s),
    };
    let v = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i64::from_str_radix(h, 16).ok()?
    } else if let Some(o) = t.strip_prefix("0o") {
        i64::from_str_radix(o, 8).ok()?
    } else {
        if t.len() > 1 && t.starts_with('0') && !t.bytes().all(|c| c == b'0') {
            return None;
        }
        t.parse().ok()?
    };
    Some(if neg { -v } else { v })
}

/// The note for a line that prints what the capture's line keeps out (`ARGUMENTS_PRINTED`,
/// `DATA_PRINTED`): `args` from the call's `(`, or a resumed line's rest. A quoted argument
/// begins after `, ` or `iov_base=`, or a resumed line's rest; `-s 0` prints each as `""...`.
/// A `-yy` path escapes its quotes, and a socket address's `sun_path="` follows no `, `; an
/// IPv6 address strace decodes (a datagram's sender, a netlink listing's) is
/// `inet_pton(AF_INET6, "::1")` under `-s 0` too, and is not the call's bytes.
fn printed(name: &str, args: &str) -> Option<&'static str> {
    if name == "execve" || name == "execveat" {
        return args.contains("\", [").then_some(ARGUMENTS_PRINTED);
    }
    if !DATA_CALLS.contains(&name) {
        return None;
    }
    let quoted = |m: &str| {
        args.match_indices(m)
            .any(|(j, _)| args.as_bytes().get(j + m.len()) != Some(&b'"') && !args[..j].ends_with("inet_pton(AF_INET6"))
    };
    (quoted(", \"") || quoted("iov_base=\"") || (args.starts_with('"') && !args.starts_with("\"\""))).then_some(DATA_PRINTED)
}

fn digits_only(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())
}

/// `iov_len=N` of every iovec.
fn iov_lens(s: &str) -> i64 {
    let mut total = 0;
    let mut from = 0;
    while let Some(i) = s[from..].find("iov_len=").map(|i| i + from) {
        let d = &s[i + 8..];
        let k = d.bytes().position(|c| !c.is_ascii_digit()).unwrap_or(d.len());
        if k > 0 {
            total += d[..k].parse::<i64>().unwrap_or(0);
        }
        from = i + 8;
    }
    total
}

/// One AIO request of an `io_submit`'s array.
#[derive(Debug, Clone)]
struct Iocb {
    data: String,
    op: String,
    fd: i64,
    path: Option<String>,
    nbytes: i64,
    offset: i64,
}

impl Iocb {
    /// `Some(true)` for a read, `Some(false)` for a write, `None` for any other request.
    fn reads(&self) -> Option<bool> {
        match self.op.as_str() {
            "PREAD" | "PREADV" => Some(true),
            "PWRITE" | "PWRITEV" => Some(false),
            _ => None,
        }
    }
}

fn word_at(s: &str, i: usize) -> &str {
    let k = s.as_bytes()[i..].iter().position(|c| !is_word(*c)).map_or(s.len(), |k| i + k);
    &s[i..k]
}

fn num_at(s: &str, i: usize, neg: bool) -> Option<(i64, usize)> {
    let b = s.as_bytes();
    let n = usize::from(neg && b.get(i) == Some(&b'-'));
    let k = b[i + n..].iter().position(|c| !c.is_ascii_digit()).map_or(b.len(), |k| i + n + k);
    if k == i + n {
        return None;
    }
    Some((s[i..k].parse().ok()?, k))
}

/// The requests of an `io_submit` array, each `{...}` read by its fields as strace prints
/// them: `aio_data`, `aio_rw_flags` if set, `aio_lio_opcode`, `aio_fildes` with its path, and
/// then by opcode nothing (the syncs, NOOP), `aio_buf` (POLL), `aio_buf`, `aio_nbytes`,
/// `aio_offset` (PREAD, PWRITE), or an iovec `aio_buf` and `aio_offset` (PREADV, PWRITEV);
/// a request strace did not print whole is left out.
fn iocbs(s: &str) -> Vec<Iocb> {
    let s = s.trim();
    let Some(inner) = s.strip_prefix('[') else { return Vec::new() };
    let inner = inner.strip_suffix(']').unwrap_or(inner);
    split_args(inner).iter().filter_map(|e| iocb(e)).collect()
}

fn iocb(e: &str) -> Option<Iocb> {
    let body = e.strip_prefix('{')?.strip_suffix('}')?;
    let (mut data, mut op, mut fd, mut path, mut nbytes, mut offset, mut iov) = (None, None, None, None, 0, 0, None);
    for f in split_args(body) {
        let (k, v) = f.split_once('=')?;
        match k {
            "aio_data" => data = Some(v.to_string()),
            "aio_lio_opcode" => op = Some(v.strip_prefix("IOCB_CMD_").unwrap_or(v).to_string()),
            "aio_fildes" => (fd, path) = fd_arg(v),
            "aio_nbytes" => nbytes = v.parse().ok()?,
            "aio_offset" => offset = v.parse().ok()?,
            "aio_buf" if v.starts_with('[') => iov = Some(iov_lens(v)),
            _ => {}
        }
    }
    let (data, op, fd) = (data.filter(|d| !d.is_empty())?, op.filter(|o| !o.is_empty())?, fd?);
    Some(Iocb { data, op, fd, path, nbytes: iov.unwrap_or(nbytes), offset })
}

/// `{data=X, obj=Y, res=N` of every event an `io_getevents` returned.
fn events(s: &str) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = s[from..].find("{data=").map(|i| i + from) {
        from = i + 1;
        let data = word_at(s, i + 6);
        if data.is_empty() {
            continue;
        }
        let at = i + 6 + data.len();
        let Some(r) = s[at..].strip_prefix(", obj=") else { continue };
        let o = s.len() - r.len();
        let obj = word_at(s, o);
        if obj.is_empty() {
            continue;
        }
        let Some(r2) = s[o + obj.len()..].strip_prefix(", res=") else { continue };
        let p = s.len() - r2.len();
        let Some((res, end)) = num_at(s, p, true) else { continue };
        out.push((data.to_string(), res));
        from = end;
    }
    out
}

// ---------------------------------------------------------------- the tracer

/// An open file description: descriptors made by `dup` and inherited over `fork` share it.
#[derive(Debug)]
pub struct File {
    pub path: Option<String>,
    pub pos: i64,
    pub append: bool,
    /// A directory stream being read: its `getdents` calls are one `readdir` op together.
    pub listed: bool,
    /// The exporter's open id, when this description was opened under the root.
    pub oid: Option<usize>,
    /// A descriptor the trace shows no open of: held from before it began.
    pub held: bool,
    /// An open under the root the export left out (`O_PATH`, `O_TMPFILE`, the root itself).
    pub unexported: bool,
    /// The file the open holds, for the metrics (`metrics::Metrics::file`): its calls reach it
    /// whatever happens to its name after.
    pub file: Option<u64>,
}

type FileRef = Rc<RefCell<File>>;
type Table = Rc<RefCell<HashMap<i64, FileRef>>>;

fn new_file(path: Option<String>, pos: i64, append: bool) -> FileRef {
    Rc::new(RefCell::new(File { path, pos, append, listed: false, oid: None, held: false, unexported: false, file: None }))
}

const OPENS: [&str; 4] = ["open", "openat", "openat2", "creat"];
const READS: [&str; 5] = ["read", "pread64", "readv", "preadv", "preadv2"];
const WRITES: [&str; 5] = ["write", "pwrite64", "writev", "pwritev", "pwritev2"];
const CLONES: [&str; 4] = ["clone", "clone3", "fork", "vfork"];
const DUPS: [&str; 4] = ["dup", "dup2", "dup3", "fcntl"];

/// Descriptor calls that are one op of the abstract language each.
fn fd_op_of(name: &str) -> Option<&'static str> {
    Some(match name {
        "fstat" => "fstat",
        "fsync" => "fsync",
        "fdatasync" => "fdatasync",
        "ftruncate" => "ftruncate",
        "fallocate" => "fallocate",
        "getdents64" | "getdents" => "readdir",
        "fadvise64" => "fadvise",
        "ioctl" => "ioctl",
        _ => return None,
    })
}

/// Path calls: the op, and where the path is (dirfd index or none, path index).
fn path_op_of(name: &str) -> Option<(&'static str, Option<usize>, usize)> {
    Some(match name {
        "stat" | "lstat" => ("stat", None, 0),
        "newfstatat" | "statx" => ("stat", Some(0), 1),
        "unlink" => ("unlink", None, 0),
        "unlinkat" => ("unlink", Some(0), 1),
        "truncate" => ("truncate", None, 0),
        "mkdir" => ("mkdir", None, 0),
        "mkdirat" => ("mkdir", Some(0), 1),
        "rmdir" => ("rmdir", None, 0),
        "rename" => ("rename", None, 0),
        "renameat" | "renameat2" => ("rename", Some(0), 1),
        _ => return None,
    })
}

/// The calls that name a mapping by its address alone.
const MAPPING_CALLS: [&str; 6] = ["msync", "madvise", "munmap", "mremap", "mlock", "mlock2"];

/// Whether a call outside `handled` may be on the root: its arguments show a string or a
/// descriptor's annotation, or it is an `io_uring_setup`, a `sync`, or a call on a mapping;
/// the test before any parsing.
fn may_name_a_path(name: &str, args: &str) -> bool {
    matches!(name, "io_uring_setup" | "sync") || MAPPING_CALLS.contains(&name) || args.contains('"') || args.contains('<')
}

/// `0x7f0000000000` at the start of `s` as a number; `None` for anything else.
fn address(s: &str) -> Option<u64> {
    let h = s.trim_start().strip_prefix("0x")?;
    let k = h.bytes().position(|c| !c.is_ascii_hexdigit()).unwrap_or(h.len());
    u64::from_str_radix(&h[..k], 16).ok()
}

/// `aio_fildes` of `aio_fildes=3</p>`: a struct field's name.
fn is_field(k: &str) -> bool {
    !k.is_empty() && k.bytes().all(|c| c == b'_' || c.is_ascii_lowercase() || c.is_ascii_digit())
}

fn handled(name: &str) -> bool {
    OPENS.contains(&name)
        || READS.contains(&name)
        || WRITES.contains(&name)
        || CLONES.contains(&name)
        || DUPS.contains(&name)
        || fd_op_of(name).is_some()
        || path_op_of(name).is_some()
        || matches!(name, "close" | "lseek" | "io_setup" | "io_submit" | "io_getevents" | "io_pgetevents" | "chdir" | "fchdir" | "mmap" | "aeiou_uring_sqe" | "aeiou_uring_cqe" | "aeiou_fault" | "aeiou_aio_issue" | "aeiou_aio_complete")
        || crate::ebpf_text::PAIO_RECORDS.contains(&name)
}

/// When a call started and returned, and by whom: `(tid, issued, done)`.
pub type When = (i64, Option<f64>, Option<f64>);

/// A request submitted and not yet reaped.
#[derive(Debug, Clone)]
struct Waiting {
    kind: usize,
    inst: i64,
    tid: i64,
    path: String,
    file: Option<u64>,
    off: i64,
    nbytes: i64,
    member: Option<(usize, usize)>,
}

/// An `io_uring` request between its `aeiou_uring_sqe` and its last `aeiou_uring_cqe`: its
/// operation (`IORING_OP_` left off), the record's keyed arguments, and, for a read or a write,
/// what its completion counts.
struct Uring {
    op: String,
    tid: i64,
    issued: Option<f64>,
    args: HashMap<String, String>,
    data: Option<Waiting>,
    /// A request at the file's position (offset -1): the open file whose position it reads at
    /// issue and moves by its result, as a `read` or `write` does.
    cur: Option<FileRef>,
}

/// A POSIX AIO request between its issue and the helper's call that serves it (DESIGN_REVIEW.md
/// §3.98 "Phase 4 step 4"): the thread that issued it and when, a read or a write to count
/// (`None` off the roots), whether its file appends (the kernel ignores the offset glibc's
/// `pwrite64` passes), or the fsync it is, as the issuer's call (its name and descriptor).
struct Paio {
    tid: i64,
    issued: Option<f64>,
    data: Option<Waiting>,
    append: bool,
    sync: Option<(&'static str, String)>,
}

/// The `key=value` arguments of an `aeiou_` record (DESIGN_REVIEW.md §3.98).
fn keyed(a: &[String]) -> HashMap<String, String> {
    a.iter().filter_map(|x| x.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// Whether an `io_uring` operation moves file data: a read (`true`) or a write.
fn uring_reads(op: &str) -> Option<bool> {
    match op {
        "READ" | "READV" | "READ_FIXED" | "READ_MULTISHOT" => Some(true),
        "WRITE" | "WRITEV" | "WRITE_FIXED" => Some(false),
        _ => None,
    }
}

/// Reads `strace` lines and feeds the metrics, the exporter, or both.
pub struct Tracer {
    pub metrics: Option<TraceMetrics>,
    pub export: Option<Exporter>,
    pub roots: Vec<String>,
    exclude: Vec<String>,
    cwd: Option<String>,
    instance_roots: Vec<i64>,
    pub chain_gap: Option<f64>,
    tables: HashMap<i64, Table>,
    /// The tids of `tables` in the order they were first given one (the Python dict's order).
    order: Vec<i64>,
    parent: HashMap<i64, i64>,
    instance_of: HashMap<i64, Option<i64>>,
    cwds: HashMap<i64, String>,
    pending: HashMap<i64, (String, String, Option<f64>)>,
    closing: HashMap<i64, FileRef>,
    chain: HashMap<i64, (u64, Option<f64>)>,
    /// Requests by `(context, aio_data)`, with the order the key was made in (the Python
    /// dict's), which is the order the unreaped are counted in at the end.
    aio: HashMap<(String, String), (u64, Vec<Waiting>)>,
    aio_seq: u64,
    /// The trace's format (`--- aeiou: trace format N` at its start; 0, a plain strace, without).
    pub format: u64,
    /// The trace says a first touch may not fault (`Some`), and whether a fault under the roots
    /// has been counted since, which notes it.
    first_touch_unseen: Option<bool>,
    /// `io_uring` requests in flight, by `(ring, id)` (`aeiou_uring_sqe`), at most a ring's depth.
    uring: HashMap<(String, String), Uring>,
    /// A thread's data requests submitted since its last other line: one fan-out.
    uring_batch: HashMap<i64, Vec<(String, String)>>,
    /// An `io_submit` was counted: its rounds' depth needs `--chain-gap-us` (`io_uring`'s are not chained).
    aio_submitted: bool,
    /// The trace's header lists libaio's completions (`aeiou_aio_issue`, `aeiou_aio_complete`,
    /// DESIGN_REVIEW.md §3.98 "Phase 4 step 3"): a request is counted at its completion, and
    /// `io_getevents` only collects. Then each thread's last `io_submit`'s requests by their place
    /// in its array, until its `aeiou_aio_issue`s number them; the requests in flight by
    /// `(context, id)`; and the completions and the results collected, in all.
    aio_records: bool,
    aio_members: HashMap<i64, (String, HashMap<usize, Waiting>)>,
    aio_ids: HashMap<(String, String), Waiting>,
    aio_completed: u64,
    aio_reaped: u64,
    aio_unused: BTreeMap<String, u64>,
    /// POSIX AIO's requests by number (`aeiou_aio_read`, ...) until served; the request each
    /// helper thread's next call serves (`aeiou_aio_serve`); those served whose result no
    /// `aio_return` has collected.
    paio: HashMap<String, Paio>,
    serving: HashMap<i64, String>,
    paio_uncollected: std::collections::HashSet<String>,
    clock: f64,
    pub notes: BTreeMap<String, u64>,
    pub tids: std::collections::HashSet<i64>,
    under_cache: HashMap<String, Option<String>>,
    /// The size of each file under the roots the trace made known (created, truncated, or
    /// renamed into place since), kept as its writes extend it: an `O_APPEND` write's base.
    extent: HashMap<String, i64>,
    /// The address ranges `mmap` returned for files under a root (start → length), so a call
    /// that names only an address can be told to be on one
    maps: BTreeMap<u64, u64>,
    /// The address a call returned (`mmap`, `mremap`), `None` for a number or an error
    ret_addr: Option<u64>,
    pub seen: Seen,
    error: Option<anyhow::Error>,
}

impl Tracer {
    pub fn new(o: &Opts, metrics: bool, export: Option<Exporter>) -> Tracer {
        Tracer {
            metrics: metrics.then(|| TraceMetrics::new(MetricsOpts { block: o.block })),
            export,
            roots: o.roots.iter().map(|r| normpath(r)).collect(),
            exclude: o.exclude.clone(),
            cwd: o.cwd.clone(),
            instance_roots: o.instance_roots.clone(),
            chain_gap: o.chain_gap_us.map(|us| us / 1e6),
            tables: HashMap::new(),
            order: Vec::new(),
            parent: HashMap::new(),
            instance_of: HashMap::new(),
            cwds: HashMap::new(),
            pending: HashMap::new(),
            closing: HashMap::new(),
            chain: HashMap::new(),
            aio: HashMap::new(),
            aio_seq: 0,
            format: 0,
            first_touch_unseen: None,
            uring: HashMap::new(),
            uring_batch: HashMap::new(),
            aio_submitted: false,
            aio_records: false,
            aio_members: HashMap::new(),
            aio_ids: HashMap::new(),
            aio_completed: 0,
            aio_reaped: 0,
            aio_unused: BTreeMap::new(),
            paio: HashMap::new(),
            serving: HashMap::new(),
            paio_uncollected: Default::default(),
            clock: 0.0,
            notes: BTreeMap::new(),
            tids: Default::default(),
            under_cache: HashMap::new(),
            extent: HashMap::new(),
            maps: BTreeMap::new(),
            ret_addr: None,
            seen: Seen::default(),
            error: None,
        }
    }

    fn note(&mut self, k: &str, n: u64) {
        *self.notes.entry(k.to_string()).or_insert(0) += n;
    }

    /// The open id the export names `f` by. A descriptor whose open the trace does not show
    /// (opened before `strace` began: attached with `-p`, inherited, received over a socket)
    /// gets an open synthesized at this, its first use, so its calls are replayed and the
    /// header says the trace is incomplete (`Exporter::synthesize`); one with no path to open
    /// (no `-yy`) is left out and noted by name.
    fn oid_or_note(&mut self, f: Option<&Rc<RefCell<File>>>, op: &str, when: When) -> Result<Option<usize>> {
        if self.export.is_none() {
            return Ok(None);
        }
        if let Some(oid) = f.and_then(|f| f.borrow().oid) {
            return Ok(Some(oid));
        }
        let path = f.and_then(|f| f.borrow().path.clone());
        let under = self.under(path.as_deref());
        let ex = self.export.as_mut().unwrap();
        match (f.filter(|f| f.borrow().held), under) {
            (Some(f), Some(p)) => {
                let oid = ex.synthesize(when, &p, op)?;
                f.borrow_mut().oid = Some(oid);
                Ok(Some(oid))
            }
            _ => {
                let why = if f.is_some_and(|f| f.borrow().unexported) { "its open not exported" } else { "descriptor without open" };
                ex.note("calls_not_exported", 1);
                ex.note(&format!("not_exported:{op}:{why}"), 1);
                Ok(None)
            }
        }
    }

    // ---- paths

    /// The path as counted (normalized), or `None` when it is outside the roots or excluded.
    pub fn under(&mut self, path: Option<&str>) -> Option<String> {
        let path = path?;
        if let Some(hit) = self.under_cache.get(path) {
            return hit.clone();
        }
        let p = normpath(path);
        let mut out = None;
        for r in &self.roots {
            let base = format!("{}/", r.trim_end_matches('/'));
            if p == *r || p.starts_with(&base) {
                let rel = p.get(base.len()..).unwrap_or("");
                if !self.exclude.iter().any(|g| glob(rel, g)) {
                    out = Some(p.clone());
                }
                break;
            }
        }
        if self.under_cache.len() < UNDER_CACHE {
            self.under_cache.insert(path.to_string(), out.clone());
        }
        out
    }

    /// A call no op covers, noted by name when it names a path under a root (the man page's
    /// "dropped and counted by name"); an `io_uring_setup` always, since the I/O of its ring
    /// is not in an strace, unless the trace holds the ring's requests (an eBPF capture's,
    /// its header's format 1 and on).
    fn not_counted(&mut self, tid: i64, name: &str, a: &[String]) {
        let hit = match name {
            "io_uring_setup" => self.format == 0,
            "sync" => true,
            _ if MAPPING_CALLS.contains(&name) => self.on_a_root_mapping(name, a),
            _ => self.names_a_root_path(tid, a),
        };
        if hit && self.instance(tid).is_some() {
            self.note(&format!("not_counted:{name}"), 1);
        }
    }

    /// Whether a call naming an address range (`a[0]`, `a[1]`) is on a mapping of a file under
    /// a root, the ranges kept as the kernel does: `munmap` takes its range out of them, and a
    /// successful `mremap` moves the mapping it names to the address it returned. Address
    /// spaces are not told apart: a range is the last mapping of a root file there.
    fn on_a_root_mapping(&mut self, name: &str, a: &[String]) -> bool {
        let (Some(addr), Some(len)) = (a.first().and_then(|t| address(t)), a.get(1).and_then(|t| int0(t)).filter(|&l| l > 0)) else { return false };
        let end = addr.saturating_add(len as u64);
        // the ranges of one address space never overlap, so the scan down from `end` stops at
        // the first that ends at or before `addr`
        let over: Vec<(u64, u64)> = self.maps.range(..end).rev().take_while(|(&s, &l)| s.saturating_add(l) > addr).map(|(&s, &l)| (s, l)).collect();
        if over.is_empty() {
            return false;
        }
        match name {
            "munmap" => {
                for (s, l) in &over {
                    self.maps.remove(s);
                    if *s < addr {
                        self.maps.insert(*s, addr - s);
                    }
                    if s.saturating_add(*l) > end {
                        self.maps.insert(end, s.saturating_add(*l) - end);
                    }
                }
            }
            "mremap" => {
                if let (Some(new), Some(n)) = (self.ret_addr, a.get(2).and_then(|t| int0(t)).filter(|&n| n > 0)) {
                    for (s, _) in &over {
                        self.maps.remove(s);
                    }
                    self.maps.insert(new, n as u64);
                }
            }
            _ => {}
        }
        true
    }

    /// Whether arguments name a path under a root: a descriptor's `<path>`, inside a struct or
    /// an array too; a string first, after a directory descriptor, or absolute after another
    /// path, as a path argument stands (never a buffer's bytes elsewhere).
    fn names_a_root_path(&mut self, tid: i64, a: &[String]) -> bool {
        let mut after_path = false;
        for (i, t) in a.iter().enumerate() {
            let t = t.split_once('=').filter(|(k, _)| is_field(k)).map_or(t.as_str(), |(_, v)| v);
            let was = std::mem::take(&mut after_path);
            if t.starts_with('{') || t.starts_with('[') {
                let inner = &t[1..t.len().saturating_sub(1).max(1)];
                if self.names_a_root_path(tid, &split_args(inner)) {
                    return true;
                }
                continue;
            }
            if let (_, Some(p)) = fd_arg(t) {
                if self.under(Some(&p)).is_some() {
                    return true;
                }
                after_path = p.starts_with('/');
                continue;
            }
            let Some(p) = c_string(t).filter(|p| !p.is_empty() && !truncated(t)) else { continue };
            let dir = i.checked_sub(1).and_then(|j| fd_arg(&a[j]).1).filter(|d| d.starts_with('/'));
            let full = if p.starts_with('/') && (i == 0 || dir.is_some() || was) {
                p
            } else if let Some(d) = dir {
                join(&d, &p)
            } else if i == 0 {
                match self.cwds.get(&tid).or(self.cwd.as_ref()).filter(|c| !c.is_empty()) {
                    Some(c) => join(c, &p),
                    None => continue,
                }
            } else {
                continue;
            };
            if self.under(Some(&full)).is_some() {
                return true;
            }
            after_path = true;
        }
        false
    }

    fn resolve(&mut self, tid: i64, dirfd_tok: Option<&str>, path: Option<String>) -> Option<String> {
        let path = path?;
        if path.starts_with('/') {
            return Some(path);
        }
        let mut base = None;
        if let Some(tok) = dirfd_tok {
            let (fd, b) = fd_arg(tok);
            base = b;
            if base.is_none() {
                if let Some(fd) = fd {
                    base = self.table(tid).borrow().get(&fd).and_then(|f| f.borrow().path.clone());
                }
            }
        }
        if base.is_none() {
            base = self.cwds.get(&tid).filter(|s| !s.is_empty()).cloned().or_else(|| self.cwd.clone().filter(|s| !s.is_empty()));
        }
        match base {
            Some(b) => Some(join(&b, &path)),
            None => {
                self.note("relative_paths_skipped", 1);
                None
            }
        }
    }

    // ---- processes

    fn set_table(&mut self, tid: i64, t: Table) {
        if self.tables.insert(tid, t).is_none() {
            self.order.push(tid);
        }
    }

    fn table(&mut self, tid: i64) -> Table {
        if let Some(t) = self.tables.get(&tid) {
            return t.clone();
        }
        self.adopt(tid)
    }

    /// A tid seen before its `clone` returned (or with no `clone` in the trace): the child of
    /// the one unfinished `clone`, if there is exactly one.
    fn adopt(&mut self, tid: i64) -> Table {
        let waiting: Vec<(i64, String, String)> =
            self.pending.iter().filter(|(p, pend)| CLONES.contains(&pend.0.as_str()) && **p != tid).map(|(p, pend)| (*p, pend.0.clone(), pend.1.clone())).collect();
        if waiting.len() == 1 {
            let (p, name, body) = &waiting[0];
            return self.child(*p, tid, shares_files(name, body));
        }
        if !self.tables.is_empty() {
            self.note("threads_without_clone", 1);
        }
        let t: Table = Rc::new(RefCell::new(HashMap::new()));
        self.set_table(tid, t.clone());
        t
    }

    fn child(&mut self, parent: i64, child: i64, shares: bool) -> Table {
        let ptable = self.table(parent);
        let old = self.tables.get(&child).cloned();
        if let Some(o) = &old {
            if self.parent.get(&child) == Some(&parent) {
                return o.clone();
            }
        }
        let t = if shares { ptable } else { Rc::new(RefCell::new(ptable.borrow().clone())) };
        if let Some(o) = old {
            let entries: Vec<(i64, FileRef)> = o.borrow().iter().map(|(k, v)| (*k, v.clone())).collect();
            t.borrow_mut().extend(entries);
        }
        self.set_table(child, t.clone());
        self.parent.insert(child, parent);
        self.instance_of.remove(&child);
        if let Some(c) = self.cwds.get(&parent).cloned() {
            self.cwds.entry(child).or_insert(c);
        }
        t
    }

    /// The instance `tid` belongs to: 0 when the trace is one instance, else the
    /// `--instance-root` at or above it, or `None` when there is none.
    fn instance(&mut self, tid: i64) -> Option<i64> {
        if self.instance_roots.is_empty() {
            return Some(0);
        }
        if let Some(i) = self.instance_of.get(&tid) {
            return *i;
        }
        let mut t = Some(tid);
        let mut seen = std::collections::HashSet::new();
        while let Some(x) = t {
            if self.instance_roots.contains(&x) || !seen.insert(x) {
                break;
            }
            t = self.parent.get(&x).copied();
        }
        let out = t.filter(|x| self.instance_roots.contains(x));
        self.instance_of.insert(tid, out);
        out
    }

    /// The open file behind `fd` as `tid` sees it. The `-yy` path wins over a stale table.
    fn file(&mut self, tid: i64, fd: Option<i64>, path: Option<&str>) -> Option<FileRef> {
        let fd = fd?;
        let table = self.table(tid);
        let f = table.borrow().get(&fd).cloned();
        if let Some(f) = &f {
            if path.is_none() || f.borrow().path.as_deref() == path {
                return Some(f.clone());
            }
        }
        let path = path?;
        // not this thread's open: a thread sharing another's table, traced without `clone`
        for t in self.order.clone() {
            let other = self.tables[&t].clone();
            let g = other.borrow().get(&fd).cloned();
            if let Some(g) = g {
                if g.borrow().path.as_deref() == Some(path) && !Rc::ptr_eq(&other, &table) {
                    if table.borrow().is_empty() {
                        for t2 in self.order.clone() {
                            if Rc::ptr_eq(&self.tables[&t2], &table) {
                                self.tables.insert(t2, other.clone());
                            }
                        }
                    } else {
                        table.borrow_mut().insert(fd, g.clone());
                    }
                    return Some(g);
                }
            }
        }
        let f = new_file(Some(path.to_string()), 0, false);
        f.borrow_mut().held = true;
        table.borrow_mut().insert(fd, f.clone());
        if self.under(Some(path)).is_some() {
            // opened before the trace began: its position is taken as 0
            self.note("descriptors_without_open", 1);
        }
        Some(f)
    }

    // ---- the stream

    pub fn feed(&mut self, input: impl BufRead) -> Result<()> {
        for_each_line(input, |line| self.feed_line(line))
    }

    /// One line of the trace, counted as read; the error a call raised, if any.
    pub fn feed_line(&mut self, line: &str) -> Result<()> {
        self.seen.lines += 1;
        self.line(line);
        match self.error.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn m(&mut self) -> Option<&mut TraceMetrics> {
        self.metrics.as_mut()
    }

    pub fn line(&mut self, line: &str) {
        let line = line.strip_suffix('\n').unwrap_or(line);
        let (tid, ts_text, body) = prefix(line);
        let mut ts = None;
        if let Some(t) = ts_text {
            let mut v = seconds(t);
            if v < 1e6 && !t.contains(':') {
                // `-r`: relative to the previous line
                self.clock += v;
                v = self.clock;
            }
            ts = Some(v);
        }
        if !self.uring_batch.is_empty() && !body.starts_with("aeiou_uring_sqe(") {
            self.flush_uring(tid);
        }
        if body.starts_with("+++") {
            if let Some(m) = self.m() {
                m.end_context(tid);
            }
            self.end_chain(tid);
            return;
        }
        if let Some(r) = body.strip_prefix("--- aeiou: trace format ") {
            // the eBPF source's header: the format's version and the record kinds it may hold
            let v: u64 = r.split([',', ' ']).next().and_then(|n| n.parse().ok()).unwrap_or(0);
            self.format = v;
            self.aio_records = r.split_once(", records ").is_some_and(|(_, k)| k.split(' ').any(|k| k == "aeiou_aio_complete"));
            if v > TRACE_FORMAT {
                self.note("trace_format_newer", v);
            }
            return;
        }
        if body.starts_with("--- aeiou: faults: ") {
            // the eBPF source's mark that its fault-around check was waived (ebpf_text.rs), noted
            // with the first fault counted: a trace with none under the roots is a call trace
            self.first_touch_unseen = Some(false);
            return;
        }
        if let Some(r) = body.strip_prefix("--- aeiou: ") {
            // the eBPF source's mark where its ring buffer lost records (capture.rs)
            if let Some(n) = r.split(' ').next().and_then(|n| n.parse().ok()).filter(|_| r.contains(" event(s) lost")) {
                self.note(EVENTS_LOST, n);
            }
            return;
        }
        if body.starts_with("---") || body.is_empty() {
            return;
        }
        let call = match resumed(body) {
            Some((name, rest)) => Some((name, rest)),
            None => body.find('(').map(|i| (&body[..i], &body[i..])),
        };
        if let Some(k) = call.and_then(|(name, args)| printed(name, args)) {
            self.note(k, 1);
        }
        let mut issued = ts;
        let owned: String;
        let mut body = body;
        let is_resumed = match resumed(body) {
            Some((name, rest)) => {
                let Some(pend) = self.pending.remove(&tid) else { return };
                if pend.0 != name {
                    return;
                }
                owned = format!("{}{}", pend.1, rest);
                body = &owned;
                issued = pend.2;
                true
            }
            None => false,
        };
        if !is_resumed && body.ends_with(UNFINISHED) {
            let name = body.split('(').next().unwrap_or("");
            let head = &body[..body.len() - UNFINISHED.len()];
            if handled(name) || may_name_a_path(name, head) {
                self.pending.insert(tid, (name.to_string(), head.to_string(), ts));
            }
            if name == "close" {
                // the kernel frees the descriptor as the close begins: another thread's open
                // can take it before this resumes
                let a = split_args(head.get(6..).unwrap_or("").trim_end());
                let (fd, path) = a.first().map(|x| fd_arg(x)).unwrap_or((None, None));
                let table = self.table(tid);
                let f = fd.and_then(|fd| table.borrow().get(&fd).cloned());
                if let Some(f) = f {
                    if path.is_none() || f.borrow().path == path {
                        table.borrow_mut().remove(&fd.unwrap());
                        self.closing.insert(tid, f);
                    }
                }
            }
            return;
        }
        let Some(i) = body.find('(') else { return };
        if i == 0 {
            return;
        }
        let name = &body[..i];
        if !handled(name) {
            if let Some((k, r)) = ret_split(body).filter(|&(k, _)| may_name_a_path(name, &body[i..k])) {
                let args = split_args(&body[i + 1..k]);
                self.ret_addr = address(&body[r..]);
                self.not_counted(tid, name, &args);
            }
            return;
        }
        let Some((k, r)) = ret_split(body) else { return };
        let ret_s = &body[r..];
        let Some((tok, tl)) = ret_token(ret_s) else { return };
        if tok == "?" {
            return;
        }
        let Some(ret) = int0(tok) else { return };
        let rest = &ret_s[tl..];
        let err = if ret < 0 {
            let r = rest.trim_start();
            let b = r.as_bytes();
            if b.first() == Some(&b'E') {
                let e = b[1..].iter().position(|c| !(c.is_ascii_uppercase() || c.is_ascii_digit())).map_or(b.len(), |p| p + 1);
                (e > 1).then(|| r[..e].to_string())
            } else {
                None
            }
        } else {
            None
        };
        let mut ret_path = None;
        if rest.starts_with('<') {
            let t = match rest.find(" <") {
                Some(j) => format!("{tok}{}", &rest[..j]),
                None => format!("{tok}{rest}"),
            };
            ret_path = fd_arg(&t).1;
        }
        let dur = ret_s.strip_suffix('>').and_then(|s| {
            let j = s.rfind('<')?;
            let d = &s[j + 1..];
            let (a, b) = d.split_once('.')?;
            (digits_only(a) && digits_only(b)).then(|| d.parse::<f64>().ok()).flatten()
        });
        let done = if is_resumed {
            ts
        } else {
            match (issued, dur) {
                (Some(t), Some(d)) => Some(t + d),
                _ => issued,
            }
        };
        self.tids.insert(tid);
        self.seen.calls += 1;
        let args = split_args(&body[i + 1..k]);
        self.ret_addr = address(ret_s);
        let name = name.to_string();
        if let Err(e) = self.call(tid, &name, &args, ret, ret_path, issued, done, err) {
            self.error = Some(e);
        }
    }

    // ---- faults on mappings (DESIGN_REVIEW.md §3.98 "Phase 4 step 2")

    /// An `aeiou_fault`: a run of a thread's faults on a file, counted apart from the calls, and
    /// not replayed (the export keeps a mapping as one read of its range, `mmap_as_read`).
    fn fault(&mut self, tid: i64, inst: Option<i64>, a: &[String], issued: Option<f64>) {
        let k = keyed(a);
        let Some(path) = k.get("path").and_then(|p| c_string(p)) else { return };
        let Some(p) = self.under(Some(&path)) else { return };
        if inst.is_none() {
            return;
        }
        let n = |key: &str| k.get(key).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
        let count = n("count").max(0) as u64;
        if self.first_touch_unseen == Some(false) {
            self.first_touch_unseen = Some(true);
            self.note(FIRST_TOUCH_UNSEEN, 1);
        }
        if let Some(ex) = self.export.as_mut() {
            ex.note("faults_not_replayed", count);
        }
        if let Some(m) = self.metrics.as_mut() {
            m.faults.get_or_insert_with(Default::default).run(tid, &p, n("stride"), count, n("major").max(0) as u64, n("write") == 1, n("in_call") == 1, n("page").max(1) as u64, issued.unwrap_or(0.0), n("span_ns").max(0) as u64);
        }
    }

    // ---- io_uring (DESIGN_REVIEW.md §3.98)

    /// An `aeiou_uring_sqe`: the request held until its completion, a read or a write also in
    /// the thread's batch, the fan-out its `io_uring_enter` submits.
    fn uring_sqe(&mut self, tid: i64, inst: Option<i64>, a: &[String], issued: Option<f64>) {
        let args = keyed(a);
        let (Some(ring), Some(id), Some(op)) = (args.get("ring"), args.get("id"), args.get("op")) else { return };
        let key = (ring.clone(), id.clone());
        let op = op.strip_prefix("IORING_OP_").unwrap_or(op).to_string();
        let mut data = None;
        let mut cur = None;
        if let (Some(read), Some(inst)) = (uring_reads(&op), inst) {
            let (fd, path) = self.uring_file(&args);
            let f = self.file(tid, fd, path.as_deref());
            let fp = path.or_else(|| f.as_ref().and_then(|f| f.borrow().path.clone()));
            if let Some(p) = self.under(fp.as_deref()) {
                let int = |k: &str| args.get(k).and_then(|v| int0(v));
                let held = f.as_ref().and_then(|f| f.borrow().file);
                // a vectored request's length is its result's (the probe copies no iovec)
                let nbytes = int("len").unwrap_or(-1);
                let mut off = int("off").unwrap_or(-1);
                if off == -1 {
                    if let Some(f) = &f {
                        off = f.borrow().pos;
                        cur = Some(f.clone());
                    }
                }
                data = Some(Waiting { kind: if read { R } else { W }, inst, tid, path: p, file: held, off, nbytes, member: None });
                self.uring_batch.entry(tid).or_default().push(key.clone());
            }
        }
        if self.uring.insert(key, Uring { op, tid, issued, args, data, cur }).is_some() {
            self.note("uring_submissions_without_completion", 1);
        }
    }

    /// The descriptor and path a request names: `fd=` the descriptor (`-yy`'s form), `fixed=`
    /// a registered file's index with the path the converter resolved it to.
    fn uring_file(&self, args: &HashMap<String, String>) -> (Option<i64>, Option<String>) {
        let (fd, mut path) = args.get("fd").map(|v| fd_arg(v)).unwrap_or((None, None));
        if path.is_none() {
            path = args.get("fixed").and_then(|v| fd_arg(v).1);
        }
        (fd, path.filter(|p| !p.is_empty()))
    }

    /// The reads and writes `tid` submitted since its last other line, as one fan-out (and,
    /// exported, one `submit`, as an `io_submit`'s are).
    fn flush_uring(&mut self, tid: i64) {
        let Some(keys) = self.uring_batch.remove(&tid) else { return };
        if let Some(m) = self.metrics.as_mut() {
            *m.m.fan_out.entry(keys.len() as u64).or_insert(0) += 1;
        }
        if self.export.is_none() {
            return;
        }
        let mut mem = Vec::new();
        let mut at = Vec::new();
        let mut when: When = (tid, None, None);
        for k in &keys {
            let Some(u) = self.uring.get(k) else { continue };
            let Some(w) = &u.data else { continue };
            if u.cur.is_some() {
                // written at its completion, as the sequential call it is
                continue;
            }
            let kind = if w.kind == R { "read" } else { "write" };
            // what a submit cannot say: a length the record does not hold, results past the first
            let left_out = if w.nbytes < 0 {
                Some(format!("not_exported:{kind}v:length not traced"))
            } else if u.op == "READ_MULTISHOT" {
                Some(format!("not_exported:{kind}:multishot"))
            } else {
                None
            };
            if let Some(n) = left_out {
                let ex = self.export.as_mut().unwrap();
                ex.note("calls_not_exported", 1);
                ex.note(&n, 1);
                continue;
            }
            when.1 = when.1.or(u.issued);
            when.2 = u.issued;
            let (fd, path) = self.uring_file(&u.args);
            let (off, nbytes) = (w.off, w.nbytes);
            let f = self.file(tid, fd, path.as_deref()).or_else(|| path.map(|p| new_file(Some(p), 0, false)));
            if let Ok(Some(oid)) = self.oid_or_note(f.as_ref(), kind, when) {
                mem.push((kind, oid, off, nbytes));
                at.push(k.clone());
            }
        }
        if mem.is_empty() {
            return;
        }
        let members = self.export.as_mut().unwrap().submit(when, &mem);
        for (k, m) in at.into_iter().zip(members) {
            if let Some(w) = self.uring.get_mut(&k).and_then(|u| u.data.as_mut()) {
                w.member = Some(m);
            }
        }
    }

    /// An `aeiou_uring_cqe`: its request's read or write counted with the bytes it moved, an
    /// open, a close, or a sync counted as the call it does, and anything else on a root by its
    /// operation's name. A completion with `IORING_CQE_F_MORE` leaves its request in flight.
    #[allow(clippy::too_many_arguments)]
    fn uring_cqe(&mut self, _tid: i64, a: &[String], ret: i64, ret_path: Option<String>, done: Option<f64>, err: Option<&str>) -> Result<()> {
        let args = keyed(a);
        let (Some(ring), Some(id)) = (args.get("ring"), args.get("id")) else { return Ok(()) };
        let key = (ring.clone(), id.clone());
        let more = args.get("flags").is_some_and(|f| f.contains("IORING_CQE_F_MORE"));
        let Some(mut u) = self.uring.remove(&key) else {
            self.note("uring_completions_without_submission", 1);
            return Ok(());
        };
        if let Some(w) = &u.data {
            let n = if ret >= 0 { ret } else { 0 };
            // a request of no length the probe saw (vectored, or a buffer the kernel picks) is its result's
            let len = if w.nbytes < 0 || (w.nbytes == 0 && n > 0) { n } else { w.nbytes };
            self.data(w.kind, w.inst, w.tid, &w.path, w.file, w.off, len, n)?;
            if w.kind == W && n > 0 && w.off >= 0 {
                if let Some(e) = self.extent.get_mut(&w.path) {
                    *e = (*e).max(w.off + n);
                }
            }
            if let (Some(m), Some(ex)) = (w.member, self.export.as_mut()) {
                ex.member_ret(m, if ret >= 0 { json!(ret) } else { json!(err.unwrap_or("EIO")) });
            }
            if let Some(f) = u.cur.clone() {
                if n > 0 {
                    f.borrow_mut().pos = w.off + n;
                }
                let (kind, off, issued) = (if w.kind == R { "read" } else { "write" }, w.off, u.issued);
                if let Some(oid) = self.oid_or_note(Some(&f), kind, (u.tid, issued, done))? {
                    self.export.as_mut().unwrap().data((u.tid, issued, done), kind, oid, off, len, false, ret, err);
                }
            }
            if more {
                let pos = u.cur.as_ref().map(|f| f.borrow().pos);
                let d = u.data.as_mut().unwrap();
                d.member = None;
                // the next result of a request at the position is where this one left it
                if let Some(p) = pos {
                    d.off = p;
                }
                self.uring.insert(key, u);
            }
            return Ok(());
        }
        let file = |k: &str| u.args.get(k).cloned();
        let call: Option<(&str, Vec<String>)> = match u.op.as_str() {
            "OPENAT" | "OPENAT2" if u.args.contains_key("file_slot") => {
                // installed in the ring's table, not the process's: no descriptor to follow
                let p = self.uring_open_path(&u);
                if self.under(p.as_deref()).is_some() {
                    self.note("uring_direct_opens_not_counted", 1);
                }
                None
            }
            "OPENAT" => Some(("openat", vec![file("dirfd").unwrap_or("AT_FDCWD".into()), file("path").unwrap_or_default(), file("flags").unwrap_or("O_RDONLY".into()), file("mode").unwrap_or("0".into())])),
            "OPENAT2" => Some((
                "openat2",
                vec![
                    file("dirfd").unwrap_or("AT_FDCWD".into()),
                    file("path").unwrap_or_default(),
                    format!("{{flags={}, mode={}, resolve={}}}", file("flags").unwrap_or("O_RDONLY".into()), file("mode").unwrap_or("0".into()), file("resolve").unwrap_or("0".into())),
                    "24".into(),
                ],
            )),
            "CLOSE" if !u.args.contains_key("file_slot") => file("fd").map(|f| ("close", vec![f])),
            "FSYNC" => {
                let data = u.args.get("fsync_flags").is_some_and(|f| f.contains("IORING_FSYNC_DATASYNC"));
                file("fd").map(|f| (if data { "fdatasync" } else { "fsync" }, vec![f]))
            }
            // as strace prints statx: on a descriptor (AT_EMPTY_PATH) an fstat, on a path a stat
            "STATX" => Some(("statx", vec![file("dirfd").unwrap_or("AT_FDCWD".into()), file("path").unwrap_or_default(), file("flags").unwrap_or("0".into()), file("mask").unwrap_or("0".into()), "0x1".into()])),
            _ => {
                let (_, path) = self.uring_file(&u.args);
                // an operation on paths the record does not hold: whether they are under a root is not known
                let pathless = matches!(u.op.as_str(), "RENAMEAT" | "UNLINKAT" | "MKDIRAT" | "SYMLINKAT" | "LINKAT" | "SETXATTR" | "GETXATTR");
                if (pathless || self.under(path.as_deref()).is_some()) && self.instance(u.tid).is_some() {
                    self.note(&format!("not_counted:aeiou_uring_sqe:IORING_OP_{}", u.op), 1);
                }
                None
            }
        };
        if more {
            self.uring.insert(key, Uring { op: u.op.clone(), tid: u.tid, issued: u.issued, args: u.args.clone(), data: None, cur: None });
        }
        match call {
            Some((name, a)) => self.call(u.tid, name, &a, ret, ret_path, u.issued, done, err.map(str::to_string)),
            None => Ok(()),
        }
    }

    /// An `aeiou_aio_complete`: its request's read or write counted with the bytes it moved.
    fn aio_complete(&mut self, a: &[String], ret: i64, err: Option<&str>) -> Result<()> {
        let args = keyed(a);
        self.aio_completed += 1;
        let (Some(ctx), Some(id)) = (args.get("ctx"), args.get("id")) else { return Ok(()) };
        let Some(w) = self.aio_ids.remove(&(ctx.clone(), id.clone())) else { return Ok(()) };
        if args.get("in_submit").is_some_and(|v| v == "1") {
            self.note("aio_completed_in_submit", 1);
        }
        let n = ret.max(0);
        self.data(w.kind, w.inst, w.tid, &w.path, w.file, w.off, w.nbytes, n)?;
        if w.kind == W && n > 0 {
            if let Some(e) = self.extent.get_mut(&w.path) {
                *e = (*e).max(w.off + n);
            }
        }
        if let (Some(m), Some(ex)) = (w.member, self.export.as_mut()) {
            ex.member_ret(m, if ret >= 0 { json!(ret) } else { json!(err.unwrap_or("EIO")) });
        }
        Ok(())
    }

    /// A POSIX AIO issue: each request held until the helper's call that serves it, a read or a
    /// write under a root one member of the fan-out the call is (and, exported, of one `submit`
    /// in the issuer's lane).
    fn paio_issue(&mut self, tid: i64, inst: Option<i64>, name: &str, a: &[String], issued: Option<f64>) -> Result<()> {
        let k = keyed(a);
        let reqs: Vec<HashMap<String, String>> = if name == "aeiou_lio_listio" {
            let list = k.get("list").map(|l| l.trim_start_matches('[').trim_end_matches(']').to_string()).unwrap_or_default();
            split_args(&list).iter().map(|m| keyed(&split_args(m.trim_start_matches('{').trim_end_matches('}')))).collect()
        } else {
            let mut r = k.clone();
            match name {
                "aeiou_aio_read" => r.insert("op".into(), "LIO_READ".into()),
                "aeiou_aio_write" => r.insert("op".into(), "LIO_WRITE".into()),
                _ => None,
            };
            vec![r]
        };
        let mut mem = Vec::new();
        let mut at = Vec::new();
        let mut fan = 0u64;
        for r in reqs {
            let Some(id) = r.get("id").cloned() else { continue };
            if self.paio_uncollected.remove(&id) {
                // its number free again: the request before was done, and no aio_return collected it
                self.note("posix_aio_never_collected", 1);
            }
            let fd_tok = r.get("fd").cloned().unwrap_or_default();
            let (fd, path) = fd_arg(&fd_tok);
            let f = self.file(tid, fd, path.as_deref());
            let fp = path.or_else(|| f.as_ref().and_then(|f| f.borrow().path.clone()));
            let p = self.under(fp.as_deref());
            let mut req = Paio { tid, issued, data: None, append: false, sync: None };
            let read = match r.get("op").map(String::as_str) {
                Some("LIO_READ") => Some(true),
                Some("LIO_WRITE") => Some(false),
                Some("O_DSYNC") => {
                    req.sync = Some(("fdatasync", fd_tok.clone()));
                    None
                }
                _ => {
                    req.sync = Some(("fsync", fd_tok.clone()));
                    None
                }
            };
            if let (Some(read), Some(p), Some(inst)) = (read, p, inst) {
                let int = |key: &str| r.get(key).and_then(|v| int0(v)).unwrap_or(-1);
                let held = f.as_ref().and_then(|f| f.borrow().file);
                req.append = !read && f.as_ref().is_some_and(|f| f.borrow().append);
                let w = Waiting { kind: if read { R } else { W }, inst, tid, path: p, file: held, off: int("off"), nbytes: int("len"), member: None };
                if let Some(Ok(Some(oid))) = self.export.is_some().then(|| self.oid_or_note(f.as_ref(), if read { "read" } else { "write" }, (tid, issued, issued))) {
                    mem.push((if read { "read" } else { "write" }, oid, w.off, w.nbytes));
                    at.push(id.clone());
                }
                req.data = Some(w);
                fan += 1;
            }
            if self.paio.insert(id, req).is_some() {
                self.note("posix_aio_never_served", 1);
            }
        }
        if fan > 0 {
            if let Some(m) = self.metrics.as_mut() {
                *m.m.fan_out.entry(fan).or_insert(0) += 1;
            }
        }
        if !mem.is_empty() {
            let members = self.export.as_mut().unwrap().submit((tid, issued, issued), &mem);
            for (id, m) in at.into_iter().zip(members) {
                if let Some(w) = self.paio.get_mut(&id).and_then(|p| p.data.as_mut()) {
                    w.member = Some(m);
                }
            }
        }
        Ok(())
    }

    /// A POSIX AIO request never done: its member of the export's `submit` left out, said why.
    fn paio_not_done(&mut self, p: Paio, why: &str) {
        if let (Some(m), Some(ex)) = (p.data.and_then(|w| w.member), self.export.as_mut()) {
            ex.member_dropped(m);
            ex.note("calls_not_exported", 1);
            ex.note(&format!("not_exported:posix_aio:{why}"), 1);
        }
    }

    /// The helper's call that serves a POSIX AIO request returned: a read or a write counted in
    /// the issuer's thread with the bytes it moved (an append at the file's end), an fsync as the
    /// issuer's call; one interrupted or refused its offset waits for the call glibc makes next.
    fn paio_served(&mut self, id: &str, ret: i64, done: Option<f64>, err: Option<&str>) -> Result<()> {
        if matches!(err, Some("EINTR" | "ESPIPE")) {
            return Ok(());
        }
        let Some(p) = self.paio.remove(id) else { return Ok(()) };
        self.paio_uncollected.insert(id.to_string());
        if let Some(w) = p.data {
            let n = ret.max(0);
            let off = if p.append { self.extent.get(&w.path).copied().unwrap_or(w.off) } else { w.off };
            self.data(w.kind, w.inst, w.tid, &w.path, w.file, off, w.nbytes, n)?;
            if w.kind == W && n > 0 && off >= 0 {
                if let Some(e) = self.extent.get_mut(&w.path) {
                    *e = (*e).max(off + n);
                }
            }
            if let (Some(m), Some(ex)) = (w.member, self.export.as_mut()) {
                ex.member_ret(m, if ret >= 0 { json!(ret) } else { json!(err.unwrap_or("EIO")) });
            }
        }
        if let Some((name, fd)) = p.sync {
            self.call(p.tid, name, &[fd], ret, None, p.issued, done, err.map(str::to_string))?;
        }
        Ok(())
    }

    /// Requests whose results the trace does not show, taken as complete.
    fn assume_complete(&mut self, ws: Vec<Waiting>) -> Result<()> {
        for w in ws {
            self.note("aio_results_assumed_complete", 1);
            self.data(w.kind, w.inst, w.tid, &w.path, w.file, w.off, w.nbytes, w.nbytes)?;
        }
        Ok(())
    }

    /// The path an `io_uring` open names, resolved as the converter printed its directory.
    fn uring_open_path(&self, u: &Uring) -> Option<String> {
        let p = c_string(u.args.get("path")?)?;
        if p.starts_with('/') {
            return Some(p);
        }
        let (_, dir) = fd_arg(u.args.get("dirfd")?);
        Some(format!("{}/{p}", dir?))
    }

    fn count(&mut self, tid: i64, kind: &str) {
        if let Some(m) = self.m() {
            m.op(kind);
        }
        self.end_chain(tid);
    }

    fn end_chain(&mut self, tid: i64) {
        if let Some((n, _)) = self.chain.remove(&tid) {
            if self.chain_gap.is_some() {
                if let Some(m) = self.metrics.as_mut() {
                    *m.m.depth.entry(n).or_insert(0) += 1;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn data(&mut self, kind: usize, inst: i64, tid: i64, p: &str, file: Option<u64>, off: i64, len: i64, n: i64) -> Result<()> {
        match self.metrics.as_mut() {
            Some(m) => m.data(kind, inst, tid, p, file, off, len, n),
            None => Ok(()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn call(&mut self, tid: i64, name: &str, a: &[String], ret: i64, ret_path: Option<String>, issued: Option<f64>, done: Option<f64>, err: Option<String>) -> Result<()> {
        let when: When = (tid, issued, done);
        let err = err.as_deref();
        if CLONES.contains(&name) {
            if ret > 0 {
                self.child(tid, ret, shares_files(name, &a.join(", ")));
            }
            return Ok(());
        }
        if name == "chdir" || name == "fchdir" {
            if ret == 0 && !a.is_empty() {
                let p = if name == "chdir" { c_string(&a[0]) } else { fd_arg(&a[0]).1 };
                if let Some(p) = p.filter(|p| !p.is_empty()) {
                    let r = self.resolve(tid, None, Some(p.clone())).unwrap_or(p);
                    self.cwds.insert(tid, r);
                }
            }
            return Ok(());
        }
        if name == "aeiou_uring_cqe" {
            return self.uring_cqe(tid, a, ret, ret_path, done, err);
        }
        if name == "aeiou_aio_issue" {
            let args = keyed(a);
            let (Some(ctx), Some(id), Some(m)) = (args.get("ctx"), args.get("id"), args.get("member").and_then(|m| m.parse::<usize>().ok())) else { return Ok(()) };
            // a request the line's thread did not count (not under a root, not a read or a write) is not waited for
            if let Some(w) = self.aio_members.get_mut(&tid).filter(|(c, _)| c == ctx).and_then(|(_, ws)| ws.remove(&m)) {
                if let Some(old) = self.aio_ids.insert((ctx.clone(), id.clone()), w) {
                    self.assume_complete(vec![old])?;
                }
            }
            return Ok(());
        }
        if name == "aeiou_aio_complete" {
            return self.aio_complete(a, ret, err);
        }
        if let Some(id) = self.serving.remove(&tid) {
            if matches!(name, "pread64" | "pwrite64" | "read" | "write" | "fsync" | "fdatasync") {
                return self.paio_served(&id, ret, done, err);
            }
        }
        match name {
            "aeiou_aio_serve" => {
                if let Some(id) = keyed(a).remove("id") {
                    self.serving.insert(tid, id);
                }
                return Ok(());
            }
            "aeiou_aio_return" => {
                if let Some(id) = keyed(a).get("id") {
                    self.paio_uncollected.remove(id);
                }
                return Ok(());
            }
            "aeiou_aio_refused" | "aeiou_aio_cancel" => {
                let k = keyed(a);
                let ids: Vec<String> = match (k.get("id"), k.get("canceled")) {
                    (Some(id), _) => vec![id.clone()],
                    (_, Some(l)) => split_args(l.trim_start_matches('[').trim_end_matches(']')),
                    _ => vec![],
                };
                for id in ids {
                    if let Some(p) = self.paio.remove(&id) {
                        let why = if name == "aeiou_aio_refused" { "refused" } else { "canceled" };
                        self.note(&format!("posix_aio_{why}"), 1);
                        self.paio_not_done(p, why);
                    }
                }
                return Ok(());
            }
            "aeiou_aio_suspend" => return Ok(()),
            _ => {}
        }
        let inst = self.instance(tid);
        if inst.is_none() {
            // the descriptor tables are still kept: a child inside an instance may inherit them
            self.note("calls_outside_instances", 1);
        }
        if name == "aeiou_uring_sqe" {
            self.uring_sqe(tid, inst, a, issued);
            return Ok(());
        }
        if name == "aeiou_fault" {
            self.fault(tid, inst, a, issued);
            return Ok(());
        }
        if matches!(name, "aeiou_aio_read" | "aeiou_aio_write" | "aeiou_aio_fsync" | "aeiou_lio_listio") {
            return self.paio_issue(tid, inst, name, a, issued);
        }
        if OPENS.contains(&name) {
            let pi = if name == "openat" || name == "openat2" { 1 } else { 0 };
            if a.len() <= pi {
                return Ok(());
            }
            let path = match ret_path.filter(|p| ret >= 0 && !p.is_empty()) {
                Some(p) => Some(p),
                None => {
                    let dir = if pi == 1 { Some(a[0].as_str()) } else { None };
                    self.resolve(tid, dir, c_string(&a[pi]))
                }
            };
            // `openat2` holds them in its `{flags=…, mode=…, resolve=…}`, `creat` has a mode only
            let (flags, mode, resolve) = match name {
                "openat2" => {
                    let how: Vec<String> = a.get(2).and_then(|h| h.strip_prefix('{')).and_then(|h| h.strip_suffix('}')).map(split_args).unwrap_or_default();
                    let field = |k: &str| how.iter().find_map(|f| f.strip_prefix(k).and_then(|v| v.strip_prefix('=')).map(str::to_string));
                    (field("flags").unwrap_or_default(), field("mode"), field("resolve").filter(|r| r != "0"))
                }
                "creat" => ("O_WRONLY|O_CREAT|O_TRUNC".to_string(), a.get(1).cloned(), None),
                _ => (a.get(pi + 1).cloned().unwrap_or_default(), a.get(pi + 2).cloned(), None),
            };
            let p = self.under(path.as_deref());
            let mut f = None;
            if ret >= 0 {
                if let Some(p) = &p {
                    if flags.contains("O_TRUNC") || (flags.contains("O_CREAT") && flags.contains("O_EXCL")) {
                        self.extent.insert(p.clone(), 0);
                    }
                    if flags.contains("O_APPEND") && !self.extent.contains_key(p) {
                        self.note("append_to_unknown_size", 1);
                    }
                }
                let file = new_file(path.clone(), 0, flags.contains("O_APPEND"));
                if let (Some(p), Some(m)) = (&p, self.metrics.as_mut()) {
                    // what a path holds (FS_API_SWEEP.md S-18): the open holds the file its name is
                    let held = m.m.file(p);
                    file.borrow_mut().file = Some(held);
                    if flags.contains("O_TRUNC") {
                        m.m.ended(held, 0, None);
                    }
                }
                self.table(tid).borrow_mut().insert(ret, file.clone());
                f = Some(file);
            }
            if let (Some(_), Some(p)) = (inst, &p) {
                self.count(tid, "open");
                if let Some(ex) = self.export.as_mut() {
                    if let Some(r) = &resolve {
                        // how the path was resolved, never what the open does to the file
                        ex.note(&format!("not_exported:openat2:resolve {r}"), 1);
                    }
                    let oid = ex.open(when, p, &flags, mode.as_deref(), err)?;
                    if let Some(f) = &f {
                        let mut fb = f.borrow_mut();
                        fb.oid = oid;
                        fb.unexported = oid.is_none();
                    }
                }
            }
            return Ok(());
        }
        if name == "close" {
            let (fd, mut path) = a.first().map(|x| fd_arg(x)).unwrap_or((None, None));
            let table = self.table(tid);
            let mut f = self.closing.remove(&tid);
            if f.is_none() {
                f = fd.and_then(|fd| table.borrow().get(&fd).cloned());
                if let (None, Some(fd), Some(p)) = (&f, fd, path.as_deref()) {
                    // a descriptor the trace never showed: held from before it began, its own
                    // and never another task's of the same path (a fork copies the table)
                    let held = new_file(Some(p.to_string()), 0, false);
                    held.borrow_mut().held = true;
                    table.borrow_mut().insert(fd, held.clone());
                    if self.under(Some(p)).is_some() {
                        self.note("descriptors_without_open", 1);
                    }
                    f = Some(held);
                }
                if let Some(ff) = &f {
                    if ret == 0 && (path.is_none() || ff.borrow().path == path) {
                        table.borrow_mut().remove(&fd.unwrap());
                    }
                }
            }
            if path.is_none() {
                path = f.as_ref().and_then(|f| f.borrow().path.clone());
            }
            let p = self.under(path.as_deref());
            if let (Some(_), Some(p)) = (inst, &p) {
                self.count(tid, "close");
                if let Some(m) = self.m() {
                    m.close(tid, p);
                }
                if let Some(oid) = self.oid_or_note(f.as_ref(), "close", when)? {
                    self.export.as_mut().unwrap().op(when, "close", oid, err, Value::Null, vec![]);
                }
            }
            return Ok(());
        }
        if DUPS.contains(&name) {
            if name == "fcntl" && !(a.len() > 1 && a[1].starts_with("F_DUPFD")) {
                // by its command, `F_SETFL` with the flags it set; none for those that change
                // nothing on the file
                let cmd = a.get(1).map(String::as_str).unwrap_or("");
                if !matches!(cmd, "F_GETFD" | "F_SETFD" | "F_GETFL" | "F_GETOWN" | "F_SETOWN" | "F_GETOWN_EX" | "F_SETOWN_EX" | "F_GETSIG" | "F_SETSIG") && inst.is_some() && self.names_a_root_path(tid, &a[..1]) {
                    let what = if cmd == "F_SETFL" { format!("F_SETFL {}", a.get(2).map(String::as_str).unwrap_or("")) } else { cmd.to_string() };
                    self.note(&format!("not_counted:fcntl:{what}"), 1);
                }
                return Ok(());
            }
            let (fd, path) = a.first().map(|x| fd_arg(x)).unwrap_or((None, None));
            let f = self.file(tid, fd, path.as_deref());
            if let (true, Some(f)) = (ret >= 0, f) {
                self.table(tid).borrow_mut().insert(ret, f);
            }
            return Ok(());
        }
        if name == "lseek" {
            let (fd, path) = a.first().map(|x| fd_arg(x)).unwrap_or((None, None));
            let f = self.file(tid, fd, path.as_deref());
            if let Some(f) = &f {
                if ret >= 0 {
                    let mut fb = f.borrow_mut();
                    fb.pos = ret;
                    fb.listed = false; // rewinddir
                }
            }
            let fp = path.clone().or_else(|| f.as_ref().and_then(|f| f.borrow().path.clone()));
            if inst.is_some() && self.under(fp.as_deref()).is_some() {
                self.count(tid, "lseek");
                if let Some(oid) = self.oid_or_note(f.as_ref(), "lseek", when)? {
                    let ex = self.export.as_mut().unwrap();
                    match (a.len() > 2).then(|| int0(&a[1])).flatten() {
                        Some(_) if matches!(a[2].as_str(), "SEEK_DATA" | "SEEK_HOLE") => {
                            // no whence of the trace file's yet (FS_API_SWEEP.md S-28, S-40): left
                            // out, and the open's later calls carry the offsets it moved them to
                            ex.note("calls_not_exported", 1);
                            ex.note(&format!("not_exported:lseek:{}", a[2]), 1);
                            ex.pin(oid);
                        }
                        Some(off) => {
                            let whence = a[2].replace("SEEK_", "");
                            ex.op(when, "lseek", oid, err, json!(ret), vec![("offset", json!(off)), ("whence", json!(whence))]);
                        }
                        None => {
                            ex.note("calls_not_exported", 1);
                            ex.note("not_exported:lseek:unparsed", 1);
                        }
                    }
                }
            }
            return Ok(());
        }
        if READS.contains(&name) || WRITES.contains(&name) {
            let kind = if READS.contains(&name) { R } else { W };
            let (fd, path) = a.first().map(|x| fd_arg(x)).unwrap_or((None, None));
            let f = self.file(tid, fd, path.as_deref());
            let path = path.or_else(|| f.as_ref().and_then(|f| f.borrow().path.clone()));
            let vector = name.ends_with('v') || name.ends_with("v2");
            let mut length;
            let off_tok = if name.starts_with('p') && a.len() > 3 { Some(a[3].as_str()) } else { None };
            if vector {
                length = a.get(1).map(|x| iov_lens(x)).unwrap_or(0);
                if a.len() > 1 && truncated(&a[1]) {
                    // the iovecs past the 32nd are not in the trace: a successful call's
                    // return value is a floor on the length the whole list asked for
                    length = length.max(ret);
                    if inst.is_some() && self.under(path.as_deref()).is_some() {
                        self.note("iov_arrays_truncated", 1);
                    }
                }
            } else {
                length = match a.get(2) {
                    Some(t) if digits_only(t.strip_prefix('-').unwrap_or(t)) => int0(t).unwrap_or(0),
                    _ => 0,
                };
            }
            let mut positioned = off_tok.is_some_and(|t| digits_only(t.strip_prefix('-').unwrap_or(t)) && t.parse::<i64>().is_ok_and(|v| v >= 0));
            let p = self.under(path.as_deref());
            // `pwritev2`'s `RWF_APPEND` appends as `O_APPEND` does, for this call only
            let rwf_append = name == "pwritev2" && a.get(4).is_some_and(|fl| fl.split('|').any(|x| x.trim() == "RWF_APPEND"));
            let append = kind == W && (rwf_append || f.as_ref().is_some_and(|f| f.borrow().append));
            let base = match (&p, &f) {
                (Some(p), Some(_)) if append => self.extent.get(p).copied(),
                _ => None,
            };
            if let (true, None, Some(_), Some(_)) = (rwf_append, base, &p, inst) {
                self.note("append_to_unknown_size", 1);
            }
            let offset;
            if let Some(b) = base {
                // an append: at the end of the file, a given offset ignored, and the position
                // moved to the end only by a call that gives none (a `pwrite` never moves it)
                offset = b;
                if ret > 0 && !positioned {
                    f.as_ref().unwrap().borrow_mut().pos = b + ret;
                }
                positioned = true;
            } else if positioned {
                offset = off_tok.unwrap().parse().unwrap();
            } else {
                offset = f.as_ref().map_or(-1, |f| f.borrow().pos);
                if let (Some(f), true) = (&f, ret > 0) {
                    f.borrow_mut().pos += ret;
                }
            }
            if kind == W && ret > 0 && offset >= 0 {
                if let Some(p) = &p {
                    if let Some(e) = self.extent.get_mut(p) {
                        *e = (*e).max(offset + ret);
                    }
                }
            }
            if let (Some(inst), Some(p)) = (inst, &p) {
                self.end_chain(tid);
                if offset < 0 {
                    self.note("data_ops_without_offset", 1);
                }
                let held = f.as_ref().and_then(|f| f.borrow().file);
                self.data(kind, inst, tid, p, held, offset, length, ret.max(0))?;
                if let (Some(oid), true) = (self.oid_or_note(f.as_ref(), KINDS[kind], when)?, offset >= 0) {
                    let ex = self.export.as_mut().unwrap();
                    if base.is_some() {
                        ex.pin(oid);
                        ex.note("append_positions_resolved", 1);
                    }
                    ex.data(when, KINDS[kind], oid, offset, length, positioned, ret, err);
                }
            }
            return Ok(());
        }
        if name == "mmap" {
            // the range of a file under a root is kept for the calls that name only an address
            if let (Some(at), Some(len), Some(tok)) = (self.ret_addr, a.get(1).and_then(|t| int0(t)).filter(|&l| l > 0), a.get(4)) {
                let (fd, path) = fd_arg(tok);
                let path = path.or_else(|| fd.and_then(|fd| self.table(tid).borrow().get(&fd).and_then(|f| f.borrow().path.clone())));
                if self.under(path.as_deref()).is_some() {
                    self.maps.insert(at, len as u64);
                }
            }
            // exported only (the metrics do not count a mapping): the mapped range as one read,
            // since the faults inside it are not in the trace
            if self.export.is_none() || a.len() < 6 || ret < 0 {
                return Ok(());
            }
            let (fd, path) = fd_arg(&a[4]);
            let f = self.file(tid, fd, path.as_deref());
            let Some(f) = f else { return Ok(()) };
            let fp = path.clone().or_else(|| f.borrow().path.clone());
            if inst.is_none() || self.under(fp.as_deref()).is_none() {
                return Ok(());
            }
            let Some(oid) = self.oid_or_note(Some(&f), "mmap", when)? else { return Ok(()) };
            let (Some(length), Some(off)) = (int0(&a[1]), int0(&a[5])) else { return Ok(()) };
            let ex = self.export.as_mut().unwrap();
            ex.note("mmap_as_read", 1);
            ex.data(when, "read", oid, off, length, true, length, None);
            return Ok(());
        }
        if name == "io_setup" {
            // `io_setup(nr, [ctx]) = 0`: a context no `io_submit` names by the end is noted
            if ret == 0 && inst.is_some() && a.len() > 1 {
                *self.aio_unused.entry(a[1].trim_matches(['[', ']']).to_string()).or_insert(0) += 1;
            }
            return Ok(());
        }
        if name == "io_submit" {
            if let Some(c) = a.first() {
                self.aio_unused.remove(c);
            }
            let cbs = iocbs(a.get(2).map(String::as_str).unwrap_or(""));
            if inst.is_some() && a.len() > 2 && truncated(&a[2]) && ret > cbs.len() as i64 {
                // the requests past the 32nd are not in the trace
                self.note("aio_requests_truncated", (ret - cbs.len() as i64) as u64);
            }
            // the kernel takes the first `ret` requests of the array and none past them
            let taken = ret.max(0) as usize;
            let mut got: Vec<(usize, String, usize, String, Option<u64>, i64, i64)> = Vec::new();
            for (k, cb) in cbs.iter().take(taken).enumerate() {
                let path = cb.path.clone().filter(|p| !p.is_empty());
                let f = self.file(tid, Some(cb.fd), path.as_deref());
                let fp = path.clone().or_else(|| f.as_ref().and_then(|f| f.borrow().path.clone()));
                let Some(p) = self.under(fp.as_deref()) else { continue };
                let Some(read) = cb.reads() else {
                    self.note("aio_requests_not_counted", 1);
                    continue;
                };
                let held = f.as_ref().and_then(|f| f.borrow().file);
                got.push((k, cb.data.clone(), if read { R } else { W }, p, held, cb.offset, cb.nbytes));
            }
            let Some(inst) = inst else { return Ok(()) };
            if got.is_empty() || ret <= 0 {
                return Ok(());
            }
            let mut c = self.chain.get(&tid).copied();
            if let (Some(cc), Some(gap), Some(iss)) = (c, self.chain_gap, issued) {
                if let Some(last) = cc.1 {
                    if iss - last > gap {
                        self.end_chain(tid);
                        c = None;
                    }
                }
            }
            let mut cc = c.unwrap_or((0, None));
            cc.0 += 1;
            cc.1 = done;
            self.chain.insert(tid, cc);
            if let Some(m) = self.metrics.as_mut() {
                *m.m.fan_out.entry(got.len() as u64).or_insert(0) += 1;
            }
            self.aio_submitted = true;
            let ctx = a[0].clone();
            let mut members: Option<Vec<(usize, usize)>> = None;
            if self.export.is_some() {
                let mut mem = Vec::new();
                for cb in cbs.iter().take(taken) {
                    let path = cb.path.clone().filter(|p| !p.is_empty());
                    let f = self.file(tid, Some(cb.fd), path.as_deref());
                    if let Some(f) = f {
                        let fp = path.clone().or_else(|| f.borrow().path.clone());
                        if let (Some(read), Some(_)) = (cb.reads(), self.under(fp.as_deref())) {
                            let kind = if read { "read" } else { "write" };
                            if let Some(oid) = self.oid_or_note(Some(&f), kind, when)? {
                                mem.push((kind, oid, cb.offset, cb.nbytes));
                            }
                        }
                    }
                }
                if !mem.is_empty() {
                    members = Some(self.export.as_mut().unwrap().submit(when, &mem));
                }
            }
            if self.aio_records {
                // counted at its completion (`aeiou_aio_complete`), once its `aeiou_aio_issue`
                // numbers it; a request of the thread's call before that none numbered was lost
                let mut take = HashMap::new();
                for (j, (k, _, kind, p, file, off, nbytes)) in got.into_iter().enumerate() {
                    let member = members.as_ref().and_then(|m| m.get(j).copied());
                    take.insert(k, Waiting { kind, inst, tid, path: p, file, off, nbytes, member });
                }
                if let Some((_, old)) = self.aio_members.insert(tid, (ctx, take)) {
                    self.assume_complete(old.into_values().collect())?;
                }
                return Ok(());
            }
            for (j, (_, data, kind, p, file, off, nbytes)) in got.into_iter().enumerate() {
                // counted when its result is reaped (`io_getevents`), with the bytes it moved
                let member = members.as_ref().and_then(|m| m.get(j).copied());
                let seq = &mut self.aio_seq;
                self.aio
                    .entry((ctx.clone(), data))
                    .or_insert_with(|| {
                        *seq += 1;
                        (*seq, Vec::new())
                    })
                    .1
                    .push(Waiting { kind, inst, tid, path: p, file, off, nbytes, member });
            }
            return Ok(());
        }
        if name == "io_getevents" || name == "io_pgetevents" {
            if let Some(c) = self.chain.get_mut(&tid) {
                c.1 = done;
            }
            let ctx = a.first().cloned();
            if self.aio_records {
                // the requests' results are their completions': collecting them counts nothing
                self.aio_reaped += ret.max(0) as u64;
                return Ok(());
            }
            let evs = events(a.get(3).map(String::as_str).unwrap_or(""));
            if a.len() > 3 && truncated(&a[3]) && ret > evs.len() as i64 && self.aio.keys().any(|k| Some(&k.0) == ctx.as_ref()) {
                // results the trace cut off: their requests stay waiting, taken as complete at the end
                self.note("aio_events_truncated", (ret - evs.len() as i64) as u64);
            }
            for (data, res) in evs {
                let key = (ctx.clone().unwrap_or_default(), data);
                let Some((_, waiting)) = self.aio.get_mut(&key) else { continue };
                if waiting.is_empty() {
                    continue;
                }
                let w = waiting.remove(0);
                if waiting.is_empty() {
                    self.aio.remove(&key);
                }
                self.data(w.kind, w.inst, w.tid, &w.path, w.file, w.off, w.nbytes, res.max(0))?;
                if w.kind == W && res > 0 {
                    if let Some(e) = self.extent.get_mut(&w.path) {
                        *e = (*e).max(w.off + res);
                    }
                }
                if let (Some(m), Some(ex)) = (w.member, self.export.as_mut()) {
                    ex.member_ret(m, if res >= 0 { json!(res) } else { json!(errno_name(-res)) });
                }
            }
            return Ok(());
        }
        if let Some(op) = fd_op_of(name) {
            let (fd, mut path) = a.first().map(|x| fd_arg(x)).unwrap_or((None, None));
            let f = self.file(tid, fd, path.as_deref());
            if path.is_none() {
                path = f.as_ref().and_then(|f| f.borrow().path.clone());
            }
            let p = self.under(path.as_deref());
            if let (0, Some(p), Some(m), "ftruncate" | "fallocate") = (ret, &p, self.metrics.as_mut(), name) {
                // what the file holds (FS_API_SWEEP.md S-18), the open's file
                let held = f.as_ref().and_then(|f| f.borrow().file).unwrap_or_else(|| m.m.file(p));
                let int = |i: usize| a.get(i).and_then(|x| int0(x));
                // a length the trace does not show ends nothing
                if name == "ftruncate" {
                    if let Some(n) = int(1) {
                        m.m.ended(held, n.max(0) as u64, None);
                    }
                } else if let (Some(mode), Some(o), Some(l)) = (a.get(1).and_then(|x| crate::export::falloc_bits(x)), int(2), int(3)) {
                    m.m.fallocated(held, o, l, mode);
                }
            }
            if let (0, Some(p)) = (ret, &p) {
                if self.extent.contains_key(p) {
                    let v = match name {
                        "ftruncate" => Some(a.get(1).and_then(|x| int0(x))),
                        // the size a mode leaves: a hole punched, a range kept, leaves it; a range
                        // collapsed or inserted moves it by the length
                        "fallocate" if a.get(1).is_some_and(|m| m.contains("COLLAPSE_RANGE") || m.contains("INSERT_RANGE")) => Some(None),
                        "fallocate" if a.get(1).is_some_and(|m| m == "0" || (m == "FALLOC_FL_ZERO_RANGE")) => {
                            Some(match (a.get(2).and_then(|x| int0(x)), a.get(3).and_then(|x| int0(x))) {
                                (Some(o), Some(l)) => Some(self.extent[p].max(o + l)),
                                _ => None,
                            })
                        }
                        _ => None,
                    };
                    match v {
                        Some(Some(n)) => {
                            self.extent.insert(p.clone(), n);
                        }
                        Some(None) => {
                            self.extent.remove(p);
                        }
                        None => {}
                    }
                }
            }
            if op == "readdir" {
                if let Some(f) = &f {
                    // the abstract's `readdir` is one listing of a directory: the calls that
                    // continue it, and the empty one that ends it, are the same op
                    if f.borrow().listed {
                        return Ok(());
                    }
                    f.borrow_mut().listed = true;
                }
            }
            if let (Some(_), Some(_)) = (inst, &p) {
                self.count(tid, op);
                if let Some(oid) = self.oid_or_note(f.as_ref(), op, when)? {
                    self.export.as_mut().unwrap().fd_op(when, op, name, oid, a, ret, err);
                }
            }
            return Ok(());
        }
        if let Some((mut kind, di, pi)) = path_op_of(name) {
            if a.len() <= pi {
                return Ok(());
            }
            let arg = c_string(&a[pi]);
            if (name == "statx" || name == "newfstatat") && arg.as_deref() == Some("") {
                // AT_EMPTY_PATH: the descriptor itself
                let (fd, mut path) = fd_arg(&a[0]);
                if path.is_none() {
                    path = self.file(tid, fd, None).and_then(|f| f.borrow().path.clone());
                }
                let f = self.file(tid, fd, path.as_deref());
                if inst.is_some() && self.under(path.as_deref()).is_some() {
                    self.count(tid, "fstat");
                    if let Some(oid) = self.oid_or_note(f.as_ref(), "fstat", when)? {
                        self.export.as_mut().unwrap().op(when, "fstat", oid, err, Value::Null, vec![]);
                    }
                }
                return Ok(());
            }
            let dir = di.map(|d| a[d].clone());
            let path = self.resolve(tid, dir.as_deref(), arg);
            if name == "unlinkat" && a.last().is_some_and(|x| x.contains("AT_REMOVEDIR")) {
                kind = "rmdir";
            }
            let p = self.under(path.as_deref());
            // renameat2's flags (exchange, no-replace, whiteout): no rename of the trace file's
            // yet (FS_API_SWEEP.md S-27, S-40)
            let flags = (name == "renameat2").then(|| a.get(4).map(|f| f.trim()).filter(|f| *f != "0")).flatten();
            if let (0, Some(p)) = (ret, &p) {
                self.known_size(tid, name, kind, p, a, di, pi);
            }
            if ret == 0 && self.metrics.is_some() {
                // what a path holds (FS_API_SWEEP.md S-18), a rename's either name may be outside the root
                if kind == "rename" {
                    let to = self.rename_target(tid, a, di, pi);
                    let exchange = name == "renameat2" && a.get(4).is_some_and(|f| f.contains("RENAME_EXCHANGE"));
                    self.metrics.as_mut().unwrap().m.renamed(p.as_deref(), to.as_deref(), exchange);
                } else if let Some(p) = &p {
                    let m = &mut self.metrics.as_mut().unwrap().m;
                    match kind {
                        "unlink" => m.unlinked(p),
                        "truncate" => {
                            if let Some(n) = a.get(pi + 1).and_then(|x| int0(x)) {
                                m.ended_name(p, n.max(0) as u64, None);
                            }
                        }
                        _ => {}
                    }
                }
            }
            if p.is_none() && kind == "rename" && inst.is_some() && self.export.is_some() {
                // from outside the root into it: the name appears with nothing the runner can make
                // it from (FS_API_SWEEP.md S-25)
                if self.rename_target(tid, a, di, pi).is_some() {
                    let ex = self.export.as_mut().unwrap();
                    ex.note("calls_not_exported", 1);
                    ex.note("not_exported:rename:source outside the root", 1);
                }
            }
            if let (Some(_), Some(p)) = (inst, &p) {
                self.count(tid, kind);
                if let (Some(f), Some(ex)) = (flags, self.export.as_mut()) {
                    ex.note("calls_not_exported", 1);
                    ex.note(&format!("not_exported:renameat2:{f}"), 1);
                } else if self.export.is_some() {
                    let to = if kind == "rename" { self.rename_target(tid, a, di, pi) } else { None };
                    self.export.as_mut().unwrap().path_op(when, kind, name, p, a, pi, to.as_deref(), err)?;
                }
            }
            return Ok(());
        }
        Ok(())
    }

    /// A rename's destination, under the root.
    fn rename_target(&mut self, tid: i64, a: &[String], di: Option<usize>, pi: usize) -> Option<String> {
        let j = if di.is_none() { pi + 1 } else { pi + 2 };
        if a.len() <= j {
            return None;
        }
        let d = if di.is_some() { Some(a[j - 1].clone()) } else { None };
        let r = self.resolve(tid, d.as_deref(), c_string(&a[j]));
        self.under(r.as_deref())
    }

    /// What a successful path call does to the sizes the trace knows (`extent`).
    #[allow(clippy::too_many_arguments)]
    fn known_size(&mut self, tid: i64, name: &str, kind: &str, p: &str, a: &[String], di: Option<usize>, pi: usize) {
        if kind == "unlink" {
            self.extent.remove(p);
        } else if name == "truncate" {
            match a.get(pi + 1).and_then(|x| int0(x)) {
                Some(n) => {
                    self.extent.insert(p.to_string(), n);
                }
                None => {
                    self.extent.remove(p);
                }
            }
        } else if kind == "rename" {
            let to = self.rename_target(tid, a, di, pi);
            if let (true, Some(to)) = (name == "renameat2" && a.get(4).is_some_and(|f| f.contains("RENAME_EXCHANGE")), &to) {
                // the two names trade files
                let (x, y) = (self.extent.remove(p), self.extent.remove(to));
                if let Some(y) = y {
                    self.extent.insert(p.to_string(), y);
                }
                if let Some(x) = x {
                    self.extent.insert(to.clone(), x);
                }
                return;
            }
            let size = self.extent.remove(p);
            if let Some(to) = to {
                match size {
                    None => {
                        self.extent.remove(&to);
                    }
                    Some(s) => {
                        self.extent.insert(to, s);
                    }
                }
            }
        }
    }

    pub fn finish(&mut self) -> Result<()> {
        let mut tids: Vec<i64> = self.chain.keys().copied().collect();
        tids.sort_unstable();
        for t in tids {
            self.end_chain(t);
        }
        let mut tids: Vec<i64> = self.uring_batch.keys().copied().collect();
        tids.sort_unstable();
        for t in tids {
            self.flush_uring(t);
        }
        // `io_uring` requests whose completion the trace does not show (the capture ended first,
        // or its record was lost): taken as complete, as AIO's are
        let mut left: Vec<((String, String), Uring)> = self.uring.drain().collect();
        left.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, u) in left {
            if let Some(w) = u.data {
                self.note("uring_results_assumed_complete", 1);
                self.data(w.kind, w.inst, w.tid, &w.path, w.file, w.off, w.nbytes, w.nbytes)?;
            }
        }
        // requests whose results the trace does not show (reaped from the ring in user space,
        // or `io_getevents` not traced): taken as complete
        let mut keys: Vec<(u64, (String, String))> = self.aio.iter().map(|(k, v)| (v.0, k.clone())).collect();
        keys.sort();
        for (_, k) in keys {
            let ws = self.aio.remove(&k).unwrap().1;
            self.assume_complete(ws)?;
        }
        // the same of a trace with completions: their records lost, or the capture ended first
        let mut left: Vec<((String, String), Waiting)> = self.aio_ids.drain().collect();
        left.sort_by(|a, b| a.0.cmp(&b.0));
        let mut members: Vec<(i64, (String, HashMap<usize, Waiting>))> = self.aio_members.drain().collect();
        members.sort_by_key(|m| m.0);
        for (_, (_, ws)) in members {
            let mut ws: Vec<(usize, Waiting)> = ws.into_iter().collect();
            ws.sort_by_key(|w| w.0);
            left.extend(ws.into_iter().map(|(_, w)| (Default::default(), w)));
        }
        self.assume_complete(left.into_iter().map(|(_, w)| w).collect())?;
        // POSIX AIO's requests no helper's call served (the capture ended first, or the record
        // was lost) are not counted: the call that would have moved the bytes is not in the trace
        if !self.paio.is_empty() {
            let mut left: Vec<(String, Paio)> = self.paio.drain().collect();
            left.sort_by(|a, b| a.0.cmp(&b.0));
            self.note("posix_aio_never_served", left.len() as u64);
            for (_, p) in left {
                self.paio_not_done(p, "never served");
            }
        }
        if !self.paio_uncollected.is_empty() {
            let n = self.paio_uncollected.drain().count() as u64;
            self.note("posix_aio_never_collected", n);
        }
        if self.aio_completed > self.aio_reaped {
            // a result read from the context's ring in user space, with no call
            self.note("aio_completed_never_reaped", self.aio_completed - self.aio_reaped);
        }
        // contexts made and never submitted to: unused, or `io_submit` left out of the trace's
        // `-e trace=` (strace classes no AIO call in `%desc`), and every request lost with it
        let unused: u64 = self.aio_unused.values().sum();
        if unused > 0 {
            self.note("aio_contexts_without_io_submit", unused);
        }
        self.aio_unused.clear();
        if let Some(m) = self.metrics.as_mut() {
            m.finish()?;
        }
        Ok(())
    }
}

/// strace's name for an errno a result carries.
pub fn errno_name(code: i64) -> String {
    match code {
        2 => "ENOENT",
        5 => "EIO",
        9 => "EBADF",
        13 => "EACCES",
        17 => "EEXIST",
        20 => "ENOTDIR",
        21 => "EISDIR",
        22 => "EINVAL",
        28 => "ENOSPC",
        39 => "ENOTEMPTY",
        _ => return format!("E{code}"),
    }
    .to_string()
}

/// Every line of `input` as UTF-8, an undecodable byte replaced (strace escapes what it prints
/// of buffers; a path in a foreign encoding is still a path to count).
pub fn for_each_line(mut input: impl BufRead, mut f: impl FnMut(&str) -> Result<()>) -> Result<()> {
    let mut buf = Vec::new();
    loop {
        buf.clear();
        if input.read_until(b'\n', &mut buf)? == 0 {
            return Ok(());
        }
        let line = String::from_utf8_lossy(&buf);
        f(&line)?;
    }
}

/// The `aeiou_metrics: 2` document of an strace, and the lines it read.
pub fn metrics(input: impl BufRead, o: &Opts) -> Result<(Value, Seen)> {
    let mut t = Tracer::new(o, true, None);
    t.feed(input)?;
    t.finish()?;
    Ok((metrics_document(&t, o), t.seen))
}

/// The `aeiou_metrics: 2` document of a finished tracer made with its metrics on.
pub fn metrics_document(t: &Tracer, o: &Opts) -> Value {
    let m = t.metrics.as_ref().unwrap();
    let mut notes: serde_json::Map<String, Value> = t.notes.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    if t.chain_gap.is_none() && t.aio_submitted {
        notes.insert("depth_not_computed".into(), json!("give --chain-gap-us: the think time that ends a chain of io_submit rounds"));
    }
    if notes.contains_key(EVENTS_LOST) {
        notes.insert("trace_incomplete".into(), json!("the eBPF capture lost events when its ring buffer was full: a call whose open fell in a gap has no path, and every number here is short; capture again with a larger --buffer, or --source strace"));
    }
    if notes.contains_key(ARGUMENTS_PRINTED) || notes.contains_key(DATA_PRINTED) {
        notes.insert("trace_may_hold_secrets".into(), json!(SECRETS));
    }
    if ["aio_requests_truncated", "aio_events_truncated", "iov_arrays_truncated"].iter().any(|k| notes.contains_key(*k)) {
        notes.insert("arrays_truncated".into(), json!("trace again with strace -v: without it strace prints only the first 32 elements of an array"));
    }
    if m.faults.is_some() {
        notes.insert(
            "faults_kernel_observed".into(),
            json!("faults counts the kernel's faults on mappings of files under the roots, apart from the calls: a page faults at its first touch only, a fault maps the whole folio its page is in (up to 2 MiB, as the file happened to be cached), and where the capture's fault-around check was waived, or the host's fault-around changed during the capture (faults_first_touch_unseen), the cached pages beside it too; where touches began, at the granularity the kernel mapped: evidence of a mapped site's touches, weaker than a call"),
        );
    }
    let instances = m.instances.len().max(usize::from(o.instance_roots.is_empty()));
    let mut doc = json!({
        "aeiou_metrics": METRICS_FORMAT,
        "source": "strace",
        "roots": t.roots,
        "block": m.m.opts_block,
        "order": "completion",
        "instances": instances,
        "contexts": t.tids.len(),
        "chain_gap_us": t.chain_gap.map(|g| g * 1e6),
        "notes": notes,
        "total": m.total(t.chain_gap.is_some()),
    });
    if let Some(f) = &m.faults {
        doc["faults"] = serde_json::to_value(f.out()).unwrap();
    }
    doc
}

/// Why a trace gave no call to count: the roots matched none of its calls, or it is no strace.
pub fn nothing_counted(roots: &[String], seen: Seen) -> String {
    let why = if seen.calls > 0 {
        "check --root against the paths the trace shows"
    } else {
        "not an strace (strace -f --seccomp-bpf -ttt -T -yy -v -e 'verbose=!execve,execveat' -s 0 -e trace=%file,%desc,%process,... -o FILE)"
    };
    format!("no call under --root {} ({} line(s) read, {} parsed as strace calls): {why}", roots.join(", "), seen.lines, seen.calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_split_at_top_level_commas_only() {
        assert_eq!(split_args(r#"3</a,b>, "x,y", [1, 2], {a=1, b=2}, 7"#), vec!["3</a,b>", r#""x,y""#, "[1, 2]", "{a=1, b=2}", "7"]);
        assert_eq!(split_args(""), Vec::<String>::new());
        assert_eq!(split_args(r#""a\"b", 2"#), vec![r#""a\"b""#, "2"]);
        assert_eq!(split_args("AT_FDCWD</d>, x"), vec!["AT_FDCWD</d>", "x"]);
        // strace pads a short call's return out to column 40
        assert_eq!(ret_split(r#"rmdir("sub")  = 0"#), Some((11, 16)));
        assert_eq!(ret_split(r#"close(3</a>) = 0 <0.1>"#), Some((11, 15)));
        assert_eq!(ret_split(r#"write(1, ") = x", 5) = 5"#), Some((19, 23)));
        assert_eq!(ret_split("exit_group(0) = ?"), Some((12, 16)));
        assert_eq!(ret_split("x = 1"), None);
        assert_eq!(ret_split("read(3, "), None);
        let mut t = Tracer::new(&Opts { roots: vec!["/d".into()], ..Opts::default() }, true, None);
        t.line("0 1.000000 --- aeiou: 77 event(s) lost on cpu 2 from 1.000000 to 2.000000: the ring buffer was full ---\n");
        t.line("0 1.000000 --- SIGCHLD {si_signo=SIGCHLD} ---\n");
        assert_eq!(t.notes.get(EVENTS_LOST), Some(&77));
        let doc = metrics_document(&t, &Opts::default());
        assert!(doc["notes"]["trace_incomplete"].as_str().unwrap().contains("--buffer"));
    }

    #[test]
    fn the_header_and_a_record_of_a_kind_not_known_are_skipped() {
        // DESIGN_REVIEW.md §3.98: a reader falls through a name it does not know, counted by name
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let mut t = Tracer::new(&o, true, None);
        t.line("0 1.000000 --- aeiou: trace format 1, records aeiou_uring_sqe aeiou_uring_cqe ---\n");
        t.line("1 1.000001 openat(AT_FDCWD</d>, \"/d/a\", O_RDONLY) = 3</d/a> <0.000010>\n");
        t.line("1 1.000002 aeiou_later_kind(fd=3</d/a>, off=0, len=4096, path=\"/d/a\") = 4096\n");
        t.line("1 1.000003 pread64(3</d/a>, \"\"..., 4096, 0) = 4096 <0.000010>\n");
        assert!(!t.notes.contains_key(EVENTS_LOST), "{:?}", t.notes);
        assert_eq!(t.notes.get("not_counted:aeiou_later_kind"), Some(&1), "{:?}", t.notes);
        assert_eq!(t.format, 1);
        let doc = metrics_document(&t, &o);
        assert!(doc["notes"].get("trace_incomplete").is_none(), "{}", doc["notes"]);
        assert_eq!(doc["total"]["bytes_read"], 4096, "{}", doc["total"]);
        // a newer format than this reader's is read as far as it can be, and said so
        let mut t = Tracer::new(&o, true, None);
        t.line(&format!("0 1.000000 --- aeiou: trace format {}, records aeiou_x ---\n", TRACE_FORMAT + 1));
        assert_eq!(t.notes.get("trace_format_newer"), Some(&(TRACE_FORMAT + 1)));
    }

    #[test]
    fn faults_are_counted_apart_from_the_calls_and_not_replayed() {
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let lines = [
            "0 1700000001.000000 --- aeiou: trace format 1, records aeiou_fault ---",
            "1 1700000001.000001 openat(AT_FDCWD</d>, \"/d/a\", O_RDONLY) = 3</d/a> <0.000010>",
            "1 1700000001.000020 mmap(NULL, 1048576, PROT_READ, MAP_SHARED, 3</d/a>, 0) = 0x7f0000000000 <0.000005>",
            r#"1 1700000001.000100 aeiou_fault(path="/d/a", off=0, stride=4096, count=64, major=64, write=0, in_call=0, page=4096, span_ns=900000, gap_max_ns=20000, addr=0x7f0000000000) = 0 <0.000800>"#,
            // the same thread on the same file 5 ms after the first run ended
            r#"1 1700000001.005900 aeiou_fault(path="/d/a", off=262144, stride=65536, count=4, major=0, write=1, in_call=0, page=4096, span_ns=10000, gap_max_ns=2000, addr=0x7f0000040000) = 0 <0.000004>"#,
            r#"2 1700000001.006000 aeiou_fault(path="/d/a", off=8192, stride=0, count=1, major=0, write=0, in_call=1, page=4096, span_ns=1000, gap_max_ns=0, addr=0x7f0000002000) = 0 <0.000001>"#,
            // a library's: not under the root
            r#"1 1700000001.007000 aeiou_fault(path="/usr/lib/libc.so.6", off=0, stride=0, count=1, major=0, write=0, in_call=0, page=4096, span_ns=1000, gap_max_ns=0, addr=0x7f1000000000) = 0 <0.000001>"#,
        ];
        let mut t = Tracer::new(&o, true, None);
        for l in lines {
            t.line(l);
        }
        t.finish().unwrap();
        let doc = metrics_document(&t, &o);
        let f = &doc["faults"];
        assert_eq!((f["faults"].as_u64(), f["major"].as_u64(), f["write"].as_u64(), f["in_call"].as_u64(), f["runs"].as_u64()), (Some(69), Some(64), Some(4), Some(1), Some(3)), "{f}");
        assert_eq!(f["bytes"], 69 * 4096);
        assert_eq!(f["stride"], json!({"4096": 63, "65536": 3}));
        // a run's span: 63 strides and a page; 3 strides of 64 KiB and a page; a page
        assert_eq!(f["run_bytes"]["max"], 64 * 4096);
        assert_eq!(f["run_faults"]["n"], 3);
        // the quiet time between thread 1's two runs on the file: 5.9 ms less the first's 0.9 ms
        assert_eq!((f["quiet_us"]["n"].as_u64(), f["quiet_us"]["max"].as_u64()), (Some(1), Some(4900)), "{f}");
        assert!(doc["notes"]["faults_kernel_observed"].as_str().unwrap().contains("weaker than a call"));
        assert!(doc["notes"].get(FIRST_TOUCH_UNSEEN).is_none());
        // the calls' numbers are the calls' alone: a mapping is not a read to the metrics
        assert!(doc["total"]["counts"].get("read").is_none(), "{}", doc["total"]["counts"]);

        // a capture whose fault-around check was waived says so
        let mut t = Tracer::new(&o, true, None);
        t.line("0 1700000001.000000 --- aeiou: trace format 1, records aeiou_fault ---");
        t.line("0 1700000001.000000 --- aeiou: faults: a first touch of a page may not fault (the capture's fault-around check was waived) ---");
        t.line(lines[3]);
        t.finish().unwrap();
        assert_eq!(metrics_document(&t, &o)["notes"][FIRST_TOUCH_UNSEEN], 1);
        // and says nothing of it where no fault under the roots was counted
        let mut t = Tracer::new(&o, true, None);
        t.line("0 1700000001.000000 --- aeiou: faults: a first touch of a page may not fault (the capture's fault-around check was waived) ---");
        t.line(lines[1]);
        t.line(lines[6]);
        t.finish().unwrap();
        assert!(metrics_document(&t, &o)["notes"].get(FIRST_TOUCH_UNSEEN).is_none());

        // without faults, no section
        let mut t = Tracer::new(&o, true, None);
        t.line(lines[1]);
        t.finish().unwrap();
        assert!(metrics_document(&t, &o).get("faults").is_none());

        // the export keeps the mapping as one read of its range and notes the faults it does not replay
        let mut t = Tracer::new(&o, false, Some(crate::export::Exporter::new("/d")));
        for l in &lines[..6] {
            t.line(l);
        }
        let ex = t.export.as_ref().unwrap();
        assert_eq!(ex.notes.get("faults_not_replayed"), Some(&json!(69)), "{:?}", ex.notes);
        assert_eq!(ex.notes.get("mmap_as_read"), Some(&json!(1)), "{:?}", ex.notes);
    }

    #[test]
    fn io_uring_requests_count_as_their_calls_and_one_enter_s_are_a_fan_out() {
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let mut t = Tracer::new(&o, true, None);
        for l in [
            "0 1.000000 --- aeiou: trace format 1, records aeiou_uring_sqe aeiou_uring_cqe ---",
            "1 1.000001 io_uring_setup(64, 0x7ffe) = 5 <0.000010>",
            // an open through the ring: the descriptor it returns names its path
            "1 1.000002 io_uring_enter(5, 1, 1, IORING_ENTER_GETEVENTS, NULL, 8 <unfinished ...>",
            r#"1 1.000003 aeiou_uring_sqe(ring=5, id=1, op=IORING_OP_OPENAT, dirfd=AT_FDCWD</d>, path="a", flags=O_RDWR|O_CREAT, mode=0644, user_data=0x1) = 0"#,
            "1 1.000004 aeiou_uring_cqe(ring=5, id=1, user_data=0x1, flags=0) = 7</d/a>",
            "1 1.000005 <... io_uring_enter resumed>) = 1 <0.000003>",
            // two reads and a write in one enter: a fan-out of 3
            "1 1.000010 io_uring_enter(5, 3, 0, 0, NULL, 8 <unfinished ...>",
            "1 1.000011 aeiou_uring_sqe(ring=5, id=1, op=IORING_OP_READ, fd=7</d/a>, off=0, len=4096, user_data=0x0) = 0",
            "1 1.000012 aeiou_uring_sqe(ring=5, id=2, op=IORING_OP_READ_FIXED, fd=7</d/a>, off=8192, len=4096, user_data=0x0) = 0",
            "1 1.000013 aeiou_uring_sqe(ring=5, id=3, op=IORING_OP_WRITE, fd=7</d/a>, fixed=0</d/a>, off=65536, len=100, user_data=0x0) = 0",
            "1 1.000014 <... io_uring_enter resumed>) = 3 <0.000004>",
            "1 1.000020 aeiou_uring_cqe(ring=5, id=2, user_data=0x0, flags=0) = 4096",
            "1 1.000021 aeiou_uring_cqe(ring=5, id=1, user_data=0x0, flags=0) = 1000",
            "1 1.000022 aeiou_uring_cqe(ring=5, id=3, user_data=0x0, flags=0) = 100",
            // a vectored read moves what its result says; an fdatasync; a direct open not counted
            "1 1.000030 aeiou_uring_sqe(ring=5, id=4, op=IORING_OP_READV, fd=7</d/a>, off=0, iovcnt=2, user_data=0x0) = 0",
            "1 1.000031 aeiou_uring_sqe(ring=5, id=5, op=IORING_OP_FSYNC, fd=7</d/a>, fsync_flags=IORING_FSYNC_DATASYNC, user_data=0x0) = 0",
            r#"1 1.000032 aeiou_uring_sqe(ring=5, id=6, op=IORING_OP_OPENAT, dirfd=AT_FDCWD</d>, path="b", flags=O_RDONLY, file_slot=1, user_data=0x0) = 0"#,
            "1 1.000033 aeiou_uring_sqe(ring=5, id=7, op=IORING_OP_FADVISE, fd=7</d/a>, user_data=0x0) = 0",
            "1 1.000040 aeiou_uring_cqe(ring=5, id=4, user_data=0x0, flags=0) = 50",
            "1 1.000041 aeiou_uring_cqe(ring=5, id=5, user_data=0x0, flags=0) = 0",
            "1 1.000042 aeiou_uring_cqe(ring=5, id=6, user_data=0x0, flags=0) = 0",
            "1 1.000043 aeiou_uring_cqe(ring=5, id=7, user_data=0x0, flags=0) = 0",
            // a failed read moves nothing; a completion of no submission the trace shows
            "1 1.000050 aeiou_uring_sqe(ring=5, id=8, op=IORING_OP_READ, fd=7</d/a>, off=0, len=10, user_data=0x0) = 0",
            "1 1.000051 aeiou_uring_cqe(ring=5, id=8, user_data=0x0, flags=0) = -1 EIO (Input/output error)",
            "1 1.000052 aeiou_uring_cqe(ring=5, id=9, user_data=0x0, flags=0) = 10",
            "1 1.000060 aeiou_uring_sqe(ring=5, id=10, op=IORING_OP_CLOSE, fd=7</d/a>, user_data=0x0) = 0",
            "1 1.000061 aeiou_uring_cqe(ring=5, id=10, user_data=0x0, flags=0) = 0",
            // a request whose result the trace does not show: taken as complete
            "1 1.000070 aeiou_uring_sqe(ring=5, id=11, op=IORING_OP_WRITE, fixed=0</d/a>, off=0, len=7, user_data=0x0) = 0",
        ] {
            t.line(&format!("{l}\n"));
        }
        t.finish().unwrap();
        let doc = metrics_document(&t, &o);
        let total = &doc["total"];
        assert_eq!(total["bytes_read"], 1000 + 4096 + 50, "{total}");
        assert_eq!(total["bytes_written"], 100 + 7, "{total}");
        assert_eq!(total["counts"]["open"], 1, "{total}");
        assert_eq!(total["counts"]["fdatasync"], 1, "{total}");
        assert_eq!(total["counts"]["close"], 1, "{total}");
        assert_eq!(total["fan_out"]["3"], 1, "{total}");
        let notes = &doc["notes"];
        assert_eq!(notes["uring_direct_opens_not_counted"], 1, "{notes}");
        assert_eq!(notes["not_counted:aeiou_uring_sqe:IORING_OP_FADVISE"], 1, "{notes}");
        assert_eq!(notes["uring_completions_without_submission"], 1, "{notes}");
        assert_eq!(notes["uring_results_assumed_complete"], 1, "{notes}");
        assert!(notes.get("not_counted:aeiou_uring_sqe").is_none() && notes.get("not_counted:aeiou_uring_cqe").is_none(), "{notes}");
        // the ring's requests are in this trace: its setup is no longer a sign they are missing
        assert!(notes.get("not_counted:io_uring_setup").is_none(), "{notes}");
    }

    /// libaio's completions in an eBPF capture (DESIGN_REVIEW.md §3.98 "Phase 4 step 3"): a
    /// request counted at its `aeiou_aio_complete` with the bytes it moved, paired by the number
    /// its `aeiou_aio_issue` gave it (not by `aio_data`, which the second call reuses and whose
    /// completions come out of order), `io_getevents` collecting only; a completion inside its
    /// call noted, the results never collected by a call noted, a request with no completion
    /// taken as complete; the export's members get their completions' results.
    #[test]
    fn libaio_requests_count_at_their_completions() {
        let lines = [
            "0 1.000000 --- aeiou: trace format 1, records aeiou_aio_issue aeiou_aio_complete ---",
            "1 1.000001 openat(AT_FDCWD</d>, \"/d/a\", O_RDWR) = 7</d/a> <0.000010>",
            "1 1.000002 io_setup(8, [0xc]) = 0 <0.000001>",
            "1 1.000003 io_submit(0xc, 3, [{aio_data=0x0, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=7</d/a>, aio_buf=0x1, aio_nbytes=4096, aio_offset=0}, {aio_data=0x0, aio_lio_opcode=IOCB_CMD_FSYNC, aio_fildes=7</d/a>}, {aio_data=0x0, aio_lio_opcode=IOCB_CMD_PWRITE, aio_fildes=7</d/a>, aio_buf=0x1, aio_nbytes=100, aio_offset=8192}]) = 3 <0.000004>",
            "1 1.000003 aeiou_aio_issue(ctx=0xc, id=1, member=0) = 0",
            "1 1.000003 aeiou_aio_issue(ctx=0xc, id=2, member=1) = 0",
            "1 1.000003 aeiou_aio_issue(ctx=0xc, id=3, member=2) = 0",
            "1 1.000004 aeiou_aio_complete(ctx=0xc, id=1, data=0x0, in_submit=1) = 1000",
            "1 1.000010 aeiou_aio_complete(ctx=0xc, id=3, data=0x0, in_submit=0) = 100",
            "1 1.000011 aeiou_aio_complete(ctx=0xc, id=2, data=0x0, in_submit=0) = 0",
            "1 1.000012 io_getevents(0xc, 1, 8, [{data=0x0, obj=0x10, res=100, res2=0}], NULL) = 1 <0.000001>",
            "1 1.000020 io_submit(0xc, 2, [{aio_data=0x0, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=7</d/a>, aio_buf=0x1, aio_nbytes=4096, aio_offset=0}, {aio_data=0x0, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=7</d/a>, aio_buf=0x1, aio_nbytes=4096, aio_offset=4096}]) = 2 <0.000004>",
            "1 1.000020 aeiou_aio_issue(ctx=0xc, id=1, member=0) = 0",
            "1 1.000020 aeiou_aio_issue(ctx=0xc, id=2, member=1) = 0",
            "1 1.000030 aeiou_aio_complete(ctx=0xc, id=2, data=0x0, in_submit=0) = 4096",
            "1 1.000031 aeiou_aio_complete(ctx=0xc, id=1, data=0x0, in_submit=0) = -1 EIO (Input/output error)",
            "1 1.000032 io_getevents(0xc, 2, 8, [{data=0x0, obj=0x20, res=4096, res2=0}, {data=0x0, obj=0x18, res=-5, res2=0}], NULL) = 2 <0.000001>",
            // a request whose completion the trace does not show
            "1 1.000040 io_submit(0xc, 1, [{aio_data=0x0, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=7</d/a>, aio_buf=0x1, aio_nbytes=10, aio_offset=0}]) = 1 <0.000004>",
            "1 1.000040 aeiou_aio_issue(ctx=0xc, id=4, member=0) = 0",
            // a completion of no request the trace shows
            "1 1.000050 aeiou_aio_complete(ctx=0xc, id=9, data=0x0, in_submit=0) = 10",
        ]
        .join("\n");
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let (doc, _) = metrics(lines.as_bytes(), &o).unwrap();
        let total = &doc["total"];
        // the short read's 1000, the second call's 4096 at 4096 (not the failed one at 0), the unanswered 10
        assert_eq!((total["bytes_read"].as_u64(), total["bytes_written"].as_u64()), (Some(1000 + 4096 + 10), Some(100)), "{total}");
        assert_eq!(total["fan_out"], json!({"1": 1, "2": 2}), "{total}");
        let notes = &doc["notes"];
        assert_eq!(notes["aio_completed_in_submit"], 1, "{notes}");
        // six completions, three results collected by a call
        assert_eq!(notes["aio_completed_never_reaped"], 3, "{notes}");
        assert_eq!(notes["aio_results_assumed_complete"], 1, "{notes}");
        assert!(notes.get("aio_requests_not_counted").is_some_and(|n| n == 1), "{notes}");
        // the same calls without the records: counted when reaped, by aio_data, as strace's are,
        // which gives the write's result to the read and the read's to the write
        let plain: String = lines.lines().filter(|l| !l.contains("aeiou_")).collect::<Vec<_>>().join("\n");
        let (doc, _) = metrics(plain.as_bytes(), &o).unwrap();
        assert!(doc["notes"].get("aio_completed_in_submit").is_none(), "{}", doc["notes"]);
        assert_eq!((doc["total"]["bytes_read"].as_u64(), doc["total"]["bytes_written"].as_u64()), (Some(100 + 4096 + 10), Some(4096)), "{}", doc["total"]);

        // the export's submits carry their members' results
        let opener = || -> std::io::Result<Box<dyn BufRead + '_>> { Ok(Box::new(lines.as_bytes())) };
        let mut out = Vec::new();
        crate::export::strace(&opener, "/d", &[], None, &mut out).unwrap();
        let ev: Vec<Value> = out.split(|b| *b == b'\n').filter(|l| !l.is_empty()).skip(1).map(|l| serde_json::from_slice(l).unwrap()).collect();
        let subs: Vec<&Value> = ev.iter().filter(|e| e["op"] == "submit").collect();
        assert_eq!(subs.len(), 3, "{ev:?}");
        let rets = |e: &Value| e["ops"].as_array().unwrap().iter().map(|m| m["ret"].clone()).collect::<Vec<_>>();
        assert_eq!(rets(subs[0]), [json!(1000), json!(100)], "{ev:?}");
        assert_eq!(rets(subs[1]), [json!("EIO"), json!(4096)], "{ev:?}");
    }

    /// POSIX AIO in an eBPF capture (DESIGN_REVIEW.md §3.98 "Phase 4 step 4"): a request counted
    /// once, in the thread that issued it, when the helper's call its `aeiou_aio_serve` names
    /// returns, with the bytes it moved (an interrupted call waiting for its retry); an fsync as
    /// the issuer's; the helper's other calls its own; a request refused, canceled, or never
    /// served not counted, and noted, as are those whose results no `aio_return` collected; the
    /// issue's members a fan-out, and the export's submits carry their results.
    #[test]
    fn posix_aio_requests_count_in_their_issuer_when_served() {
        let lines = [
            "0 1.000000 --- aeiou: trace format 1, records aeiou_aio_read aeiou_aio_write aeiou_aio_fsync aeiou_lio_listio aeiou_aio_serve aeiou_aio_suspend aeiou_aio_return aeiou_aio_cancel aeiou_aio_refused ---",
            "1 1.000001 openat(AT_FDCWD</d>, \"/d/a\", O_RDWR) = 7</d/a> <0.000010>",
            "1 1.000002 openat(AT_FDCWD</d>, \"/d/g\", O_WRONLY|O_CREAT|O_APPEND|O_TRUNC, 0644) = 8</d/g> <0.000010>",
            "1 1.000003 aeiou_aio_read(id=1, aiocb=0xa000, fd=7</d/a>, off=0, len=4096, notify=SIGEV_NONE) = 0",
            "1 1.000004 clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x1, parent_tid=0x1, exit_signal=0, stack=0x1, stack_size=0x6000, tls=0x1} => {parent_tid=[2]}, 88) = 2 <0.000010>",
            "2 1.000005 aeiou_aio_serve(id=1) = 0",
            "2 1.000005 pread64(7</d/a>, 0x1, 4096, 0) = 1000 <0.000010>",
            "1 1.000006 aeiou_aio_suspend(ids=[1], timeout=NULL) = 0 <0.000010>",
            "1 1.000016 aeiou_aio_return(id=1) = 1000",
            "1 1.000020 aeiou_aio_write(id=1, aiocb=0xa100, fd=8</d/g>, off=999999, len=100, notify=SIGEV_THREAD) = 0",
            "2 1.000021 aeiou_aio_serve(id=1) = 0",
            "2 1.000021 pwrite64(8</d/g>, 0x1, 100, 999999) = -1 EINTR (Interrupted system call) <0.000001>",
            "2 1.000022 aeiou_aio_serve(id=1) = 0",
            "2 1.000022 pwrite64(8</d/g>, 0x1, 100, 999999) = 100 <0.000001>",
            "1 1.000030 aeiou_aio_fsync(id=2, aiocb=0xa180, fd=7</d/a>, op=O_DSYNC, notify=SIGEV_NONE) = 0",
            "2 1.000031 aeiou_aio_serve(id=2) = 0",
            "2 1.000031 fdatasync(7</d/a>) = 0 <0.000100>",
            "1 1.000040 aeiou_lio_listio(mode=LIO_WAIT, list=[{id=3, aiocb=0xa200, op=LIO_READ, fd=7</d/a>, off=0, len=10, notify=SIGEV_NONE}, {id=4, aiocb=0xa300, op=LIO_READ, fd=7</d/a>, off=4096, len=10, notify=SIGEV_NONE}], nent=2) = 0",
            "2 1.000041 aeiou_aio_serve(id=3) = 0",
            "2 1.000041 pread64(7</d/a>, 0x1, 10, 0) = 10 <0.000001>",
            "2 1.000042 aeiou_aio_serve(id=4) = 0",
            "2 1.000042 pread64(7</d/a>, 0x1, 10, 4096) = 10 <0.000001>",
            "1 1.000050 aeiou_aio_read(id=5, aiocb=0xa400, fd=7</d/a>, off=0, len=1, notify=SIGEV_NONE) = 0",
            "1 1.000050 aeiou_aio_refused(id=5) = -1",
            "1 1.000060 aeiou_aio_read(id=5, aiocb=0xa500, fd=7</d/a>, off=0, len=1, notify=SIGEV_NONE) = 0",
            "1 1.000060 aeiou_aio_cancel(fd=7</d/a>, aiocb=NULL, canceled=[5]) = 0 (AIO_CANCELED) <0.000001>",
            "1 1.000070 aeiou_aio_read(id=5, aiocb=0xa600, fd=7</d/a>, off=0, len=7, notify=SIGEV_NONE) = 0",
            // the helper thread's own call
            "2 1.000080 pread64(7</d/a>, 0x1, 3, 0) = 3 <0.000001>",
        ]
        .join("\n");
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let (doc, _) = metrics(lines.as_bytes(), &o).unwrap();
        let total = &doc["total"];
        assert_eq!((total["bytes_read"].as_u64(), total["bytes_written"].as_u64()), (Some(1000 + 10 + 10 + 3), Some(100)), "{total}");
        assert_eq!((total["counts"]["read"].as_u64(), total["counts"]["write"].as_u64(), total["counts"]["fdatasync"].as_u64()), (Some(4), Some(1), Some(1)), "{total}");
        assert_eq!(total["fan_out"], json!({"1": 5, "2": 1}), "{total}");
        let notes = &doc["notes"];
        assert_eq!((notes["posix_aio_refused"].as_u64(), notes["posix_aio_canceled"].as_u64(), notes["posix_aio_never_served"].as_u64(), notes["posix_aio_never_collected"].as_u64()), (Some(1), Some(1), Some(1), Some(4)), "{notes}");
        assert!(notes.get("not_counted:aeiou_aio_suspend").is_none() && notes.get("not_counted:aeiou_aio_serve").is_none(), "{notes}");
        // the same trace without the records: the helper's calls counted as its own, as strace's are
        let plain: String = lines.lines().filter(|l| !l.contains("aeiou_")).collect::<Vec<_>>().join("\n");
        let (doc, _) = metrics(plain.as_bytes(), &o).unwrap();
        assert!(doc["notes"].get("posix_aio_never_served").is_none(), "{}", doc["notes"]);
        assert_eq!(doc["total"]["counts"]["write"], 2, "{}", doc["total"]);

        // the export: each issue a submit in the issuer's lane, its members' results from the
        // helper's calls; the fsync the issuer's; the helper's own read in its lane
        let opener = || -> std::io::Result<Box<dyn BufRead + '_>> { Ok(Box::new(lines.as_bytes())) };
        let mut out = Vec::new();
        crate::export::strace(&opener, "/d", &[], None, &mut out).unwrap();
        let ev: Vec<Value> = out.split(|b| *b == b'\n').filter(|l| !l.is_empty()).skip(1).map(|l| serde_json::from_slice(l).unwrap()).collect();
        let lane = |op: &str| ev.iter().filter(|e| e["op"] == op).map(|e| e["lane"].as_u64().unwrap()).collect::<Vec<_>>();
        let subs: Vec<&Value> = ev.iter().filter(|e| e["op"] == "submit").collect();
        let rets = |e: &Value| e["ops"].as_array().unwrap().iter().map(|m| m["ret"].clone()).collect::<Vec<_>>();
        // the requests refused, canceled, and never served are not replayed: their submits gone
        assert_eq!(subs.iter().map(|s| rets(s)).collect::<Vec<_>>(), [vec![json!(1000)], vec![json!(100)], vec![json!(10), json!(10)]], "{ev:?}");
        let head: Value = serde_json::from_slice(out.split(|b| *b == b'\n').next().unwrap()).unwrap();
        let n = &head["notes"];
        assert_eq!((n["not_exported:posix_aio:refused"].as_u64(), n["not_exported:posix_aio:canceled"].as_u64(), n["not_exported:posix_aio:never served"].as_u64(), n["calls_not_exported"].as_u64()), (Some(1), Some(1), Some(1), Some(3)), "{head}");
        let issuer = subs[0]["lane"].as_u64().unwrap();
        assert!(subs.iter().all(|s| s["lane"] == issuer), "{ev:?}");
        assert_eq!(lane("fdatasync"), [issuer], "{ev:?}");
        assert_eq!(lane("read").len(), 1, "{ev:?}");
        assert_ne!(lane("read"), [issuer], "{ev:?}");
    }

    #[test]
    fn a_request_at_the_file_s_position_reads_there_and_moves_it() {
        // offset -1: the file's position, which the kernel reads at issue and moves by the result
        let lines = [
            "1 1.000001 openat(AT_FDCWD</d>, \"/d/a\", O_RDONLY) = 7</d/a> <0.000010>",
            "1 1.000002 read(7</d/a>, \"\"..., 100, 0) = 100 <0.000010>",
            "1 1.000011 aeiou_uring_sqe(ring=5, id=1, op=IORING_OP_READ, fd=7</d/a>, off=-1, len=50, user_data=0x0) = 0",
            "1 1.000012 aeiou_uring_cqe(ring=5, id=1, user_data=0x0, flags=0) = 50",
            "1 1.000013 read(7</d/a>, \"\"..., 10) = 10 <0.000010>",
            // a vectored request's length and a multishot's results are not a submit's: left out of
            // the export, noted, and counted by the metrics
            "1 1.000021 aeiou_uring_sqe(ring=5, id=2, op=IORING_OP_READV, fd=7</d/a>, off=0, iovcnt=2, user_data=0x0) = 0",
            "1 1.000022 aeiou_uring_cqe(ring=5, id=2, user_data=0x0, flags=0) = 30",
            "1 1.000031 aeiou_uring_sqe(ring=5, id=3, op=IORING_OP_READ_MULTISHOT, fd=7</d/a>, off=0, len=0, user_data=0x0) = 0",
            "1 1.000032 aeiou_uring_cqe(ring=5, id=3, user_data=0x0, flags=IORING_CQE_F_MORE) = 5",
            "1 1.000033 aeiou_uring_cqe(ring=5, id=3, user_data=0x0, flags=0) = 0",
            // a multishot at the position: each result where the last left it
            "1 1.000041 aeiou_uring_sqe(ring=5, id=4, op=IORING_OP_READ_MULTISHOT, fd=7</d/a>, off=-1, len=0, user_data=0x0) = 0",
            "1 1.000042 aeiou_uring_cqe(ring=5, id=4, user_data=0x0, flags=IORING_CQE_F_MORE) = 7",
            "1 1.000043 aeiou_uring_cqe(ring=5, id=4, user_data=0x0, flags=0) = 3",
        ]
        .join("\n");
        let opener = || -> std::io::Result<Box<dyn BufRead + '_>> { Ok(Box::new(lines.as_bytes())) };
        let mut out = Vec::new();
        let sum = crate::export::strace(&opener, "/d", &[], None, &mut out).unwrap();
        let ev: Vec<Value> = out.split(|b| *b == b'\n').filter(|l| !l.is_empty()).skip(1).map(|l| serde_json::from_slice(l).unwrap()).collect();
        // written as the sequential read it is, so a replay moves its position as the kernel did
        let reads: Vec<&Value> = ev.iter().filter(|e| e["op"] == "read").collect();
        assert_eq!(reads.len(), 5, "{ev:?}");
        assert!(reads[1]["len"] == 50 && reads[1].get("offset").is_none(), "{ev:?}");
        assert_eq!(ev.iter().filter(|e| e["op"] == "submit").count(), 0, "{ev:?}");
        let notes = &sum.header["notes"];
        assert_eq!((&notes["not_exported:readv:length not traced"], &notes["not_exported:read:multishot"]), (&json!(1), &json!(1)), "{notes}");
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let (doc, _) = metrics(lines.as_bytes(), &o).unwrap();
        assert_eq!(doc["total"]["bytes_read"], 100 + 50 + 10 + 30 + 5 + 7 + 3, "{}", doc["total"]);
        // 0 to 100, 100 to 150, 150 to 160: one run, the ring's read where the position was; and
        // the multishot's 160 to 167 and 167 to 170 another, after the reads at 0 between
        assert_eq!(doc["total"]["run_length"]["read"]["multi_op_bytes"], 160 + 10, "{}", doc["total"]["run_length"]);
    }

    #[test]
    fn a_multishot_request_stays_in_flight_while_its_completions_say_more() {
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let mut t = Tracer::new(&o, true, None);
        for l in [
            "1 1.000001 openat(AT_FDCWD</d>, \"/d/a\", O_RDONLY) = 7</d/a> <0.000010>",
            "1 1.000011 aeiou_uring_sqe(ring=5, id=1, op=IORING_OP_READ_MULTISHOT, fd=7</d/a>, off=0, len=0, user_data=0x0) = 0",
            "1 1.000012 aeiou_uring_cqe(ring=5, id=1, user_data=0x0, flags=IORING_CQE_F_BUFFER|IORING_CQE_F_MORE) = 10",
            "1 1.000013 aeiou_uring_cqe(ring=5, id=1, user_data=0x0, flags=IORING_CQE_F_BUFFER|IORING_CQE_F_MORE) = 20",
            "1 1.000014 aeiou_uring_cqe(ring=5, id=1, user_data=0x0, flags=0) = 0",
        ] {
            t.line(&format!("{l}\n"));
        }
        t.finish().unwrap();
        let doc = metrics_document(&t, &o);
        assert_eq!(doc["total"]["bytes_read"], 30, "{}", doc["total"]);
        assert!(doc["notes"].get("uring_completions_without_submission").is_none(), "{}", doc["notes"]);
    }

    #[test]
    fn a_trace_printing_arguments_or_bytes_is_noted_and_the_capture_s_is_not() {
        let o = Opts { roots: vec!["/d".into()], ..Opts::default() };
        let clean = [
            r#"1 1.000000 execve("/bin/x", 0x7ffe, 0x7fff) = 0 <0.000100>"#,
            r#"1 1.000000 execveat(3</bin>, "x", 0x7ffe, 0x7fff, 0) = 0 <0.000100>"#,
            r#"1 1.000000 write(1</d/a, \"b\">, ""..., 5) = 5 <0.000010>"#,
            r#"1 1.000000 write(1</d/a>, "", 0) = 0 <0.000010>"#,
            r#"1 1.000000 writev(3</d/a>, [{iov_base=""..., iov_len=1}, {iov_base=""..., iov_len=2}], 2) = 3 <0.000010>"#,
            r#"1 1.000000 sendto(4<UNIX-STREAM:[9,"/run/s"]>, ""..., 3, 0, {sa_family=AF_UNIX, sun_path="/run/s"}, 9) = 3 <0.000010>"#,
            // an address strace decodes, under -s 0 too: a netlink listing's, a datagram's sender (LMCache's capture, 2026-11-05)
            r#"1 1.000000 recvmsg(21<NETLINK:[ROUTE:1]>, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[[{nlmsg_len=80, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1, nlmsg_pid=1}, {ifa_family=AF_INET6, ifa_prefixlen=128, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_HOST, ifa_index=if_nametoindex("lo")}, [[{nla_len=20, nla_type=IFA_ADDRESS}, inet_pton(AF_INET6, "::1")], [{nla_len=5, nla_type=IFA_PROTO}, ""...]]]], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 80 <0.000010>"#,
            r#"1 1.000000 recvfrom(5<UDP:[2]>, ""..., 512, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("10.0.0.1")}, [16]) = 40 <0.000010>"#,
            r#"1 1.000000 recvfrom(5<UDPv6:[3]>, ""..., 512, 0, {sa_family=AF_INET6, sin6_port=htons(53), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "fe80::1", &sin6_addr), sin6_scope_id=0}, [28]) = 40 <0.000010>"#,
            r#"1 1.000000 read(3</d/a>,  <unfinished ...>"#,
            r#"1 1.000100 <... read resumed>""..., 5) = 5 <0.000100>"#,
            r#"1 1.000000 openat(AT_FDCWD</d>, "/d/a", O_RDONLY) = 3</d/a> <0.000010>"#,
        ];
        let mut t = Tracer::new(&o, true, None);
        for l in clean {
            t.line(&format!("{l}\n"));
        }
        assert!(!t.notes.contains_key(ARGUMENTS_PRINTED) && !t.notes.contains_key(DATA_PRINTED), "{:?}", t.notes);
        assert!(metrics_document(&t, &o)["notes"].get("trace_may_hold_secrets").is_none());
        let printing = [
            (r#"1 1.000000 execve("/bin/x", ["x", "--password=p"], ["TOKEN=t"]) = 0 <0.000100>"#, ARGUMENTS_PRINTED),
            (r#"1 1.000000 execveat(3</bin>, "x", ["x"], 0x7fff, 0) = 0 <0.000100>"#, ARGUMENTS_PRINTED),
            (r#"1 1.000000 write(1</d/a>, "key=v"..., 5) = 5 <0.000010>"#, DATA_PRINTED),
            (r#"1 1.000000 writev(3</d/a>, [{iov_base=""..., iov_len=1}, {iov_base="Authorization: x", iov_len=16}], 2) = 17 <0.000010>"#, DATA_PRINTED),
            (r#"1 1.000100 <... read resumed>"secret"..., 5) = 5 <0.000100>"#, DATA_PRINTED),
            (r#"1 1.000000 recvfrom(5<UDPv6:[3]>, "secret"..., 512, 0, {sa_family=AF_INET6, sin6_port=htons(53), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "fe80::1", &sin6_addr), sin6_scope_id=0}, [28]) = 40 <0.000010>"#, DATA_PRINTED),
        ];
        for (l, k) in printing {
            let mut t = Tracer::new(&o, true, None);
            if l.contains("resumed") {
                t.line("1 1.000000 read(3</d/a>,  <unfinished ...>\n");
            }
            t.line(&format!("{l}\n"));
            assert_eq!(t.notes.get(k), Some(&1), "{l}: {:?}", t.notes);
            let doc = metrics_document(&t, &o);
            assert!(doc["notes"]["trace_may_hold_secrets"].as_str().unwrap().contains("verbose=!execve,execveat -s 0"), "{l}");
        }
    }

    #[test]
    fn descriptors_and_their_paths() {
        assert_eq!(fd_arg("3</p>"), (Some(3), Some("/p".into())));
        assert_eq!(fd_arg("AT_FDCWD</d>"), (None, Some("/d".into())));
        assert_eq!(fd_arg("3"), (Some(3), None));
        assert_eq!(fd_arg("x"), (None, None));
        assert_eq!(fd_arg("3<p"), (None, None));
        assert_eq!(fd_arg("4</dev/null<char 1:3>>"), (Some(4), Some("/dev/null".into())));
        assert_eq!(fd_arg("5</gone (deleted)>"), (Some(5), Some("/gone".into())));
        // strace 6.8 prints `(deleted)` after the brackets (FS_API_SWEEP.md S-5)
        assert_eq!(fd_arg("5</gone>(deleted)"), (Some(5), Some("/gone".into())));
        assert_eq!(fd_arg("AT_FDCWD</gone>(deleted)"), (None, Some("/gone".into())));
        assert_eq!(fd_arg("5</gone>(other)"), (None, None));
        assert_eq!(fd_path(r"/a\74b\76"), "/a<b>");
    }

    /// The Python implementation's cases (`builder/tests/test_trace.py` until 2026-10-29): a
    /// descriptor's `-yy` path is unescaped as a quoted string is, and its commas do not split.
    #[test]
    fn escaped_descriptor_paths_as_the_python_had_them() {
        assert_eq!(fd_arg(r"3</r/caf\303\251>"), (Some(3), Some("/r/café".into())));
        assert_eq!(fd_arg(r"3</r/c\74d\76e>"), (Some(3), Some("/r/c<d>e".into())));
        assert_eq!(fd_arg(r"3</r/x\\y>"), (Some(3), Some(r"/r/x\y".into())));
        assert_eq!(fd_arg("AT_FDCWD</r/a b>"), (None, Some("/r/a b".into())));
        assert_eq!(fd_arg(r"4</r/caf\303\251 (deleted)>"), (Some(4), Some("/r/café".into())));
        assert_eq!(fd_arg(r"1</dev/n\74ll\76<char 1:3>>"), (Some(1), Some("/dev/n<ll>".into())));
        assert_eq!(fd_arg("0</dev/null<char 1:3>>"), (Some(0), Some("/dev/null".into())));
        assert_eq!(split_args(r#"3</r/c\74d,e\76>, "x"..., 4096"#), vec![r"3</r/c\74d,e\76>", r#""x"..."#, "4096"]);
        assert_eq!(split_args(r#"3</d/a, b>, "x, \"y\"", 10"#), vec!["3</d/a, b>", r#""x, \"y\"""#, "10"]);
        assert_eq!(split_args(r#"AT_FDCWD</cwd>, "p", O_RDONLY|O_CLOEXEC"#), vec!["AT_FDCWD</cwd>", r#""p""#, "O_RDONLY|O_CLOEXEC"]);
        assert_eq!(split_args(r#"5, [{iov_base="a,b", iov_len=3}, {iov_base="", iov_len=5}], 2"#), vec!["5", r#"[{iov_base="a,b", iov_len=3}, {iov_base="", iov_len=5}]"#, "2"]);
        assert_eq!(c_string(r#""caf\303\251""#), Some("café".into()));
    }

    /// `CLONE_FILES` read from the flags argument only, whole flag names only.
    #[test]
    fn shares_files_reads_the_flags_argument_only() {
        let legacy = "child_stack=0x7f5c3a1fdfb0, flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, parent_tid=[101], tls=0x7f5c3a1fe700, child_tidptr=0x7f5c3a1fe9d0";
        let clone3 = "{flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f5c3a1fe9d0, parent_tid=0x7f5c3a1fe9d0, exit_signal=0, stack=0x7f5c399fd000, stack_size=0x7fff00, tls=0x7f5c3a1fe6c0} => {parent_tid=[101]}, 88";
        assert!(shares_files("clone", legacy));
        assert!(shares_files("clone3", clone3));
        assert!(!shares_files("clone", "child_stack=NULL, flags=CLONE_CHILD_CLEARTID|SIGCHLD, child_tidptr=0x7f"));
        assert!(!shares_files("clone", "child_stack=NULL, flags=CLONE_FILESYSTEM_X"));
        assert!(!shares_files("fork", "flags=CLONE_FILES"));
    }

    #[test]
    fn escapes_undone_or_left() {
        assert_eq!(unescape(r"caf\303\251"), "café");
        assert_eq!(unescape(r"a\nb\t\\\x41"), "a\nb\t\\A");
        assert_eq!(unescape(r"\q"), r"\q");
        assert_eq!(unescape(r"end\"), r"end\");
        assert_eq!(unescape(r"\777"), r"\777");
        assert_eq!(unescape(r"\xZZ"), r"\xZZ");
        assert_eq!(unescape(r"\u00e9"), "\u{fffd}");
        assert_eq!(unescape(r"\N{x}"), r"\N{x}");
        assert_eq!(unescape("plain"), "plain");
        assert_eq!(unescape("\u{4e2d}\\n"), "\u{4e2d}\\n");
        assert_eq!(c_string(r#""a\"b""#), Some("a\"b".into()));
        assert_eq!(c_string("x"), None);
        assert_eq!(c_string("\""), None);
    }

    #[test]
    fn paths_normalize_as_linux_resolves_them() {
        assert_eq!(normpath("//d"), "/d");
        assert_eq!(normpath("///d/./e/../f"), "/d/f");
        assert_eq!(normpath(""), ".");
        assert_eq!(normpath("a/../.."), "..");
        assert_eq!(normpath("/.."), "/");
        assert_eq!(join("/a/", "b"), "/a/b");
        assert_eq!(join("", "b"), "b");
        assert_eq!(join("/a", "/b"), "/b");
        assert!(abspath("x").ends_with("/x"));
        assert_eq!(abspath("/a/b/.."), "/a");
    }

    #[test]
    fn globs_as_fnmatch() {
        assert!(glob("a/b.aeiou-x", "*.aeiou-*"));
        assert!(glob("abc", "a?c"));
        assert!(glob("a1", "a[0-9]"));
        assert!(!glob("ab", "a[!b]"));
        assert!(glob("a[", "a["));
        assert!(glob("", "*"));
        assert!(!glob("abc", "a*d"));
        assert!(glob("a]", "a[]]"));
    }

    #[test]
    fn line_prefixes() {
        assert_eq!(prefix("[pid  12] 1.5 open("), (12, Some("1.5"), "open("));
        assert_eq!(prefix("12 10:00:01.25 x"), (12, Some("10:00:01.25"), "x"));
        assert_eq!(prefix("10:00:01 x"), (0, Some("10:00:01"), "x"));
        assert_eq!(prefix("x"), (0, None, "x"));
        assert_eq!(prefix("7 x"), (7, None, "x"));
        assert_eq!(seconds("01:00:01.5"), 3601.5);
        assert_eq!(resumed("<... read resumed> , 3) = 3"), Some(("read", ", 3) = 3")));
        assert_eq!(resumed("<... resumed>"), None);
        assert_eq!(ret_token("0x7f = x"), Some(("0", 1)));
        assert_eq!(ret_token("?"), Some(("?", 1)));
        assert_eq!(ret_token("x"), None);
        assert_eq!(int0("012"), None);
        assert_eq!(int0("0x10"), Some(16));
        assert_eq!(int0("-0o7"), Some(-7));
    }

    #[test]
    fn aio_arrays() {
        let s = "[{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/a, b>, aio_buf=0x1, aio_nbytes=4096, aio_offset=0}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_FSYNC, aio_fildes=4}]";
        let cbs = iocbs(s);
        assert_eq!(cbs.len(), 2);
        assert_eq!((cbs[0].fd, cbs[0].path.as_deref(), cbs[0].nbytes, cbs[0].offset), (3, Some("/d/a, b"), 4096, 0));
        assert_eq!((cbs[1].op.as_str(), cbs[1].path.as_deref(), cbs[1].nbytes, cbs[1].offset), ("FSYNC", None, 0, 0));
        assert!(iocbs("aio_data=, x").is_empty());
        // every request kind as strace 6.8 prints it (FS_API_SWEEP.md S-6): the syncs, NOOP, and
        // POLL end at the descriptor, a vectored request has an iovec and no `aio_nbytes`, and
        // `aio_rw_flags` comes before the opcode; none runs into the next request
        let s = r#"[{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</r/f>, aio_buf=0x5bc64f97d040, aio_nbytes=4096, aio_offset=4096}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_FSYNC, aio_fildes=3</r/f>}, {aio_data=0x3, aio_lio_opcode=IOCB_CMD_PWRITEV, aio_fildes=3</r/f>, aio_buf=[{iov_base="xxxxxxxx"..., iov_len=4096}, {iov_base="xxxxxxxx"..., iov_len=4096}], aio_offset=0}, {aio_data=0x4, aio_rw_flags=RWF_DSYNC, aio_lio_opcode=IOCB_CMD_PWRITE, aio_fildes=3</r/f>, aio_buf="xxxxxxxx"..., aio_nbytes=4096, aio_offset=4096}, {aio_data=0x5, aio_lio_opcode=IOCB_CMD_FDSYNC, aio_fildes=3</r/f>(deleted)}, {aio_data=0x6, aio_lio_opcode=IOCB_CMD_PREADV, aio_fildes=3</r/f>, aio_buf=[{iov_base=0x5bc64f97d040, iov_len=4096}, {iov_base=0x5bc64f97e040, iov_len=512}], aio_offset=8192}, {aio_data=0x7, aio_lio_opcode=IOCB_CMD_NOOP, aio_fildes=3</r/f>}, {aio_data=0x9, aio_lio_opcode=IOCB_CMD_POLL, aio_fildes=3</r/f>, aio_buf=POLLIN}, {aio_data=0xa, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=4</r/c\74d\76>, aio_buf=0x1, aio_nbytes=10, aio_offset=20, aio_flags=IOCB_FLAG_RESFD, aio_resfd=9}, ...]"#;
        let got: Vec<(String, String, i64, Option<String>, i64, i64)> = iocbs(s).into_iter().map(|c| (c.data, c.op, c.fd, c.path, c.nbytes, c.offset)).collect();
        let f = || Some("/r/f".to_string());
        assert_eq!(
            got,
            vec![
                ("0x1".into(), "PREAD".into(), 3, f(), 4096, 4096),
                ("0x2".into(), "FSYNC".into(), 3, f(), 0, 0),
                ("0x3".into(), "PWRITEV".into(), 3, f(), 8192, 0),
                ("0x4".into(), "PWRITE".into(), 3, f(), 4096, 4096),
                ("0x5".into(), "FDSYNC".into(), 3, f(), 0, 0),
                ("0x6".into(), "PREADV".into(), 3, f(), 4608, 8192),
                ("0x7".into(), "NOOP".into(), 3, f(), 0, 0),
                ("0x9".into(), "POLL".into(), 3, f(), 0, 0),
                ("0xa".into(), "PREAD".into(), 4, Some("/r/c<d>".into()), 10, 20),
            ]
        );
        assert_eq!(events("[{data=0x1, obj=0x2, res=4096, res2=0}, {data=0x3, obj=0x4, res=-5}]"), vec![("0x1".to_string(), 4096), ("0x3".to_string(), -5)]);
        assert_eq!(iov_lens("[{iov_base=\"\", iov_len=3}, {iov_base=\"\", iov_len=4}]"), 7);
        assert!(shares_files("clone3", "{flags=CLONE_VM|CLONE_FILES, exit_signal=0}"));
        assert!(!shares_files("clone", "child_stack=NULL, flags=CLONE_VM, x=CLONE_FILES"));
        assert!(!shares_files("fork", "flags=CLONE_FILES"));
        assert!(!shares_files("clone", "xflags=CLONE_FILES"));
        assert!(!shares_files("clone", "flags=CLONE_FILESX"));
        assert_eq!(errno_name(2), "ENOENT");
        assert_eq!(errno_name(99), "E99");
    }
}
