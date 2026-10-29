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
//! and page faults on a mapping. The trace is read once, in a stream; the state is per
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
const UNFINISHED: &str = "<unfinished ...>";
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

/// `3</path>` → `(Some(3), Some("/path"))`; `AT_FDCWD</cwd>` → `(None, Some("/cwd"))`; a
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

/// The requests of an `io_submit` array (`_IOCB` of the Python, every match in turn).
fn iocbs(s: &str) -> Vec<Iocb> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = s[from..].find("aio_data=").map(|i| i + from) {
        match iocb_at(s, i + 9) {
            Some((cb, end)) => {
                out.push(cb);
                from = end;
            }
            None => from = i + 1,
        }
    }
    out
}

fn iocb_at(s: &str, i: usize) -> Option<(Iocb, usize)> {
    let data = word_at(s, i);
    if data.is_empty() {
        return None;
    }
    let mut at = i + data.len();
    at += s[at..].strip_prefix(", aio_lio_opcode=IOCB_CMD_").map(|_| 26)?;
    let op = word_at(s, at);
    if op.is_empty() {
        return None;
    }
    at += op.len();
    // `.*?aio_fildes=(\d+)(?:<(.*?)>)?, `: the first fildes that parses
    let mut search = at;
    loop {
        let j = s[search..].find("aio_fildes=").map(|j| j + search)?;
        search = j + 1;
        let Some((fd, k)) = num_at(s, j + 11, false) else { continue };
        let (path, after) = if s[k..].starts_with('<') {
            // the shortest `<...>` followed by `, `
            let mut e = k + 1;
            let mut found = None;
            while let Some(g) = s[e..].find(">, ").map(|g| g + e) {
                found = Some(g);
                break;
            }
            match found {
                Some(g) => {
                    e = g;
                    (Some(s[k + 1..e].to_string()), e + 3)
                }
                None => continue,
            }
        } else if s[k..].starts_with(", ") {
            (None, k + 2)
        } else {
            continue;
        };
        // `.*?aio_nbytes=(\d+), aio_offset=(-?\d+)`
        let mut n_search = after;
        while let Some(m) = s[n_search..].find("aio_nbytes=").map(|m| m + n_search) {
            n_search = m + 1;
            let Some((nbytes, k2)) = num_at(s, m + 11, false) else { continue };
            let Some(rest) = s[k2..].strip_prefix(", aio_offset=") else { continue };
            let o = s.len() - rest.len();
            let Some((offset, end)) = num_at(s, o, true) else { continue };
            return Some((Iocb { data: data.to_string(), op: op.to_string(), fd, path, nbytes, offset }, end));
        }
        return None;
    }
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
}

type FileRef = Rc<RefCell<File>>;
type Table = Rc<RefCell<HashMap<i64, FileRef>>>;

fn new_file(path: Option<String>, pos: i64, append: bool) -> FileRef {
    Rc::new(RefCell::new(File { path, pos, append, listed: false, oid: None }))
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
        "truncate" => ("ftruncate", None, 0),
        "mkdir" => ("mkdir", None, 0),
        "mkdirat" => ("mkdir", Some(0), 1),
        "rmdir" => ("rmdir", None, 0),
        "rename" => ("rename", None, 0),
        "renameat" | "renameat2" => ("rename", Some(0), 1),
        _ => return None,
    })
}

fn handled(name: &str) -> bool {
    OPENS.contains(&name)
        || READS.contains(&name)
        || WRITES.contains(&name)
        || CLONES.contains(&name)
        || DUPS.contains(&name)
        || fd_op_of(name).is_some()
        || path_op_of(name).is_some()
        || matches!(name, "close" | "lseek" | "io_setup" | "io_submit" | "io_getevents" | "io_pgetevents" | "chdir" | "fchdir" | "mmap")
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
    off: i64,
    nbytes: i64,
    member: Option<(usize, usize)>,
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
    aio_unused: BTreeMap<String, u64>,
    clock: f64,
    pub notes: BTreeMap<String, u64>,
    pub tids: std::collections::HashSet<i64>,
    under_cache: HashMap<String, Option<String>>,
    /// The size of each file under the roots the trace made known (created, truncated, or
    /// renamed into place since), kept as its writes extend it: an `O_APPEND` write's base.
    extent: HashMap<String, i64>,
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
            aio_unused: BTreeMap::new(),
            clock: 0.0,
            notes: BTreeMap::new(),
            tids: Default::default(),
            under_cache: HashMap::new(),
            extent: HashMap::new(),
            seen: Seen::default(),
            error: None,
        }
    }

    fn note(&mut self, k: &str, n: u64) {
        *self.notes.entry(k.to_string()).or_insert(0) += n;
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
        table.borrow_mut().insert(fd, f.clone());
        if self.under(Some(path)).is_some() {
            // opened before the trace began: its position is taken as 0
            self.note("descriptors_without_open", 1);
        }
        Some(f)
    }

    // ---- the stream

    pub fn feed(&mut self, input: impl BufRead) -> Result<()> {
        for_each_line(input, |line| {
            self.seen.lines += 1;
            self.line(line);
            match self.error.take() {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })
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
        if body.starts_with("+++") {
            if let Some(m) = self.m() {
                m.end_context(tid);
            }
            self.end_chain(tid);
            return;
        }
        if body.starts_with("---") || body.is_empty() {
            return;
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
            if handled(name) {
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
            return;
        }
        let Some(k) = body.rfind(") = ") else { return };
        let ret_s = &body[k + 4..];
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
        let name = name.to_string();
        if let Err(e) = self.call(tid, &name, &args, ret, ret_path, issued, done, err) {
            self.error = Some(e);
        }
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

    fn data(&mut self, kind: usize, inst: i64, tid: i64, p: &str, off: i64, len: i64, n: i64) -> Result<()> {
        match self.metrics.as_mut() {
            Some(m) => m.data(kind, inst, tid, p, off, len, n),
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
        let inst = self.instance(tid);
        if inst.is_none() {
            // the descriptor tables are still kept: a child inside an instance may inherit them
            self.note("calls_outside_instances", 1);
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
            let flags = a.get(pi + 1).cloned().unwrap_or_default();
            let p = self.under(path.as_deref());
            let mut f = None;
            if ret >= 0 {
                if let Some(p) = &p {
                    if flags.contains("O_TRUNC") || (flags.contains("O_CREAT") && flags.contains("O_EXCL")) || name == "creat" {
                        self.extent.insert(p.clone(), 0);
                    }
                    if flags.contains("O_APPEND") && !self.extent.contains_key(p) {
                        self.note("append_to_unknown_size", 1);
                    }
                }
                let file = new_file(path.clone(), 0, flags.contains("O_APPEND"));
                self.table(tid).borrow_mut().insert(ret, file.clone());
                f = Some(file);
            }
            if let (Some(_), Some(p)) = (inst, &p) {
                self.count(tid, "open");
                if let Some(ex) = self.export.as_mut() {
                    let mode = a.get(pi + 2).map(String::as_str);
                    let fl = if name != "creat" { flags.as_str() } else { "O_WRONLY|O_CREAT|O_TRUNC" };
                    let oid = ex.open(when, p, fl, mode, err)?;
                    if let Some(f) = &f {
                        f.borrow_mut().oid = oid;
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
                if let (Some(ex), Some(f)) = (self.export.as_mut(), &f) {
                    if let Some(oid) = f.borrow().oid {
                        ex.op(when, "close", oid, err, Value::Null, vec![]);
                    }
                }
            }
            return Ok(());
        }
        if DUPS.contains(&name) {
            if name == "fcntl" && !(a.len() > 1 && a[1].starts_with("F_DUPFD")) {
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
                if let (Some(ex), Some(f)) = (self.export.as_mut(), &f) {
                    if let (Some(oid), true) = (f.borrow().oid, a.len() > 2) {
                        if let Some(off) = int0(&a[1]) {
                            let whence = a[2].replace("SEEK_", "");
                            ex.op(when, "lseek", oid, err, json!(ret), vec![("offset", json!(off)), ("whence", json!(whence))]);
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
            let base = match (&p, &f) {
                (Some(p), Some(f)) if kind == W && f.borrow().append => self.extent.get(p).copied(),
                _ => None,
            };
            let offset;
            if let Some(b) = base {
                // `O_APPEND`: at the end of the file, a `pwrite`'s offset ignored, the position
                // left at the end
                offset = b;
                positioned = true;
                if ret > 0 {
                    f.as_ref().unwrap().borrow_mut().pos = b + ret;
                }
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
                self.data(kind, inst, tid, p, offset, length, ret.max(0))?;
                if let (Some(ex), Some(f)) = (self.export.as_mut(), &f) {
                    if let (Some(oid), true) = (f.borrow().oid, offset >= 0) {
                        if base.is_some() {
                            ex.pin(oid);
                            ex.note("append_positions_resolved", 1);
                        }
                        ex.data(when, KINDS[kind], oid, offset, length, positioned, ret, err);
                    }
                }
            }
            return Ok(());
        }
        if name == "mmap" {
            // exported only (the metrics do not count a mapping): the mapped range as one read,
            // since the faults inside it are not in the trace
            if self.export.is_none() || a.len() < 6 || ret < 0 {
                return Ok(());
            }
            let (fd, path) = fd_arg(&a[4]);
            let f = self.file(tid, fd, path.as_deref());
            let Some(f) = f else { return Ok(()) };
            let Some(oid) = f.borrow().oid else { return Ok(()) };
            let fp = path.clone().or_else(|| f.borrow().path.clone());
            if inst.is_none() || self.under(fp.as_deref()).is_none() {
                return Ok(());
            }
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
            let mut got: Vec<(String, usize, String, i64, i64)> = Vec::new();
            for cb in &cbs {
                let path = cb.path.as_deref().map(fd_path).filter(|p| !p.is_empty());
                let f = self.file(tid, Some(cb.fd), path.as_deref());
                let fp = path.clone().or_else(|| f.as_ref().and_then(|f| f.borrow().path.clone()));
                let Some(p) = self.under(fp.as_deref()) else { continue };
                if cb.op != "PREAD" && cb.op != "PWRITE" {
                    self.note("aio_requests_not_counted", 1);
                    continue;
                }
                got.push((cb.data.clone(), if cb.op == "PREAD" { R } else { W }, p, cb.offset, cb.nbytes));
            }
            let Some(inst) = inst else { return Ok(()) };
            if got.is_empty() || ret <= 0 {
                return Ok(());
            }
            got.truncate(ret as usize);
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
            let ctx = a[0].clone();
            let mut members: Option<Vec<(usize, usize)>> = None;
            if self.export.is_some() {
                let mut mem = Vec::new();
                for cb in &cbs {
                    let path = cb.path.as_deref().map(fd_path).filter(|p| !p.is_empty());
                    let f = self.file(tid, Some(cb.fd), path.as_deref());
                    if let Some(f) = f {
                        let fp = path.clone().or_else(|| f.borrow().path.clone());
                        let oid = f.borrow().oid;
                        if let (true, Some(oid)) = (cb.op == "PREAD" || cb.op == "PWRITE", oid) {
                            if self.under(fp.as_deref()).is_some() {
                                mem.push((if cb.op == "PREAD" { "read" } else { "write" }, oid, cb.offset, cb.nbytes));
                            }
                        }
                    }
                }
                mem.truncate(ret as usize);
                if !mem.is_empty() {
                    members = Some(self.export.as_mut().unwrap().submit(when, &mem));
                }
            }
            for (j, (data, kind, p, off, nbytes)) in got.into_iter().enumerate() {
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
                    .push(Waiting { kind, inst, tid, path: p, off, nbytes, member });
            }
            return Ok(());
        }
        if name == "io_getevents" || name == "io_pgetevents" {
            if let Some(c) = self.chain.get_mut(&tid) {
                c.1 = done;
            }
            let ctx = a.first().cloned();
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
                self.data(w.kind, w.inst, w.tid, &w.path, w.off, w.nbytes, res.max(0))?;
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
            if let (0, Some(p)) = (ret, &p) {
                if self.extent.contains_key(p) {
                    let v = match name {
                        "ftruncate" => Some(a.get(1).and_then(|x| int0(x))),
                        "fallocate" if a.get(1).map(String::as_str) == Some("0") => {
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
                if let (Some(ex), Some(f)) = (self.export.as_mut(), &f) {
                    if let Some(oid) = f.borrow().oid {
                        ex.fd_op(when, op, name, oid, a, ret, err);
                    }
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
                    if let (Some(ex), Some(f)) = (self.export.as_mut(), &f) {
                        if let Some(oid) = f.borrow().oid {
                            ex.op(when, "fstat", oid, err, Value::Null, vec![]);
                        }
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
            if let (0, Some(p)) = (ret, &p) {
                self.known_size(tid, name, kind, p, a, di, pi);
            }
            if let (Some(_), Some(p)) = (inst, &p) {
                self.count(tid, kind);
                if self.export.is_some() {
                    let mut to = None;
                    if kind == "rename" {
                        let j = if di.is_none() { pi + 1 } else { pi + 2 };
                        if a.len() > j {
                            let d = if di.is_some() { Some(a[j - 1].clone()) } else { None };
                            let r = self.resolve(tid, d.as_deref(), c_string(&a[j]));
                            to = self.under(r.as_deref());
                        }
                    }
                    self.export.as_mut().unwrap().path_op(when, kind, name, p, a, pi, to.as_deref(), err)?;
                }
            }
            return Ok(());
        }
        Ok(())
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
            let j = if di.is_none() { pi + 1 } else { pi + 2 };
            let to = if a.len() > j {
                let d = if di.is_some() { Some(a[j - 1].clone()) } else { None };
                let r = self.resolve(tid, d.as_deref(), c_string(&a[j]));
                self.under(r.as_deref())
            } else {
                None
            };
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
        // requests whose results the trace does not show (reaped from the ring in user space,
        // or `io_getevents` not traced): taken as complete
        let mut keys: Vec<(u64, (String, String))> = self.aio.iter().map(|(k, v)| (v.0, k.clone())).collect();
        keys.sort();
        for (_, k) in keys {
            for w in self.aio.remove(&k).unwrap().1 {
                self.note("aio_results_assumed_complete", 1);
                self.data(w.kind, w.inst, w.tid, &w.path, w.off, w.nbytes, w.nbytes)?;
            }
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
    let m = t.metrics.as_ref().unwrap();
    let mut notes: serde_json::Map<String, Value> = t.notes.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    if t.chain_gap.is_none() && !m.m.fan_out.is_empty() {
        notes.insert("depth_not_computed".into(), json!("give --chain-gap-us: the think time that ends a chain of io_submit rounds"));
    }
    if ["aio_requests_truncated", "aio_events_truncated", "iov_arrays_truncated"].iter().any(|k| notes.contains_key(*k)) {
        notes.insert("arrays_truncated".into(), json!("trace again with strace -v: without it strace prints only the first 32 elements of an array"));
    }
    let instances = m.instances.len().max(usize::from(o.instance_roots.is_empty()));
    let doc = json!({
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
    Ok((doc, t.seen))
}

/// Why a trace gave no call to count: the roots matched none of its calls, or it is no strace.
pub fn nothing_counted(roots: &[String], seen: Seen) -> String {
    let why = if seen.calls > 0 {
        "check --root against the paths the trace shows"
    } else {
        "not an strace (strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,... -o FILE)"
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
        let s = "[{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/a, b>, aio_buf=0x1, aio_nbytes=4096, aio_offset=0}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_FSYNC, aio_fildes=4, aio_nbytes=7, aio_offset=-1}]";
        let cbs = iocbs(s);
        assert_eq!(cbs.len(), 2);
        assert_eq!((cbs[0].fd, cbs[0].path.as_deref(), cbs[0].nbytes, cbs[0].offset), (3, Some("/d/a, b"), 4096, 0));
        assert_eq!((cbs[1].op.as_str(), cbs[1].path.as_deref(), cbs[1].offset), ("FSYNC", None, -1));
        assert!(iocbs("aio_data=, x").is_empty());
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
