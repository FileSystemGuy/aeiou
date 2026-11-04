//! The eBPF source's records (`syscalls.bin`, `ebpf.rs`) as the strace text every reader of the
//! suite reads (`strace -f -ttt -T -yy`, the capture's line; DESIGN_REVIEW.md §3.96 "Phase 3"):
//! the records put back in time order, each call printed as strace prints it, a whole line when
//! nothing came between its entry and its return and `<unfinished ...>` with `<... resumed>`
//! when something did, its descriptors annotated with their paths (`3</p>`) from the opens,
//! `dup`s, forks, `chdir`s, and `exec`s recorded before it, `+++` lines at each thread's end,
//! and a `--- aeiou: ... lost ---` line where the ring buffer lost records (a line every reader
//! skips). What the parser reads is printed as strace does; what it ignores (a `stat`'s
//! structure, a read's bytes, `execve`'s arguments and environment) as the address strace
//! prints when it cannot read one.
use crate::ebpf::{calls, record::*, Call, MAGIC};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::rc::Rc;

/// Records are put back in time order within this many: the ring buffer takes them in the
/// order their CPUs reserved space, which a record stamped a moment earlier on another CPU can
/// lose.
const WINDOW: usize = 1 << 16;
const AT_FDCWD: i64 = -100;

/// One record, decoded.
#[derive(Debug, Clone)]
enum Rec {
    Enter { h: Head, nr: u32, args: [u64; 6], blobs: Vec<(u8, u8, Vec<u8>)> },
    Exit { h: Head, nr: u32, ret: i64, blobs: Vec<(u8, u8, Vec<u8>)> },
    Gap { h: Head, lost: u64, first: u64, last: u64 },
    Gone { h: Head, code: i64 },
    Fork { h: Head, child: u32, thread: bool },
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn head_of(b: &[u8]) -> Head {
    Head { kind: u16_at(b, 0), len: u16_at(b, 2), cpu: u32_at(b, 4), ts: u64_at(b, 8), pid: u32_at(b, 16), tid: u32_at(b, 20) }
}

fn blobs_of(b: &[u8], mut at: usize) -> Result<Vec<(u8, u8, Vec<u8>)>> {
    let mut out = Vec::new();
    while at + 8 <= b.len() {
        let (arg, what, len) = (b[at], b[at + 1], u16_at(b, at + 2) as usize);
        let data = b.get(at + 8..at + 8 + len).ok_or_else(|| anyhow!("a blob past its record's end"))?;
        out.push((arg, what, data.to_vec()));
        at += 8 + ((len + 7) & !7);
    }
    Ok(out)
}

/// One record of `syscalls.bin`, its whole bytes.
fn decode(b: &[u8]) -> Result<Rec> {
    let h = head_of(b);
    Ok(match h.kind {
        ENTER => {
            let mut args = [0u64; 6];
            for (i, a) in args.iter_mut().enumerate() {
                *a = u64_at(b, 32 + 8 * i);
            }
            Rec::Enter { h, nr: u32_at(b, 24), args, blobs: blobs_of(b, std::mem::size_of::<Enter>())? }
        }
        EXIT => Rec::Exit { h, nr: u32_at(b, 24), ret: u64_at(b, 32) as i64, blobs: blobs_of(b, std::mem::size_of::<Exit>())? },
        GAP => Rec::Gap { h, lost: u64_at(b, 24), first: u64_at(b, 32), last: u64_at(b, 40) },
        GONE => Rec::Gone { h, code: u64_at(b, 32) as i64 },
        FORK => Rec::Fork { h, child: u32_at(b, 24), thread: u32_at(b, 28) == 1 },
        k => bail!("a record of kind {k}"),
    })
}

/// A descriptor: the path it names (`None` for a socket, a pipe, or what the trace does not
/// show), and whether an `exec` closes it.
#[derive(Debug, Clone)]
struct Fd {
    path: Option<String>,
    cloexec: bool,
}
type Files = Rc<RefCell<HashMap<i64, Fd>>>;
type Cwd = Rc<RefCell<String>>;

#[derive(Debug, Clone)]
struct Task {
    files: Files,
    cwd: Cwd,
}

/// A call between its entry and its return.
struct Pending {
    nr: u32,
    ts: u64,
    args: [u64; 6],
    blobs: Vec<(u8, u8, Vec<u8>)>,
    /// Printed and cut short (`<unfinished ...>`).
    unfinished: bool,
    silent: bool,
    /// The entry's text, `TID TIME NAME(ARGS`, a string the probe could not read yet a
    /// placeholder (`hole`) filled when the line is written.
    line: String,
}

/// Where a string argument the entry could not read goes in its line, until the return reads it.
fn hole(arg: usize) -> String {
    format!("\u{0}{arg}\u{0}")
}

/// A line's placeholders filled with the strings as they are now.
fn fill(line: &str, args: &[u64; 6], blobs: &[(u8, u8, Vec<u8>)]) -> String {
    if !line.contains('\u{0}') {
        return line.to_string();
    }
    let mut out = line.to_string();
    for i in 0..6 {
        let h = hole(i);
        if out.contains(&h) {
            out = out.replace(&h, &string(i as u8, args[i], blobs));
        }
    }
    out
}

/// What a conversion did, for `capture.json`.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub records: u64,
    pub calls: u64,
    /// The losses' windows: CPU, records lost, first and last on the monotonic clock.
    pub windows: Vec<(u32, u64, u64, u64)>,
    pub paths_not_copied: u64,
    pub exits_without_entry: u64,
    pub entries_without_exit: u64,
    pub descriptors_unresolved: u64,
    pub vectored_aio_left_out: u64,
}

impl Stats {
    pub fn json(&self) -> Value {
        json!({
            "records": self.records,
            "calls": self.calls,
            "paths_not_copied": self.paths_not_copied,
            "exits_without_entry": self.exits_without_entry,
            "entries_without_exit": self.entries_without_exit,
            "descriptors_unresolved": self.descriptors_unresolved,
            "vectored_aio_left_out": self.vectored_aio_left_out,
        })
    }
}

/// `syscalls.bin`'s header line: the architecture, the clocks' offset at the start, the
/// process the capture started, its working directory, and the descriptors it inherited.
pub fn header(bin: &Path) -> Result<Value> {
    let mut r = BufReader::new(std::fs::File::open(bin).with_context(|| bin.display().to_string())?);
    read_header(&mut r, bin)
}

fn read_header(r: &mut impl BufRead, bin: &Path) -> Result<Value> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).with_context(|| format!("{}: shorter than its header", bin.display()))?;
    if &magic != MAGIC {
        bail!("{}: not the eBPF source's records (it begins {:?}, not {:?})", bin.display(), String::from_utf8_lossy(&magic), String::from_utf8_lossy(MAGIC));
    }
    let mut line = String::new();
    r.read_line(&mut line)?;
    serde_json::from_str(&line).with_context(|| format!("{}: its header line", bin.display()))
}

/// `bin` (`syscalls.bin`) converted to `out` (`trace.txt`).
pub fn convert(bin: &Path, out: &Path) -> Result<Stats> {
    let mut r = BufReader::with_capacity(8 << 20, std::fs::File::open(bin).with_context(|| bin.display().to_string())?);
    let hdr = read_header(&mut r, bin)?;
    if hdr["arch"].as_str() != Some(crate::ebpf::ARCH) {
        bail!("{}: recorded on {}, and this aeiou-trace reads {} records (convert it on a host of its architecture)", bin.display(), hdr["arch"], crate::ebpf::ARCH);
    }
    let w = BufWriter::with_capacity(8 << 20, std::fs::File::create(out).with_context(|| out.display().to_string())?);
    let mut t = Text::new(&hdr, w);
    let mut heap: BinaryHeap<Reverse<(u64, u64, Vec<u8>)>> = BinaryHeap::new();
    let mut seq = 0u64;
    let mut head = [0u8; 24];
    loop {
        match r.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let h = head_of(&head);
        if (h.len as usize) < 24 {
            bail!("{}: a record of {} bytes, after {} records", bin.display(), h.len, seq);
        }
        let mut b = head.to_vec();
        b.resize(h.len as usize, 0);
        r.read_exact(&mut b[24..]).with_context(|| format!("{}: cut short in a record", bin.display()))?;
        let ts = if h.kind == GAP { u64_at(&b, 32) } else { h.ts };
        heap.push(Reverse((ts, seq, b)));
        seq += 1;
        if heap.len() > WINDOW {
            let Reverse((_, _, b)) = heap.pop().unwrap();
            t.record(decode(&b)?)?;
        }
    }
    while let Some(Reverse((_, _, b))) = heap.pop() {
        t.record(decode(&b)?)?;
    }
    t.end()?;
    let mut w = t.w;
    w.flush()?;
    Ok(t.stats)
}

/// The losses no `GAP` record reported (after each CPU's last record), as `GAP` records to
/// append to `syscalls.bin`.
pub fn trailing_gaps(losses: &[Loss]) -> Vec<u8> {
    let mut out = Vec::new();
    for (cpu, l) in losses.iter().enumerate() {
        if l.pending == 0 {
            continue;
        }
        let g = Gap { head: Head { kind: GAP, len: std::mem::size_of::<Gap>() as u16, cpu: cpu as u32, ts: l.last, pid: 0, tid: 0 }, lost: l.pending, first: l.first, last: l.last };
        let b: [u8; std::mem::size_of::<Gap>()] = unsafe { std::mem::transmute(g) };
        out.extend_from_slice(&b);
    }
    out
}

struct Text<W: Write> {
    w: W,
    offset: f64,
    seed: u32,
    execd: bool,
    tasks: HashMap<u32, Task>,
    pending: HashMap<u32, Pending>,
    /// The line printed last, not yet ended: its thread.
    open: Option<u32>,
    stats: Stats,
    real: HashMap<String, String>,
    /// Threads whose end the trace showed, so a later record of theirs is a new thread's.
    gone: HashSet<u32>,
}

impl<W: Write> Text<W> {
    fn new(hdr: &Value, w: W) -> Text<W> {
        let cwd = hdr["cwd"].as_str().unwrap_or("/").to_string();
        let mut files = HashMap::new();
        for d in hdr["inherited"].as_array().cloned().unwrap_or_default() {
            let (Some(fd), Some(target)) = (d["fd"].as_i64(), d["target"].as_str()) else { continue };
            // a pipe's or socket's name too (`pipe:[123]`), as -yy prints it
            files.insert(fd, Fd { path: (!target.is_empty()).then(|| target.to_string()), cloexec: false });
        }
        let seed = hdr["seed"].as_u64().unwrap_or(0) as u32;
        let mut tasks = HashMap::new();
        tasks.insert(seed, Task { files: Rc::new(RefCell::new(files)), cwd: Rc::new(RefCell::new(cwd)) });
        Text {
            w,
            offset: hdr["realtime_minus_monotonic"].as_f64().unwrap_or(0.0),
            seed,
            execd: false,
            tasks,
            pending: HashMap::new(),
            open: None,
            stats: Stats::default(),
            real: HashMap::new(),
            gone: HashSet::new(),
        }
    }

    fn when(&self, ns: u64) -> String {
        format!("{:.6}", ns as f64 / 1e9 + self.offset)
    }

    fn task(&mut self, tid: u32) -> Task {
        if let Some(t) = self.tasks.get(&tid) {
            return t.clone();
        }
        // a thread the trace showed no fork of (its record lost): what it holds is unknown
        let t = Task { files: Rc::new(RefCell::new(HashMap::new())), cwd: Rc::new(RefCell::new("/".into())) };
        self.tasks.insert(tid, t.clone());
        t
    }

    /// The line left open written and ended as strace ends one another thread interrupts.
    fn cut(&mut self) -> Result<()> {
        if let Some(tid) = self.open.take() {
            if let Some(p) = self.pending.get_mut(&tid) {
                p.unfinished = true;
                let l = fill(&p.line, &p.args, &p.blobs);
                self.w.write_all(l.as_bytes())?;
                self.w.write_all(b" <unfinished ...>\n")?;
            }
        }
        Ok(())
    }

    fn record(&mut self, r: Rec) -> Result<()> {
        self.stats.records += 1;
        match r {
            Rec::Enter { h, nr, args, blobs } => self.enter(h, nr, args, blobs),
            Rec::Exit { h, nr, ret, blobs } => self.exit(h, nr, ret, blobs),
            Rec::Gap { h, lost, first, last } => {
                self.cut()?;
                self.stats.windows.push((h.cpu, lost, first, last));
                let line = format!("{} {} --- aeiou: {lost} event(s) lost on cpu {} from {} to {}: the ring buffer was full ---\n", h.tid, self.when(first), h.cpu, self.when(first), self.when(last));
                self.w.write_all(line.as_bytes())?;
                Ok(())
            }
            Rec::Gone { h, code } => {
                if self.silent(h.tid) {
                    return Ok(());
                }
                if self.open == Some(h.tid) {
                    // the thread's last call never returns: a whole line, as strace prints exit_group's
                    self.open = None;
                    if let Some(p) = self.pending.remove(&h.tid) {
                        self.w.write_all(format!("{}) = ?\n", fill(&p.line, &p.args, &p.blobs)).as_bytes())?;
                    }
                }
                self.cut()?;
                if let Some(p) = self.pending.remove(&h.tid) {
                    if !p.silent {
                        let name = calls().get(p.nr as usize).copied().flatten().map_or("?", |c| c.name);
                        let line = if p.unfinished { format!("{} {} <... {name} resumed>) = ?\n", h.tid, self.when(h.ts)) } else { format!("{} {} {name}() = ?\n", h.tid, self.when(p.ts)) };
                        self.w.write_all(line.as_bytes())?;
                    }
                }
                let code = code as i32;
                let how = if code & 0x7f == 0 {
                    format!("exited with {}", (code >> 8) & 0xff)
                } else {
                    format!("killed by {}{}", signal(code & 0x7f), if code & 0x80 != 0 { " (core dumped)" } else { "" })
                };
                self.w.write_all(format!("{} {} +++ {how} +++\n", h.tid, self.when(h.ts)).as_bytes())?;
                self.gone.insert(h.tid);
                Ok(())
            }
            Rec::Fork { h, child, thread } => {
                let parent = self.task(h.tid);
                let flags = self.pending.get(&h.tid).map_or(libc::SIGCHLD as u64, |p| clone_flags(p.nr, &p.args, &p.blobs));
                let files = if flags & libc::CLONE_FILES as u64 != 0 { parent.files.clone() } else { Rc::new(RefCell::new(parent.files.borrow().clone())) };
                let cwd = if flags & libc::CLONE_FS as u64 != 0 { parent.cwd.clone() } else { Rc::new(RefCell::new(parent.cwd.borrow().clone())) };
                let _ = thread;
                self.gone.remove(&child);
                self.tasks.insert(child, Task { files, cwd });
                Ok(())
            }
        }
    }

    /// Whether a thread's records are left out: the process the capture started, until it
    /// `exec`s the command (what came before is the capture's own, as under strace).
    fn silent(&self, tid: u32) -> bool {
        !self.execd && tid == self.seed
    }

    fn call(&self, nr: u32) -> Option<Call> {
        calls().get(nr as usize).copied().flatten()
    }

    fn enter(&mut self, h: Head, nr: u32, args: [u64; 6], blobs: Vec<(u8, u8, Vec<u8>)>) -> Result<()> {
        let Some(c) = self.call(nr) else { return Ok(()) };
        if self.silent(h.tid) && (c.name == "execve" || c.name == "execveat") {
            self.execd = true;
        }
        let silent = self.silent(h.tid);
        self.stats.paths_not_copied += blobs.iter().filter(|b| b.1 == STR_FAULT).count() as u64;
        let mut line = String::new();
        if !silent {
            self.cut()?;
            self.stats.calls += 1;
            let task = self.task(h.tid);
            let (head, _) = self.args(&c, &args, &blobs, &task, None);
            line = format!("{} {} {}({head}", h.tid, self.when(h.ts), c.name);
            self.open = Some(h.tid);
        }
        if c.name == "close" {
            // the kernel frees the descriptor as the close begins
            let task = self.task(h.tid);
            task.files.borrow_mut().remove(&(args[0] as i32 as i64));
        }
        if let Some(old) = self.pending.insert(h.tid, Pending { nr, ts: h.ts, args, blobs, unfinished: false, silent, line }) {
            // an entry whose exit was lost
            if !old.silent {
                self.stats.entries_without_exit += 1;
            }
        }
        Ok(())
    }

    fn exit(&mut self, h: Head, nr: u32, ret: i64, xblobs: Vec<(u8, u8, Vec<u8>)>) -> Result<()> {
        let Some(c) = self.call(nr) else { return Ok(()) };
        let Some(mut p) = self.pending.get(&h.tid).is_some_and(|p| p.nr == nr).then(|| self.pending.remove(&h.tid)).flatten() else {
            if !self.silent(h.tid) && !(ret == 0 && matches!(c.name, "clone" | "clone3" | "fork" | "vfork")) {
                // a child's own return from its clone is not a call strace prints
                self.stats.exits_without_entry += 1;
            }
            return Ok(());
        };
        // a string the entry could not read, read again at the exit
        for (arg, what, data) in &xblobs {
            if (*what == STR || *what == STR_CUT) && (*arg as usize) < 6 {
                if let Some(b) = p.blobs.iter_mut().find(|b| b.0 == *arg && b.1 == STR_FAULT) {
                    *b = (*arg, *what, data.clone());
                    self.stats.paths_not_copied -= 1;
                }
            }
        }
        let task = self.task(h.tid);
        let ret_path = self.apply(h.pid, h.tid, &c, &p, ret, &task);
        if !p.silent {
            let (_, tail) = self.args(&c, &p.args, &p.blobs, &task, Some((ret, &xblobs)));
            let r = self.ret(&c, ret, ret_path);
            let dur = format!("<{:.6}>", (h.ts.saturating_sub(p.ts)) as f64 / 1e9);
            if self.open == Some(h.tid) && !p.unfinished {
                self.open = None;
                self.w.write_all(format!("{}{tail}) = {r} {dur}\n", fill(&p.line, &p.args, &p.blobs)).as_bytes())?;
            } else {
                self.cut()?;
                p.unfinished = true;
                self.w.write_all(format!("{} {} <... {} resumed>{tail}) = {r} {dur}\n", h.tid, self.when(h.ts), c.name).as_bytes())?;
            }
        }
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        self.cut()?;
        let mut left: Vec<(u32, Pending)> = self.pending.drain().collect();
        left.sort_by_key(|(t, p)| (p.ts, *t));
        for (tid, p) in left {
            if p.silent {
                continue;
            }
            self.stats.entries_without_exit += 1;
            let name = self.call(p.nr).map_or("?", |c| c.name);
            if p.unfinished {
                self.w.write_all(format!("{tid} {} <... {name} resumed>) = ?\n", self.when(p.ts)).as_bytes())?;
            }
        }
        Ok(())
    }

    /// A descriptor as `-yy` prints it.
    fn fd(&mut self, task: &Task, raw: u64) -> String {
        let fd = raw as i32 as i64;
        if fd < 0 {
            return fd.to_string();
        }
        match task.files.borrow().get(&fd) {
            Some(Fd { path: Some(p), .. }) => format!("{fd}<{}>", annotation(p)),
            Some(_) => fd.to_string(),
            None => {
                self.stats.descriptors_unresolved += 1;
                fd.to_string()
            }
        }
    }

    fn dirfd(&mut self, task: &Task, raw: u64) -> String {
        if raw as i32 as i64 == AT_FDCWD {
            return format!("AT_FDCWD<{}>", annotation(&task.cwd.borrow()));
        }
        self.fd(task, raw)
    }

    /// The arguments strace prints before the call returns, and those it prints at its return
    /// (`e`, and what follows it); `exit` is `Some` at the return.
    fn args(&mut self, c: &Call, a: &[u64; 6], blobs: &[(u8, u8, Vec<u8>)], task: &Task, exit: Option<(i64, &[(u8, u8, Vec<u8>)])>) -> (String, String) {
        if c.name == "clone" {
            #[cfg(target_arch = "x86_64")]
            let (ptid, ctid, tls) = (a[2], a[3], a[4]);
            #[cfg(target_arch = "aarch64")]
            let (ptid, tls, ctid) = (a[2], a[3], a[4]);
            return (format!("child_stack={}, flags={}, parent_tid={}, child_tidptr={}, tls={}", ptr(a[1]), clone_names(a[0]), ptr(ptid), ptr(ctid), ptr(tls)), String::new());
        }
        let mut head = Vec::new();
        let mut tail = Vec::new();
        let mut at_exit = false;
        let kinds: Vec<char> = c.kinds.chars().collect();
        for (raw, &k) in kinds.iter().enumerate() {
            if k == '_' {
                continue;
            }
            if k == 'e' {
                at_exit = true;
            }
            // at the return only what strace prints there; the rest was printed at the entry
            if exit.is_some() != at_exit {
                continue;
            }
            let v = a[raw];
            if k == 'm' && matches!(c.name, "open" | "openat") && a[raw - 1] & (libc::O_CREAT as u64 | 0o20000000) == 0 {
                // strace prints the mode only when the open can create
                continue;
            }
            let s = match k {
                'd' => self.fd(task, v),
                // a descriptor the call makes: its path if it was open, never one unresolved
                'n' => match task.files.borrow().get(&(v as i32 as i64)) {
                    Some(Fd { path: Some(p), .. }) => format!("{}<{}>", v as i32, annotation(p)),
                    _ => (v as i32).to_string(),
                },
                'D' => self.dirfd(task, v),
                'p' | 's' if exit.is_none() && blobs.iter().any(|b| b.0 == raw as u8 && b.1 == STR_FAULT) => hole(raw),
                'p' | 's' => string(raw as u8, v, blobs),
                'i' => (v as i64).to_string(),
                'u' => v.to_string(),
                'x' => ptr(v),
                'o' => open_flags(v),
                'm' => mode(v),
                'w' => whence(v),
                'C' => self.clone_args(blobs, v),
                'F' => fcntl_cmd(v),
                'a' => bits(v, &AT_FLAGS),
                'U' => bits(v, &[(0x200, "AT_REMOVEDIR")]),
                'f' => bits(v, &FALLOC),
                'r' => bits(v, &RWF),
                'v' => iovs(blobs, v, a.get(raw + 1).copied().unwrap_or(0)),
                'A' => advice(v),
                'I' => ioctl(v),
                'P' => prot(v),
                'M' => map_flags(v),
                'b' => self.iocbs(blobs, task, v),
                'e' => match exit {
                    Some((ret, xb)) => events(xb, v, ret),
                    None => String::new(),
                },
                'H' => match blob(blobs, 2) {
                    Some(b) if b.len() >= 24 => format!("{{flags={}, mode={:#o}, resolve={:#x}}}", open_flags(u64_at(b, 0)), u64_at(b, 8), u64_at(b, 16)),
                    _ => ptr(v),
                },
                'c' => clone_names(v),
                _ => ptr(v),
            };
            if at_exit { tail.push(s) } else { head.push(s) }
        }
        let mut h = head.join(", ");
        if kinds.contains(&'e') && !h.is_empty() {
            // strace ends the entry's text before what it prints at the return
            h += ", ";
        }
        (h, if exit.is_some() { tail.join(", ") } else { String::new() })
    }

    fn clone_args(&self, blobs: &[(u8, u8, Vec<u8>)], v: u64) -> String {
        match blob(blobs, 0) {
            Some(b) if b.len() >= 64 => format!(
                "{{flags={}, child_tid={}, parent_tid={}, exit_signal={}, stack={}, stack_size={:#x}, tls={}}}",
                clone_names(u64_at(b, 0) & !0xff),
                ptr(u64_at(b, 16)),
                ptr(u64_at(b, 24)),
                match u64_at(b, 32) { 0 => "0".to_string(), s => signal(s as i32) },
                ptr(u64_at(b, 40)),
                u64_at(b, 48),
                ptr(u64_at(b, 56))
            ),
            _ => ptr(v),
        }
    }

    fn iocbs(&mut self, blobs: &[(u8, u8, Vec<u8>)], task: &Task, v: u64) -> String {
        let Some(b) = blob(blobs, 2) else { return ptr(v) };
        let mut out = Vec::new();
        for e in b.chunks_exact(IOCB_SIZE) {
            let (data, rw, op, fd, buf, n, off) = (u64_at(e, 0), u32_at(e, 12), u16_at(e, 16), u32_at(e, 20), u64_at(e, 24), u64_at(e, 32), u64_at(e, 40) as i64);
            let name = match op {
                0 => "PREAD",
                1 => "PWRITE",
                2 => "FSYNC",
                3 => "FDSYNC",
                5 => "POLL",
                6 => "NOOP",
                7 | 8 => {
                    // the iovec array inside a request is not copied: left out, and counted
                    self.stats.vectored_aio_left_out += 1;
                    continue;
                }
                _ => "",
            };
            let mut s = format!("{{aio_data={data:#x}");
            if rw != 0 {
                s += &format!(", aio_rw_flags={}", bits(rw as u64, &RWF));
            }
            s += &format!(", aio_lio_opcode={}, aio_fildes={}", if name.is_empty() { op.to_string() } else { format!("IOCB_CMD_{name}") }, self.fd(task, fd as u64));
            match op {
                0 | 1 => s += &format!(", aio_buf={}, aio_nbytes={n}, aio_offset={off}", ptr(buf)),
                5 => s += &format!(", aio_buf={}", ptr(buf)),
                _ => {}
            }
            out.push(s + "}");
        }
        format!("[{}]", out.join(", "))
    }

    /// The state a call's return changes; the path of the descriptor it returned.
    fn apply(&mut self, pid: u32, tid: u32, c: &Call, p: &Pending, ret: i64, task: &Task) -> Option<String> {
        if ret < 0 {
            return None;
        }
        let a = &p.args;
        let fd_path = |t: &Task, fd: u64| t.files.borrow().get(&(fd as i32 as i64)).and_then(|f| f.path.clone());
        match c.name {
            "open" | "creat" | "openat" | "openat2" => {
                let (dirfd, path, flags) = match c.name {
                    "open" => (AT_FDCWD as u64, str_of(&p.blobs, 0), a[1]),
                    "creat" => (AT_FDCWD as u64, str_of(&p.blobs, 0), (libc::O_CREAT | libc::O_WRONLY | libc::O_TRUNC) as u64),
                    "openat" => (a[0], str_of(&p.blobs, 1), a[2]),
                    _ => (a[0], str_of(&p.blobs, 1), blob(&p.blobs, 2).filter(|b| b.len() >= 8).map_or(0, |b| u64_at(b, 0))),
                };
                let path = path.and_then(|s| self.resolve(pid, tid, task, dirfd, &s));
                task.files.borrow_mut().insert(ret, Fd { path: path.clone(), cloexec: flags & libc::O_CLOEXEC as u64 != 0 });
                path
            }
            "dup" | "dup2" | "dup3" => {
                let path = fd_path(task, a[0]);
                let cloexec = c.name == "dup3" && a[2] & libc::O_CLOEXEC as u64 != 0;
                task.files.borrow_mut().insert(ret, Fd { path: path.clone(), cloexec });
                path
            }
            "fcntl" => {
                let cmd = a[1] as i32;
                if cmd == libc::F_DUPFD || cmd == libc::F_DUPFD_CLOEXEC {
                    let path = fd_path(task, a[0]);
                    task.files.borrow_mut().insert(ret, Fd { path: path.clone(), cloexec: cmd == libc::F_DUPFD_CLOEXEC });
                    return path;
                }
                if cmd == libc::F_SETFD {
                    if let Some(f) = task.files.borrow_mut().get_mut(&(a[0] as i32 as i64)) {
                        f.cloexec = a[2] & libc::FD_CLOEXEC as u64 != 0;
                    }
                }
                None
            }
            "chdir" => {
                if let Some(d) = str_of(&p.blobs, 0).and_then(|s| self.resolve(pid, tid, task, AT_FDCWD as u64, &s)) {
                    *task.cwd.borrow_mut() = d;
                }
                None
            }
            "fchdir" => {
                if let Some(d) = fd_path(task, a[0]) {
                    *task.cwd.borrow_mut() = d;
                }
                None
            }
            "execve" | "execveat" => {
                // the table is the process's own from here, without what closes on exec
                let kept: HashMap<i64, Fd> = task.files.borrow().iter().filter(|(_, f)| !f.cloexec).map(|(k, v)| (*k, v.clone())).collect();
                let t = Task { files: Rc::new(RefCell::new(kept)), cwd: Rc::new(RefCell::new(task.cwd.borrow().clone())) };
                // every thread of the process is gone but this one
                self.tasks.insert(tid, t);
                None
            }
            "socket" | "socketpair" | "pipe" | "pipe2" | "eventfd" | "eventfd2" | "epoll_create" | "epoll_create1" | "accept" | "accept4" | "memfd_create" | "timerfd_create" | "signalfd" | "signalfd4" | "inotify_init" | "inotify_init1" | "io_uring_setup" | "userfaultfd" | "perf_event_open" | "fanotify_init" | "pidfd_open" | "pidfd_getfd" | "bpf" | "memfd_secret" | "fsopen" | "fsmount" | "fspick" | "open_tree" | "landlock_create_ruleset" | "mq_open" => {
                // a descriptor that names no path (the pairs' second is not seen)
                if !matches!(c.name, "socketpair" | "pipe" | "pipe2" | "bpf") {
                    task.files.borrow_mut().insert(ret, Fd { path: None, cloexec: false });
                }
                None
            }
            _ => None,
        }
    }

    /// The path an open names, as `-yy` would print it: absolute, its directories' symbolic
    /// links resolved as the kernel resolves them (looked up now, on this host; the name's
    /// own last part kept), `.` and `..` gone; `/proc/self` the traced process's own, and
    /// nothing under `/proc` looked up (it is the converter's own there).
    fn resolve(&mut self, pid: u32, tid: u32, task: &Task, dirfd: u64, path: &str) -> Option<String> {
        let base = if path.starts_with('/') {
            String::new()
        } else if dirfd as i32 as i64 == AT_FDCWD {
            task.cwd.borrow().clone()
        } else {
            task.files.borrow().get(&(dirfd as i32 as i64))?.path.clone()?
        };
        let joined = crate::strace::normpath(&if base.is_empty() { path.to_string() } else { format!("{base}/{path}") });
        let joined = proc_self(&joined, pid, tid);
        if joined == "/proc" || joined.starts_with("/proc/") {
            return Some(joined);
        }
        let (dir, name) = match joined.rsplit_once('/') {
            Some(("", n)) => ("/".to_string(), n.to_string()),
            Some((d, n)) => (d.to_string(), n.to_string()),
            None => return Some(joined),
        };
        let real = match self.real.get(&dir) {
            Some(r) => r.clone(),
            None => {
                let r = std::fs::canonicalize(&dir).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| dir.clone());
                self.real.insert(dir.clone(), r.clone());
                r
            }
        };
        Some(if name.is_empty() { real } else if real == "/" { format!("/{name}") } else { format!("{real}/{name}") })
    }

    fn ret(&self, c: &Call, ret: i64, path: Option<String>) -> String {
        if (-516..=-512).contains(&ret) {
            // interrupted, to be restarted: strace prints no return value
            return format!("? {}", errno_name((-ret) as i32));
        }
        if (-4095..0).contains(&ret) {
            let e = (-ret) as i32;
            return format!("-1 {} ({})", errno_name(e), std::io::Error::from_raw_os_error(e).to_string().split(" (os error").next().unwrap_or(""));
        }
        match c.name {
            "mmap" | "mremap" => format!("{ret:#x}"),
            _ => match path {
                Some(p) => format!("{ret}<{}>", annotation(&p)),
                None => ret.to_string(),
            },
        }
    }
}

/// A path inside a descriptor's `<...>`, as `-yy` prints it: a quoted string's escapes without
/// the quotes, `<` and `>` in octal so no raw angle bracket is part of it (`strace::fd_path`).
fn annotation(p: &str) -> String {
    let mut s = String::new();
    for &c in p.as_bytes() {
        match c {
            b'\\' => s += "\\\\",
            b'\n' => s += "\\n",
            b'\t' => s += "\\t",
            b'\r' => s += "\\r",
            b'<' | b'>' => s += &format!("\\{c:o}"),
            0x20..=0x7e => s.push(c as char),
            _ => s += &format!("\\{c:o}"),
        }
    }
    s
}

/// `/proc/self` and `/proc/thread-self` as the traced thread's own.
fn proc_self(path: &str, pid: u32, tid: u32) -> String {
    for (link, real) in [("/proc/thread-self", format!("/proc/{pid}/task/{tid}")), ("/proc/self", format!("/proc/{pid}"))] {
        if path == link {
            return real;
        }
        if let Some(rest) = path.strip_prefix(link).filter(|r| r.starts_with('/')) {
            return format!("{real}{rest}");
        }
    }
    path.to_string()
}

fn blob(blobs: &[(u8, u8, Vec<u8>)], arg: u8) -> Option<&[u8]> {
    blobs.iter().find(|b| b.0 == arg && b.1 == BYTES).map(|b| b.2.as_slice())
}

fn str_of(blobs: &[(u8, u8, Vec<u8>)], arg: u8) -> Option<String> {
    blobs.iter().find(|b| b.0 == arg && (b.1 == STR || b.1 == STR_CUT)).map(|b| String::from_utf8_lossy(&b.2).into_owned())
}

/// The flags a call that makes a task passes (`clone`, `clone3`, `fork`, `vfork`).
fn clone_flags(nr: u32, a: &[u64; 6], blobs: &[(u8, u8, Vec<u8>)]) -> u64 {
    match calls().get(nr as usize).copied().flatten().map(|c| c.name) {
        Some("clone") => a[0],
        Some("clone3") => blob(blobs, 0).filter(|b| b.len() >= 8).map_or(0, |b| u64_at(b, 0)),
        Some("vfork") => (libc::CLONE_VM | libc::CLONE_VFORK) as u64,
        _ => 0,
    }
}

fn ptr(v: u64) -> String {
    if v == 0 { "NULL".into() } else { format!("{v:#x}") }
}

/// A string argument as strace prints it: quoted, escaped, `...` after one cut at `PATH_MAX`;
/// the address when the probe could not read it.
fn string(arg: u8, v: u64, blobs: &[(u8, u8, Vec<u8>)]) -> String {
    if v == 0 {
        return "NULL".into();
    }
    match blobs.iter().find(|b| b.0 == arg && b.1 != BYTES && b.1 != BYTES_FAULT) {
        Some((_, w, data)) if *w == STR || *w == STR_CUT => {
            let mut s = String::from("\"");
            for &c in data {
                match c {
                    b'"' => s += "\\\"",
                    b'\\' => s += "\\\\",
                    b'\n' => s += "\\n",
                    b'\t' => s += "\\t",
                    b'\r' => s += "\\r",
                    0x0b => s += "\\v",
                    0x0c => s += "\\f",
                    0x20..=0x7e => s.push(c as char),
                    _ => s += &format!("\\{c:o}"),
                }
            }
            s.push('"');
            if *w == STR_CUT {
                s += "...";
            }
            s
        }
        _ => ptr(v),
    }
}

fn bits(v: u64, names: &[(u64, &str)]) -> String {
    if v == 0 {
        return "0".into();
    }
    let mut left = v;
    let mut out = Vec::new();
    for &(b, n) in names {
        if left & b == b && b != 0 {
            out.push(n.to_string());
            left &= !b;
        }
    }
    if left != 0 {
        out.push(format!("{left:#x}"));
    }
    out.join("|")
}

const AT_FLAGS: [(u64, &str); 6] = [(0x100, "AT_SYMLINK_NOFOLLOW"), (0x200, "AT_EACCESS"), (0x400, "AT_SYMLINK_FOLLOW"), (0x800, "AT_NO_AUTOMOUNT"), (0x1000, "AT_EMPTY_PATH"), (0x8000, "AT_RECURSIVE")];
const FALLOC: [(u64, &str); 6] = [(0x1, "FALLOC_FL_KEEP_SIZE"), (0x2, "FALLOC_FL_PUNCH_HOLE"), (0x8, "FALLOC_FL_COLLAPSE_RANGE"), (0x10, "FALLOC_FL_ZERO_RANGE"), (0x20, "FALLOC_FL_INSERT_RANGE"), (0x40, "FALLOC_FL_UNSHARE_RANGE")];
const RWF: [(u64, &str); 7] = [(0x1, "RWF_HIPRI"), (0x2, "RWF_DSYNC"), (0x4, "RWF_SYNC"), (0x8, "RWF_NOWAIT"), (0x10, "RWF_APPEND"), (0x20, "RWF_NOAPPEND"), (0x40, "RWF_ATOMIC")];

/// The open flags as the kernel defines them on this architecture, in the order strace 6.8
/// prints them (`src/xlat/open_mode_flags.in`), which the export keeps.
fn open_flags(v: u64) -> String {
    #[cfg(target_arch = "x86_64")]
    let (direct, largefile, directory, nofollow) = (0o40000u64, 0o100000u64, 0o200000u64, 0o400000u64);
    #[cfg(target_arch = "aarch64")]
    let (direct, largefile, directory, nofollow) = (0o200000u64, 0o400000u64, 0o40000u64, 0o100000u64);
    let (dsync, sync_, tmpfile_) = (0o10000u64, 0o4000000u64, 0o20000000u64);
    let order: [(u64, &str); 20] = [
        (0o100, "O_CREAT"),
        (0o200, "O_EXCL"),
        (0o400, "O_NOCTTY"),
        (0o1000, "O_TRUNC"),
        (0o2000, "O_APPEND"),
        (0o4000, "O_NONBLOCK"),
        (sync_ | dsync, "O_SYNC"),
        (sync_, "__O_SYNC"),
        (dsync, "O_DSYNC"),
        (direct, "O_DIRECT"),
        (largefile, "O_LARGEFILE"),
        (nofollow, "O_NOFOLLOW"),
        (0o1000000, "O_NOATIME"),
        (0o2000000, "O_CLOEXEC"),
        (0o10000000, "O_PATH"),
        (tmpfile_ | directory, "O_TMPFILE"),
        (tmpfile_, "__O_TMPFILE"),
        (directory, "O_DIRECTORY"),
        (0o20000, "FASYNC"),
        (0, ""),
    ];
    let mode = match v & 3 {
        0 => "O_RDONLY",
        1 => "O_WRONLY",
        2 => "O_RDWR",
        _ => "0x3",
    };
    let mut out = vec![mode.to_string()];
    let mut left = v & !3;
    for &(b, n) in order.iter().filter(|o| o.0 != 0) {
        if left & b == b {
            out.push(n.to_string());
            left &= !b;
        }
    }
    if left != 0 {
        out.push(format!("{left:#x}"));
    }
    out.join("|")
}

/// A mode as strace prints it: `0644`, `000`.
fn mode(v: u64) -> String {
    if v == 0 { "000".into() } else { format!("0{:o}", v as u32) }
}

fn whence(v: u64) -> String {
    match v {
        0 => "SEEK_SET",
        1 => "SEEK_CUR",
        2 => "SEEK_END",
        3 => "SEEK_DATA",
        4 => "SEEK_HOLE",
        _ => return v.to_string(),
    }
    .into()
}

fn fcntl_cmd(v: u64) -> String {
    match v as i32 {
        0 => "F_DUPFD",
        1 => "F_GETFD",
        2 => "F_SETFD",
        3 => "F_GETFL",
        4 => "F_SETFL",
        5 => "F_GETLK",
        6 => "F_SETLK",
        7 => "F_SETLKW",
        36 => "F_OFD_GETLK",
        37 => "F_OFD_SETLK",
        38 => "F_OFD_SETLKW",
        1024 => "F_SETLEASE",
        1025 => "F_GETLEASE",
        1030 => "F_DUPFD_CLOEXEC",
        1031 => "F_SETPIPE_SZ",
        1032 => "F_GETPIPE_SZ",
        1033 => "F_ADD_SEALS",
        1034 => "F_GET_SEALS",
        _ => return format!("{v:#x}"),
    }
    .into()
}

fn advice(v: u64) -> String {
    match v {
        0 => "POSIX_FADV_NORMAL",
        1 => "POSIX_FADV_RANDOM",
        2 => "POSIX_FADV_SEQUENTIAL",
        3 => "POSIX_FADV_WILLNEED",
        4 => "POSIX_FADV_DONTNEED",
        5 => "POSIX_FADV_NOREUSE",
        _ => return v.to_string(),
    }
    .into()
}

fn ioctl(v: u64) -> String {
    match v as u32 {
        0x5401 => "TCGETS",
        0x5413 => "TIOCGWINSZ",
        0x541b => "FIONREAD",
        0x5421 => "FIONBIO",
        0x5450 => "FIONCLEX",
        0x5451 => "FIOCLEX",
        0x8008_1272 => "BLKGETSIZE64",
        0x4004_9409 => "FICLONE",
        _ => return format!("{:#x}", v as u32),
    }
    .into()
}

fn prot(v: u64) -> String {
    if v == 0 {
        return "PROT_NONE".into();
    }
    bits(v, &[(1, "PROT_READ"), (2, "PROT_WRITE"), (4, "PROT_EXEC"), (0x0100_0000, "PROT_GROWSDOWN"), (0x0200_0000, "PROT_GROWSUP")])
}

fn map_flags(v: u64) -> String {
    let kind = match v & 0xf {
        1 => "MAP_SHARED",
        2 => "MAP_PRIVATE",
        3 => "MAP_SHARED_VALIDATE",
        _ => "",
    };
    let rest = bits(v & !0xf, &[(0x10, "MAP_FIXED"), (0x20, "MAP_ANONYMOUS"), (0x40, "MAP_32BIT"), (0x100, "MAP_GROWSDOWN"), (0x800, "MAP_DENYWRITE"), (0x1000, "MAP_EXECUTABLE"), (0x2000, "MAP_LOCKED"), (0x4000, "MAP_NORESERVE"), (0x8000, "MAP_POPULATE"), (0x10000, "MAP_NONBLOCK"), (0x20000, "MAP_STACK"), (0x40000, "MAP_HUGETLB"), (0x80000, "MAP_SYNC"), (0x100000, "MAP_FIXED_NOREPLACE")]);
    match (kind, rest.as_str()) {
        ("", r) => r.to_string(),
        (k, "0") => k.to_string(),
        (k, r) => format!("{k}|{r}"),
    }
}

const CLONE: [(u64, &str); 24] = [
    (0x100, "CLONE_VM"),
    (0x200, "CLONE_FS"),
    (0x400, "CLONE_FILES"),
    (0x800, "CLONE_SIGHAND"),
    (0x1000, "CLONE_PIDFD"),
    (0x2000, "CLONE_PTRACE"),
    (0x4000, "CLONE_VFORK"),
    (0x8000, "CLONE_PARENT"),
    (0x10000, "CLONE_THREAD"),
    (0x20000, "CLONE_NEWNS"),
    (0x40000, "CLONE_SYSVSEM"),
    (0x80000, "CLONE_SETTLS"),
    (0x100000, "CLONE_PARENT_SETTID"),
    (0x200000, "CLONE_CHILD_CLEARTID"),
    (0x400000, "CLONE_DETACHED"),
    (0x800000, "CLONE_UNTRACED"),
    (0x1000000, "CLONE_CHILD_SETTID"),
    (0x2000000, "CLONE_NEWCGROUP"),
    (0x4000000, "CLONE_NEWUTS"),
    (0x8000000, "CLONE_NEWIPC"),
    (0x10000000, "CLONE_NEWUSER"),
    (0x20000000, "CLONE_NEWPID"),
    (0x40000000, "CLONE_NEWNET"),
    (0x80000000, "CLONE_IO"),
];

/// clone's flags, its exit signal (the low byte) last as strace prints it.
fn clone_names(v: u64) -> String {
    let sig = (v & 0xff) as i32;
    let mut s = bits(v & !0xff, &CLONE);
    if sig != 0 {
        s = if s == "0" { signal(sig) } else { format!("{s}|{}", signal(sig)) };
    }
    s
}

fn iovs(blobs: &[(u8, u8, Vec<u8>)], v: u64, count: u64) -> String {
    let Some(b) = blob(blobs, 1) else { return ptr(v) };
    let mut out: Vec<String> = b.chunks_exact(IOVEC_SIZE).map(|e| format!("{{iov_base={}, iov_len={}}}", ptr(u64_at(e, 0)), u64_at(e, 8))).collect();
    if count as usize > out.len() {
        out.push("...".into());
    }
    format!("[{}]", out.join(", "))
}

fn events(xblobs: &[(u8, u8, Vec<u8>)], v: u64, ret: i64) -> String {
    if ret < 0 {
        return ptr(v);
    }
    let b = blob(xblobs, 3).unwrap_or(&[]);
    let mut out: Vec<String> = b.chunks_exact(IO_EVENT_SIZE).map(|e| format!("{{data={:#x}, obj={:#x}, res={}, res2={}}}", u64_at(e, 0), u64_at(e, 8), u64_at(e, 16) as i64, u64_at(e, 24) as i64)).collect();
    if ret as usize > out.len() {
        out.push("...".into());
    }
    format!("[{}]", out.join(", "))
}

fn signal(n: i32) -> String {
    const NAMES: [&str; 32] = [
        "0", "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL", "SIGTRAP", "SIGABRT", "SIGBUS", "SIGFPE", "SIGKILL", "SIGUSR1", "SIGSEGV", "SIGUSR2", "SIGPIPE", "SIGALRM", "SIGTERM", "SIGSTKFLT", "SIGCHLD", "SIGCONT", "SIGSTOP", "SIGTSTP", "SIGTTIN",
        "SIGTTOU", "SIGURG", "SIGXCPU", "SIGXFSZ", "SIGVTALRM", "SIGPROF", "SIGWINCH", "SIGIO", "SIGPWR", "SIGSYS",
    ];
    NAMES.get(n as usize).map_or_else(|| format!("SIGRT_{}", n - 32), |s| s.to_string())
}

/// Linux's errno names (asm-generic, the same on x86_64 and aarch64).
fn errno_name(e: i32) -> String {
    const NAMES: [&str; 134] = [
        "0", "EPERM", "ENOENT", "ESRCH", "EINTR", "EIO", "ENXIO", "E2BIG", "ENOEXEC", "EBADF", "ECHILD", "EAGAIN", "ENOMEM", "EACCES", "EFAULT", "ENOTBLK", "EBUSY", "EEXIST", "EXDEV", "ENODEV", "ENOTDIR", "EISDIR", "EINVAL",
        "ENFILE", "EMFILE", "ENOTTY", "ETXTBSY", "EFBIG", "ENOSPC", "ESPIPE", "EROFS", "EMLINK", "EPIPE", "EDOM", "ERANGE", "EDEADLK", "ENAMETOOLONG", "ENOLCK", "ENOSYS", "ENOTEMPTY", "ELOOP", "E41", "ENOMSG", "EIDRM",
        "ECHRNG", "EL2NSYNC", "EL3HLT", "EL3RST", "ELNRNG", "EUNATCH", "ENOCSI", "EL2HLT", "EBADE", "EBADR", "EXFULL", "ENOANO", "EBADRQC", "EBADSLT", "E58", "EBFONT", "ENOSTR", "ENODATA", "ETIME", "ENOSR", "ENONET",
        "ENOPKG", "EREMOTE", "ENOLINK", "EADV", "ESRMNT", "ECOMM", "EPROTO", "EMULTIHOP", "EDOTDOT", "EBADMSG", "EOVERFLOW", "ENOTUNIQ", "EBADFD", "EREMCHG", "ELIBACC", "ELIBBAD", "ELIBSCN", "ELIBMAX", "ELIBEXEC", "EILSEQ",
        "ERESTART", "ESTRPIPE", "EUSERS", "ENOTSOCK", "EDESTADDRREQ", "EMSGSIZE", "EPROTOTYPE", "ENOPROTOOPT", "EPROTONOSUPPORT", "ESOCKTNOSUPPORT", "EOPNOTSUPP", "EPFNOSUPPORT", "EAFNOSUPPORT", "EADDRINUSE",
        "EADDRNOTAVAIL", "ENETDOWN", "ENETUNREACH", "ENETRESET", "ECONNABORTED", "ECONNRESET", "ENOBUFS", "EISCONN", "ENOTCONN", "ESHUTDOWN", "ETOOMANYREFS", "ETIMEDOUT", "ECONNREFUSED", "EHOSTDOWN", "EHOSTUNREACH",
        "EALREADY", "EINPROGRESS", "ESTALE", "EUCLEAN", "ENOTNAM", "ENAVAIL", "EISNAM", "EREMOTEIO", "EDQUOT", "ENOMEDIUM", "EMEDIUMTYPE", "ECANCELED", "ENOKEY", "EKEYEXPIRED", "EKEYREVOKED", "EKEYREJECTED",
        "EOWNERDEAD", "ENOTRECOVERABLE", "ERFKILL", "EHWPOISON",
    ];
    match e {
        512 => "ERESTARTSYS".into(),
        513 => "ERESTARTNOINTR".into(),
        514 => "ERESTARTNOHAND".into(),
        516 => "ERESTART_RESTARTBLOCK".into(),
        524 => "ENOTSUPP".into(),
        _ => NAMES.get(e as usize).map_or_else(|| format!("E{e}"), |s| s.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nr(name: &str) -> u32 {
        calls().iter().flatten().find(|c| c.name == name).unwrap_or_else(|| panic!("{name}")).nr
    }

    fn pad(b: &mut Vec<u8>) {
        while b.len() % 8 != 0 {
            b.push(0);
        }
    }

    fn head(b: &mut Vec<u8>, kind: u16, ts: u64, pid: u32, tid: u32) {
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&ts.to_le_bytes());
        b.extend_from_slice(&pid.to_le_bytes());
        b.extend_from_slice(&tid.to_le_bytes());
    }

    fn done(mut b: Vec<u8>) -> Vec<u8> {
        pad(&mut b);
        let n = b.len() as u16;
        b[2..4].copy_from_slice(&n.to_le_bytes());
        b
    }

    fn blob(b: &mut Vec<u8>, arg: u8, what: u8, data: &[u8]) {
        b.extend_from_slice(&[arg, what]);
        b.extend_from_slice(&(data.len() as u16).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(data);
        pad(b);
    }

    /// An entry: `blobs` are (argument, kind, bytes).
    fn enter(ts: u64, pid: u32, tid: u32, name: &str, args: [u64; 6], blobs: &[(u8, u8, &[u8])]) -> Vec<u8> {
        let mut b = Vec::new();
        head(&mut b, ENTER, ts, pid, tid);
        b.extend_from_slice(&nr(name).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        for a in args {
            b.extend_from_slice(&a.to_le_bytes());
        }
        for (a, w, d) in blobs {
            blob(&mut b, *a, *w, d);
        }
        done(b)
    }

    fn exit(ts: u64, pid: u32, tid: u32, name: &str, ret: i64, blobs: &[(u8, u8, &[u8])]) -> Vec<u8> {
        let mut b = Vec::new();
        head(&mut b, EXIT, ts, pid, tid);
        b.extend_from_slice(&nr(name).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&ret.to_le_bytes());
        for (a, w, d) in blobs {
            blob(&mut b, *a, *w, d);
        }
        done(b)
    }

    fn other(kind: u16, ts: u64, pid: u32, tid: u32, a: u32, p: u32, ret: i64) -> Vec<u8> {
        let mut b = Vec::new();
        head(&mut b, kind, ts, pid, tid);
        b.extend_from_slice(&a.to_le_bytes());
        b.extend_from_slice(&p.to_le_bytes());
        b.extend_from_slice(&ret.to_le_bytes());
        done(b)
    }

    fn gap(cpu: u32, lost: u64, first: u64, last: u64) -> Vec<u8> {
        trailing_gaps(&[Loss::default(), Loss::default(), Loss { pending: lost, first, last, total: lost, untraced: 0 }])
            .into_iter()
            .collect::<Vec<u8>>()
            .chunks(48)
            .map(|c| {
                let mut c = c.to_vec();
                c[4..8].copy_from_slice(&cpu.to_le_bytes());
                c
            })
            .next()
            .unwrap()
    }

    /// The text of `records` (any order: the converter sorts them), the process 10 started in
    /// `/w` with descriptor 1 a pipe and 5 a file.
    fn text(records: &[Vec<u8>]) -> (String, Stats) {
        let d = std::env::temp_dir().join(format!("aeiou-ebpf-text-{}-{}", std::process::id(), records.len() ^ records.iter().map(|r| r.len()).sum::<usize>()));
        std::fs::create_dir_all(&d).unwrap();
        let bin = d.join("syscalls.bin");
        let mut f = MAGIC.to_vec();
        let hdr = json!({"arch": crate::ebpf::ARCH, "realtime_minus_monotonic": 1000.0, "seed": 10, "cwd": "/w", "inherited": [{"fd": 1, "target": "pipe:[7]"}, {"fd": 5, "target": "/w/held"}, {"fd": 6}]});
        f.extend_from_slice(serde_json::to_string(&hdr).unwrap().as_bytes());
        f.push(b'\n');
        for r in records {
            f.extend_from_slice(r);
        }
        std::fs::write(&bin, f).unwrap();
        let out = d.join("trace.txt");
        let st = convert(&bin, &out).unwrap();
        let t = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_dir_all(&d);
        (t, st)
    }

    const S: u64 = 1_000_000_000;
    const FDCWD: u64 = -100i64 as u64;

    fn exec(ts: u64) -> Vec<Vec<u8>> {
        vec![enter(ts, 10, 10, "execve", [0x10, 0x20, 0x30, 0, 0, 0], &[(0, STR, b"/bin/app")]), exit(ts + 1000, 10, 10, "execve", 0, &[])]
    }

    #[test]
    fn calls_print_as_strace_prints_them() {
        let mut r = vec![
            // before the exec: the capture's own, left out
            exit(S / 2, 10, 10, "read", 1, &[]),
            enter(S / 2 + 1, 10, 10, "close", [9, 0, 0, 0, 0, 0], &[]),
            exit(S / 2 + 2, 10, 10, "close", 0, &[]),
        ];
        r.extend(exec(S));
        r.extend([
            enter(2 * S, 10, 10, "openat", [FDCWD, 0x1, 0o2000000 | 0o100 | 1, 0o644, 0, 0], &[(1, STR, b"sub/../f \"q\"\n\x01")]),
            exit(2 * S + 5000, 10, 10, "openat", 3, &[]),
            enter(3 * S, 10, 10, "openat", [FDCWD, 0x1, 0, 0o777, 0, 0], &[(1, STR, b"/nope")]),
            exit(3 * S + 10, 10, 10, "openat", -2, &[]),
            enter(4 * S, 10, 10, "pread64", [3, 0x7f00, 4096, 8192, 0, 0], &[]),
            exit(4 * S + 10, 10, 10, "pread64", 4096, &[]),
            enter(5 * S, 10, 10, "lseek", [5, 100, 1, 0, 0, 0], &[]),
            exit(5 * S + 10, 10, 10, "lseek", 100, &[]),
            enter(6 * S, 10, 10, "write", [1, 0x7f00, 3, 0, 0, 0], &[]),
            exit(6 * S + 10, 10, 10, "write", 3, &[]),
            enter(7 * S, 10, 10, "mmap", [0, 8192, 1, 2, 3, 0], &[]),
            exit(7 * S + 10, 10, 10, "mmap", 0x7f12_3400_0000, &[]),
            enter(8 * S, 10, 10, "close", [3, 0, 0, 0, 0, 0], &[]),
            exit(8 * S + 10, 10, 10, "close", 0, &[]),
            enter(9 * S, 10, 10, "read", [3, 0x7f00, 1, 0, 0, 0], &[]),
            exit(9 * S + 10, 10, 10, "read", -9, &[]),
            enter(10 * S, 10, 10, "unlinkat", [FDCWD, 1, 0x200, 0, 0, 0], &[(1, STR, b"d")]),
            exit(10 * S + 10, 10, 10, "unlinkat", 0, &[]),
            enter(11 * S, 10, 10, "newfstatat", [FDCWD, 1, 0x7fff, 0x1000, 0, 0], &[(1, STR_FAULT, b"")]),
            exit(11 * S + 10, 10, 10, "newfstatat", 0, &[(1, STR, b"")]),
            enter(12 * S, 10, 10, "wait4", [4, 0, 0, 0, 0, 0], &[]),
            exit(12 * S + 10, 10, 10, "wait4", -512, &[]),
            enter(13 * S, 10, 10, "exit_group", [3, 0, 0, 0, 0, 0], &[]),
            other(GONE, 13 * S + 10, 10, 10, 0, 0, 3 << 8),
        ]);
        let (t, st) = text(&r);
        let lines: Vec<&str> = t.lines().collect();
        assert!(lines[0].starts_with("10 1001.000000 execve(\"/bin/app\", 0x20, 0x30) = 0 <0.000001>"), "{t}");
        assert_eq!(lines[1], r#"10 1002.000000 openat(AT_FDCWD</w>, "sub/../f \"q\"\n\1", O_WRONLY|O_CREAT|O_CLOEXEC, 0644) = 3</w/f "q"\n\1> <0.000005>"#);
        assert_eq!(lines[2], "10 1003.000000 openat(AT_FDCWD</w>, \"/nope\", O_RDONLY) = -1 ENOENT (No such file or directory) <0.000000>");
        assert!(lines[3].contains(r#"pread64(3</w/f "q""#) && lines[3].contains(">, 0x7f00, 4096, 8192) = 4096"), "{}", lines[3]);
        assert!(lines[4].contains("lseek(5</w/held>, 100, SEEK_CUR) = 100"), "{}", lines[4]);
        assert!(lines[5].contains("write(1<pipe:[7]>, 0x7f00, 3) = 3"), "{}", lines[5]);
        assert!(lines[6].contains(r#"mmap(NULL, 8192, PROT_READ, MAP_PRIVATE, 3</w/f "q""#) && lines[6].ends_with(", 0) = 0x7f1234000000 <0.000000>"), "{}", lines[6]);
        assert!(lines[7].contains("close(3</w/f"), "{}", lines[7]);
        assert!(lines[8].contains("read(3, 0x7f00, 1) = -1 EBADF (Bad file descriptor)"), "{}", lines[8]);
        assert!(lines[9].contains(r#"unlinkat(AT_FDCWD</w>, "d", AT_REMOVEDIR) = 0"#), "{}", lines[9]);
        assert!(lines[10].contains(r#"newfstatat(AT_FDCWD</w>, "", 0x7fff, AT_EMPTY_PATH) = 0"#), "{}", lines[10]);
        assert!(lines[11].contains("wait4(4, NULL, NULL, NULL) = ? ERESTARTSYS"), "{}", lines[11]);
        assert_eq!(lines[12], "10 1013.000000 exit_group(3) = ?");
        assert!(lines[13].ends_with("+++ exited with 3 +++"), "{}", lines[13]);
        assert_eq!(st.paths_not_copied, 0, "retried at the exit");
        assert_eq!(st.descriptors_unresolved, 1, "the read of a closed descriptor");
        assert_eq!(st.exits_without_entry, 0);
    }

    #[test]
    fn threads_interleave_and_forks_take_their_tables() {
        let mut r = exec(S);
        r.extend([
            enter(2 * S, 10, 10, "openat", [FDCWD, 1, 0o2000000, 0, 0, 0], &[(1, STR, b"a")]),
            exit(2 * S + 1, 10, 10, "openat", 3, &[]),
            enter(2 * S + 2, 10, 10, "openat", [FDCWD, 1, 0, 0, 0, 0], &[(1, STR, b"b")]),
            exit(2 * S + 3, 10, 10, "openat", 4, &[]),
            // a thread: clone with CLONE_FILES|CLONE_FS, its first call before the parent returns
            enter(3 * S, 10, 10, "clone", [0x3d0f00, 0x1000, 0, 0, 0, 0], &[]),
            other(FORK, 3 * S + 1, 10, 10, 11, 1, 0),
            enter(3 * S + 2, 10, 11, "read", [3, 0, 10, 0, 0, 0], &[]),
            exit(3 * S + 3, 10, 11, "clone", 0, &[]),
            exit(3 * S + 4, 10, 10, "clone", 11, &[]),
            exit(3 * S + 5, 10, 11, "read", 10, &[]),
            // the thread changes the shared directory
            enter(4 * S, 10, 11, "chdir", [1, 0, 0, 0, 0, 0], &[(0, STR, b"/x")]),
            exit(4 * S + 1, 10, 11, "chdir", 0, &[]),
            // a process: fork copies the table and the directory
            enter(5 * S, 10, 10, "fork", [0; 6], &[]),
            other(FORK, 5 * S + 1, 10, 10, 12, 0, 0),
            exit(5 * S + 2, 10, 10, "fork", 12, &[]),
            enter(5 * S + 3, 10, 10, "close", [4, 0, 0, 0, 0, 0], &[]),
            exit(5 * S + 4, 10, 10, "close", 0, &[]),
            enter(6 * S, 12, 12, "fcntl", [4, 1030, 7, 0, 0, 0], &[]),
            exit(6 * S + 1, 12, 12, "fcntl", 7, &[]),
            enter(6 * S + 2, 12, 12, "fchdir", [4, 0, 0, 0, 0, 0], &[]),
            exit(6 * S + 3, 12, 12, "fchdir", 0, &[]),
            enter(6 * S + 4, 12, 12, "openat", [FDCWD, 1, 0, 0, 0, 0], &[(1, STR, b"/proc/self/stat")]),
            exit(6 * S + 5, 12, 12, "openat", 8, &[]),
            // the exec closes what closes on exec
            enter(7 * S, 12, 12, "execve", [1, 0, 0, 0, 0, 0], &[(0, STR, b"/bin/x")]),
            exit(7 * S + 1, 12, 12, "execve", 0, &[]),
            enter(7 * S + 2, 12, 12, "read", [7, 0, 1, 0, 0, 0], &[]),
            exit(7 * S + 3, 12, 12, "read", 1, &[]),
            enter(7 * S + 4, 12, 12, "read", [3, 0, 1, 0, 0, 0], &[]),
            exit(7 * S + 5, 12, 12, "read", 1, &[]),
            enter(7 * S + 6, 12, 12, "dup3", [4, 9, 0o2000000, 0, 0, 0], &[]),
            exit(7 * S + 7, 12, 12, "dup3", 9, &[]),
            other(GONE, 8 * S, 12, 12, 0, 0, 9 | 0x80),
            other(GONE, 8 * S + 1, 10, 11, 0, 0, 0),
        ]);
        let (t, st) = text(&r);
        assert!(t.contains("10 1003.000000 clone(child_stack=0x1000, flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, parent_tid=NULL, child_tidptr=NULL, tls=NULL <unfinished ...>"), "{t}");
        assert!(t.contains("11 1003.000000 read(3</w/a>, NULL, 10 <unfinished ...>"), "{t}");
        assert!(t.contains("10 1003.000000 <... clone resumed>) = 11 <0.000000>"), "{t}");
        assert!(t.contains("11 1003.000000 <... read resumed>) = 10 <0.000000>"), "{t}");
        // the thread's chdir is the process's
        assert!(t.contains("fork() = 12"), "{t}");
        // the child's table is a copy: 4 is still open in it, closed in the parent
        assert!(t.contains("12 1006.000000 fcntl(4</w/b>, F_DUPFD_CLOEXEC, 0x7) = 7</w/b>"), "{t}");
        assert!(t.contains("openat(AT_FDCWD</w/b>, \"/proc/self/stat\", O_RDONLY) = 8</proc/12/stat>"), "{t}");
        // after the exec: 7 (cloexec) gone, 3 (opened O_CLOEXEC) gone
        assert!(t.contains("12 1007.000000 read(7, NULL, 1)"), "{t}");
        assert!(t.contains("12 1007.000000 read(3, NULL, 1)"), "{t}");
        assert!(t.contains("dup3(4</w/b>, 9, O_RDONLY|O_CLOEXEC) = 9</w/b>"), "{t}");
        assert!(t.contains("12 1008.000000 +++ killed by SIGKILL (core dumped) +++"), "{t}");
        assert!(t.contains("11 1008.000000 +++ exited with 0 +++"), "{t}");
        assert_eq!(st.descriptors_unresolved, 2, "{t}");
    }

    #[test]
    fn structures_losses_and_the_end() {
        let iov = [0x1000u64.to_le_bytes(), 10u64.to_le_bytes(), 0x2000u64.to_le_bytes(), 30u64.to_le_bytes()].concat();
        let mut iocb = vec![0u8; 64 * 3];
        // PWRITE of 4096 at 8192 on fd 5, rw flags DSYNC; FSYNC; PREADV (left out)
        iocb[0..8].copy_from_slice(&1u64.to_le_bytes());
        iocb[12..16].copy_from_slice(&2u32.to_le_bytes());
        iocb[16..18].copy_from_slice(&1u16.to_le_bytes());
        iocb[20..24].copy_from_slice(&5u32.to_le_bytes());
        iocb[24..32].copy_from_slice(&0xb0u64.to_le_bytes());
        iocb[32..40].copy_from_slice(&4096u64.to_le_bytes());
        iocb[40..48].copy_from_slice(&8192u64.to_le_bytes());
        iocb[64..72].copy_from_slice(&2u64.to_le_bytes());
        iocb[80..82].copy_from_slice(&2u16.to_le_bytes());
        iocb[84..88].copy_from_slice(&5u32.to_le_bytes());
        iocb[144..146].copy_from_slice(&7u16.to_le_bytes());
        let ev = [1u64.to_le_bytes(), 0xb0u64.to_le_bytes(), 4096u64.to_le_bytes(), 0u64.to_le_bytes()].concat();
        let mut ca = vec![0u8; 88];
        ca[0..8].copy_from_slice(&(0x100u64 | 0x400).to_le_bytes());
        ca[32..40].copy_from_slice(&17u64.to_le_bytes());
        let how = [(0o2000000u64 | 0o100).to_le_bytes(), 0o600u64.to_le_bytes(), 0u64.to_le_bytes()].concat();
        let mut r = exec(S);
        r.extend([
            enter(2 * S, 10, 10, "writev", [5, 0x9, 3, 0, 0, 0], &[(1, BYTES, &iov)]),
            exit(2 * S + 1, 10, 10, "writev", 40, &[]),
            enter(3 * S, 10, 10, "io_submit", [0xc, 3, 0xd, 0, 0, 0], &[(2, BYTES, &iocb)]),
            exit(3 * S + 1, 10, 10, "io_submit", 3, &[]),
            enter(4 * S, 10, 10, "io_getevents", [0xc, 1, 8, 0xe, 0, 0], &[]),
            enter(4 * S + 1, 10, 13, "fsync", [5, 0, 0, 0, 0, 0], &[]),
            exit(4 * S + 2, 10, 10, "io_getevents", 2, &[(3, BYTES, &ev)]),
            enter(5 * S, 10, 10, "clone3", [0xf, 88, 0, 0, 0, 0], &[(0, BYTES, &ca)]),
            exit(5 * S + 1, 10, 10, "clone3", 14, &[]),
            enter(6 * S, 10, 10, "openat2", [FDCWD, 1, 0x10, 24, 0, 0], &[(1, STR_CUT, b"long"), (2, BYTES, &how)]),
            exit(6 * S + 1, 10, 10, "openat2", 6, &[]),
            gap(2, 77, 6 * S + 5, 7 * S),
            enter(8 * S, 10, 10, "fadvise64", [6, 0, 0, 4, 0, 0], &[]),
            exit(8 * S + 1, 10, 10, "fadvise64", 0, &[]),
            enter(8 * S + 2, 10, 10, "ioctl", [6, 0x541b, 0x9, 0, 0, 0], &[]),
            exit(8 * S + 3, 10, 10, "ioctl", 0, &[]),
            enter(8 * S + 4, 10, 10, "fallocate", [6, 0x11, 0, 4096, 0, 0], &[]),
            exit(8 * S + 5, 10, 10, "fallocate", 0, &[]),
            enter(8 * S + 6, 10, 10, "pwritev2", [6, 0x9, 40, 0, 0, 0x10], &[(1, BYTES, &iov)]),
            exit(8 * S + 7, 10, 10, "pwritev2", 40, &[]),
            exit(8 * S + 8, 10, 13, "fsync", 0, &[]),
            exit(8 * S + 9, 10, 10, "read", 1, &[]),
            enter(9 * S, 10, 10, "read", [5, 0, 1, 0, 0, 0], &[]),
            enter(9 * S + 1, 10, 15, "write", [5, 0, 1, 0, 0, 0], &[]),
        ]);
        let (t, st) = text(&r);
        assert!(t.contains("writev(5</w/held>, [{iov_base=0x1000, iov_len=10}, {iov_base=0x2000, iov_len=30}, ...], 3) = 40"), "{t}");
        assert!(t.contains("io_submit(0xc, 3, [{aio_data=0x1, aio_rw_flags=RWF_DSYNC, aio_lio_opcode=IOCB_CMD_PWRITE, aio_fildes=5</w/held>, aio_buf=0xb0, aio_nbytes=4096, aio_offset=8192}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_FSYNC, aio_fildes=5</w/held>}]) = 3"), "{t}");
        assert!(t.contains("10 1004.000000 io_getevents(0xc, 1, 8,  <unfinished ...>"), "{t}");
        assert!(t.contains("13 1004.000000 fsync(5 <unfinished ...>"), "{t}");
        assert!(t.contains("10 1004.000000 <... io_getevents resumed>[{data=0x1, obj=0xb0, res=4096, res2=0}, ...], NULL) = 2"), "{t}");
        assert!(t.contains("clone3({flags=CLONE_VM|CLONE_FILES, child_tid=NULL, parent_tid=NULL, exit_signal=SIGCHLD, stack=NULL, stack_size=0x0, tls=NULL}, 88) = 14"), "{t}");
        assert!(t.contains(r#"openat2(AT_FDCWD</w>, "long"..., {flags=O_RDONLY|O_CREAT|O_CLOEXEC, mode=0o600, resolve=0x0}, 24) = 6"#), "{t}");
        assert!(t.contains("--- aeiou: 77 event(s) lost on cpu 2 from 1006.000000 to 1007.000000: the ring buffer was full ---"), "{t}");
        assert!(t.contains("fadvise64(6</w/long>, 0, 0, POSIX_FADV_DONTNEED) = 0"), "{t}");
        assert!(t.contains("ioctl(6</w/long>, FIONREAD, 0x9) = 0"), "{t}");
        assert!(t.contains("fallocate(6</w/long>, FALLOC_FL_KEEP_SIZE|FALLOC_FL_ZERO_RANGE, 0, 4096) = 0"), "{t}");
        assert!(t.contains("pwritev2(6</w/long>, [{iov_base=0x1000, iov_len=10}, {iov_base=0x2000, iov_len=30}, ...], 40, 0, RWF_APPEND) = 40"), "{t}");
        assert!(t.contains("13 1008.000000 <... fsync resumed>) = 0 <4.000000>"), "{t}");
        // left at the end: printed with no return
        assert!(t.contains("10 1009.000000 <... read resumed>) = ?"), "{t}");
        assert_eq!(st.windows, vec![(2, 77, 6 * S + 5, 7 * S)]);
        assert_eq!(st.vectored_aio_left_out, 1);
        assert_eq!(st.exits_without_entry, 1, "the read whose entry was lost");
        assert_eq!(st.entries_without_exit, 2);
        assert!(st.json()["records"].as_u64().unwrap() > 20);
    }

    #[test]
    fn the_records_file_is_checked() {
        let d = std::env::temp_dir().join(format!("aeiou-ebpf-text-bad-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let bin = d.join("syscalls.bin");
        std::fs::write(&bin, b"NOTMAGIC{}\n").unwrap();
        assert!(convert(&bin, &d.join("t")).unwrap_err().to_string().contains("not the eBPF source's records"));
        assert!(header(&bin).is_err());
        std::fs::write(&bin, b"AEIO").unwrap();
        assert!(convert(&bin, &d.join("t")).unwrap_err().to_string().contains("shorter than its header"));
        std::fs::write(&bin, [MAGIC.as_slice(), b"{\"arch\": \"sparc\"}\n"].concat()).unwrap();
        assert!(convert(&bin, &d.join("t")).unwrap_err().to_string().contains("recorded on \"sparc\""));
        assert_eq!(header(&bin).unwrap()["arch"], "sparc");
        let hdr = format!("{{\"arch\": \"{}\"}}\n", crate::ebpf::ARCH);
        let mut short = [MAGIC.as_slice(), hdr.as_bytes()].concat();
        short.extend_from_slice(&[1, 0, 8, 0]);
        short.extend_from_slice(&[0; 20]);
        std::fs::write(&bin, &short).unwrap();
        assert!(convert(&bin, &d.join("t")).unwrap_err().to_string().contains("a record of 8 bytes"));
        let mut cut = [MAGIC.as_slice(), hdr.as_bytes()].concat();
        cut.extend_from_slice(&enter(1, 10, 10, "read", [0; 6], &[])[..40]);
        std::fs::write(&bin, &cut).unwrap();
        assert!(convert(&bin, &d.join("t")).unwrap_err().to_string().contains("cut short"));
        let mut odd = [MAGIC.as_slice(), hdr.as_bytes()].concat();
        odd.extend_from_slice(&other(9, 1, 10, 10, 0, 0, 0));
        std::fs::write(&bin, &odd).unwrap();
        assert!(convert(&bin, &d.join("t")).unwrap_err().to_string().contains("a record of kind 9"));
        assert!(convert(&d.join("missing"), &d.join("t")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_names() {
        assert_eq!(open_flags(0o2000000 | 0o200000 | 0o4000 | 2), if cfg!(target_arch = "x86_64") { "O_RDWR|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY" } else { "O_RDWR|O_NONBLOCK|O_CLOEXEC|O_DIRECT" });
        assert_eq!(open_flags(0o4010000 | 1), "O_WRONLY|O_SYNC");
        assert_eq!(open_flags(0o4000000), "O_RDONLY|__O_SYNC");
        assert_eq!(open_flags(3 | 0o100000000), "0x3|0x1000000");
        assert_eq!(annotation("/a b<c>\\\n\u{1}é"), "/a b\\74c\\76\\\\\\n\\1\\303\\251");
        assert_eq!(crate::strace::fd_path(&annotation("/a b<c>\\\n\u{1}é")), "/a b<c>\\\n\u{1}é");
        assert_eq!(clone_names(0x11), "SIGCHLD");
        assert_eq!(clone_names(0), "0");
        assert_eq!(map_flags(0x22), "MAP_PRIVATE|MAP_ANONYMOUS");
        assert_eq!(map_flags(0x20), "MAP_ANONYMOUS");
        assert_eq!(map_flags(3), "MAP_SHARED_VALIDATE");
        assert_eq!(map_flags(1 | 0x8000_0000), "MAP_SHARED|0x80000000");
        assert_eq!(prot(0), "PROT_NONE");
        assert_eq!(prot(7), "PROT_READ|PROT_WRITE|PROT_EXEC");
        assert_eq!(mode(0), "000");
        assert_eq!(whence(2), "SEEK_END");
        assert_eq!(whence(9), "9");
        assert_eq!(fcntl_cmd(0), "F_DUPFD");
        assert_eq!(fcntl_cmd(99), "0x63");
        assert_eq!(advice(9), "9");
        assert_eq!(ioctl(0x1234), "0x1234");
        assert_eq!(ioctl(0x8008_1272), "BLKGETSIZE64");
        assert_eq!(bits(0x300, &AT_FLAGS), "AT_SYMLINK_NOFOLLOW|AT_EACCESS");
        assert_eq!(errno_name(2), "ENOENT");
        assert_eq!(errno_name(524), "ENOTSUPP");
        assert_eq!(errno_name(400), "E400");
        assert_eq!(signal(9), "SIGKILL");
        assert_eq!(signal(34), "SIGRT_2");
        assert_eq!(proc_self("/proc/thread-self/fd", 3, 4), "/proc/3/task/4/fd");
        assert_eq!(proc_self("/proc/self", 3, 4), "/proc/3");
        assert_eq!(proc_self("/proc/selfish", 3, 4), "/proc/selfish");
        assert_eq!(string(0, 0, &[]), "NULL");
        assert_eq!(string(0, 0x10, &[(0, STR_FAULT, vec![])]), "0x10");
        assert_eq!(iovs(&[], 0x5, 1), "0x5");
        assert_eq!(events(&[], 0x5, -14), "0x5");
        assert_eq!(trailing_gaps(&[Loss::default()]).len(), 0);
    }
}
