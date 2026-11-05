//! The eBPF source of `aeiou-trace capture` (DESIGN_REVIEW.md §3.96 "Phase 3"): the capture's
//! BPF object (`runner/capture-ebpf/`, built by `build.rs` and embedded here) loaded and
//! attached, the reader that copies its ring buffer to `syscalls.bin` while the command runs,
//! and what the converter (`ebpf_text.rs`) needs of both: the calls traced on this
//! architecture and what each argument is.
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[path = "../../capture-ebpf/src/record.rs"]
#[allow(dead_code)]
pub mod record;
use record::*;

unsafe impl aya::Pod for Config {}
unsafe impl aya::Pod for Loss {}

/// The BPF object, for this build's architecture.
pub static OBJECT: &[u8] = aya::include_bytes_aligned!(concat!(env!("OUT_DIR"), "/aeiou-capture.bpf.o"));

/// The records' file in a capture directory, kept so a converter's bug is fixed by converting
/// again (`aeiou-trace capture --convert`), never by asking the owner for another capture.
pub const FILE: &str = "syscalls.bin";
/// `syscalls.bin`'s first 8 bytes; a JSON header line follows, then the records.
pub const MAGIC: &[u8; 8] = b"AEIOUSC1";
/// The ring buffer's default size (`--buffer`), about two seconds of the spike's peak of 2.9
/// million events a second; to be revisited with captures in the wild (the user, 2026-11-03).
pub const DEFAULT_BUFFER: u64 = 256 << 20;

#[cfg(target_arch = "x86_64")]
pub const ARCH: &str = "x86_64";
#[cfg(target_arch = "aarch64")]
pub const ARCH: &str = "aarch64";
#[cfg(target_arch = "x86_64")]
static TABLE: &str = include_str!("syscalls/x86_64.tsv");
#[cfg(target_arch = "aarch64")]
static TABLE: &str = include_str!("syscalls/aarch64.tsv");

/// A traced call: its number here, its name, and what each argument strace prints is, one
/// letter each (`syscalls/*.tsv`): `d` a descriptor, `D` a directory's (`AT_FDCWD`), `n` one
/// the call makes (`dup2`'s second), `p` a
/// path and `s` another string (both copied by the probe), `i` signed and `u` unsigned
/// decimal, `x` hexadecimal, `o` open flags, `m` a mode, `w` lseek's whence, `c` clone's flags,
/// `C` clone3's arguments, `F` fcntl's command, `a` `AT_` flags, `U` unlinkat's, `f`
/// fallocate's mode, `r` `RWF_` flags, `v` an iovec array (its count the next argument), `A`
/// fadvise's advice, `I` an ioctl's request, `P` mmap's protection, `M` its flags, `b`
/// io_submit's iocbs, `e` io_getevents' events, `H` openat2's `open_how`; `_` a raw argument
/// strace does not print.
#[derive(Debug, Clone, Copy)]
pub struct Call {
    pub nr: u32,
    pub name: &'static str,
    pub kinds: &'static str,
}

/// The calls of this architecture's table, by number.
pub fn calls() -> &'static [Option<Call>] {
    static CALLS: std::sync::OnceLock<Vec<Option<Call>>> = std::sync::OnceLock::new();
    CALLS.get_or_init(|| {
        let mut v = vec![None; NR_MAX as usize];
        for line in TABLE.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let mut f = line.split('\t');
            let (Some(nr), Some(name), Some(kinds)) = (f.next(), f.next(), f.next()) else { panic!("syscalls/{ARCH}.tsv: {line}") };
            let nr: u32 = nr.parse().expect("a call's number");
            v[nr as usize] = Some(Call { nr, name, kinds: if kinds == "-" { "" } else { kinds } });
        }
        v
    })
}

/// What the probe copies of a call (`record::SPECS`'s value).
pub fn spec(c: &Call) -> u32 {
    let mut s = TRACED;
    let mut raw = 0;
    for k in c.kinds.chars() {
        if k == 'p' || k == 's' {
            s |= 1 << raw;
        }
        raw += 1;
    }
    let special = match c.name {
        "readv" | "writev" | "preadv" | "pwritev" | "preadv2" | "pwritev2" => IOV << 8,
        "io_submit" => IOCB << 8,
        "clone3" => CLONE_ARGS << 8,
        "openat2" => HOW << 8,
        "io_getevents" | "io_pgetevents" => IO_EVENTS << 16,
        _ => 0,
    };
    s | special
}

// ---------------------------------------------------------------- BTF

/// The structs of a BTF blob (the kernel's own, `/sys/kernel/btf/vmlinux`): enough to find a
/// member's offset and a struct's size, which is all the probes need of the kernel's layout.
pub struct Btf {
    /// Each type by id (0 is void): its name, kind, whether its members carry bit-field sizes,
    /// its members (name, type, offset in bits), and its size (or the type it refers to).
    types: Vec<(String, u32, bool, Vec<(String, u32, u32)>, u32)>,
}

impl Btf {
    pub fn parse(btf: &[u8]) -> Option<Btf> {
        let u32_at = |o: usize| -> Option<u32> { btf.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())) };
        if btf.get(0..2)? != [0x9f, 0xeb] {
            return None;
        }
        let hdr_len = u32_at(4)? as usize;
        let (type_off, type_len, str_off, str_len) = (u32_at(8)? as usize, u32_at(12)? as usize, u32_at(16)? as usize, u32_at(20)? as usize);
        let types = btf.get(hdr_len + type_off..hdr_len + type_off + type_len)?;
        let strs = btf.get(hdr_len + str_off..hdr_len + str_off + str_len)?;
        let s_at = |o: u32| -> String {
            let b = strs.get(o as usize..).unwrap_or(&[]);
            String::from_utf8_lossy(&b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())]).into_owned()
        };
        let rd = |o: usize| -> Option<u32> { types.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())) };
        let mut all = vec![(String::new(), 0, false, Vec::new(), 0)];
        let mut at = 0;
        while at + 12 <= types.len() {
            let (name_off, info, size_or_type) = (rd(at)?, rd(at + 4)?, rd(at + 8)?);
            let kind = (info >> 24) & 0x1f;
            let vlen = (info & 0xffff) as usize;
            at += 12;
            let mut members = Vec::new();
            at += match kind {
                1 | 14 | 17 => 4,
                3 => 12,
                4 | 5 => {
                    for m in 0..vlen {
                        let o = at + 12 * m;
                        members.push((s_at(rd(o)?), rd(o + 4)?, rd(o + 8)?));
                    }
                    12 * vlen
                }
                13 => {
                    // a function's prototype: its parameters, as members of no offset
                    for m in 0..vlen {
                        let o = at + 8 * m;
                        members.push((s_at(rd(o)?), rd(o + 4)?, 0));
                    }
                    8 * vlen
                }
                6 => 8 * vlen,
                15 | 19 => 12 * vlen,
                _ => 0,
            };
            all.push((s_at(name_off), kind, info >> 31 == 1, members, size_or_type));
        }
        Some(Btf { types: all })
    }

    fn strukt(&self, name: &str) -> Option<usize> {
        self.types.iter().position(|(n, k, _, _, _)| *k == 4 && n == name)
    }

    /// A struct's size in bytes.
    pub fn size(&self, name: &str) -> Option<u32> {
        self.strukt(name).map(|i| self.types[i].4)
    }

    /// The byte offset of `member` in the struct `name`, looked for through anonymous structs
    /// and unions; `None` when there is no such struct or member, or it is a bit field.
    pub fn offset(&self, name: &str, member: &str) -> Option<u32> {
        let (bits, bitfield) = self.find(self.strukt(name)?, member, 0)?;
        (!bitfield && bits % 8 == 0).then_some(bits / 8)
    }

    /// What a tracepoint passes its raw program (`btf_trace_NAME`'s prototype, its first
    /// parameter `void *__data` left out): for each argument the struct it points to, or an
    /// empty name for any other type; `None` when the kernel has no such tracepoint.
    pub fn tracepoint_args(&self, name: &str) -> Option<Vec<String>> {
        let want = format!("btf_trace_{name}");
        let td = self.types.iter().position(|(n, k, _, _, _)| *k == 8 && *n == want)?;
        let skip = |mut t: usize| {
            while let Some((_, 8..=11 | 18, _, _, next)) = self.types.get(t) {
                t = *next as usize;
            }
            t
        };
        let ptr = skip(self.types[td].4 as usize);
        let (_, 2, _, _, proto) = self.types.get(ptr)? else { return None };
        let proto = skip(*proto as usize);
        let (_, 13, _, params, _) = self.types.get(proto)? else { return None };
        Some(
            params
                .iter()
                .skip(1)
                .map(|(_, t, _)| match self.types.get(skip(*t as usize)) {
                    Some((_, 2, _, _, to)) => match self.types.get(skip(*to as usize)) {
                        Some((n, 4, _, _, _)) => n.clone(),
                        _ => String::new(),
                    },
                    _ => String::new(),
                })
                .collect(),
        )
    }

    fn find(&self, id: usize, member: &str, depth: u32) -> Option<(u32, bool)> {
        let (_, _, kflag, members, _) = self.types.get(id)?;
        for (mname, mtype, off) in members {
            let (bits, size) = if *kflag { (off & 0xff_ffff, off >> 24) } else { (*off, 0) };
            if mname == member {
                return Some((bits, size != 0));
            }
            if mname.is_empty() && depth < 8 {
                // through typedefs and qualifiers to the anonymous struct or union
                let mut t = *mtype as usize;
                while let Some((_, 8..=11 | 18, _, _, next)) = self.types.get(t) {
                    t = *next as usize;
                }
                if matches!(self.types.get(t).map(|x| x.1), Some(4 | 5)) {
                    if let Some((b, bf)) = self.find(t, member, depth + 1) {
                        return Some((bits + b, bf));
                    }
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------- privilege

#[repr(C)]
struct CapHeader {
    version: u32,
    pid: i32,
}
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct CapData {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}
const CAP_V3: u32 = 0x2008_0522;
const CAP_SYS_ADMIN: u32 = 21;
const CAP_PERFMON: u32 = 38;
const CAP_BPF: u32 = 39;

fn effective() -> u64 {
    let mut h = CapHeader { version: CAP_V3, pid: 0 };
    let mut d = [CapData::default(); 2];
    if unsafe { libc::syscall(libc::SYS_capget, &mut h as *mut CapHeader, d.as_mut_ptr()) } != 0 {
        return 0;
    }
    d[0].effective as u64 | (d[1].effective as u64) << 32
}

/// How this process may load and attach the probes: as root, or under file capabilities
/// (`CAP_BPF` and `CAP_PERFMON`, or `CAP_SYS_ADMIN`); `None` when it cannot.
pub fn privilege() -> Option<&'static str> {
    if unsafe { libc::geteuid() } == 0 {
        return Some("root");
    }
    let e = effective();
    let has = |c: u32| e & (1 << c) != 0;
    if has(CAP_BPF) && has(CAP_PERFMON) {
        Some("cap_bpf,cap_perfmon")
    } else if has(CAP_SYS_ADMIN) {
        Some("cap_sys_admin")
    } else {
        None
    }
}

/// Every capability this process holds, given up for good (the ambient ones too): after the
/// probes are attached and before the command is started or the buffer read.
pub fn drop_privilege() -> Result<()> {
    unsafe { libc::prctl(libc::PR_CAP_AMBIENT, libc::PR_CAP_AMBIENT_CLEAR_ALL as libc::c_ulong, 0, 0, 0) };
    let h = CapHeader { version: CAP_V3, pid: 0 };
    let d = [CapData::default(); 2];
    if unsafe { libc::syscall(libc::SYS_capset, &h as *const CapHeader, d.as_ptr()) } != 0 {
        bail!("capset: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

// ---------------------------------------------------------------- the probes

/// This process's PID namespace, as `bpf_get_ns_current_pid_tgid` names it.
fn pid_ns() -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata("/proc/self/ns/pid").context("/proc/self/ns/pid")?;
    Ok((m.dev(), m.ino()))
}

/// The kernel structures' offsets the probes read, from the kernel's BTF, in a `Config` with
/// no namespace and no seed; and what `io_uring`'s probes need (`uring_layout`), or why they
/// cannot run here.
pub fn kernel_layout() -> Result<(Config, std::result::Result<Value, String>)> {
    let p = "/sys/kernel/btf/vmlinux";
    let raw = std::fs::read(p).with_context(|| format!("{p}: the kernel's BTF (CONFIG_DEBUG_INFO_BTF), which the eBPF source needs"))?;
    let btf = Btf::parse(&raw).ok_or_else(|| anyhow!("{p}: not BTF"))?;
    let mut cfg = base_layout(&btf, p)?;
    let uring = uring_layout(&btf, &mut cfg);
    Ok((cfg, uring))
}

fn base_layout(btf: &Btf, p: &str) -> Result<Config> {
    let off = |s: &str, m: &str| btf.offset(s, m).ok_or_else(|| anyhow!("{p}: no {s}.{m}"));
    Ok(Config {
        tgid_off: off("task_struct", "tgid")?,
        exit_code_off: off("task_struct", "exit_code")?,
        thread_pid_off: off("task_struct", "thread_pid")?,
        level_off: off("pid", "level")?,
        numbers_off: off("pid", "numbers")?,
        upid_size: btf.size("upid").ok_or_else(|| anyhow!("{p}: no struct upid"))?,
        signal_off: off("task_struct", "signal")?,
        live_off: off("signal_struct", "live")?,
        ..Config::default()
    })
}

/// What `io_uring`'s probes read (DESIGN_REVIEW.md §3.98), from the kernel's BTF, set in `cfg`
/// (`uring` 1): the tracepoints' prototypes, the request's fields, and the descriptor table's
/// way to a ring; and, for the converter, the offsets in the request's first `CMD_SIZE` bytes
/// of each operation's fields, the header's `uring.layout`. `Err` says why the probes cannot
/// run on this kernel, and the capture goes on without them.
pub fn uring_layout(btf: &Btf, cfg: &mut Config) -> std::result::Result<Value, String> {
    let sqe = btf.tracepoint_args("io_uring_submit_req").ok_or("the kernel has no io_uring tracepoints (CONFIG_IO_URING)")?;
    let cqe = btf.tracepoint_args("io_uring_complete").ok_or("the kernel has no tracepoint io_uring_complete")?;
    let req_arg = sqe.iter().position(|t| t == "io_kiocb").ok_or("io_uring_submit_req passes no struct io_kiocb")?;
    if req_arg > 1 || cqe.len() < 3 {
        return Err(format!("io_uring's tracepoints are not of a form the probes read (io_uring_submit_req {sqe:?}, io_uring_complete {cqe:?})"));
    }
    let cqe_struct = cqe[2] == "io_uring_cqe";
    if !cqe_struct && cqe.len() < 5 {
        return Err(format!("io_uring_complete is not of a form the probes read ({cqe:?})"));
    }
    let off = |s: &str, m: &str| btf.offset(s, m).ok_or_else(|| format!("the kernel's BTF has no {s}.{m}"));
    let size = |s: &str| btf.size(s).ok_or_else(|| format!("the kernel's BTF has no struct {s}"));
    // the operation's own structure is the request's first member
    if off("io_kiocb", "file")? != 0 {
        return Err("struct io_kiocb does not begin with the operation's structure".into());
    }
    for s in ["io_rw", "io_open", "io_close", "io_sync", "io_statx"] {
        if size(s)? as usize > CMD_SIZE {
            return Err(format!("struct {s} is larger than the {CMD_SIZE} bytes the probe copies"));
        }
    }
    let cqe_off = off("io_kiocb", "cqe")?;
    cfg.uring = 1;
    cfg.sqe_req_arg = req_arg as u32;
    cfg.cqe_struct = cqe_struct as u32;
    cfg.req_opcode_off = off("io_kiocb", "opcode")?;
    cfg.req_flags_off = off("io_kiocb", "flags")?;
    cfg.req_user_data_off = cqe_off + off("io_cqe", "user_data")?;
    cfg.req_fd_off = cqe_off + off("io_cqe", "fd")?;
    cfg.req_ctx_off = off("io_kiocb", "ctx")?;
    cfg.open_filename_off = off("io_open", "filename")?;
    cfg.filename_name_off = off("filename", "name")?;
    cfg.statx_filename_off = off("io_statx", "filename")?;
    cfg.task_files_off = off("task_struct", "files")?;
    cfg.files_fdt_off = off("files_struct", "fdt")?;
    cfg.fdt_fd_off = off("fdtable", "fd")?;
    cfg.file_private_off = off("file", "private_data")?;
    let how = off("io_open", "how")?;
    Ok(json!({
        "rw": {"pos": off("io_rw", "kiocb")? + off("kiocb", "ki_pos")?, "addr": off("io_rw", "addr")?, "len": off("io_rw", "len")?, "flags": off("io_rw", "flags")?},
        "open": {"dfd": off("io_open", "dfd")?, "file_slot": off("io_open", "file_slot")?, "flags": how + off("open_how", "flags")?, "mode": how + off("open_how", "mode")?, "resolve": how + off("open_how", "resolve")?},
        "close": {"fd": off("io_close", "fd")?, "file_slot": off("io_close", "file_slot")?},
        "sync": {"len": off("io_sync", "len")?, "off": off("io_sync", "off")?, "flags": off("io_sync", "flags")?},
        "statx": {"dfd": off("io_statx", "dfd")?, "mask": off("io_statx", "mask")?, "flags": off("io_statx", "flags")?},
    }))
}

/// The probes loaded (and the verifier's verdict on them) with a ring buffer of `buffer` bytes,
/// the traced calls marked, and `task_struct`'s offsets set; nothing attached.
pub fn load(buffer: u64) -> Result<(aya::Ebpf, std::result::Result<Value, String>)> {
    let (mut config, uring) = kernel_layout()?;
    (config.ns_dev, config.ns_ino) = pid_ns()?;
    // found by the probe when it adds the command (record::Config)
    config.ns_level = u32::MAX;
    let size = u32::try_from(buffer).map_err(|_| anyhow!("--buffer {buffer}: at most 2 GiB"))?;
    let mut bpf = aya::EbpfLoader::new().map_max_entries("EVENTS", size).load(OBJECT).context("loading the capture's BPF object")?;
    {
        let mut specs: aya::maps::Array<_, u32> = aya::maps::Array::try_from(bpf.map_mut("SPECS").expect("SPECS"))?;
        for c in calls().iter().flatten() {
            specs.set(c.nr, spec(c), 0)?;
        }
    }
    {
        let mut cfg: aya::maps::Array<_, Config> = aya::maps::Array::try_from(bpf.map_mut("CONFIG").expect("CONFIG"))?;
        cfg.set(0, config, 0)?;
    }
    for name in PROGRAMS.iter().chain(URING_PROGRAMS.iter()) {
        let p: &mut aya::programs::RawTracePoint = bpf.program_mut(name).expect("a program of the object").try_into()?;
        p.load().with_context(|| format!("the verifier, on the program {name}"))?;
    }
    Ok((bpf, uring))
}

const PROGRAMS: [&str; 4] = ["sys_enter", "sys_exit", "sched_process_fork", "sched_process_exit"];
/// `io_uring`'s programs, loaded always; attached when the kernel has what they read
/// (`uring_layout`), one of each tracepoint's forms (`uring_programs`).
const URING_PROGRAMS: [&str; 5] = ["io_uring_submit_req", "io_uring_submit_req_after_ring", "io_uring_complete", "io_uring_complete_fields", "uring_calls"];

/// The programs of `io_uring` this kernel's tracepoints take (`Config.sqe_req_arg`, `cqe_struct`):
/// a raw program that reads past a tracepoint's last argument is refused at attach.
fn uring_programs(c: &Config) -> [&'static str; 3] {
    [
        if c.sqe_req_arg == 0 { "io_uring_submit_req" } else { "io_uring_submit_req_after_ring" },
        if c.cqe_struct == 1 { "io_uring_complete" } else { "io_uring_complete_fields" },
        "uring_calls",
    ]
}

/// The tracepoint a program attaches to.
fn tracepoint_of(program: &str) -> &str {
    match program {
        "uring_calls" => "sys_exit",
        "io_uring_submit_req_after_ring" => "io_uring_submit_req",
        "io_uring_complete_fields" => "io_uring_complete",
        p => p,
    }
}

/// The source's check: a privilege to load with, the kernel's BTF, and the object loaded and
/// passed by the verifier (then unloaded): `Err` says why the source cannot run here.
pub fn usable() -> std::result::Result<String, String> {
    let how = privilege().ok_or_else(|| {
        "eBPF needs root or the file capabilities cap_bpf,cap_perfmon on a node-local copy of aeiou-trace (setcap cap_bpf,cap_perfmon+ep; cap_sys_admin before Linux 5.8); --source strace runs as any user".to_string()
    })?;
    let (_, uring) = load(1 << 16).map_err(|e| format!("{e:#}"))?;
    Ok(match uring {
        Ok(_) => format!("loaded and verified as {how}, kernel BTF, io_uring's requests"),
        Err(why) => format!("loaded and verified as {how}, kernel BTF; io_uring's requests not traced: {why}"),
    })
}

/// The probes attached to a command not yet started: `seed` is its pid as this process sees it.
pub struct Probes {
    bpf: aya::Ebpf,
    links: usize,
    /// `io_uring`'s layout for the converter (`uring_layout`), or why its probes are not attached.
    pub uring: std::result::Result<Value, String>,
}

impl Probes {
    pub fn attach((mut bpf, uring): (aya::Ebpf, std::result::Result<Value, String>)) -> Result<Probes> {
        let mut links = 0;
        let cfg: aya::maps::Array<_, Config> = aya::maps::Array::try_from(bpf.map("CONFIG").expect("CONFIG"))?;
        let c = cfg.get(&0, 0)?;
        let names: &[&str] = if uring.is_ok() { &[&PROGRAMS[..], &uring_programs(&c)[..]].concat() } else { &PROGRAMS };
        for name in names {
            let p: &mut aya::programs::RawTracePoint = bpf.program_mut(name).expect("a program").try_into()?;
            p.attach(tracepoint_of(name)).with_context(|| format!("attaching {name}"))?;
            links += 1;
        }
        Ok(Probes { bpf, links, uring })
    }

    /// The header's `uring`: whether `io_uring`'s requests are traced, and the converter's layout.
    pub fn uring_header(&self) -> Value {
        match &self.uring {
            Ok(layout) => json!({"traced": true, "layout": layout}),
            Err(why) => json!({"traced": false, "why": why}),
        }
    }

    /// The process to trace, by its pid here: the probe adds it at its next call.
    pub fn seed(&mut self, pid: u32) -> Result<()> {
        let mut cfg: aya::maps::Array<_, Config> = aya::maps::Array::try_from(self.bpf.map_mut("CONFIG").expect("CONFIG"))?;
        let mut c = cfg.get(&0, 0)?;
        c.seed = pid;
        cfg.set(0, c, 0)?;
        Ok(())
    }

    /// The reader, on a thread of its own, copying every record to `path` until `stop` is set
    /// and the buffer is empty.
    pub fn read_into(&mut self, path: &Path, header: &Value, stop: Arc<AtomicBool>) -> Result<std::thread::JoinHandle<Result<Reading>>> {
        let rb = aya::maps::RingBuf::try_from(self.bpf.take_map("EVENTS").expect("EVENTS"))?;
        let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path).with_context(|| path.display().to_string())?);
        f.write_all(MAGIC)?;
        f.write_all(serde_json::to_string(header)?.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(std::thread::Builder::new().name("aeiou-capture-reader".into()).spawn(move || read_loop(rb, f, stop))?)
    }

    /// Each CPU's losses, read after the reader has stopped: in all, and those after the last
    /// record it wrote (no `GAP` follows them).
    pub fn losses(&self) -> Result<Vec<Loss>> {
        let m: aya::maps::PerCpuArray<_, Loss> = aya::maps::PerCpuArray::try_from(self.bpf.map("LOSSES").expect("LOSSES"))?;
        Ok(m.get(&0, 0)?.iter().copied().collect())
    }

    /// The processes still in the tree, read after the command has ended: any left behind it
    /// (a daemon it started), since an exited one leaves.
    pub fn traced_at_end(&self) -> Result<usize> {
        let m: aya::maps::HashMap<_, u32, u8> = aya::maps::HashMap::try_from(self.bpf.map("TREE").expect("TREE"))?;
        Ok(m.keys().filter(|k| k.is_ok()).count())
    }

    pub fn attached(&self) -> usize {
        self.links
    }
}

/// What the reader did: records and bytes, and its busiest second.
#[derive(Debug, Default, Clone)]
pub struct Reading {
    pub records: u64,
    pub bytes: u64,
    pub peak_records_per_s: u64,
    pub peak_bytes_per_s: u64,
    pub seconds: f64,
}

fn read_loop(mut rb: aya::maps::RingBuf<aya::maps::MapData>, mut f: std::io::BufWriter<std::fs::File>, stop: Arc<AtomicBool>) -> Result<Reading> {
    use std::os::fd::AsRawFd;
    let fd = rb.as_raw_fd();
    let start = Instant::now();
    let mut r = Reading::default();
    let (mut window, mut w_records, mut w_bytes) = (Instant::now(), 0u64, 0u64);
    loop {
        let stopping = stop.load(Ordering::SeqCst);
        let mut got = false;
        while let Some(item) = rb.next() {
            f.write_all(&item)?;
            r.records += 1;
            r.bytes += item.len() as u64;
            w_records += 1;
            w_bytes += item.len() as u64;
            got = true;
        }
        if window.elapsed() >= Duration::from_secs(1) {
            let s = window.elapsed().as_secs_f64();
            r.peak_records_per_s = r.peak_records_per_s.max((w_records as f64 / s) as u64);
            r.peak_bytes_per_s = r.peak_bytes_per_s.max((w_bytes as f64 / s) as u64);
            (window, w_records, w_bytes) = (Instant::now(), 0, 0);
        }
        if stopping && !got {
            break;
        }
        if !got {
            let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
            unsafe { libc::poll(&mut p, 1, 50) };
        }
    }
    let s = window.elapsed().as_secs_f64();
    if s > 0.0 && r.peak_records_per_s == 0 {
        r.peak_records_per_s = (w_records as f64 / s.max(1e-3)) as u64;
        r.peak_bytes_per_s = (w_bytes as f64 / s.max(1e-3)) as u64;
    }
    f.flush()?;
    r.seconds = start.elapsed().as_secs_f64();
    Ok(r)
}

/// The `losses` of `capture.json` and the buffer to suggest: `lost` per CPU with its first and
/// last time on the monotonic clock, and a `--buffer` that would have held the reader's busiest
/// second's bytes for the longest stretch of losses (at least twice this one), a power of two;
/// and the processes of the command the full tree could not hold (`untraced_processes`).
pub fn losses_record(losses: &[Loss], reading: &Reading, buffer: u64, windows: &[(u32, u64, u64, u64)]) -> Value {
    let total: u64 = losses.iter().map(|l| l.total).sum();
    let longest = windows.iter().map(|w| w.3.saturating_sub(w.2)).max().unwrap_or(0) as f64 / 1e9;
    let need = (reading.peak_bytes_per_s as f64 * longest.max(1.0)) as u64;
    let suggested = need.max(buffer * 2).next_power_of_two().min(1 << 31);
    let per_cpu: Vec<Value> = losses.iter().enumerate().filter(|(_, l)| l.total > 0).map(|(c, l)| json!({"cpu": c, "events": l.total})).collect();
    json!({
        "events": total,
        "per_cpu": per_cpu,
        "windows": windows.iter().map(|(cpu, n, a, b)| json!({"cpu": cpu, "events": n, "first_ns": a, "last_ns": b})).collect::<Vec<_>>(),
        "buffer": buffer,
        "suggested_buffer": (total > 0).then_some(suggested),
        "untraced_processes": losses.iter().map(|l| l.untraced).sum::<u64>(),
        "untraced_rings": losses.iter().map(|l| l.rings).sum::<u64>(),
    })
}

/// Whether a capture's `losses` leave its trace incomplete: an event lost, or a process untraced.
pub fn incomplete(rec: &Value) -> bool {
    rec["events"].as_u64().unwrap_or(0) > 0 || rec["untraced_processes"].as_u64().unwrap_or(0) > 0 || rec["untraced_rings"].as_u64().unwrap_or(0) > 0
}

/// The loud box at the end of a capture that lost events: what was lost and what to do.
pub fn loss_box(rec: &Value) -> String {
    let n = rec["events"].as_u64().unwrap_or(0);
    let u = rec["untraced_processes"].as_u64().unwrap_or(0);
    let cpus: Vec<String> = rec["per_cpu"].as_array().map(|a| a.iter().map(|c| format!("cpu {} ({})", c["cpu"], c["events"])).collect()).unwrap_or_default();
    let mut lines = vec![];
    if n > 0 {
        lines.extend([
            format!("THE TRACE IS INCOMPLETE: {n} event(s) lost, the ring buffer full"),
            format!("on {}", cpus.join(", ")),
            format!("in {} window(s), marked in trace.txt; a call whose open fell in one has no path", rec["windows"].as_array().map_or(0, |a| a.len())),
        ]);
    }
    if u > 0 {
        lines.extend([
            if n > 0 { "and".to_string() } else { "THE TRACE IS INCOMPLETE:".to_string() },
            format!("{u} process(es) of the command not traced: the probes' table of {TREE_PROCESSES} live processes was full"),
        ]);
    }
    let rings = rec["untraced_rings"].as_u64().unwrap_or(0);
    if rings > 0 {
        lines.extend([
            if n > 0 || u > 0 { "and".to_string() } else { "THE TRACE IS INCOMPLETE:".to_string() },
            format!("{rings} io_uring ring(s) of the command whose completions were not traced: the probes' table of {URING_RINGS} rings was full"),
        ]);
    }
    lines.push("capture.json says complete: false. To capture it whole:".to_string());
    if n > 0 {
        lines.extend([
            format!("  1. a larger buffer: --buffer {}", human(rec["suggested_buffer"].as_u64().unwrap_or(0))),
            "  2. -o on faster local storage, if the reader's write rate was the limit (capture.json reader)".to_string(),
        ]);
    }
    lines.push(format!("  {}. --source strace, which loses nothing and slows the application", if n > 0 { 3 } else { 1 }));
    let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let bar = "*".repeat(w + 4);
    let mut s = format!("{bar}\n");
    for l in &lines {
        s += &format!("* {l:<w$} *\n");
    }
    s + &bar
}

/// The most rings the probes' `RINGS` holds (`capture-ebpf/src/main.rs`).
pub const URING_RINGS: u32 = 4096;

/// The most processes the probes' `TREE` holds at once (`capture-ebpf/src/main.rs`).
pub const TREE_PROCESSES: u32 = 65536;

/// Bytes in the `--buffer` notation: `512MiB`, `1GiB`.
pub fn human(b: u64) -> String {
    if b >= 1 << 30 && b % (1 << 30) == 0 {
        format!("{}GiB", b >> 30)
    } else if b >= 1 << 20 && b % (1 << 20) == 0 {
        format!("{}MiB", b >> 20)
    } else if b >= 1 << 10 && b % (1 << 10) == 0 {
        format!("{}KiB", b >> 10)
    } else {
        b.to_string()
    }
}

/// `--buffer`'s value: a size the ring buffer can be, a power of two from 64 KiB (four of the
/// largest record) to 2 GiB.
pub fn parse_buffer(s: &str) -> std::result::Result<u64, String> {
    let t = s.trim();
    let (num, unit) = t.split_at(t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len()));
    let mult: u64 = match unit {
        "" | "B" => 1,
        "KiB" | "K" => 1 << 10,
        "MiB" | "M" => 1 << 20,
        "GiB" | "G" => 1 << 30,
        _ => return Err(format!("--buffer {s}: a size in bytes, KiB, MiB, or GiB")),
    };
    let b = num.parse::<u64>().ok().and_then(|n| n.checked_mul(mult)).ok_or_else(|| format!("--buffer {s}: a size in bytes, KiB, MiB, or GiB"))?;
    if !b.is_power_of_two() || !(64 << 10..=1 << 31).contains(&b) {
        return Err(format!("--buffer {s}: a power of two from 64KiB to 2GiB (the kernel's ring buffer)"));
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_parses_and_marks_strings() {
        let c = calls();
        let by = |n: &str| c.iter().flatten().find(|x| x.name == n).copied().unwrap();
        let openat = by("openat");
        assert_eq!(openat.kinds, "Dpom");
        assert_eq!(spec(&openat), TRACED | 0b10);
        assert_eq!(spec(&by("renameat2")), TRACED | 0b1010);
        assert_eq!(entry_special(spec(&by("pwritev2"))), IOV);
        assert_eq!(exit_special(spec(&by("io_getevents"))), IO_EVENTS);
        assert_eq!(spec(&by("sync")), TRACED);
        // every argument count is the table's
        assert!(c.iter().flatten().all(|x| x.kinds.chars().all(|k| "dDnpsiuxomwcCFaUfrvAIPMbeH_".contains(k))));
    }

    #[test]
    fn btf_offsets_through_anonymous_members() {
        // a hand-made BTF: int, struct s { int a; struct { int b; }; int c:3; }
        let mut types = Vec::new();
        let strs = b"\0int\0s\0a\0b\0c\0".to_vec();
        let push = |t: &mut Vec<u8>, v: &[u32]| v.iter().for_each(|x| t.extend_from_slice(&x.to_le_bytes()));
        push(&mut types, &[1, 1 << 24, 4, 32]); // 1: int
        push(&mut types, &[0, (4 << 24) | 1, 4, 9, 1, 0]); // 2: anon { b at 0 }
        push(&mut types, &[0, 8 << 24, 2]); // 3: a typedef of it
        push(&mut types, &[5, (4 << 24) | (1 << 31) | 3, 12, 7, 1, 0, 0, 3, 32, 11, 1, (3 << 24) | 64]); // 4: s
        let mut b = vec![0x9f, 0xeb, 1, 0];
        push(&mut b, &[24, 0, types.len() as u32, types.len() as u32, strs.len() as u32]);
        b.extend_from_slice(&types);
        b.extend_from_slice(&strs);
        let btf = Btf::parse(&b).unwrap();
        assert_eq!(btf.offset("s", "a"), Some(0));
        assert_eq!(btf.offset("s", "b"), Some(4));
        assert_eq!(btf.offset("s", "c"), None, "a bit field");
        assert_eq!(btf.offset("s", "z"), None);
        assert_eq!(btf.offset("t", "a"), None);
        assert_eq!(btf.size("s"), Some(12));
        assert_eq!(btf.size("t"), None);
        assert!(Btf::parse(b"nope").is_none());
        assert!(Btf::parse(&b[..30]).is_none());
    }

    #[test]
    fn the_kernel_btf_has_the_layout() {
        let (c, _) = kernel_layout().unwrap();
        assert!(c.tgid_off > 0 && c.exit_code_off > 0 && c.tgid_off != c.exit_code_off && c.thread_pid_off > 0);
        assert!(c.numbers_off > c.level_off && c.upid_size >= 12);
        // where a process's living threads are counted, so the tree can let it go
        assert!(c.signal_off > 0 && c.signal_off != c.tgid_off);
    }

    #[test]
    fn buffer_sizes() {
        assert_eq!(parse_buffer("256MiB"), Ok(256 << 20));
        assert_eq!(parse_buffer("2GiB"), Ok(1 << 31));
        assert!(parse_buffer("3MiB").unwrap_err().contains("power of two"));
        assert!(parse_buffer("4GiB").is_err());
        assert!(parse_buffer("32KiB").is_err());
        assert_eq!(parse_buffer("64KiB"), Ok(65536));
        assert_eq!(parse_buffer("1G"), Ok(1 << 30));
        assert!(parse_buffer("ten").unwrap_err().contains("KiB, MiB, or GiB"));
        assert!(parse_buffer("1TiB").is_err());
        assert_eq!(human(1 << 30), "1GiB");
        assert_eq!(human(512 << 20), "512MiB");
        assert_eq!(human(8192), "8KiB");
        assert_eq!(human(100), "100");
    }

    #[test]
    fn the_loss_record_suggests_a_buffer_and_the_box_says_what_to_do() {
        let l = [Loss { total: 0, ..Default::default() }, Loss { total: 7, pending: 0, first: 1, last: 2, ..Default::default() }];
        let r = Reading { peak_bytes_per_s: 600 << 20, ..Default::default() };
        let rec = losses_record(&l, &r, 256 << 20, &[(1, 7, 1_000_000_000, 3_500_000_000)]);
        assert_eq!(rec["events"], 7);
        assert_eq!(rec["per_cpu"][0]["cpu"], 1);
        // 600 MiB/s for 2.5 s is 1.5 GiB: 2 GiB
        assert_eq!(rec["suggested_buffer"], 1u64 << 31);
        let b = loss_box(&rec);
        assert!(b.contains("THE TRACE IS INCOMPLETE: 7 event(s) lost") && b.contains("--buffer 2GiB") && b.contains("--source strace"), "{b}");
        assert!(b.lines().all(|l| l.chars().count() == b.lines().next().unwrap().chars().count()));
        let none = losses_record(&[Loss::default()], &r, 256 << 20, &[]);
        assert_eq!(none["events"], 0);
        assert_eq!(none["untraced_processes"], 0);
        assert!(none["suggested_buffer"].is_null());
        assert!(!incomplete(&none));
        assert!(incomplete(&rec));

        // a child the full tree could not hold: incomplete, and the box says so, with no event lost
        let full = losses_record(&[Loss { untraced: 2, ..Default::default() }, Loss { untraced: 1, ..Default::default() }], &r, 256 << 20, &[]);
        assert_eq!(full["events"], 0);
        assert_eq!(full["untraced_processes"], 3);
        assert!(incomplete(&full));
        let b = loss_box(&full);
        assert!(b.contains("THE TRACE IS INCOMPLETE") && b.contains("3 process(es) of the command not traced"), "{b}");
        assert!(!b.contains("--buffer"), "a larger buffer does not help: {b}");
        assert!(b.lines().all(|l| l.chars().count() == b.lines().next().unwrap().chars().count()));

        // a ring the full table of rings could not hold: its completions are not traced
        let rings = losses_record(&[Loss { rings: 1, ..Default::default() }], &r, 256 << 20, &[]);
        assert_eq!(rings["untraced_rings"], 1);
        assert!(incomplete(&rings));
        let b = loss_box(&rings);
        assert!(b.contains("1 io_uring ring(s) of the command whose completions were not traced"), "{b}");
        assert!(b.lines().all(|l| l.chars().count() == b.lines().next().unwrap().chars().count()));
    }

    #[test]
    fn a_kernel_whose_io_uring_differs_is_said_why_and_not_traced() {
        let raw = std::fs::read("/sys/kernel/btf/vmlinux").unwrap();
        // the kernel's BTF with one thing changed, as another kernel would have it
        let changed = |f: &dyn Fn(&mut Btf)| {
            let mut b = Btf::parse(&raw).unwrap();
            f(&mut b);
            uring_layout(&b, &mut Config::default()).unwrap_err()
        };
        let rename = |b: &mut Btf, kind: u32, from: &str, to: &str| {
            let t = b.types.iter_mut().find(|t| t.1 == kind && t.0 == from).unwrap();
            t.0 = to.into();
        };
        let member = |b: &mut Btf, s: &str, from: &str, to: &str| {
            let t = b.types.iter_mut().find(|t| t.1 == 4 && t.0 == s).unwrap();
            t.3.iter_mut().find(|m| m.0 == from).unwrap().0 = to.into();
        };
        assert!(changed(&|b| rename(b, 8, "btf_trace_io_uring_submit_req", "x")).contains("no io_uring tracepoints"));
        assert!(changed(&|b| rename(b, 8, "btf_trace_io_uring_complete", "x")).contains("no tracepoint io_uring_complete"));
        // the request passed as no struct io_kiocb
        assert!(changed(&|b| rename(b, 4, "io_kiocb", "x")).contains("passes no struct io_kiocb"));
        assert!(changed(&|b| member(b, "io_kiocb", "opcode", "x")).contains("no io_kiocb.opcode"));
        assert!(changed(&|b| rename(b, 4, "io_statx", "x")).contains("no struct io_statx"));
        assert!(changed(&|b| {
            let t = b.types.iter_mut().find(|t| t.1 == 4 && t.0 == "io_rw").unwrap();
            t.4 = 100;
        })
        .contains("larger than the 64 bytes"));
        // the operation's structure not first in the request
        assert!(changed(&|b| {
            let t = b.types.iter_mut().find(|t| t.1 == 4 && t.0 == "io_kiocb").unwrap();
            let m = t.3.iter_mut().find(|m| m.0.is_empty()).unwrap();
            m.2 = 64;
        })
        .contains("does not begin with the operation's structure"));
    }

    #[test]
    fn the_kernel_btf_has_io_uring_s_layout() {
        let p = "/sys/kernel/btf/vmlinux";
        let btf = Btf::parse(&std::fs::read(p).unwrap()).unwrap();
        // this box's kernel (6.18): the request alone, and the posted cqe
        assert_eq!(btf.tracepoint_args("io_uring_submit_req").unwrap(), vec!["io_kiocb"]);
        assert_eq!(btf.tracepoint_args("io_uring_complete").unwrap(), vec!["io_ring_ctx", "", "io_uring_cqe"]);
        assert!(btf.tracepoint_args("no_such_tracepoint").is_none());
        let mut c = Config::default();
        let l = uring_layout(&btf, &mut c).unwrap();
        assert_eq!((c.uring, c.sqe_req_arg, c.cqe_struct), (1, 0, 1));
        assert_eq!(uring_programs(&c), ["io_uring_submit_req", "io_uring_complete", "uring_calls"]);
        let old = Config { sqe_req_arg: 1, cqe_struct: 0, ..c };
        assert_eq!(uring_programs(&old), ["io_uring_submit_req_after_ring", "io_uring_complete_fields", "uring_calls"]);
        assert!(uring_programs(&old).iter().all(|p| URING_PROGRAMS.contains(p)));
        assert_eq!(tracepoint_of("io_uring_complete_fields"), "io_uring_complete");
        assert!(c.req_opcode_off >= CMD_SIZE as u32 && c.req_fd_off > c.req_user_data_off && c.req_ctx_off > 0, "{c:?}");
        assert!(c.task_files_off > 0 && c.fdt_fd_off > 0 && c.file_private_off > 0);
        assert_eq!(l["rw"]["pos"], 8);
        assert_eq!((&l["statx"]["dfd"], c.statx_filename_off), (&json!(8), 24));
        assert!(l["rw"]["len"].as_u64().unwrap() < CMD_SIZE as u64 && l["open"]["flags"].as_u64().unwrap() < CMD_SIZE as u64);
        // the probes' only kernel types: a kernel without io_uring is said so, the capture goes on
        let b = Btf { types: vec![(String::new(), 0, false, Vec::new(), 0)] };
        assert!(uring_layout(&b, &mut Config::default()).unwrap_err().contains("no io_uring tracepoints"));
    }
}
