//! The capture's BPF programs (`aeiou-trace capture --source ebpf`, DESIGN_REVIEW.md §3.96
//! "Phase 3"): every call of the traced process tree that `SPECS` marks, on the raw
//! tracepoints `sys_enter` and `sys_exit`, as one record at entry and one at exit, built in a
//! per-CPU scratch buffer and copied into the ring buffer `EVENTS`; the tree followed at
//! `sched_process_fork`. A record the ring buffer has no room for is counted in `LOSSES` and
//! reported by the next one the CPU writes (`GAP`): a probe cannot make the application wait.
#![no_std]
#![no_main]

mod record;

use aya_ebpf::{
    bindings::bpf_pidns_info,
    helpers::{
        bpf_get_current_pid_tgid, bpf_get_ns_current_pid_tgid, bpf_get_smp_processor_id, bpf_ktime_get_ns,
        bpf_probe_read_kernel, generated as gen,
    },
    macros::{fentry, fexit, map, raw_tracepoint, uprobe, uretprobe},
    maps::{Array, HashMap, LruHashMap, PerCpuArray, RingBuf},
    programs::{FEntryContext, FExitContext, ProbeContext, RawTracePointContext, RetProbeContext},
};
use core::mem::size_of;
use record::*;

#[map]
static EVENTS: RingBuf = RingBuf::with_byte_size(1 << 28, 0);
#[map]
static SPECS: Array<u32> = Array::with_max_entries(NR_MAX, 0);
#[map]
static CONFIG: Array<Config> = Array::with_max_entries(1, 0);
/// The traced processes, by the kernel's (the initial namespace's) process id; its size is
/// `aeiou::ebpf::TREE_PROCESSES`.
#[map]
static TREE: HashMap<u32, u8> = HashMap::with_max_entries(65536, 0);
/// The string arguments a thread's entry could not copy, retried at its exit, by thread id.
#[map]
static RETRY: HashMap<u32, u32> = HashMap::with_max_entries(65536, 0);
/// The rings traced processes submitted to, by `io_ring_ctx`'s address: a completion is
/// traced by its ring, since it may run in an interrupt or in a thread of no traced process.
#[map]
static RINGS: HashMap<u64, u8> = HashMap::with_max_entries(4096, 0);
/// The libaio requests of traced threads in flight, by the kernel's request: a completion is
/// traced by its request, since it may run in an interrupt or in a thread of no traced process
/// (DESIGN_REVIEW.md §3.98 "Phase 4 step 3"). Least recently used first out, so a request the
/// kernel dropped without a completion (one `io_submit` refused after preparing it) cannot
/// fill it.
#[map]
static AIO_REQS: LruHashMap<u64, u8> = LruHashMap::with_max_entries(65536, 0);
/// A fault between its entry and its return, by thread id.
#[map]
static FAULTS: HashMap<u32, FaultStart> = HashMap::with_max_entries(65536, 0);
#[map]
static LOSSES: PerCpuArray<Loss> = PerCpuArray::with_max_entries(1, 0);
/// Twice the largest record, so a blob at any offset below `SCRATCH` fits without a check the
/// verifier cannot follow.
#[map]
static BUF: PerCpuArray<[u8; 2 * SCRATCH]> = PerCpuArray::with_max_entries(1, 0);

/// Where each argument is in the kernel's `struct pt_regs` (arch/*/include/asm/ptrace.h), at
/// entry and at exit, and the call's number at exit.
#[cfg(bpf_target_arch = "x86_64")]
mod regs {
    // di, si, dx, r10, r8, r9; the registers keep the arguments to the exit
    pub const ENTRY: [usize; 6] = [112, 104, 96, 56, 72, 64];
    pub const EXIT: [usize; 6] = ENTRY;
    pub const NR: usize = 120; // orig_ax
}
#[cfg(bpf_target_arch = "aarch64")]
mod regs {
    // regs[0..6]; at exit regs[0] holds the return value, the first argument is orig_x0
    pub const ENTRY: [usize; 6] = [0, 8, 16, 24, 32, 40];
    pub const EXIT: [usize; 6] = [272, 8, 16, 24, 32, 40];
    pub const NR: usize = 280; // syscallno
}

#[inline(always)]
fn reg(regs: *const u8, off: usize) -> u64 {
    unsafe { bpf_probe_read_kernel(regs.add(off) as *const u64) }.unwrap_or(0)
}

/// The kernel's process id of the current thread when it is in the traced tree; the first
/// call of the process the capture starts adds it.
#[inline(always)]
fn traced() -> bool {
    let tgid = (bpf_get_current_pid_tgid() >> 32) as u32;
    if unsafe { TREE.get(&tgid) }.is_some() {
        return true;
    }
    let Some(cfg) = CONFIG.get_ptr_mut(0) else { return false };
    let cfg = unsafe { &mut *cfg };
    if cfg.seed == 0 {
        return false;
    }
    match ns_ids(cfg) {
        Some((pid, _)) if pid == cfg.seed => {
            if TREE.insert(&tgid, &1, 0).is_err() {
                untraced();
            }
            cfg.ns_level = ns_level(cfg, pid);
            cfg.seed = 0;
            true
        }
        _ => false,
    }
}

/// A process the full `TREE` could not hold, counted: it is not traced, and the capture says so.
#[inline(always)]
fn untraced() {
    if let Some(loss) = LOSSES.get_ptr_mut(0) {
        unsafe { (*loss).untraced += 1 };
    }
}

/// The level of the command's PID namespace, as the kernel numbers levels: where the current
/// task's `pid.numbers[k].nr` is its id in that namespace (`/proc`'s `NSpid` counts only from
/// the namespace `/proc` was mounted in, which WSL2, for one, puts below the kernel's own).
#[inline(always)]
fn ns_level(cfg: &Config, nspid: u32) -> u32 {
    let task = unsafe { aya_ebpf::helpers::bpf_get_current_task() } as *const u8;
    let Ok(pid) = (unsafe { bpf_probe_read_kernel(task.add(cfg.thread_pid_off as usize) as *const u64) }) else { return u32::MAX };
    let pid = pid as *const u8;
    let Ok(level) = (unsafe { bpf_probe_read_kernel(pid.add(cfg.level_off as usize) as *const u32) }) else { return u32::MAX };
    for k in 0..32u32 {
        if k > level {
            break;
        }
        let at = cfg.numbers_off as usize + k as usize * cfg.upid_size as usize;
        if unsafe { bpf_probe_read_kernel(pid.add(at) as *const u32) } == Ok(nspid) {
            return k;
        }
    }
    u32::MAX
}

/// The current thread's process and thread ids in the command's PID namespace.
#[inline(always)]
fn ns_ids(cfg: &Config) -> Option<(u32, u32)> {
    let mut info = bpf_pidns_info { pid: 0, tgid: 0 };
    let r = unsafe { bpf_get_ns_current_pid_tgid(cfg.ns_dev, cfg.ns_ino, &mut info, size_of::<bpf_pidns_info>() as u32) };
    if r != 0 {
        return None;
    }
    Some((info.tgid, info.pid))
}

#[inline(always)]
fn head(kind: u16) -> Head {
    let (pid, tid) = CONFIG.get(0).and_then(ns_ids).unwrap_or((0, 0));
    Head { kind, len: 0, cpu: unsafe { bpf_get_smp_processor_id() }, ts: unsafe { bpf_ktime_get_ns() }, pid, tid }
}

#[inline(always)]
fn pad8(n: usize) -> usize {
    (n + 7) & !7
}

/// A string argument at `off`: its blob, then the bytes; the new offset, and whether it was
/// read.
#[inline(always)]
fn put_str(buf: *mut u8, off: usize, arg: u8, src: u64) -> (usize, bool) {
    let off = off & (SCRATCH - 1);
    let data = unsafe { buf.add(off + size_of::<Blob>()) };
    let n = unsafe { gen::bpf_probe_read_user_str(data as *mut _, (PATH_MAX + 1) as u32, src as *const _) };
    let (what, len) = if n <= 0 {
        (STR_FAULT, 0usize)
    } else if n as usize > PATH_MAX {
        (STR_CUT, PATH_MAX)
    } else {
        (STR, n as usize - 1)
    };
    unsafe { *(buf.add(off) as *mut Blob) = Blob { arg, what, len: len as u16, pad: 0 } };
    (off + size_of::<Blob>() + pad8(len), what != STR_FAULT)
}

/// `len` bytes (at most `max`) of user memory at `src`, as one blob of kind `BYTES`.
#[inline(always)]
fn put_bytes(buf: *mut u8, off: usize, arg: u8, src: u64, len: usize, max: usize) -> usize {
    let off = off & (SCRATCH - 1);
    let len = if len > max { max } else { len };
    let len = len & (SCRATCH - 1);
    let data = unsafe { buf.add(off + size_of::<Blob>()) };
    let r = unsafe { gen::bpf_probe_read_user(data as *mut _, len as u32, src as *const _) };
    let (what, len) = if r == 0 { (BYTES, len) } else { (BYTES_FAULT, 0) };
    unsafe { *(buf.add(off) as *mut Blob) = Blob { arg, what, len: len as u16, pad: 0 } };
    off + size_of::<Blob>() + pad8(len)
}

/// The `struct iocb`s `io_submit` names: up to `MAX_ELEMS`, one after another in one blob.
#[inline(always)]
fn put_iocbs(buf: *mut u8, off: usize, nr: u64, list: u64) -> usize {
    let off = off & (SCRATCH - 1);
    let nr = if nr > MAX_ELEMS as u64 { MAX_ELEMS } else { nr as usize };
    let data = unsafe { buf.add(off + size_of::<Blob>()) };
    let mut got = 0usize;
    for i in 0..MAX_ELEMS {
        if i >= nr {
            break;
        }
        let mut p: u64 = 0;
        if unsafe { gen::bpf_probe_read_user(&mut p as *mut u64 as *mut _, 8, (list + 8 * i as u64) as *const _) } != 0 {
            break;
        }
        if unsafe { gen::bpf_probe_read_user(data.add(i * IOCB_SIZE) as *mut _, IOCB_SIZE as u32, p as *const _) } != 0 {
            break;
        }
        got = i + 1;
    }
    let len = got * IOCB_SIZE;
    unsafe { *(buf.add(off) as *mut Blob) = Blob { arg: 2, what: BYTES, len: len as u16, pad: 0 } };
    off + size_of::<Blob>() + len
}

/// `io_uring_register`'s files: for the opcodes taking a structure, the structure (a blob of
/// argument 2), then the descriptors it or argument 2 points to (argument 3's blob, or
/// argument 2's for `IORING_REGISTER_FILES`), at most `MAX_FILES`.
#[inline(always)]
fn put_files(buf: *mut u8, off: usize, opcode: u64, arg: u64, nr_args: u64) -> usize {
    // IORING_REGISTER_USE_REGISTERED_RING names the ring by its index: the files are the same
    let op = opcode & 0x7fff_ffff;
    let (head, size) = match op {
        6 => (true, 16usize),       // IORING_REGISTER_FILES_UPDATE: struct io_uring_files_update
        13 | 14 => (true, 32usize), // _FILES2: io_uring_rsrc_register; _FILES_UPDATE2: io_uring_rsrc_update2
        2 => (false, 0usize),       // IORING_REGISTER_FILES: the array itself
        _ => return off,
    };
    if !head {
        return put_bytes(buf, off, 2, arg, nr_args as usize * 4, MAX_FILES * 4);
    }
    let mut s = [0u8; 32];
    if unsafe { gen::bpf_probe_read_user(s.as_mut_ptr() as *mut _, size as u32, arg as *const _) } != 0 {
        return off;
    }
    let off = put_bytes(buf, off, 2, arg, size, 32);
    let u32_at = |o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
    let u64_at = |o: usize| u32_at(o) as u64 | (u32_at(o + 4) as u64) << 32;
    let (data, nr) = match op {
        6 => (u64_at(8), nr_args),
        13 => (u64_at(16), u32_at(0) as u64),
        _ => (u64_at(8), u32_at(24) as u64),
    };
    if data == 0 {
        return off;
    }
    put_bytes(buf, off, 3, data, nr as usize * 4, MAX_FILES * 4)
}

/// The ring descriptor `fd` of the current process is: its file's `private_data`, the
/// `io_ring_ctx`, as a blob of argument 0.
#[inline(always)]
fn put_ring(buf: *mut u8, off: usize, fd: u64) -> usize {
    let Some(cfg) = CONFIG.get(0) else { return off };
    if cfg.uring == 0 || fd >= 1 << 20 {
        return off;
    }
    let task = unsafe { aya_ebpf::helpers::bpf_get_current_task() } as *const u8;
    let read = |p: *const u8, o: u32| unsafe { bpf_probe_read_kernel(p.add(o as usize) as *const u64) }.ok().map(|v| v as *const u8);
    let Some(files) = read(task, cfg.task_files_off) else { return off };
    let Some(fdt) = read(files, cfg.files_fdt_off) else { return off };
    let Some(fds) = read(fdt, cfg.fdt_fd_off) else { return off };
    let Some(file) = read(fds, fd as u32 * 8) else { return off };
    let Some(ring) = read(file, cfg.file_private_off) else { return off };
    let off = off & (SCRATCH - 1);
    unsafe {
        *(buf.add(off) as *mut Blob) = Blob { arg: 0, what: BYTES, len: 8, pad: 0 };
        *(buf.add(off + size_of::<Blob>()) as *mut u64) = ring as u64;
    }
    off + size_of::<Blob>() + 8
}

/// The record in `buf`, `len` bytes, into the ring buffer, after a `GAP` for what this CPU
/// lost since its last; a record with no room is counted.
#[inline(always)]
fn emit(buf: *mut u8, len: usize) {
    let Some(loss) = LOSSES.get_ptr_mut(0) else { return };
    let loss = unsafe { &mut *loss };
    let len = len & (SCRATCH - 1);
    unsafe { (*(buf as *mut Head)).len = len as u16 };
    let ts = unsafe { (*(buf as *const Head)).ts };
    let mut ok = true;
    if loss.pending > 0 {
        let mut h = head(GAP);
        h.len = size_of::<Gap>() as u16;
        let gap = Gap { head: h, lost: loss.pending, first: loss.first, last: loss.last };
        ok = EVENTS.output::<Gap>(&gap, 0).is_ok();
        if ok {
            loss.pending = 0;
        }
    }
    if ok {
        let rec = unsafe { core::slice::from_raw_parts(buf as *const u8, len) };
        ok = EVENTS.output::<[u8]>(rec, 0).is_ok();
    }
    if !ok {
        if loss.pending == 0 {
            loss.first = ts;
        }
        loss.pending += 1;
        loss.last = ts;
        loss.total += 1;
    }
}

#[raw_tracepoint(tracepoint = "sys_enter")]
pub fn sys_enter(ctx: RawTracePointContext) -> i32 {
    let nr: i64 = ctx.arg(1);
    if nr < 0 || nr >= NR_MAX as i64 {
        return 0;
    }
    let Some(&spec) = SPECS.get(nr as u32) else { return 0 };
    if spec & TRACED == 0 || !traced() {
        return 0;
    }
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    let regs: *const u8 = ctx.arg(0);
    let mut args = [0u64; 6];
    for (i, a) in args.iter_mut().enumerate() {
        *a = reg(regs, regs::ENTRY[i]);
    }
    unsafe { *(buf as *mut Enter) = Enter { head: head(ENTER), nr: nr as u32, pad: 0, args } };
    let mut off = size_of::<Enter>();
    let mut faulted = 0u32;
    for i in 0..6 {
        if spec & (1 << i) != 0 {
            let (o, ok) = put_str(buf, off, i as u8, args[i]);
            off = o;
            if !ok {
                faulted |= 1 << i;
            }
        }
    }
    match entry_special(spec) {
        IOV => off = put_bytes(buf, off, 1, args[1], args[2] as usize * IOVEC_SIZE, MAX_ELEMS * IOVEC_SIZE),
        IOCB => off = put_iocbs(buf, off, args[1], args[2]),
        CLONE_ARGS => off = put_bytes(buf, off, 0, args[0], args[1] as usize, CLONE_ARGS_SIZE),
        HOW => off = put_bytes(buf, off, 2, args[2], args[3] as usize, OPEN_HOW_SIZE),
        _ => {}
    }
    if faulted != 0 {
        let tid = bpf_get_current_pid_tgid() as u32;
        let _ = RETRY.insert(&tid, &faulted, 0);
    }
    emit(buf, off);
    0
}

#[raw_tracepoint(tracepoint = "sys_exit")]
pub fn sys_exit(ctx: RawTracePointContext) -> i32 {
    let regs: *const u8 = ctx.arg(0);
    let nr = reg(regs, regs::NR) as i64 as i32 as i64;
    if nr < 0 || nr >= NR_MAX as i64 {
        return 0;
    }
    let Some(&spec) = SPECS.get(nr as u32) else { return 0 };
    if spec & TRACED == 0 || !traced() {
        return 0;
    }
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    let ret: i64 = ctx.arg(1);
    unsafe { *(buf as *mut Exit) = Exit { head: head(EXIT), nr: nr as u32, pad: 0, ret } };
    let mut off = size_of::<Exit>();
    if spec & 0x3f != 0 {
        let tid = bpf_get_current_pid_tgid() as u32;
        if let Some(&mask) = unsafe { RETRY.get(&tid) } {
            for i in 0..6 {
                if mask & (1 << i) != 0 {
                    off = put_str(buf, off, i as u8, reg(regs, regs::EXIT[i])).0;
                }
            }
            let _ = RETRY.remove(&tid);
        }
    }
    if exit_special(spec) == IO_EVENTS && ret > 0 {
        off = put_bytes(buf, off, 3, reg(regs, regs::EXIT[3]), ret as usize * IO_EVENT_SIZE, MAX_ELEMS * IO_EVENT_SIZE);
    }

    emit(buf, off);
    0
}

/// A child of a traced process is traced from its first instruction: the hook runs in the
/// parent before the child can run. The record names the child as the command's namespace
/// numbers it (`task->thread_pid->numbers[level].nr`), so the text knows whose child it is.
#[raw_tracepoint(tracepoint = "sched_process_fork")]
pub fn sched_process_fork(ctx: RawTracePointContext) -> i32 {
    let tgid = (bpf_get_current_pid_tgid() >> 32) as u32;
    if unsafe { TREE.get(&tgid) }.is_none() {
        return 0;
    }
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    let child: *const u8 = ctx.arg(1);
    let Ok(c) = (unsafe { bpf_probe_read_kernel(child.add(cfg.tgid_off as usize) as *const u32) }) else { return 0 };
    if TREE.insert(&c, &1, 0).is_err() {
        untraced();
    }
    let mut nstid: u32 = 0;
    if let Ok(pid) = unsafe { bpf_probe_read_kernel(child.add(cfg.thread_pid_off as usize) as *const u64) } {
        let pid = pid as *const u8;
        if let Ok(level) = unsafe { bpf_probe_read_kernel(pid.add(cfg.level_off as usize) as *const u32) } {
            if cfg.ns_level != u32::MAX && level >= cfg.ns_level {
                let at = cfg.numbers_off as usize + cfg.ns_level as usize * cfg.upid_size as usize;
                nstid = unsafe { bpf_probe_read_kernel(pid.add(at) as *const u32) }.unwrap_or(0);
            }
        }
    }
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    unsafe { *(buf as *mut Exit) = Exit { head: head(FORK), nr: nstid, pad: (c == tgid) as u32, ret: 0 } };
    emit(buf, size_of::<Exit>());
    0
}

/// A traced thread's end, with its wait status, so the text says how each one ended as
/// strace's `+++` lines do; the last of a process's threads takes it out of `TREE`, so a
/// process given its recycled pid is not traced (the leader may exit before its threads, so
/// not at the leader's end: when `signal->live`, decremented before this tracepoint, is zero).
#[raw_tracepoint(tracepoint = "sched_process_exit")]
pub fn sched_process_exit(ctx: RawTracePointContext) -> i32 {
    let tgid = (bpf_get_current_pid_tgid() >> 32) as u32;
    if unsafe { TREE.get(&tgid) }.is_none() {
        return 0;
    }
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    let task: *const u8 = ctx.arg(0);
    let code = unsafe { bpf_probe_read_kernel(task.add(cfg.exit_code_off as usize) as *const i32) }.unwrap_or(0);
    unsafe { *(buf as *mut Exit) = Exit { head: head(GONE), nr: 0, pad: 0, ret: code as i64 } };
    emit(buf, size_of::<Exit>());
    if let Ok(signal) = unsafe { bpf_probe_read_kernel(task.add(cfg.signal_off as usize) as *const u64) } {
        let live = unsafe { bpf_probe_read_kernel((signal as *const u8).add(cfg.live_off as usize) as *const i32) };
        if live == Ok(0) {
            let _ = TREE.remove(&tgid);
        }
    }
    0
}

/// An `io_uring` request of a traced process as the kernel takes it, after its `prep` read
/// the submission entry (`io_submit_sqe`), so the request holds the descriptor, offset, and
/// length; an open's or a statx's path read from the kernel's copy (`filename`). The ring is
/// added to `RINGS`, so its completions are traced.
///
/// One program per form of the tracepoint, the loader attaching the one the kernel's BTF gives
/// (a raw program may read no argument past the tracepoint's last, or it does not attach):
/// the request alone, or (before about Linux 5.19) after the ring.
#[raw_tracepoint(tracepoint = "io_uring_submit_req")]
pub fn io_uring_submit_req(ctx: RawTracePointContext) -> i32 {
    let req: u64 = ctx.arg(0);
    submit_req(req as *const u8)
}

#[raw_tracepoint(tracepoint = "io_uring_submit_req")]
pub fn io_uring_submit_req_after_ring(ctx: RawTracePointContext) -> i32 {
    let req: u64 = ctx.arg(1);
    submit_req(req as *const u8)
}

#[inline(always)]
fn submit_req(req: *const u8) -> i32 {
    if !traced() {
        return 0;
    }
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    if cfg.uring == 0 {
        return 0;
    }
    let rd64 = |o: u32| unsafe { bpf_probe_read_kernel(req.add(o as usize) as *const u64) }.unwrap_or(0);
    let ring = rd64(cfg.req_ctx_off);
    if RINGS.insert(&ring, &1, 0).is_err() {
        if let Some(loss) = LOSSES.get_ptr_mut(0) {
            unsafe { (*loss).rings += 1 };
        }
    }
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    let opcode = unsafe { bpf_probe_read_kernel(req.add(cfg.req_opcode_off as usize)) }.unwrap_or(0u8);
    let fd = unsafe { bpf_probe_read_kernel(req.add(cfg.req_fd_off as usize) as *const i32) }.unwrap_or(-1);
    let rec = buf as *mut Sqe;
    unsafe {
        (*rec).head = head(URING_SQE);
        (*rec).ring = ring;
        (*rec).req = req as u64;
        (*rec).user_data = rd64(cfg.req_user_data_off);
        (*rec).flags = rd64(cfg.req_flags_off);
        (*rec).fd = fd;
        (*rec).opcode = opcode;
        (*rec).pad = [0; 3];
        let cmd = &raw mut (*rec).cmd;
        if gen::bpf_probe_read_kernel(cmd as *mut _, CMD_SIZE as u32, req as *const _) != 0 {
            *cmd = [0; CMD_SIZE];
        }
    }
    let mut off = size_of::<Sqe>();
    if opcode == IORING_OP_OPENAT || opcode == IORING_OP_OPENAT2 || opcode == IORING_OP_STATX {
        let filename = rd64(if opcode == IORING_OP_STATX { cfg.statx_filename_off } else { cfg.open_filename_off }) as *const u8;
        if let Ok(name) = unsafe { bpf_probe_read_kernel(filename.add(cfg.filename_name_off as usize) as *const u64) } {
            let o = off & (SCRATCH - 1);
            let data = unsafe { buf.add(o + size_of::<Blob>()) };
            let n = unsafe { gen::bpf_probe_read_kernel_str(data as *mut _, (PATH_MAX + 1) as u32, name as *const _) };
            let (what, len) = if n <= 0 {
                (STR_FAULT, 0usize)
            } else if n as usize > PATH_MAX {
                (STR_CUT, PATH_MAX)
            } else {
                (STR, n as usize - 1)
            };
            unsafe { *(buf.add(o) as *mut Blob) = Blob { arg: 0, what, len: len as u16, pad: 0 } };
            off = o + size_of::<Blob>() + pad8(len);
        }
    }
    emit(buf, off);
    0
}

/// A completion on a ring a traced process submitted to, whatever thread or interrupt posts
/// it: the tracepoint passes the `struct io_uring_cqe` posted or, on older kernels, its fields
/// one by one, a program for each (`io_uring_submit_req`'s note).
#[raw_tracepoint(tracepoint = "io_uring_complete")]
pub fn io_uring_complete(ctx: RawTracePointContext) -> i32 {
    let ring: u64 = ctx.arg(0);
    if !uring_traced(ring) {
        return 0;
    }
    let req: u64 = ctx.arg(1);
    let cqe: *const u8 = ctx.arg(2);
    let ud = unsafe { bpf_probe_read_kernel(cqe as *const u64) }.unwrap_or(0);
    let res = unsafe { bpf_probe_read_kernel(cqe.add(8) as *const i32) }.unwrap_or(0);
    let fl = unsafe { bpf_probe_read_kernel(cqe.add(12) as *const u32) }.unwrap_or(0);
    complete(ring, req, ud, res, fl)
}

#[raw_tracepoint(tracepoint = "io_uring_complete")]
pub fn io_uring_complete_fields(ctx: RawTracePointContext) -> i32 {
    let ring: u64 = ctx.arg(0);
    if !uring_traced(ring) {
        return 0;
    }
    let (req, ud, res, fl): (u64, u64, u64, u64) = (ctx.arg(1), ctx.arg(2), ctx.arg(3), ctx.arg(4));
    complete(ring, req, ud, res as i32, fl as u32)
}

#[inline(always)]
fn uring_traced(ring: u64) -> bool {
    CONFIG.get(0).is_some_and(|c| c.uring != 0) && unsafe { RINGS.get(&ring) }.is_some()
}

#[inline(always)]
fn complete(ring: u64, req: u64, user_data: u64, res: i32, cflags: u32) -> i32 {
    // built on this program's stack: an interrupt's completion must not write the per-CPU
    // buffer a call's probe it interrupted is filling
    let rec = Cqe { head: Head { len: size_of::<Cqe>() as u16, ..head(URING_CQE) }, ring, req, user_data, res, cflags };
    let Some(loss) = LOSSES.get_ptr_mut(0) else { return 0 };
    let loss = unsafe { &mut *loss };
    if EVENTS.output::<Cqe>(&rec, 0).is_err() {
        if loss.pending == 0 {
            loss.first = rec.head.ts;
        }
        loss.pending += 1;
        loss.last = rec.head.ts;
        loss.total += 1;
    }
    0
}

/// The return of a traced `io_uring_setup` (the ring its descriptor is) or `io_uring_register`
/// (the files it registers), as a record of its own: a program apart from `sys_exit`'s, so
/// neither grows past what the verifier takes, attached only with `io_uring`'s probes.
#[raw_tracepoint(tracepoint = "sys_exit")]
pub fn uring_calls(ctx: RawTracePointContext) -> i32 {
    let regs: *const u8 = ctx.arg(0);
    let nr = reg(regs, regs::NR) as i64 as i32;
    if nr != NR_IO_URING_SETUP as i32 && nr != NR_IO_URING_REGISTER as i32 {
        return 0;
    }
    let ret: i64 = ctx.arg(1);
    if ret < 0 || !traced() {
        return 0;
    }
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    let mut args = [0u64; 6];
    for (i, a) in args.iter_mut().enumerate() {
        *a = reg(regs, regs::EXIT[i]);
    }
    unsafe { *(buf as *mut UringCall) = UringCall { head: head(URING_CALL), nr: nr as u32, pad: 0, ret, args } };
    let off = size_of::<UringCall>();
    let off = if nr == NR_IO_URING_SETUP as i32 { put_ring(buf, off, ret as u64) } else { put_files(buf, off, args[1], args[2], args[3]) };
    emit(buf, off);
    0
}

/// A page fault of a traced process on a file's mapping (DESIGN_REVIEW.md §3.98 "Phase 4 step
/// 2"), at `handle_mm_fault(vma, address, flags, regs)`'s entry: kept when the fault came from
/// an exception (`regs` set: the application's load or store, or a call's copy of its buffer;
/// a `get_user_pages`, an `O_DIRECT` buffer's pinning, passes none) on a mapping of a file that
/// is not code. The mapping's fields are read here, since a fault that drops the mmap lock may
/// return with the mapping gone.
#[fentry(function = "handle_mm_fault")]
pub fn fault_entry(ctx: FEntryContext) -> i32 {
    let regs: u64 = ctx.arg(3);
    if regs == 0 || !traced() {
        return 0;
    }
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    if cfg.faults == 0 {
        return 0;
    }
    let vma: *const u8 = ctx.arg(0);
    let rd = |p: *const u8, o: u32| unsafe { bpf_probe_read_kernel(p.add(o as usize) as *const u64) }.unwrap_or(0);
    let file = rd(vma, cfg.vma_file_off) as *const u8;
    if file.is_null() || rd(vma, cfg.vma_flags_off) & VM_EXEC != 0 {
        return 0;
    }
    let inode = rd(file, cfg.file_inode_off) as *const u8;
    let sb = rd(inode, cfg.inode_sb_off) as *const u8;
    let dev = unsafe { bpf_probe_read_kernel(sb.add(cfg.sb_dev_off as usize) as *const u32) }.unwrap_or(0);
    let flags: u32 = ctx.arg(2);
    let address: u64 = ctx.arg(1);
    let start = FaultStart { ts: unsafe { bpf_ktime_get_ns() }, address, vm_start: rd(vma, cfg.vma_start_off), pgoff: rd(vma, cfg.vma_pgoff_off), ino: rd(inode, cfg.inode_ino_off), dev, flags };
    let tid = bpf_get_current_pid_tgid() as u32;
    let _ = FAULTS.insert(&tid, &start, 0);
    0
}

/// The fault's return: its record, the `VM_FAULT_` bits it returned (the last of the
/// arguments' slots, so no helper newer than the trampolines is needed) and its end.
#[fexit(function = "handle_mm_fault")]
pub fn fault_exit(ctx: FExitContext) -> i32 {
    let tid = bpf_get_current_pid_tgid() as u32;
    let Some(s) = (unsafe { FAULTS.get(&tid) }).copied() else { return 0 };
    let _ = FAULTS.remove(&tid);
    let ret: u64 = ctx.arg(4);
    let mut h = head(FAULT);
    h.len = size_of::<Fault>() as u16;
    h.ts = s.ts;
    let rec = Fault { head: h, address: s.address, vm_start: s.vm_start, pgoff: s.pgoff, ino: s.ino, dev: s.dev, flags: s.flags, ret: ret as u32, pad: 0, end: unsafe { bpf_ktime_get_ns() } };
    let Some(loss) = LOSSES.get_ptr_mut(0) else { return 0 };
    let loss = unsafe { &mut *loss };
    if EVENTS.output::<Fault>(&rec, 0).is_err() {
        if loss.pending == 0 {
            loss.first = rec.head.ts;
        }
        loss.pending += 1;
        loss.last = rec.head.ts;
        loss.total += 1;
    }
    0
}

/// A libaio request of a traced thread as `io_submit` prepares it, a read or a write
/// (`aio_prep_rw(struct kiocb *, const struct iocb *)`, the request's `kiocb` its first member,
/// its context and `aio_data` set before): added to `AIO_REQS`, so its completion is traced,
/// and recorded, so the converter numbers it by its place in the call's array.
#[fentry(function = "aio_prep_rw")]
pub fn aio_prep_rw(ctx: FEntryContext) -> i32 {
    let req: *const u8 = ctx.arg(0);
    aio_issue(req)
}

/// The same of an fsync (`aio_fsync(struct fsync_iocb *, ...)`, the request's first member).
#[fentry(function = "aio_fsync")]
pub fn aio_fsync(ctx: FEntryContext) -> i32 {
    let req: *const u8 = ctx.arg(0);
    aio_issue(req)
}

#[inline(always)]
fn aio_issue(req: *const u8) -> i32 {
    if !traced() {
        return 0;
    }
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    if cfg.aio == 0 {
        return 0;
    }
    let key = req as u64;
    let _ = AIO_REQS.insert(&key, &1, 0);
    aio_emit(cfg, AIO_ISSUE, req, 0)
}

/// A libaio request's completion as the kernel posts it to the context's ring
/// (`aio_complete(struct aio_kiocb *)`, its result in `ki_res`), whatever thread or interrupt
/// posts it: kept when the request is one a traced thread issued.
#[fentry(function = "aio_complete")]
pub fn aio_complete(ctx: FEntryContext) -> i32 {
    let req: *const u8 = ctx.arg(0);
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    let res = unsafe { bpf_probe_read_kernel(req.add(cfg.ki_res_off as usize + 16) as *const i64) }.unwrap_or(0);
    aio_done(cfg, req, res)
}

/// The same where the kernel's BTF has no `aio_complete` (inlined): `aio_complete_rw(struct
/// kiocb *, long res)`, a read's or a write's completion alone.
#[fentry(function = "aio_complete_rw")]
pub fn aio_complete_rw(ctx: FEntryContext) -> i32 {
    let req: *const u8 = ctx.arg(0);
    let res: i64 = ctx.arg(1);
    let Some(cfg) = CONFIG.get(0) else { return 0 };
    aio_done(cfg, req, res)
}

#[inline(always)]
fn aio_done(cfg: &Config, req: *const u8, res: i64) -> i32 {
    let key = req as u64;
    if cfg.aio == 0 || unsafe { AIO_REQS.get(&key) }.is_none() {
        return 0;
    }
    let _ = AIO_REQS.remove(&key);
    aio_emit(cfg, AIO_DONE, req, res)
}

/// An `AioRec` of `req`, built on this program's stack: an interrupt's completion must not
/// write the per-CPU buffer a call's probe it interrupted is filling.
#[inline(always)]
fn aio_emit(cfg: &Config, kind: u16, req: *const u8, res: i64) -> i32 {
    let rd = |p: *const u8, o: usize| unsafe { bpf_probe_read_kernel(p.add(o) as *const u64) }.unwrap_or(0);
    let kctx = rd(req, cfg.ki_ctx_off as usize) as *const u8;
    let rec = AioRec { head: Head { len: size_of::<AioRec>() as u16, ..head(kind) }, req: req as u64, ctx: rd(kctx, cfg.ctx_user_id_off as usize), data: rd(req, cfg.ki_res_off as usize), res };
    let Some(loss) = LOSSES.get_ptr_mut(0) else { return 0 };
    let loss = unsafe { &mut *loss };
    if EVENTS.output::<AioRec>(&rec, 0).is_err() {
        if loss.pending == 0 {
            loss.first = rec.head.ts;
        }
        loss.pending += 1;
        loss.last = rec.head.ts;
        loss.total += 1;
    }
    0
}

/// A POSIX AIO function of the C library called by a traced thread (a uprobe on its entry,
/// DESIGN_REVIEW.md §3.98 "Phase 4 step 4"): its arguments, and the `struct aiocb`s they name
/// (`aio_read`'s and `aio_write`'s first, `aio_fsync`'s second, a `lio_listio`'s list and each
/// of its entries), `aio_suspend`'s list and timeout, as they are at the call: the converter
/// matches each request to the helper thread's call that serves it.
#[inline(always)]
fn paio_entry(ctx: &ProbeContext, func: u32) -> u32 {
    if !traced() {
        return 0;
    }
    let Some(buf) = BUF.get_ptr_mut(0) else { return 0 };
    let buf = buf as *mut u8;
    let mut args = [0u64; 6];
    for (i, a) in args.iter_mut().enumerate() {
        *a = ctx.arg(i).unwrap_or(0);
    }
    unsafe { *(buf as *mut Enter) = Enter { head: head(PAIO_ENTER), nr: func, pad: 0, args } };
    let mut off = size_of::<Enter>();
    match func {
        PAIO_READ | PAIO_WRITE => off = put_bytes(buf, off, 0, args[0], AIOCB_SIZE, AIOCB_SIZE),
        PAIO_FSYNC => off = put_bytes(buf, off, 1, args[1], AIOCB_SIZE, AIOCB_SIZE),
        PAIO_LIO_LISTIO => {
            off = put_bytes(buf, off, 1, args[1], args[2] as usize * 8, MAX_ELEMS * 8);
            off = put_aiocbs(buf, off, args[2], args[1]);
        }
        PAIO_SUSPEND => {
            off = put_bytes(buf, off, 0, args[0], args[1] as usize * 8, MAX_ELEMS * 8);
            if args[2] != 0 {
                off = put_bytes(buf, off, 2, args[2], TIMESPEC_SIZE, TIMESPEC_SIZE);
            }
        }
        _ => {}
    }
    emit(buf, off);
    0
}

/// The `struct aiocb`s a `lio_listio`'s `nr` entries at `list` point to, up to `MAX_ELEMS`, one
/// after another in one blob; an entry not read (a null pointer) marked `LIO_NOP`, so each keeps
/// its place.
#[inline(always)]
fn put_aiocbs(buf: *mut u8, off: usize, nr: u64, list: u64) -> usize {
    let off = off & (SCRATCH - 1);
    let nr = if nr > MAX_ELEMS as u64 { MAX_ELEMS } else { nr as usize };
    let data = unsafe { buf.add(off + size_of::<Blob>()) };
    let mut got = 0usize;
    for i in 0..MAX_ELEMS {
        if i >= nr {
            break;
        }
        let mut p: u64 = 0;
        if unsafe { gen::bpf_probe_read_user(&mut p as *mut u64 as *mut _, 8, (list + 8 * i as u64) as *const _) } != 0 {
            break;
        }
        let at = unsafe { data.add(i * AIOCB_SIZE) };
        if p == 0 || unsafe { gen::bpf_probe_read_user(at as *mut _, AIOCB_SIZE as u32, p as *const _) } != 0 {
            // LIO_NOP in aio_lio_opcode
            unsafe { *(at.add(4) as *mut u32) = 2 };
        }
        got = i + 1;
    }
    let len = got * AIOCB_SIZE;
    unsafe { *(buf.add(off) as *mut Blob) = Blob { arg: AIOCB_LIST, what: BYTES, len: len as u16, pad: 0 } };
    off + size_of::<Blob>() + len
}

/// Its return (a uretprobe): what it returned, the record built on this program's stack.
#[inline(always)]
fn paio_return(ctx: &RetProbeContext, func: u32) -> u32 {
    if !traced() {
        return 0;
    }
    let ret: i64 = ctx.ret();
    let rec = Exit { head: Head { len: size_of::<Exit>() as u16, ..head(PAIO_EXIT) }, nr: func, pad: 0, ret };
    let Some(loss) = LOSSES.get_ptr_mut(0) else { return 0 };
    let loss = unsafe { &mut *loss };
    if EVENTS.output::<Exit>(&rec, 0).is_err() {
        if loss.pending == 0 {
            loss.first = rec.head.ts;
        }
        loss.pending += 1;
        loss.last = rec.head.ts;
        loss.total += 1;
    }
    0
}

/// A uprobe and a uretprobe of each POSIX AIO function, named as `aeiou::ebpf::paio_program`
/// names them.
macro_rules! paio {
    ($entry:ident, $ret:ident, $func:expr) => {
        #[uprobe]
        pub fn $entry(ctx: ProbeContext) -> u32 {
            paio_entry(&ctx, $func)
        }
        #[uretprobe]
        pub fn $ret(ctx: RetProbeContext) -> u32 {
            paio_return(&ctx, $func)
        }
    };
}
paio!(paio_aio_read, paio_aio_read_ret, PAIO_READ);
paio!(paio_aio_write, paio_aio_write_ret, PAIO_WRITE);
paio!(paio_aio_fsync, paio_aio_fsync_ret, PAIO_FSYNC);
paio!(paio_lio_listio, paio_lio_listio_ret, PAIO_LIO_LISTIO);
paio!(paio_aio_return, paio_aio_return_ret, PAIO_RETURN);
paio!(paio_aio_suspend, paio_aio_suspend_ret, PAIO_SUSPEND);
paio!(paio_aio_cancel, paio_aio_cancel_ret, PAIO_CANCEL);

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
