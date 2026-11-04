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
    macros::{map, raw_tracepoint},
    maps::{Array, HashMap, PerCpuArray, RingBuf},
    programs::RawTracePointContext,
};
use core::mem::size_of;
use record::*;

#[map]
static EVENTS: RingBuf = RingBuf::with_byte_size(1 << 28, 0);
#[map]
static SPECS: Array<u32> = Array::with_max_entries(NR_MAX, 0);
#[map]
static CONFIG: Array<Config> = Array::with_max_entries(1, 0);
/// The traced processes, by the kernel's (the initial namespace's) process id.
#[map]
static TREE: HashMap<u32, u8> = HashMap::with_max_entries(65536, 0);
/// The string arguments a thread's entry could not copy, retried at its exit, by thread id.
#[map]
static RETRY: HashMap<u32, u32> = HashMap::with_max_entries(65536, 0);
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
            let _ = TREE.insert(&tgid, &1, 0);
            cfg.ns_level = ns_level(cfg, pid);
            cfg.seed = 0;
            true
        }
        _ => false,
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
    let _ = TREE.insert(&c, &1, 0);
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
/// strace's `+++` lines do.
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
    0
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
