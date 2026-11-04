//! The records the capture's BPF programs write and `aeiou-trace` reads (one definition for
//! both: the probe crate includes it as a module, and so does `aeiou/src/ebpf.rs`), the maps'
//! values, and the specification of each traced call the reader fills in. Little-endian, every
//! record a multiple of 8 bytes. DESIGN_REVIEW.md §3.96 "Phase 3".

/// A record's kind.
pub const ENTER: u16 = 1;
pub const EXIT: u16 = 2;
/// Records lost on one CPU before this one, which the probe could not write.
pub const GAP: u16 = 3;
/// A traced thread's end (`sched_process_exit`): `Exit` with `nr` 0 and `ret` the task's
/// `exit_code` (a wait status: the exit status shifted, or the signal).
pub const GONE: u16 = 4;
/// A traced thread's new task (`sched_process_fork`, in the parent's thread): `Exit` with
/// `nr` the child's thread id in the command's PID namespace, `pad` 1 for a thread of the
/// parent's process (0 for a process).
pub const FORK: u16 = 5;

/// The start of every record: its kind and whole length, the CPU, the monotonic clock, and the
/// process and thread as the traced command's own PID namespace numbers them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Head {
    pub kind: u16,
    pub len: u16,
    pub cpu: u32,
    pub ts: u64,
    pub pid: u32,
    pub tid: u32,
}

/// A call's entry: its number and six raw arguments, then its blobs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Enter {
    pub head: Head,
    pub nr: u32,
    pub pad: u32,
    pub args: [u64; 6],
}

/// A call's return, then its blobs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Exit {
    pub head: Head,
    pub nr: u32,
    pub pad: u32,
    pub ret: i64,
}

/// `lost` records the CPU could not write, the first at `first` and the last at `last`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Gap {
    pub head: Head,
    pub lost: u64,
    pub first: u64,
    pub last: u64,
}

/// Memory a call's argument names, copied after the record's fixed part: `arg` is the
/// argument's index, `what` one of the kinds below, `len` the bytes that follow (padded to 8).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Blob {
    pub arg: u8,
    pub what: u8,
    pub len: u16,
    pub pad: u32,
}

/// A NUL-terminated string, the NUL not kept.
pub const STR: u8 = 1;
/// A string longer than `PATH_MAX`, cut there.
pub const STR_CUT: u8 = 2;
/// A string the probe could not read (its page not resident); retried at the call's exit.
pub const STR_FAULT: u8 = 3;
/// Bytes of a structure or an array (`what` for `IOV`, `IOCB`, `IO_EVENTS`, `CLONE_ARGS`, `HOW`).
pub const BYTES: u8 = 4;
/// Bytes the probe could not read.
pub const BYTES_FAULT: u8 = 5;

/// The longest string copied, `PATH_MAX`.
pub const PATH_MAX: usize = 4096;
/// The most array elements copied (`strace` prints 32).
pub const MAX_ELEMS: usize = 32;
/// The scratch buffer a record is built in, per CPU: the largest record.
pub const SCRATCH: usize = 16384;

/// What the probe copies of a call, by its number (`SPECS`): bits 0 to 5, argument `i` is a
/// string; bits 8 to 15, a structure copied at entry; bits 16 to 23, one copied at exit; bit
/// 31, the call is traced.
pub const TRACED: u32 = 1 << 31;
pub const fn entry_special(s: u32) -> u32 {
    (s >> 8) & 0xff
}
pub const fn exit_special(s: u32) -> u32 {
    (s >> 16) & 0xff
}
/// `iovcnt` `struct iovec`s at argument 1, the count argument 2 (`readv`, `writev`, the
/// `p*v*` calls).
pub const IOV: u32 = 1;
/// `nr` (argument 1) pointers to `struct iocb` at argument 2 (`io_submit`), each structure
/// copied after the pointers.
pub const IOCB: u32 = 2;
/// `struct clone_args` at argument 0, its size argument 1 (`clone3`).
pub const CLONE_ARGS: u32 = 3;
/// `struct open_how` at argument 2, its size argument 3 (`openat2`).
pub const HOW: u32 = 4;
/// At exit: the return value's count of `struct io_event`s at argument 3 (`io_getevents`,
/// `io_pgetevents`).
pub const IO_EVENTS: u32 = 5;

pub const IOVEC_SIZE: usize = 16;
pub const IOCB_SIZE: usize = 64;
pub const IO_EVENT_SIZE: usize = 32;
pub const CLONE_ARGS_SIZE: usize = 88;
pub const OPEN_HOW_SIZE: usize = 24;

/// The highest call number the maps hold, plus one.
pub const NR_MAX: u32 = 512;

/// The reader's settings (`CONFIG`, index 0): the command's PID namespace, the process the
/// capture starts (in that namespace's numbering; the probe adds it to `TREE` at its first
/// call, finds the namespace's level from it, and clears `seed`), and the kernel structures' offsets the probes read, from
/// the kernel's BTF: `task_struct`'s `tgid`, `exit_code`, and `thread_pid`, `pid`'s `level`
/// and `numbers`, and the size of a `upid` (whose `nr` is first).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    pub ns_dev: u64,
    pub ns_ino: u64,
    pub seed: u32,
    pub ns_level: u32,
    pub tgid_off: u32,
    pub exit_code_off: u32,
    pub thread_pid_off: u32,
    pub level_off: u32,
    pub numbers_off: u32,
    pub upid_size: u32,
}

/// One CPU's losses (`LOSSES`): pending since the last record written, and in all.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Loss {
    pub pending: u64,
    pub first: u64,
    pub last: u64,
    pub total: u64,
}
