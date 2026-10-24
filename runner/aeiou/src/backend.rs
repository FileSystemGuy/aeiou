//! I/O backends: how an op of the abstract is issued to the system under test. The op stream
//! is the same for every backend (`PROJECT_BRIEF.md` §4); a backend maps each op to an API.
//!
//! This is the blocking half: `sync` (buffered POSIX syscalls on the calling thread, which
//! under `aeiou run` is one OS thread per actor or sub-actor: the fidelity reference, what a
//! PyTorch worker does) and `sync-direct` (the same with `O_DIRECT` on every regular-file
//! open; reads and writes must then be 4 KiB-aligned in offset and length, and the runner's
//! buffers are).
//!
//! Two more blocking backends live here, on the same thread per actor:
//! `posix-aio` (glibc's `aio_read`/`aio_write`/`aio_fsync` with `aio_suspend`: a hand-off to
//! glibc's user-space thread pool and back) and `mmap` (a read makes its range of a shared
//! mapping of the file resident, so the bytes arrive by page fault; `MmapMode` picks the
//! prefetch and `MmapConsume` whether the range is touched or copied out). Both issue every read and write at the VM's effective offset (`positional`),
//! as the event loops do. The event-loop backends are `uring.rs` (`io_uring`) and `aio.rs`
//! (`libaio`, the kernel AIO system calls).

use std::ffi::CString;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::ast::{Advice, IoctlRequest, OpenFlag, Whence};

/// The API a run issues its POSIX ops through (`DESIGN_REVIEW.md` §3.65): one axis of a
/// backend, the cache mode the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    /// Blocking syscalls on the calling thread.
    Sync,
    /// glibc POSIX AIO, one request at a time per actor thread.
    PosixAio,
    /// The kernel AIO system calls (`io_setup`/`io_submit`/`io_getevents`, what the libaio
    /// library wraps) on the event loop (`aio.rs`).
    LibAio,
    /// `io_uring` through the `io-uring` crate: an event loop per thread multiplexing many
    /// actors over one ring (`uring.rs`).
    Uring,
    /// Reads are copies out of a shared mapping; everything else is `sync`.
    Mmap,
}

/// The run-wide cache mode: `PerOpen`, each open's own flags decide whether it bypasses the
/// page cache, or `Direct`, `O_DIRECT` forced on every regular-file open (another workload,
/// `DESIGN_REVIEW.md` §3.48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cache {
    PerOpen,
    Direct,
}

/// A run's POSIX backend: an API and a cache mode, given apart (`--posix`, `--cache`). The constants
/// below are the code's and the tests' shorthand for the valid pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendKind {
    pub api: Api,
    pub cache: Cache,
}

#[allow(non_upper_case_globals)]
impl BackendKind {
    pub const Sync: BackendKind = BackendKind { api: Api::Sync, cache: Cache::PerOpen };
    pub const SyncDirect: BackendKind = BackendKind { api: Api::Sync, cache: Cache::Direct };
    pub const Uring: BackendKind = BackendKind { api: Api::Uring, cache: Cache::PerOpen };
    pub const UringDirect: BackendKind = BackendKind { api: Api::Uring, cache: Cache::Direct };
    pub const PosixAio: BackendKind = BackendKind { api: Api::PosixAio, cache: Cache::PerOpen };
    pub const PosixAioDirect: BackendKind = BackendKind { api: Api::PosixAio, cache: Cache::Direct };
    pub const LibAio: BackendKind = BackendKind { api: Api::LibAio, cache: Cache::PerOpen };
    pub const LibAioDirect: BackendKind = BackendKind { api: Api::LibAio, cache: Cache::Direct };
    pub const Mmap: BackendKind = BackendKind { api: Api::Mmap, cache: Cache::PerOpen };
}

pub const API_NAMES: &str = "sync, posix-aio, libaio, io_uring, mmap";
pub const CACHE_NAMES: &str = "per-open, direct";
pub const S3_NAMES: &str = "blocking, async";

/// How the application's S3 client uses the store (`--s3`, `DESIGN_REVIEW.md` §3.65): the
/// API axis of the `s3` protocol. Either way the requests are made by `object_store` on the
/// engine's tokio runtime (`object.rs`); the value says how the actors wait for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3Api {
    /// A thread per actor, blocked on each request (boto3).
    Blocking,
    /// Event loops, many actors' requests in flight per thread (aiobotocore, the CRT).
    Async,
}

impl S3Api {
    pub fn parse(s: &str) -> Option<S3Api> {
        match s {
            "blocking" => Some(S3Api::Blocking),
            "async" => Some(S3Api::Async),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            S3Api::Blocking => "blocking",
            S3Api::Async => "async",
        }
    }

    pub fn event_loop(self) -> bool {
        self == S3Api::Async
    }
}

/// The S3 API an abstract declares: its `s3`, `blocking` when it names none.
pub fn declared_s3(ast: &crate::ast::Ast) -> anyhow::Result<S3Api> {
    match &ast.s3 {
        None => Ok(S3Api::Blocking),
        Some(s) => S3Api::parse(s).ok_or_else(|| anyhow::anyhow!("s3 `{s}` is not one of {S3_NAMES}")),
    }
}

impl Api {
    pub fn parse(s: &str) -> Option<Api> {
        match s {
            "sync" => Some(Api::Sync),
            "posix-aio" => Some(Api::PosixAio),
            "libaio" => Some(Api::LibAio),
            "io_uring" => Some(Api::Uring),
            "mmap" => Some(Api::Mmap),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Api::Sync => "sync",
            Api::PosixAio => "posix-aio",
            Api::LibAio => "libaio",
            Api::Uring => "io_uring",
            Api::Mmap => "mmap",
        }
    }
}

impl Cache {
    pub fn parse(s: &str) -> Option<Cache> {
        match s {
            "per-open" => Some(Cache::PerOpen),
            "direct" => Some(Cache::Direct),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Cache::PerOpen => "per-open",
            Cache::Direct => "direct",
        }
    }
}

/// What the `mmap` backend does before it touches (or copies) a read's range of the mapping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmapMode {
    /// Nothing: the touch faults the pages in (fault-around and readahead decide the I/O).
    #[default]
    Fault,
    /// `madvise(MADV_POPULATE_READ)` over the range (Linux 5.14): the pages are read and
    /// mapped in one call, and an I/O error is an errno instead of a `SIGBUS`. Nothing is
    /// touched afterwards: the call returns when the range is resident and mapped.
    Populate,
    /// `madvise(MADV_WILLNEED)` over the range: readahead is started, then the touch waits
    /// for the pages.
    WillNeed,
}

impl MmapMode {
    pub fn parse(s: &str) -> Option<MmapMode> {
        match s {
            "fault" => Some(MmapMode::Fault),
            "populate" => Some(MmapMode::Populate),
            "willneed" => Some(MmapMode::WillNeed),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            MmapMode::Fault => "fault",
            MmapMode::Populate => "populate",
            MmapMode::WillNeed => "willneed",
        }
    }
}

/// How the `mmap` backend consumes a read's range once it is mapped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmapConsume {
    /// Read one byte of every page of the range: each page is resident and mapped when the
    /// read returns, and nothing is copied. What the API itself costs; the consumer's own
    /// use of the bytes (a copy to a device, a decode) is the application's, as it is after
    /// a `read(2)` or an `O_DIRECT` read.
    #[default]
    Touch,
    /// Copy the range into the actor's buffer: a loader that copies out of the mapping.
    Copy,
}

impl MmapConsume {
    pub fn parse(s: &str) -> Option<MmapConsume> {
        match s {
            "touch" => Some(MmapConsume::Touch),
            "copy" => Some(MmapConsume::Copy),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            MmapConsume::Touch => "touch",
            MmapConsume::Copy => "copy",
        }
    }
}

/// What the `mmap` backends of a run count, shared by every actor thread.
#[derive(Debug, Default)]
pub struct MmapStats {
    pub maps: AtomicU64,
    pub mapped_bytes: AtomicU64,
    pub advised: AtomicU64,
    pub touched_pages: AtomicU64,
    pub copied_bytes: AtomicU64,
    /// The `fstat` calls the backend issued itself to learn a mapping's length: one per open
    /// that is read before the abstract asks its size, none where it does.
    pub sized: AtomicU64,
}

impl BackendKind {
    /// The pair, unless it is one no backend is: `O_DIRECT` under `mmap`, whose reads are
    /// page faults on a mapping (`DESIGN_REVIEW.md` §3.65).
    pub fn of(api: Api, cache: Cache) -> Option<BackendKind> {
        (api != Api::Mmap || cache == Cache::PerOpen).then_some(BackendKind { api, cache })
    }

    /// What a run prints and records: `posix API  cache MODE`.
    pub fn describe(self) -> String {
        format!("posix {}  cache {}", self.api.name(), self.cache.name())
    }

    /// `O_DIRECT` on every regular-file open.
    pub fn direct(self) -> bool {
        self.cache == Cache::Direct
    }

    /// Runs on the event loop (`uring.rs`) rather than one thread per actor.
    pub fn uring(self) -> bool {
        self.api == Api::Uring
    }

    /// The kernel AIO system calls on the event loop (`aio.rs`).
    pub fn libaio(self) -> bool {
        self.api == Api::LibAio
    }

    /// Reads through a shared mapping.
    pub fn mmap(self) -> bool {
        self.api == Api::Mmap
    }

    /// One event loop per thread multiplexing the actors, rather than one thread per actor.
    pub fn event_loop(self) -> bool {
        self.uring() || self.libaio()
    }

    /// The blocking form: the backend itself for `sync`, `posix-aio`, and `mmap`, and what an
    /// event loop uses inline for the ops its API cannot carry.
    pub fn make(self, mmap: MmapMode, consume: MmapConsume, stats: &Arc<MmapStats>) -> Box<dyn Backend> {
        let sync = Sync { direct: self.direct() };
        match self.api {
            Api::PosixAio => Box::new(PosixAio { sync }),
            Api::Mmap => Box::new(Mmap { sync, mode: mmap, consume, stats: stats.clone() }),
            Api::Sync | Api::LibAio | Api::Uring => Box::new(sync),
        }
    }
}

/// The POSIX backend an abstract declares: its `posix` and `cache`, `sync` and `per-open` when
/// it names neither (`aeiou check` has refused names that are not, and the pair no backend is).
pub fn declared(ast: &crate::ast::Ast) -> anyhow::Result<BackendKind> {
    let api = match &ast.posix {
        None => Api::Sync,
        Some(a) => Api::parse(a).ok_or_else(|| anyhow::anyhow!("posix `{a}` is not one of {API_NAMES}"))?,
    };
    let cache = match &ast.cache {
        None => Cache::PerOpen,
        Some(c) => Cache::parse(c).ok_or_else(|| anyhow::anyhow!("cache `{c}` is not one of {CACHE_NAMES}"))?,
    };
    BackendKind::of(api, cache).ok_or_else(|| anyhow::anyhow!("cache `direct` under posix `mmap`"))
}

pub const ALIGN: usize = 4096;

pub trait Backend: Send {
    /// The backend's API has no file position: the driver passes every read and write its
    /// effective offset (the one the VM computed and the fingerprint hashes), as the event
    /// loops do (`uring.rs`).
    fn positional(&self) -> bool {
        false
    }
    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OpenFile>;
    /// `read(2)` at the file position, or `pread(2)` at `offset`.
    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize>;
    fn write(&mut self, file: &OpenFile, buf: &[u8], offset: Option<i64>) -> io::Result<usize>;
    fn lseek(&mut self, file: &OpenFile, offset: i64, whence: Whence) -> io::Result<i64>;
    fn ioctl(&mut self, file: &OpenFile, request: IoctlRequest) -> io::Result<()>;
    /// `posix_fadvise(2)` over `[offset, offset + len)` (`len` 0: to the end of the file).
    fn fadvise(&mut self, file: &OpenFile, offset: i64, len: i64, advice: Advice) -> io::Result<()>;
    /// Returns `st_size`.
    fn fstat(&mut self, file: &OpenFile) -> io::Result<i64>;
    fn stat(&mut self, path: &Path) -> io::Result<i64>;
    fn fsync(&mut self, file: &OpenFile) -> io::Result<()>;
    fn fdatasync(&mut self, file: &OpenFile) -> io::Result<()>;
    fn unlink(&mut self, path: &Path) -> io::Result<()>;
    fn ftruncate(&mut self, file: &OpenFile, len: i64) -> io::Result<()>;
    fn fallocate(&mut self, file: &OpenFile, offset: i64, len: i64) -> io::Result<()>;
    fn mkdir(&mut self, path: &Path, mode: u32) -> io::Result<()>;
    fn rmdir(&mut self, path: &Path) -> io::Result<()>;
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()>;
    /// Read the whole directory through `getdents64`; returns the number of entries, not
    /// counting `.`, `..`, and `.aeiou*` (`schema/README.md` §6).
    fn readdir(&mut self, file: &OpenFile) -> io::Result<usize>;
}

/// `flag_bits` of `vm.rs` back to `O_*`.
pub fn open_flags(bits: u64) -> libc::c_int {
    const TABLE: [(OpenFlag, libc::c_int); 14] = [
        (OpenFlag::RDONLY, libc::O_RDONLY),
        (OpenFlag::WRONLY, libc::O_WRONLY),
        (OpenFlag::RDWR, libc::O_RDWR),
        (OpenFlag::CREAT, libc::O_CREAT),
        (OpenFlag::TRUNC, libc::O_TRUNC),
        (OpenFlag::EXCL, libc::O_EXCL),
        (OpenFlag::APPEND, libc::O_APPEND),
        (OpenFlag::CLOEXEC, libc::O_CLOEXEC),
        (OpenFlag::DIRECTORY, libc::O_DIRECTORY),
        (OpenFlag::DIRECT, libc::O_DIRECT),
        (OpenFlag::SYNC, libc::O_SYNC),
        (OpenFlag::DSYNC, libc::O_DSYNC),
        (OpenFlag::NOATIME, libc::O_NOATIME),
        (OpenFlag::NOFOLLOW, libc::O_NOFOLLOW),
    ];
    let mut f = 0;
    for (flag, o) in TABLE {
        if bits & (1 << (flag as u8)) != 0 {
            f |= o;
        }
    }
    f
}

/// Every errno of the platform by its name, as `strace` prints it (and so as `aeiou-trace
/// export` writes it and an `expect` names it). A name outside a short list once came out
/// as `E?` and matched no `expect`, failing a trace that did what the application did (RC1
/// review).
const ERRNO_NAMES: &[(i32, &str)] = &[
    (libc::EPERM, "EPERM"), (libc::ENOENT, "ENOENT"), (libc::ESRCH, "ESRCH"), (libc::EINTR, "EINTR"),
    (libc::EIO, "EIO"), (libc::ENXIO, "ENXIO"), (libc::E2BIG, "E2BIG"), (libc::ENOEXEC, "ENOEXEC"),
    (libc::EBADF, "EBADF"), (libc::ECHILD, "ECHILD"), (libc::EAGAIN, "EAGAIN"), (libc::ENOMEM, "ENOMEM"),
    (libc::EACCES, "EACCES"), (libc::EFAULT, "EFAULT"), (libc::ENOTBLK, "ENOTBLK"),
    (libc::EBUSY, "EBUSY"), (libc::EEXIST, "EEXIST"), (libc::EXDEV, "EXDEV"), (libc::ENODEV, "ENODEV"),
    (libc::ENOTDIR, "ENOTDIR"), (libc::EISDIR, "EISDIR"), (libc::EINVAL, "EINVAL"),
    (libc::ENFILE, "ENFILE"), (libc::EMFILE, "EMFILE"), (libc::ENOTTY, "ENOTTY"),
    (libc::ETXTBSY, "ETXTBSY"), (libc::EFBIG, "EFBIG"), (libc::ENOSPC, "ENOSPC"),
    (libc::ESPIPE, "ESPIPE"), (libc::EROFS, "EROFS"), (libc::EMLINK, "EMLINK"), (libc::EPIPE, "EPIPE"),
    (libc::EDOM, "EDOM"), (libc::ERANGE, "ERANGE"), (libc::EDEADLK, "EDEADLK"),
    (libc::ENAMETOOLONG, "ENAMETOOLONG"), (libc::ENOLCK, "ENOLCK"), (libc::ENOSYS, "ENOSYS"),
    (libc::ENOTEMPTY, "ENOTEMPTY"), (libc::ELOOP, "ELOOP"), (libc::ENOMSG, "ENOMSG"),
    (libc::EIDRM, "EIDRM"), (libc::ECHRNG, "ECHRNG"), (libc::EL2NSYNC, "EL2NSYNC"),
    (libc::EL3HLT, "EL3HLT"), (libc::EL3RST, "EL3RST"), (libc::ELNRNG, "ELNRNG"),
    (libc::EUNATCH, "EUNATCH"), (libc::ENOCSI, "ENOCSI"), (libc::EL2HLT, "EL2HLT"),
    (libc::EBADE, "EBADE"), (libc::EBADR, "EBADR"), (libc::EXFULL, "EXFULL"), (libc::ENOANO, "ENOANO"),
    (libc::EBADRQC, "EBADRQC"), (libc::EBADSLT, "EBADSLT"), (libc::EBFONT, "EBFONT"),
    (libc::ENOSTR, "ENOSTR"), (libc::ENODATA, "ENODATA"), (libc::ETIME, "ETIME"), (libc::ENOSR, "ENOSR"),
    (libc::ENONET, "ENONET"), (libc::ENOPKG, "ENOPKG"), (libc::EREMOTE, "EREMOTE"),
    (libc::ENOLINK, "ENOLINK"), (libc::EADV, "EADV"), (libc::ESRMNT, "ESRMNT"), (libc::ECOMM, "ECOMM"),
    (libc::EPROTO, "EPROTO"), (libc::EMULTIHOP, "EMULTIHOP"), (libc::EDOTDOT, "EDOTDOT"),
    (libc::EBADMSG, "EBADMSG"), (libc::EOVERFLOW, "EOVERFLOW"), (libc::ENOTUNIQ, "ENOTUNIQ"),
    (libc::EBADFD, "EBADFD"), (libc::EREMCHG, "EREMCHG"), (libc::ELIBACC, "ELIBACC"),
    (libc::ELIBBAD, "ELIBBAD"), (libc::ELIBSCN, "ELIBSCN"), (libc::ELIBMAX, "ELIBMAX"),
    (libc::ELIBEXEC, "ELIBEXEC"), (libc::EILSEQ, "EILSEQ"), (libc::ERESTART, "ERESTART"),
    (libc::ESTRPIPE, "ESTRPIPE"), (libc::EUSERS, "EUSERS"), (libc::ENOTSOCK, "ENOTSOCK"),
    (libc::EDESTADDRREQ, "EDESTADDRREQ"), (libc::EMSGSIZE, "EMSGSIZE"), (libc::EPROTOTYPE, "EPROTOTYPE"),
    (libc::ENOPROTOOPT, "ENOPROTOOPT"), (libc::EPROTONOSUPPORT, "EPROTONOSUPPORT"),
    (libc::ESOCKTNOSUPPORT, "ESOCKTNOSUPPORT"), (libc::EOPNOTSUPP, "EOPNOTSUPP"),
    (libc::EPFNOSUPPORT, "EPFNOSUPPORT"), (libc::EAFNOSUPPORT, "EAFNOSUPPORT"),
    (libc::EADDRINUSE, "EADDRINUSE"), (libc::EADDRNOTAVAIL, "EADDRNOTAVAIL"),
    (libc::ENETDOWN, "ENETDOWN"), (libc::ENETUNREACH, "ENETUNREACH"), (libc::ENETRESET, "ENETRESET"),
    (libc::ECONNABORTED, "ECONNABORTED"), (libc::ECONNRESET, "ECONNRESET"), (libc::ENOBUFS, "ENOBUFS"),
    (libc::EISCONN, "EISCONN"), (libc::ENOTCONN, "ENOTCONN"), (libc::ESHUTDOWN, "ESHUTDOWN"),
    (libc::ETOOMANYREFS, "ETOOMANYREFS"), (libc::ETIMEDOUT, "ETIMEDOUT"),
    (libc::ECONNREFUSED, "ECONNREFUSED"), (libc::EHOSTDOWN, "EHOSTDOWN"),
    (libc::EHOSTUNREACH, "EHOSTUNREACH"), (libc::EALREADY, "EALREADY"),
    (libc::EINPROGRESS, "EINPROGRESS"), (libc::ESTALE, "ESTALE"), (libc::EUCLEAN, "EUCLEAN"),
    (libc::ENOTNAM, "ENOTNAM"), (libc::ENAVAIL, "ENAVAIL"), (libc::EISNAM, "EISNAM"),
    (libc::EREMOTEIO, "EREMOTEIO"), (libc::EDQUOT, "EDQUOT"), (libc::ENOMEDIUM, "ENOMEDIUM"),
    (libc::EMEDIUMTYPE, "EMEDIUMTYPE"), (libc::ECANCELED, "ECANCELED"), (libc::ENOKEY, "ENOKEY"),
    (libc::EKEYEXPIRED, "EKEYEXPIRED"), (libc::EKEYREVOKED, "EKEYREVOKED"),
    (libc::EKEYREJECTED, "EKEYREJECTED"), (libc::EOWNERDEAD, "EOWNERDEAD"),
    (libc::ENOTRECOVERABLE, "ENOTRECOVERABLE"), (libc::ERFKILL, "ERFKILL"),
];

/// The other names of an errno of `ERRNO_NAMES`.
const ERRNO_ALIASES: &[(i32, &str)] = &[(libc::EAGAIN, "EWOULDBLOCK"), (libc::EOPNOTSUPP, "ENOTSUP"), (libc::EDEADLK, "EDEADLOCK")];

/// The name of an errno; `E<number>` for one the platform does not name.
pub fn errno_name(code: i32) -> std::borrow::Cow<'static, str> {
    match ERRNO_NAMES.iter().find(|(c, _)| *c == code) {
        Some((_, n)) => std::borrow::Cow::Borrowed(n),
        None => std::borrow::Cow::Owned(format!("E{code}")),
    }
}

/// `name` names the errno `code`: its name, an alias of it, or `E<number>`.
pub fn errno_is(name: &str, code: i32) -> bool {
    ERRNO_NAMES.iter().chain(ERRNO_ALIASES).any(|(c, n)| *c == code && *n == name) || name.strip_prefix('E').and_then(|d| d.parse::<i32>().ok()) == Some(code)
}

fn cpath(p: &Path) -> io::Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(p.as_os_str().as_bytes()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))
}

fn check(r: libc::c_long) -> io::Result<libc::c_long> {
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(r) }
}

fn check_int(r: libc::c_int) -> io::Result<()> {
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

pub struct Sync {
    pub direct: bool,
}

impl Backend for Sync {
    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OpenFile> {
        let c = cpath(path)?;
        let mut f = open_flags(flags);
        if self.direct && f & libc::O_DIRECTORY == 0 {
            f |= libc::O_DIRECT;
        }
        let fd = unsafe { libc::open(c.as_ptr(), f, mode as libc::c_uint) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(OpenFile::from(unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize> {
                let r = match offset {
            Some(off) => unsafe { libc::pread(file.as_raw_fd(), buf.as_mut_ptr() as *mut _, buf.len(), off) },
            None => unsafe { libc::read(file.as_raw_fd(), buf.as_mut_ptr() as *mut _, buf.len()) },
        };
        check(r as libc::c_long).map(|n| n as usize)
    }

    fn write(&mut self, file: &OpenFile, buf: &[u8], offset: Option<i64>) -> io::Result<usize> {
        let r = match offset {
            Some(off) => unsafe { libc::pwrite(file.as_raw_fd(), buf.as_ptr() as *const _, buf.len(), off) },
            None => unsafe { libc::write(file.as_raw_fd(), buf.as_ptr() as *const _, buf.len()) },
        };
        check(r as libc::c_long).map(|n| n as usize)
    }

    fn lseek(&mut self, file: &OpenFile, offset: i64, whence: Whence) -> io::Result<i64> {
        let w = match whence {
            Whence::SET => libc::SEEK_SET,
            Whence::CUR => libc::SEEK_CUR,
            Whence::END => libc::SEEK_END,
        };
        check(unsafe { libc::lseek(file.as_raw_fd(), offset, w) } as libc::c_long).map(|n| n as i64)
    }

    fn ioctl(&mut self, file: &OpenFile, request: IoctlRequest) -> io::Result<()> {
        let r = match request {
            IoctlRequest::TCGETS => {
                let mut t: libc::termios = unsafe { std::mem::zeroed() };
                unsafe { libc::ioctl(file.as_raw_fd(), libc::TCGETS, &mut t as *mut _) }
            }
            IoctlRequest::FIONREAD => {
                let mut n: libc::c_int = 0;
                unsafe { libc::ioctl(file.as_raw_fd(), libc::FIONREAD, &mut n as *mut _) }
            }
            IoctlRequest::BLKGETSIZE64 => {
                let mut n: u64 = 0;
                // BLKGETSIZE64 = _IOR(0x12, 114, size_t)
                unsafe { libc::ioctl(file.as_raw_fd(), 0x8008_1272, &mut n as *mut _) }
            }
        };
        check_int(r)
    }

    fn fadvise(&mut self, file: &OpenFile, offset: i64, len: i64, advice: Advice) -> io::Result<()> {
        let adv = match advice {
            Advice::NORMAL => libc::POSIX_FADV_NORMAL,
            Advice::RANDOM => libc::POSIX_FADV_RANDOM,
            Advice::SEQUENTIAL => libc::POSIX_FADV_SEQUENTIAL,
            Advice::WILLNEED => libc::POSIX_FADV_WILLNEED,
            Advice::DONTNEED => libc::POSIX_FADV_DONTNEED,
            Advice::NOREUSE => libc::POSIX_FADV_NOREUSE,
        };
        // returns the errno directly, not -1
        let r = unsafe { libc::posix_fadvise(file.as_raw_fd(), offset, len, adv) };
        if r != 0 { Err(io::Error::from_raw_os_error(r)) } else { Ok(()) }
    }

    fn fstat(&mut self, file: &OpenFile) -> io::Result<i64> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        check_int(unsafe { libc::fstat(file.as_raw_fd(), &mut st) })?;
        Ok(st.st_size)
    }

    fn stat(&mut self, path: &Path) -> io::Result<i64> {
        let c = cpath(path)?;
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        check_int(unsafe { libc::stat(c.as_ptr(), &mut st) })?;
        Ok(st.st_size)
    }

    fn fsync(&mut self, file: &OpenFile) -> io::Result<()> {
        check_int(unsafe { libc::fsync(file.as_raw_fd()) })
    }

    fn fdatasync(&mut self, file: &OpenFile) -> io::Result<()> {
        check_int(unsafe { libc::fdatasync(file.as_raw_fd()) })
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        let c = cpath(path)?;
        check_int(unsafe { libc::unlink(c.as_ptr()) })
    }

    fn ftruncate(&mut self, file: &OpenFile, len: i64) -> io::Result<()> {
        check_int(unsafe { libc::ftruncate(file.as_raw_fd(), len) })
    }

    fn fallocate(&mut self, file: &OpenFile, offset: i64, len: i64) -> io::Result<()> {
        check_int(unsafe { libc::fallocate(file.as_raw_fd(), 0, offset, len) })
    }

    fn mkdir(&mut self, path: &Path, mode: u32) -> io::Result<()> {
        let c = cpath(path)?;
        check_int(unsafe { libc::mkdir(c.as_ptr(), mode as libc::mode_t) })
    }

    fn rmdir(&mut self, path: &Path) -> io::Result<()> {
        let c = cpath(path)?;
        check_int(unsafe { libc::rmdir(c.as_ptr()) })
    }

    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        let a = cpath(from)?;
        let b = cpath(to)?;
        check_int(unsafe { libc::rename(a.as_ptr(), b.as_ptr()) })
    }

    fn readdir(&mut self, file: &OpenFile) -> io::Result<usize> {
        let mut buf = vec![0u8; 32 * 1024];
        let mut entries = 0usize;
        loop {
            let n = unsafe { libc::syscall(libc::SYS_getdents64, file.as_raw_fd(), buf.as_mut_ptr(), buf.len()) };
            let n = check(n)? as usize;
            if n == 0 {
                return Ok(entries);
            }
            let mut off = 0usize;
            while off < n {
                // struct linux_dirent64 { u64 d_ino; i64 d_off; u16 d_reclen; u8 d_type; char d_name[]; }
                let reclen = u16::from_ne_bytes([buf[off + 16], buf[off + 17]]) as usize;
                let name = &buf[off + 19..off + reclen];
                let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
                let name = &name[..end];
                if name != b"." && name != b".." && !name.starts_with(b".aeiou") {
                    entries += 1;
                }
                off += reclen;
            }
        }
    }
}

// ---------------------------------------------------------------- posix-aio

/// glibc POSIX AIO: `aio_read`/`aio_write`/`aio_fsync`, then `aio_suspend` until the request
/// is done. glibc runs the request on a thread of its own pool (20 threads unless `aio_init`
/// says otherwise, and requests on one descriptor are served one at a time), so each op is a
/// hand-off to another thread and back around the same `pread`/`pwrite`. An actor has one op
/// in flight, so there is never a list for `lio_listio`. Everything else is `sync`.
pub struct PosixAio {
    sync: Sync,
}

impl PosixAio {
    fn cb(fd: BorrowedFd, buf: *mut u8, len: usize, offset: i64) -> libc::aiocb {
        let mut cb: libc::aiocb = unsafe { std::mem::zeroed() };
        cb.aio_fildes = fd.as_raw_fd();
        cb.aio_buf = buf as *mut libc::c_void;
        cb.aio_nbytes = len;
        cb.aio_offset = offset;
        cb.aio_sigevent.sigev_notify = libc::SIGEV_NONE;
        cb
    }

    /// Wait for a request that was accepted; its return value.
    fn wait(cb: &mut libc::aiocb) -> io::Result<isize> {
        let list = [cb as *const libc::aiocb];
        loop {
            // SAFETY: `cb` is a submitted control block that stays put until `aio_return`
            let e = unsafe { libc::aio_error(cb) };
            if e == libc::EINPROGRESS {
                unsafe { libc::aio_suspend(list.as_ptr(), 1, std::ptr::null()) };
                continue;
            }
            let n = unsafe { libc::aio_return(cb) };
            return if e != 0 { Err(io::Error::from_raw_os_error(e)) } else { Ok(n) };
        }
    }
}

impl Backend for PosixAio {
    fn positional(&self) -> bool {
        true
    }

    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OpenFile> {
        self.sync.open(path, flags, mode)
    }

    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize> {
        let Some(off) = offset else { return self.sync.read(file, buf, None) };
        let mut cb = Self::cb(file.as_fd(), buf.as_mut_ptr(), buf.len(), off);
        check_int(unsafe { libc::aio_read(&mut cb) })?;
        Self::wait(&mut cb).map(|n| n as usize)
    }

    fn write(&mut self, file: &OpenFile, buf: &[u8], offset: Option<i64>) -> io::Result<usize> {
        let Some(off) = offset else { return self.sync.write(file, buf, None) };
        let mut cb = Self::cb(file.as_fd(), buf.as_ptr() as *mut u8, buf.len(), off);
        check_int(unsafe { libc::aio_write(&mut cb) })?;
        Self::wait(&mut cb).map(|n| n as usize)
    }

    fn lseek(&mut self, file: &OpenFile, offset: i64, whence: Whence) -> io::Result<i64> {
        self.sync.lseek(file, offset, whence)
    }

    fn ioctl(&mut self, file: &OpenFile, request: IoctlRequest) -> io::Result<()> {
        self.sync.ioctl(file, request)
    }

    fn fadvise(&mut self, file: &OpenFile, offset: i64, len: i64, advice: Advice) -> io::Result<()> {
        self.sync.fadvise(file, offset, len, advice)
    }

    fn fstat(&mut self, file: &OpenFile) -> io::Result<i64> {
        self.sync.fstat(file)
    }

    fn stat(&mut self, path: &Path) -> io::Result<i64> {
        self.sync.stat(path)
    }

    fn fsync(&mut self, file: &OpenFile) -> io::Result<()> {
        let mut cb = Self::cb(file.as_fd(), std::ptr::null_mut(), 0, 0);
        check_int(unsafe { libc::aio_fsync(libc::O_SYNC, &mut cb) })?;
        Self::wait(&mut cb).map(|_| ())
    }

    fn fdatasync(&mut self, file: &OpenFile) -> io::Result<()> {
        let mut cb = Self::cb(file.as_fd(), std::ptr::null_mut(), 0, 0);
        check_int(unsafe { libc::aio_fsync(libc::O_DSYNC, &mut cb) })?;
        Self::wait(&mut cb).map(|_| ())
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        self.sync.unlink(path)
    }

    fn ftruncate(&mut self, file: &OpenFile, len: i64) -> io::Result<()> {
        self.sync.ftruncate(file, len)
    }

    fn fallocate(&mut self, file: &OpenFile, offset: i64, len: i64) -> io::Result<()> {
        self.sync.fallocate(file, offset, len)
    }

    fn mkdir(&mut self, path: &Path, mode: u32) -> io::Result<()> {
        self.sync.mkdir(path, mode)
    }

    fn rmdir(&mut self, path: &Path) -> io::Result<()> {
        self.sync.rmdir(path)
    }

    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        self.sync.rename(from, to)
    }

    fn readdir(&mut self, file: &OpenFile) -> io::Result<usize> {
        self.sync.readdir(file)
    }
}

// ---------------------------------------------------------------- mmap

/// `MADV_POPULATE_READ` (Linux 5.14).
const MADV_POPULATE_READ: libc::c_int = 22;

struct Mapping {
    ptr: *mut u8,
    len: usize,
}

/// An open file as the actors hold it, and the handle every `Backend` method takes in place
/// of a descriptor (`DESIGN_REVIEW.md` §3.65): the descriptor and, under the `mmap`
/// backend, the mapping of it. The mapping belongs to the open file, not to the actor that reads: every
/// actor that sees the descriptor (the one that opened it and the sub-actors that inherited
/// it at a fork, which are threads of one address space) reads through the one mapping,
/// made by whichever of them reads first and unmapped when the last of them lets the
/// descriptor go, just before it is closed.
///
/// Readers find the current mapping without a lock. A file that has grown past it is mapped
/// again at its new size, under the lock, and the earlier mapping stays until the close,
/// since another sub-actor may be inside it.
///
/// A file in an object store (`object.rs`) has no descriptor: its handle is the store and
/// the key, and the driver routes its ops to the object engine by path, so no `Backend`
/// ever sees one (the descriptor of such a handle is a bug, and panics).
pub struct OpenFile {
    fd: Option<OwnedFd>,
    object: Option<crate::object::Handle>,
    map: AtomicPtr<Mapping>,
    all: Mutex<Vec<Box<Mapping>>>,
    /// The file's size as this open last learned it, or -1 before it has: the abstract's
    /// `fstat` of it, else the one the `mmap` backend issues before its first map, then the
    /// end of every write through it. The length a mapping is made at.
    size: AtomicI64,
}

// SAFETY: a `Mapping` is a read-only shared mapping that is not unmapped while the
// `OpenFile` lives; the list is behind the mutex and the current one behind the atomic
unsafe impl Send for OpenFile {}
unsafe impl std::marker::Sync for OpenFile {}

/// Files the actors of this process hold open, and the most they have held since
/// `open_files_reset`: counted where an `OpenFile` is made and dropped, not sampled.
static OPEN_FILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPEN_FILES_PEAK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Start a run's count: the peak becomes what is open now.
pub fn open_files_reset() {
    OPEN_FILES_PEAK.store(OPEN_FILES.load(Ordering::Relaxed), Ordering::Relaxed);
}

pub fn open_files_peak() -> u64 {
    OPEN_FILES_PEAK.load(Ordering::Relaxed)
}

impl From<OwnedFd> for OpenFile {
    fn from(fd: OwnedFd) -> Self {
        let now = OPEN_FILES.fetch_add(1, Ordering::Relaxed) + 1;
        OPEN_FILES_PEAK.fetch_max(now, Ordering::Relaxed);
        OpenFile { fd: Some(fd), object: None, map: AtomicPtr::new(std::ptr::null_mut()), all: Mutex::new(Vec::new()), size: AtomicI64::new(-1) }
    }
}

impl OpenFile {
    /// An object's handle: no descriptor, so not counted among the open files.
    pub fn object(h: crate::object::Handle) -> OpenFile {
        OpenFile { fd: None, object: Some(h), map: AtomicPtr::new(std::ptr::null_mut()), all: Mutex::new(Vec::new()), size: AtomicI64::new(-1) }
    }

    pub fn as_object(&self) -> Option<&crate::object::Handle> {
        self.object.as_ref()
    }
}

impl std::ops::Deref for OpenFile {
    type Target = OwnedFd;
    fn deref(&self) -> &OwnedFd {
        self.fd.as_ref().expect("an object's handle has no descriptor; its ops go to the object engine")
    }
}

impl Drop for OpenFile {
    fn drop(&mut self) {
        if self.fd.is_some() {
            OPEN_FILES.fetch_sub(1, Ordering::Relaxed);
        }
        for m in self.all.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
            unsafe { libc::munmap(m.ptr as *mut libc::c_void, m.len) };
        }
    }
}

/// Reads through a mapping: the first read of an open file maps the whole of it
/// (`PROT_READ`, `MAP_SHARED`, at the size the open knows: `OpenFile::size`), a read makes its range of the
/// mapping resident (one byte of every page is read, or under `MmapConsume::Copy` the range
/// is copied into the actor's buffer), and the close unmaps (`OpenFile`: one mapping per
/// open, shared with the sub-actors that inherit the descriptor). That is the shape of a
/// safetensors or Arrow load: `open`, `fstat`, `mmap`, the consumer's accesses, `munmap`,
/// `close`. The length is the application's: the size its own `fstat` of the open returned,
/// which the abstract's `fstat` op records, so the backend issues none of its own; only an
/// open read before the abstract asks its size costs one `fstat`, the backend's, inside the
/// first read (`MmapStats::sized`). The read returns only when every page of the range has
/// arrived: a fault does not return before its page is read, and `MADV_POPULATE_READ` does
/// not return before the range is. A read past the size the open knows returns what `pread`
/// would at that size and asks nothing; a file that grows is mapped again at the next read
/// past the mapping once the open learns it, by the abstract's `fstat` or a write through
/// the open, as an application must. Writes and every other op are `sync` (no loader writes
/// through a mapping).
///
/// Under `MmapMode::Fault` and `WillNeed` an I/O error or a file truncated underneath is a
/// `SIGBUS`, as it is for the applications this models; `Populate` gets an errno.
///
/// The map, the unmap, the translation flush an unmap causes, and the faults are what the
/// technique costs and are counted against it. Every actor thread maps into the one
/// address space of the runner process, so those costs also couple the actors (one
/// `mmap_lock`, flushes sent to every core running an actor), more than they couple
/// PyTorch workers, which are processes with an address space each.
pub struct Mmap {
    sync: Sync,
    mode: MmapMode,
    consume: MmapConsume,
    stats: Arc<MmapStats>,
}

impl Mmap {
    /// The descriptor's mapping and how much of it the reads may use, made or remade so it
    /// covers `end` when the size the open knows does. The size is asked here only when the
    /// open has not learned it.
    fn mapping(&mut self, file: &OpenFile, end: usize) -> io::Result<(*mut u8, usize)> {
        // SAFETY: a published mapping lives in `file.all` until the file is dropped
        let current = |p: *mut Mapping| if p.is_null() { None } else { Some(unsafe { ((*p).ptr, (*p).len) }) };
        let covers = || {
            let known = file.size.load(Ordering::Acquire);
            match current(file.map.load(Ordering::Acquire)) {
                // all the open knows of the file is mapped: the reads stop at that size
                Some((ptr, len)) if known >= 0 && known as usize <= len => Some((ptr, known as usize)),
                Some((ptr, len)) if known < 0 && end <= len => Some((ptr, len)),
                None if known == 0 => Some((std::ptr::null_mut(), 0)),
                _ => None,
            }
        };
        if let Some(m) = covers() {
            return Ok(m);
        }
        // no mapping yet, or one the known size has outgrown: one actor at a time maps
        let mut all = file.all.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = covers() {
            return Ok(m);
        }
        if file.size.load(Ordering::Acquire) < 0 {
            // the abstract has not asked this open's size, and the technique needs it to map
            let n = self.sync.fstat(file)?;
            self.stats.sized.fetch_add(1, Ordering::Relaxed);
            file.size.store(n, Ordering::Release);
            if let Some(m) = covers() {
                return Ok(m);
            }
        }
        let size = file.size.load(Ordering::Acquire).max(0) as usize;
        if size == 0 {
            return Ok((std::ptr::null_mut(), 0));
        }
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        self.stats.maps.fetch_add(1, Ordering::Relaxed);
        self.stats.mapped_bytes.fetch_add(size as u64, Ordering::Relaxed);
        let mut m = Box::new(Mapping { ptr: ptr as *mut u8, len: size });
        file.map.store(&mut *m, Ordering::Release);
        all.push(m);
        Ok((ptr as *mut u8, size))
    }

    /// A write through the open ending at `end`: the least the file's size now is, when the
    /// open knows it.
    fn wrote(file: &OpenFile, end: Option<i64>) {
        match end {
            Some(end) => {
                let _ = file.size.try_update(Ordering::AcqRel, Ordering::Acquire, |s| (s >= 0 && s < end).then_some(end));
            }
            // at the descriptor's position, which this backend does not follow: ask again
            None => file.size.store(-1, Ordering::Release),
        }
    }
}

impl Backend for Mmap {
    fn positional(&self) -> bool {
        true
    }

    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OpenFile> {
        self.sync.open(path, flags, mode)
    }

    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize> {
        let Some(off) = offset else { return self.sync.read(file, buf, None) };
        let off = off as usize;
        let (ptr, len) = self.mapping(file, off + buf.len())?;
        if off >= len {
            return Ok(0);
        }
        let n = buf.len().min(len - off);
        if n == 0 {
            return Ok(0);
        }
        let advice = match self.mode {
            MmapMode::Fault => None,
            MmapMode::Populate => Some(MADV_POPULATE_READ),
            MmapMode::WillNeed => Some(libc::MADV_WILLNEED),
        };
        if let Some(advice) = advice {
            // madvise takes whole pages: from the page holding `off` to the end of the range
            let lo = off - off % ALIGN;
            check_int(unsafe { libc::madvise(ptr.add(lo) as *mut libc::c_void, off + n - lo, advice) })?;
            self.stats.advised.fetch_add(1, Ordering::Relaxed);
        }
        match self.consume {
            MmapConsume::Copy => {
                // SAFETY: `[off, off + n)` is inside the mapping, and `buf` is the actor's own buffer
                unsafe { std::ptr::copy_nonoverlapping(ptr.add(off), buf.as_mut_ptr(), n) };
                self.stats.copied_bytes.fetch_add(n as u64, Ordering::Relaxed);
            }
            // populated: the range is resident and mapped already
            MmapConsume::Touch if self.mode == MmapMode::Populate => {}
            MmapConsume::Touch => {
                // one byte of each page: the first byte of the range, then every page start
                let mut at = off;
                let mut pages = 0u64;
                while at < off + n {
                    // SAFETY: `at` is inside the mapping; volatile so the read is not elided
                    unsafe { std::ptr::read_volatile(ptr.add(at)) };
                    pages += 1;
                    at = at - at % ALIGN + ALIGN;
                }
                self.stats.touched_pages.fetch_add(pages, Ordering::Relaxed);
            }
        }
        Ok(n)
    }

    fn write(&mut self, file: &OpenFile, buf: &[u8], offset: Option<i64>) -> io::Result<usize> {
        let n = self.sync.write(file, buf, offset)?;
        Mmap::wrote(file, offset.map(|o| o + n as i64));
        Ok(n)
    }

    fn lseek(&mut self, file: &OpenFile, offset: i64, whence: Whence) -> io::Result<i64> {
        self.sync.lseek(file, offset, whence)
    }

    fn ioctl(&mut self, file: &OpenFile, request: IoctlRequest) -> io::Result<()> {
        self.sync.ioctl(file, request)
    }

    fn fadvise(&mut self, file: &OpenFile, offset: i64, len: i64, advice: Advice) -> io::Result<()> {
        self.sync.fadvise(file, offset, len, advice)
    }

    /// The abstract's `fstat`: the application learning the length it maps.
    fn fstat(&mut self, file: &OpenFile) -> io::Result<i64> {
        let n = self.sync.fstat(file)?;
        file.size.store(n, Ordering::Release);
        Ok(n)
    }

    fn stat(&mut self, path: &Path) -> io::Result<i64> {
        self.sync.stat(path)
    }

    fn fsync(&mut self, file: &OpenFile) -> io::Result<()> {
        self.sync.fsync(file)
    }

    fn fdatasync(&mut self, file: &OpenFile) -> io::Result<()> {
        self.sync.fdatasync(file)
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        self.sync.unlink(path)
    }

    fn ftruncate(&mut self, file: &OpenFile, len: i64) -> io::Result<()> {
        self.sync.ftruncate(file, len)?;
        // a shrink keeps the earlier mapping, whose pages past the new end are a SIGBUS:
        // the reads stop at the new size
        file.size.store(len, Ordering::Release);
        Ok(())
    }

    fn fallocate(&mut self, file: &OpenFile, offset: i64, len: i64) -> io::Result<()> {
        self.sync.fallocate(file, offset, len)?;
        Mmap::wrote(file, Some(offset + len));
        Ok(())
    }

    fn mkdir(&mut self, path: &Path, mode: u32) -> io::Result<()> {
        self.sync.mkdir(path, mode)
    }

    fn rmdir(&mut self, path: &Path) -> io::Result<()> {
        self.sync.rmdir(path)
    }

    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        self.sync.rename(from, to)
    }

    fn readdir(&mut self, file: &OpenFile) -> io::Result<usize> {
        self.sync.readdir(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_mapping_belongs_to_the_open_file_and_a_grown_file_is_mapped_again() {
        let path = std::env::temp_dir().join(format!("aeiou-openfile-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; 8192]).unwrap();
        let stats = Arc::new(MmapStats::default());
        let make = || BackendKind::Mmap.make(MmapMode::Fault, MmapConsume::Copy, &stats);
        let (mut a, mut b) = (make(), make());
        let file = Arc::new(OpenFile::from(a.open(&path, 1 << (OpenFlag::RDONLY as u8), 0).unwrap()));
        let mut buf = vec![0u8; 4096];
        // two backends (two actors) read the one open file: one mapping
        assert_eq!(a.read(&file, &mut buf, Some(0)).unwrap(), 4096);
        let f2 = file.clone();
        std::thread::spawn(move || {
            let mut buf = vec![0u8; 4096];
            assert_eq!(b.read(&f2, &mut buf, Some(4096)).unwrap(), 4096);
            assert_eq!(buf[0], 7);
        })
        .join()
        .unwrap();
        assert_eq!(stats.maps.load(Ordering::Relaxed), 1);
        // a read at the end maps nothing; a grown file is mapped again once the open's
        // `fstat` (the abstract's) learns its new size, and not before
        assert_eq!(a.read(&file, &mut buf, Some(8192)).unwrap(), 0);
        assert_eq!(stats.maps.load(Ordering::Relaxed), 1);
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(&[9u8; 4096]).unwrap();
        assert_eq!(a.read(&file, &mut buf, Some(8192)).unwrap(), 0);
        assert_eq!(a.fstat(&file).unwrap(), 12288);
        assert_eq!(a.read(&file, &mut buf, Some(8192)).unwrap(), 4096);
        assert_eq!(buf[0], 9);
        assert_eq!((stats.maps.load(Ordering::Relaxed), stats.mapped_bytes.load(Ordering::Relaxed)), (2, 8192 + 12288));
        // a second open of the same path is its own mapping
        let other = OpenFile::from(a.open(&path, 1 << (OpenFlag::RDONLY as u8), 0).unwrap());
        assert_eq!(a.read(&other, &mut buf, Some(0)).unwrap(), 4096);
        assert_eq!(stats.maps.load(Ordering::Relaxed), 3);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_mapping_is_made_at_the_size_the_abstract_asked_and_the_backend_asks_only_when_it_did_not() {
        // RC1 review: the backend asked `fstat` for itself at the first read of every open,
        // after the abstract's own, and again at every read past the mapping (each
        // `until_eof` read's last two): syscalls the application never issues, timed as reads
        let path = std::env::temp_dir().join(format!("aeiou-mmap-sized-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; 8192]).unwrap();
        let stats = Arc::new(MmapStats::default());
        let mut be = BackendKind::Mmap.make(MmapMode::Fault, MmapConsume::Copy, &stats);
        let rd = 1 << (OpenFlag::RDONLY as u8);
        let mut buf = vec![0u8; 16384];
        // the safetensors shape: `open`, `fstat`, the mapping at the size that fstat returned
        let file = OpenFile::from(be.open(&path, rd, 0).unwrap());
        assert_eq!(be.fstat(&file).unwrap(), 8192);
        // the file grows after the application learned its size: the mapping is the
        // application's length, so the reads end there, as they would through its mapping
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(&[9u8; 4096]).unwrap();
        assert_eq!(be.read(&file, &mut buf[..4096], Some(0)).unwrap(), 4096);
        assert_eq!(be.read(&file, &mut buf[..8193], Some(0)).unwrap(), 8192, "the read past the size the open knows is short");
        assert_eq!(be.read(&file, &mut buf[..1], Some(8192)).unwrap(), 0, "and one at that size is the end");
        assert_eq!((stats.sized.load(Ordering::Relaxed), stats.maps.load(Ordering::Relaxed), stats.mapped_bytes.load(Ordering::Relaxed)), (0, 1, 8192));
        drop(file);
        // an open read without the abstract's `fstat`: the backend asks once, at the first
        // read, and never again for the reads past the end (`until_eof`'s size + 1, then 0)
        let file = OpenFile::from(be.open(&path, rd, 0).unwrap());
        assert_eq!(be.read(&file, &mut buf[..12289], Some(0)).unwrap(), 12288);
        assert_eq!(be.read(&file, &mut buf[..1], Some(12288)).unwrap(), 0);
        assert_eq!(be.read(&file, &mut buf[..1], Some(12288)).unwrap(), 0);
        assert_eq!((stats.sized.load(Ordering::Relaxed), stats.maps.load(Ordering::Relaxed)), (1, 2));
        drop(file);
        // writes through the open grow the size it knows: the read-back is mapped again
        // without asking
        let file = OpenFile::from(be.open(&path, 1 << (OpenFlag::RDWR as u8), 0).unwrap());
        assert_eq!(be.fstat(&file).unwrap(), 12288);
        assert_eq!(be.read(&file, &mut buf[..1], Some(0)).unwrap(), 1);
        assert_eq!(be.write(&file, &[5u8; 4096], Some(12288)).unwrap(), 4096);
        assert_eq!(be.read(&file, &mut buf[..4097], Some(12288)).unwrap(), 4096);
        assert_eq!(buf[0], 5);
        // and a truncation shrinks it: the reads stop at the new end, never in a page past it
        be.ftruncate(&file, 4096).unwrap();
        assert_eq!(be.read(&file, &mut buf[..8192], Some(0)).unwrap(), 4096);
        assert_eq!(be.read(&file, &mut buf[..1], Some(8192)).unwrap(), 0);
        assert_eq!((stats.sized.load(Ordering::Relaxed), stats.maps.load(Ordering::Relaxed)), (1, 4));
        drop(file);
        std::fs::remove_file(&path).unwrap();
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("aeiou-backend-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn bits(flags: &[OpenFlag]) -> u64 {
        flags.iter().fold(0, |b, f| b | 1 << (*f as u8))
    }

    fn errno<T>(r: io::Result<T>) -> i32 {
        r.err().expect("an error").raw_os_error().expect("an errno")
    }

    #[test]
    fn every_blocking_backend_issues_every_op_with_the_same_results() {
        // the op is the abstract's whatever the API: each backend, the same calls, the same
        // returns and errnos (`PROJECT_BRIEF.md` §4)
        let stats = Arc::new(MmapStats::default());
        for kind in [BackendKind::Sync, BackendKind::PosixAio, BackendKind::Mmap] {
            let d = scratch(kind.api.name());
            let mut be = kind.make(MmapMode::Fault, MmapConsume::Touch, &stats);
            let what = kind.describe();
            be.mkdir(&d.join("sub"), 0o755).unwrap();
            assert_eq!(errno(be.mkdir(&d.join("sub"), 0o755)), libc::EEXIST, "{what}");
            let p = d.join("sub/f");
            let f = be.open(&p, bits(&[OpenFlag::RDWR, OpenFlag::CREAT]), 0o644).unwrap();
            assert_eq!(be.write(&f, &[3u8; 8192], Some(0)).unwrap(), 8192, "{what}");
            // at the file position: the backend's API has none, the descriptor does
            assert_eq!(be.lseek(&f, 8192, Whence::SET).unwrap(), 8192, "{what}");
            assert_eq!(be.write(&f, &[4u8; 4096], None).unwrap(), 4096, "{what}");
            be.fsync(&f).unwrap();
            be.fdatasync(&f).unwrap();
            for advice in [Advice::NORMAL, Advice::RANDOM, Advice::SEQUENTIAL, Advice::WILLNEED, Advice::DONTNEED, Advice::NOREUSE] {
                be.fadvise(&f, 0, 4096, advice).unwrap();
            }
            be.fallocate(&f, 12288, 4096).unwrap();
            assert_eq!(be.fstat(&f).unwrap(), 16384, "{what}");
            let mut buf = vec![0u8; 4096];
            assert_eq!(be.read(&f, &mut buf, Some(8192)).unwrap(), 4096, "{what}");
            assert_eq!(be.lseek(&f, 0, Whence::SET).unwrap(), 0);
            assert_eq!(be.read(&f, &mut buf, None).unwrap(), 4096, "{what}");
            assert_eq!(buf[0], 3);
            be.ftruncate(&f, 4096).unwrap();
            assert_eq!(be.lseek(&f, 0, Whence::END).unwrap(), 4096, "{what}");
            assert_eq!(be.lseek(&f, -96, Whence::CUR).unwrap(), 4000, "{what}");
            // a regular file is no terminal and no block device; it says how much is unread
            assert_eq!(errno(be.ioctl(&f, IoctlRequest::TCGETS)), libc::ENOTTY, "{what}");
            be.ioctl(&f, IoctlRequest::FIONREAD).unwrap();
            assert_eq!(errno(be.ioctl(&f, IoctlRequest::BLKGETSIZE64)), libc::ENOTTY, "{what}");
            assert_eq!(errno(be.fadvise(&f, 0, -1, Advice::NORMAL)), libc::EINVAL, "{what}");
            assert_eq!(errno(be.fallocate(&f, -1, 1)), libc::EINVAL, "{what}");
            drop(f);
            assert_eq!(be.stat(&p).unwrap(), 4096, "{what}");
            std::fs::write(d.join("sub/.aeiou-hidden"), b"").unwrap();
            let dir = be.open(&d.join("sub"), bits(&[OpenFlag::RDONLY, OpenFlag::DIRECTORY]), 0).unwrap();
            assert_eq!(be.readdir(&dir).unwrap(), 1, "{what}: `.`, `..`, and `.aeiou*` are not counted");
            drop(dir);
            assert_eq!(errno(be.rmdir(&d.join("sub"))), libc::ENOTEMPTY, "{what}");
            be.rename(&p, &d.join("g")).unwrap();
            assert_eq!(errno(be.stat(&p)), libc::ENOENT, "{what}");
            be.unlink(&d.join("g")).unwrap();
            assert_eq!(errno(be.unlink(&d.join("g"))), libc::ENOENT, "{what}");
            std::fs::remove_file(d.join("sub/.aeiou-hidden")).unwrap();
            be.rmdir(&d.join("sub")).unwrap();
            // the failures of the calls that take a path
            assert_eq!(errno(be.open(&d.join("none"), bits(&[OpenFlag::RDONLY]), 0)), libc::ENOENT, "{what}");
            assert_eq!(be.open(Path::new("a\0b"), 0, 0).err().unwrap().kind(), io::ErrorKind::InvalidInput, "{what}");
            std::fs::remove_dir_all(&d).unwrap();
        }
    }

    #[test]
    fn posix_aio_returns_a_failed_request_s_errno() {
        // the request is accepted and fails in glibc's thread: `aio_error`, not the submit
        let d = scratch("aio-errno");
        let mut be = BackendKind::PosixAio.make(MmapMode::Fault, MmapConsume::Touch, &Arc::new(MmapStats::default()));
        let f = be.open(&d.join("w"), bits(&[OpenFlag::WRONLY, OpenFlag::CREAT]), 0o644).unwrap();
        let mut buf = [0u8; 16];
        assert_eq!(errno(be.read(&f, &mut buf, Some(0))), libc::EBADF);
        let r = be.open(&d.join("w"), bits(&[OpenFlag::RDONLY]), 0).unwrap();
        assert_eq!(errno(be.write(&r, &buf, Some(0))), libc::EBADF);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_mmap_modes_make_the_range_resident_and_count_what_they_did() {
        let d = scratch("mmap-modes");
        let p = d.join("f");
        std::fs::write(&p, (0..3 * 4096 + 100).map(|i| (i / 4096) as u8 + 1).collect::<Vec<u8>>()).unwrap();
        let rd = bits(&[OpenFlag::RDONLY]);
        for (mode, consume, advised, touched, copied) in [
            (MmapMode::Fault, MmapConsume::Touch, 0, 4, 0),
            (MmapMode::Populate, MmapConsume::Touch, 1, 0, 0),
            (MmapMode::WillNeed, MmapConsume::Touch, 1, 4, 0),
            (MmapMode::WillNeed, MmapConsume::Copy, 1, 0, 8388),
        ] {
            let stats = Arc::new(MmapStats::default());
            let mut be = BackendKind::Mmap.make(mode, consume, &stats);
            let f = be.open(&p, rd, 0).unwrap();
            let mut buf = vec![0u8; 8388];
            // from inside page 0 to the end of the file: the end of page 0, pages 1 and 2, and
            // the 100 bytes of page 3 (four pages touched)
            assert_eq!(be.read(&f, &mut buf, Some(4000)).unwrap(), 8388);
            let got = |a: &AtomicU64| a.load(Ordering::Relaxed);
            assert_eq!((got(&stats.advised), got(&stats.touched_pages), got(&stats.copied_bytes)), (advised, touched, copied), "{mode:?} {consume:?}");
            if consume == MmapConsume::Copy {
                assert_eq!((buf[0], buf[96], buf[96 + 4096], buf[8387]), (1, 2, 3, 4));
            }
            assert_eq!(MmapMode::parse(mode.name()), Some(mode));
            assert_eq!(MmapConsume::parse(consume.name()), Some(consume));
        }
        assert_eq!((MmapMode::parse("mapped"), MmapConsume::parse("move")), (None, None));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn mmap_maps_nothing_for_an_empty_file_and_a_refused_mapping_is_its_errno() {
        let d = scratch("mmap-edges");
        let stats = Arc::new(MmapStats::default());
        let mut be = BackendKind::Mmap.make(MmapMode::Fault, MmapConsume::Touch, &stats);
        // an empty file read before its size is asked: the backend asks, maps nothing, reads 0
        std::fs::write(d.join("empty"), b"").unwrap();
        let f = be.open(&d.join("empty"), bits(&[OpenFlag::RDONLY]), 0).unwrap();
        let mut buf = vec![0u8; 4096];
        assert_eq!(be.read(&f, &mut buf, Some(0)).unwrap(), 0);
        assert_eq!(be.read(&f, &mut buf, Some(0)).unwrap(), 0);
        assert_eq!((stats.sized.load(Ordering::Relaxed), stats.maps.load(Ordering::Relaxed)), (1, 0));
        drop(f);
        // a read of no bytes inside a mapping touches nothing
        std::fs::write(d.join("one"), vec![1u8; 4096]).unwrap();
        let f = be.open(&d.join("one"), bits(&[OpenFlag::RDONLY]), 0).unwrap();
        assert_eq!(be.read(&f, &mut buf[..0], Some(10)).unwrap(), 0);
        assert_eq!((stats.maps.load(Ordering::Relaxed), stats.touched_pages.load(Ordering::Relaxed)), (1, 0));
        drop(f);
        // a read through a write-only open: no read mapping of it is possible
        std::fs::write(d.join("w"), vec![1u8; 4096]).unwrap();
        let f = be.open(&d.join("w"), bits(&[OpenFlag::WRONLY]), 0).unwrap();
        assert_eq!(errno(be.read(&f, &mut buf, Some(0))), libc::EACCES);
        drop(f);
        // a write at the file position, or an allocation past the end, moves the size the
        // open knows: the next read past the old end maps the file again
        let f = be.open(&d.join("w"), bits(&[OpenFlag::RDWR]), 0).unwrap();
        assert_eq!(be.fstat(&f).unwrap(), 4096);
        assert_eq!(be.read(&f, &mut buf, Some(0)).unwrap(), 4096);
        assert_eq!(be.lseek(&f, 0, Whence::END).unwrap(), 4096);
        assert_eq!(be.write(&f, &[2u8; 4096], None).unwrap(), 4096);
        assert_eq!(be.read(&f, &mut buf, Some(4096)).unwrap(), 4096, "the size was asked again");
        be.fallocate(&f, 8192, 4096).unwrap();
        assert_eq!(be.read(&f, &mut buf, Some(8192)).unwrap(), 4096, "fallocate's zeros");
        assert_eq!(buf[0], 0);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn errno_names_and_backend_names() {
        for (code, name) in [
            (libc::ENOENT, "ENOENT"), (libc::EEXIST, "EEXIST"), (libc::ENOTTY, "ENOTTY"), (libc::EISDIR, "EISDIR"), (libc::ENOTDIR, "ENOTDIR"),
            (libc::ENOTEMPTY, "ENOTEMPTY"), (libc::EACCES, "EACCES"), (libc::EPERM, "EPERM"), (libc::EINVAL, "EINVAL"), (libc::ENOSPC, "ENOSPC"),
            (libc::EBADF, "EBADF"), (libc::ENAMETOOLONG, "ENAMETOOLONG"), (libc::EIO, "EIO"), (libc::EOPNOTSUPP, "EOPNOTSUPP"), (libc::EXDEV, "EXDEV"),
            (libc::EBUSY, "EBUSY"), (libc::EMFILE, "EMFILE"), (libc::ENFILE, "ENFILE"), (libc::EFBIG, "EFBIG"), (libc::EAGAIN, "EAGAIN"),
            (libc::EINTR, "EINTR"), (libc::ESTALE, "ESTALE"), (libc::ECHILD, "ECHILD"),
        ] {
            assert_eq!(errno_name(code), name);
        }
        // a number the table does not name is `E<n>`, as builder trace.py names it
        assert_eq!(errno_name(9999), "E9999");
        for api in [Api::Sync, Api::PosixAio, Api::LibAio, Api::Uring, Api::Mmap] {
            assert_eq!(Api::parse(api.name()), Some(api));
        }
        assert_eq!(BackendKind::of(Api::Mmap, Cache::Direct), None);
        // what an abstract declares (names `aeiou check` has refused say so here too)
        let ast = |posix: &str, cache: &str| -> crate::ast::Ast {
            serde_json::from_str(&format!(r#"{{"ast": "0.13", "name": "t", "actors": {{}}{posix}{cache}}}"#)).unwrap()
        };
        assert_eq!(declared(&ast("", "")).unwrap(), BackendKind::Sync);
        assert_eq!(declared(&ast(r#", "posix": "libaio""#, r#", "cache": "direct""#)).unwrap(), BackendKind::LibAioDirect);
        assert_eq!(declared(&ast(r#", "posix": "aio""#, "")).unwrap_err().to_string(), format!("posix `aio` is not one of {API_NAMES}"));
        assert_eq!(declared(&ast("", r#", "cache": "none""#)).unwrap_err().to_string(), format!("cache `none` is not one of {CACHE_NAMES}"));
        assert_eq!(declared(&ast(r#", "posix": "mmap""#, r#", "cache": "direct""#)).unwrap_err().to_string(), "cache `direct` under posix `mmap`");
        assert_eq!(BackendKind::PosixAioDirect.describe(), "posix posix-aio  cache direct");
    }
}
