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
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::ast::{Advice, IoctlRequest, OpenFlag, Whence};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Sync,
    SyncDirect,
    /// `io_uring` through the `io-uring` crate: an event loop per thread multiplexing many
    /// actors over one ring (`uring.rs`).
    Uring,
    UringDirect,
    /// glibc POSIX AIO, one request at a time per actor thread.
    PosixAio,
    PosixAioDirect,
    /// The kernel AIO system calls (`io_setup`/`io_submit`/`io_getevents`, what the libaio
    /// library wraps) on the event loop (`aio.rs`).
    LibAio,
    LibAioDirect,
    /// Reads are copies out of a shared mapping; everything else is `sync`.
    Mmap,
}

pub const NAMES: &str = "sync, sync-direct, io_uring, io_uring-direct, posix-aio, posix-aio-direct, libaio, libaio-direct, mmap";

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
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<BackendKind> {
        match s {
            "sync" => Some(BackendKind::Sync),
            "sync-direct" => Some(BackendKind::SyncDirect),
            "io_uring" | "io-uring" => Some(BackendKind::Uring),
            "io_uring-direct" | "io-uring-direct" => Some(BackendKind::UringDirect),
            "posix-aio" => Some(BackendKind::PosixAio),
            "posix-aio-direct" => Some(BackendKind::PosixAioDirect),
            "libaio" => Some(BackendKind::LibAio),
            "libaio-direct" => Some(BackendKind::LibAioDirect),
            "mmap" => Some(BackendKind::Mmap),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Sync => "sync",
            BackendKind::SyncDirect => "sync-direct",
            BackendKind::Uring => "io_uring",
            BackendKind::UringDirect => "io_uring-direct",
            BackendKind::PosixAio => "posix-aio",
            BackendKind::PosixAioDirect => "posix-aio-direct",
            BackendKind::LibAio => "libaio",
            BackendKind::LibAioDirect => "libaio-direct",
            BackendKind::Mmap => "mmap",
        }
    }

    /// `O_DIRECT` on every regular-file open.
    pub fn direct(self) -> bool {
        matches!(self, BackendKind::SyncDirect | BackendKind::UringDirect | BackendKind::PosixAioDirect | BackendKind::LibAioDirect)
    }

    /// Runs on the event loop (`uring.rs`) rather than one thread per actor.
    pub fn uring(self) -> bool {
        matches!(self, BackendKind::Uring | BackendKind::UringDirect)
    }

    /// The kernel AIO system calls on the event loop (`aio.rs`).
    pub fn libaio(self) -> bool {
        matches!(self, BackendKind::LibAio | BackendKind::LibAioDirect)
    }

    /// One event loop per thread multiplexing the actors, rather than one thread per actor.
    pub fn event_loop(self) -> bool {
        self.uring() || self.libaio()
    }

    /// The blocking form: the backend itself for `sync`, `posix-aio`, and `mmap`, and what an
    /// event loop uses inline for the ops its API cannot carry.
    pub fn make(self, mmap: MmapMode, consume: MmapConsume, stats: &Arc<MmapStats>) -> Box<dyn Backend> {
        let sync = Sync { direct: self.direct() };
        match self {
            BackendKind::PosixAio | BackendKind::PosixAioDirect => Box::new(PosixAio { sync }),
            BackendKind::Mmap => Box::new(Mmap { sync, mode: mmap, consume, stats: stats.clone() }),
            _ => Box::new(sync),
        }
    }
}

pub const ALIGN: usize = 4096;

pub trait Backend: Send {
    fn name(&self) -> &'static str;
    /// The backend's API has no file position: the driver passes every read and write its
    /// effective offset (the one the VM computed and the fingerprint hashes), as the event
    /// loops do (`uring.rs`).
    fn positional(&self) -> bool {
        false
    }
    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OwnedFd>;
    /// `read(2)` at the file position, or `pread(2)` at `offset`.
    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize>;
    fn write(&mut self, fd: BorrowedFd, buf: &[u8], offset: Option<i64>) -> io::Result<usize>;
    fn lseek(&mut self, fd: BorrowedFd, offset: i64, whence: Whence) -> io::Result<i64>;
    fn ioctl(&mut self, fd: BorrowedFd, request: IoctlRequest) -> io::Result<()>;
    /// `posix_fadvise(2)` over `[offset, offset + len)` (`len` 0: to the end of the file).
    fn fadvise(&mut self, fd: BorrowedFd, offset: i64, len: i64, advice: Advice) -> io::Result<()>;
    /// Returns `st_size`.
    fn fstat(&mut self, fd: BorrowedFd) -> io::Result<i64>;
    fn stat(&mut self, path: &Path) -> io::Result<i64>;
    fn fsync(&mut self, fd: BorrowedFd) -> io::Result<()>;
    fn fdatasync(&mut self, fd: BorrowedFd) -> io::Result<()>;
    fn unlink(&mut self, path: &Path) -> io::Result<()>;
    fn ftruncate(&mut self, fd: BorrowedFd, len: i64) -> io::Result<()>;
    fn fallocate(&mut self, fd: BorrowedFd, offset: i64, len: i64) -> io::Result<()>;
    fn mkdir(&mut self, path: &Path, mode: u32) -> io::Result<()>;
    fn rmdir(&mut self, path: &Path) -> io::Result<()>;
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()>;
    /// Read the whole directory through `getdents64`; returns the number of entries, not
    /// counting `.`, `..`, and `.aeiou*` (`schema/README.md` §6).
    fn readdir(&mut self, fd: BorrowedFd) -> io::Result<usize>;
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

pub fn errno_name(code: i32) -> &'static str {
    match code {
        libc::ENOENT => "ENOENT",
        libc::EEXIST => "EEXIST",
        libc::ENOTTY => "ENOTTY",
        libc::EISDIR => "EISDIR",
        libc::ENOTDIR => "ENOTDIR",
        libc::ENOTEMPTY => "ENOTEMPTY",
        libc::EACCES => "EACCES",
        libc::EPERM => "EPERM",
        libc::EINVAL => "EINVAL",
        libc::ENOSPC => "ENOSPC",
        libc::EBADF => "EBADF",
        libc::ENAMETOOLONG => "ENAMETOOLONG",
        libc::EIO => "EIO",
        libc::EOPNOTSUPP => "EOPNOTSUPP",
        libc::EXDEV => "EXDEV",
        libc::EBUSY => "EBUSY",
        libc::EMFILE => "EMFILE",
        libc::ENFILE => "ENFILE",
        libc::EFBIG => "EFBIG",
        libc::EAGAIN => "EAGAIN",
        libc::EINTR => "EINTR",
        libc::ESTALE => "ESTALE",
        _ => "E?",
    }
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
    fn name(&self) -> &'static str {
        if self.direct { "sync-direct" } else { "sync" }
    }

    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OwnedFd> {
        let c = cpath(path)?;
        let mut f = open_flags(flags);
        if self.direct && f & libc::O_DIRECTORY == 0 {
            f |= libc::O_DIRECT;
        }
        let fd = unsafe { libc::open(c.as_ptr(), f, mode as libc::c_uint) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize> {
        let fd = file.as_fd();
        let r = match offset {
            Some(off) => unsafe { libc::pread(fd.as_raw_fd(), buf.as_mut_ptr() as *mut _, buf.len(), off) },
            None => unsafe { libc::read(fd.as_raw_fd(), buf.as_mut_ptr() as *mut _, buf.len()) },
        };
        check(r as libc::c_long).map(|n| n as usize)
    }

    fn write(&mut self, fd: BorrowedFd, buf: &[u8], offset: Option<i64>) -> io::Result<usize> {
        let r = match offset {
            Some(off) => unsafe { libc::pwrite(fd.as_raw_fd(), buf.as_ptr() as *const _, buf.len(), off) },
            None => unsafe { libc::write(fd.as_raw_fd(), buf.as_ptr() as *const _, buf.len()) },
        };
        check(r as libc::c_long).map(|n| n as usize)
    }

    fn lseek(&mut self, fd: BorrowedFd, offset: i64, whence: Whence) -> io::Result<i64> {
        let w = match whence {
            Whence::SET => libc::SEEK_SET,
            Whence::CUR => libc::SEEK_CUR,
            Whence::END => libc::SEEK_END,
        };
        check(unsafe { libc::lseek(fd.as_raw_fd(), offset, w) } as libc::c_long).map(|n| n as i64)
    }

    fn ioctl(&mut self, fd: BorrowedFd, request: IoctlRequest) -> io::Result<()> {
        let r = match request {
            IoctlRequest::TCGETS => {
                let mut t: libc::termios = unsafe { std::mem::zeroed() };
                unsafe { libc::ioctl(fd.as_raw_fd(), libc::TCGETS, &mut t as *mut _) }
            }
            IoctlRequest::FIONREAD => {
                let mut n: libc::c_int = 0;
                unsafe { libc::ioctl(fd.as_raw_fd(), libc::FIONREAD, &mut n as *mut _) }
            }
            IoctlRequest::BLKGETSIZE64 => {
                let mut n: u64 = 0;
                // BLKGETSIZE64 = _IOR(0x12, 114, size_t)
                unsafe { libc::ioctl(fd.as_raw_fd(), 0x8008_1272, &mut n as *mut _) }
            }
        };
        check_int(r)
    }

    fn fadvise(&mut self, fd: BorrowedFd, offset: i64, len: i64, advice: Advice) -> io::Result<()> {
        let adv = match advice {
            Advice::NORMAL => libc::POSIX_FADV_NORMAL,
            Advice::RANDOM => libc::POSIX_FADV_RANDOM,
            Advice::SEQUENTIAL => libc::POSIX_FADV_SEQUENTIAL,
            Advice::WILLNEED => libc::POSIX_FADV_WILLNEED,
            Advice::DONTNEED => libc::POSIX_FADV_DONTNEED,
            Advice::NOREUSE => libc::POSIX_FADV_NOREUSE,
        };
        // returns the errno directly, not -1
        let r = unsafe { libc::posix_fadvise(fd.as_raw_fd(), offset, len, adv) };
        if r != 0 { Err(io::Error::from_raw_os_error(r)) } else { Ok(()) }
    }

    fn fstat(&mut self, fd: BorrowedFd) -> io::Result<i64> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        check_int(unsafe { libc::fstat(fd.as_raw_fd(), &mut st) })?;
        Ok(st.st_size)
    }

    fn stat(&mut self, path: &Path) -> io::Result<i64> {
        let c = cpath(path)?;
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        check_int(unsafe { libc::stat(c.as_ptr(), &mut st) })?;
        Ok(st.st_size)
    }

    fn fsync(&mut self, fd: BorrowedFd) -> io::Result<()> {
        check_int(unsafe { libc::fsync(fd.as_raw_fd()) })
    }

    fn fdatasync(&mut self, fd: BorrowedFd) -> io::Result<()> {
        check_int(unsafe { libc::fdatasync(fd.as_raw_fd()) })
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        let c = cpath(path)?;
        check_int(unsafe { libc::unlink(c.as_ptr()) })
    }

    fn ftruncate(&mut self, fd: BorrowedFd, len: i64) -> io::Result<()> {
        check_int(unsafe { libc::ftruncate(fd.as_raw_fd(), len) })
    }

    fn fallocate(&mut self, fd: BorrowedFd, offset: i64, len: i64) -> io::Result<()> {
        check_int(unsafe { libc::fallocate(fd.as_raw_fd(), 0, offset, len) })
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

    fn readdir(&mut self, fd: BorrowedFd) -> io::Result<usize> {
        let mut buf = vec![0u8; 32 * 1024];
        let mut entries = 0usize;
        loop {
            let n = unsafe { libc::syscall(libc::SYS_getdents64, fd.as_raw_fd(), buf.as_mut_ptr(), buf.len()) };
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
    fn name(&self) -> &'static str {
        if self.sync.direct { "posix-aio-direct" } else { "posix-aio" }
    }

    fn positional(&self) -> bool {
        true
    }

    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OwnedFd> {
        self.sync.open(path, flags, mode)
    }

    fn read(&mut self, file: &OpenFile, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize> {
        let fd = file.as_fd();
        let Some(off) = offset else { return self.sync.read(file, buf, None) };
        let mut cb = Self::cb(fd, buf.as_mut_ptr(), buf.len(), off);
        check_int(unsafe { libc::aio_read(&mut cb) })?;
        Self::wait(&mut cb).map(|n| n as usize)
    }

    fn write(&mut self, fd: BorrowedFd, buf: &[u8], offset: Option<i64>) -> io::Result<usize> {
        let Some(off) = offset else { return self.sync.write(fd, buf, None) };
        let mut cb = Self::cb(fd, buf.as_ptr() as *mut u8, buf.len(), off);
        check_int(unsafe { libc::aio_write(&mut cb) })?;
        Self::wait(&mut cb).map(|n| n as usize)
    }

    fn lseek(&mut self, fd: BorrowedFd, offset: i64, whence: Whence) -> io::Result<i64> {
        self.sync.lseek(fd, offset, whence)
    }

    fn ioctl(&mut self, fd: BorrowedFd, request: IoctlRequest) -> io::Result<()> {
        self.sync.ioctl(fd, request)
    }

    fn fadvise(&mut self, fd: BorrowedFd, offset: i64, len: i64, advice: Advice) -> io::Result<()> {
        self.sync.fadvise(fd, offset, len, advice)
    }

    fn fstat(&mut self, fd: BorrowedFd) -> io::Result<i64> {
        self.sync.fstat(fd)
    }

    fn stat(&mut self, path: &Path) -> io::Result<i64> {
        self.sync.stat(path)
    }

    fn fsync(&mut self, fd: BorrowedFd) -> io::Result<()> {
        let mut cb = Self::cb(fd, std::ptr::null_mut(), 0, 0);
        check_int(unsafe { libc::aio_fsync(libc::O_SYNC, &mut cb) })?;
        Self::wait(&mut cb).map(|_| ())
    }

    fn fdatasync(&mut self, fd: BorrowedFd) -> io::Result<()> {
        let mut cb = Self::cb(fd, std::ptr::null_mut(), 0, 0);
        check_int(unsafe { libc::aio_fsync(libc::O_DSYNC, &mut cb) })?;
        Self::wait(&mut cb).map(|_| ())
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        self.sync.unlink(path)
    }

    fn ftruncate(&mut self, fd: BorrowedFd, len: i64) -> io::Result<()> {
        self.sync.ftruncate(fd, len)
    }

    fn fallocate(&mut self, fd: BorrowedFd, offset: i64, len: i64) -> io::Result<()> {
        self.sync.fallocate(fd, offset, len)
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

    fn readdir(&mut self, fd: BorrowedFd) -> io::Result<usize> {
        self.sync.readdir(fd)
    }
}

// ---------------------------------------------------------------- mmap

/// `MADV_POPULATE_READ` (Linux 5.14).
const MADV_POPULATE_READ: libc::c_int = 22;

struct Mapping {
    ptr: *mut u8,
    len: usize,
}

/// An open file as the actors hold it: the descriptor and, under the `mmap` backend, the
/// mapping of it. The mapping belongs to the open file, not to the actor that reads: every
/// actor that sees the descriptor (the one that opened it and the sub-actors that inherited
/// it at a fork, which are threads of one address space) reads through the one mapping,
/// made by whichever of them reads first and unmapped when the last of them lets the
/// descriptor go, just before it is closed.
///
/// Readers find the current mapping without a lock. A file that has grown past it is mapped
/// again at its new size, under the lock, and the earlier mapping stays until the close,
/// since another sub-actor may be inside it.
pub struct OpenFile {
    fd: OwnedFd,
    map: AtomicPtr<Mapping>,
    all: Mutex<Vec<Box<Mapping>>>,
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
        OpenFile { fd, map: AtomicPtr::new(std::ptr::null_mut()), all: Mutex::new(Vec::new()) }
    }
}

impl std::ops::Deref for OpenFile {
    type Target = OwnedFd;
    fn deref(&self) -> &OwnedFd {
        &self.fd
    }
}

impl Drop for OpenFile {
    fn drop(&mut self) {
        OPEN_FILES.fetch_sub(1, Ordering::Relaxed);
        for m in self.all.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
            unsafe { libc::munmap(m.ptr as *mut libc::c_void, m.len) };
        }
    }
}

/// Reads through a mapping: the first read of an open file maps the whole of it
/// (`PROT_READ`, `MAP_SHARED`, the size from `fstat`), a read makes its range of the
/// mapping resident (one byte of every page is read, or under `MmapConsume::Copy` the range
/// is copied into the actor's buffer), and the close unmaps (`OpenFile`: one mapping per
/// open, shared with the sub-actors that inherit the descriptor). That is the shape of a
/// safetensors or Arrow load: `open`, `fstat`, `mmap`, the consumer's accesses, `munmap`,
/// `close`. The read returns only when every page of the range has arrived: a fault does
/// not return before its page is read, and `MADV_POPULATE_READ` does not return before
/// the range is. A read
/// past the mapped length asks `fstat` again and remaps when the file has grown; a read at
/// or past the end returns what `pread` would. Writes and every other op are `sync` (no
/// loader writes through a mapping).
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
    /// The descriptor's mapping, made or remade so it covers `end` when the file does.
    fn mapping(&mut self, file: &OpenFile, end: usize) -> io::Result<(*mut u8, usize)> {
        // SAFETY: a published mapping lives in `file.all` until the file is dropped
        let current = |p: *mut Mapping| if p.is_null() { None } else { Some(unsafe { ((*p).ptr, (*p).len) }) };
        if let Some((ptr, len)) = current(file.map.load(Ordering::Acquire)) {
            if end <= len {
                return Ok((ptr, len));
            }
        }
        // no mapping yet, or a read past it: one actor at a time asks the size and maps
        let mut all = file.all.lock().unwrap_or_else(|e| e.into_inner());
        let now = current(file.map.load(Ordering::Acquire));
        if let Some((ptr, len)) = now {
            if end <= len {
                return Ok((ptr, len));
            }
        }
        let size = self.sync.fstat(file.as_fd())? as usize;
        if let Some((ptr, len)) = now {
            if len == size {
                return Ok((ptr, len));
            }
        }
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
}

impl Backend for Mmap {
    fn name(&self) -> &'static str {
        "mmap"
    }

    fn positional(&self) -> bool {
        true
    }

    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OwnedFd> {
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

    fn write(&mut self, fd: BorrowedFd, buf: &[u8], offset: Option<i64>) -> io::Result<usize> {
        self.sync.write(fd, buf, offset)
    }

    fn lseek(&mut self, fd: BorrowedFd, offset: i64, whence: Whence) -> io::Result<i64> {
        self.sync.lseek(fd, offset, whence)
    }

    fn ioctl(&mut self, fd: BorrowedFd, request: IoctlRequest) -> io::Result<()> {
        self.sync.ioctl(fd, request)
    }

    fn fadvise(&mut self, fd: BorrowedFd, offset: i64, len: i64, advice: Advice) -> io::Result<()> {
        self.sync.fadvise(fd, offset, len, advice)
    }

    fn fstat(&mut self, fd: BorrowedFd) -> io::Result<i64> {
        self.sync.fstat(fd)
    }

    fn stat(&mut self, path: &Path) -> io::Result<i64> {
        self.sync.stat(path)
    }

    fn fsync(&mut self, fd: BorrowedFd) -> io::Result<()> {
        self.sync.fsync(fd)
    }

    fn fdatasync(&mut self, fd: BorrowedFd) -> io::Result<()> {
        self.sync.fdatasync(fd)
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        self.sync.unlink(path)
    }

    fn ftruncate(&mut self, fd: BorrowedFd, len: i64) -> io::Result<()> {
        self.sync.ftruncate(fd, len)
    }

    fn fallocate(&mut self, fd: BorrowedFd, offset: i64, len: i64) -> io::Result<()> {
        self.sync.fallocate(fd, offset, len)
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

    fn readdir(&mut self, fd: BorrowedFd) -> io::Result<usize> {
        self.sync.readdir(fd)
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
        // a read at the end asks the size and maps nothing; a grown file is mapped again
        assert_eq!(a.read(&file, &mut buf, Some(8192)).unwrap(), 0);
        assert_eq!(stats.maps.load(Ordering::Relaxed), 1);
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(&[9u8; 4096]).unwrap();
        assert_eq!(a.read(&file, &mut buf, Some(8192)).unwrap(), 4096);
        assert_eq!(buf[0], 9);
        assert_eq!((stats.maps.load(Ordering::Relaxed), stats.mapped_bytes.load(Ordering::Relaxed)), (2, 8192 + 12288));
        // a second open of the same path is its own mapping
        let other = OpenFile::from(a.open(&path, 1 << (OpenFlag::RDONLY as u8), 0).unwrap());
        assert_eq!(a.read(&other, &mut buf, Some(0)).unwrap(), 4096);
        assert_eq!(stats.maps.load(Ordering::Relaxed), 3);
        std::fs::remove_file(&path).unwrap();
    }
}
