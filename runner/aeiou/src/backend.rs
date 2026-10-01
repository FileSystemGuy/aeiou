//! I/O backends: how an op of the abstract is issued to the system under test. The op stream
//! is the same for every backend (`PROJECT_BRIEF.md` §4); a backend maps each op to an API.
//!
//! This is the blocking half: `sync` (buffered POSIX syscalls on the calling thread, which
//! under `aeiou run` is one OS thread per actor or sub-actor: the fidelity reference, what a
//! PyTorch worker does) and `sync-direct` (the same with `O_DIRECT` on every regular-file
//! open; reads and writes must then be 4 KiB-aligned in offset and length, and the runner's
//! buffers are). The asynchronous backends (`io_uring`, `libaio`, …) will sit behind the same
//! op vocabulary with an issue half and a completion source (`NAPKIN_MATH.md` §4.2).

use std::ffi::CString;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::path::Path;

use crate::ast::{Advice, IoctlRequest, OpenFlag, Whence};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Sync,
    SyncDirect,
    /// `io_uring` through the `io-uring` crate: an event loop per thread multiplexing many
    /// actors over one ring (`uring.rs`).
    Uring,
    UringDirect,
}

pub const NAMES: &str = "sync, sync-direct, io_uring, io_uring-direct";

impl BackendKind {
    pub fn parse(s: &str) -> Option<BackendKind> {
        match s {
            "sync" => Some(BackendKind::Sync),
            "sync-direct" => Some(BackendKind::SyncDirect),
            "io_uring" | "io-uring" => Some(BackendKind::Uring),
            "io_uring-direct" | "io-uring-direct" => Some(BackendKind::UringDirect),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Sync => "sync",
            BackendKind::SyncDirect => "sync-direct",
            BackendKind::Uring => "io_uring",
            BackendKind::UringDirect => "io_uring-direct",
        }
    }

    /// `O_DIRECT` on every regular-file open.
    pub fn direct(self) -> bool {
        matches!(self, BackendKind::SyncDirect | BackendKind::UringDirect)
    }

    /// Runs on the event loop (`uring.rs`) rather than one thread per actor.
    pub fn uring(self) -> bool {
        matches!(self, BackendKind::Uring | BackendKind::UringDirect)
    }

    /// The blocking form: the backend itself for `sync`, and what the event loop uses inline
    /// for the ops `io_uring` has no opcode for.
    pub fn make(self) -> Box<dyn Backend> {
        Box::new(Sync { direct: self.direct() })
    }
}

pub const ALIGN: usize = 4096;

pub trait Backend: Send {
    fn name(&self) -> &'static str;
    fn open(&mut self, path: &Path, flags: u64, mode: u32) -> io::Result<OwnedFd>;
    /// `read(2)` at the file position, or `pread(2)` at `offset`.
    fn read(&mut self, fd: BorrowedFd, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize>;
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

    fn read(&mut self, fd: BorrowedFd, buf: &mut [u8], offset: Option<i64>) -> io::Result<usize> {
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
