//! The capture's `LD_PRELOAD` library, `libaeiou_capture.so` (`DESIGN_REVIEW.md` §3.96, phase 2
//! of brief item 54): `aeiou-trace capture --requests shim` starts the command with it, and it
//! records the heads of the HTTP requests an object client sends and the responses it gets,
//! where it can see them in plaintext: through a shared libssl (`SSL_write`, `SSL_read`, and
//! their `_ex` forms, the connection the `SSL *`) and through libc's calls on a socket (`send`,
//! `write`, `recv`, `read`, their vector and fortified forms). It only observes: every hook
//! passes the call on unchanged, its result and `errno` the real one's.
//!
//! Nothing it does is a system call the application's trace would show, but for its start in
//! each process: the log (`AEIOU_CAPTURE_LOG`, made by the capture) opened, mapped, and closed.
//! Records go into the mapped log (`aeiou_interpose::log`); a descriptor is known for a socket
//! by the `connect` that made it one (never an `fstat`), the thread id is asked once per thread
//! (`gettid`, outside every traced class), the time read through the vDSO. A head's text keeps
//! the request line, with a presigned URL's signature and credential redacted, and only the
//! headers that describe the request (`KEEP`, and `x-amz-` ones but the secret ones): never
//! `Authorization`, a session token, a customer's encryption key, a cookie, or user metadata.
/// The most of a buffer looked at for a head.
pub const LOOK: usize = 8192;
/// The most of a head's text kept.
pub const KEPT: usize = 2048;

const METHODS: [&[u8]; 7] = [b"GET ", b"PUT ", b"HEAD ", b"POST ", b"DELETE ", b"OPTIONS ", b"PATCH "];

/// The headers kept, besides `x-amz-` ones.
const KEEP: [&str; 18] = [
    "host",
    "range",
    "content-length",
    "content-range",
    "content-type",
    "content-encoding",
    "transfer-encoding",
    "etag",
    "last-modified",
    "if-match",
    "if-none-match",
    "if-modified-since",
    "if-unmodified-since",
    "if-range",
    "expect",
    "connection",
    "location",
    "retry-after",
];

/// The query parameters of a presigned URL that are secrets, redacted.
const SECRET_PARAMS: [&str; 8] = ["x-amz-signature", "x-amz-credential", "x-amz-security-token", "signature", "awsaccesskeyid", "x-goog-signature", "x-goog-credential", "sig"];

fn keep(name: &[u8]) -> bool {
    let Ok(n) = std::str::from_utf8(name) else { return false };
    let n = n.trim();
    if KEEP.iter().any(|k| k.eq_ignore_ascii_case(n)) {
        return true;
    }
    let lower = |s: &str, p: &str| s.len() >= p.len() && s[..p.len()].eq_ignore_ascii_case(p);
    lower(n, "x-amz-")
        && !lower(n, "x-amz-meta-")
        && !lower(n, "x-amz-security-token")
        && !lower(n, "x-amz-content-sha256")
        && !lower(n, "x-amz-server-side-encryption-customer-key")
        && !lower(n, "x-amz-copy-source-server-side-encryption-customer-key")
}

/// Whether `data` begins a head: a request line when sent, a status line when received.
pub fn starts_head(dir: u8, data: &[u8]) -> bool {
    if dir == aeiou_interpose::capture::SENT {
        METHODS.iter().any(|m| data.starts_with(m))
    } else {
        data.starts_with(b"HTTP/1.")
    }
}

struct Out<'a> {
    buf: &'a mut [u8],
    len: usize,
    full: bool,
}

impl Out<'_> {
    fn put(&mut self, parts: &[&[u8]]) {
        let n: usize = parts.iter().map(|p| p.len()).sum();
        if self.full || self.len + n > self.buf.len() {
            self.full = true;
            return;
        }
        for p in parts {
            self.buf[self.len..self.len + p.len()].copy_from_slice(p);
            self.len += p.len();
        }
    }
}

/// The request line with a presigned URL's secrets redacted.
fn request_line(line: &[u8], out: &mut Out) {
    let Some(q) = line.iter().position(|&c| c == b'?') else { return out.put(&[line, b"\n"]) };
    // the query ends at the space before the version
    let end = line[q..].iter().position(|&c| c == b' ').map_or(line.len(), |i| q + i);
    out.put(&[&line[..=q]]);
    for (i, param) in line[q + 1..end].split(|&c| c == b'&').enumerate() {
        if i > 0 {
            out.put(&[b"&"]);
        }
        let name = param.split(|&c| c == b'=').next().unwrap_or(b"");
        let secret = std::str::from_utf8(name).is_ok_and(|n| SECRET_PARAMS.iter().any(|s| s.eq_ignore_ascii_case(n)));
        if secret {
            out.put(&[name, b"=REDACTED"]);
        } else {
            out.put(&[param]);
        }
    }
    out.put(&[&line[end..], b"\n"]);
}

/// The kept text of the head `data` begins (`starts_head`), one line each, `\n`-ended, into
/// `out`: its length, and whether the head was cut short (its end not within `LOOK` bytes or
/// the buffer, or its text longer than `out`).
pub fn filter_head(dir: u8, data: &[u8], out: &mut [u8]) -> (usize, bool) {
    let data = &data[..data.len().min(LOOK)];
    let (head, ended) = match data.windows(4).position(|w| w == b"\r\n\r\n") {
        Some(e) => (&data[..e], true),
        None => (data, false),
    };
    let mut o = Out { buf: out, len: 0, full: false };
    for (i, line) in head.split(|&c| c == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if i == 0 {
            if dir == aeiou_interpose::capture::SENT {
                request_line(line, &mut o);
            } else {
                o.put(&[line, b"\n"]);
            }
        } else if let Some(colon) = line.iter().position(|&c| c == b':') {
            if keep(&line[..colon]) {
                o.put(&[line, b"\n"]);
            }
        }
    }
    (o.len, !ended || o.full)
}

#[cfg(not(test))]
mod hooks {
    use super::*;
    use aeiou_interpose::capture::{self as rec, Stamp};
    use aeiou_interpose::log::Log;
    use aeiou_interpose::{next, Guard};
    use libc::{c_char, c_int, iovec, msghdr, size_t, sockaddr, socklen_t, ssize_t};
    use std::cell::Cell;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

    static LOG: AtomicPtr<Log> = AtomicPtr::new(std::ptr::null_mut());
    static PID: AtomicU32 = AtomicU32::new(0);
    thread_local! {
        static TID: Cell<u32> = const { Cell::new(0) };
        /// The socket an SSL call of this thread wrote or read under it: an SSL connection's
        /// descriptor where `SSL_get_fd` has none (curl's own BIO).
        static UNDER: Cell<c_int> = const { Cell::new(-1) };
    }

    /// The descriptors a `connect` made sockets of, one bit each, up to `MAX_FD`.
    const MAX_FD: usize = 1 << 20;
    static SOCKETS: [AtomicU64; MAX_FD / 64] = [const { AtomicU64::new(0) }; MAX_FD / 64];

    fn is_socket(fd: c_int) -> bool {
        (0..MAX_FD as c_int).contains(&fd) && SOCKETS[fd as usize / 64].load(Ordering::Relaxed) & (1 << (fd % 64)) != 0
    }

    fn mark(fd: c_int, on: bool) {
        if (0..MAX_FD as c_int).contains(&fd) {
            let bit = 1u64 << (fd % 64);
            if on {
                SOCKETS[fd as usize / 64].fetch_or(bit, Ordering::Relaxed);
            } else {
                SOCKETS[fd as usize / 64].fetch_and(!bit, Ordering::Relaxed);
            }
        }
    }

    fn log() -> Option<&'static Log> {
        unsafe { LOG.load(Ordering::Acquire).as_ref() }
    }

    fn stamp() -> Stamp {
        let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
        unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
        let tid = TID.with(|t| {
            if t.get() == 0 {
                t.set(unsafe { libc::syscall(libc::SYS_gettid) } as u32);
            }
            t.get()
        });
        Stamp { time_ns: ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64, pid: PID.load(Ordering::Relaxed), tid }
    }

    fn process(how: u32) {
        let Some(log) = log() else { return };
        let exe = unsafe { libc::getauxval(libc::AT_EXECFN) } as *const c_char;
        let exe = if exe.is_null() { &[][..] } else { unsafe { std::ffi::CStr::from_ptr(exe) }.to_bytes() };
        log.append(rec::PROCESS, &[&stamp().bytes(), &rec::process(unsafe { libc::getppid() } as u32, how), exe]);
    }

    extern "C" fn forked() {
        PID.store(unsafe { libc::getpid() } as u32, Ordering::Relaxed);
        TID.with(|t| t.set(0));
        process(rec::FORKED);
    }

    /// At load: map the log the capture named, and say this process has the library.
    extern "C" fn init() {
        let Some(_g) = Guard::enter() else { return };
        let path = unsafe { libc::getenv(c"AEIOU_CAPTURE_LOG".as_ptr()) };
        if path.is_null() {
            return;
        }
        let fd = unsafe { libc::open(path, libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            return;
        }
        let mapped = unsafe { Log::map(fd) };
        unsafe { libc::close(fd) };
        let Some(log) = mapped else { return };
        PID.store(unsafe { libc::getpid() } as u32, Ordering::Relaxed);
        LOG.store(Box::into_raw(Box::new(log)), Ordering::Release);
        unsafe { libc::pthread_atfork(None, None, Some(forked)) };
        process(rec::LOADED);
    }

    #[used]
    #[link_section = ".init_array"]
    static INIT: extern "C" fn() = init;

    /// Record the head `data` begins, if it begins one, as of `at` (now when `None`).
    fn see(via: u8, dir: u8, fd: c_int, conn: u64, data: &[u8], at: Option<Stamp>) {
        let Some(log) = log() else { return };
        if !starts_head(dir, data) {
            return;
        }
        let at = at.unwrap_or_else(stamp);
        let mut out = [0u8; KEPT];
        let (n, cut) = filter_head(dir, data, &mut out);
        log.append(rec::HEAD, &[&at.bytes(), &rec::head(fd, via, dir, cut, conn), &out[..n]]);
    }

    /// The first `LOOK` bytes of `total` bytes spread over `iov`, when they begin a head.
    fn gathered(dir: u8, iov: *const iovec, count: usize, total: usize, out: &mut [u8; LOOK]) -> Option<usize> {
        if iov.is_null() {
            return None;
        }
        let iov = unsafe { std::slice::from_raw_parts(iov, count) };
        let mut n = 0;
        for v in iov {
            let take = v.iov_len.min(LOOK - n).min(total - n);
            if take == 0 {
                if n == LOOK || n == total {
                    break;
                }
                continue;
            }
            out[n..n + take].copy_from_slice(unsafe { std::slice::from_raw_parts(v.iov_base as *const u8, take) });
            n += take;
            // the first bytes decide: most of a socket's writes are bodies
            if n >= 8 && !starts_head(dir, &out[..n]) {
                return None;
            }
        }
        (n > 0).then_some(n)
    }

    fn see_vec(dir: u8, fd: c_int, iov: *const iovec, count: usize, total: usize) {
        if log().is_none() {
            return;
        }
        let mut buf = [0u8; LOOK];
        if let Some(n) = gathered(dir, iov, count, total, &mut buf) {
            see(rec::VIA_SOCKET, dir, fd, fd as u64, &buf[..n], None);
        }
    }

    fn bytes<'a>(buf: *const c_void, n: usize) -> &'a [u8] {
        if buf.is_null() || n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(buf as *const u8, n.min(LOOK)) }
    }

    fn errno() -> c_int {
        unsafe { *libc::__errno_location() }
    }

    fn set_errno(e: c_int) {
        unsafe { *libc::__errno_location() = e };
    }

    /// The real definition of a hooked function, found once.
    macro_rules! real {
        ($name:ident, fn($($t:ty),*) -> $r:ty) => {{
            static P: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
            let mut p = P.load(Ordering::Relaxed);
            if p.is_null() {
                let name = std::ffi::CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap();
                p = next(name, $name as *const c_void);
                P.store(p, Ordering::Relaxed);
            }
            if p.is_null() {
                None
            } else {
                Some(unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($t),*) -> $r>(p) })
            }
        }};
    }

    /// A socket's call inside a hook (an SSL call's): the socket under it, for that hook.
    fn under(fd: c_int) {
        UNDER.with(|u| u.set(fd));
    }

    /// A socket's outgoing call: its head seen first, then the call.
    macro_rules! out {
        ($fd:expr, $buf:expr, $n:expr) => {
            if is_socket($fd) {
                match Guard::enter() {
                    Some(_g) => see(rec::VIA_SOCKET, rec::SENT, $fd, $fd as u64, bytes($buf, $n), None),
                    None => under($fd),
                }
            }
        };
    }

    /// A socket's incoming call, after it: the head it read.
    macro_rules! inward {
        ($fd:expr, $buf:expr, $r:expr) => {
            if $r > 0 && is_socket($fd) {
                let e = errno();
                match Guard::enter() {
                    Some(_g) => see(rec::VIA_SOCKET, rec::RECEIVED, $fd, $fd as u64, bytes($buf, $r as usize), None),
                    None => under($fd),
                }
                set_errno(e);
            }
        };
    }

    #[no_mangle]
    pub unsafe extern "C" fn connect(fd: c_int, addr: *const sockaddr, len: socklen_t) -> c_int {
        let f = real!(connect, fn(c_int, *const sockaddr, socklen_t) -> c_int).unwrap();
        let r = f(fd, addr, len);
        let e = if r == 0 { 0 } else { errno() };
        if r == 0 || [libc::EINPROGRESS, libc::EALREADY, libc::EINTR, libc::EAGAIN].contains(&e) {
            mark(fd, true);
            if let (Some(log), Some(_g), false) = (log(), Guard::enter(), addr.is_null()) {
                let family = (*addr).sa_family;
                let mut a = [0u8; 16];
                let mut port = 0u16;
                if family as c_int == libc::AF_INET && len as usize >= std::mem::size_of::<libc::sockaddr_in>() {
                    let s = &*(addr as *const libc::sockaddr_in);
                    a[..4].copy_from_slice(&s.sin_addr.s_addr.to_ne_bytes());
                    port = u16::from_be(s.sin_port);
                } else if family as c_int == libc::AF_INET6 && len as usize >= std::mem::size_of::<libc::sockaddr_in6>() {
                    let s = &*(addr as *const libc::sockaddr_in6);
                    a = s.sin6_addr.s6_addr;
                    port = u16::from_be(s.sin6_port);
                }
                log.append(rec::CONNECT, &[&stamp().bytes(), &rec::connect(fd, e, family, port, &a)]);
            }
        }
        set_errno(e);
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn close(fd: c_int) -> c_int {
        mark(fd, false);
        real!(close, fn(c_int) -> c_int).unwrap()(fd)
    }

    #[no_mangle]
    pub unsafe extern "C" fn write(fd: c_int, buf: *const c_void, n: size_t) -> ssize_t {
        out!(fd, buf, n);
        real!(write, fn(c_int, *const c_void, size_t) -> ssize_t).unwrap()(fd, buf, n)
    }

    #[no_mangle]
    pub unsafe extern "C" fn send(fd: c_int, buf: *const c_void, n: size_t, flags: c_int) -> ssize_t {
        out!(fd, buf, n);
        real!(send, fn(c_int, *const c_void, size_t, c_int) -> ssize_t).unwrap()(fd, buf, n, flags)
    }

    #[no_mangle]
    pub unsafe extern "C" fn sendto(fd: c_int, buf: *const c_void, n: size_t, flags: c_int, addr: *const sockaddr, len: socklen_t) -> ssize_t {
        out!(fd, buf, n);
        real!(sendto, fn(c_int, *const c_void, size_t, c_int, *const sockaddr, socklen_t) -> ssize_t).unwrap()(fd, buf, n, flags, addr, len)
    }

    #[no_mangle]
    pub unsafe extern "C" fn writev(fd: c_int, iov: *const iovec, count: c_int) -> ssize_t {
        if is_socket(fd) {
            match Guard::enter() {
                Some(_g) => see_vec(rec::SENT, fd, iov, count.max(0) as usize, usize::MAX),
                None => under(fd),
            }
        }
        real!(writev, fn(c_int, *const iovec, c_int) -> ssize_t).unwrap()(fd, iov, count)
    }

    #[no_mangle]
    pub unsafe extern "C" fn sendmsg(fd: c_int, msg: *const msghdr, flags: c_int) -> ssize_t {
        if is_socket(fd) && !msg.is_null() {
            match Guard::enter() {
                Some(_g) => see_vec(rec::SENT, fd, (*msg).msg_iov, (*msg).msg_iovlen, usize::MAX),
                None => under(fd),
            }
        }
        real!(sendmsg, fn(c_int, *const msghdr, c_int) -> ssize_t).unwrap()(fd, msg, flags)
    }

    #[no_mangle]
    pub unsafe extern "C" fn read(fd: c_int, buf: *mut c_void, n: size_t) -> ssize_t {
        let r = real!(read, fn(c_int, *mut c_void, size_t) -> ssize_t).unwrap()(fd, buf, n);
        inward!(fd, buf, r);
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn __read_chk(fd: c_int, buf: *mut c_void, n: size_t, buflen: size_t) -> ssize_t {
        let r = real!(__read_chk, fn(c_int, *mut c_void, size_t, size_t) -> ssize_t).unwrap()(fd, buf, n, buflen);
        inward!(fd, buf, r);
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn recv(fd: c_int, buf: *mut c_void, n: size_t, flags: c_int) -> ssize_t {
        let r = real!(recv, fn(c_int, *mut c_void, size_t, c_int) -> ssize_t).unwrap()(fd, buf, n, flags);
        // a peek reads the same bytes again
        if flags & libc::MSG_PEEK == 0 {
            inward!(fd, buf, r);
        }
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn __recv_chk(fd: c_int, buf: *mut c_void, n: size_t, buflen: size_t, flags: c_int) -> ssize_t {
        let r = real!(__recv_chk, fn(c_int, *mut c_void, size_t, size_t, c_int) -> ssize_t).unwrap()(fd, buf, n, buflen, flags);
        if flags & libc::MSG_PEEK == 0 {
            inward!(fd, buf, r);
        }
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn recvfrom(fd: c_int, buf: *mut c_void, n: size_t, flags: c_int, addr: *mut sockaddr, len: *mut socklen_t) -> ssize_t {
        let r = real!(recvfrom, fn(c_int, *mut c_void, size_t, c_int, *mut sockaddr, *mut socklen_t) -> ssize_t).unwrap()(fd, buf, n, flags, addr, len);
        if flags & libc::MSG_PEEK == 0 {
            inward!(fd, buf, r);
        }
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn __recvfrom_chk(fd: c_int, buf: *mut c_void, n: size_t, buflen: size_t, flags: c_int, addr: *mut sockaddr, len: *mut socklen_t) -> ssize_t {
        let r = real!(__recvfrom_chk, fn(c_int, *mut c_void, size_t, size_t, c_int, *mut sockaddr, *mut socklen_t) -> ssize_t).unwrap()(fd, buf, n, buflen, flags, addr, len);
        if flags & libc::MSG_PEEK == 0 {
            inward!(fd, buf, r);
        }
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn readv(fd: c_int, iov: *const iovec, count: c_int) -> ssize_t {
        let r = real!(readv, fn(c_int, *const iovec, c_int) -> ssize_t).unwrap()(fd, iov, count);
        if r > 0 && is_socket(fd) {
            let e = errno();
            match Guard::enter() {
                Some(_g) => see_vec(rec::RECEIVED, fd, iov, count.max(0) as usize, r as usize),
                None => under(fd),
            }
            set_errno(e);
        }
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn recvmsg(fd: c_int, msg: *mut msghdr, flags: c_int) -> ssize_t {
        let r = real!(recvmsg, fn(c_int, *mut msghdr, c_int) -> ssize_t).unwrap()(fd, msg, flags);
        if r > 0 && flags & libc::MSG_PEEK == 0 && is_socket(fd) && !msg.is_null() {
            let e = errno();
            match Guard::enter() {
                Some(_g) => see_vec(rec::RECEIVED, fd, (*msg).msg_iov, (*msg).msg_iovlen, r as usize),
                None => under(fd),
            }
            set_errno(e);
        }
        r
    }

    // ------------------------------------------------ a shared libssl

    /// The descriptor of an SSL connection, -1 when it has none (or libssl has no `SSL_get_fd`).
    fn ssl_fd(ssl: *mut c_void) -> c_int {
        // not hooked: this stand-in only names the function `real!` looks past
        #[allow(non_snake_case)]
        unsafe extern "C" fn SSL_get_fd(_: *mut c_void) -> c_int {
            -1
        }
        match real!(SSL_get_fd, fn(*mut c_void) -> c_int) {
            Some(f) => unsafe { f(ssl) },
            None => -1,
        }
    }

    /// An SSL call: held in the guard across the real call, so the ciphertext its socket calls
    /// carry is not looked at and the socket under it is learned; the head recorded after it,
    /// stamped as of its start (a request's) or its end (a response's).
    unsafe fn ssl_call(ssl: *mut c_void, dir: u8, call: impl FnOnce() -> (c_int, *const c_void, usize)) -> c_int {
        let Some(g) = Guard::enter() else { return call().0 };
        UNDER.with(|u| u.set(-1));
        let before = stamp();
        let (r, buf, n) = call();
        if r > 0 && log().is_some() {
            let e = errno();
            let fd = match UNDER.with(Cell::get) {
                -1 => ssl_fd(ssl),
                fd => fd,
            };
            see(rec::VIA_SSL, dir, fd, ssl as u64, bytes(buf, n), (dir == rec::SENT).then_some(before));
            set_errno(e);
        }
        drop(g);
        r
    }

    #[no_mangle]
    #[allow(non_snake_case)]
    pub unsafe extern "C" fn SSL_write(ssl: *mut c_void, buf: *const c_void, n: c_int) -> c_int {
        let f = real!(SSL_write, fn(*mut c_void, *const c_void, c_int) -> c_int).unwrap();
        ssl_call(ssl, rec::SENT, || (f(ssl, buf, n), buf, n.max(0) as usize))
    }

    #[no_mangle]
    #[allow(non_snake_case)]
    pub unsafe extern "C" fn SSL_write_ex(ssl: *mut c_void, buf: *const c_void, n: size_t, written: *mut size_t) -> c_int {
        let f = real!(SSL_write_ex, fn(*mut c_void, *const c_void, size_t, *mut size_t) -> c_int).unwrap();
        ssl_call(ssl, rec::SENT, || (f(ssl, buf, n, written), buf, n))
    }

    #[no_mangle]
    #[allow(non_snake_case)]
    pub unsafe extern "C" fn SSL_read(ssl: *mut c_void, buf: *mut c_void, n: c_int) -> c_int {
        let f = real!(SSL_read, fn(*mut c_void, *mut c_void, c_int) -> c_int).unwrap();
        ssl_call(ssl, rec::RECEIVED, || {
            let r = f(ssl, buf, n);
            (r, buf, r.max(0) as usize)
        })
    }

    #[no_mangle]
    #[allow(non_snake_case)]
    pub unsafe extern "C" fn SSL_read_ex(ssl: *mut c_void, buf: *mut c_void, n: size_t, got: *mut size_t) -> c_int {
        let f = real!(SSL_read_ex, fn(*mut c_void, *mut c_void, size_t, *mut size_t) -> c_int).unwrap();
        ssl_call(ssl, rec::RECEIVED, || {
            let r = f(ssl, buf, n, got);
            (r, buf, if r > 0 && !got.is_null() { *got } else { 0 })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeiou_interpose::capture::{RECEIVED, SENT};

    fn kept(dir: u8, data: &[u8]) -> (String, bool) {
        let mut out = [0u8; KEPT];
        let (n, cut) = filter_head(dir, data, &mut out);
        (String::from_utf8(out[..n].to_vec()).unwrap(), cut)
    }

    #[test]
    fn a_request_keeps_its_line_and_the_headers_that_describe_it_and_never_a_secret() {
        let req = b"GET /bucket/a/b.bin?partNumber=2&X-Amz-Credential=AKIA%2F1&X-Amz-Signature=abc HTTP/1.1\r\n\
Host: 127.0.0.1:9000\r\n\
Authorization: AWS4-HMAC-SHA256 Credential=AKIA/x, Signature=deadbeef\r\n\
X-Amz-Security-Token: tok\r\n\
x-amz-content-sha256: UNSIGNED-PAYLOAD\r\n\
x-amz-date: 20261103T000000Z\r\n\
X-Amz-Meta-Owner: someone\r\n\
x-amz-server-side-encryption-customer-key: c2VjcmV0\r\n\
x-amz-server-side-encryption-customer-key-MD5: bWQ1\r\n\
Cookie: s=1\r\n\
Range: bytes=0-99\r\n\
\r\nbody";
        assert!(starts_head(SENT, req));
        let (text, cut) = kept(SENT, req);
        assert!(!cut);
        assert_eq!(text, "GET /bucket/a/b.bin?partNumber=2&X-Amz-Credential=REDACTED&X-Amz-Signature=REDACTED HTTP/1.1\nHost: 127.0.0.1:9000\nx-amz-date: 20261103T000000Z\nRange: bytes=0-99\n");
        for secret in ["AKIA", "deadbeef", "tok", "UNSIGNED", "someone", "c2VjcmV0", "bWQ1", "s=1", "abc"] {
            assert!(!text.contains(secret), "{secret} in {text}");
        }
    }

    #[test]
    fn a_response_keeps_its_status_and_the_answer_s_headers() {
        let resp = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-99/4096\r\nContent-Length: 100\r\nETag: \"e\"\r\nx-amz-request-id: R1\r\nSet-Cookie: k=v\r\nServer: MinIO\r\n\r\n";
        assert!(starts_head(RECEIVED, resp) && !starts_head(SENT, resp));
        assert_eq!(kept(RECEIVED, resp), ("HTTP/1.1 206 Partial Content\nContent-Range: bytes 0-99/4096\nContent-Length: 100\nETag: \"e\"\nx-amz-request-id: R1\n".into(), false));
    }

    #[test]
    fn a_head_without_its_end_or_too_long_to_keep_is_cut_and_says_so() {
        let (text, cut) = kept(SENT, b"PUT /b/k HTTP/1.1\r\nHost: h\r\nContent-Le");
        assert!(cut);
        assert_eq!(text, "PUT /b/k HTTP/1.1\nHost: h\n");
        let mut long = b"GET /".to_vec();
        long.extend(vec![b'a'; KEPT]);
        long.extend(b" HTTP/1.1\r\nHost: h\r\n\r\n");
        let (text, cut) = kept(SENT, &long);
        assert!(cut && text.is_empty(), "nothing kept of a line that does not fit");
        // a query without a version, and an empty parameter
        assert_eq!(kept(SENT, b"GET /b?&sig=1\r\n\r\n").0, "GET /b?&sig=REDACTED\n");
        assert!(!starts_head(SENT, b"GETX /") && !starts_head(RECEIVED, b"HTTP/2 200"));
        assert!(!keep(&[0xff]) && keep(b" content-length ") && !keep(b"x-amz-meta-a"));
    }
}
