//! The records of the capture's library (`libaeiou_capture.so`, phase 2 of brief item 54) in
//! its log: written by the library, read by `aeiou-trace capture`, one definition for both.
//! Every record starts with a stamp: the wall clock (`CLOCK_REALTIME`, read through the vDSO,
//! no system call) in nanoseconds, the process id, and the thread id.

pub const PROCESS: u16 = 1;
pub const CONNECT: u16 = 2;
pub const HEAD: u16 = 3;

/// How a process came to have the library: loaded at its start (an `exec`), or a `fork`'s
/// child, which has its parent's mapping.
pub const LOADED: u32 = 0;
pub const FORKED: u32 = 1;

/// A head's way: through a shared libssl (`SSL_write`, `SSL_read`), or libc's calls on a socket.
pub const VIA_SSL: u8 = 1;
pub const VIA_SOCKET: u8 = 2;
pub const SENT: u8 = 0;
pub const RECEIVED: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Stamp {
    pub time_ns: u64,
    pub pid: u32,
    pub tid: u32,
}

impl Stamp {
    pub fn bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&self.time_ns.to_ne_bytes());
        b[8..12].copy_from_slice(&self.pid.to_ne_bytes());
        b[12..].copy_from_slice(&self.tid.to_ne_bytes());
        b
    }

    fn of(b: &[u8]) -> Stamp {
        Stamp { time_ns: u64_of(b, 0), pid: u32_of(b, 8), tid: u32_of(b, 12) }
    }
}

fn u64_of(b: &[u8], at: usize) -> u64 {
    u64::from_ne_bytes(b[at..at + 8].try_into().unwrap())
}

fn u32_of(b: &[u8], at: usize) -> u32 {
    u32::from_ne_bytes(b[at..at + 4].try_into().unwrap())
}

/// The fixed part of a process record after its stamp: the parent and how.
pub fn process(ppid: u32, how: u32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[..4].copy_from_slice(&ppid.to_ne_bytes());
    b[4..].copy_from_slice(&how.to_ne_bytes());
    b
}

/// A connect record after its stamp: the descriptor, the call's error (0, or its errno), the
/// address family, the port, and the address (4 bytes of IPv4 or 16 of IPv6; a Unix socket's
/// path is not kept).
pub fn connect(fd: i32, errno: i32, family: u16, port: u16, addr: &[u8; 16]) -> [u8; 28] {
    let mut b = [0u8; 28];
    b[..4].copy_from_slice(&fd.to_ne_bytes());
    b[4..8].copy_from_slice(&errno.to_ne_bytes());
    b[8..10].copy_from_slice(&family.to_ne_bytes());
    b[10..12].copy_from_slice(&port.to_ne_bytes());
    b[12..].copy_from_slice(addr);
    b
}

/// The fixed part of a head record after its stamp: the descriptor (-1 when an SSL connection
/// has none), the way, the direction, whether the head was cut short, and the connection
/// (the `SSL *`, or the descriptor). The head's kept text follows.
pub fn head(fd: i32, via: u8, dir: u8, truncated: bool, conn: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..4].copy_from_slice(&fd.to_ne_bytes());
    b[4] = via;
    b[5] = dir;
    b[6] = truncated as u8;
    b[8..].copy_from_slice(&conn.to_ne_bytes());
    b
}

/// A record read back.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Process { at: Stamp, ppid: u32, how: u32, exe: String },
    Connect { at: Stamp, fd: i32, errno: i32, family: u16, port: u16, addr: [u8; 16] },
    Head { at: Stamp, fd: i32, via: u8, dir: u8, truncated: bool, conn: u64, text: String },
    Other { kind: u16 },
}

/// A record's payload decoded; trailing padding (NULs) is dropped from its text.
pub fn decode(kind: u16, p: &[u8]) -> Record {
    let text = |b: &[u8]| {
        let end = b.iter().rposition(|&c| c != 0).map_or(0, |i| i + 1);
        String::from_utf8_lossy(&b[..end]).into_owned()
    };
    match kind {
        PROCESS if p.len() >= 24 => Record::Process { at: Stamp::of(p), ppid: u32_of(p, 16), how: u32_of(p, 20), exe: text(&p[24..]) },
        CONNECT if p.len() >= 44 => Record::Connect {
            at: Stamp::of(p),
            fd: u32_of(p, 16) as i32,
            errno: u32_of(p, 20) as i32,
            family: u16::from_ne_bytes(p[24..26].try_into().unwrap()),
            port: u16::from_ne_bytes(p[26..28].try_into().unwrap()),
            addr: p[28..44].try_into().unwrap(),
        },
        HEAD if p.len() >= 32 => Record::Head { at: Stamp::of(p), fd: u32_of(p, 16) as i32, via: p[20], dir: p[21], truncated: p[22] != 0, conn: u64_of(p, 24), text: text(&p[32..]) },
        k => Record::Other { kind: k },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_record_reads_back_as_written() {
        let at = Stamp { time_ns: 1_700_000_000_123_456_789, pid: 42, tid: 43 };
        let mut p = at.bytes().to_vec();
        p.extend(process(1, FORKED));
        p.extend(b"/usr/bin/curl\0\0\0");
        assert_eq!(decode(PROCESS, &p), Record::Process { at, ppid: 1, how: FORKED, exe: "/usr/bin/curl".into() });

        let mut addr = [0u8; 16];
        addr[..4].copy_from_slice(&[127, 0, 0, 1]);
        let mut p = at.bytes().to_vec();
        p.extend(connect(5, 115, libc::AF_INET as u16, 9000, &addr));
        assert_eq!(decode(CONNECT, &p), Record::Connect { at, fd: 5, errno: 115, family: libc::AF_INET as u16, port: 9000, addr });

        let mut p = at.bytes().to_vec();
        p.extend(head(-1, VIA_SSL, RECEIVED, true, 0xdead_beef));
        p.extend(b"HTTP/1.1 200 OK\n");
        assert_eq!(decode(HEAD, &p), Record::Head { at, fd: -1, via: VIA_SSL, dir: RECEIVED, truncated: true, conn: 0xdead_beef, text: "HTTP/1.1 200 OK\n".into() });

        assert_eq!(decode(HEAD, &p[..10]), Record::Other { kind: HEAD });
        assert_eq!(decode(77, &p), Record::Other { kind: 77 });
    }
}
