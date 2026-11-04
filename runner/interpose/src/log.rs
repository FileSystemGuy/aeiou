//! An appending log in a mapped file (`DESIGN_REVIEW.md` §3.96, phase 2 of brief item 54).
//!
//! The capture makes the file and sizes it (sparse) before the command starts; each process of
//! the application maps it once, shared, and appends records by reserving their space with an
//! atomic add on the header's next offset, so a record costs no system call (a `write` would be
//! a call of the application's in its trace), and the reservation holds across `fork`, the
//! mapping being shared. A record is a 32-bit length, a 16-bit kind, 16 bits reserved, then its
//! payload, padded to 8 bytes; its length is stored with the high bit set at once, then without
//! it once the payload is in, so a record its process died writing is told from a whole one. A
//! record that does not fit is dropped and counted: the log never wraps, and never blocks.
//!
//! The header, 64 bytes: `AEIOULOG`, the version (u32), 4 bytes reserved, the capacity (the
//! file's size), the next offset, the records dropped, their bytes; every number little-endian
//! (the machine's: the file is read on the host that wrote it).
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const MAGIC: [u8; 8] = *b"AEIOULOG";
pub const VERSION: u32 = 1;
pub const HEADER: u64 = 64;
/// Set in a record's length while its payload is being written.
const PENDING: u32 = 1 << 31;
const CAPACITY_AT: usize = 16;
const NEXT_AT: usize = 24;
const DROPPED_AT: usize = 32;
const DROPPED_BYTES_AT: usize = 40;

/// The smallest log: the header and one page.
pub const MIN_CAPACITY: u64 = 4096 + HEADER;

/// Make the log at `path`, `capacity` bytes, sparse: nothing is written below the header until
/// a record is.
pub fn create(path: &Path, capacity: u64) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut h = [0u8; HEADER as usize];
    h[..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&VERSION.to_ne_bytes());
    h[CAPACITY_AT..CAPACITY_AT + 8].copy_from_slice(&capacity.to_ne_bytes());
    h[NEXT_AT..NEXT_AT + 8].copy_from_slice(&HEADER.to_ne_bytes());
    f.write_all(&h)?;
    f.set_len(capacity)
}

/// The log mapped into this process.
pub struct Log {
    base: *mut u8,
    cap: u64,
}

unsafe impl Send for Log {}
unsafe impl Sync for Log {}

impl Log {
    /// Map the log open on `fd` (shared, read and write); `None` when it is not one of this
    /// version or cannot be mapped. The descriptor may be closed afterward.
    ///
    /// # Safety
    /// `fd` is an open descriptor.
    pub unsafe fn map(fd: libc::c_int) -> Option<Log> {
        let mut st: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut st) != 0 || (st.st_size as u64) < MIN_CAPACITY {
            return None;
        }
        let len = st.st_size as usize;
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
        if p == libc::MAP_FAILED {
            return None;
        }
        let log = Log { base: p as *mut u8, cap: len as u64 };
        let h = std::slice::from_raw_parts(log.base, 16);
        if h[..8] != MAGIC || u32::from_ne_bytes(h[8..12].try_into().unwrap()) != VERSION || log.u64_at(CAPACITY_AT).load(Ordering::Relaxed) != log.cap {
            libc::munmap(p, len);
            return None;
        }
        Some(log)
    }

    fn u64_at(&self, at: usize) -> &AtomicU64 {
        unsafe { &*(self.base.add(at) as *const AtomicU64) }
    }

    /// Append a record of `kind` whose payload is `parts` one after another; false when it did
    /// not fit (counted in the header).
    pub fn append(&self, kind: u16, parts: &[&[u8]]) -> bool {
        let body: usize = parts.iter().map(|p| p.len()).sum();
        let len = (8 + body as u64 + 7) & !7;
        if len >= PENDING as u64 {
            self.u64_at(DROPPED_AT).fetch_add(1, Ordering::Relaxed);
            self.u64_at(DROPPED_BYTES_AT).fetch_add(len, Ordering::Relaxed);
            return false;
        }
        let at = self.u64_at(NEXT_AT).fetch_add(len, Ordering::AcqRel);
        if at + len > self.cap {
            self.u64_at(DROPPED_AT).fetch_add(1, Ordering::Relaxed);
            self.u64_at(DROPPED_BYTES_AT).fetch_add(len, Ordering::Relaxed);
            return false;
        }
        unsafe {
            let p = self.base.add(at as usize);
            let head = &*(p as *const AtomicU32);
            head.store(len as u32 | PENDING, Ordering::Relaxed);
            std::ptr::copy_nonoverlapping(kind.to_ne_bytes().as_ptr(), p.add(4), 2);
            let mut q = p.add(8);
            for part in parts {
                std::ptr::copy_nonoverlapping(part.as_ptr(), q, part.len());
                q = q.add(part.len());
            }
            head.store(len as u32, Ordering::Release);
        }
        true
    }
}

/// What a reading of a log found besides its records.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Reading {
    pub capacity: u64,
    /// The bytes reserved, up to the capacity.
    pub used: u64,
    pub records: u64,
    pub dropped: u64,
    pub dropped_bytes: u64,
    /// Records reserved and never finished: their process ended while writing them.
    pub torn: u64,
    /// Bytes past the last record that could be read (a record cut short, or reserved space a
    /// process never began, or the tail a dropped record left).
    pub unreadable_bytes: u64,
}

/// Read the log in `buf` (the whole file), calling `f` with each whole record's kind and
/// payload (its padding included), in the order their space was reserved.
pub fn read(buf: &[u8], mut f: impl FnMut(u16, &[u8])) -> Result<Reading, String> {
    if (buf.len() as u64) < HEADER || buf[..8] != MAGIC {
        return Err("not an aeiou log (no AEIOULOG header)".into());
    }
    let n = |at: usize| u64::from_ne_bytes(buf[at..at + 8].try_into().unwrap());
    let version = u32::from_ne_bytes(buf[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(format!("an aeiou log of version {version}; this reads {VERSION}"));
    }
    let mut r = Reading { capacity: n(CAPACITY_AT), dropped: n(DROPPED_AT), dropped_bytes: n(DROPPED_BYTES_AT), ..Default::default() };
    let limit = n(NEXT_AT).min(r.capacity).min(buf.len() as u64);
    r.used = limit;
    let mut at = HEADER;
    while at + 8 <= limit {
        let i = at as usize;
        let raw = u32::from_ne_bytes(buf[i..i + 4].try_into().unwrap());
        let len = (raw & !PENDING) as u64;
        if len < 8 || len % 8 != 0 || at + len > limit {
            break;
        }
        if raw & PENDING != 0 {
            r.torn += 1;
        } else {
            r.records += 1;
            f(u16::from_ne_bytes(buf[i + 4..i + 6].try_into().unwrap()), &buf[i + 8..i + len as usize]);
        }
        at += len;
    }
    r.unreadable_bytes = limit - at;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("aeiou-interpose-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn records(path: &Path) -> (Vec<(u16, Vec<u8>)>, Reading) {
        let mut out = Vec::new();
        let r = read(&std::fs::read(path).unwrap(), |k, p| out.push((k, p.to_vec()))).unwrap();
        (out, r)
    }

    #[test]
    fn records_are_appended_padded_and_read_back_in_order() {
        let p = tmp("order");
        create(&p, MIN_CAPACITY).unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().len(), MIN_CAPACITY);
        assert!(create(&p, MIN_CAPACITY).is_err(), "never over an existing file");
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&p).unwrap();
        let log = unsafe { Log::map(f.as_raw_fd()) }.unwrap();
        drop(f);
        assert!(log.append(3, &[b"abc", b"de"]));
        assert!(log.append(1, &[]));
        let (recs, r) = records(&p);
        assert_eq!(recs, vec![(3, b"abcde\0\0\0".to_vec()), (1, vec![])]);
        assert_eq!(r, Reading { capacity: MIN_CAPACITY, used: HEADER + 16 + 8, records: 2, ..Default::default() });
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn a_record_that_does_not_fit_is_dropped_and_counted_and_the_rest_still_read() {
        let p = tmp("full");
        create(&p, MIN_CAPACITY).unwrap();
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&p).unwrap();
        let log = unsafe { Log::map(f.as_raw_fd()) }.unwrap();
        let big = vec![7u8; 3000];
        assert!(log.append(2, &[&big]));
        assert!(!log.append(2, &[&big]));
        assert!(!log.append(2, &[&vec![0u8; PENDING as usize]]), "a record longer than a length can say");
        let (recs, r) = records(&p);
        assert_eq!(recs.len(), 1);
        assert_eq!((r.dropped, r.used, r.records), (2, MIN_CAPACITY, 1));
        assert_eq!(r.dropped_bytes, 3008 + ((8 + PENDING as u64 + 7) & !7));
        // the space the dropped record reserved below the capacity is unreadable, never a record
        assert_eq!(r.unreadable_bytes, MIN_CAPACITY - HEADER - 3008);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn a_record_left_pending_is_torn_and_skipped() {
        let mut buf = vec![0u8; (HEADER + 32) as usize];
        buf[..8].copy_from_slice(&MAGIC);
        buf[8..12].copy_from_slice(&VERSION.to_ne_bytes());
        buf[16..24].copy_from_slice(&(HEADER + 32).to_ne_bytes());
        buf[24..32].copy_from_slice(&(HEADER + 32).to_ne_bytes());
        buf[64..68].copy_from_slice(&(16u32 | PENDING).to_ne_bytes());
        buf[80..84].copy_from_slice(&16u32.to_ne_bytes());
        buf[84..86].copy_from_slice(&9u16.to_ne_bytes());
        let mut kinds = Vec::new();
        let r = read(&buf, |k, _| kinds.push(k)).unwrap();
        assert_eq!((kinds, r.torn, r.records, r.unreadable_bytes), (vec![9], 1, 1, 0));
        // not a log, another version, a mapping refused
        assert!(read(b"nope", |_, _| ()).unwrap_err().contains("not an aeiou log"));
        buf[8] = 2;
        assert!(read(&buf, |_, _| ()).unwrap_err().contains("version 2"));
        let p = tmp("small");
        std::fs::write(&p, &buf).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        assert!(unsafe { Log::map(f.as_raw_fd()) }.is_none());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn processes_sharing_the_mapping_append_without_overlap() {
        let p = tmp("fork");
        create(&p, 1 << 20).unwrap();
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&p).unwrap();
        let log = unsafe { Log::map(f.as_raw_fd()) }.unwrap();
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            for i in 0..1000u32 {
                log.append(1, &[&i.to_ne_bytes()]);
            }
            unsafe { libc::_exit(0) };
        }
        for i in 0..1000u32 {
            log.append(2, &[&i.to_ne_bytes()]);
        }
        let mut st = 0;
        unsafe { libc::waitpid(pid, &mut st, 0) };
        let (recs, r) = records(&p);
        assert_eq!(r.records, 2000);
        for k in [1u16, 2] {
            let mine: Vec<u32> = recs.iter().filter(|(kk, _)| *kk == k).map(|(_, b)| u32::from_ne_bytes(b[..4].try_into().unwrap())).collect();
            assert_eq!(mine, (0..1000).collect::<Vec<_>>(), "kind {k}");
        }
        std::fs::remove_file(&p).unwrap();
    }
}
