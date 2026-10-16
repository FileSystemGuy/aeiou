//! Cold start (`DESIGN_REVIEW.md` §3.31): what a host does about the caches before it
//! arrives at the start gate, and what it measures about them. Harness, never abstract
//! vocabulary: cache state is solution, not application.
//!
//! - `--drop-caches`: `sync`, then `3` into `/proc/sys/vm/drop_caches` (page cache,
//!   dentries, inodes; an evicted NFS inode takes its attribute and access caches with it
//!   and returns its delegation). Root only; a failed write refuses the run. Recorded with
//!   its duration and `Cached`, `dentry-state`, `inode-nr` before and after. Once, before
//!   the gate; never inside a run.
//! - The residency check, only with `--drop-caches` or `--require-cold` (it is the proof of
//!   the one and the gate of the other; a plain run does not pay its opens, and on NFS the
//!   report's server-read bytes already tell a warm run): [`SAMPLE_FILES`] files of every dataset,
//!   evenly spaced over the file ids (a formula, no per-file structure, no root), mapped and
//!   asked through `mincore` what fraction of their pages is in the page cache. `mincore`
//!   reads no data, but the open is an open: on NFS it leaves the sampled files' dentries,
//!   attributes, and possibly delegations on the client, a few hundred of the corpus.
//!   `--require-cold` refuses when any sampled page is resident.

use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::dryrun::human_bytes;
use crate::eval::{DsMeta, Model};
use crate::run::RunOpts;

/// Files sampled per dataset by the residency check.
pub const SAMPLE_FILES: i64 = 256;
/// A file larger than this is sampled in `WINDOWS` windows of `WINDOW` bytes, evenly spaced.
const WHOLE: u64 = 256 << 20;
const WINDOW: u64 = 4 << 20;
const WINDOWS: u64 = 64;

/// The kernel's cache sizes: `Cached` and `Shmem` of `/proc/meminfo` (`Cached` counts tmpfs
/// and shared memory, which a drop cannot free; the file cache is the difference), and the first two fields of
/// `/proc/sys/fs/dentry-state` and `/proc/sys/fs/inode-nr` (allocated, unused).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheSizes {
    pub cached_bytes: u64,
    #[serde(default)]
    pub shmem_bytes: u64,
    pub dentries: u64,
    pub dentries_unused: u64,
    pub inodes: u64,
    pub inodes_unused: u64,
}

impl CacheSizes {
    pub fn now() -> CacheSizes {
        let rd = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
        Self::parse(&rd("/proc/meminfo"), &rd("/proc/sys/fs/dentry-state"), &rd("/proc/sys/fs/inode-nr"))
    }

    pub fn parse(meminfo: &str, dentry_state: &str, inode_nr: &str) -> CacheSizes {
        let kb = |key: &str| meminfo.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let (cached_kb, shmem_kb) = (kb("Cached:"), kb("Shmem:"));
        let two = |s: &str| {
            let f: Vec<u64> = s.split_whitespace().map(|x| x.parse().unwrap_or(0)).collect();
            (f.first().copied().unwrap_or(0), f.get(1).copied().unwrap_or(0))
        };
        let (dentries, dentries_unused) = two(dentry_state);
        let (inodes, inodes_unused) = two(inode_nr);
        CacheSizes { cached_bytes: cached_kb << 10, shmem_bytes: shmem_kb << 10, dentries, dentries_unused, inodes, inodes_unused }
    }

    /// `Cached − Shmem`: the page cache of files a drop can free.
    pub fn file_cache_bytes(&self) -> u64 {
        self.cached_bytes.saturating_sub(self.shmem_bytes)
    }
}

/// One `--drop-caches`: how long `sync` and the write took, the cache sizes around them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DropRecord {
    pub sync_ns: u64,
    pub drop_ns: u64,
    pub before: CacheSizes,
    pub after: CacheSizes,
}

/// One dataset's residency sample.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Residency {
    pub dataset: String,
    /// Files (or chunk objects) examined, of `of` in the dataset.
    pub files: u64,
    pub of: u64,
    /// Pages examined and, of those, in the page cache.
    pub pages: u64,
    pub resident: u64,
}

/// One host's cold start: `Report::cold` holds one per host.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ColdStart {
    pub host: String,
    pub dropped: Option<DropRecord>,
    pub residency: Vec<Residency>,
}

/// `sync`, then `3` into `/proc/sys/vm/drop_caches`.
pub fn drop_caches() -> Result<DropRecord> {
    drop_caches_at(Path::new("/proc/sys/vm/drop_caches"))
}

/// The drop against another file, for tests.
pub fn drop_caches_at(path: &Path) -> Result<DropRecord> {
    // open before the sync: without the privilege the refusal should not cost a sync
    let mut f = std::fs::OpenOptions::new().write(true).open(path).with_context(|| {
        format!("--drop-caches: opening {} for writing (root or CAP_SYS_ADMIN is required; run the runner under `sudo -n`)", path.display())
    })?;
    let before = CacheSizes::now();
    let t0 = Instant::now();
    unsafe { libc::sync() };
    let sync_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    std::io::Write::write_all(&mut f, b"3\n").with_context(|| format!("--drop-caches: writing 3 to {}", path.display()))?;
    let drop_ns = t1.elapsed().as_nanos() as u64;
    Ok(DropRecord { sync_ns, drop_ns, before, after: CacheSizes::now() })
}

/// Pages of `[off, off + len)` of `fd` and how many of them are in the page cache.
fn mincore_range(fd: i32, off: u64, len: u64) -> std::io::Result<(u64, u64)> {
    if len == 0 {
        return Ok((0, 0));
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let p = unsafe { libc::mmap(std::ptr::null_mut(), len as usize, libc::PROT_READ, libc::MAP_SHARED, fd, off as libc::off_t) };
    if p == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    let pages = len.div_ceil(page);
    let mut vec = vec![0u8; pages as usize];
    let r = unsafe { libc::mincore(p, len as usize, vec.as_mut_ptr()) };
    let e = std::io::Error::last_os_error();
    unsafe { libc::munmap(p, len as usize) };
    if r != 0 {
        return Err(e);
    }
    Ok((pages, vec.iter().filter(|b| *b & 1 == 1).count() as u64))
}

/// One file: all of it up to `WHOLE`, evenly spaced windows beyond.
pub fn file_residency(path: &Path) -> Result<(u64, u64)> {
    let f = std::fs::File::open(path).with_context(|| format!("residency check: opening {}", path.display()))?;
    let size = f.metadata()?.len();
    let fd = f.as_raw_fd();
    let mut total = (0u64, 0u64);
    let mut add = |off: u64, len: u64| -> Result<()> {
        let (p, r) = mincore_range(fd, off, len).with_context(|| format!("residency check: mincore of {}", path.display()))?;
        total.0 += p;
        total.1 += r;
        Ok(())
    };
    if size <= WHOLE {
        add(0, size)?;
    } else {
        // window starts are multiples of WINDOW, so of the page size
        let slots = size / WINDOW;
        for w in 0..WINDOWS {
            add((w * slots / WINDOWS) * WINDOW, WINDOW)?;
        }
    }
    Ok(total)
}

/// The residency sample of every dataset under `root`.
pub fn residency(model: &Model<'_>, root: &(impl crate::endpoint::Place + ?Sized)) -> Result<Vec<Residency>> {
    let mut out = Vec::new();
    for meta in &model.datasets {
        let files = meta.files().max(0);
        let n = files.min(SAMPLE_FILES);
        let mut r = Residency { dataset: meta.name().to_string(), files: 0, of: files as u64, pages: 0, resident: 0 };
        for k in 0..n {
            // evenly spaced ids: a formula of (k, files), the same on every host
            let file = (k as i128 * files as i128 / n as i128) as i64;
            let chunk = match meta {
                DsMeta::Files { chunk: Some(_), .. } => Some(file % meta.chunks(file)?.max(1)),
                _ => None,
            };
            let rel = meta.file_path(file, chunk)?;
            if root.object(&rel).is_some() {
                // an object store has no page cache of this host's to sample
                break;
            }
            let (pages, resident) = file_residency(&root.at(&rel)).with_context(|| format!("dataset `{}`", meta.name()))?;
            r.files += 1;
            r.pages += pages;
            r.resident += resident;
        }
        out.push(r);
    }
    Ok(out)
}

/// The host's cold start, between the startup checks and the start gate: the drop when
/// asked, the residency sample when either flag is given, and the `--require-cold` refusal.
pub fn start(model: &Model<'_>, opts: &RunOpts) -> Result<ColdStart> {
    let dropped = if opts.drop_caches { Some(drop_caches()?) } else { None };
    let residency = if opts.drop_caches || opts.require_cold { residency(model, opts)? } else { Vec::new() };
    if opts.require_cold {
        if let Some(r) = residency.iter().find(|r| r.resident > 0) {
            bail!(
                "dataset `{}`: {} of {} sampled pages ({} files) are in this host's page cache at the start (--require-cold){}",
                r.dataset,
                r.resident,
                r.pages,
                r.files,
                if opts.drop_caches { "; they survived the drop: mapped, dirty, or on a filesystem that is its page cache (tmpfs)" } else { "; use --drop-caches or remount" }
            );
        }
    }
    Ok(ColdStart { host: crate::run::hostname(), dropped, residency })
}

/// The report's lines for one host.
pub fn write(out: &mut impl std::io::Write, c: &ColdStart) -> std::io::Result<()> {
    match &c.dropped {
        Some(d) => writeln!(
            out,
            "cold start {}: caches dropped (sync {:.3} s, drop {:.3} s, before the start gate): file cache {} -> {} (Cached less Shmem; Shmem {})  dentries {} -> {}  inodes {} -> {}",
            c.host,
            d.sync_ns as f64 / 1e9,
            d.drop_ns as f64 / 1e9,
            human_bytes(d.before.file_cache_bytes()),
            human_bytes(d.after.file_cache_bytes()),
            human_bytes(d.after.shmem_bytes),
            d.before.dentries,
            d.after.dentries,
            d.before.inodes,
            d.after.inodes
        )?,
        None => writeln!(out, "cold start {}: caches not dropped{}", c.host, if c.residency.is_empty() { ", dataset residency not sampled (--require-cold or --drop-caches samples it)" } else { "" })?,
    }
    for r in &c.residency {
        writeln!(
            out,
            "cold start {}: dataset {}: {} of {} files sampled, {} of {} pages resident ({:.1} %)",
            c.host,
            r.dataset,
            r.files,
            r.of,
            r.resident,
            r.pages,
            if r.pages > 0 { 100.0 * r.resident as f64 / r.pages as f64 } else { 0.0 }
        )?;
    }
    if c.residency.iter().any(|r| r.resident > 0) {
        writeln!(out, "WARNING: dataset pages were in {}'s page cache after the drop; reads of them are not reads of the storage (--require-cold refuses)", c.host)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sizes_parse() {
        let c = CacheSizes::parse("MemTotal: 100 kB\nCached:          2048 kB\nSwapCached: 0 kB\nShmem:   512 kB\n", "1203 400 45 0 12 0\n", "5000\t1200\n");
        assert_eq!(c.file_cache_bytes(), 1536 << 10);
        assert_eq!(c, CacheSizes { cached_bytes: 2048 << 10, shmem_bytes: 512 << 10, dentries: 1203, dentries_unused: 400, inodes: 5000, inodes_unused: 1200 });
        assert_eq!(CacheSizes::parse("", "", ""), CacheSizes::default());
        // the live files parse to something on Linux
        assert!(CacheSizes::now().dentries > 0);
    }

    #[test]
    fn the_drop_writes_3_and_refuses_without_the_privilege() {
        let f = std::env::temp_dir().join(format!("aeiou-drop-{}", std::process::id()));
        std::fs::write(&f, "").unwrap();
        let d = drop_caches_at(&f).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "3\n");
        assert!(d.before.dentries > 0 && d.after.dentries > 0);
        std::fs::remove_file(&f).unwrap();
        if unsafe { libc::geteuid() } != 0 {
            let e = format!("{:#}", drop_caches().unwrap_err());
            assert!(e.contains("root or CAP_SYS_ADMIN"), "{e}");
        }
    }

    #[test]
    fn mincore_sees_what_was_just_written() {
        let f = std::env::temp_dir().join(format!("aeiou-mincore-{}", std::process::id()));
        std::fs::write(&f, vec![7u8; 3 * 4096 + 100]).unwrap();
        let (pages, resident) = file_residency(&f).unwrap();
        assert_eq!(pages, 4);
        assert_eq!(resident, 4, "just written, so in the page cache");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(file_residency(&f).unwrap(), (0, 0));
        std::fs::remove_file(&f).unwrap();
    }

    #[test]
    fn a_large_file_is_sampled_in_windows_and_a_file_that_cannot_be_mapped_is_an_error() {
        // sparse: no data, so nothing resident, and the windows are what is examined
        let f = std::env::temp_dir().join(format!("aeiou-mincore-large-{}", std::process::id()));
        let file = std::fs::File::create(&f).unwrap();
        file.set_len(WHOLE + 3 * WINDOW + 4096).unwrap();
        std::os::unix::fs::FileExt::write_all_at(&file, &[1u8; 4096], 0).unwrap();
        drop(file);
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let (pages, resident) = file_residency(&f).unwrap();
        assert_eq!(pages, WINDOWS * WINDOW / page, "{WINDOWS} windows of {WINDOW} bytes");
        assert!(resident >= 1, "the first window holds the page just written");
        std::fs::remove_file(&f).unwrap();
        // a directory opens, and has no pages to map
        let e = format!("{:#}", file_residency(&std::env::temp_dir()).unwrap_err());
        assert!(e.contains("residency check: mincore of"), "{e}");
    }

    #[test]
    fn the_residency_of_a_chunked_dataset_samples_one_chunk_of_each_file() {
        let text = r#"{"ast": "0.10", "name": "t",
            "datasets": {"d": {"layout": "chunked", "access": "indexed", "pattern": "d/{id}/{k}", "count": 3, "chunk": 4096, "size": {"const": 10000}, "seed": 1}},
            "actors": {"gpu": {"body": []}}}"#;
        let loaded = crate::load_str(text).unwrap();
        let cfg = crate::eval::Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![] };
        let params = crate::eval::Params::new(&loaded.ast, &cfg).unwrap();
        let model = crate::eval::build_model(&loaded.ast, &cfg, &params).unwrap();
        let root = std::env::temp_dir().join(format!("aeiou-residency-chunked-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // file f's sampled chunk is f mod its 3 chunks
        for f in 0..3 {
            std::fs::create_dir_all(root.join(format!("d/{f}"))).unwrap();
            std::fs::write(root.join(format!("d/{f}/{f}")), vec![1u8; 4096 * (f as usize + 1)]).unwrap();
        }
        let r = residency(&model, &root).unwrap();
        assert_eq!(r, vec![Residency { dataset: "d".into(), files: 3, of: 3, pages: 6, resident: 6 }]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_report_says_what_the_drop_did_and_warns_of_resident_pages() {
        let sizes = |cached: u64| CacheSizes { cached_bytes: cached << 20, shmem_bytes: 1 << 20, dentries: 10, inodes: 20, ..Default::default() };
        let c = ColdStart {
            host: "h".into(),
            dropped: Some(DropRecord { sync_ns: 1_500_000_000, drop_ns: 250_000_000, before: sizes(9), after: sizes(2) }),
            residency: vec![Residency { dataset: "d".into(), files: 2, of: 5, pages: 8, resident: 2 }, Residency { dataset: "e".into(), ..Default::default() }],
        };
        let mut out = Vec::new();
        write(&mut out, &c).unwrap();
        let s = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert!(lines[0].starts_with("cold start h: caches dropped (sync 1.500 s, drop 0.250 s, before the start gate): file cache "), "{s}");
        assert!(lines[0].contains("dentries 10 -> 10  inodes 20 -> 20"), "{s}");
        assert_eq!(lines[1], "cold start h: dataset d: 2 of 5 files sampled, 2 of 8 pages resident (25.0 %)");
        assert_eq!(lines[2], "cold start h: dataset e: 0 of 0 files sampled, 0 of 0 pages resident (0.0 %)");
        assert!(lines[3].starts_with("WARNING: dataset pages were in h's page cache after the drop"), "{s}");
        // neither flag: one line, and it says how to sample
        let mut out = Vec::new();
        write(&mut out, &ColdStart { host: "h".into(), ..Default::default() }).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "cold start h: caches not dropped, dataset residency not sampled (--require-cold or --drop-caches samples it)\n");
    }
}
