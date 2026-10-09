//! `aeiou datagen`: write the corpus an abstract declares, from the same names, sizes, and
//! dataset seeds the runner computes, with the payload of `payload.rs`, and a manifest at
//! each dataset root (`schema/README.md` §6). Files are written in parallel by id with no
//! per-file structure; the manifest is written last, atomically.
//!
//! **Several hosts** (`--ranks R --rank r --coordinator`, 2026-10-04, `DESIGN_REVIEW.md`
//! §3.62). Rank `r` writes the files of every `files` dataset whose ids fall in its slice of
//! `[0, files)` (the contiguous split of `run::gpu_range`, so a host's files sit in its own
//! directories), and each `regions` dataset is written whole by one rank (the i-th regions
//! dataset by rank `i mod R`: several hosts writing one NFS file serialize on the server, so
//! the split buys nothing). Every rank checks that the roots are empty before the start gate,
//! writes after it, `syncfs`es, and sends its counts; rank 0 writes the manifests once every
//! rank has reported, so a manifest means the whole corpus is there. A single host is the
//! same path with no coordinator.
//!
//! **O_DIRECT, always.** Datagen never reads what it writes, the client's page cache is what
//! a benchmark must not have warm, and a page cache near full makes every eviction dearer, so
//! files are written with `O_DIRECT` in 1 MiB aligned blocks; a tail that is not a multiple
//! of 4 KiB is written as is where the filesystem allows (NFS does) and otherwise padded and
//! truncated to length. A root that refuses `O_DIRECT` (tmpfs before Linux 6.6) falls back
//! to the page cache, said once. Each direct write completes before the next, so a crash
//! leaves a strict prefix of every file: a file of the right size is a whole file, which is
//! what a later `--resume` will rely on.
//!
//! **Threads.** A rank's ids are handed to its threads in runs aligned to the pattern's
//! directories (`{id div N}`: a run is a directory, shrunk so that every thread has several),
//! because Linux takes the parent directory's lock exclusively for every create, and on NFS
//! the `OPEN` round trip runs under it: threads creating in one directory serialize. A regions
//! file is cut into 64 MiB pieces for the owner's threads (`pwrite` at disjoint offsets).

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::coord::{Server, Tcp};
use crate::eval::{Config, DirScheme, DsMeta, Model, Params};
use crate::payload::{self, Filler, Manifest, PayloadSpec, BLOCK};

/// The alignment `O_DIRECT` wants of a buffer and a length (the largest common logical block).
pub const ALIGN: usize = 4096;
/// A regions file is written in pieces of this many bytes, one per thread at a time.
const REGIONS_PIECE: u64 = 64 << 20;

#[derive(Debug, Clone)]
pub struct DatagenOpts {
    pub root: PathBuf,
    /// The datasets placed elsewhere (`endpoint.rs`).
    pub endpoints: crate::endpoint::Endpoints,
    pub threads: usize,
    pub dedupe: u64,
    pub compress: u64,
    /// Only these datasets (all when empty).
    pub datasets: Vec<String>,
    /// This host among `ranks`.
    pub rank: i64,
    pub ranks: i64,
}

/// The coordinator of a datagen on several hosts: this rank's client, and rank 0's server.
pub struct Hosts<'a> {
    pub tcp: &'a Tcp,
    pub server: Option<&'a Server>,
}

pub struct DatasetResult {
    pub name: String,
    pub root: PathBuf,
    /// Written by this rank.
    pub files: u64,
    pub bytes: u64,
    /// Written by every rank (this rank's alone on one host).
    pub total_files: u64,
    pub total_bytes: u64,
    pub id: String,
    pub elapsed: Duration,
    /// This rank wrote the manifest (rank 0; the only rank on one host).
    pub wrote_manifest: bool,
    /// A `regions` dataset another rank owns.
    pub owner: Option<i64>,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// This rank's slice of a `files` dataset's ids: the contiguous split of `run::gpu_range`.
pub fn file_slice(files: i64, ranks: i64, rank: i64) -> (i64, i64) {
    crate::run::gpu_range(files, ranks, rank, 0)
}

/// The rank that writes the `index`-th `regions` dataset (in the abstract's order, among the
/// datasets selected).
pub fn regions_owner(index: usize, ranks: i64) -> i64 {
    (index as i64).rem_euclid(ranks.max(1))
}

/// Ids are handed to threads in runs: a directory each for `{id div N}`, shrunk so that a
/// slice of `len` ids gives every thread several runs; one id otherwise.
pub fn run_size(dirs: &DirScheme, len: i64, threads: usize) -> i64 {
    let per_dir = match dirs {
        DirScheme::Div(n) => (*n).max(1),
        _ => 1,
    };
    let target = (len / (4 * threads.max(1) as i64)).max(1);
    per_dir.min(target).max(1)
}

// ---------------------------------------------------------------- the writer

/// A 4 KiB-aligned buffer of `BLOCK` bytes, for `O_DIRECT`.
struct Aligned {
    ptr: *mut u8,
    layout: Layout,
}

impl Aligned {
    fn new() -> Self {
        let layout = Layout::from_size_align(BLOCK as usize, ALIGN).expect("layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "allocating a {BLOCK}-byte aligned buffer");
        Aligned { ptr, layout }
    }
}

impl Deref for Aligned {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }
}

impl DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

unsafe impl Send for Aligned {}

/// What this root turned out to support, learned once and shared by every thread.
pub struct Mode {
    /// `O_DIRECT` opens succeed here.
    direct: AtomicBool,
    /// An unaligned tail under `O_DIRECT` is refused here: pad it and truncate.
    pad_tails: AtomicBool,
    notes: Mutex<Vec<String>>,
}

impl Mode {
    pub fn new() -> Self {
        Mode { direct: AtomicBool::new(true), pad_tails: AtomicBool::new(false), notes: Mutex::new(Vec::new()) }
    }
    pub fn direct(&self) -> bool {
        self.direct.load(Ordering::Relaxed)
    }
    fn note(&self, s: String) {
        let mut n = self.notes.lock().unwrap();
        if !n.contains(&s) {
            n.push(s);
        }
    }
    fn take_notes(&self) -> Vec<String> {
        std::mem::take(&mut *self.notes.lock().unwrap())
    }
}

impl Default for Mode {
    fn default() -> Self {
        Self::new()
    }
}

fn is_einval(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::EINVAL)
}

/// Create (truncating) `full` for writing, with `O_DIRECT` unless this root has refused it;
/// says whether this file's descriptor is direct.
fn create(full: &Path, mode: &Mode) -> Result<(File, bool)> {
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if mode.direct() {
        match OpenOptions::new().write(true).create(true).truncate(true).custom_flags(libc::O_DIRECT).open(full) {
            Ok(f) => return Ok((f, true)),
            Err(e) if is_einval(&e) => {
                mode.direct.store(false, Ordering::Relaxed);
                mode.note(format!("O_DIRECT is not supported under {}: writing through the page cache", full.parent().map(|p| p.display().to_string()).unwrap_or_default()));
            }
            Err(e) => return Err(e).with_context(|| format!("creating {}", full.display())),
        }
    }
    Ok((OpenOptions::new().write(true).create(true).truncate(true).open(full).with_context(|| format!("creating {}", full.display()))?, false))
}

/// Write `len` bytes of content at file offset `at`, the content being logical offset
/// `logical..` of the stream `seed_of` describes, in `BLOCK` pieces. Under `O_DIRECT` an
/// unaligned tail is tried as is, then padded to `ALIGN` and the file truncated to `at + len`.
fn write_range(f: &File, direct: bool, mode: &Mode, seed_of: &dyn Fn(u64) -> u64, logical: u64, at: u64, len: u64, filler: &mut Filler, buf: &mut Aligned, what: &Path) -> Result<()> {
    let mut done = 0u64;
    while done < len {
        let n = (len - done).min(BLOCK) as usize;
        filler.fill_range(seed_of, logical + done, &mut buf[..n]);
        let off = at + done;
        if direct && n % ALIGN != 0 {
            if !mode.pad_tails.load(Ordering::Relaxed) {
                match f.write_all_at(&buf[..n], off) {
                    Ok(()) => {
                        done += n as u64;
                        continue;
                    }
                    Err(e) if is_einval(&e) => {
                        mode.pad_tails.store(true, Ordering::Relaxed);
                        mode.note("unaligned tails are refused under O_DIRECT here: padded and truncated".into());
                    }
                    Err(e) => return Err(e).with_context(|| format!("writing {}", what.display())),
                }
            }
            let padded = (n + ALIGN - 1) / ALIGN * ALIGN;
            buf[n..padded].fill(0);
            f.write_all_at(&buf[..padded], off).with_context(|| format!("writing {}", what.display()))?;
            f.set_len(off + n as u64).with_context(|| format!("truncating {} to {}", what.display(), off + n as u64))?;
        } else {
            f.write_all_at(&buf[..n], off).with_context(|| format!("writing {}", what.display()))?;
        }
        done += n as u64;
    }
    Ok(())
}

/// Flush the filesystem holding `dir` (what `O_DIRECT` left to the page cache, the metadata).
fn syncfs(dir: &Path) -> Result<()> {
    let d = File::open(dir).with_context(|| format!("opening {}", dir.display()))?;
    if unsafe { libc::syncfs(d.as_raw_fd()) } != 0 {
        let e = std::io::Error::last_os_error();
        // not every filesystem has it; `sync` is the fallback
        if e.raw_os_error() != Some(libc::ENOSYS) {
            return Err(e).with_context(|| format!("syncfs {}", dir.display()));
        }
        unsafe { libc::sync() };
    }
    Ok(())
}

// ---------------------------------------------------------------- the plan

/// What this rank writes of one dataset.
enum Share {
    /// Files `[lo, hi)` of a `files` dataset.
    Files { lo: i64, hi: i64 },
    /// The whole `regions` file.
    Regions,
    /// Nothing: a `regions` dataset another rank owns.
    Other { owner: i64 },
}

struct Plan<'m, 'a> {
    meta: &'m DsMeta<'a>,
    name: &'a str,
    root: PathBuf,
    share: Share,
}

struct Written {
    files: u64,
    bytes: u64,
    started: u64,
    finished: u64,
    elapsed: Duration,
}

/// Write this rank's share of one dataset.
fn write_share(plan: &Plan<'_, '_>, opts: &DatagenOpts, spec: &PayloadSpec, mode: &Mode, log: &mut (impl Write + Send)) -> Result<Written> {
    let name = plan.name;
    let meta = plan.meta;
    let started = Instant::now();
    let start_secs = now_secs();
    let threads = opts.threads.max(1);
    let seed = meta.dataset_seed();
    let dedupe = spec.dedupe;
    let files = AtomicU64::new(0);
    let bytes = AtomicU64::new(0);
    let failed = AtomicBool::new(false);
    let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
    let log_mutex: Mutex<&mut (dyn Write + Send)> = Mutex::new(log);
    let last_log = Mutex::new(Instant::now());
    let progress = |done: u64, total: u64, unit: &str| {
        let mut ll = last_log.lock().unwrap();
        if ll.elapsed().as_secs() >= 5 {
            *ll = Instant::now();
            let mut l = log_mutex.lock().unwrap();
            let _ = writeln!(l, "  {name}: {done}/{total} {unit}, {} written", crate::dryrun::human_bytes(bytes.load(Ordering::Relaxed)));
        }
    };
    match (&plan.share, meta) {
        (Share::Other { .. }, _) => {}
        (Share::Files { lo, hi }, DsMeta::Files { spf, chunk, dirs, count, .. }) => {
            let (lo, hi) = (*lo, *hi);
            let len = hi - lo;
            if len > 0 {
                let run = run_size(dirs, len, threads);
                let (first, last) = (lo / run, (hi - 1) / run);
                let next = AtomicI64::new(first);
                let done_ids = AtomicU64::new(0);
                let nthreads = threads.min((last - first + 1).max(1) as usize);
                std::thread::scope(|s| {
                    for _ in 0..nthreads {
                        s.spawn(|| {
                            let mut filler = Filler::new(spec.compress);
                            let mut buf = Aligned::new();
                            loop {
                                if failed.load(Ordering::Relaxed) {
                                    break;
                                }
                                let j = next.fetch_add(1, Ordering::Relaxed);
                                if j > last {
                                    break;
                                }
                                let (a, b) = ((j * run).max(lo), ((j + 1) * run).min(hi));
                                for file in a..b {
                                    let r: Result<()> = (|| {
                                        if file * spf >= *count {
                                            return Ok(());
                                        }
                                        let size = meta.file_size(file)? as u64;
                                        let seed_of = |blk: u64| payload::file_block_seed(seed, file as u64, blk, dedupe);
                                        let objects: Vec<(String, u64, u64)> = match chunk {
                                            None => vec![(meta.file_path(file, None)?.to_string(), 0, size)],
                                            Some(c) => {
                                                let c = *c as u64;
                                                (0..meta.chunks(file)?).map(|k| Ok((meta.file_path(file, Some(k))?.to_string(), k as u64 * c, meta.chunk_size(file, k)? as u64))).collect::<Result<_>>()?
                                            }
                                        };
                                        for (path, logical, len) in objects {
                                            let full = opts.endpoints.path(&opts.root, &path);
                                            let (f, direct) = create(&full, mode)?;
                                            write_range(&f, direct, mode, &seed_of, logical, 0, len, &mut filler, &mut buf, &full)?;
                                            files.fetch_add(1, Ordering::Relaxed);
                                            bytes.fetch_add(len, Ordering::Relaxed);
                                        }
                                        Ok(())
                                    })();
                                    if let Err(e) = r {
                                        errors.lock().unwrap().push(e);
                                        failed.store(true, Ordering::Relaxed);
                                        break;
                                    }
                                    let d = done_ids.fetch_add(1, Ordering::Relaxed) + 1;
                                    if d % 256 == 0 {
                                        progress(d, len as u64, "ids");
                                    }
                                }
                            }
                        });
                    }
                });
            }
        }
        (Share::Regions, DsMeta::Regions { .. }) => {
            let path = meta.file_path(0, None)?.to_string();
            let full = opts.endpoints.path(&opts.root, &path);
            let size = meta.file_size(0)? as u64;
            let (f, direct) = create(&full, mode)?;
            let pieces = (size + REGIONS_PIECE - 1) / REGIONS_PIECE;
            let next = AtomicU64::new(0);
            let nthreads = threads.min(pieces.max(1) as usize);
            let seed_of = |blk: u64| payload::regions_block_seed(seed, blk, dedupe);
            std::thread::scope(|s| {
                for _ in 0..nthreads {
                    s.spawn(|| {
                        let mut filler = Filler::new(spec.compress);
                        let mut buf = Aligned::new();
                        loop {
                            if failed.load(Ordering::Relaxed) {
                                break;
                            }
                            let p = next.fetch_add(1, Ordering::Relaxed);
                            if p >= pieces {
                                break;
                            }
                            let at = p * REGIONS_PIECE;
                            let len = (size - at).min(REGIONS_PIECE);
                            if let Err(e) = write_range(&f, direct, mode, &seed_of, at, at, len, &mut filler, &mut buf, &full) {
                                errors.lock().unwrap().push(e);
                                failed.store(true, Ordering::Relaxed);
                                break;
                            }
                            bytes.fetch_add(len, Ordering::Relaxed);
                            if p % 16 == 15 {
                                progress(p + 1, pieces, "pieces");
                            }
                        }
                    });
                }
            });
            if !failed.load(Ordering::Relaxed) {
                files.fetch_add(1, Ordering::Relaxed);
            }
        }
        _ => bail!("dataset `{name}`: the plan does not match the dataset kind"),
    }
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e.context(format!("dataset `{name}`")));
    }
    Ok(Written { files: files.load(Ordering::Relaxed), bytes: bytes.load(Ordering::Relaxed), started: start_secs, finished: now_secs(), elapsed: started.elapsed() })
}

/// One rank's part of one dataset, as sent to rank 0 (`Done`) and recorded in the manifest.
fn part_json(rank: i64, host: &str, w: &Written) -> Value {
    json!({"rank": rank, "host": host, "files_written": w.files, "bytes_written": w.bytes, "started": w.started, "finished": w.finished})
}

pub fn datagen(loaded: &crate::Loaded, cfg: &Config, params: &Params, model: &Model<'_>, opts: &DatagenOpts, hosts: Option<&Hosts<'_>>, log: &mut (impl Write + Send)) -> Result<Vec<DatasetResult>> {
    let spec = PayloadSpec::new(opts.dedupe, opts.compress);
    let ranks = opts.ranks.max(1);
    let rank = opts.rank;
    if rank < 0 || rank >= ranks {
        bail!("rank {rank} of {ranks}");
    }
    if ranks > 1 && hosts.is_none() {
        bail!("{ranks} ranks and no coordinator");
    }
    let params_json = payload::params_json(&loaded.doc, cfg, params)?;
    let host = crate::run::hostname();

    // the plan: what each dataset is, where it goes, this rank's share; every rank checks
    // the roots are empty before the gate, so no rank sees another's files
    let mut plans = Vec::new();
    let mut regions_index = 0usize;
    for meta in model.datasets.iter() {
        let name = meta.name();
        if !opts.datasets.is_empty() && !opts.datasets.iter().any(|d| d == name) {
            continue;
        }
        if let Some(crate::ast::Dataset::Files(f)) = loaded.ast.datasets.get(name) {
            if let Some(fm) = &f.format {
                bail!("dataset `{name}` is a `{}` container: its format class writes it (the Python writer, `aeiou-datagen`; builder/REFERENCE.md), not `aeiou datagen`", fm.class);
            }
        }
        let rel = payload::dataset_root(&loaded.ast, name)?;
        let root = opts.endpoints.path(&opts.root, &rel);
        if root.exists() {
            let n = std::fs::read_dir(&root)?.count();
            if n > 0 {
                bail!("dataset `{name}`: {} is not empty ({n} entries); datasets are read-only, remove it first", root.display());
            }
        }
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        let share = match meta {
            DsMeta::Files { .. } => {
                let (lo, hi) = file_slice(meta.files(), ranks, rank);
                Share::Files { lo, hi }
            }
            DsMeta::Regions { .. } => {
                let owner = regions_owner(regions_index, ranks);
                regions_index += 1;
                if owner == rank {
                    Share::Regions
                } else {
                    Share::Other { owner }
                }
            }
        };
        plans.push(Plan { meta, name, root, share });
    }
    for p in &plans {
        match &p.share {
            Share::Files { lo, hi } => writeln!(log, "dataset {}: files [{lo}, {hi}) of {} on this rank", p.name, p.meta.files())?,
            Share::Regions => writeln!(log, "dataset {}: the regions file, written by this rank", p.name)?,
            Share::Other { owner } => writeln!(log, "dataset {}: the regions file, written by rank {owner}", p.name)?,
        }
    }
    log.flush()?;

    // the gate: no rank writes before every rank has checked
    if let Some(h) = hosts {
        let (_t0, names) = h.tcp.ready()?;
        writeln!(log, "start gate: {} host(s) ready ({})", names.len(), names.join(", "))?;
        log.flush()?;
    }

    // the writes, then the metadata to stable storage
    let mode = Mode::new();
    let mut written = Vec::new();
    for p in &plans {
        let w = write_share(p, opts, &spec, &mode, log)?;
        for n in mode.take_notes() {
            writeln!(log, "  {n}")?;
        }
        written.push(w);
    }
    syncfs(&opts.root)?;
    for p in opts.endpoints.placed() {
        syncfs(&p.dir)?;
    }

    // the manifests: rank 0's, once every rank's part is in; on one host, now
    let mut parts: BTreeMap<i64, Value> = BTreeMap::new();
    let mut wrote_manifest = false;
    let mine = json!({
        "host": host,
        "direct": mode.direct(),
        "datasets": plans.iter().zip(&written).map(|(p, w)| (p.name.to_string(), part_json(rank, &host, w))).collect::<serde_json::Map<_, _>>(),
    });
    match hosts {
        None => {
            parts.insert(0, mine);
            wrote_manifest = true;
        }
        Some(h) => {
            h.tcp.done(&mine)?;
            match h.server {
                Some(server) => {
                    parts = server.gathered().context("gathering every rank's part")?;
                    wrote_manifest = true;
                }
                None => {
                    let (ok, _, error) = h.tcp.result()?;
                    if !ok {
                        bail!("{}", error.unwrap_or_else(|| "rank 0 failed".into()));
                    }
                }
            }
        }
    }
    let mut results = Vec::new();
    let mut manifest_error = None;
    for (p, w) in plans.iter().zip(&written) {
        let (mut total_files, mut total_bytes, mut started, mut finished) = (0u64, 0u64, u64::MAX, 0u64);
        let mut host_parts = Vec::new();
        let mut direct = true;
        for (_, doc) in &parts {
            direct &= doc["direct"].as_bool().unwrap_or(false);
            let d = &doc["datasets"][p.name];
            if d.is_null() {
                continue;
            }
            total_files += d["files_written"].as_u64().unwrap_or(0);
            total_bytes += d["bytes_written"].as_u64().unwrap_or(0);
            started = started.min(d["started"].as_u64().unwrap_or(u64::MAX));
            finished = finished.max(d["finished"].as_u64().unwrap_or(0));
            host_parts.push(d.clone());
        }
        if wrote_manifest && host_parts.len() as i64 != ranks {
            manifest_error = Some(format!("dataset `{}`: {} of {ranks} ranks reported", p.name, host_parts.len()));
            break;
        }
        let manifest = Manifest {
            manifest_version: payload::MANIFEST_VERSION,
            dataset: payload::resolved_dataset(&loaded.doc, p.name, cfg)?,
            payload: spec.clone(),
            format: None,
            provenance: json!({
                "abstract": loaded.ast.name,
                "ast_sha256": loaded.sha256,
                "params": params_json,
                "param_files": cfg.sets.iter().map(|s| json!({"path": s.path, "sha256": s.sha256})).collect::<Vec<_>>(),
                "datagen": format!("aeiou {}", env!("CARGO_PKG_VERSION")),
                "host": parts.get(&0).and_then(|d| d["host"].as_str()).unwrap_or(&host),
                "ranks": ranks,
                "hosts": host_parts,
                "direct": direct,
                "started": if started == u64::MAX { w.started } else { started },
                "finished": finished.max(w.finished),
                "files_written": total_files,
                "bytes_written": total_bytes,
            }),
        };
        if wrote_manifest {
            if let Err(e) = manifest.write(&p.root) {
                manifest_error = Some(format!("dataset `{}`: {e:#}", p.name));
                break;
            }
        }
        results.push(DatasetResult {
            name: p.name.to_string(),
            root: p.root.clone(),
            files: w.files,
            bytes: w.bytes,
            total_files: if wrote_manifest { total_files } else { w.files },
            total_bytes: if wrote_manifest { total_bytes } else { w.bytes },
            id: manifest.id(),
            elapsed: w.elapsed,
            wrote_manifest,
            owner: match p.share {
                Share::Other { owner } => Some(owner),
                _ => None,
            },
        });
    }
    if let Some(h) = hosts {
        if let Some(server) = h.server {
            // the verdict: every rank exits with it
            server.finish(manifest_error.is_none(), 0, manifest_error.clone());
        }
    }
    if let Some(e) = manifest_error {
        bail!("{e}");
    }
    Ok(results)
}
