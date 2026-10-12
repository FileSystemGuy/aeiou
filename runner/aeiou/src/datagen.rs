//! `aeiou datagen`: write the corpus an abstract declares, from the same names, sizes, and
//! dataset seeds the runner computes, with the payload of `payload.rs`, and a manifest at
//! each dataset root (`schema/README.md` §6). Files are written in parallel by id with no
//! per-file structure. The manifest is written first, `writing`, and rewritten every
//! `--progress-secs` with the ids not yet written and flushed, and `complete` at the end
//! (`DESIGN_REVIEW.md` §3.70), each time atomically.
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
//! **Object datasets** (`protocol: s3`, `object.rs`): each file is one object, sent as
//! one `PUT` or, when larger than `object::DEFAULT_PART`, a multipart upload in parts of
//! that size, each part made and sent before the next; the root's emptiness is a `LIST`, the
//! manifest one `PUT` (atomic, so no temporary name), and there is no `syncfs`. A `regions`
//! dataset, a sparse layout written at offsets, has no object form and is refused.
//!
//! **Writers.** A rank's ids are handed to its writers (`--writers`, a thread each) in runs
//! aligned to the pattern's directories (`{id div N}`: a run is a directory, shrunk so that
//! every writer has several), because Linux takes the parent directory's lock exclusively for
//! every create, and on NFS the `OPEN` round trip runs under it: threads creating in one
//! directory serialize. A regions file is cut into 64 MiB pieces for the owner's writers (`pwrite` at disjoint offsets).

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
/// A regions file is written in pieces of this many bytes, one per writer at a time.
const REGIONS_PIECE: u64 = 64 << 20;

#[derive(Debug, Clone)]
pub struct DatagenOpts {
    pub root: PathBuf,
    /// The datasets placed elsewhere (`endpoint.rs`).
    pub endpoints: crate::endpoint::Endpoints,
    pub writers: usize,
    pub dedupe: u64,
    pub compress: u64,
    /// Only these datasets (all when empty).
    pub datasets: Vec<String>,
    /// This host among `ranks`.
    pub rank: i64,
    pub ranks: i64,
    /// How often each rank flushes and reports what it has finished, and rank 0 rewrites the
    /// manifests (`--progress-secs`).
    pub progress: Duration,
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
    /// A container's class and writer, `class, writer`.
    pub format: Option<String>,
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

/// Ids are handed to writers in runs: a directory each for `{id div N}`, shrunk so that a
/// slice of `len` ids gives every writer several runs; one id otherwise.
pub fn run_size(dirs: &DirScheme, len: i64, writers: usize) -> i64 {
    let per_dir = match dirs {
        DirScheme::Div(n) => (*n).max(1),
        _ => 1,
    };
    let target = (len / (4 * writers.max(1) as i64)).max(1);
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
        write_at(f, direct, mode, buf, n, at + done, what)?;
        done += n as u64;
    }
    Ok(())
}

/// Write the first `n` bytes of `buf` at `off`; only the file's last write may be unaligned.
fn write_at(f: &File, direct: bool, mode: &Mode, buf: &mut Aligned, n: usize, off: u64, what: &Path) -> Result<()> {
    if direct && n % ALIGN != 0 {
        if !mode.pad_tails.load(Ordering::Relaxed) {
            match f.write_all_at(&buf[..n], off) {
                Ok(()) => return Ok(()),
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
    Ok(())
}

// ---------------------------------------------------------------- containers

/// A container dataset (one with a format class): its writer, the manifest's `format`, the
/// resolved `format` entry (what `aeiou-header` takes), and what the provenance records of
/// the headers.
struct Container {
    #[cfg(feature = "formats")]
    writer: crate::formats::Writer,
    format: Value,
    #[cfg_attr(not(feature = "formats"), allow(dead_code))]
    spec: Value,
    headers: Value,
}

/// The writer of a format-class dataset, from its resolved definition.
fn container(name: &str, resolved: &Value) -> Result<Container> {
    let fmt = &resolved["files"]["format"];
    if !resolved["files"]["chunk"].is_null() {
        bail!("dataset `{name}`: a container dataset with `chunk` has no writer");
    }
    #[cfg(feature = "formats")]
    {
        let writer = crate::formats::Writer::from_format(name, fmt)?;
        let format = json!({"class": fmt["class"], "reader": fmt["reader"], "version": fmt["version"], "writer": writer.library()});
        Ok(Container { writer, format, spec: fmt.clone(), headers: Value::Null })
    }
    #[cfg(not(feature = "formats"))]
    bail!(
        "dataset `{name}` is a `{}` container: this build of aeiou has no container writers (the cargo feature `formats` adds TFRecord, WebDataset, and HDF5); the Python writer, `aeiou-datagen`, writes every format class (builder/REFERENCE.md §6)",
        fmt["class"].as_str().unwrap_or("?")
    )
}

/// Rank 0, before the gate: every header a container needs, from `aeiou-header`, set on its
/// writer and returned for `Start` to carry to every rank (`{dataset: {rows: hex}}`; null
/// when no dataset needs one).
#[cfg_attr(not(feature = "formats"), allow(unused_variables))]
fn make_headers(plans: &mut [Plan<'_, '_>], log: &mut impl Write) -> Result<Value> {
    #[cfg(feature = "formats")]
    {
        let mut share = serde_json::Map::new();
        let mut tool = None;
        for p in plans.iter_mut() {
            let Some(c) = p.container.as_mut() else { continue };
            let DsMeta::Files { count, spf, .. } = p.meta else { continue };
            let rows = c.writer.header_rows(*count, *spf);
            if rows.is_empty() {
                continue;
            }
            if tool.is_none() {
                tool = Some(crate::formats::header_tool()?);
            }
            let tool = tool.as_ref().unwrap();
            let got = crate::formats::run_header_tool(tool, &c.spec, &rows).with_context(|| format!("dataset `{}`", p.name))?;
            writeln!(log, "dataset {}: {} header(s) from {}", p.name, got.len(), tool.display())?;
            share.insert(p.name.to_string(), Value::Object(got.iter().map(|(n, h)| (n.to_string(), json!(crate::formats::hex(h)))).collect()));
            set_headers(c, got);
        }
        Ok(if share.is_empty() { Value::Null } else { Value::Object(share) })
    }
    #[cfg(not(feature = "formats"))]
    Ok(Value::Null)
}

/// Every other rank, after the gate: the headers rank 0 sent with `Start`.
#[cfg_attr(not(feature = "formats"), allow(unused_variables))]
fn take_headers(plans: &mut [Plan<'_, '_>], shared: &Value) -> Result<()> {
    #[cfg(feature = "formats")]
    for p in plans.iter_mut() {
        let Some(c) = p.container.as_mut() else { continue };
        let DsMeta::Files { count, spf, .. } = p.meta else { continue };
        let rows = c.writer.header_rows(*count, *spf);
        if rows.is_empty() {
            continue;
        }
        let mut got = BTreeMap::new();
        for n in rows {
            let hex = shared[p.name][n.to_string()].as_str().ok_or_else(|| anyhow::anyhow!("dataset `{}`: rank 0 sent no header for {n} rows", p.name))?;
            got.insert(n, crate::formats::unhex(hex)?);
        }
        set_headers(c, got);
    }
    Ok(())
}

#[cfg(feature = "formats")]
fn set_headers(c: &mut Container, got: BTreeMap<i64, Vec<u8>>) {
    use sha2::Digest;
    c.headers = json!({"tool": "aeiou-header", "sha256": got.iter().map(|(n, h)| (n.to_string(), json!(format!("{:x}", sha2::Sha256::digest(h))))).collect::<serde_json::Map<_, _>>()});
    c.writer.set_headers(got);
}

/// Write container file `file` to `path` (relative to its place), checked against the size
/// its layout predicts; returns the bytes written.
#[cfg_attr(not(feature = "formats"), allow(unused_variables))]
fn write_container(c: &Container, meta: &DsMeta<'_>, file: i64, seed_of: &dyn Fn(u64) -> u64, filler: &mut Filler, buf: &mut Aligned, path: &str, opts: &DatagenOpts, mode: &Mode) -> Result<u64> {
    #[cfg(feature = "formats")]
    {
        let size = meta.file_size(file)? as u64;
        let mut stream = crate::formats::Stream::new(&c.writer, meta, file, seed_of, filler)?;
        let drift = |at: u64, what: &str| anyhow::anyhow!("{what}: the writer made {at} bytes, the layout predicts {size}: the format class's layout does not match its writer");
        if let Some((store, rest)) = opts.endpoints.object(path) {
            let mut failed: Option<anyhow::Error> = None;
            store.put_parts(&rest, size, crate::object::DEFAULT_PART, &mut |_, b| {
                if failed.is_some() {
                    return;
                }
                match stream.read(b) {
                    Ok(n) if n == b.len() => {}
                    Ok(_) => failed = Some(drift(stream.pos(), path)),
                    Err(e) => failed = Some(e),
                }
            })?;
            if let Some(e) = failed {
                return Err(e);
            }
            if stream.read(&mut buf[..1])? != 0 {
                return Err(drift(stream.pos(), path));
            }
            return Ok(size);
        }
        let full = opts.endpoints.path(&opts.root, path);
        let (f, direct) = create(&full, mode)?;
        loop {
            let n = stream.read(&mut buf[..BLOCK as usize])?;
            if n == 0 {
                break;
            }
            let at = stream.pos() - n as u64;
            if at + n as u64 > size {
                return Err(drift(stream.pos(), &full.display().to_string()));
            }
            write_at(&f, direct, mode, buf, n, at, &full)?;
        }
        if stream.pos() != size {
            return Err(drift(stream.pos(), &full.display().to_string()));
        }
        Ok(size)
    }
    #[cfg(not(feature = "formats"))]
    bail!("no container writers in this build")
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
    /// The store and root key of an object dataset.
    object: Option<(std::sync::Arc<crate::object::Store>, String)>,
    /// The writer of a format-class dataset.
    container: Option<Container>,
    share: Share,
    /// The rank that writes a `regions` dataset.
    owner: Option<i64>,
    /// Ids per run handed to a writer (`run_size`).
    run: i64,
    track: Track,
}

/// What this rank has written of one dataset, kept by its writers and read every
/// `--progress-secs` (`DESIGN_REVIEW.md` §3.70). Writers claim runs in id order and write a
/// run in id order, so what is not finished is, per writer, the rest of its run, and every
/// run not yet claimed.
#[derive(Default)]
struct Track {
    files: AtomicU64,
    bytes: AtomicU64,
    claims: Mutex<Claims>,
    /// The regions file is written.
    whole: AtomicBool,
}

#[derive(Default)]
struct Claims {
    /// The next run to hand out.
    next: i64,
    /// Each writer's run `[start, end)` and the first id of it not yet written.
    slots: Vec<(i64, i64, std::sync::Arc<AtomicI64>)>,
}

impl Plan<'_, '_> {
    /// The id ranges of this rank's share not yet written, merged, in order (a regions file
    /// is id 0).
    fn unfinished(&self) -> Vec<(i64, i64)> {
        let mut v = Vec::new();
        match self.share {
            Share::Other { .. } => {}
            Share::Regions => {
                if !self.track.whole.load(Ordering::Acquire) {
                    v.push((0, 1));
                }
            }
            Share::Files { lo, hi } => {
                let c = self.track.claims.lock().unwrap();
                for (_, end, cursor) in &c.slots {
                    let at = cursor.load(Ordering::Acquire);
                    if at < *end {
                        v.push((at, *end));
                    }
                }
                let rest = (c.next * self.run).max(lo);
                if rest < hi {
                    v.push((rest, hi));
                }
            }
        }
        merge_ranges(v)
    }
}

/// Sorted, with touching and overlapping ranges joined.
fn merge_ranges(mut v: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    v.retain(|(a, b)| a < b);
    v.sort();
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (a, b) in v {
        match out.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// What rank `r` is to write of a dataset, before it has said anything.
fn share_of(meta: &DsMeta<'_>, owner: Option<i64>, ranks: i64, r: i64) -> Vec<(i64, i64)> {
    match meta {
        DsMeta::Files { .. } => {
            let (lo, hi) = file_slice(meta.files(), ranks, r);
            merge_ranges(vec![(lo, hi)])
        }
        DsMeta::Regions { .. } => {
            if owner == Some(r) {
                vec![(0, 1)]
            } else {
                Vec::new()
            }
        }
    }
}

fn ranges_json(v: &[(i64, i64)]) -> Value {
    Value::Array(v.iter().map(|(a, b)| json!([a, b])).collect())
}

fn ranges_of(v: &Value) -> Vec<(i64, i64)> {
    v.as_array().into_iter().flatten().filter_map(|x| Some((x[0].as_i64()?, x[1].as_i64()?))).collect()
}

struct Written {
    files: u64,
    bytes: u64,
    started: u64,
    finished: u64,
    elapsed: Duration,
}

/// Write this rank's share of one dataset.
fn write_share(plan: &Plan<'_, '_>, opts: &DatagenOpts, spec: &PayloadSpec, mode: &Mode) -> Result<Written> {
    let name = plan.name;
    let meta = plan.meta;
    let track = &plan.track;
    let started = Instant::now();
    let start_secs = now_secs();
    let writers = opts.writers.max(1);
    let seed = meta.dataset_seed();
    let dedupe = spec.dedupe;
    let failed = AtomicBool::new(false);
    let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
    let (files0, bytes0) = (track.files.load(Ordering::Relaxed), track.bytes.load(Ordering::Relaxed));
    match (&plan.share, meta) {
        (Share::Other { .. }, _) => {}
        (Share::Files { lo, hi }, DsMeta::Files { spf, chunk, count, .. }) => {
            let (lo, hi) = (*lo, *hi);
            let len = hi - lo;
            if len > 0 {
                let run = plan.run;
                let last = (hi - 1) / run;
                let nthreads = writers.min((last - lo / run + 1).max(1) as usize);
                track.claims.lock().unwrap().slots = (0..nthreads).map(|_| (0, 0, std::sync::Arc::new(AtomicI64::new(0)))).collect();
                std::thread::scope(|s| {
                    for w in 0..nthreads {
                        let (failed, errors) = (&failed, &errors);
                        s.spawn(move || {
                            let mut filler = Filler::new(spec.compress);
                            let mut buf = Aligned::new();
                            loop {
                                if failed.load(Ordering::Relaxed) {
                                    break;
                                }
                                // claim the next run, and say so, under one lock
                                let (a, b, cursor) = {
                                    let mut c = track.claims.lock().unwrap();
                                    let j = c.next;
                                    if j > last {
                                        break;
                                    }
                                    c.next += 1;
                                    let (a, b) = ((j * run).max(lo), ((j + 1) * run).min(hi));
                                    let cursor = c.slots[w].2.clone();
                                    cursor.store(a, Ordering::Release);
                                    c.slots[w].0 = a;
                                    c.slots[w].1 = b;
                                    (a, b, cursor)
                                };
                                for file in a..b {
                                    let r: Result<()> = (|| {
                                        if file * spf >= *count {
                                            return Ok(());
                                        }
                                        let size = meta.file_size(file)? as u64;
                                        let seed_of = |blk: u64| payload::file_block_seed(seed, file as u64, blk, dedupe);
                                        if let Some(c) = &plan.container {
                                            let size = write_container(c, meta, file, &seed_of, &mut filler, &mut buf, &meta.file_path(file, None)?, opts, mode)?;
                                            track.files.fetch_add(1, Ordering::Relaxed);
                                            track.bytes.fetch_add(size, Ordering::Relaxed);
                                            return Ok(());
                                        }
                                        let objects: Vec<(String, u64, u64)> = match chunk {
                                            None => vec![(meta.file_path(file, None)?.to_string(), 0, size)],
                                            Some(c) => {
                                                let c = *c as u64;
                                                (0..meta.chunks(file)?).map(|k| Ok((meta.file_path(file, Some(k))?.to_string(), k as u64 * c, meta.chunk_size(file, k)? as u64))).collect::<Result<_>>()?
                                            }
                                        };
                                        for (path, logical, len) in objects {
                                            if let Some((store, rest)) = opts.endpoints.object(&path) {
                                                store.put_parts(&rest, len, crate::object::DEFAULT_PART, &mut |at, b| filler.fill_range(&seed_of, logical + at, b))?;
                                                track.files.fetch_add(1, Ordering::Relaxed);
                                                track.bytes.fetch_add(len, Ordering::Relaxed);
                                                continue;
                                            }
                                            let full = opts.endpoints.path(&opts.root, &path);
                                            let (f, direct) = create(&full, mode)?;
                                            write_range(&f, direct, mode, &seed_of, logical, 0, len, &mut filler, &mut buf, &full)?;
                                            track.files.fetch_add(1, Ordering::Relaxed);
                                            track.bytes.fetch_add(len, Ordering::Relaxed);
                                        }
                                        Ok(())
                                    })();
                                    if let Err(e) = r {
                                        errors.lock().unwrap().push(e);
                                        failed.store(true, Ordering::Relaxed);
                                        break;
                                    }
                                    cursor.store(file + 1, Ordering::Release);
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
            let nthreads = writers.min(pieces.max(1) as usize);
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
                            track.bytes.fetch_add(len, Ordering::Relaxed);
                        }
                    });
                }
            });
            if !failed.load(Ordering::Relaxed) {
                track.files.fetch_add(1, Ordering::Relaxed);
                track.whole.store(true, Ordering::Release);
            }
        }
        _ => bail!("dataset `{name}`: the plan does not match the dataset kind"),
    }
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e.context(format!("dataset `{name}`")));
    }
    Ok(Written {
        files: track.files.load(Ordering::Relaxed) - files0,
        bytes: track.bytes.load(Ordering::Relaxed) - bytes0,
        started: start_secs,
        finished: now_secs(),
        elapsed: started.elapsed(),
    })
}

/// One rank's part of one dataset, as sent to rank 0 (`Done`) and recorded in the manifest.
fn part_json(rank: i64, host: &str, w: &Written) -> Value {
    json!({"rank": rank, "host": host, "files_written": w.files, "bytes_written": w.bytes, "started": w.started, "finished": w.finished})
}

/// What the manifests say about the datasets, the same in every write: the normative part
/// and the provenance known at the start.
struct Base {
    params_json: Value,
    host: String,
}

impl Base {
    fn manifest(&self, loaded: &crate::Loaded, cfg: &Config, spec: &PayloadSpec, p: &Plan<'_, '_>, ranks: i64, started: u64) -> Result<Manifest> {
        let mut m = Manifest {
            manifest_version: payload::MANIFEST_VERSION,
            state: payload::STATE_WRITING.into(),
            dataset: payload::resolved_dataset(&loaded.doc, p.name, cfg)?,
            payload: spec.clone(),
            format: p.container.as_ref().map(|c| c.format.clone()),
            provenance: json!({
                "abstract": loaded.ast.name,
                "ast_sha256": loaded.sha256,
                "params": self.params_json,
                "param_files": cfg.sets.iter().map(|s| json!({"path": s.path, "sha256": s.sha256})).collect::<Vec<_>>(),
                "datagen": format!("aeiou {}", env!("CARGO_PKG_VERSION")),
                "host": self.host,
                "ranks": ranks,
                "started": started,
            }),
            progress: Value::Null,
        };
        if let Some(c) = p.container.as_ref().filter(|c| !c.headers.is_null()) {
            m.provenance["headers"] = c.headers.clone();
        }
        Ok(m)
    }
}

fn write_manifest(p: &Plan<'_, '_>, m: &Manifest) -> Result<()> {
    match &p.object {
        Some((store, rest)) => m.write_object(store, rest).map(|_| ()),
        None => m.write(&p.root).map(|_| ()),
    }
    .with_context(|| format!("dataset `{}`: writing its manifest", p.name))
}

/// This rank's progress document: per dataset, what is not yet written and the counts.
fn progress_doc(plans: &[Plan<'_, '_>]) -> Value {
    json!({"datasets": plans.iter().map(|p| (p.name.to_string(), json!({
        "unfinished": ranges_json(&p.unfinished()),
        "files": p.track.files.load(Ordering::Relaxed),
        "bytes": p.track.bytes.load(Ordering::Relaxed),
    }))).collect::<serde_json::Map<_, _>>()})
}

/// Flush what this rank has written: what `O_DIRECT` left to the page cache, and the metadata.
fn flush(opts: &DatagenOpts) -> Result<()> {
    // --root holds nothing when every dataset is an object, and may not exist
    if opts.root.exists() {
        syncfs(&opts.root)?;
    }
    for p in opts.endpoints.placed() {
        if p.store.is_none() {
            syncfs(&p.dir)?;
        }
    }
    Ok(())
}

/// The progress window: note what is finished, flush, and tell rank 0 (`DESIGN_REVIEW.md`
/// §3.70); rank 0 merges every rank's latest note, rewrites each manifest once, and prints
/// one line. A rank not heard from yet counts as having written nothing.
struct Ticker<'p, 'm, 'a> {
    plans: &'p [Plan<'m, 'a>],
    manifests: &'p [Manifest],
    opts: &'p DatagenOpts,
    hosts: Option<&'p Hosts<'p>>,
    t0: Instant,
}

impl Ticker<'_, '_, '_> {
    fn tick(&self, log: &Mutex<&mut (dyn Write + Send)>) -> Result<()> {
        let mine = progress_doc(self.plans);
        flush(self.opts)?;
        let mut docs = BTreeMap::new();
        if let Some(h) = self.hosts {
            h.tcp.progress(&mine)?;
            match h.server {
                Some(server) => docs = server.progress(),
                None => return Ok(()),
            }
        }
        docs.insert(self.opts.rank, mine);
        let ranks = self.opts.ranks.max(1);
        let (mut total, mut left, mut bytes) = (0i64, 0i64, 0u64);
        for (p, base) in self.plans.iter().zip(self.manifests) {
            let mut unfinished = Vec::new();
            let (mut files, mut b) = (0u64, 0u64);
            for r in 0..ranks {
                match docs.get(&r) {
                    Some(d) => {
                        let d = &d["datasets"][p.name];
                        unfinished.extend(ranges_of(&d["unfinished"]));
                        files += d["files"].as_u64().unwrap_or(0);
                        b += d["bytes"].as_u64().unwrap_or(0);
                    }
                    None => unfinished.extend(share_of(p.meta, p.owner, ranks, r)),
                }
            }
            let unfinished = merge_ranges(unfinished);
            total += match p.meta {
                DsMeta::Files { .. } => p.meta.files(),
                DsMeta::Regions { .. } => 1,
            };
            left += unfinished.iter().map(|(a, b)| b - a).sum::<i64>();
            bytes += b;
            let mut m = base.clone();
            m.progress = json!({"unfinished": ranges_json(&unfinished), "files_written": files, "bytes_written": b, "updated": now_secs()});
            write_manifest(p, &m)?;
        }
        let secs = self.t0.elapsed().as_secs_f64();
        let done = total - left;
        let eta = if done > 0 && left > 0 { format!(", about {:.0} s left", secs * left as f64 / done as f64) } else { String::new() };
        let mut l = log.lock().unwrap();
        writeln!(l, "progress {secs:.1} s: {done}/{total} files, {} written, {}/s{eta}", crate::dryrun::human_bytes(bytes), crate::dryrun::human_bytes((bytes as f64 / secs.max(1e-9)) as u64))?;
        l.flush()?;
        Ok(())
    }
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
    let host = crate::run::hostname();
    let base = Base { params_json: payload::params_json(&loaded.doc, cfg, params)?, host: host.clone() };

    // the plan: what each dataset is, where it goes, this rank's share; every rank checks
    // the roots are empty before the gate, so no rank sees another's files (rank 0's
    // manifest, written before the gate, aside)
    let mut plans = Vec::new();
    let mut regions_index = 0usize;
    for meta in model.datasets.iter() {
        let name = meta.name();
        if !opts.datasets.is_empty() && !opts.datasets.iter().any(|d| d == name) {
            continue;
        }
        let container = match loaded.ast.datasets.get(name) {
            Some(crate::ast::Dataset::Files(f)) if f.format.is_some() => Some(container(name, &payload::resolved_dataset(&loaded.doc, name, cfg)?)?),
            _ => None,
        };
        let rel = payload::dataset_root(&loaded.ast, name)?;
        let root = opts.endpoints.path(&opts.root, &rel);
        let object = opts.endpoints.object(&rel);
        if let Some((store, rest)) = &object {
            if matches!(meta, DsMeta::Regions { .. }) {
                bail!("dataset `{name}`: a `regions` dataset is a sparse layout written at offsets and has no object form; place it in a directory");
            }
            let busy = if rank == 0 { store.non_empty(rest)? } else { store.non_empty_except(rest, payload::MANIFEST_NAME)? };
            if busy {
                bail!("dataset `{name}`: {} is not empty; datasets are read-only, remove it first", store.show(rest));
            }
        } else if root.exists() {
            let n = std::fs::read_dir(&root)?
                .filter(|e| rank == 0 || !e.as_ref().is_ok_and(|e| e.file_name().to_string_lossy().starts_with(payload::MANIFEST_NAME)))
                .count();
            if n > 0 {
                bail!("dataset `{name}`: {} is not empty ({n} entries); datasets are read-only, remove it first", root.display());
            }
        }
        if object.is_none() {
            std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        }
        let (share, owner, run) = match meta {
            DsMeta::Files { dirs, .. } => {
                let (lo, hi) = file_slice(meta.files(), ranks, rank);
                (Share::Files { lo, hi }, None, run_size(dirs, hi - lo, opts.writers))
            }
            DsMeta::Regions { .. } => {
                let owner = regions_owner(regions_index, ranks);
                regions_index += 1;
                (if owner == rank { Share::Regions } else { Share::Other { owner } }, Some(owner), 1)
            }
        };
        let track = Track::default();
        if let Share::Files { lo, .. } = share {
            track.claims.lock().unwrap().next = lo / run;
        }
        plans.push(Plan { meta, name, root, object, container, share, owner, run, track });
    }
    for p in &plans {
        match &p.share {
            Share::Files { lo, hi } => writeln!(log, "dataset {}: files [{lo}, {hi}) of {} on this rank", p.name, p.meta.files())?,
            Share::Regions => writeln!(log, "dataset {}: the regions file, written by this rank", p.name)?,
            Share::Other { owner } => writeln!(log, "dataset {}: the regions file, written by rank {owner}", p.name)?,
        }
    }
    log.flush()?;

    // the headers a container takes from its library: rank 0 makes them, Start carries them
    if rank == 0 {
        let share = make_headers(&mut plans, log)?;
        if let Some(server) = hosts.and_then(|h| h.server) {
            server.share(share);
        }
    }

    // the manifests from the start (DESIGN_REVIEW.md §3.70): `writing`, everything unfinished,
    // so a corpus never holds a file without saying what it is meant to be
    let started = now_secs();
    let mut manifests = Vec::new();
    for p in &plans {
        let mut m = base.manifest(loaded, cfg, &spec, p, ranks, started)?;
        let all = share_of(p.meta, Some(0), 1, 0);
        m.progress = json!({"unfinished": ranges_json(&all), "files_written": 0, "bytes_written": 0, "updated": started});
        if rank == 0 {
            write_manifest(p, &m)?;
        }
        manifests.push(m);
    }

    // the gate: no rank writes before every rank has checked; when it does not open, nothing
    // was written, and rank 0 takes back its manifests
    if let Some(h) = hosts {
        let (_t0, names, shared) = match h.tcp.ready_shared() {
            Ok(x) => x,
            Err(e) => {
                if rank == 0 {
                    for p in &plans {
                        let _ = match &p.object {
                            Some((store, rest)) => store.delete_all(&[payload::join_key(rest, payload::MANIFEST_NAME)]).map_err(anyhow::Error::from),
                            None => std::fs::remove_file(p.root.join(payload::MANIFEST_NAME)).map_err(anyhow::Error::from),
                        };
                    }
                }
                return Err(e);
            }
        };
        writeln!(log, "start gate: {} host(s) ready ({})", names.len(), names.join(", "))?;
        log.flush()?;
        if rank != 0 {
            take_headers(&mut plans, &shared)?;
        }
    }

    // the writes, with a progress window beside them; then the metadata to stable storage, a
    // last note, and (several hosts) this rank's part to rank 0, whose window runs until
    // every part is in
    let mode = Mode::new();
    let ticker = Ticker { plans: &plans, manifests: &manifests, opts, hosts, t0: Instant::now() };
    let log_m: Mutex<&mut (dyn Write + Send)> = Mutex::new(log);
    let (stop, stopped) = std::sync::mpsc::channel::<()>();
    let mut parts: BTreeMap<i64, Value> = BTreeMap::new();
    let mut written = Vec::new();
    let phase: Result<()> = std::thread::scope(|s| {
        let (ticker, log_m) = (&ticker, &log_m);
        s.spawn(move || loop {
            match stopped.recv_timeout(opts.progress) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Err(e) = ticker.tick(&log_m) {
                        let _ = writeln!(log_m.lock().unwrap(), "  progress: {e:#}");
                    }
                }
                _ => break,
            }
        });
        let r = (|| {
            for p in &plans {
                let w = write_share(p, opts, &spec, &mode);
                for n in mode.take_notes() {
                    writeln!(log_m.lock().unwrap(), "  {n}")?;
                }
                written.push(w?);
            }
            flush(opts)?;
            let mine = json!({
                "host": host,
                "direct": mode.direct(),
                "datasets": plans.iter().zip(&written).map(|(p, w)| (p.name.to_string(), part_json(rank, &host, w))).collect::<serde_json::Map<_, _>>(),
            });
            match hosts {
                None => {
                    parts.insert(0, mine);
                }
                Some(h) => {
                    // the last note, so rank 0's manifests do not wait a window for this rank
                    h.tcp.progress(&progress_doc(&plans))?;
                    h.tcp.done(&mine)?;
                    if let Some(server) = h.server {
                        parts = server.gathered().context("gathering every rank's part")?;
                    }
                }
            }
            Ok(())
        })();
        let _ = stop.send(());
        r
    });
    let log = log_m.into_inner().unwrap();
    if let Err(e) = phase {
        // what is finished, recorded before the error is reported
        if rank == 0 || hosts.is_some() {
            let _ = ticker.tick(&Mutex::new(&mut std::io::sink() as &mut (dyn Write + Send)));
        }
        return Err(e);
    }
    let wrote_manifest = rank == 0;
    if let (Some(h), false) = (hosts, wrote_manifest) {
        let (ok, _, error) = h.tcp.result()?;
        if !ok {
            bail!("{}", error.unwrap_or_else(|| "rank 0 failed".into()));
        }
    }

    // the manifests, `complete`: rank 0's, once every rank's part is in
    let mut results = Vec::new();
    let mut manifest_error = None;
    for ((p, w), base_m) in plans.iter().zip(&written).zip(&manifests) {
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
        let mut manifest = base_m.clone();
        manifest.state = payload::STATE_COMPLETE.into();
        manifest.progress = Value::Null;
        let prov = &mut manifest.provenance;
        prov["host"] = json!(parts.get(&0).and_then(|d| d["host"].as_str()).unwrap_or(&host));
        prov["hosts"] = json!(host_parts);
        prov["direct"] = json!(direct);
        prov["started"] = json!(if started == u64::MAX { w.started } else { started });
        prov["finished"] = json!(finished.max(w.finished));
        prov["files_written"] = json!(total_files);
        prov["bytes_written"] = json!(total_bytes);
        if wrote_manifest {
            if let Err(e) = write_manifest(p, &manifest) {
                manifest_error = Some(format!("{e:#}"));
                break;
            }
        }
        results.push(DatasetResult {
            name: p.name.to_string(),
            root: match &p.object {
                Some((store, rest)) => PathBuf::from(store.show(rest)),
                None => p.root.clone(),
            },
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
            format: p.container.as_ref().map(|c| format!("{}, {}", c.format["class"].as_str().unwrap_or("?"), c.format["writer"].as_str().unwrap_or("?"))),
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
    let _ = log;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_in_order() {
        assert_eq!(merge_ranges(vec![(5, 9), (0, 2), (2, 3), (8, 12), (4, 4)]), vec![(0, 3), (5, 12)]);
        assert_eq!(merge_ranges(vec![]), vec![]);
    }
}
