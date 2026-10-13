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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// `--resume`: the manifest of the one dataset to finish (`DESIGN_REVIEW.md` §3.70); the
    /// ids it lists as unfinished are written again, whole, and nothing else.
    pub resume: Option<Manifest>,
    /// `--grow` / `--shrink`, begun by this datagen: `resume` then holds `resize_manifest`'s
    /// manifest, which names the new count.
    pub resize: Option<Resize>,
}

#[derive(Debug, Clone)]
pub struct Resize {
    /// The complete corpus's manifest, put back when the start gate does not open.
    pub original: Manifest,
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

/// The manifest of the dataset at `root`, a directory or `s3://BUCKET[/PREFIX]`, for `--resume`.
pub fn resume_manifest(root: &str) -> Result<Manifest> {
    if crate::object::is_uri(root) {
        let store = crate::object::Store::open(root)?;
        Manifest::read_object(&store, "")
    } else {
        Manifest::read(Path::new(root))
    }
}

/// The abstract a resume runs (`DESIGN_REVIEW.md` §3.70): the manifest's resolved dataset, the
/// only one, under the name and abstract it was written for, placed at `root` (a name declared
/// `protocol: s3` when `root` is a bucket), with an actor that does nothing. Its datasets
/// resolve to the manifest's own, so the corpus's id is unchanged.
pub fn resume_abstract(m: &Manifest, root: &str) -> Result<(String, Vec<String>)> {
    let name = m.provenance["dataset"].as_str().unwrap_or("dataset").to_string();
    let mut dataset = m.dataset.clone();
    if crate::object::is_uri(root) {
        if let Some(inner) = dataset.as_object_mut().and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut) {
            inner.insert("protocol".into(), json!("s3"));
        }
    }
    let doc = json!({
        "ast": crate::ast::AST_VERSION,
        "name": m.provenance["abstract"].as_str().unwrap_or("resume"),
        "datasets": {name.clone(): dataset},
        "actors": {"resume": {"body": []}},
    });
    Ok((serde_json::to_string(&doc)?, vec![format!("{name}={root}")]))
}

/// The manifest `--grow` or `--shrink` starts from (`DESIGN_REVIEW.md` §3.70): the complete
/// corpus's, its dataset's `count` replaced by `count`, `growing` or `shrinking`, the count it
/// had in `progress.from` (what a resume of it needs to know what to remove), and the change
/// begun recorded in `provenance.resizes`.
pub fn resize_manifest(m: &Manifest, from: i64, count: i64) -> Result<Manifest> {
    let mut r = m.clone();
    let Some(files) = r.dataset.get_mut("files").and_then(Value::as_object_mut) else {
        bail!("only a `files` dataset grows or shrinks");
    };
    files.insert("count".into(), json!(count));
    let grow = count > from;
    r.state = if grow { payload::STATE_GROWING } else { payload::STATE_SHRINKING }.into();
    r.progress = json!({"from": from});
    let this = json!({"op": if grow { "grow" } else { "shrink" }, "from": from, "from_id": m.id(), "to": count, "started": now_secs()});
    match r.provenance["resizes"].as_array_mut() {
        Some(a) => a.push(this),
        None => r.provenance["resizes"] = json!([this]),
    }
    Ok(r)
}

/// Rank `rank`'s part of the ids `work` (sorted, disjoint ranges): the contiguous split of
/// their concatenation, so a fresh datagen's `[0, files)` splits as `file_slice` does and a
/// resume's unfinished ranges split over any host count.
pub fn work_slice(work: &[(i64, i64)], ranks: i64, rank: i64) -> Vec<(i64, i64)> {
    let total: i64 = work.iter().map(|(a, b)| b - a).sum();
    let (lo, hi) = crate::run::gpu_range(total, ranks, rank, 0);
    let (mut out, mut at) = (Vec::new(), 0i64);
    for &(a, b) in work {
        let (from, to) = (lo.max(at), hi.min(at + b - a));
        if from < to {
            out.push((a + from - at, a + to - at));
        }
        at += b - a;
    }
    out
}

/// The runs handed to writers: each range cut at multiples of `run`, so a run never spans
/// two directories of a `{id div N}` pattern when `run` divides `N`.
fn cut_runs(ranges: &[(i64, i64)], run: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for &(a, b) in ranges {
        let mut at = a;
        while at < b {
            let end = ((at / run + 1) * run).min(b);
            out.push((at, end));
            at = end;
        }
    }
    out
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
    /// The files of these id ranges of a `files` dataset.
    Files { ranges: Vec<(i64, i64)> },
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
    /// A resumed object dataset: the keys already there (one `LIST`), each a whole object,
    /// since an upload that did not complete leaves none.
    present: std::collections::HashSet<String>,
    /// The runs handed to writers, in id order (`run_size`, `cut_runs`).
    runs: Vec<(i64, i64)>,
    /// Every rank's ids, before any has said anything: `[0, files)`, or a resume's unfinished
    /// ranges (a regions file is id 0).
    work: Vec<(i64, i64)>,
    /// Files and bytes written and flushed before this datagen, by the one it resumes (a grow
    /// or shrink: what the corpus keeps).
    before: (u64, u64),
    /// A grow or shrink: the count the corpus had, and the old names to remove.
    from: Option<i64>,
    remove: Option<Remove<'a>>,
    track: Track,
}

/// What a grow or shrink removes: the names of ids `[ids.0, ids.1)` under the old count that
/// the new count does not have (the ids past it; the extra chunks of a chunked file it cuts).
struct Remove<'a> {
    old: DsMeta<'a>,
    ids: (i64, i64),
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
    /// Names a grow or shrink removed.
    removed: AtomicU64,
}

#[derive(Default)]
struct Claims {
    /// The next run to hand out, an index into the plan's `runs`.
    next: usize,
    /// Each writer's run `[start, end)` and the first id of it not yet written, which the
    /// writer advances under its lock together with the counts, so a window reads a file as
    /// finished and counted, or as neither.
    slots: Vec<(i64, i64, std::sync::Arc<Mutex<i64>>)>,
}

impl Plan<'_, '_> {
    /// An object already in the store, which a resume does not send again.
    fn present(&self, opts: &DatagenOpts, path: &str) -> bool {
        !self.present.is_empty() && opts.endpoints.object(path).is_some_and(|(_, rest)| self.present.contains(&rest))
    }

    /// The id ranges of this rank's share not yet written, merged, in order (a regions file
    /// is id 0), and the files and bytes of the rest, read together: every writer's lock is
    /// held while its cursor and the counts are read, and a regions file counts once whole.
    fn unfinished(&self) -> (Vec<(i64, i64)>, u64, u64) {
        let mut v = Vec::new();
        let counts = || (self.track.files.load(Ordering::Relaxed), self.track.bytes.load(Ordering::Relaxed));
        let (files, bytes) = match self.share {
            Share::Other { .. } => (0, 0),
            Share::Regions => {
                if self.track.whole.load(Ordering::Acquire) {
                    counts()
                } else {
                    v.push((0, 1));
                    (0, 0)
                }
            }
            Share::Files { .. } => {
                let c = self.track.claims.lock().unwrap();
                let held: Vec<_> = c.slots.iter().map(|(_, end, cursor)| (*end, cursor.lock().unwrap())).collect();
                for (end, at) in &held {
                    if **at < *end {
                        v.push((**at, *end));
                    }
                }
                v.extend_from_slice(&self.runs[c.next.min(self.runs.len())..]);
                counts()
            }
        };
        (merge_ranges(v), files, bytes)
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
fn share_of(p: &Plan<'_, '_>, ranks: i64, r: i64) -> Vec<(i64, i64)> {
    match p.meta {
        DsMeta::Files { .. } => work_slice(&p.work, ranks, r),
        DsMeta::Regions { .. } => {
            if p.owner == Some(r) {
                p.work.clone()
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

/// The names of file `id`: its chunks', or its own.
fn names(meta: &DsMeta<'_>, id: i64) -> Result<Vec<String>> {
    match meta {
        DsMeta::Files { chunk: Some(_), .. } => (0..meta.chunks(id)?).map(|k| Ok(meta.file_path(id, Some(k))?.to_string())).collect(),
        _ => Ok(vec![meta.file_path(id, None)?.to_string()]),
    }
}

/// The objects file `id` is: its chunks, or one.
fn objects_of(meta: &DsMeta<'_>, id: i64) -> Result<u64> {
    match meta {
        DsMeta::Files { chunk: Some(_), .. } => Ok(meta.chunks(id)? as u64),
        _ => Ok(1),
    }
}

/// Remove `dir` and each parent below `root` while they are empty.
fn prune(dir: &Path, root: &Path) {
    let mut d = dir;
    while d != root && d.starts_with(root) && std::fs::remove_dir(d).is_ok() {
        match d.parent() {
            Some(p) => d = p,
            None => break,
        }
    }
}

/// A grow or shrink: remove this rank's part of the old names the new count does not have,
/// and every directory that leaves empty, so the corpus is what a datagen of the new count
/// writes (`DESIGN_REVIEW.md` §3.70). A name already gone is no error, so a resume removes
/// again; objects go in batches of `DELETE_BATCH`.
fn remove_old(plan: &Plan<'_, '_>, rm: &Remove<'_>, opts: &DatagenOpts) -> Result<()> {
    const DELETE_BATCH: usize = 1000;
    let DsMeta::Files { dirs, .. } = plan.meta else { return Ok(()) };
    let new_files = plan.meta.files();
    let mine = work_slice(&[rm.ids], opts.ranks.max(1), opts.rank);
    let runs = cut_runs(&mine, run_size(dirs, mine.iter().map(|(a, b)| b - a).sum(), opts.writers));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
    let removed = &plan.track.removed;
    std::thread::scope(|s| {
        for _ in 0..opts.writers.max(1).min(runs.len()) {
            s.spawn(|| {
                let r = (|| -> Result<()> {
                    let mut batch: Option<(std::sync::Arc<crate::object::Store>, Vec<String>)> = None;
                    let send = |b: &mut Option<(std::sync::Arc<crate::object::Store>, Vec<String>)>| -> Result<()> {
                        if let Some((store, keys)) = b.as_mut().filter(|(_, k)| !k.is_empty()) {
                            store.delete_all(keys)?;
                            removed.fetch_add(keys.len() as u64, Ordering::Relaxed);
                            keys.clear();
                        }
                        Ok(())
                    };
                    while let Some(&(a, b)) = runs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let mut last_dir: Option<PathBuf> = None;
                        for id in a..b {
                            let keep = if id < new_files { names(plan.meta, id)? } else { Vec::new() };
                            for path in names(&rm.old, id)?.into_iter().filter(|n| !keep.contains(n)) {
                                if let Some((store, rest)) = opts.endpoints.object(&path) {
                                    batch.get_or_insert_with(|| (store, Vec::new())).1.push(rest);
                                    if batch.as_ref().is_some_and(|(_, k)| k.len() >= DELETE_BATCH) {
                                        send(&mut batch)?;
                                    }
                                    continue;
                                }
                                let full = opts.endpoints.path(&opts.root, &path);
                                match std::fs::remove_file(&full) {
                                    Ok(()) => {
                                        removed.fetch_add(1, Ordering::Relaxed);
                                    }
                                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                                    Err(e) => return Err(e).with_context(|| format!("removing {}", full.display())),
                                }
                                let dir = full.parent().map(Path::to_path_buf);
                                if dir != last_dir {
                                    if let Some(d) = last_dir.take() {
                                        prune(&d, &plan.root);
                                    }
                                    last_dir = dir;
                                }
                            }
                        }
                        if let Some(d) = last_dir {
                            prune(&d, &plan.root);
                        }
                    }
                    send(&mut batch)
                })();
                if let Err(e) = r {
                    errors.lock().unwrap().push(e);
                    // the others stop at their next run
                    next.store(usize::MAX / 2, Ordering::Relaxed);
                }
            });
        }
    });
    match errors.into_inner().unwrap().into_iter().next() {
        Some(e) => Err(e.context(format!("dataset `{}`", plan.name))),
        None => Ok(()),
    }
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
        (Share::Files { .. }, DsMeta::Files { spf, chunk, count, .. }) => {
            if let Some(rm) = &plan.remove {
                remove_old(plan, rm, opts)?;
            }
            let runs = &plan.runs;
            if !runs.is_empty() {
                let nthreads = writers.min(runs.len());
                track.claims.lock().unwrap().slots = (0..nthreads).map(|_| (0, 0, std::sync::Arc::new(Mutex::new(0)))).collect();
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
                                    let Some(&(a, b)) = runs.get(c.next) else { break };
                                    c.next += 1;
                                    let cursor = c.slots[w].2.clone();
                                    *cursor.lock().unwrap() = a;
                                    c.slots[w].0 = a;
                                    c.slots[w].1 = b;
                                    (a, b, cursor)
                                };
                                for file in a..b {
                                    // this file's objects and bytes, counted when it is whole
                                    let mut got = (0u64, 0u64);
                                    let r: Result<()> = (|| {
                                        if file * spf >= *count {
                                            return Ok(());
                                        }
                                        let size = meta.file_size(file)? as u64;
                                        let seed_of = |blk: u64| payload::file_block_seed(seed, file as u64, blk, dedupe);
                                        if let Some(c) = &plan.container {
                                            let path = meta.file_path(file, None)?.to_string();
                                            if plan.present(opts, &path) {
                                                return Ok(());
                                            }
                                            got = (1, write_container(c, meta, file, &seed_of, &mut filler, &mut buf, &path, opts, mode)?);
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
                                            if plan.present(opts, &path) {
                                                continue;
                                            }
                                            if let Some((store, rest)) = opts.endpoints.object(&path) {
                                                store.put_parts(&rest, len, crate::object::DEFAULT_PART, &mut |at, b| filler.fill_range(&seed_of, logical + at, b))?;
                                                got = (got.0 + 1, got.1 + len);
                                                continue;
                                            }
                                            let full = opts.endpoints.path(&opts.root, &path);
                                            let (f, direct) = create(&full, mode)?;
                                            write_range(&f, direct, mode, &seed_of, logical, 0, len, &mut filler, &mut buf, &full)?;
                                            got = (got.0 + 1, got.1 + len);
                                        }
                                        Ok(())
                                    })();
                                    if let Err(e) = r {
                                        errors.lock().unwrap().push(e);
                                        failed.store(true, Ordering::Relaxed);
                                        break;
                                    }
                                    let mut at = cursor.lock().unwrap();
                                    track.files.fetch_add(got.0, Ordering::Relaxed);
                                    track.bytes.fetch_add(got.1, Ordering::Relaxed);
                                    *at = file + 1;
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

/// A payload as an error message names it.
fn payload_line(p: &PayloadSpec) -> String {
    format!("{} {} ({}, block {}, dedupe {}, compress {})", p.generator, p.version, p.wrapper, p.block, p.dedupe, p.compress)
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
                "dataset": p.name,
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
    json!({"datasets": plans.iter().map(|p| {
        let (unfinished, files, bytes) = p.unfinished();
        (p.name.to_string(), json!({"unfinished": ranges_json(&unfinished), "files": files, "bytes": bytes}))
    }).collect::<serde_json::Map<_, _>>()})
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
                    None => unfinished.extend(share_of(p, ranks, r)),
                }
            }
            let unfinished = merge_ranges(unfinished);
            let (files, b) = (files + p.before.0, b + p.before.1);
            total += match p.meta {
                DsMeta::Files { .. } => p.meta.files(),
                DsMeta::Regions { .. } => 1,
            };
            left += unfinished.iter().map(|(a, b)| b - a).sum::<i64>();
            bytes += b - p.before.1;
            let mut m = base.clone();
            m.progress = json!({"unfinished": ranges_json(&unfinished), "files_written": files, "bytes_written": b, "updated": now_secs()});
            if let Some(from) = p.from {
                m.progress["from"] = json!(from);
            }
            write_manifest(p, &m)?;
        }
        let secs = self.t0.elapsed().as_secs_f64();
        let done = total - left;
        // the rate of this datagen alone: a resume starts with the work of the one before
        let now = self.plans.iter().map(|p| p.work.iter().map(|(a, b)| b - a).sum::<i64>()).sum::<i64>() - left;
        let eta = if now > 0 && left > 0 { format!(", about {:.0} s left", secs * left as f64 / now as f64) } else { String::new() };
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
    let resume = opts.resume.as_ref();
    // a grow or shrink, begun now or resumed: the count the corpus had
    let from = resume.and_then(|m| m.progress["from"].as_i64());
    if let Some(m) = resume {
        if m.payload != spec {
            bail!("the corpus was written with the payload {} and this aeiou writes {}: a resume would mix them", payload_line(&m.payload), payload_line(&spec));
        }
        if model.datasets.len() != 1 {
            bail!("a resume finishes one dataset, not {}", model.datasets.len());
        }
    }

    // the plan: what each dataset is, where it goes, this rank's share; every rank checks
    // the roots are empty before the gate, so no rank sees another's files (rank 0's
    // manifest, written before the gate, aside); a resume checks instead that the format
    // writer is the one that wrote the corpus
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
        if let Some(m) = resume {
            let mine = container.as_ref().map(|c| &c.format);
            if mine != m.format.as_ref() {
                let show = |f: Option<&Value>| f.map(|f| f.to_string()).unwrap_or_else(|| "none".into());
                bail!("dataset `{name}`: the corpus's format is {} and this aeiou writes {}: a resume would mix them", show(m.format.as_ref()), show(mine));
            }
        }
        if object.is_some() && matches!(meta, DsMeta::Regions { .. }) {
            bail!("dataset `{name}`: a `regions` dataset is a sparse layout written at offsets and has no object form; place it in a directory");
        }
        let mut present = std::collections::HashSet::new();
        if let (Some((store, rest)), true) = (&object, resume.is_some() && opts.resize.is_none()) {
            present = store.keys_below(rest)?.into_iter().collect();
        } else if let (Some((store, rest)), true) = (&object, resume.is_none()) {
            let busy = if rank == 0 { store.non_empty(rest)? } else { store.non_empty_except(rest, payload::MANIFEST_NAME)? };
            if busy {
                bail!("dataset `{name}`: {} is not empty; datasets are read-only, remove it first", store.show(rest));
            }
        } else if resume.is_none() && root.exists() {
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
        let mut remove = None;
        let (work, before) = match resume {
            Some(m) if opts.resize.is_none() => (merge_ranges(m.unfinished()), (m.progress["files_written"].as_u64().unwrap_or(0), m.progress["bytes_written"].as_u64().unwrap_or(0))),
            // a grow or shrink begun now: the ids from the first whose bytes change (the old last
            // file when it was short) to the new count are written, and the corpus keeps the rest
            Some(m) => {
                let (Some(c0), DsMeta::Files { spf, .. }) = (from, meta) else { bail!("dataset `{name}`: only a `files` dataset grows or shrinks") };
                let old = meta.with_count(c0);
                let w0 = c0.min(meta.count()) / spf;
                let (mut files, mut bytes) = (0u64, 0u64);
                for id in w0..old.files() {
                    files += objects_of(&old, id)?;
                    bytes += old.file_size(id)? as u64;
                }
                let kept = (m.provenance["files_written"].as_u64().unwrap_or(0).saturating_sub(files), m.provenance["bytes_written"].as_u64().unwrap_or(0).saturating_sub(bytes));
                (if w0 < meta.files() { vec![(w0, meta.files())] } else { Vec::new() }, kept)
            }
            None => (vec![(0, if let DsMeta::Files { .. } = meta { meta.files() } else { 1 })], (0, 0)),
        };
        if let (Some(c0), DsMeta::Files { spf, .. }) = (from, meta) {
            let old = meta.with_count(c0);
            let w0 = c0.min(meta.count()) / spf;
            if c0 % spf != 0 && w0 < meta.files() {
                // the old short last file is there whole but is not the new one's: write it again
                for n in names(meta, w0)? {
                    if let Some((_, rest)) = opts.endpoints.object(&n) {
                        present.remove(&rest);
                    }
                }
            }
            remove = Some(Remove { ids: (w0, old.files()), old });
        }
        let (share, owner, runs) = match meta {
            DsMeta::Files { dirs, .. } => {
                let ranges = work_slice(&work, ranks, rank);
                let run = run_size(dirs, ranges.iter().map(|(a, b)| b - a).sum(), opts.writers);
                let runs = cut_runs(&ranges, run);
                (Share::Files { ranges }, None, runs)
            }
            DsMeta::Regions { .. } => {
                let owner = regions_owner(regions_index, ranks);
                regions_index += 1;
                // a resume whose regions file was finished has nothing to write
                (if owner == rank && !work.is_empty() { Share::Regions } else { Share::Other { owner } }, Some(owner), Vec::new())
            }
        };
        plans.push(Plan { meta, name, root, object, container, share, owner, present, runs, work, before, from, remove, track: Track::default() });
    }
    for p in &plans {
        if let (Some(c0), Some(rm)) = (p.from, &p.remove) {
            let w = p.work.iter().map(|(a, b)| b - a).sum::<i64>();
            let gone = if rm.ids.0 < rm.ids.1 { format!("; old names of ids [{}, {}) removed where the new count has none", rm.ids.0, rm.ids.1) } else { String::new() };
            writeln!(log, "dataset {}: count {c0} -> {}: {w} file(s) to write{gone}", p.name, p.meta.count())?;
        }
        match &p.share {
            Share::Files { ranges } if resume.is_some() => writeln!(log, "dataset {}: {} of {} unfinished file(s) on this rank, in {} range(s) from {}", p.name, ranges.iter().map(|(a, b)| b - a).sum::<i64>(), p.work.iter().map(|(a, b)| b - a).sum::<i64>(), ranges.len(), ranges.first().map(|r| r.0).unwrap_or(0))?,
            Share::Files { ranges } => {
                let (lo, hi) = ranges.first().map(|r| (r.0, r.1)).unwrap_or((0, 0));
                writeln!(log, "dataset {}: files [{lo}, {hi}) of {} on this rank", p.name, p.meta.files())?
            }
            Share::Regions => writeln!(log, "dataset {}: the regions file, written by this rank", p.name)?,
            Share::Other { owner } => writeln!(log, "dataset {}: the regions file, written by rank {owner}", p.name)?,
        }
    }
    log.flush()?;

    // the headers a container takes from its library: rank 0 makes them, Start carries them
    if rank == 0 {
        let share = make_headers(&mut plans, log)?;
        if let Some(m) = resume {
            // the library that made the corpus's headers must make the same ones now
            for p in &plans {
                let Some(c) = &p.container else { continue };
                // a grow or shrink changes the short last file's header alone: those both have must agree
                let (now, was) = (&c.headers["sha256"], &m.provenance["headers"]["sha256"]);
                let differ = match opts.resize {
                    None => now != was,
                    Some(_) => now.as_object().into_iter().flatten().any(|(k, v)| was.get(k).is_some_and(|w| w != v)),
                };
                if !c.headers.is_null() && differ {
                    bail!("dataset `{}`: aeiou-header made headers other than the corpus's ({} against {}): a resume would mix them", p.name, c.headers["sha256"], m.provenance["headers"]["sha256"]);
                }
            }
        }
        if let Some(server) = hosts.and_then(|h| h.server) {
            server.share(share);
        }
    }

    // the manifests from the start (DESIGN_REVIEW.md §3.70): `writing`, everything unfinished,
    // so a corpus never holds a file without saying what it is meant to be; a resume keeps
    // the one there, which already says so
    let started = now_secs();
    let mut manifests = Vec::new();
    for p in &plans {
        if let Some(m) = resume {
            let mut m = m.clone();
            if opts.resize.is_some() {
                // the new count's manifest, before anything is removed or written
                m.progress = json!({"unfinished": ranges_json(&p.work), "files_written": p.before.0, "bytes_written": p.before.1, "from": p.from, "updated": started});
                if let Some(c) = p.container.as_ref().filter(|c| !c.headers.is_null()) {
                    m.provenance["headers"] = c.headers.clone();
                }
                if rank == 0 {
                    write_manifest(p, &m)?;
                }
            }
            manifests.push(m);
            continue;
        }
        let mut m = base.manifest(loaded, cfg, &spec, p, ranks, started)?;
        m.progress = json!({"unfinished": ranges_json(&p.work), "files_written": 0, "bytes_written": 0, "updated": started});
        if rank == 0 {
            write_manifest(p, &m)?;
        }
        manifests.push(m);
    }

    // the gate: no rank writes before every rank has checked; when it does not open, nothing
    // was written, and rank 0 takes back its manifests (never a resumed corpus's)
    if let Some(h) = hosts {
        let (_t0, names, shared) = match h.tcp.ready_shared() {
            Ok(x) => x,
            Err(e) => {
                if let (0, Some(rz)) = (rank, &opts.resize) {
                    // nothing was removed or written: the corpus is the complete one it was
                    for p in &plans {
                        let _ = write_manifest(p, &rz.original);
                    }
                } else if rank == 0 && resume.is_none() {
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
                if p.remove.is_some() {
                    writeln!(log_m.lock().unwrap(), "dataset {}: {} old name(s) removed on this rank", p.name, p.track.removed.load(Ordering::Relaxed))?;
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
        let started = if started == u64::MAX { w.started } else { started };
        if opts.resize.is_some() {
            // the change begun completes its record; the totals are what the corpus holds
            let finished = finished.max(w.finished);
            if let Some(last) = prov["resizes"].as_array_mut().and_then(|a| a.last_mut()) {
                for (k, v) in [("finished", json!(finished)), ("ranks", json!(ranks)), ("hosts", json!(host_parts)), ("direct", json!(direct)), ("files_written", json!(total_files)), ("bytes_written", json!(total_bytes))] {
                    last[k] = v;
                }
            }
            prov["finished"] = json!(finished);
            prov["files_written"] = json!(total_files + p.before.0);
            prov["bytes_written"] = json!(total_bytes + p.before.1);
        } else if resume.is_some() {
            // the datagen resumed keeps its record; each resume adds its own, and the totals
            // count every file written and flushed once (what the last window recorded, then
            // this datagen's)
            let this = json!({"started": started, "finished": finished.max(w.finished), "ranks": ranks, "hosts": host_parts, "direct": direct, "files_written": total_files, "bytes_written": total_bytes});
            match prov["resumes"].as_array_mut() {
                Some(a) => a.push(this),
                None => prov["resumes"] = json!([this]),
            }
            prov["finished"] = json!(finished.max(w.finished));
            prov["files_written"] = json!(total_files + p.before.0);
            prov["bytes_written"] = json!(total_bytes + p.before.1);
        } else {
            prov["host"] = json!(parts.get(&0).and_then(|d| d["host"].as_str()).unwrap_or(&host));
            prov["hosts"] = json!(host_parts);
            prov["direct"] = json!(direct);
            prov["started"] = json!(started);
            prov["finished"] = json!(finished.max(w.finished));
            prov["files_written"] = json!(total_files);
            prov["bytes_written"] = json!(total_bytes);
        }
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
