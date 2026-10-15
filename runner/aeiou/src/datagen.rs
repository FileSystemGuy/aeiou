//! `aeiou datagen`: write the corpus an abstract declares, from the same names, sizes, and
//! dataset seeds the runner computes, with the payload of `payload.rs`, and a manifest at
//! each dataset root (`schema/README.md` §6). Files are written in parallel by id with no
//! per-file structure. The manifest is written first, `writing`, and rewritten every
//! `--progress-secs` with the ids not yet written and flushed, and `complete` at the end
//! (`DESIGN_REVIEW.md` §3.70), each time atomically.
//!
//! **Several hosts** (`--ranks R --rank r --coordinator`, 2026-10-04, `DESIGN_REVIEW.md`
//! §3.62). The ids of every `files` dataset are handed out in runs from one queue on rank 0,
//! in id order (§3.70, 2026-10-12: so an interrupted corpus is whole directories, a band of
//! partial ones, and nothing past it; and the hosts share the load), and each `array`
//! dataset is written whole by one rank (the i-th array
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
//! that size (larger for an object over 78 GiB, `object::part_for`), each part made and sent
//! before the next; the root's emptiness is a `LIST`, the
//! manifest one `PUT` (atomic, so no temporary name), and there is no `syncfs`. An `array`
//! dataset, a sparse layout written at offsets, has no object form and is refused.
//!
//! **Writers.** Each writer (`--writers`, a thread each) claims a run, writes it in id order,
//! and claims the next (over the coordinator, one claim ahead). A run is a directory of the
//! pattern (`{id div N}`), or `FLAT_RUN` ids of a flat one, cut smaller only when there are
//! fewer runs than writers on all hosts, because Linux takes the parent directory's lock
//! exclusively for every create, and on NFS the `OPEN` round trip runs under it: threads
//! creating in one directory serialize. A shrink claims its runs from the top down and
//! removes each from its highest id down. Rank 0 records a claim mark in the manifest before
//! handing out a run past it, so a settle knows how far strays can reach. An array file is
//! cut into 64 MiB pieces for the owner's writers (`pwrite` at disjoint offsets), and in
//! them only the slots' contents are written: each slot's tail is a hole.
//!
//! **Stopping.** Ctrl-C (`SIGINT`) on any host stops every host: each stops claiming, leaves
//! the file it is writing, notes what it finished, flushes, and reports; rank 0 writes the
//! manifests once more (`stopped`), and every host exits 130. A second Ctrl-C exits at once.
//! A host that fails stops the others the same way, but each reports its error and exits 1.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::coord::{Coordinator, Server, Tcp};
use crate::eval::{Config, DirScheme, DsMeta, Model, Params};
use crate::payload::{self, Filler, Manifest, PayloadSpec, BLOCK};

/// The alignment `O_DIRECT` wants of a buffer and a length (the largest common logical block).
pub const ALIGN: usize = 4096;
/// An array file is written in pieces of this many bytes, one per writer at a time.
const REGIONS_PIECE: u64 = 64 << 20;
/// The ids of a run of a pattern without `{id div N}` directories.
pub const FLAT_RUN: i64 = 1000;
/// Rank 0 moves the claim mark this many bands (a run for every writer on every host) past a
/// run it hands out beyond it, so the mark costs one manifest write every few bands.
const MARK_BANDS: usize = 4;

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
    /// `--settle`, begun by this datagen: `resume` then holds `settle_manifest`'s manifest,
    /// which names the settled count.
    pub settle: Option<Resize>,
    /// Ctrl-C stops every host (`catch_sigint`; the command line's, not a library caller's).
    pub sigint: bool,
    /// `--ignore-space`: write a corpus larger than the free space (`check_space`).
    pub ignore_space: bool,
}

impl DatagenOpts {
    /// The manifest a grow, shrink, or settle begun by this datagen replaces: rank 0 writes
    /// the new one before the gate and puts this one back when the gate does not open.
    fn begun(&self) -> Option<&Manifest> {
        self.resize.as_ref().or(self.settle.as_ref()).map(|r| &r.original)
    }
}

#[derive(Debug, Clone)]
pub struct Resize {
    /// The manifest replaced (a grow's or shrink's complete corpus, a settle's unfinished
    /// one), put back when the start gate does not open.
    pub original: Manifest,
}

/// A datagen stopped by Ctrl-C (`aeiou` exits 130); what it finished is recorded.
#[derive(Debug)]
pub struct Stopped(pub String);

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Stopped {}

/// A write left because the datagen is stopping; not a failure.
#[derive(Debug)]
struct Halted;

impl std::fmt::Display for Halted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stopped")
    }
}

impl std::error::Error for Halted {}

static SIGINTS: AtomicU32 = AtomicU32::new(0);

extern "C" fn on_sigint(_: libc::c_int) {
    // the second exits at once: the manifest of the last window is still true
    if SIGINTS.fetch_add(1, Ordering::SeqCst) > 0 {
        unsafe { libc::_exit(130) };
    }
}

/// Catch `SIGINT` from here on: the first asks every host to stop, the second exits.
fn catch_sigint() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_sigint as extern "C" fn(libc::c_int) as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }
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
    /// Of a resume's object keys, those found already in the store and not sent again: by
    /// this rank, and by every rank.
    pub found: u64,
    pub total_found: u64,
    pub id: String,
    pub elapsed: Duration,
    /// This rank wrote the manifest (rank 0; the only rank on one host).
    pub wrote_manifest: bool,
    /// An `array` dataset another rank owns.
    pub owner: Option<i64>,
    /// A container's class and writer, `class, writer`.
    pub format: Option<String>,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Now, in seconds to the millisecond (`progress.updated`, which the live check compares
/// with windows of a fraction of a second in tests).
pub fn now_f64() -> f64 {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    (t * 1000.0).round() / 1000.0
}

/// Refuse to finish a corpus a datagen may still be writing: one whose manifest was updated
/// less than two progress windows ago and does not say its datagen stopped
/// (`DESIGN_REVIEW.md` §3.70; a manifest from before the window was recorded counts 30 s).
pub fn check_not_live(m: &Manifest, what: &str) -> Result<()> {
    let p = &m.progress;
    if !p["stopped"].is_null() {
        return Ok(());
    }
    let secs = p["progress_secs"].as_f64().unwrap_or(30.0);
    let age = now_f64() - p["updated"].as_f64().unwrap_or(0.0);
    if age < 2.0 * secs {
        let show = |x: f64| if x >= 10.0 { format!("{x:.0}") } else { format!("{:.3}", x).trim_end_matches('0').trim_end_matches('.').to_string() };
        // a time ahead of this clock: this host's clock was stepped back (a virtual machine's
        // under load), or the writing host's runs ahead
        let when = if age < 0.0 { format!("{} s ahead of this host's clock", show(-age)) } else { format!("{} s ago", show(age)) };
        bail!("{what}: the manifest was updated {when} and its datagen notes progress every {} s, so it may still be writing; stop it, or wait until {} s have passed since the update", show(secs), show(2.0 * secs));
    }
    Ok(())
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
        if let Some(def) = dataset.as_object_mut() {
            def.insert("protocol".into(), json!("s3"));
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
    let Some(def) = r.dataset.as_object_mut().filter(|d| d["layout"] != "array") else {
        bail!("an `array` does not grow or shrink");
    };
    def.insert("count".into(), json!(count));
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

/// The runs rank 0 hands out, in the order it hands them out: the ranges of `work` cut at
/// the pattern's directories (`{id div N}`; every `FLAT_RUN` ids otherwise), each piece cut
/// again into equal parts when there are fewer pieces than `slots` (writers on every host);
/// from the top down when `desc` (a shrink).
pub fn runs_of(work: &[(i64, i64)], dirs: &DirScheme, slots: usize, desc: bool) -> Vec<(i64, i64)> {
    let per = match dirs {
        DirScheme::Div(n) => (*n).max(1),
        _ => FLAT_RUN,
    };
    let mut pieces = Vec::new();
    for &(a, b) in work {
        let mut at = a;
        while at < b {
            let end = ((at / per + 1) * per).min(b);
            pieces.push((at, end));
            at = end;
        }
    }
    if pieces.len() < slots {
        let total: i64 = pieces.iter().map(|(a, b)| b - a).sum();
        let size = ((total + slots as i64 - 1) / slots.max(1) as i64).max(1);
        pieces = pieces.into_iter().flat_map(|(a, b)| {
            let k = (b - a + size - 1) / size;
            (0..k).map(move |i| (a + (b - a) * i / k, a + (b - a) * (i + 1) / k))
        }).collect();
    }
    if desc {
        pieces.reverse();
    }
    pieces
}

/// The rank that writes the `index`-th `array` dataset (in the abstract's order, among the
/// datasets selected).
pub fn array_owner(index: usize, ranks: i64) -> i64 {
    (index as i64).rem_euclid(ranks.max(1))
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
    /// The datagen is stopping (Ctrl-C, or another host failed): claim nothing more, and
    /// leave the file in hand.
    halt: AtomicBool,
}

impl Mode {
    pub fn new() -> Self {
        Mode { direct: AtomicBool::new(true), pad_tails: AtomicBool::new(false), notes: Mutex::new(Vec::new()), halt: AtomicBool::new(false) }
    }
    pub fn direct(&self) -> bool {
        self.direct.load(Ordering::Relaxed)
    }
    fn halted(&self) -> bool {
        self.halt.load(Ordering::Relaxed)
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
    open_for_write(full, mode, true)
}

/// `create`, or with `truncate` false, open `full` keeping what it holds.
fn open_for_write(full: &Path, mode: &Mode, truncate: bool) -> Result<(File, bool)> {
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if mode.direct() {
        match OpenOptions::new().write(true).create(true).truncate(truncate).custom_flags(libc::O_DIRECT).open(full) {
            Ok(f) => return Ok((f, true)),
            Err(e) if is_einval(&e) => {
                mode.direct.store(false, Ordering::Relaxed);
                mode.note(format!("O_DIRECT is not supported under {}: writing through the page cache", full.parent().map(|p| p.display().to_string()).unwrap_or_default()));
            }
            Err(e) => return Err(e).with_context(|| format!("creating {}", full.display())),
        }
    }
    Ok((OpenOptions::new().write(true).create(true).truncate(truncate).open(full).with_context(|| format!("creating {}", full.display()))?, false))
}

/// Write dataset `p`'s inventory, if it has one, at the count `count` the corpus ends at
/// (contract 0.9, `DESIGN_REVIEW.md` §3.73): rank 0, once every rank has reported, before the
/// manifest says `complete`, so an inventory is of a written corpus. Its content is positional
/// (`inventory_block_seed`), so a file's whole blocks are kept when it was written directly
/// (a strict prefix, as every datagen file) and only the rest is written; a grow extends it,
/// a shrink truncates it. An object is one upload, whole.
fn write_inventory(p: &Plan<'_, '_>, count: i64, opts: &DatagenOpts, spec: &PayloadSpec, mode: &Mode) -> Result<()> {
    let Some((path, size, _)) = p.meta.with_count(count).inventory() else { return Ok(()) };
    let (size, seed, dedupe) = (size as u64, p.meta.dataset_seed(), spec.dedupe);
    let seed_of = |blk: u64| payload::inventory_block_seed(seed, blk, dedupe);
    let mut filler = Filler::new(spec.compress);
    if let Some((store, rest)) = opts.endpoints.object(&path) {
        store.put_parts(&rest, size, crate::object::DEFAULT_PART, &mut |at, b| filler.fill_range(&seed_of, at, b))
            .with_context(|| format!("dataset `{}`: writing its inventory {}", p.name, store.show(&rest)))?;
        return Ok(());
    }
    let full = opts.endpoints.path(&opts.root, &path);
    let (f, direct) = open_for_write(&full, mode, false)?;
    let had = f.metadata().with_context(|| format!("{}", full.display()))?.len();
    let keep = if direct { had.min(size) / BLOCK * BLOCK } else { 0 };
    f.set_len(keep).with_context(|| format!("truncating {}", full.display()))?;
    let mut buf = Aligned::new();
    write_range(&f, direct, mode, &seed_of, keep, keep, size - keep, &mut filler, &mut buf, &full)
        .with_context(|| format!("dataset `{}`: writing its inventory", p.name))?;
    f.sync_all().with_context(|| format!("flushing {}", full.display()))?;
    Ok(())
}

/// Write `len` bytes of content at file offset `at`, the content being logical offset
/// `logical..` of the stream `seed_of` describes, in `BLOCK` pieces. Under `O_DIRECT` an
/// unaligned tail is tried as is, then padded to `ALIGN` and the file truncated to `at + len`.
fn write_range(f: &File, direct: bool, mode: &Mode, seed_of: &dyn Fn(u64) -> u64, logical: u64, at: u64, len: u64, filler: &mut Filler, buf: &mut Aligned, what: &Path) -> Result<()> {
    let mut done = 0u64;
    while done < len {
        if mode.halted() {
            return Err(Halted.into());
        }
        let n = (len - done).min(BLOCK) as usize;
        filler.fill_range(seed_of, logical + done, &mut buf[..n]);
        write_at(f, direct, mode, buf, n, at + done, what)?;
        done += n as u64;
    }
    Ok(())
}

/// Write the first `n` bytes of `buf` (aligned, with room to pad `n` to `ALIGN`) at `off`;
/// only the file's last write may be unaligned.
fn write_at(f: &File, direct: bool, mode: &Mode, buf: &mut [u8], n: usize, off: u64, what: &Path) -> Result<()> {
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

/// Write the slots' contents that fall in `[at, end)` of an array file whose `count` slots are
/// `slot` bytes apart (`DESIGN_REVIEW.md` §3.72: a sparse file, each slot's tail unwritten),
/// `at` aligned to `BLOCK` and `end` too unless it is the file's end. Each content
/// `[c·slot, c·slot + size(c))` is the bytes of its file offset (`seed_of`), as a dense file
/// would hold them; the `ALIGN` blocks holding any content are written, `BLOCK` at a time,
/// zeros where they are not content (what a hole reads as), and the rest are left holes.
/// Returns the content bytes written.
#[allow(clippy::too_many_arguments)]
fn write_slots(f: &File, direct: bool, mode: &Mode, meta: &DsMeta<'_>, slot: u64, count: u64, seed_of: &dyn Fn(u64) -> u64, at: u64, end: u64, filler: &mut Filler, buf: &mut Aligned, what: &Path) -> Result<u64> {
    let mut content = 0u64;
    if slot == 0 || count == 0 {
        return Ok(0);
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut cs = at;
    while cs < end {
        if mode.halted() {
            return Err(Halted.into());
        }
        let ce = (cs + BLOCK).min(end);
        // the contents in this block, adjoining ones (full slots) merged
        runs.clear();
        for c in cs / slot..=((ce - 1) / slot).min(count - 1) {
            let (s, e) = (c * slot, c * slot + meta.sample_size(c as i64)? as u64);
            let (s, e) = (s.max(cs), e.min(ce));
            if s >= e {
                continue;
            }
            let (s, e) = ((s - cs) as usize, (e - cs) as usize);
            match runs.last_mut() {
                Some(r) if r.1 == s => r.1 = e,
                _ => runs.push((s, e)),
            }
        }
        let n = (ce - cs) as usize;
        if !(runs.len() == 1 && runs[0] == (0, n)) {
            buf[..n].fill(0);
        }
        for &(s, e) in &runs {
            filler.fill_range(seed_of, cs + s as u64, &mut buf[s..e]);
            content += (e - s) as u64;
        }
        // the ALIGN blocks the contents touch, adjoining ones written together
        let mut i = 0;
        while i < runs.len() {
            let a = runs[i].0 / ALIGN * ALIGN;
            let mut b = (runs[i].1 + ALIGN - 1) / ALIGN * ALIGN;
            i += 1;
            while i < runs.len() && runs[i].0 / ALIGN * ALIGN <= b {
                b = (runs[i].1 + ALIGN - 1) / ALIGN * ALIGN;
                i += 1;
            }
            let b = b.min(n);
            write_at(f, direct, mode, &mut buf[a..], b - a, cs + a as u64, what)?;
        }
        cs = ce;
    }
    Ok(content)
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
    let fmt = &resolved["format"];
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
            if mode.halted() {
                return Err(Halted.into());
            }
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
    /// Runs of a `files` dataset's ids, claimed from rank 0's queue.
    Files,
    /// The whole `array` file.
    Array,
    /// Nothing: an `array` dataset another rank owns.
    Other { owner: i64 },
}

struct Plan<'m, 'a> {
    meta: &'m DsMeta<'a>,
    name: &'a str,
    root: PathBuf,
    /// The store and root key of an object dataset.
    object: Option<(Arc<crate::object::Store>, String)>,
    /// The writer of a format-class dataset.
    container: Option<Container>,
    share: Share,
    /// The rank that writes an `array` dataset.
    owner: Option<i64>,
    /// A resumed object dataset: the keys already there (one `LIST`), each a whole object,
    /// since an upload that did not complete leaves none.
    present: std::collections::HashSet<String>,
    /// Every rank's ids, before any has said anything: `[0, files)`, a resume's unfinished
    /// ranges, a grow's or shrink's ids from the first that changes (an array file is id 0).
    work: Vec<(i64, i64)>,
    /// Claimed and done from the top down (a shrink).
    desc: bool,
    /// Rank 0 (and the one host): the queue of runs every rank claims from.
    queue: Option<Mutex<Queue>>,
    /// Rank 0: the manifest last written, and the claim mark, under one lock.
    book: Mutex<Book>,
    /// Files and bytes written and flushed before this datagen, by the one it resumes (a grow
    /// or shrink: what the corpus keeps).
    before: (u64, u64),
    /// A grow or shrink: the count the corpus had.
    from: Option<i64>,
    /// What every manifest's progress carries: `from`, `settle`.
    carry: serde_json::Map<String, Value>,
    /// The count whose names may be on storage past this one's (a shrink's old count, a
    /// settle's asked one): an id's names under it that this count lacks are removed.
    remove: Option<DsMeta<'a>>,
    track: Track,
}

/// Rank 0's queue of one dataset's runs (`DESIGN_REVIEW.md` §3.70): handed out in order, to
/// its own writers directly and to the other ranks' over the coordinator.
struct Queue {
    runs: Vec<(i64, i64)>,
    next: usize,
    /// The runs granted to each rank, in the order granted (to another rank, sent).
    grants: Vec<Vec<(i64, i64)>>,
    /// The claim boundary before the first run: the start of the work, or a resume's
    /// manifest's; none when that manifest never recorded one, and then none is recorded.
    claimed0: Option<i64>,
    /// How many runs past a grant the mark moves when the grant reaches beyond it.
    step: usize,
}

#[derive(Default)]
struct Book {
    /// The manifest as rank 0 last wrote it, which a mark is written into.
    last: Option<Manifest>,
    /// `claimed_to`: no run reaching past it has been handed out, by this datagen or an
    /// earlier one of this corpus.
    mark: Option<i64>,
}

/// What this rank has written of one dataset, kept by its writers and read every
/// `--progress-secs` (`DESIGN_REVIEW.md` §3.70). A writer does its run in id order (from the
/// top down in a shrink), so what is not done is, per writer, the rest of its run.
#[derive(Default)]
struct Track {
    files: AtomicU64,
    bytes: AtomicU64,
    claims: Mutex<Claims>,
    /// Of `files` and `bytes`: the objects a resume found already in the store, which count
    /// toward the dataset's totals and not toward what this datagen wrote.
    found_files: AtomicU64,
    found_bytes: AtomicU64,
    /// The array file is written.
    whole: AtomicBool,
    /// Names removed (a shrink, a settle).
    removed: AtomicU64,
}

#[derive(Default)]
struct Claims {
    /// Each writer's run `[a, b)` and where it is (the first id not yet done; from the top
    /// down, the end of those not yet done), which the writer moves under its lock together
    /// with the counts, so a window reads a file as done and counted, or as neither.
    slots: Vec<(i64, i64, Arc<Mutex<i64>>)>,
    /// Runs this rank took from the queue itself (rank 0, the one host).
    taken: u64,
}

/// Objects removed in one request.
const DELETE_BATCH: usize = 1000;

impl Plan<'_, '_> {
    /// An object already in the store, which a resume does not send again but counts, as a
    /// directory resume counts the file it writes again.
    fn present(&self, opts: &DatagenOpts, path: &str) -> bool {
        !self.present.is_empty() && opts.endpoints.object(path).is_some_and(|(_, rest)| self.present.contains(&rest))
    }

    /// `x` lies past `than` in the order of the work.
    fn past(&self, x: i64, than: i64) -> bool {
        if self.desc { x < than } else { x > than }
    }

    /// The far end of a run in the order of the work.
    fn far(&self, (a, b): (i64, i64)) -> i64 {
        if self.desc { a } else { b }
    }

    /// The claim boundary: every id of the work on the near side of it has been handed out.
    fn claimed(&self, q: &Queue) -> Option<i64> {
        let c0 = q.claimed0?;
        Some(match q.next.checked_sub(1) {
            None => c0,
            Some(i) if self.past(self.far(q.runs[i]), c0) => self.far(q.runs[i]),
            Some(_) => c0,
        })
    }

    /// Rank 0: hand rank `rank` the next run (none when the queue is empty or the datagen is
    /// stopping), first moving the claim mark `step` runs past it, and writing it into the
    /// manifest, when the run reaches beyond the mark.
    fn take(&self, rank: i64, halted: bool) -> Result<Option<(i64, i64)>> {
        let mut q = self.queue.as_ref().expect("rank 0's queue").lock().unwrap();
        if halted || q.next >= q.runs.len() {
            return Ok(None);
        }
        let run = q.runs[q.next];
        let mut b = self.book.lock().unwrap();
        if b.mark.map_or(true, |m| self.past(self.far(run), m)) {
            let to = self.far(q.runs[(q.next + q.step).min(q.runs.len() - 1)]);
            b.mark = Some(to);
            if let Some(m) = b.last.as_mut() {
                m.progress["claimed_to"] = json!(to);
                m.progress["updated"] = json!(now_f64());
                write_manifest(self, m)?;
            }
        }
        drop(b);
        q.next += 1;
        q.grants[rank as usize].push(run);
        Ok(Some(run))
    }

    /// Writer `w`'s next run, set in its slot: from the queue (rank 0, the one host), or from
    /// rank 0 over the coordinator with one claim ahead, so the round trip overlaps a run.
    fn claim(&self, w: usize, ahead: &mut Option<u64>, tcp: Option<&Tcp>, mode: &Mode) -> Result<Option<(i64, i64, Arc<Mutex<i64>>)>> {
        let set = |c: &mut Claims, (a, b): (i64, i64)| {
            let cursor = c.slots[w].2.clone();
            *cursor.lock().unwrap() = if self.desc { b } else { a };
            c.slots[w].0 = a;
            c.slots[w].1 = b;
            (a, b, cursor)
        };
        if self.queue.is_some() {
            let mut c = self.track.claims.lock().unwrap();
            let Some(run) = self.take(0, mode.halted())? else { return Ok(None) };
            c.taken += 1;
            return Ok(Some(set(&mut c, run)));
        }
        let Some(t) = tcp else { bail!("dataset `{}`: no queue and no coordinator", self.name) };
        let ticket = match ahead.take() {
            Some(k) => k,
            None => t.claim(self.name)?,
        };
        t.wait_grant(ticket)?;
        let got = {
            let mut c = self.track.claims.lock().unwrap();
            t.take_grant(ticket).flatten().map(|run| set(&mut c, run))
        };
        if got.is_some() && !mode.halted() {
            *ahead = Some(t.claim(self.name)?);
        }
        Ok(got)
    }

    /// This rank's note on the dataset: the ids not yet done of the runs it holds (an array
    /// file is id 0), how many runs it has received (`seen`: rank 0 counts the ones it sent
    /// after them as not done), and the files and bytes of the rest (and of those bytes, those
    /// of the objects a resume found already in the store), read together: every writer's lock is
    /// held while its cursor and the counts are read, and an array file counts once whole.
    fn note(&self, tcp: Option<&Tcp>) -> Value {
        let mut v = Vec::new();
        let mut seen = 0u64;
        let counts = || (self.track.files.load(Ordering::Relaxed), self.track.bytes.load(Ordering::Relaxed), self.track.found_bytes.load(Ordering::Relaxed));
        let (files, bytes, found) = match self.share {
            Share::Other { .. } => (0, 0, 0),
            Share::Array => {
                if self.track.whole.load(Ordering::Acquire) {
                    counts()
                } else {
                    v.push((0, 1));
                    (0, 0, 0)
                }
            }
            Share::Files => {
                let c = self.track.claims.lock().unwrap();
                let held: Vec<_> = c.slots.iter().map(|(a, b, cursor)| (*a, *b, cursor.lock().unwrap())).collect();
                for (a, b, at) in &held {
                    v.push(if self.desc { (*a, **at) } else { (**at, *b) });
                }
                seen = match (tcp, &self.queue) {
                    (Some(t), None) => {
                        let (n, untaken) = t.grant_book(self.name);
                        v.extend(untaken);
                        n
                    }
                    _ => c.taken,
                };
                counts()
            }
        };
        json!({"unfinished": ranges_json(&merge_ranges(v)), "seen": seen, "files": files, "bytes": bytes, "found_bytes": found})
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

/// What rank `r` is to write of an `array` dataset, before it has said anything.
fn share_of(p: &Plan<'_, '_>, r: i64) -> Vec<(i64, i64)> {
    if p.owner == Some(r) {
        p.work.clone()
    } else {
        Vec::new()
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
    if id >= meta.files() {
        return Ok(Vec::new());
    }
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

struct Written {
    /// Written (sent) by this datagen.
    files: u64,
    bytes: u64,
    /// Found already in the store by a resume, and not sent again.
    found: (u64, u64),
    started: u64,
    finished: u64,
    elapsed: Duration,
}

/// Write this rank's share of one dataset: a `files` dataset's runs as they are claimed, each
/// id in order removing the names the count does not have (`DESIGN_REVIEW.md` §3.70: a
/// shrink's, a settle's; a directory that leaves empty goes too, so the tree is a fresh
/// datagen's) and writing the file when the count has it. A name already gone is no error,
/// so a resume removes again; objects go in batches of `DELETE_BATCH`, sent at the end of a
/// run, which counts as done only then. `tcp`: another rank's coordinator, which it claims
/// through.
fn write_share(plan: &Plan<'_, '_>, opts: &DatagenOpts, spec: &PayloadSpec, mode: &Mode, tcp: Option<&Tcp>) -> Result<Written> {
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
    let fail = |e: anyhow::Error| {
        errors.lock().unwrap().push(e);
        failed.store(true, Ordering::Relaxed);
    };
    let (files0, bytes0) = (track.files.load(Ordering::Relaxed), track.bytes.load(Ordering::Relaxed));
    let found0 = (track.found_files.load(Ordering::Relaxed), track.found_bytes.load(Ordering::Relaxed));
    match (&plan.share, meta) {
        (Share::Other { .. }, _) => {}
        (Share::Files, DsMeta::Files { spf, chunk, count, .. }) => {
            let total: i64 = plan.work.iter().map(|(a, b)| b - a).sum();
            let nthreads = writers.min(total.max(0) as usize);
            if nthreads > 0 {
                track.claims.lock().unwrap().slots = (0..nthreads).map(|_| (0, 0, Arc::new(Mutex::new(0)))).collect();
                // removals into a bucket are batched: the run counts once its batch is sent
                let hold = plan.remove.is_some() && plan.object.is_some();
                std::thread::scope(|s| {
                    for w in 0..nthreads {
                        let fail = &fail;
                        let failed = &failed;
                        s.spawn(move || {
                            let mut filler = Filler::new(spec.compress);
                            let mut buf = Aligned::new();
                            let mut ahead = None;
                            loop {
                                if failed.load(Ordering::Relaxed) || mode.halted() {
                                    break;
                                }
                                let (a, b, cursor) = match plan.claim(w, &mut ahead, tcp, mode) {
                                    Ok(Some(x)) => x,
                                    Ok(None) => break,
                                    Err(e) => {
                                        fail(e);
                                        break;
                                    }
                                };
                                let mut batch: Vec<String> = Vec::new();
                                let mut held = (0u64, 0u64);
                                let mut held_found = (0u64, 0u64);
                                let mut last_dir: Option<PathBuf> = None;
                                let mut whole = true;
                                let ids: Box<dyn Iterator<Item = i64>> = if plan.desc { Box::new((a..b).rev()) } else { Box::new(a..b) };
                                for file in ids {
                                    if mode.halted() {
                                        whole = false;
                                        break;
                                    }
                                    // this file's objects and bytes, counted when it is whole, and of
                                    // them those found already in the store
                                    let mut got = (0u64, 0u64);
                                    let mut found = (0u64, 0u64);
                                    let r: Result<()> = (|| {
                                        if let Some(rm) = &plan.remove {
                                            let keep = names(meta, file)?;
                                            for path in names(rm, file)?.into_iter().filter(|n| !keep.contains(n)) {
                                                if let Some((_, rest)) = opts.endpoints.object(&path) {
                                                    batch.push(rest);
                                                    continue;
                                                }
                                                let full = opts.endpoints.path(&opts.root, &path);
                                                match std::fs::remove_file(&full) {
                                                    Ok(()) => {
                                                        track.removed.fetch_add(1, Ordering::Relaxed);
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
                                        if file * spf >= *count {
                                            return Ok(());
                                        }
                                        let size = meta.file_size(file)? as u64;
                                        let seed_of = |blk: u64| payload::file_block_seed(seed, file as u64, blk, dedupe);
                                        if let Some(c) = &plan.container {
                                            let path = meta.file_path(file, None)?.to_string();
                                            // an object already there counts toward the totals (a
                                            // directory resume writes it again), so they hold every file
                                            // once, and not toward what this datagen wrote
                                            if plan.present(opts, &path) {
                                                got = (1, size);
                                                found = got;
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
                                                got = (got.0 + 1, got.1 + len);
                                                found = (found.0 + 1, found.1 + len);
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
                                        // a file left because the datagen is stopping is not done, and no failure
                                        if !e.is::<Halted>() {
                                            fail(e);
                                        }
                                        whole = false;
                                        break;
                                    }
                                    if hold {
                                        held = (held.0 + got.0, held.1 + got.1);
                                        held_found = (held_found.0 + found.0, held_found.1 + found.1);
                                        continue;
                                    }
                                    let mut at = cursor.lock().unwrap();
                                    track.found_files.fetch_add(found.0, Ordering::Relaxed);
                                    track.found_bytes.fetch_add(found.1, Ordering::Relaxed);
                                    track.files.fetch_add(got.0, Ordering::Relaxed);
                                    track.bytes.fetch_add(got.1, Ordering::Relaxed);
                                    *at = if plan.desc { file } else { file + 1 };
                                }
                                if let Some(d) = last_dir {
                                    prune(&d, &plan.root);
                                }
                                if hold && whole {
                                    let sent = (|| -> Result<()> {
                                        if let Some((store, _)) = &plan.object {
                                            for keys in batch.chunks(DELETE_BATCH) {
                                                store.delete_all(keys)?;
                                                track.removed.fetch_add(keys.len() as u64, Ordering::Relaxed);
                                            }
                                        }
                                        Ok(())
                                    })();
                                    if let Err(e) = sent {
                                        fail(e);
                                        break;
                                    }
                                    let mut at = cursor.lock().unwrap();
                                    track.found_files.fetch_add(held_found.0, Ordering::Relaxed);
                                    track.found_bytes.fetch_add(held_found.1, Ordering::Relaxed);
                                    track.files.fetch_add(held.0, Ordering::Relaxed);
                                    track.bytes.fetch_add(held.1, Ordering::Relaxed);
                                    *at = if plan.desc { a } else { b };
                                }
                            }
                        });
                    }
                });
            }
        }
        (Share::Array, DsMeta::Array { count, slot, .. }) => {
            let path = meta.file_path(0, None)?.to_string();
            let full = opts.endpoints.path(&opts.root, &path);
            let size = meta.file_size(0)? as u64;
            let (count, slot) = (*count as u64, *slot as u64);
            let (f, direct) = create(&full, mode)?;
            // the slots' tails, and a last slot's that is empty, are holes
            f.set_len(size).with_context(|| format!("sizing {}", full.display()))?;
            let pieces = (size + REGIONS_PIECE - 1) / REGIONS_PIECE;
            let next = AtomicU64::new(0);
            let nthreads = writers.min(pieces.max(1) as usize);
            let seed_of = |blk: u64| payload::array_block_seed(seed, blk, dedupe);
            let left = AtomicBool::new(false);
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
                            match write_slots(&f, direct, mode, meta, slot, count, &seed_of, at, at + len, &mut filler, &mut buf, &full) {
                                Ok(n) => {
                                    track.bytes.fetch_add(n, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    if e.is::<Halted>() {
                                        left.store(true, Ordering::Relaxed);
                                    } else {
                                        fail(e);
                                    }
                                    break;
                                }
                            }
                        }
                    });
                }
            });
            if !failed.load(Ordering::Relaxed) && !left.load(Ordering::Relaxed) {
                track.files.fetch_add(1, Ordering::Relaxed);
                track.whole.store(true, Ordering::Release);
            }
        }
        _ => bail!("dataset `{name}`: the plan does not match the dataset kind"),
    }
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e.context(format!("dataset `{name}`")));
    }
    let found = (track.found_files.load(Ordering::Relaxed) - found0.0, track.found_bytes.load(Ordering::Relaxed) - found0.1);
    Ok(Written {
        files: track.files.load(Ordering::Relaxed) - files0 - found.0,
        bytes: track.bytes.load(Ordering::Relaxed) - bytes0 - found.1,
        found,
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
    let mut v = json!({"rank": rank, "host": host, "files_written": w.files, "bytes_written": w.bytes, "started": w.started, "finished": w.finished});
    // a resume's objects found already in the store, which only the totals count
    if w.found.0 > 0 {
        v["files_found"] = json!(w.found.0);
        v["bytes_found"] = json!(w.found.1);
    }
    v
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

/// This rank's progress document: per dataset, its note (`Plan::note`). `tcp`: another
/// rank's coordinator, which holds the runs granted it that no writer has taken.
fn progress_doc(plans: &[Plan<'_, '_>], tcp: Option<&Tcp>) -> Value {
    json!({"datasets": plans.iter().map(|p| (p.name.to_string(), p.note(tcp))).collect::<serde_json::Map<_, _>>()})
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

/// A manifest's progress record (`schema/README.md` §6): what is not yet written and flushed,
/// the counts, when, the window, what a change carries (`from`, `settle`), the claim boundary
/// and mark, and, from a datagen that is exiting, `stopped`.
fn progress_json(p: &Plan<'_, '_>, unfinished: &[(i64, i64)], counts: (u64, u64), claimed: Option<i64>, mark: Option<i64>, opts: &DatagenOpts, stopped: bool) -> Value {
    let mut v = json!({"unfinished": ranges_json(unfinished), "files_written": counts.0, "bytes_written": counts.1, "updated": now_f64(), "progress_secs": opts.progress.as_secs_f64()});
    for (k, x) in &p.carry {
        v[k] = x.clone();
    }
    if let Some(c) = claimed {
        v["claimed"] = json!(c);
    }
    if let Some(m) = mark {
        v["claimed_to"] = json!(m);
    }
    if stopped {
        v["stopped"] = json!(now_f64());
    }
    v
}

/// The progress window: note what is finished, flush, and tell rank 0 (`DESIGN_REVIEW.md`
/// §3.70); rank 0 merges every rank's latest note with its queue (the runs not handed out,
/// and those handed to a rank after the last it had seen when it took its note), rewrites
/// each manifest once, and prints one line. A rank not heard from yet counts as having
/// written nothing.
struct Ticker<'p, 'm, 'a> {
    plans: &'p [Plan<'m, 'a>],
    manifests: &'p [Manifest],
    opts: &'p DatagenOpts,
    hosts: Option<&'p Hosts<'p>>,
    t0: Instant,
}

impl Ticker<'_, '_, '_> {
    /// `last`: this datagen is exiting (stopped, or failed), so the manifests say no datagen
    /// is writing the corpus (`progress.stopped`).
    fn tick(&self, log: &Mutex<&mut (dyn Write + Send)>, last: bool) -> Result<()> {
        let rank = self.opts.rank;
        let remote = self.hosts.filter(|_| rank != 0).map(|h| h.tcp);
        let mine = progress_doc(self.plans, remote);
        flush(self.opts)?;
        let mut docs = BTreeMap::new();
        if let Some(h) = self.hosts {
            h.tcp.progress(&mine)?;
            match h.server {
                Some(server) => docs = server.progress(),
                None => return Ok(()),
            }
        }
        docs.insert(rank, mine);
        let ranks = self.opts.ranks.max(1);
        let (mut total, mut left, mut bytes) = (0i64, 0i64, 0u64);
        for (p, base) in self.plans.iter().zip(self.manifests) {
            let mut unfinished = Vec::new();
            let (mut files, mut b, mut found_b) = (0u64, 0u64, 0u64);
            let doc = |r: i64| docs.get(&r).map(|d| &d["datasets"][p.name]);
            for r in 0..ranks {
                match doc(r) {
                    Some(d) => {
                        unfinished.extend(ranges_of(&d["unfinished"]));
                        files += d["files"].as_u64().unwrap_or(0);
                        b += d["bytes"].as_u64().unwrap_or(0);
                        found_b += d["found_bytes"].as_u64().unwrap_or(0);
                    }
                    None if p.queue.is_none() => unfinished.extend(share_of(p, r)),
                    None => {}
                }
            }
            let mut claimed = None;
            if let Some(q) = &p.queue {
                let q = q.lock().unwrap();
                unfinished.extend_from_slice(&q.runs[q.next..]);
                for r in 0..ranks {
                    let seen = doc(r).and_then(|d| d["seen"].as_u64()).unwrap_or(0) as usize;
                    let g = &q.grants[r as usize];
                    unfinished.extend_from_slice(&g[seen.min(g.len())..]);
                }
                claimed = p.claimed(&q);
            }
            let unfinished = merge_ranges(unfinished);
            let (files, b) = (files + p.before.0, b + p.before.1);
            // a shrink's ids reach past its count
            total += match p.meta {
                DsMeta::Files { .. } => p.meta.files().max(p.work.last().map_or(0, |r| r.1)),
                DsMeta::Array { .. } => 1,
            };
            left += unfinished.iter().map(|(a, b)| b - a).sum::<i64>();
            // the rate's bytes are this datagen's: not a resume's objects found in the store
            bytes += (b - p.before.1).saturating_sub(found_b);
            let mut book = p.book.lock().unwrap();
            let mut m = base.clone();
            m.progress = progress_json(p, &unfinished, (files, b), claimed, book.mark, self.opts, last);
            write_manifest(p, &m)?;
            book.last = Some(m);
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

/// The manifest `--settle` starts from (`DESIGN_REVIEW.md` §3.70): an unfinished corpus
/// brought to the count it reached, claiming nothing new. `meta` is its dataset at the count
/// the manifest names (the one asked for). The count is the end of the runs handed out (from
/// the top down in a shrink: its start), whole files, never past the asked count nor short of
/// the one a grow or shrink began from; what is left to do is what was handed out and is not
/// done, and the band between the boundary and the mark, where a datagen that died after its
/// last window may have written or removed (strays past the count are removed, ids below it
/// written again). The result is `settling` (`progress.settle`), finished as a resume is.
pub fn settle_manifest(m: &Manifest, meta: &DsMeta<'_>) -> Result<Manifest> {
    let p = &m.progress;
    if !p["settle"].is_null() {
        // a settle that did not finish: its count is decided
        return Ok(m.clone());
    }
    let DsMeta::Files { spf, .. } = meta else { bail!("an `array` dataset is one file: `--resume` finishes it") };
    let Some(claimed) = p["claimed"].as_i64() else { bail!("the manifest records no claim boundary (an earlier aeiou wrote it): `--resume` finishes it") };
    let mark = p["claimed_to"].as_i64().unwrap_or(claimed);
    let n = meta.count();
    let from = p["from"].as_i64();
    let desc = m.state == payload::STATE_SHRINKING;
    let s = if desc { n.max(from.unwrap_or(n).min(claimed * spf)) } else { from.unwrap_or(0).max(n.min(claimed * spf)) };
    if s == 0 {
        bail!("nothing was handed out, so a settled corpus would be empty: remove it instead");
    }
    let near = |(a, b): (i64, i64)| if desc { (a.max(claimed), b) } else { (a, b.min(claimed)) };
    let mut t: Vec<(i64, i64)> = m.unfinished().into_iter().map(near).collect();
    t.push(if desc { (mark, claimed) } else { (claimed, mark) });
    let t = merge_ranges(t);
    let inside = |id: i64| t.iter().any(|(a, b)| (*a..*b).contains(&id));
    let target = meta.with_count(s);
    let (mut files, mut bytes) = (p["files_written"].as_u64().unwrap_or(0), p["bytes_written"].as_u64().unwrap_or(0));
    // ids the corpus keeps that no window counted: a shrink settled above its count keeps old
    // files from the new short last one up; a grow settled at its old count keeps the old
    // short last file
    let kept = match from {
        Some(_) if desc && s > n => (n / spf, target.files()),
        Some(f) if !desc && s == f => (f / spf, target.files()),
        _ => (0, 0),
    };
    for id in kept.0..kept.1 {
        if !inside(id) {
            files += objects_of(&target, id)?;
            bytes += target.file_size(id)? as u64;
        }
    }
    let mut r = m.clone();
    let Some(def) = r.dataset.as_object_mut().filter(|d| d["layout"] != "array") else { bail!("an `array` is one file: `--resume` finishes it") };
    def.insert("count".into(), json!(s));
    let mut progress = json!({"unfinished": ranges_json(&t), "files_written": files, "bytes_written": bytes, "settle": {"asked": n, "count": s}, "updated": p["updated"], "progress_secs": p["progress_secs"]});
    if let Some(f) = from {
        progress["from"] = json!(f);
    }
    r.progress = progress;
    Ok(r)
}

/// Above this share of a file system's free space, `check_space` warns.
pub const SPACE_WARN: f64 = 0.9;

/// The capacity check before a datagen writes anything (`DESIGN_REVIEW.md` §3.74 item 4):
/// the bytes still to write for each dataset (its projection, `project.rs`, less what a
/// resumed or resized manifest says is already there), summed per file system of the
/// directory roots, against what `statvfs` reports free to an unprivileged writer there.
/// Refused above it unless `ignore`; a line of warning above `SPACE_WARN` of it, and for an
/// object root, which has no free space to ask. File-system metadata, and what a system under
/// test saves by deduplicating or compressing the payload, are not counted. Returns the
/// lines to print.
pub fn check_space(loaded: &crate::Loaded, model: &Model<'_>, opts: &DatagenOpts, ignore: bool) -> Result<Vec<String>> {
    use crate::dryrun::human_bytes;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut lines = Vec::new();
    // by file system: the roots on it, the bytes to write, its free bytes
    let mut fs: Vec<(u64, Vec<String>, u64, u64)> = Vec::new();
    for meta in model.datasets.iter() {
        let name = meta.name();
        if !opts.datasets.is_empty() && !opts.datasets.iter().any(|d| d == name) {
            continue;
        }
        let rel = payload::dataset_root(&loaded.ast, name)?;
        let p = crate::project::project(meta, threads)?;
        let have = match &opts.resume {
            None => 0,
            Some(m) if opts.resize.is_none() => m.progress["bytes_written"].as_u64().unwrap_or(0),
            Some(m) => m.provenance["bytes_written"].as_u64().unwrap_or(0),
        };
        let need = p.bytes.saturating_sub(have);
        if need == 0 {
            continue;
        }
        if let Some((store, rest)) = opts.endpoints.object(&rel) {
            lines.push(format!("space: dataset `{name}`: {} to write to {}; an object store reports no free space, not checked", human_bytes(need), store.show(&rest)));
            continue;
        }
        let root = opts.endpoints.path(&opts.root, &rel);
        let (dev, free) = free_space(&root)?;
        match fs.iter_mut().find(|f| f.0 == dev) {
            Some(f) => {
                f.1.push(name.to_string());
                f.2 += need;
            }
            None => fs.push((dev, vec![name.to_string()], need, free)),
        }
    }
    for (_, names, need, free) in &fs {
        let what = format!("dataset(s) {}: {} to write, {} free", names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", "), human_bytes(*need), human_bytes(*free));
        if need > free {
            if !ignore {
                bail!("space: {what}: the corpus does not fit (--ignore-space writes anyway, for a system that stores less than it is sent)");
            }
            lines.push(format!("space: IGNORED: {what}"));
        } else if *need as f64 > SPACE_WARN * *free as f64 {
            lines.push(format!("space: {what}: above {:.0}% of the free space", SPACE_WARN * 100.0));
        } else {
            lines.push(format!("space: {what}"));
        }
    }
    Ok(lines)
}

/// The device and the bytes `statvfs` reports available to an unprivileged writer at `path`,
/// or at its nearest existing ancestor (a dataset root datagen has yet to create).
fn free_space(path: &Path) -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let mut p = path;
    while !p.exists() {
        p = p.parent().ok_or_else(|| anyhow::anyhow!("{}: no existing directory above it", path.display()))?;
    }
    let dev = std::fs::metadata(p)?.dev();
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes())?;
    let mut v: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut v) } != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("statvfs {}", p.display()));
    }
    Ok((dev, v.f_bavail as u64 * v.f_frsize as u64))
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
    // a settle, begun now or resumed: the count asked for and the count settled at
    let settle = resume.map(|m| m.progress["settle"].clone()).filter(|v| !v.is_null());
    let desc = resume.is_some_and(|m| m.state == payload::STATE_SHRINKING);
    if let Some(m) = resume {
        if m.payload != spec {
            bail!("the corpus was written with the payload {} and this aeiou writes {}: a resume would mix them", payload_line(&m.payload), payload_line(&spec));
        }
        if model.datasets.len() != 1 {
            bail!("a resume finishes one dataset, not {}", model.datasets.len());
        }
    }

    // what each dataset is and where it goes, and the refusals those alone decide (a format
    // writer other than the one that wrote a resumed corpus, an `array` in an object store),
    // before the space, which asks the file system: a datagen refused anyway is never told
    // to free space first
    let mut checked = Vec::new();
    for meta in model.datasets.iter() {
        let name = meta.name();
        if !opts.datasets.is_empty() && !opts.datasets.iter().any(|d| d == name) {
            continue;
        }
        let container = match loaded.ast.datasets.get(name) {
            Some(d) if d.format.is_some() => Some(container(name, &payload::resolved_dataset(&loaded.doc, name, cfg)?)?),
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
        if object.is_some() && matches!(meta, DsMeta::Array { .. }) {
            bail!("dataset `{name}`: an `array` dataset is a sparse layout written at offsets and has no object form; place it in a directory");
        }
        // a geometry the format writer cannot frame (a Parquet page past 2 GiB, a page header
        // outside the layout's band): refused here, not at the file that shows it
        #[cfg(feature = "formats")]
        if let Some(c) = &container {
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            c.writer.check(meta, threads).with_context(|| format!("dataset `{name}`"))?;
        }
        checked.push((meta, name, container, root, object));
    }
    // the space, on rank 0 (the roots are the shared storage under test): a refusal there
    // reaches every rank as the empty-root checks below do
    if rank == 0 {
        for line in check_space(loaded, model, opts, opts.ignore_space)? {
            writeln!(log, "{line}")?;
        }
    }
    // the plan: this rank's share of each; every rank checks the roots are empty before the
    // gate, so no rank sees another's files (rank 0's manifest, written before the gate,
    // aside); a resume, whose format writer was checked above, does not
    let mut plans = Vec::new();
    let mut array_index = 0usize;
    for (meta, name, container, root, object) in checked {
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
        let count = meta.count();
        // the count whose names may be on storage past this one's
        let reach = [from, settle.as_ref().and_then(|s| s["asked"].as_i64())].into_iter().flatten().fold(count, i64::max);
        let (work, before) = match resume {
            Some(m) if opts.resize.is_none() => (merge_ranges(m.unfinished()), (m.progress["files_written"].as_u64().unwrap_or(0), m.progress["bytes_written"].as_u64().unwrap_or(0))),
            // a grow or shrink begun now: the ids from the first whose bytes change (the old last
            // file when it was short; a shrink's new one) to the larger count's last, and the
            // corpus keeps the rest
            Some(m) => {
                let (Some(c0), DsMeta::Files { spf, .. }) = (from, meta) else { bail!("dataset `{name}`: only a `files` dataset grows or shrinks") };
                let old = meta.with_count(c0);
                let w0 = c0.min(count) / spf;
                let (mut files, mut bytes) = (0u64, 0u64);
                for id in w0..old.files() {
                    files += objects_of(&old, id)?;
                    bytes += old.file_size(id)? as u64;
                }
                let kept = (m.provenance["files_written"].as_u64().unwrap_or(0).saturating_sub(files), m.provenance["bytes_written"].as_u64().unwrap_or(0).saturating_sub(bytes));
                let end = old.files().max(meta.files());
                (if w0 < end { vec![(w0, end)] } else { Vec::new() }, kept)
            }
            None => (vec![(0, if let DsMeta::Files { .. } = meta { meta.files() } else { 1 })], (0, 0)),
        };
        if let DsMeta::Files { spf, .. } = meta {
            if from.is_some() || reach > count {
                // a short last file in the bucket under one count is not the other's: write it again
                for c in [from.unwrap_or(count), count] {
                    if c % spf != 0 {
                        for n in names(meta, c / spf)? {
                            if let Some((_, rest)) = opts.endpoints.object(&n) {
                                present.remove(&rest);
                            }
                        }
                    }
                }
            }
        }
        let remove = (reach > count).then(|| meta.with_count(reach));
        let (share, owner, queue) = match meta {
            DsMeta::Files { dirs, .. } => {
                // rank 0 holds the queue; the boundary starts at the work's start, or where a
                // resumed manifest left it
                let queue = (rank == 0).then(|| {
                    let slots = ranks as usize * opts.writers.max(1);
                    let claimed0 = match resume {
                        Some(m) if opts.resize.is_none() => m.progress["claimed"].as_i64(),
                        _ => Some(if desc { work.last().map(|r| r.1).unwrap_or(0) } else { work.first().map(|r| r.0).unwrap_or(0) }),
                    };
                    Mutex::new(Queue { runs: runs_of(&work, dirs, slots, desc), next: 0, grants: vec![Vec::new(); ranks as usize], claimed0, step: MARK_BANDS * slots })
                });
                (Share::Files, None, queue)
            }
            DsMeta::Array { .. } => {
                let owner = array_owner(array_index, ranks);
                array_index += 1;
                // a resume whose array file was finished has nothing to write
                (if owner == rank && !work.is_empty() { Share::Array } else { Share::Other { owner } }, Some(owner), None)
            }
        };
        let mark = match resume {
            Some(m) if opts.resize.is_none() => m.progress["claimed_to"].as_i64(),
            _ => None,
        };
        let mut carry = serde_json::Map::new();
        if let Some(f) = from {
            carry.insert("from".into(), json!(f));
        }
        if let Some(s) = &settle {
            carry.insert("settle".into(), s.clone());
        }
        plans.push(Plan { meta, name, root, object, container, share, owner, present, work, desc, queue, book: Mutex::new(Book { last: None, mark }), before, from, carry, remove, track: Track::default() });
    }
    for p in &plans {
        let w = p.work.iter().map(|(a, b)| b - a).sum::<i64>();
        if let (Some(c0), true, DsMeta::Files { spf, .. }) = (p.from, opts.resize.is_some(), p.meta) {
            let n = p.meta.count();
            if p.desc {
                let again = if n % spf != 0 { format!(", and file {} written again", n / spf) } else { String::new() };
                writeln!(log, "dataset {}: count {c0} -> {n}: {w} id(s) from the top down, the names past the new count removed{again}", p.name)?;
            } else {
                writeln!(log, "dataset {}: count {c0} -> {n}: {w} file(s) to write", p.name)?;
            }
        }
        match &p.share {
            Share::Files => {
                let what = if resume.is_some() && opts.resize.is_none() { format!("{w} unfinished id(s) of {}", p.meta.files()) } else { format!("{w} id(s) of {}", p.meta.files()) };
                let order = if p.desc { ", from the top down" } else { "" };
                match p.queue.as_ref().map(|q| q.lock().unwrap().runs.len()) {
                    Some(n) if ranks > 1 => writeln!(log, "dataset {}: {what}, in {n} run(s) handed out by this rank in order{order}", p.name)?,
                    Some(n) => writeln!(log, "dataset {}: {what}, in {n} run(s){order}", p.name)?,
                    None => writeln!(log, "dataset {}: {what}, in runs claimed from rank 0", p.name)?,
                }
            }
            Share::Array => writeln!(log, "dataset {}: the array file, written by this rank", p.name)?,
            Share::Other { owner } => writeln!(log, "dataset {}: the array file, written by rank {owner}", p.name)?,
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
                // a grow, shrink, or settle changes the short last file's header alone: those both have must agree
                let (now, was) = (&c.headers["sha256"], &m.provenance["headers"]["sha256"]);
                let differ = match opts.begun() {
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
    // the one there, which already says so (no rank but 0 reads it: it came with `Welcome`)
    let started = now_secs();
    let mut manifests = Vec::new();
    for p in &plans {
        let claimed = p.queue.as_ref().and_then(|q| p.claimed(&q.lock().unwrap()));
        let mark = p.book.lock().unwrap().mark;
        let m = match resume {
            Some(m) => {
                let mut m = m.clone();
                if opts.begun().is_some() {
                    // the new count's manifest, before anything is removed or written
                    m.progress = progress_json(p, &p.work, p.before, claimed, mark, opts, false);
                    if let Some(c) = p.container.as_ref().filter(|c| !c.headers.is_null()) {
                        m.provenance["headers"] = c.headers.clone();
                    }
                    if rank == 0 {
                        write_manifest(p, &m)?;
                    }
                }
                m
            }
            None => {
                let mut m = base.manifest(loaded, cfg, &spec, p, ranks, started)?;
                m.progress = progress_json(p, &p.work, (0, 0), claimed, mark, opts, false);
                if rank == 0 {
                    write_manifest(p, &m)?;
                }
                m
            }
        };
        p.book.lock().unwrap().last = Some(m.clone());
        manifests.push(m);
    }

    // the gate: no rank writes before every rank has checked; when it does not open, nothing
    // was written, and rank 0 takes back its manifests (never a resumed corpus's)
    if let Some(h) = hosts {
        let (_t0, names, shared) = match h.tcp.ready_shared() {
            Ok(x) => x,
            Err(e) => {
                if let (0, Some(original)) = (rank, opts.begun()) {
                    // nothing was removed or written: the corpus is the one it was
                    for p in &plans {
                        let _ = write_manifest(p, original);
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

    // the writes, with a progress window beside them, a watch for Ctrl-C and for another host
    // stopping, and on rank 0 the answers to the other ranks' claims; then the metadata to
    // stable storage, a last note, and (several hosts) this rank's part to rank 0, whose
    // window runs until every part is in
    let mode = Mode::new();
    if opts.sigint {
        catch_sigint();
    }
    let remote = hosts.filter(|_| rank != 0).map(|h| h.tcp);
    let ticker = Ticker { plans: &plans, manifests: &manifests, opts, hosts, t0: Instant::now() };
    let log_m: Mutex<&mut (dyn Write + Send)> = Mutex::new(log);
    let (stop, stopped) = std::sync::mpsc::channel::<()>();
    let over = AtomicBool::new(false);
    let mut parts: BTreeMap<i64, Value> = BTreeMap::new();
    let mut written = Vec::new();
    let phase: Result<()> = std::thread::scope(|s| {
        let (ticker, log_m, mode, over, plans) = (&ticker, &log_m, &mode, &over, &plans);
        s.spawn(move || loop {
            match stopped.recv_timeout(opts.progress) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Err(e) = ticker.tick(log_m, false) {
                        let _ = writeln!(log_m.lock().unwrap(), "  progress: {e:#}");
                    }
                }
                _ => break,
            }
        });
        // Ctrl-C here reaches every host through rank 0; a halt or a failure anywhere stops
        // this host's writers
        s.spawn(move || {
            let mut sent = false;
            while !over.load(Ordering::Relaxed) {
                if opts.sigint && !sent && SIGINTS.load(Ordering::Relaxed) > 0 {
                    sent = true;
                    match hosts {
                        Some(h) => {
                            if h.tcp.halt().is_err() {
                                mode.halt.store(true, Ordering::Relaxed);
                            }
                        }
                        None => mode.halt.store(true, Ordering::Relaxed),
                    }
                }
                if hosts.is_some_and(|h| h.tcp.halted() || h.tcp.abort_reason().is_some()) {
                    mode.halt.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        if let Some(server) = hosts.and_then(|h| h.server) {
            s.spawn(move || {
                while !over.load(Ordering::Relaxed) {
                    let Some((r, ds, ticket)) = server.next_claim(Duration::from_millis(50)) else { continue };
                    let run = match plans.iter().find(|p| p.name == ds && p.queue.is_some()) {
                        Some(p) => p.take(r, mode.halted()),
                        None => Ok(None),
                    };
                    match run {
                        Ok(run) => server.grant(r, &ds, ticket, run),
                        Err(e) => {
                            server.stop(&format!("{e:#}"));
                            server.grant(r, &ds, ticket, None);
                        }
                    }
                }
            });
        }
        let r = (|| {
            for p in plans {
                let w = write_share(p, opts, &spec, mode, remote);
                for n in mode.take_notes() {
                    writeln!(log_m.lock().unwrap(), "  {n}")?;
                }
                if p.remove.is_some() {
                    writeln!(log_m.lock().unwrap(), "dataset {}: {} name(s) removed on this rank", p.name, p.track.removed.load(Ordering::Relaxed))?;
                }
                written.push(w?);
            }
            flush(opts)?;
            let mine = json!({
                "host": host,
                "direct": mode.direct(),
                "halted": mode.halted(),
                "datasets": plans.iter().zip(&written).map(|(p, w)| (p.name.to_string(), part_json(rank, &host, w))).collect::<serde_json::Map<_, _>>(),
            });
            match hosts {
                None => {
                    parts.insert(0, mine);
                }
                Some(h) => {
                    // the last note, so rank 0's manifests do not wait a window for this rank
                    h.tcp.progress(&progress_doc(plans, remote))?;
                    h.tcp.done(&mine)?;
                    if let Some(server) = h.server {
                        parts = server.gathered().context("gathering every rank's part")?;
                    }
                }
            }
            Ok(())
        })();
        over.store(true, Ordering::Relaxed);
        let _ = stop.send(());
        r
    });
    let mut sink = std::io::sink();
    let quiet: Mutex<&mut (dyn Write + Send)> = Mutex::new(&mut sink);
    if let Err(e) = phase {
        // what is finished, recorded before the error is reported
        if rank == 0 || hosts.is_some() {
            let _ = ticker.tick(&quiet, true);
        }
        return Err(e);
    }
    let log = log_m.into_inner().unwrap();
    if let Some(reason) = hosts.and_then(|h| h.tcp.abort_reason()).filter(|_| mode.halted()) {
        // stopped because another host failed (or the coordinator went away): a failure, so
        // its reason and exit 1, not a Ctrl-C's 130; rank 0's own gathering fails the same way
        return Err(anyhow::anyhow!("{reason}"));
    }
    if mode.halted() {
        // stopped: every rank has noted what it finished; rank 0 records it once more
        let mut msg = "stopped".to_string();
        if rank == 0 {
            ticker.tick(&quiet, true)?;
            msg = if ranks > 1 { format!("all {ranks} ranks stopped") } else { "stopped".into() };
            msg += "; the manifest records what is finished: `aeiou datagen --resume ROOT` finishes the corpus, `--settle ROOT` keeps what it reached";
            if let Some(server) = hosts.and_then(|h| h.server) {
                server.finish(false, 0, Some("stopped".into()));
            }
        } else if let Some(h) = hosts {
            let _ = h.tcp.result();
        }
        writeln!(log, "{msg}")?;
        log.flush()?;
        return Err(Stopped(msg).into());
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
        let (mut found_files, mut found_bytes) = (0u64, 0u64);
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
            found_files += d["files_found"].as_u64().unwrap_or(0);
            found_bytes += d["bytes_found"].as_u64().unwrap_or(0);
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
        let finished = finished.max(w.finished);
        if let Some(st) = &settle {
            // the corpus kept at the count it reached: the settle's record, and the change it
            // ended marked settled
            let this = json!({"state": base_m.state, "asked": st["asked"], "count": st["count"], "started": started, "finished": finished, "ranks": ranks, "hosts": host_parts, "direct": direct, "files_written": total_files, "bytes_written": total_bytes});
            match prov["settles"].as_array_mut() {
                Some(a) => a.push(this),
                None => prov["settles"] = json!([this]),
            }
            if let Some(last) = prov["resizes"].as_array_mut().and_then(|a| a.last_mut()).filter(|l| l["finished"].is_null()) {
                last["settled"] = st["count"].clone();
                last["finished"] = json!(finished);
            }
            prov["finished"] = json!(finished);
            prov["files_written"] = json!(total_files + p.before.0);
            prov["bytes_written"] = json!(total_bytes + p.before.1);
        } else if opts.resize.is_some() {
            // the change begun completes its record; the totals are what the corpus holds
            if let Some(last) = prov["resizes"].as_array_mut().and_then(|a| a.last_mut()) {
                for (k, v) in [("finished", json!(finished)), ("ranks", json!(ranks)), ("hosts", json!(host_parts)), ("direct", json!(direct)), ("files_written", json!(total_files)), ("bytes_written", json!(total_bytes))] {
                    last[k] = v;
                }
            }
            prov["finished"] = json!(finished);
            prov["files_written"] = json!(total_files + p.before.0);
            prov["bytes_written"] = json!(total_bytes + p.before.1);
        } else if resume.is_some() {
            // the datagen resumed keeps its record; each resume adds its own, what it wrote,
            // and the totals count every file written and flushed once (what the last window
            // recorded, then this datagen's, then the objects it found already in the store)
            let mut this = json!({"started": started, "finished": finished, "ranks": ranks, "hosts": host_parts, "direct": direct, "files_written": total_files, "bytes_written": total_bytes});
            if found_files > 0 {
                this["files_found"] = json!(found_files);
                this["bytes_found"] = json!(found_bytes);
            }
            match prov["resumes"].as_array_mut() {
                Some(a) => a.push(this),
                None => prov["resumes"] = json!([this]),
            }
            prov["finished"] = json!(finished);
            prov["files_written"] = json!(total_files + found_files + p.before.0);
            prov["bytes_written"] = json!(total_bytes + found_bytes + p.before.1);
        } else {
            prov["host"] = json!(parts.get(&0).and_then(|d| d["host"].as_str()).unwrap_or(&host));
            prov["hosts"] = json!(host_parts);
            prov["direct"] = json!(direct);
            prov["started"] = json!(started);
            prov["finished"] = json!(finished);
            prov["files_written"] = json!(total_files);
            prov["bytes_written"] = json!(total_bytes);
        }
        if wrote_manifest {
            let count = manifest.dataset["count"].as_i64().unwrap_or_else(|| p.meta.count());
            if let Err(e) = write_inventory(p, count, opts, &spec, &mode) {
                manifest_error = Some(format!("{e:#}"));
                break;
            }
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
            found: w.found.0,
            total_found: if wrote_manifest { found_files } else { w.found.0 },
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

    /// The live check against a manifest noted `ago` seconds back (negative: ahead of this
    /// clock) by a datagen with 30 s windows, or by one that stopped.
    #[test]
    fn a_manifest_younger_than_two_windows_is_live() {
        let check = |ago: f64, stopped: bool| {
            let payload = crate::payload::PayloadSpec::new(1, 1);
            let mut m = Manifest { manifest_version: 1, state: "writing".into(), dataset: json!({}), payload, format: None, provenance: Value::Null, progress: json!({"updated": now_f64() - ago, "progress_secs": 30.0}) };
            if stopped {
                m.progress["stopped"] = json!(now_f64());
            }
            check_not_live(&m, "--settle X").map_err(|e| e.to_string())
        };
        let e = check(5.0, false).unwrap_err();
        assert!(e.starts_with("--settle X: the manifest was updated 5") && e.contains(" s ago and its datagen notes progress every 30 s, so it may still be writing; stop it, or wait until 60 s have passed"), "{e}");
        assert!(check(40.0, false).is_err());
        let e = check(-2.5, false).unwrap_err();
        assert!(e.contains("the manifest was updated 2.5 s ahead of this host's clock and its datagen"), "{e}");
        check(90.0, false).unwrap();
        check(0.0, true).unwrap();
        check(-2.5, true).unwrap();
    }

    #[test]
    fn ranges_merge_in_order() {
        assert_eq!(merge_ranges(vec![(5, 9), (0, 2), (2, 3), (8, 12), (4, 4)]), vec![(0, 3), (5, 12)]);
        assert_eq!(merge_ranges(vec![]), vec![]);
    }
}
