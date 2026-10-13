//! Payload and the dataset manifest.
//!
//! **Payload.** The bytes at `(seed, unit, block)` are a pure function of that tuple
//! (`PROJECT_BRIEF.md` §5, *Data verification*; `DESIGN_REVIEW.md` §3.16–§3.17). Content is
//! cut into 1 MiB blocks; block `b` of unit `u` is the prefix of the 1 MiB block seeded
//! `labeled_key(seed, "payload", [u, b])` (`aeiou-positional/3`, 2026-10-11, `DESIGN_REVIEW.md`
//! §3.70): an SFC64 stream (`Sfc64`, numpy's bit generator, so the Python writer makes the
//! same bytes) laid out in 4 KiB pieces, each the next `⌈4096 / C⌉` bytes of the stream and
//! then `0x5A` to the end of the piece for compression ratio `C` (at `C = 1`, the stream as
//! is). Any prefix of a few KiB, and any compression unit of 4 KiB or more, has ratio `C`,
//! and no piece is all zeros (`/2` used `dgen-data`'s layout, the last `(C−1)/C` of the block
//! zero-filled, so a file below `1/C` MiB did not compress at all). Dedupe is by seed reuse
//! across units (`/2`, 2026-10-04; `/1` grouped by `id mod ceil(count / D)`, which made the
//! groups depend on the count, so that a prefix of the ids had a lower ratio and a dataset
//! could not grow; `DESIGN_REVIEW.md` §3.62):
//! - a `files` dataset: `u = id div D`, `b` = logical offset in the file ÷ 1 MiB (a chunked
//!   file is the logical file cut at `chunk`), so every run of `D` consecutive files carries
//!   the same bytes block for block, whatever the count;
//! - a `regions` dataset (one file): `u = 0`, `b = (offset ÷ 1 MiB) div D`, so every run of
//!   `D` consecutive blocks is one block repeated;
//! - a namespace object written by `aeiou run`: `seed = labeled_key(namespace seed, "object",
//!   [xxh3(path)])`, `u = 0`, `b` = offset ÷ 1 MiB, ratio from `--write-compress`, no dedupe.
//!
//! **Manifest.** `.aeiou-dataset.json` at each dataset root (`schema/README.md` §6): the
//! resolved dataset definition, the payload settings, and provenance; its id is the SHA-256 of
//! the canonical normative part.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::ast::{Ast, Dataset};
use crate::canon;
use crate::eval::{Config, Params};
use crate::pattern::Pattern;
use crate::rng::labeled_key;

pub const BLOCK: u64 = 1 << 20;
pub const WRAPPER: &str = "aeiou-positional/3";
pub const GENERATOR: &str = "sfc64";
pub const GENERATOR_VERSION: &str = "1";
/// The piece a compression ratio applies to, and its fill.
pub const PIECE: usize = 4096;
pub const FILL: u8 = 0x5A;
pub const MANIFEST_NAME: &str = ".aeiou-dataset.json";
/// 2 since 2026-10-12: `state` and `progress` (`DESIGN_REVIEW.md` §3.70), so a reader that
/// knows only 1 cannot take an unfinished corpus for a whole one.
pub const MANIFEST_VERSION: u64 = 2;
/// A dataset's `state`: written at the start as `writing`, `complete` once every file is.
pub const STATE_WRITING: &str = "writing";
pub const STATE_COMPLETE: &str = "complete";
pub const NAMESPACE_MANIFEST_NAME: &str = ".aeiou-namespace.json";
pub const NAMESPACE_MANIFEST_VERSION: u64 = 1;
/// Above this many objects a namespace manifest records the count only.
pub const NAMESPACE_OBJECT_LIMIT: usize = 100_000;

/// The payload settings recorded in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PayloadSpec {
    pub generator: String,
    pub version: String,
    pub wrapper: String,
    pub block: u64,
    pub dedupe: u64,
    pub compress: u64,
}

impl PayloadSpec {
    pub fn new(dedupe: u64, compress: u64) -> Self {
        PayloadSpec {
            generator: GENERATOR.into(),
            version: GENERATOR_VERSION.into(),
            wrapper: WRAPPER.into(),
            block: BLOCK,
            dedupe: dedupe.max(1),
            compress: compress.max(1),
        }
    }

    pub fn check(&self) -> Result<()> {
        if self.generator != GENERATOR || self.version != GENERATOR_VERSION || self.wrapper != WRAPPER || self.block != BLOCK {
            bail!(
                "payload {} {} {} block {} is not what this runner generates ({GENERATOR} {GENERATOR_VERSION} {WRAPPER} block {BLOCK})",
                self.generator,
                self.version,
                self.wrapper,
                self.block
            );
        }
        Ok(())
    }
}

/// The seed of block `block` of unit `unit` under `seed`.
pub fn block_seed(seed: u64, unit: u64, block: u64) -> u64 {
    labeled_key(seed, "payload", &[unit, block])
}

/// The seed of logical block `block` of file `id` of a `files` dataset under dedupe ratio
/// `dedupe`: `D` consecutive files share a unit (since `aeiou-positional/2`).
pub fn file_block_seed(seed: u64, id: u64, block: u64, dedupe: u64) -> u64 {
    block_seed(seed, id / dedupe.max(1), block)
}

/// The seed of block `block` of a `regions` dataset's one file under dedupe ratio `dedupe`:
/// `D` consecutive blocks are one block repeated (since `aeiou-positional/2`).
pub fn regions_block_seed(seed: u64, block: u64, dedupe: u64) -> u64 {
    block_seed(seed, 0, block / dedupe.max(1))
}

/// The seed of a namespace object's content.
pub fn object_seed(namespace_seed: u64, path: &str) -> u64 {
    labeled_key(namespace_seed, "object", &[xxhash_rust::xxh3::xxh3_64(path.as_bytes())])
}

/// Chris Doty-Humphrey's SFC64, as numpy's `SFC64` bit generator steps it: `(a, b, c)`
/// and a counter, 64 bits each.
pub struct Sfc64 {
    s: [u64; 4],
}

impl Sfc64 {
    /// The state for `seed`: `a`, `b`, `c` the first three SplitMix64 outputs from `seed`, the
    /// counter 1, then twelve outputs discarded (numpy seeds it so, from its own seed
    /// sequence).
    pub fn new(seed: u64) -> Sfc64 {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut g = Sfc64 { s: [next(), next(), next(), 1] };
        for _ in 0..12 {
            g.next_u64();
        }
        g
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let tmp = s[0].wrapping_add(s[1]).wrapping_add(s[3]);
        s[3] = s[3].wrapping_add(1);
        s[0] = s[1] ^ (s[1] >> 11);
        s[1] = s[2].wrapping_add(s[2] << 3);
        s[2] = s[2].rotate_left(24).wrapping_add(tmp);
        tmp
    }

    /// The next `out.len()` bytes of the stream, each output little-endian.
    pub fn fill(&mut self, out: &mut [u8]) {
        let mut words = out.chunks_exact_mut(8);
        for w in &mut words {
            w.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let tail = words.into_remainder();
        if !tail.is_empty() {
            let n = tail.len();
            tail.copy_from_slice(&self.next_u64().to_le_bytes()[..n]);
        }
    }
}

/// The stream bytes in each 4 KiB piece under compression ratio `compress`.
pub fn random_per_piece(compress: u64) -> usize {
    PIECE.div_ceil(compress.max(1) as usize)
}

/// A block being made in order: the block offset of its next byte, and the generator there
/// (with what is left of the last output, for a range that ended inside one).
struct Cursor {
    seed: u64,
    at: usize,
    g: Sfc64,
    spare: [u8; 8],
    spare_at: usize,
}

impl Cursor {
    fn new(seed: u64) -> Cursor {
        Cursor { seed, at: 0, g: Sfc64::new(seed), spare: [0; 8], spare_at: 8 }
    }

    /// The next `out.len()` bytes of the stream.
    fn stream(&mut self, out: &mut [u8]) {
        let k = (8 - self.spare_at).min(out.len());
        out[..k].copy_from_slice(&self.spare[self.spare_at..self.spare_at + k]);
        self.spare_at += k;
        let rest = &mut out[k..];
        let mut words = rest.chunks_exact_mut(8);
        for w in &mut words {
            w.copy_from_slice(&self.g.next_u64().to_le_bytes());
        }
        let tail = words.into_remainder();
        if !tail.is_empty() {
            let n = tail.len();
            self.spare = self.g.next_u64().to_le_bytes();
            tail.copy_from_slice(&self.spare[..n]);
            self.spare_at = n;
        }
    }

    /// The block's next `out.len()` bytes: per 4 KiB piece, `r` stream bytes then `FILL`.
    fn fill(&mut self, r: usize, out: &mut [u8]) {
        if r == PIECE {
            self.stream(out);
            self.at += out.len();
            return;
        }
        let mut done = 0;
        while done < out.len() {
            let off = self.at % PIECE;
            let k = if off < r { (r - off).min(out.len() - done) } else { (PIECE - off).min(out.len() - done) };
            if off < r {
                self.stream(&mut out[done..done + k]);
            } else {
                out[done..done + k].fill(FILL);
            }
            self.at += k;
            done += k;
        }
    }
}

/// Generates block content: a block read in order from its start is made in place as it is
/// read (a container's samples, which start anywhere in a block); any other sub-block range
/// comes from a one-block cache.
pub struct Filler {
    compress: u64,
    cache_seed: Option<u64>,
    cache: Vec<u8>,
    scratch: Vec<u8>,
    cursor: Option<Cursor>,
}

impl Filler {
    pub fn new(compress: u64) -> Self {
        Filler { compress: compress.max(1), cache_seed: None, cache: Vec::new(), scratch: Vec::new(), cursor: None }
    }

    /// The first `out.len()` bytes (≤ 1 MiB) of the block seeded `seed`.
    pub fn fill_block(&mut self, seed: u64, out: &mut [u8]) {
        Self::generate(self.compress, seed, out, &mut self.scratch)
    }

    /// The block's layout: piece `k` holds stream bytes `[k·R, k·R + R)` and then `FILL`, so a
    /// prefix needs only the stream bytes of its own pieces.
    fn generate(compress: u64, seed: u64, out: &mut [u8], scratch: &mut Vec<u8>) {
        debug_assert!(out.len() as u64 <= BLOCK);
        if out.is_empty() {
            return;
        }
        let mut g = Sfc64::new(seed);
        let r = random_per_piece(compress);
        if r == PIECE {
            g.fill(out);
            return;
        }
        let pieces = out.len().div_ceil(PIECE);
        let last = out.len() - (pieces - 1) * PIECE;
        scratch.resize((pieces - 1) * r + r.min(last), 0);
        g.fill(scratch);
        for (k, piece) in out.chunks_mut(PIECE).enumerate() {
            let n = r.min(piece.len());
            piece[..n].copy_from_slice(&scratch[k * r..k * r + n]);
            piece[n..].fill(FILL);
        }
    }

    /// Fill `buf` with the content at logical `offset` under `seed_of(block)`.
    pub fn fill_range(&mut self, seed_of: impl Fn(u64) -> u64, offset: u64, buf: &mut [u8]) {
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let block = pos / BLOCK;
            let in_block = (pos % BLOCK) as usize;
            let n = ((BLOCK as usize) - in_block).min(buf.len() - done);
            let seed = seed_of(block);
            let next = matches!(&self.cursor, Some(c) if c.seed == seed && c.at == in_block);
            if in_block == 0 && n as u64 == BLOCK {
                self.fill_block(seed, &mut buf[done..done + n]);
            } else if next || in_block == 0 {
                if !next {
                    self.cursor = Some(Cursor::new(seed));
                }
                let r = random_per_piece(self.compress);
                self.cursor.as_mut().unwrap().fill(r, &mut buf[done..done + n]);
            } else {
                if self.cache_seed != Some(seed) || self.cache.len() < in_block + n {
                    self.cache.resize(BLOCK as usize, 0);
                    Self::generate(self.compress, seed, &mut self.cache[..], &mut self.scratch);
                    self.cache_seed = Some(seed);
                }
                buf[done..done + n].copy_from_slice(&self.cache[in_block..in_block + n]);
            }
            done += n;
        }
    }
}

// ---------------------------------------------------------------- manifest

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub manifest_version: u64,
    /// `writing` from before the first file, `complete` once the last is (neither hashed).
    #[serde(default)]
    pub state: String,
    /// The resolved dataset definition: the `datasets` entry with parameters substituted,
    /// `doc` removed, canonical key order.
    pub dataset: Value,
    pub payload: PayloadSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(default)]
    pub provenance: Value,
    /// While `writing`: what is not yet written and flushed (`unfinished`, id ranges), the
    /// counts so far, and when it was recorded; absent once `complete`.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub progress: Value,
}

impl Manifest {
    pub fn normative(&self) -> Value {
        let mut m = json!({
            "manifest_version": self.manifest_version,
            "dataset": self.dataset,
            "payload": serde_json::to_value(&self.payload).unwrap(),
        });
        if let Some(f) = &self.format {
            m["format"] = f.clone();
        }
        m
    }

    /// While `writing`: the id ranges not yet written and flushed, as recorded.
    pub fn unfinished(&self) -> Vec<(i64, i64)> {
        self.progress["unfinished"].as_array().into_iter().flatten().filter_map(|x| Some((x[0].as_i64()?, x[1].as_i64()?))).collect()
    }

    /// The dataset id: SHA-256 of the canonical normative content.
    pub fn id(&self) -> String {
        canon::sha256_hex(&self.normative())
    }

    pub fn read(dir: &Path) -> Result<Manifest> {
        let path = dir.join(MANIFEST_NAME);
        let text = std::fs::read_to_string(&path).with_context(|| format!("no manifest at {}", path.display()))?;
        Manifest::parse(&text, &path.display().to_string())
    }

    /// The manifest object at the root `rest` of an object store.
    pub fn read_object(store: &crate::object::Store, rest: &str) -> Result<Manifest> {
        let key = join_key(rest, MANIFEST_NAME);
        let what = store.show(&key);
        let bytes = store.get(&key)?.ok_or_else(|| anyhow!("no manifest at {what}"))?;
        Manifest::parse(&String::from_utf8_lossy(&bytes), &what)
    }

    fn parse(text: &str, what: &str) -> Result<Manifest> {
        let m: Manifest = serde_json::from_str(text).with_context(|| format!("{what}: not a manifest"))?;
        if m.manifest_version != MANIFEST_VERSION {
            bail!("{what}: manifest_version {} (this runner writes {})", m.manifest_version, MANIFEST_VERSION);
        }
        Ok(m)
    }

    /// One `PUT` at the root `rest` of an object store, which no reader sees half written.
    pub fn write_object(&self, store: &crate::object::Store, rest: &str) -> Result<String> {
        let key = join_key(rest, MANIFEST_NAME);
        store.put(&key, serde_json::to_string_pretty(self)?.into_bytes())?;
        Ok(store.show(&key))
    }

    /// Written to a temporary name and renamed into place (`schema/README.md` §6).
    pub fn write(&self, dir: &Path) -> Result<PathBuf> {
        let path = dir.join(MANIFEST_NAME);
        let tmp = dir.join(format!("{MANIFEST_NAME}.tmp.{}", std::process::id()));
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
        Ok(path)
    }
}

/// `name` below the key `rest` (`""`: at the endpoint's prefix itself).
pub fn join_key(rest: &str, name: &str) -> String {
    if rest.is_empty() { name.to_string() } else { format!("{rest}/{name}") }
}

/// The dataset's root directory relative to the run root (`schema/README.md` §6): the
/// constant directory prefix of a `files` pattern; the directory of a `regions` file.
pub fn dataset_root(ast: &Ast, name: &str) -> Result<String> {
    match ast.datasets.get(name).ok_or_else(|| anyhow!("no dataset `{name}`"))? {
        Dataset::Files(f) => Ok(Pattern::parse(&f.pattern)?.root().to_string()),
        Dataset::Regions(r) => Ok(match r.file.rfind('/') {
            Some(i) => r.file[..i].to_string(),
            None => String::new(),
        }),
    }
}

/// One rank of the run that wrote a namespace: where it ran and which GPU ids it owned.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RankRecord {
    pub rank: i64,
    pub host: String,
    /// `[lo, hi)` of global GPU ids.
    pub gpus: [i64; 2],
}

/// `.aeiou-namespace.json` at a namespace root, written by the run that created the
/// objects there, read by a run that declares those namespaces `input` (V14).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceManifest {
    pub manifest_version: u64,
    /// The resolved definition of every namespace at this root (`pattern`, `fields`, `seed`
    /// with parameters substituted, canonical key order), by name.
    pub namespaces: BTreeMap<String, Value>,
    pub abstract_name: String,
    pub ast_sha256: String,
    pub seed: u64,
    pub gpus: i64,
    pub params: Value,
    pub ranks: Vec<RankRecord>,
    /// Unix seconds, fractional.
    pub started: f64,
    pub finished: f64,
    pub objects_created: u64,
    pub bytes_written: u64,
    /// Every object created at this root with the GPU id that created it, when there are at
    /// most `NAMESPACE_OBJECT_LIMIT`; otherwise absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objects: Option<Vec<(String, i64)>>,
}

impl NamespaceManifest {
    pub fn read(dir: &Path) -> Result<NamespaceManifest> {
        let path = dir.join(NAMESPACE_MANIFEST_NAME);
        let text = std::fs::read_to_string(&path).with_context(|| format!("no namespace manifest at {}", path.display()))?;
        NamespaceManifest::parse(&text, &path.display().to_string())
    }

    /// The manifest object at the root `rest` of an object store.
    pub fn read_object(store: &crate::object::Store, rest: &str) -> Result<NamespaceManifest> {
        let key = join_key(rest, NAMESPACE_MANIFEST_NAME);
        let what = store.show(&key);
        let bytes = store.get(&key)?.ok_or_else(|| anyhow!("no namespace manifest at {what}"))?;
        NamespaceManifest::parse(&String::from_utf8_lossy(&bytes), &what)
    }

    fn parse(text: &str, what: &str) -> Result<NamespaceManifest> {
        let m: NamespaceManifest = serde_json::from_str(text).with_context(|| format!("{what}: not a namespace manifest"))?;
        if m.manifest_version != NAMESPACE_MANIFEST_VERSION {
            bail!("{what}: manifest_version {} (this runner writes {})", m.manifest_version, NAMESPACE_MANIFEST_VERSION);
        }
        Ok(m)
    }

    /// One `PUT` at the root `rest` of an object store, which no reader sees half written.
    pub fn write_object(&self, store: &crate::object::Store, rest: &str) -> Result<String> {
        let key = join_key(rest, NAMESPACE_MANIFEST_NAME);
        store.put(&key, serde_json::to_string_pretty(self)?.into_bytes())?;
        Ok(store.show(&key))
    }

    pub fn write(&self, dir: &Path) -> Result<PathBuf> {
        let path = dir.join(NAMESPACE_MANIFEST_NAME);
        let tmp = dir.join(format!("{NAMESPACE_MANIFEST_NAME}.tmp.{}", std::process::id()));
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
        Ok(path)
    }

    /// The host that wrote GPU id `gpu`, if recorded.
    pub fn host_of_gpu(&self, gpu: i64) -> Option<&str> {
        self.ranks.iter().find(|r| gpu >= r.gpus[0] && gpu < r.gpus[1]).map(|r| r.host.as_str())
    }
}

/// The resolved definition of namespace `name` for the manifest: `pattern`, `fields`, and
/// `seed`, which fix the names and the content. `size` is each abstract's own model of the
/// objects (`as_written` for the writer, an expression for a reader, V4) and is not
/// compared; `doc` and `input` are not either (the writer declares the namespace without
/// `input`, the reader with it), nor `protocol` (where it lives, not what it is).
pub fn resolved_namespace(doc: &Value, name: &str, cfg: &Config) -> Result<Value> {
    let entry = doc
        .get("namespaces")
        .and_then(|d| d.get(name))
        .ok_or_else(|| anyhow!("no namespace `{name}` in the abstract"))?;
    let values = param_values(doc, cfg)?;
    let mut v = substitute(entry, &values)?;
    if let Some(o) = v.as_object_mut() {
        o.remove("doc");
        o.remove("input");
        o.remove("same_run");
        o.remove("size");
        o.remove("protocol");
    }
    let bytes = canon::canonical(&v);
    Ok(serde_json::from_slice(&bytes)?)
}

/// A namespace's root directory relative to the run root.
pub fn namespace_root(ast: &Ast, name: &str) -> Result<String> {
    let n = ast.namespaces.get(name).ok_or_else(|| anyhow!("no namespace `{name}`"))?;
    Ok(Pattern::parse(&n.pattern)?.root().to_string())
}

/// The parameter values in effect as JSON, for substitution and provenance.
pub fn param_values(doc: &Value, cfg: &Config) -> Result<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    if let Some(params) = doc.get("params").and_then(|p| p.as_object()) {
        for (k, p) in params {
            out.insert(k.clone(), p.get("default").cloned().unwrap_or(Value::Null));
        }
    }
    for set in &cfg.sets {
        for (name, v) in &set.values {
            out.insert(name.clone(), v.clone());
        }
    }
    for (name, text) in &cfg.overrides {
        let v = serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()));
        out.insert(name.clone(), v);
    }
    out.insert("gpus".into(), json!(cfg.gpus));
    Ok(out)
}

/// The resolved definition of dataset `name`: its `datasets` entry with every `{"param": x}`
/// replaced by the value in effect and `doc` removed, in canonical key order.
pub fn resolved_dataset(doc: &Value, name: &str, cfg: &Config) -> Result<Value> {
    let entry = doc
        .get("datasets")
        .and_then(|d| d.get(name))
        .ok_or_else(|| anyhow!("no dataset `{name}` in the abstract"))?;
    let values = param_values(doc, cfg)?;
    let mut v = substitute(entry, &values)?;
    if let Some(obj) = v.as_object_mut() {
        for inner in obj.values_mut() {
            if let Some(o) = inner.as_object_mut() {
                o.remove("doc");
                o.remove("protocol");
            }
        }
    }
    let bytes = canon::canonical(&v);
    Ok(serde_json::from_slice(&bytes)?)
}

fn substitute(v: &Value, values: &BTreeMap<String, Value>) -> Result<Value> {
    Ok(match v {
        Value::Object(o) => {
            if o.len() == 1 {
                if let Some(Value::String(name)) = o.get("param") {
                    return values.get(name).cloned().ok_or_else(|| anyhow!("param `{name}` has no value"));
                }
            }
            let mut out = serde_json::Map::new();
            for (k, x) in o {
                out.insert(k.clone(), substitute(x, values)?);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| substitute(x, values)).collect::<Result<_>>()?),
        x => x.clone(),
    })
}

/// Field-by-field differences between two JSON values, as `path: a != b` lines.
pub fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys: std::collections::BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            for k in keys {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match (x.get(k), y.get(k)) {
                    (Some(a), Some(b)) => diff(a, b, &p, out),
                    (Some(a), None) => out.push(format!("{p}: {a} vs (absent)")),
                    (None, Some(b)) => out.push(format!("{p}: (absent) vs {b}")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (a, b)) in x.iter().zip(y).enumerate() {
                diff(a, b, &format!("{path}[{i}]"), out);
            }
        }
        _ => {
            if a != b {
                out.push(format!("{path}: {a} vs {b}"));
            }
        }
    }
}

/// The parameter values a `Params` resolved to, as JSON, for the manifest's provenance.
pub fn params_json(doc: &Value, cfg: &Config, _params: &Params) -> Result<Value> {
    Ok(Value::Object(param_values(doc, cfg)?.into_iter().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    #[test]
    fn block_prefix_is_chunk_size_independent() {
        let mut f = Filler::new(1);
        let mut full = vec![0u8; BLOCK as usize];
        f.fill_block(42, &mut full);
        let mut head = vec![0u8; 4096];
        f.fill_block(42, &mut head);
        assert_eq!(&full[..4096], &head[..]);
        let mut other = vec![0u8; 4096];
        f.fill_block(43, &mut other);
        assert_ne!(&head[..], &other[..]);
        // a sub-block range through the cache
        let mut mid = vec![0u8; 10000];
        f.fill_range(|_| 42, 5000, &mut mid);
        assert_eq!(&full[5000..15000], &mid[..]);
    }

    #[test]
    fn a_range_read_in_order_is_the_blocks_bytes() {
        // a container's payload: ranges of odd lengths back to back over block boundaries,
        // made in place by the cursor, against whole blocks; then a jump back through the cache
        for c in [1u64, 2, 3, 7] {
            let seed_of = |b: u64| 100 + b;
            let mut whole = vec![0u8; 3 * BLOCK as usize];
            let mut f = Filler::new(c);
            for b in 0..3 {
                f.fill_block(seed_of(b as u64), &mut whole[b * BLOCK as usize..(b + 1) * BLOCK as usize]);
            }
            let mut f = Filler::new(c);
            let mut got = vec![0u8; whole.len()];
            let (mut at, mut k) = (0usize, 1usize);
            while at < got.len() {
                let n = (k * 7919 % 300_001 + 1).min(got.len() - at);
                f.fill_range(seed_of, at as u64, &mut got[at..at + n]);
                at += n;
                k += 1;
            }
            assert!(got == whole, "C={c}");
            let mut back = vec![0u8; 5000];
            f.fill_range(seed_of, BLOCK + 123, &mut back);
            assert_eq!(&back[..], &whole[BLOCK as usize + 123..BLOCK as usize + 5123], "C={c}");
        }
    }

    #[test]
    fn compression_fills_every_piece() {
        for c in [2u64, 3, 4] {
            let r = random_per_piece(c);
            let mut f = Filler::new(c);
            let mut b = vec![0u8; BLOCK as usize];
            f.fill_block(7, &mut b);
            for piece in b.chunks(PIECE) {
                assert!(piece[r..].iter().all(|&x| x == FILL), "C={c}");
                assert!(piece[..r].iter().any(|&x| x != FILL), "C={c}");
            }
            // the stream continues from piece to piece, and a prefix is the block's prefix
            let mut s = vec![0u8; 2 * r];
            Sfc64::new(7).fill(&mut s);
            assert_eq!(&b[..r], &s[..r]);
            assert_eq!(&b[PIECE..PIECE + r], &s[r..]);
            let mut head = vec![0u8; 5000];
            f.fill_block(7, &mut head);
            assert_eq!(&b[..5000], &head[..]);
        }
    }

    /// The bytes numpy's `SFC64` makes from the same state (`builder/tests/test_payload.py`
    /// pins the same values): the first outputs for seed 0 and the SHA-256 of whole blocks.
    #[test]
    fn the_stream_and_blocks_are_pinned() {
        let mut g = Sfc64::new(0);
        let first: Vec<u64> = (0..3).map(|_| g.next_u64()).collect();
        assert_eq!(first, PINNED_FIRST);
        for (c, want) in PINNED_BLOCKS {
            let mut b = vec![0u8; BLOCK as usize];
            Filler::new(c).fill_block(0x1234_5678_9ABC_DEF0, &mut b);
            let hex: String = sha2::Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect();
            assert_eq!(hex, want, "C={c}");
        }
    }

    const PINNED_FIRST: [u64; 3] = [0xeaf7_3661_f5e1_80bc, 0xbc90_4e12_62de_1088, 0x0653_8b07_830a_ee11];
    const PINNED_BLOCKS: [(u64, &str); 3] = [
        (1, "a2eb20e7d329c5ec5dddb0ccd476a040aea1b465701f0a7e1ec8d998a5804258"),
        (2, "931796b308ff3de12bbecbb615e24e832ed6d21449decda560b566d4fc5e746b"),
        (3, "f3d76909654cbbbe2e2f4daeba766978879f046852302ac27e63c6a99b1b2612"),
    ];
}
