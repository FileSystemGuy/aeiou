//! Payload and the dataset manifest.
//!
//! **Payload.** The bytes at `(seed, unit, block)` are a pure function of that tuple
//! (`PROJECT_BRIEF.md` §5, *Data verification*; `DESIGN_REVIEW.md` §3.16–§3.17). Content is
//! cut into 1 MiB blocks; block `b` of unit `u` is the prefix of a 1 MiB `dgen-data` stream
//! seeded `labeled_key(seed, "payload", [u, b])`, with dgen's compression layout (the last
//! `(C−1)/C` of the block zero-filled for ratio `C`) and no dgen dedupe: dedupe is this
//! wrapper's, by seed reuse across units (`aeiou-positional/1`):
//! - a `files` dataset: `u = id mod ceil(count / D)`, `b` = logical offset in the file ÷ 1 MiB
//!   (a chunked file is the logical file cut at `chunk`), so file `id` and file `id + count/D`
//!   carry the same bytes block for block;
//! - a `regions` dataset (one file): `u = 0`, `b = (offset ÷ 1 MiB) mod ceil(blocks / D)`;
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
pub const WRAPPER: &str = "aeiou-positional/1";
pub const GENERATOR: &str = "dgen-data";
pub const GENERATOR_VERSION: &str = "0.3.0";
pub const MANIFEST_NAME: &str = ".aeiou-dataset.json";
pub const MANIFEST_VERSION: u64 = 1;
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

/// The seed of a namespace object's content.
pub fn object_seed(namespace_seed: u64, path: &str) -> u64 {
    labeled_key(namespace_seed, "object", &[xxhash_rust::xxh3::xxh3_64(path.as_bytes())])
}

/// Generates block content, with a one-block cache for sub-block ranges.
pub struct Filler {
    compress: u64,
    cache_seed: Option<u64>,
    cache: Vec<u8>,
}

impl Filler {
    pub fn new(compress: u64) -> Self {
        Filler { compress: compress.max(1), cache_seed: None, cache: Vec::new() }
    }

    /// The first `out.len()` bytes (≤ 1 MiB) of the block seeded `seed`.
    pub fn fill_block(&mut self, seed: u64, out: &mut [u8]) {
        Self::generate(self.compress, seed, out)
    }

    fn generate(compress: u64, seed: u64, out: &mut [u8]) {
        debug_assert!(out.len() as u64 <= BLOCK);
        if out.is_empty() {
            return;
        }
        let cfg = dgen_data::GeneratorConfig {
            size: BLOCK as usize,
            dedup_factor: 1,
            compress_factor: compress as usize,
            numa_mode: dgen_data::NumaMode::Disabled,
            max_threads: Some(1),
            numa_node: None,
            block_size: Some(BLOCK as usize),
            seed: Some(seed),
        };
        let mut g = dgen_data::DataGenerator::new(cfg);
        let n = g.fill_chunk(out);
        debug_assert_eq!(n, out.len());
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
            if in_block == 0 && n as u64 == BLOCK {
                self.fill_block(seed, &mut buf[done..done + n]);
            } else {
                if self.cache_seed != Some(seed) || self.cache.len() < in_block + n {
                    self.cache.resize(BLOCK as usize, 0);
                    Self::generate(self.compress, seed, &mut self.cache[..]);
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
    /// The resolved dataset definition: the `datasets` entry with parameters substituted,
    /// `doc` removed, canonical key order.
    pub dataset: Value,
    pub payload: PayloadSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(default)]
    pub provenance: Value,
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

    /// The dataset id: SHA-256 of the canonical normative content.
    pub fn id(&self) -> String {
        canon::sha256_hex(&self.normative())
    }

    pub fn read(dir: &Path) -> Result<Manifest> {
        let path = dir.join(MANIFEST_NAME);
        let text = std::fs::read_to_string(&path).with_context(|| format!("no manifest at {}", path.display()))?;
        let m: Manifest = serde_json::from_str(&text).with_context(|| format!("{}: not a manifest", path.display()))?;
        if m.manifest_version != MANIFEST_VERSION {
            bail!("{}: manifest_version {} (this runner writes {})", path.display(), m.manifest_version, MANIFEST_VERSION);
        }
        Ok(m)
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
        let m: NamespaceManifest = serde_json::from_str(&text).with_context(|| format!("{}: not a namespace manifest", path.display()))?;
        if m.manifest_version != NAMESPACE_MANIFEST_VERSION {
            bail!("{}: manifest_version {} (this runner writes {})", path.display(), m.manifest_version, NAMESPACE_MANIFEST_VERSION);
        }
        Ok(m)
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
/// `input`, the reader with it).
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
        o.remove("size");
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
    fn compression_zero_fills_the_tail() {
        let mut f = Filler::new(2);
        let mut b = vec![1u8; BLOCK as usize];
        f.fill_block(7, &mut b);
        assert!(b[..1000].iter().any(|&x| x != 0));
        assert!(b[BLOCK as usize - 1000..].iter().all(|&x| x == 0));
    }
}
