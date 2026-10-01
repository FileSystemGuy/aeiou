//! `aeiou datagen`: write the corpus an abstract declares, from the same names, sizes, and
//! dataset seeds the runner computes, with the payload of `payload.rs`, and a manifest at
//! each dataset root (`schema/README.md` §6). Files are written in parallel by id with no
//! per-file structure; the manifest is written last, atomically.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::eval::{Config, DsMeta, Model, Params};
use crate::payload::{self, Filler, Manifest, PayloadSpec, BLOCK};

#[derive(Debug, Clone)]
pub struct DatagenOpts {
    pub root: PathBuf,
    pub threads: usize,
    pub dedupe: u64,
    pub compress: u64,
    /// Only these datasets (all when empty).
    pub datasets: Vec<String>,
}

pub struct DatasetResult {
    pub name: String,
    pub root: PathBuf,
    pub files: u64,
    pub bytes: u64,
    pub id: String,
    pub elapsed: std::time::Duration,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "?".into())
}

/// One unit of work: a file to write with its logical extent.
struct Job {
    path: String,
    /// Logical offset of the file's first byte within its unit (chunk k starts at k·chunk).
    logical: u64,
    len: u64,
    unit: u64,
}

fn write_file(full: &Path, job: &Job, seed: u64, filler: &mut Filler, buf: &mut Vec<u8>, dedupe_blocks: Option<u64>) -> Result<()> {
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut f = std::fs::File::create(full).with_context(|| format!("creating {}", full.display()))?;
    let mut done = 0u64;
    while done < job.len {
        let n = (job.len - done).min(BLOCK) as usize;
        buf.resize(n, 0);
        let unit = job.unit;
        filler.fill_range(
            |b| match dedupe_blocks {
                Some(m) => payload::block_seed(seed, unit, b % m),
                None => payload::block_seed(seed, unit, b),
            },
            job.logical + done,
            &mut buf[..n],
        );
        f.write_all(&buf[..n]).with_context(|| format!("writing {}", full.display()))?;
        done += n as u64;
    }
    Ok(())
}

pub fn datagen(loaded: &crate::Loaded, cfg: &Config, _params: &Params, model: &Model<'_>, opts: &DatagenOpts, log: &mut (impl Write + Send)) -> Result<Vec<DatasetResult>> {
    let spec = PayloadSpec::new(opts.dedupe, opts.compress);
    let mut results = Vec::new();
    let threads = opts.threads.max(1);
    let params_json = payload::params_json(&loaded.doc, cfg, _params)?;
    for (i, meta) in model.datasets.iter().enumerate() {
        let name = meta.name();
        if !opts.datasets.is_empty() && !opts.datasets.iter().any(|d| d == name) {
            continue;
        }
        let _ = i;
        let rel = payload::dataset_root(&loaded.ast, name)?;
        let root = opts.root.join(&rel);
        if root.exists() {
            let n = std::fs::read_dir(&root)?.count();
            if n > 0 {
                bail!("dataset `{name}`: {} is not empty ({n} entries); datasets are read-only, remove it first", root.display());
            }
        }
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        let started = Instant::now();
        let start_secs = now_secs();

        let count = meta.count();
        let seed = meta.dataset_seed();
        // the jobs are computed, not stored: an atomic cursor over ids
        let (total_ids, units) = match meta {
            DsMeta::Files { count, spf, .. } => {
                let files = (count + spf - 1) / spf;
                let units = (files + spec.dedupe as i64 - 1) / spec.dedupe as i64;
                (files, units.max(1) as u64)
            }
            DsMeta::Regions { .. } => (1, 1),
        };
        let dedupe_blocks = match meta {
            DsMeta::Regions { .. } => {
                let size = meta.file_size(0)? as u64;
                let blocks = (size + BLOCK - 1) / BLOCK;
                Some(((blocks + spec.dedupe - 1) / spec.dedupe).max(1))
            }
            _ => None,
        };
        let next = AtomicI64::new(0);
        let files = AtomicU64::new(0);
        let bytes = AtomicU64::new(0);
        let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
        let done_files = AtomicU64::new(0);
        let last_log = Mutex::new(Instant::now());
        let log_mutex: Mutex<&mut (dyn Write + Send)> = Mutex::new(log);
        std::thread::scope(|s| {
            for _ in 0..threads.min(total_ids.max(1) as usize) {
                s.spawn(|| {
                    let mut filler = Filler::new(spec.compress);
                    let mut buf = Vec::with_capacity(BLOCK as usize);
                    let mut jobs: Vec<Job> = Vec::new();
                    loop {
                        let id = next.fetch_add(1, Ordering::Relaxed);
                        if id >= total_ids {
                            break;
                        }
                        jobs.clear();
                        let r: Result<()> = (|| {
                            match meta {
                                DsMeta::Files { spf, chunk, .. } => {
                                    // ids enumerate files (spf samples each); a chunked file is several objects
                                    let file = id;
                                    if file * spf >= count {
                                        return Ok(());
                                    }
                                    let size = meta.file_size(file)? as u64;
                                    let unit = (file as u64) % units;
                                    match chunk {
                                        None => jobs.push(Job { path: meta.file_path(file, None)?.to_string(), logical: 0, len: size, unit }),
                                        Some(c) => {
                                            let c = *c as u64;
                                            let n = meta.chunks(file)?;
                                            for k in 0..n {
                                                let len = meta.chunk_size(file, k)? as u64;
                                                jobs.push(Job { path: meta.file_path(file, Some(k))?.to_string(), logical: k as u64 * c, len, unit });
                                            }
                                        }
                                    }
                                }
                                DsMeta::Regions { .. } => {
                                    jobs.push(Job { path: meta.file_path(0, None)?.to_string(), logical: 0, len: meta.file_size(0)? as u64, unit: 0 });
                                }
                            }
                            for job in &jobs {
                                let full = opts.root.join(&job.path);
                                write_file(&full, job, seed, &mut filler, &mut buf, dedupe_blocks)?;
                                files.fetch_add(1, Ordering::Relaxed);
                                bytes.fetch_add(job.len, Ordering::Relaxed);
                            }
                            Ok(())
                        })();
                        if let Err(e) = r {
                            errors.lock().unwrap().push(e);
                            break;
                        }
                        let d = done_files.fetch_add(1, Ordering::Relaxed) + 1;
                        if d % 1000 == 0 {
                            let mut ll = last_log.lock().unwrap();
                            if ll.elapsed().as_secs() >= 5 {
                                *ll = Instant::now();
                                let mut l = log_mutex.lock().unwrap();
                                let _ = writeln!(l, "  {name}: {d}/{total_ids} ids, {} written", crate::dryrun::human_bytes(bytes.load(Ordering::Relaxed)));
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
            return Err(e.context(format!("dataset `{name}`")));
        }
        let files_n = files.load(Ordering::Relaxed);
        let bytes_n = bytes.load(Ordering::Relaxed);
        // ids of files: count/spf files for a files dataset
        let _ = total_ids;
        let manifest = Manifest {
            manifest_version: payload::MANIFEST_VERSION,
            dataset: payload::resolved_dataset(&loaded.doc, name, cfg)?,
            payload: spec.clone(),
            format: None,
            provenance: json!({
                "abstract": loaded.ast.name,
                "ast_sha256": loaded.sha256,
                "params": params_json,
                "param_files": cfg.sets.iter().map(|s| json!({"path": s.path, "sha256": s.sha256})).collect::<Vec<_>>(),
                "datagen": format!("aeiou {}", env!("CARGO_PKG_VERSION")),
                "host": hostname(),
                "started": start_secs,
                "finished": now_secs(),
                "files_written": files_n,
                "bytes_written": bytes_n,
            }),
        };
        manifest.write(&root)?;
        results.push(DatasetResult { name: name.to_string(), root: root.clone(), files: files_n, bytes: bytes_n, id: manifest.id(), elapsed: started.elapsed() });
    }
    Ok(results)
}
