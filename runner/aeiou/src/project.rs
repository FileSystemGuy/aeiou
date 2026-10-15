//! The projection of a dataset: what `aeiou datagen` would write for it at these parameters,
//! summed exactly from the size expressions and the framing, without the payload
//! (`DESIGN_REVIEW.md` §3.74 item 3). `aeiou dry-run` prints it, `aeiou datagen` checks it
//! against a directory root's free space, and `aeiou run` reports it beside the hosts'
//! memory. No rounding and no rule: the numbers follow from the parameters given.

use crate::eval::{DsMeta, Model};
use anyhow::Result;
use serde_json::{json, Value};

/// One dataset's projected size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projection {
    pub name: String,
    /// Samples (`count`); an `array` dataset's slots.
    pub samples: i64,
    /// Files (objects, chunks included) and directories as datagen creates them; the
    /// inventory is not counted, as datagen does not count it in `files_written`.
    pub files: u64,
    pub directories: i64,
    /// Bytes datagen writes: every file's size (an `array`'s written slots, not its sparse
    /// length), and the inventory's.
    pub bytes: u64,
    /// The inventory's bytes, part of `bytes`.
    pub inventory_bytes: u64,
}

impl Projection {
    pub fn mean_file(&self) -> u64 {
        if self.files == 0 { 0 } else { self.bytes.saturating_sub(self.inventory_bytes) / self.files }
    }

    pub fn json(&self) -> Value {
        json!({
            "name": self.name,
            "samples": self.samples,
            "files": self.files,
            "directories": self.directories,
            "bytes": self.bytes,
            "inventory_bytes": self.inventory_bytes,
            "mean_file": self.mean_file(),
        })
    }
}

/// The projection of `meta`, its files spread over `threads` threads. Bytes are exact, not
/// rounded to any block size: allocation is the file system's (an NFS client's `statvfs`
/// block is the transfer size, which says nothing of the server's allocation).
pub fn project(meta: &DsMeta<'_>, threads: usize) -> Result<Projection> {
    let (files, bytes) = match meta {
        // the written slots, not the sparse length
        DsMeta::Array { count, .. } => (1, sum_over(meta.name(), *count, threads, |c| Ok((0, meta.sample_size(c)? as u64)))?.1),
        // a chunked file is `ceil(size / chunk)` objects (`chunks`)
        DsMeta::Files { chunk, .. } => sum_over(meta.name(), meta.files(), threads, |id| {
            let size = meta.file_size(id)? as u64;
            Ok((chunk.map_or(1, |c| size.div_ceil(c as u64)), size))
        })?,
    };
    let inventory_bytes = meta.inventory().map(|(_, n, _)| n as u64).unwrap_or(0);
    let directories = match meta {
        DsMeta::Files { ndirs, .. } => *ndirs,
        DsMeta::Array { .. } => 0,
    };
    Ok(Projection {
        name: meta.name().to_string(),
        samples: meta.count(),
        files,
        directories,
        bytes: add(meta.name(), (0, bytes), (0, inventory_bytes))?.1,
        inventory_bytes,
    })
}

/// The sums of `f(i)` for `i` in `0..n`, over `threads` threads in contiguous ranges.
fn sum_over(name: &str, n: i64, threads: usize, f: impl Fn(i64) -> Result<(u64, u64)> + Sync) -> Result<(u64, u64)> {
    let threads = threads.clamp(1, n.max(1) as usize) as i64;
    let per = crate::eval::ceil_div(n, threads);
    let f = &f;
    let parts: Vec<Result<(u64, u64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut sum = (0u64, 0u64);
                    for i in (t * per)..((t + 1).saturating_mul(per)).min(n) {
                        sum = add(name, sum, f(i)?)?;
                    }
                    Ok(sum)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("a projection thread panicked")).collect()
    });
    parts.into_iter().try_fold((0, 0), |s, p| add(name, s, p?))
}

/// Two partial sums, checked: a projection past 2^64-1 is refused, never wrapped.
fn add(name: &str, s: (u64, u64), p: (u64, u64)) -> Result<(u64, u64)> {
    match (s.0.checked_add(p.0), s.1.checked_add(p.1)) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(anyhow::anyhow!("dataset `{name}`: its projected files or bytes pass 2^64-1")),
    }
}

/// Every dataset of the model, unrounded.
pub fn project_all(model: &Model<'_>, threads: usize) -> Result<Vec<Projection>> {
    model.datasets.iter().map(|m| project(m, threads)).collect()
}
