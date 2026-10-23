//! The committed abstracts and parameter files, under `workloads/` (DESIGN_REVIEW.md §3.66), and
//! the tests' scratch directories.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn workloads() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../workloads")
}

/// The committed AST `name` (or the file `name.ast.json`), wherever its mechanism lives.
pub fn workload_ast(name: &str) -> PathBuf {
    let file = if name.ends_with(".ast.json") { name.to_string() } else { format!("{name}.ast.json") };
    find(&workloads(), &file).unwrap_or_else(|| panic!("no {file} under workloads/"))
}

fn find(dir: &Path, file: &str) -> Option<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).collect();
    entries.sort();
    entries.iter().find(|p| p.file_name().is_some_and(|n| n == file)).cloned()
        .or_else(|| entries.iter().filter(|p| p.is_dir()).find_map(|p| find(p, file)))
}

/// file_per_sample's job and GPU values and its corpus's sizes, required since 2026-10-22 (item 33,
/// part 5c; DESIGN_REVIEW.md §3.78): the defaults they held until then, which its unsettled case
/// `jpeg-50m` keeps, for the tests that ran on them.
pub const FILE_PER_SAMPLE: [(&str, &str); 7] = [("batch", "32"), ("workers", "8"), ("prefetch", "2"), ("steps", "500"),
    ("step_time", "105000000"), ("samples", "50000000"), ("sample_median", "112640")];

/// `params`, then file_per_sample's old defaults for every value `params` does not give.
pub fn file_per_sample<'a>(params: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut all = params.to_vec();
    all.extend(FILE_PER_SAMPLE.iter().filter(|(k, _)| !params.iter().any(|(p, _)| p == k)));
    all
}

/// The same as `--param NAME=VALUE` arguments, to go before a test's own (the last `--param` of a
/// name wins) and after any `--params-file` they would override.
pub fn file_per_sample_args() -> Vec<String> {
    FILE_PER_SAMPLE.iter().flat_map(|(k, v)| ["--param".to_string(), format!("{k}={v}")]).collect()
}

/// dcp_save's defaults until 2026-10-23, which a case now gives (its unsettled case four-items;
/// DESIGN_REVIEW §3.78, item 38): the tests that ran the abstract on them pass them.
pub const DCP_SAVE: [(&str, &str); 6] = [("steps", "500"), ("step_time", "323000000"), ("ckpt_every", "100"), ("items", "4"),
    ("item_bytes", "[16777216, 33554432, 16777216, 8388608]"), ("meta_bytes", "2097152")];

/// `params`, then dcp_save's old defaults for every value `params` does not give.
pub fn dcp_save<'a>(params: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut all = params.to_vec();
    all.extend(DCP_SAVE.iter().filter(|(k, _)| !params.iter().any(|(p, _)| p == k)));
    all
}

/// dcp_load's defaults until 2026-10-23, which a case now gives (its unsettled case four-items,
/// the step 100 dcp_save's four-items writes; DESIGN_REVIEW §3.78, item 38): the tests that ran
/// the abstract on them pass them.
pub const DCP_LOAD: [(&str, &str); 7] = [("items", "4"), ("item_bytes", "[16777216, 33554432, 16777216, 8388608]"),
    ("item_off", "[0, 16778793, 50334802, 67113595]"), ("read_order", "[0, 1, 2, 3]"), ("replicas", "1"), ("meta_bytes", "2097152"),
    ("restore_step", "100")];

/// `params`, then dcp_load's old defaults for every value `params` does not give.
pub fn dcp_load<'a>(params: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut all = params.to_vec();
    all.extend(DCP_LOAD.iter().filter(|(k, _)| !params.iter().any(|(p, _)| p == k)));
    all
}

/// The LMCache abstracts' defaults until 2026-10-23, which a case now gives (their unsettled cases
/// chat-64-slots; DESIGN_REVIEW §3.78, item 38): the model's on its GPU, the deployment's, and the
/// load's, the same in lmcache_local_disk, lmcache_shared_fs, and its reader. The tests that ran
/// the abstracts on them pass them.
pub const LMCACHE: [(&str, &str); 16] = [
    ("chunk_bytes", "33554432"), ("context", "8192"), ("prefill_per_token", "40000"), ("decode_per_token", "12000000"),
    ("keep", r#"{"empirical": {"values": [0, 1000000], "weights": [91, 9]}}"#), ("prefill_step", "8192"), ("retain", "5000"),
    ("concurrency", "64"), ("requests", "20000"), ("warm", "2000"),
    ("reuse", r#"{"mixture": [{"weight": 0.15, "dist": null}, {"weight": 0.85, "dist": {"const": 40}}]}"#),
    ("turn_in", r#"{"empirical": {"values": [1, 4, 6, 7, 9, 10, 11, 13, 15, 17, 19, 22, 26, 31, 39, 50, 68, 107, 214, 944], "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}"#),
    ("turn_out", r#"{"empirical": {"values": [12, 34, 62, 92, 119, 148, 175, 201, 225, 248, 271, 294, 317, 342, 370, 407, 454, 519, 615, 785], "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}"#),
    ("sys_prompts", "50"), ("sys_tokens", "1500"), ("sys_pop", r#"{"zipf": {"s": 1.1}}"#),
];

/// `params`, then the LMCache abstracts' old defaults for every value `params` does not give.
pub fn lmcache<'a>(params: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut all = params.to_vec();
    all.extend(LMCACHE.iter().filter(|(k, _)| !params.iter().any(|(p, _)| p == k)));
    all
}

/// The same as `--param NAME=VALUE` arguments, for every old default `given` (NAME=VALUE) does not name.
pub fn lmcache_args(given: &[&str]) -> Vec<String> {
    LMCACHE.iter().filter(|(k, _)| !given.iter().any(|g| g.split_once('=').map(|(n, _)| n) == Some(*k)))
        .flat_map(|(k, v)| ["--param".to_string(), format!("{k}={v}")]).collect()
}

/// safetensors_load's defaults until 2026-10-23, which a case now gives (its unsettled case
/// five-tensors; DESIGN_REVIEW §3.78, item 38): the model's shards and tensor table and the
/// deployment's tp. The tests that ran the abstract on them pass them.
pub const SAFETENSORS: [(&str, &str); 10] = [
    ("shards", "2"), ("shard_bytes", "5368709120"), ("hdr_len", "[98304, 98304]"), ("tp", "8"), ("shard", "[0, 0, 1, 1, 1]"),
    ("off", "[98312, 536969224, 98312, 67207176, 1140949000]"), ("bytes", "[536870912, 134217728, 67108864, 1073741824, 16384]"),
    ("split", r#"["column", "row", "column", "row", "full"]"#), ("rows", "[1, 4096, 1, 8192, 1]"),
    ("row_bytes", "[536870912, 32768, 67108864, 131072, 16384]"),
];

/// `params`, then safetensors_load's old defaults for every value `params` does not give.
pub fn safetensors<'a>(params: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut all = params.to_vec();
    all.extend(SAFETENSORS.iter().filter(|(k, _)| !params.iter().any(|(p, _)| p == k)));
    all
}

/// The same as `--param NAME=VALUE` arguments, for every old default `given` (NAME=VALUE) does not name.
pub fn safetensors_args(given: &[&str]) -> Vec<String> {
    SAFETENSORS.iter().filter(|(k, _)| !given.iter().any(|g| g.split_once('=').map(|(n, _)| n) == Some(*k)))
        .flat_map(|(k, v)| ["--param".to_string(), format!("{k}={v}")]).collect()
}

/// file_per_sample as it was until 2026-10-22: the committed abstract with its old defaults back
/// in place of the required values, and `hdr_read` and `inv_read` 1 MiB in place of the storage's
/// st_blksize, so that a run and a dry run of it read alike; for the tests (the coordinator's)
/// that start the binary in many ways and ran on those defaults.
pub fn file_per_sample_doc() -> serde_json::Value {
    let mut doc: serde_json::Value = serde_json::from_slice(&std::fs::read(workload_ast("file_per_sample.ast.json")).unwrap()).unwrap();
    let params = doc["params"].as_object_mut().unwrap();
    let old = FILE_PER_SAMPLE.iter().map(|(k, v)| (*k, serde_json::json!(v.parse::<i64>().unwrap()))).chain([("hdr_read", 1 << 20), ("inv_read", 1 << 20)].map(|(k, v)| (k, serde_json::json!(v))));
    for (k, v) in old {
        let p = params[k].as_object_mut().unwrap();
        p.remove("kind");
        p.remove("scale");
        p.insert("default".into(), v);
    }
    doc
}

/// `file_per_sample_doc` written once per test process, its file named as the committed one's.
pub fn file_per_sample_as_it_was() -> PathBuf {
    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file_per_sample.ast.json");
        std::fs::write(&path, serde_json::to_vec(&file_per_sample_doc()).unwrap()).unwrap();
        path
    })
    .clone()
}

/// Set (to anything but empty or `0`) to keep every test's scratch directory and print where it
/// is, for debugging; a test-only hook, reserved in `options.rs` so the runner does not warn of it.
pub const KEEP_TEST_DIRS: &str = "AEIOU_KEEP_TEST_DIRS";

static N: AtomicUsize = AtomicUsize::new(0);

/// A test's scratch directory, `<temp>/aeiou-<module>-<pid>-<n>-<tag>`, created empty and removed
/// when dropped: at the end of the test and on a panic's unwind alike, so a failing test leaves
/// nothing behind (a failed coordinator test used to leave gigabytes and fill the disk).
pub struct TempDir(PathBuf);

pub fn tmpdir(module: &str, tag: &str) -> TempDir {
    let d = std::env::temp_dir().join(format!("aeiou-{module}-{}-{}-{tag}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    TempDir(d)
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if std::env::var_os(KEEP_TEST_DIRS).is_some_and(|v| !v.is_empty() && v != "0") {
            eprintln!("{KEEP_TEST_DIRS}: kept {}", self.0.display());
        } else {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

impl std::ops::Deref for TempDir {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<std::ffi::OsStr> for TempDir {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.0.as_os_str()
    }
}

impl aeiou::endpoint::Place for TempDir {
    fn at(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}
