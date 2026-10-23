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
