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
