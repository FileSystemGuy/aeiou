//! The committed abstracts and parameter files, under `workloads/` (DESIGN_REVIEW.md §3.66).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

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
