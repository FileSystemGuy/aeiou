//! `aeiou`: the abstract-driven I/O workload runner. Layer 3 of the design in
//! `GRAMMAR_OPTIONS.md` Option D: it loads the AST contract (`schema/`), validates it, and
//! executes it: the AST, its canonical hash, the validator, positional randomness, the VM that
//! turns an AST plus `(seed, gpus, params)` into each actor's op stream and the workload
//! fingerprint (`aeiou dry-run`), the blocking I/O backends and the thread-per-actor executor
//! (`aeiou run`), the payload generator and dataset manifest (`aeiou datagen`), and the
//! in-process coordinator.

pub mod ast;
pub mod aio;
pub mod backend;
pub mod canon;
pub mod cold;
pub mod coord;
pub mod counters;
pub mod datagen;
pub mod dryrun;
pub mod eval;
pub mod limits;
pub mod metrics;
pub mod options;
pub mod pattern;
pub mod payload;
pub mod report;
pub mod rng;
pub mod run;
pub mod sites;
pub mod trace;
pub mod uring;
pub mod validate;
pub mod vm;

use std::path::Path;

/// A loaded, annotated, validated AST with its canonical hash.
pub struct Loaded {
    pub ast: ast::Ast,
    pub sha256: String,
    pub ops: std::collections::BTreeMap<&'static str, usize>,
    /// The document as parsed, for the resolved dataset definitions of the manifest.
    pub doc: serde_json::Value,
    /// The files of the `trace` nodes, by the `file` the node names (`trace.rs`); loaded
    /// relative to the document's directory, so empty for a document loaded from text.
    pub traces: std::collections::HashMap<String, std::sync::Arc<trace::TraceFile>>,
}

pub fn load(path: &Path) -> anyhow::Result<Loaded> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let mut loaded = load_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    loaded.traces = trace::load_all(&loaded.ast, &dir).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(loaded)
}

pub fn load_str(text: &str) -> anyhow::Result<Loaded> {
    let doc: serde_json::Value = serde_json::from_str(text)?;
    let sha256 = canon::sha256_hex(&doc);
    let ast = ast::parse(text)?;
    sites::annotate(&ast);
    let ops = validate::check_with_ops(&ast).map_err(|errs| anyhow::anyhow!("invalid abstract:\n  {}", errs.join("\n  ")))?;
    Ok(Loaded { ast, sha256, ops, doc, traces: Default::default() })
}
