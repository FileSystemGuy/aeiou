//! `aeiou`: the abstract-driven I/O workload runner. Layer 3 of the design in
//! `GRAMMAR_OPTIONS.md` Option D: it loads the AST contract (`schema/`), validates it, and
//! executes it. This crate holds the parts that do no I/O: the AST, its canonical hash, the
//! validator, positional randomness, and the VM that turns an AST plus `(seed, gpus, params)`
//! into each actor's op stream and the workload fingerprint (`aeiou dry-run`). The I/O
//! backends and the coordinator come next.

pub mod ast;
pub mod canon;
pub mod dryrun;
pub mod eval;
pub mod pattern;
pub mod rng;
pub mod sites;
pub mod validate;
pub mod vm;

use std::path::Path;

/// A loaded, annotated, validated AST with its canonical hash.
pub struct Loaded {
    pub ast: ast::Ast,
    pub sha256: String,
    pub ops: std::collections::BTreeMap<&'static str, usize>,
}

pub fn load(path: &Path) -> anyhow::Result<Loaded> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    load_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

pub fn load_str(text: &str) -> anyhow::Result<Loaded> {
    let doc: serde_json::Value = serde_json::from_str(text)?;
    let sha256 = canon::sha256_hex(&doc);
    let ast = ast::parse(text)?;
    sites::annotate(&ast);
    let ops = validate::check_with_ops(&ast).map_err(|errs| anyhow::anyhow!("invalid abstract:\n  {}", errs.join("\n  ")))?;
    Ok(Loaded { ast, sha256, ops })
}
