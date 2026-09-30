//! `aeiou`: the runner binary. Subcommands: `check` (validate ASTs and print their hashes,
//! as `schema/check.py` does) and `dry-run` (the op streams and the fingerprint, no I/O).
//! `run` and `datagen` come with the I/O backends.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};

use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Params};

#[derive(Parser)]
#[command(name = "aeiou", version, about = "Abstract-driven I/O workload runner")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Validate abstracts and print their canonical hashes and op counts.
    Check {
        /// `.ast.json` files.
        files: Vec<PathBuf>,
    },
    /// Compute every actor's op stream and the workload fingerprint without doing any I/O.
    DryRun(DryRunArgs),
}

#[derive(Args)]
struct RunArgs {
    /// The abstract (`.ast.json`).
    abstract_path: PathBuf,
    /// Number of instances of every actor template whose count is `gpus` (global ids 0..G).
    #[arg(long)]
    gpus: i64,
    /// The run seed. The dataset seed is separate and lives in the abstract.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Override a parameter: `--param name=value` (JSON; a bare word is a string).
    #[arg(long = "param", value_name = "NAME=VALUE")]
    params: Vec<String>,
}

#[derive(Args)]
struct DryRunArgs {
    #[command(flatten)]
    run: RunArgs,
    /// Hosts the run would be spread over, for the bytes-per-host estimate.
    #[arg(long)]
    ranks: Option<i64>,
    /// Worker threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
    /// Print the op stream of actor instance G (one thread, in order).
    #[arg(long, value_name = "G")]
    gpu: Option<i64>,
    /// With --gpu: only ops whose outermost loop index is in [A, B).
    #[arg(long, value_name = "A..B")]
    steps: Option<String>,
    /// With --gpu: stop printing after N ops.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("aeiou: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Check { files } => check(files),
        Cmd::DryRun(a) => dry_run(a),
    }
}

fn check(files: Vec<PathBuf>) -> Result<()> {
    if files.is_empty() {
        bail!("no files given");
    }
    let mut failed = 0;
    for f in &files {
        let name = f.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        match aeiou::load(f) {
            Ok(loaded) => {
                let ops: Vec<String> = loaded.ops.iter().map(|(k, v)| format!("{k}={v}")).collect();
                println!("ok   {name}  sha256={}…  ops: {}", &loaded.sha256[..16], ops.join(" "));
            }
            Err(e) => {
                failed += 1;
                println!("FAIL {name}");
                for line in format!("{e:#}").lines() {
                    println!("  {line}");
                }
            }
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn parse_config(a: &RunArgs) -> Result<Config> {
    if a.gpus < 1 {
        bail!("--gpus must be at least 1");
    }
    let mut overrides = Vec::new();
    for p in &a.params {
        let Some((k, v)) = p.split_once('=') else { bail!("--param {p}: expected NAME=VALUE") };
        overrides.push((k.to_string(), v.to_string()));
    }
    Ok(Config { seed: a.seed, gpus: a.gpus, overrides })
}

fn dry_run(a: DryRunArgs) -> Result<()> {
    let cfg = parse_config(&a.run)?;
    let loaded = aeiou::load(&a.run.abstract_path)?;
    let params = Params::new(&loaded.ast, &cfg)?;
    let model = build_model(&loaded.ast, &cfg, &params)?;

    let filter = if a.gpu.is_some() || a.steps.is_some() || a.limit.is_some() {
        let steps = match &a.steps {
            None => None,
            Some(s) => {
                let Some((lo, hi)) = s.split_once("..") else { bail!("--steps {s}: expected A..B") };
                Some((lo.parse::<i64>()?, hi.parse::<i64>()?))
            }
        };
        Some(dryrun::Filter { actor: Some(a.gpu.unwrap_or(0)), steps, limit: a.limit })
    } else {
        None
    };
    let threads = a.threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let started = std::time::Instant::now();
    let mut report = dryrun::run(&model, threads, filter)?;
    let elapsed = started.elapsed();

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    let overrides: Vec<String> = cfg.overrides.iter().map(|(k, v)| format!("{k}={v}")).collect();
    writeln!(
        out,
        "seed {}  gpus {}  params: {}",
        cfg.seed,
        cfg.gpus,
        if overrides.is_empty() { "defaults".to_string() } else { overrides.join(" ") }
    )?;
    for line in report.total.take_lines() {
        writeln!(out, "{line}")?;
    }
    for t in &mut report.templates {
        t.run.take_lines();
    }
    dryrun::write_report(&mut out, &report, a.ranks, cfg.gpus)?;
    writeln!(out, "dry-run took {:.2?} on {} thread(s)", elapsed, threads)?;
    Ok(())
}
