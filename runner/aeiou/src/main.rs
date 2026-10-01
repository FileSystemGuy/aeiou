//! `aeiou`: the runner binary. Subcommands: `check` (validate ASTs and print their hashes,
//! as `schema/check.py` does), `dry-run` (the op streams and the fingerprint, no I/O),
//! `datagen` (write the corpus and its manifests), and `run` (execute against a directory
//! with a blocking backend).

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};

use aeiou::backend::BackendKind;
use aeiou::datagen::{self, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Params};
use aeiou::run::{self, RunOpts};

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
    /// Write the datasets an abstract declares under --root, with a manifest per dataset root.
    Datagen(DatagenArgs),
    /// Execute the abstract against --root with a blocking I/O backend.
    Run(RunCmd),
}

#[derive(Args)]
struct DatagenArgs {
    /// The abstract (`.ast.json`).
    abstract_path: PathBuf,
    /// Directory the abstract's paths are relative to.
    #[arg(long)]
    root: PathBuf,
    /// Instance count, for dataset definitions that reference `gpus`.
    #[arg(long, default_value_t = 1)]
    gpus: i64,
    /// Override a parameter: `--param name=value` (only those the datasets reference matter).
    #[arg(long = "param", value_name = "NAME=VALUE")]
    params: Vec<String>,
    /// Writer threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
    /// Dedupe ratio: every `dedupe` files (or 1 MiB blocks of a regions file) share content.
    #[arg(long, default_value_t = 1)]
    dedupe: u64,
    /// Compression ratio: the last (C−1)/C of every 1 MiB block is zeros.
    #[arg(long, default_value_t = 1)]
    compress: u64,
    /// Only these datasets (default: all).
    #[arg(long = "dataset", value_name = "NAME")]
    datasets: Vec<String>,
}

#[derive(Args)]
struct RunCmd {
    #[command(flatten)]
    run: RunArgs,
    /// Directory the abstract's paths are relative to (datasets and namespaces live under it).
    #[arg(long)]
    root: PathBuf,
    /// `sync` (buffered POSIX on one thread per actor) or `sync-direct` (the same with O_DIRECT).
    #[arg(long = "io-backend", default_value = "sync")]
    backend: String,
    /// Per-thread read and write buffer ring, MiB.
    #[arg(long, default_value_t = 8)]
    buffer_mib: usize,
    /// Compression ratio of the bytes written to namespaces.
    #[arg(long, default_value_t = 1)]
    write_compress: u64,
    /// Multiply every `compute` sleep (0 runs the I/O back to back).
    #[arg(long, default_value_t = 1.0)]
    time_scale: f64,
    /// Empty the namespace roots before starting instead of refusing.
    #[arg(long)]
    clean_namespaces: bool,
    /// Fail unless the run's fingerprint is this (hex, from `aeiou dry-run`).
    #[arg(long, value_name = "HEX")]
    expect_fingerprint: Option<String>,
    /// Fail unless every dataset id is among these.
    #[arg(long = "expect-dataset-id", value_name = "SHA256")]
    expect_dataset_ids: Vec<String>,
    /// This host's index among --ranks hosts (the multi-host coordinator is not built yet: --ranks 1 only).
    #[arg(long, default_value_t = 0)]
    rank: i64,
    #[arg(long, default_value_t = 1)]
    ranks: i64,
    /// Run the GPU range of rank (rank + k) mod ranks, so each host reads what another wrote.
    #[arg(long, default_value_t = 0, value_name = "K")]
    rank_rotate: i64,
    /// Fail if an input namespace was finished more than this many seconds ago.
    #[arg(long, value_name = "SECS")]
    max_gap: Option<f64>,
    /// Fail if this host would read input objects it wrote itself.
    #[arg(long)]
    require_cold: bool,
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
        Cmd::Datagen(a) => datagen_cmd(a),
        Cmd::Run(a) => run_cmd(a),
    }
}

fn datagen_cmd(a: DatagenArgs) -> Result<()> {
    let cfg = parse_config(&RunArgs { abstract_path: a.abstract_path.clone(), gpus: a.gpus, seed: 0, params: a.params.clone() })?;
    let loaded = aeiou::load(&a.abstract_path)?;
    let params = Params::new(&loaded.ast, &cfg)?;
    let model = build_model(&loaded.ast, &cfg, &params)?;
    let threads = a.threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let opts = DatagenOpts { root: a.root.clone(), threads, dedupe: a.dedupe, compress: a.compress, datasets: a.datasets.clone() };
    let mut out = std::io::stdout();
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    let results = datagen::datagen(&loaded, &cfg, &params, &model, &opts, &mut out)?;
    for r in &results {
        writeln!(
            out,
            "dataset {}: {} file(s), {} in {:.2?} at {}  id {}",
            r.name,
            r.files,
            dryrun::human_bytes(r.bytes),
            r.elapsed,
            r.root.display(),
            r.id
        )?;
    }
    Ok(())
}

fn run_cmd(a: RunCmd) -> Result<()> {
    let cfg = parse_config(&a.run)?;
    let backend = BackendKind::parse(&a.backend).ok_or_else(|| anyhow::anyhow!("--io-backend {}: not one of sync, sync-direct", a.backend))?;
    let expect_fingerprint = match &a.expect_fingerprint {
        None => None,
        Some(h) => Some(u64::from_str_radix(h.trim_start_matches("0x"), 16).map_err(|_| anyhow::anyhow!("--expect-fingerprint {h}: not hex"))?),
    };
    // the run is the process: the abstract and the model live for the threads' lifetime
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&a.run.abstract_path)?));
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg)?));
    let model = Box::leak(Box::new(build_model(&loaded.ast, cfg, params)?));

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    let overrides: Vec<String> = cfg.overrides.iter().map(|(k, v)| format!("{k}={v}")).collect();
    writeln!(out, "seed {}  gpus {}  params: {}", cfg.seed, cfg.gpus, if overrides.is_empty() { "defaults".to_string() } else { overrides.join(" ") })?;
    writeln!(out, "backend {}  root {}", backend.name(), a.root.display())?;

    let checks = run::check_datasets(loaded, cfg, &a.root)?;
    for c in &checks {
        if !a.expect_dataset_ids.is_empty() && !a.expect_dataset_ids.iter().any(|x| *x == c.id) {
            bail!("dataset `{}` id {} is not among --expect-dataset-id", c.name, c.id);
        }
        writeln!(
            out,
            "dataset {} at {}/: id {}  payload {} {} dedupe {} compress {}{}",
            c.name,
            c.root,
            c.id,
            c.payload.generator,
            c.payload.version,
            c.payload.dedupe,
            c.payload.compress,
            c.files.map(|n| format!("  ({n} files)")).unwrap_or_default()
        )?;
    }
    if a.ranks < 1 || a.rank < 0 || a.rank >= a.ranks {
        bail!("--rank {} of --ranks {}: rank must be in [0, ranks)", a.rank, a.ranks);
    }
    if a.ranks > 1 {
        bail!("--ranks {}: several hosts need the coordinator, which is not built yet (NAPKIN_MATH.md §8.A)", a.ranks);
    }
    let opts = RunOpts {
        root: a.root.clone(),
        backend,
        buffer_bytes: a.buffer_mib.max(1) << 20,
        write_compress: a.write_compress,
        time_scale: a.time_scale.max(0.0),
        clean_namespaces: a.clean_namespaces,
        expect_fingerprint,
        expect_dataset_ids: a.expect_dataset_ids.clone(),
        rank: a.rank,
        ranks: a.ranks,
        rank_rotate: a.rank_rotate,
        max_gap: a.max_gap,
        require_cold: a.require_cold,
    };
    let (lo, hi) = run::gpu_range(cfg.gpus, opts.ranks, opts.rank, opts.rank_rotate);
    writeln!(out, "host {}  rank {} of {}  rotate {}  gpu ids [{lo}, {hi})", run::hostname(), opts.rank, opts.ranks, opts.rank_rotate)?;
    let (ns_checks, input_objects) = run::check_input_namespaces(loaded, cfg, &a.root, &opts)?;
    for c in &ns_checks {
        writeln!(
            out,
            "input namespace(s) {} at {}/: written by `{}` ({}…) on {}, finished {:.1} s ago{}{}",
            c.names.join(", "),
            c.root,
            c.writer_abstract,
            &c.writer_sha256[..16],
            c.writer_hosts.join(","),
            c.gap,
            c.objects.map(|n| format!(", {n} objects recorded")).unwrap_or_default(),
            if c.same_host { "; WARNING: this host wrote part of the range it will run" } else { "" }
        )?;
    }
    let cleaned = run::prepare_namespaces(&loaded.ast, &a.root, a.clean_namespaces)?;
    for c in &cleaned {
        writeln!(out, "namespace root {c}/ emptied")?;
    }
    out.flush()?;

    let started = run::unix_now();
    let report = run::run(model, opts.clone(), input_objects)?;
    let finished = run::unix_now();
    run::write_report(&mut out, &report)?;
    for p in run::write_namespace_manifests(loaded, cfg, &a.root, &opts, &report, started, finished)? {
        writeln!(out, "namespace manifest {}", p.display())?;
    }
    if let Some(fp) = expect_fingerprint {
        if report.stats.fingerprint != fp {
            bail!("fingerprint {:016x} does not match the expected {:016x}", report.stats.fingerprint, fp);
        }
        writeln!(out, "fingerprint matches")?;
    }
    Ok(())
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
