//! `aeiou`: the runner binary. Subcommands: `check` (validate ASTs and print their hashes,
//! as `schema/check.py` does), `dry-run` (the op streams and the fingerprint, no I/O),
//! `datagen` (write the corpus and its manifests), and `run` (execute against a directory
//! with a blocking backend, on one host or on several through the TCP coordinator).

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use clap::{Args, Parser, Subcommand};

use aeiou::backend::{BackendKind, MmapConsume, MmapMode};
use aeiou::coord::{Coordinator, Local, Server, Tcp};
use aeiou::datagen::{self, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, ParamSet, Params};
use aeiou::payload;
use aeiou::run::{self, Report, RunOpts, UringOpts};

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
    /// Execute the abstract against --root with a blocking I/O backend (several hosts: --ranks R --rank r --coordinator HOST:PORT on each).
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
    /// A parameter file (`.params.json`), applied over the defaults and under --param; repeatable, in order.
    #[arg(long = "params", value_name = "FILE")]
    param_files: Vec<PathBuf>,
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
    /// `sync` (buffered POSIX on one thread per actor), `sync-direct` (the same with O_DIRECT),
    /// `io_uring` (an event loop per thread multiplexing the actors over one ring), `io_uring-direct`,
    /// `posix-aio` (glibc aio_read/aio_write on one thread per actor), `posix-aio-direct`,
    /// `libaio` (the kernel AIO calls on the event loop; asynchronous only as `libaio-direct`),
    /// `mmap` (reads are copies out of a mapping of the file).
    #[arg(long = "io-backend", default_value = "sync")]
    backend: String,
    /// Event-loop threads for the io_uring and libaio backends (default: one per core, at most one
    /// per actor instance). The other backends run one thread per actor and ignore it.
    #[arg(long)]
    threads: Option<usize>,
    /// Per-thread read and write buffer ring, MiB.
    #[arg(long, default_value_t = 8)]
    buffer_mib: usize,
    /// Compression ratio of the bytes written to namespaces.
    #[arg(long, default_value_t = 1)]
    write_compress: u64,
    /// Multiply every `compute` sleep (0 runs the I/O back to back).
    #[arg(long, default_value_t = 1.0)]
    time_scale: f64,
    /// io_uring: cap each loop's bounded io-wq workers (IORING_REGISTER_IOWQ_MAX_WORKERS); the
    /// report shows the kernel's default either way.
    #[arg(long, value_name = "N")]
    iowq_max_workers: Option<u32>,
    /// io_uring: a kernel submission thread per loop (IORING_SETUP_SQPOLL) that sleeps after
    /// this many idle milliseconds.
    #[arg(long, value_name = "IDLE_MS")]
    sqpoll: Option<u32>,
    /// io_uring: one submission thread, and one io-wq, shared by every loop
    /// (IORING_SETUP_ATTACH_WQ) instead of one per loop.
    #[arg(long, requires = "sqpoll")]
    sqpoll_shared: bool,
    /// io_uring: IORING_SETUP_SINGLE_ISSUER with IORING_SETUP_DEFER_TASKRUN (not with --sqpoll).
    #[arg(long, conflicts_with = "sqpoll")]
    defer_taskrun: bool,
    /// io_uring: IORING_SETUP_COOP_TASKRUN.
    #[arg(long)]
    coop_taskrun: bool,
    /// libaio: requests each loop's AIO context holds (io_setup's nr_events; the host's total is
    /// bounded by fs.aio-max-nr). Default 256.
    #[arg(long, value_name = "N")]
    aio_depth: Option<u32>,
    /// mmap: the prefetch before a read's range is consumed: `fault` (none: the touch faults
    /// the pages in), `populate` (MADV_POPULATE_READ over the range), `willneed` (MADV_WILLNEED).
    #[arg(long, value_name = "MODE")]
    mmap_mode: Option<String>,
    /// mmap: how a read's range is consumed: `touch` (one byte of every page is read, so each
    /// page is resident and mapped; nothing under `populate`, which has done that) or `copy`
    /// (the range is copied into the actor's buffer). Default touch.
    #[arg(long, value_name = "HOW")]
    mmap_consume: Option<String>,
    /// Empty the namespace roots before starting instead of refusing.
    #[arg(long)]
    clean_namespaces: bool,
    /// Fail unless the run's fingerprint is this (hex, from `aeiou dry-run`).
    #[arg(long, value_name = "HEX")]
    expect_fingerprint: Option<String>,
    /// Fail unless every dataset id is among these.
    #[arg(long = "expect-dataset-id", value_name = "SHA256")]
    expect_dataset_ids: Vec<String>,
    /// This host's index among --ranks hosts.
    #[arg(long, default_value_t = 0)]
    rank: i64,
    #[arg(long, default_value_t = 1)]
    ranks: i64,
    /// With --ranks above 1: the coordinator's address. Rank 0 listens on it (in-process); every rank connects to it.
    #[arg(long, value_name = "HOST:PORT")]
    coordinator: Option<String>,
    /// Run the GPU range of rank (rank + k) mod ranks, so each host reads what another wrote.
    #[arg(long, default_value_t = 0, value_name = "K")]
    rank_rotate: i64,
    /// Fail if an input namespace was finished more than this many seconds ago.
    #[arg(long, value_name = "SECS")]
    max_gap: Option<f64>,
    /// Fail if this host would read input objects it wrote itself, or if dataset pages are
    /// in its page cache at the start (mincore over 256 sampled files per dataset).
    #[arg(long)]
    require_cold: bool,
    /// On every host, before the start gate: sync, then drop the page cache, dentries, and
    /// inodes (3 into /proc/sys/vm/drop_caches), then sample the datasets' residency. Needs
    /// root; the run refuses when it fails.
    #[arg(long)]
    drop_caches: bool,
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
    /// A parameter file (`.params.json`, schema/README.md §8), applied over the defaults and under --param; repeatable, in order.
    #[arg(long = "params", value_name = "FILE")]
    param_files: Vec<PathBuf>,
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
    let cfg = parse_config(&RunArgs { abstract_path: a.abstract_path.clone(), gpus: a.gpus, seed: 0, params: a.params.clone(), param_files: a.param_files.clone() })?;
    let loaded = aeiou::load(&a.abstract_path)?;
    cfg.check_sets(&loaded.ast.name, &loaded.sha256)?;
    let params = Params::new(&loaded.ast, &cfg)?;
    let model = build_model(&loaded.ast, &cfg, &params)?;
    let threads = a.threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let opts = DatagenOpts { root: a.root.clone(), threads, dedupe: a.dedupe, compress: a.compress, datasets: a.datasets.clone() };
    let mut out = std::io::stdout();
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    writeln!(out, "gpus {}  params: {}", cfg.gpus, params_line(&cfg))?;
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
    let backend = BackendKind::parse(&a.backend).ok_or_else(|| anyhow::anyhow!("--io-backend {}: not one of {}", a.backend, aeiou::backend::NAMES))?;
    let expect_fingerprint = match &a.expect_fingerprint {
        None => None,
        Some(h) => Some(u64::from_str_radix(h.trim_start_matches("0x"), 16).map_err(|_| anyhow::anyhow!("--expect-fingerprint {h}: not hex"))?),
    };
    if a.ranks < 1 || a.rank < 0 || a.rank >= a.ranks {
        bail!("--rank {} of --ranks {}: rank must be in [0, ranks)", a.rank, a.ranks);
    }
    if a.ranks > 1 && a.coordinator.is_none() {
        bail!("--ranks {}: several hosts need --coordinator HOST:PORT (rank 0 listens there, every rank connects to it)", a.ranks);
    }
    let uring = UringOpts { iowq_max_workers: a.iowq_max_workers.unwrap_or(0), sqpoll_idle_ms: a.sqpoll, sqpoll_shared: a.sqpoll_shared, defer_taskrun: a.defer_taskrun, coop_taskrun: a.coop_taskrun };
    if uring.any() && !backend.uring() {
        bail!("--iowq-max-workers, --sqpoll, --defer-taskrun, --coop-taskrun are io_uring knobs; --io-backend {} has no ring", backend.name());
    }
    uring.check()?;
    if a.aio_depth.is_some() && !backend.libaio() {
        bail!("--aio-depth is a libaio knob; --io-backend {} has no AIO context", backend.name());
    }
    if a.aio_depth == Some(0) {
        bail!("--aio-depth 0: a context needs room for a request");
    }
    let mmap = match &a.mmap_mode {
        None => MmapMode::default(),
        Some(_) if backend != BackendKind::Mmap => bail!("--mmap-mode is an mmap knob; --io-backend {} maps nothing", backend.name()),
        Some(m) => MmapMode::parse(m).ok_or_else(|| anyhow::anyhow!("--mmap-mode {m}: not one of fault, populate, willneed"))?,
    };
    let mmap_consume = match &a.mmap_consume {
        None => MmapConsume::default(),
        Some(_) if backend != BackendKind::Mmap => bail!("--mmap-consume is an mmap knob; --io-backend {} maps nothing", backend.name()),
        Some(c) => MmapConsume::parse(c).ok_or_else(|| anyhow::anyhow!("--mmap-consume {c}: not one of touch, copy"))?,
    };
    // the run is the process: the abstract and the model live for the threads' lifetime
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&a.run.abstract_path)?));
    cfg.check_sets(&loaded.ast.name, &loaded.sha256)?;
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg)?));
    let model = Box::leak(Box::new(build_model(&loaded.ast, cfg, params)?));

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    writeln!(out, "seed {}  gpus {}  params: {}", cfg.seed, cfg.gpus, params_line(cfg))?;
    writeln!(out, "backend {}  root {}{}", backend.name(), a.root.display(), if uring.any() {
        format!("  io_uring knobs: {}", uring.describe())
    } else if backend == BackendKind::Mmap {
        format!("  mmap mode: {}  consume: {}", mmap.name(), mmap_consume.name())
    } else {
        String::new()
    })?;

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
    let opts = RunOpts {
        root: a.root.clone(),
        backend,
        buffer_bytes: a.buffer_mib.max(1) << 20,
        threads: a.threads.unwrap_or(0),
        write_compress: a.write_compress,
        time_scale: a.time_scale.max(0.0),
        uring,
        mmap,
        mmap_consume,
        aio_depth: a.aio_depth.unwrap_or(0),
        clean_namespaces: a.clean_namespaces,
        expect_fingerprint,
        expect_dataset_ids: a.expect_dataset_ids.clone(),
        rank: a.rank,
        ranks: a.ranks,
        rank_rotate: a.rank_rotate,
        max_gap: a.max_gap,
        require_cold: a.require_cold,
        drop_caches: a.drop_caches,
    };
    let (lo, hi) = run::gpu_range(cfg.gpus, opts.ranks, opts.rank, opts.rank_rotate);
    writeln!(out, "host {}  rank {} of {}  rotate {}  gpu ids [{lo}, {hi})", run::hostname(), opts.rank, opts.ranks, opts.rank_rotate)?;

    // several hosts: rank 0 listens, every rank connects and has its configuration checked
    // before anything else happens; a host that fails later tells the others through it
    let participants = run::participants(model, &opts)?;
    let aborted = Arc::new(AtomicBool::new(false));
    let server = match &a.coordinator {
        Some(addr) if a.ranks > 1 && a.rank == 0 => {
            let s = Server::start(addr, a.ranks)?;
            writeln!(out, "coordinator listening on {}", s.addr)?;
            out.flush()?;
            Some(s)
        }
        _ => None,
    };
    let (coord, tcp): (Arc<dyn Coordinator>, Option<Arc<Tcp>>) = match &a.coordinator {
        Some(addr) if a.ranks > 1 => {
            let dataset_ids: Vec<&str> = checks.iter().map(|c| c.id.as_str()).collect();
            let config = serde_json::json!({
                "ast_sha256": loaded.sha256,
                "seed": cfg.seed,
                "gpus": cfg.gpus,
                "params": payload::params_json(&loaded.doc, cfg, params)?,
                "dataset_ids": dataset_ids,
                "backend": backend.name(),
                "mmap_mode": mmap.name(),
                "mmap_consume": mmap_consume.name(),
                "rank_rotate": a.rank_rotate,
                "time_scale": opts.time_scale,
                "write_compress": a.write_compress,
                "drop_caches": a.drop_caches,
            });
            let config = aeiou::canon::sha256_hex(&config);
            let t = Arc::new(Tcp::connect(addr, a.rank, a.ranks, &run::hostname(), &config, &participants, aborted.clone())?);
            writeln!(out, "coordinator {}: connected as rank {} of {}  config {}…", t.addr, a.rank, a.ranks, &config[..16])?;
            (t.clone(), Some(t))
        }
        _ => (Arc::new(Local::new(&participants)), None),
    };
    let r = run_connected(&a, loaded, cfg, model, opts, coord, aborted, tcp.as_deref(), server.as_ref(), &mut out);
    if let (Err(e), Some(t)) = (&r, &tcp) {
        // the coordinator relays a failure here to every other host (a no-op after a Stop)
        t.stop(&format!("{e:#}"));
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn run_connected(
    a: &RunCmd,
    loaded: &'static aeiou::Loaded,
    cfg: &'static Config,
    model: &'static aeiou::eval::Model<'static>,
    opts: RunOpts,
    coord: Arc<dyn Coordinator>,
    aborted: Arc<AtomicBool>,
    tcp: Option<&Tcp>,
    server: Option<&Server>,
    out: &mut impl Write,
) -> Result<()> {
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
    // --root is the storage under test, shared by every host: rank 0 prepares the output
    // namespace roots before the start gate; the other hosts never empty anything
    if a.rank == 0 {
        let cleaned = run::prepare_namespaces(&loaded.ast, &a.root, a.clean_namespaces)?;
        for c in &cleaned {
            writeln!(out, "namespace root {c}/ emptied")?;
        }
    }
    out.flush()?;

    // the cold start: after the checks and the namespace preparation, before the gate, so
    // every host has dropped before any op is issued and the drop is outside `elapsed`
    // (printed with the report, this host's and the merged one)
    let cold = aeiou::cold::start(model, &opts)?;

    let mut started = run::unix_now();
    if let Some(t) = tcp {
        let (t0, hosts) = t.ready().map_err(|e| anyhow!("start gate: {e:#}"))?;
        started = t0;
        writeln!(out, "start gate: {} host(s) ready: {}", hosts.len(), hosts.join(", "))?;
        out.flush()?;
    }
    let mut report = run::run_with(model, opts.clone(), input_objects, coord, aborted)?;
    report.cold = vec![cold];
    let finished = run::unix_now();
    if tcp.is_some() {
        writeln!(out, "--- this host ({}), rank {} of {}", report.host, a.rank, a.ranks)?;
    }
    run::write_report(out, &report)?;

    // the verdict: manifests for what was written and the fingerprint check, on the merged
    // report when there are several hosts
    let verdict = |out: &mut dyn Write, report: &Report| -> Result<()> {
        for p in run::write_namespace_manifests(loaded, cfg, &a.root, &opts, report, started, finished)? {
            writeln!(out, "namespace manifest {}", p.display())?;
        }
        if let Some(fp) = opts.expect_fingerprint {
            if report.stats.fingerprint != fp {
                bail!("fingerprint {:016x} does not match the expected {:016x}", report.stats.fingerprint, fp);
            }
            writeln!(out, "fingerprint matches")?;
        }
        Ok(())
    };
    match (tcp, server) {
        (None, _) => verdict(out, &report),
        (Some(t), None) => {
            t.report(&report)?;
            writeln!(out, "report sent to the coordinator; waiting for rank 0's verdict")?;
            out.flush()?;
            let (ok, fp, err) = t.result()?;
            writeln!(out, "--- all {} hosts: fingerprint {fp:016x}", a.ranks)?;
            if !ok {
                bail!("rank 0: {}", err.unwrap_or_else(|| "failed".into()));
            }
            if let Some(exp) = opts.expect_fingerprint {
                if fp != exp {
                    bail!("fingerprint {fp:016x} does not match the expected {exp:016x}");
                }
                writeln!(out, "fingerprint matches")?;
            }
            Ok(())
        }
        (Some(t), Some(s)) => {
            t.report(&report)?;
            out.flush()?;
            let merged = s.merged()?;
            writeln!(out, "--- all {} hosts ({})", merged.ranks.len(), merged.host)?;
            for r in &merged.ranks {
                writeln!(out, "rank {} {}  gpu ids [{}, {})", r.rank, r.host, r.gpus[0], r.gpus[1])?;
            }
            run::write_report(out, &merged)?;
            match verdict(out, &merged) {
                Ok(()) => {
                    s.finish(true, merged.stats.fingerprint, None);
                    Ok(())
                }
                Err(e) => {
                    s.finish(false, merged.stats.fingerprint, Some(format!("{e:#}")));
                    Err(e)
                }
            }
        }
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
    let mut sets = Vec::new();
    for p in &a.param_files {
        sets.push(ParamSet::load(p)?);
    }
    Ok(Config { seed: a.seed, gpus: a.gpus, overrides, sets })
}

/// The parameters in effect, as the header line prints them: the files (with their hashes)
/// and the `--param` overrides, or `defaults`.
fn params_line(cfg: &Config) -> String {
    let mut parts: Vec<String> = cfg.sets.iter().map(|s| format!("{} ({}…)", s.path, &s.sha256[..16])).collect();
    parts.extend(cfg.overrides.iter().map(|(k, v)| format!("{k}={v}")));
    if parts.is_empty() {
        "defaults".to_string()
    } else {
        parts.join(" ")
    }
}

fn dry_run(a: DryRunArgs) -> Result<()> {
    let cfg = parse_config(&a.run)?;
    let loaded = aeiou::load(&a.run.abstract_path)?;
    cfg.check_sets(&loaded.ast.name, &loaded.sha256)?;
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
    writeln!(out, "seed {}  gpus {}  params: {}", cfg.seed, cfg.gpus, params_line(&cfg))?;
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
