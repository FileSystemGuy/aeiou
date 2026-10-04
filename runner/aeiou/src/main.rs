//! `aeiou`: the runner binary. Subcommands: `check` (validate ASTs and print their hashes,
//! as `schema/check.py` does), `dry-run` (the op streams and the fingerprint, no I/O),
//! `datagen` (write the corpus and its manifests), and `run` (execute against a directory
//! with a blocking backend, on one host or on several through the TCP coordinator).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use aeiou::backend::{BackendKind, MmapConsume, MmapMode};
use aeiou::coord::{Coordinator, Local, Server, Tcp};
use aeiou::datagen::{self, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, ParamSet, Params};
use aeiou::options::{self, Layers};
use aeiou::payload;
use aeiou::run::{self, Report, RunOpts, UringOpts};

#[derive(Parser)]
#[command(name = "aeiou", version, about = "Abstract-driven I/O workload runner")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// A TOML config file (else $AEIOU_CONFIG, else none; never searched for): one table per
    /// subcommand, keys spelled as the long flags. Under the command line and the environment
    /// (runner/README.md §14); every subcommand prints what it resolved and from where.
    #[arg(long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,
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

/// The shape and its parameters: what every subcommand that takes an abstract takes.
#[derive(Args)]
struct ShapeArgs {
    /// The abstract (`.ast.json`).
    abstract_path: PathBuf,
    /// Override a parameter: `--param name=value` (JSON; a bare word is a string).
    #[arg(long = "param", value_name = "NAME=VALUE", help_heading = "Workload")]
    params: Vec<String>,
    /// A parameter file (`.params.json`, schema/README.md §8), applied over the defaults and under --param; repeatable, in order.
    #[arg(long = "params-file", value_name = "FILE", help_heading = "Workload")]
    param_files: Vec<PathBuf>,
}

/// The shape, its parameters, and the run's identity: `run` and `dry-run`.
#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    shape: ShapeArgs,
    /// Number of instances of every actor template whose count is `gpus` (global ids 0..G).
    #[arg(long, help_heading = "Workload")]
    gpus: i64,
    /// The run seed. The dataset seed is separate and lives in the abstract.
    #[arg(long, default_value_t = 0, help_heading = "Workload")]
    seed: u64,
}

#[derive(Args)]
struct DatagenArgs {
    #[command(flatten)]
    shape: ShapeArgs,
    /// Instance count, for dataset definitions that reference `gpus`.
    #[arg(long, default_value_t = 1, help_heading = "Workload")]
    gpus: i64,
    /// Directory the abstract's paths are relative to.
    #[arg(long, help_heading = "Writer")]
    root: Option<PathBuf>,
    /// Writer threads (default: all cores).
    #[arg(long, help_heading = "Writer")]
    threads: Option<usize>,
    /// Dedupe ratio: every `dedupe` files (or 1 MiB blocks of a regions file) share content.
    #[arg(long, default_value_t = 1, help_heading = "Writer")]
    dedupe: u64,
    /// Compression ratio: the last (C−1)/C of every 1 MiB block is zeros.
    #[arg(long, default_value_t = 1, help_heading = "Writer")]
    compress: u64,
    /// Only these datasets (default: all).
    #[arg(long = "dataset", value_name = "NAME", help_heading = "Writer")]
    datasets: Vec<String>,
}

#[derive(Args)]
struct RunCmd {
    #[command(flatten)]
    run: RunArgs,
    /// `sync` (buffered POSIX on one thread per actor), `sync-direct` (the same with O_DIRECT),
    /// `io_uring` (an event loop per thread multiplexing the actors over one ring), `io_uring-direct`,
    /// `posix-aio` (glibc aio_read/aio_write on one thread per actor), `posix-aio-direct`,
    /// `libaio` (the kernel AIO calls on the event loop; asynchronous only as `libaio-direct`),
    /// `mmap` (reads are copies out of a mapping of the file).
    /// Default: the backend the abstract declares (`sync` when it declares none). Any other is a
    /// different workload on the storage, and the run says so.
    #[arg(long = "io-backend", help_heading = "Backend")]
    backend: Option<String>,
    /// Directory the abstract's paths are relative to (datasets and namespaces live under it).
    #[arg(long, help_heading = "Backend")]
    root: Option<PathBuf>,
    /// Event-loop threads for the io_uring and libaio backends (default: one per core, at most one
    /// per actor instance). The other backends run one thread per actor and refuse it.
    #[arg(long, help_heading = "Backend")]
    threads: Option<usize>,
    /// Per-thread read and write buffer ring, MiB. Default 8.
    #[arg(long, value_name = "MIB", help_heading = "Backend")]
    buffer_mib: Option<usize>,
    /// Compression ratio of the bytes written to namespaces. Default 1.
    #[arg(long, value_name = "C", help_heading = "Backend")]
    write_compress: Option<u64>,
    /// Multiply every `compute` sleep (0 runs the I/O back to back). Default 1.
    #[arg(long, value_name = "X", help_heading = "Backend")]
    time_scale: Option<f64>,
    /// Cap each loop's bounded io-wq workers (IORING_REGISTER_IOWQ_MAX_WORKERS); the
    /// report shows the kernel's default either way.
    #[arg(long, value_name = "N", help_heading = "io_uring")]
    iowq_max_workers: Option<u32>,
    /// A kernel submission thread per loop (IORING_SETUP_SQPOLL) that sleeps after
    /// this many idle milliseconds.
    #[arg(long, value_name = "IDLE_MS", help_heading = "io_uring")]
    sqpoll: Option<u32>,
    /// One submission thread, and one io-wq, shared by every loop
    /// (IORING_SETUP_ATTACH_WQ) instead of one per loop; needs --sqpoll.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL", help_heading = "io_uring")]
    sqpoll_shared: Option<bool>,
    /// IORING_SETUP_SINGLE_ISSUER with IORING_SETUP_DEFER_TASKRUN (not with --sqpoll).
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL", help_heading = "io_uring")]
    defer_taskrun: Option<bool>,
    /// IORING_SETUP_COOP_TASKRUN.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL", help_heading = "io_uring")]
    coop_taskrun: Option<bool>,
    /// Requests each loop's AIO context holds (io_setup's nr_events; the host's total is
    /// bounded by fs.aio-max-nr). Default 256.
    #[arg(long, value_name = "N", help_heading = "libaio")]
    aio_depth: Option<u32>,
    /// The prefetch before a read's range is consumed: `fault` (none: the touch faults
    /// the pages in), `populate` (MADV_POPULATE_READ over the range), `willneed` (MADV_WILLNEED).
    #[arg(long, value_name = "MODE", help_heading = "mmap")]
    mmap_mode: Option<String>,
    /// How a read's range is consumed: `touch` (one byte of every page is read, so each
    /// page is resident and mapped; nothing under `populate`, which has done that) or `copy`
    /// (the range is copied into the actor's buffer). Default touch.
    #[arg(long, value_name = "HOW", help_heading = "mmap")]
    mmap_consume: Option<String>,
    /// This host's index among --ranks hosts. Default 0.
    #[arg(long, value_name = "R", help_heading = "Several hosts")]
    rank: Option<i64>,
    /// Hosts the run is spread over; each runs the GPU id range of its --rank. Default 1.
    #[arg(long, value_name = "N", help_heading = "Several hosts")]
    ranks: Option<i64>,
    /// With --ranks above 1: the coordinator's address. Rank 0 listens on it (in-process); every rank connects to it.
    #[arg(long, value_name = "HOST:PORT", help_heading = "Several hosts")]
    coordinator: Option<String>,
    /// Run the GPU range of rank (rank + k) mod ranks, so each host reads what another wrote. Default 0.
    #[arg(long, value_name = "K", help_heading = "Several hosts")]
    rank_rotate: Option<i64>,
    /// Fail unless the run's fingerprint is this (hex, from `aeiou dry-run`).
    #[arg(long, value_name = "HEX", help_heading = "Checks")]
    expect_fingerprint: Option<String>,
    /// Fail unless every dataset id is among these.
    #[arg(long = "expect-dataset-id", value_name = "SHA256", help_heading = "Checks")]
    expect_dataset_ids: Vec<String>,
    /// Fail if an input namespace was finished more than this many seconds ago.
    #[arg(long, value_name = "SECS", help_heading = "Checks")]
    max_gap: Option<f64>,
    /// Fail if this host would read input objects it wrote itself, or if dataset pages are
    /// in its page cache at the start (mincore over 256 sampled files per dataset).
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL", help_heading = "Checks")]
    require_cold: Option<bool>,
    /// On every host, before the start gate: sync, then drop the page cache, dentries, and
    /// inodes (3 into /proc/sys/vm/drop_caches), then sample the datasets' residency. Needs
    /// root; the run refuses when it fails.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL", help_heading = "Checks")]
    drop_caches: Option<bool>,
    /// Empty the namespace roots before starting instead of refusing.
    #[arg(long, help_heading = "Checks")]
    clean_namespaces: bool,
    /// Start even when the estimated open files, threads, or mappings exceed this host's
    /// limits (RLIMIT_NOFILE, RLIMIT_NPROC, kernel.threads-max, vm.max_map_count).
    #[arg(long, help_heading = "Checks")]
    ignore_limits: bool,
    /// Write the run's report to FILE as JSON (runner/README.md §12): the configuration, the
    /// results the text report prints with the latency histograms in full, and the verdict.
    /// Written when the run fails too, with the error; rank 0 of several hosts writes the
    /// merged report, every other rank its own.
    #[arg(long, value_name = "FILE", help_heading = "Report")]
    report_json: Option<PathBuf>,
    /// With --report-json: every take of every instance (stall and compute), not only the sums.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL", help_heading = "Report")]
    report_takes: Option<bool>,
}

/// `aeiou run`'s options as resolved through the layers (`options.rs`): the command line's
/// identity, and every layered option with its value in effect.
struct RunOptions {
    run: RunArgs,
    backend: Option<String>,
    root: PathBuf,
    threads: Option<usize>,
    buffer_mib: usize,
    write_compress: u64,
    time_scale: f64,
    iowq_max_workers: Option<u32>,
    sqpoll: Option<u32>,
    sqpoll_shared: bool,
    defer_taskrun: bool,
    coop_taskrun: bool,
    aio_depth: Option<u32>,
    mmap_mode: Option<String>,
    mmap_consume: Option<String>,
    rank: i64,
    ranks: i64,
    coordinator: Option<String>,
    rank_rotate: i64,
    expect_fingerprint: Option<String>,
    expect_dataset_ids: Vec<String>,
    max_gap: Option<f64>,
    require_cold: bool,
    drop_caches: bool,
    clean_namespaces: bool,
    ignore_limits: bool,
    report_json: Option<PathBuf>,
    report_takes: bool,
}

/// The identity of the run as the command line fixes it: the abstract, its parameters, and
/// the run's `--gpus`, in every subcommand's block first.
fn fix_shape(l: &mut Layers, shape: &ShapeArgs) -> Result<()> {
    l.fixed("abstract", &shape.abstract_path, true)?;
    l.fixed("param", &shape.params, !shape.params.is_empty())?;
    l.fixed("params-file", &shape.param_files, !shape.param_files.is_empty())
}

fn resolve_run(a: RunCmd, config: Option<&Path>) -> Result<(RunOptions, Layers)> {
    let mut l = Layers::new("run", config)?;
    fix_shape(&mut l, &a.run.shape)?;
    l.fixed("gpus", &a.run.gpus, true)?;
    l.fixed("seed", &a.run.seed, a.run.seed != 0)?;
    l.fixed("io-backend", &a.backend.clone().unwrap_or_else(|| "the abstract's".into()), a.backend.is_some())?;
    l.fixed("expect-fingerprint", &a.expect_fingerprint, a.expect_fingerprint.is_some())?;
    l.fixed("expect-dataset-id", &a.expect_dataset_ids, !a.expect_dataset_ids.is_empty())?;
    l.fixed("clean-namespaces", &a.clean_namespaces, a.clean_namespaces)?;
    l.fixed("ignore-limits", &a.ignore_limits, a.ignore_limits)?;
    let root = l.layered::<PathBuf>("root", a.root, None)?.ok_or_else(|| anyhow!("--root DIR is required (the command line, $AEIOU_ROOT, or [run] root in the config file)"))?;
    let o = RunOptions {
        root,
        threads: l.layered("threads", a.threads, None)?,
        buffer_mib: l.layered("buffer-mib", a.buffer_mib, Some(8))?.unwrap_or(8),
        write_compress: l.layered("write-compress", a.write_compress, Some(1))?.unwrap_or(1),
        time_scale: l.layered("time-scale", a.time_scale, Some(1.0))?.unwrap_or(1.0),
        iowq_max_workers: l.layered("iowq-max-workers", a.iowq_max_workers, None)?,
        sqpoll: l.layered("sqpoll", a.sqpoll, None)?,
        sqpoll_shared: l.flag("sqpoll-shared", a.sqpoll_shared, false)?,
        defer_taskrun: l.flag("defer-taskrun", a.defer_taskrun, false)?,
        coop_taskrun: l.flag("coop-taskrun", a.coop_taskrun, false)?,
        aio_depth: l.layered("aio-depth", a.aio_depth, None)?,
        mmap_mode: l.layered("mmap-mode", a.mmap_mode, None)?,
        mmap_consume: l.layered("mmap-consume", a.mmap_consume, None)?,
        rank: l.layered("rank", a.rank, Some(0))?.unwrap_or(0),
        ranks: l.layered("ranks", a.ranks, Some(1))?.unwrap_or(1),
        coordinator: l.layered("coordinator", a.coordinator, None)?,
        rank_rotate: l.layered("rank-rotate", a.rank_rotate, Some(0))?.unwrap_or(0),
        max_gap: l.layered("max-gap", a.max_gap, None)?,
        require_cold: l.flag("require-cold", a.require_cold, false)?,
        drop_caches: l.flag("drop-caches", a.drop_caches, false)?,
        report_json: l.layered("report-json", a.report_json, None)?,
        report_takes: l.flag("report-takes", a.report_takes, false)?,
        run: a.run,
        backend: a.backend,
        expect_fingerprint: a.expect_fingerprint,
        expect_dataset_ids: a.expect_dataset_ids,
        clean_namespaces: a.clean_namespaces,
        ignore_limits: a.ignore_limits,
    };
    l.finish()?;
    if o.report_takes && o.report_json.is_none() {
        bail!("--report-takes needs --report-json FILE");
    }
    if o.sqpoll_shared && o.sqpoll.is_none() {
        bail!("--sqpoll-shared needs --sqpoll IDLE_MS");
    }
    if o.defer_taskrun && o.sqpoll.is_some() {
        bail!("--defer-taskrun and --sqpoll exclude each other");
    }
    Ok((o, l))
}

#[derive(Args)]
struct DryRunArgs {
    #[command(flatten)]
    run: RunArgs,
    /// Hosts the run would be spread over, for the bytes-per-host estimate.
    #[arg(long, help_heading = "Output")]
    ranks: Option<i64>,
    /// Worker threads (default: all cores).
    #[arg(long, help_heading = "Output")]
    threads: Option<usize>,
    /// Print the op stream of actor instance G (one thread, in order).
    #[arg(long, value_name = "G", help_heading = "Output")]
    gpu: Option<i64>,
    /// With --gpu: only ops whose outermost loop index is in [A, B).
    #[arg(long, value_name = "A..B", help_heading = "Output")]
    steps: Option<String>,
    /// With --gpu: stop printing after N ops.
    #[arg(long, value_name = "N", help_heading = "Output")]
    limit: Option<usize>,
    /// Compute the locality metrics of the op stream (reuse distance, sequential runs,
    /// popularity, request sizes, fan-out and depth, read/write mix); runner/README.md §10.
    #[arg(long, help_heading = "Metrics")]
    metrics: bool,
    /// With --metrics: the block size of the reuse-distance and popularity units.
    #[arg(long, value_name = "BYTES", default_value_t = 4096, help_heading = "Metrics")]
    metrics_block: u64,
    /// With --metrics: keep one block and one object in N, chosen by hash, and scale (1: exact).
    #[arg(long, value_name = "N", default_value_t = 1, help_heading = "Metrics")]
    metrics_sample: u64,
    /// Write the metrics, with their histograms, to FILE as JSON (implies --metrics).
    #[arg(long, value_name = "FILE", help_heading = "Metrics")]
    metrics_json: Option<PathBuf>,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("aeiou: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    let config = cli.config.as_deref();
    match cli.cmd {
        Cmd::Check { files } => check(files, config),
        Cmd::DryRun(a) => dry_run(a, config),
        Cmd::Datagen(a) => datagen_cmd(a, config),
        Cmd::Run(a) => run_cmd(a, config),
    }
}

fn datagen_cmd(a: DatagenArgs, config: Option<&Path>) -> Result<()> {
    let mut l = Layers::new("datagen", config)?;
    fix_shape(&mut l, &a.shape)?;
    l.fixed("gpus", &a.gpus, a.gpus != 1)?;
    l.fixed("dedupe", &a.dedupe, a.dedupe != 1)?;
    l.fixed("compress", &a.compress, a.compress != 1)?;
    l.fixed("dataset", &a.datasets, !a.datasets.is_empty())?;
    let root = l.layered::<PathBuf>("root", a.root, None)?.ok_or_else(|| anyhow!("--root DIR is required (the command line, $AEIOU_ROOT, or [datagen] root in the config file)"))?;
    let threads = l.layered::<usize>("threads", a.threads, None)?;
    l.finish()?;
    let cfg = parse_config(&a.shape, a.gpus, 0)?;
    let loaded = aeiou::load(&a.shape.abstract_path)?;
    cfg.check_sets(&loaded.ast.name, &loaded.sha256)?;
    let params = Params::new(&loaded.ast, &cfg)?;
    let mut model = build_model(&loaded.ast, &cfg, &params)?;
    model.traces = loaded.traces.clone();
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let opts = DatagenOpts { root, threads, dedupe: a.dedupe, compress: a.compress, datasets: a.datasets.clone() };
    let mut out = std::io::stdout();
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    l.print(&mut out)?;
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

fn run_cmd(a: RunCmd, config: Option<&Path>) -> Result<()> {
    let (a, layers) = resolve_run(a, config)?;
    let mut doc = aeiou::report::Doc { full_takes: a.report_takes, ..Default::default() };
    if let Some(path) = &a.report_json {
        // never leave an earlier run's report where this run's is expected
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => bail!("--report-json {}: {e}", path.display()),
            _ => {}
        }
    }
    let r = run_checked(&a, &layers, &mut doc);
    if let Some(path) = &a.report_json {
        let error = r.as_ref().err().map(|e| format!("{e:#}"));
        match (doc.write(path, error.as_deref()), &r) {
            (Ok(()), _) => println!("report {}", path.display()),
            (Err(e), Ok(())) => return Err(e),
            (Err(e), Err(_)) => eprintln!("aeiou: {e:#}"),
        }
    }
    r
}

fn run_checked(a: &RunOptions, layers: &Layers, doc: &mut aeiou::report::Doc) -> Result<()> {
    let cfg = parse_config(&a.run.shape, a.run.gpus, a.run.seed)?;
    // the run is the process: the abstract and the model live for the threads' lifetime
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load(&a.run.shape.abstract_path)?));
    let declared = loaded.ast.backend.as_deref().unwrap_or("sync");
    let backend_name = a.backend.as_deref().unwrap_or(declared);
    let backend = BackendKind::parse(backend_name).ok_or_else(|| anyhow::anyhow!("--io-backend {}: not one of {}", backend_name, aeiou::backend::NAMES))?;
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
    if a.threads.is_some() && !backend.event_loop() {
        bail!("--threads sets the event-loop threads of io_uring and libaio; --io-backend {} runs one thread per actor", backend.name());
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
    cfg.check_sets(&loaded.ast.name, &loaded.sha256)?;
    let cfg: &'static Config = Box::leak(Box::new(cfg));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg)?));
    let model = Box::leak(Box::new(build_model(&loaded.ast, cfg, params)?));
    model.traces = loaded.traces.clone();
    // the run's identity first, so the report of a run that fails a check still says which run
    doc.set("abstract", serde_json::json!({"name": loaded.ast.name, "sha256": loaded.sha256}));
    doc.set("seed", serde_json::json!(cfg.seed));
    doc.set("gpus", serde_json::json!(cfg.gpus));
    doc.set("params", payload::params_json(&loaded.doc, cfg, params)?);
    doc.set("backend", serde_json::json!(backend.name()));
    doc.set("backend_declared", serde_json::json!(declared));
    doc.set("host", serde_json::json!(run::hostname()));
    doc.set("rank", serde_json::json!(a.rank));
    doc.set("ranks", serde_json::json!(a.ranks));
    doc.set("layers", layers.json());
    doc.expected_fingerprint = expect_fingerprint;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    layers.print(&mut out)?;
    writeln!(out, "seed {}  gpus {}  params: {}", cfg.seed, cfg.gpus, params_line(cfg))?;
    if backend.name() != declared {
        writeln!(out, "backend {} is not the abstract's ({declared}): the storage sees another workload, and this run is not comparable with runs of the abstract as declared", backend.name())?;
    }
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
    // the `trace` nodes (`DESIGN_REVIEW.md` §3.58): V16 with the resolved counts, and what
    // each file is, before any host is kept waiting; the paths under --root are checked
    // with the namespaces below
    let traces: Vec<&Arc<aeiou::trace::TraceFile>> = {
        let mut v: Vec<_> = loaded.traces.values().collect();
        v.sort_by(|x, y| x.name.cmp(&y.name));
        v
    };
    if !traces.is_empty() {
        let counts = aeiou::vm::actor_counts(model)?;
        aeiou::trace::check_counts(&loaded.ast, &loaded.traces, &counts)?;
        for t in &traces {
            writeln!(out, "trace {}: {} line(s) on {} lane(s), {} open(s), {} path(s) created; sha256 {}; never CLOSED", t.name, t.lines.len(), t.lanes(), t.opens.len(), t.header.creates.len(), &t.sha256[..16])?;
        }
    }
    doc.set("traces", serde_json::json!(traces.iter().map(|t| serde_json::json!({"file": t.name, "sha256": t.sha256, "lanes": t.lanes(), "lines": t.lines.len(), "ops": t.ops, "opens": t.opens.len(), "creates": t.header.creates.len(), "notes": t.header.notes})).collect::<Vec<_>>()));
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

    // the limits, before any other host is kept waiting: soft limits raised to the hard ones,
    // the estimated open files and threads against them
    let loops = if backend.event_loop() { run::loop_count(model, &opts)? } else { 0 };
    let limits = aeiou::limits::check(model, &opts, loops, a.ignore_limits)?;
    for line in limits.lines() {
        writeln!(out, "{line}")?;
    }
    for p in limits.problems() {
        writeln!(out, "limits: IGNORED: {p}")?;
    }

    doc.set(
        "options",
        serde_json::json!({
            "root": a.root,
            "threads": a.threads,
            "buffer_bytes": opts.buffer_bytes,
            "write_compress": opts.write_compress,
            "time_scale": opts.time_scale,
            "io_uring": backend.uring().then_some(&opts.uring),
            "aio_depth": a.aio_depth,
            "mmap_mode": (backend == BackendKind::Mmap).then(|| mmap.name()),
            "mmap_consume": (backend == BackendKind::Mmap).then(|| mmap_consume.name()),
            "clean_namespaces": a.clean_namespaces,
            "rank_rotate": a.rank_rotate,
            "max_gap": a.max_gap,
            "require_cold": a.require_cold,
            "drop_caches": a.drop_caches,
            "ignore_limits": a.ignore_limits,
        }),
    );
    doc.set("gpu_ids", serde_json::json!([lo, hi]));
    let datasets: Vec<serde_json::Value> = checks.iter().map(|c| serde_json::json!({"name": c.name, "root": c.root, "id": c.id, "payload": c.payload, "files": c.files})).collect();
    doc.set("datasets", serde_json::json!(datasets));
    doc.set("limits", serde_json::json!({"checked": limits, "ignored": limits.problems()}));

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
            let identity = serde_json::json!({
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
            let hash = aeiou::canon::sha256_hex(&identity);
            let t = Arc::new(Tcp::connect(addr, a.rank, a.ranks, &run::hostname(), &identity, &layers.json(), &participants, aborted.clone())?);
            writeln!(out, "coordinator {}: connected as rank {} of {}  identity {}…", t.addr, a.rank, a.ranks, &hash[..16])?;
            (t.clone(), Some(t))
        }
        _ => (Arc::new(Local::new(&participants)), None),
    };
    let r = run_connected(a, loaded, cfg, model, opts, coord, aborted, tcp.as_deref(), server.as_ref(), &mut out, doc);
    if let (Err(e), Some(t)) = (&r, &tcp) {
        // the coordinator relays a failure here to every other host (a no-op after a Stop)
        t.stop(&format!("{e:#}"));
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn run_connected(
    a: &RunOptions,
    loaded: &'static aeiou::Loaded,
    cfg: &'static Config,
    model: &'static aeiou::eval::Model<'static>,
    opts: RunOpts,
    coord: Arc<dyn Coordinator>,
    aborted: Arc<AtomicBool>,
    tcp: Option<&Tcp>,
    server: Option<&Server>,
    out: &mut impl Write,
    doc: &mut aeiou::report::Doc,
) -> Result<()> {
    let (ns_checks, input_objects) = run::check_input_namespaces(loaded, cfg, &a.root, &opts)?;
    let inputs: Vec<serde_json::Value> = ns_checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "names": c.names,
                "root": c.root,
                "writer_abstract": c.writer_abstract,
                "writer_sha256": c.writer_sha256,
                "writer_hosts": c.writer_hosts,
                "gap_s": c.gap,
                "objects": c.objects,
                "same_host": c.same_host,
            })
        })
        .collect();
    doc.set("input_namespaces", serde_json::json!(inputs));
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
    let traces: Vec<&Arc<aeiou::trace::TraceFile>> = {
        let mut v: Vec<_> = loaded.traces.values().collect();
        v.sort_by(|x, y| x.name.cmp(&y.name));
        v
    };
    if a.rank == 0 {
        let cleaned = run::prepare_namespaces(&loaded.ast, &a.root, a.clean_namespaces)?;
        for c in &cleaned {
            writeln!(out, "namespace root {c}/ emptied")?;
        }
        if a.clean_namespaces {
            for t in traces.iter() {
                let n = aeiou::trace::clean(t, &a.root)?;
                if n > 0 {
                    writeln!(out, "trace {}: {n} created path(s) removed", t.name)?;
                }
            }
        }
    }
    for t in traces.iter() {
        let c = aeiou::trace::check_root(t, &a.root)?;
        writeln!(out, "trace {}: {} input path(s) present under --root, {} to be created absent", c.file, c.inputs, c.creates)?;
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
        if let Some(s) = server {
            // every host's options block came with its Hello: the report records them all,
            // and the options that differ between hosts are printed (rank always does)
            let all = s.hosts();
            doc.set("hosts", serde_json::json!(all.iter().map(|(r, h, l)| serde_json::json!({"rank": r, "host": h, "layers": l})).collect::<Vec<_>>()));
            let refs: Vec<(i64, &serde_json::Value)> = all.iter().map(|(r, _, l)| (*r, l)).collect();
            let rows = options::differences(&refs);
            writeln!(out, "options differing between hosts ({}; rank always does):", if rows.len() > 1 { "layered values are each host's own" } else { "none but rank" })?;
            for (name, values) in &rows {
                writeln!(out, "  {name}: {}", values.iter().map(|(r, v)| format!("rank {r} = {v}")).collect::<Vec<_>>().join(", "))?;
            }
        }
        out.flush()?;
    }
    let mut report = run::run_with(model, opts.clone(), input_objects, coord, aborted)?;
    report.cold = vec![cold];
    let finished = run::unix_now();
    doc.started = Some(started);
    doc.finished = Some(finished);
    doc.host = Some(aeiou::report::result(&report, doc.full_takes)?);
    doc.fingerprint = Some(report.stats.fingerprint);
    doc.whole_run = tcp.is_none();
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
            doc.fingerprint = Some(fp);
            doc.whole_run = true;
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
            doc.merged = Some(aeiou::report::result(&merged, doc.full_takes)?);
            doc.fingerprint = Some(merged.stats.fingerprint);
            doc.whole_run = true;
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

fn check(files: Vec<PathBuf>, config: Option<&Path>) -> Result<()> {
    if files.is_empty() {
        bail!("no files given");
    }
    let mut l = Layers::new("check", config)?;
    l.fixed("files", &files, true)?;
    l.finish()?;
    l.print(&mut std::io::stdout())?;
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

fn parse_config(a: &ShapeArgs, gpus: i64, seed: u64) -> Result<Config> {
    if gpus < 1 {
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
    Ok(Config { seed, gpus, overrides, sets })
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

fn dry_run(a: DryRunArgs, config: Option<&Path>) -> Result<()> {
    let mut l = Layers::new("dry-run", config)?;
    fix_shape(&mut l, &a.run.shape)?;
    l.fixed("gpus", &a.run.gpus, true)?;
    l.fixed("seed", &a.run.seed, a.run.seed != 0)?;
    l.fixed("gpu", &a.gpu, a.gpu.is_some())?;
    l.fixed("steps", &a.steps, a.steps.is_some())?;
    l.fixed("limit", &a.limit, a.limit.is_some())?;
    l.fixed("metrics", &(a.metrics || a.metrics_json.is_some()), a.metrics)?;
    l.fixed("metrics-block", &a.metrics_block, a.metrics_block != 4096)?;
    l.fixed("metrics-sample", &a.metrics_sample, a.metrics_sample != 1)?;
    l.fixed("metrics-json", &a.metrics_json, a.metrics_json.is_some())?;
    let ranks = l.layered::<i64>("ranks", a.ranks, None)?;
    let threads = l.layered::<usize>("threads", a.threads, None)?;
    l.finish()?;
    let cfg = parse_config(&a.run.shape, a.run.gpus, a.run.seed)?;
    let loaded = aeiou::load(&a.run.shape.abstract_path)?;
    cfg.check_sets(&loaded.ast.name, &loaded.sha256)?;
    let params = Params::new(&loaded.ast, &cfg)?;
    let mut model = build_model(&loaded.ast, &cfg, &params)?;
    model.traces = loaded.traces.clone();

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
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let started = std::time::Instant::now();
    if a.metrics_block == 0 || a.metrics_sample == 0 {
        bail!("--metrics-block and --metrics-sample must be at least 1");
    }
    let metrics = (a.metrics || a.metrics_json.is_some()).then_some(aeiou::metrics::Opts { block: a.metrics_block, sample: a.metrics_sample });
    let mut report = dryrun::run_with(&model, threads, filter, metrics)?;
    let elapsed = started.elapsed();
    if let Some(path) = &a.metrics_json {
        let templates: std::collections::BTreeMap<&str, _> = report.templates.iter().filter_map(|t| aeiou::metrics::to_json(&t.run).map(|m| (t.name.as_str(), m))).collect();
        let doc = serde_json::json!({
            "aeiou_metrics": 1,
            "source": "dry-run",
            "abstract": loaded.ast.name,
            "sha256": loaded.sha256,
            "seed": cfg.seed,
            "gpus": cfg.gpus,
            "fingerprint": format!("{:016x}", report.total.fingerprint),
            "block": a.metrics_block,
            "sample": a.metrics_sample,
            "order": "round-robin",
            "templates": templates,
            "total": aeiou::metrics::to_json(&report.total),
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)? + "\n").with_context(|| format!("{}", path.display()))?;
    }

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    writeln!(out, "abstract {}  sha256 {}", loaded.ast.name, loaded.sha256)?;
    l.print(&mut out)?;
    if let Some(b) = &loaded.ast.backend {
        writeln!(out, "declared backend {b}")?;
    }
    writeln!(out, "seed {}  gpus {}  params: {}", cfg.seed, cfg.gpus, params_line(&cfg))?;
    for line in report.total.take_lines() {
        writeln!(out, "{line}")?;
    }
    for t in &mut report.templates {
        t.run.take_lines();
    }
    dryrun::write_report(&mut out, &report, ranks, cfg.gpus)?;
    writeln!(out, "dry-run took {:.2?} on {} thread(s)", elapsed, threads)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// `options::ALL_OPTIONS` is the set of long flags (and positionals) of every subcommand.
    #[test]
    fn all_options_lists_every_flag() {
        let cmd = Cli::command();
        let mut names = std::collections::BTreeSet::new();
        let mut collect = |c: &clap::Command| {
            for a in c.get_arguments() {
                if a.get_id() == "help" || a.get_id() == "version" {
                    continue;
                }
                names.insert(match a.get_long() {
                    Some(l) => l.to_string(),
                    None => a.get_id().as_str().trim_end_matches("_path").to_string(),
                });
            }
        };
        collect(&cmd);
        for sub in cmd.get_subcommands() {
            collect(sub);
        }
        let listed: std::collections::BTreeSet<String> = options::ALL_OPTIONS.iter().map(|s| s.to_string()).collect();
        assert_eq!(names, listed);
    }
}
