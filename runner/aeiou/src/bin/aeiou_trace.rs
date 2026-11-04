//! `aeiou-trace`: the trace side of the suite (`builder/REFERENCE.md` §7, `DESIGN_REVIEW.md`
//! §3.42, §3.89): the locality metrics of a real application from an strace of it, an exported
//! trace file, or an object-store request log; the runner's trace file for a `trace` node; the
//! structural digest a fit is made from (`aeiou::digest`), with its interview, the application
//! records, and the answers file (`aeiou::interview`, `aeiou::applications`); and the comparison
//! of the application's document with the abstract's, two digests' structure and any two
//! documents' numbers (`aeiou::compare`, §3.89 stage 4); and the capture that starts the
//! application under a trace and keeps the record beside it (`aeiou::capture`, §3.96).

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use serde_json::{json, Value};

use aeiou::usage::{self, Missing, UsageError};
use aeiou::{applications, capture, compare, digest, export, interview, objectlog, strace};

const ABOUT: &str = "Locality metrics of a real application from an strace of it, the runner's trace file, the structural digest a fit is made from, the comparison of the application's with its abstract's (the digest of aeiou dry-run --emit-trace, or aeiou dry-run --metrics-json), and the capture of the trace with its record";
const EXIT_PIPE: i32 = 141;

#[derive(Parser)]
#[command(name = "aeiou-trace", version, about = ABOUT, override_usage = "aeiou-trace <COMMAND> [ARGS]...", max_term_width = 120)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compute the metrics of a trace (strace -f --seccomp-bpf -ttt -T -yy -v -e 'verbose=!execve,execveat' -s 0 -e trace=%file,%desc,%process,sync,msync,madvise,munmap,mremap,mlock,mlock2,io_setup,io_submit,io_getevents,io_pgetevents,io_cancel,io_destroy -o FILE)
    #[command(override_usage = "aeiou-trace metrics [OPTIONS] <TRACE>")]
    Metrics(MetricsArgs),
    /// Write the runner's trace file for a `trace` node from an strace (DESIGN_REVIEW.md §3.58): JSON Lines, one lane per traced task, opens by id, paths relative to --root
    #[command(override_usage = "aeiou-trace export [OPTIONS] --root <DIR> -o <FILE> <TRACE>")]
    Export(ExportArgs),
    /// Compare the application's digest or metrics with the abstract's: structure, then numbers
    #[command(override_usage = "aeiou-trace compare [OPTIONS] <A> <B>")]
    Compare(CompareArgs),
    /// Write the structural digest of a trace: path templates, lanes and their phases, a verdict per site on what the trace decides (permutation or draw, order, size), and the open questions ranked by their divergence at scale
    #[command(override_usage = "aeiou-trace digest [OPTIONS] --root <DIR> <TRACE>")]
    Digest(DigestArgs),
    /// Start COMMAND under a trace (strace -f --seccomp-bpf ...) after checking this node (and, with --ranks, every node), and write the capture directory: trace.txt, the trace every subcommand reads, capture.json, the record of what was captured and how, and with an --endpoint requests.jsonl, the object requests the capture's library saw
    #[command(override_usage = "aeiou-trace capture [OPTIONS] -o <DIR> -- <COMMAND>...")]
    Capture(CaptureArgs),
}

#[derive(clap::Args)]
struct CaptureArgs {
    /// What traces the system calls: strace (any user, slower, loses nothing), ebpf (root or the file capabilities cap_bpf,cap_perfmon; can lose events when its --buffer fills), auto (eBPF where granted, else strace; one host only) [default: auto]
    #[arg(long, value_name = "SOURCE", value_parser = ["auto", "ebpf", "strace"])]
    source: Option<String>,
    /// A directory the application's data is under (repeatable): its mount is recorded and compared across nodes. At least one --root or --endpoint
    #[arg(long, value_name = "DIR")]
    root: Vec<String>,
    /// An object store the application talks to, as a URL (s3://BUCKET[/PREFIX], http://HOST:PORT[/BUCKET]; repeatable): recorded with its scheme, sent an anchor HEAD at the start and the end when http(s)
    #[arg(long, value_name = "URL")]
    endpoint: Vec<String>,
    /// Sync and drop the page cache (3 into /proc/sys/vm/drop_caches, root only) before the command starts; recorded either way
    #[arg(long)]
    cold: bool,
    /// shim: preload the capture's library into the command, recording the heads of its HTTP requests and responses (a shared libssl's or libc's socket calls) into requests.jsonl; none: no library [default: shim with an --endpoint, else none]
    #[arg(long, value_name = "MODE", value_parser = ["shim", "none"])]
    requests: Option<String>,
    /// The size of the library's log, sparse (KiB, MiB, GiB); a request and its response take about half a KiB [default: 1GiB]
    #[arg(long, value_name = "SIZE")]
    requests_buffer: Option<String>,
    /// The eBPF source's ring buffer, a power of two from 64KiB to 2GiB: a full one loses events, and the capture says so and suggests a larger one [default: 256MiB]
    #[arg(long, value_name = "SIZE")]
    buffer: Option<String>,
    /// What capture.json keeps of the command line: recorded, every argument, the value of an option or a variable named as a secret (password, token, key, auth, ...) and a URL's password replaced by <redacted>; none, the program alone, for a secret passed by position [default: recorded]
    #[arg(long, value_name = "MODE", value_parser = ["recorded", "none"])]
    argv: Option<String>,
    /// Start although CHECK failed (repeatable; one of the checks by name); printed and recorded
    #[arg(long, value_name = "CHECK")]
    accept: Vec<String>,
    /// Run the checks (with --ranks, on every node and across them) and start nothing; -o is checked and not made
    #[arg(long)]
    check: bool,
    /// Write DIR/trace.txt again from DIR/syscalls.bin, the eBPF source's records, with this aeiou-trace's converter, and update DIR/capture.json; starts nothing, takes no other option
    #[arg(long, value_name = "DIR", conflicts_with_all = ["out", "check", "command", "root", "endpoint"])]
    convert: Option<String>,
    /// The capture this node joins (the same on every node); default $SLURM_JOB_ID when set. Captures of one id may share a host
    #[arg(long, value_name = "ID")]
    capture_id: Option<String>,
    /// Number of nodes of the capture; with more than 1, --rank and --coordinator too (aeiou-launch appends all three)
    #[arg(long, value_name = "N")]
    ranks: Option<i64>,
    /// This node's index, 0 to N-1; rank 0 runs the coordinator, compares the nodes, and writes capture-set.json
    #[arg(long, value_name = "R")]
    rank: Option<i64>,
    /// With --ranks above 1: the coordinator's address. Rank 0 listens on it (in-process); every rank connects to it
    #[arg(long, value_name = "HOST:PORT")]
    coordinator: Option<String>,
    /// Required: the capture directory, new or empty, on a node-local disk outside every --root; %h expands to the host, %p to the pid
    #[arg(short, long, value_name = "DIR")]
    out: Option<String>,
    /// After --: the command and its arguments, started under the trace as the invoking user
    #[arg(last = true, value_name = "COMMAND")]
    command: Vec<String>,
}

#[derive(clap::Args)]
struct MetricsArgs {
    /// The strace output file, an exported trace file, an object-store request log (MinIO audit, S3 server access, a capture's requests.jsonl), a capture directory, or - for stdin
    trace: Option<String>,
    /// Count only calls on paths under DIR (repeatable; required for an strace, not for an exported trace file)
    #[arg(long, value_name = "DIR")]
    root: Vec<String>,
    /// Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)
    #[arg(long, value_name = "GLOB")]
    exclude: Vec<String>,
    /// Resolve relative paths against DIR when the trace does not say (no -y)
    #[arg(long, value_name = "DIR")]
    cwd: Option<String>,
    /// The block metrics too, reuse and popularity in blocks of BYTES, as captures taken before 2026-10-22 have them (the extent metrics are always there)
    #[arg(long, value_name = "BYTES")]
    metrics_block: Option<i64>,
    /// The process tree under PID is one instance (repeatable; needs clone in the trace); default: the whole trace is one instance
    #[arg(long, value_name = "PID")]
    instance_root: Vec<i64>,
    /// Report depth: consecutive io_submit rounds of a thread form a chain until more than US microseconds pass between a round's end and the next submit
    #[arg(long, value_name = "US", value_parser = non_negative)]
    chain_gap_us: Option<f64>,
    /// Write the JSON document (aeiou_metrics: 2) to FILE instead of stdout
    #[arg(short, long, value_name = "FILE")]
    out: Option<PathBuf>,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// The strace output file, an object-store request log (MinIO audit, S3 server access, a capture's requests.jsonl), a capture directory, or - for stdin
    trace: Option<String>,
    /// Required: export only calls on paths under DIR, written relative to it
    #[arg(long, value_name = "DIR")]
    root: Option<String>,
    /// Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)
    #[arg(long, value_name = "GLOB")]
    exclude: Vec<String>,
    /// Resolve relative paths against DIR when the trace does not say (no -y)
    #[arg(long, value_name = "DIR")]
    cwd: Option<String>,
    /// Required: the trace file to write; its sha256 is printed, for the abstract's `trace` node
    #[arg(short, long, value_name = "FILE")]
    out: Option<PathBuf>,
}

#[derive(clap::Args)]
struct DigestArgs {
    /// The strace output file, an exported trace file, an object-store request log (MinIO audit, S3 server access, a capture's requests.jsonl), or a capture directory; a file, never stdin: the comparison after the fit and a re-run with answers read it again
    trace: Option<String>,
    /// Digest only calls on paths under DIR, written relative to it (required for an strace, not for an exported trace file)
    #[arg(long, value_name = "DIR")]
    root: Option<String>,
    /// Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)
    #[arg(long, value_name = "GLOB")]
    exclude: Vec<String>,
    /// Resolve relative paths against DIR when the trace does not say (no -y)
    #[arg(long, value_name = "DIR")]
    cwd: Option<String>,
    /// Project the open questions over N passes over the set (default 1): the share a draw with replacement never reads is e^-N
    #[arg(long, value_name = "N", allow_negative_numbers = true)]
    epochs: Option<i64>,
    /// The files the application could have chosen among (the corpus); default: the distinct paths the trace touched. Decides a pass from a draw when no path repeats
    #[arg(long, value_name = "N", allow_negative_numbers = true)]
    candidates: Option<i64>,
    /// Include every raw path under each template (the digest otherwise carries templates and counts only)
    #[arg(long)]
    paths: bool,
    /// The block metrics too in the metrics section, reuse and popularity in blocks of BYTES, as captures taken before 2026-10-22 have them
    #[arg(long, value_name = "BYTES", allow_negative_numbers = true)]
    metrics_block: Option<i64>,
    /// The metrics section: the process tree under PID is one instance (repeatable; an strace's alone, and needs clone in it); default: the whole trace is one instance
    #[arg(long, value_name = "PID")]
    instance_root: Vec<i64>,
    /// The metrics section reports depth: consecutive io_submit rounds of a thread form a chain until more than US microseconds pass between a round's end and the next submit (an strace's alone)
    #[arg(long, value_name = "US", value_parser = non_negative)]
    chain_gap_us: Option<f64>,
    /// Print the text report instead of the JSON document (with -o, the report is printed beside the file anyway)
    #[arg(long)]
    text: bool,
    /// Write the JSON document (aeiou_digest: 2) to FILE and print the text report
    #[arg(short, long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// The answers file (aeiou_answers: 1): replayed when it exists, so the questions are not asked; written there after an interactive run. Default after an interactive run with -o: the document's name with .answers.json
    #[arg(long, value_name = "FILE")]
    answers: Option<String>,
    /// Ask nothing even on a terminal: every undecided verdict stays open
    #[arg(long)]
    no_questions: bool,
    /// Record NAME as the one who answered (default: user@host)
    #[arg(long, value_name = "NAME")]
    answered_by: Option<String>,
    /// The application records (default: the ones built into the tool, the checkout's applications/ it was built from)
    #[arg(long, value_name = "DIR")]
    applications: Option<String>,
    /// State the traced application instead of being asked: its record's defaults and assertions apply, with the record as their provenance
    #[arg(long, value_name = "NAME[@VERSION]")]
    application: Option<String>,
}

#[derive(clap::Args)]
struct CompareArgs {
    /// The application's document: a digest (aeiou_digest: 2), or a metrics document (aeiou_metrics: 2, or 1 from before 2026-10-22: numbers only)
    a: Option<String>,
    /// The document A is compared with, the abstract's: the digest of its aeiou dry-run --emit-trace, or its aeiou dry-run --metrics-json
    b: Option<String>,
    /// Use this actor template's metrics of A instead of its total
    #[arg(long, value_name = "NAME")]
    template_a: Option<String>,
    /// Use this actor template's metrics of B instead of its total
    #[arg(long, value_name = "NAME")]
    template_b: Option<String>,
    /// Exit 1 if any distance exceeds X (one number for every row; --judge is the rule by class)
    #[arg(long, value_name = "X", value_parser = non_negative)]
    max_distance: Option<f64>,
    /// Judge each row against the tolerance of its class (A the trace, B the abstract) and exit 1 if any is outside
    #[arg(long)]
    judge: bool,
    /// A tolerance file (aeiou_tolerances: 1): class tolerances that replace the defaults, the rows the trace cannot show, the rows recorded as outside; implies --judge
    #[arg(long, value_name = "FILE")]
    tolerances: Option<String>,
    /// A metrics document of B's abstract at the same parameters and another seed (repeatable): a row's tolerance is raised by its largest distance between B and these; implies --judge
    #[arg(long = "self", value_name = "FILE")]
    selfs: Vec<String>,
    /// Only the rows whose metric starts with PREFIX (repeatable)
    #[arg(long, value_name = "PREFIX")]
    only: Vec<String>,
    /// Pair A's template A with B's template B, written as the digests write them, when neither their text nor their shape pairs them (repeatable)
    #[arg(long, value_name = "A=B")]
    pair: Vec<String>,
    /// Print the comparison as one JSON document (aeiou_compare: 1) instead of the text
    #[arg(long)]
    json: bool,
}

/// A finite number at least 0 (`usage.non_negative` of the Python tools).
fn non_negative(s: &str) -> std::result::Result<f64, String> {
    match s.parse::<f64>() {
        Ok(x) if x.is_finite() && x >= 0.0 => Ok(x),
        _ => Err("not a finite number at least 0".into()),
    }
}

/// An OS error as the suite prints one (`No such file or directory (os error 2)`).
fn cause(e: &std::io::Error) -> String {
    e.to_string()
}

/// The trace, `-` for stdin spooled to a file: an export and an object log are read twice.
struct Input {
    path: PathBuf,
    spooled: bool,
}

impl Input {
    fn open(name: &str) -> Result<Input> {
        if name != "-" {
            File::open(name).map_err(|e| anyhow!("{name}: {}", cause(&e)))?;
            return Ok(Input { path: PathBuf::from(name), spooled: false });
        }
        let path = std::env::temp_dir().join(format!("aeiou-trace-stdin-{}", std::process::id()));
        let mut f = File::create(&path).map_err(|e| anyhow!("{}: {}", path.display(), cause(&e)))?;
        std::io::copy(&mut std::io::stdin().lock(), &mut f)?;
        Ok(Input { path, spooled: true })
    }

    fn reader(&self) -> std::io::Result<Box<dyn BufRead>> {
        Ok(Box::new(BufReader::with_capacity(1 << 20, File::open(&self.path)?)))
    }

    fn first_line(&self) -> Result<String> {
        let mut r = self.reader()?;
        let mut buf = Vec::new();
        r.read_until(b'\n', &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        if self.spooled {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// A capture directory (`aeiou-trace capture -o DIR`) read in place of a trace: its trace.txt
/// when the roots are directories, its requests.jsonl when they name a bucket; the command's
/// working directory the default `--cwd`. Anything else is read as it is.
fn in_capture(trace: String, roots: &[&str], cwd: Option<String>) -> Result<(String, Option<String>)> {
    let dir = std::path::Path::new(&trace);
    if !dir.is_dir() {
        return Ok((trace, cwd));
    }
    let rec: Value = match std::fs::read_to_string(dir.join("capture.json")) {
        Ok(t) => serde_json::from_str(&t).map_err(|e| anyhow!("{trace}/capture.json: {e}"))?,
        Err(_) => return Err(anyhow!("{trace}: a directory, and not a capture's (no capture.json)")),
    };
    let is_dir = |r: &&&str| std::path::Path::new(r).is_absolute() || r.starts_with('.') || std::path::Path::new(r).is_dir();
    let dirs = roots.iter().filter(is_dir).count();
    let file = if dirs == roots.len() {
        "trace.txt"
    } else if dirs == 0 {
        aeiou::requests::FILE
    } else {
        return Err(usage::err(format!("{trace}: a capture directory's roots are all directories (its trace.txt) or all one bucket (its requests.jsonl)")));
    };
    let path = dir.join(file);
    if !path.exists() {
        return Err(anyhow!("{}: no such file: {}", path.display(), if file == "trace.txt" { "not a whole capture" } else { "captured without the library (--requests none)" }));
    }
    let cwd = cwd.or_else(|| rec["command"]["cwd"].as_str().map(str::to_string));
    Ok((path.to_string_lossy().into_owned(), cwd))
}

fn write_out(path: &PathBuf, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|e| anyhow!("{}: {}", path.display(), cause(&e)))
}

/// Standard output, written whole; a reader that went away ends the tool quietly, as the runner.
fn print_out(text: &str) {
    let mut out = std::io::stdout().lock();
    if let Err(e) = out.write_all(text.as_bytes()).and_then(|_| out.flush()) {
        if e.kind() == std::io::ErrorKind::BrokenPipe {
            std::process::exit(EXIT_PIPE);
        }
        eprintln!("aeiou-trace: standard output: {e}");
        std::process::exit(1);
    }
}

fn notes_lines(notes: &Value) -> String {
    notes.as_object().map(|m| m.iter().map(|(k, v)| format!("note: {k}: {}\n", v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect()).unwrap_or_default()
}

/// The warning that a trace is incomplete (`DESIGN_REVIEW.md` §3.92): descriptors used without
/// the open that made them, `missing` the opens the export synthesized (`missing_opens`), or,
/// where the tool exported nothing, their count alone. Printed to stderr; the tool goes on.
fn incomplete(prog: &str, count: u64, missing: Option<&Value>, what: &str) -> String {
    if count == 0 {
        return String::new();
    }
    let mut out = format!(
        "{prog}: WARNING: the trace is incomplete: {count} descriptor(s) were used without the open that made them, held from before strace began (attached with strace -p to a running process, inherited from the shell, or received over a socket, which needs %network in -e trace=); {what}\n"
    );
    for m in missing.and_then(Value::as_array).into_iter().flatten() {
        out += &format!("  {} (first used by {} in lane {})\n", m["path"].as_str().unwrap_or(""), m["first"].as_str().unwrap_or(""), m["lane"]);
    }
    out + "  trace the command from its start: strace -f --seccomp-bpf ... -o FILE COMMAND\n"
}

/// The warning that a trace prints what the capture's line keeps out, where a secret can be
/// (brief item 62): from a metrics document's `notes`. Printed to stderr; the tool goes on.
fn secrets(prog: &str, notes: &Value) -> String {
    let n = |k: &str| notes.get(k).and_then(Value::as_u64).unwrap_or(0);
    let (args, data) = (n(strace::ARGUMENTS_PRINTED), n(strace::DATA_PRINTED));
    if args + data == 0 {
        return String::new();
    }
    format!("{prog}: WARNING: this trace may hold secrets: {args} execve(s) with their arguments and environment, {data} call(s) with their bytes; {}\n", strace::SECRETS)
}

/// The warning for a header (an export's, a trace file's) that lists synthesized opens.
fn incomplete_of(prog: &str, header: &Value) -> String {
    let missing = header.get("missing_opens");
    let n = missing.and_then(Value::as_array).map_or(0, Vec::len) as u64;
    incomplete(prog, n, missing, "an open was synthesized for each at its first use, its flags its use, its position 0, so its calls are replayed:")
}

fn missing_object_root(kind: &str) -> anyhow::Error {
    let mut m = Missing::new();
    m.need(false, "--root <BUCKET[/PREFIX]>", "the bucket, and the prefix under it, whose requests count", Some(&format!("needed for an object log ({kind})")));
    m.check().unwrap_err()
}

fn metrics(a: MetricsArgs) -> Result<()> {
    let mut m = Missing::new();
    let trace = m.want(a.trace, "<TRACE>", "the strace output file, or - for stdin", None);
    m.check()?;
    let roots: Vec<&str> = a.root.iter().map(String::as_str).collect();
    let (trace, cwd) = in_capture(trace.unwrap(), &roots, a.cwd.clone())?;
    if a.metrics_block.is_some_and(|b| b < 1) {
        return Err(usage::err("--metrics-block must be at least 1"));
    }
    let block = a.metrics_block.map(|b| b as u64);
    let input = Input::open(&trace)?;
    let first = input.first_line()?;
    let opener = || input.reader();
    let mut doc = if first.starts_with("{\"aeiou_trace\"") {
        let d = export::metrics_of_file(input.reader()?, block).map_err(|e| anyhow!("{trace}: {e:#}"))?;
        if d["total"]["ops"].as_u64() == Some(0) {
            return Err(anyhow!("{trace}: a trace file with no call in it"));
        }
        d
    } else if let Some(kind) = objectlog::source_of(&first) {
        match a.root.len() {
            0 => return Err(missing_object_root(kind)),
            1 => {}
            n => return Err(anyhow!("{trace}: an object log takes one --root BUCKET[/PREFIX], not {n}")),
        }
        let (d, requests) = objectlog::metrics(&opener, &a.root[0], kind, block).map_err(|e| anyhow!("{trace}: {e:#}"))?;
        if d["total"]["ops"].as_u64() == Some(0) {
            return Err(anyhow!("{trace}: no request under --root {} ({requests} request(s) read)", a.root[0].trim_matches('/')));
        }
        d
    } else {
        if a.root.is_empty() {
            let mut m = Missing::new();
            m.need(false, "--root <DIR>", "count only calls on paths under DIR (repeatable)", Some("needed for an strace; an exported trace file carries its root"));
            m.check()?;
        }
        let o = strace::Opts {
            roots: a.root.iter().map(|r| strace::abspath(r)).collect(),
            exclude: a.exclude,
            cwd,
            instance_roots: a.instance_root,
            chain_gap_us: a.chain_gap_us,
            block,
        };
        let (d, seen) = strace::metrics(input.reader()?, &o).map_err(|e| anyhow!("{trace}: {e:#}"))?;
        if d["total"]["ops"].as_u64() == Some(0) {
            let roots: Vec<String> = d["roots"].as_array().map(|r| r.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            return Err(anyhow!("{trace}: {}", strace::nothing_counted(&roots, seen)));
        }
        d
    };
    doc["trace"] = json!(trace);
    let held = doc["notes"]["descriptors_without_open"].as_u64().unwrap_or(0);
    eprint!("{}", incomplete("aeiou-trace metrics", held, None, "their calls are counted from position 0, the open's own work not at all"));
    let text = serde_json::to_string_pretty(&doc)? + "\n";
    match &a.out {
        Some(out) => {
            write_out(out, &text)?;
            let t = &doc["total"];
            let roots: Vec<&str> = doc["roots"].as_array().map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            print_out(&format!(
                "{} ops under {}: read {} bytes, wrote {} bytes; {} thread(s); {}\n{}",
                t["ops"],
                roots.join(", "),
                t["bytes_read"],
                t["bytes_written"],
                doc["contexts"],
                out.display(),
                notes_lines(&doc["notes"])
            ));
        }
        None => print_out(&text),
    }
    eprint!("{}", secrets("aeiou-trace metrics", &doc["notes"]));
    Ok(())
}

fn export_cmd(a: ExportArgs) -> Result<()> {
    let mut m = Missing::new();
    let trace = m.want(a.trace, "<TRACE>", "the strace output file, or - for stdin", None);
    let root = m.want(a.root, "--root <DIR>", "the directory the traced application's paths are under; the export is relative to it", None);
    let out = m.want(a.out, "-o <FILE>", "the trace file to write", None);
    m.check()?;
    let (trace, root, out) = (trace.unwrap(), root.unwrap(), out.unwrap());
    let (trace, cwd) = in_capture(trace, &[root.as_str()], a.cwd.clone())?;
    let input = Input::open(&trace)?;
    let first = input.first_line()?;
    let opener = || input.reader();
    // written beside, renamed into place: never an empty or partial trace file at `out`
    let tmp = out.with_file_name(format!(".{}.aeiou-trace-{}", out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), std::process::id()));
    let result = (|| -> Result<export::Summary> {
        let f = File::create(&tmp).map_err(|e| anyhow!("{}: {}", out.display(), cause(&e)))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        let summary = if let Some(kind) = objectlog::source_of(&first) {
            let s = objectlog::export(&opener, &root, kind, &mut w).map_err(|e| anyhow!("{trace}: {e:#}"))?;
            if s.lines == 0 {
                return Err(anyhow!("{trace}: no request under --root {} ({} request(s) read)", root.trim_matches('/'), s.header["requests"].as_u64().unwrap_or(0)));
            }
            s
        } else {
            let s = export::strace(&opener, &strace::abspath(&root), &a.exclude, cwd.as_deref(), &mut w).map_err(|e| anyhow!("{trace}: {e:#}"))?;
            if s.lines == 0 {
                // never an empty trace file: a `trace` node of it would run nothing
                let r = s.header["root"].as_str().unwrap_or("").to_string();
                return Err(anyhow!("{trace}: {}; {} not written", strace::nothing_counted(&[r], s.seen), out.display()));
            }
            s
        };
        w.flush().map_err(|e| anyhow!("{}: {}", out.display(), cause(&e)))?;
        std::fs::rename(&tmp, &out).map_err(|e| anyhow!("{}: {}", out.display(), cause(&e)))?;
        Ok(summary)
    })();
    let s = match result {
        Ok(s) => s,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    let h = &s.header;
    eprint!("{}", incomplete_of("aeiou-trace export", h));
    print_out(&format!(
        "{} line(s) on {} lane(s), {} open(s), {} path(s) created under {}; {}\nsha256 {}\n{}",
        h["lines"],
        h["lanes"],
        h["opens"],
        h["creates"].as_array().map_or(0, Vec::len),
        h["root"].as_str().unwrap_or(""),
        out.display(),
        s.sha256,
        notes_lines(&h["notes"])
    ));
    Ok(())
}

/// `user@host`, as the Python's `getpass.getuser()` finds the user.
fn who() -> String {
    let user = ["LOGNAME", "USER", "LNAME", "USERNAME"].iter().find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty())).unwrap_or_else(|| unsafe { libc::getuid() }.to_string());
    format!("{user}@{}", aeiou::run::hostname())
}

/// Now in UTC to the second, `2026-10-31T12:00:00+00:00`.
fn now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil from days (Howard Hinnant's algorithm)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn sha256_of(path: &str) -> Result<String> {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    std::io::copy(&mut File::open(path).map_err(|e| anyhow!("{path}: {}", cause(&e)))?, &mut h)?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `--application NAME[@VERSION]` checked against the records: the application as the document
/// records it, no record resolved yet.
fn stated_application(records: &applications::Records, stated: &str) -> Result<Value> {
    let (name, version) = stated.split_once('@').unwrap_or((stated, ""));
    let Some(app) = records.apps.get(name) else {
        return Err(usage::err(&format!("--application {name}: no such application record; one of {}", records.apps.keys().cloned().collect::<Vec<_>>().join(", "))));
    };
    Ok(json!({"name": app.name, "display": app.display, "version": Some(version).filter(|v| !v.is_empty()), "record": null, "exact": false, "options": {}, "blind_spots": []}))
}

/// The answers file an interactive run leaves beside the document it wrote.
fn answers_beside(out: &str) -> String {
    match out.strip_suffix(".digest.json") {
        Some(stem) => format!("{stem}.answers.json"),
        None => format!("{out}.answers.json"),
    }
}

fn digest_cmd(a: DigestArgs) -> Result<()> {
    let mut m = Missing::new();
    let trace = m.want(a.trace.clone(), "<TRACE>", "the strace output file, or an exported trace file", None);
    m.check()?;
    let roots: Vec<&str> = a.root.iter().map(String::as_str).collect();
    let (trace, cwd) = in_capture(trace.unwrap(), &roots, a.cwd.clone())?;
    let epochs = a.epochs.unwrap_or(1);
    if epochs < 1 {
        return Err(usage::err("--epochs must be at least 1"));
    }
    if a.candidates.is_some_and(|c| c < 1) {
        return Err(usage::err("--candidates must be at least 1"));
    }
    if a.metrics_block.is_some_and(|b| b < 1) {
        return Err(usage::err("--metrics-block must be at least 1"));
    }
    let mo = digest::MetricsOpts { instance_roots: a.instance_root.clone(), chain_gap_us: a.chain_gap_us, block: a.metrics_block.map(|b| b as u64) };
    if trace == "-" {
        return Err(usage::err("the digest reads the trace as a file, never stdin: the comparison after the fit and a re-run with answers read it again"));
    }
    let answers_exist = a.answers.as_deref().is_some_and(|p| std::path::Path::new(p).exists());
    let asking = !a.no_questions && unsafe { libc::isatty(0) } == 1 && !answers_exist;
    let records = if asking || a.application.is_some() || a.applications.is_some() {
        Some(match &a.applications {
            Some(dir) => applications::load(std::path::Path::new(dir))?,
            None => applications::shipped(),
        })
    } else {
        None
    };
    let input = Input::open(&trace)?;
    let first = input.first_line()?;
    let exported = first.starts_with("{\"aeiou_trace\"");
    if exported || objectlog::source_of(&first).is_some() {
        if let Some(opt) = digest::strace_only(&mo) {
            return Err(usage::err(format!("{opt}: only an strace has the process tree and the rounds' ends it needs; {trace} is {}", if exported { "an exported trace file" } else { "an object-store request log" })));
        }
    }
    if !exported && a.root.is_none() {
        let mut m = Missing::new();
        m.need(false, "--root <DIR>", "digest only calls on paths under DIR", Some("needed for an strace; an exported trace file carries its root"));
        m.check()?;
    }
    let opener = || input.reader();
    let an = digest::analyze(&opener, &first, a.root.as_deref(), &a.exclude, cwd.as_deref(), &mo).map_err(|e| anyhow!("{trace}: {e:#}"))?;
    eprint!("{}", incomplete_of("aeiou-trace digest", &an.header));
    let sha = sha256_of(&trace)?;
    let (mut by, mut at) = (a.answered_by.clone().unwrap_or_else(who), now());
    let o = digest::Opts { epochs, candidates: a.candidates, paths: a.paths };
    let stated = match (&records, &a.application) {
        (Some(r), Some(name)) => Some(stated_application(r, name)?),
        _ => None,
    };
    let mut said = vec![];
    let mut doc = if answers_exist {
        let path = a.answers.as_deref().unwrap();
        let (header, answers) = interview::read_answers(path)?;
        let given = |k: &str| header.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        by = given("by").unwrap_or(by);
        at = given("at").unwrap_or(at);
        let mut doc = an.document(&o, &answers, Some(&by), Some(&at), header.get("application").unwrap_or(&Value::Null));
        if header.get("trace_sha256").and_then(Value::as_str) != Some(sha.as_str()) {
            let named = header.get("trace").and_then(Value::as_str).unwrap_or("None");
            doc["notes"]["answers_trace"] = json!(format!("{path} was answered on another trace ({named}); its answers were checked against this one"));
        }
        doc
    } else if asking {
        let (doc, answers) = interview::interview(&an, &mut interview::Terminal, &o, &by, &at, records.as_ref(), stated);
        if !answers.is_empty() || !doc["application"].is_null() {
            let path = a.answers.clone().or_else(|| a.out.as_ref().map(|p| answers_beside(&p.to_string_lossy())));
            match path {
                Some(path) => {
                    interview::write_answers(&path, &answers, &trace, &sha, &by, &at, &doc["application"])?;
                    let mut replay = format!("aeiou-trace digest {trace}");
                    if let Some(root) = &a.root {
                        replay += &format!(" --root {root}");
                    }
                    replay += &format!(" --answers {path}");
                    if let Some(out) = &a.out {
                        replay += &format!(" -o {}", out.display());
                    }
                    said.push(format!("answers: {path}"));
                    said.push(format!("replay without questions: {replay}"));
                }
                None => said.push("answers: in the document only (give -o or --answers FILE to keep them as a file)".into()),
            }
        }
        doc
    } else {
        if let Some(path) = &a.answers {
            return Err(anyhow!("{path}: no such answers file, and no terminal to ask on"));
        }
        let mut answers = vec![];
        let mut application = Value::Null;
        if let (Some(records), Some(mut app)) = (&records, stated) {
            let known = &records.apps[app["name"].as_str().unwrap()];
            let (record, exact) = known.resolve(app["version"].as_str());
            if let Some(record) = record {
                let options: serde_json::Map<String, Value> = record.options.iter().filter_map(|o| Some((o.name.clone(), json!(o.default.as_ref()?)))).collect();
                app["record"] = json!(record.version);
                app["exact"] = json!(exact);
                app["blind_spots"] = json!(records.blind_spots(record, &options));
                app["options"] = json!(options);
                if exact {
                    let all = an.document(&digest::Opts { epochs: 1, candidates: None, paths: true }, &[], None, None, &Value::Null);
                    let templates: Vec<(String, Vec<String>)> = all["templates"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|t| (t["template"].as_str().unwrap_or("").to_string(), t["raw_paths"].as_array().map(|p| p.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()))
                        .collect();
                    answers = record.assertions(&options, &templates);
                }
            }
            application = app;
        }
        an.document(&o, &answers, Some(&by), Some(&at), &application)
    };
    doc["trace"] = json!(trace);
    doc["trace_sha256"] = json!(sha);
    let report = digest::render(&doc);
    if let Some(out) = &a.out {
        write_out(out, &(serde_json::to_string_pretty(&doc)? + "\n"))?;
        print_out(&format!("{report}written: {}\n{}", out.display(), notes_lines(&doc["notes"])));
    } else if a.text {
        print_out(&report);
    } else {
        print_out(&(serde_json::to_string_pretty(&doc)? + "\n"));
    }
    eprint!("{}", secrets("aeiou-trace digest", &doc["metrics"]["notes"]));
    for line in said {
        eprintln!("{line}");
    }
    Ok(())
}

fn compare_cmd(a: CompareArgs) -> Result<()> {
    let mut m = Missing::new();
    let first = m.want(a.a, "<A>", "the application's document: a digest, or a metrics document", None);
    let second = m.want(a.b, "<B>", "the document A is compared with, the abstract's", None);
    m.check()?;
    let mut pairs = Vec::new();
    for p in &a.pair {
        match p.split_once('=') {
            Some((x, y)) if !x.is_empty() && !y.is_empty() => pairs.push((x.to_string(), y.to_string())),
            _ => return Err(usage::err(format!("--pair {p}: give A's template and B's as A=B"))),
        }
    }
    let o = compare::Opts {
        a: first.unwrap(),
        b: second.unwrap(),
        template_a: a.template_a,
        template_b: a.template_b,
        max_distance: a.max_distance,
        judge: a.judge,
        tolerances: a.tolerances,
        selfs: a.selfs,
        only: a.only,
        pairs,
        json: a.json,
    };
    let mut out = String::new();
    let status = compare::run(&o, &mut out);
    print_out(&out);
    match status? {
        0 => Ok(()),
        n => std::process::exit(n),
    }
}

/// A URL's scheme, lowercased: `SCHEME://` and something after it.
fn url_scheme(u: &str) -> Option<String> {
    let (scheme, rest) = u.split_once("://")?;
    let ok = !scheme.is_empty() && scheme.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) && scheme.chars().all(|c| c.is_ascii_alphanumeric() || "+.-".contains(c));
    (ok && !rest.is_empty()).then(|| scheme.to_ascii_lowercase())
}

fn capture_cmd(a: CaptureArgs) -> Result<()> {
    if let Some(dir) = &a.convert {
        return capture::convert_again(std::path::Path::new(dir));
    }
    let ranks = a.ranks.unwrap_or(1);
    let mut m = Missing::new();
    let out = m.want(a.out, "-o <DIR>", "the capture directory, new or empty, on a node-local disk", None);
    m.need(!a.root.is_empty() || !a.endpoint.is_empty(), "--root <DIR> | --endpoint <URL>", "at least one: the directories the application's data is under, or the object stores it talks to", None);
    m.need(a.check || !a.command.is_empty(), "<COMMAND>...", "after --: the command to start under the trace", Some("not with --check, which starts nothing"));
    let several = ranks > 1;
    m.need(!several || a.rank.is_some(), "--rank <R>", "this node's index, 0 to N-1", Some("needed with --ranks above 1; aeiou-launch appends it"));
    m.need(!several || a.coordinator.is_some(), "--coordinator <HOST:PORT>", "the coordinator's address: rank 0 listens on it, every rank connects to it", Some("needed with --ranks above 1; aeiou-launch appends it"));
    m.need(!several || a.source.as_deref().is_some_and(|s| s != "auto"), "--source <SOURCE>", "strace on every node: auto may choose differently from node to node", Some("needed with --ranks above 1"));
    m.check()?;
    if ranks < 1 {
        return Err(usage::err(format!("--ranks {ranks}: at least 1")));
    }
    let rank = a.rank.unwrap_or(0);
    if !(0..ranks).contains(&rank) {
        return Err(usage::err(format!("--rank {rank}: 0 to {} with --ranks {ranks}", ranks - 1)));
    }
    if let Some(c) = &a.coordinator {
        if !c.rsplit_once(':').is_some_and(|(h, p)| !h.is_empty() && p.parse::<u16>().is_ok()) {
            return Err(usage::err(format!("--coordinator {c}: expected HOST:PORT, a port number 0 to 65535")));
        }
    }
    for n in &a.accept {
        if !capture::CHECKS.contains(&n.as_str()) {
            return Err(usage::err(format!("--accept {n}: no such check; one of {}", capture::CHECKS.join(", "))));
        }
    }
    let mut endpoints = Vec::new();
    for u in &a.endpoint {
        match url_scheme(u) {
            Some(s) => endpoints.push((u.clone(), s)),
            None => return Err(usage::err(format!("--endpoint {u}: expected a URL, SCHEME://HOST[/PATH] (s3://BUCKET/PREFIX, http://HOST:PORT)"))),
        }
    }
    let requests_buffer = match &a.requests_buffer {
        Some(b) => aeiou::requests::parse_size(b).map_err(usage::err)?,
        None => aeiou::requests::DEFAULT_BUFFER,
    };
    let requests = a.requests.as_deref().map_or(!endpoints.is_empty(), |r| r == "shim");
    let buffer = match &a.buffer {
        Some(_) if a.source.as_deref() == Some("strace") => return Err(usage::err("--buffer: the eBPF source's ring buffer; strace has none (not with --source strace)")),
        Some(b) => aeiou::ebpf::parse_buffer(b).map_err(usage::err)?,
        None => aeiou::ebpf::DEFAULT_BUFFER,
    };
    let source = match a.source.as_deref().unwrap_or("auto") {
        "ebpf" => capture::ebpf_source("--source ebpf"),
        "strace" => capture::strace_source("--source strace"),
        _ => match capture::ebpf_source("auto: eBPF can load here") {
            s if s.broken.is_none() => s,
            s => capture::strace_source(&format!("auto: {}", s.broken.unwrap_or_default())),
        },
    };
    let out = capture::expand_out(&out.unwrap(), &aeiou::run::hostname(), std::process::id()).map_err(usage::err)?;
    let out = std::path::Path::new(&out);
    let out = if out.is_absolute() { out.to_path_buf() } else { std::env::current_dir()?.join(out) };
    let o = capture::Opts {
        source,
        roots: a.root,
        endpoints,
        cold: a.cold,
        accept: a.accept,
        check_only: a.check,
        requests,
        requests_buffer,
        buffer,
        capture_id: a.capture_id.or_else(|| std::env::var("SLURM_JOB_ID").ok().filter(|v| !v.is_empty())),
        ranks,
        rank,
        coordinator: a.coordinator,
        out,
        command: a.command,
        argv_recorded: a.argv.as_deref() != Some("none"),
    };
    match capture::run(o)? {
        0 => Ok(()),
        n => std::process::exit(n),
    }
}

fn main() {
    let args: Vec<String> = std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect();
    let subs = ["metrics", "export", "compare", "digest", "capture"];
    let sub = args.iter().skip(1).find(|a| !a.starts_with('-')).filter(|a| subs.contains(&a.as_str())).cloned();
    let prog = match &sub {
        Some(s) => format!("aeiou-trace {s}"),
        None => "aeiou-trace".into(),
    };
    let usage_text = || {
        let mut cmd = Cli::command();
        match &sub {
            Some(s) => cmd.find_subcommand_mut(s.as_str()).expect("a subcommand of the parser").render_usage().to_string(),
            None => cmd.render_usage().to_string(),
        }
    };
    let usage_exit = |message: &str| -> ! {
        eprint!("{}", usage::render(&prog, message, &usage_text()));
        std::process::exit(2)
    };
    let cli = match Cli::command().try_get_matches_from(std::env::args_os()).and_then(|m| Cli::from_arg_matches(&m)) {
        Ok(cli) => cli,
        Err(e) => {
            use clap::error::ErrorKind;
            if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
                e.exit();
            }
            eprint!("{}", usage::reframe(&e.render().to_string(), &prog, &usage_text()));
            std::process::exit(2);
        }
    };
    let result = match cli.cmd {
        None => {
            let mut m = Missing::new();
            m.need(false, "<COMMAND>", "one of metrics, export, compare, digest, capture", None);
            m.check()
        }
        Some(Cmd::Metrics(a)) => metrics(a),
        Some(Cmd::Export(a)) => export_cmd(a),
        Some(Cmd::Compare(a)) => compare_cmd(a),
        Some(Cmd::Digest(a)) => digest_cmd(a),
        Some(Cmd::Capture(a)) => capture_cmd(a),
    };
    if let Err(e) = result {
        if e.downcast_ref::<UsageError>().is_some() {
            usage_exit(&format!("{e:#}"));
        }
        eprintln!("{prog}: {e:#}");
        std::process::exit(1);
    }
}
