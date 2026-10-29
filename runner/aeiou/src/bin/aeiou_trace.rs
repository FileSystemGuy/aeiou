//! `aeiou-trace`: the trace side of the suite (`builder/REFERENCE.md` §7, `DESIGN_REVIEW.md`
//! §3.42, §3.89): the locality metrics of a real application from an strace of it, an exported
//! trace file, or an object-store request log; the runner's trace file for a `trace` node; and,
//! until they are ported (§3.89 build order), the comparison of two metrics documents and the
//! structural digest, which run the Python package's (`python3 -m aeiou.trace`, or the
//! interpreter `$AEIOU_PYTHON` names).

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use serde_json::{json, Value};

use aeiou::usage::{self, Missing, UsageError};
use aeiou::{export, objectlog, strace};

const ABOUT: &str = "Locality metrics of a real application from an strace of it, their comparison with an abstract's (aeiou dry-run --metrics-json), the runner's trace file, and the structural digest a fit is made from";
const EXIT_PIPE: i32 = 141;

#[derive(Parser)]
#[command(name = "aeiou-trace", version, about = ABOUT, override_usage = "aeiou-trace <COMMAND> [ARGS]...", max_term_width = 120)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compute the metrics of a trace (strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o FILE)
    #[command(override_usage = "aeiou-trace metrics [OPTIONS] <TRACE>")]
    Metrics(MetricsArgs),
    /// Write the runner's trace file for a `trace` node from an strace (DESIGN_REVIEW.md §3.58): JSON Lines, one lane per traced task, opens by id, paths relative to --root
    #[command(override_usage = "aeiou-trace export [OPTIONS] --root <DIR> -o <FILE> <TRACE>")]
    Export(ExportArgs),
    /// Compare two metrics documents, from a trace or from aeiou dry-run --metrics-json
    #[command(disable_help_flag = true, override_usage = "aeiou-trace compare [OPTIONS] <A> <B>")]
    Compare(Passed),
    /// Write the structural digest of a trace: path templates, lanes and their phases, a verdict per site on what the trace decides (permutation or draw, order, size), and the open questions ranked by their divergence at scale
    #[command(disable_help_flag = true, override_usage = "aeiou-trace digest [OPTIONS] --root <DIR> <TRACE>")]
    Digest(Passed),
}

#[derive(clap::Args)]
struct MetricsArgs {
    /// The strace output file, an exported trace file, an object-store request log (MinIO audit, S3 server access), or - for stdin
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
    /// The strace output file, an object-store request log (MinIO audit, S3 server access), or - for stdin
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

/// A subcommand the Python package still runs: its arguments, untouched.
#[derive(clap::Args)]
struct Passed {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
    args: Vec<String>,
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

fn missing_object_root(kind: &str) -> anyhow::Error {
    let mut m = Missing::new();
    m.need(false, "--root <BUCKET[/PREFIX]>", "the bucket, and the prefix under it, whose requests count", Some(&format!("needed for an object log ({kind})")));
    m.check().unwrap_err()
}

fn metrics(a: MetricsArgs) -> Result<()> {
    let mut m = Missing::new();
    let trace = m.want(a.trace, "<TRACE>", "the strace output file, or - for stdin", None);
    m.check()?;
    let trace = trace.unwrap();
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
            cwd: a.cwd,
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
    Ok(())
}

fn export_cmd(a: ExportArgs) -> Result<()> {
    let mut m = Missing::new();
    let trace = m.want(a.trace, "<TRACE>", "the strace output file, or - for stdin", None);
    let root = m.want(a.root, "--root <DIR>", "the directory the traced application's paths are under; the export is relative to it", None);
    let out = m.want(a.out, "-o <FILE>", "the trace file to write", None);
    m.check()?;
    let (trace, root, out) = (trace.unwrap(), root.unwrap(), out.unwrap());
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
            let s = export::strace(&opener, &strace::abspath(&root), &a.exclude, a.cwd.as_deref(), &mut w).map_err(|e| anyhow!("{trace}: {e:#}"))?;
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

/// Run a subcommand the Python package still holds, with its arguments as given.
fn passed(sub: &str, args: &[String]) -> Result<()> {
    let python = std::env::var("AEIOU_PYTHON").unwrap_or_else(|_| "python3".into());
    let probe = std::process::Command::new(&python).args(["-c", "import aeiou.trace"]).stderr(std::process::Stdio::null()).stdout(std::process::Stdio::null()).status();
    if !probe.is_ok_and(|s| s.success()) {
        return Err(anyhow!(
            "{sub} is the Python package's until it is ported (DESIGN_REVIEW.md §3.89), and {python} cannot import aeiou: install the aeiou wheel, or set AEIOU_PYTHON to an interpreter that has it"
        ));
    }
    let status = std::process::Command::new(&python).args(["-m", "aeiou.trace", sub]).args(args).status().map_err(|e| anyhow!("{python}: {}", cause(&e)))?;
    std::process::exit(status.code().unwrap_or(1));
}

fn main() {
    let args: Vec<String> = std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect();
    let subs = ["metrics", "export", "compare", "digest"];
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
            m.need(false, "<COMMAND>", "one of metrics, export, compare, digest", None);
            m.check()
        }
        Some(Cmd::Metrics(a)) => metrics(a),
        Some(Cmd::Export(a)) => export_cmd(a),
        Some(Cmd::Compare(p)) => passed("compare", &p.args),
        Some(Cmd::Digest(p)) => passed("digest", &p.args),
    };
    if let Err(e) = result {
        if e.downcast_ref::<UsageError>().is_some() {
            usage_exit(&format!("{e:#}"));
        }
        eprintln!("{prog}: {e:#}");
        std::process::exit(1);
    }
}
