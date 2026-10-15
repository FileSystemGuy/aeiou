//! The usage-error frame (`usage.rs`, `runner/REFERENCE.md` §15): every wrong command line, the
//! parser's own mistakes included, is printed as the command's name, the message, the usage
//! line, and the pointer to the help, with exit status 2; every missing requirement is listed
//! at once, from any layer, and the list shrinks by exactly what the next attempt supplies;
//! what the options in effect make required joins the list with the reason; a wrong file for
//! the abstract says what an abstract is; a failure during the work names the command and
//! exits 1.

use std::path::PathBuf;
use std::process::Command;

const MISSING: &str = "the following required arguments were not provided:";

fn aeiou(args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.args(args);
    for (k, _) in std::env::vars() {
        if k.starts_with("AEIOU_") {
            c.env_remove(k);
        }
    }
    c
}

fn example(name: &str) -> String {
    crate::common::workload_ast(name).display().to_string()
}

/// The stderr of a usage error, split: (message, usage block); asserts the frame.
fn frame(c: &mut Command, prog: &str) -> (String, String) {
    let o = c.output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    assert_eq!(o.status.code(), Some(2), "stdout:\n{}\nstderr:\n{err}", String::from_utf8_lossy(&o.stdout));
    let footer = format!("\n\nFor more information, try '{prog} --help'.\n");
    assert!(err.starts_with(&format!("{prog}: ")), "{err}");
    assert!(err.ends_with(&footer), "{err}");
    let body = &err[prog.len() + 2..err.len() - footer.len()];
    let (message, usage) = body.split_once("\n\nUsage: ").unwrap_or_else(|| panic!("no usage block:\n{err}"));
    (message.to_string(), format!("Usage: {usage}"))
}

#[test]
fn every_missing_argument_is_listed_at_once_and_the_list_shrinks() {
    let (message, usage) = frame(&mut aeiou(&["datagen"]), "aeiou datagen");
    assert_eq!(
        message,
        format!(
            "{MISSING}\n  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n  --root <DIR>     directory the abstract's paths are relative to\n                   (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)"
        )
    );
    assert!(usage.starts_with("Usage: aeiou datagen [OPTIONS] --root <DIR> <ABSTRACT_PATH>\n"), "{usage}");
    let ast = example("file_per_sample.ast.json");
    let (message, _) = frame(&mut aeiou(&["datagen", &ast]), "aeiou datagen");
    assert_eq!(message, format!("{MISSING}\n  --root <DIR>  directory the abstract's paths are relative to\n                (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)"));
    // --root from a lower layer is not missing
    let (message, _) = frame(aeiou(&["datagen", &ast, "--gpus", "x"]).env("AEIOU_ROOT", "/nonexistent"), "aeiou datagen");
    assert!(message.starts_with("invalid value 'x' for '--gpus <GPUS>'"), "{message}");
    // run: the identity and the root, then what the options in effect make required
    let (message, _) = frame(&mut aeiou(&["run"]), "aeiou run");
    assert_eq!(
        message,
        format!(
            "{MISSING}\n  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n  --gpus <GPUS>    number of instances of every actor template whose count is `gpus`\n  --root <DIR>     directory the abstract's paths are relative to\n                   (also $AEIOU_ROOT, or `root` in the [run] table of the config file)"
        )
    );
    let (message, _) = frame(aeiou(&["run", &ast, "--gpus", "1", "--ranks", "2", "--report-takes"]).env("AEIOU_ROOT", "/nonexistent").env("AEIOU_SQPOLL_SHARED", "true"), "aeiou run");
    assert_eq!(
        message,
        format!(
            "{MISSING}\n  --coordinator <HOST:PORT>  the coordinator's address: rank 0 listens on it, every rank connects to it\n                             (needed with --ranks 2)\n  --report-json <FILE>       the file the run's report is written to as JSON\n                             (needed by --report-takes)\n  --sqpoll <IDLE_MS>         a kernel submission thread per loop\n                             (needed by --sqpoll-shared, --sqpoll-shared from env AEIOU_SQPOLL_SHARED)"
        )
    );
    let (message, usage) = frame(&mut aeiou(&["dry-run"]), "aeiou dry-run");
    assert_eq!(message, format!("{MISSING}\n  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n  --gpus <GPUS>    number of instances of every actor template whose count is `gpus`"));
    assert_eq!(usage, "Usage: aeiou dry-run [OPTIONS] --gpus <GPUS> <ABSTRACT_PATH>");
    let (message, usage) = frame(&mut aeiou(&["check"]), "aeiou check");
    assert_eq!(message, format!("{MISSING}\n  <FILES>...  the abstracts to validate (`.ast.json`, written by aeiou-build from builder scripts)"));
    assert_eq!(usage, "Usage: aeiou check [OPTIONS] <FILES>...");
}

#[test]
fn the_parsers_own_errors_and_the_layers_refusals_wear_the_frame() {
    let ast = example("file_per_sample.ast.json");
    let (message, _) = frame(&mut aeiou(&["datagen", "--bogus", &ast, "--root", "/tmp"]), "aeiou datagen");
    assert!(message.starts_with("unexpected argument '--bogus' found"), "{message}");
    let (message, _) = frame(aeiou(&["dry-run", &ast, "--gpus", "1"]).env("AEIOU_GPUS", "3"), "aeiou dry-run");
    assert!(message.starts_with("AEIOU_GPUS is set, but --gpus is the command line's alone"), "{message}");
    let (message, _) = frame(&mut aeiou(&["dry-run", &ast, "--gpus", "1", "--[no-]metrics"]), "aeiou dry-run");
    assert!(message.ends_with("type --metrics to turn it on or --no-metrics to turn it off"), "{message}");
    let (message, _) = frame(&mut aeiou(&["run", &ast, "--gpus", "1", "--root", "/tmp", "--defer-taskrun", "--sqpoll", "10"]), "aeiou run");
    assert_eq!(message, "--defer-taskrun and --sqpoll exclude each other");
    // a backend is an API and a cache mode, each its own flag (DESIGN_REVIEW §3.65)
    let run = |extra: &[&str]| frame(&mut aeiou(&[&["run", &ast, "--gpus", "1", "--root", "/tmp"], extra].concat()), "aeiou run").0;
    assert!(run(&["--posix", "nope"]).starts_with("--posix nope: not one of "));
    assert!(run(&["--cache", "nope"]).starts_with("--cache nope: not one of "));
    assert_eq!(run(&["--posix", "mmap", "--cache", "direct"]), "--cache direct under --posix mmap: its reads are page faults on a mapping");
}

#[test]
fn a_wrong_file_for_the_abstract_says_what_an_abstract_is() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../workloads/input-pipeline/map/file-per-sample/file_per_sample.py").display().to_string();
    let (message, _) = frame(&mut aeiou(&["datagen", "--root", "/tmp", &script]), "aeiou datagen");
    assert_eq!(message, format!("{script}: a builder script, not an abstract; `aeiou-build {script}` writes the abstracts it authors (`<name>.ast.json`) next to it, and those are what the runner takes"));
    let (message, _) = frame(&mut aeiou(&["datagen", "--root", "/tmp", "file_per_sample"]), "aeiou datagen");
    assert_eq!(message, "file_per_sample: No such file or directory (os error 2); an abstract is the `.ast.json` file aeiou-build writes from a builder script");
    let dir = crate::common::tmpdir("usage", "not-an-abstract");
    let bad = dir.join("x.ast.json");
    std::fs::write(&bad, "import aeiou\n").unwrap();
    let (message, _) = frame(&mut aeiou(&["dry-run", "--gpus", "1", bad.to_str().unwrap()]), "aeiou dry-run");
    assert_eq!(message, format!("{}: not an abstract (expected value at line 1 column 1); an abstract is the `.ast.json` file aeiou-build writes from a builder script", bad.display()));
    // an invalid abstract is a failure of the work, not of the line: the command's name, status 1
    std::fs::write(&bad, "{\"not\": \"an abstract\"}\n").unwrap();
    let o = aeiou(&["dry-run", "--gpus", "1", bad.to_str().unwrap()]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.starts_with(&format!("aeiou dry-run: {}: ", bad.display())) && !err.contains("Usage:"), "{err}");
}

/// A number that can only be meaningless is a usage error in the frame, never a panic, a bare
/// parse error with status 1, or a silent clamp (RC1 review): dry-run's --ranks, --steps,
/// --metrics-block, --metrics-sample, --gpu, --limit, --threads; run's and datagen's zeros.
#[test]
fn a_meaningless_number_is_a_usage_error() {
    let ast = example("file_per_sample.ast.json");
    let dry = |extra: &[&str]| frame(&mut aeiou(&[&["dry-run", &ast, "--gpus", "2"], extra].concat()), "aeiou dry-run").0;
    assert_eq!(dry(&["--ranks", "0"]), "--ranks 0: the run needs at least one host");
    let (message, _) = frame(aeiou(&["dry-run", &ast, "--gpus", "2"]).env("AEIOU_RANKS", "0"), "aeiou dry-run");
    assert_eq!(message, "--ranks 0: the run needs at least one host (--ranks from env AEIOU_RANKS)");
    assert_eq!(dry(&["--ranks=-1"]), "--ranks -1: the run needs at least one host");
    assert_eq!(dry(&["--threads", "0"]), "--threads 0: the walk needs a thread");
    for s in ["1..x", "x..2", "3", "5..1", "2..2", "-1..2"] {
        assert_eq!(dry(&["--gpu", "0", &format!("--steps={s}")]), format!("--steps {s}: expected A..B, whole numbers with 0 <= A < B"));
    }
    assert_eq!(dry(&["--metrics", "--metrics-block", "0"]), "--metrics-block 0: the block size is at least 1 byte");
    assert_eq!(dry(&["--metrics", "--metrics-sample", "0"]), "--metrics-sample 0: N is at least 1 (1 keeps every block)");
    assert_eq!(dry(&["--gpu", "0", "--limit", "0"]), "--limit 0: nothing would be printed");
    assert_eq!(dry(&["--gpu=-1"]), "--gpu -1: instance ids start at 0");
    // --gpu 2 of --gpus 2 named no instance and printed nothing, with status 0
    assert_eq!(dry(&["--gpu", "2"]), "--gpu 2: no actor has that instance (ids: gpu 0..2)");
    assert_eq!(dry(&["--gpu", "5"]), "--gpu 5: no actor has that instance (ids: gpu 0..2)");
    let o = aeiou(&["dry-run", &ast, "--gpus", "2", "--gpu", "1", "--steps", "0..1", "--limit", "3"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(String::from_utf8_lossy(&o.stdout).lines().filter(|l| l.starts_with("gpu#1 [")).count(), 3);
    let run = |extra: &[&str]| frame(&mut aeiou(&[&["run", &ast, "--gpus", "1", "--root", "/nonexistent"], extra].concat()), "aeiou run").0;
    assert_eq!(run(&["--threads", "0"]), "--threads 0: an event loop needs a thread (leave it out for one per core)");
    assert_eq!(run(&["--buffer-mib", "0"]), "--buffer-mib 0: a whole number of MiB above 0 that fits in memory");
    assert_eq!(run(&["--buffer-mib", &(usize::MAX >> 10).to_string()]), format!("--buffer-mib {}: a whole number of MiB above 0 that fits in memory", usize::MAX >> 10));
    assert_eq!(run(&["--write-compress", "0"]), "--write-compress 0: the compression ratio is at least 1");
    assert_eq!(run(&["--time-scale=-1"]), "--time-scale -1: a factor of 0 or more");
    assert_eq!(run(&["--time-scale", "inf"]), "--time-scale inf: a factor of 0 or more");
    assert_eq!(run(&["--iowq-max-workers", "0"]), "--iowq-max-workers 0: a cap of no workers (leave it out for the kernel's default)");
    assert_eq!(run(&["--max-gap", "NaN"]), "--max-gap NaN: a number of seconds, 0 or more");
    let gen = |extra: &[&str]| frame(&mut aeiou(&[&["datagen", &ast, "--root", "/nonexistent"], extra].concat()), "aeiou datagen").0;
    assert_eq!(gen(&["--writers", "0"]), "--writers 0: the datagen needs a writer (leave it out for one per core)");
    assert_eq!(gen(&["--dedupe", "0"]), "--dedupe 0: the dedupe ratio is at least 1");
    assert_eq!(gen(&["--compress", "0"]), "--compress 0: the compression ratio is at least 1");
    // a window too long for a Duration panicked in the conversion
    assert!(gen(&["--progress-secs", "1e30"]).ends_with(": a positive number of seconds"));
}

/// A name that is none of the abstract's datasets is a usage error naming the ones it has, and
/// nothing is written (RC1 review): `datagen --dataset nosuch` wrote nothing and exited 0, so
/// a typo in a tuning loop looked like success. `--endpoint` with such a name was refused already.
#[test]
fn an_unknown_dataset_name_is_a_usage_error() {
    let ast = example("file_per_sample.ast.json");
    let root = std::env::temp_dir().join(format!("aeiou-usage-dataset-{}", std::process::id()));
    let r = root.to_str().unwrap();
    let gen = |extra: &[&str]| frame(&mut aeiou(&[&["datagen", &ast, "--root", r, "--param", "samples=4"], extra].concat()), "aeiou datagen").0;
    assert_eq!(gen(&["--dataset", "nosuch"]), "--dataset nosuch: the abstract has no dataset `nosuch` (its datasets: `train`)");
    assert_eq!(gen(&["--dataset", "train", "--dataset", "a", "--dataset", "b"]), "--dataset a, b: the abstract has no dataset `a`, `b` (its datasets: `train`)");
    assert_eq!(gen(&["--endpoint", "nosuch=/tmp"]), "--endpoint nosuch=/tmp: the abstract has no dataset or namespace `nosuch`");
    assert!(!root.exists(), "{} was created", root.display());
}

/// An option that only narrows another is never ignored in silence (RC1 review): dry-run's
/// --steps and --limit without --gpu printed instance 0's stream as if --gpu 0 had been given,
/// and --metrics-block and --metrics-sample without --metrics were dropped with status 0. What
/// each needs joins the list with the options that need it, all at once, in the frame.
#[test]
fn an_option_narrowing_an_absent_one_is_a_usage_error() {
    let ast = example("file_per_sample.ast.json");
    let dry = |extra: &[&str]| frame(&mut aeiou(&[&["dry-run", &ast, "--gpus", "2"], extra].concat()), "aeiou dry-run").0;
    let gpu = "the actor instance whose op stream is printed";
    let metrics = "compute the locality metrics of the op stream (or --metrics-json <FILE>)";
    assert_eq!(dry(&["--limit", "2"]), format!("{MISSING}\n  --gpu <G>  {gpu}\n             (needed by --limit)"));
    assert_eq!(dry(&["--steps", "0..1", "--limit", "2"]), format!("{MISSING}\n  --gpu <G>  {gpu}\n             (needed by --steps, --limit)"));
    assert_eq!(dry(&["--metrics-sample", "4"]), format!("{MISSING}\n  --metrics  {metrics}\n             (needed by --metrics-sample)"));
    // --no-metrics after --metrics turns it off; an explicit default is given all the same
    assert_eq!(dry(&["--metrics", "--no-metrics", "--metrics-block", "8192"]), format!("{MISSING}\n  --metrics  {metrics}\n             (needed by --metrics-block)"));
    assert_eq!(dry(&["--metrics-block", "4096"]), format!("{MISSING}\n  --metrics  {metrics}\n             (needed by --metrics-block)"));
    // every problem at once, with what is missing outright
    let (message, _) = frame(&mut aeiou(&["dry-run", "--steps", "0..1", "--metrics-block", "8192", "--metrics-sample", "4"]), "aeiou dry-run");
    assert_eq!(
        message,
        format!(
            "{MISSING}\n  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n  --gpus <GPUS>    number of instances of every actor template whose count is `gpus`\n  --gpu <G>        {gpu}\n                   (needed by --steps)\n  --metrics        {metrics}\n                   (needed by --metrics-block, --metrics-sample)"
        )
    );
    // with what they narrow they run; --metrics-json implies --metrics
    let dir = std::env::temp_dir().join(format!("aeiou-usage-narrow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let json = dir.join("m.json");
    for extra in [
        &["--gpu", "1", "--steps", "0..1", "--limit", "2"][..],
        &["--metrics", "--metrics-block", "8192", "--metrics-sample", "4"][..],
        &["--metrics-json", json.to_str().unwrap(), "--metrics-block", "8192"][..],
    ] {
        let o = aeiou(&[&["dry-run", &ast, "--gpus", "2"], extra].concat()).output().unwrap();
        assert!(o.status.success(), "{extra:?}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
    assert_eq!((doc["block"].as_u64(), doc["sample"].as_u64()), (Some(8192), Some(1)));
    std::fs::remove_dir_all(&dir).unwrap();
}
