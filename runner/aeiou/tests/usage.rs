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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schema/examples").join(name).display().to_string()
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
    let ast = example("train_small_files.ast.json");
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
    let ast = example("train_small_files.ast.json");
    let (message, _) = frame(&mut aeiou(&["datagen", "--bogus", &ast, "--root", "/tmp"]), "aeiou datagen");
    assert!(message.starts_with("unexpected argument '--bogus' found"), "{message}");
    let (message, _) = frame(aeiou(&["dry-run", &ast, "--gpus", "1"]).env("AEIOU_GPUS", "3"), "aeiou dry-run");
    assert!(message.starts_with("AEIOU_GPUS is set, but --gpus is the command line's alone"), "{message}");
    let (message, _) = frame(&mut aeiou(&["dry-run", &ast, "--gpus", "1", "--[no-]metrics"]), "aeiou dry-run");
    assert!(message.ends_with("type --metrics to turn it on or --no-metrics to turn it off"), "{message}");
    let (message, _) = frame(&mut aeiou(&["run", &ast, "--gpus", "1", "--root", "/tmp", "--defer-taskrun", "--sqpoll", "10"]), "aeiou run");
    assert_eq!(message, "--defer-taskrun and --sqpoll exclude each other");
    let (message, _) = frame(&mut aeiou(&["run", &ast, "--gpus", "1", "--root", "/tmp", "--io-backend", "nope"]), "aeiou run");
    assert!(message.starts_with("--io-backend nope: not one of "), "{message}");
    // a backend is an API and a cache mode (DESIGN_REVIEW §3.65); --io-backend names both, alone
    let run = |extra: &[&str]| frame(&mut aeiou(&[&["run", &ast, "--gpus", "1", "--root", "/tmp"], extra].concat()), "aeiou run").0;
    assert!(run(&["--io-api", "nope"]).starts_with("--io-api nope: not one of "));
    assert!(run(&["--cache", "nope"]).starts_with("--cache nope: not one of "));
    assert_eq!(run(&["--io-backend", "sync", "--cache", "direct"]), "--io-backend sync names the API and the cache mode; not with --io-api or --cache");
    assert_eq!(run(&["--io-api", "mmap", "--cache", "direct"]), "--cache direct under --io-api mmap: its reads are page faults on a mapping");
}

#[test]
fn a_wrong_file_for_the_abstract_says_what_an_abstract_is() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../builder/abstracts/train_small_files.py").display().to_string();
    let (message, _) = frame(&mut aeiou(&["datagen", "--root", "/tmp", &script]), "aeiou datagen");
    assert_eq!(message, format!("{script}: a builder script, not an abstract; `aeiou-build {script}` writes the abstracts it authors (`<name>.ast.json`) next to it, and those are what the runner takes"));
    let (message, _) = frame(&mut aeiou(&["datagen", "--root", "/tmp", "train_small_files"]), "aeiou datagen");
    assert_eq!(message, "train_small_files: No such file or directory (os error 2); an abstract is the `.ast.json` file aeiou-build writes from a builder script");
    let dir = std::env::temp_dir().join(format!("aeiou-usage-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
    std::fs::remove_dir_all(&dir).unwrap();
}
