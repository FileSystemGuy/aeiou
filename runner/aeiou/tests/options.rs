//! The option layers through the binary (`options.rs`, `runner/REFERENCE.md` §14): the block every
//! subcommand prints, the environment and config layers, the refusal of a fixed option from a
//! lower layer, the negation of a flag the file turned on, the report's `layers`, and, over two
//! ranks, every host's block in rank 0's report with the options that differ printed, and the
//! identity refusal naming the field; and the `--[no-]x` notation: the help shows one row per
//! boolean, aligned, `--x`/`--no-x` parse with the last one winning, the brackets typed
//! literally are explained.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

fn tmpdir(tag: &str) -> crate::common::TempDir {
    crate::common::tmpdir("options", tag)
}

fn aeiou(sub: &str, ast: &str, params: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_aeiou"));
    c.arg(sub).arg(crate::common::workload_ast(ast));
    if ast == "file_per_sample.ast.json" {
        c.args(crate::common::file_per_sample_args());      // its old defaults, required since 2026-10-22
    }
    for p in params {
        c.arg("--param").arg(p);
    }
    c.env_remove("AEIOU_CONFIG");
    c
}

fn out(c: &mut Command) -> (bool, String, String) {
    let o = c.output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The `name = value  [source]` lines of the options block, as (name, value, source).
fn block(text: &str) -> Vec<(String, String, String)> {
    let mut rows = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        if line.starts_with("options (cli > env > config > default)") {
            in_block = true;
            continue;
        }
        if in_block {
            if line.is_empty() {
                break;
            }
            let Some(body) = line.strip_prefix("  ") else { continue };
            if body.starts_with("WARNING") {
                continue;
            }
            let (lhs, src) = body.rsplit_once("  [").unwrap();
            let (name, value) = lhs.trim_end().split_once(" = ").unwrap();
            rows.push((name.to_string(), value.to_string(), src.trim_end_matches(']').to_string()));
        }
    }
    rows
}

fn row<'a>(rows: &'a [(String, String, String)], name: &str) -> &'a (String, String, String) {
    rows.iter().find(|r| r.0 == name).unwrap_or_else(|| panic!("no row {name} in {rows:?}"))
}

const SMALL: [&str; 6] = ["samples=600", "batch=4", "workers=2", "prefetch=2", "steps=12", "walk=true"];

#[test]
fn every_subcommand_prints_the_block_with_sources() {
    let d = tmpdir("block");
    let cfg = d.join("aeiou.toml");
    std::fs::write(&cfg, "[dry-run]\nthreads = 2\n\n[datagen]\nwriters = 3\n\n[run]\nbuffer-mib = 16\nrequire-cold = true\n").unwrap();
    // check: the files and the config file
    let (ok, text, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("check").arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json")).env_remove("AEIOU_CONFIG"));
    assert!(ok, "{err}");
    let rows = block(&text);
    assert_eq!(row(&rows, "config").2, "default");
    assert!(text.lines().any(|l| l.starts_with("ok   file_per_sample.ast.json")), "{text}");
    // dry-run: cli, env, config, default, and a warning for an unknown AEIOU_*
    let (ok, text, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).arg("--config").arg(&cfg).env("AEIOU_RANKS", "3").env("AEIOU_BOGUS", "1"));
    assert!(ok, "{err}");
    let rows = block(&text);
    assert_eq!(row(&rows, "gpus"), &("gpus".into(), "2".into(), "cli".into()));
    assert_eq!(row(&rows, "seed"), &("seed".into(), "0".into(), "default".into()));
    assert_eq!(row(&rows, "ranks"), &("ranks".into(), "3".into(), "env AEIOU_RANKS".into()));
    assert_eq!(row(&rows, "threads"), &("threads".into(), "2".into(), format!("config {}", cfg.display())));
    assert_eq!(row(&rows, "config").2, "cli");
    assert!(text.contains("WARNING: AEIOU_BOGUS is set and is no option of any subcommand; ignored"), "{text}");
    assert!(text.contains("3 host(s)") || text.contains("per host"), "{text}");
    // the same through AEIOU_CONFIG
    let (ok, text, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).env("AEIOU_CONFIG", &cfg));
    assert!(ok, "{err}");
    assert_eq!(row(&block(&text), "config").2, "env AEIOU_CONFIG");
    // datagen: root from the environment, writers from the file
    let root = d.join("root");
    let (ok, text, err) = out(aeiou("datagen", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).env("AEIOU_ROOT", &root).env("AEIOU_CONFIG", &cfg));
    assert!(ok, "{err}");
    let rows = block(&text);
    assert_eq!(row(&rows, "root"), &("root".into(), root.display().to_string(), "env AEIOU_ROOT".into()));
    assert_eq!(row(&rows, "writers").1, "3");
    assert_eq!(row(&rows, "dedupe"), &("dedupe".into(), "1".into(), "default".into()));
    // run: the file turns require-cold on and sets the buffer, the command line turns it off
    // again; the report records the layers
    let json = root.join("report.json");
    let (ok, text, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL)
        .args(["--gpus", "2", "--time-scale", "0", "--require-cold", "--no-require-cold"])
        .arg("--report-json")
        .arg(&json)
        .env("AEIOU_ROOT", &root)
        .env("AEIOU_CONFIG", &cfg));
    assert!(ok, "{text}\n{err}");
    let rows = block(&text);
    assert_eq!(row(&rows, "buffer-mib"), &("buffer-mib".into(), "16".into(), format!("config {}", cfg.display())));
    assert_eq!(row(&rows, "require-cold"), &("require-cold".into(), "false".into(), "cli".into()));
    assert_eq!(row(&rows, "time-scale"), &("time-scale".into(), "0".into(), "cli".into()));
    assert_eq!(row(&rows, "posix"), &("posix".into(), "the abstract's".into(), "default".into()));
    assert_eq!(row(&rows, "cache"), &("cache".into(), "the abstract's".into(), "default".into()));
    assert_eq!(row(&rows, "root").2, "env AEIOU_ROOT");
    let doc = read(&json);
    let layers = &doc["layers"];
    assert_eq!(layers["config"]["path"], cfg.display().to_string());
    assert_eq!(layers["config"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(layers["env"], serde_json::json!(["AEIOU_CONFIG", "AEIOU_ROOT"]));
    assert_eq!(layers["options"]["buffer-mib"], serde_json::json!({"value": 16, "source": format!("config {}", cfg.display())}));
    assert_eq!(layers["options"]["require-cold"], serde_json::json!({"value": false, "source": "cli"}));
    assert_eq!(layers["options"]["rank"]["source"], "default");
    assert_eq!(doc["options"]["buffer_bytes"], 16 << 20);
    // the same run without the override: the file's require-cold is in effect (the dataset is
    // warm, so the run is refused for that reason)
    let (ok, text, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2", "--time-scale", "0"]).env("AEIOU_ROOT", &root).env("AEIOU_CONFIG", &cfg));
    assert!(!ok);
    assert_eq!(row(&block(&text), "require-cold").2, format!("config {}", cfg.display()));
    assert!(err.contains("--require-cold"), "{err}");
}

#[test]
fn fixed_options_are_refused_from_the_environment_and_the_file() {
    let d = tmpdir("fixed");
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).env("AEIOU_SEED", "4"));
    assert!(!ok);
    assert!(err.contains("AEIOU_SEED is set, but --seed is the command line's alone"), "{err}");
    let cfg = d.join("aeiou.toml");
    std::fs::write(&cfg, "[run]\nclean-namespaces = true\n").unwrap();
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).arg("--root").arg(&d).arg("--config").arg(&cfg));
    assert!(!ok);
    assert!(err.contains("[run] clean-namespaces is set, but --clean-namespaces is the command line's alone"), "{err}");
    // the environment and the file may not name an option that belongs to another subcommand's
    // table, nor an unknown one
    std::fs::write(&cfg, "[run]\nmetrics-block = 8\n").unwrap();
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).arg("--root").arg(&d).arg("--config").arg(&cfg));
    assert!(!ok);
    assert!(err.contains("[run] metrics-block: not an option of `aeiou run`"), "{err}");
    std::fs::write(&cfg, "buffer-mib = 8\n").unwrap();
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).arg("--root").arg(&d).arg("--config").arg(&cfg));
    assert!(!ok);
    assert!(err.contains("top level"), "{err}");
    // a knob the backend cannot take: from the command line it is refused (the knobs' own
    // tests have the messages); from the environment or the file, which may serve other runs
    // (one file for a cluster), it is passed over with a warning in the block (RC1 review)
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2", "--threads", "4"]).arg("--root").arg(&d));
    assert!(!ok);
    assert!(err.lines().next().unwrap().ends_with("has one thread per actor"), "{err}");
    std::fs::write(&cfg, "[run]\naio-depth = 64\nsqpoll = 10\nmmap-mode = \"populate\"\nobject-threads = 4\nobject-part-mib = 16\n").unwrap();
    let json = d.join("report.json");
    let (ok, text, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL)
        .args(["--gpus", "2", "--time-scale", "0"])
        .arg("--root")
        .arg(&d)
        .arg("--config")
        .arg(&cfg)
        .arg("--report-json")
        .arg(&json)
        .env("AEIOU_THREADS", "4")
        .env("AEIOU_DEFER_TASKRUN", "true"));
    // the run goes as far as the dataset, which is not there: no usage error
    assert!(!ok && !err.contains("knob") && !err.contains("Usage:"), "{err}");
    assert!(err.contains("train"), "{err}");
    let c = cfg.display();
    for w in [
        format!("--aio-depth from config {c} is not used: --posix sync has no AIO context"),
        format!("--sqpoll from config {c} is not used: --posix sync has no ring"),
        "--defer-taskrun from env AEIOU_DEFER_TASKRUN is not used: --posix sync has no ring".to_string(),
        "--threads from env AEIOU_THREADS is not used: this run has one thread per actor".to_string(),
        format!("--mmap-mode from config {c} is not used: --posix sync maps nothing"),
        format!("--object-threads from config {c} is not used: no dataset or namespace is placed in an object store"),
        format!("--object-part-mib from config {c} is not used: no dataset or namespace is placed in an object store"),
    ] {
        assert!(text.contains(&format!("  WARNING: {w}\n")), "{w}\n{text}");
    }
    let rows = block(&text);
    for name in ["aio-depth", "sqpoll", "defer-taskrun", "threads", "mmap-mode", "object-threads", "object-part-mib"] {
        assert!(!rows.iter().any(|r| r.0 == name), "{name} passed over and still in the block: {rows:?}");
    }
    let doc = read(&json);
    assert!(doc["layers"]["options"].get("threads").is_none(), "{}", doc["layers"]);
    assert_eq!(doc["layers"]["warnings"].as_array().unwrap().len(), 7, "{}", doc["layers"]);
    // datagen, likewise, for the object engine's threads from the environment
    let (ok, text, err) = out(aeiou("datagen", "file_per_sample.ast.json", &["samples=30"]).arg("--root").arg(d.join("dg")).env("AEIOU_OBJECT_THREADS", "4"));
    assert!(ok, "{err}");
    assert!(text.contains("  WARNING: --object-threads from env AEIOU_OBJECT_THREADS is not used: no dataset or namespace is placed in an object store\n"), "{text}");
    // --root is required, from any layer: listed as missing (tests/usage.rs has the frame)
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]));
    assert!(!ok);
    assert!(err.contains("the following required arguments were not provided:\n  --root <DIR>  directory the abstract's paths are relative to\n"), "{err}");
    // a typed value that does not parse names the layer and the key
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "2"]).env("AEIOU_THREADS", "many"));
    assert!(!ok);
    assert!(err.contains("AEIOU_THREADS"), "{err}");
}

#[test]
fn two_ranks_record_every_host_and_print_the_differences_and_a_mismatch_names_the_field() {
    let root = tmpdir("two");
    let params = ["steps=4", "ckpt_every=2", "item_bytes=[1048576, 2097152, 1048576, 1048576]", "meta_bytes=65536"];
    let spawn = |rank: i64, port: u16, extra: &[&str], seed: &str| {
        let mut c = aeiou("run", "dcp_save.ast.json", &params);
        c.arg("--root").arg(&root).args(["--gpus", "2", "--time-scale", "0", "--ranks", "2", "--clean-namespaces"]).arg("--rank").arg(rank.to_string()).args(["--seed", seed]);
        c.arg("--coordinator").arg(format!("127.0.0.1:{port}")).arg("--report-json").arg(root.join(format!("report-{rank}.json"))).args(extra);
        c.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap()
    };
    let collect = |children: Vec<std::process::Child>| -> Vec<(bool, String, String)> {
        children.into_iter().map(|c| {
            let o = c.wait_with_output().unwrap();
            (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
        }).collect()
    };
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let c0 = spawn(0, port, &["--buffer-mib", "4"], "1");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let c1 = spawn(1, port, &["--buffer-mib", "2"], "1");
    let outs = collect(vec![c0, c1]);
    for (ok, text, err) in &outs {
        assert!(ok, "{text}\n{err}");
    }
    // rank 0 prints the options that differ: buffer-mib, rank, and the report file
    let t0 = &outs[0].1;
    assert!(t0.contains("options differing between hosts"), "{t0}");
    assert!(t0.contains("  buffer-mib: rank 0 = 4, rank 1 = 2"), "{t0}");
    assert!(t0.contains("  rank: rank 0 = 0, rank 1 = 1"), "{t0}");
    assert!(!t0.contains("  time-scale:"), "{t0}");
    assert!(!outs[1].1.contains("options differing between hosts"), "{}", outs[1].1);
    // rank 0's report has every host's block; rank 1's its own only
    let (d0, d1) = (read(&root.join("report-0.json")), read(&root.join("report-1.json")));
    let hosts = d0["hosts"].as_array().unwrap();
    assert_eq!(hosts.len(), 2);
    assert_eq!(hosts[1]["rank"], 1);
    assert_eq!(hosts[1]["layers"]["options"]["buffer-mib"]["value"], 2);
    assert_eq!(hosts[0]["layers"]["options"]["buffer-mib"]["value"], 4);
    assert_eq!(d0["layers"]["options"]["buffer-mib"]["value"], 4);
    assert!(d1.get("hosts").is_none());
    // every host's facts came with its Hello, and the memory block sums them
    for h in hosts {
        assert!(h["facts"]["memory"].as_u64().unwrap() > 0, "{h}");
    }
    assert_eq!(d0["memory"]["hosts"], 2);
    assert_eq!(d0["memory"]["memory"].as_u64(), Some(hosts.iter().map(|h| h["facts"]["memory"].as_u64().unwrap()).sum::<u64>()));
    assert!(d1.get("memory").is_none() && d1["facts"]["memory"].as_u64().is_some());
    assert_eq!(d1["layers"]["options"]["buffer-mib"]["value"], 2);
    // a differing seed is refused with the field named, on both ranks
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let c0 = spawn(0, port, &[], "1");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let c1 = spawn(1, port, &[], "2");
    let outs = collect(vec![c0, c1]);
    assert!(outs.iter().all(|(ok, _, _)| !ok));
    assert!(outs.iter().any(|(_, _, e)| e.contains("not the same run: seed 2 differs from rank 0's 1")), "{outs:?}");
}

#[test]
fn every_boolean_reads_no_x_in_the_help_and_parses_both_ways() {
    let help = |sub: &str| {
        let (ok, text, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).arg(sub).arg("--help"));
        assert!(ok, "{err}");
        text
    };
    let run = help("run");
    for flag in ["sqpoll-shared", "defer-taskrun", "coop-taskrun", "require-cold", "drop-caches", "clean-namespaces", "ignore-limits", "report-takes"] {
        assert!(run.contains(&format!("--[no-]{flag} ")), "{flag} not shown as --[no-]{flag}:\n{run}");
        assert!(!run.contains(&format!("--no-{flag}")), "the twin --no-{flag} is shown:\n{run}");
    }
    assert!(run.contains("Every boolean reads --[no-]x: --x turns it on, --no-x turns it off, the last one on the line wins."), "{run}");
    assert!(help("dry-run").contains("--[no-]metrics "));
    // every row of a heading group starts its description in the same column: the longest
    // name sets it, and a --[no-] name is never the longest in its group (the test fails when
    // a flag is added that makes it so, and the REFERENCE.md §14 note then needs the next-line layout)
    // the column the description starts in: past the name and the padding after it
    let column = |line: &str| {
        let indent = line.len() - line.trim_start().len();
        let body = line.trim_start();
        let gap = body.find("  ")?;
        let desc = body[gap..].len() - body[gap..].trim_start().len();
        Some(indent + gap + desc)
    };
    let mut group: Vec<(String, usize)> = Vec::new();
    let mut groups = Vec::new();
    for line in run.lines() {
        if line.starts_with("      --") {
            if let Some(c) = column(line) {
                group.push((line.to_string(), c));
            }
        } else if !line.starts_with(' ') && !group.is_empty() {
            groups.push(std::mem::take(&mut group));
        }
    }
    groups.push(group);
    for g in groups.iter().filter(|g| g.len() > 1) {
        let cols: std::collections::BTreeSet<usize> = g.iter().map(|(_, c)| *c).collect();
        // names longer than the column wrap their description to the next line; those rows
        // have no two-space gap after the name and are not counted
        let names_only: Vec<&(String, usize)> = g.iter().filter(|(l, c)| l.len() > *c).collect();
        let cols2: std::collections::BTreeSet<usize> = names_only.iter().map(|(_, c)| *c).collect();
        assert!(cols2.len() <= 1, "descriptions start in different columns {cols:?}:\n{}", g.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join("\n"));
    }
    // --x then --no-x is off, --no-x then --x is on, and the notation typed literally is explained
    let metrics = |flags: &[&str]| {
        let (ok, text, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).args(flags));
        assert!(ok, "{err}");
        row(&block(&text), "metrics").clone()
    };
    assert_eq!(metrics(&["--metrics", "--no-metrics"]), ("metrics".into(), "false".into(), "cli".into()));
    assert_eq!(metrics(&["--no-metrics", "--metrics"]), ("metrics".into(), "true".into(), "cli".into()));
    assert_eq!(metrics(&["--no-metrics"]), ("metrics".into(), "false".into(), "cli".into()));
    assert_eq!(metrics(&[]), ("metrics".into(), "false".into(), "default".into()));
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1", "--[no-]metrics"]));
    assert!(!ok);
    assert!(err.contains("type --metrics to turn it on or --no-metrics to turn it off"), "{err}");
    // the file and the environment say false, never no-
    let d = tmpdir("neg");
    let cfg = d.join("aeiou.toml");
    std::fs::write(&cfg, "[run]\nno-drop-caches = true\n").unwrap();
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--root").arg(&d).arg("--config").arg(&cfg));
    assert!(!ok);
    assert!(err.contains("[run] no-drop-caches") && err.contains("`drop-caches = false`"), "{err}");
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--root").arg(&d).env("AEIOU_NO_DROP_CACHES", "1"));
    assert!(!ok);
    assert!(err.contains("AEIOU_DROP_CACHES=true or false"), "{err}");
}

/// A config file shared between a fresh datagen and the changes of the dataset it wrote:
/// `--resume`, `--grow`, `--shrink`, `--settle` take the place from the manifest, so the file's
/// (or the environment's) `root` and `endpoint` are passed over with a warning in the block,
/// never refused as an option the subcommand does not have; on the command line they are still
/// a usage error.
#[test]
fn a_change_of_a_dataset_root_passes_over_the_place_from_the_lower_layers() {
    let d = tmpdir("dg-root-layers");
    let root = d.join("root");
    let cfg = d.join("aeiou.toml");
    std::fs::write(&cfg, format!("[datagen]\nroot = {:?}\nendpoint = []\nwriters = 2\n", root.display().to_string())).unwrap();
    let (ok, _, err) = out(aeiou("datagen", "file_per_sample.ast.json", &["samples=30"]).arg("--config").arg(&cfg));
    assert!(ok, "{err}");
    let at = root.join("train");
    let (ok, text, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--resume"]).arg(&at).arg("--config").arg(&cfg).env_remove("AEIOU_CONFIG"));
    assert!(ok && text.contains("the manifest says `complete`"), "{text}\n{err}");
    let (ok, text, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--grow", "40"]).arg(&at).env("AEIOU_CONFIG", &cfg));
    assert!(ok, "{text}\n{err}");
    let rows = block(&text);
    assert_eq!(row(&rows, "writers").2, format!("config {}", cfg.display()));
    assert!(!rows.iter().any(|r| r.0 == "root" || r.0 == "endpoint"), "{rows:?}");
    for name in ["root", "endpoint"] {
        assert!(text.contains(&format!("WARNING: --{name} from config {} is not used: --grow takes the dataset's place from the manifest at its root", cfg.display())), "{text}");
    }
    // the environment's, likewise
    let (ok, text, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--shrink", "35"]).arg(&at).env_remove("AEIOU_CONFIG").env("AEIOU_ROOT", d.join("elsewhere")));
    assert!(ok, "{text}\n{err}");
    assert!(text.contains("WARNING: --root from env AEIOU_ROOT is not used: --shrink takes the dataset's place from the manifest at its root"), "{text}");
    assert!(!text.contains("--endpoint from"), "{text}");
    // the command line's is refused
    let (ok, _, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).args(["datagen", "--resume"]).arg(&at).arg("--root").arg(&root).env_remove("AEIOU_CONFIG").env_remove("AEIOU_ROOT"));
    assert!(!ok && err.contains("--root given too"), "{err}");
}

/// RC1 review: a variable or an argument that is not UTF-8 never panics (it was exit 101 for any
/// variable at all): a path takes the bytes, text is a usage error naming it; an empty value from
/// the environment or the file is a usage error (an empty root was the working directory); a
/// fixed boolean's negation below the command line says it is the command line's; a config type
/// error names the TOML type; and the block shows what a default comes to.
#[test]
fn odd_values_from_the_lower_layers_and_the_defaults_in_effect() {
    use std::os::unix::ffi::OsStrExt;
    let d = tmpdir("odd");
    let raw = |b: &[u8]| std::ffi::OsStr::from_bytes(b).to_os_string();
    // anyone's variable that is not UTF-8
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).env("AEIOU_TEST_NOT_UTF8_FOO", raw(b"a\xff")).env("NOT_OURS", raw(b"b\xff")));
    assert!(ok, "{err}");
    // a root that is not UTF-8, from the command line and from the environment: written there
    let mut at = d.as_os_str().to_os_string();
    at.push(raw(b"/cli\xff"));
    let (ok, _, err) = out(aeiou("datagen", "file_per_sample.ast.json", &["samples=30"]).arg("--root").arg(&at));
    assert!(ok, "{err}");
    assert!(Path::new(&at).join("train/.aeiou-dataset.json").exists());
    let mut at = d.as_os_str().to_os_string();
    at.push(raw(b"/env\xff"));
    let (ok, _, err) = out(aeiou("datagen", "file_per_sample.ast.json", &["samples=30"]).env("AEIOU_ROOT", &at));
    assert!(ok, "{err}");
    assert!(Path::new(&at).join("train/.aeiou-dataset.json").exists());
    // an abstract whose path is not UTF-8
    let mut dir = d.as_os_str().to_os_string();
    dir.push(raw(b"/d\xff"));
    std::fs::create_dir_all(&dir).unwrap();
    let ast = Path::new(&dir).join("a.ast.json");
    std::fs::copy(crate::common::workload_ast("file_per_sample.ast.json"), &ast).unwrap();
    let (ok, text, err) = out(Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("check").arg(&ast).env_remove("AEIOU_CONFIG"));
    assert!(ok && text.contains("ok   a.ast.json"), "{text}\n{err}");
    // text that is not UTF-8: a usage error naming it, from either layer
    let code = |c: &mut Command| c.output().unwrap().status.code();
    let mut c = aeiou("dry-run", "file_per_sample.ast.json", &SMALL);
    c.args(["--gpus", "1", "--param"]).arg(raw(b"steps=\xff"));
    assert_eq!(code(&mut c), Some(2));
    let (_, _, err) = out(&mut c);
    assert!(err.starts_with("aeiou dry-run: \"steps=\\xFF\": not UTF-8"), "{err}");
    let mut c = aeiou("run", "file_per_sample.ast.json", &SMALL);
    c.args(["--gpus", "1"]).arg("--root").arg(&d).env("AEIOU_COORDINATOR", raw(b"h\xff:1"));
    assert_eq!(code(&mut c), Some(2));
    let (_, _, err) = out(&mut c);
    assert!(err.starts_with("aeiou run: AEIOU_COORDINATOR: \"h\\xFF:1\" is not UTF-8"), "{err}");
    // empty, from the environment and the file: refused, nothing written in the working directory
    let cwd = d.join("cwd");
    std::fs::create_dir_all(&cwd).unwrap();
    let mut c = aeiou("datagen", "file_per_sample.ast.json", &["samples=30"]);
    c.env("AEIOU_ROOT", "").current_dir(&cwd);
    assert_eq!(code(&mut c), Some(2));
    let (_, _, err) = out(&mut c);
    assert!(err.starts_with("aeiou datagen: AEIOU_ROOT: empty"), "{err}");
    assert_eq!(std::fs::read_dir(&cwd).unwrap().count(), 0);
    let cfg = d.join("aeiou.toml");
    std::fs::write(&cfg, "[run]\nroot = \"\"\n").unwrap();
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--config").arg(&cfg).current_dir(&cwd));
    assert!(!ok && err.contains("[run] root: empty"), "{err}");
    for var in ["AEIOU_REPORT_JSON", "AEIOU_ENDPOINT", "AEIOU_CONFIG"] {
        let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--root").arg(&d).env(var, ""));
        assert!(!ok && err.starts_with(&format!("aeiou run: {var}: empty")), "{var}: {err}");
    }
    // the negation of a fixed boolean, from the environment and the file
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).env("AEIOU_NO_METRICS", "1"));
    assert!(!ok && err.contains("AEIOU_NO_METRICS: --metrics is the command line's alone (`--no-metrics` there)") && !err.contains("AEIOU_METRICS="), "{err}");
    std::fs::write(&cfg, "[dry-run]\nno-metrics = true\n").unwrap();
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--config").arg(&cfg));
    assert!(!ok && err.contains("[dry-run] no-metrics: --metrics is the command line's alone"), "{err}");
    let (ok, _, err) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--root").arg(&d).env("AEIOU_NO_CLEAN_NAMESPACES", "1"));
    assert!(!ok && err.contains("AEIOU_NO_CLEAN_NAMESPACES: --clean-namespaces is the command line's alone"), "{err}");
    // a type error: the TOML type and value
    std::fs::write(&cfg, "[dry-run]\nthreads = \"3\"\n").unwrap();
    let (ok, _, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--config").arg(&cfg));
    assert!(!ok && err.contains("[dry-run] threads: expected an integer, got the string \"3\""), "{err}");
    // the defaults in effect
    let cores = std::thread::available_parallelism().unwrap().get();
    let (ok, text, err) = out(aeiou("dry-run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]));
    assert!(ok, "{err}");
    assert_eq!(row(&block(&text), "threads").1, format!("{cores} (all cores)"));
    let (ok, text, err) = out(aeiou("datagen", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1"]).arg("--root").arg(d.join("dg")));
    assert!(ok, "{err}");
    assert_eq!(row(&block(&text), "writers").1, format!("{cores} (all cores)"));
    let (_, text, _) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1", "--time-scale", "0"]).arg("--root").arg(d.join("dg")));
    assert_eq!(row(&block(&text), "threads").1, "one per actor");
    let (_, text, _) = out(aeiou("run", "file_per_sample.ast.json", &SMALL).args(["--gpus", "1", "--time-scale", "0", "--posix", "libaio"]).arg("--root").arg(d.join("dg")));
    let rows = block(&text);
    assert_eq!(row(&rows, "threads").1, format!("{cores} (one per core, at most one per actor instance)"));
    assert_eq!(row(&rows, "aio-depth").1, "256");
}

/// A `--root` given as a bare relative name that does not exist yet (`dg1`, not `./dg1`): the
/// space check finds the free space of the current directory, the parent of an empty path, and
/// datagen writes the dataset there (RC1 review).
#[test]
fn datagen_takes_a_bare_relative_root_that_does_not_exist_yet() {
    let d = tmpdir("dg-bare-root");
    let (ok, text, err) = out(aeiou("datagen", "file_per_sample.ast.json", &["samples=30"]).args(["--root", "dg1"]).current_dir(&d).env_remove("AEIOU_ROOT"));
    assert!(ok, "{text}\n{err}");
    assert!(d.join("dg1/train/.aeiou-dataset.json").exists(), "{text}\n{err}");
}
