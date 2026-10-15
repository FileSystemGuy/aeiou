//! The option layers (`runner/REFERENCE.md` §14, `DESIGN_REVIEW.md` §3.60): a value comes from the
//! command line, else from the environment (`AEIOU_<FLAG>`, the long flag upper-cased with
//! underscores), else from the config file (TOML, named by `--config FILE` or `AEIOU_CONFIG`,
//! one table per subcommand, keys spelled as the long flags), else from the compiled default.
//! A higher layer replaces a lower one's value.
//!
//! Two kinds of option. A **fixed** option is the command line's alone: the environment and
//! the file are refused when they name it. Those are the ones the fingerprint, a dataset id,
//! or a safety check depends on. A **layered** option may come from any layer. Every
//! subcommand prints what it resolved and where each value came from (`print`), and `aeiou
//! run` records the same in its report (`json`) and sends it to the coordinator.
//!
//! The file is never searched for: no working directory, home, or XDG path. A config file
//! nobody remembers is how two hosts of one run come to differ.
//!
//! Every refusal here is a usage error (`usage.rs`): the invocation is wrong and nothing has
//! run, so it is printed in the suite's frame and exits 2.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::usage::err;
use serde_json::{json, Value};

pub const ENV_PREFIX: &str = "AEIOU_";
/// The environment variable that names the config file (under `--config`).
pub const ENV_CONFIG: &str = "AEIOU_CONFIG";
/// `AEIOU_*` names that are not options: the config file itself, and the names the other tools
/// of the family and the launcher use, the coordinator's test hook (`coord::timing`), and the
/// tests' `AEIOU_KEEP_TEST_DIRS` (`tests/common`), which a spawned runner inherits. Any
/// other unrecognized `AEIOU_*` is reported as a warning in the options block.
pub const RESERVED_ENV: &[&str] = &[ENV_CONFIG, "AEIOU_SCHEMA_DIR", "AEIOU_RUNNER", "AEIOU_RSH", "AEIOU_COORD_HEARTBEAT_MS", "AEIOU_KEEP_TEST_DIRS"];
/// The subcommands a config file may have a table for.
pub const SUBCOMMANDS: &[&str] = &["check", "dry-run", "datagen", "run"];
/// Every option name of every subcommand (the long flag, or the positional's name), so an
/// `AEIOU_*` for another subcommand's option is left alone, and anything else warns. A test in
/// `main.rs` keeps it equal to the clap definitions.
pub const ALL_OPTIONS: &[&str] = &[
    "files", "abstract", "param", "params-file", "gpus", "seed", "posix", "cache", "s3", "root", "endpoint", "threads", "buffer-mib", "write-compress", "time-scale",
    "iowq-max-workers", "sqpoll", "sqpoll-shared", "defer-taskrun", "coop-taskrun", "aio-depth", "mmap-mode", "mmap-consume", "object-threads", "object-part-mib", "rank", "ranks",
    "coordinator", "rank-rotate", "subset", "expect-fingerprint", "expect-dataset-id", "max-gap", "require-cold", "drop-caches", "clean-namespaces",
    "ignore-limits", "ignore-space", "report-json", "report-takes", "dedupe", "compress", "dataset", "gpu", "steps", "limit", "metrics", "metrics-block",
    "metrics-sample", "metrics-json", "writers", "progress-secs", "resume", "grow", "shrink", "settle", "config",
];

/// Where a value came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Cli,
    Env(String),
    Config(PathBuf),
    Default,
}

impl Source {
    pub fn describe(&self) -> String {
        match self {
            Source::Cli => "cli".into(),
            Source::Env(n) => format!("env {n}"),
            Source::Config(p) => format!("config {}", p.display()),
            Source::Default => "default".into(),
        }
    }
}

/// One resolved option: its name, its value as the block shows it and as JSON, its source.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub shown: String,
    pub value: Value,
    pub source: Source,
}

/// The config file, parsed: every top-level key must be a subcommand's table.
#[derive(Debug)]
pub struct ConfigFile {
    pub path: PathBuf,
    pub sha256: String,
    tables: toml::Table,
}

impl ConfigFile {
    pub fn load(path: &Path) -> Result<ConfigFile> {
        let text = std::fs::read_to_string(path).map_err(|e| err(format!("config {}: {e}", path.display())))?;
        let sha256 = {
            use sha2::Digest;
            let d = sha2::Sha256::digest(text.as_bytes());
            d.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        let tables: toml::Table = text.parse::<toml::Table>().map_err(|e| err(format!("config {}: {e}", path.display())))?;
        for (k, v) in &tables {
            if !SUBCOMMANDS.contains(&k.as_str()) {
                crate::usage!(
                    "config {}: `{k}` at the top level; options live in a subcommand's table ([run], [dry-run], [datagen], [check]), spelled as the long flags",
                    path.display()
                );
            }
            if !v.is_table() {
                crate::usage!("config {}: [{k}] must be a table", path.display());
            }
        }
        Ok(ConfigFile { path: path.to_path_buf(), sha256, tables })
    }

    fn table(&self, sub: &str) -> Option<&toml::Table> {
        self.tables.get(sub).and_then(|v| v.as_table())
    }
}

/// A value a layered option can take, from the environment's text or the file's TOML value.
pub trait Layered: Sized + Show {
    fn from_env_str(s: &str) -> Result<Self>;
    fn from_toml(v: &toml::Value) -> Result<Self>;
    /// The environment's value as the process has it: text, refused when it is not UTF-8 (a
    /// path takes the bytes as they are).
    fn from_env(s: &OsStr) -> Result<Self> {
        match s.to_str() {
            Some(t) => Self::from_env_str(t),
            None => anyhow::bail!("{s:?} is not UTF-8"),
        }
    }
}

/// A TOML value as a message names it: its type and its value (`the string "3"`).
pub fn toml_described(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => format!("the string {s:?}"),
        toml::Value::Integer(i) => format!("the integer {i}"),
        toml::Value::Float(f) => format!("the float {f}"),
        toml::Value::Boolean(b) => format!("the boolean {b}"),
        toml::Value::Datetime(d) => format!("the datetime {d}"),
        toml::Value::Array(a) => format!("an array of {} value(s)", a.len()),
        toml::Value::Table(_) => "a table".into(),
    }
}

/// An empty value where one is needed is refused, as the command line refuses `--root ""`
/// (an empty path would be the working directory).
fn non_empty(s: &str) -> Result<()> {
    if s.trim().is_empty() {
        anyhow::bail!("empty, and a value is required (leave it unset for none)");
    }
    Ok(())
}

/// How an option's value is shown in the block and recorded as JSON.
pub trait Show {
    fn shown(&self) -> String;
    fn json(&self) -> Value;
}

macro_rules! int_layered {
    ($($t:ty),*) => {$(
        impl Show for $t {
            fn shown(&self) -> String { self.to_string() }
            fn json(&self) -> Value { json!(*self) }
        }
        impl Layered for $t {
            fn from_env_str(s: &str) -> Result<Self> { s.trim().parse::<$t>().map_err(|e| anyhow::anyhow!("{s:?}: {e}")) }
            fn from_toml(v: &toml::Value) -> Result<Self> {
                let i = v.as_integer().ok_or_else(|| anyhow::anyhow!("expected an integer, got {}", toml_described(v)))?;
                <$t>::try_from(i).map_err(|_| anyhow::anyhow!("{i} is out of range"))
            }
        }
    )*};
}
int_layered!(i64, u64, u32, usize);

impl Show for f64 {
    fn shown(&self) -> String {
        self.to_string()
    }
    fn json(&self) -> Value {
        json!(*self)
    }
}
impl Layered for f64 {
    fn from_env_str(s: &str) -> Result<Self> {
        s.trim().parse::<f64>().map_err(|e| anyhow::anyhow!("{s:?}: {e}"))
    }
    fn from_toml(v: &toml::Value) -> Result<Self> {
        match v {
            toml::Value::Float(f) => Ok(*f),
            toml::Value::Integer(i) => Ok(*i as f64),
            _ => crate::usage!("expected a number, got {}", toml_described(v)),
        }
    }
}

impl Show for bool {
    fn shown(&self) -> String {
        self.to_string()
    }
    fn json(&self) -> Value {
        json!(*self)
    }
}
impl Layered for bool {
    fn from_env_str(s: &str) -> Result<Self> {
        match s.trim() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            other => crate::usage!("{other:?}: expected true or false"),
        }
    }
    fn from_toml(v: &toml::Value) -> Result<Self> {
        v.as_bool().ok_or_else(|| anyhow::anyhow!("expected true or false, got {}", toml_described(v)))
    }
}

impl Show for String {
    fn shown(&self) -> String {
        self.clone()
    }
    fn json(&self) -> Value {
        json!(self)
    }
}
impl Layered for String {
    fn from_env_str(s: &str) -> Result<Self> {
        non_empty(s)?;
        Ok(s.to_string())
    }
    fn from_toml(v: &toml::Value) -> Result<Self> {
        let s = v.as_str().ok_or_else(|| anyhow::anyhow!("expected a string, got {}", toml_described(v)))?;
        non_empty(s)?;
        Ok(s.to_string())
    }
}

impl Show for PathBuf {
    fn shown(&self) -> String {
        self.display().to_string()
    }
    fn json(&self) -> Value {
        json!(self.display().to_string())
    }
}
impl Layered for PathBuf {
    fn from_env_str(s: &str) -> Result<Self> {
        Self::from_env(OsStr::new(s))
    }
    /// A path is the bytes as the environment has them, UTF-8 or not, but never empty.
    fn from_env(s: &OsStr) -> Result<Self> {
        non_empty(&s.to_string_lossy())?;
        Ok(PathBuf::from(s))
    }
    fn from_toml(v: &toml::Value) -> Result<Self> {
        let s = v.as_str().ok_or_else(|| anyhow::anyhow!("expected a path string, got {}", toml_described(v)))?;
        non_empty(s)?;
        Ok(PathBuf::from(s))
    }
}

/// A list: whitespace-separated in the environment, an array of strings (or one string) in
/// the file. `[]` is the empty list; an empty string is refused, in the file as in the
/// environment.
impl Layered for Vec<String> {
    fn from_env_str(s: &str) -> Result<Self> {
        non_empty(s)?;
        Ok(s.split_whitespace().map(str::to_string).collect())
    }
    fn from_toml(v: &toml::Value) -> Result<Self> {
        let one = |x: &toml::Value| -> Result<String> {
            let s = x.as_str().ok_or_else(|| anyhow::anyhow!("expected strings, got {}", toml_described(x)))?;
            non_empty(s)?;
            Ok(s.to_string())
        };
        match v {
            toml::Value::String(_) => Ok(vec![one(v)?]),
            toml::Value::Array(a) => a.iter().map(one).collect(),
            _ => crate::usage!("expected a string or an array of strings, got {}", toml_described(v)),
        }
    }
}

impl<T: Show> Show for Vec<T> {
    fn shown(&self) -> String {
        if self.is_empty() {
            "none".into()
        } else {
            self.iter().map(Show::shown).collect::<Vec<_>>().join(" ")
        }
    }
    fn json(&self) -> Value {
        Value::Array(self.iter().map(Show::json).collect())
    }
}

impl<T: Show> Show for Option<T> {
    fn shown(&self) -> String {
        match self {
            Some(v) => v.shown(),
            None => "none".into(),
        }
    }
    fn json(&self) -> Value {
        match self {
            Some(v) => v.json(),
            None => Value::Null,
        }
    }
}

/// The resolver of one invocation: the subcommand's table of the config file, the `AEIOU_*`
/// environment, and the entries resolved so far, in the order they will be printed.
#[derive(Debug)]
pub struct Layers {
    sub: String,
    pub config: Option<ConfigFile>,
    config_source: Source,
    /// The `AEIOU_*` variables, their values as the process has them (a value need not be
    /// UTF-8 until an option reads it as text); a name that is not UTF-8 is kept lossily, and
    /// is no option.
    env: BTreeMap<String, OsString>,
    pub entries: Vec<Entry>,
    env_used: BTreeSet<String>,
    keys_used: BTreeSet<String>,
    /// The fixed options of this invocation, for the refusal of their negation below the
    /// command line.
    fixed_used: BTreeSet<String>,
    pub warnings: Vec<String>,
}

pub fn env_name(option: &str) -> String {
    format!("{ENV_PREFIX}{}", option.to_ascii_uppercase().replace('-', "_"))
}

impl Layers {
    /// The layers of subcommand `sub`, with the config file `--config` named (else
    /// `AEIOU_CONFIG`, else none), over the process environment.
    pub fn new(sub: &str, config_cli: Option<&Path>) -> Result<Layers> {
        // `vars_os`: a variable that is not UTF-8, ours or anyone's, is no reason to stop
        let env: BTreeMap<String, OsString> =
            std::env::vars_os().filter(|(k, _)| k.as_encoded_bytes().starts_with(ENV_PREFIX.as_bytes())).map(|(k, v)| (k.to_string_lossy().into_owned(), v)).collect();
        Layers::with_env(sub, config_cli, env)
    }

    /// `new`, over a given environment (the tests').
    pub fn with_env(sub: &str, config_cli: Option<&Path>, env: BTreeMap<String, OsString>) -> Result<Layers> {
        if !SUBCOMMANDS.contains(&sub) {
            crate::usage!("no subcommand `{sub}`");
        }
        let mut env_used = BTreeSet::new();
        let (config_path, config_source) = match config_cli {
            Some(p) => (Some(p.to_path_buf()), Source::Cli),
            None => match env.get(ENV_CONFIG) {
                Some(p) => {
                    env_used.insert(ENV_CONFIG.to_string());
                    let p = PathBuf::from_env(p).map_err(|e| err(format!("{ENV_CONFIG}: {e:#}")))?;
                    (Some(p), Source::Env(ENV_CONFIG.into()))
                }
                None => (None, Source::Default),
            },
        };
        let config = match &config_path {
            Some(p) => Some(ConfigFile::load(p)?),
            None => None,
        };
        Ok(Layers {
            sub: sub.to_string(),
            config,
            config_source,
            env,
            entries: Vec::new(),
            env_used,
            keys_used: BTreeSet::new(),
            fixed_used: BTreeSet::new(),
            warnings: Vec::new(),
        })
    }

    fn table_value(&self, name: &str) -> Option<&toml::Value> {
        self.config.as_ref().and_then(|c| c.table(&self.sub)).and_then(|t| t.get(name))
    }

    fn config_path(&self) -> PathBuf {
        self.config.as_ref().map(|c| c.path.clone()).unwrap_or_default()
    }

    /// A fixed option: the command line's or the default, never the environment's or the
    /// file's, which are refused when they name it.
    pub fn fixed<T: Show>(&mut self, name: &str, value: &T, given: bool) -> Result<()> {
        self.fixed_used.insert(name.to_string());
        let env = env_name(name);
        if self.env.contains_key(&env) {
            crate::usage!("{env} is set, but --{name} is the command line's alone: nothing the fingerprint, a dataset id, or a safety check depends on may come from the environment or the config file");
        }
        if self.table_value(name).is_some() {
            crate::usage!(
                "config {}: [{}] {name} is set, but --{name} is the command line's alone: nothing the fingerprint, a dataset id, or a safety check depends on may come from the environment or the config file",
                self.config_path().display(),
                self.sub
            );
        }
        self.entries.push(Entry { name: name.into(), shown: value.shown(), value: value.json(), source: if given { Source::Cli } else { Source::Default } });
        Ok(())
    }

    /// A layered option: the command line's value, else the environment's, else the file's,
    /// else `default`. Returns the value in effect.
    pub fn layered<T: Layered>(&mut self, name: &str, cli: Option<T>, default: Option<T>) -> Result<Option<T>> {
        let env = env_name(name);
        self.keys_used.insert(name.to_string());
        let (value, source) = if let Some(v) = cli {
            (Some(v), Source::Cli)
        } else if let Some(s) = self.env.get(&env) {
            self.env_used.insert(env.clone());
            (Some(T::from_env(s).map_err(|e| err(format!("{env}: {e:#}")))?), Source::Env(env))
        } else if let Some(v) = self.table_value(name) {
            let path = self.config_path();
            (Some(T::from_toml(v).map_err(|e| err(format!("config {}: [{}] {name}: {e:#}", path.display(), self.sub)))?), Source::Config(path))
        } else {
            (default, Source::Default)
        };
        self.entries.push(Entry { name: name.into(), shown: value.shown(), value: value.json(), source });
        Ok(value)
    }

    /// A layered option this invocation's mode does not take (`datagen --resume` and the other
    /// changes of a dataset root take `--root` and `--endpoint` from the manifest): the caller
    /// refuses it on the command line; a value from the environment or the file, which may
    /// serve the subcommand's other modes, is passed over with a warning that says `why`, and
    /// no entry is printed for it.
    pub fn unused(&mut self, name: &str, why: &str) {
        let env = env_name(name);
        self.keys_used.insert(name.to_string());
        let from = if self.env.contains_key(&env) {
            Some(Source::Env(env))
        } else if self.table_value(name).is_some() {
            Some(Source::Config(self.config_path()))
        } else {
            None
        };
        if let Some(s) = from {
            self.warnings.push(format!("--{name} from {} is not used: {why}", s.describe()));
        }
    }

    /// A layered option resolved already that this run cannot use (a backend knob the run's
    /// API has no use for, an object-engine option with no name in an object store): from the
    /// command line it is the caller's to refuse, and this returns false; from the environment
    /// or the file, which may serve other runs (one file for a cluster), it is passed over
    /// with a warning that says `why`, its entry is dropped, and this returns true.
    pub fn pass_over(&mut self, name: &str, why: &str) -> bool {
        let Some(i) = self.entries.iter().position(|e| e.name == name) else { return false };
        match self.entries[i].source.clone() {
            Source::Cli | Source::Default => false,
            s => {
                self.entries.remove(i);
                self.warnings.push(format!("--{name} from {} is not used: {why}", s.describe()));
                true
            }
        }
    }

    /// The value a default resolves to, shown in the block in place of `none` when it is known
    /// (`writers = 20 (all cores)`); the JSON keeps the default as it was given.
    pub fn show_default(&mut self, name: &str, shown: String) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.name == name && e.source == Source::Default) {
            e.shown = shown;
        }
    }

    /// A layered boolean: `--name` on the command line is true, `--name=false` false, absent
    /// is the lower layers'. Returns the value in effect.
    pub fn flag(&mut self, name: &str, cli: Option<bool>, default: bool) -> Result<bool> {
        Ok(self.layered(name, cli, Some(default))?.unwrap_or(default))
    }

    /// After the last option: unknown keys in this subcommand's table are refused; `AEIOU_*`
    /// names that are no option of any subcommand are warned about; the config file itself
    /// is recorded as an entry.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(c) = &self.config {
            if let Some(t) = c.table(&self.sub) {
                // a negation, first: the file writes a boolean under its own name, and a fixed
                // one not at all
                if let Some(k) = t.keys().find(|k| k.starts_with("no-")) {
                    let option = &k[3..];
                    if self.fixed_used.contains(option) {
                        crate::usage!(
                            "config {}: [{}] {k}: --{option} is the command line's alone (`--no-{option}` there): nothing the fingerprint, a dataset id, or a safety check depends on may come from the environment or the config file",
                            c.path.display(),
                            self.sub
                        );
                    }
                    crate::usage!("config {}: [{}] {k}: a boolean is written as its name with true or false (`{option} = false`); `--no-x` is the command line's negation", c.path.display(), self.sub);
                }
                for k in t.keys() {
                    if !self.keys_used.contains(k) {
                        if ALL_OPTIONS.contains(&k.as_str()) {
                            crate::usage!("config {}: [{}] {k}: not an option of `aeiou {}` (or the command line's alone)", c.path.display(), self.sub, self.sub);
                        }
                        crate::usage!("config {}: [{}] {k}: unknown option (keys are spelled as the long flags)", c.path.display(), self.sub);
                    }
                }
            }
        }
        for k in self.env.keys() {
            if RESERVED_ENV.contains(&k.as_str()) || self.env_used.contains(k) {
                continue;
            }
            if let Some(rest) = k.strip_prefix("AEIOU_NO_") {
                let option = rest.to_ascii_lowercase().replace('_', "-");
                if self.fixed_used.contains(&option) {
                    crate::usage!("{k}: --{option} is the command line's alone (`--no-{option}` there): nothing the fingerprint, a dataset id, or a safety check depends on may come from the environment or the config file");
                }
                if ALL_OPTIONS.contains(&option.as_str()) {
                    crate::usage!("{k}: a boolean is set in the environment as {}=true or false; `--no-x` is the command line's negation", env_name(&option));
                }
            }
            let as_option = k.trim_start_matches(ENV_PREFIX).to_ascii_lowercase().replace('_', "-");
            if !ALL_OPTIONS.contains(&as_option.as_str()) {
                self.warnings.push(format!("{k} is set and is no option of any subcommand; ignored"));
            }
        }
        let (shown, value) = match &self.config {
            Some(c) => (format!("{} (sha256 {}…)", c.path.display(), &c.sha256[..16]), json!({"path": c.path.display().to_string(), "sha256": c.sha256})),
            None => ("none".to_string(), Value::Null),
        };
        self.entries.push(Entry { name: "config".into(), shown, value, source: self.config_source.clone() });
        Ok(())
    }

    /// For a message that refuses or cross-checks the named options: where the ones that did
    /// not come from the command line came from, as ` (from env AEIOU_X, config PATH)`, or
    /// nothing when the user typed them all (`default` is named too: a default the backend
    /// cannot take is the program's business, and the message says so).
    pub fn from(&self, names: &[&str]) -> String {
        let plain = self.from_plain(names);
        if plain.is_empty() {
            String::new()
        } else {
            format!(" ({plain})")
        }
    }

    /// `from` without the parentheses: `--x from env AEIOU_X, --y by default`, or nothing.
    pub fn from_plain(&self, names: &[&str]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for n in names {
            if let Some(e) = self.entries.iter().find(|e| e.name == *n) {
                let d = match &e.source {
                    Source::Cli => continue,
                    Source::Default => format!("--{n} by default"),
                    other => format!("--{n} from {}", other.describe()),
                };
                if !parts.contains(&d) {
                    parts.push(d);
                }
            }
        }
        parts.join(", ")
    }

    /// The block every invocation prints: one line per option, its value and its source, the
    /// warnings last, then an empty line.
    pub fn print(&self, out: &mut dyn Write) -> std::io::Result<()> {
        writeln!(out, "options (cli > env > config > default)")?;
        let width = self.entries.iter().map(|e| e.name.len() + 3 + e.shown.len()).max().unwrap_or(0).min(72);
        for e in &self.entries {
            let lhs = format!("{} = {}", e.name, e.shown);
            writeln!(out, "  {lhs:<width$}  [{}]", e.source.describe())?;
        }
        for w in &self.warnings {
            writeln!(out, "  WARNING: {w}")?;
        }
        writeln!(out)
    }

    /// The same as JSON (the report's `layers`, the coordinator's `Hello`): the config file,
    /// the environment variables that contributed, every option with its value and source.
    pub fn json(&self) -> Value {
        let options: serde_json::Map<String, Value> = self.entries.iter().map(|e| (e.name.clone(), json!({"value": e.value, "source": e.source.describe()}))).collect();
        let config = self.config.as_ref().map(|c| json!({"path": c.path.display().to_string(), "sha256": c.sha256}));
        json!({"config": config, "env": self.env_used, "options": options, "warnings": self.warnings})
    }
}

/// The options whose values differ between hosts, from each host's `layers` JSON: one row per
/// option, the value per rank. Rows are over the union of the option names.
pub fn differences(hosts: &[(i64, &Value)]) -> Vec<(String, Vec<(i64, String)>)> {
    let names: BTreeSet<String> = hosts.iter().flat_map(|(_, l)| l["options"].as_object().into_iter().flat_map(|o| o.keys().cloned())).collect();
    let mut rows = Vec::new();
    for name in names {
        let values: Vec<(i64, &Value)> = hosts.iter().map(|(r, l)| (*r, &l["options"][&name]["value"])).collect();
        if values.windows(2).any(|w| w[0].1 != w[1].1) {
            rows.push((name, values.into_iter().map(|(r, v)| (r, shown(v))).collect()));
        }
    }
    rows
}

fn shown(v: &Value) -> String {
    match v {
        Value::Null => "none".into(),
        Value::String(s) => s.clone(),
        Value::Array(a) if a.is_empty() => "none".into(),
        Value::Array(a) => a.iter().map(shown).collect::<Vec<_>>().join(" "),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, OsString> {
        pairs.iter().map(|(k, v)| (k.to_string(), OsString::from(v))).collect()
    }

    fn write(dir: &Path, text: &str) -> PathBuf {
        let p = dir.join("aeiou.toml");
        std::fs::write(&p, text).unwrap();
        p
    }

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("aeiou-options-{}-{}", std::process::id(), rand_suffix()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn rand_suffix() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn precedence_cli_env_config_default() {
        let d = tmp();
        let cfg = write(&d, "[run]\nthreads = 3\nbuffer-mib = 16\nrequire-cold = true\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[("AEIOU_THREADS", "5"), ("AEIOU_TIME_SCALE", "0.5")])).unwrap();
        assert_eq!(l.layered::<usize>("threads", Some(7), None).unwrap(), Some(7));
        assert_eq!(l.layered::<usize>("buffer-mib", None, Some(8)).unwrap(), Some(16));
        assert_eq!(l.layered::<f64>("time-scale", None, Some(1.0)).unwrap(), Some(0.5));
        assert_eq!(l.layered::<i64>("rank", None, Some(0)).unwrap(), Some(0));
        assert!(l.flag("require-cold", None, false).unwrap());
        assert!(!l.flag("drop-caches", None, false).unwrap());
        l.finish().unwrap();
        let src: Vec<(&str, String)> = l.entries.iter().map(|e| (e.name.as_str(), e.source.describe())).collect();
        assert_eq!(src[0], ("threads", "cli".to_string()));
        assert_eq!(src[1], ("buffer-mib", format!("config {}", cfg.display())));
        assert_eq!(src[2], ("time-scale", "env AEIOU_TIME_SCALE".to_string()));
        assert_eq!(src[3], ("rank", "default".to_string()));
        assert_eq!(src[4], ("require-cold", format!("config {}", cfg.display())));
        assert_eq!(src[6], ("config", "cli".to_string()));
        // env over config, even when both set the same option
        let mut l = Layers::with_env("run", Some(&cfg), env(&[("AEIOU_THREADS", "5")])).unwrap();
        assert_eq!(l.layered::<usize>("threads", None, None).unwrap(), Some(5));
        // the negation: the file turns it on, the command line turns it off
        assert!(!l.flag("require-cold", Some(false), false).unwrap());
        let j = l.json();
        assert_eq!(j["options"]["threads"]["source"], "env AEIOU_THREADS");
        assert_eq!(j["env"], json!(["AEIOU_THREADS"]));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn fixed_options_are_refused_from_the_lower_layers() {
        let d = tmp();
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_SEED", "3")])).unwrap();
        let e = l.fixed("seed", &3u64, false).unwrap_err().to_string();
        assert!(e.contains("AEIOU_SEED") && e.contains("command line's alone"), "{e}");
        let cfg = write(&d, "[run]\ngpus = 8\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[])).unwrap();
        let e = l.fixed("gpus", &8i64, true).unwrap_err().to_string();
        assert!(e.contains("[run] gpus") && e.contains("command line's alone"), "{e}");
        // the same key under another subcommand's table is that subcommand's business
        let mut l = Layers::with_env("dry-run", Some(&cfg), env(&[])).unwrap();
        l.fixed("gpus", &8i64, true).unwrap();
        l.finish().unwrap();
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn unknown_keys_and_top_level_keys_are_refused_and_unknown_env_warns() {
        let d = tmp();
        let cfg = write(&d, "threads = 3\n");
        let e = Layers::with_env("run", Some(&cfg), env(&[])).unwrap_err().to_string();
        assert!(e.contains("top level"), "{e}");
        let cfg = write(&d, "[run]\nthread = 3\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[])).unwrap();
        l.layered::<usize>("threads", None, None).unwrap();
        let e = l.finish().unwrap_err().to_string();
        assert!(e.contains("[run] thread") && e.contains("unknown"), "{e}");
        let cfg = write(&d, "[run]\nmetrics-block = 3\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[])).unwrap();
        let e = l.finish().unwrap_err().to_string();
        assert!(e.contains("not an option of `aeiou run`"), "{e}");
        let cfg = write(&d, "[nope]\nthreads = 3\n");
        let e = Layers::with_env("run", Some(&cfg), env(&[])).unwrap_err().to_string();
        assert!(e.contains("`nope`"), "{e}");
        // the negation is the command line's: the file and the environment say false
        let cfg = write(&d, "[run]\nno-require-cold = true\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[])).unwrap();
        let e = l.finish().unwrap_err().to_string();
        assert!(e.contains("no-require-cold") && e.contains("`require-cold = false`"), "{e}");
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_NO_REQUIRE_COLD", "1")])).unwrap();
        let e = l.finish().unwrap_err().to_string();
        assert!(e.contains("AEIOU_REQUIRE_COLD=true or false"), "{e}");
        // the negation of a fixed boolean: the command line's alone, never `AEIOU_X=false`
        let mut l = Layers::with_env("dry-run", None, env(&[("AEIOU_NO_METRICS", "1")])).unwrap();
        l.fixed("metrics", &false, false).unwrap();
        let e = l.finish().unwrap_err().to_string();
        assert!(e.contains("AEIOU_NO_METRICS: --metrics is the command line's alone (`--no-metrics` there)"), "{e}");
        assert!(!e.contains("AEIOU_METRICS="), "{e}");
        let cfg = write(&d, "[dry-run]\nno-metrics = true\n");
        let mut l = Layers::with_env("dry-run", Some(&cfg), env(&[])).unwrap();
        l.fixed("metrics", &false, false).unwrap();
        let e = l.finish().unwrap_err().to_string();
        assert!(e.contains("[dry-run] no-metrics: --metrics is the command line's alone (`--no-metrics` there)"), "{e}");
        assert!(!e.contains("metrics = false"), "{e}");
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_BOGUS", "1"), ("AEIOU_ROOT", "/x"), ("AEIOU_SCHEMA_DIR", "/s")])).unwrap();
        l.layered::<usize>("threads", None, None).unwrap();
        l.finish().unwrap();
        assert_eq!(l.warnings, vec!["AEIOU_BOGUS is set and is no option of any subcommand; ignored".to_string()]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn typed_values_and_the_config_from_the_environment() {
        let d = tmp();
        let cfg = write(&d, "[run]\nthreads = \"three\"\n");
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_CONFIG", cfg.display().to_string().as_str())])).unwrap();
        let e = l.layered::<usize>("threads", None, None).unwrap_err();
        assert!(format!("{e:#}").contains("[run] threads: expected an integer, got the string \"three\""), "{e:#}");
        l.finish().unwrap();
        assert_eq!(l.entries.last().unwrap().source, Source::Env(ENV_CONFIG.into()));
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_THREADS", "x")])).unwrap();
        assert!(l.layered::<usize>("threads", None, None).is_err());
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_REQUIRE_COLD", "yes"), ("AEIOU_MAX_GAP", "30")])).unwrap();
        assert!(l.flag("require-cold", None, false).unwrap());
        assert_eq!(l.layered::<f64>("max-gap", None, None).unwrap(), Some(30.0));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// RC1 review: an empty value or one that is not UTF-8, from the environment or the file,
    /// is a usage error naming the variable or the key (an empty root was the working
    /// directory); a path that is not UTF-8 is taken as it is; a config type error names the
    /// TOML type and value, never Rust's Debug.
    #[test]
    fn empty_and_non_utf8_values_and_the_toml_types() {
        use std::os::unix::ffi::OsStrExt;
        let d = tmp();
        for (var, name) in [("AEIOU_ROOT", "root"), ("AEIOU_REPORT_JSON", "report-json")] {
            let mut l = Layers::with_env("run", None, env(&[(var, "")])).unwrap();
            let e = l.layered::<PathBuf>(name, None, None).unwrap_err();
            assert!(e.downcast_ref::<crate::usage::UsageError>().is_some());
            assert!(format!("{e:#}").starts_with(&format!("{var}: empty")), "{e:#}");
        }
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_COORDINATOR", " ")])).unwrap();
        assert!(format!("{:#}", l.layered::<String>("coordinator", None, None).unwrap_err()).starts_with("AEIOU_COORDINATOR: empty"));
        let mut l = Layers::with_env("run", None, env(&[("AEIOU_ENDPOINT", "")])).unwrap();
        assert!(format!("{:#}", l.layered::<Vec<String>>("endpoint", None, None).unwrap_err()).starts_with("AEIOU_ENDPOINT: empty"));
        let e = Layers::with_env("run", None, env(&[("AEIOU_CONFIG", "")])).unwrap_err();
        assert!(format!("{e:#}").starts_with("AEIOU_CONFIG: empty"), "{e:#}");
        let cfg = write(&d, "[run]\nroot = \"\"\ncoordinator = \"\"\nendpoint = [\"\"]\nreport-json = 3\nrequire-cold = 1\nmax-gap = \"5\"\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[])).unwrap();
        for (name, want) in [
            ("root", "[run] root: empty"),
            ("coordinator", "[run] coordinator: empty"),
            ("endpoint", "[run] endpoint: empty"),
            ("report-json", "[run] report-json: expected a path string, got the integer 3"),
            ("require-cold", "[run] require-cold: expected true or false, got the integer 1"),
            ("max-gap", "[run] max-gap: expected a number, got the string \"5\""),
        ] {
            let e = match name {
                "root" | "report-json" => l.layered::<PathBuf>(name, None, None).map(|_| ()),
                "coordinator" => l.layered::<String>(name, None, None).map(|_| ()),
                "endpoint" => l.layered::<Vec<String>>(name, None, None).map(|_| ()),
                "require-cold" => l.flag(name, None, false).map(|_| ()),
                _ => l.layered::<f64>(name, None, None).map(|_| ()),
            }
            .unwrap_err();
            assert!(format!("{e:#}").contains(want), "{name}: {e:#}");
            assert!(!format!("{e:#}").contains("Integer(") && !format!("{e:#}").contains("String("), "{e:#}");
        }
        // the empty list is a list
        let cfg = write(&d, "[run]\nendpoint = []\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[])).unwrap();
        assert_eq!(l.layered::<Vec<String>>("endpoint", None, None).unwrap(), Some(vec![]));
        // not UTF-8: a path as it is, text refused naming the variable
        let raw = |b: &[u8]| OsStr::from_bytes(b).to_os_string();
        let mut e = env(&[]);
        e.insert("AEIOU_ROOT".into(), raw(b"/tmp/r\xff"));
        e.insert("AEIOU_COORDINATOR".into(), raw(b"h\xff:1"));
        let mut l = Layers::with_env("run", None, e).unwrap();
        assert_eq!(l.layered::<PathBuf>("root", None, None).unwrap().unwrap().as_os_str().as_bytes(), b"/tmp/r\xff");
        let e = l.layered::<String>("coordinator", None, None).unwrap_err();
        assert!(e.downcast_ref::<crate::usage::UsageError>().is_some());
        assert_eq!(format!("{e:#}"), "AEIOU_COORDINATOR: \"h\\xFF:1\" is not UTF-8");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_knob_from_a_lower_layer_is_passed_over_and_a_default_shows_its_value() {
        let d = tmp();
        let cfg = write(&d, "[run]\nthreads = 8\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[("AEIOU_AIO_DEPTH", "64")])).unwrap();
        l.layered::<usize>("threads", None, None).unwrap();
        l.layered::<u32>("aio-depth", None, None).unwrap();
        l.layered::<u32>("sqpoll", Some(5), None).unwrap();
        l.layered::<u64>("object-part-mib", None, None).unwrap();
        assert!(l.pass_over("threads", "this run has one thread per actor"));
        assert!(l.pass_over("aio-depth", "--posix sync has no AIO context"));
        assert!(!l.pass_over("sqpoll", "--posix sync has no ring"), "the command line's is the caller's to refuse");
        assert!(!l.pass_over("object-part-mib", "x"), "a default is not passed over");
        assert_eq!(
            l.warnings,
            vec![format!("--threads from config {} is not used: this run has one thread per actor", cfg.display()), "--aio-depth from env AEIOU_AIO_DEPTH is not used: --posix sync has no AIO context".to_string()]
        );
        assert!(!l.entries.iter().any(|e| e.name == "threads" || e.name == "aio-depth"));
        l.show_default("object-part-mib", "8".into());
        l.show_default("sqpoll", "0".into());
        let shown = |l: &Layers, n: &str| l.entries.iter().find(|e| e.name == n).unwrap().shown.clone();
        assert_eq!(shown(&l, "object-part-mib"), "8");
        assert_eq!(shown(&l, "sqpoll"), "5", "only a default's value is replaced");
        assert_eq!(l.json()["options"]["object-part-mib"]["value"], Value::Null);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn from_names_the_layers_that_were_not_the_command_line() {
        let d = tmp();
        let cfg = write(&d, "[run]\naio-depth = 64\n");
        let mut l = Layers::with_env("run", Some(&cfg), env(&[("AEIOU_THREADS", "5")])).unwrap();
        l.layered::<u32>("aio-depth", None, None).unwrap();
        l.layered::<usize>("threads", None, None).unwrap();
        l.layered::<u32>("sqpoll", Some(10), None).unwrap();
        l.layered::<i64>("rank", None, Some(0)).unwrap();
        assert_eq!(l.from(&["sqpoll"]), "");
        assert_eq!(l.from(&["aio-depth", "sqpoll"]), format!(" (--aio-depth from config {})", cfg.display()));
        assert_eq!(l.from(&["threads", "aio-depth"]), format!(" (--threads from env AEIOU_THREADS, --aio-depth from config {})", cfg.display()));
        assert_eq!(l.from(&["rank"]), " (--rank by default)");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn differences_between_hosts() {
        let a = json!({"options": {"threads": {"value": 4, "source": "cli"}, "root": {"value": "/mnt/a", "source": "cli"}, "rank": {"value": 0, "source": "cli"}}});
        let b = json!({"options": {"threads": {"value": 4, "source": "env AEIOU_THREADS"}, "root": {"value": "/mnt/b", "source": "cli"}, "rank": {"value": 1, "source": "cli"}}});
        let rows = differences(&[(0, &a), (1, &b)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], ("rank".to_string(), vec![(0, "0".to_string()), (1, "1".to_string())]));
        assert_eq!(rows[1].0, "root");
    }
}
