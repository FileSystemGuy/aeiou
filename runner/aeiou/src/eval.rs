//! The resolved model of one run: parameters with overrides applied, dataset geometry
//! (counts, sizes, layouts, names), namespaces, and distribution sampling. Everything here is a
//! pure function of the AST, the CLI parameters, and the seeds; the VM (`vm.rs`) evaluates
//! expressions against it.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};

use crate::ast::*;
use crate::pattern::{FieldValue, Pattern};
use crate::rng::{dataset_key, labeled_key, Perm, Words};

/// What `aeiou run` and `aeiou dry-run` take from the command line.
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub seed: u64,
    pub gpus: i64,
    /// `--param name=value`, value as JSON (a bare word that is not JSON is a string).
    pub overrides: Vec<(String, String)>,
    /// `--params-file FILE`, in command-line order: applied over the defaults, under `overrides`.
    pub sets: Vec<ParamSet>,
}

impl Config {
    /// Every parameter file names the abstract it is for; one that also names an AST hash is
    /// for that exact shape.
    pub fn check_sets(&self, abstract_name: &str, ast_sha256: &str) -> Result<()> {
        for s in &self.sets {
            s.check_identity(abstract_name, ast_sha256)?;
        }
        Ok(())
    }
}

/// The `params_version` this runner reads (`schema/README.md` §8).
pub const PARAMS_VERSION: u64 = 1;

/// A parameter file: a named set of values for the slots of one abstract (`schema/README.md`
/// §8, the "three sources" split of `GRAMMAR_OPTIONS.md` Option D). Loaded as JSON; the
/// values are checked against the abstract's declarations when `Params` is built.
#[derive(Debug, Clone)]
pub struct ParamSet {
    pub path: String,
    /// SHA-256 of the file's bytes, printed with the parameters in effect.
    pub sha256: String,
    pub abstract_name: String,
    pub ast_sha256: Option<String>,
    pub doc: Option<String>,
    pub values: BTreeMap<String, serde_json::Value>,
}

impl ParamSet {
    pub fn load(path: &std::path::Path) -> Result<ParamSet> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text, &path.display().to_string()).with_context(|| format!("parameter file {}", path.display()))
    }

    pub fn parse(text: &str, path: &str) -> Result<ParamSet> {
        use sha2::Digest;
        let sha256 = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));
        let doc: serde_json::Value = serde_json::from_str(text).context("not JSON")?;
        // a value past the contract's 64-bit integers would run as a float (`canon::int_range_errors`)
        let ints = crate::canon::int_range_errors(text);
        if !ints.is_empty() {
            bail!("{}", ints.join("\n  "));
        }
        let obj = doc.as_object().ok_or_else(|| anyhow!("not a JSON object"))?;
        for k in obj.keys() {
            if !matches!(k.as_str(), "params_version" | "abstract" | "ast_sha256" | "doc" | "params" | "provenance") {
                bail!("unknown key `{k}` (expected params_version, abstract, ast_sha256, doc, params, provenance)");
            }
        }
        let version = obj.get("params_version").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("`params_version` missing or not an integer"))?;
        if version != PARAMS_VERSION {
            bail!("params_version {version} (this runner reads {PARAMS_VERSION})");
        }
        let abstract_name = obj.get("abstract").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("`abstract` missing or not a string"))?.to_string();
        let ast_sha256 = match obj.get("ast_sha256") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) => Some(s.to_ascii_lowercase()),
            Some(v) => bail!("`ast_sha256` is not 64 hex digits: {v}"),
        };
        let doc_text = match obj.get("doc") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(v) => bail!("`doc` is not a string: {v}"),
        };
        let values = obj
            .get("params")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("`params` missing or not an object"))?
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(ParamSet { path: path.to_string(), sha256, abstract_name, ast_sha256, doc: doc_text, values })
    }

    pub fn check_identity(&self, abstract_name: &str, ast_sha256: &str) -> Result<()> {
        if self.abstract_name != abstract_name {
            bail!("parameter file {} is for abstract `{}`, not `{abstract_name}`", self.path, self.abstract_name);
        }
        if let Some(want) = &self.ast_sha256 {
            if want != ast_sha256 {
                bail!("parameter file {} is for AST {}…, not {}…", self.path, &want[..16], &ast_sha256[..16]);
            }
        }
        Ok(())
    }
}

/// Parameters in effect: the defaults, then every `--params-file` file in order, then `--param`.
/// Owned, so `Value` can borrow it.
pub struct Params {
    pub values: BTreeMap<String, PValue>,
}

fn pvalue_kind(v: &PValue) -> &'static str {
    match v {
        PValue::Dist(_) => "a distribution",
        PValue::Array(_) => "an array",
        PValue::Scalar(_) => "a scalar",
    }
}

/// A value may replace a default of the same kind: a scalar for a scalar, an array for an
/// array (any length, elements of the default's element kind), a distribution for a
/// distribution. The builder decided at build time how each slot is consumed (a distribution
/// is drawn, an array is indexed), so a change of kind would not be a change of value.
fn check_kind(name: &str, default: &PValue, v: &PValue, source: &str) -> Result<()> {
    if std::mem::discriminant(default) != std::mem::discriminant(v) {
        bail!("{source} {name}: the default is {}, the value is {}", pvalue_kind(default), pvalue_kind(v));
    }
    if let (PValue::Array(d), PValue::Array(a)) = (default, v) {
        if let Some(first) = d.first() {
            for (i, x) in a.iter().enumerate() {
                if std::mem::discriminant(first) != std::mem::discriminant(x) {
                    bail!("{source} {name}[{i}]: the default's elements are {}, this one is {}", pvalue_kind(first), pvalue_kind(x));
                }
            }
        }
    }
    Ok(())
}

impl Params {
    pub fn new(ast: &Ast, cfg: &Config) -> Result<Params> {
        let mut values: BTreeMap<String, PValue> = ast.params.iter().map(|(k, p)| (k.clone(), p.default.clone())).collect();
        for set in &cfg.sets {
            if set.abstract_name != ast.name {
                bail!("parameter file {} is for abstract `{}`, not `{}`", set.path, set.abstract_name, ast.name);
            }
            for (name, v) in &set.values {
                let source = format!("{}:", set.path);
                if name == "gpus" {
                    bail!("{source} `gpus` is set with --gpus, not by a parameter file");
                }
                let param = ast.params.get(name).ok_or_else(|| anyhow!("{source} no parameter `{name}` in `{}`", ast.name))?;
                let pv: PValue = serde_json::from_value(v.clone()).map_err(|e| anyhow!("{source} {name}: not a parameter value (a scalar, a distribution, or an array of these): {e}"))?;
                check_kind(name, &param.default, &pv, &source)?;
                values.insert(name.clone(), pv);
            }
        }
        for (name, text) in &cfg.overrides {
            if name == "gpus" {
                bail!("`gpus` is set with --gpus, not --param");
            }
            let param = ast.params.get(name).ok_or_else(|| anyhow!("--param {name}: no such parameter"))?;
            if param.cli == Some(false) {
                bail!("--param {name}: not overridable (cli: false)");
            }
            let v: PValue = match serde_json::from_str(text) {
                Ok(v) => {
                    if let Some(e) = crate::canon::int_range_errors(text).first() {
                        bail!("--param {name}: {}", e.trim_start_matches(": "));
                    }
                    v
                }
                Err(_) => PValue::Scalar(Literal::Str(text.clone())),
            };
            check_kind(name, &param.default, &v, "--param")?;
            values.insert(name.clone(), v);
        }
        // rule V3 again on what was given: a distribution whose minimum is below the default's
        // can turn an `at` offset into zero (§3.57)
        let errs = crate::validate::check_given(ast, &values);
        if !errs.is_empty() {
            bail!("invalid with the parameters given:\n  {}", errs.join("\n  "));
        }
        Ok(Params { values })
    }
}

// ---------------------------------------------------------------- values

#[derive(Debug, Clone)]
pub enum Value<'a> {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    Dist(&'a Dist),
    Array(&'a [PValue]),
    Handle(HVal),
}

/// A resolved handle: what a `consume`, `pick`, `file`, `dir`, or `object` names.
#[derive(Debug, Clone, PartialEq)]
pub enum HVal {
    /// Sample `id` of files dataset `ds`.
    Sample { ds: usize, id: i64 },
    /// File `file` of a files dataset (the container of a sample, or a file by id).
    File { ds: usize, file: i64 },
    /// Chunk object `k` of file `file` of a `chunk`-realized dataset.
    Chunk { ds: usize, file: i64, k: i64 },
    /// Slot `c` of an array dataset.
    Slot { ds: usize, c: i64 },
    /// The single file of an array dataset.
    ArrayFile { ds: usize },
    /// Unit `u` (row group, record batch, chunk) of file `file` of a files dataset.
    Unit { ds: usize, file: i64, u: i64 },
    /// Column chunk `c` of unit `u` of file `file`.
    Column { ds: usize, file: i64, u: i64, c: i64 },
    /// Directory `d` of a files dataset's pattern.
    Dir { ds: usize, d: i64 },
    /// The inventory of dataset `ds` (contract 0.9).
    Inventory { ds: usize },
    /// A namespace object with its formatted path and computed size (`None` = as_written).
    Object { ns: usize, path: Arc<str>, size: Option<i64> },
}

impl<'a> Value<'a> {
    pub fn kind(&self) -> &'static str {
        match self {
            Value::None => "none",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Dist(_) => "dist",
            Value::Array(_) => "array",
            Value::Handle(_) => "handle",
        }
    }

    /// Integer view; floats round to nearest (rule V10).
    pub fn as_int(&self) -> Result<i64> {
        match self {
            Value::Int(n) => Ok(*n),
            Value::Bool(b) => Ok(*b as i64),
            Value::Float(x) => {
                if !x.is_finite() {
                    bail!("non-finite float where an integer is needed");
                }
                // `as` saturates: a float past the 64-bit integers would become 2^63-1 silently
                let r = x.round();
                if !(r >= -9223372036854775808.0 && r < 9223372036854775808.0) {
                    bail!("float {x} where an integer is needed is outside the 64-bit integers of the contract");
                }
                Ok(r as i64)
            }
            other => bail!("expected an integer, got {}", other.kind()),
        }
    }

    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Value::Int(n) => Ok(*n as f64),
            Value::Float(x) => Ok(*x),
            Value::Bool(b) => Ok(*b as i64 as f64),
            other => bail!("expected a number, got {}", other.kind()),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(b) => Ok(*b),
            Value::Int(n) => Ok(*n != 0),
            other => bail!("expected a condition, got {}", other.kind()),
        }
    }

    pub fn as_handle(&self) -> Result<&HVal> {
        match self {
            Value::Handle(h) => Ok(h),
            other => bail!("expected a handle, got {}", other.kind()),
        }
    }

    pub fn from_literal(l: &Literal) -> Value<'a> {
        match l {
            Literal::Null => Value::None,
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Int(n) => Value::Int(*n),
            Literal::Float(x) => Value::Float(*x),
            Literal::Str(s) => Value::Str(Arc::from(s.as_str())),
        }
    }

    pub fn from_pvalue(p: &'a PValue) -> Value<'a> {
        match p {
            PValue::Scalar(l) => Value::from_literal(l),
            PValue::Dist(d) => Value::Dist(d),
            PValue::Array(a) => Value::Array(a),
        }
    }
}

impl std::fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::None => write!(f, "none"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Dist(_) => write!(f, "<dist>"),
            Value::Array(a) => write!(f, "<array[{}]>", a.len()),
            Value::Handle(h) => write!(f, "{h:?}"),
        }
    }
}

// ---------------------------------------------------------------- resolved distributions

/// A distribution with its arguments evaluated, ready to sample.
#[derive(Debug, Clone)]
pub enum RDist<'a> {
    Const(Value<'a>),
    Uniform { lo: i64, hi: i64 },
    Uniform64,
    Normal { mean: f64, sd: f64, min: Option<f64>, max: Option<f64> },
    Lognormal { median: f64, sigma: f64, min: Option<f64>, max: Option<f64> },
    Empirical { values: Vec<Value<'a>>, cum: Vec<f64> },
    Zipf { s: f64 },
    Hotset { fraction: f64, weight: f64 },
    Mixture { arms: Vec<(f64, Option<RDist<'a>>)>, total: f64 },
}

/// Draw one value. `domain` is the id permutation of the dataset a `pick` samples over; `zipf`
/// and `hotset` need it (they draw a rank and map it through the dataset's rank order).
pub fn sample<'a>(d: &RDist<'a>, w: &mut Words, domain: Option<&Perm>) -> Result<Value<'a>> {
    Ok(match d {
        RDist::Const(v) => v.clone(),
        RDist::Uniform { lo, hi } => {
            if hi <= lo {
                bail!("uniform: empty range [{lo}, {hi})");
            }
            Value::Int(w.next_range(*lo, *hi))
        }
        RDist::Uniform64 => Value::Int(w.next_u64() as i64),
        RDist::Normal { mean, sd, min, max } => {
            let x = mean + sd * w.next_normal();
            Value::Int(clamp(x, *min, *max).round() as i64)
        }
        RDist::Lognormal { median, sigma, min, max } => {
            let x = median * (sigma * w.next_normal()).exp();
            Value::Int(clamp(x, *min, *max).round() as i64)
        }
        RDist::Empirical { values, cum } => {
            let total = *cum.last().unwrap_or(&0.0);
            if total <= 0.0 {
                bail!("empirical: total weight is zero");
            }
            let u = w.next_f64() * total;
            let i = cum.partition_point(|c| *c <= u).min(values.len() - 1);
            values[i].clone()
        }
        RDist::Zipf { s } => {
            let perm = domain.ok_or_else(|| anyhow!("zipf outside a pick over a dataset"))?;
            let n = perm.len() as f64;
            let u = w.next_f64();
            // Inverse CDF of the continuous envelope of P(r) ∝ r^-s on [1, N+1): O(1), no table.
            let x = if (*s - 1.0).abs() < 1e-12 {
                (u * (n + 1.0).ln()).exp()
            } else {
                let t = 1.0 - s;
                (u * ((n + 1.0).powf(t) - 1.0) + 1.0).powf(1.0 / t)
            };
            let rank = (x.floor() as i64).clamp(1, perm.len() as i64) - 1;
            Value::Int(perm.apply(rank as u64) as i64)
        }
        RDist::Hotset { fraction, weight } => {
            let perm = domain.ok_or_else(|| anyhow!("hotset outside a pick over a dataset"))?;
            let n = perm.len() as i64;
            let hot = ((fraction * n as f64).ceil() as i64).clamp(1, n);
            let u = w.next_f64();
            let rank = if u < *weight || hot == n { w.next_range(0, hot) } else { w.next_range(hot, n) };
            Value::Int(perm.apply(rank as u64) as i64)
        }
        RDist::Mixture { arms, total } => {
            if *total <= 0.0 {
                bail!("mixture: total weight is zero");
            }
            let u = w.next_f64() * total;
            let mut acc = 0.0;
            let mut chosen = arms.len() - 1;
            for (i, (wt, _)) in arms.iter().enumerate() {
                acc += wt;
                if u < acc {
                    chosen = i;
                    break;
                }
            }
            match &arms[chosen].1 {
                None => Value::None,
                Some(d) => sample(d, w, domain)?,
            }
        }
    })
}

fn clamp(x: f64, min: Option<f64>, max: Option<f64>) -> f64 {
    let x = match min {
        Some(m) if x < m => m,
        _ => x,
    };
    match max {
        Some(m) if x > m => m,
        _ => x,
    }
}

// ---------------------------------------------------------------- datasets and namespaces

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirScheme {
    /// No directory part: `dirs` is 0.
    NoDir,
    /// A constant directory: one directory, all files in it.
    Constant,
    /// `{id div D}`: directory d holds ids `[d·D, (d+1)·D)`.
    Div(i64),
    /// `{id mod M}`: directory d holds ids `≡ d (mod M)`.
    Mod(i64),
    /// `{id}`: one directory per id.
    Plain,
    /// Anything else (several id fields in the directory part).
    Unsupported,
}

/// A container's framing, resolved from `format.framing` (`schema/README.md` §2): how a file's
/// bytes are framed around its samples. A file is `file_header ‖ units ‖ file_footer`,
/// aligned to `file_align`; a unit is `unit_header ‖ column chunks ‖ unit_footer`, aligned;
/// a column chunk is `header ‖ rows`, aligned; a row is `row_header ‖ fixed ‖ its share of
/// the sample ‖ row_footer`, aligned. Headers and footers may grow per sample or per unit.
/// All zeros with one column of weight 1 is the packed framing: samples back to back, which
/// is every dataset without a format class.
#[derive(Debug, Clone)]
pub struct Framing {
    /// Samples per unit.
    pub unit: i64,
    pub file_header: i64,
    pub file_header_per_sample: i64,
    pub file_header_per_unit: i64,
    pub file_footer: i64,
    pub file_footer_per_sample: i64,
    pub file_footer_per_unit: i64,
    pub file_align: i64,
    pub unit_header: i64,
    pub unit_header_per_sample: i64,
    pub unit_footer: i64,
    pub unit_footer_per_sample: i64,
    pub unit_align: i64,
    pub columns: Vec<Column>,
}

#[derive(Debug, Clone)]
pub struct Column {
    pub header: i64,
    pub fixed: i64,
    pub weight: f64,
    pub row_header: i64,
    pub row_footer: i64,
    pub row_align: i64,
    pub align: i64,
    /// The last column with a positive weight: it takes the remainder of the sample's bytes.
    pub last_weighted: bool,
}

/// `x` rounded up to a multiple of `a`, `None` past 2^63-1.
pub fn align_up(x: i64, a: i64) -> Option<i64> {
    if a <= 1 {
        Some(x)
    } else {
        ceil_div(x, a).checked_mul(a)
    }
}

/// `ceil(x / d)` for `x >= 0` and `d >= 1`, without the `x + d - 1` that wraps near 2^63.
pub fn ceil_div(x: i64, d: i64) -> i64 {
    x / d + (x % d > 0) as i64
}

/// `Σ xs`, `None` past 2^63-1: the geometry's sums are checked, never wrapped (RC1 review).
fn checked_sum(xs: impl IntoIterator<Item = i64>) -> Option<i64> {
    xs.into_iter().try_fold(0i64, |s, x| s.checked_add(x))
}

impl Framing {
    /// Samples back to back: no headers, one column carrying each sample whole.
    pub fn packed(unit: i64) -> Framing {
        Framing {
            unit: unit.max(1),
            file_header: 0,
            file_header_per_sample: 0,
            file_header_per_unit: 0,
            file_footer: 0,
            file_footer_per_sample: 0,
            file_footer_per_unit: 0,
            file_align: 1,
            unit_header: 0,
            unit_header_per_sample: 0,
            unit_footer: 0,
            unit_footer_per_sample: 0,
            unit_align: 1,
            columns: vec![Column { header: 0, fixed: 0, weight: 1.0, row_header: 0, row_footer: 0, row_align: 1, align: 1, last_weighted: true }],
        }
    }

    pub fn is_packed(&self) -> bool {
        let c = &self.columns;
        self.file_header == 0 && self.file_header_per_sample == 0 && self.file_header_per_unit == 0 && self.file_footer == 0
            && self.file_footer_per_sample == 0 && self.file_footer_per_unit == 0 && self.file_align <= 1 && self.unit_header == 0
            && self.unit_header_per_sample == 0 && self.unit_footer == 0 && self.unit_footer_per_sample == 0 && self.unit_align <= 1
            && c.len() == 1 && c[0].header == 0 && c[0].fixed == 0 && c[0].row_header == 0 && c[0].row_footer == 0 && c[0].row_align <= 1 && c[0].align <= 1
    }

    /// The share of a `size`-byte sample that column `c` carries: `floor(size × weight)`,
    /// the last weighted column taking the remainder so the shares sum to `size`.
    pub fn part(&self, c: usize, size: i64) -> i64 {
        let col = &self.columns[c];
        if col.weight <= 0.0 {
            return 0;
        }
        if col.last_weighted {
            let others: i64 = self.columns.iter().enumerate().filter(|(i, k)| *i != c && k.weight > 0.0).map(|(_, k)| (size as f64 * k.weight).floor() as i64).sum();
            return size - others;
        }
        (size as f64 * col.weight).floor() as i64
    }

    /// Bytes of column `c`'s row for a `size`-byte sample, `None` past 2^63-1.
    pub fn row_bytes(&self, c: usize, size: i64) -> Option<i64> {
        let col = &self.columns[c];
        align_up(checked_sum([col.row_header, col.fixed, self.part(c, size), col.row_footer])?, col.row_align)
    }
}

/// A dataset's inventory (contract 0.9, `DESIGN_REVIEW.md` §3.73): `path`, `entry` bytes per
/// item of the count. `cap`: the run is a subset of the corpus whose inventory is on disk, so
/// reads of it are issued no further than the size at the run's count (§3.73 item 5).
#[derive(Clone, Debug)]
pub struct InvMeta {
    pub path: Arc<str>,
    pub entry: i64,
    pub cap: bool,
}

/// One container file's size and the offset of each of its samples (`DsMeta::file_geom`).
#[derive(Debug, PartialEq)]
pub struct FileGeom {
    pub size: i64,
    /// By sample index within the file; `None` where `sample_offset` refuses.
    pub offsets: Option<Vec<i64>>,
}

#[derive(Clone)]
pub enum DsMeta<'a> {
    Files {
        name: &'a str,
        pattern: Pattern,
        count: i64,
        spf: i64,
        chunk: Option<i64>,
        access: Access,
        framing: Framing,
        size: RDist<'a>,
        seed: u64,
        dirs: DirScheme,
        /// Number of directories, per `dirs`.
        ndirs: i64,
        /// `labeled_key(seed, "consume", name)`, the base of the per-epoch permutation keys.
        consume_base: u64,
        rank_key: u64,
        inventory: Option<InvMeta>,
    },
    /// An `array` dataset: one file of `count` fixed slots (contract 0.8; `regions` before).
    Array {
        name: &'a str,
        file: Arc<str>,
        count: i64,
        slot: i64,
        size: RDist<'a>,
        seed: u64,
        rank_key: u64,
    },
}

impl<'a> DsMeta<'a> {
    pub fn name(&self) -> &'a str {
        match self {
            DsMeta::Files { name, .. } | DsMeta::Array { name, .. } => name,
        }
    }

    /// The error of geometry arithmetic past 2^63-1: an offset, a size, or a count no file
    /// system holds, refused rather than wrapped into a wrong op stream (RC1 review).
    fn overflow(&self, what: String) -> anyhow::Error {
        anyhow!("dataset `{}`: {what} overflows the 64-bit integers of the contract (2^63-1)", self.name())
    }

    pub fn count(&self) -> i64 {
        match self {
            DsMeta::Files { count, .. } | DsMeta::Array { count, .. } => *count,
        }
    }

    pub fn dataset_seed(&self) -> u64 {
        match self {
            DsMeta::Files { seed, .. } | DsMeta::Array { seed, .. } => *seed,
        }
    }

    /// Files of a files dataset (its shards), one for an array dataset.
    /// The same dataset with `count` samples (files when there is one per file): what a corpus
    /// was before `datagen --grow` or `--shrink` (`DESIGN_REVIEW.md` §3.70), every name, size,
    /// and byte of an id the two share being the same.
    pub fn with_count(&self, count: i64) -> DsMeta<'a> {
        let mut m = self.clone();
        match &mut m {
            DsMeta::Files { count: c, spf, pattern, dirs, ndirs, .. } => {
                *c = count;
                let files = ceil_div(count, *spf);
                (*dirs, *ndirs) = Model::dir_scheme(pattern, files);
            }
            DsMeta::Array { count: c, .. } => *c = count,
        }
        m
    }

    /// Reads of the inventory are issued no further than its size at this count: the run is a
    /// subset of a corpus whose inventory lists more (`DESIGN_REVIEW.md` §3.73 item 5).
    pub fn cap_inventory(&mut self) {
        if let DsMeta::Files { inventory: Some(i), .. } = self {
            i.cap = true;
        }
    }

    /// The inventory's path, its size at this count, and whether reads of it are capped.
    pub fn inventory(&self) -> Option<(Arc<str>, i64, bool)> {
        match self {
            // `build_model` refuses a count whose inventory overflows; smaller counts fit
            DsMeta::Files { inventory: Some(i), count, .. } => Some((i.path.clone(), count.saturating_mul(i.entry), i.cap)),
            _ => None,
        }
    }

    pub fn files(&self) -> i64 {
        match self {
            DsMeta::Files { count, spf, .. } => ceil_div(*count, *spf),
            DsMeta::Array { .. } => 1,
        }
    }

    pub fn access(&self) -> Access {
        match self {
            DsMeta::Files { access, .. } => *access,
            DsMeta::Array { .. } => Access::Indexed,
        }
    }

    /// What `consume` and `pick` draw from: samples under `map`, files (shards) under `stream`.
    pub fn draw_domain(&self) -> i64 {
        if self.access() == Access::Stream {
            self.files()
        } else {
            self.count()
        }
    }

    /// The rank order of ids for `zipf` and `hotset`, fixed by the dataset seed.
    pub fn rank_perm(&self) -> Perm {
        let key = match self {
            DsMeta::Files { rank_key, .. } | DsMeta::Array { rank_key, .. } => *rank_key,
        };
        Perm::new(self.draw_domain().max(1) as u64, key)
    }

    pub fn framing(&self) -> Result<&Framing> {
        match self {
            DsMeta::Files { framing, .. } => Ok(framing),
            DsMeta::Array { .. } => bail!("`{}` is an array dataset: no container framing", self.name()),
        }
    }

    /// Units (row groups, record batches, chunks) in file `file`.
    pub fn units_in_file(&self, file: i64) -> Result<i64> {
        let n = self.samples_in_file(file)?;
        let l = self.framing()?;
        Ok(ceil_div(n, l.unit))
    }

    /// The unit of sample `id`, within its file.
    pub fn unit_of_sample(&self, id: i64) -> Result<i64> {
        match self {
            DsMeta::Files { spf, framing: layout, .. } => {
                let file = id / spf;
                Ok((id - file * spf) / layout.unit)
            }
            DsMeta::Array { .. } => Ok(0),
        }
    }

    pub fn samples_in_unit(&self, file: i64, u: i64) -> Result<i64> {
        let n = self.samples_in_file(file)?;
        let l = self.framing()?;
        if u < 0 || u >= ceil_div(n, l.unit) {
            bail!("unit {u} outside file {file} of `{}` ({} units)", self.name(), ceil_div(n, l.unit));
        }
        Ok((n - u * l.unit).min(l.unit))
    }

    /// Lengths of the column chunks of unit `u` of `file`, headers and alignment included.
    pub fn col_lens(&self, file: i64, u: i64) -> Result<Vec<i64>> {
        let l = self.framing()?;
        let spf = match self {
            DsMeta::Files { spf, .. } => *spf,
            _ => 1,
        };
        let n = self.samples_in_unit(file, u)?;
        let first = file * spf + u * l.unit;
        let mut lens: Vec<i64> = l.columns.iter().map(|c| c.header).collect();
        for i in 0..n {
            let size = self.sample_size(first + i)?;
            for (c, len) in lens.iter_mut().enumerate() {
                *len = l.row_bytes(c, size).and_then(|r| len.checked_add(r)).ok_or_else(|| self.overflow(format!("column {c} of unit {u} of file {file}")))?;
            }
        }
        for (c, len) in lens.iter_mut().enumerate() {
            *len = align_up(*len, l.columns[c].align).ok_or_else(|| self.overflow(format!("column {c} of unit {u} of file {file}")))?;
        }
        Ok(lens)
    }

    pub fn col_len(&self, file: i64, u: i64, c: i64) -> Result<i64> {
        let lens = self.col_lens(file, u)?;
        if c < 0 || c as usize >= lens.len() {
            bail!("column {c} outside the {} columns of `{}`", lens.len(), self.name());
        }
        Ok(lens[c as usize])
    }

    /// Offset of column chunk `c` from the start of its unit.
    pub fn col_offset_in_unit(&self, file: i64, u: i64, c: i64) -> Result<i64> {
        let l = self.framing()?;
        let lens = self.col_lens(file, u)?;
        if c < 0 || c as usize >= lens.len() {
            bail!("column {c} outside the {} columns of `{}`", lens.len(), self.name());
        }
        let n = self.samples_in_unit(file, u)?;
        (|| checked_sum([l.unit_header, l.unit_header_per_sample.checked_mul(n)?, checked_sum(lens[..c as usize].iter().copied())?]))()
            .ok_or_else(|| self.overflow(format!("the offset of column {c} in unit {u} of file {file}")))
    }

    /// Bytes of unit `u` of `file`, framing and alignment included.
    pub fn unit_len(&self, file: i64, u: i64) -> Result<i64> {
        let l = self.framing()?;
        let n = self.samples_in_unit(file, u)?;
        let cols = checked_sum(self.col_lens(file, u)?);
        (|| align_up(checked_sum([l.unit_header, l.unit_header_per_sample.checked_mul(n)?, cols?, l.unit_footer, l.unit_footer_per_sample.checked_mul(n)?])?, l.unit_align))()
            .ok_or_else(|| self.overflow(format!("the length of unit {u} of file {file}")))
    }

    /// The lengths of every unit of `file`: O(samples in the file). The VM caches this per
    /// open file (`GRAMMAR_OPTIONS.md` §6.3).
    pub fn unit_lens(&self, file: i64) -> Result<Vec<i64>> {
        (0..self.units_in_file(file)?).map(|u| self.unit_len(file, u)).collect()
    }

    /// Where the first unit starts: after the file header.
    pub fn data_start(&self, file: i64) -> Result<i64> {
        let l = self.framing()?;
        let n = self.samples_in_file(file)?;
        let units = self.units_in_file(file)?;
        (|| checked_sum([l.file_header, l.file_header_per_sample.checked_mul(n)?, l.file_header_per_unit.checked_mul(units)?]))()
            .ok_or_else(|| self.overflow(format!("the header of file {file}")))
    }

    /// Offset of unit `u` in `file` (O(u × unit) sample draws; the VM uses `unit_lens`).
    pub fn unit_offset(&self, file: i64, u: i64) -> Result<i64> {
        if u < 0 || u >= self.units_in_file(file)? {
            bail!("unit {u} outside file {file} of `{}`", self.name());
        }
        let mut off = self.data_start(file)?;
        for v in 0..u {
            off = off.checked_add(self.unit_len(file, v)?).ok_or_else(|| self.overflow(format!("the offset of unit {u} of file {file}")))?;
        }
        Ok(off)
    }

    /// Bytes of a file: the framed units, the footer, and the file alignment.
    pub fn file_size_from_units(&self, file: i64, unit_lens: &[i64]) -> Result<i64> {
        let l = self.framing()?;
        let n = self.samples_in_file(file)?;
        let units = unit_lens.len() as i64;
        let start = self.data_start(file)?;
        (|| align_up(checked_sum([start, checked_sum(unit_lens.iter().copied())?, l.file_footer, l.file_footer_per_sample.checked_mul(n)?, l.file_footer_per_unit.checked_mul(units)?])?, l.file_align))()
            .ok_or_else(|| self.overflow(format!("the size of file {file}")))
    }

    /// Bytes of sample `id` (files) or slot `c` (array, capped at the slot).
    pub fn sample_size(&self, id: i64) -> Result<i64> {
        match self {
            DsMeta::Files { size, seed, count, .. } => {
                if id < 0 || id >= *count {
                    bail!("sample {id} outside dataset `{}` of {count}", self.name());
                }
                #[cfg(test)]
                tests::SIZE_DRAWS.with(|c| c.set(c.get() + 1));
                let mut w = Words::new(dataset_key(*seed, id));
                let n = sample(size, &mut w, None)?.as_int()?;
                // rule V25 refuses what the document shows; a size it cannot see is caught
                // here, before a cast to `u64` makes it some 16 EiB
                if n < 0 {
                    bail!("sample {id} of dataset `{}` has size {n}, below 0: a size is a byte count (rule V25)", self.name());
                }
                Ok(n)
            }
            DsMeta::Array { size, seed, count, slot, .. } => {
                if id < 0 || id >= *count {
                    bail!("region {id} outside dataset `{}` of {count}", self.name());
                }
                let mut w = Words::new(dataset_key(*seed, id));
                Ok(sample(size, &mut w, None)?.as_int()?.min(*slot).max(0))
            }
        }
    }

    /// Samples held by file `file` of a files dataset (the last file may be short).
    pub fn samples_in_file(&self, file: i64) -> Result<i64> {
        match self {
            DsMeta::Files { count, spf, .. } => {
                let first = file.checked_mul(*spf).ok_or_else(|| anyhow!("file id overflow"))?;
                if file < 0 || first >= *count {
                    bail!("file {file} outside dataset `{}`", self.name());
                }
                Ok((*count - first).min(*spf))
            }
            DsMeta::Array { .. } => bail!("`{}` is an array dataset", self.name()),
        }
    }

    /// Bytes of a container file: the sum of its samples' sizes under the packed framing, else
    /// the framing of `format.framing`.
    pub fn file_size(&self, file: i64) -> Result<i64> {
        match self {
            DsMeta::Files { spf, framing: layout, .. } => {
                let n = self.samples_in_file(file)?;
                if !layout.is_packed() {
                    let lens = self.unit_lens(file)?;
                    return self.file_size_from_units(file, &lens);
                }
                if *spf == 1 {
                    return self.sample_size(file);
                }
                let mut total = 0i64;
                for i in 0..n {
                    total = total.checked_add(self.sample_size(file * spf + i)?).ok_or_else(|| self.overflow(format!("the size of file {file}")))?;
                }
                Ok(total)
            }
            DsMeta::Array { count, slot, .. } => {
                if file != 0 {
                    bail!("an array dataset has one file, id 0");
                }
                if *count == 0 {
                    return Ok(0);
                }
                // `build_model` refuses a count × slot past 2^63-1, and a size is capped at the slot
                Ok((count - 1) * slot + self.sample_size(count - 1)?)
            }
        }
    }

    /// Offset of sample `id`'s bytes inside its container: the prefix sum over the file's
    /// earlier samples under the packed framing; under a format's framing, the start of its row's
    /// payload in the one column (a sample split over several columns has no single offset:
    /// address its unit or a column).
    pub fn sample_offset(&self, id: i64) -> Result<i64> {
        match self {
            DsMeta::Files { spf, framing: layout, .. } => {
                let file = id / spf;
                if layout.is_packed() {
                    let mut off = 0i64;
                    for i in 0..(id - file * spf) {
                        off = off.checked_add(self.sample_size(file * spf + i)?).ok_or_else(|| self.overflow(format!("the offset of sample {id}")))?;
                    }
                    return Ok(off);
                }
                if layout.columns.len() != 1 {
                    bail!("offset of a sample of `{}`: its bytes are split over {} columns; address the unit or a column", self.name(), layout.columns.len());
                }
                let u = self.unit_of_sample(id)?;
                let first = file * spf + u * layout.unit;
                let ovf = || self.overflow(format!("the offset of sample {id}"));
                let mut off = checked_sum([self.unit_offset(file, u)?, self.col_offset_in_unit(file, u, 0)?, layout.columns[0].header]).ok_or_else(ovf)?;
                for s in first..id {
                    off = layout.row_bytes(0, self.sample_size(s)?).and_then(|r| off.checked_add(r)).ok_or_else(ovf)?;
                }
                off.checked_add(layout.columns[0].row_header).ok_or_else(ovf)
            }
            DsMeta::Array { slot, .. } => id.checked_mul(*slot).ok_or_else(|| self.overflow(format!("the offset of slot {id}"))),
        }
    }

    /// `file_size(file)` and `sample_offset` of every sample of a files dataset's `file`, from
    /// one pass over its samples' sizes (two under a format's framing): what `file_size` and
    /// `sample_offset` each draw again per call, O(samples in the file), held by the VM for the
    /// files it is reading (`Vm::file_geom`). The values are those functions' to the bit;
    /// `offsets` is `None` where `sample_offset` would refuse (several columns, an offset past
    /// 2^63-1) or the file has more than `max_offsets` samples (a per-sample table is never
    /// held past that bound), and an `Err` is `file_size`'s refusal, so a caller falls back to
    /// them for the error message.
    pub fn file_geom(&self, file: i64, max_offsets: usize) -> Result<FileGeom> {
        let DsMeta::Files { spf, framing: layout, .. } = self else { bail!("`{}` is an array dataset", self.name()) };
        let n = self.samples_in_file(file)?;
        let first = file * spf;
        let keep = usize::try_from(n).is_ok_and(|n| n <= max_offsets);
        if layout.is_packed() {
            let mut offsets = Vec::with_capacity(if keep { n as usize } else { 0 });
            let mut total = 0i64;
            for i in 0..n {
                if keep {
                    offsets.push(total);
                }
                total = total.checked_add(self.sample_size(first + i)?).ok_or_else(|| self.overflow(format!("the size of file {file}")))?;
            }
            return Ok(FileGeom { size: total, offsets: keep.then_some(offsets) });
        }
        let lens = self.unit_lens(file)?;
        let size = self.file_size_from_units(file, &lens)?;
        if layout.columns.len() != 1 || !keep {
            return Ok(FileGeom { size, offsets: None });
        }
        let col = &layout.columns[0];
        let offsets = (|| {
            let mut offsets = Vec::with_capacity(n as usize);
            let mut unit_at = self.data_start(file).ok()?;
            for (u, len) in lens.iter().enumerate() {
                let m = (n - u as i64 * layout.unit).min(layout.unit);
                let mut off = checked_sum([unit_at, layout.unit_header, layout.unit_header_per_sample.checked_mul(m)?, col.header])?;
                for i in 0..m {
                    offsets.push(off.checked_add(col.row_header)?);
                    off = off.checked_add(layout.row_bytes(0, self.sample_size(first + u as i64 * layout.unit + i).ok()?)?)?;
                }
                unit_at = unit_at.checked_add(*len)?;
            }
            Some(offsets)
        })();
        Ok(FileGeom { size, offsets })
    }

    pub fn chunk_size(&self, file: i64, k: i64) -> Result<i64> {
        match self {
            DsMeta::Files { chunk: Some(c), .. } => {
                let size = self.file_size(file)?;
                if k < 0 || k >= ceil_div(size, *c) {
                    bail!("chunk {k} outside file {file} of `{}` ({size} bytes)", self.name());
                }
                Ok((size - k * c).min(*c))
            }
            _ => bail!("dataset `{}` has no `chunk`", self.name()),
        }
    }

    pub fn chunks(&self, file: i64) -> Result<i64> {
        match self {
            DsMeta::Files { chunk: Some(c), .. } => {
                let size = self.file_size(file)?;
                Ok(ceil_div(size, *c))
            }
            _ => bail!("dataset `{}` has no `chunk`", self.name()),
        }
    }

    pub fn file_path(&self, file: i64, k: Option<i64>) -> Result<Arc<str>> {
        match self {
            DsMeta::Files { pattern, .. } => {
                let s = pattern.format(|f| match f {
                    "id" => Some(FieldValue::Int(file)),
                    "k" => k.map(FieldValue::Int),
                    _ => None,
                })?;
                Ok(Arc::from(s))
            }
            DsMeta::Array { file: path, .. } => {
                if file != 0 {
                    bail!("an array dataset has one file, id 0");
                }
                Ok(path.clone())
            }
        }
    }

    pub fn dir_path(&self, d: i64) -> Result<Arc<str>> {
        match self {
            DsMeta::Files { pattern, dirs, ndirs, .. } => {
                if d < 0 || d >= *ndirs {
                    bail!("directory {d} outside the {ndirs} directories of `{}`", self.name());
                }
                let id = match dirs {
                    DirScheme::Div(n) => d * n,
                    DirScheme::Mod(_) | DirScheme::Plain => d,
                    DirScheme::Constant => 0,
                    DirScheme::NoDir => bail!("dataset `{}` has no directory part", self.name()),
                    DirScheme::Unsupported => bail!("dataset `{}`: `dir` needs at most one `id` field in the directory part", self.name()),
                };
                let s = pattern.format_dir(|f| match f {
                    "id" => Some(FieldValue::Int(id)),
                    _ => None,
                })?;
                Ok(Arc::from(s))
            }
            DsMeta::Array { .. } => bail!("`dir` on array dataset `{}`", self.name()),
        }
    }

    /// Entries a `readdir` of directory `d` should return, when the layout makes it computable
    /// (one sample per file, no chunks).
    pub fn dir_entries(&self, d: i64) -> Option<i64> {
        match self {
            DsMeta::Files { count, spf, chunk, dirs, .. } if *spf == 1 && chunk.is_none() => match dirs {
                // saturating: the clamp's result is the exact one's
                DirScheme::Div(n) => Some(count.saturating_sub(d.saturating_mul(*n)).clamp(0, *n)),
                DirScheme::Plain => Some(1),
                DirScheme::Constant => Some(*count),
                DirScheme::Mod(m) => Some(if (0..*count).contains(&d) { ceil_div(count - d, *m) } else { 0 }),
                DirScheme::NoDir | DirScheme::Unsupported => None,
            },
            _ => None,
        }
    }

    /// The permutation of positions for `epoch` (`consume`): over samples under `map`, over
    /// files under `stream` (`GRAMMAR_OPTIONS.md` §6.2).
    pub fn consume_perm(&self, epoch: i64) -> Result<Perm> {
        match self {
            DsMeta::Files { consume_base, .. } => {
                let n = self.draw_domain();
                if n == 0 {
                    bail!("consume from empty dataset `{}`", self.name());
                }
                Ok(Perm::new(n as u64, labeled_key(*consume_base, "epoch", &[epoch as u64])))
            }
            DsMeta::Array { .. } => bail!("`consume` on array dataset `{}`: use `pick`", self.name()),
        }
    }
}

pub struct NsMeta<'a> {
    pub name: &'a str,
    pub pattern: Pattern,
    pub fields: &'a BTreeMap<String, FieldType>,
    pub size: Option<&'a Expr>, // None = as_written
    pub seed: u64,
    pub input: bool,
}

/// Everything the VM reads: the AST, the parameters in effect, and the resolved datasets.
pub struct Model<'a> {
    pub ast: &'a Ast,
    pub cfg: &'a Config,
    pub params: &'a Params,
    pub datasets: Vec<DsMeta<'a>>,
    pub ds_index: HashMap<&'a str, usize>,
    pub namespaces: Vec<NsMeta<'a>>,
    pub ns_index: HashMap<&'a str, usize>,
    /// Parameter-default distributions resolved once (their arguments are parameter-only).
    pub param_dists: HashMap<usize, RDist<'a>>,
    /// The `trace` nodes' files (`Loaded::traces`); set by the caller after `build_model`.
    pub traces: HashMap<String, std::sync::Arc<crate::trace::TraceFile>>,
}

impl<'a> Model<'a> {
    pub fn dataset(&self, name: &str) -> Result<usize> {
        self.ds_index.get(name).copied().ok_or_else(|| anyhow!("unknown dataset `{name}`"))
    }

    /// Cap reads of dataset `name`'s inventory at its size at this count (`--subset`).
    pub fn cap_inventory(&mut self, name: &str) {
        if let Some(&i) = self.ds_index.get(name) {
            self.datasets[i].cap_inventory();
        }
    }

    pub fn namespace(&self, name: &str) -> Result<usize> {
        self.ns_index.get(name).copied().ok_or_else(|| anyhow!("unknown namespace `{name}`"))
    }

    /// The pattern's directory scheme and count, from the `id` fields of its directory part.
    /// `files` is the file count, not the sample count: the pattern's `id` is a file's (a shard's
    /// under `compound`), so a compound dataset of 10 files of 1,024 samples under `{id div 10}`
    /// is one directory, not 1,024 (`DESIGN_REVIEW.md` §3.76, fixed 2026-10-14).
    pub fn dir_scheme(pattern: &Pattern, files: i64) -> (DirScheme, i64) {
        if pattern.dir_segments == 0 {
            return (DirScheme::NoDir, 0);
        }
        let ids = pattern.dir_id_fields();
        match ids.as_slice() {
            [] => (DirScheme::Constant, 1),
            [f] => match f.op {
                None => (DirScheme::Plain, files),
                Some((crate::pattern::FieldOp::Div, n)) => (DirScheme::Div(n), ceil_div(files, n)),
                Some((crate::pattern::FieldOp::Mod, n)) => (DirScheme::Mod(n), files.min(n)),
            },
            _ => (DirScheme::Unsupported, 0),
        }
    }
}

/// A dataset's or namespace's pattern with each parameter divisor bound to the parameter's
/// value in effect (contract 0.10, V23): an integer scalar.
pub fn bind_pattern(src: &str, params: &Params) -> Result<Pattern> {
    Pattern::parse(src)?.bind(|name| match params.values.get(name) {
        Some(PValue::Scalar(Literal::Int(n))) => Ok(*n),
        Some(_) => bail!("parameter `{name}` is not an integer"),
        None => bail!("unknown parameter `{name}`"),
    })
}

/// Build the model. Dataset and namespace metadata expressions are evaluated with a VM over
/// a model that has no datasets yet, so they may use parameters but not other datasets.
pub fn build_model<'a>(ast: &'a Ast, cfg: &'a Config, params: &'a Params) -> Result<Model<'a>> {
    let stub = Model {
        ast,
        cfg,
        params,
        datasets: Vec::new(),
        ds_index: HashMap::new(),
        namespaces: Vec::new(),
        ns_index: HashMap::new(),
        param_dists: HashMap::new(),
        traces: HashMap::new(),
    };
    let mut vm = crate::vm::Vm::new(&stub, "static", 0, cfg.gpus.max(1));

    let mut param_dists = HashMap::new();
    for (name, v) in &params.values {
        if let PValue::Dist(d) = v {
            let r = vm.resolve_dist(d).with_context(|| format!("param `{name}`"))?;
            param_dists.insert(d as *const Dist as usize, r);
        }
    }

    let mut datasets = Vec::new();
    let mut ds_index = HashMap::new();
    for (name, d) in &ast.datasets {
        let ctx = || format!("dataset `{name}`");
        let meta = match d.layout {
            Layout::Whole | Layout::Chunked | Layout::Compound => {
                let f = d;
                let pattern = bind_pattern(&f.pattern, params).with_context(ctx)?;
                let count = vm.eval_int(&f.count).with_context(ctx)?;
                let spf = match &f.samples_per_file {
                    Some(e) => vm.eval_int(e).with_context(ctx)?,
                    None => 1,
                };
                if count < 0 || spf < 1 {
                    bail!("dataset `{name}`: count {count}, samples_per_file {spf}");
                }
                let chunk = match &f.chunk {
                    Some(e) => Some(vm.eval_int(e).with_context(ctx)?),
                    None => None,
                };
                if chunk == Some(0) || chunk.map_or(false, |c| c < 0) {
                    bail!("dataset `{name}`: chunk must be positive");
                }
                if chunk.is_some() && !pattern.field_names().contains(&"k") {
                    bail!("dataset `{name}`: a chunked dataset's pattern needs a {{k}} field");
                }
                let size = vm.resolve_distref(&f.size).with_context(ctx)?;
                let access = f.access;
                let framing = match f.format.as_ref().and_then(|fm| fm.framing.as_ref()) {
                    None => Framing::packed(spf),
                    Some(l) => {
                        let mut int = |e: &'a Option<Expr>, dflt: i64, what: &str| -> Result<i64> {
                            let v = match e {
                                Some(e) => vm.eval_int(e).with_context(|| format!("dataset `{name}`: framing {what}"))?,
                                None => dflt,
                            };
                            if v < dflt.min(0) {
                                bail!("dataset `{name}`: framing {what} is {v}");
                            }
                            Ok(v)
                        };
                        let unit = int(&l.unit, spf, "unit")?;
                        if unit < 1 || unit > spf {
                            bail!("dataset `{name}`: framing unit {unit} must be in [1, samples_per_file = {spf}]");
                        }
                        let mut columns = Vec::new();
                        match &l.columns {
                            None => columns.push(Column { header: 0, fixed: 0, weight: 1.0, row_header: 0, row_footer: 0, row_align: 1, align: 1, last_weighted: true }),
                            Some(cols) => {
                                let last = cols.iter().rposition(|c| c.weight > 0.0);
                                for (i, c) in cols.iter().enumerate() {
                                    columns.push(Column {
                                        header: int(&c.header, 0, "column header")?,
                                        fixed: int(&c.fixed, 0, "column fixed")?,
                                        weight: c.weight,
                                        row_header: int(&c.row_header, 0, "row_header")?,
                                        row_footer: int(&c.row_footer, 0, "row_footer")?,
                                        row_align: int(&c.row_align, 1, "row_align")?.max(1),
                                        align: int(&c.align, 1, "column align")?.max(1),
                                        last_weighted: last == Some(i),
                                    });
                                }
                            }
                        }
                        Framing {
                            unit,
                            file_header: int(&l.file_header, 0, "file_header")?,
                            file_header_per_sample: int(&l.file_header_per_sample, 0, "file_header_per_sample")?,
                            file_header_per_unit: int(&l.file_header_per_unit, 0, "file_header_per_unit")?,
                            file_footer: int(&l.file_footer, 0, "file_footer")?,
                            file_footer_per_sample: int(&l.file_footer_per_sample, 0, "file_footer_per_sample")?,
                            file_footer_per_unit: int(&l.file_footer_per_unit, 0, "file_footer_per_unit")?,
                            file_align: int(&l.file_align, 1, "file_align")?.max(1),
                            unit_header: int(&l.unit_header, 0, "unit_header")?,
                            unit_header_per_sample: int(&l.unit_header_per_sample, 0, "unit_header_per_sample")?,
                            unit_footer: int(&l.unit_footer, 0, "unit_footer")?,
                            unit_footer_per_sample: int(&l.unit_footer_per_sample, 0, "unit_footer_per_sample")?,
                            unit_align: int(&l.unit_align, 1, "unit_align")?.max(1),
                            columns,
                        }
                    }
                };
                let (dirs, ndirs) = Model::dir_scheme(&pattern, ceil_div(count, spf));
                let inventory = match &f.inventory {
                    Some(i) => {
                        let entry = vm.eval_int(&i.entry).with_context(|| format!("dataset `{name}`: inventory entry"))?;
                        if entry < 1 {
                            bail!("dataset `{name}`: inventory entry {entry} must be positive");
                        }
                        if count.checked_mul(entry).is_none() {
                            bail!("dataset `{name}`: its inventory, count {count} × entry {entry} bytes, overflows the 64-bit integers of the contract (2^63-1)");
                        }
                        let path = crate::payload::join_key(pattern.root(), &i.name);
                        Some(InvMeta { path: Arc::from(path), entry, cap: false })
                    }
                    None => None,
                };
                let name_hash = crate::rng::labeled_key(0, name, &[]);
                DsMeta::Files {
                    name,
                    pattern,
                    count,
                    spf,
                    chunk,
                    access,
                    framing,
                    size,
                    seed: f.seed,
                    dirs,
                    ndirs,
                    consume_base: labeled_key(cfg.seed, "consume", &[name_hash]),
                    rank_key: labeled_key(f.seed, "rank", &[]),
                    inventory,
                }
            }
            Layout::Array => {
                let r = d;
                let count = vm.eval_int(&r.count).with_context(ctx)?;
                let slot = match &r.slot {
                    Some(e) => vm.eval_int(e).with_context(ctx)?,
                    None => bail!("dataset `{name}`: an `array` needs `slot`"),
                };
                if count < 0 || slot <= 0 {
                    bail!("dataset `{name}`: count {count}, slot {slot}");
                }
                if count.checked_mul(slot).is_none() {
                    bail!("dataset `{name}`: its file, count {count} × slot {slot} bytes, overflows the 64-bit integers of the contract (2^63-1)");
                }
                let size = vm.resolve_distref(&r.size).with_context(ctx)?;
                DsMeta::Array {
                    name,
                    file: Arc::from(r.pattern.as_str()),
                    count,
                    slot,
                    size,
                    seed: r.seed,
                    rank_key: labeled_key(r.seed, "rank", &[]),
                }
            }
        };
        ds_index.insert(name.as_str(), datasets.len());
        datasets.push(meta);
    }

    let mut namespaces = Vec::new();
    let mut ns_index = HashMap::new();
    for (name, n) in &ast.namespaces {
        let pattern = bind_pattern(&n.pattern, params).with_context(|| format!("namespace `{name}`"))?;
        for f in pattern.field_names() {
            if !n.fields.contains_key(f) {
                bail!("namespace `{name}`: pattern field `{{{f}}}` is not a declared field");
            }
        }
        let size = match &n.size {
            NsSize::AsWritten(_) => None,
            NsSize::Expr(e) => Some(e),
        };
        ns_index.insert(name.as_str(), namespaces.len());
        namespaces.push(NsMeta { name, pattern, fields: &n.fields, size, seed: n.seed, input: n.input.unwrap_or(false) });
    }
    drop(vm);

    Ok(Model { ast, cfg, params, datasets, ds_index, namespaces, ns_index, param_dists, traces: HashMap::new() })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::vm::{run_actor, Op, OpCtx, OpKind, Sink};

    thread_local! {
        /// Sample sizes drawn on this thread (`DsMeta::sample_size` of a files dataset).
        pub(crate) static SIZE_DRAWS: Cell<u64> = const { Cell::new(0) };
    }

    /// One file of `n` lognormal samples (or files of `spf`), every sample opened, read at its
    /// offset, and closed once, through its container's file handle (an HDF5 or TFRecord reader).
    fn dataset(n: i64, spf: i64, format: &str) -> String {
        format!(
            r#"{{
  "ast": "0.10", "name": "geom_test",
  "datasets": {{"d": {{"layout": "compound", "pattern": "d/{{id:04}}.bin", "count": {n}, "samples_per_file": {spf},
     "size": {{"lognormal": {{"median": 1000, "sigma": 0.5, "min": 1}}}}, "seed": 7, "access": "map"{format}}}}},
  "actors": {{"gpu": {{"body": [
    {{"loop": {{"index": "j", "to": {n}, "body": [
      {{"let": {{"name": "s", "value": {{"consume": "d"}}}}}},
      {{"let": {{"name": "f", "value": {{"file": {{"of": {{"ref": "s"}}}}}}}}}},
      {{"open": {{"file": {{"ref": "f"}}, "flags": ["RDONLY"]}}}},
      {{"fstat": {{"file": {{"ref": "f"}}}}}},
      {{"read": {{"file": {{"ref": "f"}}, "len": 8, "offset": 0}}}},
      {{"read": {{"file": {{"ref": "f"}}, "len": {{"size": {{"ref": "s"}}}}, "offset": {{"offset": {{"ref": "s"}}}}}}}},
      {{"close": {{"file": {{"ref": "f"}}}}}}
    ]}}}}
  ]}}}}
}}"#
        )
    }

    const TFRECORD: &str = r#", "format": {"class": "tfrecord", "reader": "tf.data", "framing": {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}}"#;
    const PARQUET: &str = r#", "format": {"class": "parquet", "reader": "pyarrow", "framing": {"unit": 7, "file_header": 4, "file_footer": 200, "file_footer_per_unit": 120,
        "columns": [{"header": 26, "row_header": 4, "weight": 1}, {"header": 22, "fixed": 8, "weight": 0}]}}"#;

    fn with_model(text: &str, f: impl FnOnce(&Model)) {
        let loaded = crate::load_str(text).expect("valid abstract");
        let cfg = Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![] };
        let params = Params::new(&loaded.ast, &cfg).unwrap();
        f(&build_model(&loaded.ast, &cfg, &params).unwrap())
    }

    #[derive(Default)]
    struct Reads(Vec<(i64, i64)>);

    impl<'m, 'a> Sink<'m, 'a> for Reads {
        fn op(&mut self, op: &Op, _: &OpCtx) -> anyhow::Result<()> {
            if op.kind == OpKind::Read && op.len != 8 {
                self.0.push((op.offset, op.len));
            }
            Ok(())
        }
    }

    /// RC1 review: every op on a sample of a container file drew the sizes of all the file's
    /// samples again (`file_size` for its target, `sample_offset` for its offset), O(samples
    /// per file) per op and O(n²) per file. The VM now holds the file's geometry while it reads
    /// it: a pass over a file's n samples draws O(n) sizes, and the offsets are the same.
    #[test]
    fn a_pass_over_a_files_samples_draws_their_sizes_a_bounded_number_of_times() {
        let n = 400i64;
        // one file; then 20 files visited in shuffled order, past the 16 the VM holds
        for (spf, format) in [(n, ""), (n, TFRECORD), (20, ""), (20, TFRECORD)] {
            with_model(&dataset(n, spf, format), |model| {
                SIZE_DRAWS.with(|c| c.set(0));
                let mut got = run_actor(model, "gpu", 0, 1, Reads::default()).unwrap().0;
                let draws = SIZE_DRAWS.with(|c| c.get());
                assert!(spf < n || draws <= 4 * n as u64, "format `{format}`: {draws} size draws for one pass over {n} samples (n² is {})", n * n);
                let meta = &model.datasets[0];
                let mut want: Vec<(i64, i64)> = (0..n).map(|id| (meta.sample_offset(id).unwrap(), meta.sample_size(id).unwrap())).collect();
                want.sort();
                got.sort();
                assert_eq!(got, want, "format `{format}`, {spf} samples per file");
            });
        }
    }

    #[test]
    fn file_geom_is_file_size_and_sample_offset() {
        // 3 files of 50, 50, and 23 samples: packed, one framed column, two columns (no offsets)
        for format in ["", TFRECORD, PARQUET] {
            with_model(&dataset(123, 50, format), |model| {
                let meta = &model.datasets[0];
                for file in 0..3 {
                    let g = meta.file_geom(file, usize::MAX).unwrap();
                    assert_eq!(g.size, meta.file_size(file).unwrap(), "format `{format}` file {file}");
                    let n = meta.samples_in_file(file).unwrap();
                    match &g.offsets {
                        Some(o) => {
                            assert_eq!(o.len() as i64, n);
                            for (i, off) in o.iter().enumerate() {
                                assert_eq!(*off, meta.sample_offset(file * 50 + i as i64).unwrap(), "format `{format}` file {file} sample {i}");
                            }
                        }
                        None => {
                            assert_eq!(format, PARQUET);
                            assert!(meta.sample_offset(file * 50).is_err());
                        }
                    }
                }
                assert!(meta.file_geom(3, usize::MAX).is_err());
            });
        }
    }
}
