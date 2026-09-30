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
}

/// Parameters in effect: defaults with the overrides applied. Owned, so `Value` can borrow it.
pub struct Params {
    pub values: BTreeMap<String, PValue>,
}

impl Params {
    pub fn new(ast: &Ast, cfg: &Config) -> Result<Params> {
        let mut values: BTreeMap<String, PValue> = ast.params.iter().map(|(k, p)| (k.clone(), p.default.clone())).collect();
        for (name, text) in &cfg.overrides {
            if name == "gpus" {
                bail!("`gpus` is set with --gpus, not --param");
            }
            let param = ast.params.get(name).ok_or_else(|| anyhow!("--param {name}: no such parameter"))?;
            if param.cli == Some(false) {
                bail!("--param {name}: not overridable (cli: false)");
            }
            let v: PValue = match serde_json::from_str(text) {
                Ok(v) => v,
                Err(_) => PValue::Scalar(Literal::Str(text.clone())),
            };
            values.insert(name.clone(), v);
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
    /// Region `c` of a regions dataset.
    Region { ds: usize, c: i64 },
    /// The single file of a regions dataset.
    RegionFile { ds: usize },
    /// Directory `d` of a files dataset's pattern.
    Dir { ds: usize, d: i64 },
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
                Ok(x.round() as i64)
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

pub enum DsMeta<'a> {
    Files {
        name: &'a str,
        pattern: Pattern,
        count: i64,
        spf: i64,
        chunk: Option<i64>,
        access: Access,
        size: RDist<'a>,
        seed: u64,
        dirs: DirScheme,
        /// Number of directories, per `dirs`.
        ndirs: i64,
        /// `labeled_key(seed, "consume", name)`, the base of the per-epoch permutation keys.
        consume_base: u64,
        rank_key: u64,
    },
    Regions {
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
            DsMeta::Files { name, .. } | DsMeta::Regions { name, .. } => name,
        }
    }

    pub fn count(&self) -> i64 {
        match self {
            DsMeta::Files { count, .. } | DsMeta::Regions { count, .. } => *count,
        }
    }

    pub fn dataset_seed(&self) -> u64 {
        match self {
            DsMeta::Files { seed, .. } | DsMeta::Regions { seed, .. } => *seed,
        }
    }

    /// The rank order of ids for `zipf` and `hotset`, fixed by the dataset seed.
    pub fn rank_perm(&self) -> Perm {
        let key = match self {
            DsMeta::Files { rank_key, .. } | DsMeta::Regions { rank_key, .. } => *rank_key,
        };
        Perm::new(self.count().max(1) as u64, key)
    }

    /// Bytes of sample `id` (files) or region `c` (regions, capped at the slot).
    pub fn sample_size(&self, id: i64) -> Result<i64> {
        match self {
            DsMeta::Files { size, seed, count, .. } => {
                if id < 0 || id >= *count {
                    bail!("sample {id} outside dataset `{}` of {count}", self.name());
                }
                let mut w = Words::new(dataset_key(*seed, id));
                sample(size, &mut w, None)?.as_int()
            }
            DsMeta::Regions { size, seed, count, slot, .. } => {
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
            DsMeta::Regions { .. } => bail!("`{}` is a regions dataset", self.name()),
        }
    }

    /// Bytes of a container file: the sum of its samples' sizes.
    pub fn file_size(&self, file: i64) -> Result<i64> {
        match self {
            DsMeta::Files { spf, .. } => {
                let n = self.samples_in_file(file)?;
                if *spf == 1 {
                    return self.sample_size(file);
                }
                let mut total = 0i64;
                for i in 0..n {
                    total += self.sample_size(file * spf + i)?;
                }
                Ok(total)
            }
            DsMeta::Regions { count, slot, .. } => {
                if file != 0 {
                    bail!("a regions dataset has one file, id 0");
                }
                if *count == 0 {
                    return Ok(0);
                }
                Ok((count - 1) * slot + self.sample_size(count - 1)?)
            }
        }
    }

    /// Offset of sample `id` inside its container (prefix sum over the file's earlier samples).
    pub fn sample_offset(&self, id: i64) -> Result<i64> {
        match self {
            DsMeta::Files { spf, .. } => {
                let file = id / spf;
                let mut off = 0i64;
                for i in 0..(id - file * spf) {
                    off += self.sample_size(file * spf + i)?;
                }
                Ok(off)
            }
            DsMeta::Regions { slot, .. } => Ok(id * slot),
        }
    }

    pub fn chunk_size(&self, file: i64, k: i64) -> Result<i64> {
        match self {
            DsMeta::Files { chunk: Some(c), .. } => {
                let size = self.file_size(file)?;
                if k < 0 || k * c >= size {
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
                Ok((size + c - 1) / c)
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
            DsMeta::Regions { file: path, .. } => {
                if file != 0 {
                    bail!("a regions dataset has one file, id 0");
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
            DsMeta::Regions { .. } => bail!("`dir` on regions dataset `{}`", self.name()),
        }
    }

    /// The permutation of sample positions for `epoch` (`consume`).
    pub fn consume_perm(&self, epoch: i64) -> Result<Perm> {
        match self {
            DsMeta::Files { count, consume_base, .. } => {
                if *count == 0 {
                    bail!("consume from empty dataset `{}`", self.name());
                }
                Ok(Perm::new(*count as u64, labeled_key(*consume_base, "epoch", &[epoch as u64])))
            }
            DsMeta::Regions { .. } => bail!("`consume` on regions dataset `{}`: use `pick`", self.name()),
        }
    }
}

pub struct NsMeta<'a> {
    pub name: &'a str,
    pub pattern: Pattern,
    pub fields: &'a BTreeMap<String, FieldType>,
    pub size: Option<&'a Expr>, // None = as_written
    pub seed: u64,
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
}

impl<'a> Model<'a> {
    pub fn dataset(&self, name: &str) -> Result<usize> {
        self.ds_index.get(name).copied().ok_or_else(|| anyhow!("unknown dataset `{name}`"))
    }

    pub fn namespace(&self, name: &str) -> Result<usize> {
        self.ns_index.get(name).copied().ok_or_else(|| anyhow!("unknown namespace `{name}`"))
    }

    /// The pattern's directory scheme and count, from the `id` fields of its directory part.
    pub fn dir_scheme(pattern: &Pattern, count: i64) -> (DirScheme, i64) {
        if pattern.dir_segments == 0 {
            return (DirScheme::NoDir, 0);
        }
        let ids = pattern.dir_id_fields();
        match ids.as_slice() {
            [] => (DirScheme::Constant, 1),
            [f] => match f.op {
                None => (DirScheme::Plain, count),
                Some((crate::pattern::FieldOp::Div, n)) => (DirScheme::Div(n), (count + n - 1) / n),
                Some((crate::pattern::FieldOp::Mod, n)) => (DirScheme::Mod(n), count.min(n)),
            },
            _ => (DirScheme::Unsupported, 0),
        }
    }
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
    };
    let mut vm = crate::vm::Vm::new(&stub, crate::vm::NullSink, "static", 0, cfg.gpus.max(1));

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
        let meta = match d {
            Dataset::Files(f) => {
                let pattern = Pattern::parse(&f.pattern).with_context(ctx)?;
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
                let access = f.access.unwrap_or(Access::Map);
                if access == Access::Stream {
                    bail!("dataset `{name}`: `stream` access is not implemented yet (schema §7)");
                }
                let (dirs, ndirs) = Model::dir_scheme(&pattern, count);
                let name_hash = crate::rng::labeled_key(0, name, &[]);
                DsMeta::Files {
                    name,
                    pattern,
                    count,
                    spf,
                    chunk,
                    access,
                    size,
                    seed: f.seed,
                    dirs,
                    ndirs,
                    consume_base: labeled_key(cfg.seed, "consume", &[name_hash]),
                    rank_key: labeled_key(f.seed, "rank", &[]),
                }
            }
            Dataset::Regions(r) => {
                let count = vm.eval_int(&r.count).with_context(ctx)?;
                let slot = vm.eval_int(&r.slot).with_context(ctx)?;
                if count < 0 || slot <= 0 {
                    bail!("dataset `{name}`: count {count}, slot {slot}");
                }
                let size = vm.resolve_distref(&r.size).with_context(ctx)?;
                DsMeta::Regions {
                    name,
                    file: Arc::from(r.file.as_str()),
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
        let pattern = Pattern::parse(&n.pattern).with_context(|| format!("namespace `{name}`"))?;
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
        namespaces.push(NsMeta { name, pattern, fields: &n.fields, size, seed: n.seed });
    }
    drop(vm);

    Ok(Model { ast, cfg, params, datasets, ds_index, namespaces, ns_index, param_dists })
}
