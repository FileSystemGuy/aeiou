//! The VM: evaluates expressions positionally and walks an actor instance's body, emitting
//! ops to a `Sink`. It holds no state that the abstract does not define: loop frames, `let`
//! bindings, open-file positions, and the as-written byte sums of objects this actor created
//! at this position. A `parallel` or `loader` body is walked once per sub-actor index, in
//! index order; that order is not observable through the sink (the fingerprint is a sum).
//!
//! Definitions the runner fixes here (`schema/README.md` §5 leaves them to it):
//! - a draw at site `s` inside frames with indices `i₁ … iₙ` is `Words(position_key(seed,
//!   actor, s, [i₁ … iₙ]))`; a `choose` is the same with the statement's site;
//! - `consume` position: with the enclosing `loader` (else the innermost loop) as the batch
//!   frame, `b` is the row-major ordinal of the frames down to and including it, `j` and `B`
//!   the ordinal and the product of the frames inside it; an epoch is `N div (G·B)` batches
//!   (`drop_last`); the id is `Perm(N, key(seed, dataset, epoch))(g + G·((b mod epoch_len)·B + j))`;
//! - `x @ i` evaluates `x`'s definition with the frame of `x`'s loop set to `i`, sibling
//!   `let`s of the same body re-evaluated at that index and memoized; below the loop's `from`
//!   the enclosing `cond` takes its other arm;
//! - `until_eof` issues reads of `len` from the position until the computed size is exhausted,
//!   then one more; `readdir` is one op.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ast::Access;
use crate::ast::*;
use crate::eval::*;
use crate::pattern::FieldValue;
use crate::rng::{position_key, Words};

// ---------------------------------------------------------------- ops and sinks

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpKind {
    Open = 1,
    Close,
    Read,
    Write,
    Lseek,
    Ioctl,
    Fstat,
    Stat,
    Fsync,
    Fdatasync,
    Unlink,
    Ftruncate,
    Fallocate,
    Mkdir,
    Rmdir,
    Rename,
    Readdir,
    Fadvise,
}

impl OpKind {
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Open => "open",
            OpKind::Close => "close",
            OpKind::Read => "read",
            OpKind::Write => "write",
            OpKind::Lseek => "lseek",
            OpKind::Ioctl => "ioctl",
            OpKind::Fstat => "fstat",
            OpKind::Stat => "stat",
            OpKind::Fsync => "fsync",
            OpKind::Fdatasync => "fdatasync",
            OpKind::Unlink => "unlink",
            OpKind::Ftruncate => "ftruncate",
            OpKind::Fallocate => "fallocate",
            OpKind::Mkdir => "mkdir",
            OpKind::Rmdir => "rmdir",
            OpKind::Rename => "rename",
            OpKind::Readdir => "readdir",
            OpKind::Fadvise => "fadvise",
        }
    }

    pub const ALL: [OpKind; 18] = [
        OpKind::Open, OpKind::Close, OpKind::Read, OpKind::Write, OpKind::Lseek, OpKind::Ioctl, OpKind::Fstat,
        OpKind::Stat, OpKind::Fsync, OpKind::Fdatasync, OpKind::Unlink, OpKind::Ftruncate, OpKind::Fallocate,
        OpKind::Mkdir, OpKind::Rmdir, OpKind::Rename, OpKind::Readdir, OpKind::Fadvise,
    ];
}

/// One operation as the abstract issues it: POSIX-shaped, backend-independent.
#[derive(Debug, Clone)]
pub struct Op<'c> {
    pub kind: OpKind,
    pub path: &'c str,
    /// `rename`'s destination.
    pub path2: Option<&'c str>,
    /// Effective byte offset (reads, writes, fallocate) or the `lseek` argument.
    pub offset: i64,
    /// Requested length (reads, writes, ftruncate, fallocate).
    pub len: i64,
    /// Bytes the op is expected to transfer: the returned count of a read, `len` of a write;
    /// for `readdir`, the entries expected (−1 when not computable).
    pub bytes: i64,
    /// Open flags and mode, `lseek` whence, `ioctl` request, `mkdir` mode.
    pub aux: u64,
    /// The op's `expect` list: errno names the op may fail with (empty: none).
    pub expect: &'c [String],
    /// A read or write that names its offset (`pread`/`pwrite`) rather than using the position.
    pub positioned: bool,
    /// The payload key of the target: the namespace seed for an object, the dataset seed for a
    /// dataset file, 0 otherwise. Not hashed; a backend that writes uses it to generate content.
    pub seed: u64,
}

/// Where an op comes from: the actor instance, its position, and the statistics label.
#[derive(Debug, Clone, Copy)]
pub struct OpCtx<'c> {
    pub template: &'c str,
    pub actor: i64,
    pub indices: &'c [i64],
    pub phase: Option<&'c str>,
}

#[derive(Debug, Clone, Copy)]
pub enum Control<'a> {
    Compute { ns: i64 },
    Barrier { scope: &'a str },
    Put { channel: &'a str, seq: i64 },
    Take { channel: &'a str },
    Channel { name: &'a str, capacity: i64, ordered: bool },
}

/// A `parallel` or `loader` node: the sub-actors a sink may run concurrently.
#[derive(Debug, Clone, Copy)]
pub enum ForkKind<'a> {
    Parallel { index: &'a str, width: i64 },
    Loader { name: &'a str, index: &'a str, workers: i64, prefetch: i64, batches: i64, ordered: bool },
}

/// Where the VM delivers what it walks. `dry-run` counts; `run` does the I/O.
pub trait Sink<'m, 'a>: Sized {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> Result<()>;
    fn control(&mut self, _c: Control<'a>, _ctx: &OpCtx) -> Result<()> {
        Ok(())
    }
    /// A `parallel` or `loader`. Return `Ok(false)` to have the VM walk the sub-actors inline,
    /// in index order (the dry run). A sink that runs them itself takes a `Snapshot` from the
    /// closure, resumes one `Vm` per sub-actor from it, and calls `run_sub` per index; a
    /// `parallel` is joined before returning, a `loader` at `finish`.
    fn fork(&mut self, _kind: &ForkKind<'a>, _snapshot: &dyn Fn() -> Snapshot<'m, 'a>) -> Result<bool> {
        Ok(false)
    }
    /// The actor instance (or sub-actor) has walked its whole body.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct NullSink;
impl<'m, 'a> Sink<'m, 'a> for NullSink {
    fn op(&mut self, _op: &Op, _ctx: &OpCtx) -> Result<()> {
        Ok(())
    }
}

/// The per-op hash whose sum over the run is the workload fingerprint. Order-independent by
/// construction (`NAPKIN_MATH.md` §8.D); `phase`, `expect`, and results are not hashed.
pub fn op_hash(op: &Op, ctx: &OpCtx) -> u64 {
    let mut buf = Vec::with_capacity(64 + op.path.len());
    buf.push(op.kind as u8);
    buf.extend_from_slice(&(ctx.actor as u64).to_le_bytes());
    buf.extend_from_slice(&(ctx.indices.len() as u32).to_le_bytes());
    for i in ctx.indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf.extend_from_slice(&op.offset.to_le_bytes());
    buf.extend_from_slice(&op.len.to_le_bytes());
    buf.extend_from_slice(&op.aux.to_le_bytes());
    buf.extend_from_slice(op.path.as_bytes());
    if let Some(p2) = op.path2 {
        buf.push(0);
        buf.extend_from_slice(p2.as_bytes());
    }
    xxhash_rust::xxh3::xxh3_64_with_seed(&buf, 0x0f1e_6e12_a1e7_0000)
}

pub fn flag_bits(flags: &[OpenFlag]) -> u64 {
    flags.iter().fold(0u64, |acc, f| acc | (1 << (*f as u8)))
}

// ---------------------------------------------------------------- the VM

#[derive(Debug, Clone, Copy)]
struct Frame<'a> {
    name: &'a str,
    idx: i64,
    from: i64,
    step: i64,
    iters: i64,
    loader: bool,
}

#[derive(Clone)]
struct Binding<'a> {
    value: Value<'a>,
    /// Frames enclosing the definition.
    depth: usize,
    body: &'a [Node],
}

#[derive(Clone)]
struct ScopeLevel<'a> {
    bindings: HashMap<&'a str, Binding<'a>>,
    body: &'a [Node],
    depth: usize,
}

/// An active `x @ i` evaluation: frame `frame` reads as `idx`, sibling lets of `body` are
/// re-evaluated at that index and memoized.
struct Shift<'a> {
    frame: usize,
    idx: i64,
    body: &'a [Node],
    depth: usize,
    memo: HashMap<&'a str, Value<'a>>,
}

#[derive(Debug, Clone)]
struct FileState {
    pos: i64,
    size: Option<i64>,
    as_written: bool,
}

/// A handle resolved to something an op can name.
#[derive(Debug, Clone)]
struct Target {
    path: Arc<str>,
    size: Option<i64>,
    as_written: bool,
    seed: u64,
    /// For a dataset directory: the entries a `readdir` is expected to return, when computable.
    entries: Option<i64>,
}

/// What a sub-actor starts from: the parent's position (frames, with the fork's frame last),
/// its visible bindings, open files, as-written sums, and the sub-actor body. Cloned once per
/// sub-actor thread by a forking sink.
#[derive(Clone)]
pub struct Snapshot<'m, 'a> {
    model: &'m Model<'a>,
    template: &'a str,
    actor: i64,
    actor_count: i64,
    frames: Vec<Frame<'a>>,
    scopes: Vec<ScopeLevel<'a>>,
    phases: Vec<Arc<str>>,
    open: HashMap<Arc<str>, FileState>,
    written: HashMap<Arc<str>, i64>,
    body: &'a [Node],
}

impl<'m, 'a> Snapshot<'m, 'a> {
    pub fn template(&self) -> &'a str {
        self.template
    }
    pub fn actor(&self) -> i64 {
        self.actor
    }
    /// The enclosing loop indices at the fork, the fork's own index last (its `from`).
    pub fn indices(&self) -> Vec<i64> {
        self.frames.iter().map(|f| f.idx).collect()
    }
    pub fn phase(&self) -> Option<Arc<str>> {
        self.phases.last().cloned()
    }
}

enum EvalError {
    BelowFrom,
    Other(anyhow::Error),
}

impl From<anyhow::Error> for EvalError {
    fn from(e: anyhow::Error) -> Self {
        EvalError::Other(e)
    }
}

type ER<T> = std::result::Result<T, EvalError>;

fn other<T>(msg: impl Into<String>) -> ER<T> {
    Err(EvalError::Other(anyhow!(msg.into())))
}

pub struct Vm<'m, 'a, S: Sink<'m, 'a>> {
    pub model: &'m Model<'a>,
    pub sink: S,
    template: &'a str,
    actor: i64,
    actor_count: i64,
    frames: Vec<Frame<'a>>,
    scopes: Vec<ScopeLevel<'a>>,
    shifts: Vec<Shift<'a>>,
    /// Frames visible to the expression being evaluated (shorter during a shifted evaluation).
    depth: usize,
    fields: Option<HashMap<&'a str, Value<'a>>>,
    open: HashMap<Arc<str>, FileState>,
    written: HashMap<Arc<str>, i64>,
    phases: Vec<Arc<str>>,
    indices_buf: Vec<i64>,
    /// The sub-actor body of a resumed VM.
    sub_body: Option<&'a [Node]>,
    /// Unit lengths of recently addressed container files (`GRAMMAR_OPTIONS.md` §6.3: the
    /// layout is computed once per open file and held). Bounded; never shared.
    unit_lens: HashMap<(usize, i64), Arc<Vec<i64>>>,
}

impl<'m, 'a: 'm, S: Sink<'m, 'a>> Vm<'m, 'a, S> {
    pub fn new(model: &'m Model<'a>, sink: S, template: &'a str, actor: i64, actor_count: i64) -> Self {
        Vm {
            model,
            sink,
            template,
            actor,
            actor_count,
            frames: Vec::new(),
            scopes: vec![ScopeLevel { bindings: HashMap::new(), body: &[], depth: 0 }],
            shifts: Vec::new(),
            depth: 0,
            fields: None,
            open: HashMap::new(),
            written: HashMap::new(),
            phases: Vec::new(),
            indices_buf: Vec::new(),
            unit_lens: HashMap::new(),
            sub_body: None,
        }
    }

    pub fn into_sink(self) -> S {
        self.sink
    }

    /// A sub-actor VM: the parent's state at the fork, with its own sink.
    pub fn resume(sink: S, snap: Snapshot<'m, 'a>) -> Self {
        let depth = snap.frames.len();
        Vm {
            model: snap.model,
            sink,
            template: snap.template,
            actor: snap.actor,
            actor_count: snap.actor_count,
            frames: snap.frames,
            scopes: snap.scopes,
            shifts: Vec::new(),
            depth,
            fields: None,
            open: snap.open,
            written: snap.written,
            phases: snap.phases,
            indices_buf: Vec::new(),
            unit_lens: HashMap::new(),
            sub_body: Some(snap.body),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn snapshot(model: &'m Model<'a>, template: &'a str, actor: i64, actor_count: i64, frames: &[Frame<'a>], scopes: &[ScopeLevel<'a>], phases: &[Arc<str>], open: &HashMap<Arc<str>, FileState>, written: &HashMap<Arc<str>, i64>, body: &'a [Node]) -> Snapshot<'m, 'a> {
        Snapshot {
            model,
            template,
            actor,
            actor_count,
            frames: frames.to_vec(),
            scopes: scopes.to_vec(),
            phases: phases.to_vec(),
            open: open.clone(),
            written: written.clone(),
            body,
        }
    }

    /// Run one actor instance to completion.
    pub fn run(&mut self, body: &'a [Node]) -> Result<()> {
        self.body(body).map_err(|e| self.wrap(e))?;
        self.sink.finish()
    }

    /// On a resumed VM: walk the sub-actor body once at index `k` of the fork (ordinal, so the
    /// frame's index is `from + k·step`).
    pub fn run_sub(&mut self, k: i64) -> Result<()> {
        let body = self.sub_body.expect("run_sub on a VM that was not resumed from a snapshot");
        let f = self.frames.last_mut().expect("a resumed VM has the fork frame");
        f.idx = f.from + k * f.step;
        self.body(body).map_err(|e| self.wrap(e))
    }

    fn wrap(&self, e: EvalError) -> anyhow::Error {
        let e = match e {
            EvalError::BelowFrom => anyhow!("`at` below the loop's start with no alternative arm"),
            EvalError::Other(e) => e,
        };
        let pos: Vec<String> = self.frames.iter().map(|f| format!("{}={}", f.name, f.idx)).collect();
        e.context(format!("actor {}#{} at [{}]", self.template, self.actor, pos.join(", ")))
    }

    // ---- positions ----

    fn frame_index(&self, f: usize) -> i64 {
        for s in self.shifts.iter().rev() {
            if s.frame == f {
                return s.idx;
            }
        }
        self.frames[f].idx
    }

    fn frame_ordinal(&self, f: usize) -> i64 {
        let fr = &self.frames[f];
        (self.frame_index(f) - fr.from).div_euclid(fr.step)
    }

    fn indices(&self, depth: usize) -> Vec<i64> {
        (0..depth).map(|f| self.frame_index(f)).collect()
    }

    fn draw_key(&self, site: u64) -> u64 {
        let idx = self.indices(self.depth);
        position_key(self.model.cfg.seed, self.actor as u64, site, &idx)
    }

    // ---- public evaluation entry points (used by the model builder) ----

    pub fn eval_int(&mut self, e: &'a Expr) -> Result<i64> {
        self.expr(e).map_err(|e| self.wrap(e))?.as_int()
    }

    pub fn resolve_distref(&mut self, d: &'a DistRef) -> Result<RDist<'a>> {
        self.distref(d).map_err(|e| self.wrap(e))
    }

    pub fn resolve_dist(&mut self, d: &'a Dist) -> Result<RDist<'a>> {
        self.dist(d).map_err(|e| self.wrap(e))
    }

    // ---- expressions ----

    fn expr(&mut self, e: &'a Expr) -> ER<Value<'a>> {
        let node = match e {
            Expr::Lit(l) => return Ok(Value::from_literal(l)),
            Expr::Node(n) => n.as_ref(),
        };
        Ok(match node {
            ExprNode::Param(name) => self.param(name)?,
            ExprNode::Index(name) => self.index(name)?,
            ExprNode::Ref(name) => self.reference(name)?,
            ExprNode::At { name, index } => {
                let i = self.expr(index)?.as_int()?;
                self.at(name, i)?
            }
            ExprNode::Actor(ActorAttr::Id) => Value::Int(self.actor),
            ExprNode::Actor(ActorAttr::Count) => Value::Int(self.actor_count),
            ExprNode::Draw(d) => {
                let rd = self.distref(&d.dist)?;
                let mut w = Words::new(self.draw_key(d.site.get()));
                sample(&rd, &mut w, None)?
            }
            ExprNode::Elem { array, index } => {
                let i = self.expr(index)?.as_int()?;
                match self.param(array)? {
                    Value::Array(a) => {
                        if i < 0 || i as usize >= a.len() {
                            return other(format!("elem: index {i} outside parameter array `{array}` of {}", a.len()));
                        }
                        Value::from_pvalue(&a[i as usize])
                    }
                    _ => return other(format!("elem: parameter `{array}` is not an array")),
                }
            }
            ExprNode::Len(array) => match self.param(array)? {
                Value::Array(a) => Value::Int(a.len() as i64),
                _ => return other(format!("len: parameter `{array}` is not an array")),
            },
            ExprNode::Sum(array) => match self.param(array)? {
                Value::Array(a) => {
                    let mut int = 0i64;
                    let mut float = 0f64;
                    let mut any_float = false;
                    for x in a {
                        match Value::from_pvalue(x) {
                            Value::Int(n) => int = int.checked_add(n).ok_or_else(|| anyhow!("sum overflow"))?,
                            Value::Float(f) => {
                                any_float = true;
                                float += f;
                            }
                            v => return other(format!("sum: element of `{array}` is {}", v.kind())),
                        }
                    }
                    if any_float { Value::Float(float + int as f64) } else { Value::Int(int) }
                }
                _ => return other(format!("sum: parameter `{array}` is not an array")),
            },
            ExprNode::Size(h) => {
                let hv = self.handle(h)?;
                Value::Int(self.size_of(&hv)?)
            }
            ExprNode::Offset(h) => {
                let hv = self.handle(h)?;
                Value::Int(self.offset_of(&hv)?)
            }
            ExprNode::UnitIndex(h) => {
                let hv = self.handle(h)?;
                match hv {
                    HVal::Sample { ds, id } => Value::Int(self.model.datasets[ds].unit_of_sample(id)?),
                    HVal::Unit { u, .. } | HVal::Column { u, .. } => Value::Int(u),
                    _ => return other("unit_index: not a sample, unit, or column handle"),
                }
            }
            ExprNode::Units(h) => {
                let hv = self.handle(h)?;
                let (ds, file) = match hv {
                    HVal::Sample { ds, id } => (ds, self.container_of(ds, id)),
                    HVal::File { ds, file } | HVal::Unit { ds, file, .. } | HVal::Column { ds, file, .. } => (ds, file),
                    _ => return other("units: not a file, sample, or unit handle"),
                };
                Value::Int(self.model.datasets[ds].units_in_file(file)?)
            }
            ExprNode::Chunks(h) => {
                let hv = self.handle(h)?;
                let (ds, file) = match hv {
                    HVal::Sample { ds, id } => (ds, self.container_of(ds, id)),
                    HVal::File { ds, file } => (ds, file),
                    _ => return other("chunks: not a file or sample handle"),
                };
                Value::Int(self.model.datasets[ds].chunks(file)?)
            }
            ExprNode::Count(name) => Value::Int(self.model.datasets[self.model.dataset(name)?].count()),
            ExprNode::Dirs(name) => match &self.model.datasets[self.model.dataset(name)?] {
                DsMeta::Files { dirs: DirScheme::Unsupported, .. } => return other(format!("dirs: dataset `{name}` has several `id` fields in its directory part")),
                DsMeta::Files { ndirs, .. } => Value::Int(*ndirs),
                _ => return other(format!("dirs: `{name}` is not a files dataset")),
            },
            ExprNode::Add(a) => self.arith(a, |x, y| x.checked_add(y), |x, y| x + y, "add")?,
            ExprNode::Sub(a) => self.arith(a, |x, y| x.checked_sub(y), |x, y| x - y, "sub")?,
            ExprNode::Mul(a) => self.arith(a, |x, y| x.checked_mul(y), |x, y| x * y, "mul")?,
            ExprNode::Div(a) => self.arith(a, |x, y| if y == 0 { None } else { Some(x.div_euclid(y)) }, |x, y| (x / y).floor(), "div")?,
            ExprNode::Mod(a) => self.arith(a, |x, y| if y == 0 { None } else { Some(x.rem_euclid(y)) }, |x, y| x.rem_euclid(y), "mod")?,
            ExprNode::CeilDiv(a) => self.arith(a, |x, y| if y == 0 { None } else { Some(-((-x).div_euclid(y))) }, |x, y| (x / y).ceil(), "ceil_div")?,
            ExprNode::Min(a) => self.arith(a, |x, y| Some(x.min(y)), f64::min, "min")?,
            ExprNode::Max(a) => self.arith(a, |x, y| Some(x.max(y)), f64::max, "max")?,
            ExprNode::Neg(x) => match self.expr(x)? {
                Value::Int(n) => Value::Int(n.checked_neg().ok_or_else(|| anyhow!("neg overflow"))?),
                Value::Float(f) => Value::Float(-f),
                v => return other(format!("neg of {}", v.kind())),
            },
            ExprNode::Eq(a) => Value::Bool(self.compare(a, "eq")? == Some(std::cmp::Ordering::Equal)),
            ExprNode::Ne(a) => Value::Bool(self.compare(a, "ne")? != Some(std::cmp::Ordering::Equal)),
            ExprNode::Lt(a) => Value::Bool(self.compare(a, "lt")? == Some(std::cmp::Ordering::Less)),
            ExprNode::Le(a) => Value::Bool(matches!(self.compare(a, "le")?, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal))),
            ExprNode::Gt(a) => Value::Bool(self.compare(a, "gt")? == Some(std::cmp::Ordering::Greater)),
            ExprNode::Ge(a) => Value::Bool(matches!(self.compare(a, "ge")?, Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))),
            ExprNode::And(a) => {
                for x in a {
                    if !self.expr(x)?.as_bool()? {
                        return Ok(Value::Bool(false));
                    }
                }
                Value::Bool(true)
            }
            ExprNode::Or(a) => {
                for x in a {
                    if self.expr(x)?.as_bool()? {
                        return Ok(Value::Bool(true));
                    }
                }
                Value::Bool(false)
            }
            ExprNode::Not(x) => Value::Bool(!self.expr(x)?.as_bool()?),
            ExprNode::Cond(c) => {
                let test = self.expr(&c.test)?.as_bool()?;
                let (first, second) = if test { (&c.then, &c.otherwise) } else { (&c.otherwise, &c.then) };
                match self.exprlike(first) {
                    Err(EvalError::BelowFrom) => self.exprlike(second)?,
                    r => r?,
                }
            }
        })
    }

    fn arith(&mut self, a: &'a [Expr; 2], int: impl Fn(i64, i64) -> Option<i64>, float: impl Fn(f64, f64) -> f64, name: &str) -> ER<Value<'a>> {
        let x = self.expr(&a[0])?;
        let y = self.expr(&a[1])?;
        match (&x, &y) {
            (Value::Int(x), Value::Int(y)) => match int(*x, *y) {
                Some(v) => Ok(Value::Int(v)),
                None => other(format!("{name}: overflow or division by zero ({x}, {y})")),
            },
            (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => Ok(Value::Float(float(x.as_f64()?, y.as_f64()?))),
            _ => other(format!("{name}: operands are {} and {}", x.kind(), y.kind())),
        }
    }

    /// `None` when the values are of different kinds or one is `none` (then only `eq`/`ne` are
    /// meaningful and they compare as unequal).
    fn compare(&mut self, a: &'a [Expr; 2], name: &str) -> ER<Option<std::cmp::Ordering>> {
        let x = self.expr(&a[0])?;
        let y = self.expr(&a[1])?;
        Ok(match (&x, &y) {
            (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
            (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
            (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
            (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
            (Value::None, Value::None) => Some(std::cmp::Ordering::Equal),
            (Value::None, _) | (_, Value::None) => None,
            (Value::Handle(x), Value::Handle(y)) => {
                if x == y { Some(std::cmp::Ordering::Equal) } else { None }
            }
            _ => return other(format!("{name}: cannot compare {} with {}", x.kind(), y.kind())),
        })
    }

    fn exprlike(&mut self, v: &'a ExprLike) -> ER<Value<'a>> {
        match v {
            ExprLike::Expr(e) => self.expr(e),
            ExprLike::Dist(d) => Ok(Value::Dist(d)),
            ExprLike::Handle(h) => Ok(Value::Handle(self.handle(h)?)),
        }
    }

    fn param(&self, name: &str) -> ER<Value<'a>> {
        if name == "gpus" {
            return Ok(Value::Int(self.model.cfg.gpus));
        }
        let params: &'a Params = self.model.params;
        match params.values.get(name) {
            Some(v) => Ok(Value::from_pvalue(v)),
            None => other(format!("unknown param `{name}`")),
        }
    }

    fn index(&self, name: &str) -> ER<Value<'a>> {
        if let Some(fields) = &self.fields {
            if let Some(v) = fields.get(name) {
                return Ok(v.clone());
            }
        }
        for f in (0..self.depth).rev() {
            if self.frames[f].name == name {
                return Ok(Value::Int(self.frame_index(f)));
            }
        }
        other(format!("index `{name}` not in scope"))
    }

    fn find_let(body: &'a [Node], name: &str) -> Option<&'a ExprLike> {
        body.iter().find_map(|n| match n {
            Node::Let { name: n, value } if n == name => Some(value),
            _ => None,
        })
    }

    fn reference(&mut self, name: &str) -> ER<Value<'a>> {
        if let Some(fields) = &self.fields {
            if let Some(v) = fields.get(name) {
                return Ok(v.clone());
            }
        }
        // inside `x @ i`: sibling lets of x's body take their value at the shifted index
        if let Some(top) = self.shifts.last() {
            if let Some(v) = top.memo.get(name) {
                return Ok(v.clone());
            }
            let body = top.body;
            if let Some(def) = Self::find_let(body, name) {
                let v = self.exprlike(def)?;
                let key: &'a str = Self::let_name(body, name).expect("found above");
                self.shifts.last_mut().unwrap().memo.insert(key, v.clone());
                return Ok(v);
            }
        }
        for level in self.scopes.iter().rev() {
            if let Some(b) = level.bindings.get(name) {
                return Ok(b.value.clone());
            }
        }
        other(format!("binding `{name}` not in scope (a forward reference needs `at`)"))
    }

    fn let_name(body: &'a [Node], name: &str) -> Option<&'a str> {
        body.iter().find_map(|n| match n {
            Node::Let { name: n, .. } if n == name => Some(n.as_str()),
            _ => None,
        })
    }

    /// `x @ i`: the value `x` takes at index `i` of its loop.
    fn at(&mut self, name: &str, i: i64) -> ER<Value<'a>> {
        // where is x defined: the current shift's body, an enclosing scope, or the current body (forward)?
        let (frame, body, depth) = if let Some(top) = self.shifts.last().filter(|s| Self::find_let(s.body, name).is_some()) {
            (top.frame, top.body, top.depth)
        } else if let Some(b) = self.scopes.iter().rev().find_map(|l| l.bindings.get(name)) {
            if b.depth == 0 {
                return other(format!("`{name} @`: binding is outside any loop"));
            }
            (b.depth - 1, b.body, b.depth)
        } else if let Some(level) = self.scopes.iter().rev().find(|l| Self::find_let(l.body, name).is_some()) {
            if level.depth == 0 {
                return other(format!("`{name} @`: binding is outside any loop"));
            }
            (level.depth - 1, level.body, level.depth)
        } else {
            return other(format!("`at` names unknown binding `{name}`"));
        };
        if i < self.frames[frame].from {
            return Err(EvalError::BelowFrom);
        }
        let def = Self::find_let(body, name).expect("checked above");
        let saved_depth = self.depth;
        self.shifts.push(Shift { frame, idx: i, body, depth, memo: HashMap::new() });
        self.depth = depth;
        let r = self.exprlike(def);
        self.depth = saved_depth;
        self.shifts.pop();
        r
    }

    // ---- distributions ----

    fn distref(&mut self, d: &'a DistRef) -> ER<RDist<'a>> {
        match d {
            DistRef::Dist(d) => self.dist(d),
            DistRef::Expr(e) => match self.expr(e)? {
                Value::Dist(d) => {
                    if let Some(r) = self.model.param_dists.get(&(d as *const Dist as usize)) {
                        return Ok(r.clone());
                    }
                    self.dist(d)
                }
                v => Ok(RDist::Const(v)),
            },
        }
    }

    fn dist(&mut self, d: &'a Dist) -> ER<RDist<'a>> {
        Ok(match d {
            Dist::Const(e) => RDist::Const(self.expr(e)?),
            Dist::Uniform { lo, hi } => RDist::Uniform { lo: self.expr(lo)?.as_int()?, hi: self.expr(hi)?.as_int()? },
            Dist::Uniform64 {} => RDist::Uniform64,
            Dist::Normal { mean, sd, min, max } => RDist::Normal {
                mean: self.expr(mean)?.as_f64()?,
                sd: self.expr(sd)?.as_f64()?,
                min: self.opt_f64(min)?,
                max: self.opt_f64(max)?,
            },
            Dist::Lognormal { median, sigma, min, max } => RDist::Lognormal {
                median: self.expr(median)?.as_f64()?,
                sigma: *sigma,
                min: self.opt_f64(min)?,
                max: self.opt_f64(max)?,
            },
            Dist::Empirical { values, weights } => {
                let mut cum = Vec::with_capacity(weights.len());
                let mut acc = 0.0;
                for w in weights {
                    acc += w;
                    cum.push(acc);
                }
                RDist::Empirical { values: values.iter().map(Value::from_literal).collect(), cum }
            }
            Dist::Zipf { s } => RDist::Zipf { s: *s },
            Dist::Hotset { fraction, weight } => RDist::Hotset { fraction: *fraction, weight: *weight },
            Dist::Mixture(arms) => {
                let mut out = Vec::with_capacity(arms.len());
                let mut total = 0.0;
                for arm in arms {
                    total += arm.weight;
                    let d = match &arm.dist {
                        None => None,
                        Some(d) => Some(self.distref(d)?),
                    };
                    out.push((arm.weight, d));
                }
                RDist::Mixture { arms: out, total }
            }
        })
    }

    fn opt_f64(&mut self, e: &'a Option<Box<Expr>>) -> ER<Option<f64>> {
        match e {
            None => Ok(None),
            Some(e) => Ok(Some(self.expr(e)?.as_f64()?)),
        }
    }

    // ---- handles ----

    fn container_of(&self, ds: usize, id: i64) -> i64 {
        match &self.model.datasets[ds] {
            DsMeta::Files { spf, .. } => id / spf,
            DsMeta::Regions { .. } => 0,
        }
    }

    fn handle(&mut self, h: &'a Handle) -> ER<HVal> {
        Ok(match h {
            Handle::Ref(name) => match self.reference(name)? {
                Value::Handle(h) => h,
                v => return other(format!("`{name}` is a {}, not a handle", v.kind())),
            },
            Handle::File(f) => match (&f.dataset, &f.id, &f.of) {
                (Some(ds), Some(id), _) => {
                    let ds = self.model.dataset(ds)?;
                    let file = self.expr(id)?.as_int()?;
                    match &self.model.datasets[ds] {
                        DsMeta::Regions { .. } => {
                            if file != 0 {
                                return other("a regions dataset has one file, id 0");
                            }
                            HVal::RegionFile { ds }
                        }
                        DsMeta::Files { .. } => match &f.chunk {
                            Some(k) => HVal::Chunk { ds, file, k: self.expr(k)?.as_int()? },
                            None => HVal::File { ds, file },
                        },
                    }
                }
                (_, _, Some(of)) => {
                    let inner = self.handle(of)?;
                    let (ds, file) = match inner {
                        HVal::Sample { ds, id } => (ds, self.container_of(ds, id)),
                        HVal::File { ds, file } => (ds, file),
                        HVal::Region { ds, .. } | HVal::RegionFile { ds } => (ds, 0),
                        HVal::Chunk { ds, file, .. } | HVal::Unit { ds, file, .. } | HVal::Column { ds, file, .. } => (ds, file),
                        other_h => return other(format!("file {{of}}: {other_h:?} has no container")),
                    };
                    match &f.chunk {
                        Some(k) => HVal::Chunk { ds, file, k: self.expr(k)?.as_int()? },
                        None => match &self.model.datasets[ds] {
                            DsMeta::Regions { .. } => HVal::RegionFile { ds },
                            _ => HVal::File { ds, file },
                        },
                    }
                }
                _ => return other("file handle needs {dataset, id} or {of}"),
            },
            Handle::Dir { dataset, id } => {
                let ds = self.model.dataset(dataset)?;
                HVal::Dir { ds, d: self.expr(id)?.as_int()? }
            }
            Handle::Object { namespace, fields } => {
                let ns = self.model.namespace(namespace)?;
                let meta = &self.model.namespaces[ns];
                let mut vals: HashMap<&'a str, Value<'a>> = HashMap::with_capacity(fields.len());
                for (name, e) in fields {
                    let v = self.expr(e)?;
                    match (meta.fields.get(name), &v) {
                        (Some(FieldType::Int), Value::Int(_)) | (Some(FieldType::Str), Value::Str(_)) => {}
                        (Some(FieldType::Int), Value::Float(_)) => {}
                        (Some(t), _) => return other(format!("object field `{name}` of `{namespace}` is {:?}, got {}", t, v.kind())),
                        (None, _) => return other(format!("object field `{name}` is not a field of `{namespace}`")),
                    }
                    vals.insert(name.as_str(), v);
                }
                let path = meta.pattern.format(|f| match vals.get(f) {
                    Some(Value::Int(n)) => Some(FieldValue::Int(*n)),
                    Some(Value::Float(x)) => Some(FieldValue::Int(x.round() as i64)),
                    Some(Value::Str(s)) => Some(FieldValue::Str(s)),
                    _ => None,
                })?;
                let size = match meta.size {
                    None => None,
                    Some(e) => {
                        let saved = self.fields.replace(vals);
                        let r = self.expr(e);
                        self.fields = saved;
                        Some(r?.as_int()?)
                    }
                };
                HVal::Object { ns, path: Arc::from(path), size }
            }
            Handle::Unit(uh) => {
                let inner = self.handle(&uh.of)?;
                let index = match &uh.index {
                    Some(e) => Some(self.expr(e)?.as_int()?),
                    None => None,
                };
                match (inner, index) {
                    (HVal::Sample { ds, id }, None) => HVal::Unit { ds, file: self.container_of(ds, id), u: self.model.datasets[ds].unit_of_sample(id)? },
                    (HVal::Sample { ds, id }, Some(u)) => HVal::Unit { ds, file: self.container_of(ds, id), u },
                    (HVal::File { ds, file }, Some(u)) => HVal::Unit { ds, file, u },
                    (HVal::File { .. }, None) => return other("unit {of: file} needs an index"),
                    (h, _) => return other(format!("unit {{of}}: {h:?} is not a sample or file handle")),
                }
            }
            Handle::Column(ch) => {
                let inner = self.handle(&ch.of)?;
                let c = self.expr(&ch.index)?.as_int()?;
                match inner {
                    HVal::Unit { ds, file, u } => HVal::Column { ds, file, u, c },
                    HVal::Sample { ds, id } => HVal::Column { ds, file: self.container_of(ds, id), u: self.model.datasets[ds].unit_of_sample(id)?, c },
                    h => return other(format!("column {{of}}: {h:?} is not a unit or sample handle")),
                }
            }
            Handle::Consume(dataset) => {
                let ds = self.model.dataset(dataset)?;
                self.consume(ds)?
            }
            Handle::Pick(pick) => {
                let ds = self.model.dataset(&pick.dataset)?;
                let meta = &self.model.datasets[ds];
                let n = meta.draw_domain();
                if n == 0 {
                    return other(format!("pick from empty dataset `{}`", pick.dataset));
                }
                let mut w = Words::new(self.draw_key(pick.site.get()));
                let id = match &pick.dist {
                    None => w.next_range(0, n),
                    Some(d) => {
                        let rd = self.distref(d)?;
                        let perm = meta.rank_perm();
                        sample(&rd, &mut w, Some(&perm))?.as_int()?
                    }
                };
                if id < 0 || id >= n {
                    return other(format!("pick: id {id} outside dataset `{}` of {n}", pick.dataset));
                }
                match meta {
                    DsMeta::Files { access: Access::Stream, .. } => HVal::File { ds, file: id },
                    DsMeta::Files { .. } => HVal::Sample { ds, id },
                    DsMeta::Regions { .. } => HVal::Region { ds, c: id },
                }
            }
        })
    }

    /// `consume`: the next sample without replacement, by position (see the module doc).
    fn consume(&mut self, ds: usize) -> ER<HVal> {
        let depth = self.depth;
        if depth == 0 {
            return other("`consume` outside any loop");
        }
        let split = (0..depth).rev().find(|f| self.frames[*f].loader).unwrap_or(depth - 1);
        let mut b: i64 = 0;
        for f in 0..=split {
            b = b.checked_mul(self.frames[f].iters).and_then(|x| x.checked_add(self.frame_ordinal(f))).ok_or_else(|| anyhow!("position overflow"))?;
        }
        let mut j: i64 = 0;
        let mut batch: i64 = 1;
        for f in split + 1..depth {
            j = j.checked_mul(self.frames[f].iters).and_then(|x| x.checked_add(self.frame_ordinal(f))).ok_or_else(|| anyhow!("position overflow"))?;
            batch = batch.checked_mul(self.frames[f].iters).ok_or_else(|| anyhow!("batch size overflow"))?;
        }
        let meta = &self.model.datasets[ds];
        let n = meta.draw_domain();
        let g = self.actor_count;
        let per_epoch = g.checked_mul(batch).ok_or_else(|| anyhow!("gpus × batch overflow"))?;
        let epoch_len = n / per_epoch;
        if epoch_len == 0 {
            return other(format!("dataset `{}` has {n} samples, fewer than one batch across {g} actors ({batch} per batch)", meta.name()));
        }
        let epoch = b / epoch_len;
        let pos = self.actor + g * ((b % epoch_len) * batch + j);
        let perm = meta.consume_perm(epoch)?;
        let id = perm.apply(pos as u64) as i64;
        Ok(if meta.access() == Access::Stream { HVal::File { ds, file: id } } else { HVal::Sample { ds, id } })
    }

    /// The unit lengths of a container file, computed once and held (bounded cache).
    fn unit_lens(&mut self, ds: usize, file: i64) -> ER<Arc<Vec<i64>>> {
        if let Some(v) = self.unit_lens.get(&(ds, file)) {
            return Ok(v.clone());
        }
        let v = Arc::new(self.model.datasets[ds].unit_lens(file)?);
        if self.unit_lens.len() >= 64 {
            self.unit_lens.clear();
        }
        self.unit_lens.insert((ds, file), v.clone());
        Ok(v)
    }

    fn size_of(&mut self, h: &HVal) -> ER<i64> {
        let ds = &self.model.datasets;
        Ok(match h {
            HVal::Sample { ds: d, id } => ds[*d].sample_size(*id)?,
            HVal::File { ds: d, file } => ds[*d].file_size(*file)?,
            HVal::Chunk { ds: d, file, k } => ds[*d].chunk_size(*file, *k)?,
            HVal::Region { ds: d, c } => ds[*d].sample_size(*c)?,
            HVal::RegionFile { ds: d } => ds[*d].file_size(0)?,
            HVal::Unit { ds: d, file, u } => {
                let lens = self.unit_lens(*d, *file)?;
                match lens.get(*u as usize) {
                    Some(n) if *u >= 0 => *n,
                    _ => return other(format!("unit {u} outside file {file} ({} units)", lens.len())),
                }
            }
            HVal::Column { ds: d, file, u, c } => ds[*d].col_len(*file, *u, *c)?,
            HVal::Dir { .. } => return other("size of a directory"),
            HVal::Object { size: Some(s), .. } => *s,
            HVal::Object { path, .. } => match self.written.get(path) {
                Some(n) => *n,
                None => return other(format!("size of as_written object `{path}`: no writes recorded at this position")),
            },
        })
    }

    fn offset_of(&mut self, h: &HVal) -> ER<i64> {
        Ok(match h {
            HVal::Sample { ds, id } => self.model.datasets[*ds].sample_offset(*id)?,
            HVal::Region { ds, c } => self.model.datasets[*ds].sample_offset(*c)?,
            HVal::Chunk { .. } | HVal::File { .. } | HVal::RegionFile { .. } => 0,
            HVal::Unit { ds, file, u } => self.unit_offset(*ds, *file, *u)?,
            HVal::Column { ds, file, u, c } => self.unit_offset(*ds, *file, *u)? + self.model.datasets[*ds].col_offset_in_unit(*file, *u, *c)?,
            _ => return other(format!("offset of {h:?}")),
        })
    }

    fn unit_offset(&mut self, ds: usize, file: i64, u: i64) -> ER<i64> {
        let lens = self.unit_lens(ds, file)?;
        if u < 0 || u as usize >= lens.len() {
            return other(format!("unit {u} outside file {file} ({} units)", lens.len()));
        }
        Ok(self.model.datasets[ds].data_start(file)? + lens[..u as usize].iter().sum::<i64>())
    }

    fn target(&mut self, h: &'a Handle) -> ER<Target> {
        let hv = self.handle(h)?;
        let ds = &self.model.datasets;
        Ok(match &hv {
            HVal::Sample { ds: d, id } => {
                let file = self.container_of(*d, *id);
                Target { path: ds[*d].file_path(file, None)?, size: Some(ds[*d].file_size(file)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None }
            }
            HVal::File { ds: d, file } | HVal::Unit { ds: d, file, .. } | HVal::Column { ds: d, file, .. } => {
                Target { path: ds[*d].file_path(*file, None)?, size: Some(ds[*d].file_size(*file)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None }
            }
            HVal::Chunk { ds: d, file, k } => Target { path: ds[*d].file_path(*file, Some(*k))?, size: Some(ds[*d].chunk_size(*file, *k)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None },
            HVal::Region { ds: d, .. } | HVal::RegionFile { ds: d } => Target { path: ds[*d].file_path(0, None)?, size: Some(ds[*d].file_size(0)?), as_written: false, seed: ds[*d].dataset_seed(), entries: None },
            HVal::Dir { ds: d, d: dir } => Target { path: ds[*d].dir_path(*dir)?, size: None, as_written: false, seed: ds[*d].dataset_seed(), entries: ds[*d].dir_entries(*dir) },
            HVal::Object { ns, path, size } => Target { path: path.clone(), size: *size, as_written: size.is_none(), seed: self.model.namespaces[*ns].seed, entries: None },
        })
    }

    // ---- statements ----

    fn body(&mut self, nodes: &'a [Node]) -> ER<()> {
        self.scopes.push(ScopeLevel { bindings: HashMap::new(), body: nodes, depth: self.frames.len() });
        let r = self.body_inner(nodes);
        self.scopes.pop();
        r
    }

    fn body_inner(&mut self, nodes: &'a [Node]) -> ER<()> {
        for node in nodes {
            match node {
                Node::Let { name, value } => {
                    let v = self.exprlike(value)?;
                    let depth = self.frames.len();
                    let level = self.scopes.last_mut().unwrap();
                    level.bindings.insert(name.as_str(), Binding { value: v, depth, body: level.body });
                }
                Node::Loop { index, from, to, step, body } => {
                    let from = match from {
                        Some(e) => self.expr(e)?.as_int()?,
                        None => 0,
                    };
                    let to = self.expr(to)?.as_int()?;
                    let step = match step {
                        Some(e) => self.expr(e)?.as_int()?,
                        None => 1,
                    };
                    if step <= 0 {
                        return other(format!("loop `{index}`: step {step} must be positive"));
                    }
                    let iters = if to > from { (to - from + step - 1) / step } else { 0 };
                    self.iterate(index, from, step, iters, false, body)?;
                }
                Node::Parallel { index, width, body } => {
                    let width = self.expr(width)?.as_int()?;
                    if width < 0 {
                        return other(format!("parallel `{index}`: width {width}"));
                    }
                    self.fork(ForkKind::Parallel { index, width }, index, width, false, body)?;
                }
                Node::Loader { name, index, workers, prefetch, batches, ordered, body } => {
                    let workers = self.expr(workers)?.as_int()?;
                    let prefetch = self.expr(prefetch)?.as_int()?;
                    let batches = self.expr(batches)?.as_int()?;
                    if workers < 1 || prefetch < 0 || batches < 0 {
                        return other(format!("loader: workers {workers}, prefetch {prefetch}, batches {batches}"));
                    }
                    let kind = ForkKind::Loader { name, index, workers, prefetch, batches, ordered: ordered.unwrap_or(true) };
                    self.fork(kind, index, batches, true, body)?;
                }
                Node::Channel { name, capacity, ordered } => {
                    let capacity = self.expr(capacity)?.as_int()?;
                    self.ctl(Control::Channel { name, capacity, ordered: ordered.unwrap_or(true) })?;
                }
                Node::Put { channel, seq } => {
                    let seq = self.expr(seq)?.as_int()?;
                    self.ctl(Control::Put { channel, seq })?;
                }
                Node::Take { channel } => {
                    self.ctl(Control::Take { channel })?;
                }
                Node::Barrier { scope } => {
                    self.ctl(Control::Barrier { scope })?;
                }
                Node::Compute { ns } => {
                    let ns = self.expr(ns)?.as_int()?;
                    self.ctl(Control::Compute { ns })?;
                }
                Node::Cond { test, then, otherwise } => {
                    if self.expr(test)?.as_bool()? {
                        self.body(then)?;
                    } else if let Some(b) = otherwise {
                        self.body(b)?;
                    }
                }
                Node::Choose { arms, site } => {
                    let total: f64 = arms.iter().map(|a| a.weight).sum();
                    if total <= 0.0 {
                        return other("choose: total weight is zero");
                    }
                    let u = Words::new(self.draw_key(site.get())).next_f64() * total;
                    let mut acc = 0.0;
                    let mut chosen = arms.len() - 1;
                    for (i, arm) in arms.iter().enumerate() {
                        acc += arm.weight;
                        if u < acc {
                            chosen = i;
                            break;
                        }
                    }
                    self.body(&arms[chosen].body)?;
                }
                Node::Phase { name, body } => {
                    let label: Arc<str> = match self.expr(name)? {
                        Value::Str(s) => s,
                        Value::Int(n) => Arc::from(n.to_string()),
                        v => return other(format!("phase name is {}", v.kind())),
                    };
                    self.phases.push(label);
                    let r = self.body(body);
                    self.phases.pop();
                    r?;
                }
                Node::Replay { trace, .. } => return other(format!("replay `{trace}`: the trace format is deferred (schema §7)")),
                _ => self.op(node)?,
            }
        }
        Ok(())
    }

    /// A `parallel` or `loader`: offer the sub-actors to the sink; walk them inline if it declines.
    fn fork(&mut self, kind: ForkKind<'a>, index: &'a str, width: i64, loader: bool, body: &'a [Node]) -> ER<()> {
        let frame = Frame { name: index, idx: 0, from: 0, step: 1, iters: width, loader };
        let forked = {
            let (model, template, actor, actor_count) = (self.model, self.template, self.actor, self.actor_count);
            let (frames, scopes, phases, open, written) = (&self.frames, &self.scopes, &self.phases, &self.open, &self.written);
            let snapshot = || {
                let mut fr = frames.clone();
                fr.push(frame);
                Self::snapshot(model, template, actor, actor_count, &fr, scopes, phases, open, written, body)
            };
            self.sink.fork(&kind, &snapshot).map_err(EvalError::Other)?
        };
        if !forked {
            self.iterate(index, 0, 1, width, loader, body)?;
        }
        Ok(())
    }

    fn iterate(&mut self, index: &'a str, from: i64, step: i64, iters: i64, loader: bool, body: &'a [Node]) -> ER<()> {
        self.frames.push(Frame { name: index, idx: from, from, step, iters, loader });
        self.depth = self.frames.len();
        let mut r = Ok(());
        for k in 0..iters {
            self.frames.last_mut().unwrap().idx = from + k * step;
            r = self.body(body);
            if r.is_err() {
                break;
            }
        }
        self.frames.pop();
        self.depth = self.frames.len();
        r
    }

    fn ctl(&mut self, c: Control<'a>) -> ER<()> {
        self.indices_buf.clear();
        for f in 0..self.frames.len() {
            self.indices_buf.push(self.frames[f].idx);
        }
        let ctx = OpCtx { template: self.template, actor: self.actor, indices: &self.indices_buf, phase: self.phases.last().map(|p| &**p) };
        self.sink.control(c, &ctx).map_err(EvalError::Other)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, kind: OpKind, t: &Target, path2: Option<&Arc<str>>, offset: i64, len: i64, bytes: i64, aux: u64, expect: &Expect, positioned: bool) -> ER<()> {
        self.indices_buf.clear();
        for f in 0..self.frames.len() {
            self.indices_buf.push(self.frames[f].idx);
        }
        let ctx = OpCtx { template: self.template, actor: self.actor, indices: &self.indices_buf, phase: self.phases.last().map(|p| &**p) };
        let op = Op { kind, path: &t.path, path2: path2.map(|p| &**p), offset, len, bytes, aux, expect: expect.as_deref().unwrap_or(&[]), positioned, seed: t.seed };
        self.sink.op(&op, &ctx).map_err(EvalError::Other)
    }

    fn opened(&self, path: &Arc<str>, what: &str) -> ER<FileState> {
        match self.open.get(path) {
            Some(s) => Ok(s.clone()),
            None => other(format!("{what} on `{path}`, which this actor has not opened")),
        }
    }

    fn op(&mut self, node: &'a Node) -> ER<()> {
        match node {
            Node::Open { file, flags, mode, expect } => {
                let t = self.target(file)?;
                let append = flags.contains(&OpenFlag::APPEND);
                let trunc = flags.contains(&OpenFlag::TRUNC);
                if trunc && t.as_written {
                    self.written.insert(t.path.clone(), 0);
                }
                let size = if t.as_written { self.written.get(&t.path).copied() } else { t.size };
                let pos = if append { size.unwrap_or(0) } else { 0 };
                self.open.insert(t.path.clone(), FileState { pos, size, as_written: t.as_written });
                let aux = flag_bits(flags) | ((mode.unwrap_or(0) as u64) << 32);
                self.emit(OpKind::Open, &t, None, 0, 0, 0, aux, expect, false)?;
            }
            Node::Close(f) => {
                let t = self.target(&f.file)?;
                self.opened(&t.path, "close")?;
                self.open.remove(&t.path);
                self.emit(OpKind::Close, &t, None, 0, 0, 0, 0, &f.expect, false)?;
            }
            Node::Read { file, len, offset, repeat, expect } => {
                let t = self.target(file)?;
                let len = self.expr(len)?.as_int()?;
                if len < 0 {
                    return other(format!("read: negative length {len}"));
                }
                let st = self.opened(&t.path, "read")?;
                let size = if st.as_written { self.written.get(&t.path).copied() } else { st.size };
                let positioned = match offset {
                    Some(e) => Some(self.expr(e)?.as_int()?),
                    None => None,
                };
                let mut off = positioned.unwrap_or(st.pos);
                match repeat {
                    None => {
                        let bytes = expected(size, off, len);
                        self.emit(OpKind::Read, &t, None, off, len, bytes, 0, expect, positioned.is_some())?;
                        off += bytes;
                    }
                    Some(Repeat::Count(n)) => {
                        let n = self.expr(n)?.as_int()?;
                        for _ in 0..n {
                            let bytes = expected(size, off, len);
                            self.emit(OpKind::Read, &t, None, off, len, bytes, 0, expect, positioned.is_some())?;
                            off += bytes;
                        }
                    }
                    Some(Repeat::UntilEof(_)) => {
                        let Some(size) = size else {
                            return other(format!("until_eof on `{}`: size unknown (an as_written object needs the creating writes at this position)", t.path));
                        };
                        if len == 0 {
                            return other("until_eof with a zero-length read");
                        }
                        while off < size {
                            let bytes = (size - off).min(len);
                            self.emit(OpKind::Read, &t, None, off, len, bytes, 0, expect, positioned.is_some())?;
                            off += bytes;
                        }
                        self.emit(OpKind::Read, &t, None, off, len, 0, 0, expect, positioned.is_some())?;
                    }
                }
                if positioned.is_none() {
                    self.open.get_mut(&t.path).unwrap().pos = off;
                }
            }
            Node::Write { file, len, offset, repeat, expect } => {
                let t = self.target(file)?;
                let len = self.expr(len)?.as_int()?;
                if len < 0 {
                    return other(format!("write: negative length {len}"));
                }
                let st = self.opened(&t.path, "write")?;
                let positioned = match offset {
                    Some(e) => Some(self.expr(e)?.as_int()?),
                    None => None,
                };
                let n = match repeat {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 1,
                };
                let mut off = positioned.unwrap_or(st.pos);
                for _ in 0..n {
                    self.emit(OpKind::Write, &t, None, off, len, len, 0, expect, positioned.is_some())?;
                    off += len;
                    if st.as_written {
                        *self.written.entry(t.path.clone()).or_insert(0) += len;
                    }
                }
                if positioned.is_none() {
                    self.open.get_mut(&t.path).unwrap().pos = off;
                }
            }
            Node::Lseek { file, offset, whence } => {
                let t = self.target(file)?;
                let off = self.expr(offset)?.as_int()?;
                let st = self.opened(&t.path, "lseek")?;
                let size = if st.as_written { self.written.get(&t.path).copied() } else { st.size };
                let pos = match whence {
                    Whence::SET => off,
                    Whence::CUR => st.pos + off,
                    Whence::END => match size {
                        Some(s) => s + off,
                        None => return other(format!("lseek END on `{}`: size unknown", t.path)),
                    },
                };
                if pos < 0 {
                    return other(format!("lseek to {pos} on `{}`", t.path));
                }
                self.open.get_mut(&t.path).unwrap().pos = pos;
                self.emit(OpKind::Lseek, &t, None, off, 0, 0, *whence as u64, &None, false)?;
            }
            Node::Ioctl { file, request, expect } => {
                let t = self.target(file)?;
                self.opened(&t.path, "ioctl")?;
                self.emit(OpKind::Ioctl, &t, None, 0, 0, 0, *request as u64, expect, false)?;
            }
            Node::Fadvise { file, advice, offset, len, expect } => {
                let t = self.target(file)?;
                let off = match offset {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                let len = match len {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                if off < 0 || len < 0 {
                    return other(format!("fadvise: offset {off}, len {len}"));
                }
                self.opened(&t.path, "fadvise")?;
                self.emit(OpKind::Fadvise, &t, None, off, len, 0, *advice as u64, expect, false)?;
            }
            Node::Fstat(f) | Node::Fsync(f) | Node::Fdatasync(f) => {
                let t = self.target(&f.file)?;
                let kind = match node {
                    Node::Fstat(_) => OpKind::Fstat,
                    Node::Fsync(_) => OpKind::Fsync,
                    _ => OpKind::Fdatasync,
                };
                self.opened(&t.path, kind.name())?;
                self.emit(kind, &t, None, 0, 0, 0, 0, &f.expect, false)?;
            }
            Node::Stat(f) => {
                let t = self.target(&f.file)?;
                self.emit(OpKind::Stat, &t, None, 0, 0, 0, 0, &f.expect, false)?;
            }
            Node::Unlink(f) => {
                let t = self.target(&f.file)?;
                self.written.remove(&t.path);
                self.emit(OpKind::Unlink, &t, None, 0, 0, 0, 0, &f.expect, false)?;
            }
            Node::Ftruncate { file, len, expect } => {
                let t = self.target(file)?;
                let len = self.expr(len)?.as_int()?;
                self.opened(&t.path, "ftruncate")?;
                self.emit(OpKind::Ftruncate, &t, None, 0, len, 0, 0, expect, false)?;
            }
            Node::Fallocate { file, offset, len, expect } => {
                let t = self.target(file)?;
                let off = match offset {
                    Some(e) => self.expr(e)?.as_int()?,
                    None => 0,
                };
                let len = self.expr(len)?.as_int()?;
                self.opened(&t.path, "fallocate")?;
                self.emit(OpKind::Fallocate, &t, None, off, len, 0, 0, expect, false)?;
            }
            Node::Mkdir { dir, mode, expect } => {
                let t = self.target(dir)?;
                self.emit(OpKind::Mkdir, &t, None, 0, 0, 0, mode.unwrap_or(0o777) as u64, expect, false)?;
            }
            Node::Rmdir { dir, expect } => {
                let t = self.target(dir)?;
                self.emit(OpKind::Rmdir, &t, None, 0, 0, 0, 0, expect, false)?;
            }
            Node::Rename { from, to, expect } => {
                let a = self.target(from)?;
                let b = self.target(to)?;
                if let Some(n) = self.written.remove(&a.path) {
                    self.written.insert(b.path.clone(), n);
                }
                self.emit(OpKind::Rename, &a, Some(&b.path), 0, 0, 0, 0, expect, false)?;
            }
            Node::Readdir { dir, expect, .. } => {
                let t = self.target(dir)?;
                self.opened(&t.path, "readdir")?;
                let entries = t.entries.unwrap_or(-1);
                self.emit(OpKind::Readdir, &t, None, 0, 0, entries, 0, expect, false)?;
            }
            _ => unreachable!("control node in op()"),
        }
        Ok(())
    }
}

/// Bytes a read of `len` at `off` returns against a file of known `size`.
fn expected(size: Option<i64>, off: i64, len: i64) -> i64 {
    match size {
        Some(s) => (s - off).clamp(0, len),
        None => len,
    }
}

/// Run every instance of every actor template through `make_sink`, in one thread.
pub fn run_actor<'m, 'a: 'm, S: Sink<'m, 'a>>(model: &'m Model<'a>, template: &'a str, actor: i64, count: i64, sink: S) -> Result<S> {
    let a = model.ast.actors.get(template).ok_or_else(|| anyhow!("no actor `{template}`"))?;
    let mut vm = Vm::new(model, sink, template, actor, count);
    vm.run(&a.body).with_context(|| format!("actor `{template}` instance {actor}"))?;
    Ok(vm.into_sink())
}

/// Instance counts of every actor template (`count` defaults to `gpus`).
pub fn actor_counts<'a>(model: &'a Model<'a>) -> Result<Vec<(&'a str, i64)>> {
    let mut out = Vec::new();
    for (name, a) in &model.ast.actors {
        let count = match &a.count {
            None => model.cfg.gpus,
            Some(e) => {
                let mut vm = Vm::new(model, NullSink, name, 0, model.cfg.gpus.max(1));
                vm.eval_int(e).with_context(|| format!("actor `{name}` count"))?
            }
        };
        if count < 0 {
            bail!("actor `{name}`: count {count}");
        }
        out.push((name.as_str(), count));
    }
    Ok(out)
}
