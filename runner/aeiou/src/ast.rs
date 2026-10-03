//! The AST contract (`schema/abstract-ast.schema.json`, v0.5) as Rust types.
//!
//! Every node, expression, distribution, and handle is externally tagged: a JSON object with
//! exactly one key naming its kind. `deny_unknown_fields` on every struct and serde's enum
//! decoding give the structural checks of the schema; the semantic rules (`schema/README.md`
//! §4) live in `validate.rs`. The types carry nothing but what the JSON carries, plus the
//! `site` of every positional draw, filled by `sites::annotate` after loading.

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::BTreeMap;

use serde::Deserialize;

pub const AST_VERSION: &str = "0.5";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ast {
    pub ast: String,
    pub name: String,
    #[serde(default)]
    pub doc: Option<String>,
    /// The API the application this abstract was traced from issues its I/O through, as an
    /// `--io-backend` name (v0.3); absent: `sync`. A run uses it unless told otherwise.
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub params: BTreeMap<String, Param>,
    #[serde(default)]
    pub datasets: BTreeMap<String, Dataset>,
    #[serde(default)]
    pub namespaces: BTreeMap<String, Namespace>,
    pub actors: BTreeMap<String, Actor>,
    #[serde(default)]
    pub provenance: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Param {
    pub default: PValue,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(default)]
    pub cli: Option<bool>,
    #[serde(default)]
    pub unit: Option<String>,
}

/// A parameter value: a scalar, a distribution, or an array of these.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PValue {
    Dist(Dist),
    Array(Vec<PValue>),
    Scalar(Literal),
}

/// A JSON scalar; `null` is the `none` outcome of a mixture.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

// ---------------------------------------------------------------- distributions

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Dist {
    Const(Box<Expr>),
    Uniform { lo: Box<Expr>, hi: Box<Expr> },
    Uniform64 {},
    Normal { mean: Box<Expr>, sd: Box<Expr>, #[serde(default)] min: Option<Box<Expr>>, #[serde(default)] max: Option<Box<Expr>> },
    Lognormal { median: Box<Expr>, sigma: f64, #[serde(default)] min: Option<Box<Expr>>, #[serde(default)] max: Option<Box<Expr>> },
    Empirical { values: Vec<Literal>, weights: Vec<f64> },
    Zipf { s: f64 },
    Hotset { fraction: f64, weight: f64 },
    Mixture(Vec<MixtureArm>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixtureArm {
    pub weight: f64,
    pub dist: Option<DistRef>,
}

/// A distribution, or an expression that evaluates to one (a parameter, or a cond).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DistRef {
    Dist(Dist),
    Expr(Expr),
}

// ---------------------------------------------------------------- expressions

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Expr {
    Lit(Literal),
    Node(Box<ExprNode>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActorAttr {
    Id,
    Count,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ExprNode {
    Param(String),
    Index(String),
    Ref(String),
    At { #[serde(rename = "ref")] name: String, index: Expr },
    Actor(ActorAttr),
    Draw(Draw),
    Elem { array: String, index: Expr },
    Len(String),
    Sum(String),
    Size(Handle),
    Offset(Handle),
    /// Index of the unit holding a sample, within its file.
    UnitIndex(Handle),
    /// Number of units in a file.
    Units(Handle),
    Chunks(Handle),
    Count(String),
    Dirs(String),
    Add([Expr; 2]),
    Sub([Expr; 2]),
    Mul([Expr; 2]),
    Div([Expr; 2]),
    Mod([Expr; 2]),
    CeilDiv([Expr; 2]),
    Min([Expr; 2]),
    Max([Expr; 2]),
    Neg(Expr),
    Eq([Expr; 2]),
    Ne([Expr; 2]),
    Lt([Expr; 2]),
    Le([Expr; 2]),
    Gt([Expr; 2]),
    Ge([Expr; 2]),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Expr),
    Cond(CondExpr),
}

/// A positional draw. `site` is the hash of the node's JSON pointer, filled after loading.
#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct Draw {
    pub dist: DistRef,
    #[serde(skip)]
    pub site: Site,
}

/// A site hash slot: written once by `sites::annotate`, read on every draw.
#[derive(Debug, Default)]
pub struct Site(AtomicU64);

impl Site {
    pub fn set(&self, v: u64) {
        self.0.store(v, Ordering::Relaxed);
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

impl Clone for Site {
    fn clone(&self) -> Self {
        Site(AtomicU64::new(self.get()))
    }
}

impl Clone for Draw {
    fn clone(&self) -> Self {
        Draw { dist: self.dist.clone(), site: self.site.clone() }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CondExpr {
    #[serde(rename = "if")]
    pub test: Expr,
    pub then: ExprLike,
    #[serde(rename = "else")]
    pub otherwise: ExprLike,
}

/// Where an expression, a distribution, or a handle may appear (`let` values, `cond` arms).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ExprLike {
    Expr(Expr),
    Dist(Dist),
    Handle(Handle),
}

// ---------------------------------------------------------------- handles

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Handle {
    Ref(String),
    File(FileHandle),
    Dir { dataset: String, id: Expr },
    Object { namespace: String, fields: BTreeMap<String, Expr> },
    Consume(String),
    Pick(Pick),
    /// A container unit (row group, record batch, chunk) of a file: `index` of the file
    /// `of` holds or is; without `index`, the unit holding the sample `of`.
    Unit(UnitHandle),
    /// Column chunk `index` of a unit.
    Column(ColumnHandle),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitHandle {
    pub of: Box<Handle>,
    #[serde(default)]
    pub index: Option<Expr>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnHandle {
    pub of: Box<Handle>,
    pub index: Expr,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileHandle {
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub id: Option<Expr>,
    #[serde(default)]
    pub of: Option<Box<Handle>>,
    #[serde(default)]
    pub chunk: Option<Expr>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pick {
    pub dataset: String,
    #[serde(default)]
    pub dist: Option<DistRef>,
    #[serde(skip)]
    pub site: Site,
}

// ---------------------------------------------------------------- datasets, namespaces, actors

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    Map,
    Stream,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Dataset {
    Files(FilesDataset),
    Regions(RegionsDataset),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilesDataset {
    pub pattern: String,
    pub count: Expr,
    pub size: DistRef,
    pub seed: u64,
    #[serde(default)]
    pub access: Option<Access>,
    #[serde(default)]
    pub samples_per_file: Option<Expr>,
    #[serde(default)]
    pub chunk: Option<Expr>,
    #[serde(default)]
    pub format: Option<Format>,
    #[serde(default)]
    pub doc: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Format {
    pub class: String,
    #[serde(default)]
    pub reader: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub layout: Option<LayoutSpec>,
}

/// The container layout a format class declares (`schema/README.md` §2, v0.2): how a file's
/// bytes are framed around its samples. Every field is an expression over parameters; absent
/// means 0 (alignments: 1). The resolved form and the formulas are `eval::Layout`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutSpec {
    /// Samples per unit (default: `samples_per_file`, one unit per file).
    #[serde(default)]
    pub unit: Option<Expr>,
    #[serde(default)]
    pub file_header: Option<Expr>,
    #[serde(default)]
    pub file_header_per_sample: Option<Expr>,
    #[serde(default)]
    pub file_header_per_unit: Option<Expr>,
    #[serde(default)]
    pub file_footer: Option<Expr>,
    #[serde(default)]
    pub file_footer_per_sample: Option<Expr>,
    #[serde(default)]
    pub file_footer_per_unit: Option<Expr>,
    #[serde(default)]
    pub file_align: Option<Expr>,
    #[serde(default)]
    pub unit_header: Option<Expr>,
    #[serde(default)]
    pub unit_header_per_sample: Option<Expr>,
    #[serde(default)]
    pub unit_footer: Option<Expr>,
    #[serde(default)]
    pub unit_footer_per_sample: Option<Expr>,
    #[serde(default)]
    pub unit_align: Option<Expr>,
    /// The column chunks of a unit (default: one column carrying every sample whole).
    #[serde(default)]
    pub columns: Option<Vec<ColumnSpec>>,
    /// Class-specific writer settings: opaque to the runner, part of the resolved definition.
    #[serde(default)]
    pub writer: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnSpec {
    #[serde(default)]
    pub header: Option<Expr>,
    /// Bytes per row that do not come from the sample (an int64 label, an offset entry).
    #[serde(default)]
    pub fixed: Option<Expr>,
    /// Share of each sample's bytes this column carries; the last column with a positive
    /// weight takes the remainder. Weights sum to 1.
    #[serde(default)]
    pub weight: f64,
    #[serde(default)]
    pub row_header: Option<Expr>,
    #[serde(default)]
    pub row_footer: Option<Expr>,
    #[serde(default)]
    pub row_align: Option<Expr>,
    #[serde(default)]
    pub align: Option<Expr>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionsDataset {
    pub file: String,
    pub count: Expr,
    pub slot: Expr,
    pub size: DistRef,
    pub seed: u64,
    #[serde(default)]
    pub doc: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Int,
    Str,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum NsSize {
    AsWritten(AsWrittenTag),
    Expr(Expr),
}

#[derive(Debug, Deserialize)]
pub enum AsWrittenTag {
    #[serde(rename = "as_written")]
    AsWritten,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Namespace {
    pub pattern: String,
    pub fields: BTreeMap<String, FieldType>,
    pub size: NsSize,
    pub seed: u64,
    /// Written by a previous run; this abstract only reads it (V14).
    #[serde(default)]
    pub input: Option<bool>,
    /// The names read are positional draws of the writing run: this run must have the
    /// writer's seed, instance count, and common parameters (V15, v0.4).
    #[serde(default)]
    pub same_run: Option<bool>,
    #[serde(default)]
    pub doc: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Actor {
    #[serde(default)]
    pub count: Option<Expr>,
    pub body: Body,
    #[serde(default)]
    pub doc: Option<String>,
}

pub type Body = Vec<Node>;

// ---------------------------------------------------------------- statements

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
pub enum OpenFlag {
    RDONLY, WRONLY, RDWR, CREAT, TRUNC, EXCL, APPEND, CLOEXEC, DIRECTORY, DIRECT, SYNC, DSYNC, NOATIME, NOFOLLOW,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Whence {
    SET, CUR, END,
}

impl Whence {
    /// Back from an op's `aux` (`*whence as u64`).
    pub fn from_code(code: u64) -> Whence {
        match code {
            1 => Whence::CUR,
            2 => Whence::END,
            _ => Whence::SET,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum IoctlRequest {
    TCGETS, FIONREAD, BLKGETSIZE64,
}

impl IoctlRequest {
    pub fn from_code(code: u64) -> IoctlRequest {
        match code {
            1 => IoctlRequest::FIONREAD,
            2 => IoctlRequest::BLKGETSIZE64,
            _ => IoctlRequest::TCGETS,
        }
    }
}

/// `posix_fadvise(2)` advice, numbered as Linux numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Advice {
    NORMAL = 0,
    RANDOM = 1,
    SEQUENTIAL = 2,
    WILLNEED = 3,
    DONTNEED = 4,
    NOREUSE = 5,
}

impl Advice {
    pub fn from_code(code: u64) -> Advice {
        match code {
            1 => Advice::RANDOM,
            2 => Advice::SEQUENTIAL,
            3 => Advice::WILLNEED,
            4 => Advice::DONTNEED,
            5 => Advice::NOREUSE,
            _ => Advice::NORMAL,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Repeat {
    UntilEof(UntilEofTag),
    Count(Expr),
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum UntilEofTag {
    #[serde(rename = "until_eof")]
    UntilEof,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum UntilEndTag {
    #[serde(rename = "until_end")]
    UntilEnd,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum BarrierScope {
    Named(String),
}

pub type Expect = Option<Vec<String>>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileOp {
    pub file: Handle,
    #[serde(default)]
    pub expect: Expect,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChooseArm {
    pub weight: f64,
    pub body: Body,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Node {
    Let { name: String, value: ExprLike },
    Loop { index: String, #[serde(default)] from: Option<Expr>, to: Expr, #[serde(default)] step: Option<Expr>, body: Body },
    Parallel { index: String, width: Expr, body: Body },
    Channel { name: String, capacity: Expr, #[serde(default)] ordered: Option<bool> },
    Put { channel: String, seq: Expr },
    Take { channel: String },
    Loader { name: String, index: String, workers: Expr, prefetch: Expr, batches: Expr, #[serde(default)] ordered: Option<bool>, body: Body },
    Barrier { scope: String },
    Compute { ns: Expr },
    Cond { #[serde(rename = "if")] test: Expr, then: Body, #[serde(default, rename = "else")] otherwise: Option<Body> },
    Choose { arms: Vec<ChooseArm>, #[serde(skip)] site: Site },
    Phase { name: Expr, body: Body },
    Trace { file: String, sha256: String },

    Open { file: Handle, flags: Vec<OpenFlag>, #[serde(default)] mode: Option<u32>, #[serde(default)] expect: Expect },
    Close(FileOp),
    Read { file: Handle, len: Expr, #[serde(default)] offset: Option<Expr>, #[serde(default)] repeat: Option<Repeat>, #[serde(default)] expect: Expect },
    Write { file: Handle, len: Expr, #[serde(default)] offset: Option<Expr>, #[serde(default)] repeat: Option<Expr>, #[serde(default)] expect: Expect },
    Lseek { file: Handle, offset: Expr, whence: Whence },
    Ioctl { file: Handle, request: IoctlRequest, #[serde(default)] expect: Expect },
    Fadvise { file: Handle, advice: Advice, #[serde(default)] offset: Option<Expr>, #[serde(default)] len: Option<Expr>, #[serde(default)] expect: Expect },
    Fstat(FileOp),
    Stat(FileOp),
    Fsync(FileOp),
    Fdatasync(FileOp),
    Unlink(FileOp),
    Ftruncate { file: Handle, len: Expr, #[serde(default)] expect: Expect },
    Fallocate { file: Handle, #[serde(default)] offset: Option<Expr>, len: Expr, #[serde(default)] expect: Expect },
    Mkdir { dir: Handle, #[serde(default)] mode: Option<u32>, #[serde(default)] expect: Expect },
    Rmdir { dir: Handle, #[serde(default)] expect: Expect },
    Rename { from: Handle, to: Handle, #[serde(default)] expect: Expect },
    Readdir { dir: Handle, repeat: UntilEndTag, #[serde(default)] expect: Expect },
}

impl Node {
    /// The JSON key of this statement, as the schema names it.
    pub fn kind(&self) -> &'static str {
        match self {
            Node::Let { .. } => "let",
            Node::Loop { .. } => "loop",
            Node::Parallel { .. } => "parallel",
            Node::Channel { .. } => "channel",
            Node::Put { .. } => "put",
            Node::Take { .. } => "take",
            Node::Loader { .. } => "loader",
            Node::Barrier { .. } => "barrier",
            Node::Compute { .. } => "compute",
            Node::Cond { .. } => "cond",
            Node::Choose { .. } => "choose",
            Node::Phase { .. } => "phase",
            Node::Trace { .. } => "trace",
            Node::Open { .. } => "open",
            Node::Close(_) => "close",
            Node::Read { .. } => "read",
            Node::Write { .. } => "write",
            Node::Lseek { .. } => "lseek",
            Node::Ioctl { .. } => "ioctl",
            Node::Fadvise { .. } => "fadvise",
            Node::Fstat(_) => "fstat",
            Node::Stat(_) => "stat",
            Node::Fsync(_) => "fsync",
            Node::Fdatasync(_) => "fdatasync",
            Node::Unlink(_) => "unlink",
            Node::Ftruncate { .. } => "ftruncate",
            Node::Fallocate { .. } => "fallocate",
            Node::Mkdir { .. } => "mkdir",
            Node::Rmdir { .. } => "rmdir",
            Node::Rename { .. } => "rename",
            Node::Readdir { .. } => "readdir",
        }
    }

    pub fn is_control(&self) -> bool {
        matches!(
            self,
            Node::Let { .. } | Node::Loop { .. } | Node::Parallel { .. } | Node::Channel { .. } | Node::Put { .. }
                | Node::Take { .. } | Node::Loader { .. } | Node::Barrier { .. } | Node::Compute { .. }
                | Node::Cond { .. } | Node::Choose { .. } | Node::Phase { .. } | Node::Trace { .. }
        )
    }
}

impl Expr {
    pub fn lit_int(&self) -> Option<i64> {
        match self {
            Expr::Lit(Literal::Int(n)) => Some(*n),
            _ => None,
        }
    }
}

/// Load an AST from JSON text: structural checks only. Semantic rules are `validate::check`.
pub fn parse(text: &str) -> anyhow::Result<Ast> {
    let ast: Ast = serde_json::from_str(text)?;
    if ast.ast != AST_VERSION {
        anyhow::bail!("ast version `{}` is not `{}`", ast.ast, AST_VERSION);
    }
    Ok(ast)
}
