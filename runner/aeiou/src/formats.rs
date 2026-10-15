//! The container writers of `aeiou datagen` (cargo feature `formats`, `DESIGN_REVIEW.md` §3.70,
//! revised 2026-10-11): a format-class dataset's files written from the same sample sizes,
//! payload, and layout the runner computes, byte for byte the files the Python writer
//! (`builder/aeiou/formats.py`) makes, which the builder's tests compare.
//!
//! A file is a byte stream (`Stream`) pulled in pieces of any size, so it goes to a file in
//! `O_DIRECT` blocks or into an upload in parts with nothing held but a piece: framing bytes
//! are made as they are reached, and the payload of each sample is the file's payload stream
//! (`payload.rs`, the samples' bytes back to back from logical offset 0) filled in place.
//!
//! - **TFRecord**: records back to back, each `u64 length ‖ masked crc32c(length) ‖ data ‖
//!   masked crc32c(data)`, the CRC taken as the data streams past.
//! - **WebDataset**: a POSIX (ustar) tar of `{id:09}.{ext}` per member per sample, as CPython's
//!   `tarfile` writes it: 512-byte headers with mode 0644, owner 0, mtime 0, no device numbers;
//!   data padded to 512; two zero blocks; the archive padded to 10240.
//!
//! - **HDF5**: the header from `aeiou-header` (the format's own libhdf5, a tool beside this
//!   binary that rank 0 alone runs, §3.70 revised 2026-10-12), then the rows, contiguous.
//!
//! - **Parquet**: our own writer (§3.70, revised 2026-10-12, in place of parquet-rs): `PAR1`, row
//!   groups of one PLAIN, uncompressed, statistics-free page per column chunk, the thrift
//!   (compact protocol) footer padded to the layout's size, its length, `PAR1`. Byte for byte
//!   what pyarrow writes with the class's options but for `created_by`, which names this writer
//!   (the pad takes up the difference, so every offset is the same). The footer is a function
//!   of the sizes alone, made at the end from the chunks as written.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::eval::DsMeta;
use crate::payload::Filler;

/// A tar block, and the record an archive is padded to (`tarfile.RECORDSIZE`).
const TAR_BLOCK: u64 = 512;
const TAR_RECORD: u64 = 20 * TAR_BLOCK;

/// The class of a container dataset this build writes.
#[derive(Debug, Clone)]
pub enum Writer {
    TfRecord,
    /// `(extension, bytes)` per member; `None` is the sample itself, every other member zeros.
    WebDataset { members: Vec<(String, Option<u64>)> },
    /// The header of a file of so many rows (`set_headers`), and the library that made it.
    Hdf5 { headers: BTreeMap<i64, Vec<u8>>, version: String },
    /// `(name, type)` per column, exactly one `Binary` (the sample).
    Parquet { columns: Vec<(String, PqType)> },
}

/// A Parquet column's type (`FIXED_WIDTH` in `builder/aeiou/formats.py`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PqType {
    Binary,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt8,
    Float64,
    Float32,
}

impl PqType {
    fn parse(t: &str) -> Option<PqType> {
        Some(match t {
            "binary" => PqType::Binary,
            "int64" => PqType::Int64,
            "int32" => PqType::Int32,
            "int16" => PqType::Int16,
            "int8" => PqType::Int8,
            "uint8" => PqType::UInt8,
            "float64" => PqType::Float64,
            "float32" => PqType::Float32,
            _ => return None,
        })
    }

    /// The physical type (`parquet.thrift` `Type`): INT32 1, INT64 2, FLOAT 4, DOUBLE 5, BYTE_ARRAY 6.
    fn physical(self) -> i32 {
        match self {
            PqType::Binary => 6,
            PqType::Int64 => 2,
            PqType::Float64 => 5,
            PqType::Float32 => 4,
            _ => 1,
        }
    }

    /// Bytes a row takes in a PLAIN page (a fixed-width column).
    fn width(self) -> u64 {
        match self {
            PqType::Int64 | PqType::Float64 => 8,
            _ => 4,
        }
    }

    /// `(converted type, bit width, signed)` of the narrow integers, written as INT32.
    fn narrow(self) -> Option<(i32, i8, bool)> {
        match self {
            PqType::Int8 => Some((15, 8, true)),
            PqType::Int16 => Some((16, 16, true)),
            PqType::UInt8 => Some((11, 8, false)),
            _ => None,
        }
    }

    /// Sample `id`'s value, PLAIN: the Python writer's `(id % 1000).astype(t)` for the
    /// integers (wrapping), `id.astype(t)` for the floats.
    fn value(self, id: i64, out: &mut Vec<u8>) {
        let m = id % 1000;
        match self {
            PqType::Int64 => out.extend_from_slice(&m.to_le_bytes()),
            PqType::Int32 => out.extend_from_slice(&(m as i32).to_le_bytes()),
            PqType::Int16 => out.extend_from_slice(&(m as i16 as i32).to_le_bytes()),
            PqType::Int8 => out.extend_from_slice(&(m as i8 as i32).to_le_bytes()),
            PqType::UInt8 => out.extend_from_slice(&(m as u8 as i32).to_le_bytes()),
            PqType::Float64 => out.extend_from_slice(&(id as f64).to_le_bytes()),
            PqType::Float32 => out.extend_from_slice(&(id as f32).to_le_bytes()),
            PqType::Binary => unreachable!("the binary column is the sample"),
        }
    }
}

impl Writer {
    /// The writer of a resolved `format` entry (`schema/README.md` §2), or why there is none.
    pub fn from_format(dataset: &str, format: &Value) -> Result<Writer> {
        let class = format["class"].as_str().unwrap_or("");
        let settings = &format["framing"]["writer"];
        match class {
            "tfrecord" => Ok(Writer::TfRecord),
            "webdataset" => {
                let mut members = Vec::new();
                let given = settings["members"].as_array().cloned().unwrap_or_else(|| vec![json!(["jpg", null]), json!(["cls", 4])]);
                for m in &given {
                    let (ext, n) = (m[0].as_str(), &m[1]);
                    let Some(ext) = ext else { bail!("dataset `{dataset}`: webdataset member {m}: expected [extension, bytes or null]") };
                    let n = if n.is_null() { None } else { Some(n.as_u64().ok_or_else(|| anyhow::anyhow!("dataset `{dataset}`: webdataset member {m}: bytes must be a non-negative integer or null"))?) };
                    members.push((ext.to_string(), n));
                }
                if members.iter().filter(|(_, n)| n.is_none()).count() != 1 {
                    bail!("dataset `{dataset}`: webdataset: exactly one member is the sample (bytes null)");
                }
                Ok(Writer::WebDataset { members })
            }
            "hdf5" => Ok(Writer::Hdf5 { headers: BTreeMap::new(), version: format["version"].as_str().unwrap_or("?").to_string() }),
            "parquet" => {
                let given = settings["columns"].as_array().cloned().unwrap_or_else(|| vec![json!(["image", "binary"]), json!(["label", "int64"])]);
                let mut columns = Vec::new();
                for c in &given {
                    let (Some(name), Some(t)) = (c[0].as_str(), c[1].as_str().and_then(PqType::parse)) else {
                        bail!("dataset `{dataset}`: parquet column {c}: expected [name, binary or a fixed-width type]")
                    };
                    columns.push((name.to_string(), t));
                }
                if columns.iter().filter(|(_, t)| *t == PqType::Binary).count() != 1 {
                    bail!("dataset `{dataset}`: parquet: exactly one `binary` column carries the sample");
                }
                Ok(Writer::Parquet { columns })
            }
            _ => bail!("dataset `{dataset}`: no writer for format class `{class}`"),
        }
    }

    /// What the manifest's `format.writer` names: this writer, whose bytes the id covers.
    pub fn library(&self) -> String {
        match self {
            Writer::TfRecord => "aeiou datagen tfrecord writer".into(),
            Writer::WebDataset { .. } => "aeiou datagen ustar writer".into(),
            Writer::Hdf5 { version, .. } => format!("aeiou datagen hdf5 writer (header: aeiou-header, {version})"),
            Writer::Parquet { .. } => PARQUET_CREATED_BY.into(),
        }
    }

    /// The row counts whose headers this writer needs, for a dataset of `count` samples in
    /// files of `spf` (the full files and a short last one).
    pub fn header_rows(&self, count: i64, spf: i64) -> Vec<i64> {
        match self {
            Writer::Hdf5 { .. } if count > 0 => {
                let mut v = vec![spf.min(count)];
                let last = count % spf;
                if last != 0 && last != v[0] {
                    v.push(last);
                }
                v
            }
            _ => Vec::new(),
        }
    }

    pub fn set_headers(&mut self, given: BTreeMap<i64, Vec<u8>>) {
        if let Writer::Hdf5 { headers, .. } = self {
            *headers = given;
        }
    }

    /// Every file of `meta` framed as this writer frames it, the payload skipped (`skim`): the
    /// refusals the dataset's geometry alone decides (a Parquet page or sample larger than its
    /// header can say, a page header outside the layout's varint band, a footer larger than
    /// the layout's, a file of other than the layout's size), made before a datagen writes
    /// anything, with the writer's own messages. HDF5 frames nothing but the header
    /// `aeiou-header` makes later. O(samples), the work the space check's projection does, over
    /// `threads` threads; the refusal of the lowest file is the one returned.
    pub fn check(&self, meta: &DsMeta<'_>, threads: usize) -> Result<()> {
        use std::sync::atomic::{AtomicI64, Ordering};
        if matches!(self, Writer::Hdf5 { .. }) || !matches!(meta, DsMeta::Files { .. }) {
            return Ok(());
        }
        let files = meta.files();
        let threads = threads.clamp(1, files.max(1) as usize) as i64;
        let per = (files + threads - 1) / threads;
        // the lowest file refused so far: a thread past it stops
        let lowest = AtomicI64::new(i64::MAX);
        let one = |file: i64, filler: &mut Filler| -> Result<()> {
            let seed_of = |_: u64| 0u64;
            let mut s = Stream::new(self, meta, file, &seed_of, filler)?;
            let (at, size) = (s.skim()?, meta.file_size(file)? as u64);
            if at != size {
                bail!("the writer made {at} bytes, the layout predicts {size}: the format class's layout does not match its writer");
            }
            Ok(())
        };
        let (one, lowest) = (&one, &lowest);
        let found: Vec<Option<anyhow::Error>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    sc.spawn(move || {
                        let mut filler = Filler::new(1);
                        for file in (t * per)..((t + 1) * per).min(files) {
                            if file > lowest.load(Ordering::Relaxed) {
                                break;
                            }
                            if let Err(e) = one(file, &mut filler) {
                                lowest.fetch_min(file, Ordering::Relaxed);
                                let path = meta.file_path(file, None).map(|p| p.to_string()).unwrap_or_else(|_| format!("{file}"));
                                return Some(e.context(format!("file {path}")));
                            }
                        }
                        None
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("a format check thread panicked")).collect()
        });
        match found.into_iter().flatten().next() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Where `aeiou-header` is: beside this binary, the real file (symlinks resolved), never on
/// `PATH` (a search could find another tool or another version).
pub fn header_tool() -> Result<PathBuf> {
    let exe = std::env::current_exe().and_then(|p| p.canonicalize()).context("locating this binary, to find aeiou-header beside it")?;
    let tool = exe.with_file_name("aeiou-header");
    if !tool.exists() {
        bail!("{} not found: aeiou-header makes HDF5 headers and must sit beside aeiou (copy the release's bin/aeiou-header, or, in a source tree, symlink the builder's .venv/bin/aeiou-header there)", tool.display());
    }
    Ok(tool)
}

/// The headers of files of each of `rows` rows of the dataset whose resolved `format` entry
/// is `format`, from one run of `aeiou-header`: back to back on its stdout, each the layout's
/// `file_header` bytes.
pub fn run_header_tool(tool: &std::path::Path, format: &Value, rows: &[i64]) -> Result<BTreeMap<i64, Vec<u8>>> {
    let mut cmd = std::process::Command::new(tool);
    cmd.arg("--format").arg(serde_json::to_string(format)?);
    for n in rows {
        cmd.arg("--rows").arg(n.to_string());
    }
    let out = cmd.output().with_context(|| format!("running {}", tool.display()))?;
    if !out.status.success() {
        bail!("{} failed ({}): {}", tool.display(), out.status, String::from_utf8_lossy(&out.stderr).trim());
    }
    let each = format["framing"]["file_header"].as_u64().ok_or_else(|| anyhow!("the framing has no file_header"))? as usize;
    if out.stdout.len() != each * rows.len() {
        bail!("{} gave {} bytes for {} header(s) of {each} bytes, the framing's file_header", tool.display(), out.stdout.len(), rows.len());
    }
    Ok(rows.iter().zip(out.stdout.chunks(each.max(1))).map(|(n, h)| (*n, h.to_vec())).collect())
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn unhex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        bail!("odd-length hex");
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| anyhow!("hex: {e}"))).collect()
}

fn masked_crc32c(c: u32) -> u32 {
    c.rotate_right(15).wrapping_add(0xA282_EAD8)
}

/// `n` in octal, `digits - 1` digits and a NUL (`tarfile.itn` for ustar); refused when `n` needs
/// more digits than the field has (ustar has no wider form, and GNU base-256 or pax would change
/// the bytes the format class predicts).
fn octal(out: &mut [u8], n: u64, field: &str) -> Result<()> {
    let s = format!("{:0w$o}", n, w = out.len() - 1);
    if s.len() >= out.len() {
        bail!("a tar {field} of {n}: more than ustar's {}-digit octal field holds ({})", out.len() - 1, (1u64 << (3 * (out.len() - 1))) - 1);
    }
    out[..s.len()].copy_from_slice(s.as_bytes());
    out[s.len()] = 0;
    Ok(())
}

/// The largest tar member ustar's 11-digit size field can say: 8 GiB − 1.
const USTAR_MAX_SIZE: u64 = (1 << 33) - 1;

/// The ustar header of a regular member, as `tarfile.TarInfo.tobuf(USTAR_FORMAT)` makes it.
fn tar_header(name: &str, size: u64) -> Result<[u8; 512]> {
    let mut h = [0u8; 512];
    if name.len() > 100 {
        bail!("tar member name `{name}` is longer than 100 bytes");
    }
    if size > USTAR_MAX_SIZE {
        bail!("tar member `{name}` of {size} bytes: more than ustar's 8 GiB − 1 ({USTAR_MAX_SIZE}) size field");
    }
    h[..name.len()].copy_from_slice(name.as_bytes());
    octal(&mut h[100..108], 0o644, "mode")?;
    octal(&mut h[108..116], 0, "uid")?;
    octal(&mut h[116..124], 0, "gid")?;
    octal(&mut h[124..136], size, "size")?;
    octal(&mut h[136..148], 0, "mtime")?;
    h[148..156].fill(b' ');
    h[156] = b'0';
    h[257..265].copy_from_slice(b"ustar\x0000");
    // at most 512 · 255 = 0o376800: six digits, a NUL, and the field's last space
    let sum: u64 = h.iter().map(|&b| b as u64).sum();
    octal(&mut h[148..155], sum, "checksum")?;
    Ok(h)
}

/// The `created_by` of the files this writer makes, and its name in the manifest.
const PARQUET_CREATED_BY: &str = "aeiou datagen parquet writer";

/// Thrift's compact protocol, the parts a Parquet footer uses: struct fields by id delta,
/// zigzag varints, binaries, lists.
struct Thrift {
    out: Vec<u8>,
    /// The last field id of each open struct.
    last: Vec<i16>,
}

const T_TRUE: u8 = 1;
const T_FALSE: u8 = 2;
const T_BYTE: u8 = 3;
const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

impl Thrift {
    fn new() -> Thrift {
        Thrift { out: Vec::new(), last: vec![0] }
    }

    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.out.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.out.push(v as u8);
    }

    fn zigzag(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }

    fn field(&mut self, id: i16, t: u8) {
        let last = self.last.last_mut().expect("an open struct");
        let d = id - *last;
        *last = id;
        if (1..=15).contains(&d) {
            self.out.push((d as u8) << 4 | t);
        } else {
            self.out.push(t);
            self.zigzag(id as i64);
        }
    }

    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.zigzag(v as i64);
    }

    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.zigzag(v);
    }

    fn byte(&mut self, id: i16, v: i8) {
        self.field(id, T_BYTE);
        self.out.push(v as u8);
    }

    fn bool(&mut self, id: i16, v: bool) {
        self.field(id, if v { T_TRUE } else { T_FALSE });
    }

    fn binary(&mut self, id: i16, b: &[u8]) {
        self.field(id, T_BINARY);
        self.elem_binary(b);
    }

    fn elem_binary(&mut self, b: &[u8]) {
        self.varint(b.len() as u64);
        self.out.extend_from_slice(b);
    }

    /// A list field of `n` elements of type `t`; the elements follow.
    fn list(&mut self, id: i16, n: usize, t: u8) {
        self.field(id, T_LIST);
        if n < 15 {
            self.out.push((n as u8) << 4 | t);
        } else {
            self.out.push(0xF0 | t);
            self.varint(n as u64);
        }
    }

    /// A struct field, or (`None`) a struct element of a list; closed by `end`.
    fn begin(&mut self, id: Option<i16>) {
        if let Some(id) = id {
            self.field(id, T_STRUCT);
        }
        self.last.push(0);
    }

    fn end(&mut self) {
        self.out.push(0);
        self.last.pop();
    }
}

/// The page header of one PLAIN, uncompressed data page of `rows` values in `data` bytes, as
/// pyarrow writes it: an empty statistics struct, RLE level encodings.
fn parquet_page_header(data: u64, rows: i64) -> Result<Vec<u8>> {
    let data = i32::try_from(data).map_err(|_| anyhow!("a Parquet page of {data} bytes: more than a page holds (2 GiB)"))?;
    let mut t = Thrift::new();
    t.i32(1, 0);
    t.i32(2, data);
    t.i32(3, data);
    t.begin(Some(5));
    t.i32(1, rows as i32);
    t.i32(2, 0);
    t.i32(3, 3);
    t.i32(4, 3);
    t.begin(Some(5));
    t.end();
    t.end();
    t.end();
    Ok(t.out)
}

/// A column chunk as written: where, how long (header included), its rows, and (the binary
/// column) the samples' bytes without their length prefixes.
#[derive(Debug, Clone, Copy)]
struct Chunk {
    at: u64,
    len: u64,
    rows: i64,
    raw: u64,
}

/// The thrift `FileMetaData` of a file of these chunks (row group by row group), with the
/// key `aeiou.pad` holding `pad` bytes, as pyarrow writes it but for `created_by`.
fn parquet_footer(columns: &[(String, PqType)], chunks: &[Chunk], pad: usize, created_by: &str) -> Vec<u8> {
    let mut t = Thrift::new();
    t.i32(1, 2);
    t.list(2, columns.len() + 1, T_STRUCT);
    t.begin(None);
    t.i32(3, 0);
    t.binary(4, b"schema");
    t.i32(5, columns.len() as i32);
    t.end();
    for (name, ty) in columns {
        t.begin(None);
        t.i32(1, ty.physical());
        t.i32(3, 0);
        t.binary(4, name.as_bytes());
        if let Some((converted, bits, signed)) = ty.narrow() {
            t.i32(6, converted);
            t.begin(Some(10));
            t.begin(Some(10));
            t.byte(1, bits);
            t.bool(2, signed);
            t.end();
            t.end();
        }
        t.end();
    }
    let groups: Vec<&[Chunk]> = chunks.chunks(columns.len()).collect();
    t.i64(3, groups.iter().map(|g| g[0].rows).sum());
    t.list(4, groups.len(), T_STRUCT);
    for g in &groups {
        t.begin(None);
        t.list(1, columns.len(), T_STRUCT);
        for ((name, ty), c) in columns.iter().zip(g.iter()) {
            t.begin(None);
            t.i64(2, 0);
            t.begin(Some(3));
            t.i32(1, ty.physical());
            t.list(2, 2, T_I32);
            t.zigzag(3);
            t.zigzag(0);
            t.list(3, 1, T_BINARY);
            t.elem_binary(name.as_bytes());
            t.i32(4, 0);
            t.i64(5, c.rows);
            t.i64(6, c.len as i64);
            t.i64(7, c.len as i64);
            t.i64(9, c.at as i64);
            t.list(13, 1, T_STRUCT);
            t.begin(None);
            t.i32(1, 0);
            t.i32(2, 0);
            t.i32(3, 1);
            t.end();
            if *ty == PqType::Binary {
                t.begin(Some(16));
                t.i64(1, c.raw as i64);
                t.list(2, 0, T_I64);
                t.list(3, 0, T_I64);
                t.end();
            }
            t.end();
            t.end();
        }
        let bytes: u64 = g.iter().map(|c| c.len).sum();
        t.i64(2, bytes as i64);
        t.i64(3, g[0].rows);
        t.i64(5, g[0].at as i64);
        t.i64(6, bytes as i64);
        t.end();
    }
    t.list(5, 1, T_STRUCT);
    t.begin(None);
    t.binary(1, b"aeiou.pad");
    t.field(2, T_BINARY);
    t.varint(pad as u64);
    t.out.resize(t.out.len() + pad, b'p');
    t.end();
    t.binary(6, created_by.as_bytes());
    t.list(7, columns.len(), T_STRUCT);
    for _ in columns {
        t.begin(None);
        t.begin(Some(1));
        t.end();
        t.end();
    }
    t.end();
    t.out
}

/// The footer padded to `target` bytes (the layout's, without the length and `PAR1`), then
/// its length and `PAR1`.
fn parquet_tail(columns: &[(String, PqType)], chunks: &[Chunk], target: u64) -> Result<Vec<u8>> {
    let natural = parquet_footer(columns, chunks, 0, PARQUET_CREATED_BY).len() as u64;
    if natural > target {
        bail!("the Parquet footer ({natural} bytes) exceeds the layout's {target}: the format class's footer margins are too small");
    }
    let mut pad = target - natural;
    for _ in 0..4 {
        let mut f = parquet_footer(columns, chunks, pad as usize, PARQUET_CREATED_BY);
        let len = f.len() as u64;
        if len == target {
            f.extend_from_slice(&(len as u32).to_le_bytes());
            f.extend_from_slice(b"PAR1");
            return Ok(f);
        }
        // the pad's length prefix grew a byte
        pad = (pad + target).checked_sub(len).ok_or_else(|| anyhow!("no pad makes a {target}-byte Parquet footer"))?;
    }
    bail!("no pad makes a {target}-byte Parquet footer")
}

/// Where a Parquet stream is.
#[derive(Default)]
struct PqAt {
    started: bool,
    /// The unit and column whose chunk is being sent, and (binary) its next row.
    u: i64,
    c: usize,
    row: usize,
    in_chunk: bool,
    /// The sizes of the unit's samples.
    sizes: Vec<u64>,
    chunks: Vec<Chunk>,
    /// Where each unit starts, by the layout.
    unit_at: Vec<u64>,
    ended: bool,
}

/// One container file as a byte stream.
pub struct Stream<'s, 'a> {
    writer: &'s Writer,
    meta: &'s DsMeta<'a>,
    seed_of: &'s dyn Fn(u64) -> u64,
    filler: &'s mut Filler,
    /// The first sample id of the file, and how many it holds.
    first: i64,
    n: i64,
    /// The next sample, and (WebDataset) the next member of it.
    sample: i64,
    member: usize,
    /// Framing bytes waiting to be sent, from `small_at`.
    small: Vec<u8>,
    small_at: usize,
    /// Payload bytes of the current sample still to send, at this logical offset.
    pay_left: u64,
    pay_at: u64,
    /// After the payload: TFRecord's data CRC, so far; WebDataset's padding.
    crc: u32,
    after: After,
    /// Bytes sent.
    pos: u64,
    /// The archive's end is queued (WebDataset).
    ended: bool,
    done: bool,
    pq: PqAt,
}

#[derive(Clone, Copy)]
enum After {
    Nothing,
    Crc,
    Pad,
}

impl<'s, 'a> Stream<'s, 'a> {
    pub fn new(writer: &'s Writer, meta: &'s DsMeta<'a>, file: i64, seed_of: &'s dyn Fn(u64) -> u64, filler: &'s mut Filler) -> Result<Self> {
        let spf = match meta {
            DsMeta::Files { spf, .. } => *spf,
            DsMeta::Array { .. } => bail!("`{}` is an array dataset: no container", meta.name()),
        };
        let n = meta.samples_in_file(file)?;
        Ok(Stream { writer, meta, seed_of, filler, first: file * spf, n, sample: 0, member: 0, small: Vec::with_capacity(TAR_RECORD as usize), small_at: 0, pay_left: 0, pay_at: 0, crc: 0, after: After::Nothing, pos: 0, ended: false, done: false, pq: PqAt::default() })
    }

    /// Bytes sent so far.
    pub fn pos(&self) -> u64 {
        self.pos
    }

    fn set_small(&mut self, b: &[u8]) {
        self.small.clear();
        self.small.extend_from_slice(b);
        self.small_at = 0;
    }

    /// Queue what follows the current item: the payload's trailer, the next sample's head,
    /// or the end of the file.
    fn advance(&mut self) -> Result<()> {
        match std::mem::replace(&mut self.after, After::Nothing) {
            After::Crc => {
                let c = masked_crc32c(self.crc);
                self.set_small(&c.to_le_bytes());
                return Ok(());
            }
            After::Pad => {
                let pad = (TAR_BLOCK - self.pos % TAR_BLOCK) % TAR_BLOCK;
                if pad > 0 {
                    self.small.clear();
                    self.small.resize(pad as usize, 0);
                    self.small_at = 0;
                    return Ok(());
                }
            }
            After::Nothing => {}
        }
        if let Writer::Parquet { columns } = self.writer {
            return self.advance_parquet(columns);
        }
        if self.sample >= self.n {
            if let (Writer::WebDataset { .. }, false) = (self.writer, std::mem::replace(&mut self.ended, true)) {
                // two zero blocks, then zeros to the record (tarfile's close)
                let end = (self.pos + 2 * TAR_BLOCK).div_ceil(TAR_RECORD) * TAR_RECORD;
                if end > self.pos {
                    self.small.clear();
                    self.small.resize((end - self.pos) as usize, 0);
                    self.small_at = 0;
                    return Ok(());
                }
            }
            self.done = true;
            return Ok(());
        }
        let size = self.meta.sample_size(self.first + self.sample)? as u64;
        match self.writer {
            Writer::TfRecord => {
                let len = size.to_le_bytes();
                let mut head = [0u8; 12];
                head[..8].copy_from_slice(&len);
                head[8..].copy_from_slice(&masked_crc32c(crc32c::crc32c(&len)).to_le_bytes());
                self.set_small(&head);
                self.pay_left = size;
                self.crc = 0;
                self.after = After::Crc;
                self.sample += 1;
            }
            Writer::Hdf5 { headers, .. } => {
                // the header, then every row: one item for the whole file
                let head = headers.get(&self.n).ok_or_else(|| anyhow!("no HDF5 header for a file of {} rows", self.n))?;
                self.set_small(head);
                let mut total = size;
                for i in 1..self.n {
                    total += self.meta.sample_size(self.first + i)? as u64;
                }
                self.pay_left = total;
                self.sample = self.n;
            }
            Writer::Parquet { .. } => unreachable!("advance_parquet"),
            Writer::WebDataset { members } => {
                let (ext, bytes) = &members[self.member];
                let n = bytes.unwrap_or(size);
                let h = tar_header(&format!("{:09}.{ext}", self.first + self.sample), n)?;
                self.set_small(&h);
                if bytes.is_none() {
                    self.pay_left = size;
                } else {
                    // a fixed member: zeros, sent with its header
                    self.small.resize(512 + n as usize, 0);
                }
                self.after = After::Pad;
                self.member += 1;
                if self.member == members.len() {
                    self.member = 0;
                    self.sample += 1;
                }
            }
        }
        Ok(())
    }

    /// Parquet: `PAR1`; per row group, per column, the page header and the rows (a binary
    /// row its 4-byte length, then the sample); the footer.
    fn advance_parquet(&mut self, columns: &[(String, PqType)]) -> Result<()> {
        let layout = self.meta.framing()?;
        if self.pq.ended {
            self.done = true;
            return Ok(());
        }
        if !self.pq.started {
            self.pq.started = true;
            let DsMeta::Files { spf, .. } = self.meta else { bail!("an array dataset has no container") };
            let lens = self.meta.unit_lens(self.first / spf)?;
            let mut at = 4u64;
            for len in lens {
                self.pq.unit_at.push(at);
                at += len as u64;
            }
            self.set_small(b"PAR1");
            return Ok(());
        }
        let pq = &mut self.pq;
        if pq.in_chunk {
            if columns[pq.c].1 == PqType::Binary && pq.row < pq.sizes.len() {
                let size = pq.sizes[pq.row];
                pq.row += 1;
                let len = u32::try_from(size).map_err(|_| anyhow!("a sample of {size} bytes in a Parquet binary column (at most 4 GiB)"))?;
                self.pay_left = size;
                self.set_small(&len.to_le_bytes());
                return Ok(());
            }
            let c = pq.chunks.last_mut().expect("the open chunk");
            c.len = self.pos - c.at;
            pq.in_chunk = false;
            pq.c += 1;
            if pq.c == columns.len() {
                pq.c = 0;
                pq.u += 1;
            }
        }
        let units = pq.unit_at.len() as i64;
        if pq.u < units {
            let first = self.first + pq.u * layout.unit;
            if pq.c == 0 {
                if self.pos != pq.unit_at[pq.u as usize] {
                    bail!("row group {} starts at {}, the layout predicts {}: the format class's layout does not match its writer", pq.u, self.pos, pq.unit_at[pq.u as usize]);
                }
                let rows = (self.n - pq.u * layout.unit).min(layout.unit);
                pq.sizes.clear();
                for i in 0..rows {
                    pq.sizes.push(self.meta.sample_size(first + i)? as u64);
                }
            }
            let rows = pq.sizes.len() as i64;
            let ty = columns[pq.c].1;
            let (data, raw) = match ty {
                PqType::Binary => {
                    let raw: u64 = pq.sizes.iter().sum();
                    (raw + 4 * rows as u64, raw)
                }
                _ => (ty.width() * rows as u64, 0),
            };
            let head = parquet_page_header(data, rows)?;
            let want = layout.columns.get(pq.c).map(|c| c.header).unwrap_or(-1);
            if head.len() as i64 != want {
                bail!("row group {} column {}: a {}-byte page header, the layout predicts {want} (a chunk outside the declared varint band; brief item 31 for a short last row group)", pq.u, pq.c, head.len());
            }
            pq.chunks.push(Chunk { at: self.pos, len: 0, rows, raw });
            pq.in_chunk = true;
            pq.row = 0;
            self.small.clear();
            self.small.extend_from_slice(&head);
            if ty != PqType::Binary {
                for i in 0..rows {
                    ty.value(first + i, &mut self.small);
                }
            }
            self.small_at = 0;
            return Ok(());
        }
        let target = layout.file_footer + layout.file_footer_per_sample * self.n + layout.file_footer_per_unit * units - 8;
        let tail = parquet_tail(columns, &pq.chunks, target.max(0) as u64)?;
        pq.ended = true;
        self.set_small(&tail);
        Ok(())
    }

    /// The file's bytes counted, not sent: every framing item built and checked as `read`
    /// builds it, the payload skipped (a TFRecord data CRC is then not the data's). Returns
    /// the file's size.
    fn skim(&mut self) -> Result<u64> {
        while !self.done {
            if self.small_at < self.small.len() {
                self.pos += (self.small.len() - self.small_at) as u64;
                self.small_at = self.small.len();
            } else if self.pay_left > 0 {
                self.pos += self.pay_left;
                self.pay_at += self.pay_left;
                self.pay_left = 0;
            } else {
                self.advance()?;
            }
        }
        Ok(self.pos)
    }

    /// Fill as much of `out` as the file has left; 0 at its end.
    pub fn read(&mut self, out: &mut [u8]) -> Result<usize> {
        let mut at = 0usize;
        while at < out.len() && !self.done {
            if self.small_at < self.small.len() {
                let k = (self.small.len() - self.small_at).min(out.len() - at);
                out[at..at + k].copy_from_slice(&self.small[self.small_at..self.small_at + k]);
                self.small_at += k;
                at += k;
                self.pos += k as u64;
            } else if self.pay_left > 0 {
                let k = (self.pay_left.min((out.len() - at) as u64)) as usize;
                let piece = &mut out[at..at + k];
                self.filler.fill_range(self.seed_of, self.pay_at, piece);
                if let After::Crc = self.after {
                    self.crc = crc32c::crc32c_append(self.crc, piece);
                }
                self.pay_at += k as u64;
                self.pay_left -= k as u64;
                at += k;
                self.pos += k as u64;
            } else {
                self.advance()?;
            }
        }
        Ok(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_crc_is_tensorflows() {
        // crc32c("123456789") = 0xE3069283 (the CRC-32C check value)
        let c = crc32c::crc32c(b"123456789");
        assert_eq!(c, 0xE306_9283);
        assert_eq!(masked_crc32c(c), ((c >> 15) | (c << 17)).wrapping_add(0xA282_EAD8));
    }

    #[test]
    fn tar_header_is_cpythons() {
        // `tarfile.TarInfo("000000007.jpg")`, size 5, mtime 0, USTAR (CPython 3.12): checksum 007410
        let h = tar_header("000000007.jpg", 5).unwrap();
        assert_eq!(&h[100..108], b"0000644\0");
        assert_eq!(&h[124..136], b"00000000005\0");
        assert_eq!(&h[148..156], b"007410\0 ");
        assert_eq!(&h[257..265], b"ustar\x0000");
        assert!(h[329..345].iter().all(|&b| b == 0));
        // the largest member ustar can say, and one byte more refused, never a panic
        let h = tar_header("000000007.bin", USTAR_MAX_SIZE).unwrap();
        assert_eq!(&h[124..136], b"77777777777\0");
        let e = tar_header("000000007.bin", USTAR_MAX_SIZE + 1).unwrap_err().to_string();
        assert!(e.contains("8589934592 bytes") && e.contains("8 GiB"), "{e}");
        let e = octal(&mut [0u8; 8], 1 << 21, "mode").unwrap_err().to_string();
        assert!(e.contains("mode of 2097152") && e.contains("7-digit"), "{e}");
    }

    fn unhex_test(s: &str) -> Vec<u8> {
        unhex(s).unwrap()
    }

    #[test]
    fn parquet_page_header_is_pyarrows() {
        // pyarrow 25.0.1, the class's options: 4 INT32 rows; 4 binary rows of 15 bytes in all
        assert_eq!(parquet_page_header(16, 4).unwrap(), unhex_test("1500152015202c15081500150615061c000000"));
        assert_eq!(parquet_page_header(31, 4).unwrap(), unhex_test("1500153e153e2c15081500150615061c000000"));
        // the length `formats.parquet_page_header` predicts: 16 + 2 varints of 2·data + 1 of 2·rows
        for (data, rows, len) in [(5, 1, 19), (64 * 12, 64, 22), (64 * 100004, 64, 26), (1000 * 1028, 1000, 24), (10000 * 1028, 10000, 27)] {
            assert_eq!(parquet_page_header(data, rows).unwrap().len(), len);
        }
    }

    #[test]
    fn parquet_footer_is_pyarrows_but_for_created_by() {
        // one row group of 4 rows (sample sizes 3, 5, 0, 7) over int8, uint8, int16, binary,
        // float32, int32, `aeiou.pad` = "ppp": pyarrow 25.0.1's footer, from the file's byte 229
        let cols: Vec<(String, PqType)> = [("a", PqType::Int8), ("b", PqType::UInt8), ("c", PqType::Int16), ("i", PqType::Binary), ("d", PqType::Float32), ("e", PqType::Int32)]
            .iter().map(|(n, t)| (n.to_string(), *t)).collect();
        let chunks: Vec<Chunk> = [(4, 35, 0), (39, 35, 0), (74, 35, 0), (109, 50, 15), (159, 35, 0), (194, 35, 0)]
            .iter().map(|&(at, len, raw)| Chunk { at, len, rows: 4, raw }).collect();
        let want = unhex_test(concat!(
            "1504197c35001806736368656d61150c0015022500180161251e4cac1308110000001502250018016225164cac1308120000001502250018016325204cac131011000000150c250018016900150825001801640015022500180165001608191c196c26001c1502192506001918016115001608164616462608491c15001500150200000026001c150219250600191801621500160816461646264e491c15001500150200000026001c150219250600191801631500160816461646269401491c15001500150200000026001c150c1925060019180169150016081664166426da01491c150015001502003c161e1906190600000026001c15081925060019180164150016081646164626be02491c15001500150200000026001c150219250600191801651500160816461646268403491c15001500150200000016c2031608260816c20300191c18096165696f752e7061641803707070001820706172717565742d6370702d6172726f772076657273696f6e2032352e302e31196c1c00001c00001c00001c00001c00001c0000",
            "008f01000050415231"));
        let f = parquet_footer(&cols, &chunks, 3, "parquet-cpp-arrow version 25.0.1");
        assert_eq!(hex(&f), hex(&want[..want.len() - 8]));
        // padded to a size: the pad takes up the length of created_by and of its own prefix
        for target in [f.len() as u64, f.len() as u64 + 200, f.len() as u64 + 20000] {
            let t = parquet_tail(&cols, &chunks, target).unwrap();
            assert_eq!(t.len() as u64, target + 8);
            assert_eq!(&t[t.len() - 8..t.len() - 4], &(target as u32).to_le_bytes());
        }
    }

    // ------------------------------------------------------------ helpers and the writers' paths

    /// The model of `text` (one dataset), handed to `f` with its first dataset and its `format`.
    fn with_meta<R>(text: &str, f: impl FnOnce(&DsMeta<'_>, &Value) -> R) -> R {
        let loaded = crate::load_str(text).unwrap();
        let cfg = crate::eval::Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![] };
        let params = crate::eval::Params::new(&loaded.ast, &cfg).unwrap();
        let model = crate::eval::build_model(&loaded.ast, &cfg, &params).unwrap();
        let doc: Value = serde_json::from_str(text).unwrap();
        let format = doc["datasets"]["d"]["format"].clone();
        f(&model.datasets[0], &format)
    }

    fn abstract_of(dataset: &str) -> String {
        format!(r#"{{"ast": "0.10", "name": "fmt_unit", "datasets": {{"d": {{{dataset}}}}}, "actors": {{"gpu": {{"body": []}}}}}}"#)
    }

    fn seed_of(b: u64) -> u64 {
        0x5eed ^ b
    }

    /// File `file` of `meta` as `writer` streams it, pulled in pieces of `piece` bytes.
    fn file_bytes(writer: &Writer, meta: &DsMeta<'_>, file: i64, piece: usize) -> Vec<u8> {
        let mut filler = Filler::new(1);
        let mut s = Stream::new(writer, meta, file, &seed_of, &mut filler).unwrap();
        let mut out = Vec::new();
        let mut buf = vec![0u8; piece];
        loop {
            let n = s.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
            assert_eq!(s.pos(), out.len() as u64);
        }
        out
    }

    /// The payload stream of a file, logical bytes `[at, at + len)`.
    fn payload(at: u64, len: u64) -> Vec<u8> {
        let mut b = vec![0u8; len as usize];
        Filler::new(1).fill_range(seed_of, at, &mut b);
        b
    }

    #[test]
    fn hex_round_trips_and_refuses_bad_text() {
        let b = [0u8, 1, 0x7f, 0x80, 0xff];
        assert_eq!(hex(&b), "00017f80ff");
        assert_eq!(unhex("00017f80ff").unwrap(), b);
        assert_eq!(unhex("").unwrap(), Vec::<u8>::new());
        assert!(unhex("abc").unwrap_err().to_string().contains("odd-length hex"));
        assert!(unhex("zz").unwrap_err().to_string().starts_with("hex: "));
    }

    #[test]
    fn tar_header_refuses_a_long_name_and_sums_a_full_name() {
        let e = tar_header(&"n".repeat(101), 1).unwrap_err().to_string();
        assert!(e.contains("longer than 100 bytes"), "{e}");
        // a 100-byte name fills the field, with no NUL after it
        let h = tar_header(&"n".repeat(100), 0).unwrap();
        assert_eq!(&h[..100], "n".repeat(100).as_bytes());
        let sum: u64 = h.iter().enumerate().map(|(i, &b)| if (148..156).contains(&i) { b' ' as u64 } else { b as u64 }).sum();
        assert_eq!(&h[148..156], format!("{sum:06o}\0 ").as_bytes());
    }

    #[test]
    fn thrift_long_field_deltas_and_long_lists() {
        // a field id delta outside 1..=15 is the type byte and the id as a zigzag varint
        let mut t = Thrift::new();
        t.i32(20, 1);
        assert_eq!(t.out, [T_I32, 40, 2]);
        t.i32(3, -1);
        assert_eq!(&t.out[3..], [T_I32, 6, 1]);
        // a list of 15 or more elements: 0xF ‖ type, then the count as a varint
        let mut t = Thrift::new();
        t.list(1, 15, T_I32);
        assert_eq!(t.out, [0x10 | T_LIST, 0xF0 | T_I32, 15]);
        let mut t = Thrift::new();
        t.list(1, 300, T_BINARY);
        assert_eq!(t.out, [0x10 | T_LIST, 0xF0 | T_BINARY, 0xAC, 0x02]);
        // zigzag: 0, −1, 1, −2, i64::MIN
        let mut t = Thrift::new();
        for v in [0, -1, 1, -2] {
            t.zigzag(v);
        }
        assert_eq!(t.out, [0, 1, 2, 3]);
        let mut t = Thrift::new();
        t.zigzag(i64::MIN);
        assert_eq!(t.out, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
    }

    #[test]
    fn parquet_types_parse_and_write_as_pyarrow_does() {
        let all = ["binary", "int64", "int32", "int16", "int8", "uint8", "float64", "float32"];
        let types: Vec<PqType> = all.iter().map(|t| PqType::parse(t).unwrap()).collect();
        assert_eq!(PqType::parse("string"), None);
        assert_eq!(types.iter().map(|t| t.physical()).collect::<Vec<_>>(), [6, 2, 1, 1, 1, 1, 5, 4]);
        assert_eq!(types[1..].iter().map(|t| t.width()).collect::<Vec<_>>(), [8, 4, 4, 4, 4, 8, 4]);
        assert_eq!(types.iter().map(|t| t.narrow()).collect::<Vec<_>>(), [None, None, None, Some((16, 16, true)), Some((15, 8, true)), Some((11, 8, false)), None, None]);
        // `(id % 1000).astype(t)`, wrapping, widened to INT32 for the narrow types; floats of the id
        let v = |t: PqType, id: i64| {
            let mut o = Vec::new();
            t.value(id, &mut o);
            o
        };
        assert_eq!(v(PqType::Int64, 1999), 999i64.to_le_bytes());
        assert_eq!(v(PqType::Int32, 1999), 999i32.to_le_bytes());
        assert_eq!(v(PqType::Int16, 1999), 999i32.to_le_bytes());
        assert_eq!(v(PqType::Int8, 1999), (999i32 as i8 as i32).to_le_bytes());
        assert_eq!(v(PqType::Int8, 200), (-56i32).to_le_bytes());
        assert_eq!(v(PqType::UInt8, 300), 44i32.to_le_bytes());
        assert_eq!(v(PqType::Float64, 1999), 1999f64.to_le_bytes());
        assert_eq!(v(PqType::Float32, 1999), 1999f32.to_le_bytes());
    }

    #[test]
    #[should_panic(expected = "the binary column is the sample")]
    fn parquet_binary_has_no_fixed_value() {
        PqType::Binary.value(0, &mut Vec::new());
    }

    #[test]
    fn writers_from_format_and_their_refusals() {
        let w = |f: Value| Writer::from_format("d", &f);
        let err = |f: Value| format!("{:#}", w(f).unwrap_err());
        assert_eq!(w(json!({"class": "tfrecord"})).unwrap().library(), "aeiou datagen tfrecord writer");
        // the defaults: a jpg sample and a 4-byte cls; an image and an int64 label
        match w(json!({"class": "webdataset"})).unwrap() {
            Writer::WebDataset { members } => assert_eq!(members, [("jpg".to_string(), None), ("cls".to_string(), Some(4))]),
            o => panic!("{o:?}"),
        }
        assert_eq!(w(json!({"class": "webdataset"})).unwrap().library(), "aeiou datagen ustar writer");
        match w(json!({"class": "parquet"})).unwrap() {
            Writer::Parquet { columns } => assert_eq!(columns, [("image".to_string(), PqType::Binary), ("label".to_string(), PqType::Int64)]),
            o => panic!("{o:?}"),
        }
        assert_eq!(w(json!({"class": "parquet"})).unwrap().library(), PARQUET_CREATED_BY);
        assert_eq!(w(json!({"class": "hdf5", "version": "h5py 3, hdf5 2"})).unwrap().library(), "aeiou datagen hdf5 writer (header: aeiou-header, h5py 3, hdf5 2)");
        assert_eq!(w(json!({"class": "hdf5"})).unwrap().library(), "aeiou datagen hdf5 writer (header: aeiou-header, ?)");
        let wds = |m: Value| json!({"class": "webdataset", "framing": {"writer": {"members": m}}});
        assert!(err(wds(json!([[3, null]]))).contains("dataset `d`: webdataset member [3,null]: expected [extension, bytes or null]"));
        assert!(err(wds(json!([["jpg", null], ["cls", -4]]))).contains("bytes must be a non-negative integer or null"));
        assert!(err(wds(json!([["jpg", null], ["png", null]]))).contains("dataset `d`: webdataset: exactly one member is the sample (bytes null)"));
        assert!(err(wds(json!([["cls", 4]]))).contains("exactly one member is the sample"));
        let pq = |c: Value| json!({"class": "parquet", "framing": {"writer": {"columns": c}}});
        assert!(err(pq(json!([["image", "binary"], ["label", "string"]]))).contains("dataset `d`: parquet column [\"label\",\"string\"]: expected [name, binary or a fixed-width type]"));
        assert!(err(pq(json!([["image", "binary"], [7, "int64"]]))).contains("expected [name, binary or a fixed-width type]"));
        assert!(err(pq(json!([["a", "binary"], ["b", "binary"]]))).contains("dataset `d`: parquet: exactly one `binary` column carries the sample"));
        assert!(err(pq(json!([["a", "int64"]]))).contains("exactly one `binary` column"));
        assert!(err(json!({"class": "npz"})).contains("dataset `d`: no writer for format class `npz`"));
        assert!(err(json!({})).contains("no writer for format class ``"));
    }

    #[test]
    fn hdf5_header_rows_are_the_full_and_the_short_last_file() {
        let h = Writer::Hdf5 { headers: BTreeMap::new(), version: String::new() };
        assert_eq!(h.header_rows(64, 16), [16]);
        assert_eq!(h.header_rows(70, 16), [16, 6]);
        assert_eq!(h.header_rows(10, 16), [10]);
        assert_eq!(h.header_rows(0, 16), Vec::<i64>::new());
        assert_eq!(Writer::TfRecord.header_rows(70, 16), Vec::<i64>::new());
        // headers are kept by an HDF5 writer alone
        let mut t = Writer::TfRecord;
        t.set_headers(BTreeMap::from([(1, vec![1])]));
        assert!(matches!(t, Writer::TfRecord));
    }

    /// A fake `aeiou-header`: a script printing `out` and exiting `code`.
    fn fake_tool(dir: &std::path::Path, out: &str, code: i32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(format!("tool-{code}-{}", out.len()));
        std::fs::write(&p, format!("#!/bin/sh\nprintf '%s' '{out}'\necho \"args: $*\" >&2\nexit {code}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[test]
    fn header_tool_output_is_split_and_checked() {
        let dir = std::env::temp_dir().join(format!("aeiou-formats-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fmt = json!({"class": "hdf5", "framing": {"file_header": 4}});
        let got = run_header_tool(&fake_tool(&dir, "AAAABBBB", 0), &fmt, &[16, 6]).unwrap();
        assert_eq!(got, BTreeMap::from([(16, b"AAAA".to_vec()), (6, b"BBBB".to_vec())]));
        let e = run_header_tool(&fake_tool(&dir, "AAAABBB", 0), &fmt, &[16, 6]).unwrap_err().to_string();
        assert!(e.contains("gave 7 bytes for 2 header(s) of 4 bytes, the framing's file_header"), "{e}");
        let e = run_header_tool(&fake_tool(&dir, "", 3), &fmt, &[16]).unwrap_err().to_string();
        assert!(e.contains("failed (exit status: 3)") && e.contains("args: --format") && e.contains("--rows 16"), "{e}");
        let e = run_header_tool(&fake_tool(&dir, "AAAA", 0), &json!({"class": "hdf5"}), &[16]).unwrap_err().to_string();
        assert!(e.contains("the framing has no file_header"), "{e}");
        let e = format!("{:#}", run_header_tool(&dir.join("absent"), &fmt, &[16]).unwrap_err());
        assert!(e.contains("running ") && e.contains("absent"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
        // the tool is looked for beside this binary alone, never on PATH
        let beside = std::env::current_exe().unwrap().canonicalize().unwrap().with_file_name("aeiou-header");
        match header_tool() {
            Ok(t) => assert_eq!(t, beside),
            Err(e) => assert!(!beside.exists() && e.to_string().contains("not found: aeiou-header makes HDF5 headers and must sit beside aeiou"), "{e}"),
        }
    }

    const TFR: &str = r#""layout": "compound", "pattern": "s/{id:04}.tfrecord", "count": 25, "samples_per_file": 10,
        "size": {"const": 1000}, "seed": 7, "access": "stream",
        "format": {"class": "tfrecord", "framing": {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}}"#;

    #[test]
    fn tfrecord_files_are_framed_records_of_the_payload_stream() {
        with_meta(&abstract_of(TFR), |meta, fmt| {
            let w = Writer::from_format("d", fmt).unwrap();
            w.check(meta, 4).unwrap();
            for file in 0..meta.files() {
                let b = file_bytes(&w, meta, file, 4096);
                assert_eq!(b.len() as i64, meta.file_size(file).unwrap());
                // pulled in odd pieces, the same bytes
                assert_eq!(file_bytes(&w, meta, file, 7), b);
                let (mut at, mut logical, mut n) = (0usize, 0u64, 0);
                while at < b.len() {
                    let len = u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
                    assert_eq!(len, 1000);
                    assert_eq!(&b[at + 8..at + 12], &masked_crc32c(crc32c::crc32c(&b[at..at + 8])).to_le_bytes());
                    let data = &b[at + 12..at + 12 + len as usize];
                    assert_eq!(data, payload(logical, len), "file {file} record {n}");
                    let crc = &b[at + 12 + len as usize..at + 16 + len as usize];
                    assert_eq!(crc, masked_crc32c(crc32c::crc32c(data)).to_le_bytes());
                    at += 16 + len as usize;
                    logical += len;
                    n += 1;
                }
                assert_eq!(n, meta.samples_in_file(file).unwrap());
            }
        });
    }

    #[test]
    fn a_layout_other_than_the_writers_is_refused_by_check() {
        // a 10-byte record header where the writer makes 12: the size, not the bytes, decides
        let bad = abstract_of(&TFR.replace("\"row_header\": 12", "\"row_header\": 10"));
        with_meta(&bad, |meta, fmt| {
            let w = Writer::from_format("d", fmt).unwrap();
            let e = format!("{:#}", w.check(meta, 3).unwrap_err());
            assert!(e.contains("file s/0000.tfrecord: the writer made 10160 bytes, the layout predicts 10140: the format class's layout does not match its writer"), "{e}");
            // one thread, the same refusal: the lowest file's
            let e1 = format!("{:#}", w.check(meta, 1).unwrap_err());
            assert_eq!(e, e1);
        });
    }

    const WDS: &str = r#""layout": "compound", "pattern": "w/{id:04}.tar", "count": 25, "samples_per_file": 10,
        "size": {"const": 1000}, "seed": 7, "access": "stream",
        "format": {"class": "webdataset", "framing": {"file_footer": 1024, "file_align": 10240,
          "columns": [{"row_header": 512, "row_footer": 1024, "row_align": 512, "weight": 1}],
          "writer": {"members": [["jpg", null], ["cls", 4]]}}}"#;

    #[test]
    fn webdataset_files_are_cpythons_ustar_archives() {
        with_meta(&abstract_of(WDS), |meta, fmt| {
            let w = Writer::from_format("d", fmt).unwrap();
            w.check(meta, 2).unwrap();
            assert_eq!(meta.files(), 3);
            for file in 0..3 {
                let b = file_bytes(&w, meta, file, 1 << 16);
                assert_eq!(b.len() as i64, meta.file_size(file).unwrap());
                assert_eq!(b.len() % TAR_RECORD as usize, 0);
                assert_eq!(file_bytes(&w, meta, file, 333), b);
                let (mut at, mut logical) = (0usize, 0u64);
                let n = meta.samples_in_file(file).unwrap();
                for k in 0..n {
                    for (ext, size) in [("jpg", 1000usize), ("cls", 4)] {
                        let h = &b[at..at + 512];
                        let name = format!("{:09}.{ext}", file * 10 + k);
                        assert_eq!(&h[..name.len()], name.as_bytes());
                        assert_eq!(h[name.len()], 0);
                        assert_eq!(&h[124..136], format!("{size:011o}\0").as_bytes());
                        assert_eq!(h[156], b'0');
                        let sum: u64 = h.iter().enumerate().map(|(i, &x)| if (148..156).contains(&i) { b' ' as u64 } else { x as u64 }).sum();
                        assert_eq!(&h[148..156], format!("{sum:06o}\0 ").as_bytes());
                        let data = &b[at + 512..at + 512 + size];
                        if ext == "jpg" {
                            assert_eq!(data, payload(logical, 1000), "file {file} sample {k}");
                            logical += 1000;
                        } else {
                            assert!(data.iter().all(|&x| x == 0));
                        }
                        let padded = size.div_ceil(512) * 512;
                        assert!(b[at + 512 + size..at + 512 + padded].iter().all(|&x| x == 0), "padding");
                        at += 512 + padded;
                    }
                }
                // two zero blocks at least, then zeros to the record
                assert!(b.len() - at >= 1024);
                assert!(b[at..].iter().all(|&x| x == 0));
            }
        });
        // members whose sizes fill their blocks exactly: no padding item
        let exact = WDS.replace("\"count\": 25, \"samples_per_file\": 10", "\"count\": 9, \"samples_per_file\": 9")
            .replace("\"const\": 1000", "\"const\": 512").replace("[\"cls\", 4]", "[\"cls\", 512]");
        with_meta(&abstract_of(&exact), |meta, fmt| {
            let w = Writer::from_format("d", fmt).unwrap();
            w.check(meta, 1).unwrap();
            let b = file_bytes(&w, meta, 0, 4096);
            // 9 × (2 × 1024) + 1024 = 19456 → 20480
            assert_eq!((b.len() as i64, meta.file_size(0).unwrap()), (20480, 20480));
        });
    }

    const H5: &str = r#""layout": "compound", "pattern": "h/{id:04}.h5", "count": 10, "samples_per_file": 4,
        "size": {"const": 300}, "seed": 7, "access": "map",
        "format": {"class": "hdf5", "version": "test", "framing": {"file_header": 16}}"#;

    #[test]
    fn hdf5_files_are_the_header_then_the_rows() {
        with_meta(&abstract_of(H5), |meta, fmt| {
            let mut w = Writer::from_format("d", fmt).unwrap();
            // HDF5 frames nothing before the header exists: check passes at once
            w.check(meta, 2).unwrap();
            assert_eq!(w.header_rows(10, 4), [4, 2]);
            // without the short file's header, that file is refused, never a panic
            w.set_headers(BTreeMap::from([(4, vec![0xA4; 16])]));
            let mut filler = Filler::new(1);
            let mut s = Stream::new(&w, meta, 2, &seed_of, &mut filler).unwrap();
            let e = s.read(&mut [0u8; 64]).unwrap_err().to_string();
            assert_eq!(e, "no HDF5 header for a file of 2 rows");
            w.set_headers(BTreeMap::from([(4, vec![0xA4; 16]), (2, vec![0xA2; 16])]));
            for (file, rows, mark) in [(0, 4u64, 0xA4u8), (1, 4, 0xA4), (2, 2, 0xA2)] {
                let b = file_bytes(&w, meta, file, 100);
                assert_eq!(b.len() as i64, meta.file_size(file).unwrap());
                assert_eq!(b.len() as u64, 16 + 300 * rows);
                assert!(b[..16].iter().all(|&x| x == mark));
                assert_eq!(&b[16..], payload(0, 300 * rows));
            }
        });
    }

    /// Parquet over every column type: 4-row groups, two per full file, a short last file of one
    /// 2-row group; the page headers 19 bytes (fixed) and 21 (binary), the footer padded to 4000.
    const PQ: &str = r#""layout": "compound", "pattern": "p/{id:04}.parquet", "count": 10, "samples_per_file": 8,
        "size": {"const": 100}, "seed": 7, "access": "stream",
        "format": {"class": "parquet", "reader": "pyarrow", "version": "25.0.1", "framing": {
          "unit": 4, "file_header": 4, "file_footer": 4000,
          "columns": [{"header": 19, "fixed": 4, "weight": 0}, {"header": 19, "fixed": 4, "weight": 0}, {"header": 19, "fixed": 4, "weight": 0},
                      {"header": 21, "fixed": 4, "weight": 1}, {"header": 19, "fixed": 4, "weight": 0}, {"header": 19, "fixed": 4, "weight": 0},
                      {"header": 19, "fixed": 8, "weight": 0}, {"header": 19, "fixed": 8, "weight": 0}],
          "writer": {"rows_per_group": 4, "columns": [["a", "int8"], ["b", "uint8"], ["c", "int16"], ["i", "binary"], ["d", "float32"], ["e", "int32"], ["f", "float64"], ["g", "int64"]]}}}"#;

    #[test]
    fn parquet_files_of_every_column_type() {
        with_meta(&abstract_of(PQ), |meta, fmt| {
            let w = Writer::from_format("d", fmt).unwrap();
            let Writer::Parquet { columns } = &w else { panic!() };
            w.check(meta, 2).unwrap();
            for file in 0..meta.files() {
                let b = file_bytes(&w, meta, file, 4096);
                assert_eq!(b.len() as i64, meta.file_size(file).unwrap());
                assert_eq!(file_bytes(&w, meta, file, 13), b);
                assert_eq!(&b[..4], b"PAR1");
                assert_eq!(&b[b.len() - 4..], b"PAR1");
                assert_eq!(u32::from_le_bytes(b[b.len() - 8..b.len() - 4].try_into().unwrap()), 4000 - 8);
                // the first row group: each chunk its page header and rows, in the layout's places
                let rows = meta.samples_in_file(file).unwrap().min(4);
                let mut at = 4usize;
                let mut logical = 0u64;
                for (ci, (_, ty)) in columns.iter().enumerate() {
                    let data = if *ty == PqType::Binary { rows as u64 * 104 } else { ty.width() * rows as u64 };
                    let head = parquet_page_header(data, rows).unwrap();
                    assert_eq!(&b[at..at + head.len()], head, "column {ci}");
                    at += head.len();
                    for r in 0..rows {
                        if *ty == PqType::Binary {
                            assert_eq!(&b[at..at + 4], 100u32.to_le_bytes());
                            assert_eq!(&b[at + 4..at + 104], payload(logical, 100));
                            logical += 100;
                            at += 104;
                        } else {
                            let mut v = Vec::new();
                            ty.value(file * 8 + r, &mut v);
                            assert_eq!(&b[at..at + v.len()], v);
                            at += v.len();
                        }
                    }
                }
                assert_eq!(at as i64 - 4, meta.unit_lens(file).unwrap()[0]);
            }
        });
    }

    #[test]
    fn parquet_geometry_the_writer_cannot_meet_is_refused() {
        let refusal = |text: String| -> String {
            with_meta(&abstract_of(&text), |meta, fmt| format!("{:#}", Writer::from_format("d", fmt).unwrap().check(meta, 2).unwrap_err()))
        };
        // a footer smaller than the writer's
        let e = refusal(PQ.replace("\"file_footer\": 4000", "\"file_footer\": 200"));
        assert!(e.contains("file p/0000.parquet: the Parquet footer (") && e.contains("bytes) exceeds the layout's 192: the format class's footer margins are too small"), "{e}");
        // an 8-byte file header, where the writer's `PAR1` is 4: the file's size drifts
        let e = refusal(PQ.replace("\"file_header\": 4,", "\"file_header\": 8,"));
        assert!(e.contains("file p/0000.parquet: the writer made 5432 bytes, the layout predicts 5436: the format class's layout does not match its writer"), "{e}");
        // an int8 column of 5 bytes a row, where the writer's are 4: the second row group drifts
        let e = refusal(PQ.replacen("{\"header\": 19, \"fixed\": 4, \"weight\": 0}", "{\"header\": 19, \"fixed\": 5, \"weight\": 0}", 1));
        assert!(e.contains("file p/0000.parquet: row group 1 starts at ") && e.contains("the format class's layout does not match its writer"), "{e}");
        // a page header the layout puts outside the writer's varint band
        let e = refusal(PQ.replace("{\"header\": 21, \"fixed\": 4, \"weight\": 1}", "{\"header\": 22, \"fixed\": 4, \"weight\": 1}"));
        assert!(e.contains("row group 0 column 3: a 21-byte page header, the layout predicts 22"), "{e}");
        // a target below the natural footer, refused by the tail itself
        let tail = parquet_tail(&[("i".to_string(), PqType::Binary)], &[Chunk { at: 4, len: 30, rows: 1, raw: 5 }], 10);
        assert!(tail.unwrap_err().to_string().contains("exceeds the layout's 10"));
    }

    #[test]
    fn an_array_dataset_has_no_container() {
        let arr = abstract_of(r#""pattern": "d/one.bin", "layout": "array", "access": "indexed", "slot": 4, "count": 10, "size": {"const": 4}, "seed": 1"#);
        with_meta(&arr, |meta, _| {
            let w = Writer::TfRecord;
            // nothing to frame: the check passes, a stream is refused
            w.check(meta, 2).unwrap();
            let mut filler = Filler::new(1);
            let e = Stream::new(&w, meta, 0, &seed_of, &mut filler).err().unwrap().to_string();
            assert_eq!(e, "`d` is an array dataset: no container");
        });
    }
}
