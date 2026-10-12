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
//! Parquet is still the Python writer's (`aeiou-datagen`).

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
}

impl Writer {
    /// The writer of a resolved `format` entry (`schema/README.md` §2), or why there is none.
    pub fn from_format(dataset: &str, format: &Value) -> Result<Writer> {
        let class = format["class"].as_str().unwrap_or("");
        let settings = &format["layout"]["writer"];
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
            "parquet" => bail!("dataset `{dataset}` is a `{class}` container: the Python writer writes it (`aeiou-datagen`; builder/REFERENCE.md §6), not yet `aeiou datagen`"),
            _ => bail!("dataset `{dataset}`: no writer for format class `{class}`"),
        }
    }

    /// What the manifest's `format.writer` names: this writer, whose bytes the id covers.
    pub fn library(&self) -> String {
        match self {
            Writer::TfRecord => "aeiou datagen tfrecord writer".into(),
            Writer::WebDataset { .. } => "aeiou datagen ustar writer".into(),
            Writer::Hdf5 { version, .. } => format!("aeiou datagen hdf5 writer (header: aeiou-header, {version})"),
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
    let each = format["layout"]["file_header"].as_u64().ok_or_else(|| anyhow!("the layout has no file_header"))? as usize;
    if out.stdout.len() != each * rows.len() {
        bail!("{} gave {} bytes for {} header(s) of {each} bytes, the layout's file_header", tool.display(), out.stdout.len(), rows.len());
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

/// `n` in octal, `digits - 1` digits and a NUL (`tarfile.itn` for ustar).
fn octal(out: &mut [u8], n: u64) {
    let s = format!("{:0w$o}", n, w = out.len() - 1);
    out[..s.len()].copy_from_slice(s.as_bytes());
    out[s.len()] = 0;
}

/// The ustar header of a regular member, as `tarfile.TarInfo.tobuf(USTAR_FORMAT)` makes it.
fn tar_header(name: &str, size: u64) -> Result<[u8; 512]> {
    let mut h = [0u8; 512];
    if name.len() > 100 {
        bail!("tar member name `{name}` is longer than 100 bytes");
    }
    h[..name.len()].copy_from_slice(name.as_bytes());
    octal(&mut h[100..108], 0o644);
    octal(&mut h[108..116], 0);
    octal(&mut h[116..124], 0);
    octal(&mut h[124..136], size);
    octal(&mut h[136..148], 0);
    h[148..156].fill(b' ');
    h[156] = b'0';
    h[257..265].copy_from_slice(b"ustar\x0000");
    let sum: u64 = h.iter().map(|&b| b as u64).sum();
    let s = format!("{sum:06o}\0");
    h[148..148 + s.len()].copy_from_slice(s.as_bytes());
    Ok(h)
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
            DsMeta::Regions { .. } => bail!("`{}` is a regions dataset: no container", meta.name()),
        };
        let n = meta.samples_in_file(file)?;
        Ok(Stream { writer, meta, seed_of, filler, first: file * spf, n, sample: 0, member: 0, small: Vec::with_capacity(TAR_RECORD as usize), small_at: 0, pay_left: 0, pay_at: 0, crc: 0, after: After::Nothing, pos: 0, ended: false, done: false })
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
    }
}
