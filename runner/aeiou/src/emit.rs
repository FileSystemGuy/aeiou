//! `aeiou dry-run --emit-trace FILE` (`DESIGN_REVIEW.md` §3.89, stage 4; `PROJECT_BRIEF.md` item
//! 49): the abstract's op stream written as the runner's trace file (`trace.rs`,
//! `runner/REFERENCE.md` §13), the run side of `aeiou-trace compare`, digested like the trace of
//! an application: no strace, none of the runner's own I/O, no dataset on disk.
//!
//! A **lane** is one context of the metrics walk (`metrics.rs`): an instance's main line, each of
//! its `parallel` sub-actors, each `loader` worker, each lane of a `trace` node, numbered in the
//! order their first ops are written. The instances are walked in turn, one op of each per
//! round (`metrics::interleave`), so lanes of different instances interleave as instances that
//! run at once would, and within an instance the order is the metrics walk's. The clock is that
//! order: the n-th line's `t` is n microseconds and every `dur` is 0; compute is not in it, so a
//! digest's think times say nothing. An open is given an id when it is written, and an op on a
//! path names the open its own lane made, else the latest open of the path in its instance.
//! Every op is written as it succeeds: a dry run has no errors to trace.

use std::collections::HashMap;
use std::io::Write;

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

use crate::ast::{Advice, IoctlRequest, Whence};
use crate::eval::Model;
use crate::vm::{Op, OpKind};

/// The open flags by their bit (`vm::flag_bits`), as the trace file spells them.
const FLAGS: [&str; 14] = ["RDONLY", "WRONLY", "RDWR", "CREAT", "TRUNC", "EXCL", "APPEND", "CLOEXEC", "DIRECTORY", "DIRECT", "SYNC", "DSYNC", "NOATIME", "NOFOLLOW"];

/// One op as the walk gave it, kept past the walk's borrow.
#[derive(Debug, Clone)]
pub struct Walked {
    pub lane: usize,
    pub kind: OpKind,
    pub path: String,
    pub path2: Option<String>,
    pub offset: i64,
    pub len: i64,
    pub bytes: i64,
    pub aux: u64,
    pub positioned: bool,
}

impl Walked {
    pub fn of(lane: usize, op: &Op) -> Walked {
        Walked {
            lane,
            kind: op.kind,
            path: op.path.to_string(),
            path2: op.path2.map(str::to_string),
            offset: op.offset,
            len: op.len,
            bytes: op.bytes,
            aux: op.aux,
            positioned: op.positioned,
        }
    }
}

/// What an emitted file holds, for its header.
#[derive(Debug, Default)]
pub struct Summary {
    pub lanes: usize,
    pub lines: usize,
    pub opens: usize,
    pub creates: Vec<String>,
}

#[derive(Default)]
struct Emitter {
    /// `(instance, lane in it)` → the file's lane.
    lanes: HashMap<(usize, usize), usize>,
    /// `(instance, path)` → the opens on it not yet closed, `(lane in the instance, id)`.
    held: HashMap<(usize, String), Vec<(usize, usize)>>,
    created: std::collections::HashSet<String>,
    s: Summary,
}

impl Emitter {
    fn lane(&mut self, inst: usize, lane: usize) -> usize {
        let n = self.lanes.len();
        *self.lanes.entry((inst, lane)).or_insert(n)
    }

    /// The open an op on `path` names: its own lane's latest, else its instance's latest; a close
    /// lets it go.
    fn open_of(&mut self, inst: usize, w: &Walked) -> Result<usize> {
        let held = self.held.get_mut(&(inst, w.path.clone())).filter(|h| !h.is_empty()).ok_or_else(|| anyhow!("{} of `{}` with no open of it", w.kind.name(), w.path))?;
        let at = held.iter().rposition(|(l, _)| *l == w.lane).unwrap_or(held.len() - 1);
        let id = held[at].1;
        if w.kind == OpKind::Close {
            held.remove(at);
        }
        Ok(id)
    }

    fn created(&mut self, path: &str) {
        if self.created.insert(path.to_string()) {
            self.s.creates.push(path.to_string());
        }
    }

    fn line(&mut self, inst: usize, w: &Walked) -> Result<Value> {
        let mut o = Map::new();
        let mut put = |k: &str, v: Value| {
            o.insert(k.to_string(), v);
        };
        put("op", json!(w.kind.name()));
        match w.kind {
            OpKind::Open => {
                let id = self.s.opens;
                self.s.opens += 1;
                self.held.entry((inst, w.path.clone())).or_default().push((w.lane, id));
                let flags: Vec<&str> = FLAGS.iter().enumerate().filter(|(i, _)| w.aux & (1 << i) != 0).map(|(_, f)| *f).collect();
                if flags.contains(&"CREAT") {
                    self.created(&w.path);
                }
                put("fd", json!(id));
                put("path", json!(w.path));
                put("flags", json!(flags));
                if w.aux >> 32 != 0 {
                    put("mode", json!(w.aux >> 32));
                }
            }
            OpKind::Stat | OpKind::Unlink | OpKind::Rmdir => put("path", json!(w.path)),
            OpKind::Mkdir => {
                self.created(&w.path);
                put("path", json!(w.path));
                put("mode", json!(w.aux));
            }
            OpKind::Rename => {
                put("path", json!(w.path));
                put("to", json!(w.path2.clone().unwrap_or_default()));
            }
            _ => {
                put("fd", json!(self.open_of(inst, w)?));
                match w.kind {
                    OpKind::Read | OpKind::Write => {
                        if w.positioned {
                            put("offset", json!(w.offset));
                        }
                        put("len", json!(w.len));
                        put("ret", json!(w.bytes));
                    }
                    OpKind::Lseek => {
                        put("offset", json!(w.offset));
                        put("whence", json!(format!("{:?}", Whence::from_code(w.aux))));
                        put("ret", json!(w.bytes));
                    }
                    OpKind::Readdir if w.bytes >= 0 => put("ret", json!(w.bytes)),
                    OpKind::Ioctl => put("request", json!(format!("{:?}", IoctlRequest::from_code(w.aux)))),
                    OpKind::Fadvise => {
                        put("offset", json!(w.offset));
                        put("len", json!(w.len));
                        put("advice", json!(format!("{:?}", Advice::from_code(w.aux))));
                    }
                    OpKind::Ftruncate => put("len", json!(w.len)),
                    OpKind::Fallocate => {
                        put("offset", json!(w.offset));
                        put("len", json!(w.len));
                    }
                    _ => {}
                }
            }
        }
        let lane = self.lane(inst, w.lane);
        o.insert("lane".into(), json!(lane));
        o.insert("t".into(), json!(self.s.lines as i64 * 1000));
        o.insert("dur".into(), json!(0));
        self.s.lines += 1;
        Ok(Value::Object(o))
    }
}

/// Walk the model's every instance and write its ops to `out` as trace lines, without the
/// header (whose counts are known only at the end).
pub fn lines(model: &Model<'_>, out: &mut dyn Write) -> Result<Summary> {
    let mut e = Emitter::default();
    crate::metrics::interleave(model, |inst, sink| {
        for w in sink.emitted.as_mut().map(std::mem::take).unwrap_or_default() {
            let line = e.line(inst, &w)?;
            serde_json::to_writer(&mut *out, &line)?;
            out.write_all(b"\n")?;
        }
        Ok(())
    })?;
    e.s.lanes = e.lanes.len();
    Ok(e.s)
}

/// The header of an emitted file: the trace file's fields, and the dry run's identity, which
/// the digest's `metrics` carry as a dry run's metrics document does.
pub fn header(s: &Summary, identity: &Map<String, Value>) -> Value {
    let mut h = json!({
        "aeiou_trace": crate::trace::FORMAT,
        "source": "dry-run",
        "root": "",
        "lanes": s.lanes,
        "lines": s.lines,
        "opens": s.opens,
        "creates": s.creates,
        "notes": {},
        "clock": "the walk's order, one microsecond per line; no compute",
    });
    for (k, v) in identity {
        h[k] = v.clone();
    }
    h
}

/// The header as one line, `aeiou_trace` its first key: what the tools look for to know a trace
/// file from an strace (the other keys follow in serde's order).
pub fn header_line(h: &Value) -> String {
    let mut rest = h.as_object().cloned().unwrap_or_default();
    let version = rest.remove("aeiou_trace").unwrap_or(Value::Null);
    let rest = serde_json::to_string(&Value::Object(rest)).expect("a header serializes");
    let rest = rest.strip_prefix('{').unwrap_or(&rest);
    if rest == "}" { format!("{{\"aeiou_trace\":{version}}}\n") } else { format!("{{\"aeiou_trace\":{version},{rest}\n") }
}

/// Write the emitted trace file at `path`: the lines beside it first, then the header and the
/// lines renamed into place, so `path` is never a partial file.
pub fn write(model: &Model<'_>, path: &std::path::Path, identity: &Map<String, Value>) -> Result<Summary> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let body = path.with_file_name(format!(".{name}.aeiou-emit-{}", std::process::id()));
    let whole = path.with_file_name(format!(".{name}.aeiou-emit-{}.whole", std::process::id()));
    let result = (|| -> Result<Summary> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&body).map_err(|e| anyhow!("{}: {e}", path.display()))?);
        let s = lines(model, &mut w)?;
        w.flush()?;
        drop(w);
        let mut out = std::io::BufWriter::new(std::fs::File::create(&whole).map_err(|e| anyhow!("{}: {e}", path.display()))?);
        out.write_all(header_line(&header(&s, identity)).as_bytes())?;
        std::io::copy(&mut std::fs::File::open(&body)?, &mut out)?;
        out.flush()?;
        std::fs::rename(&whole, path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        Ok(s)
    })();
    let _ = std::fs::remove_file(&body);
    let _ = std::fs::remove_file(&whole);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(lane: usize, kind: OpKind, path: &str) -> Walked {
        Walked { lane, kind, path: path.into(), path2: None, offset: 0, len: 0, bytes: 0, aux: 0, positioned: false }
    }

    /// Every kind of op the VM issues, written as a line the runner's trace loader reads back as
    /// that op (`trace::Line`), with what the file's header says it made.
    #[test]
    fn every_op_kind_is_a_line_the_trace_loader_reads() {
        let mut e = Emitter::default();
        let creat = (1u64 << 1) | (1 << 3) | (0o644 << 32);
        let ops = vec![
            w(0, OpKind::Mkdir, "out").with(|o| o.aux = 0o755),
            w(0, OpKind::Open, "out/a").with(|o| o.aux = creat),
            Walked { offset: 4096, len: 100, bytes: 100, positioned: true, ..w(0, OpKind::Write, "out/a") },
            Walked { len: 100, bytes: 100, ..w(0, OpKind::Write, "out/a") },
            Walked { offset: 10, bytes: 10, aux: 0, ..w(0, OpKind::Lseek, "out/a") },
            Walked { offset: 0, len: 8192, aux: 3, ..w(0, OpKind::Fadvise, "out/a") },
            Walked { len: 4096, ..w(0, OpKind::Ftruncate, "out/a") },
            Walked { offset: 0, len: 8192, ..w(0, OpKind::Fallocate, "out/a") },
            w(0, OpKind::Ioctl, "out/a").with(|o| o.aux = 1),
            w(0, OpKind::Fsync, "out/a"),
            w(0, OpKind::Fdatasync, "out/a"),
            w(0, OpKind::Fstat, "out/a"),
            w(0, OpKind::Close, "out/a"),
            w(1, OpKind::Open, "out"),
            Walked { bytes: 1, ..w(1, OpKind::Readdir, "out") },
            Walked { bytes: -1, ..w(1, OpKind::Readdir, "out") },
            w(1, OpKind::Close, "out"),
            w(0, OpKind::Stat, "out/a"),
            Walked { path2: Some("out/b".into()), ..w(0, OpKind::Rename, "out/a") },
            w(0, OpKind::Unlink, "out/b"),
            w(0, OpKind::Rmdir, "out"),
        ];
        let lines: Vec<Value> = ops.iter().map(|o| e.line(0, o).unwrap()).collect();
        for (o, l) in ops.iter().zip(&lines) {
            let _: crate::trace::Line = serde_json::from_value(l.clone()).unwrap_or_else(|err| panic!("{l}: {err}"));
            assert_eq!(l["op"], o.kind.name(), "{l}");
        }
        assert_eq!(lines[1]["flags"], json!(["WRONLY", "CREAT"]));
        assert_eq!(lines[1]["mode"], json!(0o644));
        assert_eq!((lines[2]["offset"].as_i64(), lines[3].get("offset")), (Some(4096), None));
        assert_eq!((lines[4]["whence"].as_str(), lines[4]["ret"].as_i64()), (Some("SET"), Some(10)));
        assert_eq!((lines[5]["advice"].as_str(), lines[8]["request"].as_str()), (Some("WILLNEED"), Some("FIONREAD")));
        assert_eq!((lines[14]["ret"].as_i64(), lines[15].get("ret")), (Some(1), None));
        assert_eq!(lines[18]["to"], "out/b");
        assert_eq!(e.s.creates, vec!["out".to_string(), "out/a".to_string()]);
        assert_eq!((e.s.opens, e.lanes.len(), e.s.lines), (2, 2, ops.len()));
        // an op on a path its instance has not opened is the walk's error, named
        let err = e.line(0, &w(0, OpKind::Read, "nowhere")).unwrap_err();
        assert_eq!(err.to_string(), "read of `nowhere` with no open of it");
    }

    #[test]
    fn an_op_names_its_own_lanes_open_else_its_instances_latest() {
        let mut e = Emitter::default();
        for (lane, kind) in [(0, OpKind::Open), (1, OpKind::Open)] {
            e.line(0, &w(lane, kind, "f")).unwrap();
        }
        assert_eq!(e.line(0, &w(0, OpKind::Fstat, "f")).unwrap()["fd"], 0);
        assert_eq!(e.line(0, &w(2, OpKind::Fstat, "f")).unwrap()["fd"], 1);
        assert_eq!(e.line(0, &w(1, OpKind::Close, "f")).unwrap()["fd"], 1);
        assert_eq!(e.line(0, &w(2, OpKind::Fstat, "f")).unwrap()["fd"], 0);
        // another instance's opens are its own
        assert!(e.line(1, &w(0, OpKind::Fstat, "f")).is_err());
    }

    #[test]
    fn the_header_line_begins_with_its_version() {
        let line = header_line(&json!({"abstract": "x", "aeiou_trace": 1}));
        assert_eq!(line, "{\"aeiou_trace\":1,\"abstract\":\"x\"}\n");
        assert_eq!(header_line(&json!({"aeiou_trace": 1})), "{\"aeiou_trace\":1}\n");
    }

    trait With: Sized {
        fn with(self, f: impl FnOnce(&mut Self)) -> Self;
    }
    impl With for Walked {
        fn with(mut self, f: impl FnOnce(&mut Self)) -> Self {
            f(&mut self);
            self
        }
    }
}
