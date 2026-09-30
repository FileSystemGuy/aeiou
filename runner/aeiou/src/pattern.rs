//! Path patterns: `train/{id div 1300:05}/img_{id:09}.jpg`, `kv/{conv:016x}/blk_{k:04}`,
//! `ckpt/step_{step:06}/{name}`. A pattern is parsed once; names are formatted from it and
//! never stored (`CLAUDE.md`: never materialize per-file data structures).
//!
//! Field syntax: `{name}`, `{name div N}`, `{name mod N}`, each with an optional `:format`
//! where the format is `[0][width][x]` (zero padding, minimum width, lowercase hex).

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    Lit(String),
    Field(Field),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub op: Option<(FieldOp, i64)>,
    pub zero: bool,
    pub width: usize,
    pub hex: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOp {
    Div,
    Mod,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub source: String,
    pub segments: Vec<Segment>,
    /// Segments up to and including the last `/`, if any: the directory part.
    pub dir_segments: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum FieldValue<'a> {
    Int(i64),
    Str(&'a str),
}

impl Pattern {
    pub fn parse(src: &str) -> anyhow::Result<Pattern> {
        let mut segments = Vec::new();
        let mut lit = String::new();
        let mut rest = src;
        while let Some(open) = rest.find('{') {
            lit.push_str(&rest[..open]);
            let after = &rest[open + 1..];
            let close = after.find('}').ok_or_else(|| anyhow::anyhow!("pattern `{src}`: unclosed `{{`"))?;
            if !lit.is_empty() {
                segments.push(Segment::Lit(std::mem::take(&mut lit)));
            }
            segments.push(Segment::Field(parse_field(&after[..close], src)?));
            rest = &after[close + 1..];
        }
        lit.push_str(rest);
        if !lit.is_empty() {
            segments.push(Segment::Lit(lit));
        }
        // the directory part ends at the last `/` in a literal segment
        let mut dir_segments = 0;
        for (i, s) in segments.iter().enumerate() {
            if let Segment::Lit(l) = s {
                if l.contains('/') {
                    dir_segments = i + 1;
                }
            }
        }
        Ok(Pattern { source: src.to_string(), segments, dir_segments })
    }

    pub fn field_names(&self) -> Vec<&str> {
        self.segments
            .iter()
            .filter_map(|s| match s {
                Segment::Field(f) => Some(f.name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The constant directory prefix (`train/` for `train/{id div 1300:05}/…`), without the
    /// trailing slash; empty when the pattern starts with a field or has no directory.
    pub fn root(&self) -> &str {
        let prefix = self.source.split('{').next().unwrap_or("");
        match prefix.rfind('/') {
            Some(i) => &prefix[..i],
            None => "",
        }
    }

    /// Format the whole pattern with a field lookup.
    pub fn format<'a>(&self, lookup: impl Fn(&str) -> Option<FieldValue<'a>>) -> anyhow::Result<String> {
        self.format_segments(&self.segments, lookup)
    }

    /// Format the directory part only (no trailing slash).
    pub fn format_dir<'a>(&self, lookup: impl Fn(&str) -> Option<FieldValue<'a>>) -> anyhow::Result<String> {
        let mut s = self.format_segments(&self.segments[..self.dir_segments], lookup)?;
        // the last literal segment ends with the directory's trailing slash plus a file-name prefix
        if let Some(i) = s.rfind('/') {
            s.truncate(i);
        }
        Ok(s)
    }

    fn format_segments<'a>(&self, segs: &[Segment], lookup: impl Fn(&str) -> Option<FieldValue<'a>>) -> anyhow::Result<String> {
        let mut out = String::with_capacity(self.source.len() + 16);
        for s in segs {
            match s {
                Segment::Lit(l) => out.push_str(l),
                Segment::Field(f) => {
                    let v = lookup(&f.name).ok_or_else(|| anyhow::anyhow!("pattern `{}`: no value for field `{}`", self.source, f.name))?;
                    match v {
                        FieldValue::Str(s) => out.push_str(s),
                        FieldValue::Int(mut n) => {
                            if let Some((op, m)) = f.op {
                                n = match op {
                                    FieldOp::Div => n.div_euclid(m),
                                    FieldOp::Mod => n.rem_euclid(m),
                                };
                            }
                            if f.hex {
                                if f.zero {
                                    write!(out, "{:0width$x}", n as u64, width = f.width).unwrap();
                                } else {
                                    write!(out, "{:width$x}", n as u64, width = f.width).unwrap();
                                }
                            } else if f.zero {
                                write!(out, "{:0width$}", n, width = f.width).unwrap();
                            } else {
                                write!(out, "{:width$}", n, width = f.width).unwrap();
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The `id` fields of the directory part, for `dirs` and `dir` handles.
    pub fn dir_id_fields(&self) -> Vec<&Field> {
        self.segments[..self.dir_segments]
            .iter()
            .filter_map(|s| match s {
                Segment::Field(f) if f.name == "id" => Some(f),
                _ => None,
            })
            .collect()
    }
}

fn parse_field(spec: &str, src: &str) -> anyhow::Result<Field> {
    let (expr, fmt) = match spec.split_once(':') {
        Some((e, f)) => (e.trim(), f.trim()),
        None => (spec.trim(), ""),
    };
    let words: Vec<&str> = expr.split_whitespace().collect();
    let (name, op) = match words.as_slice() {
        [name] => (name.to_string(), None),
        [name, op, n] => {
            let n: i64 = n.parse().map_err(|_| anyhow::anyhow!("pattern `{src}`: bad number in `{{{spec}}}`"))?;
            if n <= 0 {
                anyhow::bail!("pattern `{src}`: `{op} {n}` needs a positive divisor");
            }
            let op = match *op {
                "div" => FieldOp::Div,
                "mod" => FieldOp::Mod,
                _ => anyhow::bail!("pattern `{src}`: unknown field operator `{op}`"),
            };
            (name.to_string(), Some((op, n)))
        }
        _ => anyhow::bail!("pattern `{src}`: cannot parse field `{{{spec}}}`"),
    };
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        anyhow::bail!("pattern `{src}`: bad field name `{name}`");
    }
    let mut zero = false;
    let mut hex = false;
    let mut digits = fmt;
    if let Some(stripped) = digits.strip_prefix('0') {
        zero = true;
        digits = stripped;
    }
    if let Some(stripped) = digits.strip_suffix('x') {
        hex = true;
        digits = stripped;
    } else if let Some(stripped) = digits.strip_suffix('d') {
        digits = stripped;
    }
    let width = if digits.is_empty() {
        0
    } else {
        digits.parse().map_err(|_| anyhow::anyhow!("pattern `{src}`: bad format `{fmt}` in `{{{spec}}}`"))?
    };
    Ok(Field { name, op, zero, width, hex })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: i64) -> impl Fn(&str) -> Option<FieldValue<'static>> {
        move |f| if f == "id" { Some(FieldValue::Int(n)) } else { None }
    }

    #[test]
    fn formats_like_the_builder_examples() {
        let p = Pattern::parse("train/{id div 1300:05}/img_{id:09}.jpg").unwrap();
        assert_eq!(p.format(id(1300)).unwrap(), "train/00001/img_000001300.jpg");
        assert_eq!(p.format_dir(id(2600)).unwrap(), "train/00002");
        assert_eq!(p.root(), "train");
        let p = Pattern::parse("kv/{conv:016x}/blk_{k:04}").unwrap();
        let s = p
            .format(|f| match f {
                "conv" => Some(FieldValue::Int(-1)),
                "k" => Some(FieldValue::Int(7)),
                _ => None,
            })
            .unwrap();
        assert_eq!(s, "kv/ffffffffffffffff/blk_0007");
        let p = Pattern::parse("ckpt/step_{step:06}/{name}").unwrap();
        let s = p
            .format(|f| match f {
                "step" => Some(FieldValue::Int(100)),
                "name" => Some(FieldValue::Str(".metadata")),
                _ => None,
            })
            .unwrap();
        assert_eq!(s, "ckpt/step_000100/.metadata");
        let p = Pattern::parse("model-{id:05}-of-00002.safetensors").unwrap();
        assert_eq!(p.dir_segments, 0);
        assert_eq!(p.root(), "");
        assert_eq!(Pattern::parse("d/{id mod 16}/x").unwrap().format(id(17)).unwrap(), "d/1/x");
    }
}
