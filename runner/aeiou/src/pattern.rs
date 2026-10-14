//! Path patterns: `train/{id div 1300:05}/img_{id:09}.jpg`, `kv/{conv:016x}/blk_{k:04}`,
//! `ckpt/step_{step:06}/{name}`. A pattern is parsed once; names are formatted from it and
//! never stored (`CLAUDE.md`: never materialize per-file data structures).
//!
//! Field syntax: `{name}`, `{name div N}`, `{name mod N}`, each with an optional `:format`
//! where the format is `[0][width][x]` (zero padding, minimum width, lowercase hex). The
//! divisor `N` is a positive number or, from contract 0.10, the name of an integer parameter
//! (`{id div files_per_directory:05}`, V23), bound to its value by `Pattern::bind` before any
//! name is formatted (`DESIGN_REVIEW.md` §3.75).

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    Lit(String),
    Field(Field),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    /// The operator and its divisor; the divisor is 0 while `param` names it unbound.
    pub op: Option<(FieldOp, i64)>,
    /// The parameter naming the divisor, until `Pattern::bind` replaces it with its value.
    pub param: Option<String>,
    /// The text between the braces, so a bound pattern's source is the original with each
    /// parameter's name replaced by its value.
    pub spec: String,
    pub zero: bool,
    pub width: usize,
    pub hex: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOp {
    Div,
    Mod,
}

impl FieldOp {
    pub fn word(self) -> &'static str {
        match self {
            FieldOp::Div => "div",
            FieldOp::Mod => "mod",
        }
    }
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

    /// The parameters naming divisors, in pattern order.
    pub fn params(&self) -> Vec<&str> {
        self.segments
            .iter()
            .filter_map(|s| match s {
                Segment::Field(Field { param: Some(p), .. }) => Some(p.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The pattern with every parameter divisor replaced by its value from `lookup`; its
    /// source is the original text with the names replaced, so a pattern without parameters
    /// binds to itself.
    pub fn bind(&self, lookup: impl Fn(&str) -> anyhow::Result<i64>) -> anyhow::Result<Pattern> {
        let mut segments = self.segments.clone();
        let mut source = String::with_capacity(self.source.len());
        for s in &mut segments {
            match s {
                Segment::Lit(l) => source.push_str(l),
                Segment::Field(f) => {
                    if let (Some(name), Some((op, _))) = (f.param.take(), f.op) {
                        let n = lookup(&name).map_err(|e| anyhow::anyhow!("pattern `{}`: divisor `{name}`: {e}", self.source))?;
                        if n <= 0 {
                            anyhow::bail!("pattern `{}`: `{} {name}` is {n}: it needs a positive divisor", self.source, op.word());
                        }
                        f.op = Some((op, n));
                        let fmt = f.spec.split_once(':').map(|(_, x)| format!(":{x}")).unwrap_or_default();
                        f.spec = format!("{} {} {n}{fmt}", f.name, op.word());
                    }
                    source.push('{');
                    source.push_str(&f.spec);
                    source.push('}');
                }
            }
        }
        Ok(Pattern { source, segments, dir_segments: self.dir_segments })
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
                    if let Some(p) = &f.param {
                        anyhow::bail!("pattern `{}`: divisor `{p}` is not bound to a value", self.source);
                    }
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
    let (name, op, param) = match words.as_slice() {
        [name] => (name.to_string(), None, None),
        [name, op, n] => {
            let op = match *op {
                "div" => FieldOp::Div,
                "mod" => FieldOp::Mod,
                _ => anyhow::bail!("pattern `{src}`: unknown field operator `{op}`"),
            };
            if n.starts_with(|c: char| c.is_ascii_lowercase() || c == '_') {
                if !n.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
                    anyhow::bail!("pattern `{src}`: bad parameter name `{n}` in `{{{spec}}}`");
                }
                (name.to_string(), Some((op, 0)), Some(n.to_string()))
            } else {
                let n: i64 = n.parse().map_err(|_| anyhow::anyhow!("pattern `{src}`: bad number in `{{{spec}}}`"))?;
                if n <= 0 {
                    anyhow::bail!("pattern `{src}`: `{} {n}` needs a positive divisor", op.word());
                }
                (name.to_string(), Some((op, n)), None)
            }
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
    Ok(Field { name, op, param, spec: spec.to_string(), zero, width, hex })
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

    #[test]
    fn a_parameter_divisor_binds_to_its_value() {
        let p = Pattern::parse("train/{id div files_per_directory:05}/img_{id:09}.jpg").unwrap();
        assert_eq!(p.params(), ["files_per_directory"]);
        assert!(p.format(id(1)).unwrap_err().to_string().contains("not bound"));
        let b = p.bind(|n| if n == "files_per_directory" { Ok(1300) } else { anyhow::bail!("unknown") }).unwrap();
        assert_eq!(b.source, "train/{id div 1300:05}/img_{id:09}.jpg");
        assert!(b.params().is_empty());
        assert_eq!(b.format(id(1300)).unwrap(), "train/00001/img_000001300.jpg");
        assert_eq!(b.dir_id_fields()[0].op, Some((FieldOp::Div, 1300)));
        // a pattern without parameters binds to itself, so its resolved text is unchanged
        let q = Pattern::parse("kv/{conv mod 16:x}/blk_{k:04}").unwrap();
        assert_eq!(q.bind(|_| unreachable!()).unwrap().source, q.source);
        let e = p.bind(|_| Ok(0)).unwrap_err().to_string();
        assert!(e.contains("`div files_per_directory` is 0"), "{e}");
        assert!(Pattern::parse("d/{id div Bad}/x").is_err());
    }
}
