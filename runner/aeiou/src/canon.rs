//! Canonical form and identity of an AST (`schema/README.md` §1): the JSON with keys sorted,
//! no whitespace, ASCII escapes, floats in Python `repr`, and the `provenance` block removed;
//! the identity is the SHA-256 of those bytes. This must produce byte-identical output to
//! `schema/check.py::canonical`, which uses `json.dumps(sort_keys=True, separators=(",", ":"),
//! ensure_ascii=True)`; the tests compare the nine committed ASTs' hashes.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Canonical bytes of a parsed JSON document (the top-level `provenance` key removed).
pub fn canonical(doc: &Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(4096);
    match doc {
        Value::Object(map) => {
            out.push(b'{');
            let mut first = true;
            for (k, v) in sorted(map) {
                if k == "provenance" {
                    continue;
                }
                if !first {
                    out.push(b',');
                }
                first = false;
                write_str(&mut out, k);
                out.push(b':');
                write_value(&mut out, v);
            }
            out.push(b'}');
        }
        other => write_value(&mut out, other),
    }
    out
}

/// SHA-256 of the canonical form, lowercase hex.
pub fn sha256_hex(doc: &Value) -> String {
    let digest = Sha256::digest(canonical(doc));
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn sorted(map: &serde_json::Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut v: Vec<_> = map.iter().collect();
    // Python sorts str keys by code point; JSON object keys here are identifiers, but sort
    // by bytes anyway, which agrees with code-point order for UTF-8.
    v.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    v
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.extend_from_slice(i.to_string().as_bytes());
            } else if let Some(u) = n.as_u64() {
                out.extend_from_slice(u.to_string().as_bytes());
            } else {
                out.extend_from_slice(python_float_repr(n.as_f64().expect("finite float")).as_bytes());
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, x);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            for (i, (k, x)) in sorted(map).into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write_value(out, x);
            }
            out.push(b'}');
        }
    }
}

/// `json.dumps(ensure_ascii=True)` string escaping: `\" \\ \n \r \t \b \f`, every other code
/// point outside `0x20..=0x7e` as `\uXXXX` (surrogate pairs above the BMP).
fn write_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            ' '..='~' => out.push(c as u8),
            _ => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.extend_from_slice(format!("\\u{:04x}", unit).as_bytes());
                }
            }
        }
    }
    out.push(b'"');
}

/// Python's `repr(float)`: the shortest round-trip digits, fixed notation when the decimal
/// exponent is in `-4 < decpt <= 16`, else `d.ddde±XX`; a fixed form always has a `.` and at
/// least one fractional digit.
pub fn python_float_repr(x: f64) -> String {
    assert!(x.is_finite(), "canonical form has no NaN or infinity (allow_nan=False)");
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` prints the shortest round-trip digits as d[.ddd]e[-]x.
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent digits");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.d1d2... × 10^decpt
    let sign = if x < 0.0 { "-" } else { "" };
    let n = digits.len() as i32;
    if -4 < decpt && decpt <= 16 {
        let body = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
        } else {
            let (a, b) = digits.split_at(decpt as usize);
            format!("{}.{}", a, b)
        };
        format!("{}{}", sign, body)
    } else {
        let e = decpt - 1;
        let mant = if n == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        let esign = if e < 0 { '-' } else { '+' };
        format!("{}{}e{}{:02}", sign, mant, esign, e.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [
            (0.45, "0.45"),
            (1.0, "1.0"),
            (1.1, "1.1"),
            (0.3, "0.3"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1234.5, "1234.5"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (-2.5, "-2.5"),
            (0.1 + 0.2, "0.30000000000000004"),
        ] {
            assert_eq!(python_float_repr(x), want, "{x:?}");
        }
    }

    #[test]
    fn ascii_escapes_match_python() {
        let mut out = Vec::new();
        write_str(&mut out, "a\"b\\c\n\u{7f}é😀");
        assert_eq!(String::from_utf8(out).unwrap(), "\"a\\\"b\\\\c\\n\\u007f\\u00e9\\ud83d\\ude00\"");
    }
}
