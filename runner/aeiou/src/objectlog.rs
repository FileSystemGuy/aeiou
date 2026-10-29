//! Object-store request logs read as traces (`DESIGN_REVIEW.md` §3.88 step 5, §3.89). An strace
//! never shows an application's object requests (TLS, or another process), but the store's own
//! log does: one line per request with its time, client, operation, bucket, key, bytes, status,
//! and duration. Each request becomes the calls the runner's object engine issues for it
//! (`runner/REFERENCE.md` §16), through the exporter of `export.rs`, so the trace file, the
//! metrics, and the runner's `trace` node take it unchanged: a GET is an open, one read (at the
//! Range's start when the log has it), and a close; a PUT an open that creates, one write, and
//! a close; a HEAD a stat; a DELETE an unlink; a listing an open of the prefix, a readdir, and a
//! close; a multipart upload an open at its start, a write per part, and a close at its
//! completion. A lane is a client host, since the log knows no thread. The root is
//! `BUCKET[/PREFIX]`; keys outside it are left out and counted.
//!
//! Two formats, told apart by the first line: MinIO's audit log (JSON lines with an `api`
//! object; `time` taken as when the response completed, so the request starts
//! `timeToResponse` earlier; the `Range` header gives a GET's offset) and Amazon S3's server
//! access log (space-delimited, to the second, no Range, so a GET that sent fewer bytes than
//! the object holds is read from 0 and counted under `ranges_unknown`). Ceph's RGW ops log and
//! a lab proxy's log are not read yet (`PROJECT_BRIEF.md` item 49).

use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::export::{self, Event, Exporter, Opener, Summary};
use crate::strace::{for_each_line, Seen, When};

pub const FORMATS: [&str; 2] = ["minio-audit", "s3-access"];

const GET: [&str; 2] = ["GetObject", "REST.GET.OBJECT"];
const PUT: [&str; 2] = ["PutObject", "REST.PUT.OBJECT"];
const HEAD: [&str; 2] = ["HeadObject", "REST.HEAD.OBJECT"];
const DELETE: [&str; 2] = ["DeleteObject", "REST.DELETE.OBJECT"];
const LIST: [&str; 4] = ["ListObjectsV2", "ListObjectsV1", "ListObjects", "REST.GET.BUCKET"];
const MP_NEW: [&str; 2] = ["NewMultipartUpload", "REST.POST.UPLOADS"];
const MP_PART: [&str; 2] = ["PutObjectPart", "REST.PUT.PART"];
const MP_DONE: [&str; 2] = ["CompleteMultipartUpload", "REST.POST.UPLOAD"];
const MP_ABORT: [&str; 2] = ["AbortMultipartUpload", "REST.DELETE.UPLOAD"];
const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

fn errno(status: i64) -> &'static str {
    match status {
        404 => "ENOENT",
        403 => "EACCES",
        409 | 412 => "EEXIST",
        _ => "EIO",
    }
}

fn all_digits(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|c| c.is_ascii_digit())
}

/// The S3 access log's first fields: `[0-9a-f]{64} \S+ [dd/Mon/yyyy:hh:mm:ss +zzzz] `.
fn s3_first(s: &str) -> bool {
    let Some((owner, rest)) = s.split_once(' ') else { return false };
    if owner.len() != 64 || !owner.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return false;
    }
    let Some((bucket, rest)) = rest.split_once(' ') else { return false };
    if bucket.is_empty() || !rest.starts_with('[') {
        return false;
    }
    let Some(end) = rest.find(']') else { return false };
    rest[end + 1..].starts_with(' ') && access_seconds(&rest[1..end]).is_some()
}

/// The log format of a first line, or `None` when it is not an object log.
pub fn source_of(first: &str) -> Option<&'static str> {
    let s = first.trim();
    if s.starts_with('{') && s.contains("\"api\"") {
        let doc: Value = serde_json::from_str(s).ok()?;
        return doc.get("api").filter(|a| a.is_object()).map(|_| "minio-audit");
    }
    s3_first(s).then_some("s3-access")
}

/// Days from 1970-01-01 to a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn timegm(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> i64 {
    days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s
}

/// Seconds since the epoch of an RFC 3339 time, nanoseconds kept, or `None`.
pub fn iso_seconds(text: &str) -> Option<f64> {
    let t = text.trim();
    let b = t.as_bytes();
    if b.len() < 19 || !all_digits(&t[0..4], 4) || b[4] != b'-' || !all_digits(&t[5..7], 2) || b[7] != b'-' || !all_digits(&t[8..10], 2) {
        return None;
    }
    if !(b[10] == b'T' || b[10] == b' ') || !all_digits(&t[11..13], 2) || b[13] != b':' || !all_digits(&t[14..16], 2) || b[16] != b':' || !all_digits(&t[17..19], 2) {
        return None;
    }
    let num = |a: usize, z: usize| t[a..z].parse::<i64>().unwrap();
    let mut out = timegm(num(0, 4), num(5, 7), num(8, 10), num(11, 13), num(14, 16), num(17, 19)) as f64;
    let mut rest = &t[19..];
    if let Some(r) = rest.strip_prefix('.') {
        let k = r.bytes().position(|c| !c.is_ascii_digit()).unwrap_or(r.len());
        if k == 0 {
            return None;
        }
        let frac = &r[..k.min(9)];
        out += format!("{frac:0<9}").parse::<i64>().unwrap() as f64 / 1e9;
        rest = &r[k..];
    }
    match rest {
        "" | "Z" => {}
        tz => {
            let sign = match tz.as_bytes()[0] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            let z = &tz[1..];
            let (hh, mm) = if z.len() == 4 && all_digits(z, 4) {
                (&z[0..2], &z[2..4])
            } else if z.len() == 5 && z.as_bytes()[2] == b':' && all_digits(&z[0..2], 2) && all_digits(&z[3..5], 2) {
                (&z[0..2], &z[3..5])
            } else {
                return None;
            };
            out -= (sign * (hh.parse::<i64>().unwrap() * 3600 + mm.parse::<i64>().unwrap() * 60)) as f64;
        }
    }
    Some(out)
}

/// Seconds since the epoch of an access log's `06/Feb/2019:00:00:38 +0000`.
pub fn access_seconds(text: &str) -> Option<f64> {
    let t = text.trim();
    let b = t.as_bytes();
    if b.len() != 26 || b[2] != b'/' || b[6] != b'/' || b[11] != b':' || b[14] != b':' || b[17] != b':' || b[20] != b' ' {
        return None;
    }
    let ok = all_digits(&t[0..2], 2) && all_digits(&t[7..11], 4) && all_digits(&t[12..14], 2) && all_digits(&t[15..17], 2) && all_digits(&t[18..20], 2) && all_digits(&t[22..26], 4);
    let mon = MONTHS.iter().position(|m| *m == &t[3..6])? as i64 + 1;
    if !ok || !(b[21] == b'+' || b[21] == b'-') {
        return None;
    }
    let n = |a: usize, z: usize| t[a..z].parse::<i64>().unwrap();
    let base = timegm(n(7, 11), mon, n(0, 2), n(12, 14), n(15, 17), n(18, 20));
    let sign = if b[21] == b'+' { 1 } else { -1 };
    Some((base - sign * (n(22, 24) * 3600 + n(24, 26) * 60)) as f64)
}

/// One request, whatever the log: what the converter needs of it.
#[derive(Debug, Clone, Default)]
struct Request {
    op: String,
    bucket: String,
    key: String,
    status: i64,
    rx: i64,
    tx: i64,
    size: Option<i64>,
    start: Option<f64>,
    end: Option<f64>,
    host: String,
    offset: Option<i64>,
    length: Option<i64>,
    prefix: Option<String>,
}

/// Python's `int(v)` with a default: a number (truncated), a numeric string, a bool.
fn int_of(v: Option<&Value>, default: i64) -> i64 {
    match v {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(default),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => default,
    }
}

fn int_text(s: &str) -> i64 {
    s.trim().parse().unwrap_or(0)
}

fn minio_request(doc: &Value, no: usize) -> Result<Request> {
    let api = doc.get("api").filter(|a| a.is_object()).ok_or_else(|| anyhow!("line {no}: no api.name"))?;
    let name = api.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("line {no}: no api.name"))?;
    let end = doc.get("time").and_then(Value::as_str).and_then(iso_seconds);
    let ns = int_of(api.get("timeToResponseInNS"), 0);
    let start = end.map(|e| e - ns as f64 / 1e9);
    let (mut offset, mut length) = (None, None);
    if let Some(rng) = doc.get("requestHeader").and_then(|h| h.get("Range")).and_then(Value::as_str) {
        if let Some(r) = rng.trim().strip_prefix("bytes=") {
            if let Some((a, z)) = r.split_once('-') {
                if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && z.bytes().all(|c| c.is_ascii_digit()) {
                    let o: i64 = a.parse()?;
                    offset = Some(o);
                    length = if z.is_empty() { None } else { Some(z.parse::<i64>()? - o + 1) };
                }
            }
        }
    }
    let q = doc.get("requestQuery");
    let text = |v: Option<&Value>| v.and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    let host = match doc.get("remotehost") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::Number(n)) if n.as_f64() != Some(0.0) => n.to_string(),
        _ => "?".into(),
    };
    Ok(Request {
        op: name.to_string(),
        bucket: text(api.get("bucket")).unwrap_or_default(),
        key: text(api.get("object")).unwrap_or_default(),
        status: int_of(api.get("statusCode"), 0),
        rx: int_of(api.get("rx"), 0),
        tx: int_of(api.get("tx"), 0),
        size: None,
        start,
        end,
        host,
        offset,
        length,
        prefix: q.and_then(|q| q.get("prefix")).and_then(Value::as_str).map(str::to_string),
    })
}

/// `urllib.parse.unquote` (and `unquote_plus` when `plus`).
fn unquote(s: &str, plus: bool) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let h = &s[i + 1..i + 3];
            if let Ok(v) = u8::from_str_radix(h, 16) {
                if h.bytes().all(|c| c.is_ascii_hexdigit()) {
                    out.push(v);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(if plus && b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The first value of `key` in a query string, blank values dropped (`parse_qs`).
fn query_value(q: &str, key: &str) -> Option<String> {
    for part in q.split('&') {
        let (k, v) = part.split_once('=')?;
        if unquote(k, true) == key && !v.is_empty() {
            return Some(unquote(v, true));
        }
    }
    None
}

fn s3_request(line: &str, no: usize) -> Result<Request> {
    let bad = || anyhow!("line {no}: not an S3 server access log record");
    let mut rest = line;
    let mut fields: Vec<&str> = Vec::with_capacity(15);
    for i in 0..15 {
        let (f, r) = match i {
            2 => {
                let r = rest.strip_prefix('[').ok_or_else(bad)?;
                let e = r.find(']').ok_or_else(bad)?;
                if e == 0 {
                    return Err(bad());
                }
                (&r[..e], &r[e + 1..])
            }
            8 => {
                let r = rest.strip_prefix('"').ok_or_else(bad)?;
                let e = r.find('"').ok_or_else(bad)?;
                (&r[..e], &r[e + 1..])
            }
            _ => {
                let e = rest.find(char::is_whitespace).ok_or_else(bad)?;
                if e == 0 {
                    return Err(bad());
                }
                (&rest[..e], &rest[e..])
            }
        };
        fields.push(f);
        rest = r.strip_prefix(' ').ok_or_else(bad)?;
    }
    let (bucket, when, ip, op, key, uri, status, sent, size, total) = (fields[1], fields[2], fields[3], fields[6], fields[7], fields[8], fields[9], fields[11], fields[12], fields[13]);
    let start = access_seconds(when);
    let end = match start {
        Some(s) if total != "-" => Some(s + int_text(total) as f64 / 1000.0),
        s => s,
    };
    let key = if key != "-" { unquote(key, false).trim_start_matches('/').to_string() } else { String::new() };
    let target = if uri.contains(' ') { uri.split(' ').nth(1).unwrap_or("") } else { uri };
    let query = target.split_once('?').map(|(_, q)| q.split('#').next().unwrap_or("")).unwrap_or("");
    let sent_n = if sent != "-" { int_text(sent) } else { 0 };
    let size_n = if size != "-" { Some(int_text(size)) } else { None };
    let rx = if PUT.contains(&op) || MP_PART.contains(&op) { size_n.filter(|n| *n != 0).unwrap_or(0) } else { 0 };
    Ok(Request {
        op: op.to_string(),
        bucket: bucket.to_string(),
        key,
        status: int_text(status),
        rx,
        tx: sent_n,
        size: size_n,
        start,
        end,
        host: ip.to_string(),
        offset: None,
        length: None,
        prefix: query_value(query, "prefix"),
    })
}

fn under(path: &str, root: &str) -> bool {
    path == root || path.starts_with(&format!("{root}/"))
}

/// One pass over the log: every request through the exporter; the requests read and the
/// log's own notes.
fn pass(open: &Opener, root: &str, format: &str, ex: &mut Exporter, mut drain: impl FnMut(&mut Exporter) -> Result<()>) -> Result<(u64, BTreeMap<String, u64>)> {
    let mut notes: BTreeMap<String, u64> = BTreeMap::new();
    let mut note = |k: &str| *notes.entry(k.to_string()).or_insert(0) += 1;
    let mut hosts: HashMap<String, i64> = HashMap::new();
    // the object's full key -> (order, open id, bytes so far) of the upload in progress (the
    // upload id is in the start's response, which no log has)
    let mut uploads: HashMap<String, (u64, usize, i64)> = HashMap::new();
    let mut seq = 0u64;
    let mut n = 0u64;
    let mut no = 0usize;
    for_each_line(open()?, |raw| {
        no += 1;
        let line = raw.trim();
        if line.is_empty() {
            return Ok(());
        }
        let r = if format == "minio-audit" {
            let doc: Value = serde_json::from_str(line).map_err(|e| anyhow!("line {no}: {e}"))?;
            if !doc.is_object() {
                bail!("line {no}: not a JSON object");
            }
            minio_request(&doc, no)?
        } else {
            s3_request(line, no)?
        };
        n += 1;
        if r.start.is_none() {
            note("requests_without_time");
        }
        let k = hosts.len() as i64;
        let tid = *hosts.entry(r.host.clone()).or_insert(k);
        let when: When = (tid, r.start, r.end);
        let full = if r.key.is_empty() { r.bucket.clone() } else { format!("{}/{}", r.bucket, r.key) };
        let err = if (200..300).contains(&r.status) { None } else { Some(errno(r.status)) };
        let op = r.op.as_str();
        if LIST.contains(&op) {
            let path = match r.prefix.as_deref().filter(|p| !p.is_empty()) {
                Some(p) => format!("{}/{}", r.bucket, p.trim_matches('/')),
                None => r.bucket.clone(),
            };
            if !under(&path, root) {
                note("requests_outside_root");
                return drain(ex);
            }
            let oid = ex.open(when, &path, "O_RDONLY|O_DIRECTORY", None, err)?;
            if let (None, Some(oid)) = (err, oid) {
                ex.op(when, "readdir", oid, None, Value::Null, vec![]);
                ex.op(when, "close", oid, None, Value::Null, vec![]);
            }
            return drain(ex);
        }
        if MP_DONE.contains(&op) || MP_ABORT.contains(&op) {
            match uploads.remove(&full) {
                None => note(if under(&full, root) { "multipart_without_start" } else { "requests_outside_root" }),
                Some((_, oid, _)) => {
                    ex.op(when, "close", oid, None, Value::Null, vec![]);
                    if MP_ABORT.contains(&op) {
                        ex.path_event(when, "unlink", &full, None)?;
                    }
                }
            }
            return drain(ex);
        }
        if MP_PART.contains(&op) {
            match uploads.get_mut(&full) {
                None => note(if under(&full, root) { "multipart_without_start" } else { "requests_outside_root" }),
                Some(up) => {
                    // parts in the order logged, each after the last
                    ex.data(when, "write", up.1, up.2, r.rx, false, r.rx, err);
                    up.2 += r.rx;
                }
            }
            return drain(ex);
        }
        if r.key.is_empty() || !under(&full, root) {
            if GET.contains(&op) || PUT.contains(&op) || HEAD.contains(&op) || DELETE.contains(&op) || MP_NEW.contains(&op) {
                note("requests_outside_root");
            } else {
                note(&format!("requests_not_exported:{op}"));
            }
            return drain(ex);
        }
        if GET.contains(&op) {
            let oid = ex.open(when, &full, "O_RDONLY", None, err)?;
            if let (None, Some(oid)) = (err, oid) {
                let off = r.offset.unwrap_or(0);
                let length = r.length.unwrap_or(r.tx);
                if r.offset.is_none() && r.size.is_some_and(|s| r.tx < s) {
                    note("ranges_unknown");
                }
                ex.data(when, "read", oid, off, length, r.offset.is_some(), r.tx, None);
                ex.op(when, "close", oid, None, Value::Null, vec![]);
            }
        } else if PUT.contains(&op) {
            let oid = ex.open(when, &full, "O_WRONLY|O_CREAT|O_TRUNC", Some("644"), err)?;
            if let (None, Some(oid)) = (err, oid) {
                ex.data(when, "write", oid, 0, r.rx, false, r.rx, None);
                ex.op(when, "close", oid, None, Value::Null, vec![]);
            }
        } else if HEAD.contains(&op) {
            ex.path_event(when, "stat", &full, err)?;
        } else if DELETE.contains(&op) {
            ex.path_event(when, "unlink", &full, err)?;
        } else if MP_NEW.contains(&op) {
            let oid = ex.open(when, &full, "O_WRONLY|O_CREAT|O_TRUNC", Some("644"), err)?;
            if let (None, Some(oid)) = (err, oid) {
                seq += 1;
                uploads.insert(full, (seq, oid, 0));
            }
        } else {
            note(&format!("requests_not_exported:{op}"));
        }
        drain(ex)
    })?;
    let mut left: Vec<(u64, usize)> = uploads.values().map(|(s, oid, _)| (*s, *oid)).collect();
    left.sort_unstable();
    for (_, oid) in left {
        note("multipart_not_completed");
        let t0 = ex.t0();
        ex.op((0, t0, t0), "close", oid, None, Value::Null, vec![]);
    }
    drain(ex)?;
    Ok((n, notes))
}

fn check(root: &str, format: &str) -> Result<String> {
    if !FORMATS.contains(&format) {
        bail!("not an object log format: {format}");
    }
    let root = root.trim_matches('/');
    if root.is_empty() {
        bail!("an object log needs --root BUCKET[/PREFIX]");
    }
    Ok(root.to_string())
}

/// The first pass: the plan of the second and the header.
fn plan(open: &Opener, root: &str, format: &str) -> Result<(export::Plan, Value)> {
    let mut ex = Exporter::new(root);
    let (n, notes) = pass(open, root, format, &mut ex, |_| Ok(()))?;
    let (plan, mut header) = ex.finish(root, format, &BTreeMap::new());
    header["requests"] = json!(n);
    let h = header["notes"].as_object_mut().unwrap();
    for (k, v) in notes {
        h.entry(k).or_insert(json!(v));
    }
    if format == "minio-audit" {
        h.entry("time_is_response_end").or_insert(json!("a MinIO audit entry's time is taken as when the response completed; the request starts timeToResponse earlier"));
    }
    h.entry("lanes").or_insert(json!("one per client host; the log knows no thread"));
    Ok((plan, header))
}

/// Export an object log under `root` = `BUCKET[/PREFIX]`.
pub fn export(open: &Opener, root: &str, format: &str, out: &mut dyn Write) -> Result<Summary> {
    let root = check(root, format)?;
    let (plan, header) = plan(open, &root, format)?;
    let lines = header["lines"].as_u64().unwrap_or(0) as usize;
    let seen = Seen { lines: header["requests"].as_u64().unwrap_or(0), calls: header["requests"].as_u64().unwrap_or(0) };
    if lines == 0 {
        return Ok(Summary { header, lines, sha256: String::new(), seen });
    }
    let sha256 = export::write_file(out, &header, |write| {
        let mut ex = Exporter::emitting(&root, plan);
        pass(open, &root, format, &mut ex, |ex| {
            let q = std::mem::take(&mut ex.queue);
            if q.is_empty() { Ok(()) } else { write(q) }
        })
        .map(|_| ())
    })?;
    Ok(Summary { header, lines, sha256, seen })
}

/// The `aeiou_metrics: 2` document of an object log, and the requests it read.
pub fn metrics(open: &Opener, root: &str, format: &str, block: Option<u64>) -> Result<(Value, u64)> {
    let root = check(root, format)?;
    let (plan, header) = plan(open, &root, format)?;
    let mut fm = export::FileMetrics::new(block);
    let mut ex = Exporter::emitting(&root, plan);
    let mut buf = Vec::new();
    pass(open, &root, format, &mut ex, |ex| {
        let q: Vec<Event> = std::mem::take(&mut ex.queue);
        for e in q {
            buf.clear();
            e.write(&mut buf);
            fm.line(&serde_json::from_slice(&buf)?)?;
        }
        Ok(())
    })?;
    let requests = header["requests"].as_u64().unwrap_or(0);
    Ok((fm.document(&header)?, requests))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times() {
        assert_eq!(iso_seconds("1970-01-01T00:00:01Z"), Some(1.0));
        assert_eq!(iso_seconds("1970-01-01T01:00:00+01:00"), Some(0.0));
        assert_eq!(iso_seconds("1970-01-01 00:00:00.5-0100"), Some(3600.5));
        assert_eq!(iso_seconds("2024-02-29T00:00:00"), Some(1709164800.0));
        assert_eq!(iso_seconds("1970-01-01T00:00:00.Z"), None);
        assert_eq!(iso_seconds("1970-01-01T00:00:00+1"), None);
        assert_eq!(iso_seconds("nope"), None);
        assert_eq!(access_seconds("06/Feb/2019:00:00:38 +0000"), Some(1549411238.0));
        assert_eq!(access_seconds("06/Feb/2019:01:00:38 +0100"), Some(1549411238.0));
        assert_eq!(access_seconds("06/Fex/2019:00:00:38 +0000"), None);
        assert_eq!(access_seconds("06/Feb/2019:00:00:38 *0000"), None);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
    }

    #[test]
    fn urls() {
        assert_eq!(unquote("a%2Fb%zz+c%", false), "a/b%zz+c%");
        assert_eq!(unquote("a+b", true), "a b");
        assert_eq!(query_value("list-type=2&prefix=d%2Fe&x=", "prefix"), Some("d/e".into()));
        assert_eq!(query_value("prefix=", "prefix"), None);
        assert_eq!(query_value("", "prefix"), None);
    }

    #[test]
    fn formats_from_the_first_line() {
        assert_eq!(source_of(r#"{"api": {"name": "GetObject"}}"#), Some("minio-audit"));
        assert_eq!(source_of(r#"{"api": 1}"#), None);
        assert_eq!(source_of(r#"{"api": "#), None);
        assert_eq!(source_of("x"), None);
        let owner = "a".repeat(64);
        assert_eq!(source_of(&format!("{owner} b [06/Feb/2019:00:00:38 +0000] rest")), Some("s3-access"));
        assert_eq!(source_of(&format!("{owner} b [06/Feb/2019:00:00:38 +0000]rest")), None);
        assert_eq!(int_of(Some(&json!(2.7)), 0), 2);
        assert_eq!(int_of(Some(&json!(true)), 0), 1);
        assert_eq!(int_of(Some(&json!(" 5 ")), 0), 5);
        assert_eq!(int_of(None, 3), 3);
    }
}
