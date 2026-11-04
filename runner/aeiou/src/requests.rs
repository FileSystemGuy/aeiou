//! The capture's object requests (`DESIGN_REVIEW.md` §3.96, phase 2 of brief item 54): the
//! library `aeiou-trace capture --requests shim` preloads into the command
//! (`libaeiou_capture.so`, `runner/capture-shim/`), the log it records into (`requests.bin`,
//! `aeiou_interpose::log`, kept as it was written so a converter's bug is fixed by converting
//! again), and what this side makes of it after the run: `requests.jsonl`, one request a line
//! with its response, and the record's account of what the library reached and what it did not.
//! Also the anchors: an unsigned `HEAD` of a name of the capture's own sent to each `--endpoint`
//! at the start and the end, so the store's log, where it shows as a refusal, can be aligned
//! with the capture's clock.
//!
//! `requests.jsonl`: a first line `{"aeiou_requests": 1, ...}`, then a request a line in the
//! order they were sent: `start` (the wall clock when its head was sent, seconds), `end` (when
//! its response's head arrived; the body's end is not seen), `pid`, `tid`, `via` (`ssl` or
//! `socket`), `fd`, `peer` (from the `connect` that made the socket), `method`, `target`,
//! `request` (the kept headers, names in lower case), `status`, `response`, and `truncated`
//! when either head was cut short. Responses are paired with requests in order on each
//! connection, as HTTP/1.1 answers them.
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use aeiou_interpose::capture::{self as rec, Record};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map, Value};

pub const LIBRARY: &str = "libaeiou_capture.so";
pub const LOG: &str = "requests.bin";
pub const FILE: &str = "requests.jsonl";
/// The log's size unless `--requests-buffer` says: sparse, so only what is recorded is written.
pub const DEFAULT_BUFFER: u64 = 1 << 30;

/// What a capture under the library still cannot see of object requests.
pub const NOT_SEEN: &str = "object-store requests of clients the library does not reach: TLS inside the program (python-build-standalone's Python, the AWS CRT's s2n, rustls, the JVM), Go's clients (no libc), HTTP/2, a static or setuid binary, and a process whose environment drops LD_PRELOAD (processes.not_reached names those it saw start)";

/// The calls of the library's own that the trace shows, outside every `--root`.
pub const VISIBLE: &str = "each process's start opens, maps, and closes requests.bin (openat, fstat, mmap, close), and the dynamic loader opens and maps libaeiou_capture.so: both outside every --root, so no reader counts them";

/// The library of the running binary: a build tree's (`target/release/deps/`, where a test
/// build leaves it, else `target/release/`), else the release's `lib/` beside its `bin/`.
pub fn library() -> std::result::Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("the running binary: {e}"))?;
    let dir = exe.parent().unwrap_or(Path::new("/"));
    let near = [dir.join("deps").join(LIBRARY), dir.join(LIBRARY), dir.join("../lib").join(LIBRARY)];
    near.iter().find(|p| p.is_file()).map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone())).ok_or_else(|| format!("{LIBRARY} is not beside {} nor in {}: the release's lib/ holds it (a build: cargo build --release -p aeiou-capture-shim)", dir.display(), dir.join("../lib").display()))
}

/// `--requests-buffer`'s size: bytes, or with `KiB`, `MiB`, `GiB` (or `K`, `M`, `G`, binary).
pub fn parse_size(s: &str) -> std::result::Result<u64, String> {
    let t = s.trim();
    let split = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
    let (n, unit) = t.split_at(split);
    let n: u64 = n.parse().map_err(|_| format!("--requests-buffer {s}: expected a size, as 64MiB or 1GiB"))?;
    let shift = match unit {
        "" => 0,
        "K" | "KiB" => 10,
        "M" | "MiB" => 20,
        "G" | "GiB" => 30,
        _ => return Err(format!("--requests-buffer {s}: the unit is KiB, MiB, or GiB")),
    };
    let v = n.checked_shl(shift).filter(|v| v >> shift == n).ok_or_else(|| format!("--requests-buffer {s}: too large"))?;
    if v < aeiou_interpose::log::MIN_CAPACITY {
        return Err(format!("--requests-buffer {s}: at least {} bytes", aeiou_interpose::log::MIN_CAPACITY));
    }
    Ok(v)
}

/// Make the log the library records into.
pub fn create_log(path: &Path, capacity: u64) -> Result<()> {
    aeiou_interpose::log::create(path, capacity).with_context(|| path.display().to_string())
}

/// The successful `execve`s of an strace's text (`-f`, so each line begins with its pid): each
/// process and the program it began, in order.
pub fn execs(trace: &Path) -> Result<Vec<(u32, String)>> {
    let mut out = Vec::new();
    let mut pending: HashMap<u32, String> = HashMap::new();
    let path_of = |s: &str| -> String {
        // the first quoted argument: execve's path, execveat's after its descriptor
        let after = s.split_once(" execve(").or_else(|| s.split_once(" execveat(")).map_or("", |(_, r)| r);
        after.split('"').nth(1).unwrap_or("").to_string()
    };
    let ok = |s: &str| s.rsplit_once(" = ").is_some_and(|(_, r)| r.split_whitespace().next() == Some("0"));
    let f = std::fs::File::open(trace).with_context(|| trace.display().to_string())?;
    crate::strace::for_each_line(Box::new(std::io::BufReader::new(f)), |line| {
        let Some((pid, rest)) = line.split_once(char::is_whitespace) else { return Ok(()) };
        let Ok(pid) = pid.parse::<u32>() else { return Ok(()) };
        if rest.contains("<... execve resumed>") || rest.contains("<... execveat resumed>") {
            if let Some(p) = pending.remove(&pid) {
                if ok(rest) {
                    out.push((pid, p));
                }
            }
        } else if rest.contains(" execve(") || rest.contains(" execveat(") {
            let p = path_of(rest);
            if rest.trim_end().ends_with("<unfinished ...>") {
                pending.insert(pid, p);
            } else if ok(rest) {
                out.push((pid, p));
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// A head's text split: its first line, and its headers by lower-cased name (repeats joined
/// with `, `).
fn split_head(text: &str) -> (String, Map<String, Value>) {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("").to_string();
    let mut h = Map::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            match h.get_mut(&k) {
                Some(Value::String(s)) => {
                    s.push_str(", ");
                    s.push_str(v);
                }
                _ => {
                    h.insert(k, json!(v));
                }
            }
        }
    }
    (first, h)
}

fn peer_of(family: u16, port: u16, addr: &[u8; 16]) -> Option<String> {
    match family as i32 {
        libc::AF_INET => Some(format!("{}:{port}", std::net::Ipv4Addr::from(<[u8; 4]>::try_from(&addr[..4]).unwrap()))),
        libc::AF_INET6 => Some(format!("[{}]:{port}", std::net::Ipv6Addr::from(*addr))),
        libc::AF_UNIX => Some("unix".into()),
        _ => None,
    }
}

fn seconds(ns: u64) -> f64 {
    ns as f64 / 1e9
}

/// Read the log at `log`, keep it cut to what was recorded, write `requests.jsonl` at `out`, and
/// return the record's account: the requests, the log's counts, and the processes the library
/// reached against those the trace saw begin (`execs`).
pub fn convert(log: &Path, out: &Path, execs: &[(u32, String)]) -> Result<Value> {
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(log).with_context(|| log.display().to_string())?;
    let mut header = [0u8; aeiou_interpose::log::HEADER as usize];
    f.read_exact(&mut header).with_context(|| log.display().to_string())?;
    let n = |at: usize| u64::from_ne_bytes(header[at..at + 8].try_into().unwrap());
    let limit = n(24).min(n(16));
    let mut buf = header.to_vec();
    Read::by_ref(&mut f).take(limit.saturating_sub(aeiou_interpose::log::HEADER)).read_to_end(&mut buf)?;
    f.set_len(buf.len() as u64)?;

    let mut records = Vec::new();
    let reading = aeiou_interpose::log::read(&buf, |k, p| records.push(rec::decode(k, p))).map_err(|e| anyhow!("{}: {e}", log.display()))?;

    struct Proc {
        exe: String,
        ppid: u32,
        loaded: u32,
        heads: u64,
        peers: Vec<String>,
    }
    let mut procs: BTreeMap<u32, Proc> = BTreeMap::new();
    let mut peers: HashMap<(u32, i32), String> = HashMap::new();
    let mut waiting: HashMap<(u32, u8, u64), VecDeque<usize>> = HashMap::new();
    let mut reqs: Vec<Value> = Vec::new();
    let (mut unpaired, mut cut) = (0u64, 0u64);
    for r in records {
        match r {
            Record::Process { at, ppid, how, exe } => {
                let p = procs.entry(at.pid).or_insert(Proc { exe: exe.clone(), ppid, loaded: 0, heads: 0, peers: Vec::new() });
                (p.exe, p.ppid) = (exe, ppid);
                if how == rec::LOADED {
                    p.loaded += 1;
                }
            }
            Record::Connect { at, fd, family, port, addr, .. } => {
                if let Some(peer) = peer_of(family, port, &addr) {
                    if let Some(p) = procs.get_mut(&at.pid) {
                        // a resolver's datagrams to port 53 are not the object client's
                        if port != 53 && family as i32 != libc::AF_UNIX && !p.peers.contains(&peer) {
                            p.peers.push(peer.clone());
                        }
                    }
                    peers.insert((at.pid, fd), peer);
                }
            }
            Record::Head { at, fd, via, dir, truncated, conn, text } => {
                if let Some(p) = procs.get_mut(&at.pid) {
                    p.heads += 1;
                }
                cut += truncated as u64;
                let key = (at.pid, via, conn);
                let (first, headers) = split_head(&text);
                if dir == rec::SENT {
                    let mut parts = first.splitn(3, ' ');
                    let (method, target, version) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                    // a socket inherited over a fork was connected by the parent
                    let ppid = procs.get(&at.pid).map(|p| p.ppid);
                    let peer = peers.get(&(at.pid, fd)).or_else(|| ppid.and_then(|pp| peers.get(&(pp, fd))));
                    reqs.push(json!({
                        "start": seconds(at.time_ns),
                        "end": null,
                        "pid": at.pid,
                        "tid": at.tid,
                        "via": if via == rec::VIA_SSL { "ssl" } else { "socket" },
                        "fd": fd,
                        "peer": peer,
                        "method": method,
                        "target": target,
                        "version": version,
                        "request": headers,
                        "status": null,
                        "response": null,
                        "truncated": truncated,
                    }));
                    waiting.entry(key).or_default().push_back(reqs.len() - 1);
                } else {
                    let Some(i) = waiting.get_mut(&key).and_then(VecDeque::pop_front) else {
                        unpaired += 1;
                        continue;
                    };
                    let q = &mut reqs[i];
                    q["end"] = json!(seconds(at.time_ns));
                    let mut parts = first.splitn(3, ' ');
                    let _version = parts.next();
                    q["status"] = json!(parts.next().and_then(|s| s.parse::<u16>().ok()));
                    q["reason"] = json!(parts.next().unwrap_or(""));
                    q["response"] = Value::Object(headers);
                    if truncated {
                        q["truncated"] = json!(true);
                    }
                }
            }
            Record::Other { .. } => {}
        }
    }
    reqs.sort_by(|a, b| a["start"].as_f64().partial_cmp(&b["start"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    let answered = reqs.iter().filter(|q| !q["status"].is_null()).count();

    let tmp = out.with_extension("jsonl.tmp");
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).with_context(|| tmp.display().to_string())?);
        let head = json!({
            "aeiou_requests": 1,
            "clock": "CLOCK_REALTIME, seconds",
            "source": LIBRARY,
            "times": "start: the request's head was sent; end: the response's head arrived (the body's end is not seen)",
            "requests": reqs.len(),
        });
        writeln!(w, "{head}")?;
        for q in &reqs {
            writeln!(w, "{q}")?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, out).with_context(|| out.display().to_string())?;

    // every exec the trace saw is a process the library should have loaded in
    let mut execd: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
    for (pid, path) in execs {
        execd.entry(*pid).or_default().push(path);
    }
    // a full log dropped processes' records too: which it missed is then unknown
    let not_reached: Value = if reading.dropped > 0 {
        json!(null)
    } else {
        execd.iter().filter(|(pid, paths)| procs.get(pid).map_or(0, |p| p.loaded as usize) < paths.len()).map(|(pid, paths)| json!({"pid": pid, "exe": paths.last()})).collect()
    };
    let silent: Vec<Value> = procs.iter().filter(|(_, p)| p.heads == 0 && !p.peers.is_empty()).map(|(pid, p)| json!({"pid": pid, "exe": p.exe, "peers": p.peers})).collect();
    Ok(json!({
        "file": FILE,
        "requests": reqs.len(),
        "answered": answered,
        "unanswered": reqs.len() - answered,
        "responses_unpaired": unpaired,
        "heads_cut": cut,
        "log": {
            "file": LOG,
            "capacity": reading.capacity,
            "used": reading.used,
            "records": reading.records,
            "dropped": reading.dropped,
            "dropped_bytes": reading.dropped_bytes,
            "torn": reading.torn,
            "unreadable_bytes": reading.unreadable_bytes,
        },
        "processes": {
            "reached": procs.values().filter(|p| p.loaded > 0).count(),
            "forked": procs.values().filter(|p| p.loaded == 0).count(),
            "not_reached": not_reached,
            "not_reached_unknown": (reading.dropped > 0).then_some("the log was full, and a process's own record may be among those dropped"),
            "connected_without_heads": silent,
        },
    }))
}

/// The host, port, and path of an `http://` or `https://` URL.
fn http_parts(url: &str) -> Option<(bool, String, u16, String)> {
    let (tls, rest) = if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else {
        return None;
    };
    let (hostport, path) = rest.split_once('/').map_or((rest, ""), |(h, p)| (h, p));
    if hostport.is_empty() {
        return None;
    }
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) if !h.ends_with(']') || h.starts_with('[') => match p.parse::<u16>() {
            Ok(p) => (h.to_string(), p),
            Err(_) if hostport.starts_with('[') && hostport.ends_with(']') => (hostport.to_string(), if tls { 443 } else { 80 }),
            Err(_) => return None,
        },
        _ => (hostport.to_string(), if tls { 443 } else { 80 }),
    };
    Some((tls, host, port, path.trim_matches('/').to_string()))
}

/// A socket to `host:port`, five seconds to connect and to each read.
fn connect(host: &str, port: u16) -> std::result::Result<std::net::TcpStream, String> {
    use std::net::ToSocketAddrs;
    let addr = (host.trim_start_matches('[').trim_end_matches(']'), port).to_socket_addrs().map_err(|e| format!("{host}: {e}"))?.next().ok_or_else(|| format!("{host}: no address"))?;
    let t = std::time::Duration::from_secs(5);
    let s = std::net::TcpStream::connect_timeout(&addr, t).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(t)).map_err(|e| e.to_string())?;
    Ok(s)
}

/// The status of an unsigned `HEAD` of `target` sent over `s`.
fn head(mut s: impl Read + Write, host: &str, port: u16, default_port: u16, target: &str) -> std::result::Result<u16, String> {
    let hostport = if port == default_port { host.to_string() } else { format!("{host}:{port}") };
    write!(s, "HEAD {target} HTTP/1.1\r\nHost: {hostport}\r\nUser-Agent: aeiou-trace/{}\r\nConnection: close\r\n\r\n", env!("CARGO_PKG_VERSION")).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while !buf.windows(2).any(|w| w == b"\r\n") {
        let n = s.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let line = String::from_utf8_lossy(&buf);
    line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).filter(|_| line.starts_with("HTTP/")).ok_or_else(|| format!("not an HTTP answer: {:?}", line.lines().next().unwrap_or("")))
}

/// The anchor carries nothing but its name: a store's certificate is not checked (its
/// signatures still are, by the provider's algorithms).
#[cfg(feature = "object")]
#[derive(Debug)]
struct AnyCertificate(std::sync::Arc<rustls::crypto::CryptoProvider>);

#[cfg(feature = "object")]
impl rustls::client::danger::ServerCertVerifier for AnyCertificate {
    fn verify_server_cert(&self, _: &rustls::pki_types::CertificateDer<'_>, _: &[rustls::pki_types::CertificateDer<'_>], _: &rustls::pki_types::ServerName<'_>, _: &[u8], _: rustls::pki_types::UnixTime) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(&self, m: &[u8], c: &rustls::pki_types::CertificateDer<'_>, d: &rustls::DigitallySignedStruct) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(&self, m: &[u8], c: &rustls::pki_types::CertificateDer<'_>, d: &rustls::DigitallySignedStruct) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

/// `head` over TLS, blocking on the socket (rustls, the TLS the object engine brings).
#[cfg(feature = "object")]
fn https_head(host: &str, port: u16, target: &str) -> std::result::Result<u16, String> {
    use std::sync::Arc;
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AnyCertificate(provider)))
        .with_no_client_auth();
    let name = rustls::pki_types::ServerName::try_from(host.trim_start_matches('[').trim_end_matches(']').to_string()).map_err(|e| format!("{host}: {e}"))?;
    let conn = rustls::ClientConnection::new(Arc::new(config), name).map_err(|e| e.to_string())?;
    head(rustls::StreamOwned::new(conn, connect(host, port)?), host, port, 443, target)
}

#[cfg(not(feature = "object"))]
fn https_head(_: &str, _: u16, _: &str) -> std::result::Result<u16, String> {
    Err("not sent: this build has no TLS client (built without the object engine)".into())
}

/// Send the anchor `name` to the endpoint `url`: an unsigned `HEAD` of `name` under the URL's
/// path, the times it was sent and answered on the wall clock, and the status (a store refuses
/// it, 403, and logs it with its time).
pub fn anchor(url: &str, name: &str) -> Value {
    let Some((tls, host, port, path)) = http_parts(url) else {
        return json!({"url": url, "name": name, "not_sent": "the URL names no HTTP host (s3://BUCKET does not): give the store's http:// or https:// URL to anchor its log"});
    };
    let target = if path.is_empty() { format!("/{name}") } else { format!("/{path}/{name}") };
    let sent = crate::run::unix_now();
    let r = if tls { https_head(&host, port, &target) } else { connect(&host, port).and_then(|s| head(s, &host, port, 80, &target)) };
    let received = crate::run::unix_now();
    match r {
        Ok(status) => json!({"url": url, "name": name, "target": target, "sent": sent, "received": received, "status": status}),
        Err(e) => json!({"url": url, "name": name, "target": target, "sent": sent, "error": e}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_take_binary_units_and_a_floor() {
        assert_eq!(parse_size("1GiB").unwrap(), 1 << 30);
        assert_eq!(parse_size("64M").unwrap(), 64 << 20);
        assert_eq!(parse_size("8192").unwrap(), 8192);
        assert!(parse_size("4096").unwrap_err().contains("at least 4160 bytes"));
        assert!(parse_size("1TB").unwrap_err().contains("the unit is"));
        assert!(parse_size("x").unwrap_err().contains("expected a size"));
        assert!(parse_size("99999999999999GiB").unwrap_err().contains("too large"));
    }

    #[test]
    fn urls_split_into_host_port_and_path() {
        assert_eq!(http_parts("http://127.0.0.1:9000/bucket/p/"), Some((false, "127.0.0.1".into(), 9000, "bucket/p".into())));
        assert_eq!(http_parts("https://store.example"), Some((true, "store.example".into(), 443, String::new())));
        assert_eq!(http_parts("http://[::1]:9000"), Some((false, "[::1]".into(), 9000, String::new())));
        assert_eq!(http_parts("http://[::1]"), Some((false, "[::1]".into(), 80, String::new())));
        assert_eq!(http_parts("http://h:x"), None);
        assert_eq!(http_parts("http:///b"), None);
        assert_eq!(http_parts("s3://bucket"), None);
        assert!(anchor("s3://bucket", "n")["not_sent"].as_str().unwrap().contains("names no HTTP host"));
        // nothing listens on port 1
        assert!(anchor("http://127.0.0.1:1", "n")["error"].is_string());
    }

    #[test]
    fn heads_split_into_their_line_and_headers() {
        let (first, h) = split_head("GET /b HTTP/1.1\nHost: h\nX-A: 1\nx-a: 2\nnot a header\n");
        assert_eq!(first, "GET /b HTTP/1.1");
        assert_eq!(Value::Object(h), json!({"host": "h", "x-a": "1, 2"}));
        assert_eq!(peer_of(libc::AF_INET6 as u16, 80, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).unwrap(), "[::1]:80");
        assert_eq!(peer_of(99, 80, &[0; 16]), None);
    }

    #[test]
    fn a_log_s_odd_records_are_counted_and_kept_apart() {
        let d = std::env::temp_dir().join(format!("aeiou-requests-convert-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir(&d).unwrap();
        let log = d.join(LOG);
        create_log(&log, 1 << 16).unwrap();
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&log).unwrap();
        let l = unsafe { aeiou_interpose::log::Log::map(std::os::fd::AsRawFd::as_raw_fd(&f)) }.unwrap();
        let at = |ns: u64| rec::Stamp { time_ns: ns, pid: 5, tid: 6 };
        l.append(rec::PROCESS, &[&at(1).bytes(), &rec::process(1, rec::LOADED), b"/bin/c"]);
        l.append(rec::CONNECT, &[&at(2).bytes(), &rec::connect(3, 0, libc::AF_UNIX as u16, 0, &[0; 16])]);
        l.append(rec::HEAD, &[&at(3).bytes(), &rec::head(3, rec::VIA_SOCKET, rec::RECEIVED, false, 3), b"HTTP/1.1 200 OK\n"]);
        l.append(rec::HEAD, &[&at(4).bytes(), &rec::head(3, rec::VIA_SOCKET, rec::SENT, false, 3), b"GET /b/k HTTP/1.1\n"]);
        l.append(rec::HEAD, &[&at(5).bytes(), &rec::head(3, rec::VIA_SOCKET, rec::RECEIVED, true, 3), b"HTTP/1.1 200 OK\nContent-Le"]);
        l.append(99, &[b"later"]);
        let r = convert(&log, &d.join(FILE), &[(5, "/bin/c".into()), (8, "/static".into())]).unwrap();
        assert_eq!((r["requests"].as_u64(), r["answered"].as_u64(), r["responses_unpaired"].as_u64(), r["heads_cut"].as_u64()), (Some(1), Some(1), Some(1), Some(1)), "{r}");
        assert_eq!(r["processes"]["not_reached"], json!([{"pid": 8, "exe": "/static"}]));
        // a Unix socket is no store's peer: no process connected without heads
        assert_eq!(r["processes"]["connected_without_heads"], json!([]));
        let text = std::fs::read_to_string(d.join(FILE)).unwrap();
        let q: Value = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!((q["peer"].as_str(), q["truncated"].as_bool()), (Some("unix"), Some(true)), "{q}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn an_anchor_answered_by_no_http_says_so() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut b = [0u8; 256];
            let _ = c.read(&mut b);
            c.write_all(b"nope").unwrap();
        });
        let a = anchor(&format!("http://127.0.0.1:{port}/b"), "n");
        t.join().unwrap();
        assert_eq!(a["target"], "/b/n");
        assert!(a["error"].as_str().unwrap().contains("not an HTTP answer: \"nope\""), "{a}");
    }

    #[test]
    fn execs_are_read_from_an_strace_whole_or_resumed() {
        let d = std::env::temp_dir().join(format!("aeiou-requests-execs-{}", std::process::id()));
        std::fs::write(
            &d,
            "10 1.0 execve(\"/usr/bin/env\", [\"env\"], 0x1 /* 3 vars */) = 0 <0.1>\n\
11 1.1 execve(\"/bin/true\", [\"true\"], 0x1 <unfinished ...>\n\
12 1.1 execve(\"/nope\", [\"nope\"], 0x1) = -1 ENOENT (No such file or directory) <0.1>\n\
11 1.2 <... execve resumed>) = 0 <0.1>\n\
14 1.25 <... execve resumed>) = 0 <0.1>\n\
13 1.3 execveat(3, \"x\", [], 0x1, 0) = 0 <0.1>\n\
not a line\n",
        )
        .unwrap();
        assert_eq!(execs(&d).unwrap(), vec![(10, "/usr/bin/env".into()), (11, "/bin/true".into()), (13, "x".into())]);
        std::fs::remove_file(&d).unwrap();
    }
}
