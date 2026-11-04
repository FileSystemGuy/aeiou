//! The library preloaded into a real client (`DESIGN_REVIEW.md` §3.96, phase 2 of brief item
//! 54): the system's Python talking HTTP/1.1 to a server of the test's through each of libc's
//! socket calls the library hooks (`send` and `recv`, `write` and `read`, `writev` and `readv`,
//! `sendmsg` and `recvmsg`, `recvfrom`), a peek not taken for a head, and a forked child; every
//! head is in the log, each process's record too, and no secret header. `tests/capture_requests.rs`
//! of the runner takes it through `aeiou-trace capture`, TLS included.
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;

use aeiou_interpose::capture::{self as rec, decode, Record};

/// The library a test build leaves beside the test binary (`target/release/deps/`).
fn library() -> PathBuf {
    let p = std::env::current_exe().unwrap().parent().unwrap().join("libaeiou_capture.so");
    assert!(p.is_file(), "{}: not built", p.display());
    p
}

const CLIENT: &str = r#"
import ctypes, os, socket, ssl, sys, threading
port = int(sys.argv[1])
libc = ctypes.CDLL(None, use_errno=True)
def req(name):
    return b"GET /" + name + b" HTTP/1.1\r\nHost: h\r\nAuthorization: AWS4 secret\r\n\r\n"
def conn():
    return socket.create_connection(("127.0.0.1", port))
s = conn(); s.sendall(req(b"send")); s.recv(4096, socket.MSG_PEEK); s.recv(4096); s.close()
s = conn(); os.write(s.fileno(), req(b"write")); os.read(s.fileno(), 4096); s.close()
s = conn(); os.writev(s.fileno(), [b"GET /wr", b"itev HTTP/1.1\r\nHost: h\r\n\r\n"]); os.readv(s.fileno(), [bytearray(4096)]); s.close()
s = conn(); s.sendmsg([req(b"sendmsg")]); s.recvmsg(4096); s.close()
s = conn(); s.send(req(b"recvfrom")); s.recvfrom(4096); s.close()
s = conn(); s.sendto(req(b"sendto"), ("127.0.0.1", port)); s.recv(4096); s.close()
# the fortified forms a C program built with _FORTIFY_SOURCE calls, reached through ctypes
buf = ctypes.create_string_buffer(4096)
for chk in (b"__read_chk", b"__recv_chk", b"__recvfrom_chk"):
    s = conn(); s.sendall(req(chk))
    f = getattr(libc, chk.decode())
    if chk == b"__read_chk":
        n = f(s.fileno(), buf, 4096, 4096)
    elif chk == b"__recv_chk":
        n = f(s.fileno(), buf, 4096, 4096, 0)
    else:
        n = f(s.fileno(), buf, 4096, 4096, 0, None, None)
    assert n > 0, (chk, ctypes.get_errno())
    s.close()
# Python's ssl writes and reads with SSL_write_ex and SSL_read_ex; its server's heads are not a client's
cert, key = sys.argv[2], sys.argv[3]
sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); sctx.load_cert_chain(cert, key)
lsn = socket.socket(); lsn.bind(("127.0.0.1", 0)); lsn.listen()
def serve():
    c, _ = lsn.accept()
    t = sctx.wrap_socket(c, server_side=True)
    t.recv(4096); t.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"); t.close()
th = threading.Thread(target=serve); th.start()
cctx = ssl._create_unverified_context()
t = cctx.wrap_socket(socket.create_connection(lsn.getsockname()))
t.sendall(req(b"tls")); t.recv(4096); t.close(); th.join()
pid = os.fork()
if pid == 0:
    s = conn(); s.sendall(req(b"child")); s.recv(4096); s.close()
    sys.exit(0)
os.waitpid(pid, 0)
"#;

#[test]
fn every_hooked_socket_call_records_its_head_and_each_process_its_start() {
    let log = std::env::temp_dir().join(format!("aeiou-capture-shim-preload-{}", std::process::id()));
    let _ = std::fs::remove_file(&log);
    aeiou_interpose::log::create(&log, 1 << 20).unwrap();
    let dir = std::env::temp_dir().join(format!("aeiou-capture-shim-preload-{}.d", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    let (cert, key) = (dir.join("cert.pem"), dir.join("key.pem"));
    let o = Command::new("openssl").args(["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=localhost", "-days", "1", "-keyout"]).arg(&key).arg("-out").arg(&cert).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = server.local_addr().unwrap().port();
    let serving = std::thread::spawn(move || {
        for _ in 0..10 {
            let (mut c, _) = server.accept().unwrap();
            let mut got = Vec::new();
            let mut buf = [0u8; 1024];
            while !got.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = c.read(&mut buf).unwrap();
                assert!(n > 0, "the request's head never ended: {:?}", String::from_utf8_lossy(&got));
                got.extend_from_slice(&buf[..n]);
            }
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nSet-Cookie: k=v\r\n\r\nok").unwrap();
        }
    });
    let o = Command::new("python3").args(["-c", CLIENT, &port.to_string()]).arg(&cert).arg(&key).env("LD_PRELOAD", library()).env("AEIOU_CAPTURE_LOG", &log).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serving.join().unwrap();

    let mut records = Vec::new();
    let r = aeiou_interpose::log::read(&std::fs::read(&log).unwrap(), |k, p| records.push(decode(k, p))).unwrap();
    std::fs::remove_file(&log).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!((r.dropped, r.torn), (0, 0));
    let parent = records.iter().find_map(|r| if let Record::Process { at, how: rec::LOADED, .. } = r { Some(at.pid) } else { None }).expect("the client's record");
    let child = records.iter().find_map(|r| if let Record::Process { at, how: rec::FORKED, ppid, .. } = r { Some((at.pid, *ppid)) } else { None }).expect("the fork's record");
    assert_eq!(child.1, parent);

    let connects: Vec<u16> = records.iter().filter_map(|r| if let Record::Connect { port, errno, .. } = r { assert_eq!(*errno, 0); Some(*port) } else { None }).collect();
    // ten to the test's server, and the TLS client's to the script's own
    assert_eq!(connects.iter().filter(|p| **p == port).count(), 10, "{connects:?}");
    assert_eq!(connects.len(), 11, "{connects:?}");
    let heads: Vec<(u32, u8, u8, i32, String)> = records.iter().filter_map(|r| if let Record::Head { at, dir, via, fd, text, .. } = r { Some((at.pid, *dir, *via, *fd, text.clone())) } else { None }).collect();
    let sent: Vec<&str> = heads.iter().filter(|h| h.1 == rec::SENT).map(|h| h.4.lines().next().unwrap()).collect();
    assert_eq!(sent, ["GET /send HTTP/1.1", "GET /write HTTP/1.1", "GET /writev HTTP/1.1", "GET /sendmsg HTTP/1.1", "GET /recvfrom HTTP/1.1", "GET /sendto HTTP/1.1", "GET /__read_chk HTTP/1.1", "GET /__recv_chk HTTP/1.1", "GET /__recvfrom_chk HTTP/1.1", "GET /tls HTTP/1.1", "GET /child HTTP/1.1"]);
    // one response each: the peek is not one, and the TLS server's own heads are not a client's
    let received: Vec<&(u32, u8, u8, i32, String)> = heads.iter().filter(|h| h.1 == rec::RECEIVED).collect();
    assert_eq!(received.len(), 11, "{heads:?}");
    assert!(received.iter().all(|h| h.4 == "HTTP/1.1 200 OK\nContent-Length: 2\n" || h.4 == "HTTP/1.1 200 OK\nContent-Length: 0\n"), "{heads:?}");
    // the TLS client's heads came through libssl, on the socket learned under the SSL call
    let tls: Vec<_> = heads.iter().filter(|h| h.2 == rec::VIA_SSL).collect();
    assert_eq!(tls.len(), 2, "{heads:?}");
    assert!(tls.iter().all(|h| h.3 >= 0), "{tls:?}");
    assert!(heads.iter().filter(|h| h.2 != rec::VIA_SSL).all(|h| h.2 == rec::VIA_SOCKET));
    assert_eq!(heads.iter().filter(|h| h.0 == child.0).count(), 2);
    assert!(heads.iter().all(|h| !h.4.contains("secret") && !h.4.contains("Authorization")), "{heads:?}");
}
