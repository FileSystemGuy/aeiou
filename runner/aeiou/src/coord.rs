//! Cross-actor coordination behind the `Coordinator` trait (`NAPKIN_MATH.md` §8.A): named
//! barriers, departures, and (for several hosts) the start gate and the end-of-run reduction.
//! `Local` is the in-process implementation for a single host. `Tcp` is the client every host
//! of a multi-host run uses, and `Server` is the coordinator rank 0 runs in-process beside its
//! own client: star topology, blocking `std::net`, one reader thread per socket, frames of a
//! big-endian `u32` length and a JSON body (`Msg`), heartbeats while idle, a configuration
//! hash in `Hello` that refuses a host whose run differs before any I/O starts.
//!
//! A barrier's participants are fixed before the run: every instance of every actor template
//! whose body contains `barrier {scope}` (outside any `parallel` or `loader`). An instance
//! that finishes leaves its barriers; if the remaining arrivals then complete a generation,
//! it is released and counted as a departure release, which the report shows because it means
//! the abstract's instances did not all hit the barrier the same number of times. Across hosts
//! the barrier is two-level: a host's instances arrive locally, the last one tells the server
//! (`Arrive`), the server releases every host once each host with participants has arrived
//! (`Release`), and a host whose last participant left tells the server so (`Leave`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::run::{unix_now, Report};

pub trait Coordinator: Send + Sync {
    /// Wait until every participant of `scope` has arrived. Returns the time spent waiting.
    fn barrier(&self, scope: &str, aborted: &AtomicBool) -> Result<Duration>;
    /// The non-blocking half for an event loop: arrive, and get the generation this arrival
    /// belongs to; `released` says when it has completed. Every release and every abort
    /// writes to each eventfd given to `subscribe`, so a loop waiting in `io_uring_enter`
    /// with a read posted on its eventfd wakes up (`NAPKIN_MATH.md` §8.A).
    fn arrive(&self, scope: &str) -> Result<u64>;
    fn released(&self, scope: &str, generation: u64) -> bool;
    fn subscribe(&self, eventfd: RawFd);
    fn unsubscribe(&self, eventfd: RawFd);
    /// This participant will never arrive at `scope` again.
    fn leave(&self, scope: &str);
    /// Releases by departure so far, per scope.
    fn departure_releases(&self) -> Vec<(String, u64)>;
    /// Tell every other host the run failed here (nothing to do on one host).
    fn stop(&self, _reason: &str) {}
    /// Why the run was aborted from outside this process, if it was.
    fn abort_reason(&self) -> Option<String> {
        None
    }
}

// ---------------------------------------------------------------- one host

struct Bar {
    expected: usize,
    arrived: usize,
    generation: u64,
    departure_releases: u64,
}

pub struct Local {
    bars: Mutex<HashMap<String, Bar>>,
    cv: Condvar,
    efds: Eventfds,
}

/// The eventfds of the event loops waiting on this coordinator (`Coordinator::subscribe`).
#[derive(Default)]
pub(crate) struct Eventfds(Mutex<Vec<RawFd>>);

impl Eventfds {
    pub(crate) fn add(&self, fd: RawFd) {
        self.0.lock().unwrap().push(fd);
    }
    pub(crate) fn remove(&self, fd: RawFd) {
        self.0.lock().unwrap().retain(|f| *f != fd);
    }
    /// Wake every subscribed loop.
    pub(crate) fn kick(&self) {
        let one: u64 = 1;
        for fd in self.0.lock().unwrap().iter() {
            unsafe { libc::write(*fd, &one as *const u64 as *const libc::c_void, 8) };
        }
    }
}

impl Local {
    pub fn new(participants: &[(String, usize)]) -> Self {
        let bars = participants
            .iter()
            .map(|(s, n)| (s.clone(), Bar { expected: *n, arrived: 0, generation: 0, departure_releases: 0 }))
            .collect();
        Local { bars: Mutex::new(bars), cv: Condvar::new(), efds: Eventfds::default() }
    }
}

impl Coordinator for Local {
    fn barrier(&self, scope: &str, aborted: &AtomicBool) -> Result<Duration> {
        let t = Instant::now();
        let my_gen = self.arrive(scope)?;
        let mut bars = self.bars.lock().unwrap();
        loop {
            let (guard, _) = self.cv.wait_timeout(bars, Duration::from_millis(50)).unwrap();
            bars = guard;
            if bars.get(scope).map_or(true, |b| b.generation != my_gen) {
                return Ok(t.elapsed());
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("barrier `{scope}`: run aborted while waiting");
            }
        }
    }

    fn arrive(&self, scope: &str) -> Result<u64> {
        let mut bars = self.bars.lock().unwrap();
        let Some(bar) = bars.get_mut(scope) else { bail!("barrier `{scope}`: no participants registered") };
        let my_gen = bar.generation;
        bar.arrived += 1;
        if bar.arrived >= bar.expected {
            bar.arrived = 0;
            bar.generation += 1;
            self.cv.notify_all();
            self.efds.kick();
        }
        Ok(my_gen)
    }

    fn released(&self, scope: &str, generation: u64) -> bool {
        self.bars.lock().unwrap().get(scope).map_or(true, |b| b.generation != generation)
    }

    fn subscribe(&self, eventfd: RawFd) {
        self.efds.add(eventfd);
    }

    fn unsubscribe(&self, eventfd: RawFd) {
        self.efds.remove(eventfd);
    }

    fn leave(&self, scope: &str) {
        let mut bars = self.bars.lock().unwrap();
        if let Some(bar) = bars.get_mut(scope) {
            bar.expected = bar.expected.saturating_sub(1);
            if bar.arrived > 0 && bar.arrived >= bar.expected {
                bar.arrived = 0;
                bar.generation += 1;
                bar.departure_releases += 1;
                self.cv.notify_all();
                self.efds.kick();
            }
        }
    }

    fn departure_releases(&self) -> Vec<(String, u64)> {
        let bars = self.bars.lock().unwrap();
        let mut v: Vec<_> = bars.iter().filter(|(_, b)| b.departure_releases > 0).map(|(s, b)| (s.clone(), b.departure_releases)).collect();
        v.sort();
        v
    }
}

// ---------------------------------------------------------------- the wire

/// How long hosts may take to connect after the server (or the first client) is up.
pub const CONNECT_WINDOW: Duration = Duration::from_secs(120);
/// A heartbeat goes out after this long without anything to send; a peer silent for
/// `DEAD_AFTER` is gone and the run aborts on every host.
const HEARTBEAT: Duration = Duration::from_secs(5);
const DEAD_AFTER: Duration = Duration::from_secs(30);
const MAX_FRAME: usize = 256 << 20;

#[derive(Debug, Serialize, Deserialize)]
enum Msg {
    /// Client → server, first. `identity` is everything that shapes the run (the abstract,
    /// seed, G, parameters, dataset ids, backend, rotation), compared by its canonical hash
    /// and, when it differs, field by field; `layers` is the host's options block
    /// (`options::Layers::json`), recorded per rank and compared after the gate;
    /// `participants` are this host's barrier scopes with its instance counts.
    Hello { rank: i64, ranks: i64, host: String, identity: Value, layers: Value, participants: Vec<(String, usize)> },
    Welcome,
    /// Client → server once its startup checks are done.
    Ready,
    /// Server → all once every host is ready: the common start time and the host list by rank.
    Start { t0: f64, hosts: Vec<String> },
    Arrive { scope: String },
    Leave { scope: String },
    Release { scope: String, generation: u64 },
    Report(Box<Report>),
    /// Server → all after the reduction: the verdict rank 0 reached on the merged report.
    Result { ok: bool, fingerprint: u64, error: Option<String> },
    Stop { reason: String },
    Heartbeat,
}

fn send(w: &Mutex<TcpStream>, m: &Msg) -> Result<()> {
    let body = serde_json::to_vec(m)?;
    let mut s = w.lock().unwrap();
    s.write_all(&(body.len() as u32).to_be_bytes())?;
    s.write_all(&body)?;
    Ok(())
}

/// Fill `buf` from the socket, calling `idle` on every read timeout (it sends a heartbeat or
/// declares the peer dead).
fn read_full(s: &mut TcpStream, buf: &mut [u8], idle: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        match s.read(&mut buf[filled..]) {
            Ok(0) => bail!("connection closed"),
            Ok(n) => filled += n,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => idle()?,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn recv(s: &mut TcpStream, idle: &mut dyn FnMut() -> Result<()>) -> Result<Msg> {
    let mut hdr = [0u8; 4];
    read_full(s, &mut hdr, idle)?;
    let len = u32::from_be_bytes(hdr) as usize;
    if len > MAX_FRAME {
        bail!("frame of {len} bytes");
    }
    let mut body = vec![0u8; len];
    read_full(s, &mut body, idle)?;
    Ok(serde_json::from_slice(&body)?)
}

/// The idle handler of a reader thread: heartbeat the peer, give up on it after `DEAD_AFTER`.
struct Liveness {
    last: Instant,
    writer: Arc<Mutex<TcpStream>>,
    peer: String,
}

impl Liveness {
    fn idle(&mut self) -> Result<()> {
        if self.last.elapsed() > DEAD_AFTER {
            bail!("{}: no message for {:?}", self.peer, DEAD_AFTER);
        }
        send(&self.writer, &Msg::Heartbeat).with_context(|| format!("{}: heartbeat", self.peer))
    }
    fn seen(&mut self) {
        self.last = Instant::now();
    }
}

fn configure(s: &TcpStream) -> Result<()> {
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(HEARTBEAT))?;
    Ok(())
}

// ---------------------------------------------------------------- the client

struct TBar {
    expected: usize,
    arrived: usize,
    /// The server's release count for this scope, as last heard.
    generation: u64,
    departure_releases: u64,
    /// `Leave` sent: no participant of this host will arrive again.
    left: bool,
}

#[derive(Default)]
struct Events {
    welcome: bool,
    started: Option<(f64, Vec<String>)>,
    result: Option<(bool, u64, Option<String>)>,
}

struct ClientState {
    bars: Mutex<HashMap<String, TBar>>,
    cv: Condvar,
    events: Mutex<Events>,
    ecv: Condvar,
    writer: Arc<Mutex<TcpStream>>,
    aborted: Arc<AtomicBool>,
    reason: Mutex<Option<String>>,
    efds: Eventfds,
}

impl ClientState {
    fn fail(&self, reason: String) {
        let mut r = self.reason.lock().unwrap();
        if r.is_none() {
            *r = Some(reason);
        }
        self.aborted.store(true, Ordering::Relaxed);
        self.cv.notify_all();
        self.ecv.notify_all();
        self.efds.kick();
    }

    fn failure(&self) -> Option<String> {
        self.reason.lock().unwrap().clone()
    }

    /// Wait until `done` holds or the run was aborted.
    fn wait_event<T>(&self, mut done: impl FnMut(&Events) -> Option<T>) -> Result<T> {
        let mut ev = self.events.lock().unwrap();
        loop {
            if let Some(t) = done(&ev) {
                return Ok(t);
            }
            if self.aborted.load(Ordering::Relaxed) {
                return Err(anyhow!("{}", self.failure().unwrap_or_else(|| "run aborted".into())));
            }
            ev = self.ecv.wait_timeout(ev, Duration::from_millis(50)).unwrap().0;
        }
    }
}

/// A host's connection to the coordinator.
pub struct Tcp {
    st: Arc<ClientState>,
    pub rank: i64,
    pub ranks: i64,
    pub addr: SocketAddr,
    _reader: JoinHandle<()>,
}

impl Tcp {
    /// Connect (retrying for `CONNECT_WINDOW`), send `Hello`, and wait for the server's
    /// `Welcome`, which it sends once the identity matches the first host's.
    #[allow(clippy::too_many_arguments)]
    pub fn connect(addr: &str, rank: i64, ranks: i64, host: &str, identity: &Value, layers: &Value, participants: &[(String, usize)], aborted: Arc<AtomicBool>) -> Result<Tcp> {
        let sock = resolve(addr)?;
        let deadline = Instant::now() + CONNECT_WINDOW;
        let stream = loop {
            match TcpStream::connect_timeout(&sock, Duration::from_secs(2)) {
                Ok(s) => break s,
                Err(e) => {
                    if Instant::now() >= deadline {
                        bail!("coordinator {sock}: {e} (gave up after {CONNECT_WINDOW:?})");
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        };
        configure(&stream)?;
        let writer = Arc::new(Mutex::new(stream.try_clone()?));
        let bars = participants
            .iter()
            .map(|(s, n)| (s.clone(), TBar { expected: *n, arrived: 0, generation: 0, departure_releases: 0, left: false }))
            .collect();
        let st = Arc::new(ClientState {
            bars: Mutex::new(bars),
            cv: Condvar::new(),
            events: Mutex::new(Events::default()),
            ecv: Condvar::new(),
            writer: writer.clone(),
            aborted,
            reason: Mutex::new(None),
            efds: Eventfds::default(),
        });
        send(&writer, &Msg::Hello { rank, ranks, host: host.to_string(), identity: identity.clone(), layers: layers.clone(), participants: participants.to_vec() })?;
        let reader = {
            let st = st.clone();
            let mut stream = stream;
            let mut live = Liveness { last: Instant::now(), writer, peer: format!("coordinator {sock}") };
            std::thread::Builder::new()
                .name("coord-client".into())
                .spawn(move || {
                    if let Err(e) = client_loop(&st, &mut stream, &mut live) {
                        st.fail(format!("{e:#}"));
                    }
                })
                .expect("spawn coordinator client thread")
        };
        st.wait_event(|e| e.welcome.then_some(())).with_context(|| format!("coordinator {sock} refused rank {rank}"))?;
        Ok(Tcp { st, rank, ranks, addr: sock, _reader: reader })
    }

    /// Startup checks are done here; blocks until every host says so. Returns the common
    /// start time and the hosts by rank.
    pub fn ready(&self) -> Result<(f64, Vec<String>)> {
        send(&self.st.writer, &Msg::Ready)?;
        self.st.wait_event(|e| e.started.clone())
    }

    /// Send this host's report for the reduction.
    pub fn report(&self, r: &Report) -> Result<()> {
        send(&self.st.writer, &Msg::Report(Box::new(r.clone())))
    }

    /// Wait for rank 0's verdict on the merged report: `(ok, fingerprint, error)`.
    pub fn result(&self) -> Result<(bool, u64, Option<String>)> {
        self.st.wait_event(|e| e.result.clone())
    }
}

fn client_loop(st: &ClientState, stream: &mut TcpStream, live: &mut Liveness) -> Result<()> {
    loop {
        let m = {
            let mut idle = || live.idle();
            recv(stream, &mut idle)?
        };
        live.seen();
        match m {
            Msg::Welcome => {
                st.events.lock().unwrap().welcome = true;
                st.ecv.notify_all();
            }
            Msg::Start { t0, hosts } => {
                st.events.lock().unwrap().started = Some((t0, hosts));
                st.ecv.notify_all();
            }
            Msg::Release { scope, generation } => {
                if let Some(b) = st.bars.lock().unwrap().get_mut(&scope) {
                    b.generation = generation;
                }
                st.cv.notify_all();
                st.efds.kick();
            }
            Msg::Result { ok, fingerprint, error } => {
                st.events.lock().unwrap().result = Some((ok, fingerprint, error));
                st.ecv.notify_all();
                return Ok(());
            }
            Msg::Stop { reason } => bail!("{reason}"),
            Msg::Heartbeat => {}
            other => bail!("unexpected message from the coordinator: {other:?}"),
        }
    }
}

impl Coordinator for Tcp {
    fn barrier(&self, scope: &str, aborted: &AtomicBool) -> Result<Duration> {
        let t = Instant::now();
        let my_gen = self.arrive(scope)?;
        let mut bars = self.st.bars.lock().unwrap();
        loop {
            if bars.get(scope).map_or(true, |b| b.generation > my_gen) {
                return Ok(t.elapsed());
            }
            if aborted.load(Ordering::Relaxed) || self.st.aborted.load(Ordering::Relaxed) {
                bail!("barrier `{scope}`: {}", self.st.failure().unwrap_or_else(|| "run aborted while waiting".into()));
            }
            bars = self.st.cv.wait_timeout(bars, Duration::from_millis(50)).unwrap().0;
        }
    }

    fn arrive(&self, scope: &str) -> Result<u64> {
        let mut bars = self.st.bars.lock().unwrap();
        let Some(bar) = bars.get_mut(scope) else { bail!("barrier `{scope}`: no participants registered on this host") };
        let my_gen = bar.generation;
        bar.arrived += 1;
        if bar.arrived >= bar.expected {
            bar.arrived = 0;
            send(&self.st.writer, &Msg::Arrive { scope: scope.to_string() }).with_context(|| format!("barrier `{scope}`"))?;
        }
        Ok(my_gen)
    }

    fn released(&self, scope: &str, generation: u64) -> bool {
        self.st.bars.lock().unwrap().get(scope).map_or(true, |b| b.generation > generation)
    }

    fn subscribe(&self, eventfd: RawFd) {
        self.st.efds.add(eventfd);
    }

    fn unsubscribe(&self, eventfd: RawFd) {
        self.st.efds.remove(eventfd);
    }

    fn leave(&self, scope: &str) {
        let mut bars = self.st.bars.lock().unwrap();
        let Some(bar) = bars.get_mut(scope) else { return };
        bar.expected = bar.expected.saturating_sub(1);
        if bar.arrived > 0 && bar.arrived >= bar.expected {
            bar.arrived = 0;
            bar.departure_releases += 1;
            if let Err(e) = send(&self.st.writer, &Msg::Arrive { scope: scope.to_string() }) {
                self.st.fail(format!("barrier `{scope}`: {e:#}"));
            }
        }
        if bar.expected == 0 && bar.arrived == 0 && !bar.left {
            bar.left = true;
            if let Err(e) = send(&self.st.writer, &Msg::Leave { scope: scope.to_string() }) {
                self.st.fail(format!("barrier `{scope}`: {e:#}"));
            }
        }
    }

    fn departure_releases(&self) -> Vec<(String, u64)> {
        let bars = self.st.bars.lock().unwrap();
        let mut v: Vec<_> = bars.iter().filter(|(_, b)| b.departure_releases > 0).map(|(s, b)| (s.clone(), b.departure_releases)).collect();
        v.sort();
        v
    }

    fn stop(&self, reason: &str) {
        let _ = send(&self.st.writer, &Msg::Stop { reason: reason.to_string() });
    }

    fn abort_reason(&self) -> Option<String> {
        self.st.failure()
    }
}

fn resolve(addr: &str) -> Result<SocketAddr> {
    addr.to_socket_addrs().with_context(|| format!("coordinator address `{addr}`"))?.next().ok_or_else(|| anyhow!("coordinator address `{addr}` resolves to nothing"))
}

// ---------------------------------------------------------------- the server

struct SBar {
    /// Hosts with participants that have not left.
    expected: usize,
    arrived: HashSet<i64>,
    generation: u64,
    departure_releases: u64,
}

#[derive(Default)]
struct SState {
    conns: BTreeMap<i64, Arc<Mutex<TcpStream>>>,
    /// The first host's identity (its rank, the canonical hash, the document).
    identity: Option<(i64, String, Value)>,
    hosts: BTreeMap<i64, String>,
    /// Each host's options block, from its `Hello`.
    layers: BTreeMap<i64, Value>,
    ready: HashSet<i64>,
    started: Option<f64>,
    bars: BTreeMap<String, SBar>,
    reports: BTreeMap<i64, Report>,
    failed: Option<String>,
    finished: bool,
    /// Connections that have gone away.
    closed: usize,
}

struct ServerState {
    ranks: i64,
    m: Mutex<SState>,
    cv: Condvar,
}

impl ServerState {
    fn broadcast(&self, s: &SState, m: &Msg) {
        for w in s.conns.values() {
            let _ = send(w, m);
        }
    }

    fn fail(&self, s: &mut SState, reason: String) {
        if s.finished {
            return;
        }
        if s.failed.is_none() {
            s.failed = Some(reason.clone());
            self.broadcast(s, &Msg::Stop { reason });
        }
        self.cv.notify_all();
    }

    fn check_release(&self, s: &mut SState, scope: &str, by_departure: bool) {
        let Some(b) = s.bars.get_mut(scope) else { return };
        if !b.arrived.is_empty() && b.arrived.len() >= b.expected {
            b.arrived.clear();
            b.generation += 1;
            if by_departure {
                b.departure_releases += 1;
            }
            let m = Msg::Release { scope: scope.to_string(), generation: b.generation };
            self.broadcast(s, &m);
        }
    }
}

/// The coordinator rank 0 runs in-process. `start` binds and accepts `ranks` connections
/// in the background; `merged` waits for every host's report.
pub struct Server {
    st: Arc<ServerState>,
    pub addr: SocketAddr,
}

impl Server {
    pub fn start(addr: &str, ranks: i64) -> Result<Server> {
        let listener = TcpListener::bind(addr).with_context(|| format!("coordinator: binding {addr}"))?;
        let bound = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let st = Arc::new(ServerState { ranks, m: Mutex::new(SState::default()), cv: Condvar::new() });
        {
            let st = st.clone();
            std::thread::Builder::new()
                .name("coord-accept".into())
                .spawn(move || accept_loop(st, listener, ranks))
                .expect("spawn coordinator accept thread");
        }
        Ok(Server { st, addr: bound })
    }

    /// Every host's rank, name, and options block (`Hello`), by rank; complete once the start
    /// gate has opened.
    pub fn hosts(&self) -> Vec<(i64, String, Value)> {
        let s = self.st.m.lock().unwrap();
        s.hosts.iter().map(|(r, h)| (*r, h.clone(), s.layers.get(r).cloned().unwrap_or(Value::Null))).collect()
    }

    /// Host-level departure releases (a host whose participants all left completed a
    /// generation), per scope.
    pub fn departure_releases(&self) -> Vec<(String, u64)> {
        let s = self.st.m.lock().unwrap();
        s.bars.iter().filter(|(_, b)| b.departure_releases > 0).map(|(k, b)| (k.clone(), b.departure_releases)).collect()
    }

    /// Wait for every host's report and merge them (`Report::merge_all`).
    pub fn merged(&self) -> Result<Report> {
        let mut s = self.st.m.lock().unwrap();
        loop {
            if let Some(r) = &s.failed {
                bail!("{r}");
            }
            if s.reports.len() as i64 == self.st.ranks {
                let reports: Vec<Report> = std::mem::take(&mut s.reports).into_values().collect();
                let mut merged = Report::merge_all(reports);
                for (scope, n) in self.departure_releases_locked(&s) {
                    match merged.departure_releases.iter_mut().find(|(k, _)| *k == scope) {
                        Some(e) => e.1 += n,
                        None => merged.departure_releases.push((scope, n)),
                    }
                }
                merged.departure_releases.sort();
                return Ok(merged);
            }
            s = self.st.cv.wait_timeout(s, Duration::from_millis(50)).unwrap().0;
        }
    }

    fn departure_releases_locked(&self, s: &SState) -> Vec<(String, u64)> {
        s.bars.iter().filter(|(_, b)| b.departure_releases > 0).map(|(k, b)| (k.clone(), b.departure_releases)).collect()
    }

    /// Rank 0's verdict on the merged report, sent to every host; the last message. Waits
    /// (briefly) for the other hosts to close, so that rank 0 exiting with their heartbeats
    /// unread cannot reset a connection before its `Result` was read.
    pub fn finish(&self, ok: bool, fingerprint: u64, error: Option<String>) {
        let mut s = self.st.m.lock().unwrap();
        s.finished = true;
        self.st.broadcast(&s, &Msg::Result { ok, fingerprint, error });
        let deadline = Instant::now() + Duration::from_secs(5);
        while (s.closed as i64) < self.st.ranks - 1 && Instant::now() < deadline {
            s = self.st.cv.wait_timeout(s, Duration::from_millis(20)).unwrap().0;
        }
    }

    /// Abort every host (the server's own run failed before the reduction).
    pub fn stop(&self, reason: &str) {
        let mut s = self.st.m.lock().unwrap();
        self.st.fail(&mut s, reason.to_string());
    }
}

fn accept_loop(st: Arc<ServerState>, listener: TcpListener, ranks: i64) {
    let deadline = Instant::now() + CONNECT_WINDOW;
    let mut accepted = 0;
    while accepted < ranks {
        match listener.accept() {
            Ok((stream, peer)) => {
                accepted += 1;
                let st = st.clone();
                std::thread::Builder::new()
                    .name(format!("coord-{peer}"))
                    .spawn(move || serve(st, stream, peer))
                    .expect("spawn coordinator connection thread");
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted) => {
                if Instant::now() >= deadline {
                    let mut s = st.m.lock().unwrap();
                    st.fail(&mut s, format!("coordinator: {accepted} of {ranks} hosts connected within {CONNECT_WINDOW:?}"));
                    return;
                }
                if st.m.lock().unwrap().failed.is_some() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let mut s = st.m.lock().unwrap();
                st.fail(&mut s, format!("coordinator: accept: {e}"));
                return;
            }
        }
    }
}

fn serve(st: Arc<ServerState>, stream: TcpStream, peer: SocketAddr) {
    let mut stream = stream;
    if let Err(e) = configure(&stream) {
        let mut s = st.m.lock().unwrap();
        st.fail(&mut s, format!("coordinator: {peer}: {e}"));
        return;
    }
    let writer = match stream.try_clone() {
        Ok(w) => Arc::new(Mutex::new(w)),
        Err(e) => {
            let mut s = st.m.lock().unwrap();
            st.fail(&mut s, format!("coordinator: {peer}: {e}"));
            return;
        }
    };
    let mut live = Liveness { last: Instant::now(), writer: writer.clone(), peer: peer.to_string() };
    let mut who = peer.to_string();
    let r = serve_loop(&st, &mut stream, &writer, &mut live, &mut who);
    let mut s = st.m.lock().unwrap();
    if let Err(e) = r {
        if !s.finished {
            st.fail(&mut s, format!("{who}: {e:#}"));
        }
    }
    s.closed += 1;
    st.cv.notify_all();
}

/// The fields of `mine` that differ from rank `r0`'s `theirs`, one line each; `params` is
/// compared parameter by parameter.
fn identity_differences(mine: &Value, theirs: &Value, r0: i64) -> Vec<String> {
    let mut out = Vec::new();
    let keys: std::collections::BTreeSet<&String> = mine.as_object().into_iter().chain(theirs.as_object()).flat_map(|o| o.keys()).collect();
    for k in keys {
        let (a, b) = (&mine[k], &theirs[k]);
        if a == b {
            continue;
        }
        if k == "params" && a.is_object() && b.is_object() {
            let names: std::collections::BTreeSet<&String> = a.as_object().into_iter().chain(b.as_object()).flat_map(|o| o.keys()).collect();
            for n in names {
                if a[n] != b[n] {
                    out.push(format!("params.{n} {} differs from rank {r0}'s {}", a[n], b[n]));
                }
            }
        } else {
            out.push(format!("{k} {a} differs from rank {r0}'s {b}"));
        }
    }
    if out.is_empty() {
        out.push(format!("identity differs from rank {r0}'s"));
    }
    out
}

fn serve_loop(st: &ServerState, stream: &mut TcpStream, writer: &Arc<Mutex<TcpStream>>, live: &mut Liveness, who: &mut String) -> Result<()> {
    let rank = {
        let mut idle = || live.idle();
        let m = recv(stream, &mut idle)?;
        live.seen();
        let Msg::Hello { rank, ranks, host, identity, layers, participants } = m else { bail!("expected Hello, got {m:?}") };
        *who = format!("rank {rank} ({host})");
        let hash = crate::canon::sha256_hex(&identity);
        let mut s = st.m.lock().unwrap();
        let refusal = if ranks != st.ranks {
            Some(format!("{who}: --ranks {ranks}, the coordinator was started for {}", st.ranks))
        } else if rank < 0 || rank >= st.ranks || s.conns.contains_key(&rank) {
            Some(format!("{who}: rank {rank} is out of range or already connected"))
        } else {
            match &s.identity {
                Some((r0, h0, id0)) if *h0 != hash => Some(format!("{who}: not the same run: {}", identity_differences(&identity, &id0.clone(), *r0).join("; "))),
                _ => None,
            }
        };
        if let Some(e) = refusal {
            // the refused host is not registered, so it hears the reason on its own socket
            let _ = send(writer, &Msg::Stop { reason: e.clone() });
            st.fail(&mut s, e.clone());
            bail!("{e}");
        }
        if s.identity.is_none() {
            s.identity = Some((rank, hash, identity));
        }
        s.layers.insert(rank, layers);
        s.conns.insert(rank, writer.clone());
        s.hosts.insert(rank, host);
        for (scope, n) in participants {
            if n > 0 {
                s.bars.entry(scope).or_insert_with(|| SBar { expected: 0, arrived: HashSet::new(), generation: 0, departure_releases: 0 }).expected += 1;
            }
        }
        send(writer, &Msg::Welcome)?;
        rank
    };
    loop {
        let m = {
            let mut idle = || live.idle();
            recv(stream, &mut idle)?
        };
        live.seen();
        let mut s = st.m.lock().unwrap();
        if s.failed.is_some() {
            return Ok(());
        }
        match m {
            Msg::Ready => {
                s.ready.insert(rank);
                if s.ready.len() as i64 == st.ranks {
                    let t0 = unix_now();
                    s.started = Some(t0);
                    let hosts: Vec<String> = s.hosts.values().cloned().collect();
                    st.broadcast(&s, &Msg::Start { t0, hosts });
                }
            }
            Msg::Arrive { scope } => {
                let Some(b) = s.bars.get_mut(&scope) else { bail!("Arrive at `{scope}`, which no host registered") };
                b.arrived.insert(rank);
                st.check_release(&mut s, &scope, false);
            }
            Msg::Leave { scope } => {
                let Some(b) = s.bars.get_mut(&scope) else { bail!("Leave `{scope}`, which no host registered") };
                b.expected = b.expected.saturating_sub(1);
                b.arrived.remove(&rank);
                st.check_release(&mut s, &scope, true);
            }
            Msg::Report(r) => {
                s.reports.insert(rank, *r);
                st.cv.notify_all();
            }
            Msg::Stop { reason } => {
                st.fail(&mut s, format!("{who}: {reason}"));
                return Ok(());
            }
            Msg::Heartbeat => {}
            other => bail!("unexpected message: {other:?}"),
        }
        if s.finished {
            return Ok(());
        }
    }
}
