"""The coordinator of a datagen over several hosts, as `aeiou datagen` has it
(`runner/aeiou/src/coord.rs`, `runner/REFERENCE.md` §5, §6): rank 0 runs the server in
process and every rank, rank 0 included, connects to it as a client.

The wire is the runner's: a frame is a big-endian `u32` length and a JSON body, each message
serde's externally tagged form of `coord.rs`'s `Msg` (a unit variant a bare string,
`"Ready"`; any other an object of one key, `{"Done": {...}}`). A datagen uses `Hello`,
`Welcome`, `Ready`, `Start`, `Done`, `Result`, `Stop`, and `Heartbeat`, and no barrier: a
host says `Hello` with its identity (whatever shapes the corpus, compared by its canonical
hash and, when it differs, field by field), is welcomed, says `Ready` once its roots are
checked, and writes when the server says `Start`, the gate; then it sends `Done` with its
counts, which rank 0 gathers to write the manifests, and every host exits with the `Result`
rank 0 sends. A host silent for `DEAD_AFTER` is gone, and a failure anywhere stops every host.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
import struct
import threading
import time

from .nodes import BuildError

CONNECT_WINDOW = 120.0   # seconds hosts may take to connect
HEARTBEAT = 5.0          # a heartbeat goes out after this long without a message
DEAD_AFTER = 30.0        # a peer silent this long is gone
MAX_FRAME = 256 << 20


def identity_hash(identity) -> str:
    """`canon::sha256_hex`: the canonical JSON (sorted keys, no spaces, ASCII)."""
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def identity_differences(mine: dict, theirs: dict, r0: int) -> list:
    """The fields of `mine` that differ from rank `r0`'s `theirs`, one line each; `params`
    parameter by parameter (`coord.rs` `identity_differences`)."""
    show = lambda v: json.dumps(v, separators=(",", ":"))  # noqa: E731
    out = []
    for k in sorted(set(mine) | set(theirs)):
        a, b = mine.get(k), theirs.get(k)
        if a == b:
            continue
        if k == "params" and isinstance(a, dict) and isinstance(b, dict):
            for n in sorted(set(a) | set(b)):
                if a.get(n) != b.get(n):
                    out.append(f"params.{n} {show(a.get(n))} differs from rank {r0}'s {show(b.get(n))}")
        else:
            out.append(f"{k} {show(a)} differs from rank {r0}'s {show(b)}")
    return out or [f"identity differs from rank {r0}'s"]


def host_facts() -> dict:
    """What a host brings, as `run::host_facts` says it in the runner's `Hello`: host, kernel,
    CPUs, `MemTotal`, and the tightest cgroup v2 `memory.max` on the way to the root."""
    def mem_total():
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    def cgroup_max():
        try:
            rel = next(l[3:].strip() for l in open("/proc/self/cgroup") if l.startswith("0::"))
        except (OSError, StopIteration):
            return None
        best, d, top = None, pathlib.Path("/sys/fs/cgroup") / rel.lstrip("/"), pathlib.Path("/sys/fs/cgroup")
        while True:
            try:
                n = int((d / "memory.max").read_text().strip())
                best = n if best is None else min(best, n)
            except (OSError, ValueError):
                pass
            if d == top or d.parent == d:
                return best
            d = d.parent

    online = os.sysconf("SC_NPROCESSORS_ONLN") if hasattr(os, "sysconf") else -1
    return {"host": socket.gethostname(), "kernel": os.uname().release, "cpus": len(os.sched_getaffinity(0)),
            "cpus_online": online if online > 0 else None, "memory": mem_total(), "cgroup_memory_max": cgroup_max()}


class Closed(Exception):
    pass


class Conn:
    """One end of a connection: a writer lock, and a reader that heartbeats the peer on every
    read timeout and gives up on it after `DEAD_AFTER`."""

    def __init__(self, sock: socket.socket, peer: str):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(HEARTBEAT)
        self.sock, self.peer = sock, peer
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def send(self, msg) -> None:
        body = json.dumps(msg, separators=(",", ":")).encode()
        with self.lock:
            self.sock.sendall(struct.pack(">I", len(body)) + body)

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except (socket.timeout, InterruptedError):
                if time.monotonic() - self.last > DEAD_AFTER:
                    raise Closed(f"{self.peer}: no message for {DEAD_AFTER:.0f}s") from None
                self.send("Heartbeat")
                continue
            if not chunk:
                raise Closed("connection closed")
            buf += chunk
        return bytes(buf)

    def recv(self):
        (n,) = struct.unpack(">I", self._read(4))
        if n > MAX_FRAME:
            raise Closed(f"frame of {n} bytes")
        msg = json.loads(self._read(n))
        self.last = time.monotonic()
        return msg

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _variant(msg):
    """(name, body) of a message in serde's externally tagged form."""
    if isinstance(msg, str):
        return msg, None
    if isinstance(msg, dict) and len(msg) == 1:
        return next(iter(msg.items()))
    raise Closed(f"not a message: {msg!r}")


def _split(addr: str) -> tuple:
    host, sep, port = addr.rpartition(":")
    if not sep or not port.isdigit():
        raise BuildError(f"coordinator address `{addr}`: expected HOST:PORT")
    return host.strip("[]") or "0.0.0.0", int(port)


# ---------------------------------------------------------------- the client

class Client:
    """A host's connection to the coordinator (`coord.rs` `Tcp`)."""

    def __init__(self, addr: str, rank: int, ranks: int, host: str, identity: dict, layers: dict):
        self.rank, self.ranks = rank, ranks
        self.cv = threading.Condition()
        self.welcome = False
        self.started = None
        self.result_ = None
        self.reason = None
        target = _split(addr)
        deadline = time.monotonic() + CONNECT_WINDOW
        while True:
            try:
                sock = socket.create_connection(target, timeout=2)
                break
            except OSError as e:
                if time.monotonic() >= deadline:
                    raise BuildError(f"coordinator {addr}: {e} (gave up after {CONNECT_WINDOW:.0f}s)") from None
                time.sleep(0.5)
        self.addr = "%s:%d" % sock.getpeername()[:2]
        self.conn = Conn(sock, f"coordinator {self.addr}")
        self.conn.send({"Hello": {"rank": rank, "ranks": ranks, "host": host, "identity": identity, "layers": layers, "participants": [],
                                 "facts": host_facts()}})
        threading.Thread(target=self._loop, name="coord-client", daemon=True).start()
        try:
            self._wait(lambda: self.welcome or None)
        except BuildError as e:
            raise BuildError(f"coordinator {self.addr} refused rank {rank}: {e}") from None

    def _fail(self, reason: str) -> None:
        with self.cv:
            if self.reason is None:
                self.reason = reason
            self.cv.notify_all()

    def _loop(self) -> None:
        try:
            while True:
                name, body = _variant(self.conn.recv())
                with self.cv:
                    if name == "Welcome":
                        self.welcome = True
                    elif name == "Start":
                        self.started = (body["t0"], body["hosts"])
                    elif name == "Result":
                        self.result_ = (body["ok"], body.get("fingerprint", 0), body.get("error"))
                        self.cv.notify_all()
                        return
                    elif name == "Stop":
                        raise Closed(body["reason"])
                    elif name in ("Heartbeat", "Release"):
                        pass
                    else:
                        raise Closed(f"unexpected message from the coordinator: {name}")
                    self.cv.notify_all()
        except (Closed, OSError, ValueError) as e:
            self._fail(str(e))

    def _wait(self, done):
        with self.cv:
            while True:
                v = done()
                if v is not None:
                    return v
                if self.reason is not None:
                    raise BuildError(self.reason)
                self.cv.wait(0.05)

    def ready(self) -> tuple:
        """This host's checks are done; blocks until every host's are (the start gate).
        Returns the common start time and the hosts by rank."""
        self.conn.send("Ready")
        return self._wait(lambda: self.started)

    def done(self, doc: dict) -> None:
        """This host's part, for rank 0 to gather."""
        self.conn.send({"Done": doc})

    def result(self) -> tuple:
        """Rank 0's verdict: `(ok, fingerprint, error)`."""
        return self._wait(lambda: self.result_)

    def stop(self, reason: str) -> None:
        """This host failed: the coordinator relays it to every other host."""
        try:
            self.conn.send({"Stop": {"reason": reason}})
        except OSError:
            pass

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------- the server

class Server:
    """The coordinator rank 0 runs in process (`coord.rs` `Server`): accepts `ranks`
    connections within `CONNECT_WINDOW`, checks each `Hello` against the first, opens the gate
    when every host is ready, and gathers every host's `Done`."""

    def __init__(self, addr: str, ranks: int):
        self.ranks = ranks
        self.cv = threading.Condition()
        self.conns: dict = {}
        self.identity = None   # (rank, hash, document) of the first host
        self.hosts: dict = {}
        self.layers: dict = {}
        self.ready_: set = set()
        self.done_: dict = {}
        self.failed = None
        self.finished = False
        self.closed = 0
        try:
            self.listener = socket.create_server(_split(addr), reuse_port=False)
        except OSError as e:
            raise BuildError(f"coordinator: binding {addr}: {e}") from None
        self.addr = "%s:%d" % self.listener.getsockname()[:2]
        threading.Thread(target=self._accept, name="coord-accept", daemon=True).start()

    def _broadcast(self, msg) -> None:
        for c in self.conns.values():
            try:
                c.send(msg)
            except OSError:
                pass

    def _fail(self, reason: str) -> None:
        """Under `cv`."""
        if self.finished:
            return
        if self.failed is None:
            self.failed = reason
            self._broadcast({"Stop": {"reason": reason}})
        self.cv.notify_all()

    def _accept(self) -> None:
        deadline = time.monotonic() + CONNECT_WINDOW
        self.listener.settimeout(0.05)
        accepted = 0
        while accepted < self.ranks:
            try:
                sock, peer = self.listener.accept()
            except socket.timeout:
                with self.cv:
                    if self.failed is not None:
                        return
                    if time.monotonic() >= deadline:
                        self._fail(f"coordinator: {accepted} of {self.ranks} hosts connected within {CONNECT_WINDOW:.0f}s")
                        return
                continue
            except OSError as e:
                with self.cv:
                    self._fail(f"coordinator: accept: {e}")
                return
            accepted += 1
            threading.Thread(target=self._serve, args=(sock, "%s:%d" % peer[:2]), daemon=True).start()
        self.listener.close()

    def _serve(self, sock, peer: str) -> None:
        conn = Conn(sock, peer)
        who = [peer]
        try:
            self._serve_loop(conn, who)
        except (Closed, OSError, ValueError, KeyError) as e:
            with self.cv:
                if not self.finished:
                    self._fail(f"{who[0]}: {e}")
        with self.cv:
            self.closed += 1
            self.cv.notify_all()

    def _serve_loop(self, conn: Conn, who: list) -> None:
        name, h = _variant(conn.recv())
        if name != "Hello":
            raise Closed(f"expected Hello, got {name}")
        rank, ranks, host = h["rank"], h["ranks"], h["host"]
        who[0] = f"rank {rank} ({host})"
        digest = identity_hash(h["identity"])
        with self.cv:
            refusal = None
            if ranks != self.ranks:
                refusal = f"{who[0]}: --ranks {ranks}, the coordinator was started for {self.ranks}"
            elif not 0 <= rank < self.ranks or rank in self.conns:
                refusal = f"{who[0]}: rank {rank} is out of range or already connected"
            elif self.identity is not None and self.identity[1] != digest:
                r0, _, id0 = self.identity
                refusal = f"{who[0]}: not the same run: {'; '.join(identity_differences(h['identity'], id0, r0))}"
            if refusal is not None:
                # the refused host is not registered, so it hears the reason on its own socket
                try:
                    conn.send({"Stop": {"reason": refusal}})
                except OSError:
                    pass
                self._fail(refusal)
                return
            if self.identity is None:
                self.identity = (rank, digest, h["identity"])
            self.layers[rank] = h.get("layers")
            self.conns[rank] = conn
            self.hosts[rank] = host
            conn.send("Welcome")
        while True:
            name, body = _variant(conn.recv())
            with self.cv:
                if self.failed is not None:
                    return
                if name == "Ready":
                    self.ready_.add(rank)
                    if len(self.ready_) == self.ranks:
                        self._broadcast({"Start": {"t0": time.time(), "hosts": [self.hosts[r] for r in sorted(self.hosts)]}})
                elif name == "Done":
                    self.done_[rank] = body
                    self.cv.notify_all()
                elif name == "Stop":
                    self._fail(f"{who[0]}: {body['reason']}")
                    return
                elif name not in ("Heartbeat", "Progress"):   # a Rust datagen's progress: not kept here
                    raise Closed(f"unexpected message: {name}")
                if self.finished:
                    return

    def gathered(self) -> dict:
        """Every host's `Done` document, by rank; a host's failure ends the wait."""
        with self.cv:
            while True:
                if self.failed is not None:
                    raise BuildError(self.failed)
                if len(self.done_) == self.ranks:
                    return dict(sorted(self.done_.items()))
                self.cv.wait(0.05)

    def finish(self, ok: bool, error: str | None = None) -> None:
        """Rank 0's verdict, the last message; waits briefly for the other hosts to close, so
        that rank 0 exiting cannot reset a connection before its `Result` was read."""
        with self.cv:
            self.finished = True
            self._broadcast({"Result": {"ok": ok, "fingerprint": 0, "error": error}})
            deadline = time.monotonic() + 5
            while self.closed < self.ranks - 1 and time.monotonic() < deadline:
                self.cv.wait(0.02)

    def stop(self, reason: str) -> None:
        """Abort every host."""
        with self.cv:
            self._fail(reason)
