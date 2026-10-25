"""`aeiou-datagen` on several hosts (`aeiou/coord.py`, the runner's coordinator protocol):
the corpus is the one host's byte for byte, rank 0 writes the manifests once every rank has
reported, a host whose identity differs stops every host before anything is written, and
the frames are the runner's (a Python rank talks to `aeiou datagen`'s coordinator)."""
import json
import os
import re
import pathlib
import socket
import subprocess
import struct
import sys
import threading

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
sys.path.insert(0, str(BUILDER))

from aeiou import coord  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
needs_runner = pytest.mark.skipif(not RUNNER.exists(), reason="runner binary not built")
# six shards of 128 samples, the last short (700 samples)
PARAMS = ["samples=700", "samples_per_file=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000", "prefetch=1", "step_time=105000000", "decode=0"]


def ast_path(name):
    return next((ROOT / "workloads").rglob(f"{name}.ast.json"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cmd(root, *extra):
    return [sys.executable, "-m", "aeiou.datagen", str(ast_path("tfrecord_stream")), "--root", str(root),
            *sum([["--param", p] for p in PARAMS], []), *map(str, extra)]


def _env():
    return {**{k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}, "PYTHONPATH": str(BUILDER)}


def _ranks(root, ranks, extra=lambda r: []):
    """`ranks` processes on one coordinator, rank 0 first; their (returncode, stdout, stderr)."""
    addr = f"127.0.0.1:{free_port()}"
    procs = [subprocess.Popen(_cmd(root, "--writers", 2, "--ranks", ranks, "--rank", r, "--coordinator", addr, *extra(r)),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(BUILDER), env=_env())
             for r in range(ranks)]
    out = []
    for p in procs:
        o, e = p.communicate(timeout=300)
        out.append((p.returncode, o, e))
    return out


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and p.name != ".aeiou-dataset.json"}


def _manifest(root):
    return json.loads((root / "train" / ".aeiou-dataset.json").read_text())


def _id(stdout):
    return [w for line in stdout.splitlines() if line.startswith("dataset shards:") and " id " in line for w in [line.split(" id ")[1].split()[0]]]


def test_several_ranks_write_the_one_host_corpus(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("crc32c")
    one = tmp_path / "one"
    r = subprocess.run(_cmd(one, "--writers", 2), capture_output=True, text=True, cwd=str(BUILDER), env=_env())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dataset shards: files [0, 6) of 6 on this rank" in r.stdout
    m1 = _manifest(one)
    assert m1["provenance"]["ranks"] == 1 and [h["rank"] for h in m1["provenance"]["hosts"]] == [0]
    # four ranks over six files: [0, 2), [2, 4), [4, 6), and an empty slice
    many = tmp_path / "many"
    out = _ranks(many, 4)
    for rc, o, e in out:
        assert rc == 0, o + e
        assert "start gate: 4 host(s) ready" in o, o
    slices = ["[0, 2)", "[2, 4)", "[4, 6)", "[6, 6)"]
    for (_, o, _), s in zip(out, slices):
        assert f"dataset shards: files {s} of 6 on this rank" in o, o
    assert "all 4 hosts 6 file(s)" in out[0][1] and "manifest written" in out[0][1]
    for _, o, _ in out[1:]:
        assert "(manifest by rank 0)" in o
    assert _tree(many) == _tree(one)
    m = _manifest(many)
    p = m["provenance"]
    assert p["ranks"] == 4 and [h["rank"] for h in p["hosts"]] == [0, 1, 2, 3]
    assert [h["files_written"] for h in p["hosts"]] == [2, 2, 2, 0]
    assert p["files_written"] == 6 and p["bytes_written"] == m1["provenance"]["bytes_written"]
    assert {k: m[k] for k in ("manifest_version", "dataset", "payload", "format")} == {k: m1[k] for k in ("manifest_version", "dataset", "payload", "format")}
    # every rank prints the dataset id, rank 0's manifest's, the one host's
    ids = {i for _, o, _ in out for i in _id(o)}
    assert ids == set(_id(r.stdout)) and len(ids) == 1


def test_a_different_host_stops_every_host_before_anything_is_written(tmp_path):
    pytest.importorskip("numpy")
    root = tmp_path / "root"
    out = _ranks(root, 2, lambda r: ["--dedupe", 2] if r == 1 else [])
    for rc, o, e in out:
        assert rc == 1, o + e
        # the first Hello is the reference, and either rank's may arrive first
        assert re.search(r"dedupe (2 differs from rank 0's 1|1 differs from rank 1's 2)", e), e
    assert not (root / "train" / ".aeiou-dataset.json").exists()
    assert not root.exists() or not any(p.is_file() for p in root.rglob("*"))


def test_the_coordinator_in_process():
    """The server and its clients on threads: the gate, the gathering, the verdict, and the
    refusals (a different identity, a wrong --ranks, a rank taken twice)."""
    def serve(ranks, hellos):
        addr = f"127.0.0.1:{free_port()}"
        server = coord.Server(addr, ranks)
        results = [None] * len(hellos)

        def host(i, rank, ranks_, identity):
            try:
                c = coord.Client(addr, rank, ranks_, f"h{rank}", identity, {})
                t0, names = c.ready()
                c.done({"host": f"h{rank}", "n": rank})
                results[i] = ("ok", names, c.result())
            except BuildError as e:
                results[i] = ("error", str(e))
        threads = [threading.Thread(target=host, args=(i, *h)) for i, h in enumerate(hellos)]
        threads[0].start()
        # the first Hello is the reference the others are compared with
        while server.identity is None:
            threading.Event().wait(0.01)
        for t in threads[1:]:
            t.start()
        return server, threads, results

    server, threads, results = serve(2, [(0, 2, {"a": 1}), (1, 2, {"a": 1})])
    gathered = server.gathered()
    assert gathered == {0: {"host": "h0", "n": 0}, 1: {"host": "h1", "n": 1}}
    server.finish(True)
    for t in threads:
        t.join(10)
    assert results == [("ok", ["h0", "h1"], (True, 0, None))] * 2

    def refused(hellos, reason):
        server, threads, results = serve(2, hellos)
        for t in threads:
            t.join(15)
        with pytest.raises(BuildError, match=reason):
            server.gathered()
        assert all(r[0] == "error" and reason in r[1] for r in results), results
    refused([(0, 2, {"params": {"x": 1}}), (1, 2, {"params": {"x": 2}})], "params.x 2 differs from rank 0's 1")
    refused([(0, 2, {}), (1, 3, {})], "--ranks 3, the coordinator was started for 2")
    refused([(0, 2, {}), (0, 2, {})], "rank 0 is out of range or already connected")


@needs_runner
def test_the_frames_are_the_runners(tmp_path):
    """A Python rank 1 says Hello to `aeiou datagen`'s rank 0: the runner reads the frame,
    compares the identity (the Rust writer's parameters are not the Python's document), and
    names the difference to both hosts."""
    addr = f"127.0.0.1:{free_port()}"
    ast = ast_path("file_per_sample")
    # its job values, required since 2026-10-22 (DESIGN_REVIEW.md §3.78): the defaults they held
    JOB = [x for p in ("batch=32", "workers=8", "prefetch=2", "steps=500", "step_time=105000000", "sample_median=112640") for x in ("--param", p)]
    rust = subprocess.Popen([str(RUNNER), "datagen", str(ast), "--root", str(tmp_path / "r"), "--ranks", "2", "--rank", "0",
                             "--coordinator", addr, "--dedupe", "2", "--param", "samples=100", *JOB],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env())
    py = subprocess.run([sys.executable, "-m", "aeiou.datagen", str(ast), "--root", str(tmp_path / "p"), "--ranks", "2", "--rank", "1",
                         "--coordinator", addr, "--param", "samples=100", *JOB], capture_output=True, text=True, cwd=str(BUILDER), env=_env(), timeout=300)
    ro, re_ = rust.communicate(timeout=300)
    assert rust.returncode != 0 and py.returncode == 1, ro + re_ + py.stdout + py.stderr
    assert "dedupe 1 differs from rank 0's 2" in py.stderr, py.stderr
    assert "dedupe 1 differs from rank 0's 2" in re_, re_


# ---- the wire, one end at a time ----

def _frame(msg) -> bytes:
    body = json.dumps(msg).encode()
    return len(body).to_bytes(4, "big") + body


def test_a_silent_peer_is_heartbeaten_then_given_up(monkeypatch):
    """A read that times out sends `Heartbeat` and keeps waiting; after `DEAD_AFTER` without a
    message the peer is gone. A frame arriving in pieces is reassembled."""
    monkeypatch.setattr(coord, "HEARTBEAT", 0.05)
    monkeypatch.setattr(coord, "DEAD_AFTER", 0.3)
    with socket.create_server(("127.0.0.1", 0)) as srv:   # TCP: the connection sets TCP_NODELAY
        b = socket.create_connection(srv.getsockname())
        a, _ = srv.accept()
    conn = coord.Conn(a, "peer p")
    f = _frame({"Done": {"n": 1}})

    def trickle():
        threading.Event().wait(0.12)
        b.sendall(f[:3])
        threading.Event().wait(0.12)
        b.sendall(f[3:])
    t = threading.Thread(target=trickle)
    t.start()
    assert conn.recv() == {"Done": {"n": 1}}
    t.join()
    b.settimeout(1)
    beats = b.recv(1 << 16)
    assert beats.startswith(_frame("Heartbeat")), beats   # the reader beat while it waited
    with pytest.raises(coord.Closed, match=r"^peer p: no message for 0s$"):
        conn.recv()
    b.sendall((coord.MAX_FRAME + 1).to_bytes(4, "big"))
    with pytest.raises(coord.Closed, match=f"frame of {coord.MAX_FRAME + 1} bytes"):
        conn.recv()
    b.shutdown(socket.SHUT_WR)   # an end of stream (a close with heartbeats unread would reset)
    with pytest.raises(coord.Closed, match="connection closed"):
        conn.recv()
    conn.close()
    conn.close()   # twice is harmless
    b.close()


def test_what_is_not_a_message_or_an_address():
    assert coord._variant("Ready") == ("Ready", None)
    assert coord._variant({"Done": {"n": 1}}) == ("Done", {"n": 1})
    for bad in ({}, {"a": 1, "b": 2}, 3, None):
        with pytest.raises(coord.Closed, match="not a message"):
            coord._variant(bad)
    assert coord._split("[::1]:7") == ("::1", 7) and coord._split(":7") == ("0.0.0.0", 7)
    for bad in ("host", "host:", "host:x"):
        with pytest.raises(BuildError, match=f"coordinator address `{bad}`: expected HOST:PORT"):
            coord._split(bad)


def test_a_coordinator_that_never_answers_is_given_up_on(monkeypatch):
    monkeypatch.setattr(coord, "CONNECT_WINDOW", 0.0)
    addr = f"127.0.0.1:{free_port()}"
    with pytest.raises(BuildError, match=rf"coordinator {addr}: .*\(gave up after 0s\)"):
        coord.Client(addr, 0, 1, "h0", {}, {})


def test_a_port_taken_is_refused():
    with socket.create_server(("127.0.0.1", 0)) as s:
        addr = "127.0.0.1:%d" % s.getsockname()[1]
        with pytest.raises(BuildError, match=f"coordinator: binding {addr}: "):
            coord.Server(addr, 1)


def test_hosts_that_never_connect_fail_the_server(monkeypatch):
    monkeypatch.setattr(coord, "CONNECT_WINDOW", 0.2)
    server = coord.Server(f"127.0.0.1:{free_port()}", 2)
    with pytest.raises(BuildError, match=r"^coordinator: 0 of 2 hosts connected within 0s$"):
        server.gathered()


def _raw_host(server, rank=0, ranks=1, identity=None):
    """A host speaking the frames by hand: its socket, `Hello` sent."""
    host, port = coord._split(server.addr)
    s = socket.create_connection((host, port))
    s.settimeout(10)
    s.sendall(_frame({"Hello": {"rank": rank, "ranks": ranks, "host": f"h{rank}", "identity": identity or {},
                                "layers": {}, "participants": [], "facts": {}}}))
    return s


def _read_msg(s):
    n = int.from_bytes(s.recv(4, socket.MSG_WAITALL), "big")
    return json.loads(s.recv(n, socket.MSG_WAITALL))


@pytest.mark.parametrize("first, then, reason", [
    ("Ready", None, "expected Hello, got Ready"),
    (None, "Arrive", "unexpected message: Arrive"),
    (None, {"Stop": {"reason": "disk full"}}, "rank 0 (h0): disk full"),
])
def test_the_server_stops_every_host_on_a_wrong_or_failed_one(first, then, reason):
    server = coord.Server(f"127.0.0.1:{free_port()}", 1)
    if first is not None:
        host, port = coord._split(server.addr)
        s = socket.create_connection((host, port))
        s.sendall(_frame(first))
    else:
        s = _raw_host(server)
        assert _read_msg(s) == "Welcome"
        s.sendall(_frame("Heartbeat") + _frame({"Progress": {"files": 1}}) + _frame(then))
    with pytest.raises(BuildError, match=re.escape(reason)):
        server.gathered()
    if first is None and isinstance(then, str):
        assert _read_msg(s) == {"Stop": {"reason": f"rank 0 (h0): {reason}"}}
    s.close()


def test_after_the_verdict_a_host_is_no_longer_watched():
    """Once rank 0 sent `Result`, a host closing or saying more is not a failure (before it, a
    host gone is: it would never hear the verdict)."""
    server = coord.Server(f"127.0.0.1:{free_port()}", 2)
    s0, s1 = _raw_host(server, 0, 2), _raw_host(server, 1, 2)
    for s in (s0, s1):
        assert _read_msg(s) == "Welcome"
        s.sendall(_frame("Ready"))
    for s in (s0, s1):
        assert _read_msg(s)["Start"]["hosts"] == ["h0", "h1"]
        s.sendall(_frame({"Done": {}}))
    assert server.gathered() == {0: {}, 1: {}}

    def verdict():
        # rank 0 waits a moment for the hosts to close after its Result
        server.finish(True)
    t = threading.Thread(target=verdict)
    t.start()
    for s in (s0, s1):
        assert _read_msg(s) == {"Result": {"ok": True, "fingerprint": 0, "error": None}}
    s0.close()
    s1.sendall(_frame("Heartbeat"))
    s1.close()
    t.join(10)
    server.stop("late")   # a failure after the verdict changes nothing
    assert server.failed is None


def test_a_client_hears_a_stop_and_names_unexpected_messages():
    """The client's side: `Release` and `Heartbeat` are nothing to it, `Stop` is the reason it
    fails with, a message it never gets is refused; `stop` tells the coordinator."""
    srv = socket.create_server(("127.0.0.1", 0))
    addr = "127.0.0.1:%d" % srv.getsockname()[1]
    got = {}

    def coordinator():
        s, _ = srv.accept()
        s.settimeout(10)
        got["hello"] = _read_msg(s)
        s.sendall(_frame({"Release": {"scope": "x", "generation": 1}}) + _frame("Heartbeat") + _frame({"Welcome": {}}))
        got["stop"] = _read_msg(s)
        s.sendall(_frame({"Grant": {"dataset": "d", "ticket": 0, "run": None}}))
        s.close()
    t = threading.Thread(target=coordinator)
    t.start()
    c = coord.Client(addr, 0, 1, "h0", {"a": 1}, {})
    c.stop("my disk is full")
    with pytest.raises(BuildError, match="unexpected message from the coordinator: Grant"):
        c.result()
    t.join(10)
    c.close()
    c.stop("after close")   # a closed socket: nothing to tell, nothing raised
    srv.close()
    assert got["hello"]["Hello"]["identity"] == {"a": 1} and set(got["hello"]["Hello"]["facts"]) == {
        "host", "kernel", "cpus", "cpus_online", "memory", "cgroup_memory_max"}
    assert got["stop"] == {"Stop": {"reason": "my disk is full"}}


def test_a_send_after_the_coordinator_reset_fails_with_its_reason():
    """Rank 0 stops every host and exits, which resets a connection whose last message it did
    not read; the next `Ready` or `Done` of that host fails on the socket, and the host reports
    the coordinator's reason, never the reset (CI run 40153591881, one run in 200 here)."""
    srv = socket.create_server(("127.0.0.1", 0))
    addr = "127.0.0.1:%d" % srv.getsockname()[1]

    def coordinator(stop):
        s, _ = srv.accept()
        s.settimeout(10)
        _read_msg(s)
        s.sendall(_frame({"Welcome": {}}) + (_frame({"Stop": {"reason": "rank 1 (h1): not the same run"}}) if stop else b""))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))   # a reset, as an exit with unread data
        s.close()
    for stop, reason in ((True, "rank 1 \\(h1\\): not the same run"), (False, "Connection reset by peer")):
        t = threading.Thread(target=coordinator, args=(stop,))
        t.start()
        c = coord.Client(addr, 0, 2, "h0", {"a": 1}, {})
        t.join(10)
        for call in (c.ready, lambda: c.done({})):
            with pytest.raises(BuildError, match=reason):
                call()
        c.close()
    srv.close()


def test_host_facts_without_proc_or_cgroups(monkeypatch):
    """A host without /proc/meminfo or a cgroup v2 line says none; the tightest `memory.max`
    on the way to the cgroup root wins, and `max` is no limit."""
    import builtins
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: (_ for _ in ()).throw(OSError(2, "nope"))
                        if str(p).startswith("/proc/") else real_open(p, *a, **k))
    f = coord.host_facts()
    assert f["memory"] is None and f["cgroup_memory_max"] is None and f["cpus"] >= 1
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: iter(["0::/a/b\n"]) if p == "/proc/self/cgroup"
                        else iter(["MemTotal: x kB\n"]) if p == "/proc/meminfo" else real_open(p, *a, **k))
    limits = {"/sys/fs/cgroup/a/b/memory.max": "max\n", "/sys/fs/cgroup/a/memory.max": "4096\n",
              "/sys/fs/cgroup/memory.max": "8192\n"}

    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: limits[str(self)])   # the walk reads these three alone
    f = coord.host_facts()
    assert f["memory"] is None and f["cgroup_memory_max"] == 4096
