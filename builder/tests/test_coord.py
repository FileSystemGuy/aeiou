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
PARAMS = ["samples=700", "per_shard=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"]


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
    rust = subprocess.Popen([str(RUNNER), "datagen", str(ast), "--root", str(tmp_path / "r"), "--ranks", "2", "--rank", "0",
                             "--coordinator", addr, "--dedupe", "2"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env())
    py = subprocess.run([sys.executable, "-m", "aeiou.datagen", str(ast), "--root", str(tmp_path / "p"), "--ranks", "2", "--rank", "1",
                         "--coordinator", addr], capture_output=True, text=True, cwd=str(BUILDER), env=_env(), timeout=300)
    ro, re_ = rust.communicate(timeout=300)
    assert rust.returncode != 0 and py.returncode == 1, ro + re_ + py.stdout + py.stderr
    assert "dedupe 1 differs from rank 0's 2" in py.stderr, py.stderr
    assert "dedupe 1 differs from rank 0's 2" in re_, re_
