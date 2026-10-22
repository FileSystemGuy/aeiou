"""The workloads' kit scripts (`workloads/**/kit/`): what they read off a trace, what they send a
server, and what they fit."""
from __future__ import annotations

import http.server
import importlib
import importlib.util
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import threading

import pytest

from aeiou.params import values_of

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
HOPS = ROOT / "workloads" / "vector-search" / "diskann-search" / "kit" / "hops.py"


def _iocb(off: int) -> str:
    return (f"{{aio_data=0, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</i/ann_disk.index>, "
            f"aio_buf=0x7f, aio_nbytes=4096, aio_offset={off}}}")


def _submit(t: float, sectors: list, whole: bool) -> str:
    """One `io_submit` line as `strace` prints it, cut at 32 requests unless `-v` (`whole`)."""
    shown = sectors if whole or len(sectors) <= 32 else sectors[:32]
    cut = "" if len(shown) == len(sectors) else ", ..."
    body = ", ".join(_iocb(s * 4096) for s in shown)
    return f"100 {t:.6f} io_submit(0x7f00, {len(sectors)}, [{body}{cut}]) = {len(sectors)} <0.000010>\n"


def _trace(whole: bool) -> str:
    """A cached search: the breadth-first load (the medoids' level, a level of 40, then blocks of 8),
    then three queries of beam-4 rounds, each after search.py's mark."""
    t, lines = 1.0, []

    def sub(sectors):
        nonlocal t
        t += 0.001
        lines.append(_submit(t, sectors, whole))
    sub([1, 2, 3])
    sub(list(range(10, 50)))
    sub(list(range(50, 58)))
    sub(list(range(58, 66)))
    for q, rounds in enumerate((3, 4, 2)):
        t += 0.001
        lines.append(f'100 {t:.6f} stat("/aeiou-query-mark", 0x7ffd) = -1 ENOENT (No such file or directory) <0.000005>\n')
        for r in range(rounds):
            sub([100 + 7 * q + r + k for k in range(4)])
    return "".join(lines)


@pytest.fixture
def index(tmp_path):
    d = tmp_path / "index"
    d.mkdir()
    hdr = bytearray(4096)
    struct.pack_into("<QQQQQ", hdr, 8, 1000, 128, 7, 600, 5)   # npts, ndims, medoid, max_node_len, per_sector
    (d / "ann_disk.index").write_bytes(bytes(hdr) + bytes(4096 * 199))
    (d / "ann_disk.index_medoids.bin").write_bytes(struct.pack("<5I", 3, 1, 5, 10, 15))
    return d


def _hops(trace: pathlib.Path, index: pathlib.Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HOPS), str(trace), str(index), *args], capture_output=True, text=True)


def test_hops_reads_a_trace_with_and_without_v_alike(tmp_path, index):
    """The node cache's level of 40 is the load whether `strace` cut its array at 32 or printed
    it whole (`-v`, the documented capture): neither a query's rounds nor the load's count move."""
    pytest.importorskip("numpy")
    out = {}
    for whole in (False, True):
        trace = tmp_path / f"trace.{whole}.txt"
        trace.write_text(_trace(whole))
        assert ("..." in trace.read_text()) is not whole
        report = _hops(trace, index)
        params = _hops(trace, index, "--params", "test")
        assert report.returncode == 0 and params.returncode == 0, report.stderr + params.stderr
        out[whole] = (json.loads(report.stdout), json.loads(params.stdout))
    assert out[False] == out[True]
    report, params = out[True]
    assert (report["bulk_submits"], report["bulk_requests"], report["bulk_largest"]) == (1, 40, 40)
    assert report["before_first_query"] == {"submits": 3, "requests": 19}
    assert report["rounds_per_query"] == {"2": 1, "3": 1, "4": 1}
    assert values_of(params)["node_cache"] == 59
    assert values_of(params)["hops"]["empirical"] == {"values": [2, 3, 4], "weights": [1, 1, 1]}


def test_hops_refuses_a_query_round_over_the_bulk_size(tmp_path, index):
    """A round of more than 32 requests after the first mark is no load: refused, never counted."""
    pytest.importorskip("numpy")
    trace = tmp_path / "trace.txt"
    trace.write_text(_trace(True) + _submit(9.0, list(range(120, 160)), True))
    got = _hops(trace, index)
    assert got.returncode == 1 and "a query round of 40 requests" in got.stderr, got.stderr


def test_hops_without_marks_starts_a_query_at_a_medoid_read(tmp_path, index):
    """Without search.py's marks a query starts at a one-request submit of a medoid's sector (an index
    without a node cache); `--tail N` keeps each thread's last N queries; the parameters then count
    hops past the medoid's round and the first hop, whose sectors are the `near` hot set."""
    pytest.importorskip("numpy")
    t, lines = 1.0, []
    for tid, rounds in ((100, [[2], range(10, 14), range(14, 18)]), (100, [[3], range(20, 24)]), (200, [[4], range(50, 54)]),
                        (100, [[2], [10, 11, 30, 31], [40, 41]])):
        for sectors in rounds:
            t += 0.001
            lines.append(_submit(t, list(sectors), True).replace("100 ", "%d " % tid, 1))
    lines.insert(1, _submit(0.5, list(range(60, 100)), False))                         # a bulk submit, only counted
    trace = tmp_path / "trace.txt"
    trace.write_text("".join(lines))
    r = _hops(trace, index)
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    assert (rep["threads"], rep["queries"], rep["before_first_query"], rep["bulk_submits"], rep["bulk_requests"]) == (2, 4, None, 1, 40)
    assert rep["rounds_per_query"] == {"2": 2, "3": 2} and rep["first_batch"] == {"1": 4}
    assert rep["second_round"] == {"reads": 16, "distinct_sectors": 14}
    assert (rep["sector_reads"], rep["share_of_reads_on_medoid_sectors"]) == (26, round(4 / 26, 4))
    p = values_of(json.loads(_hops(trace, index, "--params", "doc").stdout))
    assert p["hops"]["empirical"] == {"values": [0, 1], "weights": [2, 2]} and p["node_cache"] == 0
    assert (p["threads"], p["queries"], p["beam"], p["nodes"], p["per_sector"]) == (2, 2, 4, 200, 5)
    assert p["entry"]["hotset"]["fraction"] == 2.5 / 200 and p["near"]["hotset"]["fraction"] == 13.5 / 200
    rep = json.loads(_hops(trace, index, "--tail", "1").stdout)
    assert (rep["queries"], rep["rounds_per_query"]) == (2, {"2": 1, "3": 1})


DISKANN_BUILD = ROOT / "workloads" / "vector-search" / "diskann-build" / "kit"


def test_diskann_build_params_from_a_trace_and_the_files_the_build_left(tmp_path):
    """`diskann-build/kit/params.py`: the base's shape from its header, the shard count and mean
    shard graph from the merge's reads, the merged graph from its writes (and its 8-byte
    header), and the sizes of the files the build left."""
    base = tmp_path / "base.fbin"
    base.write_bytes(struct.pack("<II", 1000, 16) + bytes(64))
    index = tmp_path / "index"
    index.mkdir()
    (index / "ann_pq_compressed.bin").write_bytes(bytes(8 + 1000 * 32))
    (index / "ann_disk.index").write_bytes(bytes(4096 * 30 + 100))
    (index / "ann_sample_ids.bin").write_bytes(bytes(8 + 4 * 77))
    trace = tmp_path / "trace.txt"
    trace.write_text("".join(
        [f'7 1.0 read(5</i/ann_subshard-{k}_mem.index>, "\\x00"..., {n}) = {n} <0.000010>\n' for k, n in ((0, 1000), (1, 600), (0, 500), (2, 100))]
        + ['7 1.1 writev(6</i/ann_mem.index>, [{iov_base="\\x00", iov_len=4096}], 1) = 4096 <0.000010>\n',
           '7 1.2 writev(6</i/ann_mem.index>, [{iov_base="\\x00", iov_len=100}], 1) = 100 <0.000010>\n',
           '7 1.3 read(5</i/ann_mem.index>, "", 8) = 8 <0.000010>\n']))
    r = _run(DISKANN_BUILD / "params.py", trace, base, index, "a doc")
    assert r.returncode == 0, r.stderr
    f = json.loads(r.stdout)
    assert (f["abstract"], f["doc"]) == ("diskann_build", "a doc")
    assert values_of(f) == {"dim": 16, "n": 1000, "shards": 3, "pq_code": 32, "shard_index_bytes": 2200 // 3, "index_bytes": 4204,
                           "sectors": 30, "sample_rows": 77}
    trace.write_text("")
    r = _run(DISKANN_BUILD / "params.py", trace, base, index, "a doc")
    assert r.returncode == 1 and r.stderr.strip() == ("params.py: the trace reads no shard graph (subshard-N_mem.index): "
                                                      "diskann_build models a build that partitions its base (lower --build-gb)"), r.stderr


def test_npy_corpus_writes_one_archive_per_sample_as_dlio_does(tmp_path):
    """`npy-per-sample/kit/mkcorpus.py`: `np.savez(x=<uint8 volume>, y)` per sample at
    train/{id div 10000:05}/sample_{id:09}.npz, the volume near the size asked for."""
    np = pytest.importorskip("numpy")
    r = _run(ROOT / "workloads/input-pipeline/map/npy-per-sample/kit/mkcorpus.py", tmp_path, 3, 2 << 20, 0)
    assert r.returncode == 0, r.stderr
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["train/00000/sample_%09d.npz" % i for i in range(3)]
    for f in files:
        z = np.load(tmp_path / f)
        assert sorted(z.files) == ["x", "y"] and z["x"].shape == (256, 256, 32) and z["x"].dtype == np.uint8 and list(z["y"]) == [0]


# --- the KV-cache kit (workloads/kvcache/lmcache-local-disk/kit) ---------------------------------

KV = ROOT / "workloads" / "kvcache" / "lmcache-local-disk" / "kit"


def _kit(monkeypatch, name, where=KV):
    monkeypatch.syspath_prepend(str(where))
    return importlib.import_module(name)


def _req(t, h, i, o, think=0.5, model="m"):
    return {"t": t, "hash_ids": h, "in": i, "out": o, "think_time": think, "model": model}


# Two AgentX-shaped sessions. Session 0: a main agent that extends its chain (all but the partial
# last block kept), a sub-agent group whose first prompt finds the session's [1, 2] stored and
# extends its own chain, and a main-agent request that cuts its chain to 4 blocks (a rewrite).
# Session 1: no sub-agents; a side call that keeps nothing (a chain of its own), a prompt over the
# context (skipped), and a request of another model (a chain of its own, opening with that
# model's main-agent prefix).
CORPUS = [
    [_req(0, [1, 2, 3, 4, 5], 300, 10), _req(1, [1, 2, 3, 4, 6, 7, 8], 420, 20),
     {"type": "subagent", "requests": [_req(2, [1, 2, 9, 10, 11], 300, 5), _req(3, [1, 2, 9, 10, 12, 13], 380, 6)]},
     _req(4, [1, 2, 3, 4, 20, 21], 350, 8)],
    [_req(0, [30, 31, 32], 150, 7, 1.0), _req(1, [40], 50, 3, None), _req(2, list(range(50, 130)), 5000, 9),
     _req(3, [30, 31, 32, 33], 200, 4, 2.0, "m2")],
]


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text("".join(json.dumps({"requests": s}) + "\n" for s in CORPUS))
    return p


def test_agentx_walk_builds_chains_by_continuity_and_counts_a_never_evicting_store(monkeypatch, corpus):
    """`agentx.walk` at 128-token chunks (two hash ids each): the main agent's prefix is the 4 blocks
    its cut keeps (`main_blocks`); a request extends the chain whose last prompt it keeps all but
    the last block of, rewrites it when it keeps less, and starts one when it keeps no more than
    the sub-agents' shared prefix; a chain's kind is the prefix its first prompt opens with."""
    ax = _kit(monkeypatch, "agentx")
    assert ax.main_blocks(corpus) == 4
    flat = []
    ax.flatten(CORPUS[0], flat)
    assert [(g, r["t"]) for g, r in flat] == [(-1, 0), (-1, 1), (0, 2), (0, 3), (-1, 4)]
    assert ax.shared_prefix(flat, 2) == [1, 2] and ax.shared_prefix(flat, 2, "m2") == []
    assert ax.prefixes(flat, 2, 4) == {(0, "m"): [1, 2, 3, 4], (1, "m"): [1, 2]}
    rows, shared = ax.walk(corpus, 10 ** 9, 128, 4096, None, 4)
    got = [(r["ti"], r["fi"], r["chain"], r["first"], r["kind"], r["kept"], r["stored"], r["hit"]) for r in rows]
    assert got == [(0, 0, 0, True, 0, 0, 2, 0), (0, 1, 0, False, 0, 320, 1, 2),       # extends: all 5 blocks kept
                   (0, 2, 1, True, 1, 0, 1, 1), (0, 3, 1, False, 1, 320, 0, 2),
                   (0, 4, 0, False, 0, 256, 0, 2),                                     # a rewrite: 4 of 7 blocks
                   (1, 0, 2, True, 2, 0, 1, 0), (1, 1, 3, True, 2, 0, 0, 0),           # a side call: a chain of its own
                   (1, 3, 4, True, 0, 0, 1, 0)]       # fi 2 is over the context; m2's chunks are its own, never m's
    assert shared == [128]
    assert [r["think"] for r in rows] == [0.5] * 5 + [1.0, None, 2.0]
    assert [r["fi"] for r in ax.walk(corpus, 1, 128, 4096)[0]] == [0, 1, 2, 3, 4]       # --traces 1: session 0 only
    assert [(r["ti"], r["fi"]) for r in ax.walk(corpus, 10 ** 9, 128, 4096, {(0, 1), (1, 3)}, 4)[0]] == [(0, 1), (1, 3)]   # a replay's


def test_agentx_keys_chunks_by_model(monkeypatch, tmp_path):
    """LMCache keys a chunk by its model (2026-10-15): a request of model b never hits the chunks
    its session stored for model a, in the corpus's accounting (`walk`'s hits and stores) or in
    the sub-agents' shared prefix (`shared_prefix` with a model, `sub_tokens`). A replay serves one
    model, so its walk and `shared_prefix` without a model (`holds`, `replay_agentx.py`) keep one
    store per session."""
    ax = _kit(monkeypatch, "agentx")
    session = [_req(0, [1, 2, 3, 4, 5], 300, 10, model="a"),
               {"type": "subagent", "requests": [_req(1, [1, 2, 7, 8, 9], 300, 5, model="b")]},
               _req(2, [1, 2, 3, 4, 6], 300, 8, model="b")]
    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps({"requests": session}) + "\n")
    rows, shared = ax.walk(p, 10 ** 9, 128, 4096)
    assert [(r["stored"], r["hit"]) for r in rows] == [(2, 0), (2, 0), (1, 1)]       # b hits only its sub-agent's [1, 2]
    assert shared == []                                                              # the sub-agent found nothing of b's
    flat = []
    ax.flatten(session, flat)
    assert ax.shared_prefix(flat, 2, "b") == [] and ax.shared_prefix(flat, 2, "b", pooled=True) == [1, 2] == ax.shared_prefix(flat, 2)
    rows, shared = ax.walk(p, 10 ** 9, 128, 4096, {(0, 0), (0, 1), (0, 2)})          # a replay's: one engine, one model
    assert [(r["stored"], r["hit"]) for r in rows] == [(2, 0), (1, 1), (0, 2)] and shared == [128]
    out = tmp_path / "ref.json"
    r = _agentx("reference", p, "--chunk-tokens", 128, "--context", 4096, "--bins", 1, "-o", out)
    assert r.returncode == 0, r.stderr
    ref = json.loads(out.read_text())
    assert (ref["chunks_stored"]["total"], ref["chunks_hit"]["total"], ref["subagent_shared_prefix_tokens"]) == (5, 1, {"median": 0, "n": 0})


def _agentx(*args):
    return subprocess.run([sys.executable, str(KV / "agentx.py"), *map(str, args)], capture_output=True, text=True)


def test_agentx_fit_of_the_corpus(tmp_path, corpus):
    """`agentx.py fit` without a replay: the shares are of the walk's rows (the module doc)."""
    out = tmp_path / "fit.json"
    r = _agentx("fit", corpus, "--chunk-tokens", 128, "--context", 4000, "--bins", 2, "--turn-bins", 3,
                "--set", "chunk_bytes=1024", "-o", out)
    assert r.returncode == 0, r.stderr
    assert "requests 8, chains 5 (one request: 3, longest 3)" in r.stdout
    f = json.loads(out.read_text())
    assert (f["abstract"], "provenance" in f) == ("lmcache_local_disk", False)
    assert f["doc"].startswith("Fitted by agentx.py to traces.jsonl (2 sessions): 8 requests in 5 chains; 33.3% of continuing "
                               "requests cut their chain (trim) and the chain grows by 100% of the corpus's prompt growth")
    p = values_of(f)
    assert (p["requests"], p["context"], p["chunk_tokens"], p["sys_prompts"], p["sub_prompts"], p["chunk_bytes"]) == (8, 4096, 128, 2, 2, 1024)
    assert (p["sys_tokens"], p["sub_tokens"], p["concurrency"], p["retain"]) == (256, 128, 1, 8)
    assert p["kind"]["empirical"]["weights"] == [2, 1, 2]
    assert p["turns"]["empirical"]["values"] == [1, 1, 2]                 # chain lengths 1, 1, 1, 2, 3 in three shares
    # extends: in - previous in - previous out (110, 75); the rewrite: in - kept (350 - 256)
    assert p["turn_in"]["empirical"]["values"] == [84, 110]
    # past the kind's prefix or what the session had stored, the longer: 300-256, 300-128, 150, 50, max(0, 200-256)
    assert p["first_in"]["empirical"]["values"] == [22, 124]
    assert p["turn_out"]["empirical"]["values"] == [4, 11]
    assert p["trim"]["mixture"] == [{"weight": 0.6667, "dist": {"const": 0}},
                                    {"weight": 0.3333, "dist": {"empirical": {"values": [184], "weights": [1]}}}]   # 420 + 20 - 256
    assert p["think"]["empirical"]["values"] == [500000000, 1166666667]   # None (no think time) left out
    assert p["reuse"]["mixture"][0]["weight"] == 0


def test_agentx_reference_is_the_corpus_chunk_accounting(tmp_path, corpus):
    out = tmp_path / "ref.json"
    r = _agentx("reference", corpus, "--chunk-tokens", 128, "--context", 4096, "--bins", 2, "-o", out)
    assert r.returncode == 0, r.stderr
    ref = json.loads(out.read_text())
    assert (ref["corpus"], ref["traces"], ref["requests"], ref["chains"], ref["first"]) == ("traces.jsonl", 2, 8, 5, 5)
    assert ref["chunks_stored"] == {"total": 6, "mean": 0.75, "shares": [0, 1]}
    assert ref["chunks_hit"] == {"total": 7, "mean": 0.88, "shares": [0, 2]}
    assert ref["out_tokens"] == {"total": 63, "mean": 7.9}
    assert ref["subagent_shared_prefix_tokens"] == {"median": 128, "n": 1}
    assert "requests 8, chains 5, chunks stored 6 (0.75 per request), hit 7" in r.stdout


def test_agentx_needs_its_outputs():
    for args in (("fit", "x"), ("reference", "x"), ("log", "x"), ("holds", "x")):
        r = _agentx(*args)
        assert r.returncode == 2 and "fit, reference, and log need -o; holds needs --replay" in r.stderr, args


# The replay of CORPUS as replay_agentx.py would send it (fi 2 of session 1 over the context): per
# request (session, index, prompt tokens = 64 x blocks, completion, the engine's lookups (seconds,
# tokens held), when it admitted it). Seconds in the replay's log are 1/640 s per prompt token and
# 0.01 s per reply token. Two prompts of 320 and two of 384 tokens: matched in first-lookup order.
REPLAY = [(0, 0, 320, 10, [(0.0, 0)], 0.1), (1, 0, 192, 7, [(0.5, 0)], 0.6), (0, 1, 448, 20, [(1.0, 320)], 1.2),
          (1, 1, 64, 3, [(1.5, 0)], 1.6), (0, 2, 320, 5, [(2.0, 256), (2.05, 128)], 2.1),   # waited: held fell to 128
          (1, 3, 256, 4, [(2.5, 128)], 2.6), (0, 3, 384, 6, [(3.0, 256)], 3.1), (0, 4, 384, 8, [(4.0, 128)], 4.1)]


def _stamp(t):
    return "[2026-10-08 13:40:%02d,%03d]" % (int(t), round(t * 1000) % 1000)


def _replay_logs(d):
    """The replay's log and the server's (vLLM's conditions, LMCache's lookups and stores, noise)."""
    replay = ["sessions 2 context 4096 (server 4096) vocab 151643 prefixes (main, sub) 256,128 0,0"]
    events = []
    for n, (s, i, p, c, looks, admit) in enumerate(REPLAY):
        replay.append("req %d session %d index %d prompt %d cached None completion %d %.2fs" % (n, s, i, p, c, p / 640 + c * 0.01))
        for t, h in looks:
            events.append((t, "(EngineCore pid=7) %s LMCache INFO: Reqid: cmpl-%d-%d, Total tokens %d, Inference Engine computed "
                              "tokens: %d, LMCache hit tokens: 0, need to load: 0" % (_stamp(t), s, i, p, h)))
        events.append((admit, "(EngineCore pid=7) %s LMCache INFO: [req_id=cmpl-%d-%d] %s %d out of total %d tokens." % (
            _stamp(admit), s, i, "Retrieved" if looks[-1][1] else "Stored", p, p)))
    events.append((1.3, "(EngineCore pid=7) %s LMCache INFO: \x1b[33mReqid: cmpl-0-1, Total tokens 448, Inference Engine computed "
                        "tokens: 0\x1b[0m" % _stamp(1.3)))                               # after its admission: not a hold
    events.append((1.4, "(EngineCore pid=7) INFO 10-08 13:40:01 [metrics.py:1] Avg prompt throughput: 1.0 tokens/s"))
    head = ["(APIServer pid=6) INFO 10-08 13:39:00 [api_utils.py:286] non-default args: {'model': 'Qwen/Qwen2.5-0.5B-Instruct', "
            "'max_model_len': 4096, 'kv_cache_memory_bytes': 1048576}",
            "(EngineCore pid=7) INFO 10-08 13:39:30 [kv_cache_utils.py:1] GPU KV cache size: 1,024 tokens, Maximum concurrency 1.00x",
            "(EngineCore pid=7) [2026-10-08 13:39:31,000] LMCache INFO: Creating LMCacheEngine with config: {'chunk_size': 256, "
            "'local_cpu': False, 'max_local_cpu_size': 2.0}"]
    (d / "replay.log").write_text("\n".join(replay) + "\n")
    (d / "serve.log").write_text("\n".join(head + [l for _, l in sorted(events, key=lambda e: e[0])]) + "\n")
    return d / "replay.log", d / "serve.log"


def test_agentx_replayed_takes_each_request_s_last_lookup_before_its_admission(monkeypatch, tmp_path):
    ax = _kit(monkeypatch, "agentx")
    log, serve = _replay_logs(tmp_path)
    sessions, sent = ax.replayed(log, serve)
    assert sessions == 2
    assert sent == {(s, i): (p, looks[-1][1]) for s, i, p, _, looks, _ in REPLAY}
    c_in, c_out = ax.service(log)
    assert (round(c_in, 7), round(c_out, 7)) == (round(1 / 640, 7), 0.01)
    times = ax.engine_times(log, serve)
    assert times[(0, 2)][1] - times[(0, 2)][0] == pytest.approx(0.1) and times[(1, 3)][0] - times[(0, 0)][0] == pytest.approx(2.5)
    other = tmp_path / "other.log"
    other.write_text(log.read_text().replace("prompt 64 ", "prompt 65 "))
    with pytest.raises(AssertionError, match="no request of 65 prompt tokens"):
        ax.replayed(other, serve)
    other.write_text("\n".join(log.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(AssertionError, match="requests the replay's does not"):
        ax.replayed(other, serve)


def test_agentx_fit_of_a_replay_takes_keep_from_the_engine_and_records_its_conditions(tmp_path, corpus):
    """`agentx.py fit --replay`: the rows are the requests the replay sent, keep is what the engine
    held of each continuing request's kept prefix past its prefix's whole chunks (a prefix held
    whole a lower bound), and the file's provenance is the engine the holds stand for (§3.64).
    `agentx.py log` keeps what the fit reads: the fit of the trimmed log is the same file."""
    log, serve = _replay_logs(tmp_path)
    trimmed = tmp_path / "lmcache.log"
    r = _agentx("log", serve, "-o", trimmed)
    assert r.returncode == 0, r.stderr
    kept = trimmed.read_text().splitlines()
    assert len(kept) == 3 + 2 * len(REPLAY) + 1 and "\x1b" not in trimmed.read_text()       # conditions; first, last lookup, admission
    assert not any("throughput" in l or "Total tokens 448, Inference Engine computed tokens: 0" in l for l in kept)
    outs = []
    for server in (serve, trimmed):
        out = tmp_path / f"fit.{server.name}.json"
        r = _agentx("fit", corpus, "--replay", log, server, "--chunk-tokens", 128, "--context", 4096, "--bins", 2,
                    "--condition", "gpu=\"test\"", "-o", out)
        assert r.returncode == 0, r.stderr
        outs.append(json.loads(out.read_text()))
    assert outs[0] == outs[1]
    f = outs[0]
    p = values_of(f)
    assert (p["concurrency"], p["requests"], p["sys_prompts"], p["sub_prompts"], p["think"]) == (2, 4, 1, 1, {"const": 0})
    # continuing: (64 held past 256, held whole), (256 - 128, not whole), (128 held of 256 kept, under the 256 prefix: 0)
    assert p["keep"]["empirical"]["values"] == [43, 128]
    assert p["sys_held"]["empirical"]["values"] == [64, 192] and p["sub_held"]["empirical"]["values"] == [128]   # [0, 256, 128, 128], [128]
    assert f["doc"].endswith("the requests replay_agentx.py sent (2 sessions as 2 slots, 4 requests each, back to back), and keep "
                             "from the server's log: of 3 continuing requests the engine held the whole kept prefix in 1.")
    assert f["provenance"]["conditions"] == {"model": "Qwen/Qwen2.5-0.5B-Instruct", "max_model_len": 4096, "kv_cache_memory_bytes": 1048576,
                                             "engine_pool_tokens": 1024, "cpu_tier_bytes": 0, "slots_per_engine": 2, "gpu": "test",
                                             "latency_s_per_prompt_token": 0.00156, "latency_s_per_reply_token": 0.01}
    out = tmp_path / "scaled.json"
    r = _agentx("fit", corpus, "--replay", log, serve, "--chunk-tokens", 128, "--context", 4096, "--bins", 2, "--time-scale", 2,
                "--doc", "mine", "-o", out)
    assert r.returncode == 0, r.stderr
    f = json.loads(out.read_text())
    assert f["doc"] == "mine" and values_of(f)["think"]["empirical"]["values"] == [1000000000, 2333333333]


def test_agentx_holds_simulates_the_engine_s_prefix_cache_under_the_replay(tmp_path, corpus):
    """`agentx.py holds`: vLLM's prefix cache over the replay's requests, on the engine's timeline or
    clockless, at the server's pool or `--pool`; per request, the held tokens measured and predicted."""
    log, serve = _replay_logs(tmp_path)
    with corpus.open("a") as f:                                  # a session the replay did not play
        f.write(json.dumps({"requests": [_req(0, [1, 2, 3, 4], 256, 1)]}) + "\n")
    got = {}
    for name, extra in (("engine", ()), ("clockless", ("--clockless",)), ("service", ("--clockless", "--service", "0.001", "0.01")),
                        ("small", ("--pool", "256"))):
        out = tmp_path / f"{name}.txt"
        r = _agentx("holds", corpus, "--replay", log, serve, "--chunk-tokens", 128, *extra, "-o", out)
        assert r.returncode == 0, r.stderr
        got[name] = (r.stdout, [tuple(map(int, l.split())) for l in out.read_text().splitlines()])
    stdout, rows = got["engine"]
    assert stdout.startswith("pool 1024 tokens, the engine's timeline: 8 requests, 0 preemptions;")
    assert [r[:4] for r in rows] == sorted((s, i, p, looks[-1][1]) for s, i, p, _, looks, _ in REPLAY)
    pred = {(s, i): q for s, i, _, _, q in rows}
    # 16 blocks of 64 tokens hold every session's blocks: each request finds its whole stored prefix;
    # the replay serves one model, so session 1's last request finds the other model's [30, 31, 32]
    assert pred == {(0, 0): 0, (0, 1): 256, (0, 2): 128, (0, 3): 256, (0, 4): 256, (1, 0): 0, (1, 1): 0, (1, 3): 192}
    assert got["clockless"][0].startswith("pool 1024 tokens, clockless timeline: 8 requests") and got["clockless"][1] == rows
    assert got["service"][1] == rows
    # 4 blocks: every admission preempts the running request of the other session (5 times), and only
    # s0i2 and s0i4 find two blocks of their predecessor's freed head still queued
    stdout, small = got["small"]
    assert stdout.startswith("pool 256 tokens, the engine's timeline: 8 requests, 5 preemptions; held tokens measured 960, predicted 256 ")
    assert {(s, i): q for s, i, _, _, q in small if q} == {(0, 2): 128, (0, 4): 128}


def test_serve_sh_starts_vllm_with_lmcache_s_connector(tmp_path):
    """`kit/serve.sh` (a `vllm` on the path that prints what it was given): the model and port, the
    context, the KV pool when KV_BYTES is set, LMCache's connector, VLLM_ARGS as separate flags,
    and LMCache's configuration beside the script unless LMCACHE_CONFIG_FILE names one."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    vllm = bin_ / "vllm"
    vllm.write_text('#!/bin/sh\necho "LMCACHE_CONFIG_FILE=$LMCACHE_CONFIG_FILE VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"\n'
                    'for a in "$@"; do echo "$a"; done\n')
    vllm.chmod(0o755)
    connector = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

    def serve(*args, **env):
        e = {k: v for k, v in os.environ.items() if k not in ("KV_BYTES", "MAX_MODEL_LEN", "VLLM_ARGS", "LMCACHE_CONFIG_FILE")}
        r = subprocess.run(["sh", str(KV / "serve.sh"), *args], capture_output=True, text=True, env={**e, "PATH": f"{bin_}:{e['PATH']}", **env})
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    assert serve() == [f"LMCACHE_CONFIG_FILE={KV}/lmcache.yaml VLLM_USE_FLASHINFER_SAMPLER=0", "serve", "Qwen/Qwen2.5-0.5B-Instruct",
                       "--port", "8000", "--max-model-len", "4096", "--gpu-memory-utilization", "0.7", "--kv-transfer-config", connector]
    got = serve("m/x", "9000", KV_BYTES="2147483648", MAX_MODEL_LEN="131072", VLLM_ARGS='--hf-overrides {"a":1} --seed 3',
                LMCACHE_CONFIG_FILE="/c.yaml")
    assert got == ["LMCACHE_CONFIG_FILE=/c.yaml VLLM_USE_FLASHINFER_SAMPLER=0", "serve", "m/x", "--port", "9000", "--max-model-len", "131072",
                   "--gpu-memory-utilization", "0.7", "--kv-cache-memory-bytes", "2147483648", "--kv-transfer-config", connector,
                   "--hf-overrides", '{"a":1}', "--seed", "3"]


# --- the load generators, against a stand-in for vLLM's OpenAI-compatible server ------------------

class _Engine(http.server.BaseHTTPRequestHandler):
    """vLLM's endpoints as the kit uses them: a token is a word (or an id of a prompt given as ids),
    a message costs its words and 3, the generation prompt 1; a reply is `max_tokens` words; a
    request whose prompt and reply exceed the context is refused with 400, as vLLM refuses it."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        with srv.lock:
            srv.seen.append((self.path, body))
            srv.served += 1
            serial = srv.served

        def count(b):
            if "messages" in b:
                return sum(len(m["content"].split()) + 3 for m in b["messages"]) + bool(b.get("add_generation_prompt", True))
            return len(b["prompt"]) if isinstance(b["prompt"], list) else len(b["prompt"].split())
        if self.path == "/tokenize":
            out = {"count": count(body), "max_model_len": srv.limit}
        else:
            n, want = count(body), body["max_tokens"]
            if n + want > srv.limit or srv.refuse(body):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "maximum context length"}')
                return
            text = " ".join("r%d" % serial for _ in range(want))
            out = {"choices": [{"message": {"content": text}, "text": text}],
                   "usage": {"prompt_tokens": n, "completion_tokens": want, "prompt_tokens_details": None}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def engine():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    srv.seen, srv.served, srv.lock, srv.limit, srv.refuse = [], 0, threading.Lock(), 4096, lambda body: False
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.url = "http://127.0.0.1:%d" % srv.server_address[1]
    yield srv
    srv.shutdown()
    srv.server_close()


def _run(script, *args, **kw):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, timeout=120, **kw)


SENT_LINE = re.compile(r"req (\d+) session (\d+) index (\d+) prompt (\d+) cached None completion (\d+) [\d.]+s$")


def test_replay_agentx_sends_each_session_s_share_with_the_corpus_s_prefix_structure(monkeypatch, engine, corpus):
    """`replay_agentx.py`: each session plays its own share of `--requests`, the first of its requests
    in time order, a request over the context counting against it; a block's tokens are a function of
    the session and its id, or of its place in a prefix the sessions share; `max_tokens` the corpus's
    `out`. What it sends is what `agentx.py fit --replay` reads."""
    ax = _kit(monkeypatch, "agentx")
    rx = _kit(monkeypatch, "replay_agentx")
    r = _run(KV / "replay_agentx.py", corpus, "--url", engine.url, "--sessions", 2, "--requests", 9, "--vocab", 5000,
             "--chunk-tokens", 128, "--time-scale", 0.001)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "sessions 2 context 131072 (server 4096) vocab 5000 prefixes (main, sub) 256,0 256,0"   # [1, 2] is the main prefix's
    sent = sorted(tuple(map(int, m.groups())) for m in map(SENT_LINE.match, lines[1:]))
    assert len(sent) == len(lines) - 1
    # shares of 9 over 2 sessions: 5 and 4; session 1's index 2 is over the server's context
    assert sent == [(0, 0, 0, 320, 10), (1, 0, 1, 448, 20), (2, 0, 2, 320, 5), (3, 0, 3, 384, 6), (4, 0, 4, 384, 8),
                    (5, 1, 0, 192, 7), (6, 1, 1, 64, 3), (8, 1, 3, 256, 4)]
    prompts = [b["prompt"] for p, b in engine.seen if p == "/v1/completions"]
    assert len(prompts) == 8 and all(1000 <= t < 5000 for p in prompts for t in p)
    by_len = {}
    for p in prompts:
        by_len.setdefault(len(p), []).append(p)
    main = [t for p in range(4) for t in rx.tokens_of("main %d" % p, 5000)]
    assert by_len[448][0][:256] == main and sorted(p[:256] == main for p in by_len[384]) == [False, True]   # the main agent's, a sub-agent's
    assert all(p[:128] == main[:128] for p in by_len[320] + by_len[384])
    assert by_len[256][0][:192] == [t for b in (30, 31, 32) for t in rx.tokens_of("1 %d" % b, 5000)]
    assert rx.tokens_of("main 0", 5000) == rx.tokens_of("main 0", 5000) != rx.tokens_of("0 1", 5000)
    assert ax.BLOCK == 64


def test_replay_agentx_skips_a_prompt_whose_blocks_exceed_the_context(engine, tmp_path):
    """A corpus prompt of `in` tokens is sent as its blocks, 64 tokens each, up to 63 more than `in`:
    one whose blocks leave no room for a reply is skipped like one over the context (vLLM refuses
    it), and the session goes on."""
    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps({"requests": [_req(0, [1, 2], 100, 5), _req(1, list(range(1, 65)), 4090, 5),
                                          _req(2, [1, 2, 3], 150, 5)]}) + "\n" + json.dumps({"requests": [_req(0, [7], 10, 1)]}) + "\n")
    r = _run(KV / "replay_agentx.py", p, "--url", engine.url, "--sessions", 1, "--requests", 3)      # the first session only
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr
    assert [m.group(3) for m in map(SENT_LINE.match, r.stdout.splitlines()[1:])] == ["0", "2"]


def test_replay_agentx_fails_when_the_server_fails_a_request(engine, corpus):
    """A session whose request the server fails stops; the others finish, and the replay exits
    with an error naming it, since its log is short of that session's share."""
    engine.refuse = lambda body: len(body["prompt"]) == 448             # session 0's second request
    r = _run(KV / "replay_agentx.py", corpus, "--url", engine.url, "--sessions", 2, "--requests", 9)
    assert r.returncode == 1 and r.stderr.strip() == "replay_agentx.py: session 0: request 1: HTTP Error 400: Bad Request", r.stderr
    assert sorted(m.group(2, 3) for m in map(SENT_LINE.match, r.stdout.splitlines()[1:])) == [("0", "0"), ("1", "0"), ("1", "1"), ("1", "3")]
    r = _run(KV / "replay_agentx.py", corpus, "--url", engine.url, "--skip", 1)
    assert r.returncode == 2 and "unrecognized arguments: --skip 1" in r.stderr


def _said(who, n):
    return {"from": who, "value": " ".join(["w"] * n)}


# ShareGPT's shape: a long conversation in parts that overlap by one message (a), one that starts
# with the model (b) and one of odd length (d), both unused; one whose second turn no longer fits
# a context of 200 (e), one whose first turn never fits (f), and single turns.
SHAREGPT = ([{"id": "a_0", "conversations": [_said("human", 5), _said("gpt", 5), _said("human", 6)]},
             {"id": "a_2", "conversations": [_said("human", 6), _said("gpt", 7)]},
             {"id": "b_0", "conversations": [_said("gpt", 3), _said("human", 3)]},
             {"id": "c_0", "conversations": [_said("human", 4), _said("gpt", 4)]},
             {"id": "d_0", "conversations": [_said("human", 2), _said("gpt", 2), _said("human", 2)]},
             {"id": "e_0", "conversations": [_said("human", 60), _said("gpt", 60), _said("human", 60), _said("gpt", 60)]},
             {"id": "f_0", "conversations": [_said("human", 250), _said("gpt", 5)]}]
            + [{"id": f"{x}_0", "conversations": [_said("human", 3), _said("gpt", 3)]} for x in "ghijk"])

REQ_LINE = re.compile(r"req (\d+) conv (\d+) turn (\d+) ago (\S+) prompt (\d+) cached None completion (\d+) [\d.]+s$")


def test_replay_serves_open_conversations_in_turn_and_fit_takes_its_log(monkeypatch, engine, tmp_path):
    """`replay.py`: ShareGPT's parts joined, `--active` conversations served in turn (each returns
    `--active` requests after its last turn), one that ends or no longer fits replaced in its
    place, one whose first turn never fits passed over; `--save` and `--replay` put one run's
    replies into another's history. `fit.py` reads its log and LMCache's line per request."""
    rp = _kit(monkeypatch, "replay")
    data = tmp_path / "sharegpt.json"
    data.write_text(json.dumps(SHAREGPT))
    convs = rp.conversations(data)
    assert [len(c) for c in convs] == [2, 1, 2, 1, 1, 1, 1, 1, 1]                       # a, c, e, f, g..k
    assert convs[0] == [(" ".join(["w"] * 5),) * 2, (" ".join(["w"] * 6), " ".join(["w"] * 7))]
    engine.limit = 200
    saved = tmp_path / "replies.json"
    args = ("--url", engine.url, "--requests", 7, "--active", 2, "--system-words", 20, "--max-reply", 50, "--seed", 3)
    r = _run(KV / "replay.py", data, *args, "--save", saved)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "conversations 9 context 200 system prompts [32, 32]"
    reqs = [m.groups() for m in map(REQ_LINE.match, lines[1:]) if m]
    events = [l for l in lines[1:] if not REQ_LINE.match(l)]
    assert len(reqs) == 7 and all(re.fullmatch(r"conv \d+ (ended|cut) after \d+ turns", l) for l in events)
    assert all(ago in ("-", "2") for _, _, _, ago, _, _ in reqs)
    turns = {}
    for n, conv, turn, ago, prompt, completion in reqs:
        assert int(turn) == turns.get(conv, -1) + 1 and (ago == "-") == (turn == "0")
        turns[conv] = int(turn)
        assert int(prompt) + int(completion) <= 200
    assert any(l.endswith("ended after 1 turns") for l in events)
    assert any("cut after 1 turns" in l for l in events), r.stdout                     # e's second turn: 212 + 50 > 200
    replies = json.loads(saved.read_text())
    assert sorted(replies) == sorted("%s %s" % (c, t) for _, c, t, *_ in reqs)
    # LMCache's line per request, the engine holding a returning conversation's last prompt and reply
    last, engine_log = {}, []
    for n, conv, turn, ago, prompt, completion in reqs:
        engine_log.append("LMCache INFO: Reqid: %s, Total tokens %s, Inference Engine computed tokens: %d" % (n, prompt, last.get(conv, 0)))
        last[conv] = int(prompt) + int(completion)
    (tmp_path / "replay.log").write_text(r.stdout + "a line of neither kind\n")
    (tmp_path / "lmcache.log").write_text("\n".join(engine_log) + "\n")
    out = tmp_path / "fit.json"
    f = _run(KV / "fit.py", tmp_path / "replay.log", tmp_path / "lmcache.log", "--chunk-tokens", 16, "--bins", 2, "-o", out)
    assert f.returncode == 0, f.stderr
    p = values_of(json.loads(out.read_text()))
    began, before = 0, ""                                # conversations begun where one had ended (not cut)
    for l in lines[1:]:
        m = REQ_LINE.match(l)
        if not m:
            before += l
            continue
        began += m[4] == "-" and "ended" in before and "cut" not in before
        before = ""
    assert 0 < began
    assert (p["requests"], p["context"], p["sys_tokens"], p["sys_prompts"]) == (7, 200, 32, 2)
    assert p["reuse"]["mixture"] == [{"weight": round(began / 5, 4), "dist": None}, {"weight": round(1 - began / 5, 4), "dist": {"const": 2}}]
    first_in = sorted(int(prompt) - 32 for _, _, t, _, prompt, _ in reqs if t == "0")
    assert min(p["turn_in"]["empirical"]["values"]) <= first_in[0]
    # a second run sent the first one's replies: its histories hold them byte for byte
    engine.seen.clear()
    r2 = _run(KV / "replay.py", data, *args, "--replay", saved)
    assert r2.returncode == 0, r2.stderr
    held = {m["content"] for p, b in engine.seen if p == "/v1/chat/completions" for m in b["messages"] if m["role"] == "assistant"}
    assert held and held <= set(replies.values())


def test_chat_sends_seeded_conversations_round_robin(engine, tmp_path):
    """`chat.py`: `--conversations` conversations of `--turns` turns, each from one of the shared system
    prompts, text from the seed; `--replay` puts a saved run's replies into the history."""
    saved = tmp_path / "replies.json"
    args = ("--url", engine.url, "--conversations", 3, "--turns", 2, "--system-words", 5, "--turn-words", 4, "--max-tokens", 3)
    r = _run(KV / "chat.py", *args, "--save", saved)
    assert r.returncode == 0, r.stderr
    got = [re.fullmatch(r"conv (\d) turn (\d) prompt (\d+) cached None completion 3 [\d.]+s", l).groups() for l in r.stdout.splitlines()]
    # a system prompt of 14 words and a turn of 8 (and 3 tokens each, 1 for the reply's start); the reply 3 more
    assert got == [("0", "0", "29"), ("1", "0", "29"), ("2", "0", "29"), ("0", "1", "46"), ("1", "1", "46"), ("2", "1", "46")]
    first = [b["messages"] for p, b in engine.seen]
    replies = json.loads(saved.read_text())
    assert sorted(replies) == ["0 0", "0 1", "1 0", "1 1", "2 0", "2 1"]
    assert first[0][0]["content"] == first[2][0]["content"] != first[1][0]["content"]   # two system prompts, by conversation
    engine.seen.clear()
    r = _run(KV / "chat.py", *args, "--replay", saved, "--concurrency", 2)
    assert r.returncode == 0, r.stderr
    assert [l.split(" prompt")[0] for l in r.stdout.splitlines()] == ["conv %s turn %s" % g[:2] for g in got]   # printed in order
    second = [b["messages"] for p, b in engine.seen]
    assert sorted(m["content"] for ms in second for m in ms if m["role"] == "user") == sorted(m["content"] for ms in first for m in ms if m["role"] == "user")
    assert {m["content"] for ms in second for m in ms if m["role"] == "assistant"} == {replies[k] for k in ("0 0", "1 0", "2 0")}


# --- kits of applications not installed here: their own logic, apart from the library ------------

FAISS = ROOT / "workloads" / "vector-search" / "faiss-ivf-ondisk" / "kit"


def _load(monkeypatch, path, name):
    """A kit module by its path (several kits have a `params.py` or a `search.py`), its directory first
    on the path for its own imports."""
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.delitem(sys.modules, "search", raising=False)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_faiss_params_fit_list_sizes_popularity_and_the_index_reads(monkeypatch, tmp_path):
    """`faiss-ivf-ondisk/kit/params.py`: list bytes log-normal over the non-empty lists (median the
    geometric mean), list popularity a Zipf fitted to the more probed half of the lists, and the
    lengths `read_index` asked for on the index file."""
    np = pytest.importorskip("numpy")
    import argparse
    fp = _load(monkeypatch, FAISS / "params.py", "faiss_params")
    size = np.array([1000, 4000, 1000, 4000, 2000, 2000, 0, 2000])
    hits = [144, 36, 16, 9, 5, 3, 1, 0]                       # 144 / rank^2 over the top half
    probes = np.array([l for l, n in enumerate(hits) for _ in range(n)]).reshape(-1, 1)
    trace = ['9 1.0 read(5</mnt/nfs/out/populated.index>, "IwPQ"..., 8192) = 8192 <0.000010>\n',
             '9 1.1 read(5</mnt/nfs/out/populated.index>, ""..., 516096) = 400000 <0.000010>\n',    # the length asked for
             '9 1.2 read(6</mnt/nfs/out/merged_index.ivfdata>, ""..., 4096) = 4096 <0.000010>\n']
    a = argparse.Namespace(doc="d", threads=1, calls=20, batch=2, nprobe=64)
    f = fp.params(a, size, probes, 32, 32, trace, 680212)
    p = values_of(f)
    assert (f["abstract"], f["doc"], p["lists"], p["code_size"], p["prefetch"], p["index_bytes"]) == ("faiss_ivf_ondisk", "d", 8, 32, 32, 680212)
    assert p["list_bytes"] == {"lognormal": {"median": 2000, "sigma": 0.52, "min": 0, "max": 4000}}
    assert p["list_pop"] == {"zipf": {"s": 2.0}} and p["index_reads"] == [8192, 516096]
    assert values_of(fp.params(argparse.Namespace(doc="d", threads=1, calls=20, batch=1, nprobe=16), size, probes, 32, 32, [], 1))["prefetch"] == 16
    v = np.array([[3, 0, 0, 0], [3, 0, 0, 0]], dtype="int32")
    v[:, 1:] = np.array([[1.5, 2, 3], [4, 5, 6.25]], dtype="float32").view("int32")
    v.tofile(tmp_path / "q.fvecs")
    assert fp.fvecs(tmp_path / "q.fvecs").tolist() == [[1.5, 2, 3], [4, 5, 6.25]]


def test_faiss_faults_reads_the_nfs_mount_s_counters_and_the_resident_pages(monkeypatch, tmp_path):
    """`faiss-ivf-ondisk/kit/faults.py`: the READ ops and server bytes read of the longest NFS mount
    holding the file, from mountstats; the file's resident pages from `mincore`."""
    pytest.importorskip("numpy")
    ft = _load(monkeypatch, FAISS / "faults.py", "faiss_faults")
    stats = tmp_path / "mountstats"
    stats.write_text("device sysfs mounted on /sys with fstype sysfs\n"
                     "device srv:/export mounted on /mnt with fstype nfs4 statvers=1.1\n\tbytes:\t1 2 3 4 500 6 7 8\n\tREAD:\t10 10 0 1 2\n"
                     "device srv:/export/sub mounted on /mnt/nfs with fstype nfs statvers=1.1\n\tbytes:\t1 2 3 4 900 6 7 8\n\tREAD:\t20 20 0 1 2\n"
                     "device srv:/other mounted on /mnt/nfsx with fstype nfs4 statvers=1.1\n\tbytes:\t1 2 3 4 111 6 7 8\n\tREAD:\t99 1 0 1 2\n"
                     "device /dev/sda1 mounted on /mnt/nfs/local with fstype ext4\n")
    assert ft.nfs_reads("/mnt/nfs/out/merged_index.ivfdata", stats) == (20, 900)
    assert ft.nfs_reads("/mnt/other/merged_index.ivfdata", stats) == (10, 500)
    assert ft.nfs_reads("/mnt/nfs/local/x", stats) == (20, 900)          # the ext4 mount has no counters: its NFS parent's
    assert ft.nfs_reads("/home/x", stats) is None
    f = tmp_path / "lists"
    f.write_bytes(b"\1" * (2 * ft.PAGE + 1))
    f.read_bytes()
    got = ft.resident(f)
    assert got.dtype == bool and len(got) == 3 and got.all()
    ft.drop(f)                                                          # fsync and DONTNEED: no error, the file intact
    assert len(ft.resident(f)) == 3 and f.read_bytes() == b"\1" * (2 * ft.PAGE + 1)


def test_dcp_load_params_from_a_checkpoint_s_metadata(monkeypatch):
    """`dcp-load/kit/params.py`: rank 0's items in file order, each its chunk's bytes, and the order
    `dcp.load` reads them in (by name: layer10 before layer2)."""
    import collections
    dp = _load(monkeypatch, ROOT / "workloads/checkpointing/dcp-load/kit/params.py", "dcp_params")
    Index = collections.namedtuple("MetadataIndex", "fqn offset")
    Storage = collections.namedtuple("StorageInfo", "relative_path offset length")
    Chunk = collections.namedtuple("ChunkStorageMetadata", "offsets sizes")
    f32 = collections.namedtuple("Props", "dtype")(collections.namedtuple("DType", "itemsize")(4))
    Tensor = collections.namedtuple("TensorStorageMetadata", "properties chunks")
    m = collections.namedtuple("Metadata", "storage_data state_dict_metadata")(
        {Index("layer0.weight", (0,)): Storage("__0_0.distcp", 0, 40), Index("layer0.weight", (8,)): Storage("__1_0.distcp", 0, 40),
         Index("layer10.weight", (0,)): Storage("__0_0.distcp", 100, 24), Index("layer2.weight", (0,)): Storage("__0_0.distcp", 50, 16)},
        {"layer0.weight": Tensor(f32, [Chunk((8,), (8,)), Chunk((0,), (8,))]), "layer10.weight": Tensor(f32, [Chunk((0,), (4,))]),
         "layer2.weight": Tensor(f32, [Chunk((0,), (2,))])})
    assert dp.params(m, 777, 200, "d") == {"params_version": 2, "abstract": "dcp_load", "doc": "d", "params": {k: {"default": v} for k, v in {
        "items": 3, "item_bytes": [32, 8, 16], "item_off": [0, 50, 100], "read_order": [0, 2, 1], "meta_bytes": 777, "restore_step": 200}.items()}}


# --- kit scripts run whole against the stand-ins of builder/tests/standins/ (test fakes) ---------
# Each test runs the script as its README does, the stand-ins first on its path, and checks what it
# asked of the library against the kit's README and the committed case it was traced for.

STANDINS = pathlib.Path(__file__).resolve().parent / "standins"
WL = ROOT / "workloads"


def _standin_run(tmp_path, script, *args, env=None, cwd=None):
    """`python SCRIPT ARGS` with the stand-ins on its path: the process and the calls it made."""
    log = tmp_path / "standin.log"
    log.unlink(missing_ok=True)
    path = os.pathsep.join(p for p in (str(STANDINS), os.environ.get("PYTHONPATH")) if p)
    e = dict(os.environ, PYTHONPATH=path, AEIOU_STANDIN_LOG=str(log), **(env or {}))
    r = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, timeout=300, env=e, cwd=cwd)
    return r, [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _case(path):
    return json.loads((WL / path / "params.json").read_text())


def _calls(calls, name):
    return [c for c in calls if c["call"] == name]


def _by_rank(calls):
    """The calls of each rank process, by the rank its `init_process_group` named."""
    rank = {c["pid"]: c["rank"] for c in _calls(calls, "init_process_group")}
    out = {}
    for c in calls:
        if c["pid"] in rank:
            out.setdefault(rank[c["pid"]], []).append(c)
    return out


def _numels(mib):
    """The kit's full-tensor element counts for sizes in MiB (fp32, even)."""
    return [int(m * (1 << 20)) // 4 // 2 * 2 for m in mib]


def test_dcp_save_kit_saves_each_rank_s_half_of_the_case_s_items_at_its_steps(tmp_path):
    """`dcp-save/kit/save.py ROOT 2 2` (its README): two gloo ranks on one rendezvous, each item
    a DTensor sharded on dimension 0, saved at the case's checkpoint steps, each rank's share of an
    item the case's `item_bytes`; `--torch-save` writes the same full tensors in one archive."""
    case = values_of(_case("checkpointing/dcp-save/cases/toy-6-items"))
    out = tmp_path / "out"
    r, calls = _standin_run(tmp_path, WL / "checkpointing/dcp-save/kit/save.py", out, 2, 2)
    assert r.returncode == 0, r.stderr
    ranks = _by_rank(calls)
    assert sorted(ranks) == [0, 1]
    steps = [os.path.join(out, "ckpt", f"step_{s:06}") for s in range(case["ckpt_every"], case["steps"] + 1, case["ckpt_every"])]
    for cs in ranks.values():
        assert cs[0]["call"] == "init_process_group" and cs[-1]["call"] == "destroy_process_group"
        assert (cs[0]["backend"], cs[0]["world_size"], cs[0]["master"]) == ("gloo", 2, ["127.0.0.1", "29511"])
        assert [(c["device_type"], c["mesh_shape"]) for c in _calls(cs, "init_device_mesh")] == [("cpu", [2])]
        assert {tuple(c["placements"]) for c in _calls(cs, "distribute_tensor")} == {("Shard(0)",)}
        saves = _calls(cs, "dcp.save")
        assert [c["checkpoint_id"] for c in saves] == steps
        for s in saves:
            assert list(s["items"]) == [f"layer{i}.weight" for i in range(case["items"])]
            assert [n * 4 for n in s["items"].values()] == case["item_bytes"]
    r, calls = _standin_run(tmp_path, WL / "checkpointing/dcp-save/kit/save.py", "--torch-save", tmp_path / "ts")
    assert r.returncode == 0, r.stderr
    (save,) = _calls(calls, "save")
    assert save["path"] == os.path.join(tmp_path / "ts", "ckpt", "rank0.pt") and (tmp_path / "ts" / "ckpt").is_dir()
    assert [n * 4 // 2 for n in save["items"].values()] == case["item_bytes"]


@pytest.mark.parametrize("case, mib", [
    ("small-last-6-items", "32,64,32,16,1,0.004"),
    ("mixed-9-items", "32,1,64,0.004,0.004,1,32,0.004,16"),
    ("name-order-17-items", "32,0.5,0.5,0.5,0.5,0.5,0.5,0.004,0.004,0.004,0.75,0.75,0.75,0.75,1.9,1.9,16"),
])
def test_dcp_load_kit_loads_each_rank_s_shard_of_the_case_s_checkpoint(tmp_path, case, mib):
    """`dcp-load/kit/load.py [--save] ROOT 2 STEP ITEMS` (its README): `--save` saves random
    tensors (seed 1) and a load restores into zeros in place, from `ckpt/step_STEP`; each rank's
    share of item `i` is the case's `item_bytes[i]`, and the step its `restore_step`."""
    p = values_of(_case(f"checkpointing/dcp-load/cases/{case}"))
    out, step = tmp_path / "out", p["restore_step"]
    path = os.path.join(out, "ckpt", f"step_{step:06}")
    for mode in (["--save"], []):
        r, calls = _standin_run(tmp_path, WL / "checkpointing/dcp-load/kit/load.py", *mode, out, 2, step, mib)
        assert r.returncode == 0, r.stderr
        ranks = _by_rank(calls)
        assert sorted(ranks) == [0, 1]
        for cs in ranks.values():
            assert (cs[0]["backend"], cs[0]["world_size"], cs[0]["master"]) == ("gloo", 2, ["127.0.0.1", "29512"])
            assert cs[-1]["call"] == "destroy_process_group"
            made = _calls(cs, "rand" if mode else "zeros")
            assert [c["n"] for c in made] == _numels(float(m) for m in mib.split(",")) and not _calls(cs, "zeros" if mode else "rand")
            if mode:
                assert _calls(cs, "Generator.manual_seed")[0]["seed"] == 1 and all(c["generator"] for c in made)
            (io,) = _calls(cs, "dcp.save" if mode else "dcp.load")
            assert io["checkpoint_id"] == path
            assert [n * 4 for n in io["items"].values()] == p["item_bytes"]


_DCP_METADATA = """
import pickle, sys, torch
from torch.distributed.checkpoint.metadata import *
f32 = TensorProperties(torch.float32)
m = Metadata({"layer0.weight": TensorStorageMetadata(f32, [ChunkStorageMetadata((0,), (6,)), ChunkStorageMetadata((6,), (6,))]),
              "layer1.weight": TensorStorageMetadata(f32, [ChunkStorageMetadata((0,), (2,)), ChunkStorageMetadata((2,), (2,))])},
             {MetadataIndex("layer0.weight", (0,)): StorageInfo("__0_0.distcp", 0, 30),
              MetadataIndex("layer0.weight", (6,)): StorageInfo("__1_0.distcp", 0, 30),
              MetadataIndex("layer1.weight", (0,)): StorageInfo("__0_0.distcp", 30, 14),
              MetadataIndex("layer1.weight", (2,)): StorageInfo("__1_0.distcp", 30, 14)})
with open(sys.argv[1], "wb") as f:
    pickle.dump(m, f)
"""


def test_dcp_load_params_reads_the_metadata_in_the_checkpoint_it_is_given(tmp_path):
    """`dcp-load/kit/params.py CKPT STEP DOC` (its README): the `.metadata` pickle in CKPT, a path
    relative to the caller, its size the `meta_bytes`, rank 0's items the items."""
    ckpt = tmp_path / "ckpt" / "step_000200"
    ckpt.mkdir(parents=True)
    script = tmp_path / "mkmeta.py"
    script.write_text(_DCP_METADATA)
    r, _ = _standin_run(tmp_path, script, ckpt / ".metadata")
    assert r.returncode == 0, r.stderr
    r, _ = _standin_run(tmp_path, WL / "checkpointing/dcp-load/kit/params.py", "ckpt/step_000200", 200, "a doc", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"params_version": 2, "abstract": "dcp_load", "doc": "a doc", "params": {k: {"default": v} for k, v in {
        "items": 2, "item_bytes": [24, 8], "item_off": [0, 30], "read_order": [0, 1],
        "meta_bytes": (ckpt / ".metadata").stat().st_size, "restore_step": 200}.items()}}


def test_safetensors_kit_saves_the_case_s_model_and_loads_it(tmp_path):
    """`safetensors-load/kit/mkmodel.py DIR` (its README): a seeded random GPT-2 whose tensor
    table is the case's (12 tensors a layer and 4 more; the embedding vocab x width fp32, the
    positions table positions x width), in 100MB shards unless told; `loadmodel.py` loads the
    directory last on its line and copies every parameter only with `--touch`."""
    case = values_of(_case("model-loading/safetensors-load/cases/gpt2-random"))
    r, calls = _standin_run(tmp_path, WL / "model-loading/safetensors-load/kit/mkmodel.py", tmp_path / "model")
    assert r.returncode == 0, r.stderr
    assert [c["call"] for c in calls] == ["manual_seed", "save_pretrained"] and calls[0]["seed"] == 1
    save = calls[1]
    cfg = save["config"]
    assert (save["path"], save["max_shard_size"]) == (str(tmp_path / "model"), "100MB")
    assert 12 * cfg["n_layer"] + 4 == len(case["shard"]) and cfg["n_embd"] % cfg["n_head"] == 0
    assert case["bytes"][0] == cfg["vocab_size"] * cfg["n_embd"] * 4 and cfg["n_positions"] * cfg["n_embd"] * 4 in case["bytes"]
    r, calls = _standin_run(tmp_path, WL / "model-loading/safetensors-load/kit/mkmodel.py", tmp_path / "model", "50MB")
    assert r.returncode == 0 and calls[1]["max_shard_size"] == "50MB", r.stderr
    for flag, out in (("--touch", "8\n"), ("--notouch", "")):
        r, calls = _standin_run(tmp_path, WL / "model-loading/safetensors-load/kit/loadmodel.py", flag, tmp_path / "model")
        assert (r.returncode, r.stdout) == (0, out), r.stderr
        assert [(c["call"], c["path"]) for c in calls] == [("from_pretrained", str(tmp_path / "model"))]


def test_safetensors_touch_copies_every_tensor_of_every_shard_in_file_order(tmp_path):
    """`safetensors-load/kit/touch.py DIR`: the shards in name order, each opened for torch, its
    tensors taken by their data offsets (the order of the case's `off` within a shard), never the
    header's key order; `ONE=1` first sets one thread."""
    case = values_of(_case("model-loading/safetensors-load/cases/gpt2-random"))
    for s in set(case["shard"]):
        offs = [o for o, k in zip(case["off"], case["shard"]) if k == s]
        assert offs == sorted(offs)
    d = tmp_path / "model"
    d.mkdir()
    for name, keys in (("model-00002-of-00002", ["z", "a"]), ("model-00001-of-00002", ["b", "c", "a"])):
        h = {"__metadata__": {"format": "pt"}}
        for k in reversed(keys):                         # the header lists them against file order
            i = keys.index(k)
            h[k] = {"dtype": "F32", "shape": [2], "data_offsets": [8 * i, 8 * i + 8]}
        b = json.dumps(h).encode()
        (d / f"{name}.safetensors").write_bytes(struct.pack("<Q", len(b)) + b + bytes(8 * len(keys)))
    (d / "config.json").write_text("{}")
    for env, threads in (({"ONE": "1"}, [("set_num_threads", 1)]), ({}, [])):
        r, calls = _standin_run(tmp_path, WL / "model-loading/safetensors-load/kit/touch.py", d, env=env)
        assert r.returncode == 0, r.stderr
        assert [(c["call"], c["n"]) for c in calls[:len(threads)]] == threads
        got = [(c["call"], os.path.basename(c["path"]), c.get("name", c.get("framework"))) for c in calls[len(threads):]]
        assert got == [("safe_open", "model-00001-of-00002.safetensors", "pt")] + [
            ("get_tensor", "model-00001-of-00002.safetensors", k) for k in ("b", "c", "a")] + [
            ("safe_open", "model-00002-of-00002.safetensors", "pt")] + [("get_tensor", "model-00002-of-00002.safetensors", k) for k in ("z", "a")]


def test_imagefolder_kit_writes_the_corpus_and_trains_over_it_as_the_case_says(tmp_path):
    """`file-per-sample/kit/mkcorpus.py ROOT N` (its README): JPEGs at quality 75 at
    train/{id div 1300:05}/img_{id:09}.jpg, sized from the encoder's bytes per pixel to near
    lognormal(110 KiB, 0.45); `train.py ROOT/train 16 2 2`: ImageFolder with the README's
    transforms, a seeded shuffling DataLoader with the case's batch and workers, dropping the
    last batch, so the steps are the case's samples // batch x epochs."""
    np = pytest.importorskip("numpy")
    case = values_of(_case("input-pipeline/map/file-per-sample/cases/imagefolder-jpeg"))
    n = 40
    r, calls = _standin_run(tmp_path, WL / "input-pipeline/map/file-per-sample/kit/mkcorpus.py", tmp_path / "c", n)
    assert r.returncode == 0, r.stderr
    assert {(c["format"], c["quality"]) for c in calls} == {("JPEG", 75)} and len(calls) == n + 1
    files = sorted(p.relative_to(tmp_path / "c").as_posix() for p in (tmp_path / "c").rglob("*") if p.is_file())
    assert files == [f"train/00000/img_{i:09}.jpg" for i in range(n)]
    ln = np.log([(tmp_path / "c" / f).stat().st_size for f in files])
    assert abs(ln.mean() - np.log(110 * 1024)) < 0.25 and 0.3 < ln.std() < 0.6, (np.exp(ln.mean()), ln.std())
    assert r.stdout.startswith("bytes per pixel ")
    r, calls = _standin_run(tmp_path, WL / "input-pipeline/map/file-per-sample/kit/train.py", tmp_path / "c/train", case["batch"],
                            case["workers"], 2)
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"files {n} steps {n // case['batch'] * 2}\n" and case["steps"] == case["samples"] // case["batch"] * 2
    assert [c["call"] for c in calls] == ["manual_seed", "ImageFolder", "DataLoader"] and calls[0]["seed"] == 1
    assert (calls[1]["root"], calls[1]["classes"]) == (str(tmp_path / "c/train"), 1)
    assert calls[1]["transform"] == ["RandomResizedCrop(224,)", "RandomHorizontalFlip()", "ToTensor()"]
    dl = calls[2]
    assert (dl["batch_size"], dl["num_workers"], dl["shuffle"], dl["drop_last"], dl["samples"]) == (case["batch"], case["workers"], True, True, n)


@pytest.mark.parametrize("case", ["npy-8mib", "npy-140mib"])
def test_npy_kit_trains_over_the_corpus_as_the_case_says(tmp_path, case):
    """`npy-per-sample/kit/train.py ROOT/train BATCH 2 2 x|xy` (its README) over what
    `mkcorpus.py` wrote: every archive in the class directories, `np.load` of each in a seeded
    shuffling DataLoader with the case's batch and workers, dropping the last batch, so the steps
    are the case's samples // batch x epochs."""
    pytest.importorskip("numpy")
    p = values_of(_case(f"input-pipeline/map/npy-per-sample/cases/{case}"))
    n = 2 * p["batch"] + 1
    r = _run(WL / "input-pipeline/map/npy-per-sample/kit/mkcorpus.py", tmp_path / "c", n, 1 << 20, 0)
    assert r.returncode == 0, r.stderr
    assert p["steps"] == p["samples"] // p["batch"] * 2
    for members in ("x", "xy"):
        r, calls = _standin_run(tmp_path, WL / "input-pipeline/map/npy-per-sample/kit/train.py", tmp_path / "c/train", p["batch"],
                                p["workers"], 2, members)
        assert r.returncode == 0, r.stderr
        assert r.stdout == f"files {n} steps {n // p['batch'] * 2}\n"
        assert [c["call"] for c in calls] == ["manual_seed", "DataLoader"] and calls[0]["seed"] == 1
        dl = calls[1]
        assert (dl["batch_size"], dl["num_workers"], dl["shuffle"], dl["drop_last"], dl["samples"]) == (p["batch"], p["workers"], True, True, n)


def _fvecs(path, rows):
    """An .fvecs file of float32 rows (each row its dimension, then the floats)."""
    import numpy as np
    rows = np.asarray(rows, dtype="float32")
    a = np.empty((len(rows), rows.shape[1] + 1), dtype="int32")
    a[:, 0] = rows.shape[1]
    a[:, 1:] = rows.view("int32")
    a.tofile(path)


def test_faiss_kit_builds_searches_fits_and_faults_the_case_s_index(tmp_path):
    """`faiss-ivf-ondisk/kit/` as its README runs it, OUT relative to the caller: `mkindex.py`
    builds the case's `IVF1024,PQ32` index (trained on sift_learn, sift_base added in four
    contiguous id ranges, merged on disk into OUT/merged_index.ivfdata by an absolute name, so the
    search, run from OUT, finds it); `search.py --queries 20 --threads 1` makes the case's 20 calls
    of one query at nprobe 64; `params.py` records those parameters and the trace's reads of the
    index file; `faults.py` probes as the search did."""
    np = pytest.importorskip("numpy")
    case = values_of(_case("vector-search/faiss-ivf-ondisk/cases/sift1m-ivf1024-pq32"))
    rng = np.random.default_rng(1)
    (tmp_path / "sift").mkdir()
    _fvecs(tmp_path / "sift/sift_learn.fvecs", rng.random((50, 128)))
    _fvecs(tmp_path / "sift/sift_base.fvecs", rng.random((101, 128)))
    _fvecs(tmp_path / "sift/sift_query.fvecs", rng.random((30, 128)))
    kit = WL / "vector-search/faiss-ivf-ondisk/kit"
    r, calls = _standin_run(tmp_path, kit / "mkindex.py", "sift", "out", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    (fac,) = _calls(calls, "index_factory")
    assert (fac["d"], fac["description"]) == (128, f"IVF{case['lists']},PQ{case['code_size']}")
    assert [c["shape"] for c in _calls(calls, "Index.train")] == [[50, 128]]
    assert [c["ids"] for c in _calls(calls, "Index.add_with_ids")] == [[0, 24], [25, 49], [50, 74], [75, 100]]
    (merge,) = _calls(calls, "merge_ondisk")
    assert merge["ivfdata"] == str(tmp_path / "out" / "merged_index.ivfdata")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["merged_index.ivfdata", "populated.index"]
    assert r.stderr == f"101 vectors, {case['lists']} lists, code_size {case['code_size']}\n"

    for batch, sizes in ((1, [1] * 20), (3, [3] * 6 + [2])):
        r, calls = _standin_run(tmp_path, kit / "search.py", "out", "sift/sift_query.fvecs", "--queries", case["calls"] * case["batch"],
                                "--threads", case["threads"], "--batch", batch, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert [c["n"] for c in _calls(calls, "omp_set_num_threads")] == [case["threads"]]
        assert [(c["path"], c["cwd"]) for c in _calls(calls, "read_index")] == [("populated.index", str(tmp_path / "out"))]
        s = _calls(calls, "Index.search")
        assert [c["n"] for c in s] == sizes and {(c["k"], c["nprobe"]) for c in s} == {(10, case["nprobe"])}

    (tmp_path / "trace.txt").write_text(
        f'9 1.0 read(5<{tmp_path}/out/populated.index>, "IwPQ"..., 8192) = 8192 <0.000010>\n'
        f'9 1.1 read(5<{tmp_path}/out/populated.index>, ""..., 516096) = 400 <0.000010>\n')
    r, calls = _standin_run(tmp_path, kit / "params.py", "out", "sift/sift_query.fvecs", "trace.txt", "a doc", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    f = json.loads(r.stdout)
    p = values_of(f)
    assert (f["abstract"], f["doc"], set(p)) == ("faiss_ivf_ondisk", "a doc", set(case))
    same = ("threads", "calls", "batch", "nprobe", "prefetch", "lists", "code_size")
    assert {k: p[k] for k in same} == {k: case[k] for k in same}
    assert p["index_bytes"] == (tmp_path / "out/populated.index").stat().st_size and p["index_reads"] == [8192, 516096]
    assert (p["list_bytes"]["lognormal"]["min"], p["list_bytes"]["lognormal"]["max"]) == (0, case["code_size"] + 8)
    assert [(c["n"], c["k"]) for c in _calls(calls, "quantizer.search")] == [(30, case["nprobe"])]

    r, calls = _standin_run(tmp_path, kit / "faults.py", "out", "sift/sift_query.fvecs", "--queries", 1, "--threads", 1, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == f"probes 64, distinct lists 64 of 1024, their bytes {64 * (case['code_size'] + 8)}"
    assert re.fullmatch(r"pages: resident before \d+, of the probed lists 1, resident after \d+, probed and not resident \d+, "
                        r"resident and not probed \d+", lines[1]) and len(lines) in (2, 3)
    assert [(c["n"], c["k"], c["nprobe"]) for c in _calls(calls, "Index.search")] == [(1, 10, 64)]
    assert [c["n"] for c in _calls(calls, "omp_set_num_threads")] == [1]


def test_diskann_build_kit_converts_the_base_and_builds_as_the_case_says(tmp_path):
    """`diskann-build/kit/build.py` as its README runs it: `--fbin` writes the fvecs rows as an
    .fbin; a build of the base into INDEX_DIR (made) asks for the case's graph degree, build list
    and build memory, l2, fp32, a thread per CPU unless told."""
    np = pytest.importorskip("numpy")
    doc = _case("vector-search/diskann-build/cases/sift1m-13-shards")["doc"]
    rows = np.arange(40, dtype="float32").reshape(10, 4) / 8
    _fvecs(tmp_path / "base.fvecs", rows)
    build = WL / "vector-search/diskann-build/kit/build.py"
    r, calls = _standin_run(tmp_path, build, "--fbin", tmp_path / "base.fvecs", tmp_path / "base.fbin")
    assert r.returncode == 0, r.stderr
    assert [(c["call"], c["shape"], c["dtype"]) for c in calls] == [("vectors_to_file", [10, 4], "float32")]
    b = (tmp_path / "base.fbin").read_bytes()
    assert struct.unpack("<II", b[:8]) == (10, 4) and np.frombuffer(b[8:], dtype="float32").reshape(10, 4).tolist() == rows.tolist()
    for extra, threads in (([], os.cpu_count()), (["--threads", "20"], 20)):
        r, calls = _standin_run(tmp_path, build, tmp_path / "base.fbin", tmp_path / "index", *extra)
        assert r.returncode == 0, r.stderr
        (c,) = calls
        assert (tmp_path / "index").is_dir() and c["call"] == "build_disk_index"
        assert (c["data"], c["index_directory"], c["distance_metric"], c["num_threads"]) == (
            str(tmp_path / "base.fbin"), str(tmp_path / "index"), "l2", threads)
        assert f"degree {c['graph_degree']}," in doc and f"build list {c['complexity']}," in doc
        assert f"(build memory {c['build_memory_maximum']} GB)" in doc
        assert c["search_memory_maximum"] == 0.03 and "float32" in c["vector_dtype"]


@pytest.fixture
def standins_imported(monkeypatch, tmp_path):
    """The stand-ins importable in this process for one test (their log in `tmp_path`), and gone
    from `sys.modules` after it."""
    before = set(sys.modules)
    monkeypatch.syspath_prepend(str(STANDINS))
    monkeypatch.setenv("AEIOU_STANDIN_LOG", str(tmp_path / "standin.log"))
    yield tmp_path / "standin.log"
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_diskann_search_kit_marks_every_query_and_searches_as_the_case_says(monkeypatch, tmp_path, standins_imported):
    """`diskann-search/kit/search.py` as its README runs it (in this process, `os.stat` watched):
    the case's beam 4 and search list 100, the node cache and its mechanism passed through, and
    without `--batch` a `stat` of hops.py's mark before every query; `--batch` is one call over the
    threads and no mark."""
    import runpy
    np = pytest.importorskip("numpy")
    import standin
    case = _case("vector-search/diskann-search/cases/sift1m-cache")
    assert "search list 100, beam 4," in case["doc"] and values_of(case)["beam"] == 4
    q = tmp_path / "query.fbin"
    q.write_bytes(struct.pack("<II", 5, 4) + np.arange(20, dtype="float32").tobytes())
    stat = os.stat

    def watched(path, *a, **kw):
        if path == "/aeiou-query-mark":
            standin.record("stat", path=path)
        return stat(path, *a, **kw)
    monkeypatch.setattr(os, "stat", watched)
    script = str(WL / "vector-search/diskann-search/kit/search.py")
    for extra in (["--cache", "10000", "--cache-mechanism", "2"], ["--batch"]):
        standins_imported.unlink(missing_ok=True)
        monkeypatch.setattr(sys, "argv", [script, str(tmp_path / "index"), str(q), "--queries", "3", "--threads", "2", *extra])
        runpy.run_path(script, run_name="__main__")
        calls = [json.loads(line) for line in standins_imported.read_text().splitlines()]
        assert [c["call"] for c in calls[:2]] == ["vectors_from_file", "StaticDiskIndex"]
        ix = calls[1]
        assert (ix["index_directory"], ix["num_threads"], ix["distance_metric"], ix["dimensions"]) == (str(tmp_path / "index"), 2, "l2", 4)
        assert "float32" in ix["vector_dtype"]
        if extra[0] == "--batch":
            assert (ix["num_nodes_to_cache"], ix["cache_mechanism"]) == (0, 0)
            assert [(c["call"], c["shape"], c["k"], c["complexity"], c["threads"], c["beam"]) for c in calls[2:]] == [
                ("batch_search", [3, 4], 10, 100, 2, 4)]
        else:
            assert (ix["num_nodes_to_cache"], ix["cache_mechanism"]) == (10000, 2)
            assert [c["call"] for c in calls[2:]] == ["stat", "search"] * 3
            assert {(c["dim"], c["k"], c["complexity"], c["beam"]) for c in calls[3::2]} == {(4, 10, 100, 4)}
