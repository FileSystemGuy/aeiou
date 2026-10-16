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
    assert params["params"]["node_cache"] == 59
    assert params["params"]["hops"]["empirical"] == {"values": [2, 3, 4], "weights": [1, 1, 1]}


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
    p = json.loads(_hops(trace, index, "--params", "doc").stdout)["params"]
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
    assert f["params"] == {"dim": 16, "n": 1000, "shards": 3, "pq_code": 32, "shard_index_bytes": 2200 // 3, "index_bytes": 4204,
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
                   (1, 3, 4, True, 0, 0, 0, 1)]                                        # fi 2 is over the context
    assert shared == [128]
    assert [r["think"] for r in rows] == [0.5] * 5 + [1.0, None, 2.0]
    assert [r["fi"] for r in ax.walk(corpus, 1, 128, 4096)[0]] == [0, 1, 2, 3, 4]       # --traces 1: session 0 only
    assert [(r["ti"], r["fi"]) for r in ax.walk(corpus, 10 ** 9, 128, 4096, {(0, 1), (1, 3)}, 4)[0]] == [(0, 1), (1, 3)]   # a replay's


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
    p = f["params"]
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
    assert ref["chunks_stored"] == {"total": 5, "mean": 0.62, "shares": [0, 1]}
    assert ref["chunks_hit"] == {"total": 8, "mean": 1.0, "shares": [0, 2]}
    assert ref["out_tokens"] == {"total": 63, "mean": 7.9}
    assert ref["subagent_shared_prefix_tokens"] == {"median": 128, "n": 1}
    assert "requests 8, chains 5, chunks stored 5 (0.62 per request), hit 8" in r.stdout


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
    p = f["params"]
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
    assert f["doc"] == "mine" and f["params"]["think"]["empirical"]["values"] == [1000000000, 2333333333]


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
    p = json.loads(out.read_text())["params"]
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
    p = f["params"]
    assert (f["abstract"], f["doc"], p["lists"], p["code_size"], p["prefetch"], p["index_bytes"]) == ("faiss_ivf_ondisk", "d", 8, 32, 32, 680212)
    assert p["list_bytes"] == {"lognormal": {"median": 2000, "sigma": 0.52, "min": 0, "max": 4000}}
    assert p["list_pop"] == {"zipf": {"s": 2.0}} and p["index_reads"] == [8192, 516096]
    assert fp.params(argparse.Namespace(doc="d", threads=1, calls=20, batch=1, nprobe=16), size, probes, 32, 32, [], 1)["params"]["prefetch"] == 16
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
    assert dp.params(m, 777, 200, "d") == {"params_version": 1, "abstract": "dcp_load", "doc": "d", "params": {
        "items": 3, "item_bytes": [32, 8, 16], "item_off": [0, 50, 100], "read_order": [0, 2, 1], "meta_bytes": 777, "restore_step": 200}}
