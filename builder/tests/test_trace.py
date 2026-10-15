"""`aeiou-trace`: the metrics of an `strace` against their definitions (`runner/REFERENCE.md`
§10) on hand-written traces, the ports of the runner's histogram and stack distance, the
comparison's distances, and, when `strace` and the runner binary are there, the trace of
`aeiou run` against the same abstract's `aeiou dry-run --metrics-json`: everything that
does not depend on the order must be equal."""
import importlib.util
import json, re
import os
import pathlib
import random
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
sys.path.insert(0, str(BUILDER))

from aeiou import catalog, trace  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402
from aeiou.trace import Instance, LogHist, compare, metrics_of, split_args  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
WORKLOADS = ROOT / "workloads"


def ast_path(name):
    """The committed AST `name`, wherever its mechanism lives under workloads/ (DESIGN_REVIEW.md §3.66)."""
    return next(WORKLOADS.rglob(f"{name}.ast.json"))


def case_dir(name, case):
    """A case of the mechanism whose AST is `name`: its parameter set and its captures."""
    return ast_path(name).parent / "cases" / case


def total(text, **kw):
    kw.setdefault("roots", ["/d"])
    return metrics_of(text.strip().splitlines(), **kw)


def test_loghist_is_the_runners():
    for v in [0, 1, 2, 3, 4, 5, 7, 8, 4096, 4097, 5000, 6143, 6144, 1 << 40, (1 << 64) - 1]:
        lo = LogHist.lower(v)
        assert lo <= v and (v < 4 or (v - lo) <= 0.25 * lo), v
    assert LogHist.lower(4096) == 4096 and LogHist.lower(6144) == 6144 and LogHist.lower(5000) == 4096
    h = LogHist()
    for v in range(1, 101):
        h.add(v * 4096)
    assert trace._quantile(h.out(), 0.5) == 48 * 4096 and h.max == 100 * 4096


def test_stack_distance_matches_brute_force():
    rnd = random.Random(1)
    inst, history = Instance(), []
    for i in range(6000):
        x = rnd.random()
        key = rnd.randrange(16) if x < 0.4 else 100 + rnd.randrange(600) if x < 0.9 else 10_000 + i
        want = None
        if key in history:
            p = len(history) - 1 - history[::-1].index(key)
            want = len(set(history[p:]))
        got = inst.touch(key, 0)
        assert (got[0] if got else None) == want, i
        history.append(key)


def test_split_args():
    assert split_args('3</d/a, b>, "x, \\"y\\"", 10') == ["3</d/a, b>", '"x, \\"y\\""', "10"]
    assert split_args('AT_FDCWD</cwd>, "p", O_RDONLY|O_CLOEXEC') == ["AT_FDCWD</cwd>", '"p"', "O_RDONLY|O_CLOEXEC"]
    assert split_args('5, [{iov_base="a,b", iov_len=3}, {iov_base="", iov_len=5}], 2') == ["5", '[{iov_base="a,b", iov_len=3}, {iov_base="", iov_len=5}]', "2"]
    assert trace.fd_arg("0</dev/null<char 1:3>>") == (0, "/dev/null")
    assert trace.c_string('"caf\\303\\251"') == "café"


SEQUENTIAL = """
100 1700000001.000000 openat(AT_FDCWD</cwd>, "/d/a", O_RDONLY|O_CLOEXEC) = 3</d/a> <0.000010>
100 1700000001.000100 fstat(3</d/a>, {st_mode=S_IFREG|0644, st_size=10000, ...}) = 0 <0.000010>
100 1700000001.000200 read(3</d/a>, "x"..., 4096) = 4096 <0.000010>
100 1700000001.000300 read(3</d/a>,  <unfinished ...>
101 1700000001.000350 openat(AT_FDCWD</cwd>, "/etc/hosts", O_RDONLY) = 4</etc/hosts> <0.000010>
101 1700000001.000360 read(4</etc/hosts>, "y"..., 4096) = 100 <0.000010>
100 1700000001.000400 <... read resumed>"x"..., 4096) = 4096 <0.000100>
100 1700000001.000500 read(3</d/a>, "x"..., 4096) = 1808 <0.000010>
100 1700000001.000600 read(3</d/a>, "", 4096) = 0 <0.000010>
100 1700000001.000700 lseek(3</d/a>, 0, SEEK_SET) = 0 <0.000010>
100 1700000001.000800 read(3</d/a>, "x"..., 100) = 100 <0.000010>
100 1700000001.000900 pread64(3</d/a>, "x"..., 200, 8192) = 200 <0.000010>
100 1700000001.001000 close(3</d/a>) = 0 <0.000010>
100 1700000001.001100 stat("/d/missing", 0x7ffc) = -1 ENOENT (No such file or directory) <0.000010>
100 1700000001.001200 +++ exited with 0 +++
"""


def test_positions_runs_and_reuse():
    d = total(SEQUENTIAL)
    t = d["total"]
    assert d["aeiou_metrics"] == 1 and d["source"] == "strace" and d["order"] == "completion"
    # only /d counts; the failed stat is an op, as an expected error is in the abstract
    assert t["counts"] == {"close": 1, "fstat": 1, "lseek": 1, "open": 1, "read": 6, "stat": 1}
    assert t["ops"] == 11 and t["bytes_read"] == 4096 + 4096 + 1808 + 100 + 200
    # request sizes are what was asked for, the EOF read included
    assert t["request_size"]["read"]["buckets"] == [[96, 1], [192, 1], [4096, 4]]
    # runs: 0..10000 in three ops, then the re-read at 0, then the pread at 8192
    assert sorted(map(tuple, t["run_length"]["read"]["bytes"]["buckets"])) == [(96, 1), (192, 1), (8192, 1)]
    assert t["run_length"]["read"]["ops"]["buckets"] == [[1, 2], [3, 1]]
    assert t["run_length"]["read"]["multi_op_bytes"] == 10000
    # blocks 0, 1, 2 first; block 0 again after three distinct blocks, block 2 after two
    r = t["reuse_distance_bytes"]
    assert r["first_touch"] == {"read": 3, "write": 0}
    assert r["read_after_read"]["buckets"] == [[2 * 4096, 1], [3 * 4096, 1]]
    assert t["popularity_blocks"]["counts"] == [[2, 2], [1, 1]] and t["popularity_objects"]["counts"] == [[6, 1]]
    # tid 101 appears with no clone in the trace: noted, and harmless here
    assert d["notes"] == {"threads_without_clone": 1}


THREADS = """
200 1700000002.000000 openat(AT_FDCWD</d>, "w", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3</d/w> <0.000010>
200 1700000002.000100 clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD, child_tid=0x7f, stack=0x7f, stack_size=0x1000}, 88 <unfinished ...>
201 2.000200 write(3</d/w>, "a"..., 4096) = 4096 <0.000010>
200 1700000002.000300 <... clone3 resumed> => {parent_tid=[201]}, 88) = 201 <0.000200>
200 1700000002.000400 write(3</d/w>, "b"..., 4096) = 4096 <0.000010>
201 2.000500 write(3</d/w>, "c"..., 4096) = 4096 <0.000010>
200 1700000002.000600 dup(3</d/w>) = 4</d/w> <0.000010>
200 1700000002.000700 write(4</d/w>, "d"..., 4096) = 4096 <0.000010>
200 1700000002.000800 fsync(4</d/w>) = 0 <0.000010>
200 1700000002.000900 pread64(3</d/w>, "a"..., 4096, 0) = 4096 <0.000010>
200 1700000002.001000 mkdir("sub", 0755) = 0 <0.000010>
200 1700000002.001100 chdir("/d/sub") = 0 <0.000010>
200 1700000002.001200 unlinkat(AT_FDCWD</d/sub>, "../w", 0) = 0 <0.000010>
200 1700000002.001300 openat(AT_FDCWD</d/sub>, "/d", O_RDONLY|O_DIRECTORY) = 5</d> <0.000010>
200 1700000002.001400 getdents64(5</d>, 0x55 /* 3 entries */, 32768) = 80 <0.000010>
200 1700000002.001500 getdents64(5</d>, 0x55 /* 0 entries */, 32768) = 0 <0.000010>
200 1700000002.001600 close(5</d>) = 0 <0.000010>
"""


def test_threads_share_positions_and_contexts_own_runs():
    d = total(THREADS, cwd="/d")
    t = d["total"]
    assert t["counts"] == {"close": 1, "fsync": 1, "mkdir": 1, "open": 2, "read": 1, "readdir": 1, "unlink": 1, "write": 4}
    assert t["bytes_written"] == 4 * 4096 and d["contexts"] == 2
    # one shared position: the four writes are blocks 0..3, each a first touch
    r = t["reuse_distance_bytes"]
    assert r["first_touch"] == {"read": 0, "write": 4}
    # the read of block 0 comes four distinct blocks after its write
    assert r["read_after_write"]["buckets"] == [[4 * 4096, 1]]
    # a run belongs to the thread that issues it: 201 wrote 0 and 2 (two runs), 200 wrote 1 and 3
    assert t["run_length"]["write"]["ops"]["buckets"] == [[1, 4]]


AIO = """
300 1700000003.000000 openat(AT_FDCWD</d>, "idx", O_RDONLY|O_DIRECT) = 3</d/idx> <0.000010>
300 1700000003.000100 io_submit(0x7f00, 2, [{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=0}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=40960}]) = 2 <0.000010>
300 1700000003.000200 io_getevents(0x7f00, 2, 2, [{data=0x2, obj=0x7f10, res=4096, res2=0}, {data=0x1, obj=0x7f20, res=4096, res2=0}], NULL) = 2 <0.000050>
300 1700000003.000300 io_submit(0x7f00, 2, [{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=8192}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=0}]) = 2 <0.000010>
300 1700000003.000400 io_getevents(0x7f00, 2, 2, [{data=0x1, obj=0x7f10, res=4096, res2=0}, {data=0x2, obj=0x7f20, res=100, res2=0}], NULL) = 2 <0.000050>
300 1700000003.010000 io_submit(0x7f00, 1, [{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=4096}]) = 1 <0.000010>
300 1700000003.010100 io_submit(0x7f00, 1, [{aio_data=0x9, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=4096}]) = 1 <0.000010>
"""


def test_io_submit_is_fan_out_and_chains_need_a_gap():
    d = total(AIO)
    t = d["total"]
    assert t["fan_out"] == {"1": 2, "2": 2} and t["depth"] == {}
    assert "depth_not_computed" in d["notes"]
    # results come from io_getevents (the short one too); the two never reaped are taken as complete
    assert t["counts"]["read"] == 6 and t["bytes_read"] == 3 * 4096 + 100 + 2 * 4096
    assert d["notes"]["aio_results_assumed_complete"] == 2
    # 9.6 ms between the second round's end and the third submit: a 1 ms gap cuts there
    assert total(AIO, chain_gap_us=1000)["total"]["depth"] == {"2": 2}
    assert total(AIO, chain_gap_us=50_000)["total"]["depth"] == {"4": 1}


def test_an_aio_context_no_io_submit_names_is_noted():
    # strace classes no AIO call in `%desc`: a trace taken with `-e trace=%file,%desc,…` and
    # `io_setup` but not `io_submit` shows the context made and none of its requests
    setup = '300 1700000002.900000 io_setup(64, [0x7f00]) = 0 <0.000100>\n'
    lost = setup + "\n".join(l for l in AIO.strip().splitlines() if "io_submit" not in l and "io_getevents" not in l)
    d = total(lost)
    assert d["total"]["counts"] == {"open": 1}
    assert d["notes"] == {"aio_contexts_without_io_submit": 1}
    header, _ = trace.export_of(lost.strip().splitlines(), "/d")
    assert header["notes"]["aio_contexts_without_io_submit"] == 1
    # the context used: no note; one made outside every instance: none either
    assert "aio_contexts_without_io_submit" not in total(setup + AIO)["notes"]
    assert "aio_contexts_without_io_submit" not in total(lost, instance_roots=[999])["notes"]


FORKED = """
400 1700000004.000000 clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7f) = 401 <0.000100>
400 1700000004.000100 clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7f) = 402 <0.000100>
401 1700000004.000200 openat(AT_FDCWD</d>, "a", O_RDONLY) = 3</d/a> <0.000010>
402 1700000004.000300 openat(AT_FDCWD</d>, "a", O_RDONLY) = 3</d/a> <0.000010>
401 1700000004.000400 read(3</d/a>, "x"..., 4096) = 4096 <0.000010>
402 1700000004.000500 read(3</d/a>, "x"..., 4096) = 4096 <0.000010>
401 1700000004.000600 read(3</d/a>, "x"..., 4096) = 4096 <0.000010>
400 1700000004.000700 openat(AT_FDCWD</d>, "a", O_RDONLY) = 3</d/a> <0.000010>
"""


def test_instances_are_process_trees():
    one = total(FORKED)["total"]
    # one instance: 402's read of block 0 re-reads 401's
    assert one["reuse_distance_bytes"]["first_touch"]["read"] == 2 and one["reuse_distance_bytes"]["read_after_read"]["n"] == 1
    d = total(FORKED, instance_roots=[401, 402])
    two = d["total"]
    # two instances: no reuse within either; the sharing shows as popularity; the parent is outside
    assert d["instances"] == 2 and two["counts"] == {"open": 2, "read": 3}
    assert two["reuse_distance_bytes"]["first_touch"]["read"] == 3 and two["reuse_distance_bytes"]["read_after_read"]["n"] == 0
    assert two["popularity_blocks"]["counts"] == [[2, 1], [1, 1]]
    assert d["notes"]["calls_outside_instances"] == 1


def test_exclude_sampling_and_other_roots():
    lines = ["1 openat(AT_FDCWD, \"/d/.aeiou-dataset.json\", O_RDONLY) = 3</d/.aeiou-dataset.json>", "1 read(3</d/.aeiou-dataset.json>, \"\"..., 100) = 100"]
    for i in range(400):
        lines.append(f'1 pread64(4</d/big>, ""..., 4096, {4096 * (i % 100)}) = 4096')
    d = metrics_of(lines, ["/d"], exclude=[".aeiou*"])
    assert d["total"]["counts"] == {"read": 400} and d["notes"] == {"descriptors_without_open": 1}
    exact = d["total"]["reuse_distance_bytes"]
    assert exact["first_touch"]["read"] == 100 and exact["read_after_read"]["buckets"] == [[LogHist.lower(100 * 4096), 300]]
    s = metrics_of(lines, ["/d"], sample=4, exclude=[".aeiou*"])["total"]["reuse_distance_bytes"]
    # one block in four by hash, scaled: about the same counts, distances within a bucket or two
    assert 40 <= s["first_touch"]["read"] <= 180 and s["read_after_read"]["n"] == 3 * s["first_touch"]["read"]
    assert metrics_of(lines, ["/elsewhere"])["total"]["ops"] == 0


def test_compare_distances(tmp_path, capsys):
    a = total(SEQUENTIAL)
    same = compare(a["total"], a["total"])
    assert same and all(r[3] in (0.0, None) for r in same)
    b = total(SEQUENTIAL.replace("4096) = 4096", "8192) = 4096"))
    rows = {r[0]: r for r in compare(a["total"], b["total"])}
    # two of six requests moved from the 4096 bucket to the 8192 one
    assert rows["request size, read (p50 / p90 / p99)"][3] == pytest.approx(2 / 6)
    assert rows["mix: reads / data ops"][3] == 0.0
    assert trace._tv_distance({4: 3, 2: 1}, {4: 1, 2: 1}) == pytest.approx(0.25)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert trace.main(["compare", str(pa), str(pb)]) == 0
    assert trace.main(["compare", str(pa), str(pb), "--max-distance", "0.1", "--only", "request size"]) == 1
    assert "largest distance 0.333" in capsys.readouterr().out


def _strace_works():
    if not shutil.which("strace"):
        return False
    return subprocess.run(["strace", "-f", "-o", os.devnull, "true"], capture_output=True).returncode == 0


needs_run = pytest.mark.skipif(not RUNNER.exists() or not _strace_works(), reason="needs the runner binary and a working strace")


@needs_run
@pytest.mark.parametrize(
    "name, seed, params",
    [
        ("file_per_sample", 7, ["samples=600", "batch=4", "workers=2", "prefetch=2", "steps=12"]),
        ("lmcache_local_disk", 5, ["sys_prompts=3", "sys_tokens=6", "chunk_bytes=262144", "concurrency=3", "warm=4", "requests=8",
                              'reuse={"mixture": [{"weight": 0.3, "dist": null}, {"weight": 0.7, "dist": {"const": 2}}]}',
                              'keep={"empirical": {"values": [0, 50, 100], "weights": [1, 1, 1]}}']),
        ("diskann_search", 9, ["nodes=5000", "threads=2", "queries=6"]),
    ],
)
def test_trace_of_the_runner_matches_its_dry_run(tmp_path, name, seed, params):
    ast = ast_path(name)
    root = tmp_path / "root"
    root.mkdir()
    common = [str(ast), "--gpus", "1", *[x for p in params for x in ("--param", p)]]
    run = lambda *a: subprocess.run([*map(str, a)], capture_output=True, text=True)
    r = run(RUNNER, "datagen", *common, "--root", root)
    assert r.returncode == 0, r.stderr
    dry = tmp_path / "dry.json"
    r = run(RUNNER, "dry-run", *common, "--seed", seed, "--metrics-json", dry)
    assert r.returncode == 0, r.stderr
    st = tmp_path / "strace.txt"
    r = run("strace", "-f", "-ttt", "-T", "-yy", "-e", "trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_destroy", "-o", st, RUNNER, "run", *common, "--seed", seed, "--root", root, "--time-scale", "0")
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "trace.json"
    assert trace.main(["metrics", str(st), "--root", str(root), "--exclude", "*.aeiou-*", "-o", str(out)]) == 0
    t, d = json.loads(out.read_text())["total"], json.loads(dry.read_text())
    assert d["source"] == "dry-run"
    d = d["total"]
    # the op multiset of the data path, and everything that is a sum over it
    for kind in ("read", "write"):
        assert t["counts"].get(kind, 0) == d["counts"].get(kind, 0), kind
    assert (t["bytes_read"], t["bytes_written"]) == (d["bytes_read"], d["bytes_written"])
    assert t["request_size"] == d["request_size"]
    for unit in ("popularity_blocks", "popularity_objects"):
        assert t[unit]["counts"] == [list(c) for c in d[unit]["counts"]], unit
    first = lambda m: sum(m["reuse_distance_bytes"]["first_touch"].values())
    assert first(t) == first(d)
    # runs belong to a context: a thread there, a sub-actor here, the same ops either way. A pool
    # thread runs sub-actor k of every fork in turn (runner/REFERENCE.md §4), so when a fork's last
    # read and the next fork's first are adjacent in one file and no other thread's call lands
    # between them (timing: 1 in 3 to 1 in 40 under CPU contention, 2026-10-07), the strace sees
    # one run where the dry run sees two. The ops and bytes in runs are the same either way.
    for kind, rd in d["run_length"].items():
        rt = t["run_length"][kind]
        total = lambda h: round(h["n"] * h["mean"])
        assert (total(rt["ops"]), total(rt["bytes"])) == (total(rd["ops"]), total(rd["bytes"])), kind
        assert 0 <= rd["ops"]["n"] - rt["ops"]["n"] <= 2, (kind, rt, rd)
    # what a block's previous access was depends on the order (the real one against the
    # round-robin): the shares are close, not equal. The distance histograms are not bounded
    # here: on these configurations a kind has a few dozen samples, and on a host with two
    # cores the real interleaving of three sub-actors put one of them at 0.6 (CI, 2026-10-01).
    rows = [r for r in compare(t, d) if r[0].startswith("reuse") and not r[0].startswith("reuse distance")]
    assert rows and all(r[3] is None or r[3] <= 0.5 for r in rows), rows

    # the `trace` node (DESIGN_REVIEW.md §3.58): the strace exported as a trace file, an
    # abstract that is that one node, and the runner's dry run of it in line order must
    # compute the strace's own metrics; then the trace runs against the same corpus with
    # the fingerprint of its dry run. Under libaio the strace's metrics count an io_submit
    # member when it is reaped and the file counts it at submission, so the reuse-distance
    # histograms of that pair differ by the reordering within a round, and so, now and then,
    # does a run (two adjacent sector reads); all else is equal.
    exported = tmp_path / "exported.jsonl"
    r = subprocess.run([sys.executable, "-m", "aeiou.trace", "export", str(st), "--root", str(root), "--exclude", "*.aeiou-*", "-o", str(exported)], capture_output=True, text=True, cwd=ROOT / "builder")
    assert r.returncode == 0, r.stderr
    sha = next(line.split()[1] for line in r.stdout.splitlines() if line.startswith("sha256 "))
    node_ast = tmp_path / "node.ast.json"
    node_ast.write_text(json.dumps({"ast": "0.10", "name": f"trace_{name}", "doc": "the exported trace", "params": {}, "datasets": {},
                                    "actors": {"app": {"count": 1, "body": [{"trace": {"file": exported.name, "sha256": sha}}]}}}))
    node_dry = tmp_path / "node.dry.json"
    r = run(RUNNER, "dry-run", node_ast, "--gpus", "1", "--metrics-json", node_dry)
    assert r.returncode == 0, r.stderr
    nd = json.loads(node_dry.read_text())["total"]
    assert trace.main(["metrics", str(exported), "-o", str(tmp_path / "ex.json")]) == 0
    ex = json.loads((tmp_path / "ex.json").read_text())["total"]
    assert nd == ex, "the runner's walk of the file and the Python reader's disagree"
    strip = lambda m: {k: v for k, v in m.items() if k not in ("reuse_distance_bytes", "run_length")} if name == "diskann_search" else m
    assert strip(t) == strip(nd), "the trace node's dry run is not the strace's metrics"
    fp = next(line.split()[1] for line in r.stdout.splitlines() if line.startswith("fingerprint "))
    # one thread per lane under sync; one task per lane on an event loop, a group's members
    # in flight together, under io_uring and libaio: the same run either way
    for backend in ("sync", "io_uring", "libaio"):
        r = run(RUNNER, "run", node_ast, "--gpus", "1", "--root", root, "--time-scale", "0", "--clean-namespaces", "--posix", backend)
        assert r.returncode == 0, backend + "\n" + r.stdout + r.stderr
        assert f"fingerprint {fp}" in r.stdout, backend + "\n" + r.stdout
        assert "never CLOSED" in r.stdout


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_small_file_abstract_matches_the_trace_of_the_real_loader(tmp_path):
    """`workloads/input-pipeline/map/file-per-sample`: DataLoader + ImageFolder on NFS, traced 2026-10-01.
    The abstract at the fitted parameters has the application's op mix exactly but for the
    one listing of the dataset root; the corpus of real JPEGs is near the abstract's size
    distribution, not equal to it, and the order differs."""
    case = case_dir("file_per_sample", "imagefolder-jpeg")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("file_per_sample")), "--gpus", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-01/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    root_listing = {"open": 1, "fstat": 1, "readdir": 1, "close": 1}
    assert {k: v - root_listing.get(k, 0) for k, v in t["counts"].items()} == d["counts"]
    assert t["request_size"]["read"]["buckets"] == d["request_size"]["read"]["buckets"]
    rows = compare(t, d)
    assert max(r[3] for r in rows if r[3] is not None) <= 0.06, rows

    assert subprocess.run([sys.executable, "-m", "aeiou.trace", "compare", str(tmp_path / "absent.json"), str(dry)],
                          capture_output=True, text=True, cwd=BUILDER).stderr.startswith("aeiou-trace compare: ")


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("which, reuse", [("140MiB", 0.3), ("8MiB", 0.07)])
def test_large_sample_abstract_matches_the_trace_of_np_load(tmp_path, which, reuse):
    """`workloads/input-pipeline/map/npy-per-sample`: `np.load(...)["x"]` on NFS, traced 2026-10-01.
    The abstract draws its sizes, the corpus has its own, so the counts are near, not equal
    (they are exact per file: `ABSTRACTS.md` §2). Sixteen files say little about the reuse
    distance; 256 say more."""
    case = case_dir("npy_per_sample", f"npy-{which.lower()}")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("npy_per_sample")), "--gpus", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-01/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    listing = {"open": 1, "fstat": 1, "readdir": 1, "close": 1}       # glob lists the root too; the abstract lists the sample directories (one here)
    for k, v in t["counts"].items():
        v -= listing.get(k, 0)
        assert abs(v - d["counts"].get(k, 0)) <= 0.01 * v, k
    for name, _, _, dist in compare(t, d):
        if dist is not None:
            assert dist <= (reuse if name.startswith("reuse distance") else 0.01), name


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_checkpoint_write_abstract_matches_the_trace_of_dcp_save(tmp_path):
    """`workloads/checkpointing/dcp-save`: `torch.distributed.checkpoint.save` on two ranks,
    traced 2026-10-01. Every call on the shard files and `.metadata` is in the abstract; the
    path checks around `mkdir` that it cannot express are listed here (`ABSTRACTS.md` §3)."""
    case = case_dir("dcp_save", "toy-6-items")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("dcp_save")), "--gpus", "2",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-01/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    not_modeled = {
        "mkdir": 2,   # once in the job: the step directory's parent is absent (ENOENT), then made
        "stat": 8,    # per rank and checkpoint the parent (4); the rank that lost the mkdir, the directory (2); the root, first time (2)
    }
    assert {k: v - not_modeled.get(k, 0) for k, v in t["counts"].items()} == d["counts"]
    assert t["bytes_written"] == d["bytes_written"]
    assert t["request_size"]["write"]["buckets"] == d["request_size"]["write"]["buckets"]
    assert max(r[3] for r in compare(t, d) if r[3] is not None) <= 0.05


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("which, bytes_off", [("mixed-9-items", 0), ("small-last-6-items", 88), ("name-order-17-items", 0)])
def test_checkpoint_restore_abstract_matches_the_trace_of_dcp_load(tmp_path, which, bytes_off):
    """`workloads/checkpointing/dcp-load`: `torch.distributed.checkpoint.load` on two ranks, traced
    2026-10-01. Every call on the shard files and `.metadata` is in the abstract, with its
    count; the check of the checkpoint's parent directory is not (`ABSTRACTS.md` §4). A small
    last item has its zip records 44 bytes nearer its ends than `rec2` and `tail_back` say,
    which is 88 bytes over two ranks. `name-order` has 17 items, read in the sorted order of
    their names across runs of small ones longer than the buffer: the buffer chain."""
    case = case_dir("dcp_load", which)
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("dcp_load")), "--gpus", "2",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-01/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    not_modeled = {"stat": 2}   # per rank, the parent of the checkpoint directory (and an `access`, which the trace metrics do not count)
    assert {k: v - not_modeled.get(k, 0) for k, v in t["counts"].items() if v != not_modeled.get(k)} == d["counts"]
    assert d["bytes_read"] - t["bytes_read"] == bytes_off
    assert t["request_size"]["read"]["buckets"] == d["request_size"]["read"]["buckets"]
    assert max(r[3] for r in compare(t, d) if r[3] is not None) <= 0.05


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_model_load_abstract_matches_the_trace_of_from_pretrained(tmp_path):
    """`workloads/model-loading/safetensors-load`: `from_pretrained` on five safetensors shards, traced
    2026-10-01. The library maps the shards and issues no `read` on them, so the trace has the
    calls around the mappings and the small JSON files, and nothing of the tensor bytes
    (`ABSTRACTS.md` §4)."""
    case = case_dir("safetensors_load", "gpt2-random")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("safetensors_load")), "--gpus", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-01/trace.metrics.json").read_text())["total"]["counts"], json.loads(dry.read_text())["total"]["counts"]
    for op in ("open", "close", "fstat", "ioctl", "lseek", "fadvise"):
        assert t[op] == d[op], op
    assert d["read"] - t["read"] == 2 * 5 + 76   # through the mappings, unseen by strace: 8 bytes and the header per shard, 76 tensors
    assert t["stat"] - d["stat"] == 25           # not modeled: directories (the model's 12 times, the two above each shard), a second stat of two JSON files, the probe for an unsharded model.safetensors


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_ivf_abstract_matches_the_trace_of_faiss_search(tmp_path):
    """`workloads/vector-search/faiss-ivf-ondisk`: `faiss.read_index` and 20 searches over
    `OnDiskInvertedLists`, traced 2026-10-01. FAISS maps the lists file and issues no call on
    it, so the trace has the index file's reads and the two opens, and nothing of the lists
    (`ABSTRACTS.md` §6)."""
    case = case_dir("faiss_ivf_ondisk", "sift1m-ivf1024-pq32")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("faiss_ivf_ondisk")), "--gpus", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-01/trace.metrics.json").read_text())["total"]["counts"], json.loads(dry.read_text())["total"]["counts"]
    assert {k: v for k, v in d.items() if k != "read"} == {k: v for k, v in t.items() if k != "read"}
    assert d["read"] - t["read"] == 20 * 64 * 2   # through the mapping, unseen by strace: ids and codes of 64 lists per query


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("which, reads_off", [("nocache", 0.02), ("cache", 0.04)])
def test_diskann_search_abstract_matches_the_trace(tmp_path, which, reads_off):
    """`workloads/vector-search/diskann-search`: 1,000 queries through DiskANN's `PQFlashIndex`,
    traced 2026-10-02 without a node cache and with 10,000 nodes cached. The abstract issues
    full rounds of `beam` reads where the search skips nodes it already holds, so it reads a
    little more; the load's small files (pivots, medoids, centroids, metadata) are not
    modeled (`ABSTRACTS.md` §5)."""
    case = case_dir("diskann_search", f"sift1m-{which}")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("diskann_search")), "--gpus", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-02/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    tc, dc = t["counts"], d["counts"]
    assert 0 <= (dc["read"] - tc["read"]) / tc["read"] <= reads_off
    small = {"open": 8, "close": 9, "lseek": 28, "stat": 9, "fstat": 3, "ioctl": 1}   # the unmodeled small files
    assert {k: tc[k] - dc.get(k, 0) for k in small} == small
    share = lambda m: m["fan_out"]["4"] / sum(m["fan_out"].values())
    assert share(t) > 0.85 and share(d) > 0.85
    rows = {r[0]: r[3] for r in compare(t, d)}
    assert rows["request size, read (p50 / p90 / p99)"] <= 0.001
    assert rows["popularity, blocks: accesses to the top 1 %"] <= 0.01
    assert rows["reuse: first touches / block reads"] <= 0.06


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_diskann_build_abstract_matches_the_trace(tmp_path):
    """`workloads/vector-search/diskann-build`: `build_disk_index` on SIFT1M in 13 shards, traced
    2026-10-02. The data path agrees to a part in a thousand; the shards are modeled at one
    size and the files' small headers and probes are not modeled (`ABSTRACTS.md` §7)."""
    case = case_dir("diskann_build", "sift1m-13-shards")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("diskann_build")), "--gpus", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-02/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    for k in ("read", "write", "lseek"):
        assert abs(t["counts"][k] - d["counts"][k]) / t["counts"][k] < 0.001, k
    for k in ("bytes_read", "bytes_written"):
        assert abs(t[k] - d[k]) / t[k] < 0.001, k
    rows = {r[0]: r[3] for r in compare(t, d)}
    for k in ("request size, read (p50 / p90 / p99)", "request size, write (p50 / p90 / p99)",
              "run length, read: bytes in multi-op runs", "reuse: read after write / block reads"):
        assert rows[k] <= 0.01, (k, rows[k])
    # 13 unlinks of files already gone and one of a file this abstract does not write
    assert t["counts"]["unlink"] - d["counts"]["unlink"] == 14


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_kv_cache_abstract_matches_the_trace_of_vllm_with_lmcache(tmp_path):
    """`workloads/kvcache/lmcache-local-disk`: vLLM with LMCache's local-disk backend under a replay
    of ShareGPT, 300 requests with 8 conversations open, traced 2026-10-02. Every chunk is the
    same six calls around one `read` or `write`. At the parameters `fit.py` takes from the
    run's two logs the abstract stores 382 chunks where the engine stored 370 (two of them
    the system prompts', a dataset here) and reads 596 where it read 755 (seed 1; over eight
    seeds it reads 587 to 781: one `keep` draw per round of conversations, §3.57, makes the
    loads come in bursts, so the seeds spread wider than with a draw per request)
    (`ABSTRACTS.md` §8)."""
    case = case_dir("lmcache_local_disk", "sharegpt-chat")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("lmcache_local_disk")), "--gpus", "1", "--seed", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/2026-10-02/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    tc, dc = t["counts"], d["counts"]
    for c in (tc, dc):                                    # one open, fstat, ioctl, lseek and close per data call
        assert c["open"] == c["fstat"] == c["ioctl"] == c["lseek"] == c["close"] == c["read"] + c["write"]
    assert (tc["write"], dc["write"]) == (370, 382)
    assert (tc["read"], dc["read"]) == (755, 596)
    assert t["request_size"] == {k: {**v, "n": t["request_size"][k]["n"], "buckets": [[3145728, t["request_size"][k]["n"]]]} for k, v in t["request_size"].items()}
    assert d["request_size"]["read"]["buckets"][0][0] == d["request_size"]["write"]["buckets"][0][0] == 3145728
    assert set(tc) - set(dc) == {"mkdir", "stat"}         # of the cache directory, once at start
    # no chunk is written twice, in either: a request has one continuation (one `reuse` distance)
    assert t["reuse_distance_bytes"]["write_after_write"]["n"] == d["reuse_distance_bytes"]["write_after_write"]["n"] == 0


def test_kv_cache_fitted_parameters_are_what_fit_py_takes_from_the_logs(tmp_path):
    """The ShareGPT case's `params.json` is `fit.py` on its capture's two logs: the load's, and
    LMCache's line per request from the server's. `keep` is a product-limit estimate: a
    conversation the engine held whole is a lower bound on what it would have kept."""
    case = case_dir("lmcache_local_disk", "sharegpt-chat")
    kit, cap = ast_path("lmcache_local_disk").parent / "kit", case / "captures" / "2026-10-02"
    have = json.loads((case / "params.json").read_text())
    out = tmp_path / "fit.json"
    r = subprocess.run([sys.executable, str(kit / "fit.py"), str(cap / "replay.log"), str(cap / "lmcache.log"), "--set", "chunk_bytes=3145728",
                        "--doc", have["doc"], *[x for k, v in have["provenance"]["conditions"].items() if k != "slots_per_engine"
                                                for x in ("--condition", f"{k}={json.dumps(v)}")], "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text()) == have
    p = have["params"]
    assert p["reuse"]["mixture"][1]["dist"] == {"const": 8} and p["requests"] == 300 and p["context"] == 4096
    keep = p["keep"]["empirical"]["values"]
    assert keep == sorted(keep) and keep[-1] == p["context"] and keep.count(208) == 13      # 208: the system prompt's tokens past its whole chunk, all the engine kept
    spec = importlib.util.spec_from_file_location("kv_fit", kit / "fit.py")
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    # uncensored: the sample's own equal shares; a censored observation moves its weight to the larger ones
    assert fit.kept([(10, False), (20, False), (30, False), (40, False)], 4, 99)["empirical"]["values"] == [10, 20, 30, 40]
    assert fit.kept([(10, False), (20, True), (30, False), (40, True)], 4, 100)["empirical"]["values"] == [10, 30, 65, 100]


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_kv_cache_abstract_at_the_agentx_fit_matches_the_corpus_accounting(tmp_path):
    """`workloads/kvcache/lmcache-local-disk/kit/agentx.py`: the KV abstract fitted to the AgentX corpus
    (agentic coding sessions, 2026-10-02) against the corpus's own chunk accounting
    (`agentx.reference.json`: per request, the whole chunks a store that never evicts would
    hit and would store). The abstract stores 8 % more (the reply counted whole where the
    model kept less of it) and, at the default `keep` (91 % of returns hold nothing), loads
    0.91 of its hits, which are within 10 % of the corpus's; the first 10,000 of the 98,827
    requests, one slot (`DESIGN_REVIEW.md` §3.59)."""
    kit, case = ast_path("lmcache_local_disk").parent / "kit", case_dir("lmcache_local_disk", "agentx")
    ref = json.loads((kit / "agentx.reference.json").read_text())
    n = 10_000
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("lmcache_local_disk")), "--gpus", "1", "--seed", "1",
                        "--params-file", str(case / "params.json"), "--param", f"requests={n}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kinds = dict(re.findall(r"(\w+)=(\d+)", re.search(r"by kind: (.*)", r.stdout)[1]))
    stored, loaded = int(kinds["write"]) / n, int(kinds["read"]) / n
    assert ref["chunks_stored"]["mean"] == 14.41 and ref["chunks_hit"]["mean"] == 840.14 and ref["requests"] == 98827
    assert 0.95 * ref["chunks_stored"]["mean"] < stored < 1.2 * ref["chunks_stored"]["mean"], stored
    assert 0.8 * ref["chunks_hit"]["mean"] < loaded / 0.91 < 1.2 * ref["chunks_hit"]["mean"], loaded
    p = json.loads((case / "params.json").read_text())["params"]
    assert p["reuse"]["mixture"][0]["weight"] == 0 and p["turns"]["empirical"]["values"][0] == 1 and len(p["turns"]["empirical"]["values"]) == 100
    assert p["trim"]["mixture"][0]["dist"] == {"const": 0} and 0.03 < p["trim"]["mixture"][1]["weight"] < 0.05


def test_agentx_replay_logs_give_each_request_what_the_engine_held_when_admitted(monkeypatch):
    """`workloads/kvcache/lmcache-local-disk`: the AgentX replay through vLLM and LMCache (2026-10-07,
    run again 2026-10-08 with each session's own tokens and both prefixes shared, DESIGN_REVIEW.md
    §3.63): 8 sessions at 128k context, 2 GiB of engine KV. LMCache logs a lookup
    at every step a request waits for KV memory, the held prefix falling as the running requests
    evict it; a request's is its last lookup before its first load or store. The kit's LMCache
    log keeps those two lines per request (and the first lookup, and the lines of the fit's conditions). The capture's `fitted.params.json` is `agentx.py fit
    --replay` on these logs and the corpus (not committed; the kit's README has the command)."""
    cap = case_dir("lmcache_local_disk", "agentx") / "captures" / "replay-2gib-pool"
    monkeypatch.syspath_prepend(str(ast_path("lmcache_local_disk").parent / "kit"))
    agentx = importlib.import_module("agentx")
    sessions, sent = agentx.replayed(cap / "replay.log", cap / "lmcache.log")
    assert sessions == 8 and len(sent) == 219
    assert sum(1 for _, held in sent.values() if held == 0) == 46           # the engine held nothing, not even a shared prefix
    p = json.loads((cap / "fitted.params.json").read_text())["params"]
    assert (p["concurrency"], p["requests"], p["context"], p["sys_local"], p["sys_prompts"], p["sub_prompts"]) == (8, 27, 131072, False, 1, 1)
    assert p["kind"]["empirical"]["weights"] == [8, 10, 27]                 # chains opening with the main agent's prefix, a sub-agent's, neither
    assert p["keep"]["empirical"]["values"].count(0) == 7                   # of 20 shares: nothing held past the prefix


def test_agentx_replay_fits_record_the_engine_they_were_fitted_at(monkeypatch):
    """What an engine holds is its KV pool against its load (DESIGN_REVIEW.md §3.64): a fit of
    `keep`, `sys_held`, and `sub_held` records the pool, the model, the CPU tier (0, LMCache's
    `local_cpu` off), the slots one engine serves, and the replay's seconds per token, as its
    provenance; `agentx.py log` keeps the lines of the server's log they come from. The same
    replay at 2 GiB and at 6 GiB of engine KV (3x the tokens): the larger pool held more of
    every conversation, so fewer requests found nothing held."""
    monkeypatch.syspath_prepend(str(ast_path("lmcache_local_disk").parent / "kit"))
    agentx = importlib.import_module("agentx")
    fit = importlib.import_module("fit")
    for which, pool, n, nothing in (("replay-2gib-pool", 174752, 219, 46), ("replay-6gib-pool", 524288, 232, 19)):
        cap = case_dir("lmcache_local_disk", "agentx") / "captures" / which
        sessions, sent = agentx.replayed(cap / "replay.log", cap / "lmcache.log")
        assert sessions == 8 and len(sent) == n and sum(1 for _, held in sent.values() if held == 0) == nothing
        c = fit.conditions(cap / "lmcache.log", sessions)
        c_in, c_out = agentx.service(cap / "replay.log")
        c.update(latency_s_per_prompt_token=float("%.3g" % c_in), latency_s_per_reply_token=float("%.3g" % c_out))
        assert json.loads((cap / "fitted.params.json").read_text())["provenance"]["conditions"] == c
        assert (c["engine_pool_tokens"], c["cpu_tier_bytes"], c["model"]) == (pool, 0, "Qwen/Qwen2.5-0.5B-Instruct")


def test_agentx_hold_simulation_evicts_a_freed_tail_first_and_preempts_the_last_admitted(monkeypatch):
    """`agentx.simulate`, vLLM's prefix cache as `agentx.py holds` runs it (DESIGN_REVIEW.md §3.64):
    a request's blocks freed tail first, so its head is evicted last; with nothing free to
    evict, the last admitted running request is preempted and its blocks freed."""
    monkeypatch.syspath_prepend(str(ast_path("lmcache_local_disk").parent / "kit"))
    agentx = importlib.import_module("agentx")
    blocks = {"a": ["a1", "a2"], "b": ["b1", "b2", "b3"], "c": ["a1", "a2"]}
    held, preempted = agentx.simulate([(0, 1, "a"), (1, 0, "a"), (2, 1, "b"), (3, 0, "b"), (4, 1, "c")], blocks, dict.fromkeys(blocks, 0), 4)
    assert (held, preempted) == ({"a": 0, "b": 0, "c": 1}, 0)            # b evicted a2, a's tail; a1 stayed
    blocks = {"x": ["x1", "x2", "x3"], "y": ["y1", "y2"], "z": ["x1", "x2"]}
    held, preempted = agentx.simulate([(0, 1, "x"), (1, 1, "y"), (2, 0, "y"), (3, 1, "z")], blocks, dict.fromkeys(blocks, 0), 4)
    assert (held, preempted) == ({"x": 0, "y": 0, "z": 2}, 1)            # y preempted x, whose tail x3 went first


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_kv_cache_direct_path_is_the_calls_of_lmcaches_odirect_trace(tmp_path):
    """LMCache's `use_odirect` (2026-10-07, DESIGN_REVIEW.md §3.63): a chunk is written by
    `openat(O_WRONLY|O_CREAT|O_DIRECT)`, `write`, `close`, and read by
    `openat(O_RDONLY|O_DIRECT|O_CLOEXEC)`, `fstat`, `read`, `close`; the abstract's `direct`
    issues the same, under the backend it declares (`sync`)."""
    case = case_dir("lmcache_local_disk", "agentx-odirect")
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ast_path("lmcache_local_disk")), "--gpus", "1", "--seed", "1",
                        "--params-file", str(case / "params.json"), "--metrics-json", str(dry),
                        "--metrics-sample", "64"], capture_output=True, text=True)       # the counts only, which sampling leaves exact
    assert r.returncode == 0, r.stderr
    t, d = json.loads((case / "captures/replay-2gib-pool/trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    tc, dc = t["counts"], d["counts"]
    for c in (tc, dc):
        assert c["open"] == c["close"] == c["read"] + c["write"] and c["fstat"] == c["read"] and "ioctl" not in c and "lseek" not in c
    assert (tc["write"], tc["read"]) == (5994, 41004) and (dc["write"], dc["read"]) == (5858, 29895)


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_kv_shared_abstracts_match_the_traces_of_the_fs_backend(tmp_path):
    """`workloads/kvcache/lmcache-shared-fs`: vLLM with LMCache's `fs://` backend, the 300 requests of
    the row above sent to an engine on an empty store and then to a restarted engine on the
    filled one, traced 2026-10-02. A chunk file is a 28-byte header and the chunk, read as one
    buffer and the rest; the writer renames every file into place and the reader writes
    nothing. The writer stores and loads what the local-disk engine did, chunk for chunk; the
    abstract's counts at seed 1 are the low end of its seeds (`ABSTRACTS.md` §8)."""
    case = case_dir("lmcache_shared_fs", "sharegpt-chat")
    got = {}
    for who, ast, params in (("writer", "lmcache_shared_fs", "writer.params.json"), ("reader", "lmcache_shared_fs_reader", "reader.params.json")):
        dry = tmp_path / f"{who}.json"
        r = subprocess.run([str(RUNNER), "dry-run", str(ast_path(ast)), "--gpus", "1", "--seed", "1",
                            "--params-file", str(case / params), "--metrics-json", str(dry)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        t, d = json.loads((case / f"captures/2026-10-02/{who}.trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
        got[who] = (t["counts"], d["counts"])
        for c in (t["counts"], d["counts"]):              # Python's open around every file; a load is two reads
            assert c["open"] == c["fstat"] == c["ioctl"] == c["lseek"] == c["close"]
        assert set(t["counts"]) - set(d["counts"]) == {"mkdir"}          # of the store's directory, once at start
        assert [b[0] for b in t["request_size"]["read"]["buckets"]] == [b[0] for b in d["request_size"]["read"]["buckets"]] == [1048576, 2097152]
    (tw, dw), (tr, dr) = got["writer"], got["reader"]
    assert (tw["rename"], dw["rename"]) == (370, 381) and "rename" not in tr and "rename" not in dr
    assert (tw["write"], dw["write"]) == (2 * 370, 2 * 381) and "write" not in tr and "write" not in dr
    assert (tw["open"] - tw["rename"], dw["open"] - dw["rename"]) == (755, 595)      # chunks loaded by the writer
    assert (tr["open"], dr["open"]) == (1124, 902)                                    # and by the reader
    assert (tw["read"], tr["read"]) == (2 * 755, 2 * 1124) and (dw["read"], dr["read"]) == (2 * 595, 2 * 902)
    assert (tw["stat"], dw["stat"]) == (1418, 1336)       # 1,193 hits and a miss in most requests
    assert (tr["stat"], dr["stat"]) == (1564, 1499)       # every chunk of every prompt, and the directory once in the trace
    # the same parameter values in both files: the reader is run with the writer's; and they are the local-disk
    # mechanism's ShareGPT case (the same requests, and the engine held the same of each) with the header and the buffer
    pw, pr = (json.loads((case / f).read_text())["params"] for f in ("writer.params.json", "reader.params.json"))
    assert pw == pr
    assert pw == {**json.loads((case_dir("lmcache_local_disk", "sharegpt-chat") / "params.json").read_text())["params"], "meta_bytes": 28, "buf": 1048576}



def test_judge_classes_spread_and_the_tolerance_file(tmp_path):
    """The rule of `compare --judge`: a row is within when its distance is at most its
    class's tolerance plus what the abstract differs from itself by at other seeds."""
    assert trace.tolerance_class("reuse distance, read after read (p50 / p90 / p99)") == "reuse distance"
    assert trace.tolerance_class("reuse: first touches / block reads") == "reuse"
    assert trace.tolerance_class("ops: read / all ops") == "ops"
    pop = {"distinct": 5000, "accesses": 1, "max": 1, "top_0_1_pct": 0.001, "top_1_pct": 0.01, "top_10_pct": 0.1, "counts": []}
    a, b = {"popularity_blocks": pop, "popularity_objects": {**pop, "distinct": 50}}, {"popularity_blocks": pop, "popularity_objects": pop}
    rows = [
        ("mix: reads / data ops", "", "", 0.04),
        ("request size, read (p50 / p90 / p99)", "", "", 0.06),
        ("reuse distance, read after read (p50 / p90 / p99)", "", "", 0.25),
        ("popularity, blocks: accesses to the top 0.1 %", "", "", 0.9),    # 5 blocks: says nothing
        ("popularity, blocks: accesses to the top 1 %", "", "", 0.01),     # 50 blocks
        ("popularity, objects: accesses to the top 10 %", "", "", 0.9),    # 5 objects on one side
        ("fan-out", "none", "", 1.0),
        ("depth", "none", "", 1.0),
        ("run length, write, ops (p50 / p90 / p99)", "none", "none", None),
    ]
    verdicts = lambda j: [r[5] for r in j]
    j, stale = trace.judge(rows, a, b)
    assert verdicts(j) == ["ok", "outside", "outside", "not judged", "ok", "not judged", "outside", "outside", "not judged"] and not stale
    assert j[0][4] == 0.05 and j[2][4] == 0.10
    # a trace read without --chain-gap-us has no depth to judge
    j, _ = trace.judge(rows, a, b, docs=[{"source": "strace", "chain_gap_us": None}, {"source": "dry-run"}])
    assert verdicts(j)[7] == "not judged"
    spec = {"aeiou_tolerances": 1, "tolerances": {"request size": 0.07},
            "unseen": [{"metric": "fan-out", "reason": "threads"}, {"metric": "ops: mkdir", "reason": "no such row"}],
            "outside": [{"metric": "reuse distance, read after read", "reason": "known"}, {"metric": "mix: reads", "reason": "was"}]}
    j, stale = trace.judge(rows, a, b, spec=spec)
    assert verdicts(j)[:3] == ["ok", "ok", "outside"] and j[2][6] == "known" and (j[6][5], j[6][6]) == ("unseen", "threads")
    assert len(stale) == 2 and "ops: mkdir" in stale[0] and "mix: reads" in stale[1]
    f = tmp_path / "t.json"
    f.write_text(json.dumps(spec))
    assert trace.load_tolerances(str(f)) == spec
    for bad in ({"aeiou_tolerances": 2}, {**spec, "waive": []}, {**spec, "tolerances": {"sizes": 0.1}}, {**spec, "unseen": [{"metric": "x"}]}):
        f.write_text(json.dumps(bad))
        with pytest.raises(BuildError):
            trace.load_tolerances(str(f))


# Every traced pair under the default tolerances (`DESIGN_REVIEW.md` §3.55): the trace's
# metrics, the abstract, the fitted parameters, `--gpus`, and the verdict. A pair's
# tolerance file, beside its trace document, names the rows the trace cannot show and
# records, with the reason, the rows that are outside.
# every trace of a capture with the verdict `compare --judge` gives on it, from the captures' capture.toml
# (DESIGN_REVIEW.md §3.66; aeiou-catalog checks that every trace has one): (capture, trace document's stem, AST,
# parameter file, GPUs, verdict)
CAPTURE_PAIRS = catalog.capture_pairs(WORKLOADS)


def test_every_trace_of_a_capture_is_a_pair():
    have = {(p.parent, p.name[: -len(".metrics.json")]) for p in WORKLOADS.rglob("captures/*/*.metrics.json")}
    assert have == {(c, t) for c, t, *_ in CAPTURE_PAIRS}


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("capture, which, ast, params, gpus, verdict", CAPTURE_PAIRS,
                         ids=[f"{str(c).split('/cases/')[1]}:{t}" for c, t, *_ in CAPTURE_PAIRS])
def test_capture_pair_against_the_tolerances(tmp_path, capsys, capture, which, ast, params, gpus, verdict):
    """`aeiou-trace compare --judge` on a committed trace and its abstract at the fitted
    parameters, with the abstract at three other seeds as its own spread. The verdict is
    the recorded one, every row outside has its reason in the pair's file, and the file
    has no entry that matches nothing."""
    # the trace's own unit: the KV captures measure in chunks, every access being one whole chunk (the
    # shares and distances are those at 4 KiB, the counts 768 times fewer, and so the work)
    block = json.loads((capture / f"{which}.metrics.json").read_text())["block"]
    docs = [tmp_path / f"dry.{seed}.json" for seed in (1, 2, 3, 4)]
    runs = [subprocess.Popen([str(RUNNER), "dry-run", str(ast_path(ast)), "--gpus", str(gpus), "--seed", str(seed),
                              "--params-file", str(params), "--metrics-json", str(doc), "--metrics-block", str(block)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            for seed, doc in zip((1, 2, 3, 4), docs)]               # the seeds at once: a dry run's metrics take one core
    for p in runs:
        _, err = p.communicate()
        assert p.returncode == 0, err
    spec = capture / f"{which}.tolerances.json"
    args = ["compare", str(capture / f"{which}.metrics.json"), str(docs[0]), *[x for d in docs[1:] for x in ("--self", str(d))]]
    rc = trace.main(args + (["--tolerances", str(spec)] if spec.exists() else ["--judge"]))
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith(verdict + ": "), out[-1]
    assert rc == (0 if verdict == "accepted" else 1)
    assert not [l for l in out if l.startswith("note: ")], out
    assert not [l for l in out if l.endswith("OUTSIDE")], out          # an outside row without a recorded reason
    assert (verdict == "not accepted") == any(" OUTSIDE (" in l for l in out)


# ---------------------------------------------------------------- export: the runner's trace file

WORKERS = """
100 1700000001.000000 openat(AT_FDCWD</cwd>, "/d/shared", O_RDONLY|O_CLOEXEC) = 3</d/shared> <0.000010>
100 1700000001.000100 clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7f) = 200 <0.000050>
100 1700000001.000200 read(3</d/shared>, "x"..., 4096) = 4096 <0.000010>
200 1700000001.000300 read(3</d/shared>, "x"..., 4096) = 4096 <0.000010>
200 1700000001.000400 openat(AT_FDCWD</cwd>, "/d/own", O_RDONLY) = 4</d/own> <0.000010>
200 1700000001.000500 read(4</d/own>, "x"..., 100) = 100 <0.000010>
200 1700000001.000600 read(4</d/own>, "x"..., 100) = 100 <0.000010>
200 1700000001.000700 close(4</d/own>) = 0 <0.000010>
200 1700000001.000800 close(3</d/shared>) = 0 <0.000010>
200 1700000001.000900 +++ exited with 0 +++
100 1700000001.001000 pread64(3</d/shared>, "x"..., 50, 8192) = 50 <0.000010>
100 1700000001.001100 openat(AT_FDCWD</cwd>, "/d/out/new", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 4</d/out/new> <0.000010>
100 1700000001.001200 write(4</d/out/new>, "y"..., 1000) = 1000 <0.000010>
100 1700000001.001300 rename("/d/out/new", "/d/out/final") = 0 <0.000010>
100 1700000001.001400 close(4</d/out/final>) = 0 <0.000010>
100 1700000001.001500 close(3</d/shared>) = 0 <0.000010>
100 1700000001.001600 +++ exited with 0 +++
"""


def _export(text, root="/d"):
    return trace.export_of(text.strip().splitlines(), root)


def test_export_lines_are_the_calls_relative_to_the_root():
    h, ev = _export(SEQUENTIAL)
    assert h["aeiou_trace"] == 1 and h["root"] == "/d" and h["lanes"] == 1 and h["lines"] == 11 and h["opens"] == 1 and h["creates"] == []
    ops = [(e["op"], e.get("path"), e.get("fd"), e.get("offset"), e.get("len"), e.get("ret")) for e in ev]
    assert ops[0] == ("open", "a", 0, None, None, None) and ev[0]["flags"] == ["RDONLY", "CLOEXEC"]
    # sequential reads stay sequential (one lane: no shared position); the pread keeps its offset
    assert ops[2:6] == [("read", None, 0, None, 4096, 4096), ("read", None, 0, None, 4096, 4096), ("read", None, 0, None, 4096, 1808), ("read", None, 0, None, 4096, None)]
    assert ops[6] == ("lseek", None, 0, 0, None, None) and ev[6]["whence"] == "SET"
    assert ops[8] == ("read", None, 0, 8192, 200, 200)
    assert ops[10] == ("stat", "missing", None, None, None, "ENOENT")
    # time from the first exported call, the call's duration beside it
    assert ev[0]["t"] == 0 and ev[3]["dur"] == 100136 and ev[10]["t"] == 1100063
    # a lane outside the root (tid 101) is not a lane


def test_export_resolves_shared_positions_and_lists_creates():
    h, ev = _export(WORKERS)
    assert h["lanes"] == 2 and h["creates"] == ["out/new", "out/final"]
    assert h["notes"]["shared_positions_resolved"] == 2
    reads = [(e["lane"], e["fd"], e.get("offset"), e["len"]) for e in ev if e["op"] == "read"]
    # the inherited description's reads are positioned at what each actually read; the
    # child's own file keeps its sequential reads
    assert reads == [(0, 0, 0, 4096), (1, 0, 4096, 4096), (1, 1, None, 100), (1, 1, None, 100), (0, 0, 8192, 50)]
    closes = [(e["lane"], e["fd"]) for e in ev if e["op"] == "close"]
    assert closes == [(1, 1), (1, 0), (0, 2), (0, 0)]
    w = next(e for e in ev if e["op"] == "write")
    assert (w["fd"], w["len"], w["ret"]) == (2, 1000, 1000)
    r = next(e for e in ev if e["op"] == "rename")
    assert (r["path"], r["to"]) == ("out/new", "out/final")


# a thread's close begins, the kernel frees the descriptor, and another thread's open takes it
# and is printed whole before the close resumes (seen 1 in 40 under CPU contention, 2026-10-07)
REUSED_FD = """
300 1700000003.000000 openat(AT_FDCWD</d>, "/d/a", O_RDONLY|O_CLOEXEC) = 3</d/a> <0.000010>
300 1700000003.000100 clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD, child_tid=0x7f, stack=0x7f, stack_size=0x1000}, 88) = 301 <0.000050>
300 1700000003.000200 read(3</d/a>, "a"..., 4096) = 4096 <0.000010>
300 1700000003.000300 close(3</d/a> <unfinished ...>
301 1700000003.000400 openat(AT_FDCWD</d>, "/d/b", O_RDONLY|O_CLOEXEC) = 3</d/b> <0.000010>
300 1700000003.000500 <... close resumed>) = 0 <0.000200>
301 1700000003.000600 read(3</d/b>, "b"..., 4096) = 4096 <0.000010>
301 1700000003.000700 close(3</d/b>) = 0 <0.000010>
"""


def test_a_descriptor_reused_while_its_close_resumes_closes_the_first_file():
    h, ev = _export(REUSED_FD)
    closes = [e for e in ev if e["op"] == "close"]
    opens = {e["fd"]: e["path"] for e in ev if e["op"] == "open"}
    assert [opens[e["fd"]] for e in closes] == ["a", "b"]
    assert trace.metrics_of(REUSED_FD.strip().splitlines(), ["/d"])["total"]["counts"]["close"] == 2


@pytest.mark.parametrize("text", [SEQUENTIAL, WORKERS, REUSED_FD])
def test_exported_file_measures_as_the_strace(tmp_path, text):
    h, ev = _export(text)
    out = tmp_path / "t.jsonl"
    sha = trace.write_export(str(out), h, ev)
    assert len(sha) == 64
    a = trace.metrics_of(text.strip().splitlines(), ["/d"])
    b = trace.metrics_of_export(out.read_text().splitlines())
    assert b["source"] == "trace" and b["contexts"] == h["lanes"]
    assert a["total"] == b["total"]
    # and through the command line, which sniffs the header
    m = tmp_path / "m.json"
    assert trace.main(["metrics", str(out), "-o", str(m)]) == 0
    assert json.loads(m.read_text())["total"] == a["total"]


# strace prints a descriptor's `-yy` path as it prints a quoted string, without the quotes:
# non-ASCII bytes and `<`, `>` (so that the annotation's end is found) in octal, a backslash
# doubled; the names of one file in a path argument and in an annotation are one name
ESCAPED = r'''
500 1700000005.000000 newfstatat(AT_FDCWD</r/\303\251t\303\251>, "caf\303\251", {st_mode=S_IFREG|0644, st_size=8192, ...}, 0) = 0 <0.000010>
500 1700000005.000100 openat(AT_FDCWD</r/\303\251t\303\251>, "caf\303\251", O_RDONLY) = 3</r/\303\251t\303\251/caf\303\251> <0.000010>
500 1700000005.000200 read(3</r/\303\251t\303\251/caf\303\251>, "x"..., 4096) = 4096 <0.000010>
500 1700000005.000300 close(3</r/\303\251t\303\251/caf\303\251>) = 0 <0.000010>
500 1700000005.000400 openat(AT_FDCWD</r/\303\251t\303\251>, "c<d>e", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 3</r/\303\251t\303\251/c\74d\76e> <0.000010>
500 1700000005.000500 write(3</r/\303\251t\303\251/c\74d\76e>, "x"..., 4096) = 4096 <0.000010>
500 1700000005.000600 close(3</r/\303\251t\303\251/c\74d\76e>) = 0 <0.000010>
500 1700000005.000700 openat(AT_FDCWD</r/\303\251t\303\251>, "x\\y", O_RDONLY|O_DIRECT) = 3</r/\303\251t\303\251/x\\y> <0.000010>
500 1700000005.000800 io_submit(0x7f00, 1, [{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</r/\303\251t\303\251/x\\y>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=4096}]) = 1 <0.000010>
500 1700000005.000900 io_getevents(0x7f00, 1, 1, [{data=0x1, obj=0x7f10, res=4096, res2=0}], NULL) = 1 <0.000050>
500 1700000005.001000 close(3</r/\303\251t\303\251/x\\y>) = 0 <0.000010>
500 1700000005.001100 openat(AT_FDCWD</r/\303\251t\303\251>, "a b", O_RDONLY) = 3</r/\303\251t\303\251/a b> <0.000010>
500 1700000005.001200 read(3</r/\303\251t\303\251/a b>, "x"..., 4096) = 4096 <0.000010>
500 1700000005.001300 close(3</r/\303\251t\303\251/a b>) = 0 <0.000010>
500 1700000005.001400 openat(AT_FDCWD</r/\303\251t\303\251>, "n,m", O_RDONLY) = 3</r/\303\251t\303\251/n,m> <0.000010>
500 1700000005.001500 read(3</r/\303\251t\303\251/n,m>, "x"..., 4096) = 4096 <0.000010>
500 1700000005.001600 close(3</r/\303\251t\303\251/n,m>) = 0 <0.000010>
500 1700000005.001700 openat(AT_FDCWD</r/\303\251t\303\251>, "caf\303\251", O_RDONLY) = 3</r/\303\251t\303\251/caf\303\251> <0.000010>
500 1700000005.001800 read(3</r/\303\251t\303\251/caf\303\251>, "x"..., 4096) = 4096 <0.000010>
500 1700000005.001900 close(3</r/\303\251t\303\251/caf\303\251>) = 0 <0.000010>
'''


def test_a_descriptors_escaped_path_is_unescaped():
    assert trace.fd_arg(r"3</r/caf\303\251>") == (3, "/r/café")
    assert trace.fd_arg(r"3</r/c\74d\76e>") == (3, "/r/c<d>e")
    assert trace.fd_arg(r"3</r/x\\y>") == (3, "/r/x\\y")
    assert trace.fd_arg(r"AT_FDCWD</r/a b>") == (None, "/r/a b")
    assert trace.fd_arg(r"4</r/caf\303\251 (deleted)>") == (4, "/r/café")
    assert trace.fd_arg(r"1</dev/n\74ll\76<char 1:3>>") == (1, "/dev/n<ll>")
    assert split_args(r'3</r/c\74d,e\76>, "x"..., 4096') == [r"3</r/c\74d,e\76>", '"x"...', "4096"]


def test_metrics_and_export_give_an_escaped_file_one_name():
    lines = ESCAPED.strip().splitlines()
    # the root itself has escaped characters: it matches only once the annotations are unescaped
    d = metrics_of(lines, ["/r/été"])
    assert d["total"]["counts"] == {"open": 6, "close": 6, "read": 5, "write": 1, "stat": 1}
    pop = d["total"]["popularity_objects"]
    assert pop["distinct"] == 5 and pop["counts"] == [[2, 1], [1, 4]]   # café twice, the rest once
    header, events = trace.export_of(lines, "/r/été")
    assert header["creates"] == ["c<d>e"] and header["opens"] == 6 and header["notes"] == {}
    paths = [e["path"] for e in events if "path" in e]
    assert paths == ["café", "café", "c<d>e", "x\\y", "a b", "n,m", "café"]


# a thread opens and reads, the main thread closes and reopens the path (taking the same
# descriptor), the thread reads again: one table, so the close is the thread's open's and the
# second read is on the new open. The thread is made by legacy `clone` (glibc < 2.34, as
# glibc 2.31 prints it under strace -f, flags the second argument) or by `clone3`.
SHARED_TABLE = """
100 1700000006.000000 {clone} = 101 <0.000050>
101 1700000006.000100 openat(AT_FDCWD</d>, "/d/x", O_RDONLY) = 3</d/x> <0.000010>
101 1700000006.000200 read(3</d/x>, "..."..., 4096) = 4096 <0.000010>
100 1700000006.000300 close(3</d/x>) = 0 <0.000010>
100 1700000006.000400 openat(AT_FDCWD</d>, "/d/x", O_RDONLY) = 3</d/x> <0.000010>
101 1700000006.000500 read(3</d/x>, "..."..., 4096) = 4096 <0.000010>
"""
LEGACY_CLONE = ("clone(child_stack=0x7f5c3a1fdfb0, flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM"
                "|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, parent_tid=[101], tls=0x7f5c3a1fe700, child_tidptr=0x7f5c3a1fe9d0)")
CLONE3 = "clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, child_tid=0x7f5c3a1fe9d0, parent_tid=0x7f5c3a1fe9d0, exit_signal=0, stack=0x7f5c399fd000, stack_size=0x7fff00, tls=0x7f5c3a1fe6c0} => {parent_tid=[101]}, 88)"


def _shared_table(clone, unfinished=False):
    text = SHARED_TABLE.format(clone=clone)
    if unfinished:   # the thread's first call comes before its clone returns: the `_adopt` path
        lines = text.strip().splitlines()
        lines[0] = f"100 1700000006.000000 {clone} <unfinished ...>"
        lines.insert(2, "100 1700000006.000150 <... " + clone.split("(")[0] + " resumed>) = 101 <0.000200>")
        text = "\n".join(lines)
    lines = text.strip().splitlines()
    return metrics_of(lines, ["/d"]), trace.export_of(lines, "/d")


@pytest.mark.parametrize("unfinished", [False, True])
def test_a_legacy_clone_shares_the_descriptor_table_as_clone3_does(unfinished):
    m3, (h3, ev3) = _shared_table(CLONE3, unfinished)
    m, (h, ev) = _shared_table(LEGACY_CLONE, unfinished)
    assert m == m3 and h == h3 and ev == ev3
    ops = [(e["lane"], e["op"], e["fd"], e.get("offset")) for e in ev]
    assert ops == [(0, "open", 0, None), (0, "read", 0, 0), (1, "close", 0, None), (1, "open", 1, None), (0, "read", 1, 0)]
    # a fork copies the table: the thread's descriptor 3 stays its first open
    mf, (hf, evf) = _shared_table("clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7f)", unfinished)
    assert [(e["lane"], e["op"], e["fd"]) for e in evf][-1] == (0, "read", 0)


def test_shares_files_reads_the_flags_argument_only():
    assert trace.shares_files("clone", LEGACY_CLONE[6:-1])
    assert trace.shares_files("clone3", CLONE3[7:-1])
    assert not trace.shares_files("clone", "child_stack=NULL, flags=CLONE_CHILD_CLEARTID|SIGCHLD, child_tidptr=0x7f")
    assert not trace.shares_files("clone", "child_stack=NULL, flags=CLONE_FILESYSTEM_X")   # whole flag names only
    assert not trace.shares_files("fork", "flags=CLONE_FILES")


# four opens of one log with O_APPEND, the first truncating it, 4 KiB written through each:
# the file the trace made is 16 KiB, so each write starts where the last one ended
APPENDS = "\n".join(
    f'300 1700000007.{4 * i:04d}00 openat(AT_FDCWD</d>, "/d/log", O_WRONLY|O_CREAT|O_APPEND{"|O_TRUNC" if i == 0 else ""}|O_CLOEXEC, 0644) = 3</d/log> <0.000010>\n'
    f'300 1700000007.{4 * i + 1:04d}00 write(3</d/log>, "x"..., 4096) = 4096 <0.000010>\n'
    f'300 1700000007.{4 * i + 2:04d}00 close(3</d/log>) = 0 <0.000010>'
    for i in range(4))


def test_an_append_to_a_file_the_trace_made_starts_at_its_end():
    lines = APPENDS.strip().splitlines()
    d = metrics_of(lines, ["/d"])
    t = d["total"]
    assert d["notes"] == {}
    assert t["reuse_distance_bytes"]["first_touch"] == {"read": 0, "write": 4}
    assert t["popularity_blocks"]["counts"] == [[1, 4]]
    header, events = trace.export_of(lines, "/d")
    assert [e["offset"] for e in events if e["op"] == "write"] == [0, 4096, 8192, 12288]
    assert header["notes"] == {"append_positions_resolved": 4}
    # the exported file measures as the strace does
    assert trace.metrics_of_export([json.dumps(header)] + [json.dumps(e) for e in events])["total"] == t


def test_an_append_whose_base_the_trace_cannot_know_is_noted():
    # no truncation, no creation the trace saw; then a rename and an unlink end what is known
    known = APPENDS.replace("|O_TRUNC", "")
    d = metrics_of(known.strip().splitlines(), ["/d"])
    assert d["notes"] == {"append_to_unknown_size": 4}
    lines = (APPENDS + '\n300 1700000008.000000 truncate("/d/log", 100) = 0 <0.000010>'
             '\n300 1700000008.000100 rename("/d/log", "/d/old") = 0 <0.000010>'
             '\n300 1700000008.000200 openat(AT_FDCWD</d>, "/d/old", O_WRONLY|O_APPEND) = 4</d/old> <0.000010>'
             '\n300 1700000008.000300 write(4</d/old>, "x"..., 10) = 10 <0.000010>'
             '\n300 1700000008.000400 unlink("/d/old") = 0 <0.000010>'
             '\n300 1700000008.000500 openat(AT_FDCWD</d>, "/d/old", O_WRONLY|O_CREAT|O_APPEND, 0644) = 5</d/old> <0.000010>'
             '\n300 1700000008.000600 write(5</d/old>, "x"..., 10) = 10 <0.000010>').strip().splitlines()
    header, events = trace.export_of(lines, "/d")
    assert [e.get("offset") for e in events if e["op"] == "write"] == [0, 4096, 8192, 12288, 100, None]
    assert metrics_of(lines, ["/d"])["notes"] == {"append_to_unknown_size": 1}


# strace prints the first 32 elements of an array unless given `-v`: a 40-request
# io_submit, its 40 results, and a 40-iovec preadv, as a trace taken without it shows them
_CB = "{{aio_data={d:#x}, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset={o}}}"
_EV = "{{data={d:#x}, obj=0x7f10, res=4096, res2=0}}"
_IOV = '{iov_base="\\0\\0"..., iov_len=4096}'


def _cut(elems, n=32):
    return "[" + ", ".join(elems[:n]) + (", ...]" if len(elems) > n else "]")


def _arrays(n):
    cbs = [_CB.format(d=i + 1, o=i * 4096) for i in range(40)]
    evs = [_EV.format(d=i + 1) for i in range(40)]
    return f"""
300 1700000003.000000 openat(AT_FDCWD</d>, "idx", O_RDONLY) = 3</d/idx> <0.000010>
300 1700000003.000100 io_submit(0x7f00, 40, {_cut(cbs, n)}) = 40 <0.000010>
300 1700000003.000200 io_getevents(0x7f00, 40, 40, {_cut(evs, n)}, NULL) = 40 <0.000050>
300 1700000003.000300 preadv(3</d/idx>, {_cut([_IOV] * 40, n)}, 40, 0) = 163840 <0.000050>
"""


def test_arrays_strace_cut_at_32_elements_are_noted():
    whole = total(_arrays(40))
    assert whole["total"]["fan_out"] == {"40": 1}
    assert not any("truncated" in k for k in whole["notes"])
    cut = total(_arrays(32))
    # the 8 requests past the 32nd are not in the trace: what and where they read is
    # unknown, so they are not counted, and the notes say how many and to trace with -v
    assert cut["total"]["fan_out"] == {"32": 1}
    assert cut["notes"]["aio_requests_truncated"] == 8
    assert cut["notes"]["aio_events_truncated"] == 8
    assert cut["notes"]["iov_arrays_truncated"] == 1
    assert "strace -v" in cut["notes"]["arrays_truncated"]
    header, _ = trace.export_of(_arrays(32).strip().splitlines(), "/d")
    assert header["notes"]["aio_requests_truncated"] == 8


def test_a_cut_iovec_list_takes_the_bytes_the_call_moved():
    # 32 of 40 iovecs shown sum to 131072; the call moved 163840, the whole list's length
    one = lambda n: "\n".join(l for l in _arrays(n).strip().splitlines() if "io_" not in l)
    whole, cut = total(one(40))["total"], total(one(32))["total"]
    assert cut["bytes_read"] == whole["bytes_read"] == 163840
    assert cut["request_size"] == whole["request_size"] and cut["request_size"]["read"]["max"] == 163840
# the command line (RC1 review): a file the tool cannot open is a one-line error naming it,
# exit 1, never a traceback; two rules for `compare` are a usage error, exit 2, before any
# work; `-` is UTF-8 whatever the locale, and a message the console cannot spell is escaped
def _trace_tool(*args, env=None, stdin=None):
    return subprocess.run([sys.executable, "-m", "aeiou.trace", *map(str, args)], capture_output=True, cwd=BUILDER, env=env, input=stdin)


UNICODE_TRACE = """
100 1700000001.000000 openat(AT_FDCWD</cwd>, "/d/\u00e9", O_RDONLY|O_CLOEXEC) = 3</d/\u00e9> <0.000010>
100 1700000001.000100 read(3</d/\u00e9>, "x"..., 4096) = 4096 <0.000010>
100 1700000001.000200 close(3</d/\u00e9>) = 0 <0.000010>
""".lstrip().encode("utf-8")


def test_a_file_the_tool_cannot_open_is_an_error_naming_it(tmp_path):
    st = tmp_path / "st.txt"
    st.write_bytes(UNICODE_TRACE)
    absent, nodir = tmp_path / "absent", tmp_path / "no" / "x.json"
    for args, path in ((["metrics", absent, "--root", "/d"], absent), (["export", absent, "--root", "/d", "-o", tmp_path / "x.jsonl"], absent),
                       (["metrics", st, "--root", "/d", "-o", nodir], nodir), (["export", st, "--root", "/d", "-o", nodir], nodir)):
        r = _trace_tool(*args)
        assert (r.returncode, r.stderr.decode()) == (1, f"aeiou-trace {args[0]}: {path}: No such file or directory (os error 2)\n"), r.stderr


def test_max_distance_with_a_judging_rule_is_a_usage_error(tmp_path):
    tol = tmp_path / "tol.json"
    for rule, said in ((["--judge"], "--judge"), (["--tolerances", tol], "--tolerances (which implies --judge)"), (["--self", tol], "--self (which implies --judge)")):
        r = _trace_tool("compare", tmp_path / "a.json", tmp_path / "b.json", "--max-distance", "0.1", *rule)
        err = r.stderr.decode()
        assert r.returncode == 2 and r.stdout == b"", err
        assert err.startswith(f"aeiou-trace compare: --max-distance cannot be used with {said}: they are two rules; give one\n\nUsage: aeiou-trace compare "), err
        assert err.endswith("For more information, try 'aeiou-trace compare --help'.\n"), err


def test_stdin_is_utf8_and_the_console_escapes_what_it_cannot_spell(tmp_path):
    # LC_ALL=C without UTF-8 mode: stdin decodes as ASCII (a path's é two lone surrogates in
    # the export, a different sha256); PYTHONIOENCODING=ascii is a console with no escape for
    # the -o path aeiou-trace prints after the work
    base = {k: v for k, v in os.environ.items() if not k.startswith("LC_") and k not in ("LANG", "PYTHONIOENCODING", "PYTHONUTF8")}
    out = {}
    for locale, mode in (("C.UTF-8", "1"), ("C", "0")):
        o = tmp_path / f"{locale}.jsonl"
        r = _trace_tool("export", "-", "--root", "/d", "-o", o, env={**base, "LC_ALL": locale, "PYTHONUTF8": mode}, stdin=UNICODE_TRACE)
        assert r.returncode == 0, r.stderr
        out[locale] = o.read_bytes()
    assert out["C"] == out["C.UTF-8"]
    assert [e.get("path") for e in map(json.loads, out["C"].splitlines())][1] == "\u00e9"
    st, o = tmp_path / "st.txt", tmp_path / "\u00e9.jsonl"
    st.write_bytes(UNICODE_TRACE)
    r = _trace_tool("export", st, "--root", "/d", "-o", o, env={**base, "PYTHONIOENCODING": "ascii"})
    assert r.returncode == 0 and b"\\xe9.jsonl\n" in r.stdout, r.stderr


# the exporter's paths (RC1 review): every path is what the runner accepts (`trace.rs`,
# `check_path`): relative, non-empty, no `..`, and `.` (the root) only where the call leaves
# the root as it is; a call that changes the root itself is left out with a counted note.
# Paths and roots are normalized as Linux resolves them: `//d` is `/d`, and `/` is a root.
ROOT_PATHS = """
100 1700000001.000000 mkdir("{r}", 0755) = -1 EEXIST (File exists) <0.000010>
100 1700000001.000100 openat(AT_FDCWD</cwd>, "{r}", O_RDONLY|O_CREAT, 0644) = -1 EISDIR (Is a directory) <0.000010>
100 1700000001.000200 stat("{r}/", {{st_mode=S_IFDIR}}) = 0 <0.000010>
100 1700000001.000300 openat(AT_FDCWD</cwd>, "{r}/../{b}/./a//b", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3 <0.000010>
100 1700000001.000400 write(3, "y"..., 100) = 100 <0.000010>
100 1700000001.000500 close(3) = 0 <0.000010>
100 1700000001.000600 rename("{r}/a/b", "{r}/c") = 0 <0.000010>
100 1700000001.000700 chdir("{r}") = 0 <0.000010>
100 1700000001.000800 stat("../{b}/c", {{st_mode=S_IFREG}}) = 0 <0.000010>
100 1700000001.000900 stat("/{r}/c", {{st_mode=S_IFREG}}) = 0 <0.000010>
100 1700000001.001000 rmdir("{r}") = -1 ENOTEMPTY (Directory not empty) <0.000010>
"""


def _runner_accepts(e) -> bool:
    """`check_path` of `runner/aeiou/src/trace.rs` for one exported line."""
    keep = e["op"] == "stat" or (e["op"] == "open" and "CREAT" not in e["flags"] and "TRUNC" not in e["flags"])
    for p in (e[k] for k in ("path", "to") if k in e):
        parts = [c for c in p.split("/") if c not in ("", ".")]
        if not p or p.startswith("/") or ".." in parts or (not parts and not keep):
            return False
    return True


@pytest.mark.parametrize("root,spelled", [("/d", "/d"), ("/d/", "/d"), ("//d", "//d"), ("/d", "//d"), ("/d/sub/..", "/d")])
def test_every_exported_path_is_one_the_runner_accepts(root, spelled):
    b = spelled.rstrip("/").rsplit("/", 1)[1]
    h, ev = _export(ROOT_PATHS.format(r=spelled, b=b), root)
    assert h["root"] == "/d"
    assert all(_runner_accepts(e) for e in ev), ev
    assert [(e["op"], e.get("path"), e.get("to")) for e in ev if "path" in e] == [
        ("stat", ".", None), ("open", "a/b", None), ("rename", "a/b", "c"), ("stat", "c", None), ("stat", "c", None)]
    assert h["creates"] == ["a/b", "c"]
    n = h["notes"]
    assert n["calls_not_exported"] == 3 and n["not_exported:mkdir:the root itself"] == n["not_exported:rmdir:the root itself"] == 1
    assert n["not_exported:open:creates or truncates the root"] == 1


def test_the_root_slash_is_every_path_and_double_slash_is_slash():
    for root in ("/", "//"):
        h, ev = _export('100 1700000001.000000 stat("//", {st_mode=S_IFDIR}) = 0 <0.000010>\n'
                        '100 1700000001.000100 stat("/x", {st_mode=S_IFREG}) = 0 <0.000010>\n', root)
        assert h["root"] == "/" and [e["path"] for e in ev] == [".", "x"], (root, ev)
    assert trace.metrics_of(['100 1700000001.000000 stat("/x", {st_mode=S_IFREG}) = 0 <0.000010>'], ["/"])["total"]["ops"] == 1


def test_a_path_not_under_the_exporters_root_is_an_internal_error():
    ex = trace.Exporter("/d")
    assert (ex.rel("/d"), ex.rel("/d/x")) == (".", "x")
    for bad in ("/e/x", "/dx", "/d//x", "/d/../x", "x"):
        with pytest.raises(AssertionError, match="internal error"):
            ex.rel(bad)


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_the_runner_loads_an_export_that_mkdirs_its_root(tmp_path):
    h, ev = _export(ROOT_PATHS.format(r="/d", b="d"))
    out = tmp_path / "app.jsonl"
    sha = trace.write_export(str(out), h, ev)
    node_ast = tmp_path / "node.ast.json"
    node_ast.write_text(json.dumps({"ast": "0.10", "name": "trace_root", "doc": "calls on the root itself", "params": {}, "datasets": {},
                                    "actors": {"app": {"count": 1, "body": [{"trace": {"file": out.name, "sha256": sha}}]}}}))
    r = subprocess.run([str(RUNNER), "dry-run", str(node_ast), "--gpus", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# a short write and the retry of its rest, between them a failed one (RC1 review): a
# write contributes what the call returned, as a read does, to the bytes, the blocks touched,
# the runs, and the position a sequential call leaves; a failed one contributes nothing. The
# file the trace leaves is 8192 bytes, and the read of it ends at its end.
SHORT_WRITE = """
700 1700000007.000000 openat(AT_FDCWD</d>, "/d/out", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3</d/out> <0.000010>
700 1700000007.000100 write(3</d/out>, "y"..., 8192) = 5000 <0.000010>
700 1700000007.000200 write(3</d/out>, "y"..., 3192) = -1 ENOSPC (No space left on device) <0.000010>
700 1700000007.000300 write(3</d/out>, "y"..., 3192) = 3192 <0.000010>
700 1700000007.000400 close(3</d/out>) = 0 <0.000010>
700 1700000007.000500 openat(AT_FDCWD</d>, "/d/out", O_RDONLY) = 3</d/out> <0.000010>
700 1700000007.000600 read(3</d/out>, "y"..., 8192) = 8192 <0.000010>
700 1700000007.000700 close(3</d/out>) = 0 <0.000010>
"""


def _short_write_export(tmp_path):
    h, ev = _export(SHORT_WRITE)
    out = tmp_path / "app.jsonl"
    sha = trace.write_export(str(out), h, ev)
    return h, ev, out, sha


def test_a_short_or_failed_write_moves_what_it_returned(tmp_path):
    h, ev, out, _ = _short_write_export(tmp_path)
    writes = [(e.get("offset"), e["len"], e.get("ret")) for e in ev if e["op"] == "write"]
    assert writes == [(None, 8192, 5000), (None, 3192, "ENOSPC"), (None, 3192, 3192)]
    a = trace.metrics_of(SHORT_WRITE.strip().splitlines(), ["/d"])["total"]
    assert a["bytes_written"] == 8192 and a["bytes_read"] == 8192
    # one run of the two writes that moved bytes, 0..5000 and 5000..8192
    assert a["run_length"]["write"]["bytes"]["max"] == 8192 and a["run_length"]["write"]["ops"]["max"] == 2
    assert trace.metrics_of_export(out.read_text().splitlines())["total"] == a


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_the_trace_nodes_dry_run_moves_what_a_short_write_returned(tmp_path):
    _, _, out, sha = _short_write_export(tmp_path)
    node_ast = tmp_path / "node.ast.json"
    node_ast.write_text(json.dumps({"ast": "0.10", "name": "trace_short_write", "doc": "a short and a failed write", "params": {}, "datasets": {},
                                    "actors": {"app": {"count": 1, "body": [{"trace": {"file": out.name, "sha256": sha}}]}}}))
    dry = tmp_path / "node.dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(node_ast), "--gpus", "1", "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    nd = json.loads(dry.read_text())["total"]
    assert nd == trace.metrics_of_export(out.read_text().splitlines())["total"]
    assert nd["bytes_written"] == 8192


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_the_trace_nodes_run_leaves_the_file_a_short_write_left(tmp_path):
    """The run issues each write at its length and checks it so, but a sequential one leaves
    the position the traced call left: the file is the application's 8192 bytes under every
    backend, the sync one's `write(2)` at the file position included."""
    _, _, out, sha = _short_write_export(tmp_path)
    node_ast = tmp_path / "node.ast.json"
    node_ast.write_text(json.dumps({"ast": "0.10", "name": "trace_short_write", "doc": "a short and a failed write", "params": {}, "datasets": {},
                                    "actors": {"app": {"count": 1, "body": [{"trace": {"file": out.name, "sha256": sha}}]}}}))
    for backend in ("sync", "io_uring", "libaio"):
        root = tmp_path / ("root-" + backend)
        root.mkdir()
        r = subprocess.run([str(RUNNER), "run", str(node_ast), "--gpus", "1", "--root", str(root), "--time-scale", "0", "--posix", backend], capture_output=True, text=True)
        assert r.returncode == 0, backend + "\n" + r.stdout + r.stderr
        assert (root / "out").stat().st_size == 8192, backend


def test_an_exported_trace_is_read_as_the_runner_reads_it(tmp_path):
    """`aeiou-trace metrics` read an exported trace file, which the runner's `trace` node reads
    too, with plain `json.loads` (RC1 review, R21): a line with a lone surrogate (`\\ud800`)
    or a repeated key passed, and a line that is not JSON ended in a traceback. It reads with
    `emit.loads` and names the file and the line, as the runner names them."""
    head = '{"aeiou_trace": 1, "root": "/x", "lanes": 1, "lines": 1}\n'
    for i, (line, said) in enumerate([
        ('{"lane": 0, "op": "stat", "path": "a\\ud800"}', "line 2: lone surrogate in hex escape (\\ud800)"),
        ('{"lane": 0, "op": "stat", "path": "a", "path": "b"}', "line 2: duplicate field `path`"),
        ('{"lane": 0, "op"', "line 2: Expecting ':' delimiter: line 1 column 17 (char 16)"),
        ('[1]', "line 2: not a JSON object"),
    ]):
        f = tmp_path / f"t{i}.jsonl"
        f.write_text(head + line + "\n", encoding="utf-8")
        r = _trace_tool("metrics", f)
        assert (r.returncode, r.stderr.decode()) == (1, f"aeiou-trace metrics: {f}: {said}\n"), r.stderr
