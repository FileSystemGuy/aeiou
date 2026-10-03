"""`aeiou-trace`: the metrics of an `strace` against their definitions (`runner/README.md`
§10) on hand-written traces, the ports of the runner's histogram and stack distance, the
comparison's distances, and, when `strace` and the runner binary are there, the trace of
`aeiou run` against the same abstract's `aeiou dry-run --metrics-json`: everything that
does not depend on the order must be equal."""
import importlib.util
import json
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

from aeiou import trace  # noqa: E402
from aeiou.trace import Instance, LogHist, compare, metrics_of, split_args  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))


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
        ("train_small_files", 7, ["files=600", "batch=4", "workers=2", "prefetch=2", "steps=12"]),
        ("kv_cache_serving", 5, ["sys_prompts=3", "sys_tokens=6", "chunk_bytes=262144", "concurrency=3", "warm=4", "requests=8",
                              'reuse={"mixture": [{"weight": 0.3, "dist": null}, {"weight": 0.7, "dist": {"const": 2}}]}',
                              'keep={"empirical": {"values": [0, 50, 100], "weights": [1, 1, 1]}}']),
        ("vdb_search_diskann", 9, ["nodes=5000", "threads=2", "queries=6"]),
    ],
)
def test_trace_of_the_runner_matches_its_dry_run(tmp_path, name, seed, params):
    ast = ROOT / "schema" / "examples" / f"{name}.ast.json"
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
    # runs belong to a context: a thread there, a sub-actor here, the same ops either way
    assert t["run_length"] == d["run_length"]
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
    # histograms of that pair differ by the reordering within a round; all else is equal.
    exported = tmp_path / "exported.jsonl"
    r = subprocess.run([sys.executable, "-m", "aeiou.trace", "export", str(st), "--root", str(root), "--exclude", "*.aeiou-*", "-o", str(exported)], capture_output=True, text=True, cwd=ROOT / "builder")
    assert r.returncode == 0, r.stderr
    sha = next(line.split()[1] for line in r.stdout.splitlines() if line.startswith("sha256 "))
    node_ast = tmp_path / "node.ast.json"
    node_ast.write_text(json.dumps({"ast": "0.5", "name": f"trace_{name}", "doc": "the exported trace", "params": {}, "datasets": {},
                                    "actors": {"app": {"count": 1, "body": [{"trace": {"file": exported.name, "sha256": sha}}]}}}))
    node_dry = tmp_path / "node.dry.json"
    r = run(RUNNER, "dry-run", node_ast, "--gpus", "1", "--metrics-json", node_dry)
    assert r.returncode == 0, r.stderr
    nd = json.loads(node_dry.read_text())["total"]
    assert trace.main(["metrics", str(exported), "-o", str(tmp_path / "ex.json")]) == 0
    ex = json.loads((tmp_path / "ex.json").read_text())["total"]
    assert nd == ex, "the runner's walk of the file and the Python reader's disagree"
    strip = lambda m: {k: v for k, v in m.items() if k != "reuse_distance_bytes"} if name == "vdb_search_diskann" else m
    assert strip(t) == strip(nd), "the trace node's dry run is not the strace's metrics"
    fp = next(line.split()[1] for line in r.stdout.splitlines() if line.startswith("fingerprint "))
    r = run(RUNNER, "run", node_ast, "--gpus", "1", "--root", root, "--time-scale", "0", "--clean-namespaces")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"fingerprint {fp}" in r.stdout, r.stdout
    assert "never CLOSED" in r.stdout


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_small_file_abstract_matches_the_trace_of_the_real_loader(tmp_path):
    """`builder/traces/train_small_files`: DataLoader + ImageFolder on NFS, traced 2026-10-01.
    The abstract at the fitted parameters has the application's op mix exactly but for the
    one listing of the dataset root; the corpus of real JPEGs is near the abstract's size
    distribution, not equal to it, and the order differs."""
    kit = BUILDER / "traces" / "train_small_files"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "train_small_files.ast.json"), "--gpus", "1",
                        "--params", str(kit / "fitted.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / "trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    root_listing = {"open": 1, "fstat": 1, "readdir": 1, "close": 1}
    assert {k: v - root_listing.get(k, 0) for k, v in t["counts"].items()} == d["counts"]
    assert t["request_size"]["read"]["buckets"] == d["request_size"]["read"]["buckets"]
    rows = compare(t, d)
    assert max(r[3] for r in rows if r[3] is not None) <= 0.06, rows

    assert subprocess.run([sys.executable, "-m", "aeiou.trace", "compare", str(tmp_path / "absent.json"), str(dry)],
                          capture_output=True, text=True, cwd=BUILDER).stderr.startswith("aeiou-trace: ")


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("which, reuse", [("140MiB", 0.3), ("8MiB", 0.07)])
def test_large_sample_abstract_matches_the_trace_of_np_load(tmp_path, which, reuse):
    """`builder/traces/train_large_samples`: `np.load(...)["x"]` on NFS, traced 2026-10-01.
    The abstract draws its sizes, the corpus has its own, so the counts are near, not equal
    (they are exact per file: `ABSTRACTS.md` §2). Sixteen files say little about the reuse
    distance; 256 say more."""
    kit = BUILDER / "traces" / "train_large_samples"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "train_large_samples.ast.json"), "--gpus", "1",
                        "--params", str(kit / f"fitted.{which}.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / f"trace.{which}.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    listing = {"open": 1, "fstat": 1, "readdir": 1, "close": 1}       # glob lists the root too; the abstract lists the sample directories (one here)
    for k, v in t["counts"].items():
        v -= listing.get(k, 0)
        assert abs(v - d["counts"].get(k, 0)) <= 0.01 * v, k
    for name, _, _, dist in compare(t, d):
        if dist is not None:
            assert dist <= (reuse if name.startswith("reuse distance") else 0.01), name


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_checkpoint_write_abstract_matches_the_trace_of_dcp_save(tmp_path):
    """`builder/traces/ckpt_write_dcp`: `torch.distributed.checkpoint.save` on two ranks,
    traced 2026-10-01. Every call on the shard files and `.metadata` is in the abstract; the
    path checks around `mkdir` that it cannot express are listed here (`ABSTRACTS.md` §3)."""
    kit = BUILDER / "traces" / "ckpt_write_dcp"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "ckpt_write_dcp.ast.json"), "--gpus", "2",
                        "--params", str(kit / "fitted.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / "trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    not_modeled = {
        "mkdir": 2,   # once in the job: the step directory's parent is absent (ENOENT), then made
        "stat": 8,    # per rank and checkpoint the parent (4); the rank that lost the mkdir, the directory (2); the root, first time (2)
    }
    assert {k: v - not_modeled.get(k, 0) for k, v in t["counts"].items()} == d["counts"]
    assert t["bytes_written"] == d["bytes_written"]
    assert t["request_size"]["write"]["buckets"] == d["request_size"]["write"]["buckets"]
    assert max(r[3] for r in compare(t, d) if r[3] is not None) <= 0.05


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("which, bytes_off", [("mixed", 0), ("small-last", 88), ("name-order", 0)])
def test_checkpoint_restore_abstract_matches_the_trace_of_dcp_load(tmp_path, which, bytes_off):
    """`builder/traces/ckpt_restore`: `torch.distributed.checkpoint.load` on two ranks, traced
    2026-10-01. Every call on the shard files and `.metadata` is in the abstract, with its
    count; the check of the checkpoint's parent directory is not (`ABSTRACTS.md` §4). A small
    last item has its zip records 44 bytes nearer its ends than `rec2` and `tail_back` say,
    which is 88 bytes over two ranks. `name-order` has 17 items, read in the sorted order of
    their names across runs of small ones longer than the buffer: the buffer chain."""
    kit = BUILDER / "traces" / "ckpt_restore"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "ckpt_restore.ast.json"), "--gpus", "2",
                        "--params", str(kit / f"fitted.{which}.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / f"trace.{which}.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
    not_modeled = {"stat": 2}   # per rank, the parent of the checkpoint directory (and an `access`, which the trace metrics do not count)
    assert {k: v - not_modeled.get(k, 0) for k, v in t["counts"].items() if v != not_modeled.get(k)} == d["counts"]
    assert d["bytes_read"] - t["bytes_read"] == bytes_off
    assert t["request_size"]["read"]["buckets"] == d["request_size"]["read"]["buckets"]
    assert max(r[3] for r in compare(t, d) if r[3] is not None) <= 0.05


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_model_load_abstract_matches_the_trace_of_from_pretrained(tmp_path):
    """`builder/traces/model_load`: `from_pretrained` on five safetensors shards, traced
    2026-10-01. The library maps the shards and issues no `read` on them, so the trace has the
    calls around the mappings and the small JSON files, and nothing of the tensor bytes
    (`ABSTRACTS.md` §4)."""
    kit = BUILDER / "traces" / "model_load"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "model_load.ast.json"), "--gpus", "1",
                        "--params", str(kit / "fitted.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / "trace.metrics.json").read_text())["total"]["counts"], json.loads(dry.read_text())["total"]["counts"]
    for op in ("open", "close", "fstat", "ioctl", "lseek", "fadvise"):
        assert t[op] == d[op], op
    assert d["read"] - t["read"] == 2 * 5 + 76   # through the mappings, unseen by strace: 8 bytes and the header per shard, 76 tensors
    assert t["stat"] - d["stat"] == 25           # not modeled: directories (the model's 12 times, the two above each shard), a second stat of two JSON files, the probe for an unsharded model.safetensors


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_ivf_abstract_matches_the_trace_of_faiss_search(tmp_path):
    """`builder/traces/vdb_search_ivf`: `faiss.read_index` and 20 searches over
    `OnDiskInvertedLists`, traced 2026-10-01. FAISS maps the lists file and issues no call on
    it, so the trace has the index file's reads and the two opens, and nothing of the lists
    (`ABSTRACTS.md` §6)."""
    kit = BUILDER / "traces" / "vdb_search_ivf"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "vdb_search_ivf.ast.json"), "--gpus", "1",
                        "--params", str(kit / "fitted.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / "trace.metrics.json").read_text())["total"]["counts"], json.loads(dry.read_text())["total"]["counts"]
    assert {k: v for k, v in d.items() if k != "read"} == {k: v for k, v in t.items() if k != "read"}
    assert d["read"] - t["read"] == 20 * 64 * 2   # through the mapping, unseen by strace: ids and codes of 64 lists per query


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("which, reads_off", [("nocache", 0.02), ("cache", 0.04)])
def test_diskann_search_abstract_matches_the_trace(tmp_path, which, reads_off):
    """`builder/traces/vdb_search_diskann`: 1,000 queries through DiskANN's `PQFlashIndex`,
    traced 2026-10-02 without a node cache and with 10,000 nodes cached. The abstract issues
    full rounds of `beam` reads where the search skips nodes it already holds, so it reads a
    little more; the load's small files (pivots, medoids, centroids, metadata) are not
    modeled (`ABSTRACTS.md` §5)."""
    kit = BUILDER / "traces" / "vdb_search_diskann"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "vdb_search_diskann.ast.json"), "--gpus", "1",
                        "--params", str(kit / f"fitted.{which}.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / f"trace.{which}.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
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
    """`builder/traces/vdb_build_diskann`: `build_disk_index` on SIFT1M in 13 shards, traced
    2026-10-02. The data path agrees to a part in a thousand; the shards are modeled at one
    size and the files' small headers and probes are not modeled (`ABSTRACTS.md` §7)."""
    kit = BUILDER / "traces" / "vdb_build_diskann"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "vdb_build_diskann.ast.json"), "--gpus", "1",
                        "--params", str(kit / "fitted.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / "trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
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
    """`builder/traces/kv_cache_serving`: vLLM with LMCache's local-disk backend under a replay
    of ShareGPT, 300 requests with 8 conversations open, traced 2026-10-02. Every chunk is the
    same six calls around one `read` or `write`. At the parameters `fit.py` takes from the
    run's two logs the abstract stores 382 chunks where the engine stored 370 (two of them
    the system prompts', a dataset here) and reads 596 where it read 755 (seed 1; over eight
    seeds it reads 587 to 781: one `keep` draw per round of conversations, §3.57, makes the
    loads come in bursts, so the seeds spread wider than with a draw per request)
    (`ABSTRACTS.md` §8)."""
    kit = BUILDER / "traces" / "kv_cache_serving"
    dry = tmp_path / "dry.json"
    r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / "kv_cache_serving.ast.json"), "--gpus", "1", "--seed", "1",
                        "--params", str(kit / "fitted.params.json"), "--metrics-json", str(dry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t, d = json.loads((kit / "trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
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
    """The kit's `fitted.params.json` is `fit.py` on the kit's two logs: the load's, and
    LMCache's line per request from the server's. `keep` is a product-limit estimate: a
    conversation the engine held whole is a lower bound on what it would have kept."""
    kit = BUILDER / "traces" / "kv_cache_serving"
    have = json.loads((kit / "fitted.params.json").read_text())
    out = tmp_path / "fit.json"
    r = subprocess.run([sys.executable, str(kit / "fit.py"), str(kit / "replay.log"), str(kit / "lmcache.log"), "--set", "chunk_bytes=3145728",
                        "--doc", have["doc"], "-o", str(out)], capture_output=True, text=True)
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
def test_kv_shared_abstracts_match_the_traces_of_the_fs_backend(tmp_path):
    """`builder/traces/kv_cache_shared`: vLLM with LMCache's `fs://` backend, the 300 requests of
    the row above sent to an engine on an empty store and then to a restarted engine on the
    filled one, traced 2026-10-02. A chunk file is a 28-byte header and the chunk, read as one
    buffer and the rest; the writer renames every file into place and the reader writes
    nothing. The writer stores and loads what the local-disk engine did, chunk for chunk; the
    abstract's counts at seed 1 are the low end of its seeds (`ABSTRACTS.md` §8)."""
    kit = BUILDER / "traces" / "kv_cache_shared"
    got = {}
    for who, ast, params in (("writer", "kv_cache_shared", "fitted.params.json"), ("reader", "kv_cache_shared_reader", "fitted.reader.params.json")):
        dry = tmp_path / f"{who}.json"
        r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / f"{ast}.ast.json"), "--gpus", "1", "--seed", "1",
                            "--params", str(kit / params), "--metrics-json", str(dry)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        t, d = json.loads((kit / f"{who}.trace.metrics.json").read_text())["total"], json.loads(dry.read_text())["total"]
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
    # kit's (the same requests, and the engine held the same of each) with the header and the buffer
    pw, pr = (json.loads((kit / f).read_text())["params"] for f in ("fitted.params.json", "fitted.reader.params.json"))
    assert pw == pr
    assert pw == {**json.loads((BUILDER / "traces" / "kv_cache_serving" / "fitted.params.json").read_text())["params"], "meta_bytes": 28, "buf": 1048576}



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
        with pytest.raises(SystemExit):
            trace.load_tolerances(str(f))


# Every traced pair under the default tolerances (`DESIGN_REVIEW.md` §3.55): the trace's
# metrics, the abstract, the fitted parameters, `--gpus`, and the verdict. A pair's
# tolerance file, beside its trace document, names the rows the trace cannot show and
# records, with the reason, the rows that are outside.
KIT_PAIRS = [
    ("train_small_files", "trace", "train_small_files", "fitted.params.json", 1, "accepted"),
    ("train_large_samples", "trace.140MiB", "train_large_samples", "fitted.140MiB.params.json", 1, "accepted"),
    ("train_large_samples", "trace.8MiB", "train_large_samples", "fitted.8MiB.params.json", 1, "accepted"),
    ("ckpt_write_dcp", "trace", "ckpt_write_dcp", "fitted.params.json", 2, "accepted"),
    ("ckpt_restore", "trace.mixed", "ckpt_restore", "fitted.mixed.params.json", 2, "accepted"),
    ("ckpt_restore", "trace.small-last", "ckpt_restore", "fitted.small-last.params.json", 2, "accepted"),
    ("ckpt_restore", "trace.name-order", "ckpt_restore", "fitted.name-order.params.json", 2, "accepted"),
    ("model_load", "trace", "model_load", "fitted.params.json", 1, "nothing judged"),
    ("vdb_search_ivf", "trace", "vdb_search_ivf", "fitted.params.json", 1, "nothing judged"),
    ("vdb_search_diskann", "trace.nocache", "vdb_search_diskann", "fitted.nocache.params.json", 1, "accepted"),
    ("vdb_search_diskann", "trace.cache", "vdb_search_diskann", "fitted.cache.params.json", 1, "accepted"),
    ("vdb_build_diskann", "trace", "vdb_build_diskann", "fitted.params.json", 1, "not accepted"),
    # the three KV-cache pairs are the ShareGPT replay (DESIGN_REVIEW.md §3.56); with one `keep` draw per round of
    # conversations (§3.57) the reuse distance is within, and only the writer's store order (two threads) stays outside
    ("kv_cache_serving", "trace", "kv_cache_serving", "fitted.params.json", 1, "accepted"),
    ("kv_cache_shared", "writer.trace", "kv_cache_shared", "fitted.params.json", 1, "not accepted"),
    ("kv_cache_shared", "reader.trace", "kv_cache_shared_reader", "fitted.reader.params.json", 1, "accepted"),
]


def test_every_trace_of_a_kit_is_a_pair():
    have = {(p.parent.name, p.name[: -len(".metrics.json")]) for p in (BUILDER / "traces").glob("*/*.metrics.json")}
    assert have == {(k, t) for k, t, *_ in KIT_PAIRS}
    files = {(p.parent.name, p.name[: -len(".tolerances.json")]) for p in (BUILDER / "traces").glob("*/*.tolerances.json")}
    assert files <= have


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
@pytest.mark.parametrize("kit, which, ast, params, gpus, verdict", KIT_PAIRS, ids=[f"{k}:{t}" for k, t, *_ in KIT_PAIRS])
def test_kit_pair_against_the_tolerances(tmp_path, capsys, kit, which, ast, params, gpus, verdict):
    """`aeiou-trace compare --judge` on a committed trace and its abstract at the fitted
    parameters, with the abstract at three other seeds as its own spread. The verdict is
    the recorded one, every row outside has its reason in the pair's file, and the file
    has no entry that matches nothing."""
    kit = BUILDER / "traces" / kit
    docs = []
    for seed in (1, 2, 3, 4):
        docs.append(tmp_path / f"dry.{seed}.json")
        r = subprocess.run([str(RUNNER), "dry-run", str(ROOT / "schema" / "examples" / f"{ast}.ast.json"), "--gpus", str(gpus), "--seed", str(seed),
                            "--params", str(kit / params), "--metrics-json", str(docs[-1])], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    spec = kit / f"{which}.tolerances.json"
    args = ["compare", str(kit / f"{which}.metrics.json"), str(docs[0]), *[x for d in docs[1:] for x in ("--self", str(d))]]
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


@pytest.mark.parametrize("text", [SEQUENTIAL, WORKERS])
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
