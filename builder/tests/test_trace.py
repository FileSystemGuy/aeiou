"""`aeiou-trace`: the metrics of an `strace` against their definitions (`runner/README.md`
§10) on hand-written traces, the ports of the runner's histogram and stack distance, the
comparison's distances, and, when `strace` and the runner binary are there, the trace of
`aeiou run` against the same abstract's `aeiou dry-run --metrics-json`: everything that
does not depend on the order must be equal."""
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
        ("kv_cache_serving", 5, ["sys_prompts=3", "sys_tokens=6", "chunk_bytes=262144", "concurrency=3", "warm=4", "requests=8"]),
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
    r = run("strace", "-f", "-ttt", "-T", "-yy", "-e", "trace=%file,%desc,%process", "-o", st, RUNNER, "run", *common, "--seed", seed, "--root", root, "--time-scale", "0")
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
    listing = {"open": 2, "fstat": 2, "readdir": 2, "close": 2}       # glob over the two directories
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
