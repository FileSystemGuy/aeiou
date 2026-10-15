"""The workloads' kit scripts (`workloads/**/kit/`): what they read off a trace."""
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys

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
