"""The payload (`aeiou-positional/3`): numpy's SFC64 in the state the runner gives it, laid out
in 4 KiB pieces, the same bytes as `runner/aeiou/src/payload.rs` (whose tests pin the same
values), and, when the runner is built, the same bytes as the files `aeiou datagen` writes."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import zlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
sys.path.insert(0, str(BUILDER))
np = pytest.importorskip("numpy")

from aeiou.datagen import BLOCK, FILL, PIECE, Payload, payload_block, sfc64  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
needs_runner = pytest.mark.skipif(not RUNNER.exists(), reason="runner binary not built")


def test_the_stream_and_blocks_are_the_runners():
    assert [int(v) for v in sfc64(0).random_raw(3)] == [0xEAF73661F5E180BC, 0xBC904E1262DE1088, 0x06538B07830AEE11]
    pinned = {1: "a2eb20e7d329c5ec5dddb0ccd476a040aea1b465701f0a7e1ec8d998a5804258",
              2: "931796b308ff3de12bbecbb615e24e832ed6d21449decda560b566d4fc5e746b",
              3: "f3d76909654cbbbe2e2f4daeba766978879f046852302ac27e63c6a99b1b2612"}
    for c, want in pinned.items():
        assert hashlib.sha256(payload_block(0x123456789ABCDEF0, c).tobytes()).hexdigest() == want, c


@pytest.mark.parametrize("c", [2, 3, 4])
def test_every_piece_and_every_prefix_compresses_at_the_ratio(c):
    b = payload_block(99, c).tobytes()
    assert len(b) == BLOCK
    r = -(-PIECE // c)
    pieces = [b[i:i + PIECE] for i in range(0, BLOCK, PIECE)]
    assert all(p[r:] == bytes([FILL]) * (PIECE - r) for p in pieces)
    assert all(p[:r].count(0) < r // 2 for p in pieces), "no piece is zeros"
    # an image-sized file compresses as a whole block does (`/2` gave such a file ratio 1)
    for n in (110_000, BLOCK):
        ratio = n / len(zlib.compress(b[:n], 1))
        assert c * 0.9 < ratio <= c * 1.02, (n, ratio)
    assert zlib.compress(payload_block(99, 1).tobytes(), 1).__len__() >= BLOCK


@needs_runner
def test_the_rust_writer_writes_the_same_bytes(tmp_path):
    """`aeiou datagen` writes `file_per_sample` with dedupe and compression; every file is the
    Python payload of its unit, so the two writers can share a corpus's definition."""
    ast = next((ROOT / "workloads").rglob("file_per_sample.ast.json"))
    root = tmp_path / "r"
    r = subprocess.run([str(RUNNER), "datagen", str(ast), "--root", str(root), "--param", "files=12", "--dedupe", "2",
                        "--compress", "3", "--writers", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    seed = json.loads(ast.read_text())["datasets"]["train"]["files"]["seed"]
    m = json.loads((root / "train" / ".aeiou-dataset.json").read_text())
    assert m["payload"]["generator"] == "sfc64" and m["payload"]["wrapper"] == "aeiou-positional/3"
    for i in range(12):
        data = (root / "train" / f"{i // 1300:05}" / f"img_{i:09}.jpg").read_bytes()
        assert data == bytes(Payload(seed, i // 2, 3)(0, len(data))), i
