"""Format classes: the ports of the runner's definitions, the class specs and probes,
`aeiou-datagen` writing real containers that the libraries read back and that match the
layout's geometry, and, when the runner binary is built (`AEIOU_RUNNER` or
runner/target/release/aeiou), `aeiou run` executing the new abstracts over the generated
corpus to the dry-run fingerprint."""
import json
import os
import pathlib
import struct
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
EXAMPLES = ROOT / "schema" / "examples"
sys.path.insert(0, str(BUILDER))
pytest.importorskip("pyarrow")
pytest.importorskip("h5py")
pytest.importorskip("dgen_py")

from aeiou import *  # noqa: E402,F401
from aeiou import emit, formats  # noqa: E402
from aeiou.layout import FileGeometry, Layout  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402
from aeiou.pattern import Pattern  # noqa: E402
from aeiou.rng import Words, labeled_key, sample_size  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
needs_runner = pytest.mark.skipif(not RUNNER.exists(), reason="runner binary not built")


def _tool(mod, args, cwd=BUILDER):
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    return subprocess.run([sys.executable, "-m", f"aeiou.{mod}", *map(str, args)], capture_output=True, text=True, cwd=str(cwd), env=env)


def test_pattern_port():
    p = Pattern("train/{id div 1300:05}/img_{id:09}.jpg")
    assert p.format(id=1300) == "train/00001/img_000001300.jpg"
    assert p.root() == "train"
    assert Pattern("kv/{conv:016x}/blk_{k:04}").format(conv=-1, k=7) == "kv/ffffffffffffffff/blk_0007"
    assert Pattern("d/{id mod 16}/x").format(id=17) == "d/1/x"
    assert Pattern("model-{id:05}.safetensors").root() == ""


def test_words_and_keys_are_the_runners():
    # SplitMix64 from key 0: the first output is mix64(golden), a published value
    w = Words(0)
    assert w.next_u64() == 0xE220A8397B1DCDAF
    assert 0 <= Words(1).next_f64() < 1
    k = labeled_key(7, "payload", [1, 2])
    assert k == labeled_key(7, "payload", [1, 2]) != labeled_key(7, "payload", [2, 1])


def test_layout_geometry_by_hand_matches_the_rust_test():
    # tests/layout.rs `framed_layout_geometry_by_hand`: three 64-row groups, two columns
    spec = {"unit": 64, "file_header": 4, "file_footer": 200, "file_footer_per_unit": 120,
            "columns": [{"header": 26, "fixed": 4, "weight": 1.0}, {"header": 22, "fixed": 8, "weight": 0}]}
    lay = Layout.from_spec(spec, 192, int)
    geo = FileGeometry(lay, [100000] * 192)
    col0, col1 = 26 + 64 * (4 + 100000), 22 + 64 * 8
    unit = col0 + col1
    assert geo.units == 3 and geo.col_lens[0] == [col0, col1] and geo.unit_lens == [unit] * 3
    assert geo.unit_offsets == [4, 4 + unit, 4 + 2 * unit]
    assert geo.col_offset(1, 1) == 4 + unit + col0
    assert geo.file_size == 4 + 3 * unit + 200 + 3 * 120
    # tar: 512-byte headers, data padded to 512, 1024 footer, 10240 alignment
    tar = Layout.from_spec({"file_footer": 1024, "file_align": 10240, "columns": [{"row_header": 512, "row_align": 512, "weight": 1}]}, 10, int)
    assert FileGeometry(tar, [1000] * 10).file_size == 20480
    assert FileGeometry(tar, [1000] * 5).file_size == 10240
    assert FileGeometry(tar, [1000] * 10).sample_offset(1) == 1536 + 512


def test_specs_probes_and_v9():
    pq = formats.parquet(rows_per_group=64)
    spec = pq.spec("stream", const(100000), {})
    assert spec["class"] == "parquet" and spec["version"].startswith("pyarrow ")
    lay = spec["layout"]
    assert lay["unit"] == 64 and lay["file_header"] == 4
    assert lay["columns"] == [{"header": 26, "fixed": 4, "weight": 1.0}, {"header": 22, "fixed": 8, "weight": 0}]
    assert lay["writer"] == {"rows_per_group": 64, "columns": [["image", "binary"], ["label", "int64"]]}
    with pytest.raises(BuildError, match="V9"):
        pq.spec("map", const(100000), {})
    assert formats.parquet_page_header(5, 1) == 19
    assert formats.parquet_page_header(64 * 12, 64) == 22
    assert formats.parquet_page_header(64 * 100004, 64) == 26
    assert formats.parquet_page_header(1000 * 1028, 1000) == 24
    assert formats.parquet_page_header(10000 * 1028, 10000) == 27
    h5 = formats.hdf5(dataset="records", shape=(64, 64, 3), dtype="u1")
    assert h5.row_bytes == 12288
    assert h5.spec("map", const(12288), {})["layout"]["file_header"] == 2048
    with pytest.raises(BuildError, match="fixed-size"):
        h5.spec("map", const(100), {})
    with pytest.raises(BuildError, match="V9"):
        formats.tfrecord().spec("map", const(10), {})
    assert formats.webdataset().spec("stream", const(10), {})["layout"]["columns"][0]["row_footer"] == 1024
    assert formats.tfrecord().spec("stream", lognormal(median=110 * KiB, sigma=0.45), {})["layout"]["columns"] == [{"row_header": 12, "row_footer": 4, "weight": 1}]
    with pytest.raises(BuildError, match="binary"):
        formats.parquet(columns=(("a", "int64"),))
    assert formats.FormatClass.from_spec(spec).rows == 64
    assert formats.expected_sample_bytes(lognormal(median=1000, sigma=0.001), {}) == pytest.approx(1000.0, abs=0.01)
    assert formats.expected_sample_bytes({"normal": {"mean": {"param": "m"}, "sd": 1}}, {"m": 42}) == 42.0


# ---- datagen round trips ----

CASES = {
    "train_stream_tfrecord": ["samples=768", "per_shard=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"],
    "train_stream_parquet": ["samples=768", "per_shard=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"],
    "train_map_hdf5": ["samples=64", "per_file=16", "batch=4", "workers=2", "steps=4"],
}


def _datagen(name, tmp_path, params):
    root = tmp_path / "root"
    r = _tool("datagen", [EXAMPLES / f"{name}.ast.json", "--root", root, "--threads", "3", *sum([["--param", p] for p in params], [])])
    assert r.returncode == 0, r.stdout + r.stderr
    return root, r.stdout


def _parse_tfrecord(path):
    import crc32c
    out = []
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                return out
            (n,) = struct.unpack("<Q", head)
            (c,) = struct.unpack("<I", f.read(4))
            assert c == formats._masked_crc32c(head)
            data = f.read(n)
            (c2,) = struct.unpack("<I", f.read(4))
            assert c2 == formats._masked_crc32c(data)
            out.append(len(data))


def test_datagen_tfrecord_round_trip(tmp_path):
    root, out = _datagen("train_stream_tfrecord", tmp_path, CASES["train_stream_tfrecord"])
    files = sorted((root / "train").glob("*.tfrecord"))
    assert len(files) == 6
    ast = emit.load(EXAMPLES / "train_stream_tfrecord.ast.json")
    seed = ast["datasets"]["shards"]["files"]["seed"]
    sizes = _parse_tfrecord(files[2])
    assert sizes == [sample_size({"lognormal": {"median": 20000, "sigma": 0.45}}, seed, 2 * 128 + i) for i in range(128)]
    m = json.loads((root / "train" / ".aeiou-dataset.json").read_text())
    assert m["manifest_version"] == 1 and m["format"]["class"] == "tfrecord" and m["payload"]["generator"] == "dgen-data"
    assert m["dataset"]["files"]["count"] == 768 and m["dataset"]["files"]["samples_per_file"] == 128
    assert "doc" not in m["dataset"]["files"]
    assert m["provenance"]["files_written"] == 6
    assert "id " in out
    # a second datagen into the same root is refused: datasets are read-only
    r = _tool("datagen", [EXAMPLES / "train_stream_tfrecord.ast.json", "--root", root, *sum([["--param", p] for p in CASES["train_stream_tfrecord"]], [])])
    assert r.returncode == 1 and "not empty" in r.stderr


def test_datagen_parquet_reads_back_with_pyarrow(tmp_path):
    import pyarrow.parquet as pq
    root, _ = _datagen("train_stream_parquet", tmp_path, CASES["train_stream_parquet"])
    files = sorted((root / "train").glob("*.parquet"))
    assert len(files) == 6
    t = pq.read_table(files[1])
    assert t.num_rows == 128 and t.column_names == ["image", "label"]
    ast = emit.load(EXAMPLES / "train_stream_parquet.ast.json")
    seed = ast["datasets"]["shards"]["files"]["seed"]
    lens = [len(v.as_py()) for v in t.column("image")]
    assert lens == [sample_size({"lognormal": {"median": 20000, "sigma": 0.45}}, seed, 128 + i) for i in range(128)]
    md = pq.ParquetFile(files[1]).metadata
    assert md.num_row_groups == 2
    lay = ast["datasets"]["shards"]["files"]["format"]["layout"]
    footer = int.from_bytes(files[1].read_bytes()[-8:-4], "little")
    assert footer == lay["file_footer"] - 8 + lay["file_footer_per_unit"] * 2, "the thrift footer is padded to the declared size"
    assert files[1].stat().st_size == 4 + sum(
        sum(pq.ParquetFile(files[1]).metadata.row_group(r).column(c).total_compressed_size for c in range(2)) for r in range(2)) + footer + 8


def test_datagen_hdf5_reads_back_with_h5py(tmp_path):
    import h5py
    import numpy as np
    root, _ = _datagen("train_map_hdf5", tmp_path, CASES["train_map_hdf5"])
    files = sorted((root / "train").glob("*.h5"))
    assert len(files) == 4
    with h5py.File(files[3], "r") as h:
        d = h["records"]
        assert d.shape == (16, 256, 256, 3) and d.dtype == np.uint8
        assert d.id.get_offset() == 2048
        row = d[5]
        assert row.any(), "payload, not zeros"
    assert files[3].stat().st_size == 2048 + 16 * 256 * 256 * 3


@needs_runner
@pytest.mark.parametrize("name", sorted(CASES))
def test_runner_executes_the_generated_corpus(name, tmp_path):
    params = CASES[name]
    root, _ = _datagen(name, tmp_path, params)
    ast_path = EXAMPLES / f"{name}.ast.json"
    flags = sum([["--param", p] for p in params], [])
    dry = subprocess.run([RUNNER, "dry-run", ast_path, "--gpus", "2", "--seed", "3", *flags], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    fp = [l.split()[1] for l in dry.stdout.splitlines() if l.startswith("fingerprint ")][0]
    run = subprocess.run([RUNNER, "run", ast_path, "--root", root, "--gpus", "2", "--seed", "3", "--time-scale", "0",
                          "--expect-fingerprint", fp, *flags], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "fingerprint matches" in run.stdout
    if name == "train_stream_parquet":
        assert "fadvise" in run.stdout
    # the Rust writer refuses a dataset that has a format class
    r = subprocess.run([RUNNER, "datagen", ast_path, "--root", tmp_path / "other", *flags], capture_output=True, text=True)
    assert r.returncode != 0 and "format class" in r.stderr


@needs_runner
def test_size_draws_match_the_runner(tmp_path):
    """The Python `sample_size` port against sizes the Rust datagen wrote."""
    ast = {"ast": "0.2", "name": "sizes", "datasets": {"d": {"files": {"pattern": "d/{id:04}", "count": 40, "seed": 99,
           "size": {"mixture": [{"weight": 1, "dist": {"normal": {"mean": 50000, "sd": 20000, "min": 100}}},
                                {"weight": 1, "dist": {"lognormal": {"median": 30000, "sigma": 0.7, "min": 1, "max": 90000}}},
                                {"weight": 1, "dist": {"uniform": {"lo": 10, "hi": 5000}}},
                                {"weight": 1, "dist": {"empirical": {"values": [7, 8, 9], "weights": [1, 2, 3]}}}]}}}},
           "actors": {"gpu": {"body": [{"stat": {"file": {"file": {"dataset": "d", "id": 0}}}}]}}}
    p = tmp_path / "s.ast.json"
    p.write_text(json.dumps(ast))
    r = subprocess.run([RUNNER, "datagen", p, "--root", tmp_path / "r"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for i in range(40):
        assert (tmp_path / "r" / "d" / f"{i:04}").stat().st_size == sample_size(ast["datasets"]["d"]["files"]["size"], 99, i), i
