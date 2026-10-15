"""Format classes: the ports of the runner's definitions, the class specs and probes,
`aeiou-datagen` writing real containers that the libraries read back and that match the
framing's geometry, and, when the runner binary is built (`AEIOU_RUNNER` or
runner/target/release/aeiou), `aeiou run` executing the new abstracts over the generated
corpus to the dry-run fingerprint."""
import io
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
WORKLOADS = ROOT / "workloads"


def ast_path(name):
    """The committed AST `name`, wherever its mechanism lives under workloads/ (DESIGN_REVIEW.md §3.66)."""
    return next(WORKLOADS.rglob(f"{name}.ast.json"))
sys.path.insert(0, str(BUILDER))
pytest.importorskip("pyarrow")
pytest.importorskip("h5py")
pytest.importorskip("numpy")

from aeiou import *  # noqa: E402,F401
from aeiou import emit, formats  # noqa: E402
from aeiou.framing import FileGeometry, Framing  # noqa: E402
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
    lay = Framing.from_spec(spec, 192, int)
    geo = FileGeometry(lay, [100000] * 192)
    col0, col1 = 26 + 64 * (4 + 100000), 22 + 64 * 8
    unit = col0 + col1
    assert geo.units == 3 and geo.col_lens[0] == [col0, col1] and geo.unit_lens == [unit] * 3
    assert geo.unit_offsets == [4, 4 + unit, 4 + 2 * unit]
    assert geo.col_offset(1, 1) == 4 + unit + col0
    assert geo.file_size == 4 + 3 * unit + 200 + 3 * 120
    # tar: 512-byte headers, data padded to 512, 1024 footer, 10240 alignment
    tar = Framing.from_spec({"file_footer": 1024, "file_align": 10240, "columns": [{"row_header": 512, "row_align": 512, "weight": 1}]}, 10, int)
    assert FileGeometry(tar, [1000] * 10).file_size == 20480
    assert FileGeometry(tar, [1000] * 5).file_size == 10240
    assert FileGeometry(tar, [1000] * 10).sample_offset(1) == 1536 + 512


def test_specs_probes_and_v9():
    pq = formats.parquet(rows_per_group=64)
    spec = pq.spec("stream", const(100000), {})
    assert spec["class"] == "parquet" and spec["version"].startswith("pyarrow ")
    lay = spec["framing"]
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
    assert h5.spec("map", const(12288), {})["framing"]["file_header"] == 2048
    with pytest.raises(BuildError, match="fixed-size"):
        h5.spec("map", const(100), {})
    with pytest.raises(BuildError, match="V9"):
        formats.tfrecord().spec("map", const(10), {})
    assert formats.webdataset().spec("stream", const(10), {})["framing"]["columns"][0]["row_footer"] == 1024
    assert formats.tfrecord().spec("stream", lognormal(median=110 * KiB, sigma=0.45), {})["framing"]["columns"] == [{"row_header": 12, "row_footer": 4, "weight": 1}]
    with pytest.raises(BuildError, match="binary"):
        formats.parquet(columns=(("a", "int64"),))
    # the narrow integers are INT32 in a PLAIN page, 4 bytes a row
    narrow = formats.parquet(columns=(("image", "binary"), ("a", "int8"), ("b", "uint8"), ("c", "int16"))).spec("stream", const(1000), {})
    assert [c["fixed"] for c in narrow["framing"]["columns"]] == [4, 4, 4, 4]
    assert formats.FormatClass.from_spec(spec).rows == 64
    assert formats.expected_sample_bytes(lognormal(median=1000, sigma=0.001), {}) == pytest.approx(1000.0, abs=0.01)
    assert formats.expected_sample_bytes({"normal": {"mean": {"param": "m"}, "sd": 1}}, {"m": 42}) == 42.0


# ---- datagen round trips ----

CASES = {
    "tfrecord_stream": ["samples=768", "samples_per_file=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"],
    "parquet_stream": ["samples=768", "samples_per_file=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"],
    "hdf5_map": ["samples=64", "samples_per_file=16", "batch=4", "workers=2", "steps=4"],
}


def _datagen(name, tmp_path, params):
    root = tmp_path / "root"
    r = _tool("datagen", [ast_path(name), "--root", root, "--writers", "3", *sum([["--param", p] for p in params], [])])
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
    root, out = _datagen("tfrecord_stream", tmp_path, CASES["tfrecord_stream"])
    files = sorted((root / "train").glob("*.tfrecord"))
    assert len(files) == 6
    ast = emit.load(ast_path("tfrecord_stream"))
    seed = ast["datasets"]["shards"]["seed"]
    sizes = _parse_tfrecord(files[2])
    assert sizes == [sample_size({"lognormal": {"median": 20000, "sigma": 0.45}}, seed, 2 * 128 + i) for i in range(128)]
    m = json.loads((root / "train" / ".aeiou-dataset.json").read_text())
    assert m["manifest_version"] == 3 and m["state"] == "complete" and m["format"]["class"] == "tfrecord" and m["payload"]["generator"] == "sfc64" and m["payload"]["wrapper"] == "aeiou-positional/3"
    assert m["dataset"]["count"] == 768 and m["dataset"]["samples_per_file"] == 128
    assert "doc" not in m["dataset"]
    assert m["provenance"]["files_written"] == 6
    assert "id " in out
    # a second datagen into the same root is refused: datasets are read-only
    r = _tool("datagen", [ast_path("tfrecord_stream"), "--root", root, *sum([["--param", p] for p in CASES["tfrecord_stream"]], [])])
    assert r.returncode == 1 and "not empty" in r.stderr


def test_datagen_parquet_reads_back_with_pyarrow(tmp_path):
    import pyarrow.parquet as pq
    root, _ = _datagen("parquet_stream", tmp_path, CASES["parquet_stream"])
    files = sorted((root / "train").glob("*.parquet"))
    assert len(files) == 6
    t = pq.read_table(files[1])
    assert t.num_rows == 128 and t.column_names == ["image", "label"]
    ast = emit.load(ast_path("parquet_stream"))
    seed = ast["datasets"]["shards"]["seed"]
    lens = [len(v.as_py()) for v in t.column("image")]
    assert lens == [sample_size({"lognormal": {"median": 20000, "sigma": 0.45}}, seed, 128 + i) for i in range(128)]
    md = pq.ParquetFile(files[1]).metadata
    assert md.num_row_groups == 2
    lay = ast["datasets"]["shards"]["format"]["framing"]
    footer = int.from_bytes(files[1].read_bytes()[-8:-4], "little")
    assert footer == lay["file_footer"] - 8 + lay["file_footer_per_unit"] * 2, "the thrift footer is padded to the declared size"
    assert files[1].stat().st_size == 4 + sum(
        sum(pq.ParquetFile(files[1]).metadata.row_group(r).column(c).total_compressed_size for c in range(2)) for r in range(2)) + footer + 8


def test_datagen_hdf5_reads_back_with_h5py(tmp_path):
    import h5py
    import numpy as np
    root, _ = _datagen("hdf5_map", tmp_path, CASES["hdf5_map"])
    files = sorted((root / "train").glob("*.h5"))
    assert len(files) == 4
    with h5py.File(files[3], "r") as h:
        d = h["records"]
        assert d.shape == (16, 256, 256, 3) and d.dtype == np.uint8
        assert d.id.get_offset() == 2048
        row = d[5]
        assert row.any(), "payload, not zeros"
    assert files[3].stat().st_size == 2048 + 16 * 256 * 256 * 3


def _geometry(klass, access, size, sizes):
    from aeiou.datagen import ev
    return FileGeometry(Framing.from_spec(klass.spec(access, size)["framing"], len(sizes), ev), sizes)


def test_streamed_hdf5_and_parquet_are_the_files_the_libraries_write(tmp_path):
    """The writers stream (DESIGN_REVIEW.md §3.69): HDF5 as libhdf5's header then the rows,
    Parquet a row group at a time; each file is byte for byte the one the library writes
    when handed every row at once."""
    import io
    import h5py
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from aeiou.datagen import Payload
    h = formats.hdf5(shape=(16, 16, 3))
    for n in (1, 6, 16):
        sizes = [h.row_bytes] * n
        out = io.BytesIO()
        h.write_file(formats.Sink(out), 0, sizes, Payload(7, 0, 1), _geometry(h, "map", const(h.row_bytes), sizes))
        path = tmp_path / f"whole-{n}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("records", data=np.frombuffer(Payload(7, 0, 1)(0, n * h.row_bytes), dtype="u1").reshape((n, 16, 16, 3)))
        assert out.getvalue() == path.read_bytes(), f"{n} rows"
    p = formats.parquet(rows_per_group=8)
    sizes = [sample_size({"lognormal": {"median": 3000, "sigma": 0.45}}, 11, i) for i in range(32)]
    out = io.BytesIO()
    m = p.write_file(formats.Sink(out), 100, sizes, Payload(7, 0, 1), _geometry(p, "stream", lognormal(3000, 0.45), sizes))
    p.check_file("streamed", len(out.getvalue()), sizes, _geometry(p, "stream", lognormal(3000, 0.45), sizes), m)
    streamed = out.getvalue()
    t = pq.read_table(io.BytesIO(streamed))
    whole = pa.BufferOutputStream()
    w = pq.ParquetWriter(whole, t.schema, **p.writer_options())
    w.write_table(t, row_group_size=8)
    w.add_key_value_metadata({"aeiou.pad": pq.ParquetFile(io.BytesIO(streamed)).metadata.metadata[b"aeiou.pad"].decode()})
    w.close()
    assert streamed == whole.getvalue().to_pybytes()


def test_the_writers_hold_a_piece_of_a_file_not_the_file():
    """Peak memory while writing a 256 MiB file stays a small fraction of it."""
    script = """
import resource, sys
sys.path.insert(0, %r)
from aeiou import formats
from aeiou.datagen import Payload, ev
from aeiou.framing import FileGeometry, Framing
from aeiou.dists import const, lognormal
def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
def geo(k, access, size, sizes):
    return FileGeometry(Framing.from_spec(k.spec(access, size)["framing"], len(sizes), ev), sizes)
h = formats.hdf5()
p = formats.parquet()
small = [h.row_bytes] * 2
h.write_file(formats.Sink(), 0, small, Payload(1, 0, 1), geo(h, "map", const(h.row_bytes), small))
s = [64 * 1024] * 128
p.write_file(formats.Sink(), 0, s, Payload(1, 0, 1), geo(p, "stream", lognormal(64 * 1024, 0.0001), s))
base = peak()
rows = [h.row_bytes] * ((256 << 20) // h.row_bytes)
h.write_file(formats.Sink(), 0, rows, Payload(1, 0, 1), geo(h, "map", const(h.row_bytes), rows))
s = [64 * 1024] * 4096
p.write_file(formats.Sink(), 0, s, Payload(1, 0, 1), geo(p, "stream", lognormal(64 * 1024, 0.0001), s))
print(peak() - base)
""" % str(BUILDER)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) < 64 << 20, f"{int(r.stdout) >> 20} MiB above the warm process for two 256 MiB files"


@needs_runner
@pytest.mark.parametrize("name", sorted(CASES))
def test_runner_executes_the_generated_corpus(name, tmp_path):
    params = CASES[name]
    root, _ = _datagen(name, tmp_path, params)
    ast_file = ast_path(name)
    flags = sum([["--param", p] for p in params], [])
    dry = subprocess.run([RUNNER, "dry-run", ast_file, "--gpus", "2", "--seed", "3", *flags], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    fp = [l.split()[1] for l in dry.stdout.splitlines() if l.startswith("fingerprint ")][0]
    # the thread-per-actor backends and both event-loop engines issue the same ops over the same corpus
    for backend in ("sync", "io_uring", "posix-aio", "libaio", "mmap"):
        run = subprocess.run([RUNNER, "run", ast_file, "--root", root, "--gpus", "2", "--seed", "3", "--time-scale", "0",
                              "--posix", backend, "--expect-fingerprint", fp, *flags], capture_output=True, text=True)
        assert run.returncode == 0, backend + ": " + run.stdout + run.stderr
        assert "fingerprint matches" in run.stdout
        if name == "parquet_stream":
            assert "fadvise" in run.stdout


def webdataset_ast(tmp_path, members=(("jpg", None), ("cls", 4), ("json", 700))):
    """`tfrecord_stream` with its shards a WebDataset tar (no committed abstract has one)."""
    a = json.loads(ast_path("tfrecord_stream").read_text())
    f = a["datasets"]["shards"]
    f["pattern"] = "train/shard-{id:05}.tar"
    f["format"] = formats.webdataset(members=members).spec("stream", f["size"], {k: p["default"] for k, p in a["params"].items()})
    a["name"] = "webdataset_stream"
    p = tmp_path / "webdataset_stream.ast.json"
    p.write_text(json.dumps(a))
    return p


def parquet_types_ast(tmp_path):
    """`parquet_stream` with a column of every fixed-width type around the sample's."""
    a = json.loads(ast_path("parquet_stream").read_text())
    f = a["datasets"]["shards"]
    cols = (("a", "int8"), ("b", "uint8"), ("c", "int16"), ("image", "binary"), ("d", "float32"), ("e", "int32"), ("g", "float64"), ("h", "int64"))
    f["format"] = formats.parquet(rows_per_group=64, columns=cols).spec("stream", f["size"], {k: p["default"] for k, p in a["params"].items()})
    a["name"] = "parquet_types"
    p = tmp_path / "parquet_types.ast.json"
    p.write_text(json.dumps(a))
    return p


# each writer's case: the abstract, its corpus parameters (a short last file), and the run's
RUST_CASES = {
    "tfrecord": (lambda tmp: ast_path("tfrecord_stream"), ["samples=700", "samples_per_file=128", "sample_median=20000"], ["batch=32", "steps=4"]),
    "webdataset": (webdataset_ast, ["samples=700", "samples_per_file=128", "sample_median=20000"], ["batch=32", "steps=4"]),
    "hdf5": (lambda tmp: ast_path("hdf5_map"), ["samples=70", "samples_per_file=16"], ["batch=4", "workers=2", "steps=4"]),
    # whole row groups and no short last file (brief item 31)
    "parquet": (lambda tmp: ast_path("parquet_stream"), ["samples=768", "samples_per_file=128", "sample_median=20000"], ["batch=32", "steps=4"]),
    "parquet_types": (parquet_types_ast, ["samples=768", "samples_per_file=128", "sample_median=20000"], ["batch=32", "steps=4"]),
}


def _params(names):
    return sum([["--param", p] for p in names], [])


@needs_runner
@pytest.mark.parametrize("name", sorted(RUST_CASES))
def test_aeiou_datagen_writes_the_python_writers_bytes(name, tmp_path):
    """`aeiou datagen` built with the cargo feature `formats` (DESIGN_REVIEW.md §3.70): every
    TFRecord, WebDataset, HDF5, and Parquet file is the Python writer's byte for byte (a
    Parquet footer but for its `created_by` and the pad that makes up for it), with dedupe,
    compression, and a short last file; the manifest the same definition and payload, its
    `format.writer` the Rust writer (so the dataset id differs). HDF5's headers come from
    `aeiou-header` beside the binary."""
    make, corpus, run_params = RUST_CASES[name]
    ast_file = make(tmp_path)
    params = _params(corpus)
    flags = [*params, "--dedupe", "2", "--compress", "3"]
    rs = subprocess.run([RUNNER, "datagen", ast_file, "--root", tmp_path / "rs", "--writers", "3", *flags], capture_output=True, text=True)
    if "--features formats" in rs.stderr:
        pytest.skip(f"{RUNNER} is not built with the cargo feature `formats`")
    assert rs.returncode == 0, rs.stdout + rs.stderr
    assert f"({name.split('_')[0]}, aeiou datagen " in rs.stdout
    py = _tool("datagen", [ast_file, "--root", tmp_path / "py", "--writers", "3", *flags])
    assert py.returncode == 0, py.stdout + py.stderr
    _same_files(tmp_path / "rs", tmp_path / "py", 6 if name != "hdf5" else 5)
    m_rs, m_py = (json.loads((tmp_path / w / "train" / ".aeiou-dataset.json").read_text()) for w in ("rs", "py"))
    for k in ("manifest_version", "dataset", "payload"):
        assert m_rs[k] == m_py[k], k
    assert {k: v for k, v in m_rs["format"].items() if k != "writer"} == {k: v for k, v in m_py["format"].items() if k != "writer"}
    assert m_rs["format"]["writer"].startswith("aeiou datagen ")
    if name == "hdf5":
        assert set(m_rs["provenance"]["headers"]["sha256"]) == {"16", "6"}
    # the runner reads what it wrote to the dry run's fingerprint
    run_flags = [*params, *_params(run_params)]
    dry = subprocess.run([RUNNER, "dry-run", ast_file, "--gpus", "2", "--seed", "3", *run_flags], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    fp = [l.split()[1] for l in dry.stdout.splitlines() if l.startswith("fingerprint ")][0]
    run = subprocess.run([RUNNER, "run", ast_file, "--root", tmp_path / "rs", "--gpus", "2", "--seed", "3", "--time-scale", "0",
                          "--expect-fingerprint", fp, *run_flags], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "fingerprint matches" in run.stdout


def _unfinish(root, ranges, delete=(), shorten=()):
    """Make a complete corpus look like one a crash left: the manifest `writing` with `ranges`
    unfinished, some of those files gone and some cut short."""
    path = root / ".aeiou-dataset.json"
    m = json.loads(path.read_text())
    m["state"] = "writing"
    m["progress"] = {"unfinished": ranges, "files_written": 0, "bytes_written": 0, "updated": 0}
    path.write_text(json.dumps(m))
    files = sorted(p for p in root.iterdir() if p.name != ".aeiou-dataset.json")
    for i in delete:
        files[i].unlink()
    for i in shorten:
        with open(files[i], "r+b") as f:
            f.truncate(files[i].stat().st_size // 2)
    return m


@needs_runner
@pytest.mark.parametrize("name", sorted(RUST_CASES))
def test_aeiou_datagen_resumes_a_container_corpus(name, tmp_path):
    """`aeiou datagen --resume ROOT` (DESIGN_REVIEW.md §3.70) rewrites the files the manifest
    lists as unfinished, whole, from the manifest alone, the HDF5 headers made again by
    `aeiou-header` and checked against the corpus's; the files are those of a datagen that ran
    through, and the dataset id is unchanged. A manifest naming another format writer is
    refused before anything is written."""
    make, corpus, _ = RUST_CASES[name]
    ast_file = make(tmp_path)
    root = tmp_path / "rs" / "train"
    rs = subprocess.run([RUNNER, "datagen", ast_file, "--root", tmp_path / "rs", "--writers", "3", *_params(corpus), "--dedupe", "2", "--compress", "3"],
                        capture_output=True, text=True)
    if "--features formats" in rs.stderr:
        pytest.skip(f"{RUNNER} is not built with the cargo feature `formats`")
    assert rs.returncode == 0, rs.stdout + rs.stderr
    whole = {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".aeiou-dataset.json"}
    before = json.loads((root / ".aeiou-dataset.json").read_text())
    _unfinish(root, [[1, 3], [4, 5]], delete=[1], shorten=[2, 4])
    r = subprocess.run([RUNNER, "datagen", "--resume", root, "--writers", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3 file(s) unfinished" in r.stdout
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".aeiou-dataset.json"} == whole
    after = json.loads((root / ".aeiou-dataset.json").read_text())
    assert after["state"] == "complete" and "progress" not in after
    assert {k: after[k] for k in ("dataset", "payload", "format")} == {k: before[k] for k in ("dataset", "payload", "format")}
    assert after["provenance"]["resumes"][0]["files_written"] == 3
    if name == "hdf5":
        assert after["provenance"]["headers"] == before["provenance"]["headers"]
    # another writer's corpus is not finished by this one
    m = _unfinish(root, [[0, 1]])
    m["format"]["writer"] = "another writer"
    (root / ".aeiou-dataset.json").write_text(json.dumps(m))
    r = subprocess.run([RUNNER, "datagen", "--resume", root], capture_output=True, text=True)
    assert r.returncode != 0 and "a resume would mix them" in r.stderr, r.stdout + r.stderr
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".aeiou-dataset.json"} == whole


# a grow from a short last file and a shrink to one (Parquet: whole files, brief item 31)
RESIZES = {"tfrecord": (900, 300), "webdataset": (900, 300), "hdf5": (100, 40), "parquet": (1024, 512), "parquet_types": (1024, 512)}


@needs_runner
@pytest.mark.parametrize("name", sorted(RUST_CASES))
def test_aeiou_datagen_grows_and_shrinks_a_container_corpus(name, tmp_path):
    """`aeiou datagen --grow N ROOT` and `--shrink N ROOT` (DESIGN_REVIEW.md §3.70) leave the
    corpus a datagen of the new count writes, byte for byte: the old short last file written
    again with its new rows (an HDF5 header among them), the files past a shrink removed; the
    manifest the new count's, its headers the new count's."""
    make, corpus, _ = RUST_CASES[name]
    ast_file = make(tmp_path)

    def fresh(where, samples):
        params = [p if not p.startswith("samples=") else f"samples={samples}" for p in corpus]
        r = subprocess.run([RUNNER, "datagen", ast_file, "--root", where, "--writers", "3", *_params(params), "--dedupe", "2"], capture_output=True, text=True)
        if "--features formats" in r.stderr:
            pytest.skip(f"{RUNNER} is not built with the cargo feature `formats`")
        assert r.returncode == 0, r.stdout + r.stderr
        return where / "train"

    def files(root):
        return {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".aeiou-dataset.json"}

    root = fresh(tmp_path / "rs", int(next(p for p in corpus if p.startswith("samples=")).split("=")[1]))
    for op, n in zip(("--grow", "--shrink"), RESIZES[name]):
        r = subprocess.run([RUNNER, "datagen", op, str(n), root, "--writers", "2"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        want = fresh(tmp_path / f"fresh{n}", n)
        assert files(root) == files(want), f"{op} {n}"
        got, exp = (json.loads((d / ".aeiou-dataset.json").read_text()) for d in (root, want))
        assert got["state"] == "complete" and "progress" not in got
        assert {k: got[k] for k in ("dataset", "payload", "format")} == {k: exp[k] for k in ("dataset", "payload", "format")}
        assert got["provenance"].get("headers") == exp["provenance"].get("headers")
        assert got["provenance"]["files_written"] == exp["provenance"]["files_written"]
        assert got["provenance"]["bytes_written"] == exp["provenance"]["bytes_written"]
    assert [x["op"] for x in got["provenance"]["resizes"]] == ["grow", "shrink"]


@needs_runner
@pytest.mark.parametrize("name", sorted(RUST_CASES))
def test_aeiou_datagen_settles_an_interrupted_grow(name, tmp_path):
    """`aeiou datagen --settle ROOT` (DESIGN_REVIEW.md §3.70) keeps what an interrupted grow
    reached: the count is the claim boundary's, the file in flight is written again, the stray
    past the boundary removed, and the corpus is, byte for byte, a fresh datagen's of the
    settled count (its HDF5 header that count's, checked against the corpus's)."""
    make, corpus, _ = RUST_CASES[name]
    ast_file = make(tmp_path)

    def fresh(where, samples):
        params = [p if not p.startswith("samples=") else f"samples={samples}" for p in corpus]
        r = subprocess.run([RUNNER, "datagen", ast_file, "--root", where, "--writers", "3", *_params(params), "--dedupe", "2"], capture_output=True, text=True)
        if "--features formats" in r.stderr:
            pytest.skip(f"{RUNNER} is not built with the cargo feature `formats`")
        assert r.returncode == 0, r.stdout + r.stderr
        return where / "train"

    def files(root):
        return {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".aeiou-dataset.json"}

    c0 = int(next(p for p in corpus if p.startswith("samples=")).split("=")[1])
    root = fresh(tmp_path / "rs", c0)
    f0 = len(files(root))
    n = RESIZES[name][0]
    r = subprocess.run([RUNNER, "datagen", "--grow", str(n), root, "--writers", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # what a grow killed after handing out the run that ends at file c leaves: file c - 1 in
    # flight (cut short), file c written past the last window (a stray), nothing beyond
    names = sorted(files(root))
    c = f0 + 1
    assert c < len(names), names
    path = root / ".aeiou-dataset.json"
    m = json.loads(path.read_text())
    done = [root / x for x in names[:c - 1]]
    m["state"] = "growing"
    for k in ("finished", "ranks", "hosts", "direct", "files_written", "bytes_written"):
        m["provenance"]["resizes"][-1].pop(k, None)
    m["progress"] = {"unfinished": [[c - 1, len(names)]], "files_written": len(done), "bytes_written": sum(p.stat().st_size for p in done),
                     "from": c0, "claimed": c, "claimed_to": c + 1, "updated": 0, "progress_secs": 30}
    path.write_text(json.dumps(m))
    with open(root / names[c - 1], "r+b") as f:
        f.truncate((root / names[c - 1]).stat().st_size // 2)
    for x in names[c + 1:]:
        (root / x).unlink()
    r = subprocess.run([RUNNER, "datagen", "--settle", root, "--writers", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    got = json.loads(path.read_text())
    s = got["dataset"]["count"]
    assert c0 < s < n
    want = fresh(tmp_path / f"fresh{s}", s)
    assert files(root) == files(want), f"settled at {s}"
    exp = json.loads((want / ".aeiou-dataset.json").read_text())
    assert got["state"] == "complete" and "progress" not in got
    assert {k: got[k] for k in ("dataset", "payload", "format")} == {k: exp[k] for k in ("dataset", "payload", "format")}
    assert got["provenance"].get("headers") == exp["provenance"].get("headers")
    assert (got["provenance"]["files_written"], got["provenance"]["bytes_written"]) == (exp["provenance"]["files_written"], exp["provenance"]["bytes_written"])
    assert got["provenance"]["resizes"][-1]["settled"] == s and got["provenance"]["settles"][-1]["asked"] == n


def _parquet_parts(b):
    """A Parquet file as its row groups and its footer with `created_by` and the `aeiou.pad`
    value taken out (the two things the Rust and Python writers' files differ in)."""
    import pyarrow.parquet as pq
    n = int.from_bytes(b[-8:-4], "little")
    footer = b[-8 - n:-8]
    created = pq.ParquetFile(io.BytesIO(b)).metadata.created_by.encode()
    assert len(created) < 128 and footer.count(bytes([len(created)]) + created) == 1
    footer = footer.replace(bytes([len(created)]) + created, b"")
    footer, k = re.subn(rb"\taeiou\.pad\x18[\x80-\xff]*[\x00-\x7f]p*\x00", b"", footer)
    assert k == 1
    return b[:-8 - n], footer, b[-8:]


def _same_files(a, b, n):
    """The trees `a` and `b` hold the same `n` data files, byte for byte (Parquet: but for the
    footer's `created_by` and pad), and a manifest each."""
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert len(files) == n + 1
    for f in files:
        if f.name == ".aeiou-dataset.json":
            continue
        x, y = (a / f).read_bytes(), (b / f).read_bytes()
        if f.suffix == ".parquet":
            assert len(x) == len(y), f
            x, y = _parquet_parts(x), _parquet_parts(y)
        assert x == y, f


@needs_runner
def test_hdf5_headers_come_from_rank_0_alone(tmp_path):
    """`aeiou-header` is found beside the binary and nowhere else: a copy of `aeiou` alone in a
    directory cannot write HDF5 on one host, yet runs rank 1 of two, since rank 0 makes the
    headers and `Start` carries them (DESIGN_REVIEW.md §3.70, revised 2026-10-12)."""
    import shutil
    import socket
    make, corpus, _ = RUST_CASES["hdf5"]
    flags = _params(corpus)
    if not (RUNNER.parent / "aeiou-header").exists():
        pytest.skip(f"no aeiou-header beside {RUNNER}")
    lone = tmp_path / "lone" / "aeiou"
    lone.parent.mkdir()
    shutil.copy2(RUNNER, lone)
    alone = subprocess.run([lone, "datagen", make(tmp_path), "--root", tmp_path / "x", *flags], capture_output=True, text=True)
    if "--features formats" in alone.stderr:
        pytest.skip(f"{RUNNER} is not built with the cargo feature `formats`")
    assert alone.returncode == 1 and f"{lone.parent / 'aeiou-header'} not found" in alone.stderr, alone.stderr
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        addr = f"127.0.0.1:{s.getsockname()[1]}"
    procs = [subprocess.Popen([exe, "datagen", make(tmp_path), "--root", tmp_path / "two", "--writers", "2", *flags, "--ranks", "2",
                               "--rank", str(r), "--coordinator", addr], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for r, exe in ((0, RUNNER), (1, lone))]
    outs = [p.communicate(timeout=120) for p in procs]
    for p, (o, e) in zip(procs, outs):
        assert p.returncode == 0, o + e
    assert "header(s) from" in outs[0][0] and "header(s) from" not in outs[1][0]
    one = subprocess.run([RUNNER, "datagen", make(tmp_path), "--root", tmp_path / "one", *flags], capture_output=True, text=True)
    assert one.returncode == 0, one.stderr
    _same_files(tmp_path / "two", tmp_path / "one", 5)


def test_aeiou_header_refuses_another_library_version():
    from aeiou import header
    fmt = formats.hdf5(dataset="records", shape=(4, 4, 3), dtype="u1").spec("map", const(48), {})
    assert len(header.header(fmt, 5)) == fmt["framing"]["file_header"]
    with pytest.raises(BuildError, match="probed with h5py 0.0"):
        header.header({**fmt, "version": "h5py 0.0, hdf5 0.0"}, 5)
    with pytest.raises(BuildError, match="only hdf5"):
        header.header({"class": "tfrecord"}, 5)
    out = subprocess.run([sys.executable, "-m", "aeiou.header", "--format", json.dumps(fmt), "--rows", "5"], capture_output=True,
                         cwd=str(BUILDER), env={**os.environ, "PYTHONPATH": str(BUILDER)})
    assert out.returncode == 0 and out.stdout == header.header(fmt, 5) and out.stderr == b""


@needs_runner
def test_an_endpoint_places_a_container_dataset_for_both_writers_and_the_runner(tmp_path):
    """`--endpoint NAME=DIR` (DESIGN_REVIEW.md §3.65): the Python writer puts the dataset's root at DIR,
    the runner finds it there, and the fingerprint is the one of the corpus under --root."""
    name, params = "tfrecord_stream", CASES["tfrecord_stream"]
    flags = sum([["--param", p] for p in params], [])
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    r = _tool("datagen", [ast_path(name), "--root", root, "--endpoint", f"shards={elsewhere}", "--writers", "3", *flags])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (elsewhere / ".aeiou-dataset.json").exists() and not any(root.iterdir())
    ast_file = ast_path(name)
    dry = subprocess.run([RUNNER, "dry-run", ast_file, "--gpus", "2", "--seed", "3", *flags], capture_output=True, text=True)
    fp = [l.split()[1] for l in dry.stdout.splitlines() if l.startswith("fingerprint ")][0]
    run = subprocess.run([RUNNER, "run", ast_file, "--root", root, "--endpoint", f"shards={elsewhere}", "--gpus", "2", "--seed", "3",
                          "--time-scale", "0", "--expect-fingerprint", fp, *flags], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "fingerprint matches" in run.stdout and f"endpoint shards  root train/  at {elsewhere}" in run.stdout
    bad = _tool("datagen", [ast_file, "--root", root, "--endpoint", "nope=/x", *flags])
    assert bad.returncode == 2 and "no dataset or namespace `nope`" in bad.stderr


@needs_runner
def test_size_draws_match_the_runner(tmp_path):
    """The Python `sample_size` port against sizes the Rust datagen wrote."""
    ast = {"ast": "0.10", "name": "sizes", "datasets": {"d": {"pattern": "d/{id:04}", "layout": "whole", "access": "indexed", "count": 40, "seed": 99,
           "size": {"mixture": [{"weight": 1, "dist": {"normal": {"mean": 50000, "sd": 20000, "min": 100}}},
                                {"weight": 1, "dist": {"lognormal": {"median": 30000, "sigma": 0.7, "min": 1, "max": 90000}}},
                                {"weight": 1, "dist": {"uniform": {"lo": 10, "hi": 5000}}},
                                {"weight": 1, "dist": {"empirical": {"values": [7, 8, 9], "weights": [1, 2, 3]}}}]}}},
           "actors": {"gpu": {"body": [{"stat": {"file": {"file": {"dataset": "d", "id": 0}}}}]}}}
    p = tmp_path / "s.ast.json"
    p.write_text(json.dumps(ast))
    r = subprocess.run([RUNNER, "datagen", p, "--root", tmp_path / "r"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for i in range(40):
        assert (tmp_path / "r" / "d" / f"{i:04}").stat().st_size == sample_size(ast["datasets"]["d"]["size"], 99, i), i


def test_npz_framing_of_the_installed_numpy_is_the_abstracts_default():
    """`npy_per_sample` takes `framing` and `cd_len` as parameters whose defaults were read
    off a trace (498 and 102 for `np.savez(x=<uint8 volume>, y=[0])`). They are facts of NumPy's
    writer (the padded `.npy` header, the forced zip64 local headers) and of the member names, so
    this writes such an archive in memory with the installed NumPy and compares. A failure here
    means the defaults, and any parameter file fitted with them, describe another NumPy."""
    import io
    import zipfile
    np = pytest.importorskip("numpy")
    ast = json.loads(ast_path("npy_per_sample").read_text())
    want = {k: ast["params"][k]["default"] for k in ("framing", "cd_len")}
    for shape in ((256, 256, 16), (256, 256, 300)):          # independent of the volume's size
        x = np.zeros(shape, np.uint8)
        b = io.BytesIO()
        np.savez(b, x=x, y=np.array([0]))
        size, cd = len(b.getvalue()), zipfile.ZipFile(b).start_dir
        assert {"framing": size - x.nbytes, "cd_len": size - cd - 22} == want, np.__version__


def test_rust_round_is_rusts_round_then_saturating_cast():
    """`rust_round` is `x.round() as i64` (eval.rs, V10): half away from zero with no `x + 0.5`
    precision loss, then saturation, NaN to 0. Values checked against a compiled `rustc` program."""
    from aeiou.rng import rust_round

    i64_max, i64_min = (1 << 63) - 1, -(1 << 63)
    cases = [
        (0.49999999999999994, 0), (-0.49999999999999994, 0),
        (4503599627370497.0, 4503599627370497), (-4503599627370497.0, -4503599627370497),
        (1e30, i64_max), (-1e30, i64_min), (float("nan"), 0),
        (float("inf"), i64_max), (float("-inf"), i64_min),
        (9.223372036854775807e18, i64_max), (-9.223372036854775808e18, i64_min),
        (2.5, 3), (-2.5, -3), (0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2),
        (-0.0, 0), (3.0, 3), (20000.4999, 20000),
    ]
    for x, want in cases:
        got = rust_round(x)
        assert got == want and type(got) is int, (x, got, want)
