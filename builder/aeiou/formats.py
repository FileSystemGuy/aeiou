"""Format classes (`GRAMMAR_OPTIONS.md` §6.4, `DESIGN_REVIEW.md` §3.16, §3.28): the container
formats a dataset can be stored in, each a library with two halves.

The **format contract** is what a trace of the named reader library shows regardless of the
application that drives it: the *framing* (`schema/README.md` §2, *Container framing*: how the
file frames its samples, from which the runner computes every offset), the reader's fixed
*protocol* as ordinary POSIX nodes emitted on a cursor, the *access modes* it supports (V9),
and the *writer* that `aeiou-datagen` uses to produce real files with the runner's payload,
into a byte stream, so a file goes to a directory or an object store with no copy on a local
file system (`DESIGN_REVIEW.md` §3.69).
The **loader contract** stays in the abstract: access mode, how many shards a worker streams,
column projection, batch composition.

Each class is pinned to a reader library and, where the constants come from the library, to
its version: the constants a framing needs (a Parquet page header's bytes, an HDF5 file's data
offset, a Parquet footer's growth per row group) are probed at build time by writing a
one-row file in memory with the installed library, so the AST records the version the framing
was derived for, and the writer checks every file it produces against the formula. The
protocols were read off `strace` of the libraries on the loopback NFS mount of
`runner/REFERENCE.md` §7 (2026-09-30, pyarrow 25.0.1, h5py 3.16 / HDF5 2.0, CPython 3.12's
`tarfile`); TFRecord's is from TensorFlow's source and is tagged [verify].

    from aeiou.formats import tfrecord, parquet, hdf5, webdataset
    shards = w.dataset("shards", pattern="train/shard-{id:05}.tfrecord", layout="compound", access="stream",
                       count=P.samples, samples_per_file=P.samples_per_file, size=lognormal(110 * KiB, 0.45),
                       seed=0x5eed_0001, format=tfrecord(xfer=P.xfer))
    with gpu.loader("shards", workers=P.cycle, prefetch=1, batches=P.shards) as worker:
        s = worker.let("s", shards.consume())        # a shard: stream datasets consume files
        shards.format.stream(worker, s)              # open, sequential reads to EOF, close

The libraries (`pyarrow`, `h5py`, `numpy`, `crc32c`) are the `formats` extra of the package;
a class imports its library when first used and names the extra if it is missing.
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import threading

from .nodes import BuildError, Dist, Handle, Node, Param, ast_of, lift
from .units import KiB, MiB


def _lib(name: str):
    try:
        return __import__(name)
    except ImportError:
        raise BuildError(f"format class needs `{name}`: install the builder's `formats` extra "
                         f"(`uv sync --extra formats` or `pip install 'aeiou[formats]'`)") from None


def _plain_int(x, what: str, lo: int = 0) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < lo:
        raise BuildError(f"{what} must be a plain integer >= {lo} (it fixes the framing at build time), got {x!r}")
    return x


def varint_len(v: int) -> int:
    """Bytes of an unsigned LEB128 varint."""
    n = 1
    while v >= 128:
        v >>= 7
        n += 1
    return n


def expected_sample_bytes(size, params: dict | None) -> float:
    """The mean of a size distribution, with parameter references resolved to their defaults:
    what a class needs to place a framing constant in the right band."""
    params = params or {}

    def num(v):
        if isinstance(v, Param):
            v = params.get(v.name)
        if isinstance(v, dict) and len(v) == 1 and "param" in v:
            v = params.get(v["param"])
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BuildError(f"a format class needs size distribution arguments that are numbers or parameters, got {v!r}")
        return float(v)

    if isinstance(size, Param):
        size = params.get(size.name)
    d = size.ast() if isinstance(size, Dist) else size
    if not isinstance(d, dict) or len(d) != 1:
        raise BuildError(f"not a size distribution: {size!r}")
    kind, a = next(iter(d.items()))
    if kind == "const":
        return num(a)
    if kind == "uniform":
        return (num(a["lo"]) + num(a["hi"])) / 2
    if kind == "normal":
        return num(a["mean"])
    if kind == "lognormal":
        return num(a["median"]) * math.exp(float(a["sigma"]) ** 2 / 2)
    if kind == "empirical":
        ws = [num(w) for w in a["weights"]]
        return sum(num(v) * w for v, w in zip(a["values"], ws)) / sum(ws)
    if kind == "mixture":
        arms = [(float(arm["weight"]), arm["dist"]) for arm in a if arm["dist"] is not None]
        return sum(w * expected_sample_bytes(dd, params) for w, dd in arms) / sum(w for w, _ in arms)
    raise BuildError(f"cannot take the mean of a `{kind}` size distribution")


class Sink:
    """The byte stream a writer writes into: counts the bytes and answers `tell` for libraries
    that ask (`tarfile`, pyarrow), over `out` (a file, an object upload) or over nothing (a
    count alone); keeps the last 8 bytes (a Parquet footer's length)."""

    closed = False
    mode = "wb"

    def __init__(self, out=None):
        self.out, self.n, self.tail = out, 0, b""

    def write(self, b):
        m = memoryview(b).cast("B")
        if self.out is not None:
            self.out.write(m)
        self.n += len(m)
        self.tail = (self.tail + bytes(m[-8:]))[-8:]
        return len(m)

    def tell(self):
        return self.n

    def flush(self):
        pass

    def close(self):
        pass

    def writable(self):
        return True

    def seekable(self):
        return False


class FormatClass:
    """Base of the format classes. Subclasses set `name`, `reader`, `modes`, and implement
    `framing`, the protocol methods, and the writer."""
    name = ""
    reader = ""
    modes: frozenset = frozenset()

    def version(self) -> str | None:
        return None

    def framing(self, mean_bytes: float) -> dict:
        raise NotImplementedError

    def writer_settings(self) -> dict:
        """What `aeiou-datagen` needs to rebuild this class from the AST (`framing.writer`)."""
        raise NotImplementedError

    def spec(self, access: str, size=None, params: dict | None = None) -> dict:
        """The dataset's `format` entry. Refuses an access mode the class does not support (V9)."""
        if access not in self.modes:
            raise BuildError(f"format class `{self.name}` supports {sorted(self.modes)} access, not `{access}` "
                             f"(schema/README.md V9, GRAMMAR_OPTIONS.md §6.2)")
        self.check_size(size, params)
        mean = expected_sample_bytes(size, params) if size is not None else 0.0
        framing = dict(self.framing(mean))
        framing["writer"] = self.writer_settings()
        spec = {"class": self.name, "reader": self.reader}
        v = self.version()
        if v:
            spec["version"] = v
        spec["framing"] = framing
        return spec

    def check_size(self, size, params):
        pass

    # ---- writer ----
    def library(self) -> str:
        raise NotImplementedError

    def write_file(self, out, first_id: int, sizes: list, payload, geo):
        """Write one container holding samples `first_id …` of the given payload sizes into
        `out`, a writable byte stream that need not seek (a `Sink`); `payload(logical_offset,
        n)` returns the sample bytes of the file's payload stream (`payload.pieces(offset, n)`
        the same bytes as views of its blocks, with no copy). Returns what `check_file`
        needs of the writer beyond the byte count (the library's own record of the framing)."""
        raise NotImplementedError

    def check_file(self, name: str, written: int, sizes: list, geo, info) -> None:
        """Compare the file as written with the geometry the framing predicts; raise on drift.
        `written` is the bytes the stream took, `info` what `write_file` returned."""
        if written != geo.file_size:
            raise BuildError(f"{name}: {written} bytes written, the framing predicts {geo.file_size}: the format class's framing "
                             f"does not match its writer (library version?)")

    @classmethod
    def from_spec(cls, spec: dict) -> "FormatClass":
        """Rebuild the class instance `aeiou-datagen` needs from a dataset's `format` entry."""
        name = spec["class"]
        klass = CLASSES.get(name)
        if klass is None:
            raise BuildError(f"unknown format class `{name}` (known: {sorted(CLASSES)})")
        return klass.from_writer(spec.get("framing", {}).get("writer", {}))

    @classmethod
    def from_writer(cls, writer: dict) -> "FormatClass":
        return cls(**writer)


# ----------------------------------------------------------------------------------------------
# TFRecord
# ----------------------------------------------------------------------------------------------

def _masked_crc32c(data: bytes) -> int:
    c = _lib("crc32c").crc32c(data)
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


class TfRecord(FormatClass):
    """TFRecord: records back to back, each `u64 length ‖ masked crc32c(length) ‖ data ‖ masked
    crc32c(data)`, 16 bytes of framing per record; no index, so `stream` only. The reader is
    `tf.data.TFRecordDataset(buffer_size)`: positioned reads of `buffer_size` from offset 0 to
    the short read at EOF [verify: TensorFlow is not installed here; from
    `tensorflow/core/lib/io/record_reader.cc` and `PosixRandomAccessFile::Read`, the default
    buffer is 256 KiB]."""
    name = "tfrecord"
    reader = "tf.data.TFRecordDataset"
    modes = frozenset({"stream"})

    def __init__(self, xfer=256 * KiB):
        self.xfer = lift(xfer, "tfrecord xfer")

    def framing(self, mean_bytes):
        return {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}

    def writer_settings(self):
        return {}

    def stream(self, cur, f: Handle):
        """The reader's whole-shard protocol."""
        cur.open(f, "RDONLY")
        cur.read(f, self.xfer, offset=0, repeat="until_eof")
        cur.close(f)

    def library(self):
        return f"aeiou tfrecord writer (crc32c {_lib('crc32c').__version__})"

    def write_file(self, out, first_id, sizes, payload, geo):
        off = 0
        for s in sizes:
            data = payload(off, s)
            off += s
            length = struct.pack("<Q", s)
            out.write(length)
            out.write(struct.pack("<I", _masked_crc32c(length)))
            out.write(data)
            out.write(struct.pack("<I", _masked_crc32c(data)))


# ----------------------------------------------------------------------------------------------
# WebDataset (tar)
# ----------------------------------------------------------------------------------------------

class WebDataset(FormatClass):
    """WebDataset: a POSIX tar of `{id}.jpg`, `{id}.cls`, … per sample (512-byte headers, data
    padded to 512, two zero blocks, the archive padded to 10240). Streamed through Python's
    `tarfile` over `open(path, 'rb')`: the BufferedReader issues `read(st_blksize)` until EOF
    (traced 2026-09-30 on the NFS mount: 1 MiB reads, a short one, a zero one). `members` are
    `(extension, bytes)` pairs; `None` bytes means the sample itself."""
    name = "webdataset"
    reader = "tarfile (CPython)"
    modes = frozenset({"stream"})

    def __init__(self, members=(("jpg", None), ("cls", 4)), xfer=1 * MiB):
        self.members = [(str(ext), None if n is None else _plain_int(n, f"member {ext} bytes")) for ext, n in members]
        if sum(1 for _, n in self.members if n is None) != 1:
            raise BuildError("webdataset: exactly one member is the sample (bytes None)")
        self.xfer = lift(xfer, "webdataset xfer")

    def framing(self, mean_bytes):
        extra = sum(512 + (n + 511) // 512 * 512 for _, n in self.members if n is not None)
        return {"file_footer": 1024, "file_align": 10240,
                "columns": [{"row_header": 512, "row_footer": extra, "row_align": 512, "weight": 1}]}

    def writer_settings(self):
        return {"members": [[ext, n] for ext, n in self.members]}

    @classmethod
    def from_writer(cls, writer):
        return cls(members=[(e, n) for e, n in writer.get("members", [["jpg", None], ["cls", 4]])])

    def stream(self, cur, f: Handle):
        """`open()` through `io.open` then `tarfile` in streaming mode: the Python file protocol."""
        cur.open(f, "RDONLY|CLOEXEC")
        cur.fstat(f)
        cur.ioctl(f, "TCGETS", expect=["ENOTTY"])
        cur.lseek(f, 0, "CUR")
        cur.read(f, self.xfer, repeat="until_eof")
        cur.close(f)

    def library(self):
        import platform
        return f"tarfile (CPython {platform.python_version()})"

    def write_file(self, out, first_id, sizes, payload, geo):
        import io
        import tarfile
        with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as tf:
            off = 0
            for i, s in enumerate(sizes):
                sid = first_id + i
                for ext, n in self.members:
                    data = payload(off, s) if n is None else bytes(n)
                    if n is None:
                        off += s
                    ti = tarfile.TarInfo(f"{sid:09}.{ext}")
                    ti.size = len(data)
                    ti.mtime = 0
                    tf.addfile(ti, io.BytesIO(data))


# ----------------------------------------------------------------------------------------------
# HDF5
# ----------------------------------------------------------------------------------------------

class Hdf5(FormatClass):
    """HDF5: one contiguous dataset `name` of shape `(samples, *shape)` and `dtype` at the root,
    so every sample is a fixed-size row at `data_offset + i × row`. The data offset is probed
    at build time (an in-memory file with the `core` driver lays out metadata as the disk
    driver does; 2048 for a rank-4 `u1` dataset named `records`). The reader is h5py's
    `f[name][i]`: eight small metadata reads at open (the superblock, the root group's object
    header, heap, B-tree and symbol node, the dataset's object header; traced 2026-09-30, h5py
    3.16 / HDF5 2.0, hard-coded for this one-dataset structure), then one `pread` per sample
    through libhdf5's 64 KiB sieve buffer: `min(64 KiB, EOF − offset)` for a row smaller than
    the sieve, the row itself otherwise. Sieve hits between neighbouring rows are not modeled
    (a cut: they depend on access order). `map` only."""
    name = "hdf5"
    reader = "h5py"
    modes = frozenset({"map"})
    META_READS = ((8, 0), (48, 0), (48, 48), (512, 96), (512, 680), (544, 136), (328, 1072), (512, 800))

    def __init__(self, dataset="records", shape=(64, 64, 3), dtype="u1", sieve=64 * KiB):
        self.dataset = str(dataset)
        self.shape = tuple(_plain_int(x, "hdf5 shape", 1) for x in shape)
        self.dtype = str(dtype)
        self.sieve = _plain_int(sieve, "hdf5 sieve", 1)
        self._offset = None
        self._headers, self._lock = {}, threading.Lock()

    @property
    def row_bytes(self) -> int:
        np = _lib("numpy")
        n = np.dtype(self.dtype).itemsize
        for x in self.shape:
            n *= x
        return n

    def version(self):
        h5py = _lib("h5py")
        return f"h5py {h5py.__version__}, hdf5 {h5py.version.hdf5_version}"

    def data_offset(self) -> int:
        if self._offset is None:
            h5py = _lib("h5py")
            with h5py.File(f"aeiou-probe-{os.getpid()}.h5", "w", driver="core", backing_store=False) as h:
                d = h.create_dataset(self.dataset, shape=(4, *self.shape), dtype=self.dtype)
                d[0] = _lib("numpy").zeros(self.shape, self.dtype)   # a scalar broadcasts slowly (0.6 s a row)
                self._offset = int(d.id.get_offset())
        return self._offset

    def check_size(self, size, params):
        if size is None:
            return
        d = size.ast() if isinstance(size, Dist) else size
        if not (isinstance(d, dict) and "const" in d and d["const"] == self.row_bytes):
            raise BuildError(f"hdf5: samples are fixed-size rows; the dataset's size must be const({self.row_bytes}) for "
                             f"shape {self.shape} {self.dtype}, got {d!r}")

    def framing(self, mean_bytes):
        return {"file_header": self.data_offset()}

    def writer_settings(self):
        return {"dataset": self.dataset, "shape": list(self.shape), "dtype": self.dtype, "sieve": self.sieve}

    def open_reads(self, cur, f: Handle):
        """`h5py.File(path, 'r')` and the first access to the dataset."""
        cur.open(f, "RDONLY|CLOEXEC")
        cur.fstat(f)
        for n, off in self.META_READS:
            cur.read(f, n, offset=off)

    def read_sample(self, cur, f: Handle, s: Handle):
        from .dists import min_
        if self.row_bytes < self.sieve:
            cur.read(f, min_(self.sieve, f.size - s.offset), offset=s.offset)
        else:
            cur.read(f, s.size, offset=s.offset)

    def close(self, cur, f: Handle):
        cur.close(f)

    def library(self):
        return self.version()

    def header(self, n: int) -> bytes:
        """The file's first `data_offset` bytes for `n` rows, from libhdf5: a file of the same
        dataset written to a sparse scratch file (in /dev/shm where there is one) with one row
        stored, its metadata read back, the file removed. The metadata of a contiguous dataset
        with no filter and no fill does not depend on the rows' bytes (the superblock records
        the end of file, so it depends on `n`), so this is byte for byte the start of the file
        with every row written (verified 2026-10-11, h5py 3.16 / HDF5 2.0: an 18 GiB file
        costs 260 KiB of scratch and 0.65 s). Cached per row count: a dataset has at most two."""
        with self._lock:
            got = self._headers.get(n)
        if got is not None:
            return got
        h5py = _lib("h5py")
        shm = "/dev/shm"
        fd, path = tempfile.mkstemp(prefix="aeiou-", suffix=".h5",
                                    dir=shm if os.path.isdir(shm) and os.access(shm, os.W_OK) else None)
        os.close(fd)
        try:
            with h5py.File(path, "w") as h:
                d = h.create_dataset(self.dataset, shape=(n, *self.shape), dtype=self.dtype)
                d[0] = _lib("numpy").zeros(self.shape, self.dtype)   # a scalar broadcasts slowly (0.6 s a row)
                off = int(d.id.get_offset())
            with open(path, "rb") as f:
                got = f.read(off)
        finally:
            os.unlink(path)
        if len(got) != off:
            raise BuildError(f"hdf5: read {len(got)} bytes of the {off}-byte header back from the scratch file")
        with self._lock:
            self._headers[n] = got
        return got

    def write_file(self, out, first_id, sizes, payload, geo):
        """The header from libhdf5 (`header`), then the rows straight from the payload's
        blocks: libhdf5 seeks back over what it writes, so it cannot write
        into a stream itself, and h5py's Python file-object driver lays the file out otherwise
        (the data at 1400, not 2048, for the default shape; measured 2026-10-11, h5py 3.16)."""
        row = self.row_bytes
        n = len(sizes)
        head = self.header(n)
        if len(head) != geo.framing.file_header:
            raise BuildError(f"the dataset's data starts at {len(head)}, the framing says {geo.framing.file_header}")
        out.write(head)
        for piece in payload.pieces(0, n * row):
            out.write(piece)


# ----------------------------------------------------------------------------------------------
# Parquet
# ----------------------------------------------------------------------------------------------

# a fixed-width column's bytes per row in a Parquet PLAIN page: its physical type's width (int8,
# int16, and uint8 are INT32 with a logical type, 4 bytes each; found 2026-10-12, when the
# widths were the Arrow types' and the writer's own check refused such a column)
FIXED_WIDTH = {"int64": 8, "int32": 4, "int16": 4, "int8": 4, "uint8": 4, "float64": 8, "float32": 4}


def parquet_page_header(data_bytes: int, rows: int) -> int:
    """Bytes of the thrift-compact page header pyarrow writes for one PLAIN, uncompressed,
    statistics-free data page: 16 + two varints of the (zigzag) page size + one of the row
    count (probed 2026-09-30, pyarrow 25.0.1; the probe below re-derives it at build time)."""
    return 16 + 2 * varint_len(2 * data_bytes) + varint_len(2 * rows)


class Parquet(FormatClass):
    """Parquet as pyarrow writes and reads it: `PAR1`, row groups of `rows_per_group` rows,
    each column chunk one PLAIN uncompressed page (no dictionary, no statistics, no page
    index), a thrift footer padded to a declared size, a 4-byte footer length and `PAR1`. The
    sample is the one `binary` column (a 4-byte length prefix per value); the other columns
    are fixed-width. The reader is `pyarrow.parquet.ParquetFile` (pre_buffer on, the
    default): two `fstat`s, a 64 KiB speculative footer read at EOF − 64 KiB (and the rest of
    the footer if larger), then `posix_fadvise(WILLNEED)` plus one `pread` per coalesced range:
    adjacent row groups merge into pieces of at most 32 MiB (`iter_batches`), and a column
    projection reads one range per run of adjacent projected chunks per group (traced
    2026-09-30, pyarrow 25.0.1 on NFS). `stream` only: a row costs its row group."""
    name = "parquet"
    reader = "pyarrow.parquet.ParquetFile"
    modes = frozenset({"stream"})
    FOOTER_READ = 64 * KiB
    RANGE_LIMIT = 32 * MiB

    def __init__(self, rows_per_group=64, columns=(("image", "binary"), ("label", "int64"))):
        self.rows = _plain_int(rows_per_group, "parquet rows_per_group", 1)
        self.columns = [(str(n), str(t)) for n, t in columns]
        if sum(1 for _, t in self.columns if t == "binary") != 1:
            raise BuildError("parquet: exactly one `binary` column carries the sample")
        for n, t in self.columns:
            if t != "binary" and t not in FIXED_WIDTH:
                raise BuildError(f"parquet: column {n}: type {t} (binary or one of {sorted(FIXED_WIDTH)})")
        self._probe = None

    def version(self):
        return f"pyarrow {_lib('pyarrow').__version__}"

    def footer_base(self) -> int:
        return 128 + 32 * len(self.columns)

    def footer_per_unit(self) -> int:
        return 32 + 64 * len(self.columns)

    def probe(self):
        """Re-derive the page header formula with the installed pyarrow on an in-memory file."""
        if self._probe is None:
            pa = _lib("pyarrow")
            pq = _lib("pyarrow.parquet").parquet
            for rows, length in ((1, 1), (64, 100000), (1000, 1024)):
                t = pa.table({"c": pa.array([b"x" * length] * rows, pa.binary())},
                             schema=pa.schema([pa.field("c", pa.binary(), nullable=False)]))
                buf = pa.BufferOutputStream()
                pq.write_table(t, buf, **self.writer_options(), row_group_size=rows)
                cc = pq.ParquetFile(pa.BufferReader(buf.getvalue())).metadata.row_group(0).column(0)
                data = rows * (4 + length)
                if cc.total_compressed_size - data != parquet_page_header(data, rows):
                    raise BuildError(f"parquet: {self.version()} writes a {cc.total_compressed_size - data}-byte page header for "
                                     f"{rows} rows of {length} bytes, the class predicts {parquet_page_header(data, rows)}: "
                                     f"the formula needs updating for this pyarrow")
            self._probe = True
        return True

    def writer_options(self) -> dict:
        return {"compression": "none", "use_dictionary": False, "write_statistics": False,
                "data_page_size": 1 << 30, "write_page_index": False, "store_schema": False}

    def column_specs(self, mean_bytes: float) -> list:
        """The framing's columns, with page headers for the band the expected chunk sizes fall in."""
        self.probe()
        specs = []
        for _, t in self.columns:
            if t == "binary":
                data = self.rows * (4 + int(round(mean_bytes)))
                specs.append({"header": parquet_page_header(data, self.rows), "fixed": 4, "weight": 1.0})
            else:
                w = FIXED_WIDTH[t]
                specs.append({"header": parquet_page_header(self.rows * w, self.rows), "fixed": w, "weight": 0})
        return specs

    def framing(self, mean_bytes):
        # the file's tail: the thrift footer, padded to footer_base + footer_per_unit × units,
        # then the 4-byte footer length and `PAR1`
        return {"unit": self.rows, "file_header": 4, "file_footer": self.footer_base() + 8,
                "file_footer_per_unit": self.footer_per_unit(), "columns": self.column_specs(mean_bytes)}

    def writer_settings(self):
        return {"rows_per_group": self.rows, "columns": [[n, t] for n, t in self.columns]}

    @classmethod
    def from_writer(cls, writer):
        return cls(rows_per_group=writer["rows_per_group"], columns=[tuple(c) for c in writer["columns"]])

    # ---- protocol ----
    def open_reads(self, cur, f: Handle):
        """`ParquetFile(path)`: the footer."""
        from .dists import min_
        cur.open(f, "RDONLY")
        cur.fstat(f)
        cur.fstat(f)
        tail = min_(self.FOOTER_READ, f.size)
        cur.read(f, tail, offset=f.size - tail)
        footer_total = self.footer_base() + self.footer_per_unit() * f.units + 8
        with cur.when(footer_total > self.FOOTER_READ):
            cur.read(f, footer_total - self.FOOTER_READ, offset=f.size - footer_total)

    def _column_runs(self, columns):
        names = [n for n, _ in self.columns]
        idx = sorted(names.index(c) for c in columns)
        if len(set(idx)) != len(idx) or not idx:
            raise BuildError(f"parquet: projection {columns!r} over columns {names}")
        runs, start = [], idx[0]
        for a, b in zip(idx, idx[1:] + [None]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        return runs

    def read_all(self, cur, f: Handle, columns=None):
        """`iter_batches()` over the whole shard: WILLNEED on every range first, then the
        reads. Full projection coalesces adjacent row groups into pieces of at most 32 MiB
        (pieces are sized from the first row group's length); a projection reads each run of
        adjacent projected column chunks of every group."""
        from .dists import ceil_div, max_, min_
        if columns is None or set(columns) == {n for n, _ in self.columns}:
            k = max_(1, self.RANGE_LIMIT // f.unit(0).size)
            pieces = ceil_div(f.units, k)
            for op in ("fadvise", "read"):
                with cur.loop("piece", pieces) as p:
                    first = cur.let("u0", f.unit(p * k))
                    last = cur.let("u1", f.unit(min_(p * k + k, f.units) - 1))
                    length = last.offset + last.size - first.offset
                    if op == "fadvise":
                        cur.fadvise(f, "WILLNEED", offset=first.offset, len=length)
                    else:
                        cur.read(f, length, offset=first.offset)
            return
        runs = self._column_runs(columns)
        for op in ("fadvise", "read"):
            with cur.loop("rg", f.units) as g:
                u = cur.let("u", f.unit(g))
                for c0, c1 in runs:
                    first = cur.let(f"c{c0}", u.column(c0))
                    last = first if c1 == c0 else cur.let(f"c{c1}", u.column(c1))
                    length = last.offset + last.size - first.offset
                    if op == "fadvise":
                        cur.fadvise(f, "WILLNEED", offset=first.offset, len=length)
                    else:
                        cur.read(f, length, offset=first.offset)

    def close(self, cur, f: Handle):
        cur.close(f)

    # ---- writer ----
    def library(self):
        return self.version()

    def _schema(self):
        pa = _lib("pyarrow")
        return pa.schema([pa.field(n, pa.binary() if t == "binary" else getattr(pa, t)(), nullable=False)
                          for n, t in self.columns])

    def _group(self, schema, first_id, sizes, data):
        """One row group's table: the samples are `data` (their bytes end to end) cut at `sizes`."""
        pa, np = _lib("pyarrow"), _lib("numpy")
        n = len(sizes)
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(sizes, out=offsets[1:])
        arrays = []
        for name, t in self.columns:
            if t == "binary":
                arrays.append(pa.Array.from_buffers(pa.binary(), n, [None, pa.py_buffer(offsets), pa.py_buffer(data)]))
            else:
                dt = getattr(np, t)
                ids = np.arange(first_id, first_id + n, dtype=np.int64)
                arr = (ids % 1000).astype(dt) if t.startswith("int") or t.startswith("uint") else ids.astype(dt)
                arrays.append(pa.array(arr))
        return pa.Table.from_arrays(arrays, schema=schema)

    def _write(self, out, first_id, sizes, data, pad: int):
        """Write the file one row group at a time, `data(offset, n)` giving each group's
        sample bytes; returns the footer's metadata as the writer wrote it."""
        pa, pq = _lib("pyarrow"), _lib("pyarrow.parquet").parquet
        schema = self._schema()
        w = pq.ParquetWriter(pa.PythonFile(out, mode="w"), schema, **self.writer_options())
        off = 0
        for g in range(0, len(sizes), self.rows):
            group = sizes[g:g + self.rows]
            k = sum(group)
            w.write_table(self._group(schema, first_id + g, group, data(off, k)), row_group_size=self.rows)
            off += k
        w.add_key_value_metadata({"aeiou.pad": "p" * pad})
        w.close()
        return w.writer.metadata

    def write_file(self, out, first_id, sizes, payload, geo):
        """Streamed a row group at a time, in two passes: the first, of zero bytes in the
        samples' sizes into a counting sink, measures the natural footer (its bytes do not
        depend on the samples' bytes, since no statistics are written), so the second can pad
        it to the size the framing declares."""
        zeros = bytes(max(sum(sizes[g:g + self.rows]) for g in range(0, len(sizes), self.rows)))
        sink = Sink()
        self._write(sink, first_id, sizes, lambda off, n: memoryview(zeros)[:n], 0)
        footer0 = int.from_bytes(sink.tail[:4], "little")
        target = geo.framing.file_footer - 8 + geo.framing.file_footer_per_unit * geo.units
        pad = target - footer0
        for _ in range(4):
            pad = target - footer0 - (varint_len(pad) - 1)
        if pad < 0:
            raise BuildError(f"the natural footer ({footer0} bytes) exceeds the declared {target}: raise the class's footer margins")
        return self._write(out, first_id, sizes, payload, pad)

    def check_file(self, name, written, sizes, geo, m):
        """`m`: the footer's metadata as the writer wrote it."""
        super().check_file(name, written, sizes, geo, m)
        if m.num_row_groups != geo.units:
            raise BuildError(f"{name}: {m.num_row_groups} row groups, the framing predicts {geo.units}")
        for r in range(m.num_row_groups):
            rg = m.row_group(r)
            for c in range(rg.num_columns):
                cc = rg.column(c)
                want_off, want_len = geo.col_offset(r, c), geo.col_lens[r][c]
                if (cc.data_page_offset, cc.total_compressed_size) != (want_off, want_len):
                    raise BuildError(f"{name}: row group {r} column {c} at {cc.data_page_offset} for {cc.total_compressed_size} bytes; the framing "
                                     f"predicts {want_off} for {want_len} (a page header outside the declared band? keep every "
                                     f"column chunk's bytes within one varint band of the expected size)")


# ----------------------------------------------------------------------------------------------

def tfrecord(**kw) -> TfRecord:
    return TfRecord(**kw)


def parquet(**kw) -> Parquet:
    return Parquet(**kw)


def hdf5(**kw) -> Hdf5:
    return Hdf5(**kw)


def webdataset(**kw) -> WebDataset:
    return WebDataset(**kw)


CLASSES = {c.name: c for c in (TfRecord, Parquet, Hdf5, WebDataset)}
