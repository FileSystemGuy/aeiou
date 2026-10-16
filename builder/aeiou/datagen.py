"""`aeiou-datagen`: write the container datasets an abstract declares, through their format
classes, with the runner's payload (`runner/REFERENCE.md` §5: `aeiou-positional/4`, an SFC64
stream from numpy laid out in 4 KiB pieces), and the manifest `aeiou run` checks (`schema/README.md` §6).

    aeiou-datagen AST --root DIR [--params-file FILE]… [--param k=v]… [--gpus G]
                      [--dedupe D] [--compress C] [--writers N] [--dataset NAME]…
                      [--ranks R --rank r --coordinator HOST:PORT] [--config FILE]

`--root`, `--endpoint`, `--writers`, `--rank`, `--ranks`, and `--coordinator` come through the
runner's option layers (`options.py`: the command line, else `AEIOU_<FLAG>`, else the `[datagen]` table of the TOML file `--config`
or `AEIOU_CONFIG` names, else the default); the rest is the command line's alone, and the
block of what was resolved, with each value's source, is printed as `aeiou datagen` prints it.

Datasets without a format class are `aeiou datagen`'s (the Rust writer); this tool skips
them, and `aeiou datagen` refuses the ones with a class (built with the cargo feature
`formats`, it writes TFRecord, WebDataset, and HDF5 too, these bytes), so a corpus is written by both, each
dataset by exactly one of them. Names, sizes, offsets, and bytes follow the runner's
definitions bit for bit (`rng.py`, `pattern.py`, `framing.py`); every file is checked against
the geometry its framing predicts before the manifest is written.

A dataset declared `protocol: s3` is written to the object store its endpoint names
(`--endpoint NAME=s3://BUCKET[/PREFIX]`, the store from the environment as the runner's
object engine has it, `runner/REFERENCE.md` §16) through `obstore`, the Python bindings of
the runner's `object_store`: each file streams from its format class's writer into an upload
(one `PUT`, or a multipart upload in parts of 8 MiB), never through a local file system, and
the manifest is one `PUT` at the prefix (`DESIGN_REVIEW.md` §3.69). Every writer streams,
so what a writer holds is a piece of its file (a Parquet row group, a 1 MiB payload block), never
the file.

On several hosts (`--ranks`, 2026-10-11) each rank writes its contiguous slice of every
dataset's file ids, as `aeiou datagen` does (`runner/REFERENCE.md` §5), through the runner's
coordinator protocol (`coord.py`): every rank's identity is checked before anything is
written, no rank writes before every rank has checked its roots, and rank 0 writes the
manifests once every rank has reported, so a manifest means the whole corpus is there.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import ctypes
import hashlib
import json
import multiprocessing
import os
import pathlib
import platform
import socket
import sys
import tempfile
import time

from . import __version__, coord, emit, options, params as params_mod, usage
from .formats import FormatClass, Sink
from .framing import FileGeometry, Framing
from .nodes import BuildError
from .pattern import Pattern
from .rng import block_seed, sample_size

MANIFEST_NAME = ".aeiou-dataset.json"
PART = 8 << 20   # the runner's `object::DEFAULT_PART`: what `aeiou datagen` sends a part at
MANIFEST_VERSION = 3   # `state` since 2026-10-12 (DESIGN_REVIEW.md §3.70), contract 0.8's dataset since 2026-10-13 (§3.72); this writer writes it once, `complete`
BLOCK = 1 << 20
GENERATOR, GENERATOR_VERSION, WRAPPER = "sfc64", "1", "aeiou-positional/4"
DEDUPE_SETS = 8   # payload.rs `DEDUPE_SETS`: a group of one-sample files is 8 × D ids
PIECE, FILL = 4096, 0x5A
_M64 = (1 << 64) - 1


# ---- parameters and the resolved definition (payload.rs: param_values, substitute, resolved_dataset) ----

def param_values(ast: dict, sets: list, overrides: list, gpus: int) -> dict:
    values = {k: p["default"] for k, p in ast.get("params", {}).items()}
    for pset in sets:
        values.update(pset["params"])
    for name, text in overrides:
        try:
            values[name] = json.loads(text)
        except ValueError:
            values[name] = text
    values["gpus"] = gpus
    return values


def substitute(v, values: dict):
    if isinstance(v, dict):
        if len(v) == 1 and "param" in v:
            if v["param"] not in values:
                raise BuildError(f"param `{v['param']}` has no value")
            return copy.deepcopy(values[v["param"]])
        return {k: substitute(x, values) for k, x in v.items()}
    if isinstance(v, list):
        return [substitute(x, values) for x in v]
    return v


def canonical_json(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def resolved_dataset(ast: dict, name: str, values: dict) -> dict:
    entry = substitute(ast["datasets"][name], values)
    entry["pattern"] = Pattern(entry["pattern"]).bind(values).source    # contract 0.10: the width it was made at
    entry.pop("doc", None)
    entry.pop("protocol", None)   # where the dataset lives, not what it is
    return json.loads(canonical_json(entry))


def ev(x) -> int:
    """Evaluate a dataset-level expression (parameters already substituted) to an int."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(round(x))
    if isinstance(x, dict) and len(x) == 1:
        k, a = next(iter(x.items()))
        if k in ("add", "sub", "mul", "div", "mod", "ceil_div", "min", "max"):
            p, q = ev(a[0]), ev(a[1])
            return {"add": p + q, "sub": p - q, "mul": p * q, "div": p // q if q else 0, "mod": p % q if q else 0,
                    "ceil_div": -(-p // q) if q else 0, "min": min(p, q), "max": max(p, q)}[k]
        if k == "neg":
            return -ev(a)
    raise BuildError(f"cannot evaluate {x!r} at datagen time (dataset-level expressions are over parameters and literals)")


# ---- payload ----

def sfc64(seed: int):
    """numpy's `SFC64` in the state `payload.rs` `Sfc64::new(seed)` gives it: `a`, `b`, `c` the
    first three SplitMix64 outputs from `seed`, the counter 1, twelve outputs discarded."""
    import numpy as np
    x, state = seed, []
    for _ in range(3):
        x = (x + 0x9E3779B97F4A7C15) & _M64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        state.append(z ^ (z >> 31))
    g = np.random.SFC64(0)
    g.state = {"bit_generator": "SFC64", "state": {"state": np.array(state + [1], dtype=np.uint64)}, "has_uint32": 0, "uinteger": 0}
    g.random_raw(12)
    return g


def payload_block(seed: int, compress: int):
    """The 1 MiB block seeded `seed` (`aeiou-positional/3`, `payload.rs` `Filler`): in each 4 KiB
    piece the next `⌈4096 / C⌉` bytes of the SFC64 stream, each output little-endian, then
    `0x5A`; at `C = 1` the stream as is. A new array, never refilled."""
    import numpy as np
    r = -(-PIECE // max(1, compress))
    pieces = BLOCK // PIECE
    raw = sfc64(seed).random_raw(-(-pieces * r // 8)).astype("<u8", copy=False).view(np.uint8)
    if r == PIECE:
        return raw
    out = np.full((pieces, PIECE), FILL, dtype=np.uint8)
    out[:, :r] = raw[:pieces * r].reshape(pieces, r)
    return out.reshape(-1)


def dedupe_unit(size: dict, seed: int, file: int, dedupe: int, one_sample: bool) -> int:
    """The payload unit of file `file` (`aeiou-positional/4`, payload.rs `Units`): `file div D`,
    or for files of one sample the set of `D` files of near sizes in the group of
    `DEDUPE_SETS × D` ids ordered by (size draw, id), whatever the count."""
    d = max(1, dedupe)
    if d == 1 or not one_sample:
        return file // d
    g = DEDUPE_SETS * d
    first = file // g * g
    order = sorted((max(0, sample_size(size, seed, i)), i) for i in range(first, first + g))
    rank = next(r for r, (_, i) in enumerate(order) if i == file)
    return first // g * DEDUPE_SETS + rank // d


class Payload:
    """The bytes of one file's payload stream: 1 MiB blocks seeded `block_seed(seed, unit, b)`
    (`payload_block`), one block cached."""

    def __init__(self, seed: int, unit: int, compress: int):
        self.seed, self.unit, self.compress = seed, unit, compress
        self._b, self._block = None, b""

    def block(self, b: int):
        if self._b != b:
            # a new buffer per block, never refilled, so the views handed out stay valid
            self._b, self._block = b, payload_block(block_seed(self.seed, self.unit, b), self.compress)
        return self._block

    def pieces(self, off: int, n: int):
        """The bytes at `off` as views of the blocks, no copy."""
        while n > 0:
            b, i = divmod(off, BLOCK)
            take = min(n, BLOCK - i)
            yield memoryview(self.block(b))[i:i + take]
            off += take
            n -= take

    def __call__(self, off: int, n: int):
        """The bytes at `off`, contiguous: a view when they lie in one block, else a copy."""
        b, i = divmod(off, BLOCK)
        if i + n <= BLOCK:
            return memoryview(self.block(b))[i:i + n]
        out = bytearray(n)
        at = 0
        for p in self.pieces(off, n):
            out[at:at + len(p)] = p
            at += len(p)
        return out


# ---- where each dataset lives (the runner's endpoint.rs) ----

def _roots(ast: dict) -> dict:
    """Each dataset's and namespace's (protocol, root relative to --root)."""
    out = {}
    for name, d in ast.get("datasets", {}).items():
        root = d["pattern"].rpartition("/")[0] if d["layout"] == "array" else Pattern(d["pattern"]).root()
        out[name] = (d.get("protocol", "posix"), root)
    for name, n in ast.get("namespaces", {}).items():
        out[name] = (n.get("protocol", "posix"), Pattern(n["pattern"]).root())
    return out


def _obstore():
    try:
        import obstore
    except ImportError:
        raise BuildError("an object endpoint needs `obstore`: install the builder's `s3` extra "
                         "(`uv sync --inexact --extra s3` or `pip install 'aeiou[s3]'`)") from None
    return obstore


class Bucket:
    """An object endpoint, `s3://BUCKET[/PREFIX]` (the runner's `Store::open`): the store from
    the environment (`AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    `AWS_REGION`, `AWS_ALLOW_HTTP`), path-style unless `AWS_VIRTUAL_HOSTED_STYLE_REQUEST` is
    set."""

    def __init__(self, uri: str):
        rest = uri[len("s3://"):] if uri.startswith("s3://") else None
        if rest is None:
            raise BuildError(f"{uri}: an object endpoint is s3://BUCKET[/PREFIX]")
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise BuildError(f"{uri}: no bucket")
        _obstore()
        from obstore.store import S3Store
        kw = {} if os.environ.get("AWS_VIRTUAL_HOSTED_STYLE_REQUEST") else {"virtual_hosted_style_request": False}
        try:
            self.store = S3Store(bucket, **kw)
        except Exception as e:
            raise BuildError(f"{uri}: {e}") from None
        self.uri, self.bucket, self.prefix = uri.rstrip("/"), bucket, prefix.strip("/")

    def __str__(self):
        return self.uri

    def key(self, rest: str) -> str:
        rest = rest.strip("/")
        return "/".join(x for x in (self.prefix, rest) if x)


class Obj:
    """A key in a `Bucket`: where a path of an object name lives."""

    def __init__(self, bucket: Bucket, key: str):
        self.bucket, self.key = bucket, key

    def __truediv__(self, rel: str) -> "Obj":
        return Obj(self.bucket, "/".join(x for x in (self.key, rel) if x))

    def __str__(self):
        return f"s3://{self.bucket.bucket}/{self.key}"


def placement(ast: dict, root: pathlib.Path, given: list):
    """`--endpoint NAME=DIR` or `NAME=s3://BUCKET[/PREFIX]`: the function from a path relative
    to --root to where it lives, a `pathlib.Path` or an `Obj`. An endpoint places a name's root
    directory; a path falls under the longest placed root that is a prefix of it, else under
    --root; names sharing a root share its place (DESIGN_REVIEW.md §3.65). As the runner has
    it, an object name has no default place, an endpoint's kind is its name's protocol's,
    and a `posix` name never falls under a root placed in an object store."""
    roots = _roots(ast)
    placed: dict[str, tuple[list, object]] = {}
    for g in given:
        name, sep, d = g.partition("=")
        if not sep:
            raise usage.UsageError(f"--endpoint {g}: expected NAME=DIR")
        if name not in roots:
            raise usage.UsageError(f"--endpoint {g}: the abstract has no dataset or namespace `{name}`")
        uri = "://" in d
        proto = roots[name][0]
        if proto == "posix" and uri:
            raise usage.UsageError(f"--endpoint {g}: `{name}` is a `posix` dataset or namespace; its endpoint is a directory")
        if proto == "s3" and not uri:
            raise usage.UsageError(f"--endpoint {g}: `{name}` is declared `protocol: s3`; its endpoint is s3://BUCKET[/PREFIX]")
        if not d:
            raise usage.UsageError(f"--endpoint {g}: no directory")
        r = roots[name][1]
        if r in placed:
            names, at = placed[r]
            if name in names:
                raise usage.UsageError(f"--endpoint {name}: given twice")
            if (at.uri != d.rstrip("/")) if isinstance(at, Bucket) else (at != pathlib.Path(d)):
                raise usage.UsageError(f"--endpoint {g}: `{name}` shares root `{r}/` with `{'`, `'.join(names)}`, placed at {at}; a root is in one place")
            names.append(name)
        else:
            try:
                placed[r] = ([name], Bucket(d) if uri else pathlib.Path(d))
            except BuildError as e:
                raise usage.UsageError(f"--endpoint {g}: {e}") from None
    order = sorted(placed.items(), key=lambda kv: (-len(kv[0]), kv[0]))

    def under(rel: str):
        for r, (names, d) in order:
            if r == "" or rel == r or rel.startswith(r + "/"):
                return r, names, d
        return None

    for name in sorted(roots):
        proto, r = roots[name]
        u = under(r)
        is_object = u is not None and isinstance(u[2], Bucket)
        if proto == "s3" and not is_object:
            where = f" (it falls under `{'`, `'.join(u[1])}`, placed at {u[2]})" if u else ""
            raise usage.UsageError(f"`{name}` is declared `protocol: s3` and has no object endpoint{where}: give it --endpoint {name}=s3://BUCKET[/PREFIX]")
        if proto == "posix" and is_object:
            raise usage.UsageError(f"`{name}` is a `posix` dataset or namespace under `{'`, `'.join(u[1])}`, placed in the object store {u[2].uri}: give it a directory, --endpoint {name}=DIR")

    def at(rel: str):
        u = under(rel)
        if u is None:
            return root / rel
        r, _, d = u
        rest = rel if r == "" else rel[len(r):].lstrip("/")
        if isinstance(d, Bucket):
            return Obj(d, d.key(rest))
        return d / rest if rest else d
    return at


# ---- one dataset ----

class Job:
    def __init__(self, ast, name, values, at, dedupe: int, compress: int):
        """`at`: where a path relative to --root lives (`placement`)."""
        self.ast, self.name, self.values, self.at = ast, name, values, at
        self.resolved = resolved_dataset(ast, name, values)
        body = self.resolved
        if "inventory" in body:
            raise BuildError(f"dataset {name}: an inventory (contract 0.9) is written by the Rust `aeiou datagen` "
                             f"alone; this writer is the byte reference of the container formats")
        self.fmt = body["format"]
        self.klass = FormatClass.from_spec(self.fmt)
        self.count = ev(body["count"])
        self.spf = ev(body.get("samples_per_file", 1))
        if self.count < 0 or self.spf < 1:
            raise BuildError(f"dataset {name}: count {self.count}, samples_per_file {self.spf}")
        self.files = (self.count + self.spf - 1) // self.spf
        self.framing = Framing.from_spec(self.fmt.get("framing"), self.spf, ev)
        self.size = body["size"]
        self.seed = int(body["seed"])
        self.pattern = Pattern(body["pattern"])
        self.rel_root = self.pattern.root()
        self.dedupe, self.compress = max(1, dedupe), max(1, compress)

    def sizes(self, file: int) -> list:
        first = file * self.spf
        n = min(self.spf, self.count - first)
        return [sample_size(self.size, self.seed, first + i) for i in range(n)]

    def unit(self, file: int) -> int:
        return dedupe_unit(self.size, self.seed, file, self.dedupe, self.spf == 1)

    def write(self, file: int) -> tuple:
        sizes = self.sizes(file)
        geo = FileGeometry(self.framing, sizes)
        rel = self.pattern.format(id=file)
        place = self.at(rel)
        payload = Payload(self.seed, self.unit(file), self.compress)
        with opened(place) as out:
            sink = Sink(out)
            try:
                info = self.klass.write_file(sink, file * self.spf, sizes, payload, geo)
            except BuildError as e:
                raise BuildError(f"{place}: {e}") from None
            # checked before the file is closed, so an upload that fails it is aborted
            self.klass.check_file(str(place), sink.n, sizes, geo, info)
        return len(sizes), geo.file_size

    def manifest(self, files_written: int, bytes_written: int, started: float, provenance: dict, finished: float | None = None) -> dict:
        m = {"manifest_version": MANIFEST_VERSION, "state": "complete", "dataset": self.resolved,
             "payload": {"generator": GENERATOR, "version": GENERATOR_VERSION, "wrapper": WRAPPER, "block": BLOCK,
                         "dedupe": self.dedupe, "compress": self.compress},
             "format": {"class": self.fmt["class"], "reader": self.fmt.get("reader"), "version": self.fmt.get("version"),
                        "writer": self.klass.library()}}
        m["provenance"] = {**provenance, "datagen": f"aeiou-datagen {__version__}", "host": socket.gethostname(),
                           "python": platform.python_version(), "started": int(started),
                           "finished": int(time.time() if finished is None else finished),
                           "files_written": files_written, "bytes_written": bytes_written}
        return m

    @staticmethod
    def manifest_id(m: dict) -> str:
        normative = {k: m[k] for k in ("manifest_version", "dataset", "payload", "format")}
        return hashlib.sha256(canonical_json(normative).encode()).hexdigest()


# ---- writer processes ----

_job = None


def _start(spec):
    """A writer process's job, rebuilt from what pickles (the store is opened again here)."""
    global _job
    ast, name, values, root, endpoints, dedupe, compress = spec
    _job = Job(ast, name, values, placement(ast, root, endpoints), dedupe, compress)


def _write(file: int) -> tuple:
    return _job.write(file)


def _context():
    """Writers forked from a server process that imported the libraries once, before any
    thread existed (this process may hold the object store's runtime threads, which a fork
    must not copy)."""
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload(["numpy", "h5py", "pyarrow.parquet", "crc32c"])
    return ctx


class opened:
    """A file to write: a `pathlib.Path` opened `wb` (its directory made first), or an `Obj`'s
    upload, one `PUT` when it fits in a part and a multipart upload in parts of `PART`
    otherwise, each sent before the next is filled (as `aeiou datagen` sends them); an
    exception before the close aborts the upload, so no object is left."""

    def __init__(self, place):
        self.place = place

    def __enter__(self):
        if isinstance(self.place, Obj):
            self.f = _obstore().open_writer(self.place.bucket.store, self.place.key, buffer_size=PART, max_concurrency=1)
        else:
            self.place.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.place, "wb")
        return self.f.__enter__()

    def __exit__(self, *exc):
        return self.f.__exit__(*exc)


def is_empty(root) -> bool:
    """No file below a directory, no key below a prefix (one `LIST` page)."""
    if isinstance(root, Obj):
        prefix = root.key + "/" if root.key else None
        for chunk in _obstore().list(root.bucket.store, prefix=prefix, chunk_size=1):
            if chunk:
                return False
        return True
    return not (root.exists() and any(root.iterdir()))


def file_slice(files: int, ranks: int, rank: int) -> tuple:
    """Rank `rank`'s ids of `files`, `[lo, hi)`, contiguous and possibly empty (`datagen.rs`
    `file_slice`, over `run.rs` `gpu_range`)."""
    per = -(-files // ranks)
    return min(rank * per, files), min((rank + 1) * per, files)


def syncfs(path) -> None:
    """The file system holding `path` to stable storage, as `aeiou datagen` leaves it before
    it reports (`syncfs(2)`; `sync(2)` where libc has no `syncfs`)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "syncfs"):
            os.sync()
        elif libc.syncfs(fd) != 0:
            e = ctypes.get_errno()
            raise BuildError(f"syncfs {path}: {os.strerror(e)}")
    finally:
        os.close(fd)


def write_manifest(dir_, m: dict):
    """In a directory, through a temporary name renamed into place; in a bucket, one `PUT`
    at the prefix, atomic as the rename is."""
    text = json.dumps(m, indent=2) + "\n"
    if isinstance(dir_, Obj):
        obj = dir_ / MANIFEST_NAME
        _obstore().put(obj.bucket.store, obj.key, text.encode(), use_multipart=False)
        return obj
    path = dir_ / MANIFEST_NAME
    fd, tmp = tempfile.mkstemp(prefix=MANIFEST_NAME + ".tmp.", dir=str(dir_))
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def parser() -> usage.Parser:
    ap = usage.Parser(prog="aeiou-datagen", description="Write the container datasets an abstract declares under --root, through their format classes, with a manifest per dataset root (the Python twin of `aeiou datagen`)",
                      usage="aeiou-datagen [OPTIONS] --root <DIR> <ABSTRACT_PATH>\n       (--root may also come from $AEIOU_ROOT or `root` in the [datagen] table of the config file)")
    ap.add_argument("ast", nargs="?", type=pathlib.Path, metavar="ABSTRACT_PATH", help="The abstract (`.ast.json`)")
    ap.add_argument("--params-file", action="append", default=[], type=pathlib.Path, metavar="FILE", help="A parameter file (`.params.json`, schema/README.md §8), applied over the defaults and under --param; repeatable, in order")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="Override a parameter: `--param name=value` (JSON; a bare word is a string)")
    ap.add_argument("--gpus", type=int, default=1, help="Instance count, for dataset definitions that reference `gpus` [default: 1]")
    ap.add_argument("--root", type=pathlib.Path, metavar="DIR", help="Required: directory the abstract's paths are relative to. From the command line, else $AEIOU_ROOT, else `root` in the [datagen] table of the config file")
    ap.add_argument("--endpoint", action="append", metavar="NAME=DIR", help="Put the root directory of the abstract's dataset NAME at DIR instead of under --root, or, for a name declared `protocol: s3`, at s3://BUCKET[/PREFIX] in the store the AWS_* environment names (repeatable; names sharing a root share its place). From the command line, else $AEIOU_ENDPOINT (whitespace-separated), else `endpoint` in the [datagen] table of the config file (an array)")
    ap.add_argument("--writers", type=int, metavar="N", help="Files written at once, a process each (Python holds its lock between the library calls) (default: all cores)")
    ap.add_argument("--dedupe", type=int, default=1, help="Dedupe ratio: files share their content `dedupe` by `dedupe`, at the same offsets (files of one sample in sets of near sizes), so 4 KiB blocks repeat near `dedupe` times [default: 1]")
    ap.add_argument("--compress", type=int, default=1, help="Compression ratio: in every 4 KiB, the first ⌈4096/C⌉ bytes are random and the rest 0x5A [default: 1]")
    ap.add_argument("--dataset", action="append", default=[], metavar="NAME", help="Only these datasets (default: all)")
    ap.add_argument("--rank", type=int, metavar="R", help="This host's index among --ranks hosts (default: 0)")
    ap.add_argument("--ranks", type=int, metavar="N", help="Hosts the datagen is spread over: each writes its slice of every dataset's file ids, and rank 0 writes the manifests once every rank has reported (default: 1)")
    ap.add_argument("--coordinator", metavar="HOST:PORT", help="With --ranks above 1: the coordinator's address. Rank 0 listens on it (in process); every rank connects to it")
    ap.add_argument("--config", type=pathlib.Path, metavar="FILE", help="A TOML config file (else $AEIOU_CONFIG, else none; never searched for): one table per subcommand, keys spelled as the long flags (runner/REFERENCE.md §14)")
    ap.version(f"aeiou-datagen {__version__}")
    return ap


def main(argv=None) -> int:
    ap = parser()
    try:
        a = ap.parse_args(argv)
        return _main(a, ap)
    except usage.UsageError as e:
        return usage.fail(e, ap)
    except BuildError as e:
        return usage.failed(ap.prog, e)


def resolve(a, ap: usage.Parser | None = None) -> options.Layers:
    """`aeiou datagen`'s layers (main.rs `datagen_cmd`): the shape and the writer's fixed
    options first, then `root` and `writers` through the layers; what is missing from every
    layer is listed in one usage error (`usage.Missing`). Sets `a.root` and `a.writers` to
    the values in effect."""
    missing = usage.Missing(ap or parser())
    missing.want(a.ast, *usage.ABSTRACT)
    layers = options.Layers("datagen", a.config)
    layers.fixed("abstract", a.ast, a.ast is not None)
    layers.fixed("param", a.param, bool(a.param))
    layers.fixed("params-file", a.params_file, bool(a.params_file))
    layers.fixed("gpus", a.gpus, a.gpus != 1)
    layers.fixed("dedupe", a.dedupe, a.dedupe != 1)
    layers.fixed("compress", a.compress, a.compress != 1)
    layers.fixed("dataset", a.dataset, bool(a.dataset))
    a.root = missing.want(layers.layered("root", a.root, pathlib.Path), *usage.root("datagen"))
    a.endpoint = layers.layered("endpoint", a.endpoint, list) or []
    a.writers = layers.layered("writers", a.writers, int)
    if a.writers is None:
        a.writers = os.cpu_count() or 1
    a.rank = layers.layered("rank", a.rank, int, 0)
    a.ranks = layers.layered("ranks", a.ranks, int, 1)
    a.coordinator = layers.layered("coordinator", a.coordinator, str)
    because = layers.from_plain(["ranks"])
    missing.need(not (a.ranks > 1 and a.coordinator is None), "--coordinator <HOST:PORT>",
                 "the coordinator's address: rank 0 listens on it, every rank connects to it",
                 f"needed with --ranks {a.ranks}" + (f", {because}" if because else ""))
    missing.check()
    layers.finish()
    if a.ranks < 1 or not 0 <= a.rank < a.ranks:
        raise usage.UsageError(f"--rank {a.rank} of --ranks {a.ranks}: rank must be in [0, ranks){layers.from_(['rank', 'ranks'])}")
    return layers


def _main(a, ap: usage.Parser) -> int:
    from .validate import validate
    layers = resolve(a, ap)
    ast = emit.load(a.ast)
    validate(ast)
    sha = emit.sha256(ast)
    sets = []
    for p in a.params_file:
        pset = params_mod.load(p)
        errors = params_mod.check(ast, pset, ast_sha256=sha, name=p.name)
        if errors:
            raise BuildError("\n  ".join(errors))
        sets.append(pset)
    overrides = []
    for s in a.param:
        if "=" not in s:
            raise usage.UsageError(f"--param {s}: expected NAME=VALUE")
        k, v = s.split("=", 1)
        if k not in ast.get("params", {}):
            raise usage.UsageError(f"--param {k}: no such parameter")
        overrides.append((k, v))
    values = param_values(ast, sets, overrides, a.gpus)
    print(f"abstract {ast['name']}  sha256 {sha}")
    layers.print()
    at = placement(ast, a.root, a.endpoint)
    host = socket.gethostname()
    provenance = {"abstract": ast["name"], "ast_sha256": sha, "params": values,
                  "param_files": [{"path": str(p), "sha256": params_mod.file_sha256(p)} for p in a.params_file]}

    # several hosts: rank 0 listens, every rank connects and has its identity checked before
    # anything is written; a host that fails tells the others through it
    server = client = None
    if a.ranks > 1:
        if a.rank == 0:
            server = coord.Server(a.coordinator, a.ranks)
            print(f"coordinator listening on {server.addr}", flush=True)
        identity = {"ast_sha256": sha, "gpus": a.gpus, "params": values, "dedupe": a.dedupe, "compress": a.compress,
                    "datasets": a.dataset}
        client = coord.Client(a.coordinator, a.rank, a.ranks, host, identity, layers.json())
        print(f"coordinator {client.addr}: connected as rank {a.rank} of {a.ranks}  identity {coord.identity_hash(identity)[:16]}…", flush=True)
    try:
        jobs = _plan(a, ast, values, at)
        for job, root, lo, hi in jobs:
            print(f"dataset {job.name}: files [{lo}, {hi}) of {job.files} on this rank")
        sys.stdout.flush()
        # the gate: no rank writes before every rank has checked its roots
        if client is not None:
            _, names = client.ready()
            print(f"start gate: {len(names)} host(s) ready ({', '.join(names)})", flush=True)
        parts = {}
        for job, root, lo, hi in jobs:
            parts[job.name] = _write_share(a, ast, values, job, lo, hi)
        # the metadata to stable storage before this rank reports
        for root in {str(root) for _, root, _, _ in jobs if not isinstance(root, Obj)}:
            syncfs(root)
        # one rank's part of each dataset, as rank 0 records it (`datagen.rs` `part_json`)
        mine = {"host": host, "datasets": {n: {"rank": a.rank, "host": host, **{k: v for k, v in w.items() if k != "elapsed"}}
                                           for n, w in parts.items()}}
        gathered = {0: mine}
        if client is not None:
            client.done(mine)
            if server is not None:
                gathered = server.gathered()
            else:
                ok, _, error = client.result()
                if not ok:
                    raise BuildError(error or "rank 0 failed")
    except BaseException as e:
        reason = str(e) or type(e).__name__
        if server is not None:
            server.stop(reason)
        elif client is not None:
            client.stop(reason)
        raise

    # the manifests: rank 0's, once every rank's part is in; on one host, now
    error = None
    for job, root, lo, hi in jobs:
        w = parts[job.name]
        hosts = [d["datasets"][job.name] for _, d in sorted(gathered.items()) if job.name in d.get("datasets", {})]
        if a.rank == 0 and len(hosts) != a.ranks:
            error = f"dataset `{job.name}`: {len(hosts)} of {a.ranks} ranks reported"
            break
        total_files = sum(h["files_written"] for h in hosts)
        total_bytes = sum(h["bytes_written"] for h in hosts)
        prov = {**provenance, "ranks": a.ranks, "hosts": hosts}
        m = job.manifest(total_files, total_bytes, min(h["started"] for h in hosts), prov,
                         finished=max(h["finished"] for h in hosts))
        mine_line = f"{w['files_written']} file(s), {w['bytes_written'] / 2**20:.2f} MiB in {w['elapsed']:.2f}s"
        if a.ranks == 1:
            write_manifest(root, m)
            print(f"dataset {job.name}: {mine_line} at {root}  id {Job.manifest_id(m)}  ({job.klass.name}, {job.klass.library()})")
        elif a.rank == 0:
            try:
                write_manifest(root, m)
            except (OSError, BuildError) as e:
                error = f"dataset `{job.name}`: {e}"
                break
            print(f"dataset {job.name}: this rank {mine_line}; all {a.ranks} hosts {total_files} file(s), {total_bytes / 2**20:.2f} MiB "
                  f"at {root}  id {Job.manifest_id(m)}  manifest written")
        else:
            print(f"dataset {job.name}: this rank {mine_line} at {root}  id {Job.manifest_id(m)}  (manifest by rank 0)")
    if server is not None:
        # the verdict: every rank exits with it
        server.finish(error is None, error)
    if error is not None:
        raise BuildError(error)
    return 0


def _plan(a, ast, values, at) -> list:
    """Every dataset this tool writes, its root checked empty and made, and this rank's slice
    of its ids: `(job, root, lo, hi)`. Every rank checks before the gate, so no rank sees
    another's files."""
    wanted = set(a.dataset)
    jobs = []
    for name, d in ast.get("datasets", {}).items():
        has_format = "format" in d
        if wanted and name not in wanted:
            continue
        if not has_format:
            if wanted:
                raise BuildError(f"dataset `{name}` has no format class: `aeiou datagen` (the Rust writer) writes it")
            print(f"dataset {name}: no format class, left to `aeiou datagen`")
            continue
        job = Job(ast, name, values, at, a.dedupe, a.compress)
        root = at(job.rel_root)
        if not is_empty(root):
            raise BuildError(f"dataset `{name}`: {root} is not empty; datasets are read-only, remove it first")
        if not isinstance(root, Obj):
            root.mkdir(parents=True, exist_ok=True)
        jobs.append((job, root, *file_slice(job.files, a.ranks, a.rank)))
    if wanted and len(jobs) < len(wanted):
        raise BuildError(f"not every --dataset was found: {sorted(wanted)}")
    return jobs


def _write_share(a, ast, values, job: Job, lo: int, hi: int) -> dict:
    """Write ids `[lo, hi)` of one dataset; this rank's part of it, as rank 0 records it."""
    started = time.time()
    t0 = time.monotonic()
    files_written = bytes_written = 0
    # processes, not threads: the writers' Python holds the GIL between the library
    # calls, so threads write no faster than one (DESIGN_REVIEW.md §3.69, measured
    # 2026-10-11); a file is a pure function of its id, so the processes share nothing
    workers = max(1, min(a.writers, hi - lo))
    with contextlib.ExitStack() as stack:
        if workers == 1:
            results = map(job.write, range(lo, hi))
        else:
            ex = stack.enter_context(concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=_context(), initializer=_start,
                initargs=((ast, job.name, values, a.root, a.endpoint, a.dedupe, a.compress),)))
            results = ex.map(_write, range(lo, hi))
        last = time.time()
        for i, (n, b) in enumerate(results):
            files_written += 1
            bytes_written += b
            if time.time() - last > 5:
                last = time.time()
                print(f"  {job.name}: {i + 1}/{hi - lo} files, {bytes_written / 2**20:.1f} MiB written", flush=True)
    return {"files_written": files_written, "bytes_written": bytes_written, "started": int(started),
            "finished": int(time.time()), "elapsed": time.monotonic() - t0}

if __name__ == "__main__":
    sys.exit(main())
