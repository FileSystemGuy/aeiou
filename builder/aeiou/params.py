"""`aeiou-params`: parameter files, the second artifact of the shape/parameters split.

An abstract is a shape with named slots; its parameter defaults are placeholders or one
reference set. A parameter file (`<abstract>.<set>.params.json`, `schema/params.schema.json`,
`schema/README.md` §8) fills the slots from the three sources of `GRAMMAR_OPTIONS.md` Option D
(configuration, measurement, a trace), so one published shape can carry several parameter
sets and a fitted set can be reviewed and hashed on its own. The runner applies it with
`aeiou run --params-file FILE`, over the defaults and under `--param`.

    aeiou-params defaults  AST.json [-o FILE] [--doc TEXT] [--pin]   the defaults as a starting set
    aeiou-params check     AST.json FILE...                          validate sets against an abstract
    aeiou-params safetensors AST.json SHARD... [-o FILE] [--tp N]    safetensors_load's tensor table from real shards
    aeiou-params npz       AST.json ARCHIVE... [-o FILE] [--member x] npy_per_sample' framing from real .npz files
    aeiou-params lock      CASE [--rules FILE] [-o FILE]           a use case's lock (lock.py)

The value rules are the runner's (`runner/aeiou/src/eval.rs`): a name must be declared, `gpus`
may not appear, and a value replaces a default of the same kind (scalar, array, distribution;
an array may change length, its elements keep the default's element kind), because the
builder decided at build time how each slot is consumed, and where the default is an integer
an integral float is that integer and a fractional one is refused; and, as the runner does, rule V3 is
judged again on the distributions given and a pattern's divisor must bind to a positive integer
(V23), so `check` refuses every file `aeiou run --params-file` refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

from . import __version__, checkout, usage
from .nodes import BuildError

PARAMS_VERSION = 2
KEYS = ("params_version", "abstract", "ast_sha256", "doc", "params", "run", "provenance")
IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
STATES = ("default", "locked", "required")
# the runner's identity options a fragment's `run` table may set (DESIGN_REVIEW.md §3.78 point 3)
RUN_OPTIONS = ("seed", "posix", "cache", "s3", "time_scale", "rank_rotate", "write_compress")


def kind_of(v) -> str:
    if isinstance(v, dict):
        return "a distribution"
    if isinstance(v, list):
        return "an array"
    return "a scalar"


def entry(e) -> tuple[str, object]:
    """(state, value) of an entry, `{"default": v}`, `{"locked": v}`, or `{"required": true}`
    (contract 0.11); the value of a required one is None."""
    if isinstance(e, dict) and len(e) == 1:
        (k, v), = e.items()
        if k in ("default", "locked"):
            return k, v
        if k == "required" and v is True:
            return k, None
    raise BuildError(f"not an entry: {{\"default\": v}}, {{\"locked\": v}}, or {{\"required\": true}}, got {json.dumps(e)}")


def values_of(pset: dict) -> dict:
    """The values of a set's `default` and `locked` entries, by name."""
    return {k: entry(e)[1] for k, e in pset.get("params", {}).items() if entry(e)[0] != "required"}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def declared_kind(decl: dict) -> tuple[str, bool, bool]:
    """A declaration's (coarse kind, whether an integer, whether an array of integers), as the
    runner's `Param` has it: the default's, or, required, its `kind` and `items` (contract
    0.11); a storage-reported default is an integer."""
    if "default" not in decl:
        k = decl.get("kind")
        coarse = {"dist": "a distribution", "array": "an array"}.get(k, "a scalar")
        return coarse, k == "int", k == "array" and decl.get("items") == "int"
    d = decl["default"]
    if isinstance(d, dict) and "from" in d:
        return "a scalar", True, False
    return kind_of(d), _is_int(d), isinstance(d, list) and bool(d) and _is_int(d[0])


def keep_int(pname: str, decl: dict, v):
    """A value keeps its parameter's kind down to the number, as the runner has it
    (`eval::keep_int`): where the kind is an integer (an array's elements are), an integral
    float is that integer (`3e3`, `3000.0`), so the dataset id is the integer's, and a
    fractional one is refused, never rounded. Returns (value, errors)."""
    _, is_int, ints = declared_kind(decl)
    what = "the default" if "default" in decl else "the kind"
    if is_int and isinstance(v, float):
        if v.is_integer() and -2**63 <= v < 2**63:
            return int(v), []
        return v, [f"{pname}: {what} is an integer, the value {v!r} is not"]
    if ints and isinstance(v, list):
        out, errors = [], []
        for i, x in enumerate(v):
            if isinstance(x, float):
                if x.is_integer() and -2**63 <= x < 2**63:
                    x = int(x)
                else:
                    errors.append(f"{pname}[{i}]: {what} is an integer, the value {x!r} is not")
            out.append(x)
        return out, errors[:1]
    return v, []


def default_value(decl: dict):
    """The value a declaration gives before any layer: its default, a storage-reported one's
    fallback, None for a required one."""
    d = decl.get("default")
    if isinstance(d, dict) and "from" in d:
        return d.get("fallback")
    return d


def load(path) -> dict:
    """Read and structurally validate a parameter file (the published JSON Schema)."""
    import jsonschema
    from .validate import schema_dir
    from .emit import loads
    path = pathlib.Path(path)
    try:
        doc = loads(path.read_text(encoding="utf-8"))
    except OSError as e:   # worded as the runner words a file it cannot read
        raise BuildError(f"{path}: {e.strerror} (os error {e.errno})") from None
    except ValueError as e:
        raise BuildError(f"{path}: {e}") from None
    d = schema_dir()
    registry = None
    try:
        from referencing import Registry, Resource
        ast_schema = json.loads((d / "abstract-ast.schema.json").read_text(encoding="utf-8"))
        registry = Registry().with_resource("abstract-ast.schema.json", Resource.from_contents(ast_schema))
    except ImportError:  # old jsonschema without `referencing`: skip the cross-file refs
        pass
    schema = json.loads((d / "params.schema.json").read_text(encoding="utf-8"))
    if registry is not None:
        v = jsonschema.Draft202012Validator(schema, registry=registry)
        errors = [f"{path.name}: {'/'.join(map(str, e.absolute_path))}: {e.message[:300]}"
                  for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))]
        if errors:
            raise BuildError("parameter file rejected:\n  " + "\n  ".join(errors))
    if isinstance(doc, dict) and doc.get("params_version") == 1:
        raise BuildError(f"{path.name}: params_version 1, whose values have no state; params_version 2 (contract 0.11) wraps each: {{\"default\": v}}, {{\"locked\": v}}, or {{\"required\": true}}")
    if not isinstance(doc, dict) or doc.get("params_version") != PARAMS_VERSION or not isinstance(doc.get("params"), dict):
        raise BuildError(f"{path.name}: not a parameter file (params_version {PARAMS_VERSION} with a `params` object)")
    for k in doc:
        if k not in KEYS:
            raise BuildError(f"{path.name}: unknown key `{k}`")
    return doc


def check(ast: dict, pset: dict, *, ast_sha256: str | None = None, name: str = "params") -> list[str]:
    """The runner's rules, as a list of errors (empty when the set fits the abstract)."""
    from .validate import _load
    _, chk = _load()
    # the runner refuses a value past the contract's 64-bit integers before anything else
    errors = [f"{name}: {'/'.join(map(str, p))}: {m}" for p, m in chk.int_range(pset)]
    if errors:
        return errors
    if pset.get("abstract") != ast["name"]:
        errors.append(f"{name}: for abstract `{pset.get('abstract')}`, not `{ast['name']}`")
    want = pset.get("ast_sha256")
    if want and ast_sha256 and want != ast_sha256:
        errors.append(f"{name}: for AST {want[:16]}…, not {ast_sha256[:16]}…")
    declared = ast.get("params", {})
    given = {}
    for pname, e in pset.get("params", {}).items():
        if pname == "gpus":
            errors.append(f"{name}: `gpus` is set with --gpus, not by a parameter file")
            continue
        if pname not in declared:
            errors.append(f"{name}: no parameter `{pname}` in `{ast['name']}`")
            continue
        try:
            state, v = entry(e)
        except BuildError as x:
            errors.append(f"{name}: params: {pname}: {x}")
            continue
        if state == "required":
            continue
        decl = declared[pname]
        coarse, _, _ = declared_kind(decl)
        what = "the default" if "default" in decl else "the kind"
        if coarse != kind_of(v):
            errors.append(f"{name}: {pname}: {what} is {coarse}, the value is {kind_of(v)}")
        elif coarse == "an array":
            items = decl.get("items") if "default" not in decl else None
            first = kind_of(decl["default"][0]) if "default" in decl and decl["default"] else \
                ({"dist": "a distribution"}.get(items, "a scalar") if items else None)
            for i, x in enumerate(v):
                if first is not None and kind_of(x) != first:
                    errors.append(f"{name}: {pname}[{i}]: {what}'s elements are {first}, this one is {kind_of(x)}")
                    break
        v, x = keep_int(pname, decl, v)
        errors += [f"{name}: {m}" for m in x]
        given[pname] = v
    if not errors:
        errors += given_rules(ast, given, name=name)
    return errors


def given_rules(ast: dict, given: dict, *, name: str = "params") -> list[str]:
    """The rules the runner judges again on the values a run is given, past their kinds:
    the document's rules with each parameter's value in effect (`validate::check_given`: V3,
    where a value whose minimum is below the default's can turn an `at` offset into zero,
    `DESIGN_REVIEW.md` §3.57; a normal/lognormal `min` above its `max`; V25) and
    V23 on the pattern divisors they bind (`eval::bind_pattern`: a positive integer)."""
    from .pattern import Pattern
    from .validate import _load
    _, chk = _load()

    class Given(chk.Check):
        # every rule the runner's `check_given` judges on the value in effect (V3's minimums,
        # the normal/lognormal min-above-max rule, V25) reads a parameter through this hook
        def param_value(self, pname):
            return given[pname] if pname in given else super().param_value(pname)

    errors = [f"{name}: invalid with the parameters given: {e}" for e in Given(ast, ast["name"]).run()]
    values = {k: default_value(p) for k, p in ast.get("params", {}).items()} | given
    patterns = [(f"dataset `{n}`", d["pattern"]) for n, d in ast.get("datasets", {}).items() if d.get("layout") != "array"]
    patterns += [(f"namespace `{n}`", s["pattern"]) for n, s in ast.get("namespaces", {}).items()]
    for where, src in patterns:
        try:
            pt = Pattern(src)
            for p in pt.params():
                v = values.get(p)
                if not isinstance(v, int) or isinstance(v, bool):
                    raise ValueError(f"pattern `{src}`: divisor `{p}`: parameter `{p}` is not an integer")
            pt.bind(values)
        except ValueError as e:
            errors.append(f"{name}: {where}: {e}")
    return errors


def defaults(ast: dict, *, doc: str | None = None, pin: bool = False) -> dict:
    """The abstract's defaults as a parameter set: the starting point for a fitted set."""
    from .emit import sha256
    out = {"params_version": PARAMS_VERSION, "abstract": ast["name"]}
    if pin:
        out["ast_sha256"] = sha256(ast)
    out["doc"] = doc or f"the defaults of `{ast['name']}` as declared in the abstract"
    # a required parameter stays required; a storage-reported one is left to the storage
    out["params"] = {k: ({"required": True} if "default" not in p else {"default": p["default"]})
                     for k, p in ast.get("params", {}).items()
                     if not (isinstance(p.get("default"), dict) and "from" in p["default"])}
    return out


def render(pset: dict, provenance: dict | None = None) -> str:
    doc = {k: pset[k] for k in KEYS if k in pset and k != "provenance"}
    if provenance:
        doc["provenance"] = provenance
    return json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# safetensors: safetensors_load's tensor table from the real shards (source 1, configuration)
# ----------------------------------------------------------------------------------------------

DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2,
               "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
# Which tensors a tensor-parallel serving engine slices by column (an output-dimension split:
# q/k/v and gate/up projections, embeddings, the LM head) or by row (an input-dimension split:
# the o and down projections). Everything else (norms, biases) is replicated: every rank reads
# it whole. These are the Megatron conventions for Llama-shaped checkpoints; pass your own.
COLUMN_RE = r"(q_proj|k_proj|v_proj|gate_proj|up_proj|embed_tokens|lm_head)\.weight$"
ROW_RE = r"(o_proj|down_proj)\.weight$"


def read_safetensors_header(path) -> tuple[int, dict]:
    """(header length N, header JSON) of a safetensors file: 8 bytes little-endian N, then N bytes;
    BuildError, in one line, for a file that is not there or not one."""
    path = pathlib.Path(path)
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if size < 8 or n > size - 8:            # never allocate what the file cannot hold
                raise BuildError(f"{path.name}: not a safetensors file (a header of {n} bytes in a file of {size})")
            header = json.loads(f.read(n))
    except OSError as e:
        raise BuildError(f"{path}: {e.strerror}") from None
    except ValueError as e:
        raise BuildError(f"{path.name}: not a safetensors file (its header is not JSON: {e})") from None
    if not isinstance(header, dict):
        raise BuildError(f"{path.name}: not a safetensors file (its header is not a JSON object)")
    for name, t in header.items():
        if name == "__metadata__":
            continue
        ok = isinstance(t, dict) and isinstance(t.get("dtype"), str) and isinstance(t.get("shape"), list) \
            and all(isinstance(x, int) and not isinstance(x, bool) for x in t["shape"]) \
            and isinstance(t.get("data_offsets"), list) and len(t["data_offsets"]) == 2 \
            and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in t["data_offsets"]) \
            and t["data_offsets"][0] <= t["data_offsets"][1]
        if not ok:
            raise BuildError(f"{path.name}: not a safetensors file (tensor `{name}` has no dtype, shape, and data_offsets [begin, end])")
    return n, header


def safetensors_table(shards: list, *, tp: int = 8, column: str = COLUMN_RE, row: str = ROW_RE) -> dict:
    """The `safetensors_load` parameters for these shard files: one row per tensor, shards in the
    order given (the dataset's file id), tensors in file order."""
    col_re, row_re = re.compile(column), re.compile(row)
    shards = [pathlib.Path(s) for s in shards]
    table = {k: [] for k in ("shard", "off", "bytes", "split", "rows", "row_bytes")}
    hdr_len, sizes = [], []
    for s, path in enumerate(shards):
        n, header = read_safetensors_header(path)
        hdr_len.append(n)
        sizes.append(path.stat().st_size)
        tensors = sorted(((v["data_offsets"][0], k, v) for k, v in header.items() if k != "__metadata__"), key=lambda t: t[0])
        for start, name, t in tensors:
            end = t["data_offsets"][1]
            nbytes = end - start
            shape = t["shape"]
            width = DTYPE_BYTES.get(t["dtype"])
            if width is None:
                raise BuildError(f"{path.name}: {name}: unknown dtype {t['dtype']}")
            rows = shape[0] if len(shape) >= 2 else 1
            if rows < 1 or nbytes % rows:
                raise BuildError(f"{path.name}: {name}: {nbytes} bytes do not divide into {rows} rows")
            split = "column" if col_re.search(name) else "row" if row_re.search(name) else "full"
            table["shard"].append(s)
            table["off"].append(8 + n + start)
            table["bytes"].append(nbytes)
            table["split"].append(split)
            table["rows"].append(rows if split == "row" else 1)
            table["row_bytes"].append(nbytes // rows if split == "row" else nbytes)
    if not table["shard"]:
        raise BuildError("no tensors found")
    return {"shards": len(shards), "shard_bytes": max(sizes), "hdr_len": hdr_len, "tp": tp, **table}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def _load_ast(path) -> tuple[dict, str]:
    """The abstract and its hash. Not there, or not JSON: a usage error (`emit.load`); JSON that
    is no abstract: `AST rejected`, in one line per error, as every tool that reads one."""
    from .emit import load, sha256
    from .validate import validate
    path = pathlib.Path(path)
    ast = load(path)
    validate(ast, path.name)
    return ast, sha256(ast)


def _write(out: pathlib.Path, text: str) -> None:
    try:
        out.write_text(text, encoding="utf-8")
    except OSError as e:
        raise BuildError(f"{out}: {e.strerror}") from None


# npz: npy_per_sample' archive framing from real files (source 1, configuration)
#
# The abstract reads one member of an uncompressed .npz front to back and takes two numbers
# about the archive as parameters: `framing`, the archive's bytes that are not that member's
# array data, and `cd_len`, the central directory's length. Both are facts of the writer (the
# library, its version, the member names and their count) and are the same for every file one
# writer produces. This reads them from the files with the standard library only.

def npz_framing(path, member: str = "x") -> dict:
    """`framing` and `cd_len` of one archive, or BuildError when it does not have the shape the
    abstract reads (the member first and stored, no archive comment, no zip64 end record)."""
    import struct
    import zipfile
    path = pathlib.Path(path)
    try:
        size = path.stat().st_size
    except OSError as e:
        raise BuildError(f"{path}: {e.strerror}") from None
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise BuildError(f"{path.name}: not a zip archive ({e})") from None
    with z:
        names = z.namelist()
        if f"{member}.npy" not in names:
            raise BuildError(f"{path.name}: no member `{member}.npy` (members: {', '.join(names)})")
        info = z.getinfo(f"{member}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise BuildError(f"{path.name}: `{member}.npy` is compressed (np.savez_compressed); the abstract models the stored layout only")
        if info.header_offset != 0:
            raise BuildError(f"{path.name}: `{member}.npy` is not the first member (local header at {info.header_offset}); the abstract reads the archive from offset 0")
        if z.comment:
            raise BuildError(f"{path.name}: the archive has a comment; the abstract places the end record in the last 22 bytes")
        with z.open(info) as f:                              # the .npy header: magic, version, header length, header
            head = f.read(12)
        if head[:6] != b"\x93NUMPY":
            raise BuildError(f"{path.name}: `{member}.npy` is not an .npy array")
        hlen = 10 + struct.unpack("<H", head[8:10])[0] if head[6] == 1 else 12 + struct.unpack("<I", head[8:12])[0]
        cd_len = size - z.start_dir - 22
        with open(path, "rb") as f:                          # a zip64 end record would sit between the directory and the end record
            f.seek(size - 22)
            eocd = f.read(22)
        if eocd[:4] != b"PK\x05\x06" or struct.unpack("<I", eocd[12:16])[0] != cd_len:
            raise BuildError(f"{path.name}: the central directory does not end at the end record (a zip64 end record: an archive of 4 GiB or more, or 65,535 members)")
        return {"framing": size - (info.file_size - hlen), "cd_len": cd_len, "members": names, "bytes": size, "data": info.file_size - hlen}


def npz_params(archives: list, member: str = "x") -> tuple[dict, list]:
    """The parameter values the archives agree on, and one record per archive. Archives that
    disagree are refused: the abstract has one `framing` for the corpus."""
    recs = [npz_framing(p, member) for p in archives]
    for key in ("framing", "cd_len"):
        seen = sorted({r[key] for r in recs})
        if len(seen) > 1:
            which = {v: next(pathlib.Path(p).name for p, r in zip(archives, recs) if r[key] == v) for v in seen}
            raise BuildError(f"`{key}` differs between the archives: " + ", ".join(f"{v} ({n})" for v, n in which.items())
                             + "; they were not written by one writer with one set of members")
    return {"framing": recs[0]["framing"], "cd_len": recs[0]["cd_len"]}, recs


def _positive(v: str) -> int:
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"'{v}' is not a positive integer")
    return n


def _regex(v: str) -> str:
    try:
        re.compile(v)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"'{v}' is not a regular expression ({e})") from None
    return v


def parser() -> tuple[usage.Parser, dict[str, usage.Parser]]:
    """The parser and its subcommands' parsers by name (the usage a `Missing` list shows)."""
    ap = usage.Parser(prog="aeiou-params", description="Parameter files of an abstract (schema/README.md §8): write its defaults as a set, validate sets against it, build a set from real files",
                      usage="aeiou-params <COMMAND> [ARGS]...")
    ap.version(f"aeiou-params {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    abstract = dict(nargs="?", type=pathlib.Path, metavar="ABSTRACT_PATH")
    d = sub.add_parser("defaults", help="Write the abstract's defaults as a parameter set", usage="aeiou-params defaults [OPTIONS] <ABSTRACT_PATH>")
    d.add_argument("ast", **abstract, help="The abstract (`.ast.json`)")
    d.add_argument("-o", "--out", type=pathlib.Path, metavar="FILE", help="Output file (default: <abstract>.defaults.params.json)")
    d.add_argument("--doc", metavar="TEXT", help="The set's `doc` line")
    d.add_argument("--pin", action="store_true", help="Record the AST hash so the set is for this exact shape")
    c = sub.add_parser("check", help="Validate parameter files against an abstract", usage="aeiou-params check <ABSTRACT_PATH> <FILES>...")
    c.add_argument("ast", **abstract, help="The abstract (`.ast.json`)")
    c.add_argument("files", nargs="*", type=pathlib.Path, metavar="FILES", help="Parameter files (`.params.json`)")
    s = sub.add_parser("safetensors", help="Build safetensors_load's tensor table from safetensors shards", usage="aeiou-params safetensors [OPTIONS] -o <FILE> <ABSTRACT_PATH> <SHARDS>...")
    s.add_argument("ast", **abstract, help="The safetensors_load abstract (its parameter names are checked)")
    s.add_argument("shards", nargs="*", type=pathlib.Path, metavar="SHARDS", help="Shard files in dataset id order")
    s.add_argument("-o", "--out", type=pathlib.Path, metavar="FILE", help="Required: the parameter file to write")
    s.add_argument("--tp", type=_positive, default=8, help="Tensor-parallel degree [default: 8]")
    s.add_argument("--column", type=_regex, default=COLUMN_RE, metavar="REGEX", help="Regex of column-parallel tensors")
    s.add_argument("--row", type=_regex, default=ROW_RE, metavar="REGEX", help="Regex of row-parallel tensors")
    s.add_argument("--doc", metavar="TEXT", help="The set's `doc` line")
    s.add_argument("--pin", action="store_true", help="Record the AST hash so the set is for this exact shape")
    n = sub.add_parser("npz", help="Read npy_per_sample' framing and cd_len from real .npz archives and compare them with the abstract's defaults", usage="aeiou-params npz [OPTIONS] <ABSTRACT_PATH> <ARCHIVES>...")
    n.add_argument("ast", **abstract, help="The npy_per_sample abstract")
    n.add_argument("archives", nargs="*", type=pathlib.Path, metavar="ARCHIVES", help="Archives of the corpus (a few are enough; all must agree)")
    n.add_argument("-o", "--out", type=pathlib.Path, metavar="FILE", help="Write a parameter file with the two values")
    n.add_argument("--member", default="x", metavar="NAME", help="The member the application reads [default: x]")
    n.add_argument("--doc", metavar="TEXT", help="The set's `doc` line")
    n.add_argument("--pin", action="store_true", help="Record the AST hash so the set is for this exact shape")
    k = sub.add_parser("lock", help="Resolve a use case (and a benchmark's rules) into the lock a run names", usage="aeiou-params lock [OPTIONS] <CASE>")
    k.add_argument("case", nargs="?", type=pathlib.Path, metavar="CASE", help="The case directory (`workloads/…/cases/<case>`, with its case.toml)")
    k.add_argument("--rules", type=pathlib.Path, metavar="FILE", help="A benchmark's rules file: its freeze and constraints go into the lock")
    k.add_argument("-o", "--out", type=pathlib.Path, metavar="FILE", help="Output file (default: CASE/case.lock.json)")
    k.add_argument("--workloads", type=pathlib.Path, metavar="DIR", help="The workloads/ directory the lock's paths are relative to (default: the one above CASE)")
    return ap, {"defaults": d, "check": c, "safetensors": s, "npz": n, "lock": k}


@usage.tool
def main(argv=None) -> int:
    ap, subs = parser()
    checkout.warn(ap.prog)
    sp = ap
    try:
        a = ap.parse_args(argv)
        missing = usage.Missing(ap)
        missing.want(a.cmd, "<COMMAND>", "one of defaults, check, safetensors, npz, lock")
        missing.check()
        sp = subs[a.cmd]
        missing = usage.Missing(sp)
        if a.cmd == "lock":
            missing.want(a.case, "<CASE>", "the case directory, with its case.toml")
            missing.check()
            return _lock(a)
        missing.want(a.ast, *usage.ABSTRACT)
        if a.cmd == "check":
            missing.want(a.files, "<FILES>...", "the parameter files (`.params.json`) to validate against the abstract")
        if a.cmd == "safetensors":
            missing.want(a.shards, "<SHARDS>...", "the safetensors shard files, in dataset id order")
            missing.want(a.out, "-o <FILE>", "the parameter file to write")
        if a.cmd == "npz":
            missing.want(a.archives, "<ARCHIVES>...", "archives of the corpus (a few are enough; all must agree)")
        missing.check()
        return _main(a)
    except usage.UsageError as e:
        return usage.fail(e, sp)
    except BuildError as e:
        return usage.failed(sp.prog, e)


def _lock(a) -> int:
    """`aeiou-params lock`: the use case's lock, beside its case.toml unless -o says elsewhere."""
    from . import lock
    workloads = a.workloads or lock.workloads_of(a.case)
    out = a.out or a.case / lock.LOCK_FILE
    doc = lock.make_lock(a.case, workloads=workloads, rules=a.rules)
    _write(out, lock.render(doc))
    required = sorted({k for r in doc["runs"] for k, e in r["params"].items() if "required" in e})
    print(f"wrote {out}  {doc['name']}  {len(doc['runs'])} run(s)  sha256={file_sha256(out)[:16]}…"
          + (f"  required: {', '.join(required)}" if required else ""))
    return 0


def _main(a) -> int:
    ast, sha = _load_ast(a.ast)
    if a.cmd == "defaults":
        pset = defaults(ast, doc=a.doc, pin=a.pin)
        out = a.out or pathlib.Path(f"{ast['name']}.defaults.params.json")
        _write(out, render(pset, {"tool": f"aeiou-params {__version__}", "command": "defaults", "ast_sha256": sha}))
        print(f"wrote {out}  ({len(pset['params'])} parameters)")
        return 0
    if a.cmd == "check":
        status = 0
        for f in a.files:
            try:
                pset = load(f)
                errors = check(ast, pset, ast_sha256=sha, name=f.name)
            except BuildError as e:
                errors = [str(e)]
            if errors:
                status = 1
                print(f"FAIL {f.name}")
                for e in errors:
                    print("  " + e)
            else:
                print(f"ok   {f.name}  sha256={file_sha256(f)[:16]}…  {len(pset['params'])} parameters")
        return status
    if a.cmd == "safetensors":
        values = safetensors_table(a.shards, tp=a.tp, column=a.column, row=a.row)
        pset = {"params_version": PARAMS_VERSION, "abstract": ast["name"]}
        if a.pin:
            pset["ast_sha256"] = sha
        pset["doc"] = a.doc or f"tensor table of {len(a.shards)} safetensors shard(s), tp={a.tp}"
        pset["params"] = {k: {"default": v} for k, v in values.items()}
        errors = check(ast, pset, ast_sha256=sha, name=a.out.name)
        if errors:
            raise BuildError("\n  ".join(errors))
        prov = {"tool": f"aeiou-params {__version__}", "command": "safetensors", "ast_sha256": sha,
                "shards": [{"file": p.name, "sha256": file_sha256(p)} for p in a.shards],
                "column": a.column, "row": a.row}
        _write(a.out, render(pset, prov))
        print(f"wrote {a.out}  ({len(values['shard'])} tensors in {values['shards']} shard(s))")
        return 0
    if a.cmd == "npz":
        values, recs = npz_params(a.archives, a.member)
        defaults_ = {k: default_value(ast["params"][k]) for k in values if k in ast.get("params", {})}
        print(f"{len(recs)} archive(s), members {', '.join(recs[0]['members'])}: framing {values['framing']}, cd_len {values['cd_len']}")
        differs = {k: (defaults_.get(k), v) for k, v in values.items() if defaults_.get(k) != v}
        for k, (was, now) in differs.items():
            print(f"DIFFERS {k}: the abstract's default is {was}, these archives have {now}")
        if not differs:
            print("the abstract's defaults describe these archives")
        if a.out:
            pset = {"params_version": PARAMS_VERSION, "abstract": ast["name"]}
            if a.pin:
                pset["ast_sha256"] = sha
            pset["doc"] = a.doc or f"archive framing of {len(recs)} .npz file(s), member {a.member}"
            pset["params"] = {k: {"default": v} for k, v in values.items()}
            errors = check(ast, pset, ast_sha256=sha, name=a.out.name)
            if errors:
                raise BuildError("\n  ".join(errors))
            prov = {"tool": f"aeiou-params {__version__}", "command": "npz", "ast_sha256": sha, "member": a.member,
                    "archives": [{"file": p.name, "bytes": r["bytes"], "members": r["members"]} for p, r in zip(a.archives, recs)]}
            _write(a.out, render(pset, prov))
            print(f"wrote {a.out}")
        return 1 if differs and not a.out else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
