"""`aeiou-params`: parameter files, the second artifact of the shape/parameters split.

An abstract is a shape with named slots; its parameter defaults are placeholders or one
reference set. A parameter file (`<abstract>.<set>.params.json`, `schema/params.schema.json`,
`schema/README.md` §8) fills the slots from the three sources of `GRAMMAR_OPTIONS.md` Option D
(configuration, measurement, a trace), so one published shape can carry several parameter
sets and a fitted set can be reviewed and hashed on its own. The runner applies it with
`aeiou run --params-file FILE`, over the defaults and under `--param`.

    aeiou-params defaults  AST.json [-o FILE] [--doc TEXT] [--pin]   the defaults as a starting set
    aeiou-params check     AST.json FILE...                          validate sets against an abstract
    aeiou-params safetensors AST.json SHARD... [-o FILE] [--tp N]    model_load's tensor table from real shards
    aeiou-params npz       AST.json ARCHIVE... [-o FILE] [--member x] train_large_samples' framing from real .npz files

The value rules are the runner's (`runner/aeiou/src/eval.rs`): a name must be declared, `gpus`
may not appear, and a value replaces a default of the same kind (scalar, array, distribution;
an array may change length, its elements keep the default's element kind), because the
builder decided at build time how each slot is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

from . import __version__
from .nodes import BuildError

PARAMS_VERSION = 1
KEYS = ("params_version", "abstract", "ast_sha256", "doc", "params", "provenance")
IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def kind_of(v) -> str:
    if isinstance(v, dict):
        return "a distribution"
    if isinstance(v, list):
        return "an array"
    return "a scalar"


def load(path) -> dict:
    """Read and structurally validate a parameter file (the published JSON Schema)."""
    import jsonschema
    from .validate import schema_dir
    path = pathlib.Path(path)
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise BuildError(f"{path}: {e}") from None
    d = schema_dir()
    registry = None
    try:
        from referencing import Registry, Resource
        ast_schema = json.loads((d / "abstract-ast.schema.json").read_text())
        registry = Registry().with_resource("abstract-ast.schema.json", Resource.from_contents(ast_schema))
    except ImportError:  # old jsonschema without `referencing`: skip the cross-file refs
        pass
    schema = json.loads((d / "params.schema.json").read_text())
    if registry is not None:
        v = jsonschema.Draft202012Validator(schema, registry=registry)
        errors = [f"{path.name}: {'/'.join(map(str, e.absolute_path))}: {e.message[:300]}"
                  for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))]
        if errors:
            raise BuildError("parameter file rejected:\n  " + "\n  ".join(errors))
    if not isinstance(doc, dict) or doc.get("params_version") != PARAMS_VERSION or not isinstance(doc.get("params"), dict):
        raise BuildError(f"{path.name}: not a parameter file (params_version {PARAMS_VERSION} with a `params` object)")
    for k in doc:
        if k not in KEYS:
            raise BuildError(f"{path.name}: unknown key `{k}`")
    return doc


def check(ast: dict, pset: dict, *, ast_sha256: str | None = None, name: str = "params") -> list[str]:
    """The runner's rules, as a list of errors (empty when the set fits the abstract)."""
    errors = []
    if pset.get("abstract") != ast["name"]:
        errors.append(f"{name}: for abstract `{pset.get('abstract')}`, not `{ast['name']}`")
    want = pset.get("ast_sha256")
    if want and ast_sha256 and want != ast_sha256:
        errors.append(f"{name}: for AST {want[:16]}…, not {ast_sha256[:16]}…")
    declared = ast.get("params", {})
    for pname, v in pset.get("params", {}).items():
        if pname == "gpus":
            errors.append(f"{name}: `gpus` is set with --gpus, not by a parameter file")
            continue
        if pname not in declared:
            errors.append(f"{name}: no parameter `{pname}` in `{ast['name']}`")
            continue
        default = declared[pname]["default"]
        if kind_of(default) != kind_of(v):
            errors.append(f"{name}: {pname}: the default is {kind_of(default)}, the value is {kind_of(v)}")
        elif isinstance(default, list) and default:
            for i, x in enumerate(v):
                if kind_of(x) != kind_of(default[0]):
                    errors.append(f"{name}: {pname}[{i}]: the default's elements are {kind_of(default[0])}, this one is {kind_of(x)}")
                    break
    return errors


def defaults(ast: dict, *, doc: str | None = None, pin: bool = False) -> dict:
    """The abstract's defaults as a parameter set: the starting point for a fitted set."""
    from .emit import sha256
    out = {"params_version": PARAMS_VERSION, "abstract": ast["name"]}
    if pin:
        out["ast_sha256"] = sha256(ast)
    out["doc"] = doc or f"the defaults of `{ast['name']}` as declared in the abstract"
    out["params"] = {k: p["default"] for k, p in ast.get("params", {}).items()}
    return out


def render(pset: dict, provenance: dict | None = None) -> str:
    doc = {k: pset[k] for k in KEYS if k in pset and k != "provenance"}
    if provenance:
        doc["provenance"] = provenance
    return json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# safetensors: model_load's tensor table from the real shards (source 1, configuration)
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
    """(header length N, header JSON) of a safetensors file: 8 bytes little-endian N, then N bytes."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return n, header


def safetensors_table(shards: list, *, tp: int = 8, column: str = COLUMN_RE, row: str = ROW_RE) -> dict:
    """The `model_load` parameters for these shard files: one row per tensor, shards in the
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
    from .emit import load, sha256
    ast = load(path)
    return ast, sha256(ast)


# npz: train_large_samples' archive framing from real files (source 1, configuration)
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
    size = path.stat().st_size
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aeiou-params", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"aeiou-params {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("defaults", help="write the abstract's defaults as a parameter set")
    d.add_argument("ast", type=pathlib.Path)
    d.add_argument("-o", "--out", type=pathlib.Path, help="output file (default: <abstract>.defaults.params.json)")
    d.add_argument("--doc")
    d.add_argument("--pin", action="store_true", help="record the AST hash so the set is for this exact shape")
    c = sub.add_parser("check", help="validate parameter files against an abstract")
    c.add_argument("ast", type=pathlib.Path)
    c.add_argument("files", nargs="+", type=pathlib.Path)
    s = sub.add_parser("safetensors", help="build model_load's tensor table from safetensors shards")
    s.add_argument("ast", type=pathlib.Path, help="the model_load abstract (its parameter names are checked)")
    s.add_argument("shards", nargs="+", type=pathlib.Path, help="shard files in dataset id order")
    s.add_argument("-o", "--out", type=pathlib.Path, required=True)
    s.add_argument("--tp", type=int, default=8)
    s.add_argument("--column", default=COLUMN_RE, help="regex of column-parallel tensors")
    s.add_argument("--row", default=ROW_RE, help="regex of row-parallel tensors")
    s.add_argument("--doc")
    s.add_argument("--pin", action="store_true")
    n = sub.add_parser("npz", help="read train_large_samples' framing and cd_len from real .npz archives and compare them with the abstract's defaults")
    n.add_argument("ast", type=pathlib.Path, help="the train_large_samples abstract")
    n.add_argument("archives", nargs="+", type=pathlib.Path, help="archives of the corpus (a few are enough; all must agree)")
    n.add_argument("-o", "--out", type=pathlib.Path, help="write a parameter file with the two values")
    n.add_argument("--member", default="x", help="the member the application reads (default x)")
    n.add_argument("--doc")
    n.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    try:
        return _main(a)
    except BuildError as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 1


def _main(a) -> int:
    ast, sha = _load_ast(a.ast)
    if a.cmd == "defaults":
        pset = defaults(ast, doc=a.doc, pin=a.pin)
        out = a.out or pathlib.Path(f"{ast['name']}.defaults.params.json")
        out.write_text(render(pset, {"tool": f"aeiou-params {__version__}", "command": "defaults", "ast_sha256": sha}))
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
        pset["params"] = values
        errors = check(ast, pset, ast_sha256=sha, name=a.out.name)
        if errors:
            raise BuildError("\n  ".join(errors))
        prov = {"tool": f"aeiou-params {__version__}", "command": "safetensors", "ast_sha256": sha,
                "shards": [{"file": p.name, "sha256": file_sha256(p)} for p in a.shards],
                "column": a.column, "row": a.row}
        a.out.write_text(render(pset, prov))
        print(f"wrote {a.out}  ({len(values['shard'])} tensors in {values['shards']} shard(s))")
        return 0
    if a.cmd == "npz":
        values, recs = npz_params(a.archives, a.member)
        defaults_ = {k: ast["params"][k]["default"] for k in values if k in ast.get("params", {})}
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
            pset["params"] = values
            errors = check(ast, pset, ast_sha256=sha, name=a.out.name)
            if errors:
                raise BuildError("\n  ".join(errors))
            prov = {"tool": f"aeiou-params {__version__}", "command": "npz", "ast_sha256": sha, "member": a.member,
                    "archives": [{"file": p.name, "bytes": r["bytes"], "members": r["members"]} for p, r in zip(a.archives, recs)]}
            a.out.write_text(render(pset, prov))
            print(f"wrote {a.out}")
        return 2 if differs and not a.out else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
