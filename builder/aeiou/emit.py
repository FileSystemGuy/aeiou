"""Canonical form, hash, provenance, and the JSON writer.

One format on the contract: the AST is JSON on disk (pretty-printed, keys in the builder's
order) and JSON in canonical form for the hash (sorted keys, no whitespace). See
schema/README.md §1 and DESIGN_REVIEW.md §3.20 for why YAML was dropped."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import platform
import re
import subprocess
import sys

from . import __version__
from .nodes import BuildError

# Values the hermetic harness computes before it installs the audit hooks (git and the
# lockfile need subprocesses and file reads the sandbox denies).
PRECOMPUTED: dict = {}


def canonical(ast: dict) -> bytes:
    """Sorted keys, no whitespace, ASCII escapes, floats in Python repr, provenance removed.
    Identical to schema/check.py."""
    stripped = {k: v for k, v in ast.items() if k != "provenance"}
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def sha256(ast: dict) -> str:
    return hashlib.sha256(canonical(ast)).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def git_describe(path) -> str | None:
    """Short HEAD of the repository holding `path`, with `-dirty` if `path` has uncommitted
    changes or is untracked. None outside a repository."""
    path = pathlib.Path(path).resolve()
    try:
        head = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        if head.returncode:
            return None
        status = subprocess.run(["git", "-C", str(path.parent), "status", "--porcelain", "--", str(path)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        dirty = "-dirty" if status.stdout.strip() else ""
        return head.stdout.strip() + dirty
    except (OSError, subprocess.SubprocessError):
        return None


def find_lock(start) -> pathlib.Path | None:
    for parent in [pathlib.Path(start).resolve(), *pathlib.Path(start).resolve().parents]:
        for name in ("uv.lock", "requirements.lock"):
            if (parent / name).exists():
                return parent / name
    return None


def provenance(script, ast: dict, *, built_twice: bool | None = None) -> dict:
    """The provenance block (not part of the hash)."""
    script = pathlib.Path(script)
    pre = PRECOMPUTED.get(str(script.resolve()), {})
    gen = {"script": script.name, "sha256": pre.get("script_sha256") or file_sha256(script),
           "builder_version": __version__}
    git = pre["git"] if "git" in pre else git_describe(script)
    if git:
        gen["git"] = git
    out = {"generator": gen, "python": platform.python_version()}
    lock = pre.get("lock")
    if lock is None and "lock" not in pre:
        lp = find_lock(script.parent)
        lock = {"file": lp.name, "sha256": file_sha256(lp)} if lp else None
    if lock:
        out["lock"] = lock
    if built_twice is not None:
        out["built_twice_identical"] = built_twice
    out["ast_sha256"] = sha256(ast)
    return out


def render(ast: dict, prov: dict | None = None) -> str:
    """The on-disk form: JSON, two-space indent, the builder's key order, provenance last,
    trailing newline. Non-ASCII is kept as-is; the canonical form escapes it."""
    doc = dict(ast)
    if prov:
        doc["provenance"] = prov
    return json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _no_constant(name):
    raise ValueError(f"`{name}` is not JSON")


def _finite(text):
    x = float(text)
    if math.isinf(x):
        raise ValueError(f"number {text} is out of range")
    return x


def _no_duplicates(pairs):
    """An object whose key appears twice: serde_json refuses a repeated field of a struct
    ("duplicate field") and keeps the last of a repeated key of a map (`datasets`,
    `params`, a parameter file's top level); Python's `json` keeps the last of both. No
    writer of these files repeats a key, so every repeat is refused, the map's too."""
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"duplicate field `{k}`")
        seen[k] = v
    return seen


def _no_lone_surrogate(doc):
    """serde_json refuses a `\\u` escape of half a UTF-16 pair; Python's `json` keeps it as a
    lone surrogate in the string (a whole pair it joins into one character, so a surrogate
    left in a key or a value is a lone one)."""
    stack = [doc]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str) and (m := _SURROGATE.search(x)):
            raise ValueError(f"lone surrogate in hex escape (\\u{ord(m.group()):04x})")
    return doc


_SURROGATE = re.compile("[\ud800-\udfff]")


def loads(text):
    """JSON as the runner's serde_json reads it (schema/check.py's `loads`): Python's `json`
    also takes `NaN`, `Infinity`, and `-Infinity`, and turns a number past the 64-bit float
    range (`1e400`) into an infinity; serde_json refuses all four, and so does this, and a
    repeated key or a lone surrogate in a string as well. Every reader of an abstract or a
    parameter file in the builder's tools reads with it."""
    return _no_lone_surrogate(json.loads(text, parse_constant=_no_constant, parse_float=_finite,
                                         object_pairs_hook=_no_duplicates))


def load(path) -> dict:
    """An abstract from its `.ast.json`. A path that is no abstract is a usage error
    (`usage.py`, as the runner's `aeiou::load`): a builder script, a file that does not
    exist, a file that is not JSON; each message says what an abstract is."""
    from .usage import UsageError, WHAT_AN_ABSTRACT
    path = pathlib.Path(path)
    if path.suffix == ".py":
        raise UsageError(f"{path}: a builder script, not an abstract; `aeiou-build {path}` writes the abstracts it authors "
                         f"(`<name>.ast.json`) next to it, and those are what this tool takes")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        hint = "" if path.suffix == ".json" else f"; {WHAT_AN_ABSTRACT}"
        raise UsageError(f"{path}: {e.strerror} (os error {e.errno}){hint}") from None
    try:
        return loads(text)
    except ValueError as e:
        raise UsageError(f"{path}: not an abstract ({e}); {WHAT_AN_ABSTRACT}") from None


def write(wl, path=None, *, provenance: bool = True, built_twice: bool | None = None):
    ast = wl.build()
    path = pathlib.Path(path) if path else pathlib.Path(f"{wl.name}.ast.json")
    if path.is_dir():
        path = path / f"{wl.name}.ast.json"
    prov = None
    if provenance:
        prov = globals()["provenance"](wl.source, ast, built_twice=built_twice)
    text = render(ast, prov)
    if len(text) > 16 * 1024 * 1024:
        raise BuildError("AST too large")
    path.write_text(text, encoding="utf-8")
    return path
