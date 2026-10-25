"""`aeiou-catalog`: the workload catalog's records, their dictionary, and what is derived from them.

`workloads/` holds one directory per I/O mechanism and, inside it, one directory per case (an
AST and a parameter set: a workload a user would ask for) with its captures
(`DESIGN_REVIEW.md` §3.66). Building is a compiler's problem (`aeiou-build`, a script to an
AST); the catalog is a data-management one: curated records, the vocabulary they are written
in, and their consistency with the abstracts they describe.

    aeiou-catalog check [--workloads DIR] [--runner PATH] [--no-shapes]   validate, re-derive, exit 1 on drift
    aeiou-catalog write [--workloads DIR] [--runner PATH] [--no-shapes]   re-derive and write
    aeiou-catalog describe [--params-file FILE] [--gpus N] [--runner PATH] [--no-shapes] AST   print its description

The records, each a TOML file:

- `<category>/[<class>/]<mechanism>/mechanism.toml`: the mechanism's curated attributes
  (`category`, `class` where the category has classes, `application`, `format`, `phase`,
  `domain`), and a `[defaults]` table (`ast`, `gpus`, `length`) when it has no
  case yet, so the catalog still shows it, at the abstract's defaults.
- `cases/<case>/case.toml`: the case's curated attributes (`kind`, `modality`), the GPU counts
  its parameters hold at (`gpus`), its `length` parameter (the one that only makes the run
  longer) when it has one, and one `[[run]]` per AST it runs (`ast`, `params`, a file of the
  case or a list of the case's and its mechanism's).
- `captures/<capture>/capture.toml`: `raw_trace` (a pointer or "not retained") and one
  `[[pair]]` per trace of the capture: `trace` (the stem of its `.metrics.json`), `ast`
  (one of the case's runs), `params` (a file of the capture, when its fit differs from the
  case's), and `verdict`, what `aeiou-trace compare --judge` gives (`builder/tests/test_trace.py`
  checks it).

`workloads/attributes.toml` defines every attribute with its meaning, source, level, and
values; a record may use only attributes and values defined there, and only the curated
ones. The derived attributes are read from the AST and the captures; the measured ones from
`aeiou dry-run --metrics-json` of each run (`shape.json` in the case, `shape.defaults.json`
in a mechanism with no case) at the case's own parameters: the median of seeds 1, 2, and 3,
their spread, and the numbers they disagree on beyond a tolerance. A shorter run is never
taken for the case (on a fixed dataset reuse grows with the run, and a fixed load is amortized
only over a long one), but for a case with a length parameter `write` also halves it, down to
1/64, while the numbers stay the case's within their tolerances, and records the smallest
such length: `check` re-measures there, so CI never pays three seeds of a long case; the op
counts, bytes, and datasets' projection the shape also records (the first seed's at the case's
own length, which the descriptions print) are compared exactly, from one more dry run, the
first seed's at the case's length, when the check length is shorter. `CATALOG.md` is
rendered from the records, the shapes, and the dictionary. `check` re-measures every shape and re-renders the catalog, and fails when a
committed one differs; `--no-shapes` skips the dry runs (no runner needed) and checks instead
that each shape was measured on the AST, parameter file, and case record committed now.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import heapq
import itertools
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib

from .params import values_of
from . import __version__, checkout, emit, lock, params, usage
from .describe import Describer
from .nodes import BuildError
from .validate import validate

SHAPE_VERSION = 3               # 2 (2026-10-14): the op counts, bytes, and each dataset's projection, which the description quotes;
                                # 3 (2026-10-22): reread and skew over extents, no block (DESIGN_REVIEW.md §3.79)
SEEDS = (1, 2, 3)
START = 64                      # the shortest check length tried is the case's over this
METADATA = ("open", "close", "stat", "fstat", "readdir", "mkdir", "rmdir", "unlink", "rename", "ftruncate", "fallocate")
DATA = ("read", "write")
MECHANISM_KEYS = {"defaults"}
DEFAULTS_KEYS = {"ast", "gpus", "length"}
CASE_KEYS = {"gpus", "length", "run"}
RUN_KEYS = {"ast", "params"}
CAPTURE_KEYS = {"raw_trace", "pair"}
PAIR_KEYS = {"trace", "ast", "params", "verdict"}
BEST = ("accepted", "not-accepted", "nothing-judged")       # status, best first


# ---------------------------------------------------------------- the dictionary

def load_dictionary(root: pathlib.Path) -> dict:
    path = root / "attributes.toml"
    try:
        d = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise BuildError(f"{path}: {e}")
    attrs = d.get("attribute", {})
    for name, a in attrs.items():
        if a.get("source") not in ("curated", "derived", "measured") or a.get("level") not in ("mechanism", "case"):
            raise BuildError(f"{path}: attribute {name}: source must be curated, derived, or measured, level mechanism or case")
        if a["source"] == "measured":
            if name not in MEASURES:
                raise BuildError(f"{path}: attribute {name}: measured, but aeiou-catalog has no measure of that name")
            if not a.get("classes") or any("name" not in c for c in a["classes"]):
                raise BuildError(f"{path}: attribute {name}: a measured attribute needs classes, each with a name")
        elif not a.get("values"):
            raise BuildError(f"{path}: attribute {name}: needs values")
        if a.get("multiple") and a["source"] != "curated":
            raise BuildError(f"{path}: attribute {name}: only a curated attribute takes several values")
    for name, a in attrs.items():
        # required: always, never, or when another curated attribute of the record has a value
        r = a.get("required", False)
        if not isinstance(r, bool) and not (
                isinstance(r, dict) and len(r) == 1 and (k := next(iter(r))) in attrs and attrs[k]["source"] == "curated"
                and attrs[k]["level"] == a["level"] and r[k] in attrs[k].get("values", {})):
            raise BuildError(f"{path}: attribute {name}: required is true, false, or a table of one curated attribute of its level and a value of it")
    cats = attrs.get("category", {}).get("values", {})
    for v, d in attrs.get("class", {}).get("values", {}).items():
        if d.get("category") not in cats:
            raise BuildError(f"{path}: class {v}: its category {d.get('category')!r} is not a value of category")
    missing = sorted(set(MEASURES) - {n for n, a in attrs.items() if a["source"] == "measured"})
    if missing:
        raise BuildError(f"{path}: aeiou-catalog measures {', '.join(missing)}, which the dictionary does not define")
    return attrs


def classify(attr: dict, value) -> str | None:
    """The first class whose bounds hold (inclusive); none for a number that is not there."""
    if value is None:
        return None
    for c in attr["classes"]:
        if ("min" not in c or value >= c["min"]) and ("max" not in c or value <= c["max"]):
            return c["name"]
    return None


def agree(attr: dict, a, b) -> bool:
    """Whether two measurements of one attribute are the same within its tolerance."""
    if a is None or b is None:
        return a is None and b is None
    tol = attr.get("tolerance")
    if tol is None:
        return a == b
    # either bound is enough: a relative one alone would hold a ratio near zero to noise no class sees
    return abs(a - b) <= max(tol.get("relative", 0) * max(abs(a), abs(b)), tol.get("absolute", 0))


# ---------------------------------------------------------------- the measures (of a dry run's `total`)

def _weighted_median(h: dict):
    """The byte-weighted median of a histogram of sizes: the bucket (its lower bound) at or
    above which half the bytes lie. A count median misleads here (np.load's many header reads
    make 140 MiB samples look like 112-byte runs)."""
    weights = [(lo, lo * n) for lo, n in h["buckets"]]
    total = sum(w for _, w in weights)
    if not total:
        return None
    acc = 0
    for lo, w in weights[:-1]:
        acc += w
        if 2 * acc >= total:
            return lo
    return weights[-1][0]   # the last bucket brings the sum to the total


def _accesses(r: dict, this: str) -> int:
    return r["first_touch"][this] + sum(r[f"{this}_after_{prev}"]["n"] for prev in DATA)


def _share(a, b):
    return None if not b else a / b


def _data_ops(t: dict) -> int:
    return sum(t["counts"].get(k, 0) for k in DATA)


MEASURES = {
    "run_length_read": lambda t: _weighted_median(t["run_length"]["read"]["bytes"]),
    "run_length_write": lambda t: _weighted_median(t["run_length"]["write"]["bytes"]),
    "request_size_read": lambda t: _weighted_median(t["request_size"]["read"]),
    "request_size_write": lambda t: _weighted_median(t["request_size"]["write"]),
    "read_share": lambda t: _share(t["bytes_read"], t["bytes_read"] + t["bytes_written"]),
    "metadata": lambda t: _share(sum(t["counts"].get(k, 0) for k in METADATA), _data_ops(t)),
    "reads_own_writes": lambda t: 1.0 if t["reuse_extents"]["read_after_write"]["n"] else 0.0,
    "reread": lambda t: _share(t["reuse_extents"]["read_after_read"]["n"], _accesses(t["reuse_extents"], "read")),
    "skew": lambda t: t["popularity_bytes"]["top_1_pct"] if t["popularity_bytes"]["accesses"] else None,
}


def measure(total: dict) -> dict:
    return {name: f(total) for name, f in MEASURES.items()}


# ---------------------------------------------------------------- the records

class Problems(list):
    def add(self, path: pathlib.Path, root: pathlib.Path, message: str) -> None:
        self.append(f"{_rel(path, root)}: {message}")


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _toml(path: pathlib.Path, root: pathlib.Path, problems: Problems) -> dict | None:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        problems.add(path, root, str(e))
        return None


def _attributes(rec: dict, level: str, structural: set, attrs: dict, path, root, problems: Problems) -> dict:
    """The record's curated attributes, checked against the dictionary."""
    out = {}
    for key, value in rec.items():
        if key in structural:
            continue
        a = attrs.get(key)
        if a is None:
            problems.add(path, root, f"`{key}` is not an attribute of workloads/attributes.toml")
        elif a["source"] != "curated":
            problems.add(path, root, f"`{key}` is {a['source']}: aeiou-catalog computes it, a record never states it")
        elif a["level"] != level:
            problems.add(path, root, f"`{key}` is an attribute of a {a['level']}, not of a {level}")
        elif a.get("multiple") and isinstance(value, list):
            bad = [v for v in value if not isinstance(v, str) or v not in a["values"]]
            if not value or bad or len(set(value)) != len(value):
                problems.add(path, root, f"`{key}` = {value!r}: a list of distinct values among {', '.join(a['values'])}")
            else:
                out[key] = value
        elif not isinstance(value, str) or value not in a["values"]:
            several = " (or a list of them)" if a.get("multiple") else ""
            problems.add(path, root, f"`{key}` = {value!r}: not one of {', '.join(a['values'])}{several}")
        else:
            out[key] = [value] if a.get("multiple") else value
    for key, a in attrs.items():
        if a["source"] != "curated" or a["level"] != level or key in rec:
            continue
        r = a.get("required")
        if r is True:
            problems.add(path, root, f"`{key}` is required")
        elif isinstance(r, dict) and rec.get(k := next(iter(r))) == r[k]:
            problems.add(path, root, f"`{key}` is required when `{k}` is {r[k]}")
    return out


def classes_of(attrs: dict, category: str) -> list[str]:
    """The class values of a category, in the dictionary's order."""
    return [v for v, d in attrs.get("class", {}).get("values", {}).items() if d.get("category") == category]


def check_tree(root: pathlib.Path, attrs: dict, problems: Problems) -> None:
    """Every directory of the tree's first level is a category, every category has its directory
    (with no mechanism yet, its README says what belongs there), and a category with classes holds
    nothing but its class directories."""
    have = {d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))}
    cats = list(attrs["category"]["values"])
    for d in sorted(have - set(cats)):
        problems.add(root / d, root, f"not a category of workloads/attributes.toml ({', '.join(cats)})")
    for c in cats:
        if c not in have:
            problems.add(root / c, root, "a category without its directory (aeiou-catalog write makes it)")
            continue
        classes = classes_of(attrs, c)
        if classes:
            sub = {d.name for d in (root / c).iterdir() if d.is_dir() and not d.name.startswith((".", "_"))}
            for d in sorted(sub - set(classes)):
                problems.add(root / c / d, root, f"not a class of {c} ({', '.join(classes)})")
            for k in classes:
                if k not in sub:
                    problems.add(root / c / k, root, "a class without its directory (aeiou-catalog write makes it)")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params_of(ast: dict) -> dict:
    return ast.get("params") or {}


def _length_ok(ast: dict, name: str) -> bool:
    """`name` is an integer parameter: its default an integer, or required of kind `int` (a case gives it)."""
    p = _params_of(ast).get(name)
    if isinstance(p, dict) and "default" not in p:
        return p.get("kind") == "int"
    return isinstance(p, dict) and isinstance(p.get("default"), int) and not isinstance(p.get("default"), bool)


def load(root: pathlib.Path, attrs: dict) -> tuple[list[dict], Problems]:
    """Every mechanism with its cases and their captures, checked; the problems found."""
    problems = Problems()
    check_tree(root, attrs, problems)
    mech_dirs = sorted({p.parent for p in root.rglob("*.ast.json") if "kit" not in p.relative_to(root).parts})
    recorded = sorted(p.parent for p in root.rglob("mechanism.toml"))
    for d in sorted(set(recorded) - set(mech_dirs)):
        problems.add(d / "mechanism.toml", root, "a mechanism.toml where no abstract is")
    mechanisms = []
    for d in mech_dirs:
        path = d / "mechanism.toml"
        if not path.exists():
            problems.add(d, root, "a mechanism without mechanism.toml")
            continue
        rec = _toml(path, root, problems)
        if rec is None:
            continue
        m = {"dir": d, "rel": _rel(d, root), "attributes": _attributes(rec, "mechanism", MECHANISM_KEYS, attrs, path, root, problems),
             "asts": {}, "cases": [], "defaults": None}
        parts = d.relative_to(root).parts
        cat, cls = m["attributes"].get("category"), m["attributes"].get("class")
        if cat and cls and classes_of(attrs, cat) and cls not in classes_of(attrs, cat):
            problems.add(path, root, f"class {cls!r} is not a class of {cat} ({', '.join(classes_of(attrs, cat))})")
        elif cat and cls and not classes_of(attrs, cat):
            problems.add(path, root, f"class {cls!r}: {cat} has no classes")
        elif cat and classes_of(attrs, cat) and not cls:
            problems.add(path, root, f"{cat} has classes ({', '.join(classes_of(attrs, cat))}): the mechanism names its class")
        elif cat and (parts[:-1] != ((cat, cls) if cls else (cat,))):
            problems.add(path, root, f"category {cat!r}" + (f" and class {cls!r}" if cls else "") +
                         f" say workloads/{cat}/{cls + '/' if cls else ''}<mechanism>, not workloads/{m['rel']}")
        for a in sorted(d.glob("*.ast.json")):
            try:
                doc = emit.loads(a.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                problems.add(a, root, str(e))
                continue
            m["asts"][doc.get("name")] = {"path": a, "doc": doc, "sha256": _sha256(a)}
        for c in sorted((d / "cases").glob("*/")) if (d / "cases").is_dir() else ():
            case = _load_case(c, m, attrs, root, problems)
            if case:
                m["cases"].append(case)
        dflt = rec.get("defaults")
        if dflt is not None:
            if m["cases"]:
                problems.add(path, root, "[defaults] is for a mechanism with no case; this one has cases")
            elif _check_keys(dflt, DEFAULTS_KEYS, "[defaults]", path, root, problems):
                if dflt.get("ast") not in m["asts"]:
                    problems.add(path, root, f"[defaults] ast {dflt.get('ast')!r} is not an abstract of this mechanism")
                elif "length" in dflt and not _length_ok(m["asts"][dflt["ast"]]["doc"], dflt["length"]):
                    problems.add(path, root, f"[defaults] length {dflt['length']!r} is not an integer parameter of {dflt['ast']}")
                elif not _gpus_ok(dflt.get("gpus", 1)):
                    problems.add(path, root, "[defaults] gpus must be a positive integer")
                else:
                    m["defaults"] = {"ast": dflt["ast"], "gpus": dflt.get("gpus", 1), "length": dflt.get("length")}
        elif not m["cases"]:
            problems.add(path, root, "a mechanism with no case needs a [defaults] table (ast, gpus, length), its catalog row")
        mechanisms.append(m)
    names = {}
    for m in mechanisms:
        for name, a in m["asts"].items():
            if name in names:
                problems.add(a["path"], root, f"the AST name {name} is also {_rel(names[name], root)}")
            names[name] = a["path"]
    # one lock to a name, so `aeiou run --case NAME` finds one (§3.78 point 13)
    locks = {}
    for m in mechanisms:
        for case in m["cases"]:
            name = f"{m['dir'].name}/{case['name']}"
            if name in locks:
                problems.add(case["dir"] / lock.LOCK_FILE, root, f"the lock name {name} is also {_rel(locks[name], root)}'s")
            locks[name] = case["dir"]
    return mechanisms, problems


def _gpus_ok(g) -> bool:
    return isinstance(g, int) and not isinstance(g, bool) and g >= 1


def _check_keys(rec: dict, allowed: set, what: str, path, root, problems: Problems) -> bool:
    extra = sorted(set(rec) - allowed)
    if extra:
        problems.add(path, root, f"{what}: unknown key(s) {', '.join(extra)} (allowed: {', '.join(sorted(allowed))})")
    return not extra


def _load_case(c: pathlib.Path, m: dict, attrs: dict, root, problems: Problems) -> dict | None:
    path = c / "case.toml"
    if not path.exists():
        problems.add(c, root, "a case without case.toml")
        return None
    rec = _toml(path, root, problems)
    if rec is None:
        return None
    case = {"dir": c, "rel": _rel(c, root), "name": c.name, "mechanism": m,
            "attributes": _attributes(rec, "case", CASE_KEYS, attrs, path, root, problems),
            "gpus": rec.get("gpus"), "length": rec.get("length"),
            "runs": [], "captures": []}
    if not isinstance(case["gpus"], list) or not case["gpus"] or not all(_gpus_ok(g) for g in case["gpus"]):
        problems.add(path, root, "gpus must be a non-empty list of positive integers: the GPU counts its parameters hold at")
        case["gpus"] = [1]
    runs = rec.get("run") or []
    if not runs:
        problems.add(path, root, "a case runs at least one AST: a [[run]] with ast and params")
    for r in runs:
        if not _check_keys(r, RUN_KEYS, "[[run]]", path, root, problems):
            continue
        ast = m["asts"].get(r.get("ast"))
        if ast is None:
            problems.add(path, root, f"[[run]] ast {r.get('ast')!r} is not an abstract of {m['rel']}")
            continue
        # its fragments (contract 0.11): one file or a list, each the case's or its mechanism's
        frags = r.get("params")
        frags = [frags] if isinstance(frags, str) else frags if isinstance(frags, list) else [None]
        files = [c / str(f) for f in frags]
        bad = [f for f, p in zip(frags, files) if not isinstance(f, str) or not p.is_file() or m["dir"].resolve() not in p.resolve().parents]
        if bad:
            problems.add(path, root, f"[[run]] params {bad[0]!r} is not a file of the case or its mechanism")
            continue
        wrong = False
        for f in files:
            try:
                pset = emit.loads(f.read_text(encoding="utf-8"))
            except ValueError as e:
                problems.add(f, root, str(e))
                wrong = True
                continue
            if pset.get("abstract") != r["ast"]:
                problems.add(f, root, f"its abstract is {pset.get('abstract')!r}, the case runs it as {r['ast']}")
        if wrong:
            continue
        # what a run of the case reads, and what its shape is measured through: the lock
        params = c / lock.LOCK_FILE
        if case["length"] is not None and not _length_ok(ast["doc"], case["length"]):
            problems.add(path, root, f"length {case['length']!r} is not an integer parameter of {r['ast']}")
            case["length"] = None
        case["runs"].append({"ast": r["ast"], "params": params if params.is_file() else None, "fragments": files, "abstract": ast})
    if len({r["ast"] for r in case["runs"]}) != len(case["runs"]):
        problems.add(path, root, "two runs of one AST: a case runs each abstract once")
    named = {f.name for r in case["runs"] for f in r["fragments"] if f.parent == c}
    for p in sorted(c.glob("*params.json")):
        if p.name not in named:
            problems.add(p, root, "a parameter file no [[run]] of case.toml names")
    # the use case's lock (DESIGN_REVIEW.md §3.78 point 17): there, and the one its inputs make now
    if case["runs"] and (why := lock.stale(c, workloads=root)):
        problems.add(c / lock.LOCK_FILE, root, why)
    caps = c / "captures"
    for cap in sorted(caps.glob("*/")) if caps.is_dir() else ():
        capture = _load_capture(cap, case, attrs, root, problems)
        if capture:
            case["captures"].append(capture)
    return case


def _load_capture(cap: pathlib.Path, case: dict, attrs: dict, root, problems: Problems) -> dict | None:
    path = cap / "capture.toml"
    if not path.exists():
        problems.add(cap, root, "a capture without capture.toml")
        return None
    rec = _toml(path, root, problems)
    if rec is None or not _check_keys(rec, CAPTURE_KEYS, "capture.toml", path, root, problems):
        return None
    if not isinstance(rec.get("raw_trace"), str):
        problems.add(path, root, "raw_trace: where the raw trace is (a URL with its SHA-256), or \"not retained\"")
    verdicts = [v for v in attrs["status"]["values"] if v != "untraced"]
    runs = {r["ast"]: r for r in case["runs"]}
    pairs = []
    for p in rec.get("pair") or []:
        if not _check_keys(p, PAIR_KEYS, "[[pair]]", path, root, problems):
            continue
        metrics = cap / f"{p.get('trace')}.metrics.json"
        if not metrics.is_file():
            problems.add(path, root, f"[[pair]] trace {p.get('trace')!r}: no {metrics.name} in the capture")
            continue
        if p.get("ast") not in runs:
            problems.add(path, root, f"[[pair]] ast {p.get('ast')!r} is not one of the case's runs ({', '.join(runs)})")
            continue
        params = cap / p["params"] if "params" in p else runs[p["ast"]]["params"]      # the case's lock
        if params is None:
            continue                # a case with no lock, refused where its runs are read
        if not params.is_file():
            problems.add(path, root, f"[[pair]] params {p['params']!r} is not a file of the capture")
            continue
        if p.get("verdict") not in verdicts:
            problems.add(path, root, f"[[pair]] verdict {p.get('verdict')!r}: not one of {', '.join(verdicts)}")
            continue
        pairs.append({"trace": p["trace"], "ast": p["ast"], "params": params, "verdict": p["verdict"]})
    have = {q.name[: -len(".metrics.json")] for q in cap.glob("*.metrics.json")}
    for stem in sorted(have - {p["trace"] for p in pairs}):
        problems.add(cap / f"{stem}.metrics.json", root, "a trace no [[pair]] of capture.toml judges")
    for q in sorted(cap.glob("*.tolerances.json")):
        if q.name[: -len(".tolerances.json")] not in have:
            problems.add(q, root, "a tolerance file with no trace beside it")
    return {"dir": cap, "rel": _rel(cap, root), "pairs": pairs}


def capture_pairs(root: pathlib.Path) -> list[tuple]:
    """Every judged pair, for builder/tests/test_trace.py: (capture dir, trace stem, AST,
    parameter file, GPUs, verdict as compare prints it)."""
    attrs = load_dictionary(root)
    mechanisms, problems = load(root, attrs)
    if problems:
        raise BuildError("\n".join(problems))
    return [(cap["dir"], p["trace"], p["ast"], p["params"], case["gpus"][0], p["verdict"].replace("-", " "))
            for m in mechanisms for case in m["cases"] for cap in case["captures"] for p in cap["pairs"]]


# ---------------------------------------------------------------- derived

def derived(abstract: dict, verdicts: list[str]) -> dict:
    doc = abstract["doc"]
    ns = (doc.get("namespaces") or {}).values()
    return {
        "api": doc.get("posix") or "sync",
        "writes": "yes" if any(not n.get("input") for n in ns) else "no",
        "inputs": "yes" if any(n.get("input") for n in ns) else "no",
        "status": next((v for v in BEST if v in verdicts), "untraced"),
    }


# ---------------------------------------------------------------- measured: the shape

def runner_args(ast: pathlib.Path, name: str, params: pathlib.Path | None) -> list[str]:
    """What names the workload to a dry run: the AST, with a parameter file, or a case's lock,
    as a run of the case reads it (`--case-run` when the lock has several runs)."""
    if params is None:
        return [str(ast)]
    if params.name != lock.LOCK_FILE:
        return [str(ast), "--params-file", str(params)]
    out = ["--case-file", str(params), "--workloads", str(lock.workloads_of(params))]
    if len(json.loads(params.read_text(encoding="utf-8")).get("runs", [])) > 1:
        out += ["--case-run", name]
    return out


def _dry_run(runner: str, run: dict, gpus: int, seed: int, length: tuple | None, tmp: pathlib.Path) -> dict:
    out = tmp / f"{run['abstract']['doc']['name']}.{seed}.{length[1] if length else 'case'}.json"
    cmd = [runner, "dry-run", *runner_args(run["abstract"]["path"], run.get("ast"), run["params"]), "--gpus", str(gpus), "--seed", str(seed),
           "--metrics-json", str(out)]
    if length:
        cmd += ["--param", f"{length[0]}={length[1]}"]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, encoding="utf-8", errors="replace")
    if p.returncode:
        raise BuildError(f"{' '.join(cmd)}: exit {p.returncode}\n{p.stderr.strip()}")
    return json.loads(out.read_text(encoding="utf-8"))


def _median(xs: list):
    """The median of the seeds' numbers (the middle one of three); none when a seed has none."""
    if any(x is None for x in xs):
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


class Scheduler:
    """`jobs` dry runs at once, from one queue: the highest cost first (a case's recorded op
    count; a case with none, infinity, first of all), in order of submission within one cost,
    so the longest case starts first and never ends the write alone. Nothing starts while it
    is `held`, so every case's first dry runs are queued before one is chosen. Leaving it waits
    on what runs; on an error what is still queued is dropped."""

    def __init__(self, jobs: int):
        self._queue, self._order = [], itertools.count()
        self._cv = threading.Condition()
        self._held = self._closed = False
        self._workers = [threading.Thread(target=self._work, daemon=True) for _ in range(jobs)]
        for w in self._workers:
            w.start()

    def submit(self, cost, fn, *args) -> concurrent.futures.Future:
        f = concurrent.futures.Future()
        with self._cv:
            heapq.heappush(self._queue, (-cost, next(self._order), f, fn, args))
            self._cv.notify()
        return f

    @contextlib.contextmanager
    def held(self):
        with self._cv:
            self._held = True
        try:
            yield self
        finally:
            with self._cv:
                self._held = False
                self._cv.notify_all()

    def _work(self) -> None:
        while True:
            with self._cv:
                while self._held or not (self._queue or self._closed):
                    self._cv.wait()
                if not self._queue:
                    return
                *_, f, fn, args = heapq.heappop(self._queue)
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(fn(*args))
                except BaseException as e:
                    f.set_exception(e)

    def __enter__(self):
        return self

    def __exit__(self, kind, *_):
        with self._cv:
            if kind is not None:
                for *_, f, _, _ in self._queue:
                    f.cancel()
            self._held, self._closed = False, True
            self._cv.notify_all()
        for w in self._workers:
            w.join()


def _cost(recorded: dict | None) -> float:
    """What orders a case's dry runs on the scheduler: the op count its shape recorded, at the
    case's own length; a case never measured, infinity (it may be the longest)."""
    ops = (recorded or {}).get("ops")
    return ops if isinstance(ops, int) else math.inf


def _start(runner: str, run: dict, gpus: int, length: tuple | None, pool, tmp, cost=math.inf) -> list:
    """The dry runs of every seed at one length, queued on the scheduler."""
    return [pool.submit(cost, _dry_run, runner, run, gpus, seed, length, tmp) for seed in SEEDS]


def _gathered(started: list) -> tuple[dict, list[dict], dict]:
    """The median of each measure over the seeds, each seed's numbers, and the first seed's dry run."""
    docs = [f.result() for f in started]
    values = [measure(d["total"]) for d in docs]
    return {n: _median([v[n] for v in values]) for n in MEASURES}, values, docs[0]


def _agrees(attrs: dict, a: dict, b: dict) -> list[str]:
    """The measures on which two sets of numbers differ beyond their tolerances."""
    return [n for n in MEASURES if not agree(attrs[n], a[n], b[n])]


def _levels(full: int | None) -> list[int]:
    """The lengths a check length may be, longest first: the case's own, then its halves down to 1/64."""
    if not full:
        return []
    out, n = [full], full // 2
    while n >= max(1, math.ceil(full / START)):
        out.append(n)
        n //= 2
    return out


def _first_level(levels: list[int], previous: dict | None) -> int:
    """The level a search starts at: the check length the shape recorded before (half the case's
    when that was the case's own), and for a new case, or a length no longer a level, the smallest."""
    check = ((previous or {}).get("length") or {}).get("check")
    if check not in levels:
        return len(levels) - 1
    return max(levels.index(check), 1) if len(levels) > 1 else 0


def begin_shape(runner: str, run: dict, gpus: int, length_param: str | None, pool, previous: dict | None = None) -> dict:
    """`shape`'s dry runs that never wait on a decision, queued: the case's own length's three
    seeds and, for a case with a length parameter, the first level's beside them."""
    pset = _values(run)
    full = pset.get(length_param, _params_of(run["abstract"]["doc"])[length_param].get("default")) if length_param else None
    levels = _levels(full)
    tmp = tempfile.TemporaryDirectory(prefix="aeiou-catalog.")
    state = {"run": run, "runner": runner, "gpus": gpus, "param": length_param, "full": full, "levels": levels,
             "pool": pool, "tmp": tmp, "cost": _cost(previous), "first": _first_level(levels, previous) if levels else 0}
    state["own"] = _start(runner, run, gpus, (length_param, full) if full else None, pool, pathlib.Path(tmp.name), state["cost"])
    state["started"] = {}
    if state["first"]:
        state["started"][state["first"]] = _start(runner, run, gpus, (length_param, levels[state["first"]]), pool, pathlib.Path(tmp.name), state["cost"])
    return state


def finish_shape(state: dict, attrs: dict) -> dict:
    """One run's shape (`shape.json`'s entry) from what `begin_shape` queued: the measures at
    the case's own parameters, the median over the seeds and its spread; and, for a case with
    a length parameter, its check length, where `check` re-measures it: the smallest level
    (1/2 to 1/64 of the case's) whose numbers are the case's own length's, each level judged on
    its three seeds against the full run, never against another level. The search starts at
    the check length recorded before: down while a level agrees, up while it differs; a new
    case starts at the smallest, since most shapes hold there or nowhere. A level is queued
    only once the one before it is decided, so no dry run is wasted. A shorter run is never
    taken for the case: a reuse share grows with the run on a fixed dataset, and a fixed load
    is amortized only over a long one."""
    run, levels, started, tmp = state["run"], state["levels"], state["started"], pathlib.Path(state["tmp"].name)
    try:
        med, seeds, first = _gathered(state["own"])

        def holds(i):
            if i == 0:
                return True
            if i not in started:
                started[i] = _start(state["runner"], run, state["gpus"], (state["param"], levels[i]), state["pool"], tmp, state["cost"])
            return not _agrees(attrs, _gathered(started[i])[0], med)
        i = state["first"]
        if holds(i):
            while i + 1 < len(levels) and holds(i + 1):
                i += 1
        else:
            i -= 1
            while not holds(i):
                i -= 1
        check = levels[i] if levels else None
    finally:
        # on an error the dry runs not begun are dropped, the rest finish before their directory goes
        waiting = [f for fs in started.values() for f in fs] + state["own"]
        for f in waiting:
            f.cancel()
        concurrent.futures.wait(waiting)
        state["tmp"].cleanup()
    full, length_param = state["full"], state["param"]
    # the numbers the seeds disagree on: the case's own statistical error is larger than the tolerance
    moving = [n for n in MEASURES if not all(agree(attrs[n], v[n], w[n]) for v in seeds for w in seeds)]
    spread = {n: None if med[n] is None else max(v[n] for v in seeds) - min(v[n] for v in seeds) for n in MEASURES}
    out = {"ast": run["abstract"]["doc"]["name"], "ast_sha256": run["abstract"]["sha256"],
           "params": run["params"].name if run["params"] else None, "params_sha256": _sha256(run["params"]) if run["params"] else None,
           "gpus": state["gpus"], "seeds": list(SEEDS),
           "length": None if full is None else {"param": length_param, "case": full, "check": check},
           "stable": not moving, "moving": moving, "ops": first["total"]["ops"],
           "counts": dict(sorted(first["total"]["counts"].items())), "bytes_read": first["total"]["bytes_read"],
           "bytes_written": first["total"]["bytes_written"], "datasets": first.get("datasets", []), "attributes": {}}
    for name in MEASURES:
        out["attributes"][name] = {"value": _round(med[name]), "spread": _round(spread[name]), "class": classify(attrs[name], med[name])}
    return out


def shape(runner: str, run: dict, gpus: int, length_param: str | None, attrs: dict, pool, previous: dict | None = None) -> dict:
    """One run's shape, measured on the scheduler `pool` (`begin_shape`, then `finish_shape`),
    its search started from `previous`, the entry the shape recorded before."""
    return finish_shape(begin_shape(runner, run, gpus, length_param, pool, previous), attrs)


COUNTED = ("ops", "counts", "bytes_read", "bytes_written", "datasets")   # the first seed's, at the case's length; exact


def _counted(first: dict) -> dict:
    """What a shape records of the first seed's dry run at the case's own length, as `shape` records it."""
    return {"ops": first["total"]["ops"], "counts": dict(first["total"]["counts"]), "bytes_read": first["total"]["bytes_read"],
            "bytes_written": first["total"]["bytes_written"], "datasets": first.get("datasets", [])}


def _short(recorded: dict) -> bool:
    """Whether a shape's check length is shorter than its case's: its counts then need a run of their own."""
    return recorded["length"] is not None and recorded["length"]["check"] != recorded["length"]["case"]


def _case_run(runner: str, run: dict, gpus: int, recorded: dict, tmp: pathlib.Path) -> dict:
    """The first seed's dry run at the case's own length, the one a shape's counts come from."""
    length = recorded["length"]
    return _dry_run(runner, run, gpus, SEEDS[0], (length["param"], length["case"]) if length else None, tmp)


def begin_recheck(runner: str, run: dict, gpus: int, recorded: dict, pool) -> dict:
    """`recheck`'s dry runs, queued: the three seeds at the recorded check length and, when it is
    shorter than the case's, the first seed's at the case's, which the counts are taken from."""
    length, cost = recorded["length"], _cost(recorded)
    tmp = tempfile.TemporaryDirectory(prefix="aeiou-catalog.")
    full = pool.submit(cost, _case_run, runner, run, gpus, recorded, pathlib.Path(tmp.name)) if _short(recorded) else None
    seeds = _start(runner, run, gpus, (length["param"], length["check"]) if length else None, pool, pathlib.Path(tmp.name), cost)
    return {"recorded": recorded, "tmp": tmp, "full": full, "seeds": seeds}


def finish_recheck(state: dict, attrs: dict) -> list[str]:
    """The measures and counts that differ from the recorded ones, from what `begin_recheck` queued."""
    recorded, waiting = state["recorded"], state["seeds"] + ([state["full"]] if state["full"] else [])
    try:
        med, _, first = _gathered(state["seeds"])
        if state["full"] is not None:
            first = state["full"].result()
    finally:
        for f in waiting:
            f.cancel()
        concurrent.futures.wait(waiting)
        state["tmp"].cleanup()
    counted = _counted(first)
    return (_agrees(attrs, med, {n: a["value"] for n, a in recorded["attributes"].items()})
            + [n for n in COUNTED if counted[n] != recorded.get(n)])


def recheck(runner: str, run: dict, gpus: int, recorded: dict, attrs: dict, pool) -> list[str]:
    """`check` with the runner: the run re-measured at its recorded check length, against
    the recorded numbers, and the first seed's op counts, bytes, and datasets' projection
    against the recorded ones exactly (the description and CATALOG.md print them); the
    measures and counts that differ. The counts are the case's own length's: when the check
    length is shorter, one more dry run, the first seed's at the case's length, gives them."""
    return finish_recheck(begin_recheck(runner, run, gpus, recorded, pool), attrs)


def _round(v):
    return None if v is None else (v if isinstance(v, int) else float(f"{v:.6g}"))


def render_shape(runs: list[dict]) -> str:
    return json.dumps({"shape_version": SHAPE_VERSION, "tool": "aeiou-catalog", "runs": runs}, indent=2) + "\n"


def _shape_path(m: dict, case: dict | None) -> pathlib.Path:
    return case["dir"] / "shape.json" if case else m["dir"] / "shape.defaults.json"


def _targets(mechanisms: list[dict]):
    """(mechanism, case or None, [(run, gpus, length)]) for every shape file."""
    for m in mechanisms:
        for case in m["cases"]:
            yield m, case, [(r, case["gpus"][0], case["length"]) for r in case["runs"]]
        if m["defaults"]:
            d = m["defaults"]
            yield m, None, [({"ast": d["ast"], "params": None, "abstract": m["asts"][d["ast"]]}, d["gpus"], d["length"])]


def jobs_default() -> int:
    """Dry runs at once by default: one per CPU this process may run on."""
    return len(os.sched_getaffinity(0))


def _recorded(m: dict, case: dict | None, runs: list) -> tuple[list[str], list[tuple]]:
    """A shape file against the records committed now, without the runner: why it is stale
    (missing, of another `shape_version`, other runs, or a run measured on another AST,
    parameter file, GPU count, or length parameter), and the runs it holds that are current,
    `(run, gpus, recorded)` each."""
    path = _shape_path(m, case)
    if not path.exists():
        return ["missing"], []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("shape_version") != SHAPE_VERSION:
        return [f"shape_version {doc.get('shape_version')}, this aeiou-catalog writes {SHAPE_VERSION}"], []
    have = {s["ast"]: s for s in doc["runs"]}
    if set(have) != {run["ast"] for run, *_ in runs}:
        return ["its runs are not the case's"], []
    stale, current = [], []
    for run, gpus, length in runs:
        s = have[run["ast"]]
        if (s["ast_sha256"] != run["abstract"]["sha256"] or (run["params"] and s["params_sha256"] != _sha256(run["params"]))
                or s["gpus"] != gpus or (s["length"] or {}).get("param") != length):
            stale.append(f"measured on another {run['ast']}, parameter file, or case.toml than the one committed")
        else:
            current.append((run, gpus, s))
    return stale, current


def _previous(path: pathlib.Path) -> dict:
    """The entries a shape file holds now, by AST name, whatever made it stale: where each
    search starts and what orders its dry runs. None when it is not there or not a shape."""
    try:
        return {s["ast"]: s for s in json.loads(path.read_text(encoding="utf-8"))["runs"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _finished(finish, begun: list, attrs: dict) -> list:
    """`finish` of every begun state, each on a thread of its own (they only wait on the
    scheduler), in their order."""
    if not begun:
        return []
    with concurrent.futures.ThreadPoolExecutor(len(begun)) as finishing:
        return list(finishing.map(lambda s: finish(s, attrs), begun))


def measure_all(mechanisms: list[dict], attrs: dict, runner: str, jobs: int = 1, every: bool = True) -> dict[pathlib.Path, str]:
    """Every shape file's text: measured now, or, unless `every`, the committed text of a
    shape `_recorded` finds current (one measured on the records committed now; that its
    numbers still hold is `check`'s to say). Every run's dry runs share one scheduler of
    `jobs`, the most expensive case's first; every case's first ones are queued before any
    starts. The texts are the same at any `jobs`: the dry runs are deterministic, and each
    search's next level depends on its own case's numbers alone."""
    texts, units = {}, []
    for m, case, runs in _targets(mechanisms):
        path = _shape_path(m, case)
        if not every and not _recorded(m, case, runs)[0]:
            texts[path] = path.read_text(encoding="utf-8")
            continue
        texts[path] = None
        previous = _previous(path)
        units += [(path, run, gpus, length, previous.get(run["ast"])) for run, gpus, length in runs]
    with Scheduler(jobs) as pool:
        with pool.held():
            begun = [begin_shape(runner, run, gpus, length, pool, previous) for _, run, gpus, length, previous in units]
        shapes = _finished(finish_shape, begun, attrs)
    for path in texts:
        if texts[path] is None:
            texts[path] = render_shape([s for (p, *_), s in zip(units, shapes) if p == path])
    return texts


def stale_shapes(mechanisms: list[dict], root: pathlib.Path, attrs: dict, runner: str | None, jobs: int = 1) -> list[str]:
    """The shape files missing, measured on an AST or parameter file other than the one
    committed now, or (with the runner) whose numbers the dry run no longer gives: the
    measures at the recorded check length, and the op counts, bytes, and datasets'
    projection at the case's. The dry runs share one scheduler of `jobs`, as `measure_all`'s."""
    problems, current = [], []
    for m, case, runs in _targets(mechanisms):
        path = _shape_path(m, case)
        stale, ok = _recorded(m, case, runs)
        problems += [f"{_rel(path, root)}: {why} (aeiou-catalog write)" for why in stale]
        current += [(path, run, gpus, s) for run, gpus, s in ok]
    if not runner:
        return problems
    with Scheduler(jobs) as pool:
        with pool.held():
            begun = [begin_recheck(runner, run, gpus, s, pool) for _, run, gpus, s in current]
        differs = _finished(finish_recheck, begun, attrs)
    problems += [f"{_rel(path, root)}: {run['ast']}: the dry run no longer gives the recorded {', '.join(differ)} (aeiou-catalog write)"
                 for (path, run, _, _), differ in zip(current, differs) if differ]
    return problems


# ---------------------------------------------------------------- the descriptions

def _description_path(m: dict, case: dict | None) -> pathlib.Path:
    return case["dir"] / "DESCRIPTION.md" if case else m["dir"] / "DESCRIPTION.defaults.md"


def _values(run: dict) -> dict:
    """The values a run takes: its parameter file's, or a case's lock's for this run (none for
    a required one)."""
    if not run["params"]:
        return {}
    if run["params"].name != lock.LOCK_FILE:
        return values_of(emit.loads(run["params"].read_text(encoding="utf-8")))
    doc = json.loads(run["params"].read_text(encoding="utf-8"))
    entries = next((r["params"] for r in doc.get("runs", []) if r.get("name") == run["ast"]), {})
    # a storage-reported value, at its fallback, as a dry run without --st-blksize takes it
    return {k: v.get("fallback") if isinstance(v, dict) and "from" in v else v for k, v in values_of({"params": entries}).items()}


def context(m: dict, case: dict | None, run: dict, gpus: int, attrs: dict) -> list[str]:
    """What the records say of a run, in sentences: where the catalog files it, what the case is,
    and what its captures say."""
    a = m["attributes"]
    joined = lambda v: ", ".join(v) if isinstance(v, list) else v
    out = [f"In the catalog it is `{m['rel']}`: application `{a.get('application', '—')}`, format `{a.get('format', '—')}`, "
           f"phase {joined(a.get('phase', '—'))}, domain {joined(a.get('domain', '—'))}"]
    if case is None:
        return out + [f"No case yet: the abstract at its defaults, at {gpus} GPU{'' if gpus == 1 else 's'}"]
    ca = case["attributes"]
    head = attrs["kind"]["values"][ca["kind"]]["meaning"].split(":")[0].split(".")[0]
    family = ca.get("model-family")
    about = ([f"{ca['modality']} data"] if "modality" in ca else []) + (
        [f"{'an' if family[0] in 'aeiou' else 'a'} {family} model"] if family else [])
    gp = " or ".join(map(str, case["gpus"]))
    out.append(f"{head} (`{case['name']}`{': ' + ', '.join(about) if about else ''}), its parameters holding at {gp} "
               f"GPU{'' if case['gpus'] == [1] else 's'}")
    verdicts = [p["verdict"] for cap in case["captures"] for p in cap["pairs"] if p["ast"] == run["ast"]]
    status = derived(run["abstract"], verdicts)["status"]
    meaning = attrs["status"]["values"][status]["meaning"]
    out.append("Not traced yet" if status == "untraced" else f"Traced: {meaning[0].lower()}{meaning[1:]}")
    return out


def describe_run(m: dict | None, case: dict | None, run: dict, gpus: int, attrs: dict | None, shape_run: dict | None) -> tuple[str, list[str]]:
    """One run's paragraph and account."""
    ctx = context(m, case, run, gpus, attrs) if m else []
    return (Describer(run["abstract"]["doc"], _values(run), gpus, shape=shape_run).paragraph(ctx),
            Describer(run["abstract"]["doc"], _values(run), gpus, shape=shape_run).account())


def render_description(title: str, sources: str, described: list[tuple[str, str, list[str]]]) -> str:
    """A description file: the paragraph of each run, then each run's account."""
    out = [f"# {title}", "",
           f"Generated by `aeiou-catalog` from {sources}; never edited by hand (`PROJECT_BRIEF.md` §6 item 27,",
           "`DESIGN_REVIEW.md` §3.76). It says what the abstract does, not why the application does it (the",
           "mechanism's README) or what a storage system should do about it.", "",
           "## In brief", ""]
    many = len(described) > 1
    for ast, para, _ in described:
        out += [(f"**`{ast}`.** " if many else "") + para, ""]
    for ast, _, account in described:
        out += [f"## The account of `{ast}`" if many else "## The account", ""] + account
    while out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def render_descriptions(mechanisms: list[dict], attrs: dict, shapes: dict[pathlib.Path, str]) -> tuple[dict[pathlib.Path, str], list[str]]:
    """Every case's DESCRIPTION.md (DESCRIPTION.defaults.md for a mechanism with no case), and the
    paragraphs CATALOG.md shows."""
    files, briefs = {}, []
    for m, case, runs in _targets(mechanisms):
        sp = _shape_path(m, case)
        doc = json.loads(shapes[sp]) if sp in shapes else {"runs": []}
        by_ast = {s["ast"]: s for s in doc["runs"]}
        described = []
        for run, gpus, *_ in runs:
            para, account = describe_run(m, case, run, gpus, attrs, by_ast.get(run["ast"]))
            described.append((run["ast"], para, account))
        if case:
            title = f"`{case['name']}`, a case of `{m['dir'].name}`"
            sources = "the AST" + ("s" if len(runs) > 1 else "") + ", the case's parameter file" + ("s" if len(runs) > 1 else "") + ", its records, and its measured shape"
        else:
            title = f"`{m['dir'].name}` at its defaults"
            sources = "the AST at its defaults, the mechanism's record, and its measured shape"
        path = _description_path(m, case)
        files[path] = render_description(title, sources, described)
        rel = pathlib.Path(m["rel"]) / path.relative_to(m["dir"])
        name = case["name"] if case else "*defaults*"
        for ast, para, _ in described:
            briefs += [f"**`{m['rel']}`, {name}" + (f" (`{ast}`)" if len(described) > 1 else "") + f"** ([description]({rel})). {para}", ""]
    return files, briefs


# ---------------------------------------------------------------- CATALOG.md

def _size(v) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:g} {unit}" if unit == "B" or v == int(v) else f"{v:.1f} {unit}"
        v /= 1024


def _shown(attr: dict, a: dict) -> str:
    if a["value"] is None:
        return "—"
    v = a["value"]
    unit = attr.get("unit")
    num = _size(v) if unit == "bytes" else "" if unit == "flag" else f"{v:.2f}"
    return f"{a['class']} ({num})" if num else a["class"]


def render_catalog(mechanisms: list[dict], attrs: dict, shapes: dict[pathlib.Path, str], briefs: list[str]) -> str:
    measured = [n for n, a in attrs.items() if a["source"] == "measured"]
    rows, shape_rows = [], []
    for m, case, runs in _targets(mechanisms):
        doc = json.loads(shapes[_shape_path(m, case)]) if _shape_path(m, case) in shapes else {"runs": []}
        by_ast = {s["ast"]: s for s in doc["runs"]}
        for run, gpus, *_ in runs:
            name = (case["name"] if case else "*defaults*") + (f" ({run['ast']})" if case and len(case["runs"]) > 1 else "")
            where = f"`{m['rel']}`"
            verdicts = [p["verdict"] for cap in (case["captures"] if case else []) for p in cap["pairs"] if p["ast"] == run["ast"]]
            d = derived(run["abstract"], verdicts)
            ca = case["attributes"] if case else {}
            gp = ", ".join(map(str, case["gpus"])) if case else str(gpus)
            rows.append(f"| {where} | {name} | {ca.get('kind', '—')} | {ca.get('modality', '—')} | {ca.get('model-family', '—')} | "
                        f"`{run['ast']}` | {d['api']} | {d['writes']} | {d['inputs']} | {d['status']} | {gp} |")
            s = by_ast.get(run["ast"])
            if s is None:
                shape_rows.append(f"| {where} | {name} | not measured |" + " |" * len(measured))
                continue
            ln = "—" if s["length"] is None else f"{s['length']['param']} {s['length']['case']:,}" + (
                "" if s["length"]["check"] == s["length"]["case"] else f" ({s['length']['check']:,})")
            if not s["stable"]:
                ln += f"; seeds differ: {', '.join(s['moving'])}"
            shape_rows.append(f"| {where} | {name} | {ln} | " + " | ".join(_shown(attrs[n], s["attributes"][n]) for n in measured) + " |")
    joined = lambda v: ", ".join(v) if isinstance(v, list) else (v or "—")
    coverage = []
    for cat, d in attrs["category"]["values"].items():
        for cls in classes_of(attrs, cat) or [None]:
            ms = [m for m in mechanisms if m["attributes"].get("category") == cat and m["attributes"].get("class") == cls]
            where = f"`{cat}/{cls}/`" if cls else f"`{cat}/`"
            what = attrs["class"]["values"][cls]["meaning"] if cls else d["meaning"]
            coverage.append(f"| {where} | {len(ms) or '—'} | {sum(len(m['cases']) for m in ms) or '—'} | {what} |")
    mech_rows = [f"| `{m['rel']}` | {m['attributes'].get('application', '—')} | {m['attributes'].get('format', '—')} | "
                 f"{joined(m['attributes'].get('phase'))} | {joined(m['attributes'].get('domain'))} | "
                 f"{', '.join(c['name'] for c in m['cases']) or '*defaults*'} |" for m in mechanisms]
    out = ["# The workload catalog", "",
           "Generated by `aeiou-catalog write` from the records under `workloads/` (`mechanism.toml`, `case.toml`,",
           "`capture.toml`), the shapes measured from each case's dry run (`shape.json`), and the dictionary",
           "(`attributes.toml`); never edited by hand (`DESIGN_REVIEW.md` §3.66, §3.68). A mechanism is a",
           "directory: an authoring script and its ASTs. A case is an AST and a parameter set; a mechanism",
           "with no case yet shows its abstract at the defaults.", "",
           "## Coverage", "",
           "Every category of the dictionary and its classes, with what is in each so far: the tree's",
           "directories, a contribution going in the one whose description fits.", "",
           "| Directory | Mechanisms | Cases | What belongs here |",
           "|---|---|---|---|", *coverage, "",
           "## Mechanisms", "",
           "| Mechanism | Application | Format | Phase | Domain | Cases |",
           "|---|---|---|---|---|---|", *mech_rows, "",
           "## Cases", "",
           "| Mechanism | Case | Kind | Modality | Model family | AST | API | Writes | Inputs | Status | GPUs |",
           "|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
           "## In brief", "",
           "Each case in a paragraph, generated from its AST, parameters, records, and measured shape; its",
           "`DESCRIPTION.md` has the account in full (`DESIGN_REVIEW.md` §3.76).", "", *briefs,
           "## Shape", "",
           "Measured by `aeiou dry-run --metrics-json` of the case at its own parameters and first GPU count,",
           "the median of seeds 1, 2, and 3. Each cell is the class and its number; sizes are byte-weighted",
           "medians. The length column is the case's length parameter and, in parentheses, the smallest length",
           "that still gives these numbers within their tolerances, where `aeiou-catalog check` re-measures",
           "them; where the seeds differ by more than a tolerance, the numbers carry that error and are named.",
           "The numbers are the abstract's, not the trace's, and reuse grows with a run's length on a fixed",
           "dataset: they describe the case at its length.", "",
           "| Mechanism | Case | Length | " + " | ".join(f"`{n}`" for n in measured) + " |",
           "|---|---|---|" + "---|" * len(measured), *shape_rows, "",
           "## Glossary", "",
           "Every attribute and value a record may use (`attributes.toml`). Curated attributes are written in the",
           "records, derived ones read from the AST and the captures, measured ones computed from the dry run.", ""]
    for name, a in attrs.items():
        r = a.get("required")
        req = (", required" if r is True else "".join(f", required when `{k}` is {v}" for k, v in r.items()) if isinstance(r, dict) else "") + \
            (", several values" if a.get("multiple") else "")
        out += [f"### `{name}`", "", f"{a['meaning']} *({a['source']}, {a['level']}{req})*", ""]
        if a["source"] == "measured":
            tol = a.get("tolerance")
            if tol:
                bounds = ([f"±{tol['relative'] * 100:g} %"] if "relative" in tol else []) + ([f"±{tol['absolute']:g}"] if "absolute" in tol else [])
                out += [f"Stable within {' or '.join(bounds)}.", ""]
            for c in a["classes"]:
                out.append(f"- `{c['name']}`: {c['meaning']}")
        else:
            for v, d in a["values"].items():
                search = f" *Also: {', '.join(d['search'])}.*" if d.get("search") else ""
                within = f" *(In `{d['category']}`.)*" if "category" in d else ""
                out.append(f"- `{v}`: {d['meaning']}{within}{search}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- catalog.json, what `aeiou cases` reads

INDEX_VERSION = 1


def render_index(mechanisms: list[dict], attrs: dict, shapes: dict[pathlib.Path, str]) -> str:
    """`catalog.json` (DESIGN_REVIEW.md §3.80): what `aeiou cases` lists and filters, so the
    runner reads one generated file and never the records. The vocabulary, every attribute with
    its values (a measured one's classes), in the dictionary's order; and every case a lock
    names, in the tree's order: its curated attributes (the mechanism's and its own), GPU
    counts, lock, and description, and per run what it does (the AST's doc), what it is (its
    parameter files' docs, in order), and its derived attributes and measured classes. A
    mechanism with no case has no lock to run, and no row."""
    vocabulary = [{"name": name, "source": a["source"], "level": a["level"], "multiple": bool(a.get("multiple")),
                   "values": [c["name"] for c in a["classes"]] if a["source"] == "measured" else list(a["values"])}
                  for name, a in attrs.items()]       # a list: a reader keeps the dictionary's order
    measured = [n for n, a in attrs.items() if a["source"] == "measured"]
    cases = []
    for m, case, runs in _targets(mechanisms):
        if case is None:
            continue
        sp = _shape_path(m, case)
        by_ast = {s["ast"]: s for s in json.loads(shapes[sp])["runs"]} if sp in shapes else {}
        rows = []
        for run, *_ in runs:
            verdicts = [p["verdict"] for cap in case["captures"] for p in cap["pairs"] if p["ast"] == run["ast"]]
            got = derived(run["abstract"], verdicts)
            s = by_ast.get(run["ast"])
            for n in measured if s else ():
                if s["attributes"][n]["class"] is not None:
                    got[n] = s["attributes"][n]["class"]
            docs = [emit.loads(f.read_text(encoding="utf-8")).get("doc") for f in run["fragments"]]
            rows.append({"ast": run["ast"], "does": run["abstract"]["doc"].get("doc", ""), "is": " ".join(d for d in docs if d),
                         "attributes": got})
        rel = lambda p: str(pathlib.Path(case["rel"]) / p)
        cases.append({"name": f"{m['dir'].name}/{case['name']}", "dir": case["rel"], "lock": rel(lock.LOCK_FILE),
                      "description": rel("DESCRIPTION.md"), "gpus": case["gpus"],
                      "attributes": {**m["attributes"], **case["attributes"]}, "runs": rows})
    return json.dumps({"catalog_version": INDEX_VERSION, "tool": "aeiou-catalog",
                       "doc": "Generated by aeiou-catalog write from the records under workloads/; read by aeiou cases. Never edited by hand.",
                       "attributes": vocabulary, "cases": cases}, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- the directories' READMEs

def make_tree(root: pathlib.Path, attrs: dict) -> None:
    """Every category's directory, and its classes', so the tree shows what the catalog covers."""
    for cat in attrs["category"]["values"]:
        (root / cat).mkdir(exist_ok=True)
        for cls in classes_of(attrs, cat):
            (root / cat / cls).mkdir(exist_ok=True)


def render_readmes(root: pathlib.Path, attrs: dict, mechanisms: list[dict]) -> dict[pathlib.Path, str]:
    """The README of every category and class directory: what belongs there, from the dictionary,
    and what is there now."""
    out = {}
    head = "Generated by `aeiou-catalog write` from `workloads/attributes.toml`; never edited by hand."

    def mech_list(cat, cls):
        ms = [m for m in mechanisms if m["attributes"].get("category") == cat and m["attributes"].get("class") == cls]
        if not ms:
            where = f"workloads/{cat}/{cls + '/' if cls else ''}<mechanism>/"
            return [f"None yet. A mechanism goes in `{where}` (`workloads/README.md`, \"Adding one\")."]
        return [f"- [`{m['dir'].name}`]({m['dir'].name}/): {m['attributes'].get('application', '—')}, "
                f"{m['attributes'].get('format', '—')}; cases: {', '.join(c['name'] for c in m['cases']) or 'none yet'}" for m in ms]

    for cat, d in attrs["category"]["values"].items():
        search = [f"Also searched for as: {', '.join(d['search'])}.", ""] if d.get("search") else []
        lines = [f"# `{cat}`", "", head, "", d["meaning"], "", *search]
        classes = classes_of(attrs, cat)
        if classes:
            lines += ["## Classes", ""]
            for cls in classes:
                n = sum(1 for m in mechanisms if m["attributes"].get("category") == cat and m["attributes"].get("class") == cls)
                lines.append(f"- [`{cls}`]({cls}/): {attrs['class']['values'][cls]['meaning']} ({n or 'no'} mechanism{'' if n == 1 else 's'})")
                c = attrs["class"]["values"][cls]
                cs = [f"Also searched for as: {', '.join(c['search'])}.", ""] if c.get("search") else []
                out[root / cat / cls / "README.md"] = "\n".join([f"# `{cat}/{cls}`", "", head, "", c["meaning"], "", *cs,
                                                               "## Mechanisms", "", *mech_list(cat, cls), ""])
        else:
            lines += ["## Mechanisms", "", *mech_list(cat, None)]
        lines += ["", "The catalog of every case: [`CATALOG.md`](../CATALOG.md).", ""]
        out[root / cat / "README.md"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------- the command

def parser() -> tuple[usage.Parser, dict[str, usage.Parser]]:
    ap = usage.Parser(prog="aeiou-catalog", description="The workload catalog (DESIGN_REVIEW.md §3.66): check the records under workloads/ against their dictionary, measure each case's shape, and render CATALOG.md and each case's description",
                      usage="aeiou-catalog <COMMAND> [OPTIONS]")
    ap.version(f"aeiou-catalog {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    subs = {}
    for name, about in (("check", "Validate the records, re-derive the shapes and CATALOG.md, and fail when a committed one differs"),
                        ("write", "Validate the records, measure the shapes, and write them and CATALOG.md")):
        p = sub.add_parser(name, help=about, usage=f"aeiou-catalog {name} [OPTIONS]")
        p.add_argument("--workloads", type=pathlib.Path, default=pathlib.Path("workloads"), metavar="DIR", help="The workload tree [default: workloads]")
        p.add_argument("--runner", metavar="PATH", default="aeiou", help="The runner binary the shapes are measured with [default: aeiou, on the PATH]")
        p.add_argument("--no-shapes", action="store_true", help="Leave the shapes as committed (no dry run); check that each was measured on the AST and parameter file committed now")
        p.add_argument("--jobs", type=int, default=jobs_default(), metavar="N",
                       help="Dry runs at once, over every shape, the most expensive case's first [default: the CPUs this process may run on]")
        if name == "write":
            p.add_argument("--all-shapes", action="store_true",
                           help="Measure every shape, not only those measured on records other than the ones committed now (after a change to the runner's metrics)")
        subs[name] = p
    p = sub.add_parser("describe", help="Print an abstract's description at a parameter set: its paragraph and its account, with its shape measured by dry run",
                       usage="aeiou-catalog describe [OPTIONS] <AST>")
    p.add_argument("ast", nargs="?", metavar="AST", help="The abstract (*.ast.json)")
    p.add_argument("--params-file", type=pathlib.Path, metavar="FILE", help="A parameter file for it (aeiou-params); its defaults when absent")
    p.add_argument("--gpus", type=int, default=1, metavar="N", help="The GPU count [default: 1]")
    p.add_argument("--workloads", type=pathlib.Path, default=pathlib.Path("workloads"), metavar="DIR", help="The workload tree whose dictionary classes the shape [default: workloads]")
    p.add_argument("--runner", metavar="PATH", default="aeiou", help="The runner binary the shape is measured with [default: aeiou, on the PATH]")
    p.add_argument("--no-shapes", action="store_true", help="No dry run: the description without its measured shape, op counts, or dataset projections")
    subs["describe"] = p
    return ap, subs


@usage.tool
def main(argv=None) -> int:
    ap, subs = parser()
    checkout.warn(ap.prog)
    sp = ap
    try:
        a = ap.parse_args(argv)
        missing = usage.Missing(ap)
        missing.want(a.cmd, "<COMMAND>", "one of check, write, describe")
        missing.check()
        sp = subs[a.cmd]
        if a.cmd != "describe" and a.jobs < 1:
            raise usage.UsageError(f"--jobs {a.jobs}: must be at least 1", sp)
        if a.cmd == "describe":
            missing = usage.Missing(sp)
            missing.want(a.ast, "<AST>", "the abstract to describe")
            missing.check()
            if a.gpus < 1:
                raise usage.UsageError(f"--gpus {a.gpus}: must be at least 1", sp)
            return _describe(a)
        return _main(a)
    except usage.UsageError as e:
        return usage.fail(e, sp)
    except BuildError as e:
        return usage.failed(sp.prog, e)


def _describe(a) -> int:
    """`describe`: any AST, at its defaults or a parameter file, outside the catalog's records."""
    path = pathlib.Path(a.ast)
    doc = emit.load(path)           # not there, or not JSON: a usage error, as every tool's
    validate(doc, path.name)        # JSON that is no abstract: one line, as the runner's
    if a.params_file:
        if not a.params_file.is_file():
            raise BuildError(f"{a.params_file}: no such parameter file")
        errors = params.check(doc, params.load(a.params_file), ast_sha256=emit.sha256(doc), name=a.params_file.name)
        if errors:
            raise BuildError("parameter file rejected:\n  " + "\n  ".join(errors))
    run = {"ast": doc.get("name"), "params": a.params_file, "abstract": {"path": path, "doc": doc, "sha256": _sha256(path)}}
    measured = None
    if not a.no_shapes:
        if shutil.which(a.runner) is None:
            raise BuildError(f"the runner {a.runner!r} is not there (--runner PATH; --no-shapes for the description without its shape)")
        if not (a.workloads / "attributes.toml").is_file():
            raise BuildError(f"{a.workloads}: not a workload tree (no attributes.toml), whose dictionary classes the shape (--workloads DIR; --no-shapes)")
        with Scheduler(len(SEEDS)) as pool:
            measured = shape(a.runner, run, a.gpus, None, load_dictionary(a.workloads), pool)
    para, account = describe_run(None, None, run, a.gpus, None, measured)
    sources = "the AST" + (f", `{a.params_file.name}`" if a.params_file else " at its defaults") + (", and its shape by dry run" if measured else "")
    text = render_description(f"`{doc.get('name')}`", sources, [(doc.get("name"), para, account)])
    # a document, the bytes `write` puts in DESCRIPTION.md whatever the locale (`usage.console`
    # is for messages); a stream with no bytes under it (a test's) takes the text
    out = getattr(sys.stdout, "buffer", None)
    if out is None:
        sys.stdout.write(text)
    else:
        sys.stdout.flush()
        out.write(text.encode("utf-8"))
        out.flush()
    return 0


def _main(a) -> int:
    root = a.workloads
    if not (root / "attributes.toml").is_file():
        raise BuildError(f"{root}: not a workload tree (no attributes.toml)")
    if not a.no_shapes and shutil.which(a.runner) is None:
        raise BuildError(f"the runner {a.runner!r} is not there (--runner PATH; --no-shapes to skip the dry runs)")
    attrs = load_dictionary(root)
    if a.cmd == "write":
        make_tree(root, attrs)
    mechanisms, problems = load(root, attrs)
    if problems:
        for p in problems:
            print(p)
        print(f"{len(problems)} problem(s) in the records")
        return 1
    cases = sum(len(m["cases"]) for m in mechanisms)
    measured = None
    if a.cmd == "write" and not a.no_shapes:
        targets = list(_targets(mechanisms))
        measured = sum(1 for m, c, runs in targets if a.all_shapes or _recorded(m, c, runs)[0]), len(targets)
        shapes, stale = measure_all(mechanisms, attrs, a.runner, a.jobs, every=a.all_shapes), []
    else:
        stale = stale_shapes(mechanisms, root, attrs, None if a.no_shapes else a.runner, a.jobs)
        shapes = {_shape_path(m, c): _shape_path(m, c).read_text(encoding="utf-8") for m, c, _ in _targets(mechanisms) if _shape_path(m, c).exists()}
    files = dict(shapes) if a.cmd == "write" else {}
    descriptions, briefs = render_descriptions(mechanisms, attrs, shapes)
    files.update(descriptions)
    files[root / "CATALOG.md"] = render_catalog(mechanisms, attrs, shapes, briefs)
    files[root / "catalog.json"] = render_index(mechanisms, attrs, shapes)
    files.update(render_readmes(root, attrs, mechanisms))
    if a.cmd == "write":
        for path, text in files.items():
            if not path.exists() or path.read_bytes() != text.encode("utf-8"):
                path.write_text(text, encoding="utf-8")
                print(f"wrote {_rel(path, root)}")
        for p in stale:
            print(p)
        print(f"{len(mechanisms)} mechanisms, {cases} cases" + ("" if measured is None else f"; {measured[0]} of {measured[1]} shapes measured")
              + ("" if not stale else f"; {len(stale)} shape(s) stale"))
        return 1 if stale else 0
    drift = [f"{_rel(path, root)}: differs from what aeiou-catalog derives now (aeiou-catalog write)"
             for path, text in files.items() if not path.exists() or path.read_bytes() != text.encode("utf-8")]
    for p in stale + drift:
        print(p)
    print(f"{len(mechanisms)} mechanisms, {cases} cases: " + ("ok" if not stale + drift else f"{len(stale + drift)} problem(s)"))
    return 1 if stale + drift else 0


if __name__ == "__main__":
    sys.exit(main())
