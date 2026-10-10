"""`aeiou-catalog`: the workload catalog's records, their dictionary, and what is derived from them.

`workloads/` holds one directory per I/O mechanism and, inside it, one directory per case (an
AST and a parameter set: a workload a user would ask for) with its captures
(`DESIGN_REVIEW.md` §3.66). Building is a compiler's problem (`aeiou-build`, a script to an
AST); the catalog is a data-management one: curated records, the vocabulary they are written
in, and their consistency with the abstracts they describe.

    aeiou-catalog check [--workloads DIR] [--runner PATH] [--no-shapes]   validate, re-derive, exit 1 on drift
    aeiou-catalog write [--workloads DIR] [--runner PATH] [--no-shapes]   re-derive and write

The records, each a TOML file:

- `<mechanism>/mechanism.toml`: the mechanism's curated attributes (`category`, `access`,
  `application`, `format`), and a `[defaults]` table (`ast`, `gpus`, `length`) when it has no
  case yet, so the catalog still shows it, at the abstract's defaults.
- `cases/<case>/case.toml`: the case's curated attributes (`kind`, `modality`), the GPU counts
  its parameters hold at (`gpus`), its `length` parameter (the one that only makes the run
  longer) when it has one, `metrics_block` (the unit of the locality metrics, 4 KiB unless
  the case says), and one `[[run]]` per AST it runs (`ast`, `params`, a file of the case).
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
such length: `check` re-measures there, so CI never pays for a long case. `CATALOG.md` is
rendered from the records, the shapes, and the dictionary. `check` re-measures every shape and re-renders the catalog, and fails when a
committed one differs; `--no-shapes` skips the dry runs (no runner needed) and checks instead
that each shape was measured on the AST, parameter file, and case record committed now.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

from . import __version__, usage
from .nodes import BuildError

SHAPE_VERSION = 1
SEEDS = (1, 2, 3)
START = 64                      # the shortest check length tried is the case's over this
BLOCK = 4096                    # the dry run's own default unit of the locality metrics
METADATA = ("open", "close", "stat", "fstat", "readdir", "mkdir", "rmdir", "unlink", "rename", "ftruncate", "fallocate")
DATA = ("read", "write")
MECHANISM_KEYS = {"defaults"}
DEFAULTS_KEYS = {"ast", "gpus", "length"}
CASE_KEYS = {"gpus", "length", "metrics_block", "run"}
RUN_KEYS = {"ast", "params"}
CAPTURE_KEYS = {"raw_trace", "pair"}
PAIR_KEYS = {"trace", "ast", "params", "verdict"}
BEST = ("accepted", "not-accepted", "nothing-judged")       # status, best first


# ---------------------------------------------------------------- the dictionary

def load_dictionary(root: pathlib.Path) -> dict:
    path = root / "attributes.toml"
    try:
        d = tomllib.loads(path.read_text())
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
    for lo, w in weights:
        acc += w
        if 2 * acc >= total:
            return lo
    return weights[-1][0]


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
    "reads_own_writes": lambda t: 1.0 if t["reuse_distance_bytes"]["read_after_write"]["n"] else 0.0,
    "reread": lambda t: _share(t["reuse_distance_bytes"]["read_after_read"]["n"], _accesses(t["reuse_distance_bytes"], "read")),
    "skew": lambda t: t["popularity_blocks"]["top_1_pct"] if t["popularity_blocks"]["accesses"] else None,
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
        return tomllib.loads(path.read_text())
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
        elif not isinstance(value, str) or value not in a["values"]:
            problems.add(path, root, f"`{key}` = {value!r}: not one of {', '.join(a['values'])}")
        else:
            out[key] = value
    for key, a in attrs.items():
        if a["source"] == "curated" and a["level"] == level and a.get("required") and key not in rec:
            problems.add(path, root, f"`{key}` is required")
    return out


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params_of(ast: dict) -> dict:
    return ast.get("params") or {}


def _length_ok(ast: dict, name: str) -> bool:
    p = _params_of(ast).get(name)
    return isinstance(p, dict) and isinstance(p.get("default"), int) and not isinstance(p.get("default"), bool)


def load(root: pathlib.Path, attrs: dict) -> tuple[list[dict], Problems]:
    """Every mechanism with its cases and their captures, checked; the problems found."""
    problems = Problems()
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
        cat, acc = m["attributes"].get("category"), m["attributes"].get("access")
        want = (cat, acc, None) if acc else (cat, None)
        if cat and (len(parts) != len(want) or parts[0] != cat or (acc and parts[1] != acc)):
            problems.add(path, root, f"category {cat!r}" + (f" and access {acc!r}" if acc else "") +
                         f" say workloads/{cat}/{acc + '/' if acc else ''}<mechanism>, not workloads/{m['rel']}")
        for a in sorted(d.glob("*.ast.json")):
            try:
                doc = json.loads(a.read_text())
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
            "gpus": rec.get("gpus"), "length": rec.get("length"), "block": rec.get("metrics_block", BLOCK),
            "runs": [], "captures": []}
    if not isinstance(case["gpus"], list) or not case["gpus"] or not all(_gpus_ok(g) for g in case["gpus"]):
        problems.add(path, root, "gpus must be a non-empty list of positive integers: the GPU counts its parameters hold at")
        case["gpus"] = [1]
    if not _gpus_ok(case["block"]):
        problems.add(path, root, "metrics_block must be a positive integer (bytes)")
        case["block"] = BLOCK
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
        params = c / str(r.get("params"))
        if not params.is_file() or params.parent != c:
            problems.add(path, root, f"[[run]] params {r.get('params')!r} is not a file of the case")
            continue
        try:
            pset = json.loads(params.read_text())
        except ValueError as e:
            problems.add(params, root, str(e))
            continue
        if pset.get("abstract") != r["ast"]:
            problems.add(params, root, f"its abstract is {pset.get('abstract')!r}, the case runs it as {r['ast']}")
        if case["length"] is not None and not _length_ok(ast["doc"], case["length"]):
            problems.add(path, root, f"length {case['length']!r} is not an integer parameter of {r['ast']}")
            case["length"] = None
        case["runs"].append({"ast": r["ast"], "params": params, "abstract": ast})
    if len({r["ast"] for r in case["runs"]}) != len(case["runs"]):
        problems.add(path, root, "two runs of one AST: a case runs each abstract once")
    named = {r["params"].name for r in case["runs"]}
    for p in sorted(c.glob("*params.json")):
        if p.name not in named:
            problems.add(p, root, "a parameter file no [[run]] of case.toml names")
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
        params = cap / p["params"] if "params" in p else runs[p["ast"]]["params"]
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

def _dry_run(runner: str, run: dict, gpus: int, block: int, seed: int, length: tuple | None, tmp: pathlib.Path) -> dict:
    out = tmp / f"{run['abstract']['doc']['name']}.{seed}.{length[1] if length else 'case'}.json"
    cmd = [runner, "dry-run", str(run["abstract"]["path"]), "--gpus", str(gpus), "--seed", str(seed),
           "--metrics-block", str(block), "--metrics-json", str(out)]
    if run["params"]:
        cmd += ["--params-file", str(run["params"])]
    if length:
        cmd += ["--param", f"{length[0]}={length[1]}"]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise BuildError(f"{' '.join(cmd)}: exit {p.returncode}\n{p.stderr.strip()}")
    return json.loads(out.read_text())["total"]


def _median(xs: list):
    """The median of the seeds' numbers (the middle one of three); none when a seed has none."""
    if any(x is None for x in xs):
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _at(runner: str, run: dict, gpus: int, block: int, length: tuple | None, pool, tmp) -> tuple[dict, list[dict], int]:
    """The median of each measure over the seeds, each seed's numbers, and the ops of one seed's run."""
    totals = list(pool.map(lambda seed: _dry_run(runner, run, gpus, block, seed, length, tmp), SEEDS))
    values = [measure(t) for t in totals]
    return {n: _median([v[n] for v in values]) for n in MEASURES}, values, totals[0]["ops"]


def _agrees(attrs: dict, a: dict, b: dict) -> list[str]:
    """The measures on which two sets of numbers differ beyond their tolerances."""
    return [n for n in MEASURES if not agree(attrs[n], a[n], b[n])]


def shape(runner: str, run: dict, gpus: int, block: int, length_param: str | None, attrs: dict, pool) -> dict:
    """One run's shape (`shape.json`'s entry): the measures at the case's own parameters, the
    median over the seeds and its spread; and, for a case with a length parameter, the
    smallest length (halving down to 1/64) whose numbers are still the case's, where `check`
    re-measures them. A shorter run is never taken for the case: a reuse share grows with the
    run on a fixed dataset, and a fixed load is amortized only over a long one."""
    pset = json.loads(run["params"].read_text())["params"] if run["params"] else {}
    full = pset.get(length_param, _params_of(run["abstract"]["doc"])[length_param]["default"]) if length_param else None
    with tempfile.TemporaryDirectory(prefix="aeiou-catalog.") as tmp:
        tmp = pathlib.Path(tmp)
        med, seeds, ops = _at(runner, run, gpus, block, (length_param, full) if full else None, pool, tmp)
        check = full
        if full:
            n = full // 2
            while n >= max(1, math.ceil(full / START)):
                if _agrees(attrs, _at(runner, run, gpus, block, (length_param, n), pool, tmp)[0], med):
                    break
                check, n = n, n // 2
    # the numbers the seeds disagree on: the case's own statistical error is larger than the tolerance
    moving = [n for n in MEASURES if not all(agree(attrs[n], v[n], w[n]) for v in seeds for w in seeds)]
    spread = {n: None if med[n] is None else max(v[n] for v in seeds) - min(v[n] for v in seeds) for n in MEASURES}
    out = {"ast": run["abstract"]["doc"]["name"], "ast_sha256": run["abstract"]["sha256"],
           "params": run["params"].name if run["params"] else None, "params_sha256": _sha256(run["params"]) if run["params"] else None,
           "gpus": gpus, "block": block, "seeds": list(SEEDS),
           "length": None if full is None else {"param": length_param, "case": full, "check": check},
           "stable": not moving, "moving": moving, "ops": ops, "attributes": {}}
    for name in MEASURES:
        out["attributes"][name] = {"value": _round(med[name]), "spread": _round(spread[name]), "class": classify(attrs[name], med[name])}
    return out


def recheck(runner: str, run: dict, gpus: int, block: int, recorded: dict, attrs: dict, pool) -> list[str]:
    """`check` with the runner: the run re-measured at its recorded check length, against
    the recorded numbers; the measures that differ."""
    length = recorded["length"]
    with tempfile.TemporaryDirectory(prefix="aeiou-catalog.") as tmp:
        med = _at(runner, run, gpus, block, (length["param"], length["check"]) if length else None, pool, pathlib.Path(tmp))[0]
    return _agrees(attrs, med, {n: a["value"] for n, a in recorded["attributes"].items()})


def _round(v):
    return None if v is None else (v if isinstance(v, int) else float(f"{v:.6g}"))


def render_shape(runs: list[dict]) -> str:
    return json.dumps({"shape_version": SHAPE_VERSION, "tool": "aeiou-catalog", "runs": runs}, indent=2) + "\n"


def _shape_path(m: dict, case: dict | None) -> pathlib.Path:
    return case["dir"] / "shape.json" if case else m["dir"] / "shape.defaults.json"


def _targets(mechanisms: list[dict]):
    """(mechanism, case or None, [(run, gpus, block, length)]) for every shape file."""
    for m in mechanisms:
        for case in m["cases"]:
            yield m, case, [(r, case["gpus"][0], case["block"], case["length"]) for r in case["runs"]]
        if m["defaults"]:
            d = m["defaults"]
            yield m, None, [({"ast": d["ast"], "params": None, "abstract": m["asts"][d["ast"]]}, d["gpus"], BLOCK, d["length"])]


def measure_all(mechanisms: list[dict], attrs: dict, runner: str) -> dict[pathlib.Path, str]:
    """Every shape file's text, measured now."""
    out = {}
    with concurrent.futures.ThreadPoolExecutor(len(SEEDS)) as pool:
        for m, case, runs in _targets(mechanisms):
            out[_shape_path(m, case)] = render_shape([shape(runner, run, gpus, block, length, attrs, pool) for run, gpus, block, length in runs])
    return out


def stale_shapes(mechanisms: list[dict], root: pathlib.Path, attrs: dict, runner: str | None) -> list[str]:
    """The shape files missing, measured on an AST or parameter file other than the one
    committed now, or (with the runner) whose numbers the dry run no longer gives at the
    recorded check length."""
    problems = []
    with concurrent.futures.ThreadPoolExecutor(len(SEEDS)) as pool:
        for m, case, runs in _targets(mechanisms):
            path = _shape_path(m, case)
            if not path.exists():
                problems.append(f"{_rel(path, root)}: missing (aeiou-catalog write)")
                continue
            have = {s["ast"]: s for s in json.loads(path.read_text())["runs"]}
            if set(have) != {run["ast"] for run, *_ in runs}:
                problems.append(f"{_rel(path, root)}: its runs are not the case's (aeiou-catalog write)")
                continue
            for run, gpus, block, length in runs:
                s = have[run["ast"]]
                if (s["ast_sha256"] != run["abstract"]["sha256"] or (run["params"] and s["params_sha256"] != _sha256(run["params"]))
                        or s["gpus"] != gpus or s["block"] != block or (s["length"] or {}).get("param") != length):
                    problems.append(f"{_rel(path, root)}: measured on another {run['ast']}, parameter file, or case.toml than the one committed (aeiou-catalog write)")
                elif runner:
                    differ = recheck(runner, run, gpus, block, s, attrs, pool)
                    if differ:
                        problems.append(f"{_rel(path, root)}: {run['ast']}: the dry run no longer gives the recorded {', '.join(differ)} (aeiou-catalog write)")
    return problems


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


def render_catalog(mechanisms: list[dict], attrs: dict, shapes: dict[pathlib.Path, str]) -> str:
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
            rows.append(f"| {where} | {name} | {ca.get('kind', '—')} | {ca.get('modality', '—')} | {m['attributes'].get('application', '—')} | "
                        f"{m['attributes'].get('format', '—')} | `{run['ast']}` | {d['api']} | {d['writes']} | {d['inputs']} | {d['status']} | {gp} |")
            s = by_ast.get(run["ast"])
            if s is None:
                shape_rows.append(f"| {where} | {name} | not measured |" + " |" * len(measured))
                continue
            ln = "—" if s["length"] is None else f"{s['length']['param']} {s['length']['case']:,}" + (
                "" if s["length"]["check"] == s["length"]["case"] else f" ({s['length']['check']:,})")
            if not s["stable"]:
                ln += f"; seeds differ: {', '.join(s['moving'])}"
            shape_rows.append(f"| {where} | {name} | {ln} | " + " | ".join(_shown(attrs[n], s["attributes"][n]) for n in measured) + " |")
    out = ["# The workload catalog", "",
           "Generated by `aeiou-catalog write` from the records under `workloads/` (`mechanism.toml`, `case.toml`,",
           "`capture.toml`), the shapes measured from each case's dry run (`shape.json`), and the dictionary",
           "(`attributes.toml`); never edited by hand (`DESIGN_REVIEW.md` §3.66). A row is a case: an AST and",
           "a parameter set; a mechanism with no case yet shows its abstract at the defaults.", "",
           "## Cases", "",
           "| Mechanism | Case | Kind | Modality | Application | Format | AST | API | Writes | Inputs | Status | GPUs |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
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
        req = ", required" if a.get("required") else ""
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
                out.append(f"- `{v}`: {d['meaning']}{search}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- the command

def parser() -> tuple[usage.Parser, dict[str, usage.Parser]]:
    ap = usage.Parser(prog="aeiou-catalog", description="The workload catalog (DESIGN_REVIEW.md §3.66): check the records under workloads/ against their dictionary, measure each case's shape, and render CATALOG.md",
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
        subs[name] = p
    return ap, subs


def main(argv=None) -> int:
    ap, subs = parser()
    sp = ap
    try:
        a = ap.parse_args(argv)
        missing = usage.Missing(ap)
        missing.want(a.cmd, "<COMMAND>", "one of check, write")
        missing.check()
        sp = subs[a.cmd]
        return _main(a)
    except usage.UsageError as e:
        return usage.fail(e, sp)
    except BuildError as e:
        return usage.failed(sp.prog, e)


def _main(a) -> int:
    root = a.workloads
    if not (root / "attributes.toml").is_file():
        raise BuildError(f"{root}: not a workload tree (no attributes.toml)")
    if not a.no_shapes and shutil.which(a.runner) is None:
        raise BuildError(f"the runner {a.runner!r} is not there (--runner PATH; --no-shapes to skip the dry runs)")
    attrs = load_dictionary(root)
    mechanisms, problems = load(root, attrs)
    if problems:
        for p in problems:
            print(p)
        print(f"{len(problems)} problem(s) in the records")
        return 1
    cases = sum(len(m["cases"]) for m in mechanisms)
    if a.cmd == "write" and not a.no_shapes:
        shapes, stale = measure_all(mechanisms, attrs, a.runner), []
    else:
        stale = stale_shapes(mechanisms, root, attrs, None if a.no_shapes else a.runner)
        shapes = {_shape_path(m, c): _shape_path(m, c).read_text() for m, c, _ in _targets(mechanisms) if _shape_path(m, c).exists()}
    files = dict(shapes) if a.cmd == "write" else {}
    files[root / "CATALOG.md"] = render_catalog(mechanisms, attrs, shapes)
    if a.cmd == "write":
        for path, text in files.items():
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
                print(f"wrote {_rel(path, root)}")
        for p in stale:
            print(p)
        print(f"{len(mechanisms)} mechanisms, {cases} cases" + ("" if not stale else f"; {len(stale)} shape(s) stale"))
        return 1 if stale else 0
    drift = [f"{_rel(path, root)}: differs from what aeiou-catalog derives now (aeiou-catalog write)"
             for path, text in files.items() if not path.exists() or path.read_text() != text]
    for p in stale + drift:
        print(p)
    print(f"{len(mechanisms)} mechanisms, {cases} cases: " + ("ok" if not stale + drift else f"{len(stale + drift)} problem(s)"))
    return 1 if stale + drift else 0


if __name__ == "__main__":
    sys.exit(main())
