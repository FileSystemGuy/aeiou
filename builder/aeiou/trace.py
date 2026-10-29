"""`aeiou-trace compare` and `aeiou-trace digest`, the two subcommands of the trace-side tool still in
Python (`DESIGN_REVIEW.md` §3.89). The tool is the runner's binary `aeiou-trace` (`runner/aeiou/src/
bin/aeiou_trace.rs`): `metrics` and `export` are Rust since 2026-10-29 (`strace.rs`, `export.rs`,
`objectlog.rs`), and the binary runs these two through `python3 -m aeiou.trace` (or `$AEIOU_PYTHON`)
until their stages of §3.89 port them. The digest reads a trace's calls as the exported lines, which
`export_of` asks the binary for (`aeiou-trace export`), whatever the trace: an strace, an object-store
request log, or an exported trace file already.

`compare` holds two metrics documents (`aeiou_metrics: 2`, from `aeiou-trace metrics` or `aeiou dry-run
--metrics-json`; version 1, a capture's from before 2026-10-22) row by row, and `--judge` holds each row
to the tolerance of its class (`TOLERANCES` below, `builder/REFERENCE.md` §7), raised by what the
abstract differs from itself by at other seeds.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

from . import __version__, checkout, usage
from .nodes import BuildError

FORMAT = 2
FORMATS_READ = (1, 2)    # version 1: a capture taken before 2026-10-22, its locality in blocks only (DESIGN_REVIEW.md §3.79)
TRACE_FORMAT = 1
KINDS = ("read", "write")
R, W = 0, 1

# ---------------------------------------------------------------- histograms


class LogHist:
    """Quarter-octave buckets, as `metrics::LogHist`: a bucket's lower bound keeps the
    value's top three bits."""

    __slots__ = ("n", "sum", "max", "buckets")

    def __init__(self):
        self.n = 0
        self.sum = 0
        self.max = 0
        self.buckets: dict[int, int] = {}

    @staticmethod
    def lower(v: int) -> int:
        if v < 4:
            return v
        e = v.bit_length() - 1
        return (v >> (e - 2)) << (e - 2)

    def add(self, v: int, weight: int = 1):
        self.n += weight
        self.sum += v * weight
        if v > self.max:
            self.max = v
        b = self.lower(v)
        self.buckets[b] = self.buckets.get(b, 0) + weight

    def out(self) -> dict:
        return {
            "n": self.n,
            "mean": self.sum / self.n if self.n else 0.0,
            "max": self.max,
            "buckets": [[b, w] for b, w in sorted(self.buckets.items())],
        }


# ---------------------------------------------------------------- the binary


def binary() -> str:
    """The runner's `aeiou-trace` (`DESIGN_REVIEW.md` §3.89): `$AEIOU_TRACE`, else the one beside
    `$AEIOU_RUNNER`, else the one on the PATH, else the checkout's release build."""
    if os.environ.get("AEIOU_TRACE"):
        return os.environ["AEIOU_TRACE"]
    if os.environ.get("AEIOU_RUNNER"):
        beside = pathlib.Path(os.environ["AEIOU_RUNNER"]).with_name("aeiou-trace")
        if beside.exists():
            return str(beside)
    found = shutil.which("aeiou-trace")
    if found:
        return found
    built = pathlib.Path(__file__).resolve().parents[2] / "runner" / "target" / "release" / "aeiou-trace"
    if built.exists():
        return str(built)
    raise BuildError("the runner's aeiou-trace binary is not found: put it on the PATH, or set AEIOU_TRACE to it (DESIGN_REVIEW.md §3.89)")


def export_of(lines, root: str, exclude=(), cwd: str | None = None) -> tuple[dict, list[dict]]:
    """The exported header and lines of a trace given as lines, an strace or an object-store
    request log, from `aeiou-trace export` under `root`; its refusal as a `BuildError` with the
    binary's own words."""
    with tempfile.TemporaryDirectory(prefix="aeiou-trace-") as d:
        src, out = os.path.join(d, "trace"), os.path.join(d, "trace.jsonl")
        with open(src, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line if line.endswith("\n") else line + "\n")
        args = [binary(), "export", src, "--root", root, "-o", out]
        for g in exclude:
            args += ["--exclude", g]
        if cwd:
            args += ["--cwd", cwd]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            msg = r.stderr.strip().removeprefix("aeiou-trace export: ").removeprefix(f"{src}: ")
            raise BuildError(msg.removesuffix(f"; {out} not written"))
        with open(out, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
    return docs[0], docs[1:]


# ---------------------------------------------------------------- compare


def _cdf_distance(a: dict, b: dict) -> float | None:
    """The largest difference between two histograms' cumulative shares over the bucket
    bounds of both (the Kolmogorov-Smirnov statistic on the bucketed values)."""
    if not a["n"] or not b["n"]:
        return None if a["n"] == b["n"] else 1.0
    da, db = dict(map(tuple, a["buckets"])), dict(map(tuple, b["buckets"]))
    ca = cb = 0
    worst = 0.0
    for bound in sorted(set(da) | set(db)):
        ca += da.get(bound, 0)
        cb += db.get(bound, 0)
        worst = max(worst, abs(ca / a["n"] - cb / b["n"]))
    return worst


def _quantile(h: dict, p: float) -> int:
    want = max(-(-h["n"] * p // 1), 1)
    seen = 0
    for bound, w in h["buckets"]:
        seen += w
        if seen >= want:
            return bound
    return 0


def _tv_distance(a: dict, b: dict) -> float | None:
    """Total-variation distance between two exact distributions (fan-out, depth)."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None if na == nb else 1.0
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def _size(v: float) -> str:
    v = float(v)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:.0f} {unit}" if unit == "B" or v == int(v) else f"{v:.1f} {unit}"
        v /= 1024
    return str(v)


def select(doc: dict, template: str | None) -> dict:
    if template is None:
        return doc["total"]
    try:
        return doc["templates"][template]
    except KeyError:
        raise BuildError(f"no template `{template}` in the document (it has: {', '.join(doc.get('templates', {})) or 'none'})")


def compare(a: dict, b: dict) -> list[tuple[str, str, str, float | None]]:
    """Rows `(metric, a, b, distance)` for two `total`s. A distance is in [0, 1]: a share
    difference, a CDF distance for a histogram, a total-variation distance for an exact
    distribution; `None` when both sides are empty."""
    rows = []

    def share(x, y):
        return x / y if y else 0.0

    def pct(name, va, vb):
        rows.append((name, f"{100 * va:.1f} %", f"{100 * vb:.1f} %", abs(va - vb)))

    def data(t):
        return t["counts"].get("read", 0) + t["counts"].get("write", 0)

    pct("mix: data ops / all ops", share(data(a), a["ops"]), share(data(b), b["ops"]))
    pct("mix: reads / data ops", share(a["counts"].get("read", 0), data(a)), share(b["counts"].get("read", 0), data(b)))
    pct("mix: read bytes / data bytes", share(a["bytes_read"], a["bytes_read"] + a["bytes_written"]), share(b["bytes_read"], b["bytes_read"] + b["bytes_written"]))
    for kind in sorted(set(a["counts"]) | set(b["counts"])):
        pct(f"ops: {kind} / all ops", share(a["counts"].get(kind, 0), a["ops"]), share(b["counts"].get(kind, 0), b["ops"]))

    def hist(name, ha, hb, fmt=_size):
        if not ha["n"] and not hb["n"]:
            return
        show = lambda h: "none" if not h["n"] else " / ".join(fmt(_quantile(h, p)) for p in (0.5, 0.9, 0.99))
        rows.append((f"{name} (p50 / p90 / p99)", show(ha), show(hb), _cdf_distance(ha, hb)))

    plain = lambda v: str(v)
    for k in KINDS:
        hist(f"request size, {k}", a["request_size"][k], b["request_size"][k])
    for k, total in (("read", "bytes_read"), ("write", "bytes_written")):
        hist(f"run length, {k}, bytes", a["run_length"][k]["bytes"], b["run_length"][k]["bytes"])
        hist(f"run length, {k}, ops", a["run_length"][k]["ops"], b["run_length"][k]["ops"], plain)
        if a[total] or b[total]:
            pct(f"run length, {k}: bytes in multi-op runs", share(a["run_length"][k]["multi_op_bytes"], a[total]), share(b["run_length"][k]["multi_op_bytes"], b[total]))
    # the locality sections both have (DESIGN_REVIEW.md §3.79): extents, and blocks, which a
    # capture taken before 2026-10-22 has alone
    sections = [(key, prefix, unit) for key, prefix, unit in (("reuse_extents", "extent reuse", None), ("reuse_distance_bytes", "reuse", "block"))
                if key in a and key in b]
    if not sections:
        raise BuildError("no locality metrics in common: one document has extents only and the other blocks only "
                         "(take the dry run with --metrics-block, the capture's block size)")
    for key, prefix, unit in sections:
        ra, rb = a[key], b[key]
        for this in KINDS:
            per = f"block {this}s" if unit else f"bytes {'read' if this == 'read' else 'written'}"
            def accesses(r):
                return r["first_touch"][this] + sum(r[f"{this}_after_{prev}"]["n"] for prev in KINDS)
            if accesses(ra) or accesses(rb):
                pct(f"{prefix}: first touches / {per}", share(ra["first_touch"][this], accesses(ra)), share(rb["first_touch"][this], accesses(rb)))
            for prev in KINDS:
                k = f"{this}_after_{prev}"
                if ra[k]["n"] or rb[k]["n"]:
                    pct(f"{prefix}: {this} after {prev} / {per}", share(ra[k]["n"], accesses(ra)), share(rb[k]["n"], accesses(rb)))
                hist(f"{prefix} distance, {this} after {prev}", ra[k], rb[k])
        if not unit and (ra["whole"] + ra["partial"] or rb["whole"] + rb["partial"]):
            pct(f"{prefix}: partial re-reads / bytes read again", share(ra["partial"], ra["whole"] + ra["partial"]), share(rb["partial"], rb["whole"] + rb["partial"]))
    for unit in [u for u in ("bytes", "blocks") if f"popularity_{u}" in a and f"popularity_{u}" in b] + ["objects"]:
        pa, pb = a[f"popularity_{unit}"], b[f"popularity_{unit}"]
        for top, label in (("top_0_1_pct", "0.1 %"), ("top_1_pct", "1 %"), ("top_10_pct", "10 %")):
            pct(f"popularity, {unit}: accesses to the top {label}", pa[top], pb[top])
    for key in ("fan_out", "depth"):
        da = {int(k): v for k, v in a[key].items()}
        db = {int(k): v for k, v in b[key].items()}
        if da or db:
            show = lambda d: "none" if not d else " ".join(f"{k}:{100 * v / sum(d.values()):.0f}%" for k, v in sorted(d.items()))
            rows.append((key.replace("_", "-"), show(da), show(db), _tv_distance(da, db)))
    return rows


# ---------------------------------------------------------------- tolerances

TOLERANCES_FORMAT = 1
# The largest distance at which a row of `compare` still counts as a match, by the class
# of the row (the longest of these prefixes of its name). The repository's defaults; a
# tolerance file may replace any of them. `DESIGN_REVIEW.md` §3.55 has where they come from.
TOLERANCES = {
    "mix": 0.05,
    "ops": 0.05,
    "request size": 0.05,
    "run length": 0.10,
    "reuse": 0.10,
    "reuse distance": 0.10,
    "extent reuse": 0.10,
    "extent reuse distance": 0.10,
    "popularity": 0.05,
    "fan-out": 0.10,
    "depth": 0.10,
}
# A popularity row is the share of accesses that go to the top fraction of the units; it
# says something only when that fraction is at least this many units on both sides.
POPULARITY_MIN_UNITS = 10
_TOP = {"0.1 %": 0.001, "1 %": 0.01, "10 %": 0.1}


def tolerance_class(metric: str) -> str:
    return max((c for c in TOLERANCES if metric.startswith(c)), key=len)


def _cause(e: BaseException) -> str:
    """An OSError as the runner prints one (`No such file or directory (os error 2)`)."""
    if isinstance(e, OSError) and e.strerror:
        return f"{e.strerror} (os error {e.errno})"
    return str(e)


def load_tolerances(path: str) -> dict:
    """A tolerance file: `{"aeiou_tolerances": 1, "tolerances": {class: x},
    "unseen": [{"metric": prefix, "reason": text}], "outside": [the same]}`. `unseen` names
    rows the trace cannot show (they are not judged); `outside` records rows known to be
    outside, with the reason (they are judged, and still outside)."""
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise BuildError(f"{path}: {_cause(e)}")
    if not isinstance(spec, dict) or spec.get("aeiou_tolerances") != TOLERANCES_FORMAT:
        raise BuildError(f"{path}: not an aeiou_tolerances: {TOLERANCES_FORMAT} document")
    extra = set(spec) - {"aeiou_tolerances", "tolerances", "unseen", "outside", "comment"}
    if extra:
        raise BuildError(f"{path}: unknown key(s) {', '.join(sorted(extra))}")
    for c, x in spec.get("tolerances", {}).items():
        if c not in TOLERANCES or not isinstance(x, (int, float)) or isinstance(x, bool) or not 0 <= x <= 1:
            raise BuildError(f"{path}: tolerances: `{c}` must be one of {', '.join(TOLERANCES)} with a value in [0, 1]")
    for key in ("unseen", "outside"):
        for e in spec.get(key, []):
            if not isinstance(e, dict) or set(e) != {"metric", "reason"} or not all(isinstance(v, str) and v for v in e.values()):
                raise BuildError(f"{path}: {key}: each entry is {{\"metric\": prefix, \"reason\": text}}")
    return spec


def judge(rows, a: dict, b: dict, docs=(), selfs=(), spec: dict | None = None):
    """Rows `(metric, a, b, distance, allowed, verdict, note)` for the rows of `compare(a, b)`.
    `allowed` is the class tolerance plus the largest distance of that row between `b` and
    each of `selfs` (the same abstract at other seeds: what the abstract differs from itself
    by). Verdicts: `ok`, `outside`, `unseen` (the tolerance file says the trace cannot show
    it), `not judged` (nothing on either side, too few units for a popularity share, or a
    depth without `--chain-gap-us`). Also returns the entries of the file that matched no
    row, or recorded as outside a row that is not."""
    spec = spec or {}
    tol = {**TOLERANCES, **spec.get("tolerances", {})}
    spread: dict[str, float] = {}
    for s in selfs:
        for name, _, _, dist in compare(b, s):
            spread[name] = max(spread.get(name, 0.0), dist or 0.0)
    no_depth = any(d.get("source") == "strace" and d.get("chain_gap_us") is None for d in docs)
    used = set()

    def listed(key, name):
        for i, e in enumerate(spec.get(key, [])):
            if name.startswith(e["metric"]):
                used.add((key, i))
                return e["reason"]
        return None

    out = []
    for name, va, vb, dist in rows:
        allowed = tol[tolerance_class(name)] + spread.get(name, 0.0)
        why = listed("unseen", name)
        if why is not None:
            out.append((name, va, vb, dist, None, "unseen", why))
            continue
        if dist is None:
            out.append((name, va, vb, dist, None, "not judged", "nothing on either side"))
            continue
        if name.startswith("popularity"):
            unit, top = name.split(":")[0].split(", ")[1], name.rsplit("top ", 1)[1]
            fewest = min(a[f"popularity_{unit}"]["distinct"], b[f"popularity_{unit}"]["distinct"])
            if fewest * _TOP[top] < POPULARITY_MIN_UNITS:
                out.append((name, va, vb, dist, None, "not judged", f"the top {top} of {fewest} {unit} is fewer than {POPULARITY_MIN_UNITS}"))
                continue
        if name == "depth" and no_depth:
            out.append((name, va, vb, dist, None, "not judged", "the trace was read without --chain-gap-us"))
            continue
        why = listed("outside", name)
        if dist <= allowed:
            out.append((name, va, vb, dist, allowed, "ok", "recorded as outside, now within" if why is not None else ""))
        else:
            out.append((name, va, vb, dist, allowed, "outside", why or ""))
    stale = [f"{key}: `{e['metric']}` matches no row" for key in ("unseen", "outside") for i, e in enumerate(spec.get(key, [])) if (key, i) not in used]
    stale += [f"outside: `{r[0]}` is recorded as outside and is within" for r in out if r[5] == "ok" and r[6]]
    return out, stale


# ---------------------------------------------------------------- CLI


def _describe(doc: dict) -> str:
    src = doc.get("source", "dry-run")
    if src == "strace":
        return f"strace of {', '.join(doc.get('roots', []))} (order: completion; {doc.get('instances')} instance(s), {doc.get('contexts')} thread(s))"
    if src == "trace":
        return f"trace file {doc.get('trace')} of {', '.join(doc.get('roots', []))} (order: completion; {doc.get('instances')} instance(s), {doc.get('contexts')} lane(s))"
    return f"{src} of {doc.get('abstract')} seed {doc.get('seed')} gpus {doc.get('gpus')} (order: {doc.get('order')})"


def parser() -> tuple[usage.Parser, dict[str, usage.Parser]]:
    """The parser and its subcommands' parsers by name (the usage a `Missing` list shows)."""
    ap = usage.Parser(prog="aeiou-trace", description="The comparison of two metrics documents and the structural digest a fit is made from: the subcommands of aeiou-trace still in Python, which the runner's aeiou-trace binary runs (DESIGN_REVIEW.md §3.89)",
                      usage="aeiou-trace <COMMAND> [ARGS]...")
    sub = ap.add_subparsers(dest="cmd")
    cp = sub.add_parser("compare", help="Compare two metrics documents, from a trace or from aeiou dry-run --metrics-json", usage="aeiou-trace compare [OPTIONS] <A> <B>")
    cp.add_argument("a", nargs="?", help="A metrics document (aeiou_metrics: 2, or 1 from before 2026-10-22), the trace's")
    cp.add_argument("b", nargs="?", help="The metrics document A is compared with, the abstract's")
    cp.add_argument("--template-a", metavar="NAME", help="Use this actor template's metrics of A instead of its total")
    cp.add_argument("--template-b", metavar="NAME", help="Use this actor template's metrics of B instead of its total")
    cp.add_argument("--max-distance", type=usage.non_negative, metavar="X", help="Exit 1 if any distance exceeds X (one number for every row; --judge is the rule by class)")
    cp.add_argument("--judge", action="store_true", help="Judge each row against the tolerance of its class (A the trace, B the abstract) and exit 1 if any is outside")
    cp.add_argument("--tolerances", metavar="FILE", help="A tolerance file (aeiou_tolerances: 1): class tolerances that replace the defaults, the rows the trace cannot show, the rows recorded as outside; implies --judge")
    cp.add_argument("--self", dest="selfs", action="append", default=[], metavar="FILE", help="A metrics document of B's abstract at the same parameters and another seed (repeatable): a row's tolerance is raised by its largest distance between B and these; implies --judge")
    cp.add_argument("--only", action="append", default=[], metavar="PREFIX", help="Only the rows whose metric starts with PREFIX (repeatable)")
    dp = sub.add_parser("digest", help="Write the structural digest of a trace: path templates, lanes and their phases, a verdict per site on what the trace decides (permutation or draw, order, size), and the open questions ranked by their divergence at scale", usage="aeiou-trace digest [OPTIONS] --root <DIR> <TRACE>")
    dp.add_argument("trace", nargs="?", help="The strace output file, an exported trace file, or an object-store request log (MinIO audit, S3 server access); a file, never stdin: the comparison after the fit and a re-run with answers read it again")
    dp.add_argument("--root", metavar="DIR", help="Digest only calls on paths under DIR, written relative to it (required for an strace, not for an exported trace file)")
    dp.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    dp.add_argument("--cwd", metavar="DIR", help="Resolve relative paths against DIR when the trace does not say (no -y)")
    dp.add_argument("--epochs", type=int, default=1, metavar="N", help="Project the open questions over N passes over the set (default 1): the share a draw with replacement never reads is e^-N")
    dp.add_argument("--candidates", type=int, metavar="N", help="The files the application could have chosen among (the corpus); default: the distinct paths the trace touched. Decides a pass from a draw when no path repeats")
    dp.add_argument("--paths", action="store_true", help="Include every raw path under each template (the digest otherwise carries templates and counts only)")
    dp.add_argument("--text", action="store_true", help="Print the text report instead of the JSON document (with -o, the report is printed beside the file anyway)")
    dp.add_argument("-o", "--out", metavar="FILE", help="Write the JSON document (aeiou_digest: 1) to FILE and print the text report")
    dp.add_argument("--answers", metavar="FILE", help="The answers file (aeiou_answers: 1): replayed when it exists, so the questions are not asked; written there after an interactive run. Default after an interactive run with -o: the document's name with .answers.json")
    dp.add_argument("--no-questions", action="store_true", help="Ask nothing even on a terminal: every undecided verdict stays open")
    dp.add_argument("--answered-by", metavar="NAME", help="Record NAME as the one who answered (default: user@host)")
    dp.add_argument("--applications", metavar="DIR", help="The application records (default: the ones shipped with the tool, or the checkout's applications/)")
    dp.add_argument("--application", metavar="NAME[@VERSION]", help="State the traced application instead of being asked: its record's defaults and assertions apply, with the record as their provenance")
    ap.version(f"aeiou-trace {__version__}")
    return ap, {"compare": cp, "digest": dp}


@usage.tool
def main(argv=None) -> int:
    ap, subs = parser()
    checkout.warn(ap.prog)
    sp = ap
    try:
        args = ap.parse_args(argv)
        missing = usage.Missing(ap)
        missing.want(args.cmd, "<COMMAND>", "one of compare, digest (metrics and export are the binary's)")
        missing.check()
        sp = subs[args.cmd]
        missing = usage.Missing(sp)
        if args.cmd == "digest":
            missing.want(args.trace, "<TRACE>", "the strace output file, or an exported trace file")
        if args.cmd == "compare":
            missing.want(args.a, "<A>", "a metrics document (aeiou_metrics: 2, or 1 from before 2026-10-22), the trace's")
            missing.want(args.b, "<B>", "the metrics document A is compared with, the abstract's")
        missing.check()
        if args.cmd == "compare" and args.max_distance is not None and (args.judge or args.tolerances or args.selfs):
            rule = "--judge" if args.judge else "--tolerances (which implies --judge)" if args.tolerances else "--self (which implies --judge)"
            raise usage.UsageError(f"--max-distance cannot be used with {rule}: they are two rules; give one", sp)
        return _main(args, subs)
    except usage.UsageError as e:
        return usage.fail(e, sp)
    except BuildError as e:
        return usage.failed(sp.prog, e)


def _main(args, subs: dict[str, usage.Parser]) -> int:
    if args.cmd == "digest":
        return _digest(args, subs["digest"])

    docs = []
    for path in (args.a, args.b):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise BuildError(f"{path}: {_cause(e)}")
        if not isinstance(d, dict) or d.get("aeiou_metrics") not in FORMATS_READ:
            raise BuildError(f"{path}: not an aeiou_metrics: {FORMAT} document (or version 1, a capture's from before 2026-10-22)")
        docs.append(d)
    a, b = select(docs[0], args.template_a), select(docs[1], args.template_b)
    # where only the blocks are in common, they are in common only at one size
    if not ("reuse_extents" in a and "reuse_extents" in b) and docs[0].get("block") != docs[1].get("block"):
        raise BuildError(f"block sizes differ ({docs[0].get('block')} and {docs[1].get('block')}): reuse distance and block popularity are not comparable")
    rows = compare(a, b)
    if args.only:
        rows = [r for r in rows if any(r[0].startswith(p) for p in args.only)]
    print(f"A: {args.a}: {_describe(docs[0])}")
    print(f"B: {args.b}: {_describe(docs[1])}")
    if args.judge or args.tolerances or args.selfs:
        return _judged(args, rows, a, b, docs)
    w = [max(len(r[i]) for r in rows + [("metric", "A", "B", 0)]) for i in range(3)]
    print(f"{'metric':<{w[0]}}  {'A':>{w[1]}}  {'B':>{w[2]}}  distance")
    worst = 0.0
    for name, va, vb, dist in rows:
        worst = max(worst, dist or 0.0)
        flag = "  <" if args.max_distance is not None and dist is not None and dist > args.max_distance else ""
        print(f"{name:<{w[0]}}  {va:>{w[1]}}  {vb:>{w[2]}}  {'-' if dist is None else f'{dist:.3f}'}{flag}")
    print(f"largest distance {worst:.3f}")
    if args.max_distance is not None and worst > args.max_distance:
        print(f"aeiou-trace compare: exceeds --max-distance {args.max_distance}", file=sys.stderr)
        return 1
    return 0


def _digest(args, dp: usage.Parser) -> int:
    import pathlib
    from . import applications as ap
    from . import digest as dg
    if args.epochs < 1:
        dp.error("--epochs must be at least 1")
    if args.candidates is not None and args.candidates < 1:
        dp.error("--candidates must be at least 1")
    if args.trace == "-":
        dp.error("the digest reads the trace as a file, never stdin: the comparison after the fit and a re-run with answers read it again")
    f = _open_trace(args.trace)
    with f:
        first = f.readline()
        lines = itertools.chain([first], f)
        if not first.startswith('{"aeiou_trace"') and not args.root:
            missing = usage.Missing(dp)
            missing.need(False, "--root <DIR>", "digest only calls on paths under DIR", "needed for an strace; an exported trace file carries its root")
            missing.check()
        try:
            an = dg.analyze(lines, args.root, exclude=args.exclude, cwd=args.cwd)
        except BuildError as e:
            raise BuildError(f"{args.trace}: {e}") from None
    sha = _sha256_of(args.trace)
    by = args.answered_by or _who()
    at = _now()
    kw = dict(epochs=args.epochs, candidates=args.candidates, paths=args.paths)
    said = []
    apps = ap.load(pathlib.Path(args.applications) if args.applications else ap.default_dir())
    stated = _stated_application(apps, args.application, dp) if args.application else None
    if args.answers and os.path.exists(args.answers):
        header, answers = dg.read_answers(args.answers)
        by, at = header.get("by") or by, header.get("at") or at
        doc = dg.document(an, answers=answers, by=by, at=at, application=header.get("application"), **kw)
        if header.get("trace_sha256") != sha:
            doc["notes"]["answers_trace"] = f"{args.answers} was answered on another trace ({header.get('trace')}); its answers were checked against this one"
    elif args.no_questions or not sys.stdin.isatty():
        if args.answers:
            raise BuildError(f"{args.answers}: no such answers file, and no terminal to ask on")
        answers = []
        if stated:
            record, exact = ap.resolve(stated["app"], stated["version"])
            if record is not None:
                stated["record"], stated["exact"] = record.version, exact
                stated["options"] = {o["name"]: o["default"] for o in record.options if "default" in o}
                stated["blind_spots"] = ap.blind_spots(record, stated["options"])
                if exact:
                    answers = ap.assertions(record, stated["options"], [(t.name(), sorted(t.paths)) for t in an.templates])
            del stated["app"]
        doc = dg.document(an, answers=answers, by=by, at=at, application=stated, **kw)
    else:
        preset = None
        if stated:
            preset = {k: v for k, v in stated.items() if k != "app"}
        doc, answers = dg.interview(an, dg.Console(), by=by, at=at, apps=apps, application=preset, **kw)
        if answers or doc.get("application"):
            path = args.answers or (_answers_beside(args.out) if args.out else None)
            if path:
                try:
                    dg.write_answers(path, answers, trace=args.trace, trace_sha256=sha, by=by, at=at, application=doc.get("application"))
                except OSError as e:
                    raise BuildError(f"{path}: {_cause(e)}")
                replay = ["aeiou-trace", "digest", args.trace] + (["--root", args.root] if args.root else []) + ["--answers", path] + (["-o", args.out] if args.out else [])
                said.append(f"answers: {path}")
                said.append("replay without questions: " + " ".join(replay))
            else:
                said.append("answers: in the document only (give -o or --answers FILE to keep them as a file)")
    doc["trace"] = args.trace
    doc["trace_sha256"] = sha
    report = dg.render(doc)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as out:
                out.write(json.dumps(doc, indent=2) + "\n")
        except OSError as e:
            raise BuildError(f"{args.out}: {_cause(e)}")
        sys.stdout.write(report)
        print(f"written: {args.out}")
        for k, v in doc["notes"].items():
            print(f"note: {k}: {v}")
    elif args.text:
        sys.stdout.write(report)
    else:
        sys.stdout.write(json.dumps(doc, indent=2) + "\n")
    for line in said:
        print(line, file=sys.stderr)
    return 0


def _stated_application(apps: dict, stated: str, dp: usage.Parser) -> dict:
    """`--application NAME[@VERSION]` checked against the records."""
    name, _, version = stated.partition("@")
    if name not in apps:
        dp.error(f"--application {name}: no such application record; one of {', '.join(sorted(apps))}")
    app = apps[name]
    return {"app": app, "name": app.name, "display": app.display, "version": version or None, "record": None, "exact": False, "options": {}, "blind_spots": []}


def _answers_beside(out: str) -> str:
    """The answers file an interactive run leaves beside the document it wrote."""
    return out[: -len(".digest.json")] + ".answers.json" if out.endswith(".digest.json") else out + ".answers.json"


def _sha256_of(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _who() -> str:
    import getpass
    import socket
    try:
        user = getpass.getuser()
    except (KeyError, OSError):   # no passwd entry for the uid (a container)
        user = str(os.getuid())
    return f"{user}@{socket.gethostname()}"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _open_trace(path: str):
    """The trace to read, `-` for stdin: UTF-8 whatever the locale, an undecodable byte
    replaced (strace escapes what it prints of buffers; a path in a foreign encoding is still
    a path to count)."""
    if path == "-":
        return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    try:
        return open(path, encoding="utf-8", errors="replace")
    except OSError as e:
        raise BuildError(f"{path}: {_cause(e)}")


def _judged(args, rows, a, b, docs) -> int:
    selfs = []
    for path in args.selfs:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise BuildError(f"{path}: {_cause(e)}")
        if not isinstance(d, dict) or d.get("aeiou_metrics") != docs[1].get("aeiou_metrics"):
            raise BuildError(f"{path}: --self is B's abstract at another seed, and its aeiou_metrics is not B's ({d.get('aeiou_metrics') if isinstance(d, dict) else None} and {docs[1].get('aeiou_metrics')})")
        for key in ("source", "sha256", "gpus", "block"):
            if d.get(key) != docs[1].get(key):
                raise BuildError(f"{path}: --self is B's abstract at another seed, and its `{key}` is not B's ({d.get(key)} and {docs[1].get(key)})")
        if d.get("seed") == docs[1].get("seed"):
            raise BuildError(f"{path}: --self has B's seed ({d.get('seed')}); another seed is the point")
        print(f"self: {path}: seed {d.get('seed')}")
        selfs.append(select(d, args.template_b))
    spec = load_tolerances(args.tolerances) if args.tolerances else None
    judged, stale = judge(rows, a, b, docs, selfs, spec)
    w = [max(len(r[i]) for r in rows + [("metric", "A", "B", 0)]) for i in range(3)]
    print(f"{'metric':<{w[0]}}  {'A':>{w[1]}}  {'B':>{w[2]}}  distance  allowed  verdict")
    for name, va, vb, dist, allowed, verdict, note in judged:
        print(f"{name:<{w[0]}}  {va:>{w[1]}}  {vb:>{w[2]}}  {'-' if dist is None else f'{dist:.3f}':>8}  {'-' if allowed is None else f'{allowed:.3f}':>7}  {verdict.upper() if verdict == 'outside' else verdict}{f' ({note})' if note else ''}")
    n = Counter(r[5] for r in judged)
    for line in stale:
        print(f"note: {line}")
    counts = f"{n['ok']} row(s) within tolerance, {n['outside']} outside, {n['unseen']} unseen, {n['not judged']} not judged"
    if n["outside"]:
        print(f"not accepted: {counts}")
        return 1
    if not n["ok"]:
        print(f"nothing judged: {counts}")
        return 1
    print(f"accepted: {counts}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
