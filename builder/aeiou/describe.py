"""The abstract in prose (`PROJECT_BRIEF.md` §6 item 27, `DESIGN_REVIEW.md` §3.76): a deterministic
walk of an AST, at a parameter set, joined with what the catalog knows of it.

An AST is a compact account of a workload's I/O; this renders it in English for readers who will
not read the JSON. Two lengths: a paragraph (context, pattern, measured shape) and an account,
section by section (parameters, data, every actor's body in order, every op named). Generated, not
written: the same AST, parameters, records, and shape give the same text, so a parameter change
shows up as a prose diff and `aeiou-catalog check` fails a description that has drifted. Every node,
expression, and distribution kind of the schema has its words here (`builder/tests/test_catalog.py`
holds the lists to the schema's).

What it reads: the AST (never the authoring script), the parameter values in effect (the defaults,
then the case's parameter file), the GPU count, and, when the catalog has them, the mechanism's and
case's curated attributes and the case's measured shape (its dry run's op counts and each dataset's
projection). What it cannot say is why the application does it (the mechanism's README) or what a
storage system should do about it.

Numbers: an expression over parameters and literals alone is worked out and shown with its
parameters (`1 MiB (hdr_read)`); one that depends on a loop index, a binding, or a draw stays a
formula. Distributions are given in words with percentiles computed exactly from their parameters,
never sampled.
"""
from __future__ import annotations

import math
import statistics

from .params import default_value
from .pattern import Pattern

# The kinds the schema defines (`schema/abstract-ast.schema.json`); a test holds these to it.
OPS = ("open", "close", "read", "write", "lseek", "ioctl", "fadvise", "fstat", "stat", "fsync", "fdatasync",
       "unlink", "ftruncate", "fallocate", "mkdir", "rmdir", "rename", "readdir")
STATEMENTS = ("let", "loop", "parallel", "channel", "put", "take", "loader", "barrier", "compute", "cond", "choose",
              "phase", "trace") + OPS
DISTS = ("const", "uniform", "uniform64", "normal", "lognormal", "empirical", "zipf", "hotset", "mixture")
EXPRS = ("param", "index", "ref", "at", "actor", "draw", "elem", "len", "sum", "size", "offset", "unit_index", "units",
         "chunks", "count", "dirs", "add", "sub", "mul", "div", "mod", "ceil_div", "min", "max", "neg", "eq", "ne", "lt",
         "le", "gt", "ge", "and", "or", "not", "cond")
HANDLES = ("ref", "file", "dir", "object", "inventory", "consume", "pick", "unit", "column")

# The measured attributes in a few words each (the dictionary has their meanings in full).
MEASURE_WORDS = {
    "run_length_read": "read runs", "run_length_write": "write runs",
    "request_size_read": "read requests", "request_size_write": "write requests",
    "read_share": "share of bytes read", "metadata": "metadata ops per data op",
    "reads_own_writes": "reads its own writes", "reread": "share of block reads that are re-reads",
    "skew": "share of accesses to the hottest 1% of blocks",
}

_ARITH = {"add": "+", "sub": "−", "mul": "×", "div": "div", "mod": "mod"}
_CMP = {"eq": "=", "ne": "≠", "lt": "<", "le": "≤", "gt": ">", "ge": "≥"}
_PREC = {"or": 1, "and": 2, "not": 3, **{k: 4 for k in _CMP}, "add": 5, "sub": 5, "mul": 6, "div": 6, "mod": 6, "neg": 7}
_WHENCE = {"SET": "the start", "CUR": "the current position", "END": "the end"}
_Z95 = statistics.NormalDist().inv_cdf(0.95)


class K:
    """A value worked out from parameters and literals."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _only(node: dict):
    return next(iter(node.items()))


# ---------------------------------------------------------------- numbers in words

def _num(v) -> str:
    """A number as a reader takes it in: thousands separated, three significant digits past the point."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1e15):
        return f"{int(v):,}"
    a = abs(v)
    return f"{v:,.0f}" if a >= 1000 else f"{float(f'{v:.3g}'):g}"


def size(n) -> str:
    if n is None:
        return "none"
    if abs(n) < 1024:
        return f"{_num(n)} byte" + ("" if n == 1 else "s")
    for i, unit in enumerate(("KiB", "MiB", "GiB", "TiB", "PiB"), 1):
        v = n / 1024 ** i
        if abs(v) < 1024 or unit == "PiB":
            return f"{v:g} {unit}" if v == int(v) else f"{v:.1f} {unit}"


def duration(n) -> str:
    for scale, unit in ((1e9, "s"), (1e6, "ms"), (1e3, "µs")):
        if abs(n) >= scale:
            return f"{_num(n / scale)} {unit}"
    return f"{_num(n)} ns"


def amount(v, unit: str | None) -> str:
    """A scalar value in its unit."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f"`{v}`"
    if unit == "bytes":
        return size(v)
    if unit == "ns":
        return duration(v)
    if unit == "tokens":
        return f"{_num(v)} token" + ("" if v == 1 else "s")
    return _num(v)


def _pct(p: float) -> str:
    return f"{float(f'{100 * p:.3g}'):g}%"


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{_num(n)} times")


def _plural(n: int, word: str) -> str:
    return f"{_num(n)} {word}" + ("" if n == 1 else "s")


# ---------------------------------------------------------------- the describer

class Describer:
    """One AST at one parameter set. `shape` is a run of the case's `shape.json` (op counts,
    each dataset's projection, the measured attributes) when the catalog has one."""

    def __init__(self, ast: dict, values: dict, gpus: int, *, shape: dict | None = None):
        self.ast = ast
        self.params = ast.get("params") or {}
        self.values = {**{k: default_value(p) for k, p in self.params.items()}, **values, "gpus": gpus}
        self.gpus = gpus
        self.shape = shape
        self.datasets = ast.get("datasets") or {}
        self.namespaces = ast.get("namespaces") or {}
        self.projection = {d["name"]: d for d in (shape or {}).get("datasets") or []}
        self.units: dict[str, str | None] = {}         # a binding's unit, as its value's
        self.loaders: set[str] = set()

    # ---- values

    def unit_of_param(self, name: str) -> str | None:
        u = (self.params.get(name) or {}).get("unit")
        return None if u == "none" else u

    def fold(self, e):
        """`e` worked out to a `K` when it depends on parameters and literals alone, else `e`."""
        if not isinstance(e, (dict, list)):
            return K(e)
        if not isinstance(e, dict) or len(e) != 1:
            return e
        k, a = _only(e)
        if k == "param":
            v = self.values.get(a)
            return K(v) if not isinstance(v, (dict, list)) and a in self.values else e
        if k == "count" and a in self.datasets:
            c = self.fold(self.datasets[a]["count"])
            return c if isinstance(c, K) else e
        if k == "dirs" and a in self.datasets:
            n = self.dirs(a)
            return e if n is None else K(n)
        if k in ("len", "sum"):
            v = self.values.get(a)
            if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
                return K(len(v) if k == "len" else sum(v))
            return e
        if k == "elem":
            v, i = self.values.get(a["array"]), self.fold(a["index"])
            if isinstance(v, list) and isinstance(i, K) and isinstance(i.value, int) and 0 <= i.value < len(v) \
                    and not isinstance(v[i.value], (dict, list)):
                return K(v[i.value])
            return e
        if k == "actor" and a == "count":
            return e
        if k in _ARITH or k in _CMP or k in ("ceil_div", "min", "max"):
            x, y = self.fold(a[0]), self.fold(a[1])
            if isinstance(x, K) and isinstance(y, K):
                return K(self._apply(k, x.value, y.value))
            if k == "mod" and isinstance(y, K) and y.value == 1:
                return K(0)
            return e
        if k in ("and", "or"):
            xs = [self.fold(x) for x in a]
            ks = [x.value for x in xs if isinstance(x, K)]
            if k == "and" and any(v is False for v in ks) or k == "or" and any(v is True for v in ks):
                return K(k == "or")
            return K(k == "and") if len(ks) == len(xs) else e
        if k == "not":
            x = self.fold(a)
            return K(not x.value) if isinstance(x, K) else e
        if k == "neg":
            x = self.fold(a)
            return K(-x.value) if isinstance(x, K) and x.value is not None else e
        if k == "cond":
            t = self.fold(a["if"])
            if isinstance(t, K):
                return self.fold(a["then"] if t.value else a["else"])
            return e
        return e

    @staticmethod
    def _apply(k, x, y):
        if k in _CMP:
            return {"eq": x == y, "ne": x != y, "lt": x is not None and y is not None and x < y,
                    "le": x is not None and y is not None and x <= y, "gt": x is not None and y is not None and x > y,
                    "ge": x is not None and y is not None and x >= y}[k]
        if x is None or y is None:
            return None
        whole = isinstance(x, int) and isinstance(y, int)
        if k in ("div", "mod", "ceil_div") and y == 0:
            return 0                                    # as the runner and datagen evaluate it
        return {"add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y,
                "div": lambda: x // y if whole else math.floor(x / y), "mod": lambda: x % y,
                "ceil_div": lambda: -(-x // y) if whole else math.ceil(x / y),
                "min": lambda: min(x, y), "max": lambda: max(x, y)}[k]()

    def const(self, e):
        f = self.fold(e)
        return f.value if isinstance(f, K) else None

    def unit_of(self, e) -> str | None:
        if not isinstance(e, dict) or len(e) != 1:
            return None
        k, a = _only(e)
        if k == "param":
            return self.unit_of_param(a)
        if k in ("elem", "sum"):
            return self.unit_of_param(a["array"] if k == "elem" else a)
        if k == "ref":
            return self.units.get(a)
        if k == "at":
            return self.units.get(a["ref"])
        if k in ("size", "offset"):
            return "bytes"
        if k in ("count", "dirs", "len", "units", "chunks", "unit_index"):
            return "count"
        if k == "draw":
            return self.unit_of(a)
        if k in ("add", "sub", "min", "max", "mod"):
            return next((u for u in map(self.unit_of, a) if u), None)
        if k == "neg":
            return self.unit_of(a)
        if k == "mul":
            us = [u for u in map(self.unit_of, a) if u]
            return "bytes" if "bytes" in us else next((u for u in us if u != "count"), us[0] if us else None)
        if k in ("div", "ceil_div"):
            x, y = self.unit_of(a[0]), self.unit_of(a[1])
            return "count" if x and x == y else x
        if k == "cond":
            return self.unit_of(a["then"]) or self.unit_of(a["else"])
        return None

    def value(self, e, hint: str | None = None) -> str:
        """An expression in words: worked out when it can be, with its formula when that names
        parameters; a formula otherwise."""
        f = self.fold(e)
        unit = hint or self.unit_of(e)
        if isinstance(f, K):
            s = amount(f.value, unit)
            return s + (f" (`{self.sym(e)}`)" if isinstance(e, dict) else "")      # the formula, when it names something
        return f"`{self.sym(e)}`"

    def counted(self, e, noun: str) -> str:
        """`3,200 samples (samples)`: a count with its noun, then the formula that names it."""
        f = self.fold(e)
        if isinstance(f, K) and isinstance(f.value, (int, float)):
            return _plural(f.value, noun) + (f" (`{self.sym(e)}`)" if isinstance(e, dict) else "")
        return f"`{self.sym(e)}` {noun}s"

    def bare(self, e, hint: str | None = None) -> str:
        """An expression's value alone, for the paragraph: worked out, or its formula."""
        f = self.fold(e)
        return amount(f.value, hint or self.unit_of(e)) if isinstance(f, K) else f"`{self.sym(e)}`"

    # ---- formulas

    def sym(self, e, outer: int = 0) -> str:
        """An expression as a formula, the parameters by name."""
        if not isinstance(e, (dict, list)):
            if e is None:
                return "none"
            if isinstance(e, bool):
                return "true" if e else "false"
            if isinstance(e, str):
                return repr(e)
            return f"{e:g}" if isinstance(e, float) else str(e)
        if isinstance(e, list) or len(e) != 1:
            return str(e)
        k, a = _only(e)
        if k in DISTS:
            return self.dist_short(e)
        if k in HANDLES and k != "ref":
            return self.handle_short(e)
        if k in ("param", "index", "ref"):
            return a
        if k == "at":
            return f"{a['ref']}@({self.sym(a['index'])})"
        if k == "actor":
            return "actor id" if a == "id" else "actors"
        if k == "draw":
            return f"draw({self.dist_short(a)})"
        if k == "elem":
            return f"{a['array']}[{self.sym(a['index'])}]"
        if k in ("len", "sum", "count", "dirs"):
            return f"{k}({a})"
        if k in ("size", "offset", "unit_index", "units", "chunks"):
            return f"{k}({self.handle_short(a)})"
        if k == "ceil_div":
            return f"⌈{self.sym(a[0], 6)} / {self.sym(a[1], 7)}⌉"
        if k in ("min", "max"):
            return f"{k}({self.sym(a[0])}, {self.sym(a[1])})"
        if k == "cond":
            return f"(if {self.sym(a['if'])} then {self.sym(a['then'])} else {self.sym(a['else'])})"
        p = _PREC.get(k)
        if p is None:
            return str(e)
        if k == "neg":
            s = f"−{self.sym(a, p)}"
        elif k == "not":
            s = f"not {self.sym(a, p)}"
        elif k in ("and", "or"):
            s = f" {k} ".join(self.sym(x, p) for x in a)
        else:
            op = _ARITH.get(k) or _CMP[k]
            s = f"{self.sym(a[0], p)} {op} {self.sym(a[1], p + 1)}"
        # `mod` and `div` inside another operator are bracketed: `a − b mod 64` is read wrongly
        return f"({s})" if p < outer or (outer and k in ("div", "mod")) else s

    def handle_short(self, h) -> str:
        if not isinstance(h, dict) or len(h) != 1:
            return self.sym(h)
        k, a = _only(h)
        if k == "ref":
            return a
        if k == "file":
            if "of" in a:
                s = f"file of {self.handle_short(a['of'])}"
            else:
                s = f"{a['dataset']}[{self.sym(a['id'])}]"
            return s + (f" chunk {self.sym(a['chunk'])}" if "chunk" in a else "")
        if k == "dir":
            return f"dir {a['dataset']}[{self.sym(a['id'])}]"
        if k == "object":
            return f"{a['namespace']}{{{', '.join(f'{f}: {self.sym(v)}' for f, v in a['fields'].items())}}}"
        if k in ("inventory", "consume"):
            return f"{k}({a})"
        if k == "pick":
            return f"pick({a['dataset']})"
        if k in ("unit", "column"):
            return f"{k} {self.sym(a['index']) if 'index' in a else ''} of {self.handle_short(a['of'])}".replace("  ", " ")
        return str(h)

    # ---- distributions

    def resolve_dist(self, d):
        """(the distribution literal, the parameter it came from) for a distref; (None, None)
        when it is chosen by something not worked out here."""
        name = None
        for _ in range(16):
            if isinstance(d, list):
                return {"mixture": d}, name
            if not isinstance(d, dict) or len(d) != 1:
                return None, name
            k, a = _only(d)
            if k in DISTS:
                return d, name
            if k == "param" and isinstance(self.values.get(a), (dict, list)):
                d, name = self.values[a], a
                continue
            if k == "cond":
                t = self.fold(a["if"])
                if not isinstance(t, K):
                    return None, name
                d = a["then"] if t.value else a["else"]
                continue
            if k == "draw":
                d = a
                continue
            return None, name
        return None, name

    def dist_short(self, d) -> str:
        lit, name = self.resolve_dist(d)
        if name:
            return name
        if lit is None:
            return self.sym(d) if not (isinstance(d, dict) and "cond" in d) else "a distribution chosen by a condition"
        return _only(lit)[0]

    def dist(self, d, unit: str | None = None, over: str | None = None) -> str:
        """A distribution in words; `over` names the dataset a draw over ids is applied to."""
        lit, name = self.resolve_dist(d)
        if lit is None:
            return f"the distribution `{self.sym(d)}` selects"
        unit = unit or (self.unit_of_param(name) if name else None)
        words = self._dist(lit, unit, over)
        return words + (f" (`{name}`)" if name else "")

    def _dist(self, lit: dict, unit, over) -> str:
        k, a = _only(lit)
        fmt = lambda e: self.bare(e, unit)
        if k == "const":
            return f"always {fmt(a)}"
        if k == "uniform":
            lo, hi = self.const(a["lo"]), self.const(a["hi"])
            if isinstance(lo, int) and isinstance(hi, int):
                return f"uniform over {amount(lo, unit)} to {amount(hi - 1, unit)}"
            return f"uniform over [{fmt(a['lo'])}, {fmt(a['hi'])})"
        if k == "uniform64":
            return "a random 64-bit identifier"
        if k in ("normal", "lognormal"):
            lo, hi = (self.const(a[b]) if b in a else None for b in ("min", "max"))
            if lo is not None and lo == hi:
                return f"always {amount(lo, unit)} (a {k} clamped to it)"
            bounds = self._bounds(a, unit)
            if k == "normal":
                m, sd = self.const(a["mean"]), self.const(a["sd"])
                head = f"normal, mean {fmt(a['mean'])}, standard deviation {fmt(a['sd'])}"
                lo, hi = (m - _Z95 * sd, m + _Z95 * sd) if isinstance(m, (int, float)) and isinstance(sd, (int, float)) else (None, None)
            else:
                m, s = self.const(a["median"]), a["sigma"]
                head = f"lognormal, median {fmt(a['median'])}, σ {s:g}"
                lo, hi = (m * math.exp(-_Z95 * s), m * math.exp(_Z95 * s)) if isinstance(m, (int, float)) else (None, None)
            if lo is not None:
                lo, hi = (amount(round(self._clamp(v, a), 1), unit) for v in (lo, hi))
                head += f" (5th and 95th percentiles both {lo})" if lo == hi else f" (5th–95th percentile {lo} to {hi})"
            return head + bounds
        if k == "empirical":
            vals, ws = a["values"], a["weights"]
            total = sum(ws)
            if len(vals) <= 6 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                return "one of " + ", ".join(f"{amount(v, unit)} ({_pct(w / total)})" for v, w in zip(vals, ws))
            pairs = sorted(zip(vals, ws))
            q = lambda p: next(v for v, c in zip([v for v, _ in pairs], _cum([w for _, w in pairs])) if c >= p * total)
            mean = sum(v * w for v, w in pairs) / total
            return (f"one of {len(vals)} weighted values, median {amount(q(0.5), unit)}, 5th–95th percentile "
                    f"{amount(q(0.05), unit)} to {amount(q(0.95), unit)}, mean {amount(round(mean, 1), unit)}")
        if k == "zipf":
            return f"Zipf over the ids{f' of `{over}`' if over else ''} with exponent {a['s']:g}, in a rank order fixed by the dataset seed"
        if k == "hotset":
            return (f"{_pct(a['weight'])} of draws over the hottest {_pct(a['fraction'])} of the ids{f' of `{over}`' if over else ''}"
                    " (a rank order fixed by the dataset seed), the rest over the others")
        if k == "mixture":
            total = sum(arm["weight"] for arm in a) or 1
            return "a mixture: " + "; ".join(f"{_pct(arm['weight'] / total)} " + ("none" if arm["dist"] is None else self.dist(arm["dist"], unit, over))
                                             for arm in a)
        return str(lit)

    def _clamp(self, v, a):
        """A value as the runner clamps a draw: raised to `min`, then lowered to `max` (eval.rs
        `clamp`), so a clamp past both percentiles moves both."""
        for key, f in (("min", max), ("max", min)):
            b = self.const(a[key]) if key in a else None
            if isinstance(b, (int, float)):
                v = f(v, b)
        return v

    def _bounds(self, a, unit) -> str:
        have = [k for k in ("min", "max") if k in a]
        if not have:
            return ""
        return ", clamped to " + " and ".join(f"{'at least' if k == 'min' else 'at most'} {self.bare(a[k], unit)}" for k in have)

    # ---- handles

    def handle(self, h) -> str:
        """A handle in words."""
        if not isinstance(h, dict) or len(h) != 1:
            return f"`{self.sym(h)}`"
        k, a = _only(h)
        if k == "ref":
            return f"`{a}`"
        if k == "file":
            if "of" in a:
                s = f"the file holding {self.handle(a['of'])}"
            else:
                s = f"file {self.value(a['id'], 'count')} of `{a['dataset']}`"
            return (f"chunk object {self.value(a['chunk'], 'count')} of " + s) if "chunk" in a else s
        if k == "dir":
            return f"directory {self.value(a['id'], 'count')} of `{a['dataset']}`"
        if k == "object":
            return f"the object of `{a['namespace']}` named by " + ", ".join(f"{f} = {self.value(v)}" for f, v in a["fields"].items())
        if k == "inventory":
            return f"the inventory of `{a}`"
        if k == "consume":
            what = "shard (a whole file)" if self.datasets.get(a, {}).get("access") == "stream" else "sample"
            return f"the next {what} of `{a}` from the shuffle, without replacement"
        if k == "pick":
            how = self.dist(a["dist"], over=a["dataset"]) if "dist" in a else "uniform over the ids"
            return f"a sample of `{a['dataset']}` drawn with replacement, {how}"
        if k == "unit":
            if "index" in a:
                return f"unit {self.value(a['index'], 'count')} of {self.handle(a['of'])}"
            return f"the unit holding {self.handle(a['of'])}"
        if k == "column":
            return f"column chunk {self.value(a['index'], 'count')} of {self.handle(a['of'])}"
        return f"`{self.sym(h)}`"

    def _handle_target(self, h) -> str | None:
        """The dataset or namespace a handle reaches, for the paragraph."""
        if not isinstance(h, dict) or len(h) != 1:
            return None
        k, a = _only(h)
        if k in ("consume", "inventory"):
            return a
        if k in ("file", "dir", "pick"):
            return a.get("dataset") if "dataset" in a else self._handle_target(a.get("of"))
        if k == "object":
            return a["namespace"]
        if k in ("unit", "column"):
            return self._handle_target(a["of"])
        return None

    # ---- the data

    def dirs(self, name: str) -> int | None:
        """The dataset's directories: the projection's when the shape has it, else from the pattern
        (a field of `id` in the directory part, `{id div D}` making ceil(files / D) of them)."""
        if name in self.projection:
            return self.projection[name]["directories"]
        d = self.datasets.get(name)
        count = self.const(d["count"]) if d else None
        if not isinstance(count, int):
            return None
        ints = {k: v for k, v in self.values.items() if isinstance(v, int) and not isinstance(v, bool)}
        try:
            head = Pattern(d["pattern"]).bind(ints).source.rpartition("/")[0]
        except ValueError:
            return None
        if not head:
            return 0
        spf = self.const(d.get("samples_per_file", 1)) if d["layout"] == "compound" else 1
        files = -(-count // spf) if isinstance(spf, int) and spf > 0 else count
        ids = [seg for seg in Pattern(head).segments if seg[0] == "field" and seg[1] == "id"]
        if not ids:
            return 1
        op, n = ids[0][2], ids[0][3]
        return files if op is None else -(-files // n) if op == "div" else min(files, n)

    def dataset(self, name: str, d: dict) -> list[str]:
        out = []
        count = self.bare(d["count"], "count")
        layout = d["layout"]
        what = {"whole": "one sample a file",
                "chunked": f"each sample stored as chunk objects of {self.value(d.get('chunk'), 'bytes')} (`{{k}}` in the name)",
                "compound": f"{self.value(d.get('samples_per_file'), 'count')} samples a file",
                "array": f"one file of {count} slots of {self.value(d.get('slot'), 'bytes')} each"}[layout]
        access = {"map": "consumed as samples, drawn without replacement from a keyed shuffle and split over the GPUs by "
                         "position (`g + G·(b·B + j)`)",
                  "stream": "consumed as whole files (shards), drawn without replacement from a keyed shuffle and split "
                            "over the GPUs by position, each read in order",
                  "indexed": "looked up by id, never consumed"}[d["access"]]
        where = " on an S3 store" if d.get("protocol") == "s3" else ""
        out.append(f"**`{name}`**, a dataset{where} (`{layout}`, `{d['access']}`): {self.counted(d['count'], 'sample')}, {what}; {access}.")
        pat = d["pattern"]
        bound = Pattern(pat).params()
        out.append(f"Named `{pat}`" + (" (" + ", ".join(f"`{p}` = {amount(self.values.get(p), 'count')}" for p in bound) + ")" if bound else "") + ".")
        out.append(f"Sample sizes: {self.dist(d['size'], 'bytes')}.")
        fmt = d.get("format")
        if fmt:
            parts = [f"format class `{fmt['class']}`"] + [f"{k} `{fmt[k]}`" for k in ("reader", "version") if k in fmt]
            fr = fmt.get("framing") or {}
            if fr:
                cols = fr.get("columns") or []
                parts.append(f"framing: {_plural(len(cols), 'column') if cols else 'one column'}"
                             + (f", {self.value(fr['unit'], 'count')} samples a unit" if "unit" in fr else ""))
            out.append("Written as " + ", ".join(parts) + ".")
        if d.get("inventory"):
            inv = d["inventory"]
            out.append(f"An inventory, `{inv['name']}` in the dataset's root, {self.value(inv['entry'], 'bytes')} a sample.")
        p = self.projection.get(name)
        if p:
            inv = f", the inventory's {size(p['inventory_bytes'])} included" if p["inventory_bytes"] else ""
            out.append(f"At these parameters `aeiou datagen` writes {_plural(p['files'], 'file')} in "
                       f"{_plural(p['directories'], 'directory').replace('directorys', 'directories')}, "
                       f"{size(p['bytes'])} in all{inv} (a mean file of {size(p['mean_file'])}).")
        out.append(f"Dataset seed `{d['seed']:#x}`." + (f" {d['doc']}" if d.get("doc") else ""))
        return out

    def namespace(self, name: str, n: dict) -> list[str]:
        where = " on an S3 store" if n.get("protocol") == "s3" else ""
        if n.get("input"):
            role = ("read, written by an earlier run of the same seed, GPUs, and common parameters (`same_run`)" if n.get("same_run")
                    else "read, written by an earlier run")
        else:
            role = "written by this workload, empty at the start"
        sz = "the sum of the writes that create it" if n["size"] == "as_written" else self.value(n["size"], "bytes")
        fields = ", ".join(f"`{f}`" for f in n["fields"])
        out = [f"**`{name}`**, a namespace{where}: objects {role}. Named `{n['pattern']}` by {fields}; an object's size is {sz}.",
               f"Content seed `{n['seed']:#x}`." + (f" {n['doc']}" if n.get("doc") else "")]
        return out

    # ---- the account: every statement in order

    def body(self, body: list, depth: int = 0) -> list[str]:
        out = []
        for node in body:
            out += self.statement(node, depth)
        return out

    def statement(self, node: dict, depth: int) -> list[str]:
        k, a = _only(node)
        pad = "  " * depth + "- "
        if k == "let":
            return [pad + self.let(a)]
        if k == "loop":
            return [pad + self.loop_head(a) + ":"] + self.body(a["body"], depth + 1)
        if k == "parallel":
            w = self.value(a["width"], "count")
            return [pad + f"{w} in parallel, `{a['index']}` = 0 onward, joined at the end:"] + self.body(a["body"], depth + 1)
        if k == "channel":
            return [pad + f"A channel `{a['name']}` of {self.value(a['capacity'], 'count')} slots"
                    + (", delivering in sequence order" if a.get("ordered", True) else ", delivering as items arrive") + "."]
        if k == "put":
            return [pad + f"Puts item {self.value(a['seq'], 'count')} on `{a['channel']}`, waiting while it is full."]
        if k == "take":
            what = "batch from the loader" if a["channel"] in self.loaders else "item from"
            return [pad + f"Takes the next {what} `{a['channel']}`."]
        if k == "loader":
            self.loaders.add(a["name"])
            w, p, b = (self.value(a[x], "count") for x in ("workers", "prefetch", "batches"))
            order = "in batch order" if a.get("ordered", True) else "as they are ready"
            return [pad + f"A loader `{a['name']}`: {w} workers, worker w building batches w, w + W, …, each up to {p} "
                    f"batches ahead; {b} batches in all, delivered {order}. For batch `{a['index']}`:"] + self.body(a["body"], depth + 1)
        if k == "barrier":
            return [pad + self.barrier(a["scope"]).capitalize() + "."]
        if k == "compute":
            return [pad + f"Computes for {self.value(a['ns'], 'ns')}, no I/O."]
        if k == "cond":
            t = self.fold(a["if"])
            if isinstance(t, K):
                here = " (true here)" if t.value else " (false here: skipped)"
                other = " (taken here)" if not t.value else " (not taken here)"
            else:
                here, other = "", ""
            out = [pad + f"If `{self.sym(a['if'])}`{here}:"] + self.body(a["then"], depth + 1)
            if a.get("else"):
                out += [pad + f"Otherwise{other}:"] + self.body(a["else"], depth + 1)
            return out
        if k == "choose":
            total = sum(arm["weight"] for arm in a["arms"]) or 1
            out = [pad + "One of these, chosen by a positional draw:"]
            for arm in a["arms"]:
                out += ["  " * (depth + 1) + f"- With probability {_pct(arm['weight'] / total)}:"] + self.body(arm["body"], depth + 2)
            return out
        if k == "phase":
            return [pad + f"Phase {self.label(a['name'])}:"] + self.body(a["body"], depth + 1)
        if k == "trace":
            return [pad + f"Replays the captured trace `{a['file']}` literally (SHA-256 `{a['sha256'][:12]}…`), one lane per traced task."]
        return [pad + self.op(k, a) + "."]

    def label(self, name) -> str:
        """A phase's name; one chosen by a condition between two names reads as both."""
        if isinstance(name, dict) and "cond" in name:
            c = name["cond"]
            if isinstance(c["then"], str) and isinstance(c["else"], str):
                t = self.fold(c["if"])
                if isinstance(t, K):
                    return f"`{c['then'] if t.value else c['else']}`"
                return f"`{c['then']}` while `{self.sym(c['if'])}`, else `{c['else']}`,"
        return f"`{name}`" if isinstance(name, str) else self.value(name)

    def barrier(self, scope) -> str:
        if scope == "global":
            return "waits at a barrier with every actor on every host"
        if scope == "host":
            return "waits at a barrier with every actor on its host"
        return f"waits at the barrier `{scope}`"

    def let(self, a: dict) -> str:
        name, v = a["name"], a["value"]
        if isinstance(v, dict) and len(v) == 1:
            k, x = _only(v)
            if k in HANDLES and k != "ref" and (k not in ("file",) or isinstance(x, dict)):
                self.units[name] = None
                return f"`{name}` is {self.handle(v)}."
            if k == "draw" or k in DISTS:
                lit = x if k == "draw" else v
                _, pname = self.resolve_dist(lit)
                self.units[name] = self.unit_of_param(pname) if pname else self.unit_of(lit)
                return f"`{name}` is drawn from {self.dist(lit, self.units[name])}."
        self.units[name] = self.unit_of(v)
        return f"`{name}` = {self.value(v)}."

    def loop_head(self, a: dict) -> str:
        i, to = a["index"], a["to"]
        start, step = a.get("from"), a.get("step")
        if isinstance(to, dict) and "dirs" in to and start is None and step is None:
            n = self.const(to)
            many = f"each of the {_num(n)} directories" if n is not None else "each directory"
            return f"For {many} `{i}` of `{to['dirs']}`"
        n = self.const(to)
        if (start is None or self.const(start) == 0) and step is None and isinstance(n, int):
            src = f" (`{self.sym(to)}`)" if isinstance(to, dict) else ""
            return f"{_times(n).capitalize()}{src}, `{i}` = 0 to {_num(n - 1)}"
        rng = f"from {self.value(start) if start is not None else '0'} while below {self.value(to)}"
        return f"For `{i}` {rng}" + (f", in steps of {self.value(step)}" if step is not None else "")

    def op(self, k: str, a: dict) -> str:
        f = self.handle(a.get("file") or a.get("dir") or a.get("from")) if k != "rename" else None
        expect = f" (may fail with {', '.join(a['expect'])})" if a.get("expect") else ""
        if k == "open":
            mode = f", mode {a['mode']:#o}" if "mode" in a else ""
            return f"`open` {f} ({'|'.join(a['flags'])}{mode}){expect}"
        if k in ("read", "write"):
            ln = self.value(a["len"], "bytes")
            at = f" at offset {self.value(a['offset'], 'bytes')}" if "offset" in a else " at the file position"
            rep = a.get("repeat")
            if rep == "until_eof":
                return f"`read` {f}: requests of {ln}{at} until end of file, the short or empty last one included{expect}"
            if rep is not None:
                return f"`{k}` {f}: {self.value(rep, 'count')} requests of {ln}{at}{expect}"
            return f"`{k}` {f}: {ln}{at}{expect}"
        if k == "lseek":
            to = "0" if self.const(a["offset"]) == 0 else self.value(a["offset"], "bytes")
            return f"`lseek` {f} to {to} from {_WHENCE[a['whence']]}"
        if k == "ioctl":
            return f"`ioctl` {f} `{a['request']}`{expect}"
        if k == "fadvise":
            rng = ""
            if "offset" in a or "len" in a:
                ln = self.const(a["len"]) if "len" in a else None
                rng = f", {'to the end' if ln == 0 else self.value(a['len'], 'bytes')} from {self.value(a.get('offset', 0), 'bytes')}"
            return f"`posix_fadvise` {f} `{a['advice']}`{rng}{expect}"
        if k == "ftruncate":
            return f"`ftruncate` {f} to {self.value(a['len'], 'bytes')}{expect}"
        if k == "fallocate":
            return f"`fallocate` {f}: {self.value(a['len'], 'bytes')} from {self.value(a.get('offset', 0), 'bytes')}{expect}"
        if k == "rename":
            return f"`rename` {self.handle(a['from'])} to {self.handle(a['to'])}{expect}"
        if k == "readdir":
            return f"`readdir` {f} to the end{expect}"
        return f"`{k}` {f}{expect}"

    # ---- the paragraph: the same walk, the parameters' values in place and dead branches out

    def gist(self, body: list) -> str:
        parts, ops, subject = [], [], None

        def flush():
            nonlocal subject
            if ops:
                parts.append((f"on {subject}, " if subject else "") + _runs([self.op_gist(k, a) for k, a in ops]))
                ops.clear()
                subject = None

        for node in body:
            k, a = _only(node)
            if k in OPS:
                ops.append((k, a))
                continue
            if k == "let":
                v = a["value"]
                if isinstance(v, dict) and len(v) == 1 and _only(v)[0] in HANDLES and _only(v)[0] != "ref":
                    flush()
                    subject = self.handle_gist(v, subject)
                self.let(a)                             # the binding's unit, for what follows
                continue
            flush()
            g = self.node_gist(k, a)
            if g:
                parts.append(g)
        flush()
        return "; ".join(parts)

    def node_gist(self, k: str, a: dict) -> str:
        if k == "loop":
            inner = self.gist(a["body"])
            if not inner:
                return ""
            to = a["to"]
            n = self.const(to)
            if isinstance(to, dict) and "dirs" in to:
                return f"for each of the {_num(n) + ' ' if n is not None else ''}directories of `{to['dirs']}`: {inner}"
            if a.get("from") is None and a.get("step") is None and isinstance(n, int):
                return f"{_times(n)}: {inner}" if n > 0 else ""
            return f"for each `{a['index']}` below {self.bare(to, 'count')}: {inner}"
        if k == "parallel":
            if self.const(a["width"]) == 1:
                return self.gist(a["body"])
            return f"{self.bare(a['width'], 'count')} in parallel, each: {self.gist(a['body'])}"
        if k == "loader":
            self.loaders.add(a["name"])
            return (f"{self.bare(a['workers'], 'count')} loader workers build {self.bare(a['batches'], 'count')} batches, "
                    f"up to {self.bare(a['prefetch'], 'count')} ahead each; per batch, {self.gist(a['body'])}")
        if k == "phase":
            inner = self.gist(a["body"])
            return f"in phase {self.label(a['name'])}, {inner}" if inner else ""
        if k == "cond":
            t = self.fold(a["if"])
            if isinstance(t, K):
                return self.gist(a["then"] if t.value else a.get("else") or [])
            inner, other = self.gist(a["then"]), self.gist(a.get("else") or [])
            s = f"when `{self.sym(a['if'])}`, {inner or 'nothing'}"
            return s + (f", otherwise {other}" if other else "")
        if k == "choose":
            total = sum(arm["weight"] for arm in a["arms"]) or 1
            return "one of: " + "; ".join(f"({_pct(arm['weight'] / total)}) {self.gist(arm['body']) or 'nothing'}" for arm in a["arms"])
        if k == "take":
            return f"take a batch from `{a['channel']}`" if a["channel"] in self.loaders else f"take an item from `{a['channel']}`"
        if k == "put":
            return f"put an item on `{a['channel']}`"
        if k == "compute":
            return "" if self.const(a["ns"]) == 0 else f"compute for {self.bare(a['ns'], 'ns')}"
        if k == "barrier":
            return {"global": "a barrier with every GPU", "host": "a barrier with its host's GPUs"}.get(a["scope"], f"barrier `{a['scope']}`")
        if k == "trace":
            return "replay a captured trace"
        return ""                                       # a channel's declaration

    def handle_gist(self, h, pending: str | None = None) -> str:
        """A handle in a few words; `pending` is the one the last binding named, no op between."""
        k, a = _only(h)
        if k == "consume":
            what = "shard" if self.datasets.get(a, {}).get("access") == "stream" else "sample"
            return f"the next {what} of `{a}`"
        if k == "pick":
            return f"a sample of `{a['dataset']}` drawn with replacement"
        if k == "file":
            return f"a file of `{self._handle_target(h)}`" if "dataset" in a else f"the file holding {pending or 'it'}"
        if k == "dir":
            return f"a directory of `{a['dataset']}`"
        if k == "object":
            return f"an object of `{a['namespace']}`"
        if k == "inventory":
            return f"the inventory of `{a}`"
        if k == "unit":
            return f"a unit of {pending or 'it'}" if _only(a["of"])[0] == "ref" else f"a unit of {self.handle_gist(a['of'])}"
        if k == "column":
            return "a column chunk"
        return ""

    def op_gist(self, k: str, a: dict) -> str:
        if k == "open":
            fl = set(a["flags"])
            how = "for writing" if fl & {"WRONLY", "RDWR", "CREAT"} else ""
            how += (" with O_DIRECT" if "DIRECT" in fl else "")
            return f"open{(' ' + how.strip()) if how else ''}"
        if k in ("read", "write"):
            ln = self.bare(a["len"], "bytes")
            rep = a.get("repeat")
            where = " at an offset" if "offset" in a else ""
            if rep == "until_eof":
                return f"read in {ln} requests{where} to end of file"
            if rep is not None:
                return f"{self.bare(rep, 'count')} {k}s of {ln}{where}"
            return f"{k} {ln}{where}"
        if k == "ioctl":
            return f"ioctl {a['request']}"
        if k == "readdir":
            return "readdir to the end"
        if k == "fadvise":
            return f"fadvise {a['advice']}"
        return k

    # ---- the whole

    def actor_count(self, name: str, actor: dict):
        return self.const(actor.get("count", {"param": "gpus"}))

    def paragraph(self, context: list[str]) -> str:
        """The paragraph: what it is, what it reads and writes, what each actor does, and its shape."""
        s = [_sentence(self.ast.get("doc") or f"The abstract `{self.ast['name']}`.")]
        s += [_sentence(c) for c in context if c]
        for name, d in self.datasets.items():
            p = self.projection.get(name)
            proj = (f" ({size(p['bytes'])} in {_plural(p['files'], 'file')}"
                    + (f" and {_num(p['directories'])} directories" if p["directories"] > 1 else "") + ")") if p else ""
            lit, _ = self.resolve_dist(d["size"])
            sizes = self._dist(lit, "bytes", None) if lit else "sizes a distribution sets"
            per = {"whole": "one a file", "chunked": "stored as chunk objects", "compound": f"{self.bare(d.get('samples_per_file'), 'count')} a file",
                   "array": "slots of one file"}[d["layout"]]
            how = {"map": "consumed in a keyed shuffle without replacement", "stream": "consumed shard by shard in a keyed shuffle",
                   "indexed": "looked up by id"}[d["access"]]
            n = self.const(d["count"])
            many = _plural(n, "sample") if isinstance(n, int) else f"`{self.sym(d['count'])}` samples"
            s.append(f"`{name}` holds {many}, {per}{proj}, {how}; sizes {sizes}.")
        for name, n in self.namespaces.items():
            sz = "as written" if n["size"] == "as_written" else self.bare(n["size"], "bytes")
            role = "reads" if n.get("input") else "writes"
            s.append(f"It {role} the objects of `{name}`, each {sz}.")
        for name, actor in (self.ast.get("actors") or {}).items():
            n = self.actor_count(name, actor)
            who = (f"each of the {_num(n)} `{name}` actors" if n != 1 else f"the one `{name}` actor") + (
                " (one per GPU)" if "count" not in actor else "")
            s.append(_sentence(f"{who[0].upper()}{who[1:]}: {self.gist(actor['body'])}"))
        if self.shape:
            s.append(_sentence(self.shape_gist()))
        return " ".join(s)

    def shape_gist(self) -> str:
        sh = self.shape
        c = sh.get("counts") or {}
        counts = ", ".join(f"{_num(n)} {k}" for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
        rw = f"{_bytes_moved(sh['bytes_read'])} read, {_bytes_moved(sh['bytes_written'])} written" if "bytes_read" in sh else ""
        attrs = []
        for n, a in sh["attributes"].items():
            if a["class"] is None:
                continue
            v = self._measure(n, a["value"]) if a["value"] is not None else a["class"]
            attrs.append(f"{MEASURE_WORDS.get(n, n)}: {a['class']}" + ("" if v == a["class"] else f" ({v})"))
        head = f"Measured by dry run at {_plural(sh['gpus'], 'GPU')}"
        return head + (f" ({_num(sh['ops'])} ops: {counts}; {rw})" if counts else "") + ": " + "; ".join(attrs)

    @staticmethod
    def _measure(name: str, v) -> str:
        if name.startswith(("run_length", "request_size")):
            return size(v)
        if name in ("read_share", "reread", "skew"):
            return _pct(v)
        if name == "reads_own_writes":
            return "yes" if v else "no"
        return _num(v)

    def parameters(self) -> list[str]:
        if not self.params:
            return ["None."]
        out = ["| Parameter | Value | Default | Unit | What it is |", "|---|---|---|---|---|"]
        for name, p in self.params.items():
            unit = self.unit_of_param(name)
            v, dflt = self.values.get(name), default_value(p)
            shown = lambda x: (self.dist(x, unit) if isinstance(x, dict) else _array(x, unit) if isinstance(x, list) else amount(x, unit)).replace("|", "\\|")
            doc = (p.get("doc") or "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| `{name}` | {shown(v)} | {'same' if v == dflt else shown(dflt)} | {p.get('unit', '—')} | {doc} |")
        return out

    def account(self) -> list[str]:
        out = ["### Parameters", ""] + self.parameters() + [""]
        api = [f"`posix` {self.ast['posix']}" if "posix" in self.ast else "`posix` sync (none declared)"]
        api += [f"`cache` {self.ast['cache']}"] if "cache" in self.ast else []
        api += [f"`s3` {self.ast['s3']}"] if "s3" in self.ast else []
        out += [f"The API the traced application used: {', '.join(api)}; a run under another backend is another workload.", ""]
        if self.datasets or self.namespaces:
            out += ["### Data", ""]
            for name, d in self.datasets.items():
                out += [" ".join(self.dataset(name, d)), ""]
            for name, n in self.namespaces.items():
                out += [" ".join(self.namespace(name, n)), ""]
        for name, actor in (self.ast.get("actors") or {}).items():
            n = self.actor_count(name, actor)
            many = (f"{_plural(n, 'instance')}, one per GPU" if "count" not in actor
                    else f"{self.value(actor['count'], 'count')} instances")
            out += [f"### The `{name}` actor ({many})", ""]
            if actor.get("doc"):
                out += [actor["doc"], ""]
            self.units.clear()
            out += self.body(actor["body"]) + [""]
        if self.shape:
            out += self.shape_table()
        return out

    def shape_table(self) -> list[str]:
        sh = self.shape
        out = ["### Measured shape", "",
               f"From `aeiou dry-run --metrics-json` at {_plural(sh['gpus'], 'GPU')}, the median of seeds "
               f"{', '.join(map(str, sh['seeds']))}" + ("" if sh["stable"] else f"; the seeds differ on {', '.join(sh['moving'])}") + "."]
        if sh.get("counts"):
            out += ["", f"Ops in all (seed {sh['seeds'][0]}): {_num(sh['ops'])}: " +
                    ", ".join(f"{_num(n)} `{k}`" for k, n in sorted(sh["counts"].items(), key=lambda kv: (-kv[1], kv[0])))
                    + f"; {_bytes_moved(sh['bytes_read'])} read, {_bytes_moved(sh['bytes_written'])} written."]
        out += ["", "| Attribute | Class | Value |", "|---|---|---|"]
        for n, a in sh["attributes"].items():
            out.append(f"| `{n}` ({MEASURE_WORDS.get(n, n)}) | {a['class'] or '—'} | {self._measure(n, a['value']) if a['value'] is not None else '—'} |")
        return out + [""]


def _bytes_moved(n: int) -> str:
    return "nothing" if not n else size(n)


def _cum(ws):
    acc = 0
    for w in ws:
        acc += w
        yield acc


def _array(v: list, unit) -> str:
    head = ", ".join(amount(x, unit) if not isinstance(x, (dict, list)) else "…" for x in v[:8])
    return f"[{head}{', …' if len(v) > 8 else ''}] ({len(v)} values)" if len(v) > 8 else f"[{head}]"


def _runs(words: list[str]) -> str:
    """Consecutive equal steps counted: `lseek, lseek` is `lseek twice`."""
    out = []
    for w in words:
        if out and out[-1][0] == w:
            out[-1][1] += 1
        else:
            out.append([w, 1])
    return ", ".join(w if n == 1 else f"{w} {_times(n)}" for w, n in out)


def _sentence(s: str) -> str:
    s = s.strip()
    return s if s.endswith((".", "!", "?")) else s + "."
