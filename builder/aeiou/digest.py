"""`aeiou-trace digest`: the structural summary of a trace that a fit is made from
(`PROJECT_BRIEF.md` item 47; `DESIGN_REVIEW.md` §3.86 for what it can and cannot decide, §3.87
for its design), with a verdict per site on what the trace decides and an `open` list of what
it does not, ranked by how far the candidates diverge at the scale the benchmark will run.

The digest reads the same calls `aeiou-trace export` writes (the exporter's event stream, so
descriptor tables, `clone`, `dup`, and positions are the one resolver's), or an exported trace
file, and reduces them to:

- **path templates**: every run of digits in a path is a field (`train/{a:05}/sample_{b:09}.npz`),
  a run of 16 or more hex digits with a letter in it a hex field (`{a:x32}`); per template, how
  many values each field took and their range, the op mix, bytes, request sizes, and how each
  open moved through its file (`whole`: one sequential run from offset 0; `sequential`: one run
  from elsewhere; `positioned`: more than one run). No raw path unless `paths=True`.
- **lanes and phases**: a lane is a traced task; a unit is the calls on one open from `open` to
  `close` (a path call, or an `io_submit`, is a unit of one op); a unit's motif is its op
  sequence with repeats collapsed (`read+`, `(lseek read)+`); a phase is a run of units of one
  motif on one template, with the repeat counts and the think time between units.
- **sites**: a template opened at least twice, its opens in order. The selection verdict says
  what the sequence of paths decides between a permutation pass (`consume`: each path once per
  pass) and a draw with replacement (`choose`): a repeat before the set is exhausted refutes
  the pass; passes of equal length refute the draw; no repeat at all decides nothing unless
  the corpus is small enough that a draw would have repeated (the birthday bound, at 5 %), and
  then says how many touches would decide it. Order within a pass (sequential, permuted) and
  the bytes per open (constant, per path, drawn) are judged the same way.
- **open**: every undecided verdict as a question in the application's terms, its candidate
  constructs, what evidence settles it, and the divergence of the candidates at `epochs` passes
  (the share of the set a draw with replacement never reads, `e^-E`; zero for a pass), ranked.

Standard library only. The digest never asks anything; the asking is a person's or the fitting
skill's (item 48), and answers belong beside the capture with their provenance.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
from collections import Counter

from .nodes import BuildError
from .trace import KINDS, TRACE_FORMAT, LogHist, export_of, nothing_counted

FORMAT = 1
P_REFUTE = 0.05                 # a candidate whose chance of producing the evidence is below this is refuted
_LN = math.log(1 / P_REFUTE)    # ln 20
_FIELD = re.compile(r"[0-9a-f]{16,}|[0-9]+")
_MARK = "\0"
MAX_PERIOD = 4                  # the longest cycle of ops collapsed in a motif


# ---------------------------------------------------------------- templates


class Field:
    """One varying run in a template: its kind, the widths and values it took."""

    __slots__ = ("kind", "widths", "values", "lo", "hi")

    def __init__(self, kind: str):
        self.kind = kind
        self.widths: set[int] = set()
        self.values: set[str] = set()
        self.lo = self.hi = None

    def add(self, text: str):
        self.widths.add(len(text))
        self.values.add(text)
        if self.kind == "int":
            v = int(text)
            self.lo = v if self.lo is None else min(self.lo, v)
            self.hi = v if self.hi is None else max(self.hi, v)

    def name(self, i: int) -> str:
        return string.ascii_lowercase[i] if i < 26 else f"f{i}"

    def width(self) -> int | None:
        (w,) = self.widths if len(self.widths) == 1 else (None,)
        return w if w is not None and w > 1 else None

    def placeholder(self, i: int) -> str:
        w = self.width()
        if self.kind == "hex":
            return "{%s:x%d}" % (self.name(i), w) if w else "{%s:x}" % self.name(i)
        return "{%s:0%d}" % (self.name(i), w) if w else "{%s}" % self.name(i)

    def out(self, i: int) -> dict:
        d = {"name": self.name(i), "kind": self.kind, "width": self.width(), "values": len(self.values)}
        if self.kind == "int":
            d["min"], d["max"] = self.lo, self.hi
        return d


def split(path: str) -> tuple[str, list[tuple[str, str]]]:
    """The template key of a path (fields marked) and its field values with their kinds."""
    fields = []

    def mark(m):
        text = m.group(0)
        kind = "hex" if len(text) >= 16 and not text.isdigit() else "int"
        fields.append((kind, text))
        return _MARK

    return _FIELD.sub(mark, path), fields


class Template:
    def __init__(self, tid: int, key: str, kinds: list[str]):
        self.id = tid
        self.key = key
        self.fields = [Field(k) for k in kinds]
        self.paths: set[str] = set()
        self.opens = 0
        self.ops: Counter = Counter()
        self.bytes = [0, 0]
        self.size = [LogHist(), LogHist()]
        self.offsets: Counter = Counter()
        self.runs = LogHist()
        self.touches: list = []     # (t, path, oid) of every open, for the site

    def add_path(self, path: str, values: list[tuple[str, str]]):
        self.paths.add(path)
        for f, (_, text) in zip(self.fields, values):
            f.add(text)

    def name(self) -> str:
        parts = self.key.split(_MARK)
        out = [parts[0]]
        for i, f in enumerate(self.fields):
            out.append(f.placeholder(i))
            out.append(parts[i + 1])
        return "".join(out)

    def out(self, raw: bool) -> dict:
        d = {
            "id": self.id,
            "template": self.name(),
            "paths": len(self.paths),
            "fields": [f.out(i) for i, f in enumerate(self.fields)],
            "opens": self.opens,
            "ops": dict(sorted(self.ops.items(), key=lambda kv: -kv[1])),
            "bytes_read": self.bytes[0],
            "bytes_written": self.bytes[1],
            "read_size": self.size[0].out(),
            "write_size": self.size[1].out(),
            "offsets": dict(sorted(self.offsets.items())),
            "runs_per_open": self.runs.out(),
        }
        if raw:
            d["raw_paths"] = sorted(self.paths)
        return d


# ---------------------------------------------------------------- units, motifs, phases


def compress(ops: list[str]) -> list[tuple[str, int]]:
    """The op sequence as groups: the shortest cycle of up to `MAX_PERIOD` ops that repeats at a
    position is one group with its count; anything else a group of one."""
    out, i, n = [], 0, len(ops)
    while i < n:
        hit = None
        for L in range(1, MAX_PERIOD + 1):
            if i + 2 * L > n:
                break
            unit = ops[i : i + L]
            r = 1
            while ops[i + r * L : i + (r + 1) * L] == unit:
                r += 1
            if r >= 2:
                hit = (L, r)
                break
        if hit:
            L, r = hit
            out.append((" ".join(ops[i : i + L]), r))
            i += L * r
        else:
            out.append((ops[i], 1))
            i += 1
    return out


def flat(groups: list[tuple[str, int]]) -> str:
    return " ".join(g for g, _ in groups)


def plus(group: str) -> str:
    return f"({group})+" if " " in group else f"{group}+"


class Unit:
    __slots__ = ("lane", "template", "ops", "first", "last", "bytes", "data", "path")

    def __init__(self, lane: int, template: int, path: str, t: int, dur: int):
        self.lane = lane
        self.template = template
        self.path = path
        self.ops: list[str] = []
        self.first = t
        self.last = t + dur
        self.bytes = [0, 0]
        self.data: list[tuple[int, int, int]] = []    # (offset, requested, moved) in order

    def op(self, name: str, t: int, dur: int):
        self.ops.append(name)
        self.last = max(self.last, t + dur)


class Phase:
    def __init__(self, key: str, template: int):
        self.key = key
        self.template = template
        self.units = 0
        self.ops = 0
        self.bytes = [0, 0]
        self.repeats: dict[str, LogHist] = {}
        self.think = LogHist()
        self.first = None
        self.last = None
        self.groups: list[tuple[str, int]] = []
        self.rich = -1

    def add(self, u: Unit, groups: list[tuple[str, int]]):
        self.units += 1
        self.ops += len(u.ops)
        self.bytes[0] += u.bytes[0]
        self.bytes[1] += u.bytes[1]
        repeated = 0
        for g, r in groups:
            if r >= 2:
                repeated += 1
                self.repeats.setdefault(plus(g), LogHist()).add(r)
        if repeated > self.rich:
            self.rich, self.groups = repeated, groups
        if self.last is not None and u.first > self.last:
            self.think.add(u.first - self.last)
        self.first = u.first if self.first is None else min(self.first, u.first)
        self.last = u.last if self.last is None else max(self.last, u.last)

    def motif(self) -> str:
        return " ".join(plus(g) if plus(g) in self.repeats else g for g, _ in self.groups)

    def out(self) -> dict:
        return {
            "motif": self.motif(),
            "template": self.template,
            "units": self.units,
            "repeats": {k: h.out() for k, h in sorted(self.repeats.items())},
            "ops": self.ops,
            "bytes_read": self.bytes[0],
            "bytes_written": self.bytes[1],
            "think": self.think.out(),
            "first": self.first,
            "last": self.last,
        }


def offsets_class(data: list[tuple[int, int, int]]) -> tuple[str, int]:
    """How an open moved through its file: the class and the number of sequential runs."""
    if not data:
        return "none", 0
    runs, end = 1, None
    for off, _, moved in data:
        if end is not None and off != end:
            runs += 1
        end = off + moved
    if runs > 1:
        return "positioned", runs
    return ("whole" if data[0][0] == 0 else "sequential"), 1


# ---------------------------------------------------------------- the digester


class Digester:
    def __init__(self):
        self.templates: dict[str, Template] = {}
        self.by_id: list[Template] = []
        self.opens: dict[int, Unit] = {}
        self.pos: dict[int, int] = {}
        self.units: dict[int, list[Unit]] = {}       # lane -> units in the order they began
        self.lane_ops: Counter = Counter()
        self.ops = 0
        self.notes: Counter = Counter()

    def template(self, path: str) -> Template:
        key, values = split(path)
        t = self.templates.get(key)
        if t is None:
            t = self.templates[key] = Template(len(self.templates), key, [k for k, _ in values])
            self.by_id.append(t)
        t.add_path(path, values)
        return t

    def _unit(self, lane: int, template: Template, path: str, t: int, dur: int) -> Unit:
        u = Unit(lane, template.id, path, t, dur)
        self.units.setdefault(lane, []).append(u)
        return u

    def _data(self, e: dict, t: int, dur: int, member: Unit | None = None):
        oid = e["fd"]
        u = member or self.opens.get(oid)
        if u is None:
            self.notes["data_on_unknown_open"] += 1
            return
        kind = 0 if e["op"] == "read" else 1
        ret = e.get("ret", 0)
        n = ret if isinstance(ret, int) else 0
        if "offset" in e:
            off = e["offset"]
        else:
            off = self.pos.get(oid, 0)
            self.pos[oid] = off + n
        tpl = self.by_id[u.template]
        tpl.ops[e["op"]] += 1
        tpl.bytes[kind] += n
        tpl.size[kind].add(max(e.get("len", 0), 0))
        u.bytes[kind] += n
        u.data.append((off, e.get("len", 0), n))
        if member is None:
            u.op(e["op"], t, dur)       # a submit's members are the submit unit's ops, not the open's

    def event(self, e: dict):
        lane, op, t, dur = e["lane"], e["op"], e["t"], e.get("dur", 0)
        self.ops += 1
        self.lane_ops[lane] += 1
        if op == "open":
            tpl = self.template(e["path"])
            tpl.opens += 1
            tpl.ops["open"] += 1
            u = self._unit(lane, tpl, e["path"], t, dur)
            u.ops.append("open")
            self.opens[e["fd"]] = u
            self.pos[e["fd"]] = 0
            tpl.touches.append((t, e["path"], u))
            return
        if op in KINDS:
            self._data(e, t, dur)
            return
        if op == "submit":
            members = e.get("ops", [])
            u = Unit(lane, -1, "", t, dur)
            u.ops.append("submit")
            self.units.setdefault(lane, []).append(u)
            self.notes["submit_members"] += len(members)
            for sub in members:
                m = self.opens.get(sub["fd"])
                if m is None:
                    self.notes["data_on_unknown_open"] += 1
                    continue
                self._data(sub, t, dur, m)
                if u.template < 0:
                    u.template = m.template
            u.ops.extend(["member"] * len(members))
            return
        if op == "close":
            u = self.opens.pop(e["fd"], None)
            self.pos.pop(e["fd"], None)
            if u is None:
                self.notes["close_of_unknown_open"] += 1
                return
            u.op("close", t, dur)
            self.by_id[u.template].ops["close"] += 1
            self._close(u)
            return
        if "fd" in e:
            u = self.opens.get(e["fd"])
            if u is None:
                self.notes[f"{op}_on_unknown_open"] += 1
                return
            if op == "lseek":
                ret = e.get("ret", 0)
                if isinstance(ret, int):
                    self.pos[e["fd"]] = ret
            u.op(op, t, dur)
            self.by_id[u.template].ops[op] += 1
            return
        # a path op: a unit of one op
        tpl = self.template(e["path"])
        tpl.ops[op] += 1
        u = self._unit(lane, tpl, e["path"], t, dur)
        u.ops.append(op)
        if op == "rename" and "to" in e:
            self.template(e["to"]).ops["renamed_to"] += 1

    def _close(self, u: Unit):
        cls, runs = offsets_class(u.data)
        tpl = self.by_id[u.template]
        tpl.offsets[cls] += 1
        if runs:
            tpl.runs.add(runs)

    def finish(self):
        for oid, u in list(self.opens.items()):
            self.notes["opens_not_closed"] += 1
            self._close(u)
        self.opens = {}

    # ---- output

    def lanes_out(self) -> list[dict]:
        out = []
        for lane in sorted(self.units):
            phases: list[Phase] = []
            for u in sorted(self.units[lane], key=lambda u: u.first):
                groups = compress(u.ops)
                key = flat(groups)
                if phases and phases[-1].key == key and phases[-1].template == u.template:
                    phases[-1].add(u, groups)
                else:
                    p = Phase(key, u.template)
                    p.add(u, groups)
                    phases.append(p)
            out.append({"lane": lane, "ops": self.lane_ops[lane], "phases": [p.out() for p in phases]})
        return out


# ---------------------------------------------------------------- the verdicts


def epochs_of(paths: list[str]) -> list[list[str]]:
    """Greedy passes: a pass ends at the first path it already holds."""
    out, cur, seen = [], [], set()
    for p in paths:
        if p in seen:
            out.append(cur)
            cur, seen = [], set()
        cur.append(p)
        seen.add(p)
    if cur:
        out.append(cur)
    return out


def p_no_repeat(k: int, n: int) -> float:
    """The chance that `k` draws with replacement from `n` repeat nothing."""
    return math.exp(-k * (k - 1) / (2 * n)) if n > 0 else 0.0


def candidates_for_choose(k: int) -> int:
    """The smallest corpus from which `k` draws with replacement have at least a 5 % chance of no repeat."""
    return max(k, math.ceil(k * (k - 1) / (2 * _LN)))


def touches_to_decide(n: int) -> int:
    """The fewest draws with replacement from `n` that repeat with 95 % probability."""
    k = max(2, math.floor((1 + math.sqrt(1 + 8 * n * _LN)) / 2))   # the real root, never above the answer
    while p_no_repeat(k, n) > P_REFUTE:
        k += 1
    return k


def full_passes(epochs: list[list[str]]) -> list[list[str]]:
    """The passes the trace completed: the last one is partial (the trace ended inside it)
    unless it is as long as the first."""
    full = epochs[:-1] if len(epochs) > 1 else []
    if full and len(epochs[-1]) == len(epochs[0]):
        full = epochs
    return full


def selection_of(paths: list[str], epochs: list[list[str]], candidates: int | None, notes: Counter, tid: int) -> dict:
    k, d = len(paths), len(set(paths))
    full = full_passes(epochs)
    lengths = [len(e) for e in full]
    if candidates is not None and candidates < d:
        notes["candidates"] = f"--candidates {candidates} is fewer than the {d} distinct paths of template {tid}; the paths seen were used"
        candidates = d
    out: dict = {"verdict": "undecided", "consistent": [], "refuted": {}}
    if not full:
        # no repeat: a pass of k, or a draw from a corpus large enough not to have repeated
        out["consume"] = {"epoch": k}
        n_star = candidates_for_choose(k)
        out["candidates_for_choose"] = n_star
        if candidates is not None:
            p = p_no_repeat(k, candidates)
            out["p_no_repeat_under_choose"] = p
            if p < P_REFUTE:
                out["verdict"] = "consume"
                out["consistent"] = ["consume"]
                out["refuted"]["choose"] = f"no repeat in {k} touches of {candidates} candidates has probability {p:.2g} under a draw with replacement"
                out["settles"] = "decided"
                return out
            out["touches_to_decide"] = touches_to_decide(candidates)
            out["consistent"] = ["consume", "choose"]
            out["settles"] = (f"a second pass over the set: a repeat only after all {k} decides a permutation, an earlier one a draw; "
                              f"or {out['touches_to_decide']} touches of {candidates} candidates, after which a draw with replacement has repeated with 95 % probability")
            return out
        out["consistent"] = ["consume", "choose"]
        out["settles"] = (f"a second pass over the set (a repeat decides it), or the corpus size: fewer than {n_star} candidates "
                          f"refutes a draw with replacement (no repeat in {k} touches would have probability below {P_REFUTE})")
        return out
    L = lengths[0]
    equal = all(x == L for x in lengths)
    if len(full) >= 2 and equal and d < 1.5 * L:
        out["verdict"] = "consume"
        out["consistent"] = ["consume"]
        out["consume"] = {"epoch": L}
        if d > L:
            out["consume"]["seen_not_every_pass"] = d - L
        out["refuted"]["choose"] = (f"{len(full)} full passes of equal length {L} over {d} paths; a draw with replacement repeats after about "
                                    f"sqrt(pi N / 2) touches and its passes have unequal lengths")
        out["settles"] = "decided"
        return out
    per_path = Counter(paths)
    counts = LogHist()
    for c in per_path.values():
        counts.add(c)
    top = max(1, math.ceil(d / 10))
    share = sum(sorted(per_path.values(), reverse=True)[:top]) / k
    mean = sum(lengths) / len(lengths)
    choose = {"candidates_implied": round(2 * mean * mean / math.pi), "touches_per_path": counts.out(), "top_10pct_share": share}
    if len(full) == 1:
        out["consume"] = {"epoch": L}
        out["choose"] = choose
        out["consistent"] = ["consume", "choose"]
        out["settles"] = (f"a second full pass: the same length {L} again is a permutation over the set; an earlier repeat, a draw with replacement "
                          f"from about {choose['candidates_implied']} candidates")
        return out
    out["verdict"] = "choose"
    out["consistent"] = ["choose"]
    out["choose"] = choose
    why = f"passes of unequal length {min(lengths)}-{max(lengths)}" if not equal else f"passes of equal length {L} but {d} distinct paths, more than a set of {L}"
    out["refuted"]["consume"] = f"{why}: a permutation pass over the set touches each path once"
    out["settles"] = "decided"
    return out


def order_of(epochs: list[list[str]], ranks: dict[str, int]) -> dict:
    pairs = asc = one = 0
    for e in epochs:
        for a, b in zip(e, e[1:]):
            pairs += 1
            delta = ranks[b] - ranks[a]
            asc += delta > 0
            one += delta == 1
    out = {"pairs": pairs, "ascending": asc / pairs if pairs else None, "step_of_one": one / pairs if pairs else None}
    if pairs < 8:
        out["verdict"] = "undecided"
        out["settles"] = "at least 8 consecutive touches within one pass"
    elif out["step_of_one"] >= 0.9:
        out["verdict"] = "sequential"
    elif abs(out["ascending"] - 0.5) <= 3 * math.sqrt(0.25 / pairs) and out["step_of_one"] < 0.2:
        out["verdict"] = "permuted"
    else:
        out["verdict"] = "other"
        out["note"] = "neither a scan (steps of one) nor a permutation (half the pairs ascending): interleaved scans, or a sort the trace does not show"
    return out


def size_of(touches: list[tuple[int, str, Unit]]) -> dict:
    totals = [u.bytes[0] + u.bytes[1] for _, _, u in touches]
    if len(set(totals)) == 1:
        return {"verdict": "constant", "bytes": totals[0]}
    by_path: dict[str, set[int]] = {}
    for (_, p, u), b in zip(touches, totals):
        by_path.setdefault(p, set()).add(b)
    repeated = Counter(p for _, p, _ in touches)
    repeated = [p for p, c in repeated.items() if c > 1]
    if not repeated:
        return {"verdict": "undecided", "settles": "a repeat of a path: the same bytes again is a size per path, other bytes a draw per open"}
    differing = sum(1 for p in repeated if len(by_path[p]) > 1)
    if differing:
        return {"verdict": "drawn", "paths_with_differing_bytes": differing, "repeated_paths": len(repeated)}
    return {"verdict": "per-path", "repeated_paths": len(repeated)}


def paths_of(tpl: Template) -> list[str]:
    """The paths of a template's opens, in time order."""
    return [p for _, p, _ in sorted(tpl.touches, key=lambda x: x[0])]


def site_of(tpl: Template, candidates: int | None, notes: Counter) -> dict:
    touches = sorted(tpl.touches, key=lambda x: x[0])
    paths = [p for _, p, _ in touches]
    epochs = epochs_of(paths)
    ranks = {p: i for i, p in enumerate(sorted(set(paths)))}
    return {
        "template": tpl.id,
        "touches": len(paths),
        "distinct": len(ranks),
        "epochs": [len(e) for e in epochs],
        "selection": selection_of(paths, epochs, candidates, notes, tpl.id),
        "order": order_of(epochs, ranks),
        "size": size_of(touches),
    }


ASK = {
    "selection": "Does the application revisit a sample within one pass over the set (a draw with replacement), or touch each exactly once per pass (a permutation)?",
    "size": "Is the number of bytes read from a file a property of the file (its size), or chosen per access?",
    "order": "Within a pass, does the application take the files in their sorted order, or in a shuffled one?",
}


CANDIDATES = {"selection": ["consume", "choose"], "size": ["per-path", "drawn"], "order": ["sequential", "permuted"]}


def open_of(sites: list[dict], epochs: int) -> list[dict]:
    """The undecided verdicts as questions; a site's `projection` (its passes at scale, a fact
    the owner gave) overrides `epochs` for that site."""
    items = []
    for s in sites:
        sel = s["selection"]
        epochs = s.get("projection", {}).get("passes", epochs)
        if sel["verdict"] == "undecided":
            e = epochs
            never = math.exp(-e)
            items.append({
                "site": s["template"], "question": "selection", "candidates": ["consume", "choose"],
                "ask": ASK["selection"], "settles": sel["settles"], "divergence": never,
                "at_scale": {
                    "epochs": e,
                    "measure": "share of the set never read; share of touches that re-read a path",
                    "consume": {"never_read_share": 0.0, "repeat_share": (e - 1) / e},
                    "choose": {"never_read_share": never, "repeat_share": 1 - (1 - never) / e},
                },
            })
        if s["size"]["verdict"] == "undecided":
            items.append({
                "site": s["template"], "question": "size", "candidates": ["per-path", "drawn"],
                "ask": ASK["size"], "settles": s["size"]["settles"], "divergence": 0.0,
                "at_scale": {"epochs": epochs, "note": "the op multiset is the same in distribution; which file gets which bytes differs"},
            })
        if s["order"]["verdict"] == "undecided":
            items.append({
                "site": s["template"], "question": "order", "candidates": ["sequential", "permuted"],
                "ask": ASK["order"], "settles": s["order"]["settles"], "divergence": 0.0,
                "at_scale": {"epochs": epochs, "note": "the op multiset is the same; the order within a pass differs, which read-ahead sees"},
            })
    return ranked(items)


def ranked(items: list[dict]) -> list[dict]:
    items.sort(key=lambda it: -it["divergence"])
    for i, it in enumerate(items, 1):
        it["rank"] = i
    return items


# ---------------------------------------------------------------- the document


def _events_of_file(first: str, lines) -> tuple[dict, list[dict]]:
    import itertools
    header, events = None, []
    for no, line in enumerate(itertools.chain([first], lines), 1):
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except ValueError as e:
            raise BuildError(f"line {no}: {e}") from None
        if not isinstance(doc, dict):
            raise BuildError(f"line {no}: not a JSON object")
        if header is None:
            if doc.get("aeiou_trace") != TRACE_FORMAT:
                raise BuildError(f"not an aeiou_trace: {TRACE_FORMAT} file")
            header = doc
        else:
            events.append(doc)
    return header, events


class Analysis:
    """A trace reduced to what the document is made from, kept so that answers can be applied
    and the document remade without reading the trace again (the interview does this twice)."""

    def __init__(self, header: dict, source: str, root: str, digester: Digester):
        self.header = header
        self.source = source
        self.root = root
        self.d = digester
        self.templates = sorted(digester.templates.values(), key=lambda t: t.id)
        self.by_name = {t.name(): t for t in self.templates}


def analyze(lines, root: str | None, *, exclude=(), cwd=None) -> Analysis:
    """Read an strace (needs `root`) or an exported trace file (carries its root; `root`, when
    given, must be it)."""
    import itertools
    it = iter(lines)
    first = next(it, "")
    rest = itertools.chain([first], it)
    if first.startswith('{"aeiou_trace"'):
        header, events = _events_of_file(first, it)
        source = "trace"
        if root is not None and os.path.abspath(root) != header["root"]:
            raise BuildError(f"an exported trace file carries its root ({header['root']}); --root {root} is not it")
        root = header["root"]
        if not events:
            raise BuildError("a trace file with no call in it")
    else:
        if root is None:
            raise BuildError("an strace needs --root")
        seen: dict = {}
        header, events = export_of(rest, os.path.abspath(root), seen=seen, exclude=list(exclude), cwd=cwd)
        source = "strace"
        if not events:
            raise BuildError(nothing_counted([header["root"]], seen))
        root = header["root"]
    d = Digester()
    for e in events:
        d.event(e)
    d.finish()
    return Analysis(header, source, root, d)


def document(an: Analysis, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, answers=(), by: str | None = None, at: str | None = None) -> dict:
    """The digest document, with `answers` (facts and answers to open questions, §3.88) checked
    against the trace and recorded under `asserted` with `by` and `at`, or under `rejected`
    with why; the verdicts stay the trace's."""
    notes = Counter()
    for k, v in an.header.get("notes", {}).items():
        notes[k] = v
    for k, v in an.d.notes.items():
        notes[k] = v
    asserted: list[dict] = []
    rejected: list[tuple[int, dict]] = []     # (the answer's index, so rejections keep the answers' order)
    facts: dict[int, dict] = {}
    questions = []
    index = {id(a): i for i, a in enumerate(answers)}

    def reject(a: dict, why: str):
        rejected.append((index[id(a)], {**a, "why": why}))

    def record(tpl: Template, a: dict, kind: str, value):
        e = {"site": tpl.id, "template": tpl.name(), kind: a[kind], ("value" if kind == "fact" else "answer"): value}
        if a.get("note"):
            e["note"] = a["note"]
        e["by"], e["at"] = by, at
        asserted.append(e)
        return e

    for a in answers:
        tpl = an.by_name.get(a.get("template"))
        if tpl is None:
            reject(a, "no such template in the trace")
        elif "fact" in a:
            v = a.get("value")
            if a["fact"] not in FACTS:
                reject(a, f"no such fact: {', '.join(FACTS)}")
            elif not isinstance(v, int) or isinstance(v, bool) or v < 1:
                reject(a, "not a whole number of at least 1")
            elif (why := fact_check(tpl, a["fact"], v)):
                reject(a, why)
            else:
                facts.setdefault(tpl.id, {})[a["fact"]] = v
                record(tpl, a, "fact", v)
        elif "question" in a:
            questions.append((tpl, a))
        else:
            reject(a, "neither a fact nor a question")

    sites = []
    for t in an.templates:
        if t.opens < 2:
            continue
        f = facts.get(t.id, {})
        c, p = f.get("candidates", candidates), f.get("passes", epochs)
        s = site_of(t, c, notes)
        s["projection"] = {"candidates": c, "passes": p, "asserted": sorted(f)}
        sites.append(s)
    by_tid = {s["template"]: s for s in sites}

    for tpl, a in questions:
        q, ans = a["question"], a.get("answer")
        s = by_tid.get(tpl.id)
        if q not in ASK:
            reject(a, f"no such question: {', '.join(ASK)}")
        elif s is None:
            reject(a, "not a site: the template was opened once")
        elif ans not in CANDIDATES[q] and ans != UNKNOWN:
            reject(a, f"not one of {', '.join(CANDIDATES[q])}, {UNKNOWN}")
        elif s[q]["verdict"] != "undecided":
            if ans in (s[q]["verdict"], UNKNOWN):
                notes["redundant_answers"] += 1
            else:
                reject(a, f"the trace decided {s[q]['verdict']}: {evidence(q, s[q], ans)}")
        else:
            e = record(tpl, a, "question", ans)
            s[q]["asserted"] = {k: e[k] for k in ("answer", "by", "at", "note") if k in e}

    open_items = []
    for it in open_of(sites, epochs):
        asserted_answer = by_tid[it["site"]][it["question"]].get("asserted", {}).get("answer")
        if asserted_answer is None:
            open_items.append(it)
        elif asserted_answer == UNKNOWN:
            it["answered"] = UNKNOWN
            open_items.append(it)
    rejected.sort(key=lambda r: r[0])
    rejected = [r for _, r in rejected]
    if rejected:
        notes["rejected_answers"] = len(rejected)
    return {
        "aeiou_digest": FORMAT,
        "source": an.source,
        "root": an.root,
        "ops": an.d.ops,
        "contexts": len(an.d.lane_ops),
        "projection": {"epochs": epochs, "candidates": candidates},
        "notes": dict(sorted(notes.items())),
        "templates": [t.out(paths) for t in an.templates],
        "lanes": an.d.lanes_out(),
        "sites": sites,
        "open": ranked(open_items),
        "asserted": asserted,
        "rejected": rejected,
    }


def digest_of(lines, root: str | None, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, exclude=(), cwd=None, answers=(), by=None, at=None) -> dict:
    """`analyze` then `document`, in one call."""
    return document(analyze(lines, root, exclude=exclude, cwd=cwd), epochs=epochs, candidates=candidates, paths=paths, answers=answers, by=by, at=at)


# ---------------------------------------------------------------- the questions (§3.88)


UNKNOWN = "unknown"
ANSWERS_FORMAT = 1
FACTS = {
    "candidates": "How many files of this kind does the application have at the scale it runs?",
    "passes": "How many passes over them will the benchmark run?",
}
LABELS = {
    "consume": "each file exactly once per pass (a permutation over the set)",
    "choose": "a draw with replacement: a file may come up again within a pass",
    "per-path": "the bytes are a property of the file (its size)",
    "drawn": "the bytes are chosen per access",
    "sequential": "in their sorted order",
    "permuted": "in a shuffled order",
}


def fact_check(tpl: Template, fact: str, value: int) -> str | None:
    """Why the trace contradicts a fact, or None."""
    paths = paths_of(tpl)
    if fact == "candidates":
        d = len(set(paths))
        if value < d:
            return f"the trace touched {d} distinct paths of this template; {value} is fewer"
    else:
        n = len(full_passes(epochs_of(paths)))
        if value < n:
            return f"the trace shows {n} full passes over the set; {value} is fewer"
    return None


def evidence(q: str, v: dict, against: str) -> str:
    """What decided a verdict, said against the candidate an answer named."""
    if q == "selection":
        return v["refuted"].get(against) or f"passes {v.get('consume', {}).get('epoch', '')}".strip()
    if q == "order":
        return f"ascending {v['ascending']:.2f} of {v['pairs']} adjacent pairs, step of one {v['step_of_one']:.2f}"
    if v["verdict"] == "constant":
        return f"every open moved {v['bytes']} B"
    if v["verdict"] == "per-path":
        return f"{v['repeated_paths']} repeated path(s) always moved the same bytes"
    return f"{v['paths_with_differing_bytes']} of {v['repeated_paths']} repeated path(s) moved differing bytes"


class Console:
    """The terminal: questions on stderr (stdout may be the document), answers read from stdin."""

    def say(self, text: str) -> None:
        import sys
        sys.stderr.write(text + "\n")

    def ask(self, prompt: str) -> str | None:
        import sys
        sys.stderr.write(prompt)
        sys.stderr.flush()
        line = sys.stdin.readline()
        return None if line == "" else line.rstrip("\n")


def interview(an: Analysis, io, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, by: str, at: str) -> tuple[dict, list[dict]]:
    """Ask the facts of every site with an open question, remake the document, then ask the
    open items in rank order, each checked against the trace and re-asked when it contradicts
    it; end of input stops the asking. Returns the document and the answers given, the list
    `write_answers` keeps and `document` replays."""
    kw = dict(epochs=epochs, candidates=candidates, paths=paths, by=by, at=at)
    answers: list[dict] = []

    def finish():
        return document(an, answers=answers, **kw), answers

    doc = document(an, **kw)
    open_sites = {it["site"] for it in doc["open"]}
    for s in doc["sites"]:
        if s["template"] not in open_sites:
            continue
        tpl = an.templates[s["template"]]
        io.say("")
        io.say(f"site {s['template']}  {tpl.name()}: {s['distinct']} distinct files touched in {s['touches']} opens, passes {s['epochs']}")
        for fact, question in FACTS.items():
            io.say(f"  {question}")
            while True:
                line = io.ask("number (Enter for don't know): ")
                if line is None:
                    return finish()
                line = line.strip()
                if not line:
                    break
                if not line.isdigit() or int(line) < 1:
                    io.say("  a whole number of at least 1, or Enter for don't know")
                    continue
                why = fact_check(tpl, fact, int(line))
                if why:
                    io.say(f"  {why}")
                    continue
                answers.append({"template": tpl.name(), "fact": fact, "value": int(line)})
                break
    doc = document(an, answers=answers, **kw)
    for it in doc["open"]:
        s = next(x for x in doc["sites"] if x["template"] == it["site"])
        tpl = an.templates[it["site"]]
        cands = it["candidates"]
        io.say("")
        io.say(f"site {it['site']}  {tpl.name()}: {s['touches']} touches of {s['distinct']} paths, passes {s['epochs']}; {it['question']} undecided")
        io.say(f"  {it['ask']}")
        for i, c in enumerate(cands, 1):
            io.say(f"    {i}. {c}: {LABELS[c]}")
        io.say(f"    {len(cands) + 1}. don't know")
        sc = it["at_scale"]
        if "choose" in sc:
            io.say(f"  at {sc['epochs']} pass(es): a draw with replacement never reads {sc['choose']['never_read_share']:.1%} of the set and re-reads "
                   f"{sc['choose']['repeat_share']:.1%} of what it touches; a permutation reads every file each pass")
        else:
            io.say(f"  at {sc['epochs']} pass(es): {sc['note']}")
        io.say(f"  settles: {it['settles']}")
        while True:
            line = io.ask(f"answer [1-{len(cands) + 1}]: ")
            if line is None:
                return finish()
            line = line.strip()
            if line.isdigit() and 1 <= int(line) <= len(cands) + 1:
                break
            io.say(f"  a number from 1 to {len(cands) + 1}")
        ans = cands[int(line) - 1] if int(line) <= len(cands) else UNKNOWN
        a = {"template": tpl.name(), "question": it["question"], "answer": ans}
        answers.append(a)
        note = io.ask("how do you know? (Enter to skip): ")
        if note is None:
            return finish()
        if note.strip():
            a["note"] = note.strip()
    return finish()


def write_answers(path: str, answers: list[dict], *, trace: str, trace_sha256: str, by: str, at: str) -> None:
    doc = {"aeiou_answers": ANSWERS_FORMAT, "trace": trace, "trace_sha256": trace_sha256, "by": by, "at": at, "answers": list(answers)}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2) + "\n")


def read_answers(path: str) -> tuple[dict, list[dict]]:
    """The answers file: its header (trace, hash, by, at) and its answers."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        raise BuildError(f"{path}: {e.strerror}") from None
    except ValueError as e:
        raise BuildError(f"{path}: {e}") from None
    if not isinstance(doc, dict) or doc.get("aeiou_answers") != ANSWERS_FORMAT:
        raise BuildError(f"{path}: not an aeiou_answers: {ANSWERS_FORMAT} file")
    answers = doc.get("answers")
    if not isinstance(answers, list) or not all(isinstance(a, dict) for a in answers):
        raise BuildError(f"{path}: answers must be a list of objects")
    return doc, answers


# ---------------------------------------------------------------- the text


def _bytes(n: int) -> str:
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def _ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.1f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.0f} us"
    return f"{ns:.0f} ns"


def _p50(h: dict) -> int:
    half, acc = h["n"] / 2, 0
    for b, w in h["buckets"]:
        acc += w
        if acc >= half:
            return b
    return h["max"]


def render(doc: dict) -> str:
    out = []
    what = doc.get("trace") or doc["root"]
    head = (f"digest of {what}: {doc['ops']} ops on {doc['contexts']} lane(s) under {doc['root']}; "
            f"{len(doc['templates'])} path template(s), {len(doc['sites'])} site(s), {len(doc['open'])} open question(s)")
    asserted, rejected = doc.get("asserted", []), doc.get("rejected", [])
    if asserted:
        head += f", {len(asserted)} asserted"
    if rejected:
        head += f", {len(rejected)} rejected"
    out.append(head)
    out.append("templates:")
    for t in doc["templates"]:
        ops = ", ".join(f"{k} {v}" for k, v in t["ops"].items())
        line = f"  {t['id']:>3}  {t['template']}  {t['paths']} paths, {t['opens']} opens; ops {ops}"
        if t["bytes_read"]:
            line += f"; read {_bytes(t['bytes_read'])} (request p50 {_bytes(_p50(t['read_size']))})"
        if t["bytes_written"]:
            line += f"; wrote {_bytes(t['bytes_written'])} (request p50 {_bytes(_p50(t['write_size']))})"
        if t["offsets"]:
            line += "; offsets " + ", ".join(f"{k} {v}" for k, v in t["offsets"].items())
        out.append(line)
        for f in t["fields"]:
            rng = f" {f['min']}..{f['max']}" if f["kind"] == "int" else ""
            width = f"width {f['width']}" if f["width"] else "no fixed width"
            out.append(f"       {f['name']}: {f['kind']}, {width}, {f['values']} value(s){rng}")
        if "raw_paths" in t:
            for p in t["raw_paths"]:
                out.append(f"       {p}")
    out.append("lanes:")
    for lane in doc["lanes"]:
        out.append(f"  lane {lane['lane']}: {lane['ops']} ops")
        for i, p in enumerate(lane["phases"], 1):
            line = f"    phase {i}  {p['motif']} x{p['units']} on template {p['template']}"
            if p["repeats"]:
                line += "; repeats " + ", ".join(f"{k} p50 {_p50(h)} max {h['max']}" for k, h in p["repeats"].items())
            if p["think"]["n"]:
                line += f"; think p50 {_ns(_p50(p['think']))}"
            if p["bytes_read"]:
                line += f"; read {_bytes(p['bytes_read'])}"
            if p["bytes_written"]:
                line += f"; wrote {_bytes(p['bytes_written'])}"
            out.append(line)
    out.append("sites:")
    for s in doc["sites"]:
        line = f"  site {s['template']} (template {s['template']}): {s['touches']} touches of {s['distinct']} paths, passes {s['epochs']}"
        pr = s.get("projection")
        if pr and pr["asserted"]:
            line += f"; projected at {pr['passes']} pass(es)" + (f", {pr['candidates']} candidates" if pr["candidates"] else "") + f" (asserted: {', '.join(pr['asserted'])})"
        out.append(line)
        sel = s["selection"]
        line = f"    selection: {sel['verdict']}"
        if sel["verdict"] == "undecided":
            line += " - " + " or ".join(sel["consistent"])
            if "candidates_for_choose" in sel:
                line += f" (a draw needs a corpus of at least {sel['candidates_for_choose']})"
        for k, why in sel["refuted"].items():
            line += f"; {k} refuted: {why}"
        if sel.get("settles") and sel["settles"] != "decided":
            line += f"; settles: {sel['settles']}"
        out.append(line + _asserted(sel))
        o = s["order"]
        line = f"    order: {o['verdict']}"
        if o["pairs"]:
            line += f" (ascending {o['ascending']:.2f} of {o['pairs']} pairs, step of one {o['step_of_one']:.2f})"
        if o.get("note"):
            line += f"; {o['note']}"
        out.append(line + _asserted(o))
        z = s["size"]
        line = f"    size: {z['verdict']}"
        if z["verdict"] == "constant":
            line += f" {z['bytes']} B"
        elif z["verdict"] == "undecided":
            line += f"; settles: {z['settles']}"
        out.append(line + _asserted(z))
    pr = doc["projection"]
    out.append(f"open questions (projected over {pr['epochs']} pass(es)" + (f", {pr['candidates']} candidates" if pr["candidates"] else "") + "):")
    if not doc["open"]:
        out.append("  none: the trace decides every verdict")
    for it in doc["open"]:
        line = f"  {it['rank']}. site {it['site']} {it['question']}: {' | '.join(it['candidates'])}; divergence {it['divergence']:.3f}"
        if "answered" in it:
            line += f"; answered: {it['answered']}"
        out.append(line)
        out.append(f"     ask: {it['ask']}")
        out.append(f"     settles: {it['settles']}")
        sc = it["at_scale"]
        if "choose" in sc:
            out.append(f"     at {sc['epochs']} pass(es): never read {sc['choose']['never_read_share']:.3f} under choose, {sc['consume']['never_read_share']:.3f} under consume; "
                       f"re-reads {sc['choose']['repeat_share']:.3f} and {sc['consume']['repeat_share']:.3f} of the touches")
        elif "note" in sc:
            out.append(f"     at scale: {sc['note']}")
    if asserted:
        out.append(f"asserted (by {asserted[0]['by']}, {asserted[0]['at']}):")
        for a in asserted:
            kind = "fact" if "fact" in a else "question"
            line = f"  site {a['site']} {a[kind]}: {a['value'] if kind == 'fact' else a['answer']}"
            if a.get("note"):
                line += f" ({a['note']})"
            out.append(line)
    if rejected:
        out.append("rejected:")
        for r in rejected:
            what = f"{r['fact']} {r.get('value')}" if "fact" in r else f"{r.get('question')} {r.get('answer')}"
            out.append(f"  {r.get('template')} {what}: {r['why']}")
    return "\n".join(out) + "\n"


def _asserted(v: dict) -> str:
    a = v.get("asserted")
    return f"; asserted {a['answer']}" if a else ""
