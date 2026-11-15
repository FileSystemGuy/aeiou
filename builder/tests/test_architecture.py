"""`ARCHITECTURE.md` §5's table and the checkers say the same thing about where a statement may
stand. The section gives every statement a set of effects and every context (an actor's body
outside a timeline, a timeline's lane, a `span`'s body, a `submit`'s body, an `on_evict` body)
the effects it allows; rules V30, V34, V35, and V36 are those two tables read together. Here the
tables are read from the document itself, every statement of the schema is placed in every
context, and `schema/check.py`'s `Check` (and, when the runner is built, `aeiou check`) must
refuse exactly the placements the tables refuse. A statement the schema gains and the document
does not classify fails, as does a ban list changed in a checker without the document."""
from __future__ import annotations

import copy
import json
import os
import pathlib
import re
import subprocess

import pytest

from aeiou.validate import _load

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
DOC = ROOT / "ARCHITECTURE.md"
_, check = _load()
KINDS = set(check.SCHEMA["$defs"]["node"]["properties"])
OPS = KINDS - check.CONTROL

# the contexts, by the words of the table's first column
CONTEXTS = {"outside": "outside a timeline", "lane": "A timeline's lane", "span": "`span`'s body",
            "submit": "`submit`'s body", "evict": "`on_evict` body"}

# a refusal for where a statement stands, in either checker's words (the two word them alike)
PLACEMENT = re.compile(r"inside a timeline|outside a timeline|in a `submit` body|in an `on_evict` body|in span `"
                       r"|inside span `|inside timeline `|no timeline `tl` encloses|no timeline declares")

F = {"file": {"dataset": "d", "id": 1}}
O = {"object": {"namespace": "o", "fields": {"k": 0}}}
DIR = {"dir": {"dataset": "d", "id": 0}}
SEQ = {"shared": {"id": 1, "n": 2}}
STAT = {"stat": {"file": F}}
CACHE = {"cache": "c", "tier": "u", "seq": SEQ, "per": 1}

# one small instance of every statement of the schema; a body holds one `stat`, so a statement
# that only encloses others shows the effect `op` alone
INSTANCES = {
    "let": {"name": "x", "value": 1},
    "loop": {"index": "i", "to": 1, "body": [STAT]},
    "cond": {"if": True, "then": [STAT]},
    "choose": {"arms": [{"weight": 1, "body": [STAT]}]},
    "phase": {"name": "p", "body": [STAT]},
    "array": {"name": "arr", "count": 2, "index": "ai", "value": 1},
    "compute": {"ns": 1},
    "parallel": {"index": "k", "width": 1, "body": []},
    "channel": {"name": "ch", "capacity": 1},
    "put": {"channel": "ch", "seq": 0},
    "take": {"channel": "ch"},
    "barrier": {"scope": "host"},
    "loader": {"name": "ld", "index": "b", "workers": 1, "prefetch": 1, "batches": 1, "body": []},
    "trace": {"file": "t.jsonl", "sha256": "0" * 64},
    "timeline": {"name": "t2", "body": []},
    "publish": {"timeline": "tl", "values": {"a": 1}},
    "lookup": {**CACHE, "bind": "b"},
    "probe": {**CACHE, "from": 0, "bind": "b"},
    "touch": {**CACHE, "from": 0, "count": 1},
    "store": {**CACHE, "from": 0, "count": 1},
    "acquire": {"resource": "r", "n": 1, "hold": "h"},
    "release": {"hold": "h0"},
    "submit": {"workers": "w", "body": [STAT]},
    "span": {"name": "sp", "body": [STAT]},
    "open": {"file": F, "flags": ["RDONLY"]},
    "close": {"file": F},
    "read": {"file": F, "len": 1},
    "write": {"file": O, "len": 1},
    "lseek": {"file": F, "offset": 0, "whence": "SET"},
    "ioctl": {"file": F, "request": "TCGETS"},
    "fadvise": {"file": F, "advice": "WILLNEED"},
    "fstat": {"file": F},
    "stat": {"file": F},
    "fsync": {"file": O},
    "fdatasync": {"file": O},
    "unlink": {"file": O},
    "truncate": {"file": O, "len": 0},
    "ftruncate": {"file": O, "len": 0},
    "fallocate": {"file": O, "len": 1},
    "mkdir": {"dir": DIR},
    "rmdir": {"dir": DIR},
    "rename": {"from": O, "to": {"object": {"namespace": "o", "fields": {"k": 1}}}},
    "readdir": {"dir": DIR, "repeat": "until_eof"},
}


def tables(text: str):
    """§5 of the document: each statement's effects, and each context's allowed effects (with
    whether its `fork` is the joined one alone)."""
    sec = text.split("### 5. The verifier", 1)[1].split("\n### 6.", 1)[0]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in sec.splitlines() if ln.startswith("| `") or ln.startswith("| A")]
    effects, allows, joined_only = {k: set() for k in KINDS}, {}, {}
    names = set()
    for r in rows:
        m = re.fullmatch(r"`(\w+)`", r[0])
        if m and len(r) == 3 and not any(w in r[0] for w in CONTEXTS.values()):
            names.add(m.group(1))
            named = OPS if m.group(1) == "op" else set(re.findall(r"`(\w+)`", r[2])) & KINDS
            for k in named:
                effects[k].add(m.group(1))
    for r in rows:
        ctx = next((c for c, words in CONTEXTS.items() if words in r[0]), None)
        if ctx:
            allows[ctx] = set(re.findall(r"`(\w+)`", r[1]))
            assert allows[ctx] <= names, f"context `{r[0]}` allows an effect the first table lacks: {allows[ctx] - names}"
            joined_only[ctx] = "`fork` (joined)" in r[1]
    quiet = re.search(r"^((?:`\w+`,? (?:and )?)+)have no effect of their own", sec, re.M)
    none = set(re.findall(r"`(\w+)`", quiet.group(1)))
    return effects, allows, joined_only, none


def document(ctx: str, node: dict) -> dict:
    """An abstract with `node` in the context: a timeline with a cache, a resource, and workers,
    a hold `h0` taken for a `release` to give back."""
    ordered = {"name": "t", "order": [{"cached": "touched"}], "capacity": {"param": "n"}}
    tl = {"name": "tl", "workers": [{"name": "w", "lanes": 1, "staging": 1024}],
          "resources": [{"name": "r", "capacity": 4}],
          "caches": [{"name": "c", "tiers": [ordered, {"name": "u"}]}],
          "body": [{"publish": {"timeline": "tl", "values": {"a": 1}}}, {"acquire": {"resource": "r", "n": 1, "hold": "h0"}}]}
    if ctx == "lane":
        tl["body"].append(node)
    elif ctx == "span":
        tl["body"].append({"span": {"name": "outer", "body": [node]}})
    elif ctx == "submit":
        tl["body"].append({"submit": {"workers": "w", "body": [node]}})
    elif ctx == "evict":
        ordered["on_evict"] = {"lanes": 1, "body": [node]}
    if not (ctx == "lane" and "release" in node):
        tl["body"].append({"release": {"hold": "h0"}})
    return {"ast": check.SCHEMA["properties"]["ast"]["const"], "name": "t", "params": {"n": {"default": 3}},
            "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1}},
            "namespaces": {"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 2}},
            "actors": {"gpu": {"body": [node] if ctx == "outside" else [{"timeline": tl}]}}}


def placements():
    """Every statement, and `parallel` not joined beside the joined one."""
    for kind in sorted(KINDS):
        yield kind, {kind: copy.deepcopy(INSTANCES[kind])}
    yield "parallel join:false", {"parallel": {**INSTANCES["parallel"], "join": False}}


def by_table(name, node, ctx, effects, allows, joined_only):
    kind = next(iter(node))
    eff = set(effects[kind])
    if kind in ("submit", "span", "loop", "cond", "choose", "phase"):
        eff.add("op")                                   # the `stat` in its body
    if name == "parallel join:false" and joined_only[ctx]:
        return True
    return not eff <= allows[ctx]


def disagreements(refused_by_checker, text=None):
    effects, allows, joined_only, _ = tables(text or DOC.read_text(encoding="utf-8"))
    out = []
    for ctx in CONTEXTS:
        for name, node in placements():
            doc = document(ctx, node)
            table, code = by_table(name, node, ctx, effects, allows, joined_only), refused_by_checker(doc)
            if table != code:
                out.append(f"`{name}` in context `{ctx}`: the table {'refuses' if table else 'allows'} it, the checker {'refuses' if code else 'allows'} it")
    return out


def python_refuses(doc):
    return any(PLACEMENT.search(e) for e in check.Check(doc, "t").run())


def test_every_statement_has_an_instance_here():
    assert set(INSTANCES) == KINDS, f"statements of the schema with no instance in this test, or the reverse: {sorted(set(INSTANCES) ^ KINDS)}"


def test_the_document_classifies_every_statement():
    effects, allows, _, none = tables(DOC.read_text(encoding="utf-8"))
    assert set(allows) == set(CONTEXTS)
    unclassified = sorted(k for k in KINDS if not effects[k] and k not in none)
    assert not unclassified, f"ARCHITECTURE.md §5 gives these statements no effect and does not list them as having none: {unclassified}"
    both = sorted(k for k in none if effects[k])
    assert not both, f"ARCHITECTURE.md §5 lists these as having no effect and gives them one: {both}"


def test_the_table_and_the_python_checker_place_statements_alike():
    assert disagreements(python_refuses) == []


@pytest.mark.skipif(not RUNNER.exists(), reason="runner binary not built")
def test_the_table_and_the_runner_place_statements_alike(tmp_path):

    def runner_refuses(doc, n=[0]):
        n[0] += 1
        f = tmp_path / f"t{n[0]}.ast.json"
        f.write_text(json.dumps(doc))
        r = subprocess.run([str(RUNNER), "check", str(f)], capture_output=True, text=True)
        return bool(PLACEMENT.search(r.stdout + r.stderr))
    assert disagreements(runner_refuses) == []


@pytest.mark.parametrize("old, new, found", [
    # the span's row gaining `time`: a `compute` in a span, which V36 refuses
    ("| A `span`'s body | `op`, ", "| A `span`'s body | `op`, `time`, ", "`compute` in context `span`: the table allows it, the checker refuses it"),
    # `probe` taken out of the first table: unclassified, so allowed everywhere by the table
    ("| `read` | Reads a tier by a statement | `probe` |", "| `read` | Reads a tier by a statement | nothing |", "`probe` in context `outside`: the table allows it, the checker refuses it"),
    # the lane's row losing `write`
    ("`wait`, `read`, `write`, `fork` (joined or not)", "`wait`, `read`, `fork` (joined or not)", "`publish` in context `lane`: the table refuses it, the checker allows it"),
])
def test_a_wrong_table_is_caught(old, new, found):
    text = DOC.read_text(encoding="utf-8")
    assert text.count(old) == 1
    assert found in disagreements(python_refuses, text.replace(old, new))
