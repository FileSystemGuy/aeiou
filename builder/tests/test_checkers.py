"""The two checkers refuse alike: `schema/check.py`'s semantic rules (`Check`) and the runner's
`aeiou check` (`validate.rs`) give every refusal at the same path in the same words. Each
document here breaks one rule; `Check` judges it directly (a document the JSON schema refuses
first in check.py still reaches the runner's own copy of the rule, so both copies are held to
one wording), and, when the runner is built, `aeiou check` judges the file."""
from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess

import pytest

from aeiou.validate import _load

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
_, check = _load()

BASE = {"ast": "0.11", "name": "t",
        "params": {"n": {"default": 4}},
        "datasets": {"d": {"pattern": "d/{id}", "layout": "whole", "access": "map", "count": 8, "size": {"const": 10}, "seed": 1},
                     "arr": {"pattern": "a/v.bin", "layout": "array", "access": "indexed", "count": 4, "size": {"const": 4}, "slot": 4, "seed": 5},
                     "ix": {"pattern": "i/{id}", "layout": "whole", "access": "indexed", "count": 4, "size": {"const": 4}, "seed": 6}},
        "namespaces": {"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 2},
                       "inp": {"pattern": "in/{k}", "fields": {"k": "int"}, "size": 8, "seed": 7, "input": True},
                       "s3o": {"pattern": "s3/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 8, "protocol": "s3"}},
        "actors": {"a": {"body": [{"stat": {"file": {"pick": {"dataset": "d"}}}}]}}}

PICK = {"pick": {"dataset": "d"}}


def obj(ns, **fields):
    return {"object": {"namespace": ns, "fields": fields or {"k": 0}}}


def loop(*nodes):
    return {"loop": {"index": "i", "to": 2, "body": list(nodes)}}


def body(*nodes):
    def edit(doc):
        doc["actors"]["a"]["body"] = list(nodes)
    return edit


def ds(**kw):
    def edit(doc):
        doc["datasets"]["d"].update(kw)
    return edit


def framing(*weights):
    return ds(layout="compound", access="stream", samples_per_file=2,
              format={"class": "x", "framing": {"columns": [{"weight": w} for w in weights]}})


def extra(where, name, value):
    def edit(doc):
        doc[where][name] = value
    return edit


A = "actors/a/body/0"
CASES = {
    # datasets: V20, V21, V13, the framing (the first four the JSON schema refuses first in check.py)
    "compound without samples_per_file": (ds(layout="compound", access="stream"), "datasets/d: layout `compound` needs `samples_per_file` (V20)"),
    "whole with a chunk": (ds(chunk=4), "datasets/d/chunk: layout `whole` takes no `chunk` (V20)"),
    "a consumed array": (ds(pattern="d/x.bin", layout="array", slot=4), "datasets/d/access: an `array` is `indexed`: its slots are looked up by position (V21)"),
    "a negative column weight": (framing(-1, 2), "datasets/d/format/framing/columns: negative weight"),
    "an array with fields": (ds(layout="array", access="indexed", slot=4), "datasets/d/pattern: an `array` is one file: its pattern has no fields (V20)"),
    "chunked without k": (ds(layout="chunked", chunk=4), "datasets/d/pattern: a `chunked` dataset's pattern needs a `{k}` field (V20)"),
    "whole with k": (ds(pattern="d/{id}/{k}"), "datasets/d/pattern: `{k}` names a chunk: only a `chunked` dataset has one (V20)"),
    "no id": (ds(pattern="d/x{k}", layout="chunked", chunk=4), "datasets/d/pattern: the pattern needs an `{id}` field (V20)"),
    "consumed directories by mod": (ds(pattern="d/{id mod 4}/{id}"),
                                    "datasets/d/pattern: a consumed dataset's directories are `{id div D}` or none, never `mod`, so a subset of it is whole directories (V21)"),
    "weights short of one": (framing(0.5), "datasets/d/format/framing/columns: weights sum to 0.5, not 1: the sample's bytes must land somewhere once"),
    "gpus declared": (extra("params", "gpus", {"default": 2}), "params: `gpus` is reserved and set by the runner"),
    "nested roots": (extra("datasets", "e", dict(BASE["datasets"]["d"], pattern="d/e/{id}", seed=3)),
                     "datasets/e: root `d/e/` and root `d/` of dataset `d` are nested (V13)"),
    "empirical lengths": (ds(size={"empirical": {"values": [1, 2], "weights": [1]}}), "datasets/d/size: empirical: values and weights differ in length"),
    "uniform64 size": (ds(size={"uniform64": {}}), "datasets/d/size: a size below 0: the distribution is unbounded below; give it a `min` of 0 or more (V25)"),
    "a mixture arm below 0": (ds(size={"mixture": [{"weight": 1, "dist": {"zipf": {"s": 1.1}}}, {"weight": 1, "dist": {"const": -3}}]}),
                              "datasets/d/size: a size below 0: it can be -3 (V25)"),
    # namespaces: V13, V14, V15, V17
    "a namespace inside a dataset": (extra("namespaces", "q", {"pattern": "d/q/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 9}),
                                     "namespaces/q: root `d/q/` lies inside dataset `d`'s root `d/` (V13)"),
    "input differs on one root": (extra("namespaces", "q", {"pattern": "o/q{k}", "fields": {"k": "int"}, "size": 8, "seed": 9, "input": True}),
                                  "namespaces/q: shares root `o/` with namespace `o` but `input` differs (V14)"),
    "protocol differs on one root": (extra("namespaces", "q", {"pattern": "o/q{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 9, "protocol": "s3"}),
                                     "namespaces/q: shares root `o/` with namespace `o` but `protocol` differs (V17)"),
    "same_run without input": (extra("namespaces", "q", {"pattern": "q/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 9, "same_run": True}),
                               "namespaces/q: `same_run` without `input`: only an input namespace has a writer to compare with (V15)"),
    # statements
    "a put to no channel": (body({"put": {"channel": "c", "seq": 0}}), f"{A}/put: channel `c` not declared"),
    "a take from no channel": (body({"take": {"channel": "c"}}), f"{A}/take: channel `c` not declared"),
    "a write to an input namespace": (body({"write": {"file": obj("inp"), "len": 4}}), f"{A}/write/file: write on input namespace `inp`: input namespaces are read-only (V14)"),
    "an open for writing of an input namespace": (body({"open": {"file": obj("inp"), "flags": ["WRONLY"]}}, {"close": {"file": obj("inp")}}),
                                                  f"{A}/open/flags: open for writing on input namespace `inp`: input namespaces are read-only (V14)"),
    "ftruncate of an s3 object": (body({"open": {"file": obj("s3o"), "flags": ["WRONLY"]}}, {"ftruncate": {"file": obj("s3o"), "len": 0}}, {"close": {"file": obj("s3o")}}),
                                  "actors/a/body/1/ftruncate/file: ftruncate on namespace `s3o`, declared `protocol: s3`: an object is written once, in order from 0, and has no ftruncate (V18)"),
    "until_eof of an object never written": (body({"let": {"name": "f", "value": obj("o")}}, {"read": {"file": {"ref": "f"}, "len": 4, "repeat": "until_eof"}}),
                                             "actors/a/body/1/read: until_eof on an as_written object needs the handle binding the creating writes used (same position)"),
    "a write to a container": (body({"write": {"file": {"file": {"of": PICK}}, "len": 4}}), f"{A}/write/file: write on dataset `d`: datasets are read-only (V12)"),
    "a write to a unit": (body({"write": {"file": {"unit": {"of": PICK, "index": 0}}, "len": 4}}), f"{A}/write/file: write on dataset `d`: datasets are read-only (V12)"),
    # expressions
    "an unknown param": (body({"compute": {"ns": {"param": "zz"}}}), f"{A}/compute/ns: unknown param `zz`"),
    "an index out of scope": (body({"compute": {"ns": {"index": "i"}}}), f"{A}/compute/ns: index `i` not in scope"),
    "elem of an unknown array": (body({"compute": {"ns": {"elem": {"array": "zz", "index": 0}}}}), f"{A}/compute/ns: unknown parameter array `zz`"),
    "len of an unknown array": (body({"compute": {"ns": {"len": "zz"}}}), f"{A}/compute/ns: unknown parameter array `zz`"),
    "count of an unknown dataset": (body({"compute": {"ns": {"count": "zz"}}}), f"{A}/compute/ns: unknown dataset `zz`"),
    "dirs of an array": (body({"compute": {"ns": {"dirs": "arr"}}}), f"{A}/compute/ns: `dirs` needs a dataset named by ids, not an `array`: got `arr`"),
    "dirs of an unknown dataset": (body({"compute": {"ns": {"dirs": "zz"}}}), f"{A}/compute/ns: `dirs` needs a dataset named by ids, not an `array`: got `zz`"),
    "at of an unknown binding": (body(loop({"let": {"name": "x", "value": {"at": {"ref": "y", "index": {"sub": [{"index": "i"}, 1]}}}}})),
                                 f"{A}/loop/body/0/let/value: `at` names unknown binding `y`"),
    "at outside a loop": (body({"let": {"name": "x", "value": {"at": {"ref": "x", "index": 0}}}}), f"{A}/let/value: `at` outside any loop"),
    # handles
    "a handle binding out of scope": (body({"stat": {"file": {"ref": "nope"}}}), f"{A}/stat/file: handle binding `nope` not in scope"),
    "a file of a null id": (body({"stat": {"file": {"file": {"dataset": "d", "id": None}}}}), f"{A}/stat/file: file handle needs either {{dataset, id}} or {{of}}"),
    "a file of an unknown dataset": (body({"stat": {"file": {"file": {"dataset": "zz", "id": 0}}}}), f"{A}/stat/file: unknown dataset `zz`"),
    "an inventory of an unknown dataset": (body({"stat": {"file": {"inventory": "zz"}}}), f"{A}/stat/file: unknown dataset `zz`"),
    "a dir of an array": (body({"readdir": {"dir": {"dir": {"dataset": "arr", "id": 0}}, "repeat": "until_end"}}),
                          f"{A}/readdir/dir: `dir` needs a dataset named by ids, not an `array`: got `arr`"),
    "an object of an unknown namespace": (body({"stat": {"file": obj("zz")}}), f"{A}/stat/file: unknown namespace `zz`"),
    "an object of other fields": (body({"stat": {"file": obj("o", j=0, a=1)}}), f'{A}/stat/file: object fields ["a", "j"] != namespace fields ["k"]'),
    "a consume of an unknown dataset": (body(loop({"stat": {"file": {"consume": "zz"}}})), f"{A}/loop/body/0/stat/file: unknown dataset `zz`"),
    "a consume of an indexed dataset": (body(loop({"stat": {"file": {"consume": "ix"}}})),
                                        f"{A}/loop/body/0/stat/file: `consume` from `ix`, which is `indexed`: its items are looked up by id (`file`, `pick`), never drawn from the shuffle (V21)"),
    "a pick of an unknown dataset": (body({"stat": {"file": {"pick": {"dataset": "zz"}}}}), f"{A}/stat/file: unknown dataset `zz`"),
}


def _runner_errors(path: pathlib.Path) -> list[str]:
    r = subprocess.run([str(RUNNER), "check", str(path)], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    return [ln.strip() for ln in r.stdout.splitlines() if ln.startswith("    t: ")]


@pytest.mark.parametrize("name", sorted(CASES))
def test_both_checkers_refuse_at_one_path_in_one_wording(name, tmp_path):
    edit, want = CASES[name]
    doc = copy.deepcopy(BASE)
    edit(doc)
    errors = check.Check(copy.deepcopy(doc), "t").run()
    assert f"t: {want}" in errors, errors
    if RUNNER.exists():
        f = tmp_path / "t.ast.json"
        f.write_text(json.dumps(doc))
        assert _runner_errors(f) == errors


@pytest.mark.parametrize("edit", [
    body({"channel": {"name": "c", "capacity": 2}},
         {"choose": {"arms": [{"weight": 1, "body": [{"put": {"channel": "c", "seq": 0}}, {"take": {"channel": "c"}}]}]}}),
    body({"read": {"file": PICK, "len": 4, "repeat": {"param": "n"}}}),
    body(loop({"let": {"name": "x", "value": {"at": {"ref": "x", "index": {"sub": [{"index": "i"}, {"draw": {"empirical": {"values": [1, 2], "weights": [1, 1]}}}]}}}}})),
    ds(size={"mixture": [{"weight": 1, "dist": {"zipf": {"s": 1.1}}}, {"weight": 1, "dist": {"const": 3}}]}),
], ids=["channels in a choose arm", "a read repeat of an expression", "at of a draw of at least 1", "a mixture with an arm of no known least"])
def test_both_checkers_accept(edit, tmp_path):
    doc = copy.deepcopy(BASE)
    edit(doc)
    assert check.Check(copy.deepcopy(doc), "t").run() == []
    if RUNNER.exists():
        f = tmp_path / "t.ast.json"
        f.write_text(json.dumps(doc))
        r = subprocess.run([str(RUNNER), "check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0 and "ok   t.ast.json" in r.stdout, r.stdout + r.stderr


def test_a_kind_no_schema_has_is_named_by_check():
    """An expression of a kind the schema does not define (the schema refuses it before `Check`
    in check.py, and the runner's parser before its rules) is named, never a crash."""
    doc = copy.deepcopy(BASE)
    body({"compute": {"ns": {"bogus": 1}}})(doc)
    assert check.Check(doc, "t").run() == [f"t: {A}/compute/ns: unhandled expression kind `bogus`"]
