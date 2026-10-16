"""The builder's refusals of API misuse: each raises `BuildError` at the call that misuses the
API, naming what was misused, so an author sees the line of the script and never a later
validation error or a Python exception from inside the builder."""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from aeiou import *  # noqa: F401,F403
from aeiou.nodes import BuildError


def _kit():
    """A workload with one dataset of each kind of lookup, a namespace, an input namespace,
    and an open actor body."""
    w = Workload("m")
    P = w.params(n=4, spf=8)
    k = SimpleNamespace(w=w, P=P)
    k.ds = w.dataset("d", pattern="d/{id}", layout="whole", access="map", count=P.n, size=const(10), seed=1)
    k.arr = w.dataset("arr", pattern="a/vectors.bin", layout="array", access="indexed", count=8, size=const(4), slot=4, seed=2)
    k.idx = w.dataset("ix", pattern="i/{id}", layout="whole", access="indexed", count=8, size=const(4), seed=3)
    k.ns = w.namespace("o", pattern="o/{k}", fields={"k": int}, size="as_written", seed=4)
    k.inp = w.namespace("in_", pattern="in/{k}", fields={"k": int}, size=8, seed=5, input=True)
    return k


DECLARATIONS = [
    ("an inventory not a dict", lambda k: k.w.dataset("e", pattern="e/{id}", layout="whole", access="map", count=1, size=const(1), seed=1,
                                                       inventory=["inv.txt"]), "dataset e: inventory is {'name': file, 'entry': bytes}"),
    ("a parameter twice", lambda k: k.w.param("n", 3), "parameter 'n' declared twice"),
    ("an unknown unit", lambda k: k.w.param("q", 3, unit="furlongs"), "parameter q: unknown unit 'furlongs'"),
    ("a formula default", lambda k: k.w.param("q", k.P.n + 1), "parameter q: a default is a scalar, a distribution, or an array of these, not a formula"),
    ("an unparsable divisor", lambda k: k.w.dataset("e", pattern="e/{id div 0}/{id}", layout="whole", access="map", count=1, size=const(1), seed=1),
     "dataset e: pattern `e/{id div 0}/{id}`"),
    ("a bad layout", lambda k: k.w.dataset("e", pattern="e/{id}", layout="flat", access="map", count=1, size=const(1), seed=1),
     "dataset e: layout must be one of whole, chunked, compound, array, got 'flat'"),
    ("a bad access", lambda k: k.w.dataset("e", pattern="e/{id}", layout="whole", access="random", count=1, size=const(1), seed=1),
     "dataset e: access must be one of map, stream, indexed, got 'random'"),
    ("a chunked dataset without chunk", lambda k: k.w.dataset("e", pattern="e/{id}/{k}", layout="chunked", access="map", count=1, size=const(1), seed=1),
     "dataset e: layout `chunked` needs `chunk` (V20)"),
    ("a whole dataset with samples_per_file", lambda k: k.w.dataset("e", pattern="e/{id}", layout="whole", access="map", count=1, size=const(1), seed=1,
                                                                     samples_per_file=2), "dataset e: layout `whole` takes no `samples_per_file` (V20)"),
    ("an array with fields", lambda k: k.w.dataset("e", pattern="e/{id}", layout="array", access="indexed", count=1, size=const(1), seed=1, slot=1),
     "dataset e: an array is one file: its pattern has no fields, got ['id'] (V20)"),
    ("a consumed array", lambda k: k.w.dataset("e", pattern="e/x.bin", layout="array", access="map", count=1, size=const(1), seed=1, slot=1),
     "dataset e: an array is `indexed`: its slots are looked up by position (V21)"),
    ("a chunked pattern without k", lambda k: k.w.dataset("e", pattern="e/{id}", layout="chunked", access="map", count=1, size=const(1), seed=1, chunk=4),
     "dataset e: a `chunked` dataset's pattern fields are ['id', 'k'], got ['id'] (V20)"),
    ("consumed directories by mod", lambda k: k.w.dataset("e", pattern="e/{id mod 4}/{id}", layout="whole", access="map", count=1, size=const(1), seed=1),
     "dataset e: a consumed dataset's directories are `{id div D}` or none, never `mod`"),
    ("a namespace field of a float", lambda k: k.w.namespace("p", pattern="p/{k}", fields={"k": float}, size="as_written", seed=1),
     "namespace p: field k must be int or str"),
    ("a namespace size over another name", lambda k: k.w.namespace("p", pattern="p/{k}", fields={"k": int}, size=k.w.field("j") * 2, seed=1),
     "namespace p: size may reference only its fields and params"),
    ("a name used twice", lambda k: k.w.namespace("d", pattern="q/{k}", fields={"k": int}, size="as_written", seed=1),
     "namespace 'd': name already used"),
    ("a format without a class", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                        samples_per_file=1, format={"reader": "x"}), "dataset e: format needs a `class`"),
    ("an unknown format key", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                     samples_per_file=1, format={"class": "x", "layout": {}}), "dataset e: unknown format key 'layout'"),
    ("a framing not a dict", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                    samples_per_file=1, format={"class": "x", "framing": []}), "dataset e: format framing must be a dict"),
    ("an unknown framing key", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                      samples_per_file=1, format={"class": "x", "framing": {"tail": 1}}), "dataset e: unknown framing key 'tail'"),
    ("no columns", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                          samples_per_file=1, format={"class": "x", "framing": {"columns": []}}), "dataset e: framing columns must be a non-empty list"),
    ("an unknown column key", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                     samples_per_file=1, format={"class": "x", "framing": {"columns": [{"width": 1}]}}),
     "dataset e: unknown column key 'width'"),
    ("a negative column weight", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                        samples_per_file=1, format={"class": "x", "framing": {"columns": [{"weight": -1}]}}),
     "dataset e: column weight must be a non-negative number"),
    ("column weights short of one", lambda k: k.w.dataset("e", pattern="e/{id}", layout="compound", access="stream", count=1, size=const(1), seed=1,
                                                           samples_per_file=1, format={"class": "x", "framing": {"columns": [{"weight": 0.5}]}}),
     "dataset e: column weights sum to 0.5, not 1"),
    ("a parameter set on P", lambda k: setattr(k.P, "n", 3), "declare parameters with Workload.param() or Workload.params()"),
    ("an undeclared parameter", lambda k: k.P.nope, "no parameter 'nope'; declared: ['n', 'spf']"),
    ("dirs() of an array", lambda k: k.arr.dirs, "dirs() on array dataset arr: an array is one file"),
    ("file(id) of an array", lambda k: k.arr.file(3), "array dataset arr has one file; call file() with no id"),
    ("file() without an id", lambda k: k.ds.file(), "file() on dataset d needs an id"),
    ("dir() of an array", lambda k: k.arr.dir(0), "dir() on array dataset arr: an array is one file"),
    ("consume() of an indexed dataset", lambda k: k.idx.consume(), "consume() on dataset ix, which is `indexed`"),
]


@pytest.mark.parametrize("what, misuse, why", DECLARATIONS, ids=[c[0] for c in DECLARATIONS])
def test_a_declaration_misused_is_refused_by_name(what, misuse, why):
    k = _kit()
    with pytest.raises((BuildError, AttributeError), match=re.escape(why)):
        misuse(k)


def _choose_none(a):
    with a.choose():
        pass


def _choose_zero(a):
    with a.choose() as c, c.arm(0):
        a.compute(1)


def _otherwise_alone(a):
    a.otherwise().__enter__()


STATEMENTS = [
    ("a duplicate errno", lambda k, a: a.stat(k.ds.pick(), expect=["ENOENT", "ENOENT"]), "expect: duplicate errno"),
    ("a duplicate flag", lambda k, a: a.open(k.ds.pick(), "RDONLY|RDONLY"), "open: duplicate flag"),
    ("no flags", lambda k, a: a.open(k.ds.pick(), ""), "open: at least one flag"),
    ("a file that is no handle", lambda k, a: a.stat("d/0"), "file: expected a handle (a binding, dataset file/dir, or namespace object), got 'd/0'"),
    ("let of a distribution", lambda k, a: a.let("x", uniform(1, 2)), "let x: bind draw(dist), not the distribution itself"),
    ("a channel twice", lambda k, a: (a.channel("c", 2), a.channel("c", 2)), "channel 'c' already declared"),
    ("a loader over a channel", lambda k, a: (a.channel("c", 2), a.loader("c", workers=1, prefetch=1, batches=1).__enter__()),
     "channel 'c' already declared"),
    ("a put to no channel", lambda k, a: a.put("c", 0), "channel 'c' not declared in this actor"),
    ("a bad barrier group", lambda k, a: a.barrier("Global!"), "barrier group"),
    ("otherwise without when", lambda k, a: _otherwise_alone(a), "otherwise() must directly follow a when() block"),
    ("choose without arms", lambda k, a: _choose_none(a), "choose() needs at least one arm"),
    ("choose of zero weight", lambda k, a: _choose_zero(a), "choose: total weight is zero"),
    ("a phase named by a number", lambda k, a: a.phase(3).__enter__(), "phase name must be a string or an expression"),
    ("a trace without its hash", lambda k, a: a.trace("t.jsonl", "abc"), "trace sha256 must be 64 hex digits"),
    ("a write to a container", lambda k, a: a.write(k.ds.pick().container, 4), "write on dataset d: datasets are read-only"),
    ("a write to an input namespace", lambda k, a: a.write(k.inp.object(k=0), 4), "write on input namespace in_: input namespaces are read-only (schema/README.md V14)"),
    ("a read repeat of a word", lambda k, a: a.read(k.ds.pick(), 4, repeat="forever"), "read repeat must be until_eof, a count, or an expression; got 'forever'"),
    ("a write until_eof", lambda k, a: a.write(k.ns.object(k=0), 4, repeat="until_eof"), "write has no until_eof; give a count"),
    ("an lseek whence", lambda k, a: a.lseek(k.ds.pick(), 0, "HOLE"), "lseek whence must be one of ('SET', 'CUR', 'END')"),
    ("an ioctl request", lambda k, a: a.ioctl(k.ds.pick(), "FIOCLEX"), "ioctl request must be one of"),
    ("an fadvise advice", lambda k, a: a.fadvise(k.ds.pick(), "SOON"), "fadvise advice must be one of"),
    ("rmdir of a dataset directory", lambda k, a: a.rmdir(k.ds.dir(0)), "rmdir on dataset d: datasets are read-only"),
    ("a readdir repeat", lambda k, a: a.readdir(k.ds.dir(0), repeat="once"), "readdir repeat is always until_end"),
]


@pytest.mark.parametrize("what, misuse, why", STATEMENTS, ids=[c[0] for c in STATEMENTS])
def test_a_statement_misused_is_refused_by_name(what, misuse, why):
    k = _kit()
    with k.w.actor("x") as a:
        with pytest.raises(BuildError, match=re.escape(why)):
            misuse(k, a)


def test_what_the_misused_forms_build_when_used_right():
    """The forms next to each refusal, built and validated: a single errno, a parameter's
    listing, `count`, channels with put and take, a compute of a distribution, a loop step, a
    repeat inside a loop named `rep`, ftruncate and fallocate on a posix namespace, and rmdir."""
    k = _kit()
    assert dir(k.P) == ["n", "spf"]
    with k.w.actor("x") as a:
        assert a.indices == ()
        a.stat(k.ds.pick(), expect="ENOENT")
        a.channel("c", 2)
        with a.loop("i", k.ds.count, step=2):
            a.put("c", a.index)
            a.take("c")
            a.compute(uniform(1, 2))
        with a.loop("rep", 2):
            a.read(k.ds.pick(), 4, repeat=3)
        f = a.let("f", k.ns.object(k=0))
        a.open(f, "WRONLY|CREAT")
        a.write(f, 8)
        a.ftruncate(f, 4)
        a.fallocate(f, 8, offset=0)
        a.close(f)
        a.mkdir(k.ns.object(k=1))
        a.rmdir(k.ns.object(k=1))
    ast = k.w.build()
    body = ast["actors"]["x"]["body"]
    assert body[0] == {"stat": {"file": {"pick": {"dataset": "d"}}, "expect": ["ENOENT"]}}
    assert body[2]["loop"]["to"] == {"count": "d"} and body[2]["loop"]["step"] == 2
    assert body[2]["loop"]["body"][2] == {"compute": {"ns": {"draw": {"uniform": {"lo": 1, "hi": 2}}}}}
    # the repeat's generated index does not shadow the author's `rep`
    assert body[3]["loop"]["body"][0]["loop"]["index"] == "rep_1"
    assert [next(iter(n)) for n in body[5:]] == ["open", "write", "ftruncate", "fallocate", "close", "mkdir", "rmdir"]


def test_a_workload_without_actors_is_refused():
    w = Workload("empty")
    with pytest.raises(BuildError, match="workload empty: no actors"):
        w.build()
