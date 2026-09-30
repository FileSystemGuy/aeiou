"""Symbolic nodes: expressions, distributions, and handles.

Every value an abstract computes at run time is a node here, never a Python number. Arithmetic
and comparison operators on nodes build more nodes (`Op`), so `step % P.sync_every == 0` is a
formula, not a boolean. A node has no truth value, cannot be iterated, and cannot be converted
to an int: Python control flow over a node is a TypeError that tells the author to use the
cursor's `loop` / `when` instead (GRAMMAR_OPTIONS.md Option D, "the one discipline").

`ast()` on any node returns the externally tagged JSON form of schema/abstract-ast.schema.json.
"""
from __future__ import annotations

import re
from typing import Any

IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class BuildError(Exception):
    """A mistake the builder can see at construction time."""


def check_ident(name: str, what: str = "identifier") -> str:
    if not isinstance(name, str) or not IDENT.match(name) or len(name) > 64:
        raise BuildError(f"{what} {name!r} must match [a-z_][a-z0-9_]* (at most 64 chars)")
    return name


# ----------------------------------------------------------------------------------------------
# lifting
# ----------------------------------------------------------------------------------------------

Literal = int | float | bool | str | None


def lift(x: Any, what: str = "value") -> Any:
    """A JSON scalar stays a scalar (integral floats become ints); a node stays a node."""
    if isinstance(x, Node):
        return x
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise BuildError(f"{what}: non-finite float")
        return int(x) if x.is_integer() else x
    if callable(x):
        raise BuildError(f"{what}: nodes hold no callables; write the formula with node arithmetic")
    raise BuildError(f"{what}: cannot use {type(x).__name__} in an abstract; expected a number, "
                     f"string, bool, None, or a node")


def ast_of(x: Any) -> Any:
    x = lift(x)
    return x.ast() if isinstance(x, Node) else x


# ----------------------------------------------------------------------------------------------
# base classes
# ----------------------------------------------------------------------------------------------

class Node:
    __slots__ = ()

    def ast(self) -> Any:
        raise NotImplementedError

    # Python control flow over a symbolic value is the error JAX raises for a traced value.
    def __bool__(self):
        raise TypeError(f"{self!r} is symbolic and has no truth value; use cursor.when(...) "
                        f"for a statement conditional or when(c, a, b) for an expression")

    def __iter__(self):
        raise TypeError(f"{self!r} is symbolic and cannot be iterated; use cursor.loop(...)")

    def __int__(self):
        raise TypeError(f"{self!r} is symbolic; it has no value at build time")

    __float__ = __index__ = __int__
    __hash__ = object.__hash__

    def __repr__(self):
        return f"<{type(self).__name__} {self.ast()!r}>"


class Expr(Node):
    """A scalar-valued positional expression."""
    __slots__ = ()

    def _bin(self, kind, other, swap=False):
        a, b = self, lift(other, kind)
        return Op(kind, (b, a) if swap else (a, b))

    def __add__(self, o): return self._bin("add", o)
    def __radd__(self, o): return self._bin("add", o, True)
    def __sub__(self, o): return self._bin("sub", o)
    def __rsub__(self, o): return self._bin("sub", o, True)
    def __mul__(self, o): return self._bin("mul", o)
    def __rmul__(self, o): return self._bin("mul", o, True)
    def __floordiv__(self, o): return self._bin("div", o)
    def __rfloordiv__(self, o): return self._bin("div", o, True)
    def __mod__(self, o): return self._bin("mod", o)
    def __rmod__(self, o): return self._bin("mod", o, True)
    def __neg__(self): return Op("neg", self)

    def __truediv__(self, o):
        raise TypeError("`/` is not defined on symbolic values: use `//` (floor) or ceil_div(a, b)")
    __rtruediv__ = __truediv__

    def __eq__(self, o): return self._bin("eq", o)          # noqa: E301
    def __ne__(self, o): return self._bin("ne", o)
    def __lt__(self, o): return self._bin("lt", o)
    def __le__(self, o): return self._bin("le", o)
    def __gt__(self, o): return self._bin("gt", o)
    def __ge__(self, o): return self._bin("ge", o)
    def __and__(self, o): return self._bin("and", o)
    def __rand__(self, o): return self._bin("and", o, True)
    def __or__(self, o): return self._bin("or", o)
    def __ror__(self, o): return self._bin("or", o, True)
    def __invert__(self): return Op("not", self)
    __hash__ = object.__hash__


class Dist(Node):
    """A distribution literal. `draw(d)` is the positional draw; a distribution is also a
    legal parameter default and a legal `size` of a dataset."""
    __slots__ = ("kind", "args")

    def __init__(self, kind: str, args: Any):
        self.kind, self.args = kind, args

    def ast(self):
        return {self.kind: _ast_dist_args(self.args)}

    def lower_bound(self) -> float | None:
        """Provable minimum of a draw, or None. Used by the `at` rule (schema/README.md V3)."""
        a = self.args
        if self.kind == "const":
            return a if isinstance(a, (int, float)) and not isinstance(a, bool) else None
        if self.kind == "uniform":
            lo = a["lo"]
            return lo if isinstance(lo, (int, float)) else None
        if self.kind in ("normal", "lognormal"):
            lo = a.get("min")
            return lo if isinstance(lo, (int, float)) else None
        if self.kind == "empirical":
            vs = a["values"]
            return min(vs) if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs) else None
        if self.kind == "mixture":
            los = [arm["dist"] for arm in a if arm["dist"] is not None]
            bounds = [lower_bound_of(d) for d in los]
            return min(bounds) if bounds and all(b is not None for b in bounds) else None
        return None


def _ast_dist_args(a):
    if isinstance(a, dict):
        return {k: _ast_dist_args(v) for k, v in a.items()}
    if isinstance(a, list):
        return [_ast_dist_args(v) for v in a]
    return ast_of(a)


def lower_bound_of(d: Any) -> float | None:
    """Lower bound of a serialized distribution (dict) or Dist."""
    if isinstance(d, Dist):
        return d.lower_bound()
    if isinstance(d, dict) and len(d) == 1:
        k, a = next(iter(d.items()))
        return Dist(k, a).lower_bound()
    return None


class Handle(Node):
    """A reference to a file, directory, sample, region, chunk object, or namespace object."""
    __slots__ = ()

    @property
    def size(self) -> Expr:
        return Meta("size", self)

    @property
    def offset(self) -> Expr:
        return Meta("offset", self)

    @property
    def unit(self) -> Expr:
        return Meta("unit", self)

    @property
    def chunks(self) -> Expr:
        return Meta("chunks", self)

    def chunk(self, k) -> "Handle":
        """Chunk object `k` of a chunk-realized file."""
        return FileOf(self, k)

    @property
    def container(self) -> "Handle":
        """The container file holding this sample (a sample handle used as a file already
        means its container; this spells it out)."""
        return FileOf(self, None)


# ----------------------------------------------------------------------------------------------
# expressions
# ----------------------------------------------------------------------------------------------

class Op(Expr):
    __slots__ = ("kind", "args")
    UNARY = {"neg", "not"}

    def __init__(self, kind: str, args):
        self.kind, self.args = kind, args

    def ast(self):
        if self.kind in self.UNARY:
            return {self.kind: ast_of(self.args)}
        return {self.kind: [ast_of(a) for a in self.args]}


class Param(Expr):
    """A reference to a named parameter. Indexing (`P.xs[t]`) is `elem`; `.len` / `.sum` are
    the array reductions."""
    __slots__ = ("name", "_default")

    def __init__(self, name: str, default: Any = None):
        self.name, self._default = check_ident(name, "parameter"), default

    def ast(self):
        return {"param": self.name}

    def __getitem__(self, i):
        return Elem(self.name, i)

    @property
    def len(self) -> Expr:
        return Reduce("len", self.name)

    @property
    def sum(self) -> Expr:
        return Reduce("sum", self.name)


class Index(Expr):
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = check_ident(name, "loop index")

    def ast(self):
        return {"index": self.name}


class Ref(Expr, Handle):
    """A let binding. It is an expression or a handle depending on what it bound."""
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = check_ident(name, "binding")

    def ast(self):
        return {"ref": self.name}

    def at(self, index) -> Expr:
        """The value this binding takes at another index of its enclosing loop (`x @ i`)."""
        return At(self.name, index)

    __hash__ = object.__hash__


class At(Expr):
    __slots__ = ("name", "index")

    def __init__(self, name: str, index):
        self.name, self.index = name, lift(index, "at index")

    def ast(self):
        return {"at": {"ref": self.name, "index": ast_of(self.index)}}


class Actor(Expr):
    __slots__ = ("which",)

    def __init__(self, which: str):
        self.which = which

    def ast(self):
        return {"actor": self.which}


class Draw(Expr):
    __slots__ = ("dist",)

    def __init__(self, dist):
        self.dist = distref(dist)

    def ast(self):
        return {"draw": ast_of(self.dist)}


class Elem(Expr):
    __slots__ = ("array", "index")

    def __init__(self, array: str, index):
        self.array, self.index = array, lift(index, "elem index")

    def ast(self):
        return {"elem": {"array": self.array, "index": ast_of(self.index)}}


class Reduce(Expr):
    __slots__ = ("kind", "array")

    def __init__(self, kind: str, array: str):
        self.kind, self.array = kind, array

    def ast(self):
        return {self.kind: self.array}


class Meta(Expr):
    """size / offset / unit / chunks of a handle."""
    __slots__ = ("kind", "handle")

    def __init__(self, kind: str, handle: Handle):
        if not isinstance(handle, Handle):
            raise BuildError(f"{kind}() needs a handle, got {handle!r}")
        self.kind, self.handle = kind, handle

    def ast(self):
        return {self.kind: self.handle.ast()}


class DatasetMeta(Expr):
    """count / dirs of a dataset."""
    __slots__ = ("kind", "dataset")

    def __init__(self, kind: str, dataset: str):
        self.kind, self.dataset = kind, dataset

    def ast(self):
        return {self.kind: self.dataset}


class Cond(Expr):
    """Expression conditional; arms may be expressions, distributions, or handles."""
    __slots__ = ("test", "then", "otherwise")

    def __init__(self, test, then, otherwise):
        self.test = lift(test, "when test")
        self.then = lift(then, "when then")
        self.otherwise = lift(otherwise, "when else")

    def ast(self):
        return {"cond": {"if": ast_of(self.test), "then": ast_of(self.then), "else": ast_of(self.otherwise)}}


def distref(d) -> Any:
    """A distribution or an expression that evaluates to one (a param, a cond)."""
    if isinstance(d, (Dist, Param, Cond)):
        return d
    raise BuildError(f"expected a distribution, a parameter holding one, or when() selecting "
                     f"one; got {d!r}")


# ----------------------------------------------------------------------------------------------
# handles
# ----------------------------------------------------------------------------------------------

class FileHandle(Handle):
    __slots__ = ("dataset", "id")

    def __init__(self, dataset: str, id):
        self.dataset, self.id = dataset, lift(id, "file id")

    def ast(self):
        return {"file": {"dataset": self.dataset, "id": ast_of(self.id)}}


class FileOf(Handle):
    __slots__ = ("of", "k")

    def __init__(self, of: Handle, k):
        if not isinstance(of, Handle):
            raise BuildError(f"file-of needs a handle, got {of!r}")
        self.of, self.k = of, lift(k, "chunk")

    def ast(self):
        a = {"of": self.of.ast()}
        if self.k is not None:
            a["chunk"] = ast_of(self.k)
        return {"file": a}


class DirHandle(Handle):
    __slots__ = ("dataset", "id")

    def __init__(self, dataset: str, id):
        self.dataset, self.id = dataset, lift(id, "dir id")

    def ast(self):
        return {"dir": {"dataset": self.dataset, "id": ast_of(self.id)}}


class ObjectHandle(Handle):
    __slots__ = ("namespace", "fields")

    def __init__(self, namespace: str, fields: dict):
        self.namespace = namespace
        self.fields = {k: lift(v, f"field {k}") for k, v in fields.items()}

    def ast(self):
        return {"object": {"namespace": self.namespace,
                           "fields": {k: ast_of(v) for k, v in self.fields.items()}}}


class Consume(Handle):
    __slots__ = ("dataset",)

    def __init__(self, dataset: str):
        self.dataset = dataset

    def ast(self):
        return {"consume": self.dataset}


class Pick(Handle):
    __slots__ = ("dataset", "dist")

    def __init__(self, dataset: str, dist=None):
        self.dataset = dataset
        self.dist = None if dist is None else distref(dist)

    def ast(self):
        a = {"dataset": self.dataset}
        if self.dist is not None:
            a["dist"] = ast_of(self.dist)
        return {"pick": a}


# ----------------------------------------------------------------------------------------------
# walking
# ----------------------------------------------------------------------------------------------

def walk(x: Any):
    """Yield every node inside a node or serialized value (depth-first)."""
    if isinstance(x, Node):
        yield x
        for attr in _children(x):
            yield from walk(attr)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from walk(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from walk(v)


def _children(n: Node):
    for cls in type(n).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot.startswith("_"):
                continue
            v = getattr(n, slot, None)
            if isinstance(v, (Node, list, tuple, dict)):
                yield v
