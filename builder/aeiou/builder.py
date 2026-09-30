"""The Workload builder: declarations (params, datasets, namespaces, actors) and the Cursor
that appends statements to an actor body.

Python control flow runs at build time and generates structure; cursor control flow (`loop`,
`parallel`, `when`, `every`, `choose`) becomes AST nodes and runs at execution time. The
`with` blocks push a body, hand back a symbolic index or a sub-cursor, and pop on exit.
"""
from __future__ import annotations

import contextlib
import re
import sys
from typing import Any

from .nodes import (Actor, At, BuildError, Consume, Cond, DatasetMeta, DirHandle, Dist, Draw,
                    Expr, FileHandle, Handle, Index, Node, ObjectHandle, Op, Param, Pick, Ref,
                    ast_of, check_ident, distref, lift, lower_bound_of, walk)

OPEN_FLAGS = ("RDONLY", "WRONLY", "RDWR", "CREAT", "TRUNC", "EXCL", "APPEND", "CLOEXEC",
              "DIRECTORY", "DIRECT", "SYNC", "DSYNC", "NOATIME", "NOFOLLOW")
IOCTLS = ("TCGETS", "FIONREAD", "BLKGETSIZE64")
WHENCE = ("SET", "CUR", "END")
ERRNO = re.compile(r"^E[A-Z0-9]{1,15}$")
FIELD = re.compile(r"\{([a-z_][a-z0-9_]*)(?:\s+(?:div|mod)\s+\d+)?(?::[^}]*)?\}")

gpu_id = Actor("id")        # this actor's global id
actor_count = Actor("count")


def _expect(expect):
    if expect is None:
        return None
    if isinstance(expect, str):
        expect = [expect]
    expect = list(expect)
    for e in expect:
        if not isinstance(e, str) or not ERRNO.match(e):
            raise BuildError(f"expect: {e!r} is not an errno name")
    if len(set(expect)) != len(expect):
        raise BuildError("expect: duplicate errno")
    return expect or None


def _flags(flags):
    if isinstance(flags, str):
        flags = [f.strip() for f in flags.replace(",", "|").split("|") if f.strip()]
    flags = list(flags)
    for f in flags:
        if f not in OPEN_FLAGS:
            raise BuildError(f"open: unknown flag {f!r}; known: {', '.join(OPEN_FLAGS)}")
    if len(set(flags)) != len(flags):
        raise BuildError("open: duplicate flag")
    if not flags:
        raise BuildError("open: at least one flag")
    return flags


def _handle(h, what="file") -> Handle:
    if not isinstance(h, Handle):
        raise BuildError(f"{what}: expected a handle (a binding, dataset file/dir, or namespace "
                         f"object), got {h!r}")
    return h


def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# ----------------------------------------------------------------------------------------------
# declarations
# ----------------------------------------------------------------------------------------------

class Params:
    """Attribute view of a workload's parameters: `P.steps` is a `Param` node."""

    def __init__(self, wl: "Workload"):
        object.__setattr__(self, "_wl", wl)

    def __getattr__(self, name):
        try:
            return self._wl._params[name]
        except KeyError:
            raise AttributeError(f"no parameter {name!r}; declared: {sorted(self._wl._params)}") from None

    def __setattr__(self, name, value):
        raise AttributeError("declare parameters with Workload.param() or Workload.params()")

    def __dir__(self):
        return sorted(self._wl._params)


class Dataset:
    def __init__(self, name: str, kind: str, spec: dict):
        self.name, self.kind, self.spec = name, kind, spec

    def ast(self):
        return {self.kind: _strip_none(self.spec)}

    @property
    def count(self) -> Expr:
        return DatasetMeta("count", self.name)

    @property
    def dirs(self) -> Expr:
        if self.kind != "files":
            raise BuildError("dirs() needs a files dataset")
        return DatasetMeta("dirs", self.name)

    def file(self, id=None) -> Handle:
        """Dataset file by id. A `regions` dataset has one file, addressed as id 0."""
        if self.kind == "regions":
            if id is not None:
                raise BuildError(f"regions dataset {self.name} has one file; call file() with no id")
            return FileHandle(self.name, 0)
        if id is None:
            raise BuildError(f"file() on files dataset {self.name} needs an id")
        return FileHandle(self.name, id)

    def dir(self, id) -> Handle:
        if self.kind != "files":
            raise BuildError("dir() needs a files dataset")
        return DirHandle(self.name, id)

    def consume(self) -> Handle:
        """Next sample without replacement (positional; must sit inside a loop)."""
        return Consume(self.name)

    def pick(self, dist=None) -> Handle:
        """A sample with replacement, uniform unless a distribution over ids is given."""
        return Pick(self.name, dist)


class Namespace:
    def __init__(self, name: str, spec: dict, fields: dict):
        self.name, self.spec, self.fields = name, spec, fields

    def ast(self):
        return _strip_none(self.spec)

    def object(self, **fields) -> Handle:
        """The object named by filling the pattern's fields."""
        if set(fields) != set(self.fields):
            raise BuildError(f"namespace {self.name} has fields {sorted(self.fields)}, got "
                             f"{sorted(fields)}")
        return ObjectHandle(self.name, {k: fields[k] for k in self.fields})


class Workload:
    """One abstract. Declarations first, then actors; `build()` validates and returns the AST."""

    _registry: list["Workload"] = []
    AST_VERSION = "0.1"
    MAX_CANONICAL_BYTES = 4 * 1024 * 1024

    def __init__(self, name: str, doc: str | None = None, *, lint: bool = True):
        self.name = check_ident(name, "workload name")
        self.doc = doc
        self.lint = lint
        self._params: dict[str, Param] = {}
        self._param_specs: dict[str, dict] = {}
        self._datasets: dict[str, Dataset] = {}
        self._namespaces: dict[str, Namespace] = {}
        self._actors: dict[str, dict] = {}
        self.P = Params(self)
        frame = sys._getframe(1)
        self.source = frame.f_code.co_filename
        Workload._registry.append(self)

    # ---- parameters ----
    def param(self, name: str, default, *, unit: str | None = None, doc: str | None = None,
              cli: bool | None = None) -> Param:
        check_ident(name, "parameter")
        if name == "gpus":
            raise BuildError("`gpus` is reserved; the runner sets it from --gpus")
        if name in self._params:
            raise BuildError(f"parameter {name!r} declared twice")
        if unit is not None and unit not in ("bytes", "ns", "count", "ratio", "tokens", "none"):
            raise BuildError(f"parameter {name}: unknown unit {unit!r}")
        default = _param_value(default, name)
        spec = {"default": default, "unit": unit, "doc": doc, "cli": cli}
        self._param_specs[name] = spec
        self._params[name] = p = Param(name, default)
        return p

    def params(self, **defaults) -> Params:
        for k, v in defaults.items():
            self.param(k, v)
        return self.P

    # ---- datasets ----
    def dataset(self, name: str, *, pattern: str, count, size, seed: int, access: str | None = None,
                samples_per_file=None, chunk=None, format: dict | None = None,
                doc: str | None = None) -> Dataset:
        """A `files` dataset: files named by `pattern`, sizes drawn from `seed`."""
        self._declare(name, "dataset")
        fields = set(FIELD.findall(pattern))
        if not fields <= {"id", "k"} or "id" not in fields:
            raise BuildError(f"dataset {name}: pattern fields must be `id` (and `k` for chunks), "
                             f"got {sorted(fields)}")
        if (chunk is not None) != ("k" in fields):
            raise BuildError(f"dataset {name}: `chunk` and a `{{k}}` field go together")
        if access is not None and access not in ("map", "stream"):
            raise BuildError(f"dataset {name}: access must be map or stream")
        if format is not None and "class" not in format:
            raise BuildError(f"dataset {name}: format needs a `class`")
        spec = {"pattern": pattern, "count": lift(count, "count"), "size": distref(size),
                "seed": _seed(seed, name), "access": access,
                "samples_per_file": None if samples_per_file is None else lift(samples_per_file),
                "chunk": None if chunk is None else lift(chunk, "chunk"), "format": format, "doc": doc}
        self._datasets[name] = ds = Dataset(name, "files", spec)
        return ds

    def regions(self, name: str, *, file: str, count, slot, size, seed: int,
                doc: str | None = None) -> Dataset:
        """A `regions` dataset: `count` fixed-slot regions inside one file."""
        self._declare(name, "dataset")
        spec = {"file": file, "count": lift(count, "count"), "slot": lift(slot, "slot"),
                "size": distref(size), "seed": _seed(seed, name), "doc": doc}
        self._datasets[name] = ds = Dataset(name, "regions", spec)
        return ds

    # ---- namespaces ----
    def namespace(self, name: str, *, pattern: str, fields: dict, size, seed: int,
                  doc: str | None = None) -> Namespace:
        """Workload-created objects. `size` is an expression over the fields and params, or
        "as_written" (the sum of the writes that create the object; schema/README.md V4)."""
        self._declare(name, "namespace")
        fields = {k: ("int" if v in (int, "int") else "str" if v in (str, "str") else v)
                  for k, v in fields.items()}
        for k, v in fields.items():
            check_ident(k, "namespace field")
            if v not in ("int", "str"):
                raise BuildError(f"namespace {name}: field {k} must be int or str")
        pat = set(FIELD.findall(pattern))
        if pat != set(fields):
            raise BuildError(f"namespace {name}: pattern fields {sorted(pat)} != declared {sorted(fields)}")
        if not (isinstance(size, str) and size == "as_written"):
            size = lift(size, "namespace size")
            for n in walk(size):
                if isinstance(n, Ref) and n.name not in fields:
                    raise BuildError(f"namespace {name}: size may reference only its fields and params")
        spec = {"pattern": pattern, "fields": fields, "size": size, "seed": _seed(seed, name), "doc": doc}
        self._namespaces[name] = ns = Namespace(name, spec, fields)
        return ns

    def field(self, name: str) -> Ref:
        """A namespace field, for use in a namespace `size` expression."""
        return Ref(name)

    # ---- actors ----
    @contextlib.contextmanager
    def actor(self, name: str, *, count=None, doc: str | None = None):
        """An actor template. `count` defaults to the reserved `gpus` parameter."""
        self._declare(name, "actor")
        body: list = []
        self._actors[name] = _strip_none({"count": None if count is None else lift(count, "count"),
                                          "body": body, "doc": doc})
        yield Cursor(self, body)

    def _declare(self, name, what):
        check_ident(name, what)
        for table in (self._datasets, self._namespaces, self._actors):
            if name in table:
                raise BuildError(f"{what} {name!r}: name already used")

    # ---- output ----
    def build(self, validate: bool = True) -> dict:
        """The AST as a plain dict (no provenance). Validates against the schema and the
        semantic rules unless `validate=False`."""
        ast = {"ast": self.AST_VERSION, "name": self.name}
        if self.doc:
            ast["doc"] = self.doc
        if self._param_specs:
            ast["params"] = {k: _strip_none({**v, "default": _ast_value(v["default"])})
                             for k, v in self._param_specs.items()}
        if self._datasets:
            ast["datasets"] = {k: _ast_deep(v.ast()) for k, v in self._datasets.items()}
        if self._namespaces:
            ast["namespaces"] = {k: _ast_deep(v.ast()) for k, v in self._namespaces.items()}
        if not self._actors:
            raise BuildError(f"workload {self.name}: no actors")
        ast["actors"] = {k: _ast_deep(v) for k, v in self._actors.items()}
        if self.lint:
            _lint(ast)
        if validate:
            from .validate import validate as _validate
            _validate(ast)
        return ast

    def canonical(self) -> bytes:
        from .emit import canonical
        return canonical(self.build())

    def sha256(self) -> str:
        from .emit import sha256
        return sha256(self.build())

    def write(self, path=None, *, provenance: bool = True):
        """Validate, then write YAML to `path` (default `<name>.ast.yaml`). Returns the path."""
        from .emit import write
        return write(self, path, provenance=provenance)


def _seed(seed, name):
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed < 2 ** 64):
        raise BuildError(f"{name}: seed must be an integer in [0, 2^64)")
    return seed


def _param_value(v, name):
    if isinstance(v, (list, tuple)):
        return [_param_value(x, name) for x in v]
    if isinstance(v, Dist):
        return v
    v = lift(v, f"parameter {name}")
    if isinstance(v, Node):
        raise BuildError(f"parameter {name}: a default is a scalar, a distribution, or an "
                         f"array of these, not a formula")
    return v


def _ast_value(v):
    if isinstance(v, list):
        return [_ast_value(x) for x in v]
    return ast_of(v)


def _ast_deep(x):
    """Serialize a structure that may contain nodes, node lists, and plain dicts."""
    if isinstance(x, Node):
        return x.ast()
    if isinstance(x, dict):
        return {k: _ast_deep(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ast_deep(v) for v in x]
    return x


def _lint(ast):
    """Refuse runs of identical sibling statements: hand-unrolled iterations."""
    import json

    def bodies(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("body", "then", "else") and isinstance(v, list):
                    yield v
                yield from bodies(v)
        elif isinstance(x, list):
            for v in x:
                yield from bodies(v)

    for body in bodies(ast["actors"]):
        run, prev = 1, None
        for node in body:
            cur = json.dumps(node, sort_keys=True)
            run = run + 1 if cur == prev else 1
            prev = cur
            if run >= 3:
                kind = next(iter(node))
                raise BuildError(f"three or more identical `{kind}` statements in a row: a "
                                 f"hand-unrolled iteration; use cursor.loop() or repeat=n "
                                 f"(pass lint=False to Workload to allow it)")


# ----------------------------------------------------------------------------------------------
# the cursor
# ----------------------------------------------------------------------------------------------

class _Frame:
    __slots__ = ("body", "indices", "bindings", "channels", "written", "kind")

    def __init__(self, body, indices, bindings, channels, written, kind):
        self.body, self.indices, self.bindings = body, indices, bindings
        self.channels, self.written, self.kind = channels, written, kind


class Cursor:
    """Appends statements to the current body of one actor. Context managers (`loop`,
    `parallel`, `loader`, `when`, `otherwise`, `every`, `choose`, `phase`) push a body."""

    def __init__(self, wl: Workload, body: list):
        self.wl = wl
        self._frames = [_Frame(body, (), {}, set(), set(), "actor")]

    # ---- frame plumbing ----
    @property
    def _top(self) -> _Frame:
        return self._frames[-1]

    @property
    def indices(self) -> tuple[Index, ...]:
        """Loop indices in scope, outermost first."""
        return self._top.indices

    @property
    def index(self) -> Index:
        """The innermost index in scope."""
        if not self._top.indices:
            raise BuildError("no loop index in scope")
        return self._top.indices[-1]

    def _emit(self, node: dict):
        self._top.body.append(node)

    @contextlib.contextmanager
    def _push(self, body: list, kind: str, index: Index | None = None):
        top = self._top
        indices = top.indices + ((index,) if index is not None else ())
        # bindings are lexically scoped (V11); channels and the written set are per actor,
        # as in schema/check.py
        self._frames.append(_Frame(body, indices, dict(top.bindings), top.channels,
                                   top.written, kind))
        try:
            yield
        finally:
            self._frames.pop()

    def _new_index(self, name: str) -> Index:
        check_ident(name, "index")
        if any(i.name == name for i in self._top.indices):
            raise BuildError(f"index {name!r} shadows an enclosing index; loop indices are RNG "
                             f"keys and must be distinct")
        return Index(name)

    def _lookup(self, name):
        return self._top.bindings.get(name)

    # ---- bindings ----
    def let(self, name: str, value) -> Ref:
        """Bind a name to a positional value or a handle for the rest of this body and its
        children. Returns the reference."""
        check_ident(name, "binding")
        if isinstance(value, Dist):
            raise BuildError(f"let {name}: bind draw(dist), not the distribution itself")
        value = lift(value, f"let {name}")
        self._check_at(name, value)
        self._emit({"let": {"name": name, "value": ast_of(value)}})
        self._top.bindings[name] = value
        return Ref(name)

    def draw(self, name: str, dist) -> Ref:
        """`let name = draw(dist)`."""
        return self.let(name, Draw(dist))

    def ref(self, name: str) -> Ref:
        """A reference to a binding, including one defined later in the same loop body
        (for `x @ i` chains). Unresolved references fail validation."""
        return Ref(check_ident(name, "binding"))

    def _check_at(self, binding: str, value):
        """Rule V3 at construction: a self- or forward-reference `x @ e` needs e = i - d with
        d provably >= 1, i the innermost index."""
        for n in walk(value):
            if not isinstance(n, At):
                continue
            target = n.name
            is_self, is_forward = target == binding, self._lookup(target) is None
            if not (is_self or is_forward):
                continue
            if not self._top.indices:
                raise BuildError(f"let {binding}: `{target} @` outside any loop")
            i = self._top.indices[-1]
            idx = n.index
            ok = (isinstance(idx, Op) and idx.kind == "sub" and isinstance(idx.args[0], Index)
                  and idx.args[0].name == i.name and self._positive(idx.args[1]))
            if not ok:
                raise BuildError(f"let {binding}: `{target} @ e` refers to the binding being "
                                 f"defined (or one defined later), so e must be `{i.name} - d` "
                                 f"with d provably >= 1 (a positive literal, or a draw from a "
                                 f"distribution whose minimum is >= 1); got {ast_of(idx)}")

    def _positive(self, e) -> bool:
        if isinstance(e, bool):
            return False
        if isinstance(e, (int, float)):
            return e >= 1
        if isinstance(e, Ref):
            v = self._lookup(e.name)
            return v is not None and self._positive(v)
        if isinstance(e, Param):
            d = self.wl._param_specs.get(e.name, {}).get("default")
            return isinstance(d, Dist) and (d.lower_bound() or 0) >= 1
        if isinstance(e, Draw):
            d = e.dist
            if isinstance(d, Param):
                return self._positive(d)
            return isinstance(d, Dist) and (d.lower_bound() or 0) >= 1
        if isinstance(e, Dist):
            return (e.lower_bound() or 0) >= 1
        return False

    # ---- control statements ----
    @contextlib.contextmanager
    def loop(self, index: str, to, *, start=None, step=None):
        """`for index in [start, to) step step`. Yields the symbolic index."""
        idx = self._new_index(index)
        body: list = []
        node = {"index": index}
        if start is not None:
            node["from"] = ast_of(lift(start, "loop start"))
        node["to"] = ast_of(lift(to, "loop bound"))
        if step is not None:
            node["step"] = ast_of(lift(step, "loop step"))
        node["body"] = body
        self._emit({"loop": node})
        with self._push(body, "loop", idx):
            yield idx

    @contextlib.contextmanager
    def parallel(self, index: str, width):
        """`width` concurrent sub-actors, index 0..width, joined on exit. Yields a sub-cursor
        whose `.index` is the sub-actor index."""
        idx = self._new_index(index)
        body: list = []
        self._emit({"parallel": {"index": index, "width": ast_of(lift(width, "width")), "body": body}})
        with self._push(body, "parallel", idx):
            yield Sub(self, idx)

    @contextlib.contextmanager
    def loader(self, name: str, *, workers, prefetch, batches, ordered: bool = True, index: str = "b"):
        """PyTorch-style loader: `workers` sub-actors filling an ordered channel `name` of
        workers x prefetch slots with exactly `batches` batches. Yields the worker cursor
        whose `.index` is the batch index."""
        check_ident(name, "loader")
        idx = self._new_index(index)
        if name in self._top.channels:
            raise BuildError(f"channel {name!r} already declared")
        self._top.channels.add(name)
        body: list = []
        self._emit({"loader": {"name": name, "index": index,
                               "workers": ast_of(lift(workers, "workers")),
                               "prefetch": ast_of(lift(prefetch, "prefetch")),
                               "batches": ast_of(lift(batches, "batches")),
                               "ordered": bool(ordered), "body": body}})
        with self._push(body, "loader", idx):
            yield Sub(self, idx)

    def channel(self, name: str, capacity, *, ordered: bool = True) -> str:
        check_ident(name, "channel")
        if name in self._top.channels:
            raise BuildError(f"channel {name!r} already declared")
        self._top.channels.add(name)
        self._emit({"channel": {"name": name, "capacity": ast_of(lift(capacity, "capacity")),
                                "ordered": bool(ordered)}})
        return name

    def put(self, channel: str, seq):
        self._need_channel(channel)
        self._emit({"put": {"channel": channel, "seq": ast_of(lift(seq, "seq"))}})

    def take(self, channel: str):
        self._need_channel(channel)
        self._emit({"take": {"channel": channel}})

    def _need_channel(self, name):
        if name not in self._top.channels:
            raise BuildError(f"channel {name!r} not declared in this actor")

    def barrier(self, scope: str = "global"):
        if scope not in ("global", "host"):
            check_ident(scope, "barrier group")
        self._emit({"barrier": {"scope": scope}})

    def compute(self, ns):
        """Emulated compute of `ns` nanoseconds (an expression, or draw(dist))."""
        if isinstance(ns, Dist):
            ns = Draw(ns)
        elif isinstance(ns, Param) and isinstance(ns._default, Dist):
            ns = Draw(ns)
        self._emit({"compute": {"ns": ast_of(lift(ns, "compute"))}})

    @contextlib.contextmanager
    def when(self, test):
        """Statement conditional over indices, parameters, bindings, and the actor id."""
        test = lift(test, "when")
        body: list = []
        self._emit({"cond": {"if": ast_of(test), "then": body}})
        with self._push(body, "then"):
            yield

    @contextlib.contextmanager
    def otherwise(self):
        """The else branch of the immediately preceding `when`."""
        prev = self._top.body[-1] if self._top.body else None
        if not (prev and "cond" in prev and "else" not in prev["cond"]):
            raise BuildError("otherwise() must directly follow a when() block")
        body: list = []
        prev["cond"]["else"] = body
        with self._push(body, "else"):
            yield

    @contextlib.contextmanager
    def every(self, n, *, index: Index | None = None):
        """When the innermost (or the given) loop index is a multiple of n."""
        idx = self.index if index is None else index
        with self.when(idx % n == 0):
            yield

    @contextlib.contextmanager
    def choose(self):
        """Run one arm, chosen positionally by weight: `with c.arm(0.8): ...`."""
        arms: list = []
        self._emit({"choose": {"arms": arms}})
        yield Chooser(self, arms)
        if not arms:
            raise BuildError("choose() needs at least one arm")

    @contextlib.contextmanager
    def phase(self, name):
        """Statistics label; no effect on execution or the fingerprint. `name` may be a
        `when` expression choosing between strings."""
        name = lift(name, "phase name")
        if not isinstance(name, (str, Node)):
            raise BuildError("phase name must be a string or an expression")
        body: list = []
        self._emit({"phase": {"name": ast_of(name), "body": body}})
        with self._push(body, "phase"):
            yield

    def replay(self, trace: str, sha256: str):
        if not re.match(r"^[0-9a-f]{64}$", sha256):
            raise BuildError("replay sha256 must be 64 hex digits")
        self._emit({"replay": {"trace": trace, "sha256": sha256}})

    # ---- ops ----
    def _op(self, kind: str, args: dict, repeat=None):
        node = {kind: _strip_none(args)}
        if repeat is None:
            self._emit(node)
        else:
            # integer repeat: a loop with a generated index (schema/README.md §3)
            name = "rep"
            n = sum(1 for i in self._top.indices if i.name == "rep" or i.name.startswith("rep_"))
            if n:
                name = f"rep_{n}"
            idx = self._new_index(name)
            with self.loop(name, repeat):
                self._emit(node)

    def open(self, file, flags, *, mode: int | None = None, expect=None):
        self._op("open", {"file": _handle(file).ast(), "flags": _flags(flags), "mode": mode,
                          "expect": _expect(expect)})

    def _fileop(self, kind, file, expect):
        self._op(kind, {"file": _handle(file).ast(), "expect": _expect(expect)})

    def close(self, file, *, expect=None): self._fileop("close", file, expect)
    def fstat(self, file, *, expect=None): self._fileop("fstat", file, expect)
    def stat(self, file, *, expect=None): self._fileop("stat", file, expect)
    def fsync(self, file, *, expect=None): self._fileop("fsync", file, expect)
    def fdatasync(self, file, *, expect=None): self._fileop("fdatasync", file, expect)
    def unlink(self, file, *, expect=None): self._fileop("unlink", file, expect)

    def read(self, file, len, *, offset=None, repeat=None, expect=None):
        """`read(f, len)` sequential from the current position; with `offset`, positioned
        (pread). `repeat="until_eof"` reads until the known size is exhausted; an integer or
        expression repeat is a loop."""
        h = _handle(file)
        args = {"file": h.ast(), "len": ast_of(lift(len, "read len")),
                "offset": None if offset is None else ast_of(lift(offset, "read offset")),
                "expect": _expect(expect)}
        if isinstance(repeat, str) and repeat == "until_eof":
            args["repeat"] = "until_eof"
            self._note_until_eof(h)
            self._op("read", args)
        else:
            if isinstance(repeat, str):
                raise BuildError(f"read repeat must be until_eof, a count, or an expression; got {repeat!r}")
            self._op("read", args, repeat=None if repeat is None else lift(repeat, "repeat"))

    def write(self, file, len, *, offset=None, repeat=None, expect=None):
        h = _handle(file)
        if isinstance(repeat, str):
            raise BuildError("write has no until_eof; give a count")
        args = {"file": h.ast(), "len": ast_of(lift(len, "write len")),
                "offset": None if offset is None else ast_of(lift(offset, "write offset")),
                "expect": _expect(expect)}
        if isinstance(h, Ref):
            self._top.written.add(h.name)
        self._op("write", args, repeat=None if repeat is None else lift(repeat, "repeat"))

    def _note_until_eof(self, h: Handle):
        """Rule V4 at construction: until_eof on an as_written object needs the binding the
        creating writes used."""
        target = h
        if isinstance(h, Ref):
            target = self._lookup(h.name)
        if isinstance(target, ObjectHandle):
            ns = self.wl._namespaces.get(target.namespace)
            if ns and isinstance(ns.spec["size"], str) and ns.spec["size"] == "as_written":
                if not (isinstance(h, Ref) and h.name in self._top.written):
                    raise BuildError("until_eof on an as_written object needs the handle binding "
                                     "the creating writes used (same position)")

    def lseek(self, file, offset, whence: str = "SET"):
        if whence not in WHENCE:
            raise BuildError(f"lseek whence must be one of {WHENCE}")
        self._op("lseek", {"file": _handle(file).ast(), "offset": ast_of(lift(offset, "lseek offset")),
                           "whence": whence})

    def ioctl(self, file, request: str, *, expect=None):
        if request not in IOCTLS:
            raise BuildError(f"ioctl request must be one of {IOCTLS}")
        self._op("ioctl", {"file": _handle(file).ast(), "request": request, "expect": _expect(expect)})

    def ftruncate(self, file, len, *, expect=None):
        self._op("ftruncate", {"file": _handle(file).ast(), "len": ast_of(lift(len, "len")),
                               "expect": _expect(expect)})

    def fallocate(self, file, len, *, offset=None, expect=None):
        self._op("fallocate", {"file": _handle(file).ast(),
                               "offset": None if offset is None else ast_of(lift(offset, "offset")),
                               "len": ast_of(lift(len, "len")), "expect": _expect(expect)})

    def mkdir(self, dir, *, mode: int | None = None, expect=None):
        self._op("mkdir", {"dir": _handle(dir, "dir").ast(), "mode": mode, "expect": _expect(expect)})

    def rmdir(self, dir, *, expect=None):
        self._op("rmdir", {"dir": _handle(dir, "dir").ast(), "expect": _expect(expect)})

    def rename(self, src, dst, *, expect=None):
        self._op("rename", {"from": _handle(src, "from").ast(), "to": _handle(dst, "to").ast(),
                            "expect": _expect(expect)})

    def readdir(self, dir, *, repeat: str = "until_end", expect=None):
        if repeat != "until_end":
            raise BuildError("readdir repeat is always until_end")
        self._op("readdir", {"dir": _handle(dir, "dir").ast(), "repeat": "until_end",
                             "expect": _expect(expect)})


class Sub:
    """A view of a cursor inside `parallel` or `loader`, carrying that construct's index."""

    def __init__(self, cursor: Cursor, index: Index):
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "index", index)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class Chooser:
    def __init__(self, cursor: Cursor, arms: list):
        self._cursor, self._arms = cursor, arms

    @contextlib.contextmanager
    def arm(self, weight):
        if not isinstance(weight, (int, float)) or weight < 0:
            raise BuildError("arm weight must be a non-negative number")
        body: list = []
        self._arms.append({"weight": weight, "body": body})
        with self._cursor._push(body, "arm"):
            yield
