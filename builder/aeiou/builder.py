"""The Workload builder: declarations (params, datasets, namespaces, actors) and the Cursor
that appends statements to an actor body.

Python control flow runs at build time and generates structure; cursor control flow (`loop`,
`parallel`, `when`, `every`, `choose`) becomes AST nodes and runs at execution time. The
`with` blocks push a body, hand back a symbolic index or a sub-cursor, and pop on exit.
"""
from __future__ import annotations

import contextlib
import math
import re
import sys
from typing import Any

from .pattern import Pattern
from .nodes import (Actor, At, BuildError, Consume, Cond, DatasetMeta, DirHandle, Dist, Draw, InventoryHandle,
                    Expr, FileHandle, FileOf, Handle, Index, Node, ObjectHandle, Op, Param, Pick, Ref,
                    ast_of, check_ident, distref, lift, lower_bound_of, walk)

OPEN_FLAGS = ("RDONLY", "WRONLY", "RDWR", "CREAT", "TRUNC", "EXCL", "APPEND", "CLOEXEC",
              "DIRECTORY", "DIRECT", "SYNC", "DSYNC", "NOATIME", "NOFOLLOW")
IOCTLS = ("TCGETS", "FIONREAD", "BLKGETSIZE64")
ADVICE = ("NORMAL", "RANDOM", "SEQUENTIAL", "WILLNEED", "DONTNEED", "NOREUSE")
LAYOUTS = ("whole", "chunked", "compound", "array")
ACCESSES = ("map", "stream", "indexed")
FRAMING_KEYS = ("unit", "file_header", "file_header_per_sample", "file_header_per_unit", "file_footer",
               "file_footer_per_sample", "file_footer_per_unit", "file_align", "unit_header",
               "unit_header_per_sample", "unit_footer", "unit_footer_per_sample", "unit_align", "columns", "writer")
COLUMN_KEYS = ("header", "fixed", "weight", "row_header", "row_footer", "row_align", "align")
WHENCE = ("SET", "CUR", "END")
ERRNO = re.compile(r"^E[A-Z0-9]{1,15}$")
RESERVED_PREFIX = ".aeiou"      # V13: the dataset manifest `.aeiou-dataset.json` and future sidecars
WRITE_FLAGS = {"WRONLY", "RDWR", "CREAT", "TRUNC", "APPEND"}
FIELD = re.compile(r"\{([a-z_][a-z0-9_]*)(?:\s+(?:div|mod)\s+(?:\d+|[a-z_][a-z0-9_]*))?(?::[^}]*)?\}")

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


def _reserved(pattern: str, what: str):
    for comp in pattern.split("/"):
        if comp.startswith(RESERVED_PREFIX):
            raise BuildError(f"{what}: path component {comp!r} begins with {RESERVED_PREFIX!r}, which is "
                             f"reserved for the dataset manifest (schema/README.md V13)")


def _protocol(protocol, what: str) -> None:
    """A dataset's or namespace's protocol (contract 0.6; `s3` was `object` until 0.7): `posix`
    (absent) or `s3`."""
    if protocol is not None and protocol not in ("posix", "s3"):
        raise BuildError(f"{what}: protocol must be posix or s3")

def _root(pattern: str) -> str:
    return pattern.split("{", 1)[0].rpartition("/")[0]


def _field_glob(component: str) -> re.Pattern:
    """The names a pattern component can produce: each field one or more of the characters an
    integer field formats to."""
    parts = re.split(r"\{[^}]*\}", component)
    return re.compile("[0-9a-f -]+".join(re.escape(p) for p in parts))


def _inventory(name: str, pattern: str, access: str, inv) -> dict:
    """V22: an inventory of a consumed dataset, a constant file name directly in its root that
    the pattern cannot produce there."""
    if not isinstance(inv, dict) or set(inv) != {"name", "entry"}:
        raise BuildError(f"dataset {name}: inventory is {{'name': file, 'entry': bytes}}")
    n = inv["name"]
    if access not in ("map", "stream"):
        raise BuildError(f"dataset {name}: only a consumed dataset (`map` or `stream`) has an inventory (V22)")
    if not isinstance(n, str) or n in ("", ".", "..") or any(c in n for c in "/{}"):
        raise BuildError(f"dataset {name}: inventory name {n!r} is not a file name: it lies directly in "
                         f"the dataset's root, a constant (V22)")
    if n.startswith(RESERVED_PREFIX):
        raise BuildError(f"dataset {name}: inventory name {n!r} begins with {RESERVED_PREFIX!r}, reserved "
                         f"for the dataset manifest (V13)")
    root = _root(pattern)
    sibling = (pattern[len(root) + 1:] if root else pattern).split("/")[0]
    if _field_glob(sibling).fullmatch(n):
        raise BuildError(f"dataset {name}: inventory name {n!r} is a name the pattern {pattern!r} can "
                         f"produce in the dataset's root (V22)")
    return {"name": n, "entry": lift(inv["entry"], "inventory entry")}


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
    def __init__(self, name: str, layout: str, spec: dict):
        self.name, self.layout, self.spec = name, layout, spec
        #: the format class instance the dataset was declared with (`aeiou.formats`), for its
        #: reader protocol: `ds.format.stream(cursor, shard)`; None without one
        self.format = None

    def ast(self):
        return _strip_none(self.spec)

    @property
    def count(self) -> Expr:
        return DatasetMeta("count", self.name)

    @property
    def dirs(self) -> Expr:
        if self.layout == "array":
            raise BuildError(f"dirs() on array dataset {self.name}: an array is one file")
        return DatasetMeta("dirs", self.name)

    def file(self, id=None) -> Handle:
        """Dataset file by id. An `array` dataset has one file, addressed as id 0."""
        if self.layout == "array":
            if id is not None:
                raise BuildError(f"array dataset {self.name} has one file; call file() with no id")
            return FileHandle(self.name, 0)
        if id is None:
            raise BuildError(f"file() on dataset {self.name} needs an id")
        return FileHandle(self.name, id)

    def dir(self, id) -> Handle:
        if self.layout == "array":
            raise BuildError(f"dir() on array dataset {self.name}: an array is one file")
        return DirHandle(self.name, id)

    def inventory(self) -> Handle:
        """The dataset's inventory (contract 0.9): the file list declared with `inventory=`,
        read in place of a walk (DESIGN_REVIEW.md §3.73)."""
        if not self.spec.get("inventory"):
            raise BuildError(f"inventory() on dataset {self.name}, which declares none: pass "
                             f"inventory={{'name': …, 'entry': …}} to Workload.dataset (V22)")
        return InventoryHandle(self.name)

    def consume(self) -> Handle:
        """Next sample without replacement (positional; must sit inside a loop). Only a `map`
        or `stream` dataset is consumed (V21)."""
        if self.spec.get("access") not in ("map", "stream"):
            raise BuildError(f"consume() on dataset {self.name}, which is `indexed`: look its items up by id "
                             f"(file(), pick()), or declare it `map` or `stream` (V21)")
        return Consume(self.name)

    def pick(self, dist=None) -> Handle:
        """A sample with replacement, uniform unless a distribution over ids is given."""
        return Pick(self.name, dist)


class Namespace:
    def __init__(self, name: str, spec: dict, fields: dict, input: bool = False):
        self.name, self.spec, self.fields, self.input = name, spec, fields, input

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
    AST_VERSION = "0.10"
    MAX_CANONICAL_BYTES = 4 * 1024 * 1024

    POSIX_APIS = ("sync", "posix-aio", "libaio", "io_uring", "mmap")
    CACHES = ("per-open", "direct")
    S3_APIS = ("blocking", "async")

    def __init__(self, name: str, doc: str | None = None, *, posix: str | None = None, cache: str | None = None,
                 s3: str | None = None, lint: bool = True):
        """`posix`: the API the traced application issues the I/O of its `posix` names through
        (a runner `--posix` name); a run uses it by default. Leave it out for an application
        that calls read and write (`sync`). `cache`: `direct` for an application that opens
        every regular file `O_DIRECT`; leave it out when the open flags of the ops say what it
        does. `s3`: how its S3 client uses the store for its `s3` names, `blocking` (left out)
        or `async` (contract 0.7, `DESIGN_REVIEW.md` §3.65; V19 relates the three to the
        protocols the workload has names on)."""
        self.name = check_ident(name, "workload name")
        self.doc = doc
        if posix is not None and posix not in self.POSIX_APIS:
            raise BuildError(f"workload {name}: posix must be one of {self.POSIX_APIS}")
        if cache is not None and cache not in self.CACHES:
            raise BuildError(f"workload {name}: cache must be one of {self.CACHES}")
        if s3 is not None and s3 not in self.S3_APIS:
            raise BuildError(f"workload {name}: s3 must be one of {self.S3_APIS}")
        if posix == "mmap" and cache == "direct":
            raise BuildError(f"workload {name}: cache `direct` under posix `mmap`: its reads are page faults on a mapping")
        self.posix = posix
        self.cache = cache
        self.s3 = s3
        self.lint = lint
        self._params: dict[str, Param] = {}
        self._param_specs: dict[str, dict] = {}
        self._datasets: dict[str, Dataset] = {}
        self._namespaces: dict[str, Namespace] = {}
        self._actors: dict[str, dict] = {}
        self._roots: dict[str, str] = {}
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

    def _divisors(self, pattern: str, what: str) -> None:
        """V23 (contract 0.10, DESIGN_REVIEW.md §3.75): a divisor named in a pattern
        (`{id div files_per_directory}`) is a parameter declared before it, its default an integer."""
        try:
            names = Pattern(pattern).params()
        except ValueError as e:
            raise BuildError(f"{what}: {e}") from None
        for n in names:
            spec = self._param_specs.get(n)
            if spec is None:
                raise BuildError(f"{what}: divisor `{n}` is not a declared parameter (V23)")
            d = spec["default"]
            if not isinstance(d, int) or isinstance(d, bool):
                raise BuildError(f"{what}: divisor `{n}` is a parameter whose default is not an integer (V23)")
            if d <= 0:
                raise BuildError(f"{what}: divisor `{n}` is a parameter whose default {d} is not positive (V23)")

    # ---- datasets ----
    def dataset(self, name: str, *, pattern: str, layout: str, access: str, count, size, seed: int,
                samples_per_file=None, chunk=None, slot=None, format: dict | None = None,
                protocol: str | None = None, inventory: dict | None = None, doc: str | None = None) -> Dataset:
        """A dataset (contract 0.8, DESIGN_REVIEW.md §3.72): files or objects named by `pattern`,
        sizes drawn from `seed`. `layout`, how a file holds the items: `whole` (one item),
        `chunked` (part of one: `chunk` bytes a file, a `{k}` field), `compound` (several
        variably sized ones: `samples_per_file`, framed by `format`), `array` (`count` fixed
        `slot`s in one file; a pattern without fields). `access`, how they are found: `map`
        (samples drawn from the shuffle), `stream` (files drawn from the shuffle), `indexed`
        (looked up by id). `protocol`: `s3` for one the application reads from an object store;
        absent, a file system. `inventory`, `{"name": file, "entry": bytes}`: a file list of the
        dataset in its root, `entry` bytes per item of `count`, read in place of a walk (a
        consumed dataset's; contract 0.9, DESIGN_REVIEW.md §3.73, V22)."""
        self._declare(name, "dataset")
        _protocol(protocol, f"dataset {name}")
        if layout not in LAYOUTS:
            raise BuildError(f"dataset {name}: layout must be one of {', '.join(LAYOUTS)}, got {layout!r}")
        if access not in ACCESSES:
            raise BuildError(f"dataset {name}: access must be one of {', '.join(ACCESSES)}, got {access!r}")
        given = {"samples_per_file": samples_per_file, "chunk": chunk, "slot": slot, "format": format}
        need, may = {"whole": ((), ()), "chunked": (("chunk",), ()), "compound": (("samples_per_file",), ("format",)),
                     "array": (("slot",), ())}[layout]
        for k, v in given.items():
            if k in need and v is None:
                raise BuildError(f"dataset {name}: layout `{layout}` needs `{k}` (V20)")
            if v is not None and k not in need and k not in may:
                raise BuildError(f"dataset {name}: layout `{layout}` takes no `{k}` (V20)")
        fields = set(FIELD.findall(pattern))
        if layout == "array":
            if fields:
                raise BuildError(f"dataset {name}: an array is one file: its pattern has no fields, got {sorted(fields)} (V20)")
            if access != "indexed":
                raise BuildError(f"dataset {name}: an array is `indexed`: its slots are looked up by position (V21)")
        else:
            want = {"id", "k"} if layout == "chunked" else {"id"}
            if fields != want:
                raise BuildError(f"dataset {name}: a `{layout}` dataset's pattern fields are {sorted(want)}, "
                                 f"got {sorted(fields)} (V20)")
        if access in ("map", "stream") and re.search(r"\{id mod ", pattern.rpartition("/")[0]):
            raise BuildError(f"dataset {name}: a consumed dataset's directories are `{{id div D}}` or none, never `mod`, "
                             f"so a subset of it is whole directories (V21)")
        fmt_obj = format if hasattr(format, "spec") else None
        if format is not None:
            format = _format_spec(name, format, access, size, {k: v["default"] for k, v in self._param_specs.items()})
        _reserved(pattern, f"dataset {name}")
        self._divisors(pattern, f"dataset {name}")
        if inventory is not None:
            inventory = _inventory(name, pattern, access, inventory)
        self._claim_root(pattern.rpartition("/")[0] if layout == "array" else _root(pattern), name)
        spec = {"pattern": pattern, "layout": layout, "access": access, "count": lift(count, "count"),
                "size": distref(size), "seed": _seed(seed, name), "protocol": protocol,
                "samples_per_file": None if samples_per_file is None else lift(samples_per_file),
                "chunk": None if chunk is None else lift(chunk, "chunk"),
                "slot": None if slot is None else lift(slot, "slot"), "format": format,
                "inventory": inventory, "doc": doc}
        self._datasets[name] = ds = Dataset(name, layout, spec)
        ds.format = fmt_obj
        return ds

    # ---- namespaces ----
    def namespace(self, name: str, *, pattern: str, fields: dict, size, seed: int,
                  input: bool = False, same_run: bool = False, protocol: str | None = None,
                  doc: str | None = None) -> Namespace:
        """Workload-created objects. `size` is an expression over the fields and params, or
        "as_written" (the sum of the writes that create the object; schema/README.md V4).
        `input=True`: a previous run wrote the objects and this abstract only reads them; the
        runner requires that run's `.aeiou-namespace.json` (schema/README.md V14).
        `same_run=True` (with `input`): the names read are positional draws of the writing run,
        so the runner refuses a run whose seed, instance count, or common parameters differ from
        the writer's (schema/README.md V15)."""
        if same_run and not input:
            raise BuildError(f"namespace {name}: `same_run` without `input`: only an input namespace has a writer "
                             f"to compare with (schema/README.md V15)")
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
        _reserved(pattern, f"namespace {name}")
        self._divisors(pattern, f"namespace {name}")
        nroot = _root(pattern)
        for root, dname in self._roots.items():
            if root and (nroot == root or nroot.startswith(root + "/")):
                raise BuildError(f"namespace {name}: root {nroot!r}/ lies inside dataset {dname}'s root "
                                 f"{root!r}/ (schema/README.md V13)")
        if not (isinstance(size, str) and size == "as_written"):
            size = lift(size, "namespace size")
            for n in walk(size):
                if isinstance(n, Ref) and n.name not in fields:
                    raise BuildError(f"namespace {name}: size may reference only its fields and params")
        _protocol(protocol, f"namespace {name}")
        for other in self._namespaces.values():
            if _root(other.spec["pattern"]) == nroot and other.input != bool(input):
                raise BuildError(f"namespace {name}: shares root {nroot!r}/ with namespace {other.name} but "
                                 f"`input` differs; a root has one manifest (schema/README.md V14)")
            if _root(other.spec["pattern"]) == nroot and (other.spec.get("protocol") or "posix") != (protocol or "posix"):
                raise BuildError(f"namespace {name}: shares root {nroot!r}/ with namespace {other.name} but "
                                 f"`protocol` differs; a root is in one place (schema/README.md V17)")
        spec = {"pattern": pattern, "fields": fields, "size": size, "seed": _seed(seed, name),
                "input": True if input else None, "same_run": True if same_run else None,
                "protocol": protocol, "doc": doc}
        self._namespaces[name] = ns = Namespace(name, spec, fields, input=bool(input))
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

    def _claim_root(self, root: str, name: str):
        if root in self._roots:
            raise BuildError(f"dataset {name}: shares root {root!r}/ with dataset {self._roots[root]}; "
                             f"one manifest per root (schema/README.md V13)")
        for other, owner in self._roots.items():   # a dataset root is emptied, generated and checked as a whole
            inner, outer = (root, other) if len(root) > len(other) else (other, root)
            if outer == "" or inner.startswith(outer + "/"):
                raise BuildError(f"dataset {name}: root {root!r}/ and root {other!r}/ of dataset {owner} are nested; "
                                 f"`aeiou datagen` wants each dataset root to itself (schema/README.md V13)")
        self._roots[root] = name

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
        if self.posix:
            ast["posix"] = self.posix
        if self.cache:
            ast["cache"] = self.cache
        if self.s3:
            ast["s3"] = self.s3
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
        """Validate, then write the AST as JSON to `path` (default `<name>.ast.json`), with a
        provenance block. Returns the path."""
        from .emit import write
        return write(self, path, provenance=provenance)


def _format_spec(name: str, format, access: str, size=None, params: dict | None = None) -> dict:
    """The `format` entry of a dataset: a dict `{class, reader?, version?, framing?}` (a format
    class object of `aeiou.formats` renders to one through `.spec(access, size, params)`)."""
    spec = format.spec(access, size, params) if hasattr(format, "spec") else format
    if not isinstance(spec, dict) or "class" not in spec:
        raise BuildError(f"dataset {name}: format needs a `class`")
    check_ident(spec["class"], "format class")
    for k in spec:
        if k not in ("class", "reader", "version", "framing"):
            raise BuildError(f"dataset {name}: unknown format key {k!r}")
    framing = spec.get("framing")
    if framing is not None:
        if not isinstance(framing, dict):
            raise BuildError(f"dataset {name}: format framing must be a dict")
        for k, v in framing.items():
            if k not in FRAMING_KEYS:
                raise BuildError(f"dataset {name}: unknown framing key {k!r} (one of {FRAMING_KEYS})")
            if k == "columns":
                if not isinstance(v, list) or not v:
                    raise BuildError(f"dataset {name}: framing columns must be a non-empty list")
                weights = []
                for c in v:
                    for ck in c:
                        if ck not in COLUMN_KEYS:
                            raise BuildError(f"dataset {name}: unknown column key {ck!r} (one of {COLUMN_KEYS})")
                    w = c.get("weight", 0)
                    if not isinstance(w, (int, float)) or w < 0:
                        raise BuildError(f"dataset {name}: column weight must be a non-negative number")
                    weights.append(w)
                if abs(sum(weights) - 1) > 1e-9:
                    raise BuildError(f"dataset {name}: column weights sum to {sum(weights)}, not 1")
                framing["columns"] = [{ck: (float(cv) if ck == "weight" else lift(cv, f"column {ck}")) for ck, cv in c.items()} for c in v]
            elif k != "writer":
                framing[k] = lift(v, f"framing {k}")
    return spec


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
        self._nonneg_indices: set[str] = set()   # indices provably >= 0 (V3); names are distinct in scope

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
        """Rule V3 at construction: an `x @ e` on a binding of the same loop body (the one being
        defined, one defined later, or one defined earlier in that body) needs e = i - d with d
        provably >= 1, i the innermost index. A binding of an enclosing body is free."""
        same_body = {n["let"]["name"] for n in self._top.body if "let" in n}
        for n in walk(value):
            if not isinstance(n, At):
                continue
            target = n.name
            if not (target == binding or target in same_body or self._lookup(target) is None):
                continue
            if not self._top.indices:
                raise BuildError(f"let {binding}: `{target} @` outside any loop")
            i = self._top.indices[-1]
            idx = n.index
            ok = (isinstance(idx, Op) and idx.kind == "sub" and isinstance(idx.args[0], Index)
                  and idx.args[0].name == i.name and self._at_least(idx.args[1], 1))
            if not ok:
                raise BuildError(f"let {binding}: `{target} @ e` refers to a binding of this loop "
                                 f"body, so e must be `{i.name} - d` with d provably >= 1 (a "
                                 f"positive literal, a draw from a distribution whose minimum is "
                                 f">= 1, or `add` of such a term and one provably >= 0: a `mod`, "
                                 f"a loop index, a literal, a draw with minimum >= 0); got {ast_of(idx)}")

    def _at_least(self, e, k: int) -> bool:
        """`e` provably >= k, for k in {0, 1} (rule V3). Judged on the defaults; the runner judges
        the same forms again on what it was given."""
        if isinstance(e, bool):
            return False
        if isinstance(e, (int, float)):
            return e >= k
        if isinstance(e, Ref):
            v = self._lookup(e.name)
            return v is not None and self._at_least(v, k)
        if isinstance(e, Param):
            d = self.wl._param_specs.get(e.name, {}).get("default")
            return isinstance(d, Dist) and self._dist_at_least(d, k)
        if isinstance(e, Draw):
            d = e.dist
            if isinstance(d, Param):
                return self._at_least(d, k)
            return isinstance(d, Dist) and self._dist_at_least(d, k)
        if isinstance(e, Dist):
            return self._dist_at_least(e, k)
        if isinstance(e, Op) and e.kind == "add":
            a, b = e.args
            return ((self._at_least(a, k) and self._at_least(b, 0))
                    or (self._at_least(a, 0) and self._at_least(b, k)))
        if k == 0:
            if isinstance(e, Op) and e.kind == "mod":          # Euclidean in the runner: in [0, |b|)
                return True
            if isinstance(e, Index):                            # steps are positive: index >= from
                return e.name in self._nonneg_indices
        return False

    def _dist_at_least(self, d: Dist, k: int) -> bool:
        lb = d.lower_bound(self._number)
        return lb is not None and lb >= k

    def _number(self, e):
        """The number `e` is known to be: a parameter's default when it is one, else None."""
        if isinstance(e, Param):
            e = self.wl._param_specs.get(e.name, {}).get("default")
        return e if isinstance(e, (int, float)) and not isinstance(e, bool) else None

    # ---- control statements ----
    @contextlib.contextmanager
    def loop(self, index: str, to, *, start=None, step=None):
        """`for index in [start, to) step step`. Yields the symbolic index."""
        idx = self._new_index(index)
        body: list = []
        node = {"index": index}
        self._nonneg_indices.discard(index)
        if start is not None:
            start = lift(start, "loop start")
            node["from"] = ast_of(start)
        if start is None or self._at_least(start, 0):
            self._nonneg_indices.add(index)
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
        self._nonneg_indices.add(index)
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
        self._nonneg_indices.add(index)
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
        if sum(a["weight"] for a in arms) <= 0:
            raise BuildError("choose: total weight is zero")

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

    def trace(self, file: str, sha256: str):
        """A captured trace of a real application, executed literally (schema: the `trace`
        node; its file format is deferred). `file` is relative to the AST document."""
        if not re.match(r"^[0-9a-f]{64}$", sha256):
            raise BuildError("trace sha256 must be 64 hex digits")
        self._emit({"trace": {"file": file, "sha256": sha256}})

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

    def _dataset_of(self, h):
        """Dataset name if the handle is (a binding to) a dataset file, dir, sample, region, or chunk."""
        if isinstance(h, Ref):
            return self._dataset_of(self._lookup(h.name))
        if isinstance(h, (FileHandle, DirHandle, Consume, Pick, InventoryHandle)):
            return h.dataset
        if isinstance(h, FileOf):
            return self._dataset_of(h.of)
        return None

    def _read_only(self, h, what):
        ds = self._dataset_of(h)
        if ds:
            raise BuildError(f"{what} on dataset {ds}: datasets are read-only, writes go to namespaces "
                             f"(schema/README.md V12)")
        ns = self._input_namespace_of(h)
        if ns:
            raise BuildError(f"{what} on input namespace {ns}: input namespaces are read-only "
                             f"(schema/README.md V14)")

    def _object_store_namespace(self, h):
        """Namespace name if the handle is (a binding to) an object of a namespace declared
        `protocol: s3`."""
        if isinstance(h, Ref):
            return self._object_store_namespace(self._lookup(h.name))
        if isinstance(h, ObjectHandle) and self.wl._namespaces[h.namespace].spec.get("protocol") == "s3":
            return h.namespace
        return None

    def _once_in_order(self, h, what):
        """V18: an object in an object store is written once, in order from 0, by one upload."""
        ns = self._object_store_namespace(h)
        if ns:   # in the reference checker's words (schema/check.py, validate.rs)
            why = ("an object is made only by the upload an open with WRONLY begins, and a read-only handle writes none"
                   if what.endswith("without WRONLY") else f"an object is written once, in order from 0, and has no {what}")
            raise BuildError(f"{what} on namespace `{ns}`, declared `protocol: s3`: {why} (V18)")

    def _input_namespace_of(self, h):
        if isinstance(h, Ref):
            return self._input_namespace_of(self._lookup(h.name))
        if isinstance(h, ObjectHandle) and self.wl._namespaces[h.namespace].input:
            return h.namespace
        return None

    def open(self, file, flags, *, mode: int | None = None, expect=None):
        flags = _flags(flags)
        if set(flags) & WRITE_FLAGS:
            self._read_only(file, "open for writing")
        for f in flags:
            if f in ("APPEND", "RDWR", "EXCL"):
                self._once_in_order(file, f"open with {f}")
            # an upload is begun by WRONLY alone (a close missing from the body is the
            # checker's, at build)
            if f in ("CREAT", "TRUNC") and "WRONLY" not in flags:
                self._once_in_order(file, f"open with {f} without WRONLY")
        self._op("open", {"file": _handle(file).ast(), "flags": flags, "mode": mode,
                          "expect": _expect(expect)})

    def _fileop(self, kind, file, expect):
        self._op(kind, {"file": _handle(file).ast(), "expect": _expect(expect)})

    def close(self, file, *, expect=None): self._fileop("close", file, expect)
    def fstat(self, file, *, expect=None): self._fileop("fstat", file, expect)
    def stat(self, file, *, expect=None): self._fileop("stat", file, expect)
    def fsync(self, file, *, expect=None): self._fileop("fsync", file, expect)
    def fdatasync(self, file, *, expect=None): self._fileop("fdatasync", file, expect)
    def unlink(self, file, *, expect=None):
        self._read_only(file, "unlink")
        self._fileop("unlink", file, expect)

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
        self._read_only(h, "write")
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

    def fadvise(self, file, advice: str, *, offset=None, len=None, expect=None):
        """`posix_fadvise(fd, offset, len, advice)` on an open file (`len` 0 or absent: to EOF).
        WILLNEED starts readahead on NFS; pyarrow issues one before every coalesced read."""
        if advice not in ADVICE:
            raise BuildError(f"fadvise advice must be one of {ADVICE}")
        self._op("fadvise", {"file": _handle(file).ast(), "advice": advice,
                             "offset": None if offset is None else ast_of(lift(offset, "fadvise offset")),
                             "len": None if len is None else ast_of(lift(len, "fadvise len")),
                             "expect": _expect(expect)})

    def ftruncate(self, file, len, *, expect=None):
        self._read_only(file, "ftruncate")
        self._once_in_order(file, "ftruncate")
        self._op("ftruncate", {"file": _handle(file).ast(), "len": ast_of(lift(len, "len")),
                               "expect": _expect(expect)})

    def fallocate(self, file, len, *, offset=None, expect=None):
        self._read_only(file, "fallocate")
        self._once_in_order(file, "fallocate")
        self._op("fallocate", {"file": _handle(file).ast(),
                               "offset": None if offset is None else ast_of(lift(offset, "offset")),
                               "len": ast_of(lift(len, "len")), "expect": _expect(expect)})

    def mkdir(self, dir, *, mode: int | None = None, expect=None):
        self._read_only(dir, "mkdir")
        self._op("mkdir", {"dir": _handle(dir, "dir").ast(), "mode": mode, "expect": _expect(expect)})

    def rmdir(self, dir, *, expect=None):
        self._read_only(dir, "rmdir")
        self._op("rmdir", {"dir": _handle(dir, "dir").ast(), "expect": _expect(expect)})

    def rename(self, src, dst, *, expect=None):
        self._read_only(src, "rename")
        self._read_only(dst, "rename")
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
        if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
                or not math.isfinite(weight) or weight < 0):
            raise BuildError(f"arm weight must be a finite non-negative number, got {weight!r}")
        body: list = []
        self._arms.append({"weight": weight, "body": body})
        with self._cursor._push(body, "arm"):
            yield
