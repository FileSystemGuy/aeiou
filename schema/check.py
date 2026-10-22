#!/usr/bin/env python3
"""Validate abstract ASTs against schema/abstract-ast.schema.json and the semantic rules of
schema/README.md, print each AST's canonical hash and op-kind counts.

Usage: python3 schema/check.py [file.ast.json ...]   (default: workloads/**/*.ast.json)

The reference checker: the Rust validator (`aeiou check`) must refuse what this refuses, at the
same path with the same message, and CI diffs the two over the committed ASTs. It needs only jsonschema.
"""
import hashlib
import re
import json
import math
import pathlib
import sys
from collections import Counter

import jsonschema

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "abstract-ast.schema.json").read_text(encoding="utf-8"))

CONTROL = {"let", "loop", "parallel", "channel", "put", "take", "loader", "barrier",
           "compute", "cond", "choose", "phase", "trace"}
INDEX_BINDERS = {"loop", "parallel", "loader"}
MUTATING_OPS = {"write", "ftruncate", "fallocate", "unlink"}          # V12 and V14, plus rename, mkdir, rmdir, and open flags
WRITE_FLAGS = {"WRONLY", "RDWR", "CREAT", "TRUNC", "APPEND"}
RESERVED_PREFIX = ".aeiou"                                            # V13: the manifest and sidecars


def param_kind(p: dict):
    """A parameter's kind (contract 0.11): its declared `kind` when required, else its
    default's (`int`, `float`, `bool`, `str`, `dist`, `array`), as the runner's `Kind::of`."""
    if "kind" in p:
        return p["kind"]
    d = p.get("default")
    if isinstance(d, dict):
        return "int" if "from" in d else "dist"
    if isinstance(d, list):
        return "array"
    if isinstance(d, bool):
        return "bool"
    return "int" if isinstance(d, int) else "float" if isinstance(d, float) else "str"


def dataset_root(d: dict) -> str:
    """Constant directory prefix of a dataset's pattern (an array's: its file's directory)."""
    prefix = d["pattern"].split("{", 1)[0]
    return prefix.rpartition("/")[0]


_LOWER, _DIGITS = set("abcdefghijklmnopqrstuvwxyz"), set("0123456789")


def pattern_fields(src: str) -> list:
    """The fields of a path pattern as the runner parses it (`runner/aeiou/src/pattern.rs`), each
    `(name, op, param, in_dir)`: `op` `div`, `mod`, or None, `param` the parameter naming the
    divisor or None, `in_dir` whether the field is in the directory part. A pattern the runner
    cannot parse raises ValueError with the runner's message."""
    fields, rest = [], src
    while "{" in rest:
        after = rest[rest.index("{") + 1:]
        if "}" not in after:
            raise ValueError(f"pattern `{src}`: unclosed `{{`")
        spec, rest = after.split("}", 1)
        expr, colon, fmt = spec.partition(":")
        expr, fmt = expr.strip(), fmt.strip() if colon else ""
        words = expr.split()
        op = param = None
        if len(words) == 1:
            name = words[0]
        elif len(words) == 3:
            name, op, n = words
            if op not in ("div", "mod"):
                raise ValueError(f"pattern `{src}`: unknown field operator `{op}`")
            if n[0] in _LOWER or n[0] == "_":
                if not set(n) <= _LOWER | _DIGITS | {"_"}:
                    raise ValueError(f"pattern `{src}`: bad parameter name `{n}` in `{{{spec}}}`")
                param = n
            else:
                if not re.fullmatch(r"[+-]?[0-9]+", n) or not -2**63 <= int(n) < 2**63:
                    raise ValueError(f"pattern `{src}`: bad number in `{{{spec}}}`")
                if int(n) <= 0:
                    raise ValueError(f"pattern `{src}`: `{op} {int(n)}` needs a positive divisor")
        else:
            raise ValueError(f"pattern `{src}`: cannot parse field `{{{spec}}}`")
        if not name or not set(name) <= _LOWER | _DIGITS | {"_"}:
            raise ValueError(f"pattern `{src}`: bad field name `{name}`")
        digits = fmt[1:] if fmt.startswith("0") else fmt
        digits = digits[:-1] if digits.endswith(("x", "d")) else digits
        if digits and (not re.fullmatch(r"\+?[0-9]+", digits) or int(digits) >= 2**64):
            raise ValueError(f"pattern `{src}`: bad format `{fmt}` in `{{{spec}}}`")
        fields.append((name, op, param, "/" in rest))
    return fields


def parses(src: str) -> bool:
    """Whether the runner can parse the path pattern `src` (`pattern_fields`)."""
    try:
        pattern_fields(src)
    except ValueError:
        return False
    return True


def protocols(ast: dict, traces: bool) -> tuple:
    """(posix, s3): the protocols an abstract has names on. `posix` when a dataset or namespace
    declares it (or none), when a `trace` node runs (its files are under --root), or when no
    name is `s3`."""
    declared = [d.get("protocol", "posix") for d in ast.get("datasets", {}).values()]
    declared += [n.get("protocol", "posix") for n in ast.get("namespaces", {}).values()]
    s3 = "s3" in declared
    return ("posix" in declared or traces or not s3), s3


def round_away(x: float) -> float:
    """Round to the nearest integer, halves away from zero, as the runner rounds a draw (V10):
    Rust's `f64::round`, the algorithm of `aeiou.rng.rust_round` without its cast to i64 (a
    float, unsaturated; NaN and the infinities as they are). No `x + 0.5`: that sum rounds
    itself, -0.49999999999999994 to -1 and -(2^52 + 1) to -(2^52 + 2)."""
    if not math.isfinite(x):
        return x
    a = abs(x)
    r = math.floor(a)
    if a - r >= 0.5:  # exact: `a - r` is a float's fraction part
        r += 1
    return math.copysign(float(r), x)


def least_of(x: float, y: float) -> float:
    """The lesser of two V25 values, NaN when either is (an undefined value, refused as no finite
    size). Python's `min` answers by argument order with a NaN, and Rust's `f64::min` drops it:
    neither is the value, and the two checkers must judge one document alike."""
    return math.nan if math.isnan(x) or math.isnan(y) else min(x, y)


def most_of(x: float, y: float) -> float:
    """The greater of two V25 values, NaN when either is (`least_of`)."""
    return math.nan if math.isnan(x) or math.isnan(y) else max(x, y)


def integral(x: float) -> str:
    """A rounded value as the runner prints it in a message (`{:.0}` of the f64): every digit,
    never saturated at 64 bits nor in exponent form."""
    return f"{x:.0f}"


def _no_constant(name):
    raise ValueError(f"`{name}` is not JSON")


def _finite(text):
    x = float(text)
    if math.isinf(x):
        raise ValueError(f"number {text} is out of range")
    return x


def _no_duplicates(pairs):
    """An object whose key appears twice: serde_json refuses a repeated field of a struct
    ("duplicate field") and keeps the last of a repeated key of a map (`datasets`,
    `params`, a parameter file's top level); Python's `json` keeps the last of both. No
    writer of these files repeats a key, so every repeat is refused, the map's too."""
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"duplicate field `{k}`")
        seen[k] = v
    return seen


def _no_lone_surrogate(doc):
    """serde_json refuses a `\\u` escape of half a UTF-16 pair; Python's `json` keeps it as a
    lone surrogate in the string (a whole pair it joins into one character, so a surrogate
    left in a key or a value is a lone one)."""
    stack = [doc]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str) and (m := _SURROGATE.search(x)):
            raise ValueError(f"lone surrogate in hex escape (\\u{ord(m.group()):04x})")
    return doc


_SURROGATE = re.compile("[\ud800-\udfff]")


def loads(text):
    """JSON as the runner's serde_json reads it: Python's `json` also takes `NaN`, `Infinity`,
    and `-Infinity`, and turns a number past the 64-bit float range (`1e400`) into an
    infinity; serde_json refuses all four, and so does this, and a repeated key or a
    lone surrogate in a string as well."""
    return _no_lone_surrogate(json.loads(text, parse_constant=_no_constant, parse_float=_finite,
                                         object_pairs_hook=_no_duplicates))


def int_range(doc, path=()) -> list:
    """(path, message) for every integer outside the contract's range (§1): 64-bit signed,
    a dataset's or namespace's `seed` 64-bit unsigned. JSON integers are unbounded and Python
    keeps them exact; the runner's would parse as a float, so the AST would hash and run
    differently there. The runner's `canon::int_range_errors` has the same rule and wording.
    A non-finite float, which JSON cannot hold (`loads` refuses it in a file), is refused in an
    AST the builder made."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            out += int_range(v, path + (k,))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            out += int_range(v, path + (i,))
    elif isinstance(doc, int) and not isinstance(doc, bool):
        if len(path) == 3 and path[0] in ("datasets", "namespaces") and path[2] == "seed":
            if not 0 <= doc <= 2 ** 64 - 1:
                out.append((path, f"seed {doc} is outside 0 to 2^64-1"))
        elif not -2 ** 63 <= doc <= 2 ** 63 - 1:
            out.append((path, f"integer {doc} is outside the 64-bit signed range of the contract's integers "
                              f"(-2^63 to 2^63-1)"))
    elif isinstance(doc, float) and not math.isfinite(doc):
        out.append((path, f"{doc} is not a JSON number (NaN and the infinities have no JSON form)"))
    return out


def canonical(ast: dict) -> bytes:
    """Canonical form: sorted keys, no whitespace, ASCII escapes, provenance removed. Floats
    serialize with Python's repr (shortest round-trip), which is what the builder emits. The
    on-disk file is the same JSON pretty-printed; only these bytes are hashed."""
    stripped = {k: v for k, v in ast.items() if k != "provenance"}
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def param_refs(x, out: set):
    """V1: the parameters a value names with `param`, anywhere in it (a distribution's
    arguments, a mixture's arms, an expression's operands, a handle's expressions; not a
    namespace object's field names, which are names, not nodes). As the runner's `pvalue_refs`."""
    if isinstance(x, dict):
        if len(x) == 1 and isinstance(x.get("param"), str):
            out.add(x["param"])
            return
        for k, v in x.items():
            if k == "fields" and isinstance(v, dict):
                for f in v.values():
                    param_refs(f, out)
            else:
                param_refs(v, out)
    elif isinstance(x, list):
        for v in x:
            param_refs(v, out)


class Check:
    def __init__(self, ast: dict, name: str):
        self.ast, self.name, self.errors = ast, name, []
        self.params = set(ast.get("params", {}))
        self.datasets = ast.get("datasets", {})
        self.namespaces = ast.get("namespaces", {})
        self.ops = Counter()
        self.traces = False
        # inside a `parallel` or `loader` body: the handles opened so far in the innermost one (V24)
        self.fork_opens = None
        # V26: inside a `loader` body, the loader's name and the path of the body it is in
        self.loader = None
        # V26: an actor's channel declarations (`channel` and `loader`), by name: the path of
        # the body each is in, and whether a loop encloses it
        self.sites = {}
        # the `consume`s inside a loop, by dataset, in document order (V6)
        self.consumes = {}
        # V6: a `consume` under a loop whose count varies, by path: the innermost such loop's path
        self.consume_varying = {}
        # V27: the parameters whose distributions `drawkind` is following
        self.drawing = set()

    def param_value(self, name):
        """The value in effect of parameter `name` (None when it is not declared): its default
        when the document is judged; `aeiou-params check` overrides this with the value a
        parameter file gives, as the runner's `check_given` judges a run (`DESIGN_REVIEW.md` §3.57).
        None too for a required parameter (contract 0.11), and a storage-reported one's is its
        fallback."""
        d = self.ast.get("params", {}).get(name, {}).get("default")
        if isinstance(d, dict) and "from" in d:
            return d.get("fallback")
        return d

    def param_cycles(self) -> bool:
        """V1: a parameter whose value in effect names, through `param`s, the parameter itself
        is defined by nothing (and the rules that follow a `param` to its value, V3, V25, would
        never return). Each parameter on a cycle is refused with the shortest one, its names in
        sorted order at each step. True when there is one. As the runner's `param_cycles`."""
        declared = self.ast.get("params", {})

        def refs(name):
            out = set()
            param_refs(self.param_value(name), out)
            return sorted(n for n in out if n in declared)
        found = False
        for start in sorted(declared):
            prev, queue = {}, []
            for n in refs(start):
                prev.setdefault(n, start)
                queue.append(n)
            while queue:
                n = queue.pop(0)
                if n == start:
                    chain, cur = [start], prev[start]
                    while cur != start:
                        chain.append(cur)
                        cur = prev[cur]
                    chain.append(start)
                    self.err(["params", start], f"parameter `{start}` is defined through itself ({' -> '.join(reversed(chain))}) (V1)")
                    found = True
                    break
                for m in refs(n):
                    if m not in prev:
                        prev[m] = n
                        queue.append(m)
        return found

    def err(self, path, msg):
        self.errors.append(f"{self.name}: {'/'.join(map(str, path))}: {msg}")

    def layout_fields(self, d: dict, path: list):
        """V20 (the fields each layout takes) and V21 (access), contract 0.8."""
        layout = d["layout"]
        need, may = {"whole": ((), ()), "chunked": (("chunk",), ()), "compound": (("samples_per_file",), ("format",)),
                     "array": (("slot",), ())}[layout]
        for k in ("samples_per_file", "chunk", "slot", "format"):
            if k in need and k not in d:
                self.err(path, f"layout `{layout}` needs `{k}` (V20)")
            elif k in d and k not in need and k not in may:
                self.err(path + [k], f"layout `{layout}` takes no `{k}` (V20)")
        # a pattern the runner cannot parse is refused by `reserved`, with the runner's message,
        # and has no fields to judge here
        parsed = pattern_fields(d["pattern"]) if parses(d["pattern"]) else None
        fields = {f[0] for f in parsed or ()}
        if parsed is not None:
            if layout == "array" and fields:
                self.err(path + ["pattern"], "an `array` is one file: its pattern has no fields (V20)")
            elif layout == "chunked" and "k" not in fields:
                self.err(path + ["pattern"], "a `chunked` dataset's pattern needs a `{k}` field (V20)")
            elif layout in ("whole", "compound") and "k" in fields:
                self.err(path + ["pattern"], "`{k}` names a chunk: only a `chunked` dataset has one (V20)")
            elif layout != "array" and "id" not in fields:
                self.err(path + ["pattern"], "the pattern needs an `{id}` field (V20)")
            # the run formats a dataset's names from `id` and `k` alone
            for f in sorted(fields - {"id", "k"}):
                self.err(path + ["pattern"], f"`{{{f}}}` is no field of a dataset: its names are formatted from `{{id}}` and, `chunked`, `{{k}}` (V20)")
        if layout == "array" and d["access"] != "indexed":
            self.err(path + ["access"], "an `array` is `indexed`: its slots are looked up by position (V21)")
        # the `id` fields of the directory part, as the runner reads them (`{ id  mod 16}` too)
        if d["access"] in ("map", "stream") and any(f[0] == "id" and f[1] == "mod" and f[3] for f in parsed or ()):
            self.err(path + ["pattern"], "a consumed dataset's directories are `{id div D}` or none, never `mod`, so a subset of it is whole directories (V21)")

    def inventory(self, d: dict, path: list):
        """V22 (contract 0.9, DESIGN_REVIEW.md §3.73): a dataset's inventory is a file directly in
        its root, of a consumed dataset, named so the pattern cannot produce it."""
        inv = d.get("inventory")
        if inv is None:
            return
        ip = path + ["inventory"]
        if d["access"] not in ("map", "stream"):
            self.err(ip, "only a consumed dataset (`map` or `stream`) has an inventory: it stands in for the walk of a corpus the job draws from (V22)")
        n = inv["name"]
        root = dataset_root(d)
        sibling = (d["pattern"][len(root) + 1:] if root else d["pattern"]).split("/")[0]
        glob = "[0-9a-f -]+".join(re.escape(s) for s in re.split(r"\{[^}]*\}", sibling))
        if n in ("", ".", "..") or any(c in n for c in "/{}"):
            self.err(ip + ["name"], f"`{n}` is not a file name: the inventory lies directly in the dataset's root, its name a constant (V22)")
        elif n.startswith(RESERVED_PREFIX):
            self.err(ip + ["name"], f"`{n}` begins with `{RESERVED_PREFIX}`, reserved for the manifest (V13)")
        elif parses(d["pattern"]) and re.fullmatch(glob, n):
            self.err(ip + ["name"], f"`{n}` is a name the pattern `{d['pattern']}` can produce in the dataset's root (V22)")

    # ---- entry ----
    def run(self):
        for path, msg in int_range(self.ast):
            self.err(list(path), msg)
        if "gpus" in self.params:
            self.err(["params"], "`gpus` is reserved and set by the runner")
        for pname, p in self.ast.get("params", {}).items():
            self.value(self.param_value(pname), ["params", pname])
            # V28 (contract 0.11): a storage-reported value is read from a declared dataset's storage
            src = p.get("default") if isinstance(p, dict) else None
            if isinstance(src, dict) and "from" in src:
                if src.get("of") not in self.datasets:
                    self.err(["params", pname, "default", "of"], f"`{src.get('of')}` is not a declared dataset (V28)")
                if not (isinstance(src.get("fallback"), int) and src["fallback"] >= 1):
                    self.err(["params", pname, "default", "fallback"], f"{src.get('fallback')} is not a positive byte count (V28)")
        # every rule after this one follows a `param` to its value; through a cycle it would
        # never return, so a document with one is judged no further
        if self.param_cycles():
            return self.errors
        roots = {}
        for dname, d in self.datasets.items():
            body = d
            self.layout_fields(body, ["datasets", dname])
            self.inventory(body, ["datasets", dname])
            self.exprlike(body["size"], ["datasets", dname, "size"], Scope(self), dist_ok=True)
            self.drawn(body["size"], ["datasets", dname, "size"], Scope(self))
            if body.get("layout") != "array":
                self.size_not_negative(body["size"], ["datasets", dname, "size"])
            self.expr(body["count"], ["datasets", dname, "count"], Scope(self))
            self.typed(body["count"], ["datasets", dname, "count"], Scope(self))
            for k in ("samples_per_file", "chunk", "slot"):
                if k in body:
                    self.expr(body[k], ["datasets", dname, k], Scope(self))
                    self.typed(body[k], ["datasets", dname, k], Scope(self), optional=True)
            if "inventory" in body:
                self.expr(body["inventory"]["entry"], ["datasets", dname, "inventory", "entry"], Scope(self))
                self.typed(body["inventory"]["entry"], ["datasets", dname, "inventory", "entry"], Scope(self))
            framing = body.get("format", {}).get("framing")
            if framing:
                lp = ["datasets", dname, "format", "framing"]
                for k, v in framing.items():
                    if k == "columns":
                        weights = [c.get("weight", 0) for c in v]
                        if any(w < 0 for w in weights):
                            self.err(lp + ["columns"], "negative weight")
                        elif abs(sum(weights) - 1) > 1e-9:
                            self.err(lp + ["columns"], f"weights sum to {sum(weights)}, not 1: the sample's bytes must land somewhere once")
                        for i, c in enumerate(v):
                            for ck, cv in c.items():
                                if ck != "weight":
                                    self.expr(cv, lp + ["columns", i, ck], Scope(self))
                                    self.typed(cv, lp + ["columns", i, ck], Scope(self), optional=True)
                    elif k != "writer":
                        self.expr(v, lp + [k], Scope(self))
                        self.typed(v, lp + [k], Scope(self), optional=True)
            self.reserved(body["pattern"], ["datasets", dname])
            root = dataset_root(d)
            if root in roots:
                self.err(["datasets", dname], f"shares root `{root}/` with dataset `{roots[root]}` (V13)")
            for other, owner in roots.items():
                inner, outer = (root, other) if len(root) > len(other) else (other, root)
                if inner != outer and (outer == "" or inner.startswith(outer + "/")):
                    self.err(["datasets", dname], f"root `{root}/` and root `{other}/` of dataset `{owner}` are nested (V13)")
            roots[root] = dname
        nroots = {}
        for nname, n in self.namespaces.items():
            if n["size"] != "as_written":
                self.expr(n["size"], ["namespaces", nname, "size"], Scope(self, fields=set(n["fields"])))
                self.typed(n["size"], ["namespaces", nname, "size"], Scope(self, fields=set(n["fields"])))
            self.reserved(n["pattern"], ["namespaces", nname])
            nroot = n["pattern"].split("{", 1)[0].rpartition("/")[0]
            for root, dname in roots.items():
                if root and (nroot == root or nroot.startswith(root + "/")):
                    self.err(["namespaces", nname], f"root `{nroot}/` lies inside dataset `{dname}`'s root `{root}/` (V13)")
            # V14: namespaces sharing a root share its manifest, so they agree on `input`
            inp = bool(n.get("input", False))
            if nroot in nroots and nroots[nroot][1] != inp:
                self.err(["namespaces", nname], f"shares root `{nroot}/` with namespace `{nroots[nroot][0]}` but `input` differs (V14)")
            # V17: and a root is in one place, so they agree on `protocol`
            proto = n.get("protocol", "posix")
            if nroot in nroots and nroots[nroot][2] != proto:
                self.err(["namespaces", nname], f"shares root `{nroot}/` with namespace `{nroots[nroot][0]}` but `protocol` differs (V17)")
            nroots.setdefault(nroot, (nname, inp, proto))
            # V15: `same_run` compares this run with the writer's manifest, which only an input has
            if n.get("same_run") and not inp:
                self.err(["namespaces", nname], "`same_run` without `input`: only an input namespace has a writer to compare with (V15)")
        for aname, a in self.ast["actors"].items():
            scope = Scope(self)
            if "count" in a:
                self.expr(a["count"], ["actors", aname, "count"], scope)
                self.typed(a["count"], ["actors", aname, "count"], scope, optional=True)
            self.sites = {}
            self.channel_sites(a["body"], ["actors", aname, "body"], False)
            self.body(a["body"], ["actors", aname, "body"], scope)
        self.v6_one_consume()
        self.v19()
        return self.errors

    def v6_one_consume(self):
        """V6: a dataset is consumed in one place. A `consume` has no site: its sample is a
        function of the actor's number and the ordinals of its enclosing frames alone
        (runner/REFERENCE.md §2), so two `consume`s of one dataset (in one body, in sibling
        loops, or in two actors) meet at the same positions and return the same samples.
        A `consume` under a loop whose count varies with an enclosing loop repeats positions alike:
        the ordinal assumes fixed counts. Reported in the runner's order: datasets by name, actors
        by name, then document order."""
        for ds in sorted(self.consumes):
            paths = sorted(self.consumes[ds], key=lambda q: q[1])  # stable: document order within an actor
            first = "/".join(map(str, paths[0]))
            for i, q in enumerate(paths):
                loop = self.consume_varying.get(tuple(q))
                if loop is not None:
                    self.err(q, f"a `consume` of `{ds}` under a loop whose count varies with an enclosing loop "
                                f"({'/'.join(map(str, loop))}): its position is the row-major ordinal of the "
                                "enclosing loops at fixed counts, so a count that varies repeats positions and draws "
                                "the same samples, a reuse the job does not have; give the loop a fixed count (the "
                                "most it can be, with a `cond`, the builder's `when`, inside to skip the rest) (V6)")
                if i == 0:
                    continue
                self.err(q, f"a second `consume` of `{ds}` (the first is at {first}): `consume` has no site, so its sample "
                            "depends only on the actor and the enclosing loop indices, and two of them draw the same samples, "
                            "a reuse the job does not have; consume once and use the handle in each place, or put the one "
                            "`consume` in an inner loop so each draw has its own position (V6)")

    def v19(self):
        """V19: an API is declared only for a protocol the abstract has names on, and an
        abstract with both declares APIs of one scheduling (event-driven or a thread per actor),
        defaults counted."""
        posix, s3 = protocols(self.ast, self.traces)
        if "s3" in self.ast and not s3:
            self.err(["s3"], "declared, but no dataset or namespace is `protocol: s3` (V19)")
        for key in ("posix", "cache"):
            if key in self.ast and not posix:
                self.err([key], "declared, but every dataset and namespace is `protocol: s3` (V19)")
        if posix and s3:
            api, s = self.ast.get("posix", "sync"), self.ast.get("s3", "blocking")
            if (api in ("io_uring", "libaio")) != (s == "async"):
                self.err(["s3" if "s3" in self.ast else "posix"],
                         f"posix `{api}` and s3 `{s}`: one is event-driven and the other a thread per actor; "
                         "an abstract with both protocols declares APIs of one kind (V19)")

    def reserved(self, pattern, path):
        for comp in pattern.split("/"):
            if comp.startswith(RESERVED_PREFIX):
                self.err(path, f"path component `{comp}` begins with `{RESERVED_PREFIX}`, reserved for the manifest (V13)")
        # the pattern as the runner parses it: a field it cannot read is refused with its message
        try:
            fields = pattern_fields(pattern)
        except ValueError as e:
            return self.err(path + ["pattern"], str(e))
        # V23 (contract 0.10, DESIGN_REVIEW.md §3.75): a divisor named in a pattern is a declared
        # parameter whose kind is an integer (its default's, or, required, its `kind`; 0.11)
        for name in [f[2] for f in fields if f[2] is not None]:
            p = self.ast.get("params", {}).get(name)
            if p is None:
                self.err(path + ["pattern"], f"divisor `{name}` is not a declared parameter (V23)")
            elif param_kind(p) != "int":
                self.err(path + ["pattern"], f"divisor `{name}` is a parameter whose kind is not an integer (V23)")

    # ---- V27: the kind a value is known to have ----
    KINDS = {"int": "an integer", "float": "a float", "bool": "a bool", "str": "a string", "none": "null",
             "dist": "a distribution", "array": "an array", "handle": "a handle"}
    # where the runner needs an integer or a number it reads a float, rounded, and a bool as 0 or 1
    NOT_NUMBER = {"str", "none", "dist", "array", "handle"}
    # an operand of arithmetic is an integer or a float
    NOT_OPERAND = NOT_NUMBER | {"bool"}
    # a condition is a bool or an integer
    NOT_CONDITION = NOT_NUMBER | {"float"}

    @staticmethod
    def pkind(v):
        """The kind of a parameter value."""
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        if v is None:
            return "none"
        return "array" if isinstance(v, list) else "dist"

    def kind(self, e, scope):
        """V27: the kind (`KINDS`) a value, distribution, or handle is known to have where the
        document shows it: a literal, a parameter by its value in effect, an element of a
        parameter array whose elements are of one kind, a binding by its value, a handle, a
        distribution, an expression whose result is an integer or a bool whatever its operands, a
        `cond` whose arms are of one kind. None when it is not known before the run."""
        if not isinstance(e, dict):
            return self.pkind(e)
        if len(e) != 1:
            return None
        k, a = next(iter(e.items()))
        if k in ("ref", "index") and a in scope.fields:
            return None
        if k == "ref":
            return scope.kind_of(a)
        if k == "at":
            return None if a["ref"] in scope.forward else scope.kind_of(a["ref"])
        if k == "param":
            if a in self.params and "default" not in self.ast["params"][a] and self.param_value(a) is None:
                return self.ast["params"][a].get("kind")       # required, no value yet (contract 0.11)
            return self.pkind(self.param_value(a)) if a in self.params else None
        if k == "elem":
            v = self.param_value(a["array"]) if a["array"] in self.params else None
            if v is None and a["array"] in self.params and "default" not in self.ast["params"][a["array"]]:
                return self.ast["params"][a["array"]].get("items")
            kinds = {self.pkind(x) for x in v} if isinstance(v, list) else set()
            return kinds.pop() if len(kinds) == 1 else None
        if k in ("index", "actor", "count", "dirs", "len", "size", "offset", "unit_index", "units", "chunks"):
            return "int"
        if k in ("eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"):
            return "bool"
        if k == "cond":
            t = self.kind(a["then"], scope)
            return t if t == self.kind(a["else"], scope) else None
        if k == "draw":
            return self.drawkind(a, scope)
        if k in SCHEMA["$defs"]["handle"]["properties"]:
            return "handle"
        if k in SCHEMA["$defs"]["dist"]["properties"]:
            return "dist"
        return None

    def drawkind(self, d, scope):
        """V27: the kind a draw of `d` (a distribution, or an expression) yields where the
        document shows it: a `const` its argument's kind; `uniform`, `uniform64`, `normal`,
        `lognormal`, `zipf`, `hotset` an integer; an `empirical` whose values are of one kind
        that kind; a `mixture` whose arms yield one kind that kind (a `null` arm null); a
        parameter whose value is a distribution what that distribution yields; a `cond` whose
        arms yield one kind that kind; any other expression its own kind, drawn as a constant.
        None when it is not known before the run. As the runner's `draw_kind`."""
        if isinstance(d, dict) and len(d) == 1:
            k, a = next(iter(d.items()))
            if k in SCHEMA["$defs"]["dist"]["properties"]:
                if k == "const":
                    return self.kind(a, scope)
                if k == "empirical":
                    kinds = {self.pkind(v) for v in a["values"]}
                elif k == "mixture":
                    kinds = {"none" if arm["dist"] is None else self.drawkind(arm["dist"], scope) for arm in a}
                else:
                    return "int"
                return kinds.pop() if len(kinds) == 1 else None
            if k == "param" and a in self.params and isinstance(self.param_value(a), dict):
                # a parameter defined through itself is refused by V1 (after the parameters'
                # values are walked, so this walk may meet one first)
                if a in self.drawing:
                    return None
                self.drawing.add(a)
                try:
                    # a parameter's distribution is judged where it is declared, no binding in scope
                    return self.drawkind(self.param_value(a), Scope(self))
                finally:
                    self.drawing.discard(a)
            if k == "cond":
                t = self.drawkind(a["then"], scope)
                return t if t == self.drawkind(a["else"], scope) else None
        k = self.kind(d, scope)
        return None if k == "dist" else k

    def typed(self, e, p, scope, want="an integer", refuse=NOT_NUMBER, optional=False):
        """V27: a value the runner needs as `want` is not of a kind it refuses there. A `null`
        where a key is `optional` is the key left out, as the runner reads it."""
        if optional and e is None:
            return
        k = self.kind(e, scope)
        if k in refuse:
            self.err(p, f"expected {want}, got {self.KINDS[k]} (V27)")

    def drawn(self, d, p, scope, optional=False):
        """V27: a size or a pick's distribution is drawn (an expression that is not a
        distribution as a constant) and what it yields is read as an integer."""
        if optional and d is None:
            return
        k = self.drawkind(d, scope)
        if k in self.NOT_NUMBER:
            self.err(p, f"expected an integer, got {self.KINDS[k]} (V27)")

    def value(self, v, path):
        if isinstance(v, list):
            for i, x in enumerate(v):
                self.value(x, path + [i])
        elif isinstance(v, dict):
            self.dist(v, path, Scope(self))

    # ---- statements ----
    def body(self, nodes, path, scope):
        scope = scope.child()
        # `at` may forward-reference a binding defined later in the same loop body
        lets = [n["let"]["name"] for n in nodes if "let" in n]
        scope.forward = set(lets)
        scope.twice = {x for x in lets if lets.count(x) > 1}
        for i, node in enumerate(nodes):
            kind, args = next(iter(node.items()))
            p = path + [i, kind]
            if kind not in CONTROL:
                self.ops[kind] += 1
            if kind == "open":
                self.closed_later(args, nodes[i + 1:], p, scope)
            getattr(self, "n_" + kind, self.n_op)(args, p, scope)

    # V18: an object is made by the `close` that sends its upload, so an s3 name opened for
    # writing is closed by a later op of the same body, on the same handle, before an `open`
    # of it again or a `let` of a name it reads: a close in a nested body may not run, and an
    # upload never closed would be discarded
    def closed_later(self, a, rest, p, scope):
        if "WRONLY" not in a["flags"]:
            return
        ns = self.object_store_namespace(a["file"], scope)
        if not ns:
            return
        h = json.dumps(a["file"], sort_keys=True)
        for node in rest:
            kind, b = next(iter(node.items()))
            if kind == "close" and json.dumps(b["file"], sort_keys=True) == h:
                return
            if (kind == "open" and json.dumps(b["file"], sort_keys=True) == h) or (kind == "let" and json.dumps({"ref": b["name"]})[1:-1] in h):
                break
        self.err(p + ["file"], f"open for writing on namespace `{ns}`, declared `protocol: s3`, with no close of that handle later in the same body (before it is opened again or a name it reads is bound again): an object is made by one upload that its close sends, so an upload never closed would be discarded (V18)")

    def n_let(self, a, p, scope):
        self.exprlike(a["value"], p + ["value"], scope, dist_ok=False, binding=a["name"])
        # an alias (`{ref: g}`) is bound to what `g` is here, as the run binds it: the rules that
        # follow a binding to its handle or distribution (V3, V12, V14, V18) read through it, and
        # a later `let` of `g` (`let g = {ref: f}`) changes nothing, nor makes a cycle to follow
        v = a["value"]
        if isinstance(v, dict) and set(v) == {"ref"} and scope.lookup(v["ref"]) is not None:
            v = scope.lookup(v["ref"])
        scope.bind(a["name"], v, self.varies(a["value"], scope), self.kind(a["value"], scope), self.may_name(a["value"], scope))

    def n_loop(self, a, p, scope):
        for k in ("from", "to", "step"):
            if k in a:
                self.expr(a[k], p + [k], scope)
                self.typed(a[k], p + [k], scope, optional=k != "to")
        nonneg = "from" not in a or self.at_least(a["from"], 0, scope)
        varies = any(self.varies(a[k], scope) for k in ("from", "to", "step") if k in a)
        self.body(a["body"], p + ["body"], scope.with_index(a["index"], nonneg, p if varies else None))

    def n_parallel(self, a, p, scope):
        self.expr(a["width"], p + ["width"], scope)
        self.typed(a["width"], p + ["width"], scope)
        varies = self.varies(a["width"], scope)
        self.fork_body(a["body"], p + ["body"], scope.with_index(a["index"], True, p if varies else None))

    # V26: a channel is the actor's, shared by its sub-actors, so one declared in a sub-actor
    # would be declared by each of them, and which declaration replaced which depends on timing
    def declared_outside_forks(self, kind, p):
        if self.fork_opens is not None:
            self.err(p, f"a `{kind}` inside a `parallel` or `loader` body: its channel is the actor's, shared by every sub-actor, so each sub-actor would declare it again over the others' and what it holds would depend on timing; declare it before the fork (V26)")

    def channel_sites(self, nodes, path, in_loop):
        for i, node in enumerate(nodes):
            kind, a = next(iter(node.items()))
            p = path + [i, kind]
            if kind in ("channel", "loader"):
                self.sites.setdefault(a["name"], []).append((path, in_loop))
            if kind == "cond":
                self.channel_sites(a["then"], p + ["then"], in_loop)
                self.channel_sites(a.get("else", []), p + ["else"], in_loop)
            elif kind == "choose":
                for j, arm in enumerate(a["arms"]):
                    self.channel_sites(arm["body"], p + ["arms", j, "body"], in_loop)
            elif kind in ("loop", "parallel", "loader", "phase"):
                self.channel_sites(a["body"], p + ["body"], in_loop or kind == "loop")

    # V26: a loader's workers run beside the actor, so a channel they put or take on that the
    # actor declares again while they run (in a loop the loader is not declared again in, or
    # at a second site) would be the old one or the new one as timing had it
    # A `take` there is refused outright (2026-10-15): the workers take beside each other and
    # the actor, so which of them got which item, and whether the run finished, was timing
    def used_by_workers(self, kind, channel, p):
        if self.loader is not None and kind == "take":
            self.err(p, f"a `take` on channel `{channel}` inside a `loader` body: the loader's workers run beside each other and the actor, so which of them took which item, and whether the run finished, would depend on timing; take in the actor, a loader's body only puts (V26)")
            return
        sites = self.sites.get(channel)
        if self.loader is None or not sites:
            return
        lname, lbody = self.loader
        if len(sites) == 1 and (not sites[0][1] or (sites[0][0] == lbody and len(self.sites[lname]) == 1)):
            return
        self.err(p, f"a `{kind}` on channel `{channel}` inside a `loader` body: the loader's workers run beside the actor, so the channel is declared at one site, outside every loop or next to a loader declared at one site; declared again while they run, which declaration a worker reached would depend on timing (V26)")

    def n_loader(self, a, p, scope):
        for k in ("workers", "prefetch", "batches"):
            self.expr(a[k], p + [k], scope)
            self.typed(a[k], p + [k], scope)
        self.declared_outside_forks("loader", p)
        scope.channels.add(a["name"])
        varies = self.varies(a["batches"], scope)
        outer, self.loader = self.loader, (a["name"], p[:-2])
        self.fork_body(a["body"], p + ["body"], scope.with_index(a["index"], True, p if varies else None))
        self.loader = outer

    # V6: the kinds that draw (keyed on the enclosing indices), and those fixed for an actor
    DRAWS = {"draw", "consume", "pick", "uniform", "uniform64", "normal", "lognormal", "empirical",
             "zipf", "hotset", "mixture"}
    INVARIANT = {"param", "actor", "len", "sum", "count", "dirs", "inventory"}

    def varies(self, e, scope):
        """V6: whether an expression, distribution, or handle can take another value at another
        iteration of an enclosing frame: it names a loop index (or `at`), a binding that varies,
        or draws inside a frame (a draw is keyed on the enclosing indices). Parameters and the
        actor number are fixed for an actor's walk."""
        if isinstance(e, list):
            return any(self.varies(x, scope) for x in e)
        if not isinstance(e, dict):
            return False
        k, a = next(iter(e.items()))
        if k in ("index", "at"):
            return True
        if k == "ref":
            return scope.is_varying(a)
        if k in self.DRAWS:
            return bool(scope.indices)
        if k in self.INVARIANT:
            return False
        if k == "elem":
            return self.varies(a["index"], scope)
        if k in ("file", "dir", "unit", "column", "cond"):
            return any(self.varies(v, scope) for v in a.values())
        if k == "object":
            return any(self.varies(v, scope) for v in a["fields"].values())
        return self.varies(a, scope)

    def fork_body(self, nodes, path, scope):
        outer, self.fork_opens = self.fork_opens, []
        self.body(nodes, path, scope)
        self.fork_opens = outer

    def n_channel(self, a, p, scope):
        self.expr(a["capacity"], p + ["capacity"], scope)
        self.typed(a["capacity"], p + ["capacity"], scope)
        self.declared_outside_forks("channel", p)
        scope.channels.add(a["name"])

    def n_put(self, a, p, scope):
        if a["channel"] not in scope.channels:
            self.err(p, f"channel `{a['channel']}` not declared")
        self.used_by_workers("put", a["channel"], p)
        self.expr(a["seq"], p + ["seq"], scope)
        self.typed(a["seq"], p + ["seq"], scope)

    def n_take(self, a, p, scope):
        if a["channel"] not in scope.channels:
            self.err(p, f"channel `{a['channel']}` not declared")
        self.used_by_workers("take", a["channel"], p)

    def n_barrier(self, a, p, scope):
        pass

    def n_compute(self, a, p, scope):
        self.expr(a["ns"], p + ["ns"], scope)
        self.typed(a["ns"], p + ["ns"], scope)

    def n_cond(self, a, p, scope):
        self.expr(a["if"], p + ["if"], scope)
        self.typed(a["if"], p + ["if"], scope, "a condition", self.NOT_CONDITION)
        self.body(a["then"], p + ["then"], scope)
        if "else" in a:
            self.body(a["else"], p + ["else"], scope)

    def n_choose(self, a, p, scope):
        for i, arm in enumerate(a["arms"]):
            self.body(arm["body"], p + ["arms", i, "body"], scope)

    def n_phase(self, a, p, scope):
        self.expr(a["name"], p + ["name"], scope)
        self.body(a["body"], p + ["body"], scope)

    def n_trace(self, a, p, scope):
        self.traces = True

    def n_op(self, a, p, scope):
        kind = p[-1]
        for key in ("file", "dir", "from", "to"):
            if key in a:
                self.handle(a[key], p + [key], scope)
        for key in ("len", "offset"):
            if key in a:
                self.expr(a[key], p + [key], scope)
                self.typed(a[key], p + [key], scope, optional=key == "offset" or kind == "fadvise")
        rep = a.get("repeat")
        if rep not in (None, "until_eof", "until_end"):
            self.expr(rep, p + ["repeat"], scope)
            self.typed(rep, p + ["repeat"], scope)
        if kind == "write":
            self.note_write(a["file"], scope)
        # V24: a sub-actor shares the descriptors it inherits, and with them the kernel's file
        # position, so an op at that position on one is timing-dependent
        if self.fork_opens is not None:
            if kind == "open":
                self.fork_opens.append(json.dumps(a["file"], sort_keys=True))
            elif (kind in ("read", "write") and "offset" not in a) or kind == "lseek":
                if json.dumps(a["file"], sort_keys=True) not in self.fork_opens:
                    self.err(p + ["file"], f"{kind} that uses the file position of a handle this `parallel` or `loader` body did not open: its sub-actors share the descriptor and its position, so the offsets would depend on timing; give an `offset`, or open the handle in the body (V24)")
        if kind == "read" and rep == "until_eof":
            self.check_until_eof(a["file"], p, scope)
        # V12: datasets are read-only
        if kind in MUTATING_OPS or kind in ("rename", "mkdir", "rmdir"):
            for key in ("file", "dir", "from", "to"):
                if key in a and self.dataset_of(a[key], scope):
                    self.err(p + [key], f"{kind} on dataset `{self.dataset_of(a[key], scope)}`: datasets are read-only (V12)")
        if kind == "open" and set(a["flags"]) & WRITE_FLAGS and self.dataset_of(a["file"], scope):
            self.err(p + ["flags"], f"open for writing on dataset `{self.dataset_of(a['file'], scope)}`: datasets are read-only (V12)")
        # V14: input namespaces are read-only
        if kind in MUTATING_OPS or kind in ("rename", "mkdir", "rmdir"):
            for key in ("file", "dir", "from", "to"):
                if key in a:
                    ns = self.input_namespace_of(a[key], scope)
                    if ns:
                        self.err(p + [key], f"{kind} on input namespace `{ns}`: input namespaces are read-only (V14)")
        if kind == "open" and set(a["flags"]) & WRITE_FLAGS and self.input_namespace_of(a["file"], scope):
            self.err(p + ["flags"], f"open for writing on input namespace `{self.input_namespace_of(a['file'], scope)}`: input namespaces are read-only (V14)")
        # V18: an object in an object store is written once, in order from 0, by one upload
        v18 = None
        if kind == "open":
            bad = [f for f in a["flags"] if f in ("APPEND", "RDWR", "EXCL")]
            if bad:
                v18 = ("flags", f"open with {bad[0]}")
            elif "WRONLY" not in a["flags"]:
                # an upload is begun by WRONLY alone: CREAT or TRUNC on a read-only handle
                # would make or empty an object the handle never writes
                bad = [f for f in a["flags"] if f in ("CREAT", "TRUNC")]
                if bad:
                    v18 = ("flags", f"open with {bad[0]} without WRONLY")
        elif kind in ("ftruncate", "fallocate"):
            v18 = ("file", kind)
        if v18:
            ns = self.object_store_namespace(a["file"], scope)
            if ns:
                why = "an object is made only by the upload an open with WRONLY begins, and a read-only handle writes none" if v18[1].endswith("without WRONLY") else f"an object is written once, in order from 0, and has no {v18[1]}"
                self.err(p + [v18[0]], f"{v18[1]} on namespace `{ns}`, declared `protocol: s3`: {why} (V18)")

    def object_store_namespace(self, h, scope):
        """Namespace name if the handle is (a binding to) an object of a namespace declared `protocol: s3`."""
        ns = self.object_namespace(h, scope) if isinstance(h, dict) else None
        if ns and self.namespaces.get(ns, {}).get("protocol") == "s3":
            return ns
        return None

    def input_namespace_of(self, h, scope):
        """Namespace name if the handle is (a binding that may be) an object of an `input` namespace."""
        if not isinstance(h, dict) or len(h) != 1:
            return None
        k, a = next(iter(h.items()))
        if k == "ref":
            return scope.reads_of(a)[1]
        if k == "object" and self.namespaces.get(a["namespace"], {}).get("input"):
            return a["namespace"]
        return None

    def dataset_of(self, h, scope):
        """Dataset name if the handle is (a binding that may be) a dataset file, dir, sample,
        region, or chunk."""
        if not isinstance(h, dict) or len(h) != 1:
            return None
        k, a = next(iter(h.items()))
        if k == "ref":
            return scope.reads_of(a)[0]
        if k == "file":
            return a["dataset"] if "dataset" in a else self.dataset_of(a["of"], scope)
        if k in ("dir",):
            return a["dataset"]
        if k in ("consume", "inventory"):
            return a
        if k == "pick":
            return a["dataset"]
        if k in ("unit", "column"):
            return self.dataset_of(a["of"], scope)
        return None

    def may_name(self, v, scope):
        """V12, V14: the dataset and the `input` namespace a `let` value may name, so a write
        through the binding is refused if any handle it may hold is read-only: a handle, an
        alias or an `at` of a binding by what that binding may name, a `cond` by either arm, the
        `then` arm's first (RC1 review: a handle chosen by `cond` from a dataset handle was
        written through unrefused). As the runner's `may_name`."""
        if not isinstance(v, dict) or len(v) != 1:
            return (None, None)
        k, a = next(iter(v.items()))
        if k == "cond":
            t, e = self.may_name(a["then"], scope), self.may_name(a["else"], scope)
            return (t[0] or e[0], t[1] or e[1])
        if k == "at":
            return (None, None) if a["ref"] in scope.forward else scope.reads_of(a["ref"])
        return (self.dataset_of(v, scope), self.input_namespace_of(v, scope))

    # ---- the as_written rule (ABSTRACTS.md §9.7) ----
    def object_namespace(self, h, scope):
        """Namespace name if the handle is (a binding to) a namespace object, else None."""
        if "ref" in h:
            v = scope.lookup(h["ref"])
            return self.object_namespace(v, scope) if isinstance(v, dict) else None
        if "object" in h:
            return h["object"]["namespace"]
        return None

    def note_write(self, h, scope):
        if "ref" in h and self.object_namespace(h, scope):
            scope.written.add(h["ref"])

    def check_until_eof(self, h, p, scope):
        ns = self.object_namespace(h, scope)
        if ns and self.namespaces[ns]["size"] == "as_written":
            if not ("ref" in h and h["ref"] in scope.written):
                self.err(p, "until_eof on an as_written object needs the handle binding the "
                            "creating writes used (same position)")

    # ---- expressions ----
    def exprlike(self, v, p, scope, dist_ok, binding=None):
        if isinstance(v, dict) and len(v) == 1:
            k = next(iter(v))
            # a `ref` is an expression first, as the runner reads it
            if k in SCHEMA["$defs"]["handle"]["properties"] and k != "ref":
                return self.handle(v, p, scope)
            if k in SCHEMA["$defs"]["dist"]["properties"]:
                return self.dist(v, p, scope)
        self.expr(v, p, scope, binding)

    def dist(self, d, p, scope):
        k, a = next(iter(d.items()))
        if k == "empirical" and len(a["values"]) != len(a["weights"]):
            self.err(p, "empirical: values and weights differ in length")
        elif k == "mixture":
            for i, arm in enumerate(a):
                if arm["dist"] is not None:
                    self.distref(arm["dist"], p + [i, "dist"], scope)
        elif k == "const":
            # before the argument maps below: a `const` of an expression node is that expression
            self.expr(a, p, scope)
        elif isinstance(a, dict):
            for kk, vv in a.items():
                if isinstance(vv, (dict, list)) or kk in ("lo", "hi", "mean", "sd", "median", "min", "max"):
                    self.expr(vv, p + [kk], scope)
                if kk in ("lo", "hi", "mean", "sd", "median", "min", "max"):
                    self.typed(vv, p + [kk], scope, "an integer" if k == "uniform" else "a number", optional=kk in ("min", "max"))
            if k in ("normal", "lognormal"):
                # clamped to `min` and then to `max`: every draw `max`, and a `min` rule V3 would believe.
                # Compared as the runner clamps, in f64: 2^53 + 1 is not above 2^53 there.
                lo, hi = self.number(a.get("min")), self.number(a.get("max"))
                if lo is not None and hi is not None and float(lo) > float(hi):
                    self.err(p, f"{k}: min {lo} is above max {hi}")

    def distref(self, v, p, scope):
        if isinstance(v, dict) and next(iter(v)) in SCHEMA["$defs"]["dist"]["properties"]:
            self.dist(v, p, scope)
        else:
            self.expr(v, p, scope)

    def expr(self, e, p, scope, binding=None):
        if not isinstance(e, dict):
            return
        k, a = next(iter(e.items()))
        if k == "param":
            if a not in self.params:
                self.err(p, f"unknown param `{a}`")
        elif k == "index":
            if a not in scope.indices and a not in scope.fields:
                self.err(p, f"index `{a}` not in scope")
        elif k == "ref":
            if scope.lookup(a) is None and a not in scope.fields:
                self.err(p, f"binding `{a}` not in scope")
        elif k == "at":
            self.at(a, p, scope, binding)
        elif k == "draw":
            self.distref(a, p, scope)
        elif k in ("elem",):
            if a["array"] not in self.params:
                self.err(p, f"unknown parameter array `{a['array']}`")
            else:
                self.array(a["array"], "elem", p)
            self.expr(a["index"], p + ["index"], scope)
            self.typed(a["index"], p + ["index"], scope)
        elif k in ("len", "sum"):
            if a not in self.params:
                self.err(p, f"unknown parameter array `{a}`")
            else:
                self.array(a, k, p, numbers=k == "sum")
        elif k in ("size", "offset", "unit_index", "units", "chunks"):
            self.handle(a, p, scope)
        elif k == "count":
            if a not in self.datasets:
                self.err(p, f"unknown dataset `{a}`")
        elif k == "dirs":
            if a not in self.datasets or self.datasets[a]["layout"] == "array":
                self.err(p, f"`dirs` needs a dataset named by ids, not an `array`: got `{a}`")
        elif k == "cond":
            self.expr(a["if"], p + ["if"], scope)
            self.typed(a["if"], p + ["if"], scope, "a condition", self.NOT_CONDITION)
            self.exprlike(a["then"], p + ["then"], scope, dist_ok=True, binding=binding)
            self.exprlike(a["else"], p + ["else"], scope, dist_ok=True, binding=binding)
        elif isinstance(a, list):
            for i, x in enumerate(a):
                self.expr(x, p + [i], scope)
                self.operand(k, x, p + [i], scope)
            if k in self.COMPARISONS and len(a) == 2:
                self.compared(k, a, p, scope)
        elif k in ("neg", "not"):
            self.expr(a, p, scope)
            self.operand(k, a, p, scope)
        elif k == "actor":
            pass
        else:
            self.err(p, f"unhandled expression kind `{k}`")

    ARITHMETIC = {"add", "sub", "mul", "div", "mod", "ceil_div", "min", "max", "neg"}

    def operand(self, k, x, p, scope):
        """V27: an operand of arithmetic is a number, of `and`, `or`, `not` a condition."""
        if k in self.ARITHMETIC:
            self.typed(x, p, scope, "a number", self.NOT_OPERAND)
        elif k in ("and", "or", "not"):
            self.typed(x, p, scope, "a condition", self.NOT_CONDITION)

    COMPARISONS = {"eq", "ne", "lt", "le", "gt", "ge"}

    def compared(self, k, a, p, scope):
        """V27: the runner compares two numbers, two strings, two bools, two handles, or `null`
        with anything (unequal unless both are); any other pair of kinds the document shows is
        refused, as the run would. As the runner's `compared`."""
        x, y = self.kind(a[0], scope), self.kind(a[1], scope)
        if x is None or y is None or "none" in (x, y):
            return
        if {x, y} <= {"int", "float"} or (x == y and x in ("str", "bool", "handle")):
            return
        self.err(p, f"{k}: cannot compare {self.KINDS[x]} with {self.KINDS[y]} (V27)")

    def array(self, name, op, p, numbers=False):
        """V27: the parameter `elem`, `len`, or `sum` reads is an array by its value in effect,
        and for `sum` (`numbers`) of integers and floats only, as the runner reads it."""
        v = self.param_value(name)
        decl = self.ast["params"][name]
        if v is None and "default" not in decl:                 # required, no value yet (contract 0.11)
            if decl.get("kind") != "array":
                self.err(p, f"{op}: parameter `{name}` is not an array (V27)")
            elif numbers and decl.get("items") not in ("int", "float"):
                self.err(p, f"{op}: an element of `{name}` is {self.KINDS[decl.get('items')]} (V27)")
        elif not isinstance(v, list):
            self.err(p, f"{op}: parameter `{name}` is not an array (V27)")
        elif numbers:
            k = next((k for k in map(self.pkind, v) if k not in ("int", "float")), None)
            if k is not None:
                self.err(p, f"{op}: an element of `{name}` is {self.KINDS[k]} (V27)")

    def at(self, a, p, scope, binding):
        name, idx = a["ref"], a["index"]
        if scope.lookup(name) is None and name not in scope.forward:
            self.err(p, f"`at` names unknown binding `{name}`")
        if not scope.indices:
            self.err(p, "`at` outside any loop")
            return
        loop_index = scope.indices[-1]
        ok = (isinstance(idx, dict) and "sub" in idx and idx["sub"][0] == {"index": loop_index}
              and self.at_least(idx["sub"][1], 1, scope))
        # a binding of the same loop body: the one being defined, or any `let` of this body
        if binding == name or name in scope.forward:
            if not ok:
                self.err(p, f"`{name} @` names a binding of this loop body, so its index must be "
                            f"{{sub: [{{index: {loop_index}}}, e]}} with e provably >= 1")
        # of two `let`s of one name in this body, which one an `at` reads is not defined
        if name in scope.twice:
            self.err(p, f"`{name} @` names a binding this body defines more than once, "
                        f"so which one it reads is not defined")
        self.expr(idx, p + ["index"], scope)
        self.typed(idx, p + ["index"], scope)

    def at_least(self, e, k, scope):
        """e provably >= k, k in {0, 1} (rule V3): a literal >= k; a ref/param/draw whose
        distribution has min >= k (uniform lo, normal/lognormal min, every empirical value, every
        non-null mixture arm, const); `add` of a term >= k and a term >= 0. For k = 0 also any
        `mod` (Euclidean: the result is in [0, |b|)) and a loop index whose `from` is absent or
        provably >= 0 (steps are positive; `parallel` and `loader` indices start at 0)."""
        if isinstance(e, (int, float)) and not isinstance(e, bool):
            return e >= k
        if not isinstance(e, dict):
            return False
        kind, a = next(iter(e.items()))
        if kind == "ref":
            v = scope.lookup(a)
            return v is not None and self.at_least(v, k, scope)
        if kind == "param":
            d = self.param_value(a)
            return isinstance(d, dict) and self.at_least(d, k, scope)
        if kind == "draw":
            return self.at_least(a, k, scope)
        if kind == "uniform":
            return isinstance(a["lo"], (int, float)) and a["lo"] >= k
        if kind in ("normal", "lognormal"):
            # the runner clamps to `min` and then to `max`: `min` bounds a draw only when no
            # `max` below it can win (none, or one known and not below it)
            lo = self.number(a.get("min"))
            hi = self.number(a.get("max"))
            return (lo is not None and lo >= k and isinstance(a["min"], (int, float))
                    and ("max" not in a or (hi is not None and float(hi) >= float(lo))))
        if kind == "empirical":
            return all(isinstance(v, (int, float)) and v >= k for v in a["values"])
        if kind == "mixture":
            # a `none` arm draws no number, so at least one arm must draw one
            return (any(arm["dist"] is not None for arm in a)
                    and all(arm["dist"] is None or self.at_least(arm["dist"], k, scope) for arm in a))
        if kind == "const":
            return self.at_least(a, k, scope)
        if kind == "add":
            x, y = a
            return ((self.at_least(x, k, scope) and self.at_least(y, 0, scope))
                    or (self.at_least(x, 0, scope) and self.at_least(y, k, scope)))
        if k == 0:
            if kind == "mod":
                return True
            if kind == "index":
                return a in scope.nonneg_indices
        return False

    def number(self, e):
        """The number `e` is known to be: a literal, or a `param` whose value in effect is a number."""
        if isinstance(e, dict) and set(e) == {"param"}:
            e = self.param_value(e["param"])
        return e if isinstance(e, (int, float)) and not isinstance(e, bool) else None

    # rule V25: the least a dataset's size can be; UNBOUNDED when nothing bounds it below
    UNBOUNDED = "unbounded"

    def size_not_negative(self, size, p):
        """Rule V25: a dataset's `size` (not an `array`'s, which the runner clamps to [0, slot])
        that can be below 0 where the document shows it: a number known below 0 (rounded as the
        runner rounds), or a distribution unbounded below (a `normal` without `min`)."""
        least = self.least(size)
        if least == self.UNBOUNDED:
            self.err(p, "a size below 0: the distribution is unbounded below; give it a `min` of 0 or more (V25)")
        elif least is not None and not math.isfinite(least):
            self.err(p, "a size that is not a finite number (V25)")
        elif least is not None and round_away(least) < 0:
            self.err(p, f"a size below 0: it can be {integral(round_away(least))} (V25)")

    def exact(self, e):
        """The value `e` is known to have: a number, or add/sub/mul/min/max/neg of known values."""
        if not isinstance(e, dict) or set(e) == {"param"}:
            n = self.number(e)
            return None if n is None else float(n)
        kind, a = next(iter(e.items()))
        ops = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
               "min": least_of, "max": most_of}
        if kind in ops:
            x, y = self.exact(a[0]), self.exact(a[1])
            return None if x is None or y is None else ops[kind](x, y)
        if kind == "neg":
            x = self.exact(a)
            return None if x is None else -x
        return None

    def least(self, d):
        """The least a size (an expression or a distribution) can be, when the document shows
        it: a float, UNBOUNDED, or None when it does not."""
        if not isinstance(d, dict):
            return self.exact(d)
        kind, a = next(iter(d.items()))
        if kind == "param":
            v = self.param_value(a)
            return self.least(v) if isinstance(v, dict) else self.exact(d)
        if kind == "const":
            return self.least(a)
        if kind == "uniform":
            return self.exact(a["lo"])
        if kind == "uniform64":
            return self.UNBOUNDED
        if kind in ("normal", "lognormal"):
            if kind == "normal":
                sd = self.exact(a["sd"])
                if sd is None:
                    return None
                raw = self.UNBOUNDED if sd != 0 else self.exact(a["mean"])
            else:
                m = self.exact(a["median"])
                raw = None if m is None else self.UNBOUNDED if m < 0 else 0.0
            if raw is None:
                return None
            # the runner clamps to `min` and then to `max`
            if "min" in a:
                m = self.exact(a["min"])
                if m is None:
                    return None
                raw = m if raw == self.UNBOUNDED else most_of(raw, m)
            if "max" in a and raw != self.UNBOUNDED:
                m = self.exact(a["max"])
                if m is None:
                    return None
                raw = least_of(raw, m)
            return raw
        if kind == "empirical":
            vals = a["values"]
            if not vals or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                return None
            return float(min(vals))
        if kind == "mixture":
            # the arms that can be drawn: any one not a finite number is no finite size, whatever
            # the order of the arms; else any one unbounded or below 0 is; else every one known
            least, unknown, unbounded = None, False, False
            for arm in a:
                if not arm["weight"] > 0 or arm["dist"] is None:
                    continue
                x = self.least(arm["dist"])
                if x == self.UNBOUNDED:
                    unbounded = True
                elif x is None:
                    unknown = True
                elif not math.isfinite(x):
                    return math.nan
                else:
                    least = x if least is None else min(least, x)
            if unbounded:
                return self.UNBOUNDED
            return least if least is not None and (round_away(least) < 0 or not unknown) else None
        if kind in ("zipf", "hotset"):
            return None
        return self.exact(d)

    def handle(self, h, p, scope):
        k, a = next(iter(h.items()))
        if k == "ref":
            if scope.lookup(a) is None:
                self.err(p, f"handle binding `{a}` not in scope")
            elif a not in scope.fields and scope.kind_of(a) not in (None, "handle"):
                # V27: a handle position names a binding to a handle (RC1 review: `let x = 1`
                # then a write on `{ref: x}` passed and failed at run)
                self.err(p, f"expected a handle, got {self.KINDS[scope.kind_of(a)]} (V27)")
        elif k == "file":
            if "dataset" in a and a.get("id") is None:
                # a `null` id is no id, as the runner reads it
                self.err(p, "file handle needs either {dataset, id} or {of}")
            elif "dataset" in a:
                if a["dataset"] not in self.datasets:
                    self.err(p, f"unknown dataset `{a['dataset']}`")
                self.expr(a["id"], p + ["id"], scope)
                self.typed(a["id"], p + ["id"], scope)
            else:
                self.handle(a["of"], p + ["of"], scope)
            if "chunk" in a:
                self.expr(a["chunk"], p + ["chunk"], scope)
                self.typed(a["chunk"], p + ["chunk"], scope, optional=True)
        elif k == "inventory":
            if a not in self.datasets:
                self.err(p, f"unknown dataset `{a}`")
            elif "inventory" not in self.datasets[a]:
                self.err(p, f"`inventory` of `{a}`, which declares none (V22)")
        elif k == "dir":
            if a["dataset"] not in self.datasets or self.datasets[a["dataset"]]["layout"] == "array":
                self.err(p, f"`dir` needs a dataset named by ids, not an `array`: got `{a['dataset']}`")
            self.expr(a["id"], p + ["id"], scope)
            self.typed(a["id"], p + ["id"], scope)
        elif k == "object":
            ns = self.namespaces.get(a["namespace"])
            if ns is None:
                self.err(p, f"unknown namespace `{a['namespace']}`")
            elif set(a["fields"]) != set(ns["fields"]):
                # the names as the runner prints them (`{:?}` of the sorted names: `["j"]`)
                self.err(p, f"object fields {json.dumps(sorted(a['fields']))} != namespace fields {json.dumps(sorted(ns['fields']))}")
            for f, e in a["fields"].items():
                self.expr(e, p + ["fields", f], scope)
                # V27: an `int` field takes an integer or a float, rounded (never a bool); a `str` field a string
                t = ns["fields"].get(f) if ns is not None else None
                if t == "int":
                    self.typed(e, p + ["fields", f], scope, "an integer", self.NOT_OPERAND)
                elif t == "str":
                    self.typed(e, p + ["fields", f], scope, "a string", set(self.KINDS) - {"str"})
        elif k == "consume":
            if a not in self.datasets:
                self.err(p, f"unknown dataset `{a}`")
            elif self.datasets[a]["access"] not in ("map", "stream"):
                self.err(p, f"`consume` from `{a}`, which is `indexed`: its items are looked up by id (`file`, `pick`), never drawn from the shuffle (V21)")
            if not scope.indices:
                self.err(p, "`consume` outside any loop: its position needs an index")
            elif p not in self.consumes.setdefault(a, []):
                self.consumes[a].append(p)
                if scope.count_varies is not None:
                    self.consume_varying[tuple(p)] = scope.count_varies
        elif k == "pick":
            if a["dataset"] not in self.datasets:
                self.err(p, f"unknown dataset `{a['dataset']}`")
            if "dist" in a:
                self.distref(a["dist"], p + ["dist"], scope)
                self.drawn(a["dist"], p + ["dist"], scope, optional=True)
        elif k in ("unit", "column"):
            self.handle(a["of"], p + ["of"], scope)
            if "index" in a:
                self.expr(a["index"], p + ["index"], scope)
                self.typed(a["index"], p + ["index"], scope, optional=k == "unit")


class Scope:
    def __init__(self, check, parent=None, fields=()):
        self.check, self.parent = check, parent
        self.bindings, self.indices = {}, list(parent.indices) if parent else []
        self.varying = set()  # the bindings here whose value varies (V6)
        self.kinds = {}  # the kind each binding here is known to have, or None (V27)
        self.reads = {}  # a dataset and an `input` namespace each binding here may name (V12, V14)
        # the innermost enclosing frame whose count varies with an enclosing one (V6), or None
        self.count_varies = parent.count_varies if parent else None
        self.nonneg_indices = set(parent.nonneg_indices) if parent else set()
        self.channels = parent.channels if parent else set()
        self.written = parent.written if parent else set()
        self.forward = set()
        self.twice = set()
        self.fields = set(fields) | (parent.fields if parent else set())

    def child(self):
        return Scope(self.check, self)

    def with_index(self, name, nonneg=True, count_varies=None):
        s = Scope(self.check, self)
        s.indices.append(name)
        if count_varies is not None:
            s.count_varies = count_varies
        if nonneg:
            s.nonneg_indices.add(name)
        else:
            s.nonneg_indices.discard(name)
        return s

    def bind(self, name, value, varies=False, kind=None, reads=(None, None)):
        self.bindings[name] = value
        self.kinds[name] = kind
        self.reads[name] = reads
        if varies:
            self.varying.add(name)
        else:
            self.varying.discard(name)

    def is_varying(self, name):
        s = self
        while s:
            if name in s.bindings:
                return name in s.varying
            s = s.parent
        return False

    def kind_of(self, name):
        s = self
        while s:
            if name in s.bindings:
                return s.kinds[name]
            s = s.parent
        return None

    def reads_of(self, name):
        """The dataset and the `input` namespace the binding `name` may name (V12, V14)."""
        s = self
        while s:
            if name in s.bindings:
                return s.reads[name]
            s = s.parent
        return (None, None)

    def lookup(self, name):
        s = self
        while s:
            if name in s.bindings:
                return s.bindings[name]
            s = s.parent
        return None


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted((HERE.parent / "workloads").rglob("*.ast.json"))
    validator = jsonschema.Draft202012Validator(SCHEMA)
    failed = 0
    for f in files:
        try:
            ast = loads(f.read_text(encoding="utf-8"))
        except ValueError as e:     # not JSON as the runner reads it: `aeiou check` says the same
            ast, errors = None, [f"{f.name}: not an abstract ({e})"]
        else:
            # the integer range first, as the runner reads it: a seed past 2^64-1 has the same message in both
            errors = [f"{f.name}: {'/'.join(map(str, p))}: {m}" for p, m in int_range(ast)]
        if not errors:
            schema_errors = sorted(validator.iter_errors(ast), key=lambda e: list(e.absolute_path))
            errors = [f"{f.name}: {'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in schema_errors]
        if not errors:
            chk = Check(ast, f.name)
            errors = chk.run()
        if errors:
            failed += 1
            print(f"FAIL {f.name}")
            for e in errors:
                print("  " + e)
            continue
        digest = hashlib.sha256(canonical(ast)).hexdigest()
        ops = " ".join(f"{k}={v}" for k, v in sorted(chk.ops.items()))
        print(f"ok   {f.name}  sha256={digest[:16]}…  ops: {ops}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
