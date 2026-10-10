#!/usr/bin/env python3
"""Validate abstract ASTs against schema/abstract-ast.schema.json and the semantic rules of
schema/README.md, print each AST's canonical hash and op-kind counts.

Usage: python3 schema/check.py [file.ast.json ...]   (default: schema/examples/*.ast.json)

This is the reference for the rules until the Rust validator exists; the Rust validator must
reject everything this rejects. It needs only jsonschema.
"""
import hashlib
import json
import pathlib
import sys
from collections import Counter

import jsonschema

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "abstract-ast.schema.json").read_text())

CONTROL = {"let", "loop", "parallel", "channel", "put", "take", "loader", "barrier",
           "compute", "cond", "choose", "phase", "trace"}
INDEX_BINDERS = {"loop", "parallel", "loader"}
MUTATING_OPS = {"write", "ftruncate", "fallocate", "unlink"}          # V12 and V14, plus rename and open flags
WRITE_FLAGS = {"WRONLY", "RDWR", "CREAT", "TRUNC", "APPEND"}
RESERVED_PREFIX = ".aeiou"                                            # V13: the manifest and sidecars


def dataset_root(d: dict) -> str:
    """Constant directory prefix of a files pattern, or the directory of a regions file."""
    kind, body = next(iter(d.items()))
    if kind == "regions":
        return body["file"].rpartition("/")[0]
    prefix = body["pattern"].split("{", 1)[0]
    return prefix.rpartition("/")[0]


def protocols(ast: dict, traces: bool) -> tuple:
    """(posix, s3): the protocols an abstract has names on. `posix` when a dataset or namespace
    declares it (or none), when a `trace` node runs (its files are under --root), or when no
    name is `s3`."""
    declared = [next(iter(d.values())).get("protocol", "posix") for d in ast.get("datasets", {}).values()]
    declared += [n.get("protocol", "posix") for n in ast.get("namespaces", {}).values()]
    s3 = "s3" in declared
    return ("posix" in declared or traces or not s3), s3


def canonical(ast: dict) -> bytes:
    """Canonical form: sorted keys, no whitespace, ASCII escapes, provenance removed. Floats
    serialize with Python's repr (shortest round-trip), which is what the builder emits. The
    on-disk file is the same JSON pretty-printed; only these bytes are hashed."""
    stripped = {k: v for k, v in ast.items() if k != "provenance"}
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


class Check:
    def __init__(self, ast: dict, name: str):
        self.ast, self.name, self.errors = ast, name, []
        self.params = set(ast.get("params", {}))
        self.datasets = ast.get("datasets", {})
        self.namespaces = ast.get("namespaces", {})
        self.ops = Counter()
        self.traces = False

    def err(self, path, msg):
        self.errors.append(f"{self.name}: {'/'.join(map(str, path))}: {msg}")

    # ---- entry ----
    def run(self):
        if "gpus" in self.params:
            self.err(["params"], "`gpus` is reserved and set by the runner")
        for pname, p in self.ast.get("params", {}).items():
            self.value(p["default"], ["params", pname])
        roots = {}
        for dname, d in self.datasets.items():
            kind, body = next(iter(d.items()))
            self.exprlike(body["size"], ["datasets", dname, "size"], Scope(self), dist_ok=True)
            self.expr(body["count"], ["datasets", dname, "count"], Scope(self))
            for k in ("samples_per_file", "chunk", "slot"):
                if k in body:
                    self.expr(body[k], ["datasets", dname, k], Scope(self))
            layout = body.get("format", {}).get("layout")
            if layout:
                lp = ["datasets", dname, "format", "layout"]
                for k, v in layout.items():
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
                    elif k != "writer":
                        self.expr(v, lp + [k], Scope(self))
            self.reserved(body.get("pattern") or body.get("file"), ["datasets", dname])
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
            self.body(a["body"], ["actors", aname, "body"], scope)
        self.v19()
        return self.errors

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
        scope.forward = {n["let"]["name"] for n in nodes if "let" in n}
        for i, node in enumerate(nodes):
            kind, args = next(iter(node.items()))
            p = path + [i, kind]
            if kind not in CONTROL:
                self.ops[kind] += 1
            getattr(self, "n_" + kind, self.n_op)(args, p, scope)

    def n_let(self, a, p, scope):
        self.exprlike(a["value"], p + ["value"], scope, dist_ok=False, binding=a["name"])
        scope.bind(a["name"], a["value"])

    def n_loop(self, a, p, scope):
        for k in ("from", "to", "step"):
            if k in a:
                self.expr(a[k], p + [k], scope)
        nonneg = "from" not in a or self.at_least(a["from"], 0, scope)
        self.body(a["body"], p + ["body"], scope.with_index(a["index"], nonneg))

    def n_parallel(self, a, p, scope):
        self.expr(a["width"], p + ["width"], scope)
        self.body(a["body"], p + ["body"], scope.with_index(a["index"]))

    def n_loader(self, a, p, scope):
        for k in ("workers", "prefetch", "batches"):
            self.expr(a[k], p + [k], scope)
        scope.channels.add(a["name"])
        self.body(a["body"], p + ["body"], scope.with_index(a["index"]))

    def n_channel(self, a, p, scope):
        self.expr(a["capacity"], p + ["capacity"], scope)
        scope.channels.add(a["name"])

    def n_put(self, a, p, scope):
        if a["channel"] not in scope.channels:
            self.err(p, f"channel `{a['channel']}` not declared")
        self.expr(a["seq"], p + ["seq"], scope)

    def n_take(self, a, p, scope):
        if a["channel"] not in scope.channels:
            self.err(p, f"channel `{a['channel']}` not declared")

    def n_barrier(self, a, p, scope):
        pass

    def n_compute(self, a, p, scope):
        self.expr(a["ns"], p + ["ns"], scope)

    def n_cond(self, a, p, scope):
        self.expr(a["if"], p + ["if"], scope)
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
        rep = a.get("repeat")
        if rep not in (None, "until_eof", "until_end"):
            self.expr(rep, p + ["repeat"], scope)
        if kind == "write":
            self.note_write(a["file"], scope)
        if kind == "read" and rep == "until_eof":
            self.check_until_eof(a["file"], p, scope)
        # V12: datasets are read-only
        if kind in MUTATING_OPS or kind == "rename":
            for key in ("file", "from", "to"):
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
        elif kind in ("ftruncate", "fallocate"):
            v18 = ("file", kind)
        if v18:
            ns = self.object_store_namespace(a["file"], scope)
            if ns:
                self.err(p + [v18[0]], f"{v18[1]} on namespace `{ns}`, declared `protocol: s3`: an object is written once, in order from 0, and has no {v18[1]} (V18)")

    def object_store_namespace(self, h, scope):
        """Namespace name if the handle is (a binding to) an object of a namespace declared `protocol: s3`."""
        ns = self.object_namespace(h, scope) if isinstance(h, dict) else None
        if ns and self.namespaces.get(ns, {}).get("protocol") == "s3":
            return ns
        return None

    def input_namespace_of(self, h, scope):
        """Namespace name if the handle is (a binding to) an object of an `input` namespace."""
        if not isinstance(h, dict) or len(h) != 1:
            return None
        k, a = next(iter(h.items()))
        if k == "ref":
            v = scope.lookup(a)
            return self.input_namespace_of(v, scope) if isinstance(v, dict) else None
        if k == "object" and self.namespaces.get(a["namespace"], {}).get("input"):
            return a["namespace"]
        return None

    def dataset_of(self, h, scope):
        """Dataset name if the handle is (a binding to) a dataset file, dir, sample, region, or chunk."""
        if not isinstance(h, dict) or len(h) != 1:
            return None
        k, a = next(iter(h.items()))
        if k == "ref":
            v = scope.lookup(a)
            return self.dataset_of(v, scope) if isinstance(v, dict) else None
        if k == "file":
            return a["dataset"] if "dataset" in a else self.dataset_of(a["of"], scope)
        if k in ("dir",):
            return a["dataset"]
        if k == "consume":
            return a
        if k == "pick":
            return a["dataset"]
        if k in ("unit", "column"):
            return self.dataset_of(a["of"], scope)
        return None

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
            if k in SCHEMA["$defs"]["handle"]["properties"]:
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
        elif isinstance(a, dict):
            for kk, vv in a.items():
                if isinstance(vv, (dict, list)) or kk in ("lo", "hi", "mean", "sd", "median", "min", "max"):
                    self.expr(vv, p + [kk], scope)
        elif k == "const":
            self.expr(a, p, scope)

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
            self.expr(a["index"], p + ["index"], scope)
        elif k in ("len", "sum"):
            if a not in self.params:
                self.err(p, f"unknown parameter array `{a}`")
        elif k in ("size", "offset", "unit_index", "units", "chunks"):
            self.handle(a, p, scope)
        elif k == "count":
            if a not in self.datasets:
                self.err(p, f"unknown dataset `{a}`")
        elif k == "dirs":
            if a not in self.datasets or "files" not in self.datasets[a]:
                self.err(p, f"`dirs` needs a files dataset, got `{a}`")
        elif k == "cond":
            self.expr(a["if"], p + ["if"], scope)
            self.exprlike(a["then"], p + ["then"], scope, dist_ok=True, binding=binding)
            self.exprlike(a["else"], p + ["else"], scope, dist_ok=True, binding=binding)
        elif isinstance(a, list):
            for i, x in enumerate(a):
                self.expr(x, p + [i], scope)
        elif k in ("neg", "not"):
            self.expr(a, p, scope)
        elif k == "actor":
            pass
        else:
            self.err(p, f"unhandled expression kind `{k}`")

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
        self.expr(idx, p + ["index"], scope)

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
            d = self.ast["params"].get(a, {}).get("default")
            return isinstance(d, dict) and self.at_least(d, k, scope)
        if kind == "draw":
            return self.at_least(a, k, scope)
        if kind == "uniform":
            return isinstance(a["lo"], (int, float)) and a["lo"] >= k
        if kind in ("normal", "lognormal"):
            return isinstance(a.get("min"), (int, float)) and a["min"] >= k
        if kind == "empirical":
            return all(isinstance(v, (int, float)) and v >= k for v in a["values"])
        if kind == "mixture":
            return all(arm["dist"] is None or self.at_least(arm["dist"], k, scope) for arm in a)
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

    def handle(self, h, p, scope):
        k, a = next(iter(h.items()))
        if k == "ref":
            if scope.lookup(a) is None:
                self.err(p, f"handle binding `{a}` not in scope")
        elif k == "file":
            if "dataset" in a:
                if a["dataset"] not in self.datasets:
                    self.err(p, f"unknown dataset `{a['dataset']}`")
                self.expr(a["id"], p + ["id"], scope)
            else:
                self.handle(a["of"], p + ["of"], scope)
            if "chunk" in a:
                self.expr(a["chunk"], p + ["chunk"], scope)
        elif k == "dir":
            if a["dataset"] not in self.datasets or "files" not in self.datasets[a["dataset"]]:
                self.err(p, f"`dir` needs a files dataset, got `{a['dataset']}`")
            self.expr(a["id"], p + ["id"], scope)
        elif k == "object":
            ns = self.namespaces.get(a["namespace"])
            if ns is None:
                self.err(p, f"unknown namespace `{a['namespace']}`")
            elif set(a["fields"]) != set(ns["fields"]):
                self.err(p, f"object fields {sorted(a['fields'])} != namespace fields {sorted(ns['fields'])}")
            for f, e in a["fields"].items():
                self.expr(e, p + ["fields", f], scope)
        elif k == "consume":
            if a not in self.datasets:
                self.err(p, f"unknown dataset `{a}`")
            if not scope.indices:
                self.err(p, "`consume` outside any loop: its position needs an index")
        elif k == "pick":
            if a["dataset"] not in self.datasets:
                self.err(p, f"unknown dataset `{a['dataset']}`")
            if "dist" in a:
                self.distref(a["dist"], p + ["dist"], scope)
        elif k in ("unit", "column"):
            self.handle(a["of"], p + ["of"], scope)
            if "index" in a:
                self.expr(a["index"], p + ["index"], scope)


class Scope:
    def __init__(self, check, parent=None, fields=()):
        self.check, self.parent = check, parent
        self.bindings, self.indices = {}, list(parent.indices) if parent else []
        self.nonneg_indices = set(parent.nonneg_indices) if parent else set()
        self.channels = parent.channels if parent else set()
        self.written = parent.written if parent else set()
        self.forward = set()
        self.fields = set(fields) | (parent.fields if parent else set())

    def child(self):
        return Scope(self.check, self)

    def with_index(self, name, nonneg=True):
        s = Scope(self.check, self)
        s.indices.append(name)
        if nonneg:
            s.nonneg_indices.add(name)
        else:
            s.nonneg_indices.discard(name)
        return s

    def bind(self, name, value):
        self.bindings[name] = value

    def lookup(self, name):
        s = self
        while s:
            if name in s.bindings:
                return s.bindings[name]
            s = s.parent
        return None


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted((HERE / "examples").glob("*.ast.json"))
    validator = jsonschema.Draft202012Validator(SCHEMA)
    failed = 0
    for f in files:
        ast = json.loads(f.read_text())
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
