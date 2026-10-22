"""Use cases, benchmark rules, and the lock (`DESIGN_REVIEW.md` §3.78, contract 0.11).

A use case is a case directory under `workloads/` whose `case.toml` lists, for each run, its
abstract and the fragments (`params_version` 2 parameter files) that set the job and the
accelerator. A benchmark's rules file names no parameter: it freezes by state and holds the
constraints a run must meet, over the runner's `fact.` and `result.` names. Resolving a use
case, and a rules file for a benchmark entry, writes the **lock**: every value with its state
and where it came from, the abstract pinned by its path under `workloads/` and its
`ast_sha256`, the freeze and the constraints copied in, the fragments and the rules file as
provenance. The lock is generated and never edited, byte-identical on every site, and it is
the one file `aeiou run --case`/`--case-file` reads; the runner computes its workload id.

Layers only narrow: every value is `locked`, `default`, or `required`; a later layer may
replace a default or require a value, never change one an earlier layer locked; fragments of
one layer combine by agreement (two that set a name differently are refused, both named),
never last-one-wins.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import tomllib

from .nodes import BuildError
from . import params as P

LOCK_VERSION = 1
RULES_VERSION = 1
LOCK_FILE = "case.lock.json"
FREEZE = {"defaults": "locked", "scale": "required", "identity_options": "locked"}


# ---------------------------------------------------------------------------- constraints

_TOKEN = re.compile(r"\s*(?:(?P<num>\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
                    r"|(?P<op>>=|<=|==|!=|[-+*()<>,]))")
_CMP = {">=": "ge", "<=": "le", ">": "gt", "<": "lt", "==": "eq", "!=": "ne"}
_FUNCS = {"min", "max"}


def parse_constraint(text: str) -> tuple[dict, list[str]]:
    """A constraint's infix text as the AST's expression form, and the runner names it reads.

    The grammar: `or`, `and`, `not`; one comparison (`>=`, `<=`, `>`, `<`, `==`, `!=`); `+`,
    `-`; `*`, `div`, `mod` (integer division and remainder, as the AST's); unary `-`; numbers;
    `min(a, b)`, `max(a, b)`; parentheses; and the names `fact.NAME` and `result.NAME`, as
    `{"fact": NAME}` and `{"result": NAME}`. Nothing else is a name: a rules file names no
    parameter (§3.78 point 6)."""
    toks, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise BuildError(f"constraint `{text}`: cannot read `{text[pos:].strip()}`")
        kind = m.lastgroup
        toks.append((kind, m.group(kind)))
        pos = m.end()
    names: list[str] = []
    i = 0

    def peek():
        return toks[i] if i < len(toks) else (None, None)

    def take(want=None):
        nonlocal i
        t = peek()
        if t[0] is None or (want is not None and t[1] != want):
            raise BuildError(f"constraint `{text}`: expected {want or 'more'}, got {t[1] or 'the end'}")
        i += 1
        return t

    def word(w):
        return peek() == ("name", w)

    def or_():
        e = and_()
        while word("or"):
            take()
            e = {"or": [e, and_()]}
        return e

    def and_():
        e = not_()
        while word("and"):
            take()
            e = {"and": [e, not_()]}
        return e

    def not_():
        if word("not"):
            take()
            return {"not": not_()}
        return cmp()

    def cmp():
        e = sum_()
        if peek()[0] == "op" and peek()[1] in _CMP:
            op = take()[1]
            e = {_CMP[op]: [e, sum_()]}
            if peek()[0] == "op" and peek()[1] in _CMP:
                raise BuildError(f"constraint `{text}`: one comparison at a time (`a <= b and b <= c`)")
        return e

    def sum_():
        e = term()
        while peek()[0] == "op" and peek()[1] in "+-":
            op = take()[1]
            e = {"add" if op == "+" else "sub": [e, term()]}
        return e

    def term():
        e = unary()
        while (peek()[0] == "op" and peek()[1] == "*") or word("div") or word("mod"):
            op = take()[1]
            e = {{"*": "mul"}.get(op, op): [e, unary()]}
        return e

    def unary():
        if peek() == ("op", "-"):
            take()
            return {"neg": unary()}
        return atom()

    def atom():
        kind, v = take()
        if kind == "num":
            s = v.replace("_", "")
            return float(s) if any(c in s for c in ".eE") else int(s)
        if kind == "op" and v == "(":
            e = or_()
            take(")")
            return e
        if kind == "name" and v in _FUNCS:
            take("(")
            a = sum_()
            take(",")
            b = sum_()
            take(")")
            return {v: [a, b]}
        if kind == "name" and "." in v:
            space, n = v.split(".", 1)
            if space in ("fact", "result"):
                names.append(v)
                return {space: n}
        raise BuildError(f"constraint `{text}`: `{v}` is not a number, a `fact.` or `result.` name, or an operator "
                         f"(a rules file names no parameter: DESIGN_REVIEW.md §3.78 point 6)")

    e = or_()
    if i != len(toks):
        raise BuildError(f"constraint `{text}`: unexpected `{toks[i][1]}`")
    return e, sorted(set(names))


# ---------------------------------------------------------------------------- inputs

def load_rules(path) -> dict:
    """A benchmark's rules file (`schema/rules.schema.json`): `freeze` by state and
    `constraints` over the runner's names, never a parameter's name."""
    path = pathlib.Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BuildError(f"{path}: {e.strerror}") from None
    except ValueError as e:
        raise BuildError(f"{path}: not JSON: {e}") from None
    if not isinstance(doc, dict) or doc.get("rules_version") != RULES_VERSION:
        raise BuildError(f"{path.name}: not a rules file (rules_version {RULES_VERSION})")
    for k in doc:
        if k not in ("rules_version", "name", "doc", "freeze", "constraints"):
            raise BuildError(f"{path.name}: unknown key `{k}` (a rules file names no parameter: freeze by state, constraints over fact. and result. names)")
    freeze = doc.get("freeze") or {}
    for k, v in freeze.items():
        if FREEZE.get(k) != v:
            raise BuildError(f"{path.name}: freeze: `{k}: {json.dumps(v)}` is not one of " + ", ".join(f"{a}: {b}" for a, b in FREEZE.items()))
    for c in doc.get("constraints", []):
        if not isinstance(c, dict) or set(c) - {"expr", "doc"} or not isinstance(c.get("expr"), str):
            raise BuildError(f"{path.name}: a constraint is {{\"expr\": TEXT, \"doc\": TEXT}}, got {json.dumps(c)}")
        parse_constraint(c["expr"])
    return doc


def read_case(case_dir) -> list[dict]:
    """A case's runs from its `case.toml`: each `{ast, params}`, `params` one fragment or a list
    of them, paths relative to the case directory."""
    case_dir = pathlib.Path(case_dir)
    f = case_dir / "case.toml"
    try:
        doc = tomllib.loads(f.read_text(encoding="utf-8"))
    except OSError as e:
        raise BuildError(f"{f}: {e.strerror}") from None
    except tomllib.TOMLDecodeError as e:
        raise BuildError(f"{f}: {e}") from None
    runs = doc.get("run")
    if not isinstance(runs, list) or not runs:
        raise BuildError(f"{f}: no [[run]] table: a case names the abstract each run executes")
    out = []
    for r in runs:
        frags = r.get("params", [])
        frags = [frags] if isinstance(frags, str) else frags
        if not isinstance(r.get("ast"), str) or not all(isinstance(x, str) for x in frags):
            raise BuildError(f"{f}: a run is `ast = NAME` and `params = FILE` or a list of files")
        out.append({"ast": r["ast"], "fragments": [case_dir / x for x in frags]})
    return out


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _rel(path: pathlib.Path, workloads: pathlib.Path) -> str:
    """A path as a lock records it: relative to `workloads/`, the basename for a file outside
    it (so a lock is byte-identical on every site)."""
    try:
        return path.resolve().relative_to(workloads.resolve()).as_posix()
    except ValueError:
        return path.name


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------- the lock

def _entry(state: str, value=None) -> dict:
    return {"required": True} if state == "required" else {state: value}


def resolve_run(ast: dict, ast_rel: str, ast_sha: str, fragments: list, workloads: pathlib.Path, freeze: dict) -> dict:
    """One run of the lock: the abstract's declarations, then the use case's fragments by
    agreement, then the rules' freeze, every entry with the file it came from."""
    states: dict[str, tuple[str, object]] = {}
    sources: dict[str, str] = {}
    for k, p in ast.get("params", {}).items():
        states[k] = ("default", p["default"]) if "default" in p else ("required", None)
        sources[k] = "abstract"
    # the use case: its fragments, order-free, by agreement
    given: dict[str, tuple[str, object, str]] = {}
    run_given: dict[str, tuple[str, object, str]] = {}
    for f in fragments:
        pset = P.load(f)
        errors = P.check(ast, pset, ast_sha256=ast_sha, name=f.name)
        if errors:
            raise BuildError("\n  ".join(errors))
        rel = _rel(f, workloads)
        for table, into in (("params", given), ("run", run_given)):
            for k, e in pset.get(table, {}).items():
                state, v = P.entry(e)
                if table == "params" and state != "required":
                    v, _ = P.keep_int(k, ast["params"][k], v)
                if k in into and not _same(into[k][:2], (state, v)):
                    raise BuildError(f"{table} `{k}`: {into[k][2]} sets {json.dumps(_entry(*into[k][:2]))}, {rel} sets "
                                     f"{json.dumps(_entry(state, v))}; fragments of one use case agree (DESIGN_REVIEW.md §3.78 point 5)")
                into[k] = (state, v, rel)
    for k, (state, v, rel) in given.items():
        states[k], sources[k] = (state, v), rel
    run = {k: (state, v) for k, (state, v, _) in run_given.items()}
    run_sources = {k: rel for k, (_, _, rel) in run_given.items()}
    # the rules: by state, never by name
    if freeze.get("defaults") == "locked":
        for k, (state, v) in states.items():
            if state == "default":
                states[k] = ("locked", v)
    if freeze.get("scale") == "required":
        for k, p in ast.get("params", {}).items():
            if p.get("scale"):
                if k in given and given[k][0] == "locked":
                    raise BuildError(f"params `{k}`: a `scale` parameter, which the rules make required, is locked by {sources[k]}")
                states[k] = ("required", None)
                sources[k] = "rules"
    if freeze.get("identity_options") == "locked":
        for k, (state, v) in run.items():
            if state == "default":
                run[k] = ("locked", v)
    out = {"name": ast["name"], "abstract": ast_rel, "ast_sha256": ast_sha,
           "params": {k: _entry(*states[k]) for k in sorted(states)},
           "sources": {k: sources[k] for k in sorted(sources)}}
    if run:
        out["run"] = {k: _entry(*run[k]) for k in sorted(run)}
        out["run_sources"] = {k: run_sources[k] for k in sorted(run_sources)}
    return out


def make_lock(case_dir, *, workloads, rules=None) -> dict:
    """The lock of the use case at `case_dir` (and of a benchmark entry, with `rules`)."""
    from .emit import load, sha256
    from .validate import validate
    case_dir, workloads = pathlib.Path(case_dir), pathlib.Path(workloads)
    mechanism = case_dir.parent.parent
    rules_doc = load_rules(rules) if rules else None
    freeze = (rules_doc or {}).get("freeze") or {}
    runs = []
    for r in read_case(case_dir):
        ast_path = mechanism / f"{r['ast']}.ast.json"
        if not ast_path.exists():
            raise BuildError(f"{case_dir / 'case.toml'}: no abstract `{r['ast']}` in {mechanism} ({ast_path.name})")
        ast = load(ast_path)
        validate(ast, ast_path.name)
        runs.append(resolve_run(ast, _rel(ast_path, workloads), sha256(ast), r["fragments"], workloads, freeze))
    lock = {"lock_version": LOCK_VERSION, "name": f"{mechanism.name}/{case_dir.name}"}
    if rules_doc and rules_doc.get("doc"):
        lock["doc"] = rules_doc["doc"]
    lock["runs"] = runs
    if freeze:
        lock["freeze"] = {k: freeze[k] for k in FREEZE if k in freeze}
    if rules_doc and rules_doc.get("constraints"):
        lock["constraints"] = []
        for c in rules_doc["constraints"]:
            expr, names = parse_constraint(c["expr"])
            lock["constraints"].append({"text": c["expr"], "expr": expr, "names": names, **({"doc": c["doc"]} if "doc" in c else {})})
    frags = sorted({_rel(f, workloads): f for r in read_case(case_dir) for f in r["fragments"]}.items())
    lock["provenance"] = {
        "tool": "aeiou-params lock",
        # the case's path alone: its case.toml holds the catalog's attributes too, and what of it a
        # lock depends on, the runs' abstracts and fragments, is in the lock itself
        "case": _rel(case_dir / "case.toml", workloads),
        "fragments": [{"path": rel, "sha256": _sha(f)} for rel, f in frags],
        **({"rules": {"path": _rel(pathlib.Path(rules), workloads), "sha256": _sha(rules), "name": rules_doc.get("name")}} if rules_doc else {}),
    }
    return lock


def render(lock: dict) -> str:
    """The lock as its file holds it: indented, one parameter, source, and constraint to a
    line, so a lock's diff, what a reviewer reads (§3.78 point 12), is a line per value."""
    def one(v):
        return json.dumps(v, ensure_ascii=False, allow_nan=False)

    def table(d, ind):
        if not d:
            return "{}"
        pad = " " * ind
        return "{\n" + ",\n".join(f"{pad}  {one(k)}: {one(v)}" for k, v in d.items()) + f"\n{pad}}}"

    out = ["{"]
    items = []
    for k, v in lock.items():
        if k == "runs":
            runs = []
            for r in v:
                fields = []
                for rk, rv in r.items():
                    if rk in ("params", "sources", "run", "run_sources"):
                        fields.append(f'      {one(rk)}: {table(rv, 6)}')
                    else:
                        fields.append(f'      {one(rk)}: {one(rv)}')
                runs.append("    {\n" + ",\n".join(fields) + "\n    }")
            items.append('  "runs": [\n' + ",\n".join(runs) + "\n  ]")
        elif k == "constraints":
            items.append('  "constraints": [\n' + ",\n".join(f"    {one(c)}" for c in v) + "\n  ]")
        elif k == "provenance":
            items.append(f'  "provenance": {table(v, 2)}')
        else:
            items.append(f"  {one(k)}: {one(v)}")
    out.append(",\n".join(items))
    out.append("}")
    text = "\n".join(out) + "\n"
    assert json.loads(text) == lock
    return text


def stale(case_dir, *, workloads) -> str | None:
    """Why the committed lock of a case is not the one its inputs make now, or None. A case's
    committed lock is made without rules (a rules file is a benchmark's, outside the tree)."""
    f = pathlib.Path(case_dir) / LOCK_FILE
    if not f.exists():
        return f"no {LOCK_FILE}: make it with `aeiou-params lock {case_dir}`"
    try:
        want = render(make_lock(case_dir, workloads=workloads))
    except BuildError as e:
        return f"its use case does not resolve: {e}"
    if f.read_text(encoding="utf-8") != want:
        return f"{LOCK_FILE} is stale (its abstract, case.toml, or a fragment changed): remake it with `aeiou-params lock {case_dir}`"
    return None


def cases_of(ast_path) -> list[pathlib.Path]:
    """The committed cases of the abstract at `ast_path` that have a lock: what `aeiou-build`
    re-locks after it rebuilt that abstract (§3.78 point 17)."""
    ast_path = pathlib.Path(ast_path)
    name = ast_path.name.removesuffix(".ast.json")
    out = []
    for lock in sorted((ast_path.parent / "cases").glob(f"*/{LOCK_FILE}")):
        try:
            runs = [r["ast"] for r in read_case(lock.parent)]
        except BuildError:
            continue
        if name in runs:
            out.append(lock.parent)
    return out


def workloads_of(path) -> pathlib.Path:
    """The `workloads/` directory above `path`."""
    for p in [pathlib.Path(path).resolve(), *pathlib.Path(path).resolve().parents]:
        if p.name == "workloads":
            return p
    raise BuildError(f"{path}: not under a `workloads/` directory: a lock names its abstract by its path below workloads/ (--workloads DIR)")
