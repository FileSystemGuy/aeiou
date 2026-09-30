"""Validation against the published schema and the semantic rules of schema/README.md.

The schema and the reference checker live in `schema/` at the repository root; the builder
loads them from there (or from $AEIOU_SCHEMA_DIR). They are the runner's contract and the
builder does not keep a second copy.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import pathlib

from .nodes import BuildError


def schema_dir() -> pathlib.Path:
    env = os.environ.get("AEIOU_SCHEMA_DIR")
    if env:
        return pathlib.Path(env)
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "schema" / "abstract-ast.schema.json"
        if cand.exists():
            return cand.parent
    raise BuildError("cannot find schema/abstract-ast.schema.json; set AEIOU_SCHEMA_DIR")


@functools.lru_cache(maxsize=1)
def _load():
    d = schema_dir()
    schema = json.loads((d / "abstract-ast.schema.json").read_text())
    spec = importlib.util.spec_from_file_location("aeiou_schema_check", d / "check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return schema, mod


def validate(ast: dict, name: str | None = None) -> None:
    """Raise BuildError listing every schema and semantic error."""
    import jsonschema
    schema, check = _load()
    name = name or ast.get("name", "ast")
    validator = jsonschema.Draft202012Validator(schema)
    errors = [f"{name}: {'/'.join(map(str, e.absolute_path))}: {e.message[:300]}"
              for e in sorted(validator.iter_errors(ast), key=lambda e: list(e.absolute_path))]
    if not errors:
        errors = check.Check(ast, name).run()
    if errors:
        raise BuildError("AST rejected:\n  " + "\n  ".join(errors))


def op_counts(ast: dict):
    _, check = _load()
    chk = check.Check(ast, ast.get("name", "ast"))
    chk.run()
    return dict(sorted(chk.ops.items()))
