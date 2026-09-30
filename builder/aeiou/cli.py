"""`aeiou-build`: run an authoring script and write its workloads as AST JSON.

    aeiou-build script.py [script.py ...] [-o DIR] [--hermetic] [--twice] [--check]

Default: run each script in this process and write `<name>.ast.json` per Workload into DIR
(default: the script's directory) with a provenance block.

--hermetic   Run each script in a fresh, isolated child (`python -s -B`, scrubbed environment,
             audit hooks, stubbed clocks and entropy; see hermetic.py). The child returns the
             AST on stdout; this process validates it again and writes the file.
--twice      Build in two children with different PYTHONHASHSEED values and refuse to write
             unless the canonical hashes agree; records built_twice_identical in provenance.
--check      Do not write; compare the canonical hash with the existing file's and exit 1 on a
             difference (CI drift check).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import runpy
import subprocess
import sys

from . import __version__
from .nodes import BuildError


def _run_script(script: pathlib.Path) -> list:
    from .builder import Workload
    before = len(Workload._registry)
    runpy.run_path(str(script), run_name="__abstract__")
    made = Workload._registry[before:]
    if not made:
        raise BuildError(f"{script}: no Workload was constructed")
    return made


def _child_main(argv=None):
    """Entry point of the isolated child: build, validate, print JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--hermetic", action="store_true")
    ap.add_argument("--schema-dir")
    a = ap.parse_args(argv)
    script = pathlib.Path(a.script).resolve()
    if a.schema_dir:
        os.environ["AEIOU_SCHEMA_DIR"] = a.schema_dir
    if a.hermetic:
        from . import hermetic
        hermetic.install(script, extra_read_roots=[a.schema_dir] if a.schema_dir else ())
    out = {}
    for wl in _run_script(script):
        out[wl.name] = wl.build()
    sys.stdout.write(json.dumps(out))   # insertion order kept for the on-disk rendering
    sys.stdout.flush()


def _spawn_child(script: pathlib.Path, hermetic: bool, hashseed: int) -> dict:
    from .validate import schema_dir
    pkg_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": str(hashseed),
           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    boot = (f"import sys; sys.path.insert(0, {pkg_root!r}); "
            f"from aeiou.cli import _child_main; _child_main()")
    cmd = [sys.executable, "-s", "-B", "-c", boot, str(script), "--schema-dir", str(schema_dir())]
    if hermetic:
        cmd.append("--hermetic")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(script.parent))
    if r.returncode:
        tail = r.stderr.strip().splitlines()[-12:]
        raise BuildError(f"{script.name}: child build failed (PYTHONHASHSEED={hashseed}):\n  "
                         + "\n  ".join(tail))
    return json.loads(r.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aeiou-build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, help="output directory (default: the script's)")
    ap.add_argument("--hermetic", action="store_true")
    ap.add_argument("--twice", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-provenance", action="store_true")
    ap.add_argument("--version", action="version", version=f"aeiou-build {__version__}")
    a = ap.parse_args(argv)

    from . import emit
    status = 0
    for script in a.scripts:
        script = script.resolve()
        try:
            asts = _build_one(script, a, emit)
        except BuildError as e:
            print(f"FAIL {script.name}: {e}", file=sys.stderr)
            status = 1
            continue
        except Exception as e:  # the author's script raised: show where, keep going
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f for f in reversed(tb) if f.filename == str(script)), tb[-1])
            print(f"FAIL {script.name}: {type(e).__name__} at line {where.lineno}: {e}", file=sys.stderr)
            status = 1
            continue
        out_dir = a.out or script.parent
        for name, (ast, prov) in asts.items():
            path = out_dir / f"{name}.ast.json"
            digest = emit.sha256(ast)
            if a.check:
                if not path.exists():
                    print(f"MISSING {path}")
                    status = 1
                    continue
                old = emit.sha256(emit.load(path))
                if old != digest:
                    print(f"DRIFT {path.name}: committed {old[:16]}… rebuilt {digest[:16]}…")
                    status = 1
                else:
                    print(f"ok   {path.name}  sha256={digest[:16]}…")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(emit.render(ast, prov))
            from .validate import op_counts
            ops = " ".join(f"{k}={v}" for k, v in op_counts(ast).items())
            print(f"wrote {path}  sha256={digest[:16]}…  ops: {ops}")
    return status


def _build_one(script, a, emit) -> dict:
    """name -> (ast, provenance)."""
    pre = {"script_sha256": emit.file_sha256(script), "git": emit.git_describe(script)}
    lp = emit.find_lock(script.parent)
    pre["lock"] = {"file": lp.name, "sha256": emit.file_sha256(lp)} if lp else None
    emit.PRECOMPUTED[str(script)] = pre
    built_twice = None
    if a.hermetic or a.twice:
        first = _spawn_child(script, a.hermetic, 1)
        if a.twice:
            second = _spawn_child(script, a.hermetic, 4242)
            for name, ast in first.items():
                h1, h2 = emit.sha256(ast), emit.sha256(second.get(name, {}))
                if h1 != h2:
                    raise BuildError(f"{name}: not reproducible: {h1[:16]}… vs {h2[:16]}… under "
                                     f"different PYTHONHASHSEED values (set iteration order, "
                                     f"id()-based ordering, or an unseeded draw in the script?)")
            built_twice = True
        from .validate import validate
        asts = {}
        for name, ast in first.items():
            validate(ast, name)
            asts[name] = ast
    else:
        asts = {wl.name: wl.build() for wl in _run_script(script)}
    return {name: (ast, None if a.no_provenance else emit.provenance(script, ast, built_twice=built_twice))
            for name, ast in asts.items()}


if __name__ == "__main__":
    sys.exit(main())
