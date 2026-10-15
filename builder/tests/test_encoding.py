"""Files are UTF-8 whatever the locale. `render` writes non-ASCII as itself (`ensure_ascii=False`)
and the committed docs carry `×`, `−`, `—`, `§`; the runner reads an abstract with serde_json,
which takes UTF-8 only. So every text read and write of an abstract, a parameter file, or a
schema names its encoding, and a tool run under `LC_ALL=C` without UTF-8 mode (where Python's
default text encoding is ASCII) writes the same bytes as one run under `C.UTF-8`, and prints
its `…` as an escape rather than failing after the work is done."""
from __future__ import annotations

import os
import subprocess
import sys

from test_builder import BUILDER

SCRIPT = ("from aeiou import *\n"
          "w = Workload('enc', doc='a 2 × 3 grid — one read per cell, r − 1 before it')\n"
          "w.param('n', 3, unit='count', doc='cells × reads')\n"
          "P = w.P\n"
          "ds = w.dataset('d', pattern='d/{id}', layout='whole', access='map', count=10, size=const(1), seed=1)\n"
          "with w.actor('a') as a:\n"
          "    with a.loop('i', P.n):\n"
          "        a.stat(ds.pick())\n")


def _env(locale: str, utf8_mode: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LC_") and k not in ("LANG", "PYTHONIOENCODING", "PYTHONUTF8")}
    return {**env, "PYTHONPATH": str(BUILDER), "LC_ALL": locale, "PYTHONUTF8": utf8_mode}


def _tool(module: str, args, locale: str, utf8_mode: str, cwd):
    return subprocess.run([sys.executable, "-m", module, *map(str, args)], capture_output=True, cwd=str(cwd),
                          env=_env(locale, utf8_mode))


def test_build_and_params_under_an_ascii_locale(tmp_path):
    # LC_ALL=C with PYTHONUTF8=0 is ASCII where glibc has no coercion to C.UTF-8 (LC_ALL set
    # disables Python's own, PEP 538); a host where it is UTF-8 passes without showing anything
    script = tmp_path / "enc.py"
    script.write_text(SCRIPT, encoding="utf-8")
    out = {}
    for locale, mode in (("C.UTF-8", "1"), ("C", "0")):
        d = tmp_path / locale
        d.mkdir()
        r = _tool("aeiou.cli", ["-o", d, script], locale, mode, tmp_path)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        ast = (d / "enc.ast.json").read_bytes()
        r = _tool("aeiou.cli", ["--check", "-o", d, script], locale, mode, tmp_path)
        assert r.returncode == 0 and b"ok   enc.ast.json" in r.stdout, r.stdout + r.stderr
        r = _tool("aeiou.params", ["defaults", d / "enc.ast.json", "-o", d / "enc.params.json"], locale, mode, tmp_path)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        r = _tool("aeiou.params", ["check", d / "enc.ast.json", d / "enc.params.json"], locale, mode, tmp_path)
        assert r.returncode == 0 and b"ok   enc.params.json" in r.stdout, r.stdout + r.stderr
        out[locale] = (ast, (d / "enc.params.json").read_bytes())
    ast, pset = out["C"]
    assert "2 × 3 grid — one read per cell, r − 1".encode("utf-8") in ast   # UTF-8, written as itself
    ast.decode("utf-8")
    assert (ast, pset) == out["C.UTF-8"]
