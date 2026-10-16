"""The writer of `emit.py` that a script run directly reaches (`if __name__ == "__main__":
w.write()`), and the provenance it records with and without a repository, a lockfile, or git."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from aeiou import Workload, const, emit
from test_builder import BUILDER, ast_path


def _tiny(name="tiny"):
    w = Workload(name)
    d = w.dataset("d", pattern="d/{id}", layout="whole", access="map", count=4, size=const(1), seed=1)
    with w.actor("a") as a:
        a.stat(d.pick())
    return w


def test_a_script_run_directly_writes_the_committed_abstract_outside_a_repository(tmp_path):
    """Outside a repository and with no lockfile above it, the provenance has neither `git`
    nor `lock`; the AST is the one `aeiou-build` committed."""
    src = ast_path("file_per_sample").parent / "file_per_sample.py"
    script = tmp_path / src.name
    shutil.copy(src, script)
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads((tmp_path / "file_per_sample.ast.json").read_text())
    prov = doc["provenance"]
    assert prov["ast_sha256"] == emit.sha256(emit.load(ast_path("file_per_sample")))
    assert prov["generator"] == {"script": "file_per_sample.py", "sha256": emit.file_sha256(script),
                                 "builder_version": emit.__version__}
    assert "lock" not in prov and "built_twice_identical" not in prov
    assert emit.git_describe(script) is None and emit.find_lock(tmp_path) is None


def test_write_into_a_directory_names_the_file_for_the_workload_and_finds_the_lock(tmp_path):
    """`w.write(dir)` writes `<dir>/<name>.ast.json`; the source (this test file) is in the
    repository, so `git` is its HEAD and `lock` the builder's `uv.lock`."""
    w = _tiny()
    path = w.write(tmp_path)
    assert path == tmp_path / "tiny.ast.json"
    prov = emit.load(path)["provenance"]
    assert prov["ast_sha256"] == w.sha256()
    assert prov["lock"] == {"file": "uv.lock", "sha256": emit.file_sha256(BUILDER / "uv.lock")}
    head = subprocess.run(["git", "-C", str(BUILDER), "rev-parse", "--short=12", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert prov["generator"]["git"].removesuffix("-dirty") == head


def test_write_without_provenance_and_to_a_named_file(tmp_path):
    w = _tiny("plain")
    path = w.write(tmp_path / "x.json", provenance=False)
    assert path == tmp_path / "x.json"
    doc = emit.load(path)
    assert "provenance" not in doc and emit.render(doc) == path.read_text()


def test_without_git_the_provenance_has_no_git(tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "git")
    monkeypatch.setattr(subprocess, "run", no_git)
    assert emit.git_describe(BUILDER / "aeiou" / "emit.py") is None
    prov = emit.provenance(BUILDER / "aeiou" / "emit.py", {"ast": "0.10"})
    assert "git" not in prov["generator"]


@pytest.mark.parametrize("pre, has_lock", [({"lock": None}, False), ({}, True)])
def test_a_precomputed_lock_of_none_is_kept(pre, has_lock):
    """The hermetic harness precomputes the lock before the sandbox; a precomputed None (no
    lockfile) is an answer, never a reason to look again."""
    script = BUILDER / "aeiou" / "emit.py"
    emit.PRECOMPUTED[str(script.resolve())] = {"git": None, **pre}
    try:
        prov = emit.provenance(script, {"ast": "0.10"})
    finally:
        del emit.PRECOMPUTED[str(script.resolve())]
    assert ("lock" in prov) is has_lock and "git" not in prov["generator"]
