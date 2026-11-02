"""`AEIOU_REQUIRE` (`conftest.py`): a skip for a capability the job names is a failure, a skip
for one it does not name stays a skip, an unknown name is a usage error, and every skip in the
suite names a capability, so no test can lack one unseen (DESIGN_REVIEW.md §3.75). And
`AEIOU_WITHOUT`, what a job deliberately lacks: a test marked `skipif` for it is deselected,
counted by capability, never a skip, and one that skips for it at run time fails, so such a job
reports no skip (2026-11-02, the user's choice: CI's quick job ran 227 tests as skips that the
integration job runs)."""
import os
import pathlib
import re
import subprocess
import sys

from conftest import CAPABILITIES

TESTS = pathlib.Path(__file__).resolve().parent

PROBE = '''
import pytest

needs_runner = pytest.mark.skipif(True, reason="runner binary not built")


@needs_runner
def test_marked():
    pass


def test_called():
    pytest.skip("the runner is not built with the cargo feature `formats`")


def test_minio():
    pytest.skip("TEST_MINIO_BIN is not set (a MinIO server binary)")


def test_library():
    pytest.importorskip("no_such_module_for_aeiou")
'''


def probe(tmp_path, require, without=None):
    # the suite's conftest itself, loaded from where it is (a copy would be another file, its
    # coverage nobody's), its hooks exposed under the names pytest looks for
    (tmp_path / "conftest.py").write_text(
        "import importlib.util\n"
        f"_spec = importlib.util.spec_from_file_location('aeiou_conftest', {str(TESTS / 'conftest.py')!r})\n"
        "_real = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_real)\n"
        "pytest_configure = _real.pytest_configure\n"
        "pytest_runtest_makereport = _real.pytest_runtest_makereport\n"
        "pytest_collection_modifyitems = _real.pytest_collection_modifyitems\n"
        "pytest_terminal_summary = _real.pytest_terminal_summary\n"
        "pytest_sessionfinish = _real.pytest_sessionfinish\n")
    (tmp_path / "test_probe.py").write_text(PROBE)
    env = {k: v for k, v in os.environ.items() if k not in ("AEIOU_REQUIRE", "AEIOU_WITHOUT")}
    if require is not None:
        env["AEIOU_REQUIRE"] = require
    if without is not None:
        env["AEIOU_WITHOUT"] = without
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-rA", str(tmp_path)],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    return r.returncode, r.stdout + r.stderr


def test_a_skip_for_a_required_capability_fails(tmp_path):
    code, out = probe(tmp_path, None)
    assert code == 0 and "4 skipped" in out, out
    code, out = probe(tmp_path, "runner,formats")
    # a `skipif` mark skips in setup, where a failure is counted an error; either fails the run
    assert code == 1 and "1 failed, 2 skipped, 1 error" in out, out
    assert "AEIOU_REQUIRE names `runner`: runner binary not built" in out, out
    assert "AEIOU_REQUIRE names `formats`" in out, out
    code, out = probe(tmp_path, "minio")
    assert code == 1 and "1 failed, 3 skipped" in out, out
    code, out = probe(tmp_path, "runner,gpu")
    assert code != 0 and "AEIOU_REQUIRE: unknown capability gpu" in out, out


def test_a_test_for_a_capability_the_job_is_without_is_deselected(tmp_path):
    code, out = probe(tmp_path, None, "runner")
    # the marked test is deselected; the one that skips for a missing capability at run time
    # stays a skip only for one the job does not say it is without
    assert code == 0 and "3 skipped, 1 deselected" in out, out
    assert "AEIOU_WITHOUT: 1 deselected, needing runner 1" in out, out
    code, out = probe(tmp_path, None, "runner,formats,minio")
    # a run-time skip for one is a failure: mark the test `skipif`, so it is deselected
    assert code == 1 and "2 failed, 1 skipped, 1 deselected" in out, out
    assert "skipped at run time for `formats`, which AEIOU_WITHOUT names: mark the test skipif" in out, out
    code, out = probe(tmp_path, None, "gpu")
    assert code != 0 and "AEIOU_WITHOUT: unknown capability gpu" in out, out
    code, out = probe(tmp_path, "runner", "runner")
    assert code != 0 and "AEIOU_REQUIRE and AEIOU_WITHOUT both name runner" in out, out


def test_every_skip_known_before_a_test_runs_is_a_mark():
    """A run-time `pytest.skip` cannot be deselected by `AEIOU_WITHOUT`, so a skip known before
    the test runs (the runner or `aeiou-trace` not built, no MinIO) is a `skipif` mark. What only
    the binary can say (a cargo feature, `aeiou-header` beside it) is asked at run time, in a
    test marked as needing the runner, which a job without it never runs."""
    runtime = []
    for f in sorted(TESTS.glob("test_*.py")):
        if f.name == "test_require.py":
            continue
        runtime += [(f.name, r) for r in re.findall(r'pytest\.skip\(f?"([^"]+)"', f.read_text())]
    static = [(f, r) for f, r in runtime if any(w in r for w in CAPABILITIES["runner"] + CAPABILITIES["minio"])]
    assert not static, static


def test_every_skip_in_the_suite_names_a_capability():
    """A skip whose reason names no capability could never be required; name one in
    `conftest.CAPABILITIES` (or say the reason with its words)."""
    words = [w for ws in CAPABILITIES.values() for w in ws]
    reasons = []
    for f in sorted(TESTS.glob("test_*.py")):
        if f.name == "test_require.py":
            continue
        text = f.read_text()
        reasons += [(f.name, r) for r in re.findall(r'skipif\([^\n]*reason="([^"]+)"', text)]
        reasons += [(f.name, r) for r in re.findall(r'pytest\.skip\(f?"([^"]+)"', text)]
        reasons += [(f.name, f"could not import '{m}'") for m in re.findall(r'importorskip\("([^"]+)"\)', text)]
    assert reasons, "no skips found: the patterns above no longer match the suite"
    loose = [(f, r) for f, r in reasons if not any(w in r for w in words)]
    assert not loose, loose
