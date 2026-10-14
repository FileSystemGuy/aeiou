"""`AEIOU_REQUIRE` (`conftest.py`): a skip for a capability the job names is a failure, a skip
for one it does not name stays a skip, an unknown name is a usage error, and every skip in the
suite names a capability, so no test can lack one unseen (DESIGN_REVIEW.md §3.75)."""
import os
import pathlib
import re
import shutil
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


def probe(tmp_path, require):
    shutil.copy(TESTS / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(PROBE)
    env = {k: v for k, v in os.environ.items() if k != "AEIOU_REQUIRE"}
    if require is not None:
        env["AEIOU_REQUIRE"] = require
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
