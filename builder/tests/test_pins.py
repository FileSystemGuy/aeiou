"""Every tool a build uses is a version the repository names (DESIGN_REVIEW.md §3.82): the
Python (`builder/.python-version`, exact, always one uv downloads, and the one these tests run
under), uv (`required-version`, which CI's setup-uv reads, by checksum), the build backend and
PyInstaller (the lock's), Rust (`rust-toolchain.toml`), nextest, the runner image, and pandoc; no
workflow takes a version the day it runs, and the release ships the lock's requirements and the
Python beside its wheel, as QUICKSTART.md installs them."""
import pathlib
import re
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
WORKFLOWS = ROOT / ".github" / "workflows"
EXACT = re.compile(r"\d+\.\d+\.\d+")


def pyproject():
    return tomllib.loads((BUILDER / "pyproject.toml").read_text(encoding="utf-8"))


def workflows():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(WORKFLOWS.glob("*.yml"))}


def test_python_is_one_exact_version_and_the_one_running():
    pin = (BUILDER / ".python-version").read_text(encoding="utf-8").strip()
    assert EXACT.fullmatch(pin), pin
    assert "%d.%d.%d" % sys.version_info[:3] == pin


def test_uv_is_pinned_and_takes_only_its_own_python():
    uv = pyproject()["tool"]["uv"]
    assert re.fullmatch(r"==\d+\.\d+\.\d+", uv["required-version"]), uv["required-version"]
    assert uv["python-preference"] == "only-managed"


def test_build_backend_and_release_tools_are_pinned():
    p = pyproject()
    assert all("==" in r for r in p["build-system"]["requires"]), p["build-system"]["requires"]
    assert any(r.startswith("pyinstaller==") for r in p["dependency-groups"]["release"])


def test_rust_is_one_exact_version():
    t = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))["toolchain"]
    assert EXACT.fullmatch(t["channel"]), t["channel"]


def test_no_workflow_takes_a_version_the_day_it_runs():
    for name, text in workflows().items():
        assert "ubuntu-latest" not in text, name
        assert "rust-toolchain@stable" not in text, name
        assert "@nextest" not in text, name                      # taiki-e/install-action's latest
        assert "--with " not in text, name                        # a package beside the lock
        assert "apt-get install" not in text or "strace" in text, name
        assert not re.search(r"^\s+(?:run: )?python3? ", text, re.M), name
        for m in re.finditer(r"uses: astral-sh/setup-uv@\S+\n\s+with:\n((?:\s{10}.*\n)+)", text):
            assert "version-file: builder/pyproject.toml" in m.group(1) and re.search(r"checksum: [0-9a-f]{64}", m.group(1)), name
        assert text.count("uses: astral-sh/setup-uv@") == len(re.findall(r"uses: astral-sh/setup-uv@\S+\n\s+with:", text)), name
        for m in re.finditer(r"tool: nextest@(\S+)", text):
            assert EXACT.fullmatch(m.group(1)), name
        for m in re.finditer(r"curl [^\n]*-o (\S+) (https://\S+)", text):
            assert re.search(r"[0-9a-f]{64}  " + re.escape(m.group(1)), text), (name, m.group(2))


def test_the_release_ships_the_lock_and_the_python_and_quickstart_installs_them():
    rel = workflows()["release.yml"]
    assert "uv export --locked" in rel and "wheel/requirements.txt" in rel and "wheel/python-version" in rel
    qs = (ROOT / "QUICKSTART.md").read_text(encoding="utf-8")
    assert '--python "$(cat wheel/python-version)" --managed-python' in qs
    assert "--require-hashes -r wheel/requirements.txt" in qs
