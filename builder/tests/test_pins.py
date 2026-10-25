"""Every tool a build uses is a version the repository names (DESIGN_REVIEW.md §3.82): the
Python (`builder/.python-version`, exact, always one uv downloads, and the one these tests run
under), uv (`required-version`, which CI's setup-uv reads, by checksum), the build backend and
PyInstaller (the lock's), Rust (`rust-toolchain.toml`), nextest, the runner image, and pandoc; no
workflow takes a version the day it runs, and the release ships the lock's requirements and the
Python beside its wheel, as QUICKSTART.md installs them. Then the checkout's own (§3.83): every
package installed is the lock's, at the lock's version (a `pip install` is found here, by the
first test run), each CI job is `./dev.sh check JOB` (the downloads pinned in `dev.sh` alone),
and a tool run from a checkout under another Python warns once, on stderr."""
import importlib.metadata
import pathlib
import re
import shutil
import subprocess
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
        assert "install-action" not in text, name                 # nextest is dev.sh's
        assert "github.com/minio" not in text, name                 # so is MinIO
        for m in re.finditer(r"uses: astral-sh/setup-uv@(\S+)", text):
            assert re.fullmatch(r"v\d+\.\d+\.\d+", m.group(1)), (name, m.group(1))   # no major tag exists
        assert "--with " not in text, name                        # a package beside the lock
        assert "apt-get install" not in text or "strace" in text, name
        assert not re.search(r"^\s+(?:run: )?python3? ", text, re.M), name
        for m in re.finditer(r"uses: astral-sh/setup-uv@\S+\n\s+with:\n((?:\s{10}.*\n)+)", text):
            assert "version-file: builder/pyproject.toml" in m.group(1) and re.search(
                r"checksum: (?:[0-9a-f]{64}|\$\{\{ matrix\.[a-z0-9-]+ \}\})$", m.group(1), re.M), name
        assert text.count("uses: astral-sh/setup-uv@") == len(re.findall(r"uses: astral-sh/setup-uv@\S+\n\s+with:", text)), name
        for m in re.finditer(r"curl [^\n]*-o (\S+) (https://\S+)", text):
            assert re.search(r"(?:[0-9a-f]{64}|\$\{\{ matrix\.[a-z0-9-]+ \}\})  " + re.escape(m.group(1)), text), (name, m.group(2))


def test_dev_sh_pins_every_download_by_version_and_hash():
    text = (ROOT / "dev.sh").read_text(encoding="utf-8")
    pins = re.findall(r"^pin (\S+) (https://\S+) ([0-9a-f]{64})$", text, re.M)
    assert sorted(t for t, _, _ in pins) == ["llvm-cov", "minio", "nextest"], pins
    for tool, url, _ in pins:
        assert re.search(r"\d+\.\d+\.\d+|RELEASE\.\d{4}-", url) and "latest" not in url, (tool, url)


def test_every_ci_job_is_its_dev_sh_check():
    ci = "\n" + workflows()["ci.yml"].split("\njobs:\n", 1)[1]
    jobs = re.findall(r"^  ([a-z-]+):\n", ci, re.M)
    assert jobs == ["abstracts", "runner-tests", "runner-object", "runner-integration"], jobs
    for job, check in zip(jobs, ["abstracts", "runner", "object", "integration"]):
        body = ci.split(f"\n  {job}:\n", 1)[1].split("\n  runner-", 1)[0]
        assert re.search(rf"run: \./dev\.sh check {check}$", body, re.M), job
        runs = " ".join(re.findall(r"run: (.*)", body))
        assert "cargo " not in runs and "pytest" not in runs and ".venv/bin" not in runs, (job, runs)


def test_installed_packages_are_the_lock_s():
    lock = tomllib.loads((BUILDER / "uv.lock").read_text(encoding="utf-8"))
    locked = {}
    for p in lock["package"]:
        locked.setdefault(p["name"], set()).add(p["version"])
    drift = []
    for d in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower()
        if name == "aeiou":
            continue
        if d.version not in locked.get(name, ()):
            drift.append(f"{name} {d.version} (the lock's: {', '.join(sorted(locked.get(name, ()))) or 'none'})")
    assert not drift, "this environment is not uv.lock's; run ./dev.sh setup: " + "; ".join(sorted(drift))


def test_a_checkout_run_under_another_python_warns(tmp_path):
    from aeiou import checkout
    pin = tmp_path / ".python-version"
    pin.write_text("3.12.13\n")
    assert checkout.foreign(pin, (3, 12, 13, "final", 0)) is None
    assert checkout.foreign(tmp_path / "absent", (3, 13, 1, "final", 0)) is None     # a wheel's install
    assert checkout.foreign(pin, (3, 13, 1, "final", 0)) == (
        "this checkout pins Python 3.12.13 (builder/.python-version) and Python 3.13.1 runs it; "
        "./dev.sh setup makes the venv that pins it, and builder/.venv/bin/ has every tool")


def test_every_tool_warns_from_a_checkout_of_another_pin_and_not_from_a_wheel(tmp_path):
    clone, wheel = tmp_path / "builder", tmp_path / "site-packages"
    for d in (clone, wheel):
        shutil.copytree(BUILDER / "aeiou", d / "aeiou", ignore=shutil.ignore_patterns("__pycache__"))
    (clone / ".python-version").write_text("3.0.1\n")
    now = "%d.%d.%d" % sys.version_info[:3]
    for module, prog in [("cli", "aeiou-build"), ("params", "aeiou-params"), ("datagen", "aeiou-datagen"),
                         ("trace", "aeiou-trace"), ("catalog", "aeiou-catalog"), ("header", "aeiou-header")]:
        for where, warns in ((clone, True), (wheel, False)):
            r = subprocess.run([sys.executable, "-m", f"aeiou.{module}", "--help"], capture_output=True, text=True,
                               env={"PYTHONPATH": str(where), "PATH": "/usr/bin:/bin"}, cwd=tmp_path)   # `-m` puts the cwd first
            assert r.returncode == 0, r.stderr
            expect = (f"{prog}: warning: this checkout pins Python 3.0.1 (builder/.python-version) and Python {now} runs it; "
                      "./dev.sh setup makes the venv that pins it, and builder/.venv/bin/ has every tool\n") if warns else ""
            assert r.stderr == expect, (module, where)


def test_the_release_ships_the_lock_and_the_python_and_quickstart_installs_them():
    rel = workflows()["release.yml"]
    assert "uv export --locked" in rel and "wheel/requirements.txt" in rel and "wheel/python-version" in rel
    qs = (ROOT / "QUICKSTART.md").read_text(encoding="utf-8")
    assert '--python "$(cat wheel/python-version)" --managed-python' in qs
    assert "--require-hashes -r wheel/requirements.txt" in qs


def test_the_release_builds_each_architecture_on_its_own_hardware():
    """PyInstaller builds `aeiou-header` for the machine it runs on, so each tarball is built and
    smoke-tested natively, on the glibc 2.35 image, with pandoc for that architecture by checksum;
    a dispatch builds both and publishes nothing (DESIGN_REVIEW.md §3.84)."""
    rel = workflows()["release.yml"]
    legs = re.findall(r"- arch: (\S+)\n\s+runner: (\S+)\n\s+pandoc: (\S+)\n\s+pandoc-sha256: ([0-9a-f]{64})\n"
                      r"\s+uv-sha256: ([0-9a-f]{64})\n", rel)
    assert [(a, r, p) for a, r, p, _, _ in legs] == [("x86_64", "ubuntu-22.04", "linux-amd64"),
                                                  ("aarch64", "ubuntu-22.04-arm", "linux-arm64")], legs
    assert "runs-on: ${{ matrix.runner }}" in rel
    assert 'echo "${{ matrix.pandoc-sha256 }}  /tmp/pandoc.tgz" | sha256sum -c' in rel
    assert "checksum: ${{ matrix.uv-sha256 }}" in rel     # uv's checksum is its architecture's (run 40152549826)
    assert "fail-fast: false" in rel                      # one leg's failure leaves the other's answer
    assert "DIST=aeiou-$ver-${{ matrix.arch }}-linux" in rel
    assert "workflow_dispatch:" in rel
    assert "if: startsWith(github.ref, 'refs/tags/v')" in rel
