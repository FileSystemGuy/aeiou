"""`dev.sh` (the checkout's developer script, POSIX sh; DESIGN_REVIEW.md §3.83) without building
anything: its help, the usage frame for a missing or wrong command, job, or tool (exit 2), the
refusal to start without uv or cargo, naming how to install each (exit 1, before anything runs),
and a pinned download refused when its SHA-256 is not the pin's, with nothing left in the cache.
The jobs themselves are what CI runs, through `./dev.sh check JOB`."""
import os
import pathlib
import shutil
import stat
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEV = ROOT / "dev.sh"
USAGE = "Usage: dev.sh <COMMAND> [ARGS]..."
POINTER = "For more information, try 'dev.sh --help'."
# what the script needs of the system, and nothing of uv's or cargo's (a distribution may put
# rustup's cargo in /usr/bin, so the PATH is these alone)
UTILITIES = ["sh", "dirname", "sed", "cat", "grep", "cut", "uname", "mkdir", "rm", "mv", "chmod", "tar", "gzip",
             "sha256sum", "head", "find", "sort", "mktemp", "diff", "ln"]


def system(tmp_path):
    d = tmp_path / "system-bin"
    if not d.exists():
        d.mkdir()
        for u in UTILITIES:
            (d / u).symlink_to(shutil.which(u, path="/usr/bin:/bin"))
    return str(d)


def dev(*args, path=None, cache=None, cwd=None, tmp_path=None):
    env = {"PATH": path or system(tmp_path), "HOME": os.environ.get("HOME", "/tmp")}
    if cache is not None:
        env["AEIOU_DEV_CACHE"] = str(cache)
    return subprocess.run([str(DEV), *args], capture_output=True, text=True, env=env, cwd=cwd or ROOT)


def frame(message):
    return f"dev.sh: {message}\n\n{USAGE}\n\n{POINTER}\n"


def fake(tmp_path, name, body):
    d = tmp_path / "fake-bin"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return d


def test_help_names_every_command_and_job(tmp_path):
    r = dev("--help", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    for word in ["setup", "check", "cov", "fetch", "abstracts", "runner", "object", "integration", "all",
                 "minio", "nextest", "llvm-cov", "AEIOU_DEV_CACHE", USAGE]:
        assert word in r.stdout, word
    assert dev("-h", tmp_path=tmp_path).stdout == r.stdout


def test_no_command_is_the_suite_s_usage_frame(tmp_path):
    r = dev(tmp_path=tmp_path)
    assert (r.returncode, r.stdout) == (2, "")
    assert r.stderr == frame("the following required arguments were not provided:\n"
                             "  <COMMAND>  one of setup, check, cov, fetch")


def test_a_wrong_command_job_or_tool_is_a_usage_error(tmp_path):
    r = dev("bogus", tmp_path=tmp_path)
    assert (r.returncode, r.stderr) == (2, frame("unrecognized subcommand 'bogus'"))
    r = dev("check", "runner", "bogus", tmp_path=tmp_path)
    assert (r.returncode, r.stderr) == (2, frame("invalid value 'bogus' for '[JOB]...'\n"
                                                 "  [possible values: abstracts, runner, object, integration, all]"))
    r = dev("fetch", tmp_path=tmp_path)
    assert (r.returncode, r.stderr) == (2, frame("the following required arguments were not provided:\n"
                                                 "  <TOOL>  one of minio, nextest, llvm-cov"))
    r = dev("fetch", "pandoc", tmp_path=tmp_path)
    assert (r.returncode, r.stderr) == (2, frame("invalid value 'pandoc' for '<TOOL>'\n"
                                                 "  [possible values: minio, nextest, llvm-cov]"))
    r = dev("setup", "extra", tmp_path=tmp_path)
    assert (r.returncode, r.stderr) == (2, frame("unexpected argument 'extra' found"))
    r = dev("--foo", tmp_path=tmp_path)
    assert (r.returncode, r.stderr) == (2, frame("unexpected argument '--foo' found"))


def test_without_uv_or_cargo_nothing_runs_and_both_installers_are_named(tmp_path):
    # each job needs what its steps run: abstracts no Rust, runner no Python (CI sets up no uv there)
    for args, rust, python in ((["setup"], 1, 1), (["check"], 1, 1), (["check", "abstracts"], 0, 1),
                               (["check", "runner"], 1, 0), (["check", "object"], 1, 1),
                               (["check", "integration"], 1, 1), (["check", "abstracts", "runner"], 1, 1), (["cov"], 1, 1)):
        r = dev(*args, cache=tmp_path / "cache", tmp_path=tmp_path)
        assert r.returncode == 1, (args, r.stderr)
        assert r.stdout == ""
        assert ("https://rustup.rs" in r.stderr) == bool(rust), args
        assert ("https://docs.astral.sh/uv/getting-started/installation/" in r.stderr) == bool(python), args
    assert not (tmp_path / "cache").exists()


def test_a_download_whose_hash_is_not_the_pin_s_is_refused_and_not_kept(tmp_path):
    bad = fake(tmp_path, "curl", 'while [ $# -gt 1 ]; do [ "$1" = -o ] && out=$2; shift; done; echo not-minio > "$out"\n')
    cache = tmp_path / "cache"
    r = dev("fetch", "minio", path=f"{bad}:{system(tmp_path)}", cache=cache)
    assert r.returncode == 1 and r.stdout == ""
    assert "dev.sh: minio: the download's SHA-256 is not the pin's" in r.stderr
    assert not any(cache.rglob("*minio*")), list(cache.rglob("*"))
    # a corrupt file already in the cache is not trusted: it is fetched again, and refused again
    arch = os.uname().machine
    (cache / arch).mkdir(parents=True, exist_ok=True)
    (cache / arch / "minio").write_text("stale")
    r = dev("fetch", "minio", path=f"{bad}:{system(tmp_path)}", cache=cache)
    assert r.returncode == 1 and "SHA-256 is not the pin's" in r.stderr
    assert not (cache / arch / "minio").exists()


def test_a_failed_download_is_reported_as_one(tmp_path):
    down = fake(tmp_path, "curl", "echo 'curl: (6) Could not resolve host' >&2; exit 6\n")
    r = dev("fetch", "nextest", path=f"{down}:{system(tmp_path)}", cache=tmp_path / "cache")
    assert r.returncode == 1
    assert "dev.sh: nextest: the download failed" in r.stderr and "Could not resolve host" in r.stderr
    assert not any((tmp_path / "cache").rglob("*nextest*"))


def test_each_architecture_fetches_its_own_download_into_its_own_cache(tmp_path):
    # a cache on a home directory both kinds of host mount: one host's file never stands for the other's
    log = tmp_path / "urls"
    curl = f'while [ $# -gt 1 ]; do [ "$1" = -o ] && out=$2; shift; done; echo "$1" >> {log}; echo x > "$out"\n'
    fake(tmp_path, "curl", curl)
    cache = tmp_path / "cache"
    for arch, file in (("aarch64", "minio.linux-arm64."), ("x86_64", "minio.linux-amd64.")):
        d = fake(tmp_path, "uname", f"echo {arch}\n")
        (cache / arch).mkdir(parents=True, exist_ok=True)
        (cache / arch / "minio").write_text("stale")
        r = dev("fetch", "minio", path=f"{d}:{system(tmp_path)}", cache=cache)
        assert r.returncode == 1 and "SHA-256 is not the pin's" in r.stderr, r.stderr
        assert file in log.read_text().splitlines()[-1], log.read_text()
        assert not (cache / arch / "minio").exists()


def test_an_architecture_with_no_pinned_download_is_refused(tmp_path):
    d = fake(tmp_path, "uname", "echo riscv64\n")
    r = dev("fetch", "minio", path=f"{d}:{system(tmp_path)}", cache=tmp_path / "cache")
    assert (r.returncode, r.stdout) == (1, "")
    assert r.stderr == "dev.sh: minio: no download is pinned for riscv64, only x86_64 and aarch64\n"
    assert not (tmp_path / "cache").exists()
