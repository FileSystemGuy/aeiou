"""The hermetic harness (`aeiou-build --hermetic`, hermetic.py): the stubs and the audit hook
in this process, then what an author sees from the command line.

The stubs replace module attributes process-wide and an audit hook cannot be removed, so the
in-process tests restore every attribute the stubs touch and call the hook directly instead of
adding it."""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import random
import resource
import subprocess
import sys
import time
import uuid

import pytest

from aeiou import hermetic
from aeiou.hermetic import HermeticViolation

from test_builder import BUILDER


@pytest.fixture
def stubbed():
    yield from _stubbing()


def _stubbing():
    """Run `_stub_entropy_and_clocks` and put every attribute it replaces back afterwards."""
    try:
        import numpy.random as npr
    except ImportError:
        npr = None
    saved = [(m, dict(vars(m))) for m in (time, os, uuid, random, datetime, resource) + ((npr,) if npr else ())]
    cls = [(c, {k: c.__dict__[k] for k in ("__init__", "random", "getrandbits", "randbytes") if k in c.__dict__})
           for c in (random.Random, random.SystemRandom)]
    hermetic._stub_entropy_and_clocks()
    try:
        yield npr
    finally:
        for m, d in saved:
            for k in set(vars(m)) - set(d):
                delattr(m, k)
            for k, v in d.items():
                if vars(m).get(k) is not v:
                    setattr(m, k, v)
        for c, d in cls:
            for k, v in d.items():
                setattr(c, k, v)


CLOCKS = [
    lambda: time.time(), lambda: time.time_ns(), lambda: time.monotonic(), lambda: time.monotonic_ns(),
    lambda: time.perf_counter(), lambda: time.perf_counter_ns(), lambda: time.process_time(),
    lambda: time.process_time_ns(), lambda: time.thread_time(), lambda: time.thread_time_ns(),
    lambda: time.clock_gettime(time.CLOCK_REALTIME), lambda: time.clock_gettime_ns(time.CLOCK_REALTIME),
    lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC), lambda: time.localtime(), lambda: time.gmtime(),
    lambda: time.ctime(), lambda: time.asctime(), lambda: time.strftime("%Y"),
    lambda: os.times(), lambda: resource.getrusage(resource.RUSAGE_SELF),
    lambda: datetime.datetime.now(), lambda: datetime.datetime.utcnow(), lambda: datetime.datetime.today(),
    lambda: datetime.date.today(),
]


@pytest.mark.parametrize("read", CLOCKS, ids=range(len(CLOCKS)))
def test_every_clock_is_denied(stubbed, read):
    with pytest.raises(HermeticViolation, match="not available under --hermetic"):
        read()


def test_entropy_and_unseeded_draws_are_denied_and_seeded_ones_work(stubbed):
    for draw in (lambda: os.urandom(4), lambda: uuid.uuid4(), lambda: uuid.uuid1(), lambda: random.randint(1, 9),
                 lambda: random.random(), lambda: random.shuffle([1, 2]), lambda: random.Random(),
                 lambda: random.SystemRandom().random()):
        with pytest.raises(HermeticViolation):
            draw()
    if hasattr(os, "getrandom"):
        with pytest.raises(HermeticViolation):
            os.getrandom(4)
    assert random.Random(7).random() == random.Random(7).random()
    npr = stubbed
    if npr is not None:
        for draw in (lambda: npr.rand(), lambda: npr.default_rng(), lambda: npr.seed(1)):
            with pytest.raises(HermeticViolation):
                draw()
        assert npr.default_rng(7).integers(1 << 30) == npr.default_rng(7).integers(1 << 30)
    # what an abstract may use is untouched
    assert datetime.datetime(2026, 1, 1).year == 2026 and os.getpid() > 0


def test_the_audit_hook_denies_the_network_processes_writes_and_foreign_reads(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("")
    hook = hermetic._audit_hook(script, extra_read_roots=[tmp_path / "schema"])
    for event, args in [("socket.connect", (None, ("127.0.0.1", 9))), ("socket.__new__", ()),
                        ("subprocess.Popen", ("true", ["true"], None, None)), ("os.system", ("true",)),
                        ("os.remove", ("x", -1)), ("os.putenv", ("HOME", "/")),
                        ("ctypes.dlopen", ("/tmp/evil.so",)),
                        ("open", (str(tmp_path / "out.json"), "w", 0)),
                        ("open", (str(tmp_path / "out.json"), None, os.O_WRONLY | os.O_CREAT)),
                        ("open", ("/etc/hostname", "r", 0))]:
        with pytest.raises(HermeticViolation, match="denied under --hermetic"):
            hook(event, args)
    for event, args in [("open", (str(tmp_path / "data.bin"), "rb", 0)),          # the script's directory
                        ("open", (os.fsencode(tmp_path / "data.bin"), "r", 0)),
                        ("open", (str(tmp_path / "schema" / "x.json"), "r", 0)),   # an extra root
                        ("open", (str(pathlib.Path(os.__file__).resolve()), "r", 0)),          # the interpreter's
                        ("open", (3, "r", 0)),                                     # an fd already open
                        ("os.putenv", ("OPENBLAS_MAIN_FREE", "1")), ("os.putenv", (b"GOTOBLAS_MAIN_FREE", b"1")),
                        ("ctypes.dlopen", (None,)), ("ctypes.dlopen", ("libm.so.6",)),
                        ("compile", (None, "<x>"))]:
        hook(event, args)   # allowed: does not raise


# ---- from the command line ----

def _cli(args, cwd, env=None):
    env = {**os.environ, "PYTHONPATH": str(BUILDER), **(env or {})}
    return subprocess.run([sys.executable, "-m", "aeiou.cli", *map(str, args)],
                          capture_output=True, text=True, cwd=str(cwd), env=env)


def _script(tmp_path, line, name="w"):
    script = tmp_path / f"{name}.py"
    script.write_text("import os, time\nfrom aeiou import *\n"
                      f"w = Workload('{name}')\n{line}\n"
                      "ds = w.dataset('d', pattern='d/{id}', layout='whole', access='map', count=10, size=const(1), seed=1)\n"
                      "with w.actor('a') as a:\n"
                      "    a.stat(ds.pick())\n")
    return script


@pytest.mark.parametrize("read, needle", [
    ("time.clock_gettime_ns(time.CLOCK_REALTIME)", "time.clock_gettime_ns"),
    ("int(time.clock_gettime(time.CLOCK_MONOTONIC))", "time.clock_gettime"),
    ("int(os.times().elapsed)", "os.times"),
])
def test_a_clock_in_a_default_is_refused_by_name(tmp_path, read, needle):
    """A default baked from `clock_gettime` used to build under --hermetic, and --twice then
    blamed PYTHONHASHSEED."""
    script = _script(tmp_path, f"w.param('t', {read} % 1000003)")
    r = _cli(["--hermetic", "--twice", "-o", tmp_path, script], cwd=tmp_path)
    assert r.returncode == 1
    assert f"{needle} is not available under --hermetic" in r.stderr, r.stderr
    assert not (tmp_path / "w.ast.json").exists()


def test_the_environment_is_scrubbed(tmp_path):
    script = _script(tmp_path, "assert 'AEIOU_LEAK' not in os.environ and 'HOME' not in os.environ, dict(os.environ)\n"
                               "w.param('e', len(os.environ.get('AEIOU_LEAK', '')))")
    r = _cli(["--hermetic", "-o", tmp_path, script], cwd=tmp_path, env={"AEIOU_LEAK": "xyz"})
    assert r.returncode == 0, r.stderr
    # in this process the variable is there: the scrub, not the script, kept it out
    r = _cli(["-o", tmp_path, script], cwd=tmp_path, env={"AEIOU_LEAK": "xyz"})
    assert r.returncode == 1 and "AssertionError" in r.stderr


def test_a_pid_default_passes_hermetic_and_twice_refuses_it_naming_the_pid(tmp_path):
    script = _script(tmp_path, "w.param('p', os.getpid())")
    assert _cli(["--hermetic", "-o", tmp_path, script], cwd=tmp_path).returncode == 0
    (tmp_path / "w.ast.json").unlink()
    r = _cli(["--hermetic", "--twice", "-o", tmp_path, script], cwd=tmp_path)
    assert r.returncode == 1
    assert "not reproducible" in r.stderr and "the pid" in r.stderr, r.stderr
    assert "under different PYTHONHASHSEED values" not in r.stderr
    assert not (tmp_path / "w.ast.json").exists()


def test_a_failed_child_names_its_script_once(tmp_path):
    script = _script(tmp_path, "Workload('w')", name="w")
    r = _cli(["--hermetic", "-o", tmp_path, script], cwd=tmp_path)
    assert r.returncode == 1
    assert "FAIL w.py: child build failed (PYTHONHASHSEED=1):" in r.stderr, r.stderr
    assert "w.py: w.py:" not in r.stderr
    assert "two Workloads named 'w'" in r.stderr


def test_check_names_an_abstract_never_written(tmp_path):
    script = _script(tmp_path, "")
    r = _cli(["--check", "-o", tmp_path, script], cwd=tmp_path)
    assert r.returncode == 1
    assert f"MISSING {tmp_path / 'w.ast.json'}" in r.stdout, r.stdout
    assert not (tmp_path / "w.ast.json").exists()


def test_a_relative_read_resolves_against_the_working_directory(tmp_path, monkeypatch):
    """`open('data.bin')` in a script reads from the child's cwd, which the harness sets to the
    script's directory: allowed there, denied from anywhere else."""
    (tmp_path / "w").mkdir()
    script = tmp_path / "w" / "s.py"
    script.write_text("")
    hook = hermetic._audit_hook(script)
    monkeypatch.chdir(tmp_path / "w")
    hook("open", ("data.bin", "r", 0))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(HermeticViolation, match=f"read of {tmp_path}/data.bin is denied"):
        hook("open", ("data.bin", "r", 0))


def test_without_numpy_the_stdlib_stubs_still_go_in(monkeypatch):
    """numpy is optional: with it absent (`import numpy.random` raises) the clocks, entropy, and
    `random` are stubbed all the same."""
    monkeypatch.setitem(sys.modules, "numpy.random", None)
    gen = _stubbing()
    assert next(gen) is None
    try:
        for draw in (lambda: time.time(), lambda: os.urandom(1), lambda: random.random(), lambda: random.Random()):
            with pytest.raises(HermeticViolation):
                draw()
    finally:
        next(gen, None)


# ---- the child `aeiou-build --hermetic` spawns ----

def _child(tmp_path, script, *flags):
    """The child's protocol, run with this environment (the parent scrubs it, which would keep
    coverage out): the script's ASTs as one JSON object on stdout, or exit 1 with the reason on
    the last line of stderr."""
    boot = "from aeiou.cli import _child_main; _child_main()"
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    return subprocess.run([sys.executable, "-c", boot, str(script), "--schema-dir", str(BUILDER.parent / "schema"),
                           *flags], capture_output=True, text=True, cwd=str(tmp_path), env=env)


@pytest.mark.parametrize("flags", [(), ("--hermetic",)])
def test_the_child_prints_every_ast_of_its_script(tmp_path, flags):
    script = _script(tmp_path, "w.param('n', 3)")
    script.write_text(script.read_text() + "v = Workload('v')\n"
                      "e = v.dataset('e', pattern='e/{id}', layout='whole', access='map', count=1, size=const(1), seed=1)\n"
                      "with v.actor('b') as b:\n    b.stat(e.pick())\n")
    r = _child(tmp_path, script, *flags)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert list(out) == ["w", "v"] and out["w"]["params"]["n"]["default"] == 3 and out["v"]["name"] == "v"


@pytest.mark.parametrize("body, reason", [
    ("x = 1\n", "BuildError: {script}: no Workload was constructed"),
    ("from aeiou import *\nWorkload('w')\nWorkload('w')\n", "BuildError: two Workloads named 'w'"),
])
def test_a_child_that_fails_ends_its_stderr_with_the_reason(tmp_path, body, reason):
    script = tmp_path / "s.py"
    script.write_text(body)
    r = _child(tmp_path, script)   # unsandboxed: under coverage, the hermetic child cannot save its data, and says so last
    assert r.returncode == 1 and r.stdout == ""
    assert r.stderr.strip().splitlines()[-1].startswith(reason.format(script=script)), r.stderr
