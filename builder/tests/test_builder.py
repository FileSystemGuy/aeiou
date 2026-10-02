"""Builder tests: every abstract builds and validates; committed ASTs match; construction-time
errors fire; the hermetic harness denies what it should; build-twice agrees."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
ABSTRACTS = sorted((BUILDER / "abstracts").glob("*.py"))
EXAMPLES = ROOT / "schema" / "examples"
sys.path.insert(0, str(BUILDER))

from aeiou import *  # noqa: E402
from aeiou import emit, cli  # noqa: E402
from aeiou.validate import validate, op_counts  # noqa: E402


def build_script(path):
    return {wl.name: wl.build() for wl in cli._run_script(path)}


@pytest.mark.parametrize("script", ABSTRACTS, ids=[p.stem for p in ABSTRACTS])
def test_abstract_builds_and_matches_committed(script):
    asts = build_script(script)
    assert asts, "script made no Workload"
    for name, ast in asts.items():
        validate(ast, name)                                   # schema + semantic rules
        assert op_counts(ast)
        committed = EXAMPLES / f"{name}.ast.json"
        assert committed.exists(), f"{committed} missing: run `aeiou-build -o schema/examples {script}`"
        assert emit.sha256(emit.load(committed)) == emit.sha256(ast), \
            f"{name}: committed AST differs from the builder's output; regenerate and review the diff"


def test_canonical_form_matches_check_py():
    """The builder's canonical bytes are the reference checker's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", ROOT / "schema" / "check.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    ast = emit.load(EXAMPLES / "train_small_files.ast.json")
    assert chk.canonical(ast) == emit.canonical(ast)


def test_on_disk_form_round_trips():
    for script in ABSTRACTS:
        for name, ast in build_script(script).items():
            text = emit.render(ast, {"ast_sha256": emit.sha256(ast)})
            back = json.loads(text)
            assert emit.sha256(back) == emit.sha256(ast), name
            assert list(back)[-1] == "provenance" and list(back)[:2] == ["ast", "name"], name


# ---- construction-time discipline ----

def _wl():
    w = Workload("t")
    P = w.params(n=10, xfer=1 * MiB)
    ds = w.dataset("d", pattern="d/{id:06}", count=1000, size=const(1 * MiB), seed=1)
    return w, P, ds


def test_python_control_flow_over_symbolic_values_is_refused():
    w, P, ds = _wl()
    with pytest.raises(TypeError, match="cannot be iterated"):
        for _ in P.n:
            pass
    with pytest.raises(TypeError, match="no truth value"):
        if P.n > 3:
            pass
    with pytest.raises(TypeError, match="symbolic"):
        range(P.n)
    with pytest.raises(TypeError, match="ceil_div"):
        P.n / 2


def test_callables_and_foreign_objects_do_not_enter_nodes():
    w, P, ds = _wl()
    with w.actor("a") as a:
        f = a.let("f", ds.pick())
        with pytest.raises(BuildError, match="callables"):
            a.read(f, lambda: 5)
        with pytest.raises(BuildError, match="cannot use"):
            a.read(f, object())


def test_at_rule_at_construction():
    w, P, ds = _wl()
    w.param("gap", uniform(1, 5))
    w.param("bad", uniform(0, 5))
    with w.actor("a") as a:
        with a.loop("r", P.n) as r:
            d = a.draw("d", P.gap)
            a.let("x", when(d <= r, a.ref("x").at(r - d), 0))          # ok: d >= 1
            a.let("y", when(True, a.ref("y").at(r - 1), 0))            # ok: literal
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", a.ref("z").at(r))
            with pytest.raises(BuildError, match="provably >= 1"):
                e = a.draw("e", P.bad)
                a.let("z", a.ref("z").at(r - e))
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", a.ref("z").at(r + 1))
        with pytest.raises(BuildError, match="outside any loop"):
            a.let("q", a.ref("q").at(0))


def test_as_written_until_eof_needs_the_creating_handle():
    w, P, ds = _wl()
    ns = w.namespace("o", pattern="o/{k}", fields={"k": int}, size="as_written", seed=2)
    with w.actor("a") as a:
        with a.loop("k", P.n) as k:
            c = a.let("c", ns.object(k=k))
            a.open(c, "WRONLY|CREAT")
            a.write(c, P.xfer)
            a.close(c)
            a.open(c, "RDONLY")
            a.read(c, P.xfer, repeat="until_eof")                      # same binding: ok
            other = a.let("other", ns.object(k=k))
            with pytest.raises(BuildError, match="as_written"):
                a.read(other, P.xfer, repeat="until_eof")
    w.build()


def test_misc_construction_errors():
    w, P, ds = _wl()
    with pytest.raises(BuildError, match="reserved"):
        w.param("gpus", 4)
    with pytest.raises(BuildError, match="unknown flag"):
        with w.actor("a") as a:
            a.open(ds.file(0), "RDONLY|BOGUS")
    w, P, ds = _wl()
    with w.actor("a") as a:
        with pytest.raises(BuildError, match="no loop index"):
            with a.every(5):
                pass
        with a.loop("i", P.n):
            with pytest.raises(BuildError, match="shadows"):
                with a.loop("i", P.n):
                    pass
        with pytest.raises(BuildError, match="not declared"):
            a.take("nope")
        with pytest.raises(BuildError, match="errno"):
            a.stat(ds.file(0), expect=["nope"])
    ns = w.namespace("o", pattern="o/{k}", fields={"k": int}, size=0, seed=2)
    with pytest.raises(BuildError, match="fields"):
        ns.object(j=1)
    with pytest.raises(BuildError, match="pattern fields"):
        w.namespace("p", pattern="p/{k}/{j}", fields={"k": int}, size=0, seed=2)


def test_integer_repeat_becomes_a_loop_and_lint_catches_unrolling():
    w, P, ds = _wl()
    with w.actor("a") as a:
        f = a.let("f", ds.file(0))
        a.read(f, P.xfer, repeat=P.n)
    ast = w.build()
    node = ast["actors"]["a"]["body"][1]
    assert "loop" in node and node["loop"]["index"] == "rep" and node["loop"]["to"] == {"param": "n"}
    assert "read" in node["loop"]["body"][0]

    w, P, ds = _wl()
    with w.actor("a") as a:
        f = a.let("f", ds.file(0))
        for _ in range(3):                                            # hand-unrolled
            a.read(f, P.xfer)
    with pytest.raises(BuildError, match="hand-unrolled"):
        w.build()
    w.lint = False
    w.build()


def test_validator_catches_what_construction_cannot():
    w, P, ds = _wl()
    with w.actor("a") as a:
        a.let("f", ds.consume())                                      # consume outside any loop (V6)
    with pytest.raises(BuildError, match="consume"):
        w.build()
    w, P, ds = _wl()
    with w.actor("a") as a:
        with a.loop("i", P.n):
            a.let("x", a.ref("never_bound"))                          # unresolved forward reference (V1)
    with pytest.raises(BuildError, match="never_bound"):
        w.build()


def test_datasets_are_read_only_and_names_are_reserved():
    w, P, ds = _wl()
    with w.actor("a") as a:
        f = a.let("f", ds.file(0))
        a.open(f, "RDONLY")
        with pytest.raises(BuildError, match="read-only"):
            a.open(f, "WRONLY|CREAT")
        with pytest.raises(BuildError, match="read-only"):
            a.write(f, 1)
        with pytest.raises(BuildError, match="read-only"):
            a.unlink(ds.pick())
        with pytest.raises(BuildError, match="read-only"):
            a.rename(ds.file(1), ds.file(2))
    with pytest.raises(BuildError, match="reserved"):
        w.dataset("m", pattern="m/.aeiou-{id}", count=1, size=const(1), seed=1)
    with pytest.raises(BuildError, match="reserved"):
        w.namespace("n", pattern="x/.aeiou-dataset.json/{k}", fields={"k": int}, size=0, seed=1)
    with pytest.raises(BuildError, match="shares root"):
        w.dataset("d2", pattern="d/other_{id:06}", count=1, size=const(1), seed=1)
    with pytest.raises(BuildError, match="inside dataset"):
        w.namespace("n2", pattern="d/out/{k}", fields={"k": int}, size=0, seed=1)
    with pytest.raises(BuildError, match="are nested"):                          # either way round
        w.dataset("d3", pattern="d/sub/f_{id:06}", count=1, size=const(1), seed=1)
    with pytest.raises(BuildError, match="are nested"):
        w.dataset("d4", pattern="top_{id:06}", count=1, size=const(1), seed=1)
    w.namespace("ok", pattern="dd/{k}", fields={"k": int}, size=0, seed=1)      # `dd/` is not under `d/`


def test_check_py_enforces_v12_v13_on_a_hand_written_ast():
    """The reference checker, not only the builder, refuses these."""
    import copy
    from aeiou.validate import _load
    _, check = _load()
    base = emit.load(EXAMPLES / "vdb_search_ivf.ast.json")
    ast = copy.deepcopy(base)
    ast["actors"]["gpu"]["body"][0]["parallel"]["body"][1]["open"]["flags"] = ["RDWR"]
    assert any("V12" in e for e in check.Check(ast, "x").run())
    ast = copy.deepcopy(base)
    ast["datasets"]["lists"]["regions"]["file"] = "ivf/.aeiou-lists.bin"
    assert any("V13" in e for e in check.Check(ast, "x").run())
    ast = copy.deepcopy(base)
    ast["datasets"]["lists2"] = copy.deepcopy(ast["datasets"]["lists"])
    assert any("shares root" in e for e in check.Check(ast, "x").run())
    assert not check.Check(copy.deepcopy(base), "x").run()


# ---- the harness ----

def _run_cli(args, cwd=BUILDER):
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    return subprocess.run([sys.executable, "-m", "aeiou.cli", *map(str, args)],
                          capture_output=True, text=True, cwd=str(cwd), env=env)


def test_hermetic_build_twice(tmp_path):
    script = BUILDER / "abstracts" / "kv_cache_serving.py"
    r = _run_cli(["--hermetic", "--twice", "-o", tmp_path, script])
    assert r.returncode == 0, r.stderr
    out = emit.load(tmp_path / "kv_cache_serving.ast.json")
    assert out["provenance"]["built_twice_identical"] is True
    assert out["provenance"]["ast_sha256"] == emit.sha256(out)
    assert out["provenance"]["generator"]["script"] == "kv_cache_serving.py"


@pytest.mark.parametrize("poison, needle", [
    ("import random\nn = random.randint(1, 9)", "random.randint"),
    ("import time\nt = time.time()", "time.time"),
    ("import os\nb = os.urandom(4)", "os.urandom"),
    ("import uuid\nu = uuid.uuid4()", "uuid.uuid4"),
    ("import datetime\nd = datetime.datetime.now()", "datetime.now"),
    ("import socket\ns = socket.socket()", "socket"),
    ("import subprocess\nsubprocess.run(['true'])", "subprocess"),
    ("open('/etc/hostname').read()", "denied"),
    ("open('leak.txt', 'w').write('x')", "write"),
])
def test_hermetic_denies(tmp_path, poison, needle):
    script = tmp_path / "poisoned.py"
    script.write_text("from aeiou import *\n" + poison + "\n"
                      "w = Workload('p')\nP = w.params(n=3)\n"
                      "ds = w.dataset('d', pattern='d/{id}', count=10, size=const(1), seed=1)\n"
                      "with w.actor('a') as a:\n"
                      "    with a.loop('i', P.n):\n"
                      "        a.stat(ds.pick())\n")
    r = _run_cli(["--hermetic", "-o", tmp_path, script])
    assert r.returncode == 1
    assert needle in r.stderr, r.stderr
    assert not (tmp_path / "p.ast.json").exists()
    # the same script builds without --hermetic (the poison is inert), so the harness is what refused it;
    # run it from tmp_path so the write poison's relative path lands there, not in the repo
    r = _run_cli(["-o", tmp_path, script], cwd=tmp_path)
    assert r.returncode == 0, r.stderr


def test_build_twice_detects_hash_order_dependence(tmp_path):
    script = tmp_path / "unstable.py"
    script.write_text("from aeiou import *\n"
                      "w = Workload('u')\nP = w.params(n=3)\n"
                      "ds = w.dataset('d', pattern='d/{id}', count=10, size=const(1), seed=1)\n"
                      "names = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'}\n"
                      "with w.actor('a') as a:\n"
                      "    with a.loop('i', P.n):\n"
                      "        for nm in names:\n"                  # set order depends on PYTHONHASHSEED
                      "            with a.phase(nm):\n"
                      "                a.stat(ds.pick())\n")
    r = _run_cli(["--twice", "-o", tmp_path, script])
    assert r.returncode == 1
    assert "not reproducible" in r.stderr, r.stderr


def test_check_mode_reports_drift(tmp_path):
    script = BUILDER / "abstracts" / "vdb_search_ivf.py"
    r = _run_cli(["-o", tmp_path, script])
    assert r.returncode == 0, r.stderr
    r = _run_cli(["--check", "-o", tmp_path, script])
    assert r.returncode == 0 and "ok" in r.stdout
    p = tmp_path / "vdb_search_ivf.ast.json"
    doc = emit.load(p)
    doc["params"]["nprobe"]["default"] = 65
    p.write_text(emit.render(doc))
    r = _run_cli(["--check", "-o", tmp_path, script])
    assert r.returncode == 1 and "DRIFT" in r.stdout
