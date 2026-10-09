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


def test_at_rule_offset_may_be_an_expression_provably_at_least_one():
    """Contract 0.5 (DESIGN_REVIEW.md §3.57): the offset may be `add` of a term >= 1 and a
    term >= 0, where `mod` (Euclidean in the runner), a loop index whose start is not below
    zero, a literal, and a draw with min >= 0 are >= 0. The rule covers every binding of the
    same loop body, earlier ones included, and both validators (the builder's at construction
    and `schema/check.py` at build) agree."""
    w, P, ds = _wl()
    w.param("gap", uniform(1, 5))
    w.param("bad", uniform(0, 5))
    with w.actor("a") as a:
        with a.loop("r", P.n) as r:
            d = a.draw("d", P.gap)
            kp = a.draw("kp", P.bad)
            a.let("x", when(d <= r, a.ref("x").at(r - (r % d + 1)), 0))   # the per-round draw: r − (r mod d + 1)
            a.let("x2", when(True, a.ref("x2").at(r - (r + d)), 0))       # an index (start 0) plus a draw >= 1
            a.let("x3", when(True, a.ref("x3").at(r - (kp + 1)), 0))      # a draw >= 0 plus a literal >= 1
            a.let("x4", when(d <= r, a.ref("kp").at(r - (r % d + 1)), 0))  # an earlier binding of this body, stepping back: fine
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", when(True, a.ref("z").at(r - (r % d)), 0))     # >= 0 only
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", when(True, a.ref("z").at(r - (kp + kp)), 0))
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", when(True, a.ref("z").at(r - d * 2), 0))       # `mul` is not a form the rule knows
            with pytest.raises(BuildError, match="binding of this loop body"):
                a.let("z", a.ref("kp").at(r + 1))                         # an earlier binding of this body, looking ahead
            with a.loop("j", P.n, start=-2) as j:
                with pytest.raises(BuildError, match="provably >= 1"):
                    a.let("z", when(True, a.ref("z").at(j - (j + 1)), 0))  # an index that may be negative
            with a.loop("k", P.n, start=d) as k:
                a.let("ok", when(True, a.ref("ok").at(k - (k + 1)), 0))    # a start provably >= 0 (here >= 1)
            with a.loop("m", P.n) as m:
                a.let("ok2", a.ref("kp").at(m + 1))                        # a binding of an enclosing body is free
    w.build()


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
    ast["actors"]["gpu"]["body"][5]["open"]["flags"] = ["RDWR"]
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


def test_a_workload_may_declare_its_api_and_cache():
    """Contract 0.6 (`backend` from 0.3 until then): the API of the traced application and its
    cache mode, a run's default backend; `direct` under `mmap` is no backend."""
    with pytest.raises(BuildError, match="api must be one of"):
        Workload("w", api="pread")
    with pytest.raises(BuildError, match="cache must be one of"):
        Workload("w", cache="dontcache")
    with pytest.raises(BuildError, match="page faults"):
        Workload("w", api="mmap", cache="direct")
    w = Workload("w", api="libaio", cache="direct")
    d = w.dataset("d", pattern="d/f_{id:06}", count=1, size=const(1), seed=1)
    with w.actor("a", count=1) as a:
        a.stat(d.file(0))
    ast = w.build()
    assert ast["api"] == "libaio" and ast["cache"] == "direct" and list(ast)[:2] == ["ast", "name"]
    examples = pathlib.Path(__file__).resolve().parents[2] / "schema" / "examples"
    small = json.loads((examples / "train_small_files.ast.json").read_text())
    assert "api" not in small and "cache" not in small and "backend" not in small
    assert json.loads((examples / "model_load.ast.json").read_text())["api"] == "mmap"


def test_protocol_per_dataset_and_namespace():
    """Contract 0.6: a dataset or namespace may live in an object store; namespaces sharing a root
    declare one protocol (V17); the protocol is not part of a dataset's identity."""
    from aeiou.datagen import resolved_dataset
    with pytest.raises(BuildError, match="protocol must be posix or object"):
        Workload("p").namespace("n", pattern="o/{k:04}", fields={"k": int}, size=1, seed=1, protocol="s3")
    w = Workload("p")
    d = w.dataset("d", pattern="d/f_{id:06}", count=1, size=const(1), seed=1, protocol="object")
    w.namespace("a", pattern="o/{k:04}.a", fields={"k": int}, size=1, seed=2, protocol="object")
    with pytest.raises(BuildError, match="V17"):
        w.namespace("b", pattern="o/{k:04}.b", fields={"k": int}, size=1, seed=3)
    with w.actor("x", count=1) as x:
        x.stat(d.file(0))
    ast = w.build()
    assert ast["datasets"]["d"]["files"]["protocol"] == "object" and ast["namespaces"]["a"]["protocol"] == "object"
    assert "protocol" not in resolved_dataset(ast, "d", {})["files"]


def test_same_run_needs_input():
    """Contract 0.4, V15: `same_run` compares a run with the writer's manifest, which only an
    input namespace has; with `input` it is emitted on the namespace."""
    w = Workload("v15")
    with pytest.raises(BuildError, match="V15"):
        w.namespace("out", pattern="o/{k:04}", fields={"k": int}, size=4096, seed=1, same_run=True)
    ns = w.namespace("inp", pattern="i/{k:04}", fields={"k": int}, size=4096, seed=1, input=True, same_run=True)
    assert ns.spec["input"] is True and ns.spec["same_run"] is True
