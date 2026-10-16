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
WORKLOADS = ROOT / "workloads"


def ast_path(name):
    """The committed AST `name`, wherever its mechanism lives under workloads/ (DESIGN_REVIEW.md §3.66)."""
    return next(WORKLOADS.rglob(f"{name}.ast.json"))


ABSTRACTS = sorted(p for p in WORKLOADS.rglob("*.py") if "kit" not in p.relative_to(WORKLOADS).parts)
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
        committed = script.parent / f"{name}.ast.json"
        assert committed.exists(), f"{committed} missing: run `aeiou-build {script}`"
        assert emit.sha256(emit.load(committed)) == emit.sha256(ast), \
            f"{name}: committed AST differs from the builder's output; regenerate and review the diff"


def test_canonical_form_matches_check_py():
    """The builder's canonical bytes are the reference checker's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", ROOT / "schema" / "check.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    ast = emit.load(ast_path("file_per_sample"))
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
    ds = w.dataset("d", pattern="d/{id:06}", layout="whole", access="map", count=1000, size=const(1 * MiB), seed=1)
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
        w.dataset("m", pattern="m/.aeiou-{id}", layout="whole", access="map", count=1, size=const(1), seed=1)
    with pytest.raises(BuildError, match="reserved"):
        w.namespace("n", pattern="x/.aeiou-dataset.json/{k}", fields={"k": int}, size=0, seed=1)
    with pytest.raises(BuildError, match="shares root"):
        w.dataset("d2", pattern="d/other_{id:06}", layout="whole", access="map", count=1, size=const(1), seed=1)
    with pytest.raises(BuildError, match="inside dataset"):
        w.namespace("n2", pattern="d/out/{k}", fields={"k": int}, size=0, seed=1)
    with pytest.raises(BuildError, match="are nested"):                          # either way round
        w.dataset("d3", pattern="d/sub/f_{id:06}", layout="whole", access="map", count=1, size=const(1), seed=1)
    with pytest.raises(BuildError, match="are nested"):
        w.dataset("d4", pattern="top_{id:06}", layout="whole", access="map", count=1, size=const(1), seed=1)
    w.namespace("ok", pattern="dd/{k}", fields={"k": int}, size=0, seed=1)      # `dd/` is not under `d/`


def test_check_py_enforces_v12_v13_on_a_hand_written_ast():
    """The reference checker, not only the builder, refuses these."""
    import copy
    from aeiou.validate import _load
    _, check = _load()
    base = emit.load(ast_path("faiss_ivf_ondisk"))
    ast = copy.deepcopy(base)
    ast["actors"]["gpu"]["body"][5]["open"]["flags"] = ["RDWR"]
    assert any("V12" in e for e in check.Check(ast, "x").run())
    ast = copy.deepcopy(base)
    ast["datasets"]["lists"]["pattern"] = "ivf/.aeiou-lists.bin"
    assert any("V13" in e for e in check.Check(ast, "x").run())
    ast = copy.deepcopy(base)
    ast["datasets"]["lists2"] = copy.deepcopy(ast["datasets"]["lists"])
    assert any("shares root" in e for e in check.Check(ast, "x").run())
    assert not check.Check(copy.deepcopy(base), "x").run()


def test_an_inventory_is_a_constant_name_in_a_consumed_datasets_root():
    """Contract 0.9 (DESIGN_REVIEW.md §3.73): V22 in the builder and in check.py alike."""
    import copy
    from aeiou.validate import _load
    w = Workload("t")
    P = w.params(entry=24)

    def ds(name, pattern, access="map", inv="inventory.txt"):
        return w.dataset(name, pattern=pattern, layout="whole", access=access, count=10, size=const(1), seed=1,
                         inventory={"name": inv, "entry": P.entry})
    with pytest.raises(BuildError, match="only a consumed dataset"):
        ds("a", "a/{id}", access="indexed")
    with pytest.raises(BuildError, match="not a file name"):
        ds("b", "b/{id}", inv="x/list.txt")
    with pytest.raises(BuildError, match="reserved"):
        ds("c", "c/{id}", inv=".aeiou-list")
    with pytest.raises(BuildError, match="can produce"):
        ds("e", "e/f_{id:04}.jpg", inv="f_0012.jpg")
    with pytest.raises(BuildError, match="declares none"):
        w.dataset("plain", pattern="p/{id}", layout="whole", access="map", count=1, size=const(1), seed=1).inventory()
    train = ds("train", "train/{id div 1300:05}/img_{id:09}.jpg")
    with w.actor("gpu") as g:
        inv = g.let("inv", train.inventory())
        g.open(inv, "RDONLY")
        g.read(inv, 1 * MiB, repeat="until_eof")
        with pytest.raises(BuildError, match="read-only"):
            g.write(inv, 1)
        g.close(inv)
    ast = w.build()
    assert ast["ast"] == "0.10"
    assert ast["datasets"]["train"]["inventory"] == {"name": "inventory.txt", "entry": {"param": "entry"}}
    assert ast["actors"]["gpu"]["body"][0]["let"]["value"] == {"inventory": "train"}
    _, check = _load()
    assert not check.Check(copy.deepcopy(ast), "x").run()
    for mutate, needle in [
        (lambda a: a["datasets"]["train"].__setitem__("access", "indexed"), "only a consumed dataset"),
        (lambda a: a["datasets"]["train"]["inventory"].__setitem__("name", "01234"), "can produce in the dataset's root (V22)"),
        (lambda a: a["datasets"]["train"]["inventory"].__setitem__("name", ".aeiou-x"), "V13"),
        (lambda a: a["datasets"]["train"]["inventory"].__setitem__("name", "a/b"), "not a file name"),
        (lambda a: a["datasets"]["train"].pop("inventory"), "which declares none (V22)"),
    ]:
        bad = copy.deepcopy(ast)
        mutate(bad)
        assert any(needle in e for e in check.Check(bad, "x").run()), needle


def test_a_divisor_is_a_declared_integer_parameter():
    """Contract 0.10 (DESIGN_REVIEW.md §3.75): V23 in the builder and in check.py alike, and the
    Python port binds a divisor as the runner does, so the resolved definition names the width."""
    import copy
    from aeiou.validate import _load
    from aeiou.pattern import Pattern
    from aeiou.datagen import resolved_dataset
    w = Workload("t")
    P = w.params(files_per_directory=1300, ratio=0.5)
    with pytest.raises(BuildError, match="divisor `width` is not a declared parameter"):
        w.dataset("a", pattern="a/{id div width}/f_{id}", layout="whole", access="map", count=10, size=const(1), seed=1)
    with pytest.raises(BuildError, match="whose default is not an integer"):
        w.dataset("b", pattern="b/{id div ratio}/f_{id}", layout="whole", access="map", count=10, size=const(1), seed=1)
    train = w.dataset("train", pattern="train/{id div files_per_directory:05}/img_{id:09}.jpg", layout="whole",
                      access="map", count=5000, size=const(1), seed=1)
    with w.actor("gpu") as g:
        with g.loop("d", train.dirs) as d:
            g.stat(g.let("dh", train.dir(d)))
    ast = w.build()
    assert ast["datasets"]["train"]["pattern"] == "train/{id div files_per_directory:05}/img_{id:09}.jpg"
    _, check = _load()
    assert not check.Check(copy.deepcopy(ast), "x").run()
    for mutate, needle in [
        (lambda a: a["params"].pop("files_per_directory"), "divisor `files_per_directory` is not a declared parameter (V23)"),
        (lambda a: a["params"]["files_per_directory"].__setitem__("default", 2.5), "whose default is not an integer (V23)"),
    ]:
        bad = copy.deepcopy(ast)
        mutate(bad)
        assert any(needle in e for e in check.Check(bad, "x").run()), needle
    p = Pattern(ast["datasets"]["train"]["pattern"])
    assert p.params() == ["files_per_directory"]
    with pytest.raises(ValueError, match="not bound"):
        p.format(id=1)
    assert p.bind({"files_per_directory": 4096}).format(id=4096) == "train/00001/img_000004096.jpg"
    with pytest.raises(ValueError, match="positive divisor"):
        p.bind({"files_per_directory": 0})
    assert resolved_dataset(ast, "train", {"files_per_directory": 1300})["pattern"] == "train/{id div 1300:05}/img_{id:09}.jpg"


def test_a_sub_actor_positions_its_ops_on_an_inherited_handle():
    """V24 (2026-10-14) in check.py as in the runner's validator: in a `parallel` or `loader`
    body, a read or write without an offset, or an lseek, names a handle the body opened."""
    import copy
    from aeiou.validate import _load
    _, check = _load()
    f = {"file": {"dataset": "d", "id": 0}}
    ast = {"ast": "0.10", "name": "t",
           "datasets": {"d": {"pattern": "d/f{id}", "layout": "whole", "access": "indexed", "count": 1,
                              "size": {"const": 4096}, "seed": 1}},
           "actors": {"gpu": {"body": [
               {"open": {"file": f, "flags": ["RDONLY"]}},
               {"parallel": {"index": "p", "width": 2, "body": [
                   {"read": {"file": f, "len": 512, "offset": 0}},
                   {"open": {"file": f, "flags": ["RDONLY"]}},
                   {"lseek": {"file": f, "offset": 512, "whence": "SET"}},
                   {"read": {"file": f, "len": 512}},
                   {"close": {"file": f}}]}},
               {"read": {"file": f, "len": 512}},
               {"close": {"file": f}}]}}}
    assert not check.Check(copy.deepcopy(ast), "x").run()
    body = lambda a: a["actors"]["gpu"]["body"][1]["parallel"]["body"]
    for mutate, needle in [
        (lambda a: body(a)[0]["read"].pop("offset"), "parallel/body/0/read/file: read that uses the file position"),
        (lambda a: body(a).insert(0, {"lseek": {"file": f, "offset": 0, "whence": "SET"}}), "lseek that uses the file position"),
        (lambda a: body(a).pop(1), "lseek that uses the file position"),
        # an inner fork shares what the outer body opened
        (lambda a: body(a).insert(4, {"parallel": {"index": "q", "width": 2, "body": [{"read": {"file": f, "len": 1}}]}}),
         "parallel/body/4/parallel/body/0/read/file"),
    ]:
        bad = copy.deepcopy(ast)
        mutate(bad)
        errs = check.Check(bad, "x").run()
        assert any(needle in e and "(V24)" in e for e in errs), (needle, errs)


# ---- the harness ----

def _run_cli(args, cwd=BUILDER):
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    return subprocess.run([sys.executable, "-m", "aeiou.cli", *map(str, args)],
                          capture_output=True, text=True, cwd=str(cwd), env=env)


def test_hermetic_build_twice(tmp_path):
    script = ast_path("lmcache_local_disk").parent / "lmcache_local_disk.py"
    r = _run_cli(["--hermetic", "--twice", "-o", tmp_path, script])
    assert r.returncode == 0, r.stderr
    out = emit.load(tmp_path / "lmcache_local_disk.ast.json")
    assert out["provenance"]["built_twice_identical"] is True
    assert out["provenance"]["ast_sha256"] == emit.sha256(out)
    assert out["provenance"]["generator"]["script"] == "lmcache_local_disk.py"


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
                      "ds = w.dataset('d', pattern='d/{id}', layout='whole', access='map', count=10, size=const(1), seed=1)\n"
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
                      "ds = w.dataset('d', pattern='d/{id}', layout='whole', access='map', count=10, size=const(1), seed=1)\n"
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
    script = ast_path("faiss_ivf_ondisk").parent / "faiss_ivf_ondisk.py"
    r = _run_cli(["-o", tmp_path, script])
    assert r.returncode == 0, r.stderr
    r = _run_cli(["--check", "-o", tmp_path, script])
    assert r.returncode == 0 and "ok" in r.stdout
    p = tmp_path / "faiss_ivf_ondisk.ast.json"
    doc = emit.load(p)
    doc["params"]["nprobe"]["default"] = 65
    p.write_text(emit.render(doc))
    r = _run_cli(["--check", "-o", tmp_path, script])
    assert r.returncode == 1 and "DRIFT" in r.stdout


def test_a_workload_may_declare_its_apis_and_cache():
    """Contract 0.7 (`api` in 0.6, `backend` from 0.3): the APIs of the traced application, one
    per protocol, and its cache mode, a run's defaults; `direct` under `mmap` is no backend."""
    with pytest.raises(BuildError, match="posix must be one of"):
        Workload("w", posix="pread")
    with pytest.raises(BuildError, match="cache must be one of"):
        Workload("w", cache="dontcache")
    with pytest.raises(BuildError, match="s3 must be one of"):
        Workload("w", s3="threads")
    with pytest.raises(BuildError, match="page faults"):
        Workload("w", posix="mmap", cache="direct")
    w = Workload("w", posix="libaio", cache="direct")
    d = w.dataset("d", pattern="d/f_{id:06}", layout="whole", access="map", count=1, size=const(1), seed=1)
    with w.actor("a", count=1) as a:
        a.stat(d.file(0))
    ast = w.build()
    assert ast["posix"] == "libaio" and ast["cache"] == "direct" and list(ast)[:2] == ["ast", "name"]
    small = json.loads(ast_path("file_per_sample").read_text())
    assert not {"posix", "api", "cache", "s3", "backend"} & set(small)
    assert json.loads(ast_path("safetensors_load").read_text())["posix"] == "mmap"


def test_v19_apis_follow_the_protocols():
    """V19: an API is declared only for a protocol with names, and posix and s3 names together
    take APIs of one kind (event-driven or a thread per actor), defaults counted."""
    def build(protocols, **apis):
        w = Workload("v19", **apis)
        ds = [w.dataset(f"d{i}", pattern=f"d{i}/f_{{id:06}}", layout="whole", access="map", count=1, size=const(1), seed=1, protocol=p)
              for i, p in enumerate(protocols)]
        with w.actor("a", count=1) as a:
            for d in ds:
                a.stat(d.file(0))
        return w.build()

    with pytest.raises(BuildError, match=r"no dataset or namespace is `protocol: s3` \(V19\)"):
        build(["posix"], s3="async")
    with pytest.raises(BuildError, match=r"every dataset and namespace is `protocol: s3` \(V19\)"):
        build(["s3"], posix="io_uring")
    with pytest.raises(BuildError, match=r"every dataset and namespace is `protocol: s3` \(V19\)"):
        build(["s3"], cache="direct")
    with pytest.raises(BuildError, match="one is event-driven and the other a thread per actor"):
        build(["posix", "s3"], posix="io_uring")
    with pytest.raises(BuildError, match="one is event-driven and the other a thread per actor"):
        build(["posix", "s3"], s3="async")
    assert build(["s3"], s3="async")["s3"] == "async"
    assert build(["posix", "s3"], posix="libaio", s3="async")["posix"] == "libaio"
    assert "s3" not in build(["posix", "s3"], posix="mmap")


def test_protocol_per_dataset_and_namespace():
    """Contract 0.6 (`s3` was `object` until 0.7): a dataset or namespace may live in an S3
    store; namespaces sharing a root declare one protocol (V17); the protocol is not part of a
    dataset's identity."""
    from aeiou.datagen import resolved_dataset
    with pytest.raises(BuildError, match="protocol must be posix or s3"):
        Workload("p").namespace("n", pattern="o/{k:04}", fields={"k": int}, size=1, seed=1, protocol="object")
    w = Workload("p")
    d = w.dataset("d", pattern="d/f_{id:06}", layout="whole", access="map", count=1, size=const(1), seed=1, protocol="s3")
    w.namespace("a", pattern="o/{k:04}.a", fields={"k": int}, size=1, seed=2, protocol="s3")
    with pytest.raises(BuildError, match="V17"):
        w.namespace("b", pattern="o/{k:04}.b", fields={"k": int}, size=1, seed=3)
    with w.actor("x", count=1) as x:
        x.stat(d.file(0))
    ast = w.build()
    assert ast["datasets"]["d"]["protocol"] == "s3" and ast["namespaces"]["a"]["protocol"] == "s3"
    assert "protocol" not in resolved_dataset(ast, "d", {})


def test_v18_an_object_is_written_once_in_order():
    """V18: an object of a namespace in an object store is written once, in order from 0, by
    one upload, so `O_APPEND`, `O_RDWR`, `O_EXCL`, `ftruncate`, and `fallocate` on one are
    refused, by the builder and by the reference checker; a posix namespace keeps them."""
    import copy
    from aeiou.validate import _load
    _, check = _load()

    def build(protocol, body):
        w = Workload("v18")
        ns = w.namespace("o", pattern="o/{k:04}", fields={"k": int}, size="as_written", seed=1, protocol=protocol)
        with w.actor("x", count=1) as x:
            body(x, ns.object(k=0))
        return w.build()

    def plain(x, f):
        x.open(f, "WRONLY|CREAT|TRUNC")
        x.write(f, 4096)
        x.close(f)

    for bad in ("APPEND", "RDWR", "EXCL"):
        with pytest.raises(BuildError, match="V18"):
            build("s3", lambda x, f: x.open(f, f"WRONLY|CREAT|{bad}"))
    with pytest.raises(BuildError, match="V18"):
        build("s3", lambda x, f: x.ftruncate(f, 0))
    with pytest.raises(BuildError, match="V18"):
        build("s3", lambda x, f: x.fallocate(f, 4096))
    build(None, lambda x, f: x.open(f, "RDWR|CREAT"))
    ast = build("s3", plain)
    assert not check.Check(copy.deepcopy(ast), "x").run()
    ast["actors"]["x"]["body"][0]["open"]["flags"].append("APPEND")
    assert any("V18" in e for e in check.Check(ast, "x").run())
    # an upload is begun by WRONLY alone: CREAT or TRUNC on a read-only handle would make or
    # empty an object the handle never writes (RC1 review, bug R30)
    for bad in ("RDONLY|CREAT", "RDONLY|TRUNC", "CREAT"):
        with pytest.raises(BuildError, match="V18"):
            build("s3", lambda x, f: x.open(f, bad))
    build(None, lambda x, f: (x.open(f, "RDONLY|CREAT"), x.close(f)))
    ast = build("s3", plain)
    ast["actors"]["x"]["body"][0]["open"]["flags"] = ["RDONLY", "TRUNC"]
    assert any("open with TRUNC without WRONLY" in e and "(V18)" in e for e in check.Check(ast, "x").run())
    # the builder refuses in the checker's words (it said `declared protocol object`, the
    # protocol's name before contract 0.7)
    for flags, edit in [("RDONLY|TRUNC", None), ("WRONLY|CREAT|APPEND", None),
                        (None, lambda b: b.insert(1, {"ftruncate": {"file": b[0]["open"]["file"], "len": 0}})),
                        (None, lambda b: b.insert(1, {"fallocate": {"file": b[0]["open"]["file"], "len": 1}}))]:
        doc = build("s3", plain)
        body = doc["actors"]["x"]["body"]
        if flags:
            body[0]["open"]["flags"] = flags.split("|")
            with pytest.raises(BuildError) as e:
                build("s3", lambda x, f: (x.open(f, flags), x.close(f)))
        else:
            edit(body)
            kind = next(iter(body[1]))
            with pytest.raises(BuildError) as e:
                build("s3", lambda x, f: getattr(x, kind)(f, body[1][kind]["len"]))
        said = [m.split(": ", 2)[2] for m in check.Check(doc, "x").run() if "(V18)" in m]   # `x: path: why`
        assert said == [str(e.value)], (said, str(e.value))

    # the upload is sent by the close, so an s3 name opened for writing is closed later in the
    # same body, on the same handle, no `let` rebinding a name it reads in between (bug R29)
    def unclosed(x, f):
        x.open(f, "WRONLY|CREAT|TRUNC")
        x.write(f, 4096)
    with pytest.raises(BuildError, match="no close of that handle later in the same body"):
        build("s3", unclosed)
    build(None, unclosed)

    def nested(x, f):
        x.open(f, "WRONLY|CREAT|TRUNC")
        x.write(f, 4096)
        with x.when(gpu_id == 0):
            x.close(f)
        with x.otherwise():
            x.close(f)
    with pytest.raises(BuildError, match="V18"):
        build("s3", nested)

    def across_loop(x, f):
        x.open(f, "WRONLY|CREAT|TRUNC")
        with x.loop("i", 3):
            x.write(f, 4096)
        x.close(f)
    build("s3", across_loop)

    def reopened(x, f):
        x.open(f, "WRONLY|CREAT|TRUNC")
        x.write(f, 4096)
        x.open(f, "WRONLY|CREAT|TRUNC")
        x.write(f, 4096)
        x.close(f)
    with pytest.raises(BuildError, match="before it is opened again"):
        build("s3", reopened)
    ast = build("s3", plain)
    body = ast["actors"]["x"]["body"]
    body.insert(2, {"let": {"name": "g", "value": 1}})
    assert not check.Check(copy.deepcopy(ast), "x").run()
    ast["actors"]["x"]["body"] = [{"let": {"name": "f", "value": body[0]["open"]["file"]}},
                                  {"open": {"file": {"ref": "f"}, "flags": ["WRONLY", "CREAT"]}},
                                  {"let": {"name": "f", "value": body[0]["open"]["file"]}},
                                  {"close": {"file": {"ref": "f"}}}]
    assert any("no close of that handle" in e for e in check.Check(ast, "x").run())


def test_same_run_needs_input():
    """Contract 0.4, V15: `same_run` compares a run with the writer's manifest, which only an
    input namespace has; with `input` it is emitted on the namespace."""
    w = Workload("v15")
    with pytest.raises(BuildError, match="V15"):
        w.namespace("out", pattern="o/{k:04}", fields={"k": int}, size=4096, seed=1, same_run=True)
    ns = w.namespace("inp", pattern="i/{k:04}", fields={"k": int}, size=4096, seed=1, input=True, same_run=True)
    assert ns.spec["input"] is True and ns.spec["same_run"] is True


def test_build_refuses_two_workloads_of_one_name(tmp_path):
    """An abstract's name is its file's: two Workloads of one name, in one script or in two
    scripts written to one directory, are refused, never one silently written over the other."""
    body = ("w = Workload('dup')\n"
            "ds = w.dataset('d', pattern='d/f{{id}}', layout='whole', access='indexed', count=4, size=const({n}), seed=1)\n"
            "with w.actor('gpu', count=1) as c:\n"
            "    f = c.let('f', ds.file(0))\n"
            "    c.open(f, 'RDONLY')\n"
            "    c.close(f)\n")
    one = tmp_path / "one.py"
    one.write_text("from aeiou import *\nfor n in (4096, 8192):\n"
                   + "".join("    " + line + "\n" for line in body.format(n="n").splitlines()))
    for flags in ([], ["--hermetic"]):
        r = _run_cli([*flags, "-o", tmp_path / "out", one])
        assert r.returncode == 1, (flags, r.stdout, r.stderr)
        assert "two Workloads named 'dup'" in r.stderr, (flags, r.stderr)
        assert not (tmp_path / "out" / "dup.ast.json").exists(), flags
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "s.py").write_text("from aeiou import *\n" + body.format(n=4096))
    (tmp_path / "b" / "s.py").write_text("from aeiou import *\n" + body.format(n=8192))
    r = _run_cli(["-o", tmp_path / "two", tmp_path / "a" / "s.py", tmp_path / "b" / "s.py"])
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "also authored by" in r.stderr and str(tmp_path / "a" / "s.py") in r.stderr, r.stderr
    written = (tmp_path / "two" / "dup.ast.json").read_text()   # the first script's, not overwritten
    assert "4096" in written and "8192" not in written
    # each beside its own script: two directories, two files, no clash; one script named twice is one author
    r = _run_cli([tmp_path / "a" / "s.py", tmp_path / "b" / "s.py", tmp_path / "a" / "s.py"])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "a" / "dup.ast.json").exists() and (tmp_path / "b" / "dup.ast.json").exists()


def test_literal_distribution_arguments_the_runner_cannot_draw_from_are_refused():
    """A literal argument that can only fail or mean nothing at run time is a BuildError naming
    the distribution, never an abstract that passes `aeiou check` and fails in `dry-run`
    (RC1 review): the runner's uniform is [lo, hi), its weights must total above zero."""
    nan, inf = float("nan"), float("inf")
    refused = [
        (lambda: uniform(4096, 4096), "uniform: empty range"),
        (lambda: uniform(8192, 4096), "uniform: empty range"),
        (lambda: empirical([4096, 8192], [0, 0]), "empirical: total weight is zero"),
        (lambda: empirical([4096, 8192], [1, -1]), "empirical weight"),
        (lambda: mixture((0, none), (0, const(4096))), "mixture: total weight is zero"),
        (lambda: mixture((nan, none), (1, const(4096))), "mixture weight"),
        (lambda: mixture((inf, none), (1, const(4096))), "mixture weight"),
        (lambda: lognormal(-5, 1.0), "lognormal: median -5"),
        (lambda: lognormal(0, 1.0), "lognormal: median 0"),
        (lambda: lognormal(4096, inf), "lognormal sigma"),
        (lambda: lognormal(4096, 1.0, min=8192, max=4096), "lognormal: min 8192 is above max 4096"),
        (lambda: normal(4096, -3), "normal: sd -3"),
        (lambda: normal(4096, 10, min=8192, max=4096), "normal: min 8192 is above max 4096"),
        (lambda: zipf(inf), "zipf s"),
        (lambda: zipf(nan), "zipf s"),
    ]
    for mk, msg in refused:
        with pytest.raises(BuildError, match=msg):
            mk()
    # the edges the runner draws from, and parameter references, which the runner checks
    w, P, ds = _wl()
    uniform(4095, 4096), normal(4096, 0), normal(4096, 1, min=4096, max=4096)
    empirical([1, 2], [0, 1]), mixture((0, none), (1, const(1)))
    uniform(P.n, P.n), normal(4096, P.n, min=P.n, max=0), lognormal(P.n, 1.0)


def test_choose_weights_and_divisor_defaults_the_runner_cannot_use_are_refused():
    """`choose` arms of zero total weight or non-finite weight, and a `{id div D}` divisor
    whose default is not positive, are BuildErrors, not a dry-run failure or a bare
    ValueError from json (RC1 review)."""
    for weights, msg in (([0, 0], "choose: total weight is zero"),
                         ([float("nan"), 1], "arm weight"), ([float("inf"), 1], "arm weight")):
        w, P, ds = _wl()
        with w.actor("gpu", count=1) as c:
            with pytest.raises(BuildError, match=msg):
                with c.choose() as ch:
                    for a in weights:
                        with ch.arm(a):
                            c.compute(1)
    for d in (0, -2):
        w = Workload("t")
        w.param("dd", d)
        with pytest.raises(BuildError, match=f"divisor `dd` is a parameter whose default {d} is not positive"):
            w.dataset("d", pattern="d{id div dd}/f{id}", layout="whole", access="indexed", count=4,
                      size=const(4096), seed=1)
    w = Workload("t")
    w.param("dd", 2)
    w.dataset("d", pattern="d{id div dd}/f{id}", layout="whole", access="indexed", count=4,
              size=const(4096), seed=1)


def test_check_py_v12_refuses_mkdir_and_rmdir_on_a_dataset_directory():
    """V12 covers a dataset's directories as the builder and V14 do: a hand-written AST that
    creates or removes one is refused by the reference checker (RC1 review)."""
    import copy
    from aeiou.validate import _load
    _, check = _load()
    base = {"ast": "0.10", "name": "m", "params": {},
            "datasets": {"d": {"pattern": "d/f{id}", "layout": "whole", "access": "map",
                               "count": 8, "size": {"const": 4096}, "seed": 1}},
            "actors": {"gpu": {"count": 1, "body": []}}}
    assert check.Check(copy.deepcopy(base), "x").run() == []
    for kind in ("mkdir", "rmdir"):
        ast = copy.deepcopy(base)
        ast["actors"]["gpu"]["body"] = [{kind: {"dir": {"dir": {"dataset": "d", "id": 0}}}}]
        errs = check.Check(ast, "x").run()
        assert any(f"{kind} on dataset `d`: datasets are read-only (V12)" in e for e in errs), errs


def _v3_chain(dist, params=None):
    """The `x @ i - d` loop of the rule V3 review (RC1, bug 30), `d` drawn from `dist`."""
    return {
        "ast": "0.10", "name": "v3_chain", "params": params or {},
        "datasets": {"d": {"pattern": "d/f{id}", "layout": "whole", "access": "indexed", "count": 4,
                           "size": {"const": 4096}, "seed": 1}},
        "actors": {"gpu": {"count": 1, "body": [{"loop": {"index": "i", "to": 4, "body": [
            {"let": {"name": "d", "value": {"draw": dist}}},
            {"let": {"name": "x", "value": {"cond": {"if": {"gt": [{"index": "i"}, 0]},
             "then": {"add": [{"at": {"ref": "x", "index": {"sub": [{"index": "i"}, {"ref": "d"}]}}}, 1]},
             "else": 0}}}},
            {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": 0}}}},
            {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
            {"read": {"file": {"ref": "f"}, "len": {"add": [{"ref": "x"}, 1]}, "offset": 0}},
            {"close": {"file": {"ref": "f"}}}]}}]}},
    }


def test_at_rule_believes_no_minimum_a_maximum_undercuts():
    """The runner clamps a normal or lognormal to `min` and then to `max`, so a `max` below
    `min` makes every draw `max`; and a mixture whose every arm is `none` draws no number.
    Neither is an offset provably >= 1 (V3), in check.py or in the builder (RC1, bug 30)."""
    from aeiou.validate import _load
    _, check = _load()

    def errs(dist, params=None):
        return check.Check(_v3_chain(dist, params), "x").run()

    e = errs({"normal": {"mean": 5, "sd": 1, "min": 1, "max": 0}})
    assert any("normal: min 1 is above max 0" in x for x in e) and any("provably >= 1" in x for x in e), e
    e = errs({"lognormal": {"median": 5, "sigma": 1.0, "min": 2.5, "max": 1}})
    assert any("lognormal: min 2.5 is above max 1" in x for x in e) and any("provably >= 1" in x for x in e), e
    assert any("provably >= 1" in x for x in errs({"mixture": [{"weight": 1, "dist": None}]}))
    assert not errs({"normal": {"mean": 5, "sd": 1, "min": 1, "max": 9}})
    assert not errs({"mixture": [{"weight": 1, "dist": None}, {"weight": 1, "dist": {"const": 2}}]})
    # a `max` that is a parameter is judged on its default (the runner judges the value given)
    hi = {"normal": {"mean": 5, "sd": 1, "min": 1, "max": {"param": "hi"}}}
    assert not errs(hi, {"hi": {"default": 9}})
    e = errs(hi, {"hi": {"default": 0}})
    assert any("normal: min 1 is above max 0" in x for x in e) and any("provably >= 1" in x for x in e), e
    # the builder: the same judgement at construction
    w = Workload("t")
    P = w.params(n=4, hi=9, lo=0)
    with w.actor("a") as a:
        with a.loop("r", P.n) as r:
            d = a.draw("d", normal(5, 1, min=1, max=P.hi))
            a.let("x", when(d <= r, a.ref("x").at(r - d), 0))
            e2 = a.draw("e", normal(5, 1, min=1, max=P.lo))
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", when(True, a.ref("z").at(r - e2), 0))
            e3 = a.draw("e3", mixture((1, none)))
            with pytest.raises(BuildError, match="provably >= 1"):
                a.let("z", when(True, a.ref("z").at(r - e3), 0))


def _consume_twice(shape):
    """A workload whose dataset `d` is consumed by two `consume` nodes: in one loop body, in two
    sibling loops (a train phase, then an eval phase), or in two actors."""
    w = Workload("t")
    d = w.dataset("d", pattern="d/f{id:04}", layout="whole", access="map", count=16, size=const(1), seed=1)
    with w.actor("gpu") as a:
        with a.loop("j", 4):
            x = a.let("x", d.consume())
            a.stat(x)
            if shape == "body":
                y = a.let("y", d.consume())
                a.stat(y)
            elif shape == "one":
                a.stat(x)
        if shape == "siblings":
            with a.loop("k", 2):
                a.stat(d.consume())
    if shape == "actors":
        with w.actor("eval") as e:
            with e.loop("k", 2):
                e.stat(d.consume())
    return w


def test_a_dataset_is_consumed_in_one_place():
    """`consume` has no site: its sample is a function of the actor and the enclosing loop
    indices, so two `consume`s of one dataset return the same samples, a reuse the job does not
    have. Both checkers and the builder refuse the second (V6, RC1 review)."""
    for shape, first, second in [
        ("body", "actors/gpu/body/0/loop/body/0/let/value", "actors/gpu/body/0/loop/body/2/let/value"),
        ("siblings", "actors/gpu/body/0/loop/body/0/let/value", "actors/gpu/body/1/loop/body/0/stat/file"),
        ("actors", "actors/eval/body/0/loop/body/0/stat/file", "actors/gpu/body/0/loop/body/0/let/value"),
    ]:
        with pytest.raises(BuildError) as e:
            _consume_twice(shape).build()
        msg = str(e.value)
        assert f"t: {second}: a second `consume` of `d` (the first is at {first})" in msg, msg
        assert "consume once and use the handle in each place" in msg and "(V6)" in msg, msg
        assert msg.count("a second `consume`") == 1, msg
    # one `consume`, its handle used twice: one sample per position, accepted
    _consume_twice("one").build()


def test_a_dataset_size_is_never_below_zero():
    """A dataset's size below 0 passed check.py and reached `aeiou datagen` cast to u64 (a
    16384 PiB projection for `small - 100` at `small` 10). Rule V25 refuses it where the
    document shows it, with the runner's message and path, in check.py and so at build (RC1
    review); an `array`'s sizes are clamped to [0, slot] by the runner and are left alone."""
    from aeiou.validate import _load
    _, check = _load()

    def errs(size, params=None, layout="whole"):
        d = {"layout": layout, "access": "indexed", "pattern": "d/f{id:04}", "count": 4, "size": size, "seed": 1}
        if layout == "array":
            d.update(pattern="d/all", slot=64)
        ast = {"ast": "0.10", "name": "t", "params": params or {}, "datasets": {"d": d},
               "actors": {"gpu": {"body": [{"stat": {"file": {"file": {"dataset": "d", "id": 0}}}}]}}}
        return check.Check(ast, "t").run()

    def below(n):
        return [f"t: datasets/d/size: a size below 0: it can be {n} (V25)"]

    unbounded = ["t: datasets/d/size: a size below 0: the distribution is unbounded below; give it a `min` of 0 or more (V25)"]
    small = {"small": {"default": 10}}
    assert errs({"const": {"sub": [{"param": "small"}, 100]}}, small) == below(-90)
    assert errs({"sub": [{"param": "small"}, 100]}, small) == below(-90)
    assert errs({"const": -1}) == below(-1)
    assert errs({"uniform": {"lo": -5, "hi": 10}}) == below(-5)
    assert errs({"normal": {"mean": 4096, "sd": 100}}) == unbounded
    assert errs({"normal": {"mean": 4096, "sd": 100, "min": -3}}) == below(-3)
    assert errs({"normal": {"mean": 4096, "sd": 100, "min": 1, "max": {"neg": 7}}}) == below(-7)
    assert errs({"lognormal": {"median": -4096, "sigma": 0.5}}) == unbounded
    assert errs({"empirical": {"values": [4096, -2], "weights": [1, 1]}}) == below(-2)
    assert errs({"mixture": [{"weight": 1, "dist": {"const": 4096}},
                             {"weight": 1, "dist": {"normal": {"mean": 1, "sd": 1}}}]}) == unbounded
    assert errs({"param": "sz"}, {"sz": {"default": {"normal": {"mean": 4096, "sd": 9}}}}) == unbounded
    for ok in [{"normal": {"mean": 4096, "sd": 100, "min": 0}}, {"normal": {"mean": 4096, "sd": 0}},
               {"lognormal": {"median": 4096, "sigma": 0.5}}, {"const": {"sub": [{"param": "small"}, 10]}},
               {"const": -0.4},
               {"mixture": [{"weight": 0, "dist": {"const": -1}}, {"weight": 1, "dist": {"const": 4096}}]},
               {"normal": {"mean": 10, "sd": 100, "min": {"div": [-100, 1]}}}]:
        assert not errs(ok, small), ok
    assert not errs({"normal": {"mean": 32, "sd": 100}}, layout="array")
    # the builder: refused at build, through the same checker
    w = Workload("t")
    P = w.params(small=10)
    ds = w.dataset("d", pattern="d/{id:06}", layout="whole", access="indexed", count=4, size=const(P.small - 100), seed=1)
    with w.actor("gpu") as a:
        a.stat(ds.file(0))
    with pytest.raises(BuildError, match=r"datasets/d/size: a size below 0: it can be -90 \(V25\)"):
        w.build()


def test_integers_outside_64_bits_are_refused_by_builder_checker_and_runner(tmp_path):
    """The contract's integers are i64, a seed u64 (schema/README.md §1). A Python int is
    unbounded; the runner's serde_json parses one past u64 as a float, so such an AST hashed
    and ran differently in the two languages. All three refuse it, with the same words."""
    import re
    from aeiou.validate import _load
    from aeiou import params as params_mod
    _, chk = _load()
    rng = "is outside the 64-bit signed range of the contract's integers (-2^63 to 2^63-1)"
    # the builder, where the author wrote it
    for big in (2 ** 63, 2 ** 64, -2 ** 63 - 1):
        w = Workload("t")
        with pytest.raises(BuildError, match=rf"parameter n: integer {big} {re.escape(rng)}"):
            w.params(n=big)
        with pytest.raises(BuildError, match=rf"const: integer {big} is outside"):
            const(big)
    w = Workload("t")
    P = w.params(n=2 ** 63 - 1, m=-2 ** 63, f=1e30)            # the ends fit; an integral float past them stays a float
    assert isinstance(w._param_specs["f"]["default"], float)
    ds = w.dataset("d", pattern="d/{id}", layout="whole", access="map", count=10, size=const(1), seed=2 ** 64 - 1)
    with w.actor("gpu") as a:
        a.compute(P.n)
    ast = w.build()
    with pytest.raises(BuildError, match=r"seed must be an integer in \[0, 2\^64\)"):
        w.dataset("e", pattern="e/{id}", layout="whole", access="map", count=10, size=const(1), seed=2 ** 64)
    # the checker, on a document the builder did not write; the runner says the same
    ast["params"]["n"]["default"] = 2 ** 64
    ast["datasets"]["d"]["seed"] = 2 ** 64
    ast["actors"]["gpu"]["body"][0]["compute"]["ns"] = {"add": [1, -2 ** 70]}
    want = [f"t: params/n/default: integer {2 ** 64} {rng}",
            f"t: datasets/d/seed: seed {2 ** 64} is outside 0 to 2^64-1",
            f"t: actors/gpu/body/0/compute/ns/add/1: integer {-2 ** 70} {rng}"]
    assert [e for e in chk.Check(ast, "t").run() if "outside" in e] == want
    with pytest.raises(BuildError, match=re.escape(want[0])):
        validate(ast, "t")
    f = tmp_path / "t.ast.json"
    f.write_text(json.dumps(ast, indent=2))
    out = subprocess.run([sys.executable, str(ROOT / "schema" / "check.py"), str(f)], capture_output=True, text=True)
    assert out.returncode == 1 and [l.strip() for l in out.stdout.splitlines()[1:]] == [w.replace("t:", "t.ast.json:", 1) for w in want]
    # a parameter file
    pset = {"params_version": 1, "abstract": "t", "params": {"n": [1, 2 ** 64]}}
    assert params_mod.check(ast, pset, name="x") == [f"x: params/n/1: integer {2 ** 64} {rng}"]
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
    if not runner.exists():
        pytest.skip("runner binary not built")
    out = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
    assert out.returncode != 0
    assert [l.strip() for l in (out.stdout + out.stderr).splitlines() if "is outside" in l] == want


def test_at_names_the_binding_its_body_shadows():
    """`x @ i` names a `let` of the body it is written in, before or after it, over an outer
    `x` (rule V3, "which binding"): check.py holds the inner one to the step-back rule, as the
    runner evaluates it, and refuses an `at` on a name its body binds twice (RC1 review)."""
    from aeiou.validate import _load
    _, check = _load()

    def doc(inner):
        return {
            "ast": "0.10", "name": "shadow",
            "namespaces": {"o": {"pattern": "o/{v}", "fields": {"v": "int"}, "size": 1, "seed": 3}},
            "actors": {"gpu": {"count": 1, "body": [{"loop": {"index": "i", "to": 3, "body": [
                {"let": {"name": "x", "value": {"mul": [{"index": "i"}, 1000]}}},
                {"loop": {"index": "j", "to": 4, "body": inner + [
                    {"stat": {"file": {"object": {"namespace": "o", "fields": {"v": {"ref": "x"}}}}}}]}}]}}]}},
        }

    def at(index):
        return {"cond": {"if": {"eq": [{"index": "j"}, 0]}, "then": 0,
                         "else": {"at": {"ref": "x", "index": index}}}}

    back = {"sub": [{"index": "j"}, 1]}
    assert not check.Check(doc([{"let": {"name": "y", "value": at(back)}},
                                {"let": {"name": "x", "value": at(back)}}]), "x").run()
    # the inner `x`, defined after the `at`, is the one named: `j` itself does not step back
    e = check.Check(doc([{"let": {"name": "y", "value": at({"index": "j"})}},
                         {"let": {"name": "x", "value": {"index": "j"}}}]), "x").run()
    assert any("names a binding of this loop body" in x for x in e), e
    e = check.Check(doc([{"let": {"name": "x", "value": {"index": "j"}}},
                         {"let": {"name": "x", "value": at(back)}}]), "x").run()
    assert any("`x @` names a binding this body defines more than once" in x for x in e), e


def _consume_under(shape):
    """A workload whose one `consume` of `d` sits under a loop whose count varies with an
    enclosing loop (`to`, `from` through a binding, a draw, a `parallel` width), or under loops
    of fixed count."""
    w = Workload("t")
    n = w.param("n", 3)
    d = w.dataset("d", pattern="d/f{id:04}", layout="whole", access="map", count=64, size=const(1), seed=1)
    with w.actor("gpu") as a:
        if shape == "actor_draw":
            m = a.let("m", draw(uniform(1, 4)))
        with a.loop("i", 3) as i:
            if shape == "to":
                with a.loop("j", 3 - i):
                    a.stat(d.consume())
            elif shape == "from":
                m = a.let("m", i)
                with a.loop("j", 3, start=m):
                    a.stat(d.consume())
            elif shape == "draw":
                m = a.let("m", draw(uniform(1, 4)))
                with a.loop("j", m):
                    a.stat(d.consume())
            elif shape == "parallel":
                with a.parallel("w", i + 1):
                    with a.loop("j", 2):
                        a.stat(d.consume())
            elif shape == "when":
                with a.loop("j", 3) as j:
                    with a.when(j < 3 - i):
                        a.stat(d.consume())
            elif shape == "actor_draw":
                with a.loop("j", m + n):
                    a.stat(d.consume())
            elif shape == "inside":
                x = a.let("x", d.consume())
                with a.loop("j", i + 1):
                    a.stat(x)
    return w


def test_a_consume_is_under_loops_of_fixed_count(tmp_path):
    """`consume`'s position is the row-major ordinal of its enclosing frames at fixed counts, so
    a loop around it whose count varies with an enclosing loop repeats positions: `loop i < 3 {
    loop j < 3 - i { stat consume d } }` statted `d/f0002` three times. Both checkers and the
    builder refuse it, naming the loop, with one message (V6, RC1 review)."""
    l0, l1 = "actors/gpu/body/0/loop/body/0", "actors/gpu/body/0/loop/body/1"

    def want(at, lp):
        return (f"t: {at}: a `consume` of `d` under a loop whose count varies with an enclosing loop ({lp}): "
                "its position is the row-major ordinal of the enclosing loops at fixed counts, so a count that "
                "varies repeats positions and draws the same samples, a reuse the job does not have; give the "
                "loop a fixed count (the most it can be, with a `cond`, the builder's `when`, inside to skip the "
                "rest) (V6)")

    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
    for shape, at, lp in [
        ("to", f"{l0}/loop/body/0/stat/file", f"{l0}/loop"),
        ("from", f"{l1}/loop/body/0/stat/file", f"{l1}/loop"),
        ("draw", f"{l1}/loop/body/0/stat/file", f"{l1}/loop"),
        ("parallel", f"{l0}/parallel/body/0/loop/body/0/stat/file", f"{l0}/parallel"),
    ]:
        with pytest.raises(BuildError) as e:
            _consume_under(shape).build()
        assert want(at, lp) in str(e.value), str(e.value)
        # the runner says the same of the same document
        if runner.exists():
            f = tmp_path / f"{shape}.ast.json"
            f.write_text(json.dumps(_consume_under(shape).build(validate=False)))
            out = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
            assert out.returncode != 0 and want(at, lp) in out.stdout + out.stderr, out.stdout + out.stderr
    # a fixed count, the rest skipped by a `when`; a count fixed for the actor; a varying loop
    # inside the `consume`'s, not around it: accepted
    for shape in ("when", "actor_draw", "inside"):
        _consume_under(shape).build()


def _size_errs(size):
    from aeiou.validate import _load
    _, check = _load()
    ast = {"ast": "0.10", "name": "t",
           "datasets": {"d": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4,
                              "size": size, "seed": 1}},
           "actors": {"gpu": {"body": [{"stat": {"file": {"file": {"dataset": "d", "id": 0}}}}]}}}
    return check.Check(ast, "t").run()


def test_a_dataset_size_is_judged_and_printed_as_the_runner_judges_and_prints_it():
    """Rule V25 rounds a size as the runner rounds a draw (`f64::round`, `rng.rust_round`'s
    algorithm), not `int(abs(x) + 0.5)`, which refused -0.49999999999999994 and printed
    -(2^52 + 1) as -(2^52 + 2); every digit printed, as the runner prints it. A size that is no
    finite number is refused with the runner's message, not a traceback (RC1 review)."""
    def below(n):
        return [f"t: datasets/d/size: a size below 0: it can be {n} (V25)"]

    infinite = ["t: datasets/d/size: a size that is not a finite number (V25)"]
    assert _size_errs(-0.49999999999999994) == []
    assert _size_errs(-0.5) == below(-1)
    assert _size_errs(-4503599627370497.0) == below(-4503599627370497)
    assert _size_errs(-1e20) == below(-100000000000000000000)
    assert _size_errs({"mul": [-1e308, 10]}) == infinite
    assert _size_errs({"mul": [1e308, 10]}) == infinite
    assert _size_errs({"add": [{"mul": [1e308, 10]}, {"mul": [-1e308, 10]}]}) == infinite


def test_min_above_max_is_judged_and_printed_as_the_runner_judges_and_prints_it():
    """`min` above `max` is compared as the runner's clamp compares, in f64 (2^53 + 1 is not
    above 2^53 there), and printed with Python's repr, which the runner's message now uses too
    (RC1 review). The same comparison where rule V3 believes a `min` only under a `max` not below it."""
    from aeiou.validate import _load
    _, check = _load()
    assert _size_errs({"normal": {"mean": 0, "sd": 1, "min": 1e20, "max": 1e19}}) == [
        "t: datasets/d/size: normal: min 1e+20 is above max 1e+19"]
    assert _size_errs({"lognormal": {"median": 1, "sigma": 1.0, "min": 1e-5, "max": 1e-7}}) == [
        "t: datasets/d/size: lognormal: min 1e-05 is above max 1e-07"]
    assert _size_errs({"normal": {"mean": 0, "sd": 1, "min": 9007199254740993, "max": 9007199254740992}}) == []
    chain = _v3_chain({"normal": {"mean": 5, "sd": 1, "min": 9007199254740993, "max": 9007199254740992}})
    assert check.Check(chain, "x").run() == []


def test_a_size_that_is_not_a_number_is_refused_whatever_the_order():
    """A size whose least value is not a number (`inf - inf`) is judged as the runner judges it
    whatever the order of a `mixture`'s arms or of a `min`'s or `max`'s operands, and in the
    clamp of a `normal`: Python's `min` answered by argument order where the runner's
    `f64::min` dropped the NaN (RC1 review). An arm of `+inf` is no finite size either, and
    outranks an arm unbounded below."""
    nan = {"add": [{"mul": [1e308, 10]}, {"mul": [-1e308, 10]}]}

    def arm(x):
        return {"weight": 1, "dist": {"const": x}}

    normal = {"weight": 1, "dist": {"normal": {"mean": 0, "sd": 1}}}
    infinite = ["t: datasets/d/size: a size that is not a finite number (V25)"]
    for a, b in [(arm(nan), arm(5)), (arm(nan), arm(-3)), (arm({"mul": [1e308, 10]}), arm(5)), (arm(nan), normal)]:
        assert _size_errs({"mixture": [a, b]}) == infinite
        assert _size_errs({"mixture": [b, a]}) == infinite
    for op, x in [("min", 5), ("max", -5)]:
        assert _size_errs({op: [nan, x]}) == infinite
        assert _size_errs({op: [x, nan]}) == infinite
    assert _size_errs({"normal": {"mean": nan, "sd": 0, "min": 5}}) == infinite
    assert _size_errs({"normal": {"mean": nan, "sd": 0, "max": -5}}) == infinite
    assert _size_errs({"normal": {"mean": 0, "sd": 1, "min": nan, "max": 5}}) == infinite
    # a weight of 0 is never drawn
    assert _size_errs({"mixture": [{"weight": 0, "dist": {"const": nan}}, arm(5)]}) == []


def test_a_channel_or_loader_inside_a_fork_is_refused():
    """V26 (RC1 regression review), as the runner has it: a `channel` or `loader` inside a
    `parallel` or `loader` body is refused, at any depth; one declared by the actor and used by
    its sub-actors, or declared again in each iteration of the actor's loop, is not."""
    from aeiou.validate import _load
    _, check = _load()

    def errs(*body):
        return check.Check({"ast": "0.10", "name": "v26", "actors": {"gpu": {"body": list(body)}}}, "v26").run()

    chan = {"channel": {"name": "c", "capacity": 2}}

    def loader(*body):
        return {"loader": {"name": "q", "index": "b", "workers": 2, "prefetch": 1, "batches": 2, "body": list(body)}}

    def par(*body):
        return {"parallel": {"index": "p", "width": 2, "body": list(body)}}

    put_take = [{"put": {"channel": "c", "seq": {"index": "p"}}}, {"take": {"channel": "c"}}]
    take_q = {"loop": {"index": "s", "to": 2, "body": [{"take": {"channel": "q"}}]}}
    compute = {"compute": {"ns": 1}}
    # accepted: the actor declares, its sub-actors put and take; a channel and a loader per iteration
    assert errs(chan, par(*put_take)) == []
    assert errs({"loop": {"index": "e", "to": 2, "body": [chan, loader(compute), par(*put_take), take_q]}}) == []
    for body, path in [
        ([par(chan, *put_take)], "parallel/body/0/channel"),
        ([loader(chan), take_q], "loader/body/0/channel"),
        ([par(loader(compute), take_q)], "parallel/body/0/loader"),
        ([loader(par(chan)), take_q], "loader/body/0/parallel/body/0/channel"),
    ]:
        kind = path.rsplit("/", 1)[1]
        e = errs(*body)
        assert len(e) == 1, e
        assert e[0].startswith(f"v26: actors/gpu/body/0/{path}: a `{kind}` inside a `parallel` or `loader` body")
        assert e[0].endswith("(V26)")


NON_JSON = ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"]


@pytest.mark.parametrize("lit", NON_JSON)
def test_non_json_numbers_are_refused_as_the_runner_refuses_them(tmp_path, lit):
    """Python's `json` takes `NaN`, `Infinity`, and `-Infinity`, and reads `1e400` as an
    infinity; the runner's serde_json refuses all of them at parse time. check.py crashed on
    such an AST and the builder's tools (emit.load, so aeiou-params, aeiou-catalog, aeiou-build
    --check) took it (RC1 review): every reader refuses it, as `aeiou check` does, wherever the
    literal sits (a scalar, inside an array, inside an object)."""
    from aeiou.usage import UsageError
    text = ast_path("file_per_sample").read_text(encoding="utf-8")
    assert '"default": 105000000' in text and '"default": 32' in text
    for i, sub in enumerate([lit, f"[1, {lit}]", f'{{"x": [{lit}]}}']):
        f = tmp_path / f"t{i}.ast.json"
        f.write_text(text.replace('"default": 105000000', f'"default": {sub}', 1), encoding="utf-8")
        out = subprocess.run([sys.executable, str(ROOT / "schema" / "check.py"), str(f)], capture_output=True, text=True)
        assert out.returncode == 1 and out.stdout.splitlines()[0] == f"FAIL {f.name}", out.stdout + out.stderr
        assert f"{f.name}: not an abstract (" in out.stdout and "Traceback" not in out.stderr, out.stdout + out.stderr
        with pytest.raises(UsageError, match="not an abstract"):
            emit.load(f)
        runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
        if runner.exists():
            out = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
            assert out.returncode == 1 and "not an abstract (" in out.stdout + out.stderr, out.stdout + out.stderr
    # a valid document reads the same, and hashes the same, as plain json reads it
    for p in sorted(WORKLOADS.rglob("*.ast.json")):
        t = p.read_text(encoding="utf-8")
        assert emit.loads(t) == json.loads(t)


def test_non_finite_floats_authored_are_refused_with_their_path(tmp_path):
    """A NaN or infinity a script slips past the builder's scalar checks (a framing column's
    weight, a format writer's option) has no JSON form: the build refused it only when the
    writer's `allow_nan=False` raised, as "ValueError at line 258", and a hermetic child printed
    a traceback (RC1 review). validate() names it, in process and in the hermetic child."""
    script = tmp_path / "nanw.py"
    script.write_text("from aeiou import *\n"
                      "w = Workload('nanw')\n"
                      "ds = w.dataset('d', pattern='d/{id}.tfr', layout='compound', access='stream', count=4,\n"
                      "               samples_per_file=2, size=uniform(100, 200), seed=1,\n"
                      "               format={'class': 'tfrecord', 'framing': {'unit': 1, 'columns': [{'weight': float('nan')}],\n"
                      "                       'writer': {'x': float('inf')}}})\n"
                      "with w.actor('gpu') as g:\n"
                      "    g.compute(1)\n")
    want = ["nanw: datasets/d/format/framing/columns/0/weight: nan is not a JSON number",
            "nanw: datasets/d/format/framing/writer/x: inf is not a JSON number"]
    for flags in ([], ["--hermetic"]):
        r = _run_cli([*flags, "-o", tmp_path, script])
        assert r.returncode == 1, r.stderr
        assert all(w in r.stderr for w in want) and "Traceback" not in r.stderr and "ValueError" not in r.stderr, r.stderr
        assert not (tmp_path / "nanw.ast.json").exists()


def test_a_channel_a_loaders_workers_use_is_declared_once_or_beside_it():
    """V26 (RC1 regression review), as the runner has it: a channel a loader's workers put or
    take on is declared at one site, outside every loop or in the body the loader is in, the
    loader declared at one site; the workers run beside the actor, so one declared again under
    them was the old channel or the new one as timing had it."""
    from aeiou.validate import _load
    _, check = _load()

    def errs(*body):
        return check.Check({"ast": "0.10", "name": "v26", "actors": {"gpu": {"body": list(body)}}}, "v26").run()

    chan = {"channel": {"name": "c", "capacity": 2}}

    def loader(name, *body):
        return {"loader": {"name": name, "index": "b", "workers": 2, "prefetch": 1, "batches": 2, "body": list(body)}}

    def lp(*body):
        return {"loop": {"index": "e", "to": 2, "body": list(body)}}

    def cond(*body):
        return {"cond": {"if": 1, "then": list(body)}}

    lput = loader("q", {"put": {"channel": "c", "seq": {"index": "b"}}})
    ltake = loader("q", {"take": {"channel": "c"}})
    compute = {"compute": {"ns": 1}}
    # accepted: declared once, outside every loop; declared beside the loader in its loop
    assert errs(chan, lp(lput)) == []
    assert errs(lp(chan, lput)) == []
    assert errs(cond(chan), lp(lput)) == []
    for body, path, kind, ch in [
        ([chan, lput, lp(chan)], "1/loader/body/0", "put", "c"),
        ([lp(chan), lput], "1/loader/body/0", "put", "c"),
        ([chan, lput, chan], "1/loader/body/0", "put", "c"),
        ([lp(chan), ltake], "1/loader/body/0", "take", "c"),
        ([lp(chan, cond(lput))], "0/loop/body/1/cond/then/0/loader/body/0", "put", "c"),
        ([loader("q", compute), lp(chan, lput)], "1/loop/body/1/loader/body/0", "put", "c"),
        ([lp(loader("r", compute)), loader("q", {"take": {"channel": "r"}})], "1/loader/body/0", "take", "r"),
    ]:
        e = errs(*body)
        assert len(e) == 1, e
        assert e[0].startswith(f"v26: actors/gpu/body/{path}/{kind}: a `{kind}` on channel `{ch}` inside a `loader` body")
        assert "would depend on timing" in e[0] and e[0].endswith("(V26)")


def test_a_value_of_a_kind_the_runner_refuses_is_refused_at_check():
    """V27 (RC1 review, R18), as the runner has it: a value whose kind the document shows (a
    literal, a parameter by its value in effect, a binding by its value) where the runner
    refuses that kind is refused at check time, not when the run reaches it: `"count": "gpus"`
    passed both checkers and failed `run` with "expected an integer, got str". A float or a
    bool where an integer is needed is read as the runner reads it, and a kind the document
    does not show (a draw, a `cond` whose arms differ) is left to the run."""
    from aeiou.validate import _load
    _, check = _load()

    def errs(count, *body):
        actor = {"body": list(body)}
        if count is not None:
            actor["count"] = count
        return check.Check({
            "ast": "0.10", "name": "v27",
            "params": {"s": {"default": "x"}, "n": {"default": 2}, "d": {"default": {"uniform": {"lo": 1, "hi": 4}}},
                       "a": {"default": [1, 2]}},
            "datasets": {"ds": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4, "size": 10, "seed": 1}},
            "namespaces": {"o": {"pattern": "o/{name}-{k:03}", "fields": {"name": "str", "k": "int"}, "size": 10, "seed": 2}},
            "actors": {"gpu": actor}}, "v27").run()

    def compute(ns):
        return {"compute": {"ns": ns}}

    def obj(name, k):
        return {"open": {"file": {"object": {"namespace": "o", "fields": {"name": name, "k": k}}}, "flags": ["WRONLY", "CREAT"]}}

    # accepted: what the runner reads as an integer, and what the document does not show
    for count, body in [
        (1.5, [compute(True)]),
        ({"param": "n"}, [compute({"draw": {"param": "d"}})]),
        ({"actor": "count"}, [compute({"cond": {"if": True, "then": 1, "else": "x"}})]),
        (1, [{"let": {"name": "x", "value": 3}}, compute({"add": [{"ref": "x"}, 1.5]})]),
        (1, [obj({"param": "s"}, 1.5)]),
        (None, [compute(1)]),
    ]:
        assert errs(count, *body) == [], (count, body)
    for count, body, want in [
        ("gpus", [compute(1)], "actors/gpu/count: expected an integer, got a string (V27)"),
        ({"param": "s"}, [compute(1)], "actors/gpu/count: expected an integer, got a string (V27)"),
        (1, [compute({"param": "d"})], "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        (1, [compute({"param": "a"})], "actors/gpu/body/0/compute/ns: expected an integer, got an array (V27)"),
        (1, [{"let": {"name": "x", "value": "x"}}, {"let": {"name": "y", "value": {"ref": "x"}}}, compute({"ref": "y"})],
         "actors/gpu/body/2/compute/ns: expected an integer, got a string (V27)"),
        (1, [{"let": {"name": "f", "value": {"file": {"dataset": "ds", "id": 0}}}}, compute({"ref": "f"})],
         "actors/gpu/body/1/compute/ns: expected an integer, got a handle (V27)"),
        (1, [compute({"add": [1, True]})], "actors/gpu/body/0/compute/ns/1: expected a number, got a bool (V27)"),
        (1, [compute({"neg": {"elem": {"array": "a", "index": "x"}}})], "actors/gpu/body/0/compute/ns/index: expected an integer, got a string (V27)"),
        (1, [compute({"cond": {"if": 0.5, "then": 1, "else": 2}})], "actors/gpu/body/0/compute/ns/if: expected a condition, got a float (V27)"),
        (1, [compute({"draw": {"normal": {"mean": "m", "sd": 1, "min": 0}}})], "actors/gpu/body/0/compute/ns/mean: expected a number, got a string (V27)"),
        (1, [{"loop": {"index": "i", "to": {"cond": {"if": True, "then": "a", "else": "b"}}, "body": []}}],
         "actors/gpu/body/0/loop/to: expected an integer, got a string (V27)"),
        (1, [{"stat": {"file": {"pick": {"dataset": "ds", "dist": "x"}}}}], "actors/gpu/body/0/stat/file/dist: expected an integer, got a string (V27)"),
        (1, [obj("a", True)], "actors/gpu/body/0/open/file/fields/k: expected an integer, got a bool (V27)"),
        (1, [obj(3, 1)], "actors/gpu/body/0/open/file/fields/name: expected a string, got an integer (V27)"),
    ]:
        assert errs(count, *body) == [f"v27: {want}"], (count, body)


def test_a_take_inside_a_loader_body_is_refused():
    """V26 (RC1 review, R20), as the runner has it: a `take` inside a `loader` body, at any
    depth, is refused; the workers run beside each other and the actor, so which of them took
    the item the actor put, and whether the run finished, was timing. A loader's body puts."""
    from aeiou.validate import _load
    _, check = _load()

    def errs(*body):
        return check.Check({"ast": "0.10", "name": "v26", "actors": {"gpu": {"body": list(body)}}}, "v26").run()

    chan = {"channel": {"name": "c", "capacity": 4}}

    def loader(name, *body):
        return {"loader": {"name": name, "index": "b", "workers": 2, "prefetch": 1, "batches": 8, "body": list(body)}}

    take = {"take": {"channel": "c"}}
    take_q = {"take": {"channel": "q"}}
    feed = {"loop": {"index": "j", "to": 8, "body": [{"put": {"channel": "c", "seq": {"index": "j"}}}, take_q]}}
    # accepted: the loader's body puts, the actor takes
    assert errs(chan, loader("q", {"put": {"channel": "c", "seq": {"index": "b"}}}), {"loop": {"index": "j", "to": 8, "body": [take_q, take]}}) == []
    for body, path, ch in [
        ([chan, loader("q", {"compute": {"ns": 1000}}, take), feed], "1/loader/body/1", "c"),
        ([chan, loader("q", {"loop": {"index": "i", "to": 1, "body": [{"cond": {"if": 1, "then": [take]}}]}}), feed], "1/loader/body/0/loop/body/0/cond/then/0", "c"),
        ([chan, loader("q", {"parallel": {"index": "p", "width": 1, "body": [take]}}), feed], "1/loader/body/0/parallel/body/0", "c"),
        ([loader("r", {"compute": {"ns": 1}}), loader("q", {"take": {"channel": "r"}}), {"loop": {"index": "j", "to": 8, "body": [take_q]}}], "1/loader/body/0", "r"),
    ]:
        assert errs(*body) == [f"v26: actors/gpu/body/{path}/take: a `take` on channel `{ch}` inside a `loader` body: the loader's workers run beside each other and the actor, so which of them took which item, and whether the run finished, would depend on timing; take in the actor, a loader's body only puts (V26)"]


def test_duplicate_keys_and_lone_surrogates_are_refused_as_the_runner_refuses_them(tmp_path):
    """Bug 76's `loads` still took JSON the runner's serde_json refuses (RC1 review, R21): a
    repeated field (`"name": "x", "name": "y"`, serde: "duplicate field `name`") and a `\\u`
    escape of half a UTF-16 pair (serde: "unexpected end of hex escape"); check.py said `ok`
    and `aeiou check` FAIL. Every reader refuses both now, a surrogate in a key too. A
    repeated key of a map (`params`) the runner reads as its last value; no writer repeats a
    key, so the Python readers refuse that as well."""
    from aeiou.usage import UsageError
    text = ast_path("file_per_sample").read_text(encoding="utf-8")
    name = '"name": "file_per_sample"'
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
    for i, (old, new, said, runner_refuses) in enumerate([
        (name, f'{name}, {name}', "duplicate field `name`", True),
        (name, '"name": "file_per\\ud800sample"', "lone surrogate in hex escape (\\ud800)", True),
        (name, '"name": "file_per_sample\\udc00"', "lone surrogate in hex escape (\\udc00)", True),
        ('"params": {', '"params": {"a\\ud800": {"default": 1}, ', "lone surrogate in hex escape (\\ud800)", True),
        ('"params": {', '"params": {"step_time": {"default": 1}, ', "duplicate field `step_time`", False),
    ]):
        f = tmp_path / f"t{i}.ast.json"
        assert old in text
        f.write_text(text.replace(old, new, 1), encoding="utf-8")
        out = subprocess.run([sys.executable, str(ROOT / "schema" / "check.py"), str(f)], capture_output=True, text=True)
        assert out.returncode == 1 and out.stdout.splitlines()[:2] == [f"FAIL {f.name}", f"  {f.name}: not an abstract ({said})"], out.stdout + out.stderr
        with pytest.raises(UsageError, match="not an abstract"):
            emit.load(f)
        if runner.exists() and runner_refuses:
            out = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
            assert out.returncode == 1 and f"FAIL {f.name}" in out.stdout, out.stdout + out.stderr
    # a whole pair is one character, not a lone surrogate
    assert emit.loads('{"a": "\\ud83d\\ude00"}') == {"a": "\U0001F600"}
    # a valid document (every committed abstract and parameter file) reads as plain json reads it
    for p in sorted(WORKLOADS.rglob("*.json")):
        t = p.read_text(encoding="utf-8")
        assert emit.loads(t) == json.loads(t), p


def test_a_parameter_defined_through_itself_is_refused_with_its_cycle(tmp_path):
    """A parameter whose default names itself (`{"const": {"param": "pa"}}`), read as a dataset's
    size, overflowed the stack of `aeiou check` (exit 134) and raised RecursionError in
    check.py (RC1 review, R22): the rules that follow a `param` to its value (V3, V25) never
    returned. Both checkers refuse every parameter on a cycle, with the cycle, the same path
    and message, and judge the document no further; a parameter naming another is accepted."""
    def doc(params):
        return json.dumps({"ast": "0.10", "name": "cyc", "params": params,
                           "datasets": {"d": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4,
                                              "size": {"param": "pa"}, "seed": 1}},
                           "actors": {"gpu": {"body": [{"stat": {"file": {"file": {"dataset": "d", "id": {"draw": {"param": "pa"}}}}}}]}}})
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
    for i, (params, want) in enumerate([
        ({"pa": {"default": {"const": {"param": "pa"}}}}, ["params/pa: parameter `pa` is defined through itself (pa -> pa) (V1)"]),
        ({"pa": {"default": {"mixture": [{"weight": 1, "dist": {"param": "pb"}}]}},
          "pb": {"default": {"normal": {"mean": {"add": [{"param": "pc"}, 1]}, "sd": 1, "min": 1}}},
          "pc": {"default": {"const": {"param": "pa"}}}, "pd": {"default": {"const": {"param": "pa"}}}},
         ["params/pa: parameter `pa` is defined through itself (pa -> pb -> pc -> pa) (V1)",
          "params/pb: parameter `pb` is defined through itself (pb -> pc -> pa -> pb) (V1)",
          "params/pc: parameter `pc` is defined through itself (pc -> pa -> pb -> pc) (V1)"]),
        ({"pa": {"default": {"mixture": [{"weight": 1, "dist": {"param": "pb"}}]}}, "pb": {"default": {"const": 3}}}, []),
    ]):
        f = tmp_path / f"c{i}.ast.json"
        f.write_text(doc(params), encoding="utf-8")
        out = subprocess.run([sys.executable, str(ROOT / "schema" / "check.py"), str(f)], capture_output=True, text=True)
        assert "Traceback" not in out.stderr, out.stderr
        got = sorted(ln.split(": ", 1)[1] for ln in out.stdout.splitlines() if "(V1)" in ln)
        assert got == want and out.returncode == (1 if want else 0), out.stdout + out.stderr
        if runner.exists():
            r = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
            got = sorted(ln.strip().split(": ", 1)[1] for ln in r.stdout.splitlines() if "(V1)" in ln)
            assert got == want and r.returncode == (1 if want else 0), r.stdout + r.stderr


def test_a_draw_of_a_kind_the_runner_refuses_is_refused_at_check(tmp_path):
    """V27 (RC1 review, R25), as the runner has it: a draw yields a kind the document shows (a
    `const` its argument's, an `empirical` of one kind of value that kind, a `mixture` or `cond`
    whose arms yield one kind that kind, a parameter whose value is a distribution what that
    distribution yields), judged where the drawn value is read: `pa` given `{"const": {"param":
    "pb"}}` with `pb` a distribution, drawn as a `compute`'s `ns`, passed both checkers and
    failed `run` with "expected an integer, got dist". What the document does not show is left
    to the run, and a value a parameter file gives is held to it too."""
    from aeiou.validate import _load
    _, check = _load()
    U = {"uniform": {"lo": 1, "hi": 4}}

    def doc(size, *body):
        return {"ast": "0.10", "name": "v27",
                "params": {"n": {"default": 3}, "d": {"default": U}, "cd": {"default": {"const": {"param": "d"}}},
                           "cn": {"default": {"const": {"param": "n"}}}, "cs": {"default": {"const": "x"}}},
                "datasets": {"ds": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4, "size": size, "seed": 1}},
                "namespaces": {"o": {"pattern": "o/{name}-{k:03}", "fields": {"name": "str", "k": "int"}, "size": 10, "seed": 2}},
                "actors": {"gpu": {"body": list(body)}}}

    def errs(size, *body):
        return check.Check(doc(size, *body), "v27").run()

    def compute(ns):
        return {"compute": {"ns": ns}}

    def obj(name, k):
        return {"open": {"file": {"object": {"namespace": "o", "fields": {"name": name, "k": k}}}, "flags": ["WRONLY", "CREAT"]}}

    redraw = {"let": {"name": "x", "value": {"draw": {"param": "cd"}}}}
    accepted = [
        ({"param": "cn"}, [compute({"draw": {"param": "cn"}})]),
        (10, [compute({"draw": {"cond": {"if": True, "then": {"param": "cn"}, "else": U}}})]),
        (10, [compute({"draw": {"mixture": [{"weight": 1, "dist": None}, {"weight": 1, "dist": 2}]}})]),
        (10, [redraw, compute({"draw": {"ref": "x"}})]),
        (10, [obj({"draw": {"param": "cs"}}, {"draw": {"empirical": {"values": [1.5, 2], "weights": [1, 1]}}})]),
    ]
    for size, body in accepted:
        assert errs(size, *body) == [], (size, body)
    refused = [
        (10, [compute({"draw": {"param": "cd"}})], "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        (10, [compute({"draw": {"const": {"param": "d"}}})], "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        (10, [compute({"draw": {"mixture": [{"weight": 1, "dist": {"param": "cd"}}]}})],
         "actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"),
        (10, [compute({"draw": {"param": "cs"}})], "actors/gpu/body/0/compute/ns: expected an integer, got a string (V27)"),
        (10, [compute({"add": [1, {"draw": {"empirical": {"values": ["a", "b"], "weights": [1, 1]}}}]})],
         "actors/gpu/body/0/compute/ns/1: expected a number, got a string (V27)"),
        (10, [{"cond": {"if": {"draw": {"const": 0.5}}, "then": []}}], "actors/gpu/body/0/cond/if: expected a condition, got a float (V27)"),
        (10, [redraw, compute({"ref": "x"})], "actors/gpu/body/1/compute/ns: expected an integer, got a distribution (V27)"),
        (10, [obj("a", {"draw": {"param": "cs"}})], "actors/gpu/body/0/open/file/fields/k: expected an integer, got a string (V27)"),
        (10, [{"stat": {"file": {"pick": {"dataset": "ds", "dist": {"param": "cd"}}}}}],
         "actors/gpu/body/0/stat/file/dist: expected an integer, got a distribution (V27)"),
        ({"const": {"param": "d"}}, [compute(1)], "datasets/ds/size: expected an integer, got a distribution (V27)"),
        ({"mixture": [{"weight": 1, "dist": None}]}, [compute(1)], "datasets/ds/size: expected an integer, got null (V27)"),
    ]
    for size, body, want in refused:
        assert errs(size, *body) == [f"v27: {want}"], (size, body)
    # a parameter drawn through itself is V1's, and following it does not recurse
    cyc = doc(10, compute(1))
    cyc["params"]["n"] = {"default": {"uniform": {"lo": {"draw": {"param": "n"}}, "hi": 9}}}
    assert check.Check(cyc, "v27").run() == ["v27: params/n: parameter `n` is defined through itself (n -> n) (V1)"]
    # the runner refuses the same documents with the same messages, and a parameter file giving
    # `cn` the const of a distribution is refused by aeiou-params check and the run alike
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    if runner.exists():
        for i, (size, body, *want) in enumerate(accepted + refused):
            f = tmp_path / f"c{i}.ast.json"
            f.write_text(json.dumps(doc(size, *body)), encoding="utf-8")
            r = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
            got = [ln.strip() for ln in r.stdout.splitlines() if "(V27)" in ln]
            assert got == [f"v27: {w}" for w in want] and r.returncode == (1 if want else 0), r.stdout + r.stderr
    a, f = tmp_path / "v27.ast.json", tmp_path / "v27.params.json"
    a.write_text(json.dumps(doc(10, compute({"draw": {"param": "cn"}}))), encoding="utf-8")
    f.write_text(json.dumps({"params_version": 1, "abstract": "v27", "params": {"cn": {"const": {"param": "d"}}}}), encoding="utf-8")
    want = "v27: actors/gpu/body/0/compute/ns: expected an integer, got a distribution (V27)"
    r = subprocess.run([sys.executable, "-m", "aeiou.params", "check", str(a), str(f)], capture_output=True, text=True,
                       cwd=str(BUILDER), env={**os.environ, "PYTHONPATH": str(BUILDER)})
    assert r.returncode == 1 and want in r.stdout, r.stdout + r.stderr
    if runner.exists():
        r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(a), "--params-file", str(f)], capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 1 and want in r.stderr, r.stderr


def test_a_comparison_or_array_read_the_runner_refuses_is_refused_at_check(tmp_path):
    """V27 (RC1 review, R26), as the runner has it: the runner compares two numbers, two
    strings, two bools, two handles, or `null` with anything, so `{"eq": [1, "a"]}` as a
    condition passed both checkers and failed `run` with "eq: cannot compare int with str"; an
    `elem`, `len`, or `sum` of a parameter that is not an array, or a `sum` of one holding other
    than numbers, likewise. What the document does not show is left to the run, and a value a
    parameter file gives is held to it too."""
    from aeiou.validate import _load
    _, check = _load()

    def doc(body):
        return {"ast": "0.10", "name": "v27",
                "params": {"n": {"default": 3}, "s": {"default": "x"}, "z": {"default": None}, "arr": {"default": [1, 2.5]},
                           "sarr": {"default": ["a", "b"]}, "d": {"default": {"uniform": {"lo": 1, "hi": 4}}}},
                "datasets": {"ds": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4, "size": 10, "seed": 1}},
                "actors": {"gpu": {"body": [{"let": {"name": "h", "value": {"file": {"dataset": "ds", "id": 0}}}}, body]}}}

    def when(c):
        return {"cond": {"if": c, "then": [{"compute": {"ns": 1}}]}}

    def compute(ns):
        return {"compute": {"ns": ns}}

    accepted = [
        when({"eq": [1, 2.5]}),
        when({"lt": [{"param": "s"}, "a"]}),
        when({"ne": [True, {"lt": [1, 2]}]}),
        when({"eq": [None, {"param": "d"}]}),
        when({"eq": [{"ref": "h"}, {"ref": "h"}]}),
        when({"eq": [{"add": [1, 2]}, "a"]}),
        when({"eq": [{"elem": {"array": "sarr", "index": 0}}, "a"]}),
        compute({"add": [{"len": "sarr"}, {"sum": "arr"}]}),
    ]
    for body in accepted:
        assert check.Check(doc(body), "v27").run() == [], body
    refused = [
        (when({"eq": [1, "a"]}), "actors/gpu/body/1/cond/if: eq: cannot compare an integer with a string (V27)"),
        (when({"ge": [{"param": "n"}, True]}), "actors/gpu/body/1/cond/if: ge: cannot compare an integer with a bool (V27)"),
        (when({"not": {"ne": [{"param": "d"}, {"param": "d"}]}}),
         "actors/gpu/body/1/cond/if: ne: cannot compare a distribution with a distribution (V27)"),
        (when({"lt": [{"ref": "h"}, {"elem": {"array": "sarr", "index": 1}}]}),
         "actors/gpu/body/1/cond/if: lt: cannot compare a handle with a string (V27)"),
        (when({"eq": [{"param": "arr"}, {"count": "ds"}]}), "actors/gpu/body/1/cond/if: eq: cannot compare an array with an integer (V27)"),
        (compute({"len": "n"}), "actors/gpu/body/1/compute/ns: len: parameter `n` is not an array (V27)"),
        (compute({"elem": {"array": "z", "index": 0}}), "actors/gpu/body/1/compute/ns: elem: parameter `z` is not an array (V27)"),
        (compute({"sum": "sarr"}), "actors/gpu/body/1/compute/ns: sum: an element of `sarr` is a string (V27)"),
    ]
    for body, want in refused:
        assert check.Check(doc(body), "v27").run() == [f"v27: {want}"], body
    # the runner refuses the same documents with the same messages, and a parameter file giving
    # `arr` a string element is refused by aeiou-params check and the run alike
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    if runner.exists():
        for i, (body, *want) in enumerate([(b,) for b in accepted] + refused):
            f = tmp_path / f"c{i}.ast.json"
            f.write_text(json.dumps(doc(body)), encoding="utf-8")
            r = subprocess.run([str(runner), "check", str(f)], capture_output=True, text=True)
            got = [ln.strip() for ln in r.stdout.splitlines() if "(V27)" in ln]
            assert got == [f"v27: {w}" for w in want] and r.returncode == (1 if want else 0), r.stdout + r.stderr
    a, f = tmp_path / "v27.ast.json", tmp_path / "v27.params.json"
    a.write_text(json.dumps(doc(compute({"sum": "arr"}))), encoding="utf-8")
    f.write_text(json.dumps({"params_version": 1, "abstract": "v27", "params": {"arr": [4, "a"]}}), encoding="utf-8")
    want = "v27: actors/gpu/body/1/compute/ns: sum: an element of `arr` is a string (V27)"
    r = subprocess.run([sys.executable, "-m", "aeiou.params", "check", str(a), str(f)], capture_output=True, text=True,
                       cwd=str(BUILDER), env={**os.environ, "PYTHONPATH": str(BUILDER)})
    assert r.returncode == 1 and want in r.stdout, r.stdout + r.stderr
    if runner.exists():
        r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(a), "--params-file", str(f)], capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 1 and want in r.stderr, r.stderr
