"""Parameter files: the committed sets fit their abstracts, `aeiou-params` writes and checks
sets, the value rules refuse what the runner refuses, and the safetensors table builder
reproduces the committed synthetic example."""
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
sys.path.insert(0, str(BUILDER))

from aeiou import emit, params  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402


def v2(values: dict) -> dict:
    """Values as the entries of a params_version 2 set, each a default (contract 0.11)."""
    return {k: {"default": v} for k, v in values.items()}


def _run(args, cwd=BUILDER):
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    return subprocess.run([sys.executable, "-m", "aeiou.params", *map(str, args)], capture_output=True, text=True, cwd=str(cwd), env=env)


def _ast(name):
    ast = emit.load(ast_path(name))
    return ast, emit.sha256(ast)


def _mechanism(path):
    """The directory of the mechanism a file belongs to: the nearest one holding an AST."""
    return next(d for d in path.parents if any(d.glob("*.ast.json")))


SETS = sorted(WORKLOADS.rglob("*params.json"))


@pytest.mark.parametrize("path", SETS, ids=lambda p: str(p.relative_to(WORKLOADS)))
def test_committed_sets_fit_their_abstracts(path):
    pset = params.load(path)
    ast, sha = _ast(pset["abstract"])
    assert params.check(ast, pset, ast_sha256=sha, name=path.name) == []
    assert (_mechanism(path) / f"{pset['abstract']}.ast.json").exists(), "a parameter file sits under its abstract's mechanism"


def test_defaults_round_trip(tmp_path):
    ast, sha = _ast("file_per_sample")
    out = tmp_path / "d.params.json"
    r = _run(["defaults", ast_path("file_per_sample"), "-o", out, "--pin"])
    assert r.returncode == 0, r.stderr
    pset = params.load(out)
    assert pset["ast_sha256"] == sha
    # a required parameter stays required, a storage-reported one (hdr_read, inv_read) is the storage's
    assert pset["params"] == {k: {"default": p["default"]} if "default" in p else {"required": True}
                              for k, p in ast["params"].items() if k not in ("hdr_read", "inv_read")}
    assert pset["params"]["steps"] == {"required": True} and pset["params"]["walk"] == {"default": True}
    assert params.check(ast, pset, ast_sha256=sha) == []
    r = _run(["check", ast_path("file_per_sample"), out, ast_path("file_per_sample").parent / "params" / "smoke.params.json"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("ok   ") == 2


def test_value_rules():
    ast, sha = _ast("file_per_sample")
    base = {"params_version": 2, "abstract": "file_per_sample", "params": {}}
    assert params.check(ast, {**base, "abstract": "other"}) == ["params: for abstract `other`, not `file_per_sample`"]
    assert params.check(ast, {**base, "ast_sha256": "0" * 64}, ast_sha256=sha)[0].startswith("params: for AST 0000")
    assert params.check(ast, {**base, "params": v2({"nope": 1})}) == ["params: no parameter `nope` in `file_per_sample`"]
    assert params.check(ast, {**base, "params": v2({"gpus": 4})})[0].endswith("--gpus, not by a parameter file")
    assert "the default is a scalar, the value is an array" in params.check(ast, {**base, "params": v2({"inv_entry": [1]})})[0]
    assert "the default is a scalar, the value is a distribution" in params.check(ast, {**base, "params": v2({"inv_entry": {"const": 1}})})[0]
    assert "the kind is a scalar, the value is an array" in params.check(ast, {**base, "params": v2({"steps": [1]})})[0]     # required since 2026-10-22
    assert params.check(ast, {**base, "params": v2({"steps": 7, "walk": True, "hdr_read": 4096})}) == []
    ast, _ = _ast("safetensors_load")
    assert params.check(ast, {**base, "abstract": "safetensors_load", "params": v2({"off": [1, 2, 3]})}) == []      # arrays may change length
    assert "elements are a scalar" in params.check(ast, {**base, "abstract": "safetensors_load", "params": v2({"off": [1, [2]]})})[0]


def test_an_integral_float_is_the_integer_its_default_is(tmp_path):
    """A value keeps its default's kind down to the number, as the runner has it (RC1 review):
    where the default is an integer, 3000.0 is 3000 (a divisor too: `files_per_directory`
    1300.0 was refused "not an integer" while `samples` 3000.0 went into the manifest as
    3000.0) and 20.7 is refused (the runner rounded it); an array's integer elements alike."""
    ast, _ = _ast("file_per_sample")
    base = {"params_version": 2, "abstract": "file_per_sample", "params": {}}
    assert params.check(ast, {**base, "params": v2({"samples": 3000.0, "files_per_directory": 1.3e3})}) == []
    assert params.check(ast, {**base, "params": v2({"samples": 20.7})}) == ["params: samples: the kind is an integer, the value 20.7 is not"]
    assert params.check(ast, {**base, "params": v2({"inv_entry": 20.7})}) == ["params: inv_entry: the default is an integer, the value 20.7 is not"]
    sast, _ = _ast("safetensors_load")
    sbase = {**base, "abstract": "safetensors_load"}
    assert params.check(sast, {**sbase, "params": v2({"off": [1, 2.0]})}) == []
    assert params.check(sast, {**sbase, "params": v2({"off": [1, 2.5]})}) == ["params: off[1]: the default is an integer, the value 2.5 is not"]
    # the runner takes and refuses the same files
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    if not runner.exists():
        pytest.skip("runner binary not built")
    for value, ok in ((3000.0, True), (20.7, False)):
        f = tmp_path / "f.params.json"
        job = {"batch": 32, "workers": 8, "prefetch": 2, "step_time": 105_000_000, "sample_median": 112_640}   # required since 2026-10-22
        f.write_text(json.dumps({**base, "params": v2({**job, "samples": value, "files_per_directory": 1300.0})}))
        r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(ast_path("file_per_sample")), "--params-file", str(f), "--param", "steps=1"],
                           capture_output=True, text=True, cwd=str(tmp_path))
        assert (r.returncode == 0) == ok, r.stderr
        assert ok or "samples: the kind is an integer, the value 20.7 is not" in r.stderr, r.stderr


def test_given_values_are_judged_as_the_runner_judges_them(tmp_path):
    """Past their kinds, the runner judges given values again (V3, `validate::check_given`; V23,
    `eval::bind_pattern`): `check` refuses what `aeiou dry-run --params-file` refuses."""
    ast, sha = _ast("file_per_sample")
    base = {"params_version": 2, "abstract": "file_per_sample", "params": {}}
    # a fractional value for an integer default is refused before the pattern sees it (RC1 review)
    errs = params.check(ast, {**base, "params": v2({"files_per_directory": 2.5})})
    assert errs == ["params: files_per_directory: the default is an integer, the value 2.5 is not"]
    assert params.check(ast, {**base, "params": v2({"files_per_directory": True})}) == [
        "params: dataset `train`: pattern `train/{id div files_per_directory:05}/img_{id:09}.jpg`: "
        "divisor `files_per_directory`: parameter `files_per_directory` is not an integer"]
    assert params.check(ast, {**base, "params": v2({"files_per_directory": 0})})[0].endswith(
        "`div files_per_directory` is 0: it needs a positive divisor")
    assert params.check(ast, {**base, "params": v2({"files_per_directory": 7})}) == []
    # V3: a reuse distance whose given arm reaches 0 makes `x @ r - reuse` name the binding being defined
    ast, sha = _ast("lmcache_local_disk")
    reuse = {"mixture": [{"weight": 0.15, "dist": None}, {"weight": 0.85, "dist": {"const": 0}}]}
    errs = params.check(ast, {**base, "abstract": "lmcache_local_disk", "params": v2({"reuse": reuse})})
    assert errs and all(e.startswith("params: invalid with the parameters given: lmcache_local_disk: actors/") for e in errs)
    assert any("`kind @` names a binding of this loop body, so its index must be" in e and "provably >= 1" in e for e in errs)
    assert params.check(ast, {**base, "abstract": "lmcache_local_disk", "params": v2({"reuse": ast["params"]["reuse"]["default"]})}) == []
    # and the command says so, exit 1
    f = tmp_path / "reuse.params.json"
    f.write_text(json.dumps({**base, "abstract": "lmcache_local_disk", "params": v2({"reuse": reuse})}))
    r = _run(["check", ast_path("lmcache_local_disk"), f])
    assert r.returncode == 1 and r.stdout.startswith("FAIL reuse.params.json") and "`kind @`" in r.stdout


def test_schema_rejects_malformed_files(tmp_path):
    f = tmp_path / "x.params.json"
    f.write_text(json.dumps({"params_version": 2, "abstract": "a", "params": v2({}), "extra": 1}))
    with pytest.raises(BuildError, match="extra"):
        params.load(f)
    f.write_text(json.dumps({"params_version": 3, "abstract": "a", "params": v2({})}))
    with pytest.raises(BuildError):
        params.load(f)
    f.write_text(json.dumps({"params_version": 2, "abstract": "a", "params": v2({"Bad-Name": 1})}))
    with pytest.raises(BuildError):
        params.load(f)
    f.write_text("{not json")
    with pytest.raises(BuildError):
        params.load(f)


# ---- safetensors ----

def make_synthetic_model(d: pathlib.Path) -> list:
    """Two safetensors shards of a two-layer Llama-shaped model (hidden 64, intermediate 128,
    vocab 256, F16), with real headers and zero data. The committed example set was made from
    these files."""
    h, i, v = 64, 128, 256
    layer = lambda n: [  # noqa: E731
        (f"model.layers.{n}.input_layernorm.weight", [h]),
        (f"model.layers.{n}.mlp.down_proj.weight", [h, i]),
        (f"model.layers.{n}.mlp.gate_proj.weight", [i, h]),
        (f"model.layers.{n}.mlp.up_proj.weight", [i, h]),
        (f"model.layers.{n}.post_attention_layernorm.weight", [h]),
        (f"model.layers.{n}.self_attn.k_proj.weight", [h // 4, h]),
        (f"model.layers.{n}.self_attn.o_proj.weight", [h, h]),
        (f"model.layers.{n}.self_attn.q_proj.weight", [h, h]),
        (f"model.layers.{n}.self_attn.v_proj.weight", [h // 4, h]),
    ]
    shards = [[("model.embed_tokens.weight", [v, h])] + layer(0),
              layer(1) + [("model.norm.weight", [h]), ("lm_head.weight", [v, h])]]
    paths = []
    for s, tensors in enumerate(shards):
        header, off = {"__metadata__": {"format": "pt"}}, 0
        for name, shape in sorted(tensors):
            n = 2
            for x in shape:
                n *= x
            header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [off, off + n]}
            off += n
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * (-len(hj) % 8)
        p = d / f"model-{s + 1:05}-of-00002.safetensors"
        p.write_bytes(len(hj).to_bytes(8, "little") + hj + bytes(off))
        paths.append(p)
    return paths


def test_safetensors_table_matches_committed(tmp_path):
    shards = make_synthetic_model(tmp_path)
    out = tmp_path / "m.params.json"
    r = _run(["safetensors", ast_path("safetensors_load"), *shards, "-o", out, "--tp", "8",
              "--doc", "x"])
    assert r.returncode == 0, r.stderr
    made = params.load(out)
    committed = params.load(ast_path("safetensors_load").parent / "params" / "synthetic.params.json")
    assert made["params"] == committed["params"], \
        "regenerate with: aeiou-params safetensors ../workloads/model-loading/safetensors-load/safetensors_load.ast.json <the fixture's shards> -o ../workloads/model-loading/safetensors-load/params/synthetic.params.json"
    t = params.values_of(made)
    assert t["shards"] == 2 and len(t["hdr_len"]) == 2 and t["tp"] == 8
    assert len(t["shard"]) == 21 and set(t["split"]) == {"column", "row", "full"}
    # a row-parallel tensor is one piece per row of the input dimension; a replicated one is read whole
    k = t["shard"].index(0)
    for s, nb, split, rows, rb in zip(t["shard"], t["bytes"], t["split"], t["rows"], t["row_bytes"]):
        assert rows * rb == nb
        assert (rows == 1) == (split != "row")
    assert all(off >= 8 + t["hdr_len"][s] for off, s in zip(t["off"], t["shard"]))
    assert t["shard_bytes"] == max(p.stat().st_size for p in shards)


# ---- npz ----

def test_npz_framing_is_read_from_real_archives_and_compared_with_the_abstract(tmp_path):
    """`aeiou-params npz`: the two archive numbers `npy_per_sample` takes as parameters,
    read from files. The traced writer's archives give the defaults; another set of members
    gives a larger number, which is reported and can be written as a parameter file; archives
    that disagree, a compressed member, and a member that is not first are refused."""
    np = pytest.importorskip("numpy")
    ast = ast_path("npy_per_sample")
    x = np.zeros((64, 64, 40), np.uint8)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, x=x, y=np.array([0]))
    np.savez(b, x=np.zeros((64, 64, 90), np.uint8), y=np.array([0]))
    r = _run(["npz", ast, a, b])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "framing 498, cd_len 102" in r.stdout and "defaults describe" in r.stdout

    # another writer: a third member and longer names
    c = tmp_path / "c.npz"
    np.savez(c, x=x, label=np.array([0]), spacing=np.array([1.0, 1.0, 2.5]))
    r = _run(["npz", ast, c])
    assert r.returncode == 1 and "DIFFERS framing: the abstract's default is 498" in r.stdout, r.stdout + r.stderr
    got = params.npz_framing(c)
    assert got["framing"] > 498 and got["cd_len"] > 102 and got["framing"] == c.stat().st_size - x.nbytes
    out = tmp_path / "c.params.json"
    r = _run(["npz", ast, c, "-o", out])
    assert r.returncode == 0, r.stdout + r.stderr
    assert params.values_of(params.load(out)) == {"framing": got["framing"], "cd_len": got["cd_len"]}
    assert _run(["check", ast, out]).returncode == 0

    r = _run(["npz", ast, a, c])
    assert r.returncode == 1 and "differs between the archives" in r.stderr
    z = tmp_path / "z.npz"
    np.savez_compressed(z, x=x, y=np.array([0]))
    r = _run(["npz", ast, z])
    assert r.returncode == 1 and "compressed" in r.stderr
    s = tmp_path / "s.npz"
    np.savez(s, y=np.array([0]), x=x)
    r = _run(["npz", ast, s])
    assert r.returncode == 1 and "not the first member" in r.stderr


def _one_line_failure(r, code=1):
    """A refusal in the suite's frame: the exit status, no traceback, the tool's name first."""
    assert r.returncode == code, (r.stdout, r.stderr)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.stderr.startswith("aeiou-params "), r.stderr


@pytest.mark.parametrize("cmd", ["defaults", "check", "safetensors", "npz"])
@pytest.mark.parametrize("text", ["[]", "{}"])
def test_json_that_is_no_abstract_is_rejected_in_one_line(tmp_path, cmd, text):
    """JSON that is no abstract: `AST rejected`, exit 1, as aeiou-catalog describe and the runner
    say it, never a traceback from the first key the command reads (RC1 review)."""
    ast = tmp_path / "x.ast.json"
    ast.write_text(text)
    pset = tmp_path / "p.params.json"
    pset.write_text('{"params_version": 2, "abstract": "x", "params": {}}')
    args = {"defaults": [ast, "-o", tmp_path / "o.json"], "check": [ast, pset],
            "safetensors": [ast, pset, "-o", tmp_path / "o.json"], "npz": [ast, pset]}[cmd]
    r = _run([cmd, *args])
    _one_line_failure(r)
    assert "AST rejected" in r.stderr
    assert not (tmp_path / "o.json").exists()


def test_bad_inputs_and_options_are_one_line_errors(tmp_path):
    """Every input aeiou-params reads or writes fails in one line, and a bad option value is a
    usage error in the frame (exit 2), never a traceback (RC1 review)."""
    st, npz = ast_path("safetensors_load"), ast_path("npy_per_sample")
    out = tmp_path / "o.json"
    huge = tmp_path / "huge.safetensors"                     # a header longer than the file: no allocation
    huge.write_bytes((1 << 60).to_bytes(8, "little") + b"{}")
    flat = tmp_path / "flat.safetensors"                     # a header that is JSON but no tensor table
    head = b'{"a": 1}'
    flat.write_bytes(len(head).to_bytes(8, "little") + head)
    nojson = tmp_path / "nojson.safetensors"
    nojson.write_bytes((4).to_bytes(8, "little") + b"{{{{")
    for shard, words in [(huge, "not a safetensors file"), (flat, "not a safetensors file"),
                         (nojson, "not a safetensors file"), (tmp_path / "nope.safetensors", "No such file"),
                         (tmp_path, "Is a directory")]:
        r = _run(["safetensors", st, shard, "-o", out])
        _one_line_failure(r)
        assert words in r.stderr, r.stderr
    r = _run(["npz", npz, tmp_path / "nope.npz"])
    _one_line_failure(r)
    assert "No such file" in r.stderr
    r = _run(["defaults", npz, "-o", tmp_path / "no" / "dir.json"])
    _one_line_failure(r)
    assert "No such file" in r.stderr
    for opt in (["--tp", "0"], ["--tp", "-2"], ["--tp", "x"], ["--column", "("], ["--row", "[a"]):
        r = _run(["safetensors", st, huge, "-o", out, *opt])
        _one_line_failure(r, 2)
        assert f"argument {opt[0]}:" in r.stderr and "Usage: aeiou-params safetensors" in r.stderr, r.stderr
    assert not out.exists()


def _given_abstracts():
    """Two abstracts whose rules read a scalar parameter: a `normal` whose `max` is `hi`, drawn as
    the reuse distance of `x @ i - d` (V3 and min-above-max), and a dataset whose size is `sz` (V25)."""
    from aeiou import Workload, const, normal, when
    w = Workload("t")
    P = w.params(hi=10)
    ds = w.dataset("d", pattern="d/f{id:04}", layout="whole", access="indexed", count=4, size=const(4096), seed=1)
    with w.actor("gpu") as a:
        with a.loop("i", 4) as i:
            d = a.draw("d", normal(5, 1, min=1, max=P.hi))
            a.let("x", when(i > 0, a.ref("x").at(i - d), 0))
            a.stat(ds.file(a.ref("x")))
    v3 = w.build()
    w = Workload("t")
    P = w.params(sz=4096)
    ds = w.dataset("d", pattern="d/f{id:04}", layout="whole", access="indexed", count=4, size=P.sz, seed=1)
    with w.actor("gpu") as a:
        a.stat(ds.file(0))
    return v3, w.build()


GIVEN = [  # (abstract, given, the runner's `check_given` errors)
    ("v3", {"hi": 0}, ["t: actors/gpu/body/0/loop/body/0/let/value: normal: min 1 is above max 0",
                       "t: actors/gpu/body/0/loop/body/1/let/value/then: `x @` names a binding of this loop body, "
                       "so its index must be {sub: [{index: i}, e]} with e provably >= 1"]),
    ("v3", {"hi": 9}, []),
    ("v3", {"hi": 1}, []),
    ("v25", {"sz": -5}, ["t: datasets/d/size: a size below 0: it can be -5 (V25)"]),
    ("v25", {"sz": 8192}, []),
    ("v25", {"sz": 0}, []),
]


@pytest.mark.parametrize("which,given,want", GIVEN, ids=[f"{w}-{g}" for w, g, _ in GIVEN])
def test_given_scalars_are_judged_by_every_rule_that_reads_them(tmp_path, which, given, want):
    """A given scalar is the value in effect for every rule the runner's `check_given` applies,
    not only V3's distribution minimums: a `max` below its `min` (and with it the V3 proof the
    `min` gave) and a dataset size below 0 (V25) were judged on the default, so `check` passed a
    file `run --params-file` refuses (RC1 review). Same verdict and words as the runner's."""
    v3, v25 = _given_abstracts()
    ast = {"v3": v3, "v25": v25}[which]
    pset = {"params_version": 2, "abstract": "t", "params": v2(given)}
    errs = params.check(ast, pset)
    assert sorted(errs) == sorted(f"params: invalid with the parameters given: {e}" for e in want), errs
    a, f = tmp_path / "t.ast.json", tmp_path / "t.params.json"
    a.write_text(json.dumps(ast))
    f.write_text(json.dumps(pset))
    r = _run(["check", a, f])
    assert r.returncode == (1 if want else 0), r.stdout + r.stderr
    # resolved here: CI gives a path relative to builder/, and the run below is in tmp_path
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    if not runner.exists():
        pytest.skip("runner binary not built")
    r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(a), "--params-file", str(f)],
                       capture_output=True, text=True, cwd=str(tmp_path))
    if want:
        assert r.returncode == 1 and "invalid with the parameters given:" in r.stderr, r.stderr
        got = [ln.strip() for ln in r.stderr.split("invalid with the parameters given:", 1)[1].splitlines() if ln.strip()]
        assert sorted(got) == sorted(want), r.stderr
    else:
        assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("lit", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_non_json_numbers_in_a_parameter_file_are_refused(tmp_path, lit):
    """A parameter given as `Infinity` (or `NaN`, `-Infinity`, `1e400`), alone or nested, is not
    JSON to the runner's serde_json, which refuses the file; Python's `json` took it, so
    `aeiou-params check` judged a file `run --params-file` cannot read (RC1 review)."""
    ast = ast_path("file_per_sample")
    for i, value in enumerate([lit, f"[1, {lit}]", f'{{"x": {lit}}}']):
        f = tmp_path / f"t{i}.params.json"
        f.write_text(f'{{"params_version": 2, "abstract": "file_per_sample", "params": {{"step_time": {{"default": {value}}}}}}}\n')
        with pytest.raises(BuildError, match=f"{f.name}: .*(is not JSON|is out of range)"):
            params.load(f)
        r = _run(["check", ast, f])
        assert r.returncode == 1 and f"FAIL {f.name}" in r.stdout, r.stdout + r.stderr
        assert "is not JSON" in r.stdout or "is out of range" in r.stdout, r.stdout + r.stderr
        runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
        if runner.exists():
            r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(ast), "--params-file", str(f)],
                               capture_output=True, text=True, cwd=str(tmp_path))
            assert r.returncode == 1 and "not JSON" in r.stderr, r.stderr


def test_a_file_pinned_to_the_builders_hash_is_taken_by_the_runner(tmp_path):
    """A float literal is read to the nearest f64 by both sides: the runner's serde_json read
    123456789.12345679 one ULP off, so a parameter file pinned to the hash `aeiou-params` computes
    passed `check` and was refused by `dry-run --params-file` as a file for another AST (RC1 review)."""
    a, f = tmp_path / "h.ast.json", tmp_path / "h.params.json"
    a.write_text('{"ast": "0.11", "name": "h", "params": {"sz": {"default": 123456789.12345679}}, '
                 '"datasets": {"d": {"layout": "whole", "access": "indexed", "pattern": "d/f{id:04}", "count": 4, "size": 4096, "seed": 1}}, '
                 '"actors": {"gpu": {"body": [{"compute": {"ns": {"param": "sz"}}}]}}}')
    f.write_text(json.dumps({"params_version": 2, "abstract": "h", "ast_sha256": emit.sha256(emit.load(a)), "params": v2({})}))
    r = _run(["check", a, f])
    assert r.returncode == 0, r.stdout + r.stderr
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    if not runner.exists():
        pytest.skip("runner binary not built")
    r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(a), "--params-file", str(f)],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr


def test_duplicate_keys_and_lone_surrogates_in_a_parameter_file_are_refused(tmp_path):
    """A parameter file with a `\\u` escape of half a UTF-16 pair (`"doc": "a\\ud800"`) is not
    JSON to the runner's serde_json, which refuses it; Python's `json` took it, so
    `aeiou-params check` passed a file `run --params-file` cannot read (RC1 review, R21). A
    repeated key, which the runner reads as its last value, is refused too: no writer repeats
    one, and which of two values a file meant is not for a reader to guess."""
    import re
    ast = ast_path("file_per_sample")
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    for i, (body, said) in enumerate([
        ('"doc": "a\\ud800", "params": {}', "lone surrogate in hex escape (\\ud800)"),
        ('"params": {"step_time\\udfff": {"default": 1}}', "lone surrogate in hex escape (\\udfff)"),
        ('"params": {"step_time": {"default": 1}, "step_time": {"default": 2}}', "duplicate field `step_time`"),
        ('"params": {}, "params": {"step_time": {"default": 1}}', "duplicate field `params`"),
    ]):
        f = tmp_path / f"t{i}.params.json"
        f.write_text(f'{{"params_version": 2, "abstract": "file_per_sample", {body}}}\n')
        with pytest.raises(BuildError, match=re.escape(f"{f.name}: {said}")):
            params.load(f)
        r = _run(["check", ast, f])
        assert r.returncode == 1 and f"FAIL {f.name}" in r.stdout and said in r.stdout, r.stdout + r.stderr
        if runner.exists() and "surrogate" in said:
            r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(ast), "--params-file", str(f)],
                               capture_output=True, text=True, cwd=str(tmp_path))
            assert r.returncode == 1 and "not JSON" in r.stderr, r.stderr


def test_a_parameter_file_closing_a_cycle_is_refused_as_the_runner_refuses_it(tmp_path):
    """A value given to a parameter that names, through `param`s, the parameter itself (RC1
    review, R22): the runner's `check_given` and `aeiou-params check` refuse it with the cycle,
    rather than recursing without end in the rules that follow a `param` to its value."""
    a, f = tmp_path / "g.ast.json", tmp_path / "g.params.json"
    a.write_text(json.dumps({"ast": "0.11", "name": "g", "params": {"pa": {"default": {"mixture": [{"weight": 1, "dist": {"param": "pb"}}]}}, "pb": {"default": {"const": 3}}},
                             "actors": {"gpu": {"body": [{"compute": {"ns": {"draw": {"param": "pa"}}}}]}}}))
    f.write_text(json.dumps({"params_version": 2, "abstract": "g", "params": v2({"pb": {"const": {"param": "pa"}}})}))
    want = ["g: params/pa: parameter `pa` is defined through itself (pa -> pb -> pa) (V1)",
            "g: params/pb: parameter `pb` is defined through itself (pb -> pa -> pb) (V1)"]
    r = _run(["check", a, f])
    assert r.returncode == 1 and "Traceback" not in r.stderr, r.stdout + r.stderr
    assert [ln.split("invalid with the parameters given: ", 1)[1] for ln in r.stdout.splitlines() if "(V1)" in ln] == want, r.stdout
    runner = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
    if runner.exists():
        r = subprocess.run([str(runner), "dry-run", "--gpus", "1", str(a), "--params-file", str(f)], capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 1 and [ln.strip() for ln in r.stderr.splitlines() if "(V1)" in ln] == want, r.stderr
