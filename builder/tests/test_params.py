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
EXAMPLES = ROOT / "schema" / "examples"
PARAMS = EXAMPLES / "params"
sys.path.insert(0, str(BUILDER))

from aeiou import emit, params  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402


def _run(args, cwd=BUILDER):
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    return subprocess.run([sys.executable, "-m", "aeiou.params", *map(str, args)], capture_output=True, text=True, cwd=str(cwd), env=env)


def _ast(name):
    ast = emit.load(EXAMPLES / f"{name}.ast.json")
    return ast, emit.sha256(ast)


@pytest.mark.parametrize("path", sorted(PARAMS.glob("*.params.json")), ids=lambda p: p.name)
def test_committed_sets_fit_their_abstracts(path):
    pset = params.load(path)
    ast, sha = _ast(pset["abstract"])
    assert params.check(ast, pset, ast_sha256=sha, name=path.name) == []
    assert path.name.startswith(pset["abstract"] + "."), "named <abstract>.<set>.params.json"


def test_defaults_round_trip(tmp_path):
    ast, sha = _ast("train_small_files")
    out = tmp_path / "d.params.json"
    r = _run(["defaults", EXAMPLES / "train_small_files.ast.json", "-o", out, "--pin"])
    assert r.returncode == 0, r.stderr
    pset = params.load(out)
    assert pset["ast_sha256"] == sha
    assert pset["params"] == {k: p["default"] for k, p in ast["params"].items()}
    assert params.check(ast, pset, ast_sha256=sha) == []
    r = _run(["check", EXAMPLES / "train_small_files.ast.json", out, PARAMS / "train_small_files.smoke.params.json"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("ok   ") == 2


def test_value_rules():
    ast, sha = _ast("train_small_files")
    base = {"params_version": 1, "abstract": "train_small_files", "params": {}}
    assert params.check(ast, {**base, "abstract": "other"}) == ["params: for abstract `other`, not `train_small_files`"]
    assert params.check(ast, {**base, "ast_sha256": "0" * 64}, ast_sha256=sha)[0].startswith("params: for AST 0000")
    assert params.check(ast, {**base, "params": {"nope": 1}}) == ["params: no parameter `nope` in `train_small_files`"]
    assert params.check(ast, {**base, "params": {"gpus": 4}})[0].endswith("--gpus, not by a parameter file")
    assert "the default is a scalar, the value is an array" in params.check(ast, {**base, "params": {"steps": [1]}})[0]
    assert "the default is a scalar, the value is a distribution" in params.check(ast, {**base, "params": {"steps": {"const": 1}}})[0]
    assert params.check(ast, {**base, "params": {"steps": 7, "enumerate": True, "hdr_read": 4096}}) == []
    ast, _ = _ast("model_load")
    assert params.check(ast, {**base, "abstract": "model_load", "params": {"off": [1, 2, 3]}}) == []      # arrays may change length
    assert "elements are a scalar" in params.check(ast, {**base, "abstract": "model_load", "params": {"off": [1, [2]]}})[0]


def test_schema_rejects_malformed_files(tmp_path):
    f = tmp_path / "x.params.json"
    f.write_text(json.dumps({"params_version": 1, "abstract": "a", "params": {}, "extra": 1}))
    with pytest.raises(BuildError, match="extra"):
        params.load(f)
    f.write_text(json.dumps({"params_version": 2, "abstract": "a", "params": {}}))
    with pytest.raises(BuildError):
        params.load(f)
    f.write_text(json.dumps({"params_version": 1, "abstract": "a", "params": {"Bad-Name": 1}}))
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
    r = _run(["safetensors", EXAMPLES / "model_load.ast.json", *shards, "-o", out, "--tp", "8",
              "--doc", "x"])
    assert r.returncode == 0, r.stderr
    made = params.load(out)
    committed = params.load(PARAMS / "model_load.synthetic.params.json")
    assert made["params"] == committed["params"], \
        "regenerate with: aeiou-params safetensors ../schema/examples/model_load.ast.json <the fixture's shards> -o ../schema/examples/params/model_load.synthetic.params.json"
    t = made["params"]
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
    """`aeiou-params npz`: the two archive numbers `train_large_samples` takes as parameters,
    read from files. The traced writer's archives give the defaults; another set of members
    gives a larger number, which is reported and can be written as a parameter file; archives
    that disagree, a compressed member, and a member that is not first are refused."""
    np = pytest.importorskip("numpy")
    ast = EXAMPLES / "train_large_samples.ast.json"
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
    assert params.load(out)["params"] == {"framing": got["framing"], "cd_len": got["cd_len"]}
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
