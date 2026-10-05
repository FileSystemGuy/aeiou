"""The option layers in Python (`aeiou/options.py`), the mirror of the runner's `options.rs`
(`runner/README.md` §14): precedence, the negation, the refusals, the block, and
`aeiou-datagen` through them."""
import os
import pathlib
import subprocess
import sys

import pytest

BUILDER = pathlib.Path(__file__).resolve().parents[1]
ROOT = BUILDER.parent
EXAMPLES = ROOT / "schema" / "examples"
sys.path.insert(0, str(BUILDER))

from aeiou import options  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402


def test_precedence_cli_env_config_default(tmp_path):
    cfg = tmp_path / "aeiou.toml"
    cfg.write_text("[datagen]\nthreads = 3\nroot = '/mnt/cfg'\n\n[run]\nrequire-cold = true\n")
    l = options.Layers("datagen", cfg, env={"AEIOU_THREADS": "5", "PATH": "/bin"})
    assert l.layered("threads", 7, int) == 7
    assert l.layered("root", None, pathlib.Path) == pathlib.Path("/mnt/cfg")
    l.finish()
    assert [(n, s) for n, _, _, s in l.entries] == [("threads", "cli"), ("root", f"config {cfg}"), ("config", "cli")]
    l = options.Layers("datagen", None, env={"AEIOU_THREADS": "5", "AEIOU_CONFIG": str(cfg)})
    assert l.layered("threads", None, int) == 5
    assert l.layered("root", None, pathlib.Path) == pathlib.Path("/mnt/cfg")
    l.finish()
    j = l.json()
    assert j["options"]["threads"] == {"value": 5, "source": "env AEIOU_THREADS"}
    assert j["options"]["config"]["source"] == "env AEIOU_CONFIG"
    assert j["config"]["path"] == str(cfg) and len(j["config"]["sha256"]) == 64
    assert j["env"] == ["AEIOU_CONFIG", "AEIOU_THREADS"]
    # the file turns a flag on, the command line turns it off
    l = options.Layers("run", cfg, env={})
    assert l.flag("require-cold", None) is True
    l = options.Layers("run", cfg, env={})
    assert l.flag("require-cold", False) is False
    assert l.entries[-1][3] == "cli"


def test_fixed_refusals_unknown_keys_and_warnings(tmp_path):
    with pytest.raises(BuildError, match="AEIOU_SEED is set, but --seed is the command line's alone"):
        options.Layers("run", None, env={"AEIOU_SEED": "3"}).fixed("seed", 3, False)
    cfg = tmp_path / "aeiou.toml"
    cfg.write_text("[datagen]\ngpus = 8\n")
    with pytest.raises(BuildError, match=r"\[datagen\] gpus is set, but --gpus is the command line's alone"):
        options.Layers("datagen", cfg, env={}).fixed("gpus", 8, True)
    # the same key under another subcommand's table is that subcommand's business
    l = options.Layers("run", cfg, env={})
    l.fixed("gpus", 8, True)
    l.finish()
    cfg.write_text("threads = 3\n")
    with pytest.raises(BuildError, match="top level"):
        options.Layers("datagen", cfg, env={})
    cfg.write_text("[datagen]\nthread = 3\n")
    l = options.Layers("datagen", cfg, env={})
    with pytest.raises(BuildError, match=r"\[datagen\] thread: unknown option"):
        l.finish()
    cfg.write_text("[datagen]\nbuffer-mib = 3\n")
    l = options.Layers("datagen", cfg, env={})
    with pytest.raises(BuildError, match="not an option of `aeiou datagen`"):
        l.finish()
    cfg.write_text("[datagen]\nthreads = 'three'\n")
    l = options.Layers("datagen", cfg, env={})
    with pytest.raises(BuildError, match=r"\[datagen\] threads: expected an integer"):
        l.layered("threads", None, int)
    l = options.Layers("datagen", None, env={"AEIOU_BOGUS": "1", "AEIOU_ROOT": "/x", "AEIOU_SCHEMA_DIR": "/s", "AEIOU_THREADS": "x"})
    with pytest.raises(BuildError, match="AEIOU_THREADS"):
        l.layered("threads", None, int)
    l = options.Layers("datagen", None, env={"AEIOU_BOGUS": "1", "AEIOU_ROOT": "/x", "AEIOU_SCHEMA_DIR": "/s"})
    l.layered("threads", None, int)
    l.finish()
    assert l.warnings == ["AEIOU_BOGUS is set and is no option of any subcommand; ignored"]
    # the negation is the command line's: the file and the environment say false
    cfg.write_text("[run]\nno-require-cold = true\n")
    with pytest.raises(BuildError, match=r"no-require-cold: a boolean is written as its name with true or false \(`require-cold = false`\)"):
        options.Layers("run", cfg, env={})
    l = options.Layers("run", None, env={"AEIOU_NO_REQUIRE_COLD": "1"})
    with pytest.raises(BuildError, match="AEIOU_REQUIRE_COLD=true or false"):
        l.finish()


def test_the_block_is_the_runners(tmp_path, capsys):
    cfg = tmp_path / "aeiou.toml"
    cfg.write_text("[datagen]\nthreads = 3\n")
    l = options.Layers("datagen", cfg, env={"AEIOU_ROOT": "/mnt/x", "AEIOU_BOGUS": "1"})
    l.fixed("abstract", pathlib.Path("a.ast.json"), True)
    l.fixed("param", [], False)
    l.fixed("dedupe", 1, False)
    assert l.layered("root", None, pathlib.Path) == pathlib.Path("/mnt/x")
    assert l.layered("threads", None, int) == 3
    l.finish()
    l.print()
    text = capsys.readouterr().out
    lines = text.splitlines()
    assert lines[0] == "options (cli > env > config > default)"
    rows = {}
    for line in lines[1:]:
        if not line:
            break
        if line.startswith("  WARNING"):
            continue
        lhs, src = line[2:].rsplit("  [", 1)
        name, value = lhs.rstrip().split(" = ", 1)
        rows[name] = (value, src.rstrip("]"))
    assert rows["abstract"] == ("a.ast.json", "cli")
    assert rows["param"] == ("none", "default")
    assert rows["dedupe"] == ("1", "default")
    assert rows["root"] == ("/mnt/x", "env AEIOU_ROOT")
    assert rows["threads"] == ("3", f"config {cfg}")
    assert rows["config"][1] == "cli" and rows["config"][0].startswith(f"{cfg} (sha256 ")
    assert "  WARNING: AEIOU_BOGUS is set and is no option of any subcommand; ignored" in lines
    assert text.endswith("\n\n")


def _datagen(args, env):
    full = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
    full.update({"PYTHONPATH": str(BUILDER), **env})
    return subprocess.run([sys.executable, "-m", "aeiou.datagen", *map(str, args)], capture_output=True, text=True, cwd=str(BUILDER), env=full)


def test_aeiou_datagen_through_the_layers(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("dgen_py")
    cfg = tmp_path / "aeiou.toml"
    cfg.write_text("[datagen]\nthreads = 2\n")
    root = tmp_path / "root"
    ast = EXAMPLES / "train_stream_tfrecord.ast.json"
    params = ["samples=64", "per_shard=16", "batch=8", "steps=2", "cycle=1", "sample_median=2000"]
    r = _datagen([ast, *sum([["--param", p] for p in params], [])], {"AEIOU_ROOT": str(root), "AEIOU_CONFIG": str(cfg), "AEIOU_BOGUS": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines()[1] == "options (cli > env > config > default)"
    assert f"root = {root}" in r.stdout and "[env AEIOU_ROOT]" in r.stdout
    assert "threads = 2" in r.stdout and f"[config {cfg}]" in r.stdout
    assert "[env AEIOU_CONFIG]" in r.stdout
    assert "WARNING: AEIOU_BOGUS" in r.stdout
    assert (root / "shards").exists() or any(root.iterdir())
    # a fixed option from the environment is refused before anything is written, as a usage
    # error in the suite's frame (test_usage.py has the frame itself)
    r = _datagen([ast, "--root", tmp_path / "other"], {"AEIOU_GPUS": "4"})
    assert r.returncode == 2
    assert r.stderr.startswith("aeiou-datagen: AEIOU_GPUS is set, but --gpus is the command line's alone")
    assert not (tmp_path / "other").exists()
    # --root from no layer: listed as missing
    r = _datagen([ast], {})
    assert r.returncode == 2 and "the following required arguments were not provided:\n  --root <DIR>  directory the abstract's paths are relative to\n" in r.stderr
