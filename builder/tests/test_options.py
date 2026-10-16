"""The option layers in Python (`aeiou/options.py`), the mirror of the runner's `options.rs`
(`runner/REFERENCE.md` §14): precedence, the negation, the refusals, the block, and
`aeiou-datagen` through them."""
import os
import pathlib
import re
import subprocess
import sys

import pytest

BUILDER = pathlib.Path(__file__).resolve().parents[1]
ROOT = BUILDER.parent
WORKLOADS = ROOT / "workloads"


def ast_path(name):
    """The committed AST `name`, wherever its mechanism lives under workloads/ (DESIGN_REVIEW.md §3.66)."""
    return next(WORKLOADS.rglob(f"{name}.ast.json"))
sys.path.insert(0, str(BUILDER))

from aeiou import options  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402


def test_precedence_cli_env_config_default(tmp_path):
    cfg = tmp_path / "aeiou.toml"
    cfg.write_text("[datagen]\nwriters = 3\nroot = '/mnt/cfg'\n\n[run]\nrequire-cold = true\n")
    l = options.Layers("datagen", cfg, env={"AEIOU_WRITERS": "5", "PATH": "/bin"})
    assert l.layered("writers", 7, int) == 7
    assert l.layered("root", None, pathlib.Path) == pathlib.Path("/mnt/cfg")
    l.finish()
    assert [(n, s) for n, _, _, s in l.entries] == [("writers", "cli"), ("root", f"config {cfg}"), ("config", "cli")]
    l = options.Layers("datagen", None, env={"AEIOU_WRITERS": "5", "AEIOU_CONFIG": str(cfg)})
    assert l.layered("writers", None, int) == 5
    assert l.layered("root", None, pathlib.Path) == pathlib.Path("/mnt/cfg")
    l.finish()
    j = l.json()
    assert j["options"]["writers"] == {"value": 5, "source": "env AEIOU_WRITERS"}
    assert j["options"]["config"]["source"] == "env AEIOU_CONFIG"
    assert j["config"]["path"] == str(cfg) and len(j["config"]["sha256"]) == 64
    assert j["env"] == ["AEIOU_CONFIG", "AEIOU_WRITERS"]
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
    cfg.write_text("writers = 3\n")
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
    cfg.write_text("[datagen]\nwriters = 'three'\n")
    l = options.Layers("datagen", cfg, env={})
    with pytest.raises(BuildError, match=r"\[datagen\] writers: expected an integer"):
        l.layered("writers", None, int)
    l = options.Layers("datagen", None, env={"AEIOU_BOGUS": "1", "AEIOU_ROOT": "/x", "AEIOU_SCHEMA_DIR": "/s", "AEIOU_WRITERS": "x"})
    with pytest.raises(BuildError, match="AEIOU_WRITERS"):
        l.layered("writers", None, int)
    l = options.Layers("datagen", None, env={"AEIOU_BOGUS": "1", "AEIOU_ROOT": "/x", "AEIOU_SCHEMA_DIR": "/s"})
    l.layered("writers", None, int)
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
    cfg.write_text("[datagen]\nwriters = 3\n")
    l = options.Layers("datagen", cfg, env={"AEIOU_ROOT": "/mnt/x", "AEIOU_BOGUS": "1"})
    l.fixed("abstract", pathlib.Path("a.ast.json"), True)
    l.fixed("param", [], False)
    l.fixed("dedupe", 1, False)
    assert l.layered("root", None, pathlib.Path) == pathlib.Path("/mnt/x")
    assert l.layered("writers", None, int) == 3
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
    assert rows["writers"] == ("3", f"config {cfg}")
    assert rows["config"][1] == "cli" and rows["config"][0].startswith(f"{cfg} (sha256 ")
    assert "  WARNING: AEIOU_BOGUS is set and is no option of any subcommand; ignored" in lines
    assert text.endswith("\n\n")


def _datagen(args, env):
    full = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
    full.update({"PYTHONPATH": str(BUILDER), **env})
    return subprocess.run([sys.executable, "-m", "aeiou.datagen", *map(str, args)], capture_output=True, text=True, cwd=str(BUILDER), env=full)


def test_aeiou_datagen_through_the_layers(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("numpy")
    cfg = tmp_path / "aeiou.toml"
    cfg.write_text("[datagen]\nwriters = 2\n")
    root = tmp_path / "root"
    ast = ast_path("tfrecord_stream")
    params = ["samples=64", "samples_per_file=16", "batch=8", "steps=2", "cycle=1", "sample_median=2000"]
    r = _datagen([ast, *sum([["--param", p] for p in params], [])], {"AEIOU_ROOT": str(root), "AEIOU_CONFIG": str(cfg), "AEIOU_BOGUS": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines()[1] == "options (cli > env > config > default)"
    assert f"root = {root}" in r.stdout and "[env AEIOU_ROOT]" in r.stdout
    assert "writers = 2" in r.stdout and f"[config {cfg}]" in r.stdout
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


RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))

# (subcommand, TOML line, option, kind, the refusal options.rs words); every kind, every refusal
CONFIG_REFUSALS = [
    ("datagen", 'root = ""', "root", pathlib.Path, "empty, and a value is required (leave it unset for none)"),
    ("datagen", "root = 3", "root", pathlib.Path, "expected a path string, got the integer 3"),
    ("datagen", 'endpoint = ["a", ""]', "endpoint", list, "empty, and a value is required (leave it unset for none)"),
    ("datagen", 'endpoint = ""', "endpoint", list, "empty, and a value is required (leave it unset for none)"),
    ("datagen", "endpoint = [1]", "endpoint", list, "expected strings, got the integer 1"),
    ("datagen", "endpoint = 1.5", "endpoint", list, "expected a string or an array of strings, got the float 1.5"),
    ("datagen", "root = 2.0", "root", pathlib.Path, "expected a path string, got the float 2"),
    ("datagen", "root = 1e20", "root", pathlib.Path, "expected a path string, got the float 100000000000000000000"),
    ("datagen", "root = -inf", "root", pathlib.Path, "expected a path string, got the float -inf"),
    ("datagen", "root = nan", "root", pathlib.Path, "expected a path string, got the float NaN"),
    ("datagen", "root = 0.000125", "root", pathlib.Path, "expected a path string, got the float 0.000125"),
    ("datagen", "writers = true", "writers", int, "expected an integer, got the boolean true"),
    ("datagen", 'coordinator = ""', "coordinator", str, "empty, and a value is required (leave it unset for none)"),
    ("datagen", "coordinator = []", "coordinator", str, "expected a string, got an array of 0 value(s)"),
    ("run", 'require-cold = "yes"', "require-cold", bool, 'expected true or false, got the string "yes"'),
    ("run", "require-cold = 1979-05-27", "require-cold", bool, "expected true or false, got the datetime 1979-05-27"),
    ("run", 'time-scale = "x"', "time-scale", float, 'expected a number, got the string "x"'),
    ("run", "root = {a = 1}", "root", pathlib.Path, "expected a path string, got a table"),
]

ENV_REFUSALS = [
    ("run", "AEIOU_REQUIRE_COLD", "maybe", "require-cold", bool, '"maybe": expected true or false'),
    ("datagen", "AEIOU_ROOT", "", "root", pathlib.Path, "empty, and a value is required (leave it unset for none)"),
    ("datagen", "AEIOU_ENDPOINT", " ", "endpoint", list, "empty, and a value is required (leave it unset for none)"),
    ("datagen", "AEIOU_COORDINATOR", "", "coordinator", str, "empty, and a value is required (leave it unset for none)"),
]


@pytest.mark.parametrize("sub, line, name, kind, why", CONFIG_REFUSALS, ids=[c[1] for c in CONFIG_REFUSALS])
def test_a_config_value_is_refused_in_the_runners_words(tmp_path, sub, line, name, kind, why):
    """An empty value is refused, never the working directory (`root = ""` used to write a
    dataset into it), and every refusal is the runner's, word for word."""
    cfg = tmp_path / "o.toml"
    cfg.write_text(f"[{sub}]\n{line}\n")
    want = f"config {cfg}: [{sub}] {name}: {why}"
    with pytest.raises(BuildError) as e:
        options.Layers(sub, cfg, env={}).layered(name, None, kind)
    assert str(e.value) == want
    if RUNNER.exists():
        env = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
        r = subprocess.run([RUNNER, sub, "--config", cfg, ast_path("file_per_sample")], capture_output=True, text=True, env=env)
        assert r.returncode == 2 and r.stderr.splitlines()[0] == f"aeiou {sub}: {want}", r.stderr


@pytest.mark.parametrize("sub, var, value, name, kind, why", ENV_REFUSALS, ids=[f"{c[1]}={c[2]!r}" for c in ENV_REFUSALS])
def test_an_environment_value_is_refused_in_the_runners_words(sub, var, value, name, kind, why):
    with pytest.raises(BuildError) as e:
        options.Layers(sub, None, env={var: value}).layered(name, None, kind)
    assert str(e.value) == f"{var}: {why}"
    if RUNNER.exists():
        env = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
        r = subprocess.run([RUNNER, sub, ast_path("file_per_sample")], capture_output=True, text=True, env={**env, var: value})
        assert r.returncode == 2 and r.stderr.splitlines()[0] == f"aeiou {sub}: {var}: {why}", r.stderr


def test_values_the_layers_accept_from_each_source(tmp_path):
    """Every kind from the environment and the file: a path is kept as the environment has it
    (never trimmed, as the runner keeps it), a list splits on whitespace, a number is trimmed."""
    l = options.Layers("datagen", None, env={"AEIOU_ROOT": " /x ", "AEIOU_ENDPOINT": "a=s3://b  c=s3://d",
                                             "AEIOU_WRITERS": " 4 ", "AEIOU_COORDINATOR": "h:1"})
    assert l.layered("root", None, pathlib.Path) == pathlib.Path(" /x ")
    assert l.layered("endpoint", None, list) == ["a=s3://b", "c=s3://d"]
    assert l.layered("writers", None, int) == 4
    assert l.layered("coordinator", None, str) == "h:1"
    l = options.Layers("run", None, env={"AEIOU_REQUIRE_COLD": "off", "AEIOU_TIME_SCALE": "0.5"})
    assert l.flag("require-cold", None, True) is False and l.layered("time-scale", None, float) == 0.5
    cfg = tmp_path / "o.toml"
    cfg.write_text('[datagen]\nendpoint = "a=s3://b"\ncoordinator = "h:1"\n[run]\ntime-scale = 2\nendpoint = []\n')
    l = options.Layers("datagen", cfg, env={})
    assert l.layered("endpoint", None, list) == ["a=s3://b"] and l.layered("coordinator", None, str) == "h:1"
    l = options.Layers("run", cfg, env={})
    scale = l.layered("time-scale", None, float)
    assert scale == 2.0 and isinstance(scale, float)
    assert l.layered("endpoint", None, list) == []
    with pytest.raises(TypeError):
        options._from_toml(1, dict)


@pytest.mark.parametrize("text, why", [
    (None, "No such file or directory (os error 2)"),
    ("[run\n", "Expected ']'"),
    (b"\xff = 1\n", "invalid start byte"),
    ("run = 3\n", "[run] must be a table"),
])
def test_a_config_file_that_cannot_be_read_is_refused(tmp_path, text, why):
    cfg = tmp_path / "o.toml"
    if isinstance(text, bytes):
        cfg.write_bytes(text)
    elif text is not None:
        cfg.write_text(text)
    with pytest.raises(BuildError, match=f"^config {re.escape(str(cfg))}: .*{re.escape(why)}"):
        options.Layers("run", cfg, env={})
    with pytest.raises(BuildError, match="no subcommand `walk`"):
        options.Layers("walk", None, env={})
