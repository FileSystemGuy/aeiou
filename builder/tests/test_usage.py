"""The usage-error frame of the suite (`aeiou/usage.py`, `runner/aeiou/src/usage.rs`,
`runner/REFERENCE.md` §15): every Python tool prints a wrong command line the same way, lists
every missing requirement at once, and the list shrinks by exactly what the next attempt
supplies; the Rust runner prints the same bytes for the same mistake (when its binary is
built); a wrong file for the abstract says what an abstract is; the help has the runner's
look. Exit status 2 for a usage error, 1 for a failure during the work."""
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
EXAMPLES = ROOT / "schema" / "examples"
# the runner binary: $AEIOU_RUNNER, else the newest build (a stale release build next to a
# fresh debug one would compare yesterday's wording)
_BUILDS = [p for p in (ROOT / "runner" / "target" / "release" / "aeiou", ROOT / "runner" / "target" / "debug" / "aeiou") if p.exists()]
RUNNER = pathlib.Path(os.environ["AEIOU_RUNNER"]) if "AEIOU_RUNNER" in os.environ else max(_BUILDS, key=lambda p: p.stat().st_mtime, default=ROOT / "runner" / "target" / "release" / "aeiou")

FOOTER = "For more information, try '{prog} --help'.\n"
MISSING = "the following required arguments were not provided:"


def tool(module, *args, env=None, stdin=None):
    full = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
    full.update({"PYTHONPATH": str(BUILDER), **(env or {})})
    return subprocess.run([sys.executable, "-m", f"aeiou.{module}", *map(str, args)], input=stdin, capture_output=True, text=True, cwd=str(BUILDER), env=full)


def frame(r, prog):
    """The stderr of a usage error, split: the message and the usage block."""
    assert r.returncode == 2, r.stdout + r.stderr
    assert r.stderr.startswith(f"{prog}: "), r.stderr
    assert r.stderr.endswith("\n\n" + FOOTER.format(prog=prog)), r.stderr
    body = r.stderr[len(prog) + 2 : -len(FOOTER.format(prog=prog)) - 2]
    message, usage = body.split("\n\nUsage: ", 1)
    return message, "Usage: " + usage


def test_aeiou_datagen_lists_every_missing_argument_and_the_list_shrinks(tmp_path):
    ast = EXAMPLES / "train_stream_tfrecord.ast.json"
    message, usage = frame(tool("datagen"), "aeiou-datagen")
    assert message == (
        f"{MISSING}\n"
        "  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n"
        "  --root <DIR>     directory the abstract's paths are relative to\n"
        "                   (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)"
    )
    assert usage.startswith("Usage: aeiou-datagen [OPTIONS] --root <DIR> <ABSTRACT_PATH>\n")
    # the abstract given: only --root remains, and the column narrows to it
    message, _ = frame(tool("datagen", ast), "aeiou-datagen")
    assert message == (
        f"{MISSING}\n"
        "  --root <DIR>  directory the abstract's paths are relative to\n"
        "                (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)"
    )
    # --root from a lower layer is not missing (the line's next mistake is reported instead)
    r = tool("datagen", ast, "--gpus", "x", env={"AEIOU_ROOT": str(tmp_path)})
    message, _ = frame(r, "aeiou-datagen")
    assert message == "argument --gpus: invalid int value: 'x'"
    # the parser's own errors wear the same frame
    message, _ = frame(tool("datagen", "--bogus", ast, "--root", tmp_path), "aeiou-datagen")
    assert message == "unrecognized arguments: --bogus"


def test_a_wrong_file_for_the_abstract_says_what_an_abstract_is(tmp_path):
    script = BUILDER / "abstracts" / "train_small_files.py"
    message, _ = frame(tool("datagen", "--root", tmp_path, script), "aeiou-datagen")
    assert message == f"{script}: a builder script, not an abstract; `aeiou-build {script}` writes the abstracts it authors (`<name>.ast.json`) next to it, and those are what this tool takes"
    message, _ = frame(tool("datagen", "--root", tmp_path, "train_small_files"), "aeiou-datagen")
    assert message == "train_small_files: No such file or directory (os error 2); an abstract is the `.ast.json` file aeiou-build writes from a builder script"
    bad = tmp_path / "x.ast.json"
    bad.write_text("import aeiou\n")
    message, _ = frame(tool("params", "defaults", bad), "aeiou-params defaults")
    assert message == f"{bad}: not an abstract (Expecting value: line 1 column 1 (char 0)); an abstract is the `.ast.json` file aeiou-build writes from a builder script"
    # the other way round, a per-script failure of aeiou-build, status 1
    r = tool("cli", EXAMPLES / "train_small_files.ast.json")
    assert r.returncode == 1 and r.stderr.startswith("FAIL train_small_files.ast.json: an abstract, not a builder script; aeiou-build takes the `.py` that authors it")


def test_every_tool_and_subcommand_lists_what_it_lacks():
    message, usage = frame(tool("cli"), "aeiou-build")
    assert message == f"{MISSING}\n  <SCRIPTS>...  builder scripts (`.py`), each constructing one or more Workloads"
    assert usage == "Usage: aeiou-build [OPTIONS] <SCRIPTS>..."
    message, _ = frame(tool("params"), "aeiou-params")
    assert message == f"{MISSING}\n  <COMMAND>  one of defaults, check, safetensors, npz"
    message, usage = frame(tool("params", "safetensors"), "aeiou-params safetensors")
    assert message == (
        f"{MISSING}\n"
        "  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n"
        "  <SHARDS>...      the safetensors shard files, in dataset id order\n"
        "  -o <FILE>        the parameter file to write"
    )
    assert usage == "Usage: aeiou-params safetensors [OPTIONS] -o <FILE> <ABSTRACT_PATH> <SHARDS>..."
    message, _ = frame(tool("params", "check", EXAMPLES / "train_small_files.ast.json"), "aeiou-params check")
    assert message == f"{MISSING}\n  <FILES>...  the parameter files (`.params.json`) to validate against the abstract"
    message, _ = frame(tool("trace"), "aeiou-trace")
    assert message == f"{MISSING}\n  <COMMAND>  one of metrics, export, compare"
    message, usage = frame(tool("trace", "export"), "aeiou-trace export")
    assert message == (
        f"{MISSING}\n"
        "  <TRACE>       the strace output file, or - for stdin\n"
        "  --root <DIR>  the directory the traced application's paths are under; the export is relative to it\n"
        "  -o <FILE>     the trace file to write"
    )
    assert usage == "Usage: aeiou-trace export [OPTIONS] --root <DIR> -o <FILE> <TRACE>"
    # a requirement that holds only for some input is listed when it does, with the reason
    r = tool("trace", "metrics", "-", stdin='1 0.0 openat(AT_FDCWD, "/x", O_RDONLY) = 3 <0.0001>\n')
    message, _ = frame(r, "aeiou-trace metrics")
    assert message == (
        f"{MISSING}\n"
        "  --root <DIR>  count only calls on paths under DIR (repeatable)\n"
        "                (needed for an strace; an exported trace file carries its root)"
    )


def test_a_failure_during_the_work_names_the_command_and_exits_1(tmp_path):
    r = tool("trace", "compare", tmp_path / "absent.json", tmp_path / "b.json")
    assert r.returncode == 1
    assert r.stderr == f"aeiou-trace compare: {tmp_path / 'absent.json'}: No such file or directory (os error 2)\n"


def test_the_help_has_the_runners_look():
    for module, args, prog in (("datagen", (), "aeiou-datagen"), ("cli", (), "aeiou-build"), ("params", ("npz",), "aeiou-params npz"), ("trace", ("compare",), "aeiou-trace compare")):
        r = tool(module, *args, "--help")
        assert r.returncode == 0, r.stderr
        lines = r.stdout.splitlines()
        assert lines[0] and not lines[0].startswith("Usage"), "the description comes first"
        assert lines[1] == "" and lines[2].startswith(f"Usage: {prog} "), r.stdout
        assert "Arguments:" in lines and "Options:" in lines, r.stdout
        assert any(line.startswith("  -h, --help") and line.rstrip().endswith("Print help") for line in lines), r.stdout
        assert "usage:" not in r.stdout and "positional arguments" not in r.stdout
    out = tool("params", "npz", "--help").stdout
    assert "  -o, --out <FILE>" in out and "<ABSTRACT_PATH>" in out


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_the_runner_prints_the_same_list_for_the_same_mistake(tmp_path):
    """`aeiou datagen` and `aeiou-datagen` differ only in the command's name: the message and
    the frame are the same bytes, so the suite reads as one tool."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
    for args in ((), (EXAMPLES / "train_stream_tfrecord.ast.json",), ("--root", tmp_path, "nope"), ("--root", tmp_path, BUILDER / "abstracts" / "train_small_files.py")):
        rs = subprocess.run([str(RUNNER), "datagen", *map(str, args)], capture_output=True, text=True, env=env)
        py = tool("datagen", *args)
        assert rs.returncode == py.returncode == 2, rs.stderr + py.stderr
        a = rs.stderr.replace("aeiou datagen", "aeiou-datagen").replace("what the runner takes", "what this tool takes")
        assert a == py.stderr, f"runner:\n{rs.stderr}\npython:\n{py.stderr}"
