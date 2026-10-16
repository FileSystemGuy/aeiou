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
WORKLOADS = ROOT / "workloads"


def ast_path(name):
    """The committed AST `name`, wherever its mechanism lives under workloads/ (DESIGN_REVIEW.md §3.66)."""
    return next(WORKLOADS.rglob(f"{name}.ast.json"))
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
    ast = ast_path("tfrecord_stream")
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
    script = ast_path("file_per_sample").parent / "file_per_sample.py"
    message, _ = frame(tool("datagen", "--root", tmp_path, script), "aeiou-datagen")
    assert message == f"{script}: a builder script, not an abstract; `aeiou-build {script}` writes the abstracts it authors (`<name>.ast.json`) next to it, and those are what this tool takes"
    message, _ = frame(tool("datagen", "--root", tmp_path, "file_per_sample"), "aeiou-datagen")
    assert message == "file_per_sample: No such file or directory (os error 2); an abstract is the `.ast.json` file aeiou-build writes from a builder script"
    bad = tmp_path / "x.ast.json"
    bad.write_text("import aeiou\n")
    message, _ = frame(tool("params", "defaults", bad), "aeiou-params defaults")
    assert message == f"{bad}: not an abstract (Expecting value: line 1 column 1 (char 0)); an abstract is the `.ast.json` file aeiou-build writes from a builder script"
    # the other way round, a per-script failure of aeiou-build, status 1
    r = tool("cli", ast_path("file_per_sample"))
    assert r.returncode == 1 and r.stderr.startswith("FAIL file_per_sample.ast.json: an abstract, not a builder script; aeiou-build takes the `.py` that authors it")


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
    message, _ = frame(tool("params", "check", ast_path("file_per_sample")), "aeiou-params check")
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


HELPS = [("cli",), ("params",), ("params", "defaults"), ("params", "check"), ("params", "safetensors"), ("params", "npz"),
         ("datagen",), ("trace",), ("trace", "metrics"), ("trace", "export"), ("trace", "compare"),
         ("catalog",), ("catalog", "check"), ("catalog", "write"), ("catalog", "describe"), ("header",)]


@pytest.mark.parametrize("columns", ["80", "120"])
def test_the_help_wraps_at_spaces_only(columns):
    """(RC1 review, R50) the help wrapped a long word mid-word (`-e trac` / `e=%file,...`) and
    at its hyphens (`aeiou dry-` / `run`): clap wraps at spaces, and so does `usage.Formatter`."""
    for args in HELPS:
        r = tool(*args, "--help", env={"COLUMNS": columns})
        assert r.returncode == 0 and r.stdout.startswith(r.stdout.splitlines()[0]), (args, r.stderr)
        for line in r.stdout.splitlines():
            assert not (line.endswith("-") and line[-2:-1].isalnum()), f"{args}: a word broken at a hyphen: {line!r}"
    out = tool("trace", "--help", env={"COLUMNS": columns}).stdout
    assert " trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy" in out, out
    assert "aeiou dry-run --metrics-json" in " ".join(out.split()), out


def test_aeiou_header_refuses_what_is_no_format_entry_or_row_count():
    """(RC1 review, R49) a `--format` that is not an object with `class`, and a `--rows` below
    one, are usage errors (status 2), not a format class `None` with status 1."""
    for fmt in ("{}", "null", "[]", '{"reader": "h5py"}', '"hdf5"'):
        message, usage = frame(tool("header", "--format", fmt, "--rows", "1"), "aeiou-header")
        assert message == "--format: not a format entry (an object with `class`)", fmt
        assert usage == "Usage: aeiou-header --format <JSON> --rows <N> [--rows <N>]..."
    for rows in ("0", "-1"):
        message, _ = frame(tool("header", "--format", '{"class": "hdf5"}', "--rows", "4", "--rows", rows), "aeiou-header")
        assert message == f"--rows {rows}: a file holds at least one row"
    # an entry of a class with no header is still a failure during the work
    r = tool("header", "--format", '{"class": "tfrecord"}', "--rows", "1")
    assert (r.returncode, r.stderr) == (1, "aeiou-header: no header for format class `tfrecord`: only hdf5 has one\n")


def test_aeiou_build_says_a_script_does_not_exist(tmp_path):
    """(RC1 review, R60) a missing script was `FileNotFoundError at line 1013`, a line of runpy,
    not of any script; an exception is placed at a line of the script only when it has one."""
    r = tool("cli", tmp_path / "nonexist.py", "-o", tmp_path)
    assert (r.returncode, r.stderr) == (1, f"FAIL nonexist.py: no such file ({tmp_path / 'nonexist.py'}); aeiou-build takes builder scripts (`.py`)\n")
    r = tool("cli", tmp_path, "-o", tmp_path)
    assert r.returncode == 1 and r.stderr.startswith(f"FAIL {tmp_path.name}: not a file ({tmp_path});"), r.stderr
    raises = tmp_path / "raises.py"
    raises.write_text("import json\njson.loads('{')\n")
    r = tool("cli", raises, "-o", tmp_path)
    assert r.returncode == 1 and r.stderr.startswith("FAIL raises.py: JSONDecodeError at line 2: "), r.stderr


def test_aeiou_params_check_words_a_missing_file_as_the_runner_does(tmp_path):
    """(RC1 review, R65) `[Errno 2] No such file or directory: 'nonexist.json'` was Python's
    text; the suite's is `aeiou check`'s, `<path>: No such file or directory (os error 2)`."""
    missing = tmp_path / "nonexist.json"
    r = tool("params", "check", ast_path("file_per_sample"), missing)
    assert (r.returncode, r.stdout) == (1, f"FAIL nonexist.json\n  {missing}: No such file or directory (os error 2)\n"), r.stdout + r.stderr
    if RUNNER.exists():
        rs = subprocess.run([str(RUNNER), "check", str(missing)], capture_output=True, text=True)
        assert f"  {missing}: No such file or directory (os error 2)\n" in rs.stdout + rs.stderr, rs.stdout + rs.stderr


def _head(*argv, merge=False):
    """`argv | head -c1` (`2>&1 | head -c1` when `merge`): one byte read, then the reader goes
    away. The exit status and the stderr left."""
    full = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
    full["PYTHONPATH"] = str(BUILDER)
    p = subprocess.Popen([sys.executable, *map(str, argv)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT if merge else subprocess.PIPE, cwd=str(BUILDER), env=full)
    assert p.stdout.read(1)
    p.stdout.close()
    err = b"" if merge else p.stderr.read()
    return p.wait(), err.decode()


def test_a_reader_that_goes_away_is_exit_141_and_no_traceback(tmp_path):
    """(RC1 review, R66) `aeiou-params check ... | head -1` with more than a pipe buffer (64 KiB)
    to print was a BrokenPipeError traceback and exit 1; it is the runner's 141 and nothing said
    (runner/REFERENCE.md §15, "A reader that goes away"), stdout's pipe or stderr's."""
    ast = ast_path("hdf5_map")
    pset = tmp_path / ("p" * 200 + ".params.json")   # a long name: few files fill the pipe
    assert tool("params", "defaults", "-o", pset, ast).returncode == 0
    many = [pset] * 300   # ~85 KB of `ok` lines
    assert _head("-m", "aeiou.params", "check", ast, *many) == (141, "")
    assert _head("-m", "aeiou.params", "check", ast, *many, merge=True) == (141, "")


@pytest.mark.parametrize("status, expect", [(0, 141), (None, 141), (1, 1), (2, 2)])
def test_a_reader_that_goes_away_leaves_a_failures_status(status, expect):
    """`usage.tool`: the work goes on to its end with the pipe gone (what follows the broken
    write still runs), and its failure's status stands; only a success becomes 141."""
    main = ("import sys\nfrom aeiou import usage\n@usage.tool\ndef main(argv=None):\n"
            "    sys.stdout.write('x' * 200000)\n    print('after', file=sys.stderr)\n"
            f"    return {status}\nsys.exit(main())\n")
    assert _head("-c", main) == (expect, "after\n")
    assert _head("-c", main, merge=True) == (expect, "")


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_the_runner_prints_the_same_list_for_the_same_mistake(tmp_path):
    """`aeiou datagen` and `aeiou-datagen` differ only in the command's name: the message and
    the frame are the same bytes, so the suite reads as one tool."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AEIOU_")}
    for args in ((), (ast_path("tfrecord_stream"),), ("--root", tmp_path, "nope"), ("--root", tmp_path, ast_path("file_per_sample").parent / "file_per_sample.py")):
        rs = subprocess.run([str(RUNNER), "datagen", *map(str, args)], capture_output=True, text=True, env=env)
        py = tool("datagen", *args)
        assert rs.returncode == py.returncode == 2, rs.stderr + py.stderr
        a = rs.stderr.replace("aeiou datagen", "aeiou-datagen").replace("what the runner takes", "what this tool takes")
        assert a == py.stderr, f"runner:\n{rs.stderr}\npython:\n{py.stderr}"
