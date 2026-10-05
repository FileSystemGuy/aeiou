"""`aeiou-launch` (`runner/aeiou-launch`, POSIX sh) with a shim for ssh that prints what it was
asked to run: the hosts from the command line and from `-f FILE`, pdsh-style bracket ranges
(`n[1-4]`, `n[01-10]` zero-padded, `n[1,3-4]-ib`), the rank flags appended to `aeiou run` and
`aeiou datagen` alike, the quoting of the remote command, the exit status, and the usage frame."""
import os
import pathlib
import stat
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LAUNCH = ROOT / "runner" / "aeiou-launch"


def shim(tmp_path, body='echo "$@"\n'):
    p = tmp_path / "rsh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def launch(tmp_path, *args, body='echo "$@"\n'):
    env = {**os.environ, "AEIOU_RSH": str(shim(tmp_path, body))}
    r = subprocess.run([str(LAUNCH), *args], capture_output=True, text=True, env=env)
    return r.returncode, sorted(r.stdout.splitlines()), r.stderr


def test_hosts_ranges_hostfile_and_the_appended_flags(tmp_path):
    hosts = tmp_path / "hosts"
    hosts.write_text("# the file's hosts\nf[01-02] g\n\n")
    code, lines, err = launch(tmp_path, "-p", "9", "n[1-2]", "m[1,3-4]-ib", "-f", str(hosts), "--", "aeiou", "datagen", "x.ast.json", "--root", "/mnt/a b")
    assert code == 0, err
    expect = ["n1", "n2", "m1-ib", "m3-ib", "m4-ib", "f01", "f02", "g"]
    assert lines == sorted(
        f"[{i} {h}] {h}  'aeiou' 'datagen' 'x.ast.json' '--root' '/mnt/a b' --ranks 8 --rank {i} --coordinator n1:9" for i, h in enumerate(expect)
    )


def test_run_and_the_default_port(tmp_path):
    code, lines, err = launch(tmp_path, "a", "b", "--", "aeiou", "run", "x.ast.json", "--gpus", "8")
    assert code == 0, err
    assert lines == [
        "[0 a] a  'aeiou' 'run' 'x.ast.json' '--gpus' '8' --ranks 2 --rank 0 --coordinator a:7311",
        "[1 b] b  'aeiou' 'run' 'x.ast.json' '--gpus' '8' --ranks 2 --rank 1 --coordinator a:7311",
    ]


def test_a_failing_rank_fails_the_launch(tmp_path):
    code, lines, err = launch(tmp_path, "a", "b", "--", "aeiou", "run", "x", body='case $1 in b) echo "$1 fails"; exit 3;; esac\necho "$1 ok"\n')
    assert code == 1
    assert lines == ["[0 a] a ok", "[1 b] b fails"]
    assert "aeiou-launch: rank 1 (b) exited with status 3" in err


def test_usage_errors_are_the_frame(tmp_path):
    def frame(*args):
        r = subprocess.run([str(LAUNCH), *args], capture_output=True, text=True)
        assert r.returncode == 2, r.stderr
        assert "\nUsage: aeiou-launch [OPTIONS] [HOSTS]... -- aeiou run|datagen <ARGS>...\n" in r.stderr
        assert r.stderr.rstrip().endswith("For more information, try 'aeiou-launch --help'.")
        return r.stderr.split("\n\n", 1)[0]

    assert frame() == ("aeiou-launch: the following required arguments were not provided:\n"
                       "  [HOSTS]...  the hosts (on the command line or in -f FILE), one rank each; the first is rank 0 and the coordinator\n"
                       "  <ARGS>...   after --: the `aeiou run` or `aeiou datagen` command line every host runs")
    assert frame("a", "--").startswith("aeiou-launch: the following required arguments were not provided:\n  <ARGS>...")
    assert frame("n[a-b]", "--", "aeiou", "run", "x") == "aeiou-launch: bad bracket range 'a-b' in host 'n[a-b]': want N or LO-HI, digits only"
    assert frame("n[4-1]", "--", "aeiou", "run", "x") == "aeiou-launch: bad bracket range '4-1' in host 'n[4-1]': 4 is above 1"
    assert frame("-f", str(tmp_path / "none"), "--", "aeiou", "run", "x").startswith("aeiou-launch: cannot read the host file")
    assert frame("-p") == "aeiou-launch: a value is required for '-p <PORT>' but none was supplied"
    assert frame("--bogus", "a", "--", "aeiou", "run", "x") == "aeiou-launch: unexpected argument '--bogus' found"
