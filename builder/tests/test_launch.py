"""`aeiou-launch` (`runner/aeiou-launch`, POSIX sh) with a shim for ssh that prints what it was
asked to run: the hosts from the command line and from `-f FILE`, pdsh-style bracket ranges
(`n[1-4]`, `n[01-10]` zero-padded, `n[1,3-4]-ib`), the rank flags appended to `aeiou run` and
`aeiou datagen` alike, the quoting of the remote command, the exit status, and the usage frame;
then Ctrl-C (and SIGTERM) forwarded to every rank over the session's stdin, and each rank's lines
kept whole in a file."""
import os
import pathlib
import re
import signal
import stat
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LAUNCH = ROOT / "runner" / "aeiou-launch"
# the watcher in front of every remote command: a line INT or TERM on the session's stdin is
# that signal for the command
WATCH = "exec 3<&0 </dev/null; { while read -r s <&3; do case $s in INT|TERM) kill -s $s $$ ;; esac; done; } >/dev/null 2>&1 & exec 3<&-"


def shim(tmp_path, body='echo "$@"\n'):
    p = tmp_path / "rsh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def launch(tmp_path, *args, body='echo "$@"\n'):
    env = {**os.environ, "AEIOU_RSH": str(shim(tmp_path, body))}
    r = subprocess.run([str(LAUNCH), *args], capture_output=True, text=True, env=env)
    lines = r.stdout.splitlines()
    if body == 'echo "$@"\n':
        assert all(WATCH in line for line in lines), lines
        lines = [line.replace(WATCH, "") for line in lines]
    return r.returncode, sorted(lines), r.stderr


def test_hosts_ranges_hostfile_and_the_appended_flags(tmp_path):
    hosts = tmp_path / "hosts"
    hosts.write_text("# the file's hosts\nf[01-02] g\n\n")
    code, lines, err = launch(tmp_path, "-p", "9", "n[1-2]", "m[1,3-4]-ib", "-f", str(hosts), "--", "aeiou", "datagen", "x.ast.json", "--root", "/mnt/a b")
    assert code == 0, err
    expect = ["n1", "n2", "m1-ib", "m3-ib", "m4-ib", "f01", "f02", "g"]
    assert lines == sorted(
        f"[{i} {h}] {h}  'aeiou' 'datagen' 'x.ast.json' '--root' '/mnt/a b' --ranks 8 --rank {i} --coordinator n1:9" for i, h in enumerate(expect)
    )


def test_a_host_file_s_ranges_are_never_file_names(tmp_path):
    """`n[1-2]` in a host file is a bracket range even where a file `n1` exists: the launcher reads
    the file's words without pathname expansion (it took the range for a glob and launched on `n1`
    alone, 2026-10-15); `*` is a bad range, not the directory's files. The last line needs no newline."""
    (tmp_path / "n1").write_text("")
    hosts = tmp_path / "hosts"
    hosts.write_text("n[1-2]   # two\nm[3]")
    env = {**os.environ, "AEIOU_RSH": str(shim(tmp_path))}
    r = subprocess.run([str(LAUNCH), "-f", str(hosts), "--", "aeiou", "run", "x"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    got = sorted(line.replace(WATCH, "") for line in r.stdout.splitlines())
    assert got == [f"[{i}] {h}  'aeiou' 'run' 'x' --ranks 3 --rank {i} --coordinator n1:7311".replace(f"[{i}]", f"[{i} {h}]")
                   for i, h in enumerate(["n1", "n2", "m3"])]
    hosts.write_text("n[*]\n")
    r = subprocess.run([str(LAUNCH), "-f", str(hosts), "--", "aeiou", "run", "x"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 2 and "bad bracket range '*' in host 'n[*]'" in r.stderr, r.stderr


def test_help(tmp_path):
    r = subprocess.run([str(LAUNCH), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == ""
    assert "\nUsage: aeiou-launch [OPTIONS] [HOSTS]... -- aeiou run|datagen|aeiou-datagen <ARGS>...\n" in r.stdout
    assert re.findall(r"^  (-\S+)", r.stdout, re.M) == ["-p", "-f", "-h,"]
    assert subprocess.run([str(LAUNCH), "-h"], capture_output=True, text=True).stdout == r.stdout


def test_run_and_the_default_port(tmp_path):
    code, lines, err = launch(tmp_path, "a", "b", "--", "aeiou", "run", "x.ast.json", "--gpus", "8")
    assert code == 0, err
    assert lines == [
        "[0 a] a  'aeiou' 'run' 'x.ast.json' '--gpus' '8' --ranks 2 --rank 0 --coordinator a:7311",
        "[1 b] b  'aeiou' 'run' 'x.ast.json' '--gpus' '8' --ranks 2 --rank 1 --coordinator a:7311",
    ]


def test_a_failing_rank_fails_the_launch(tmp_path):
    code, lines, err = launch(tmp_path, "a", "b", "--", "aeiou", "run", "x", body='case $1 in b) echo "$1 fails"; exit 4;; esac\necho "$1 ok"\n')
    assert code == 1
    assert lines == ["[0 a] a ok", "[1 b] b fails"]
    assert "aeiou-launch: rank 1 (b) exited with status 4" in err


def test_a_run_that_did_not_pass_keeps_its_status(tmp_path):
    """Status 3, a valid run of a lock that did not pass a constraint (rank 0's verdict,
    contract 0.11), is the launch's when no rank failed otherwise, so a driver tells it from an
    error; beside a failure the failure wins."""
    code, _, err = launch(tmp_path, "a", "b", "--", "aeiou", "run", "x", body='case $1 in a) exit 3;; esac\n')
    assert code == 3 and "aeiou-launch: rank 0 (a): the run did not pass (status 3)" in err
    code, _, err = launch(tmp_path, "a", "b", "--", "aeiou", "run", "x", body='case $1 in a) exit 3;; b) exit 1;; esac\n')
    assert code == 1 and "rank 1 (b) exited with status 1" in err


def test_usage_errors_are_the_frame(tmp_path):
    def frame(*args):
        r = subprocess.run([str(LAUNCH), *args], capture_output=True, text=True)
        assert r.returncode == 2, r.stderr
        assert "\nUsage: aeiou-launch [OPTIONS] [HOSTS]... -- aeiou run|datagen|aeiou-datagen <ARGS>...\n" in r.stderr
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
    assert frame("a", "-f") == "aeiou-launch: a value is required for '-f <FILE>' but none was supplied"
    assert frame("n[]", "--", "aeiou", "run", "x") == "aeiou-launch: empty bracket range in host 'n[]'"
    assert frame("n[1,,2]", "--", "aeiou", "run", "x") == "aeiou-launch: bad bracket range '' in host 'n[1,,2]': want N or LO-HI, digits only"
    assert frame("n[1-2-3]", "--", "aeiou", "run", "x") == "aeiou-launch: bad bracket range '1-2-3' in host 'n[1-2-3]': want N or LO-HI, digits only"
    assert frame("--bogus", "a", "--", "aeiou", "run", "x") == "aeiou-launch: unexpected argument '--bogus' found"


def sshd_shim(tmp_path):
    """ssh's stand-in: the remote command in a session of its own, out of the launcher's process
    group, with SIGINT and SIGTERM at their defaults as sshd leaves them, and stdin passed on."""
    p = tmp_path / "rsh"
    p.write_text(f"#!{sys.executable}\n"
                 "import os, signal, sys\n"
                 "os.setsid()\n"
                 "signal.signal(signal.SIGINT, signal.SIG_DFL)\n"
                 "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
                 "os.execvp('sh', ['sh', '-c', sys.argv[2]])\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


@pytest.mark.parametrize("sig,code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_ctrl_c_stops_every_rank(tmp_path, sig, code):
    # Ctrl-C reaches the launcher's process group and nothing else: a command run over ssh
    # without a tty gets no signal, so the launcher forwards it, waits, cleans up, and exits
    rank = tmp_path / "rank"
    rank.write_text("#!/bin/sh\n"
                    "trap 'echo stopped by INT; exit 130' INT\n"
                    "trap 'echo stopped by TERM; exit 143' TERM\n"
                    f": > {tmp_path}/started.$4; i=0\n"
                    "while [ $i -lt 300 ]; do sleep 0.1; i=$((i+1)); done\n"
                    "echo finished\n")
    rank.chmod(rank.stat().st_mode | stat.S_IXUSR)
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    out = tmp_path / "out"
    env = {**os.environ, "AEIOU_RSH": str(sshd_shim(tmp_path)), "TMPDIR": str(tmpdir)}
    with open(out, "w") as f:
        p = subprocess.Popen([str(LAUNCH), "a", "b", "--", str(rank)], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.time() + 20
            while len(list(tmp_path.glob("started.*"))) < 2:
                assert time.time() < deadline, out.read_text()
                time.sleep(0.05)
            os.killpg(p.pid, sig)
            assert p.wait(timeout=20) == code, out.read_text()
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    name = signal.Signals(sig).name
    text = out.read_text()
    assert f"aeiou-launch: {name}: stopping every rank" in text
    for r, h in ((0, "a"), (1, "b")):
        assert f"[{r} {h}] stopped by {name[3:]}" in text, text
        assert f"aeiou-launch: rank {r} ({h}) exited with status {code}" in text
    assert "finished" not in text
    assert list(tmpdir.iterdir()) == []


def test_ctrl_c_before_every_rank_started(tmp_path):
    """Ctrl-C while rank 0 has its second to come up: rank 0 is stopped, the later ranks are never
    started and are reported so, and the exit status is still Ctrl-C's."""
    rank = tmp_path / "rank"
    rank.write_text("#!/bin/sh\n"
                    "trap 'echo stopped by INT; exit 130' INT\n"
                    f": > {tmp_path}/started.$4; i=0\n"
                    "while [ $i -lt 300 ]; do sleep 0.1; i=$((i+1)); done\n")
    rank.chmod(rank.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "out"
    env = {**os.environ, "AEIOU_RSH": str(sshd_shim(tmp_path))}
    with open(out, "w") as f:
        p = subprocess.Popen([str(LAUNCH), "a", "b", "c", "--", str(rank)], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.time() + 20
            while not (tmp_path / "started.0").exists():
                assert time.time() < deadline, out.read_text()
                time.sleep(0.01)
            os.killpg(p.pid, signal.SIGINT)                   # within the second rank 0 is given
            assert p.wait(timeout=20) == 130, out.read_text()
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    text = out.read_text()
    assert [p.name for p in tmp_path.glob("started.*")] == ["started.0"], text
    assert "[0 a] stopped by INT" in text and "aeiou-launch: rank 0 (a) exited with status 130" in text
    assert "aeiou-launch: rank 1 (b) was not started" in text and "aeiou-launch: rank 2 (c) was not started" in text


def test_lines_stay_whole_in_a_file(tmp_path):
    # two ranks writing at once, the launcher's output a file: every line whole, with its own prefix
    body = "line of the rank that wrote it, long enough to fill a filter's block in a few dozen lines"
    script = f'i=0; while [ $i -lt 2000 ]; do [ $((i % 50)) = 0 ] && sleep 0.05; echo "$i {body}"; i=$((i+1)); done'
    out = tmp_path / "out"
    env = {**os.environ, "AEIOU_RSH": str(shim(tmp_path, 'sh -c "$2"\n'))}
    with open(out, "w") as f:
        r = subprocess.run([str(LAUNCH), "a", "b", "--", "sh", "-c", script], stdout=f, env=env)
    assert r.returncode == 0
    lines = out.read_text().splitlines()
    bad = [line for line in lines if not re.fullmatch(rf"\[(0 a|1 b)\] \d+ {body}", line)]
    assert bad == [] and len(lines) == 4000, (len(lines), bad[:4])
