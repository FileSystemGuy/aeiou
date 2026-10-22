"""No I/O of aeiou's own during the measured run (DESIGN_REVIEW.md §3.78 point 16, a rule):
nothing the runner writes for its own records (the report, the step times, the counters)
touches a file system between the gate and the end of the run; they are held in memory and
written after it. Checked under `strace -f -y`: between the run's first and last op on a file of
its dataset (the manifest checks come before the gate), every path a call names, and every file
a write goes to, is under the root, a pseudo file system (`/proc`, `/sys`, `/dev`), a pipe, or a
socket."""
import os
import pathlib
import re
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou")).resolve()
AST = next((ROOT / "workloads").rglob("file_per_sample.ast.json"))
PARAMS = AST.parent / "params" / "smoke.params.json"


def _strace_works():
    if not shutil.which("strace"):
        return False
    return subprocess.run(["strace", "-f", "-o", os.devnull, "true"], capture_output=True).returncode == 0


needs_run = pytest.mark.skipif(not RUNNER.exists() or not _strace_works(), reason="needs the runner binary and a working strace")

PATH = re.compile(r'"(/[^"]*)"|<(/[^>]*)>')
CWD = re.compile(r"AT_FDCWD<[^>]*>")
PSEUDO = ("/proc/", "/sys/", "/dev/")


@needs_run
def test_a_run_touches_no_file_system_of_its_own_between_the_gate_and_the_end(tmp_path):
    root = (tmp_path / "root").resolve()
    shape = [str(AST), "--params-file", str(PARAMS), "--param", "samples=200"]
    subprocess.run([str(RUNNER), "datagen", *shape, "--root", str(root)], check=True, capture_output=True)
    trace, report = tmp_path / "run.strace", tmp_path / "report.json"
    r = subprocess.run(["strace", "-f", "-y", "-qq", "-e", "trace=%file,write,pwrite64,writev,pwritev,pwritev2,ftruncate,fsync,fdatasync",
                        "-o", str(trace), str(RUNNER), "run", *shape, "--param", "steps=12", "--gpus", "2", "--root", str(root),
                        "--time-scale", "0.01", "--report-json", str(report)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = trace.read_text().splitlines()
    on_data = [i for i, line in enumerate(lines) if f"{root}/train/" in line and "/.aeiou" not in line]
    assert len(on_data) > 100, "the run's own ops are in the trace"
    first, last = on_data[0], on_data[-1]
    own = []
    for line in lines[first:last + 1]:
        for m in PATH.finditer(CWD.sub("AT_FDCWD", line)):
            p = m.group(1) or m.group(2)
            if not (p == str(root) or p.startswith(str(root) + "/") or p.startswith(PSEUDO)):
                own.append(line)
                break
    assert not own, "calls on files of aeiou's own during the run:\n" + "\n".join(own[:20])
    # and the report is written after it, from memory
    writes = [i for i, line in enumerate(lines) if str(report) in line]
    assert writes and min(writes) < first and max(writes) > last, "the report is made empty before the gate and written after the end"
