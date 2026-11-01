"""The runner's `aeiou-trace` from Python, for the tests (`DESIGN_REVIEW.md` §3.89). The tool is the
binary (`runner/aeiou/src/bin/aeiou_trace.rs`), Rust whole since stage 4 of §3.89 (2026-11-01, when
`compare` was ported, `compare.rs`): `metrics` and `export` since 2026-10-29, the digest, its
interview, the application records, and the answers file since 2026-10-31. What is left here is
for the tests: the binary found (`binary`), a trace given as lines exported through it
(`export_of`), and the runner's histogram buckets (`LogHist`).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile

from .nodes import BuildError

# ---------------------------------------------------------------- histograms


class LogHist:
    """Quarter-octave buckets, as `metrics::LogHist`: a bucket's lower bound keeps the
    value's top three bits."""

    __slots__ = ("n", "sum", "max", "buckets")

    def __init__(self):
        self.n = 0
        self.sum = 0
        self.max = 0
        self.buckets: dict[int, int] = {}

    @staticmethod
    def lower(v: int) -> int:
        if v < 4:
            return v
        e = v.bit_length() - 1
        return (v >> (e - 2)) << (e - 2)

    def add(self, v: int, weight: int = 1):
        self.n += weight
        self.sum += v * weight
        if v > self.max:
            self.max = v
        b = self.lower(v)
        self.buckets[b] = self.buckets.get(b, 0) + weight


# ---------------------------------------------------------------- the binary


def binary() -> str:
    """The runner's `aeiou-trace` (`DESIGN_REVIEW.md` §3.89): `$AEIOU_TRACE`, else the one beside
    `$AEIOU_RUNNER`, else the one on the PATH, else the checkout's release build."""
    if os.environ.get("AEIOU_TRACE"):
        return os.environ["AEIOU_TRACE"]
    if os.environ.get("AEIOU_RUNNER"):
        beside = pathlib.Path(os.environ["AEIOU_RUNNER"]).with_name("aeiou-trace")
        if beside.exists():
            return str(beside)
    found = shutil.which("aeiou-trace")
    if found:
        return found
    built = pathlib.Path(__file__).resolve().parents[2] / "runner" / "target" / "release" / "aeiou-trace"
    if built.exists():
        return str(built)
    raise BuildError("the runner's aeiou-trace binary is not found: put it on the PATH, or set AEIOU_TRACE to it (DESIGN_REVIEW.md §3.89)")


def export_of(lines, root: str, exclude=(), cwd: str | None = None) -> tuple[dict, list[dict]]:
    """The exported header and lines of a trace given as lines, an strace or an object-store
    request log, from `aeiou-trace export` under `root`; its refusal as a `BuildError` with the
    binary's own words."""
    with tempfile.TemporaryDirectory(prefix="aeiou-trace-") as d:
        src, out = os.path.join(d, "trace"), os.path.join(d, "trace.jsonl")
        with open(src, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line if line.endswith("\n") else line + "\n")
        args = [binary(), "export", src, "--root", root, "-o", out]
        for g in exclude:
            args += ["--exclude", g]
        if cwd:
            args += ["--cwd", cwd]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            msg = r.stderr.strip().removeprefix("aeiou-trace export: ").removeprefix(f"{src}: ")
            raise BuildError(msg.removesuffix(f"; {out} not written"))
        with open(out, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
    return docs[0], docs[1:]
