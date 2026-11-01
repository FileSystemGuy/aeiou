"""`aeiou-trace digest` from Python, for the tests (`PROJECT_BRIEF.md` item 47; `DESIGN_REVIEW.md`
§3.86 to §3.89).

The digest, the structural summary of a trace with a verdict per site on what the trace decides
and an `open` list of what it does not, is the runner's `aeiou-trace digest`
(`runner/aeiou/src/digest.rs`, stage 2 of §3.89), and so, since stage 3, are its interview, the
application records, and the answers file (`interview.rs`, `applications.rs`). What is left here
runs the binary: `analyze` and `document` give it a trace and the answers to apply as an answers
file, and read back its document or its report. Stage 5 retires it with the Python tests that use it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import weakref

from .nodes import BuildError
from .trace import binary


# ---------------------------------------------------------------- the binary


class Trace:
    """A trace the binary digests: the file (a temporary copy when it was given as lines), what
    selects its calls, and its sha256; the document without answers is made once, when it is
    analyzed, so a trace the binary refuses is refused there."""

    def __init__(self, path: str, root: str | None, exclude=(), cwd: str | None = None, owned: str | None = None):
        self.path, self.root, self.exclude, self.cwd = path, root, list(exclude), cwd
        if owned:
            self._cleanup = weakref.finalize(self, shutil.rmtree, owned, True)
        with open(path, "rb") as f:
            self.sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        self.base = self.run()
        self.source = self.base["source"]

    def args(self, *, epochs: int = 1, candidates: int | None = None, paths: bool = False) -> list[str]:
        a = [self.path] + (["--root", self.root] if self.root else [])
        for g in self.exclude:
            a += ["--exclude", g]
        if self.cwd:
            a += ["--cwd", self.cwd]
        a += ["--epochs", str(epochs)] + (["--candidates", str(candidates)] if candidates is not None else []) + (["--paths"] if paths else [])
        return a

    def run(self, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, answers=(), by: str | None = None, at: str | None = None, application: dict | None = None, text: bool = False):
        """The document, from `aeiou-trace digest --no-questions`, with `answers` given to it as an
        answers file; with `text`, its report instead."""
        with tempfile.TemporaryDirectory(prefix="aeiou-digest-") as d:
            cmd = [binary(), "digest"] + self.args(epochs=epochs, candidates=candidates, paths=paths) + ["--no-questions"]
            if answers or application or by or at:
                path = os.path.join(d, "answers.json")
                write_answers(path, list(answers), trace=self.path, trace_sha256=self.sha256, by=by, at=at, application=application)
                cmd += ["--answers", path]
            r = subprocess.run(cmd + (["--text"] if text else []), capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise BuildError(r.stderr.strip().removeprefix("aeiou-trace digest: ").removeprefix(f"{self.path}: "))
        if text:
            return r.stdout
        doc = json.loads(r.stdout)
        doc.pop("trace", None)
        doc.pop("trace_sha256", None)
        return doc


def analyze(lines, root: str | None, *, exclude=(), cwd=None) -> Trace:
    """A trace given as lines (an exported trace file, an strace, or an object-store request log),
    copied to a temporary file the binary reads."""
    d = tempfile.mkdtemp(prefix="aeiou-digest-")
    path = os.path.join(d, "trace")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
    return Trace(path, root, exclude, cwd, owned=d)


def analyze_file(path: str, root: str | None, *, exclude=(), cwd=None) -> Trace:
    return Trace(path, root, exclude, cwd)


def document(an: Trace, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, answers=(), by: str | None = None, at: str | None = None, application: dict | None = None) -> dict:
    """The digest document, with `answers` checked against the trace and recorded under
    `asserted` with `by` and `at`, or under `rejected` with why (the binary's)."""
    if not (answers or application or by or at) and (epochs, candidates, paths) == (1, None, False):
        return json.loads(json.dumps(an.base))
    return an.run(epochs=epochs, candidates=candidates, paths=paths, answers=answers, by=by, at=at, application=application)


def report(an: Trace, **kw) -> str:
    """The text report of `document(an, **kw)`, the binary's."""
    return an.run(text=True, **kw)


def digest_of(lines, root: str | None, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, exclude=(), cwd=None, answers=(), by=None, at=None, application=None) -> dict:
    """`analyze` then `document`, in one call."""
    return document(analyze(lines, root, exclude=exclude, cwd=cwd), epochs=epochs, candidates=candidates, paths=paths, answers=answers, by=by, at=at, application=application)


ANSWERS_FORMAT = 1


def write_answers(path: str, answers: list[dict], *, trace: str, trace_sha256: str, by: str, at: str, application: dict | None = None) -> None:
    """The answers file (`aeiou_answers: 1`) the binary applies."""
    doc = {"aeiou_answers": ANSWERS_FORMAT, "trace": trace, "trace_sha256": trace_sha256, "by": by, "at": at, "application": application, "answers": list(answers)}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
