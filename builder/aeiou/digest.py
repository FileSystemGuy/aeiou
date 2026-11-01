"""`aeiou-trace digest`: the questions it asks about a trace and the answers file (`PROJECT_BRIEF.md`
item 47; `DESIGN_REVIEW.md` §3.86 to §3.89).

The digest itself, the structural summary of a trace with a verdict per site on what the trace
decides and an `open` list of what it does not, is the runner's `aeiou-trace digest`
(`runner/aeiou/src/digest.rs`, stage 2 of §3.89): its document, `aeiou_digest: 2`, with the
answers of an answers file applied, and its text report. What stays here until stage 3 ports it
is what asks: the interview, which identifies the application from the records, asks the facts
and then the open items, and checks every answer against the trace (each site's facts carry the
least value the trace allows); and the answers file it writes. `analyze` and `document` run the
binary (`trace.binary()`), the trace read again for every document.
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


def fact_check(f: dict, value: int) -> str | None:
    """Why the trace contradicts a value of a site's fact (`f`, one of the site's `facts`), or None."""
    if "least" in f and value < f["least"]:
        return f"{f['why']}; {value} is {'smaller' if f['name'] == 'size' else 'fewer'}"
    return None


# ---------------------------------------------------------------- the questions (§3.88)


UNKNOWN = "unknown"
ANSWERS_FORMAT = 1
LABELS = {
    "consume": "each file exactly once per pass (a permutation over the set)",
    "choose": "a draw with replacement: a file may come up again within a pass",
    "per-path": "the bytes are a property of the file (its size)",
    "drawn": "the bytes are chosen per access",
    "sequential": "in their sorted order",
    "permuted": "in a shuffled order",
    "earlier_run": "an earlier run of the application wrote them (a warm cache)",
    "other_writer": "another process the trace does not cover writes them",
    "once": "each key is read once",
    "repeated": "a key is read again after its first read",
    "uniform": "spread evenly over the file",
    "skewed": "a few parts of the file take most of the reads",
    "fits": "the file fits in memory at scale",
    "exceeds": "the file exceeds memory at scale",
    "dependent": "each offset depends on what the previous read returned",
    "independent": "the offsets are independent of one another",
    "checkpoint": "a checkpoint: the application's state, read back only after a failure",
    "output": "output: an index build, a dataset, results",
}


def site_name(site: int, phase: str | None) -> str:
    """A site as the report and the interview name it: its template, and its phase when the template is two."""
    return f"{site}/{phase}" if phase else str(site)


def site_line(s: dict) -> str:
    """One line of evidence about a site, by kind."""
    if s["kind"] == "set":
        return f"{s['touches']} touches of {s['distinct']} paths, passes {s['epochs']}"
    if s["kind"] == "keyed":
        a = s["access"]
        return f"keyed: {a['keys']} keys, {a['written']} written, {a['reads']} reads ({a['read_after_write']} after a write, {a['rereads']} re-reads)"
    if s["kind"] == "bulk_write":
        return f"bulk write: {s['distinct']} files in {s['saves']['count']} save(s), {s['rewrite']['rewritten_paths']} rewritten, fsync on {s['sync']['fsync_share']:.0%}"
    if s["kind"] == "restore":
        return f"restore: {s['restore']['files']} files, {_bytes(s['restore']['bytes'])} read before the first write"
    return f"within one open: {s['reads']} reads at {s['distinct_offsets']} offsets over {_bytes(s['span'])}, top tenth {s['skew']['top_10pct_share']:.2f}"


class Console:
    """The terminal: questions on stderr (stdout may be the document), answers read from stdin."""

    def say(self, text: str) -> None:
        import sys
        sys.stderr.write(text + "\n")

    def ask(self, prompt: str) -> str | None:
        import sys
        sys.stderr.write(prompt)
        sys.stderr.flush()
        line = sys.stdin.readline()
        return None if line == "" else line.rstrip("\n")


def interview(an: Trace, io, *, epochs: int = 1, candidates: int | None = None, paths: bool = False, by: str, at: str, apps: dict | None = None, application: dict | None = None) -> tuple[dict, list[dict]]:
    """Ask what the trace cannot decide, in this order: with `apps` (the application records),
    which application the trace is of, confirmed from its path signatures or chosen from the
    list, its version, and the record's options, whose assertions then answer what they can
    (as suggestions the person confirms when no record claims the version); then the facts of
    every site with an open question; then the open items in rank order. Every answer is
    checked against the trace and re-asked when it contradicts it; end of input stops the
    asking. `application` given skips the identification (`--application NAME@VERSION`).
    Returns the document and the answers given, the list `write_answers` keeps and `document`
    replays, with the record's assertions among them carrying their own `by`."""
    kw = dict(epochs=epochs, candidates=candidates, paths=paths, by=by, at=at)
    answers: list[dict] = []
    state = {"application": application}

    def finish():
        return document(an, answers=answers, application=state["application"], **kw), answers

    def choose(prompt: str, n: int) -> int | None:
        """A number from 1 to n, or None at end of input."""
        while True:
            line = io.ask(prompt)
            if line is None:
                return None
            line = line.strip()
            if line.isdigit() and 1 <= int(line) <= n:
                return int(line)
            io.say(f"  a number from 1 to {n}")

    if apps is not None:
        from . import applications as ap
        templates = [(t["template"], t["raw_paths"]) for t in document(an, paths=True)["templates"]]
        app = None
        if application is not None:
            app = apps.get(application["name"])
            version = application.get("version")
        else:
            found = ap.identify(apps, (p for _, ps in templates for p in ps))
            io.say("")
            if found:
                a0, sig = found[0]
                io.say(f"This trace looks like {a0.display} (a path matched {sig}).")
                io.say("    1. yes")
                io.say("    2. another application")
                io.say("    3. none of these, or don't know")
                n = choose("answer [1-3]: ", 3)
                if n is None:
                    return finish()
                app = a0 if n == 1 else None
                listed = n == 2
            else:
                io.say("No path in the trace identifies the application.")
                listed = True
            if listed:
                names = sorted(apps)
                io.say("Which application produced this trace?")
                for i, name in enumerate(names, 1):
                    io.say(f"    {i}. {apps[name].display}: {apps[name].summary}")
                io.say(f"    {len(names) + 1}. none of these, or don't know")
                n = choose(f"answer [1-{len(names) + 1}]: ", len(names) + 1)
                if n is None:
                    return finish()
                app = apps[names[n - 1]] if n <= len(names) else None
            version = None
            if app is not None:
                version = io.ask(f"which version of {app.display}? (Enter for don't know): ")
                if version is None:
                    return finish()
                version = version.strip() or None
        if app is not None:
            record, exact = ap.resolve(app, version)
            chosen = {"name": app.name, "display": app.display, "version": version, "record": None, "exact": False, "options": {}, "blind_spots": []}
            state["application"] = chosen
            if record is None:
                io.say(f"  no record of {app.display} at {version} or below; nothing is assumed about it")
            else:
                chosen["record"], chosen["exact"] = record.version, exact
                if not exact:
                    io.say(f"  no record claims {app.display} {version or 'of unknown version'}; the nearest is {record.version}, read {record.read}: its answers are suggestions you confirm")
                options = chosen["options"]
                if application is not None:
                    options.update(application.get("options", {}))
                for o in record.options:
                    if o["name"] in options:
                        continue
                    io.say("")
                    io.say(f"  {o['question']}")
                    for i, c in enumerate(o["choices"], 1):
                        io.say(f"    {i}. {c}" + (" (the default)" if o.get("default") == c else ""))
                    io.say(f"    {len(o['choices']) + 1}. don't know")
                    n = choose(f"answer [1-{len(o['choices']) + 1}]: ", len(o["choices"]) + 1)
                    if n is None:
                        return finish()
                    if n <= len(o["choices"]):
                        options[o["name"]] = o["choices"][n - 1]
                chosen["blind_spots"] = ap.blind_spots(record, options)
                for b in chosen["blind_spots"]:
                    io.say(f"  blind spot ({b['kind']}): {b['note']}")
                for a in ap.assertions(record, options, templates):
                    if exact:
                        answers.append(a)
                        continue
                    what = f"{a['fact']} {a['value']}" if "fact" in a else f"{a['question']} {a['answer']}"
                    io.say("")
                    io.say(f"  the {record.app.display} {record.version} record says, for {a['template']}: {what}" + (f" ({a['note']})" if a.get("note") else ""))
                    io.say("    1. yes, that holds here")
                    io.say("    2. no")
                    io.say("    3. don't know")
                    n = choose("answer [1-3]: ", 3)
                    if n is None:
                        return finish()
                    if n == 1:
                        mine = {k: v for k, v in a.items() if k not in ("by", "source")}
                        mine["note"] = f"confirmed from the {record.app.display} {record.version} record" + (f": {a['note']}" if a.get("note") else "")
                        answers.append(mine)

    doc = document(an, answers=answers, application=state["application"], **kw)
    open_sites = {(it["site"], it.get("phase")) for it in doc["open"]}
    for s in doc["sites"]:
        if (s["template"], s["phase"]) not in open_sites:
            continue
        name = doc["templates"][s["template"]]["template"]
        io.say("")
        io.say(f"site {site_name(s['template'], s['phase'])}  {name}: {site_line(s)}")
        for f in s["facts"]:
            io.say(f"  {f['ask']}")
            while True:
                line = io.ask("number (Enter for don't know): ")
                if line is None:
                    return finish()
                line = line.strip()
                if not line:
                    break
                if not line.isdigit() or int(line) < 1:
                    io.say("  a whole number of at least 1, or Enter for don't know")
                    continue
                why = fact_check(f, int(line))
                if why:
                    io.say(f"  {why}")
                    continue
                answers.append({"template": name, "fact": f["name"], "value": int(line)} | ({"phase": s["phase"]} if s["phase"] else {}))
                break
    doc = document(an, answers=answers, application=state["application"], **kw)
    for it in doc["open"]:
        s = next(x for x in doc["sites"] if (x["template"], x["phase"]) == (it["site"], it.get("phase")))
        name = doc["templates"][it["site"]]["template"]
        cands = it["candidates"]
        io.say("")
        io.say(f"site {site_name(it['site'], it.get('phase'))}  {name}: {site_line(s)}; {it['question']} undecided")
        io.say(f"  {it['ask']}")
        for i, c in enumerate(cands, 1):
            io.say(f"    {i}. {c}: {LABELS[c]}")
        io.say(f"    {len(cands) + 1}. don't know")
        sc = it["at_scale"]
        if "choose" in sc:
            io.say(f"  at {sc['epochs']} pass(es): a draw with replacement never reads {sc['choose']['never_read_share']:.1%} of the set and re-reads "
                   f"{sc['choose']['repeat_share']:.1%} of what it touches; a permutation reads every file each pass")
        else:
            io.say(f"  at {sc['epochs']} pass(es): {sc['note']}")
        io.say(f"  settles: {it['settles']}")
        while True:
            line = io.ask(f"answer [1-{len(cands) + 1}]: ")
            if line is None:
                return finish()
            line = line.strip()
            if line.isdigit() and 1 <= int(line) <= len(cands) + 1:
                break
            io.say(f"  a number from 1 to {len(cands) + 1}")
        ans = cands[int(line) - 1] if int(line) <= len(cands) else UNKNOWN
        a = {"template": name, "question": it["question"], "answer": ans} | ({"phase": it["phase"]} if "phase" in it else {})
        answers.append(a)
        note = io.ask("how do you know? (Enter to skip): ")
        if note is None:
            return finish()
        if note.strip():
            a["note"] = note.strip()
    return finish()


def write_answers(path: str, answers: list[dict], *, trace: str, trace_sha256: str, by: str, at: str, application: dict | None = None) -> None:
    doc = {"aeiou_answers": ANSWERS_FORMAT, "trace": trace, "trace_sha256": trace_sha256, "by": by, "at": at, "application": application, "answers": list(answers)}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2) + "\n")


def read_answers(path: str) -> tuple[dict, list[dict]]:
    """The answers file: its header (trace, hash, by, at) and its answers."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        raise BuildError(f"{path}: {e.strerror}") from None
    except ValueError as e:
        raise BuildError(f"{path}: {e}") from None
    if not isinstance(doc, dict) or doc.get("aeiou_answers") != ANSWERS_FORMAT:
        raise BuildError(f"{path}: not an aeiou_answers: {ANSWERS_FORMAT} file")
    answers = doc.get("answers")
    if not isinstance(answers, list) or not all(isinstance(a, dict) for a in answers):
        raise BuildError(f"{path}: answers must be a list of objects")
    return doc, answers


# ---------------------------------------------------------------- the text


def _bytes(n: int) -> str:
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"
