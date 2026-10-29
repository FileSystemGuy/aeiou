"""The application records (`applications/`, `DESIGN_REVIEW.md` §3.88): what `aeiou-trace digest`
knows about the applications whose traces it reads, as data. One directory per application:
`application.toml` (name, display, kind, summary, home, source, path signatures, catalog
mechanisms) and `<version>.toml` per version range the facts were read for (`from`, optional
`until`, `read`, `sources`, the `[[option]]`s that change the storage workload with the blind
spots a choice causes, the `[[blind_spot]]`s that hold regardless, and the `[[assert]]`s the
record answers for the digest). `attributes.toml` is the vocabulary; `load` refuses a record
outside it, every problem listed. The three rules live in the digest: the trace always wins, a
record's assertion ranks below the owner's answer and says so in its provenance, and a version
no record was read for gets the nearest lower record as suggestions (`resolve`)."""
from __future__ import annotations

import fnmatch
import pathlib
import re
import tomllib

from .nodes import BuildError

FORMAT = 1
KINDS = ("library", "service", "managed")
CONFIDENCE = ("high", "medium")
# the digest's own vocabulary for what a record may assert (kept here so the loader does not import
# the digest; `tests/test_digest.py` holds the two equal): facts, plus `values:<field>`; questions,
# plus `field:<field>` with the candidates `new_values` and `constant`
FACTS = ("candidates", "passes", "keys", "requests", "size")
QUESTIONS = {"selection": ("consume", "choose"), "size": ("per-path", "drawn"), "order": ("sequential", "permuted"),
             "origin": ("earlier_run", "other_writer"), "reuse": ("once", "repeated"),
             "skew": ("uniform", "skewed"), "memory": ("fits", "exceeds"), "dependency": ("dependent", "independent")}
FIELD_CANDIDATES = ("new_values", "constant")


def question_candidates(q: str) -> tuple | None:
    if q in QUESTIONS:
        return QUESTIONS[q]
    return FIELD_CANDIDATES if q.startswith("field:") else None


class Record:
    def __init__(self, app: "Application", path: pathlib.Path, d: dict):
        self.app = app
        self.path = path
        self.version: str = d["from"]
        self.key = version_key(self.version)
        self.until: str | None = d.get("until")
        self.until_key = version_key(self.until) if self.until else None
        self.read: str = d["read"]
        self.sources: list[str] = d["sources"]
        self.options: list[dict] = d.get("option", [])
        self.blind_spots: list[dict] = d.get("blind_spot", [])
        self.asserts: list[dict] = d.get("assert", [])

    def covers(self, key: tuple) -> bool:
        return self.key <= key and (self.until_key is None or key < self.until_key)


class Application:
    def __init__(self, dir: pathlib.Path, d: dict):
        self.dir = dir
        self.name: str = d["name"]
        self.display: str = d["display"]
        self.kind: str = d["kind"]
        self.summary: str = d["summary"]
        self.home: str = d.get("home", "")
        self.source: str = d.get("source", "")
        self.signatures: list[str] = d.get("signatures", [])
        self.mechanisms: list[str] = d.get("mechanisms", [])
        self.records: list[Record] = []

    def matches(self, path: str) -> str | None:
        """The signature that identifies this application in `path`, or None."""
        for sig in self.signatures:
            if matches(path, sig):
                return sig
        return None


def matches(path: str, sig: str) -> bool:
    """A signature matches the path, or its part after any `/` (so `__*_*.distcp` matches a shard at any depth)."""
    return fnmatch.fnmatchcase(path, sig) or fnmatch.fnmatchcase(path, "*/" + sig)


def version_key(v: str | None) -> tuple | None:
    """`3.0.3` -> (3, 0, 3); a leading `v` and anything after the dotted numbers are dropped; None when there is no number."""
    if not v:
        return None
    m = re.match(r"v?(\d+(?:\.\d+)*)", v.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def default_dir() -> pathlib.Path:
    """The records shipped in the wheel (`aeiou/applications`), or the checkout's `applications/`."""
    here = pathlib.Path(__file__).resolve().parent
    packaged = here / "applications"
    if packaged.is_dir():
        return packaged
    return here.parent.parent / "applications"


class _Problems:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.items: list[str] = []

    def add(self, path: pathlib.Path, what: str):
        try:
            shown = path.relative_to(self.root)
        except ValueError:
            shown = path
        self.items.append(f"{shown}: {what}")


def load(root: pathlib.Path) -> dict[str, Application]:
    """Every application under `root`, checked against `root/attributes.toml`; a BuildError lists every problem."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise BuildError(f"{root}: no such directory of application records")
    problems = _Problems(root)
    vocab = _toml(root / "attributes.toml", problems)
    if vocab is None:
        raise BuildError("\n".join(problems.items))
    kinds, spots, conf = vocab.get("kind", {}), vocab.get("blind_spot", {}), vocab.get("confidence", {})
    apps: dict[str, Application] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        app_file = d / "application.toml"
        if not app_file.is_file():
            problems.add(d, "no application.toml")
            continue
        a = _toml(app_file, problems)
        if a is None:
            continue
        app = _application(d, a, kinds, app_file, problems)
        if app is None:
            continue
        for f in sorted(d.iterdir()):
            if f == app_file:
                continue
            if f.suffix != ".toml" or f.name == "application.toml":
                problems.add(f, "not application.toml or a <version>.toml")
                continue
            r = _toml(f, problems)
            if r is None:
                continue
            rec = _record(app, f, r, spots, conf, problems)
            if rec is not None:
                app.records.append(rec)
        app.records.sort(key=lambda r: r.key)
        if not app.records:
            problems.add(d, "no <version>.toml")
        apps[app.name] = app
    if problems.items:
        raise BuildError("\n".join(problems.items))
    return apps


def _toml(path: pathlib.Path, problems: _Problems) -> dict | None:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        problems.add(path, f"TOML: {e}")
        return None


def _application(d: pathlib.Path, a: dict, kinds: dict, path: pathlib.Path, problems: _Problems) -> Application | None:
    ok = True
    for key in ("name", "display", "kind", "summary"):
        if not isinstance(a.get(key), str) or not a[key]:
            problems.add(path, f"{key} is required")
            ok = False
    if ok and a["name"] != d.name:
        problems.add(path, f"name must be the directory's, {d.name}")
        ok = False
    if ok and a["kind"] not in kinds:
        problems.add(path, f"kind must be one of {', '.join(kinds)}")
        ok = False
    for key in ("signatures", "mechanisms"):
        v = a.get(key, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            problems.add(path, f"{key} must be a list of strings")
            ok = False
    return Application(d, a) if ok else None


def _record(app: Application, path: pathlib.Path, r: dict, spots: dict, conf: dict, problems: _Problems) -> Record | None:
    ok = True
    for key in ("from", "read"):
        if not isinstance(r.get(key), str) or not r[key]:
            problems.add(path, f"{key} is required")
            ok = False
    if not isinstance(r.get("sources"), list) or not r["sources"] or not all(isinstance(s, str) for s in r["sources"]):
        problems.add(path, "sources must be a non-empty list of strings")
        ok = False
    if not ok:
        return None
    if r["from"] != path.stem or version_key(r["from"]) is None:
        problems.add(path, f"from must be the file's stem, {path.stem}, a dotted version")
        return None
    if r.get("until") is not None and (version_key(r["until"]) is None or version_key(r["until"]) <= version_key(r["from"])):
        problems.add(path, "until must be above from")
        ok = False
    options = r.get("option", [])
    names = {}
    for i, o in enumerate(options, 1):
        name = o.get("name")
        if not isinstance(name, str) or not isinstance(o.get("question"), str):
            problems.add(path, f"option {i}: name and question are required")
            ok = False
            continue
        choices = o.get("choices")
        if not isinstance(choices, list) or len(choices) < 2 or not all(isinstance(c, str) for c in choices):
            problems.add(path, f"option {name}: choices must be a list of at least two strings")
            ok = False
            continue
        names[name] = choices
        if "default" in o and o["default"] not in choices:
            problems.add(path, f"option {name}: default must be one of its choices")
            ok = False
        if o.get("confidence") not in conf:
            problems.add(path, f"option {name}: confidence must be {' or '.join(conf)}")
            ok = False
        for choice, spot in o.get("blind_spots", {}).items():
            if choice not in choices:
                problems.add(path, f"option {name}: blind spot for a choice it does not have, {choice}")
                ok = False
            elif spot not in spots:
                problems.add(path, f"option {name}: blind spot {spot} is not in applications/attributes.toml")
                ok = False
    for i, b in enumerate(r.get("blind_spot", []), 1):
        if b.get("kind") not in spots:
            problems.add(path, f"blind spot {b.get('kind')} is not in applications/attributes.toml")
            ok = False
        if b.get("confidence") not in conf:
            problems.add(path, f"blind spot {i}: confidence must be {' or '.join(conf)}")
            ok = False
    for i, a in enumerate(r.get("assert", []), 1):
        if not isinstance(a.get("template"), str):
            problems.add(path, f"assert {i}: template is required")
            ok = False
        if "fact" in a:
            if a["fact"] not in FACTS and not (isinstance(a["fact"], str) and a["fact"].startswith("values:")):
                problems.add(path, f"assert {i}: fact must be one of {', '.join(FACTS)}, or values:<field>")
                ok = False
            v = a.get("value")
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                problems.add(path, f"assert {i}: value must be a whole number of at least 1")
                ok = False
        elif "question" in a:
            cands = question_candidates(a["question"]) if isinstance(a["question"], str) else None
            if cands is None:
                problems.add(path, f"assert {i}: question must be one of {', '.join(QUESTIONS)}, or field:<field>")
                ok = False
            elif a.get("answer") not in cands:
                problems.add(path, f"assert {i}: answer must be one of {', '.join(cands)}")
                ok = False
        else:
            problems.add(path, f"assert {i}: either fact and value or question and answer")
            ok = False
        if a.get("confidence") not in conf:
            problems.add(path, f"assert {i}: confidence must be {' or '.join(conf)}")
            ok = False
        when = a.get("when")
        if when is not None:
            if when.get("option") not in names:
                problems.add(path, f"assert {i}: when names no option of the record, {when.get('option')}")
                ok = False
            elif when.get("choice") not in names[when["option"]]:
                problems.add(path, f"assert {i}: when names no choice of option {when['option']}, {when.get('choice')}")
                ok = False
    return Record(app, path, r) if ok else None


def identify(apps: dict[str, Application], paths) -> list[tuple[Application, str]]:
    """The applications a path signature identifies, by name, each with the first signature that matched."""
    found: dict[str, tuple[Application, str]] = {}
    for p in paths:
        for name, app in apps.items():
            if name in found:
                continue
            sig = app.matches(p)
            if sig is not None:
                found[name] = (app, sig)
    return [found[n] for n in sorted(found)]


def resolve(app: Application, version: str | None) -> tuple[Record | None, bool]:
    """The record for a version and whether it claims it: the one whose range holds the version
    (exact); else the nearest lower record, as suggestions (not exact); none below the lowest; the
    newest record, not exact, when the version is unknown or not a number."""
    key = version_key(version)
    if key is None:
        return (app.records[-1] if app.records else None), False
    lower = None
    for r in app.records:
        if r.covers(key):
            return r, True
        if r.key <= key:
            lower = r
    return lower, False


def provenance(r: Record) -> str:
    return f"{r.app.display} {r.version} record, read {r.read}"


def assertions(r: Record, options: dict[str, str], templates: list[tuple[str, list[str]]]) -> list[dict]:
    """The record's assertions as the digest's answers, for every template one of whose paths
    matches the assertion's glob and whose `when` the chosen options satisfy, each with the record
    as its `by`."""
    out = []
    for a in r.asserts:
        when = a.get("when")
        if when and options.get(when["option"]) != when["choice"]:
            continue
        for name, paths in templates:
            if not any(matches(p, a["template"]) for p in paths):
                continue
            e: dict = {"template": name}
            if "fact" in a:
                e["fact"], e["value"] = a["fact"], a["value"]
            else:
                e["question"], e["answer"] = a["question"], a["answer"]
            if a.get("note"):
                e["note"] = a["note"]
            e["by"] = f"{provenance(r)} ({a['confidence']} confidence)"
            e["source"] = a.get("source", "")
            out.append(e)
    return out


def blind_spots(r: Record, options: dict[str, str], vocab: dict | None = None) -> list[dict]:
    """The blind spots that hold for the chosen options: the record's unconditional ones, then
    those the choices cause, each with the vocabulary's description as `note` when the record gives none."""
    if vocab is None:
        vocab = tomllib.loads((r.path.parent.parent / "attributes.toml").read_text(encoding="utf-8")).get("blind_spot", {})
    out = [{"kind": b["kind"], "note": b.get("note") or vocab.get(b["kind"], ""), "source": b.get("source", "")} for b in r.blind_spots]
    for o in r.options:
        choice = options.get(o["name"])
        spot = o.get("blind_spots", {}).get(choice) if choice else None
        if spot:
            out.append({"kind": spot, "note": vocab.get(spot, ""), "option": o["name"], "choice": choice, "source": o.get("source", "")})
    return out
