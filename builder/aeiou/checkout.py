"""A tool run from a checkout under a Python other than the one it pins (`DESIGN_REVIEW.md`
§3.83): a `pip install -e` takes whatever Python the system has, and none of uv's pins, so every
tool says so once, on stderr, and goes on. The checkout is told by `builder/.python-version`
beside the package; a wheel's install has none, and its users may run any Python it allows."""
from __future__ import annotations

import pathlib
import sys

PIN = pathlib.Path(__file__).resolve().parent.parent / ".python-version"


def foreign(pin: pathlib.Path, version_info) -> str | None:
    """What to say when `version_info` is not the version `pin` names, or None."""
    try:
        want = pin.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    now = "%d.%d.%d" % tuple(version_info[:3])
    if now == want:
        return None
    return (f"this checkout pins Python {want} (builder/.python-version) and Python {now} runs it; "
            "./dev.sh setup makes the venv that pins it, and builder/.venv/bin/ has every tool")


def warn(prog: str) -> None:
    message = foreign(PIN, sys.version_info)
    if message:
        print(f"{prog}: warning: {message}", file=sys.stderr)
