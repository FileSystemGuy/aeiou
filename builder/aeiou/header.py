"""`aeiou-header`: the header bytes of a container file, made by the format's own library, on
stdout (`DESIGN_REVIEW.md` §3.70, revised 2026-10-12). `aeiou datagen` runs it, from beside
its own binary, for each row count a dataset's files have (at most two: the full files and a
short last one), and writes the rows after it; the user never calls it.

    aeiou-header --format JSON --rows N [--rows N]…

one header per `--rows`, back to back in the order given (each is the framing's `file_header`
bytes long, so the reader splits them by length). `--format` is the dataset's resolved
`format` entry (parameters substituted, as the manifest records it). HDF5 is the one class: the first `data_offset` bytes of a file of `N` rows from
libhdf5 (`formats.Hdf5.header`), refused when the installed h5py and libhdf5 differ from
the entry's `version`, the one its framing was probed with. Nothing but the header goes to
stdout.
"""
from __future__ import annotations

import sys

from . import __version__, emit, usage
from .formats import FormatClass
from .nodes import BuildError


def header(fmt: dict, rows: int) -> bytes:
    """The header of a file of `rows` rows of the dataset whose resolved format entry is `fmt`."""
    if not isinstance(fmt, dict) or fmt.get("class") != "hdf5":
        raise BuildError(f"no header for format class `{fmt.get('class') if isinstance(fmt, dict) else fmt}`: only hdf5 has one")
    if rows < 1:
        raise BuildError(f"--rows {rows}: a file holds at least one row")
    klass = FormatClass.from_spec(fmt)
    have, want = klass.version(), fmt.get("version")
    if have != want:
        raise BuildError(f"the abstract's framing was probed with {want}, this tool has {have}: the header could differ; "
                         "rebuild the abstract with this library, or use the aeiou-header of the release that built it")
    got = klass.header(rows)
    expect = fmt.get("framing", {}).get("file_header")
    if expect is not None and len(got) != expect:
        raise BuildError(f"the header is {len(got)} bytes, the framing says {expect}")
    return got


def parser() -> usage.Parser:
    ap = usage.Parser(prog="aeiou-header", description="Write the header bytes of a container file, made by the format's own library, to stdout (run by `aeiou datagen`)",
                      usage="aeiou-header --format <JSON> --rows <N> [--rows <N>]...")
    ap.add_argument("--format", metavar="JSON", help="Required: the dataset's resolved `format` entry")
    ap.add_argument("--rows", type=int, action="append", metavar="N", help="Required: the rows the file holds; repeatable, one header each, back to back in order")
    ap.version(f"aeiou-header {__version__}")
    return ap


def main(argv=None) -> int:
    ap = parser()
    try:
        a = ap.parse_args(argv)
        missing = usage.Missing(ap)
        missing.want(a.format, "--format <JSON>", "the dataset's resolved `format` entry")
        missing.want(a.rows, "--rows <N>", "the rows the file holds")
        missing.check()
        try:
            fmt = emit.loads(a.format)
        except ValueError as e:
            raise usage.UsageError(f"--format: not JSON: {e}") from None
        if not isinstance(fmt, dict) or "class" not in fmt:
            raise usage.UsageError("--format: not a format entry (an object with `class`)")
        for n in a.rows:
            if n < 1:
                raise usage.UsageError(f"--rows {n}: a file holds at least one row")
        got = b"".join(header(fmt, n) for n in a.rows)
    except usage.UsageError as e:
        return usage.fail(e, ap)
    except BuildError as e:
        return usage.failed(ap.prog, e)
    sys.stdout.buffer.write(got)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
