"""`AEIOU_REQUIRE`: the capabilities a job provides, so a test skipped for lacking one fails.

The tests that drive the runner binary skip when it is not built, which is right on a laptop
and wrong in a CI job that built it: a job that runs no such test, or runs them with the binary
missing, passes while testing nothing (DESIGN_REVIEW.md §3.75: `test_coord.py` was skipped in
every job, and two bugs went unseen). A job names what it built, comma-separated, and a skip
whose reason names one of them is reported as a failure, naming the capability:

    AEIOU_REQUIRE=runner,formats,strace,header,py-formats pytest ...
"""
from __future__ import annotations

import os

import pytest

# capability -> the words every skip for lacking it says
CAPABILITIES = {
    "runner": ("runner binary",),
    "formats": ("cargo feature `formats`",),
    "object": ("cargo feature `object`",),
    "strace": ("working strace",),
    "header": ("aeiou-header",),
    "minio": ("TEST_MINIO_BIN",),
    # the builder's extras (pyproject.toml), as `pytest.importorskip` says a module is missing
    "py-formats": tuple(f"could not import '{m}'" for m in ("pyarrow", "h5py", "numpy", "crc32c", "xxhash")),
    "py-s3": ("could not import 'obstore'",),
}


def required() -> list:
    names = [n.strip() for n in os.environ.get("AEIOU_REQUIRE", "").split(",") if n.strip()]
    unknown = [n for n in names if n not in CAPABILITIES]
    if unknown:
        raise pytest.UsageError(f"AEIOU_REQUIRE: unknown capability {', '.join(unknown)} (known: {', '.join(CAPABILITIES)})")
    return names


def pytest_configure(config):
    config._aeiou_require = required()


def _reason(report) -> str:
    lr = report.longrepr
    reason = lr[2] if isinstance(lr, tuple) and len(lr) == 3 else str(lr)
    return reason.removeprefix("Skipped: ")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = _reason(report)
    for cap in item.config._aeiou_require:
        if any(w in reason for w in CAPABILITIES[cap]):
            report.outcome = "failed"
            report.longrepr = f"skipped, but AEIOU_REQUIRE names `{cap}`: {reason}"
            return
