"""`AEIOU_REQUIRE`: the capabilities a job provides, so a test skipped for lacking one fails.

The tests that drive the runner binary skip when it is not built, which is right on a laptop
and wrong in a CI job that built it: a job that runs no such test, or runs them with the binary
missing, passes while testing nothing (DESIGN_REVIEW.md §3.75: `test_coord.py` was skipped in
every job, and two bugs went unseen). A job names what it built, comma-separated, and a skip
whose reason names one of them is reported as a failure, naming the capability:

    AEIOU_REQUIRE=runner,formats,strace,header,py-formats pytest ...

`AEIOU_WITHOUT` is the other side: the capabilities a job deliberately does not provide,
because another job runs what needs them (CI's `abstracts` job builds no runner; its
`runner-integration` job runs every test). A test marked `skipif` for one of them is
deselected, not skipped, and counted by capability at the end; a test that skips for one at run
time fails, since it cannot be deselected: give it the mark. So such a job reports no skip.

    AEIOU_WITHOUT=runner,formats,object,strace,header,minio AEIOU_REQUIRE=py-formats pytest ...
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


def required(var: str = "AEIOU_REQUIRE") -> list:
    names = [n.strip() for n in os.environ.get(var, "").split(",") if n.strip()]
    unknown = [n for n in names if n not in CAPABILITIES]
    if unknown:
        raise pytest.UsageError(f"{var}: unknown capability {', '.join(unknown)} (known: {', '.join(CAPABILITIES)})")
    return names


def pytest_configure(config):
    config._aeiou_require = required()
    config._aeiou_without = required("AEIOU_WITHOUT")
    both = [n for n in config._aeiou_require if n in config._aeiou_without]
    if both:
        raise pytest.UsageError(f"AEIOU_REQUIRE and AEIOU_WITHOUT both name {', '.join(both)}")
    config._aeiou_deselected = {}


def _capability(reason: str, among) -> str | None:
    return next((cap for cap in among if any(w in reason for w in CAPABILITIES[cap])), None)


def pytest_collection_modifyitems(config, items):
    """Deselect each test a `skipif` mark would skip for a capability `AEIOU_WITHOUT` names."""
    if not config._aeiou_without:
        return
    keep, drop = [], []
    for item in items:
        cap = None
        for m in item.iter_markers("skipif"):
            cond = m.args[0] if m.args else m.kwargs.get("condition", True)
            if cond is True or (not isinstance(cond, str) and bool(cond)):
                cap = _capability(m.kwargs.get("reason", ""), config._aeiou_without)
                if cap:
                    break
        if cap:
            drop.append(item)
            config._aeiou_deselected[cap] = config._aeiou_deselected.get(cap, 0) + 1
        else:
            keep.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_sessionfinish(session):
    # an xdist worker deselects what the controller never collects: it hands the counts over
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["aeiou_deselected"] = session.config._aeiou_deselected


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    # every worker collects the same tests, so one worker's counts are the run's
    got = getattr(node, "workeroutput", {}).get("aeiou_deselected")
    if got:
        node.config._aeiou_deselected = got


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    d = config._aeiou_deselected
    if d:
        by = ", ".join(f"{cap} {n}" for cap, n in sorted(d.items()))
        terminalreporter.write_line(f"AEIOU_WITHOUT: {sum(d.values())} deselected, needing {by} (another job runs them)")


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
    cap = _capability(reason, item.config._aeiou_without)
    if cap:
        report.outcome = "failed"
        report.longrepr = f"skipped at run time for `{cap}`, which AEIOU_WITHOUT names: mark the test skipif, so it is deselected: {reason}"
        return
    for cap in item.config._aeiou_require:
        if any(w in reason for w in CAPABILITIES[cap]):
            report.outcome = "failed"
            report.longrepr = f"skipped, but AEIOU_REQUIRE names `{cap}`: {reason}"
            return
