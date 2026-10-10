"""`aeiou-catalog`: the records under workloads/ against their dictionary, the measures of a
dry run's metrics, and the refusals (DESIGN_REVIEW.md §3.66). The full check, every shape
re-measured with the runner, is a CI step of its own; here the committed tree is checked
without the dry runs, and one small case is re-measured when the runner binary is there."""
import json
import os
import pathlib
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
sys.path.insert(0, str(BUILDER))

from aeiou import catalog  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
WORKLOADS = ROOT / "workloads"
ATTRS = catalog.load_dictionary(WORKLOADS)


def test_the_committed_tree_is_current(capsys):
    """Every record is valid, every shape was measured on the AST and parameter file committed,
    and CATALOG.md is what the records render."""
    rc = catalog.main(["check", "--workloads", str(WORKLOADS), "--no-shapes"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.splitlines()[-1].endswith(": ok")


def test_every_capture_pair_is_read_from_its_record():
    pairs = catalog.capture_pairs(WORKLOADS)
    assert len(pairs) == len(list(WORKLOADS.rglob("captures/*/*.metrics.json")))
    assert {v for *_, v in pairs} == {"accepted", "not accepted", "nothing judged"}
    # a capture's own fit when it has one, the case's set otherwise; the case's GPU count
    fitted = [p for _, _, _, p, _, _ in pairs if p.name == "fitted.params.json"]
    assert len(fitted) == 2
    assert {g for *_, g, _ in pairs} == {1, 2}


# ---------------------------------------------------------------- the measures

def test_weighted_median_is_by_bytes_not_by_count():
    # 1,000 header reads of 112 bytes and 16 reads of 128 MiB: half the bytes are in the large ones
    h = {"buckets": [[112, 1000], [134217728, 16]]}
    assert catalog._weighted_median(h) == 134217728
    assert catalog._weighted_median({"buckets": []}) is None


def test_classes_are_first_match_with_inclusive_bounds():
    share = ATTRS["read_share"]
    assert catalog.classify(share, 0.0) == "write-only"
    assert catalog.classify(share, 1.0) == "read-only"
    assert catalog.classify(share, 0.8) == "read-mostly"
    assert catalog.classify(share, 0.5) == "mixed"
    assert catalog.classify(share, 0.1) == "write-mostly"
    assert catalog.classify(share, None) is None
    size = ATTRS["run_length_read"]
    assert catalog.classify(size, 16384) == "small" and catalog.classify(size, 16383) == "tiny"


def test_tolerances_relative_absolute_or_either():
    assert catalog.agree(ATTRS["run_length_read"], 1048576, 1310720)          # one quarter-octave bucket
    assert not catalog.agree(ATTRS["run_length_read"], 1048576, 2097152)
    assert catalog.agree(ATTRS["read_share"], 0.61, 0.625) and not catalog.agree(ATTRS["read_share"], 0.61, 0.64)
    # a ratio near zero: the absolute bound holds where the relative one alone would not
    assert catalog.agree(ATTRS["metadata"], 5.2e-5, 1.0e-4)
    assert catalog.agree(ATTRS["reads_own_writes"], 1.0, 1.0) and not catalog.agree(ATTRS["reads_own_writes"], 0.0, 1.0)
    assert catalog.agree(ATTRS["skew"], None, None) and not catalog.agree(ATTRS["skew"], None, 0.0)


def test_measures_of_a_dry_runs_totals():
    hist = lambda *b: {"n": sum(n for _, n in b), "buckets": [list(x) for x in b]}
    reuse = {"first_touch": {"read": 60, "write": 20}, "read_after_read": hist((4096, 30)), "read_after_write": hist((4096, 10)),
             "write_after_read": hist(), "write_after_write": hist()}
    total = {"bytes_read": 300, "bytes_written": 100, "counts": {"open": 4, "close": 4, "lseek": 50, "read": 6, "write": 2},
             "run_length": {"read": {"bytes": hist((65536, 3))}, "write": {"bytes": hist()}},
             "request_size": {"read": hist((4096, 6)), "write": hist((8192, 2))},
             "reuse_distance_bytes": reuse, "popularity_blocks": {"accesses": 120, "top_1_pct": 0.125}}
    m = catalog.measure(total)
    assert m["read_share"] == 0.75 and m["metadata"] == 1.0          # lseek is neither metadata nor data
    assert m["reread"] == 0.3 and m["reads_own_writes"] == 1.0       # 30 of 100 block reads read a block again
    assert m["run_length_read"] == 65536 and m["run_length_write"] is None and m["request_size_write"] == 8192
    assert m["skew"] == 0.125


# ---------------------------------------------------------------- the refusals

@pytest.fixture
def tree(tmp_path):
    """A workload tree of one mechanism, a copy of checkpointing/dcp-save with its case and capture."""
    root = tmp_path / "workloads"
    (root / "checkpointing").mkdir(parents=True)
    shutil.copy(WORKLOADS / "attributes.toml", root)
    shutil.copytree(WORKLOADS / "checkpointing" / "dcp-save", root / "checkpointing" / "dcp-save", ignore=shutil.ignore_patterns("kit"))
    return root


def problems(root):
    return catalog.load(root, catalog.load_dictionary(root))[1]


def edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_the_copy_is_clean(tree):
    assert problems(tree) == []


CASE = "checkpointing/dcp-save/cases/toy-6-items"


@pytest.mark.parametrize("old, new, message", [
    ('kind = "example"', 'kind = "example"\nflavour = "x"', "`flavour` is not an attribute"),
    ('modality = "model-state"', 'modality = "weights"', "not one of image, volume"),
    ('kind = "example"', 'kind = "example"\napi = "sync"', "`api` is derived"),
    ('kind = "example"', 'kind = "example"\napplication = "torch-dcp"', "an attribute of a mechanism, not of a case"),
    ('kind = "example"\n', '', "`kind` is required"),
    ('gpus = [2]', 'gpus = []', "gpus must be a non-empty list"),
    ('ast = "dcp_save"', 'ast = "dcp_load"', "is not an abstract of checkpointing/dcp-save"),
    ('params = "params.json"', 'params = "other.json"', "is not a file of the case"),
    ('kind = "example"', 'kind = "example"\nlength = "items_total"', "not an integer parameter of dcp_save"),
])
def test_case_records_are_refused(tree, old, new, message):
    edit(tree / CASE / "case.toml", old, new)
    assert any(message in p for p in problems(tree)), problems(tree)


def test_a_parameter_file_no_run_names(tree):
    shutil.copy(tree / CASE / "params.json", tree / CASE / "spare.params.json")
    assert any("spare.params.json: a parameter file no [[run]]" in p for p in problems(tree))


@pytest.mark.parametrize("old, new, message", [
    ('verdict = "accepted"', 'verdict = "fine"', "not one of accepted, not-accepted, nothing-judged"),
    ('trace = "trace"', 'trace = "other"', "no other.metrics.json in the capture"),
    ('raw_trace = "not retained"', 'raw_trace = 1', "raw_trace: where the raw trace is"),
])
def test_capture_records_are_refused(tree, old, new, message):
    edit(tree / CASE / "captures" / "2026-10-01" / "capture.toml", old, new)
    assert any(message in p for p in problems(tree)), problems(tree)


def test_a_trace_without_a_pair(tree):
    shutil.copy(tree / CASE / "captures" / "2026-10-01" / "trace.metrics.json", tree / CASE / "captures" / "2026-10-01" / "second.metrics.json")
    assert any("second.metrics.json: a trace no [[pair]]" in p for p in problems(tree))


def test_the_category_is_the_directory(tree):
    shutil.move(tree / "checkpointing", tree / "inference")
    assert any("category 'checkpointing' say workloads/checkpointing/<mechanism>" in p for p in problems(tree))


def test_a_mechanism_with_no_case_needs_its_defaults(tree):
    shutil.rmtree(tree / "checkpointing" / "dcp-save" / "cases")
    assert any("needs a [defaults] table" in p for p in problems(tree))
    with open(tree / "checkpointing" / "dcp-save" / "mechanism.toml", "a") as f:
        f.write('\n[defaults]\nast = "dcp_save"\ngpus = 2\n')
    assert problems(tree) == []


def test_two_asts_of_one_name(tree):
    shutil.copytree(tree / "checkpointing" / "dcp-save", tree / "checkpointing" / "dcp-save-2")
    assert any("the AST name dcp_save is also" in p for p in problems(tree))


def test_a_mechanism_without_its_record(tree):
    (tree / "checkpointing" / "dcp-save" / "mechanism.toml").unlink()
    assert any("a mechanism without mechanism.toml" in p for p in problems(tree))


def test_usage_errors_are_in_the_frame(capsys):
    assert catalog.main([]) == 2
    err = capsys.readouterr().err
    assert err.startswith("aeiou-catalog: the following required arguments were not provided:\n  <COMMAND>")


def test_a_stale_shape_is_reported_without_the_runner(tree, capsys):
    shape = tree / CASE / "shape.json"
    doc = json.loads(shape.read_text())
    doc["runs"][0]["params_sha256"] = "0" * 64
    shape.write_text(json.dumps(doc))
    assert catalog.main(["check", "--workloads", str(tree), "--no-shapes"]) == 1
    assert "measured on another dcp_save" in capsys.readouterr().out


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_a_case_remeasured_is_its_shape(tree):
    """`write` on the copy gives the committed shape byte for byte (the dry run is deterministic)."""
    mechanisms, _ = catalog.load(tree, ATTRS)
    shapes = catalog.measure_all(mechanisms, ATTRS, str(RUNNER))
    assert shapes[tree / CASE / "shape.json"] == (WORKLOADS / CASE / "shape.json").read_text()
    assert catalog.stale_shapes(mechanisms, tree, ATTRS, str(RUNNER)) == []
