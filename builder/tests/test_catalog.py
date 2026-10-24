"""`aeiou-catalog`: the records under workloads/ against their dictionary, the measures of a
dry run's metrics, and the refusals (DESIGN_REVIEW.md §3.66). The full check, every shape
re-measured with the runner, is a CI step of its own; here the committed tree is checked
without the dry runs, and one small case is re-measured when the runner binary is there."""
import json
import math
import os
import pathlib
import re
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
    # bytes (DESIGN_REVIEW.md §3.79): 6,000 read first, 3,000 read again, 1,000 read after a write
    reuse = {"first_touch": {"read": 6000, "write": 2000}, "read_after_read": hist((4096, 3000)), "read_after_write": hist((4096, 1000)),
             "write_after_read": hist(), "write_after_write": hist(), "whole": 0, "partial": 4000}
    total = {"bytes_read": 300, "bytes_written": 100, "counts": {"open": 4, "close": 4, "lseek": 50, "read": 6, "write": 2},
             "run_length": {"read": {"bytes": hist((65536, 3))}, "write": {"bytes": hist()}},
             "request_size": {"read": hist((4096, 6)), "write": hist((8192, 2))},
             "reuse_extents": reuse, "popularity_bytes": {"accesses": 12000, "top_1_pct": 0.125}}
    m = catalog.measure(total)
    assert m["read_share"] == 0.75 and m["metadata"] == 1.0          # lseek is neither metadata nor data
    assert m["reread"] == 0.3 and m["reads_own_writes"] == 1.0       # 3,000 of 10,000 bytes read had been read
    assert m["run_length_read"] == 65536 and m["run_length_write"] is None and m["request_size_write"] == 8192
    assert m["skew"] == 0.125


# ---------------------------------------------------------------- the refusals

@pytest.fixture
def tree(tmp_path):
    """A workload tree of one mechanism, a copy of checkpointing/dcp-save with its case and capture,
    and every other category's and class's directory empty."""
    root = tmp_path / "workloads"
    root.mkdir()
    shutil.copy(WORKLOADS / "attributes.toml", root)
    catalog.make_tree(root, ATTRS)
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
    ('kind = "example"', 'kind = "example"\nmetrics_block = 4096', "`metrics_block` is not an attribute"),     # gone with blocks, DESIGN_REVIEW.md §3.79
    ('kind = "example"', 'kind = "use"', "`accelerator` is required when `kind` is use"),
    ('kind = "example"', 'kind = "example"\naccelerator = "H100-80G"', "`accelerator` = 'H100-80G': not one of a100, h100, b200"),
])
def test_case_records_are_refused(tree, old, new, message):
    edit(tree / CASE / "case.toml", old, new)
    assert any(message in p for p in problems(tree)), problems(tree)


def test_a_case_names_its_accelerator(tree):
    """DESIGN_REVIEW.md §3.78 point 18: a use case names the accelerator its batch and step time
    are for; an example, traced on a stand-in, may."""
    original = (tree / CASE / "case.toml").read_text()
    for kind in ("use", "example"):
        (tree / CASE / "case.toml").write_text(original.replace('kind = "example"', f'kind = "{kind}"\naccelerator = "h100"'))
        assert [p for p in problems(tree) if "case.lock.json" not in p] == [], kind


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
    shutil.move(tree / "checkpointing" / "dcp-save", tree / "kvcache" / "dcp-save")
    assert any("category 'checkpointing' say workloads/checkpointing/<mechanism>" in p for p in problems(tree))


MECH = "checkpointing/dcp-save/mechanism.toml"


@pytest.mark.parametrize("old, new, message", [
    ('category = "checkpointing"', 'category = "checkpointing"\nclass = "map"', "class 'map': checkpointing has no classes"),
    ('phase = ["training"]', 'phase = ["training", "training"]', "a list of distinct values"),
    ('phase = ["training"]', 'phase = []', "a list of distinct values"),
    ('domain = ["ai"]', 'domain = ["ai", "astrology"]', "a list of distinct values"),
    ('phase = ["training"]', 'phase = "training"', None),                       # one value needs no list
    ('phase = ["training"]', 'phase = ["training", "inference"]', None),
    ('application = "torch-dcp"', 'application = ["torch-dcp"]', "not one of"),  # one value only
    ('domain = ["ai"]\n', '', "`domain` is required"),
])
def test_mechanism_records(tree, old, new, message):
    edit(tree / MECH, old, new)
    found = problems(tree)
    assert (not found) if message is None else any(message in p for p in found), found


def test_a_category_with_classes_takes_its_mechanisms_inside_them(tree):
    shutil.move(tree / "checkpointing" / "dcp-save", tree / "input-pipeline" / "dcp-save")
    edit(tree / "input-pipeline" / "dcp-save" / "mechanism.toml", 'category = "checkpointing"', 'category = "input-pipeline"')
    assert any("input-pipeline has classes (map, stream): the mechanism names its class" in p for p in problems(tree))
    edit(tree / "input-pipeline" / "dcp-save" / "mechanism.toml", 'category = "input-pipeline"', 'category = "input-pipeline"\nclass = "map"')
    assert any("say workloads/input-pipeline/map/<mechanism>" in p for p in problems(tree))
    assert any("input-pipeline/dcp-save: not a class of input-pipeline" in p for p in problems(tree))
    shutil.move(tree / "input-pipeline" / "dcp-save", tree / "input-pipeline" / "map" / "dcp-save")
    # a lock names its abstract by its path below workloads/: a mechanism moved re-locks its cases
    found = problems(tree)
    cases = sorted((tree / "input-pipeline" / "map" / "dcp-save" / "cases").iterdir())
    assert len(found) == len(cases) == 2, found
    assert all(any(f"{c.name}/case.lock.json: case.lock.json is stale" in p for p in found) for c in cases), found
    from aeiou import lock
    for case in cases:
        (case / lock.LOCK_FILE).write_text(lock.render(lock.make_lock(case, workloads=tree)))
    assert problems(tree) == []


def test_every_category_has_its_directory_and_no_other(tree):
    shutil.rmtree(tree / "rl")
    (tree / "training").mkdir()
    found = problems(tree)
    assert any(p.startswith("rl: a category without its directory") for p in found)
    assert any(p.startswith("training: not a category") for p in found)


def test_write_makes_the_tree_and_its_readmes(tree, capsys):
    shutil.rmtree(tree / "dataprep")
    assert catalog.main(["write", "--workloads", str(tree), "--no-shapes"]) == 0
    readme = (tree / "dataprep" / "shuffle" / "README.md").read_text()
    assert readme.startswith("# `dataprep/shuffle`") and "workloads/dataprep/shuffle/<mechanism>/" in readme
    assert "[`dcp-save`](dcp-save/)" in (tree / "checkpointing" / "README.md").read_text()


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
    shapes = catalog.measure_all(mechanisms, ATTRS, str(RUNNER), every=True)
    assert shapes[tree / CASE / "shape.json"] == (WORKLOADS / CASE / "shape.json").read_text()
    assert catalog.stale_shapes(mechanisms, tree, ATTRS, str(RUNNER)) == []


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_a_shapes_counts_are_checked_with_the_runner(tree, capsys):
    """The op counts, bytes, and datasets' projection a shape records (the descriptions and
    CATALOG.md print them) are held exactly to the dry run, not only its measures."""
    shape = tree / CASE / "shape.json"
    doc = json.loads(shape.read_text())
    doc["runs"][0]["counts"]["close"] += 12345
    doc["runs"][0]["bytes_written"] += 1
    shape.write_text(json.dumps(doc))
    assert catalog.main(["write", "--workloads", str(tree), "--no-shapes"]) == 0   # the description follows the numbers
    assert catalog.main(["check", "--workloads", str(tree), "--no-shapes"]) == 0, capsys.readouterr().out
    capsys.readouterr()
    assert catalog.main(["check", "--workloads", str(tree), "--runner", str(RUNNER)]) == 1
    assert "dcp_save: the dry run no longer gives the recorded counts, bytes_written (aeiou-catalog write)" in capsys.readouterr().out


FAISS = "vector-search/faiss-ivf-ondisk/cases/sift1m-ivf1024-pq32"


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_counts_are_the_cases_length_when_the_check_length_is_shorter():
    """A check length shorter than the case's re-measures there, but the counts the shape
    records are the case's length's, and they are compared with a run at that length."""
    mechanisms, problems = catalog.load(WORKLOADS, ATTRS)
    assert problems == []
    m = next(m for m in mechanisms if m["rel"] == str(pathlib.Path(FAISS).parent.parent))
    case = next(c for c in m["cases"] if c["name"] == pathlib.Path(FAISS).name)
    run = case["runs"][0]
    recorded = json.loads((WORKLOADS / FAISS / "shape.json").read_text())["runs"][0]
    length = recorded["length"]
    assert length["check"] == length["case"] == 20
    length["check"] = 10
    with catalog.Scheduler(len(catalog.SEEDS)) as pool:
        differ = catalog.recheck(str(RUNNER), run, case["gpus"][0], recorded, ATTRS, pool)
        assert not set(differ) & set(catalog.COUNTED), differ
        recorded["datasets"][0]["directories"] += 7
        differ = catalog.recheck(str(RUNNER), run, case["gpus"][0], recorded, ATTRS, pool)
        assert "datasets" in differ and "ops" not in differ, differ


# ---------------------------------------------------------------- the descriptions (DESIGN_REVIEW.md §3.76)

from aeiou import describe  # noqa: E402

SCHEMA = json.loads((ROOT / "schema" / "abstract-ast.schema.json").read_text())["$defs"]


def test_the_describer_has_words_for_every_kind_of_the_schema():
    """A kind added to the contract fails here until the prose has words for it."""
    assert set(describe.STATEMENTS) == set(SCHEMA["node"]["properties"])
    assert set(describe.OPS) <= set(describe.STATEMENTS)
    assert set(describe.DISTS) == set(SCHEMA["dist"]["properties"])
    assert set(describe.EXPRS) == set(SCHEMA["expr"]["anyOf"][1]["properties"])
    assert set(describe.HANDLES) == set(SCHEMA["handle"]["properties"])
    assert set(describe.MEASURE_WORDS) == set(catalog.MEASURES)


def every_kind() -> dict:
    """An AST (not a valid workload) holding every statement, expression, handle, and distribution kind."""
    p, i, r = (lambda n: {"param": n}), (lambda n: {"index": n}), (lambda n: {"ref": n})
    exprs = [{"at": {"ref": "x", "index": {"sub": [i("i"), 1]}}}, {"actor": "id"}, {"actor": "count"},
             {"elem": {"array": "arr", "index": 1}}, {"len": "arr"}, {"sum": "arr"}, {"size": r("f")}, {"offset": r("s")},
             {"unit_index": r("s")}, {"units": r("f")}, {"chunks": r("f")}, {"count": "ds"}, {"dirs": "ds"},
             {"mod": [i("i"), 3]}, {"ceil_div": [p("n"), 3]}, {"min": [1, 2]}, {"max": [1, 2]}, {"neg": 1}, {"div": [7, 2]},
             {"and": [{"eq": [1, 1]}, {"ne": [1, 2]}]}, {"or": [{"lt": [i("i"), 1]}, {"le": [1, 2]}, {"gt": [r("x"), 1]}, {"ge": [1, 1]}]},
             {"not": r("x")}, {"cond": {"if": r("x"), "then": 1, "else": 2}}, {"draw": {"uniform": {"lo": 0, "hi": 9}}}]
    dists = [{"const": 4}, {"uniform64": {}}, {"normal": {"mean": 100, "sd": 10, "min": 0}},
             {"lognormal": {"median": 4096, "sigma": 0.5, "max": 65536}}, {"empirical": {"values": [1, 2], "weights": [3, 1]}},
             {"mixture": [{"weight": 1, "dist": None}, {"weight": 3, "dist": {"const": 1}}]}]
    body = [{"let": {"name": f"e{n}", "value": e}} for n, e in enumerate(exprs)]
    body += [{"let": {"name": f"d{n}", "value": {"draw": d}}} for n, d in enumerate(dists)]
    body += [
        {"let": {"name": "s", "value": {"consume": "ds"}}},
        {"let": {"name": "f", "value": {"file": {"of": r("s")}}}},
        {"let": {"name": "g", "value": {"file": {"dataset": "ds", "id": 0, "chunk": 1}}}},
        {"let": {"name": "h", "value": {"dir": {"dataset": "ds", "id": 0}}}},
        {"let": {"name": "o", "value": {"object": {"namespace": "out", "fields": {"k": i("i")}}}}},
        {"let": {"name": "v", "value": {"inventory": "ds"}}},
        {"let": {"name": "q", "value": {"pick": {"dataset": "ds", "dist": {"zipf": {"s": 1.1}}}}}},
        {"let": {"name": "q2", "value": {"pick": {"dataset": "ds", "dist": {"hotset": {"fraction": 0.1, "weight": 0.9}}}}}},
        {"let": {"name": "u", "value": {"unit": {"of": r("s")}}}},
        {"let": {"name": "c", "value": {"column": {"of": r("u"), "index": 0}}}},
        {"channel": {"name": "ch", "capacity": 4, "ordered": False}},
        {"parallel": {"index": "w", "width": 2, "body": [{"put": {"channel": "ch", "seq": i("w")}}]}},
        {"take": {"channel": "ch"}},
        {"loader": {"name": "ld", "index": "b", "workers": 2, "prefetch": 2, "batches": p("n"), "body": [
            {"loop": {"index": "j", "from": 1, "to": p("n"), "step": 2, "body": [{"compute": {"ns": 5000}}]}}]}},
        {"take": {"channel": "ld"}},
        {"barrier": {"scope": "global"}}, {"barrier": {"scope": "host"}}, {"barrier": {"scope": "pair"}},
        {"choose": {"arms": [{"weight": 3, "body": [{"stat": {"file": r("f")}}]}, {"weight": 1, "body": []}]}},
        {"cond": {"if": p("on"), "then": [{"fsync": {"file": r("f")}}], "else": [{"fdatasync": {"file": r("f")}}]}},
        {"phase": {"name": "p1", "body": [{"trace": {"file": "app.trace", "sha256": "ab" * 32}}]}},
        {"loop": {"index": "i", "to": {"dirs": "ds"}, "body": [
            {"open": {"file": r("o"), "flags": ["WRONLY", "CREAT"], "mode": 0o644, "expect": ["EEXIST"]}},
            {"write": {"file": r("o"), "len": p("n"), "offset": 0, "repeat": 2}},
            {"read": {"file": r("o"), "len": 4096, "repeat": "until_eof"}},
            {"read": {"file": r("o"), "len": 4096, "offset": {"offset": r("s")}}},
            {"lseek": {"file": r("o"), "offset": 0, "whence": "END"}},
            {"ioctl": {"file": r("o"), "request": "FIONREAD"}},
            {"fadvise": {"file": r("o"), "advice": "WILLNEED", "offset": 0, "len": 0}},
            {"ftruncate": {"file": r("o"), "len": 0}}, {"fallocate": {"file": r("o"), "len": 4096}},
            {"close": {"file": r("o")}}, {"unlink": {"file": r("o")}},
            {"mkdir": {"dir": r("h")}}, {"readdir": {"dir": r("h"), "repeat": "until_end"}}, {"rmdir": {"dir": r("h")}},
            {"rename": {"from": r("o"), "to": r("o")}}]}},
    ]
    return {"ast": "0.13", "name": "every_kind", "doc": "Every kind.",
            "params": {"n": {"default": 8, "unit": "bytes"}, "on": {"default": False}, "arr": {"default": [1, 2, 3]}},
            "datasets": {"ds": {"pattern": "d/{id div 4}/f{id}", "layout": "compound", "access": "map", "count": 100,
                                "samples_per_file": 5, "size": {"const": 4096}, "seed": 1,
                                "format": {"class": "tfrecord", "framing": {"unit": 5, "columns": [{"weight": 1}]}},
                                "inventory": {"name": "list.txt", "entry": 8}}},
            "namespaces": {"out": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 2}},
            "actors": {"gpu": {"body": body}}}


def test_every_kind_is_put_in_words():
    d = describe.Describer(every_kind(), {}, 2)
    text = d.paragraph([]) + "\n" + "\n".join(describe.Describer(every_kind(), {}, 2).account())
    assert "{'" not in text and "None" not in text, text
    for words in ("the next sample of `ds` from the shuffle", "Zipf over the ids of `ds`", "90% of draws over the hottest 10%",
                  "a mixture: 25% none; 75% always 1", "One of these, chosen by a positional draw", "Replays the captured trace",
                  "`rename` `o` to `o`", "For each of the 5 directories `i` of `ds`", "`fallocate`", "If `on` (false here: skipped)",
                  "Otherwise (taken here)", "Waits at the barrier `pair`", "2 requests of 8 bytes (`n`) at offset 0"):
        assert words in text, words
    # the compound dataset's directories: ceil(20 files / 4)
    assert d.dirs("ds") == 5


def test_numbers_are_worked_out_from_parameters():
    ast = {"name": "t", "params": {"xfer": {"default": 1 << 20, "unit": "bytes"}, "n": {"default": 3, "unit": "count"},
                                   "every": {"default": 1, "unit": "count"}}, "actors": {}}
    d = describe.Describer(ast, {"n": 4}, 1)
    assert d.value({"mul": [{"param": "xfer"}, 2]}) == "2 MiB (`xfer × 2`)"
    assert d.value({"param": "n"}, "count") == "4 (`n`)"                        # the parameter file's value
    assert d.value({"add": [{"index": "i"}, {"param": "n"}]}) == "`i + n`"     # a loop index keeps it a formula
    assert d.fold({"eq": [{"mod": [{"index": "step"}, {"param": "every"}]}, 0]}).value is True   # every step
    assert d.sym({"sub": [{"param": "a"}, {"mod": [{"param": "b"}, 64]}]}) == "a − (b mod 64)"
    assert d.sym({"ceil_div": [{"sub": [{"param": "a"}, 1]}, {"param": "b"}]}) == "⌈(a − 1) / b⌉"


def test_distributions_are_given_with_exact_percentiles():
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    assert d.dist({"lognormal": {"median": 110 * 1024, "sigma": 0.45}}, "bytes") == \
        "lognormal, median 110 KiB, σ 0.45 (5th–95th percentile 52.5 KiB to 230.6 KiB)"
    assert d.dist({"lognormal": {"median": 100, "sigma": 0.3, "min": 7, "max": 7}}, "bytes") == "always 7 bytes (a lognormal clamped to it)"
    assert d.dist({"uniform": {"lo": 0, "hi": 10}}) == "uniform over 0 to 9"
    many = {"empirical": {"values": list(range(1, 21)), "weights": [1] * 20}}
    assert d.dist(many) == "one of 20 weighted values, median 10, 5th–95th percentile 1 to 19, mean 10.5"


def test_dead_branches_leave_the_paragraph():
    body = [{"cond": {"if": {"param": "walk"}, "then": [{"stat": {"file": {"ref": "f"}}}]}},
            {"cond": {"if": {"not": {"param": "walk"}}, "then": [{"unlink": {"file": {"ref": "f"}}}]}}]
    ast = {"name": "t", "doc": "T.", "params": {"walk": {"default": True}}, "actors": {"gpu": {"body": body}}}
    assert describe.Describer(ast, {}, 1).paragraph([]).endswith("The one `gpu` actor (one per GPU): stat.")
    assert describe.Describer(ast, {"walk": False}, 4).paragraph([]).endswith("Each of the 4 `gpu` actors (one per GPU): unlink.")


def test_a_description_edited_by_hand_is_drift(tree, capsys):
    path = tree / CASE / "DESCRIPTION.md"
    path.write_text(path.read_text().replace("## In brief", "## In brief\n\nA note."))
    assert catalog.main(["check", "--workloads", str(tree), "--no-shapes"]) == 1
    assert f"{CASE}/DESCRIPTION.md: differs" in capsys.readouterr().out


def test_a_shape_of_another_version_is_stale(tree, capsys):
    shape = tree / CASE / "shape.json"
    doc = json.loads(shape.read_text())
    doc["shape_version"] = 1
    shape.write_text(json.dumps(doc))
    assert catalog.main(["check", "--workloads", str(tree), "--no-shapes"]) == 1
    assert f"shape_version 1, this aeiou-catalog writes {catalog.SHAPE_VERSION}" in capsys.readouterr().out


def test_describe_prints_any_abstract(capsys):
    case = WORKLOADS / CASE
    assert catalog.main(["describe", str(case.parent.parent / "dcp_save.ast.json"), "--params-file", str(case / "params.json"),
                         "--gpus", "2", "--no-shapes"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# `dcp_save`\n") and "from the AST, `params.json`;" in out
    assert "### The `gpu` actor (2 instances, one per GPU)" in out and "### Measured shape" not in out
    assert catalog.main(["describe"]) == 2
    assert "<AST>" in capsys.readouterr().err


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_describe_measures_the_shape_with_the_runner(capsys):
    case = WORKLOADS / CASE
    assert catalog.main(["describe", str(case.parent.parent / "dcp_save.ast.json"), "--params-file", str(case / "params.json"),
                         "--gpus", "2", "--workloads", str(WORKLOADS), "--runner", str(RUNNER)]) == 0
    out = capsys.readouterr().out
    committed = json.loads((case / "shape.json").read_text())["runs"][0]
    assert "### Measured shape" in out and f"Ops in all (seed 1): {committed['ops']:,}:" in out


def test_describe_refuses_a_bad_option_value_in_the_frame(tmp_path, capsys):
    """`--gpus` below 1 is the command line's mistake, as the runner's `--gpus 0` is: the usage
    frame and exit 2, before the abstract is read; an abstract that is not there is the frame
    too (as every tool's `aeiou::load`); a JSON file that is no abstract, or a parameter file
    that is not one or not the abstract's, is one line and exit 1, never a traceback."""
    ast = WORKLOADS / CASE / ".." / ".." / "dcp_save.ast.json"
    for gpus in ("0", "-3"):
        assert catalog.main(["describe", "--gpus", gpus, str(tmp_path / "absent.ast.json"), "--no-shapes"]) == 2
        err = capsys.readouterr().err
        assert err.startswith(f"aeiou-catalog describe: --gpus {gpus}: must be at least 1\n\nUsage: aeiou-catalog describe "), err
        assert err.endswith("For more information, try 'aeiou-catalog describe --help'.\n")
    assert catalog.main(["describe", str(tmp_path / "absent.ast.json"), "--no-shapes"]) == 2
    assert "absent.ast.json: No such file or directory" in capsys.readouterr().err
    other = WORKLOADS / "checkpointing" / "dcp-load" / "cases" / "mixed-9-items" / "params.json"
    for name, text, params, message in (("list.json", "[]", None, "AST rejected"), ("empty.json", "{}", None, "AST rejected"),
                                        (None, None, "[]", "parameter file rejected"), (None, None, "{}", "parameter file rejected"),
                                        (None, None, other.read_text(encoding="utf-8"), "for abstract `dcp_load`, not `dcp_save`")):
        path = ast
        if name:
            path = tmp_path / name
            path.write_text(text, encoding="utf-8")
        argv = ["describe", str(path), "--no-shapes"]
        if params:
            (tmp_path / "p.json").write_text(params, encoding="utf-8")
            argv += ["--params-file", str(tmp_path / "p.json")]
        assert catalog.main(argv) == 1
        err = capsys.readouterr().err
        assert err.startswith("aeiou-catalog describe: ") and message in err and "Traceback" not in err, err


def _ascii_env(locale: str, utf8_mode: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LC_") and k not in ("LANG", "PYTHONIOENCODING", "PYTHONUTF8")}
    return {**env, "PYTHONPATH": str(BUILDER), "LC_ALL": locale, "PYTHONUTF8": utf8_mode}


def _catalog(args, locale: str, utf8_mode: str):
    import subprocess
    return subprocess.run([sys.executable, "-m", "aeiou.catalog", *map(str, args)], capture_output=True, env=_ascii_env(locale, utf8_mode))


def test_check_write_and_describe_under_an_ascii_locale(tree):
    """The records, CATALOG.md, and the descriptions are UTF-8 (`×`, `—`, `§`) whatever the
    locale: under `LC_ALL=C` without UTF-8 mode (ASCII) `check` reads the committed tree,
    `write` writes the same bytes, and `describe` prints the same bytes, as under C.UTF-8."""
    r = _catalog(["check", "--workloads", WORKLOADS, "--no-shapes"], "C", "0")
    assert r.returncode == 0, (r.stdout + r.stderr).decode("utf-8", "replace")
    out = {}
    for locale, mode in (("C.UTF-8", "1"), ("C", "0")):
        root = tree.parent / locale
        shutil.copytree(tree, root)
        r = _catalog(["write", "--workloads", root, "--no-shapes"], locale, mode)
        assert r.returncode == 0, (r.stdout + r.stderr).decode("utf-8", "replace")
        files = {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*.md"))}
        r = _catalog(["check", "--workloads", root, "--no-shapes"], locale, mode)
        assert r.returncode == 0, (r.stdout + r.stderr).decode("utf-8", "replace")
        case = root / CASE
        r = _catalog(["describe", case.parent.parent / "dcp_save.ast.json", "--params-file", case / "params.json", "--gpus", "2", "--no-shapes"], locale, mode)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        out[locale] = (files, r.stdout)
    files, described = out["C"]
    assert "±".encode("utf-8") in files[pathlib.Path("CATALOG.md")]
    files[pathlib.Path("CATALOG.md")].decode("utf-8")
    assert out["C"] == out["C.UTF-8"]


# ---------------------------------------------------------------- every refusal of the records

@pytest.mark.parametrize("extra, measures, message", [
    ("[[[\n", None, "attributes.toml: "),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'guessed'\n", None, "attribute bogus: source must be curated, derived, or measured, level mechanism or case"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'measured'\n", None, "attribute bogus: measured, but aeiou-catalog has no measure of that name"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'measured'\nclasses = [{min = 1}]\n", {"bogus"}, "attribute bogus: a measured attribute needs classes, each with a name"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'curated'\n", None, "attribute bogus: needs values"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'derived'\nmultiple = true\nvalues = {a = {}}\n", None, "attribute bogus: only a curated attribute takes several values"),
    ("[attribute.class.values.zzz]\ncategory = 'nowhere'\n", None, "class zzz: its category 'nowhere' is not a value of category"),
    ("", {"unheard"}, "aeiou-catalog measures unheard, which the dictionary does not define"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'curated'\nvalues = {a = {}}\nrequired = 'yes'\n", None, "attribute bogus: required is true, false, or a table of one curated attribute of its level and a value of it"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'curated'\nvalues = {a = {}}\nrequired = {kind = 'nope'}\n", None, "attribute bogus: required is true, false, or a table of one curated attribute of its level and a value of it"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'curated'\nvalues = {a = {}}\nrequired = {category = 'hpc'}\n", None, "attribute bogus: required is true, false, or a table of one curated attribute of its level and a value of it"),
    ("[attribute.bogus]\nlevel = 'case'\nsource = 'curated'\nvalues = {a = {}}\nrequired = {kind = 'use', modality = 'volume'}\n", None, "attribute bogus: required is true, false, or a table of one curated attribute of its level and a value of it"),
])
def test_the_dictionary_is_refused_by_name(tmp_path, monkeypatch, extra, measures, message):
    text = (WORKLOADS / "attributes.toml").read_text()
    (tmp_path / "attributes.toml").write_text(text + "\n" + extra)
    if measures:
        monkeypatch.setattr(catalog, "MEASURES", {**catalog.MEASURES, **{m: None for m in measures}})
    with pytest.raises(catalog.BuildError, match=re.escape(message)):
        catalog.load_dictionary(tmp_path)


def test_a_number_outside_every_class_and_a_path_outside_the_tree():
    assert catalog.classify({"classes": [{"name": "low", "max": 1}, {"name": "mid", "min": 2, "max": 3}]}, 1.5) is None
    assert catalog._rel(pathlib.Path("/elsewhere/x"), pathlib.Path("/workloads")) == "/elsewhere/x"
    # the byte-weighted median of one bucket, and of the last of several
    assert catalog._weighted_median({"buckets": [[4096, 3]]}) == 4096
    assert catalog._weighted_median({"buckets": [[1, 1], [2, 1], [4096, 1]]}) == 4096


MECH_DIR = "checkpointing/dcp-save"


def _rewrite(path, text):
    path.write_text(text)


@pytest.mark.parametrize("change, message", [
    # the records' files
    (lambda t: _rewrite(t / MECH, "= ="), "checkpointing/dcp-save/mechanism.toml: "),
    (lambda t: _rewrite(t / CASE / "case.toml", "[[run\n"), f"{CASE}/case.toml: "),
    (lambda t: (t / CASE / "case.toml").unlink(), f"{CASE}: a case without case.toml"),
    (lambda t: (t / CASE / "captures" / "2026-10-01" / "capture.toml").unlink(), f"{CASE}/captures/2026-10-01: a capture without capture.toml"),
    (lambda t: _rewrite(t / MECH_DIR / "dcp_save.ast.json", "{"), f"{MECH_DIR}/dcp_save.ast.json: "),
    (lambda t: ((t / "checkpointing" / "lone").mkdir(), _rewrite(t / "checkpointing" / "lone" / "mechanism.toml", "")),
     "checkpointing/lone/mechanism.toml: a mechanism.toml where no abstract is"),
    (lambda t: shutil.rmtree(t / "input-pipeline" / "map"), "input-pipeline/map: a class without its directory"),
    # mechanism.toml
    (lambda t: edit(t / MECH, 'category = "checkpointing"', 'category = "input-pipeline"\nclass = "shuffle"'),
     "class 'shuffle' is not a class of input-pipeline (map, stream)"),
    (lambda t: edit(t / MECH, 'domain = ["ai"]', 'domain = ["ai"]\n[defaults]\nast = "dcp_save"'),
     "[defaults] is for a mechanism with no case; this one has cases"),
    # case.toml
    (lambda t: edit(t / CASE / "case.toml", '[[run]]\nast = "dcp_save"\nparams = "params.json"', ''),
     "a case runs at least one AST: a [[run]] with ast and params"),
    (lambda t: edit(t / CASE / "case.toml", 'params = "params.json"', 'params = "params.json"\nseed = 3'),
     "[[run]]: unknown key(s) seed (allowed: ast, params)"),
    (lambda t: edit(t / CASE / "case.toml", 'params = "params.json"', 'params = "params.json"\n[[run]]\nast = "dcp_save"\nparams = "params.json"'),
     "two runs of one AST: a case runs each abstract once"),
    (lambda t: _rewrite(t / CASE / "params.json", "{"), f"{CASE}/params.json: "),
    (lambda t: edit(t / CASE / "params.json", '"abstract": "dcp_save"', '"abstract": "dcp_load"'),
     "its abstract is 'dcp_load', the case runs it as dcp_save"),
    # capture.toml
    (lambda t: edit(t / CASE / "captures" / "2026-10-01" / "capture.toml", 'raw_trace', 'kept = 1\nraw_trace'),
     "capture.toml: unknown key(s) kept (allowed: pair, raw_trace)"),
    (lambda t: edit(t / CASE / "captures" / "2026-10-01" / "capture.toml", 'verdict = "accepted"', 'verdict = "accepted"\nnote = "x"'),
     "[[pair]]: unknown key(s) note (allowed: ast, params, trace, verdict)"),
    (lambda t: edit(t / CASE / "captures" / "2026-10-01" / "capture.toml", 'ast = "dcp_save"', 'ast = "dcp_load"'),
     "[[pair]] ast 'dcp_load' is not one of the case's runs (dcp_save)"),
    (lambda t: edit(t / CASE / "captures" / "2026-10-01" / "capture.toml", 'ast = "dcp_save"', 'ast = "dcp_save"\nparams = "fitted.params.json"'),
     "[[pair]] params 'fitted.params.json' is not a file of the capture"),
    (lambda t: _rewrite(t / CASE / "captures" / "2026-10-01" / "other.tolerances.json", "{}"),
     f"{CASE}/captures/2026-10-01/other.tolerances.json: a tolerance file with no trace beside it"),
])
def test_every_record_refusal_names_its_file(tree, change, message):
    change(tree)
    found = problems(tree)
    assert any(p.startswith(message) or message in p for p in found), found
    with pytest.raises(catalog.BuildError, match=re.escape(message)):
        catalog.capture_pairs(tree)


def test_a_length_is_an_integer_parameter_with_a_default_or_required():
    """A case's `length` names an integer parameter of its abstract: one whose default is an
    integer, or a required one of kind `int` (contract 0.11), whose value the case gives (the
    LMCache abstracts' `requests`, required since 2026-10-23, DESIGN_REVIEW.md §3.78)."""
    from aeiou import catalog
    ast = {"params": {"n": {"default": 3}, "r": {"kind": "int"}, "f": {"kind": "float"}, "b": {"default": True},
                      "d": {"kind": "dist"}, "x": {"default": 1.5}}}
    assert catalog._length_ok(ast, "n") and catalog._length_ok(ast, "r")
    assert not any(catalog._length_ok(ast, k) for k in ("f", "b", "d", "x", "nope"))


@pytest.mark.parametrize("defaults, message", [
    ('ast = "dcp_load"', "[defaults] ast 'dcp_load' is not an abstract of this mechanism"),
    ('ast = "dcp_save"\nlength = "items_total"', "[defaults] length 'items_total' is not an integer parameter of dcp_save"),
    ('ast = "dcp_save"\ngpus = 0', "[defaults] gpus must be a positive integer"),
    ('ast = "dcp_save"\nseeds = 3', "[defaults]: unknown key(s) seeds (allowed: ast, gpus, length)"),
])
def test_the_defaults_of_a_mechanism_without_cases(tree, defaults, message):
    shutil.rmtree(tree / MECH_DIR / "cases")
    with open(tree / MECH, "a") as f:
        f.write(f"\n[defaults]\n{defaults}\n")
    assert any(message in p for p in problems(tree)), problems(tree)


def test_a_dry_run_that_fails_names_its_command(tmp_path):
    run = {"abstract": {"path": tmp_path / "x.ast.json", "doc": {"name": "x"}}, "params": None}
    with pytest.raises(catalog.BuildError, match=r"^false dry-run .*x\.ast\.json --gpus 2 --seed 1 --metrics-json .*: exit 1"):
        catalog._dry_run("false", run, 2, 1, None, tmp_path)


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_a_case_with_a_length_is_measured_down_to_its_check_length():
    """The halving of `shape`: the case's length, then shorter ones while the numbers hold;
    measured again here, the committed faiss shape entry, byte for byte (its check length is
    the case's: at half of it the numbers differ)."""
    mechanisms, _ = catalog.load(WORKLOADS, ATTRS)
    m = next(m for m in mechanisms if m["rel"] == str(pathlib.Path(FAISS).parent.parent))
    case = next(c for c in m["cases"] if c["name"] == pathlib.Path(FAISS).name)
    with catalog.Scheduler(len(catalog.SEEDS)) as pool:
        s = catalog.shape(str(RUNNER), case["runs"][0], case["gpus"][0], case["length"], ATTRS, pool)
    assert s == json.loads((WORKLOADS / FAISS / "shape.json").read_text())["runs"][0]


# ---------------------------------------------------------------- what write measures, and how many at once

OTHER = "checkpointing/dcp-save/cases/four-items"


def _fake_shape(monkeypatch, calls, tree=None):
    """`begin_shape` and `finish_shape` that record which run was measured and with what the
    shape committed before gave; with `tree`, the shape given is the committed one of that case
    (for the descriptions)."""
    def begin(runner, run, gpus, length, pool, previous=None):
        calls.setdefault("runs", []).append(str(run["params"]))
        calls.setdefault("previous", []).append(previous)
        return {"run": run}

    def finish(state, attrs):
        run = state["run"]
        if tree is not None:
            rel = pathlib.Path(run["params"]).parent.relative_to(tree)
            return json.loads((WORKLOADS / rel / "shape.json").read_text())["runs"][0]
        return {"ast": run["ast"], "fake": True}
    monkeypatch.setattr(catalog, "begin_shape", begin)
    monkeypatch.setattr(catalog, "finish_shape", finish)


def test_write_measures_only_the_stale_shapes(tree, monkeypatch, capsys):
    """`write` keeps a shape measured on the AST, parameter file, and case record committed now,
    byte for byte, and measures the others (here one whose parameter file's hash differs), from
    what the stale shape recorded; `--all-shapes` measures every one."""
    calls = {}
    _fake_shape(monkeypatch, calls)
    shape = tree / CASE / "shape.json"
    doc = json.loads(shape.read_text())
    doc["runs"][0]["params_sha256"] = "0" * 64
    shape.write_text(json.dumps(doc))
    kept = (tree / OTHER / "shape.json").read_text()
    mechanisms, _ = catalog.load(tree, ATTRS)
    shapes = catalog.measure_all(mechanisms, ATTRS, "true", every=False)
    assert [pathlib.Path(p).parent.name for p in calls["runs"]] == ["toy-6-items"]
    assert calls["previous"] == [doc["runs"][0]]
    assert shapes[tree / OTHER / "shape.json"] == kept and '"fake": true' in shapes[shape]
    calls.clear()
    assert len(catalog.measure_all(mechanisms, ATTRS, "true", every=True)) == 2 and len(calls["runs"]) == 2
    calls.clear()
    _fake_shape(monkeypatch, calls, tree=tree)
    assert catalog.main(["write", "--workloads", str(tree), "--runner", "true"]) == 0
    assert "1 of 2 shapes measured" in capsys.readouterr().out and len(calls["runs"]) == 1
    calls.clear()
    assert catalog.main(["write", "--workloads", str(tree), "--runner", "true", "--all-shapes"]) == 0
    assert "2 of 2 shapes measured" in capsys.readouterr().out and len(calls["runs"]) == 2


def test_the_scheduler_runs_jobs_at_once_the_most_expensive_first():
    """`Scheduler(jobs)` runs at most `jobs` calls at once, from one queue: the highest cost first
    (an unknown cost, infinity, before any), in order of submission within one cost; nothing
    starts while it is held, so every case's first dry runs are queued before one is chosen."""
    import threading
    import time
    lock, now, peak, order = threading.Lock(), [0], [0], []

    def job(name):
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        time.sleep(0.05)
        with lock:
            now[0] -= 1
            order.append(name)
        return name
    with catalog.Scheduler(1) as pool:
        with pool.held():
            fs = [pool.submit(1, job, "cheap"), pool.submit(5, job, "dear"), pool.submit(5, job, "dear2"),
                  pool.submit(math.inf, job, "new"), pool.submit(3, job, "mid")]
            time.sleep(0.1)
            assert order == []
        assert [f.result() for f in fs] == ["cheap", "dear", "dear2", "new", "mid"]
    assert order == ["new", "dear", "dear2", "mid", "cheap"] and peak[0] == 1
    peak[0] = 0
    with catalog.Scheduler(3) as pool:
        fs = [pool.submit(0, job, i) for i in range(9)]
    assert peak[0] == 3 and all(f.done() for f in fs)

    def boom():
        raise ValueError("no")
    with catalog.Scheduler(1) as pool:
        f = pool.submit(0, boom)
        with pytest.raises(ValueError):
            f.result()


def test_a_failure_drops_the_dry_runs_not_begun():
    """Leaving the scheduler on an error cancels what is still queued, so a failed write waits
    on the dry runs running and on no others."""
    import time
    ran = []
    with pytest.raises(RuntimeError):
        with catalog.Scheduler(1) as pool:
            with pool.held():
                fs = [pool.submit(0, lambda i=i: (time.sleep(0.05), ran.append(i))) for i in range(5)]
            time.sleep(0.02)
            raise RuntimeError("stop")
    assert ran == [0] and all(f.cancelled() for f in fs[1:])


def _search(monkeypatch, tmp_path, holds, jobs=8, previous=None, full=640):
    """`shape` over a fake dry run whose numbers are the case's at any length >= `holds`; the
    events, in order: ("start" | "end", length) per seed's dry run, and the shape."""
    import threading
    import time
    events, lock = [], threading.Lock()

    def dry(runner, run, gpus, seed, length, tmp):
        with lock:
            events.append(("start", length[1]))
        time.sleep(0.01)
        with lock:
            events.append(("end", length[1]))
        n = length[1]
        return {"total": {"ops": n, "counts": {"read": n}, "bytes_read": n, "bytes_written": 0}, "datasets": []}
    monkeypatch.setattr(catalog, "_dry_run", dry)
    monkeypatch.setattr(catalog, "measure", lambda total: {m: (1.0 if total["ops"] >= holds else 9.0) for m in catalog.MEASURES})
    run = {"abstract": {"doc": {"name": "x"}, "path": tmp_path / "x.ast.json", "sha256": "0"}, "params": None, "ast": "x"}
    monkeypatch.setattr(catalog, "_values", lambda run: {})
    monkeypatch.setattr(catalog, "_params_of", lambda doc: {"calls": {"default": full}})
    with catalog.Scheduler(jobs) as pool:
        s = catalog.shape("true", run, 1, "calls", ATTRS, pool, previous)
    return events, s


def _tried(events):
    """The lengths in the order their first dry run started."""
    out = []
    for kind, n in events:
        if kind == "start" and n not in out:
            out.append(n)
    return out


@pytest.mark.parametrize("previous,tried,check", [
    (None, [640, 10, 20, 40, 80, 160], 160),            # a new case: smallest first, then up
    (160, [640, 160, 80], 160),                         # its recorded check length holds: one smaller
    (40, [640, 40, 80, 160], 160),                      # it no longer holds: up
    (640, [640, 320, 160, 80], 160),                    # the case's own: half of it first
    (320, [640, 320, 160, 80], 160),                    # holds: down while it holds
    (7, [640, 10, 20, 40, 80, 160], 160),               # a length not among the levels: a new case
])
def test_the_search_starts_from_the_recorded_check_length(monkeypatch, tmp_path, previous, tried, check):
    """The check length is the smallest level (1/2 to 1/64 of the case's) whose numbers agree
    with the case's own length's, each level judged on its three seeds against the full run
    (never against another level); the search starts at the length the shape recorded before
    (down while a level agrees, up while it differs), and a new case starts at the smallest."""
    recorded = None if previous is None else {"ops": 1, "length": {"param": "calls", "case": 640, "check": previous}}
    events, s = _search(monkeypatch, tmp_path, holds=160, previous=recorded)
    assert _tried(events) == tried
    assert s["length"] == {"param": "calls", "case": 640, "check": check}
    assert [n for kind, n in events if kind == "start"].count(640) == 3


def test_no_level_starts_before_its_cases_previous_decision(monkeypatch, tmp_path):
    """Only the first level tried runs beside the case's own length; every later one starts
    after the dry runs that decided the one before it (and the full run) have ended: no
    speculation, so no dry run is wasted however many jobs there are."""
    events, _ = _search(monkeypatch, tmp_path, holds=160, jobs=32)
    tried = _tried(events)
    for before, n in zip(tried[1:], tried[2:]):
        first = events.index(("start", n))
        assert events[:first].count(("end", before)) == 3 and events[:first].count(("end", 640)) == 3


def test_the_check_length_is_the_same_at_any_jobs(monkeypatch, tmp_path):
    for holds in (1, 10, 80, 160, 320, 640):
        found = {_search(monkeypatch, tmp_path, holds=holds, jobs=jobs)[1]["length"]["check"] for jobs in (1, 2, 32)}
        assert found == {max(10, holds)}, (holds, found)


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_jobs_bound_the_dry_runs_at_once_and_write_the_same(tree, monkeypatch):
    """`--jobs N` runs at most N dry runs at once, over every shape being measured; the shapes
    are the ones a single job measures, in the same files."""
    import threading
    import time
    real, lock, now, peak = catalog._dry_run, threading.Lock(), [0], [0]

    def dry(*args):
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        time.sleep(0.05)
        try:
            return real(*args)
        finally:
            with lock:
                now[0] -= 1
    monkeypatch.setattr(catalog, "_dry_run", dry)
    mechanisms, _ = catalog.load(tree, ATTRS)
    one = catalog.measure_all(mechanisms, ATTRS, str(RUNNER), jobs=1)
    assert peak[0] == 1
    peak[0] = 0
    four = catalog.measure_all(mechanisms, ATTRS, str(RUNNER), jobs=4)
    assert peak[0] == 4 and four == one and list(four) == list(one)


@pytest.mark.skipif(not RUNNER.exists(), reason="needs the runner binary")
def test_the_most_expensive_case_is_measured_first(tree, monkeypatch):
    """The dry runs are chosen by the op count the case's shape recorded, the largest first:
    here four-items (325 ops) before toy-6-items (154), whichever the tree lists first, and
    toy-6-items first once its recorded count is the larger."""
    real, order = catalog._dry_run, []

    def dry(runner, run, *args):
        order.append(pathlib.Path(run["params"]).parent.name)
        return real(runner, run, *args)
    monkeypatch.setattr(catalog, "_dry_run", dry)
    mechanisms, _ = catalog.load(tree, ATTRS)
    catalog.measure_all(mechanisms, ATTRS, str(RUNNER), jobs=1)
    assert order == ["four-items"] * 3 + ["toy-6-items"] * 3
    shape = tree / CASE / "shape.json"
    doc = json.loads(shape.read_text())
    doc["runs"][0]["ops"] = 10 ** 9
    shape.write_text(json.dumps(doc))
    order.clear()
    catalog.measure_all(mechanisms, ATTRS, str(RUNNER), jobs=1)
    assert order == ["toy-6-items"] * 3 + ["four-items"] * 3
    order.clear()
    catalog.stale_shapes(mechanisms, tree, ATTRS, str(RUNNER), jobs=1)
    assert order[:3] == ["toy-6-items"] * 3


def test_jobs_default_to_the_cpus_this_process_may_use_and_refuse_zero(monkeypatch, capsys):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(20)))
    assert catalog.parser()[0].parse_args(["write"]).jobs == 20
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3})
    assert catalog.parser()[0].parse_args(["check"]).jobs == 1
    assert catalog.main(["check", "--jobs", "0", "--no-shapes"]) == 2
    assert "--jobs 0: must be at least 1" in capsys.readouterr().err


def test_a_search_starts_new_when_the_old_shape_is_unreadable_and_nothing_stale_runs_nothing(tree, tmp_path):
    """What a stale shape recorded is where its search starts; a shape file that is not there or
    not a shape gives nothing, so its case searches as a new one. With no shape stale, `write`
    runs no dry run (the runner is never called: here it is one that fails)."""
    assert catalog._previous(tmp_path / "absent.json") == {}
    for text in ("not json", "[]", '{"runs": [{"no": "ast"}]}'):
        (tmp_path / "s.json").write_text(text)
        assert catalog._previous(tmp_path / "s.json") == {}
    mechanisms, _ = catalog.load(tree, ATTRS)
    shapes = catalog.measure_all(mechanisms, ATTRS, "false", every=False)
    assert shapes[tree / CASE / "shape.json"] == (WORKLOADS / CASE / "shape.json").read_text()


# ---------------------------------------------------------------- catalog.json, what `aeiou cases` reads

def test_the_index_holds_every_case_its_attributes_and_the_vocabulary():
    """`catalog.json` (item 40, DESIGN_REVIEW §3.80): every case a lock names, in the tree's
    order, with its curated attributes (the mechanism's and its own), its GPU counts, its lock
    and description, and per run what it does (the AST's doc), what it is (its parameter files'
    docs), and its derived attributes and measured classes; and the vocabulary, every attribute
    with its values (a measured one's classes), in the dictionary's order."""
    mechanisms, problems = catalog.load(WORKLOADS, ATTRS)
    assert problems == []
    shapes = {catalog._shape_path(m, c): catalog._shape_path(m, c).read_text() for m, c, _ in catalog._targets(mechanisms)}
    doc = json.loads(catalog.render_index(mechanisms, ATTRS, shapes))
    assert (doc["catalog_version"], doc["tool"]) == (1, "aeiou-catalog")
    assert [v["name"] for v in doc["attributes"]] == list(ATTRS)
    for v, (name, a) in zip(doc["attributes"], ATTRS.items()):
        assert (v["source"], v["level"], v["multiple"]) == (a["source"], a["level"], bool(a.get("multiple")))
        assert v["values"] == ([c["name"] for c in a["classes"]] if a["source"] == "measured" else list(a["values"]))
    locks = sorted(WORKLOADS.rglob("case.lock.json"))
    assert sorted(c["lock"] for c in doc["cases"]) == sorted(str(p.relative_to(WORKLOADS)) for p in locks)
    assert [c["name"] for c in doc["cases"]] == [json.loads((WORKLOADS / c["lock"]).read_text())["name"] for c in doc["cases"]]
    faiss = next(c for c in doc["cases"] if c["name"] == "faiss-ivf-ondisk/sift1m-ivf1024-pq32")
    assert faiss["dir"] == FAISS and faiss["description"] == f"{FAISS}/DESCRIPTION.md" and faiss["gpus"] == [1]
    assert {k: faiss["attributes"][k] for k in ("category", "application", "kind", "modality")} == {
        "category": "vector-search", "application": "faiss", "kind": "example", "modality": "vectors"}
    (run,) = faiss["runs"]
    ast = json.loads((WORKLOADS / "vector-search/faiss-ivf-ondisk/faiss_ivf_ondisk.ast.json").read_text())
    pset = json.loads((WORKLOADS / FAISS / "params.json").read_text())
    shape = json.loads((WORKLOADS / FAISS / "shape.json").read_text())["runs"][0]
    assert (run["ast"], run["does"], run["is"]) == ("faiss_ivf_ondisk", ast["doc"], pset["doc"])
    assert {k: run["attributes"][k] for k in ("api", "writes", "inputs")} == {"api": "mmap", "writes": "no", "inputs": "no"}
    assert run["attributes"]["status"] in ATTRS["status"]["values"]
    for name, a in ATTRS.items():
        if a["source"] == "measured" and shape["attributes"][name]["class"] is not None:
            assert run["attributes"][name] == shape["attributes"][name]["class"], name
    # a case of two runs: each its own derived attributes; fragments' docs joined in order
    shared = next(c for c in doc["cases"] if c["name"] == "lmcache-shared-fs/sharegpt-chat")
    assert [(r["ast"], r["attributes"]["writes"], r["attributes"]["inputs"]) for r in shared["runs"]] == [
        ("lmcache_shared_fs", "yes", "no"), ("lmcache_shared_fs_reader", "no", "yes")]
    unet = next(c for c in doc["cases"] if c["name"] == "npy-per-sample/unet3d-h100")
    frags = [json.loads((WORKLOADS / unet["dir"] / f).read_text())["doc"] for f in ("../../jobs/unet3d.params.json", "h100.params.json")]
    assert unet["runs"][0]["is"] == " ".join(frags) and unet["attributes"]["accelerator"] == "h100"


def test_the_index_is_written_and_checked_as_catalog_md_is(tree, capsys):
    """`write` writes catalog.json beside CATALOG.md and `check` fails when it differs from what
    the records give now; a mechanism with no case is in no row (no lock to run)."""
    assert catalog.main(["write", "--workloads", str(tree), "--no-shapes"]) == 0
    index = tree / "catalog.json"
    assert "wrote catalog.json" in capsys.readouterr().out
    doc = json.loads(index.read_text())
    assert sorted(c["name"] for c in doc["cases"]) == ["dcp-save/four-items", "dcp-save/toy-6-items"]
    assert catalog.main(["check", "--workloads", str(tree), "--no-shapes"]) == 0
    capsys.readouterr()
    doc["cases"].pop()
    index.write_text(json.dumps(doc))
    assert catalog.main(["check", "--workloads", str(tree), "--no-shapes"]) == 1
    assert "catalog.json: differs from what aeiou-catalog derives now (aeiou-catalog write)" in capsys.readouterr().out
