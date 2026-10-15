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
    shapes = catalog.measure_all(mechanisms, ATTRS, str(RUNNER))
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
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(len(catalog.SEEDS)) as pool:
        differ = catalog.recheck(str(RUNNER), run, case["gpus"][0], case["block"], recorded, ATTRS, pool)
        assert not set(differ) & set(catalog.COUNTED), differ
        recorded["datasets"][0]["directories"] += 7
        differ = catalog.recheck(str(RUNNER), run, case["gpus"][0], case["block"], recorded, ATTRS, pool)
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
    return {"ast": "0.10", "name": "every_kind", "doc": "Every kind.",
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
