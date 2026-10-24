"""Use cases, benchmark rules, and the lock (contract 0.11, DESIGN_REVIEW.md §3.78): the
constraint parser, a use case's fragments combined by agreement, the rules' freeze by state,
`aeiou-params lock`, a stale or missing lock, `aeiou-build` re-locking the cases of what it
rebuilt, and every committed case's lock current."""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
WORKLOADS = ROOT / "workloads"
sys.path.insert(0, str(BUILDER))

from aeiou import cli, lock  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402

SCRIPT = '''from aeiou import *
w = Workload("toy")
w.param("batch", kind=int, unit="count")
w.param("step_time", kind=int, unit="ns")
w.param("workers", 2, unit="count")
w.param("samples", 64, unit="count", scale=True)
w.param("hdr_read", st_blksize(of="train", fallback=1 * MiB), unit="bytes")
train = w.dataset("train", pattern="train/{id:05}", layout="whole", access="map", count=w.P.samples, size=const(4096), seed=1)
with w.actor("gpu") as g:
    with g.loop("s", 2):
        with g.loop("j", w.P.batch):
            f = g.let("f", train.consume())
            g.open(f, "RDONLY"); g.read(f, w.P.hdr_read, repeat="until_eof"); g.close(f)
        g.compute(w.P.step_time)
if __name__ == "__main__":
    w.write()
'''


def _tree(tmp_path, fragments: dict, runs=None) -> pathlib.Path:
    """A workloads/ tree with the toy mechanism built and a case `job-h100` of `fragments`."""
    mech = tmp_path / "workloads" / "input-pipeline" / "toy"
    case = mech / "cases" / "job-h100"
    case.mkdir(parents=True)
    (mech / "toy.py").write_text(SCRIPT, encoding="utf-8")
    assert cli.main([str(mech / "toy.py")]) == 0
    for name, body in fragments.items():
        (case / name).write_text(json.dumps({"params_version": 2, "abstract": "toy", **body}), encoding="utf-8")
    runs = runs or [("toy", list(fragments))]
    (case / "case.toml").write_text("kind = \"use-case\"\n" + "".join(
        f"\n[[run]]\nast = \"{a}\"\nparams = {json.dumps(fs)}\n" for a, fs in runs), encoding="utf-8")
    return case


def _lock(case, **kw):
    return lock.make_lock(case, workloads=case.parents[3], **kw)


def test_a_constraint_is_parsed_into_the_ast_expression_form():
    expr, names = lock.parse_constraint("fact.dataset_bytes >= 5 * fact.sum_host_mem")
    assert expr == {"ge": [{"fact": "dataset_bytes"}, {"mul": [5, {"fact": "sum_host_mem"}]}]}
    assert names == ["fact.dataset_bytes", "fact.sum_host_mem"]
    expr, names = lock.parse_constraint("not (result.min_gpu_busy_fraction < 0.9) and fact.min_samples_per_gpu >= 2_000 or -1 == min(fact.a div 2, fact.b mod 3) - 4")
    assert expr == {"or": [{"and": [{"not": {"lt": [{"result": "min_gpu_busy_fraction"}, 0.9]}}, {"ge": [{"fact": "min_samples_per_gpu"}, 2000]}]},
                           {"eq": [{"neg": 1}, {"sub": [{"min": [{"div": [{"fact": "a"}, 2]}, {"mod": [{"fact": "b"}, 3]}]}, 4]}]}]}
    assert names == ["fact.a", "fact.b", "fact.min_samples_per_gpu", "result.min_gpu_busy_fraction"]
    for bad, needle in [("samples >= 4", "`samples` is not a number, a `fact.` or `result.` name"),
                        ("params.batch >= 4", "`params.batch` is not"),
                        ("fact.a <= fact.b <= fact.c", "one comparison at a time"),
                        ("fact.a >= ", "expected more, got the end"),
                        ("fact.a >= 1 )", "unexpected `)`"),
                        ("fact.a ^ 2", "cannot read `^ 2`")]:
        with pytest.raises(BuildError, match=re.escape(needle)):
            lock.parse_constraint(bad)


def test_a_use_case_resolves_by_agreement_into_a_lock(tmp_path):
    case = _tree(tmp_path, {"job.params.json": {"params": {"workers": {"locked": 4}, "batch": {"default": 7}}},
                            "h100.params.json": {"params": {"batch": {"default": 7}, "step_time": {"locked": 323000000}},
                                                 "run": {"posix": {"default": "sync"}}}})
    doc = _lock(case)
    assert doc["name"] == "toy/job-h100"
    run, = doc["runs"]
    assert (run["name"], run["abstract"]) == ("toy", "input-pipeline/toy/toy.ast.json")
    assert run["params"] == {"batch": {"default": 7}, "hdr_read": {"default": {"from": "st_blksize", "of": "train", "fallback": 1048576}},
                             "samples": {"default": 64}, "step_time": {"locked": 323000000}, "workers": {"locked": 4}}
    assert run["sources"]["batch"] in ("input-pipeline/toy/cases/job-h100/job.params.json", "input-pipeline/toy/cases/job-h100/h100.params.json")
    assert run["sources"]["hdr_read"] == "abstract" and run["sources"]["step_time"].endswith("h100.params.json")
    assert (run["run"], run["run_sources"]) == ({"posix": {"default": "sync"}}, {"posix": "input-pipeline/toy/cases/job-h100/h100.params.json"})
    assert doc["provenance"]["case"] == "input-pipeline/toy/cases/job-h100/case.toml"
    assert [f["path"] for f in doc["provenance"]["fragments"]] == ["input-pipeline/toy/cases/job-h100/h100.params.json", "input-pipeline/toy/cases/job-h100/job.params.json"]
    text = lock.render(doc)
    assert json.loads(text) == doc and '        "batch": {"default": 7},\n' in text, "one value to a line"
    # a parameter the use case leaves required stays required
    case2 = _tree(tmp_path / "b", {"job.params.json": {"params": {"batch": {"default": 7}}}})
    assert _lock(case2)["runs"][0]["params"]["step_time"] == {"required": True}


def test_fragments_that_disagree_are_refused_naming_both(tmp_path):
    case = _tree(tmp_path, {"a.params.json": {"params": {"batch": {"default": 7}}}, "b.params.json": {"params": {"batch": {"default": 8}}}})
    with pytest.raises(BuildError, match=r"params `batch`: input-pipeline/toy/cases/job-h100/a.params.json sets \{\"default\": 7\}, "
                                         r"input-pipeline/toy/cases/job-h100/b.params.json sets \{\"default\": 8\}; fragments of one use case agree"):
        _lock(case)
    case = _tree(tmp_path / "s", {"a.params.json": {"params": {"batch": {"default": 7}}}, "b.params.json": {"params": {"batch": {"locked": 7}}}})
    with pytest.raises(BuildError, match="sets \\{\"default\": 7\\}.*sets \\{\"locked\": 7\\}"):
        _lock(case)
    # a fragment's value is held to the abstract's kind, as `aeiou-params check` holds it
    case = _tree(tmp_path / "k", {"a.params.json": {"params": {"batch": {"default": 2.5}}}})
    with pytest.raises(BuildError, match="batch: the kind is an integer, the value 2.5 is not"):
        _lock(case)


def test_the_rules_freeze_by_state_and_copy_their_constraints_in(tmp_path):
    case = _tree(tmp_path, {"h100.params.json": {"params": {"batch": {"default": 7}, "step_time": {"default": 323000000}},
                                                 "run": {"posix": {"default": "sync"}}}})
    rules = tmp_path / "bench.rules.json"
    rules.write_text(json.dumps({"rules_version": 1, "name": "bench-2026", "doc": "a benchmark",
                                 "freeze": {"defaults": "locked", "scale": "required", "identity_options": "locked"},
                                 "constraints": [{"expr": "fact.dataset_bytes >= 5 * fact.sum_host_mem", "doc": "5x DRAM"}]}))
    doc = _lock(case, rules=rules)
    run, = doc["runs"]
    assert run["params"]["batch"] == {"locked": 7} and run["params"]["workers"] == {"locked": 2}
    assert run["params"]["hdr_read"] == {"locked": {"from": "st_blksize", "of": "train", "fallback": 1048576}}, "the rule is locked, not a number"
    assert run["params"]["samples"] == {"required": True} and run["sources"]["samples"] == "rules"
    assert run["run"] == {"posix": {"locked": "sync"}}
    assert doc["freeze"] == {"defaults": "locked", "scale": "required", "identity_options": "locked"} and doc["doc"] == "a benchmark"
    assert doc["constraints"] == [{"text": "fact.dataset_bytes >= 5 * fact.sum_host_mem", "names": ["fact.dataset_bytes", "fact.sum_host_mem"], "doc": "5x DRAM",
                                   "expr": {"ge": [{"fact": "dataset_bytes"}, {"mul": [5, {"fact": "sum_host_mem"}]}]}}]
    assert doc["provenance"]["rules"]["path"] == "bench.rules.json", "a file outside workloads/ by its name, so the lock is the same on every site"
    # a scale parameter the use case locked cannot be made required
    case = _tree(tmp_path / "s", {"h.params.json": {"params": {"samples": {"locked": 64}}}})
    with pytest.raises(BuildError, match="a `scale` parameter, which the rules make required, is locked by"):
        _lock(case, rules=rules)
    for body, needle in [({"rules_version": 1, "params": {}}, "unknown key `params`"),
                         ({"rules_version": 1, "freeze": {"defaults": "required"}}, 'freeze: `defaults: "required"` is not one of'),
                         ({"rules_version": 1, "constraints": [{"expr": "batch > 1"}]}, "`batch` is not a number"),
                         ({"rules_version": 2}, "not a rules file")]:
        rules.write_text(json.dumps(body))
        with pytest.raises(BuildError, match=needle):
            lock.load_rules(rules)


def test_aeiou_params_lock_writes_beside_the_case_and_a_stale_lock_is_found(tmp_path):
    case = _tree(tmp_path, {"h100.params.json": {"params": {"batch": {"default": 7}}}})
    wl = case.parents[3]
    assert lock.stale(case, workloads=wl).startswith("no case.lock.json: make it with `aeiou-params lock")
    env = {**os.environ, "PYTHONPATH": str(BUILDER)}
    r = subprocess.run([sys.executable, "-m", "aeiou.params", "lock", str(case)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "toy/job-h100  1 run(s)" in r.stdout and "required: step_time" in r.stdout
    assert lock.stale(case, workloads=wl) is None
    (case / "h100.params.json").write_text(json.dumps({"params_version": 2, "abstract": "toy", "params": {"batch": {"default": 8}}}))
    assert "case.lock.json is stale" in lock.stale(case, workloads=wl)
    r = subprocess.run([sys.executable, "-m", "aeiou.params", "lock"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "<CASE>" in r.stderr


def test_aeiou_build_relocks_the_cases_of_what_it_rebuilt(tmp_path, capsys):
    case = _tree(tmp_path, {"h100.params.json": {"params": {"batch": {"default": 7}}}})
    wl = case.parents[3]
    (case / lock.LOCK_FILE).write_text(lock.render(_lock(case)))
    before = json.loads((case / lock.LOCK_FILE).read_text())["runs"][0]["ast_sha256"]
    script = case.parents[1] / "toy.py"
    script.write_text(SCRIPT.replace('w.param("workers", 2,', 'w.param("workers", 3,'))
    capsys.readouterr()
    assert cli.main([str(script)]) == 0
    assert f"relocked {case / lock.LOCK_FILE}" in capsys.readouterr().out
    after = json.loads((case / lock.LOCK_FILE).read_text())["runs"][0]
    assert after["ast_sha256"] != before and after["params"]["workers"] == {"default": 3}
    assert lock.stale(case, workloads=wl) is None
    # a case with no lock is its author's to lock: the build leaves it alone
    (case / lock.LOCK_FILE).unlink()
    assert cli.main([str(script)]) == 0 and not (case / lock.LOCK_FILE).exists()


def test_every_committed_case_has_a_current_lock():
    cases = sorted(p.parent for p in WORKLOADS.rglob("case.toml"))
    assert len(cases) >= 16
    problems = [f"{c.relative_to(WORKLOADS)}: {why}" for c in cases if (why := lock.stale(c, workloads=WORKLOADS))]
    assert not problems, "\n".join(problems)


def test_every_lock_fits_its_schema():
    import jsonschema
    from referencing import Registry, Resource
    reg = Registry()
    for n in ("abstract-ast", "rules"):
        reg = reg.with_resource(f"{n}.schema.json", Resource.from_contents(json.loads((ROOT / "schema" / f"{n}.schema.json").read_text())))
    v = jsonschema.Draft202012Validator(json.loads((ROOT / "schema" / "lock.schema.json").read_text()), registry=reg)
    for f in sorted(WORKLOADS.rglob(lock.LOCK_FILE)):
        assert not list(v.iter_errors(json.loads(f.read_text()))), f


# ---------------------------------------------------------------- npy_per_sample and its use case

NPY = WORKLOADS / "input-pipeline/map/npy-per-sample"
# the Unet3D job: the WG's reference workload (its DLIO configs unet3d_{a100,h100,b200}.yaml, sizes
# provisional, 2026-10-23), the job values shared and one fragment per accelerator (item 39)
UNET3D_JOB = {"workers": 4, "prefetch": 2, "steps": 500, "samples": 50_000,
              "sample_mean": 146_600_628, "sample_sd": 68_341_808}
UNET3D_GPU = {"a100": {"batch": 7, "step_time": 636_000_000}, "h100": {"batch": 7, "step_time": 323_000_000},
              "b200": {"batch": 7, "step_time": 162_000_000}}
UNET3D_H100 = UNET3D_JOB | UNET3D_GPU["h100"]


def test_npy_per_sample_leaves_the_job_and_the_gpu_to_its_use_case():
    """DESIGN_REVIEW.md §3.78 points 1, 4, and 11: what the job and the model on a GPU set has no
    default (a use case gives it), `samples` is the submission's scale, and the reads Python's
    BufferedReader sizes are st_blksize, which the storage reports."""
    p = json.loads((NPY / "npy_per_sample.ast.json").read_text())["params"]
    for k in UNET3D_H100:
        assert "default" not in p[k] and p[k]["kind"] == "int", k
    assert p["samples"].get("scale") is True
    for k in ("hdr_read", "xfer", "inv_read"):
        assert p[k]["default"] == {"from": "st_blksize", "of": "train", "fallback": 1 << 20}, k


def test_unet3d_is_a_use_case_per_accelerator_sharing_one_job_fragment():
    """Item 39 (DESIGN_REVIEW.md §3.81): the Unet3D job is one use case per accelerator, each the
    mechanism's job fragment and its accelerator's (batch, step_time), the WG's reference values;
    B200's step time is the WG's H100 halved, not a measurement, and says so."""
    import tomllib
    job = json.loads((NPY / "jobs/unet3d.params.json").read_text())
    assert job["params"] == {k: {"default": v} for k, v in UNET3D_JOB.items()}
    assert "provisional" in job["doc"]
    vocabulary = tomllib.loads((WORKLOADS / "attributes.toml").read_text())["attribute"]["accelerator"]["values"]
    for acc, values in UNET3D_GPU.items():
        case = NPY / f"cases/unet3d-{acc}"
        rec = tomllib.loads((case / "case.toml").read_text())
        assert (rec["kind"], rec["accelerator"]) == ("use", acc) and acc in vocabulary
        assert rec["run"] == [{"ast": "npy_per_sample", "params": ["../../jobs/unet3d.params.json", f"{acc}.params.json"]}]
        gpu = json.loads((case / f"{acc}.params.json").read_text())
        assert gpu["params"] == {k: {"default": v} for k, v in values.items()}, "the model on a GPU (§3.78 point 1)"
        assert f"unet3d_{acc}.yaml" in gpu["doc"], acc
        (run,) = json.loads((case / lock.LOCK_FILE).read_text())["runs"]
        assert {k: run["params"][k] for k in UNET3D_JOB | values} == {k: {"default": v} for k, v in (UNET3D_JOB | values).items()}
        assert run["sources"]["sample_sd"] == "input-pipeline/map/npy-per-sample/jobs/unet3d.params.json"
        assert lock.stale(case, workloads=WORKLOADS) is None
    assert "not measured" in json.loads((NPY / "cases/unet3d-b200/b200.params.json").read_text())["doc"]


def test_the_example_cases_of_npy_per_sample_give_every_required_value():
    """The traced examples give every value the abstract requires, `step_time` the old default
    and said so (never fitted: the trace's compute is the stand-in's)."""
    required = {k for k, p in json.loads((NPY / "npy_per_sample.ast.json").read_text())["params"].items() if "default" not in p}
    for case in ("npy-8mib", "npy-140mib"):
        doc = json.loads((NPY / "cases" / case / "params.json").read_text())
        assert required <= set(doc["params"]), case
        assert doc["params"]["step_time"] == {"default": 323_000_000} and "not fitted" in doc["doc"], case


# ---------------------------------------------------------------- the untraced containers and their cases

# what each abstract's defaults held until 2026-10-22, placeholders never measured: a case of kind
# `unsettled` holds them, untraced (item 33, part 5b; DESIGN_REVIEW.md §3.78 points 1 and 4)
STREAM_DEFAULTS = {"batch": 256, "samples_per_file": 1024, "cycle": 4, "prefetch": 1, "steps": 500,
                   "step_time": 105_000_000, "decode": 0, "samples": 1_281_167, "sample_median": 110 << 10}
UNTRACED = {
    "hdf5_map": (WORKLOADS / "input-pipeline/map/hdf5", "rgb256-rows",
                 {"batch": 32, "workers": 8, "prefetch": 2, "steps": 500, "step_time": 105_000_000,
                  "samples_per_file": 512, "samples": 1_000_000}),
    "parquet_stream": (WORKLOADS / "input-pipeline/stream/parquet", "imagenet-rows", STREAM_DEFAULTS),
    "tfrecord_stream": (WORKLOADS / "input-pipeline/stream/tfrecord", "imagenet-records", STREAM_DEFAULTS),
}


@pytest.mark.parametrize("ast", sorted(UNTRACED))
def test_the_untraced_containers_leave_the_job_and_the_gpu_to_a_case(ast):
    """DESIGN_REVIEW.md §3.78 points 1 and 4: the job's values and the model's on a GPU have no
    default, and `samples` is the submission's scale; the library's values keep theirs."""
    mech, _, old = UNTRACED[ast]
    p = json.loads((mech / f"{ast}.ast.json").read_text())["params"]
    for k in old:
        assert "default" not in p[k] and p[k]["kind"] == "int", k
    assert p["samples"].get("scale") is True
    assert {k for k, v in p.items() if "default" not in v} == set(old)


@pytest.mark.parametrize("ast", sorted(UNTRACED))
def test_an_untraced_container_has_an_unsettled_case_of_its_old_defaults(ast):
    """Each mechanism's one case holds the values its defaults held, each a default, said to be
    untraced: kind `unsettled`, no capture, a lock that is current, and no `[defaults]` row left."""
    import tomllib
    mech, name, old = UNTRACED[ast]
    case = mech / "cases" / name
    rec = tomllib.loads((case / "case.toml").read_text())
    assert rec["kind"] == "unsettled" and "accelerator" not in rec
    assert rec["run"] == [{"ast": ast, "params": "params.json"}]
    assert not (case / "captures").exists()
    doc = json.loads((case / "params.json").read_text())
    assert "untraced" in doc["doc"] and "not measured" in doc["doc"]
    assert doc["params"] == {k: {"default": v} for k, v in old.items()}
    (run,) = json.loads((case / lock.LOCK_FILE).read_text())["runs"]
    assert {k: run["params"][k] for k in old} == {k: {"default": v} for k, v in old.items()}
    assert lock.stale(case, workloads=WORKLOADS) is None
    assert "untraced" in (case / "README.md").read_text()
    assert "defaults" not in tomllib.loads((mech / "mechanism.toml").read_text())
    assert not (mech / "shape.defaults.json").exists() and not (mech / "DESCRIPTION.defaults.md").exists()


def test_the_release_tarball_carries_every_lock_and_what_it_pins():
    """The release workflow's file filter (`.github/workflows/release.yml`) keeps every case's
    lock and the abstract each lock pins, so `aeiou run --case` works in the unpacked tarball."""
    m = re.search(r"git ls-files workloads \| grep -E '([^']+)'", (ROOT / ".github/workflows/release.yml").read_text())
    keep = re.compile(m.group(1))
    files = subprocess.run(["git", "ls-files", "workloads"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    shipped = {f for f in files if keep.search(f)}
    locks = sorted(f for f in files if f.endswith("/" + lock.LOCK_FILE))
    assert locks and set(locks) <= shipped, sorted(set(locks) - shipped)
    for f in locks:
        for run in json.loads((ROOT / f).read_text())["runs"]:
            assert "workloads/" + run["abstract"] in shipped, (f, run["abstract"])


# ---------------------------------------------------------------- file_per_sample and its case

FPS = WORKLOADS / "input-pipeline/map/file-per-sample"
# what file_per_sample's defaults held until 2026-10-22, placeholders never measured as a job: a
# case of kind `unsettled` holds them (item 33, part 5c; DESIGN_REVIEW.md §3.78 points 1 and 4)
FPS_DEFAULTS = {"batch": 32, "workers": 8, "prefetch": 2, "steps": 500, "step_time": 105_000_000,
                "samples": 50_000_000, "sample_median": 110 << 10}


def test_file_per_sample_leaves_the_job_and_the_gpu_to_a_case():
    """DESIGN_REVIEW.md §3.78 points 1, 4, and 11: the job's values, the model's on a GPU, and the
    corpus's size distribution have no default, `samples` is the submission's scale, and the reads
    Python's BufferedReader sizes are st_blksize, which the storage reports."""
    p = json.loads((FPS / "file_per_sample.ast.json").read_text())["params"]
    for k, v in FPS_DEFAULTS.items():
        assert "default" not in p[k] and p[k]["kind"] == type(v).__name__, k
    assert p["samples"].get("scale") is True
    assert {k for k, v in p.items() if "default" not in v} == set(FPS_DEFAULTS)
    for k in ("hdr_read", "inv_read"):
        assert p[k]["default"] == {"from": "st_blksize", "of": "train", "fallback": 1 << 20}, k


def test_file_per_sample_has_an_unsettled_case_of_its_old_defaults():
    """`jpeg-50m` holds the values the defaults held, each a default, said to be not measured: kind
    `unsettled`, no capture, and a lock that is current."""
    import tomllib
    case = FPS / "cases/jpeg-50m"
    rec = tomllib.loads((case / "case.toml").read_text())
    assert rec["kind"] == "unsettled" and "accelerator" not in rec
    assert rec["run"] == [{"ast": "file_per_sample", "params": "params.json"}]
    assert not (case / "captures").exists()
    doc = json.loads((case / "params.json").read_text())
    assert "not measured" in doc["doc"]
    assert doc["params"] == {k: {"default": v} for k, v in FPS_DEFAULTS.items()}
    (run,) = json.loads((case / lock.LOCK_FILE).read_text())["runs"]
    assert {k: run["params"][k] for k in FPS_DEFAULTS} == {k: {"default": v} for k, v in FPS_DEFAULTS.items()}
    assert lock.stale(case, workloads=WORKLOADS) is None
    assert "not measured" in (case / "README.md").read_text()


def test_the_example_case_of_file_per_sample_gives_every_required_value():
    """The traced example gives every value the abstract requires: the corpus kit/mkcorpus.py
    writes, and `step_time` the old default, said so (never fitted: the trace's compute is the
    stand-in's)."""
    required = {k for k, p in json.loads((FPS / "file_per_sample.ast.json").read_text())["params"].items() if "default" not in p}
    doc = json.loads((FPS / "cases/imagefolder-jpeg/params.json").read_text())
    assert required and required <= set(doc["params"])
    assert doc["params"]["step_time"] == {"default": 105_000_000} and "not fitted" in doc["doc"]
    assert doc["params"]["sample_median"] == {"default": 110 << 10}
