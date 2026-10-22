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
