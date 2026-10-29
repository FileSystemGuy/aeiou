"""The application records (`applications/`, `DESIGN_REVIEW.md` §3.88): what `aeiou-trace digest`
knows about the applications whose traces it reads. Every committed record is walked; the loader's
refusals, the signature matching, the version resolution, and the assertions a record makes are
held to constructed trees."""
import pathlib
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
sys.path.insert(0, str(BUILDER))

from aeiou import applications as ap  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402

REPO = ap.default_dir()


def test_the_default_directory_is_the_checkout_or_the_wheel(tmp_path, monkeypatch):
    assert REPO.is_dir() and (REPO / "attributes.toml").is_file()
    (tmp_path / "pkg" / "applications").mkdir(parents=True)
    monkeypatch.setattr(ap, "__file__", str(tmp_path / "pkg" / "applications.py"))
    assert ap.default_dir() == tmp_path / "pkg" / "applications"   # shipped in the wheel: beside the module


def test_every_committed_record_loads():
    apps = ap.load(REPO)
    assert set(apps) >= {"milvus", "qdrant", "pytorch-dcp", "lmcache", "safetensors", "diskann", "faiss"}
    for name, a in apps.items():
        assert a.name == name and a.kind in ("library", "service", "managed") and a.records, name
        for r in a.records:
            assert r.path.stem == r.version and r.read and r.sources
            for o in r.options:
                assert o["choices"] and ("default" not in o or o["default"] in o["choices"])
    assert apps["faiss"].signatures == [] and apps["pytorch-dcp"].mechanisms == ["checkpointing/dcp-save", "checkpointing/dcp-load"]


def test_identification_by_path_signatures():
    apps = ap.load(REPO)
    found = ap.identify(apps, ["ckpt/step_000001/__0_0.distcp", "ckpt/step_000001/.metadata"])
    assert [(a.name, sig) for a, sig in found] == [("pytorch-dcp", "__*_*.distcp")]
    assert [a.name for a, _ in ap.identify(apps, [".metadata"])] == ["pytorch-dcp"]   # a signature matches at the root too
    assert [a.name for a, _ in ap.identify(apps, ["kv/meta-llama-Llama-3.1-8B-Instruct@1@0@3f2a9c0e1b7d4e6f@torch.bfloat16.pt"])] == ["lmcache"]
    assert [a.name for a, _ in ap.identify(apps, ["files/insert_log/1/2/3/4/5", "a/b/sift_disk.index"])] == ["diskann", "milvus"]
    assert ap.identify(apps, ["ivf/merged_index.ivfdata", "train/00000/sample_000000001.npz"]) == []
    assert [a.name for a, _ in ap.identify(apps, ["storage/collections/c/0/segments/u/vector_storage/0.dat"])] == ["qdrant"]


def test_version_resolution():
    apps = ap.load(REPO)
    dcp = apps["pytorch-dcp"]
    r, exact = ap.resolve(dcp, "2.14.1")
    assert r.version == "2.14" and exact
    r, exact = ap.resolve(dcp, "2.15.0")
    assert r.version == "2.14" and exact   # no `until`: the record claims every later version until one says otherwise
    r, exact = ap.resolve(dcp, "2.9")
    assert r is None and not exact   # below the lowest record: nothing claims it
    r, exact = ap.resolve(dcp, None)
    assert r.version == "2.14" and not exact   # unknown: the newest record, as a suggestion
    r, exact = ap.resolve(dcp, "two")
    assert r.version == "2.14" and not exact
    assert ap.version_key("3.0.3") == (3, 0, 3) and ap.version_key("v1.20.0") == (1, 20, 0) and ap.version_key("1.20rc1") == (1, 20) and ap.version_key("latest") is None


@pytest.fixture
def tree(tmp_path):
    """A copy of the committed records to break."""
    shutil.copytree(REPO, tmp_path / "applications")
    return tmp_path / "applications"


def test_a_range_with_an_end_and_the_nearest_lower_record(tree):
    (tree / "pytorch-dcp" / "2.14.toml").write_text((tree / "pytorch-dcp" / "2.14.toml").read_text().replace('from = "2.14"\n', 'from = "2.14"\nuntil = "2.15"\n'))
    (tree / "pytorch-dcp" / "2.16.toml").write_text('from = "2.16"\nread = "2026-10-29"\nsources = ["x"]\n')
    dcp = ap.load(tree)["pytorch-dcp"]
    assert [r.version for r in dcp.records] == ["2.14", "2.16"]
    assert ap.resolve(dcp, "2.14.9") == (dcp.records[0], True)
    assert ap.resolve(dcp, "2.15.2") == (dcp.records[0], False)   # between the ranges: the nearest lower, downgraded
    assert ap.resolve(dcp, "2.16") == (dcp.records[1], True)
    assert ap.resolve(dcp, None) == (dcp.records[1], False)


@pytest.mark.parametrize("file, old, new, message", [
    ("milvus/application.toml", 'kind = "service"', 'kind = "cloud"', "kind must be one of library, service, managed"),
    ("milvus/application.toml", 'name = "milvus"', 'name = "milvos"', "name must be the directory's, milvus"),
    ("milvus/application.toml", 'display = "Milvus"\n', "", "display is required"),
    ("milvus/application.toml", 'signatures = [', 'signatures = "x"\nold = [', "signatures must be a list of strings"),
    ("milvus/3.0.toml", 'from = "3.0"', 'from = "3.1"', "from must be the file's stem, 3.0"),
    ("milvus/3.0.toml", 'from = "3.0"\n', 'from = "3.0"\nuntil = "2.9"\n', "until must be above from"),
    ("milvus/3.0.toml", 'default = "in-memory index"', 'default = "GPU index"', "option index: default must be one of its choices"),
    ("milvus/3.0.toml", '"memory-mapped" = "mmap"', '"memory-mapped" = "swap"', "option mmap_vector_field: blind spot swap is not in applications/attributes.toml"),
    ("milvus/3.0.toml", '"memory-mapped" = "mmap"', '"mapped" = "mmap"', "option mmap_vector_field: blind spot for a choice it does not have, mapped"),
    ("milvus/3.0.toml", 'confidence = "medium"', 'confidence = "guess"', "confidence must be high or medium"),
    ("milvus/3.0.toml", 'kind = "object"', 'kind = "network"', "blind spot network is not in applications/attributes.toml"),
    ("pytorch-dcp/2.14.toml", 'question = "size"', 'question = "colour"', "assert 1: question must be one of selection, size, order"),
    ("pytorch-dcp/2.14.toml", 'answer = "per-path"', 'answer = "huge"', "assert 1: answer must be one of per-path, drawn"),
    ("pytorch-dcp/2.14.toml", 'template = "__*_*.distcp"\n', 'template = "__*_*.distcp"\nwhen = { option = "sync", choice = "fsync on" }\n', "assert 1: when names no option of the record, sync"),
    ("pytorch-dcp/2.14.toml", 'template = "__*_*.distcp"\n', 'template = "__*_*.distcp"\nwhen = { option = "sync_files", choice = "maybe" }\n', "assert 1: when names no choice of option sync_files, maybe"),
    ("pytorch-dcp/2.14.toml", 'question = "size"\nanswer = "per-path"', 'fact = "passes"\nvalue = "four"', "assert 1: value must be a whole number of at least 1"),
    ("pytorch-dcp/2.14.toml", 'question = "size"\nanswer = "per-path"', 'fact = "sizes"\nvalue = 4', "assert 1: fact must be one of candidates, passes"),
    ("pytorch-dcp/2.14.toml", 'question = "size"\nanswer = "per-path"', 'extra = "nothing"', "assert 1: either fact and value or question and answer"),
    ("pytorch-dcp/2.14.toml", 'read = "2026-10-29"\n', "", "read is required"),
    ("pytorch-dcp/2.14.toml", 'sources = [', 'sources = "x"\nold = [', "sources must be a non-empty list of strings"),
    ("pytorch-dcp/2.14.toml", 'name = "sync_files"\n', "", "option 2: name and question are required"),
    ("pytorch-dcp/2.14.toml", 'choices = ["fsync on", "fsync off"]', 'choices = ["fsync on"]', "option sync_files: choices must be a list of at least two strings"),
    ("safetensors/0.8.toml", 'confidence = "high"', 'confidence = "sure"', "blind spot 1: confidence must be high or medium"),
    ("pytorch-dcp/2.14.toml", 'template = "__*_*.distcp"\n', "", "assert 1: template is required"),
    ("pytorch-dcp/2.14.toml", 'source = "workloads/checkpointing/dcp-load/kit/params.py (the plan read from .metadata)"\nconfidence = "high"\n', 'source = "x"\n', "assert 1: confidence must be high or medium"),
    ("milvus/application.toml", 'name = "milvus"', 'name = = "milvus"', "TOML"),
    ("pytorch-dcp/2.14.toml", 'sources = [', 'sources = [[', "TOML"),
])
def test_records_are_refused(tree, file, old, new, message):
    p = tree / file
    text = p.read_text()
    assert old in text, old
    p.write_text(text.replace(old, new, 1))
    with pytest.raises(BuildError, match=message) as e:
        ap.load(tree)
    assert str(p.relative_to(tree)) in str(e.value)


def test_a_directory_without_a_record_or_a_stray_file_is_refused(tree):
    (tree / "nothing").mkdir()
    with pytest.raises(BuildError, match="nothing: no application.toml"):
        ap.load(tree)
    shutil.rmtree(tree / "nothing")
    (tree / "milvus" / "notes.txt").write_text("x")
    with pytest.raises(BuildError, match="milvus/notes.txt: not application.toml or a <version>.toml"):
        ap.load(tree)
    (tree / "milvus" / "notes.txt").unlink()
    (tree / "attributes.toml").unlink()
    with pytest.raises(BuildError, match="attributes.toml"):
        ap.load(tree)
    with pytest.raises(BuildError, match="no such directory"):
        ap.load(tmp := tree.parent / "absent")
    del tmp


def test_assertions_and_blind_spots_of_a_record(tree):
    p = tree / "pytorch-dcp" / "2.14.toml"
    p.write_text(p.read_text() + '''
[[assert]]
template = "__*_*.distcp"
fact = "passes"
value = 3
note = "only when fsync is off"
source = "made up for the test"
confidence = "medium"
when = { option = "sync_files", choice = "fsync off" }
''')
    dcp = ap.load(tree)["pytorch-dcp"]
    r = dcp.records[0]
    templates = [("ckpt/step_{a:06}/__{b}_0.distcp", ["ckpt/step_000001/__0_0.distcp", "ckpt/step_000001/__1_0.distcp"]), ("ckpt/step_{a:06}/.metadata", ["ckpt/step_000001/.metadata"])]
    got = ap.assertions(r, {}, templates)
    assert got == [{"template": "ckpt/step_{a:06}/__{b}_0.distcp", "question": "size", "answer": "per-path",
                    "note": "a rank's shard holds the bytes of its plan's items; the bytes are a property of the file, the same at every load",
                    "by": "PyTorch Distributed Checkpoint 2.14 record, read 2026-10-29 (high confidence)", "source": "workloads/checkpointing/dcp-load/kit/params.py (the plan read from .metadata)"}]
    got = ap.assertions(r, {"sync_files": "fsync off"}, templates)
    assert [a.get("fact") or a["question"] for a in got] == ["size", "passes"] and got[1]["value"] == 3 and "medium confidence" in got[1]["by"]
    assert ap.assertions(r, {}, [("other/{a}.bin", ["other/1.bin"])]) == []
    milvus = ap.load(tree)["milvus"].records[0]
    spots = ap.blind_spots(milvus, {"mmap_vector_field": "memory-mapped", "mmap_vector_index": "loaded into memory"})
    assert [(s["kind"], s.get("option")) for s in spots] == [("object", None), ("mmap", "mmap_vector_field")]
    assert spots[1]["note"].startswith("the application maps its files")
    assert ap.provenance(milvus) == "Milvus 3.0 record, read 2026-10-29"
