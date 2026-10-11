"""`aeiou-datagen` into an object store (DESIGN_REVIEW.md §3.69): each format class's container
dataset, declared `protocol: s3`, written into a MinIO bucket by streaming the writer into
uploads, every object byte for byte the file the directory writer makes under the same name,
the manifest the same dataset id; and `aeiou run` (built with the cargo feature `object`)
reading the bucket to the dry run's fingerprint. The server is the binary `TEST_MINIO_BIN`
names, started on a free port with a directory of its own, as `runner/aeiou/tests/object.rs`
starts it; the tests that need it are skipped, saying so, when it is unset."""
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
sys.path.insert(0, str(BUILDER))
pytest.importorskip("pyarrow")
pytest.importorskip("h5py")
pytest.importorskip("dgen_py")
obstore = pytest.importorskip("obstore")

from aeiou import datagen  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", ROOT / "runner" / "target" / "release" / "aeiou"))
USER, SECRET = "aeiou", "aeiou-secret"

CASES = {
    "tfrecord_stream": ["samples=768", "per_shard=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"],
    "parquet_stream": ["samples=768", "per_shard=128", "batch=32", "steps=8", "cycle=2", "sample_median=20000"],
    "hdf5_map": ["samples=64", "per_file=16", "batch=4", "workers=2", "steps=4"],
}


def ast_path(name):
    return next((ROOT / "workloads").rglob(f"{name}.ast.json"))


def object_variant(name, tmp_path, names=("shards",)):
    """The committed AST with its datasets `names` declared `protocol: s3`, and, when no name
    is left `posix`, without its POSIX API and cache mode (V19), as the Rust tests make it."""
    d = json.loads(ast_path(name).read_text())
    for n in names:
        body = next(iter(d["datasets"][n].values()))
        body["protocol"] = "s3"
    if len(names) == len(d.get("datasets", {})) + len(d.get("namespaces", {})):
        d.pop("posix", None)
        d.pop("cache", None)
    p = tmp_path / f"{name}.object.ast.json"
    p.write_text(json.dumps(d))
    return p


def dataset_names(name):
    return list(json.loads(ast_path(name).read_text())["datasets"])


class Minio:
    def __init__(self, tmp):
        bin_ = os.environ["TEST_MINIO_BIN"]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "MINIO_ROOT_USER": USER, "MINIO_ROOT_PASSWORD": SECRET}
        self.proc = subprocess.Popen([bin_, "server", str(tmp / "minio"), "--address", f"127.0.0.1:{port}",
                                      "--console-address", "127.0.0.1:0", "--quiet"], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while self.curl("-X", "PUT", f"{self.url}/bench").returncode != 0:
            if time.time() > deadline:
                self.proc.kill()
                raise RuntimeError("MinIO did not come up")
            time.sleep(0.2)
        self.env = {**os.environ, "AWS_ENDPOINT_URL": self.url, "AWS_ACCESS_KEY_ID": USER, "AWS_SECRET_ACCESS_KEY": SECRET,
                    "AWS_REGION": "us-east-1", "AWS_ALLOW_HTTP": "true", "PYTHONPATH": str(BUILDER)}
        self.env.pop("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", None)

    def curl(self, *args):
        env = {k: v for k, v in os.environ.items() if k.lower() != "http_proxy"}
        return subprocess.run(["curl", "-sf", "--aws-sigv4", "aws:amz:us-east-1:s3", "--user", f"{USER}:{SECRET}", *args],
                              capture_output=True, env=env)

    def store(self):
        from obstore.store import S3Store
        return S3Store("bench", endpoint=self.url, access_key_id=USER, secret_access_key=SECRET, region="us-east-1",
                       virtual_hosted_style_request=False, client_options={"allow_http": True})

    def objects(self, prefix):
        return {o["path"]: o["size"] for chunk in obstore.list(self.store(), prefix=prefix) for o in chunk}

    def get(self, key):
        return bytes(obstore.get(self.store(), key).bytes())


@pytest.fixture(scope="module")
def minio(tmp_path_factory):
    if not os.environ.get("TEST_MINIO_BIN"):
        pytest.skip("TEST_MINIO_BIN is not set (a MinIO server binary)")
    m = Minio(tmp_path_factory.mktemp("minio"))
    yield m
    m.proc.kill()
    m.proc.wait()


def _datagen(args, env=None):
    return subprocess.run([sys.executable, "-m", "aeiou.datagen", *map(str, args)], capture_output=True, text=True,
                          cwd=str(BUILDER), env=env or {**os.environ, "PYTHONPATH": str(BUILDER)})


def _has_object_engine():
    if not RUNNER.exists():
        return False
    r = subprocess.run([RUNNER, "run", "--help"], capture_output=True, text=True)
    return "--object-threads" in r.stdout



def test_two_ranks_write_one_bucket_prefix(minio, tmp_path):
    """`--ranks 2` into a bucket: each rank its slice of the objects, rank 0 the manifest,
    the objects those of one host (`test_coord.py` has the protocol)."""
    import socket
    flags = sum([["--param", p] for p in CASES["tfrecord_stream"]], [])
    ast = object_variant("tfrecord_stream", tmp_path)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        addr = f"127.0.0.1:{s.getsockname()[1]}"
    common = [sys.executable, "-m", "aeiou.datagen", str(ast), "--root", str(tmp_path / "unused"), *flags, "--writers", "2",
              "--ranks", "2", "--coordinator", addr]
    procs = [subprocess.Popen([*common, "--rank", str(r), "--endpoint", "shards=s3://bench/ranks/data"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, cwd=str(BUILDER), env=minio.env) for r in (0, 1)]
    outs = [p.communicate(timeout=300) for p in procs]
    for p, (o, e) in zip(procs, outs):
        assert p.returncode == 0, o + e
    assert "manifest written" in outs[0][0] and "(manifest by rank 0)" in outs[1][0]
    one = _datagen([ast, "--root", tmp_path / "unused", "--endpoint", "shards=s3://bench/one/data", *flags], minio.env)
    assert one.returncode == 0, one.stdout + one.stderr
    two, single = minio.objects("ranks/data"), minio.objects("one/data")
    assert sorted(k[len("ranks/"):] for k in two) == sorted(k[len("one/"):] for k in single)
    for k in two:
        if not k.endswith(".aeiou-dataset.json"):
            assert minio.get(k) == minio.get("one/" + k[len("ranks/"):]), k
    m2, m1 = json.loads(minio.get("ranks/data/.aeiou-dataset.json")), json.loads(minio.get("one/data/.aeiou-dataset.json"))
    assert datagen.Job.manifest_id(m2) == datagen.Job.manifest_id(m1)
    assert m2["provenance"]["ranks"] == 2 and [h["files_written"] for h in m2["provenance"]["hosts"]] == [3, 3]

@pytest.mark.parametrize("name", sorted(CASES))
def test_containers_stream_into_the_bucket_as_the_directory_writer_writes_them(name, minio, tmp_path):
    flags = sum([["--param", p] for p in CASES[name]], [])
    names = dataset_names(name)
    ast = object_variant(name, tmp_path, names)
    prefix = f"{name}/data"
    endpoints = sum([["--endpoint", f"{n}=s3://bench/{prefix}"] for n in names], [])
    r = _datagen([ast, "--root", tmp_path / "unused", *endpoints, "--writers", "3", *flags], minio.env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (tmp_path / "unused").exists(), "nothing written to a local file system"
    on_disk = tmp_path / "posix"
    r2 = _datagen([ast_path(name), "--root", on_disk, "--writers", "3", *flags])
    assert r2.returncode == 0, r2.stdout + r2.stderr
    disk_root = next(p.parent for p in on_disk.rglob(".aeiou-dataset.json"))
    files = sorted(p for p in disk_root.rglob("*") if p.is_file() and p.name != ".aeiou-dataset.json")
    objects = minio.objects(prefix)
    assert set(objects) == {f"{prefix}/{p.relative_to(disk_root)}" for p in files} | {f"{prefix}/.aeiou-dataset.json"}
    for p in files:
        assert minio.get(f"{prefix}/{p.relative_to(disk_root)}") == p.read_bytes(), p
    m_obj = json.loads(minio.get(f"{prefix}/.aeiou-dataset.json"))
    m_disk = json.loads((disk_root / ".aeiou-dataset.json").read_text())
    assert datagen.Job.manifest_id(m_obj) == datagen.Job.manifest_id(m_disk), "the protocol is not part of a dataset id"
    assert f"at s3://bench/{prefix}" in r.stdout
    # a second datagen into the same prefix is refused: datasets are read-only
    again = _datagen([ast, "--root", tmp_path / "unused", *endpoints, *flags], minio.env)
    assert again.returncode == 1 and "is not empty" in again.stderr
    if not _has_object_engine():
        pytest.skip(f"{RUNNER} is not built with the cargo feature `object`")
    dry = subprocess.run([RUNNER, "dry-run", ast, "--gpus", "2", "--seed", "3", *flags], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    fp = [l.split()[1] for l in dry.stdout.splitlines() if l.startswith("fingerprint ")][0]
    for api in ("blocking", "async"):
        run = subprocess.run([RUNNER, "run", ast, "--root", tmp_path / "unused", *endpoints, "--gpus", "2", "--seed", "3",
                              "--time-scale", "0", "--s3", api, "--expect-fingerprint", fp, *flags],
                             capture_output=True, text=True, env=minio.env)
        assert run.returncode == 0, api + ": " + run.stdout + run.stderr
        assert "fingerprint matches" in run.stdout


def test_a_multipart_upload_is_the_file(minio, tmp_path):
    """A file of several parts (the part is 8 MiB) arrives whole; an upload a writer fails
    before its close is aborted and leaves no object."""
    name = "hdf5_map"
    flags = sum([["--param", p] for p in ["samples=96", "per_file=96", "batch=4", "workers=2", "steps=4"]], [])
    ast = object_variant(name, tmp_path, dataset_names(name))
    r = _datagen([ast, "--root", tmp_path / "unused", *sum([["--endpoint", f"{n}=s3://bench/big"] for n in dataset_names(name)], []), *flags], minio.env)
    assert r.returncode == 0, r.stdout + r.stderr
    on_disk = tmp_path / "posix"
    assert _datagen([ast_path(name), "--root", on_disk, *flags]).returncode == 0
    (f,) = [p for p in on_disk.rglob("*.h5")]
    assert f.stat().st_size > 2 * datagen.PART
    assert minio.get(f"big/{f.name}") == f.read_bytes()
    store = datagen.Bucket.__new__(datagen.Bucket)
    store.store, store.bucket, store.prefix, store.uri = minio.store(), "bench", "aborted", "s3://bench/aborted"
    with pytest.raises(RuntimeError):
        with datagen.opened(datagen.Obj(store, "aborted/x")) as out:
            out.write(b"x" * (datagen.PART + 1))
            raise RuntimeError("the writer failed")
    assert minio.objects("aborted") == {}


def test_object_endpoints_are_placed_as_the_runner_places_them(tmp_path):
    """The runner's refusals (endpoint.rs), with no server: an object name with no endpoint,
    an endpoint of the wrong kind, a posix name under an object root."""
    name = "tfrecord_stream"
    flags = sum([["--param", p] for p in CASES[name]], [])
    ast = object_variant(name, tmp_path)
    r = _datagen([ast, "--root", tmp_path / "r", *flags])
    assert r.returncode == 2 and "`shards` is declared `protocol: s3` and has no object endpoint: give it --endpoint shards=s3://BUCKET[/PREFIX]" in r.stderr
    r = _datagen([ast, "--root", tmp_path / "r", "--endpoint", "shards=/somewhere", *flags])
    assert r.returncode == 2 and "`shards` is declared `protocol: s3`; its endpoint is s3://BUCKET[/PREFIX]" in r.stderr
    r = _datagen([ast_path(name), "--root", tmp_path / "r", "--endpoint", "shards=s3://bench/x", *flags])
    assert r.returncode == 2 and "`shards` is a `posix` dataset or namespace; its endpoint is a directory" in r.stderr
