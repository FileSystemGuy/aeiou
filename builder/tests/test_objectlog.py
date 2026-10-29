"""Object-store request logs read as traces (`DESIGN_REVIEW.md` §3.88 step 5): MinIO's audit log
and Amazon S3's server access log become the exporter's event stream, so metrics, digest, export,
and the trace node read them unchanged."""
import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
sys.path.insert(0, str(BUILDER))

from aeiou import digest, objectlog, trace  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402


def minio(name, bucket, obj, status=200, rx=0, tx=0, t="2026-10-29T12:00:00.000000000Z", ns=2_000_000, host="10.0.0.5:4321", rng=None, query=None):
    e = {"version": "1", "deploymentid": "d", "time": t, "trigger": "incoming",
         "api": {"name": name, "bucket": bucket, "object": obj, "status": "OK" if status == 200 else "Error", "statusCode": status, "rx": rx, "tx": tx,
                 "timeToFirstByteInNS": str(ns // 2), "timeToResponseInNS": str(ns)},
         "remotehost": host, "requestID": "r", "userAgent": "test", "requestHeader": {"Host": "minio"}, "responseHeader": {}}
    if rng:
        e["requestHeader"]["Range"] = rng
    if query:
        e["requestQuery"] = query
    return json.dumps(e)


def at(ms):
    return f"2026-10-29T12:00:{ms // 1000:02}.{(ms % 1000) * 1_000_000:09}Z"


MINIO = "\n".join([
    minio("PutObject", "a-bucket", "files/insert_log/1/2/3/100/7", rx=4096, t=at(0)),
    minio("PutObject", "a-bucket", "files/insert_log/1/2/3/100/8", rx=8192, t=at(10)),
    minio("GetObject", "a-bucket", "files/insert_log/1/2/3/100/7", tx=4096, t=at(20), host="10.0.0.6:1"),
    minio("GetObject", "a-bucket", "files/insert_log/1/2/3/100/8", tx=1024, t=at(30), host="10.0.0.6:1", rng="bytes=4096-5119"),
    minio("GetObject", "a-bucket", "files/insert_log/1/2/3/100/9", status=404, t=at(40), host="10.0.0.6:1"),
    minio("HeadObject", "a-bucket", "files/insert_log/1/2/3/100/7", t=at(50)),
    minio("HeadObject", "a-bucket", "files/insert_log/1/2/3/100/9", status=404, t=at(51)),
    minio("ListObjectsV2", "a-bucket", "", t=at(60), query={"prefix": "files/insert_log/1/"}),
    minio("ListObjectsV2", "a-bucket", "", t=at(61)),
    minio("DeleteObject", "a-bucket", "files/insert_log/1/2/3/100/8", t=at(70)),
    minio("NewMultipartUpload", "a-bucket", "files/index_files/5/1/2/3/big", t=at(80), query={"uploads": ""}, host="10.0.0.7:2"),
    minio("PutObjectPart", "a-bucket", "files/index_files/5/1/2/3/big", rx=1 << 20, t=at(90), query={"uploadId": "u1", "partNumber": "1"}, host="10.0.0.7:2"),
    minio("PutObjectPart", "a-bucket", "files/index_files/5/1/2/3/big", rx=1 << 20, t=at(91), query={"uploadId": "u1", "partNumber": "2"}, host="10.0.0.8:3"),
    minio("CompleteMultipartUpload", "a-bucket", "files/index_files/5/1/2/3/big", t=at(100), query={"uploadId": "u1"}, host="10.0.0.7:2"),
    minio("GetObject", "other-bucket", "x", tx=10, t=at(110)),
    minio("GetBucketLocation", "a-bucket", "", t=at(120)),
    minio("CopyObject", "a-bucket", "files/insert_log/1/2/3/100/7", t=at(130)),
    minio("PutObjectPart", "a-bucket", "files/y", rx=5, t=at(140), query={"uploadId": "nope"}),
    minio("NewMultipartUpload", "a-bucket", "files/dangling", t=at(150), query={"uploadId": "u2"}),
]) + "\n"


def test_source_detection():
    assert objectlog.source_of(MINIO.splitlines()[0]) == "minio-audit"
    assert objectlog.source_of('{"aeiou_trace": 1}') is None and objectlog.source_of('{"api": "x"}') is None and objectlog.source_of("{not json") is None and objectlog.source_of('{"api": x') is None
    assert objectlog.source_of(S3.splitlines()[0]) == "s3-access"
    assert objectlog.source_of('100 1700000000.000000 openat(AT_FDCWD, "/d/x", O_RDONLY) = 3') is None


def test_minio_audit_requests_become_the_engines_calls():
    header, events = objectlog.export_of(MINIO.splitlines(True), "a-bucket/files", "minio-audit")
    assert header["aeiou_trace"] == 1 and header["source"] == "minio-audit" and header["root"] == "a-bucket/files" and header["lanes"] == 4
    ops = [(e["lane"], e["op"], e.get("path"), e.get("fd"), e.get("offset"), e.get("len"), e.get("ret")) for e in events]
    assert ops[:3] == [(0, "open", "insert_log/1/2/3/100/7", 0, None, None, None), (0, "write", None, 0, None, 4096, 4096), (0, "close", None, 0, None, None, None)]
    get = [e for e in events if e["op"] == "read"]
    assert [(e["fd"], e.get("offset"), e["len"], e["ret"], e["lane"]) for e in get] == [(2, None, 4096, 4096, 1), (3, 4096, 1024, 1024, 1)]
    missing = [e for e in events if e["op"] == "open" and "ret" in e]
    assert [(e["path"], e["ret"]) for e in missing] == [("insert_log/1/2/3/100/9", "ENOENT")]
    stats = [e for e in events if e["op"] == "stat"]
    assert [(e["path"], e.get("ret")) for e in stats] == [("insert_log/1/2/3/100/7", None), ("insert_log/1/2/3/100/9", "ENOENT")]
    lists = [e for e in events if e["op"] == "readdir"]
    assert len(lists) == 1 and events[events.index(lists[0]) - 1]["path"] == "insert_log/1" and events[events.index(lists[0]) - 1]["flags"] == ["RDONLY", "DIRECTORY"]
    assert [e["path"] for e in events if e["op"] == "unlink"] == ["insert_log/1/2/3/100/8"]
    big = [e for e in events if e["op"] == "open" and e["path"] == "index_files/5/1/2/3/big"]
    assert len(big) == 1 and big[0]["flags"] == ["WRONLY", "CREAT", "TRUNC"] and big[0]["mode"] == 0o644
    parts = [e for e in events if e["op"] == "write" and e["fd"] == big[0]["fd"]]
    assert [(e["len"], e["lane"], e.get("offset")) for e in parts] == [(1 << 20, 2, 0), (1 << 20, 3, 1 << 20)]   # two lanes: the parts' positions resolved
    assert sum(1 for e in events if e["op"] == "close" and e["fd"] == big[0]["fd"]) == 1
    assert header["creates"] == ["insert_log/1/2/3/100/7", "insert_log/1/2/3/100/8", "index_files/5/1/2/3/big", "dangling"]
    n = header["notes"]
    assert n["requests_outside_root"] == 2 and n["requests_not_exported:GetBucketLocation"] == 1 and n["requests_not_exported:CopyObject"] == 1
    assert n["multipart_without_start"] == 1 and n["multipart_not_completed"] == 1 and "time_is_response_end" in n and n["lanes"].startswith("one per client host")
    # times: the first request's end is 2 ms after its start; events are in time order of the log, t from the earliest start
    first = events[0]
    assert first["t"] == 0 and abs(first["dur"] - 2_000_000) < 1000   # float seconds near 1.8e9 carry about 200 ns
    get7 = [e for e in events if e["op"] == "open" and e["path"] == "insert_log/1/2/3/100/7" and "ret" not in e][1]
    assert abs(get7["t"] - 20_000_000) < 1000


S3 = "\n".join([
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:00 +0000] 192.0.2.3 - R1 REST.PUT.OBJECT data/img_000000001.jpg "PUT /amzn-bucket/data/img_000000001.jpg HTTP/1.1" 200 - - 110000 41 28 "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:01 +0000] 192.0.2.4 - R2 REST.GET.OBJECT data/img_000000001.jpg "GET /amzn-bucket/data/img_000000001.jpg HTTP/1.1" 200 - 110000 110000 70 10 "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:02 +0000] 192.0.2.4 - R3 REST.GET.OBJECT data/img_000000001.jpg "GET /amzn-bucket/data/img_000000001.jpg HTTP/1.1" 206 - 4096 110000 7 1 "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:03 +0000] 192.0.2.4 - R4 REST.HEAD.OBJECT data/img_000000002.jpg "HEAD /amzn-bucket/data/img_000000002.jpg HTTP/1.1" 404 NoSuchKey - - 5 - "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:04 +0000] 192.0.2.4 - R5 REST.GET.BUCKET - "GET /amzn-bucket?prefix=data%2F&list-type=2 HTTP/1.1" 200 - 2000 - 9 - "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:05 +0000] 192.0.2.3 - R6 REST.DELETE.OBJECT data/img_000000001.jpg "DELETE /amzn-bucket/data/img_000000001.jpg HTTP/1.1" 204 - - - 3 - "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:06 +0000] 192.0.2.3 - R7 REST.GET.VERSIONING - "GET /amzn-bucket?versioning HTTP/1.1" 200 - 113 - 7 - "-" "console" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:07 +0000] 192.0.2.3 - R8 REST.POST.UPLOADS data/big.bin "POST /amzn-bucket/data/big.bin?uploads HTTP/1.1" 200 - 300 - 7 - "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:08 +0000] 192.0.2.3 - R9 REST.PUT.PART data/big.bin "PUT /amzn-bucket/data/big.bin?partNumber=1&uploadId=u9 HTTP/1.1" 200 - - 5242880 100 90 "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
    f'79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be amzn-bucket [29/Oct/2026:12:00:09 +0000] 192.0.2.3 - R10 REST.POST.UPLOAD data/big.bin "POST /amzn-bucket/data/big.bin?uploadId=u9 HTTP/1.1" 200 - 300 - 20 - "-" "aws-sdk" - h SigV4 - AuthHeader s3.amazonaws.com TLSv1.2 - - us-east-1',
]) + "\n"


def test_s3_access_log_records_become_the_engines_calls():
    header, events = objectlog.export_of(S3.splitlines(True), "amzn-bucket", "s3-access")
    assert header["source"] == "s3-access" and header["lanes"] == 2
    ops = [(e["op"], e.get("path"), e.get("offset"), e.get("len"), e.get("ret")) for e in events]
    assert ops[0] == ("open", "data/img_000000001.jpg", None, None, None) and ops[1] == ("write", None, None, 110000, 110000)
    reads = [e for e in events if e["op"] == "read"]
    assert [(e["len"], e["ret"], e.get("offset")) for e in reads] == [(110000, 110000, None), (4096, 4096, None)]
    assert header["notes"]["ranges_unknown"] == 1 and "time_is_response_end" not in header["notes"]
    assert [(e["path"], e.get("ret")) for e in events if e["op"] == "stat"] == [("data/img_000000002.jpg", "ENOENT")]
    assert [e["path"] for e in events if e["op"] == "open" and "DIRECTORY" in e["flags"]] == ["data"]
    assert [e["path"] for e in events if e["op"] == "unlink"] == ["data/img_000000001.jpg"]
    assert header["notes"]["requests_not_exported:REST.GET.VERSIONING"] == 1
    big = [e for e in events if e["op"] == "open" and e["path"] == "data/big.bin"]
    parts = [e for e in events if e["op"] == "write" and e["fd"] == big[0]["fd"]]
    assert [e["len"] for e in parts] == [5242880] and sum(1 for e in events if e["op"] == "close" and e["fd"] == big[0]["fd"]) == 1
    # the second's start is a second after the first's, and a GET's duration is its total time
    assert abs(events[3]["t"] - 1_000_000_000) < 1000 and abs(events[3]["dur"] - 70_000_000) < 1000


def test_refusals_and_edge_cases():
    with pytest.raises(BuildError, match="not an object log format"):
        objectlog.export_of([], "b", "ceph")
    with pytest.raises(BuildError, match="needs --root"):
        objectlog.export_of([], "/", "minio-audit")
    with pytest.raises(BuildError, match="line 1: Expecting"):
        objectlog.export_of(["{bad\n"], "b", "minio-audit")
    with pytest.raises(BuildError, match="line 1: not a JSON object"):
        objectlog.export_of(["[]\n"], "b", "minio-audit")
    with pytest.raises(BuildError, match="line 1: no api.name"):
        objectlog.export_of(['{"api": {}}\n'], "b", "minio-audit")
    with pytest.raises(BuildError, match="line 1: not an S3 server access log record"):
        objectlog.export_of(["nonsense\n"], "b", "s3-access")
    # a request without a time is counted, an abort closes and unlinks, a HEAD outside the root is counted
    lines = [minio("NewMultipartUpload", "b", "k", t="soon"), "", minio("AbortMultipartUpload", "b", "k", t="2026-10-29T12:00:01Z"),
             minio("HeadObject", "c", "k"), minio("ListObjectsV2", "c", "", query={"prefix": "p"}), minio("GetObject", "b", "k", status=500),
             minio("CompleteMultipartUpload", "c", "z"), minio("PutObjectPart", "c", "z", rx=1)]
    header, events = objectlog.export_of([line + "\n" for line in lines], "b", "minio-audit")
    assert [e["op"] for e in events] == ["open", "close", "unlink", "open"] and events[-1]["ret"] == "EIO"
    assert header["notes"]["requests_without_time"] == 1 and header["notes"]["requests_outside_root"] == 4 and header["requests"] == 7
    assert objectlog.iso_seconds("2026-10-29T12:00:00+02:00") == objectlog.iso_seconds("2026-10-29T10:00:00Z") and objectlog.iso_seconds("later") is None
    assert objectlog.access_seconds("29/Oct/2026:12:00:00 -0100") == objectlog.access_seconds("29/Oct/2026:13:00:00 +0000") and objectlog.access_seconds("29/Xyz/2026:12:00:00 +0000") is None
    assert objectlog._int("x", 7) == 7


def run(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "aeiou.trace", *args], capture_output=True, text=True, cwd=BUILDER, input=stdin or "")


def test_the_commands_read_an_object_log(tmp_path):
    log = tmp_path / "audit.jsonl"
    log.write_text(MINIO)
    r = run("metrics", str(log), "--root", "a-bucket/files")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["source"] == "minio-audit" and doc["roots"] == ["a-bucket/files"] and doc["total"]["counts"]["read"] == 2 and doc["contexts"] == 4
    r = run("metrics", str(log))
    assert r.returncode == 2 and "--root" in r.stderr
    r = run("metrics", str(log), "--root", "a", "--root", "b")
    assert r.returncode == 1 and "one --root" in r.stderr
    out = tmp_path / "t.jsonl"
    r = run("export", str(log), "--root", "a-bucket/files", "-o", str(out))
    assert r.returncode == 0, r.stderr
    assert "minio-audit" in out.read_text().splitlines()[0]
    r = run("digest", str(out))
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["source"] == "trace" and d["root"] == "a-bucket/files"
    r = run("digest", str(log), "--root", "a-bucket/files")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["source"] == "minio-audit" and d["notes"]["lanes"].startswith("one per client host")
    s3 = tmp_path / "access.log"
    s3.write_text(S3)
    r = run("metrics", str(s3), "--root", "amzn-bucket", "--metrics-block", "4096")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["source"] == "s3-access"
    r = run("digest", str(s3), "--root", "amzn-bucket", "--no-questions", "--text")
    assert r.returncode == 0 and "digest of" in r.stdout
    # an object log with nothing under the root
    r = run("metrics", str(log), "--root", "nowhere")
    assert r.returncode == 1 and "no request under --root nowhere" in r.stderr


def test_an_object_log_digests_in_process():
    an = digest.analyze(MINIO.splitlines(True), "a-bucket/files")
    assert an.source == "minio-audit" and an.root == "a-bucket/files"
    doc = digest.document(an)
    assert doc["source"] == "minio-audit" and len(doc["templates"]) >= 2
    with pytest.raises(BuildError, match=r"needs --root BUCKET\[/PREFIX\]"):
        digest.analyze(MINIO.splitlines(True), None)
    with pytest.raises(BuildError, match="no request under --root"):
        digest.analyze(MINIO.splitlines(True), "nowhere")
    m = trace.metrics_of_events(*objectlog.export_of(S3.splitlines(True), "amzn-bucket", "s3-access"))
    assert m["source"] == "s3-access" and m["total"]["counts"]["read"] == 2
