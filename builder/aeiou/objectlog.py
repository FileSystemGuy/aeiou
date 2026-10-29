"""Object-store request logs read as traces (`DESIGN_REVIEW.md` §3.88 step 5). An strace never
shows an application's object requests (TLS, or another process), but the store's own log does:
one line per request with its time, client, operation, bucket, key, bytes, status, and duration.
This module turns such a log into the exporter's event stream, the same header and lines
`aeiou-trace export` writes for an strace, so `metrics`, `digest`, and the runner's `trace` node
read it unchanged. A request becomes the calls the runner's object engine issues for it
(`runner/REFERENCE.md` §16): a GET is an open, one read (at the Range's start when the log has
it), and a close; a PUT an open that creates, one write, and a close; a HEAD a stat; a DELETE an
unlink; a listing an open of the prefix, a readdir, and a close; a multipart upload an open at
its start, a write per part, and a close at its completion. A lane is a client host, since the
log knows no thread. The root is `BUCKET[/PREFIX]`; keys outside it are left out and counted.

Two formats, told apart by the first line: MinIO's audit log (JSON lines with an `api` object:
`name`, `bucket`, `object`, `statusCode`, `rx`, `tx`, `timeToResponseInNS`; `time` is taken as
when the response completed, since the entry is written then, so the request starts
`timeToResponse` earlier; the `Range` request header gives a GET's offset) and Amazon S3's
server access log (space-delimited: bucket owner, bucket, `[time]` to the second, remote IP,
requester, request id, `REST.<METHOD>.<RESOURCE>`, key, "request URI", status, error, bytes
sent, object size, total time in ms, ...; it carries no Range, so a GET that sent fewer bytes
than the object holds is read from offset 0 and counted under `ranges_unknown`). Ceph's RGW ops
log and a lab proxy's log are not read yet (PROJECT_BRIEF.md item 49)."""
from __future__ import annotations

import calendar
import json
import re
import urllib.parse
from collections import Counter

from .nodes import BuildError
from .trace import Exporter

KINDS = ("minio-audit", "s3-access")
_S3_LINE = re.compile(r'^(\S+) (\S+) \[([^\]]+)\] (\S+) (\S+) (\S+) (\S+) (\S+) "([^"]*)" (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) ')
_S3_FIRST = re.compile(r"^[0-9a-f]{64} \S+ \[\d\d/[A-Z][a-z]{2}/\d{4}:\d\d:\d\d:\d\d [+-]\d{4}\] ")
_ISO = re.compile(r"^(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)?$")
_RANGE = re.compile(r"^bytes=(\d+)-(\d*)$")
_MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
_ERRNO = {404: "ENOENT", 403: "EACCES", 409: "EEXIST", 412: "EEXIST"}

# MinIO API names and S3 access-log operations, to what the engine would have done
_GET = {"GetObject", "REST.GET.OBJECT"}
_PUT = {"PutObject", "REST.PUT.OBJECT"}
_HEAD = {"HeadObject", "REST.HEAD.OBJECT"}
_DELETE = {"DeleteObject", "REST.DELETE.OBJECT"}
_LIST = {"ListObjectsV2", "ListObjectsV1", "ListObjects", "REST.GET.BUCKET"}
_MP_NEW = {"NewMultipartUpload", "REST.POST.UPLOADS"}
_MP_PART = {"PutObjectPart", "REST.PUT.PART"}
_MP_DONE = {"CompleteMultipartUpload", "REST.POST.UPLOAD"}
_MP_ABORT = {"AbortMultipartUpload", "REST.DELETE.UPLOAD"}


def source_of(first: str) -> str | None:
    """The log format of a first line, or None when it is not an object log."""
    s = first.strip()
    if s.startswith("{") and '"api"' in s:
        try:
            doc = json.loads(s)
        except ValueError:
            return None
        return "minio-audit" if isinstance(doc, dict) and isinstance(doc.get("api"), dict) else None
    return "s3-access" if _S3_FIRST.match(s) else None


def iso_seconds(text: str) -> float | None:
    """Seconds since the epoch of an RFC 3339 time, nanoseconds kept, or None."""
    m = _ISO.match(text.strip())
    if not m:
        return None
    y, mo, d, h, mi, s, frac, tz = m.groups()
    t = calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s), 0, 0, 0))
    if frac:
        t += int(frac[:9].ljust(9, "0")) / 1e9
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        hh, mm = tz[1:3], tz[-2:]
        t -= sign * (int(hh) * 3600 + int(mm) * 60)
    return t


def access_seconds(text: str) -> float | None:
    """Seconds since the epoch of an access log's `06/Feb/2019:00:00:38 +0000`."""
    m = re.match(r"^(\d\d)/([A-Z][a-z]{2})/(\d{4}):(\d\d):(\d\d):(\d\d) ([+-])(\d\d)(\d\d)$", text.strip())
    if not m or m.group(2) not in _MONTHS:
        return None
    d, mon, y, h, mi, s, sign, oh, om = m.groups()
    t = calendar.timegm((int(y), _MONTHS[mon], int(d), int(h), int(mi), int(s), 0, 0, 0))
    return t - (1 if sign == "+" else -1) * (int(oh) * 3600 + int(om) * 60)


class Request:
    """One request, whatever the log: what the converter needs of it."""
    __slots__ = ("op", "bucket", "key", "status", "rx", "tx", "size", "start", "end", "host", "offset", "length", "prefix", "upload")

    def __init__(self, op, bucket, key, status, rx, tx, size, start, end, host, offset=None, length=None, prefix=None, upload=None):
        self.op, self.bucket, self.key, self.status = op, bucket, key, status
        self.rx, self.tx, self.size, self.start, self.end, self.host = rx, tx, size, start, end, host
        self.offset, self.length, self.prefix, self.upload = offset, length, prefix, upload


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def minio_request(doc: dict, no: int) -> Request:
    api = doc.get("api") or {}
    if not isinstance(api, dict) or not isinstance(api.get("name"), str):
        raise BuildError(f"line {no}: no api.name")
    end = iso_seconds(doc.get("time", "")) if isinstance(doc.get("time"), str) else None
    ns = _int(api.get("timeToResponseInNS"), 0)
    start = end - ns / 1e9 if end is not None else None
    headers = doc.get("requestHeader") or {}
    query = doc.get("requestQuery") or {}
    offset = length = None
    rng = headers.get("Range") if isinstance(headers, dict) else None
    if isinstance(rng, str) and (m := _RANGE.match(rng.strip())):
        offset = int(m.group(1))
        length = int(m.group(2)) - offset + 1 if m.group(2) else None
    prefix = query.get("prefix") if isinstance(query, dict) else None
    upload = query.get("uploadId") if isinstance(query, dict) else None
    return Request(api["name"], api.get("bucket") or "", api.get("object") or "", _int(api.get("statusCode"), 0), _int(api.get("rx")), _int(api.get("tx")), None,
                   start, end, str(doc.get("remotehost") or "?"), offset, length, prefix if isinstance(prefix, str) else None, upload if isinstance(upload, str) else None)


def s3_request(line: str, no: int) -> Request | None:
    m = _S3_LINE.match(line)
    if not m:
        raise BuildError(f"line {no}: not an S3 server access log record")
    _, bucket, when, ip, _, _, op, key, uri, status, _, sent, size, total, _ = m.groups()
    start = access_seconds(when)
    end = start + _int(total) / 1000 if start is not None and total != "-" else start
    key = urllib.parse.unquote(key).lstrip("/") if key != "-" else ""
    q = urllib.parse.urlparse(uri.split(" ")[1] if " " in uri else uri).query
    params = urllib.parse.parse_qs(q)
    prefix = params.get("prefix", [None])[0]
    upload = params.get("uploadId", [None])[0]
    sent_n = _int(sent) if sent != "-" else 0
    size_n = _int(size) if size != "-" else None
    rx = size_n or 0 if op in _PUT or op in _MP_PART else 0
    return Request(op, bucket, key, _int(status), rx, sent_n, size_n, start, end, ip, None, None, prefix, upload)


def requests_of(lines, kind: str):
    for no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if kind == "minio-audit":
            try:
                doc = json.loads(line)
            except ValueError as e:
                raise BuildError(f"line {no}: {e}") from None
            if not isinstance(doc, dict):
                raise BuildError(f"line {no}: not a JSON object")
            yield minio_request(doc, no)
        else:
            yield s3_request(line, no)


def export_of(lines, root: str, kind: str) -> tuple[dict, list[dict]]:
    """The exporter's header and events of an object log, for `root` = `BUCKET[/PREFIX]`."""
    if kind not in KINDS:
        raise BuildError(f"not an object log format: {kind}")
    root = root.strip("/")
    if not root:
        raise BuildError("an object log needs --root BUCKET[/PREFIX]")
    ex = Exporter(root)
    notes: Counter = Counter()
    hosts: dict[str, int] = {}
    uploads: dict[str, list] = {}    # the object's full key -> [oid, bytes so far] of the upload in progress (the id is in the start's response, which no log has)
    n = 0
    for r in requests_of(lines, kind):
        n += 1
        if r.start is None:
            notes["requests_without_time"] += 1
        tid = hosts.setdefault(r.host, len(hosts))
        when = (tid, r.start, r.end)
        full = f"{r.bucket}/{r.key}" if r.key else r.bucket
        err = None if 200 <= r.status < 300 else _ERRNO.get(r.status, "EIO")
        if r.op in _LIST:
            path = f"{r.bucket}/{r.prefix.strip('/')}" if r.prefix else r.bucket
            if not _under(path, root):
                notes["requests_outside_root"] += 1
                continue
            oid = ex.open(when, path, "O_RDONLY|O_DIRECTORY", None, err)
            if err is None and oid is not None:
                ex.op(when, "readdir", oid, None)
                ex.op(when, "close", oid, None)
            continue
        if r.op in _MP_DONE or r.op in _MP_ABORT:
            up = uploads.pop(full, None)
            if up is None:
                notes["multipart_without_start" if _under(full, root) else "requests_outside_root"] += 1
                continue
            ex.op(when, "close", up[0], None)
            if r.op in _MP_ABORT:
                ex._event(when, "unlink", path=full)
            continue
        if r.op in _MP_PART:
            up = uploads.get(full)
            if up is None:
                notes["multipart_without_start" if _under(full, root) else "requests_outside_root"] += 1
                continue
            ex.data(when, "write", up[0], up[1], r.rx, False, r.rx, err)   # parts in the order logged, each after the last
            up[1] += r.rx
            continue
        if not r.key or not _under(full, root):
            if r.op in _GET | _PUT | _HEAD | _DELETE | _MP_NEW:
                notes["requests_outside_root"] += 1
            else:
                notes[f"requests_not_exported:{r.op}"] += 1
            continue
        if r.op in _GET:
            oid = ex.open(when, full, "O_RDONLY", None, err)
            if err is None and oid is not None:
                off = r.offset if r.offset is not None else 0
                length = r.length if r.length is not None else r.tx
                if r.offset is None and r.size is not None and r.tx < r.size:
                    notes["ranges_unknown"] += 1
                ex.data(when, "read", oid, off, length, r.offset is not None, r.tx, None)
                ex.op(when, "close", oid, None)
        elif r.op in _PUT:
            oid = ex.open(when, full, "O_WRONLY|O_CREAT|O_TRUNC", "644", err)
            if err is None and oid is not None:
                ex.data(when, "write", oid, 0, r.rx, False, r.rx, None)
                ex.op(when, "close", oid, None)
        elif r.op in _HEAD:
            ex._event(when, "stat", path=full, ret=err)
        elif r.op in _DELETE:
            ex._event(when, "unlink", path=full, ret=err)
        elif r.op in _MP_NEW:
            oid = ex.open(when, full, "O_WRONLY|O_CREAT|O_TRUNC", "644", err)
            if err is None and oid is not None:
                uploads[full] = [oid, 0]
        else:
            notes[f"requests_not_exported:{r.op}"] += 1
    for oid, _ in uploads.values():
        notes["multipart_not_completed"] += 1
        ex.op((0, ex.t0, ex.t0), "close", oid, None)
    if kind == "minio-audit":
        notes["time_is_response_end"] = "a MinIO audit entry's time is taken as when the response completed; the request starts timeToResponse earlier"
    notes["lanes"] = "one per client host; the log knows no thread"
    header, events = ex.finish(root, {})
    header["source"] = kind
    header["requests"] = n
    for k, v in notes.items():
        header["notes"].setdefault(k, v)
    header["notes"] = dict(sorted(header["notes"].items()))
    return header, events


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")
