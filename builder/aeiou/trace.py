"""`aeiou-trace`: the locality metrics of a real application, from an `strace` of it.

`aeiou dry-run --metrics` computes reuse distance, sequential runs, popularity, request
sizes, fan-out, depth, and the read/write mix on an abstract's op stream
(`runner/README.md` §10). This computes the same numbers, with the same definitions and in
the same JSON form (`aeiou_metrics: 1`), from a trace of the application the abstract
models, so the two can be compared (`GRAMMAR_OPTIONS.md` §5.4):

    strace -f -ttt -T -yy -e trace=%file,%desc,%process -o trace.txt <command>
    aeiou-trace metrics trace.txt --root /mnt/data -o trace.metrics.json
    aeiou dry-run x.ast.json --gpus 1 --metrics-json abstract.metrics.json
    aeiou-trace compare trace.metrics.json abstract.metrics.json
    aeiou-trace compare trace.metrics.json abstract.metrics.json --judge --self other-seed.metrics.json

`--judge` holds each row to the tolerance of its class (`TOLERANCES` below, `builder/README.md`
§7), raised by what the abstract differs from itself by at other seeds.

What differs from the dry-run side, because a trace is not an abstract:

- **Which calls count.** Only calls on paths under `--root` (the `-yy` annotation of the
  descriptor, or the path argument). Everything else a process does is not the workload.
- **Order.** A trace has a real order: calls are taken in the order they returned (the
  order of the trace's lines). The dry run uses a round-robin of an instance's sub-actors.
- **Instance.** The whole trace is one instance unless `--instance-root PID` names several
  process trees. Reuse distance is taken per instance, as on the dry-run side.
- **Sequential context.** A thread (a tid of the trace), where the dry run has a sub-actor.
- **Offsets** of `read` and `write` come from the descriptor's tracked position (`open`,
  `lseek`, and the bytes each call moved); `pread`/`pwrite` carry their own.
- **Fan-out and depth** exist only where the trace shows a fork: the requests of one
  `io_submit` are a fan-out, and consecutive `io_submit`s of a thread, with no other
  counted call of its own between them and at most `--chain-gap-us` of think time, are a
  chain. Without `--chain-gap-us` no depth is reported: there is no default think time.

Not visible to `strace`, and so not counted: `io_uring` submissions and page faults on a
mapping. Standard library only; the trace is read once, in a stream.

`aeiou-trace export` writes the same calls as a trace file for the runner's `trace` node
(`DESIGN_REVIEW.md` §3.58): JSON Lines, a header and one op per line in the order the calls
returned, each on a lane (a traced task) and naming its open by id, so that
`aeiou dry-run --metrics` of an abstract that is one `trace` node computes these same
numbers. `aeiou-trace metrics` reads an exported file too (`TraceReader`).
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import re
import sys
from collections import Counter

FORMAT = 1
MASK = (1 << 64) - 1
MAX_ENTRIES = 1 << 26
KINDS = ("read", "write")
R, W = 0, 1

# ---------------------------------------------------------------- histograms


def mix64(z: int) -> int:
    """`rng::mix64` of the runner (SplitMix64's finalizer)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


class LogHist:
    """Quarter-octave buckets, as `metrics::LogHist`: a bucket's lower bound keeps the
    value's top three bits."""

    __slots__ = ("n", "sum", "max", "buckets")

    def __init__(self):
        self.n = 0
        self.sum = 0
        self.max = 0
        self.buckets: dict[int, int] = {}

    @staticmethod
    def lower(v: int) -> int:
        if v < 4:
            return v
        e = v.bit_length() - 1
        return (v >> (e - 2)) << (e - 2)

    def add(self, v: int, weight: int = 1):
        self.n += weight
        self.sum += v * weight
        if v > self.max:
            self.max = v
        b = self.lower(v)
        self.buckets[b] = self.buckets.get(b, 0) + weight

    def out(self) -> dict:
        return {
            "n": self.n,
            "mean": self.sum / self.n if self.n else 0.0,
            "max": self.max,
            "buckets": [[b, w] for b, w in sorted(self.buckets.items())],
        }


class Instance:
    """One instance's access history: Bennett and Kruskal's stack distance, a Fenwick tree
    with a mark at the last access time of every block (as `metrics::Instance`)."""

    def __init__(self):
        self.last: dict[int, list] = {}  # key -> [time, kind, count]
        self.tree: list[int] = []
        self.now = 0

    def _add(self, i: int, d: int):
        tree, n = self.tree, len(self.tree)
        i += 1
        while i <= n:
            tree[i - 1] += d
            i += i & -i

    def _prefix(self, i: int) -> int:
        tree, s = self.tree, 0
        i += 1
        while i > 0:
            s += tree[i - 1]
            i -= i & -i
        return s

    def _compact(self):
        live = sorted(self.last.values(), key=lambda s: s[0])
        cap = 1024
        while cap < 2 * len(live):
            cap *= 2
        for t, slot in enumerate(live):
            slot[0] = t
        n = len(live)
        self.tree = [min(max(n - (i - (i & -i)), 0), i & -i) for i in range(1, cap + 1)]
        self.now = n

    def touch(self, key: int, kind: int):
        """The distinct blocks since this block's last access (itself included) and that
        access's kind, or `None` on first touch."""
        if self.now == len(self.tree):
            self._compact()
        now = self.now
        slot = self.last.get(key)
        out = None
        if slot is None:
            self.last[key] = [now, kind, 1]
        else:
            t, prev = slot[0], slot[1]
            out = (len(self.last) - self._prefix(t) + 1, prev)
            self._add(t, -1)
            slot[0], slot[1] = now, kind
            slot[2] += 1
        self._add(now, 1)
        self.now += 1
        return out


def popularity(counts: dict[int, int], sample: int) -> dict:
    """The rank-frequency curve as counts of counts, as `metrics::Popularity`."""
    cc = Counter(counts.values())
    rows = sorted(cc.items(), reverse=True)
    items = sum(n for _, n in rows)
    total = sum(c * n for c, n in rows)

    def share(frac: float) -> float:
        if total == 0:
            return 0.0
        left, got = items * frac, 0.0
        for c, n in rows:
            take = min(left, n)
            got += take * c
            left -= take
            if left <= 0:
                break
        return got / total

    return {
        "distinct": items * sample,
        "accesses": total * sample,
        "max": rows[0][0] if rows else 0,
        "top_0_1_pct": share(0.001),
        "top_1_pct": share(0.01),
        "top_10_pct": share(0.1),
        "counts": [[c, n] for c, n in rows],
    }


class Metrics:
    """The accumulated metrics of a trace; the definitions are `runner/README.md` §10's."""

    def __init__(self, block: int = 4096, sample: int = 1):
        self.block = max(block, 1)
        self.sample = max(sample, 1)
        self.counts: Counter = Counter()
        self.bytes = [0, 0]
        self.request = [LogHist(), LogHist()]
        self.run_bytes = [LogHist(), LogHist()]
        self.run_ops = [LogHist(), LogHist()]
        self.multi_op_bytes = [0, 0]
        self.reuse = [[LogHist(), LogHist()], [LogHist(), LogHist()]]  # [previous][this]
        self.first = [0, 0]
        self.fan_out: Counter = Counter()
        self.depth: Counter = Counter()
        self.blocks: dict[int, int] = {}
        self.objects: dict[int, int] = {}
        self.instances: dict[int, Instance] = {}
        self.path_ids: dict[str, int] = {}
        # open runs: context -> {(path id, kind): [end, bytes, ops]}
        self.runs: dict[int, dict] = {}

    def path_id(self, path: str) -> int:
        i = self.path_ids.get(path)
        if i is None:
            i = self.path_ids[path] = len(self.path_ids)
        return i

    def op(self, kind: str):
        self.counts[kind] += 1

    def data(self, kind: int, instance: int, context: int, path: str, offset: int, length: int, nbytes: int):
        """One `read` or `write`: `length` requested at `offset`, `nbytes` moved."""
        self.counts[KINDS[kind]] += 1
        self.request[kind].add(max(length, 0))
        pid = self.path_id(path)
        n = self.sample
        if n == 1 or mix64(pid) % n == 0:
            self.objects[pid] = self.objects.get(pid, 0) + 1
        self.bytes[kind] += max(nbytes, 0)
        if nbytes <= 0 or offset < 0:
            return
        b = self.block
        inst = self.instances.get(instance)
        if inst is None:
            inst = self.instances[instance] = Instance()
        for blk in range(offset // b, (offset + nbytes - 1) // b + 1):
            key = (pid << 44) | blk
            if n != 1:
                key = mix64(key & MASK)
                if key % n:
                    continue
            hit = inst.touch(key, kind)
            if hit is None:
                self.first[kind] += n
            else:
                self.reuse[hit[1]][kind].add(hit[0] * n * b, n)
        if len(inst.last) > MAX_ENTRIES:
            raise SystemExit(f"aeiou-trace: more than {MAX_ENTRIES} distinct blocks in one instance; use --sample N or a larger --block")
        runs = self.runs.setdefault(context, {})
        run = runs.get((pid, kind))
        if run is not None and run[0] != offset:
            self._run_done(kind, run)
            run = None
        if run is None:
            run = runs[(pid, kind)] = [offset, 0, 0]
        run[0] = offset + nbytes
        run[1] += nbytes
        run[2] += 1

    def _run_done(self, kind: int, run: list):
        self.run_bytes[kind].add(run[1])
        self.run_ops[kind].add(run[2])
        if run[2] >= 2:
            self.multi_op_bytes[kind] += run[1]

    def close(self, context: int, path: str):
        """A `close` of `path` by `context` ends that context's runs on it."""
        runs = self.runs.get(context)
        pid = self.path_ids.get(path)
        if runs and pid is not None:
            for kind in (R, W):
                run = runs.pop((pid, kind), None)
                if run is not None:
                    self._run_done(kind, run)

    def end_context(self, context: int):
        for (_, kind), run in self.runs.pop(context, {}).items():
            self._run_done(kind, run)

    def finish(self):
        for c in list(self.runs):
            self.end_context(c)
        for inst in self.instances.values():
            for key, slot in inst.last.items():
                self.blocks[key] = self.blocks.get(key, 0) + slot[2]
            inst.last = {}

    def out(self) -> dict:
        """The `total` of an `aeiou_metrics: 1` document (`metrics::MetricsOut`)."""
        runs = lambda k: {"bytes": self.run_bytes[k].out(), "ops": self.run_ops[k].out(), "multi_op_bytes": self.multi_op_bytes[k]}
        return {
            "ops": sum(self.counts.values()),
            "counts": dict(sorted(self.counts.items())),
            "bytes_read": self.bytes[R],
            "bytes_written": self.bytes[W],
            "request_size": {"read": self.request[R].out(), "write": self.request[W].out()},
            "run_length": {"read": runs(R), "write": runs(W)},
            "reuse_distance_bytes": {
                "first_touch": {"read": self.first[R], "write": self.first[W]},
                "read_after_read": self.reuse[R][R].out(),
                "read_after_write": self.reuse[W][R].out(),
                "write_after_read": self.reuse[R][W].out(),
                "write_after_write": self.reuse[W][W].out(),
            },
            "popularity_blocks": popularity(self.blocks, self.sample),
            "popularity_objects": popularity(self.objects, self.sample),
            "fan_out": {str(k): v for k, v in sorted(self.fan_out.items())},
            "depth": {str(k): v for k, v in sorted(self.depth.items())},
        }


# ---------------------------------------------------------------- strace lines

_PREFIX = re.compile(r"^(?:\[pid\s+(\d+)\]\s+|(\d+)\s+)?(?:(\d+\.\d+|\d\d:\d\d:\d\d(?:\.\d+)?)\s+)?(.*)$", re.S)
_RESUMED = re.compile(r"^<\.\.\. (\w+) resumed>\s*(.*)$", re.S)
_RET = re.compile(r"^(-?\d+|0x[0-9a-f]+|\?)")
_DUR = re.compile(r"<(\d+\.\d+)>$")
_ERRNO = re.compile(r"^\s*(E[A-Z0-9]+)")
_FD = re.compile(r"^(-?\d+|AT_FDCWD)(?:<(.*)>)?$", re.S)
_IOV_LEN = re.compile(r"iov_len=(\d+)")
_IOCB = re.compile(r"aio_data=(\w+), aio_lio_opcode=IOCB_CMD_(\w+).*?aio_fildes=(\d+)(?:<(.*?)>)?, .*?aio_nbytes=(\d+), aio_offset=(-?\d+)")
_EVENT = re.compile(r"\{data=(\w+), obj=\w+, res=(-?\d+)")
UNFINISHED = "<unfinished ...>"


def split_args(s: str) -> list[str]:
    """The top-level arguments of a call: commas inside strings, brackets, and the `<path>`
    annotation of a descriptor do not split."""
    out, depth, angle, quote, start, i, n = [], 0, 0, False, 0, 0, len(s)
    while i < n:
        c = s[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == '"':
                quote = False
        elif angle:
            if c == "<":
                angle += 1
            elif c == ">":
                angle -= 1
        elif c == '"':
            quote = True
        elif c == "<" and i > start and (s[i - 1].isdigit() or s[i - 1] == "D"):
            angle = 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
        i += 1
    tail = s[start:].strip()
    if tail or out:
        out.append(tail)
    return out


def fd_arg(tok: str):
    """`3</path>` → `(3, "/path")`; `AT_FDCWD</cwd>` → `(None, "/cwd")`; no annotation → `None`."""
    m = _FD.match(tok)
    if not m:
        return None, None
    fd = None if m.group(1) == "AT_FDCWD" else int(m.group(1))
    path = m.group(2)
    if path is not None:
        if path.endswith(">") and "<" in path:  # `/dev/null<char 1:3>`
            path = path[: path.rindex("<")]
        if path.endswith(" (deleted)"):
            path = path[: -len(" (deleted)")]
    return fd, path


def c_string(tok: str):
    """A quoted string argument, unescaped; `None` if `tok` is not one."""
    if not tok.startswith('"'):
        return None
    end = tok.rfind('"')
    if end <= 0:
        return None
    raw = tok[1:end]
    if "\\" not in raw:
        return raw
    try:
        return raw.encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw


def _seconds(ts: str) -> float:
    if ":" in ts:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(ts)


class File:
    """An open file description: descriptors made by `dup` and inherited over `fork` share it."""

    __slots__ = ("path", "pos", "append", "listed", "oid")

    def __init__(self, path, pos=0, append=False):
        self.path = path
        self.pos = pos
        self.append = append
        # a directory stream being read: its `getdents` calls are one `readdir` op together
        self.listed = False
        # the exporter's open id, when this description was opened under the root
        self.oid = None


OPENS = {"open", "openat", "openat2", "creat"}
READS = {"read", "pread64", "readv", "preadv", "preadv2"}
WRITES = {"write", "pwrite64", "writev", "pwritev", "pwritev2"}
CLONES = {"clone", "clone3", "fork", "vfork"}
DUPS = {"dup", "dup2", "dup3", "fcntl"}
# descriptor calls that are one op of the abstract language each
FD_OPS = {
    "fstat": "fstat", "fsync": "fsync", "fdatasync": "fdatasync", "ftruncate": "ftruncate",
    "fallocate": "fallocate", "getdents64": "readdir", "getdents": "readdir",
    "fadvise64": "fadvise", "ioctl": "ioctl",
}
# path calls: the op, and where the path is: (dirfd index or None, path index)
PATH_OPS = {
    "stat": ("stat", None, 0), "lstat": ("stat", None, 0), "newfstatat": ("stat", 0, 1), "statx": ("stat", 0, 1),
    "unlink": ("unlink", None, 0), "unlinkat": ("unlink", 0, 1), "truncate": ("ftruncate", None, 0),
    "mkdir": ("mkdir", None, 0), "mkdirat": ("mkdir", 0, 1), "rmdir": ("rmdir", None, 0),
    "rename": ("rename", None, 0), "renameat": ("rename", 0, 1), "renameat2": ("rename", 0, 1),
}
HANDLED = OPENS | READS | WRITES | CLONES | DUPS | set(FD_OPS) | set(PATH_OPS) | {"close", "lseek", "io_submit", "io_getevents", "io_pgetevents", "chdir", "fchdir", "mmap"}


class Tracer:
    """Reads `strace` lines and feeds a `Metrics`. Descriptor tables follow `clone` when the
    trace has it (`-e trace=…,%process`); otherwise a thread that uses a descriptor it never
    opened joins the table of the one that did (the `-yy` path decides which)."""

    def __init__(self, metrics: Metrics, roots: list[str], exclude: list[str] = (), cwd: str | None = None, instance_roots: list[int] = (), chain_gap_us: float | None = None):
        self.m = metrics
        self.roots = [os.path.normpath(r) for r in roots]
        self.exclude = list(exclude)
        self.cwd = cwd
        self.instance_roots = set(instance_roots)
        self.chain_gap = None if chain_gap_us is None else chain_gap_us / 1e6
        self.tables: dict[int, dict[int, File]] = {}
        self.parent: dict[int, int] = {}
        self.instance_of: dict[int, int | None] = {}
        self.cwds: dict[int, str] = {}
        self.pending: dict[int, tuple] = {}
        self.chain: dict[int, list] = {}  # tid -> [length, time the last round ended]
        # AIO requests submitted and not yet reaped: (context, aio_data) -> the requests, oldest first
        self.aio: dict[tuple, list] = {}
        self.clock = 0.0
        self.relative = False
        self.notes: Counter = Counter()
        self.other: Counter = Counter()
        self.tids: set[int] = set()
        self._under: dict[str, str | None] = {}
        # `aeiou-trace export`: an `Exporter` that gets every counted call
        self.export: Exporter | None = None

    # ---- paths

    def under(self, path: str | None) -> str | None:
        """The path as counted (normalized), or `None` when it is outside the roots or excluded."""
        if path is None:
            return None
        hit = self._under.get(path, 0)
        if hit != 0:
            return hit
        p = os.path.normpath(path)
        out = None
        for r in self.roots:
            if p == r or p.startswith(r + "/"):
                rel = p[len(r) + 1 :]
                if not any(fnmatch.fnmatchcase(rel, g) for g in self.exclude):
                    out = p
                break
        if len(self._under) < 1 << 20:
            self._under[path] = out
        return out

    def resolve(self, tid: int, dirfd_tok: str | None, path: str | None) -> str | None:
        if path is None:
            return None
        if path.startswith("/"):
            return path
        base = None
        if dirfd_tok is not None:
            fd, base = fd_arg(dirfd_tok)
            if base is None and fd is not None:
                f = self.table(tid).get(fd)
                base = f.path if f else None
        if base is None:
            base = self.cwds.get(tid) or self.cwd
        if base is None:
            self.notes["relative_paths_skipped"] += 1
            return None
        return os.path.join(base, path)

    # ---- processes

    def table(self, tid: int) -> dict[int, File]:
        t = self.tables.get(tid)
        if t is None:
            t = self._adopt(tid)
        return t

    def _adopt(self, tid: int) -> dict[int, File]:
        """A tid seen before its `clone` returned (or with no `clone` in the trace): the child
        of the one unfinished `clone`, if there is exactly one."""
        waiting = [(p, pend) for p, pend in self.pending.items() if pend[0] in CLONES]
        if len(waiting) == 1:
            p, pend = waiting[0]
            return self._child(p, tid, "CLONE_FILES" in pend[1])
        if self.tables:
            self.notes["threads_without_clone"] += 1
        t = self.tables[tid] = {}
        return t

    def _child(self, parent: int, child: int, shares: bool) -> dict[int, File]:
        ptable = self.table(parent)
        old = self.tables.get(child)
        if old is not None and self.parent.get(child) == parent:
            return old
        t = ptable if shares else dict(ptable)
        if old:
            t.update(old)
        self.tables[child] = t
        self.parent[child] = parent
        self.instance_of.pop(child, None)
        if parent in self.cwds:
            self.cwds.setdefault(child, self.cwds[parent])
        return t

    def instance(self, tid: int) -> int | None:
        """The instance `tid` belongs to: 0 when the trace is one instance, else the
        `--instance-root` at or above it, or `None` when there is none."""
        if not self.instance_roots:
            return 0
        if tid in self.instance_of:
            return self.instance_of[tid]
        t, seen = tid, set()
        while t is not None and t not in self.instance_roots and t not in seen:
            seen.add(t)
            t = self.parent.get(t)
        out = t if t in self.instance_roots else None
        self.instance_of[tid] = out
        return out

    def file(self, tid: int, fd: int | None, path: str | None) -> File | None:
        """The open file behind `fd` as `tid` sees it. The `-yy` path wins over a stale table."""
        if fd is None:
            return None
        table = self.table(tid)
        f = table.get(fd)
        if f is not None and (path is None or f.path == path):
            return f
        if path is None:
            return None
        # not this thread's open: a thread sharing another's table, traced without `clone`
        for other in self.tables.values():
            g = other.get(fd)
            if g is not None and g.path == path and other is not table:
                if not table:
                    for t, tab in list(self.tables.items()):
                        if tab is table:
                            self.tables[t] = other
                else:
                    table[fd] = g
                return g
        f = table[fd] = File(path)
        if self.under(path):
            # opened before the trace began: its position is taken as 0
            self.notes["descriptors_without_open"] += 1
        return f

    # ---- the stream

    def feed(self, lines):
        for line in lines:
            self.line(line.rstrip("\n"))

    def line(self, line: str):
        m = _PREFIX.match(line.lstrip())
        if not m:
            return
        tid = int(m.group(1) or m.group(2) or 0)
        ts = None
        if m.group(3):
            ts = _seconds(m.group(3))
            if ts < 1e6 and ":" not in m.group(3):  # `-r`: relative to the previous line
                self.clock += ts
                ts = self.clock
        body = m.group(4)
        if body.startswith("+++"):
            self.m.end_context(tid)
            self._end_chain(tid)
            return
        if body.startswith("---") or not body:
            return
        issued = ts
        r = _RESUMED.match(body)
        if r:
            pend = self.pending.pop(tid, None)
            if pend is None or pend[0] != r.group(1):
                return
            body, issued = pend[1] + r.group(2), pend[2]
        elif body.endswith(UNFINISHED):
            name = body.split("(", 1)[0]
            if name in HANDLED:
                self.pending[tid] = (name, body[: -len(UNFINISHED)], ts)
            return
        i = body.find("(")
        if i <= 0:
            return
        name = body[:i]
        if name not in HANDLED:
            return
        k = body.rfind(") = ")
        if k < 0:
            return
        ret_s = body[k + 4 :]
        rm = _RET.match(ret_s)
        if not rm or rm.group(1) == "?":
            return
        ret = int(rm.group(1), 0)
        ret_path = None
        rest = ret_s[rm.end() :]
        em = _ERRNO.match(rest) if ret < 0 else None
        err = em.group(1) if em else None
        if rest.startswith("<"):
            _, ret_path = fd_arg(rm.group(1) + rest.split(" <")[0] if " <" in rest else rm.group(1) + rest)
        d = _DUR.search(ret_s)
        done = ts if r else (issued + float(d.group(1)) if issued is not None and d else issued)
        self.tids.add(tid)
        self.call(tid, name, split_args(body[i + 1 : k]), ret, ret_path, issued, done, err)

    def _count(self, tid: int, kind: str):
        self.m.op(kind)
        self._end_chain(tid)

    def _end_chain(self, tid: int):
        c = self.chain.pop(tid, None)
        if c and self.chain_gap is not None:
            self.m.depth[c[0]] += 1

    def call(self, tid: int, name: str, a: list[str], ret: int, ret_path: str | None, issued: float | None, done: float | None, err: str | None = None):
        m = self.m
        ex = self.export
        when = (tid, issued, done)
        if name in CLONES:
            if ret > 0:
                self._child(tid, ret, name in ("clone", "clone3") and "CLONE_FILES" in (a[0] if a else ""))
            return
        if name in ("chdir", "fchdir"):
            if ret == 0 and a:
                p = c_string(a[0]) if name == "chdir" else fd_arg(a[0])[1]
                if p:
                    self.cwds[tid] = self.resolve(tid, None, p) or p
            return
        inst = self.instance(tid)
        if inst is None:
            self.notes["calls_outside_instances"] += 1
            # the descriptor tables are still kept: a child inside an instance may inherit them
        if name in OPENS:
            pi = 1 if name in ("openat", "openat2") else 0
            if len(a) <= pi:
                return
            path = ret_path if ret >= 0 and ret_path else self.resolve(tid, a[0] if pi else None, c_string(a[pi]))
            flags = a[pi + 1] if len(a) > pi + 1 else ""
            f = None
            if ret >= 0:
                known_empty = "O_TRUNC" in flags or "O_EXCL" in flags or name == "creat"
                if "O_APPEND" in flags and not known_empty:
                    self.notes["append_to_unknown_size"] += 1
                f = self.table(tid)[ret] = File(path, 0, "O_APPEND" in flags)
            p = self.under(path)
            if inst is not None and p:
                self._count(tid, "open")
                if ex is not None:
                    mode = a[pi + 2] if len(a) > pi + 2 else None
                    oid = ex.open(when, p, flags if name != "creat" else "O_WRONLY|O_CREAT|O_TRUNC", mode, err)
                    if f is not None:
                        f.oid = oid
            return
        if name == "close":
            fd, path = fd_arg(a[0]) if a else (None, None)
            table = self.table(tid)
            f = table.get(fd)
            if path is None and f is not None:
                path = f.path
            if ret == 0 and f is not None and f.path == path:
                del table[fd]
            p = self.under(path)
            if inst is not None and p:
                self._count(tid, "close")
                m.close(tid, p)
                if ex is not None and f is not None and f.oid is not None:
                    ex.op(when, "close", f.oid, err)
            return
        if name in DUPS:
            if name == "fcntl" and not (len(a) > 1 and a[1].startswith("F_DUPFD")):
                return
            fd, path = fd_arg(a[0]) if a else (None, None)
            f = self.file(tid, fd, path)
            if ret >= 0 and f is not None:
                self.table(tid)[ret] = f
            return
        if name == "lseek":
            fd, path = fd_arg(a[0]) if a else (None, None)
            f = self.file(tid, fd, path)
            if f is not None and ret >= 0:
                f.pos = ret
                f.listed = False  # rewinddir
            if inst is not None and self.under(path or (f.path if f else None)):
                self._count(tid, "lseek")
                if ex is not None and f is not None and f.oid is not None and len(a) > 2:
                    ex.op(when, "lseek", f.oid, err, offset=int(a[1], 0), whence=a[2].replace("SEEK_", ""), ret=ret)
            return
        if name in READS or name in WRITES:
            kind = R if name in READS else W
            fd, path = fd_arg(a[0]) if a else (None, None)
            f = self.file(tid, fd, path)
            path = path or (f.path if f else None)
            vector = name.endswith("v") or name.endswith("v2")
            if vector:
                length = sum(int(x) for x in _IOV_LEN.findall(a[1])) if len(a) > 1 else 0
                off_tok = a[3] if name.startswith("p") and len(a) > 3 else None
            else:
                length = int(a[2], 0) if len(a) > 2 and a[2].lstrip("-").isdigit() else 0
                off_tok = a[3] if name.startswith("p") and len(a) > 3 else None
            positioned = off_tok is not None and off_tok.lstrip("-").isdigit() and int(off_tok) >= 0
            if positioned:
                offset = int(off_tok)
            else:
                offset = f.pos if f is not None else -1
                if f is not None and ret > 0:
                    f.pos += ret
            p = self.under(path)
            if inst is not None and p:
                self._end_chain(tid)
                if offset < 0:
                    self.notes["data_ops_without_offset"] += 1
                m.data(kind, inst, tid, p, offset, length, max(ret, 0))
                if ex is not None and f is not None and f.oid is not None and offset >= 0:
                    ex.data(when, KINDS[kind], f.oid, offset, length, positioned, ret, err)
            return
        if name == "mmap":
            # exported only (the metrics do not count a mapping, `runner/README.md` §10): the
            # mapped range as one read, since the faults inside it are not in the trace
            if ex is None or len(a) < 6 or ret < 0:
                return
            fd, path = fd_arg(a[4])
            f = self.file(tid, fd, path)
            if f is None or f.oid is None or inst is None or not self.under(path or f.path):
                return
            length, off = int(a[1], 0), int(a[5], 0)
            ex.notes["mmap_as_read"] += 1
            ex.data(when, "read", f.oid, off, length, True, length, None)
            return
        if name == "io_submit":
            got = []
            for data, op, fd, path, nbytes, off in _IOCB.findall(a[2] if len(a) > 2 else ""):
                f = self.file(tid, int(fd), path or None)
                p = self.under(path or (f.path if f else None))
                if not p:
                    continue
                if op not in ("PREAD", "PWRITE"):
                    self.notes["aio_requests_not_counted"] += 1
                    continue
                got.append((data, R if op == "PREAD" else W, p, int(off), int(nbytes)))
            if inst is None or not got or ret <= 0:
                return
            got = got[:ret]
            c = self.chain.get(tid)
            if c and self.chain_gap is not None and issued is not None and c[1] is not None and issued - c[1] > self.chain_gap:
                self._end_chain(tid)
                c = None
            if c is None:
                c = self.chain[tid] = [0, None]
            c[0] += 1
            c[1] = done
            m.fan_out[len(got)] += 1
            ctx = a[0]
            members = None
            if ex is not None:
                mem = []
                for data, op, fd, path, nbytes, off in _IOCB.findall(a[2] if len(a) > 2 else ""):
                    f = self.file(tid, int(fd), path or None)
                    if op in ("PREAD", "PWRITE") and f is not None and f.oid is not None and self.under(path or f.path):
                        mem.append(("read" if op == "PREAD" else "write", f.oid, int(off), int(nbytes)))
                members = ex.submit(when, mem[:ret]) if mem else None
            for j, (data, kind, p, off, nbytes) in enumerate(got):
                # counted when its result is reaped (`io_getevents`), with the bytes it moved
                member = members[j] if members is not None and j < len(members) else None
                self.aio.setdefault((ctx, data), []).append((kind, inst, tid, p, off, nbytes, member))
            return
        if name in ("io_getevents", "io_pgetevents"):
            c = self.chain.get(tid)
            if c:
                c[1] = done
            ctx = a[0] if a else None
            for data, res in _EVENT.findall(a[3] if len(a) > 3 else ""):
                waiting = self.aio.get((ctx, data))
                if waiting:
                    kind, i, t, p, off, nbytes, member = waiting.pop(0)
                    if not waiting:
                        del self.aio[(ctx, data)]
                    m.data(kind, i, t, p, off, nbytes, max(int(res), 0))
                    if member is not None:
                        member["ret"] = int(res) if int(res) >= 0 else _errno_name(-int(res))
            return
        if name in FD_OPS:
            fd, path = fd_arg(a[0]) if a else (None, None)
            f = self.file(tid, fd, path)
            if path is None:
                path = f.path if f else None
            if FD_OPS[name] == "readdir" and f is not None:
                # the abstract's `readdir` is one listing of a directory: the calls that
                # continue it, and the empty one that ends it, are the same op
                if f.listed:
                    return
                f.listed = True
            if inst is not None and self.under(path):
                self._count(tid, FD_OPS[name])
                if ex is not None and f is not None and f.oid is not None:
                    ex.fd_op(when, FD_OPS[name], name, f.oid, a, ret, err)
            return
        if name in PATH_OPS:
            kind, di, pi = PATH_OPS[name]
            if len(a) <= pi:
                return
            arg = c_string(a[pi])
            if name in ("statx", "newfstatat") and arg == "":
                # AT_EMPTY_PATH: the descriptor itself
                fd, path = fd_arg(a[0])
                if path is None:
                    f = self.file(tid, fd, None)
                    path = f.path if f else None
                kind = "fstat"
                f = self.file(tid, fd, path)
                if inst is not None and self.under(path):
                    self._count(tid, kind)
                    if ex is not None and f is not None and f.oid is not None:
                        ex.op(when, "fstat", f.oid, err)
                return
            path = self.resolve(tid, a[di] if di is not None else None, arg)
            if name == "unlinkat" and "AT_REMOVEDIR" in a[-1]:
                kind = "rmdir"
            p = self.under(path)
            if inst is not None and p:
                self._count(tid, kind)
                if ex is not None:
                    to = None
                    if kind == "rename":
                        j = pi + 1 if di is None else pi + 2
                        to = self.under(self.resolve(tid, a[j - 1] if di is not None else None, c_string(a[j]))) if len(a) > j else None
                    ex.path_op(when, kind, name, p, a, pi, to, err)
            return

    def finish(self):
        for tid in list(self.chain):
            self._end_chain(tid)
        # requests whose results the trace does not show (reaped from the ring in user
        # space, or `io_getevents` not traced): taken as complete
        for waiting in self.aio.values():
            for kind, i, t, p, off, nbytes, member in waiting:
                self.notes["aio_results_assumed_complete"] += 1
                self.m.data(kind, i, t, p, off, nbytes, nbytes)
        self.aio = {}
        self.m.finish()


# ---------------------------------------------------------------- export: the runner's trace file

TRACE_FORMAT = 1
# the open flags the schema knows (`OpenFlag`); anything else strace shows is dropped and counted
OPEN_FLAGS = {"O_RDONLY": "RDONLY", "O_WRONLY": "WRONLY", "O_RDWR": "RDWR", "O_CREAT": "CREAT", "O_TRUNC": "TRUNC", "O_EXCL": "EXCL",
              "O_APPEND": "APPEND", "O_CLOEXEC": "CLOEXEC", "O_DIRECTORY": "DIRECTORY", "O_DIRECT": "DIRECT", "O_SYNC": "SYNC",
              "O_DSYNC": "DSYNC", "O_NOATIME": "NOATIME", "O_NOFOLLOW": "NOFOLLOW"}
IOCTLS = {"TCGETS", "FIONREAD", "BLKGETSIZE64"}
ADVICE = {"POSIX_FADV_NORMAL": "NORMAL", "POSIX_FADV_RANDOM": "RANDOM", "POSIX_FADV_SEQUENTIAL": "SEQUENTIAL",
          "POSIX_FADV_WILLNEED": "WILLNEED", "POSIX_FADV_DONTNEED": "DONTNEED", "POSIX_FADV_NOREUSE": "NOREUSE"}
_ERRNOS = {2: "ENOENT", 5: "EIO", 9: "EBADF", 13: "EACCES", 17: "EEXIST", 20: "ENOTDIR", 21: "EISDIR", 22: "EINVAL", 28: "ENOSPC", 39: "ENOTEMPTY"}


def _errno_name(code: int) -> str:
    return _ERRNOS.get(code, f"E{code}")


class Exporter:
    """Collects the counted calls of a `Tracer` and writes the runner's trace file
    (`DESIGN_REVIEW.md` §3.58): a header line, then one op per line in the order the calls
    returned. A lane is a traced task; an op names its open by id (`fd`), through every
    `dup` and inheritance the `Tracer` resolved. Two passes: the second decides which
    sequential reads and writes are written positioned (those on an open that more than one
    lane used, whose position the lanes' interleaving decided) and which opens created
    their path."""

    def __init__(self, root: str):
        self.root = root.rstrip("/") + "/"
        self.events: list[dict] = []
        self.lanes: dict[int, int] = {}
        self.opens: list[dict] = []  # oid -> {"path", "lanes": set, "creat": bool}
        self.t0: float | None = None
        self.notes: Counter = Counter()

    def _lane(self, tid: int) -> int:
        lane = self.lanes.get(tid)
        if lane is None:
            lane = self.lanes[tid] = len(self.lanes)
        return lane

    def _event(self, when, op: str, **fields) -> dict:
        tid, issued, done = when
        if issued is None:
            self.notes["lines_without_timestamps"] += 1
            t = dur = 0
        else:
            if self.t0 is None:
                self.t0 = issued
            t = int(round((issued - self.t0) * 1e9))
            dur = int(round((done - issued) * 1e9)) if done is not None else 0
        e = {"lane": self._lane(tid), "t": t, "dur": max(dur, 0), "op": op}
        for k, v in fields.items():
            if v is None:
                continue
            if k in ("path", "to"):
                v = self.rel(v)
            e[k] = v
        self.events.append(e)
        if "fd" in fields and fields["fd"] is not None:
            self.opens[fields["fd"]]["lanes"].add(e["lane"])
        return e

    def rel(self, path: str) -> str:
        """The path below the root (the root itself is `.`)."""
        if path.startswith(self.root):
            return path[len(self.root) :] or "."
        return "." if path == self.root[:-1] else path

    @staticmethod
    def _ret(ret, err):
        if err is not None:
            return err
        return ret if ret not in (None, 0) else None

    def open(self, when, path: str, flags: str, mode: str | None, err: str | None) -> int:
        names, dropped = [], 0
        for f in flags.split("|"):
            if f in OPEN_FLAGS:
                names.append(OPEN_FLAGS[f])
            elif f and f != "O_LARGEFILE":
                dropped += 1
        if dropped:
            self.notes["open_flags_dropped"] += dropped
        if not any(n in ("RDONLY", "WRONLY", "RDWR") for n in names):
            names.insert(0, "RDONLY")  # O_RDONLY is 0 and strace prints it, but be sure
        oid = len(self.opens)
        self.opens.append({"path": path, "lanes": set(), "creat": "CREAT" in names and err is None})
        m = None
        if mode is not None and mode.isdigit():
            m = int(mode, 8)
        self._event(when, "open", fd=oid, path=path, flags=names, mode=m, ret=err)
        return oid

    def op(self, when, op: str, oid: int, err: str | None, **fields):
        ret = fields.pop("ret", None)
        self._event(when, op, fd=oid, ret=self._ret(ret, err), **fields)

    def data(self, when, kind: str, oid: int, offset: int, length: int, positioned: bool, ret: int, err: str | None):
        e = self._event(when, kind, fd=oid, len=length, ret=self._ret(max(ret, 0), err))
        e["_off"] = offset
        if positioned:
            e["offset"] = offset

    def submit(self, when, members: list) -> list[dict]:
        ops = []
        for kind, oid, off, nbytes in members:
            self.opens[oid]["lanes"].add(self._lane(when[0]))
            # `ret` is filled when the result is reaped; a result the trace never shows is
            # taken as complete, as the metrics take it
            ops.append({"op": kind, "fd": oid, "offset": off, "len": nbytes, "ret": nbytes})
        self._event(when, "submit", ops=ops)
        return ops

    def fd_op(self, when, kind: str, call: str, oid: int, a: list[str], ret: int, err: str | None):
        f = {}
        try:
            if kind == "ftruncate":
                f["len"] = int(a[1], 0)
            elif kind == "fallocate":
                f["offset"], f["len"] = int(a[2], 0), int(a[3], 0)
            elif kind == "fadvise":
                adv = ADVICE.get(a[3])
                if adv is None:
                    self.notes["calls_not_exported"] += 1
                    self.notes[f"not_exported:{call}:{a[3]}"] += 1
                    return
                f["offset"], f["len"], f["advice"] = int(a[1], 0), int(a[2], 0), adv
            elif kind == "ioctl":
                req = a[1].split(" ")[0] if len(a) > 1 else ""
                if req not in IOCTLS:
                    self.notes["calls_not_exported"] += 1
                    self.notes[f"not_exported:ioctl:{req or '?'}"] += 1
                    return
                f["request"] = req
        except (IndexError, ValueError):
            self.notes["calls_not_exported"] += 1
            self.notes[f"not_exported:{call}:unparsed"] += 1
            return
        if kind == "readdir":
            ret = 0  # the entries are not known from the trace: not checked
        self.op(when, kind, oid, err, ret=ret if kind != "readdir" else None, **f)

    def path_op(self, when, kind: str, call: str, path: str, a: list[str], pi: int, to: str | None, err: str | None):
        if call == "truncate":
            self.notes["calls_not_exported"] += 1
            self.notes["not_exported:truncate"] += 1
            return
        f = {}
        if kind == "rename":
            if to is None:
                self.notes["calls_not_exported"] += 1
                self.notes["not_exported:rename:destination outside the root"] += 1
                return
            f["to"] = to
        elif kind == "mkdir":
            m = a[pi + 1] if len(a) > pi + 1 else None
            if m is not None and m.isdigit():
                f["mode"] = int(m, 8)
        self._event(when, kind, path=path, ret=err, **f)

    def finish(self, root: str, source_notes: dict) -> tuple[dict, list[dict]]:
        """The header and the lines: positions of shared opens resolved, `creates` collected."""
        shared = 0
        for e in self.events:
            if e["op"] in KINDS and "_off" in e:
                off = e.pop("_off")
                if "offset" not in e and len(self.opens[e["fd"]]["lanes"]) > 1:
                    e["offset"] = off
                    shared += 1
        if shared:
            self.notes["shared_positions_resolved"] = shared
        seen, creates = set(), []
        for e in self.events:
            op = e["op"]
            if op == "open":
                if self.opens[e["fd"]]["creat"] and e["path"] not in seen and e["path"] not in creates and "ret" not in e:
                    creates.append(e["path"])
                seen.add(e["path"])
            elif op == "mkdir" and "ret" not in e and e["path"] not in creates:
                creates.append(e["path"])
            elif op == "rename" and "ret" not in e and e["to"] not in creates:
                creates.append(e["to"])
            elif op == "stat":
                if "ret" not in e:
                    seen.add(e["path"])
        notes = dict(sorted(self.notes.items()))
        for k, v in source_notes.items():
            notes.setdefault(k, v)
        header = {"aeiou_trace": TRACE_FORMAT, "source": "strace", "root": root, "lanes": len(self.lanes), "lines": len(self.events),
                  "opens": len(self.opens), "creates": creates, "notes": notes}
        return header, self.events


def export_of(lines, root: str, **kw) -> tuple[dict, list[dict]]:
    """Header and op lines of the runner's trace file for an strace given as lines."""
    m = Metrics()
    t = Tracer(m, [root], **kw)
    t.export = Exporter(t.roots[0])
    t.feed(lines)
    t.finish()
    return t.export.finish(t.roots[0], dict(sorted(t.notes.items())))


def write_export(path: str, header: dict, events: list[dict]) -> str:
    """Write the file; returns its sha256 (what the abstract's `trace` node names)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "wb") as out:
        for doc in (header, *events):
            line = (json.dumps(doc, separators=(",", ":")) + "\n").encode()
            h.update(line)
            out.write(line)
    return h.hexdigest()


class TraceReader:
    """The metrics of an exported trace file, with the same definitions as the strace's: a lane
    is a context, the whole file one instance, positions tracked per open id as the runner
    tracks them. What `aeiou dry-run --metrics` computes for a `trace` node must equal this."""

    def __init__(self, metrics: Metrics):
        self.m = metrics
        self.header: dict | None = None
        self.pos: dict[int, int] = {}
        self.paths: dict[int, str] = {}
        self.lanes: set[int] = set()
        self.notes: Counter = Counter()

    def feed(self, lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if self.header is None:
                if doc.get("aeiou_trace") != TRACE_FORMAT:
                    raise SystemExit(f"aeiou-trace: not an aeiou_trace: {TRACE_FORMAT} file")
                self.header = doc
                continue
            self.line(doc)

    def _data(self, lane: int, e: dict):
        kind = R if e["op"] == "read" else W
        oid = e["fd"]
        ret = e.get("ret", 0)
        n = ret if isinstance(ret, int) else 0
        if "offset" in e:
            off = e["offset"]
        else:
            off = self.pos.get(oid, 0)
            self.pos[oid] = off + n
        self.m.data(kind, 0, lane, self.paths[oid], off, e["len"], n)

    def line(self, e: dict):
        lane, op = e["lane"], e["op"]
        self.lanes.add(lane)
        if op == "open":
            self.paths[e["fd"]] = e["path"]
            self.pos[e["fd"]] = 0
            self.m.op("open")
        elif op in KINDS:
            self._data(lane, e)
        elif op == "submit":
            self.m.fan_out[len(e["ops"])] += 1
            for sub in e["ops"]:
                self._data(lane, sub)
        elif op == "close":
            self.m.op("close")
            self.m.close(lane, self.paths[e["fd"]])
        elif op == "lseek":
            self.m.op("lseek")
            ret = e.get("ret", 0)
            if isinstance(ret, int):
                self.pos[e["fd"]] = ret
        else:
            self.m.op(op)

    def finish(self):
        self.m.finish()


def metrics_of_export(lines, block=4096, sample=1) -> dict:
    m = Metrics(block, sample)
    r = TraceReader(m)
    r.feed(lines)
    r.finish()
    h = r.header or {}
    total = m.out()
    total["depth"] = {}
    return {
        "aeiou_metrics": FORMAT,
        "source": "trace",
        "roots": [h.get("root", "")],
        "block": m.block,
        "sample": m.sample,
        "order": "completion",
        "instances": 1,
        "contexts": len(r.lanes),
        "chain_gap_us": None,
        "notes": dict(h.get("notes", {})),
        "total": total,
    }


def metrics_of(lines, roots, block=4096, sample=1, **kw) -> dict:
    """The `aeiou_metrics: 1` document of a trace given as an iterable of lines."""
    m = Metrics(block, sample)
    t = Tracer(m, roots, **kw)
    t.feed(lines)
    t.finish()
    notes = dict(sorted(t.notes.items()))
    if t.chain_gap is None and m.fan_out:
        notes["depth_not_computed"] = "give --chain-gap-us: the think time that ends a chain of io_submit rounds"
    total = m.out()
    if t.chain_gap is None:
        total["depth"] = {}
    return {
        "aeiou_metrics": FORMAT,
        "source": "strace",
        "roots": t.roots,
        "block": m.block,
        "sample": m.sample,
        "order": "completion",
        "instances": max(len(m.instances), 1 if not t.instance_roots else 0),
        "contexts": len(t.tids),
        "chain_gap_us": None if t.chain_gap is None else t.chain_gap * 1e6,
        "notes": notes,
        "total": total,
    }


# ---------------------------------------------------------------- compare


def _cdf_distance(a: dict, b: dict) -> float | None:
    """The largest difference between two histograms' cumulative shares over the bucket
    bounds of both (the Kolmogorov-Smirnov statistic on the bucketed values)."""
    if not a["n"] or not b["n"]:
        return None if a["n"] == b["n"] else 1.0
    da, db = dict(map(tuple, a["buckets"])), dict(map(tuple, b["buckets"]))
    ca = cb = 0
    worst = 0.0
    for bound in sorted(set(da) | set(db)):
        ca += da.get(bound, 0)
        cb += db.get(bound, 0)
        worst = max(worst, abs(ca / a["n"] - cb / b["n"]))
    return worst


def _quantile(h: dict, p: float) -> int:
    want = max(-(-h["n"] * p // 1), 1)
    seen = 0
    for bound, w in h["buckets"]:
        seen += w
        if seen >= want:
            return bound
    return 0


def _tv_distance(a: dict, b: dict) -> float | None:
    """Total-variation distance between two exact distributions (fan-out, depth)."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None if na == nb else 1.0
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def _size(v: float) -> str:
    v = float(v)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:.0f} {unit}" if unit == "B" or v == int(v) else f"{v:.1f} {unit}"
        v /= 1024
    return str(v)


def select(doc: dict, template: str | None) -> dict:
    if template is None:
        return doc["total"]
    try:
        return doc["templates"][template]
    except KeyError:
        raise SystemExit(f"aeiou-trace: no template `{template}` in the document (it has: {', '.join(doc.get('templates', {})) or 'none'})")


def compare(a: dict, b: dict) -> list[tuple[str, str, str, float | None]]:
    """Rows `(metric, a, b, distance)` for two `total`s. A distance is in [0, 1]: a share
    difference, a CDF distance for a histogram, a total-variation distance for an exact
    distribution; `None` when both sides are empty."""
    rows = []

    def share(x, y):
        return x / y if y else 0.0

    def pct(name, va, vb):
        rows.append((name, f"{100 * va:.1f} %", f"{100 * vb:.1f} %", abs(va - vb)))

    def data(t):
        return t["counts"].get("read", 0) + t["counts"].get("write", 0)

    pct("mix: data ops / all ops", share(data(a), a["ops"]), share(data(b), b["ops"]))
    pct("mix: reads / data ops", share(a["counts"].get("read", 0), data(a)), share(b["counts"].get("read", 0), data(b)))
    pct("mix: read bytes / data bytes", share(a["bytes_read"], a["bytes_read"] + a["bytes_written"]), share(b["bytes_read"], b["bytes_read"] + b["bytes_written"]))
    for kind in sorted(set(a["counts"]) | set(b["counts"])):
        pct(f"ops: {kind} / all ops", share(a["counts"].get(kind, 0), a["ops"]), share(b["counts"].get(kind, 0), b["ops"]))

    def hist(name, ha, hb, fmt=_size):
        if not ha["n"] and not hb["n"]:
            return
        show = lambda h: "none" if not h["n"] else " / ".join(fmt(_quantile(h, p)) for p in (0.5, 0.9, 0.99))
        rows.append((f"{name} (p50 / p90 / p99)", show(ha), show(hb), _cdf_distance(ha, hb)))

    plain = lambda v: str(v)
    for k in KINDS:
        hist(f"request size, {k}", a["request_size"][k], b["request_size"][k])
    for k, total in (("read", "bytes_read"), ("write", "bytes_written")):
        hist(f"run length, {k}, bytes", a["run_length"][k]["bytes"], b["run_length"][k]["bytes"])
        hist(f"run length, {k}, ops", a["run_length"][k]["ops"], b["run_length"][k]["ops"], plain)
        if a[total] or b[total]:
            pct(f"run length, {k}: bytes in multi-op runs", share(a["run_length"][k]["multi_op_bytes"], a[total]), share(b["run_length"][k]["multi_op_bytes"], b[total]))
    ra, rb = a["reuse_distance_bytes"], b["reuse_distance_bytes"]
    for this in KINDS:
        def accesses(r):
            return r["first_touch"][this] + sum(r[f"{this}_after_{prev}"]["n"] for prev in KINDS)
        if accesses(ra) or accesses(rb):
            pct(f"reuse: first touches / block {this}s", share(ra["first_touch"][this], accesses(ra)), share(rb["first_touch"][this], accesses(rb)))
        for prev in KINDS:
            key = f"{this}_after_{prev}"
            if ra[key]["n"] or rb[key]["n"]:
                pct(f"reuse: {this} after {prev} / block {this}s", share(ra[key]["n"], accesses(ra)), share(rb[key]["n"], accesses(rb)))
            hist(f"reuse distance, {this} after {prev}", ra[key], rb[key])
    for unit in ("blocks", "objects"):
        pa, pb = a[f"popularity_{unit}"], b[f"popularity_{unit}"]
        for top, label in (("top_0_1_pct", "0.1 %"), ("top_1_pct", "1 %"), ("top_10_pct", "10 %")):
            pct(f"popularity, {unit}: accesses to the top {label}", pa[top], pb[top])
    for key in ("fan_out", "depth"):
        da = {int(k): v for k, v in a[key].items()}
        db = {int(k): v for k, v in b[key].items()}
        if da or db:
            show = lambda d: "none" if not d else " ".join(f"{k}:{100 * v / sum(d.values()):.0f}%" for k, v in sorted(d.items()))
            rows.append((key.replace("_", "-"), show(da), show(db), _tv_distance(da, db)))
    return rows


# ---------------------------------------------------------------- tolerances

TOLERANCES_FORMAT = 1
# The largest distance at which a row of `compare` still counts as a match, by the class
# of the row (the longest of these prefixes of its name). The repository's defaults; a
# tolerance file may replace any of them. `DESIGN_REVIEW.md` §3.55 has where they come from.
TOLERANCES = {
    "mix": 0.05,
    "ops": 0.05,
    "request size": 0.05,
    "run length": 0.10,
    "reuse": 0.10,
    "reuse distance": 0.10,
    "popularity": 0.05,
    "fan-out": 0.10,
    "depth": 0.10,
}
# A popularity row is the share of accesses that go to the top fraction of the units; it
# says something only when that fraction is at least this many units on both sides.
POPULARITY_MIN_UNITS = 10
_TOP = {"0.1 %": 0.001, "1 %": 0.01, "10 %": 0.1}


def tolerance_class(metric: str) -> str:
    return max((c for c in TOLERANCES if metric.startswith(c)), key=len)


def load_tolerances(path: str) -> dict:
    """A tolerance file: `{"aeiou_tolerances": 1, "tolerances": {class: x},
    "unseen": [{"metric": prefix, "reason": text}], "outside": [the same]}`. `unseen` names
    rows the trace cannot show (they are not judged); `outside` records rows known to be
    outside, with the reason (they are judged, and still outside)."""
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"aeiou-trace: {path}: {e}")
    if not isinstance(spec, dict) or spec.get("aeiou_tolerances") != TOLERANCES_FORMAT:
        raise SystemExit(f"aeiou-trace: {path}: not an aeiou_tolerances: {TOLERANCES_FORMAT} document")
    extra = set(spec) - {"aeiou_tolerances", "tolerances", "unseen", "outside", "comment"}
    if extra:
        raise SystemExit(f"aeiou-trace: {path}: unknown key(s) {', '.join(sorted(extra))}")
    for c, x in spec.get("tolerances", {}).items():
        if c not in TOLERANCES or not isinstance(x, (int, float)) or isinstance(x, bool) or not 0 <= x <= 1:
            raise SystemExit(f"aeiou-trace: {path}: tolerances: `{c}` must be one of {', '.join(TOLERANCES)} with a value in [0, 1]")
    for key in ("unseen", "outside"):
        for e in spec.get(key, []):
            if not isinstance(e, dict) or set(e) != {"metric", "reason"} or not all(isinstance(v, str) and v for v in e.values()):
                raise SystemExit(f"aeiou-trace: {path}: {key}: each entry is {{\"metric\": prefix, \"reason\": text}}")
    return spec


def judge(rows, a: dict, b: dict, docs=(), selfs=(), spec: dict | None = None):
    """Rows `(metric, a, b, distance, allowed, verdict, note)` for the rows of `compare(a, b)`.
    `allowed` is the class tolerance plus the largest distance of that row between `b` and
    each of `selfs` (the same abstract at other seeds: what the abstract differs from itself
    by). Verdicts: `ok`, `outside`, `unseen` (the tolerance file says the trace cannot show
    it), `not judged` (nothing on either side, too few units for a popularity share, or a
    depth without `--chain-gap-us`). Also returns the entries of the file that matched no
    row, or recorded as outside a row that is not."""
    spec = spec or {}
    tol = {**TOLERANCES, **spec.get("tolerances", {})}
    spread: dict[str, float] = {}
    for s in selfs:
        for name, _, _, dist in compare(b, s):
            spread[name] = max(spread.get(name, 0.0), dist or 0.0)
    no_depth = any(d.get("source") == "strace" and d.get("chain_gap_us") is None for d in docs)
    used = set()

    def listed(key, name):
        for i, e in enumerate(spec.get(key, [])):
            if name.startswith(e["metric"]):
                used.add((key, i))
                return e["reason"]
        return None

    out = []
    for name, va, vb, dist in rows:
        allowed = tol[tolerance_class(name)] + spread.get(name, 0.0)
        why = listed("unseen", name)
        if why is not None:
            out.append((name, va, vb, dist, None, "unseen", why))
            continue
        if dist is None:
            out.append((name, va, vb, dist, None, "not judged", "nothing on either side"))
            continue
        if name.startswith("popularity"):
            unit, top = name.split(":")[0].split(", ")[1], name.rsplit("top ", 1)[1]
            fewest = min(a[f"popularity_{unit}"]["distinct"], b[f"popularity_{unit}"]["distinct"])
            if fewest * _TOP[top] < POPULARITY_MIN_UNITS:
                out.append((name, va, vb, dist, None, "not judged", f"the top {top} of {fewest} {unit} is fewer than {POPULARITY_MIN_UNITS}"))
                continue
        if name == "depth" and no_depth:
            out.append((name, va, vb, dist, None, "not judged", "the trace was read without --chain-gap-us"))
            continue
        why = listed("outside", name)
        if dist <= allowed:
            out.append((name, va, vb, dist, allowed, "ok", "recorded as outside, now within" if why is not None else ""))
        else:
            out.append((name, va, vb, dist, allowed, "outside", why or ""))
    stale = [f"{key}: `{e['metric']}` matches no row" for key in ("unseen", "outside") for i, e in enumerate(spec.get(key, [])) if (key, i) not in used]
    stale += [f"outside: `{r[0]}` is recorded as outside and is within" for r in out if r[5] == "ok" and r[6]]
    return out, stale


# ---------------------------------------------------------------- CLI


def _describe(doc: dict) -> str:
    src = doc.get("source", "dry-run")
    if src == "strace":
        return f"strace of {', '.join(doc.get('roots', []))} (order: completion; {doc.get('instances')} instance(s), {doc.get('contexts')} thread(s))"
    return f"{src} of {doc.get('abstract')} seed {doc.get('seed')} gpus {doc.get('gpus')} (order: {doc.get('order')})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aeiou-trace", description="Locality metrics of a real application from an strace of it, and their comparison with an abstract's (aeiou dry-run --metrics-json).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("metrics", help="compute the metrics of a trace (strace -f -ttt -T -yy -e trace=%%file,%%desc,%%process -o FILE)")
    mp.add_argument("trace", help="the strace output file, or - for stdin")
    mp.add_argument("--root", action="append", default=[], metavar="DIR", help="count only calls on paths under DIR (repeatable; not needed for an exported trace file)")
    mp.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    mp.add_argument("--cwd", metavar="DIR", help="resolve relative paths against DIR when the trace does not say (no -y)")
    mp.add_argument("--block", type=int, default=4096, metavar="BYTES", help="the block size of the reuse-distance and popularity units (default 4096)")
    mp.add_argument("--sample", type=int, default=1, metavar="N", help="keep one block and one object in N, chosen by hash, and scale (default 1: exact)")
    mp.add_argument("--instance-root", action="append", type=int, default=[], metavar="PID", help="the process tree under PID is one instance (repeatable; needs clone in the trace); default: the whole trace is one instance")
    mp.add_argument("--chain-gap-us", type=float, metavar="US", help="report depth: consecutive io_submit rounds of a thread form a chain until more than US microseconds pass between a round's end and the next submit")
    mp.add_argument("-o", "--output", metavar="FILE", help="write the JSON document (aeiou_metrics: 1) to FILE instead of stdout")
    ep = sub.add_parser("export", help="write the runner's trace file for a `trace` node from an strace (DESIGN_REVIEW.md §3.58): JSON Lines, one lane per traced task, opens by id, paths relative to --root")
    ep.add_argument("trace", help="the strace output file, or - for stdin")
    ep.add_argument("--root", required=True, metavar="DIR", help="export only calls on paths under DIR, written relative to it")
    ep.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    ep.add_argument("--cwd", metavar="DIR", help="resolve relative paths against DIR when the trace does not say (no -y)")
    ep.add_argument("-o", "--output", required=True, metavar="FILE", help="the trace file to write; its sha256 is printed, for the abstract's `trace` node")
    cp = sub.add_parser("compare", help="compare two metrics documents, from a trace or from aeiou dry-run --metrics-json")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--template-a", metavar="NAME", help="use this actor template's metrics of A instead of its total")
    cp.add_argument("--template-b", metavar="NAME", help="use this actor template's metrics of B instead of its total")
    cp.add_argument("--max-distance", type=float, metavar="X", help="exit 1 if any distance exceeds X (one number for every row; --judge is the rule by class)")
    cp.add_argument("--judge", action="store_true", help="judge each row against the tolerance of its class (A the trace, B the abstract) and exit 1 if any is outside")
    cp.add_argument("--tolerances", metavar="FILE", help="a tolerance file (aeiou_tolerances: 1): class tolerances that replace the defaults, the rows the trace cannot show, the rows recorded as outside; implies --judge")
    cp.add_argument("--self", dest="selfs", action="append", default=[], metavar="FILE", help="a metrics document of B's abstract at the same parameters and another seed (repeatable): a row's tolerance is raised by its largest distance between B and these; implies --judge")
    cp.add_argument("--only", action="append", default=[], metavar="PREFIX", help="only the rows whose metric starts with PREFIX (repeatable)")
    args = ap.parse_args(argv)

    if args.cmd == "metrics":
        if args.block < 1 or args.sample < 1:
            ap.error("--block and --sample must be at least 1")
        f = sys.stdin if args.trace == "-" else open(args.trace, encoding="utf-8", errors="replace")
        with f:
            first = f.readline()
            lines = itertools.chain([first], f)
            if first.startswith('{"aeiou_trace"'):
                doc = metrics_of_export(lines, args.block, args.sample)
            elif not args.root:
                ap.error("--root DIR is required for an strace")
            else:
                doc = metrics_of(lines, [os.path.abspath(r) for r in args.root], args.block, args.sample, exclude=args.exclude, cwd=args.cwd, instance_roots=args.instance_root, chain_gap_us=args.chain_gap_us)
        doc["trace"] = args.trace
        text = json.dumps(doc, indent=2) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as out:
                out.write(text)
            t = doc["total"]
            print(f"{t['ops']} ops under {', '.join(doc['roots'])}: read {t['bytes_read']} bytes, wrote {t['bytes_written']} bytes; {doc['contexts']} thread(s); {args.output}")
            for k, v in doc["notes"].items():
                print(f"note: {k}: {v}")
        else:
            sys.stdout.write(text)
        return 0

    if args.cmd == "export":
        f = sys.stdin if args.trace == "-" else open(args.trace, encoding="utf-8", errors="replace")
        with f:
            header, events = export_of(f, os.path.abspath(args.root), exclude=args.exclude, cwd=args.cwd)
        sha = write_export(args.output, header, events)
        print(f"{header['lines']} line(s) on {header['lanes']} lane(s), {header['opens']} open(s), {len(header['creates'])} path(s) created under {header['root']}; {args.output}")
        print(f"sha256 {sha}")
        for k, v in header["notes"].items():
            print(f"note: {k}: {v}")
        return 0

    docs = []
    for path in (args.a, args.b):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(f"aeiou-trace: {path}: {e}")
        if not isinstance(d, dict) or d.get("aeiou_metrics") != FORMAT:
            raise SystemExit(f"aeiou-trace: {path}: not an aeiou_metrics: {FORMAT} document")
        docs.append(d)
    if docs[0]["block"] != docs[1]["block"]:
        raise SystemExit(f"aeiou-trace: block sizes differ ({docs[0]['block']} and {docs[1]['block']}): reuse distance and block popularity are not comparable")
    a, b = select(docs[0], args.template_a), select(docs[1], args.template_b)
    rows = compare(a, b)
    if args.only:
        rows = [r for r in rows if any(r[0].startswith(p) for p in args.only)]
    print(f"A: {args.a}: {_describe(docs[0])}")
    print(f"B: {args.b}: {_describe(docs[1])}")
    if args.judge or args.tolerances or args.selfs:
        return _judged(args, rows, a, b, docs)
    w = [max(len(r[i]) for r in rows + [("metric", "A", "B", 0)]) for i in range(3)]
    print(f"{'metric':<{w[0]}}  {'A':>{w[1]}}  {'B':>{w[2]}}  distance")
    worst = 0.0
    for name, va, vb, dist in rows:
        worst = max(worst, dist or 0.0)
        flag = "  <" if args.max_distance is not None and dist is not None and dist > args.max_distance else ""
        print(f"{name:<{w[0]}}  {va:>{w[1]}}  {vb:>{w[2]}}  {'-' if dist is None else f'{dist:.3f}'}{flag}")
    print(f"largest distance {worst:.3f}")
    if args.max_distance is not None and worst > args.max_distance:
        print(f"aeiou-trace: exceeds --max-distance {args.max_distance}", file=sys.stderr)
        return 1
    return 0


def _judged(args, rows, a, b, docs) -> int:
    if args.max_distance is not None:
        raise SystemExit("aeiou-trace: --max-distance and --judge are two rules; give one")
    selfs = []
    for path in args.selfs:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(f"aeiou-trace: {path}: {e}")
        if not isinstance(d, dict) or d.get("aeiou_metrics") != FORMAT:
            raise SystemExit(f"aeiou-trace: {path}: not an aeiou_metrics: {FORMAT} document")
        for key in ("source", "sha256", "gpus", "block", "sample"):
            if d.get(key) != docs[1].get(key):
                raise SystemExit(f"aeiou-trace: {path}: --self is B's abstract at another seed, and its `{key}` is not B's ({d.get(key)} and {docs[1].get(key)})")
        if d.get("seed") == docs[1].get("seed"):
            raise SystemExit(f"aeiou-trace: {path}: --self has B's seed ({d.get('seed')}); another seed is the point")
        print(f"self: {path}: seed {d.get('seed')}")
        selfs.append(select(d, args.template_b))
    spec = load_tolerances(args.tolerances) if args.tolerances else None
    judged, stale = judge(rows, a, b, docs, selfs, spec)
    w = [max(len(r[i]) for r in rows + [("metric", "A", "B", 0)]) for i in range(3)]
    print(f"{'metric':<{w[0]}}  {'A':>{w[1]}}  {'B':>{w[2]}}  distance  allowed  verdict")
    for name, va, vb, dist, allowed, verdict, note in judged:
        print(f"{name:<{w[0]}}  {va:>{w[1]}}  {vb:>{w[2]}}  {'-' if dist is None else f'{dist:.3f}':>8}  {'-' if allowed is None else f'{allowed:.3f}':>7}  {verdict.upper() if verdict == 'outside' else verdict}{f' ({note})' if note else ''}")
    n = Counter(r[5] for r in judged)
    for line in stale:
        print(f"note: {line}")
    counts = f"{n['ok']} row(s) within tolerance, {n['outside']} outside, {n['unseen']} unseen, {n['not judged']} not judged"
    if n["outside"]:
        print(f"not accepted: {counts}")
        return 1
    if not n["ok"]:
        print(f"nothing judged: {counts}")
        return 1
    print(f"accepted: {counts}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
