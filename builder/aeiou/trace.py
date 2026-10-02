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
"""

from __future__ import annotations

import argparse
import fnmatch
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

    __slots__ = ("path", "pos", "append", "listed")

    def __init__(self, path, pos=0, append=False):
        self.path = path
        self.pos = pos
        self.append = append
        # a directory stream being read: its `getdents` calls are one `readdir` op together
        self.listed = False


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
HANDLED = OPENS | READS | WRITES | CLONES | DUPS | set(FD_OPS) | set(PATH_OPS) | {"close", "lseek", "io_submit", "io_getevents", "io_pgetevents", "chdir", "fchdir"}


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
        if rest.startswith("<"):
            _, ret_path = fd_arg(rm.group(1) + rest.split(" <")[0] if " <" in rest else rm.group(1) + rest)
        d = _DUR.search(ret_s)
        done = ts if r else (issued + float(d.group(1)) if issued is not None and d else issued)
        self.tids.add(tid)
        self.call(tid, name, split_args(body[i + 1 : k]), ret, ret_path, issued, done)

    def _count(self, tid: int, kind: str):
        self.m.op(kind)
        self._end_chain(tid)

    def _end_chain(self, tid: int):
        c = self.chain.pop(tid, None)
        if c and self.chain_gap is not None:
            self.m.depth[c[0]] += 1

    def call(self, tid: int, name: str, a: list[str], ret: int, ret_path: str | None, issued: float | None, done: float | None):
        m = self.m
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
            if ret >= 0:
                known_empty = "O_TRUNC" in flags or "O_EXCL" in flags or name == "creat"
                if "O_APPEND" in flags and not known_empty:
                    self.notes["append_to_unknown_size"] += 1
                self.table(tid)[ret] = File(path, 0, "O_APPEND" in flags)
            if inst is not None and self.under(path):
                self._count(tid, "open")
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
            for data, kind, p, off, nbytes in got:
                # counted when its result is reaped (`io_getevents`), with the bytes it moved
                self.aio.setdefault((ctx, data), []).append((kind, inst, tid, p, off, nbytes))
            return
        if name in ("io_getevents", "io_pgetevents"):
            c = self.chain.get(tid)
            if c:
                c[1] = done
            ctx = a[0] if a else None
            for data, res in _EVENT.findall(a[3] if len(a) > 3 else ""):
                waiting = self.aio.get((ctx, data))
                if waiting:
                    kind, i, t, p, off, nbytes = waiting.pop(0)
                    if not waiting:
                        del self.aio[(ctx, data)]
                    m.data(kind, i, t, p, off, nbytes, max(int(res), 0))
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
            else:
                path = self.resolve(tid, a[di] if di is not None else None, arg)
                if name == "unlinkat" and "AT_REMOVEDIR" in a[-1]:
                    kind = "rmdir"
            if inst is not None and self.under(path):
                self._count(tid, kind)
            return

    def finish(self):
        for tid in list(self.chain):
            self._end_chain(tid)
        # requests whose results the trace does not show (reaped from the ring in user
        # space, or `io_getevents` not traced): taken as complete
        for waiting in self.aio.values():
            for kind, i, t, p, off, nbytes in waiting:
                self.notes["aio_results_assumed_complete"] += 1
                self.m.data(kind, i, t, p, off, nbytes, nbytes)
        self.aio = {}
        self.m.finish()


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
    mp.add_argument("--root", action="append", required=True, metavar="DIR", help="count only calls on paths under DIR (repeatable)")
    mp.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    mp.add_argument("--cwd", metavar="DIR", help="resolve relative paths against DIR when the trace does not say (no -y)")
    mp.add_argument("--block", type=int, default=4096, metavar="BYTES", help="the block size of the reuse-distance and popularity units (default 4096)")
    mp.add_argument("--sample", type=int, default=1, metavar="N", help="keep one block and one object in N, chosen by hash, and scale (default 1: exact)")
    mp.add_argument("--instance-root", action="append", type=int, default=[], metavar="PID", help="the process tree under PID is one instance (repeatable; needs clone in the trace); default: the whole trace is one instance")
    mp.add_argument("--chain-gap-us", type=float, metavar="US", help="report depth: consecutive io_submit rounds of a thread form a chain until more than US microseconds pass between a round's end and the next submit")
    mp.add_argument("-o", "--output", metavar="FILE", help="write the JSON document (aeiou_metrics: 1) to FILE instead of stdout")
    cp = sub.add_parser("compare", help="compare two metrics documents, from a trace or from aeiou dry-run --metrics-json")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--template-a", metavar="NAME", help="use this actor template's metrics of A instead of its total")
    cp.add_argument("--template-b", metavar="NAME", help="use this actor template's metrics of B instead of its total")
    cp.add_argument("--max-distance", type=float, metavar="X", help="exit 1 if any distance exceeds X (no default: tolerances are not part of this tool)")
    cp.add_argument("--only", action="append", default=[], metavar="PREFIX", help="only the rows whose metric starts with PREFIX (repeatable)")
    args = ap.parse_args(argv)

    if args.cmd == "metrics":
        if args.block < 1 or args.sample < 1:
            ap.error("--block and --sample must be at least 1")
        f = sys.stdin if args.trace == "-" else open(args.trace, encoding="utf-8", errors="replace")
        with f:
            doc = metrics_of(f, [os.path.abspath(r) for r in args.root], args.block, args.sample, exclude=args.exclude, cwd=args.cwd, instance_roots=args.instance_root, chain_gap_us=args.chain_gap_us)
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
    rows = compare(select(docs[0], args.template_a), select(docs[1], args.template_b))
    if args.only:
        rows = [r for r in rows if any(r[0].startswith(p) for p in args.only)]
    print(f"A: {args.a}: {_describe(docs[0])}")
    print(f"B: {args.b}: {_describe(docs[1])}")
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


if __name__ == "__main__":
    sys.exit(main())
