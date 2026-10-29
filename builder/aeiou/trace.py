"""`aeiou-trace`: the locality metrics of a real application, from an `strace` of it.

`aeiou dry-run --metrics` computes reuse distance, sequential runs, popularity, request
sizes, fan-out, depth, and the read/write mix on an abstract's op stream
(`runner/REFERENCE.md` §10). This computes the same numbers, with the same definitions and in
the same JSON form (`aeiou_metrics: 2`; reuse and popularity over extents, byte ranges, and
over blocks too under `--metrics-block`, `DESIGN_REVIEW.md` §3.79), from a trace of the
application the abstract models, so the two can be compared (`GRAMMAR_OPTIONS.md` §5.4):

    strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy \
        -o trace.txt <command>
    aeiou-trace metrics trace.txt --root /mnt/data -o trace.metrics.json
    aeiou dry-run x.ast.json --gpus 1 --metrics-json abstract.metrics.json
    aeiou-trace compare trace.metrics.json abstract.metrics.json
    aeiou-trace compare trace.metrics.json abstract.metrics.json --judge --self other-seed.metrics.json

`--judge` holds each row to the tolerance of its class (`TOLERANCES` below, `builder/REFERENCE.md`
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

The AIO calls are named: `strace` classes none of them in `%desc`, and a trace without
them loses every libaio read and write. A context the trace shows `io_setup` make and no
`io_submit` use is noted (`aio_contexts_without_io_submit`): either the application never
used it or the trace left `io_submit` out. `-v` prints whole arrays: without it `strace`
cuts one at 32 elements, and the requests of a larger `io_submit`, the results of an
`io_getevents`, and the iovecs of a `readv` or `writev` past the 32nd are lost; a cut
array is noted (`aio_requests_truncated`, `aio_events_truncated`, `iov_arrays_truncated`).

Not visible to `strace`, and so not counted: `io_uring` submissions and page faults on a
mapping. Standard library only; the trace is read once, in a stream.

`aeiou-trace export` writes the same calls as a trace file for the runner's `trace` node
(`DESIGN_REVIEW.md` §3.58): JSON Lines, a header and one op per line in the order the calls
returned, each on a lane (a traced task) and naming its open by id, so that
`aeiou dry-run --metrics` of an abstract that is one `trace` node computes these same
numbers. `aeiou-trace metrics` reads an exported file too (`TraceReader`).

`aeiou-trace digest` (`digest.py`) reduces the same calls to the structural summary a fit is
made from: path templates, lanes and phases, a verdict per site on what the trace decides
(a permutation pass or a draw, order, size), and the open questions ranked by divergence.
"""

from __future__ import annotations

import fnmatch
import io
import bisect
import itertools
import json
import os
import re
import sys
from collections import Counter

from . import __version__, checkout, usage
from .nodes import BuildError

FORMAT = 2
FORMATS_READ = (1, 2)    # version 1: a capture taken before 2026-10-22, its locality in blocks only (DESIGN_REVIEW.md §3.79)
MAX_ENTRIES = 1 << 26
KINDS = ("read", "write")
R, W = 0, 1

# ---------------------------------------------------------------- histograms


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


class Extents:
    """One instance's history by byte range, as `metrics::Extents` (`DESIGN_REVIEW.md` §3.79):
    pieces `[start, end)` of a path, each last touched by one op (`[end, time, kind, count,
    run]`), and a Fenwick tree over op times holding the bytes last touched at each."""

    def __init__(self):
        self.starts: dict[int, list[int]] = {}         # path -> sorted piece starts
        self.pieces: dict[tuple[int, int], list] = {}   # (path, start) -> [end, time, kind, count, run]
        self.tree: list[int] = []
        self.live = 0
        self.now = 0
        self.runs: dict[int, int] = {}                  # run -> live bytes
        self.compactions = 0

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
        times = sorted({p[1] for p in self.pieces.values()})
        rank = {t: i for i, t in enumerate(times)}
        cap = 1024
        while cap < 2 * len(times):
            cap *= 2
        self.tree = [0] * cap
        for (_, start), p in self.pieces.items():
            p[1] = rank[p[1]]
            self._add(p[1], p[0] - start)
        self.now = len(times)
        self.compactions += 1

    def _put(self, path: int, start: int, piece: list):
        n = piece[0] - start
        self._add(piece[1], n)
        self.runs[piece[4]] = self.runs.get(piece[4], 0) + n
        self.pieces[(path, start)] = piece
        bisect.insort(self.starts.setdefault(path, []), start)

    def _take(self, path: int, start: int) -> list:
        piece = self.pieces.pop((path, start))
        n = piece[0] - start
        self._add(piece[1], -n)
        left = self.runs[piece[4]] - n
        if left:
            self.runs[piece[4]] = left
        else:
            del self.runs[piece[4]]
        starts = self.starts[path]
        del starts[bisect.bisect_left(starts, start)]
        return piece

    def live_of(self, run: int) -> int:
        return self.runs.get(run, 0)

    def touch(self, path: int, start: int, end: int, kind: int, run: int, hits: list) -> int:
        """One op touching `[start, end)`: the groups `(distance, bytes, previous kind, run)` of
        the bytes it touches again appended to `hits`, ascending; returns its first touches."""
        if self.now == len(self.tree):
            self._compact()
        now = self.now
        starts = self.starts.get(path, [])
        i = bisect.bisect_left(starts, start)
        over = starts[i - 1:i] if i and self.pieces[(path, starts[i - 1])][0] > start else []
        over += starts[i:bisect.bisect_left(starts, end)]
        at, first, group, group_end = start, 0, None, start
        for s0 in over:
            old = self.pieces[(path, s0)]
            lo, hi = max(s0, start), min(old[0], end)
            if lo > at:
                # bytes no op of the instance touched before
                if group is not None:
                    hits.append((group[4] + group[3], group[3], group[1], group[2]))
                    group = None
                self._put(path, at, [lo, now, kind, 1, run])
                self.live += lo - at
                first += lo - at
            if not (group is not None and group[0] == old[1] and group_end == lo):
                if group is not None:
                    hits.append((group[4] + group[3], group[3], group[1], group[2]))
                # [old time, kind, run, bytes, bytes touched after the old op, this op's so far
                # included (the old piece is still at its time)]
                group = [old[1], old[2], old[4], 0, self.live - self._prefix(old[1])]
            group[3] += hi - lo
            group_end = hi
            self._take(path, s0)
            if s0 < lo:
                self._put(path, s0, [lo, *old[1:]])
            if old[0] > hi:
                self._put(path, hi, old[:])
            self._put(path, lo, [hi, now, kind, old[3] + 1, run])
            at = hi
        if group is not None:
            hits.append((group[4] + group[3], group[3], group[1], group[2]))
        if end > at:
            self._put(path, at, [end, now, kind, 1, run])
            self.live += end - at
            first += end - at
        self.now += 1
        return first


class Bytes:
    """Accesses per byte, as `metrics::Bytes`: pieces `(path, start) -> [end, accesses]`."""

    def __init__(self):
        self.starts: dict[int, list[int]] = {}
        self.pieces: dict[tuple[int, int], list] = {}

    def _put(self, path, start, end, count):
        self.pieces[(path, start)] = [end, count]
        bisect.insort(self.starts.setdefault(path, []), start)

    def add(self, path: int, start: int, end: int, count: int):
        """`count` more accesses of `[start, end)` of `path`."""
        starts = self.starts.setdefault(path, [])
        i = bisect.bisect_left(starts, start)
        over = starts[i - 1:i] if i and self.pieces[(path, starts[i - 1])][0] > start else []
        over += starts[i:bisect.bisect_left(starts, end)]
        at = start
        for s0 in over:
            e, c = self.pieces.pop((path, s0))
            del starts[bisect.bisect_left(starts, s0)]
            lo, hi = max(s0, start), min(e, end)
            if lo > at:
                self._put(path, at, lo, count)
            if s0 < lo:
                self._put(path, s0, lo, c)
            if e > hi:
                self._put(path, hi, e, c)
            self._put(path, lo, hi, c + count)
            at = hi
        if end > at:
            self._put(path, at, end, count)

    def merge(self, other: "Bytes"):
        for (path, start), (end, c) in other.pieces.items():
            self.add(path, start, end, c)

    def popularity(self) -> dict:
        return popularity((c, e - s) for (_, s), (e, c) in self.pieces.items())


def popularity(items) -> dict:
    """The rank-frequency curve as counts of counts, as `metrics::Popularity`: from
    `(accesses, items)` pairs, or from a dict of accesses per item."""
    if isinstance(items, dict):
        items = ((c, 1) for c in items.values())
    cc: Counter = Counter()
    for c, n in items:
        cc[c] += n
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
        "distinct": items,
        "accesses": total,
        "max": rows[0][0] if rows else 0,
        "top_0_1_pct": share(0.001),
        "top_1_pct": share(0.01),
        "top_10_pct": share(0.1),
        "counts": [[c, n] for c, n in rows],
    }


class Metrics:
    """The accumulated metrics of a trace; the definitions are `runner/REFERENCE.md` §10's."""

    def __init__(self, block: int | None = None):
        self.block = None if block is None else max(block, 1)
        self.counts: Counter = Counter()
        self.bytes = [0, 0]
        self.request = [LogHist(), LogHist()]
        self.run_bytes = [LogHist(), LogHist()]
        self.run_ops = [LogHist(), LogHist()]
        self.multi_op_bytes = [0, 0]
        self.ext_reuse = [[LogHist(), LogHist()], [LogHist(), LogHist()]]  # extents, in bytes: [previous][this]
        self.ext_first = [0, 0]
        self.whole = self.partial = 0
        self.extents: dict[int, Extents] = {}
        self.instance_ids: set = set()
        self.byte_counts = Bytes()
        self.next_run = 0
        self.reuse = [[LogHist(), LogHist()], [LogHist(), LogHist()]]  # blocks, under --metrics-block
        self.first = [0, 0]
        self.fan_out: Counter = Counter()
        self.depth: Counter = Counter()
        self.blocks: dict[int, int] = {}
        self.objects: dict[int, int] = {}
        self.instances: dict[int, Instance] = {}
        self.path_ids: dict[str, int] = {}
        # open runs: context -> {(path id, kind): [end, bytes, ops, id, {extent: bytes read again}, instance]}
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
        self.objects[pid] = self.objects.get(pid, 0) + 1
        self.bytes[kind] += max(nbytes, 0)
        if nbytes <= 0 or offset < 0:
            return
        # a run ends before the op that does not continue it
        runs = self.runs.setdefault(context, {})
        run = runs.get((pid, kind))
        if run is not None and run[0] != offset:
            self._run_done(kind, run)
            run = None
        if run is None:
            self.next_run += 1
            run = runs[(pid, kind)] = [offset, 0, 0, self.next_run, {}, instance]
        ext = self.extents.get(instance)
        if ext is None:
            ext = self.extents[instance] = Extents()
            self.instance_ids.add(instance)
        hits = []
        self.ext_first[kind] += ext.touch(pid, offset, offset + nbytes, kind, run[3], hits)
        again = run[4]
        for dist, n, prev, x in hits:
            self.ext_reuse[prev][kind].add(dist, n)
            if kind == R:
                again[x] = again.get(x, 0) + n
        if len(ext.pieces) > MAX_ENTRIES:
            raise BuildError(f"more than {MAX_ENTRIES} pieces in one instance (byte ranges touched, as fragmented as the accesses)")
        b = self.block
        if b is not None:
            inst = self.instances.get(instance)
            if inst is None:
                inst = self.instances[instance] = Instance()
            for blk in range(offset // b, (offset + nbytes - 1) // b + 1):
                hit = inst.touch((pid << 44) | blk, kind)
                if hit is None:
                    self.first[kind] += 1
                else:
                    self.reuse[hit[1]][kind].add(hit[0] * b)
            if len(inst.last) > MAX_ENTRIES:
                raise BuildError(f"more than {MAX_ENTRIES} distinct blocks in one instance; give a larger --metrics-block")
        run[0] = offset + nbytes
        run[1] += nbytes
        run[2] += 1

    def _run_done(self, kind: int, run: list):
        self.run_bytes[kind].add(run[1])
        self.run_ops[kind].add(run[2])
        if run[2] >= 2:
            self.multi_op_bytes[kind] += run[1]
        ext = self.extents[run[5]]
        for x, n in run[4].items():
            if ext.live_of(x) == 0:
                self.whole += n
            else:
                self.partial += n

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
        for ext in self.extents.values():
            for (pid, start), piece in ext.pieces.items():
                self.byte_counts.add(pid, start, piece[0], piece[3])
        self.extents = {}

    def out(self) -> dict:
        """The `total` of an `aeiou_metrics: 2` document (`metrics::MetricsOut`)."""
        runs = lambda k: {"bytes": self.run_bytes[k].out(), "ops": self.run_ops[k].out(), "multi_op_bytes": self.multi_op_bytes[k]}
        reuse = lambda first, h: {
            "first_touch": {"read": first[R], "write": first[W]},
            "read_after_read": h[R][R].out(),
            "read_after_write": h[W][R].out(),
            "write_after_read": h[R][W].out(),
            "write_after_write": h[W][W].out(),
        }
        out = {
            "ops": sum(self.counts.values()),
            "counts": dict(sorted(self.counts.items())),
            "bytes_read": self.bytes[R],
            "bytes_written": self.bytes[W],
            "request_size": {"read": self.request[R].out(), "write": self.request[W].out()},
            "run_length": {"read": runs(R), "write": runs(W)},
            "reuse_extents": {**reuse(self.ext_first, self.ext_reuse), "whole": self.whole, "partial": self.partial},
            "popularity_bytes": self.byte_counts.popularity(),
        }
        if self.block is not None:
            out["reuse_distance_bytes"] = reuse(self.first, self.reuse)
            out["popularity_blocks"] = popularity(self.blocks)
        out["popularity_objects"] = popularity(self.objects)
        out["fan_out"] = {str(k): v for k, v in sorted(self.fan_out.items())}
        out["depth"] = {str(k): v for k, v in sorted(self.depth.items())}
        return out


# ---------------------------------------------------------------- strace lines

_PREFIX = re.compile(r"^(?:\[pid\s+(\d+)\]\s+|(\d+)\s+)?(?:(\d+\.\d+|\d\d:\d\d:\d\d(?:\.\d+)?)\s+)?(.*)$", re.S)
_RESUMED = re.compile(r"^<\.\.\. (\w+) resumed>\s*(.*)$", re.S)
_RET = re.compile(r"^(-?\d+|0x[0-9a-f]+|\?)")
_DUR = re.compile(r"<(\d+\.\d+)>$")
_ERRNO = re.compile(r"^\s*(E[A-Z0-9]+)")
# CLONE_FILES in the flags of a `clone` (its `flags=` argument, the second: `child_stack=` is
# first) or a `clone3` (the `flags` field of its struct, the first argument)
_CLONE_FILES = re.compile(r"\bflags=[^,}]*\bCLONE_FILES\b")
_FD = re.compile(r"^(-?\d+|AT_FDCWD)(?:<(.*)>)?$", re.S)
_IOV_LEN = re.compile(r"iov_len=(\d+)")
_IOCB = re.compile(r"aio_data=(\w+), aio_lio_opcode=IOCB_CMD_(\w+).*?aio_fildes=(\d+)(?:<(.*?)>)?, .*?aio_nbytes=(\d+), aio_offset=(-?\d+)")
_EVENT = re.compile(r"\{data=(\w+), obj=\w+, res=(-?\d+)")
UNFINISHED = "<unfinished ...>"


def truncated(arg: str) -> bool:
    """Whether `strace` cut the array `arg` short (`[..., ...]`): it prints only the first
    32 elements of an array unless `-v` is given."""
    return arg.rstrip().endswith("...]")


def shares_files(name: str, args: str) -> bool:
    """Whether a `clone`/`clone3` (whose argument text is `args`) shares the descriptor table
    with its child; `fork` and `vfork` copy it."""
    return name in ("clone", "clone3") and _CLONE_FILES.search(args) is not None


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
    return fd, fd_path(m.group(2))


def fd_path(raw: str | None):
    """The path of a descriptor's `<...>` annotation, unescaped as a quoted string is: strace
    prints it as one without the quotes, with `<` and `>` as octal (`\\74`, `\\76`) so that no
    raw angle bracket is ever part of the path, and a `<char 1:3>` it appends to a device's is."""
    if raw is None:
        return None
    if raw.endswith(">") and "<" in raw:  # `/dev/null<char 1:3>`
        raw = raw[: raw.rindex("<")]
    if raw.endswith(" (deleted)"):
        raw = raw[: -len(" (deleted)")]
    return unescape(raw)


def c_string(tok: str):
    """A quoted string argument, unescaped; `None` if `tok` is not one."""
    if not tok.startswith('"'):
        return None
    end = tok.rfind('"')
    if end <= 0:
        return None
    return unescape(tok[1:end])


def unescape(raw: str) -> str:
    """strace's escapes (`\\303\\251`, `\\n`, `\\\\`, `\\"`) undone, the bytes read as UTF-8."""
    if "\\" not in raw:
        return raw
    try:
        return raw.encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw


def normpath(path: str) -> str:
    """`os.path.normpath` as Linux resolves names: POSIX lets a leading `//` mean something
    else and `normpath` keeps it, but on Linux `//d` is `/d` (and `//` is `/`)."""
    p = os.path.normpath(path)
    return p[1:] if p.startswith("//") else p


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
HANDLED = OPENS | READS | WRITES | CLONES | DUPS | set(FD_OPS) | set(PATH_OPS) | {"close", "lseek", "io_setup", "io_submit", "io_getevents", "io_pgetevents", "chdir", "fchdir", "mmap"}


class Tracer:
    """Reads `strace` lines and feeds a `Metrics`. Descriptor tables follow `clone` when the
    trace has it (`-e trace=…,%process`); otherwise a thread that uses a descriptor it never
    opened joins the table of the one that did (the `-yy` path decides which)."""

    def __init__(self, metrics: Metrics, roots: list[str], exclude: list[str] = (), cwd: str | None = None, instance_roots: list[int] = (), chain_gap_us: float | None = None):
        self.m = metrics
        self.roots = [normpath(r) for r in roots]
        self.exclude = list(exclude)
        self.cwd = cwd
        self.instance_roots = set(instance_roots)
        self.chain_gap = None if chain_gap_us is None else chain_gap_us / 1e6
        self.tables: dict[int, dict[int, File]] = {}
        self.parent: dict[int, int] = {}
        self.instance_of: dict[int, int | None] = {}
        self.cwds: dict[int, str] = {}
        self.pending: dict[int, tuple] = {}
        self.closing: dict[int, "object"] = {}      # tid -> the file its unfinished close took from the table
        self.chain: dict[int, list] = {}  # tid -> [length, time the last round ended]
        # AIO requests submitted and not yet reaped: (context, aio_data) -> the requests, oldest first
        self.aio: dict[tuple, list] = {}
        # AIO contexts `io_setup` made (inside an instance) that no `io_submit` has named yet
        self.aio_unused: Counter = Counter()
        self.clock = 0.0
        self.relative = False
        self.notes: Counter = Counter()
        self.other: Counter = Counter()
        self.tids: set[int] = set()
        self._under: dict[str, str | None] = {}
        # the size of each file under the roots that the trace made known (created, truncated,
        # or renamed into place since), kept as its writes extend it: an `O_APPEND` write's base
        self.extent: dict[str, int] = {}
        # `aeiou-trace export`: an `Exporter` that gets every counted call
        self.export: Exporter | None = None
        # the lines read, and those of them that parsed as a finished call strace printed
        # (what the tool says when nothing was counted: the wrong root, or not an strace)
        self.lines_read = 0
        self.calls_parsed = 0

    # ---- paths

    def under(self, path: str | None) -> str | None:
        """The path as counted (normalized), or `None` when it is outside the roots or excluded."""
        if path is None:
            return None
        hit = self._under.get(path, 0)
        if hit != 0:
            return hit
        p = normpath(path)
        out = None
        for r in self.roots:
            base = r.rstrip("/") + "/"   # the root `/` is its own prefix
            if p == r or p.startswith(base):
                rel = p[len(base) :]
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
        waiting = [(p, pend) for p, pend in self.pending.items() if pend[0] in CLONES and p != tid]   # never its own child
        if len(waiting) == 1:
            p, pend = waiting[0]
            return self._child(p, tid, shares_files(pend[0], pend[1]))
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
            self.lines_read += 1
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
            if name == "close":                        # the kernel frees the descriptor as the close begins: another
                a = split_args(body[6 : -len(UNFINISHED)].rstrip())   # thread's open can take it before this resumes
                fd, path = fd_arg(a[0]) if a else (None, None)
                table = self.table(tid)
                f = table.get(fd)
                if f is not None and (path is None or f.path == path):
                    del table[fd]
                    self.closing[tid] = f
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
        self.calls_parsed += 1
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
                self._child(tid, ret, shares_files(name, ", ".join(a)))
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
            p = self.under(path)
            if ret >= 0:
                if p and ("O_TRUNC" in flags or ("O_CREAT" in flags and "O_EXCL" in flags) or name == "creat"):
                    self.extent[p] = 0
                if "O_APPEND" in flags and p and p not in self.extent:
                    self.notes["append_to_unknown_size"] += 1
                f = self.table(tid)[ret] = File(path, 0, "O_APPEND" in flags)
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
            f = self.closing.pop(tid, None)                # taken from the table when the close began
            if f is None:
                f = table.get(fd)
                if ret == 0 and f is not None and (path is None or f.path == path):
                    del table[fd]
            if path is None and f is not None:
                path = f.path
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
                if len(a) > 1 and truncated(a[1]):
                    # the iovecs past the 32nd are not in the trace: a successful call's
                    # return value is a floor on the length the whole list asked for
                    length = max(length, ret)
                    if inst is not None and self.under(path or (f.path if f else None)):
                        self.notes["iov_arrays_truncated"] += 1
                off_tok = a[3] if name.startswith("p") and len(a) > 3 else None
            else:
                length = int(a[2], 0) if len(a) > 2 and a[2].lstrip("-").isdigit() else 0
                off_tok = a[3] if name.startswith("p") and len(a) > 3 else None
            positioned = off_tok is not None and off_tok.lstrip("-").isdigit() and int(off_tok) >= 0
            p = self.under(path)
            base = self.extent.get(p) if kind == W and f is not None and f.append and p else None
            if base is not None:
                # `O_APPEND`: at the end of the file, a `pwrite`'s offset ignored, the
                # position left at the end
                offset, positioned = base, True
                if ret > 0:
                    f.pos = base + ret
            elif positioned:
                offset = int(off_tok)
            else:
                offset = f.pos if f is not None else -1
                if f is not None and ret > 0:
                    f.pos += ret
            if kind == W and ret > 0 and offset >= 0 and p in self.extent:
                self.extent[p] = max(self.extent[p], offset + ret)
            if inst is not None and p:
                self._end_chain(tid)
                if offset < 0:
                    self.notes["data_ops_without_offset"] += 1
                m.data(kind, inst, tid, p, offset, length, max(ret, 0))
                if ex is not None and f is not None and f.oid is not None and offset >= 0:
                    if base is not None:
                        ex.opens[f.oid]["pinned"] = True
                        ex.notes["append_positions_resolved"] += 1
                    ex.data(when, KINDS[kind], f.oid, offset, length, positioned, ret, err)
            return
        if name == "mmap":
            # exported only (the metrics do not count a mapping, `runner/REFERENCE.md` §10): the
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
        if name == "io_setup":
            # `io_setup(nr, [ctx]) = 0`: a context no `io_submit` names by the end is noted
            if ret == 0 and inst is not None and len(a) > 1:
                self.aio_unused[a[1].strip("[]")] += 1
            return
        if name == "io_submit":
            if a:
                self.aio_unused.pop(a[0], None)
            got = []
            iocbs = _IOCB.findall(a[2] if len(a) > 2 else "")
            if inst is not None and len(a) > 2 and truncated(a[2]) and ret > len(iocbs):
                # the requests past the 32nd are not in the trace, so neither is what or
                # where they read or wrote: not counted, and the fan-out is short by them
                self.notes["aio_requests_truncated"] += ret - len(iocbs)
            for data, op, fd, path, nbytes, off in iocbs:
                path = fd_path(path)
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
                for data, op, fd, path, nbytes, off in iocbs:
                    path = fd_path(path)
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
            events = _EVENT.findall(a[3] if len(a) > 3 else "")
            if len(a) > 3 and truncated(a[3]) and ret > len(events) and any(k[0] == ctx for k in self.aio):
                # results the trace cut off: their requests stay waiting, and are taken as
                # complete at the end (`aio_results_assumed_complete`)
                self.notes["aio_events_truncated"] += ret - len(events)
            for data, res in events:
                waiting = self.aio.get((ctx, data))
                if waiting:
                    kind, i, t, p, off, nbytes, member = waiting.pop(0)
                    if not waiting:
                        del self.aio[(ctx, data)]
                    m.data(kind, i, t, p, off, nbytes, max(int(res), 0))
                    if kind == W and p in self.extent and int(res) > 0:
                        self.extent[p] = max(self.extent[p], off + int(res))
                    if member is not None:
                        member["ret"] = int(res) if int(res) >= 0 else _errno_name(-int(res))
            return
        if name in FD_OPS:
            fd, path = fd_arg(a[0]) if a else (None, None)
            f = self.file(tid, fd, path)
            if path is None:
                path = f.path if f else None
            p = self.under(path)
            if ret == 0 and p in self.extent:
                try:
                    if name == "ftruncate":
                        self.extent[p] = int(a[1], 0)
                    elif name == "fallocate" and a[1] == "0":
                        self.extent[p] = max(self.extent[p], int(a[2], 0) + int(a[3], 0))
                except (IndexError, ValueError):
                    del self.extent[p]
            if FD_OPS[name] == "readdir" and f is not None:
                # the abstract's `readdir` is one listing of a directory: the calls that
                # continue it, and the empty one that ends it, are the same op
                if f.listed:
                    return
                f.listed = True
            if inst is not None and p:
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
            if ret == 0 and p:
                self._known_size(tid, name, kind, p, a, di, pi)
            if inst is not None and p:
                self._count(tid, kind)
                if ex is not None:
                    to = None
                    if kind == "rename":
                        j = pi + 1 if di is None else pi + 2
                        to = self.under(self.resolve(tid, a[j - 1] if di is not None else None, c_string(a[j]))) if len(a) > j else None
                    ex.path_op(when, kind, name, p, a, pi, to, err)
            return

    def _known_size(self, tid: int, name: str, kind: str, p: str, a: list[str], di, pi: int):
        """What a successful path call does to the sizes the trace knows (`extent`)."""
        ext = self.extent
        if kind == "unlink":
            ext.pop(p, None)
        elif name == "truncate":
            try:
                ext[p] = int(a[pi + 1], 0)
            except (IndexError, ValueError):
                ext.pop(p, None)
        elif kind == "rename":
            j = pi + 1 if di is None else pi + 2
            to = self.under(self.resolve(tid, a[j - 1] if di is not None else None, c_string(a[j]))) if len(a) > j else None
            size = ext.pop(p, None)
            if to:
                if size is None:
                    ext.pop(to, None)
                else:
                    ext[to] = size

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
        # contexts made and never submitted to: unused, or `io_submit` left out of the trace's
        # `-e trace=` (strace classes no AIO call in `%desc`), and every request lost with it
        unused = sum(self.aio_unused.values())
        if unused:
            self.notes["aio_contexts_without_io_submit"] += unused
        self.aio_unused = Counter()
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
            dur = 0
        else:
            # `t` is set in `finish`, from the earliest start: the lines are in the order the
            # calls returned, and a call (an `<unfinished ...>` one) may have started before
            # the call that returned first
            if self.t0 is None or issued < self.t0:
                self.t0 = issued
            dur = int(round((done - issued) * 1e9)) if done is not None else 0
        e = {"lane": self._lane(tid), "t": issued, "dur": max(dur, 0), "op": op}
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
        """The path below the root (the root itself is `.`). `Tracer.under` hands over only
        normalized paths at or under the normalized root, so the result is never absolute,
        never empty, and has no `..`: what the runner accepts (`trace.rs`, `check_path`)."""
        if path == self.root or path + "/" == self.root:
            return "."
        out = path[len(self.root) :] if path.startswith(self.root) else None
        if not out or out.startswith("/") or ".." in out.split("/"):
            raise AssertionError(f"aeiou-trace export: internal error: {path!r} is not a normalized path under the root {self.root!r}")
        return out

    @staticmethod
    def _ret(ret, err):
        if err is not None:
            return err
        return ret if ret not in (None, 0) else None

    def open(self, when, path: str, flags: str, mode: str | None, err: str | None) -> int | None:
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
        if ("CREAT" in names or "TRUNC" in names) and self.rel(path) == ".":
            # the runner refuses a trace that changes --root itself: the open is left out,
            # and the calls on its descriptor with it
            self.notes["calls_not_exported"] += 1
            self.notes["not_exported:open:creates or truncates the root"] += 1
            return None
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
        if kind != "stat" and (self.rel(path) == "." or (to is not None and self.rel(to) == ".")):
            # `mkdir` of the root (an `os.makedirs(..., exist_ok=True)`), its `rmdir`: the
            # runner refuses a trace that changes --root itself, whatever the call returned
            self.notes["calls_not_exported"] += 1
            self.notes[f"not_exported:{kind}:the root itself"] += 1
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
            e["t"] = 0 if e["t"] is None else int(round((e["t"] - self.t0) * 1e9))
            if e["op"] in KINDS and "_off" in e:
                off = e.pop("_off")
                o = self.opens[e["fd"]]
                if "offset" not in e and (len(o["lanes"]) > 1 or o.get("pinned")):
                    # an open some `O_APPEND` write of was written positioned: the runner's
                    # position for it is not the application's after that write
                    e["offset"] = off
                    shared += len(o["lanes"]) > 1
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


def export_of(lines, root: str, seen: dict | None = None, **kw) -> tuple[dict, list[dict]]:
    """Header and op lines of the runner's trace file for an strace given as lines; `seen`,
    when given, gets the lines read and those parsed as calls (`nothing_counted`)."""
    m = Metrics()
    t = Tracer(m, [root], **kw)
    t.export = Exporter(t.roots[0])
    t.feed(lines)
    t.finish()
    if seen is not None:
        seen.update(lines=t.lines_read, calls=t.calls_parsed)
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
        # read as the runner reads the file (`emit.loads`, serde_json's JSON), a line that is
        # not JSON refused with its number as `trace::TraceFile::load` refuses it
        from .emit import loads
        for no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = loads(line)
            except ValueError as e:
                raise BuildError(f"line {no}: {e}") from None
            if not isinstance(doc, dict):
                raise BuildError(f"line {no}: not a JSON object")
            if self.header is None:
                if doc.get("aeiou_trace") != TRACE_FORMAT:
                    raise BuildError(f"not an aeiou_trace: {TRACE_FORMAT} file")
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


def metrics_of_export(lines, block=None) -> dict:
    m = Metrics(block)
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
        "order": "completion",
        "instances": 1,
        "contexts": len(r.lanes),
        "chain_gap_us": None,
        "notes": dict(h.get("notes", {})),
        "total": total,
    }


def metrics_of(lines, roots, block=None, seen: dict | None = None, **kw) -> dict:
    """The `aeiou_metrics: 2` document of a trace given as an iterable of lines; `seen`, when
    given, gets the lines read and those parsed as calls (`nothing_counted`)."""
    m = Metrics(block)
    t = Tracer(m, roots, **kw)
    t.feed(lines)
    t.finish()
    if seen is not None:
        seen.update(lines=t.lines_read, calls=t.calls_parsed)
    notes = dict(sorted(t.notes.items()))
    if t.chain_gap is None and m.fan_out:
        notes["depth_not_computed"] = "give --chain-gap-us: the think time that ends a chain of io_submit rounds"
    if any(k in notes for k in ("aio_requests_truncated", "aio_events_truncated", "iov_arrays_truncated")):
        notes["arrays_truncated"] = "trace again with strace -v: without it strace prints only the first 32 elements of an array"
    total = m.out()
    if t.chain_gap is None:
        total["depth"] = {}
    return {
        "aeiou_metrics": FORMAT,
        "source": "strace",
        "roots": t.roots,
        "block": m.block,
        "order": "completion",
        "instances": max(len(m.instance_ids), 1 if not t.instance_roots else 0),
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
        raise BuildError(f"no template `{template}` in the document (it has: {', '.join(doc.get('templates', {})) or 'none'})")


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
    # the locality sections both have (DESIGN_REVIEW.md §3.79): extents, and blocks, which a
    # capture taken before 2026-10-22 has alone
    sections = [(key, prefix, unit) for key, prefix, unit in (("reuse_extents", "extent reuse", None), ("reuse_distance_bytes", "reuse", "block"))
                if key in a and key in b]
    if not sections:
        raise BuildError("no locality metrics in common: one document has extents only and the other blocks only "
                         "(take the dry run with --metrics-block, the capture's block size)")
    for key, prefix, unit in sections:
        ra, rb = a[key], b[key]
        for this in KINDS:
            per = f"block {this}s" if unit else f"bytes {'read' if this == 'read' else 'written'}"
            def accesses(r):
                return r["first_touch"][this] + sum(r[f"{this}_after_{prev}"]["n"] for prev in KINDS)
            if accesses(ra) or accesses(rb):
                pct(f"{prefix}: first touches / {per}", share(ra["first_touch"][this], accesses(ra)), share(rb["first_touch"][this], accesses(rb)))
            for prev in KINDS:
                k = f"{this}_after_{prev}"
                if ra[k]["n"] or rb[k]["n"]:
                    pct(f"{prefix}: {this} after {prev} / {per}", share(ra[k]["n"], accesses(ra)), share(rb[k]["n"], accesses(rb)))
                hist(f"{prefix} distance, {this} after {prev}", ra[k], rb[k])
        if not unit and (ra["whole"] + ra["partial"] or rb["whole"] + rb["partial"]):
            pct(f"{prefix}: partial re-reads / bytes read again", share(ra["partial"], ra["whole"] + ra["partial"]), share(rb["partial"], rb["whole"] + rb["partial"]))
    for unit in [u for u in ("bytes", "blocks") if f"popularity_{u}" in a and f"popularity_{u}" in b] + ["objects"]:
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
    "extent reuse": 0.10,
    "extent reuse distance": 0.10,
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


def _cause(e: BaseException) -> str:
    """An OSError as the runner prints one (`No such file or directory (os error 2)`)."""
    if isinstance(e, OSError) and e.strerror:
        return f"{e.strerror} (os error {e.errno})"
    return str(e)


def load_tolerances(path: str) -> dict:
    """A tolerance file: `{"aeiou_tolerances": 1, "tolerances": {class: x},
    "unseen": [{"metric": prefix, "reason": text}], "outside": [the same]}`. `unseen` names
    rows the trace cannot show (they are not judged); `outside` records rows known to be
    outside, with the reason (they are judged, and still outside)."""
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise BuildError(f"{path}: {_cause(e)}")
    if not isinstance(spec, dict) or spec.get("aeiou_tolerances") != TOLERANCES_FORMAT:
        raise BuildError(f"{path}: not an aeiou_tolerances: {TOLERANCES_FORMAT} document")
    extra = set(spec) - {"aeiou_tolerances", "tolerances", "unseen", "outside", "comment"}
    if extra:
        raise BuildError(f"{path}: unknown key(s) {', '.join(sorted(extra))}")
    for c, x in spec.get("tolerances", {}).items():
        if c not in TOLERANCES or not isinstance(x, (int, float)) or isinstance(x, bool) or not 0 <= x <= 1:
            raise BuildError(f"{path}: tolerances: `{c}` must be one of {', '.join(TOLERANCES)} with a value in [0, 1]")
    for key in ("unseen", "outside"):
        for e in spec.get(key, []):
            if not isinstance(e, dict) or set(e) != {"metric", "reason"} or not all(isinstance(v, str) and v for v in e.values()):
                raise BuildError(f"{path}: {key}: each entry is {{\"metric\": prefix, \"reason\": text}}")
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


def nothing_counted(roots: list[str], seen: dict) -> str:
    """Why a trace gave no call to count: the roots matched none of its calls, or it is no strace."""
    why = ("check --root against the paths the trace shows" if seen["calls"] else
           "not an strace (strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,... -o FILE)")
    return f"no call under --root {', '.join(roots)} ({seen['lines']} line(s) read, {seen['calls']} parsed as strace calls): {why}"


def _describe(doc: dict) -> str:
    src = doc.get("source", "dry-run")
    if src == "strace":
        return f"strace of {', '.join(doc.get('roots', []))} (order: completion; {doc.get('instances')} instance(s), {doc.get('contexts')} thread(s))"
    if src == "trace":
        return f"trace file {doc.get('trace')} of {', '.join(doc.get('roots', []))} (order: completion; {doc.get('instances')} instance(s), {doc.get('contexts')} lane(s))"
    return f"{src} of {doc.get('abstract')} seed {doc.get('seed')} gpus {doc.get('gpus')} (order: {doc.get('order')})"


def parser() -> tuple[usage.Parser, dict[str, usage.Parser]]:
    """The parser and its subcommands' parsers by name (the usage a `Missing` list shows)."""
    ap = usage.Parser(prog="aeiou-trace", description="Locality metrics of a real application from an strace of it, their comparison with an abstract's (aeiou dry-run --metrics-json), the runner's trace file, and the structural digest a fit is made from",
                      usage="aeiou-trace <COMMAND> [ARGS]...")
    sub = ap.add_subparsers(dest="cmd")
    mp = sub.add_parser("metrics", help="Compute the metrics of a trace (strace -f -ttt -T -yy -v -e trace=%%file,%%desc,%%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o FILE)", usage="aeiou-trace metrics [OPTIONS] <TRACE>")
    mp.add_argument("trace", nargs="?", help="The strace output file, or - for stdin")
    mp.add_argument("--root", action="append", default=[], metavar="DIR", help="Count only calls on paths under DIR (repeatable; required for an strace, not for an exported trace file)")
    mp.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    mp.add_argument("--cwd", metavar="DIR", help="Resolve relative paths against DIR when the trace does not say (no -y)")
    mp.add_argument("--metrics-block", type=int, metavar="BYTES", help="The block metrics too, reuse and popularity in blocks of BYTES, as captures taken before 2026-10-22 have them (the extent metrics are always there)")
    mp.add_argument("--instance-root", action="append", type=int, default=[], metavar="PID", help="The process tree under PID is one instance (repeatable; needs clone in the trace); default: the whole trace is one instance")
    mp.add_argument("--chain-gap-us", type=usage.non_negative, metavar="US", help="Report depth: consecutive io_submit rounds of a thread form a chain until more than US microseconds pass between a round's end and the next submit")
    mp.add_argument("-o", "--out", metavar="FILE", help="Write the JSON document (aeiou_metrics: 2) to FILE instead of stdout")
    ep = sub.add_parser("export", help="Write the runner's trace file for a `trace` node from an strace (DESIGN_REVIEW.md §3.58): JSON Lines, one lane per traced task, opens by id, paths relative to --root", usage="aeiou-trace export [OPTIONS] --root <DIR> -o <FILE> <TRACE>")
    ep.add_argument("trace", nargs="?", help="The strace output file, or - for stdin")
    ep.add_argument("--root", metavar="DIR", help="Required: export only calls on paths under DIR, written relative to it")
    ep.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    ep.add_argument("--cwd", metavar="DIR", help="Resolve relative paths against DIR when the trace does not say (no -y)")
    ep.add_argument("-o", "--out", metavar="FILE", help="Required: the trace file to write; its sha256 is printed, for the abstract's `trace` node")
    cp = sub.add_parser("compare", help="Compare two metrics documents, from a trace or from aeiou dry-run --metrics-json", usage="aeiou-trace compare [OPTIONS] <A> <B>")
    cp.add_argument("a", nargs="?", help="A metrics document (aeiou_metrics: 2, or 1 from before 2026-10-22), the trace's")
    cp.add_argument("b", nargs="?", help="The metrics document A is compared with, the abstract's")
    cp.add_argument("--template-a", metavar="NAME", help="Use this actor template's metrics of A instead of its total")
    cp.add_argument("--template-b", metavar="NAME", help="Use this actor template's metrics of B instead of its total")
    cp.add_argument("--max-distance", type=usage.non_negative, metavar="X", help="Exit 1 if any distance exceeds X (one number for every row; --judge is the rule by class)")
    cp.add_argument("--judge", action="store_true", help="Judge each row against the tolerance of its class (A the trace, B the abstract) and exit 1 if any is outside")
    cp.add_argument("--tolerances", metavar="FILE", help="A tolerance file (aeiou_tolerances: 1): class tolerances that replace the defaults, the rows the trace cannot show, the rows recorded as outside; implies --judge")
    cp.add_argument("--self", dest="selfs", action="append", default=[], metavar="FILE", help="A metrics document of B's abstract at the same parameters and another seed (repeatable): a row's tolerance is raised by its largest distance between B and these; implies --judge")
    cp.add_argument("--only", action="append", default=[], metavar="PREFIX", help="Only the rows whose metric starts with PREFIX (repeatable)")
    dp = sub.add_parser("digest", help="Write the structural digest of a trace: path templates, lanes and their phases, a verdict per site on what the trace decides (permutation or draw, order, size), and the open questions ranked by their divergence at scale", usage="aeiou-trace digest [OPTIONS] --root <DIR> <TRACE>")
    dp.add_argument("trace", nargs="?", help="The strace output file, or an exported trace file (a file, never stdin: the comparison after the fit and a re-run with answers read it again)")
    dp.add_argument("--root", metavar="DIR", help="Digest only calls on paths under DIR, written relative to it (required for an strace, not for an exported trace file)")
    dp.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Leave out paths whose part below the root matches GLOB (repeatable; * crosses /)")
    dp.add_argument("--cwd", metavar="DIR", help="Resolve relative paths against DIR when the trace does not say (no -y)")
    dp.add_argument("--epochs", type=int, default=1, metavar="N", help="Project the open questions over N passes over the set (default 1): the share a draw with replacement never reads is e^-N")
    dp.add_argument("--candidates", type=int, metavar="N", help="The files the application could have chosen among (the corpus); default: the distinct paths the trace touched. Decides a pass from a draw when no path repeats")
    dp.add_argument("--paths", action="store_true", help="Include every raw path under each template (the digest otherwise carries templates and counts only)")
    dp.add_argument("--text", action="store_true", help="Print the text report instead of the JSON document (with -o, the report is printed beside the file anyway)")
    dp.add_argument("-o", "--out", metavar="FILE", help="Write the JSON document (aeiou_digest: 1) to FILE and print the text report")
    dp.add_argument("--answers", metavar="FILE", help="The answers file (aeiou_answers: 1): replayed when it exists, so the questions are not asked; written there after an interactive run. Default after an interactive run with -o: the document's name with .answers.json")
    dp.add_argument("--no-questions", action="store_true", help="Ask nothing even on a terminal: every undecided verdict stays open")
    dp.add_argument("--answered-by", metavar="NAME", help="Record NAME as the one who answered (default: user@host)")
    ap.version(f"aeiou-trace {__version__}")
    return ap, {"metrics": mp, "export": ep, "compare": cp, "digest": dp}


@usage.tool
def main(argv=None) -> int:
    ap, subs = parser()
    checkout.warn(ap.prog)
    sp = ap
    try:
        args = ap.parse_args(argv)
        missing = usage.Missing(ap)
        missing.want(args.cmd, "<COMMAND>", "one of metrics, export, compare, digest")
        missing.check()
        sp = subs[args.cmd]
        missing = usage.Missing(sp)
        if args.cmd in ("metrics", "export"):
            missing.want(args.trace, "<TRACE>", "the strace output file, or - for stdin")
        if args.cmd == "digest":
            missing.want(args.trace, "<TRACE>", "the strace output file, or an exported trace file")
        if args.cmd == "export":
            missing.want(args.root, "--root <DIR>", "the directory the traced application's paths are under; the export is relative to it")
            missing.want(args.out, "-o <FILE>", "the trace file to write")
        if args.cmd == "compare":
            missing.want(args.a, "<A>", "a metrics document (aeiou_metrics: 2, or 1 from before 2026-10-22), the trace's")
            missing.want(args.b, "<B>", "the metrics document A is compared with, the abstract's")
        missing.check()
        if args.cmd == "compare" and args.max_distance is not None and (args.judge or args.tolerances or args.selfs):
            rule = "--judge" if args.judge else "--tolerances (which implies --judge)" if args.tolerances else "--self (which implies --judge)"
            raise usage.UsageError(f"--max-distance cannot be used with {rule}: they are two rules; give one", sp)
        return _main(args, subs)
    except usage.UsageError as e:
        return usage.fail(e, sp)
    except BuildError as e:
        return usage.failed(sp.prog, e)


def _main(args, subs: dict[str, usage.Parser]) -> int:
    mp = subs["metrics"]
    if args.cmd == "metrics":
        if args.metrics_block is not None and args.metrics_block < 1:
            mp.error("--metrics-block must be at least 1")
        f = _open_trace(args.trace)
        with f:
            first = f.readline()
            lines = itertools.chain([first], f)
            if first.startswith('{"aeiou_trace"'):
                try:
                    doc = metrics_of_export(lines, args.metrics_block)
                except BuildError as e:
                    raise BuildError(f"{args.trace}: {e}") from None
                if not doc["total"]["ops"]:
                    raise BuildError(f"{args.trace}: a trace file with no call in it")
            elif not args.root:
                missing = usage.Missing(mp)
                missing.need(False, "--root <DIR>", "count only calls on paths under DIR (repeatable)", "needed for an strace; an exported trace file carries its root")
                missing.check()
            else:
                seen = {}
                doc = metrics_of(lines, [os.path.abspath(r) for r in args.root], args.metrics_block, seen=seen,
                                 exclude=args.exclude, cwd=args.cwd, instance_roots=args.instance_root, chain_gap_us=args.chain_gap_us)
                if not doc["total"]["ops"]:
                    raise BuildError(f"{args.trace}: {nothing_counted(doc['roots'], seen)}")
        doc["trace"] = args.trace
        text = json.dumps(doc, indent=2) + "\n"
        if args.out:
            try:
                with open(args.out, "w", encoding="utf-8") as out:
                    out.write(text)
            except OSError as e:
                raise BuildError(f"{args.out}: {_cause(e)}")
            t = doc["total"]
            print(f"{t['ops']} ops under {', '.join(doc['roots'])}: read {t['bytes_read']} bytes, wrote {t['bytes_written']} bytes; {doc['contexts']} thread(s); {args.out}")
            for k, v in doc["notes"].items():
                print(f"note: {k}: {v}")
        else:
            sys.stdout.write(text)
        return 0

    if args.cmd == "digest":
        return _digest(args, subs["digest"])

    if args.cmd == "export":
        f = _open_trace(args.trace)
        seen = {}
        with f:
            header, events = export_of(f, os.path.abspath(args.root), seen=seen, exclude=args.exclude, cwd=args.cwd)
        if not events:   # never an empty trace file: a `trace` node of it would run nothing
            raise BuildError(f"{args.trace}: {nothing_counted([header['root']], seen)}; {args.out} not written")
        try:
            sha = write_export(args.out, header, events)
        except OSError as e:
            raise BuildError(f"{args.out}: {_cause(e)}")
        print(f"{header['lines']} line(s) on {header['lanes']} lane(s), {header['opens']} open(s), {len(header['creates'])} path(s) created under {header['root']}; {args.out}")
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
            raise BuildError(f"{path}: {_cause(e)}")
        if not isinstance(d, dict) or d.get("aeiou_metrics") not in FORMATS_READ:
            raise BuildError(f"{path}: not an aeiou_metrics: {FORMAT} document (or version 1, a capture's from before 2026-10-22)")
        docs.append(d)
    a, b = select(docs[0], args.template_a), select(docs[1], args.template_b)
    # where only the blocks are in common, they are in common only at one size
    if not ("reuse_extents" in a and "reuse_extents" in b) and docs[0].get("block") != docs[1].get("block"):
        raise BuildError(f"block sizes differ ({docs[0].get('block')} and {docs[1].get('block')}): reuse distance and block popularity are not comparable")
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
        print(f"aeiou-trace compare: exceeds --max-distance {args.max_distance}", file=sys.stderr)
        return 1
    return 0


def _digest(args, dp: usage.Parser) -> int:
    from . import digest as dg
    if args.epochs < 1:
        dp.error("--epochs must be at least 1")
    if args.candidates is not None and args.candidates < 1:
        dp.error("--candidates must be at least 1")
    if args.trace == "-":
        dp.error("the digest reads the trace as a file, never stdin: the comparison after the fit and a re-run with answers read it again")
    f = _open_trace(args.trace)
    with f:
        first = f.readline()
        lines = itertools.chain([first], f)
        if not first.startswith('{"aeiou_trace"') and not args.root:
            missing = usage.Missing(dp)
            missing.need(False, "--root <DIR>", "digest only calls on paths under DIR", "needed for an strace; an exported trace file carries its root")
            missing.check()
        try:
            an = dg.analyze(lines, args.root, exclude=args.exclude, cwd=args.cwd)
        except BuildError as e:
            raise BuildError(f"{args.trace}: {e}") from None
    sha = _sha256_of(args.trace)
    by = args.answered_by or _who()
    at = _now()
    kw = dict(epochs=args.epochs, candidates=args.candidates, paths=args.paths)
    said = []
    if args.answers and os.path.exists(args.answers):
        header, answers = dg.read_answers(args.answers)
        by, at = header.get("by") or by, header.get("at") or at
        doc = dg.document(an, answers=answers, by=by, at=at, **kw)
        if header.get("trace_sha256") != sha:
            doc["notes"]["answers_trace"] = f"{args.answers} was answered on another trace ({header.get('trace')}); its answers were checked against this one"
    elif args.no_questions or not sys.stdin.isatty():
        if args.answers:
            raise BuildError(f"{args.answers}: no such answers file, and no terminal to ask on")
        doc = dg.document(an, by=by, at=at, **kw)
    else:
        doc, answers = dg.interview(an, dg.Console(), by=by, at=at, **kw)
        if answers:
            path = args.answers or (_answers_beside(args.out) if args.out else None)
            if path:
                try:
                    dg.write_answers(path, answers, trace=args.trace, trace_sha256=sha, by=by, at=at)
                except OSError as e:
                    raise BuildError(f"{path}: {_cause(e)}")
                replay = ["aeiou-trace", "digest", args.trace] + (["--root", args.root] if args.root else []) + ["--answers", path] + (["-o", args.out] if args.out else [])
                said.append(f"answers: {path}")
                said.append("replay without questions: " + " ".join(replay))
            else:
                said.append("answers: in the document only (give -o or --answers FILE to keep them as a file)")
    doc["trace"] = args.trace
    doc["trace_sha256"] = sha
    report = dg.render(doc)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as out:
                out.write(json.dumps(doc, indent=2) + "\n")
        except OSError as e:
            raise BuildError(f"{args.out}: {_cause(e)}")
        sys.stdout.write(report)
        print(f"written: {args.out}")
        for k, v in doc["notes"].items():
            print(f"note: {k}: {v}")
    elif args.text:
        sys.stdout.write(report)
    else:
        sys.stdout.write(json.dumps(doc, indent=2) + "\n")
    for line in said:
        print(line, file=sys.stderr)
    return 0


def _answers_beside(out: str) -> str:
    """The answers file an interactive run leaves beside the document it wrote."""
    return out[: -len(".digest.json")] + ".answers.json" if out.endswith(".digest.json") else out + ".answers.json"


def _sha256_of(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _who() -> str:
    import getpass
    import socket
    try:
        user = getpass.getuser()
    except (KeyError, OSError):   # no passwd entry for the uid (a container)
        user = str(os.getuid())
    return f"{user}@{socket.gethostname()}"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _open_trace(path: str):
    """The trace to read, `-` for stdin: UTF-8 whatever the locale, an undecodable byte
    replaced (strace escapes what it prints of buffers; a path in a foreign encoding is still
    a path to count)."""
    if path == "-":
        return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    try:
        return open(path, encoding="utf-8", errors="replace")
    except OSError as e:
        raise BuildError(f"{path}: {_cause(e)}")


def _judged(args, rows, a, b, docs) -> int:
    selfs = []
    for path in args.selfs:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise BuildError(f"{path}: {_cause(e)}")
        if not isinstance(d, dict) or d.get("aeiou_metrics") != docs[1].get("aeiou_metrics"):
            raise BuildError(f"{path}: --self is B's abstract at another seed, and its aeiou_metrics is not B's ({d.get('aeiou_metrics') if isinstance(d, dict) else None} and {docs[1].get('aeiou_metrics')})")
        for key in ("source", "sha256", "gpus", "block"):
            if d.get(key) != docs[1].get(key):
                raise BuildError(f"{path}: --self is B's abstract at another seed, and its `{key}` is not B's ({d.get(key)} and {docs[1].get(key)})")
        if d.get("seed") == docs[1].get("seed"):
            raise BuildError(f"{path}: --self has B's seed ({d.get('seed')}); another seed is the point")
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
