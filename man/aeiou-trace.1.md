# aeiou-trace(1)

## NAME

aeiou-trace - locality metrics of a real application from an strace of it, their comparison with an abstract's, and the trace file of the runner's `trace` node

## SYNOPSIS

```
aeiou-trace metrics [OPTIONS] [--root DIR]... [-o FILE] TRACE
aeiou-trace export [OPTIONS] --root DIR -o FILE TRACE
aeiou-trace compare [OPTIONS] A B
aeiou-trace --help
```

## DESCRIPTION

An abstract's fingerprint proves which operation stream ran; the locality metrics are for
proving it is the *right* stream, by comparison with the same numbers taken from a trace of
the real application the abstract models. `aeiou dry-run --metrics-json` (**aeiou**(1))
gives the abstract's numbers; **aeiou-trace** gives the same numbers, by the same
definitions and in the same document (`aeiou_metrics: 1`, with `"source": "strace"`),
from an `strace` of the application, and compares two such documents. It also exports an
strace as the file the runner's `trace` node executes literally. Standard library only.

The trace is taken with

```
strace -f -ttt -T -yy -e trace=%file,%desc,%process -o trace.txt <command>
```

`-f` follows threads and children; `-yy` puts the path behind every descriptor on the
line, which is how a call is known to be on the storage under test; `%process` records the
`clone`s, so descriptor tables and process trees are followed; `-ttt -T` are needed only
for `--chain-gap-us` and for `export`'s timings.

**What counts.** Calls on paths under `--root`, less `--exclude`. Library loading, `/proc`,
pipes, and sockets are not the workload. `open`, `openat`, `creat` are `open`; `read`,
`pread64`, `readv`, `preadv` are `read` and likewise for `write`; `fstat` and `statx` on a
descriptor are `fstat`, on a path `stat`; `getdents64` is `readdir`, once per listing;
`unlinkat` with `AT_REMOVEDIR` is `rmdir`; `lseek`, `ioctl`, `fsync`, `fdatasync`,
`ftruncate`, `fallocate`, `mkdir`, `rename`, `fadvise64`, and `close` by name. A failed call
counts as its op. The position of a sequential `read` or `write` is tracked per open file
description from `open`, `lseek`, and the bytes each call moved, shared by `dup`ed
descriptors, by threads, and across `fork`. The requests of one `io_submit` are a fan-out,
reaped by the matching `io_getevents`. Not visible to `strace`: `io_uring` submissions and
page faults on a mapping; an application that reads through a mapping leaves every row
unseen.

## COMMANDS

### metrics

Computes the metrics document of a trace: the read/write mix, request sizes, reuse
distance (per instance, in the order the calls returned), sequential run lengths (per
thread, per path), popularity of blocks and objects, and, where the trace shows a fork,
fan-out and depth. An exported trace file (`export`'s output) measures too, with a lane as
the context and the file as one instance; the header is sniffed.

### export

Writes the runner's trace file for a `trace` node: JSON Lines, a header, then one op per
line in the order the calls returned, a lane per traced task, opens by id, paths relative
to the one `--root`. A sequential read or write on an open that more than one lane used is
written positioned at the offset the trace shows it used; an `mmap` of a file range is one
positioned `read` of the range; a call the schema has no op for is dropped and counted by
name in the header's `notes`. The file's sha256 is printed, for the abstract's
`w.trace("app.jsonl", sha256)`; the runner refuses the file when it differs.

### compare

Compares two metrics documents, *A* the trace's and *B* the abstract's, one row per metric
with both values and a distance in [0, 1]: the difference of two shares (mix, ops by kind,
first touches, bytes in multi-op runs, popularity of the top 0.1 %, 1 %, 10 %), the largest
difference between two histograms' cumulative shares (request size, run length, reuse
distance), or the total-variation distance of two exact distributions (fan-out, depth).
`--max-distance` is one bound for every row; `--judge` is the rule by class: each row's
allowed distance is the tolerance of its class plus the row's largest distance between *B*
and each `--self` document, the same abstract at another seed, so an abstract is not asked
to be nearer the trace than it is to itself. Rows with nothing on either side, popularity
rows over fewer than 10 units, and `depth` without `--chain-gap-us` are not judged. The
class tolerances built in are the repository's defaults: 0.05 for `mix`, `ops`,
`request size`, and `popularity`; 0.10 for `run length`, `reuse`, `reuse distance`,
`fan-out`, and `depth`. A tolerance file replaces them.

## OPTIONS

### aeiou-trace metrics

- *TRACE*

  The strace output file, or an exported trace file, or `-` for stdin.
- **--root** *DIR*

  Count only calls on paths under *DIR*. Repeatable. Required for an strace, not for an
  exported trace file.
- **--exclude** *GLOB*

  Leave out paths whose part below the root matches *GLOB*. Repeatable; `*` crosses `/`.
- **--cwd** *DIR*

  Resolve relative paths against *DIR* when the trace does not say (no `-y`).
- **--metrics-block** *BYTES*

  The block size of the reuse-distance and popularity units. Default 4096.
- **--metrics-sample** *N*

  Keep one block and one object in *N*, chosen by hash, and scale. Default 1, exact.
- **--instance-root** *PID*

  The process tree under *PID* is one instance; what is under none is left out.
  Repeatable; needs `clone` in the trace. Default: the whole trace is one instance.
- **--chain-gap-us** *US*

  Report depth: consecutive `io_submit` rounds of a thread form a chain until more than
  *US* microseconds pass between a round's end and the next submit. There is no default,
  because the think time that separates one search from the next is a property of the
  application; without the flag no depth is reported.
- **-o**, **--out** *FILE*

  Write the JSON document to *FILE* instead of stdout.

### aeiou-trace export

- *TRACE*

  The strace output file, or `-` for stdin.
- **--root** *DIR*

  Required. Export only calls on paths under *DIR*, written relative to it.
- **--exclude** *GLOB*

  As for `metrics`.
- **--cwd** *DIR*

  As for `metrics`.
- **-o**, **--out** *FILE*

  Required. The trace file to write; its sha256 is printed.

### aeiou-trace compare

- *A*

  A metrics document (`aeiou_metrics: 1`), the trace's.
- *B*

  The metrics document *A* is compared with, the abstract's.
- **--template-a** *NAME*

  Use this actor template's metrics of *A* instead of its total.
- **--template-b** *NAME*

  Use this actor template's metrics of *B* instead of its total.
- **--max-distance** *X*

  Exit 1 if any distance exceeds *X*. One number for every row.
- **--judge**

  Judge each row against the tolerance of its class and exit 1 if any is outside. The
  comparison exits 0 only when at least one row was judged and none is outside.
- **--tolerances** *FILE*

  A tolerance file (`aeiou_tolerances: 1`): `tolerances` replaces the class values;
  `unseen` lists rows the trace cannot show, with the reason (not judged); `outside` records
  rows known to be outside, with the reason (judged, still outside, the reason printed). An
  entry that matches no row, or that records as outside a row that is within, is reported.
  Implies `--judge`.
- **--self** *FILE*

  A metrics document of *B*'s abstract at the same parameters and `--gpus` and another seed.
  Repeatable; a row's tolerance is raised by its largest distance between *B* and these.
  The abstract's hash, `--gpus`, block, and sample are checked; the parameters cannot be.
  Implies `--judge`.
- **--only** *PREFIX*

  Only the rows whose metric name starts with *PREFIX*. Repeatable.

### Common

- **-h**, **--help**

  Print the help of the command or subcommand.

## FILES

- `*.metrics.json`

  A metrics document, `aeiou_metrics: 1`, from this tool or from `aeiou dry-run
  --metrics-json`. The histograms are in full, per actor template and in total.
- `*.tolerances.json`

  A tolerance file, `aeiou_tolerances: 1`, kept beside a capture kit's trace document.
- The exported trace file

  JSON Lines: a header `{"aeiou_trace": 1, "source", "root", "lanes", "lines", "opens",
  "creates", "notes"}`, then lines `{"lane", "t", "dur", "op", "fd", "path", ...}` with the
  schema's field names and enum words, `t` in nanoseconds from the first exported call,
  `fd` an open id (the ordinal of the `open` line), `ret` the traced result (a count, or an
  errno name that becomes the op's `expect`), and `submit` lines holding the members of one
  `io_submit`. Defined in `runner/REFERENCE.md` §13.
- `builder/traces/<workload>/`

  The capture kit of each real application traced so far: the traced script, the corpus
  writer, the fitted parameter file, the trace's metrics document and tolerance file.

## EXIT STATUS

- **0**

  Success. For `compare` with `--judge`, at least one row was judged and none is outside;
  with `--max-distance`, no row exceeds it.
- **1**

  A row outside its bound, a document the tool cannot read, a `--self` document for
  another abstract or shape, a trace with no counted call.
- **2**

  A usage error, in the frame every tool of the suite shares.

## EXAMPLES

Measure a trace of the application, measure the abstract at its fitted parameters, and judge
the abstract against three other seeds of itself:

```
strace -f -ttt -T -yy -e trace=%file,%desc,%process -o trace.txt python train.py /mnt/data/train
aeiou-trace metrics trace.txt --root /mnt/data -o trace.metrics.json
aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --seed 1 --metrics-json abstract.metrics.json
for s in 2 3 4; do aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --seed $s --metrics-json self$s.json; done
aeiou-trace compare trace.metrics.json abstract.metrics.json --judge --self self2.json --self self3.json --self self4.json
```

Export the same trace for the runner's `trace` node, and check that the exported file
measures as the strace did:

```
aeiou-trace export trace.txt --root /mnt/data -o app.jsonl       # prints the sha256
aeiou-trace metrics app.jsonl -o app.metrics.json
aeiou-trace compare trace.metrics.json app.metrics.json --max-distance 0
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-abstract**(7),
**strace**(1).

The reference: `builder/REFERENCE.md` §7 (the tool, the tolerances, the capture kits),
`runner/REFERENCE.md` §10 (the definitions of the metrics) and §13 (the `trace` node and
its file).
