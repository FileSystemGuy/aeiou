# aeiou-trace(1)

## NAME

aeiou-trace - locality metrics of a real application from an strace of it, their comparison with an abstract's, the trace file of the runner's `trace` node, and the structural digest a fit is made from

## SYNOPSIS

```
aeiou-trace metrics [OPTIONS] [--root DIR]... [-o FILE] TRACE
aeiou-trace export [OPTIONS] --root DIR -o FILE TRACE
aeiou-trace compare [OPTIONS] A B
aeiou-trace digest [OPTIONS] --root DIR [-o FILE] TRACE
aeiou-trace --help | --version
```

## DESCRIPTION

An abstract's fingerprint proves which operation stream ran; the locality metrics are for
proving it is the *right* stream, by comparison with the same numbers taken from a trace of
the real application the abstract models. `aeiou dry-run --metrics-json` (**aeiou**(1))
gives the abstract's numbers; **aeiou-trace** gives the same numbers, by the same
definitions and in the same document (`aeiou_metrics: 1`, with `"source": "strace"`),
from an `strace` of the application, and compares two such documents. It also exports an
strace as the file the runner's `trace` node executes literally, and digests one into the
structural summary an abstract is fitted from. Standard library only.

The trace is taken with

```
strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt <command>
```

`-f` follows threads and children; `-yy` puts the path behind every descriptor on the
line, which is how a call is known to be on the storage under test; `%process` records the
`clone`s, so descriptor tables and process trees are followed; the AIO calls are named
because `strace` classes none of them in `%desc`, and without them every libaio read and
write is lost (a context `io_setup` made and no `io_submit` named is noted,
`aio_contexts_without_io_submit`); `-v` prints whole arrays, where `strace` otherwise cuts
one at 32 elements and the requests of a larger `io_submit`, the results of an
`io_getevents`, and the iovecs of a `readv` or `writev` past the 32nd are lost (noted,
`aio_requests_truncated`, `aio_events_truncated`, `iov_arrays_truncated`, with
`arrays_truncated` saying to trace again); `-ttt -T` are needed only for `--chain-gap-us`
and for `export`'s timings.

**What counts.** Calls on paths under `--root`, less `--exclude`. Library loading, `/proc`,
pipes, and sockets are not the workload. `open`, `openat`, `creat` are `open`; `read`,
`pread64`, `readv`, `preadv` are `read` and likewise for `write`; `fstat` and `statx` on a
descriptor are `fstat`, on a path `stat`; `getdents64` is `readdir`, once per listing;
`unlinkat` with `AT_REMOVEDIR` is `rmdir`; `lseek`, `ioctl`, `fsync`, `fdatasync`,
`ftruncate`, `fallocate`, `mkdir`, `rename`, `fadvise64`, and `close` by name. A failed call
counts as its op. The position of a sequential `read` or `write` is tracked per open file
description from `open`, `lseek`, and the bytes each call moved, shared by `dup`ed
descriptors, by threads, and across `fork`. An `O_APPEND` write goes to the end of the file
wherever the trace knows the file's size (from an open that truncated or created it, kept
through every call that changes it); where it cannot, the write is counted from 0 and noted
(`append_to_unknown_size`). The requests of one `io_submit` are a fan-out,
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
written positioned at the offset the trace shows it used, as is every read and write on an
open whose `O_APPEND` write went to a known end of the file; an `mmap` of a file range is one
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

### digest

Writes the digest document (`aeiou_digest: 1`) of a trace, the structural summary a fit is
made from, without the trace leaving its owner: no raw path unless `--paths` asks, no data
bytes ever. It reads the same calls `export` writes, through the same resolver, or an
exported trace file. **Path templates**: every run of digits in a path under the root is a
field (`train/{a:05}/sample_{b:09}.npz`; a run of 16 or more hex digits with a letter in it a
hex field, `{a:x32}`), with how many values each field took and their range, and per
template the op mix, bytes, request sizes, and how each open moved through its file
(`whole`, one sequential run from offset 0; `sequential`, one run from elsewhere;
`positioned`, more than one run). **Lanes and phases**: a lane is a traced task; a unit is
the calls on one open from `open` to `close` (a path call or an `io_submit` is a unit of
one op); a unit's motif is its op sequence with repeats collapsed (`read+`, `(lseek
read)+`); a phase is a run of units of one motif on one template, with the repeat counts
and the think time between units. **Sites**: a template opened at least twice. Its
*selection* verdict says what the sequence of paths decides between a permutation pass
over the set (`consume`, each path once per pass) and a draw with replacement (`choose`): a
repeat before the set is exhausted refutes the pass; two or more passes of equal length
refute the draw; no repeat at all decides nothing unless the corpus (`--candidates`) is
small enough that a draw would have repeated (the birthday bound, at 5 %), and then the
verdict says how many touches would decide it. *Order* within a pass (`sequential`,
`permuted`, `other`) and the bytes per open (`constant`, `per-path`, `drawn`) are judged
the same way, each `undecided` with what settles it. **Open**: every undecided verdict as a
question in the application's terms, its candidate constructs, what evidence settles it,
and the candidates' divergence at `--epochs` passes (the share of the set a draw with
replacement never reads, e^-N, against zero for a pass), ranked largest first. The digest
asks nothing; a person or the fitting skill asks, and an answer is recorded beside the
capture with who gave it, never as a measurement.

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

  The block metrics too, reuse and popularity in blocks of *BYTES*, as captures taken
  before 2026-10-22 have them; the extent metrics, over byte ranges, are always there.
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

### aeiou-trace digest

- *TRACE*

  The strace output file, or an exported trace file, or `-` for stdin.
- **--root** *DIR*

  Digest only calls on paths under *DIR*, written relative to it. Required for an strace;
  an exported trace file carries its root, and a `--root` that is not it is refused.
- **--exclude** *GLOB*

  Leave out paths whose part below the root matches *GLOB*. Repeatable; `*` crosses `/`.
- **--cwd** *DIR*

  Resolve relative paths against *DIR* when the trace does not say (no `-y`).
- **--epochs** *N*

  Project the open questions over *N* passes over the set (default 1): a draw with
  replacement never reads a share e^-N of the set, a permutation pass reads all of it; the
  open list is ranked by that divergence. At least 1.
- **--candidates** *N*

  The files the application could have chosen among, the corpus. Default: the distinct
  paths the trace touched. With no repeat in the trace, a corpus of fewer than
  *k*(*k*-1)/(2 ln 20) files, *k* the touches, refutes a draw with replacement (no repeat
  would have had under 5 % probability); a larger one leaves the verdict undecided and the
  digest counts the touches that would decide it. Fewer than the paths seen is noted and
  the seen count used. At least 1.
- **--paths**

  Include every raw path under each template. Without it the document carries templates
  and counts only, which is what leaves the trace's owner.
- **--text**

  Print the text report instead of the JSON document.
- **-o**, **--out** *FILE*

  Write the JSON document to *FILE* and print the text report.

### Common

- **-h**, **--help**

  Print the help of the command or subcommand.
- **-V**, **--version**

  Print the version.

## FILES

- `*.metrics.json`

  A metrics document, `aeiou_metrics: 1`, from this tool or from `aeiou dry-run
  --metrics-json`. The histograms are in full, per actor template and in total.
- `*.digest.json`

  A digest document, `aeiou_digest: 1`: `templates`, `lanes` with their `phases`, `sites`
  with the `selection`, `order`, and `size` verdicts, and `open`, the ranked questions.
  Times are nanoseconds from the first exported call; histograms are the metrics' quarter-
  octave buckets.
- `*.tolerances.json`

  A tolerance file, `aeiou_tolerances: 1`, kept beside a capture kit's trace document.
- The exported trace file

  JSON Lines: a header `{"aeiou_trace": 1, "source", "root", "lanes", "lines", "opens",
  "creates", "notes"}`, then lines `{"lane", "t", "dur", "op", "fd", "path", ...}` with the
  schema's field names and enum words, `t` the call's start in nanoseconds from the
  earliest start of an exported call, `fd` an open id (the ordinal of the `open` line),
  `ret` the traced result (a count, or an errno name that becomes the op's `expect`), and
  `submit` lines holding the members of one `io_submit`. Defined in `runner/REFERENCE.md` §13.
- `workloads/<category>/…/<mechanism>/kit/`, `…/cases/<case>/captures/<capture>/`

  The capture kit of each real application traced so far (the traced script, the corpus
  writer, the fitting script), and per capture the trace's metrics document, its
  tolerance file, and the parameters fitted to it.

## EXIT STATUS

- **0**

  Success. For `compare` with `--judge`, at least one row was judged and none is outside;
  with `--max-distance`, no row exceeds it.
- **1**

  A row outside its bound, a document the tool cannot read, a `--self` document for
  another abstract or shape, a trace with no counted call (a `--root` that none of its
  paths is under, or an input that is not an strace; `export` then writes no file).
- **2**

  A usage error, in the frame every tool of the suite shares; a `--max-distance` or
  `--chain-gap-us` that is negative or not a finite number is one.

## EXAMPLES

Measure a trace of the application, measure the abstract at its fitted parameters, and judge
the abstract against three other seeds of itself:

```
strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt python train.py /mnt/data/train
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

Digest a trace for a fit, projecting its open questions over the four passes the benchmark
will run, and send the document rather than the trace:

```
aeiou-trace digest trace.txt --root /mnt/data --epochs 4 --candidates 168 -o trace.digest.json
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-abstract**(7),
**strace**(1).

The reference: `builder/REFERENCE.md` §7 (the tool, the tolerances, the capture kits),
`runner/REFERENCE.md` §10 (the definitions of the metrics) and §13 (the `trace` node and
its file).
