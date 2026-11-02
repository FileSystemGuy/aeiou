# aeiou-trace(1)

## NAME

aeiou-trace - locality metrics of a real application from an strace of it, the trace file of the runner's `trace` node, the structural digest a fit is made from, and the comparison of the application's with its abstract's

## SYNOPSIS

```
aeiou-trace metrics [OPTIONS] [--root DIR]... [-o FILE] TRACE
aeiou-trace export [OPTIONS] --root DIR -o FILE TRACE
aeiou-trace compare [OPTIONS] A B
aeiou-trace digest [OPTIONS] --root DIR [-o FILE] [--answers FILE] [--no-questions] TRACE
aeiou-trace --help | --version
```

## DESCRIPTION

An abstract's fingerprint proves which operation stream ran; the locality metrics are for
proving it is the *right* stream, by comparison with the same numbers taken from a trace of
the real application the abstract models. `aeiou dry-run --metrics-json` (**aeiou**(1))
gives the abstract's numbers; **aeiou-trace** gives the same numbers, by the same
definitions and in the same document (`aeiou_metrics: 2`, with `"source": "strace"`),
from an `strace` of the application. It also exports an strace as the file the runner's
`trace` node executes literally, digests one into the structural summary an abstract is
fitted from, and compares the application's document with the abstract's: two digests'
structure by equality and the numbers of any two documents by distance.

**aeiou-trace** is a binary of the runner's crate, built and shipped beside **aeiou**(1).
`metrics` and `export` are its own (Rust, since 2026-10-29), and so is `digest`: its
document and report, its questions, the application records, and the answers file
(since 2026-10-31); and `compare` (since 2026-11-01): no Python is needed for any of it
(`DESIGN_REVIEW.md` §3.89). The trace is read in a stream, and the state kept is per open, per
thread, and per path under the root, never per call, so a trace of several gigabytes is read
in the time it takes to read it.

The trace is taken with

```
strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt <command>
```

Start the command under `strace`, never attach to it running (`strace -p`): a descriptor
the command opened before `strace` attached has no open in the trace (see **What counts**).
Add `%network` to `-e trace=` when the application passes descriptors over a socket
(`SCM_RIGHTS`), so the `recvmsg` that delivers one is seen.

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
`truncate`, `ftruncate`, `fallocate`, `mkdir`, `rename`, `fadvise64`, and `close` by name. A
failed call counts as its op. Nothing that frees or moves a file's space is left out: `export`
writes a `truncate` with its length and a `fallocate` with its mode by flag name, and a mode
with a bit no flag names is left out and noted, never written as another. A descriptor used
without the open that made it, held from before `strace` began (attached with `strace -p` to
a running process, inherited from the shell, received over a socket), is a trace that is
incomplete: its calls are counted from position 0, `export` writes an open of it at its first
use (`"synthesized": true`, its flags `RDONLY`, `WRONLY`, or `RDWR` as the calls on it read
and write) so a `trace` node replays them, the header lists each such open
(`missing_opens`), and `export`, `digest`, and `metrics` print a warning on standard error
naming them, and go on. A descriptor with no path to open (a trace without `-yy`) is left
out, noted by name. The position of a sequential `read` or `write` is tracked per open file
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
`w.trace("app.jsonl", sha256)`; the runner refuses the file when it differs. The trace is read
twice (the first pass learns which opens several lanes shared, the earliest start, and what each
AIO request returned; the second writes), so standard input is first copied to a temporary
file; the trace file is written beside *FILE* and renamed into place, so a failure leaves none.

### compare

Compares two observation documents, *A* the application's and *B* the abstract's. Each is a
digest (`aeiou_digest: 2`) or a metrics document (`aeiou_metrics: 2`, or 1, a capture's from
before 2026-10-22), which has numbers and no structure; the abstract's digest is the digest of
`aeiou dry-run --emit-trace` (**aeiou**(1)), or of an strace of `aeiou run`, which checks that
a backend issues the abstract's calls.

**Structure**, when both are digests, by equality. *B*'s own I/O is set aside first and
listed: a name beginning `.aeiou` (no workload name may, V13) and a template of the root that
is only stats; the runner's checks before the gate (a `stat` of a dataset's file) are not, and
show on an strace of `aeiou run`. Templates pair by identical text, then by `--pair`, then by
shape (the same fields with the same kinds and widths, and the same kinds of site) where one
template of each side has it. A template of *A* that nothing pairs is a difference; one of *B*
is listed. Per pair, each aspect is `same`, `DIFFERENT`, `not compared`, or `shown`: the fields
(kind and width); per site, by phase, its kind and each verdict, a side's undecided verdict
not compared and its owner's asserted answer compared in its place (labelled `assertion`);
the set of op kinds; the set of motifs (a motif that depends on file size is structure, so the
run takes the trace's sizes); where either side's template has a `space`, the kinds of what
its calls did to the space its files hold (truncating opens, of a path written before,
`truncate`, `ftruncate`, to length 0, `fallocate` by mode), never their counts, which grow with
the scale; the peak of lanes at once on the template (each phase an
interval of its lane); and the lanes on it and in all, shown and never judged. A difference
of structure exits 1, and under `--judge` it decides the verdict whatever the rows.

**Numbers**, one row per metric
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

Writes the digest document (`aeiou_digest: 2`) of a trace, the structural summary a fit is
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
and the think time between units. **Space**: per template, only where something happened,
what its calls did to the space its files hold, a load of the filesystem's own (taking a
file's extents apart and reclaiming them, deferred): its opens with `TRUNC`, of them those on a
path an earlier open had written and those on a path the trace had not seen before, its
`truncate` and `ftruncate` calls and those to length 0, its `fallocate` calls by mode, and the
bytes of the ranges `PUNCH_HOLE` freed and `COLLAPSE_RANGE` removed, and the bytes its
truncates discarded where the trace shows the size (set by a truncate and moved by every write
and allocation after it, carried through a `rename`), with the truncates of a size it does not
show counted apart, never guessed; an open with `TRUNC` reads
nothing where one without may read the whole file. **Sites**: what the digest judges, a template or, where
a template is two phases, the opens of one (`0/write`, `0/read`), of five kinds. A
*set*, a template opened at least twice and read (a sample set), has the three verdicts
below; read-only names with a hex field are a set, noted that a keyed store written
elsewhere fits as well. A *keyed* site is a template whose writes and reads interleave
(a read of a path before a later write, or an open both read and written; a KV cache); a
hex field in its name supports it and never makes it alone. It counts its keys, the reads
after a write and before any, the re-reads and the share of reads the top tenth of the
keys take, and the lag from a key's write to each later read, in time and in touches; its
verdicts are *origin* (`traced` when every key read was written in the trace, else
undecided between an earlier run's warm cache and another writer), *reuse* (`repeated`
when a key is read again, else undecided), and *size*. All of a template's writes before
all its reads is two phases, a *bulk write* and a reader (a vector database's leaves,
written by a build and read by a search); reads before its first write, of paths it never
writes, starting in the trace's first quarter (from its first call to its last), are a
*restore* (a load of an earlier checkpoint, after the application's start-up I/O and its
model) and the writes after it a bulk write; starting later, they are a reader. A *bulk write* site, opened at least twice and only written (a checkpoint, an index
build), finds its *saves*: bursts of opens, each starting within the burst's own length of
the one before, `simultaneous` when every save's lanes open their first files within a
quarter of its length (a distributed checkpoint), `one_writer` when one lane writes them
all (plain DDP's rank 0, whose writer count at scale is asked), `staggered` or `mixed`
otherwise, and their interval; every field of its name, `varies`, `constant` once a second
save keeps it, else undecided until one; its *rewrite* (a checkpoint is never overwritten,
so a path written again is evidence against one), the share of opens with an `fsync`;
*retention*, from its unlinks: `kept` (nothing deleted), `keeps` N (saves deleted only once
a later one was stable; N the most held after a deletion), or `unsafe` (a save deleted
before the next was stable); *label*, `checkpoint` or `output`, which a rewrite or a read
after the first write decides as `output` and nothing in a trace decides as `checkpoint`,
with the evidence that supports it listed; and *size*. A *within* site is one open read at
eight or more offsets in more than one run (an index traversal): the reads, the distinct
offsets and the share that revisit one, the span read, the sequential runs, *skew* (the top
tenth of the offsets' share of the reads against a uniform spread's, undecided under fifty
reads), and two verdicts no trace decides, *memory* (does the file fit at scale) and
*dependency* (does each offset depend on the last read). For a set, its
*selection* verdict says what the sequence of paths decides between a permutation pass
over the set (`consume`, each path once per pass) and a draw with replacement (`choose`): a
repeat before the set is exhausted refutes the pass; two or more passes of equal length
refute the draw; no repeat at all decides nothing unless the corpus (`--candidates`) is
small enough that a draw would have repeated (the birthday bound, at 5 %), and then the
verdict says how many touches would decide it. *Order* within a pass (`sequential`,
`permuted`, `other`) and the bytes per open (`constant`, `per-path`, `drawn`) are judged
the same way, each `undecided` with what settles it. **Facts**: every site lists the facts a
person can give about it at scale (below, under asking), each with its question and, where the
trace bounds it, the least value the trace allows and what it shows. **Metrics**: the document
`metrics` writes of the same input under the same `--instance-root`, `--chain-gap-us`, and
`--metrics-block` (`aeiou_metrics: 2`), so one document carries a trace's structure and its
numbers: of an strace, the strace's own reading, taken beside the export's first pass (every
call under the root counted, a libaio request when it is reaped); of an exported trace file or
an object log, its exported lines (a lane is a context, the whole input one instance). **Open**: every undecided verdict as a
question in the application's terms, its candidate constructs, what evidence settles it,
and the candidates' divergence at `--epochs` passes (the share of the set a draw with
replacement never reads, e^-N, against zero for a pass), ranked largest first. **Asking**:
on a terminal, before it writes, the digest asks the trace's owner what the trace cannot
decide. First the application: when a path matches a record's signature (`applications/`,
see FILES) it asks "this looks like X, is that right?", otherwise which application from
the list, then the version, then the record's options, the few that change the storage
workload, each a list with the default marked and "don't know" present; the record's
assertions then answer what they can, with the record as their provenance, and the blind
spots the choices cause (mmap, io_uring, an object store) are said and noted. A version no
record claims gets the nearest lower record, its assertions offered one by one for the
person to confirm. Then, facts first and the open questions in rank order: for every site with an open
question, the facts of its kind (a set: how many files of its kind exist at the scale the
application runs and how many passes the benchmark will make; a keyed site: the keys it
holds at scale and the requests the benchmark serves; a bulk write: every field's count of
values at scale (ranks or steps) unless two saves showed it constant, how many times the
benchmark writes this kind of file, how many saves the application keeps, and, for one
writing lane, how many processes write each save at scale; a within site: the file's size at scale and the queries the benchmark runs;
Enter for don't know; a fact decides what it can, so a corpus small enough refutes a draw
and no behavioral question follows), then each open item
on one screen, the evidence, the question in the application's terms, numbered choices with
"don't know" always among them, and what the choice changes at scale; an answer the trace
contradicts is shown the evidence and asked again, never recorded. Every answer is recorded
in the document under `asserted` with who gave it and when, beside the verdict it answers,
which stays the trace's (`undecided`), never as a measurement; an optional "how do you know?"
note goes with it; an answered item leaves the open list, a "don't know" stays on it, marked.
The answers are also written as an answers file (`aeiou_answers: 1`), given with `--answers`
or named after the document, which a later run replays without a terminal, so trace file plus
answers file reproduces the digest. Without a terminal, or with `--no-questions`, nothing is
asked and every undecided verdict stays open. An answer a replayed file carries against a
verdict the trace decides is rejected with the evidence and listed under `rejected`; one
agreeing with it is redundant and noted. The trace is read as a file, never from stdin: the
comparison after the fit and a re-run with answers read it again.

## OBJECT LOGS

An strace never shows an application's object requests (TLS, or another process), but the
store's own log does, one line per request. `metrics`, `export`, and `digest` read two such
logs in place of an strace, told apart by the first line, and turn each request into the
calls the runner's object engine issues for it (`runner/REFERENCE.md` §16): a GET is an
open, one read (at the Range header's start when the log has it, else from 0), and a
close; a PUT an open that creates, one write, and a close; a HEAD a stat; a DELETE an
unlink; a listing an open of the prefix, a readdir, and a close; a multipart upload an open
at its start, a write per part, and a close at its completion. A lane is a client host,
since the log knows no thread; `--root` is `BUCKET[/PREFIX]`, one, and requests outside it
are counted and left out, as are the requests with no call of the engine behind them
(bucket operations, copies), by name under `notes`. The exported trace file of an object
log is an ordinary trace file: the `trace` node runs it against the declared endpoint.

- **MinIO audit log** (JSON lines with an `api` object: `name`, `bucket`, `object`,
  `statusCode`, `rx`, `tx`, `timeToResponseInNS`; the `Range` request header; the
  `prefix` and `uploadId` query parameters). An entry's `time` is taken as when the
  response completed, the request starting `timeToResponse` earlier (noted as
  `time_is_response_end`). Enabled as MinIO's audit webhook or Kafka target, or its audit
  event queue, in MinIO's own documentation.
- **Amazon S3 server access log** (space-delimited records: bucket owner, bucket, `[time]`
  to the second, remote IP, requester, request id, `REST.<METHOD>.<RESOURCE>`, key,
  "request URI", status, error, bytes sent, object size, total time in milliseconds, ...).
  It carries no Range, so a GET that sent fewer bytes than the object holds is read from
  offset 0 and counted under `ranges_unknown`; its times are to the second. AWS delivers
  these logs best-effort and late; CloudTrail data events are the complete source.

Not read yet: Ceph RGW's ops log, a lab proxy's log (`PROJECT_BRIEF.md` item 49).

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

  The application's document: a digest (`aeiou_digest: 2`), or a metrics document
  (`aeiou_metrics: 2`, or 1 from before 2026-10-22: numbers only).
- *B*

  The document *A* is compared with, the abstract's: the digest of its `aeiou dry-run
  --emit-trace`, or its `aeiou dry-run --metrics-json`.
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

  A metrics document (or digest) of *B*'s abstract at the same parameters and `--gpus` and
  another seed. Repeatable; a row's tolerance is raised by its largest distance between *B*
  and these. The abstract's hash, `--gpus`, block, and source are checked; the parameters
  cannot be. A digest of an emitted file names its dry run, so its numbers are checked as a
  dry run's. Implies `--judge`.
- **--only** *PREFIX*

  Only the rows whose metric name starts with *PREFIX*. Repeatable.
- **--pair** *A*`=`*B*

  Pair *A*'s template *A* with *B*'s template *B*, written as the digests write them
  (`train/{a:05}.npz`), when neither their text nor their shape pairs them. Repeatable; two
  digests only.
- **--json**

  Print the comparison as one JSON document (`aeiou_compare: 1`): `a` and `b` (path and
  description), `structure` (`pairs` with their `aspects`, `unpaired_a`, `unpaired_b`,
  `set_aside`, `lanes`, `differences`; null unless both are digests), `rows` (`metric`, `a`,
  `b`, `distance`, and under `--judge` `allowed`, `verdict`, `note`), `selfs`, `notes`, and
  `verdict` (`same` or `different`; under `--judge` `accepted`, `not accepted`, or `nothing
  judged`). The exit status is the text's.

### aeiou-trace digest

- *TRACE*

  The strace output file, an exported trace file, or an object-store request log (see
  OBJECT LOGS). A file, never stdin (`-` is refused): the comparison after the fit and a
  re-run with answers read it again. Its SHA-256 is recorded in the document and in the
  answers file.
- **--root** *DIR*

  Digest only calls on paths under *DIR*, written relative to it. Required for an strace;
  an exported trace file carries its root, and a `--root` that is not it is refused. For an
  object log, `BUCKET[/PREFIX]`.
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
- **--metrics-block** *BYTES*

  The block metrics too in the `metrics` section, as `metrics --metrics-block` computes
  them. At least 1.
- **--instance-root** *PID*

  The `metrics` section's instances, as `metrics --instance-root` makes them. Repeatable.
  An strace's alone: for an exported trace file or an object log, which carry no process
  tree, it is a usage error.
- **--chain-gap-us** *US*

  The `metrics` section's depth, as `metrics --chain-gap-us` reports it. An strace's
  alone: an exported trace file or an object log does not show where a round of
  `io_submit` ends, and it is a usage error there.
- **--text**

  Print the text report instead of the JSON document.
- **-o**, **--out** *FILE*

  Write the JSON document to *FILE* and print the text report.
- **--answers** *FILE*

  The answers file (`aeiou_answers: 1`). When it exists it is replayed and nothing is asked:
  its facts and answers are checked against the trace and recorded, with the file's `by`
  and `at` as their provenance, and a file answered on another trace is applied and noted.
  When it does not exist and there is a terminal, the questions are asked and the answers
  written there; without a terminal that is an error. Without this option an interactive
  run with `-o` writes the answers beside the document, `NAME.answers.json` for
  `NAME.digest.json` (else `FILE.answers.json`), and prints the command line that replays
  them; without `-o` they live in the document only.
- **--no-questions**

  Ask nothing even on a terminal: every undecided verdict stays open.
- **--answered-by** *NAME*

  Record *NAME* as the one who answered. Default: `user@host`.
- **--applications** *DIR*

  The application records, `applications/` (see FILES). Default: the records built into the
  tool, the checkout's `applications/` it was built from.
- **--application** *NAME*[@*VERSION*]

  State the traced application instead of being asked, by the name of its record directory
  (`pytorch-dcp@2.14.1`). Without a terminal, the record's options take their defaults and
  its assertions apply with the record as their provenance; on a terminal, the identification
  is skipped and the options are still asked. A name no record has is a usage error.

### Common

- **-h**, **--help**

  Print the help of the command or subcommand.
- **-V**, **--version**

  Print the version.

## ENVIRONMENT

- **AEIOU_TRACE**

  Read by the Python package's tests, which run the binary for every digest document: the
  path of **aeiou-trace**. Default: the one beside `AEIOU_RUNNER`, else the one on
  the `PATH`, else the checkout's release build.

## FILES

- `*.metrics.json`

  A metrics document, `aeiou_metrics: 2` (or 1, a capture's from before 2026-10-22), from this tool or from `aeiou dry-run
  --metrics-json`. The histograms are in full, per actor template and in total.
- `*.digest.json`

  A digest document, `aeiou_digest: 2`: `trace` and `trace_sha256`, `templates`, `lanes`
  with their `phases`, `sites` with their kind's verdicts, their `facts` (`name`, `ask`, and,
  where the trace bounds it, `least` and `why`), and a `projection` (the facts the site was
  judged at, and which of them were asserted), `open`, the ranked questions, `asserted`, every fact and answer recorded with
  `site`, `template`, `by`, `at`, and its `note`, and `rejected`, every answer the trace
  contradicted, with `why`. Times are nanoseconds from the first exported call; histograms
  are the metrics' quarter-octave buckets. `metrics` is the metrics document of the same
  calls (`aeiou_metrics: 2`; its `notes` count, as `lines_not_counted`, the lines of an
  exported file the runner would refuse, such as an op on an open no line made, which the
  structure notes and the numbers leave out).
- `applications/`

  The application records (`applications/README.md`): one directory per application,
  `application.toml` with its kind, path signatures, and catalog mechanisms, and
  `<version>.toml` per version range read, with the options that change its storage
  workload, the blind spots, and the assertions it makes, each with its source and
  confidence. Built into the binary from the checkout it was built from; `--applications DIR`
  reads another tree, which is how a record is tried before it is committed.
- `*.answers.json`

  An answers file, `aeiou_answers: 1`: `trace`, `trace_sha256`, `by`, `at`, `application`
  (what the owner said the application is, its record, options, and blind spots), and `answers`,
  a list of facts (`{"template", "fact", "value", "note"}`, the fact one of the site's kind's:
  `candidates`, `passes`, `keys`, `requests`, `size`, `keep`, `writers`, `values:<field>`) and
  answers (`{"template", "question", "answer", "note"}`, the question one of `selection`,
  `size`, `order`, `origin`, `reuse`, `skew`, `memory`, `dependency`, `label`, the answer one
  of its candidates or `unknown`), either with `phase` (`write` or `read`) when its template is
  two sites; without one it answers every site of the template that has the fact or question. Written by an interactive
  `digest`, replayed by `--answers`; a hand-off, not a thing to edit.
- `*.tolerances.json`

  A tolerance file, `aeiou_tolerances: 1`, kept beside a capture kit's trace document.
- The exported trace file

  JSON Lines: a header `{"aeiou_trace": 1, "source", "root", "lanes", "lines", "opens",
  "creates", "notes"}`, then lines `{"lane", "t", "dur", "op", "fd", "path", ...}` with the
  schema's field names and enum words, `t` the call's start in nanoseconds from the
  earliest start of an exported call, `fd` an open id (the ordinal of the `open` line),
  `ret` the traced result (a count, or an errno name that becomes the op's `expect`), and
  `submit` lines holding the members of one `io_submit`. Defined in `runner/REFERENCE.md` §13.
  The file `aeiou dry-run --emit-trace` writes is one too, its header with `source: dry-run`
  and the dry run's `abstract`, `sha256`, `seed`, `gpus`, `fingerprint`, and `order`, which
  `digest` keeps as the source of its document and copies into its `metrics`.
- `workloads/<category>/…/<mechanism>/kit/`, `…/cases/<case>/captures/<capture>/`

  The capture kit of each real application traced so far (the traced script, the corpus
  writer, the fitting script), and per capture the trace's metrics document, its
  tolerance file, and the parameters fitted to it.

## EXIT STATUS

- **0**

  Success. For `compare`, the structure of two digests is the same; with `--judge`, at
  least one row was judged and none is outside; with `--max-distance`, no row exceeds it.
- **1**

  A difference of structure, a row outside its bound, a document the tool cannot read, a `--self` document for
  another abstract or shape, a trace with no counted call (a `--root` that none of its
  paths is under, or an input that is not an strace; `export` then writes no file).
- **2**

  A usage error, in the frame every tool of the suite shares; a `--max-distance` or
  `--chain-gap-us` that is negative or not a finite number is one, and so is a digest's
  `--instance-root` or `--chain-gap-us` for an input that is not an strace.

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

Hold an abstract fitted to a digest against it: emit the abstract's operations at the trace's
scale, digest them like a trace, and compare structure and numbers, the abstract's dry run at
three other seeds its own spread:

```
aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --seed 1 --emit-trace abstract.jsonl
aeiou-trace digest abstract.jsonl --no-questions -o abstract.digest.json
for s in 2 3 4; do aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --seed $s --metrics-json self$s.json; done
aeiou-trace compare trace.digest.json abstract.digest.json --judge --self self2.json --self self3.json --self self4.json
```

Digest a trace for a fit, projecting its open questions over the four passes the benchmark
will run, and send the document rather than the trace:

```
aeiou-trace digest trace.txt --root /mnt/data --epochs 4 --candidates 168 -o trace.digest.json
aeiou-trace digest trace.txt --root /mnt/data --answers trace.answers.json -o trace.digest.json
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-abstract**(7),
**strace**(1).

The reference: `builder/REFERENCE.md` §7 (the tool, the tolerances, the capture kits),
`runner/REFERENCE.md` §10 (the definitions of the metrics) and §13 (the `trace` node and
its file).
