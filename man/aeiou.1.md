# aeiou(1)

## NAME

aeiou - Author Execute I/O workload runner: validate, dry-run, generate datasets for, and execute an abstract

## SYNOPSIS

```
aeiou [--config FILE] check FILES...
aeiou [--config FILE] dry-run [OPTIONS] --gpus GPUS ABSTRACT
aeiou [--config FILE] datagen [OPTIONS] --root DIR ABSTRACT
aeiou [--config FILE] run [OPTIONS] --gpus GPUS --root DIR ABSTRACT
aeiou --help | --version
```

## DESCRIPTION

**aeiou** is the Rust runner of the aeiou suite. It executes an *abstract*, a workload
written as a program over POSIX-shaped I/O operations and compiled by **aeiou-build**(1) to
the JSON document `ABSTRACT` (suffix `.ast.json`, described in **aeiou-abstract**(7)). The
abstract is the only thing the runner executes: every instance of every actor it declares
is walked as a deterministic program whose randomness is a pure function of the run seed,
the actor id, and the position of each draw, so two runs with the same abstract, seed,
instance count, and parameters issue the same multiset of operations on every system under
test.

The four subcommands are the stages of a benchmark:

- **check** validates abstracts and prints their identities.
- **dry-run** computes the operation stream without doing any I/O and prints its
  **fingerprint**, the order-independent hash of the multiset of positioned operations,
  with counts, bytes, and, on request, the locality metrics of the stream.
- **datagen** writes the datasets an abstract declares, with content reproducible from the
  dataset seed, and a manifest per dataset root.
- **run** executes the abstract against `--root` on one host, or on several hosts through a
  coordinator, checks every result structurally, and reports latencies, throughput, host
  counters, and the fingerprint, which must equal the dry run's.

A result is identified by three things the run prints together: the abstract's SHA-256,
the parameters in effect, and the dataset ids. The run seed selects one of many equivalent
streams and is not part of the identity.

Every subcommand prints, after the `abstract` line, a block listing every option with its
value in effect and the layer it came from: the command line, the environment, the config
file, or the default (**aeiou-config**(5)). Options that the workload's identity, a dataset
id, or a safety check depends on are *fixed*: they are the command line's alone, and a lower
layer naming one is refused. The others are *layered* and are marked so below.

## COMMANDS

### check

Loads each `.ast.json` file, validates it against the schema and the semantic rules
of **aeiou-abstract**(7), and prints its canonical SHA-256 and its counts of
operations by kind, in the format of the reference checker `schema/check.py`. Prints
`ok` or `FAIL` per file.

### dry-run

Walks every actor instance without I/O, in parallel over instances, and prints: the
operation counts by kind and by phase, bytes read and written, emulated compute time,
barriers, the declared API and cache mode when the abstract declares one, and the workload fingerprint.
With `--ranks` it adds the bytes per host, against this host's memory. With `--gpu` it
prints one instance's operation stream, one line per operation, in the round-robin order
of that instance's concurrent sub-actors. With `--metrics` it computes the locality
metrics of the stream (see METRICS below) for comparison with a trace of the real
application through **aeiou-trace**(1).

### datagen

Writes every `files` and `regions` dataset the abstract declares under `--root`: names from
the pattern, sizes from the dataset seed, content from the positional payload generator
(1 MiB blocks of a `dgen-data` stream keyed by dataset seed, unit, and block, with the
dedupe and compression ratios given), in parallel by id, with `O_DIRECT` (datagen never
reads what it writes, and the client's page cache is what a benchmark must not have warm; a
root that refuses `O_DIRECT` falls back to the page cache, said once). The manifest
`.aeiou-dataset.json` is written last, atomically, at each dataset root, and its id is
printed. A non-empty root is refused: datasets are read-only. A dataset that declares a
container format class (Parquet, HDF5, TFRecord, tar) is refused here and written by
**aeiou-datagen**(1).

On several hosts (`--ranks`, `--rank`, `--coordinator`, as for `run`; **aeiou-launch**(1)
starts them) rank *r* writes its contiguous slice of every `files` dataset's ids, and each
`regions` dataset is written whole by one rank (the *i*-th by rank *i* mod *R*: several
hosts writing one file serialize on the server). Every rank checks the roots are empty
before the start gate, writes after it, and sends its counts; rank 0 writes the manifests
once every rank has reported, so a manifest means the whole corpus is there, and records
every rank's host and counts in the provenance. The coordinator refuses a host whose
abstract, parameters, `--gpus`, `--dedupe`, `--compress`, or `--dataset` list differ.

### run

Executes the abstract against `--root`. Before the start gate it compares every dataset
with its manifest and every input namespace with the manifest of the run that wrote it,
requires every output namespace root to be empty, estimates the open files, threads, and
mappings the host will need and refuses when a limit cannot hold them, and, on several
hosts, compares its identity with rank 0's. Then every actor instance runs: one operating
system thread per actor and sub-actor under the blocking backends, or many actors
multiplexed over one ring or AIO context per event-loop thread under `io_uring` and
`libaio`. Every result is checked structurally (a read returns the computed count, a
failing call's errno is one the statement expects); anything else aborts the run naming the
actor, position, and operation. The runner never looks at the bytes.

At the end it prints latency histograms per operation kind, per-phase totals, per-step
stall and busy fractions, host counters (task and io-wq worker peaks, CPU, RSS, the mount's
NFS RPC deltas), the fingerprint, and the verdict, and leaves `.aeiou-namespace.json` at
every namespace root it wrote. `--report-json` writes the same as one JSON document.

## OPTIONS

Options marked *Layered* may also come from the environment variable `AEIOU_<FLAG>` or
from the subcommand's table in the config file (**aeiou-config**(5)); every other option
is the command line's alone. Every boolean reads `--[no-]x`: `--x` turns it on, `--no-x`
turns it off, and the last one on the line wins.

### Global options

- **--config** *FILE*

  A TOML config file, else the one `$AEIOU_CONFIG` names, else none. It is never searched
  for. One table per subcommand, keys spelled as the long flags. Below the command line and
  the environment in precedence.
- **-h**, **--help**

  Print the help of the command or subcommand.
- **-V**, **--version**

  Print the version.

### aeiou check

- *FILES...*

  The `.ast.json` files to validate.

### aeiou dry-run

- *ABSTRACT*

  The abstract (`.ast.json`).

**Workload**

- **--param** *NAME=VALUE*

  Override a parameter. The value is JSON; a bare word is a string. Repeatable; applied
  after every parameter file. A value keeps the kind of the default it replaces (scalar,
  array, or distribution).
- **--params-file** *FILE*

  A parameter file (`.params.json`, **aeiou-params**(1)), applied over the defaults and
  under `--param`. Repeatable, applied in order.
- **--gpus** *GPUS*

  Required. The number of instances of every actor template whose count is the reserved
  parameter `gpus`; instances have global ids `0..GPUS`.
- **--seed** *SEED*

  The run seed, default 0. The dataset seed is separate and lives in the abstract.

**Output**

- **--ranks** *RANKS*

  The hosts the run would be spread over, for the bytes-per-host estimate. *Layered.*
- **--threads** *THREADS*

  Worker threads for the walk; default all cores. *Layered.*
- **--gpu** *G*

  Print the operation stream of actor instance *G*, on one thread, in order.
- **--steps** *A..B*

  With `--gpu`: only the operations whose outermost loop index is in `[A, B)`.
- **--limit** *N*

  With `--gpu`: stop printing after *N* operations.

**Metrics**

- **--[no-]metrics**

  Compute the locality metrics of the operation stream: reuse distance, sequential runs,
  popularity, request sizes, fan-out and depth, read/write mix. Nothing here changes the
  stream or the fingerprint.
- **--metrics-block** *BYTES*

  With `--metrics`: the block size of the reuse-distance and popularity units. Default 4096.
- **--metrics-sample** *N*

  With `--metrics`: keep one block and one object in *N*, chosen by hash, and scale the
  counts and distances. Default 1, exact.
- **--metrics-json** *FILE*

  Write the metrics, with their histograms in full, to *FILE* as JSON (format
  `aeiou_metrics: 1`). Implies `--metrics`.

### aeiou datagen

- *ABSTRACT*

  The abstract (`.ast.json`).

**Workload**

- **--param** *NAME=VALUE*

  As for `dry-run`.
- **--params-file** *FILE*

  As for `dry-run`.
- **--gpus** *GPUS*

  Instance count, for dataset definitions that reference `gpus`. Default 1.

**Writer**

- **--root** *DIR*

  Required, from some layer. The directory the abstract's paths are relative to. From the
  command line, else `$AEIOU_ROOT`, else `root` in the `[datagen]` table of the config
  file. *Layered.*
- **--endpoint** *NAME=DIR|URI*

  Puts the root directory of the abstract's dataset NAME at DIR instead of under `--root`,
  or, for a dataset declared `protocol: s3`, at `s3://BUCKET[/PREFIX]` (repeatable); see
  ENDPOINTS. From the command line, else `$AEIOU_ENDPOINT`, else `endpoint` in the
  `[datagen]` table of the config file. *Layered.*
- **--threads** *THREADS*

  Writer threads; default all cores. *Layered.*
- **--object-threads** *N*

  Worker threads of the object engine's runtime, which drive its connections (the writer
  threads make the requests); see OBJECT STORES. Only with a dataset placed in an object
  store. Default 2. *Layered.*
- **--dedupe** *DEDUPE*

  Dedupe ratio: every *DEDUPE* consecutive files (or 1 MiB blocks of a `regions` file) share
  content, whatever the count, so a prefix of the ids has the ratio too. Default 1. Part of
  the payload the manifest records.
- **--compress** *COMPRESS*

  Compression ratio: the last (C-1)/C of every 1 MiB block is zeros. Default 1. Part of the
  payload the manifest records.
- **--dataset** *NAME*

  Write only these datasets; repeatable. Default all.

**Several hosts**

- **--rank** *R*

  This host's index among `--ranks` hosts. Default 0. *Layered.*
- **--ranks** *N*

  Hosts the datagen is spread over: each writes its slice of every `files` dataset's ids, and
  each `regions` file is written by one of them. Default 1. *Layered.*
- **--coordinator** *HOST:PORT*

  With `--ranks` above 1, required: the coordinator's address. Rank 0 listens on it
  (in-process); every rank connects to it. *Layered.*

### aeiou run

- *ABSTRACT*

  The abstract (`.ast.json`).

**Workload**

- **--param** *NAME=VALUE*

  As for `dry-run`.
- **--params-file** *FILE*

  As for `dry-run`.
- **--gpus** *GPUS*

  Required. As for `dry-run`.
- **--seed** *SEED*

  As for `dry-run`.

**Backend**

- **--posix** *API*

  The API the operations on `posix` names are issued through: `sync`, `io_uring`,
  `posix-aio`, `libaio`, or `mmap`; see BACKENDS. Default: the API the abstract declares,
  `sync` when it declares none. Any other is a different workload on the storage, and the
  run says so and records both. Refused when no dataset or namespace is `posix`; the value
  is required.
- **--cache** *MODE*

  The `posix` names' cache mode: `per-open` (each open's own flags decide whether it
  bypasses the page cache) or `direct` (`O_DIRECT` on every regular-file open; not with
  `mmap`). Default: the abstract's, `per-open` when it declares none. Any other is a
  different workload on the storage, as for `--posix`. Refused when no dataset or
  namespace is `posix`.
- **--s3** *API*

  How the application's S3 client uses the store for the `s3` names: `blocking` (a thread
  per actor, waiting on each request) or `async` (event loops, many actors' requests in
  flight per thread); see BACKENDS and OBJECT STORES. Default: the abstract's, `blocking`
  when it declares none. Any other is a different workload, as for `--posix`. With `posix`
  names too, both APIs are event-driven (`io_uring` or `libaio` with `async`) or neither.
  Refused when no dataset or namespace is `s3`.
- **--root** *DIR*

  Required, from some layer. The directory the abstract's paths are relative to; datasets
  and namespaces live under it. From the command line, else `$AEIOU_ROOT`, else `root` in
  the `[run]` table of the config file. *Layered.*
- **--endpoint** *NAME=DIR|URI*

  Puts the root directory of the abstract's dataset or namespace NAME at DIR instead of
  under `--root`, or, for a name declared `protocol: s3`, at `s3://BUCKET[/PREFIX]`
  (repeatable); see ENDPOINTS. From the command line, else
  `$AEIOU_ENDPOINT`, else `endpoint` in the `[run]` table of the config file. *Layered.*
- **--threads** *THREADS*

  Event-loop threads for the `io_uring` and `libaio` backends; default one per core, at
  most one per actor instance. The other backends run one thread per actor and refuse
  the flag. *Layered.*
- **--buffer-mib** *MIB*

  The per-thread read and write buffer rings, in MiB. Default 8. *Layered.*
- **--write-compress** *C*

  Compression ratio of the bytes written to namespaces. Default 1. *Layered.*
- **--time-scale** *X*

  Multiply every `compute` sleep by *X*; 0 runs the I/O back to back. Default 1.
  *Layered.*

**io_uring**

- **--iowq-max-workers** *N*

  Cap each loop's bounded io-wq workers (`IORING_REGISTER_IOWQ_MAX_WORKERS`). The report
  shows the kernel's default either way. *Layered.*
- **--sqpoll** *IDLE_MS*

  A kernel submission thread per loop (`IORING_SETUP_SQPOLL`) that sleeps after this many
  idle milliseconds. *Layered.*
- **--[no-]sqpoll-shared**

  One submission thread, and one io-wq, shared by every loop (`IORING_SETUP_ATTACH_WQ`)
  instead of one per loop. Needs `--sqpoll`. *Layered.*
- **--[no-]defer-taskrun**

  `IORING_SETUP_SINGLE_ISSUER` with `IORING_SETUP_DEFER_TASKRUN`. Not with `--sqpoll`.
  *Layered.*
- **--[no-]coop-taskrun**

  `IORING_SETUP_COOP_TASKRUN`. *Layered.*

**libaio**

- **--aio-depth** *N*

  Requests each loop's AIO context holds (`io_setup`'s `nr_events`; the host's total is
  bounded by `fs.aio-max-nr`). Default 256. *Layered.*

**mmap**

- **--mmap-mode** *MODE*

  The prefetch before a read's range is consumed: `fault` (none; the touch faults the
  pages in), `populate` (`MADV_POPULATE_READ` over the range), or `willneed`
  (`MADV_WILLNEED`). *Layered.*
- **--mmap-consume** *HOW*

  How a read's range is consumed: `touch` (one byte of every page is read, so each page is
  resident and mapped; nothing under `populate`, which has done that) or `copy` (the range
  is copied into the actor's buffer). Default `touch`. *Layered.*

**Object engine**

- **--object-threads** *N*

  Worker threads of the object engine's runtime, which drive its connections; each actor
  makes its requests on its own thread, one at a time. Only with a name placed in an
  object store; see OBJECT STORES. Default 2. *Layered.*
- **--object-part-mib** *MIB*

  Part size of the run's uploads to an object store, MiB (5 to 5120): an object written
  through one handle goes out in parts of this size as it is written, and as one `PUT` at
  its close when it fits in one. Only with a name placed in an object store; see OBJECT
  STORES. Default 8. *Layered.*

**Several hosts**

- **--rank** *R*

  This host's index among `--ranks` hosts. Default 0. *Layered*; the environment is the
  intended source (`AEIOU_RANK` set by a launcher from its own rank variable, so one command
  line runs on every host).
- **--ranks** *N*

  The hosts the run is spread over; each runs the GPU id range of its `--rank`. Default 1.
  *Layered.*
- **--coordinator** *HOST:PORT*

  With `--ranks` above 1: the coordinator's address. Rank 0 listens on it, in-process;
  every rank connects to it. *Layered.*
- **--rank-rotate** *K*

  Run the GPU range of rank `(rank + K) mod ranks`, so each host reads what another wrote.
  Default 0. *Layered.*

**Checks**

- **--expect-fingerprint** *HEX*

  Fail unless the run's fingerprint is this (hex, from `aeiou dry-run`). On several hosts
  it is applied to the merged fingerprint.
- **--expect-dataset-id** *SHA256*

  Fail unless every dataset id is among these. Repeatable.
- **--max-gap** *SECS*

  Fail if an input namespace was finished more than this many seconds ago. *Layered.*
- **--[no-]require-cold**

  Fail if this host would read input objects it wrote itself, or if dataset pages are in
  its page cache at the start (`mincore` over 256 sampled files per dataset). *Layered.*
- **--[no-]drop-caches**

  On every host, before the start gate: `sync`, then drop the page cache, dentries, and
  inodes (3 into `/proc/sys/vm/drop_caches`), then sample the datasets' residency. Needs
  root; the run refuses when it fails. *Layered.*
- **--[no-]clean-namespaces**

  Empty the output namespace roots before starting, and remove the files a `trace` node
  creates, instead of refusing. On several hosts only rank 0 does it.
- **--[no-]ignore-limits**

  Start even when the estimated open files, threads, or mappings exceed this host's limits
  (`RLIMIT_NOFILE`, `RLIMIT_NPROC`, `kernel.threads-max`, `vm.max_map_count`); the problem
  is printed as `limits: IGNORED:`.

**Report**

- **--report-json** *FILE*

  Write the run's report to *FILE* as JSON (format `aeiou_report: 1`): the configuration,
  the results the text report prints with the latency histograms in full, and the verdict.
  Written when the run fails too, with the error. Rank 0 of several hosts writes the merged
  report, every other rank its own. *Layered.*
- **--[no-]report-takes**

  With `--report-json`: every take of every instance (stall and compute), not only the
  sums. *Layered.*

## BACKENDS

The abstract is identical under every backend; a backend maps operations to an API and
never changes the stream or the fingerprint. Each protocol has its own API: `--posix` for
the `posix` names, with the cache mode `--cache`, and `--s3` for the `s3` names. A run
compares only with runs under the same choices. An option for a protocol the abstract has
no names on is refused, since it would change nothing.

| `--posix` | API |
|---|---|
| `sync` | POSIX calls on one thread per actor. The fidelity reference. |
| `io_uring` | An event loop per thread multiplexing its actors over one ring, one operation in flight per actor. |
| `posix-aio` | glibc `aio_read`/`aio_write` on one thread per actor. |
| `libaio` | The kernel AIO calls (`io_submit`/`io_getevents`) as a second engine of the event loop; the kernel runs requests through the page cache synchronously, so it is asynchronous only under `--cache direct`. |
| `mmap` | Reads are consumed out of a mapping of the file; see `--mmap-mode` and `--mmap-consume`. |

| `--cache` | Opens |
|---|---|
| `per-open` | Each open's own flags decide whether it bypasses the page cache. |
| `direct` | `O_DIRECT` on every regular-file open; an unaligned read is rounded out to 4 KiB and the requested part counted, an unaligned write is refused. Not with `mmap`. |

| `--s3` | Client |
|---|---|
| `blocking` | A thread per actor, waiting on each request (as boto3 does). |
| `async` | Event loops, many actors' requests in flight per thread (as aiobotocore or the AWS CRT do). |

The actors run on event loops when either API is event-driven (`io_uring`, `libaio`,
`async`), on a thread each otherwise; a run with both protocols whose APIs disagree is
refused. A run of `s3` names alone under `async` has loops that wait on the engine's
answers, with no ring. Under either `--s3` value the requests themselves are made by the
object engine on its runtime threads, never by the POSIX API (OBJECT STORES).

## ENDPOINTS

Every dataset and namespace lives under `--root` unless `--endpoint NAME=DIR` places it.
An endpoint places a name's root directory, the constant directory prefix of its pattern:
under `--endpoint sysp=/fast` the dataset whose pattern is `kv/sys/{id:04}/blk_{k:04}`
has its file `kv/sys/0001/blk_0002` at `/fast/0001/blk_0002`. A path falls under the
longest root that has an endpoint and is a prefix of it, so a dataset whose root lies
inside a placed namespace's moves with the namespace unless it has its own endpoint; every
other path stays under `--root`. Namespaces sharing a root share its place, and two
endpoints for one root must agree. The run prints each endpoint, and warns when one lies on
another file system than `--root`: the report's mount counters cover `--root`'s mount
only. Placement never changes the operation stream, the fingerprint, or a dataset id.

A dataset or namespace declares its protocol (`posix`, the default, or `s3`;
**aeiou-abstract**(7)), and its endpoint fits it: a directory for `posix`,
`s3://BUCKET[/PREFIX]` for `s3`. An `s3` name has no default place, so every one
needs an endpoint, and a `posix` name may not fall under a root placed in an object store
(give it its own directory). The paths below an object endpoint are keys below its prefix:
under `--endpoint train=s3://data/imagenet` the file `train/00001/img_000001300.jpg` is the
key `imagenet/00001/img_000001300.jpg` of the bucket `data`.

## OBJECT STORES

A name declared `protocol: s3` is read and written through the object engine, built on Apache
`object_store` with the cargo feature `object` (a build without it refuses an object
endpoint). The store is the one the environment names: `AWS_ENDPOINT_URL`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and `AWS_ALLOW_HTTP=true` for a
plain-HTTP endpoint; requests are path-style unless `AWS_VIRTUAL_HOSTED_STYLE_REQUEST` is
set. Operations map as follows; the stream, the fingerprint, and the dataset ids are the
same as under a directory.

| Operation | Request |
|---|---|
| `open` for reading | none: the handle is the key |
| `read` at an offset | one ranged `GET`, streamed into the actor's buffer; a short read at the object's end, 0 bytes past it |
| `open` for writing, `write`, `close` | an upload: the writes, each at the end of what the handle has written, fill a part of `--object-part-mib`; a full part goes out as a part of a multipart upload before the write returns; the `close` sends one `PUT` when no part has gone out, else the last part and the completion. The object exists from its `close` |
| `stat`, `fstat` | `HEAD`; for a key that is no object, a `LIST` below it, which succeeds (size 0) when there is a key there; `fstat` of a handle being written is its written size, sent nothing |
| `readdir` | `LIST` with the delimiter `/`: the objects and common prefixes, not counting names that begin with `.aeiou` |
| `unlink` | `DELETE` (a missing key is no error, as S3 has it) |
| `rename` | a copy on the server and a `DELETE` of the source; not atomic |
| `lseek`, `fadvise`, `fsync`, `fdatasync`, `ioctl`, `mkdir`, `rmdir` | none (an `ioctl` answers as for a regular file: `TCGETS` is `ENOTTY`; a prefix is no object) |
| `O_APPEND`, `O_RDWR`, `O_EXCL`, `ftruncate`, `fallocate` | refused by rule V18 at `aeiou check` (**aeiou-abstract**(7)) |

The requests go over ordinary sockets (`epoll`) on the engine's `--object-threads` tokio
workers, whatever `--posix` says; the run prints `s3 engine: object_store VERSION on tokio
(epoll sockets)`. Each actor has one request in flight. Under `--s3 blocking` the actor's
thread builds and signs it and waits, the workers driving the connections; under `--s3
async` the workers do all of it and the loop goes on with its other actors until the
answer wakes it. Two actors writing one object at once (a sub-actor and the actor it was
forked from) fail the run with `EBUSY`. **aeiou datagen**
writes each file of an object dataset as one `PUT`, or as a multipart upload in 8 MiB
parts when it is larger, and the manifest as one `PUT` at the prefix's root; the prefix
must hold no key. A `regions` dataset has no object form. An object namespace's root is a
prefix: rank 0 finds it empty with a `LIST` (`--clean-namespaces` sends a `DELETE` of every
key below it but those of a dataset placed there), and writes the namespace manifest as one
`PUT` at it, where a run that reads the namespace as `input` finds it. A write at an offset
the upload has not reached fails the run with `ESPIPE`: V18 refuses what it can see in the
abstract, and offsets are positional, so the order is checked at the write. The report's
mount counters and the residency sample of `--require-cold` do not cover an object store,
and the run says so.

## METRICS

`dry-run --metrics` computes, for comparison with `aeiou-trace metrics` of a trace of the
real application: the read/write mix; request sizes per kind; the reuse distance of every
block access (the smallest LRU cache, in bytes, in which the access is a hit), in four
histograms by the kinds of the previous and this access, with first touches counted apart;
sequential run lengths in bytes and operations; the popularity of blocks and of objects
(the share of accesses going to the most popular 0.1 %, 1 %, and 10 %); the fan-out of
every `parallel`; and the depth of consecutive dependent `parallel`s. Order-dependent
metrics are taken one instance at a time in the round-robin order of its sub-actors, then
summed over instances. Histograms of bytes use quarter-octave buckets.

## SEVERAL HOSTS

`aeiou run --ranks R --rank r --coordinator HOST:PORT` on each of *R* hosts, rank 0 first;
**aeiou-launch**(1) does it over ssh. Rank 0 runs the coordinator in-process and connects to
it like every other host. Each host sends its identity document, which must match rank 0's
field for field, and its options block; after the gate rank 0 prints the layered options
whose values differ between hosts. Barriers are two-level (in-process, then across hosts).
Rank 0 merges every host's report, writes the namespace manifests, applies
`--expect-fingerprint` to the merged fingerprint, and sends one verdict to every host. A
failure on any host stops the run on all of them, naming the originating rank. Only rank 0
empties namespace roots: `--root` is the storage under test, shared by every host.

## ENVIRONMENT

- **AEIOU_CONFIG**

  The config file, when `--config` is not given. See **aeiou-config**(5).
- **AEIOU_ROOT**, **AEIOU_THREADS**, **AEIOU_RANK**, and every `AEIOU_<FLAG>`

  A layered option's value, the long flag upper-cased with `_` for `-`
  (`AEIOU_BUFFER_MIB`, `AEIOU_REQUIRE_COLD`). Booleans take `true`/`false`, `1`/`0`,
  `yes`/`no`, `on`/`off`; `AEIOU_NO_X` is refused with the spelling to use. A variable
  naming a fixed option is refused; one naming no option of any subcommand is warned about
  and ignored.

## FILES

- `*.ast.json`

  An abstract, written by **aeiou-build**(1); **aeiou-abstract**(7).
- `*.params.json`

  A parameter file; **aeiou-params**(1).
- `.aeiou-dataset.json`

  The manifest `datagen` leaves at a dataset root and `run` compares. Its normative part
  (the resolved dataset definition, the payload block, the format block) hashes to the
  dataset id.
- `.aeiou-namespace.json`

  The manifest `run` leaves at every namespace root it wrote, and requires at the root of
  every namespace declared `input`.
- The file a `trace` node names

  JSON Lines written by `aeiou-trace export`, relative to the abstract, pinned by the
  node's sha256.
- The TOML config file

  **aeiou-config**(5).

## EXIT STATUS

- **0**

  Success. For `run`, the verdict was good on this host (on several hosts the same verdict
  is sent to every host).
- **1**

  A failure during the work after the command line was accepted: an invalid abstract, a
  manifest that does not match, a non-empty namespace, a limit the host cannot hold, an
  operation whose result was not the computed one, a fingerprint or dataset id that is not
  the expected one. `check` exits 1 if any file failed.
- **2**

  A usage error. Every missing argument is listed at once, from every layer, in the frame
  every tool of the suite shares.

## EXAMPLES

Validate the committed abstracts and print their identities:

```
aeiou check schema/examples/*.ast.json
```

The fingerprint and totals of eight instances of a training workload at seed 1:

```
aeiou dry-run schema/examples/train_small_files.ast.json --gpus 8 --seed 1
```

One instance's stream, the first fifty operations of its second step:

```
aeiou dry-run schema/examples/kv_cache_serving.ast.json --gpus 1 \
    --param concurrency=2 --param requests=20 --gpu 0 --steps 1..2 --limit 50
```

Generate a small corpus, then run against it under the abstract's own backend, holding the
run to the dry run's fingerprint:

```
aeiou datagen schema/examples/train_small_files.ast.json --root /mnt/sut --param files=4000
aeiou run schema/examples/train_small_files.ast.json --root /mnt/sut --gpus 8 --seed 1 \
    --param files=4000 --param steps=50 --expect-fingerprint 3f1c... --report-json run.json
```

The same run on io_uring with a capped io-wq, the metrics of the stream to a file:

```
aeiou run ... --posix io_uring --threads 4 --iowq-max-workers 2
aeiou dry-run schema/examples/train_small_files.ast.json --gpus 8 --metrics-json abstract.metrics.json
```

A parameter file over the defaults:

```
aeiou dry-run schema/examples/model_load.ast.json --gpus 8 \
    --params-file schema/examples/params/model_load.synthetic.params.json
```

## SEE ALSO

**aeiou-build**(1), **aeiou-params**(1), **aeiou-datagen**(1), **aeiou-trace**(1),
**aeiou-launch**(1), **aeiou-config**(5), **aeiou-abstract**(7).

The reference: `runner/REFERENCE.md` (the definitions the runner fixes, §2; `aeiou run`,
§4; the payload and manifest, §5; the coordinator, §6; the backends, §8 and §9; the
metrics, §10; the limits, §11; the JSON report, §12; the `trace` node, §13; the option
layers, §14; usage errors, §15).
