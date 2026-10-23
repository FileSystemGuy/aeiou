# aeiou(1)

## NAME

aeiou - Author Execute I/O workload runner: validate, dry-run, generate datasets for, and execute an abstract

## SYNOPSIS

```
aeiou [--config FILE] check FILES...
aeiou [--config FILE] dry-run [OPTIONS] --gpus GPUS ABSTRACT | --case NAME | --case-file PATH
aeiou [--config FILE] datagen [OPTIONS] --root DIR ABSTRACT | --case NAME | --case-file PATH
aeiou [--config FILE] datagen [OPTIONS] --resume ROOT
aeiou [--config FILE] datagen [OPTIONS] --grow N ROOT
aeiou [--config FILE] datagen [OPTIONS] --shrink N ROOT
aeiou [--config FILE] datagen [OPTIONS] --settle ROOT
aeiou [--config FILE] run [OPTIONS] --gpus GPUS --root DIR ABSTRACT | --case NAME | --case-file PATH
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

A **use case** names everything that goes into a run in one file, its **lock**
(`case.lock.json`, made by **aeiou-params**(1) `lock`; contract 0.11): `--case NAME` finds it
under aeiou's `workloads/`, `--case-file PATH` reads it where it is, and the lock pins its
abstract by its path and SHA-256 and holds every parameter with its state, **locked**,
**default**, or **required**, and where it came from. A run of a lock prints those values and
two hashes, the lock's **file hash** (is this the exact file published?) and its **workload
id** (does this run do the same thing?), which results are compared by; it covers the values
in effect, the command line's replacements of a default included, and the identity options
(`--seed`, `--posix`, `--cache`, `--s3`, `--time-scale`, `--rank-rotate`, `--write-compress`),
and leaves out the values the lock requires the submission to give and the numbers the
storage reports. A lock made with a benchmark's rules may hold **constraints** over the
runner's `fact.` and `result.` names (FACTS AND RESULTS), each judged when its names are
known: at resolve, at the start gate (before any I/O; one that fails refuses the run), or
after the run (one that fails is a valid run that did not pass: exit status 3).

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
With `--ranks` it adds the bytes per host, against this host's memory. Last, one line per
dataset projects what **datagen** writes for it at these parameters: samples, files (chunks
and objects counted), directories, mean file size, and total bytes, the inventory's
included, summed exactly from the size expressions and the framing. With `--gpu` it
prints one instance's operation stream, one line per operation, in the round-robin order
of that instance's concurrent sub-actors. With `--metrics` it computes the locality
metrics of the stream (see METRICS below) for comparison with a trace of the real
application through **aeiou-trace**(1).

### datagen

Writes every dataset the abstract declares under `--root`: names from
the pattern, sizes from the dataset seed, content from the positional payload generator
(1 MiB blocks of an SFC64 stream keyed by dataset seed, unit, and block, laid out in 4 KiB
pieces for the compression ratio, with the dedupe and compression ratios given), in parallel by id, with `O_DIRECT` (datagen never
reads what it writes, and the client's page cache is what a benchmark must not have warm; a
root that refuses `O_DIRECT` falls back to the page cache, said once). The manifest
`.aeiou-dataset.json` is written at each dataset root before the first file, its state
`writing` and every id unfinished; every `--progress-secs` each rank flushes what it has
written and reports it, and the manifest is rewritten (atomically, by rank 0) with the ids
not yet written, and a progress line printed; once every file is written it says
`complete`, and its id is printed. A dataset that declares an `inventory` gets it in its
root just before (rank 0, at the count the corpus ends at, whichever operation ended it).
`run` refuses a dataset that is not `complete`. A
non-empty root is refused: datasets are read-only. A dataset that declares a
container format class is written by **aeiou-datagen**(1); a build with the cargo feature
`formats` (the release's) writes TFRecord, tar (WebDataset), HDF5, and Parquet here too,
the same bytes (a Parquet footer names this writer as `created_by`). An HDF5 file's header comes from **aeiou-header**(1), which must
sit beside this binary; only rank 0 runs it.

The ids of a dataset (every layout but `array`) are written in runs, handed out in id order from one queue
(rank 0's on several hosts): a run is a directory of a `{id div N}` pattern, or 1,000 ids of a
flat one, cut smaller only when there are fewer runs than writers on all hosts. Each writer
writes its run in id order and claims the next, so a datagen stopped anywhere leaves whole
directories, a band of partial ones no wider than a run per writer on every host (and one
claimed ahead per writer on a host other than rank 0), and nothing past it. Before handing out
a run past it, rank 0 records a claim mark in the manifest (`progress.claimed_to`, a few bands
ahead), so `--settle` knows how far a datagen that died may have written.

On several hosts (`--ranks`, `--rank`, `--coordinator`, as for `run`; **aeiou-launch**(1)
starts them) every rank claims the runs of every dataset from rank 0 over the
coordinator, and each `array` dataset is written whole by one rank (the *i*-th by rank *i*
mod *R*: several hosts writing one file serialize on the server). Every rank checks the roots are empty
before the start gate, writes after it, reports its progress every window, and sends its
counts at the end; rank 0 writes the manifests, `complete` once every rank has reported, and records
every rank's host and counts in the provenance. The coordinator refuses a host whose
abstract, parameters, `--gpus`, `--dedupe`, `--compress`, or `--dataset` list differ.

Ctrl-C (`SIGINT`) or `SIGTERM` on any host stops every host: each stops claiming, leaves the
file it is writing, notes what it finished, flushes, and reports to rank 0, which rewrites the
manifests once more, marked `stopped` (no datagen is writing the corpus), prints that every
rank stopped, and every host exits 130 (143 the host a `SIGTERM` reached). The state is
unchanged, so `--resume ROOT` finishes the corpus and `--settle ROOT` keeps what it reached.
A second signal exits at once; the manifest
of the last window is still true. A host that fails stops the others the same way, as a
failure: every host prints the failing host's rank and error and exits 1.

`datagen --resume ROOT` finishes a dataset a datagen left `writing`, `growing`, or
`shrinking` (killed, crashed, or failed), from the manifest at ROOT alone: the dataset, its place, the payload, and the
format are the manifest's, so it takes no abstract. Every id the manifest lists as
unfinished is written again, whole (a file opened with truncation, so a partial or stale one
loses its tail); in an object store, where an upload that did not complete leaves no object,
only the unfinished ids whose keys are missing (one `LIST`); the keys found there count
toward the dataset's totals, so they are the same, and not toward the resume's own record
(`provenance.resumes`, where they are `files_found`) or its summary, which says how many
were already there. The unfinished ids are split
over the hosts of this datagen, whatever their count when the corpus was begun. It refuses a
corpus whose payload or format writer is not this aeiou's, or whose HDF5 headers
**aeiou-header**(1) would make differently, since the files would mix them. The manifest
is rewritten every window as for a first datagen and `complete` at the end, with the same
dataset id; each resume's hosts and counts are added to the provenance (`resumes`). A
manifest already `complete` has nothing to write, and the command says so and exits 0. A
manifest updated less than two progress windows ago (`progress.updated`,
`progress.progress_secs`) may still have a datagen writing it and is refused, unless it says
`stopped`. Not
done: the parts of multipart uploads a crash left in a bucket are not aborted (the object
engine has no call that lists them).

`datagen --grow N ROOT` and `datagen --shrink N ROOT` change the count of the complete
dataset at ROOT to N (its `count`: samples where a file holds several, files
otherwise), larger for a grow and smaller, above 0, for a shrink; each refuses the other's
direction, so a typo cannot delete a corpus. Everything else comes from the manifest, as for
`--resume`, and is refused beside it, as is a corpus not `complete`. The files a datagen of
N would write and the corpus does not hold are written: a grow's new ids, and the old last
file again when it was short (it gains samples, and an HDF5 file a header). A shrink removes
the names past N (a chunked file's chunks; a key in an object store) and every directory that
leaves empty, and writes the new last file again when it is short. The corpus is then, byte
for byte, what a datagen of N writes, with that datagen's dataset id. The manifest is
rewritten first, `growing` or `shrinking` with the new count and the old one
(`progress.from`), so `run` refuses the corpus until the change is done and `--resume ROOT`
finishes an interrupted one (removing again is no error); it ends `complete`, with the change
recorded in the provenance (`resizes`: the old count and id, hosts, counts, times). The work
splits over the hosts as a resume's does. A shrink hands out its runs from the top down and
removes each from its highest id down.

`datagen --settle ROOT` keeps what an interrupted datagen, grow, or shrink reached, claiming
nothing new. The count is the boundary of the runs handed out (`progress.claimed`; whole
files, a directory boundary), never past the count asked for nor short of the count a grow or
shrink began from. The runs handed out and not finished are finished (a shrink's removals
among them, never more: a shrink is never continued, and `--grow` undoes one); ids between
the boundary and the mark, where a datagen that died after its last window may have written
or removed, are removed past the count and written again below it. The manifest is rewritten
first with the settled count and what is left to do (`progress.settle`: the counts asked and
settled), so an interrupted settle is finished by `--settle ROOT` or `--resume ROOT`, and ends
`complete`, with that count's dataset id, the settle in the provenance (`settles`), and a
grow or shrink it ended marked `settled`. A count of 0 is refused (remove the corpus). As
`--resume` otherwise, the live check included; a manifest from an aeiou that recorded no
claim boundary is refused (`--resume` finishes it).

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
every namespace root it wrote. On one host, and on rank 0 of several, a line sets the
datasets' projected bytes against the hosts' memory: every host sends its memory, CPUs,
kernel, and cgroup memory limit to rank 0 with its first message. These are facts, never a
verdict: whether a ratio is enough is a methodology's rule, not the runner's.
`--report-json` writes the same as one JSON document.

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

  The run seed, default 0 (a lock's, when it sets one). The dataset seed is separate and lives in the abstract.
- **--case** *NAME*

  A use case by its lock's name, in place of *ABSTRACT*: `mechanism/case`, or the case alone
  when one lock matches. The lock is found by a walk of aeiou's `workloads/` for every
  `case.lock.json`; none or several are refused with every match listed. The lock's run
  becomes the bottom layer of the parameters: a parameter file or `--param` may replace a
  default and must give a required value, and never replaces a locked one. Its identity
  options are the run's where the command line gives none, and one the lock locks (or its
  rules freeze) is refused when given otherwise.
- **--case-file** *PATH*

  A use case by its lock's path, no search: what a driver that keeps its own locks passes.
- **--case-run** *NAME*

  The run of a lock of several runs (a writer and its reader), by its abstract's name;
  required for such a lock and refused for a lock of one run.
- **--workloads** *DIR*

  aeiou's `workloads/` directory, where `--case` searches and a lock's abstract path is
  resolved. Default: `../workloads` beside the executable (the release's `bin/`), else the
  source checkout's.
- **--expect-case-sha** *SHA256*

  Refuse a lock whose file SHA-256 is not this one, before anything is done.
- **--expect-workload-id** *SHA256*

  Refuse a run whose workload id is not this one, before anything is done.
- **--st-blksize** *[DATASET=]BYTES*

  The `st_blksize` the storage reports, for the parameters whose default is
  `{"from": "st_blksize"}` (contract 0.11): *BYTES* for every dataset they read, or
  *DATASET*`=`*BYTES* for one. Repeatable. Without it each such parameter takes its
  `fallback`; `aeiou run` reads the number from the storage itself, by `stat` of the
  dataset's first file, and prints and reports it, so a dry run of another run's ops takes
  the number that run's report gives.

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

  With `--gpu`: stop printing after *N* operations. `--steps` or `--limit` without `--gpu`
  is a usage error.

**Metrics**

- **--[no-]metrics**

  Compute the locality metrics of the operation stream: reuse distance, sequential runs,
  popularity, request sizes, fan-out and depth, read/write mix. Nothing here changes the
  stream or the fingerprint.
- **--metrics-block** *BYTES*

  With `--metrics`: the block metrics too, reuse and popularity in blocks of *BYTES*, as the
  captures taken before 2026-10-22 have them; the extent metrics, over byte ranges, are
  always there. `--metrics-block` without `--metrics` or `--metrics-json` is a usage error.
- **--metrics-json** *FILE*

  Write the metrics, with their histograms in full, to *FILE* as JSON (format
  `aeiou_metrics: 2`). Implies `--metrics`.

### aeiou datagen

- *ABSTRACT*

  The abstract (`.ast.json`).

**Workload**

- **--param** *NAME=VALUE*

  As for `dry-run`.
- **--params-file** *FILE*

  As for `dry-run`.
- **--case** *NAME*

  As for `dry-run`: the corpus is written from the values the lock's run checks against.
- **--case-file** *PATH*

  As for `dry-run`.
- **--case-run** *NAME*

  As for `dry-run`.
- **--workloads** *DIR*

  As for `dry-run`.
- **--expect-case-sha** *SHA256*

  As for `dry-run`.
- **--expect-workload-id** *SHA256*

  Refused: a run's check (a dry run prints the id); `--expect-dataset-id` of the run checks
  the corpus.
- **--gpus** *GPUS*

  Instance count, for dataset definitions that reference `gpus`. Default 1.

**Writer**

- **--root** *DIR*

  Required, from some layer. The directory the abstract's paths are relative to. From the
  command line, else `$AEIOU_ROOT`, else `root` in the `[datagen]` table of the config
  file. *Layered.*
- **--endpoint** *NAME=DIR|URI*

  Puts the root directory of the abstract's dataset or namespace NAME at DIR instead of
  under `--root` (a dataset whose root lies inside a namespace's moves with it), or, for a
  name declared `protocol: s3`, at `s3://BUCKET[/PREFIX]` (repeatable); see ENDPOINTS. From the command line, else `$AEIOU_ENDPOINT`, else `endpoint` in the
  `[datagen]` table of the config file. *Layered.*
- **--writers** *N*

  Writer threads: files, or 64 MiB pieces of an `array` file, written at once; default all
  cores. *Layered.*
- **--object-threads** *N*

  Worker threads of the object engine's runtime, which drive its connections (the writers'
  threads make the requests); see OBJECT STORES. Only with a dataset placed in an object
  store. Default 2. *Layered.*
- **--dedupe** *DEDUPE*

  Dedupe ratio: files share their content *DEDUPE* by *DEDUPE*, each byte at its own file
  offset, so whole 4 KiB blocks repeat near *DEDUPE* times (`aeiou-positional/4`). Files of
  one sample share in sets of near sizes among each 8 × *DEDUPE* consecutive ids, files of
  several samples *DEDUPE* consecutive ids; a block holding a container's framing differs,
  and a set's largest file keeps its tail (`--dedupe 2`: TFRecord and Parquet shards near
  1.86, lognormal image files near 1.85, equal sizes 2). An `array` file repeats each 1 MiB
  block *DEDUPE* times. The sets are functions of the id, whatever the count. Default 1. Part
  of the payload the manifest records.
- **--compress** *COMPRESS*

  Compression ratio: in every 4 KiB, the first ⌈4096/C⌉ bytes are random and the rest
  0x5A, so every file and every piece compresses at C. Default 1. Part of the payload the
  manifest records.
  At most 4096; a C that does not divide 4096 makes 4096/⌈4096/C⌉ (3 makes 2.9985, 1000
  makes 819.2), which datagen prints and the manifest's provenance records as
  `compress_achieved`.
- **--dataset** *NAME*

  Write only these datasets; repeatable. Default all.
- **--progress-secs** *SECS*

  Seconds between progress notes: each rank flushes what it has written and reports it,
  and rank 0 rewrites each manifest with the ids not yet written and prints one line
  (files, bytes, rate, time left). Default 30. *Layered.*

**Checks**

- **--[no-]ignore-space**

  Before anything is written, rank 0 sums the bytes still to write per file system of the
  directory roots (the projection **dry-run** prints, less what a resumed or resized
  manifest says is there) and refuses when they exceed what `statvfs` reports available
  there; a line `space:` gives each sum and the free space, and warns above 90% of it or
  for an object root, which reports none. This flag writes anyway (for a system under test
  that stores less than it is sent, by deduplicating or compressing), the line then
  `space: IGNORED:`. File-system metadata is not counted.

**Several hosts**

- **--rank** *R*

  This host's index among `--ranks` hosts. Default 0. *Layered.*
- **--ranks** *N*

  Hosts the datagen is spread over: each claims runs of every dataset's ids from
  rank 0, and each `array` file is written by one of them. Default 1. *Layered.*
- **--coordinator** *HOST:PORT*

  With `--ranks` above 1, required: the coordinator's address. Rank 0 listens on it
  (in-process); every rank connects to it. *Layered.*

**Resume**

- **--resume** *ROOT*

  In place of *ABSTRACT*: finish the dataset whose root is ROOT (a directory, or
  `s3://BUCKET[/PREFIX]`) from its manifest; see **datagen** above. No abstract, `--root`,
  `--endpoint`, `--param`, `--params-file`, `--gpus`, `--dedupe`, `--compress`, or
  `--dataset` with it (a usage error; a `root` or `endpoint` from the environment or the config
  file is passed over, with a warning in the options block); `--writers`, `--object-threads`, `--progress-secs`, and
  the several-hosts options as for any datagen. With several hosts, rank 0 alone reads the
  manifest and sends every host what to write; every host gives the same ROOT, written the
  same way (the coordinator refuses one that does not), and a refusal on rank 0 is reported
  on every host.

- **--grow** *N* *ROOT*

  In place of *ABSTRACT*: grow the complete dataset at ROOT to a count of N, larger than its
  own; see **datagen** above. As `--resume` for what may stand beside it; one of `--resume`,
  `--grow`, `--shrink`, `--settle` at a time. With several hosts, the coordinator refuses a host growing
  another ROOT or to another count.

- **--shrink** *N* *ROOT*

  In place of *ABSTRACT*: shrink the complete dataset at ROOT to a count of N, smaller than
  its own and above 0; as `--grow` otherwise.

- **--settle** *ROOT*

  In place of *ABSTRACT*: keep what an interrupted datagen, grow, or shrink of the dataset at
  ROOT reached, at the count of the runs it had handed out; see **datagen** above. As
  `--resume` for what may stand beside it; one of `--resume`, `--grow`, `--shrink`,
  `--settle` at a time.

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
- **--case** *NAME*

  As for `dry-run`. The report holds the lock itself, its file hash, the workload id, every
  parameter with its value, state, and source (`case`), and every constraint's verdict
  (`constraints`).
- **--case-file** *PATH*

  As for `dry-run`.
- **--case-run** *NAME*

  As for `dry-run`.
- **--workloads** *DIR*

  As for `dry-run`.
- **--expect-case-sha** *SHA256*

  As for `dry-run`.
- **--expect-workload-id** *SHA256*

  As for `dry-run`.
- **--[no-]subset**

  Run a dataset whose count (from `--param`, as always) is below its corpus's as the
  corpus's first ids. The count is rounded up to whole directories of a `{id div D}`
  pattern, `D` times the samples per file (to whole files when the pattern has no
  directories), which are byte for byte a fresh corpus of that count, so the run is that
  corpus's workload and fingerprint; the dataset id is the corpus's. The options block shows
  the count asked and the count run, and the report's dataset entry has a `subset` block.
  The subset is the corpus's lowest ids, the first written: a tiering system may serve them
  from its coldest tier, and the run says so. Only for a consumed dataset (`map`, `stream`)
  whose `count` is a parameter and whose definition differs from the corpus's in nothing
  else. Refused: a count above the corpus's (`aeiou datagen --grow` makes it), one that
  rounds up past it (the message names the largest whole subset), an `indexed` dataset, a
  flat dataset whose directory the abstract lists (the listing would be the whole corpus),
  and `--subset` when no dataset is smaller. A dataset's inventory is read as a corpus of the
  count run would have it: the corpus's, whose first entries are that list, with every read
  issued no further than its end there (the tail short, the read that finds the end with
  length 0), so the ops and the fingerprint are the fresh corpus's. The command line's alone.

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

  Event-loop threads for `--posix io_uring` and `libaio` and `--s3 async`; default one per
  core, at most one per actor instance. The other APIs run one thread per actor and refuse
  the flag (from the environment or the config file it is passed over with a warning, as
  is every backend knob the run's API cannot use; **aeiou-config**(5)). *Layered.*
- **--buffer-mib** *MIB*

  The per-thread read and write buffer rings, in MiB. Default 8. *Layered.*
- **--write-compress** *C*

  Compression ratio of the bytes written to namespaces, at most 4096. Default 1. *Layered.*
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
  (`MADV_WILLNEED`). Default `fault`. *Layered.*
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
  Written when the run fails too, with the error, and when Ctrl-C or `SIGTERM` stops it
  (`interrupted (SIGINT)`). The file is created (empty) before the
  run starts, so a place it cannot be written is a usage error, not a finding after the
  run. Rank 0 of several hosts writes the merged report, every other rank its own.
  *Layered.*
- **--[no-]report-takes**

  With `--report-json`: every take of every instance (stall and compute), not only the
  sums. *Layered.*

## BACKENDS

The abstract is identical under every backend; a backend maps operations to an API and
never changes the stream or the fingerprint. Each protocol has its own API: `--posix` for
the `posix` names, with the cache mode `--cache`, and `--s3` for the `s3` names. A run
compares only with runs under the same choices. An option for a protocol the abstract has
no names on is refused, since it would change nothing (a knob from the environment or the
config file is passed over with a warning instead).

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
endpoints for one root must agree. Two datasets never share a place: an endpoint that would
put one dataset's root at another's, or inside it, or around it (under `--root` included) is
a usage error, since each dataset root holds its own manifest. The run prints each endpoint, and warns when one lies on
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
`object_store` with the cargo feature `object`, on by default (the release's binary has it; a build with
cargo's default features turned off refuses an object endpoint). The store is the one the environment names: `AWS_ENDPOINT_URL`,
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
| `rename` | a copy on the server and a `DELETE` of the source; not atomic; an object over 5 GiB, which one `CopyObject` cannot copy, fails with `EFBIG` |
| `lseek`, `fadvise`, `fsync`, `fdatasync`, `ioctl`, `mkdir`, `rmdir` | none (an `ioctl` answers as for a regular file: `TCGETS` is `ENOTTY`; a prefix is no object) |
| `O_APPEND`, `O_RDWR`, `O_EXCL`, `O_CREAT` or `O_TRUNC` without `O_WRONLY`, `ftruncate`, `fallocate`; an open for writing with no `close` after it | refused by rule V18 at `aeiou check` (**aeiou-abstract**(7)) |

The requests go over ordinary sockets (`epoll`) on the engine's `--object-threads` tokio
workers, whatever `--posix` says; the run prints `s3 engine: object_store VERSION on tokio
(epoll sockets)`. Each actor has one request in flight. Under `--s3 blocking` the actor's
thread builds and signs it and waits, the workers driving the connections; under `--s3
async` the workers do all of it and the loop goes on with its other actors until the
answer wakes it. Two actors writing one object at once (a sub-actor and the actor it was
forked from) fail the run with `EBUSY`. **aeiou datagen**
writes each file of an object dataset as one `PUT`, or as a multipart upload in 8 MiB
parts when it is larger (larger parts when 10,000 would not hold it), and the manifest as one `PUT` at the prefix's root; the prefix
must hold no key. An `array` dataset has no object form. An object namespace's root is a
prefix: rank 0 finds it empty with a `LIST` (`--clean-namespaces` sends a `DELETE` of every
key below it but those of a dataset placed there), and writes the namespace manifest as one
`PUT` at it, where a run that reads the namespace as `input` finds it. A write at an offset
the upload has not reached fails the run with `ESPIPE`: V18 refuses what it can see in the
abstract, and offsets are positional, so the order is checked at the write. The report's
mount counters and the residency sample of `--require-cold` do not cover an object store,
and the run says so.

A request has no deadline: a large part takes what the store takes, as a POSIX op does
(`AWS_TIMEOUT` sets one; the connect timeout is 5 s). A request the store answers 5xx, 429,
or 408, or whose connection fails, is sent again, as S3 SDKs do, up to 10 times within 180 s,
inside its op's latency; the run prints `object retries N` with the causes, and a warning
when there was any, and the JSON report has them under `result.object`, so a throttling
store (`503 SlowDown`) shows.

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
  naming a fixed option is refused, as is an empty one, or one that is not UTF-8 where the
  option takes text; one naming no option of any subcommand is warned about and ignored;
  a backend knob the run cannot use is passed over with a warning. `AEIOU_ENDPOINT`, the
  one list, is whitespace-separated (`AEIOU_ENDPOINT="kv=/a sysp=/b"`).
- **AWS_ENDPOINT_URL**, **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**, **AWS_REGION**,
  **AWS_ALLOW_HTTP**, **AWS_VIRTUAL_HOSTED_STYLE_REQUEST**, **AWS_TIMEOUT**

  The object store and its client, for a name placed at `s3://BUCKET[/PREFIX]`; see OBJECT
  STORES.

## FILES

- `*.ast.json`

  An abstract, written by **aeiou-build**(1); **aeiou-abstract**(7).
- `*.params.json`

  A parameter file; **aeiou-params**(1).
- `.aeiou-dataset.json`

  The manifest `datagen` leaves at a dataset root and `run` compares. Its normative part
  (the resolved dataset definition, the payload block, the format block) hashes to the
  dataset id; its `state` (`writing`, `growing`, or `shrinking`, then `complete`) and, while
  writing, its `progress` (the ids not yet written, the claim mark, whether its datagen
  `stopped`) are not hashed.
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
- **3**

  `run` of a lock: a valid run that did not pass a constraint known after it (a `result.`
  name). The run completed and its report is written with every verdict and no error; on
  several hosts rank 0 exits 3 and the others 0. A constraint that fails at resolve or at
  the gate is an invalid run, refused before any I/O, and exits 1.
- **130**, **143**

  Stopped by Ctrl-C (`SIGINT`, 130) or `SIGTERM` (143, what Slurm, systemd, docker, and
  `timeout` send) on this host. `run` stops its actors, prints `interrupted (SIGINT)` (or
  `SIGTERM`), and writes its `--report-json` with that as the error; the other hosts of the
  run fail with the interrupted rank and the reason and exit 1. `datagen` stops every host,
  and the manifest records what is finished; a host stopped by another host's signal exits
  130. A second signal exits at once.
- **141**

  Standard output's reader went away (`| head`) and nothing else failed: the rest of the
  output is dropped, silently, and the work goes on to its end (a run measures and writes
  its `--report-json` as it would have), as 128 + `SIGPIPE`, the status the shell shows for
  a tool that signal stopped.

## FACTS AND RESULTS

The names a lock's constraints read (contract 0.11), each defined here and in
`runner/REFERENCE.md` §17, with the time it is known:

- **fact.gpus** (resolve): *G*, the GPU instances of the run.
- **fact.dataset_bytes** (resolve): every dataset's bytes as `datagen` writes them.
- **fact.min_samples_per_gpu** (resolve): the least, over the consumed datasets, of
  floor(count / *G*).
- **fact.hosts** (gate): the hosts of the run.
- **fact.sum_host_mem**, **fact.min_host_mem** (gate): the hosts' memory summed and its
  least, each host's `MemTotal` or cgroup `memory.max`, whichever is less.
- **fact.max_gpus_per_host** (gate): ceil(*G* / hosts).
- **result.elapsed_s** (after): the run's seconds from the start gate to its end.
- **result.busy_fraction** (after): compute × time-scale / (compute × time-scale + stall)
  over every take of every instance but each instance's first.
- **result.min_gpu_busy_fraction** (after): the least, over the instances that take, of
  that busy fraction.

## EXAMPLES

Validate the committed abstracts and print their identities:

```
aeiou check $(find workloads -name '*.ast.json')
```

The fingerprint and totals of eight instances of a training workload at seed 1:

```
aeiou dry-run --case jpeg-50m --gpus 8 --seed 1
```

One instance's stream, the first fifty operations of its second step:

```
aeiou dry-run workloads/kvcache/lmcache-local-disk/lmcache_local_disk.ast.json --gpus 1 \
    --param concurrency=2 --param requests=20 --gpu 0 --steps 1..2 --limit 50
```

Generate a small corpus, then run against it under the abstract's own backend, holding the
run to the dry run's fingerprint:

```
aeiou datagen --case jpeg-50m --root /mnt/sut --param samples=4000
aeiou run --case jpeg-50m --root /mnt/sut --gpus 8 --seed 1 \
    --param samples=4000 --param steps=50 --expect-fingerprint 3f1c... --report-json run.json
```

The first 2,600 files of that corpus (two whole directories of 1,300), the fingerprint the
dry run gives for `samples=2600`:

```
aeiou run --case jpeg-50m --root /mnt/sut --gpus 8 --seed 1 \
    --param samples=2000 --param steps=50 --subset
```

The same run on io_uring with a capped io-wq, the metrics of the stream to a file:

```
aeiou run ... --posix io_uring --threads 4 --iowq-max-workers 2
aeiou dry-run --case jpeg-50m --gpus 8 --metrics-json abstract.metrics.json
```

A parameter file over the defaults:

```
aeiou dry-run workloads/model-loading/safetensors-load/safetensors_load.ast.json --gpus 8 \
    --params-file workloads/model-loading/safetensors-load/params/synthetic.params.json
```

## SEE ALSO

**aeiou-build**(1), **aeiou-params**(1), **aeiou-datagen**(1), **aeiou-header**(1), **aeiou-trace**(1),
**aeiou-launch**(1), **aeiou-config**(5), **aeiou-abstract**(7).

The reference: `runner/REFERENCE.md` (the definitions the runner fixes, §2; `aeiou run`,
§4; the payload and manifest, §5; the coordinator, §6; the backends, §8 and §9; the
metrics, §10; the limits, §11; the JSON report, §12; the `trace` node, §13; the option
layers, §14; usage errors, §15; the object engine, §16).
