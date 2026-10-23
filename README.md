# aeiou: Author Execute I/O

**aeiou** is a reproducible, abstract-driven storage workload benchmark. You *author* a
workload as a short program over POSIX I/O operations, the shape of what a real
application does, and *execute* it against a storage system, on one host or many, so that
every system under test sees the same multiset of operations and the results compare.

It was written at [Hammerspace](https://hammerspace.com) out of the MLPerf Storage working
group's need for benchmarks that reproduce the scripted I/O sequences of AI training and
inference jobs: the `open`, two `lseek`s, `read` to EOF, `close` of a PyTorch loader
worker; a checkpoint's items streamed through one buffer; the dependent rounds of a
DiskANN beam search; a KV cache's write-then-read lag. The tool itself is general. Nothing
in it is named for MLPerf, and what is specific to the working group's process is listed in
one place (`PROJECT_BRIEF.md` §8) so that any other user, a storage vendor's QA, a
filesystem developer, a researcher, can ignore it.

## Where to start

Pick the row that matches what you have; each path is spelled out below.

| You have | You want | Path | Needs |
|---|---|---|---|
| A storage system | To measure it with a workload already written | **A** | the runner |
| A job like a shipped workload, at other sizes | The shipped workload at your job's sizes | **B** | the runner, `aeiou-params` |
| An application not yet modeled | An abstract of it that scales and compares | **C** | `strace`, the builder, analysis (a person, or Claude) |
| An application and one capture of it | That capture played back exactly, as soon as possible | **D** | `strace`, `aeiou-trace` |

**What the tools do and do not do.** `aeiou-build` does not read traces: it compiles a Python
builder script, which someone wrote, into an abstract. `aeiou-trace` measures an `strace`
(`metrics`), judges an abstract against it (`compare --judge`), and exports it for literal
playback (`export`). Nothing in the suite turns a trace into a builder script: recognizing
the application's loops, its datasets and how it reads them, what it draws at random, and
which calls depend on which is analysis, done by a person or by asking Claude (Claude Code)
in this repository, which has the tools, the existing workloads, and the rules at hand.
Fitting the parameters of a script to a trace is a script per capture kit today
(`workloads/<category>/<mechanism>/kit/`); a general `aeiou-fit` is not built.

**A. Run a shipped workload.** Start with [QUICKSTART.md](QUICKSTART.md) (the release tarball)
or [Quick start](#quick-start) below (from source). Choose a workload from
[`workloads/CATALOG.md`](workloads/CATALOG.md), where every case has a description in prose,
then `aeiou check`, `aeiou dry-run` (op counts and the fingerprint, no I/O), `aeiou datagen`
(the dataset, once per root), `aeiou run`. The fingerprint `dry-run` prints must equal the one
`run` prints.

**B. A shipped workload at your job's sizes.** Write the abstract's defaults as a parameter
file, edit it to your job's values, check it, and run with it; for model loads and `.npz`
datasets, `aeiou-params safetensors` and `aeiou-params npz` read the values from your real
files:

```
aeiou-params defaults -o mine.params.json ABSTRACT
aeiou-params check ABSTRACT mine.params.json
aeiou dry-run ABSTRACT --gpus 8 --seed 1 --params-file mine.params.json
```

If the trace of your job would differ from the shipped one in *sequence* (other calls, another
order, another reader), not just in sizes, it is path C.

**C. Model a new application (the recommended sequence).**

1. **Read the application first.** Its source and vendor documentation say what it does and
   why; a trace only shows what it did once. Note its I/O API: an `strace` sees POSIX calls
   and the kernel AIO calls, but not `io_uring` submissions or reads through a mapping (for
   those, exact call counts and a page-residency probe stand in, as for the model-load and
   IVF-search workloads).
2. **Capture an strace** of the real application, at a size small enough to trace, on the
   storage you will name with `--root`:

   ```
   strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy \
       -o trace.txt <command>
   aeiou-trace metrics trace.txt --root /mnt/data -o trace.metrics.json
   ```

   Keep the command, the application's versions, and its logs: they go in the capture's
   record.
3. **Ask Claude to analyze the trace and write the abstract**: give it `trace.txt`,
   `trace.metrics.json`, the application's source or name and version, and the command line.
   Ask for a builder script under the right `workloads/<category>/<mechanism>/` directory.
   Claude chooses that directory: the category from the seventeen in
   [workloads/attributes.toml](workloads/attributes.toml), and the mechanism's name for the I/O
   mechanism, never the use case. Have it say which it chose and why, and confirm before it
   builds: every case built on the abstract names it by that path, so a later rename touches
   them all. You need to agree with the reasoning, not to know the vocabulary. Ask for the script
   built with `aeiou-build --hermetic --twice`, and a parameter file fitted to the capture, then judged
   against the trace. Use Claude Opus 5.5 or a more capable model for this step. (A person can do the same by hand: the builder's language is
   [aeiou-abstract(7)](man/aeiou-abstract.7.md), the shipped scripts are the examples.)
4. **Judge the abstract against the trace**, at its fitted parameters and against other seeds
   of itself, so it is not asked to be nearer the trace than it is to itself:

   ```
   aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --seed 1 --metrics-json abstract.metrics.json
   for s in 2 3 4; do aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --seed $s --metrics-json self$s.json; done
   aeiou-trace compare trace.metrics.json abstract.metrics.json --judge --self self2.json --self self3.json --self self4.json
   ```

   A row outside its tolerance means the script, not just a parameter, is wrong more often
   than not: back to step 3 with the rows.
5. **Commit it as a case** with its capture (`cases/<case>/captures/<capture>/`, the verdict
   in `capture.toml`), and regenerate the catalog: [workloads/README.md](workloads/README.md),
   "Adding one". Traces you may not publish stay out of the tree; the metrics document and
   the verdict are what is committed.
6. **Scale it.** The abstract is now a shape: a parameter file at a real job's sizes is a
   new case (path B), with no new trace.

**D. Play back one capture exactly.** `aeiou-trace export` writes the trace as the file the
runner's `trace` node executes literally, one lane per traced task, with no analysis:

```
aeiou-trace export trace.txt --root /mnt/data -o app.jsonl       # prints the sha256
```

then a builder script with one actor of `count=1` whose body is `w.trace("app.jsonl",
sha256)`, and `aeiou run` with `--gpus 1` ([runner/REFERENCE.md](runner/REFERENCE.md) §13).
It is that one capture and nothing more: one process on one host, at the sizes it was
traced at, in the order that day's timing produced, and never CLOSED, so it is not a
result to compare across systems. It is a quick check of a storage system against an
application, and a reference to run beside path C's abstract; it does not replace it.

## Why another benchmark

Most storage benchmarks describe a workload as a probability per operation: so many percent
reads of such a size at such a queue depth. Real jobs are sequences, and the sequence is
what storage sees: the metadata calls around every file, the order a reader walks a
checkpoint, how far apart a block's write and its re-read are, which blocks are hot and for
how long. A probability mix has no sequence, so it can neither reproduce those effects nor
be checked against the application it claims to model.

An aeiou **abstract** keeps the sequence and makes everything the application draws at
random a *positional* function of a seed, so the workload is exact: two runs with the same
abstract, seed, instance count, and parameters issue the same operations on every system,
the run's **fingerprint** proves it, and the locality metrics of the stream are compared
with the same metrics taken from an `strace` of the real application.

## Theory of operation

An abstract goes through five stages, each a tool of the suite.

1. **Author.** A Python script declares the workload with the builder package: parameters,
   datasets (sample spaces that exist before the run), namespaces (the objects the run
   creates), and actors whose bodies are loops, bindings, forks into sub-actors, channels,
   barriers, emulated compute, and the eighteen operations. The calls that look like I/O
   are declarations: Python runs once at build time and generates *structure*. `aeiou-build`
   runs the script (with `--hermetic --twice`, as CI does: in two isolated children whose
   hashes must agree) and writes the **AST**, one JSON document
   whose canonical SHA-256 is the workload's identity. The AST is the only thing the runner
   executes and the only thing an author publishes. Its slots are filled by **parameter
   files**, so one published shape carries several measured parameter sets.

2. **Check and dry-run.** `aeiou check` validates a document against the schema and the
   semantic rules. `aeiou dry-run` walks every actor instance without I/O and prints the
   operation counts, bytes, and the fingerprint: the order-independent hash of the multiset
   of positioned operations. It can print one instance's stream, and it can compute the
   stream's **locality metrics**: reuse distance, sequential runs, popularity skew, request
   sizes, fan-out and depth, read/write mix.

3. **Generate data.** `aeiou datagen` writes the datasets the abstract declares, container
   formats (TFRecord, tar, HDF5, Parquet) included, to directories or object stores: names
   from the pattern, sizes from the dataset seed, content from a positional generator with
   controlled dedupe and compression ratios, so any byte is a pure function of (dataset
   seed, file, offset) and nothing is stored to describe it. A manifest at each dataset root
   hashes to the **dataset id**, which a run checks before it starts.

4. **Run.** `aeiou run` executes the abstract against the storage under one of nine I/O
   backends (POSIX calls, io_uring, POSIX AIO, and libaio, each buffered or direct, and
   mmap), and the names it places in an object store through Apache `object_store`, blocking
   or async; one host or several through a TCP coordinator with no MPI. It checks every
   result structurally and never looks at the bytes; it reports latency histograms, per-step
   stall, host counters (tasks, io-wq workers, CPU, RSS, the mount's NFS RPC deltas), and
   the fingerprint, which must be the dry run's. A result is the abstract's hash, the
   parameters in effect, and the dataset ids.

5. **Calibrate.** `aeiou-trace` computes the same locality metrics from an `strace` of the
   real application and compares the two documents row by row against a tolerance per
   class; an abstract is accepted for a workload class when it is within tolerance of its
   application, and no nearer than it is to itself at another seed. For the dynamic half of
   the calibration, a trace can also be exported as the file a `trace` node executes
   literally, one lane per traced task, so the application's own call sequence runs through
   the same runner against the same storage.

### The ideas underneath

- **Randomness is positional.** A draw is keyed on (seed, actor id, the site of the draw,
  the enclosing loop indices). No draw counters, no shared random state, no clock, no
  completion order: every instance's stream can be computed alone, in any order, on any
  host, and the multiset of operations is fixed before the run.
- **Sharding is a formula.** `consume` draws samples without replacement through a keyed
  Feistel permutation of the dataset at position `g + G·(b·B + j)`, which splits an epoch
  over `G` instances with no coordination and no per-file data structure. Datasets of fifty
  million files cost nothing to hold.
- **The fingerprint is order-independent.** It sums per-operation hashes, so timing can
  never change it, and two runs that agree executed the same positioned operations.
- **Data is reproducible, not self-describing.** No block headers (they force a dedupe
  ratio of one); dedupe and compression are ratios the generator honours (dedupe in whole
  4 KiB blocks, near `D` where sizes vary or framing sits between samples:
  `runner/REFERENCE.md` §5); the runner checks
  counts and errnos only, and content verification is an offline tool, never in a scored run.
- **Datasets are sample spaces; the runner is format-ignorant.** A container format (Parquet,
  HDF5, TFRecord, tar) is a generic framing formula in the AST plus a reader protocol the
  builder emits as plain operations; the runner computes every offset from the formula and
  reads no container metadata.
- **Random access is the null model.** The structure storage can exploit in data-dependent
  workloads (reuse, skew, dependency shape, write-then-read lag) is expressed with positional
  distributions and `x @ i`, a binding read at an earlier index of its loop, and checked
  against traces.
- **The abstract declares its application's API.** A workload is comparable across systems
  only under the backend the traced application uses (`sync` for PyTorch, `mmap` for
  safetensors, `libaio` for DiskANN); any other backend is a different workload on the
  storage and the run says so. Anything an `LD_PRELOAD` shim could do under an unmodified
  application belongs to the system under test.
- **Access is layered, and one run may mix protocols.** Formats sit above the access APIs,
  and below them each choice is a separate axis: the protocol (POSIX or S3) and the
  endpoint of each dataset and namespace, the API, one per protocol (`--posix sync`,
  `io_uring`, `libaio`, …; `--s3 blocking` or `async`), the cache mode of the POSIX names
  (`--cache per-open` or `direct`), the transport (TCP or RDMA), and the buffer (host or
  GPU memory). Since the protocol belongs to a dataset or namespace rather than to the run,
  one run can read its prompts from a POSIX tier and its KV chunks from an object tier, as
  KV-cache engines with several storage tiers do, with the same op stream and fingerprint
  either way (decided 2026-10-09; the API and cache axes and the per-namespace protocol and
  `--endpoint` built, and the object engine's reads and writes on Apache `object_store`, blocking or async, beside any POSIX API;
  `PROJECT_BRIEF.md` §6 item 25).
- **Every option has a provenance.** Options resolve through the command line, the
  environment, a TOML file, and the default; what the identity depends on is the command
  line's alone; every invocation prints what it resolved and from where, and on several
  hosts the coordinator records every host's and prints what differs.

## The tools

| Tool | Language | What it does | Manual |
|---|---|---|---|
| `aeiou` | Rust | `check`, `dry-run`, `datagen`, `run` | [aeiou(1)](man/aeiou.1.md) |
| `aeiou-build` | Python | builder scripts to abstracts, hermetically | [aeiou-build(1)](man/aeiou-build.1.md) |
| `aeiou-params` | Python | parameter files: defaults, check, from real shards and archives | [aeiou-params(1)](man/aeiou-params.1.md) |
| `aeiou-datagen` | Python | container datasets through their format classes; deprecated, since `aeiou datagen` writes every format | [aeiou-datagen(1)](man/aeiou-datagen.1.md) |
| `aeiou-trace` | Python | metrics of an strace, comparison, the `trace` node's file | [aeiou-trace(1)](man/aeiou-trace.1.md) |
| `aeiou-catalog` | Python | the workload catalog: records against their dictionary, each case's shape, `CATALOG.md`, each case's description in prose, and `describe` for any abstract | [aeiou-catalog(1)](man/aeiou-catalog.1.md) |
| `aeiou-header` | Python | the header bytes of an HDF5 file from libhdf5, run by `aeiou datagen` beside it | [aeiou-header(1)](man/aeiou-header.1.md) |
| `aeiou-launch` | sh | one `aeiou run`, `aeiou datagen`, or `aeiou-datagen` rank per host over ssh | [aeiou-launch(1)](man/aeiou-launch.1.md) |

The option layers and the config file are [aeiou-config(5)](man/aeiou-config.5.md); the
abstract, its JSON form, its rules, parameter files, and manifests are
[aeiou-abstract(7)](man/aeiou-abstract.7.md). Every tool reports a wrong command line in
one frame, listing every missing argument at once, and exits 2.

## Repository layout

| Path | What |
|---|---|
| `builder/` | The Python package `aeiou` and its six tools (`aeiou/`), and its tests (`tests/`). The workloads it builds live under `workloads/`. [README](builder/README.md), [reference](builder/REFERENCE.md). |
| `schema/` | The AST contract: the JSON Schema, the reference validator `check.py`, and the parameter-file schema. [README](schema/README.md). |
| `workloads/` | The committed abstracts, one directory per I/O mechanism under its category (`input-pipeline/map/file-per-sample/`, `kvcache/lmcache-local-disk/`, …): the authoring script and the AST generated beside it (never edited by hand), the capture kit (`kit/`), test-sized parameter sets (`params/`), and the cases, each a parameter set describing one workload with its description in prose and the captures of the traced application under it; the catalog, `CATALOG.md`, generated from the records in the vocabulary of `attributes.toml` (`DESIGN_REVIEW.md` §3.66). [README](workloads/README.md), [catalog](workloads/CATALOG.md). |
| `runner/` | The Rust workspace: the `aeiou` binary and `aeiou-launch`. [README](runner/README.md), [reference](runner/REFERENCE.md). |
| `man/` | The manual pages, in Markdown with man-page sections; the test suites hold them to the tools' `--help`. [index](man/README.md). |
| `QUICKSTART.md` | The guide shipped in the release tarball: first run, first bug report. |
| `.github/workflows/` | CI (`ci.yml`) and the release tarball (`release.yml`). |
| `PROJECT_BRIEF.md` | Requirements, decisions, open items (§6), and the one list of what is MLPerf-specific (§8). |
| `NAPKIN_MATH.md` | Memory and IOPS estimates, the risk register, the spikes. |
| `GRAMMAR_OPTIONS.md` | The semantic model and the authoring options; Option D, the Python builder over a JSON AST, is what was built. |
| `ABSTRACTS.md` | The target workloads written on paper first, the constructs they needed, and what each trace later corrected. |
| `DESIGN_REVIEW.md` | The reasoning log: every design decision with its date, what it replaced, and why. |
| `RC1_BUG_LOG.md` | The release-candidate review's bug log, append-only: each bug, its fix, and its evidence, replayable from the tag `pre-rc1-review`. |

## Technologies

**Runner (Rust, edition 2021).** `clap` for the command line, `serde`/`serde_json` for the
AST and the reports, `toml` for the config file, `sha2` for identities, `xxhash-rust`
(xxh3) for positional keys and the fingerprint, its own SFC64 for the positional payload,
`libc` and the `io-uring` crate for the backends, `anyhow` for errors. The object engine,
the cargo feature `object`, on by default: Apache `object_store` (S3), `tokio`, `futures`,
`bytes`; `formats`, also on by default, the container writers of `aeiou datagen`, with
`crc32c`. Blocking `std::net` for the coordinator, tokio only in the
object engine; no MPI. Linux interfaces: `io_uring` (with `SQPOLL`,
`DEFER_TASKRUN`, `COOP_TASKRUN`, io-wq worker caps), the kernel AIO calls, glibc POSIX AIO,
`mmap` with `madvise`, `O_DIRECT`, `posix_fadvise`, `mincore`, `/proc/sys/vm/drop_caches`,
`/proc/self/mountstats` for NFS RPC counts, `getrusage`, the rlimits and sysctls a run
checks before its gate.

**Builder (Python ≥ 3.12).** `jsonschema` is the only dependency of the package; `hatchling`
builds it and `uv` locks it. The `formats` extra brings the libraries of the format
classes, `pyarrow`, `h5py`, `numpy` (whose `SFC64` makes the payload, the Rust writer's
bytes), `crc32c`, and `xxhash`; the `s3` extra brings `obstore`, Python's binding of
Apache `object_store`, for `aeiou-datagen` into a bucket. PEP 578 audit hooks make the
hermetic build. `aeiou-trace` and `aeiou-catalog` are standard library only; `aeiou-header`
needs h5py, and the release ships it as one PyInstaller file carrying Python, numpy, and h5py.

**Contract.** JSON Schema draft 2020-12 for the AST and the parameter file; canonical JSON
(sorted keys, no whitespace, shortest-round-trip floats) under SHA-256 for identities.

**Calibration.** `strace -f -ttt -T -yy -v` of the real applications: PyTorch `DataLoader`,
NumPy, `torch.distributed.checkpoint`, safetensors, DiskANN, FAISS, vLLM with LMCache.

**CI.** GitHub Actions, four jobs: the builder tests with the hermetic rebuild of every
abstract against the committed ASTs and every parameter file against its abstract; the
runner's whole suite (`cargo nextest`), the object engine's tests against a pinned MinIO
included; the object engine's tests again in a build without `formats`, with
`aeiou-datagen` into the bucket read back to the dry-run fingerprint; and
the integration job that diffs the Rust checker against the reference checker, runs every
Python test against the binary (the container datasets to their fingerprints, the metrics of
an strace of the runner held to its own dry run), and checks the workload catalog. A test
skipped for something its job built fails (`AEIOU_REQUIRE`). A tag `vX.Y.Z` runs a second
workflow that builds the release tarball and attaches it to a GitHub Release (`.github/workflows/release.yml`).

## Quick start

A release tarball (the runner, `aeiou-header`, `aeiou-launch`, the manual pages, the
abstracts with their cases and catalog, and the builder's wheel, for x86_64 Linux) is
attached to every tag; [QUICKSTART.md](QUICKSTART.md) is the guide for it. From source:

```
# the runner, with its default features as the release binary has them (`formats`: aeiou
# datagen's container writers; `object`: the object engine, for S3 stores); its object tests
# start the MinIO server TEST_MINIO_BIN names (runner/REFERENCE.md §16)
cd runner && cargo build --release && TEST_MINIO_BIN=/path/to/minio cargo test --release && cd ..

# the builder (or: pip install -e 'builder[test,formats]')
cd builder && uv sync --extra test --extra formats && uv run pytest && cd ..

# check, dry-run, generate, run
runner/target/release/aeiou check $(find workloads -name '*.ast.json')
runner/target/release/aeiou dry-run --case jpeg-50m --workloads workloads --gpus 8 --seed 1
runner/target/release/aeiou datagen --case jpeg-50m --workloads workloads --root /mnt/sut --param samples=4000
runner/target/release/aeiou run --case jpeg-50m --workloads workloads --root /mnt/sut --gpus 8 --seed 1 \
    --param samples=4000 --param steps=50 --report-json run.json
```

The workloads shipped as abstracts: small-file training, large-sample training (`.npz`),
checkpoint write and restore (`torch.distributed.checkpoint`), model load (safetensors),
DiskANN search and index build, IVF search (FAISS), KV-cache serving and a shared KV store
(vLLM with LMCache), and streaming training over TFRecord, Parquet, and HDF5 containers.
The eight application workloads have each been traced against their real application (two
of them, model load and IVF search, read through a mapping, which an `strace` cannot see,
so their fit rests on exact call counts and a page-residency probe instead); the container
workloads' reader protocols were traced at the library level, pyarrow, h5py, and `tarfile`
on a loopback NFS mount, TFRecord's from source.

## Status (2026-10-22)

A working proof of concept, measured so far on ext4 and on a loopback NFS v4.2 mount on
one development box, whose GPU has now replayed the agentic KV-cache load through vLLM and
LMCache (item 21: `keep` measured, buffered and `O_DIRECT` both traced; the buffered replay corrected 2026-10-08 and the main agent's and sub-agents' prompts modelled apart, its pair not yet accepted; run again at three times the engine's KV pool, that pair accepted, the engine and CPU tier a fixed reference, item 24). Built and tested: everything described above, the nine backends, the
multi-host coordinator (two ranks on `localhost`), the limit checks, the cold-start
controls, the JSON report, the locality metrics on both sides with tolerances (over extents, byte ranges, since 2026-10-22, `DESIGN_REVIEW.md` §3.79), the `trace`
node under every backend, the option layers, the usage-error frame, and `datagen` across
hosts with `O_DIRECT` (each host its slice of the ids, one manifest once every host has
reported; the launcher takes host files and `pdsh`-style ranges), and `aeiou-datagen`, the
container writer, across hosts the same way, speaking the same coordinator protocol
(2026-10-11), and `aeiou datagen` writing TFRecord and tar itself behind the cargo feature
`formats`, the Python writer's bytes at two to four times its speed, and HDF5 with its
header from `aeiou-header`, a one-file tool beside `aeiou` that rank 0 alone runs, and Parquet by our own
writer, pyarrow's bytes but for `created_by` (2026-10-12), and the datagen manifest written from
the start with its progress, and `datagen --resume ROOT`, which finishes a corpus from that
manifest alone on any host count, and `datagen --grow N ROOT` and `--shrink N ROOT`, which
make a corpus the one a datagen of count N writes, and `datagen --settle ROOT`, which keeps
what an interrupted one reached at whole directories, every datagen handing its ids out in
id order from one queue on rank 0, and Ctrl-C stopping every host with the manifest true
(2026-10-12), and contract 0.8, one dataset shape: a `layout` (`whole`, `chunked`,
`compound`, `array`), an `access` (`map`, `stream`, `indexed`), and `format.framing`
(2026-10-13, `DESIGN_REVIEW.md` §3.72), and contract 0.9, a consumed dataset's `inventory`
(the same day, §3.73), and contract 0.10, a pattern's divisor named by a parameter
(`{id div files_per_directory}`) and the layout parameters' standard names (the same day, §3.75), and the step barrier every step by default in the training workloads, its latency measured on up to 32 ranks (2026-10-14, §3.71 point 4), and the abstract in prose: every case's `DESCRIPTION.md`, a paragraph and an account generated from its AST, parameters, records, and shape, each paragraph in `CATALOG.md`, and `aeiou-catalog describe` for any AST (the same day, §3.76), and the release-candidate
review (2026-10-14 to 2026-10-16, `RC1_BUG_LOG.md`: 176 bugs, each fixed with its evidence)
and release 0.2.0 (2026-10-16), then 0.2.1 the same day, every build with the object engine and
the container writers (`object` and `formats` default features, `DESIGN_REVIEW.md` §3.65). The workload catalog
(2026-10-10, `workloads/CATALOG.md`; seventeen categories since 2026-10-11, AI and not, each
with its directory, twelve of them still empty): thirteen mechanisms and twenty-six cases under
`workloads/` (fifteen examples, one use case, ten `unsettled`), each labeled in the vocabulary of `workloads/attributes.toml`, with the shape
its dry run measures, checked in CI by `aeiou-catalog`. Use cases and the lock (2026-10-22,
`PROJECT_BRIEF.md` §6 item 33, `DESIGN_REVIEW.md` §3.78, `runner/REFERENCE.md` §17): contract
0.11 (required parameters, the `scale` tag, values the storage reports, `st_blksize`),
parameter files of version 2 whose entries are locked, default, or required, combined by
agreement, a benchmark's rules file, and `aeiou-params lock`, which writes a case's
`case.lock.json` (re-locked by `aeiou-build`, checked by `aeiou-catalog`); `run`, `dry-run`,
and `datagen` by `--case NAME` or `--case-file`, the workload id beside the lock's hash,
constraints on the `fact.` and `result.` names judged at resolve, at the gate, and after the
run (exit 3 for a valid run that did not pass), and the step-time histograms with the ten
slowest delays; the five input-pipeline abstracts take their job and GPU values from cases,
`npy_per_sample`'s from the use case `unet3d-h100`, and since 2026-10-23 `dcp_save` (item 38's
first), its write buffer the `st_blksize` of its output namespace, read by a probe file
(contract 0.12), `dcp_load`, its read buffers the `st_blksize` of its input namespaces, and the
three LMCache abstracts, the model's, the deployment's, and the load's values (contract 0.13:
rule V3 judges a required parameter in an `at` offset when its value is given),
`safetensors_load`, the model's shards and tensor table and the deployment's `tp`, and
`diskann_build`, the base's shape, the build's configuration, its graphs, and its compute.

Not built, in rough order of interest:

- ~~The abstract rendered as prose (`PROJECT_BRIEF.md` §6 item 27), from the AST and the case's
  measured shape; the workload catalog it would join is built (2026-10-10, below).~~ Built
  2026-10-14 (`DESIGN_REVIEW.md` §3.76); the same for a raw trace not yet.
- The access layers (decided 2026-10-09, `PROJECT_BRIEF.md` §6 item 25; mixed protocols included; the API
  and the cache mode as separate axes, contract 0.6, and the protocol and `--endpoint` per
  dataset and namespace built the same day, the mount counters still `--root`'s alone):
  transport and buffer beside them; ~~then object storage through Apache `object_store` behind a feature flag (item 17; ~~`s3dlio`~~ decided 2026-10-09, `DESIGN_REVIEW.md` §3.65),~~
  ~~the object engine's writes (object namespaces, multipart at `close`, V18) and~~ ~~the object engine's event-loop
  bridge,~~ its reads of datasets and its writes of namespaces (multipart at `close`, V18) on Apache `object_store` (the cargo feature `object`, on by default since 2026-10-16),
  under `--s3 blocking` or `async` (an API per protocol, contract 0.7), built 2026-10-09 (item 17, `runner/REFERENCE.md` §16); `aeiou-datagen` (the Python container writer) to object stores built 2026-10-11, each file streamed into an upload with no local copy, and every writer, HDF5 and Parquet included, holding a piece of its file rather than the file (`DESIGN_REVIEW.md` §3.69); a measurement against a real store not yet; and the backends from the brief's list not yet written: `gds`, `nixl-posix`, `libnfs`.
- A run on real hosts (the coordinator's connect window and heartbeat constants have only
  met `localhost`), and tensor-parallel sharding of a KV chunk, where `--gpus` is today
  independent engines rather than one TP group (`PROJECT_BRIEF.md` §6 item 20).
- `aeiou-verify`, the offline content verifier that takes a dataset manifest and
  regenerates any byte; `aeiou-fit`, the fitting of parameter files from a trace (each
  capture kit today has its own fitting script).
- Format classes for Arrow IPC, MDS, and Megatron `.bin`/`.idx`, and the Parquet-to-Arrow
  conversion abstract (item 15).
- An `O_DIRECT` checkpoint writer to trace (item 19); a recorded wait time per lane and a
  cap on the file for the `trace` node; `RLIMIT_MEMLOCK` in the limit checks;
  wall-clock-bounded phases (a contract change).
- Computed holds (item 24): what a KV engine holds derived from its pool in the runner,
  not drawn at one reference engine.
- ~~`datagen --resume` (rewrite what is missing or short, by size alone),~~ growing a corpus
  in place, and a run over the first *S* ids of a larger corpus (item 23): one long datagen
  on a fleet, then a series of runs growing toward the saturation point. First, decided
  2026-10-11 (`DESIGN_REVIEW.md` §3.70): the container formats written by `aeiou datagen`
  in Rust behind a cargo feature `formats`, so that item 23 is built once, in Rust;
  TFRecord, tar, and HDF5 built 2026-10-12 (HDF5's header from `aeiou-header` beside
  `aeiou`), ~~Parquet not yet~~ Parquet the same day (our own writer, pyarrow's bytes but for
  `created_by`); the manifest written from the start, `writing` with the ids not yet
  written, rewritten every `--progress-secs`, and `complete` at the end, the same day.
  `datagen --resume ROOT` the same day (the unfinished ids written again, whole, from the
  manifest alone, on any host count); ~~Resume, whole directories,~~ grow
  and shrink the same day, settle, claims in id order, and Ctrl-C the same day; subset runs (`aeiou run --subset`, whole directories of the corpus, 2026-10-13; whole directories not required of a corpus, §3.71), ~~the inventory variant not yet;~~ the inventory the same day (contract 0.9, `DESIGN_REVIEW.md` §3.73: a consumed dataset's file list, read in place of a walk, its shape `[measure]` until a traced job reads one); ~~a sizing step for
  training corpora proposed the same day (`DESIGN_REVIEW.md` §3.71).~~ the sizing step for
  training corpora left to the WG's `mlpstorage` (2026-10-13, `DESIGN_REVIEW.md` §3.74);
  the host facts in the report, the dataset projection in `dry-run`, and the capacity check
  in `datagen` built the same day; `div` by a parameter (contract 0.10, `DESIGN_REVIEW.md`
  §3.75) the same day.
- Fidelity for small datasets (item 28: per-node miss-ratio curves, cold-start time, a
  validation run on one realistic node), the harness's topology in the report (item 29),
  and concurrent mixes of cases (item 30, critical but not yet; `DESIGN_REVIEW.md` §3.67).
- Retiring `aeiou-datagen`, the Python writer, now that `aeiou datagen` writes every format
  class (item 32, decided 2026-10-14): deprecated meanwhile, no new features, the byte
  reference in tests until then.
- Recapturing the seventeen committed traces as extent metrics (item 34, 2026-10-22), until
  which they are judged on the block metrics `--metrics-block` keeps; and the extent metrics
  split by byte role, a format's framing against its payload (item 35).
- Traces of the three containers never traced, `hdf5_map`, `parquet_stream`, and
  `tfrecord_stream`, each fitted to an example case that replaces its `unsettled` one (item
  36); a lognormal's `sigma` as an expression, so a case gives it (item 37); the job values of
  the two abstracts outside the input pipelines still holding them moved to cases, one mechanism at a time
  with the user (item 38); and a use case per accelerator, `unet3d-h100` the first (item 39).

## License

Apache License 2.0. See `LICENSE`.
