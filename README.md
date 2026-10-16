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
   runs the script in a hermetic child, twice, and writes the **AST**, one JSON document
   whose canonical SHA-256 is the workload's identity. The AST is the only thing the runner
   executes and the only thing an author publishes. Its slots are filled by **parameter
   files**, so one published shape carries several measured parameter sets.

2. **Check and dry-run.** `aeiou check` validates a document against the schema and the
   semantic rules. `aeiou dry-run` walks every actor instance without I/O and prints the
   operation counts, bytes, and the fingerprint: the order-independent hash of the multiset
   of positioned operations. It can print one instance's stream, and it can compute the
   stream's **locality metrics**: reuse distance, sequential runs, popularity skew, request
   sizes, fan-out and depth, read/write mix.

3. **Generate data.** `aeiou datagen` (and `aeiou-datagen` for container formats) writes
   the datasets the abstract declares: names from the pattern, sizes from the dataset seed,
   content from a positional generator with controlled dedupe and compression ratios, so
   any byte is a pure function of (dataset seed, file, offset) and nothing is stored to
   describe it. A manifest at each dataset root hashes to the **dataset id**, which a run
   checks before it starts.

4. **Run.** `aeiou run` executes the abstract against the storage under one of nine I/O
   backends (buffered and direct POSIX, io_uring, POSIX AIO, libaio, mmap), one host or
   several through a TCP coordinator with no MPI. It checks every result structurally and
   never looks at the bytes; it reports latency histograms, per-step stall, host counters
   (tasks, io-wq workers, CPU, RSS, the mount's NFS RPC deltas), and the fingerprint, which
   must be the dry run's. A result is the abstract's hash, the parameters in effect, and the
   dataset ids.

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
| `aeiou-datagen` | Python | container datasets through their format classes | [aeiou-datagen(1)](man/aeiou-datagen.1.md) |
| `aeiou-trace` | Python | metrics of an strace, comparison, the `trace` node's file | [aeiou-trace(1)](man/aeiou-trace.1.md) |
| `aeiou-catalog` | Python | the workload catalog: records against their dictionary, each case's shape, `CATALOG.md`, each case's description in prose | [aeiou-catalog(1)](man/aeiou-catalog.1.md) |
| `aeiou-launch` | sh | one `aeiou run` or `aeiou datagen` rank per host over ssh | [aeiou-launch(1)](man/aeiou-launch.1.md) |

The option layers and the config file are [aeiou-config(5)](man/aeiou-config.5.md); the
abstract, its JSON form, its rules, parameter files, and manifests are
[aeiou-abstract(7)](man/aeiou-abstract.7.md). Every tool reports a wrong command line in
one frame, listing every missing argument at once, and exits 2.

## Repository layout

| Path | What |
|---|---|
| `builder/` | The Python package `aeiou` and its four tools; `abstracts/` holds the twelve authoring scripts of the fourteen committed workloads; `traces/` the capture kit and trace metrics of each real application traced. [README](builder/README.md), [reference](builder/REFERENCE.md). |
| `schema/` | The AST contract: the JSON Schema, the reference validator `check.py`, and the parameter-file schema. [README](schema/README.md). |
| `workloads/` | The committed abstracts, one directory per I/O mechanism under its category (`input-pipeline/map/file-per-sample/`, `kvcache/lmcache-local-disk/`, …): the authoring script and the AST generated beside it (never edited by hand), the capture kit, and the cases, each a parameter set describing one workload with the captures of the traced application under it (`DESIGN_REVIEW.md` §3.66). |
| `runner/` | The Rust workspace: the `aeiou` binary and `aeiou-launch`. [README](runner/README.md), [reference](runner/REFERENCE.md). |
| `man/` | The manual pages, in Markdown with man-page sections; the test suites hold them to the tools' `--help`. [index](man/README.md). |
| `QUICKSTART.md` | The guide shipped in the release tarball: first run, first bug report. |
| `PROJECT_BRIEF.md` | Requirements, decisions, open items (§6), and the one list of what is MLPerf-specific (§8). |
| `NAPKIN_MATH.md` | Memory and IOPS estimates, the risk register, the spikes. |
| `GRAMMAR_OPTIONS.md` | The semantic model and the authoring options; Option D, the Python builder over a JSON AST, is what was built. |
| `ABSTRACTS.md` | The target workloads written on paper first, the constructs they needed, and what each trace later corrected. |
| `DESIGN_REVIEW.md` | The reasoning log: every design decision with its date, what it replaced, and why. |

## Technologies

**Runner (Rust, edition 2021).** `clap` for the command line, `serde`/`serde_json` for the
AST and the reports, `toml` for the config file, `sha2` for identities, `xxhash-rust`
(xxh3) for positional keys and the fingerprint, its own SFC64 for the positional payload,
`libc` and the `io-uring` crate for the backends, `anyhow` for errors. Blocking `std::net`
for the coordinator; no tokio, no MPI. Linux interfaces: `io_uring` (with `SQPOLL`,
`DEFER_TASKRUN`, `COOP_TASKRUN`, io-wq worker caps), the kernel AIO calls, glibc POSIX AIO,
`mmap` with `madvise`, `O_DIRECT`, `posix_fadvise`, `mincore`, `/proc/sys/vm/drop_caches`,
`/proc/self/mountstats` for NFS RPC counts, `getrusage`, the rlimits and sysctls a run
checks before its gate.

**Builder (Python ≥ 3.12).** `jsonschema` is the only dependency of the package; `hatchling`
builds it and `uv` locks it. The `formats` extra brings the libraries of the format
classes, `pyarrow`, `h5py`, `numpy` (whose `SFC64` makes the payload, the Rust writer's
bytes), `crc32c`, and `xxhash`. PEP 578 audit hooks make the hermetic build.
`aeiou-trace` and `aeiou-catalog` are standard library only.

**Contract.** JSON Schema draft 2020-12 for the AST and the parameter file; canonical JSON
(sorted keys, no whitespace, shortest-round-trip floats) under SHA-256 for identities.

**Calibration.** `strace -f -ttt -T -yy` of the real applications: PyTorch `DataLoader`,
NumPy, `torch.distributed.checkpoint`, safetensors, DiskANN, FAISS, vLLM with LMCache.

**CI.** GitHub Actions, three jobs: the builder tests with the hermetic rebuild of every
abstract against the committed ASTs and every parameter file against its abstract; the
runner's `cargo test`; and the integration job that diffs the Rust checker against the
reference checker, runs the container datasets to their fingerprints, and holds the metrics
of an strace of the runner to its own dry run. A tag `vX.Y.Z` runs a fourth workflow that
builds the release tarball and attaches it to a GitHub Release (`.github/workflows/release.yml`).

## Quick start

A release tarball (the runner, `aeiou-launch`, the manual pages, the abstracts, and the
builder's wheel, for x86_64 Linux) is attached to every tag; [QUICKSTART.md](QUICKSTART.md)
is the guide for it. From source:

```
# the runner (`formats`: aeiou datagen writes TFRecord and tar, as the release binary does)
cd runner && cargo build --release --features formats && cargo test --release --features formats && cd ..

# the builder (or: pip install -e 'builder[test,formats]')
cd builder && uv sync --extra test --extra formats && uv run pytest && cd ..

# check, dry-run, generate, run
runner/target/release/aeiou check $(find workloads -name '*.ast.json')
runner/target/release/aeiou dry-run workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json --gpus 8 --seed 1
runner/target/release/aeiou datagen workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json --root /mnt/sut --param samples=4000
runner/target/release/aeiou run workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json --root /mnt/sut --gpus 8 --seed 1 \
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

## Status (2026-10-14)

A working proof of concept, measured so far on ext4 and on a loopback NFS v4.2 mount on
one development box, whose GPU has now replayed the agentic KV-cache load through vLLM and
LMCache (item 21: `keep` measured, buffered and `O_DIRECT` both traced; the buffered replay corrected 2026-10-08 and the main agent's and sub-agents' prompts modelled apart, its pair not yet accepted; run again at three times the engine's KV pool, that pair accepted, the engine and CPU tier a fixed reference, item 24). Built and tested: everything described above, the nine backends, the
multi-host coordinator (two ranks on `localhost`), the limit checks, the cold-start
controls, the JSON report, the locality metrics on both sides with tolerances, the `trace`
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
(`{id div files_per_directory}`) and the layout parameters' standard names (the same day, §3.75), and the step barrier every step by default in the training workloads, its latency measured on up to 32 ranks (2026-10-14, §3.71 point 4), and the abstract in prose: every case's `DESCRIPTION.md`, a paragraph and an account generated from its AST, parameters, records, and shape, each paragraph in `CATALOG.md`, and `aeiou-catalog describe` for any AST (the same day, §3.76). The workload catalog
(2026-10-10, `workloads/CATALOG.md`; seventeen categories since 2026-10-11, AI and not, each
with its directory, twelve of them still empty): thirteen mechanisms and sixteen cases under
`workloads/`, each labeled in the vocabulary of `workloads/attributes.toml`, with the shape
its dry run measures, checked in CI by `aeiou-catalog`.

Not built, in rough order of interest:

- ~~The abstract rendered as prose (`PROJECT_BRIEF.md` §6 item 27), from the AST and the case's
  measured shape; the workload catalog it would join is built (2026-10-10, below).~~ Built
  2026-10-14 (`DESIGN_REVIEW.md` §3.76); the same for a raw trace not yet.
- The access layers (decided 2026-10-09, `PROJECT_BRIEF.md` §6 item 25; mixed protocols included; the API
  and the cache mode as separate axes, contract 0.6, and the protocol and `--endpoint` per
  dataset and namespace built the same day, the mount counters still `--root`'s alone):
  transport and buffer beside them; ~~then object storage through Apache `object_store` behind a feature flag (item 17; ~~`s3dlio`~~ decided 2026-10-09, `DESIGN_REVIEW.md` §3.65),~~
  ~~the object engine's writes (object namespaces, multipart at `close`, V18) and~~ ~~the object engine's event-loop
  bridge,~~ its reads of datasets and its writes of namespaces (multipart at `close`, V18) on Apache `object_store` behind the cargo feature `object`,
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

## License

Apache License 2.0. See `LICENSE`.
