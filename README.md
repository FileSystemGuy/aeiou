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
  ratio of one); dedupe and compression are ratios the generator honours; the runner checks
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
| `aeiou-launch` | sh | one `aeiou run` rank per host over ssh | [aeiou-launch(1)](man/aeiou-launch.1.md) |

The option layers and the config file are [aeiou-config(5)](man/aeiou-config.5.md); the
abstract, its JSON form, its rules, parameter files, and manifests are
[aeiou-abstract(7)](man/aeiou-abstract.7.md). Every tool reports a wrong command line in
one frame, listing every missing argument at once, and exits 2.

## Repository layout

| Path | What |
|---|---|
| `builder/` | The Python package `aeiou` and its four tools; `abstracts/` holds the twelve authoring scripts of the fourteen committed workloads; `traces/` the capture kit and trace metrics of each real application traced. [README](builder/README.md), [reference](builder/REFERENCE.md). |
| `schema/` | The AST contract: the JSON Schema, the reference validator `check.py`, the parameter-file schema, and `examples/`, the committed abstracts (generated, never edited by hand) with example parameter sets. [README](schema/README.md). |
| `runner/` | The Rust workspace: the `aeiou` binary and `aeiou-launch`. [README](runner/README.md), [reference](runner/REFERENCE.md). |
| `man/` | The manual pages, in Markdown with man-page sections; the test suites hold them to the tools' `--help`. [index](man/README.md). |
| `PROJECT_BRIEF.md` | Requirements, decisions, open items (§6), and the one list of what is MLPerf-specific (§8). |
| `NAPKIN_MATH.md` | Memory and IOPS estimates, the risk register, the spikes. |
| `GRAMMAR_OPTIONS.md` | The semantic model and the authoring options; Option D, the Python builder over a JSON AST, is what was built. |
| `ABSTRACTS.md` | The target workloads written on paper first, the constructs they needed, and what each trace later corrected. |
| `DESIGN_REVIEW.md` | The reasoning log: every design decision with its date, what it replaced, and why. |

## Technologies

**Runner (Rust, edition 2021).** `clap` for the command line, `serde`/`serde_json` for the
AST and the reports, `toml` for the config file, `sha2` for identities, `xxhash-rust`
(xxh3) for positional keys and the fingerprint, `dgen-data` for the positional payload,
`libc` and the `io-uring` crate for the backends, `anyhow` for errors. Blocking `std::net`
for the coordinator; no tokio, no MPI. Linux interfaces: `io_uring` (with `SQPOLL`,
`DEFER_TASKRUN`, `COOP_TASKRUN`, io-wq worker caps), the kernel AIO calls, glibc POSIX AIO,
`mmap` with `madvise`, `O_DIRECT`, `posix_fadvise`, `mincore`, `/proc/sys/vm/drop_caches`,
`/proc/self/mountstats` for NFS RPC counts, `getrusage`, the rlimits and sysctls a run
checks before its gate.

**Builder (Python ≥ 3.12).** `jsonschema` is the only dependency of the package; `hatchling`
builds it and `uv` locks it. The `formats` extra brings the libraries of the format
classes, `pyarrow`, `h5py`, `numpy`, `crc32c`, `xxhash`, and `dgen-py`, the Python wheel
that reproduces the Rust generator's bytes. PEP 578 audit hooks make the hermetic build.
`aeiou-trace` is standard library only.

**Contract.** JSON Schema draft 2020-12 for the AST and the parameter file; canonical JSON
(sorted keys, no whitespace, shortest-round-trip floats) under SHA-256 for identities.

**Calibration.** `strace -f -ttt -T -yy` of the real applications: PyTorch `DataLoader`,
NumPy, `torch.distributed.checkpoint`, safetensors, DiskANN, FAISS, vLLM with LMCache.

**CI.** GitHub Actions, three jobs: the builder tests with the hermetic rebuild of every
abstract against the committed ASTs and every parameter file against its abstract; the
runner's `cargo test`; and the integration job that diffs the Rust checker against the
reference checker, runs the container datasets to their fingerprints, and holds the metrics
of an strace of the runner to its own dry run.

## Quick start

```
# the runner
cd runner && cargo build --release && cargo test --release && cd ..

# the builder (or: pip install -e 'builder[test,formats]')
cd builder && uv sync --extra test --extra formats && uv run pytest && cd ..

# check, dry-run, generate, run
runner/target/release/aeiou check schema/examples/*.ast.json
runner/target/release/aeiou dry-run schema/examples/train_small_files.ast.json --gpus 8 --seed 1
runner/target/release/aeiou datagen schema/examples/train_small_files.ast.json --root /mnt/sut --param files=4000
runner/target/release/aeiou run schema/examples/train_small_files.ast.json --root /mnt/sut --gpus 8 --seed 1 \
    --param files=4000 --param steps=50 --report-json run.json
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

## Status (2026-10-04)

A working proof of concept, measured so far on ext4 and on a loopback NFS v4.2 mount on
one development box. Built and tested: everything described above, the nine backends, the
multi-host coordinator (two ranks on `localhost`), the limit checks, the cold-start
controls, the JSON report, the locality metrics on both sides with tolerances, the `trace`
node under every backend, the option layers, and the usage-error frame.

Not built, in rough order of interest:

- A run on real hosts (the coordinator's connect window and heartbeat constants have only
  met `localhost`), and the measurements of `PROJECT_BRIEF.md` §6 that need a GPU box: the
  agentic KV-cache load replayed through vLLM (item 21), and tensor-parallel sharding of a
  KV chunk, where `--gpus` is today independent engines rather than one TP group (item 20).
- Backends from the brief's list that are not written: `gds`, `nixl-posix`, `libnfs`; object
  storage through `s3dlio` behind a feature flag (item 17).
- `aeiou-verify`, the offline content verifier that takes a dataset manifest and
  regenerates any byte; `aeiou-fit`, the fitting of parameter files from a trace (each
  capture kit today has its own fitting script).
- Format classes for Arrow IPC, MDS, and Megatron `.bin`/`.idx`, and the Parquet-to-Arrow
  conversion abstract (item 15).
- An `O_DIRECT` checkpoint writer to trace (item 19); a recorded wait time per lane and a
  cap on the file for the `trace` node; `RLIMIT_MEMLOCK` in the limit checks;
  wall-clock-bounded phases (a contract change).

## License

Apache License 2.0. See `LICENSE`.
