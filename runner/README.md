# The runner

The Rust half of aeiou: the binary `aeiou`, which loads, validates, and executes the AST
contract of `../schema/`, and `aeiou-launch`, the ssh loop that starts one rank per host.
Nothing here depends on Python. The manual is [aeiou(1)](../man/aeiou.1.md) and
[aeiou-launch(1)](../man/aeiou-launch.1.md); the reference, every definition the runner
fixes and every choice made in building it, is [REFERENCE.md](REFERENCE.md).

```
cargo build --release
cargo test --release
./target/release/aeiou check ../schema/examples/*.ast.json
./target/release/aeiou dry-run ../schema/examples/train_small_files.ast.json --gpus 8 --seed 1
./target/release/aeiou run ../schema/examples/train_small_files.ast.json --root /mnt/sut --gpus 8 --seed 1 \
    --param files=4000 --param steps=50
```

## What it is

The runner is a deterministic virtual machine for workload abstracts, with I/O backends
behind it. It never authors anything and knows nothing about file formats or applications:
an abstract arrives as JSON, the runner resolves its parameters, computes every actor
instance's operation stream from the seed and the instance's position, and issues the
operations through the API the abstract declares. The same program computes the stream
without I/O (`dry-run`), writes the datasets the abstract needs (`datagen`), and executes
it (`run`), so the fingerprint a run reports is checked against the fingerprint the dry run
predicted, on every system under test.

## Theory of operation

**Load and validate.** The document is decoded into serde types that deny unknown fields
(`ast.rs`), hashed in canonical form byte-identically to the Python reference
(`canon.rs`), and checked against the semantic rules the schema cannot express
(`validate.rs`; CI diffs `aeiou check` against `schema/check.py` over every committed
abstract). Every draw site is the JSON pointer of its node (`sites.rs`), so two builds of
one source agree on every site without coordination.

**Resolve.** Parameters come from the document's defaults, then parameter files in order,
then `--param`, each value keeping the kind of its default (`eval.rs`). From them the
runner resolves every dataset (count, sizes, names, container layout), every namespace,
and every distribution, and judges again the rules whose truth depends on parameter values.
Options come through four layers with provenance (`options.rs`): the command line, the
environment, the TOML file, the default; the identity is the command line's alone.

**Compute.** Every actor instance is a resumable state machine (`vm.rs`): `next` yields
one event at a time, an operation, a control (barrier, channel, compute), or a fork, and
the machine can be parked between any two. Randomness is positional (`rng.rs`): the key of
a draw is `xxh3(actor, site, loop indices)` under the seed, its words a SplitMix64 sequence,
and `consume` goes through a four-round Feistel permutation of the dataset, cycle-walked to
the dataset's size and tested as a bijection at fifty million. Paths are formatted from
patterns at the moment of use (`pattern.rs`); no per-file structure is ever built. The dry
run (`dryrun.rs`) drives the machines in parallel over instances and sums per-operation
hashes into the fingerprint; `metrics.rs` walks the same stream in the round-robin order of
an instance's sub-actors for the order-dependent metrics.

**Execute.** A `Backend` has an issue half and a completion half (`backend.rs`). Under the
blocking backends (`sync`, `sync-direct`, `posix-aio`, `mmap`) every actor and sub-actor
is an operating system thread and a blocking call is simply blocking (`run.rs`): this is the
fidelity reference. Under `io_uring` and `libaio` an event loop per thread parks one machine
per actor and keeps one operation in flight per actor, so thousands of actors share a few
rings (`uring.rs`, `aio.rs`). Every result is checked structurally: the computed count, the
expected errno, the computed entry count of a listing. Writes carry positional content
(`payload.rs`), the same bytes `datagen` would write for that path and offset.

**Coordinate.** Several hosts run the same command line with their rank; rank 0 serves
the coordinator in-process over blocking `std::net` (`coord.rs`), compares every host's
identity document and options block, gates the start, runs two-level barriers, merges the
reports, and sends one verdict. There is no MPI and no async runtime.

**Guard and report.** Before the gate: dataset and namespace manifests compared field by
field (`payload.rs`), an estimate of open files, threads, and mappings against the host's
limits with soft limits raised (`limits.rs`), optional cache dropping and a residency
sample (`cold.rs`). Around the run: task and io-wq worker peaks, `getrusage`, and the mount's
NFS RPC deltas (`counters.rs`). After: latency histograms, per-phase totals, per-step stall,
the fingerprint, the verdict, as text and as one JSON document (`report.rs`).

**Trace nodes.** A `trace` node is an exported `strace` executed literally (`trace.rs`),
one lane per traced task, opens shared by id, gaps between calls as compute; the lanes
observe the two orders the trace fixed, the open table and the per-path order of changing
operations, so they cannot deadlock.

## Technologies

Rust 2021, one workspace, one crate. `clap` (derive, wrapped help) for the command line
with the `--[no-]x` boolean convention; `serde`/`serde_json`; `toml`; `sha2`; `xxhash-rust`
(xxh3) for keys and the fingerprint; `dgen-data` 0.3.0 for the payload stream; `libc` for
the POSIX, AIO, and `mmap` calls; the `io-uring` crate for the rings; `anyhow`. Linux
interfaces: `io_uring` with `IORING_SETUP_SQPOLL`, `IORING_SETUP_ATTACH_WQ`,
`IORING_SETUP_SINGLE_ISSUER` with `DEFER_TASKRUN`, `COOP_TASKRUN`, and
`IORING_REGISTER_IOWQ_MAX_WORKERS`; `io_setup`/`io_submit`/`io_getevents`; glibc
`aio_read`/`aio_write`; `mmap` with `MADV_POPULATE_READ` and `MADV_WILLNEED`; `O_DIRECT`
with 4 KiB alignment; `posix_fadvise`; `mincore`; `/proc/sys/vm/drop_caches`;
`/proc/self/mountstats`; `RLIMIT_NOFILE`, `RLIMIT_NPROC`, `kernel.threads-max`,
`vm.max_map_count`.

## Layout

```
Cargo.toml          the workspace
aeiou/              the crate: src/ (one module per concern, listed in REFERENCE.md §1) and tests/
aeiou-launch        the ssh loop, POSIX sh
REFERENCE.md        the reference: definitions (§2), run semantics (§4), payload and manifest (§5),
                    the coordinator (§6), loopback NFS (§7), the backends (§8, §9), metrics (§10),
                    limits (§11), the JSON report (§12), the trace node (§13), option layers (§14),
                    usage errors (§15)
```

The tests (`cargo test --release`) are golden fingerprints and hash parity with the Python
checker, container layouts by hand, datagen-and-run round trips over every committed
abstract under every backend on a temporary directory, the io_uring knobs, the metrics,
the limit checks, the JSON report through the binary, two-rank runs as threads and as
processes, the option layers, the usage frame, and the manual page against `--help`.

## Pointers

- `../man/aeiou.1.md`, `../man/aeiou-config.5.md`: the manual.
- `REFERENCE.md`: what the runner fixes on top of the contract; change a definition in §2
  and every fingerprint changes.
- `../DESIGN_REVIEW.md` §3.22 onward: why each piece is the way it is.
- `../NAPKIN_MATH.md`: the memory and IOPS arithmetic the design answers to.
