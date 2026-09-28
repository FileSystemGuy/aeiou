# Project Brief — Abstract-Driven I/O Benchmark Runner

Captured 2026-09-24 from the design conversation; revised 2026-09-25 after design review. It
records the original requirements, the decisions made so far, and what is still open. Details
are in:
- `NAPKIN_MATH.md`: DRAM and IOPS estimates, risk register, spikes, and the round-2 decisions (§8)
- `GRAMMAR_OPTIONS.md`: options for extending the abstract language, with a recommendation
- `DESIGN_REVIEW.md`: the 2026-09-25 review, with the reasoning behind the fixes applied here

## 1. Core idea

Many real storage workloads are *partly random, partly scripted*. A random choice decides which of
a set of fixed sequences runs next. Example: PyTorch training with one sample per file on POSIX.
A random file is picked, then `open → read → read → … → close` on that file is completely
fixed.

Benchmarks built from "POSIX ops + a probability per op" can't represent those scripted
sequences. This project describes a workload as an **abstract**: a nested, regex-like structure.
Its random parts are driven by a user-supplied seed, so the entire I/O pattern is planned and
repeatable. It can be viewed as a dependency graph on operation completions, executed by a simple
state machine that scales to any number of simulated GPUs.

The goal is **not** that the storage system can predict the pattern. The goals are:
1. the generated operation sequence is repeatable;
2. it is highly representative of real-world I/O;
3. a simple state machine can scale it up until the storage system under test is saturated.
   The graph and state machine are the same whether 10 or 100 GPUs are simulated.

## 2. Reference workload (training)

- The batch size is 7. Each GPU picks 7 files at random, reads them all into memory, and hands
  them to the GPU (emulated compute time). This repeats until every file has been read.
- When a batch is submitted to the GPU, reading of the next batch starts in the background. The
  next batch is submitted once all of its files have arrived *and* the GPU is ready.
- Every N batches (500 in the original description), all GPUs hit an MPI barrier to exchange
  weights synchronously.
- The only randomness is which files are picked. Everything else is fixed.

## 3. Original grammar concepts (user's)

- **operation**: one POSIX syscall (`open`, `read`, `write`, `lseek`, `close`, `sleep`, …)
- **element**: an operation or a sequence
- **sequence**: an ordered list of elements plus a distribution for its repeat count, e.g.
  `(open, (read)[1..10], close)[1..8192]`
- **selection**: an unordered set of sequences plus a probability distribution over which runs
  next, e.g. `read[80%]`, `write[20%]`
- **consumer**: a selection in which each chosen element is marked and not chosen again until
  the set has been used up

Agreed additions (see `GRAMMAR_OPTIONS.md`): binding, per-GPU replication, explicit loop
indices (`for step in $steps`), `parallel(W)` and bounded ordered/unordered `channel`s as the
concurrency primitives (the PyTorch-style `loader` with prefetch and in-order delivery is sugar
over them, and is finite: it produces exactly `steps` batches), `take`, `barrier(scope)`,
`every N`, and named parameters that can be overridden from the CLI.

The abstract stays POSIX-shaped (`open`, `read`, `write`, `close`, …); it has no `mmap` op.
`mmap`-based loaders (safetensors, Arrow-backed Hugging Face datasets, `np.load(mmap_mode=…)`)
are covered by the `mmap` **backend** (§4), which maps `read(f, off, len)` onto populating that
range of a mapping. This reverses the 2026-09-25 exclusion.

## 4. Requirements

- A Rust **benchmark runner**, highly parallel, using io_uring through the **`io-uring` crate**
  (decided; not C liburing).
- Inputs: an abstract file, plus CLI arguments `--seed`, `--gpus`, and per-run parameters
  (batch size, readahead/transfer size, workers, prefetch, compute time, steps, …).
- It must handle corpora of **50M–100M files** without one data structure per file. Filenames
  follow patterns (e.g. `data_NNNN_of_MMMM`, or hierarchical) and are computed, not stored.
- Tracking "consumed" files: the user asked about a bitmap vs. a fully populated shuffled list.
  The **recommendation is a keyed Feistel permutation**, which needs no memory and has O(1),
  constant-cost draws.
- The target storage is **NFS** (the user is at Hammerspace; pNFS/FlexFiles is relevant).
- **Selectable I/O initiation API** (added 2026-09-28). The same abstract must run unchanged over
  many base-level I/O APIs, chosen on the CLI. Purpose: the CLOSED benchmark uses the API the
  real framework uses (see §5, *Divisions*); the other backends let the same codebase measure
  the **speed of light** of the system under test and give solution implementors and framework
  implementors concrete evidence of which I/O APIs would perform better. A backend is a point on
  three axes:
  - *initiation*: blocking syscall · submit-and-poll (libaio, io_uring) · library call (cuFile,
    NIXL, libnfs) · page fault (mmap);
  - *completion*: return value · kernel completion queue · library callback or status poll ·
    none (page fault);
  - *memory target*: pageable host · pinned host · GPU memory.

  Required backends, `--io-backend`:

  | Backend | Initiation / completion | Notes |
  |---|---|---|
  | `sync` | blocking `pread`/`pwrite` on a thread pool, buffered | fidelity reference: what PyTorch does |
  | `sync-direct` | same, `O_DIRECT` | |
  | `posix-aio` | glibc `aio_read`/`lio_listio` | user-space thread pool inside glibc; included for completeness, expect it to track `sync` |
  | `libaio` | `io_submit`/`io_getevents` | the kernel AIO path; truly async only with `O_DIRECT`; what fio and vendors mean by "AIO" |
  | `io_uring` | `io-uring` crate | feature knobs (SQPOLL, fixed files/buffers, linking) are options, not backends |
  | `mmap` | `mmap` + page touch, or `MADV_POPULATE_READ` / `MADV_WILLNEED` as prefetch variants | how safetensors, Arrow/HF datasets, and llama.cpp load; runs on the thread pool. The abstract's `read(f, off, len)` maps to populating that range |
  | `gds` | cuFile: `cuFileRead` (sync), batch API, stream-ordered | needs a CUDA device on the client. **Must detect and report compat mode** (POSIX bounce buffer fallback), which is the common case on NFS |
  | `nixl-posix` | NIXL with its POSIX plugin (`nixl-sys` Rust bindings) | needs a CUDA device; per-transfer descriptor setup will dominate small reads, which is a valid result |
  | `libnfs` | user-space NFS client | no kernel page/dentry/attribute cache; every LOOKUP goes over the wire; a floor for "client CPU with no kernel in the path" |
  | *deferred:* `s3` | object GET/PUT | not POSIX-shaped; needs `get`/`put` in the abstract's vocabulary. MLPerf Storage is heading there |

  Orthogonal flags: `--buffer pageable|pinned|gpu` and `--cache buffered|direct|dontcache|fadv-dontneed`,
  with a validity table that rejects impossible combinations (buffered `gds`, `direct` `mmap`, …).
  `preadv2` flags, vectored reads, `readahead(2)`/`fadvise` hints, and io_uring features are
  options on a backend, not backends. `splice`/`sendfile`/`copy_file_range` only matter for a
  data-mover abstract and are not planned. SPDK, xNVMe, and NVMe passthrough do not apply to NFS.

  Every backend must report, per run, its own counters (io-wq workers, libaio context depth,
  cuFile compat-mode flag and `cufile_stats`, NIXL backend selected) alongside the common
  `mountstats` RPC counts. fio's engine list is the cross-check for this set, and fio itself is
  used to validate each backend's raw numbers before ours are trusted.

## 5. Decisions so far

| Topic | Decision |
|---|---|
| Multi-host | **Pure-Rust TCP coordinator, not MPI** (decided 2026-09-28, replacing the earlier MPI decision). Same seed on every rank; each rank takes its own subset with no negotiation. Work is split **by global GPU id** over **positions in the Feistel shuffle**. "Rank" is only the host index that selects a GPU id range. Processes are launched with `pdsh` or an ssh loop instead of `mpirun`. |
| Run length | Step-bounded, currently 500 steps (adjustable). `steps` and `sync_every` are separate parameters. |
| Dataset sizing rule | Dataset capacity ≥ **5× the total DRAM of all client nodes**. |
| Page cache | The main concern: Linux gets non-linearly slower at finding pages to evict as memory fills, even with clean pages. This motivates O_DIRECT / io_uring / DONTCACHE. |
| Reproducibility | **Exact** run-to-run reproducibility of each GPU's op stream is required; training-style randomness across runs is not needed. Plan: positional RNG keyed on `(seed, gpu, site, loop indices)` with no per-actor counters; consumer position by formula `g + G·(b·B + j)` with `drop_last` epochs; finite producers; an **order-independent** fingerprint (sum of per-op hashes, summed by the coordinator), since issue order within a GPU's workers is timing-dependent. `--dry-run` computes the fingerprint and can print any GPU's op stream for any step range. |
| Dataset identity | The dataset definition has its own seed, separate from `--seed`. `datagen` writes a manifest at the corpus root (pattern, count, size distribution, dataset seed, generator version); the runner validates the abstract against it before starting. |
| Backend order | The blocking thread pool (one OS thread per worker actor) is the **fidelity reference** and is built first. io_uring is the scaling lever. Both sit behind one trait. Spike 1 tests the hypothesis that O_DIRECT reads on NFS complete asynchronously and so keep the io-wq worker count bounded, while opens/closes still punt. |
| Measurement | Per-step stall time per GPU (not per-op logs); histograms bucketed by step range so steady state is selected after the run; `--dry-run` prints total bytes per host against host DRAM. Read sink buffers are sized larger than L3 so copies cost what a real loader's do. |
| Divisions and comparison policy | **Decided 2026-09-28.** The backend is part of the application, so the line between "must be the same" and "may vary" is the backend's call boundary: the backend implementation is benchmark code; everything it calls (libc, kernel, NFS client, libnfs, cuFile, drivers, network, server) is the solution under test. **CLOSED** fixes the application-level I/O interface to what the real framework uses: for PyTorch that is buffered POSIX with PyTorch's concurrency structure, i.e. the `sync` backend. The other backends are for speed-of-light measurement of the SUT and for advising implementors; results from them are reported separately and never compared against CLOSED results as if they were storage differences. Compare storage solutions with the backend fixed; compare backends with the storage fixed. **The boundary is stated as an interposition test (decided 2026-09-28):** the solution may do anything that a dynamic-linker shim (`LD_PRELOAD`) placed under the *unmodified* application could do, provided the application's dependency graph of operations, the buffers they target, their data, and their semantics are preserved. What a shim cannot do is therefore application: the concurrency structure (workers, prefetch depth, in-order delivery), the memory target (host vs. GPU), and the choice of the next file. What a shim can do is solution: user-space clients (libnfs, vendor clients), O_DIRECT or io_uring or libaio underneath buffered calls, kernel and mount configuration, transport, and the server. A vendor-supplied shim under CLOSED is legal by this test, which is why the two requirements below (data verification, seed privacy) exist. Below the line, an optimization counts only if the real workload would get it too: no benefit from the synthetic data, no cache warmth from a previous run, no re-reads within a run, no relaxed semantics the customer would not run. RPC counts, client CPU, and client DRAM are reported as audit data and cost, not as a score. |
| Data verification | **Required (2026-09-28).** Because a shim or client may return anything, the application must be able to check what it read. `datagen` writes verifiable content: every 4 KiB block starts with a small header `(magic, dataset seed, file id, block offset)` followed by non-dedupable PRNG fill derived from the same tuple. The runner verifies headers on a sampled fraction of reads (`--verify-sample`, default 1 in 64 blocks; 100% in a `--verify` run) at negligible CPU cost, and counts mismatches as run failures. Checkpoint-write abstracts are paired with a read-back phase that verifies the written content after `fsync`/`close`, which is also the durability check. Verification reads the buffer the backend delivered, so it works for every backend, including GPU-memory ones via a device-side check or a sampled copy-back. |
| Seed privacy | **Required (2026-09-28).** The run seed, the abstract, and the resulting file order are application-private. The solution (anything below the interposition line) may not use knowledge of them, for example to prefetch the next file. A submission review may re-run with a fresh seed; the results must match within noise. The dataset seed is not secret (it is in the manifest) because it determines content, not order. |
| Coordinator | Star topology over plain TCP with blocking `std::net` on one coordinator thread; length-prefixed `postcard` messages (`Hello` with a config hash, `Ready`/`Start`, `Arrive`/`Release`, `Stats`, `Stop`, `Heartbeat`). No tokio, no tonic/gRPC. Sits behind a `Coordinator` trait; single-host runs use an in-process implementation. See `NAPKIN_MATH.md` §8.A. |
| Deployment | Bare Linux on the client nodes, **no containers**. |
| I/O crate | `io-uring` (Rust). |
| A/B testing | Agreed. Backend × cache mode × io_uring features × NFS mount options (`NAPKIN_MATH.md` §8.5). The key metric is client CPU per op. |
| Grammar | Three layers: authoring language, the AST contract, the Rust VM. The AST (serde YAML/JSON) is the contract and the only thing the runner executes; the WG publishes ASTs and their hashes, submitters run those. Leading candidate for authoring (2026-09-28): **Option D**, a Python builder package that emits the AST, with source→AST reproducibility enforced by CI (build twice, compare) and by the runner's validator. Python stays on the authoring station, never on client nodes. **User has not yet chosen.** |

## 6. Open items / next steps

1. **Write paper abstracts** for the target workloads, derived from `strace` of the real
   applications: small-file training, large-sample training, checkpoint write, checkpoint
   restore, and (added 2026-09-28) **vector-database search (DiskANN-style and IVF), index build,
   and KV-cache serving**. Sketches in `GRAMMAR_OPTIONS.md` §5.5.
2. **User to choose an authoring option** (A/B/C/D in `GRAMMAR_OPTIONS.md`; D is the leading
   candidate). Then: publish the AST JSON Schema, write the builder package and the
   `abstract-build --hermetic` harness, and add the build-twice check to CI.
3. Build the VM with `--dry-run` and the fingerprint; test against ext4 and a loopback NFS mount
   on WSL2 (see §7). Golden-test the fingerprint in CI.
4. **Spike 1:** blocking thread pool first, then io_uring; buffered vs. O_DIRECT, for
   `open → read → close` against the real NFS target, measuring `iou-wrk` count separately for
   open-heavy and read-heavy phases. It decides the I/O backend and the cache strategy.
5. Spike 2: correctness and speed of the Feistel permutation at 50M/100M.
6. Spike 3: client dentry/inode slab growth and NFS op mix when touching 50M files.
7. Check `RWF_DONTCACHE` support in the NFS client on the target kernels.
8. Build `datagen` mode, driven by the same filename pattern, size distribution, and dataset seed
   as the abstract, writing **verifiable, non-dedupable** data (per-block headers, §5) and a
   manifest at the corpus root.
9. Startup checks: `kernel.io_uring_disabled`, `RLIMIT_MEMLOCK`, and `RLIMIT_NOFILE` computed
   from G, W, and the abstract.
10. Coordinator protocol and launch script (`pdsh`/ssh loop); test it on WSL2 with several
    ranks on `localhost`.
11. ~~Comparison policy~~ Decided 2026-09-28, including the interposition test, data
    verification, and seed privacy; see §5.
12. Define the block header format and the PRNG fill for `datagen` and the verifier, and add
    `--verify-sample` / `--verify` to the CLI. Decide how GPU-memory backends verify (device-side
    kernel vs. sampled copy-back).
13. **Data-dependent workloads (2026-09-28).** Add to the semantic model: distributions over ids
    and offsets (`zipf`, `hotset`, random `offset` in `read`), recency references
    (`recent(site, d)`, positional, no stored history), and positional names for
    write-then-read. Cross-actor read-after-write is limited to barrier-separated phases or a
    statistical hit model; state this limit in each such workload's documentation.
    `GRAMMAR_OPTIONS.md` §5.
14. **Locality-metrics check and replay mode (2026-09-28).** `--dry-run --metrics` computes
    reuse-distance, sequential run-length, popularity skew, request-size, dependency depth, and
    read/write mix on the abstract's stream; a trace tool computes the same from a real trace; an
    abstract is accepted for a workload class only when they match within tolerances. A `replay`
    AST node holds a literal captured sequence for small-scale calibration; never CLOSED.

## 7. Environment

The development machine is WSL2 (kernel 6.18, 20 cores, 31 GB RAM), with no realistic NFS
target. A loopback NFS mount (`nfs-kernel-server` exporting a tmpfs directory, mounted from
`localhost`) exercises the real NFS client code paths for correctness. All performance work must
run on real Linux client nodes.
