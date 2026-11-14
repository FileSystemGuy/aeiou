# Project Brief — Abstract-Driven I/O Benchmark Runner

Captured 2026-09-24 from the design conversation; revised 2026-09-25 after design review. It
records the original requirements, the decisions made so far, and what is still open. Details
are in:
- `NAPKIN_MATH.md`: DRAM and IOPS estimates, risk register, spikes, and the round-2 decisions (§8)
- `GRAMMAR_OPTIONS.md`: options for extending the abstract language, with a recommendation
- `DESIGN_REVIEW.md`: the 2026-09-25 review, with the reasoning behind the fixes applied here
- `ABSTRACTS.md`: paper abstracts for the eight target workloads (2026-09-29) and the constructs they surfaced
- `schema/`: the AST contract (JSON Schema v0.1, 2026-09-30), its canonical form and validator rules, and example ASTs
- `builder/`: the Python builder (`aeiou`), the workloads as authoring scripts, the hermetic build harness

The tool is built for the MLPerf Storage working group but is meant to be usable by anyone who
needs a reproducible, abstract-driven I/O workload; §8 collects what is specific to the WG so
the rest can be read as general.

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
- **Multi-sample container formats** (added 2026-09-29): Parquet, HDF5, TFRecord, Arrow IPC,
  WebDataset tar, MDS shards, Megatron token files. The unit of shuffling is the sample; the
  container is found by formula; the format's reader protocol comes from a built-in format class.
  See §5 and `GRAMMAR_OPTIONS.md` §6.
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

  Required backends, ~~`--io-backend`~~ (since 2026-10-09 ~~`--io-api`~~ `--posix` and `--cache` for the POSIX names, `--s3 blocking|async` for the S3 names, `DESIGN_REVIEW.md` §3.65; a `-direct` name below is the API with `--cache direct`):

  | Backend | Initiation / completion | Notes |
  |---|---|---|
  | `sync` | blocking `pread`/`pwrite` on a thread pool, buffered | fidelity reference: what PyTorch does |
  | `sync-direct` | same, `O_DIRECT` | |
  | `posix-aio` | glibc `aio_read`/`lio_listio` | user-space thread pool inside glibc; included for completeness, expect it to track `sync`. **Built 2026-10-01** (`runner/REFERENCE.md` §9, also `posix-aio-direct`): it tracks `sync` in RPCs and costs 2.5 to 3 times the CPU on the loopback |
  | `libaio` | `io_submit`/`io_getevents` | the kernel AIO path; truly async only with `O_DIRECT`; what fio and vendors mean by "AIO". **Built 2026-10-01** as `libaio` and `libaio-direct` on the event loop (the system calls directly, `--aio-depth`; `DESIGN_REVIEW.md` §3.35); the report gives the time inside `io_submit` |
  | `io_uring` | `io-uring` crate | feature knobs (SQPOLL, fixed files/buffers, linking) are options, not backends; **built 2026-10-01** (`runner/REFERENCE.md` §8); ~~the knobs not yet~~ the ring and io-wq knobs the same day (`--iowq-max-workers`, `--sqpoll`, `--sqpoll-shared`, `--defer-taskrun`, `--coop-taskrun`; `DESIGN_REVIEW.md` §3.34), fixed files/buffers and linking not |
  | `mmap` | `mmap` + page touch, or `MADV_POPULATE_READ` / `MADV_WILLNEED` as prefetch variants | how safetensors, Arrow/HF datasets, and llama.cpp load; runs on the thread pool. The abstract's `read(f, off, len)` maps to populating that range. **Built 2026-10-01**: the file is mapped on first read and unmapped at `close` (one mapping per open, shared by the sub-actors that inherit the descriptor, since the same day: `DESIGN_REVIEW.md` §3.36 Revised), ~~a read is a copy out of the mapping,~~ a read touches one byte of every page of its range (`--mmap-consume touch`, the default since later the same day; `copy` is the option), and `--mmap-mode fault\|populate\|willneed` picks the prefetch before it. The map, unmap, flush, and fault costs are counted against the backend (`DESIGN_REVIEW.md` §3.35, Revised) |
  | `gds` | cuFile: `cuFileRead` (sync), batch API, stream-ordered | needs a CUDA device on the client. **Must detect and report compat mode** (POSIX bounce buffer fallback), which is the common case on NFS |
  | `nixl-posix` | NIXL with its POSIX plugin (`nixl-sys` Rust bindings) | needs a CUDA device; per-transfer descriptor setup will dominate small reads, which is a valid result |
  | `libnfs` | user-space NFS client | no kernel page/dentry/attribute cache; every LOOKUP goes over the wire; a floor for "client CPU with no kernel in the path" |
  | *deferred:* `s3`, `az`, `gs` | object GET/PUT | ~~not POSIX-shaped; needs `get`/`put` in the abstract's vocabulary.~~ Preliminary opinion 2026-10-01 (`DESIGN_REVIEW.md` §3.32): the abstract stays POSIX-shaped and the backend maps (`open, read…, close` to one streaming GET or ranged GETs, sequential writes to a multipart upload), refusing what has no mapping at `check` time; via the `s3dlio` crate behind a cargo feature (§6 item 17). MLPerf Storage is heading there. *2026-10-09:* bound per namespace, with the protocol family and the API as separate axes (§6 item 25, `DESIGN_REVIEW.md` §3.65, decided 2026-10-09) |

  Orthogonal flags: `--buffer pageable|pinned|gpu` and `--cache buffered|direct|dontcache|fadv-dontneed`,
  with a validity table that rejects impossible combinations (buffered `gds`, `direct` `mmap`, …).
  (*2026-10-09:* built only as the `-direct` names so far; separating them again, with the
  protocol family, transport, and an endpoint per namespace, is §6 item 25, decided the same day.)
  `preadv2` flags, vectored reads, `readahead(2)`/`fadvise` hints, and io_uring features are
  options on a backend, not backends. `splice`/`sendfile`/`copy_file_range` only matter for a
  data-mover abstract and are not planned. SPDK, xNVMe, and NVMe passthrough do not apply to NFS.

  Every backend must report, per run, its own counters (io-wq workers, libaio context depth,
  cuFile compat-mode flag and `cufile_stats`, NIXL backend selected) alongside the common
  `mountstats` RPC counts. fio's engine list is the cross-check for this set, and fio itself is
  used to validate each backend's raw numbers before ours are trusted. (The common set is in
  the report since 2026-10-01: task, io-wq worker, and `SQPOLL` thread peaks, CPU, RSS, the mount's options and NFS byte
  and RPC deltas; `runner/REFERENCE.md` §4, `DESIGN_REVIEW.md` §3.30.)

**Names (2026-10-10).** The abstracts were renamed for their directories under `workloads/`
(`DESIGN_REVIEW.md` §3.66, which has the table: `train_small_files` is `file_per_sample`,
`kv_cache_serving` is `lmcache_local_disk`, …). This document keeps the names it was written
with; paths in it are the new ones.

## 5. Decisions so far

| Topic | Decision |
|---|---|
| Multi-host | **Pure-Rust TCP coordinator, not MPI** (decided 2026-09-28, replacing the earlier MPI decision). Same seed on every rank; each rank takes its own subset with no negotiation. Work is split **by global GPU id** over **positions in the Feistel shuffle**. "Rank" is only the host index that selects a GPU id range. Processes are launched with `pdsh` or an ssh loop instead of `mpirun`. |
| Run length | Step-bounded, currently 500 steps (adjustable). `steps` and `sync_every` are separate parameters. *(2026-10-14: `sync_every` is 1 by default, a barrier every step as synchronous data-parallel training has; `DESIGN_REVIEW.md` §3.71 point 4.)* |
| Dataset sizing rule | Dataset capacity ≥ **5× the total DRAM of all client nodes**. *(2026-10-10, `DESIGN_REVIEW.md` §3.67: this is the WG's run rule, listed in §8; its reason is that the benchmark's clients are not production nodes, so client count must not buy cache. aeiou keeps dataset size a parameter of the case, and small datasets that fit in cache are legitimate use cases.)* *(2026-10-21, `DESIGN_REVIEW.md` §3.78 point 10: aeiou judges the rule at the gate when a rules file carries it as a constraint, since only aeiou knows every client host's memory; the equation stays the WG's, aeiou hard-codes none. Not yet built.)* |
| Use cases, benchmark rules, and the lock | **Decided 2026-10-21 and 2026-10-22** (`DESIGN_REVIEW.md` §3.78; not yet built, §6 item 33). Four layers that only narrow: the abstract (the application's and library's values), the use case (a job on an accelerator: `unet3d-h100`), the benchmark rules (a file that names no parameter: it freezes by state and holds constraints over the runner's `fact.` and `result.` names), and the submission (the scale and the system under test's settings, never shipped). Every value is locked, default, or required; fragments of one layer combine by agreement, never last-one-wins. Resolving them writes a JSON **lock**, the one file a run names (`--case`, `--case-file`), pinning its abstract by its path under `workloads/` and its `ast_sha256`; it has a file hash and a **workload id** (its canonical resolved values), and results are compared by the id. The fingerprint is the op multiset and does not cover compute. Made by `aeiou-params lock`, remade by `aeiou-build` for the committed cases of what it rebuilt; CI refuses a case with no lock or a stale one, and the runner an abstract whose hash is not its lock's. A run that does not pass a result constraint completes and exits 3. `st_blksize` is read from the storage at the gate (`{"from": "st_blksize"}`), the system under test's to report. |
| Page cache | The main concern: Linux gets non-linearly slower at finding pages to evict as memory fills, even with clean pages. This motivates O_DIRECT / io_uring / DONTCACHE. |
| Reproducibility | **Exact** run-to-run reproducibility of each GPU's op stream is required; training-style randomness across runs is not needed. Plan: positional RNG keyed on `(seed, gpu, site, loop indices)` with no per-actor counters; consumer position by formula `g + G·(b·B + j)` with `drop_last` epochs; finite producers; an **order-independent** fingerprint (sum of per-op hashes, summed by the coordinator), since issue order within a GPU's workers is timing-dependent. `--dry-run` computes the fingerprint and can print any GPU's op stream for any step range. |
| Dataset identity | The dataset definition has its own seed, separate from `--seed`. `datagen` writes a manifest at the corpus root (pattern, count, size distribution, dataset seed, generator version); the runner validates the abstract against it before starting. **Extended 2026-09-30 (reasoning in `DESIGN_REVIEW.md` §3.21; contract in `schema/README.md` §6):** the manifest is `.aeiou-dataset.json` at each dataset's root, and its normative part is the *resolved dataset definition*, the dataset's `datasets` entry with every referenced parameter substituted, in canonical JSON, plus what the abstract does not know (payload generator and version, dedupe/compression settings, block size, format-class writer version). `aeiou run` resolves the same closure from its own abstract and the parameters in effect and requires an exact match; it never compares the whole parameter set (most parameters do not shape the dataset) nor the abstract hash (one corpus legitimately serves several abstracts). The abstract's hash and name and the full parameter values at datagen time are recorded as provenance, not compared. The manifest's canonical hash is the dataset id, printed with every result next to the AST hash and the parameters in effect. The seed alone is not the identity: it fixes the bytes given a definition, not the definition. |
| Backend order | The blocking thread pool (one OS thread per worker actor) is the **fidelity reference** and is built first. io_uring is the scaling lever. Both sit behind one trait. Spike 1 tests the hypothesis that O_DIRECT reads on NFS complete asynchronously and so keep the io-wq worker count bounded, while opens/closes still punt. **Built 2026-09-30:** `sync` and `sync-direct` as `aeiou run` (`runner/REFERENCE.md` §4, `DESIGN_REVIEW.md` §3.23); ~~the asynchronous backends need a resumable VM, which comes with `io_uring`.~~ **Built 2026-10-01:** the resumable VM and `io_uring` / `io_uring-direct` as one event loop per thread (`runner/REFERENCE.md` §8, `DESIGN_REVIEW.md` §3.29); the Spike 1 measurement itself still waits for the real target. |
| Measurement | Per-step stall time per GPU (not per-op logs); histograms bucketed by step range so steady state is selected after the run; `--dry-run` prints total bytes per host against host DRAM. Read sink buffers are sized larger than L3 so copies cost what a real loader's do. |
| Divisions and comparison policy | **Decided 2026-09-28.** The backend is part of the application, so the line between "must be the same" and "may vary" is the backend's call boundary: the backend implementation is benchmark code; everything it calls (libc, kernel, NFS client, libnfs, cuFile, drivers, network, server) is the solution under test. **CLOSED** (an MLPerf division; §8) fixes the application-level I/O interface to what the real framework uses: for PyTorch that is buffered POSIX with PyTorch's concurrency structure, i.e. the `sync` backend. The other backends are for speed-of-light measurement of the SUT and for advising implementors; results from them are reported separately and never compared against CLOSED results as if they were storage differences. Compare storage solutions with the backend fixed; compare backends with the storage fixed. **Refined 2026-10-01** (`DESIGN_REVIEW.md` §3.48): the user's definition is that CLOSED means every system under test sees exactly the same operation sequence, which is what makes results comparable, and OPEN is what a stated change to the application buys, comparable with nothing. So the interface is fixed per abstract, not per framework: the abstract declares the backend its traced application uses (contract 0.3; `sync` when it declares none, `mmap` for `model_load`), `aeiou run` defaults to it, and a run under any other backend says so and is not comparable. Buffered and `O_DIRECT` are two workloads here; MLPerf Storage v3.0 let a submitter choose either and stay CLOSED, which this work does not repeat. **The boundary is stated as an interposition test (decided 2026-09-28):** the solution may do anything that a dynamic-linker shim (`LD_PRELOAD`) placed under the *unmodified* application could do, provided the application's dependency graph of operations, the buffers they target, their data, and their semantics are preserved. What a shim cannot do is therefore application: the concurrency structure (workers, prefetch depth, in-order delivery), the memory target (host vs. GPU), and the choice of the next file. What a shim can do is solution: user-space clients (libnfs, vendor clients), O_DIRECT or io_uring or libaio underneath buffered calls, kernel and mount configuration, transport, and the server. A vendor-supplied shim under CLOSED is legal by this test, which is why the two requirements below (data verification, seed privacy) exist. Below the line, an optimization counts only if the real workload would get it too: no benefit from the synthetic data, no cache warmth from a previous run, no re-reads within a run, no relaxed semantics the customer would not run. RPC counts, client CPU, and client DRAM are reported as audit data and cost, not as a score. |
| Data verification | ~~**Required (2026-09-28).** Because a shim or client may return anything, the application must be able to check what it read. `datagen` writes verifiable content: every 4 KiB block starts with a small header `(magic, dataset seed, file id, block offset)` followed by non-dedupable PRNG fill derived from the same tuple. The runner verifies headers on a sampled fraction of reads (`--verify-sample`, default 1 in 64 blocks; 100% in a `--verify` run) at negligible CPU cost, and counts mismatches as run failures.~~ **Revised 2026-09-29: data is reproducible, not self-describing.** A per-block header makes every block unique and defeats any dedupe control, so there are no headers. Instead the expected bytes at `(dataset seed, file id, offset)` are a pure function that a verifier can regenerate, at block granularity, with dedupe and compression ratios as generator parameters (§6 item 12). The **runner does structural checks only**: byte counts, sizes, short reads, and that index bytes it reads (footers, chunk indexes) match the computed layout. **Content verification is a separate Python `verify` tool** next to `datagen`, run after generation and by reviewers on request, never in a scored run. Rationale: the realistic failure is a wrong corpus (old generator, partial regeneration, wrong seed, wrong dedupe setting), which invalidates the result through data reduction; a fabricating shim is fraud and is handled by rules and review, and a warm cache from a previous run has correct content and is not detectable by any content check (it is handled by fresh seeds and dataset sizing). Checkpoint read-back remains as a workload phase (the restore shape), not a content check. The benchmark never interprets sample data; it moves the bytes into the destination memory (copy, or page-fault under `mmap`) and stops. Reasoning in `DESIGN_REVIEW.md` §3.16. |
| Multi-sample containers | **Decided 2026-09-29.** Datasets are sample spaces, not file lists. The Feistel shuffle runs over **sample ids**; one sample per file is the special case. A dataset declares its **access mode**: `map` (Feistel over samples, positions `g + G·(b·B + j)`, `drop_last` over samples; what map-style PyTorch does over HDF5, Arrow, MDS, Megatron token files) or `stream` (Feistel over *shards* with the same sharding formula, sequential reads within a shard, an in-memory shuffle buffer that produces no I/O; what tf.data, WebDataset, HF streaming and Ray Data do over TFRecord, Parquet, tar). Random sample access inside Parquet or TFRecord is refused by the validator because no production reader does it. Container layout is computed by formula: fixed samples per file, sizes from the dataset seed, per-file prefix sums computed once per `open` and held while open (O(open files), so the per-file invariant survives). `GRAMMAR_OPTIONS.md` §6. |
| Dataset shape | **Decided 2026-10-13** (contract 0.8, `DESIGN_REVIEW.md` §3.72). One shape for every dataset, read as a hierarchy: a unit of storage is a file or an object (`protocol`); `layout` says how it holds the items: `whole` (one), `chunked` (part of one), `compound` (several variably sized, framed by the format's `framing`, formerly `format.layout`), or `array` (fixed-size slots in one file; formerly the `regions` kind). `access` is required: `map` or `stream` (consumed from the shuffle) or `indexed` (looked up by id, never consumed; every `array`). A consumed dataset's directories are `{id div D}` or none, so a subset of it is whole directories; `mod` stays for other uses. |
| Format classes | **Decided 2026-09-29.** Each container format is a built-in library with **two contracts**. The *format* contract carries only what a trace of the reader library shows regardless of application: the layout writer for `datagen`, locate formulas (sample → file, unit, offset, length), the reader's fixed protocol emitted as ordinary POSIX nodes (Parquet: 8 bytes at EOF−8 then the footer; HDF5: superblock and chunk index; TFRecord: none), the access modes it supports, and compute slots for decode and index parsing. The *loader* contract stays with the abstract: access mode, interleave, prefetch, column projection. The class is tied to a reader library and version (`parquet(reader="pyarrow")`) and is validated against a trace like any abstract. It lives in the Python builder; **the Rust runner stays POSIX-only and format-ignorant**, and index bytes are read as I/O but not parsed. `GRAMMAR_OPTIONS.md` §6. |
| Payload generator | **Decided 2026-09-29; check resolved the same day.** `datagen` is the term for writing the synthetic corpus (as in MLPerf Storage). Use **dgen-py / the `dgen-data` Rust crate** (Russ Fellows, MIT or Apache-2.0) as the payload engine behind our own positional wrapper. Investigation of `dgen-data` 0.3.0 source plus a test of the wheel (`DESIGN_REVIEW.md` §3.17): there is no seek API, but content is positional by construction. Block `i` (1 MiB) is the Xoshiro256++ keystream seeded with `seed + (i mod unique_blocks)`, with the last `(N−1)/N` of the block zero-filled for compression ratio N, so block `i` of a stream equals block 0 of a 1 MiB generator seeded `seed + (i mod unique_blocks)`; verified byte-for-byte, including the dedupe case. Consequences: (1) **dedupe is scoped to one generator's stream**, so corpus-wide dedupe must be our layer: the seed of a file (or of a 1 MiB block of a large file) is `hash(dataset seed, index mod (total / D))`, so that D units share content; a small-file corpus otherwise has a dedupe ratio of 1 whatever is requested. (2) Ratios that do not divide 1 MiB (3, 5) are off by one byte at the random/zero boundary under seed arithmetic; exact for 2, 4, 8. (3) Compression is bimodal per 4 KiB (pure keystream or pure zeros), the DLIO convention; stated in the manifest. (4) The fill algorithm changed once (back-references → zero fill, January 2026), so the manifest pins the `dgen-data` version and the verifier uses the same one. **Revised 2026-10-11 (`DESIGN_REVIEW.md` §3.70, decided by the user): ~~dgen-data~~ our own SFC64, `aeiou-positional/3`.** The zeros sat at the end of each 1 MiB block, so a file below `1/C` MiB did not compress at all (a 110 KB image at ratio 2 measured 1.00) and storage that compresses in small units saw pure-noise and pure-zero pieces. Now every 4 KiB is `⌈4096/C⌉` stream bytes then `0x5A`; the stream is SFC64, numpy's bit generator in Python and a few lines in Rust, so neither writer depends on dgen (both faster than with it: Rust 22 GB/s at 16 writers, was 11.6). (5) Always seed; the library's default is unseeded. The Rust crate is called directly from `datagen` and the verifier. **Upstream request:** a public `fill_block(seed, block_index, ratios)` or `seek(offset)` would remove the seed-arithmetic workaround; the author and the user are both on the MLPerf Storage leadership team, and the request will be made once this code shows it is what the WG needs. |
| Seed privacy | **Required (2026-09-28).** The run seed, the abstract, and the resulting file order are application-private. The solution (anything below the interposition line) may not use knowledge of them, for example to prefetch the next file. A submission review may re-run with a fresh seed; the results must match within noise. The dataset seed is not secret (it is in the manifest) because it determines content, not order. |
| Coordinator | Star topology over plain TCP with blocking `std::net` (~~on one coordinator thread~~ a reader thread per socket, built 2026-09-30); length-prefixed ~~`postcard`~~ JSON messages (`Hello` with a config hash, `Ready`/`Start`, `Arrive`/`Leave`/`Release`, `Report`/`Result`, `Stop`, `Heartbeat`). No tokio, no tonic/gRPC. Sits behind a `Coordinator` trait; single-host runs use an in-process implementation; rank 0 runs the server in-process and is the only host that empties namespace roots and writes manifests. See `NAPKIN_MATH.md` §8.A, `runner/REFERENCE.md` §6, `DESIGN_REVIEW.md` §3.25. |
| Deployment | Bare Linux on the client nodes, **no containers**. |
| I/O crate | `io-uring` (Rust). |
| Access layers | **Decided 2026-10-09** (`DESIGN_REVIEW.md` §3.65, §6 item 25). Below the format classes, separate axes each owned by the abstract or the solution by the interposition test: the protocol family (POSIX or object) and the endpoint per namespace, mixed in one run; the API declared by the abstract; the run-wide cache mode (`--cache`, by default `per-open`, the abstract's flags as written) as an override, ~~the nine backend names its aliases on the command line~~ no combined names (removed the same day); transport (TCP or RDMA) the solution's, under a kernel or a user-space client alike when the application's calls are unchanged; the buffer axis with GDS and `nixl-posix`. Contract 0.6 replaces `backend` and regenerates every AST. Object namespaces map ops by a table, ~~V16~~ V18 refusing what has none. ~~`s3dlio` is one engine on these axes;~~ The object engine is built on Apache `object_store` behind the cargo feature `object` (decided 2026-10-09, §3.65: 3–4× less client CPU per small request than `s3dlio`'s AWS SDK path); the runner's POSIX path never goes through it. |
| A/B testing | Agreed. Backend × cache mode × io_uring features × NFS mount options (`NAPKIN_MATH.md` §8.5). The key metric is client CPU per op. |
| Checkpoint restore inputs | **Decided 2026-09-30** (`DESIGN_REVIEW.md` §3.24). The restore reads the files a previous checkpoint-write run created, not a dataset: `ckpt_restore` declares its namespaces `input` (`schema/README.md` V14), the write run leaves `.aeiou-namespace.json` at the namespace root, and the restore run requires it, may not modify the namespace, reports the write-to-read gap, and counts reads served from the host that wrote them. `--rank-rotate k` runs the read on a rotated rank-to-host mapping so every host reads what another wrote; the per-object writer record in the manifest makes the warm-read count exact. The read-back phase of `ckpt_write_dcp` is off by default (`readback`). |
| Grammar | Three layers: authoring language, the AST contract, the Rust VM. The AST (JSON; ~~serde YAML/JSON~~ YAML dropped 2026-09-30, `DESIGN_REVIEW.md` §3.20) is the contract and the only thing the runner executes; the WG publishes ASTs and their hashes, submitters run those (WG process; §8). Leading candidate for authoring (2026-09-28): **Option D**, a Python builder package that emits the AST, with source→AST reproducibility enforced by CI (build twice, compare) and by the runner's validator. Python stays on the authoring station, never on client nodes. ~~**User has not yet chosen.**~~ **Decided 2026-09-30: Option D.** The AST JSON Schema is `schema/abstract-ast.schema.json` (v0.1, draft 2020-12), with the canonical form and the validator's semantic rules in `schema/README.md` and the first abstracts in AST form under `schema/examples/`. The nine constructs of `ABSTRACTS.md` §9 were accepted the same day and are in the schema. Next: the builder package, `aeiou-build --hermetic`, and the build-twice CI check. |

## 6. Open items / next steps

**Order of work (2026-10-09, the user's).** The access layers come first (item 25,
`DESIGN_REVIEW.md` §3.65): (1) ~~decide the axes, the per-namespace binding, the object mapping,
and the contract change;~~ decided 2026-10-09; (2) refactor the runner to those axes with no
change of behaviour, with contract 0.6 and mixed protocols (the two axes, the handle, and
contract 0.6 built 2026-10-09, and the protocol and `--endpoint` per dataset and namespace the same day; the mount counters of several mounts not yet);
(3) ~~`s3dlio` as an engine over `file://` against `sync` (item 17)~~ the object engine on `object_store`, run against an S3 server on the development box (item 17; decided 2026-10-09, `DESIGN_REVIEW.md` §3.65). The KV work (the AgentX
conversation reads, item 21; the `O_DIRECT` rerun; reference-configuration holds) goes on
beside it when the GPU is free, then datagen resume (item 23). The items below keep their
numbers; the order is this note's.

1. **Write paper abstracts** for the target workloads, derived from `strace` of the real
   applications: small-file training, large-sample training, checkpoint write, checkpoint
   restore, and (added 2026-09-28) **vector-database search (DiskANN-style and IVF), index build,
   and KV-cache serving**. Sketches in `GRAMMAR_OPTIONS.md` §5.5.
   **Drafted 2026-09-29 in `ABSTRACTS.md`**, from how the applications are built; no trace has
   been captured yet, so every skeleton detail is tagged `[verify]` and every quantity
   `[measure]`, with a capture plan in its §11. The exercise surfaced nine construct proposals
   (`ABSTRACTS.md` §9: actor-id conditionals, `until_eof` semantics, `expect` errnos, offset
   expressions, `x @ i` and chains replacing `recent`, `regions`, `namespace`, `phase`/`readdir`,
   distribution-selecting `when`); ~~they need a decision before the AST schema (item 2) is
   written~~ **all accepted 2026-09-30** (`ABSTRACTS.md` §9; §9.6 with the naive slot layout,
   §9.7 with the rule that object sizes are computed from the writes, never observed; reasoning
   in `DESIGN_REVIEW.md` §3.18). Remaining: capture the traces and fill the slots.
2. ~~**User to choose an authoring option** (A/B/C/D in `GRAMMAR_OPTIONS.md`; D is the leading
   candidate).~~ **Option D chosen 2026-09-30.** ~~Then: publish the AST JSON Schema,~~ Schema
   v0.1 drafted the same day (`schema/`), with the small-file training, checkpoint-write, and KV-cache
   abstracts as the first examples and a `check.py` that validates them against the schema and
   the semantic rules. ~~Still to do: the remaining five abstracts in AST form (they will come out of the
   builder), the builder package, the `aeiou-build --hermetic` harness, and the build-twice
   check in CI.~~ **Done 2026-09-30** (`builder/`, `DESIGN_REVIEW.md` §3.19): the `aeiou`
   package, all of `ABSTRACTS.md` §1–§8 as authoring scripts (`builder/abstracts/`, nine
   workloads since §4 is two), the generated ASTs in `schema/examples/` (the three hand-written
   ones regenerate hash-identical), `aeiou-build --hermetic --twice --check`, and a CI job
   that rebuilds every abstract hermetically twice and fails on drift from the committed ASTs.
   Still to do here: the format-class reader protocols (item 15) ~~and the parameter-file
   split~~ (**the split done 2026-09-30**: `schema/README.md` §8, `DESIGN_REVIEW.md` §3.27,
   `aeiou-params`, `aeiou run --params-file FILE`).
3. Build the VM with `--dry-run` and the fingerprint; test against ext4 and a loopback NFS mount
   on WSL2 (see §7). Golden-test the fingerprint in CI. **`aeiou dry-run` done 2026-09-30**
   (`runner/`, `DESIGN_REVIEW.md` §3.22): the loader, the canonical hash, the validator (CI
   diffs it against `check.py`), the positional VM over all nine committed ASTs, and the
   fingerprint, with golden tests. The definitions the schema left to the runner (key, words,
   permutation, `consume` position, `x @ i`, `until_eof`, fingerprint) are in
   `runner/REFERENCE.md` §2. ~~Still to do: the I/O backends (`aeiou run`) and the ext4 / loopback
   NFS test,~~ **`aeiou run` with `sync` / `sync-direct` done 2026-09-30** (`runner/REFERENCE.md`
   §4, `DESIGN_REVIEW.md` §3.23) and run against ext4: every committed abstract with inputs
   reproduces its dry-run fingerprint; `runner/aeiou/tests/run.rs` covers the round trips,
   including the checkpoint write-then-restore handoff through the namespace manifest
   (`DESIGN_REVIEW.md` §3.24).
   ~~Still to do: the loopback NFS run (needs root on the development box),~~ **The loopback
   NFS run done 2026-09-30** (`runner/REFERENCE.md` §7, `DESIGN_REVIEW.md` §3.26): every
   abstract reproduces its fingerprint on the NFS v4.2 mount under `sync` cold, `sync` warm,
   and `sync-direct`, the two-rank tests pass with their roots on it, and the NFS client's
   RPC counts per backend are tabulated (a same-host restore issues zero READ RPCs; O_DIRECT
   sends one per application read). Still to do: ~~`--metrics` (item 14),~~ (built 2026-10-01, item 14) ~~`stream` access~~
   (contract 0.2, 2026-09-30), and the per-op cost (150–350 ns; caching a bound handle's
   path is the first fix). **The resumable VM and the `io_uring` backends done 2026-10-01**
   (`runner/REFERENCE.md` §8, `DESIGN_REVIEW.md` §3.29): every committed abstract reproduces its
   fingerprint on the event loop, on ext4 and on the loopback mount. **`posix-aio`,
   `libaio`, and `mmap` done the same day** (`runner/REFERENCE.md` §9, `DESIGN_REVIEW.md`
   §3.35); of the brief's list `gds`, `nixl-posix`, and `libnfs` remain.
4. **Spike 1:** blocking thread pool first, then io_uring; buffered vs. O_DIRECT, for
   `open → read → close` against the real NFS target, measuring `iou-wrk` count separately for
   open-heavy and read-heavy phases. It decides the I/O backend and the cache strategy. The
   thread-pool half exists as `aeiou run --io-backend sync|sync-direct` (2026-09-30), **the
   io_uring half as `io_uring|io_uring-direct` (2026-10-01, `runner/REFERENCE.md` §8)**; the
   measurements wait for the real target. On the loopback mount the io-wq worker count peaks
   ~~at the core count~~ at the count of actors with a punted op (20 with 20 actors, 30 to 46
   with 72; the kernel's cap is 80 per loop) for buffered and direct reads alike, because
   every open punts, so the hypothesis needs open-heavy and read-heavy phases measured apart.
   `--iowq-max-workers` (2026-10-01) is the knob for it: on the loopback a cap of one worker
   per loop lost nothing (`DESIGN_REVIEW.md` §3.34).
5. Spike 2: correctness and speed of the Feistel permutation at 50M/100M.
6. Spike 3: client dentry/inode slab growth and NFS op mix when touching 50M files.
7. Check `RWF_DONTCACHE` support in the NFS client on the target kernels.
8. Build `datagen` mode, driven by the same filename pattern, size distribution, and dataset seed
   as the abstract, writing ~~**verifiable, non-dedupable** data (per-block headers, §5)~~
   **reproducible data with controlled dedupe and compression ratios (revised 2026-09-29, §5)**
   and a manifest at the corpus root (`.aeiou-dataset.json` per dataset root, written last and
   atomically so a crashed datagen leaves none; fields and rules in `schema/README.md` §6,
   decided 2026-09-30). Container formats are written through their format class
   (real Parquet/HDF5/TFRecord/Arrow files, uncompressed, PLAIN encoding, payload from the
   generator); containers mean thousands of files, so a Python `datagen` for them is acceptable.
   **Done 2026-09-30 for `files` and `regions` datasets** as `aeiou datagen` in Rust
   (`runner/REFERENCE.md` §5): the positional wrapper over ~~`dgen-data` 0.3.0~~ SFC64 (since 2026-10-11), corpus-wide dedupe
   by seed reuse, the manifest with the resolved definition and the id, parallel by id. **The
   format-class writers done the same day** as `aeiou-datagen` in Python (`builder/REFERENCE.md`
   §6, `DESIGN_REVIEW.md` §3.28): real Parquet, TFRecord, HDF5, and tar files with the same
   payload (~~the `dgen-py` wheel reproduces the Rust crate's bytes~~ numpy's `SFC64` since 2026-10-11), sizes, names, and manifest;
   `aeiou datagen` refuses datasets with a format class (~~all~~ ~~HDF5 and Parquet~~ ~~Parquet~~ none since
   2026-10-12: built with the cargo feature `formats` it writes TFRecord, tar, HDF5, and, later the same
   day, Parquet, `DESIGN_REVIEW.md` §3.70; without the feature, all). The Python `aeiou-verify` tool is
   still to come. **Extended 2026-10-04 (`DESIGN_REVIEW.md` §3.62):** `aeiou datagen` runs
   across hosts with the same `--ranks`/`--rank`/`--coordinator` as `run` (each rank its
   contiguous slice of the ids, a `regions` file by one rank, the manifest by rank 0 once
   every rank has reported), always with `O_DIRECT`, and the payload wrapper is
   `aeiou-positional/2` (dedupe groups `D` consecutive files, so a prefix of the ids has the
   ratio and a corpus can grow). `aeiou-launch` takes `-f FILE` and `pdsh` bracket ranges.
9. Startup checks: `kernel.io_uring_disabled`, `RLIMIT_MEMLOCK`, and `RLIMIT_NOFILE` computed
   from G, W, and the abstract. (Since 2026-10-01 a disabled `io_uring` shows up as the ring
   setup failing, with the sysctl named in the error; ~~the limit computations are not done.~~)
   **Built 2026-10-01** (`runner/REFERENCE.md` §11, `DESIGN_REVIEW.md` §3.40; the choices
   decided 2026-10-01): open files and threads estimated from a bounded walk of one instance
   per template, soft limits raised, a refusal naming `RLIMIT_NOFILE`, `RLIMIT_NPROC`,
   `kernel.threads-max`, or `vm.max_map_count` (`--ignore-limits` overrides), and the counted
   open-file peak in the report. `RLIMIT_MEMLOCK` is not computed (pre-5.12 kernels only).
10. ~~Coordinator protocol and launch script (`pdsh`/ssh loop); test it on WSL2 with several
    ranks on `localhost`.~~ **Done 2026-09-30** (`runner/REFERENCE.md` §6, `DESIGN_REVIEW.md`
    §3.25): `aeiou run --ranks R --rank r --coordinator HOST:PORT`, rank 0 serving
    in-process, the configuration hash in `Hello`, the start gate, two-level barriers with
    departures, the merged report and manifests, one verdict on every host;
    `runner/aeiou-launch` is the ssh loop. Tested as two processes on `localhost`
    (`runner/aeiou/tests/coord.rs`). Still to do: a run on real hosts, and the connect
    window and heartbeat constants against real launch latency.
11. ~~Comparison policy~~ Decided 2026-09-28, including the interposition test, data
    verification, and seed privacy; see §5.
12. ~~Define the block header format and the PRNG fill for `datagen` and the verifier, and add
    `--verify-sample` / `--verify` to the CLI. Decide how GPU-memory backends verify (device-side
    kernel vs. sampled copy-back).~~ **Revised 2026-09-29.** ~~Check whether dgen-py generates
    positionally (bytes at an arbitrary offset without the prefix) **[verify]**~~ Resolved the
    same day (§5, *Payload generator*): positional by seed arithmetic, no seek API. Define the
    positional wrapper (seed per file or per 1 MiB block from `hash(dataset seed, index mod
    (total / D))`), the manifest fields for regeneration (generator, `dgen-data` version,
    parameters, dedupe/compression settings, block size), and the `verify` tool. The runner gets
    structural checks only; no `--verify-sample` in scored runs. **Manifest decided 2026-09-30**
    (`schema/README.md` §6): the verifier (`aeiou-verify`) takes the manifest as its input, so
    the manifest is to the dataset what the AST is to the workload. Later: ask upstream for
    `fill_block(seed, block_index, ratios)` or `seek(offset)`.
13. **Data-dependent workloads (2026-09-28).** Add to the semantic model: distributions over ids
    and offsets (`zipf`, `hotset`, random `offset` in `read`), recency references
    (~~`recent(site, d)`~~ `x @ i` since 2026-09-30, positional, no stored history), and
    positional names for write-then-read. Cross-actor read-after-write is limited to barrier-separated phases or a
    statistical hit model; state this limit in each such workload's documentation.
    `GRAMMAR_OPTIONS.md` §5.
14. **Locality-metrics check and trace mode (2026-09-28; the node was `replay` until 2026-10-02).** `--dry-run --metrics` computes
    reuse-distance, sequential run-length, popularity skew, request-size, dependency depth, and
    read/write mix on the abstract's stream; a trace tool computes the same from a real trace; an
    abstract is accepted for a workload class only when they match within tolerances. A `trace`
    AST node holds a literal captured sequence for small-scale calibration; never CLOSED.
    **The abstract's side built 2026-10-01** (`aeiou dry-run --metrics`, `runner/REFERENCE.md`
    §10, `DESIGN_REVIEW.md` §3.39): order-free metrics over the run, order-dependent ones
    per instance in a round-robin order of its sub-actors, block-level stack distance in
    bytes, hash sampling, a JSON form. The definitions were confirmed by the user the same day (**decided 2026-10-01**).
    ~~Still to do: the trace tool, the tolerances, the `trace` node.~~
    **The trace's side built 2026-10-01** (`aeiou-trace`, `builder/REFERENCE.md` §7,
    `DESIGN_REVIEW.md` §3.42; choices confirmed by the user, **decided 2026-10-01**): the same numbers from
    an `strace` (calls under `--root`, completion order, the trace as one instance, a thread
    as a context, fan-out and depth from `io_submit` only) and `aeiou-trace compare`, a
    distance per metric. Checked against traces of the runner itself. ~~Still to do: a trace
    of a real application, the tolerances, the `trace` node.~~
    **First real application traced 2026-10-01** (`ABSTRACTS.md` §1 "Trace",
    `DESIGN_REVIEW.md` §3.43, `workloads/input-pipeline/map/file-per-sample`): PyTorch `DataLoader` +
    `ImageFolder` on loopback NFS; the abstract gained the second `lseek` per file and the
    walk's `stat`/`fstat`, after which every op share and request size is equal and the
    largest distance is 0.048 (reuse distance, the order). Choices confirmed by the user (**decided 2026-10-01**). 
    **Second, 2026-10-01** (`ABSTRACTS.md` §2 "Trace", `DESIGN_REVIEW.md` §3.44,
    `workloads/input-pipeline/map/npy-per-sample`): `np.load(...)["x"]` as upstream DLIO and any NumPy user issues it. The
    drafted abstract was wrong in shape (two members, header reads, no read at 0) and was
    rewritten from the trace; largest distance 0.040 on a 256-file corpus. Explicit seeks
    confirmed by the user (**decided 2026-10-01**); the other choices not yet.
    **Third, 2026-10-01** (`ABSTRACTS.md` §3 "Trace", `DESIGN_REVIEW.md` §3.45,
    `workloads/checkpointing/dcp-save`): `dcp.save` on two ranks. The frame was right, the item
    loop was not (704 + item + 873 per item, split at the buffer size, two tells); counts and
    bytes now equal but for 10 path calls the contract cannot express. The traced writes are
    unaligned, so a direct backend cannot run them. ~~Choices not yet confirmed.~~ Choices decided 2026-10-02.
    **Fourth, 2026-10-01** (`ABSTRACTS.md` §4 "Trace", `DESIGN_REVIEW.md` §3.47,
    `workloads/checkpointing/dcp-load`, `workloads/model-loading/safetensors-load`): `dcp.load` on two ranks and
    safetensors `from_pretrained`. The restore's item loop was rewritten (six reads and 34
    seeks per item through one 1 MiB buffer; counts now equal). The model load issues no
    `read`: two mappings per shard, `fadvise(SEQUENTIAL)`, and nothing touched until a tensor
    is used; the abstract gained those calls and the small JSON files, and its per-tensor
    slices stay unverified without a GPU engine. ~~Choices not yet confirmed; open for the
    user: whether CLOSED means `mmap` for an application that maps.~~ **Decided 2026-10-01**
    (`DESIGN_REVIEW.md` §3.48): the abstract declares its application's backend and the
    runner defaults to it (`mmap` here); the restore mimics the reader exactly, which a
    third trace showed needs the read order (sorted names, not file order) and the buffer
    carried from item to item. ~~The other §3.47 choices are not yet confirmed.~~ Those that need no GPU were confirmed 2026-10-02. ~~Still
    to do: the other five rows of `ABSTRACTS.md` §11, the tolerances, the `trace` node.~~ Still
    to do: ~~rows 5 to 8~~ ~~rows 5, 7, and 8~~ ~~row 8~~ ~~a chat replay for row 8 of `ABSTRACTS.md` §11,~~ ~~the tolerances,~~ the `trace` node.
    **Tolerances built 2026-10-02** (`aeiou-trace compare --judge`, `builder/REFERENCE.md` §7, `DESIGN_REVIEW.md` §3.55; decided 2026-10-02: the class values and the two thin margins are fine for now): a tolerance per class of metric (0.05 for the op mix, request sizes, and popularity; 0.10 for run lengths, reuse, fan-out, and depth), raised by what the abstract differs from itself by at other seeds; rows the trace cannot show are named per pair and not judged. Of the fifteen committed pairs ~~nine are accepted, four are not (the DiskANN build and the three KV-cache pairs, each row with its recorded reason; the KV pairs' rows are those of the replay since §3.56)~~ eleven are accepted, two are not (the DiskANN build and the KV-cache writer, each row with its recorded reason; since §3.57 and one `keep` draw per round, 2026-10-02), and the two applications that read through a mapping cannot be judged from an `strace` at all.
    **Sixth row, 2026-10-01** (`ABSTRACTS.md` §6 "Trace", `DESIGN_REVIEW.md` §3.49,
    `workloads/vector-search/faiss-ivf-ondisk`): FAISS `IndexIVFPQ` over `OnDiskInvertedLists` on SIFT1M.
    FAISS maps the lists file and issues no call on it; 32 prefetch threads per search slice
    touch every probed list, ids before codes. The abstract declares `mmap` and gained the
    index file, the prefetch fan-out, and that order; list-size spread and popularity are
    now SIFT1M's. ~~Choices not yet confirmed.~~ Choices decided 2026-10-01.
    **Fifth and seventh rows, 2026-10-02** (`ABSTRACTS.md` §5 and §7 "Trace", `DESIGN_REVIEW.md`
    §3.50, `workloads/vector-search/diskann-search`, `workloads/vector-search/diskann-build`): DiskANN
    through diskannpy on SIFT1M. Search: 28 dependent rounds and 108 sector reads per query
    (the draft had 3 to 8 rounds), the first two rounds on the medoids and their neighbours,
    which a 1 % node cache holds; the abstract declares `libaio` and opens once per process.
    Build: the base file read whole twenty times, 64 MiB layout writes, no `fsync`; the
    abstract was rewritten and agrees with the trace to 0.1 % in calls and bytes, and with
    the wire in READ, WRITE, and COMMIT counts. ~~Choices not yet confirmed.~~ Choices decided 2026-10-02. ~~Row 8 remains.~~
    **Eighth row, 2026-10-02** (`ABSTRACTS.md` §8 "Trace", `DESIGN_REVIEW.md` §3.51,
    `workloads/kvcache/lmcache-local-disk`): vLLM with LMCache's local-disk backend on a GPU. No
    lookup reaches storage; a chunk is one file in one flat directory, one `write`, one
    `read`; only whole chunks of the prompt are stored and nothing during decode; reads
    happen only for what the engine's own GPU cache lost. The abstract lost its `stat`
    loops, directories, and decode writes and gained `local` and `sys_local`. ~~Choices not
    yet confirmed.~~ Choices decided 2026-10-02. Every row now has a trace; ~~the chat replay for the KV distributions is
    still to do.~~
    **Chat replay, 2026-10-02** (`ABSTRACTS.md` §8 "Replay", `DESIGN_REVIEW.md` §3.56,
    `replay.py` and `fit.py` in `workloads/kvcache/lmcache-local-disk`): ShareGPT through the same
    server, on local disk and on the shared store, 300 requests with eight conversations
    open. The dataset gives the lengths and the share of first turns, now the defaults; it
    has no timestamps, so the reuse distance is a load parameter and no longer
    **[measure]**. `local` is replaced by `keep` (a draw of the tokens the engine still
    holds, since it keeps the start of a conversation and not all or nothing), `reuse` has
    one distance (several make two requests continue one conversation and rewrite its
    chunks), and `context` ends a conversation that no longer fits. The three KV pairs went
    from 6, 7, and 4 rows outside the tolerances to 1, 2, and 1, ~~and are still not
    accepted: the reuse distance of the reads misses by 0.003 to 0.018, because the traced
    loads come in bursts that an independent draw per request does not make.~~ ~~Choices not
    yet confirmed.~~ Choices decided 2026-10-02, and the user agreed to the wider `at` rule
    that one draw per round of conversations needs (§3.57). **Built later the same day
    (`DESIGN_REVIEW.md` §3.57, contract 0.5):** the `at` offset may be an expression
    provably ≥ 1, `keep` is drawn once per round of the open conversations (`kp @ (r − (r
    mod d + 1))`), the loads per round spread as the trace's do, and the reuse distance is
    within in all three pairs: `kv_cache_serving` and the reader are **accepted**, the writer
    keeps one row (the trace's two store threads). The runner also judges V3 again on the
    parameter values in effect (a `--param` distribution with minimum 0 was a stack overflow).
    ~~Choices of §3.57 not yet confirmed.~~ §3.57 decided 2026-10-02. The wire counts were not taken again under the replay.
    Wire counts taken later the same day: three WRITEs, one COMMIT, one OPEN per chunk for
    the server and the abstract alike, and no READ in either, since the client still holds
    the chunks it wrote. ~~Which cold-reader configuration to score is open (§3.51).~~
    **Decided 2026-10-02:** the cold reader is two runs, a writer and then a reader over an
    `input` namespace after `--drop-caches` (§3.51); ~~the reader abstract is not built yet.~~
    **Built 2026-10-02** (§3.52, `workloads/kvcache/lmcache-shared-fs`): the local-disk backend
    has no reader (a restarted engine rewrites the store), so the pair is traced from
    LMCache's `fs://` backend: `kv_cache_shared` (stat lookups, header and chunk written to
    a temporary name and renamed) and `kv_cache_shared_reader` (the same request stream on
    the filled store: no write, every chunk it does not hold loaded). The reader runs with
    the writer's seed and parameters. 188 READs on the wire for vLLM, 192 for the abstract.
    ~~Choices not yet confirmed.~~ Choices decided 2026-10-02, with those of §3.50 and §3.51.
    **Review of 2026-10-02** (`DESIGN_REVIEW.md` §3.53): the GETATTR per READ of the second
    row was `strace -yy`'s, so wire counts are taken from untraced repeats from now on;
    `train_large_samples` gained the training script's listing as an optional `enumerate`
    phase; `framing` is checked against the installed NumPy by the builder's tests. §3.47
    waits for a GPU environment. ~~Open for the user: whether `enumerate` is on by default
    in both training abstracts, and an `aeiou-params npz FILE` for corpora written by
    other tools.~~ **Decided and built 2026-10-02** (`DESIGN_REVIEW.md` §3.54): the
    directory walk is on by default in both training abstracts (the metadata load is part
    of the workload); `aeiou-params npz` reads the framing from real archives; the
    remaining §3.44, §3.45 and non-GPU §3.47 choices are confirmed; contract 0.4 adds
    `same_run` (V15), under which a reader whose seed, instance count, or common parameters
    differ from the writer's is refused before the gate.
    **The `trace` node designed, built, and decided 2026-10-02** (`DESIGN_REVIEW.md` §3.58,
    `runner/REFERENCE.md` §13, `builder/REFERENCE.md` §7; renamed from `replay` the same day at the user's call): a JSON Lines trace file written by `aeiou-trace export` from the same
    `strace` the metrics read, one lane per traced task, opens referenced by id so lanes
    share descriptors, shared positions resolved at export, gaps from
    the timestamps as `compute` under `--time-scale`, two cross-lane orders from the trace (the open table and path order, the second found necessary by the first write-then-read trace); the runner's dry run walks the file in
    line order so that `dry-run --metrics` of a trace node equals `aeiou-trace metrics` of its
    source, checked on traces of the runner itself; the trace then runs with its dry run's fingerprint under ~~the blocking backends~~ every backend (the event loops since 2026-10-03). ~~Still to do: all of it.~~ ~~Still to do: traces under the event-loop backends.~~ **Item 14 is complete (2026-10-03).**

15. **Container formats (2026-09-29).** Write the format classes (Parquet/pyarrow, TFRecord,
    HDF5/h5py, Arrow IPC, WebDataset tar, MDS, Megatron `.bin`/`.idx`), each from a trace of its
    reader library; add a ninth abstract for streaming training over TFRecord or Parquet (the
    MLPerf Storage ResNet50/CosmoFlow shape) and a tenth for the Parquet→Arrow conversion pass
    plus training from a memory-mapped Arrow cache, with cache placement (local vs. SUT) as a
    parameter. ~~Extend the `loader` sugar so items per batch are decoupled from reads per item.~~
    `GRAMMAR_OPTIONS.md` §6. **Done 2026-09-30 for Parquet, TFRecord, HDF5, and WebDataset
    tar** (`DESIGN_REVIEW.md` §3.28, contract 0.2 in `schema/README.md` §9,
    `builder/aeiou/formats.py`, `aeiou-datagen`): pyarrow, h5py, and `tarfile` traced on the
    loopback mount, TFRecord from source [verify]; the ninth abstract is
    `train_stream_shards.py` (two workloads, TFRecord and Parquet) plus `train_map_hdf5.py`;
    the loader needs no new knob (under `stream` its unit of work is the shard). Still to do:
    Arrow IPC, MDS, and Megatron classes, and the tenth abstract.
16. **Cold start: `--drop-caches` (decided and built 2026-10-01, `DESIGN_REVIEW.md` §3.31,
    `runner/REFERENCE.md` §4; the drop verified as root on the loopback mount the same day, `--remount` not built).** One run
    option, applied on every host after the dataset and namespace checks and immediately
    before the host arrives at the start gate: `sync`, then `3` into `/proc/sys/vm/drop_caches`
    (page cache, dentries, inodes; evicting an NFS inode drops its attribute and access
    caches and returns its delegation). Never between batches, epochs, or phases: the goal is
    only that caching from before the run does not affect the run, and the checkpoint
    write-then-read case is two runs already (`ckpt_write_dcp` then `ckpt_restore` through
    the namespace manifest). Root is required; the runner refuses to start when the write
    fails on any host. The report records the drop, its duration (outside `elapsed` by
    construction), and before/after `Cached` (printed less `Shmem` since 2026-10-01), `dentry-state`, `inode-nr`; a residency check
    (`mincore` over a few hundred dataset files chosen by formula, no root) reports the
    fraction of sampled dataset pages resident at the start, ~~with or without the flag,~~ only
    with `--drop-caches` or `--require-cold` (revised the same day: a plain run does not pay
    the sample's opens), and `--require-cold` covers it as it covers input namespaces. The mount's `opts:` line
    (`actimeo`, `lookupcache`, `nconnect`) goes into the host counters. The full reset
    (`fscache`, NFSv4 state) is `aeiou-launch --remount`, outside the runner, which never
    unmounts the storage under test.
17. **Object backends via `s3dlio` (preliminary opinion 2026-10-01, `DESIGN_REVIEW.md`
    §3.32).** Behind a cargo feature so the default build stays tokio-free; design the
    POSIX-to-object mapping table and the validity rule first; first measurement is the same
    fingerprint over `file://` through the library versus the `sync` backend, pricing the
    library before any cloud is involved. *2026-10-09:* promoted to the top of the order, after
    the access layers it plugs into (item 25, `DESIGN_REVIEW.md` §3.65). *Later on
    2026-10-09:* the engine is built on Apache `object_store`, not `s3dlio` (measured: 3–4×
    less client CPU per small request; `s3dlio`'s S3 path is the AWS SDK, and its
    `object_store` backend does not compile), and it is measured against an S3 server, not
    over `file://` (`DESIGN_REVIEW.md` §3.65). *Built the same night, reads first:* object
    datasets written by `aeiou datagen` and read by `aeiou run` against MinIO, to the dry
    run's fingerprint (`runner/REFERENCE.md` §16); ~~next, object namespaces (the writes,
    `--object-part-size`, V18) and the event loops.~~ *The writes the same night:* object
    namespaces written as uploads (multipart in parts of `--object-part-mib`, sent by
    `close`), `rename` as copy and `DELETE`, the namespace manifest as an object, rule V18;
    ~~next, the event loops.~~ *The event loops the same night:* `io_uring` and `libaio` take
    object names, each op's request spawned on the engine's runtime and its answer waking the
    loop through its eventfd (`DESIGN_REVIEW.md` §3.65); ~~next, `aeiou-datagen` to object
    stores and a measurement against a real store.~~ *`aeiou-datagen` to object stores
    2026-10-11:* each container file streamed from its writer into an upload through
    `obstore` (the runner's `object_store`), with no local copy (`DESIGN_REVIEW.md` §3.69);
    next, a measurement against a real store.
18. **JSON report (built and decided 2026-10-01, `DESIGN_REVIEW.md` §3.41,
    `runner/REFERENCE.md` §12).** `aeiou run --report-json FILE` writes the
    run's identity, results (latency histograms in full), and verdict as format
    `aeiou_report: 1`; written on failure too; rank 0 holds the merged report, every other
    rank its own; `--report-takes` adds every take. No JSON Schema yet.

19. **`O_DIRECT` checkpointing (deferred 2026-10-01, `DESIGN_REVIEW.md` §3.45, §3.46).**
    The traced PyTorch checkpoint writer issues unaligned writes, so `ckpt_write_dcp` does
    not run under a direct backend. Low priority, to come back to: find and trace a real
    application that checkpoints with `O_DIRECT` and write its abstract. Not from DLIO
    (decided 2026-10-01: DLIO is an emulator and is not traced), and no pad-and-coalesce
    behaviour in the runner.
20. **Tensor-parallel sharding of a KV chunk (2026-10-02, `DESIGN_REVIEW.md` §3.59).** LMCache
    under TP stores one logical chunk as one shard file per rank, all under one key, the
    ranks' writes and reads balanced. The runner's positional draws are keyed on the GPU id,
    so `--gpus 8` is eight independent engines, not one TP group. A TP group needs a shared
    draw (the group's request stream) and a per-rank name; not designed.
21. **The agentic load on a GPU (2026-10-02, `DESIGN_REVIEW.md` §3.59).** The AgentX corpus
    is fitted (`agentx.py`) and checked against its own chunk accounting; what it cannot
    give needs a run of `replay_agentx.py` through vLLM and LMCache on the trace box: `keep`
    at a chosen KV memory, the `strace` for `aeiou-trace compare --judge`, and the store's
    view of the prefill-step write bursts. Also from that box, when convenient: the
    local-disk backend's `O_DIRECT` option, which LMCache's source shows to be the same one
    write and one read per chunk through `os.open` with `O_DIRECT` (buffered when the chunk
    is not a multiple of the file system's block size), so that `sync-direct` can be the
    declared backend of a traced configuration rather than another workload.
    *2026-10-07 (`DESIGN_REVIEW.md` §3.63):* both run on the development box's GPU, buffered
    and with `use_odirect`. `keep` measured; the system prompt is now read when the engine
    holds none of the conversation and can be one per slot (decided and built), and the
    agentic pair is accepted. `direct` (LMCache's `O_DIRECT` calls, under the declared
    `sync` with `DIRECT` on the chunk opens, not `sync-direct`) decided and built the same
    day; its pair has one row outside, recorded (the server's admission order).
22. **The option layers (designed, built, and decided 2026-10-04, `DESIGN_REVIEW.md` §3.60,
    `runner/REFERENCE.md` §14).** Command line, then `AEIOU_<FLAG>`, then the TOML file
    `--config`/`AEIOU_CONFIG` names, then the default; the workload's identity, the checks
    that pin it, and the unsafe overrides are the command line's alone; every invocation
    prints its options with their sources; `aeiou run` records them in the report and the
    coordinator compares them across hosts, naming what differs. Not built, on purpose: a
    check that refuses differing layered options across hosts (a WG rule, if ever).

23. **Datagen resume, growth, and subset runs (planned 2026-10-04, `DESIGN_REVIEW.md` §3.62).**
    A submitter pre-generates one very large corpus on the fleet it will run on, then spends
    days running ever larger workloads against it to find the SUT's saturation point; it must
    not regenerate per step. Three operations, one arithmetic: `--resume` walks the ids in
    scope (a formula over `--ranks`/`--rank` at resume time, nothing persisted, so the host
    count may change) and rewrites in full every file that is missing or not of its size (a
    crashed `O_DIRECT` writer leaves a strict prefix; allocated block counts are not evidence
    on reducing storage); growth is a resume under a larger count; a run over the first *S*
    ids of a larger corpus relaxes the manifest's `count` comparison to "at least". Names,
    sizes, directories, and payload groups are already functions of the id alone. Not built.
    *Noted 2026-10-09, with the object engine's writes (`runner/REFERENCE.md` §16):* in an
    object store the same arithmetic differs twice. No object is ever short, since an upload
    that dies before its completion leaves no object, so resume looks for missing keys only,
    and one paged `LIST` of the prefix gives every key's size, with no `HEAD` per file. And a
    crashed writer can leave a multipart upload's parts behind, invisible to `LIST` but
    stored (and billed): resume lists them (`ListMultipartUploads` below the prefix) and
    aborts them, which POSIX has no counterpart for.
    *Decided with the user 2026-10-11 (`DESIGN_REVIEW.md` §3.70):* the manifest written at the
    start with a state and per-writer progress, sent to rank 0 every `--progress-secs` and
    written once per window, everything after the last recorded point rewritten on resume
    (any host count); whole directories (counts rounded up, work by directory); explicit grow
    and shrink from the manifest; `O_DIRECT` in the Python writer; runs check the manifest and
    take `--subset` for a smaller count, whole directories from 0; an inventory variant for
    large corpora. The payload became `aeiou-positional/3` (SFC64 in 4 KiB pieces, built the
    same day). The rest is not built.
    *Revised later the same day (§3.70):* ~~`O_DIRECT` in the Python writer~~ dropped; the
    container formats move to `aeiou datagen` behind a cargo feature `formats` (TFRecord and
    tar, then HDF5 with its header made by h5py, then Parquet ~~through parquet-rs~~ by our
    own writer, revised 2026-10-12), and the
    rest of this item is built in Rust only, after them. TFRecord and tar **built
    2026-10-12** (byte for byte the Python writer's; 20.7 and 22.6 GB/s to tmpfs at 16
    writers against Python's 5.9 and 4.8); HDF5 the same day, its header from `aeiou-header`
    beside `aeiou`, run by rank 0 alone (14.8 GB/s against 6.9); Parquet the same day, our
    own writer, pyarrow's bytes but for `created_by` (22.6 GB/s against 4.5). The manifest
    from the start with its state and progress (§3.70 item 3) built the same day
    (`manifest_version` 2, `--progress-secs`); resume, whole directories, grow and shrink,
    and subset runs remain. *Revised 2026-10-12 (§3.71, proposed):* whole directories are
    not required of a corpus (only of a subset run's count); a sizing step writes a
    parameter file with the count rounded up, the layout, and the steps (*revised
    2026-10-13, §3.74:* the sizing step is the WG's, in `mlpstorage`; aeiou keeps the
    projection in `dry-run`, the capacity check in `datagen`, and the host facts in the
    report). *Decided
    2026-10-12:* resume, grow, and shrink are flags of `aeiou datagen` taking the dataset
    root in place of the abstract (`--resume ROOT`, `--grow N ROOT`, `--shrink N ROOT`).
    Resume **built 2026-10-12** (§3.70): the unfinished ids written again, whole, over any
    host count, from the manifest alone. Grow and shrink **built 2026-10-12** (§3.70): the
    new count's manifest first, the files a fresh datagen of N would write and the corpus
    lacks written (the old short last file among them), the names past N removed with the
    directories they leave empty, so the corpus is a fresh datagen's byte for byte with its
    id; an interrupted one finished by `--resume`. *Decided and built 2026-10-12 (§3.70):*
    `--settle ROOT` keeps what an interrupted datagen, grow, or shrink reached (the runs
    handed out finished, a shrink never continued, strays past the claim mark removed), at
    whole directories, because every datagen now hands its ids out in runs from one queue on
    rank 0, in id order (from the top down in a shrink); Ctrl-C stops every host with the
    manifest true (exit 130); `--resume` and `--settle` refuse a manifest less than two
    progress windows old unless its datagen recorded that it stopped. ~~Subset runs remain.~~
    Subset runs **built 2026-10-13** (§3.70 item 6, §3.72): `aeiou run --subset` runs a
    consumed dataset's count below its corpus's as the corpus's first ids, rounded up to
    whole `div` directories (whole files when flat), the workload and fingerprint of a fresh
    corpus of that count, the corpus's dataset id, and the age bias said. ~~The inventory
    variant remains.~~ The inventory **built 2026-10-13** (contract 0.9, §3.73): a consumed
    dataset's `inventory {name, entry}`, a file list in its root written by datagen and read
    through the `inventory` handle in place of a walk; under `--subset` its reads are issued
    no further than a corpus of the run's count would end. Its shape (format, columns, who
    reads it, read size) is `[measure]` until a job that reads one is traced.

24. **Computed holds (deferred 2026-10-08, `DESIGN_REVIEW.md` §3.64).** The KV abstracts draw
    what the engine holds (`keep`, `sys_held`, `sub_held`), fitted at one engine pool; a pool
    three times larger read a fifth as much, and only a refit of those three followed it. A
    simulation of vLLM's prefix cache (LRU, a request freed tail first, FCFS preemption,
    `agentx.py holds`) explains the measured holds at both pools on the engine's own timeline;
    on a timeline an abstract could compute it holds without pool pressure and over-predicts
    under it. Computed holds would be a per-engine cache in the runner (a contract change,
    `GRAMMAR_OPTIONS.md` §5.3 amended) with a `pool_tokens` input, plus the admission's
    dynamics under pressure; deferred because the WG fixes the engine and the CPU tier (§8),
    so drawn holds at the reference configuration serve CLOSED. Kept for it: the fits'
    conditions, the simulation, and the 2 GiB and 6 GiB pairs. *Reopened 2026-11-06 by the user: the
    pool as a computed cache, proposed and undecided, with the morning's debate and measurements
    (`DESIGN_REVIEW.md` §3.99); the demand is to be fixed first.*

25. **The access layers (proposed and decided 2026-10-09, `DESIGN_REVIEW.md` §3.65).**
    Format classes are already above the access APIs (the builder emits POSIX-shaped ops);
    "backend" below them binds three choices into one name (API, `O_DIRECT` on every open,
    blocking or event-loop completion) over file descriptors. Proposed: separate axes, each
    bound where it belongs and owned by the abstract or the solution by the interposition
    test (§5): protocol family (POSIX or object) and endpoint per namespace, so a run may mix
    protocols (built now, decided: KV caches mix tiers today, NVIDIA's G3 POSIX and G4 object
    by the user's account [verify]); the API per run as the abstract declares it; `--cache` as the run-wide
    modifier §4 planned, ~~with the nine names as aliases~~ (aliases removed the same day: `--io-api` and `--cache` alone; *later the same night* an API per protocol, `--posix` (renamed from `--io-api`) and `--s3`, the protocol `object` renamed `s3`, contract 0.7); transport (TCP or RDMA) the
    solution's, a mount option under a kernel client and a library option under a user-space
    one; the buffer axis with GDS and `nixl-posix`. The object mapping drafted with a validity
    rule (~~V16~~ V18; V16 is the `trace` rule); ~~contract 0.6 keeps every AST, fingerprint, and dataset id.~~ contract 0.6
    replaces `backend` with separate fields and regenerates every AST (decided: the only ASTs
    are this repository's); fingerprints and dataset ids are unchanged. ~~`s3dlio` is one
    engine on these axes;~~ the runner's POSIX path does not go through ~~it~~ an object library. Then: the refactor
    with no change of behaviour, ~~`s3dlio` over `file://` (item 17), an S3 server on the
    development box, and only then PRs to `s3dlio`, the narrow ones the measurements call for
    (decided; no shared POSIX-engine crate for now)~~ the object engine on Apache `object_store`
    against an S3 server on the development box (item 17; decided later the same day after
    measuring it against `s3dlio`; no PRs to `s3dlio` and no shared POSIX-engine crate). Deferred until ~~`s3dlio` over `file://` is
    measured~~ the object engine runs: the first object abstract's application (under CLOSED, S3 under a POSIX-traced
    abstract is another workload). Decided the same day: `--cache` defaults to the abstract's
    flags as written.
26. **A catalog of workloads (proposed 2026-10-10, `DESIGN_REVIEW.md` §3.66).** One tree,
    `workloads/<category>/[<class>/]<mechanism>/cases/<case>/captures/<capture>`, replaces
    `builder/abstracts/`, `builder/traces/`, and `schema/examples/`: a directory per I/O
    mechanism (one authoring script and its ASTs, named alike), a case per workload a user
    would ask for (an AST and a parameter set), captures under the case that was traced. Cases
    are labeled from an enforced dictionary (`workloads/attributes.toml`): curated, derived
    from the AST, and measured from the dry run at the smallest length where the numbers are
    stable. A separate tool, `aeiou-catalog`, checks and generates the catalog. Before
    `aeiou-datagen` to S3 (item 17). **Built 2026-10-10** (`builder/REFERENCE.md` §8,
    `man/aeiou-catalog.1.md`, `DESIGN_REVIEW.md` §3.66): the tree, the dictionary, the
    records (`mechanism.toml`, `case.toml`, `capture.toml`), `shape.json` per case, and
    `CATALOG.md`; the shape is measured at the case's own length, not a shorter one (a short
    run's reuse is not the case's), and re-checked at the smallest length that reproduces it.
    **Restructured 2026-10-11** (decided by the user, `DESIGN_REVIEW.md` §3.68): the first
    level is a class of system named as storage people name it, seventeen categories
    (`input-pipeline`, `dataprep`, `checkpointing`, `model-loading`, `kvcache`, `vector-search`,
    `embedding-tables`, `rl`, `agentic`, `software-build`, `eda`, `vdi`, `media-ingest`,
    `database`, `genomics`, `hpc`, `benchmarks`), each with its directory from the start; a
    `class` level only where a category is large; `phase` and `domain` as attributes of
    several values in place of an AI directory; `model-family` per case.
27. **The abstract in prose (backlog, raised by the user 2026-10-10; built 2026-10-14,
    `DESIGN_REVIEW.md` §3.76: `DESCRIPTION.md` in every case, its paragraph in `CATALOG.md`,
    `aeiou-catalog describe` for any AST; the raw-trace prose below not yet).** An AST is a compact,
    machine-readable account of a workload's I/O, so it can be rendered as English: a
    literal abstract, in the sense of a paper's, of the workload's patterns, for readers who
    will not read the JSON (WG reviewers, submitters, a catalog's visitors). Generated, not
    written: a deterministic walk of the AST (actors, loops, loaders, what `consume` draws and
    in what order, each op with its sizes, distributions in words and percentiles, parameters
    by name and value with their `doc` strings), joined with the case's measured shape
    (`shape.json`, item 26) and the mechanism's curated attributes. Two lengths: a paragraph
    (context, pattern, measured shape) and a section-by-section account. Checked in CI against
    the AST like the man pages against `--help`, so it cannot drift; a parameter change shows
    up as a prose diff. What it cannot say is why the application does it (the mechanism's
    README) or what a storage system should do about it, which is interpretation and stays
    labeled as such if ever added. A language model may polish the generated text only if
    every number and claim is then checked against the generated facts; deterministic
    templates first. Natural home: `aeiou-catalog describe`, after item 26. The user's note (2026-10-10): the same
    for a raw trace, an accurate and succinct prose description of what a captured
    application did, has value of its own (through `aeiou-trace metrics` and, for the call
    sequence, the `trace` node's lanes), whether or not an abstract is ever fitted to it.
28. **Comparability and fidelity (backlog, 2026-10-10, `DESIGN_REVIEW.md` §3.67).** Two goals,
    named so every rule says which it serves: comparability (a score does not move with the
    harness's choices: client count, GPUs per client) and fidelity (a result predicts what a
    real deployment sees from storage). The 5× rule serves the first and distorts the second
    for small datasets, since a real cluster at 8 GPUs per node has about 8× the client DRAM
    per GPU of a benchmark at 64. To build: per-node miss-ratio curves in `dry-run --metrics`
    (the reuse-distance histogram is the LRU curve; `--gpus-per-node`, 8 by default), shown
    in the catalog's shape; the prediction "GPUs the storage can feed ≈ measured capability ÷
    (demand per GPU × client miss ratio)"; cold-start time as the storage metric of a case
    whose working set fits; one realistic node (8 GPUs, the application's memory held by a
    ballast) run to validate the curve against Linux's page cache, which is not an exact LRU.
29. **The harness's topology in the report (backlog, 2026-10-10, `DESIGN_REVIEW.md` §3.67).**
    The report says how many GPUs each client host emulated, the client hosts' memory, and the
    datasets' bytes against it, and the client hit ratio where the client exposes counters
    (NFS mount counters today); reported, never capped. *2026-10-13 (`DESIGN_REVIEW.md`
    §3.74):* built the same day: every host's memory, CPUs, kernel, and cgroup memory limit
    sent with its Hello, and the datasets' projected bytes against their sum (the report's
    `facts`, `hosts[].facts`, `memory`); the GPUs per host were there already (`gpu_ids`,
    `ranks[].gpus`); the client hit ratio remains. ~~The 5× judgment is `mlpstorage`'s, from
    `run --report-json`.~~ *(Revised 2026-10-21, `DESIGN_REVIEW.md` §3.78 point 10: aeiou
    judges it when the WG's rules file carries it; item 33.)* Simulated nodes on one host (cgroup
    cache budgets) are not pursued: the groups share one page cache, separable only under NFS
    `nosharecache`.
30. **Concurrent mixes of cases (backlog, critical but not yet, 2026-10-10, `DESIGN_REVIEW.md`
    §3.67).** Several cases run at once against one system: the research site where many
    small jobs share the storage and compete for its caches, which a single case (one job
    with every cache to itself) cannot show. Needs a run of several ASTs under one
    coordinator, each with its own seed, GPUs, and report, and a fingerprint per case.
31. **A Parquet file with a short last row group (queued by the user 2026-10-11, soon but not
    now; `DESIGN_REVIEW.md` §3.69).** The layout gives each column one page-header size, and
    a row group of fewer rows has shorter headers (with 64 rows a group, a row count below 64
    is one varint byte, not two), so `aeiou-datagen` refuses such a file at its layout check,
    in the old writer and the streaming one alike. Today `samples_per_file` must be a
    multiple of `rows_per_group` and `count` must leave no short last file; the committed
    abstracts comply. The fix is a choice: the builder refuses those sizes at build time, or
    the layout gives a short last group its own header size (the runner's offset formula
    changes, and so does every Parquet dataset id).
32. **Retire `aeiou-datagen` (backlog, decided by the user 2026-10-14, low on the list).** `aeiou
    datagen` writes every format class (TFRecord, tar, HDF5, Parquet behind the cargo
    feature `formats`, 2026-10-12), to directories and object stores, faster, and alone has
    `--resume`, `--grow`, `--shrink`, and `--settle`; the Python writer is deprecated and
    gets no new features (its free-space check, the twin of `DESIGN_REVIEW.md` §3.71
    point 5, was dropped from the list for that reason). It stays the byte and behaviour reference in tests until it is
    retired (`DESIGN_REVIEW.md` §3.70 item 5). Retiring it: whatever the tests compare
    against it compared against committed bytes or hashes instead; its man page, CLI entry
    point, and CI steps removed; the refusal of a container dataset by a build without
    `formats` (`runner/aeiou/src/datagen.rs`), which today points at it, pointing at the
    feature alone; the docs that name it (README, builder REFERENCE §6, the workloads'
    READMEs, QUICKSTART). `aeiou.formats` stays: the format classes are the builder's.
33. **Use cases, benchmark rules, and the lock (~~backlog,~~ decided 2026-10-21 and 2026-10-22,
    `DESIGN_REVIEW.md` §3.78; ~~nothing built~~ built 2026-10-22, `runner/REFERENCE.md` §17,
    but for the abstracts outside the input pipelines, item 38).** Contract 0.11: the `scale` tag, the dot
    refused in parameter names, `{"from": "st_blksize"}` values, `params_version` 2 with the
    states locked, default, and required and a `run` table, a rules schema. The builder:
    `aeiou-params lock CASE [--rules FILE]`, `aeiou-build` re-locking the committed cases of
    what it rebuilt. The runner: `--case`, `--case-file`, `--workloads`, `--expect-case-sha`,
    `--expect-workload-id` in `run` and `datagen`; the `fact.` and `result.` names and
    constraints checked at resolve, the gate, and after the run; exit status 3 for a run that
    did not pass; the lock embedded in the report; the step-time histograms and the slowest
    ten delays; a test that `aeiou run` touches no file system between the gate and the end.
    `aeiou-catalog check`: one lock per name, no case without a lock, no stale lock, the
    `accelerator` attribute. The abstracts: job and accelerator values become required
    (`npy_per_sample`'s H100 set moves to a `unet3d-h100` use case), `[config]` becomes
    `scale`, the block-size reads become storage-reported; every AST hash changes. *Built
    2026-10-22 in six parts:* contract 0.11; `params_version` 2, the lock, and the catalog's
    checks; the runner's side; the step times and the no-I/O test; the five input-pipeline
    abstracts (`npy_per_sample` to the use case `unet3d-h100`, whose lock runs the golden
    fingerprint; `file_per_sample`, `hdf5_map`, `parquet_stream`, `tfrecord_stream`, whose
    placeholder defaults became cases of kind `unsettled`, `jpeg-50m`, `rgb256-rows`,
    `imagenet-rows`, `imagenet-records`); the docs. The eight other abstracts keep their job
    values as defaults (item 38), and a use case per accelerator is item 39.
34. **Recapture the committed traces (backlog, 2026-10-22, `DESIGN_REVIEW.md` §3.79).** The
    locality metrics are taken over extents since 2026-10-22; the seventeen committed captures
    were taken before, kept no raw trace, and are judged on the block metrics that
    `--metrics-block` still computes. Each is retaken with its kit (the commands are in its
    mechanism's README; the traces whose read sizes follow `st_blksize` on an NFS mount), its
    metrics taken as `aeiou_metrics: 2` and its tolerances re-derived; when none is left at
    version 1, the block metrics, `--metrics-block`, and version 1 in `aeiou-trace compare` go.
35. **Locality by byte role (backlog, raised by the user 2026-10-22, `DESIGN_REVIEW.md` §3.79).**
    The extent metrics split by the role of each byte: a format's framing (headers, footers,
    directories, indexes) against its samples' payload, from the layout formula of
    `format.framing`, in the dry run and, from the same layout, in `aeiou-trace`. Payload
    reuse under `consume` is the epoch count; framing reuse is what rewards a footer or
    metadata cache, and a fit that matches the total re-read with the wrong mix would show.
    To design with the HDF5, Parquet, and TFRecord example cases (~~item 33, part 5b~~ item 36,
    their traces; 5b gave them untraced cases, the user's choice 2026-10-22).
36. **Trace the untraced containers (backlog, the user's choice 2026-10-22, `DESIGN_REVIEW.md`
    §3.78).** `hdf5_map`, `parquet_stream`, and `tfrecord_stream` have never been traced; each
    has one case of kind `unsettled` (`rgb256-rows`, `imagenet-rows`, `imagenet-records`)
    holding the placeholders its defaults held until 2026-10-22. Each gets a kit (h5py under a
    PyTorch `DataLoader`; `pyarrow.parquet.ParquetFile`; `tf.data.TFRecordDataset` with
    interleave, which needs TensorFlow installed), a capture taken as `aeiou_metrics: 2`, and an
    example case fitted to it, which replaces the unsettled one; item 35 is designed on the
    framing those traces show.
37. **A lognormal's sigma as an expression (backlog, the user's choice 2026-10-22,
    `DESIGN_REVIEW.md` §3.78).** A size distribution varies with every corpus, so its values
    are a case's (point 4), but a lognormal's `sigma` is a number in the contract (`ast.rs`,
    the schema, the builder's `lognormal`), where only `median` may be an expression; so
    `file_per_sample`, `parquet_stream`, and `tfrecord_stream` require the median and keep
    sigma 0.45 in the abstract. A contract change makes `sigma` an expression (the schema, the
    runner's AST and validator, the builder's `dists`, `rng`, `describe`, and `formats`), and
    then the three require it too.
38. **Job values to cases in the other abstracts (backlog, the user's choice 2026-10-22,
    `DESIGN_REVIEW.md` §3.78).** Item 33 moved the job and GPU values of the five
    input-pipeline abstracts to cases; `dcp_save`, `dcp_load`, `lmcache_local_disk`,
    `lmcache_shared_fs` (and its reader), `safetensors_load`, `diskann_build`,
    `diskann_search`, and `faiss_ivf_ondisk` still hold theirs as defaults, with no required,
    `scale`, or storage-reported parameter. Their job values are of other kinds (a model's
    size and its ranks, a session's prompts, an index's size and its queries), so each is
    taken with the user one mechanism at a time: which values are required, which is the
    scale, which reads follow `st_blksize`, and which case holds the values the defaults hold
    (a use case where they were measured as a job, `unsettled` where not). *2026-10-23:*
    `dcp_save` done (contract 0.12: a namespace's `st_blksize` for its write buffer; the case
    `four-items` of kind `unsettled`; `DESIGN_REVIEW.md` §3.78), then `dcp_load` the same day
    (its two read buffers the `st_blksize` of its input namespaces; the case `four-items`, what
    `dcp_save`'s writes at step 100); then the three LMCache abstracts the same day (contract 0.13:
    rule V3 judges a required `at` offset when it is given; the cases `chat-64-slots` of kind
    `unsettled`); then `safetensors_load` the same day (the model's shards and tensor table
    and the deployment's `tp`; the case `five-tensors` of kind `unsettled`); then `diskann_build`
    the same day (the base's shape, the build's configuration with the graph's `degree`, from
    which the disk index's sectors are derived, its graphs, and its compute; no case for the
    billion-point defaults, which the catalog cannot measure, the traced `sift1m-13-shards` the
    only one); then `diskann_search` the same day (the index as `diskann_build` names it, its
    sectors derived; the search's configuration, the deployment's `node_cache`, the load, and
    the CPU; no case for the defaults, the traced `sift1m-nocache` and `sift1m-cache` the only
    ones); then `faiss_ivf_ondisk` the same day (the index, the search's configuration with the
    lists' popularity, the load, and the CPU; the prefetch threads derived, `min_(prefetch_nthread,
    batch·nprobe)`; no case for the million-list defaults, the traced `sift1m-ivf1024-pq32` the
    only one). Every abstract outside the input pipelines is done. Open from it: the index
    file's stdio fills follow its `st_blksize` (glibc, [verify]), so `index_reads` derived from
    the arrays' sizes; nothing checks that the lists' 256 KiB slot holds the largest list.
39. **A use case per accelerator (raised by the user 2026-10-22, `DESIGN_REVIEW.md`
    §3.78 point 1; the Unet3D family built 2026-10-23, §3.81).** A job's compute per step and its batch (bounded by the accelerator's
    memory) are the model's on one accelerator, so a job is one use case per accelerator type
    (`unet3d-h100`, `unet3d-b300`, …): a shared job fragment and one accelerator fragment
    each, the case named from the `accelerator` attribute's vocabulary. Until then the
    traced example cases carry the abstracts' old `step_time`, said in their parameter files
    to be not fitted, and the number of cases multiplies when it is built. *Built for Unet3D
    (2026-10-23, §3.81):* `unet3d-a100`, `unet3d-h100`, `unet3d-b200`, sharing the mechanism's
    `jobs/unet3d.params.json`, the WG's reference values (its DLIO configurations: batch 7,
    636/323/162 ms, B200's the H100's halved, not measured; the sample sizes the WG's too,
    provisional until the KiTS19 corpus is measured). Open: B300 and any accelerator
    measured; the other jobs' accelerators as their use cases come.
40. **`aeiou cases`: the use cases a binary can run, listed and filtered (raised by the user
    2026-10-23, `DESIGN_REVIEW.md` §3.80; built the same day, reading the generated
    `workloads/catalog.json` rather than the records).** `--case NAME` runs a case by name, but
    nothing lists the names, and with item 39 the cases reach the middle double digits. A
    subcommand of the runner, reading ~~the records the release already ships under
    `workloads/` (`case.toml`, `mechanism.toml`, each case's `shape.json`, and
    `attributes.toml`)~~ the index `aeiou-catalog write` generates from them,
    `workloads/catalog.json`: `aeiou cases [--where ATTR=VALUE | --where ATTR!=VALUE]... [--json]`
    lists every case, of every kind, one row each (`mechanism/case`, kind, GPU counts,
    category and class, its one-line doc); `--where ATTR=VALUE` keeps the cases with that value
    and `--where ATTR!=VALUE` drops them (`--where 'kind!=example'`), several ANDed; an attribute
    or value outside the dictionary is a usage error naming the valid ones. `aeiou cases NAME`
    shows one: its attributes, its GPU counts, what its lock fixes and what a submission must
    still give, its `DESCRIPTION.md`, and the `datagen` and `run` commands that run it.
    `aeiou cases --help` gives the syntax and the whole vocabulary, every attribute and its
    values, curated and measured, read from the tree's `attributes.toml` (no explanation of
    the terms: `CATALOG.md` has them). A `--case NAME` that matches nothing names the near
    matches and `aeiou cases`; the `--case` help of `dry-run`, `datagen`, and `run` points at
    `aeiou cases` and `CATALOG.md`. ~~A test holds `aeiou cases --json` to what `aeiou-catalog`
    loads from the committed tree, so the two readers of the records never drift.~~ One reader
    of the records: `aeiou-catalog check` holds `catalog.json` to them.

41. **A schema for the report, and a reader (raised by the mlpstorage planning note,
    2026-10-28).** `runner/REFERENCE.md` §12 is prose; a driver that scores from
    `--report-json` wants a contract. `schema/report.schema.json` for `aeiou_report: 1`, held to
    `report.rs` by a test that validates every report the suite writes and fails on a schema
    field no report exercises; the report gains the binary's build identity (`runner`
    {version, commit, features, target}) beside its version; §12 gains a table of the fields a
    scorer reads (identity: abstract sha, case sha, workload id, fingerprint; the figures:
    `result.busy_fraction`, `result.min_gpu_busy_fraction`, elapsed, bytes and rates, latency)
    and what each means against DLIO's AU (compute ÷ (compute + stall), every take but the
    first of each instance, under `time_scale`). The Python package gains `aeiou.report`, a
    reader that validates a file against the schema and returns it as typed objects, so a
    driver imports a reader instead of writing one; it reads a file, never links the runner
    (§8, driver integration).

42. **`aeiou report check`: does this report belong to this lock? (2026-10-28).** `PROJECT_BRIEF.md`
    §8 point 14 has the WG validating an uploaded report by recomputing the lock's hash, the
    workload id, and the fingerprint at the report's aeiou version. The definitions live in the
    runner, so the recomputation does too: `aeiou report check REPORT [--case-file LOCK]`
    re-derives all three from the report's embedded lock and abstract hash (the abstract found
    under `--workloads`), compares them with what the report claims, checks the constraints'
    recorded numbers against their expressions, and exits 0, 1, or 2 by the usage frame. A
    report of another runner version says so and stops.

43. **The dry run's projection as a document (2026-10-28).** `dry-run` prints each dataset's
    samples, files, directories, and bytes, the op counts, and the compute per GPU, and writes
    none of it as JSON (`--metrics-json` is the locality metrics alone). A sizing loop in a
    driver re-implements the math today. `dry-run --report-json FILE` writes the report's
    identity block, `datasets[]` with their projections, the per-kind op counts and bytes per
    GPU and in total, the fingerprint, and the compute at `time_scale` 1 (the shortest a run
    can take), as `aeiou_report: 1` with `scope: dry-run`, under item 41's schema.

44. **A case names its headline figures (2026-10-28).** The report gives bytes, ops, and
    latency; a benchmark's headline is a figure in its own units: samples per second for a
    training case, tokens per second for a KV cache, queries per second for a search, GB/s for
    a checkpoint. aeiou computes none of them because it never interprets content, but each is
    a formula over `result.` names and parameters (`tokens_per_s = bytes_read / bytes_per_token
    / elapsed_s`, with `bytes_per_token` a parameter of the model fragment). A `results` table
    in the parameter file and the lock declares named expressions; the runner evaluates them
    after the run, reports them under `result.derived`, prints them in the text report, and
    lets a rules file's constraints name them. A figure's units and its formula are the case's,
    published in the lock, never the runner's.

45. **The WG's closed workloads in the catalog (2026-10-28; hardware-bound in parts).** What
    the planning note finds missing, each in the catalog's own shape, the values from the
    WG's DLIO configurations and tables where no job is measured, every fragment naming its
    source as `unet3d`'s do (§3.81), the cases `unsettled` until traced:
    (a) RetinaNet is a job over `file_per_sample`, not a new abstract: a job fragment
    (OpenImages JPEGs, the WG's 322,957 B mean) and accelerator fragments `b200` and `mi355`,
    the cases `retinanet-b200`, `retinanet-mi355`; the vocabulary gains `mi355` (and `b300`
    when measured). A trace of the real loader (torchvision's detection reference) later,
    never of DLIO.
    (b) The Llama checkpoints over `dcp_save` and `dcp_load`: job fragments `llama3-8b`,
    `-70b`, `-405b`, `-1t` at the WG's TP/PP/DP and Table 2 sizes. `item_bytes` is one list
    for every rank (1/G of each tensor, FSDP-shaped), and pipeline stages hold different
    layers, so first check whether the WG's sharding is expressible; if not, a small contract
    change (an item list per rank class, indexed by an expression of `gpu_id`) comes first.
    The WG's benchmark wrote checkpoints with its own writer; whether DCP's op stream is what
    it scores is the WG's decision, raised to it.
    (c) KV cache: a model fragment `llama3.1-8b` (layers, KV heads, head size, dtype, so
    `chunk_bytes` and `bytes_per_token` follow) and the WG's loads as cases of
    `lmcache_local_disk`; its 300 s become a request count (wall-clock phases are never
    CLOSED). The holds fitted at the reference engine when the GPU can hold the model (item 24).
    (d) Milvus: trace Milvus standalone with a DiskANN (Knowhere) index under a query load on
    the development box and compare its metrics with `diskann_search`'s; a case
    (`milvus-1m-1536`) if the op stream matches, a mechanism `vector-search/milvus-diskann` if
    its segment loading makes it another workload. Recall stays out: the fit's `hops` and
    `beam` carry the work a recall target costs (a sentence in the vector-search README).

    *Scope, 2026-10-29 (`DESIGN_REVIEW.md` §3.88, the user's point):* vector search is three
    sub-workloads, the index traversal, the *leaf fetch* (an ANN query returns the neighbors'
    identities; the application reads the neighbors themselves, chunks or media, from a store
    that is never inside the index: an `indexed` dataset, POSIX or object), and the rebuild,
    merge, or compaction that incorporates new content, a sub-workload the lookups invoke
    periodically as training invokes checkpointing, with ingest's read pass over the new media
    before it. For an index held in memory the leaf fetch is the whole storage workload. Whether
    the leaf fetch and the rebuild are inside the WG's vector-search benchmark is the WG's
    scope call (its Milvus harness measures queries only); aeiou models the lifecycle, cases
    mark what a division scores, and §8 records the decision when made. The Milvus trace of
    (d) then shows traversal, payload fetch, and compaction together, and needs the object
    request log of item 49 beside the strace.

    *Measured 2026-10-28 (`DESIGN_REVIEW.md` §3.46):* the fork's Unet3D op stream is not the
    loader's (one whole-file read per sample, nothing decoded, the corpus re-read 1.3–2.5× by
    its worker pre-warm), so a case here and the WG's DLIO result of the same name are not
    comparable, and results across a transition to aeiou will very likely not be, aeiou being
    the more accurate of the two. A configuration supplies parameters, never an op stream; a
    case's `unsettled` lifts only on a trace of the real application.

46. **The integrator's page (2026-10-28).** One manual page, `aeiou-integration(7)`, for a
    driver that runs aeiou as subprocesses: the process model (one process per phase, never
    linked), the launchers (`aeiou-launch`; `mpirun`, `srun`, and `pdsh` one-liners that set
    `AEIOU_RANK` from the launcher's rank variable, tested under a fake launcher), the exit
    codes, the checkpoint write-then-restore sequence, the datagen manifest and `--resume`, what
    a scorer reads (item 41's table), and an example rules file under `schema/examples/rules/`
    carrying constraints of the shape the WG's would (5× memory, 500 steps), labelled an
    example and not the WG's. Held to the tools' `--help` like every page.

47. **A trace's digest, and what an abstract discloses (2026-10-28).** Two deterministic tools
    under the fitting of item 48, each useful alone. `aeiou-trace digest TRACE --root DIR`
    writes the structural summary a fit is made from: path templates with every literal
    turned into a field (`train/{a:05}/sample_{b:09}.npz`, with how many values each field
    took and their range), the processes and threads and what each does in order (phases of
    open-read-close, the think time between, the fan-out), the op mix, size and offset
    histograms per template, the reuse distances, and the notes the metrics already carry
    (truncation, I/O invisible to strace); no raw path unless `--paths` asks. `aeiou-catalog
    disclose CASE` (and the same over a loose abstract and params) lists every string and
    number that would leave with a contribution, by kind: path literals, the prose (`doc`,
    `DESCRIPTION.md`), parameter values, provenance (hosts, roots, dates, logs under
    `captures/`), so a contributor's reviewer reads exactly what is published; `--rename` maps
    path literals to generic names and keeps the mapping locally (the fingerprint changes, the
    shape does not).
    *2026-10-28, `DESIGN_REVIEW.md` §3.86:* what the digest can and cannot decide without a model
    of the application, recorded before its design; the next item to build. *2026-10-29,
    §3.87:* `aeiou-trace digest` built: templates, lanes and phases, a verdict per site on
    selection (`consume` or `choose`, by passes and the birthday bound under `--candidates`),
    order, and size, each `undecided` with what settles it, and the `open` list ranked by the
    candidates' divergence at `--epochs`. Decided the same day: the open questions are worth
    asking the application's owner, with provenance, as behavior not constructs, and with the
    capture that would settle each; ~~the digest never asks~~. *2026-10-29, §3.88 (the user's
    plan, which §3.87 had departed from):* the digest asks, on a terminal, facts first (a field's
    range at scale, a phase's repeat count at scale, generic across training, checkpointing,
    vector search, and KV cache) and then the open questions in rank order, checks every answer
    against the trace, writes the answers under `asserted` with provenance and as an answers
    file beside the document, and runs without a terminal from `--answers FILE` or with every
    item left open; the trace is a file, never stdin. Its site model gains a keyed-lookup kind
    (KV cache, RAG leaf fetch), a within-one-open kind (index traversal), and a bulk-write kind
    (checkpoint, build, merge). Beside it, `applications/`, records of traced applications by
    product directory and version file (signatures, storage-relevant options, defaults as
    assertions weaker than the owner's, dated sources), so a user of a product answers from a
    list. *Built the same day:* the asking (`--answers`, `--no-questions`, `--answered-by`,
    `asserted` and `rejected` in the document, the answers file) and `applications/` with seven
    records (`--applications`, `--application`; `aeiou/applications.py`, Rust since 2026-10-31), and the three site
    kinds beside the set, keyed, bulk write, and within one open, each with its facts (keys and
    requests; values per field, ranks and steps; size and queries) and questions, and the
    object-log readers of item 49. *Revised 2026-10-30 and built 2026-10-31 (§3.88):* keyed is
    writes and reads interleaving; all writes before all reads two sites; a restore kind; a bulk
    write's saves, retention, and `label` (checkpoint or output), the facts `keep` and `writers`,
    the `field:` questions gone. Not yet: digest-to-digest comparison, `aeiou-catalog disclose`. *2026-10-29, `DESIGN_REVIEW.md` §3.89 (designed with the user):* one
    observation document, `aeiou_digest: 2`, the digest's structure and the metrics of the same
    events, `aeiou-trace metrics` retired and `dry-run --metrics-json` writing the same document;
    `aeiou-trace compare` over two of them, structure by equality and numbers by the tolerance
    classes, templates paired by text, shape, or `--pair`, the runner's own I/O set aside; the run
    side from `dry-run --emit-trace` (item 49) or an strace of `aeiou run`; and the trace-side tools
    in Rust as the binary `aeiou-trace`, streaming, since a real trace runs to several GiB and the
    Python parser holds every event. Build order in §3.89. *Stage 2 built 2026-10-31:* the digest
    in Rust (`runner/aeiou/src/digest.rs`), `aeiou_digest: 2` with its `metrics` section and each
    site's `facts`; ~~the interview and the records stay Python until stage 3 (§3.89).~~ *Stage 3
    built the same day:* the interview, the records (built into the binary), and the answers file
    in Rust (`interview.rs`, `applications.rs`), the Python's `applications.py` and interview gone;
    ~~next, stage 4, `compare` over two version-2 documents and `dry-run --emit-trace`.~~ *Stage 4
    built 2026-11-01:* `aeiou-trace compare` in Rust (`compare.rs`), the numbers held to every
    comparison the Python's tests made and two digests' structure compared as designed, `--pair`
    and `--json` added; `aeiou dry-run --emit-trace` (`emit.rs`). Not done in stage 4:
    `aeiou-trace metrics` is not retired, since ~~its strace-only options (`--instance-root`,
    `--chain-gap-us`, `--metrics-block`) have no place in the digest yet and~~ the committed
    captures are metrics documents until stage 5 rewraps them (the three options on the digest
    since 2026-11-02, its `metrics` section the strace's own reading, `DESIGN_REVIEW.md` §3.89); nor does `dry-run --metrics-json`
    write a digest (its numbers are compared as a metrics document's). Next, stage 5.

48. **A fitting skill: from a trace to a fitted abstract and case (raised by the user
    2026-10-28).** A skill in the open skill format (`skills/fit-abstract/SKILL.md` and its
    references, plain Markdown plus the suite's CLIs, usable by any capable LLM, not one vendor's),
    that takes a user from the strace recipe to a catalog-shaped directory: the digest (item
    47), the choice of a committed mechanism from the catalog's vocabulary or the authoring of
    a new `<name>.py`, the parameters (the numeric fit by `aeiou-fit`, the deterministic
    fitter the kits' `fit.py` scripts become; the LLM decides structure, never numbers),
    `dry-run --metrics-json` against `aeiou-trace metrics` under `compare --judge`, the
    scaffold of the mechanism or case directory (`aeiou-catalog new`), `DESCRIPTION.md`, and
    the disclosure report. Its evaluation set is the committed captures: given a capture's
    trace, the skill must pick its mechanism and reach a fit the judge accepts; measured, and
    kept green as the catalog grows. Three audiences, one artifact: a contributor who will not
    release a trace but will release its abstract; a vendor recreating a customer's workload in
    its lab; and this repository, which grows its library by it. A contributing guide names
    the path. Names stay generic (the skill is "fit an abstract to a trace", not a vendor's or
    the WG's). *2026-10-29 (`DESIGN_REVIEW.md` §3.88):* two pathways, one set of tools: a
    customer who keeps an LLM away from its traces follows the same commands from a written
    procedure page the skill follows too; the skill owns judgment and the authoring of a new
    mechanism, never mechanics, and the digest's questions are the digest's (§3.88), asked by
    the tool, in the skill's voice only from the tool's own list.

49. **Is aeiou a "scalable trace"? (raised by the user 2026-10-28).** A fixed trace is one
    sequence of one length; replicating it in parallel amplifies its hot spots, in series
    repeats accesses the application never made, and offset in time poses the offset as a free
    parameter. aeiou generates a stream of any length over any GPU and host count from
    positional randomness and a Feistel split, so no replication artifact exists by
    construction, and it can emit that stream as a trace (`dry-run --gpu G --steps A..B`). The
    item is the honest case for the term and what it lacks: a note defining the term (a
    model fitted to a trace that regenerates the trace's per-instance statistics at any
    scale, with the aggregate scaling as the modelled sharing says and never as the copy count
    says), the measured evidence (one workload on the loopback mount: the `trace` node at one
    instance against its abstract at one, then N copies of the trace offset in time against
    the abstract at N, their merged locality metrics side by side, which needs `aeiou-trace
    metrics` over several traces as one stream), a scale-invariance check in `aeiou-catalog`
    (a case's per-GPU metrics at G and at 8G agree), and `dry-run --emit-trace FILE` in the
    `aeiou_trace` format so the scaled trace is an inspectable file (*built 2026-11-01*,
    `DESIGN_REVIEW.md` §3.89 stage 4: the instances interleaved, a lane per context). The gaps the note must
    state: every fit is validated at one scale (a trace of one application at two scales,
    fitted at one and predicted at the other, has never been taken); shared structure (a
    search index's hot nodes, a KV cache's shared prefixes, the step barrier) is amplified
    exactly as the abstract models it, which is fidelity when the real system shares it and
    only as right as the model; checkpoint sharding scales by parameter math, not by itself;
    the coordinator has met only `localhost`; and the client-cache topology of emulated GPUs
    is not scaled (item 28, the WG's 5× rule). *Added 2026-10-29 (`DESIGN_REVIEW.md` §3.88):*
    object-store I/O has no strace; a capture there needs a request log (MinIO audit, Ceph RGW
    ops log, S3 server access logs or CloudTrail data events, a logging proxy in a lab) in a
    format `aeiou-trace` reads, one line per request as issued, on a wall clock the strace's
    timestamps merge with; a gap the note states until the reader exists. *Built the same day
    (`aeiou/objectlog.py`; Rust since the port of §3.89 the same day, `runner/aeiou/src/objectlog.rs`):* MinIO's audit log and S3's server access log read by `metrics`,
    `export`, and `digest`; not yet Ceph RGW's ops log, a proxy's log, or the merge of an
    object log with an strace on the clock.

50. **The catalog's speed (raised by the user 2026-10-29, queued, not started).** `aeiou-catalog
    write` takes about 36 minutes, and porting the trace tools to Rust made them 13 to 15 times
    faster (`DESIGN_REVIEW.md` §3.89), so the user asked whether a Rust catalog would take three.
    It would not: the catalog's time is the runner's `aeiou dry-run --metrics-json`, already
    Rust, one run per case and scanned length (one LMCache case held a core at 94 % for over six
    minutes while `aeiou-catalog` itself sat at 0.4 %). The levers to measure, when the user
    takes this up: a profile of the metrics walk on the slowest case (one thread per case, the
    reuse bookkeeping per byte range); how many dry runs a case needs (the LMCache length scans
    took the catalog from about 15 to 36 minutes, 2026-10-23); and the tail once one long run is
    left (metrics sampling, shorter check lengths for the heaviest cases).

51. **Phases detected in a trace (raised by the user 2026-10-30, queued, not started).** A
    digest treats a trace as one phase (`DESIGN_REVIEW.md` §3.88, revision of 2026-10-30), but
    real traces hold several: training's restore, epochs, periodic evaluation passes, and
    periodic checkpoints; a vector database's index load, searches, and periodic index rebuilds
    or compactions; a KV cache, whose mix varies but never switches into a new regime and back.
    The user's approach: a rolling window of significant size over the broad I/O mix, a phase
    change being an abrupt change (a step, never a drift; a KV cache's warm-up is a drift) that
    lasts and, for a periodic phase, returns to the original mix. Points agreed: segment first
    by which templates are active, the op mix (bytes read and written, request sizes, metadata
    per data op, creation rate) as the second signal for phases that change the mix within a
    template; windows sized in the application's own steps, not seconds, with a minimum dwell;
    the outcomes named periodic (A B A B: the period and duty as measured facts), sequential
    (A then B: each side digested on its own), or stationary; the owner confirms each label, as
    the digest asks. Bounded memory, so it streams over 7 GiB traces. Needs a real trace with
    checkpoints or evaluation in it to design against. *The smaller phases to look for, listed
    the same day (from understanding, not yet read from sources; each checked before it becomes
    a rule or a record):* training: the restore at the start, epoch boundaries (workers
    re-forked and the corpus listed again, a metadata burst; the §3.89 spike saw the re-fork),
    evaluation passes, a first-epoch cache of the corpus to local disk (one-way), and periodic
    small appends of logs and metrics (continuous, not a phase); the checkpoint itself: the save
    burst, the fsyncs, the commit marker, then retention's deletes; vector search: the index
    load and cache warm-up at the start (DiskANN's node cache), ingestion mixed with search,
    periodic segment flushes, compactions, and index builds on sealed segments (Milvus),
    optimizer runs and snapshots (Qdrant); DiskANN's build, a chain of one-way phases (sample,
    partition passes, shard builds, merge, disk layout) the abstract already models; KV cache:
    warm-up (a drift, so no phase), the local tier filling so every write evicts (abrupt and
    one-way, so sequential), and the model load and engine start; RAG: periodic re-ingestion
    and re-indexing. *Phases in the catalog (agreed with the user 2026-10-31):* a phase item 51
    finds is in the trace's digest; it reaches a case's description (`DESCRIPTION.md`,
    `CATALOG.md`) through the abstract's `phase` nodes, which `describe` already renders ("in
    phase `restore`, ..."), once a fit models it. Three parts beside the detection, the first
    two for the accuracy of the abstract: (1) per-phase measured lines, the bytes, op mix, and
    request sizes of each phase and a periodic phase's period and duty, where a description
    today gives one "Measured by dry run" line over the whole run, which averages a training
    job's epoch reads with its checkpoint bursts; this needs the dry run's metrics split by
    phase, which the runner does not do yet; (2) the catalog's `phase` attribute
    (`workloads/attributes.toml`: `training`, `inference`, `indexing`, `preparation`,
    `production`, when in a system's life a mechanism's I/O happens) renamed (`stage` or
    `lifecycle`, to choose) before a description puts both on one line, the records and docs
    that use it with it; (3) a trace described in prose from its digest, in the catalog beside a
    capture's "Traced: ..." line, before any abstract is fitted to it, since describing a trace
    has value of its own (the user's reasoning, 2026-10-31: a researcher who wants to understand
    an application and has no interest in running an abstract gets that from the trace alone).

52. **All-rank traces of a scale-out run (raised by the user 2026-10-30, queued, not started).**
    Ground truth for a distributed application is every rank's trace at once, aligned on
    timestamp. A degenerate single-node run shows every operation the scale-out deployment
    performs, but not which node issues it, so locality of operations to nodes (and the
    simultaneity of a checkpoint's burst across nodes, `DESIGN_REVIEW.md` §3.88 revision point
    4) is seen only in the scale-out run. The item: the capture (one strace per node, or per
    process, with the clock's quality recorded: NTP, chrony, or PTP, against the millisecond
    simultaneity it must resolve; an anchor event every node performs, such as a stat of an
    agreed path, to align on beside the clock), and the reading (`aeiou-trace` over several
    traces as one stream, a lane per host and process, which item 49 also needs, and the merge
    of an object log with an strace on the clock). *What one node can and cannot stand in for
    (discussed with the user 2026-10-31):* merging the per-process traces of a scale-out run on
    the clock gives the same operations, each process's in the same order, as the same
    processes on one node only for an application that splits work by global rank alone and
    decides nothing by timing (the rank-only model the runner assumes). It differs where the
    application acts once per node (`local_rank == 0` work such as a dataset or model download,
    a node-local cache in `/dev/shm` or on local NVMe; MosaicML Streaming partitions samples by
    node; checkpoint layouts that depend on ranks per node, from memory, medium confidence, to
    be read from source), where work goes to whoever asks first (shared queues, elastic
    sharding), and in node-local paths and PIDs, which repeat across hosts, so a merge names a
    lane by host and process and a local path by host. And one node distorts timing, the point
    of the item: its processes share the CPUs, memory bandwidth, NIC, and storage client, so a
    burst's start spread and length and the ranks' interleaving all shift; a GPU job often
    cannot run at its rank count on one node at all. So the tooling (several traces read as one
    stream, host-qualified lanes and local paths, alignment on an anchor) is built and tested on
    one node, a trace split into hosts its fixture; the capture is multi-node. A topology check
    beside it: the same small job traced as 1 node of 8 processes and as 2 nodes of 4, the
    digests compared; a structural difference (a per-node template, another site kind, other
    counts) shows the application depends on topology, which a single-node trace hides, and
    none supports the rank-only model for that application. Two nodes are enough for it.

53. **A digest redacted for publication (queued by the user 2026-10-31, not started;
    `DESIGN_REVIEW.md` §3.90).** An owner who will not publish a trace, since it names their
    files, may publish its digest, from which a benchmarking organization fits an abstract; so
    a published digest must carry the structure a fit needs and nothing that identifies the
    owner's data. Today's digest is not that: `trace` (the trace file's path), `root`, `by`
    (`user@host` by default), every free-text `note` of the answers, the raw paths under
    `--paths`, and the literal text between the fields of every path template (a project's,
    a user's, a dataset's name) all survive into it. The item: a redaction mode of `aeiou-trace
    digest` (or a subcommand over a written digest) that replaces each template's literals with
    tokens consistent across the document (so two templates that share a directory still share
    it), keeps the field kinds and widths, the site kinds, every verdict, fact, and number;
    drops or generalizes `trace`, `root`, `by`, and the notes (the owner's choice per field, a
    note kept only when asked); keeps `trace_sha256` only on request (it links the digest to a
    trace the owner holds, which may be wanted); and lists everything that remains, every
    string and number, as item 47's `aeiou-catalog disclose` does for an abstract. A test holds
    that no raw path string of the trace survives in a redacted digest and that a fit's inputs
    (stage 4's `compare` of the digest against a run's) are unchanged by the redaction.
    Questions to settle: whether a template's literal that names a well-known format or
    framework (`.distcp`, `.safetensors`, `.npz`) is kept, since the application records match
    on it and it is not the owner's; whether timings (think time, a save's interval) can
    identify an owner; and the application's identity, which a record match adds and an owner
    may still want left out.

54. **What `strace` cannot see: an eBPF capture beside it, and the instructions for a capture
    (raised by the user 2026-10-31, queued; taken up by the user 2026-11-03 for a capture with
    both POSIX calls and object requests, measured by a spike the same day, `DESIGN_REVIEW.md`
    §3.96; decided; phase 1, the capture under strace, and phase 2, the HTTP-head shim and the
    anchors, built 2026-11-03).** An `strace` misses three kinds of
    I/O the abstracts model: the accesses through a mapping (it sees `mmap`, `madvise`,
    `msync`, `munmap`, never the loads; `builder/REFERENCE.md` §7), the operations of an
    `io_uring` (it sees `io_uring_setup`, `io_uring_register`, and `io_uring_enter`, never the
    SQEs, and under `SQPOLL` not even the enter), and object requests (HTTP over a socket).
    The user was reminded that eBPF tracing in the kernel can report mapping and `io_uring`
    activity; `DESIGN_REVIEW.md` §3.42 already named an eBPF capture as "a second front end to
    the same `Metrics`; none is written". This side's first assessment, from memory, each claim
    to be checked against the kernel's source for the kernels targeted before anything is
    built (the confidence after each):
    - **Mappings.** A probe on `filemap_fault` gives the file and page offset of each fault
      that reaches the page cache, and the tracepoint `filemap:mm_filemap_add_to_page_cache`
      each page as it enters the cache (buffered reads too) (medium). What it shows is the
      kernel's I/O, not the application's calls: a warm cache hides faults, and fault-around
      and readahead make the unit larger than the touch. So a capture needs a cold cache (or a
      dataset larger than memory) and says which, and a site fitted from it is marked
      kernel-observed, weaker evidence than a call, beside the principle that an abstract is
      the application's call stream.
    - **`io_uring`.** The tracepoints `io_uring:io_uring_submit_req` (named
      `io_uring_submit_sqe` before about 6.0) and `io_uring:io_uring_complete` give the
      opcode, `user_data`, and the result, not, as far as recalled, the descriptor, offset, or
      length (medium); ~~those need a probe on `io_issue_sqe` reading the request, whose layout
      changes with the kernel version, so BTF and CO-RE.~~ *Corrected 2026-11-04
      (`DESIGN_REVIEW.md` §3.98):* `io_uring_submit_req`'s argument is the request after its
      `prep`, read at offsets taken from the kernel's BTF at load; no CO-RE.
    - **Object requests (still open).** Server-side logs are read already (MinIO audit, S3
      server access, `runner/aeiou/src/objectlog.rs`). On the client, probes on OpenSSL's
      `SSL_write`/`SSL_read` (as bcc's `sslsniff`) see the plaintext request line, method,
      key, and `Range`, but only for a client linked to the system's OpenSSL: the AWS CRT
      (`s3torchconnector`, Mountpoint) uses s2n-tls on Linux, and Go clients carry their own
      TLS (low to medium; each client's source read first). An SDK's debug log, and a packet
      capture against a plain-HTTP test endpoint, are the other leads. *Measured 2026-11-03
      (§3.96):* a libssl uprobe and an `LD_PRELOAD` shim both gave boto3's request heads over
      TLS under the system's Python (not under uv's standalone Python, whose OpenSSL is static),
      at +1 to +2 %; the CRT's s2n-tls is inside its own `.so`, reached over plain HTTP only.
    - **The capture.** `strace` stays the primary record (its `-yy` paths are better than
      what a probe resolves), and the eBPF log is a second input joined to it by thread and
      time, as the object logs are a second input today. *Proposed 2026-11-03 to change, not
      decided (§3.96):* eBPF the primary record where the owner can grant it (0.2 to 0.4 µs a call against
      strace's 88 µs here; paths copied by the probe, descriptors resolved after the run from
      the recorded opens, as the export does), ptrace where not. `strace -ttt` is wall-clock and BPF
      the monotonic clock, so the capture records their offset. It needs root, or `CAP_BPF`
      and `CAP_PERFMON`, and `bpftrace` or a small tool of ours, which some owners' hosts and
      containers will not allow; its overhead is far below `strace`'s.
    - **The instructions.** A trace's owner today is given one `strace` line (the man page of
      `aeiou-trace`). The item gives them a capture kit: one script that checks the kernel and
      the tracepoints present, starts `strace` and the probes together, records the kernel
      version, the clock offset, and whether the cache was cold, and says what the capture
      could not see; the instructions as a page an owner follows without us. The kit is worth
      writing for `strace` alone before any probe is added. *Noted 2026-11-02 (the user):* the
      instructions start the command under `strace` (`strace -f … -o FILE COMMAND`), never
      attach to it running (`strace -p`), which loses the opens made before it attached and
      leaves a trace whose export must synthesize them (§3.92); and add `%network` when the
      application passes descriptors over a socket, so the `recvmsg` that delivers one is seen.
    Questions to settle: whether a mapping's faults are evidence enough for a site of an
    abstract, or only a check on one; which kernels the probes must support; whether the
    digest's record of what a trace cannot see (an application record's blind spots) becomes
    a record of what this capture did see.
    *Revised 2026-11-03 (§3.96, the spike's measurements; this side's proposal, the user to
    decide):* a capture tool of the suite that starts the command and takes four sources,
    chosen per host and client, all converted afterward to the strace text the suite reads:
    (1) eBPF where root or `CAP_BPF` and `CAP_PERFMON` are granted, the default: system calls
    on raw tracepoints in variable-length binary records (a path copied by the probe at entry,
    retried at exit, every failure counted), a binary reader (bpftrace's text reader sustains
    about a million events a second and loses half beyond that, whatever its buffer), a buffer
    sized to seconds of the peak, every loss counted by CPU and time range, libssl uprobes for
    request heads, and later the mapping faults and `io_uring` requests above; (2) ptrace
    where not: strace with `--seccomp-bpf` (missing from the man page's line, 3.6 times slower
    without it; added to every capture line the same day), or a binary tracer of ours (1.6 to
    5 times faster than strace); (3) the
    `LD_PRELOAD` shim for request heads where the client loads a shared `libssl` and eBPF is
    not granted; (4) the store's log for the rest, a reader per log format, aligned by anchor
    requests. `-s 512`, proposed earlier the same day, is withdrawn (every file buffer printed, the
    trace 5.7 times larger). The record of a capture: kernel, tools, command, each mount's
    filesystem, endpoints and scheme, cold cache or not, the clocks' offset, the buffer size,
    every loss, and what it could not see. Timing parameters (think time, a save's interval)
    come from an untraced run, never a traced one's gaps. Open: aya or libbpf-rs; a ptrace
    tracer of ours or strace; the shim in one library with item 55's `fstat` shim; the anchor
    requests (a 403 in the owner's audit log) or the owner's own command; and the spike's
    scripts (`spikes/2026-11-03-capture/`) re-run on a real Linux client node before a choice
    rests on a ratio (the numbers are WSL2's, whose cross-CPU ptrace wakeups cost about 50 µs).
    *Decided by the user the same day:* strace with `--seccomp-bpf` (a ptrace tracer of ours
    only if a need appears); the unsigned anchor `HEAD`s; the real-node measurements when a
    node is available; aya for the eBPF side (the BPF object embedded in `aeiou-trace`; ~~CO-RE
    for kernel structures later from a C-built object, to be checked~~ kernel structures read at
    offsets from the kernel's BTF, no CO-RE, checked 2026-11-04, §3.98); the shim one crate of
    interposition machinery built as two libraries, the capture's (observes only) and item 55's
    `fstat` one (changes what the application sees).
    *The tool's design proposed and debated the same day (§3.96, "The capture tool's design";
    recorded before a context clear, not approved, nothing built until the user says
    proceed):* `aeiou-trace capture ... -o DIR -- COMMAND`, a capture directory with
    `trace.txt`, `capture.json`, and the sources' raw files; the command never runs as root
    and nothing prompts (file capabilities an administrator grants once on a node-local copy,
    or a `NOPASSWD` rule for an argument-free helper under `sudo -n`; descriptors handed back,
    capabilities dropped before the command starts); several nodes through `aeiou-launch`
    (password-less ssh, the baseline) and the runner's coordinator for a start barrier, clock
    offsets, and a capture-set record, or inside the application's own launcher; the checks
    the first step of every capture, compared across nodes, any failure starting no node,
    waived only by name (`--accept`), run again at the end. Phases: strace and everything
    that holds under it; the shim and anchors; eBPF with aya (nightly and `bpf-linker`, the
    user to approve); uprobes, faults, `io_uring`.
    *Approved by the user 2026-11-03 with the proposed defaults (`%network` always, a `--root`
    or `--endpoint` required, `--cold` a sync and drop, `CAPTURE.md` in the tarball, the
    readers taking `DIR/trace.txt`), and phase 1 built the same day* (`aeiou-trace capture`,
    `runner/aeiou/src/capture.rs`, its tests `tests/capture.rs`; `aeiou-launch` puts the rank
    flags after `capture`; the coordinator's clock rounds, `Server::clocks`; `aeiou-trace(1)`
    capture, `CAPTURE.md`). Not in phase 1, and not implemented yet: a node's interruption
    stopping the others (each node stops on its own signal, which `aeiou-launch` sends to every
    rank), the cross-node comparison made afterward from records taken without a coordinator,
    and the sources' raw files (none under strace). ~~Next: phase 2, the HTTP-head shim.~~
    *Phase 2 built the same day, as proposed and approved* (`DESIGN_REVIEW.md` §3.96 "Phase 2
    as built"): `--requests shim|none` (`shim` with an `--endpoint`) preloads
    `libaeiou_capture.so` (`runner/capture-shim/`, its machinery `runner/interpose/`, shared
    with item 55's library to come), which records request and response heads through a
    shared libssl or libc's socket calls into a mapped log (`requests.bin`) with no system call
    per request and no credential kept; the tool writes `requests.jsonl`, names the processes
    the library missed, and sends the anchor `HEAD`s; `objectlog.rs` reads `requests.jsonl` as
    a third format, and the readers take the capture directory; the tarball's `lib/` holds the
    library. Not in phase 2: the join of requests to calls (items 49, 61). ~~Next: phase 3, eBPF
    with aya (nightly and `bpf-linker`, the user to approve installing them).~~ *Phase 3
    decided by the user the same day* (`DESIGN_REVIEW.md` §3.96 "Phase 3 decided"): a dated
    nightly (`runner/capture-ebpf/rust-toolchain.toml`) and `bpf-linker` pinned in `dev.sh`,
    the object built by `aeiou`'s `build.rs` in every build; file capabilities or root, the
    `sudo -n` helper phase 3b; one ring buffer, `--buffer` 256 MiB by default, to be revisited;
    any loss loud (a suggested `--buffer`, `complete: false`, the readers warning and marking
    what a gap left unresolved). ~~Not implemented yet: phase 3 itself, being built.~~ *Phase 3
    built the same day* (§3.96 "Phase 3 as built"): `--source ebpf|auto`, `--buffer`, `--convert`,
    `runner/capture-ebpf/`, `ebpf.rs`, `ebpf_text.rs`; the same exported ops and metrics as
    strace on the equivalence test; and a strace-parser bug it found (calls shorter than 40
    columns dropped) fixed. Not implemented yet: the `sudo -n` helper (phase 3b), uprobes,
    faults, `io_uring` (phase 4).
    *Phase 4 and the trace format decided by the user 2026-11-04* (`DESIGN_REVIEW.md` §3.98,
    after a prior-art survey, the hooks per mechanism from the kernel's BTF, and an experiment
    of POSIX AIO under strace): one capture format, strace's text extended by records strace
    never prints (a reserved `aeiou_` prefix, a keyed subset of strace's syntax, a versioned
    header line, completions paired to issues by `user_data` or the request's address, object
    requests in the same stream), so the strace source's output is a pure subset of eBPF's;
    JSON lines an export only (`aeiou-trace export --jsonl`, with a schema). The probes in
    order: `io_uring`, mapping faults, libaio's real completion, POSIX AIO uprobes, libssl
    uprobes, cuFile and libnfs. Also decided and built the same day (§3.98): no merged trace of
    several nodes (the set's directory the unit, readers to merge as they stream, events closer
    than the clock bound in unknown order, nodes grouped by a set digest; not implemented yet);
    and the eBPF tree's two defects fixed (an exited process leaves it, so a recycled pid is
    not traced; a child a full tree cannot hold is counted and makes the capture incomplete).
    ~~Not implemented yet: all of phase 4, phase 3b, readers of a set, the set digest.~~
    *Phase 4 step 1, `io_uring`, designed with the user and built 2026-11-04* (§3.98 "Phase 4
    step 1 as built"): probes on `io_uring_submit_req` and `io_uring_complete` (a program for
    each tracepoint form, chosen from BTF), the ring at `io_uring_setup`'s return and the
    registered files at `io_uring_register`'s, written as `aeiou_uring_sqe` and
    `aeiou_uring_cqe` records after the format's header line, paired by a number the converter
    gives (not `user_data`), and read as the calls they do; `STATX` decoded too, beyond the
    agreed set, since the runner's own engine does its `fstat`s so. ~~Not implemented yet: the
    rest of phase 4 (mapping faults next), phase 3b, readers of a set, the set digest.~~
    *Phase 4 step 2, mapping faults, debated and agreed with the user and built 2026-11-04*
    (§3.98 "Phase 4 step 2"): `fentry` and `fexit` of `handle_mm_fault` for faults an exception
    raised on mappings of files, the converter naming each fault's file from the process's
    mappings and writing a thread's faults as `aeiou_fault` runs (one file, one kind, one stride,
    ended by `--fault-gap` of quiet, 1 ms by default, measured end to start), counted by
    `metrics` apart from the calls (`faults`) and not replayed by `export`; the `fault-around`
    check, so the kernel's fault-around maps no page beside a faulting one (root sets `fault_around_bytes` to a page and puts it
    back; anyone else is told how, or waives it). Decided with it, the theme: the abstract holds
    what the application demands, the storage stack's response (readahead, caching) is the
    system under test's and is judged by the performance it delivers; a capture measures the
    application's demand and think time, not the capture host's performance (items 65, 66). ~~Not
    implemented yet: libaio's real completion (phase 4 step 3, next), the rest of phase 4,
    phase 3b, readers of a set, the set digest; readahead traced for diagnosis.~~
    *Phase 4 step 3, libaio's completions, proposed and agreed with the user and built
    2026-11-04* (§3.98 "Phase 4 step 3"): an `fentry` of `aio_complete` (every request's
    completion as it is posted; `aio_complete_rw`, reads' and writes' alone, where the kernel's
    BTF has no `aio_complete`), of the requests the command's threads issued, recorded at
    `aio_prep_rw` and `aio_fsync` and paired by the kernel's request (the application's
    `struct iocb` address is reused, found at the grant); written as `aeiou_aio_issue` (after the
    `io_submit` line, a request's number and its place in the array) and `aeiou_aio_complete`
    (`in_submit=1` for one posted before its call returned) records; a request counted at its
    completion with the bytes it moved, paired by its number, not `aio_data`, `io_getevents`
    only collecting (notes `aio_completed_in_submit`, `aio_completed_never_reaped`). ~~Not
    implemented yet: POSIX AIO's uprobes (phase 4 step 4, next), the rest of phase 4, phase 3b,
    readers of a set, the set digest; readahead traced for diagnosis.~~
    *Phase 4 step 4, POSIX AIO, proposed and agreed with the user 2026-11-05 and built the same
    day* (§3.98 "Phase 4 step 4"): a uprobe and a uretprobe of glibc's `aio_read`, `aio_write`,
    `aio_fsync`, `lio_listio`, `aio_return`, `aio_suspend`, and `aio_cancel` (not `aio_error`,
    polled in loops) in the C library the capture runs on (`librt.so.1` too before glibc 2.34),
    attached by file offset from its `.dynsym`, as root alone (a uprobe needs `CAP_SYS_ADMIN`,
    found at the grant; under file capabilities said why, item 68 to lift it); each request written at its issue
    (`aeiou_aio_read`, `aeiou_aio_write`, `aeiou_aio_fsync`, `aeiou_lio_listio`), the glibc
    helper's call that serves it preceded by an `aeiou_aio_serve` naming it (matched by process,
    descriptor, buffer, length, and offset), and `aeiou_aio_suspend`, `aeiou_aio_return`,
    `aeiou_aio_cancel`, `aeiou_aio_refused`; a request counted once, in the issuing thread, with
    the bytes the helper's call moved. A program linked statically or on another C library
    (musl, a container's) is not traced, and the C libraries the command mapped that the probes
    were not on are named. Not implemented yet: vectored libaio requests counted (their iovec
    arrays copied), the request heads by libssl uprobes (phase 4 step 5), cuFile and libnfs
    (step 6), phase 3b, readers of a set, the set digest; readahead traced for diagnosis.

55. **A storage-reported value needs evidence the application derives from it (decided with the
    user 2026-11-01, `DESIGN_REVIEW.md` §3.91; not started).** The trace side records no
    `st_blksize`, so a capture's dry run takes the fallback and a local-disk trace would be held
    against the wrong reads. Parts: the digest records the observed `st_blksize` per template
    and asks, as an open question, whether the site's size is derived from it (settled by a shim
    run or a second trace); a capture keeps the value and its dry runs take it; `--emit-trace`
    records the resolved values and `compare` flags a mismatch; `{"from": "st_blksize"}` only
    with provenance (interposition run, two traces, or primary source), else fixed at the
    traced value; an audit of every committed `st_blksize` parameter; an `fstat` shim and the
    runs that confirm each derivation. Also queued the same day, before `aeiou-trace metrics`
    is retired (§3.89 stage 5): `--instance-root`, `--chain-gap-us`, and `--metrics-block` on
    `aeiou-trace digest` (*built 2026-11-02*, item 47).

56. **Nothing drops a truncate (decided and built with the user 2026-11-02, `DESIGN_REVIEW.md`
    §3.92).** A truncate is the filesystem's deferred work of taking a file apart and reclaiming
    its space, and an open with `O_TRUNC` reads nothing where one without may read the whole
    file. Built: `truncate(2)` exported, run by the `trace` node, an op of the abstract
    (contract 0.14), and its own kind in the metrics; `fallocate`'s mode carried end to end and
    rule V29 (a hole punched is never replayed as an allocation); the digest's `space` per
    template; ~~open: the bytes a truncate discarded, and `compare` over two digests' `space`~~
    both built the same day, with every call the export leaves out for want of an open now
    named (it was left out silently); ~~open, for a decision: an open of its own for a
    descriptor held from before the trace~~ decided with the user the same day and built: an
    open synthesized at its first use, a loud warning that the trace is incomplete and how, and
    on.


57. **The filesystem API sweep's bugs (found 2026-11-02, `FS_API_SWEEP.md`, `DESIGN_REVIEW.md`
    §3.93; started 2026-11-02).** The findings of group A of the register: calls dropped with
    no note, parsing that loses calls, implicit and duplicate closes and `close(2)`'s errors,
    traced errors not reproduced (a loud warning), bookkeeping blind to truncate, unlink,
    rename, and rmdir, a create listed for a file that existed, the object-log reader. Each fix
    updates its finding's status in the register.

58. **The calls the abstract cannot express (decided by the user 2026-11-02, §3.93; not
    started).** Every op of `FS_API_SWEEP.md` S-40 in the abstract and the trace file
    (contract 0.15), with the trace file naming descriptors (a `dup` line, a descriptor per
    forked lane, every close line a real `close(2)`, the implicit closes recorded; S-11, S-12,
    S-14, decided 2026-11-02, §3.93 note): copy, link and symlink and readlink, setattr and xattrs, locks, `syncfs`
    and `sync_file_range`, `msync`, `readahead` and `madvise`, `access`, `statfs`, `lstat`,
    `SEEK_DATA`/`SEEK_HOLE` (left out of exports until then, S-28), `renameat2` with its flags
    (exchange, no-replace, whiteout; left out of exports until then, S-27, the user's reminder
    2026-11-02 that it must come back), `RWF_*` flags, multipart uploads in the object engine
    (a write naming its part number, so parts uploaded as they came replay, refused at the
    gate until then; a discard for an abort or an upload never completed, which export as a
    plain close with a note until then; S-29, S-32, decided 2026-11-02), a recursive listing
(an object listing without a delimiter, every key below the prefix, exported as a one-level
`readdir` noted `list_recursive` until then; S-33, decided 2026-11-02), `O_PATH`, `O_TMPFILE`, `F_SETFL`, the
    IOCB fsyncs and vectored requests, batch deletes; and `compare` judging a kind on one side
    only, the flags, and the notes (S-37 to S-39). mmap's page references (S-41) wait on item 54.

59. **A third kind of state, modified in place, and its reset (decided by the user 2026-11-02,
    `DESIGN_REVIEW.md` §3.94; not started).** Content made before the run (by `aeiou datagen`
    or an earlier run with its manifest) that the run modifies (a KV cache or a vector index
    restarted over what it left); its manifest the initial conditions. Reset by datagen over
    the **computed write set** (from the abstract, parameters, seed, GPU count; never a tree
    walk), bounded for an interrupted run by a durable per-actor high-water mark of ops
    completed; a dirty marker written at a modifying run's start, refused by the next run,
    cleared by reset. The content check is `aeiou-verify`'s, sampled (§3.94 item 5). *Refined by
    the user 2026-11-10 (`DESIGN_REVIEW.md` §3.99, the joint review's issue 7):* a path of the
    write set that exists at the right size is left alone (one `stat`, never a walk) when the
    run never wrote into it while it existed, which the computed write set says; a path written
    in place is regenerated, a size check not seeing an overwrite of the same length. The dirty
    marker carries the run's nonce (item 77).

60. **The `trace` node's preconditions, recorded and prepared (decided by the user 2026-11-02,
    §3.94 item 6; not started).** Whether each path existed before the trace, from all the
    trace's evidence (earlier and later), else asked of the trace's owner by the digest's
    interview and recorded with provenance; a prepare step that makes the inputs at their read
    extents and removes the creates. `FS_API_SWEEP.md` S-17 is built with it.

61. **Object requests as the abstract's own ops (decided by the user 2026-11-03,
    `DESIGN_REVIEW.md` §3.95; not started, no WG need short-term).** An object namespace's ops
    are the requests the application sent, taken from its trace at face value: ranged,
    suffix, part, and conditional GETs, HEAD, LIST pages, PUT, multipart create, part N,
    complete, and abort, CopyObject, UploadPartCopy, DeleteObjects; no mapping between POSIX
    and object anywhere, and no converter (open-source projects do that). Replaces §3.65's
    mapping and V18, `object.rs`'s POSIX emulation, and the export of an object log to POSIX
    calls (request lines in the trace file's version 2); takes item 58's object entries.
    First step: check whether `object_store` can issue what replay needs, else `aws-sdk-s3`
    or a small signed-HTTP client. A trace merging an strace with an object log (Milvus: a
    query node's local files and its bucket) joins item 49.
62. **Secrets scrubbed from a capture (raised by the user 2026-11-03; decided and built
    2026-11-04, `DESIGN_REVIEW.md` §3.97: capture less, not scrub after).** Built: the strace
    line adds `-e verbose=!execve,execveat -s 0` (no `execve` list or environment, no call's
    bytes; `-v` kept for arrays), `capture.json` keeps the command line with values named as
    secrets and URL passwords redacted (`--argv none` the program alone), and `metrics` and
    `digest` note and warn of a trace that prints them. Correction: no reader uses the
    environment or any buffer's contents. The text as queued:
    A capture is sent to people who were not there, so no secret of the owner's may be in any
    of its files. Phase 2 keeps none in `requests.jsonl` and `requests.bin` (the header
    allowlist, presigned URLs' signatures redacted; `DESIGN_REVIEW.md` §3.96), but two
    places do not scrub yet: strace's `-v` prints every `execve`'s whole argument list and
    environment into `trace.txt` (an `AWS_SECRET_ACCESS_KEY`, a token, a password on a
    command line), and `capture.json` records the command's `argv`. To decide: scrub after
    the capture (by name and pattern, values replaced, what was replaced recorded) or capture
    less (drop `-v`, losing the environment the digest and a reader use), and whether the
    raw trace is kept for the owner alone beside a scrubbed one to send; the same holds for
    the eBPF converter of phase 3, and for a trace taken by hand.
63. **`aeiou-trace scrub`, an old trace rewritten without its secrets (queued 2026-11-04 with
    item 62's design, approved by the user; not started).** A trace taken before 2026-11-04 or by
    hand with another line prints `execve`'s arguments and environment and the first bytes of
    each buffer; `metrics` and `digest` warn of it (`trace_may_hold_secrets`). To decide: a
    command that writes the trace again as the capture's line would have (the lists as
    addresses, every buffer `""...`), streamed, with what it replaced counted; until then the
    owner sends the digest.
64. **`uring_knobs`'s SQPOLL count flaky under load (queued by the user 2026-11-04; not
    started).** `tests/uring_knobs.rs` `io_uring_knobs_keep_the_fingerprint` asserts one kernel
    submission thread per loop (`HostCounters.sqpoll_threads`, read from each ring's fdinfo at
    the end of its loop), "exact however short the row is"; it counted 2 of 3 once, row `sqpoll
    per loop`, in the instrumented coverage run beside every other test (2026-11-04), and passed
    in `./dev.sh check all`, the other coverage pass, and five runs alone. To find: whether a
    loop's fdinfo can be read before its SQPOLL thread exists or after it is gone, and make the
    count exact or the assertion what the kernel guarantees.
65. **A `trace` node that replays faults as touches through a mapping (queued 2026-11-04 with
    item 54 phase 4 step 2, agreed by the user; not started).** The export writes a mapping as one
    read of its whole range (`mmap_as_read`), one large demand where the application made many
    small touches over time, and through `pread`'s readahead where the application's went through
    a mapping's read-around: another workload by the theme of `DESIGN_REVIEW.md` §3.98 "Phase 4
    step 2". To design: the `trace` node's op for a run of faults (its file, offset, stride,
    count, and the quiet time between runs) executed under `--posix mmap` as touches of the
    mapping, the export writing it where the trace holds faults and the mapping's range read
    where not.
66. **Readahead and fault-around as host facts (queued 2026-11-04 with item 54 phase 4 step 2,
    agreed by the user; not started).** Readahead is the solution's by the interposition test, in
    the client too (`read_ahead_kb` per backing device, NFS's `rsize` window), so a result
    cannot be read without it. To build: `aeiou run` and `aeiou-trace capture` record, for each
    `--root`'s mount, its backing device's `read_ahead_kb` and the host's `fault_around_bytes`
    (where readable) beside the mount options; whether a submitter may tune them is the WG's rule
    (§8), not the tool's.
67. **The pages each fault mapped (queued by the user 2026-11-04 with item 54 phase 4 step 2; not
    started).** A fault maps the whole folio its page is in (up to 2 MiB, as the file was cached
    by large writes or readahead), and where the `fault-around` check was waived the cached
    pages beside it too, so an `aeiou_fault` shows where touches began, not how much the kernel
    mapped ahead unseen (`DESIGN_REVIEW.md` §3.98 "Phase 4 step 2", found at the grant). To
    research before proposing: a probe of the step that maps them (`filemap_map_pages`'s range,
    `set_pte_range`'s count, `finish_fault`), which kernels have it in BTF and not inlined, and
    the record's field (the pages mapped with each fault, a run's sum), so a reader knows each
    fault's extent.

68. **POSIX AIO under file capabilities, by `uprobe_multi` links (queued by the user 2026-11-05
    with item 54 phase 4 step 4; not started).** A uprobe attached through `perf_event_open`
    needs `CAP_SYS_ADMIN` (`perf_uprobe_event_init`, Linux 6.18, src), so a capture under
    `cap_bpf,cap_perfmon` does not trace POSIX AIO, and says so (`DESIGN_REVIEW.md` §3.98
    "Phase 4 step 4", found at the grant). A `uprobe_multi` link (`BPF_LINK_CREATE`,
    `BPF_TRACE_UPROBE_MULTI`, Linux 6.6) needs only `CAP_BPF` and `CAP_PERFMON`, but its program
    must be loaded with that expected attach type, which aya 0.14's `UProbe::load` never sets.
    To build: a patch of aya under `[patch.crates-io]` (load with the attach type, create the
    link with every offset of a library at once), or the same upstream; root's `perf_event_open`
    kept below 6.6.
69. **The context tree of a KV trace, a digest finding (queued by the user 2026-11-06 with the
    AgentX side runs, `DESIGN_REVIEW.md` §3.64 note of 2026-11-06; not started).** An agentic
    session is a main agent's context with sub-agent and side-call contexts run beside it while
    it waits (two to five served in turn in the replay, the main one resumed after 6 to 31 of
    their requests), and that tree, with each context's tokens, turns, and compute, is what a
    person should read of the workload. A storage trace shows it only in part: LMCache names a
    chunk by its prefix's hash, so requests link into contexts only where a chunk one wrote is
    read by a later one, and siblings by the prefix chunks they share; at a large pool almost
    nothing is read. An engine's request log (vLLM's and LMCache's lookups) or a request corpus
    gives it whole. To design: a site kind of the digest for it, what the trace decides and what
    it leaves to an engine log or the owner's answers, and the factoids (per context its
    requests, tokens, prefill and decode compute; per side run its lanes and length).

70. **KV engines larger than one GPU, and the deployment as parameters (queued by the user
    2026-11-07 with the pool debate, `DESIGN_REVIEW.md` §3.99; not started).** The KV abstracts
    take one actor as one engine at TP 1, the only shape traced. LMCache 0.5.5 names a chunk by
    model, `world_size`, `worker_id`, hash, and dtype (`CacheEngineKey`, read in its source): under
    tensor parallelism every rank's worker stores and loads its own file of each chunk, its share
    of the KV heads, at about `chunk_bytes / TP`, the ranks in step; for MLA models
    (`save_only_first_rank`, on by default with MLA) the first rank alone, the whole chunk; for
    pipeline stages, by the same key, presumably per stage (not read); data-parallel attention
    is an engine per replica. Proposed, ~~not decided~~ decided below: one abstract per storage mechanism, the
    model's family a parameter (sharded per rank or first rank only), not an abstract per family;
    one process per GPU as in training, GPU g of engine g div TP at rank g mod TP~~, the request
    stream's draws keyed by the engine (a change to positional keying by GPU id, a recorded
    decision)~~ (the draws keyed by session and domain, below); the deployment stated as KV bytes per token, pool tokens and slots per engine,
    and the CPU tier, with a model, an accelerator, and a parallelism as a case that derives
    them (the WG's reference choices, §8). To do: a TP 2 capture (this laptop cannot) to see
    whether the ranks' calls interleave in a way values cannot express, which would split the
    abstract; LMCache's layer-wise mode (its key adds a layer id), likely a call structure of
    its own.

    *Decided by the user 2026-11-09 and 2026-11-10 (`DESIGN_REVIEW.md` §3.99 items 2, 3, 9, 11, and
    the note "The generator in the abstract, a shared tier, and events between runners"):* one
    abstract per storage mechanism, the attention family a parameter, on condition of the TP 2
    capture (a call structure across ranks that values cannot express reopens it). The case fixes
    the engine, the submitter none of it: the model and its KV bytes a token; the accelerator and its
    service model (prefill and decode time a token, measured once on the accelerator and published
    with the case, as training's step time is); the engine as one TP × PP replica; the slots derived
    from the pool and the mean live context, as vLLM bounds them, so pool and slots are one setting;
    ~~the CPU tier 0~~ G1 and G2 at realistic sizes the WG fixes (the user, 2026-11-13, `DESIGN_REVIEW.md`
    §3.99 note "Tier names"). The submitter's one knob is engines: the headline is the largest whole number of
    engines that passes, as concurrent sessions served; the count by slots, the last engine partly
    filled, is reported beside it, supplementary and never ranked on. Engines are grouped in domains,
    the engines sharing one storage tier (item 72); every draw is keyed by the stream's seed (the
    WG's, as the dataset seed is), sessions by (seed, session), placed on a domain's engines by its
    routing rule, so adding a domain leaves the others' ops unchanged. The keys are the abstract's
    choice, not the runner's: the KV abstracts key by session and domain, and every other abstract
    keeps its GPU-id keys; the positional-randomness invariant's wording changes for the KV
    abstracts only, a recorded decision. Engines ramp together by default; staggering is a WG
    what-if. Accepted when every committed abstract outside `kvcache/` keeps its fingerprint and
    measured shape (the catalog check). *The joint review, 2026-11-10 (§3.99):* the two
    mechanisms are redefined as LMCache's multi-process connector adapters on a local and on a
    shared path (the committed abstracts being captures under the in-process connector), read
    from the source first (item 72) and confirmed by item 74's capture. ~~Open: how the score
    counts a domain of more than one engine (item 72, phase 2).~~ *Decided by the user,
    2026-11-10 (§3.99, the note on LMCache's multi-process storage):* the adapter is `fs_native`
    with a capacity; local and shared paths are one mechanism, the sharing a parameter (other
    engines' chunks found by `stat` and skipped, or never present); its `stat` per lookup and per
    store and its file per object group are in the abstract, real metadata accesses. How the score
    counts a domain of more than one engine is answered in phase 2 (item 72 (5)).

71. **The KV request stream from a session generator written in the abstract (decided by the user
    2026-11-10, `DESIGN_REVIEW.md` §3.99 item 10 and the notes "The fifth generator held out" and
    "The generator in the abstract, a shared tier, and events between runners"; started 2026-11-10).**
    Sessions come from a generator fitted to the AgentX corpus (393 sessions, 98,827 requests), not
    the corpus as a table of rows and not archetypes; the scratch's `gen5` carries 5,839 fitted
    numbers, quantile tables, not requests. A session is a tree of concurrent contexts, the main
    agent and its sub-agents in parallel, with their fan-out and concurrency, each context's turns,
    think times (the main agent's and a sub-agent's apart), and step kinds (extend, extend an older
    tip, fresh, trim, deep cut, cut to the main prefix, compaction) fitted at depth 1, and idle gaps
    with the corpus's clustering of long gaps and its spread between sessions. Depth and an
    orchestrator's fan-out are WG what-if parameters, 1 by default, documented as supported by no
    trace. The generator is expressed in the abstract language, the runner agnostic to the workload;
    the runner streams it, each session yielding its next request on demand, merged in planned order,
    never the whole timeline held. First step: `gen5` ported to an abstract with today's language and
    the fewest additions, held to the Python program's output; what the port cannot do without is
    the list of constructs to add (expected: a prompt as parent, kept length, and new blocks; a choice
    among earlier prompts; item 73), marked provisional in the contract until a second workload uses
    them. What it claims, and the tolerance file says so: writes within ±7 % of the pool model on the
    ramp (held out, −1 % and +1 %); reads reproduce the pool model on this corpus only, since they do
    not carry between halves of it. Known and not diagnosed: in steady runs its writes come to 15.4 to
    16.0 a second against the table's 13.1 to 13.8. *The port, run 2026-11-10 (`DESIGN_REVIEW.md`
    §3.99, the note "The generator ported to an abstract"):* request structure held to `gen5`
    within +4 % at 400 sessions, no chunk ops yet; eight gaps: chunk names from lineage (item
    75), a binding of another lane, lanes never joined and arrays drawn once a session (item 72),
    `x @ i` recursion (item 73), float to integer, empirical weights as expressions, and lazy
    bindings. The second exposed a fault in `gen5`: a sub-agent may reuse main prompts sent after
    it; its pool to hold only prompts sent before the request's planned time. *`gen6`, 2026-11-12
    (`DESIGN_REVIEW.md` §3.99, the note "The sixth generator, in planned-time order"):* `gen5`'s
    tables generated in planned-time order, the generator the port follows; against the table,
    ramp writes −7 % and −4 %, reads 8 of 8 cases within two standard errors; held out, ramp writes
    −9 ± 4 and −3 ± 6, the first 2 points outside the ±7 % this item claims.
    *Contract 0.15, 2026-11-12 (`DESIGN_REVIEW.md` §3.99, the note "The timeline built"):* the
    timeline (gap 2), a `parallel` not joined (gap 3), the `array` (gap 4), and `prev`, the
    value of a binding in its loop's previous iteration, for lane state that depends on the
    pool, which `@` cannot recompute; the port resumes on them.
    *Contract 0.16, 2026-11-12 (the note "Chunks named from lineage"):* gap 1, each prompt a
    `seq` and its chunks found or stored by `lookup` in a cache of its timeline; the port's chunk
    reads and writes equal the reference's request by request (20 sessions, held 0).
    *Contract 0.17, the same day:* a timeline's entries in groups (the user's choice), so one
    timeline and one store serve every session of an engine, each drawing from its own prompts;
    equal to the reference at 20 and 40 sessions.

72. **The plan, domains, and events in the contract (decided by the user 2026-11-09 and 2026-11-10,
    `DESIGN_REVIEW.md` §3.99 items 4 to 9 and 13, and the note "The generator in the abstract, a
    shared tier, and events between runners"; the plan side of (1) and (2) built 2026-11-12).** The KV abstracts need what the
    contract cannot yet say.
    (1) *The plan*, computed before the run from positional draws and never from the run: people
    arrive on the WG's ramp (30 minutes from 5 % to 95 % by default, then 60 minutes of plateau, a
    tuning form a third as long), a return reviving k sessions (3 to 8) over 1 to 5 minutes; each
    context's requests at planned times. A pass over a domain's merged timeline fixes the op set: a
    timeline construct merging its lanes in planned order, a queue with a fixed number of servers
    (the GPU slots), and the cache construct of item 75 (the pool, and LMCache's eviction: LRU,
    watermark 0.8, 20 % of the used capacity a trigger, checked every second, over a WG-fixed L2
    capacity an engine). (2) *Lanes* as contexts, spawning concurrent lanes, nested; causal edges
    carrying the planned gap and the lag, validity edges only where reversing two ops changes a
    result. (3) *Store lanes*, a fixed worker count draining a queue, bounded by a WG-fixed staging
    size; *unlink lanes*, each unlink an op of the plan at its planned time, its victims computed
    from the WG-fixed capacity (the application's setting, never asked of the storage), which
    lane issues it (the chunk's writer, or one eviction lane a domain) following LMCache, read
    first. (4) *The resumed contexts* written by `aeiou datagen`, filling the tier to just under
    its watermark: item 59's state, made before the run and modified by it (not an `input`
    namespace, read-only by V14), its reset the diff (unlink the chunks the run made, recreate the
    pre-filled ones it evicted). A chunk's bytes are a function of its key, so the same key gives
    the same bytes; its compression ratio a case parameter, near incompressible until measured.
    First, without a GPU: which LMCache multi-process connector adapter serves a local and a
    shared path, its eviction trigger, and its staging (size, drop policy, process), read from the
    source, the two mechanisms redefined as those adapters (item 70); the datagen time of the
    pre-fill and of its reset estimated at Rust datagen's measured speed; the ramp rerun at the
    chosen operating point with clustered returns (k 1, 3, 8), the heaviest window possibly the
    returns, not the plateau.
    (5) *Domains*: the engines sharing one storage tier, the unit of scale-out; sessions route only
    within their domain, no state crosses domains, and a host replays only its domains' plans. A
    domain may span hosts and is a barrier and event scope in the run's one coordinator. Phase 1: a
    domain of one engine, the storage tier one shared namespace (`lmcache-local-disk` and
    `lmcache-shared-fs`). Phase 2: contexts routed across a domain's engines by a WG-fixed rule
    (prefix hash and planned load, an assumption no trace supports); before it is built, the reads
    crossing engines under prefix routing on the corpus and the pass's cost at the domain sizes
    proposed, measured in the scratch.
    (6) *Events*: an op at an edge's head signals, the op at its tail waits, consuming an event that
    came early. The abstract names engines and lanes, never ranks; event ids are computed from the
    edge's keys; a causal event carries its sender's lag; one event a request's stores. Delivered
    in-process on one host, by the coordinator between hosts (engine to rank by the GPU-range
    formula, the coordinator holding every rank's connection), two messages through it an event.
    The coordinator's one-way latency is measured and reported with the edge waits and never charged
    to the system under test; a run whose messaging p99 is a meaningful share of the latency budget
    is invalid. A wait times out into a failed run, never a hang. General primitives (a checkpoint's
    manifest after its shards). The coordinator stays in rank 0's process; a standalone one, or a
    launcher that stays alive to host it, on evidence from that measurement.
    (7) *The runner*: `launch = plan + L`, L a lane's inherited lag, never shrinking; D, a request's
    own storage wait, judged at p99 per 5-minute window assigned by planned time, the ramp's
    included, with a single-response ceiling (30 s as the example); a store unfinished at its
    reader's launch waits in the reader's D, staging overflow a gate failure; ~~an eviction batch past
    the next trigger a failure~~ an eviction batch past the next tick delays the next batch, the overage
    the eviction lanes' L, reported only (the user, 2026-11-13); a submittable run stops at certain failure, a tuning run under a lag
    ceiling; the benchmark host's own lateness a validity check. Reported: the load curve, read and
    write bandwidth, capacity held, creates, renames, and unlinks a second, eviction bursts, each
    session's L, the sum of D, edge waits and messaging latency, write latency, backlog peak,
    would-be drops, and GPU idle time as modelled.
    *Decided 2026-11-13, not built (`DESIGN_REVIEW.md` §3.99, the note "The run side of
    timelines"; the user chose each recommendation):* a planner (the walk `dry-run` makes) a
    bounded horizon ahead of an executor issuing a lane's ops one at a time at
    `max(plan + L, previous completion, edges)`; D measured by a `span` node; validity edges by
    path in planned order, the `trace` node's rule; (3)'s store lanes and staging built with it,
    as `workers` and `submit` (contract 0.20, with `span`); a thread a live lane under the
    blocking backends; `--on-certain-failure stop|continue` and `--lag-ceiling`; GPU idle as
    modelled later.
    *Step 1 built the same day, contract 0.20 (plan side; the note's addendum "Contract 0.20
    built"):* `workers`, `submit`, and `span` (rule V36), and `lmcache_local_disk_tiered`'s G3
    writes on 4 store workers, its staging G2's bytes, its reads span `load`, its fingerprint
    unchanged. Next: the planner (step 2).
    The WG's values, none chosen: the queue-wait target that places the operating point below the
    knee (the user's decision of 2026-11-10: no provider queue policy in the workload), the TTFT
    target behind the budget, the ramp, the resumed share, the L2 capacity, the staging size, the
    domain size, the routing rule. A contract version and every KV fingerprint change.
    *Built 2026-11-12, contract 0.15, plan side only (`DESIGN_REVIEW.md` §3.99, the note "The
    timeline built"):* (1)'s timeline construct as `timeline`, `publish`, `published`, `entry`,
    and `parallel`'s `join: false`, walked in planned order by `check` and `dry-run`; `run`
    refuses a timeline until (7) is built; the planned clock not yet in `--emit-trace`.
    *Contract 0.18, the same day (the user's choices; `DESIGN_REVIEW.md` §3.99, the note "Tiers as
    general pieces"):* (1)'s queue with a fixed number of servers as a timeline's `resources`,
    taken by `acquire` and given back by `release`, a lane waiting in planned time, first come
    first; with item 75's held GPU tier, the engine's admission; `now`, a lane's planned time.
    (8) *Opt-in.* An abstract without a timeline runs exactly as today, independent actors, no
    per-op cost of the plan; D and the lag are the case's results, judged through the `result.`
    constraints, never runner-wide (training's pass stays accelerator utilization). Accepted when
    every committed abstract outside `kvcache/` keeps its fingerprint and measured shape and a
    training run is no slower.

73. **`x @ i` evaluated without recursion (found 2026-11-07, `DESIGN_REVIEW.md` §3.99 items 9 and
    13; not started).** An `x @ (r − dd)` reaching back thousands of requests overflowed the
    runner's stack at 2,000 requests a slot; `RUST_MIN_STACK` of 1 GiB got the scratch past it. A
    valid abstract must not need that: the evaluation made iterative, or its depth bounded and
    refused by `check` with the bound named, with a test at the depth that failed. The `gen5` port
    (item 71) leans on it. Accepted when every committed abstract keeps its fingerprint: the
    evaluation changes, never the value.

74. **The AgentX case retargeted, with its tolerances (decided by the user 2026-11-09 and
    2026-11-10, `DESIGN_REVIEW.md` §3.99 items 1 and 12; not started, after items 71 and 72).** The
    committed target (2,321 reads at 6 GiB) is the 131k cut's artefact; the case is retargeted against
    the pool model (`agentx.simulate` with `lmcache_accesses`) on the whole corpus at the WG's
    operating point, and the §3.64 note of 2026-11-06 amended. Compared: the abstract's dry-run
    counts against the pool model. Reads: a band of about two standard deviations of the pool
    model's reads over at least 8 seeds, measured at the operating point, never carried over from a
    neighbouring one. Writes: ±7 %, scored on the ramp; steady-state runs are for exploration and
    not scored on writes. The tolerance file says the band claims that the abstract reproduces the
    pool model and nothing about how near the pool model is to a real service; the corpus's sampling
    limit, 1.3 to 1.9 times in reads between its halves, is stated beside it. The lab replay (233
    requests) stays a consistency check at its own point, with its own band (20.6k to 48.0k reads).
    The GPU session's checks, in order: the pool model with several slots in flight near the knee;
    contexts past 485k tokens; a capture under LMCache's multi-process connector and its filesystem
    L2 adapter (temporary file, `rename`, one flat directory, 64 unlinks at once) against what the
    abstract emits; the TP 2 capture of item 70; the compressibility of real chunk files for the
    case's model, KV dtype, and LMCache encoding, which sets the case's compression ratio.

75. **A cache construct, a stack of tiers (decided by the user 2026-11-10, `DESIGN_REVIEW.md`
    §3.99 note "The generator in the abstract, a shared tier, and events between runners"; not
    started).** The pool and the storage tiers behind it as one construct of the language, named for
    the mechanism: a stack of tiers, each with a capacity, an eviction policy, and a write policy
    (written when made, as LMCache stores every new chunk, or when evicted from the tier above, as
    Dynamo's block manager cascades), the stack with a rule for what a hit promotes, and where a tier
    puts a block (a file a chunk, or a slot on a device) a parameter. Policies include a static,
    pinned tier filled at load (DiskANN's node cache, modelled by hand in `diskann_search` today).
    Keys are opaque to the construct: LMCache's chained prefix keys, under which an early turn loses
    every later chunk, belong to the KV generator, not the cache. Only the application's own caches
    are modelled, never the operating system's page cache, which is solution by the interposition
    test. LMCache's settings built first; Dynamo's source read before the shape is fixed; expected
    users beyond KV: a VDB's tiers (DiskANN's node cache, a local disk cache over an object store as
    Milvus keeps), and training's epoch cache (`tf.data`'s `cache()`); a system that does not fit
    gets a construct of its own. Its state scales with the WG-fixed
    capacities, never the dataset (the rule against per-file structures concerns the dataset). Its
    exact rules (LRU ties, the order within a planned instant) are definitions the runner fixes in
    `runner/REFERENCE.md`, a change to one changing every KV fingerprint; the runner is the only
    reproducer of the op set, a run's exported trace the way to analyse it elsewhere. Provisional in
    the contract until a second workload uses it.
    *Built 2026-11-12, contract 0.16 (`DESIGN_REVIEW.md` §3.99, the note "Chunks named from
    lineage"):* a timeline's `caches`, a tier keeping every entry or evicting the least recently
    used past a capacity in entries or bytes, and `lookup`, with the `seq` and `chunk_key` that
    name a chunk by its prefix (gap 1 of item 71). Not built: LMCache's watermark eviction, a tier
    of more than one level and promotion, the write policies, ~~vLLM's pool as a tier~~, the unlinks
    an eviction issues (item 72 (3)), Dynamo's source read.
    *Contract 0.18, the same day (the user's choices of 2026-11-12: the policies in the abstract,
    the runner carrying mechanisms; `DESIGN_REVIEW.md` §3.99, the note "Tiers as general
    pieces"):* a tier's victim `order` as expressions over `cached` (`lru()` one of them), holds
    (a `lookup`'s `hold`, `release`), and `full` (`skip`, `wait`, `fail`); vLLM's pool written as
    a held tier in vLLM's free-queue order, with `full: wait` as its admission, equal request by
    request to `agentx.simulate` corrected (it overfilled the pool by one block a hit, found by
    the comparison; fixed the same day and its numbers rerun, the conclusions unchanged). Not built: the timed trigger
    (LMCache's watermark), a statement body on eviction (an unlink, a write to the tier below),
    `preempt`, promotion between tiers, Dynamo's source read.
    *Contract 0.19 ~~proposed 2026-11-13, not decided~~ decided and built 2026-11-13 (`DESIGN_REVIEW.md` §3.99, the note "Contract
    0.19 proposed" and its addendum "built"), after LMCache 0.5.5's eviction was read and NVIDIA's G1 to G4 adopted as the
    tier names:* timed eviction (`evict {every, when, count}`), eviction bodies on eviction lanes
    (the unlinks), and `probe`, `touch`, `store` over a range of chunks, or over an array of them,
    a store's `taken` (what it newly stored) and `present: renew`, for G2 above G3, written as
    LMCache writes it; the gen10 ports equal to LMCache's own `LRUEvictionPolicy` request by
    request and unlink by unlink. Not built: ~~the KV abstract's G2 and G3 (the committed
    `lmcache-local-disk` still has none)~~, `preempt`, Dynamo's tiers, the run side (item 72 (7)).
    *The KV abstract's G2 and G3, 2026-11-13 (the user's choices; `DESIGN_REVIEW.md` §3.99, the note
    "The tiered abstract committed"):* `lmcache_local_disk_tiered`, beside `lmcache_local_disk` in
    its mechanism: the session generator of item 71 (its fit `kit/sessions.fit.json`, written by
    `kit/agentx.py sessions`), one engine's slots and G1, LMCache's G2 and G3 at the WG's example
    sizes (G1 180 GiB less the weights, G2 490 GiB, G3 7.68 TB), the chunk calls traced under the
    in-process connector; its case `agentx-tiered` unsettled and dry-run only until (7). The older
    abstract and its four cases are retired when (7) lands.

76. **Dynamo's block tier (queued by the user 2026-11-10, `DESIGN_REVIEW.md` §3.99 note "The
    generator in the abstract, a shared tier, and events between runners"; not started, not
    near-term).** ~~The user's understanding, to check against the source: Dynamo's tiers are G1 HBM,
    G2 host DRAM, G3 node-local NVMe, G3.5 an NVMe-oF block layer reached from user space, G4 an
    archive by an object API.~~ NVIDIA's definitions (read 2026-11-13, adopted as this project's
    tier names, `DESIGN_REVIEW.md` §3.99 note "Tier names"): G1 HBM, G2 host DRAM, G3 local SSDs,
    G3.5 Ethernet-attached flash for KV shared across a pod (CMX, over NVMe and NVMe-oF, NVMe KV
    extensions among them), G4 shared object and file storage; KVBM's G4 is an S3 store in its
    configuration reference. If G3.5 issues block commands from user space, supporting it needs
    block-device ops in the language (a shim cannot turn file or object calls into block calls, so
    it is another API and another workload), a backend for them, and the extent index Dynamo keeps
    computed in the plan as an allocator's state, its writes modelled if it is persisted. A kernel
    NVMe-oF device would instead be POSIX at offsets on a block device. First: the call path and the
    index read from the source.

77. **A per-run nonce in every written namespace's content (decided by the user 2026-11-10,
    `DESIGN_REVIEW.md` §3.99, the joint review; not started).** Namespace content is a function
    of (namespace seed, path, offset), all known from the published case, so anyone can compute
    every byte a run will write and stash a copy outside the benchmark's directories for the
    system's dedupe to find, and a system keeping dedupe fingerprints of deleted data gains the
    same from a prior run; checkpoints today as much as KV. Every run draws a nonce from the
    operating system's randomness; rank 0 draws it and sends it with `t0`, so the same path and
    offset give the same bytes on every host and duplication within the run stays real. Mixed
    into the seed of every written namespace's content, never a dataset's: content written
    before the run (datagen's, a pre-filled tier) is unscored and comes back exactly at reset, so
    it keeps the dataset seed alone. Recorded in item 59's dirty marker while a state is dirty
    (after reset no file carries it), in an output namespace's manifest, and in the run report.
    Ops, paths, sizes, the fingerprint, CLOSED, and the compression ratio unchanged. A change to
    the runner's payload definition for namespaces (`runner/REFERENCE.md` §5), a recorded
    decision; `aeiou-verify` takes the nonce from the manifest or the marker.

78. **G3 and G4 active together in one abstract (queued by the user 2026-11-13; not started).**
    Some products serve a node-local G3 and a shared G4 at once (the user's example: Hammerspace),
    so a KV abstract must be able to have both, each tier a namespace of its own on its own
    storage (G3 the node's drive, G4 the shared file system or object store), each issuing its
    own reads, writes, and unlinks, while G1 and G2 issue nothing (as in
    `lmcache_local_disk_tiered`, held by `test_kv_tiered_issues_storage_ops_on_g3_alone`).
    First, read from LMCache's source (and Dynamo's, item 76) what a server with two L2 adapters
    does: every chunk written to G2 copied to each adapter (0.5.5's store loop, as read
    2026-11-13), the lookup order across them and where a G4 hit is loaded, each adapter's own
    eviction, and whether a chunk evicted from G3 is still found in G4. Then: whether 0.19's
    tiers, `probe`/`store`/`touch` and `on_evict`, already say it (expected: a G4 tier in the
    timeline's cache with its own namespace and eviction body), or what the contract lacks; how
    a run places G3's namespace on node-local storage and G4's on the shared system (`--root`
    and `--endpoint` today name one root and S3 endpoints); and G4 as one tier shared by a
    domain's engines (item 72 (5)). A case with both at the WG's sizes; a WG what-if, not a
    default.

## 7. Environment

The development machine is WSL2 (kernel 6.18, 20 cores, 31 GB RAM), with no realistic NFS
target. A loopback NFS mount (`nfs-kernel-server` exporting a tmpfs directory, mounted from
`localhost`) exercises the real NFS client code paths for correctness; set up 2026-09-30, the
sequence and what it showed are `runner/REFERENCE.md` §7 (the tmpfs and the mount must be
redone after a WSL restart). All performance work must run on real Linux client nodes.

## 8. MLPerf Storage use (added 2026-09-30)

Everything above describes a general tool: a reproducible, abstract-driven I/O workload runner
with a content-addressed workload definition. The MLPerf Storage WG is its first user and the
reason it exists, and this section is the one place that lists what is specific to that use, so
that another user (a storage vendor's own QA, a filesystem developer, a researcher) can ignore it
without losing anything. The rule for the code and the docs: nothing is named after MLPerf
unless it is about the WG's process, and a WG-specific default or behaviour says so where it
appears.

**What is generic.** The abstract model, the AST contract and its hash, the builder and its
hermetic harness, the positional RNG and Feistel sharding, the datasets and datagen, the
fingerprint, the backends, the coordinator, the eight workload shapes (they are shapes of real
applications, PyTorch, DCP, safetensors, DiskANN, FAISS, vLLM, not MLPerf benchmarks), and the
`--dry-run --metrics` acceptance method. `datagen` is a term borrowed from MLPerf Storage;
dgen-py is a generic payload generator that happens to come from the same community (used
until 2026-10-11, when the payload moved to SFC64, `DESIGN_REVIEW.md` §3.70).

**What is WG process, and where it is referenced.**

| Topic | Generic fact underneath | WG-specific part | Where |
|---|---|---|---|
| Roles | The AST is content-addressed; anyone can author and run one. A lock (`DESIGN_REVIEW.md` §3.78, decided 2026-10-22) names an abstract and every value of a use case, with a file hash and a workload id. | The WG authors the reference workloads and publishes their ASTs and hashes; submitters run the published hash and nothing else counts as "official". *(2026-10-22: what the WG publishes becomes a lock per benchmark entry, kept in `mlpstorage`'s install image with its file hash and workload id, never in aeiou's tree; the WG validates an uploaded report by recomputing both and the fingerprint at the report's aeiou version, §3.78 point 14.)* | `GRAMMAR_OPTIONS.md` Option D "Roles"; `schema/README.md` §0 |
| Divisions | The backend is part of the application; the interposition test (§5) draws the line between application and solution. | ~~**CLOSED** scores only the backend the real framework uses (`sync` for PyTorch);~~ **CLOSED** means the same operation sequence for every system under test, so it scores only the backend the abstract declares (2026-10-01; buffered and direct are not compared, unlike MLPerf Storage v3.0); other backends are speed-of-light rows; `trace` (`replay` until 2026-10-02) and wall-clock-bounded phases are never CLOSED. Whether a startup phase (the ImageFolder walk) is inside the measured window. | §4 backends, §5 comparison policy; `DESIGN_REVIEW.md` §3.12–3.13, §3.15; `ABSTRACTS.md` §1, §9.8; `NAPKIN_MATH.md` §8 |
| Data verification and seed privacy | Data is reproducible from (dataset seed, id, offset); the run seed and file order are private to the run. | Motivated by submission fraud under the interposition test; the offline verifier is an audit tool for the WG's review process. | §5; `DESIGN_REVIEW.md` §3.13, §3.16 |
| Reference parameters | Every workload has parameter slots filled from configuration, measurement, and traces; a parameter file (`schema/README.md` §8) is the published form of one set. | Which values are the reference set (batch sizes, step times, dataset scale relative to client DRAM, the 500-step bound) is a WG decision recorded in the published parameter files. *(2026-10-22, `DESIGN_REVIEW.md` §3.78: the WG's benchmark rules, a rules file that freezes by state and holds the constraints a run must meet, the 5× rule among them, are WG process; aeiou ships the mechanism and no WG rules file.)* *(2026-10-23, `DESIGN_REVIEW.md` §3.81: the Unet3D use cases' batch, step times, and sample sizes are the WG's reference values, from its DLIO workload configurations, and each fragment names its source.)* | §5 decisions, use cases row; `ABSTRACTS.md` `[measure]` slots; `schema/examples/params/` |
| Acceptance tolerances | `aeiou-trace compare --judge` holds a trace and an abstract to a tolerance per class of metric; the values in the tool are this repository's defaults (`builder/REFERENCE.md` §7; decided 2026-10-02, for now). | Which values accept an abstract for a workload class, and which recorded differences are tolerated, is a WG decision published as a tolerance file. | §6 item 14; `DESIGN_REVIEW.md` §3.55 |
| Workload selection | The builder can express any POSIX-shaped skeleton. | The ninth and tenth abstracts follow the MLPerf Storage ResNet50/CosmoFlow and Parquet→Arrow shapes because those are what the WG submits (`train_stream_tfrecord` / `train_stream_parquet` done 2026-09-30; the tenth pending). | §6 item 15; `GRAMMAR_OPTIONS.md` §6.5 |
| The KV workloads' engine and CPU tier | What an inference engine holds (its KV pool, HBM less the model) and what LMCache's CPU tier absorbs change which reads reach the storage; the KV abstracts draw the holds, fitted at the conditions their parameter file records (`provenance.conditions`), and any other engine or tier is another workload. | The reference model, accelerator, TP, and CPU tier are fixed by the WG, not chosen by a submitter (the accelerator is emulated; a submitter would otherwise buy the largest tier to unload the storage). ~~The CPU tier is 0 (LMCache's `local_cpu` off; decided 2026-10-08);~~ G1 (HBM) and G2 (host DRAM, LMCache's L1) at realistic WG-fixed sizes, the whole hierarchy modelled (the user, 2026-11-13, superseding the CPU tier 0 of 2026-10-08; NVIDIA's G1 to G4 adopted as the tier names, `DESIGN_REVIEW.md` §3.99 note "Tier names"); the reference holds must be fitted at the reference engine. | §6 item 24; `DESIGN_REVIEW.md` §3.64 |
| Dataset sizing against client DRAM (added 2026-10-10) | Dataset size is a parameter of the case (`DESIGN_REVIEW.md` §3.66, kind (a)); a use case's working set may fit in cache. Caching anywhere inside the solution, kernel client or user-space object client included, is fair (interposition test, §5). aeiou reports client hosts, GPUs per client, client memory, dataset bytes against it, and the client hit ratio where counters exist; it caps nothing. *(2026-10-21, `DESIGN_REVIEW.md` §3.78 point 10: a rules file's constraint over `fact.` names is checked at the gate, so aeiou enforces a rule like this one when a rules file carries it, without hard-coding it.)* | The dataset is at least 5× the clients' total DRAM, on every path (`O_DIRECT` and object included, since a solution's client may cache there too), because the clients are not production nodes: a submitter emulates as many GPUs per client as it can (64 against a real node's 8), and without the rule adding clients adds cache. It serves comparability and costs fidelity for small datasets (§6 item 28). | §5 sizing row; `DESIGN_REVIEW.md` §3.67 |
| Corpus sizing and layout (added 2026-10-12; revised 2026-10-13) | ~~A sizing step derives a training corpus's layout (samples per file from a format's byte target, files per directory, directories) and `steps` from epochs, rounds the samples up to whole directories, prints the projection, and writes a parameter file; the runner stays exact (`DESIGN_REVIEW.md` §3.71).~~ *Revised 2026-10-13 (`DESIGN_REVIEW.md` §3.74):* aeiou takes the counts as parameters and stays exact; `aeiou dry-run` projects each dataset's samples, files, directories, and bytes; `datagen` refuses a corpus that will not fit a directory root's free space; the report lists every host's memory, CPUs, and GPU ids beside the datasets' bytes; `div` in a pattern takes a parameter, so a directory width scales. | The sizing step itself is the WG's, in `mlpstorage` (MLCommons' driver, to drive aeiou as the generator of all four MLPerf Storage workloads in place of DLIO, the Milvus harness, and the KV-cache simulator): it probes the hosts, applies the rules below, and writes the parameter file aeiou takes. The layout rule's targets (the directory width, the class cap, the byte targets), the epoch count, and the reference `sync_every` are the WG's, fixed for every submitter at a corpus size; MLCommons' rules: data at least 5× node memory, at least 500 steps an epoch, one warm-up run of at least 500 steps then five independent runs of 500 steps (each at most one pass; confirmed by the user 2026-10-12). | `DESIGN_REVIEW.md` §3.71, §3.74 |
| Upstream requests | ~~dgen-py's API is what it is.~~ No longer a dependency (2026-10-11). | ~~The `fill_block`/`seek` request goes through the WG leadership channel.~~ None outstanding. | `DESIGN_REVIEW.md` §3.17, §3.70 |
| Driver integration (added 2026-10-09) | `aeiou` is a command: one invocation is one process, its report counts that process's CPU, RSS, and thread peaks (`getrusage`, §4 of `runner/REFERENCE.md`), its limit checks are that process's, and the multi-host launch is a process per host. Under the object feature, `s3dlio`'s tokio runtime starts at the first object call and stays until the process exits (`DESIGN_REVIEW.md` §3.65). | When `mlpstorage` drives `aeiou`, it runs each phase as its own `aeiou` process (fork/exec, `subprocess`), never by linking the runner or `s3dlio` into the Python process: otherwise a phase that touched object storage leaves tokio worker threads (and the Rayon pool that importing `s3dlio`'s Python module builds) running through every later phase, and the Python process's own CPU, memory, and threads land in a phase's measurements. | `DESIGN_REVIEW.md` §3.65, *Driver integration* |
| Checkpoint write and restore | A write run leaves a namespace manifest; a restore run declares the namespace `input`, reads it, and reports the gap and the warm reads (§5, *Checkpoint restore inputs*). | The benchmark runs them as two invocations of the same host list, the restore with `--rank-rotate 1` so no host reads its own shards; the gap between the end of the write and the start of the restore is capped at 30 s (`--max-gap 30`); a failed DP=N job restarts as DP=N (no resharding, `replicas` stays 1 for fully sharded state); `readback` stays off in a scored write. | `DESIGN_REVIEW.md` §3.24; `ABSTRACTS.md` §3–§4 |

**Naming convention (decided 2026-09-30).** One name family, `aeiou`, for everything a user
types or imports. The name expands to **Author Execute I/O** (2026-10-04): an I/O benchmark
where one *authors* a workload and then *executes* it, which is what the binary's `about`
line says:

- The Rust runner is the binary `aeiou`, with subcommands for its modes: `aeiou run`,
  `aeiou dry-run` (`--metrics`), `aeiou datagen`, `aeiou check`, and the coordinator/launch
  helper when it exists. This is the command a submitter, or anyone else, runs. (`check` and
  `dry-run` exist since 2026-09-30; `runner/REFERENCE.md`.)
- The Python authoring tools are dash-suffixed helpers in the same family, one per job:
  `aeiou-build` (the builder; renamed from `abstract-build`), `aeiou-params` (parameter files,
  2026-09-30), `aeiou-trace` (the metrics of an `strace` and their comparison, 2026-10-01; a
  binary of the runner's crate since 2026-10-29, `DESIGN_REVIEW.md` §3.89, its name kept, its
  digest and its interview Rust since 2026-10-31, its `compare` ~~the Python package's until
  ported~~ since 2026-11-01),
  `aeiou-catalog` (the workload catalog's records, shapes, and `CATALOG.md`, 2026-10-10),
  `aeiou-header` (a container header from the format's own library, run by `aeiou datagen`
  from beside its binary, 2026-10-12), and later
  `aeiou-verify` (the offline content verifier) and `aeiou-fit` (trace fitting). Dash-suffixed because they live in
  a separate install from the runner and must not fight it for one `aeiou` command.
- The Python package is `aeiou`; Rust crates are `aeiou` and `aeiou-<part>`.
- Environment variables are `AEIOU_<FLAG>` (the long flag upper-cased, `_` for `-`), the
  config file is TOML named by `--config FILE` or `AEIOU_CONFIG` and never searched for, and
  the precedence is command line, environment, file, default; the workload's identity is the
  command line's alone (2026-10-04, `runner/REFERENCE.md` §14, `DESIGN_REVIEW.md` §3.60).
- Nothing is named after MLPerf; the package was renamed from `mlps_abstract` on 2026-09-30
  for this reason.
