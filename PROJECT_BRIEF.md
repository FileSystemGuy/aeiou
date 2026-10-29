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
    conditions, the simulation, and the 2 GiB and 6 GiB pairs.

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
    records (`--applications`, `--application`; `aeiou/applications.py`), and the three site
    kinds beside the set, keyed, bulk write, and within one open, each with its facts (keys and
    requests; values per field, ranks and steps; size and queries) and questions, and the
    object-log readers of item 49. Not yet: digest-to-digest comparison, `aeiou-catalog disclose`. *2026-10-29, `DESIGN_REVIEW.md` §3.89 (designed with the user):* one
    observation document, `aeiou_digest: 2`, the digest's structure and the metrics of the same
    events, `aeiou-trace metrics` retired and `dry-run --metrics-json` writing the same document;
    `aeiou-trace compare` over two of them, structure by equality and numbers by the tolerance
    classes, templates paired by text, shape, or `--pair`, the runner's own I/O set aside; the run
    side from `dry-run --emit-trace` (item 49) or an strace of `aeiou run`; and the trace-side tools
    in Rust as the binary `aeiou-trace`, streaming, since a real trace runs to several GiB and the
    Python parser holds every event. Build order in §3.89.

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
    `aeiou_trace` format so the scaled trace is an inspectable file. The gaps the note must
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
    (`aeiou/objectlog.py`):* MinIO's audit log and S3's server access log read by `metrics`,
    `export`, and `digest`; not yet Ceph RGW's ops log, a proxy's log, or the merge of an
    object log with an strace on the clock.

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
| The KV workloads' engine and CPU tier | What an inference engine holds (its KV pool, HBM less the model) and what LMCache's CPU tier absorbs change which reads reach the storage; the KV abstracts draw the holds, fitted at the conditions their parameter file records (`provenance.conditions`), and any other engine or tier is another workload. | The reference model, accelerator, TP, and CPU tier are fixed by the WG, not chosen by a submitter (the accelerator is emulated; a submitter would otherwise buy the largest tier to unload the storage). The CPU tier is 0 (LMCache's `local_cpu` off; decided 2026-10-08); the reference holds must be fitted at the reference engine. | §6 item 24; `DESIGN_REVIEW.md` §3.64 |
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
  2026-09-30), `aeiou-trace` (the metrics of an `strace` and their comparison, 2026-10-01),
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
