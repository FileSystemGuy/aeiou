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
| Dataset identity | The dataset definition has its own seed, separate from `--seed`. `datagen` writes a manifest at the corpus root (pattern, count, size distribution, dataset seed, generator version); the runner validates the abstract against it before starting. **Extended 2026-09-30 (reasoning in `DESIGN_REVIEW.md` §3.21; contract in `schema/README.md` §6):** the manifest is `.aeiou-dataset.json` at each dataset's root, and its normative part is the *resolved dataset definition*, the dataset's `datasets` entry with every referenced parameter substituted, in canonical JSON, plus what the abstract does not know (payload generator and version, dedupe/compression settings, block size, format-class writer version). `aeiou run` resolves the same closure from its own abstract and the parameters in effect and requires an exact match; it never compares the whole parameter set (most parameters do not shape the dataset) nor the abstract hash (one corpus legitimately serves several abstracts). The abstract's hash and name and the full parameter values at datagen time are recorded as provenance, not compared. The manifest's canonical hash is the dataset id, printed with every result next to the AST hash and the parameters in effect. The seed alone is not the identity: it fixes the bytes given a definition, not the definition. |
| Backend order | The blocking thread pool (one OS thread per worker actor) is the **fidelity reference** and is built first. io_uring is the scaling lever. Both sit behind one trait. Spike 1 tests the hypothesis that O_DIRECT reads on NFS complete asynchronously and so keep the io-wq worker count bounded, while opens/closes still punt. |
| Measurement | Per-step stall time per GPU (not per-op logs); histograms bucketed by step range so steady state is selected after the run; `--dry-run` prints total bytes per host against host DRAM. Read sink buffers are sized larger than L3 so copies cost what a real loader's do. |
| Divisions and comparison policy | **Decided 2026-09-28.** The backend is part of the application, so the line between "must be the same" and "may vary" is the backend's call boundary: the backend implementation is benchmark code; everything it calls (libc, kernel, NFS client, libnfs, cuFile, drivers, network, server) is the solution under test. **CLOSED** (an MLPerf division; §8) fixes the application-level I/O interface to what the real framework uses: for PyTorch that is buffered POSIX with PyTorch's concurrency structure, i.e. the `sync` backend. The other backends are for speed-of-light measurement of the SUT and for advising implementors; results from them are reported separately and never compared against CLOSED results as if they were storage differences. Compare storage solutions with the backend fixed; compare backends with the storage fixed. **The boundary is stated as an interposition test (decided 2026-09-28):** the solution may do anything that a dynamic-linker shim (`LD_PRELOAD`) placed under the *unmodified* application could do, provided the application's dependency graph of operations, the buffers they target, their data, and their semantics are preserved. What a shim cannot do is therefore application: the concurrency structure (workers, prefetch depth, in-order delivery), the memory target (host vs. GPU), and the choice of the next file. What a shim can do is solution: user-space clients (libnfs, vendor clients), O_DIRECT or io_uring or libaio underneath buffered calls, kernel and mount configuration, transport, and the server. A vendor-supplied shim under CLOSED is legal by this test, which is why the two requirements below (data verification, seed privacy) exist. Below the line, an optimization counts only if the real workload would get it too: no benefit from the synthetic data, no cache warmth from a previous run, no re-reads within a run, no relaxed semantics the customer would not run. RPC counts, client CPU, and client DRAM are reported as audit data and cost, not as a score. |
| Data verification | ~~**Required (2026-09-28).** Because a shim or client may return anything, the application must be able to check what it read. `datagen` writes verifiable content: every 4 KiB block starts with a small header `(magic, dataset seed, file id, block offset)` followed by non-dedupable PRNG fill derived from the same tuple. The runner verifies headers on a sampled fraction of reads (`--verify-sample`, default 1 in 64 blocks; 100% in a `--verify` run) at negligible CPU cost, and counts mismatches as run failures.~~ **Revised 2026-09-29: data is reproducible, not self-describing.** A per-block header makes every block unique and defeats any dedupe control, so there are no headers. Instead the expected bytes at `(dataset seed, file id, offset)` are a pure function that a verifier can regenerate, at block granularity, with dedupe and compression ratios as generator parameters (§6 item 12). The **runner does structural checks only**: byte counts, sizes, short reads, and that index bytes it reads (footers, chunk indexes) match the computed layout. **Content verification is a separate Python `verify` tool** next to `datagen`, run after generation and by reviewers on request, never in a scored run. Rationale: the realistic failure is a wrong corpus (old generator, partial regeneration, wrong seed, wrong dedupe setting), which invalidates the result through data reduction; a fabricating shim is fraud and is handled by rules and review, and a warm cache from a previous run has correct content and is not detectable by any content check (it is handled by fresh seeds and dataset sizing). Checkpoint read-back remains as a workload phase (the restore shape), not a content check. The benchmark never interprets sample data; it moves the bytes into the destination memory (copy, or page-fault under `mmap`) and stops. Reasoning in `DESIGN_REVIEW.md` §3.16. |
| Multi-sample containers | **Decided 2026-09-29.** Datasets are sample spaces, not file lists. The Feistel shuffle runs over **sample ids**; one sample per file is the special case. A dataset declares its **access mode**: `map` (Feistel over samples, positions `g + G·(b·B + j)`, `drop_last` over samples; what map-style PyTorch does over HDF5, Arrow, MDS, Megatron token files) or `stream` (Feistel over *shards* with the same sharding formula, sequential reads within a shard, an in-memory shuffle buffer that produces no I/O; what tf.data, WebDataset, HF streaming and Ray Data do over TFRecord, Parquet, tar). Random sample access inside Parquet or TFRecord is refused by the validator because no production reader does it. Container layout is computed by formula: fixed samples per file, sizes from the dataset seed, per-file prefix sums computed once per `open` and held while open (O(open files), so the per-file invariant survives). `GRAMMAR_OPTIONS.md` §6. |
| Format classes | **Decided 2026-09-29.** Each container format is a built-in library with **two contracts**. The *format* contract carries only what a trace of the reader library shows regardless of application: the layout writer for `datagen`, locate formulas (sample → file, unit, offset, length), the reader's fixed protocol emitted as ordinary POSIX nodes (Parquet: 8 bytes at EOF−8 then the footer; HDF5: superblock and chunk index; TFRecord: none), the access modes it supports, and compute slots for decode and index parsing. The *loader* contract stays with the abstract: access mode, interleave, prefetch, column projection. The class is tied to a reader library and version (`parquet(reader="pyarrow")`) and is validated against a trace like any abstract. It lives in the Python builder; **the Rust runner stays POSIX-only and format-ignorant**, and index bytes are read as I/O but not parsed. `GRAMMAR_OPTIONS.md` §6. |
| Payload generator | **Decided 2026-09-29; check resolved the same day.** `datagen` is the term for writing the synthetic corpus (as in MLPerf Storage). Use **dgen-py / the `dgen-data` Rust crate** (Russ Fellows, MIT or Apache-2.0) as the payload engine behind our own positional wrapper. Investigation of `dgen-data` 0.3.0 source plus a test of the wheel (`DESIGN_REVIEW.md` §3.17): there is no seek API, but content is positional by construction. Block `i` (1 MiB) is the Xoshiro256++ keystream seeded with `seed + (i mod unique_blocks)`, with the last `(N−1)/N` of the block zero-filled for compression ratio N, so block `i` of a stream equals block 0 of a 1 MiB generator seeded `seed + (i mod unique_blocks)`; verified byte-for-byte, including the dedupe case. Consequences: (1) **dedupe is scoped to one generator's stream**, so corpus-wide dedupe must be our layer: the seed of a file (or of a 1 MiB block of a large file) is `hash(dataset seed, index mod (total / D))`, so that D units share content; a small-file corpus otherwise has a dedupe ratio of 1 whatever is requested. (2) Ratios that do not divide 1 MiB (3, 5) are off by one byte at the random/zero boundary under seed arithmetic; exact for 2, 4, 8. (3) Compression is bimodal per 4 KiB (pure keystream or pure zeros), the DLIO convention; stated in the manifest. (4) The fill algorithm changed once (back-references → zero fill, January 2026), so the manifest pins the `dgen-data` version and the verifier uses the same one. (5) Always seed; the library's default is unseeded. The Rust crate is called directly from `datagen` and the verifier. **Upstream request:** a public `fill_block(seed, block_index, ratios)` or `seek(offset)` would remove the seed-arithmetic workaround; the author and the user are both on the MLPerf Storage leadership team, and the request will be made once this code shows it is what the WG needs. |
| Seed privacy | **Required (2026-09-28).** The run seed, the abstract, and the resulting file order are application-private. The solution (anything below the interposition line) may not use knowledge of them, for example to prefetch the next file. A submission review may re-run with a fresh seed; the results must match within noise. The dataset seed is not secret (it is in the manifest) because it determines content, not order. |
| Coordinator | Star topology over plain TCP with blocking `std::net` on one coordinator thread; length-prefixed `postcard` messages (`Hello` with a config hash, `Ready`/`Start`, `Arrive`/`Release`, `Stats`, `Stop`, `Heartbeat`). No tokio, no tonic/gRPC. Sits behind a `Coordinator` trait; single-host runs use an in-process implementation. See `NAPKIN_MATH.md` §8.A. |
| Deployment | Bare Linux on the client nodes, **no containers**. |
| I/O crate | `io-uring` (Rust). |
| A/B testing | Agreed. Backend × cache mode × io_uring features × NFS mount options (`NAPKIN_MATH.md` §8.5). The key metric is client CPU per op. |
| Grammar | Three layers: authoring language, the AST contract, the Rust VM. The AST (JSON; ~~serde YAML/JSON~~ YAML dropped 2026-09-30, `DESIGN_REVIEW.md` §3.20) is the contract and the only thing the runner executes; the WG publishes ASTs and their hashes, submitters run those (WG process; §8). Leading candidate for authoring (2026-09-28): **Option D**, a Python builder package that emits the AST, with source→AST reproducibility enforced by CI (build twice, compare) and by the runner's validator. Python stays on the authoring station, never on client nodes. ~~**User has not yet chosen.**~~ **Decided 2026-09-30: Option D.** The AST JSON Schema is `schema/abstract-ast.schema.json` (v0.1, draft 2020-12), with the canonical form and the validator's semantic rules in `schema/README.md` and the first abstracts in AST form under `schema/examples/`. The nine constructs of `ABSTRACTS.md` §9 were accepted the same day and are in the schema. Next: the builder package, `aeiou-build --hermetic`, and the build-twice CI check. |

## 6. Open items / next steps

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
   Still to do here: the format-class reader protocols (item 15) and the parameter-file split.
3. Build the VM with `--dry-run` and the fingerprint; test against ext4 and a loopback NFS mount
   on WSL2 (see §7). Golden-test the fingerprint in CI. **`aeiou dry-run` done 2026-09-30**
   (`runner/`, `DESIGN_REVIEW.md` §3.22): the loader, the canonical hash, the validator (CI
   diffs it against `check.py`), the positional VM over all nine committed ASTs, and the
   fingerprint, with golden tests. The definitions the schema left to the runner (key, words,
   permutation, `consume` position, `x @ i`, `until_eof`, fingerprint) are in
   `runner/README.md` §2. Still to do: the I/O backends (`aeiou run`) and the ext4 / loopback
   NFS test, `--metrics` (item 14), `stream` access, and the per-op cost (150–350 ns; caching
   a bound handle's path is the first fix).
4. **Spike 1:** blocking thread pool first, then io_uring; buffered vs. O_DIRECT, for
   `open → read → close` against the real NFS target, measuring `iou-wrk` count separately for
   open-heavy and read-heavy phases. It decides the I/O backend and the cache strategy.
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
9. Startup checks: `kernel.io_uring_disabled`, `RLIMIT_MEMLOCK`, and `RLIMIT_NOFILE` computed
   from G, W, and the abstract.
10. Coordinator protocol and launch script (`pdsh`/ssh loop); test it on WSL2 with several
    ranks on `localhost`.
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
14. **Locality-metrics check and replay mode (2026-09-28).** `--dry-run --metrics` computes
    reuse-distance, sequential run-length, popularity skew, request-size, dependency depth, and
    read/write mix on the abstract's stream; a trace tool computes the same from a real trace; an
    abstract is accepted for a workload class only when they match within tolerances. A `replay`
    AST node holds a literal captured sequence for small-scale calibration; never CLOSED.

15. **Container formats (2026-09-29).** Write the format classes (Parquet/pyarrow, TFRecord,
    HDF5/h5py, Arrow IPC, WebDataset tar, MDS, Megatron `.bin`/`.idx`), each from a trace of its
    reader library; add a ninth abstract for streaming training over TFRecord or Parquet (the
    MLPerf Storage ResNet50/CosmoFlow shape) and a tenth for the Parquet→Arrow conversion pass
    plus training from a memory-mapped Arrow cache, with cache placement (local vs. SUT) as a
    parameter. Extend the `loader` sugar so items per batch are decoupled from reads per item.
    `GRAMMAR_OPTIONS.md` §6.

## 7. Environment

The development machine is WSL2 (kernel 6.18, 20 cores, 31 GB RAM), with no realistic NFS
target. A loopback NFS mount (`nfs-kernel-server` exporting a tmpfs directory, mounted from
`localhost`) exercises the real NFS client code paths for correctness. All performance work must
run on real Linux client nodes.

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
dgen-py is a generic payload generator that happens to come from the same community.

**What is WG process, and where it is referenced.**

| Topic | Generic fact underneath | WG-specific part | Where |
|---|---|---|---|
| Roles | The AST is content-addressed; anyone can author and run one. | The WG authors the reference workloads and publishes their ASTs and hashes; submitters run the published hash and nothing else counts as "official". | `GRAMMAR_OPTIONS.md` Option D "Roles"; `schema/README.md` §0 |
| Divisions | The backend is part of the application; the interposition test (§5) draws the line between application and solution. | **CLOSED** scores only the backend the real framework uses (`sync` for PyTorch); other backends are speed-of-light rows; `replay` and wall-clock-bounded phases are never CLOSED. Whether a startup phase (the ImageFolder walk) is inside the measured window. | §4 backends, §5 comparison policy; `DESIGN_REVIEW.md` §3.12–3.13, §3.15; `ABSTRACTS.md` §1, §9.8; `NAPKIN_MATH.md` §8 |
| Data verification and seed privacy | Data is reproducible from (dataset seed, id, offset); the run seed and file order are private to the run. | Motivated by submission fraud under the interposition test; the offline verifier is an audit tool for the WG's review process. | §5; `DESIGN_REVIEW.md` §3.13, §3.16 |
| Reference parameters | Every workload has parameter slots filled from configuration, measurement, and traces. | Which values are the reference set (batch sizes, step times, dataset scale relative to client DRAM, the 500-step bound) is a WG decision recorded in the published parameter files. | §5 decisions; `ABSTRACTS.md` `[measure]` slots |
| Workload selection | The builder can express any POSIX-shaped skeleton. | The ninth and tenth abstracts follow the MLPerf Storage ResNet50/CosmoFlow and Parquet→Arrow shapes because those are what the WG submits. | §6 item 15; `GRAMMAR_OPTIONS.md` §6.5 |
| Upstream requests | dgen-py's API is what it is. | The `fill_block`/`seek` request goes through the WG leadership channel. | `DESIGN_REVIEW.md` §3.17 |

**Naming convention (decided 2026-09-30).** One name family, `aeiou`, for everything a user
types or imports:

- The Rust runner is the binary `aeiou`, with subcommands for its modes: `aeiou run`,
  `aeiou dry-run` (`--metrics`), `aeiou datagen`, `aeiou check`, and the coordinator/launch
  helper when it exists. This is the command a submitter, or anyone else, runs. (`check` and
  `dry-run` exist since 2026-09-30; `runner/README.md`.)
- The Python authoring tools are dash-suffixed helpers in the same family, one per job:
  `aeiou-build` (the builder; renamed from `abstract-build`), and later `aeiou-verify` (the
  offline content verifier) and `aeiou-fit` (trace fitting). Dash-suffixed because they live in
  a separate install from the runner and must not fight it for one `aeiou` command.
- The Python package is `aeiou`; Rust crates are `aeiou` and `aeiou-<part>`.
- Nothing is named after MLPerf; the package was renamed from `mlps_abstract` on 2026-09-30
  for this reason.
