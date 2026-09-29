# Design Review — Abstract-Driven I/O Benchmark Runner

Status: review of the pre-implementation design (2026-09-25). Reviewed documents:
`PROJECT_BRIEF.md`, `NAPKIN_MATH.md`, `GRAMMAR_OPTIONS.md`.

The fixes listed here have been applied to those documents. This file records the reasoning so
it is not lost when the documents are later condensed.

## 1. Verdict

The design is sound. The Feistel consumer, positional sharding by global GPU id, counter-based
RNG, computed filenames, and the closed-loop actor model are the right calls, and the napkin math
holds up. The gaps are concentrated in the determinism story: several places where the documents
say "exact" but the mechanism as written is not. All of them are cheap to fix before code exists.

## 2. Determinism gaps (fixed)

### 2.1 The fingerprint was order-dependent

The original plan kept a per-actor rolling hash of ops "in issue order". A GPU actor has W worker
sub-actors that run concurrently, so its issue order changes with storage timing. Two identical
workloads would produce different fingerprints.

**Fix.** The fingerprint is a sum modulo 2^64 of per-op hashes. Each op hashes
`(actor id, enclosing loop indices, op, file id, offset, len)`. The sum is order-independent,
still collision-resistant for this purpose, parallelizable, and reduces across ranks by simple
addition. See `NAPKIN_MATH.md` §8.D.

### 2.2 Per-actor position counters break with workers

"Position counters are per actor" is ambiguous. If the W workers of one GPU share a counter, the
file each worker draws depends on completion order.

**Fix.** The permutation position is a formula, not a counter. GPU g, batch b, item j uses
position `g + G·(b·B + j)`, and batch b is built by worker `b mod W`. This is exactly
`DistributedSampler` (rank takes positions `rank, rank+R, …`) followed by the round-robin
`BatchSampler` and the DataLoader's round-robin worker assignment. See `GRAMMAR_OPTIONS.md` §2.

### 2.3 Randomness should be positional, not counted

A `draw#` counter is per-actor state that must be maintained and reset correctly, and the
"step" of a worker sub-actor is a batch index, not the GPU's step.

**Fix.** Every loop binds an explicit index variable. Randomness at any site is
`hash(seed, actor id, site id, vector of enclosing loop indices)`. Consequences:
- no draw counters, so nothing to reset at epoch boundaries;
- `{step:06}` in a checkpoint path has a defined referent;
- "what does GPU 17 issue at step 300" is computable without simulating steps 0–299, which makes
  `--dry-run` fast and parallel and makes debugging a single actor practical.

### 2.4 The loader must be bounded to exactly `steps` batches

As written, workers keep prefetching after the GPU actor's last step. The op multiset then
depends on when the run stops and how in-flight ops are handled.

**Fix.** The loader knows its total batch count (`steps`, like a sampler of finite length) and
dispatches exactly that many batches. Nothing needs cancelling at the end of the run and the
fingerprint stays exact.

### 2.5 Epoch wrap must be aligned across GPUs

When N is not divisible by G·B, different GPUs would switch permutation keys at different batches.

**Fix.** `drop_last` semantics: an epoch is `floor(N / (G·B))` steps long, and every GPU switches
keys at the same step. Positions beyond that within an epoch are never drawn.

### 2.6 Dataset seed must be separate from run seed

`dist.sample(hash(seed, file_id))` was ambiguous about which seed. If it is `--seed`, changing
the run seed silently changes the expected file sizes and the dataset no longer matches.

**Fix.** The dataset definition carries its own fixed seed. `datagen` writes a manifest at the
corpus root (pattern, count, size distribution, dataset seed, generator version). The runner
validates the abstract's dataset declaration against the manifest before starting.

## 3. Architecture improvements (applied)

### 3.1 Generalize `loader` into `parallel` + `channel`

The VM needs channels anyway for `take`. Making `parallel(W)` and a bounded
`channel(capacity, ordered)` the primitives means PyTorch's loader is syntactic sugar, and
tf.data-style unordered interleave, DALI, or a checkpoint-writer pool need no new VM instructions.
This is the cheapest insurance against R4 (language under-designed).

### 3.2 Spike 1 gets a hypothesis

NFS direct I/O is AIO-capable: with a non-synchronous kiocb the NFS client issues the RPCs and
returns `-EIOCBQUEUED`. An io_uring O_DIRECT read on NFS should therefore release its io-wq worker
immediately, while a buffered cache-missing read holds a worker for the whole RPC round trip.
If confirmed, O_DIRECT is doubly motivated: it avoids reclaim, and it keeps the `iou-wrk` count
small at tens of thousands of outstanding reads. `openat`/`close` will still punt either way, so
Spike 1 must measure worker count separately for open-heavy and read-heavy phases. **[verify]**

### 3.3 The blocking thread pool is the fidelity reference

One blocking OS thread per worker actor is literally what PyTorch does. It is the ground truth for
the A/B, not a fallback. io_uring is the scaling lever. Build the pool backend first: it is smaller
and gives the baseline.

### 3.4 Shared sink buffers understate client cost

Copying every read into a hot 2 MiB region per thread keeps it in L2 and hides the memory
bandwidth a real loader pays. Since client CPU per op is the key A/B metric, each thread gets a
buffer ring larger than L3 (tens of MiB) so the copies behave like real ones.

### 3.5 Per-step stall time per GPU

A few bytes per step per GPU, not per-op logging. It shows the cache-fill transition of §8.C
directly. Histograms are additionally bucketed by step range so steady state can be selected
after the run rather than by a warmup flag guessed beforehand.

### 3.6 Dry-run reports the numbers that matter

`--dry-run` prints total ops and bytes per host and compares bytes to host DRAM. The 5× rule does
not govern a 500-step run (§8.C); bytes read per host versus DRAM per host does.

### 3.7 Coordination behind a trait; MPI replaced by a TCP coordinator (2026-09-28)

The review originally kept MPI behind a `Coordinator` trait and a cargo feature. On 2026-09-28
the user adopted a follow-up suggestion (originating from a Gemini review) to drop MPI entirely:
the runner needs only a start gate, a periodic barrier, one end-of-run reduction, and stop, which
is a few hundred lines over TCP, and MPI's cost is an OpenMPI/MPICH install plus libclang on the
build host and an ABI-matched MPI runtime on every bare-Linux client. tonic/gRPC was considered
and rejected: HTTP/2, protobuf codegen, and a second async runtime buy nothing for one barrier
per ~150 s. The adopted design is blocking `std::net` on one coordinator thread, star topology,
length-prefixed `postcard` messages, a config hash in `Hello` to replace the protection
`mpirun`'s single command line gave, and `pdsh`/ssh for launch. Details in `NAPKIN_MATH.md` §8.A.

### 3.8 Open-file limits are a startup check and a reported number

2,000 GPUs per host × W=8 is 16K open files per host: `RLIMIT_NOFILE`, a matching fixed-file
table, and hundreds of thousands of NFSv4 open stateids server-side across the cluster.

### 3.9 Event-loop gotcha

With only timers pending and no I/O outstanding, a plain `submit_and_wait` never wakes. Use the
submit variant with a timeout equal to the next timer-wheel expiry.

### 3.10 Local NFS on WSL2

Export a tmpfs directory with `nfs-kernel-server` and mount it from `localhost`. No performance
meaning, but it exercises the real NFS client paths (io-wq punting, O_DIRECT on NFS, attribute
caches), which ext4 does not.

### 3.11 mmap-based loaders ~~are out of scope~~ (reversed 2026-09-28)

The review excluded mmap because page faults cannot be expressed as io_uring ops. On 2026-09-28
the user added the requirement that the same abstract run over many I/O initiation APIs chosen
on the CLI, and mmap became one of those backends (it is how safetensors, Arrow-backed Hugging
Face datasets, and llama.cpp load). The abstract stays POSIX-shaped; the `mmap` backend maps
`read(f, off, len)` onto populating that range of a mapping on the thread pool.

### 3.12 Selectable I/O backends (added 2026-09-28)

The user's list was buffered POSIX, O_DIRECT POSIX, NIXL-to-POSIX, GDS, POSIX AIO, and io_uring.
Additions from the review: libaio (the kernel AIO path; glibc POSIX AIO is a user-space thread
pool and tracks `sync`), mmap, libnfs (user-space NFS client with no kernel caches, a floor for
client CPU per op), GDS sub-modes with mandatory compat-mode detection (R15), and S3/object as a
deferred non-POSIX class. Backends are classified on three axes (initiation, completion, memory
target) with `--buffer` and `--cache` as orthogonal flags and a validity table. Full table in
`PROJECT_BRIEF.md` §4; risks R14–R17 in `NAPKIN_MATH.md`. The comparison policy (backend is part
of the application; fix it when comparing storage) was decided by the user on 2026-09-28:
CLOSED uses the framework's real API, `sync` for PyTorch; the other backends exist to measure the
SUT's speed of light and to advise solution and framework implementors. Recorded in the brief's
§5.

### 3.13 The interposition test, data verification, and seed privacy (added 2026-09-28)

The user asked whether `LD_PRELOAD`-style interposition could translate almost any of the I/O
APIs into any other, and whether that observation could set the application/solution boundary.
The answer, now adopted: mostly yes, and the places where it fails are exactly the things the
abstract specifies. A shim cannot create concurrency the application lacks, cannot deliver into
GPU memory the application did not ask for, cannot intercept page faults, and cannot pick the
next file. Those are application. Everything a shim *can* do (user-space clients, O_DIRECT or
io_uring underneath buffered calls, kernel and mount configuration, transport, server) is
solution. Precedents: DAOS `libpil4dfs`, libnfs `ld_nfs.so`, cuFile compat mode, Darshan.
Coverage caveats (stdio bypasses the PLT, raw syscalls, fork-unsafe io_uring rings) are
implementation nuisances, not boundary questions.

Two requirements follow, both adopted. A vendor shim under CLOSED is legal by the test, and a
shim can return anything fast, so ~~`datagen` writes per-block headers `(magic, dataset seed, file
id, block offset)` and the runner verifies a sample of reads (R18)~~ data must be verifiable
(R18; *how* was revised on 2026-09-29, §3.16: reproducible content and an offline tool, no
headers). And a shim that knows the seed could prefetch, so the seed, abstract, and file order
are application-private and reviews re-run with a fresh seed (R19).

### 3.14 Data-dependent workloads: random is the null model, not the model (added 2026-09-28)

The user challenged an earlier claim that a data-dependent access pattern (VDB search, KV cache)
can be approximated by a random one, citing SSD firmware that recognizes and prefetches for
application-specific patterns. The claim was too cavalier. The corrected position, now in
`GRAMMAR_OPTIONS.md` §5: storage exploits a short list of properties (reuse, sequential runs,
size and popularity skew, dependency shape, write-then-read lag); random matches a real trace
only on dimensions storage cannot see; the Feistel shuffle removes accidental structure, so real
structure must be deliberately added back. Three model additions (distributions over ids and
offsets, positional recency references, positional write-then-read names), one stated limit
(cross-actor read-after-write is phase-separated or statistical), and two method additions
(a locality-metrics check next to the fingerprint, and a bounded replay mode for calibration).
Risks R20 and R21.

### 3.15 Paper abstracts written; what they demanded of the grammar (added 2026-09-29)

The eight abstracts of `ABSTRACTS.md` were written against the semantic model of
`GRAMMAR_OPTIONS.md` §2 and §5.2 without a parser or VM, as R4 prescribed. The model held for
the training shapes with no change. The other shapes each exposed one missing piece, and the
common thread is that the model had *files* and *steps* but the real applications also have
*pieces of files* (zip members, index sectors, inverted lists), *objects the run creates*
(checkpoints, KV blocks), *ops that are supposed to fail* (`ENOTTY`, `EEXIST`, `ENOENT`), and
*the actor's own identity* in names and conditions. None of these threatens the invariants;
they are node types.

The one substantive finding is in the KV-cache shape. `recent(site, d)` as specified in §5.2
is not self-consistent: a request that continues a conversation must name the blocks by the
conversation's *original* id, which the request `d` back may itself have inherited. The
reference has to be a recurrence, `conv @ r = conv @ (r − d)` or a fresh draw, evaluated by
walking the chain positionally. That is still a pure function of the position and needs no
stored history, but `--dry-run` becomes O(chain length) per request rather than O(1), and the
index space needs a warm prefix so early requests have something to reach back to. The
derived hit length (blocks the earlier request actually wrote) replaces a fitted `hit_len`
distribution, which is a fidelity gain: the reads land on blocks that exist. The proposal is
to make `x @ i` the primitive and `recent` sugar (`ABSTRACTS.md` §9.5).

Two things the abstracts made concrete for policy rather than grammar: the ImageFolder
directory walk at startup (G full walks of the tree) is real application I/O that no current
benchmark includes, and whether it is CLOSED is a WG question; and the `mmap` backend's value
shows up in model load (§4b), where tensor-parallel ranks fault in strided slices of every
shard and the RPC pattern is the kernel's choice, not the application's.

### 3.16 Containers, format classes, and the end of block headers (added 2026-09-29)

The user asked whether the Feistel shuffle could run over *samples* rather than files, with a
mapping from sample id to the containing Parquet, HDF5, or TFRecord file. Three decisions came
out of the discussion; the reasoning is kept here.

**Samples are the unit, but the shuffle structure is format-dependent.** Feistel over sample
ids is exactly `DistributedSampler` plus `__getitem__`, and one sample per file is the special
case, so it unifies the model rather than extending it. But it is faithful only for formats
that support cheap random access (HDF5, Arrow IPC, MDS, Megatron token files). Parquet and
TFRecord readers do not access samples randomly: TFRecord has no index, and a Parquet row is
reachable only by reading and decoding its row group. Production loaders over those formats
shuffle *shards* and stream sequentially within them, with an in-memory shuffle buffer. Random
sample reads inside Parquet would therefore be the null-model error of §3.14 in reverse, adding
randomness the real workload lacks. Hence two access modes declared per dataset (`map`,
`stream`), with the validator refusing combinations the format cannot do (R22). HF's
non-streaming path is a third shape worth its own abstract: a one-time sequential conversion of
Parquet to Arrow cache files, then memory-mapped random row access on the cache, whose location
(local disk or the shared filesystem) is a parameter the author states.

**Format classes with two contracts.** A built-in "Parquet" library is a good idea as long as
it stops at the format's edge. What a trace of the reader library shows regardless of
application (footer protocol, locate formulas, supported modes, layout writer, decode compute
slots) goes in the format class; what differs between Ray Data, HF streaming, tf.data and a
map-style dataset (access mode, interleave, prefetch, column projection) stays in the abstract.
A class that absorbed the second half would cost fidelity. Column projection is the example
that makes the split concrete: reading 2 of 50 columns turns one sequential row-group read into
many small column-chunk reads, and only the loader knows the projection. The classes live in
the Python builder and emit ordinary POSIX nodes; the Rust runner stays format-ignorant. Index
bytes (footers, chunk indexes) are read as I/O because the real reader reads them, but not
parsed: the layout is already known from the formula, and comparing the bytes to the expected
layout is free drift detection. This is also why the benchmark never needs to interpret sample
data: content-dependent control flow in real readers (TFRecord's length prefix, Parquet page
headers) is recomputable from the layout, decode CPU is a `compute` slot, and under `mmap` the
backend must still touch every page so the fault actually happens.

**Reproducible, not self-describing, data.** The 2026-09-28 header design was wrong on two
counts. A unique 32-byte header in every 4 KiB block makes every block unique, so any dedupe
engine, fixed-size or content-defined, sees a ratio of one; the dedupe control the generator is
supposed to provide is destroyed. And R18 claimed the header check would catch a warm cache from
a previous run; it cannot, because cached content is correct. What verification is actually for
is the benchmark's own validity: a corpus generated with the wrong generator version, seed, or
dedupe setting hands the SUT's data reduction a free lunch. A fabricating shim is fraud, and
MLPerf handles fraud with rules and review. So the expected bytes at `(dataset seed, file id,
offset)` become a pure function that a verifier regenerates, the runner keeps only structural
checks (byte counts, short reads, index bytes), and content verification moves to a Python tool
beside `datagen`, sharing its generator, run after generation and by reviewers, never in a
scored run. The user proposed dgen-py (MLPerf Storage's generator, fast, with controllable
dedupe and compression) for the payload; the one property it must have is positional generation,
producing the bytes at an arbitrary offset without the prefix. If it is a stateful stream
generator, a block-wise wrapper (block seed from `hash(seed, file, block)`, dedupe from a pool of
K block seeds) gives the same controls at the granularity dedupe engines already use.

## 4. Plan changes

- Paper abstracts first, derived from `strace` of real loaders. Added a fourth: checkpoint
  restore, the one workload where many ranks read the same files, so delegations and page cache
  matter. Drafted 2026-09-29 (`ABSTRACTS.md`, §3.15); traces still to be captured.
- Then: fix the determinism items in the docs (done), build the VM with `--dry-run` and the
  fingerprint against ext4 and loopback NFS, then Spike 1 on the real target with the thread-pool
  backend first.

## 5. Things reviewed and left as-is

- Feistel + cycle-walking over the smallest 2^k ≥ N. Correct and O(1). Small test domains
  (N ≈ 1000) give a weak permutation at 4 rounds but still a bijection, which is all tests need.
- The bitmap probe count N·H(N) and the DRAM table are correct.
- Op chaining off by default. A short read cancels the rest of an `IOSQE_IO_LINK` chain, and the
  round trip is µs against 100s of µs per NFS RPC.
- ~~MPI as the multi-host mechanism.~~ Superseded on 2026-09-28 by the TCP coordinator; see
  §3.7.
- Timer wheel for compute sleeps rather than one `IORING_OP_TIMEOUT` per actor.
