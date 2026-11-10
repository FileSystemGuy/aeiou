# Design Review — Abstract-Driven I/O Benchmark Runner

Status: review of the pre-implementation design (2026-09-25). Reviewed documents:
`PROJECT_BRIEF.md`, `NAPKIN_MATH.md`, `GRAMMAR_OPTIONS.md`.

The fixes listed here have been applied to those documents. This file records the reasoning so
it is not lost when the documents are later condensed.

**Names (2026-10-10).** The abstracts were renamed for their directories under `workloads/`
(`DESIGN_REVIEW.md` §3.66, which has the table: `train_small_files` is `file_per_sample`,
`kv_cache_serving` is `lmcache_local_disk`, …). Sections before §3.66 keep the names and paths they were
written with.

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
validates the abstract's dataset declaration against the manifest before starting. Extended on
2026-09-30 to the resolved dataset definition and a dataset id (§3.21).

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
(a locality-metrics check next to the fingerprint, and a bounded trace mode for calibration).
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
to make `x @ i` the primitive and `recent` sugar (`ABSTRACTS.md` §9.5). Accepted 2026-09-30
(§3.18).

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

### 3.17 dgen-py: positional by construction, not by API (added 2026-09-29)

The open question from §3.16 was whether dgen-py can produce the bytes at an arbitrary offset
without generating the prefix. Answered by reading `dgen-data` 0.3.0 (`src/generator.rs`,
`src/rolling_pool.rs`) and testing the PyPI wheel on Python 3.12.

**API:** the streaming `Generator` has `fill_chunk` from the current position, `reset`,
read-only `position`, and `set_seed`, which restarts the block-index epoch at the current
block. No seek. `generate_buffer` and `BufferPool` take no seed at all.

**Algorithm:** the stream is cut into 1 MiB blocks (`BLOCK_SIZE`). For a generator of
`nblocks` blocks and dedupe ratio D, `unique_blocks = round(nblocks / D)`. Block `i` is filled
by `Xoshiro256PlusPlus::seed_from_u64(seed + (i mod unique_blocks))`; for compression ratio N
the last `(N−1)/N` of the block is zero-filled (a January 2026 change from back-references,
adopted because it matches DLIO and is much faster). No state crosses block boundaries. Hence
block `i` of stream S equals block 0 of a 1 MiB generator seeded `S + (i mod unique_blocks)`.

**Test on the wheel:** an 8 MiB stream with seed S, no dedupe: every block `i` matched a 1 MiB
generator seeded `S + i`. Dedupe 2:1 over 8 blocks: block `i` equalled block `i mod 4`, and
block 5 equalled the positional seed `S + 1`. Compression 2:1: first half keystream, second
half zeros, as read.

**What follows for the design.**
- Positional regeneration is seed arithmetic on the public API, at 1 MiB granularity; a 4 KiB
  piece is a slice. Datagen and the verifier call the Rust crate directly.
- Dedupe is scoped to a generator's stream, so it is ours to build corpus-wide: a file's seed
  (or a 1 MiB block's, for large files) is `hash(dataset seed, index mod (total / D))`. Without
  this a small-file corpus, where every file is a single block with its own seed, has a dedupe
  ratio of 1 regardless of the requested value. The "block-wise wrapper" of §3.16 is therefore
  required, not a fallback; dgen supplies the fast fill and the compression layout.
- Ratios whose zero-fill length does not divide 1 MiB (3, 5) spread a one-byte remainder over
  unique blocks by error accumulation; seed arithmetic then differs by one byte at the boundary.
  Exact for 2, 4, 8. Reproducing the accumulation is trivial if ever needed.
- Compression is bimodal per 4 KiB (pure keystream or pure zeros). A per-chunk compressor still
  lands at ratio N over a block; it is not what real data looks like. Stated in the manifest.
- Version pinning is mandatory: the fill algorithm has already changed once. The manifest
  records the `dgen-data` version and the verifier uses the same one.
- The library defaults to unseeded generation and its documentation discourages seeds; we always
  seed.

**Upstream.** A public `fill_block(seed, block_index, ratios)` or `seek(offset)` would remove
the workaround. The author (Russ Fellows) and the user are both on the MLPerf Storage
leadership team; the request will be made once this code shows it is what the WG needs.

### 3.18 The nine constructs accepted, Option D chosen, and what the AST schema fixes (added 2026-09-30)

The user accepted the nine construct proposals of `ABSTRACTS.md` §9 on the recommendation given
at the end of the 2026-09-29 session, chose Option D for authoring, and asked for the AST schema
to be started. Three of the decisions carry reasoning worth recording.

**§9.7 `namespace`: sizes are computed, never observed.** The proposal said the sizes of
workload-created objects are "known only from the writes", and `size = as_written` invited the
reading that the runner learns an object's size by remembering what it wrote or by `fstat`.
Either is state: a per-object record that must survive across positions and actors, and a value
that depends on which writer ran first. Neither is needed. Every write length in the model is
already a positional expression (the schema has no other kind), so the size of an object is the
sum of the write lengths in the sequence that created it, computable at the creator's position
with no I/O and no record. A reader at another position can reach the same expression through
`@` when it involves drawn values (the KV-cache `total @ (r − d)`), and otherwise states the
length outright. `as_written` therefore stays as a label meaning "the sum of the creating
writes", with one validator rule: `until_eof` on such an object is allowed only through the
handle binding the writes used, which is the same position by construction. `fstat` on a
namespace object is issued when the application issues it, checked structurally when the size
is computable, and never consumed. This keeps the "never materialize per-file structures"
invariant intact for objects the run itself creates.

**§9.6 `regions`: the naive layout, on purpose.** The alternative (slot groups with prefix sums
computed on the fly) was raised and declined on 2026-09-29; the user prefers a simple
assumption about how the filesystem treats unwritten ranges, and the assumption is sound: with
delayed allocation the written regions land contiguously and holes cost bounded metadata, and
real scientific formats with alignment already contain such holes. The residual cost is the cap
of `size(c)` at `slot`, which truncates the request-size distribution; it is an acceptance
metric, so a trace will show if it matters.

**§9.5 `x @ i` replaces `recent`.** No counter-argument survived: `recent(x, d)` is exactly
`x @ (i − d)`, so keeping both would be two spellings of one node, and the chain recurrence the
KV-cache shape needs is expressible only with the general form. The syntactic rule (a
self-reference's index is `i − e` with `e ≥ 1`) is what makes termination checkable without
evaluation.

**Option D, and what the schema commits to.** Option D was chosen as recommended; the
arguments in `GRAMMAR_OPTIONS.md` Option D stand and nothing new was raised. Writing the schema
forced the choices that the prose had left open, listed here so they are visible:

- *Externally tagged nodes.* Every node, expression, distribution, and handle is an object
  with a single key naming its kind (`{read: {…}}`, `{add: […]}`, `{zipf: {…}}`), which is
  serde's default enum encoding, diffs well, and reads as the Option B sketch did.
- *Canonical units.* Sizes are byte integers and durations are nanosecond integers; `1MiB`,
  `105ms`, and `40us` are builder-side spellings. Floats appear only where the value is a
  probability, an exponent, or a distribution parameter, and the builder formats them with
  `repr`. No site ids: a draw's site is its structural path in the tree, so two builds of the
  same source cannot disagree about it.
- *Small node set, sugar in the builder.* `every`, integer `repeat`, parameter tables,
  `recent`, `file("literal pattern")`, and the format-class protocols are all builder
  constructs that serialize as loops, conditionals, arithmetic, and namespace objects. The VM
  sees thirteen control statements and seventeen ops. `until_eof` and `until_end` are the two data-dependent repeats
  and stay in the AST because the runner resolves them from the layout.
- *`read` with an `offset` is positioned I/O; without one it is sequential from the current
  file position.* The abstracts mixed the two spellings; the builder emits `lseek` + `read`
  when the trace shows that pair and `read{offset}` when the reader library used `pread`.
- *Every actor template has a `count` expression* defaulting to the reserved parameter `gpus`,
  which the CLI sets; a single-builder workload (`ABSTRACTS.md` §7) sets `count: 1`.
- *Deferred, marked as such in the schema:* the `replay` node's trace format (the node is `trace` since 2026-10-02, §3.58), the container
  layout fields the format classes will need beyond `samples_per_file`, and the `stream`
  loader knob for records per batch. Each is a version bump, not a redesign.

### 3.19 The builder package, and what building the eight abstracts changed (added 2026-09-30)

`builder/aeiou` implements Option D (`builder/REFERENCE.md`). Every `ABSTRACTS.md`
workload is now an authoring script, and the three ASTs hand-written against the schema
regenerate hash-identical from their scripts, which is the test that the builder emits exactly
the contract and nothing else. Decisions taken while writing it, each small, listed so they are
visible:

- **No pydantic, no second copy of the rules.** The sketch said "typed constructors validated
  with pydantic against the same schema the runner uses". A pydantic model of the AST would be
  a second schema to keep in step with the JSON Schema. Instead the builder's nodes are plain
  slotted classes with construction-time checks for what only the builder can see (Python
  control flow over a symbolic value, a callable in a node, an `x @ e` on the binding being
  defined with a non-decreasing index, `until_eof` through the wrong binding, index shadowing,
  a hand-unrolled loop), and `build()` runs the published `schema/abstract-ast.schema.json`
  and `schema/check.py` on the emitted dict. The reference checker is the one enforcement
  point for the semantic rules, as technique 2 of Option D intended. Dependencies: ~~PyYAML and~~
  jsonschema only (§3.20), locked with `uv.lock`.
- **Bindings are named at the `let`.** `f = worker.let("f", train.consume())`, not
  `f = worker.consume(train)`: the builder cannot see the Python variable name, and the AST
  needs a name the validator and the runner's error messages can use. `draw(name, dist)` is the
  same for draws, and `ref(name)` names a binding defined later in the loop body for the `x @
  i` chains of §9.5. The construction-time V3 check reads the parameter defaults to prove
  `d ≥ 1`, so the KV-cache chain fails at the `let` if `reuse` loses its `min: 1`.
- **Integer repeats are loops, always.** `read(f, xfer, repeat=n)` emits a `loop` with a
  generated index `rep`; the schema's numeric `repeat` field stays for hand-written or
  converted ASTs but the builder never emits it. The loop gives each repetition its own RNG
  position, so a drawn length is re-drawn per repetition, and it keeps one spelling for the
  runner to optimize.
- **Sequential runs are `lseek` + sequential reads.** The §7 paper form read a shard as
  `read(b, offset = k·bytes, xfer)[n]`, which taken literally re-reads one offset `n` times.
  The script issues `lseek(SET)` then `n` sequential reads, which is what `ifstream` does. The
  same choice fixes the §2 member reads (positioned header read, `lseek`, sequential chunks).
- **`file("literal/pattern")` is not offered.** A namespace needs a seed and a size rule; the
  scripts declare namespaces explicitly and build objects with `ns.object(...)`. §4a therefore
  declares the `ckpt` namespace with the same seed §3 wrote it with, and computes its size from
  the item table, which is the rule of §3.18 applied from the reader's side.
- **A `regions` dataset's one file is `file {dataset, id: 0}`.** The schema had no spelling
  for it; `ds.file()` with no id emits this and `schema/README.md` records it.
- **Named for the repository, not the WG.** First written as `mlps_abstract`, renamed `aeiou`
  the same day at the user's request: nothing in the builder is MLPerf-specific, and names are
  the hardest thing to change later. `PROJECT_BRIEF.md` §8 now lists what is WG process.
- **Provenance is informational; the hash is the identity.** `git` is the short HEAD with
  `-dirty` when the script has uncommitted changes, so it is always one commit behind the
  commit that lands both script and AST. The script's `sha256` and the `ast_sha256` are exact,
  and CI compares hashes, not file bytes, so provenance never causes drift failures.
- **The hermetic harness is a child process, not a mode of the library.** `--hermetic` runs
  the script under `python -s -B` with a three-variable environment, audit hooks, and stubbed
  clocks and entropy, and returns the AST on stdout; the parent validates again and writes. The
  tests poison a script nine ways (unseeded `random`, `time.time`, `os.urandom`, `uuid4`,
  `datetime.now`, a socket, a subprocess, a read outside the allowed roots, a write) and check
  that each is refused under `--hermetic` and builds without it, so the harness is what
  refused it. `--twice` is checked with a script that iterates a set of phase names: it builds,
  validates, and fails the build-twice comparison, which is the failure mode technique 5 exists
  for.

Two things the exercise showed about the abstracts themselves: §4b's tensor table is four
parallel parameter arrays with a `when` on `split[t]`, which reads fine at four tensors and
will not at four hundred (the parameter-file split of Option D, "three sources", is the fix and
is still to do); and §5's `hops` needed a concrete `empirical` to build at all, so the
`[measure]` slots now hold placeholder distributions that a trace must replace before any
number from these ASTs is quoted.

### 3.20 One format on the contract: JSON, and why YAML went (added 2026-09-30)

The user asked why the builder used both JSON and YAML. The answer was that the hash had
always been computed over canonical JSON bytes (sorted keys, no whitespace, fixed float
formatting, which YAML has no agreed equivalent of), the schema is JSON Schema, and YAML was
on disk only for readability of generated files. On inspection that readability was buying
less than it cost:

- **Two grammars, two parsers, on the contract.** PyYAML implements YAML 1.1; the Rust YAML
  crates implement 1.2. They disagree on `yes`/`no`/`on`/`off` as booleans, on `1_000`, on
  `0777`, and on `1:30`. The emitter quoted defensively (`ast: '0.1'`), but a contract whose two
  implementations can parse the same bytes into different values is a determinism hazard of
  exactly the kind this project exists to exclude.
- **The Rust side is weaker for YAML.** `serde_yaml` was archived by its author in 2024; the
  forks are less trusted than `serde_json`. The runner would carry a second parser for a format
  nobody hand-writes.
- **Nobody hand-writes ASTs.** The builder emits them; review and diffs are of generated files.
  Pretty-printed JSON with the builder's key order reads well enough and diffs more cleanly
  than YAML flow style. The one thing lost is the header comment with the hash, which the
  `provenance` block already carries.

Decision: JSON only. `<name>.ast.json` on disk, pretty-printed with two-space indentation and
insertion order; the canonical form for the hash is the same document minified with sorted
keys. PyYAML leaves the builder's dependencies, `check.py` needs only `jsonschema`, and the
runner will need only `serde_json`. The nine ASTs were regenerated; their hashes did not
change, which is the point of separating the identity from the on-disk form. Files are
roughly twice the line count of the YAML, accepted.

### 3.21 The dataset manifest: what it compares, what it records, and where it hides (added 2026-09-30)

The user raised that an abstract has many parameters, some overridable on the CLI, and that a
hash mismatch between the abstract used for `aeiou datagen` and the one used for `aeiou run`
says only that they differ, not what parameters built the data. The proposal: the dataset must
be self-describing at its root, datagen writes a sentinel file with at least the parameters,
and `aeiou run` validates against it. Accepted, with three refinements that came out of asking
what exactly should be compared.

**Compare the resolved dataset definition, not the parameters and not the abstract.** A
dataset is a function of a small closure: its `datasets` entry plus whichever parameters those
expressions reference (`sys_prompts` and `chunk_bytes` shape the KV-cache dataset; `batch` and
`steps` do not). Comparing the whole parameter set would make every legitimate
`--param steps=1000` a mismatch; comparing the abstract hash would stop one corpus serving
several abstracts, which it must (checkpoint write and restore share a namespace, an evaluation
abstract will read the training corpus). So the normative content of the manifest is each
dataset's definition with parameters substituted, in canonical JSON, plus what the abstract
cannot know: the payload generator and its settings and the format-class writer version. The
abstract's hash and name and the full parameter values go in as provenance, the same split the
AST's own `provenance` block makes, so they are there for audit and never invite the wrong
comparison. The canonical hash of the normative content is the dataset id, and a result is the
triple (AST hash, parameters in effect, dataset ids). One counter-argument was considered and
rejected: "the dataset seed already identifies the data". It fixes the bytes given a
definition, not the definition; two abstracts with the same seed and different size
distributions produce different corpora.

**Not a reversal of "reproducible, not self-describing".** That decision (§3.16) was about
per-block headers, which force a dedupe ratio of one. One file at the dataset root has no such
cost, and it is metadata about the data, not data. The distinction is now stated in the docs
so the two decisions do not read as contradicting each other.

**The manifest must stay out of the data.** The user asked that the manifest be protected from
being taken for a dataset file, suggesting a leading `.`. Adopted, and made a rule rather than
a convention: the name is `.aeiou-dataset.json`, and validator rule V13 forbids any dataset or
namespace pattern component beginning with `.aeiou`, so no generated name can collide with it
or look like it; the runner's structural check on `readdir` ignores `.aeiou*` entries (the real
`getdents64` returns them, as it returns any dotfile, and ImageFolder-style walkers skip them);
datagen and the verifier treat the prefix as metadata. The leading dot also keeps it out of
naive globs and `ls`. Two datasets may not share a root, so "the manifest at the root" is
unambiguous.

Two rules the proposal implied and did not name, both now in `schema/README.md` §4 and §6.
Datasets are read-only (V12): nothing stopped an abstract from opening a dataset file for
writing, and the manifest can only describe the data if the run cannot change it. Namespaces
are the mirror image: they have no manifest, but stale contents change the workload (the
KV-cache `stat` with `expect: [ENOENT]`, `CREAT|TRUNC` semantics, the hit model), so
`aeiou run` refuses a non-empty namespace directory unless told to clean it. Also decided:
the manifest is written last and atomically, so a crashed datagen leaves none; exact match with
no relaxation for larger datasets, since `dirs`, `readdir`, and the `zipf`/`hotset` rank orders
all depend on the declared count; and the manifest is not a security boundary, since it sits on
the system under test, which is the fraud domain of rules, the published dataset id, and the
offline verifier, whose input the manifest now is.

### 3.22 The runner's first half: the VM, `aeiou dry-run`, and the definitions it fixed (added 2026-09-30)

`runner/` holds the Rust crate `aeiou` (`runner/REFERENCE.md`): the AST loader, the canonical
hash, the validator, the positional VM, and `aeiou dry-run`, which walks every actor
instance of an AST without I/O and prints op counts, bytes, and the fingerprint. All nine
committed ASTs run; `aeiou check` prints byte-identical output to `schema/check.py` over
them and CI diffs the two. Building it forced the choices the schema had left to the runner
(`schema/README.md` §5, "its exact definition belongs to the runner"). Each is listed in
`runner/REFERENCE.md` §2 and pinned by a golden test; the reasoning behind the ones that were
not obvious:

- **One hash family.** xxh3-64 for every key (draw sites, per-id dataset draws, permutation
  keys, the per-op fingerprint hash) and SplitMix64 for the words a draw consumes. One
  function to implement in any future verifier, no dependence on a crate's RNG stream
  semantics, and the key of a draw is a pure function of five integers. The site is the hash
  of the node's JSON pointer, which is what "no site ids" in §3.18 amounts to: the site is the
  structural path, and the pointer is the canonical spelling of a path.
- **`consume` generalized without a new construct.** The formula `g + G·(b·B + j)` is stated
  for the loader. The VM defines the batch frame as the nearest enclosing `loader` (else the
  innermost loop), `b` as the row-major ordinal of the frames down to it, and `j`, `B` from
  the frames inside it. For the loader sugar this is the formula; for an `epoch` loop around a
  loader, or a plain loop, it is the formula's obvious extension, and there is no second
  spelling for the author to choose.
- **`x @ i` by re-evaluation, not by history.** The shifted evaluation re-runs the binding's
  definition with its loop frame set to `i`; sibling `let`s of the same body are recomputed at
  that index and memoized for the duration; draws inside use the shifted index. This is the
  "walk the chain, one hash per link" of §3.15, and it needs no per-actor storage. The
  below-`from` rule is implemented literally: an `at` below the loop's start raises a signal
  that the enclosing `cond` expression catches by taking its other arm. The KV-cache chain
  test checks that every continued request names its origin's conversation.
- **`as_written` is an accumulator over the actor's own writes, by path.** The rule of §3.18
  makes the reader that uses `until_eof` on such an object the writer at the same position,
  so the VM keeps the sum of write lengths it issued per object path (reset by `TRUNC`,
  moved by `rename`, dropped by `unlink`) and resolves `until_eof` from it. Nothing is
  observed; a different actor or position cannot see the sum, which is what V4 requires.
  *(Revised 2026-11-03, `FS_API_SWEEP.md` S-19, the user's choice: the sum counted a rewrite
  twice and ignored a write past the end, `truncate`, `ftruncate`, and `fallocate`; the VM
  keeps the size POSIX gives instead, the end of the furthest write, set by a truncate,
  extended by a `fallocate` without `KEEP_SIZE`, shrunk and grown by a collapsed and an
  inserted range. A writer appending in order has the same size either way.)*
- **`zipf` without a table.** A rank comes from the inverse CDF of the continuous envelope
  of `r^-s` on `[1, N+1)`, O(1) per draw with no per-dataset array, and ranks map to ids
  through a permutation keyed by the *dataset* seed, so which ids are popular is a property
  of the corpus, not of the run. The envelope is an approximation of the discrete Zipf; the
  locality-metrics check (`PROJECT_BRIEF.md` §6 item 14) is where it would show if it mattered.
- **`readdir` is one op** in the fingerprint. How many `getdents64` calls it takes depends on
  the buffer size and entry lengths, which are backend and kernel facts, like the RPC count of
  a read.
- **The fingerprint hashes the effective offset and the requested length.** A sequential read
  therefore hashes the same as the positioned read at the same place (plus the `lseek` op
  itself), and a short tail read hashes by what was asked, with the expected count reported
  separately for byte totals. Actor id and the index vector are in the hash (`NAPKIN_MATH.md`
  §8.D), so the sum is sensitive to which position issued what.

Two observations from running the nine. Three of them (checkpoint write, restore, model
load) have no draw at all: their fingerprints are identical for every seed, which is the
right answer and a useful sanity check. And the per-op cost of the dry run is 150–350 ns,
dominated by formatting the path from the pattern on every op; `vdb_build_diskann` at its
defaults (203M ops) takes 33 s on one core. Caching a bound handle's resolved path is the
obvious fix and is not done.

### 3.23 The runner's second half: `aeiou run` with the `sync` backends, `aeiou datagen`, and the payload wrapper (added 2026-09-30)

`runner/REFERENCE.md` §4–§5 state what exists; this is why it is shaped that way.

- **A thread per actor and sub-actor, the VM's tree walk on it.** The alternative was to
  make the VM resumable (a state machine an event loop can park at every blocking call) and
  build the thread-pool backend on top of that. The brief's own ordering argues against it:
  the blocking pool is the *fidelity reference*, "literally what PyTorch does", and a PyTorch
  worker is a process that blocks in `read`. Running the walk on the thread makes the
  backend trivially faithful and small (`run.rs` is the `Sink`; `backend.rs` is a table of
  syscalls). The resumable VM is still needed for `io_uring`, and it will be built for that
  step, with this implementation as its oracle: same abstract, same fingerprint, same
  per-step stalls within noise.
- **The fork protocol keeps the dry run inline.** `Sink::fork` receives the fork kind and a
  closure that yields a `Snapshot` of the parent (frames with the fork frame last, visible
  bindings, phases, open files, as-written sums, the sub-actor body). The dry run declines
  and the VM walks the sub-actors in index order as before; the run sink takes the snapshot,
  resumes one VM per thread from it, and calls `run_sub(k)`. The closure is there so the
  dry run never pays for a snapshot (a `parallel` per hop in the search abstracts would
  otherwise clone bindings tens of millions of times).
- **Loader semantics: bound the batches started, not the batches delivered.** PyTorch's
  `_MultiProcessingDataLoaderIter` sends `prefetch_factor × num_workers` batch indices ahead
  and one more per batch returned; a worker cannot begin a batch it has not been sent. So
  worker `w` may start batch `b` only while fewer than `W × prefetch` batches are started
  and untaken, and delivery is in batch order. A `take` past `batches` and an actor ending
  with batches untaken are both errors rather than a hang or a silent truncation: the
  producer is finite and the consumer's count must match it (`NAPKIN_MATH.md` §4.1), and a
  mismatch is an authoring bug the run should name.
- **Barrier participants are counted statically**, as the instances of every template whose
  body contains the scope outside a `parallel` or `loader`. A dynamic count would be racy at
  startup. An instance that finishes leaves the barrier, and a generation completed by
  departures is released *and reported*, because a barrier some instances skip (one inside a
  `when (gpu == 0)`, say) is a workload that does not mean what its author thinks. Barriers
  inside sub-actors are refused for now; none of the nine abstracts needs one.
- **`sync-direct` rounds reads and refuses writes.** The `until_eof` idiom ends with a read
  that starts at EOF, which is unaligned for almost every file; a real application under an
  `O_DIRECT` shim (the interposition test, `PROJECT_BRIEF.md` §5) gets exactly this problem
  and the shim must round the read out and copy the requested part. The runner does that, so
  the op stream and the fingerprint are unchanged and only the bytes moved differ (audit
  data). An unaligned write would need read-modify-write, which is a different workload; it
  is refused with the position.
- **The payload wrapper: a 1 MiB dgen generator per block.** §3.17 established that dgen's
  content is positional by seed arithmetic and that dedupe is scoped to one generator's
  stream. Instead of reproducing dgen's stream layout, every block is its own 1 MiB generator
  seeded `labeled_key(seed, "payload", [unit, block])`; the prefix of a block does not depend
  on how it is read out (checked in `payload::tests`), so a verifier regenerates any piece
  through dgen-py's public API with no seek. Dedupe is then a property of which units share a
  seed: `unit = id mod ceil(files / D)` for a files dataset, a block modulus for the single
  file of a regions dataset. Namespace writes get the same treatment keyed by
  `(namespace seed, xxh3(path))`, which is what `schema/README.md` §5 promised
  ("namespace content is a function of (namespace seed, hash(path), offset)").
- **What the manifest compares.** The resolved definition (parameters substituted, `doc`
  stripped since documentation must not invalidate a corpus) is compared field by field; the
  payload block cannot be derived from the abstract, so it is recorded, folded into the id,
  and pinned by `--expect-dataset-id` where a WG publishes ids. This made the corpus sizes
  parameters: five abstracts had literal counts (`train_small_files` 50M, `train_large_samples`
  50k, the two search indexes, the KV-cache system prompts' length), which was fine for the
  fingerprint but wrong for the dataset rule, under which the count is chosen per submission,
  and impossible for a test that needs a 600-file corpus. The counts are now `files`, `nodes`,
  `lists`, `sys_tokens`, `sample_mean`/`sample_sd`; the defaults are the previous literals, so
  the golden fingerprints did not move.
- **Two abstracts changed on contact with a filesystem.** The KV-cache abstract wrote
  `kv/{conv}/blk_{k}` into a directory nobody created; LMCache and vLLM's file backends create
  it, so the abstract now does a `mkdir` with `expect: [EEXIST]` before a conversation's first
  write (fingerprint re-recorded). The DiskANN build kept its base vectors at the run root,
  where the manifest would have shared a directory with the namespaces; the file moved to
  `base/base.fbin`.
- **Open: the restore abstract's inputs.** `ckpt_restore` reads namespace objects that no
  actor wrote. Either the checkpoint being restored is a dataset (then datagen writes it, but
  a `files` pattern has only `id` and `k` fields and the shard files are named by rank inside a
  step directory), or a run must be able to keep a namespace another run wrote (a
  `--keep-namespaces` that the KV-cache hit model must never see). Not decided; the abstract
  is the one committed workload `aeiou run` cannot execute today.
- **Not done here**, deliberately: the TCP coordinator (the trait exists), the io_uring
  half of Spike 1, `mountstats` and per-backend counters, a per-actor pool for `parallel`
  sub-actors (the search abstracts spawn a thread per beam per hop), `RLIMIT_NOFILE` checks,
  and the loopback NFS run of `PROJECT_BRIEF.md` §7, which needs root on the development box.

### 3.24 Checkpoint restore reads what the write wrote: input namespaces, the namespace manifest, rank rotation (added 2026-09-30)

§3.23 left `ckpt_restore` as the one abstract the runner could not execute, because it reads
namespace objects no actor in it writes. The user's framing settled it: the benchmark's
restore is the real event of a job restarting on other nodes and loading the checkpoint it
just wrote, so the restore's inputs *are* the previous run's output, the gap between the two
is part of what is measured (the WG caps it at 30 s), and the read must not touch the
namespace it reads. Decisions and reasoning:

- **A namespace can be an input.** `input: true` on a namespace (schema, builder `input=`,
  rule V14 in `check.py` and `validate.rs`) means a previous run wrote it; every write-mode
  op on it is rejected at validation, the runner never empties its root, and it requires the
  writer's manifest there. Datasets were the wrong vehicle: a `files` pattern has only `id`
  and `k`, the shards are named by rank inside a step directory, and `datagen` would be
  inventing a checkpoint instead of the benchmark producing one. A `--keep-namespaces`
  escape would have put the decision in the launcher and silently legitimised the KV-cache
  hit model reading a stale namespace; `input` puts it in the abstract where the validator
  sees it.
- **One manifest per namespace root**, `.aeiou-namespace.json`, mirroring the dataset
  manifest: written last and atomically by the run that created objects there, read by the
  run that declares the namespaces `input`. Namespaces sharing a root share the manifest, so
  V14 also requires them to agree on `input`. The compared part is `pattern`, `fields`, and
  `seed`, the things that fix names and content. `size` is deliberately not compared: the
  writer declares `as_written`, the reader must state an expression (V4), and they are two
  models of the same bytes. A disagreement between them surfaces as a short read, which the
  structural check reports with the position; the first test run did exactly that when the
  test's item offsets were wrong by one tail.
- **Rotation is free because sharding is by GPU id.** Rank is only the host index that
  selects a GPU id range (`NAPKIN_MATH.md` §8.A), so `--rank-rotate k` changes which host
  runs which ids and nothing else: same ops, same names, same fingerprint. With the same host
  list as the write run and `k ≠ 0`, every host reads shards another host wrote. The manifest
  records each rank's host and range, and the reading run checks its own range against them
  (`--require-cold` turns the overlap into a refusal). The exact count comes from the
  per-object writer record: every object created is listed with its creating GPU id (capped
  at 100 000 objects, beyond which only the count is kept; a checkpoint namespace is G files
  per step, the KV cache is the case the cap is for), so the reader knows at each `open`
  whether the file was written on this host and reports the warm reads. With `--ranks 1`
  today every restore read is warm and the report says so; the TCP coordinator is what makes
  `--ranks` above 1 runnable.
- **Fan-in is a parameter, not an assumption.** `dp` in the restore abstract meant "ranks
  reading the same shard", not the job's data-parallel degree, and the conversation showed
  how easily the two are conflated; it is now `replicas`, 1 for fully sharded state (the
  MLPS case: every rank writes and reads its own shard), the replica count when DCP
  deduplicated replicated state. With `replicas = 1` a single reader per shard makes any
  non-zero rotation exact; with more, readers of one shard spread over hosts and the warm
  count is the honest answer.
- **The write's read-back is off by default.** `ckpt_write_dcp` kept its `ckpt_readback`
  phase from §3.15, reading each shard on the node that wrote it right after `fsync`. Once the
  restore is a separate run on other nodes, that phase measures the page cache; it stays
  behind `readback = false` for correctness runs and never appears in a scored write. The
  golden fingerprint moved accordingly.
- **Fidelity beyond the MLPS configuration.** The general structure (per-rank size tables,
  `replicas`, the intra-file protocol slots) is kept and the benchmark configures the fully
  sharded point of it. The gain from tensor and pipeline parallelism alone is small for
  storage (file names, uneven sizes); the real gaps are replicated state saved once and
  restored fan-in, the write protocol inside a shard, and asynchronous checkpointing that
  turns the burst into a background stream. Those are parameter and trace questions, not
  structural ones, which is why the structure stays general.

### 3.25 The TCP coordinator: several hosts, one run (added 2026-09-30)

`NAPKIN_MATH.md` §8.A fixed the design on 2026-09-28 (§3.7); building it
(`runner/aeiou/src/coord.rs`, `runner/REFERENCE.md` §6) changed four details and settled four
questions the design had not asked.

- **JSON frames, not `postcard`.** The coordinator exchanges a few dozen messages per run
  plus one report per host at the end; `serde_json` was already a dependency of the
  contract, and a readable wire costs nothing at this rate. The report is a few hundred KB
  per host at 500 steps (the per-take records dominate). A binary codec is a one-line change
  behind `send`/`recv` if a reduction ever grows.
- **A reader thread per socket, not one event loop.** The design said one coordinator
  thread; with blocking `std::net` the simplest correct shape is one reader thread per
  connection on the server and one on each client, with writes serialised through a mutex.
  Rank 0 runs the server in-process and connects to it as an ordinary client, so there is one
  code path for every rank and the server never special-cases itself.
- **The release wakes a condvar, not an eventfd.** The eventfd belongs to the `io_uring`
  backend, where a release must surface as a completion; the `sync` backends block on a
  condvar as the in-process barrier does. The two-level barrier (in-host arrivals, then
  `Arrive`/`Release` between hosts) keeps the host count out of the per-barrier cost: a
  1,000-GPU run on 20 hosts is 20 messages per generation.
- **Departures across hosts.** §3.23's departure rule (an instance that finishes leaves its
  barriers; a generation its departure completes is released and reported) needed a
  host-level counterpart: a host whose participants have all left sends `Leave`, and a
  generation that a host's leaving completes is a host-level departure release, merged into
  the same report line. The server tracks arrivals as a set of ranks, so a host that arrived
  and then left is never counted twice.
- **What the configuration hash covers.** `Hello`'s hash is over everything that shapes what
  the storage sees: the abstract's canonical hash, seed, G, the resolved parameters, the
  dataset ids from the manifests, the backend, the rotation, the time scale, and the write
  compression. It deliberately excludes `--root` (hosts may mount the same storage at
  different paths), `--buffer-mib`, `--max-gap`, `--require-cold`, and `--clean-namespaces`
  (rank 0's business). The dataset ids being in it means a host with a stale copy of a
  dataset is refused before the start gate, which replaces the protection `mpirun`'s single
  command line gave and goes further.
- **Only rank 0 touches the output namespace roots.** `--root` is the storage under test,
  shared by every host, so emptying it is one host's job, done before the start gate; the
  other hosts never empty anything and start their actors only after `Start`. The namespace
  manifests are likewise rank 0's, written from the merged report, so they record every
  rank's host and GPU range and the objects every host created net of what any host removed.
  A per-host root (a local cache) would need a flag; none is added until a workload needs it.
- **The verdict is one message.** Rank 0 merges, prints, writes the manifests, applies
  `--expect-fingerprint` to the merged fingerprint, and sends `Result` to every host; each
  host exits with it, so `pdsh` or `aeiou-launch` sees the same status on every host. Each
  host prints its own partial fingerprint too, labelled as such, since the sum is the
  fingerprint and a partial is not.
- **Faults.** Any actor failing, a refused host, a dropped socket, or 30 s of silence
  (heartbeats every 5 s while idle) sends `Stop` to every host, and actors now check the
  abort flag before every op as well as in every wait, so a failure elsewhere ends a host's
  I/O within one op. A host that fails its startup checks before connecting leaves the
  others to the 120 s connect window; connecting earlier would put the dataset ids, which
  the checks produce, outside the hash. Rank 0 waits up to 5 s after `Result` for the others
  to close, so its exit cannot reset a connection with the verdict still unread.

### 3.26 The loopback NFS run: what a real NFS client does with the abstracts (added 2026-09-30)

The loopback mount of `PROJECT_BRIEF.md` §7 came up (sequence and the full table in
`runner/REFERENCE.md` §7). It was a correctness exercise: the same kernel on both sides, tmpfs
behind the server, so only RPC overhead is measured. Every committed abstract reproduced its
dry-run fingerprint under `sync` cold, `sync` warm, and `sync-direct`, and the two-rank
tests pass with their roots on the mount. Six things the NFS client's counters showed, and
what each settles:

- **The warm restore is invisible to storage.** `ckpt_restore` run on the client that ran
  `ckpt_write_dcp` issued zero READ RPCs for its 18 MiB: the page cache served it all.
  §3.24's rule (restore on other hosts; `--rank-rotate`; `--require-cold`) was argued from
  how the page cache works; this is the measurement. It also fixes the single-host fallback:
  `sync-direct` is the only backend under which a same-host restore reaches the server
  (30 READ RPCs, 18.2 MiB), so a one-box checkpoint test that means anything runs direct.
- **O_DIRECT makes the abstract's transfer size the wire size; buffered I/O does not.**
  Buffered cold reads were merged by readahead (train_large_samples: 244 application reads,
  129 READ RPCs; the small-file case 192 reads, 97 RPCs), while `sync-direct` sent one RPC per
  application read where reads are aligned (model_load 76 of 76, IVF 32 of 32) and more
  where they are not (an unaligned tail read is rounded out to 4 KiB, and every `until_eof`
  probe is an RPC that returns EOF). That is `NAPKIN_MATH.md` §8.B's "no kernel readahead"
  consequence seen on the real client, and it confirms the decision there: under O_DIRECT the
  benchmark, not the client's heuristics, sets the wire pattern, so `xfer` and `hdr_read` are
  parameters that matter and must be measured from the real applications, not defaulted.
- **The abstract's flags and the backend are independent, as the design requires.** DiskANN's
  abstract opens its index `O_DIRECT` itself; all three backends produced the same 240 READ
  RPCs of 4 KiB. The backend adds `O_DIRECT` to opens that lack it and changes nothing an
  abstract already says (`PROJECT_BRIEF.md` §4, backends never change the op stream).
- **Writes behave symmetrically:** buffered writes coalesce to `wsize` (vdb_build: 262
  application writes, 8 WRITE RPCs; the checkpoint's 34 writes, 27 RPCs), O_DIRECT writes go
  out as issued (263 RPCs), and the checkpoint's `fsync` becomes COMMIT under both. The
  §8.B caveat that O_DIRECT NFS writes wait for stability is consistent with the higher
  direct write latency (790 µs against 650 µs mean), but the loopback cannot separate that
  from the extra RPCs; the real target can.
- **Close-to-open is a GETATTR per open, cached or not** (99 GETATTR for 98 opens in the warm
  small-file run, 0 READ). For a 50M-file epoch that is the metadata floor
  `NAPKIN_MATH.md` §3.2 counts, and no backend removes it; it is the reason the small-file
  regime is metadata-bound before it is data-bound.
- **Generating data behind the server interacts with the lookup cache.** A tree removed
  through the mount and recreated on the export was invisible for up to `acdirmax` (60 s):
  the client's negative dentries made `aeiou run` report no manifest and the namespace
  `mkdir` fail with `EEXIST`. Two practices are recorded in `runner/REFERENCE.md` §7: generate
  behind the server (so the client's cache starts cold, which is the only way a first
  buffered run shows what reaches the server) and under a name the client has never looked
  up. On a real deployment the same applies to a dataset regenerated under a name a client
  has already seen missing; a dataset generated on one host and first used from another is
  unaffected.

No design change came out of it. What it adds is evidence for three existing decisions
(O_DIRECT as the fix for the cache-fill regime, the cross-host restore rule, and the
transfer-size parameters being measured rather than defaulted) and two operating practices.

### 3.27 Parameter files: the shape/parameters split as built (added 2026-09-30)

Option D's "three sources" paragraph promised that shape and fitted parameters would be
separate artifacts, and §3.19 named the case that makes it necessary: `model_load`'s tensor
table, four parallel parameter arrays that read fine at four tensors and will not at four
hundred, and that no author should type by hand. Built as `schema/params.schema.json`, the
runner's `--params-file FILE`, and the `aeiou-params` helper (`schema/README.md` §8). Decisions
taken on the way:

- **The identity of a run does not change.** A result was already (AST hash, parameters in
  effect, dataset ids), and `aeiou run` already resolved the parameters before anything else.
  A parameter file is one more source of values in that resolution, applied over the defaults
  and under `--param`, so the manifest's resolved dataset definition, the coordinator's
  configuration hash, and the fingerprint all see exactly what they saw before. The file's
  name and SHA-256 are printed and recorded as provenance, like the abstract's `provenance`
  block, and never compared: two sets with the same values are the same run.
- **A value replaces a default of the same kind.** The builder decides at build time how a
  slot is consumed: `compute(P.step_time)` wraps a parameter whose default is a distribution in
  a `draw`, `P.off[t]` indexes an array. A file that turned a scalar into a distribution would
  not change a value, it would change the shape, and the runner would fail later with a less
  useful message. So a scalar replaces a scalar, a distribution a distribution, and an array an
  array of any length whose elements keep the default's element kind. `--param` is held to the
  same rule now; it was not before.
- **`cli: false` is about `--param`, not about files.** The flag protects a published shape
  from ad-hoc overrides at the prompt. A parameter file is the published set itself, so it may
  set any declared parameter; `gpus` stays the runner's.
- **An optional AST hash pins a set to a shape.** A set fitted against a trace is fitted for
  one exact shape; renaming a parameter or changing a loop invalidates it. `ast_sha256` is
  optional because the smoke sets and the defaults set legitimately follow the shape as it
  evolves; when present, a mismatch is a refusal, not a warning.
- **The abstract name is required.** Without it a set for `ckpt_restore` applied to
  `ckpt_write_dcp` would succeed on the shared parameter names and silently diverge on the
  rest.

**What building the safetensors tool changed in `model_load`.** Reading real shard headers
showed two things the §4b paper form had simplified away. First, a tensor-parallel plan has
three cases, not two: besides column-parallel (q/k/v, gate/up, embeddings, the LM head) and
row-parallel (o, down) weights, the norms and biases are replicated and every rank reads them
whole; `split` gained `full`. Second, shards differ in size, and the dataset models every
shard at the largest (`shard_bytes`), which over-reads nothing (the table's offsets are the
real ones) but means a `size(f)`-based `until_eof` would be wrong on this dataset; none is
used. The split decision itself (which tensor is column or row) is not in the safetensors
header: it belongs to the serving engine's parallel plan, so the tool takes it as name
patterns with Llama-shaped defaults, which is the right place for a configuration-source
input. The committed example set is generated from synthetic shards with real headers
(`builder/tests/test_params.py`), named `model_load.synthetic`, and says so in its `doc`; the
first real set will come from a real model's shards with the same command.

Not done here: `aeiou-fit` (`ABSTRACTS.md` §11), which will write the same form from a trace.

### 3.28 Format classes as built: the layout in the contract, traced protocols, a Python writer (added 2026-09-30)

§3.16 decided that a container format is a library with two halves and that the Rust runner
stays format-ignorant. Building it fixed where each piece lives and settled four things the
decision had left open. Contract 0.2 (`schema/README.md` §9) carries the result; the classes
are `builder/aeiou/formats.py`; the writer is `aeiou-datagen`; the shapes are
`train_stream_tfrecord`, `train_stream_parquet`, and `train_map_hdf5`.

**The layout is a generic framing formula in the AST, not a format name.** The runner has to
compute `offset(s)`, the extent of a row group, and a file's size without knowing Parquet,
so `format.layout` says, in format-free terms, how a file frames its samples: file, unit, and
column headers and footers, per-row framing, alignment, and column weights that split each
sample's bytes (`schema/README.md` §2 *Container layout*). Every real format tried fits it:
TFRecord is a 12-byte row header and a 4-byte row footer; a tar member is a 512-byte header
and a 512-byte row alignment, with the archive's two zero blocks and 10240-byte alignment as
file footer and file alignment; HDF5 is a file header at the data offset; Parquet is a 4-byte
magic, row groups of column chunks with a page header each, a footer that grows per row
group, and the 8-byte tail. The invariants hold unchanged: offsets are formulas, nothing
per-file is stored (the unit lengths of a file are recomputed from the size draws and held in
a bounded per-actor cache while the file is in use, `GRAMMAR_OPTIONS.md` §6.3), and the
runner never interprets a byte.

**Constants come from the installed library at build time and are pinned by its version.**
A page header is 16 bytes plus three thrift varints; an HDF5 file's data offset depends on
the library's metadata layout; an Arrow record batch's framing depends on the schema. Rather
than hand-maintain tables, a class derives its constants by writing a one-row file *in
memory* with the library (pyarrow's `BufferOutputStream`, HDF5's `core` driver without a
backing store) and records the library version in `format.version`, so the AST says which
pyarrow it is for and `--twice` proves the probe is deterministic. The writer then checks
every file it produces against the geometry the layout predicts (file size, and for Parquet
every column chunk's offset and length from the real metadata), so a library that lays files
out differently fails at datagen with a named mismatch, never at run time with a short read.
Two format quirks needed a decision: Parquet's footer varies with varint widths, so the class
declares a footer size with a margin and the writer pads it exactly through a key-value
entry (one counting pass through a sink to measure the natural footer, then the real write);
and a page header's varint band depends on the chunk's bytes, so the class places the header
for the band the expected chunk size falls in and the writer refuses a corpus whose chunks
cross a band. Parquet row counts must divide: `samples_per_file` by `rows_per_group` and the
dataset's count by `samples_per_file`, or a short last group changes a varint.

**`stream` is the shuffle over shards; the loader needs no records-per-batch knob.** The
reader library streams a shard; the input pipeline interleaves `cycle` of them; records are
batched from the interleaved stream and the shuffle buffer produces no I/O. So the loader's
unit of work under `stream` is a shard (`consume` returns file handles, epochs are counted in
files, the same position formula), the worker runs the format class's whole-shard protocol,
and the training loop takes a shard every `per_shard / batch` steps. The op multiset is
exact; what the model gives up is the timing of reads inside a shard relative to steps,
which the reader's own buffering hides from the application anyway.

**`fadvise` is the eighteenth op.** pyarrow issues `posix_fadvise(WILLNEED)` on every range
before it reads it (pre-buffering), and on an NFS client that advice starts readahead, so the
wire pattern of a Parquet read is readahead followed by a `pread` that hits the cache. An op
vocabulary without it would model the right bytes with the wrong RPC timing. The advice is
hashed like any other argument.

**What the traces showed** (loopback NFS, 2026-09-30; the scratch scripts were run through
the mount after writing behind the server, as `runner/REFERENCE.md` §7 prescribes):

| Library | Protocol as traced |
|---|---|
| pyarrow 25.0.1, `ParquetFile` | `open`, two `fstat`, `pread` 64 KiB at EOF − 64 KiB (the speculative footer read; a second read if the footer is larger); `iter_batches`: `fadvise(WILLNEED)` for every range, then one `pread` per range, where adjacent row groups coalesce into pieces of at most 32 MiB; a column projection reads one range per run of adjacent projected chunks per group; `read_table` opens, reads the footer, closes, and reopens |
| h5py 3.16 / HDF5 2.0, `f['records'][i]` | `open`, `fstat`, eight small metadata `pread`s (superblock, root object header, heap, B-tree, symbol node, dataset header), then one `pread` per row through the 64 KiB sieve buffer: `min(64 KiB, EOF − offset)` for a row smaller than the sieve, the row itself otherwise; a chunked dataset adds B-tree node reads per row |
| pyarrow 25.0.1, Arrow IPC file | memory-mapped: one `mmap` and no visible reads (the `mmap` backend's case); `OSFile`: `pread` 10 at EOF − 10, the footer, one `pread` per record batch |
| CPython 3.12 `tarfile` streaming | `open(O_RDONLY\|O_CLOEXEC)`, `fstat`, `ioctl(TCGETS)`, `lseek(0, CUR)`, `read(st_blksize)` to the short read and the zero read: the `until_eof` idiom of §1 |
| TensorFlow `TFRecordDataset` | not traced (TensorFlow is not installed): positioned reads of the buffer size from the source, tagged [verify] |

**The hermetic harness learned what a library does at import.** numpy seeds its global
`RandomState` from `/dev/urandom` and sets a BLAS guard in the environment, h5py asks
`uname` through `platform`, pyarrow loads its shared objects. None of it can reach an AST
(the module-level draws stay denied, the probes are pure), so the child imports the format
libraries before the hooks go in, allows those two environment names and library loads from
the package roots, and lets `SystemRandom` be constructed while denying its draws. The
build-twice check is still the proof; the hooks still make an actual leak immediate.

**The Python writer owns containers.** `aeiou-datagen` writes real files with the libraries
(a Parquet file pyarrow reads back, an HDF5 file h5py reads back, a tar `tarfile` lists), the
runner's payload bit for bit (the `dgen-py` 0.3.0 wheel was checked against the Rust crate's
output for the same block seeds, including the compression layout), the runner's size draws
(`rng.py` ports keys, SplitMix64, and the sampler; a Rust-written corpus is compared in the
tests), and the manifest `aeiou run` compares. `aeiou datagen` refuses a dataset with a
format class, so every dataset has one writer. One caveat is recorded rather than solved:
drawn sizes go through libm (`exp`, `log`, `cos`), so a corpus written on one host class and
run on another could in principle differ at a rounding boundary; constant sizes do not.

**Not built, and why.** Arrow IPC's record-batch read starts 16 bytes into the block and
ends 24 bytes short of it in the trace, which the probes did not pin to a formula in the time
spent; MDS and Megatron have no installable reader here to trace; the tenth abstract
(Parquet-to-Arrow conversion, then map-style training from the memory-mapped cache) needs a
namespace with a format class, which namespaces do not have yet. h5py's sieve hits between
neighbouring rows and tf.data's exact batch timing are stated cuts. Column projection is
implemented (`read_all(columns=[…])`) and traced but no committed abstract uses it.

### 3.29 The resumable VM and the `io_uring` backends as built (added 2026-10-01)

`runner/REFERENCE.md` §8 states what exists; this is why it is shaped that way.

- **Three ways to park a tree walk, and the one taken.** §3.23 left the VM recursive (the
  walk on the actor's thread) and promised a resumable VM for `io_uring`. The options were
  (a) stackful coroutines (a crate such as `corosensei`: the recursive walk untouched, the
  sink's `op` suspends the coroutine, the event loop resumes it on the completion; a stack
  per actor, lazily committed, and an unsafe stack-switching dependency), (b) `async fn`
  through the walk (a boxed future per recursion level, so an allocation per loop iteration
  on the dry run's hot path, and a hand-written executor), (c) an explicit state machine:
  the recursion becomes a stack of continuations (a body, a loop, a phase end, a read or
  write sequence) and `next()` yields one event. (c) was taken: no new dependency, no
  per-actor stack, the dry run and the `sync` sink unchanged in behaviour (`drive` feeds a
  `Sink` from the events, the fork protocol is the same), and a VM between two events is
  plain data a loop can hold by the thousand. The oracle was the one §3.23 named: the golden
  fingerprints, the run round trips, and the loader-order test, all unchanged.
- **What it cost.** The dry run of `vdb_build_diskann` (203M ops) went from 32 s to 36 s on
  one core (+12 %): a continuation push and pop per body and per loop iteration, and the
  pending op held in the VM instead of on the stack. Two things were taken back on the way:
  a single `read` or `write` (no `repeat`) is emitted without the sequence machinery, and
  the op's path moves into the pending slot rather than being reference-counted. The
  "cache a bound handle's path" fix of §3.22 is still the lever that matters, and still not
  done.
- **One op in flight per actor, by design.** An actor is a sequential program; the brief's
  fidelity argument (a PyTorch worker blocks in `read`) is why the `sync` backend is one
  thread per actor, and the same argument says an `io_uring` backend must not issue an
  actor's next op before the previous one completes. Concurrency is the number of actors on
  a loop, as it is the number of worker processes under PyTorch. Op linking
  (`IOSQE_IO_LINK`, §5) would be the way to let one actor keep several ops in flight and
  stays off.
- **Sub-actors stay on their instance's loop.** Channels then need no lock and a wake is a
  push onto the run queue; a `parallel` fan-out is `width` tasks on the same loop, not
  threads (the DiskANN search no longer spawns 923 threads for 924 reads, the complaint of
  `runner/REFERENCE.md` §4). The price is that one instance's sub-actors share one core; a
  PyTorch rank's loader workers do not, so if a workload ever needs the parallelism inside
  one instance, instances are the unit to spread and `--threads` already spreads them.
- **Explicit offsets, not the kernel's file position.** The first run of the checkpoint
  abstract under `io_uring-direct` failed its structural check: the fifth read of a 4.3 MiB
  file returned a full megabyte where 320 KiB remained. A probe showed Linux 6.18 does not
  advance the file position for an `IORING_OP_READ` at offset −1 on an `O_DIRECT` file (it
  does for a buffered one, and `pread` is right in both modes). The VM computes every read's
  effective offset anyway, since the fingerprint hashes it, so the backend now issues every
  read and write at that offset; the rounded-out direct read no longer needs its `lseek`
  fix-up either. Under the interposition test this is a shim rewriting `read` as `pread` at
  a position it knows: legal, and the op stream is the same. The `sync` backend keeps using
  the kernel's position, which is the fidelity reference and shows the two agree.
- **Inline where the ring has no opcode.** `lseek`, `ioctl`, and `getdents64` have no
  `io_uring` opcode; `ftruncate` needs 6.9. The loop runs those through the blocking backend
  on its own thread, after probing the ring at startup for the rest. An `io_uring`
  application has the same gap and makes the same call; the alternative, a helper thread
  pool, would be a second backend inside the first.
- **The eventfd, as designed.** `NAPKIN_MATH.md` §8.A planned a read posted on an eventfd
  the coordinator writes on release; §3.25 built the condvar half and deferred the eventfd to
  this step. The `Coordinator` trait grew the non-blocking half (`arrive`, `released`,
  `subscribe`), and both the local and the TCP coordinator kick every subscribed eventfd on
  a release and on an abort. A 50 ms idle timeout on the enter call is the belt to those
  braces: another thread's abort flag is seen without a kick.
- **The buffer pool replaces the ring.** The `sync` sink's per-thread buffer ring assumed one
  op at a time per thread. With tens of ops in flight per loop a chunk must stay allocated
  until its CQE, so the pool hands out aligned chunks and takes them back FIFO; the
  aggregate-exceeds-L3 property of `NAPKIN_MATH.md` §2.2 is kept by not reusing a chunk
  until the pool holds `--buffer-mib`.
- **What the first numbers say, and do not.** Four loops did the 64-GPU small-file run from
  the page cache in the time of 320 threads with a third of the client CPU; 20 loops were
  faster still but used as much CPU as the threads (`runner/REFERENCE.md` §8). On the loopback
  NFS mount the RPC counts are identical between `sync` and `io_uring`, which is the
  correctness statement (the NFS client sees the same calls), and the io-wq worker count
  peaked at ~~the core count~~ 20 for buffered and direct reads alike, because `openat` punts
  whatever the reads do (20 was the actor count, not the core count or a kernel cap:
  corrected in §3.34). Spike 1's hypothesis (R1) is neither confirmed nor refuted here: it
  needs the real target and phases that separate opens from reads. ~~The knobs it will need
  (`IORING_REGISTER_IOWQ_MAX_WORKERS`, `ATTACH_WQ`, fixed files) are listed and not built.~~
  The ring and io-wq knobs built the same day (§3.34); fixed files and buffers not.
- **Open.** ~~The per-backend counters (io-wq workers sampled from `/proc`, `mountstats`
  deltas) are shell scripts around the runner, not report fields;~~ (report fields the same
  day, §3.30) ~~`libaio`, `posix-aio`, and `mmap` now have the VM they need and are the next
  backends in the brief's order.~~ Built the same day (§3.35).

### 3.30 Host counters in the report (added 2026-10-01)

**Issue.** The brief (§4) wants every backend to report its own counters next to the
common `mountstats` RPC counts, and R2 wants client CPU and RPC statistics with every run;
through §3.29 those were shell scripts around the runner (`grep` of `/proc/self/task`,
`mountstats` before and after), which is how the io-wq peak of 20 and the RPC tables of
§3.26 and §3.29 were produced. Numbers that live outside the report are not reproduced by
the next person and cannot be merged across hosts.

**Decision.** A `counters` module with one `HostCounters` per host in `Report`, summed by
`merge_all` like the rest (each host is its own process, so peaks add), printed after the
totals (`runner/REFERENCE.md` §4). Three sources, each optional where the kernel lacks it: a
10 ms sampler of the process's task count that scans thread names for `iou-wrk-*` whenever
the count changes (io-wq workers linger idle for seconds, so the peak survives the sampling
rate; a 100 Hz scan of `/proc/self/task` only on change costs nothing under `sync`, where the
count never changes); `getrusage` deltas for user and system CPU and the peak RSS;
`/proc/self/mountstats` deltas for the mount `--root` is on, parsed once before and once
after. The NFS block gives the bytes line and every procedure's count, transmissions,
timeouts, bytes, queue, RTT, and execute totals; the report prints counts with mean RTT and
the retransmission and error figures only when they are non-zero. Nothing here touches the
op stream or the fingerprint; the counters are a measurement of the solution and are never
compared by `--expect-fingerprint` or the namespace manifest.

**Two things the counters are not.** `mountstats` is the mount's view, so every process on
the host using the mount is in the delta (the line says so); a benchmark host runs nothing
else on the mount, a developer's laptop might. And the NFS client counts `O_DIRECT` bytes as
requested, buffered bytes as returned, so the direct figure for a small-file read with 1 MiB
aligned buffers is the buffer size times the reads; the report labels it `requested`, and
the server-side bytes are the comparable number.

**What the first run of the fields showed.** The RPC counts and the io-wq peak equal the
hand-measured ones (§3.29); the only surprise was a warm `GETATTR` per open under `sync`
and not under `io_uring`, which cross-running the backends on each other's directories
showed to be the attribute-cache timeout and the gap between runs, not the backend
(`runner/REFERENCE.md` §8). That is the kind of finding the fields exist for: without them the
difference would have been read as a backend property.

**Backend-specific counters beyond these** (libaio context depth, `cufile_stats`, NIXL plugin,
`mmap` fault counts, NAPKIN_MATH §8.5) come with each backend as an optional block in the
same structure; the open-file high-water mark is one `RLIMIT` check away and goes in with
the startup checks.

### 3.31 Cold start: dropping caches at the start gate (decided 2026-10-01)

**Question (user).** Can the runner make Linux drop every cache that could carry storage
state from before a run into the run: page cache, and the metadata caches too (attribute
cache, dentries)? Clarified the same day: only before a run, or between the write and the
read of a checkpoint pair; never between batches or epochs of a training run. The goal is
only that caching from before the start of a run does not affect that run.

**What the kernel offers.** `/proc/sys/vm/drop_caches`, root only: `1` drops clean page
cache, `2` drops unreferenced dentries and inodes, `3` both. The NFS metadata lives on
those objects: the attribute cache, the access cache, and the readdir pages hang off the
inode; the lookup cache, negative entries included, is the dentry; evicting an NFS inode
returns its delegation. So one write of `3` after `sync` covers data and metadata, with
two holes: inodes still referenced keep their state (open files, the mount root, the
current directory), and `fscache` and the NFSv4 client state are untouched. The shrinker
is single-threaded and global; after an enumerate of 50M files it takes tens of seconds
and stalls the host.

**Decision.** One option, `--drop-caches`, on every host after the dataset and
input-namespace checks and after rank 0 has emptied the namespace roots, immediately
before the host arrives at the start gate. The gate then guarantees every host has dropped
before any op is issued, the drop time is outside `elapsed` by construction, and at the
gate the runner holds no files open, so the referenced-inode hole is the mount root alone.
No barrier-scoped form: the checkpoint case is already two runs (`ckpt_write_dcp` leaves
the namespace and its manifest; `ckpt_restore` reads it on other hosts under
`--rank-rotate`), so a drop at the start of the restore run is the "between write and
read" the question asked for; if one abstract ever holds both phases, the coordinator can
grow a drop exchange at a named barrier then. Root is required: the operator runs the
runner under `sudo -n` or grants the capability, and `--drop-caches` refuses to start when
the write fails on any host. The option is a harness parameter, never abstract vocabulary:
cache state is solution, not application (§3.12).

**Verify, not only act.** The goal is a property, so the report measures it: the drop and
its duration, before/after `Cached` from `/proc/meminfo` and `/proc/sys/fs/dentry-state`
and `inode-nr` (world-readable), and a residency check that samples a few hundred dataset
files by formula (no per-file structure, no root) and asks `mincore` what fraction of their
pages is resident at the start. ~~That line is useful without the flag,~~ (revised below:
sampled only with `--drop-caches` or `--require-cold`) as the dataset
counterpart of the warm-open count for input namespaces (§3.24), and `--require-cold`
covers both. The host counters gain the mount's `opts:` line, since `actimeo`,
`lookupcache`, and `nconnect` decide cache behaviour more than any drop does, and §3.30's
GETATTR finding showed the counters often make a drop unnecessary. The full reset
(`fscache`, delegations, session) is unmount and mount, which belongs in `aeiou-launch` as
`--remount` before `aeiou run`; the runner never unmounts the storage under test.

**Built 2026-10-01** (`runner/aeiou/src/cold.rs`, `runner/REFERENCE.md` §4), as decided, with
these points settled in the building:

- The drop file is opened before the `sync`, so the refusal costs nothing, and the error
  travels the coordinator's existing stop path: no host passes the gate.
- `drop_caches` joined the configuration hash the coordinator compares. Unlike `--threads`
  or the ring knobs it changes what the run measures, and a run cold on some hosts only is
  not a run anyone wants by accident.
- The residency sample is 256 files per dataset at ids `⌊k·files/256⌋`; a file over
  256 MiB is sampled in 64 windows of 4 MiB, so a 100 GiB `regions` file costs 64 small
  `mincore` calls, not one over 26M pages.
- `--require-cold` refuses on any resident sampled page. A tolerance would need a number,
  and nothing yet says what number.
- The sample's own opens warm the metadata of the files it samples. Accepted and stated:
  256 of 50M is nothing, and sampling without opening is not possible.
- Seen on the loopback mount: a first run reported 0 of 7,959 sampled pages resident, and
  a `--require-cold` run straight after it was refused with 280 resident, the files the
  first run had read. The mount options line showed `acregmin=3,acregmax=60`, the cause of
  §3.30's GETATTR finding, without anyone looking for it.
- ~~**Not verified here:** the drop itself as root. The development box has no passwordless
  `sudo`; the write, the refusal without privilege, the parsing of the three `/proc` files,
  and the report lines are tested, the effect of a real drop on a real NFS client is not.~~
  **Verified as root on the loopback mount the same day** (run by the user under `sudo`,
  `--drop-caches --require-cold`, `train_small_files`, 8 GPUs, 4 steps): sync 4 ms, drop
  81 ms; 0 of 7,959 sampled pages resident, so `--require-cold` passed; the run then read
  every byte from the server (119.78 MiB server read for 119.78 MiB read, 1,024 `READ`)
  and sent 1,014 `OPEN` for 1,024 opens where a warm run sends `OPEN_NOATTR` and
  `DELEGRETURN`. Two things the numbers show. The ten opens without an `OPEN` are the
  sample's own footprint: 256 sampled of 16,000 files, 1,024 files read, about 16 expected
  in both. And `Cached` fell only from 2.24 to 2.10 GiB, because `Cached` in
  `/proc/meminfo` includes shmem and the loopback export is a 1.9 GiB tmpfs; on a host
  with a large tmpfs the before/after pair understates the drop, and `Cached − Shmem`
  would be the better figure ~~(not changed yet)~~ (the report prints that difference as
  `file cache` since later the same day, with `Shmem` beside it). Still untried: a real NFS client against a
  real server, and the drop's duration after an enumerate of tens of millions of files.
- `aeiou-launch --remount` is not built.

**Revised 2026-10-01 (user): the residency sample is opt-in.** It runs only with
`--drop-caches` (where it is the proof the drop worked; `Cached` proved a poor witness) or
`--require-cold` (where it is the gate that refuses before an hour is spent). A plain run
no longer samples. Reasons: the sample perturbs what it measures (the root run above sent
ten fewer `OPEN`s because of it); it is 256 files of data pages, blind to dentries and
attributes; on NFS the report's server-read bytes against bytes read already expose a warm
run exactly, after the fact; and under `O_DIRECT` it is beside the point. What is given
up: the unprompted warning on filesystems with no client counters. An operator who wants
it there passes `--require-cold`.

### 3.32 Object backends through `s3dlio`: a preliminary opinion (added 2026-10-01)

**Question (user).** The repository that hosts `dgen-py` also hosts `s3dlio`, a Rust crate
reading samples over POSIX, S3, Azure Blob, and GCS behind one API chosen at run time. Can
its object support be used without hurting the short path from queued op to I/O that the
POSIX backends have? Preliminary only; a task for later.

**Facts checked (2026-10-01).** `s3dlio` 0.9.x, Apache 2.0, a Rust `ObjectStore` trait with
async `get`, `get_range(uri, offset, length)`, `put`, `put_multipart`, `stat`, `list`,
`delete`, `rename`, `mkdir`, `exists`, and `get_writer` for streaming uploads; backends by
URI scheme (`s3://`, `az://`, `gs://`, `file://`, `direct://`); results as `bytes::Bytes`;
tokio throughout, with the AWS, Azure, and Google SDKs behind it; its blocking wrappers
exist for the Python layer.

**Opinion.** Feasible behind a cargo feature (`object`), at no cost to the POSIX path, and
it touches one written invariant. The invariant "no tokio in the runner" would become "no
tokio in the default build": the feature brings `s3dlio` and a tokio runtime in, confined to
that backend behind the `Backend` trait's completion-source half, which exists for exactly
this; `sync` and `io_uring` never see a line of it, and the default binary is unchanged.
The brief's reason for deferring `s3` ("needs `get`/`put` in the abstract") is reversed: the
abstract stays POSIX-shaped and the backend maps, as `gds` or `nixl-posix` will. Reads at
an offset become `get_range`; a sequential `open, read…, close` can become one streaming
GET per open, which is what a real object loader does and what a shim could do (§3.12), so
both a ranged and a streaming mode are legal rows; sequential writes to a new object become
a multipart upload; `fstat` is `stat`, `readdir` is `list`, `rename` a server-side copy and
delete; `mkdir`, `fsync`, `lseek`, `ioctl` are local. What has no mapping (a write at a
non-sequential offset, an overwrite in place, `until_eof` on a growing object) is refused at
`aeiou check` by a validity rule, as V9 refuses access modes a format class lacks. The
fingerprint is untouched. Expected costs: compile time and size from three cloud SDKs,
version churn on a 0.9 crate, the library's own thread pools to size and pin, one copy
from `Bytes` into the sink buffer (deliberate, as the sink copy always is), and `aeiou
datagen` and the manifests through the same backend. First measurement, before any cloud:
the same `train_small_files` fingerprint over `file://` through the library against the
`sync` backend, which prices the library's overhead alone. Recorded as brief §6 item 17.

*2026-10-09:* the mapping and the cargo feature stand; the library does not. The object engine
is built on Apache `object_store` (3–4× less client CPU per small request than `s3dlio`'s AWS
SDK path, measured), and the `file://` measurement is dropped, since the runner's POSIX path
never goes through an object library (§3.65).

### 3.33 Speed of light of the benchmark code (added 2026-10-01)

**Question (user).** Given what the counters now show, what is the most a 96-core node with
two 400 GbE ports could drive, and ten of them, against a storage system that keeps up?

**Answer.** `NAPKIN_MATH.md` §3.4: the runner costs about 2 µs of CPU per op and nothing
measurable per byte (`io_uring` with four loops; twenty loops cost 3.5× more on the
small-file mix through io-wq contention), so the ceiling is the Linux NFS client's: wire
near 92 GB/s of NFS payload for large reads and writes, 0.5–0.7M small files per second
where the wire and the RPC rate meet, 1–3M RPCs per second for metadata and 4 KiB reads;
ten nodes linear. The caveats are the ones the loopback cannot remove: per-RPC CPU and
connection scaling on a real client, NUMA placement of loops and buffers at 280 GB/s of
DRAM traffic, and io-wq as the `io_uring` lever rather than loop count.

### 3.34 The `io_uring` knobs as built (added 2026-10-01)

**What was open.** §3.29 left the A/B knobs of `NAPKIN_MATH.md` §8.5 unbuilt, and §3.33
named the io-wq cap as the first one a big node needs. Built: `--iowq-max-workers`,
`--sqpoll` with `--sqpoll-shared`, `--defer-taskrun`, `--coop-taskrun`
(`runner/REFERENCE.md` §8, `RunOpts::uring`, `tests/uring_knobs.rs`).

**Choices, and why.**

- **Options, not backends** (brief §4): the op stream and the fingerprint are untouched,
  and every knob combination reproduces the dry-run fingerprint in the tests.
- **Refused under `sync`, not ignored.** `--threads` is ignored there, which is harmless;
  a ring knob silently ignored would let a result be labeled with a setup it did not have.
- **Outside the coordinator's configuration hash,** like `--threads`: host tuning, printed
  in the report. Whether a published comparison must hold them equal across hosts is the
  comparison policy's question (brief §5, R17), not the runner's.
- **The cap is per loop because io-wq is per task.** Since 5.12 the worker pool belongs to
  the task that owns the ring, so each loop thread has its own and the host total is
  `N × --threads`. `IORING_SETUP_ATTACH_WQ` no longer shares workers between tasks; it
  shares the `SQPOLL` thread, whose io-wq then serves every attached ring. That is why
  `--sqpoll-shared` requires `--sqpoll` and why a bare "attach" knob was not built: without
  `SQPOLL` it would do nothing.
- **The kernel's caps are reported with or without the flag.** The register call returns
  the previous values, and a zero leaves a cap alone, so one call per ring both reads and
  sets. On this box: 80 bounded (`min(SQ entries, 4 × cores)`), 127,569 unbounded
  (`RLIMIT_NPROC`).
- **Not built:** fixed files, fixed buffers, op linking (reasons in the README; linking
  stays off per §5).

**What the knobs corrected.** §3.29 and the README read the loopback io-wq peak of 20 as
"the core count (io-wq's bounded-worker cap)". The cap is 80 per loop. 20 was the number of
actors (4 GPUs × 5), each with one op in flight, on a 20-core box. With 72 actors the peak
was 30 to 46.

**What the first runs showed (loopback, `runner/REFERENCE.md` §8 table).** Capping the workers
at one per loop (four in all) ran the 72-actor small-file mix in the same time as the
default's 30 to 46, with the same RPCs and a little less CPU. `SQPOLL` per loop cost a third
more CPU for nothing; one shared poll thread halved the rate. `DEFER_TASKRUN` and
`COOP_TASKRUN` were neutral. So the lever §3.33 predicted is real in direction (fewer
workers lose nothing) and unmeasured in size: the loopback server competes for the same
cores, and Spike 1 on the real target should run the cap at 1, 2, and default with open-heavy
and read-heavy phases apart.

**A test artifact worth knowing.** A closed ring's `SQPOLL` thread and io-wq workers exit
asynchronously, some milliseconds after the run returns. Runs in one process (the tests)
must wait for them before counting the next run's; the CLI is one run per process and is
unaffected. The knob test lives in its own file, hence its own process, for the same reason.

### 3.35 `posix-aio`, `libaio`, and `mmap` as built (added 2026-10-01)

**What was open.** The brief's §4 lists nine backends; four existed. §3.29 left the three
that need nothing but the VM as the next step. Built: `posix-aio`, `posix-aio-direct`,
`libaio`, `libaio-direct`, `mmap` (`runner/REFERENCE.md` §9, `backend.rs`, `aio.rs`,
`tests/run.rs`). None changes the op stream; all reproduce the dry-run fingerprint.

**Choices made while building, and why.** None of these was discussed first; each is
open to revision.

- **Where each one runs.** `NAPKIN_MATH.md` §4.2 sorted the APIs: blocking calls on the
  thread per actor, submit-and-poll on the event loop. `posix-aio` and `mmap` are blocking
  backends beside `sync` (`aio_suspend` and a page fault both block the caller). `libaio`
  is a second engine under the loop of `uring.rs`, which became generic over an `Engine`
  trait (issue, complete, wait, wake) rather than being copied: tasks, channels, timers,
  barriers, and the deadlock report are one implementation, and the `io_uring` tests still
  pass unchanged. §4.2 planned a poller thread per loop feeding an eventfd; none is needed,
  since `io_getevents` takes a timeout and `IOCB_CMD_POLL` puts the barrier eventfd in the
  same context (Linux 4.18).
- **`libaio` is the kernel ABI, called directly.** The libaio library is a wrapper over
  five system calls; linking it would add a build dependency for nothing. The name stays
  `libaio` because that is what fio and vendors call the path. Counter-argument: a user
  comparing with fio's `libaio` engine may assume the library's user-space ring fast path
  for `io_getevents`; this backend always enters the kernel. On NFS the difference is
  nanoseconds against RPCs.
- **What has no AIO form runs inline and blocks the loop.** That is the interface, not a
  shortcut: an AIO application calls `open` and `close` itself. It makes `libaio` a poor
  fit for small files and the measurement says so honestly. The alternative (a helper
  pool for opens) would be modelling a specific application design, which belongs to a
  backend of its own if anyone needs it.
- **`libaio` (buffered) is offered although it is synchronous**, because people run it and
  the report shows what it is: time inside `io_submit`. Counter-argument: a row labelled
  `libaio` invites the reading "asynchronous"; the `libaio:` line and README §9 are the
  guard.
- **`mmap` maps the whole file on first read and unmaps at `close`**, with its own
  `fstat` for the size, because that is what safetensors, Arrow, and `np.load(mmap_mode)`
  do. The alternative, a map and unmap around every read, would measure `mmap(2)` itself.
  The mapping is per actor thread and per descriptor; a sub-actor reading an inherited
  descriptor makes its own mapping.
- ~~**A read is a copy out of the mapping into the actor's buffer**, under all three modes,
  so that every backend delivers the bytes to pageable host memory and CPU per op is
  comparable. Counter-argument: a zero-copy consumer (a tensor that stays a view of the
  mapping) touches pages later and elsewhere, or never; that is a different workload, not
  a different backend, and the abstract would have to say when the touch happens.~~
  Superseded the same day by **Revised** below: a read touches one byte per page, and the
  copy is an option.
- **The three prefetch variants are a mode (`--mmap-mode`), not three backends**, per the
  brief's rule that feature knobs are options. The mode is in the coordinator's
  configuration hash (it changes what the hosts ask of the kernel, like `--drop-caches`);
  `--aio-depth` is not (a host's tuning, like `--threads` and the ring knobs).
- **Writes under `mmap` are `pwrite`.** No loader writes through a mapping, and a
  `MAP_SHARED` write path would need `msync` semantics the abstract does not have.
- **One address space.** Actors are threads, so their mappings share one `mmap_lock`;
  PyTorch workers are processes. ~~The `mmap` rows therefore overstate client CPU for a
  process-per-worker loader. The fix is the per-actor sub-actor pool as processes, which
  nothing else needs; recorded as a caveat instead.~~ See **Revised** below: the costs are
  counted against `mmap`; what the one address space adds is how far they spread.
- **`SIGBUS`.** `fault` and `willneed` take the signal on an I/O error, as the applications
  do. The runner does not install a handler: a structural check that turned a `SIGBUS` into
  an op error would need `sigsetjmp` around every copy. `populate` returns an errno and is
  the mode to use where errors are expected.
- **Both glibc and kernel AIO issue at the effective offset**, as `io_uring` does (§3.29):
  the APIs have no file position. `Backend::positional` says so to the blocking driver.
- **Page faults joined the host counters** for every backend (`getrusage`), rather than
  living in the `mmap` block: they are a process-wide figure and say something under
  `sync` too (the buffer rings).

**What the loopback showed** (`runner/REFERENCE.md` §9 has the tables): `posix-aio` sends the
RPCs of `sync` for 2.5 to 3 times the CPU; buffered `libaio` on four loops was the cheapest
row in CPU and tasks because nothing is handed off, and would be the first to suffer from
real latency; `libaio-direct` spent a seventh of its loop time inside `io_submit` and
blocks on every open and close besides; `mmap` under
`fault` sent 39 % more READs cold than `read(2)` and one GETATTR per mapping, `willneed`
restored one READ per file, and `populate` changed nothing on the wire. R14's sentence,
that fault-around and readahead and not the abstract decide the RPC sizes, is now a
measurement.

**Revised 2026-10-01 (user): a read touches, it does not copy; and the coupling counts.**

The user asked how arrival is ensured under `mmap`, and whether copying out of the mapping
is fair to it. Three points came out of the exchange.

- *Arrival.* A read of a byte in a page that is not resident faults, and the fault
  returns when the page has been read. So reading the range, by copy or by one byte per
  page, cannot finish before the data has arrived; `MADV_POPULATE_READ` gives the same
  guarantee before it returns, and `MADV_WILLNEED` gives none (it starts readahead), so it
  needs the touch. The byte count the backend returns is computed from the file size,
  not handed back by the kernel as with `read(2)`; a short file is still caught, since
  the size is the `fstat`'s.
- *What each API is charged.* The first answer here defended the copy as "the same one
  copy a buffered read makes". That was the wrong frame. The rule the brief's
  application/solution boundary implies is: charge a backend what its API needs to make
  the bytes addressable by the application, and nothing the application does afterwards.
  For `read(2)` that includes the kernel's copy out of the page cache, which the call
  cannot avoid. For `O_DIRECT` and for `mmap` there is no such copy: the data is
  addressable where it landed. The copy to the GPU follows under every backend and is
  charged to none (it is `compute`). Copying out of the mapping charged `mmap` alone for
  the application's copy, and took away the one advantage the technique has (user). The
  brief's row said "`mmap` + page touch" from the start; the copy was a deviation made
  while building.
- *Memory traffic.* A copy reads every byte and writes it again; a touch of one byte per
  4 KiB page pulls one cache line in 64. Nobody has characterized what the copy's memory
  bandwidth costs the other actors, so a default that spends it would put an unmeasured
  cost in every `mmap` row (user).

**Decided:** `--mmap-consume touch` is the default: one byte of every page of the range,
read volatile so it is not elided, and nothing at all under `populate`, which has already
made the range resident. `--mmap-consume copy` remains for a loader that does copy out of
the mapping into host memory, and is in the configuration hash like the mode.
Counter-arguments kept on record: (1) a touch leaves the data cache-cold, where the
kernel's copy under `read(2)` leaves part of it in the cache for the consumer; that is a
benefit to the application's next step, outside what is measured, in either direction;
(2) fault-around maps up to 16 cached pages per fault, so most touches of a warm file take
no fault, which is the real behaviour and not an artifact; (3) with `touch` the actor's
buffer is never written, so RSS and the "buffers larger than L3" rule of
`NAPKIN_MATH.md` §2.2 do not apply to this backend: there is no destination to keep cold.

**Decided (user): the cross-thread costs of `mmap` are counted against it.** The map, the
unmap, the translation flush an unmap forces on other cores, and the faults are
consequences of the technique, disadvantages against `O_DIRECT` that belong in its row.
The earlier wording here called the `mmap` rows an overstatement and filed that as a
caveat; it is withdrawn. What remains true, as a statement about magnitude and not about
whether to count: the runner's actors are threads of one process, so an unmap's flush
goes to every core running any actor and all of them share one `mmap_lock`, where a
loader made of worker processes confines both to one worker. On a real loader of that
shape the same costs exist and spread less. Blocking is not part of the coupling: a
fault stalls only the faulting thread, the fault path gives up the lock before it sleeps
on I/O, and `mmap` runs on one thread per actor, not on an event loop (where a fault
would stall every actor on the loop, which is why it was never put there).

**What the loopback showed after the change** (`runner/REFERENCE.md` §9): warm, `mmap` with
`touch` costs 20 to 35 % more CPU than `sync` (16.2 s against 13.4 s, and 15.1 s against
11.2 s in the delegation regime) with no copy at all against `sync`'s copy of every byte,
so the map, unmap, flush, and fault overhead is larger than the copy `read(2)` makes on
this 116 KiB-per-file mix; the copy added 3.6 s for 14.8 GiB on top. The RPCs are the
same under `touch` and `copy`. On 140 MiB files (`train_large_samples`, 7.8 GiB in 1 MiB
reads) the balance reverses, since the per-mapping cost is fixed and the copy is per byte:
warm, `mmap` with `touch` used a tenth of the CPU of `sync` (0.32 to 0.40 s against 3.1
to 3.4 s) in under a third of the time, and with `copy` it matched `sync`. So the
advantage the user argued `mmap` should be credited with is large where `mmap` is actually
used, and the copy default would have hidden all of it. Cold, the same run showed the
other side: the fault path sent 36,981 READs of about 109 KiB where `read(2)` sent 6,087
of about 660 KiB, and took 1.9 s against 1.2 s; consistent with the mount's
`read_ahead_kb` (128) bounding fault readahead, not confirmed by changing it.

**Open.** The real target. ~~`read_ahead_kb` of the mount in the host counters, and~~ ~~A run
with `read_ahead_kb` raised, to confirm what sets the size of a fault's READ~~ Confirmed
the same day (user, as root; `runner/REFERENCE.md` §9): at 1024 the cold fault path sent
5,376 to 6,587 READs where it had sent 36,981, in 0.85 to 1.04 s where it had taken 1.93 s,
and `sync` did not change (6,120 READs, 1.38 s), so cold `mmap` went from the slowest row
to the fastest on a setting of the NFS client that defaults to 128 KiB whatever `rsize`
is. A comparison of `mmap` with `read(2)` on NFS is therefore a comparison at a stated
`read_ahead_kb`; whether a run rule should fix or merely report it is a WG question (the value
is in the host counters since later the same day, user: a `mount read_ahead_kb` line from
`/sys/class/bdi` for the device `--root` is on; like the mount options it is the
solution's setting, reported and never changed by the runner, and it is not in the
configuration the coordinator compares, since hosts may legitimately differ and the
merged report lists the distinct values). `aio_init` (glibc's pool size) as a knob if anyone runs
`posix-aio` in earnest. `RWF_NOWAIT`/`preadv2` flags on the AIO requests. A filled
context under test. The `--cache` and `--buffer` axes of the brief's validity table are
still only the `-direct` suffix.

### 3.36 The sub-actor pool (added 2026-10-01)

`runner/REFERENCE.md` §4 states what exists; this is why it is shaped that way. It was the
planned fix of §3.23's follow-ups for the blocking backends (`sync`, `posix-aio`, `mmap`
and their `-direct` forms); the event loops never had the problem (§3.29: a `parallel`
there is `width` tasks on the instance's loop).

- **The problem was the runner's, not the workload's.** A `parallel` inside a loop spawned
  and joined `width` OS threads on every iteration, so the DiskANN search (`beam`
  concurrent reads per hop) created one thread per read: 507,617 threads for 507,608 reads
  in the run measured, with 135 s of system time against 41 s elapsed. No search library
  does that; a thread per beam slot that lives as long as the search thread is the nearest
  thing to what one does (a pool, or asynchronous I/O). The latency clock covers only the
  read, yet the mean was 611 µs under that churn and is 197 µs with the pool (41 threads,
  36 s of system time, the fingerprint unchanged): two thirds of the measured read time
  was the client's own thread creation getting in the way of the reads beside it.
- **The pool belongs to the forking actor, and the assignment is positional.** Sub-actor
  `k` of a fork runs on pool thread `k` over that thread's own queue; there is no shared
  queue for idle threads to race on. (Since §3.38 the forking thread runs sub-actor 0
  and pool thread `k − 1` runs sub-actor `k`; still positional.) Which OS thread runs a sub-actor could not change the
  op multiset in any case (a sub-actor's randomness is keyed on its indices), but the rule
  of `CLAUDE.md` is that no work is distributed based on timing, and a fixed assignment
  keeps that true without an argument. The pool grows to the widest fork the actor issues
  and never shrinks.
- **What a pool thread keeps, and what it does not.** It keeps the two buffer rings and a
  nested pool. ~~The backend object and the file table are made per sub-actor, as before: a
  sub-actor sees the files its parent had open *at that fork*, and the `mmap` backend's
  table of mappings is keyed by descriptor number, which the parent may have closed and
  the kernel reused by the next fork. So a sub-actor's own files and mappings still end
  with it, before the parent's join returns.~~ The file table is made per sub-actor (a
  sub-actor sees the files its parent had open *at that fork*, and its own files end with
  it, before the parent's join returns). The backend object was made per sub-actor too,
  because the `mmap` backend kept its mappings by descriptor number; see **Revised**: the
  backend holds no mappings now and the thread keeps it.
- **Loader workers are unchanged.** They already lived until their actor ended. A worker
  whose body forks gets a pool of its own, as any actor does.
- **`threads` in the report changes meaning** from sub-actors run to threads created
  (the `tasks peak` of the host counters was always the latter).
- **A side effect on memory.** The resident set of the `sync` run went from 7.8 MiB to
  264 MiB. That is the ring doing what `NAPKIN_MATH.md` §2.2 asks (successive reads land in
  successive slices, so the copy is not cache-hot): a thread that lived for one read only
  ever used the first slice. The earlier figures for `parallel`-heavy abstracts under the
  blocking backends therefore had cache-hot copies as well as thread creation in them.
- **Not done: sub-actors as processes.** §3.35 considered it for `mmap` (one
  `mmap_lock`) and dropped it; nothing here needs it.

**Open.** ~~Under `mmap` a sub-actor inherits its parent's descriptors but not its mappings,
so each DiskANN beam sub-actor maps the whole index for one 4 KiB read and unmaps it (one
map and one fault per read: 508,669 minor faults, 48 s of system time, 12.0 s elapsed
against 6.3 s for `sync`). A search library that reads through a mapping maps the index
once. Whether a sub-actor should read through the mapping its parent made is a question
about what the backend is charged with (§3.35, Revised), not about the pool, and is left
for the user.~~ Decided the same day, **Revised** below. The pool's idle threads hold their rings (`--buffer-mib` each, twice once
the actor writes), which at `threads × beam` per instance is the figure to watch on a
host with thousands of instances.

**Revised (2026-10-01, later): a mapping belongs to the open file.** Raised by the user:
if a sub-actor stands for a thread of the application, sharing the parent's mapping is
the efficient architecture, with no repeated `mmap`/`munmap`. Agreed, on two grounds
already recorded: a `parallel` sub-actor is a thread of its parent's address space, where
a mapping is visible to every thread; and a map and an unmap per sub-actor is work the
API does not need, which §3.35 (Revised) says is not to be charged to the backend.

- **Not "inherit the parent's mapping".** In the DiskANN abstract the search thread opens
  the index and never reads it; only its beam sub-actors do, so at the fork there is
  nothing to inherit. The rule is that the mapping belongs to the open file: the first
  reader makes it, whoever that is, everyone holding the descriptor reads through it, and
  it is unmapped when the last holder lets the descriptor go (`backend::OpenFile`, which
  the actors' file tables now hold in place of the bare descriptor). `mappings` in the
  report is therefore one per open that was read. Which thread makes it is a race; the
  count, the op stream, and the fingerprint do not depend on who wins.
- **It needs no special case for loaders.** A PyTorch loader worker is a process, but the
  files it opens are its own in the model too, so its mappings are its own. Only a file
  opened before the fork is shared, and a mapping made before a real `fork` is inherited
  by the child as well. (One made after it is not; no abstract depends on that.)
  Instances never share: each opens its own descriptors.
- **No lock on the read path.** The current mapping is published through an atomic
  pointer. A lock per read, or one table for the process, would have been the runner
  adding a cost to `mmap` that the application does not pay.
- **A file that grows** is mapped again at its new size, under the file's lock, and the
  earlier mapping is kept until the close, since another sub-actor may be touching it.
  The cost is address space only. Before, the one actor that owned the mapping could
  replace it.
- **What stays charged to `mmap`:** one map and one unmap per open, the faults, and the
  translation flush at the unmap, which now reaches every thread of the instance. §3.35's
  "one address space" caveat is unchanged between instances.
- **The pool thread keeps its backend** now that no backend holds state per descriptor.

Measured (same run as above, warm): `mmap` went from 12.0 s to 2.1 s, 507,608 mappings to
8, 508,669 minor faults to 7,267, 48 s of system time to 13 s, and the read from 198 µs
to 1 µs; `sync` did not change (6.2 s). `copy`, `populate`, and `willneed` are within
0.1 s of `touch`. Counter-arguments and limits, so the row is not over-read: (a) it is a
hypothetical, since DiskANN reads with `O_DIRECT` and asynchronous I/O and the mapping
ignores the abstract's `DIRECT`; warm, it compares a page touch with a device read, which
is the page cache against the device and says little about `mmap`; the cold run (as
root) is the comparison that matters and has not been made. (b) The resident set reads
6.11 GiB because the 781 MiB index is counted once per mapping. (c) With the read at
1 µs the run is the runner's own fork and join: about 130 µs elapsed per hop of four
sub-actors, and about 15 µs of user and 26 µs of system time per sub-actor (the wake of
the pool thread, the VM snapshot, the file table and the payload filler made per
sub-actor). The `sync` rows carry the same cost, about a third of their elapsed time.

**Open.** ~~The per-sub-actor cost above: against roughly 2 µs per op for the VM (§3.33) it
is the next thing to reduce for the `parallel`-heavy abstracts under the blocking
backends (the event loops fork a task, not a thread, and do not pay the wake).~~ Reduced
the same day, §3.38. The cold DiskANN rows.

### 3.37 Count what the runner does; sample only what the kernel does (added 2026-10-01)

CI failed on `tests/uring_knobs.rs`: the shared-`SQPOLL` row reported no `SQPOLL` thread.
The count came from the 10 ms sampler of §3.30, and on the CI host the row ended inside
one period (the sampler's last look comes after the rings are closed). The first fix made
the test's rows last 50 ms. The user asked whether that hurt anything (no: the test only)
and then whether an atomic counter would not be better than a polling loop whose result
depends on how long the task runs. The answer sorts the counters into three kinds:

- **Done by the kernel, on its own schedule: sampled.** io-wq workers are created when an
  op would block and retired after an idle time; no code of the runner runs at either
  moment, and without privilege the thread list is the only view. `iowq_workers_peak`
  stays on the sampler, as does `tasks_peak` (which includes them and glibc's AIO threads).
- **Done by the kernel, at a moment the runner knows: counted once.** An `SQPOLL` thread
  exists from `io_uring_setup` to the ring's close. ~~The loops now meet when each has built
  its ring (`uring.rs`, `Built`), the last to arrive reads the thread list once, and then
  all run.~~ That first form failed in CI as well (2 threads seen of 3): the thread list
  finds them by name, and an `SQPOLL` thread names itself `iou-sqp-*` only when it first
  runs, which on a two-core host can be after all three rings are built. ~~Each loop now
  reads its own ring's `fdinfo` once the ring is built, where the kernel states the poll
  thread from the moment setup returns (`SqThread:`)~~ A second form read each ring's
  `fdinfo` (`SqThread:`) once the ring was built and failed in CI too (2 distinct threads
  stated by three rings sharing one): the CI kernel (Ubuntu 24.04) states the pid of the
  ring's creator in that field until the poll thread first runs, and the thread's own
  after. Each loop now reads its ring's `fdinfo` when its work is done and the ring is
  still open, by which time the thread has carried every submission; the kernel fills
  the field under a trylock and states -1 when it loses, so the read is tried a few
  times. The host's count is the number
  of distinct threads stated; the loops do not wait for each other. The lesson for the
  rule in this section's title: "a moment the runner knows" has to be a moment at which
  the kernel's own statement has settled, and just after setup is not one.
  `HostCounters::sqpoll_threads` (was `sqpoll_threads_peak`) is that count: exact
  at any run length, still an observation of the kernel (a shared poll thread shows as
  one), and taken before any op is issued. The 50 ms was taken back out of the test.
  The same naming delay means the sampled counts can miss a thread that has not run yet;
  for io-wq workers that is harmless, since a worker exists to run.
  Rejected: holding every ring open until all loops end so a last look would see them; a
  finished loop's poll thread would spin on, and its CPU would be charged to the run.
- **Done by the runner: counted where it happens.** The ops a loop has on its ring are the
  loop's own bookkeeping (it already kept the count to detect a stall), so the peak is a
  plain maximum in the single-threaded loop: `UringReport::in_flight_peak`, printed on the
  `io_uring:` line as the `libaio:` line always did, the largest over loops and hosts. An
  actor has one op in flight at most (§3.29), so it is the number of a loop's actors
  waiting on I/O together: the queue depth actually reached.

Not done for the blocking backends: an exact count of threads inside a system call would
be an atomic shared by every actor thread and written twice per op, a contended cache
line the application does not have (a backend is charged only what its API needs, §3.35). It is
bounded by the actor count in any case.

### 3.38 The fork and join of a sub-actor, reduced (added 2026-10-01)

§3.36 left the cost of a `parallel` fork under the blocking backends open: with the
DiskANN read at 1 µs through the shared mapping, the run was the runner's own fork and
join. The user asked for it to be reduced. What it was made of, and what was done:

- **Statistics per sub-actor.** Each sub-actor had its own `Stats` (a latency histogram
  allocated per op kind, a phase map) that the parent merged at the join: for a sub-actor
  of one read, more work than the read's own bookkeeping. A pool thread now keeps one
  `Stats` over all the sub-actors it runs and hands it to the forking actor when the pool
  ends. Nothing reported is per sub-actor, only sums, so the report is unchanged.
- **Copies per sub-actor.** The VM snapshot and the frozen file table were cloned once per
  sub-actor; they are now made once per fork and shared, and a pool thread resumes its VM
  from the snapshot in place (`Vm::resume_from`), keeping its allocations.
- **Wakes of the parent.** Results came back on a channel, which woke the parent once per
  sub-actor. The join is now a count the pool threads take down, and the last one wakes
  the parent once.
- **The forking thread runs sub-actor 0.** It used to sleep for the whole fork. Running
  one sub-actor itself removes a wake and a sleep per fork, and one thread per pool. The
  assignment stays positional (sub-actor 0 here, `k` on pool thread `k − 1`). A sub-actor
  run this way is still a sub-actor: it sees the frozen file table, its own opens end with
  it, a barrier inside it is refused, and its takes are not the main line's. Because it
  may fork while its parent's pool is busy, an actor has a pool per depth of fork it runs
  itself. This changes the thread counts of a run (`parallel` of width W makes W − 1
  threads, and of width 1 none) and nothing else; it is the convention of the thread
  libraries the search codes use, where the calling thread is one of the team.

Measured on the run of §3.36 (507,608 sub-actors of one read, warm):

| | elapsed | cpu user | cpu sys | threads |
|---|---|---|---|---|
| `mmap`, before | 1.87 s | 6.6 s | 11.6 s | 41 |
| `mmap`, statistics, copies, one wake | 1.84 s | 4.4 s | 11.4 s | 41 |
| `mmap`, and sub-actor 0 on the forking thread | 1.55 s | 4.0 s | 9.3 s | 32 |
| `sync`, before | 6.3 s | 13.2 s | 35.5 s | 41 |
| `sync`, after | 6.0 s | 7.6 s | 23.8 s | 32 |
| `io_uring`, one loop | 5.6 s | 1.5 s | 2.4 s | 1 |

User time per sub-actor went from 13 µs to 8 µs, system time from 23 µs to 18 µs, and the
elapsed time of a hop of four from 118 µs to 98 µs.

**What is left is the wake, and it is not the runner's to remove.** The first three
changes took a third of the user time and left the elapsed time where it was: a hop is
three threads woken from sleep and one woken back, and on this host (WSL2, where waking
an idle virtual CPU is a trip through the hypervisor) a wake and its sleep cost about
24 µs of system time. A thread per concurrent blocking read is what the blocking
backends are, so that cost is theirs, and the last row is the evidence: the event loop
does the same 507,608 direct reads with an eighth of the CPU, because a fork there is a
task on the loop. Rejected: spinning before sleeping, in the pool threads or the parent.
It would hide the wake by burning CPU that is then charged to the run, and it would be
wrong for any read slower than the spin. The figure should be re-taken on bare metal
before anyone reads much into it.

### 3.39 `--metrics`: what can be measured on a stream with no global order (added 2026-10-01)

Built as the next item in the recorded order (`runner/REFERENCE.md` §10, `metrics.rs`,
`tests/metrics.rs`). The user asked for the work to resume; the definitions were chosen
while building and **confirmed by the user the same day (decided 2026-10-01)**. The choices and what
argues against each:

- **Order-free and order-dependent metrics are kept apart.** Issue order within a GPU is
  timing and across GPUs there is none, so a reuse distance "of the run" does not exist.
  Popularity, request sizes, the mix, and fan-out are sums over the op multiset and are
  reported over all instances. Reuse distance, runs, and depth need an order and are taken
  per actor instance. *Against:* a server cache sees every instance at once, and the
  per-instance distance understates what it needs for disjoint work (by about the instance
  count) and overstates it for shared hot blocks. A whole-run order would need a model of
  time across instances; a virtual clock from the `compute` nodes with zero-time I/O is
  the obvious candidate and was not built.
- **Round-robin over an instance's sub-actors, not the inline order.** The plain dry run
  walks sub-actor 0 to its end, then 1. For `vdb_search_diskann` that is 32 search threads
  one after another, so blocks shared between threads show distances of a thread's whole
  work. Round-robin (one op per live sub-actor per turn, a nested fork inside its parent's
  turn, a `loader`'s workers beside the forking line) is the linearization in which
  concurrent contexts advance at equal op rates, and it needs no timing. It costs a VM
  per live sub-actor, which the resumable VM of §3.29 already allows. *Against:* it is
  still a model. A loader's workers are not held back by the channel, and a line that
  computes for a long time between ops advances as fast as one that does not.
- **Blocks, not objects, are the unit of reuse.** The DiskANN index is one file, so
  object-level reuse says nothing there, and caches hold blocks. 4 KiB by default,
  `--metrics-block` to change it. Object popularity is reported beside block popularity
  because IVF lists and KV objects are files.
- **Stack distance in bytes, not the gap in accesses.** The distinct blocks since the last
  access, times the block size, is the LRU cache size at which the access hits, which is
  the question storage asks. Split by (previous, current) kind so that read-after-write
  is the write-then-read lag of `GRAMMAR_OPTIONS.md` §5.1.
- **Runs belong to a sequential context.** A run is tracked by the context that issues
  the ops, so two sub-actors reading the same file at their own offsets are two streams,
  as a de-interleaving prefetcher would see them. Tracked per path rather than per open
  handle, since the op carries the path.
- **Depth is structural.** Consecutive `parallel`s of one context with nothing of its own
  between them. On the DiskANN abstract this returns the `hops` distribution exactly.
  *Against:* an abstract that puts a `compute` between hops would report chains of one;
  the trace side has the mirror problem (it needs a think-time threshold to cut chains).
- **Per-block state is accepted here and bounded by sampling.** The invariant against
  per-file data structures is the runner's; this pass does no I/O and is not part of a
  scored run. Hash sampling (SHARDS) keeps one block in `N` and scales; a cap of 2^26
  entries stops the pass with the flag named.
- **A JSON form now, ahead of the JSON report.** A comparison within tolerances is done by
  a tool, so the histograms have to be machine-readable. `--metrics-json` writes
  `aeiou_metrics: 1`; the run's JSON report may absorb it.

Not done: the trace-side tool (strace to the same numbers), tolerances, a whole-run order,
the `replay` node (`trace` since 2026-10-02).

### 3.40 Limits before the gate: an estimate, a refusal, and a counted peak (added 2026-10-01)

§3.8 and R8 asked for a startup probe that computes the needed descriptors "from G, W,
and the abstract", raises the soft limit, fails early with the limit named, and reports
the open-file high-water mark. Built as `limits.rs` (`runner/REFERENCE.md` §11). The choices
were made while building and **confirmed by the user the same day (decided 2026-10-01)**:

- **A bounded walk, not a formula.** G and W do not determine the count: the DiskANN
  abstract holds one file per search thread, the checkpoint abstracts one per writer, a
  loader one per worker. The count is a property of the fork structure and of where the
  opens and closes sit in it, which only the walk knows. An exact answer is a full dry
  run of every instance, minutes at scale, before a host may even say it is ready. So
  the first instance of each template is walked for at most 2^20 ops and scaled.
  *Against:* it is an estimate in both directions. It misses opens past the budget and
  instances that differ from the first, and it counts opens that are expected to fail.
  That is why the line prints `~`, the refusal has an override, and the report carries
  the counted peak to check the estimate against.
- **Concurrency is taken from the structure, not from an interleaving.** Sub-actors of a
  `parallel` are assumed all live at their own peaks at once; a loader's workers the
  same, beside their parent. That is an upper bound for the part walked, which is the
  right side to err on for a limit.
- **Refuse by default.** A host that starts and dies of `EMFILE` takes the whole run
  with it through the coordinator, after the others have dropped caches and waited at
  the gate. The check therefore sits before the coordinator connection. `--ignore-limits`
  is for when the estimate is known to be high.
- **Always raise the soft limits.** The hard limit is the administrator's decision; the
  soft one is a default for programs that do not ask. *Against:* a run then behaves the
  same under `ulimit -Sn 1024` and without it, which hides a misconfigured launch
  environment; the `raised from` note is what is left of that signal.
- **Threads and mappings came with it.** The same walk counts contexts, which are threads
  under the blocking backends, and at 16,000 threads the limit met first is
  `vm.max_map_count` (two mappings per thread), not `RLIMIT_NPROC`. `RLIMIT_NPROC` counts
  the user's tasks in every process, so passing it proves little; it is still compared.
- **The peak is counted, not sampled** (§3.37): opens and closes are the runner's own
  acts, in one place (`OpenFile`).
- **`RLIMIT_MEMLOCK` is not computed.** It bounds ring memory only before Linux 5.12 (since then
  rings are charged to the memory cgroup). No target that old is in view, so the check
  was dropped rather than built; on such a kernel the ring setup would fail with `ENOMEM`
  and the error would not name the limit. Item 9 of the brief named it; this records why.

### 3.41 The JSON report: one document per host, written on failure too (added 2026-10-01)

The text report is for a person; comparing runs, plotting latency, and a submission checker
need the same numbers in a form a tool reads. Built as `report.rs` and
`aeiou run --report-json FILE` (`runner/REFERENCE.md` §12, format `aeiou_report: 1`). The
choices were made while building and **confirmed by the user the same day (decided
2026-10-01)**:

- **A document built for the purpose, not the wire `Report` dumped.** The struct the hosts
  send to the coordinator holds the list of created paths (per file, unbounded), durations
  as `{secs, nanos}`, a 64-bit fingerprint as a number, and histograms as 256 raw buckets
  whose bounds only the runner knows. The document gives a count, integer nanoseconds, hex
  strings, and buckets with their lower bounds. The wire format stays free to change.
- **The identity of the run is in the file.** Abstract hash, seed, GPUs, resolved
  parameters, backend and its options, dataset ids, input-namespace writers, the limits.
  A report without them cannot be compared with another or checked against a published
  hash; the text report prints them, so the JSON holds them.
- **The verdict is a field, and a failed run still writes.** A harness that finds last
  run's file after this run failed reads a pass. So the file is removed at the start and
  written at the end either way, with `verdict.ok` and the error text. The exit status is
  unchanged.
- **Each rank writes its own file; rank 0's holds the merged report.** The alternative,
  rank 0 writing every host's report, would need the per-host reports kept after the merge
  and gains little: the merged report is the result, and a host's own file is on that host
  for whoever wants the split. Rank 0's own part is kept beside the merged one
  (`this_host`) because it costs nothing.
- **Takes: sums always, records on request.** Per-instance stall and compute sums are G
  entries. Every take is steps × G pairs, tens of megabytes for a long run on many GPUs,
  so `--report-takes` asks for them. Per-op records are not offered at all: that is a
  trace, and tracing is a different tool.
- **Quantiles as the text report defines them, plus the buckets.** Bucket lower bounds, so
  the two reports agree to the digit; anything finer is the reader's computation over
  `buckets`.
- **`--metrics-json` is not absorbed** (§3.39 left that open). The metrics come from a dry
  run and describe the stream; a run does not compute them, and computing them in a scored
  run would be work the application does not do.
- **Not done:** a JSON Schema for the document. The format is described in the README and
  pinned by `tests/report.rs`; a schema is worth writing when a second tool reads it.

### 3.42 The trace side of the metrics: what a trace has that an abstract does not (added 2026-10-01)

§3.39 defined the numbers on the abstract's stream and left the trace side unwritten.
Built as `aeiou-trace` (`builder/aeiou/trace.py`, `builder/REFERENCE.md` §7): `metrics` turns
an `strace` into the same `aeiou_metrics: 1` document, `compare` puts two documents side by
side. The choices were made while building and **confirmed by the user the same day
(decided 2026-10-01)**:

- **Python, in the builder package, as its own command.** Trace analysis is authoring-side
  work and never runs on a client under test; the brief already put it in Python (§8).
  It is not `aeiou-fit`: that tool will write parameter files, this one measures. Standard
  library only, so it runs where the trace was taken. *Against:* a second implementation
  of the definitions (histogram buckets, stack distance, popularity) that must stay equal
  to `metrics.rs`; the test that traces the runner is what holds them together.
- **`strace` text is the input.** It is what `ABSTRACTS.md` §11 already asks for, it is on
  every client, and `-yy` gives the path behind each descriptor. *Against:* it slows the
  application several-fold (irrelevant to these metrics, which are not timings, except the
  chain gap), and it cannot see `io_uring` or page faults. An eBPF or `LD_PRELOAD` capture
  would be a second front end to the same `Metrics`; none is written. *(2026-10-31: queued
  as `PROJECT_BRIEF.md` item 54, an eBPF capture of mapping faults and `io_uring` operations
  beside the `strace`, and a capture kit with its instructions; not started.)*
- **Only paths under `--root` count.** A real process loads libraries, reads `/proc`,
  talks on sockets. The abstract models the I/O on the storage under test, so the trace is
  cut to that before anything is counted.
- **Completion order.** The lines of an `strace` file are in the order calls returned;
  an unfinished call is placed where it resumed. Issue order would need the trace buffered
  until every earlier call is known to have ended. Reuse distance is the only metric that
  sees the difference, and a cache sees data when it arrives.
- **The trace is one instance unless told otherwise.** The capture plan traces one
  application process with its workers, which is one GPU's instance. `--instance-root`
  splits a trace of several by process tree. The alternative, one instance per process,
  would make every DataLoader worker its own instance and lose the reuse between them that
  the abstract's sub-actors share.
- **A thread is a sequential context**, the counterpart of a sub-actor. The runner's
  inline sub-actor 0 (§3.38) runs on the forking thread, so a parent's runs and its first
  sub-actor's could join in a trace of the runner; none did in the three abstracts tested.
- **One listing is one `readdir`.** `getdents64` is called until it returns nothing; the
  abstract's `readdir` is the listing. Counting calls would make the mix depend on the
  directory size and the libc buffer.
- **Fan-out and depth only from `io_submit`, and no default think time.** A batch of AIO
  requests is a fork the trace shows. A pool of threads each doing `pread` is not: nothing
  in the trace says which reads belong to one round, and guessing from timestamps would
  turn scheduling noise into structure. Depth needs a think-time threshold (§3.39 said
  so); any default would be a number about someone's application, so there is none and
  depth is absent without `--chain-gap-us`. *Against:* DiskANN through `io_uring`, or
  through threads, cannot be checked for depth with this tool at all.
- **AIO results are matched, not assumed.** The first version took every submitted
  request as fully transferred; on the runner's `libaio-direct` trace that counted 201 MB
  for 11.5 MB read, since reads past EOF are requested at full length. Results are now
  taken from `io_getevents` by context and `aio_data`.
- **`compare` reports distances and decides nothing.** Share differences, the largest CDF
  difference over the shared bucket grid (Kolmogorov–Smirnov on bucketed values), and
  total-variation distance for exact distributions, all in [0, 1]. Tolerances are the
  open part of item 14 and belong to whoever accepts an abstract for a workload class;
  `--max-distance` is there for when they exist. (Built 2026-10-02 as `--judge`, §3.55.)
- **The format gains `source`.** `"dry-run"` or `"strace"`; the `total` is the same shape,
  so the format number stays 1. A trace document has no templates and no fingerprint.

Checked: the runner under `strace` (`sync`, one GPU) on `train_small_files`,
`kv_cache_serving`, and `vdb_search_diskann` gives the dry run's request sizes, run
lengths, popularity, first touches, and data op counts exactly; reuse distance differs by
the order (CDF distance 0.07 to 0.20 at these sizes). No trace of a real application has
been taken; that, and the tolerances, are what remains of item 14 besides `replay` (the `trace` node since 2026-10-02).

### 3.43 The first trace of a real application: `train_small_files` against PyTorch (added 2026-10-01)

Until now `aeiou-trace` had only seen the runner itself (§3.42). The first row of the
capture plan (`ABSTRACTS.md` §11) was run on the loopback NFS mount: `ImageFolder` +
`DataLoader`, batch 16, 2 workers, 2 epochs of 200 steps over 3,200 real JPEGs. The kit
(corpus writer, the traced script, the fitted parameters, the trace's metrics) is
`builder/traces/train_small_files`; the findings are in `ABSTRACTS.md` §1 "Trace". **The
choices below were made while building and confirmed by the user the same day (decided
2026-10-01).**

- **The abstract was corrected from the trace, and its hash and fingerprints changed.**
  Two `lseek`s per file, not one (Pillow's `fp.seek(0)`); `stat` and `fstat` per directory
  in the walk. AST `11ccbefe…` became `46a86b00…`; the golden fingerprint of the abstract
  was re-recorded. Both are local calls on NFS, so no wire count moves; they are in the
  abstract because the abstract is the application's call stream, and client CPU per op is
  the number this workload watches. *Against:* the second `lseek` is Pillow's, and a loader
  that decodes with another library (DALI, `torchvision.io.decode_jpeg` on bytes read
  whole) does not issue it. The abstract models the reference pipeline; another decoder is
  another abstract or a boolean parameter.
- **The root listing is not modeled.** One `open`/`fstat`/`getdents`/`close` of the dataset
  root per actor. It would need a handle for a dataset's root directory, which the builder
  does not have, for 4 ops in a run.
- **`O_NONBLOCK` on the directory open is dropped.** The contract's flag list does not
  have it and it has no effect on a directory. Adding a flag is a contract change for
  nothing measurable.
- **The decode time is recorded, not modeled.** About 3 ms of CPU per file, with the file
  open, against 0.3 ms in its calls. Modeling it means a `compute` between the last read
  and `close` inside the loader's worker, with a `decode` parameter that is [measure] per
  CPU and per image size. It matters only when the loader, not the step, is the limit.
  Not added (decided 2026-10-01): a default of 0 changes nothing and any other default is a
  claim about the client's CPU.
- **A committed regression test, not a tolerance.** `tests/test_trace.py` compares the
  abstract at the fitted parameters with the committed trace metrics: op counts equal but
  for the root listing, request-size buckets equal, largest distance at most 0.06
  (measured 0.048, the reuse distance). This is a guard for this one pair, not the
  acceptance tolerance of `PROJECT_BRIEF.md` §6 item 14, which is still not set.
- **What the comparison could not see.** Both epochs touch every file once, so popularity
  is flat on both sides by construction, and the reuse distance only tests the reshuffle.
  The trace says nothing about timing that survives `strace` (the 3 ms is from an untraced
  run). The second epoch never reached the server for data: with a corpus smaller than
  RAM the abstract and the application agree on the calls and the page cache decides the
  rest, which is the dataset-size rule's business (`PROJECT_BRIEF.md` §5).

Also fixed: `aeiou-trace compare` on a missing or malformed file printed a traceback.

### 3.44 The second trace: `train_large_samples` was a guess, and the trace replaced it (added 2026-10-01)

Row 2 of the capture plan: `np.load(path, allow_pickle=True)["x"]`, the call in upstream
DLIO's `npz_reader.py` (argonne-lcf; qualified 2026-10-01, §3.46), through a `DataLoader` on loopback NFS, over archives written as DLIO's
`npz_generator.py` writes them (`np.savez(x=volume, y=labels)`). Kit in
`builder/traces/train_large_samples`, findings in `ABSTRACTS.md` §2. Unlike §3.43, where
the draft was one call short, here the draft was wrong in shape. **The choices below were
made while building. The user confirmed the explicit seeks (decided 2026-10-01); ~~the
others are not yet confirmed.~~** **Reviewed by the user 2026-10-02** (§3.53): `framing`
stays a parameter and gains a check against the installed NumPy; the listing follows the
real training script and is now in the abstract; the GETATTRs were the tracer's. ~~The
remaining items (the two-loop seek count, the two corpora and their tolerances) were not
commented on.~~ **Decided 2026-10-02:** the user confirmed the remaining choices (§3.54).

- **What the draft had wrong.** Two members of equal size, each found by a read of its
  local header and then read from a re-seeked offset; no read at offset 0 before the tail;
  an EOF read after the tail read; one `lseek` per member. The application reads one
  member (`x`; `y` is 191 bytes and is never asked for), reads the first megabyte twice
  (the magic, then the data), probes for a zip64 locator, never issues a zero-length read,
  and issues one `lseek(0, SEEK_CUR)` per 256 KiB NumPy chunk, so four of five calls are
  seeks.
- **The abstract was rewritten; hash and fingerprints changed.** AST `ee7689cb…` became
  `5e93a2b6…`. Parameters `members` and `lh_len` are removed, `np_chunk` and `framing`
  added, `cd_len` goes from 200 to the traced 102. A parameter file naming the removed
  ones is now refused, which is the intended failure.
- **Seeks are explicit ops, reads are sequential.** The draft used reads with an `offset`
  (a `pread`), which is one call where the application makes two. The op stream is the
  application's, so the seeks are in it. *Against:* 80 % of the abstract's ops now do
  nothing on any storage. They are kept because client CPU per byte is part of what this
  workload measures, and a backend is charged what the application does.
  **Decided 2026-10-01** after the user asked whether the seeks spend client CPU that
  more virtual GPUs could use. Measured on warm local files, 4.43 GiB read, `sync`:
  18,382 seeks cost 0.57 s of runner CPU in two runs, 4,872 seeks (`np_chunk = xfer`)
  cost 0.65 s and 0.54 s. The 13,500 seeks are below the noise; the CPU is the copy. On
  the authoring side explicit seeks make the per-file skeleton mechanical (one line per
  distinct call, no position tracking by hand, and the op mix is checked by `compare`)
  and make the count rules somewhat more work (position queries tied to a library's
  chunking need their own loop). The abstract stays compact either way: 15 op lines for
  a trace of 23,000 calls. If a workload ever is seek-bound, the saving belongs in a
  runner option, which changes the op stream and is its own decision.
- **The seek count is exact, by two loops.** Whole buffer fills carry four seeks each; the
  remainder of the reads and of the seeks follow. The order inside the last megabyte
  differs from the application's (reads, then seeks); the counts per file are equal for
  every file of both corpora. An approximation (four per fill throughout) was 0 to 3
  seeks per file off and was replaced once the rule was understood.
- **`framing = 498` is a parameter, not a constant.** It is the layout of
  `np.savez(x=, y=[0])` with NumPy's forced zip64 local headers. Another generator
  (more members, longer names, `savez_compressed`) has another value or another shape.
- **Two corpora, two committed metrics documents.** Sixteen 140 MiB files are the
  reference shape and settle everything but the reuse distance; 256 files of 8 MiB settle
  that (0.040 against a seed-to-seed 0.062). The test allows 0.3 and 0.07 on the reuse
  distance and 0.01 on every other row; counts within 1 %, since the abstract draws its
  sizes and the corpus has its own.
- ~~**Not modeled.** The `glob` that lists the corpus (two directories). DLIO lists files
  itself and differently; the walk is not part of this abstract.~~ **Modeled 2026-10-02**
  at the user's direction (follow the real training script, not DLIO): an optional
  `enumerate` phase (§3.53).
- ~~**Seen and not explained.** About one GETATTR per READ on the wire. Recorded in
  `ABSTRACTS.md` §2; whether it is the client revalidating during a long buffered read
  without a delegation is a question for a run with `rpcdebug`, not for the abstract.~~
  **Explained 2026-10-02:** they are `strace -yy`'s, not the application's (§3.53).

The lesson for the remaining six rows: the drafts marked **[verify]** are hypotheses. One
was nearly right and one was not.

### 3.45 The third trace: `ckpt_write_dcp`, and what it says about direct backends (added 2026-10-01)

Row 3 of the capture plan: `torch.distributed.checkpoint.save` on two ranks over a state
dict of `DTensor`s, and `torch.save` on one. Kit in `builder/traces/ckpt_write_dcp`,
findings in `ABSTRACTS.md` §3. The draft had the right frame (mkdir, shard file, fsync,
`.metadata` through a rename) and the wrong inside of the item loop. **The choices below
were made while building and are not yet confirmed by the user.** *2026-10-02:* the user
reviewed this section and confirmed one thing, that `O_DIRECT` checkpointing stays on the
list to be evaluated at low priority (§3.46, `PROJECT_BRIEF.md` §6 item 19); ~~the other
choices were not commented on.~~ **Decided 2026-10-02:** the user confirmed the other
choices (§3.54).

- **The item loop is the traced one; hash and fingerprints changed.** Per item a tell, the
  writes, a tell; an item above the buffer is three writes (704, the item, 873), one
  within it is a single coalesced write. `tail` (a guessed 64 KiB) is replaced by `hdr`,
  `trailer`, and `buf`. The `.metadata` writer gains the `fstat`/`ioctl`/`lseek` every
  Python `open` issues, and two `stat`s of `.metadata` that expect ENOENT are added. AST
  `d537f867…` became `54872239…`.
- **The split at `buf` is a `when`/`otherwise` on a parameter array element.** The op
  multiset is still fixed before the run: the condition is over parameters. *Not modeled:*
  the band between `buf − 704` and `buf`, where CPython flushes the header and buffers the
  item; and Linux's cap of 2 GiB − 4 KiB per `write`, which splits an item larger than
  that (a shard of a very large embedding) into several.
- **`ckpt_restore` follows the writer.** Its default `item_off` table is now the prefix sums
  of `hdr + item + trailer`; the tests that write and then restore pass the same. Its own
  read protocol is still the draft's and is row 4.
- **Ten path calls are left out**, listed in `ABSTRACTS.md` §3: no `access` op in the
  contract, no handle for a namespace's parent directory (the schema refuses a namespace
  with no fields), and one `stat` whose issuer is decided by a race. Adding `access` and a
  parent handle would be a contract change for 8 calls per checkpoint; the race cannot be
  in a fixed op multiset at all.
- **Per-rank instances on the trace side.** `--instance-root` per rank is the documented way
  to compare a multi-rank trace (§3.42); this is the first trace that needed it.
- **A direct backend cannot run this abstract at its traced parameters, and that is
  correct.** The runner refuses unaligned `O_DIRECT` writes (`check_align`), and 704 and 873
  are not aligned. Under the interposition test (`PROJECT_BRIEF.md` §5) a shim that turns
  these writes into aligned direct I/O must buffer and pad, which is solution-side work the
  runner does not do for it. So `ABSTRACTS.md` §3's cut "an `O_DIRECT` checkpoint writer is
  a backend choice" is struck: it is another application. The runner's tests that ran this
  abstract under the direct backends now pass aligned `hdr` and `trailer` (4096) to keep
  the direct write path covered. ~~*Open for the user:* whether the runner should offer what
  such a shim does (coalesce and pad to alignment, final truncate) as a named backend
  behaviour, which would let OPEN-style direct runs use the PyTorch stream; nothing built.~~
  **Deferred by the user 2026-10-01** (§3.46): no abstract and no runner behaviour for
  `O_DIRECT` checkpointing yet; it is item 19 of `PROJECT_BRIEF.md` §6, low on the list.
- **`torch.save` is recorded as a variant, not built.** One `writev` per tensor; the
  contract has no vectored op, and a `write` of the summed length is the same request to
  the file system.

### 3.46 DLIO is not an application to trace; `O_DIRECT` checkpointing is deferred (decided 2026-10-01)

The user asked whether the traces were of DLIO, whose MLCommons fork
(`mlcommons/DLIO_local_changes`, v3.0.5 read here) has `O_DIRECT` paths that could be the
model for a direct checkpoint abstract. They were not: the traces are of PyTorch and NumPy.
Reading the fork beside the traces gave the comparison the user wanted:

| | Real library (traced) | MLCommons DLIO fork (source read) |
|---|---|---|
| `.npz` sample read | `np.load(...)["x"]`: magic at 0, zip tail, then 1 MiB buffer fills front to back with one seek per 256 KiB; about 720 calls for 140 MiB | default: whole-file buffered `open().read()` on a 64-thread prefetch pool, decode skipped; `odirect`: `O_DIRECT` open and one `readv` of the file rounded up to 4 KiB, zip parsed in memory; `direct://`: s3dlio's Rust runtime, 64 in flight |
| Checkpoint write | `dcp.save`: per item 704 + storage + 873 bytes, split at the buffer size, `fsync`, `.metadata` by rename; `torch.save`: one `writev` per tensor, no `fsync` | a streaming writer of 32 MiB chunks of generated data, buffered with `fadvise`, or direct through s3dlio; neither `torch.save` nor DCP is called |

Upstream DLIO (argonne-lcf) does call `np.load(...)["x"]`, which is where row 2 took the
call from; §3.44 and `ABSTRACTS.md` §2 said "DLIO" without that qualifier and now carry it.

**Decided by the user:**

- **No abstract is built from DLIO, with or without `O_DIRECT`.** DLIO is itself an
  emulator, and the fork's read and checkpoint paths are not what the frameworks issue;
  an abstract traced from it would carry those liberties into the thing meant to remove
  them. The applications are the frameworks.
- **`O_DIRECT` checkpointing is deferred.** No abstract, and no pad-and-coalesce behaviour
  in the runner (§3.45's open question, struck there). `ckpt_write_dcp` stays a buffered
  workload; a direct backend refuses it at its traced parameters. To come back to, from a
  trace of a real direct checkpoint writer if one is found (`PROJECT_BRIEF.md` §6 item 19).

**Measured 2026-10-28.** The user asked for the comparison the source reading above
predicted. The fork (3.0.5, the driver's venv) and the loader behind `npy_per_sample`
(`kit/train.py`, `np.load(...)["x"]` through a `DataLoader`) were traced on one corpus, 28
`.npz` of about 140 MiB, two epochs, at batch 2 with 2 workers and at the WG's H100 shape,
batch 7 with 4 workers, on local ext4 (the loopback mount needs root, which the session did not
have). `aeiou-trace metrics` on each, `compare --judge` with the fork as *A* and the loader as
*B*. Nothing of DLIO is modeled or committed; this is the evidence for the decision.

| arm | ops under the root | reads | bytes read | whole-file reads | one sample read |
|---|---|---|---|---|---|
| loader, both shapes | 95,306 | 63,084 | 8.22 GB (1.00×) | 56 | 1,108 reads, 562 lseeks |
| fork, batch 2 / 2 workers | 674 | 148 | 10.84 GB (1.32×) | 74 | 1 read of the file, 1 read of 1 byte |
| fork, batch 7 / 4 workers | 1,268 | 280 | 20.55 GB (2.50×) | 140 | same |

- The fork's sample read is `open`, `fstat`, `ioctl`, two `lseek`, `fstat`, one `read` of the
  whole archive, one 1-byte `read` at EOF, `close`; nothing is decoded. The loader's is the zip
  tail probes and then NumPy's 256 KiB chunks with one `lseek` each; on ext4 Python's buffer is
  `st_blksize` = 4 KiB, so each chunk is a 4 KiB fill and a 252 KiB direct read (on the traced
  NFS mount, 1 MiB fills, 144 reads per file).
- The fork reads the corpus more often than its training loop does: `main.py` "pre-warms" the
  loader's workers before epoch 1 and after every epoch for the next one, the epoch after the
  last included, and each pre-warm iterates the dataset, whose reader prefetches the worker's
  whole shard (`ThreadPoolExecutor(64)`). At the H100 shape a two-epoch run read the corpus
  five times, 2.50× the loader's bytes; with 2 workers, 1.32×. Bytes-per-epoch figures taken
  from this fork carry that factor.
- Verdict: 15 of 21 judged rows outside at both shapes (data ops 22 % of ops against 66 %;
  read request p50 1 B and p90 128 MiB against 4 KiB and 224 KiB; runs of one op against
  1,024; first touches 38 % and 20 % against 50 %). Within: the read/write mix, the readdir
  share, popularity of the top 0.1 % and 1 %.
- Found beside it, a gap of the abstract and not of this decision: `xfer` and `hdr_read`
  follow `st_blksize`, but the model issues one read per buffer fill; where `st_blksize` is
  smaller than `np_chunk` the real reader issues a fill of `st_blksize` and a direct read of
  the rest, so against the ext4 trace the abstract has half the reads and an lseek share of
  50 % against 34 %, while reuse, popularity, run-length bytes, and mix are within. The
  committed NFS captures are unaffected. Open, for `npy_per_sample`'s next revision.

**The user's conclusion (2026-10-28).** Traces of the real application workloads are the
requirement for supporting a workload in aeiou, not a refinement: the emulations the WG has
been running have drifted further from the applications than believed. Said plainly, "Unet3D"
run by aeiou and "Unet3D" run by DLIO are not comparable workloads, and if MLPerf Storage
moves to aeiou as its workload generator, results from before the transition will very likely
not be comparable with results after it, because aeiou is more accurate than the code before
it, not less. For item 45 (§3.85): a case built from the WG's DLIO configuration takes DLIO's
parameters, batch, step time, and sizes, never its op stream, and stays `unsettled` until the
real application is traced. The traces, metrics, and compare tables stayed in the session's
scratch; none are retained here.

### 3.47 The fourth row: `dcp.load` and safetensors `from_pretrained` (added 2026-10-01)

Row 4 of the capture plan, both halves. Kits in `builder/traces/ckpt_restore` and
`builder/traces/model_load`, findings in `ABSTRACTS.md` §4. Both drafts were wrong inside:
the restore had two reads per item and the real reader has six and 34 seeks; the model load
had header reads and the real library issues no read at all. **The choices below were made
while building and are not yet confirmed by the user**, except the two put to the user as
questions, which were decided the same day and are struck here (§3.48). *2026-10-02:*
deferred by the user until a GPU environment is available; the tensor table and the
copy's thread count cannot be settled without one. **Decided 2026-10-02:** the user
confirmed the choices that need no GPU (§3.54); those two stay open.

**`ckpt_restore`.**

- **The item loop is the traced one; hash and fingerprints changed.** DCP calls `torch.load`
  once per item on a view of the shard file, and everything under it goes through one Python
  `BufferedReader` of `st_blksize`. The sequence the zip reader asks for is fixed (magic,
  length, magic, end record, central directory, the leading records, the storage); what
  reaches the kernel depends on whether each seek lands in the buffer.
- ~~**The buffer is modeled with four conditions, not simulated.** `far` (the item is at
  least a buffer long), `held` (the item before was small, so its fill holds this head),
  `last` (the fill at the end record stops at EOF), and the storage's remainder past the
  buffer. A general model needs the buffer's start as state carried from item to item,
  which the language can express as a chain (`x @ t-1`) at the price of an abstract nobody
  can read. The conditions reproduce both traces call for call. They are inexact for a run
  of small items longer than the buffer (one fill missing per megabyte of such a run), which
  no trace here contains. *Open for the user:* whether that is acceptable or the chain is
  wanted.~~ **Superseded 2026-10-01** (§3.48): the chain, after a third trace.
- **`rec2`, `eocd_scan`, `tail_back` are parameters with traced values.** They are facts of
  torch 2.14's archive layout and of its zip reader, as `hdr` and `trailer` are in §3.45;
  another version changes the numbers, not the shape. A small last item has them 44 bytes
  off (its records are shorter); the test allows the 88 bytes that costs two ranks.
- **Explicit seeks, again.** 87 % of the calls are `lseek`, most of them `lseek(0, CUR)`.
  They stay in the abstract by §3.44's decision; here they are also the only way the
  abstract shows why a restore is six reads per item.
- **`.metadata` is `ceil(meta_bytes / buf)` fills and no EOF read.** Traced only below one
  buffer; the general form is an extrapolation and is marked so in `ABSTRACTS.md`.
- **Two kits' worth of evidence in one directory.** `mixed` is where the buffer effects
  were read off; `small-last` is the checkpoint §3's kit writes, kept so that write and
  restore are checked on the same files. Both are in `tests/test_trace.py`.
- **Without root, a cold read is `fsync` then `posix_fadvise(DONTNEED)` per file.** It was
  enough to see each byte come from the server exactly once (159 READs, 153 MB) while the
  application read 182 MB.

**`model_load`.**

- **The reads stay, as the shape of what the mappings touch.** The library maps each shard
  twice and never calls `read`. The abstract is POSIX-shaped and the same for every backend
  (CLAUDE.md), so the touches are `read` ops and the `mmap` backend turns them back into
  faults. ~~*Open for the user:* under the interposition test the application's API here is
  `mmap`, not `read`; a CLOSED run of this abstract under `sync` measures 1 MiB READs the
  application never issues (451 against about 1,650). Whether CLOSED for this abstract
  means `mmap` is a rule to decide, not something built.~~ **Decided 2026-10-01** (§3.48):
  the abstract declares `mmap` and the runner defaults to it.
- **One descriptor per file and actor.** The application holds two descriptors on a shard
  at once. The runner keys an actor's open files by path, so the abstract closes the first
  before the second open; same calls, one order differs. Supporting two would change the
  runner's handle definition for the sake of an order no file system can observe on its own.
- **The second close comes after the touches.** The application closes before touching;
  the abstract cannot read a closed file. On NFSv4 this moves a CLOSE, nothing else.
- **A second dataset for the three JSON files, and `model/` for the shards.** One size for
  the three (`config_bytes`), their own root, and the shards out of the top level. This
  changes the `model` dataset's id.
- **V13 gained a rule: no dataset root inside another.** The first layout (shards at the
  top, `config/` below) built, validated, and failed only at `aeiou datagen`, which wants
  each root empty and to itself. The builder, `schema/check.py` and the runner's validator
  now refuse it. A validator rule is part of the contract's README; no AST that validated
  and ran before is refused, since such a layout never got past datagen.
- **The tensor table is still the draft's.** What a tensor-parallel GPU engine touches per
  rank (column slices, row strides) was not traced: it needs GPUs and the engine. What the
  trace does settle is when the bytes move (at the copy to the device, not at load) and
  through what (page faults on a mapping advised sequential).
- **The copy's thread count is visible on the wire.** One copying thread: 1,648 READs of
  about 260 KiB. Torch's default parallel copy on CPU: 2,629 to 2,848 READs of about
  160 KiB, because several threads fault one tensor and readahead restarts. The runner's
  `mmap` backend, one touch per op and one thread per actor, gives 1,687. A device copy
  from pageable memory is one thread as far as is known, so the runner's number is taken as
  the model; **[verify]** on a GPU host. Nothing was added to the runner to imitate the
  parallel copy (the fan-out, if real, belongs in the
  abstract as sub-actors, not in the backend).

### 3.48 What CLOSED means; the declared backend; the restore mimics the reader exactly (decided 2026-10-01)

Three answers from the user to §3.47's questions, and what was built from them.

**CLOSED and OPEN are about comparability, not about an API.** The user's definition, from
how MLPerf Storage uses the words: CLOSED means every system under test sees exactly the
same operation sequence, so results across makes and models are comparable. OPEN lets a
submitter say "you saw our CLOSED result; change your application like this and look how
much better we do", and is comparable with nothing. v2.0 supported only buffered POSIX;
v3.0 added `O_DIRECT` for training and checkpointing (and the S3 object API), and let a
submitter choose buffered or direct and still be CLOSED. The user calls that a place where
more error crept into the imposed workload than should have, and does not want it here.
The words themselves are workgroup process (`PROJECT_BRIEF.md` §8); the mechanism below is
general.

**The aim is the pattern the real application imposes on storage, warts and all.** Nobody
claims PyTorch issues efficient I/O; it is what runs. The price was assessed before
building: nothing at run time (the runner spends about 2 µs per op and nothing per byte;
the warts are seeks and re-reads of cached megabytes), and upkeep instead, since the warts
belong to library versions (torch 2.14, safetensors 0.8) and an abstract needs a retrace
when its library changes.

**Decided and built:**

- **An abstract declares the backend its application uses; a run defaults to it** (contract
  0.3). Optional root key `backend`, a runner backend name, absent meaning `sync`.
  `aeiou run` without `--io-backend` uses it. With another backend the run proceeds, prints
  that it is not the abstract's, and the JSON report carries `backend` and
  `backend_declared`; a comparison takes only runs where they are equal. `model_load`
  declares `mmap`. This keeps the invariant that the abstract is POSIX-shaped and the same
  for every backend: the op stream and the fingerprint do not change, and the field is in
  the AST's hash and nowhere else. The alternative, `mmap` and touch ops in the contract,
  would have put an API into the op stream for the same pattern on the wire.
- **One abstract, one API: buffered and direct are two workloads.** `sync-direct` on an
  abstract that declares `sync` is reported as not the abstract's, like any other
  difference. An application that opens a file `O_DIRECT` says so in its own `open` flags
  (DiskANN's index), which is part of the op stream. `O_DIRECT` checkpointing stays
  deferred until a real direct writer is traced (§3.46).
- **Every committed AST was regenerated for 0.3**; hashes changed, no fingerprint did. The
  untraced abstracts declare nothing yet; DiskANN's `io_submit` and FAISS's choice of
  `pread` or a mapping are set when rows 5 to 7 are traced.
- **The restore carries the buffer, and reads in name order.** Before building the chain a
  third trace was taken to see what a run of small items longer than the buffer does. It
  showed more than that: `dcp.load` sorts the state dict's keys, so items are read in the
  sorted order of their names (`layer1`, `layer10`, …, `layer2`), while `dcp.save` wrote
  them in the state dict's order. On a real model (`layers.0` … `layers.31`) the restore
  jumps about the shard file. The four conditions of §3.47 could not express that; the
  abstract now has `read_order` (the place in the file of the k-th item read) and `bst`,
  the buffer's start after each item, defined from `bst @ k-1`. Three tests on it
  (`head_in`, `end_in`, `far`) and `last` decide every seek and fill. All three traces match
  call for call, the third in bytes too.
- **The `x @ i` values are kept in the VM.** The chain as first run recursed to the start of
  the loop from every item: 900 items overflowed the stack. `x @ i` is a pure function of
  its definition, the index, the indices of the enclosing loops, and the actor, so the VM
  keeps what it has evaluated under that key (bounded; cleared when full). It stores
  nothing the definition does not say, which is the line the positional rule draws: no
  counter, no state a timing could change. A 900-item restore on 8 ranks dry-runs in 17 ms.
  `kv_cache_serving`'s chains go through the same path and its fingerprint is unchanged.
- **The archive's trailer is 873 less the storage's length mod 64.** Found fitting the third
  trace (a 2,096-byte storage has 825 bytes after it: the record after the storage is
  aligned). `ckpt_write_dcp` writes that now; at its defaults, all multiples of 64, nothing
  changes. Its kit's smallest item was 2,096 bytes all along, recorded as 2,048; the byte
  totals had agreed because the two errors cancelled.
- **Still unverified:** which touch pattern a GPU engine's device copy produces (one thread
  or several: 1,650 against 2,700 READs, §3.47); `rec2` and `tail_back` for a small item
  at the end of the file; an item within 4 KiB above the buffer size.

### 3.49 The sixth row: FAISS IVF over `OnDiskInvertedLists` (added 2026-10-01; choices decided the same day)

Row 6 of the capture plan. Kit in `builder/traces/vdb_search_ivf`, findings in `ABSTRACTS.md`
§6. The draft had the right shape (per query, `nprobe` whole lists, no dependency between
them) and the wrong API: its **[verify]** said a thread pool issues `pread`s, and FAISS
issues nothing at all on the lists file after mapping it. ~~**The choices below were made
while building and are not yet confirmed by the user.**~~ **Decided 2026-10-01:** the user
confirmed every choice below.

**What `strace` could and could not give.** The trace has `read_index` (seven reads of the
index file, the lists file opened read-write and mapped shared, no advice) and the threads
(32 new prefetch threads per slice per search call). It has nothing of the lists. The rest
came from the index and from the kernel: list offsets and sizes through the Python binding,
the probed lists from the coarse quantizer, the touched pages from `mincore` on a second
mapping after a cold search, the wire from the mount's READ counters (`faults.py`). Every
page of every probed list was resident, none missing, on 1, 20, and 200 queries.

**Choices.**

- **The abstract declares `mmap`** (contract 0.3, §3.48). This is the second abstract whose
  application never calls `read` on its data.
- **Two reads per list, the ids first.** The prefetch thread sums a list's ids and then its
  codes, and a list on storage is its codes followed by its ids, so the first fault lands
  four fifths into the list. The kernel's read-around window on a mapping is centred on the
  fault (`read_ahead_kb` 128 here), and from that point one window covers the typical 36 KB
  list; from the list's start it does not, and a second window follows. Measured cold, one
  query, 64 lists: FAISS 62 READs and 7.5 MiB; the abstract with one read per list from its
  start 109 to 112 READs and 12.9 to 13.5 MiB; with the ids first, 71 READs and 8.7 MiB,
  the index file's 0.65 MiB included. The order is the application's and costs one `let`
  and one read; `code_size` became a parameter for it.
- **Only the prefetch is issued.** The slice's thread scans the same lists while the
  prefetch runs; whichever arrives first faults. The abstract issues the prefetch's touches
  and then the scan as `compute`, so a call is the fan-out followed by the scan time, where
  in FAISS they overlap. Issuing both would double the reads for pages already asked for.
- **A prefetch thread's lists are a fixed share** (`batch × nprobe / prefetch` each). FAISS's
  threads pull from a queue, which is work distributed by timing and so not expressible
  here by rule; the multiset of lists is the same.
- **The index file is a second dataset with its read lengths as a parameter array.** The
  lengths are stdio's arithmetic over the three large arrays in the file; `params.py` takes
  them from the trace. The defaults are the same pattern scaled to a million lists of 128
  dimensions (528 MiB, mostly centroids).
- **`RDONLY` where FAISS opens `O_RDWR`.** A dataset is read-only (V12), and FAISS writes
  nothing in a search. On NFS the share access of the OPEN differs; nothing else was seen
  to. FAISS has a read-only flag that an application may set.
- **Defaults moved to what SIFT1M shows:** sigma 0.4 for list bytes (was 0.9), Zipf `s` 0.2
  for popularity (was 0.8). One corpus, with its own query set, and a small index (1,024
  lists); a corpus whose queries are concentrated would give another `s`, which is what
  the parameter is for. `queries` became `calls` and `batch`, since the batch decides the
  slices and the prefetch width.
- **Thread creation is not modeled.** 32 `clone3` per slice per call is CPU, not storage.

**What the abstract does not carry, measured.**

- **Read-around in a packed file lands on other lists.** 20 queries, cold: FAISS 435 READs
  and 37 MiB for 23 MiB of distinct lists, in a 40 MB file that the 20 queries probe 59 %
  of; the abstract 764 to 778 READs and 94 MiB, in a 256 MiB file of slots where a window
  around a list reads padding. At one query the two agree (62 against about 66), so the gap
  is the small real file saturating. It should close when the lists file is much larger
  than what a run touches, and that case is not measured: SIFT1M is the largest corpus
  tried. The slot layout stays (§3.18, the naive layout on purpose); this is its cost
  under a mapping, recorded.
- **Draws are independent.** Real queries share lists (607 distinct in 1,280 probes, where
  independent draws at this popularity give about 730), and larger lists are probed more
  (5.7 % more bytes per query than `nprobe` mean lists). Both could be fitted with `x @ i`
  or a joint distribution if a tolerance later says they matter.
- **Under `sync` the same ops cost 1,486 READs and 30 MiB** for the 20 queries: no
  read-around, about one READ per list part not yet in the page cache. It is another workload (§3.48), listed for the size of the difference the backend
  declaration guards against.

### 3.50 The fifth and seventh rows: DiskANN search and build (added 2026-10-02)

Rows 5 and 7 of the capture plan, from one index: DiskANN through `diskannpy` 0.7.0 on
SIFT1M, built in 13 shards on the loopback NFS mount and then searched. Kits in
`builder/traces/vdb_build_diskann` and `builder/traces/vdb_search_diskann`, findings in
`ABSTRACTS.md` §5 and §7. The search draft had the right skeleton and wrong numbers (3 to 8
rounds where there are 28); the build draft had the wrong skeleton (three reads of the base
where there are twenty, 4 KiB layout writes where there are 64 MiB ones, an `fsync` that
does not exist). ~~**The choices below were made while building and are not yet confirmed by
the user.**~~ **Decided 2026-10-02:** the user confirmed the search and the build choices.

**The capture.**

- **`--seccomp-bpf`.** `strace -f` stops every thread at every system call, traced or not,
  unless the filter is installed in the kernel. A build with 20 OpenMP threads did 1/16 of
  its PQ training in 20 minutes without it and the whole build in 407 s with it (161 s
  untraced). Recorded in `ABSTRACTS.md` §11.
- **`%desc` has no AIO calls**; they are named in the kit's command and in the runner-trace
  test, which now runs `vdb_search_diskann` under the backend it declares.
- **Query boundaries are not in a trace.** `aeiou-trace`'s chain gap (§3.42) found no
  threshold under `strace`: between 30 and 200 µs it goes from every round a chain to
  chains of hundreds. The kit's `hops.py` cuts at the read of a medoid's sector, which is
  the application's own first step, and with a node cache at a `stat` the search script
  issues between queries. The depth metric of the abstract is compared with that, not
  with `aeiou-trace`'s.

**Search: choices.**

- **The abstract declares `libaio`**, with `DIRECT` on the open as before. The runner's
  `libaio` submits a `parallel` of four as one `io_submit` of four here (27,403 submits for
  110,043 requests), which is the application's shape.
- **One descriptor per process.** The draft opened the index in every search thread; the
  trace has one `O_DIRECT` open and sub-actors inherit it. The limit estimate went from 64
  open files to 2 for two instances.
- **Three kinds of round.** A `parallel` of one on the `entry` hot set, a `parallel(beam)`
  on the `near` hot set, then `draw(hops)` uniform rounds. `hotset` ranks go through the
  dataset's permutation, so the hot sectors are scattered, as medoids are. The draft's
  single `when (hop < 1)` and a 1 % hot set at weight 0.30 are gone. Both hot-set
  fractions are configuration (medoids over sectors, and about 0.8 · medoids · degree over
  sectors); `hops.py --params` fits them.
- **A node cache is a count of sector reads at load and a switch.** `node_cache > 0` issues
  that many reads eight at a time and drops the two entry rounds from every query, which
  is what 1 % cached does (the `hops` distribution of the cached trace is the uncached one
  less two, to within sampling). A larger cache also shortens the later rounds (87 reads
  per query at 10 % cached against 101) and that needs its own fitted `hops`.
- **Full rounds.** The search submits fewer than `beam` when a frontier node is already
  held (3 % of submits without a cache, more with one). The abstract always submits
  `beam`: 1.5 % more reads uncached, 3 % with 1 % cached. Modeling it would take a drawn
  width per round; left out.
- **CPU once per query.** Measured 0.5 ms of user time per query (0.46 to 0.62), spread
  over 28 rounds. A `compute` inside the round loop ends a chain by the depth metric's
  definition (§3.39), and the metric is worth more than the placement: the storage sees
  the same closed loop either way, with the think time at the end.
- **The load is the PQ file and the index's first sector.** The PQ file is `nodes ·
  per_sector · pq_code` bytes, derived so that a small test index has a small PQ file. Its
  second read is capped at 2 GiB less a page per call, which is what Linux returns; a
  billion-point index takes 15 calls where the trace took one.
- **diskannpy's sample warm-up is not modeled.** It is that binding's default and not
  DiskANN's, and it is 100 times the traced queries. An abstract for it would be this one
  with `queries` raised and `hops` refitted at search list 15.
- **Small files not modeled** (8 opens, 9 `stat`s, 28 seeks, about 150 KB), once per process.

**Build: choices.**

- **Two helper functions in the script, not two constructs.** `block_pass` and `load_pass`
  are Python functions that emit the nodes: five block passes and three load passes in the
  AST, one of the loads inside the shard loop. The script stays readable; the AST is longer.
- **The no-op seeks are issued.** 825,000 `lseek(0, SEEK_CUR)`, two per row of the last
  partial block of each block pass, are the application's calls, and local calls stay explicit (§3.44).
  They are bounded by the block size, not by the corpus: at most 262,144 per pass at 128
  dimensions.
- **Single calls for whole arrays**, in pieces of 2 GiB less a page: the PQ file and the
  merged graph are written, and a shard's graph read, with one call each in the trace. At
  the default scale the merged graph is 103 calls.
- **Shards at one size**, `overlap · n / shards` rows. Traced 52 to 111 MB around a mean of
  79. A drawn size per shard would need the namespace's sizes to follow a distribution the
  reader also evaluates; `as_written` covers it, but the per-shard graph size would have
  to follow the same draw, and one size keeps the parameter file to eight numbers.
- **Sequential where the builder interleaves** (ids files, the layout's three streams). One
  actor issues one op at a time; the multiset and the bytes are the same.
- **No `fsync`.** The draft's was invented. The index a build leaves is as durable as the
  client's writeback made it.
- **`sample` and `xfer` are gone; `sample_rows` is the warm-up sample's size.** Parameter
  names changed with the shape; fingerprints and the golden configuration with them.
- **The unsharded build is not modeled.** With enough memory DiskANN builds one graph and
  skips partition, shards, and merge. The sharded path is the one whose I/O matters.

**What was measured.**

| | DiskANN | abstract |
|---|---|---|
| search, 1,000 queries, no cache: NFS READ | 108,431 | 110,080 |
| search, 1,000 queries, 10,000 nodes cached (load included) | 112,049 | 115,588 |
| build: reads / writes / seeks issued | 176,070 / 292,159 / 825,306 | 176,041 / 292,414 / 825,205 |
| build: NFS READ / WRITE / COMMIT / REMOVE | 491 / 3,337 / 49 / 61 | 497 / 3,334 / 43 / 52 |

The build's twenty reads of the base are one read on the wire: the client's page cache
serves the other nineteen, for DiskANN and for the runner alike, because the file is
smaller than the client's memory. A base larger than memory would be read from the server
up to twenty times, and that is the case the build abstract exists for; it is not measured.
The untraced build takes 161 s and the abstract's I/O alone 6.5 s on this mount: the build
is compute, as §7 says.

### 3.51 The eighth row: vLLM with LMCache (added 2026-10-02)

Row 8 of the capture plan, the last. Kit in `builder/traces/kv_cache_serving`, findings in
`ABSTRACTS.md` §8. The draft's chain of conversations stands; its I/O skeleton was wrong in
three places (a `stat` per chunk per request, a directory per conversation, writes during
decode) and missed the fact that decides the read volume: the engine has a cache of its own
in front of this one. ~~**The choices below were made while building and are not yet confirmed
by the user.**~~ **Decided 2026-10-02:** the user confirmed the choices.

**What was traced, and what was not.** The call sequence of LMCache's local-disk backend
under a real vLLM on a GPU, with a synthetic chat load. Not traced: a public chat replay.
The reuse distance, the prompt and reply lengths, and the system-prompt popularity stay
**[measure]**; they are properties of a workload's users, not of the software, and need a
replay (ShareGPT or a production log) through the same kit. Other tiers (vLLM's own
connectors, Mooncake, HiCache, a remote LMCache server) are other traces.

**Choices.**

- **The `stat` loops are removed, not parameterized.** The draft kept them as a stated
  choice for "a shared-filesystem backend with no index". The traced backend has an index
  and issues none; a workload for a backend that has none should come from that backend's
  trace. Ops per request fall by more than half.
- **Six calls per chunk**, including the terminal probe that fails `ENOTTY` and the no-op
  `lseek`: they are Python's `open`, as in `model_load`'s small files (§3.47).
- **One flat namespace**, `kv/{conv:016x}-{k:04}.pt`; `kv_dir` and the `mkdir` are gone. The
  real name is a hash of the token prefix; a conversation id and a chunk index give the
  same sharing structure for a conversation's own chunks.
- **Tokens are counted along the chain.** The draft added `ceil((in + out) / chunk)` blocks
  per turn. The trace stores `floor(prompt tokens / chunk)` chunks in total, the reply
  entering with the next turn, so the abstract carries the conversation's own prompt tokens
  (`ptoks @ (r − d) + out @ (r − d) + in`) and stores the whole chunks not yet stored. A
  new conversation starts with the system prompt's tokens past its last whole chunk.
- **`local`: what the engine still holds, in requests.** A conversation back within `local`
  requests reads nothing; one back later reads all the chunks it had. The real quantity is
  GPU KV memory in bytes against the tokens of the conversations in between, which is not
  positional without a simulated cache, and `GRAMMAR_OPTIONS.md` §5.3 rules that out as it
  does for `retain`. On the traced load the abstract reads 88 chunks where the engine read
  47: the engine's memory was still filling during the first turns, and it kept the
  shared first chunk. A steady-state trace with a replay is what would fit `local`.
  *2026-10-02, later:* the replay showed a threshold cannot be fitted; `local` is replaced
  by `keep`, a draw of the tokens the engine still holds (§3.56).
- **`sys_local`.** The system prompts' whole chunks are never read while the engine holds
  them, which on one engine is always. `false` is the cold engine beside a filled cache
  (a restart, or a second engine), where they are read by every new conversation. *2026-10-07:* under `false` they are read when the engine holds none of the
  conversation, a continuing one included, as the AgentX replay showed (§3.63).
- **A request's chunk reads are a `parallel` as wide as the chunks.** The trace shows four
  at once from a pool of four or five threads; a request that loads more than the pool is
  wide would queue, which is not modeled.
- **Writes are issued in the request's line**, after the prefill `compute`. LMCache issues
  them from two background threads while the request decodes.
- **The system prompts stay a dataset** written by datagen (V12, V13): the trace has two
  more writes, the first use of each prompt.
- **`sys_tokens_min` and `sys_tokens_max`** bound the system prompt length so that a fitted
  file can give every prompt one length; a log-normal's sigma is a literal in the contract.

**Measured.** 40 requests: 47 chunk writes and 47 chunk reads of 3,145,728 bytes in the
trace; 48 writes and 88 reads in the abstract at the fitted parameters (three more of the
conversations' own chunks, because some replies ended short of the 100 tokens the fitted
file gives every reply, and two fewer for the system prompts). ~~The NFS RPC counts were not taken for this row.~~

**On the wire (added 2026-10-02, later the same day).** An untraced repeat of the load and
`aeiou run` at the fitted parameters, both on the loopback mount, `mountstats` delta:

| | vLLM + LMCache | abstract |
|---|---|---|
| chunks stored / read back | 47 / 47 | 48 / 88 |
| WRITE / COMMIT / OPEN | 141 / 47 / 47 | 144 / 48 / 48 |
| GETATTR | 35 | 46 |
| READ | 0 | 0 |

Three WRITEs, one COMMIT, and one OPEN per chunk stored, in both. No read of a chunk
reaches the server in either: the client wrote the chunk minutes earlier and still holds
its pages. This is a property of the workload on one client with free memory, and the
abstract reproduces it; it also means that a run of this abstract as traced measures the
write path and nothing of the read path. The reads become wire reads when the store has
outgrown the client's memory, or when the reader is not the writer (a second engine, a
restart; `sys_local = false` is the abstract's piece of that case). Neither case was
measured, ~~and which of them a scored configuration should be is open for the user:~~ it
needs either a parameter set whose store exceeds client memory, or two runs (a writer,
then a reader over an `input` namespace after `--drop-caches`), which is the shape the
checkpoint pair already has (§3.31).

**Decided 2026-10-02** (the user): the cold reader is two runs. A writer run fills the
store; a reader run takes it as an `input` namespace after `--drop-caches`. No parameter
set sized against the client's memory, which would tie the workload to the client's DRAM.
~~Not built yet: the reader half is a second abstract over the writer's namespace, as
`ckpt_restore` is to `ckpt_write_dcp`.~~ Built the same day, from traces of another
backend, since this one turned out to have no reader (§3.52).

**Capture notes.** vLLM keeps a small chat entirely in GPU memory and LMCache is then never
read: the kit limits the engine's KV memory (`--kv-cache-memory-bytes`). The flashinfer
sampler compiles kernels at first use and needs the CUDA compiler; the kit turns it off.
`%network` is left out of the trace.

### 3.52 The shared store: a writer and a cold reader, from LMCache's `fs://` backend (added 2026-10-02)

§3.51 ended with a decision: the KV read path is measured with two runs, a writer and then
a reader over the writer's namespace after `--drop-caches`. The user asked for the reader.
Building it began, as every row has, with what the application does, and the first thing
found was that the application traced in §3.51 has no such reader. ~~**The choices below
were made while building and are not yet confirmed by the user.**~~ **Decided 2026-10-02:**
the user confirmed the choices; the items under "Not done" wait.

**What was tried, in order.**

1. *The local-disk backend, restarted on a filled directory.* It wrote 47 new files beside
   the 47 old ones. LMCache names a chunk by Python's `hash` of the token prefix, which is
   seeded per process (its log says `Using hash algorithm: builtin` and warns of
   "inconsistencies in distributed caching").
2. *The same with `PYTHONHASHSEED=0`.* The names now agree, and the restarted engine opens
   each with `O_TRUNC` and writes it again (OPEN_NOATTR and SETATTR on the wire in place
   of OPEN). It reads back only what it wrote itself. The index of that backend is in
   memory and is not rebuilt from the directory.
3. *The `fs://` remote backend* (`fs_connector.py`): existence by `stat`, a put through a
   temporary name and a rename, a get through `aiofiles`. A restarted engine finds the
   first one's chunks. But its second turn's prompts differed: a model that loaded its KV
   cache replies differently from one that computed it, so from the second turn on the
   history hashes to chunks the store does not have, and 27 of 47 were written again.
4. *The same with the replies replayed* (`chat.py --save`, `--replay`): the restarted
   engine is sent the first one's requests byte for byte. No write, 168 `stat`s, 91
   loads, 188 READs on the wire. This is the reader's trace.

So the draft of this row (before §3.51) had been closer to the `fs://` backend than to the
one §3.51 traced: its `stat` loop and its rename are here.

**Choices.**

- **Two new abstracts; `kv_cache_serving` is unchanged.** The local-disk backend is a real
  configuration with its own call sequence, and as measured it exercises writes only
  (§3.51). `kv_cache_shared` is the same request stream on the `fs://` backend, and
  `kv_cache_shared_reader` is that stream in front of the store the first run left. The
  decision of §3.51 named "a reader over an `input` namespace"; it did not say on which
  backend, because the finding that only one of them can have a reader came after it.
  *Against:* three KV abstracts where there was one, two of them differing only in the
  state of the store.
- **One script, two ASTs, and the draws at the same sites.** The reader has to name the
  chunks the writer left, and the names are positional draws (`conv` is a `uniform64` at
  a new conversation). A draw's key is its actor, its site (the JSON pointer of its node),
  and the loop indices (`runner/REFERENCE.md` §2), so two abstracts draw the same values when
  the draws stand at the same places and the run has the same `--seed`, `--gpus`, and
  parameters. `shape(name, reader)` emits the chain's statements first and identically,
  then the ops. Nothing in the contract or the runner changed. *Against:* the coupling is
  by construction in the script and is not checked by the validator; it is checked by the
  run (a reader with another seed, or built from a script whose prelude moved, fails at
  its first `stat` or `open` with `ENOENT`) and by `tests/run.rs`. The namespace manifest
  compares the namespace's definition, not the seed or the parameters of the run that
  wrote it; ~~recording those and comparing them at the reader's start would turn the late
  failure into a refusal before the gate, and is a contract change left for the user.~~
  **Built 2026-10-02** at the user's decision: `same_run`, V15, contract 0.4 (§3.54).
- **`hit` is the one difference in the chain.** The chunks the store has for a prompt are
  `had` (what earlier turns stored) on an empty store and `stored` (every whole chunk of
  the prompt) on a filled one. Lookups walk `hit` chunks; loads are `hit − held`, where
  `held = (ptoks − inn) / chunk_tokens` for a conversation back within `local` requests
  and 0 otherwise. This is the rule LMCache's log shows in all 80 requests (hit chunks
  less the whole chunks of `Inference Engine computed tokens`). For the writer it gives
  what `kv_cache_serving` already had (`had` when the engine lost the conversation, else
  nothing).
- **The reader issues no prefill `compute`.** Every whole chunk of the prompt is loaded or
  held; the tokens past the last whole chunk are computed by the real engine and are not
  charged here, as they are not in the writer.
- **The writer's terminating `stat` expects `ENOENT` and tolerates a hit.** After an
  eviction (`d > retain`) the abstract stores the chunks again under names that exist
  (no simulated cache, `GRAMMAR_OPTIONS.md` §5.3), where the real store would have
  removed them.
- **`buf` is a parameter** (1 MiB, the mount's `st_blksize`), as in `ckpt_restore`: the
  first read of a chunk file is the buffered reader's fill and its length is the file
  system's answer, not the application's. A chunk file no longer than `buf` is one read.
- **The system prompts stay a dataset**, in chunk files of the same size as the store's
  (header included), with a `stat` per whole chunk in every lookup. The traces have a miss
  and two stores at the first use of each prompt.
- **Backend: none declared, so `sync`.** `aiofiles` runs ordinary blocking calls on a
  thread pool. The connector's `O_DIRECT` option (`fs_connector_use_odirect`) is another
  call sequence and was not traced.
- **`chat.py` gained `--save` and `--replay`**, and the kit sets `PYTHONHASHSEED`. Both are
  conditions for a second engine to hit at all, and worth knowing about LMCache on a
  shared file system apart from this benchmark.

**Measured** (tables in `ABSTRACTS.md` §8). Calls: 47 / 48 chunks stored, 47 / 88 and
91 / 136 loaded (trace / abstract; `local = 8` gives 0 and 32, the traced engine lies
between as in §3.51). Wire, reader: 188 READs for vLLM, 192 for the abstract, four per
chunk file, each file once. ~~Not explained: five WRITEs per chunk from vLLM against the
abstract's four, and three times the GETATTRs.~~ Both were the tracer's: an untraced
repeat sends four WRITEs per chunk and 39 and 37 GETATTRs (§3.53). The reader was made cold with `fsync` and
`POSIX_FADV_DONTNEED` per file, since `--drop-caches` needs root; `aeiou run` warned, as
it should, that the reading host had written the objects.

**Not done.** A reader on a second client (the OPEN and delegation traffic of a client
that did not write the files); two engines on one store at once; the `O_DIRECT` option;
the chat replay for the distributions, which is unchanged from §3.51.

### 3.53 The user's review of §3.44 to §3.52; `strace -yy` is on the wire (added 2026-10-02)

The user reviewed the sections that still carried unconfirmed choices. §3.50, §3.51, and
§3.52 are decided as written. §3.47 waits for a GPU environment. `O_DIRECT` checkpointing
(§3.45, §3.46) stays on the list at low priority. Three remarks on §3.44 led to work.

**The GETATTRs were ours.** The user asked whether the GETATTR per READ seen under
`np.load` (§3.44) came from something the benchmark does, while holding that whether a
client should send them is the file system's business and not the benchmark's. The runner
does not send them: `train_large_samples` at the fitted parameters on the loopback mount
is 2,285 READs and 16 GETATTRs, one per file. Neither does NumPy: `np.load` alone, a warm
second pass, a pass after the attribute timeout, two processes, and the kit's `DataLoader`
script untraced all give 0 to 10. The same script under `strace`:

| | READ | GETATTR |
|---|---|---|
| untraced | 561 | 10 |
| `strace -f` | 561 | 6 |
| `strace -f -y` | 561 | 6 |
| `strace -f -yy` | 561 | 559 |

`-yy` makes the tracer look at every descriptor it prints, and on NFS that revalidates the
attributes each READ has just invalidated. So a wire count taken during a `-yy` capture
overstates GETATTR by about one per READ. The §3.44 count was one; so were the two things
§3.52 could not explain. An untraced repeat of the shared-store pair gives vLLM 188 WRITEs
(four per chunk, as the abstract) where the traced run had 235, and 39 and 37 GETATTRs
where it had 133 and 141; READ, COMMIT, RENAME, and LOOKUP are unchanged, and the
untraced reader sends no OPEN, as the abstract's did not. *Rule from now on:* the call
sequence comes from the trace, the wire counts from an untraced repeat. Recorded in
`ABSTRACTS.md` §11. Counts already in the docs that were taken untraced say so (§3, §8
local disk); the ImageFolder counts of §1 were not re-taken and their GETATTR figure
should be read with this in mind.

**`framing` can be checked, and now is.** The user agreed it is a parameter and asked how
one would know it needs another value, a NumPy version for instance. The version is not
the right key: the number depends on the writer's layout (the `.npy` header padded to 64
bytes, the zip64 local headers `savez` forces), on the member names, and
on how many members there are, and it does not depend on the volume's size. What can be
done is to ask the library: write `np.savez(x=<uint8 volume>, y=[0])` in memory and
subtract. NumPy 2.2.6 and 2.5.3 both give 498, and 102 for the central directory.
`builder/tests/test_formats.py` now does this against the abstract's defaults, so a NumPy
that lays the archive out differently fails the builder's tests when the lock file moves
to it. That covers the defaults. It does not cover a corpus written by something else
(more members, other names, `savez_compressed`): for that the number has to come from a
file of the corpus, which is what `aeiou-params safetensors` does for a model. An
`aeiou-params npz FILE` that prints `framing` and `cd_len` from a real archive is the
matching tool; ~~proposed, not built.~~ built the same day at the user's request (§3.54). The runner cannot detect a wrong value at run time
and must not try: it never interprets sample data, and a generated corpus has no zip
structure to look at.

**The listing follows the real training script.** §3.44 left the kit's `glob` out with
the argument that DLIO lists differently. The user's rule is the other way round: what a
real training job does is the reference, and a job with one file per sample lists its
corpus before the first step. `glob("…/*/*.npz")` is, per directory, `open(O_DIRECTORY)`,
`fstat`, `getdents64` until empty, `close`, with no `stat` (ImageFolder's walk has one,
§3.43). `train_large_samples` gains a parameter `enumerate` and a phase of that name with
those four ops per directory and actor. Choices made while building, ~~**not yet
confirmed**~~ the first reversed and the others confirmed by the user the same day (§3.54):

- ~~*Off by default*, as in `train_small_files`, where whether the walk belongs to the
  measurement was left as WG policy (`ABSTRACTS.md` §1). The user's rule argues for on in
  both. Left as it is until the user says, because turning it on changes the default
  fingerprints of two abstracts.~~ **On by default in both, decided 2026-10-02** (§3.54).
- *Per actor.* Each rank of a distributed job builds its own `Dataset`, as in §1.
- *The root's own listing is not modeled*, as in §3.43 (no handle for a dataset's root).

The abstract's hash changed (`a614e020…` to `a147556b…`); its fingerprints with the phase
off did not.

### 3.54 The walk is on by default; `aeiou-params npz`; `same_run` (decided 2026-10-02)

Four answers from the user to the questions §3.53 left, all decided and built the same day.

**The directory walk is part of the workload.** Asked whether real training runs under
PyTorch walk the tree, the answer is yes with one distinction. `ImageFolder` does, in
every process that builds the dataset: one `scandir` of every class directory, traced
from torchvision itself (§3.43). For one-file-per-sample corpora there is no such library
class; the job's own `Dataset` lists the files with `glob`, `os.listdir`, or `os.walk`,
and the kit's script is one instance of that, not a trace of a named framework. Any
map-style dataset needs the list before the first step, so some listing is always there;
what varies is whether it is read from the file system or from an index file shipped
with the corpus (a manifest, a CSV), in which case the walk is one file read. The user's
rule: stay as close to the metadata load the storage sees as to the data load, even though
the runner needs no list (the Feistel permutation and the name pattern replace it).
So `enumerate` is **on by default** in `train_small_files` and `train_large_samples`. The
runner issues the real `getdents64` calls and keeps nothing. *Consequences:* the default
fingerprints and op counts of both abstracts changed (38,462 directories of five ops per
actor in the small-files default); `enumerate=false` gives the earlier ones, and both are
golden cases. The earlier text that left the default to WG policy (`ABSTRACTS.md` §1) is
struck: whether a division scores the phase is still policy, whether the run issues it is
not. A corpus with an index file is `enumerate=false` plus that file's read, which no
abstract models yet.

**`aeiou-params npz AST ARCHIVE…`.** The user does not want to be surprised by a trace
over a corpus whose archives have another framing. The tool reads `framing` and `cd_len`
from real files (standard library only: the zip directory and the `.npy` header), prints
them against the abstract's defaults, ~~exits 2~~ exits 1 (2026-10-04, §3.61: 2 is the suite's usage-error status) when they differ, and writes a parameter
file with `-o`. It refuses what the abstract does not read: archives that disagree with
each other, a compressed member, the member not first, a zip64 end record, an archive
comment. `tests/test_params.py` covers the traced writer (498 and 102), a writer with
three members and longer names (larger numbers, reported and written), and each refusal.
With the in-memory probe of §3.53 this gives two checks: the defaults against the
installed NumPy in CI, and any corpus against the abstract by hand before a fit.

**The remaining choices of §3.44, §3.45, and §3.47 are confirmed** (§3.44: the two-loop
seek count, the two corpora and their tolerances; §3.45: the split of the item loop at
`buf`, the ten path calls left out, `torch.save` as a variant only; §3.47: everything that
needs no GPU). Still open in §3.47: the tensor table and the copy's thread count.

**`same_run`: the reader of §3.52 is refused before the gate.** The manifest already held
the writer's seed, instance count, and resolved parameters; nothing compared them, and
nothing could by default, because `ckpt_restore` rightly runs under any seed. So the
abstract declares it: `same_run: true` on an `input` namespace (V15, contract 0.4). The
runner then refuses, in the check that already compares the namespace's definition, when
`--seed` or `--gpus` differ or when a parameter both abstracts declare has another
resolved value, and names each difference. *Choices:* a parameter only one side declares
is not compared (the reader may have its own); values are compared in canonical form
after resolution, so a parameter file and a `--param` that say the same thing agree; the
key is on the namespace and not on the abstract, since an abstract may read one namespace
by draw and another by formula. *Not covered:* that the two abstracts draw at the same
sites, which is a property of the script and is tested by running the pair. Every
committed AST was regenerated for 0.4; no fingerprint changed.

### 3.55 Tolerances: `aeiou-trace compare --judge` (added and decided 2026-10-02)

The last open part of the locality-metrics check (`PROJECT_BRIEF.md` §6 item 14) besides
`replay` (the `trace` node since 2026-10-02). Built at the user's request ("#2", the tolerances); the numbers and the rule
below were chosen while building. **Decided by the user the same day (2026-10-02): the class
values are fine for now, and so are the two thin margins** (0.364 against 0.409, 0.106
against 0.107). "For now" is the user's: the values are to be looked at again when more
pairs exist. The `--self` gap under "Not done" was not commented on and stays open. Definition and the table
of verdicts: `builder/REFERENCE.md` §7.

What the fifteen committed pairs said before any rule was written (every trace against its
abstract at the fitted parameters, and the abstract against itself at seeds 2, 3, 4):

- **One number cannot be the tolerance.** The op mix, request sizes, run lengths, and
  popularity do not move with the seed at all (spread 0.000 to 0.02) and are equal or
  within 0.05 on every pair whose abstract was written from its trace. Reuse distance
  moves with the seed and with the size of the corpus: the abstract differs from itself
  by 0.027 on 3,200 files, 0.10 on 256, 0.31 on 16, and the traces differ from it by
  0.048, 0.118, 0.364. A tolerance that passes the 16-file pair passes anything on the
  3,200-file one.
- **The noise is measured, not computed.** A Kolmogorov–Smirnov bound needs the number of
  independent samples, and that is the file for a loader that reads whole files and the
  read for DiskANN (one file, 108,000 independent sectors); no count in the document is
  right for both. So a row's allowed distance is its class tolerance plus the abstract's
  own largest distance to itself at other seeds (`--self`). *Against:* it costs three
  more dry runs; three seeds are a sample, not a bound (the 16-file pair passes at 0.364
  against 0.409, the cached DiskANN pair at 0.106 against 0.107); and an abstract with no
  draws has no spread, so its order differences are held to the class value alone.
- **The class values.** 0.05 for the op mix, request size, and popularity: the largest
  such distance on an accepted pair is 0.046 (`ckpt_write_dcp`, the ten path calls the
  contract cannot express, out of 164 ops). 0.10 for run length, the reuse shares, reuse
  distance, fan-out, and depth, which depend on the order or on thread scheduling: the
  runner traced against its own dry run differs by 0.07 to 0.20 in reuse distance on
  small configurations from the order alone (§3.42); the largest on an accepted real pair
  after the spread is taken off is 0.10. They are round numbers fitted to nine pairs, and
  a pair of another size or another application may show they are wrong.
- **Blindness is declared, not inferred.** `strace` shows no page fault and no thread-pool
  fork, and a run belongs to a thread. A row the trace cannot show is listed in the pair's
  file with the reason and is not judged. The tool does not guess: a trace with no fan-out
  and an abstract with one is outside unless the file says why. Two things are inferred
  because the document states them: `depth` without `--chain-gap-us`, and a popularity
  share over fewer than ten units.
- **A known difference is recorded and still fails.** The pair's file can give the reason
  for a row that is outside; the row stays outside and the pair is not accepted. A waiver
  that turns a known difference into a pass would make the verdict mean "someone
  explained it". The test pins each pair's verdict and fails on an outside row without a
  reason and on a stale entry.
- **These are this repository's defaults,** not the WG's. Accepting an abstract for a
  workload class is the accepting body's decision; it can publish its own file
  (`PROJECT_BRIEF.md` §8).

What the rule found:

- **Nine pairs accepted:** small files, large samples (two corpora), checkpoint write,
  restore (three traces), DiskANN search (two).
- **`vdb_build_diskann`, nine rows outside.** The run-length histograms count runs, and
  33 of the trace's 106 read runs and 50 of its 107 write runs are small files and file
  headers the abstract leaves out, while calls and bytes agree to 0.1 %. The traced files
  begin with a small header, so the stream's 8,192-byte writes start off a block boundary
  and every write rewrites the last block of the one before (291,000 block rewrites in
  the trace, none in the abstract). Two reuse distances (0.20 and 0.37) are not
  explained row by row; shards modeled at one size is the candidate.
- **The three KV-cache pairs, 6, 7, and 4 rows outside,** one cause: at the fitted
  parameters the abstract loads 88 (136 for the reader) chunks where the traced engine
  loaded 47 (91). §3.51 and §3.52 said the synthetic load fixes the call sequence and not
  the distributions; the tolerance now says so with a verdict. The chat replay is what
  can change it. *2026-10-02, later:* it did, to 1, 2, and 1 rows (§3.56).
- **`model_load` and `vdb_search_ivf` cannot be judged this way.** They read through a
  mapping; with the reads on one side only, every share is of a different total. Their
  evidence is the exact call counts of their tests and the `mincore` measurements.
- **A tool limit shown by the shared store:** LMCache issues the two reads of one load
  from different pool threads on one descriptor, so the trace has two runs where the
  abstract has one. The sequential context could be the open file description instead of
  the thread; that is a change to a decided definition (§3.42) and was not made.

Not done: the parameters are not in a metrics document, so a `--self` document of another
parameter set is not refused; the existing per-pair assertions of `tests/test_trace.py`
(0.3, 0.07, 0.01, confirmed in §3.54) were left beside the new test; no tolerance on wire
counts (RPCs), which are compared by hand in `ABSTRACTS.md`.

### 3.56 The chat replay: ShareGPT through vLLM and LMCache (added and decided 2026-10-02)

The item §3.51 left open ("a steady-state trace with a replay is what would fit `local`")
and §3.55 turned into three verdicts. Done at the user's instruction as the first task of
the session. Kits: `builder/traces/kv_cache_serving` (`replay.py`, `fit.py`, the two logs)
and `builder/traces/kv_cache_shared`; findings in `ABSTRACTS.md` §8 "Replay". ~~**The choices
below were made while building and are not yet confirmed by the user.**~~ **Decided
2026-10-02:** the user confirmed the choices, and agreed to widening the `at` rule for the
per-round draw under "Not done" (§3.57).

**What was run.** ShareGPT (the file vLLM's benchmarks use) through the server of §3.51,
five times: two untraced runs to see what the engine does, then the three traced runs that
are now the committed pairs (local disk; `fs://` on an empty store; a restarted engine on
the filled store, sent the writer's requests byte for byte). 300 requests each, 8
conversations open, context 4,096 tokens, 96 MiB of GPU KV memory, two synthetic system
prompts.

**What the dataset gives, and what it does not.**

- The file holds long conversations in parts whose ids end in the index of their first
  message, and consecutive parts overlap by one message. Joined: 50,142 conversations,
  49,699 of which alternate human and gpt from a human turn. Read as it is, the file says
  a conversation has 3.4 turns and never exceeds about 2,200 tokens; joined, 6.7 turns
  (median 3, p90 15, p99 62), and 15 % of conversations exceed 4,096 tokens.
- Lengths with the Qwen tokenizer over 26,996 turns of 4,000 conversations: user turn
  median 18, mean 81, p99 1,226; reply median 260, mean 284, p99 782. A log-normal fits
  neither (the reply's is skewed the other way), so the defaults are twenty equal shares
  of the sample, each its mean, which keeps the totals.
- **No timestamps.** How conversations interleave, and so the lag between a store and its
  load, is not in the data. No public dataset was found that has both the turns and their
  times (the Azure and BurstGPT traces have times and token counts but no conversations).
  The reuse distance therefore stays a load parameter, and the brief's **[measure]** on it
  becomes **[config: load]**.
- **No system prompts.** `sys_tokens` and `sys_pop` stay **[measure]**.

**What the runs showed.**

- **A reply must go back as the tokens the engine generated.** Replies are forced to the
  dataset's lengths (`max_tokens` and `ignore_eos`). With special tokens stripped from the
  text, as the API does by default, the next prompt was shorter than the engine's own copy
  of the conversation and the engine's cache missed from that point: 8 % of a conversation
  lost at a distance of one request. `skip_special_tokens: false` removes it (0 of 222
  returning prompts shorter than the engine's copy).
- **What the engine holds is not a threshold in requests.** vLLM frees a finished request's
  blocks tail first into one LRU queue, so a returning conversation is held from its start
  up to some token, and the shared system prompt is nearly always held. In the traced run,
  of 220 returning requests the engine held the whole conversation in 37, the system prompt
  and nothing else in 147, a part in 28, and nothing in 8. With conversations chosen at
  random (the second untraced run), the best `local` (8) was wrong by 105 chunks, summed
  request by request, of 344 loaded.
- **A distribution of distances forks conversations.** `conv @ (r − d)` lets every request
  choose its parent, so with more than one distance two requests can choose the same one.
  Fitted to the first untraced run (distances 1 to 35), the abstract issued 348 chunk
  writes onto 249 files: 99 rewrites of a name, where the real store never writes a chunk
  twice (a real branch has other hashes and other files). The defaults (log-normal, median
  40) had the same property since §9.5.
- **The loads come in bursts.** The engine's memory is shared by the open conversations:
  when they are long each is cut short and all of them load. Loads per round of eight
  requests have a standard deviation of 13.9 in the trace and 9.8 (at most 11.5 in 200
  trials) when what the engine kept is drawn independently for the same conversations.

**Choices.**

- **The load serves its open conversations in turn.** With no times in the data any
  interleaving is the load's choice, and in turn is the one the chain expresses without
  forks: one distance, every request at most one continuation. `reuse` stays a mixture
  with a `none` arm; its other arm is now `const(40)` by default, and the parameter's text
  says what a second distance does. *Against:* the lag between a store and its load is one
  value in requests where real users give a broad one; in time it still varies with the
  requests in between.
- **`keep` replaces `local`.** A positional draw per request of the tokens the engine
  still holds of a returning conversation, counted from its start; `held = min(prior,
  keep) / chunk_tokens`, and the chunks past it are loaded. It is the engine's quantity
  (memory left after the other open conversations), so it is a number of tokens and not a
  share: fitted as a share, the abstract loaded 519 to 668 chunks against the trace's 755;
  as tokens, 621 to 781. The default, `empirical(0: 91, 1_000_000: 9)`, is the old
  default's meaning (9 % of returns within `local = 8` under the old log-normal): nothing
  or everything. All three abstracts use one formula now; `kv_cache_serving` loads from
  chunk `held` on, as the shared ones already did.
- **The fit of `keep` treats a conversation held whole as a lower bound** (the engine
  would have kept at least that much): the product-limit estimate, the weight past the
  longest observation put at the context length.
- **`context`.** A conversation whose next prompt and reply would not fit starts anew
  (`replay.py` cuts the same way; 14 of 80 conversations). Without it the chain's token
  count has no bound. Default 8,192, **[config: model]**.
- **The measured values are the defaults**: the none weight 0.15 (ShareGPT's first turns
  are 0.148 of requests) and the two length tables. They are not the chat template's:
  `turn_in` in a fitted file includes its few tokens per turn, the default does not.
- **`fit.py` lives in the kit**, not in `aeiou-fit`: it reads the load generator's log and
  LMCache's, not an `strace`. Both logs are committed (the server's reduced to LMCache's
  line per request) and a test repeats the fit.
- **The synthetic traces are replaced.** The 40-request pairs of §3.51 and §3.52 fixed
  the call sequence; their fitted files name `local`, which no longer exists. Their
  numbers stay in `ABSTRACTS.md` §8 and in this file, and `chat.py` stays in the kit.
- **The reader's `keep` is the writer's.** `same_run` requires one parameter set, and the
  restarted engine sees the same requests, so it holds about what the writer's did.

**Verdicts** (`aeiou-trace compare --judge`, seed 1 against the trace, seeds 2 to 4 as the
spread; before the replay the three pairs had 6, 7, and 4 rows outside):

| pair | rows outside before | now | the rows |
|---|---|---|---|
| `kv_cache_serving` | 6 | 1 | reuse distance, read after read: 0.274 against 0.271 allowed |
| `kv_cache_shared`, writer | 7 | 2 | the same row, 0.274 against 0.256; reuse distance, write after write (a file's chunk write after its header write: in 231 of 370 another thread's call came between, the store threads' order against the dry run's) |
| `kv_cache_shared_reader` | 4 | 1 | the same row, 0.280 against 0.275 |

Still not accepted, all three: a row outside is outside (§3.55), by 0.003 to 0.018 here.
The read share, the reuse shares, the popularity, and the `stat` share are inside now.
Chunks, trace against abstract at seed 1 (and over the seeds tried): stored 370 against 382
(355 to 389); loaded by the writer 755 against 633 (621 to 781, eight seeds); loaded by the
reader 1,124 against 950 (950 to 1,111, four seeds). The engine on `fs://` stored and
loaded what the engine on local disk did, chunk for chunk, and held the same of every
request: `fit.py` gives one parameter set from either run.

**The one row.** Its distance is the bytes accessed between two reads of a chunk, and the
trace's is longer (median 112 MiB against 96) in every seed tried; seed 2 is inside, seed
1 is the lowest of the seeds in chunks loaded. The cause that was measured is the bursts
above. For the reader there is a second, which is the capture's and not the abstract's: the
restarted engine was sent the writer's replies as history, so it never held a reply it had
generated itself (0 of 220 returning requests, 37 for the writer) and loaded about one
chunk more for each conversation it otherwise held whole. A real second engine holds its
own replies, as the abstract has it.

**Not done.** ~~One `keep` draw per round of open conversations, which the bursts call for:
`kp @ (r − (r % d + 1))` is refused by the validator (the offset of an `at` must be
provably at least one, and `d` may be `none`), and widening that rule is a contract
change.~~ (Done, §3.57, contract 0.5.) The wire counts under the replay (the tables of
`ABSTRACTS.md` §8 are the synthetic load's). More than one request in flight
(`--concurrency`), a second engine on the store at the same time, a model with a longer
context, and a `keep` measured at another ratio of GPU memory to open conversations.

### 3.57 One `keep` draw per round: the `at` offset as an expression (contract 0.5, built and decided 2026-10-02)

The user agreed in §3.56 to widen rule V3 so that the per-round draw can be written. Built
the same day: contract 0.5, the three KV-cache abstracts changed under it, two of the three
pairs accepted.

**The need.** The engine's GPU memory is one state shared by its open conversations: when it
runs short, every conversation is cut and every one of them loads in the same round. `keep`
drawn per request cannot say that (§3.56: loads per round of eight requests have a standard
deviation of 13.9 in the trace, 9.8 under independent draws), and the reuse distance of the
reads was the one row outside in each KV pair. The draw a round shares is the one made at
the last request of the previous round: `kp @ (r − (r mod d + 1))`, where `d` is the one
reuse distance and the rounds are the blocks of `d` requests. Under the `cont` guard `d` is
never `none` and `r ≥ d`, so the index is never below the loop's start.

**The rule as it was, and as it is.** V3 read: a self- or forward-reference's index is `i −
e` with `e` a positive literal, or a draw or parameter whose distribution has `min ≥ 1`. Two
things were found while widening it:

- **The three validators disagreed on whom the rule binds.** The README and the builder said
  the binding being defined or one defined later; the runner and `check.py` applied it to
  *every* `let` of the same loop body, earlier ones included (`scope.forward` is the body's
  whole set of names). The runner's reading is the sound one: `x = y @ (i + 1)` with `y`
  earlier in the body and `y = x @ (i − 1)` is a cycle (`x @ i → y @ (i + 1) → x @ i`) that
  the self/forward reading lets through. Every committed AST passed the runner, so the
  README and the builder now say what the runner always enforced: **on a binding of the same
  loop body, itself included,** the index is `i − e` with `e` provably `≥ 1`. A binding of
  an enclosing body is free, as before. The runner's error text changed from "self- or
  forward-reference" to "names a binding of this loop body".
- **"Provably `≥ 1`" is now one predicate with a threshold,** `at_least(e, k)` for `k ∈ {0,
  1}`, in all three (`builder.py`, `check.py`, `validate.rs`): a literal `≥ k`; a `ref`,
  `param`, or `draw` whose distribution has `min ≥ k` (the forms already known: `uniform`'s
  `lo`, `normal` and `lognormal`'s `min`, every `empirical` value, every non-null `mixture`
  arm, `const`); and **`add` of a term `≥ k` and a term `≥ 0`.** For `k = 0` two more forms:
  **any `mod`**, because the runner's `mod` is Euclidean (`rem_euclid`: the result is in `[0,
  |b|)`, and `mod` by zero is an error, never a value), and **a loop index whose loop has no
  `from` or a `from` provably `≥ 0`** (steps are checked positive at run time, so an index
  is never below its `from`; `parallel` and `loader` indices start at 0). The scopes carry
  the set of such indices. Nothing else: `mul`, `min`, `max`, `div`, `when`, and a parameter
  with a literal default are not forms the rule knows, and a `sub` never is. Each is a
  one-line addition when an abstract needs it; none does today.

**What the proof judges, and a crash found on the way.** The document: parameter
*defaults*. That was so before, and it was a hole: `--param reuse='{"const": 0}'` passed V3
because the default's minimum is 1, and the VM then evaluated `conv @ r` while defining
`conv @ r`, with no guard but the stack (verified: `dry-run` aborted with a stack overflow).
Fixed here: `Params::new` runs the rules once more with the values in effect in place of the
defaults (`validate::check_given`) and refuses before anything is evaluated, naming each
`at`; every subcommand resolves parameters through it. The builder judges the defaults, as
it must; `check.py` judges the document.

**Contract 0.5.** The schema's `at` description says the rule; the version constant is 0.5;
every committed AST was regenerated. Unlike 0.2 to 0.4, **three fingerprints changed**, not
because of the contract but because `kv_cache_serving`, `kv_cache_shared`, and
`kv_cache_shared_reader` now use the per-round draw (the goldens in `golden.rs` are
re-recorded with the reason). The widening is backward compatible: every 0.4 document is a
valid 0.5 document after the version string.

**Against the traces** (`fit.py`'s parameters, unchanged; eight seeds for the counts, seeds
2 to 4 as the spread for the judge):

| | trace | before (seed 1; seeds) | now (seed 1; seeds) |
|---|---|---|---|
| `kv_cache_serving`, chunks loaded | 755 | 633 (621 to 781) | 596 (587 to 781) |
| loads per round of 8, sd | 13.9 | 9.8 | 13.1 (10.6 to 13.4 over four seeds) |
| `kv_cache_shared`, writer, chunks loaded | 755 | 632 (632 to 780) | 595 (595 to 781) |
| `kv_cache_shared_reader`, chunks loaded | 1,124 | 950 (950 to 1,111) | 902 (902 to 1,113) |
| reuse distance, read after read: distance / allowed | | 0.274 / 0.271, 0.274 / 0.256, 0.280 / 0.275 | 0.110 / 0.274, 0.096 / 0.286, 0.131 / 0.273 |

The chunks stored and the `stat` counts did not move (`keep` touches only the loads). The
bursts are there: the spread of loads per round is the trace's, and the spread between
seeds is wider than before (587 to 781 against 621 to 781), as a shared draw makes it. Seed
1 is a low seed in both before and after. **Verdicts:** `kv_cache_serving` and the reader
**accepted**; the writer still **not accepted** on one row, the distance from a file's header
write to its chunk write, which is the trace's two store threads interleaving and was
recorded as outside in §3.52. The three tolerance files lose their `reuse distance, read
after read` entries (the judge reports an entry recorded as outside that is within).
Eleven of the fifteen committed pairs are now accepted, the DiskANN build and the KV writer
are not, and two cannot be judged.

**Choices made here** (~~for the user to confirm~~ **decided 2026-10-02**, the user confirmed all four):

- The rule binds every `let` of the same loop body (the runner's reading, documented now),
  not only self and forward references.
- The forms added: `add`, `mod`, a nonnegative loop index; and the forms left out.
- A contract bump (0.5) for a widening of a validator rule, since a 0.4 validator refuses a
  0.5 document that uses it. The alternative, calling it 0.4 and widening in place, would
  have left committed 0.4 documents that the published 0.4 rule rejects.
- The check of the parameters in effect lives in `Params::new`, so it is one place and no
  subcommand can miss it, at the cost of walking the AST a second time (microseconds).

**Not done.** `max` and `mul` as forms. The rest of §3.56's list.

### 3.58 The `trace` node: a trace's lanes, its descriptors, and its gaps (designed, renamed from `replay`, built, and decided 2026-10-02)

The last open piece of brief item 14. The schema carried `replay {trace, sha256}` from 0.1
with the trace format deferred (schema §7), the builder emitted the node, and the runner
refused it ("the trace format is deferred"). What follows is the design as written in the
morning; the user confirmed every choice ("the rest of §3.58 is fine, build it") and it was
built the same day. **"As built"** at the end says where the build departed from the text,
one of them a rule the design was missing.

**The name (decided 2026-10-02).** The node is `trace {file, sha256}`. "Replay" had come to
mean two things: this node, which runs a captured *trace* through the runner, and the chat
replay of §3.56, which runs *requests* through a server. The user asked for a name that
says what the run is for; the repo's rule is to name the mechanism, and what the node holds
is a captured trace, executed literally, so `trace` it is: it lines up with `aeiou-trace`,
`aeiou-trace export`, and `builder/traces/`, which all mean the application's captured
calls, and "replay" now means only the chat replay. The cost is a busy word, so the text
says "the strace" for the raw capture and "the trace file" for the node's input.
`validation`, `cross-check`, and `calibration` were considered and set aside: the first two
name what is done with the run afterwards, by comparison, and the last is also what
`fit.py` does to parameters. Renamed in place in the 0.5 schema, `check.py`, the builder
(`cursor.trace(file, sha256)`), and the runner (`Node::Trace`), without a version bump: no
committed document used the node and no validator ever accepted one past the runner's
refusal, so there is no 0.5 document the rename breaks; the bump comes with the file
format.

**What it is for.** The metrics (§3.39, §3.42, §3.55) compare the abstract's op stream with
the application's statically: the same numbers from both sides, within tolerances. They say
nothing about what the storage sees in time: the concurrency of the application's threads,
the think time between its calls, the order in which its descriptors are opened and closed.
The `trace` node is the dynamic half: the application's literal call sequence, with its
lanes and its gaps, run through the same runner, the same backend, and the same
instrumentation as the abstract, on the same storage. Throughput, latencies, the host
counters, and the per-phase totals of the two runs are then comparable in a way a run of the
application itself never is, because the client side is the same program. It is bounded to
calibration: a trace is a materialized op list (the one place where "never materialize" is
the point), it is one process on one host, and it is never CLOSED (`PROJECT_BRIEF.md` §8).

**The trace file.** Written by a new subcommand, `aeiou-trace export TRACE --root DIR -o
FILE`, from the same `strace -f -ttt -T` an `aeiou-trace metrics` reads, through the same
`Tracer`: the descriptor tables per task, the inheritance across `clone` and `fork`, the
`cwd`, the paths under `--root`, the positions kept per open file. The file is JSON Lines,
since a trace of an epoch over 100,000 files is 400,000 ops and a single document of that
size is read all at once for no reason. The first line is the header, every other line one
op, in the trace's issue order:

```
{"aeiou_trace": 1, "source": "strace", "root": "data", "lanes": 9, "opens": 100012,
 "creates": ["ckpt/step-100/__0_0.distcp", …], "notes": {"calls_not_exported": {"sendfile": 2}, "shared_positions_resolved": 14}}
{"lane": 0, "t": 0, "dur": 41200, "op": "open", "fd": 0, "path": "train/n01440764/x.JPEG", "flags": ["RDONLY","CLOEXEC"], "ret": 0}
{"lane": 0, "t": 61900, "dur": 2100, "op": "fstat", "fd": 0, "ret": 0}
{"lane": 3, "t": 70250, "dur": 912300, "op": "read", "fd": 0, "len": 131072, "ret": 109383}
{"lane": 3, "t": 990100, "dur": 18000, "op": "read", "fd": 0, "len": 131072, "ret": 0}
{"lane": 0, "t": 1011000, "dur": 3300, "op": "close", "fd": 0, "ret": 0}
{"lane": 2, "t": 1200000, "dur": 802000, "op": "submit", "ops": [{"op": "read", "fd": 7, "offset": 4096, "len": 4096, "ret": 4096}, …]}
```

- **A lane is a traced task** (a thread or a process under `strace -f`): the trace's
  "context" of §3.42. A lane that issues no op under the root is not exported. Lanes are
  numbered in order of first op. `t` is nanoseconds from the trace's first exported op,
  `dur` the call's duration (`-T`); both are facts of the trace, kept so that the file can
  be re-measured, and the runner derives the gaps from them (below).
- **`fd` is an open id**, not a descriptor number: the ordinal of the `open` in the file.
  An op names the open that produced its descriptor, through every `dup`, `fcntl(F_DUPFD)`,
  inheritance across a `fork`, and shared table across a `clone(CLONE_FILES)` the `Tracer`
  resolved. A lane may use an open another lane made (the main thread opens, the workers
  read): that is the dependency a trace run has to keep, and it is in the file.
- **The op vocabulary is the schema's**, with the same names and fields (`open`, `close`,
  `read`, `write`, `lseek` with `whence`, `fstat`, `stat`, `fsync`, `fdatasync`, `unlink`,
  `mkdir`, `rmdir`, `rename`, `readdir`, `ioctl`, `fadvise`, `ftruncate`, `fallocate`), the
  paths relative to `--root`, and `ret` the traced result: a byte count, `0`, or an errno
  name, which is what the runner checks against (a traced `ENOENT` on a `stat` is expected,
  any other result is a run failure, as `expect` works today). A `read` without `offset` is
  the traced `read`; with one it is the traced `pread`. `lseek` stays an op (the abstract is
  the application's call stream, §3.44). Three translations, each counted in the header's
  notes: **`io_submit` becomes `submit`**, a group of positioned ops issued together and
  reaped before the lane's next line, which is what the libaio backend does with a
  `parallel` today and what the metrics count as a fan-out; **an `mmap` of a file range
  becomes a `read` of that range** with the mapping's offset and length, because the page
  faults inside it are not in an `strace` (the same limit the metrics have, §3.47), so the
  touch pattern is the backend's (`--mmap-mode`), not the application's; and **a call the
  schema has no op for** (`sendfile`, `splice`, `copy_file_range`, `getxattr`) is dropped
  and counted by name. `aeiou-trace metrics` of the strace and of the exported file must
  agree, which is a test the exporter carries.
- **A shared position is resolved at export.** A `read` or `write` without an offset on a
  descriptor that more than one lane uses (an inherited open file description; the DataLoader
  workers after a `fork`, a thread pool over one `fd`) reads from a position the lanes'
  interleaving decides, and the run cannot reproduce that interleaving without
  serializing the lanes. The exporter knows the offset each such call actually used (the
  `Tracer` tracks it) and writes the call **positioned**, at that offset, counting the
  rewrites in the header (`shared_positions_resolved`). A lane-local sequential `read`
  stays a `read`. This changes the call (`read` to `pread`) in exactly the cases where the
  traced call's meaning depended on another thread; the count says how often.
- **`creates`** lists the paths the trace brought into being under the root (an `open` with
  `O_CREAT` of a path not seen before, a `mkdir`, the destination of a `rename`), the
  run's output set; see the checks below.

**The runner.** `Node::Trace` loads the file named by `file` (a path relative to the AST
document's directory; every host of a multi-host run reads its own copy), checks its
`sha256` against the node's and refuses on a mismatch before the gate, and executes it as
one fork of `lanes` sub-actors, which is what the sub-actor pool (§3.36) already runs: a
lane is a sub-actor walking its lines in order, one op in flight (a `submit` is the lane's
one op until its last member completes, as a `parallel` under libaio is). Three things are
new in the VM:

- **An open table per trace instance**, open id → the lane's `OpenFile`, filled by whichever
  lane executes the `open` line, and **a wait**: a lane whose next line names an open id
  not yet in the table parks until it is (the VM is already parked per actor, §3.29; this is
  one more reason to park, resumed when the `open` completes), and the `close` of a shared
  open is executed by the lane the trace closed it in, after the table shows every other
  lane's last use of that id has completed (~~the exporter writes the count of users per open
  into the `open` line as `users`, so the close knows how many to wait for~~ the runner
  counts the uses and the closes of each id from the file itself at load, as built). ~~No
  lane waits for anything else: the only cross-lane order a trace run keeps is the one its
  descriptors force.~~ **Wrong, found by the first write-then-read trace: see "As built".**
  Everything else about the lanes' relative timing is the storage's and the gaps'.
- **The gaps.** Before each line a lane emits `compute` of `t − (t_prev + dur_prev)` of its
  own previous line (for its first line, `t` itself: a worker that started late in the trace
  starts late in the run), scaled by `--time-scale` like every `compute` and recorded
  unscaled. `--time-scale 1` replays the application's think time as `strace` saw it, which
  is an upper bound (`strace` slows the application; the kit says by how much for each
  trace); `--time-scale 0` is the storage-bound run, every lane issuing as fast as the
  storage answers, the usual mode. The gaps are the per-step compute of §3.5's stall
  model, so the stall and busy fractions of a trace run mean what they mean elsewhere.
- **Writes carry the runner's payload** (§3.23: a function of the seed, the path's hash in
  place of a file id, and the offset, at the run's `--write-compress`), since a trace has no
  data. Reads are checked structurally against `ret`, as every op is.

**The fingerprint and the dry run.** Every op of a trace hashes like any op (`op_hash`: kind,
actor, indices, offset, length, path), with the lane index and the line's ordinal in the
lane as its indices, so the fingerprint of a trace node is a function of the file, the sum is
order-independent as always, `--expect-fingerprint` works, and two hosts holding different
files are caught. `aeiou dry-run` walks the file **in its line order**, not lane by lane:
each op tagged with its lane as the metrics' context. So `dry-run --metrics` of an AST that
is one `trace` node computes the trace's own metrics, and must equal `aeiou-trace metrics`
of the strace it was exported from, row for row, the reuse distance included (both are the
trace in completion order as one instance). That equality is the test that the exporter,
the loader, and the two metric implementations agree, and it is the first check of any new
trace. The report (§3.41) carries `trace: {file, sha256, lanes, ops}` on the run and the
comparison policy says such a run is never CLOSED.

**Checks before the gate** (§3.40's place). Every path the trace reads without creating must
exist under `--root` with a size of at least the largest offset plus length the trace read
from it (so the `ret` checks can hold), and every path in `creates` must not exist, both
checked in trace order (a file created and then read is fine). `--clean-namespaces` removes
the `creates` set and nothing else, since a trace's root is the application's own data,
which the runner never empties: the user put it there (the kit's `mkcorpus.py`, or a copy
of what the application ran over). V12 to V15 are untouched: a trace node declares no dataset
and no namespace. **V16:** a `trace` node in an actor template whose instance count can
exceed one is refused when the trace has a nonempty `creates` (G copies would write the same
files); with no creates it is allowed, and G instances replay the trace against the same
files, which is a load test with the application's sequence, not a model of G ranks (whose
shards differ), and the report says so.

**Not in a trace run.** `io_uring` submissions (not in an `strace`; a trace of such an
application is its control calls only, and the exporter says so in the notes). The touch
order inside a mapping (above). The GPU side of anything. A trace longer than memory: the
runner holds the file's lines as parsed ops (about 64 bytes each; a million lines is 64 MiB),
which calibration never reaches; there is no cap and no streaming.

**Choices made here (~~for the user to confirm~~ confirmed 2026-10-03):**

- JSON Lines for the trace file, the first line a header, one op per line in issue order;
  the file is not the AST and not on the AST contract, but it is JSON.
- A lane is a traced task; opens are referenced by id so lanes share descriptors; the only
  cross-lane waits are on opens and on the close of a shared open.
- Shared positions resolved to positioned calls at export, counted; lane-local sequential
  reads kept as traced.
- `io_submit` as a `submit` group; `mmap` as one `read` of the range; unknown calls dropped
  and counted.
- Gaps from the trace's timestamps, as `compute` under `--time-scale`, the first gap being
  the lane's start.
- The dry run walks the file in line order so that `dry-run --metrics` equals `aeiou-trace
  metrics` of the source; this is the acceptance test of an exported trace.
- `file` is relative to the AST's directory; the pre-gate existence checks; `creates` as
  the only thing `--clean-namespaces` touches; V16.
- ~~The name `replay` kept, with the two meanings told apart in the text.~~ Renamed `trace`
  (decided 2026-10-02, above).

~~**Not done.** Any of it.~~ Built 2026-10-02, in the order listed: the exporter and its
equality test, the loader and the dry run, the lanes, the checks, the report field, the
documents (`runner/REFERENCE.md` §13, `builder/REFERENCE.md` §7, schema README).

**As built (2026-10-02).**

- **Path order, the rule the design lacked.** The design kept one cross-lane order, the
  descriptors'. The first write-then-read trace (the runner itself under `strace` on
  `kv_cache_serving`) failed at once: a lane opened a chunk file that another lane had not
  created yet, since nothing ordered an open by *path* after the `creat` of that path by
  another lane. The rule added: on a path the trace changes, a changing op (an open with
  `CREAT` or `TRUNC`, a write, a truncate, an allocate, a sync or close of a writable open,
  a rename on both its paths, an unlink, a mkdir, a rmdir) waits for every earlier op on
  that path, and a reading op waits for every earlier changing op on it; reads never wait
  for reads, and a path the trace never changes carries no wait. It is the usual
  read-after-write, write-after-write, write-after-read dependency, computed per op at
  load (`Dep {path, k, mk, mutating}`: the op's index among the path's ops and the changing
  ops before it) and kept at run time as two counters per path under one lock. With it the
  KV trace runs every time with its dry run's fingerprint. Both orders are the trace's own:
  every wait is for an event earlier in the trace, so the earliest unfinished op never
  waits and the lanes cannot deadlock.
- **The runner, not the VM, runs the lanes.** The node is an `Event::Trace` the VM yields
  with the file and the position; the sink does the rest. The thread-per-actor driver gives
  each lane a child `Runner` on a scoped thread, and a `submit` group one thread per member
  (a fan-out under a blocking backend). ~~The event-loop drivers (`io_uring`, `libaio`)
  refuse a trace before the gate; they are the next step, and until then a trace of a
  libaio application runs under `sync` with its groups fanned out on threads.~~ **Built
  2026-10-03:** the event loop (`uring.rs`, both engines) runs a lane as a task whose
  program is a lane of the file instead of a VM (`Prog::Lane`), one op in flight, joined by
  the task that reached the node as a `parallel` parent joins its sub-actors; a `submit`
  group puts its members in flight together as member tasks, which the loop submits in one
  `io_uring_enter` or `io_submit`, the application's own call. The open table and the path
  order are loop-local and lock-free there (every lane of an instance is on its loop), and
  a lane that must wait parks on its instance and is woken by every change to its tables
  (all waiters, every time: simple, and a trace's lanes are tens, not thousands). The
  three fixture runs of `runner/aeiou/tests/trace.rs` and the three traces of the runner in
  `builder/tests/test_trace.py` give the dry run's fingerprint under `io_uring` and `libaio`
  as under the blocking backends.
- **Three equalities, checked on traces of the runner itself** (`train_small_files`,
  `kv_cache_serving`, `vdb_search_diskann`, in `builder/tests/test_trace.py`):
  `aeiou-trace metrics` of the strace, the same of the exported file, and `aeiou dry-run
  --metrics` of an abstract that is one `trace` node give the same document, row for row,
  with one exception: under libaio the strace's metrics count an `io_submit` member when it
  is reaped and the file places the group at submission (where the runner issues it), so
  the reuse-distance histogram differs by the reordering within a round. Then the trace
  runs against the same corpus with its dry run's fingerprint, under `sync`, `sync-direct`,
  `posix-aio`, and `mmap`.
- **Smaller departures.** The exporter writes the ops on the lane the call was made on and
  the header's `creates`; the runner recomputes uses, closes, read extents, inputs, and the
  peak of open descriptors from the lines (nothing in the header is trusted for
  correctness). A write's count is checked as its length (a short write in a trace would
  not reproduce anyway); `readdir` is one line per listing with the entries unchecked
  (the trace does not say how many). Waiting time in the two orders is not recorded. V16 is
  checked in `aeiou run` with the resolved counts (`count` is an expression; `--gpus`
  decides it), not by the validator. The limits of §3.40 count a trace's lanes as threads
  and its peak of open ids as files. `aeiou-trace metrics` reads an exported file as well
  as an strace. The runner's dry run reports the gaps as its compute total.

**Choices made while building (decided with the rest, 2026-10-02):** path order as the
second cross-lane rule, with reads free among themselves; the event-loop backends deferred;
the uses and closes counted by the runner from the file; the strace-side placement
difference under libaio documented rather than changed (the metrics' definition of §3.42,
completion order, stands). **Choices made building the event-loop lanes (2026-10-03,
confirmed by the user the same day):** a lane is a task on the loop, not a thread, and the
report and the limits count it as such; a `submit` group's members are member tasks with
their own actor state, merged into the lane when they end; every lane parked on an
instance is woken on every change to its tables rather than keyed by what it waits for.

### 3.59 The agentic load: the AgentX corpus, and what fitting it changed in the KV abstracts (added 2026-10-02, decided 2026-10-03)

The user had access to a set of LMCache captures taken by the SNIA AIWD TWG (private, not in
this repo, and nothing here is derived from them) and asked whether they would help validate
the KV abstracts. Their value was that they pointed at a load the kit did not have: agentic
coding sessions replayed through vLLM and LMCache, where a request's prefix is tens to
hundreds of thousands of tokens and every turn appends a few thousand. The load they replay
is public: SemiAnalysis's InferenceX AgentX corpus (`semianalysisai/cc-traces-weka-062126`
on HuggingFace, Apache-2.0; 393 Claude Code sessions, 98,827 requests, 21.6 G prompt
tokens), with what §3.56 said no public dataset had: a timestamp and a think time per
request, and, in place of text, the prompt's 64-token KV blocks as hash ids, so the reuse
structure of every request is in the file. The user's decision: fit from the corpus, keep
the captures out of the repo. Kit: `builder/traces/kv_cache_serving/agentx.py` (`fit`,
`reference`), `fitted.agentx.params.json`, `agentx.reference.json`, and `replay_agentx.py`
for the GPU box. ~~**The choices below were made while building and are not yet confirmed by
the user.**~~ **Confirmed by the user 2026-10-03.**

**Is this the end-to-end workflow the user asked about?** Mostly. The corpus gives the
distributions (step 2 of the user's description) and a chunk-level reference to validate the
abstract against here, with no GPU: `agentx.py reference` walks the corpus with a store that
never evicts and counts, per request, the whole 256-token chunks the store would hit and
those it would store; `aeiou dry-run` at the fitted parameters gives the abstract's. What it
cannot give is the call sequence (fixed by the kit's own `strace`s, §3.51, and unchanged
here), `keep` (the engine's memory against its open sessions; the proxy that recorded the
corpus saw no engine), and a judged `aeiou-trace compare`, which needs the corpus replayed
through vLLM and LMCache under `strace` on the GPU box: `replay_agentx.py`, written and not
yet run.

**What the corpus showed, in order of what it changed.**

1. **A conversation's length is a property of the conversation.** The abstracts ended a
   conversation by a per-request draw (`reuse`'s none arm), so chain lengths were geometric.
   The corpus has 8,138 chains of one request (side calls of a few hundred tokens between an
   agent's turns) beside sessions of hundreds of turns (median 65 main-agent turns, longest
   1,190), and the long sessions hold nearly all the prefix hits: at the fitted lengths with
   the none arm the abstract read a seventh of the corpus's chunks. **`turns`**, a draw at a
   conversation's start of the requests it will have (0 for no limit, the default: the
   ShareGPT fits are unchanged), with `reuse`'s none arm at 0 in the agentic file.
2. **Writes come per prefill step.** vLLM prefills `max_num_batched_tokens` tokens per
   step (8,192 for an API server on a large GPU, 2,048 otherwise, from its source) and
   LMCache stores the chunks each step completes, so a 117k-token prompt's writes are 58
   bursts of eight chunks over the prefill. At 4k context the kit's traces could not show
   it. **`prefill_step`**: the chunk writes sit in a loop over steps, each after its
   `compute`. The op multiset is the same; the writes carry one more loop index, so the two
   writers' fingerprints changed.
3. **An agent edits its context.** 3.7 % of continuing requests keep a shorter prefix of the
   chain's last prompt than the whole (a rewrite of the end of the context), dropping a mean
   of 32k tokens and storing what they add past the kept prefix: a quarter of all chunk
   writes. The chain could not shrink. **`trim`**, a draw of the tokens dropped before a
   request (`prior = max(ptoks + out − trim, 0)`), and the store's hit for the request is
   `min(stored @ (r − d), prior div chunk_tokens)`. *The infidelity:* the chunks past the
   kept prefix are written again under their names (a `trunc` of an existing file), where
   the real store writes new files under new hashes; a per-chunk generation would be a
   per-file structure (`GRAMMAR_OPTIONS.md` §5.3).
4. **The client waits.** `think_time` per request, median 4.8 s, p90 121 s, p99 51 min
   (users leave). **`think`**, a draw slept before each request, default 0. Its tail makes
   a slot idle for an hour at a time; the fit does not cap it, and a parameter file may.
5. **The reply is not all kept.** `out` counts generated tokens, and in 22 % of turns the
   next prompt grew by less than the reply (thinking is not retained), so `turn_in`, fitted
   as growth less the reply and at least 0, makes the chain grow by 108 % of the corpus's.
   Left as a stated bias; a `turn_out` for the decode and another for the chain is a
   change not made.
6. **Sub-agents are chains of their own.** 43 % of requests are sub-agents', run while the
   parent waits; their first prompt shares a median of 27,648 tokens with what the session
   stored (the system prompt and the tool definitions), which is the abstract's `sysp`
   dataset, so `sys_tokens` is that median and `sys_prompts` the session count. The
   parent's think time spans the sub-agents' run.
7. **The hash of a partial block changes when it fills.** `in` is a count of 64-token
   blocks; the last one is partial as often as not and its id differs once the prompt has
   grown. Read naively, 6.5 % of turns "rewrote" one block. A request keeping all but the
   last block extends its chain, and the block is left out of the chunk accounting.
8. **The VM's cache of `x @ i` values emptied itself when full** (65,536 entries), after
   which a chain's history was recomputed recursively from its start: fine for chains of
   40 turns, a stack overflow for 1,190. It now sheds the half with the smaller indices
   (`vm.rs`), which keeps the recent entries that an `at` chain reaches.

**Against the corpus** (`agentx.reference.json` against `aeiou dry-run` at
`fitted.agentx.params.json`, one slot, the corpus's 98,827 requests; reads at the default
`keep`, 91 % of returns holding nothing, so the abstract's loads are 0.91 of its hits):

| per request | corpus | abstract, seeds 1 and 2 |
|---|---|---|
| chunks stored | ~~14.41~~ 14.46 | ~~15.6, 15.4 (the 108 % above)~~ 15.5, 15.0 (2026-10-15) |
| chunks hit | 840 | ~~784, 850 (loads 713, 773 at 0.91)~~ 784, 808 (loads 713, 735 at 0.91; 2026-10-15) |
| prompt tokens, mean | 218,922 | — |

The hit count depends on how finely `turns` resolves its tail (`--turn-bins`, 100 by
default): with 50 shares the abstract hits 890 to 920, with 200 it hits 715. The cause is
the context: the abstract draws every turn's growth independently of the session's length,
and the corpus's long sessions grow more slowly per turn (and are compacted), so the
abstract's longest conversations reach the 990k-token cap and are cut. A growth that
depends on the chain's length is not drawn here. The fit at 100 shares is within 10 % on
hits and 9 % on stores; `tests/test_trace.py` repeats it on 10,000 requests.

**Chunks keyed by model (decided by the user 2026-10-15, RC1 review).** The chains were
keyed by model and the store was not: one store per session served every model, so a
request counted hits on another model's chunks, and a sub-agent's first prompt found its
prefix in another model's (`agentx.py` `walk` and `shared_prefix`). LMCache keys a chunk by
its model (another model's tokens are other tokens), and 187 of the 393 sessions mix models
(Opus, Haiku, Sonnet, and others). The store, and the shared prefix that a request must keep
more of to join a chain, are now per session and model. The corpus fit and the reference,
refitted: 10,079 chains (10,391), 7,762 of one request (8,138), the longest 1,519 requests
(1,190); rewrites 3.86 % of continuations (3.74 %), the chain growing by 107 % of the
corpus's (108 %); `kind` 429, 722, 8,928 (432, 870, 9,089); `sub_tokens` 29,696 unchanged
(1,516 sub-agent groups found a prefix, 1,519); `first_in` and `turn_in` shift by a few
percent in their upper shares. The reference stores 14.46 chunks a request (14.41) and hits
840.08 (840.14); its sub-agent prefix median, 27,648 over 1,697 groups since 2026-10-02,
was the walk of that day (it counted groups that found nothing) and is now the fit's,
29,696. The abstract at the new fit, all 98,827 requests, seeds 1 and 2: 15.5 and 15.0
chunks stored a request (15.3, 15.1 at the old fit), loads 713 and 735 (717, 769). The
replays' fits are unchanged, by design: a replay serves one model, so the store a replay's
requests met is one per session, the server's, as `holds` already took it.

**Choices (~~for the user to confirm~~ decided 2026-10-03).**

- Four new parameters on the three KV abstracts (`turns`, `prefill_step`, `trim`, `think`)
  with defaults that leave the ShareGPT fits' op counts as they were (the fingerprints
  changed: the writes under the step loop's index, and the draws before the conversation
  and system-prompt picks moved those picks' sites, so a seed draws other prompts where the
  system prompts vary in size; `golden.rs` re-recorded with the reason).
- `fit.py` (ShareGPT) is unchanged: it keeps the none arm, which is the right model for a
  dataset whose conversations are short and bounded by the context. `agentx.py` is a second
  fit, not a mode of the first: it reads a corpus, not a replay's logs.
- The rewrite as a `trunc` of existing names (item 3), rather than new names.
- `turns` resolved in 100 shares against 20 for the lengths, and the dependence of the hit
  count on that choice stated rather than removed.
- `agentx.reference.json` is committed (the corpus's accounting, 2 KB); the corpus is not
  (1.8 GB, public, fetched by the command in the kit's docstring).

**Not done.** The replay on the GPU box (`replay_agentx.py`: `keep` for an agentic load at a
chosen KV memory, the `strace` for the judge, and the store's view of the prefill-step
bursts); a growth that depends on the conversation's length; the chunk-level reference's
loads (it has the hits; the loads need `keep`); tensor-parallel sharding (brief item 20:
a TP group writes one chunk as one shard file per rank under one key, and the runner's
positional draws keyed on the GPU id would make the ranks independent engines); the
`O_DIRECT` option of the local-disk backend, which from LMCache's source is the same one
write and one read per chunk through `os.open` with `O_DIRECT`, falling back to buffered
I/O when the chunk is not a multiple of the file system's block size (brief item 21).

### 3.60 The option layers: CLI > environment > config file > default, with provenance (designed, built, and decided 2026-10-04)

**The review.** With `aeiou run` at about thirty flags, the user asked for a UX pass over the
command-line structure and an opinion on environment variables and a config file for the
options that change rarely, under a strict precedence: the command line, then the
environment, then the file, then the compiled default.

**The structure as found.** Sound: one binary, four subcommands in lifecycle order, the
abstract as the one positional, long flags only, prefix-grouped families (`expect-`,
`metrics-`, `report-`, `mmap-`), verb-phrased booleans, and knobs of one backend refused
under another. Irregularities, all fixed the same day: the flat thirty-flag help (grouped
with headings: Workload, Backend, io_uring, libaio, mmap, Several hosts, Checks, Report);
`--params FILE` one letter from `--param NAME=VALUE` with a different kind of value
(`--params-file`); the trace tool spelling the runner's `--metrics-block`/`--metrics-sample`
as `--block`/`--sample` although the two outputs are compared with each other (the runner's
spelling); `-o/--output` in `aeiou-trace` against `-o/--out` in the other helpers (`--out`);
`--threads` accepted and ignored under the thread-per-actor backends (refused, as the ring
knobs are); the shared core hand-copied into `datagen` instead of flattened (one `ShapeArgs`
for the abstract, `--param`, `--params-file`, with `--gpus` declared per subcommand and
`--seed` only where it means something); `--ranks` without a doc comment.

**The layers (decided).** The precedence the user proposed is the conventional one (git,
Cargo, Docker, kubectl, pip) and the right one here: the environment above the file because
the environment is per process and per host, which is what a launcher sets, while a file is
per site. It simplifies the usage model for exactly one bucket of flags, so the flags were
sorted by how often they change, and the sorting became a rule:

- **Fixed (the command line's alone):** the workload's identity (the abstract, `--gpus`,
  `--seed`, `--param`, `--params-file`, `--io-backend`), the checks that pin it
  (`--expect-fingerprint`, `--expect-dataset-id`), the unsafe overrides
  (`--clean-namespaces`, `--ignore-limits`), and `datagen`'s payload (`--dedupe`,
  `--compress`, `--dataset`). The environment and the file are refused when they name one.
  In one sentence: nothing the fingerprint, a dataset id, or a safety check depends on may
  come from a layer the command line does not show. This matches the builder, whose hermetic
  harness scrubs the environment so that it can never shape an AST (§3.3): the layers affect
  how a run is executed, never what the workload is.
- **Layered:** site and host tuning (`--root`, `--threads`, `--buffer-mib`,
  `--write-compress`, `--time-scale`, the io_uring, libaio, and mmap knobs, `--max-gap`,
  `--require-cold`, `--drop-caches`, `--report-json`, `--report-takes`) and the multi-host
  wiring (`--rank`, `--ranks`, `--coordinator`, `--rank-rotate`). `AEIOU_RANK` set by a
  launcher from its own rank variable is the strongest case for the environment layer.

Three conditions were set and accepted, and a fourth replaced: **no implicit discovery** (the
file is named by `--config` or `AEIOU_CONFIG`, never found in the working directory, home, or
an XDG path, because two hosts with different hidden files is the classic way a benchmark
site gets unexplainable results); **the report records the provenance** (`layers`: the
file's path and sha256, the variables that contributed, every option's value and source);
**every layered boolean has a negation** (~~built as `--flag=false` rather than a `--no-flag`
twin: one flag per option in the help, the form mirrors the file's `flag = false`, and `=`
is required so a bare value is never taken for the abstract~~ built first as `--flag=false`;
replaced the same day, at the user's preference, by the `--no-flag` twin shown in the help as
one row `--[no-]flag`: the notation git's documentation uses, so a Linux audience reads it at
once, one row per boolean, the regularity visible, and `--no-x` reads as English in a shell
history where `--x=false` reads as a config value that wandered onto the command line. The
UX assessment extended it to every boolean, the fixed ones included, since a convention with
exceptions has to be checked and one without is learned once; dropped the `=false` spelling,
one way to do a thing; kept `true`/`false` as the file's and the environment's form, with
`no-x` keys and `AEIOU_NO_X` refused so the double negative cannot be written; and stated the
notation once at the foot of the help, with the brackets typed literally explained rather
than parsed. Clap renders it from a long name spelled `[no-]x` with `x` as alias and a hidden
twin, so alignment is clap's own; a test pins it); and, instead of a
`--show-config`, **every invocation prints the block** of options with their sources (the
user's choice: the information is wanted every time, not on request). The merge rule for
repeated values is *replace*, so a command-line value can always displace a file value; as
built no layered option is a list, so the rule is trivially met. Workload parameters stay out
of the layers entirely: they have their own layered file format with pinning (§3.27), and
nesting one layering scheme inside another is where users lose track. The honest summary
given and accepted: the keystroke savings are modest, since operators write wrapper scripts
anyway; the value of a built-in layer is that the report can then say where every option
came from, which a wrapper never does.

**TOML (decided).** The file is TOML, keyed by subcommand, keys spelled as the long flags,
because a site file needs comments and JSON has none. JSON stays the only format on the
contract; the config file is not on the contract, it is a convenience for the operator, and
this is the one conscious departure (a `toml` crate dependency, parse only).

**Two proposals declined, and what replaced them.** The user raised the concern that the
effective options of the non-rank-0 processes might not match the invocation's, and proposed
(a) a leading positional "client" argument that honors only the command line, or (b)
clients taking their argument set from a message of rank 0. Both were assessed against what
the coordinator already does: `Hello` carried a hash over the identity (the abstract, seed,
G, parameters, dataset ids, backend, rotation, time scale, write compression), so a host
whose identity differs was already refused before any I/O; what the hash did not cover was
exactly the layered bucket. (a) was declined because it creates the mismatch it wants to
prevent: rank 0 would run with all four layers and the clients with one, and `aeiou-launch`
forwards one command line to every host, so the clients would systematically lack what rank
0 got from its file; and a leading positional that changes how the rest of the line is
parsed is the irregularity just removed elsewhere. (b) was declined because the goal is
already met for the identity, and for the rest several options legitimately differ per host
(`--rank` by definition, `--threads` with the cores, the ring knobs with the kernel,
`--root` as a mount point), so pushing rank 0's values would break those hosts or need a
per-rank override mechanism; a client assigned its rank by connection order would also make
host placement timing-dependent (the fingerprint survives, the manifest's host-to-GPU map
does not); and it breaks the stated symmetry that rank is only a host index and every host
runs the same binary with the same arguments. What was built instead, all accepted: `Hello`
carries the identity document and the host's options block; the identity refusal names the
differing field (`seed 2 differs from rank 0's 1`, `params.files …`) instead of two hash
prefixes; rank 0 records every host's block in its report (`hosts`) and prints, after the
gate, the options whose values differ between hosts (only those, so a few lines on a
healthy run: rank, and whatever is each host's own). A strictness check that refuses
differing layered options was considered and not built: the table and the report give the
operator the facts, and a WG rule can decide later whether differing thread counts are a
comparability problem. The most likely way a layer goes wrong across hosts is noted in the
README: ssh's non-interactive shell does not read the profile the operator's terminal did.

**As built.** `runner/aeiou/src/options.rs` (`Layers`: `fixed`, `layered`, `flag`, `finish`,
`print`, `json`, `differences`; `ALL_OPTIONS` kept equal to the clap definitions by a test),
`main.rs` (a `RunOptions` resolved once, the block after the `abstract` line in every
subcommand, `--config` as a global flag), `coord.rs` (`Hello { identity, layers }`,
`identity_differences`, `Server::hosts`), `report.rs` unchanged in shape (`layers` and
`hosts` are header keys), `builder/aeiou/options.py` and `aeiou-datagen` for the Python
writer (`root` and `threads` layered; the same block), CI's checker diff skipping the block.
`runner/REFERENCE.md` §14 describes it; `PROJECT_BRIEF.md` §8 records the `AEIOU_` prefix in
the naming convention. Later the same day, from the user's testing of the help: the name's
expansion "Author Execute I/O" as the `about` line (recorded in the brief's naming
convention), help text wrapped by clap at the terminal's width up to 120 columns, `--root`
shown as required in the usage of `run`, `datagen`, and `aeiou-datagen` with its help and
its missing-value message naming the three layers it may come from, and an audit of every
error message of every tool: all flag names current after the renames; every refusal or
cross-check of a layered option made where the resolver is in hand now says which layer set
the value (`Layers::from`), while the checks made during the run name the flag only, its
source being in the block above.

### 3.61 Usage errors: one frame for the suite, every missing argument at once (designed, built, and decided 2026-10-04)

**The observation.** With the option layers in (§3.60) the user tried the tools as a new
user would and read the transcript back: `aeiou datagen` with nothing said only
`<ABSTRACT_PATH>` was missing, in clap's voice with a usage line and a `--help` pointer;
with the abstract given it said `--root DIR is required`, in the runner's own voice, with
neither; a builder script passed as the abstract produced `trailing characters at line 1
column 3`, a JSON parser's complaint with no word about what was expected. Two checkers
decided what was required, each stopped at the first failure, each in its own format, and
the Python tools had the same split with argparse. The user's requirement: every tool of the
suite, Rust or Python, lists every argument the command still lacks, at once, on every
attempt, and the whole suite reads as one tool, for a user who has never seen any of them.

**Decided.** One required-argument check per command, after the layers resolve, over a list
the command builds (`usage::Missing`); nothing is declared required to clap or argparse any
more, so the parsers cannot speak first. One frame for every usage error, the parser's own
included: the command as typed, the message, the usage line, the pointer to the help naming
the command, exit status 2; a failure of the work is `command: message`, exit 1
(`runner/REFERENCE.md` §15). The wording is the parsers' own (`the following required
arguments were not provided:`), because it is what every other tool says and because clap's
and argparse's own errors then read as the same voice once reframed. The Python tools get
`usage.Parser`, argparse dressed in clap's look (`Usage:`, `Arguments:`/`Options:`,
`<METAVAR>`, `-o, --out <FILE>`, help and version last, description above the usage), so
`--help` reads the same across the suite too; `aeiou-trace`'s `SystemExit` strings and the
tools' `FAIL message` lines became `BuildError`s named by the command. A requirement that
holds only for some options or inputs (`--coordinator` with several ranks, `--report-json`
with `--report-takes`, `--sqpoll` with `--sqpoll-shared`, `aeiou-trace metrics --root` for an
strace) joins the same list with the reason and, when a lower layer set the condition, the
layer. A path given for the abstract that is not one is a usage error that says what an
abstract is and which tool writes one; an abstract that is JSON and invalid is a failure of
the work. `aeiou-params npz` exits 1 when the archives differ (was 2, §3.27), so that 2
means a usage error everywhere. `aeiou-launch` prints the frame by hand.

**Considered and not done.** Declaring `--root` required to clap and reading the lower layers
in a value parser: clap would then list it with the positional, but the config layer needs
the file, which needs `--config`, which is another argument, and the provenance the block
prints would be lost; the list after the layers is the general mechanism, and it covers the
conditional requirements the parsers cannot know about. Keeping the runner's `aeiou: message`
voice for everything, the parsers reframed into it: the user wanted one look, and the
parsers' voice is the one the rest of the world speaks. Making every parser message identical
between clap and argparse (an unknown flag is `unexpected argument '--x' found` in one and
`unrecognized arguments: --x` in the other): the frame, the lists, the exit codes, and the
hints are identical and tested so (`tests/usage.rs`, `builder/tests/test_usage.py`, the
latter comparing `aeiou datagen` and `aeiou-datagen` byte for byte when the binary is
built); the parsers' own one-line wordings differ in a few words and were left alone.

### 3.62 Datagen across hosts, `O_DIRECT`, and prefix-stable dedupe (designed, built, and decided 2026-10-04)

**The requirement.** The reference corpus must be created as fast as the storage system can
take it, and one host cannot saturate a modern system; submitters will use the same 50 hosts
for datagen as for the runs, and 200 or more must work. `aeiou datagen` therefore runs across
hosts through the same coordinator and launcher as `aeiou run`.

**What the user's framing changed.** Two parts of the first framing were corrected in the
design discussion: the Feistel permutation is not what keeps hosts apart (datagen walks a
dense id range, and the contiguous split `run` already uses for GPU ids applies to file ids;
the permutation is read order), and `--gpus` on `datagen` cannot be the parallelism knob,
because it is the instance count the dataset definition may reference and must equal the
run's or the manifest will not match. Per-host parallelism is `--threads`; across hosts it
is `--ranks`. Several processes per host buy nothing over threads, because the ceilings are
per directory (the VFS lock) and per mount (the NFS session), not per process.

**Decided.**

- *Slices.* Rank `r` writes the files of every `files` dataset whose ids fall in
  `gpu_range(files, R, r, 0)`, a contiguous slice, so under `{id div N}` a host's files sit
  in its own directories and two hosts share at most a boundary directory. Within a rank, ids
  go to threads in directory-aligned runs: Linux takes the parent's `i_rwsem` exclusively for
  every create and NFS issues the `OPEN` under it, so threads creating in one directory
  serialize at one create per round trip. A flat pattern is bounded that way on the server
  whatever the host count; the docs say so.
- *Regions files belong to one rank.* The user's observation: several NFSv4.2 clients
  writing offsets of one file serialize on the server, so a range split would still be one
  host's bandwidth. The `i`-th regions dataset is written whole by rank `i mod R`, its
  threads on 64 MiB pieces (parallel on the client with `O_DIRECT`; the server may still
  serialize). A regions corpus is bounded by one stream, a few GB/s, and that is accepted.
- *The manifest is the completeness token, so rank 0 writes it last.* Every rank checks the
  roots are empty before the start gate (no rank sees another's files), writes after it,
  `syncfs`es, and sends `Done` with its counts; rank 0 writes the manifests once
  `Server::gathered` has every rank's document, and the provenance records `ranks`, every
  rank's host and counts, the totals, and `direct`. A failed rank sends `Stop` and no
  manifest appears. The identity check covers the abstract's hash, `--gpus`, the parameters,
  `--dedupe`, `--compress`, and the `--dataset` list.
- *`O_DIRECT`, always.* The user's rule, with the reason: Linux's page cache near full makes
  choosing a victim page cost more than linearly, clean pages included, so datagen must never
  fill it. Datagen also never reads what it writes, and the client's cache is exactly what a
  benchmark must not have warm, since in the common case the SUT is an appliance that runs no
  user process, the corpus goes through the mount, and `--drop-caches` on the clients is the
  only cold start there is (DRAM inside the SUT is the SUT's to use). Writes are 1 MiB from
  4 KiB-aligned buffers; an unaligned tail is tried as is (NFS accepts it) and, on `EINVAL`,
  padded and the file truncated, learned once per datagen. (Seen 2026-10-05, on CI: the
  truncate zeroes the rest of the last block through the page cache, and on Linux 6.17 with
  ext4 `data=writeback` that folio stays resident after write-back in 2 to 6 files of 256,
  the last page or two of each, stable over time and gone after an fsync-and-`DONTNEED`
  pass; never on 6.18 locally. An eviction right after the truncate, `sync_file_range` then
  `POSIX_FADV_DONTNEED`, did not clear it, so datagen is left as is, the residency test
  allows one folio per padded file, and `--drop-caches` is the documented way to a fully
  cold padded root; NFS accepts the unaligned tail and pads nothing.) A root that refuses
  `O_DIRECT` falls back to the page cache, said once (tmpfs before Linux 6.6). A side effect that
  matters later: each direct write completes before the next, so a crash leaves a strict
  prefix of every file, and a file of the right size is a whole file.
- *Prefix-stable dedupe: `aeiou-positional/2`.* The user wants one very large corpus
  generated once and then runs over ever larger subsets of it, and one day a corpus grown in
  place. Names, sizes, and directories were already functions of the id alone; dedupe was
  not: `u = id mod ceil(files / D)` made the groups depend on the count, so a prefix of the
  ids had a ratio below `D` (down to 1 when the prefix is at most `files / D`) and growing
  the count would have changed what existing files should contain. The wrapper now groups
  `D` consecutive files (`u = id div D`) and `D` consecutive blocks of a regions file; ~~every
  prefix and every extension has ratio `D`~~ (true only of files of equal sizes without
  framing between samples; 2026-10-15, §3.77: `aeiou-positional/4`). No dataset id had been published, so the version
  bump cost nothing; with `D = 1` the bytes are unchanged. The Python writer and the Rust
  writer moved together.
- *The launcher.* `aeiou-launch` already passed the command verbatim; it now says `run` or
  `datagen`, takes `-f FILE` (one or more hosts per line, `#` comments), and expands one
  `pdsh`-style bracket range per name (`n[1-4]`, `n[01-10]` zero-padded, `n[1,3,5-7]`,
  `n[1-4]-ib`) in the script itself, so a usage error exits once. `n{1..4}` is the shell's
  and worked already. Two hundred hosts are one argument or a file.
- *No aggregate rate.* Datagen is a tool that makes the corpus, not a benchmark; each rank
  prints its counts and rank 0 the totals, and that is all.

**Deferred, and kept cheap (`PROJECT_BRIEF.md` item 23).** `--resume` walks the ids in scope
and rewrites in full every file that is missing or not of its size; the scope is a formula
over `--ranks`/`--rank` at resume time, nothing is persisted, so the host count may differ
from the first attempt. Size is the test, not allocated blocks: appliances that dedupe or
compress report reduced allocation, and the compressible payload is explicit zeros, so
`st_blocks` is advisory at best. Growth is a resume under a larger count. A run over the
first `S` ids relaxes the manifest's `count` comparison to "at least" (the Feistel over
`[0, S)` reads only that prefix). The inner loop is already "for each id in scope: write",
which is the shape resume needs.

**Considered and not done.** Work stealing across hosts (static slices assume similar hosts;
`run` makes the same assumption, and a resume pass fills what a slow host left). Several
`aeiou` processes per host (threads reach the same ceilings). A range split of a regions file
(serialized on the server). An `st_blocks` completeness check (unreliable on the targets).
An aggregate progress line at rank 0 (not required; every rank's log is prefixed by the
launcher). Generating behind the server (not possible on an appliance; the loopback-NFS
practice in `runner/REFERENCE.md` §7 remains a development convenience). Buffered writes with
`posix_fadvise(DONTNEED)` instead of `O_DIRECT` (the eviction cost the user described is
paid before the advice runs).

### 3.63 The AgentX replay on a GPU: `keep`, the system prompt as a prefix, and LMCache's `O_DIRECT` path (run, decided, and built 2026-10-07; the replay corrected and run again 2026-10-08)

**The runs.** Brief item 21: `replay_agentx.py` through vLLM 0.30.0 and LMCache 0.5.5 on the
development box's 8 GB GPU (RTX 2000 Ada), local-disk backend on the loopback NFS mount, under
the kit's `strace`. Qwen2.5-0.5B served at 131,072 tokens with YaRN (factor 4; vLLM 0.30 reads
`rope_parameters` and wants `max_position_embeddings` already scaled): 10 % of the corpus's
requests fit its native 32k, 47 % fit 128k, and the replay skips the rest as it was written
to. Engine KV 2 GiB (174,752 tokens, 1.33 sequences of the maximum length), vLLM's default
prefill step on this GPU (2,048 tokens, eight chunks), the first 8 sessions played at once,
300 requests in all (225 sent, 75 over the context), back to back. LMCache's disk limit was
raised to 30 GB and the export moved from the 8 GB tmpfs to the ext4 root, so the store never
evicts, as the corpus reference assumes (18 GB stored). Raw traces and logs stay outside the
repo (`~/MLPerfStorage/agentx/`); the kit has the replay's log, a reduced LMCache log, the
trace metrics, and the fitted file. Two runs, identical but for LMCache's `use_odirect`.

**What the buffered run showed.**

- *The call sequence holds under the agentic load*: every chunk is the six calls of a Python
  `open` around one 3 MiB `read` or `write` (§3.51); no chunk is written twice. 6,042 chunks
  written, 41,763 read: LMCache's own log retrieves exactly that many, 183 requests loading
  once each, a mean of 58k tokens.
- *LMCache logs a lookup at every step a request waits for KV memory*, its held prefix
  falling as the running requests evict it (one request: 53,088 tokens held at its first
  lookup, then a block of 16 tokens fewer every few steps). What the engine held is taken at admission: the last
  lookup before the request's first load or store. `agentx.py fit --replay` does that, fits
  only the requests the replay sent (its sessions as as many slots), and takes `keep` the
  way `fit.py` does: 16 of 20 shares are 0, the whole kept prefix held in 8 of 180.
- *The engine holds the system prompt and some of the conversation, or neither*: of 180
  continuing requests, 141 held nothing, 36 the system prompt and more, 3 part of the system
  prompt, none the system prompt alone. The engine loses a prompt from its end, the system
  prompt last. In the replay the session's shared prefix is chunks like the rest of the
  chain, stored by its first request and loaded whenever the engine lost it.

**Against the abstract.** At the parameters fitted from the run the abstract writes 5,570 to
6,414 chunks over four seeds (6,042). With `sys_local` true it reads 13,713 to 23,854 (41,763):
the system prompt is never read. With `sys_local` false, as it was, it reads 34k to 44k, and
the judge finds three rows outside, all saying the reads are too concentrated (the top 10 % of
blocks take 51 % of the reads against 36 %; read-after-read reuse median 2 GiB against 5): the
system prompt was read by *every* request, and the conversations drew their system prompts by
`sys_pop` (Zipf over the eight) where each session has its own.

**A (decided by the user 2026-10-07, built).** Two changes to the three KV abstracts:

1. Under `sys_local = false` the system prompt's whole chunks are read when the engine holds
   none of the conversation (`held == 0`: a new conversation, or one whose round's `keep`
   draw is under a chunk), not on every request. 177 of the 180 continuing requests above
   are that rule. With `sys_local = true` (every committed fit but this one) nothing changes.
2. `sys_per_slot` (default false): a slot's conversations all start with the system prompt
   of its own index, for sessions of an agent, each with its own. False keeps the pick by
   `sys_pop`.

At the replay's parameters (`fitted.agentx-replay.params.json`: `sys_local` false,
`sys_per_slot` true) the abstract reads 31,549 to 41,235 and is ~~**accepted**~~ (not since the replay was corrected, 2026-10-08, below): 33 rows within,
none outside, fan-out unseen as before. A new pair in `KIT_PAIRS`. The pick now sits inside
`sys_per_slot`'s `when`, which moves its draw site: a seed draws other system prompts, so the
fingerprints and the op counts at the goldens' parameters change (1,578 to 1,590 for
`kv_cache_serving`); with one prompt size the op counts are unchanged (1,764 both), which is
how the fitted files run. Goldens re-recorded.

*Stated, not changed.* The first request of a session writes the system prompt's chunks in
the replay, where the abstract's system prompts are a dataset (datagen's, never written by a
run: eight times 91 writes here). A conversation in the abstract counts its reply against the
context, as vLLM does, and starts anew sooner than the replay, which skips a request over
the context and carries on (lifting the cap raises the abstract's chain loads from 14k to 16k
to 22k); at the fitted context the loads are within the judge's tolerance.

**What the `O_DIRECT` run showed.** `extra_config: {use_odirect: true}`. LMCache 0.5.5's
local-disk backend takes the direct path when the chunk is a multiple of `statvfs`'s
`f_bsize` (1 MiB on the NFS mount; the chunk is 3 MiB), and every chunk took it. The calls are
not Python `open`'s six: a write is `openat(O_WRONLY|O_CREAT|O_DIRECT)`, `write`, `close`
(`os.open` and `os.write`); a read is `openat(O_RDONLY|O_DIRECT|O_CLOEXEC)`, `fstat`, `read`,
`close` (`os.fdopen(fd, buffering=0).readinto`). The same replay: 5,994 chunks written,
41,004 read (within 1 to 2 % of the buffered run); 182,001 calls under the root against
286,833.

**B (decided by the user 2026-10-07, built the same day).** `direct` (default false), a
parameter of `kv_cache_serving` (the local-disk backend's; the `fs://` pair has no such
option): every chunk read and write takes the direct path's calls, with `DIRECT` on the
chunk's open. *One choice made while building:* the brief said the configuration would
declare `sync-direct`; the abstract keeps `sync` instead and puts `DIRECT` on the opens
LMCache makes with it, as `vdb_search_diskann` does. That is the application's own call
stream (it passes `O_DIRECT` itself, on its chunk files only, through plain `os.open` and
`os.write`), so a run under the declared backend is CLOSED, and `sync-direct` would add
nothing to it. At `direct` false the fingerprints are unchanged (the AST's hash is not).

At the second run's parameters (`fitted.agentx-odirect-replay.params.json`, `direct` true)
the calls match the trace's (open, fstat, read, close; open, write, close; no `ioctl` or
`lseek`), the abstract writes 5,594 to 6,440 (5,994) and reads 29,895 to 38,554 (41,004),
and the judge finds one row outside: read-after-write reuse distance, 0.434 against 0.374
(p99 10 GiB against 6). Lifting the context cap leaves it there. The cause is the server's
admission order: a request waits for KV memory while the other sessions run (71 of 219
waited over 10 s, up to 72 s, between their first lookup and their admission), so a chunk is
read back after more of their traffic than in the dry run's slots in turn. The buffered run
has the same mechanism and its row is within (0.274 of 0.392). Recorded as outside in
`agentx-odirect.trace.tolerances.json`; the pair is **not accepted**, as the shared store's
writer is (§3.52). Modelling the admission queue would need the engine's memory as state
across slots, which `GRAMMAR_OPTIONS.md` §5.3 rules out as it does for `retain`.

**The replay corrected (2026-10-08).** The runs above shared chunks between sessions by
accident. The corpus's hash ids are local to a session (every session numbers its blocks
from 0), and `replay_agentx.py` generated a block's tokens from its id alone, so sessions
whose ids lined up sent the same tokens: in the first 60 requests of the eight sessions,
729 of 9,039 chunks were shared by two sessions, sessions 0 and 2 (and 1 and 6) sharing
whole conversations up to 328 chunks deep. The engine and LMCache then served one session
from another's chunks. Everything measured on the 2026-10-07 runs (the held counts, `keep`,
the pairs) carries that.

The replay now generates a block's tokens from its session and its id, except the blocks of
the session's shared prefix (`agentx.shared_prefix`: the longest prefix a sub-agent's first
prompt found stored), whose tokens come from their position in it, so the sessions share
one system prompt as far as the shorter prefix goes. That they share one is not in the
corpus; it is in a public corpus with the text, `sammshen/lmcache-agentic-traces` (CC-BY,
787 agentic sessions): three system prompts serve 756 of its 767 distinct sessions (552,
110, 94), and the first prompts of the 552 share about 35k characters. Offline, only the
shared prefix's chunks are now shared (134 of 9,457, by two to five sessions).

The buffered run again, same setup (`agentx.*` in the kit replaced): 220 requests sent of
the 300 counted (80 over the context skipped, 75 before: which 300 the sessions' threads
reach first is timing), 6,406 chunks written, 45,707 read. The engine
held nothing at admission for 157 of the 220, and for the other 63 part or all of the shared
prefix and never more: `keep` past the system prompt is now 0 in 18 of 20 shares (16
before; the conversations it held before were the ones another session had sent).

*What the shared prefix is.* `agentx.py`'s `sys_tokens` is the prefix a sub-agent's first
prompt shares with what its session stored: the prefix of the session's sub-agents and side
calls. The main agent's prompts never begin with it (in all eight sessions their common
prefix with it is 0); they begin with the main agent's own opening prompt. So "every chain
starts past it" is not what the corpus says. **Tried and rejected (the user chose to try it,
2026-10-08):** a parameter `sys_starts`, the share of new conversations that begin with the
system prompt (fitted 10 of 45 chains), in both KV abstracts. Against the new trace it is
worse, five rows outside (reads 23k to 29k against 45.7k, too many writes), and with
`sys_local` true seven: `turn_in` is one distribution for every turn, so the main agent's
opening prompt, once it is not the system prompt, is spread over turns instead of standing
at the chain's start and being read back when the engine has lost it. The abstract's system
prompt is the prefix a chain opens with (the sub-agents' shared one, or the main agent's
own), and the model of §3.63 A stands; `sys_starts` was removed before it was committed.

*The pair.* At A's settings (`sys_local` false, `sys_per_slot` true, eight prompts) the
abstract writes 5,615 to 6,527 (6,406) and reads 35,076 to 43,920 (45,707), and the judge
finds one row outside: read-after-read reuse distance, p50 3 GiB against 5 (p90 5 against
7). Recorded with that reason in `agentx.trace.tolerances.json`; the pair is **not
accepted**. Why the abstract's re-reads come sooner is not established; the candidate is the
server's admission order, the cause of the `O_DIRECT` pair's row. Next: examine it.

*Not redone yet:* the `O_DIRECT` run and its pair (`agentx-odirect.*`) are still the
2026-10-07 replay's, with the shared chunks.

*Checked against the device.* The SNIA AI workloads TWG's block traces of LMCache with a
CPU tier serving the public agentic corpus above (the TWG's, private: read for insight only,
never quoted here) agree with A's first rule and say more: no chunk is read the thousands of times a
system prompt shared by most sessions would be, so in a large engine the shared prompt stays
in memory (`sys_local` true), while each conversation's own chunks are read back turn after
turn. They also say every write reaches the device when a CPU tier absorbs reads, which is
for the CPU tier's design (next after this).

**Why the abstract's re-reads come sooner (found 2026-10-08).** Not the server's admission
order. The replay's chunk accesses rebuilt from LMCache's log (per request: the chain, what
the engine held, the chunks loaded and stored) give the same read-after-read distance in
any order: p50 6.0 GiB in admission order, 5.8 with the sessions taking turns a request
each, 5.7 a chunk each (the dry run's round-robin); between two loads of a session the
others sent 6 requests (p50), the dry run's slots 7. Lifting the context cap does not
move it either. The difference is what the abstract reads. Split by path, its re-reads of
conversation chunks are at p50 4.4 GiB, near the replay's; its re-reads of the system
prompt are 63 % of all its re-reads (19,110 of 30,121 at seed 1) at p50 2.8 GiB. Under
`sys_per_slot` every new conversation on a slot reads the slot's prompt when the engine
holds none of it, the one-request side calls and the sub-agents' chains included, while
in the replay only the main agent's chains begin with its opening prompt (main agent: 113
requests, 32.6k chunks loaded, 16 chain starts; sub-agents and side calls: 107, 13.1k, 29).
The totals agree and the mix does not. *Tried and rejected (2026-10-08):* C, `sys_starts`
again with the prompt on the main agent's chain starts only (16 of 45): seven rows outside
(the sub-agents' shared prefix, turned into their own tokens, is written by every
sub-agent chain, 2.6k writes more than the server's one per session), one with the
context cap lifted. Not committed. The pair's reason in `agentx.trace.tolerances.json`
says this.

**Two prefixes (proposed 2026-10-08, decided by the user and built the same day).** From primary sources
instead of a fit to the trace that produced them. Each claim with its source, in falling
confidence:

- *Prompt order* (Anthropic's prompt-caching documentation): a request's prompt is its
  `tools`, then `system`, then `messages`, so the first blocks of every prompt are its tool
  definitions.
- *Sub-agents* (Claude Code's sub-agent documentation): a sub-agent gets its own system
  prompt plus environment details, "not the Claude Code system prompt", a filtered set of
  the tools, and a fresh context without the parent's history. So a sub-agent's prompt
  differs from the main agent's at its first block, and sub-agents of one type share a
  prefix. The corpus agrees: the main agent's prompts share no block with the sub-agents'
  prefix (all eight replayed sessions), and the sub-agents share 21k to 33k tokens
  (quartiles over the 146 sessions with sub-agents; median 27k).
- *The main agent's prefix* (the corpus, an inference): when the main agent's prompt cuts
  the previous one (3,301 cuts over the corpus), it keeps nothing in 22 % (its first block,
  a tool definition, changed) or most often 30k to 45k tokens: its tools and system prompt,
  which a restart keeps. Whether two sessions share it is not in the corpus (its ids are
  local); the documentation's static and session-specific parts of the system prompt
  (secondary sources) say sessions of one version and configuration share the static
  part. The public agentic corpus with text agrees (three system prompts for 756 of 767
  sessions).
- *LMCache 0.5.5* (its source, under `lmcache/`: `v1/token_database.py` for the key,
  `integration/vllm/vllm_v1_adapter.py` for what is stored and loaded,
  `v1/storage_backend/local_disk_backend.py` and `storage_manager.py` for the I/O): a chunk's key is a
  chained hash of its whole prefix, with the TP rank; a chunk already stored is not written
  again; the last partial chunk and decode are not stored; a request loads one contiguous
  range, from the chunk holding the engine's cached tokens to LMCache's longest prefix hit,
  on four threads; the index is in memory. So a request whose engine holds nothing loads
  its prefix and its conversation as one range, and a prefix shared by many requests is
  written once.

*The model.* Each slot has two prefixes: the main agent's (`sysp`, as now) and a
sub-agent's (a second dataset, `subp`, with `sub_prompts`, `sub_tokens` and its bounds,
`sub_pop`, and `sub_per_slot` like `sys_per_slot`). A new conversation draws its `kind`:
main, sub-agent, or none (a side call that begins with no stored prefix); a continuing one
keeps its conversation's. The prefix of its kind takes the place the system prompt has
now: its whole chunks read when the engine holds none of the conversation (`sys_local`
false) or never (true), its tail counted with the conversation's own tokens, the context
check counting it. Both prefixes stay datasets, never written by a run (LMCache writes a
shared chunk once). `kind`'s draw goes after every existing draw, so its default (every
conversation the main agent's) keeps the fingerprints and op counts of every committed
parameter file; the AST's hash changes. No contract change.

*The fit* (`agentx.py`): `kind` from the chain starts (a main-agent chain, a sub-agent
group's chain, or neither); `sys_tokens` becomes the main agent's prefix (the most common
kept prefix of its cuts, or the longest common prefix of a session's main-agent chain
starts), and the present measure moves to `sub_tokens`; `turn_in` of a chain start is its
prompt less its kind's prefix. *The replay* (`replay_agentx.py`): both prefixes generated
by position, so sessions share them as far as the shorter goes, as the documentation says
sessions of one configuration do. Then the buffered and `O_DIRECT` runs again (the second
not yet redone), refit, and judge.

*Expected, not yet measured:* side calls and sub-agents stop reading the main agent's
prompt and read their own shorter one, shared by the session's sub-agents and so re-read
more often by them; the system-prompt share of re-reads should fall toward the replay's.
*Open:* whether `none` is worth a kind (the side calls are a few hundred tokens, under a
chunk, and store nothing); whether a sub-agent prefix is per slot or per type across slots
(`sub_per_slot`, the same choice as `sys_per_slot`); how the CPU tier (next) holds a prefix
many requests share, which LMCache's source answers: the CPU tier is written through, LRU,
5 GB by default, and a chunk read from disk is promoted into it.

**Built (2026-10-08), with what building it found.** The defaults the open points took: a
`none` kind kept (the fit gives it its weight), `sub_per_slot` added (default false), each
prefix measured per session and model (another model's tokens are other tokens). The main
agent's prefix is 511 blocks, 32,704 tokens: the length a main-agent prompt keeps when it
cuts the previous one, 134 times in the corpus against at most 19 for any other length
(`agentx.main_blocks`); 80 % of main-agent prompts begin with their session's first 511
blocks, the rest after a tool set changed. Three more parameters, each found by a fit that
failed without it, each drawn after every existing draw and defaulting to the old
behaviour (so every committed parameter file keeps its fingerprints; the three ASTs'
hashes re-recorded):

- `first_in`: a new conversation's first prompt past its prefix, apart from `turn_in` (a
  turn's growth). One pooled distribution is what made B and C fail. The fit takes it past
  the prefix of the chain's kind or what its session had already stored, the longer (a
  chain of neither kind often opens with another sub-agent type's tools).
- `sys_held`, `sub_held`: the tokens of each prefix the engine holds when a request
  arrives, apart from its conversation. With the main prefix shared by the sessions, vLLM
  held all of it for 89 of 103 main-agent requests; reading the prefix whenever no
  conversation is held was wrong. The fit takes them from the requests whose engine held no
  whole chunk of the conversation.

The corpus fit (`fitted.agentx.params.json`) stores 15.3 chunks a request against the
reference's 14.41 (+6 %, +8 % before) and loads within 3 % of the default `keep`'s share
of its hits. The replay run again with both prefixes shared (`agentx.*` replaced; the
replay's code shares one main and one sub-agent prefix per session, the first of any
model): 219 requests, 5,454 chunks written and 33,804 read (6,406 and 45,707 with one
prefix: the shared main prefix is written once and mostly stays in the engine). At the
fitted parameters the abstract writes 5,579 to 6,656 and reads 13,510 to 24,383, and four
rows are outside: the reads' share of data ops and of bytes, and both reuse distances
(p50 2.5 and 2 GiB against 4). With the context cap at 256k, one (read after write). The
pair stays **not accepted**, its rows recorded as a checkpoint. *Next:* where the
conversation reads go (their spread over seeds is wide: the long main-agent chains a slot
draws carry most of them), and what the primary sources say about `first_in` and the
prefix holds; then the `O_DIRECT` run again (still the first replay's).

### 3.64 The engine's KV pool and the holds: what three times the pool changes, computed holds deferred, the CPU tier fixed at 0 (run and decided 2026-10-08)

**The question (the user's).** Can an abstract be scaled along any dimension and keep the
characteristics it was fitted to, or must the engine's KV pool be part of it? The pool is HBM
less the model (the GPU's make and model) and LMCache's CPU tier (DRAM, which a submitter
would buy as large as the budget allows to unload the storage). A training abstract has one
stand-in for the hardware, the batch's compute time; the KV abstracts have the pool.

**What the primary sources say** (read 2026-10-08):

- *vLLM 0.30.0* (its source: `v1/core/kv_cache_utils.py`, `FreeKVCacheBlockQueue`;
  `v1/core/block_pool.py`; `single_type_kv_cache_manager.py:581`; `sched/scheduler.py:753`):
  a freed block that has a hash joins an LRU queue, a request's blocks freed in reverse so
  that its head is evicted last; a hit takes the blocks back out of the queue; when a step
  cannot get blocks, the FCFS scheduler preempts the last running request and frees its
  blocks. So an engine holds a prefix of a prompt, never a middle, and a prefix every
  request touches stays: the fitted `sys_held` is the whole main prefix in 17 of its 20
  shares at 2 GiB.
- *LMCache 0.5.5* (`v1/storage_backend/local_cpu_backend.py`, `v1/cache_engine.py`): the
  CPU tier's size is `max_local_cpu_size` GiB (`× 1024**3`), not the machine's DRAM; written
  through (§3.63), so it removes reads and never writes.
- *Claude Code's documentation* (the sub-agents and prompt-caching pages, as published
  2026-10-08; primary for the application, not for an engine): a request is its system
  prompt and tools, then the project context (CLAUDE.md, memory), then the conversation,
  which opens with the working directory and the git status; compaction keeps the first
  layer (the corpus's cuts keeping 30k to 45k tokens). A sub-agent's first prompt is its own
  system prompt and environment, the CLAUDE.md hierarchy (not for Explore and Plan), the git
  status, any preloaded skills, then the delegation message: sub-agents of one type in one
  session share all but the delegation message (`sub_tokens`, 21k to 33k), and a sub-agent
  chain's `first_in` is that message. A *fork* inherits the parent's whole prompt and reads
  its cache, a kind the corpus predates; not modelled. The cache is scoped to one machine and
  directory: sessions in parallel in one directory share the prefix, sequential ones only
  with the same git snapshot, so the replay's one main prefix across its eight sessions is
  the parallel case.

**The run.** The replay of §3.63 again, with `KV_BYTES` 6 GiB: 524,288 tokens, 3.0 times
the 174,752 of 2 GiB, all else the same (raw in `~/MLPerfStorage/agentx/run-20261008c`; the
kit's `agentx-6gib.*`). 232 requests were sent, 219 at 2 GiB: the replay's 300-request
budget is shared by its session threads, so which requests go depends on the server's
speed (recorded; ~~a per-session budget would fix it~~ fixed 2026-10-09: each session plays
its own share of the 300, the first of its requests in time order, 38 or 37 each, the
over-context ones it skips included, so what is sent depends on the corpus and the
arguments alone; checked against a stub server of random latency, two runs sent the same
233 requests. Both runs above were made under the shared count).

| | 2 GiB (174,752 tokens) | 6 GiB (524,288 tokens) |
|---|---|---|
| chunks written | 5,454 | 5,112 |
| chunks read | 33,804 | 6,770 |
| reads / data ops | 86 % | 57 % |
| block reads after a read / after a write | 88 % / 12 % | 64 % / 36 % |
| read-after-read distance p50 | 4 GiB | 2 GiB |
| request size, runs, popularity | 3 MiB, single-op runs, top 10 % of blocks 35 % | the same (31 %) |

The writes do not depend on the pool (LMCache stores a chunk once); the reads fall five
times and come sooner. The refit moves the demand parameters only as the requests sent
differ, and the holds: `keep`'s median from 192 tokens to about 65,000, `sub_held` up,
`sys_held` already whole. Against the 6 GiB trace, four seeds each:

| parameters | chunks read | verdict |
|---|---|---|
| the 6 GiB run's demand, the 2 GiB holds | 20k to 33k | 4 rows outside (reads' share, both reuse distances) |
| the 6 GiB refit | 4.2k to 13k | accepted (fan-out unseen, as at 2 GiB) |
| the 2 GiB set as committed | 13.5k to 24k | 2 outside (its wide seed spread widens the tolerance) |

The first two differ in `keep`, `sys_held`, and `sub_held` alone: the pool reaches the
storage through those three, and the rest of the fit carries over.

**Whether the pool explains the holds** (`agentx.py holds`; `agentx.simulate`: vLLM's prefix
cache over the corpus's 64-token blocks as the sources above describe it, a request taking
its prompt's and reply's blocks when admitted). Held tokens predicted over measured, and the
requests within one chunk of their measured hold:

| timeline | pool | 2 GiB run | 6 GiB run |
|---|---|---|---|
| the engine's (LMCache's first lookup and first load or store), with preemption | the run's | 1.14, 78 % | 1.02, 79 % |
| the engine's | the other run's | 2.93, 21 % | 0.62, 41 % |
| clockless (each session back to back, seconds = c_in × prompt + c_out × reply) | the run's, its own c | 1.34, 55 % | 1.03, 76 % |
| clockless, the 6 GiB run's c (the engine's speed with little queueing) | the run's | 1.79, 47 % | — |

On the engine's own timeline LRU and preemption explain the holds at both pools, and the
wrong pool does not. A first version without preemption predicted the same at either pool:
in the replay a session's next request arrives as its last finishes, and only the pool's
pressure (at 2 GiB a request waited 4.1 s at the median, LMCache looking it up 162 times;
173 preemptions simulated, 11 at 6 GiB) evicts it meanwhile. The clockless timeline, the
one an abstract could compute, is good without pressure and over-predicts under it, and no
single share of the pool fits both runs (0.8: 1.44 and 0.95): what it lacks is the
admission's dynamics, the waiting and a preempted request's recompute, not a fixed overhead.

**Three kinds of dimension.** *Replication* (GPUs, hosts, steps, run length) scales exactly:
each GPU's stream is positional and independent. *Data size* keeps its shape (layout and
order are formulas); what changes is what the storage can cache, which is the solution's
business. *Application-side capacity* (the engine's pool, LMCache's CPU tier) is a filter
between the application's demand and the storage, and a fitted draw of a filter's output
does not carry to another capacity. It is self-similar at constant pressure, roughly the
context of the requests running at once over the pool (about 2.5 at 2 GiB, under 1 at 6
GiB), as a scale model is faithful at a matching dimensionless number. A training step's
compute time only paces the op stream; a KV pool changes which ops there are.

**Decided (the user, 2026-10-08).**

- *Computed holds are deferred.* The holds stay drawn, fitted at a reference engine and CPU
  tier that a CLOSED submitter does not choose: the accelerator is emulated, so the WG fixes
  the model, the GPU, TP, and the tier, and a submitter cannot buy them (WG process,
  `PROJECT_BRIEF.md` §8). Any other value is another workload, as any parameter is; a
  researcher or a vendor may set them, and comparability is then theirs to argue.
- *The CPU tier is 0* (LMCache's `local_cpu` off), as in every replay so far. A nonzero
  reference is the WG's to choose; it needs a replay with the tier at that size and a draw of
  the loads it absorbs, or a plain LRU over the chunk stream (the tier has no preemption),
  which is the likeliest first piece of computed holds.
- *The reference holds must be fitted at the reference configuration.* The committed ones
  are the development box's (Qwen2.5-0.5B, 2 GiB, eight slots), right for it and nothing
  else. Without computed holds that needs a replay on the reference hardware, or a smaller
  one at the same pressure, which would need validating.
- *Kept so that computed holds stay cheap:* every fit of the holds records its conditions;
  the hold simulation is a kit tool; the 6 GiB pair is a judged pair. `GRAMMAR_OPTIONS.md`
  §5.3 (capacity an input, not a simulated cache) stands. A later design would be a cache
  per engine, over one actor's slots and so not the cross-actor timing §5.3 rules out,
  deterministic in the slots' round-robin order, with the admission's dynamics under
  pressure, and a contract change; its parameters would default to the drawn holds, keeping
  every fingerprint.

**Built (2026-10-08).** `fit.conditions` and `fit.write`: a fit of the holds writes its
conditions as the parameter file's `provenance` (never compared): the model, its context,
`kv_cache_memory_bytes`, `engine_pool_tokens`, `cpu_tier_bytes`, `slots_per_engine`, and for
`agentx.py` the replay's seconds per prompt and reply token (queueing included), from the
server's log or `--condition`. Every KV parameter file fitted to an engine has them; the
O_DIRECT and `kv_cache_shared` files were given them without a refit, from their runs' logs
and the kit's README. `agentx.py log` writes the reduced server log (the conditions' lines,
and per request its first lookup, its last before admission, and its admission; 580 lines at
2 GiB, a superset of the 438 before); `agentx.py holds` runs the simulation (the corpus is
not committed, so as with `fit` the tests cover `simulate` on a synthetic case and the
conditions against the kit's logs). The 6 GiB pair (`agentx-6gib.*`) is accepted. The chat
replay's pool, given in the kit's README as 7,800 tokens, is 8,192 (vLLM's log at 96 MiB,
2026-10-08; the 2 GiB run's 174,752 is 2 GiB over Qwen2.5-0.5B's 12,288 bytes a token,
rounded down to a 16-token block).

**Note of 2026-11-05: the AgentX and ShareGPT replays captured again (the user's choice, after
the shared-fs pair, §3.98's note of the same day).** Every capture of `lmcache-local-disk`
taken again under `aeiou-trace capture`, once with the eBPF source and once with strace, the
same load, on the loopback mount of `runner/REFERENCE.md` §7 (the AgentX store on a directory
on disk, the export no longer the 8 GB tmpfs for it). The two sources give the same calls and
bytes in every buffered run; under `O_DIRECT` the engine held other prefixes as the timing
fell (39,406 chunk reads under eBPF, 38,818 under strace), the fits differing in the holds
alone. The ShareGPT pair gives the 2026-10-02 capture's counts and bytes exactly
and `fit.py` gives the case's values. The AgentX replays did not repeat: `replay_agentx.py`
has played a fixed share per session since 2026-10-09 and skips a request the server would
refuse since 2026-10-15 (RC1 bug 153), so the committed runs were of a replay the kit no
longer makes. Now the 2 GiB and 6 GiB runs send the same 233 requests and write the same
5,663 chunks (219 and 232 requests, 5,454 and 5,112 chunks before), reading 38,818 and 2,321
(33,804 and 6,770), and their fits differ in `keep`, `sys_held`, and `sub_held` alone, which
is this section's claim without the replay's own difference between the runs. Verdicts: the
2 GiB pair stays not accepted with four rows outside (read after read now within; the
extents' read after write, measured since §3.79, outside); the 6 GiB pair ~~is accepted~~
is not accepted, read after write (blocks and extents, p50 3 GiB against 5) outside as at
2 GiB, where at 2,321 reads the reads' share is within. The `O_DIRECT` run (§3.63) is now
the buffered run's replay, its case's `params.json` refitted (the buffered fit but for `keep`,
`sys_held`, `sub_held`, and `direct`), and has the buffered pair's four rows outside; its
earlier reason (the server's admission order, with the buffered row within) no longer holds
and is struck in its tolerance file. The read-after-write distance being short in the
abstract at both pools is the next question for the two-prefix model (§3.63's checkpoint).
The `holds` simulation's agreement (14 % and 2 %) was not run again.

**Note of 2026-11-06: why the abstract's reads of a written chunk came sooner, the side runs
(the user's choice of the open question above; built the same day, the pool as a computed cache
proposed).** *Found.* The replay's chunk accesses rebuilt from the corpus, the replay's log, the
server's, and LMCache's rule (a request loads from the chunk holding what the engine kept to its
store's longest hit and stores the rest) give the 2 GiB trace's 5,663 writes exactly, 38,945
reads against 38,818, its 4,496 reads after a write, and their mean distance within 1 %. In it a
session plays its requests in time order, and the main agent's chain is suspended while its
sub-agents and side calls run, then resumed: a sub-agent's chain is interrupted by its session's
other requests at 59 % of its continuing requests, a side call's at 50 %, the main agent's at 4 %
(but those after 6 to 31 requests). The abstract's `reuse` is 1, so a slot ran each chain to its
end. At 6 GiB nearly every read is of a chain resumed after such a run (the main agent's: 75 % of
its resumptions read, 88 chunks each, against 2 % of its straight continuations), so a chunk is
read after the run's traffic (made contiguous in the rebuild, read after write falls from p50
4.8 GiB to 1.25). At 2 GiB nearly every continuing request reads, 97 % of the reads after a write
at the chain's next request, and the abstract's distance is short for two reasons together: it
reads too little (the replay's reads thinned to the abstract's count give a mean of 3.3 GiB
against 4.6) and runs its chains contiguously (another 0.7). The session order of the replay is
the sessions' order in the corpus: a main agent's chain, then a run of sub-agents served in turn
(often two to five beside one another, or one among one-request side calls), then the main
agent again.

*Built: side runs* (`lmcache_local_disk`, five parameters drawn after every other draw, so every
committed parameter file keeps its fingerprints; the AST's hash changes). After each request of
a slot's main lane, `side` (0 none) gives the requests of a side run's leading chain; the run
serves `side_lanes` chains in turn for `side_lanes` × `side` requests, the leading one in its
first lane and the others of `side_turns`, chains of `side_kind`; then the main lane goes on
where it was. In the corpus a run's longest chain holds two thirds of its requests at the median
and 45 % of all side requests, so a run lasts about as long as its leading sub-agent; a run of
drawn length with chains of drawn turns cut the long sub-agents short and lost a third of the
corpus fit's loads. What a request does at a run is read from bindings the request before left
at the end of the loop body (`x @ (r − 1)`); the distance to the request a chain goes on from is
`d + (x mod (r + 1))`, the form rule V3 proves. The state is read only where `d` is 1: a slot of
one agent; without that guard `chat-64-slots` (`reuse` 40) overflowed the stack, a miss in the
VM's `at` cache walking the state back a request at a time. `agentx.py`: `lanes_of` takes the
runs from each session's order, `kind` and `turns` are now the main agent's chains', `side` a
mixture of none and the runs' shares, and `keep` is counted past a chain's base (its kind's prefix
or what its session had stored when it began, as `first_in`), since 19 of the replay's 26 side
chains open with another sub-agent type's stored tools. *Tried and rejected the same day:*
`keep_back`, the hold of a chain resumed after others, apart from `keep`: at 6 GiB the engine
held such a chain whole or nothing past its prefix, by its size (60k to 120k tokens lost, 18k to
56k held), which a drawn hold independent of size cannot give; the user asked whether the loss is
better a share of the context (it is: nothing censored, the bytes scale with the chain), and then
whether to model the pool itself, below.

*Verdicts.* The 6 GiB pairs (eBPF and strace) are **accepted** (40 rows within, fan-out unseen),
read after write within. The 2 GiB pairs keep four rows outside, read after write now p50 3 GiB
against 4 (2.5 before) and the reads' share (17.8k to 25.7k reads against 38.8k). The `O_DIRECT`
pairs keep their four, and strace's adds the top tenth's popularity (27 % against 36 %, 0.084
against 0.083), the case being the eBPF capture's fit. The corpus fit loads 15 % under the
corpus's hits at the default `keep` (10 % before), judged now over four seeds: one seed's 10,000
requests spread by a fifth either way.

*The pool as a memory pool (the user's question; the experiment run, the design to be drafted).*
The holds are a fixed pool's filter: vLLM's prefix cache evicts freed blocks least recently used
first, a request's tail first, so what a chain loses while it waits is the interruption's demand
against the pool, and the shared prefixes stay. `agentx.simulate` over the replay's own requests,
its holds driving LMCache's loads in the rebuild, on timelines an abstract can compute:

| timeline | 2 GiB reads | after write, p50 | after read, p50 | 6 GiB reads | after write | after read |
|---|---|---|---|---|---|---|
| the engine's measured holds | 38,945 | 4.6 GiB | 4.4 | 2,455 | 4.8 | 4.9 |
| the engine's own timeline | 36,276 | 5.0 | 4.7 | 2,280 | 4.8 | 4.7 |
| one request at a time, the slots in turn | 39,881 | 5.0 | 4.8 | 2,865 | 3.9 | 3.7 |
| each slot's next request admitted together | 37,098 | 5.0 | 4.8 | 3,330 | 4.4 | 3.7 |
| sessions back to back at the replay's speed | 31,623 | 5.0 | 4.7 | 2,343 | 4.8 | 2.7 |

The order the abstract's slots already take, one request at a time in turn, with the pool's size
alone, gives the reads within 2.4 % at 2 GiB and 17 % at 6 GiB and both distances near; per
request it agrees within a chunk in only 65 % at 2 GiB, the admission's waits and preemptions
moving single requests, not the stream. The drawn holds give 15k to 26k reads at 2 GiB. This is
the replay's demand, not the abstract's drawn one. It reopens "computed holds are deferred"
above: the pool would replace `keep`, `sys_held`, and `sub_held` with one size from the GPU
model, and so carry an abstract to another accelerator, which drawn holds cannot. Built the same
day as a kit tool: `agentx.py holds --timeline engine|paced|serial|rounds` (~~`--clockless`~~,
now `paced`) prints the storage LMCache would see at the predicted holds and at the measured ones
(`lmcache_accesses`, `stack_distances`), the table above from the committed logs. Next: the design
drafted for the user's decision, a contract change. The context tree the side runs expose, as a digest finding, is
`PROJECT_BRIEF.md` §6 item 69.

### 3.65 The access layers: protocol, API, cache, transport, and buffer as separate axes, `s3dlio` as one engine among them (proposed and decided 2026-10-09; the object engine on `object_store` instead, decided the same day)

**The question (the user's).** Polishing the KV abstracts kept running into "backend"
questions that seemed unconnected: whether `s3dlio` needs PRs to give it buffered,
`O_DIRECT`, POSIX AIO, and `io_uring` file access beside its S3, Azure, and GCS stores;
whether `direct://` against `file://` in the AST would allow a run that mixes object and
POSIX storage, and whether mixed-protocol workloads should be supported at all; whether
`O_DIRECT` is a modifier of several APIs rather than part of a backend's name; and RDMA,
which NFS and S3 can both use. Is that one design or a pile of ideas? The user's framing:
the file structures (Parquet, HDF5, …) must be layered above the access-API work, and the
access APIs are what is tangled.

**What exists.** The file structures are already above it: format classes live in the
Python builder and emit POSIX-shaped ops, and the runner is format-ignorant (brief §5,
"Format classes"). The tangle is below, in "backend". The nine names of
`BackendKind` (`runner/aeiou/src/backend.rs`) are the product of three choices: the API
(syscalls, glibc AIO, kernel AIO, `io_uring`, `mmap`), whether `O_DIRECT` is added to every
regular-file open (the `-direct` names), and how completions come back (a blocking thread
per actor behind the `Backend` trait, or an event loop behind the `Engine` trait of
`uring.rs`). The `Backend` trait is written over `OwnedFd`/`BorrowedFd`, which no object
store has. The AST declares one `backend` for the whole abstract (contract 0.3, §3.48), and
`--root` is one directory. None of this was the plan: §3.12 classified backends on three
axes (initiation, completion, memory target) with `--buffer pageable|pinned|gpu` and
`--cache buffered|direct|dontcache|fadv-dontneed` as orthogonal flags and a validity table
(brief §4, `NAPKIN_MATH.md` §8.5), and §3.35 noted that the build had reduced both axes to
the `-direct` suffix.

**Assessment: one design.** Every one of the questions asks the same thing of a different
axis: is the choice the application's (in the abstract, held fixed by CLOSED) or the
solution's (the submitter's, recorded in the report)? The interposition test (brief §5)
already answers that. The questions looked unconnected because the runner gives every axis
one home, the backend name, when each binds to something different:

| Axis | Binds to | Whose | Today |
|---|---|---|---|
| Op stream, with the application's own flags (`O_DIRECT` on the opens it uses it on) | the op | the abstract | built (`OpenFlag::DIRECT`; LMCache's `direct`, §3.63) |
| Protocol family: POSIX or object | the namespace or dataset | the abstract (the application's API) | POSIX, implicitly |
| Endpoint: a directory, a bucket, a URI | the namespace or dataset | the run (it is the system under test) | one `--root` |
| API (engine): `sync`, `posix-aio`, `libaio`, `io_uring`, `mmap`; for objects the client library | the run, per protocol family | the abstract declares it; another is another workload (§3.48) | the backend name |
| Cache mode forced on every open: `direct`, `dontcache`, `fadv-dontneed` | the run | an override, so another workload | the `-direct` names |
| Transport: TCP or RDMA | the mount, or the client library's configuration | the solution | not modelled (`NAPKIN_MATH.md` §8.5 lists TCP against RDMA as an NFS mount row) |
| Buffer: pageable host, pinned host, GPU memory | the run | open (GDS, `nixl-posix`) | pageable host only |

**Proposed answers to the questions.**

- **Mixed protocols: support them, bound per namespace.** Real jobs mix: a training set
  read from an object store while checkpoints go to a file system is common, and KV engines
  have remote tiers beside local disk (LMCache's remote connectors **[verify]** against its
  source before relying on it). The structure is mostly in place: an abstract already has
  namespaces and datasets, each with its own root under `--root`, and the event loop already
  wakes on an eventfd, so a completion source on another runtime can wake the same loop that
  owns the ring. What changes is that the protocol and API are declared per namespace (with
  the abstract's top-level declaration as the default) and that a run is given an endpoint
  per namespace rather than one `--root`. ~~Design it per namespace now so nothing forecloses
  it; build the single-protocol case first.~~ Decided 2026-10-09: designed and built now
  (below). Counter-argument: no traced application of the
  eight mixes protocols today, and every mixed run doubles the setup a submitter must
  reproduce; the per-namespace binding costs little, but a WG workload that mixes should wait
  for a trace that does.
- **No `direct://` in the AST.** `s3dlio`'s scheme puts a modifier in the protocol's name,
  the same tangle as the `-direct` backend names. `O_DIRECT` already has its two right
  places: in the op stream, on the opens where the application uses it (CLOSED), and as a
  run-wide cache mode that forces it on every open (another workload). The scheme the AST may
  carry is the protocol family; the cache mode is never in it.
- **`O_DIRECT` is a modifier, orthogonal with known exceptions.** It applies to every POSIX
  API but `mmap` (refused, as the brief's validity table planned); for `libaio` it is what
  makes submission asynchronous (buffered kernel AIO mostly blocks in `io_submit`), which is
  a property to report, not a refusal; for objects it has no meaning and is refused. The
  run-wide flag becomes `--cache`, as planned in §3.12, with the abstract's own flags as
  written the default (not "buffered": an abstract may carry `DIRECT` on some opens), and
  today's nine names stay as aliases (`io_uring-direct` = `io_uring` + `--cache direct`).
- **RDMA is below the protocol, and whose it is depends on where the client runs.** Under a
  kernel client (NFS with `proto=rdma`, Lustre, GPFS) it is a mount option the application
  never sees: the solution's, chosen by the submitter and recorded with the other mount
  options the report already carries (§3.30). Under a user-space client in the runner (S3
  over RDMA **[verify]** which clients and servers support it, `libnfs`, NIXL), it is a client
  library option: a run option recorded in the report, solution side by the same reasoning,
  since the application's calls are unchanged. Counter-argument: where the RDMA path needs a
  different client API than the traced application uses, it is not a swap under an
  unmodified application, so it is another workload. RDMA and the buffer axis are one
  design: RDMA matters most when the bytes land in GPU memory.
- **PRs to `s3dlio` for the POSIX APIs: not on aeiou's path.** The runner should not send
  its POSIX ops through `s3dlio`. The object API has no descriptor, `open`, `close`, `lseek`,
  `fstat` of a descriptor, `fadvise`, or `getdents`, so it cannot reproduce a traced call
  sequence; a tokio task and a `bytes::Bytes` per op would charge every POSIX API costs that
  are not its own; and the runner parks one VM per actor with one op in flight, which async
  tasks do not model. If the WG wants one set of POSIX engines shared between the two, the
  direction is reversed: aeiou's engines, as a crate with no tokio, which `s3dlio`'s
  `file://` and `direct://` could use. What aeiou may need from `s3dlio` is narrower, and
  guesses until its current source is read **[verify]**: a ranged read into the caller's
  buffer (no allocation and copy per op), control of its runtime's and pools' sizes, and
  per-request timing. Those PRs come from measuring it (step 3 below), not before.
  *2026-10-09:* moot; the object engine is built on `object_store` (decided below).

**The object mapping (draft; §3.32 extended).** *2026-11-03: to be replaced by object
requests as the abstract's own ops, with no mapping (§3.95, brief item 61); built as below
until then.* Per op, for a namespace declared object:

| Op | Object operation | Notes |
|---|---|---|
| `open` (read) | none, or `HEAD` when the abstract `fstat`s it | the handle is the key, the mode, and a position |
| `read` at an offset | ranged `GET` | `streaming` mode: the reads of one open, sequential from its first, as one `GET` streamed; a legal row since a shim could do it (§3.12) |
| `open` (write, `CREAT`/`TRUNC`) + sequential `write`s + `close` | one `PUT` at `close`, or a multipart upload in parts of a set size | the size is the writes' positional expressions, as for a POSIX namespace |
| `fstat`, `stat` | `HEAD` | |
| `readdir` | `LIST` of the prefix, paged | the page size an option of the run |
| `unlink` | `DELETE` | |
| `rename` | copy and `DELETE` | refusable per store; not atomic |
| `mkdir`, `rmdir` | none | prefixes are not objects |
| `lseek`, `fadvise`, `ioctl`, `fsync`, `fdatasync` | local | an upload is durable at its `close` |
| `write` not sequential from 0, `O_APPEND`, `ftruncate`, `fallocate`, a read past a writer's end | refused | a new validity rule (~~V16~~ V18: V16 was already the `trace` rule of §3.58, and V17 went to the protocol of namespaces sharing a root, both noted 2026-10-09) at `aeiou check`, as V9 refuses an access mode a format class lacks |

The manifests (`.aeiou-dataset.json`, `.aeiou-namespace.json`) become objects at the
prefix's root; `datagen` writes through the same engine. The fingerprint is untouched, as
for every backend.

**The contract.** Declaring the protocol and API per namespace, and splitting today's
`backend` into an API and a cache mode, is a contract change (0.6). ~~Proposed: the top-level
`backend` keeps its meaning as the default API of POSIX namespaces, with the `-direct` names
accepted and read as the API plus `cache: direct`; a namespace or dataset may declare
`protocol: object` and its client. Every existing AST stays valid and every fingerprint and
dataset id stays the same.~~ Decided 2026-10-09 (below): `backend` is replaced by separate
fields and every AST regenerated.

**Order of work.**

1. This entry, decided: the axes, the per-namespace binding, the mapping and ~~V16~~ V18, the
   contract change. Decided 2026-10-09.
2. The runner refactored without a change of behaviour: a handle the engine owns instead of
   a descriptor, the backend name split into API and `--cache` with the nine names as
   aliases. The proof is that every fingerprint, golden, and test is unchanged.
   *2026-10-09:* with contract 0.6 (the AST's `backend` replaced, every AST regenerated, so
   the goldens that hold whole ASTs change and the fingerprints and dataset ids do not) and
   the protocol and endpoint per namespace.
3. ~~`s3dlio` as an engine behind a cargo feature, over `file://` first: the same
   `train_small_files` fingerprint against `sync`, pricing the library alone (§3.32's first
   measurement). This finds which PRs to `s3dlio` are needed.~~
4. ~~An S3 server on the development box: a run over `s3://`, with `datagen` and the
   manifests through it.~~
5. ~~The PRs to `s3dlio` that steps 3 and 4 showed are needed.~~

*Superseded 2026-10-09 (below, the object engine on `object_store`):* steps 3 and 4 are one
step, the object engine on `object_store` behind the cargo feature `object`, run against an S3
server on the development box with `datagen` and the manifests through it; step 5 is dropped.

Then `libnfs`, `gds`, and `nixl-posix` come in on the same axes (the last two with the
buffer axis), and a transport row joins the report.

**Decided (2026-10-09, the user).**

- **The axes and their owners as proposed** (the table above), transport and buffer
  included, rather than flat names such as `s3-direct` or the axes without those two.
- **Mixed protocols designed and built now**, not only bound per namespace with the
  single-protocol case first. The user's reason: KV caches are mixed workloads today, just
  not yet in our traces; NVIDIA's G3 tier is POSIX and its G4 tier object (the user's
  statement; **[verify]** against NVIDIA's documentation before the docs cite it as fact).
  Mixed traces will be needed to debug this support; until there are some, what they show
  is treated as bugs when it comes. This supersedes the counter-argument above as the
  reason to wait.
- **Contract 0.6 replaces `backend`** with separate fields (the API, the cache mode, and per
  namespace the protocol family and its client) and regenerates every AST, rather than
  keeping `backend` with the `-direct` names read as aliases. The user's reason: the only
  ASTs that exist are this repository's. ~~The command line keeps the nine names as aliases.~~
  *Superseded the same day (the user):* the aliases are dropped too, from the command line,
  the output, and the report; with no users yet, old names only clutter the page and the
  usage, and orthogonal flags let an expert name exactly the layered behaviour wanted.
- **The first object abstract's application: deferred** until ~~`s3dlio` over `file://` is
  measured (step 3)~~ the object engine runs against an S3 server (step 3 as revised the same
  day); steps 2 and 3 do not depend on it.
- **No shared POSIX-engine crate yet;** ~~only the narrow PRs to `s3dlio` that the
  measurements show aeiou needs. The crate may be raised again after step 3.~~ *Superseded the
  same day:* with the object engine on `object_store`, no PRs to `s3dlio` are on aeiou's path;
  a crate shared with `s3dlio` would be the WG's offer to make, not a step of this work.
- **RDMA under a user-space client is the solution's** when the application's calls are
  unchanged: a run option recorded in the report, as `proto=rdma` is under a kernel client.
  An RDMA path that needs a client API other than the traced application's is another
  workload.

- **The run-wide `--cache` defaults to the abstract's flags as written**, not to "buffered"
  (decided the same day): an abstract may carry `DIRECT` on some opens, and "buffered" would
  strip it. The value is named `per-open` (the user's choice, the same day, over the first
  name `as-written`, which said what the runner does, nothing, rather than what happens to
  the I/O: each open's own flags decide whether it goes through the page cache). A future
  `buffered`, stripping `O_DIRECT` from every open, would be a third value.

**Driver integration (the user's note, 2026-10-09).** Once `s3dlio` starts its runtime it
never stops it: the first object call builds a global multi-thread tokio runtime
(`global_rt_handle`, a `OnceCell` in its `src/s3_client.rs`, read 2026-10-09) and parks it on
a thread of its own until the process exits; importing its Python module also builds a Rayon
pool (`configure_thread_pools`, called at import). Until that first call nothing runs, and
the default build links none of it. So when `mlpstorage` drives `aeiou` (brief §8), each
phase is its own `aeiou` process by fork/exec, never the runner or `s3dlio` linked into the
Python process: a phase that touched object storage would leave those threads running
through every later phase, and the Python process's CPU, memory, and threads would land in
a phase's report, which counts its own process. A process per phase also gives each phase
its own limit checks and failure isolation, and matches the multi-host launch, which is
already a process per host.

**Built (2026-10-09), the first two parts of step 2.** *The runner's backend as two axes:*
`BackendKind` is a pair, `Api` (`sync`, `posix-aio`, `libaio`, `io_uring`, `mmap`) and
`Cache` (`per-open`, `direct`); the nine names are aliases of pairs and print as before,
and `BackendKind::of` refuses `direct` under `mmap`. Every `Backend` method takes the open
file's handle (`OpenFile`) in place of a descriptor, and `open` returns one: the handle is
still a descriptor (and, under `mmap`, its mapping), and an object handle joins it in step 3.
*Contract 0.6:* `backend` replaced by `api` and `cache` (the user chose both keys over `api`
alone, though no committed abstract declares `cache`: an application that opens every file
`O_DIRECT` says so in one key rather than on every open); the schema and the validator refuse
`direct` under `mmap`; `Workload(api=, cache=)`; every AST regenerated, every hash re-recorded,
no fingerprint changed. *The command line:* `--io-api` and `--cache` (default: the abstract's),
~~`--io-backend` kept as the pair in one name and refused with either.~~ *Later the same day:*
`--io-backend` and the nine names removed (above); a run prints and reports `io-api X  cache
Y` (`io_api`, `cache`, `io_api_declared`, `cache_declared` in the report), and the nine
constants remain only as the code's shorthand for the valid pairs. ~~Not yet: the endpoint
and protocol per namespace (`--endpoint`), the rest of step 2.~~

**Built (2026-10-09), step 2c: the protocol and the endpoint per dataset and namespace.**
*Contract:* the optional `protocol` (`posix`, `object`) on `files` and `regions` datasets and
on namespaces, folded into 0.6 since nobody had used 0.6 yet and no committed AST changes;
rule V17, namespaces sharing a root declare one protocol (datasets cannot share a root, V13).
The protocol is left out of a dataset id and of a namespace's resolved definition, in both
writers: where a corpus lives is not what it is. *The runner:* `--endpoint NAME=DIR`
(repeatable, layered like `--root`, the first list among the layered options: a higher
layer's list replaces a lower one's whole) on `aeiou run`, `aeiou datagen`, and
`aeiou-datagen`. Choices made here, ~~open to revision~~ confirmed by the user 2026-10-09: an endpoint places the name's *root
directory*, not a second `--root` (a path keeps its part below the root); a path falls under
the longest placed root that is a prefix of it, so a dataset rooted inside a placed namespace
moves with it unless placed itself (the KV abstracts' `kv/sys` inside `kv`); names sharing a
root share its place, and two endpoints for one root are refused; a `trace` node's paths
stay under `--root`. Every path goes through one function (`Endpoints::path`, and
`datagen.placement` in Python): both drivers' ops, the manifests, the emptiness check and
clean, the residency check, both writers. *Gaps, recorded:* the mount counters still sample
`--root`'s mount alone, so the run warns when an endpoint is on another file system (several
mounts in the report is a format change, left for when it is needed); `object` is refused by
`run` and both writers until the object engine (step 3), and endpoints are directories only.
A run placing the KV abstract's system prompts and its namespace in two directories issues
the dry run's fingerprint and leaves nothing under `--root` (`tests/run.rs`); the Python
writer and the runner agree on a placed container dataset (`test_formats.py`).

**Decided (2026-10-09, later, the user): the object engine is built on Apache
`object_store`, not on `s3dlio`.** The user's condition before believing an alternative was
viable: `s3dlio`'s author put a lot of effort into its performance, so the other library had
to be nearly as fast. Measured, it is faster.

*What was read (`s3dlio` at `4542f82`, 0.9.114, still its default branch's head on
2026-10-09).* Its S3 store is `aws-sdk-s3` (`GetObject` and `PutObject` with a `reqwest`
transport of its own, `src/reqwest_client.rs`); a range read is one SDK `GetObject` on the
caller's runtime (`get_object_range_uri_async`, `src/s3_utils.rs`), and a whole-object `get`
first issues a `HEAD` for the size (its range optimization, on by default). Its own documents
reach the same finding as below: `docs/performance/AWS_SDK_vs_Thin_Client_Benchmark.md`
measures the SDK at 3.6× the per-request latency of a thin client against the same server,
`docs/CUSTOM_S3_CLIENT_ANALYSIS.md` attributes about a third of client CPU to the SDK's
middleware, and `docs/enhancement/AWS_SDK_OVERHEAD_BYPASS.md` proposes a thin path for the hot
`GET` and `PUT`, status "Proposed — Not yet implemented". There is no run-time switch away
from the SDK (the user asked; none found). The one way around it is the build feature
`arrow-backend`, which puts `s3dlio`'s trait over `object_store` 0.13
(`src/object_store_arrow.rs`): it does not compile at `4542f82` (16 errors: the adapter still
has `Vec<u8>` and `&[u8]` where the trait moved to `Bytes`), its CI does not build it, and its
contributor notes call it experimental. The author's own route around the SDK ends at
`object_store`.

*How it was measured.* Scratch programs, not in the repository: a minimal in-memory S3 server
(`hyper`, one fixed pattern buffer, signatures ignored, requests counted per method) on cores
10–19, so the server's cost is small and the same for both; one client binary driving either
library through the same loop (`s3dlio::object_store::store_for_uri` with its default
features less Azure and GCS, against `object_store` 0.14.2 with the `aws` feature), pinned to
the stated cores with that many runtime threads (`S3DLIO_RT_THREADS` the same); keys
`obj-SIZE-i` over 4,096 objects; 1 s warm-up, 4 s measured; client CPU from `getrusage`; each
row run twice, libraries alternated. Loopback, plain HTTP/1.1, the WSL2 development box. Both
libraries sign every request, payload included.

| Case | `s3dlio` ops/s | `object_store` ops/s | CPU per op, `s3dlio` / `object_store` |
|---|---|---|---|
| 4 KiB range `GET`, 1 core, 1 in flight | 4.8–4.9k (p50 187 µs) | 12.2–12.4k (p50 75 µs) | 159–161 / 50–51 µs |
| 4 KiB range `GET`, 4 cores, 64 in flight | 13.6–15.8k | 56.1–60.0k | 201–233 / 54–57 µs |
| 4 KiB range `GET`, 8 cores, 128 in flight | 20.1–20.5k | 78.2–82.4k | 298–303 / 76–80 µs |
| 128 KiB `GET`, 4 cores, 64 in flight | 11.8–12.8k | 41.8–43.2k | 246–269 / 76–78 µs |
| 128 KiB `PUT`, 4 cores, 64 in flight | 10.7–11.7k | 26.4–26.6k | 287–314 / 134 µs |
| 8 MiB `GET`, 4 cores, 64 in flight | 5.4–5.5 GB/s | 7.2–8.1 GB/s | 5.4 / 3.7–4.1 ms |
| 8 MiB `PUT`, 4 cores, 64 in flight | 5.2 GB/s | 5.3–5.4 GB/s | 5.8 / 5.5–5.7 ms |

The large `PUT` is even because both hash every byte for the signature; `s3dlio` with
`S3DLIO_UNSIGNED_PAYLOAD=1` took 261 µs per 128 KiB `PUT` (`object_store`'s unsigned payload
was not measured). Weight, each library alone in a probe with `tokio`, counted as distinct
packages in `cargo tree -e normal`: `object_store` 131 packages, 7.3 MB stripped, a 32 s clean
release build on 20 cores; `s3dlio` 405 packages, 24.8 MB, 131 s (the runner today: 81).
Both bring `reqwest` 0.13, `rustls`, and `aws-lc-sys`, which is C; `s3dlio` also a second
`hyper`, a second `rustls`, and `ring`, and non-optional dependencies the runner would never
call (its data loaders and formats, `clap`, `rayon`, `ndarray`).

*Why it matters here.* The runner is a load generator: its cost per op decides how many
client cores it takes to drive a fast system, and the key metric of the A/B row (brief §5) is
client CPU per op. Against real storage at milliseconds the latency gap shrinks; the CPU gap
does not. Charging a library its own API is the rule (`s3dlio`'s `HEAD` before a whole-object
`get` would be a cost of the engine, not of the system under test); choosing the cheaper
library that issues the same requests is not unfair to anyone.

*The op stream does not depend on the library.* The engine issues the requests of the object
mapping above and nothing else; which crate formats them is the runner's, as `io_uring` against
`sync` is under the same abstract. Where a traced application's client adds requests of its
own (a `HEAD` for the size, a large `GET` split into ranges), they are that application's op
stream and belong in its abstract, whatever library the runner uses.

*Counter-arguments, weighed.* `s3dlio` is the WG's library, its author is in the WG, and an
engine built on it would have carried that standing; the WG may still ask for it, and the
`Backend` trait leaves room for a second object engine. `object_store` is pre-1.0 and has
changed its API between minor versions (0.13 to 0.14 moved the range calls to an extension
trait), so the version is pinned and an upgrade is a recorded change. `s3dlio` has things
`object_store` lacks (multi-endpoint stores, its range engine, NUMA and thread placement); none
is an op the mapping issues. The measurement is loopback against a server that does almost
nothing, which isolates the client's cost; it does not show either library's behaviour at
network latency or over TLS, which the step-4 server and a real store will.

*What changes.* The object engine: `object_store` with the `aws` feature (Azure and GCS
features when a workload needs them), behind the cargo feature `object`, on a `tokio` runtime
the engine builds on first use and sizes itself, so the default build stays tokio-free and
`CLAUDE.md`'s no-tokio invariant becomes "none in the default build" when the engine is built
(the change §3.32 anticipated). Steps 3 and 4 above become one: the engine measured against an
S3 server on the development box, never over `file://` (the runner's POSIX path stays on its
own engines, decided above). No PRs to `s3dlio` are on aeiou's path. The driver note above
holds for any library with a runtime: the engine's runtime lives as long as its process, one
process per phase.

**Built (2026-10-09, night), step 3 reads first** (the user's choice of order, test store,
and part size the same night: reads before writes; MinIO in CI, the last release with
binaries pinned by SHA-256; 8 MiB parts as a run option when the writes come). `object.rs`
behind the cargo feature `object`: `--endpoint NAME=s3://BUCKET[/PREFIX]` for a name declared
`object` (every one needs it, and a `posix` name may not fall under one); the driver routes
an object path to the engine before the run's API sees it; reads are one ranged `GET` each,
`stat` a `HEAD` (and a `LIST` for a directory), `readdir` a `LIST` with the delimiter; one
tokio runtime per process (`--object-threads`, default 2), each actor's op run with
`block_on` on its own thread, so the event loops refuse object names for now; `aeiou
datagen` writes object datasets as `PUT`s and 8 MiB multipart uploads, the manifest as one
`PUT`. MinIO tests: the objects' bytes equal the POSIX writer's files, and runs under
`sync`, `posix-aio`, and `mmap` reproduce the dry run's fingerprint. Choices made here, ~~open
to revision~~ confirmed by the user 2026-10-09: `--object-threads` defaults to 2 (the workers only drive connections; the
actor threads sign and build the requests), a directory `stat` costs a `HEAD` and a `LIST`
(a real client checking a prefix does the same), and `aeiou-datagen` (the Python container
writer) still writes to directories only. Not yet: ~~object namespaces and every write (V18,
`--object-part-size`),~~ ~~the event-loop bridge,~~ ~~the input-namespace manifest as an object,~~ and
a measurement against a real store (`runner/REFERENCE.md` §16).

**Built (2026-10-09, later the same night), step 3 writes.** Object namespaces, by the table
above: an `open` for writing gives the handle an upload; its writes fill a part of
`--object-part-mib` (default 8) from the payload, a full part goes out before the write
returns (the first starting the multipart upload), and the `close` sends one `PUT` when no
part went out, else the last part and the completion, so an object exists from its `close`
and the close is timed like any op; `unlink` is a `DELETE`, `rename` a server copy and a
`DELETE`, `mkdir` and `rmdir` send nothing. A namespace root is a prefix: rank 0 refuses a
non-empty one or, with `--clean-namespaces`, `DELETE`s its keys but a placed dataset's; the
namespace manifest is one `PUT` at it, which an `input` reader `GET`s for V14, V15, and
`--max-gap`. V18 in the three checkers (`validate.rs`, `schema/check.py`, the builder): no
`open` with `APPEND`, `RDWR`, or `EXCL`, no `ftruncate`, no `fallocate` on an object
namespace. Tested against MinIO with the shared KV pair: the writer's chunks (6 MiB, in two
5 MiB parts, then renamed) reach the dry run's fingerprint and equal a POSIX run's files byte
for byte, and the reader reads them back as its input to its own. *Choices made here, ~~open to
revision~~ confirmed by the user 2026-10-09:* (1) V18 sees ops, not offsets, which are positional expressions, so "a write not
sequential from 0" is refused at the write (`ESPIPE`, failing the run like a structural
check) rather than at `aeiou check`; `dry-run` could find it before a run, since the op
stream is fixed, but it knows no endpoints and would need the namespace of each path. (2)
`O_EXCL` is refused rather than sent as a conditional create (`If-None-Match: *`), which
`object_store` has for a single `PUT` but not for a multipart one; no traced application
uses it. (3) An `open` for writing without `CREAT` sends nothing, so it never fails with
`ENOENT` as POSIX would for a missing file, and `unlink` of a missing key succeeds, as S3
has it: answering as POSIX would cost a `HEAD` each, a request no S3 application makes
(charge the API only its own work). (4) `fstat` of a handle being written answers its
written size locally; `stat` of an object before its `close` is `ENOENT`. (5) The option is
`--object-part-mib`, after `--buffer-mib`, not the `--object-part-size` named above; it is
the run's only, and `aeiou datagen` keeps 8 MiB parts (a dataset's part size is not what a
run measures). Not yet: ~~the event-loop bridge,~~ `aeiou-datagen` to object stores, a
measurement against a real store.

**Built (2026-10-09, later still), the event-loop bridge.** `io_uring` and `libaio` take
object names. Each object op is split in three: `object::start` on the actor's thread does
what is local (the actor's tables, a write's payload into the part, the checks) and returns
either the result or a future that owns what it sends; the future runs on the runtime; and
`object::settle` applies its answer on the actor's thread (created and removed paths, the
upload back to its handle). A thread-per-actor API runs the three in a row with `block_on`,
as before. An event loop `spawn`s the future and parks the actor; the worker that finishes it
queues the answer and writes the loop's eventfd, the one barrier releases already write, on
which the loop keeps a read posted (a ring read, an AIO poll) while any object op is in
flight; the loop settles the queue on its own thread, so no worker touches an actor. The
recorded idea held (an eventfd the runtime signals, ops spawned rather than `block_on`), with
one change: the loop routes an object path before its engine sees the op, rather than making
the engine's inline calls non-blocking, so the ring and the AIO context stay object-free.
Tested against MinIO: the 20 MiB dataset under both loops, the listing walk with its short
reads under `io_uring`, the shared KV writer under `io_uring` (its objects byte-equal to a
POSIX run's files) and its reader under `libaio`, each to the dry run's fingerprint. *Choices
made here, ~~open to revision~~ confirmed by the user 2026-10-09:* (1) under a loop the workers build and sign the requests as
well as drive the connections, so `--object-threads` bounds the CPU object ops get, as it
would in an asynchronous application; under `sync` the actor's thread signs, as before. (2) A
read's buffer under a loop is plain heap memory owned by the op while in flight, from a FIFO
that grows to `--buffer-mib` like the ring's pool; it is not aligned, since nothing reads a
socket with `O_DIRECT`. (3) An upload is lent to its future while a part goes out, so a
second op on it meanwhile (two actors writing one object at once, which V18's order from 0
could meet only by timing) fails with `EBUSY`, under every API; before, `sync` serialized
them on a lock. (4) The report counts no object op in the ring's in-flight peak.

**Decided (2026-10-09, after the bridge), an API per protocol; contract 0.7.** The bridge
showed that `--io-api io_uring` never named what carried an object op: the S3 requests went
through `object_store` over tokio and `epoll` on the runtime's workers, the ring only waking
the loop, and the run printed `io-api io_uring` all the same. One API flag pretended there
was one choice where there is one per protocol (no POSIX API carries a `GET`, no S3 client a
`pread`). Decided with the user:
- **A flag per protocol, its value the API on it.** `--posix sync|posix-aio|libaio|io_uring|mmap`
  replaces `--io-api` (renamed outright, no alias); `--s3 blocking|async` names how the
  application's S3 client uses the store: a thread blocked per request (boto3) or an async
  client with many requests in flight per thread (aiobotocore, the CRT). The value names the
  client's behaviour, which changes the requests the store sees; the library that emulates
  it (`object_store`) is the runner's tooling, printed and recorded, never chosen. `--posix`
  with no value fails loudly (the short form chosen over `--posix-api`).
- **Dialects are protocols.** `protocol: s3` replaces `protocol: object` (and `azblob`, `gcs`
  come with their engines, never before: no flag or value for an engine not built). Their
  write paths differ (multipart, put-block and commit-list, resumable uploads), and §3.65's
  mapping table and V18 are S3's. An endpoint's scheme must agree with its name's protocol.
- **`--cache` is POSIX-only.** Given to a run with no POSIX name it fails loudly; in a run with
  both it applies to the POSIX names and says nothing. The same holds for `--posix` and the
  ring knobs; `--s3` and the object options fail loudly with no S3 name.
- **Scheduling follows the APIs.** A run uses event loops when any protocol's API is
  event-driven (`io_uring`, `libaio`, `async`), a thread per actor otherwise; a run with both
  protocols whose APIs disagree (`--posix sync --s3 async`, `--posix io_uring --s3 blocking`)
  is refused until a traced application shows that mix. A run of S3 names alone under `async`
  waits on its eventfd alone, with no ring.
- **The abstract declares both, mirroring the flags (contract 0.7).** Its top-level keys
  `posix`, `cache`, and `s3` replace `api` and `cache`; an undeclared one defaults (`sync`,
  `per-open`, `blocking`); a run under another value of a declared one is another workload,
  as for the POSIX API since §3.48. The report and the output name each: `posix`, `cache`,
  `s3`, each with its declared value.

*Built the same night.* Contract 0.7 in the schema, `check.py`, the builder
(`Workload(posix=, cache=, s3=)`), and the runner; every committed AST regenerated, no
fingerprint changed; rule V19 in the three checkers; `--posix` and `--s3` on `aeiou run`
(fixed options, as the API always was); the report's `posix`, `cache`, `s3` and their
`_declared` twins; the run's line `posix X  cache Y  s3 Z  root …` naming only the protocols
with names, and `s3 engine: object_store 0.14.2 on tokio (epoll sockets)  runtime threads N`.
Tested against MinIO: S3 names alone under `blocking` and `async`, the KV namespace written
under `async`, and a mixed run (the system prompts in directories, the chunks in the bucket)
under `--posix io_uring --s3 async`, each to the dry run's fingerprint, and the refusals.
*Choices made here, ~~open to revision~~ confirmed by the user 2026-10-09:* (1) a `trace` node counts as a POSIX name (its files
are under `--root`), and an abstract with no dataset or namespace at all is POSIX; (2) S3
names alone under `async` run on loops with no ring (`uring::WaitIo`, a `poll(2)` of the
loop's eventfd), so such a run needs no io_uring in the kernel; (3) "POSIX options" refused
without a `posix` name are `--posix`, `--cache`, the ring knobs, `--aio-depth`, and the
`mmap` knobs, listed together in one message; (4) the report's fields for a protocol
without names are `null`, not absent; (5) the old flag and key are gone, not aliased (no
users).

~~**Open for the user.** Whether the axes and their owners are right; whether mixed protocols
are designed in now (proposed) or deferred; the contract change; which real application's
object I/O the first object abstract stands for, since under CLOSED a run over S3 of an
abstract traced on POSIX is another workload (candidates: `s3torchconnector`, `s3dlio`'s own
data loader, or one the WG names); and whether the POSIX-engine crate shared with `s3dlio`
is worth offering.~~

**Every build has the object engine (decided by the user 2026-10-16).** `object` is a default
feature, so every build of `aeiou`, the release tarball's included, reads and writes S3 and a
user who wants S3 needs no other binary ("minimize the surprise factor"). This revises the
choice above to keep the default build tokio-free: tokio and `object_store` are now in every
build, the runtime still built only on the first object call, so a run that touches no object
store starts no tokio thread. `--no-default-features` builds without the engine, which refuses an
object endpoint; CI tests that refusal, and its main runner job, `runner-tests`, now runs every
test the engine has against MinIO (until then only `runner-object` did). The cost: the binary grows from about 45 MB to about
88 MB, and its libraries stay libc, libm, and libgcc_s (the S3 client is rustls and ring, no
OpenSSL). The release's smoke test refuses a binary that carries the no-engine refusal. The same
day, for the same reason, `formats` (the container writers, §3.70, off by default until then) became
a default feature too: a plain `cargo build` writes every container format, as the release
binary always did.

### 3.66 A catalog of workloads: `workloads/<category>/…/<mechanism>/cases/<case>`, and a dictionary of the attributes it is searched by (proposed 2026-10-10; the shape of it decided the same day, the details open)

**The question (the user's).** Traces and the abstracts fitted to them are accumulating, and
what belongs to one workload is spread over three directories that share only a name: the
authoring script in `builder/abstracts/`, its AST in `schema/examples/`, and the capture kit
with its metrics, fitted parameters, logs, and README in `builder/traces/`. The user asked for
a taxonomically structured repository at the root, `workloads/{training,checkpointing,vdb,
kvcache}/<name>/*`, holding everything curated for a workload (the abstract, the traces or
pointers to them, the analysis, the READMEs), with categories growing over time (IO500 and
others), and short names that let a reader pick the directory to look into for a workload of
a given shape.

**What exists.** Twelve scripts emit fourteen ASTs (`train_stream_shards.py` emits the Parquet
and TFRecord ASTs, `kv_cache_shared.py` the writer and its reader). Ten trace kits, each a flat
directory. The names do not line up: one script, two ASTs, one kit (`kv_cache_shared`), and one
kit holding four loads distinguished by file-name prefixes (`kv_cache_serving`: the ShareGPT
replay, `agentx`, `agentx-6gib`, `agentx-odirect`, each with its own logs, metrics,
tolerances, and fitted parameters). The pairs `aeiou-trace compare --judge` is pinned on are a
hand-kept list in `builder/tests/test_trace.py`. Three smoke parameter files sit in
`schema/examples/params/`. Nothing records a workload's attributes except prose in READMEs and
`ABSTRACTS.md`.

**What the user decided in discussion (2026-10-10).**

- **The unit on disk is the mechanism; the unit searched for is the case.** An abstract and a
  workload are not one to one in either direction. One abstract reproduces many workloads:
  `kv_cache_serving` is the ShareGPT chat and the AgentX sessions with different parameter
  files, and `train_small_files` would be ImageNet JPEGs or audio clips read one file per
  sample. One workload can need several abstracts: the shared KV store is a writer and a
  reader. The runner already says what a reproducible workload is: an AST and a parameter set
  (with the seed and the GPU count), which is what the fingerprint covers. So a directory holds
  one mechanism (one authoring script, the AST or ASTs it emits, and what was learned about
  the application's I/O), and each workload a user would ask for is a **case** inside it: a
  parameter set, its README, its attributes, and the captures taken under that load. The
  catalog has one row per case.
- **Directories are named for the mechanism, never the use case.** Image and audio
  classification over one file per sample through a map-style loader are one workload to
  storage; only the parameters differ. Named for use cases, `training/` would grow a
  directory per modality and per dataset; named for mechanisms it grows only with new I/O
  shapes, and the use cases become cases and tags. This is already how the KV names read
  (a technology combination). Training gains a level for the access mode (`map/`,
  `stream/`), the first split storage feels.
- **Categories:** `training`, `inference` (model loading belongs here, not under
  checkpointing), `checkpointing`, `vdb`, `kvcache`, and `synthetic` for benchmarks that are
  not applications (IO500 and the like). A synthetic case reproduces a benchmark's defined
  operations, so it is not traced and is never presented as an application's I/O. This keeps
  §3.46 clear: DLIO emulates applications and is not an abstract's source; IO500's op stream
  is the benchmark itself.
- **AgentX is a case of the local-disk mechanism**, not a mechanism of its own: same AST,
  other parameters. Its `O_DIRECT` run is a sibling case, not another capture of the same
  case, because under CLOSED buffered and direct are two workloads (§3.48).
- **A dictionary of attributes, enforced.** Every attribute a case is labeled with is defined
  in one file with its meaning and allowed values; a case may use only defined attributes and
  values, and CI fails otherwise. Without that the catalog fills with near-synonyms (`image`,
  `images`, `vision`, `cv`), each tagging a part of the cases, and a search misses the rest.
- **Captures stay pointers when large, and the SNIA AIWD TWG traces stay out entirely**,
  pointers included, until the user says permission to point to them (or to traces like them)
  has been given.

**Proposed layout (proposed, not yet decided).**

```
workloads/
  README.md                 hand-written: how the tree is organized, how to add a mechanism or case
  attributes.toml           the dictionary (below)
  CATALOG.md                generated: one row per case, and the glossary rendered from the dictionary
  <category>/[<access>/]<mechanism>/
    README.md               the mechanism: what the application does, what storage sees, what the
                            traces taught, known gaps (today's kit README, less the per-load parts)
    mechanism.toml          curated attributes that hold for every case (application, format, …)
    <mechanism>.py          the authoring script (from builder/abstracts/)
    <ast>.ast.json          the AST(s) it emits, generated beside it, never edited by hand
    kit/                    capture and fitting scripts its cases share (train.py, serve.sh, fit.py, …)
    cases/<case>/
      README.md             the load: which job, which data, which scale, why it is a case
      case.toml             curated attributes of this case; which AST(s) and parameter file(s) it runs
      params.json           the parameter set (or one per AST: writer.params.json, reader.params.json)
      shape.json            generated: the measured attributes and their raw numbers
      captures/<capture>/
        README.md           the command lines, host, kernel, file system, versions, date
        capture.toml        the raw trace: URL or "not retained", bytes, SHA-256; the inputs' sources,
                            licences, and SHA-256 (ShareGPT, SIFT1M, the model); the expected verdict
        trace.metrics.json, trace.tolerances.json, fitted.params.json, *.log
```

- `capture.toml` carries the verdict `compare --judge` is expected to give (accepted, not
  accepted with the `DESIGN_REVIEW` section that says why, or nothing judged), so the test walks
  the tree instead of a list kept in `test_trace.py`.
- Size rule: no file over about 1 MiB in the tree; a raw trace is a pointer with its SHA-256,
  or "not retained" when it was a scratch capture (every one so far).
- The AST's `name` (an `ident`, `^[a-z_][a-z0-9_]*$`) is the mechanism directory's last part
  with underscores; names stay unique across the tree, checked. A writer/reader pair keeps a
  suffix (`lmcache_shared_fs`, `lmcache_shared_fs_reader`).
- A mechanism with no case yet (the two untraced stream formats, HDF5) still has a catalog row:
  its defaults, marked `defaults`, so the catalog shows the gaps.
- `schema/examples/` goes away with its smoke parameter files, which become cases or move
  into the mechanism's `params/` when they are test fixtures rather than workloads.

**The mapping (proposed names).**

| Today: script → AST(s); kit | Mechanism | AST name(s) | Cases, from |
|---|---|---|---|
| `train_small_files`; kit `train_small_files` | `training/map/file-per-sample` | `file_per_sample` | `imagefolder-jpeg` (the 2026-10-01 trace; 3,200 JPEGs) |
| `train_large_samples`; kit `train_large_samples` | `training/map/npy-per-sample` | `npy_per_sample` | `npy-8mib`, `npy-140mib` (the two traces) |
| `train_map_hdf5` | `training/map/hdf5` | `hdf5_map` | none yet (defaults) |
| `train_stream_shards` → `train_stream_tfrecord`, `train_stream_parquet` | `training/stream/tfrecord`, `training/stream/parquet` | `tfrecord_stream`, `parquet_stream` | none yet (defaults) |
| `ckpt_write_dcp`; kit `ckpt_write_dcp` | `checkpointing/dcp-save` | `dcp_save` | `toy-6-items` |
| `ckpt_restore`; kit `ckpt_restore` | `checkpointing/dcp-load` | `dcp_load` | `mixed-9-items`, `small-last`, `name-order` (validation cases) |
| `model_load`; kit `model_load` | `inference/safetensors-load` | `safetensors_load` | `gpt2-random` (today's `model_load.synthetic` params) |
| `vdb_search_ivf`; kit `vdb_search_ivf` | `vdb/faiss-ivf-ondisk` | `faiss_ivf_ondisk` | `sift1m-ivf1024-pq32` |
| `vdb_search_diskann`; kit `vdb_search_diskann` | `vdb/diskann-search` | `diskann_search` | `sift1m-nocache`, `sift1m-cache` |
| `vdb_build_diskann`; kit `vdb_build_diskann` | `vdb/diskann-build` | `diskann_build` | `sift1m-13-shards` |
| `kv_cache_serving`; kit `kv_cache_serving` | `kvcache/lmcache-local-disk` | `lmcache_local_disk` | `sharegpt-chat`, `agentx`, `agentx-6gib`, `agentx-odirect` |
| `kv_cache_shared` → writer, `…_reader`; kit `kv_cache_shared` | `kvcache/lmcache-shared-fs` | `lmcache_shared_fs`, `lmcache_shared_fs_reader` | `sharegpt-chat` (writer and reader captures) |

The kit of `kvcache/lmcache-shared-fs` runs the replay scripts of `lmcache-local-disk`; its
README points there rather than copying them.

**Case kinds.** The three `dcp-load` captures are not workloads anyone shops for: they are
small corpora chosen to prove the abstract mimics `dcp.load` call for call. A case says which
it is: `validation` (a capture-scale load that proves the mechanism) or `reference` (a load at
a scale meant to be run). Every case today is `validation`; reference cases come with the
first parameter sets at real scale (and a WG's reference parameters, when they come, are
reference cases that say so, per `PROJECT_BRIEF.md` §8).

**The dictionary (proposed, not yet decided).** `workloads/attributes.toml`, one table per
attribute:

```toml
[attribute.modality]
meaning = "The kind of data the job reads, as its users name it."
source  = "curated"           # curated | derived | measured
level   = "case"              # mechanism | case
required = false
[attribute.modality.values]
image = { meaning = "Still images (JPEG, PNG, …).", search = ["vision", "cv"] }
audio = { meaning = "Sound clips or streams." }
```

`search` lists words a searcher may type; it changes nothing any tool accepts (it is not an
alias of the kind §3.65 removed from the command line; it only finds a case). Values are defined before use;
adding one is a one-line change, renaming or removing one is a recorded decision. The catalog's
glossary is rendered from this file, so a reader sees what "small" or "random" means where they
search.

Three sources, and the split matters more than the list:

1. **Curated** (only a person knows them). Mechanism level: `category` (required),
   `application` (the library traced: torchvision, NumPy, `torch.distributed.checkpoint`,
   safetensors, FAISS, DiskANN, vLLM with LMCache), `format` (jpeg-files, npy, hdf5,
   tfrecord, parquet, dcp, safetensors, faiss-ivf, diskann-index, lmcache-chunks), `sharing`
   (one job; a writer and a later reader; several hosts at once). Case level: `modality`,
   `kind` (validation | reference, required), and, as free text rather than tags, the model
   and dataset named in its README.
2. **Derived** from the AST and the case's parameters, statically: the declared APIs and
   protocols (`posix`, `cache`, `s3` keys), whether it writes (namespaces) and reads its own
   output (`input`, `same_run`), the file-size class and file-count class of each dataset at the
   case's parameters, the `trace` node's presence, and `status` (untraced | traced | accepted |
   not accepted, from the captures' verdicts).
3. **Measured**: the storage shape, from `aeiou dry-run --metrics-json` of the case (its AST and
   parameter file, seed 1, its GPU count), written to `shape.json` and re-derived in CI so a
   label cannot drift from the abstract. Measured from the abstract, not the trace, because the
   trace of an `mmap` application sees almost nothing (`model_load`'s trace counts 12.5
   metadata calls per read; `vdb_search_ivf`'s has 12 calls in all), mechanisms without a
   trace need a shape too, and the dry run is deterministic. The traces stay as the evidence
   in `captures/`.

Proposed measured attributes, each shown as its class and its raw number. Sizes are
byte-weighted medians (half the bytes move in runs at least this long): a count-weighted
median is wrong here, since `np.load`'s many tiny header reads make the 140 MiB samples look
like 112-byte runs. The column on the right places each traced workload, from its
`trace.metrics.json`, as a check that the thresholds separate what should be separate:

| Attribute | Classes | Where the traces fall |
|---|---|---|
| `run_length` (read; write) | tiny < 16 KiB, small < 1 MiB, medium < 32 MiB, large ≥ 32 MiB | tiny: DiskANN search (4 KiB). small: ImageFolder (128 KiB), FAISS IVF (640 KiB). medium: KV (2–3 MiB), `npy` 8 MiB, `dcp.load` (8–16 MiB). large: `npy` 140 MiB (128 MiB), `dcp.save` (64 MiB write), DiskANN build (448 MiB read, 80 MiB write) |
| `request_size` (read; write) | the same classes | differs from runs where it matters: ImageFolder asks for 1 MiB and gets 128 KiB files; DiskANN build writes 8 KiB requests in 80 MiB runs |
| `read_share` (bytes) | read-only = 1, read-mostly ≥ 0.8, mixed ≥ 0.2, write-mostly > 0, write-only = 0 | read-only: training, restore, search, the KV reader. read-mostly: AgentX (0.86–0.87). mixed: ShareGPT serving (0.67), the KV writer (0.67), DiskANN build (0.77). write-only: `dcp.save` |
| `metadata` (metadata calls per data call) | none < 0.01, low < 0.5, medium < 2, high ≥ 2 | none: DiskANN. low: `dcp.load` (0.2), `npy`. medium: ImageFolder (1.5), `dcp.save` (0.8), IVF (0.7). high: every KV load (2.2–3) |
| `reads_own_writes` | yes when any read follows a write of the same block | yes: KV serving, DiskANN build. no: the rest |
| `reread` (block reads that read a block again / all block reads) | none < 0.05, some < 0.5, repeated ≥ 0.5 | repeated: training at two epochs (0.50–0.56), every KV load (0.64–0.89), DiskANN build (0.84). some: `dcp.load` (0.16–0.22, its buffer refills ranges it has read), DiskANN search (0.26–0.29). none: `dcp.save`. Training's value is its epoch count, a parameter, which is why this is measured per case |
| `skew` (share of accesses on the busiest 1 % of blocks) | uniform < 0.1, skewed ≥ 0.1 | every trace so far is uniform (0.01–0.06); kept for the KV and VDB loads expected to be skewed |
| `concurrency` | the actor count of the case | from the AST at the case's parameters |

**Tooling (proposed).**

- `aeiou-build` writes each AST beside its script and `--check` walks `workloads/**/*.py`; it
  refuses two ASTs with one name.
- A new tool `aeiou-catalog` (convention of `PROJECT_BRIEF.md` §8; a man page like the others):
  `check` validates every `mechanism.toml`, `case.toml`, and `capture.toml` against
  `attributes.toml`, re-derives `shape.json` (it needs the runner binary, so it runs in a CI job
  that builds one) and `CATALOG.md`, and fails on drift; `write` regenerates them.
- The tests that name AST paths (`golden.rs`, `run.rs`, `metrics.rs`, `limits.rs`,
  `uring_knobs.rs`, `report.rs`, `coord.rs`, `options.rs`, `object.rs`, `usage.rs`,
  `test_builder.py`, `test_params.py`, `test_trace.py`), CI and the release workflow, the kit
  READMEs' command lines, the man pages, and every document citing `builder/abstracts/`,
  `builder/traces/`, or `schema/examples/` follow. The files move with `git mv` so their
  history follows.
- Renaming the ASTs changes each AST's SHA-256 (the golden hashes are re-recorded) and the
  `abstract` field of every parameter file. It changes no fingerprint (a per-op hash covers the
  actor, loop indices, op, file, offset, and length, §2.1, never the abstract's name) and no
  dataset id (the SHA-256 of the resolved dataset definition and payload, `payload.rs`
  `Manifest::normative`). The golden test confirms both.

**Open for the user.**

1. The names in the mapping table, mechanisms and cases, and renaming the ASTs to match
   (recommended: there are no users, so no aliases) rather than keeping today's AST names
   under the new directories. *Decided 2026-10-10: the AST names match the directory names.*
2. Splitting `train_stream_shards.py` into one script per format (recommended: their reader
   protocols differ, Parquet reading a footer and then row-group ranges, TFRecord whole shards
   front to back; each script repeats the ~20 lines of loader they share) or one
   `training/stream/shards` mechanism with two ASTs. *Decided 2026-10-10: two directories,
   the shared lines duplicated and kept minimal.*
3. Case kinds `validation` and `reference`, and `defaults` rows for mechanisms with no case.
   *Decided 2026-10-10: a mechanism with no case yet still gets its directory and its
   `defaults` row.* The kinds are under discussion: the user asked why a validation case
   differs from a reference one when an abstract grows with `--gpus`, and proposed "test
   case" and "use case" if it does. What grows with `--gpus` is the number of actors (every
   actor template defaults to `count = gpus`), so the op count; what a case's parameters fix
   is the application's data and configuration: the state dict's tensors and their sizes,
   the corpus's file sizes and count, the model's KV chunk bytes, the index's geometry. More
   GPUs never turn the nine-item state dict traced for `dcp.load` into a real model's
   checkpoint. So the AST is the same and the difference is the provenance of the values: a
   configuration chosen small enough to trace, or one describing a real job. The smoke
   parameter files of `schema/examples/params/` are a third thing, test fixtures, and stay out
   of the catalog. *Decided 2026-10-10: the kinds are named **example case** (a configuration
   small enough to trace, "example" saying it is not a job anyone runs) and **use case** (a
   configuration describing a real job).*

   **What scales with `--gpus`, and the definition it gives a use case (proposed).** The user
   separated two things: the load an abstract imposes grows with `--gpus` (a fact: each actor
   template defaults to `count = gpus`), and whether the data must grow with it (a need,
   which depends on the workload). The system allows both: `gpus` is a reserved parameter any
   expression may use, a dataset's `count` included, and `aeiou datagen` takes `--gpus` as a
   fixed option for that reason; no abstract uses it today. Sizes fall in four kinds:
   (a) the application's data, fixed whatever the GPU count: a corpus, a model's tensors, an
   index; (b) each GPU's share of it, a function of `gpus`: a rank's checkpoint shard, how fast
   an epoch is consumed (the latter already is, through the position formula); (c) what the
   run creates, which grows with the load: KV chunks, checkpoint files per step; (d) a run
   rule's override of (a): the dataset sized to five times the clients' DRAM (`PROJECT_BRIEF.md`
   §5), which follows the hosts, not the application, and is the WG's (`PROJECT_BRIEF.md` §8; its
   reason is in §3.67). This gives a test
   for a use case: its parameters hold ~~at any `--gpus`~~ at every GPU count the case declares
   (*corrected 2026-10-10, §3.67: a use case declares its GPU counts, one, a list, or a range,
   as MLPerf Storage's CLOSED division fixes the GPU count by model size, so no one checkpoints
   an 8B model from 2,048 GPUs*), every size being an application fact or an expression of
   `gpus`. An example case may be pinned to the GPU count it was traced at
   (`dcp.load` on 2 ranks), and its `case.toml` says so. **Finding:** the checkpoint
   abstracts fail that test today. `ckpt_write_dcp`'s `item_bytes` is documented as "1/G of
   each tensor" (`ABSTRACTS.md` §3) but is a fixed array, so `--gpus 64` writes 64 times the
   bytes of one rank's shard of a checkpoint whose size is fixed by the model; `ckpt_restore`
   inherits the same. The fix is a parameter of tensor sizes divided by `gpus` (fully
   sharded) or by `gpus / replicas`; unbuilt, to be done with the first checkpoint use case.
4. The dictionary's starting attributes, which are required, and the thresholds in the table
   (proposed from the spread of the eighteen captures we have; a threshold is a judgement and
   moves cases between classes). *Decided 2026-10-10: fine for now; measured classes are
   computed, so moving a threshold later re-labels every case by regeneration.*
5. ~~The shape measured from the abstract's dry run at the case's parameters. A reference case at
   real scale may take long to dry-run; such a case would name a smaller parameter set for its
   shape, at the cost that some attributes (`reread`, `skew`) change with scale.~~ *Decided
   2026-10-10 (the user): measured at the smallest scale at which the numbers are stable, so
   that the reported numbers carry as little statistical error as the measurement needs and no
   more cost.* What that means in the tool (proposed): a case names its length parameter (the
   one that only makes the run longer: `steps`, `queries`, `requests`), never one that
   changes the pattern (epochs, file sizes, a pool's size); ~~`aeiou-catalog` dry-runs the case
   at growing lengths (doubling) and at a few seeds, and stops at the first length where every
   measured number moves less than its class tolerance between one length and the next and
   between seeds; `shape.json` records that length and the spread.~~ A number that keeps moving
   with length (a reuse distance bounded by the run, `reread` under a growing working set) is
   a property of the length, and the case says what length it is reported at. *Revised in
   the build, 2026-10-10 (open to revision; the evidence is in "Built 2026-10-10: the catalog"
   below): the shape is measured at the case's own length, and the smallest length that still
   reproduces it is where `check` re-measures, so the rule serves the cost and no short run
   stands for the case.*
6. ~~A new tool `aeiou-catalog` rather than a flag of `aeiou-build`.~~ *Decided 2026-10-10:
   a separate tool. The user's reason: building is a compiler's problem (a script to an AST),
   the catalog a data-management problem (curated records, their vocabulary, their
   consistency).*
7. ~~The order of work: the user asked for this before `aeiou-datagen` to S3. Proposed as two
   commits, the move (layout, names, tests, documents) and then the dictionary, `shape.json`,
   `aeiou-catalog`, and `CATALOG.md`.~~ *Left to me by the user 2026-10-10: two commits, the
   move (layout, names, tests, documents) first, then the dictionary, `shape.json`,
   `aeiou-catalog`, and `CATALOG.md`; before `aeiou-datagen` to S3.*

**Built 2026-10-10: the move (the first of the two commits).** `builder/abstracts/`,
`builder/traces/`, and `schema/examples/` are gone; every file moved with `git mv` into the
tree of the mapping table, and every AST renamed for its directory and regenerated beside its
script (`aeiou-build` writes there by default). Each AST's hash changed (its name and its
provenance's script name); the golden test re-records the hashes and confirms every
fingerprint unchanged. `train_stream_shards.py` is split into `tfrecord_stream.py` and
`parquet_stream.py`, each repeating the shape. The tests find an AST by name anywhere under
`workloads/` (`runner/aeiou/tests/common/mod.rs`, `ast_path` in each Python test file);
every parameter file under the tree is checked against its abstract, and must sit under its
abstract's mechanism; CI builds the scripts outside `kit/` and checks every `*params.json`;
the release tarball ships the tree's ASTs, parameter files, and READMEs. A README per case
and for the three mechanisms without one; `workloads/README.md` with the layout and a
hand-kept table until `CATALOG.md` is generated. *Choices made in the build, open to
revision:* (1) a capture is named by its date when the case has one, by what distinguishes it
otherwise (`replay-2gib-pool`, `replay-6gib-pool`); (2) a capture holds `fitted.params.json`
only when its fit differs from the case's `params.json` (the AgentX replays; for every other
capture the case's set is the fit); (3) the 6 GiB AgentX replay is a capture of `agentx`, not
a case: the engine's pool is a condition of the capture (§3.64), the abstract's parameters
follow it; (4) the AgentX case's `params.json` is the fit to the whole corpus (393 sessions,
32 MiB chunks), a candidate use case left unsettled in its README (`keep` not measured, never
traced at that scale), every other case an example case; (5) the verdicts stay in
`builder/tests/test_trace.py` (`CAPTURE_PAIRS`) until `capture.toml` holds them; (6) the
`provenance.ast_sha256` of the two safetensors parameter files is left as recorded: it says
which AST the tool read when it wrote them, and it was already older than the current one.

**Built 2026-10-10: the catalog (the second of the two commits).** The dictionary
(`workloads/attributes.toml`: four curated attributes of a mechanism, two of a case, four
derived, nine measured), the records (`mechanism.toml` ×13, `case.toml` ×16,
`capture.toml` ×17), `shape.json` per case and `shape.defaults.json` for the three
mechanisms with no case, `workloads/CATALOG.md`, and the tool `aeiou-catalog check|write`
(`builder/aeiou/catalog.py`, `builder/REFERENCE.md` §8, `man/aeiou-catalog.1.md`). The verdicts
moved from `CAPTURE_PAIRS` in `test_trace.py` into the capture records, which the test now
reads. CI's integration job runs `aeiou-catalog check` with the runner; the builder's tests run
it with `--no-shapes`.

*What building the shape taught (the revision of item 5 above).* The doubling search was
built first and gave wrong labels. Shortening `steps` drops a training job's later epochs:
ImageFolder's `reread` came out 0.00 at 14 of its 400 steps, where the two epochs give 0.50,
and 7 and 14 steps agreed with each other, so the search stopped there. On a fixed dataset a
search's reuse grows with the run: DiskANN over SIFT1M (the `sift1m-nocache` parameters)
re-reads 0.2 % of its block reads at 8 queries, 1.6 % at 32, 7 % at 128, 24 % at 500, 59 % at
2,000, 88 % at 8,000, 97 % at 32,000, and the 2 GiB PQ load holds the byte-weighted read
request size at 1.8 GiB until about 8,000 queries. A test of agreement between successive
lengths, even over three of them, accepts the short end of such a trend, where it moves by
less than its tolerance per doubling. So no length below the case's is a measurement of the
case, and the numbers are the case's own: the median of seeds 1, 2, and 3 at its parameters.
The user's rule then governs the cost: `write` halves the length while the medians stay the
case's within their tolerances and records the smallest such length, and `check` re-measures
there (the full AgentX corpus, 98,827 requests, takes five minutes on three cores once, when
written). The traced example cases are small, and some carry seed spreads larger than a
tolerance at their own length (the ShareGPT replay's `read_share` moves by 0.07 over three
seeds at 300 requests); the catalog names those numbers rather than lengthening the run,
which would describe another workload. *Choices made in the build, open to revision:* (1)
the above; (2) KV cases measured in blocks of one chunk (`metrics_block` in `case.toml`, the
shares those at 4 KiB, the work far less); (3) a training job's `steps` is never a case's
`length`; (4) a tolerance may give a relative and an absolute bound, either enough
(`metadata`: 5 % or 0.005, since a ratio near zero, DiskANN's 5·10⁻⁵, moves by more than 5 %
with no class seeing it); (5) a case of two runs is two rows, and the derived attributes are
per run (the shared-FS reader has inputs, the writer does not); (6) `gpus`, `length`, and
`metrics_block` live in `case.toml` as what the case runs, not in the dictionary as what it
is; (7) a capture's GPU count is its case's first; (8) the dictionary keeps the classes of
the proposal and leaves out `sharing` (the derived `writes` and `inputs` say it for every
case so far) and `concurrency` (an actor count is a parameter, read from the case); (9)
`status` is the best verdict over a run's captures, so `agentx` reads accepted on its 6 GiB
replay though its 2 GiB one is not.

### 3.67 Client caches, the 5× rule, the harness's topology, and fidelity for small datasets (discussed and decided 2026-10-10; the fidelity tooling on the backlog)

**The question (the user's).** The dataset sizing rule (`PROJECT_BRIEF.md` §5: capacity at
least five times the clients' total DRAM) exists so the workload's randomness overwhelms the
clients' page caches and the load reaches the storage. `O_DIRECT` and object APIs do not go
through the page cache, so for them the rule seems to add only a longer run. Should it bend
for them, and how? The discussion that followed corrected two of my assumptions and turned up
a question the rule was never meant to answer.

**Caching inside the solution is fair (decided).** By the interposition test (§5 of the brief)
and the WG's long-standing line that the POSIX layer is the boundary between benchmark and
storage solution, a cache in the kernel's file-system client, in a user-space object client,
or in a shim under an unmodified application is the solution's. I had proposed checking a
measured client hit ratio against a target; that would prohibit caching the solution is
entitled to, and would also penalize hits the application really gets (KV reuse, DiskANN's hot
nodes, a restore of what was just written). Struck: the hit ratio is **reported**, never
capped. The WG's stance is the same: any cache in the storage architecture is fair game, its
cost visible in the rack units and power a submission reports (MLPerf cannot ask for prices).

**Why the 5× rule exists (corrected).** I first argued the rule protects against hits that
exist only because a benchmark dataset is smaller than the corpus it stands for. Both halves of
that were wrong:

1. **Small datasets are real.** At a research institution many researchers share one storage
   system, each with a small dataset, and when the others are away one job may have every
   cache in the architecture to itself. Dataset size is a fact of the use case (kind (a) of
   §3.66), and a use case whose working set fits in cache is legitimate. The busy-term version
   of that site, many small jobs competing for the same caches, is a different workload: a mix
   of cases run concurrently, which aeiou cannot express yet and must eventually (the user:
   critical, not yet).
2. **The rule's reason is the client, not the dataset.** The benchmark's clients are not
   production machines. A real node carries at most eight GPUs (NVIDIA's practical limit),
   but a submitter emulates as many GPUs per client as it can, 64 on one host being a
   realistic choice and an unrealistic node, so that few clients load the system under test.
   The number of clients is then the submitter's choice, and each brings its DRAM: without the
   rule, adding clients adds cache and raises the score. Tying the dataset to the clients'
   total DRAM keeps the saturation point a function of the GPU count, not the client count.

**What the rule does not cover.** Only memory. Client count also sets NIC bandwidth, client
CPU, and the kernel client's own limits (one NFS client per host, its slots and connections).
The WG relies on self-interest there: a client-bound submission understates its storage, so a
savvy submitter rebalances (the user; MLPerf's accelerator-utilization threshold catches part
of it). Accepted as the WG's answer.

**Fidelity: what the rule costs small workloads.** The rule serves **comparability** (a score
must not move with the harness's choices). It works against **fidelity** (a result should
predict what a real deployment of the workload sees from storage), and for small datasets the
distortion runs the other way from the one the rule guards against: at eight GPUs per real
node, a real cluster has about eight times the client DRAM per GPU that a benchmark at 64 GPUs
per host has, so with a dataset that fits, the real cluster caches far more than the benchmark
and asks far less of the storage. The WG has worried about submitters scoring too high; the
accuracy of the benchmark's predictions for small workloads was not in scope (the user: not
something the WG had considered). Proposed to bring it in scope without giving up
comparability:

- **Name the two goals** in the brief, and have every rule say which it serves, so a rule that
  helps one at the other's expense is a visible trade.
- **Measure cache-defeated, predict through a model of the clients' caches.** The 5× run
  measures what the storage can deliver; a real cluster asks it only for what its clients
  miss. GPUs the storage can feed ≈ measured capability ÷ (storage demand per GPU × client miss
  ratio). The miss ratio is computable without I/O: for an LRU cache the reuse-distance
  histogram the dry run already computes is the miss-ratio curve (Mattson's stack algorithm),
  and grouping GPU ids into simulated nodes (`--gpus-per-node`, 8 by default) gives each node's
  curve at any cache size. A catalog case can then say "at 8 GPUs and 512 GiB of usable cache
  per node, storage sees 3 % of reads after the first epoch".
- **When the dataset fits, the question changes.** After the first epoch the miss ratio is
  near zero; what storage must deliver is the cold load, checkpoints, and writes. The fidelity
  answer for such a case is the cold-start time, which the benchmark measures directly, not a
  steady state that never reaches storage.
- **Validate the model before trusting it.** Linux's page cache is not an exact LRU
  (active and inactive lists, readahead, the multi-generation LRU), so the curve is an
  estimate: one realistic node (8 GPUs on one host, the application's memory held by a
  ballast) is run and its measured storage traffic compared with the prediction.
- **Storage-side caches** have the same question (the researcher alone on the system gets all
  of it) and wait for concurrent mixes, which are what make contention for them honest.

**Simulated nodes on one host: not in general.** Grouping simulated GPUs into nodes, each with
a realistic cache budget (a cgroup's `memory.max`, which the kernel charges page cache to),
fails because the groups share one page cache: a block one group reads is a hit for the others,
which separate nodes would never get. NFS can separate them with one mount per group under
`nosharecache`; most clients cannot. This is why the 5× rule is the general fallback, and why
the prediction above is computed, not emulated.

**What changes (decided 2026-10-10).**

- The 5× rule is the WG's run rule, justified by the non-representative client; listed in
  `PROJECT_BRIEF.md` §8, and the brief's §5 row says so. aeiou keeps dataset size a parameter.
- aeiou reports what a reader needs to judge a result: client hosts, GPUs per client, client
  memory, dataset bytes against it, and the client hit ratio where the client exposes counters.
- §3.66's use-case test is corrected: a use case declares its GPU counts, as MLPerf Storage's
  CLOSED division fixes the GPU count by model size.
- The catalog gains a derived attribute: whether a case's working set fits in a reference
  node's memory.
- On the backlog (`PROJECT_BRIEF.md` §6 items 28–30): the two goals and the fidelity tooling
  (per-node miss-ratio curves, cold-start time, the validation run); the harness's topology
  in the report; concurrent mixes of cases.

### 3.68 The catalog's first level: classes of system, not lifecycle phases; every category a directory from the start (discussed and decided 2026-10-11)

**The question (the user's).** Why were `kvcache` and `vdb` not under `inference`? There was no
reason beyond the user's first list (§3.66: `training`, `checkpointing`, `vdb`, `kvcache`, with
`inference` added for model loading). Drawing out the inconsistencies, the user observed that
checkpointing runs only in training yet a checkpoint read does what model loading does; that
`inference` held the model while the KV cache and the vector database it uses stood beside it as
peers; that a pure taxonomy would be five layers deep with multi-homed topics, which is overkill,
and that a broad, shallow hierarchy with a loose taxonomic structure is easier for people.

**The diagnosis.** The first level mixed two axes: the lifecycle phase (`training`,
`inference`) and the subsystem (`checkpointing`, `kvcache`, `vdb`). A phase contains subsystems,
so with both at one level every oddity above follows. And the rule of §3.66 already answers the
fear of a deep `training` tree: a directory is a mechanism, never a model or a use case, so an
image classifier and an audio classifier over one file per sample are one directory with two
cases, and the model's purpose and data are attributes. Jev (TypeSafe AI's discriminative model,
2026-09-15; closed weights, served as an API) was the test: to storage its serving is model
loading and perhaps a KV cache, so it would be cases of those mechanisms with its family an
attribute, and no category of its own.

**Decided by the user, 2026-10-11.**

1. **The first level is a class of system whose I/O is reproduced**, named in the words storage
   people use for it: a subsystem for AI, the application class elsewhere. The lifecycle phase
   becomes an attribute (`phase`: training, inference, indexing, preparation, production),
   several values allowed, so that multi-homing lives in attributes and never in duplicated
   directories (a DCP load used for serving is a `dcp-load` case; a vector index build has phase
   indexing).
2. **`input-pipeline`** in place of `training` (`data-loading` was accurate but said neither
   training nor batch inference; `input-pipeline` is TensorFlow's and the field's name for the
   read path that feeds a model, in both). `inference` becomes **`model-loading`** (what is in it:
   weights loaded to serve). `vdb` becomes **`vector-search`**: FAISS and DiskANN are libraries,
   not databases, and VDB is the name of a working group's workload (`PROJECT_BRIEF.md` §8).
3. **New AI categories:** `dataprep`, `embedding-tables` (recommender tables larger than
   memory, looked up by id in tiny random reads with a power-law skew, in training and serving: a
   shape of its own; generating embeddings is batch inference, searching them is
   `vector-search`), `rl` (rollouts, a trainer, weights pushed between them, replay buffers),
   `agentic` (agents' own tool I/O; an agent's KV cache stays `kvcache`). A category for pulling
   runtime images at cluster start was left out (not AI-specific), and if ever added avoids
   "container" and "image", both overloaded here.
4. **`dataprep` splits by the shape of its load**, not by tool, into six classes: `scan-filter`,
   `shuffle`, `dedup-index`, `repack`, `tokenize-pack`, `ingest` (each defined in the
   dictionary). Frameworks such as Spark are hard to trace into compact abstracts; some of these
   may begin as reproductions of a documented shape.
5. **The second level generalizes:** `access` (training's `map`/`stream`) becomes `class`, its
   values defined per category in the dictionary, used only where a category is large
   (`input-pipeline`, `dataprep`, `benchmarks`).
6. **Non-AI workloads at the same level, and no `ai/` directory.** `software-build`, `eda`,
   `vdi`, `media-ingest` (SPEC's VDA, named for the mechanism), `database`, `genomics`, `hpc`
   (whose checkpoint and restart go in `checkpointing`). A `domain` attribute (ai, hpc,
   enterprise, media, life-sciences, software-development; several values) makes "only AI" a
   filter.
7. **`synthetic` becomes `benchmarks`**, with the suite as its class (`io500`, `specstorage`):
   a benchmark whose definition is the workload, reproduced from its specification and never
   presented as an application's I/O. SPECstorage's workloads are themselves emulations of
   applications (as DLIO's are, §3.46), so a traced real build goes in `software-build` and
   SPEC's SWBUILD in `benchmarks/specstorage`, side by side for comparison.
8. **`model-family`** per case (llm, vision, speech, recommender, discriminative) beside
   `modality`.
9. **Every category and class has its directory from the start**, empty or not (the user: it
   shows the intended coverage of the tool and where an external contributor's pull request
   puts its files). The naming rule is in the dictionary: hyphenated nouns, the words storage
   people use, avoiding overloaded ones (container, image, object, cache alone).

**Built 2026-10-11.** `git mv` of `training`, `inference`, `vdb` to `input-pipeline`,
`model-loading`, `vector-search`; the dictionary's `category` (seventeen values), `class` (ten,
each naming its category), `phase` and `domain` (`multiple = true`, a list of distinct values),
`model-family`; `aeiou-catalog` checks that the first level is the categories and nothing else,
that a category with classes holds only them and its mechanisms name one, and lists where
marked; `write` makes any missing category or class directory and generates every category's
and class's `README.md` from the dictionary (what belongs there, what is there, where a first
mechanism goes), which `check` holds as it does `CATALOG.md`; `CATALOG.md` opens with the
coverage (every category and class, its counts, empty ones included) and a table of mechanisms
with phase and domain. No AST, fingerprint, dataset id, or shape changed (the authoring scripts'
docstrings name their directories but are not in the AST). *Choices made in the build, open to
revision:* (1) `phase` and `domain` at the mechanism's level, the phases its cases span, rather
than per case; (2) the phase `production` for non-AI systems doing their work, a value the
discussion did not name; (3) `model-family` set only where the traced model is known (the
ImageFolder classifier, the GPT-2-shaped model, the KV loads' LLMs), absent for the toy
checkpoints, the NumPy volumes, and the vector indexes; (4) the category and class READMEs
generated rather than written, so they cannot drift from the dictionary; (5) the earlier
sections keep the old names as written, this section being the map (`training` →
`input-pipeline`, `inference` → `model-loading`, `vdb` → `vector-search`, `synthetic` →
`benchmarks`, `access` → `class`).

### 3.69 `aeiou-datagen` to object stores: each file streamed into an upload, never staged on a local file system (discussed and decided 2026-10-11; built the same day)

**The question (the user's).** The format libraries write to a local path, and h5py needs a
file it can seek in; the obvious port (write the file locally, upload it, delete the copy)
was called absurd by the user: a benchmark of a storage system with an object API cannot mix
a local file system's performance with the system under test's.

**The answer.** Two facts and a change. First, the staging could never have reached a scored
number: `aeiou-datagen` prepares a dataset before the run and nothing times it; the scored run
reads the objects through the runner's object engine with plain ranged `GET`s at offsets the
layout computes (§3.65), and a run's own writes (checkpoints, KV chunks into an `s3`
namespace) go out from the Rust engine as one upload per handle, with no format library and
no local file system in either. So staging would have cost setup time and scratch capacity
(both real at 50M files), not fairness. Second, an application that does write a container
locally and then uploads it (HDF5's S3 driver, ROS3, is read-only, so pipelines do) has that
in its trace, and the local write is part of its I/O under the interposition test, CLOSED:
the abstract keeps it; that is the workload's property, not the tool's. The change: no
local file system on the path, because none of the four writers needs one.

**Decided (the user agreed to the recommendation and asked for it built, 2026-10-11).**

1. **The client is `obstore`**, the Python bindings of Apache `object_store`, the runner's
   engine (§3.65): one reading of the `AWS_*` environment and one multipart behavior on both
   sides, a small wheel, no AWS SDK. Rejected: `boto3` (heavy, and it reads endpoints and
   addressing its own way) and pyarrow's `S3FileSystem` (the AWS C++ SDK, its own
   configuration, missing from some distributions' pyarrow). It is the builder's `s3` extra.
2. **A writer writes into a byte stream, never a path.** TFRecord, tar, and Parquet append, so
   they stream into the upload (one `PUT`, or a multipart upload in the runner's 8 MiB parts,
   each sent before the next is filled). ~~HDF5 cannot stream (libhdf5 seeks back over its
   metadata), so it is built whole in memory with the `core` driver and the image written out.~~
   (Revised the same day, below: HDF5 and Parquet stream too.) Measured the same day: the `core` driver's image is byte for byte the disk driver's file,
   while h5py's Python file-object driver lays the file out otherwise (the default shape's
   data at 1400, not 2048), so the file-object route was not taken. The same path serves a
   directory, so the two destinations write the same bytes by construction (the test compares
   them anyway).
3. **The check runs on what the writer wrote, before the close**: the byte count the stream
   took and, for Parquet, the footer's metadata as the writer wrote it (the check read the file
   back before; a library's own record of the layout is the same evidence and needs no `GET`).
   A file that fails it aborts its upload, and so does any exception, so no partial object is
   left; content verification stays the separate tool's.
4. ~~**A memory budget, never a fallback to local disk.**~~ (Superseded the same day: no
   writer holds a file whole, so there is no budget; see *Revised* below.) ~~What a writer holds whole (HDF5: its
   image and the copy out, twice the file; Parquet: its table, the file) is held within half
   of `MemAvailable` at the start; a thread waits until its file fits, and a file larger than
   the budget is refused with its size and the budget. No option: the budget is a property of
   the host, as the default thread count is, and a flag in the `[datagen]` table would be one
   the Rust `aeiou datagen` shares a table with and does not know. *Open to revision:* a
   `--memory-mib` if a shared writer host needs it.~~
5. **Placement as the runner's.** `--endpoint NAME=s3://BUCKET[/PREFIX]`, an object name with
   no default place, an endpoint of its name's protocol's kind, a `posix` name never under a
   root placed in a store, each refused in `endpoint.rs`'s words; the prefix must hold no key;
   the manifest one `PUT` at the prefix.

**Built 2026-10-11.** `builder/aeiou/formats.py` (`Sink`, `held`, the stream writers, HDF5 by
the `core` driver), `builder/aeiou/datagen.py` (`Bucket`, `Obj`, `placement`, `opened`,
`is_empty`, `Budget`, the manifest as an object), `builder/tests/test_object.py` against MinIO
(every class into a bucket, every object equal to the directory writer's file and the
manifests' ids equal, the runner's fingerprint under `--s3 blocking` and `async`, a
three-part upload, an aborted one, the refusals), CI's `runner-object` job runs it,
`man/aeiou-datagen.1.md`, `builder/REFERENCE.md` §6. HDF5 to a directory now goes through
memory too (it wrote to disk in 8 MiB slices before): one path, at the cost of holding a
file, within the budget. Not measured: datagen throughput into a real store (the measurement
of brief item 17 is still to come).

**Revised 2026-10-11, later (the user approved it): every writer streams, and the budget is
gone.** Measured before the change, four 1 GiB files on this host: the in-memory HDF5
writer peaked at 2.1 GiB with one thread and 6.1 GiB with four, and Parquet, whose table
the budget counted once, at 4.0 GiB and 9.5 GiB. Memory held per writer caps how many
writers run at once, and so the bandwidth datagen can offer a store; a budget only
serialized them. Now:
- **HDF5**: the file's first `data_offset` bytes come from libhdf5 itself: the same dataset
  of `n` rows created in a sparse scratch file (in `/dev/shm` where there is one, else the
  temporary directory), one row stored, the metadata read back, the file removed; cached per
  row count, so once or twice per dataset. Then the rows go out straight from the payload.
  For a contiguous dataset with no filter and no fill the metadata does not depend on the
  rows' bytes (only the superblock's end of file depends on `n`), so the header followed by
  the rows is byte for byte the file libhdf5 writes with every row stored: an 18 GiB file
  costs 260 KiB of tmpfs, 44 MiB of memory, and 0.65 s. Rejected on the way: `memfd_create`
  (absent from this Python build), a `/proc/self/fd` path (HDF5 2.0 refuses it, "can't
  retrieve real path"), the `core` driver's VFD handle (crashes or a truncated image), and
  h5py's file-object driver (its own layout). The header is still libhdf5's, not a second
  implementation of the format.
- **Parquet**: one row group at a time through `ParquetWriter.write_table`, in two passes as
  before; the first writes zero bytes in the samples' sizes (one zero buffer, no payload),
  valid because with no statistics the footer's bytes do not depend on the samples' bytes.
  pyarrow holds about three copies of a row group while writing it (one page per column
  chunk, the layout's contract), so a thread's memory follows the row group, not the file.
- The payload fills one buffer per call instead of joining slices, and `Sink` passes the
  writer's buffers through without copying them.
- Gone: `Budget`, `available_memory`, `held`, the printed budget, the refusal of a file
  larger than the budget, and its test. Added: the streamed files compared with the
  libraries' own (HDF5 at 1, 6, and 16 rows; Parquet over four row groups), and a memory test
  (two 256 MiB files raise the peak 4 MiB over a warm process; the limit is 64 MiB).
- Measured after, same four 1 GiB files to tmpfs, two runs each: HDF5 1 thread 3.8–4.1 s
  and 72 MiB peak (was 7.8 s and 2.1 GiB), 4 threads 1.8–4.9 s and 108 MiB (was 5.5 s and
  6.1 GiB); Parquet 1 thread 0.40 GiB peak (was 4.0 GiB), 4 threads 3.3–4.8 s and 1.4 GiB
  (was 8.7 s and 9.5 GiB). Times on this WSL host vary by up to a factor of three between
  runs; step 2 (throughput against `aeiou datagen`) is the real measurement.
- **Throughput against `aeiou datagen`, and processes for threads** (step 2 of the same
  plan). 8 GiB per run to tmpfs (`/dev/shm`; this host's 20 cores), 64 files of about
  128 MiB, the median of three on a monotonic clock (the WSL wall clock steps backwards);
  the Rust writer over plain files of the same bytes; samples as each abstract has them
  (TFRecord and Parquet about 120 KiB, HDF5 192 KiB rows). With threads every Python
  writer stayed at about 1 GB/s at 1, 4, and 16 (HDF5 fell to 0.4 GB/s at 16) while Rust
  went 2.3 → 10 → 11.6 GB/s: dgen releases the interpreter lock (5.7 GB/s a thread, 22 at
  four), the writers' Python between the calls does not. So the writers are processes from a
  `forkserver` (the parent may hold the object store's runtime threads, which a `fork` must
  not copy; the server preloads numpy, h5py, pyarrow, crc32c, dgen once); each rebuilds its
  `Job` from what pickles and opens its own store. Two more findings on the way: h5py's
  `d[0] = 0` broadcasts a scalar across a row in 0.6 s, so every process's first HDF5 file
  (and the build's `data_offset` probe) paid it; a zero row costs 1 ms. And the payload now
  hands out views of its 1 MiB blocks (`pieces`), so HDF5 writes the blocks as dgen fills
  them; a copy is made only for a request that spans blocks. Result, GB/s at 1 / 4 / 16
  writers:

  | writer | 1 | 4 | 16 |
  |---|---|---|---|
  | `aeiou datagen` (Rust, plain files) | 2.3 | 10.2 | 9.5–11.6 |
  | TFRecord | 1.2 | 3.4 | 5.1 |
  | HDF5 | 1.3 | 3.7 | 5.3 |
  | Parquet | 0.7 | 1.7 | 2.1 |

  The Python times include about 0.5 s of interpreter start and validation, a large share
  of an 8 GiB run and none of a real one. Bare Python processes generating with dgen and
  writing 1 MiB blocks reach 9–11 GB/s at 16, as Rust does, so the remaining gap is the
  writers' own work. Parquet is bounded by memory traffic, not the interpreter: a row group
  (70 MB here) is copied whole into one buffer, through pyarrow into one page per column
  chunk (the layout's contract), and out, each pass through DRAM where HDF5's 1 MiB blocks
  stay in cache; its footer-sizing pass is about a quarter of its time. Not pursued (each a
  trade the user has not asked for): ~~a Rust container writer from captured headers (a second
  implementation of the formats),~~ (decided later the same day, §3.70: the formats move to
  `aeiou datagen` behind the feature `formats`, HDF5's header from h5py, Parquet ~~through
  parquet-rs~~ by our own writer, revised 2026-10-12) smaller pages (a different layout). Peak memory of a
  writer process: 130 MiB (Parquet), 65 MiB (HDF5), 33 MiB (TFRecord).
- *Found, not fixed (recorded for the user):* a Parquet file whose last row group is short
  fails the layout check, in the old writer as in the new: the layout gives each column one
  page-header size, and a short group's headers are shorter (with 64 rows a group, a row
  count below 64 is one varint byte, not two, in every column's header). With the default 64 rows a
  group, any `samples_per_file` that is not a multiple of 64, or a `count` that leaves a
  short last file, is refused at datagen time. The committed abstracts use multiples.
- **`--threads` renamed `--writers` on both datagen tools** (decided by the user 2026-10-11,
  built 2026-10-11). Once `aeiou-datagen`'s writers were processes, `--threads` named a
  mechanism one of the two tools no longer used; `--writers N` names what happens, files
  written at once (or, for `aeiou datagen`, 64 MiB pieces of a `regions` file), true of the
  Rust writer's threads and the Python writer's processes alike. Renamed with its layers
  (`AEIOU_WRITERS`, `writers` in the `[datagen]` table) on `aeiou datagen` and
  `aeiou-datagen` together, since the table is shared; no alias (no users yet). `run
  --threads` (event-loop threads), `dry-run --threads` (walk workers), and `--object-threads`
  are threads and keep their names. Earlier sections keep `--threads` where they record what
  was decided then.
- **`aeiou-datagen` on several hosts** (proposed and approved by the user 2026-10-11, built
  the same day; `builder/REFERENCE.md` §6). `--ranks`, `--rank`, `--coordinator`, layered,
  as `aeiou datagen` has them. The coordinator is the runner's protocol spoken from Python
  (`builder/aeiou/coord.py`, under 400 lines: the frames, the datagen's messages, the
  server's checks and refusals, the heartbeat), chosen over two alternatives: a runner
  subcommand that Python ranks would join (the Python tool would depend on the Rust binary
  for a few hundred lines of its own, and the runner would gain a subcommand for the
  other's sake), and no coordinator at all, ranks leaving part files on the shared storage
  for rank 0 to poll (a second mechanism, and files inside a dataset root). The flow is
  the runner's step for step, so the manifests mean the same thing: every rank checks its
  roots before the gate, rank 0 writes the manifests once every rank has reported. Two
  changes on one host follow from it: the manifests are written after every dataset rather
  than after each, and the file systems written are `syncfs`ed before them, as the Rust
  writer always did.

### 3.70 Datagen resume, growth, shrink, and subset runs; the payload `aeiou-positional/3` (proposed and decided with the user 2026-10-11; the payload built the same day)

**The use.** `PROJECT_BRIEF.md` item 23 and §3.62: a submitter generates one corpus once, on
the fleet it will run on, and then searches for the SUT's saturation point by running ever
larger steps against it. Under the sizing rule (§3.67, at least five times the clients'
DRAM) a step with more clients needs a larger dataset, so the corpus is sized for the
largest step and a smaller step reads a prefix of it, or the corpus grows or shrinks between
steps. Names, sizes, directories, and dedupe groups are functions of the id alone (§3.62).
The first draft of this note (the same day, proposed) found three gaps: the runner checks
every `readdir`'s entry count, the Python writer wrote through the page cache, and a resume
had no record of what an unfinished corpus was meant to be. The decisions below came out of
the discussion that followed; the user's corrections are marked.

**1. The payload, `aeiou-positional/3` (decided and built).** Looking at the tail check the
user proposed (read the last few transfers of a file after a crash) turned up two faults in
`/2`. dgen-data's layout put the compressible part at the end of each 1 MiB block as zeros,
and a file is a prefix of its blocks, so a file below `1/C` MiB did not compress at all
(measured: a 110 KB image at `C = 2`, zlib ratio 1.00; 600 KB, 1.14; 1 MiB, 1.99), and
storage that compresses in 8 to 32 KiB units saw pure noise and pure zeros, the zeros
rewarding zero detection. (`PROJECT_BRIEF.md` §5 had recorded the bimodal 4 KiB pieces as
the DLIO convention on 2026-09-29; the small-file effect had not been seen.) The user
proposed `0x5A` for the compressible bytes, so that they compress and dedupe as zeros do
but are never zeros, and asked whether dgen could make 4 KiB blocks: its streaming
generator, the only one Python can seed, clamps blocks to at least 1 MiB
(`generator.rs:1372`), and a generator per 4 KiB ran at 3.9 GB/s. Its random bytes are
`Xoshiro256PlusPlus::seed_from_u64(seed + block)`, so the user asked whether we could make
them ourselves. Decided: **SFC64**, numpy's bit generator in Python (`numpy.random.SFC64`,
its stream fixed across numpy versions by NEP 19) and about forty lines of Rust
(`payload.rs` `Sfc64`); a block's state is `a`, `b`, `c` the first three SplitMix64 outputs
from the block seed (`labeled_key(seed, "payload", [u, b])`, unchanged), the counter 1, and
twelve outputs discarded, each output little-endian; each 4 KiB piece is the next
`⌈4096 / C⌉` bytes of the stream and then `0x5A`; at `C = 1` the stream as is. Dedupe is
unchanged (`/2`'s units). `dgen-data` and `dgen-py` are no longer dependencies; the manifest
says `generator: sfc64`, `version: 1`, `wrapper: aeiou-positional/3`, so every dataset id
changed once (none published). Measured, a 110 KB prefix at `C = 2`, 3, 4: ratio 1.96, 2.92,
3.87; 8 GiB to tmpfs, GB/s at 1 / 4 / 16 writers:

| writer | `/2` (dgen) | `/3` (SFC64) |
|---|---|---|
| `aeiou datagen`, `C = 1` | 2.3 / 10.2 / 9.5–11.6 | 2.2 / 12.4 / 22.5 |
| `aeiou datagen`, `C = 2` | | 3.2 / 11.0 / 20.9 |
| `aeiou-datagen`, TFRecord | 1.2 / 3.4 / 5.1 | 1.3 / 5.5 / 8.2 |

Tests: `payload.rs` and `builder/tests/test_payload.py` pin the same stream and block hashes
(numpy's values, which the Rust matched the first time), every piece and an image-sized
prefix compress at the ratio, and the files `aeiou datagen` writes with dedupe 2 and ratio 3
equal the Python payload byte for byte. dgen remains a fine generator; the change is that
our definition now depends on one small, fixed algorithm we hold on both sides.

**2. ~~`O_DIRECT` in the Python writer, always (decided).~~ Dropped 2026-10-11, later
(below): the container formats move to the Rust writer, which already writes `O_DIRECT`.** The user: `O_DIRECT` is almost
always faster than buffered, and §3.62 forbids filling the client's page cache. The Python
`Sink` writes directory roots through a 4 KiB-aligned 1 MiB staging buffer, with the tail
handled as the Rust writer does (tried unaligned, padded and truncated on `EINVAL`).

**3. The manifest from the start, and progress (decided).** The user proposed writing the
manifest at the start, with the parameters of the whole dataset, as the record of what an
unfinished corpus is meant to be; it is one file, `.aeiou-dataset.json`, with a state
(`writing`, `growing`, `shrinking`, `complete`) and the progress, and a run refuses
anything not `complete`; the id is unchanged by the progress, since only the normative part
is hashed. The user proposed a checkpoint every `N` files behind a global barrier and a
check of the last few transfers of each file after it; decided instead, in the discussion:

- *No check after a checkpoint: rewrite.* Everything after the last recorded point is
  rewritten whole, opened `O_CREAT|O_TRUNC` (never `O_EXCL`, the user's note), so a partial
  or stale file loses its tail. With `O_DIRECT` a crash leaves a strict prefix anyway (XFS
  and ext4 record an extending direct write's size at its completion, after the data; the
  Linux NFS client commits a direct write before it returns; mechanisms, not tested here),
  and with `0x5A` a hole could no longer pass for payload; the reads are not needed.
- *No barrier: messages, and one manifest write per window.* Every `--progress-secs`
  (layered, default 30) each rank notes what is finished, flushes (cheap under `O_DIRECT`),
  and sends rank 0 that note; rank 0 keeps every rank's latest and rewrites the manifest
  once per window. The user's point: one write per rank per window would be fifty writes;
  a global low-water mark is one write but makes the leaders rewrite. The compromise keeps
  per-writer precision at one write: each writer writes its run in id order, so a rank
  reports its finished prefix plus one point per active run, a few bytes per writer. Rank 0
  also prints one progress line per window (files, bytes, rate, time left).
- *A different host count on resume* (the user: a case not thought of, and valuable): the
  manifest holds the unfinished ranges, so the new ranks split them by whole directories.
- *Objects:* a completed upload is the whole object, so resume looks for missing keys
  (one paged `LIST`), and it aborts the multipart uploads left below the prefix.

**4. Whole directories (the user's proposal, decided; revised 2026-10-12, §3.71 and below: not
required of a corpus, only of a subset run's count).** Every directory holds the same
number of files, for an even load: the count is rounded up to a multiple of `D·k` under
`{id div D}` (`k` the samples per file), `M·k` under `{id mod M}`, and `k` for a flat
pattern; the requested and the rounded counts are printed and recorded. Work is handed to
ranks by whole directories, so no two hosts create in one directory (§3.62 allowed one
shared boundary directory), with a fallback when there are fewer directories than ranks.

**5. Grow and shrink (the user's proposal, decided).** Explicit operations, separate because
shrink deletes data (a single count that grows or shrinks could delete a corpus on a typo).
Grow, shrink, and resume read everything but the target count from the manifest and need no
abstract, and they refuse when this tool's payload version or container library version
differs from the manifest's (a Parquet footer names the pyarrow version, so a newer pyarrow
would change new files' footer size and break the layout every file shares). ~~Grow adds
whole directories; shrink removes trailing ids, whole trailing directories under `div`,~~
(revised 2026-10-12 with item 4: any count; below) Shrink removes trailing ids,
with no hole in the sequence; each marks the manifest first and `complete` last. The
dataset id changes with the count.

**6. Walks, inventories, and subset runs (decided).** The user's correction: large shops do
not walk a training corpus or `LIST` a bucket at 20 M files; they read an inventory captured
once. The manifest cannot stand in for that: the walk in `file_per_sample` models the
application's own walk, which under CLOSED is part of the workload. So the large-corpus
workload is an inventory variant: the abstract reads a file of the size an inventory of the
run's count would be (written by datagen as payload; nobody reads its content), and the
names come from the patterns, as they always have. Small corpora keep the live walk, whose
entries the runner checks and otherwise ignores. A run's parameters come from its command
line (the user, narrowing an earlier over-generalization: reading parameters from the
manifest is datagen's, for grow, shrink, and resume); the manifest only checks that the
corpus fits the run: the definition equal but for the count, the run's count no larger, and
`aeiou run --subset` required for a smaller count, so a typo is refused rather than run. A
subset is a contiguous set of whole directories from 0, the count rounded up the same way
and both counts printed; the directories beyond are ignored. A live walk on a subset walks
those directories~~, and only the root listing differs from a corpus of exactly that size: it
returns every directory's name, so the runner expects the corpus's count there and the
subset's below, and the report says so~~ (struck 2026-10-13: no handle names a dataset's
root, `dir` names one of its directories, so a walk lists whole directories only and a subset
is the fresh corpus's workload, §3.72). A flat pattern has no whole-directory subset, so a
live-walk subset of a flat dataset is refused (no committed flat abstract walks). Subsets
apply to sample spaces (`access: map` or `stream`) whose count is a bare parameter
reference; a `regions` dataset or a dataset without an access mode (a DiskANN index is a
graph over every node) is refused (in contract 0.8's terms, §3.72: an `array` or an `indexed`
dataset). Whether a scored step may run on a subset is the WG's
question (`PROJECT_BRIEF.md` §8), not aeiou's.
*Built 2026-10-13* (`runner/REFERENCE.md` §5, *Subset runs*), as decided in §3.72: the count
asked is rounded up before the model is built (`D·k` ids under `{id div D}`, `k` for a flat
pattern), so the op stream and the fingerprint are a fresh corpus's of the rounded count and
the dataset id is the corpus's; the options block shows the count asked, run, and the
corpus's, the report's dataset entry a `subset` block, and the run prints the age bias. A
flat subset is refused whenever the abstract has a `dir` handle on the dataset, whatever its
parameters (so `file_per_sample` made flat is refused even with `enumerate=false`: the check
reads the abstract, and no committed consumed dataset is flat with a walk). Rounding that
reaches the whole corpus is no subset (usage error with nothing else smaller); one that
passes it is refused, naming the largest whole subset. Tested on one host and two ranks
(`tests/coord.rs`, `a_subset_runs_whole_directories_of_the_corpus`), the fingerprint held to
the dry run's at the rounded count, and every refusal. ~~The inventory variant is next.~~
The inventory decided and built the same day (§3.73).

**Order.** ~~(1) The payload (built). (2) `O_DIRECT` in the Python writer, measured. (3) The
manifest from the start, the progress messages, and resume, on both writers. (4) Whole
directories and per-rank directories. (5) Grow and shrink. (6) Subset runs and the
inventory variant. Each with tests on one host and on two ranks.~~ Superseded 2026-10-11,
later (below): the container formats move to Rust first, and items (3) to (6) are built in
Rust only.

**Revised 2026-10-11, later (decided with the user; nothing built): the container formats
move to `aeiou datagen`, and item 23 is built in Rust alone.** Building resume, progress,
and grow/shrink twice, once in each writer, raised the question of keeping two writers at
all. The user: "attracted to the rust foundation, python is invariably slow" (the Python
writer reached 8.2 GB/s at 16 writers for TFRecord, the Rust writer 22.5 for plain files;
Parquet under `/2` 2.1). Decided:

1. *A writer module beside `datagen.rs`, behind a cargo feature `formats`*, off by default
   as `object` is. The runner proper (an AST executed by `run`) is not the basis, and the
   user accepted why: an AST writes only positional payload, with no headers, CRCs, or
   footers (the runner is format-ignorant); a namespace's rules (content keyed on the path
   hash, `--seed` and `--gpus` in its identity) are not a dataset's; and progress, resume,
   and `O_DIRECT` staging are datagen's logic, not the op language's. `aeiou datagen`
   refuses format-class datasets today; that refusal is lifted per format as Rust takes it
   over.
2. *TFRecord and tar first.* TFRecord is our own framing (u64 length, masked crc32c of the
   length, the data, masked crc32c of the data) with the crate `crc32c`; tar (WebDataset)
   through the crate `tar` or our own 512-byte headers, at the offsets Python's `tarfile`
   gives. Both byte-identical to the Python writer, which the tests compare against.
3. *HDF5: the header from Python, the rows from Rust.* The header bytes are made by h5py,
   the reader's own libhdf5, so their version always matches the reader's; Rust writes
   header and rows. No C link: `hdf5-metno-src` 0.10.4 bundles HDF5 2.2.0 while the reader
   here is h5py 3.16 on HDF5 2.0.0, and the header bytes may differ between them; the pure
   Rust crates (`hdf5-pure`, `hdf5-writer`) are young and are not what real producers use.
   So an HDF5 datagen has a Python step first (about 1 ms a header), carried in the
   manifest written at the start (item 3 above). Open: who writes that manifest, and how
   Rust receives the header.
4. ~~*Parquet through parquet-rs* (the arrow-rs crate `parquet`), not byte-identical to
   pyarrow: PLAIN, uncompressed, no statistics, no page index (to verify: arrow-rs may write
   them by default), one page per column chunk; the layout constants come from the Rust
   writer, the manifest's `format.writer` names parquet-rs, so its dataset ids differ from
   pyarrow-written ones. Checked with `aeiou-trace`: pyarrow reading Rust-written files,
   compared with the abstract. Brief item 31 (the short last row group) is re-examined
   there.~~ Revised 2026-10-12: Parquet by our own writer, below.
5. *`O_DIRECT` in the Python writer is dropped* (item 2 above, superseded). The Python
   `aeiou-datagen` stays as the byte and behaviour reference in tests until it is retired.
6. *Order:* (a) TFRecord and tar in Rust; (b) HDF5 with the header from the start
   manifest; (c) Parquet ~~through parquet-rs with the pyarrow trace check~~ by our own
   writer (revised 2026-10-12, below); (d) items 3 to 6
   above in Rust only: the manifest from the start with its state and per-writer progress,
   resume, whole and per-rank directories, grow and shrink, `run --subset`, and the
   inventory variant.

Crates checked on crates.io 2026-10-11, for this and later formats: `parquet` 60.0.0,
`arrow-ipc` 60.0.0, `tfrecord` 0.15.0, `tar` 0.4.46, `crc32c` 0.6.8, `safetensors` 0.8.0
(Hugging Face's own), `zarrs` 0.23.14, `lance` 13.0.0, `orc-rust` 0.9.0, `apache-avro`
0.22.0, `heed` 0.22.1 (LMDB), `npyz` 0.9.1, `zip` 9.0.2. The speeds to beat, 8 GiB to tmpfs
at 1 / 4 / 16 writers under `/3`: Rust plain files 2.2 / 12.4 / 22.5 GB/s (`C = 1`), 3.2 /
11.0 / 20.9 (`C = 2`); Python TFRecord 1.3 / 5.5 / 8.2. Parquet and HDF5 under `/3` are not
measured (under `/2`: 0.7 / 1.7 / 2.1 and 1.3 / 3.7 / 5.3).

**Built 2026-10-12: (a), TFRecord and WebDataset in Rust.** `formats.rs` behind the cargo
feature `formats` (one dependency, `crc32c`; the tar headers are ours, 512 bytes as CPython
3.12's `tarfile` writes a ustar member: mode `0000644`, owner 0, mtime 0, the checksum as
`%06o\0 `, devmajor and devminor left NUL for a regular file). A file is a byte stream
pulled in pieces of any size, framing made as it is reached and each sample's payload
filled in place, so the directory writer takes 1 MiB `O_DIRECT` blocks from it and the
object writer 8 MiB parts, holding a piece and never the file; every file is checked
against the size its layout predicts, which caught the one bug on the way (the tar end
padded again on every call). Both formats matched the Python writer byte for byte, with
dedupe 2, compression 3, and a short last shard, into directories and into a MinIO bucket
(multipart). The manifest's `format.writer` names the Rust writer, so its dataset ids
differ from the Python writer's for the same bytes, as decided for Parquet; `aeiou run`
compares the definition, not the id, so either corpus runs the same abstract.

Speed, 8 GiB to tmpfs, 64 shards of about 128 MiB, samples of about 120 KiB, the median of
three (GB/s at 1 / 4 / 16 writers, this session):

| writer | TFRecord | WebDataset |
|---|---|---|
| `aeiou-datagen` (Python) | 1.4 / 4.8 / 5.9 | 0.8 / 2.9 / 4.8 |
| `aeiou datagen`, first cut | 2.1 / 9.1 / 10.1 | 2.2 / 9.2 / 12.3 |
| `aeiou datagen` | 2.2 / 9.9 / 20.7 | 2.7 / 12.3 / 22.6 |

Plain files of the same shape: 2.7 / 8.3 / 21.2. The first cut was held at half that by
the payload: a container's samples start anywhere in a block, so `Filler` made each block
whole into its one-block cache and copied out of it, 2 MiB a writer and a copy more for
every byte. A block read in order from its start is now made in place as it is read, the
generator continued where the last range stopped (`payload.rs` `Cursor`; the bytes are
unchanged, `a_range_read_in_order_is_the_blocks_bytes`), and at 16 writers containers write
as fast as plain files. Next: (b), HDF5 with its header from h5py.

**Revised 2026-10-12 (the user's proposal, decided): the HDF5 header from a tool beside
`aeiou`, not from the start manifest.** Decision 3 above carried the header in the manifest
written at the start, which tied HDF5 to item 23's manifest work. The user proposed instead
that the HDF5 knowledge stay in a separate tool that does the one thing, `aeiou datagen`
running it transparently (fork, exec, the header read from its stdout). The header is a pure
function of the dataset's name, shape, dtype, and row count, so a dataset has at most two
(the full files and a short last one), made before the start gate, where a missing or
failing tool stops the run before a byte is written. Decided:

- *`aeiou-header --format JSON --rows N`*: the dataset's resolved `format` entry and a row
  count in, the raw header bytes on stdout and nothing else (warnings and errors on
  stderr, which `aeiou datagen` relays). Named for what it does, so another format whose
  bytes must come from its own library could use it; HDF5 is the only class it takes. It
  refuses when its h5py and libhdf5 differ from the abstract's `format.version`, the
  version the layout's data offset was probed with; `aeiou datagen` checks the length
  against the layout's `file_header`.
- *Found beside `aeiou`, never on `PATH`* (the user: a `PATH` search may find another
  executable or another version). `std::env::current_exe()`, canonicalized, so beside the
  real binary whatever symlink started it. In a source tree the builder's tests and CI put
  a symlink named `aeiou-header` beside `runner/target/release/aeiou`, pointing at the
  venv's entry point.
- *A standalone executable in the release* (the user: someone copying `aeiou` to the
  clients would not know to copy a second tool): the release builds it with PyInstaller,
  one file carrying Python, numpy, h5py, and libhdf5 at the versions `uv.lock` pins, into
  `bin/` beside `aeiou`.
- *Only rank 0 runs it.* Rank 0 makes the headers before the gate and gives them to its
  in-process coordinator, which sends them to every rank with `Start` (a field `share`, an
  addition the Python coordinator ignores; `Ready` unchanged), so client hosts never need
  the tool, and one source makes a header-hash identity check unnecessary. The manifest's
  `format.writer` names the Rust writer and the header's versions; its provenance records
  each header's SHA-256.

**Built 2026-10-12: (b), HDF5.** As decided, with one refinement: `--rows` repeats, the
headers coming back to back (each is the layout's `file_header` long, so they split by
length), so a dataset with a short last file costs one run of the tool, not two. The bundled
tool unpacks itself on every run: 0.56 s (35 MB, PyInstaller 6.22.3; it needed h5py and
numpy named as hidden imports, since `formats.py` imports them by name), against 0.16 s for
the venv's script. Byte for byte the Python writer's files, with dedupe 2, compression 3,
and a short last file, to a directory and into MinIO (multipart); a copy of `aeiou` alone in
a directory refuses HDF5 on one host, naming the path it looked at, and runs rank 1 of two,
the headers arriving with `Start`. 8 GiB to tmpfs, 64 files of 682 rows of 192 KiB, GB/s at
1 / 4 / 16 writers: Python 2.0 / 5.3 / 6.9, Rust 2.8 / 8.4 / 14.8. At 16 the Rust figure is
the 0.16 s of the tool on a 0.4 s write; 8 GiB at plain files' 21.2 GB/s plus 0.16 s is
15.2, so the rows go as fast as plain files and the tool is a fixed cost a real corpus does
not see. ~~Next: (c), Parquet through parquet-rs.~~

**Revised 2026-10-12: Parquet by our own writer, not parquet-rs (decision 4 above).**
Before building (c) the question was who owns the layout's constants (the page header
formula, the footer's size), which the Python class probes from pyarrow at build time. The
footer pyarrow writes with the class's options, decoded, turned out small and of one shape:
`FileMetaData` with the schema (the narrow integers with their converted and logical
types), `num_rows`, per row group its chunks' `ColumnMetaData` (type, encodings
`[RLE, PLAIN]`, path, codec, counts, sizes, `data_page_offset`, `encoding_stats`, and for
the binary column `size_statistics`), the key `aeiou.pad`, `created_by`, `column_orders`;
the page header `16 + 2 varints + 1`; every field a function of the sizes, none of the
bytes. Recommended and decided by the user: write it ourselves, as tar and TFRecord are, for
these reasons. It is about 200 lines (a compact-protocol encoder for these structs, the
header, the footer). It is one pass: the footer follows from the chunks as written, where
pyarrow needs a counting pass to size its pad and parquet-rs would hold a row group in
memory and bring its own page and offset-index choices. The question of ownership goes
away: the layout stays the Python class's (the AST's `format.layout`), and the Rust writer
checks every chunk's header and every row group's offset against it and refuses a
mismatch, the rule tar, TFRecord, and HDF5 follow. The files are pyarrow's byte for byte
but for `created_by`, which names our writer rather than claim `parquet-cpp-arrow` (the
pad takes up the difference in length, so the size and every offset are the same), so the
tests compare bytes with the Python writer rather than only reading the files back. And it
needs no arrow-rs dependency tree. Brief item 31 is not changed by either writer (the
runner's layout gives a column one header size), so the refusal stays.

**Built 2026-10-12: (c), Parquet.** `formats.rs`: `Thrift` (compact protocol: field
deltas, zigzag varints, binaries, lists), `parquet_page_header`, `parquet_footer`,
`parquet_tail` (the pad found in at most four tries, since its length prefix can grow), and
the stream: `PAR1`, per row group per column the page header and the rows (a binary row its
4-byte length, then the payload in place; a fixed column's values, the Python writer's
`(id % 1000)` cast to the type, or the id as a float), the footer. `format.writer` and
`created_by` are `aeiou datagen parquet writer`. Unit tests pin pyarrow 25.0.1's bytes: two
page headers, and the footer of a file of six columns (every narrow type, a float, the
binary) given pyarrow's `created_by`. The builder's tests compare `aeiou datagen` with
`aeiou-datagen` on `parquet_stream` and on a variant with a column of every fixed type, to a
directory and into MinIO (multipart), everything before the footer byte for byte and the
footer with `created_by` and the pad taken out, then run the abstract over the files to the
dry run's fingerprint. Found on the way: `FIXED_WIDTH` gave `int8`, `uint8`, and `int16`
their Arrow widths (1, 1, 2), but a PLAIN page stores them as INT32, 4 bytes, so the
Python writer's own check refused any such column; fixed in `formats.py` (no committed
abstract had one). 8 GiB (9.5 GB, lognormal around 128 KiB, 64 files of 1024 rows) to
tmpfs, GB/s at 1 / 4 / 16 writers: Python 0.7 / 1.8 / 4.5, Rust 2.2 / 12.1 / 22.6, plain
files' speed. Next: (d), brief item 23 in Rust.

**Built 2026-10-12: (d), part 1, the manifest from the start and progress (item 3 above;
resume, whole directories, grow and shrink, and subsets not yet).** `manifest_version` 2
(so a reader that knows 1 cannot take an unfinished corpus for a whole one; every dataset
id changes once, none published), with `state` and `progress` outside the normative part.
Rank 0 writes each manifest before the start gate, `writing` with every id unfinished, so
a corpus never holds a file without saying what it is meant to be; the other ranks'
emptiness check ignores the manifest's name, and rank 0 removes its manifests if the gate
does not open. Every `--progress-secs` (layered, default 30) each rank notes what is
finished, `syncfs`es, and sends rank 0 the note (`Progress`, a new coordinator message the
Python coordinator ignores); rank 0 merges every rank's latest note (a rank not heard from
counts as having written nothing), rewrites each manifest once, and prints one line
(files, bytes, rate, time left). The note is per writer as decided: writers claim runs in
id order under one lock and advance a cursor after each file, so what is left is the rest
of each writer's run plus the runs not yet claimed. Each rank sends a last note before
`Done`, and rank 0's window runs until every part is in, then writes `complete`. Choices
made while building: the note is taken *before* the flush, so an id it calls finished is
on stable storage; the old every-5-seconds line per dataset is replaced by rank 0's
window line; a regions file is one id, unfinished until whole; on an error every rank
makes one last note, so rank 0's record is as late as it can be. `run` refuses any state
but `complete`, naming the files not written. The Python writer writes version 2, once,
last, `complete`. Tested by killing a datagen, on one host and on two, after a few
windows: every id outside `unfinished` is a file of its full size, and `run` refuses the
corpus. The bookkeeping costs nothing measurable (8 GiB of TFRecord to tmpfs at 16
writers: 14–17 GB/s against 11–15 for the previous build, this machine's tmpfs varying
run to run). Next, before building: the command lines of resume, grow, and shrink, and
the count rounding of whole directories.

**Decided 2026-10-12 (the user): the command lines.** Resume, grow, and shrink are flags of
`aeiou datagen` that take the dataset root in place of the abstract: `--resume ROOT`,
`--grow N ROOT`, `--shrink N ROOT`, with `--ranks`, `--rank`, `--coordinator`, and refused
when this binary's payload or container library differs from the manifest's. Whole
directories are not required of a corpus (§3.71: the runner stays exact; rounding is the
sizing step's), so item 4's rounding is withdrawn and a resume splits work by ids.

**Built 2026-10-12: (d), part 2, resume.** `aeiou datagen --resume ROOT`, ROOT a dataset
root or `s3://BUCKET[/PREFIX]`. Choices made while building:

- *The abstract from the manifest.* The manifest's resolved dataset, wrapped in an abstract
  with one actor that does nothing and placed at ROOT with an endpoint, goes through the same
  loader and validator as any abstract, so the code that writes is the code that wrote and
  the dataset id comes out the manifest's. The manifest did not record the dataset's name;
  `provenance.dataset` now does (not hashed, so no id changes).
- *Work by ids, not directories.* The unfinished ranges are split over the ranks as the
  contiguous split of their concatenation, the same arithmetic as a first datagen's
  `[0, files)`, and cut into the writers' runs at multiples of the run size. With whole
  directories not required, nothing better is owed; two ranks may share the boundary
  directory, as §3.62 allowed.
- *Objects: skip what is there.* One `LIST` of the prefix; an unfinished id whose key is
  present is whole (an upload that did not complete leaves no object) and is not sent.
  Aborting the parts of uploads a crash left is not built: `object_store` 0.14 has no call
  that lists multipart uploads, and a signed `ListMultipartUploads` of our own is more than
  this step is worth; the parts are invisible to readers, and a bucket lifecycle rule
  (`AbortIncompleteMultipartUpload`) can remove them. Open.
- *The checks.* The payload block and the format block (class, reader, library version,
  writer) must equal what this binary writes, and rank 0's HDF5 headers from `aeiou-header`
  must have the provenance's SHA-256s, or nothing is written (`a resume would mix them`).
- *The record.* The manifest is kept as it is (never removed when the gate does not open),
  rewritten every window with the earlier counts added, and `complete` at the end with
  `provenance.resumes` gaining this datagen's ranks, hosts, counts, times, and `direct`;
  `files_written` and `bytes_written` count each file once (the last recorded window's
  counts, then this datagen's: a file written after the last window and lost is not counted
  twice). A `complete` manifest has nothing to write, said, exit 0, so a script may resume
  until it succeeds.

Tests: a datagen killed after three windows on one host and on two, an unfinished file
stretched 300 KB past its size, resumed on the other host count: the same id, `complete`,
every file counted once, `run` accepts the corpus, and the rewritten files are the payload's
(`tests/coord.rs`); the usage refusals and a complete corpus. The builder's tests resume
TFRecord, WebDataset, HDF5, and Parquet corpora with files missing and cut short, to the
bytes of a datagen that ran through (HDF5's headers made again and compared), refuse a
manifest naming another writer, and resume a bucket, sending only the two missing objects.
By hand: a 20,000-file `file_per_sample` corpus killed at 900 files and resumed on one host
and on two is byte for byte the uninterrupted one. Next: grow and shrink.

**Decided 2026-10-12 (the user, on the points proposed): grow and shrink.**

1. *N is the dataset's `count`*, the quantity its parameter sets (samples where a file holds
   several, files otherwise); the manifest's resolved `count` is replaced by the literal.
2. *The short last file is work.* A corpus whose count is not a multiple of
   `samples_per_file` ends in a short file; a grow from such a count fills it and a shrink to
   one cuts the new last file, so that file is written again, whole. Without it a grown
   corpus would not be a fresh datagen of N.
3. *The manifest first.* The new count's manifest is written before anything is removed or
   written, `growing` or `shrinking`, with the old count in `progress.from`, and `complete`
   last; `run` refuses it meanwhile, and `--resume ROOT` finishes an interrupted change with
   no machinery of its own (removing again is no error).
4. *Direction guards:* a grow refuses N no larger than the count, a shrink N no smaller, and
   both N of 0; no `--yes`, the two flags being the guard.
5. *Directories a shrink empties are removed*, so the tree is a fresh datagen's.
6. *`files` datasets only*; only the count changes, so the layout is fixed for the corpus's
   life (§3.71).
7. *Age, noted:* after a grow the trailing files are younger than the rest, the bias §3.71
   states for subsets; recorded, not acted on.

**Built 2026-10-12: (d), part 3, grow and shrink.** `aeiou datagen --grow N ROOT`,
`--shrink N ROOT` (`runner/REFERENCE.md` §5). Choices made while building:

- *The old corpus from the new one.* The old names and sizes come from the new `DsMeta`
  with the old count (`DsMeta::with_count`), since everything of an id but a short last
  file's samples is a function of the id; no second abstract is loaded.
- *One formula for both.* The work is `[w0, files(N))`, `w0 = min(old, N) / samples_per_file`
  (empty for a shrink to whole files), and the removal is the old names of `[w0, files(old))`
  that the new count lacks: the files past N, and the extra chunks of a chunked file a shrink
  cuts. Each rank removes its share of that range before writing; objects in batches of
  1,000 keys.
- *The counts.* The corpus keeps its totals less the old ids from `w0`, computed from the
  old sizes, so `files_written` and `bytes_written` end equal to a fresh datagen's.
- *Headers.* HDF5 headers are made for N; those both counts share must equal the
  provenance's, and the new set replaces it.
- *Resume of a change.* An object for the old short last file is in the bucket whole but is
  not the new one's, so a resume does not skip it.

Found on the way and fixed: a writer counted a file before advancing its cursor and a window
read the two apart, so a resume could count a file twice (`files_written` one high, seen
once under a loaded test run); the cursor and the counts now move under one lock per writer,
read together, and a regions file counts once whole. Tests: `tests/coord.rs` grows 3,000
files to 5,000 and shrinks to 2,100 on one host and two, each against a fresh datagen of the
new count (files byte for byte, emptied directories gone, id, totals, `run` accepts it);
kills a grow at its third window and finishes it with `--resume` on two hosts; refuses a
count in the wrong direction, 0, a parameter beside it, and two operations at once. The
builder's tests grow and shrink TFRecord, WebDataset, HDF5 across short last files (headers
included) and Parquet (whole files, brief item 31) to the bytes of a fresh datagen, and a
bucket of TFRecord shards and of plain files. By hand: a chunked dataset (`lmcache_local_disk`)
grown and shrunk to a fresh datagen's tree. ~~Next: subset runs (`run --subset`), then the
§3.71 sizing step.~~ (2026-10-12: settle, global claims, and Ctrl-C first, below.)

**Decided 2026-10-12 (the user, on the points proposed): settle, global claims, and Ctrl-C.**
The user's question: a grow started by mistake, or one that must stop, has only `--resume`
to finish it, so a corpus can be stuck between two counts for hours. Decided:

1. *`aeiou datagen --settle ROOT`* (the name preferred to `--stop`): finish what was
   claimed, claim nothing new. An interrupted `writing`, `growing`, or `shrinking` corpus is
   brought to a self-consistent count it reached, the manifest `complete` with that count's
   id; the asked-for and the settled counts are recorded in the provenance (a `resizes`
   entry, or a `settles` entry for a first datagen). Same command-line rules as `--resume`.
2. *Monotonic in whole directories, by a global claim order* (the user's point: with each
   host writing a contiguous slice and each writer a run of it, an interrupted corpus has
   holes all through the id sequence, and the longest finished prefix would keep about one
   host's share of the work). Every datagen (first, resume, grow, shrink) hands out work
   from one queue on rank 0, in id order: a run is a whole directory under `{id div D}`, a
   fixed block of ids for a flat pattern, smaller runs when there are fewer directories than
   hosts × writers (two hosts may then share a directory, as §3.62 allowed). Writers ask
   rank 0 for the next run, one ahead to hide the round trip (about 150 claims a second at
   today's speeds; rank 0's own writers claim in process). So an interrupted corpus is whole
   directories `[0, k)`, a band of partial ones no wider than hosts × writers runs, and
   nothing beyond. This replaces the contiguous host slices (`work_slice`) and balances the
   load across hosts as well. The Python writer stays static.
3. *A shrink removes strictly in reverse:* runs claimed from the top down, each removed from
   its highest id down (the built removal went upward). Settling a shrink completes only
   the removals in flight, deleting no more than the band, and never continues the shrink
   (the user: a typo of 50 for 50M must not cost hours of work); `--grow` writes it back. A
   shrink that had reached the new short last file's rewrite has it completed.
4. *Settling a grow or a first datagen* completes the runs in flight; the count is the end
   of the highest claimed run (whole files; a directory boundary under `div`), or N. The old
   short last file is in the first run claimed, so it ends whole. A settled count of 0 is
   refused (remove the corpus instead).
5. *A claim mark written ahead:* rank 0 records `claimed_to` in the manifest before handing
   out runs beyond it (a few bands at a time), so after a crash a settle removes the strays
   in `[settled count, claimed_to)`; a clean stop makes the mark exact. Progress tells runs
   in flight from runs never claimed.
6. *Ctrl-C:* rank 0 catches `SIGINT` (a small `sigaction` handler setting an atomic flag; no
   new crate) and sends every rank Stop; each rank stops claiming, abandons the file it is
   writing, takes a last note, flushes, and sends note and Done; rank 0 waits for every rank,
   writes the manifest once more (state unchanged, and true), prints that all ranks stopped,
   and exits 130. `SIGINT` on another rank goes the same way through rank 0 (as a failure
   does today). A second Ctrl-C exits at once (the manifest of the last window is still
   true). A rank whose connection drops counts as stopped, its last note standing.
7. *A live corpus is refused:* `--settle` and `--resume` refuse a manifest updated less
   than two progress windows ago, since a datagen may still be writing it; the manifest
   records `progress_secs` beside `progress.updated`, and the refusal names the wait.

**Built 2026-10-12: (d), part 4, claims, settle, and Ctrl-C** (`runner/REFERENCE.md` §5, §6).
As decided, with these choices made while building:

- *What rank 0 knows of a rank's claims.* Grants carry the run (`Grant { dataset, ticket,
  run }`), so only rank 0 cuts the work and other hosts' `--writers` need not agree. A
  rank's note lists its writers' unfinished remainders, the grants it holds that no writer
  has taken, and `seen`, the runs it has received; rank 0 counts the runs it granted that
  rank after its `seen` as not done. Grants to one rank go out in the order made and are
  counted by one reader thread, so the count names them; rank 0's own writers take a run and
  set their slot under one lock. No list of every run a rank took is ever sent.
- *The band is wider by the claim ahead.* Each writer on a host other than rank 0 holds one
  run ahead, so a stopped corpus's band is up to a run per writer on every host plus one per
  writer on the other hosts (seen in the test: 7,452 ids below a boundary of 7,800 on two
  hosts of two writers).
- *One step per id.* Removal and writing are one pass: each id's names under `reach` (the
  largest of the count, `progress.from`, a settle's asked count) that the count lacks are
  removed, then the file written when the count has it. A grow and a shrink are one work
  range, `[w0, max(files(N), files(old)))`, a shrink's from the top down, which replaced the
  separate removal pass (`remove_old`) and the host slices (`work_slice`).
- *Settle through the resume path.* `settle_manifest` turns the manifest into one at the
  settled count with what is left (`progress.settle`), written before the gate as a grow's
  is, and the resume path finishes it; an interrupted settle is finished as it is. Strays
  between the boundary and the mark are removed past the count and written again below it,
  so a shrink is never continued: a shrink caught at a boundary of 13,000 with the mark at
  7,800 kept 13,000 files, removing what was in hand and rewriting the 5,200 a stray had
  taken down.
- *`stopped`.* A manifest written by a datagen that is exiting (Ctrl-C, or a failure)
  says so, and the live check lets it through at once; the progress total of a shrink counts
  to the larger end of its work.
- *Found on the way:* a grow's other ranks read the manifest as rank 0 was rewriting it and
  could refuse it as `growing`; ~~rank 0 now waits for every `Hello` before writing a grow's,
  shrink's, or settle's manifest~~ (superseded 2026-10-13: no rank but 0 reads it, below).

Tests: `tests/coord.rs` settles a datagen killed on two hosts and a grow killed on one at a
directory boundary, to the names, sizes, sampled bytes, and totals of that count, after the
live check refuses; settles a killed shrink without removing past the run in hand; Ctrl-C on
rank 1 of two and on one host, every host exiting 130, the manifest true and `stopped`,
settled with no wait; the killed datagen's band and mark. The builder's tests settle an
interrupted grow of every container format to a fresh datagen's bytes (HDF5's headers
included) and an interrupted shrink of a bucket (objects in flight deleted, a stray's sent
again). Next: subset runs (`run --subset`), then the §3.71 sizing step.

**Decided and built 2026-10-13 (the user): rank 0 alone reads the manifest at ROOT.** Asked
whether only rank 0 touches the manifest: it alone wrote it, but every rank read it for
`--resume`, `--grow`, `--shrink`, and `--settle`, ran the live check and the settle or resize
on its own, and the coordinator compared the results; the race above came from that. The user
noted a cluster runs one rank 0, so the manifest is its alone: rank 0 listens with every
`Welcome` held, reads and decides, and sends the manifest to run (and the one it replaces)
with `Welcome`; the other ranks run what they are given. One place decides the settled count,
the wait for every `Hello` is gone, and a refusal or a `complete` corpus reaches every host
from rank 0 (`Server::last_word`, rank 0 not connecting) instead of each host reaching it
alone. The identity becomes the operation, its count, and ROOT as each host names it, so
every host must name the corpus the same way; before, a host given a different path would
have been refused only because the manifest it read differed. Decided by the user the same day:
every host mounts the system under test at the same local pathname, so ROOT is spelled the
same on every host. Found on the way: a client
whose send broke because rank 0 had gone reported `Broken pipe`, not rank 0's reason; a send
now waits briefly for the reader to take the `Stop` that precedes the close. Not changed:
`aeiou run` still reads each dataset's manifest on every rank, a read-only check of a
`complete` corpus (`runner/REFERENCE.md` §5, §6).

### 3.71 Sizing a training corpus: a sizing step that writes a parameter file, scale-free layout, and the step barrier (proposed 2026-10-12, discussed with the user; revised 2026-10-13, §3.74: the sizing step, the layout rule, and the epochs are the WG's, in `mlpstorage`)

**Where it came from.** Building §3.70's whole directories, the user chose (2026-10-12) to
refuse any count that is not whole directories (option (a)) over rounding the count inside
the runner (option (b)). A survey of the committed workloads then showed (a) refusing the
defaults of `file_per_sample` (50,000,000 against directories of 1,300), `hdf5_map`,
`tfrecord_stream` and `parquet_stream` (1,281,167 against 1,024 a shard), and the
calibrated cases `imagefolder-jpeg` (3,200 files) and `npy-8mib`/`npy-140mib` (256 and 16
against 10,000), which cannot comply without changing what was traced. The user then
reopened (b), argued for rounding up the independent variable generously and showing every
dependent value before a datagen (the total capacity above all, so a user can see that a
corpus will not fit), and for layout counts that scale the way a data scientist would set
them. The discussion that followed, corrections included, is below. MLCommons' own
benchmark has the same three steps (`datasize`, `datagen`, `run`; its Rules.md), the user
noted.

**The arguments, as settled.**

- *Rounding is harmless at scale*, so it can be generous: 50,000,000 → 50,000,600 is 0.001%.
  The expected total of the sizes is exact enough to decide whether a corpus fits: the sum
  of `n` lognormal draws strays from its expectation by about `σ/√n` relative (≈ 0.01% at
  50 M); a small corpus can be summed exactly.
- *An epoch is a full pass* (the user's correction of their own first statement, "a few epochs
  read a small fraction"): with the Feistel permutation and `drop_last` every sample is read
  once an epoch, and the 5× rule exists because epochs reuse. Within a pass, shuffled access
  has no reuse, so every read is a first touch whatever the corpus size; corpus size then
  matters through the metadata working set (dentries, inodes, directory blocks, on clients
  and servers), the fullness of the system under test, and the data's age. So a subset run
  (§3.70 item 6) stands for a corpus of the subset's size on a system holding the whole: its
  metadata footprint is the subset's. **The age bias** (the subset is the oldest data
  written, and a system that tiers by age serves it from its coldest tier) must be stated
  wherever subset results are (the user: not thought of before).
- *Whole directories are not about skew.* With uniform shuffling every file is equally
  likely, so a short last directory is colder only in proportion to its size: noise among
  thousands of directories (it matters only for tiny corpora: 3,200 files in directories of
  1,300 leave the last one 46% as busy). Whole directories serve subsets, a clean ownership
  of directories by datagen's hosts, and growth; the user agreed this weakens the skew
  argument.
- *The calibrated cases stay exact.* A case exists to check its abstract against the trace
  that fitted it, so its counts are the traced ones; it says nothing about how a scaled run
  is laid out. Rounding inside the runner (option (b) as a definition) would quietly change
  the cases (3,200 → 3,900) and break that check. Hence the sizing step below: it gives
  (b)'s ease without touching any definition, fingerprint, or case.
- *The layout counts are the data scientist's choice, not the application's semantics*, so
  they may scale with the corpus. The user's terms: samples per file, files per directory,
  directories. Checked against primary sources the same day: TensorFlow's TFRecord guide
  ("at least 10 times as many files as there will be hosts reading data … at least 10 MB+
  and ideally 100 MB+" a file), Hugging Face's WebDataset page ("Each shard is often
  ~1GB"), Parquet's configuration page ("large row groups (512MB - 1GB)"), Transformers'
  `max_shard_size` (5 GB by default). For directory width there is no authoritative guidance:
  ext4's htree and XFS's hashed directories look a name up in `O(log n)`, so width costs
  elsewhere (a listing is linear; creates serialize on the parent; some scale-out systems
  place a whole directory on one metadata server; the directory blocks a client keeps
  resident). Since §3.70 item 6 replaced the walk of a large corpus with an inventory (the
  user's reminder), the listing cost applies only to small corpora, which favours wide
  directories at scale. Depth: a third level adds few directories (the leaves are files ÷
  width whatever the depth) but one more lookup per open; the user prefers one level made
  wider, which is also what the runner supports today.
- *ImageNet's 1,300 is a class size* (ImageFolder: a directory is a label; 1,281,167 images
  in 1,000 classes, 732 to 1,300 each), a fact of how the corpus was assembled, not a
  constraint. Practitioners grow a classification corpus by classes and do not inflate the
  class size, but taxonomies stop near 20,000 classes (ImageNet-21k about 21,000 for 14 M
  images; JFT-300M about 18,000 for 300 M, from memory of Sun et al. 2017, not stored as
  directories). So a class-structured corpus grows by classes up to a cap and then by class
  size; past about 26 M samples real practice is sharded, and a larger ImageFolder corpus is
  the metadata-heavy variant, to be labelled so, not "ImageNet at scale".
- *Comparability.* The layout counts change the operation stream (how many opens, read and
  `readdir` sizes), so a CLOSED result needs every submitter at one corpus size to get one
  layout: the WG fixes the *rule* (the targets below), never the totals, and the corpus grows
  with the system under test. Users outside the WG may set the counts directly.
- *Growth.* A path is a function of the files per directory and the samples per file, so
  changing either renames every file: a new corpus. With scale-free targets (bytes per file,
  files per directory) only the directory count grows, so a corpus grown fifty-fold keeps a
  layout in the norms (the user's caveat about small corpora grown large). The exception is
  a corpus whose file count is below the readers' floor at the larger scale; §3.70's
  original answer holds there: size the corpus for the largest step and run the smaller ones
  as subsets.
- *The barrier.* The user recalled an earlier exchange: gradient exchange in modern training
  overlaps the backward pass (DDP reduces buckets as they are ready; FSDP and ZeRO
  prefetch), so a stop-the-world wait added after `step_time` overstates the cost.
  Overlap hides the exchange's *time*, not its *dependency*: synchronous SGD cannot start
  step `k+1` until every GPU's step-`k` gradients are reduced, so a GPU that starts late
  because its loader ran dry delays every GPU, and the step is the slowest GPU's data wait
  plus compute. Coupling every step is the straggler effect a storage benchmark exists to
  expose; its cost must be only "wait for the slowest", the exchange's own time being inside
  the measured `step_time`. The loaders still read ahead across it (`workers × prefetch`
  batches). The default `sync_every = 500` most likely followed MLPerf's 500 steps.
- *MLPerf's numbers* (checked in MLCommons' Rules.md and Submission_guidelines.md the same
  day): the dataset at least 5× the memory of all its nodes; at least 500 steps an epoch
  (`500 × batch_size × num_accelerators` samples); `datagen` at least what `datasize`
  computed; a run phase of six runs, one warm-up and five measured. Neither document states
  a number of epochs a run; "5" may be the five measured runs. WG process, `PROJECT_BRIEF.md`
  §8. *Confirmed by the user the same day:* one warm-up run of at least 500 steps, then five
  independent runs of 500 steps each; the 5 is runs, not epochs. Since the dataset holds at
  least 500 steps an epoch, a run reads at most one pass, so a run has no reuse within it;
  reuse is across runs (each starts over), which is what the 5× rule defeats.

**Proposed** (the user agreed to all six points 2026-10-12; to be built after resume, which
does not depend on it). *Revised 2026-10-13 (§3.74):* points 1 to 3 are the WG's methodology
and belong to `mlpstorage`, the WG's driver, not to aeiou; struck below. Points 4 to 6 stay,
with point 1's projection moved into `aeiou dry-run` and the host facts into the report.

1. ~~*A sizing step* that writes nothing to storage: input one size (samples, bytes, or the 5×
   inputs: hosts, their memory, GPUs), the format, epochs, GPUs, and any layout count the
   user fixes; output a parameter file (`schema/README.md` §8) with the samples rounded up to
   whole directories (and whole files), the layout counts, and `steps` from epochs (`epochs ×
   ⌊samples / (G × batch)⌋`), and a printed projection: samples, files, directories, files
   per directory, samples per file, mean file size, expected total bytes, steps. `datagen`,
   `dry-run`, and `run` take the file as they take any parameter file, so the runner stays
   exact and the calibrated cases untouched. An `aeiou` subcommand in Rust (exact per-sample
   sums for small corpora; it sits beside `datagen`); its name is open (named for the
   mechanism; `datasize` is MLCommons' word for the same step).~~ *(Superseded 2026-10-13 by §3.74: the WG's, in `mlpstorage`; the projection is `aeiou dry-run`'s, without the rounding or the 5× inputs.)*
2. ~~*The layout rule.* Sharded formats: samples per file from the format's byte target
   (TFRecord 100 MB and up, WebDataset about 1 GB, Parquet a 512 MB to 1 GB row group),
   subject to files ≥ 10 × readers (GPUs × loader workers under `stream` access).
   Anonymous directories: one level, a width fixed by the WG (one constant, its value open:
   4,096 or 10,000), wider before deeper. Class-structured (ImageFolder): files per
   directory the class size (1,300) until the class count reaches a cap of about 20,000,
   then larger classes. Applied once, for the target size: the layout is fixed for the
   corpus's life, and grow and shrink refuse to change it.~~ *(Superseded 2026-10-13 by §3.74: the targets are the WG's, applied by `mlpstorage`; aeiou keeps the parameter names and `div` by a parameter, point 6.)*
3. ~~*Epochs.* The sizing step derives `steps` from a step count or from epochs; the run
   structure is the WG's (§8): MLPerf's is one warm-up run of at least 500 steps and five
   independent measured runs of 500 steps, each at most one pass.~~ *(Superseded 2026-10-13 by §3.74: the WG's run structure, applied by `mlpstorage`; `steps` is an ordinary parameter here.)*
4. *The barrier.* `sync_every` defaults to every step for synchronous data-parallel
   training, adding no time of its own; the coordinator's barrier latency is measured (2, 8,
   32 hosts) to confirm it is small against `step_time`, and if it is not, the fix is a
   cheaper release, not a rarer barrier.
   *(Built 2026-10-14.)* The five training workloads (`file_per_sample`, `npy_per_sample`,
   `hdf5_map`, `tfrecord_stream`, `parquet_stream`) default `sync_every` to 1; only their
   AST hashes changed (a barrier is no op, so no fingerprint and no dataset id did). The
   measurement, an abstract of 2,000 steps of `compute(0)` and `barrier global` with four GPUs
   a rank, found a bug first: on one host 2,000 barriers of eight GPUs took 104.8 s, 52 ms
   each. `Local::barrier` waited out its 50 ms poll before looking at the generation, and the
   arrival that completes a generation releases it inside `arrive`, so that thread, with no
   one left to notify it, slept 50 ms every barrier and every other GPU waited for it at the
   next. (`Tcp::barrier` looks first; the event loops use `arrive` and `released` and never
   had it.) Under the old default a run paid it once; at one barrier a step it would have
   added 50 ms to every step, half of `file_per_sample`'s 105 ms `step_time`. Fixed (look,
   then wait), with a test that fails at 1.07 s for what now takes 0.01 s. Then the latency,
   every rank a process on one 20-core host over loopback, a barrier being `elapsed / 2,000`:

   | ranks (GPUs) | `sync` | `io_uring` |
   |---|---|---|
   | 1 (8) | 0.18 ms | 0.17 ms |
   | 2 (8) | 0.11 ms | 0.12 ms |
   | 8 (32) | 0.25 ms | 0.31 ms |
   | 32 (128) | 0.79 ms | 0.90 ms |

   (`sync` after the cheaper release below; `io_uring` before it.) The 32-rank row runs 128
   GPU threads, 32 reader threads, and the server's 32 on 20 cores, so it is mostly
   scheduling; real hosts add a round trip to the server and back (tens of microseconds on
   a data-centre network) and remove the oversubscription. Against `step_time` (105 ms for
   `file_per_sample`, 323 ms for `npy_per_sample`) that is under 1% at 32 hosts. A cheaper
   release, done on the way: a frame is encoded once into one buffer and written in one
   `write` (it was two, so two segments under `TCP_NODELAY`, and a broadcast encoded the
   message once per host), 300 → 250 µs at eight ranks and 866 → 785 µs at 32. Not done:
   a release tree or binary frames, which nothing measured asks for yet. A workload whose
   step is a few milliseconds would; measure it on real hosts first.
5. *The capacity check.* `datagen` refuses when the expected bytes plus a margin exceed the
   free space `statvfs` reports for a directory root; an object root cannot be checked and
   is warned about.
6. *Terms.* Defined once in `schema/README.md` and used everywhere: **sample**, **file**,
   **directory**; `samples`, `samples_per_file`, `files_per_directory`, and the derived
   `files`, `directories`; standard parameter names across the training abstracts, passed as
   parameters (they are workload identity), ~~the sizing step's flags the same names~~ (2026-10-13: and `mlpstorage`'s, §3.74). A
   contract extension lets `div` in a pattern take a parameter (`{id div
   files_per_directory}`; today the width is a literal, so the directory *count* already
   scales but the width cannot). *(Built 2026-10-13, contract 0.10, §3.75.)*

**Not changed by this.** §3.70's resume, grow, and shrink take any count (whole directories
are not required of a corpus); `aeiou run --subset` requires whole directories of the
subset's count. Other workloads (checkpoints, model loading, KV caches, vector indexes,
dataprep) have no such layout; their sizes come from domain parameters, and they get the
same projection in a later pass.

### 3.72 One dataset shape: `layout`, `access`, and `format.framing` (decided with the user 2026-10-13; built the same day, contract 0.8)

**How it came up.** Designing `aeiou run --subset` (§3.70 item 6), whose rule admits subsets of
`map` and `stream` datasets and refuses a `regions` dataset and a dataset without an access
mode. The user asked why, and whether "regions" and "no access mode" were community terms.
Neither is: `regions` was coined here (`ABSTRACTS.md` §9.6), and "no access mode" is an absence.
The community terms are on the consumed side only: `map` and `stream` are PyTorch's
*map-style* and *iterable-style* datasets.

**Two axes, tangled.** A `regions` dataset was described as a layout (one file of `count`
fixed slots, read with `pick`), but the relationships the user had in mind (graph edges, list
membership, reuse) were never in its definition; they are in the abstract's access pattern
(positional distributions, `x @ i`). The datasets "without an access mode" (the safetensors
shards and config files, LMCache's blocks, the DiskANN PQ file, the FAISS index file) were
`files` in layout and looked up by id in access, like a `regions` dataset. So the line that
matters is *consumed versus looked up*, and `regions` was one layout of the second.

**Access (the user's terms).** `map` and `stream`: drawn from the positional shuffle, each
item once an epoch, interchangeable. `indexed`: the application looks each item up by id
(`file`, `pick`); "the items are not anonymous" (the user). `access` is now required, so no
dataset is silent about it, and `consume` takes a `map` or `stream` dataset only (V21).

**Layout (the user's hierarchy).** Storage is a file or an object (`protocol`, unchanged); a
unit of storage then holds one item (`whole`), part of one (`chunked`), several variably sized
items (`compound`), or fixed-size slots (`array`). The user chose `compound` because it is
deliberately vague on alignment and `array` because arrays store fixed-size entries, so the
contrast shows in one field. "Shard" was considered for the split case and rejected: in ML
usage a shard is a file holding many samples (a TFRecord or WebDataset shard), our `compound`.
`chunked` matches the existing `chunk` field.

**No kind key.** An earlier proposal (this conversation) kept `files` and `array` as two kinds
beside `layout`; the user's point that nobody would tell kind from layout without reading the
documentation removed it. Every dataset has a `pattern`; an array in one file is a pattern
without fields. "One file" is not what makes an array: it is a limit of today's contract (the
three committed arrays are each one file), and real systems shard the same structure (a vector
database's index per segment, a partitioned DiskANN, Lucene segments, RocksDB SSTs). A sharded
array would be a pattern with `{id}` under `array`; not built, and refused until it is (V20).

**Fixed against variable size.** Asked by the user whether it matters: it decides where an
item's offset comes from (an array's from the id alone, `c × slot`, as a database computes a
page; a compound file's from its format's framing and the sizes before it) and what storage
sees (aligned reads at a fixed stride over a sparse file, the slot's tail unwritten, against
unaligned reads of variable length). That is real workload shape, but a property of how
offsets are found, not of what the dataset is, so it is the difference between two layout
values and not a separate axis.

**`format.framing`.** The container's headers, footers, alignment, and columns were
`format.layout` (contract 0.2). With `layout` naming how a unit holds items, the old name
would have meant two things in one definition; `framing` is also the better word for headers
and alignment. The runner's `eval::Layout` and the builder's `aeiou/layout.py` follow
(`eval::Framing`, `aeiou/framing.py`). §3.71's "layout counts" (samples per file, files per
directory, directories) are the counts that shape a corpus's tree, not this field.

**Consumed datasets' directories: `{id div D}` or none (V21).** The user: if a consumed
dataset has directories at all, they are `div`. The shuffle makes every directory equally
likely whatever the scheme, so `div` against `mod` is moot for a training read's load, and
`div` is what makes a subset whole directories. Asked when `mod` would be better: only for
access in id order (a single writer or a sequential scan, under `div` serialized on one
directory and, on some scale-out systems, one metadata server; growth, which under `mod`
fills every directory evenly), and a consumed dataset has none of it (our own datagen spreads
creates by handing out whole directories per rank). Real applications that spread by name
hash rather than use `mod`, and they write namespaces. `mod` stays in the pattern language
for `indexed` datasets and namespaces, and a future workload that fits it will use it. A flat
pattern (one directory, or a bucket prefix) is allowed: shard corpora are usually flat.

**Decided on the way, for `run --subset` (~~built next~~ built 2026-10-13, §3.70 item 6).**
- A subset means a smaller corpus: the run is the same workload as a fresh corpus of the
  rounded count (same op stream and fingerprint), so subsets are whole `div` directories from
  0, or whole files of a flat dataset; a flat subset whose abstract lists the directory is
  refused (the listing returns the whole corpus). `--subset` with no dataset smaller than its
  manifest is a usage error.
- *Selection from a corpus* is another workload, recorded as a candidate: data scientists
  train on a subset to study data quality or bias (the user), chosen by content and so
  scattered over every directory; the walk lists the whole corpus (a `torch.utils.data.Subset`
  wraps an ImageFolder that has walked everything) and the reads are a positional selection of
  N out of C. It belongs in an abstract (a corpus count and a selected count), not a runner
  flag; not built.
- The inventory variant (§3.70 item 6) comes after `--subset`; whether a subset reads the
  corpus's inventory whole or a list cut for it is decided then, with the selection workload.
  (Decided 2026-10-13, §3.73: the list a corpus of the run's count would have; the whole list
  is the selection workload's read.)

**Contract 0.8.** `datasets.<name>` is one object: `pattern`, `layout`, `access`, `count`,
`size`, `seed`, and by layout `chunk` (`chunked`), `samples_per_file` and optionally `format`
(`compound`), `slot` (`array`); `protocol` and `doc` as before. V20: the fields a layout takes
and needs, its pattern's fields (`array` none, `chunked` `{id}` and `{k}`, the others `{id}`).
V21: an `array` is `indexed`; `consume` needs `map` or `stream`; a consumed dataset's
directory part has no `mod`. The mapping from 0.7:

| 0.7 | 0.8 |
|---|---|
| `{"files": {…}}`, one sample per file | `layout: whole` |
| `{"files": {…, "chunk": c}}` | `layout: chunked` |
| `{"files": {…, "samples_per_file": n}}` | `layout: compound` |
| `{"regions": {"file": f, …}}` | `layout: array`, `pattern: f`, `access: indexed` |
| no `access` | `access: indexed` |
| `format.layout` | `format.framing` |

**What changed and what did not.** Every abstract's hash and every dataset id (the resolved
definition is the new shape; `manifest_version` 3, so an older manifest is refused by its
version, not by a diff). No fingerprint (`tests/golden.rs`, every committed abstract),
no byte of data (the payload is keyed on seed, id, and offset), and no measured shape (the
catalog's shapes changed in their hashes only). A dataset's draw sites are now the JSON
pointers of the flat shape (`/datasets/d/size`), as the site rule requires; no committed
dataset has a draw in its expressions. Built in the runner (`ast.rs`, `validate.rs`,
`eval.rs`, `sites.rs`, `payload.rs`, `datagen.rs`), the schema and `schema/check.py`, the
builder (`Workload.dataset(…, layout=, access=, slot=)`; `Workload.regions` removed), every
committed abstract, and the documents; tests for V20 and V21 in `tests/golden.rs`.

### 3.73 The inventory: a file list of the corpus, read in place of the walk (decided with the user 2026-10-13; contract 0.9)

§3.70 item 6 decided that a large corpus is listed from an inventory captured once, not walked,
and left two things open: what the inventory is in the abstract, and what a `--subset` run
reads of it (§3.72). Discussed with the user 2026-10-13.

**No trace has one yet.** Every committed walk was traced from ImageFolder or a glob; no
traced job reads a file list. So the inventory's format (text, Parquet, gzip'd CSV), its
columns, its header, who reads it (every process, or rank 0 then a broadcast), and its read
size are guesses until a real inventory-reading loader is traced. The user chose to build the
mechanism now with the guesses labelled `[measure]` ("we know that an inventory is
essentially a required feature for large datasets, so let's get closer than we are now and
refine it as we get more information"). The walk cannot be scaled into an inventory: it says
nothing of the inventory's shape. Once one is traced, a header plus `count × entry` (names are
fixed-width) scales it to any count, and down for a subset.

**Decided:**
1. *The inventory is a key of the dataset it lists, not a dataset.* V13 gives every dataset a
   root of its own, so a separate inventory dataset would need another root and another
   manifest that the subset logic would have to tie to the corpus's. Instead
   `datasets.<name>.inventory` is `{name, entry}`: `name` a constant file name (no fields, not
   beginning with `.aeiou`, not one the pattern can produce) directly in the dataset's root,
   `entry` the bytes per listed item. Its size is `count × entry` (`count` the dataset's,
   files or samples by layout: an inventory lists what the job draws). It is part of the
   resolved definition, so of the dataset id and the manifest. A dataset carries no access
   parameters; how the inventory is opened (buffered or `O_DIRECT`) is the abstract's, as for
   any file: a Python reader (`f.read()`, pandas, pyarrow) is buffered.
2. *Datagen writes it as payload*, with the corpus's ratios, keyed on the dataset seed and a
   ~~file id of its own (`u64::MAX`)~~ label of its own (`labeled_key(seed, "inventory", [b
   div D])`; a reserved id could meet a corpus's under dedupe), so its bytes at an offset do
   not depend on the count and a grow only extends it. It is written after the last id and before `complete`, by
   every datagen that ends complete (first, resume, grow, shrink, settle), at the final
   count: an inventory is captured after the corpus is written.
3. *Only a consumed dataset (`map` or `stream`) has one* (V22): it stands in for the walk of
   a corpus the job draws from. A whole or a chunked or a compound dataset can; an `array`
   cannot (it is `indexed`).
4. *The abstract reads it through a handle*, `{"inventory": "<dataset>"}` (builder:
   `train.inventory()`), opened read-only like any dataset file, its known size the formula's.
5. *Under `--subset` the run reads the list a corpus of the rounded count would have*
   (option A; option B, the whole corpus's list, is the selection workload's read, §3.72).
   Entries are fixed-width and in id order, so that list is the first `run × entry` bytes of
   the corpus's inventory, and no other file is written. The fingerprint hashes each read's
   requested length, as always, so it is a fresh corpus's. The length *issued* is capped at
   the subset's size (the user: "limit the read operations we issue to the size of inventory
   file we're simulating"): the tail read is short, as it would be on a file that size, and
   the read that finds EOF on one (a reader looping until `read` returns 0, which it must,
   since a file of a whole number of buffers has no short read) is issued with length 0, so
   the op count is the fresh corpus's and no byte past the subset's list is asked for. The
   cap applies to the inventory under `--subset` only. What the runner cannot stop: a
   buffered client's readahead may fetch past the subset's end (solution behaviour), and an
   `O_DIRECT` reader's unaligned tail is rounded out by up to one alignment unit.
6. *The committed mechanisms that walk* (`file_per_sample`, `npy_per_sample`) get a parameter
   `inventory` (false by default) and their `enumerate` parameter is renamed `walk` (no
   alias, no users yet); both off, the job lists nothing. Parameters are scalars, so the two
   are booleans rather than one three-valued `listing`. The inventory read: every GPU's
   process opens, `fstat`s, reads `inv_read` bytes (`[measure]`, 1 MiB) until EOF, and closes;
   `entry` defaults to the name's width plus a newline (`[measure]`).

**The known refinement for objects.** S3 Inventory is AWS's, not a standard, but Google Cloud
(Storage Insights inventory reports) and Azure (blob inventory) built the same thing: the
provider writes, daily or weekly, into a destination bucket, a `manifest.json` (source
bucket, format, column schema, and the list of data files with sizes and MD5s) plus data files
in gzip'd CSV, ORC, or Parquet (AWS; Google and Azure CSV or Parquet). So an object dataset's
real inventory is a manifest and several files, possibly in another bucket, read by the
application's own code (fetch the manifest, then each file, usually through pyarrow). Until a
job that reads one is traced, an `s3` dataset's inventory is the same single object at its
prefix.

*Built 2026-10-13.* Runner: `ast::Inventory` and `Handle::Inventory`, `eval::InvMeta` on a
`files` dataset (its size `count × entry` follows `with_count`), `HVal::Inventory`; the cap is
`Op::capped` and `Op::issue_len` (the length a backend issues; `len` stays the length asked and
the hashed one), set through the target, the open file, and the read sequence, and
`Model::cap_inventory` per dataset `--subset` runs smaller; every backend's read issues
`issue_len` (`run.rs` for `sync`, `posix-aio`, `mmap`; `uring.rs`; `aio.rs`; `object.rs`). V22 in
`validate.rs` and `schema/check.py` (a field of the sibling component read as the characters an
integer formats to, so `inventory.txt` beside `{id div 1300:05}` directories is allowed). Datagen:
`write_inventory`, rank 0, before `complete`, at the count in the manifest written (so a settle's
count), keeping a direct file's whole blocks; `payload::inventory_block_seed`. The Python
`aeiou-datagen` refuses a dataset with an inventory. Builder: `Workload.dataset(…, inventory=)`,
`Dataset.inventory()`, `InventoryHandle`. `file_per_sample` and `npy_per_sample`: `enumerate`
renamed `walk` (and its phase), an `inventory` phase off by default (`inv_entry` 24 and 27,
`inv_read` 1 MiB, `[measure]`), named `inventory.txt`. Every committed AST regenerated (contract
0.9): no fingerprint changed; those two corpora's dataset ids changed (the definition holds the
inventory). Tests: `tests/golden.rs` (V22, and the capped reads: the same ops and fingerprint,
issued to the subset's end); `tests/coord.rs` (a corpus's inventory read whole; a subset's on
one and two ranks under `sync`, `io_uring`, and `libaio`, its fingerprint the dry run's at the
rounded count, its ops and bytes; without the cap the run fails as predicted, "expected 62400
bytes, got 76800"; grow, shrink, and settle leave the fresh corpus's inventory); builder
`test_an_inventory_is_a_constant_name_in_a_consumed_datasets_root`.

### 3.74 The sizing step is the WG's: aeiou is the workload generator, `mlpstorage` the methodology (decided with the user 2026-10-13)

**The question (the user's).** Before building §3.71's sizing subcommand, the user asked whether
this tool needs it at all. `fio` has nothing like it: it takes `size=` and `nrfiles=` and never
asks why. The 5× rule may belong to MLPerf and not intrude here. `datagen` also parallels
MLPerf, but making the corpus a run needs is within the tool's scope. Would a calculator that
asks every host for its memory and applies the 5× rule on rank 0 be an advantage?

**The assessment (mine, agreed by the user).**

- *Choosing a test's size is methodology.* The 5× rule exists to defeat client caching in a
  CLOSED submission (§3.67); it is WG process, beside divisions and published hashes
  (`PROJECT_BRIEF.md` §8). A researcher may want a corpus that fits in cache (§3.67 item 1:
  small datasets are real). An engine that sizes the test enforces one methodology on every
  user.
- *Checking is stronger than calculating.* A sizing step trusts what it is told: the hosts,
  their memory, which of them will run. Datagen's hosts need not be the run's. What the WG
  needs is evidence that *this run's* hosts read a dataset at least 5× their memory: facts
  collected at run time, in the report, judged afterwards. That is reporting, which is
  generic, plus a rule, which is not.
- *datagen is different.* A run cannot happen without its corpus, and the corpus's names,
  sizes, bytes, and manifest are fixed by the runner's definitions. Writing it is the tool's
  job; choosing its count is the user's input, as in `fio`.
- *What is lost:* one place that turns "32 hosts" into a parameter file, unless the WG
  writes it. It does: below.

**`mlpstorage` (the user's context).** MLCommons' `mlpstorage` command wraps the workload
generators the benchmark uses today across all the hosts and applies every WG-specific probe
and rule. It is to be changed to drive aeiou as the workload generator of all four workloads
MLPerf Storage supports today, replacing the three generators it uses now (DLIO for training and
checkpointing, a Milvus-based harness for vector search, and a KV-cache simulator). So the
sizing step has a home that is already the WG's: `mlpstorage` probes the hosts, applies the 5×
rule, the layout targets, and the run structure, and writes the parameter file (the form
`aeiou-params` writes, `schema/README.md` §8) that `aeiou datagen` and `aeiou run` take. It runs
each phase as its own `aeiou` process (`PROJECT_BRIEF.md` §8, *Driver integration*).

**Decided.**

1. *Not built here:* §3.71 points 1 to 3 (the sizing subcommand, the layout rule's targets, the
   epochs and run structure), struck there. Their open questions (the subcommand's name, a
   directory width of 4,096 or 10,000, the class cap) are the WG's to answer in `mlpstorage`.
2. *The host facts in the report* (`PROJECT_BRIEF.md` §6 item 29): every rank's memory, CPUs,
   kernel, and the GPU ids it emulated, sent to rank 0 with its Hello and listed in the merged
   report, with the datasets' bytes beside the hosts' total memory. ~~Reported, never judged:
   the 5× check is `mlpstorage`'s, read from `run --report-json`.~~ Judged by aeiou since
   2026-10-21 (§3.78 point 10), by the rule a WG rules file carries; aeiou hard-codes none.
3. *A projection in `aeiou dry-run`:* each dataset's samples, files, directories, mean file
   size, and total bytes, summed exactly from the size expressions (the same evaluation
   `datagen` does, without the payload), so a user sees whether a corpus fits before writing
   it. No rounding and no rule: the numbers follow from the parameters given.
4. *The capacity check in `datagen`* (§3.71 point 5): refused before anything is written when
   the projected bytes exceed what `statvfs` reports free at a directory root; an object root
   cannot be checked and is warned about.
5. *Kept from §3.71:* point 4 (the barrier: it is the abstract's fidelity to synchronous
   training, not the WG's choice) and point 6 (the terms, and `div` by a parameter, so any user
   can scale a corpus's directory width; `mlpstorage` sets the same parameter names).

**Decided while building (2026-10-13).** The projection's bytes are exact and unrounded: a
block-rounded "allocated" figure was tried and dropped, because an NFS client's `statvfs`
reports its transfer size (often 1 MiB) as `f_frsize`, which would overstate a corpus of
128 KiB files eightfold and says nothing of the server's allocation. So the check refuses on
the exact bytes against `f_bavail × f_frsize` and warns above 90% of it, metadata uncounted.
It runs on rank 0 alone, since the roots are the shared storage under test, and its refusal
reaches every rank as an empty-root refusal does. `--[no-]ignore-space` overrides it, for a
system under test that stores less than it is sent (the payload's `--dedupe` and
`--compress`). It is fixed (command line only) as a safety check, like `--ignore-limits`. The
memory block is computed after the run, so no host's gate waits on a projection (0.34 s for
`file_per_sample`'s 50,000,000 files on 20 threads, 1.2 s for DiskANN's 10⁹ slots).

*Built 2026-10-13* (items 2 to 4). `project.rs` (`Projection`, `project`, `project_all`: files
with chunks counted, an array's written slots, the inventory's bytes included and shown
apart, summed over threads). `dry-run` prints a `dataset` line per dataset. `run`:
`run::host_facts` (host, kernel, CPUs, `MemTotal`, the tightest cgroup v2 `memory.max`) goes in
each `Hello` (`Msg::Hello.facts`, `Server::facts`), and the report gains `facts`,
`hosts[].facts`, and `memory` (`run::memory_block`, printed by `run::memory_line`). `datagen`:
`datagen::check_space` and `DatagenOpts::ignore_space`. Tests: every corpus `tests/run.rs`
writes is held to its projection (files and bytes, chunked, array, and whole datasets);
`a_corpus_larger_than_the_free_space_is_refused_before_anything_is_written`; the report's
`facts` and `memory` against the bytes on disk, and the dry run's line (`tests/report.rs`); every
host's facts on rank 0 of two (`tests/options.rs`). Not yet: the Python `aeiou-datagen` has no
space check; ~~item 5's `div` by a parameter is next.~~ item 5's `div` by a parameter built the
same day (contract 0.10, §3.75).

### 3.75 `div` by a parameter and the standard layout names (decided with the user 2026-10-13; built the same day, contract 0.10)

**The problem.** §3.71 point 6 and §3.74 item 5: a pattern's divisor was a literal
(`{id div 1300:05}`), so a corpus's directory *count* scaled with `count` but its *width*, the
files a directory holds, was fixed in the abstract. Who sizes a corpus (a researcher, or
`mlpstorage` applying the WG's layout targets, §3.74) must be able to set it as a parameter, under
a name every training abstract shares.

**Options weighed.** (a) A parameter's name in the pattern string, `{id div
files_per_directory:05}`. (b) A structured pattern, `{text: "…{id div $D}…", args: {D: <expr>}}`,
taking any expression. (b) is more general, but no one needs an expression here (the width is
computed by whoever sizes the corpus, and given as a number) and every pattern would read worse.
**Decided: (a).**

**Decided.**

1. *Syntax.* The word after `div` or `mod` is a positive number or the name of a parameter; the
   two cannot be confused (an identifier starts with a letter or `_`), so the name takes no sigil.
   Datasets and namespaces alike. `mod` by a parameter is allowed where `mod` is (an `indexed`
   dataset, a namespace); a consumed dataset's directories stay `div` (V21).
2. *Rule V23.* The name is a declared parameter whose default is an integer (`validate.rs`,
   `schema/check.py`, the builder). `gpus` is not a declared parameter, so it cannot be a divisor:
   a corpus's names do not depend on how many GPUs read it. A value of zero or less, or not an
   integer, is refused when the model is built, naming the parameter.
3. *Bound before use.* `Pattern::bind` replaces each name with its value when the model is built
   (`eval::bind_pattern`); a bound pattern's source is the original text with the name replaced,
   and formatting an unbound one is an error. The directory scheme, `dirs`, `dir`, datagen's runs,
   and `--subset`'s rounding all see the number.
4. *The resolved definition says the number.* `resolved_dataset` and `resolved_namespace`
   substitute `{"param": x}` nodes, which would not see a name inside a string, so a corpus made at
   width 1,300 would have matched a run at 4,096. They now bind the pattern too
   (`payload::bind_pattern`, and the Python `resolved_dataset`), so another width is another
   dataset (a run is refused, `--subset` included, naming the pattern), and at the default the
   resolved pattern is exactly the literal one: **no dataset id changed**, and no fingerprint.
5. *Padding stays literal.* `:05` is a minimum width; a narrower directory over a large corpus
   gives more than 99,999 directories, whose names print wider and no longer sort by name in id
   order. Harmless to the runner (directories are enumerated by id, never sorted); noted rather
   than given an automatic width.
6. *The terms* (§3.71 point 6), defined once in `schema/README.md` §2: **sample** (an item a job
   draws), **file** (a file or object of the dataset), **directory**; the parameters `samples`,
   `samples_per_file`, `files_per_directory`, and the derived (never parameters) `files` and
   `directories`. Renamed: `per_shard` (`tfrecord_stream`, `parquet_stream`) and `per_file`
   (`hdf5_map`) to `samples_per_file`; `files` (`file_per_sample`, `npy_per_sample`, whose
   items are whole files) to `samples`; those two gained `files_per_directory` (1,300, ImageNet-1k's
   most images in a class; 10,000), as their kits' corpora are laid out. The three cases'
   `params.json` follow. No alias for the old names (no users yet).

*Built 2026-10-13.* Runner: `pattern.rs` (`Field::param`, `Field::spec`, `Pattern::params`,
`Pattern::bind`), `eval::bind_pattern`, `payload::bind_pattern`, V23 in `validate.rs`;
`AST_VERSION` 0.10. Builder: `aeiou/pattern.py` (`params`, `bind`), `Workload._divisors`, the
`FIELD` pattern accepting a name, `datagen.resolved_dataset`. Every committed AST regenerated:
every AST hash changed, no fingerprint, no dataset id; the catalog's shapes changed in their
hashes only. Tests: `tests/golden.rs` (V23; the binding, the directory count, the id at the
default and at another width, a zero width refused), `tests/coord.rs` (`--subset` at another
width refused), `test_builder.py` (V23 in the builder and `check.py`; the Python binding).

**Found on the way, fixed.** (i) The Python `aeiou-datagen`'s `Hello` had no `facts`, which
§3.74 made a required field of the frame, so a Python rank was refused by a Rust rank 0
("missing field `facts`"); `coord.host_facts` now sends what `run::host_facts` does. (ii)
`test_coord.py::test_the_frames_are_the_runners` wrote `file_per_sample` at its default 50,000,000
files, which §3.74's space check refuses on rank 0 before the identities are compared; it now
writes 100. Neither showed in CI, whose Python job builds no runner, so the test was skipped.
(iii) So CI fails what it skips (asked by the user the same day): `AEIOU_REQUIRE` in
`builder/tests/conftest.py` names the capabilities a job built (`runner`, `formats`, `object`,
`strace`, `header`, `minio`, and the Python extras `py-formats` and `py-s3`, which
`pytest.importorskip` skips for), and a test skipped for lacking one of them fails, naming it. The
integration job now runs the whole Python suite against its binary with
`runner,formats,strace,header` (`test_coord.py`, `test_payload.py`, and `test_usage.py` had run
in no job), the object job `test_object.py` with `object` and `minio` added; the `abstracts` job
keeps its runner-less run, where those skips are right. `test_require.py` holds the hook to that
and requires every skip in the suite to name a capability, so a new one cannot escape it.
*Revised 2026-11-02 (the user's choice, of two this side offered):* the `abstracts` job reported
579 passed and 227 skipped, all of them run by `runner-integration`, so it now names what it is
without, `AEIOU_WITHOUT=runner,formats,object,strace,header,minio`; a test marked `skipif` for
one of them is deselected and counted by capability in the summary (`AEIOU_WITHOUT: 228
deselected, needing minio 15, runner 213`), never a skip, and a test that skips for one at run
time fails, so the seven run-time skips known before a test runs (the runner or `aeiou-trace`
not built, no MinIO) became marks (`test_require.py` holds both). The other choice, the runner,
`strace`, and MinIO built into the quick job, would have run the integration job twice.

(iv) The Rust tests fail what they lack (asked by the user 2026-10-14). A Rust test has no
skipped state: the three MinIO tests in `tests/object.rs` returned early when `TEST_MINIO_BIN`
was unset, and `tests/uring_knobs.rs` passed over a row whose ring setup the kernel refused
(SQPOLL, DEFER_TASKRUN), each printing a message that cargo and nextest hide for a passing
test, so the test passed with nothing tested. Both now panic, naming what is missing. A build
with the cargo feature `object` therefore needs `TEST_MINIO_BIN` for its tests, as CI's
`runner-object` job has it. Checked the same day: every one of the 123 Rust tests a local
build with `object,formats` lists passed in some CI job of run 52d050b (119 under `formats`,
8 object and endpoint tests under `object`), and the Python suite's 263 ran between the
integration job (248, and `test_object.py` skipped for `obstore`, which that job does not
install) and the object job (15); every early exit in the Python tests is a `pytest.skip`.

### 3.76 The abstract in prose (decided with the user 2026-10-14; built the same day)

**The problem.** `PROJECT_BRIEF.md` §6 item 27: an AST is a compact, machine-readable account of
a workload's I/O, and the readers who most need that account (WG reviewers, submitters, a
catalog's visitors) will not read the JSON. The item fixed the goal (generated, never written;
checked like the man pages; two lengths; deterministic templates before any language model) and
left the design open.

**Decided** (proposed and accepted 2026-10-14):

1. *Where.* `aeiou-catalog`, which already loads the records, the shapes, and the dictionary;
   the walk is `builder/aeiou/describe.py`. It reads the AST (never the authoring script: the AST
   is the contract), the parameter values in effect, the records, and the shape.
2. *What it writes.* `DESCRIPTION.md` in every case directory (`DESCRIPTION.defaults.md` in a
   mechanism with no case, as `shape.defaults.json`), and each case's paragraph in `CATALOG.md`.
   `write` writes them; `check` re-renders and fails on a difference, so a parameter change is
   a prose diff in review. Not `ABSTRACT.md`: here "abstract" already means the AST.
3. *Two lengths.* A paragraph: the AST's `doc`, where the catalog files it and what the case
   is, what each dataset holds, what each actor does in order, and the measured shape. An
   account: every parameter with its value, default, unit, and `doc`; every dataset and
   namespace; every actor's body in order with every op named (local calls like `lseek` and
   `ioctl` included: the abstract is the application's call stream); the shape in full.
4. *Numbers.* An expression over parameters and literals is worked out and shown with its
   formula (`1 MiB (hdr_read)`); one that depends on a loop index, a binding, or a draw stays a
   formula. A distribution is given in words with its 5th and 95th percentiles computed from its
   parameters (and its bounds), never sampled.
5. *Coverage.* `test_catalog.py` holds the describer's lists of statement, expression, handle, and
   distribution kinds to the schema's, and renders one synthetic AST holding every kind, so a
   contract change cannot leave the prose behind (as `test_man.py` holds the pages to `--help`).
6. *Not now.* Prose for a raw trace (the user's note in item 27) and any polishing by a language
   model.

**Choices made in the build, open to revision.**

- *The shape keeps what the prose quotes* (`shape_version` 2). The first seed's op counts by
  kind and its bytes read and written, and each dataset's projection (samples, files,
  directories, bytes), so the paragraph can say "3,200 files in 3 directories, 376.2 MiB" and the
  description stays renderable without a runner (`check --no-shapes`). `aeiou dry-run
  --metrics-json` gained `datasets`, the projection it already printed (`runner/REFERENCE.md`
  §10). Every shape was re-measured: the measured attributes did not change.
- *The paragraph follows this case.* A branch whose test is worked out from the parameters is
  kept or dropped (ImageFolder's walk is there, its inventory read is not; a barrier under `every
  1` is every step), a parallel of width 1 is its body, a loop of zero times and a compute of
  zero are left out. The account keeps every branch, marked taken or not here.
- *Ops in the paragraph* run together, consecutive equal ones counted (`lseek twice`), each run
  prefixed by the handle the last binding named (`on the next sample of train`, `on the file
  holding the next sample of train`). A long abstract gives a long paragraph (`dcp_load`'s item
  reads); it is the call stream, and shortening it would be interpretation.
- *Formulas.* Infix, parameters by name, `x@(i − d)` for a binding at an earlier index; `div` and
  `mod` inside another operator are bracketed (`trailer − (item_bytes[t] mod 64)`), since the
  precedence is not what a reader assumes.
- *The unit of a value* is the slot's (a read's length is bytes, a compute's is time), else the
  parameter's declared `unit`, carried through arithmetic (bytes over tokens times tokens is
  bytes).
- *A case of two runs* (a writer and its reader) is one file: both paragraphs, then each
  account.
- *`aeiou-catalog describe AST`* prints the same for any abstract, at its defaults or a
  parameter file, its shape measured at seeds 1, 2, and 3 (`--no-shapes` for none).

**Found on the way, ~~not changed~~ fixed the same day in a commit of its own (the user's
choice).** A `compound` dataset whose pattern has a directory part got its directory count from
the sample count, not the file count: `Model::dir_scheme(pattern, count)` at model build and in
`DsMeta::with_count`, where `{id}` is a file id. `tfrecord_stream` with
`train/{id div 10:03}/shard-{id:05}.tfrecord` and 10,240 samples projected 10 files in 1,024
directories, and a walk over `dirs` would have listed 1,023 that do not exist. `dir_scheme` now
takes the file count, `ceil(count / samples_per_file)`; `tests/golden.rs`
`a_compound_datasets_directories_are_counted_from_its_files` fails without it. No committed
abstract has a compound dataset with directories, so no fingerprint, dataset id, or shape
changed; the describer, without a projection, counts directories from files as well.

### 3.77 Dedupe in 4 KiB blocks: the payload at its own file offsets, and sets of near sizes (`aeiou-positional/4`; found in the RC1 review and decided 2026-10-15)

**This changes the payload bytes, and so every dataset id** (the manifest's `payload.wrapper`
is `aeiou-positional/4`; a corpus written before reads as another payload and `aeiou run`
refuses it until it is generated again). No fingerprint, shape, or op count changes: the op
stream never depended on the bytes.

**What was wrong.** `--dedupe D` is documented as a ratio the generator honours, and §3.62
said every prefix and every extension has ratio `D`. That held for files of equal sizes and
nothing else. A storage system deduplicates fixed blocks (4 KiB here, `scratchpad/rv3c/dedupe.py`
of the review), and two things kept the shared bytes from lining up:
- *Containers.* A file's payload was its unit's stream laid end to end in the samples, the
  framing between them, so after the first sample the shared bytes sat at offsets that
  depended on the file's own sample sizes: TFRecord and Parquet at `--dedupe 2` measured
  1.068, while the manifest said 2. (HDF5, whose rows are contiguous after a header of one
  size, measured 2.0.)
- *Sizes that vary.* The `D` files of a unit (`id div D`) share its prefix, so its unique
  bytes are the largest file's: `file_per_sample` (lognormal, σ 0.45) measured 1.58 at
  `--dedupe 2`, and `lmcache_shared_fs` 2.33 at `--dedupe 3`.

**What changed.**
1. *The payload at its own offsets.* The byte at file offset `x` of a file of unit `u` is the
   byte at `x` of `u`'s stream (block `x ÷ 1 MiB`), and a container's framing is written
   over it instead of pushing it along. The files of a unit then share every 4 KiB block but
   those holding framing at differing offsets, and a block of a sample is the same in every
   file whose sample covers it. A chunked file was already so (its chunks at their offsets in
   the logical file); a whole file is unchanged.
2. *Sets of near sizes.* A file of one sample (`whole`, `chunked`, or a container of one) is
   in the set of `D` files of nearest sizes among its group of `G = 8·D` consecutive ids
   (`id div G`): the group ordered by (size draw, id), cut into sets of `D`, `u = group·8 +
   rank div D`. The draws are the ids' own, past the count too, so a file's set, and its
   bytes, do not depend on the count, and a grow, a shrink, a resume, and a run over a prefix
   are the same arithmetic as before (`builder/tests/test_formats.py` grows and shrinks a
   corpus of one-sample TFRecord files and compares it with a fresh one byte for byte). With
   sizes all equal, or `D = 1`, the rank is the id's place and `u = id div D`, so such files'
   bytes are `/3`'s. Files of several samples keep `u = id div D`: their sizes are sums of many
   draws and close, and ranking them would cost the framing of `G` files per group.

**Measured** at 4 KiB blocks (the review's five cases; `--dedupe 2` but the last):

| corpus | `/3` | `/4` |
|---|---|---|
| `tfrecord_stream`, 2,048 samples of 256 per file | 1.068 | 1.864 |
| `parquet_stream`, the same | 1.068 | 1.865 |
| `hdf5_map`, 2,048 rows of 256 per file | 2.0 | 2.0 |
| `file_per_sample`, 4,000 images | 1.581 | 1.851 |
| `lmcache_shared_fs` `sysp`, 60 prompts, 1 MiB chunks, `--dedupe 3` | 2.328 | 2.683 |

The compression ratio is unchanged (`--compress 2`: zlib over 64 KiB pieces 1.957 and 1.971
before, 1.957 and 1.974 after), and so is datagen's rate (1 GiB corpora to tmpfs, four
writers, best of three: TFRecord 6,140 and 6,089 MB/s, images 7,512 and 7,609, LMCache
chunks 4,764 and 4,789, before and after). `Filler` steps a block's generator over the framing
between two samples (up to 64 KiB) rather than making the block in its one-block cache, so
each block is still made once.

**What stays below `D`, and why it is not removed.** A block holding framing is unique in each
file (a TFRecord record's 16 bytes, a tar header, a Parquet row's 4-byte length): the smaller
the samples, the more such blocks. A set's largest file keeps its tail, by how much its sizes
differ; a larger group makes the sets' sizes closer (simulated over lognormal σ 0.45: 1.61,
1.81, 1.89, 1.93 for `G` = `D`, `4·D`, `8·D`, `16·D` at `D = 2`), but the last group of a
dataset is partial, its sets missing ids past the count, and a large group then costs a small
dataset: at 50 files and `D = 3`, `16·D` gave 2.67 where `8·D` gave 2.60 and `D` 2.11, but
at 20 files 1.54 where `8·D` gave 2.32. `8·D` was below `id div D` only at 20 files of
the LMCache sizes (σ 0.3: 1.66 against 1.74) of the counts tried (20 to 10,000 files). Exact `D` for files of varying sizes would need a file's place among
all the files of its size, a table or a pass over the ids, which the corpus's rules forbid
(no per-file structures; bytes a function of the dataset seed, the id, and the offset). The
manifest records `D`, the setting; the ratio a store sees is the table's.

**Considered and not done.** Repeating blocks within a file (`array`'s rule) for every file:
exact for large files, but a file under `D` blocks would not dedupe at all, and a repeat
within a compressor's window is compressed away (zstd's window is megabytes), so `--compress`
would no longer be the ratio. A projected ratio printed by datagen: exact for files of one
sample (the sets are known), not for containers without walking every framing, and the
table above tells a submitter what to expect.

### 3.78 Use cases, benchmark rules, and one id per run: layers that only narrow (discussed with the user 2026-10-21 and 2026-10-22; the points marked *agreed* decided 2026-10-21, the rest 2026-10-22)

**The question (the user's).** Unet3D on an H100 and Unet3D on a B300 are the same I/O
mechanism with a different batch and a different compute time per step: the GPU's memory sets
the batch (convergence second), and the batch and the GPU set the step time. A benchmark needs
one handle that names *everything* that goes into an `aeiou run`, so that results submitted by
several people are directly comparable, while a developer keeps the freedom to vary what a
benchmark would lock (say `--posix io_uring` against the traced `sync`). Is a library of named,
hashed parameter files that handle, and what hierarchy does it need?

**What exists, and what is missing.**

- The abstract's defaults are one reference set. `npy_per_sample`'s are the brief's Unet3D
  numbers (`batch` 7, `step_time` 323 ms, `sample_mean` 140 MiB, `steps` 500, `samples`
  50,000), so a run that forgets a value runs the H100 job.
- Parameter files (§3.27, `schema/README.md` §8) stack over the defaults, last one wins, and
  `--param` applies last. A file may pin its abstract's hash (`ast_sha256`, `aeiou-params --pin`).
- A case's `case.toml` binds an abstract to a parameter file and the GPU counts it holds at
  (§3.66), but it is a catalog record: the runner never reads it and nothing routes a run
  through it.
- A run's identity is the resolved values (`schema/README.md` §8), but nothing names that set
  or checks it before the gate.
- **The fingerprint does not cover compute** (checked 2026-10-21): `aeiou dry-run
  npy_per_sample.ast.json --gpus 1 --param steps=4 --param samples=64` prints `2af23f1b741599ad`
  with `step_time` 323 ms and with 100 ms. It is the multiset of positioned ops by definition
  (`runner/REFERENCE.md` §2), so `--expect-fingerprint` accepts a run with the wrong step time.

**The design.** Each point says whether the user agreed to it in the discussion (*agreed*:
decided 2026-10-21) or it is mine and still open (*proposed*). *2026-10-22:* every point
proposed that day was decided, as the notes in points 8, 9, 11, and 12 say, and points 17 and
18 were added and decided the same day.

1. *A value in a fragment is one the application decides* (agreed). How storage carries a
   request out (an NFS client's rsize split, the block layer's `max_sectors`) is the system
   under test's and is never a parameter: the interposition test (`PROJECT_BRIEF.md` §5). The
   user's correction of an earlier conflation: the largest transfer an API or a file system
   takes is an implementation question for the file system, not a value of the API layer.
   What sets each value, which is what decides where it is written:

   | Set by | Examples (`npy_per_sample`) | Written in |
   |---|---|---|
   | the application's structure | the calls, their order, the loops | the abstract (its AST) |
   | the library | `np_chunk` 256 KiB, `framing` 498, `cd_len` 102 | the abstract, as defaults |
   | the job | `sample_mean`, `sample_sd`, `steps`, `workers`, `prefetch` | the use case |
   | the model on a GPU (the user's point) | `batch`, `step_time` | the use case |
   | the API the application uses | `posix`, `cache`, `s3` | the abstract, as a default (point 4) |
   | the storage, as the application reads it | `hdr_read` = `st_blksize` | the storage, at the gate (point 11) |
   | the scale of a submission | `gpus`, hosts, `samples` | the submission |
   | the system under test | `--root`, `--threads`, io_uring knobs | the submission |

2. *Four layers, the last never shipped* (agreed):

   | Layer | What it does | Written by | Reuse |
   |---|---|---|---|
   | abstract | declares the parameters, their kinds, and the library's values | whoever modeled the application | one per mechanism |
   | use case | the job on a GPU (`unet3d-h100`, `unet3d-b300`); permissive | anyone | one per job × GPU |
   | benchmark rules | locks for comparability and the constraints a run must meet | the WG, for its benchmark | one file for many use cases |
   | submission | the required values and the system under test's settings | the submitter, per site | none; never shipped |

   A benchmark entry is a use case and a rules file. A plain use case runs without rules, for
   development and research.

3. *Two declarers* (agreed; the user's correction of "the abstract declares every input"). The
   abstract declares the parameters of the op stream; the runner declares its options (seed,
   `--posix`/`--cache`/`--s3`, `--time-scale`, `--threads`, …) and classifies each as an
   *identity* option, which a use case or rules file may lock (§3.60's fixed list is the
   start), or a *solution* option, which is always the submitter's. A fragment sets `params`
   (the abstract's names) and `run` (the runner's identity options).

4. *Three states, narrowing only* (agreed). Every value is **locked** (final), **default** (a
   value a later layer may replace), or **required** (no value; a later layer must supply one).
   The abstract gives each parameter its first state; each later layer may only narrow it
   (required to default or locked, default to locked) and never reopens what an earlier one
   locked. A default is for a real fact of the layer that sets it (NumPy's chunk); a value that
   varies with every use case is required, never a placeholder, so a B300 use case that forgets
   `batch` is refused with the names missing instead of running the H100 job. The user weighed
   defaults against requiring every unlocked value and kept defaults for this reason. *Lock
   where the reason for locking lives* (agreed): the abstract records the traced API as a
   default; `unet3d-h100` leaves it a default, so a developer compares `--posix io_uring` (and
   the report already says the run used another backend); the benchmark's rules lock it,
   because comparability is the benchmark's reason.

5. *Composition by agreement, one level* (agreed). A fragment is a parameter file. Layers are
   ordered (abstract, use case, rules, submission, command line), and a later layer may replace
   an earlier one's default. Within one layer the fragments are order-free: two that set the
   same name to the same value agree, two that set different values are refused, the refusal
   naming both files (CUE's and the Nix module system's rule, not "last one wins"). A use case
   lists its fragments and a fragment imports nothing: the reuse is at the job level (one job
   fragment, several GPU fragments) and in the shared rules file, not deep enough to pay for an
   import graph (the user's conclusion: collapse layers that bring no reuse). Today's
   `--params-file A --params-file B` is the same mechanism with last-wins semantics.

6. *The rules file names no parameter* (agreed; the user's point that names are per abstract
   and a shared dictionary of them is too much complexity). It acts on states as a whole:

   - `freeze`: every default becomes locked, every `scale` parameter required, every identity
     option locked;
   - `constraints`: point 9, over facts, results, and nothing named by an abstract.

   A **`scale`** role tag on an abstract parameter (`w.param("samples", …, scale=True)`) marks
   what is sized per site; it makes formal the `[config]` convention the docs already use, and
   it is the only new per-parameter attribute. A rule that must name a parameter belongs in
   that use case, never in the shared file.

7. *Equations between parameters* (agreed). Already valid: a parameter's value may name others
   (V1, cycles refused on the defaults and again on the values given). Composition by agreement
   leaves one consistent meaning: equations are evaluated once, after every layer has merged,
   so `step_time = 46 ms × batch` follows whatever `batch` ends up being. A derived parameter
   is locked by construction (overriding it while its inputs still drive it would contradict
   them). One that reads a required parameter varies per submission: the id covers its
   expression, the report its number.

8. *Known-at times, and two runner vocabularies* (agreed). Every name is known at **resolve**
   (parameters: before any host is looked at), the **gate** (facts, point 11's
   storage-reported values: on the hosts, before any I/O), or **after the run** (results). A
   value may read only names known no later than itself: a parameter never reads a fact, so a
   run's identity never depends on what its hardware reported (`MemTotal` moves by megabytes
   across kernels and boots). Facts and results are declared by the runner, born locked, and
   named for their aggregation and window (`fact.min_samples_per_gpu`, `fact.sum_host_mem`,
   `result.min_gpu_busy_fraction` over the steady state the runner already selects), each
   defined once in `runner/REFERENCE.md`; changing a definition is a recorded decision. The
   `fact.` and `result.` prefixes (proposed 2026-10-21, decided 2026-10-22) keep them from colliding with a parameter now or
   when the vocabulary grows (the user asked that no parameter take a fact's or result's name):
   a dot is refused in parameter names, so the check holds by construction and `gpus` (V2)
   stays the only reserved name.

9. *Constraints and verdicts* (agreed). A constraint is an expression in the abstract's
   expression language with a `doc`:

   ```json
   {"rules_version": 1, "name": "…",
    "freeze": {"defaults": "locked", "scale": "required", "identity_options": "locked"},
    "constraints": [
      {"expr": "fact.dataset_bytes >= 5 * fact.sum_host_mem", "doc": "dataset at least 5x the clients' total DRAM"},
      {"expr": "fact.min_samples_per_gpu >= 2000",           "doc": "minimum distinct samples read per GPU"},
      {"expr": "result.min_gpu_busy_fraction >= 0.90",       "doc": "every GPU busy at least 90% of steady state"}]}
   ```

   It is checked at the latest known-at of the names it reads, with no `when` field (the user
   agreed it is not needed): at resolve (`aeiou dry-run`, before datagen), at the gate, or after
   the run. Every name must resolve at resolve time, so a misspelled result fails before a
   two-hour run, and the header lists each constraint with its time. The test and the message
   are the runner's, once, for every constraint (the constraint, its `doc`, its file, the
   numbers), so a rules file stays data, never code. A failed resolve or gate constraint is an
   invalid run, refused before any I/O. A failed result constraint is a valid run that did not
   pass: the run completes, the report is written with each constraint passed or failed, and the
   exit status is its own (agreed; 3, decided 2026-10-22: unused today beside 0, 1, 2, 130,
   141, 143, and below the 126 and up the shell reserves), so a
   driver searching for the largest passing GPU count tells "did not pass" from an error. Rules
   across runs (repeats, medians, the search itself) are the driver's (`mlpstorage`, §3.74);
   a rules file judges one run.

10. *`samples` is checked, never derived* (agreed). `--subset` (§3.70 item 6) makes the
    samples a run reads a run-time choice up to the corpus's count, so it is not fixed at
    datagen. Its floor is per GPU (the user's point): work is split by global GPU id with
    `drop_last`, so each GPU's distinct samples per epoch is `floor(samples / G)`, whatever the
    GPUs per host, and the GPUs per host are the submitter's choice (simulated GPUs have no
    hardware's packing). That layout leaves the op multiset unchanged but not how the ops
    arrive: fewer, fuller clients mean fewer mounts and connections, more GPUs sharing a page
    cache across epochs, more streams merged by one client's readahead, and a client that may
    become the bottleneck (the per-step stall and busy fraction show it). The 5× rule guards the
    cache effect, so the count's floor is per GPU and the cache's per summed host memory. The
    submitter supplies `samples`; a dry-run helper prints the smallest value meeting every
    constraint for a planned cluster; the runner checks it at the gate. Deriving it from
    measured memory would let the run's values and its subset move with `MemTotal`.

    *This revises §3.74 item 2* ("reported, never judged: the 5× check is `mlpstorage`'s";
    agreed). The user's reason: aeiou is the only part that knows every client host's memory
    (each rank's `MemTotal` and cgroup `memory.max` reach rank 0 in its Hello, §3.74), so
    `mlpstorage` need not collect them a second time, and aeiou enforces the rule when the
    WG's file carries it. The equation is the WG's, aeiou hard-codes none, and `mlpstorage`
    keeps the sizing step, writing the submission.

11. *Values the storage reports* (agreed that `st_blksize` is the system under test's; the
    mechanism proposed 2026-10-21 and decided, as written, 2026-10-22). Python's `BufferedReader` asks for `st_blksize`: 1 MiB on NFS at rsize
    1 MiB, 4 KiB on a local disk. By the interposition test the file system's report is the
    solution's, and pinning it would charge a file system that reports 4 MiB for reads the real
    application would not issue against it. Decided: a value source `{"from": "st_blksize"}`
    for `hdr_read`, `xfer`, `inv_read`, read by `fstat` on the dataset's root at the gate,
    printed, and reported; the id covers the rule, the report the number. This refines CLOSED
    (§3.48): every system under test sees the same op sequence *as a function of what it
    reports*, and two fingerprints differ only through reported values, which the report shows.
    An object store reports none, so the source declares a fallback. A rules file may lock the
    value anyway if the WG prefers strict sameness to fidelity.

12. *The lock, the id, and the handle* (agreed). Resolving a use case (and a rules file, for a
    benchmark entry) writes a **lock**, a JSON file (JSON or TOML as everywhere in the repo,
    never YAML, whose implicit typing makes a canonical form fragile; agreed): every value with
    its state and the file it came from, the freeze and the constraints copied in, the
    abstract's path relative to aeiou's `workloads/` directory and its `ast_sha256` (the hash of
    the AST's canonical form, `schema/README.md` §1, never of the authoring script), and the
    fragments' and rules file's paths and hashes as provenance. It is generated (by
    `aeiou-params lock CASE [--rules FILE]`, proposed 2026-10-21 and decided 2026-10-22 with
    when it runs, point 17) and never edited, and it is the file a
    run names: at run time it depends on the abstract alone, and the runner refuses an
    abstract whose hash is not the lock's. A lock may hold several runs (a writer and its
    reader), each pinning its own abstract; `aeiou datagen` takes the same lock, so the corpus
    is written from the values the run checks against. Two hashes answer two questions:

    | Hash | Of | Answers | Changes when |
    |---|---|---|---|
    | **file hash** (SHA-256 of the bytes) | the lock | is this the exact file published? | any edit, a `doc` typo included |
    | **workload id** | the lock's canonical resolved values, without the required values and the storage-reported numbers the report carries | does this run do the same thing? | a value that changes the workload |

    The **name** (`npy-per-sample/unet3d-h100`) is a mutable pointer, like a git tag; the id is
    what results are compared by, so a corrected `doc` does not split them. A changed abstract
    changes its hash and so every id built on it; re-resolving rewrites every lock, and the
    lock diffs are what a reviewer reads. Two compositions that resolve alike are the same
    workload; the fragment graph is provenance, recorded, never compared. Every run prints
    every resolved value with its state and source in its header and report (the user's point:
    necessary, though not sufficient without a check before the gate); `--expect-case-sha` and
    `--expect-workload-id` refuse a mismatch before the gate, as `--expect-dataset-id` does.
    The raw `AST --params-file` path stays for authors and researchers; its runs report no case.

13. *Finding the lock* (agreed). Two paths, resolved by two parties:

    | Path | Known to | Resolved by |
    |---|---|---|
    | the lock's own location | the caller: `mlpstorage` its install image, a developer the case's name | the command line |
    | the abstract's, inside the lock (relative to `workloads/`) | aeiou, which knows where it is installed | aeiou, from the lock |

    Only the lock's author cannot know where aeiou will be installed; aeiou always knows (its
    own executable's path; `workloads/` beside `bin/` in the release, `--workloads DIR` for
    another layout such as a source checkout). So nobody types or builds the abstract's path:
    aeiou reads every value, the path included, from the lock itself, and a lock is
    byte-identical on every site, which its file hash needs. Two mutually exclusive options,
    each named for what it does:

    - `--case-file PATH`: the lock at that path, no search. `mlpstorage` finds its entry in its
      own install image (its business, by its own index or walk) and passes the absolute path;
      a developer's lock outside aeiou's tree is run the same way.
    - `--case NAME`: for a developer running without `mlpstorage`; aeiou walks its own
      `workloads/` (hundreds of files, milliseconds) for the lock of that name, refusing none
      or several and listing every match; `mechanism/name` narrows a short name.
      `aeiou-catalog check` refuses two committed locks of one name, so every shipped name is
      unique. A walk, not a generated index: one less generated file to keep current.

    WG entries (`mlperf-unet3d-h100.lock.json`; their names are the WG's) live in
    `mlpstorage`'s install image, not in aeiou's tree, which keeps WG content out of the repo
    (`PROJECT_BRIEF.md` §8); a WG lock pins an aeiou abstract by its relative path and hash.

14. *Validation after upload* (agreed). The submitter uploads the report to the WG's store,
    and the WG runs a clean checkout of `mlpstorage`, which holds the published file hash and
    workload id of every entry, against it. The report is the submitter's file, so nothing
    printed in it proves anything by itself; the checker recomputes:

    1. the file hash, from the lock the report **embeds** (the file itself, not only its hash);
    2. the id, from that lock and the submission's required values;
    3. the op counts by kind, bytes read and written, and the fingerprint, by `aeiou dry-run`
       on that lock with the submission's GPU count and storage-reported values, in a pristine
       image of **the aeiou version the report names**. The fingerprint and dataset ids
       depend on the runner's definitions, so the WG keeps every release it accepts (agreed).
       No storage is needed, and no dry-run before the run: at 150–350 ns per op, a job of
       10⁹ ops is minutes.

    What hashes cannot prove is that the results came from that run. What makes forging hard
    is numbers that must agree exactly: the report's totals, histogram counts, and per-phase
    sums against the dry-run's; the kernel's NFS counters (mountstats, §3.30) against the op
    stream; `started`, `finished`, and the step times (point 15) against `steps × step_time`
    and the busy fraction; and the upload store's object times, the one clock the submitter
    does not keep. Reproduction (same lock, same id) remains the WG's.

15. *Step times* (agreed; the user's request). Beside today's per-step stall and busy fraction,
    the report holds two histograms over every GPU's every step (50,000 entries each at 100
    GPUs × 500 steps), in the report's latency buckets: the **compute** time of each step (the
    emulated compute, so its spread is the timer's overshoot and shows an overloaded client)
    and the **delay** before the next step could start (the wait for data, and the barrier
    where the abstract has one: the storage's signal). Minimum, median, and maximum alone are
    too coarse to see oddities; a table per GPU and step is more than a reviewer needs.
    With them, the slowest N delays (N = 10; agreed) with each one's GPU and step index, so a
    tail can be traced to where it happened. Each is a fixed-size accumulator (a few hundred
    counters, or N entries), kept by each GPU's own thread with two clock reads and one
    increment per step, merged per host and on rank 0 bucket by bucket at the end
    (`Report::merge_all`, as the latencies), so it keeps point 16: no per-entry storage, no
    file, no network during the measured run. The user's caution: a naive implementation that
    wrote each entry as it happened would break point 16.

16. *No I/O of aeiou's own during the measured run* (agreed; a rule). Nothing aeiou writes for
    its own records touches any file system between the gate and the end of the run: the
    report, the step times, and the counters are held in memory and written after the end.
    The user's reason: submitters have kept log files on the system under test, and the
    logging perturbed the measurement. The build confirms every path of `aeiou run` keeps it
    and adds a test. Outside aeiou's reach, and for `mlpstorage`'s documentation: a shell
    redirect of the output, and the `--report-json` file, belong off the system under test.

17. *When a lock is made, and by what* (decided 2026-10-22). The user asked why locking is not
    an unavoidable step at the end of `aeiou-build`, since it fixes every value of a use case
    and should not be optional. The goal is kept, and the trigger is not the build alone,
    because a lock's inputs arrive at different times from different owners: the build sees
    one authoring script, while a lock also needs a use case (written after its abstract,
    since it sets the abstract's parameters; several per abstract) and, for a benchmark
    entry, a rules file (the WG's, per round, outside aeiou's tree, where no build can find
    it). The submission is not in the way: a lock is byte-identical on every site and the
    required values arrive at run time (point 12). Locking is made unavoidable by what
    refuses a run or a commit without one:

    | When | What makes or remakes the lock |
    |---|---|
    | a new use case is written | its author runs `aeiou-params lock CASE`; CI refuses a committed case without a lock |
    | the abstract's script is edited and rebuilt | `aeiou-build` re-locks every committed case of that abstract that has a lock, as its last step |
    | a use case's fragments are edited | its author re-locks; CI refuses a stale lock |
    | the WG publishes a round's rules | the WG runs `aeiou-params lock CASE --rules FILE` |
    | a run | the runner refuses an abstract whose hash is not its lock's (point 12) |

    On an abstract's first build there is nothing to re-lock; the build's step matters on a
    rebuild, when the abstract's hash changes and every lock pinned to the old one goes stale.
    `aeiou-catalog check` refuses a committed case with no lock, and one whose `ast_sha256` or
    fragment hashes do not match the files on disk. `aeiou run --case`/`--case-file` reads
    only a lock, so a use case never locked has nothing a run can name; the raw
    `AST --params-file` path stays and reports no case (point 12).

    The lock is Python, in the builder, because the build calls it: merging fragments by state
    and agreement and hashing them is what `aeiou-params` already half does, and a builder
    that called the runner binary would depend on it. The runner still owns the names of
    facts and results: `aeiou dry-run` on a lock resolves every constraint's names (point 9),
    before datagen, so a misspelled one still fails before a long run. A lock carries a
    `lock_version` and the runner refuses one it does not know.

    *Where an abstract lives.* The lock names its abstract by its path relative to aeiou's
    `workloads/` (point 13), so a use case lock exists only for an abstract in that tree; a
    workload kept outside it runs by the raw path and reports no case, which fits use cases
    being shared and comparable. Because that path is in every lock built on it, the
    mechanism's directory name is effectively permanent once a case is locked (renaming it
    re-locks every case). The user's point: an abstract fitted from a trace is placed by
    Claude, not by a person who knows the vocabulary. The README's path C gains a step: Claude
    states the category it chose (one of the seventeen in `workloads/attributes.toml`, each
    with its directory already) and the mechanism's name (named for the I/O mechanism, never
    the use case, §3.68), with its reasons, and the person confirms before it builds. The
    person needs to agree with the reasoning, not to know the vocabulary. `aeiou-build`
    itself creates no directory but an `-o` that does not exist.

18. *Accelerator names* (decided 2026-10-22). A use case directory is named for its job and
    its accelerator (`unet3d-h100`, `unet3d-b300`); the accelerator names are a vocabulary,
    an `accelerator` attribute in `workloads/attributes.toml`, each case's record naming one, and
    `aeiou-catalog check` refuses a name not in it, as it does every other attribute, so
    `h100` and `H100-80G` cannot both appear. The directory name stays a label; code reads
    the record.

**What changes, if decided.** *(Decided 2026-10-22; ~~not yet built~~ built 2026-10-22, the parts below, but for the abstracts outside the input pipelines, `PROJECT_BRIEF.md` §6 item 38.)*

- Contract 0.11: the `scale` tag, the dot refused in parameter names, `{"from": …}` values;
  `params_version` 2 with states and a `run` table; a rules schema; both checkers.
- The abstracts: job and GPU values that are defaults today become required (`npy_per_sample`'s
  H100 set moves to a `unet3d-h100` use case), `[config]` becomes `scale`, `hdr_read`/`xfer`/
  `inv_read` and their counterparts in other abstracts (`buf`) become storage-reported. Every AST hash, and the
  fingerprints of runs on storage that does not report 1 MiB, change.
- The runner: the lock and the id; `--case-file`, `--case`, `--workloads`,
  `--expect-case-sha`, `--expect-workload-id`, in `run` and `datagen`; facts and results as
  names; constraint evaluation at three times; the new exit status; the header block; the lock
  embedded in the report; the step-time histograms; the no-I/O test; a lock's names resolved by
  `dry-run`; `lock_version`. The builder: `scale`, the name rule, `aeiou-params lock`, and
  `aeiou-build` re-locking the committed cases of what it built (point 17). `aeiou-catalog
  check`: one lock per name, no case without a lock, no stale lock, the `accelerator`
  attribute (point 18). Every man page whose flags change, and the exit status 3 in each
  page that lists exit statuses.
- The docs: `PROJECT_BRIEF.md` §5 and §8 (a rules file is WG process: aeiou ships the mechanism
  and no WG rules file), §6 a new item, CLAUDE.md's invariants (fingerprint and id), §3.74;
  the README's path C, the confirmation of the category and mechanism (point 17). *2026-10-22:*
  the docs part done that day (the brief's §5 row and item 33, §8's rows, CLAUDE.md, the
  README); the rest is item 33.

**Decided at the start of the build (2026-10-22, the user's answers to four questions the
points above leave open).**

- *A required parameter declares its kind and has no default* (`{"kind": "int"}`; `items` for
  an array's elements), never a default kept as an example, which is the placeholder point 4
  rules out. The key is `kind`, not `type`: the user asked whether `type` was better and kept
  `kind` on the answer that the contract already calls scalar, distribution, and array a
  value's *kind* (`schema/README.md` §8, the runner's messages), that `type` beside JSON
  Schema's own `type` keyword in the schema file misreads, and that `type=` in the builder
  shadows a Python builtin.
- *Every entry of a `params_version` 2 fragment and of a lock names its state*:
  `{"default": v}`, `{"locked": v}`, or `{"required": true}`, never a bare value, which
  beside a distribution's object would leave the parser to tell a wrapper from a value by its
  keys.
- *A rules file's constraint is infix text* (`fact.dataset_bytes >= 5 * fact.sum_host_mem`),
  parsed by `aeiou-params lock` into the AST's expression form, which the lock holds beside
  the text; the runner evaluates the JSON alone, so there is one parser, in Python.
- *A lock of several runs is run one at a time with `--case-run NAME`* (the run's abstract),
  required when the lock has more than one run and refused when it has one; one lock per run
  would revise point 12.

**Built: contract 0.11** (2026-10-22). The schema, both checkers, the builder
(`w.param(name, kind=int)`, `scale=True`, `st_blksize(of=…, fallback=…)`), and the runner
(`Params::new` refuses the required names with no value, a value is held to the declared
kind, `aeiou run` reads `st_blksize` before the model is built, `aeiou dry-run
--st-blksize [DATASET=]BYTES`, the `storage-reported:` header line, and `storage_reported`
in the report). Every committed AST regenerated for the version, no fingerprint and no
dataset id changed; no abstract declares a required, `scale`, or storage-reported parameter
yet (the abstracts move with the lock, below). Choices made in the build, open to revision:

- A storage-reported value names its dataset (`"of"`), since an abstract may read several
  roots and a parameter is tied to none; its root is the dataset's, wherever its endpoint
  puts it. It is read from the dataset's **first file**, not the root directory: Linux's NFS
  client reports a directory's `st_blksize` as its READDIR size, not the file's (from the
  kernel's `nfs_getattr`, which sets a directory's to `dtsize`; recalled from the source, not
  measured here). An output namespace has no file at the gate, so `of` names a dataset only:
  the write-side counterparts (`buf`) stay values for now.
- A fragment cannot hold a `{"from": …}` value; a use case that wants a fixed number gives the
  number, which replaces the default as any value does. The source lives in the abstract.
- The kind rule keeps its coarse form for every parameter (scalar, distribution, array, and an
  integer's number rule): a required `float`, `bool`, or `str` takes any scalar, as a default
  of those kinds always has; V27 judges the finer kind where the document uses it.
- `aeiou dry-run --st-blksize BYTES` applies to every dataset a storage-reported parameter
  reads, `DATASET=BYTES` to one; a name no such parameter reads is a usage error.
- V28 is the rule's number; check.py and the runner give the same messages, held by
  `runner/aeiou/tests/refusals.json`.

**Built: `params_version` 2 and the lock** (2026-10-22). The parameter-file schema and both
readers (the runner's `ParamSet`, `aeiou-params`), every committed set converted in place
(each value a `default`, the hand formatting kept) and the kits that write sets with them;
`aeiou-params lock` (`builder/aeiou/lock.py`), `schema/rules.schema.json` and
`schema/lock.schema.json`, a lock for every committed case, and `aeiou-build` re-locking the
cases of what it wrote. Choices made in the build, open to revision:

- **The workload id is the runner's alone.** It must cover every identity option in effect,
  and only the runner knows their defaults; a Python copy would be a second definition to keep
  equal. The lock holds no id; `aeiou dry-run` on a lock prints it.
- **A freeze of the identity options is carried as the freeze**, not as one entry per option:
  the lock's `run` table holds what a fragment set, and the runner holds every other identity
  option at its default when the lock's `freeze` locks them.
- **The identity options a `run` table may set** are `seed`, `posix`, `cache`, `s3`,
  `time_scale`, `rank_rotate`, and `write_compress` (§3.60's fixed list and what the
  coordinator's identity hash covers); `aeiou run --params-file` refuses a file with one.
- **A lock's entries are the fragments' form** (`{"locked": 7}`), each run's sources in a table
  beside them (`sources`, `run_sources`); one value to a line when rendered.
- **Paths in a lock are relative to `workloads/`**, a file outside it (a WG rules file) by its
  name, and the lock records the case's `case.toml` by path alone (it holds the catalog's
  attributes too, so its hash would make an attribute edit stale every lock) and no tool
  version (a release would make every lock stale).
- **Agreement is on the whole entry**: `{"default": 7}` and `{"locked": 7}` from two fragments
  of one run are refused like two values, so no rule has to decide which state wins.
- **A later layer may replace a default and may require a value; only a lock is final.** A
  `scale` parameter a use case locked is refused by a rules file that makes `scale` required.

**Built: the runner's side of the lock** (2026-10-22; `runner/REFERENCE.md` §17,
`runner/aeiou/src/case.rs`). `--case`, `--case-file`, `--case-run`, `--workloads`,
`--expect-case-sha`, `--expect-workload-id`; the lock's run as the bottom parameter layer and
its identity options set and enforced; the workload id; the header's case block and the
report's `case` (the lock embedded) and `constraints`; the facts and results and constraints
judged at resolve, at the gate (rank 0, before its `ready`), and after the run; exit status
3, kept by `aeiou-launch`. Choices made in the build, open to revision:

- **`dry-run` takes a lock too**, not only `run` and `datagen` (item 33 named those two): the
  WG's validation (point 14) is a dry run of the lock, and the dry run is where an author
  reads the workload id and finds a misspelled constraint name before a long run.
- **The first vocabulary** is ten names (`fact.gpus`, `fact.dataset_bytes`,
  `fact.min_samples_per_gpu`, `fact.hosts`, `fact.sum_host_mem`, `fact.min_host_mem`,
  `fact.max_gpus_per_host`, `result.elapsed_s`, `result.busy_fraction`,
  `result.min_gpu_busy_fraction`), each from what the runner already computes. Point 8 spoke
  of "the steady state the runner already selects"; the runner selects none, so the busy
  fractions leave out each instance's first take, which waits on the pipeline filling, and a
  steadier window is a later, recorded change of the definition.
- **A host's memory is its `MemTotal` or its cgroup `memory.max`, whichever is less**, so a
  container's limit counts as the memory a cache could use.
- **The workload id covers the lock's other runs by their entries** (a reader's id moves when
  its writer's values do), and never `--gpus`, the submission's scale.
- **`datagen` judges no constraint and refuses `--expect-workload-id`**: its `--gpus` is not
  the run's, so a per-GPU fact would be judged on the wrong count; the run judges them all.
- **A run that did not pass exits 3 on rank 0 alone**; the other hosts' verdict is the run's
  (valid), so they exit 0, and `aeiou-launch` exits 3 when the only non-zero status is 3.
- **`--seed` has no default in the parser** (0 is applied after it), so a lock's seed is the
  run's where the command line gives none and a `--seed 0` given against a lock's 7 is seen.
- **Found in the build**: rank 0's own failure before the gate (a gate constraint) reached the
  other hosts only as a `Stop` its client sent its own server, which the process could exit
  before broadcasting, so a host heard "connection closed" in place of the reason under load;
  rank 0 now stops through its server, under the state lock, before it returns.

**Built: step times and the no-I/O rule** (2026-10-22, `runner/REFERENCE.md` §17). Choices
made in the build, open to revision:

- **A step is what follows a `take`**, to the next one, so steps exist for the instances that
  take (a training GPU); the compute is the measured sleep of its `compute`s, the delay runs
  from the end of the last one to the next take's return, and an instance's last step has no
  delay (its counts are the steps less one per instance).
- **The accumulator lives in the statistics** (`Stats::steps`) and merges with them, so it
  reaches rank 0 with the report each host already sends; the ten slowest are kept sorted,
  ties by template, instance, and step, so a merge is order-free.
- **The no-I/O test runs under `strace`** in the Python suite (CI's integration job has it),
  on the `sync` backend, whose ops are calls strace sees; the event loops' reporting is the
  same code. Reads of `/proc`, `/sys`, and `/dev` are allowed during the run (the counters'
  sampler reads `/proc`): they are no storage. The build confirmed no path of `aeiou run`
  wrote during a run before the test was written; the test keeps it so.

**Built: the catalog through the lock, and the first use case** (2026-10-22, part 5a). Choices
made in the build, open to revision:

- **A case's shape is measured through its lock** (`dry-run --case-file`), as a run of the case
  reads it, and a capture pair with no parameter file of its own is judged at the lock; a
  `[[run]]`'s `params` is one fragment or a list, each the case's or its mechanism's.
- **`npy_per_sample`'s job and GPU values are required**, `samples` is `scale` and required,
  and `hdr_read`, `xfer`, and `inv_read` are `st_blksize` (fallback 1 MiB). The values the
  defaults held are the use case `unet3d-h100` (`job.params.json`, `h100.params.json`), whose
  lock runs the fingerprints the defaults did (`tests/golden.rs` and `tests/cases.rs` hold
  them). The two traced examples carry the old `step_time`, 323 ms, said to be not fitted (the
  user's answer). ~~Both fragments sit in the case's directory until a second accelerator shares
  the job fragment; where a shared fragment lives is decided then.~~ *2026-10-23, §3.81: the
  job fragment is the mechanism's (`jobs/unet3d.params.json`), shared by `unet3d-a100`,
  `unet3d-h100`, and `unet3d-b200`, and holds the WG's sample sizes.*
- **Its shape is the first at a real job's size**: about 490 GB read by one instance, which
  overflowed the metrics pass at 4 KiB blocks and led to the extent metrics (§3.79), built
  before this part was committed; it is measured exactly, in about two seconds a seed.
- **Point 18 read as: a use case names its accelerator; an example may.** The examples were
  traced on stand-ins with no accelerator, so `accelerator` is `required = { kind = "use" }`
  in `attributes.toml`, a conditional requirement any curated attribute may now declare
  (`aeiou-catalog` checks that the condition names a curated attribute of the same level and
  one of its values). The vocabulary starts with `h100` alone; each accelerator is added with
  its first case.

**Built: the untraced containers' cases** (2026-10-22, part 5b). Choices made in the build, open
to revision:

- **Untraced, by the user's choice.** `hdf5_map`, `parquet_stream`, and `tfrecord_stream` had
  no trace and no case. Every example case has a capture, which these cannot have without
  kits, and two of the three applications (PyTorch with h5py, TensorFlow) are not installed;
  the user chose to finish item 33 first. Each abstract's job and GPU values are required
  (`batch`, `step_time`, `steps`, `prefetch`, `samples_per_file`; `workers` in `hdf5_map`;
  `cycle`, `decode`, `sample_median` in the streams) and `samples` is `scale`; the values
  their defaults held, placeholders never measured, are one case each of kind `unsettled`
  (`rgb256-rows`, `imagenet-rows`, `imagenet-records`), with a lock and no capture, whose
  locks run the fingerprints the defaults did (`tests/cases.rs`). The mechanisms' `[defaults]`
  rows are gone; the catalog keeps `[defaults]` for a mechanism added before its first case.
  Their traces are `PROJECT_BRIEF.md` §6 item 36, and item 35 is designed on them.
- **`decode` is required.** It was 0 by default, "I/O bound", a placeholder marked
  `[measure]`; point 4 makes a value that varies with every use case required. The cases give 0.
- **A format class takes the mean sample size only when its framing needs it.** Parquet's
  page headers are fixed at build time by the varint band of a row group's chunk, which was
  read from `sample_median`'s default. A size naming a required parameter has no mean, so
  `parquet(band_sample_bytes=…)` places the band (`parquet_stream` gives 110 KiB: the same
  4-byte band, an identical framing), and the class refuses without it, naming the argument;
  TFRecord, WebDataset, and HDF5 frame without a mean.
- **The release tarball carries the locks.** Its file filter left out `case.lock.json`, so a
  released tree could not run `--case` (unet3d-h100's included); found writing the HDF5
  smoke step through a lock, held by `test_lock.py` since.

**Built: `file_per_sample`'s case** (2026-10-22, part 5c). Choices made in the build, open to
revision:

- **An unsettled case, not a use case** (the user's choice). The defaults (batch 32, 8 workers,
  2 ahead, 500 steps of 105 ms, 50,000,000 files of a median 110 KiB) were never measured as a
  job and are not the brief's reference job, unlike `npy_per_sample`'s, so they are the case
  `jpeg-50m` of kind `unsettled`, whose lock runs the fingerprint the defaults did
  (`tests/cases.rs`). The traced example `imagefolder-jpeg` gives every required value:
  `step_time` the old 105 ms, said to be not fitted, and the median its kit's corpus has.
- **The median is required, the sigma is not** (the user's choice). A lognormal's `sigma` is a
  number in the contract, so it stays 0.45 in the abstract, as in the streams; making it an
  expression is a contract change, `PROJECT_BRIEF.md` §6 item 37.
- **`hdr_read` and `inv_read` are `st_blksize`** (fallback 1 MiB), as in `npy_per_sample`.
  The run's `stat` of the first file, before the gate, is now in every run of the mechanism;
  `test_no_io.py` takes the run's window from the actors' threads, so the gate's own `stat`
  is outside it, and the runner-against-strace test of `test_trace.py` tells the dry run
  the `st_blksize` the run reads.
- **A run whose storage cannot report says which run it was.** `aeiou run` read `st_blksize`
  before it recorded the run's identity and printed its options, so a run on a root with no
  dataset failed with a report naming no abstract and no warnings for options passed over
  (`npy_per_sample` since part 5a; no test ran one there until this part). The identity and
  the option block now come first.
- **The tests keep their values.** Each test that relied on a default is given it: the runner's
  tests through one helper (`tests/common`, `FILE_PER_SAMPLE`), the Python tests where they
  call, every fingerprint unchanged.

**Built: the docs, and where item 33 stops** (2026-10-22, part 6). The README's status,
CLAUDE.md (contract 0.11, `runner/REFERENCE.md` §17, a case's lock and fragments, the
workload id built), and `PROJECT_BRIEF.md` §6 item 33 marked built. Choices made, open to
revision:

- **The five input-pipeline abstracts are the migration; the eight others are item 38** (the
  user's choice). `dcp_save`, `dcp_load`, the two LMCache abstracts, `safetensors_load`, the
  two DiskANN abstracts, and `faiss_ivf_ondisk` keep their job values as defaults. Their
  values are of other kinds than a training job's batch and step time (a model's size and
  ranks, a session's prompts, an index and its queries), and which are required, which is the
  scale, and which reads follow `st_blksize` is decided with the user per mechanism, not
  assumed from the training five.
- **A use case per accelerator is item 39.** Point 1 makes the batch and the step time the
  model's on one accelerator, so `unet3d-h100` is the first of a family (`unet3d-b300`, …),
  each a shared job fragment and an accelerator fragment; until then the traced examples carry
  the old `step_time`, said to be not fitted.

**Built: contract 0.12 and `dcp_save`'s case** (2026-10-23, item 38's first mechanism;
`PROJECT_BRIEF.md` §6 item 38). Decided with the user the same day, on four questions:

- *A write buffer is the storage's too* (the user's choice of the namespace form over a
  library default kept at 1 MiB, or a `stat` of the root directory). `dcp_save`'s `buf` is the
  `BufferedWriter` of the shard file, sized by its `st_blksize` like a reader's, and decides
  which items are written in one call and which coalesce with their framing; a file system
  reporting 4 KiB, 1 MiB (NFS at wsize 1 MiB), or a Lustre stripe of 4 MiB sees another
  split. Point 11 named a dataset, and `dcp_save` has none, so contract 0.12 lets `of` name a
  namespace. An output namespace is empty at the start, so the runner creates a probe file
  under its root, `stat`s it, and removes it before the gate: a file created there inherits
  what the probe did (a directory's own number is the READDIR size on NFS, the reason 0.11
  reads a file). An input namespace is read-only (V14), so it is read from the first object
  its manifest lists, or from the manifest where it lists none (past the listing's limit).
- *The old defaults are an unsettled case* (the user's choice): `four-items`, whose lock runs
  the fingerprint the defaults ran (`tests/cases.rs`; golden.rs's configuration).
- *`meta_bytes` is required* (the user's choice over an equation of 217 bytes per item and
  rank): the `.metadata` pickle grows with the parameter names and the plan, and two traces of
  the same six items measured 2,602 and 2,606 bytes, so a case measures it.
- *`item_bytes` stays each rank's bytes* (the user's choice over tensor shapes divided by
  `gpus`): FSDP2 shards dim 0 in chunks of ceil(n/G), so later ranks get less or nothing and
  an even split would be wrong; a case holds at the GPU counts it declares.

Choices made in the build, open to revision:

- **Required:** `steps`, `step_time`, `ckpt_every`, `items`, `item_bytes`, `meta_bytes`; the
  library's `hdr` and `trailer` and the scored write's `readback` (off) stay defaults; `buf`
  and the readback's `xfer` are `st_blksize` of `ckpt`, fallback 1 MiB. No `scale` parameter:
  there is no dataset, and the checkpoint's size is the model's, so the submission's scale is
  `gpus` alone.
- **`toy-6-items` carries the old `step_time`, 323 ms, said to be not fitted**: the kit saves
  back to back and computes nothing between checkpoints.
- **Probes are named apart and passed over.** Every host probes the shared root before the
  gate, with no order between them and rank 0's emptying of the root, so a probe is
  `.aeiou-blksize-HOST-PID`, created exclusively, and the empty-root check skips that prefix
  and leaves such a file alone, even under `--clean-namespaces` (it is another host's, there
  for an instant). The coordinator test runs two ranks probing one root.
- **A name both a dataset and a namespace is refused** by both checkers (V28); the builder
  already refuses the second declaration of a name.
- **The tests keep their values**: the runner's through one helper (`tests/common`,
  `DCP_SAVE`), every fingerprint unchanged; a run on a file system reporting 4 KiB splits a
  1 MiB item that a dry run's fallback coalesces, so the coordinator test gives its dry run
  the number the root reports.

**Built: `dcp_load`'s case** (2026-10-23, item 38's second mechanism). Decided with the user
the same day, on the recommendation: the model's and the job's values are required, `items`,
`item_bytes`, `item_off`, `read_order`, `meta_bytes`, with `restore_step` (a step the writing
run saved) and `replicas` (the job's parallelism, 1 when fully sharded); the library's `hdr`,
`trailer`, `rec2`, `eocd_scan`, and `tail_back` stay defaults; and *two buffers* (the user's
choice over one): the shard's `buf` is `st_blksize` of `ckpt`, and `pickle.load`'s reader of
the `.metadata`, which read through `buf` until now, is `meta_buf`, `st_blksize` of
`ckpt_meta`, each fallback 1 MiB, since the application opens two files and sizes each reader by
its own. No contract change: 0.12 reads an input namespace's number from its manifest.

Choices made in the build, open to revision:

- **`item_off` stays a table**, required with the others: it is the prefix sums of `hdr +
  item_bytes + trailer − item_bytes mod 64`, and the contract has no prefix sum to compute it.
- **The old defaults are the unsettled case `four-items`**, the checkpoint `dcp_save`'s case of
  that name writes at step 100, so a run of the one leaves what the other reads; its lock runs
  the fingerprint the defaults ran (22990a88db4e6d4f, 368 ops at 2 GPUs; `tests/cases.rs`). The
  shared name makes `--case four-items` ambiguous, which the runner already refuses with both
  locks named, so the pair is `dcp-save/four-items` and `dcp-load/four-items`.
- **The three traced examples gain `replicas` 1**, and the kit's `params.py` writes it (every
  rank of its save wrote a file of its own); their fingerprints are unchanged.
- **Not checked: that a load case agrees with the save case it reads** (the same `item_bytes`
  and `meta_bytes`); the input namespace's manifest is the guard a run has.
- **`meta_buf` is read where `buf` is.** `ckpt` and `ckpt_meta` share the root `ckpt/` and one
  manifest, and the probe of an input namespace stats the first object the manifest lists,
  which may be a shard rather than the `.metadata`: the same directory, so the same number on
  every file system we know of. Choosing the first object of the namespace's own pattern is
  open.

**Built: contract 0.13 and the LMCache abstracts' cases** (2026-10-23, item 38's third and
fourth mechanisms, `lmcache_local_disk` and `lmcache_shared_fs` with its reader). Decided with
the user the same day, on the recommendation:

- *Required*: the model's on its GPU, `chunk_bytes` (KV bytes per token × `chunk_tokens`),
  `context`, `keep` (the GPU's KV memory against the open conversations), `prefill_per_token`,
  `decode_per_token`, and `prefill_step` (vLLM's `max_num_batched_tokens`, whose own default
  is the GPU's: 8192 or 2048); the deployment's `retain` (the store's capacity); and the load's
  `concurrency`, `warm`, `requests`, `reuse`, `turn_in`, `turn_out`, `sys_prompts`,
  `sys_tokens`, and `sys_pop`. The library's `chunk_tokens`, `meta_bytes`, and `direct`
  (`use_odirect`) stay defaults.
- *The load's features stay defaults, off* (the user's choice over requiring them):
  `think`, `turns`, `trim`, `kind`, `first_in`, `sys_held`, `sub_held`, the per-slot
  switches, the sub-agent prefixes, the prompt-length bounds, and `sys_local`; each default
  turns its feature off, so a chat case does not spell out a sub-agent's prefix it never
  draws.
- *The shared store's `buf` is its `st_blksize`* (contract 0.12): the writer probes its
  output root, the reader stats what the writer's manifest lists; `sharegpt-chat` no longer
  gives it (1 MiB, the traced NFS mount's and the fallback).
- *The old defaults are the unsettled cases `chat-64-slots`*, one per mechanism, the shared
  store's with its writer and reader as two runs; each lock runs the fingerprints the
  defaults ran (golden.rs's: 9f7017e7011639d5, c7d83399ee6948bc, 382dba518a89c85f at 1 GPU,
  2 slots, 50 requests after 50 warm; `tests/cases.rs`).
- *Contract 0.13* (the user's choice of option A over a bound declared with the parameter or
  `reuse` kept a default): `reuse` is drawn as the distance of `turns @ (r − d)`, and rule V3,
  which proved it from the default's minimum, could prove nothing of a required parameter, so
  0.11 refused one in an `at` offset. Now a required parameter passes on the document, in the
  builder, check.py, and the runner alike, and its value is judged when one is given:
  `aeiou-params check` and `lock` on each fragment and the runner's `check_given` on the
  values in effect, as every given value already was (§3.57). A declared bound
  (`kind=Dist, min=1`) would have said more in the abstract, at the cost of schema surface only
  `reuse` uses.

Choices made in the build, open to revision:

- **The traced cases give the values they fell back on**, the old defaults said to be not
  fitted in each doc: `sharegpt-chat` (both mechanisms) and the AgentX fits the per-token
  times, `prefill_step` where absent, and `sys_pop`, and `agentx` `keep` too. The kits write
  no placeholder: `fit.py` and `agentx.py fit` take them by `--set`, which was already where
  the model's and the backend's values went, and the READMEs' commands give them. Every
  fingerprint is unchanged. The AgentX replays record the seconds per prompt and per reply
  token they measured (`latency_s_per_prompt_token` and `_reply_token` in their provenance),
  which a fit of `prefill_per_token` and `decode_per_token` could take: left for a use case.
- **A case's `length` may name a required parameter of kind `int`** (`requests`): the
  catalog took only an integer default.
- **The tests keep their values** through one helper (`tests/common`, `LMCACHE`; the Python
  trace test's `LMCACHE_OLD`).

**Built: `safetensors_load`'s case** (2026-10-23, item 38's fifth mechanism). Decided with the
user the same day, on the recommendation: the model's values are required, its layout
(`shards`, `shard_bytes`, `hdr_len`) and its tensor table (`shard`, `off`, `bytes`, `split`,
`rows`, `row_bytes`), which only `aeiou-params safetensors` makes from a real model's shards,
and so is the deployment's `tp`, which defaulted to 8 unseen. `configs` (the JSON files
transformers reads, the library's) and `config_bytes` (three small files modeled at one size:
the user's choice to keep it a default) stay defaults. No `scale`: the model fixes every byte,
and only the GPU count grows. No `st_blksize`: the application maps the shards and calls no
`read` there, so there is no buffer to size, and Python reads each JSON file in one call. No
contract change: a loop bound taken from a required array's length (`shard.len`) already passed.

Choices made in the build, open to revision:

- **The old defaults are the unsettled case `five-tensors`**, the stand-in of two 5 GiB shards
  at `tp` 8; its lock runs the fingerprint the defaults ran (2668387c7aa4126b, 24,682 ops at 2
  GPUs; `tests/cases.rs`). The example `gpt2-random` and the fixture
  `params/synthetic.params.json` gave every value already: re-locked, unchanged.
- **Not checked: that the GPU count is a multiple of `tp`**, nor that the table's arrays agree
  in length; a table from `aeiou-params safetensors` is consistent by construction, and the
  abstract indexes the others by `shard`'s length, as before.
- **The tests keep their values** through one helper (`tests/common`, `SAFETENSORS`);
  `test_params.py` judges an integer array's default by `dcp_load`'s `tail_back`, since `off`
  is now a required array (judged by its kind).

**Built: `diskann_build`'s values** (2026-10-23, item 38's sixth mechanism). Decided with the
user the same day, on the recommendation: the base's shape (`dim`, `n`), the build's
configuration (`shards`, from its memory budget; `pq_code`; the graph's `degree`, new), the
graphs it makes (`shard_index_bytes`, `index_bytes`, about 132 and 205 bytes a point at
degree 64, which only a real build gives), the warm-up sample's `sample_rows`, and the
compute (`pq_train`, `partition_time`, `build_time`, now integers like `step_time`) are
required. The library's `overlap`, `cache`, `stream`, `chunk`, `big`, and `sector` stay
defaults, all traced. No `scale`: `n` is the index being built, the job, not a dataset a site
sizes to its memory (whether a benchmark lets submitters size it is the WG's). No
`st_blksize`: the reads and writes are libstdc++ streams, whose `filebuf` holds `BUFSIZ`
whatever the file system reports (hence the 8,191-byte reads). No contract change.

- **`sectors` is derived from `degree`**: DiskANN's disk layout puts a header sector, then
  whole nodes of `dim·4 + 4 + 4·degree` bytes to a 4 KiB sector, or a node bigger than a
  sector in whole sectors (`max_(1, …)`, so a dimension of 1,024 and more no longer divides by
  zero). 200,001 at the traced million points, as the trace's file was; 200,000,001 at the old
  billion. From recollection of DiskANN's `create_disk_layout`, confirmed by the traced
  file's size; the multi-sector branch is not traced. `kit/params.py` reads `degree` from
  the disk index's header (`max_node_len`, DiskANN's `save_bin` of u64s) and refuses a file
  whose size disagrees with it, the check on that reading.
- **The old defaults are no case** (the user's choice over an unsettled case at them or at a
  smaller `n`): the billion-point stand-in (40 shards, 5 and 205 GiB graphs, 60 s, 90 s, and
  1,800 s per shard) was never measured, and the catalog could not measure it either: its
  dry run is 436M ops (fingerprint 8e6cb4086b7bdf42, 96 s), and `--metrics` stops after 67 s
  at 6.8 GB with more than 2^26 pieces in one instance. The traced example
  `sift1m-13-shards` is the mechanism's only case; its fingerprint is unchanged
  (17a7cb1242f291bd, 1,293,972 ops; `tests/cases.rs`).
- **The example's compute is not fitted.** No trace metric holds time: PQ training 59 s and
  the partition 91 s as traced, and 20 s per shard, the traced build's 407 s less those two
  over its 13 shards, an upper bound with the merge, the layout, and the I/O in it.
  `kit/params.py --set` gives them, as LMCache's `fit.py` does.
- **The golden configuration moved:** `degree` 64 where `sectors` 1,000 was given, so 200,001
  sectors at its million points and twelve more layout writes (0x93142e1f7cf15fb4, 1,229,219
  ops).

**Built: `diskann_search`'s values** (2026-10-23, item 38's seventh mechanism). Decided with
the user the same day, on the recommendation: the index is required and named as
`diskann_build` names it (`n`, `dim`, `degree`, `pq_code`), in place of `nodes` and
`per_sector`, so that a search case and a build case of one index give the same values; its
sectors are derived by the build's formula (200,001 at SIFT1M, what the cases gave), and the PQ
file is `n·pq_code + 8` bytes, the traced file's size (`nodes·per_sector·pq_code + 8` was 160
bytes more at SIFT1M). The search's configuration (`beam`; `hops`, the index's at its search
list and beam; the hot sets `entry` and `near`, fitted, not derived from a medoid count, since
`near`'s 0.8 is measured), the deployment's `node_cache` (a silent 0 changes the op set: the
user's choice to require it, unlike LMCache's load features off by default), the load
(`threads`, `queries`), and the CPU (`rerank`) are required. `cache_batch`, `sector`, `stream`,
and `pq_read` stay defaults. No `scale`: the index fixes every byte. No `st_blksize`: the
queries are 4 KiB `O_DIRECT` reads, aligned, not buffered, and the two first reads go through a
libstdc++ stream, which ignores it. No contract change: a required `Dist` (LMCache's `sys_pop`)
already passed, and a hot set is one.

- **The old defaults are no case**, as for the build: 200M sectors and 32 threads of 100,000
  queries is about 360M reads, past what the catalog measures; their `hops`, `near`, and
  `rerank` were SIFT1M's anyway. The tests keep them through `tests/common`'s
  `DISKANN_SEARCH` (`n` 999,999,995, whose derived sectors are the 200,000,000 `nodes` gave;
  likewise 99,995 and 24,995 for the tests' 20,000 and 5,000).
- **The examples' fingerprints are unchanged** (sift1m-nocache ecbaca064f7812d0, 109,840 ops;
  sift1m-cache 49dcd7182152e40a, 115,416; `tests/cases.rs`). Both gained `rerank` 0.5 ms (the
  traces' user time, not fitted per case) and the index's four values; sift1m-cache gained
  `near`, the uncached trace's, the same index (a cache skips those rounds).
- **The golden configuration did not move** (0x4b25fcd827dbfa0f, 21,970 ops), its values now
  from `DISKANN_SEARCH`; only the AST's hash was re-recorded.
- **`kit/hops.py --params`** writes the index as the build names it, from the disk index's
  header and the PQ file's size, refuses a disk index whose header and size disagree (the
  derivation's check), and takes `--set` for `rerank`.

**Built: `faiss_ivf_ondisk`'s values** (2026-10-23, item 38's eighth mechanism, the last outside
the input pipelines). Decided with the user the same day ("let's start on faiss_ivf_ondisk", on
the recommendation): the index (`lists`, `code_size`, `list_bytes`, `index_bytes`,
`index_reads`), the search's configuration (`nprobe`, `batch`, and `list_pop`, the popularity of
lists over the job's queries, fitted, so the job's), the load (`threads`, `calls`), and the CPU
(`distance`, the scan time of a query) are required. The prefetch threads are derived as FAISS
starts them: `min_(prefetch_nthread, batch·nprobe)` per slice, `prefetch_nthread` a new
default of 32, the library's `OnDiskInvertedLists::prefetch_nthread`, in place of `prefetch`,
which a case had to keep consistent with `batch` and `nprobe` by hand (at 16 probes a fixed 32
read nothing: the inner loop's `batch·nprobe // prefetch` was 0). `batch·nprobe` is still
modeled as a multiple of the threads; a remainder is dropped, as before. No `scale`: the index
fixes every byte. No contract change: a required `list[int]` (dcp_save's `item_bytes`) and a
required `Dist` already passed.

- **No `st_blksize`, and a follow-up.** The lists are a mapping, with no buffer to size. The
  index file is read through stdio, whose buffer glibc sizes from the `st_blksize` of the file
  (`min(BUFSIZ, st_blksize)`, a [verify] claim from glibc's `_IO_file_doallocate`, not
  re-read for this decision): on a local file system the 8,192-byte fills would be 4,096. The
  trace's 8,192-byte fills on the NFS mount imply its `st_blksize` was at least that (inferred,
  not recorded), so the case's lengths hold there; modeling it
  needs `index_reads` derived from the arrays' sizes, which `read_index` reads directly once
  they exceed the buffer. Left as an open item, not built.
- **The old defaults are no case**: a million lists of 40 KiB, 32 threads of 100,000 calls at
  `nprobe` 64 is about 410M reads (two per list probed), past what the catalog measures. The
  tests keep them through `tests/common`'s `FAISS`.
- **The example's fingerprint is unchanged** (sift1m-ivf1024-pq32 d7deee5f942abc7e, 2,572 ops;
  `tests/cases.rs`): its `prefetch` 32 is what the derivation gives at `batch` 1 and `nprobe`
  64. It gained `distance` 0.4 ms, the value the defaults held: no trace measures the scan
  (the lists are read through the mapping), so it is not fitted.
- **The golden configuration did not move** (0x42c16b5296ebaefc, 25,612 ops), its values now
  from `FAISS`; only the AST's hash was re-recorded.
- **`kit/params.py`** writes no `prefetch` (the abstract derives it) and takes `--set` for
  `distance`, or to replace a fitted value.
- **Noted, not checked:** the lists dataset's slot is 256 KiB, which must hold the largest
  list (`list_bytes`'s max; 175,960 bytes in the case). A case with longer lists needs a larger
  slot; nothing refuses one whose lists overflow it.

~~**Open.** The exit status's number (3 proposed); whether accelerator names in use case
directories (`unet3d-h100`) need a vocabulary in `workloads/attributes.toml`; the proposed
points (8's prefixes, 11's mechanism, 12's tool name).~~ Every one decided 2026-10-22 (points
8, 9, 11, 12, 17, 18).

**Considered and not done.** Inheritance resolved at run time (editing a parent silently changes
every child; a flat lock with its own id does not); "last one wins" between fragments (which
file wins is the question users get wrong; agreement makes order irrelevant and conflicts
visible); a shared dictionary of parameter names so a rules file could name them (the
complexity the user ruled out; freeze by state needs none); `samples` derived from measured
memory (point 10); rules files holding code (constraints are data, checked by the runner); a
`when` field on constraints (the names already say); a reserved-word list for facts and
results (it breaks committed abstracts when the vocabulary grows; the prefix cannot); an
absolute path to the abstract in the lock (the lock would differ per site, and so its file
hash); `mlpstorage` building the abstract's path (the dependency would pass through a command
line again); one `--case` guessing a path from a slash (two options say what they do); a hash of
the results written into the report (the user's idea, dropped by the user: anyone who edits the
report recomputes it, and a key in a binary the submitter runs can be extracted); hashing the
run's output data (a read-back pass over the system under test; `aeiou-verify`'s job);
signatures on locks (the clean `mlpstorage` checkout is the trust anchor; worth it only if
entries are ever fetched from elsewhere); a full per-GPU step table by default (point 15);
the lock made only at the end of `aeiou-build` (the user's question, 2026-10-22: the build sees
neither a use case written after it nor a rules file outside the tree; point 17 keeps the build
re-locking what it can see); `aeiou lock`, a runner subcommand (proposed 2026-10-22 so the lock
and its check came from one binary, and withdrawn the same day: once the build calls the lock
code, a Python builder calling the runner would depend on it, and the runner's `dry-run` checks
the names anyway).

### 3.79 Locality metrics by extent, not by block (discussed with the user 2026-10-22; decided the same day)

**The problem.** The first use case at a real job's size, `unet3d-h100` (500 steps of 7 samples
of about 140 MiB, ~490 GB read per GPU), passed the metrics pass's guard of 2^26 entries:
`--metrics` keeps one entry per distinct 4 KiB block an instance touches (§3.39). The guard is
a memory bound, not a field width (times and counters are 32 bits); at 4 KiB the case needs
on the order of 20 GB. Sampling (`--metrics-sample`) fits it, but the user asked why we count
4 KiB units at all when almost every access is larger: a sample, a buffer fill, a KV chunk.

**The discussion.** A per-mechanism block (the application's own unit: 1 MiB for
`npy_per_sample`, the chunk for the KV mechanisms) fixed the size (1.2 s and 115 MB, exact)
but over-counted a 22-byte tail read as a whole block touched (`reread` 0.0072 at 4 KiB,
0.0277 at 1 MiB, the same class). The user proposed extents instead: byte ranges read through
sequentially, where a re-read of a whole extent is the boring case (an epoch, which `consume`
already says) and a re-read of a sub-range the interesting one (format overhead, a buffered
reader's overlap). The two kinds of byte a format has (framing and payload) are a separate,
later split (backlog).

**Decided (2026-10-22): extents are the locality unit; block metrics remain for the committed
captures until they are recaptured** (the user's choice of three: recapturing all seventeen now,
or judging them on less, were the others). No capture kept its raw trace, so the block sections
of `aeiou_metrics` stay, computed only when `--metrics-block` asks, and `aeiou-trace compare`
judges a pair on the sections both documents have. `--metrics-sample` is dropped: sampling at a
fixed granule would cut extents back into blocks; the guard stays, on pieces.

The definitions, chosen in the discussion and confirmed by the user:

1. **Pieces.** Per instance, the bytes it has touched, per path, as pieces: byte ranges each
   last touched by one op, with that op's time and kind. An op splits the pieces it overlaps
   at its own bounds; its bytes take its time. Memory grows with how fragmented the accesses
   are, never with the bytes touched.
2. **Reuse distance.** The bytes an op touches again are taken in ascending offset order, in
   groups: contiguous bytes last touched by one earlier op. A group's distance is the distinct
   bytes the instance touched after that op (this op's earlier groups included) plus the
   group's own bytes: Mattson's stack distance at byte granularity, the smallest LRU cache in
   bytes in which the group hits. The four histograms (by previous and this access's kind) and
   the first touches are **weighted by bytes**.
3. **Whole and partial re-reads.** An *extent* is the bytes one sequential run (§10's run) last
   touched. When a run ends, each earlier extent its reads touched again counts as **whole** if
   nothing of it is left untouched since, **partial** otherwise; its bytes so touched are added
   to `whole` or `partial`. On `npy_per_sample` a second epoch's pass over the payload is whole,
   its header and zip-tail reads partial; in the first epoch the payload pass re-reads the
   header reads before it whole, at a short distance, which the distances tell apart.
4. **Popularity** per byte: the accesses of each byte over all instances; `skew` is the share of
   the accessed bytes that go to the most accessed 1 % of the bytes. Object popularity is as
   before.

**What changes.** `aeiou_metrics: 2`: `reuse_extents` (the four histograms, first touches,
`whole`, `partial`, all in bytes) and `popularity_bytes`; `block` is null unless
`--metrics-block` gives one, and then `reuse_distance_bytes` and `popularity_blocks` are
there as in version 1. The same in `aeiou-trace metrics`. `aeiou-trace compare` adds the
extent rows (`extent reuse …`, `popularity, bytes …`) when both sides have them, the block rows
when both have them at one block size, and refuses a pair with neither. The catalog measures
`reread` and `skew` from the extent sections (a shape version), with no block. A backlog item
recaptures the seventeen captures with their kits; the block sections go once nothing compares
against them.

**Considered and not done.** Sampling by object (a single large file, as DiskANN's index, is all
or nothing); a per-mechanism block (a knob that extents make unnecessary); a larger guard (20 GB
for a unit no application reads in).

*Revised 2026-11-03 (`FS_API_SWEEP.md` S-18, the user's choices): the pieces were keyed by the
path's hash and never reset, so a file rewritten under its name (an `O_TRUNC` checkpoint, a
truncate, a punched hole) showed false reuse, and a write-to-a-temporary-then-rename-then-read
lost its write-then-read link. They are keyed by the data a file holds: a path names a file, an
open holds the file (its calls reach it after an `unlink` or `rename` of the name, as the
kernel's do), and what ends the data, whole or a range, ends it in every instance; the ended
bytes stay touched when they were, so distances do not shrink. An object of the popularity is a
file's data, so an `O_TRUNC` rewrite is a new object and a renamed file the same one.
`runner/REFERENCE.md` §10.*

### 3.80 Listing the use cases: `aeiou cases` (raised by the user 2026-10-23; the shape decided and built the same day)

**The problem.** A use case runs by name (`--case NAME`, §3.78), but nothing tells a user which
names exist or what each one is. `CATALOG.md` has it, as a document to read; the user expects
the cases to reach the middle double digits once a job is one case per accelerator (item 39),
and the attributes are a fixed vocabulary (`workloads/attributes.toml`), so filtering on them is
both useful and simple.

**Decided (the user, on the recommendation).** A subcommand of the runner, not of
`aeiou-catalog`: the person running a case on a system under test has the binary, while the
Python wheel ships for authors; the release already carries every record under `workloads/`
beside the binary, where `--case` looks; ~~and the measured attributes' classes are already
written in each `shape.json`, so the runner reads tags and re-derives nothing~~.

**What it reads: a generated index, not the records** (the user, 2026-10-23, on the
recommendation, once building showed it). Four attributes, `api`, `writes`, `inputs`, and
`status`, are written in no record: `aeiou-catalog` derives them from the AST and the captures'
verdicts. A Rust reader of the records would re-derive them, a second implementation held to the
first by a test. Instead `aeiou-catalog write` also writes `workloads/catalog.json`, every case's
attributes (curated, derived, and the measured classes), GPU counts, lock, description, and
docs, with the vocabulary; `check` holds it byte for byte as it holds `CATALOG.md`, and the
release ships it. The runner reads one JSON file and parses no TOML. The cost: a case someone
adds appears only after `aeiou-catalog write`, but authoring a case already needs the Python
tools (`aeiou-params lock`). The vocabulary is a list, not an object, so a reader keeps the
dictionary's order without depending on its JSON library's (the runner's `serde_json` sorts
keys).

- `aeiou cases` lists **every** case of every kind, the kind its own column. Most cases today
  are traced examples or unsettled, so hiding them would leave the list nearly empty; the
  default may change to leave out examples once validated use cases are many (the user's
  note).
- `--where ATTR=VALUE` keeps, `--where ATTR!=VALUE` drops, several ANDed; `--json`
  for scripts. A bare `aeiou cases` lists (the user's choice over printing the help, as every
  subcommand's bare form does its default action); `aeiou cases --help` gives the syntax and
  the vocabulary, every attribute with its values, curated and measured, without their
  meanings, which `CATALOG.md` gives. The vocabulary is read from the tree's `attributes.toml`
  when the help is printed, not compiled in, so a binary and its `workloads/` never disagree.
- `aeiou cases NAME`: one case's attributes, GPU counts, what its lock fixes and what a
  submission must still give, its `DESCRIPTION.md`, and the commands that run it.
- A `--case NAME` matching nothing names the near matches and `aeiou cases`.

~~**The cost.** A second reader of the records, in Rust, beside the Python one that writes the
catalog. A test holds `aeiou cases --json` equal to what `aeiou-catalog` loads from the
committed tree, as the two checkers are held to agree.~~ No second reader: the index above.
`aeiou cases --json` prints the index's rows as they are.

**Exclusion is `!=`, not `--no-where`** (the user, 2026-10-23, after first asking for
`--no-where`): every `--no-x` of the suite turns a boolean off, the last one on the line winning
(`runner/REFERENCE.md` §14), and `--no-where` is no boolean, so `--where kind=example --no-where
kind=example` would read as the last one winning where both apply. One flag, its operator in
the value, keeps the convention; `!` wants quoting in some shells.

~~**Open.** How a repeated `where` reads from a `--config` file (a TOML array, to be settled
against `runner/REFERENCE.md` §14).~~ Settled in the build: `where` is a layered list, as
`--endpoint` is (§14): an array of strings in the `[cases]` table, whitespace-separated in
`AEIOU_WHERE`, replaced whole by a higher layer; *NAME* and `--workloads` are fixed, as for
`--case`.

**Built** (`runner/REFERENCE.md` §18, `builder/REFERENCE.md` §8): `aeiou cases`, the index, and
the near names of a `--case` that matches nothing.

**Considered and not done.** A pointer to `CATALOG.md` alone (a document, not a query, and the
cases multiply); a search tool in the Python wheel (not on the hosts that run cases); free-text
search (the tags and a name match cover it, and `grep` the rest).

### 3.81 A use case per accelerator: the Unet3D family (item 39; decided with the user and built 2026-10-23)

**The problem.** §3.78 point 1 makes a job's batch and step time the model's on one
accelerator, so one case cannot hold them for every accelerator; `unet3d-h100` held both
fragments in its own directory until a second accelerator shared the job's (part 5a).

**Decided with the user, on four questions:**

1. *A shared job fragment is its mechanism's*, `npy-per-sample/jobs/unet3d.params.json`, each
   case listing it beside its accelerator's (`["../../jobs/unet3d.params.json",
   "h100.params.json"]`). `aeiou-catalog check` already takes a fragment of the case or its
   mechanism, and a lock records it by its path under `workloads/`; no contract or checker
   change.
2. *The accelerators the WG has numbers for*: `unet3d-a100`, `unet3d-h100`, `unet3d-b200`, the
   `accelerator` vocabulary gaining `a100` and `b200`. No B300 until one is measured: no
   configuration gives one.
3. *The values are the WG's reference parameters* (`PROJECT_BRIEF.md` §8), from its DLIO
   workload configurations `unet3d_{a100,h100,b200}.yaml` (`batch_size`, `computation_time`):
   batch 7 on all three, 636, 323, and 162 ms. These are the WG's per-GPU compute, not DLIO's
   I/O, so the rule against DLIO abstracts does not apply. B200's 162 ms is the WG's H100
   halved, not a measurement, and its fragment says so. The batch does not change with the
   accelerator's memory in the WG's numbers.
4. *The sample sizes are the WG's too, for now* (the user's answer): mean 146,600,628 bytes and
   standard deviation 68,341,808 (`record_length_bytes`, `record_length_bytes_stdev`), in place
   of 140 MiB and 4 MiB, whose source no record gave; provisional until the preprocessed KiTS19
   corpus is measured. The normal stays, clamped at 1 MiB.

**Built.** The three cases and the job fragment. The step time is compute, which the
fingerprint does not cover, so the three issue one op multiset (`958ae6fd74afc74b` at 2 GPUs,
seed 1, 5 steps, 51,143 ops, against `f755bd4187b75b78` and 50,491 at the old sizes; the
abstract's own golden run in `tests/golden.rs` keeps the old values as parameters) and are three
workloads by their ids (`tests/cases.rs`).

**Open.** The sizes measured on the real corpus; a B300, and any accelerator, measured; the
other jobs' accelerators when their use cases come (the input pipelines' other mechanisms have
example and `unsettled` cases only).

### 3.82 Every tool a version the repository names (decided with the user and built 2026-10-24)

**The problem.** `ubuntu-latest` began moving to Ubuntu 26 on 2026-10-19, and CI took its
Python from the image: run 40068672434 used `/usr/bin/python3`, 3.12.3, while development used
uv's 3.12.13, and `requires-python = ">=3.12"` would have taken the next image's Python without
a word. uv was the newest each run (0.13.5 in CI, 0.11.2 locally), Rust the newest stable
(1.99.1 in CI, 1.95.0 locally), nextest the newest, pandoc the release image's apt package, the
build backend (`hatchling`) unpinned, PyInstaller pinned but beside the lock (`--with`), and the
coverage tools installed by hand. The packages alone were locked.

**Decided with the user** (the user's request: a uv venv, every Python version locked, nothing
from outside the venv where that can be had):

1. *The Python is one exact version, always one uv downloads*: `builder/.python-version`
   (3.12.13, the version development used) and `python-preference = "only-managed"` in
   `[tool.uv]`, so uv refuses the system's interpreter even where it would satisfy the version.
   The interpreter is python-build-standalone, the same bytes on every host; what it needs of the
   host is glibc, as every manylinux wheel does. `requires-python` stays `>=3.12`: it is the
   wheel's, for its users.
2. *uv is one version*, 0.13.5 (the user's choice, the version CI had passed with), as
   `required-version`, which setup-uv (v10, a Node 24 action) reads from `builder/pyproject.toml`
   and installs by its SHA-256.
3. *Everything Python is in the lock*: PyInstaller as the dependency group `release`, the
   coverage tools as `coverage` (neither installed unless asked), and `hatchling==1.32.4`.
4. *Rust is one version*: `rust-toolchain.toml` at the root (1.99.1, CI's; with
   `llvm-tools-preview` for `cargo llvm-cov`), so cargo run from `builder/` with
   `--manifest-path` takes it too; CI runs `rustup toolchain install` in place of
   `dtolnay/rust-toolchain@stable`. nextest is `0.9.149`.
5. *The image is named*, `ubuntu-24.04` (the release stays on `ubuntu-22.04` for glibc 2.34).
   strace is the image's, so a new image is a deliberate change. pandoc is its release
   3.12.1, downloaded and checked by SHA-256 as MinIO is.
6. *The release ships the environment it was tested in*: `wheel/requirements.txt` (`uv export
   --locked`, versions and hashes) and `wheel/python-version`; `QUICKSTART.md` and the release's
   smoke test install by them, `--require-hashes` and the wheel `--no-deps`. Plain pip stays the
   fallback, the same libraries under the system's Python.

`schema/check.py` loses its `python3` shebang (it was never executable) and names `uv run
--project builder python` in its usage. `builder/tests/test_pins.py` holds all of this: the
tests run under the pinned Python, and no workflow takes a version the day it runs.

~~**Open.** How the tools are invoked outside `builder/` (the user, 2026-10-24, from the WG
repository's `uv run --project` wrapper): `uv run --locked` checks the venv against the lock on
every start; the entry points in `builder/.venv/bin/` already run the venv's interpreter by
their shebang without uv, but do not notice a stale venv.~~ Decided the same day: §3.83.

### 3.83 The release for running, `dev.sh` for building (decided with the user and built 2026-10-24)

**The problem.** The WG's own repository is cloned and run, there being no releases, so its
`mlpstorage` is a `uv run --project` wrapper: a user who never ran uv, with no venv on PATH,
still gets the locked environment. aeiou has releases.

**Decided with the user:**

1. *Submitters run the release*, never a clone: the tarball's runner, its one-file
   `aeiou-header`, and the abstracts need no Rust, no Python, and no build. So no wrapper
   scripts for running (the user: "we don't need uv or (re)building the Rust code or all the
   complexity those bring"). Python is for authoring (`aeiou-build`, `aeiou-params lock`,
   `aeiou-trace`, the catalog), which the wheel serves (§3.82 point 6). Not yet: ~~an aarch64
   release (Grace hosts build from source),~~ (built the same day, §3.84) and `aeiou-header` under a `noexec` `/tmp` (PyInstaller
   unpacks a one-file executable into `$TMPDIR` each start; not tested).
2. *A clone is for changing aeiou*, and needs rustup and uv; the rest the checkout names
   (§3.82) and they fetch. What a contributor would otherwise have to know (the header link,
   MinIO, the extras, the `AEIOU_REQUIRE` lists) is `dev.sh` at the root (the name kept over
   `environment.sh`, which reads as a script to `source`, and this one exits on every error, and
   `setup_environment.sh`, which names one of its four commands; and over `make`, whose download
   checks would be a script behind it, two entry points): `setup` (the runner `--locked`, the venv
   with every extra and the `coverage` group, `aeiou-header` beside the binary), `check [JOB]...`
   (CI's four jobs, the same steps), `cov` (Rust by cargo-llvm-cov, Python by pytest-cov, the CLI
   subprocesses included), and `fetch TOOL` (MinIO, nextest, cargo-llvm-cov, each one release by
   SHA-256, kept in `$AEIOU_DEV_CACHE` and checked again on every use). A missing rustup, uv, or
   strace is named with where to get it before anything runs. Each CI job is now
   `./dev.sh check JOB` after setting up Rust and uv, so a checkout runs exactly what CI runs, and
   `taiki-e/install-action` and the MinIO download leave the workflow. POSIX sh, usage errors in
   the suite's frame (exit 2).
3. *pip is noticed, not refused.* `pip install -e builder` takes none of the pins. A tool run from
   a checkout (told by `builder/.python-version` beside the package) under another Python prints
   one warning on stderr and goes on (`aeiou/checkout.py`; a wheel's install has no pin file, and
   its users may run any Python it allows), since the HDF5 header bytes are libhdf5's: a pip-built
   `aeiou-header` writes another dataset. And `test_pins.py` refuses any installed package that
   is not `uv.lock`'s at the lock's version, naming `./dev.sh setup`, so the first test run finds a
   pip install.

`setup-uv` is named at its exact release (`v10.3.2`): it publishes no major tag, which run
40143194028 found (three jobs failed at setup), and `test_pins.py` now requires the exact form.
The tools in the docs run from `builder/.venv/bin/` (their first line is the venv's Python) in
place of `uv run`, which without `--locked` re-locks a drifted `pyproject.toml`.

### 3.84 An aarch64 release, built on its own hardware (built 2026-10-24, at the user's request)

Grace hosts (and any aarch64 client) had to build from source (§3.83 point 1). The release
workflow is now a matrix of two legs, `x86_64` on `ubuntu-22.04` and `aarch64` on
`ubuntu-22.04-arm`, each building, assembling, and smoke-testing its own tarball,
`aeiou-VER-ARCH-linux.tar.gz`, and a last job that attaches both to the tag's release.

- *Native, not cross-compiled.* The runner would cross-compile (it links nothing but libc), but
  `aeiou-header` is a PyInstaller one-file executable, and PyInstaller builds only for the machine
  it runs on; the smoke test, which runs both binaries and an HDF5 datagen through the header,
  needs the hardware too. GitHub's arm runners are free for a public repository.
- *The same image generation*, 22.04, so both binaries need glibc 2.34 or newer, as before.
- *pandoc and uv for each architecture*, the same releases (3.12.1, 0.13.5), each tarball by its
  own SHA-256 in the matrix (`test_pins.py` holds both legs and the checksum form); the first
  dispatch, run 40152549826, failed at setup-uv on the arm leg, whose download the x86_64
  checksum refused. `fail-fast: false`, so one leg's failure no longer cancels the other's.
- *The smoke test checks the architecture*: `file(1)` must name the leg's for both `aeiou` and
  `aeiou-header`, so a leg that ran on the wrong runner fails.
- *A dispatch builds without publishing* (`workflow_dispatch`): both tarballs become the run's
  artifacts and no release is made, so the arm leg can be tried before a tag; the version is then
  the checkout's (Cargo.toml and pyproject.toml must still agree).

~~Not done: `dev.sh fetch` stays x86_64-only (its pinned nextest, cargo-llvm-cov, and MinIO
downloads), so a contributor on aarch64 runs `./dev.sh check` with those tools from elsewhere.~~
*Done 2026-10-25, at the user's request:* `dev.sh` pins each download per architecture
(`pin TOOL ARCH URL SHA`, the architecture `uname -m`'s), the aarch64 files of the same three
releases (MinIO's hash checked against its `.sha256sum`, nextest's and cargo-llvm-cov's against
the digests GitHub records for the release assets, as neither project publishes a checksum
file). The cache holds a directory per architecture, since a home directory, and so the default
cache, is often one both kinds of host mount; an architecture with no pin is refused by name.
Red first: `test_pins.py` requires both architectures' file of one version for each tool, and
`test_dev.py` an aarch64 host fetching the arm64 file into its own directory.

### 3.85 The driver's planning note and the items it raises (discussed with the user 2026-10-28; items 41–49 queued the same day)

The MLPerf Storage driver, `mlpstorage`, wrote a scoping note (2026-10-28, local to its tree)
for replacing DLIO, its Milvus harness, and its KV-cache simulator with aeiou. Its survey of
aeiou is accurate; its asks of this side are a report schema, the WG's closed workloads as
catalog cases, the `mi355` accelerator, and three metrics it cannot find (AU, tokens/s, recall).
The user asked for the appropriate asks to join the backlog as written, the rest restructured
so the driver's integration gets easier, and three items of their own: a fitting skill, and the
case for calling aeiou a "scalable trace". `PROJECT_BRIEF.md` §6 items 41–49.

- *Taken as written:* a schema for `aeiou_report: 1` and the build identity in the report
  (item 41); the closed workloads in the catalog (item 45). Two of the note's shortfalls are
  smaller than it says: RetinaNet is a job over `file_per_sample`, not an abstract, and AU is
  `result.busy_fraction` already. One is larger: `dcp_save`'s `item_bytes` is one list for every
  rank (FSDP's 1/G of each tensor), and pipeline stages hold different layers, so the WG's
  TP/PP cases may need an item list per rank class first; and whether DCP's op stream is what the
  WG scores (its benchmark wrote with its own writer) is the WG's decision, raised to it.
- *Restructured:* the note plans a result reader, a fingerprint check, sizing math, and a
  tokens/s derivation in the driver. Each is a definition of ours, so each moves here as a
  mechanism and stays out of the driver: a reader in the wheel (`aeiou.report`, item 41; it
  reads a file, the process rule of §3.65 holds), `aeiou report check` recomputing lock hash,
  workload id, and fingerprint from a report (item 42, the WG's validation step of §3.78
  point 14), the dry run's projection as a report document (item 43), and headline figures as
  expressions a lock declares (item 44: tokens/s, samples/s, queries/s, GB/s are formulas over
  `result.` names and parameters; the runner evaluates and never interprets, and the WG's
  "which metric scores" becomes a line in a lock it owns). An integrator's page gathers the
  launcher one-liners (`AEIOU_RANK` from mpirun's or srun's rank variable), the exit codes, the
  checkpoint sequence, and an example rules file (item 46). Refused: nothing; the five-run loop,
  warm-up, sizing policy, and the lock in the driver's image stay the driver's (§3.74, §3.78).
- *The fitting skill (items 47, 48).* The user's reasons: contributors who will not release a
  trace may release its abstract; vendors recreating a customer's workload; the library's
  growth; and an analysis that reassures a contributor about what leaves. Two flaws found in
  the plan as stated, and the design around them. First, a hosted LLM reading a raw trace
  discloses it to the LLM's vendor, which the contributor's policy usually forbids as much as
  disclosure to us; so the raw trace is read only by a deterministic local tool, `aeiou-trace
  digest`, whose output (path templates with every literal a field, op mix, size and offset
  histograms, reuse, phases per thread) is what an LLM, hosted or local, or a person fits from,
  and a contributor may send the digest alone. Second, an abstract is not free of proprietary
  content until its path literals, prose, and provenance are scrubbed, and sizes alone can
  identify a model or corpus; the honest statement is that it discloses the I/O shape and
  scale, which is the point, and `aeiou-catalog disclose` lists exactly what would leave so the
  contributor decides. The LLM decides structure, never numbers: the numeric fit is
  `aeiou-fit`, the deterministic fitter the kits' `fit.py` scripts become, and
  `aeiou-trace compare --judge` is the only acceptance gate. The skill is in the open skill
  format, vendor-neutral, and measured: the committed captures are its evaluation set.
- *"Scalable trace" (item 49).* The user's framing: a fixed trace replicated in parallel
  amplifies its hot spots, in series repeats accesses never made, offset in time makes the
  offset a free parameter; aeiou generates a stream of any length over any GPU and host count
  with no replication artifact, by construction (positional randomness, the Feistel split).
  Agreed, with the honest limits the note must carry: the claim is "no artifact from
  replication", not "no amplification", since shared structure (a step barrier, an index's hot
  nodes, a KV cache's shared prompts) is amplified exactly as the abstract models it, and every
  fit is validated at one scale (no application traced at two scales, fitted at one, predicted
  at the other). The evidence is measurable here: the `trace` node is the fixed trace and the
  abstract the scalable one in the same runner (§3.58), so N offset copies of a trace against
  the abstract at N, merged metrics side by side, is the experiment; `dry-run --emit-trace`
  makes the scaled stream a file. The term is defined on first use; "trace-driven",
  "model-driven", and "synthetic trace generation" are the literature's neighbours.

- *DLIO as a stand-in (measured 2026-10-28, §3.46).* A WG member's position that tracing DLIO
  is tracing the application, since both drive PyTorch's `DataLoader`, was put to the test the
  same day: the fork and the real `np.load` loader traced on one corpus at the WG's Unet3D
  shape. True of the scheduling layer, false of the bytes: the fork reads each sample in one
  call and never decodes it, and its worker pre-warm re-reads the corpus, 1.32× the loader's
  bytes at 2 workers and 2.50× at the H100 configuration's 4; 15 of 21 judged rows outside.
  The user's conclusion is recorded in §3.46: real-application traces are the requirement for
  every supported workload, the WG's configurations supply parameters and never op streams
  (item 45), and a transition to aeiou is a break in comparability with earlier results, to be
  stated, caused by the new generator being the more accurate one.

### 3.86 How a trace becomes an abstract today, and what the digest must add (the user's question, 2026-10-28)

The user asked how `aeiou-trace` and the builder scan a trace, find its patterns, and generate
the AST, and what confidence to have that scripted sequences are correctly separated from
operations that need randomness. The driving case: a vendor asks a prospective customer to trace
their application and run `aeiou-trace` on it to find and describe the patterns, with no LLM
anywhere. The answer, recorded as the starting point for item 47's design:

**Nothing in aeiou scans a trace and finds its patterns.** The structure of every committed
abstract was written by a person who read the trace beside the application's source and encoded
the loops, draws, and parameters in the builder. `aeiou-trace` measures a trace, exports it for
literal replay, and judges an abstract against it. The kits' fitters fill in numbers for an
abstract whose shape already exists. Pattern detection is backlog (items 47, 48), and even the
planned digest stops short of deciding what is scripted and what is random.

What each piece does:

- `aeiou-trace metrics` (§7 of `builder/REFERENCE.md`) parses the strace line by line, keeps the
  calls on paths under the root, tracks each open file description's position through `open`,
  `lseek`, and the bytes moved, follows `dup`, threads, and `fork`, and reduces the result to
  the metrics document: read/write mix, request-size histograms, sequential run lengths per
  thread and path, reuse distance over byte extents in completion order, popularity of extents
  and objects, fan-out and depth where `io_submit` shows them. A reducer; it infers no loop,
  template, or distribution.
- `aeiou-trace export` writes the trace as the runner's `trace` node (§3.58): every call in
  order, one lane per task. The zero-inference path: replayed literally, never CLOSED, no
  scaling.
- `aeiou-trace compare --judge` is the only acceptance gate: two metrics documents row by row,
  a tolerance per metric class, widened by the abstract's own spread across seeds (`--self`).
- The builder is a DSL. A person writes the loops, the `consume`, the reads, and names which
  quantities are parameters; it emits the symbolic AST and never looks at a trace.
- The kits' `fit.py` and `params.py` are per-abstract numeric fitters: given the shape, they
  measure the trace or the application's own metadata (FAISS: list-size distribution and
  popularity curve from a search trace; DCP: item offsets from the checkpoint's `.metadata`;
  KV: reuse gaps from the replay log). Each knows what to look for because the structure was
  decided first.

**How scripted and random were separated in practice: by reading.** For each abstract the
application's source gave the call stream, one file's merged call sequence from the trace
confirmed it, and the two were matched: a sequence of calls recurring identically per file
became a fixed sub-sequence, a position that varied became a draw whose distribution was
checked against the metrics. Where the source was unavailable or the pattern data-dependent
(DiskANN's hops, KV reuse), the distribution was a hypothesis and the gate the test. The
reasoning for every such choice is in this document with its date, which is also the honest
statement of how much judgment each abstract holds.

**The confidence the gate warrants.** It is statistical, not sequential. Two abstracts that
reproduce the same reuse-distance, run-length, request-size, and popularity histograms both
pass, one of them with the loop nesting wrong. Three things it cannot see:

- Order within a GPU, by design: issue order is timing-dependent and never hashed.
- Causality: an `x @ i` dependency and a draw from a fitted reuse distribution give the same
  reuse histogram at one scale and diverge at another, which is why item 49 says every fit is
  validated at one scale only.
- Anything strace cannot see: `io_uring` submissions and `mmap` faults.

An abstract that passes reproduces the trace's locality at the traced scale within the stated
tolerances. It does not prove the structure is the application's.

**The customer's case.** Today the customer can run `aeiou-trace metrics` and send the metrics
document, which carries no paths and no bytes of their data, only histograms and shares; the
vendor authors the abstract from it and from knowledge of the application, and the customer runs
`compare --judge` locally against the vendor's dry-run metrics. What the case needs and does not
have is `aeiou-trace digest` (item 47): a deterministic structural summary with path templates
in which every varying field becomes a variable, per-thread phase segmentation by repeated call
motifs, offset-delta histograms that classify runs as sequential or not, and size and reuse
histograms. That is as far as a deterministic tool goes without a model of the application: it
can say "this thread repeated this 9-call motif 560 times per file with offsets advancing by
256 KiB", a scripted sequence, and "file choice has a reuse-distance distribution of this
shape", which needs a draw. It cannot name the loop nesting, tell a Feistel-shuffled epoch from
a uniform draw with replacement, or tell a data-dependent dependency from a fitted distribution.
Those remain a person's or an LLM's call, which is why item 48 puts the LLM on the structure,
`aeiou-fit` on the numbers, and the gate on acceptance. With no LLM and no source reading, what
a customer produces today is a metrics document, and after item 47 a digest. Either is
something a vendor can fit to. Neither is an abstract.

**Next (the user, 2026-10-28):** design and build the digest, starting from this section and
item 47's list.

### 3.87 `aeiou-trace digest`: what a trace decides, what a person must, and asking them (designed and built 2026-10-29)

Item 47's first half, built from §3.86. Two questions from the user set the design.

**The first: can a workload be generated that matches the trace, with the digest calling out
where the trace is non-specific?** The user accepts that many abstracts produce the same trace
and wants fidelity to the trace in hand, with the places where two constructs both fit (a
`consume`, each sample once per pass, against a `choose`, a draw with replacement) named rather
than silently chosen. The answer is that most such questions are statistical with a computable
answer, and the digest makes "indistinguishable here" a first-class verdict: for every site it
reports the set of constructs the evidence allows, the number behind each verdict, and what
evidence would separate the candidates it cannot.

**The second: is it worth asking the application's owner the questions the trace leaves open?**
The trace will be short, since a short trace is easier to take and to hand over, and the
benchmark will run the workload wider and longer, where a draw and a permutation diverge. The
answer recorded here: strongly positive, under three conditions. The undecided verdicts are
exactly the questions whose answer changes the op stream at scale, so they are the most valuable
thing a person can add, and the set is finite and small. The conditions: (1) answers carry
provenance, who asserted them, and are never written as measurements; "don't know" is a legal
answer that leaves the construct `unsettled`, as the catalog already allows; (2) questions are
about observable behavior ("does the loader revisit a sample within one pass?"), never about
constructs, so a third truth is not foreclosed by an A-or-B framing; (3) every open item states
what capture would settle it, because a slightly longer trace is sometimes cheaper than an
answer. The representativeness of the fitted abstract at scale is then conditional on the
asserted answers, and the disclosure report (item 47's second half) must list them beside the
measured facts. In the WG setting these are app-side facts fixed once per workload by whoever
fits it (the rule of §3.64 applies: app-side facts are the WG's, never a submitter's choice). ~~**The
digest itself asks nothing**; the asking is a person's or the fitting skill's (item 48), and an
answers file the digest reads to move an item from `open` to `asserted` is that item's work,
not built today.~~ *Reversed the same day, §3.88:* the digest asks, on a terminal, and writes the
answers with provenance into the document and an answers file beside it; the sentence struck was
this side's departure from the user's plan, not a decision of the user's.

**What was built** (`builder/aeiou/digest.py`, standard library; `aeiou-trace digest`,
`man/aeiou-trace.1.md`; 30 tests, the module at 100 % line coverage):

- *Input.* The exporter's event stream (`export_of`), so the descriptor tables, `clone`,
  `dup`, `cwd`, and positions are the one resolver's and the digest can never disagree with
  `export` about which calls there were; or an exported trace file, which carries its root.
  Positions of unpositioned reads and writes are tracked per open id as `TraceReader` tracks
  them (a `pread` leaves the position where it was).
- *Path templates.* Every run of decimal digits in a path under the root is a field; a run
  of 16 or more hex digits with a letter in it is a hex field (KV-cache keys, content hashes).
  Rendering: a fixed width when every value had the same width above one (`{a:05}`,
  `{a:x32}`), else `{a}`. Per field the count of distinct values and, for integers, the range;
  per template the op mix, bytes, request-size histograms, and per open how it moved through
  the file: `whole` (one sequential run from offset 0), `sequential` (one run from elsewhere),
  `positioned` (more than one run), `none`. No raw path unless `--paths`. A `rename` counts
  under its source's template and `renamed_to` under its destination's, so the final names
  are a template too.
- *Units, motifs, phases.* A lane is a traced task. A unit is the calls on one open from
  `open` to `close`; a path call, or an `io_submit` with its members, is a unit of one op. A
  unit's motif is its op sequence with the shortest repeating cycle of up to four ops collapsed
  at each position (`read+`, `(lseek read)+`), the counts kept as histograms. A phase is a run
  of consecutive units on one lane with the same motif once the repeat marks are dropped, on
  one template, so a file read in one call and a file read in seven are one phase; the phase
  records units, ops, bytes, repeat counts, and the think time between its units.
- *Sites and verdicts.* A site is a template opened at least twice; its opens in time order
  are a sequence of paths. Passes are greedy: a pass ends at the first path it already holds;
  the last pass is partial unless it is as long as the first. **Selection**: with two or more
  full passes of equal length *L* and fewer than 1.5 *L* distinct paths, `consume` over *L*
  (paths seen but not in every pass reported, the signature of `drop_last`), `choose` refuted
  because a draw repeats after about √(πN/2) touches and its passes are unequal; with passes of
  unequal length, `choose`, `consume` refuted, with the corpus the mean first-repeat distance
  implies (2·mean²/π), the touches per path, and the top-decile share; with one full pass and a
  partial second, undecided between both, settled by the second pass; with no repeat at all,
  undecided, settled by a second pass or by the corpus size: `--candidates N` refutes the draw
  when no repeat in *k* touches has probability exp(−k(k−1)/2N) below 5 %, and otherwise the
  digest reports the touches after which a draw would have repeated at 95 %. Without
  `--candidates`, the smallest corpus a draw needs, k(k−1)/(2 ln 20), is reported instead. For
  the 28 touches of the npy capture that is 126 files. **Order** within a pass: the share of
  adjacent pairs ascending in the sorted order and the share with a step of exactly one;
  `sequential` at 90 % steps of one, `permuted` when the ascending share is within three
  standard errors of one half, `other` otherwise (interleaved scans, a sort the trace does not
  show), undecided under eight pairs. **Size**: bytes per open `constant`, `per-path` (a
  repeated path always moves the same bytes), `drawn` (it does not), undecided with no repeat.
- *The open list.* Every undecided verdict as an item: the behavioral question, the candidate
  constructs, what settles it, and the candidates' divergence at `--epochs E` passes: for
  selection the share of the set a draw never reads, e^−E (0.37 at one pass, 0.14 at two,
  0.007 at five), against zero for a pass, and the share of touches that re-read a path under
  each; for order and size the divergence is zero in the op multiset, and the item says what
  does differ (the order read-ahead sees; which file gets which bytes). Ranked largest first.
  Because the ranking is a formula per dimension, it needs no model and stays in the standard
  library.

**First run on a real trace** (the npy kit's `train.py`, 24 archives of 1 MiB, batch 2, two
workers, two epochs, local ext4): 1,610 ops on five lanes; three templates, `train`,
`train/{a:05}`, and `train/{a:05}/sample_{b:09}.npz` with 24 paths and 48 opens; one site, passes
[24, 24], **selection `consume`**, choose refuted by two full passes of equal length; **order
`permuted`** (0.46 of 46 pairs ascending, 0.07 steps of one); **size `per-path`**; no open
question. Each worker lane is one phase of twelve units whose motif is np.load's zip reading
(`open fstat ioctl lseek+ read lseek+ (read lseek)+ …`), with the think time between units at
131 µs. The same loader at one epoch would have left selection undecided with 126 files named
as the corpus below which a draw is refuted, which is the case the user described: the digest
cannot decide it, a second epoch or the owner can.

**Not decided by any trace length, stated in the digest's terms as in §3.86:** a
data-dependent `x @ i` against a fitted distribution (item 49), the loop nesting when two
nestings give the same sequence, and a Feistel pass against any other permutation (identical by
construction, so no fidelity question). For the trace in hand, indistinguishable candidates are
equally good; the open list exists for the scale the trace was not taken at.

**Not built, in order:** ~~an answers file with provenance that moves open items to `asserted`
(item 48, where the questions are asked)~~ the question loop and the answers file (§3.88, the
digest's own); digest-to-digest comparison, the fidelity loop made
literal by digesting an `strace` of `aeiou run` under the abstract's declared backend beside the
application's and comparing verdict by verdict, since the runner has no op-stream dump and the
interposition test says it should not need one; `aeiou-catalog disclose` (item 47's second
half); a uniform-against-skewed verdict for `choose` (the numbers are reported, no verdict yet).

### 3.88 The digest asks; the four workload types; application records; object-store logs (designed with the user 2026-10-29)

Four design conversations with the user on the day the digest was built, each changing or
widening §3.87. Recorded before any of it is built; the build order is at the end.

**The digest asks: a reversal of §3.87.** §3.87 said the digest never asks and left the asking
to a person or the fitting skill (item 48). That was this side's recommendation, written into the
design reply without flagging that it departed from the user's stated plan, which was that
`aeiou-trace digest` asks its questions interactively before it writes its conclusions; the
user's "proceed" to the whole was taken as agreement to that point too. The user disputed it
and the point is reversed. The reasons given for "never asks" were real, but they argue for a
non-interactive *mode*, not for no interactive mode: the person at the console when the digest
runs is the one who knows the application, and asking them then, with their answers shipped
inside the digest, beats asking a vendor later who does not. Item 48 is about the fitting
skill's prompt and workflow, a different and larger thing; the digest is a tool the skill
runs, and since the skill will be run by people with little or no aeiou experience, answering
must be easy: a question on the terminal, never a file to edit.

Two consequences, both requirements rather than objections. (1) *The record stays a
measurement.* A verdict that changes because of an answer must show the answer and who gave
it, so answers go into the document under `asserted` with provenance, and a second run with
the same answers produces the same document. (2) *The non-interactive callers.* The skill, a
script, and the test suite need the tool to run without a terminal: the digest asks only when
its standard input is a terminal and no answers were supplied; `--answers FILE` supplies them;
a flag leaves every item open and asks nothing. Nothing else conflicts: the usage-error frame,
the manual-page tests, and the catalog are untouched by a question loop.

**The trace is a file.** The user asked whether reading the trace from standard input, carried
over from `--metrics` and `export` where one reduction is the whole job, defeats the comparison
of the fitted abstract's dry run against the trace and the re-run of the digest with answers.
It does, twice: `aeiou-trace TRACE --metrics` after the fit needs the trace, and a re-run with
the same answers must read the same trace. To be exact: answers never change how the trace is
read; they settle verdicts the trace left undecided, and an answer the trace contradicts is
rejected with the evidence shown. So trace file plus answers file reproduces the digest, and
that pair plus the digest and the metrics is the record a capture keeps. Decided: the digest
refuses `-`, records the trace's path and hash in the document, and keeps accepting the
exported trace file as the compact form an owner may retain instead of a large raw strace.
`--metrics` and `export` keep `-`.

**The question experience.** Agreed with the user: (1) facts first, then behavior; the facts a
person usually knows outright often decide verdicts without any behavioral question; (2) every
answer is checked against the trace, and a contradicted answer is rejected, not recorded, the
one place a tool can protect a fit from a confident wrong answer; (3) one question, one screen:
the evidence in two lines, the behavioral question, numbered choices with "don't know" always
present, one line on what the choice changes at scale, answered by number, then an optional
"how do you know?" note for provenance; (4) the answers file is a by-product, written beside
the document by the interactive run together with the command line that reproduces the run
without questions, never something a person edits; (5) the digest owns the questions, their
text, the consistency checks, and the answers format; the skill owns the workflow around the
tool and may ask in its own voice from the tool's question list and pass the answers back, and
both paths produce the same record.

**The facts are generic, not training's.** The user objected that the two fact questions first
proposed, how many files the dataset holds and how many passes the benchmark runs, are
training's questions and meaningless for a KV cache, and asked that every question and verdict
be measured against the four major types: training, checkpointing, vector search and RAG, and
KV cache. Done below from the kits, the abstracts, and the committed captures (no raw trace is
retained in the repository, and the kit libraries, torch, diskannpy, faiss, are not installed
on the development box, so this is read from the sources, not run). The generic facts that
replace the two: for each template field, its value range at scale (sample ids, ranks, steps,
key counts); for each phase, its repeat count at scale (epochs, checkpoints, queries,
requests); for a single open read at offsets, the file's size at scale. `--candidates` and
`--epochs` are the command-line statements of two instances of these and stay as such.

| type | what the digest of §3.87 sees | what is missing | the facts | the open questions |
|---|---|---|---|---|
| training (`npy-per-sample`, `file-per-sample`, `safetensors-load` as model loading) | sites, passes, `consume`/`choose`, order, size per path | nothing structural | field range (samples), phase repeats (epochs) | revisits within a pass; order; bytes per file |
| checkpointing (`dcp-save`, `dcp-load`) | a site on the save template with fields rank and step, each path written once per checkpoint, so one pass per checkpoint | read-shaped wording ("does the application revisit a sample") on a write site; load is a fixed item list per rank class (item 45(b)), not a selection verdict | field ranges (ranks, steps), phase repeats (checkpoints), bytes per rank | whether names are new per step or overwritten; `fsync` placement; bytes scaling with model and ranks |
| vector search and RAG (`diskann-search`, `diskann-build`, `faiss-ivf-ondisk`) | one index file opened once and read at many offsets: **no site at all** (a site needs two opens), `offsets: positioned` and nothing more | the within-one-open structure: reuse, runs, skew, dependency, the null-model rule of the invariants; the leaf fetch and the rebuild below | index size at scale; whether the index fits in memory at scale; k; query count; leaf size distribution; growth rate and merge policy | where the leaves live and under what protocol; full or incremental rebuild and its period |
| KV cache (`lmcache-local-disk`, `lmcache-shared-fs`) | nothing: the committed captures hold server and replay logs, no strace (the fitter reads logs) | an strace would show hex-keyed files written once and read on a hit; the verdicts that matter are write-then-read lag and re-read share per key, which `consume`/`choose` does not express | requests at scale; pool size, which the WG fixes (§3.64) | hit rate at scale; prefix sharing |

**Vector search is three sub-workloads (the user's scope point).** The user's understanding,
confirmed with the confidence marked, since it is from this side's knowledge of the systems and
not from reading their source this session, and the Milvus facts are to be checked against its
documentation before any catalog record states them: an ANN query returns the *identities* of
the k nearest neighbors, and the application needs the neighbors themselves, which are not in
the index. The pure libraries, DiskANN, FAISS, hnswlib, ScaNN, return ids and distances only
(high confidence); the managed databases, Milvus, Qdrant, Weaviate, pgvector, Pinecone, store a
payload beside each vector and return it in the same query (high), but the payload is never
inside the graph or IVF structure: Qdrant keeps it in a separate store on disk beside the
index; Milvus keeps scalar fields in per-segment binlog files on its object store, pulled to
local disk and memory by the query nodes (medium; confirm). So the *leaf fetch* is I/O distinct
from the index traversal in every case, and in the WG's own Milvus harness it is inside Milvus's
query path on local disk and S3 rather than a user-visible lookup. In text RAG the leaf is a
chunk of a few hundred to a thousand tokens, one to four kilobytes (high), small beside a disk
index's traversal of about a hundred 4 KiB reads per query; in multimodal retrieval, images,
video, DICOM series, one leaf is megabytes and the traversal is noise; and where a production
deployment holds an HNSW index entirely in memory, storage sees nothing of the traversal and
the leaf fetch plus periodic persistence is the whole storage workload. The leaf fetch is not an
addition to the vector-search workload; for in-memory indexes it *is* the workload.

Rebuilds, the user's second scope point, are partially true as suspected: the static libraries
need a rebuild to add content, classic DiskANN, FAISS IVF-PQ when the centroids drift, ScaNN
(high); HNSW inserts incrementally by design with deletes as tombstones and an occasional
rebuild to clean them (high); FreshDiskANN adds streaming inserts and deletes to DiskANN with a
periodic background merge that rewrites the disk index, a large read-everything-write-everything
event (medium); Milvus never rebuilds the collection: new vectors go to growing segments, sealed
segments get their index built asynchronously, and compaction merges segments, so its background
load is continuous rather than periodic (medium). The lifecycle is the same in every case, only
the period and the grain differ, whole index or one segment at a time; and ingest adds a
dataprep read pass over the new source media to embed it before the index work starts. The
user's framing is adopted: as checkpointing is a sub-workload training invokes periodically to
protect progress, an index rebuild (or merge, or compaction) is a sub-workload lookups invoke
periodically to incorporate content, and ingest precedes it.

Consequences: the language already has the parts. The leaf store is an `indexed` dataset,
looked up by id and never consumed, which contract 0.8 (§3.72) created for exactly this, with
its protocol declared per dataset so leaves on S3 use the object engine with no new grammar;
the query phase draws k ids per query and reads them with the leaf size a positional
distribution; the rebuild is a phase invoked every N queries or by elapsed time as checkpointing
sits inside training; the ingest read pass precedes it. The catalog gains mechanisms named for
the I/O: a merge or compaction mechanism, and search cases composing a search mechanism with a
leaf dataset; item 45(d)'s Milvus trace becomes more valuable since one strace of Milvus under
query load shows the traversal, the payload fetch from the segment cache, and the compaction
writes together. Whether the leaf fetch and the rebuild are inside MLPerf Storage's vector-search
benchmark is the WG's scope call (the Milvus harness as it stands measures queries only); aeiou
models the whole lifecycle because it is general, and cases can mark which parts a division
scores; this goes into the brief's §8 when the WG decides. The object-store gap stays real: no
strace for leaves on S3, and the trace survey found no public object trace, so a capture there
needs a request log, below.

**Three site kinds, not four types.** With vector search decomposed, the digest's site model
needs three kinds and every type is covered by them: a *keyed lookup* site, hex or integer ids
read by id with a lag after their write, serving the KV cache and the leaf fetch alike; a
*within-one-open* site, the index traversal, with verdicts on offset selection (reuse, runs,
skew, dependency); a *bulk write* site, checkpoint save, index build, merge. The template site
of §3.87 is the read side of a sample set and stays. Read sites and write sites get different
question wording. The keyed kind comes first since two types share it.

**Two pathways, one set of tools.** The user recalled that a customer may refuse an LLM near its
traces, the reason `aeiou-trace` is deterministic rather than a prompt (§3.86), and asked whether
needing both a skill path and a no-LLM path doubles the work or halves the skill's value. Neither.
The skill owns no mechanics: everything it would do is a command, digest, answer the questions,
pick or write a mechanism, `aeiou-fit`, `aeiou-catalog new`, `compare --judge`. The no-LLM path
is those commands plus one written procedure page, which the skill follows too, so the page is
written once. What the no-LLM path cannot do is write the builder script for a mechanism the
catalog lacks; a person does that in Python with the digest in hand, which is how every
mechanism so far was written. The skill's value is judgment, which mechanism matches, what a
failed judge row means, authoring a new mechanism, not plumbing; and keeping the tools complete
makes the skill checkable, since every step it takes is a command whose output item 48's
evaluation harness can compare.

**Application records (decided with the user).** A person may know the product they use and not
its storage design. The user proposed asking "you appear to be using a vector database, which
one?" from a list and then only the product's storage-relevant options, Milvus answering most
questions itself. Assessed as modest code and real curation, and adopted as *data, not code*:
adding a product or a version is a new file. Layout, the user's: a top-level `applications/`
beside `workloads/`, reference data about traced applications rather than workloads, visible in
the repository because that is where it is edited, and shipped in the wheel (hatch
`force-include` into `aeiou/applications`, read through `importlib.resources`) so the digest at a
customer site has it without a checkout; `--applications DIR` points at a checkout or extra
records as `--workloads` does for cases. One directory per application, named as a user picks
it from a list, lowercase and hyphenated (`milvus`, `qdrant`, `weaviate`, `opensearch`,
`pgvector`, `diskann`, `faiss`, `pytorch-dcp`, `lmcache`, `safetensors`); inside, mirroring
`mechanism.toml` and `case.toml`: `application.toml`, what does not change across versions, the
display name, the kind (library, self-hosted service, managed service), the path signatures that
identify it in a trace, the catalog mechanisms it maps to, home page and source repository; and
`<version>.toml` per version range the facts were read for, named by the lowest version it
covers (`2.5.toml`, `2.6.toml`), stating the range it claims, the structures offered with the
default marked, the options that change the storage workload and the answer each implies, the
mmap and blind-spot warnings, and per fact its source URL, the version read, and the date read.
`applications/README.md` documents the format, `applications/attributes.toml` the shared
vocabulary for kinds, structures, and option names, and the test suite walks every record
against it as the trace tests walk the captures.

Three rules keep it safe. (1) *The trace always wins*: a product default fills an undecided
verdict or an at-scale fact, never overrides what the trace shows, and the consistency check
that rejects a wrong human answer rejects a wrong default; Milvus's segment layout is in the
trace already, the record is for what a short trace cannot show. (2) *Defaults are weaker than
answers*: a fact from a record carries its own provenance ("product default, Milvus 2.5 docs,
dated") and ranks below the owner's statement, so a reviewer sees which verdicts rest on
documentation. (3) *Records are dated and versioned*: the digest asks the version once it has
confirmed the application, picks the record whose range contains it, and otherwise takes the
nearest lower record with every fact downgraded to a suggestion the person confirms; a record
never claims a version it was not read for.

Why it is worth more than it looks: signatures work across all four types (DCP's `.distcp` and
`.metadata`, LMCache's directory, safetensors, DiskANN's `_disk.index` and `_pq_pivots.bin`,
Qdrant's `payload_storage` and `vector_storage`, Lucene segment files under OpenSearch), so
recognition is the first question for every trace and the bridge the no-LLM path needs ("this
trace matches mechanism X; its cases are these"); the record knows when strace is blind (Milvus
and Qdrant can serve vectors through mmap, where reads become page faults the trace never shows,
and a record saying "mmap on by default for this index type" lets the digest warn that the
capture undercounts, which no trace reading can discover); and it names the managed case
(Pinecone, Zilliz Cloud, Atlas: the index I/O is on the vendor's storage, the measurable
workload is the leaf store and ingest). Caveats: coverage is long-tailed and FAISS, the most
common library, has no signature, so a "none of these" path always falls through to the plain
questions; the question structure stays fixed, identify, confirm, then the open list, with the
record only removing items from it; and the records are a maintenance commitment someone owns,
a project decision rather than a tool one.

**Object-store request logs (the user: "we will need to suggest and/or provide operation
tracing techniques for object stores").** In order: (1) define the log `aeiou-trace` reads, one
line per request with wall-clock time, thread, operation, bucket, key, byte range, bytes,
status, and duration, keeping range requests and SDK retries as issued since the abstract is
the application's request stream, with wall-clock time so a mixed POSIX-and-object application
(Milvus) merges its two logs on a clock as the strace timestamps allow; (2) readers for the
server-side logs that exist: MinIO's audit webhook, complete and real-time and what Milvus
standalone stores segments on, so item 45(d) needs it anyway; Ceph RGW's ops log; S3 server
access logs with the caveat that AWS delivers them best-effort and late, so CloudTrail data
events are the complete source there; (3) for a lab with no server log, a small logging proxy
the application's endpoint variable points at, since strace of a TLS client sees ciphertext;
client-side SDK logging as a verbose, library-specific fallback. Item 49 states the reader as a
gap until built.

**Build order, each its own commit with `./dev.sh check all` and coverage:** (1) this record
and the amendments to §3.87, brief items 45, 47, 48, 49, README's Status, and REFERENCE §7;
(2) the question loop in the digest, tests first: the generic fact questions, the behavioral
questions in rank order, the consistency check, `asserted` with provenance, the answers file
beside the document, `--answers FILE`, the no-questions flag, `-` refused, man page and
REFERENCE §7; (3) `applications/`: format, loader, signature matcher, confirm question, product
defaults as weaker assertions, first records for the applications the kits trace (DCP, LMCache,
safetensors, DiskANN) and Milvus and Qdrant from their documentation with sources and versions;
(4) the keyed site kind, then within-one-open; (5) the object log format and the MinIO audit
reader. Real traces for checkpointing and vector search would make (2) through (4) honest; the
kits produce them but need libraries this box lacks, so where to run them is the user's call.

*Built 2026-10-29, step (2):* `analyze` and `document` split the digest so answers apply without
re-reading the trace; `interview` over a console object (`say`, `ask`; end of input stops the
asking); `fact_check` and the candidate check, with `evidence` naming what decided a verdict a
replayed answer contradicts; `asserted` and `rejected` in the document and `asserted` beside a
verdict that stays `undecided`; `--answers FILE`, `--no-questions`, `--answered-by NAME`, the
answers file written beside `-o`'s document with the replay command printed, `-` refused, the
trace's SHA-256 in both files. The facts in this step are per site, the files of its kind at
scale and the passes the benchmark runs, the two `--candidates` and `--epochs` state; the
field-level facts (ranks, steps, keys) come with the site kinds of step (4), whose questions
need them. Seventeen tests, the module at 100 % line coverage; man page and REFERENCE §7.

*Built 2026-10-29, step (3):* `applications/` as decided, with `attributes.toml` (kinds,
confidence, blind spots), the README, and seven records: `pytorch-dcp/2.14` (from the kits and
the DCP documentation; options `single_file_per_rank`, `sync_files`; asserts size per path for
the shards), `lmcache/0.5` (from the 0.5.5 source: files named `model@world_size@worker_id@
chunk_hash_hex@dtype.pt`; options `use_odirect`, `local_cpu`), `safetensors/0.8` (blind spot
mmap), `diskann/0.7` (option `num_nodes_to_cache`), `faiss/1.15` (no signature; blind spot
mmap), `milvus/3.0` (docs v3.0.x and the master `milvus.yaml`, read today: blind spot object;
options index, mmap of the vector field, whose default the configuration file gives as on and
the mmap page as off, taken as on at medium confidence, mmap of the index, tiered-storage
eviction), `qdrant/1.19` (the storage concepts page and the 1.19 and 1.20 release notes:
vectors always memory-mapped, payload through io_uring, as blind spots; options the three
memory tiers). `aeiou/applications.py` loads, validates, identifies, resolves, and turns a
record into the digest's answers with the record as `by`; the interview asks the application
first, then the version, then the options, says the blind spots, applies an exact record's
assertions or offers a lower record's for confirmation; `--application NAME[@VERSION]`,
`--applications DIR`; the document carries `application`, the answers file too, so a replay
needs no records; the wheel ships the directory. Thirty-four tests of the records and nine of
the interview with them; both modules at 98–100 % line coverage. Milvus's layout signatures
(`insert_log/`, `index_files/`) and Qdrant's directory signatures are at medium confidence
until a trace of each confirms them, which item 45(d) will.

*Built 2026-10-29, step (4), the three site kinds, with the verdicts chosen here and open to
the user's review:* `kind_of` sorts every template: a *set* (opened twice or more, read; the
verdicts of §3.87), *keyed* (~~opened twice or more with a hex field or a path both written and
read; the KV cache and the leaf fetch~~ writes and reads interleaving, the KV cache, revised
2026-10-30, point 1 below), *bulk write* (opened twice or more, only written; the
checkpoint and the index build), *within* (one open read at eight or more offsets in more than
one run; the index traversal). Keyed: keys, writes, reads after a write and before any, re-reads
and the top tenth's share, the lag from write to read in time and in touches; `origin`
(`traced` when every key read was written in the trace, else undecided between a warm cache
from an earlier run and another writer the trace does not cover, settled by a trace from the
start or of every writer), `reuse` (`repeated` when a key is read again, else undecided, settled
by a longer trace), `size`. Bulk write: every int field of the name `varies`, ~~or `constant` once
a path is written again, or undecided between a new value per write and a rewrite, settled by a
second write (this is where ranks and steps become the field-level facts: `values:<field>` at
scale for a varying field, and the question `field:<field>` for a constant one)~~ (revised
2026-10-30, point 2 below); `rewrite`, `sync` (the share of opens with an fsync), `size`. Within: reads, distinct offsets, the share
revisiting one, the span, sequential runs; `skew` judged against a uniform spread's top-tenth
share, 0.1 (1 + 1.75 / sqrt(reads per offset)) from the Poisson counts a uniform draw gives,
`skewed` above 1.5 of it, `uniform` below 1.2, undecided between or under fifty reads; and
`memory` and `dependency`, which no trace decides, asked always with what would settle them (the
file's size against the hosts' memory; the application's source, or the same query traced
twice, since a dependent walk repeats its offsets exactly). Each kind's facts: keys and
requests; values per field and the writes the benchmark makes; the file's size and the queries.
Divergence at scale is a formula only for a set's selection; the other items rank by order with
what differs at scale said in words. The records' vocabulary follows (`values:<field>`~~,
`field:<field>`~~). Thirteen tests over constructed traces (`calls_strace`), the module at 100 %.
Not done here: the leaf fetch as a keyed site is read from an strace of the application that
reads the leaves, which exists for no vector-search kit yet; the keyed verdicts were checked
against the LMCache source's behavior (write once, read on a hit, never rewritten), not a trace.

*Built 2026-10-29, step (5):* `aeiou/objectlog.py`. Rather than a format of its own, an object
log is read into the exporter's event stream: each request becomes the calls the runner's object
engine issues for it (§16: a GET an open, a read at the Range's offset, a close; a PUT an open
that creates, a write, a close; HEAD a stat, DELETE an unlink, a listing a readdir of the prefix,
a multipart upload an open at its start, a write per part, a close at completion), on a lane per
client host, under `--root BUCKET[/PREFIX]`, so `metrics`, `export`, `digest`, and the `trace`
node take it unchanged and the digest's site kinds apply to it. MinIO's audit log (the `api`
object and the `Range` header; its `time` taken as the response's end, the request starting
`timeToResponse` earlier, noted) and S3's server access log (to the second, no Range, so a
partial GET is read from 0 and counted). Read from AWS's log-format page today; MinIO's entry
format from its documentation's example and field list (its source could not be fetched from
here), at medium confidence until a real audit log is read. Ceph RGW's ops log and a proxy's
log stay as item 49's gap, as does the merge of an object log with an strace on the clock. Six
tests over constructed logs of both formats.

*Revised with the user 2026-10-30 (the step-4 verdicts reviewed; ~~recorded, not yet built, so
the step-4 note above still describes the code; its superseded parts are struck when the digest
changes, in Python or in the Rust of §3.89 stage 2, whichever comes first~~ built in the Python
digest 2026-10-31, the note below; the step-4 note's superseded parts struck):*

1. **Keyed is the access pattern, not the name.** A keyed store (the KV cache) is written and
   then read at least once within one phase of a run; a KV cache has no phases. A leaf fetch in
   a vector database is not keyed: its build phase writes the leaves and never reads them, its
   search phase reads them and never writes them. A template is keyed only when its writes and
   reads interleave (a read of one key before a later write); all writes before all reads is
   two phases, a bulk write and a reader; a hex field in the name is supporting evidence that
   raises confidence in a pattern that already matches, never sufficient alone; a read-only
   template with hex or id names is a sample set, noted as a store written elsewhere fits it. A
   trace is one phase unless its owner says otherwise (phase detection is brief item 51).
2. **A checkpoint is never overwritten.** The prior checkpoint stays intact until the next is
   fully written and stable, then may be deleted under the application's retention policy. So
   the step-4 rule `constant` *once a path is written again* goes, and with it the question of
   §3.88's table, row "checkpointing", *whether names are new per step or overwritten*: a
   rewritten path is evidence against a checkpoint. A field holding one value over the trace's
   saves is undecided until a second save and is asked as `values:<field>` (how many at scale:
   the ranks). New verdict `retention`: `kept` (nothing deleted), `keeps N`, or *unsafe* (an
   earlier save deleted before the next was stable, which contradicts what a checkpoint is
   for), with the number kept also asked as a fact, a policy.
3. **A checkpoint is read only after a failure.** So a read of a template's path after the
   template's first write means it is not a checkpoint. A read of the checkpoint pattern at or
   near the start of a trace, of paths this trace does not write (an earlier step), is a
   restore: a load site (`dcp-load`) before the save site (`dcp-save`), then a repeating
   pattern of other I/O and a checkpoint write. A failure and the restart after it are always
   two traces. Asynchronous or tiered checkpointing shows only the save to the local tier in
   the traced process; the drain to shared storage runs elsewhere and is not in the trace.
4. **A distributed checkpoint is one burst.** Each rank (process or node) writes its own files,
   all starting simultaneously and finishing nearly so, since a checkpoint is the application's
   instantaneous state. The save is recognised as a burst: one new file per lane, the opens
   within a window small against the burst's length, quiet between bursts; the bursts' cadence
   is the checkpoint interval as a measured fact. Simultaneous start is the strong test;
   completion depends on per-rank size and on the storage under test, so it only supports.
   Plain DDP writes from rank 0 alone, which the trace cannot tell from the other ranks
   blocking on it with any reliability (the user's point): one writing lane is accepted, and
   the writer count at scale asked.
5. **Commit markers and framework names belong in the application records.** PyTorch DCP's
   coordinator alone writes `.metadata` after every rank's `__<rank>_<n>.distcp`; no rank
   writes an equivalent, and plain DDP's `torch.save` writes neither. Several frameworks
   overwrite a small pointer after a save completes (DeepSpeed's `latest`, Megatron's
   `latest_checkpointed_iteration.txt`), the one rewritten path a checkpoint has, and a
   commit marker rather than checkpoint data (from memory, medium confidence: each to be read
   from the framework's source before its record is written). Such names are evidence per
   application, so they go in `applications/` records (DeepSpeed, Megatron-LM, Hugging Face
   Trainer, Lightning beside pytorch-dcp) as assertions, which the trace still checks, and the
   owner confirms the label when neither decides.
6. **Phase detection and all-rank traces are their own items**, queued at the user's request:
   brief items 51 and 52.

*Built 2026-10-31, points 1 to 4 in the Python digest (test first; Rust is §3.89 stage 2):* a
site is now a template or, where a template is two phases, the opens of one (`phase` `write`
or `read`, shown `0/write`), and an answer may name the phase; without one it answers every
site of the template that has its fact or question, so a record's assertion needs no phase.
`parts_of` sorts a template: reads and writes interleaving (a read before a later write, or an
open both read and written) is `keyed`, its hex name recorded as `hex_name`, support only;
every write before every read is a `bulk_write` part and a `set` part, the bulk write's label
refuted as a checkpoint ("read after its first write"); every read before the first write, of
paths never written, is a new kind, `restore` (files, bytes, `size`; no facts), then the bulk
write, which lists the restore among its supports; read-only hex names are a set, noted. Bulk
write: the saves are bursts (`saves_of`: an open begins the next once it starts later after
the burst than the burst has lasted), `simultaneous` when each save's lanes open their first
files within a quarter of its length (`SIMULTANEOUS`, chosen here: a start window small
against the save, which the user's "nearly simultaneous" leaves to us), `one_writer` (one lane:
DDP accepted, the fact `writers` asked), `staggered` (a save whose lanes start far apart), or
`mixed`, with the interval between saves' starts as the cadence; a field is `varies`,
`constant` once two saves keep it, else undecided until a second save and asked as
`values:<field>` (the `field:<field>` question and its candidates gone, from the records'
vocabulary too); `rewrite` notes that a rewritten path is evidence against a checkpoint;
`retention` from the template's successful unlinks (a new `removes` list per template),
`kept`, `keeps` N (the most saves held just after a deletion, every deletion after a later
save was stable), or `unsafe` (a save, or a path the trace never wrote, deleted before any
later save was stable), with the fact `keep` checked against it; and a new question, `label`,
`checkpoint` or `output`, decided `output` by a rewrite or a read after the first write, never
`checkpoint` by a trace, with the evidence that `supports` a checkpoint listed (a simultaneous
start, one writing lane, every file fsynced, a restore, a safe retention). The
pytorch-dcp 2.14 record asserts `label` `checkpoint` for its shards, from the DCP documentation
it already cites. Point 5's records (DeepSpeed, Megatron-LM, Hugging Face Trainer, Lightning)
wait for their sources to be read. Not built: a burst's completion spread, which point 4 says
only supports~~; a "near the start" test for a restore, read as "before the first write"~~ (built
2026-10-31, below).
Eleven tests changed or added in `test_digest.py`, the record test in `test_applications.py`.

*Revised with the user 2026-10-31, "near the start":* the user agreed the quarter-length
window for simultaneous starts and bounded the restore: its reads start within the first
quarter of the trace (from its first call, of any lane, to its last). The user's account of a
restart: the application does some I/O just to start, then reads the model into the GPUs, then
reads the training parameters and the optimizer state; the parameters and the optimizer state
are the only part of that which is the checkpoint, and after the restore the job goes on with
the stereotypical training I/O, new checkpoints written periodically among it. So the start-up
I/O and the model read are other templates and other sites, and a restore is the checkpoint's
own pattern read early. Reads of a never-written pattern that start later, still before its
first write, are a reader beside the bulk write, neither a restore nor support for a
checkpoint, nor evidence against one. Built the same day, test first
(`test_reads_before_the_first_write_of_paths_never_written_are_a_restore`: a restore at 2 s of
a 10 s trace, a reader at 6 s).

### 3.89 One observation document, the digest-to-digest comparison, and `aeiou-trace` in Rust (designed with the user 2026-10-29)

The comparison item 47 left open, designed in a conversation that widened it twice. Recorded
before any of it is built; the build order is at the end.

**The spike.** `aeiou run` of `npy_per_sample` under `strace`, at the shape of the npy kit's
trace of §3.87 (24 files of 1 MiB, two workers, two passes), digested beside the kit's digest.
The site verdicts agree (`consume`, `permuted`, `per-path` over passes [24, 24]). Two real
differences show: the application lists `train` and then `train/00000`, the run only the leaf;
and the read motif differs, the application's `(read lseek read)+` group absent from the run,
684 reads against 268 (possibly the spike's `xfer` guess, below). Two differences are not
workload: the application re-forks its workers each pass (four lanes of twelve units, two at a
time) where the runner keeps two threads of 24, and the runner reads its `.aeiou-dataset.json`
and stats the root. A bug on the way: `npy_per_sample` divides by zero on storage whose
`st_blksize` is 4096, ext4's default, since `xfer` is the block size and the fill count is
`xfer div np_chunk`; the spike ran with `--param xfer=1048576`. ~~Not yet fixed.~~ *Fixed
2026-10-30:* the read loop as `np.load` issues it under a 4096-byte buffer, from an strace of
three archives (the mechanism's README): per NumPy chunk, a read of the whole blocks past the
buffer (the first 2 blocks short of a chunk, the rest 1, the last stopping at the block before
the zip tail and skipped when that is none), a refill of one block, and a tell; the header's two
fills and four tells first, so the tells stay `chunks + 4`. Under `when(xfer ≥ np_chunk)`, the
1 MiB loop is unchanged and so are its fingerprints; a buffer strictly between 4 KiB and 256 KiB
is not traced.

**Four documents, two sides (the user's question).** A trace yields a digest (structure) and a
metrics document (numbers); a model is an abstract (structure) and a parameters file (numbers).
The observation side describes one trace that happened, finite and with what it cannot decide
left open, the owner's answers asserted with provenance; the model side generates traces at any
scale and must commit to every choice. A metrics document fills no parameter: it is the
measured effect a parameter is fitted to (`sample_mean` a cause, the request-size histogram one
of its effects). The abstract and parameters split stays (several cases share an abstract). The
digest and the abstract stay apart: their overlap is a path template against a dataset's
pattern and a motif against a loop body, and removing it would need an abstract that carries
uncertainty, which the runner cannot execute, or a digest that executes, which is the `trace`
node and does not scale. The digest and metrics split goes, below. A fit is checked on the
observation side, since the application has no abstract: both sides reduced to the same
observation document and compared there.

**One observation document.** `aeiou_digest: 2`: the structure of version 1 (templates, lanes
and phases, sites and verdicts, open items, assertions) and a `metrics` section, the numbers of
today's metrics document over the same events. `aeiou-trace metrics` is retired with no alias;
`aeiou dry-run --metrics-json` writes the same document. This side first argued for keeping the
metrics document for full-scale dry runs, on the ground that the digest was built for
trace-sized inputs; the user pointed out that the SNIA AIWD TWG's real traces run 3.5 to 7 GiB,
which removes the distinction, and the measurement agrees. Synthetic straces of npz-shaped
reads (ten lines per open, four threads, two shuffled passes), under the Python tools of today:

| trace | lines | digest | metrics |
|---|---|---|---|
| 53 MB | 0.5 M | 11 s, 0.37 GB | 9 s, 0.16 GB |
| 211 MB | 2 M | 51 s, 1.5 GB | 36 s, 0.6 GB |
| 7 GiB, extrapolated | ~71 M | ~30 min, ~52 GB | ~21 min, ~22 GB |

Both linear in time and memory; on the 211 MB trace the strace parser takes 50 s and 1.5 GB of
the digest's 51 s (the digest of the exported file takes 11 s), and holds every event. So both
halves of the document must scale, and the work is the parser first. The analysis streams:
an open is compressed to its motif at its close and its ops dropped; a site keeps one integer
per open (the path's index) for its pass and order verdicts; fields and keyed sites keep state
per distinct path, each path stored once; a within-one-open site's offsets go to a bounded
histogram. Memory follows the opens and distinct paths, never the ops, and a test holds it.

**The comparison.** `aeiou-trace compare A B` takes two observation documents and judges both
halves: structure by equality, numbers by today's tolerance classes (`--judge`, `--tolerances`,
`--self`); a side without structure (a version-1 metrics document converted, below) is judged
on numbers alone. It works on documents, never traces, since the application's side is often a
digest the owner released while keeping the trace. Templates pair by identical text, then by
shape (the same fields with the same kinds and widths, the same site kind), then by `--pair
A=B`; an unpaired application template is a difference, an unpaired run template is listed.
Per pair: fields, site kind, each verdict (one side `undecided` is not compared; a decided
verdict against the owner's asserted answer is compared and labelled an assertion), the set of
op kinds, the set of motifs (a motif that depends on file size is structure the abstract must
reproduce, so the run takes the trace's size distribution), and the peak of lanes concurrently
on the template; lane counts and the process model are shown, not judged. The runner's own
I/O is set aside on the run side and noted: names beginning `.aeiou` (V13 keeps them out of
every workload) and the root's stat. Exit 0, 1 on any difference, 2 for usage.

**Two sources for the run side.** `aeiou dry-run --emit-trace FILE` (item 49) writes the op
stream as an `aeiou_trace` file, digested like any trace: no strace, none of the runner's own
I/O, no dataset on disk; the default when fitting. The strace of a real `aeiou run` stays as
the check that a backend issues exactly the abstract's calls (the interposition test).

**Rust, keeping the name.** The trace-side tools move to Rust as the binary `aeiou-trace`, built
from the runner's crate, which already reads trace files (`trace.rs`) and computes the metrics
(`metrics.rs`). The reasons: the parser is where the time goes; bounded memory is easier to
hold; the metrics get one implementation, the Python one and its row-for-row parity tests going;
the trace's owner runs one binary from the release tarball, either architecture, where the
trace lives, with no Python environment; and the user's rule of Rust for throughput-bound tools
and Python only where a real library makes the bytes. The costs: about 1,100 lines of parser
and exporter, 1,150 of analysis, 250 of object-log reader, slower iteration while the keyed,
bulk-write, and within-one-open verdicts (§3.88, step 4) await the user's review, and a document
shape still changing. What lowers the risk: the Python tests feed strace text and check JSON,
so they drive the Rust binary and are the port's acceptance tests. The name `aeiou-trace`, its
man page, and its subcommands stay (the user's choice over `aeiou trace`).

**Build order.** (1) The strace parser and exporter in Rust, `aeiou-trace export` and the
metrics over the exported events, held to the existing tests; (2) the analysis and the
version-2 document, streaming, after the user reviews the step-4 verdicts; (3) the interview,
the application records, and the answers file; (4) the two-sided `compare` and `dry-run
--emit-trace`; (5) the Python versions retired with no aliases, the 28 catalog shapes
regenerated and the 18 committed capture metrics rewrapped as numbers-only documents (their
traces were not retained, item 34), each saying so.

**Stage 1 built (2026-10-29).** `aeiou-trace` is a binary of the runner's crate
(`src/bin/aeiou_trace.rs`), its `metrics` and `export` Rust: `strace.rs` (the line grammar, the
descriptor tables through `clone`, `dup`, and an unfinished `close`, positions and `O_APPEND`
ends, instances, AIO rounds), `export.rs` (the trace file in two passes, the first learning the
shared opens, the pinned ones, the earliest start, the creates, and every AIO member's result,
so the second writes in order with the header first; and the metrics of an exported file),
`objectlog.rs`, and `metrics.rs`'s `TraceMetrics` (a context per thread or lane, instances
parked and swapped in as their calls interleave, which the dry run never needs). The Python
parser, metrics, exporter, and object-log reader were deleted; before that, every input their
tests fed them was recorded with what they made of it (`runner/aeiou/tests/strace/`, 144 cases
with five written to reach the rare calls), and the port reproduces every one exactly
(`tests/strace_corpus.rs`), the documents compared whole. The Python tests drive the binary
now; the seven that tested the Python's internals (stack and extent distances, byte
popularity, argument splitting, escaped descriptor paths, `CLONE_FILES`, the exporter's root)
gave their assertions to the Rust unit tests that cover the same code. `compare` and `digest`
stay in the Python package, which the binary runs (`python3 -m aeiou.trace`, or
`$AEIOU_PYTHON`), and the digest takes a trace's calls from `aeiou-trace export`. On the
synthetic strace of 211 MB, the metrics take 2.8 s and 0.15 GB (the Python 36 s and 0.61 GB)
and the export 3.2 s and 0.06 GB for both passes (the Python 50 s and 1.46 GB), the outputs
equal. The binary ships in the release tarball beside `aeiou`. ~~Next, stage 2, waits on the
user's review of the step-4 verdicts of §3.88.~~ *The verdicts were reviewed 2026-10-30 and built
in the Python 2026-10-31 (§3.88); stage 2 followed the same day, below.*

**Stage 2 built (2026-10-31).** The digest is Rust, `src/digest.rs`, and `aeiou-trace digest`
writes the document and the report. The cut, chosen while building: the analysis, the document,
an answers file applied, and the text report are the port; the interview, the application
records, and `--application` stay Python until stage 3, since the records' loader, their
signatures, and the console are its work. The binary decides which: a digest that would ask (a
terminal, no `--answers` file there, no `--no-questions`) or that names the application
(`--application`, `--applications`) goes to `python3 -m aeiou.trace`, which asks and gets every
document it shows or writes from the binary, the answers so far given as an answers file and
the trace read again each time (four reads at most, until stage 3 keeps the analysis in one
process). For the interview to check a fact without the analysis, each site carries its `facts`:
the name, the question, and, where the trace bounds it, the least value it allows and what it
shows (every bound `fact_check` applies is a least value: the distinct paths or keys, the full
passes or the saves, the saves kept, the bytes read, a field's values). The golden corpus is
stage 1's method again: every trace the Python digest's tests fed it, every document made from
one with the answers and options given, and every report, recorded before the Python was cut
(`tests/digest/`, 140 documents over 74 traces, 22 reports), and reproduced exactly; one
difference is chosen, a report lists a template's ops of equal count by name, since the order
they first appeared in is not in the document, a JSON object read back (serde_json's maps are
sorted, and `preserve_order` would change the bytes of every committed shape). Streaming as
designed above, with one refinement: a `within` site's offsets are kept exactly, as a count per
offset of the one open that could still be a template's only open or only reader, not in a
bounded histogram, so its verdicts stay the Python's; every other open's counts are dropped at
its close. The `metrics` section is the metrics of the exported calls (`export::FileMetrics`,
what `aeiou dry-run --metrics` of a `trace` node must equal) rather than of the strace
(`strace::metrics`, with its instances and `io_submit` chains): the same events the structure is
made from, and the same definitions the run side will have from `dry-run --emit-trace`. A line
the runner would refuse (an op on an open no line made) is noted in the structure, as before,
and counted out of the numbers (`lines_not_counted`). `aeiou-trace metrics` stays until stage
4, since `compare` still reads metrics documents. On a synthetic strace of 200 MB, 2 M lines
(npz-shaped, four threads, two shuffled passes): 7.8 s and 0.21 GB, metrics included, against
the Python digest's 14.5 s and 1.7 GB over the same Rust export, the documents equal; half the
time is the export's two passes, and of the rest the metrics and the structure take about half
each. The peak memory of 960,000 reads equals that of 3,200 over the same 200 opens within a
few MiB (`tests/trace_tool.rs`). ~~Next, stage 3: the interview, the records, and the answers
file in Rust.~~ Built the same day, below.

**Stage 3 built (2026-10-31).** The interview, the application records, and the answers file
are Rust (`src/interview.rs`, `src/applications.rs`), and `aeiou-trace digest` asks itself: the
hand-off to `python3 -m aeiou.trace` for a terminal, `--application`, or `--applications` is
gone, and with it the four reads of the trace an interview cost, every document it shows
remade from the one analysis. The golden corpus once more: every interview the Python's tests
ran, scripted and on the console, recorded before the Python was cut, with the lines typed, the
record tree and stated application, and what it said, what it asked, the answers, and the
document (`tests/digest/`, 29 interviews over 9 traces, 2 record trees), reproduced exactly by
`tests/digest_corpus.rs`, red against a stub and then green with no difference; the loader's
tests ported to `tests/applications.rs`; the command's on a pseudo-terminal (`tests/trace_tool.rs`).
Two choices made while building. The records are built into the binary (`build.rs` embeds the
checkout's `applications/`), not shipped beside it: the release tarball's `aeiou-trace` carries
them with no Python and no data directory, `--applications DIR` still reads another tree under
the same checks, and the wheel no longer carries them, since nothing in Python reads them now.
And one behaviour differs: on a terminal with an `--answers` path not yet there and nothing
answered, the Python's hand-back to the binary failed ("no such answers file, and no terminal to
ask on"); the binary writes the document and no answers file. A TOML error in a record is one
line with its line and column (`tomllib` and the `toml` crate word their messages differently;
the tests match the path and `TOML`). The vocabulary's order (kinds, confidences) is the file's,
read from the values' spans, as Python's dicts kept it. Python keeps `digest.py`'s handle for
its tests of the analysis until stage 5. ~~Next, stage 4: `compare` over two version-2
documents, and `dry-run --emit-trace`.~~ Built 2026-11-01, below.

**Stage 4 built (2026-11-01).** `aeiou-trace compare` is Rust (`src/compare.rs`), the last of the
tool that ran Python, and `aeiou dry-run --emit-trace` writes the run side (`src/emit.rs`). The
numbers first, by stage 1's method: every comparison the Python's tests made, recorded before the
Python was cut, with written cases for the rare refusals (`tests/compare/`: 75 row sets, 25
judgements, 25 tolerance files, 63 command lines with their files, output, and exit status, the
capture pairs among them), reproduced by `tests/compare_corpus.rs`, every string exactly. Five
differences are chosen: a file that is not JSON is refused in serde's words; a bound the parser
refuses (`--max-distance nan`) in the binary parser's; a total-variation distance is summed over
the values in order where the Python summed a set in hash order, which can differ in the last bit
(the corpus holds distances to 1e-12; one row of 75 sets differs, at 2e-16); a document of neither
kind is refused naming both, since digests are read now; an object log's metrics are described as
one, where the Python described them as a dry run's with every field `None` (a bug of the kind
R52 fixed for trace files).

The structure is as designed above, with these choices made while building. Pairing: identical
text, then `--pair` (applied before the shape, so a hand-made pair is never taken by a guess),
then shape, only where exactly one template of each side has it. A site pairs by phase; each
top-level verdict object of a site is compared, and a `fields` list's by field name. A side's
undecided verdict is replaced by its owner's asserted answer where there is one (not `unknown`),
and the aspect is labelled `assertion`; with neither, `not compared`. The peak of lanes is
computed from the phases on the template, each an interval `[first, last]` of its lane, a start
before an end at one instant (a phase of one call has its first equal to its last). B's own I/O
is set aside on B only: names with a component beginning `.aeiou`, and a root template whose ops
are only stats, since an application may stat its own root. A difference of structure exits 1
and, under `--judge`, makes the last line `not accepted: the structure differs (N); ...`, so a
capture's verdict cannot be accepted with a different shape. `--json` prints the whole
comparison (`aeiou_compare: 1`), which the Python tests now read their rows from.

`--emit-trace`: a lane is a context of the metrics walk (main line, `parallel` sub-actor, `loader`
worker, a `trace` node's lane), a `parallel` sub-actor keeping its lane across forks as a pool
thread does; the instances are walked in turn, one op each per round (`metrics::interleave`),
so they interleave as instances at once would, where the metrics walk takes each alone; the clock
is the line's ordinal (1 µs a line, `dur` 0, no compute). Two changes elsewhere make the file
exact. The VM's `lseek` now carries the position it sets in `Op.bytes` (not hashed: no
fingerprint moves), written as the line's `ret`, since the trace loader takes a sequential op's
position from the last `lseek`'s result; the first sweep without it put every read after an
`lseek` at 0. And the header names the dry run (`source: dry-run`, `abstract`, `sha256`, `seed`,
`gpus`, `fingerprint`, `order`) with `aeiou_trace` its first key, which the tools look for; the
digest keeps `dry-run` as its source and copies the rest into its `metrics`, so `--self` checks
an emitted digest as a dry run's metrics document. Checked on every example and unsettled case
at its first GPU count (all but three full-scale LMCache cases, which emit gigabytes): at one
instance the digest's numbers are the dry run's own except fan-out and depth (no forks in a
file) and, for `faiss_ivf_ondisk`, one run of 2,561 that the end of a `parallel` closes in the
walk and the same lane continues at the next fork; at two instances the reuse distances differ,
as the trace of two instances would. The emitted file loads as a `trace` node whose dry run
measures the same (`tests/emit.rs`).

The spike again, now with both tools: `npy_per_sample` run under `strace` against its emitted
digest. At the dry run's default `st_blksize` the two differ in request size and read count,
which is the parameter, not the tools: the run takes the storage's 4 KiB (ext4 here) and the
dry run its fallback, so a run side is emitted with the `--st-blksize` the run reports. At 4
KiB: every verdict the same, every row within tolerance but the reuse distance (the threads'
completion order against the walk's round-robin), and two differences of structure, both one
`stat` of a sample file, the runner's dataset check before the gate, which the strace of a run
shows as an op kind and a motif and the emitted file does not have. The set-aside found the
dataset manifest and the root's stat. Not done in stage 4, left for stage 5 or a decision:
`aeiou-trace metrics` is not retired (its strace-only options, `--instance-root`,
`--chain-gap-us`, `--metrics-block`, ~~have no place in the digest yet~~ *on the digest since
2026-11-02, below*, and the committed captures are metrics documents until stage 5 rewraps them), and `dry-run --metrics-json` still writes a
metrics document, which `compare` reads as numbers only. One test fixed on the way: stage 2's
memory test (`tests/trace_tool.rs`) read its digest child's `ru_maxrss`, which on Linux counts
the peak of the process the child was forked from; under coverage the test process, running
every test of the crate, reached 9.7 GB, and the new tests moved the schedule so that both
digests reported 2.8 GB of it. The child is now measured from a fresh process (the test binary
run again for a helper alone).

*The metrics options on the digest, 2026-11-02 (the user's instruction of 2026-11-01: before
`metrics` is retired).* Measured first: the digest's `metrics` were the exported lines'
(`export::FileMetrics`), which differ from `aeiou-trace metrics` of the same strace in 5 of the
27 golden straces taken without options: two libaio traces in read-after-read reuse (the strace
counts an `io_submit` member when it is reaped, the file in line order), one with a `truncate`
the export left out (`not_exported:truncate`, §3.92), two with a `close` the export does not
write, and the `depth_not_computed` note missing. And the file has neither a process tree nor a
round's end, so `--instance-root` and `--chain-gap-us` could not have worked through it. Decided
with the user: of an strace, the section is the strace's own reading, `aeiou-trace metrics` of
it, by a second tracer (its instances and notes must not change the export's) fed the lines of
the export's first pass; of an exported file or an object log (whose numbers were already equal,
three of three) the exported lines' as before. `--metrics-block` works on every input;
`--instance-root` and `--chain-gap-us` for a file or a log are usage errors, where `metrics`
ignores them. `tests/strace_corpus.rs` holds the section to every metrics document of the corpus.
The cost, on a synthetic strace of 570 MB (a million npz opens of four reads, eight threads): the
digest took 35.0 s and 1.47 GB; parsing every line twice on one thread, 48.3 s; with the metrics
tracer on its own thread, fed 1 MB batches, 40.2 s and 2.05 GB, the extra memory the metrics' own
(`metrics` alone, 8.8 s and 0.86 GB). The section changes for the five traces above, the goldens
of the digest corpus not at all (they set the section aside).

### 3.90 Two audiences, two tool suites, and the digest between them (the user's reasoning, 2026-10-31)

Raised while placing a trace's phases in the catalog (brief item 51), where the user asked for a
trace to be described in prose before any abstract is fitted to it, and explained why. Recorded
as the reason the trace-side tools of items 47 to 52 exist beside the runner, and as the test
each of them is held to.

**The user's account.** The library of abstracts serves a benchmark engineer, who picks which
workload to simulate and runs it. The trace work of the last items overlaps it as a Venn
diagram does: it serves a researcher who wants to understand an application and may have no
interest in running an abstract. A prose description of a trace, made from the trace, has value
apart from the runner; transforming a trace into a digest and back with little loss supports
insight into and characterization of the application; a 2 GB trace turned into a description
of a few hundred characters is data compression for humans. A digest turned into a fitted
abstract (an abstract and a parameter file) lets a benchmark engineer recreate the
application's I/O without the application. Whoever makes a digest gets part of the value,
perhaps all they want; turning it into an abstract gets more, for whoever wants it. And a digest
separates interests: an owner who will not publish a trace, since it holds private information
such as file names, may publish a digest, from which a benchmarking organization recreates the
I/O pattern of the now anonymous application.

**How the parts line up (this side's assessment, discussed with the user the same day).** The
digest is the shared document, and what is built already supports both audiences: the stage-4
comparison (§3.89) reads documents and never traces, so a fit is checked against a published
digest alone; version 2 carries the metrics of the same calls, under the runner's definitions,
beside the structure, so one document serves the prose and the fit; what the trace cannot
decide is marked open and the owner's answers travel in the document with their provenance; the
application records let an owner answer from a list. Five links are not yet aligned:

1. *The digest is not anonymous.* `trace`, `root`, `by`, the answers' notes, `--paths`, and the
   literals of every path template survive into it. The publication workflow rests on a
   redaction that keeps the structure a fit needs and lists what remains: brief item 53, queued.
2. *"Back" goes through the abstract.* A digest is lossy by design (templates, histograms, open
   questions); the way back is a fitted abstract and `dry-run --emit-trace`, a trace equivalent
   under the digest, never the original. "Little loss" has an operational definition: the
   stage-4 comparison of the emitted trace's digest with the original passes, structure equal
   and numbers within the tolerance classes. Stage 4 is what makes the round trip real.
3. *One vocabulary for both descriptions.* The catalog's words for a measured stream ("medium
   (16 MiB)", "read-only", "uniform (4.1%)") come from the thresholds in
   `workloads/attributes.toml`; a trace's prose (item 51, part 3) must use the same thresholds,
   so a researcher can set a trace's description beside a case's and compare them line by line.
   The binary does not carry that file today; it would be built in as the application records
   are (§3.89 stage 3).
4. *Prose does not go back to a digest.* The prose is for people; for them "back" is the digest.
   In the compression analogy, the digest is the compressed file and the prose its table of
   contents.
5. *Two meanings of "phase"*: the catalog's attribute (a system's stage of life) and a regime
   within one run (item 51, part 2, a rename).

Links beside these, not in the user's account: the data's content, which a trace never shows
(compressibility and dedupe ratios), so an owner who publishes no data would publish ratios
measured locally on a sample beside the digest (`aeiou-params` already reads real shards); the
`trace` node, a third route for whoever holds the trace and needs the traced scale exactly,
unpublishable since it holds the paths, and the ground truth for a fit at that scale (§3.85);
provenance, a case fitted from a published digest naming it by hash in its capture records, so
digest, case, lock, and results chain without the trace (stage 5, when the committed captures
are rewrapped, could make them digests); and items 52 and 49, all-rank traces and object-store
logs, both reaching both audiences through the one document.

### 3.91 A storage-reported value needs evidence that the application derives from it (debated with the user, decided 2026-11-01)

**The gap.** On §3.89's spike the run read `st_blksize` 4 KiB from ext4 and the dry run took the
1 MiB fallback, so the two issued different reads until the dry run was given `--st-blksize
4096`. Nothing on the trace side records the value the traced application saw: not the digest,
not a capture. The value is in the trace (`strace -v` prints `st_blksize=` in every `fstat`;
the spike's strace has 75, all 4096). The committed capture pairs compare at the fallback and
pass only because their traces were taken on NFS at `rsize` 1 MiB; a local-disk trace would be
held against the wrong op sequence (4 KiB reads against 1 MiB).

**The debate.** The user: by the interposition test `st_blksize` is the SUT's (a shim could
report any value to an unmodified application), so a run on storage reporting another value may
issue other reads, *if* the application derives its I/O size from it; but we will not read every
application's source, and letting anyone set the value to a convenient one is a risk. This side:
agreed, with the condition that the derivation is a claim about the application needing
evidence. One trace cannot separate "4 KiB because the file system said so" from "4 KiB hard-coded"
(4096 is the commonest hard-coded size), and the relation need not be identity (glibc stdio
`min(BUFSIZ, st_blksize)`, a [verify] claim; a recollection, low confidence, that recent CPython
changed `io`'s buffer sizing, which would make the rule version-dependent). Evidence, strongest
first: (1) interposition as an experiment, the application under a shim whose `fstat` reports
another value on the same storage, a second value giving the relation's form; (2) two traces on
storage reporting different values; (3) primary sources, tagged with confidence; (4) consistency in
one trace (the `fstat` before the reads, the read size equal to the value), which supports and
never settles.

**Decided (the user, 2026-11-01).** A parameter takes `{"from": "st_blksize"}` only with recorded
provenance of kind 1, 2, or 3; without it the parameter is fixed at the traced value, the same
for every SUT (strict sameness by default, fidelity when proven). The digest records the
`st_blksize` its trace's `fstat`/`stat` results show, per template, and treats the derivation as
an open question of the site (derived or fixed, undecided from one trace, settled by a shim run or
a second trace; an owner's answer recorded as an assertion with its source). A capture keeps the
observed value beside its fitted parameters, and a dry run of a capture or a fit takes it as
`--st-blksize`. `aeiou dry-run --emit-trace` records the resolved storage-reported values in its
header, and `aeiou-trace compare` flags (not refuses) a run side whose value differs from the
observed one: with a proven derivation the difference is legitimate and the report shows it; a
rules file may still lock the value. Every committed `st_blksize` parameter (npy, file-per-sample,
tfrecord, parquet, hdf5, dcp, LMCache shared-fs) is audited for its provenance, one resting only
on a matching trace fixed until a shim run confirms it; the shim and those runs are part of the
item. Brief item 55. Not started.

### 3.92 Nothing drops a truncate, and what frees space is replayed as the call it was (the user's reasoning, decided and built 2026-11-02)

Raised by the user while the digest took `metrics`'s options (§3.89, the 2026-11-02 note), whose
measurement found the export dropping `truncate(2)` (`not_exported:truncate`): *ensure that
truncate operations are not dropped, by anything.*

**The user's account.** Truncating a large file makes the filesystem take apart a significant
amount of the file's metadata and then recover all the bulk space the file occupied, which is
another, different metadata workload. Those are typically done asynchronously to foreground
operations, so they apply a headwind of background work that must be scheduled to finish soon.
And opening a file for read with `O_TRUNC` gives a completely different I/O pattern than opening
it without: the latter may read a GB, if the file is that big, while the former reads nothing.

**What this side found, every place a truncate or a freeing of space could be lost:**

- `truncate(2)` by path: dropped by the export, so by the `trace` node and the digest; counted as
  `ftruncate` in the metrics; no op in the abstract language, so no fit could say it.
- `fallocate`'s mode: the export kept the offset and length and dropped the mode, and the trace
  line and the runner had none, so a `FALLOC_FL_PUNCH_HOLE|FALLOC_FL_KEEP_SIZE`, which frees the
  range, would have been replayed as an allocation of it, the opposite workload; `KEEP_SIZE` as
  an extension, `COLLAPSE_RANGE`, `INSERT_RANGE`, `ZERO_RANGE` as something else. No committed
  trace or abstract has one.
- `O_TRUNC` (and `creat`): kept by the export, the trace node, and the runner (the flag reaches
  the kernel); the digest read an open's path and never its flags, so a template whose opens
  truncate looked like one whose opens read.
- `ftruncate`: kept everywhere.

**Decided with the user (the recommendation, then both of the further two):**

1. **`truncate` everywhere** (contract 0.14): a trace line `{"op":"truncate","path","len"}`, the
   runner's op kind `truncate` (numbered after every earlier kind, so no fingerprint moved),
   `Backend::truncate` (`truncate(2)`; inline under `io_uring`, which has no opcode for a path;
   under `mmap` this actor's mapping of the path stops at the new size, as at `ftruncate`), the
   abstract's `truncate {file, len, expect}` node, a changing op under V12, V14, and V18, the
   builder's `truncate(f, len)`, and its own kind in the metrics. The four golden cases of the
   strace corpus with a `truncate` were regenerated by the port, their difference checked to be
   that alone.
2. **`fallocate`'s mode end to end**: the trace line's and the abstract's `mode`, the flags by
   name, absent for mode 0 (which hashes as before: `aux` was 0); issued as written under every
   API (`io_uring`'s opcode takes it); a mode with a bit no flag names left out of an export and
   noted, never written as another; under `mmap`, the mapped size kept by `KEEP_SIZE` and
   `UNSHARE_RANGE`, asked again after `COLLAPSE_RANGE` and `INSERT_RANGE`. **Rule V29**: a mode
   the kernel takes, as `vfs_fallocate` refuses one with `EOPNOTSUPP`, checked against Linux's
   `fs/open.c` and `linux/falloc.h` the same day (the modes are exclusive; `PUNCH_HOLE` only with
   `KEEP_SIZE`; no `KEEP_SIZE` with `COLLAPSE_RANGE`, `INSERT_RANGE`, or `WRITE_ZEROES`); what a
   filesystem supports beyond that is the run's, as any failed op.
3. **The digest's `space`** per template, only where something happened: truncating opens, of them
   those on a path an earlier open had written (data discarded) and those on a path the trace had
   not seen (what it held is not known), `truncate` and `ftruncate` and those to 0, `fallocate` by
   mode, and the bytes freed by `PUNCH_HOLE` and removed by `COLLAPSE_RANGE`, with a line in the
   report. Motifs are unchanged (an open is `open` either way); the version stays 2, the record
   being additive, and the golden digests set it aside as they set aside `metrics` and `facts`.

Contract 0.14 carries both abstract changes in one bump; every committed AST was regenerated and
every case re-locked, the workload ids, fingerprints, and dataset ids unchanged, the shapes
measured again (their records name the abstract's hash). ~~Not done, for a decision: the digest
does not yet say how many bytes a truncate discarded (the size before it is known only for a
file the trace wrote or one an `fstat` showed), nor does `compare` hold two digests' `space` to
each other;~~ *Both built the same day, below.* The deferred reclaim itself is the storage's and
is not modelled beyond issuing the call, which is the interposition test's line (the call is the
application's, its cost the SUT's).

*The three left over, built 2026-11-02 at the user's request.* (1) **A call left out for want of
an open, now named.** Measuring for §3.89 had shown the export one `close` short in two
traces: ~~a fork's parent closing a descriptor only its child had opened~~ (*wrong, as the user
pointed out: a fork copies the descriptor table, so the parent's `close(3)` is of its own
descriptor 3, never the child's; both traces are synthetic, written by the Python's tests to
check that the table is copied, and the parent's descriptor 3 is one the trace never shows
opened*). The cause was wider:
every call on a descriptor whose open the trace does not show (held from before `strace`
attached) was left out of the export silently, counted in the metrics, the header saying only
`descriptors_without_open` once per descriptor. One golden case (`metrics-9f1a54ef`) exported
nothing of its 400 `pread64`s and said nothing of them. Such a call has no open id for the
`trace` node, so it is still left out, now noted by name (`not_exported:read:descriptor without
open`, `calls_not_exported`) at every site: data, `lseek`, `fstat`, the descriptor ops, `close`,
`mmap`, and an `io_submit` member; seven goldens regenerated, their difference notes alone.
~~Whether the export should instead give such a descriptor an open of its own, so that a
`trace` node replays its reads, is a decision not taken: the open was the application's, made
before the trace, with flags the trace does not show.~~ *Decided by the user the same day,
below.* (2) **The bytes a truncate discarded**:
the digest's `space` gains `truncated_bytes` and `truncates_of_unknown_size`; a size is known
once a truncate set it and is moved by every write and allocation after it and carried through
a `rename`, never guessed for a file the trace found already there (an `fstat`'s size is not in
an export). (3) **`compare` over `space`**: a pair's structure gains a `space` aspect where
either side has one, the kinds of it (truncating opens, of a path written before, `truncate`,
`ftruncate`, to length 0, `fallocate` by mode) and never the counts, which grow with the scale;
an abstract that writes where the application truncated is another workload, and a dry run's
emitted digest is now held to the trace's on it.

*An open synthesized for a descriptor the trace never shows opened (decided by the user and
built 2026-11-02).* The user: synthesize an open call rather than drop 400 reads, raise a loud
warning that the trace is incomplete and how (the missing open), and continue. And the cause:
a trace taken by attaching to a running job (`strace -p PID`) misses the opens made before it
attached, where one taken by starting the command under `strace` sees every open the command
makes; so the tracing instructions start the command under `strace` (brief item 54). This
side's addition: an open can still be missing from a trace started that way, for a descriptor
inherited from the shell or received over a socket (`SCM_RIGHTS`, seen only with `%network`).
Built: the exporter writes, at the descriptor's first use, `{"op":"open",…,"synthesized":true}`
with the flags its use needs (`RDONLY`, `WRONLY`, or `RDWR`, learned in the first pass; never
`CREAT` or `TRUNC`, the file being there), at position 0, as the reader already took it; the
header lists each (`missing_opens`: path, lane, the call that first used it) and counts them
(`opens_synthesized`); `aeiou-trace export`, `digest`, and `metrics` print a warning naming them
and saying to trace the command from its start, and `aeiou dry-run` and `run` of a `trace` node
print one too; the run issues the open, an open of an existing path (the path an input the
root must hold). A `close` of a descriptor the table does not have is given a descriptor of
its own, never another process's of the same path, so the fork golden keeps the parent's and
the child's apart and its metrics unchanged (they gain the `descriptors_without_open` note,
which they had lacked). Seven goldens regenerated, each checked to keep every call it had and to
add only calls on the paths its `missing_opens` names; `metrics-9f1a54ef` now exports its 400
reads. A synthesized open is an op the run issues and the application made before the trace,
so a trace node's fingerprint, its dry run's counts, and its digest's motifs include it, which
the warning says; no committed capture had a descriptor without an open.


### 3.93 Every filesystem call and flag, at every layer: the sweep and the user's decisions on it (2026-11-02)

**Why.** A `truncate`, an `O_TRUNC`, and a `close` had been ignored (§3.92); the user asked for
every operation and flag of the Linux filesystem APIs to be collected and every layer checked
against them, so the next such oversight is found now rather than later. The method, the
statuses, the confidence, and the 41 findings are the register `FS_API_SWEEP.md`, which the
fixes are worked from and which each fix updates.

**What it found, by type.** Calls dropped with no note where the man page promises they are
counted by name (S-1 to S-4); parsing that loses or corrupts calls (the `(deleted)` descriptor,
the `io_submit` parse, `openat2`, `RWF_APPEND`; S-5 to S-10); closes, implicit and duplicate and
inherited, and `close(2)`'s errors (S-11 to S-14); traced errors not reproduced (S-15, S-16);
bookkeeping blind to what an op means, the worst a create listed for a file that existed, whose
gate advice deletes the application's input (S-17 to S-28); the object-log reader (S-29 to
S-36); what `compare` never judges (S-37 to S-39); what the language cannot express (S-40); mmap
(S-41).

**Decided (the user, 2026-11-02).**
1. *New ops:* all of S-40, in the abstract and the trace file (contract 0.15). Brief item 58.
2. *mmap:* the page references that cause I/O are not in an strace; how to capture them (eBPF
   hooks in the kernel, brief item 54) is designed first, and S-41 stays an open work item.
3. *`compare`:* yes to all three: a kind present on one side only is a difference; open flags,
   advice, whence, and ioctl requests are compared; the documents' notes are shown in the
   verdict (S-37 to S-39).
4. *A close of a descriptor inherited over a fork* issues its own `close(2)` on replay (S-14).
5. *A traced error that does not recur on replay* is warned of loudly, not refused: an ENOSPC in
   the trace may well not reproduce, and the warning says the systems under test differ (S-15).
6. Group A (the bugs with no design question) is fixed first, test first, one commit per
   finding or tight group, each updating its status in the register. Brief item 57.

*Note (2026-11-02, later; the user's decision).* S-11 and S-12 turned out to hold a design
question, found while fixing them: the replay closes an open at its **last close line**, after
every use of it, so a trace showing fewer closes than descriptors (a duplicate closed and the
original closed implicitly at exit, which the export did not record) closes the open early,
and that close waits on a later use in its own lane: a hang. Two fixes were put: (a) record the
implicit closes and let the loader count only the closes after an open's last use, no format
change, but a close that is not the last issues no `close(2)` and NFS loses its flush; (b)
record the implicit closes and let the trace file **name descriptors**: a `dup` line, a
descriptor of its own for each lane a fork gave one, so every close line is a real `close(2)`
and the last one of an open still waits for its uses. **Decided: (b)**, the per-close flush of
item 4 carried to duplicates and to every implicit close (exit, `dup2`/`dup3` onto an open
descriptor, `execve` closing the `O_CLOEXEC` ones, `close_range`); a trace-file change made
with contract 0.15 (brief item 58), S-11, S-12, and S-14 together. The other group A items
go on meanwhile.

### 3.94 A run's preconditions: a third kind of state, modified in place, and its reset (debated with the user, decided 2026-11-02)

**How it came up.** Fixing S-17 (a create listed for a file that existed, `FS_API_SWEEP.md`),
the user saw the larger lesson: re-executing a trace, or an abstract with its parameters, may
depend on the storage holding at the start exactly what it held when the trace was taken, and
the suite had no notion of resetting the system under test to known preconditions. A tree
walk comparing what exists with what should exist costs hours of metadata load on a 50M-file
corpus (and pollutes what it would measure); reading every file to compare its content and its
compression and dedupe ratios costs a full read. The user was hesitant about both.

**What already holds, without a walk.** For an abstract, the precondition is a claim recorded
by the tool that built the state and checked by identity: a dataset's manifest
(`.aeiou-dataset.json`, its definition compared by `aeiou run`), never changed by a run (V12);
an output namespace empty at start; an input namespace with the manifest of the run that wrote
it, read-only (V14, V15). A run cannot change a dataset's compression or dedupe ratios, since
what a run writes goes to a namespace.

**What does not fit (the user's point).** "Full and read-only" or "empty and writable" is too
limiting: a KV cache restarted over the entries it left, or a vector index restarted over the
index on disk, finds content made before the run *and modifies it in place* (new entries,
evictions, overwrites, compaction). `lmcache-shared-fs` models only the cold reader.

**Decided (the user, 2026-11-02).**
1. **A third kind of state: made before the run and modified in place by it.** Its initial
   content comes from `aeiou datagen` or from an earlier run with its manifest; the manifest
   describes the *initial conditions*, not the current ones. Its name is settled when it is
   built (naming for the mechanism, `PROJECT_BRIEF.md` §8). This replaces the workaround of
   modelling the rewritten part as an earlier run's namespace restored by running the writer
   again: that is now the rule, formalized.
2. **Reset is datagen's, and costs the write set, never the corpus.** No tree walk, ever, in a
   run or in a reset. The op multiset is fixed before the run starts, so the **write set** (every
   path the run may create, write, truncate, unlink, or rename) is computed from the abstract,
   the parameters, the seed, and the GPU count (`dry-run`), not recorded; the user agreed the
   computed set meets the need ("efficient cleanup"). Reset regenerates each path of the set
   that the initial manifest holds (its bytes a pure function of the seed and position, so the
   initial content comes back exactly), removes each one it does not, and makes again each one
   the run removed. Datagen is given the manifest, the run's identity (to compute the set),
   and the progress marks.
3. **A per-actor high-water mark bounds an interrupted run** (the user liked this best): an
   actor's op sequence is deterministic, so a durable "ops completed" count per actor, saved
   from time to time, bounds what a stopped run touched at O(actors), not O(files); an
   in-memory list of paths is lost exactly when it is needed. Without the marks, reset
   regenerates the whole write set: correct, slower.
4. **A dirty marker.** A run that modifies a pre-existing state writes a marker at its root
   first; the next run refuses a dirty state rather than measure from another starting point,
   and reset clears it. One stat per root.
5. **Content is verified optionally and by sample**, in `aeiou-verify`, never in a scored run:
   a file's bytes are a pure function of (dataset seed, file id, offset), so checking N random
   files costs CPU, not a reference copy, and gives a confidence the user chooses, compression
   and dedupe ratios included; a full read stays possible, never required. Changes made out of
   band are found only this way.
6. **For the `trace` node** (brief item 14, §3.58: a captured trace executed literally inside
   an abstract, for calibration, never CLOSED): whether a path existed before the trace is a
   recorded fact of the capture, not a guess. The export decides it from all the trace's
   evidence, earlier and later (an `O_EXCL` create, a `mkdir`, or an ENOENT first: absent; a
   stat, an open without `O_CREAT`, an unlink before the create, or later a read of bytes it
   never wrote, an `fstat` or `SEEK_END` showing content: existed); where the evidence does not
   decide, the digest's interview asks the trace's owner and records the answer with its
   provenance; a **prepare** step puts the recorded state in place (inputs made at their read
   extents, creates removed). S-17 waits for this (the user, 2026-11-02): the first fix, which
   listed a create from a path's first appearance only, ignoring what the trace shows later,
   was reverted (57ba393).

Recorded as brief items 59 (the third kind of state and its reset) and 60 (the trace node's
recorded preconditions and prepare, with S-17); the sampled verify joins `aeiou-verify`.

### 3.95 Object requests are the abstract's own ops: no mapping between POSIX and object (debated with the user, decided 2026-11-03; not started)

**How it came up.** Fixing the object-log reader's findings (`FS_API_SWEEP.md` S-29 to S-36,
2026-11-02 and 11-03), the user saw that every one was a loss in a round trip. An abstract's
POSIX ops are turned into S3 requests by §3.65's mapping (`runner/REFERENCE.md` §16,
`object.rs`), and an object-store log is turned into POSIX calls (`objectlog.rs`, §3.88 step
5) that the engine turns into requests again. Each finding was something the POSIX shape could
not carry: part numbers and the abort of a multipart upload (S-29, S-32), suffix ranges and
part GETs (S-30), listing pages (S-33), a 404 GET that replays as nothing and a 404 HEAD that
replays as HEAD and LIST (S-36). The user's question: should a trace be taken at face value, an
S3 GET replayed as an S3 GET with no POSIX mapping, since no solid and comprehensive mapping
between the two exists and the open/read/close tuple does not recreate the application's
requests with fidelity?

**Assessment.** §3.65 had already said half of it (2026-10-09): "The object API has no
descriptor, `open`, `close`, `lseek`, `fstat` of a descriptor, `fadvise`, or `getdents`, so it
cannot reproduce a traced call sequence", for the direction from POSIX to object; the suite
then built the reverse direction anyway.

- *CLOSED already decides it.* The API is the application's (the interposition test, brief
  §5; §3.48), and §3.65's axis table binds the protocol family to the abstract. Declaring the
  protocol per namespace while keeping the op vocabulary POSIX stopped halfway.
- *The POSIX shape buys no comparability.* A run under another API is another workload and is
  never compared with it (§3.48), so a POSIX abstract run against S3 is not comparable with its
  POSIX run: the mapping pays in fidelity for a property nothing scored uses.
- *POSIX backends are interchangeable; objects are not.* A `pread`, a `libaio` read, and an
  `io_uring` read are the same kernel operation on the same file, so their mapping is exact. A
  GET as open, read, and close is a guess at what some client library would do.
- *What fidelity gains.* Each request replays as one request: ranged, suffix, part, and
  conditional GETs (the 304s), HEAD, LIST pages with their own `max-keys` and delimiter, PUT,
  multipart create, part N, complete, and abort, CopyObject, UploadPartCopy, DeleteObjects
  (S-34 solved, not deferred). A 404 reproduces itself. The parallel part uploads and parallel
  ranged GETs of one object that high-throughput clients issue (the AWS CRT, s5cmd) can be
  expressed, where the one-op-per-handle model and V18 serialize them. Object storage is
  measured per request (rate and latency by type, request size), which falls out directly.
- *A clean boundary.* An application reaching S3 through a FUSE file system (mountpoint-s3,
  s3fs) is a POSIX application: its strace is POSIX, and the FUSE layer is the solution under
  the interposition test. Object ops are for applications that call an object API themselves.
  Each trace keeps the shape it was captured in.
- *Against.* A second op vocabulary through the contract, schema, validator, builder, VM,
  `dry-run --metrics`, the trace file (a request line), the digest's sites, and `compare`; the
  format classes need object reader protocols, per format and per client library (pyarrow's
  S3 file system, fsspec's block cache, the CRT coalesce differently, which the POSIX reader
  protocol hides); the "this POSIX job on an object store" question is no longer an engine
  property; and the POSIX emulation in `object.rs` and the export to POSIX in `objectlog.rs` are
  given up (the logs' parsing carries over: `Content-Range`, `uploadId` binding, aws-chunked
  lengths, the notes of S-30 to S-35).

**Decided (the user, 2026-11-03).**
1. **An object namespace's ops are object requests, taken from the trace at face value.** No
   mapping from POSIX to object or back, in the runner, the trace file, or the export of a log.
   The POSIX APIs keep one POSIX op stream; the object client is an API with its own ops.
2. **No converter.** Many open-source projects map or convert between POSIX and object; the
   suite neither reinvents nor competes with them. A POSIX application on such a layer is
   traced as POSIX, the layer the system under test.
3. **Not short-term.** No one in the WG needs an object workload soon, and no committed
   workload declares `s3` (checked 2026-11-03), so no abstract or fingerprint changes when it
   is built. Queued as brief item 61.
4. **`FS_API_SWEEP.md` S-36 paused**: each of its four items (a read-only open of a missing
   key, HEAD and LIST for a 404, `BLKGETSIZE64`, the ignored `SYNC`, `DSYNC`, and `DIRECT`) is
   an artifact of the emulation. The fixes made for S-29 to S-35 stand until the rework; their
   log parsing carries into it.

**What it reverses or changes, when built.**
- The invariant "the abstract is POSIX-shaped and identical for every I/O backend" (CLAUDE.md,
  brief §5) becomes: identical for every backend *of one protocol*.
- §3.65's object mapping table and V18 (an object written once, in order): the table goes;
  V18 is moot (a PUT is whole by construction; parts come in any order). V19 (an API per
  protocol) stays, and matters more.
- `runner/REFERENCE.md` §16's mapping and `object.rs`'s emulation: `stat` as HEAD then LIST,
  `readdir` as every page, `rename` as a copy and a DELETE, an unlink marking open uploads,
  `Upload` buffering, `EBUSY` and `ESPIPE`. The transport stays: tokio inside the object engine
  (the cargo feature `object`), `--s3 blocking|async`, the eventfd that wakes the loop.
- §3.88 step 5 and §3.89: an object log exports request lines (the trace file's version 2);
  `trace::check_root`'s refusal of an object written out of order (S-29) goes.
- Brief item 58's object entries (a write's part number, a discard for an abort, the
  recursive listing of S-33) become native ops of item 61, out of item 58.
- The digest gains object sites, and `compare` judges request kinds; the builder gains object
  op nodes, and a format class an object reader protocol, refused (as V9 refuses an access
  mode) where it has none.
- Open: whether `object_store` can issue what faithful replay needs. From memory of the crate,
  not checked against its source: it numbers and orders a multipart upload's parts itself,
  pages a listing with no `max-keys`, and has no part GET or UploadPartCopy. If so, the engine
  needs `aws-sdk-s3` or a small signed-HTTP client of its own, which reopens §3.65's library
  choice (still inside the object engine's tokio boundary). To be checked first when item 61
  starts.

**One trace, one digest, one abstract over both protocols (the user's question, the same
day).** The motivating case is an application that uses POSIX and an object store for
different purposes. Milvus is one, checked against its own sources on 2026-11-03:
- Its storage layer has two backends behind one interface (`ChunkManager`): a local one, keys
  absolute paths under `localStorage.path` (`/var/lib/milvus/data/insert_log/1/2/3/...`), and a
  remote one, keys in a bucket under `minio.rootPath` (`files/insert_log/1/2/3/...`)
  (`docs/agent_guides/storage/path_contract.md` in milvus-io/milvus; high confidence).
- Object storage persists data and index files; query nodes cache what they serve and, under
  tiered storage (2.6.4 on), fetch field data and indexes on demand; Woodpecker, the default
  WAL, writes directly to object storage (milvus.io architecture overview, v3.0.x; high).
- Memory-mapped fields and indexes are files under `localStorage.path/mmap`, and a DISKANN
  index is read from local disk (`applications/milvus/3.0.toml`, read 2026-10-29: milvus.io
  `mmap.md`, `disk_index.md`, `configs/milvus.yaml`; high for the paths). That an index node
  builds a DISKANN index on local disk and uploads it is likely and not checked here (medium).
- No FUSE use was found in Milvus's documentation or source (a code search of milvus-io/milvus
  for "fuse" finds only words such as "refuse"). A FUSE file system or a CSI driver under
  `localStorage.path` would be a deployment's choice, and the solution's under the
  interposition test (medium: absence of evidence).

So a Milvus query node, in one process, GETs index files and binlogs and reads, writes, and
maps local files. Where the suite stands for such an application:
- *The abstract: yes, by design.* Protocol and API are bound per namespace (§3.65, decided and
  built 2026-10-09: per-name endpoints, V19), so one abstract holds a POSIX namespace and an S3
  one, one actor issuing both. With item 61 its object ops are requests; the data flowing
  between them (an object fetched and written to local disk, then read) is two namespaces'
  ops, sizes from the same positional expressions.
- *The trace: not yet; one source at a time today.* An strace never shows the object requests
  (TLS, `applications/milvus` records the blind spot), and an object log has no thread and the
  server's clock. One trace of both is a merge of the two: lanes from the strace's threads,
  requests attributed to them by the client's address and by time, with the clocks' offset
  bounded (the strace's `%network` calls give the client side of each request's connection
  and its timing; over plain HTTP, its request line). Not designed; it joins brief item 49
  (the logs not yet read and the merged trace) and item 52 (all-rank traces).
- *The digest: not yet.* Its four site kinds (a sample set, a keyed lookup, a bulk write, one
  open read at many offsets) are POSIX-shaped; item 61 adds object sites, and a digest of a
  merged trace has sites of both.

### 3.96 The capture: what strace costs, what eBPF and a shim reach, and the kit item 54 builds (debated with the user and measured, 2026-11-03; decided, not started)

**How it came up.** The user took up brief item 54 to be able to ask a trace's owner for one
capture holding both POSIX calls and object requests. This side proposed the strace line with
`%network` and `-s 512`, the store's own log, and anchor requests for the clocks; the user
asked how much slower `%network` makes the capture and whether that lowers what the runner
demands, whether a client-side `LD_PRELOAD` shim could record the object requests as it passes
them to the library, whether strace is itself slow (it flushes per call, so a binary format
parsed afterward might be much faster), and how eBPF compares. A spike measured each, the same
day.

**The spike.** WSL2, kernel 6.18.54.1, 20 cores (i7-13800H); strace 6.8, bpftrace 0.20.2 (the
eBPF runs by the user, as root), MinIO RELEASE.2025-09-07 on 127.0.0.1 with its data in
`/dev/shm`, boto3 1.43.123, `s3torchconnectorclient` 1.5.0 (the AWS CRT, no torch), the
system's Python 3.12.3 (OpenSSL 3.0.13, shared) and uv's standalone Python 3.12.13. Programs
written for it, in `spikes/2026-11-03-capture/` with their raw results and a `setup.sh` to
repeat them (spike code, not tested, part of no tool): a microbenchmark (cached 4 KiB `pread`s of one
file, each followed by K untraced `getppid`s), a minimal ptrace tracer (`PTRACE_GET_SYSCALL_INFO`,
an optional seccomp filter on the strace set's calls, binary records of time, tid, number, and
six arguments at entry and the return at exit, a path read with `process_vm_readv` at entry,
one `write` per MiB), an `LD_PRELOAD` shim logging HTTP request and response heads seen through
OpenSSL (`SSL_write`/`SSL_read` and their `_ex` forms) and through libc on sockets, with tid and
`CLOCK_REALTIME`, and S3 loads (`s3load.py`: PUT then GET, a ranged GET each; `s3get.py`: GETs
only over preloaded objects, 200 of 4 KiB and 8 of 8 MiB read twice, 416 requests and 128 MiB).
Every number below is WSL2's, whose ptrace wakeups across CPUs cost about 50 µs; a real Linux
client node is to be measured with the same scripts before a choice rests on a ratio.

*ptrace, per traced call* (100,000 `pread`s, K = 4, best of 3): untraced 0.9 µs; strace with
the man page's line, which has no `--seccomp-bpf`, 320 µs; with `--seccomp-bpf` 88 µs (90 as
root); the same writing to `/dev/null` 67; without `-ttt -T -yy -v` 81; with `-s 512` 93 µs and
a trace 5.7 times larger (159 MB against 28); the minimal tracer without a filter 280, with
one 56 µs. Tracer and application pinned to one CPU: strace 33 µs, the minimal tracer 6.5 µs.
So the wakeup dominates when the two run on different CPUs, and when it does not, strace's own
work (about 26 µs a call) is five times a lean tracer's.

*strace's own work* (an outer `strace -c` of strace, pinned): per traced call two `write`s,
one at its entry and one at its exit, unbuffered (54 % of strace's own system-call time; the
user's guess), four `wait4` and four `ptrace` (the floor), three `newfstatat` and one
`readlink` of `/proc/PID/fd/N` (`-yy`), and one `process_vm_readv`; about 15 calls where the
minimal tracer makes about 6 and one `write` per MiB.

*S3 GETs* (`s3get.py`, medians of 7 interleaved runs, then of 5 as root):

| | untraced | shim | strace `--seccomp-bpf` | + `%network` | strace, no seccomp | minimal tracer | eBPF record |
|---|---|---|---|---|---|---|---|
| boto3 | 0.96 s | +5 % | +38 % | +44 % | +40 % | +18 % | +2 % |
| CRT | 0.41 s | +4 % | 7.1× | 6.9× | 8.2× | 3.7× | +24 % |

`%network` adds little: the CRT reads and writes its sockets with `read` and `write`, which
`%desc` traces already, and boto3's `recv_into` is one call per buffer. The CRT under strace
was bimodal (1.1 s to 3.0 s), its event loop's timing changed by the stops: for an
asynchronous client the slowdown is a fidelity question as well as a cost.

*What the shim reaches.* Over HTTP every request and response of both clients (boto3 47 of 47,
the 46 issued and the bucket's create; CRT 90 of 90), with `Range` and `Content-Range`. Over
TLS, boto3 under the system's Python, 47 of 47 in plaintext, once the shim found the real
`SSL_read_ex` by walking the loaded objects: Python loads `_ssl` and its `libssl` with
`RTLD_LOCAL`, so `dlsym(RTLD_NEXT)` returns NULL (the first version crashed on it). Not over
TLS: uv's standalone Python (python-build-standalone links OpenSSL into the interpreter,
nothing exported) and the CRT (s2n-tls inside its own `.so`; it imports no `libssl`, but
libc's `send`, `sendmsg`, `write`, `writev`, `read`, `recvmsg`, so over HTTP it is reached).
Go clients (Milvus's minio-go) make their system calls without libc, and Rust's rustls and the
JVM carry their own TLS (from knowledge, not checked in the spike). Two facts about the CRT
that bear on an abstract: every PUT is a multipart upload (create, one part, complete), a
4 KiB object's too, and every request is issued from its event-loop thread, not the thread
that asked, so a shim's or a probe's thread for an asynchronous client is the loop's.

*eBPF* (bpftrace on the raw tracepoints `sys_enter`/`sys_exit`, the strace set's calls, the
target in a cgroup of its own: bpftrace's `pid` is the kernel's, and WSL2 runs the distribution
in a PID namespace, so `pid == cpid` never matched; reading a tracepoint's arguments made
bpftrace look for kernel headers WSL2 does not ship, so the arguments were read from `pt_regs`
at fixed offsets; WSL2's `perf` has no `trace`). Per call: counting only +0.2 µs, recording
every call with four arguments +0.36 µs; on the S3 loads boto3 +2 %, the CRT +24 % (its many
socket calls and bpftrace's formatting competing for the CPUs). A uprobe on the system's
`libssl` `SSL_write_ex`, copying 64 bytes of the buffer, gave boto3's request lines over TLS at
+2 % (the shim +1 %), the same clients the shim reaches; the user's expectation that eBPF gets
no TLS plaintext holds only for clients without a shared `libssl`.

*eBPF's losses.* A probe cannot make the application wait, so where the reader falls behind,
events are lost (counted). At bpftrace's default 64 pages per CPU the 100,000-`pread` run lost
40 to 52 % of its events; from 1,024 pages (4 MiB) none were lost, at the same per-call cost;
one record per call (arguments kept per thread at entry, printed at exit) halved the events
but cost +0.25 µs in bpftrace's six hash-map writes. Sustained (no untraced calls, about 2.9
million events a second from one thread): 1 million `pread`s lost 50 % at 1,024 pages and 3 %
at 16,384 (64 MiB); 5 million lost 52 % and 43 %. bpftrace's text reader sustains roughly a
million events a second, about 1 µs of CPU each; a larger buffer only delays the loss.

**Assessment, this side's, the user to decide.**

- *Throughput is the reader's, not the probe's.* The probe costs 0.2 to 0.4 µs a call; a
  reader that copies binary records rather than formatting text should keep up with tens of
  millions of events a second (estimated, not measured), leaving the disk as the limit (2.9
  million events a second at about 48 bytes is 140 MB/s). The rate tested is one thread making
  cached calls flat out; storage-bound calls are orders of magnitude slower, but many threads
  reading from the page cache could approach it. So the tool sizes its buffer to hold seconds
  of its peak, can run a reader per group of CPUs, writes to fast local storage and reports its
  write rate, and counts every loss by CPU and time range in the capture's record, a trace
  with a gap flagged as one; the owner can re-capture under ptrace, which is lossless by
  construction.
- *Records are variable length, and the probe copies what a pointer names.* A record of a call
  on a descriptor (`read`, `pread`, `write`, `fsync`, `close`), the bulk, is 32 to 64 bytes;
  a call that takes a path carries the string, copied by the probe at entry on the
  application's thread (`bpf_probe_read_user_str`), since the pointer means nothing to the
  reader and the memory may be reused before it reads. A probe in a raw tracepoint cannot take
  a page fault, so a string whose page is not resident is not copied: the probe retries at the
  call's exit, when the kernel has just copied it itself, or a sleepable program
  (`fentry.s`, `bpf_copy_from_user`) faults it in; every path not copied is counted. The same
  holds for an HTTP head copied at `SSL_write`. A descriptor's path is resolved after the run
  from the recorded opens, `dup`s, forks, and `chdir`s, as the export already does, where
  strace's `-yy` reads `/proc` per call; the kernel's resolved path at an open's exit
  (`bpf_d_path`, allowed only at some hooks) is an option to check per kernel.
- *The clock.* eBPF's `nsecs` is the monotonic clock, strace's `-ttt` and the shim's records
  the wall clock; a capture records the offset.
- *The pause is a fidelity cost.* An abstract does not replay a trace's timing, so a slowed
  capture does not make the runner demand less, except where a parameter is taken from the
  trace's gaps (the digest's think time, a save's interval: under ptrace they are inflated, so
  they come from an untraced run or the application's own log, as `mountstats` comes from runs
  without strace) or where the application's behavior depends on timing (an asynchronous
  client's concurrency and part sizes, a prefetch queue's depth, work handed to whoever asks
  first), and ptrace's serialization of stops lowers the concurrency a trace shows. The `trace`
  node replays think time unless run with `--time-scale 0`.
- *`-s 512` is withdrawn.* It applies to every buffer strace prints, file data included (the
  owner's contents in the trace, 5.7 times its size); the object requests come from the shim,
  a uprobe, or the store's log instead. And the man page's strace line lacks `--seccomp-bpf`,
  3.6 times slower without it on the microbenchmark (the DiskANN kit used it, §3.42's capture
  notes). *Added the same day* to every capture line (the man page, both references, the
  README, `ABSTRACTS.md` §11, the workloads' recipes, `aeiou-trace`'s help and messages):
  strace prints the same lines with it, and proceeds without it where the filter cannot be
  installed.
- *Object-store logs need a reader per format.* No standard exists: S3's server access log and
  CloudTrail differ, MinIO's audit log, Ceph RGW's ops log, Azure's resource logs or Storage
  Analytics, GCS's audit logs, and OCI's service logs each have their own (from knowledge,
  medium confidence, each read from its vendor's documentation before a reader is written),
  and each lacks something (S3's has no `Range` and whole seconds). Azure Blob and GCS's JSON
  API are not S3 requests at all, so item 61's ops would need their own or a capture through an
  S3-compatible interface. A logging proxy or an HTTP capture in a lab is one format whatever
  the store.

**The proposal** (brief item 54, revised the same day): a capture tool of the suite that
starts the command (never attaches), with four sources chosen per host and client, all
converted afterward to the strace text the suite reads, so every reader and golden keeps
working: eBPF where the owner can grant root or `CAP_BPF` and `CAP_PERFMON` (the default:
system calls through raw tracepoints, libssl uprobes for request heads, and later the mapping
faults and `io_uring` requests the item began with); ptrace where it cannot (strace with
`--seccomp-bpf`, or a binary tracer of ours, 1.6 to 5 times faster here); the `LD_PRELOAD` shim
for request heads where the client loads a shared `libssl` and eBPF is not available; the
store's log for the rest, aligned by anchor requests (an unsigned `HEAD` logged on both sides).
It records the kernel, the tools and versions, the command, each mount's filesystem, the
endpoints and their scheme, whether the cache was cold, the clocks' offset, the buffer size,
every loss, and what it could not see. *Open for the user:* aya or libbpf-rs for the eBPF
side (gcc cannot build BPF programs; clang with the BPF target or aya's toolchain is needed);
a ptrace tracer of ours or strace with `--seccomp-bpf`; whether the shim shares a library with
item 55's `fstat` shim; the anchor requests (an unsigned `HEAD` shows as a 403 in the owner's
audit log) or a command the owner supplies; and the measurements on a real Linux client node
first. *Decided by the user the same day:* the ptrace source is strace with `--seccomp-bpf`,
a tracer of ours only if a need appears; the anchors are the unsigned `HEAD`s; the
measurements on a real Linux node wait until one is available (none now). Then, the same
day: **aya**, both sides in Rust, the BPF object embedded in `aeiou-trace` so an owner needs
only the release tarball (the user's leaning for programs that stay simple, this side
agreeing); its costs a nightly toolchain and `bpf-linker` in CI and `pt_regs` offsets per
architecture; ~~the later probes on kernel structures (mapping faults, `io_uring`) need CO-RE
relocations, which aya loads from a C-built object (medium confidence, checked when they are
built).~~ the later probes on kernel structures read their offsets from the kernel's BTF at
load, as phase 3's do, and need no CO-RE (checked 2026-11-04, §3.98). **The shim: shared code, two libraries.** One crate holds the interposition machinery
(the real symbol found past `RTLD_LOCAL`, the reentrancy guard, logging without allocation),
built as two shared objects, the capture's and item 55's `fstat` one: the capture observes and
item 55's changes what the application sees, so the capture's library holds no code that can
alter a result, and no stray variable turns a capture into an experiment.

**The capture tool's design (debated with the user the same day, 2026-11-03; this side's
proposal, recorded at the user's request before a context clear; ~~not approved, nothing
built: the user says "proceed" before any of it is coded~~ approved by the user later on
2026-11-03, defaults included, and phase 1 built the same day: below, "Phase 1 as built").**

*Shape.* `aeiou-trace capture [--source auto|ebpf|strace] --root DIR... [--endpoint URL]...
[--cold] [--accept CHECK]... -o DIR -- COMMAND [ARGS...]`, a subcommand of `aeiou-trace`
(the release tarball is all an owner needs). It always starts COMMAND (no `-p`, which also
means a privileged binary can watch nothing but its own child), exits with COMMAND's status
as strace does, forwards `SIGINT`/`SIGTERM` (the scheduler's at a time limit too), drains,
and writes its record marked interrupted. Usage errors in the §15 frame; `aeiou-trace(1)`
gets a `capture` section, and an owner's page `CAPTURE.md` ships in the tarball beside
`QUICKSTART.md` (recommended; the man page alone is the alternative).

*The capture directory* (`-o DIR`, empty or absent; `%h` host and `%p` pid expand, so one
template serves every node): `trace.txt`, the strace text every reader reads (strace's own,
or converted from the eBPF records); `capture.json` (`aeiou_capture: 1`): tool version,
kernel, distribution, CPUs, memory, source and its version and why it was chosen, the exact
strace line or probe set, argv and cwd, exit status, start and end, realtime minus monotonic
at start and end, each `--root`'s mount (filesystem, server and export, options from
`/proc/self/mountinfo`), endpoints and scheme, cache `dropped` (`--cold`, root) or `not
dropped`, losses (0 under strace; by CPU and time range under eBPF), paths not copied, what
the capture could not see, the uid the command ran as, waivers, the trace's size and sha256;
and the sources' raw files kept (`syscalls.bin`, `requests.jsonl`) so a converter's bug is
fixed by re-converting, never by asking the owner again. No `mountstats` from a traced run
(the record says to take them untraced). Proposed defaults, the user to confirm: `%network`
always in the strace set (+6 % on boto3; an owner cannot know whether descriptors pass over a
socket); at least one `--root` or `--endpoint` required; `--cold` as sync and drop (or a
`mincore` sample of `--root`, the user's choice); the readers take `DIR/trace.txt` in phase 1
and the directory itself (its `--root`, `--cwd`) later, once phase 2 gives it a second file.

*Starting the command.* The tool forks the child, which waits on a pipe; the tool records
the child's inherited descriptor table (for the converter), attaches, then releases it to
`exec`. Under strace, strace starts the command as the man page's line does.

*Privilege (eBPF only; the user: the traced command must not run as root, and a cluster
capture cannot prompt for a password, since under ssh, `srun`, or `mpirun` there is no
terminal).* The tool always runs as the invoking user in their environment (sudo would
reset `PATH`, `LD_LIBRARY_PATH`, the virtualenv: another run); it never prompts. Privilege is
needed only to load and attach, and comes, in order, from (1) file capabilities an
administrator grants once on a node-local copy (`setcap cap_bpf,cap_perfmon+ep`,
`CAP_SYS_ADMIN` before 5.8; not over NFS, which carries no `security.*` xattr; execution
limited to a group, since the binary can probe the host): the cluster default; (2) a
`NOPASSWD` sudoers rule for a helper mode of the same binary that takes no arguments (its
requests arrive over a socket, so the rule names one exact command), called with `sudo -n`,
the binary root-owned; (3) on a single host, `sudo -v` beforehand so `sudo -n` succeeds.
The helper loads and attaches and passes the link and ring-buffer descriptors back over a
Unix socket (`SCM_RIGHTS`) and exits: privilege is checked at load and attach, not at read,
so the reader is unprivileged and the probes live as long as it holds the descriptors.
Under capabilities the tool attaches first and drops every capability for good before it
releases the command or reads the buffer; a capability binary is secure-exec (no
`LD_PRELOAD`), and `exec` of the command carries none. The target is filtered by process
tree, not a cgroup (which would need delegation): the namespace-local pid
(`bpf_get_ns_current_pid_tgid`, 5.7+, which also fixes WSL2's mismatch) and children added
at `sched_process_fork`. Run wholly as root, the record says so and the tool warns (some
applications run as root, in containers especially), never refuses.

*Several nodes (the user: `aeiou-launch` needs only password-less ssh, and that is the
baseline; Slurm and its kind supported, never assumed; torchrun is training's alone, VDB
and KV cache are servers and load generators).* Two cases. (1) The owner can start the
application per node over ssh: `aeiou-launch HOSTS -- aeiou-trace capture ... -- COMMAND`,
a fourth command of the launcher (it accepts only `aeiou run`, `aeiou datagen`,
`aeiou-datagen` today) with the same three flags appended, rank meaning only host index.
The capture joins the runner's TCP coordinator for a start barrier (every node attached
before any node's command is released; one that cannot, none start), clock offsets between
hosts measured over its connections against rank 0's, bounded by half the round trip
(beside `ntp_adjtime`'s unprivileged synchronized flag and error bounds, and the anchor
`HEAD`s tying hosts to the store's log), stopping together, and a `capture-set.json` on
rank 0 (each node's directory, source, losses, offset, status; only records cross the
network, traces stay on each node's local disk for the owner to collect). One capture per
node: every attached probe runs on every call on the host, so N per node cost N times.
(2) The application has its own multi-host launcher (`mpirun`, `srun`, Kubernetes, a
service manager): the capture goes inside that line, joined by `--capture-id` (default
`SLURM_JOB_ID` when set), and accepts `--ranks`, `--rank`, `--coordinator` directly so such
a line can supply them from its environment and get the barrier and comparisons; without
them, per-node only, the cross-node comparison made afterward from the records and marked
so. Several captures per node (one per `mpirun` rank) are recorded with their count; one
probe set shared among them on a node is deferred until a case needs it. Recorded from the
environment: a fixed list only (`SLURM_JOB_ID`, `SLURM_NODEID`, `SLURM_PROCID`, `RANK`,
`LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`), never the whole environment. No silent mix of
sources (one node under strace slows a synchronized job and distorts every node's trace):
the barrier refuses it; `--source auto` is for a single host.

*The checks are the capture's first step, every time (the user: protect against a runner
who forgets a preflight by hand, or whose cluster changed overnight).* In `aeiou-trace
capture`, not `aeiou-launch`: case (2) has no launcher, the knowledge (capabilities, BTF,
lockdown, strace) is the binary's, and the launcher stays a generic ssh loop; the pattern is
the runner's own (§11's limit checks before the gate, rank 0's differences table). Each node
checks itself: the source it can use and why (capabilities or `sudo -n`, lockdown, BTF,
kernel), the strace version, the release, each `--root`'s server, export, and options,
clock synchronization, `-o` on local disk, not under any `--root`, its free space, no other
capture running; rank 0 gathers and compares across nodes (same release, same mounts and
options: a node mounted with another `vers`, `rsize`, or `nconnect` is the classic bad node;
clock offsets) and prints one table. Any check failing on any node: no node starts, every
capture exits non-zero, the table names each node and check. Every check fails by default;
a deliberate exception (a mixed cluster under test) is waived by name, `--accept CHECK`
(repeatable, one flag for every check), printed and recorded. The checks run again at the
end (mounts, clock state) and every change is recorded. `--check` remains a dry mode for
planning (no command), never required.

*The phases* (each tests first): (1) strace source, the record, the checks, the barrier,
clock offsets, `capture-set.json`, `aeiou-launch` support, `CAPTURE.md`: everything that
holds under strace, so phase 1 is a coordinated tool, not a wrapper; (2) the HTTP-head shim
(a Rust `cdylib`, the capture's library of the interposition crate) writing
`requests.jsonl` on the wall clock with tids, the anchor `HEAD`s at start and end of each
`--endpoint`, a third format for `objectlog.rs` (not implemented in it: the join of requests
to calls, items 49 and 61); (3) eBPF system calls with aya: raw tracepoints, variable-length
binary records, a binary reader, a buffer of seconds of the peak, every loss counted, the
converter to strace text resolving descriptors from the recorded opens, `dup`s, forks, and
`chdir`s, and its test an equivalence of the *parsed* calls of one command under strace and
under eBPF; needs nightly Rust and `bpf-linker` locally and in CI, pinned by `dev.sh`, the
user to approve installing them (GitHub's runners have `sudo` for the root tests); (4)
~~libssl uprobes, then mapping faults and `io_uring` (CO-RE).~~ the probes of §3.98, in its
order: `io_uring`, mapping faults, libaio's completion, POSIX AIO, libssl, cuFile and libnfs.

*Open, later.* Servers started in containers (Milvus, vLLM with LMCache): the capture inside
the container (the binary there, `--cap-add BPF --cap-add PERFMON`) or on the host against a
container it did not start, against "start the command, never attach"; the user decides
when those workloads come. Whether `aeiou run` should compare mounts across hosts too (not
checked whether it does; a brief item if the user wants it, not folded in here).

**Phase 1 as built (2026-11-03, the user: "All your recommendations are good, proceed").**
The defaults proposed above are decided as proposed: `%network` always in the strace set; at
least one `--root` or `--endpoint`; `--cold` a sync and drop, which needs root (the `cold`
check fails without it, waivable when the owner dropped the cache beforehand); `CAPTURE.md`
in the tarball; the readers take `DIR/trace.txt` (the directory itself when phase 2 gives it
a second file). The nightly toolchain and `bpf-linker` wait for phase 3; containers, and
mounts compared by `aeiou run`, stay open. Built: `aeiou-trace capture` (`capture.rs`, the
subcommand in `bin/aeiou_trace.rs`), the record and the checks as proposed, twelve by name
(eight a node makes of itself: `source`, `root`, `clock`, `output-local`,
`output-outside-roots`, `space` at 1 GiB, `alone`, `cold`; four rank 0 makes across nodes:
`release`, `same-source`, `mounts`, without a client's own `clientaddr`, and `clock-offset`
at 10 ms beyond the bound), the coordinator's clock rounds (`Clock` and `ClockReply`, new
messages that only rank 0 of a capture sends, so the Python datagen's coordinator, which
never sends them, is unaffected; `Server::clocks`, eight rounds, the shortest kept, Cristian's
method), and `aeiou-launch` putting the rank flags right after `capture`, since appended
they would follow the capture's own `--` and be the traced command's.

Choices made in the building, each the simpler of two: the protocol is `Hello` (the node's
document as its options block), rank 0 judging once every `Hello` is in, then `Stop` with the
table or `Start` carrying it; the directory is made and the cache dropped only after
`Start`, and the commands then wait at a coordinator barrier (`capture-start`), so a check
failing anywhere leaves no directory anywhere. A `-o` that is not empty is an error, not a
check: no waiver could make a capture into a used directory sound (on several nodes the
other ranks then wait out the coordinator's connect window, as for any rank that fails
before it connects). A signal sent to the tool alone (`si_code` not `SI_KERNEL`) is passed to
the command, strace's child, found by its parent in `/proc`; a terminal's Ctrl-C is not
passed again, since the command's process group had it already, and strace itself holds
fatal signals under `-o FILE PROG` (`-I3`). The tool polls `waitpid` every 10 ms rather than
blocking in it, since a signal may land on a coordinator thread. Not built in phase 1: a node's
interruption stopping the others (each node stops on its own signal, and `aeiou-launch`
sends it to every rank; a `Halt` broadcast would deliver a second signal to a command that
had the first), the comparison afterward of records taken without a coordinator, and the
sources' raw files (strace has none). Brief item 54.

**Phase 2 as built (2026-11-03, the user: "All your recommendations are good, proceed").**
The proposal, as made and approved: (1) the record written into a mapped log, not a `write`
per head, since the spike's shim's `write` and its `fstat` per descriptor are calls in `%desc`
the trace would show as the application's; (2) a header allowlist, credentials never kept;
(3) `--requests shim|none`, `shim` the default with an `--endpoint`; (4) the library shipped
in the tarball's `lib/` beside `bin/`, not embedded (stable cargo cannot build a `cdylib` as
another crate's dependency); (5) ~~`reqwest`, which `object_store` already brings, for the
anchors over `https`~~ *changed in the building:* `reqwest` is asynchronous and needs a tokio
runtime, which the invariant keeps to the object engine (`CLAUDE.md`), so the anchors use
`rustls`, the TLS `reqwest` itself uses, blocking over a `std` socket, under the same feature
`object`. Built: `runner/interpose/` (`aeiou-interpose`: `next`, the real symbol
past `RTLD_LOCAL`; `Guard`, the per-thread reentrancy guard; `log`, the appending log; and
`capture`, the library's records, one definition for writer and reader) and
`runner/capture-shim/` (`libaeiou_capture.so`, its hooks `connect`, `close`, `write`, `send`,
`sendto`, `writev`, `sendmsg`, `read`, `__read_chk`, `recv`, `__recv_chk`, `recvfrom`,
`__recvfrom_chk`, `readv`, `recvmsg`, and `SSL_write`, `SSL_write_ex`, `SSL_read`,
`SSL_read_ex`), `runner/aeiou/src/requests.rs` (the log made, converted to `requests.jsonl`,
the execs of the trace against the processes the library reached, the anchors), the
`requests` check, `objectlog.rs`'s third format (`aeiou-requests`), and the readers taking
the capture directory (`aeiou-trace(1)`, CAPTURE DIRECTORIES).

Choices made in the building, each the simpler of two:

- *The log.* A file in `-o`, sized by `--requests-buffer` (1 GiB by default, sparse), each
  record's space reserved by an atomic add on the header's next offset, its length stored
  with a pending bit at once and without it once the payload is in, so a record its process
  died writing is told from a whole one (`torn`); a record that does not fit is dropped and
  counted, never wraps or blocks. Named to the library by `AEIOU_CAPTURE_LOG`, a path, not an
  inherited descriptor: Python's `subprocess` closes inherited descriptors by default, and an
  environment survives it as `LD_PRELOAD` does. Both are put in the command's environment by
  strace's `-E`, so strace itself runs without the library. The log is cut to what was
  recorded at the end and kept (`requests.bin`).
- *What the trace shows of the library.* Each process's start: `openat`, `fstat`, `mmap`, and
  `close` of `requests.bin`, and the dynamic loader's open of the library, all outside every
  `--root`, so no reader counts them and the readers need no filter. The program's name
  comes from the auxiliary vector (`AT_EXECFN`), not a `readlink` of `/proc/self/exe`; the
  thread id is asked once per thread (`gettid`, in no traced class), the parent's once per
  process (`getppid`, likewise); a socket is known by its `connect`, never an `fstat`.
- *An SSL connection's socket.* `SSL_get_fd` returns -1 under curl, whose OpenSSL backend
  reads and writes through a BIO of its own, so the socket is learned from the libc call made
  under the SSL call (the guard held across it); the head is recorded after the SSL call
  returns, stamped as of its start for a request. `SSL_get_fd` is the fallback.
- *Which processes it missed.* Every successful `execve` in the trace is a process the
  library should have loaded in; each one it did not is named (`processes.not_reached`, with
  a warning), and a process that connected and showed no head is noted. When the log filled,
  a process's own record may be among those dropped, so `not_reached` is then null and says
  why rather than naming processes falsely.
- *Pairing.* Responses with requests in order per process and connection (the `SSL *` or the
  descriptor), as HTTP/1.1 answers them; a response without a request is counted. A request's
  end is its response's head, the body's end not seen.
- *The readers.* A capture directory is read as its `trace.txt` when every `--root` is a
  directory (absolute, beginning with `.`, or existing) and as its `requests.jsonl` when
  every one names a bucket; both kinds at once is a usage error. The record's `command.cwd`
  is the default `--cwd`. The library's requests are named by the S3 REST API from method,
  path, query, and headers, path-style or virtual-hosted under an `s3.` host, and their lane
  is the client thread.
- *Finding the library.* A build tree's `target/release/deps/` (where a test build leaves it:
  cargo builds a `cdylib` for tests only when the crate is also an `rlib`, and never uplifts
  it), then `target/release/`, then the release's `lib/`. A missing library fails the
  `requests` check; waived, the capture runs without it and the record says why.
- *Anchors.* Sent whenever an `--endpoint` is an `http://` or `https://` URL, with or without
  the library, by the tool over a plain socket or blocking `rustls` (the certificate not
  checked, its signatures still: the anchor carries nothing but its name); an `s3://` URL names no host and records `not_sent`.

Tests: the shim's own (`capture-shim/tests/preload.rs`: the system's Python through each
hooked libc call and a fork), and `aeiou/tests/capture_requests.rs` (curl against MinIO,
signed, with no secret in any capture file; curl over TLS through a shared libssl; a full
log; a process without the library; `--requests none`; a binary without its library). Not
built in phase 2: the join of requests to the calls under them and their replay (items 49
and 61); TLS inside the program, Go, HTTP/2 (phase 4's uprobes reach only shared libssl
too). Brief item 54.

**Phase 3 decided (2026-11-03, the user: the toolchain approved, "I like your recommended
options", the 256 MiB buffer "but we might decide to increase the default once we get
experience with tracing in the wild"; the loss policy: "approved, proceed").** This side's
proposal, as approved:

- *The toolchain.* The BPF target (`bpfel-unknown-none`) has no prebuilt `core`, so it is
  built with `-Z build-std=core`, which only a nightly compiler allows. "Nightly" is Rust's
  channel, not a schedule: a dated nightly is pinned (`nightly-2026-10-15`, rustc 1.101.0, LLVM
  23.1.3) in `runner/capture-ebpf/rust-toolchain.toml`, as the stable one is in the root's, and
  every CI run and every release builds with it; the rest of the runner stays on the stable
  pin. `bpf-linker` 0.12.0 (2026-10-13, LLVM 23; 0.13.0 was a day old) is pinned in `dev.sh`
  by URL and SHA-256 as its static musl build, unpacked with `zstd`, a prerequisite `dev.sh`
  names; aya 0.14.0 and aya-ebpf 0.2.1 (June 2026; 0.15 and 0.16 were days old).
- *The object built by `aeiou`'s `build.rs`, so every build needs the nightly* (one binary,
  no surprises), never a committed object. *Changed in the proposing, the simpler of two:*
  not `aya-build`, which builds the probe crate as a member of the workspace (so `--workspace`
  and coverage would compile it for the host too), but a `build.rs` of ours that runs the
  probe crate's own `cargo build` in its own directory with a clean environment (no
  coverage flags reach it), the crate outside the workspace with its own lock.
- *The `sudo -n` helper* (descriptors handed back over `SCM_RIGHTS`) *is phase 3b*: phase 3
  attaches under file capabilities (`cap_bpf,cap_perfmon+ep` on a node-local copy, dropped for
  good before the command is released) or as root (recorded, warned).
- *The buffer:* one ring buffer, `--buffer SIZE`, 256 MiB by default (about two seconds of
  the spike's 2.9 million events a second), the default to be revisited with captures in the
  wild; a ring per group of CPUs only if one reader is measured falling behind.
- *Losses: any loss is loud, no threshold.* The damage is not proportional to the count: one
  lost `openat` leaves every later call on its descriptor without a path, one lost `clone` a
  thread without a table. At the end the tool still exits with the command's status, and
  prints a box on stderr: the events lost, on which CPUs, in which windows, and a `--buffer`
  computed from the reader's measured peak (events a second times bytes a record times the
  longest burst, rounded up to a power of two); `capture.json` says `complete: false`. The
  converter writes a marker line at each window; `metrics`, `digest`, and `export` still
  read the trace, warn, carry the loss in their output, and mark a descriptor whose open fell
  in a gap unresolved, never guessed; `compare` flags a side with losses. The remedies named,
  in order: a larger `--buffer`; `-o` on faster local storage when the reader's write rate was
  the limit (it is measured); `--source strace`, lossless and slower. *Later, not now:* a
  lossless eBPF mode, the probe stopping a thread (`bpf_send_signal`) near a full buffer and
  the reader resuming it once drained, ptrace's cost paid only in a burst.
- *Found in the proposing:* a raw tracepoint's `sched_process_fork` passes the child as a
  `task_struct *`, and aya has no CO-RE, so the reader takes the three offsets it needs
  (`pid`, `tgid`, `real_parent`) from the kernel's own BTF (`/sys/kernel/btf/vmlinux`) at load
  and hands them to the probe in a map: the object stays one per architecture, and a kernel
  without BTF fails the `source` check. ~~The `pt_regs` layouts per architecture come from
  aya's bindings.~~ The `pt_regs` layouts per architecture are the kernel's fixed ones
  (`arch/*/include/asm/ptrace.h`), constants in the probe: aya's aarch64 bindings leave the
  struct opaque.

**Phase 3 as built (2026-11-03, the same day, as decided above).** Built: `runner/capture-ebpf/`
(the four programs: `sys_enter` and `sys_exit` on the raw tracepoints, one record at each
call's entry with its six arguments and the strings and structures it names, one at its
return; `sched_process_fork`, which adds the child to the traced tree before it can run and
records it as the command's namespace numbers it; `sched_process_exit`, each thread's wait
status), `runner/aeiou/src/ebpf.rs` (the object embedded, loaded with `--buffer`, the calls'
table, the kernel's BTF, the capabilities, the reader), `runner/aeiou/src/ebpf_text.rs` (the
records as strace's text), `--source ebpf|auto`, `--buffer`, `--convert DIR`, and the loss
policy as decided, the readers' `events_lost` note and `compare`'s warning included. Tests:
`aeiou/tests/capture_ebpf.rs`, whose first is the equivalence the phase was to be judged by: one
program (files of every read and write family, `dup`, `lseek`, `chdir` and relative paths, a
rename, a listing, four threads, a fork, a failed open) captured under strace and under eBPF
gives the same exported ops, lane by lane, and the same metrics. Choices and findings in the
building:

- *The calls traced* are strace 6.8's own expansion of the capture's line for each architecture
  (its `syscallent.h` tables, 209 calls on x86_64 and 175 on aarch64), committed as
  `runner/aeiou/src/syscalls/{x86_64,aarch64}.tsv` with what each argument is, one letter each.
  The probe copies only what a reader uses: paths and other strings, iovec and iocb arrays,
  `clone3`'s and `openat2`'s structures at entry, `io_event`s at the return; never a buffer's
  bytes and never `execve`'s arguments or environment (its path only), so no secret in them
  reaches the trace (brief item 62). Not copied, and recorded in `not_seen`: the iovecs inside
  vectored AIO requests (those requests are left out and counted), the descriptors `recvmsg`
  receives and those `pipe` and `socketpair` return.
- *The child's id.* A thread's first call can come before its parent's `clone` returns, so the
  fork hook records the child, read as `task->thread_pid->numbers[level].nr` with every offset
  from BTF. The level is found by the probe itself when it adds the command, the index where
  the command's own `pid.numbers[k].nr` is its pid as the tool sees it: `/proc`'s `NSpid` counts
  levels only from the namespace `/proc` was mounted in, and WSL2 runs the distribution one
  below the kernel's own, where `NSpid` said 0 and the level is 1.
- *The command* is forked by the tool and held on a pipe until the probes are attached, the
  command seeded, and the reader running, then `execve`d (its path found on the `PATH` before
  the fork); its records before that `execve` are left out, so the text begins where strace's
  does. The tool is a child subreaper and waits for every process of the tree, as `strace -f`
  does.
- *Secure-exec.* A binary with file capabilities runs in glibc's secure mode, which strips
  `LD_PRELOAD`, `LD_LIBRARY_PATH`, `TMPDIR`, and others from its own environment, and is not
  dumpable, so its `/proc/self/environ` is root's: the command's environment is the strings the
  kernel placed (`env_start` to `env_end` of `/proc/self/stat`), which glibc's removals leave in
  place, so it is the user's.
- *The text.* A line is held until the next record: the call's return makes it whole, another
  thread's record cuts it (`<unfinished ...>`, then `<... resumed>`), as strace prints them;
  a string the entry could not read is filled from the return's copy before the line is
  written. Descriptors' paths come from the recorded opens, `dup`s, `fcntl` duplicates, forks
  (`CLONE_FILES` shares the table, `CLONE_FS` the directory), `chdir`s, and `exec`s (which
  close what closes on exec); `/proc/self` is the traced process's, and a directory's symbolic
  links are resolved when the capture is converted. Open flags are printed in strace's order
  (`src/xlat/open_mode_flags.in`), which the export keeps.
- *A pre-existing bug the equivalence test found, in the strace parser:* strace pads a short
  call's ` = ` out to column 40 (`rmdir("sub")  = 0`), and the parser looked for `") = "`, so
  every call whose line was shorter than 40 columns was dropped from every strace-based
  metric, export, and digest (an `rmdir` or `sync` of a short path; `-yy`'s paths make most
  lines longer). Fixed (`strace.rs` `ret_split`).
- *The tests' privilege.* Each eBPF test runs a copy of the built binary given its
  capabilities by `sudo -n setcap`: password-less in CI, here granted by the user once per
  build. `AEIOU_WITHOUT=ebpf` leaves them out of `dev.sh`, and says so.

Not built in phase 3: the `sudo -n` helper (phase 3b); the request heads by uprobe, mapping
faults, and `io_uring` (phase 4); aarch64 run only in the release's build (its tests run when
an aarch64 CI job exists).

### 3.97 No secret in a capture: capture less, not scrub after (brief item 62; designed and approved by the user 2026-11-04, built the same day)

**How it came up.** Queued by the user on 2026-11-03 (brief item 62). A capture is sent to
people who were not there, so no secret of its owner's may be in any of its files. Phase 2
already kept none in `requests.jsonl` (§3.96), and the eBPF source copies no `execve`
argument, but the strace line's `-v` printed every `execve`'s whole argument list and
environment into `trace.txt`, and `capture.json` recorded `argv` as given.

**What was found first.** No reader uses `execve`'s arguments or environment, or the contents
of any string or buffer: `strace.rs`, the digest, `applications.rs`, export, and compare read
paths, `-yy`'s annotations, sizes, results, and arrays (checked 2026-11-04). The brief's
worry that dropping `-v` loses "the environment the digest and a reader use" was wrong on
the readers' side, but `-v` cannot go: it is what prints an `io_submit`'s requests and a
`readv`'s iovecs past the 32nd. And a second leak the item had not named: strace's default
`-s 32` prints the first 32 bytes of every read, write, and send buffer, so a credentials file
read (`[default]\naws_access_key_id = AK...`), a configuration's password, or, under
`%network`, an `Authorization` header a client writes as its own iovec reached the trace in
part.

**Decided (the user approved all, 2026-11-04).**

- *Capture less, not scrub after.* The strace line keeps `-v` and adds
  `-e verbose=!execve,execveat` and `-s 0`: strace 6.8 prints `execve("/bin/x", 0x7ffe...,
  0x7fff...)`, the eBPF converter's own form, and every buffer as `""...`, with paths, `-yy`'s,
  and a 40-element `writev` whole (tried 2026-11-04). Nothing secret is taken, so there is
  nothing to scrub and no raw trace to keep for the owner beside a scrubbed one. The man
  page's line for a trace by hand, and every kit's, gets the same flags.
- *The command line kept, redacted.* `capture.json`'s `command.argv` (and the command at the
  end of `source.line`) keeps every argument but replaces by `<redacted>` the value of a
  `NAME=value`, a `--NAME=value`, or the argument after a `--NAME` (unless itself an option)
  whose name (letters, digits, `_.-`, at most two dashes, so a script's text is never taken
  for one) holds password, passwd, secret, token, key, credential, auth, signature, session,
  or cookie, and the password of any URL (`scheme://user:password@host`, in `--endpoint` too);
  `command.redacted` lists the places. Pattern matching misses a secret passed by position, so
  `--argv none` keeps the program's name alone; `recorded` is the default, since the arguments
  say what the job was.
- *A trace by hand warned of.* The parser notes a line that prints an `execve` argument list
  (`execve_arguments_printed`) or a call's bytes (`buffer_contents_printed`, a quoted
  argument after `, ` or `iov_base=`, or at a resumed line's start, not `""`; a `-yy` path
  escapes its quotes and a `sun_path="` follows no `, `), counted by call, and
  `trace_may_hold_secrets` in the metrics says to keep it and send its digest; `metrics` and
  `digest` warn on standard error after their output is written. The golden corpus prints
  bytes as strace does by default, so 260 of its files were regenerated, each difference
  checked by script to be these notes alone (`runner/aeiou/tests/strace_corpus.rs`).
- *Rewriting an old trace* (an `aeiou-trace scrub`) is not built: queued as brief item 63.

**Built (2026-11-04).** `capture::strace_line`, `scrub_argv`, `scrub_url`, `--argv`;
`strace::printed`, the notes and `SECRETS`; the warning in `aeiou_trace.rs`. Tests: the unit
tests of each; `tests/capture.rs` and `tests/capture_ebpf.rs` run a command with a secret in
its environment, its options, a URL, and the bytes it writes, reads, and sends over a socket,
under each source, and find none in any file of the capture; `tests/trace_tool.rs` the
warning and its absence.

**A decoded address is not a call's bytes (found 2026-11-05).** The strace capture of
LMCache's shared store (`workloads/kvcache/lmcache-shared-fs`, captures `2026-11-05-strace`)
was warned of, 20 calls with their bytes printed, under the capture's own line: every one a
`recvmsg` of a netlink socket, whose messages strace decodes under `-s 0` too, an IPv6 address
written `inet_pton(AF_INET6, "::1")` after a `, ` (a datagram's sender in `recvfrom` and
`recvmsg` likewise; `inet_addr("…")` and `if_nametoindex("…")` follow a `(`, and never
matched). `strace::printed` no longer takes a quoted argument after `inet_pton(AF_INET6, ` for
a call's bytes; one after it still is. Tests: the unit test's clean lines (a netlink listing,
a sender of each family) and a datagram whose bytes are printed beside its sender's address.

### 3.98 One trace format for every source: strace's text, extended by records of our own, and the probes of phase 4 (researched, debated with the user, and decided 2026-11-04; phase 4 step 1, `io_uring`, built the same day)

**How it came up.** After item 62 the user asked when the probes for `io_uring` and mapping
faults come (item 54 phase 4, nothing scheduled), noted that LMCache uses `io_uring` for some
operations, and that strace's text holds no object request either: what is needed is a
*general* extension of the trace format, not one for `io_uring` alone. With it two questions:
can strace capture all a POSIX AIO application does, and is every hook each backend needs
known. Researched the same day: a survey of prior art for such a format, the hook points per
mechanism from the kernel's BTF on this box (6.18.54.1-WSL2) and primary sources, and an
experiment of POSIX AIO under strace. Confidence tags: *box* (verified here), *src* (primary
source read), *rec* (recollection, unverified).

**POSIX AIO under strace (experiment, glibc 2.39, strace 6.8, box).** glibc implements
`aio_*` in user space (`rt/aio_misc.c`, src): `aio_read` and `lio_listio` queue the request in
glibc's memory and wake or create a helper thread (a `futex` or `clone3`, neither naming the
request); the helper calls `pread64`, `pwrite64`, or `fsync` (`aio_fsync`), and wakes the
waiter; `aio_error` and `aio_return` read the result from memory with no call, `aio_suspend`
waits on a `futex` only if the request is not done. So the storage's view is whole in the
trace (the helper's calls, with descriptor, offset, length, result), and the application's is
not: which thread issued a request, when (the wait in glibc's queue), that it was
asynchronous at all (a helper's `clone3` and `pread64` look like any thread's), how many were
outstanding, and when the result was collected. And glibc serializes by descriptor: 8 reads
queued at once on one descriptor ran one at a time on one helper; 16 over 4 descriptors ran on
4 helpers, 4 each (at most 20 helpers by default, `optim.aio_threads`, src). The user's reading,
sharpened: the I/O does reach the kernel, as an ordinary read of a glibc thread, and a syscall
tracepoint cannot tell whose request it serves. Not a limit of eBPF: uprobes on libc's
exported `aio_read`, `aio_write`, `lio_listio`, `aio_fsync`, `aio_error`, `aio_return`,
`aio_suspend` (box) see the issuer, the time, and the `aiocb`, joined to the helper's call by
the `aiocb`'s address or by descriptor, offset, and length. glibc's internal `__aio_notify`
cannot be probed: libc ships no symbol table (box).

**The hooks per mechanism (field lists from `trace_event_raw_*` and the structs in
`/sys/kernel/btf/vmlinux`, box; nothing attached, tracefs is root-only here).**

| Mechanism | Issue | Completion | Gap |
|---|---|---|---|
| `sync`, `sync-direct` | syscall tracepoints (built) | the same | `O_DIRECT` asked is not done: `iomap:iomap_dio_rw_begin`/`iomap_dio_complete` prove it on iomap filesystems (XFS, ext4) only; NFS goes through `nfs_direct_*` |
| `posix-aio` | uprobes on `aio_read`, `aio_write`, `lio_listio`, `aio_fsync`; the helper's call traced | uprobes on `aio_error`, `aio_return`, `aio_suspend` | the join by `aiocb` address |
| `libaio` | `io_submit` entry (its `iocb`s copied, phase 3); fentry `aio_prep_rw` and `aio_fsync`, the kernel's request (phase 4 step 3) | `io_getevents` exit is the reap; fentry `aio_complete_rw` (static, in BTF and kallsyms, box) the real completion, `ki_res.obj` the user `iocb` (box); built at `aio_complete`, every request's posting, `aio_complete_rw` the fallback (2026-11-04, "Phase 4 step 3") | `fs/aio.c` has no tracepoints (src); a completion in interrupt context has no meaningful thread, so joined by pointer; a buffered request completes inside `io_submit` (rec) |
| `io_uring` | tracepoint `io_uring_submit_req`: its record has no descriptor, offset, or length, but its argument is the `io_kiocb` after `prep` (`io_uring.c`, src), so a raw or BTF tracepoint reads `opcode`, `flags` (`REQ_F_FIXED_FILE`), `cqe.fd`, and `io_rw`'s `kiocb.ki_pos` and `len` (box) | tracepoint `io_uring_complete` (`req`, `user_data`, `res`, `cflags`, box) | under SQPOLL the issuer is the `iou-sqp` thread; a ring read without a call hides the reap; opens and closes inside the ring need fentry on `io_openat` or reading `io_open`; a fixed file's `cqe.fd` is an index, resolved from `io_uring_register(IORING_REGISTER_FILES)`'s array; `URING_CMD` opaque; a request punted to `iou-wrk` seen by `io_uring_queue_async_work` |
| `mmap` faults | fentry `handle_mm_fault(vma, address, flags, regs)` (box); the file offset `vm_pgoff·PAGE + (address − vm_start)` | fexit: `VM_FAULT_MAJOR` in the return | `exceptions:page_fault_user` is x86 only and has no exit; ~~filter `FAULT_FLAG_USER`~~ filter on `regs` set (a `get_user_pages` of an `O_DIRECT` buffer faults too, and passes none; the flag would drop a call's copy of a mapped buffer; 2026-11-04, "Phase 4 step 2"); a `VM_FAULT_RETRY` attempt summed with its retry (rec); the readahead a fault starts by `readahead:page_cache_ra_*` and `filemap:mm_filemap_*` (box), by inode |
| `gds` (cuFile) | uprobes `cuFileRead`/`cuFileWrite`, `cuFileBatchIOSubmit`, the `*Async` forms; or the `nvidia-fs` ioctls (`nvfs-core.h`, src) | uretprobe, `cuFileBatchIOGetStatus` by cookie | an asynchronous completion is a GPU stream's, untimed from the CPU; the handle mapped to a descriptor by `cuFileHandleRegister`; compatibility mode appears to use `io_uring` through `syscall()` (box, strings only); untested without the hardware |
| `nixl-posix` | libaio by default, liburing by option (README, src): the rows above | — | NIXL's own request identity needs uprobes on mangled C++ names |
| `libnfs` | uprobes `nfs_pread_async`, `nfs_pwrite_async`, `nfs_open_async` (API v2 adds a buffer argument) | the callback, an address known at run time | the sockets alone show the wire only; matching by XID would read payload bytes, which §3.97 forbids |
| S3 over TLS | uprobes `SSL_write`, `SSL_write_ex` in `libssl.so.3` (box), phase 2's head filter in the probe | `SSL_read`/`SSL_read_ex`, the response head | Go's TLS, rustls, a static OpenSSL: sockets only (§3.96) |

*Corrections of what was believed.* ~~The probes on kernel structures need CO-RE relocations,
loaded from a C-built object (§3.96).~~ No probe here needs CO-RE: phase 3 already takes
`task_struct`'s offsets from the kernel's BTF at load and hands them to the probe in a map, and
the same serves `io_kiocb`, `io_rw`, `kiocb`, `aio_kiocb`, and `vm_area_struct`, a missing
field failing the `source` check. Rust cannot emit CO-RE relocations yet (a pre-RFC of
2026-04-09, an LLVM prototype; `bpf-linker`'s `--btf` emits types, not relocations), and clang
is not installed. And `io_uring_submit_req` carries the request itself, so no probe on the
static `io_issue_sqe` (which may be inlined) is needed for its descriptor, offset, and length.

**Prior art for the format.** No format holds POSIX calls, `io_uring` submissions and
completions, AIO, faults, and object requests in one stream per task. DFTracer (LLNL, JSON
lines, MIT), Darshan DXT, and Recorder stop at intercepted library calls (POSIX, STDIO,
MPI-IO, HDF5), and DFTracer pairs no issue with its completion; CTF/LTTng, Perfetto, and OTF2
hold anything under a schema of one's own, behind a binary decoder or a C library (OTF2's
issued and completed I/O records, paired by a matching id, the best model of asynchronous I/O,
rec); Falco's scap and Tracee see the `io_uring` calls, not the requests in the ring (Tracee's
events a draft); strace's JSON output (a 2014 Summer of Code series) was never merged;
OpenTelemetry names S3's attributes (`aws.s3.bucket`, `aws.s3.key`) and nothing for files.

**The debate (the user, this side).** The user: an extension of strace's text that strace
parsers read, if they skip names they do not know, makes the strace source, where eBPF will
not load, a pure subset of what eBPF writes, so one format for every case; against it, JSON
lines would not depend on the vagaries of parsing text (quoted strings, spaces in paths), and
if JSON had a significant advantage the raw strace lines could be embedded in it, or
transformed into it by an awk script. This side, after a test of strace 6.8's escapes (box): a
quoted string is C-escaped, a `-yy` path is bounded by `<…>` with `>` and every byte not
printable ASCII in octal (`x\76y`, `\303\251`), a space literal, a newline `\n`, so one call is
one line and the text is unambiguous by construction; its cost is the size of strace's grammar
(each decoder's own layout, unfinished and resumed halves, signals, truncation, changes between
versions), not ambiguity. JSON's real advantages: typed and named fields, a new one breaking no
reader; a schema; every language reads it with no parser of its own; nesting. But the text
parser never goes away (the TWG's traces, anyone's `strace -f`, the fallback), so JSON is a
second format, not the one; JSON holding raw strace lines still needs the strace parser, and an
awk transform would be a second, weaker one (nesting, escapes, resumed halves) beside
`strace.rs` and its 260-file golden corpus; the fragility is in strace's lines, not in records
we write, which can keep to a strict, keyed subset; and JSON is about 1.5 to 2 times larger on
traces of 3.5 to 7 GiB. JSON's advantages for researchers are had by an export. The user
agreed.

**Decided (the user, 2026-11-04: "Agreed, strace text with JSONL export"; the build order, "no
objection").**

- *One capture format: strace's text, extended.* The strace source's output is a pure subset
  of the eBPF source's. Records strace never prints carry a reserved prefix (`aeiou_`, to be
  confirmed when built) and keep strace's line form, `TID HH:MM:SS.uuuuuu name(args) = ret
  <dur>`. A trace that holds them begins with a header line giving the format's version and
  the record kinds present.
- *Our records keep to a keyed subset of strace's syntax:* every argument `key=value`, never by
  position; a value an integer, a C-escaped quoted string, `fd<path>` with strace's escapes, or
  a nested `{…}` or `[…]`; its grammar written in `aeiou-trace(1)` and `runner/REFERENCE.md`
  §13, and each record kind held by the golden corpus. Names: `aeiou_uring_sqe`,
  `aeiou_uring_cqe`, `aeiou_aio_issue`, `aeiou_aio_complete`, `aeiou_fault` (faults as runs, a
  count per record, faults being the size risk), `aeiou_obj_req`, `aeiou_obj_resp` (OpenTelemetry's S3
  names), later `aeiou_cufile_*` and `aeiou_nfs_*`; for example
  `4711 10:00:00.000101 aeiou_uring_sqe(5</ring>, {op=IORING_OP_READ, fd=7</data/f>, off=4096, len=131072, user_data=0x7}) = 0`.
- *Pairing.* A completion carries its issue's key, ~~`user_data` and the ring~~ a number the
  converter gives the request while it is in flight (decided 2026-11-04, "Phase 4 step 1"
  below) for `io_uring`, the `iocb` or `aiocb` address for AIO, so a reader holds only the requests in flight, at most a
  ring's depth: memory per open file, never per call.
- *Object requests join the stream;* `requests.jsonl` becomes a view of it.
- *Readers skip a name they do not know:* `strace.rs`'s tables (`fd_op_of`, `path_op_of`) fall
  through for one, which a test confirms before anything relies on it.
- *JSON lines an export only:* `aeiou-trace export --jsonl`, a typed record per line, lossless,
  with a JSON Schema, so researchers read a trace with jq or pandas and never parse strace;
  Perfetto's trace JSON (flows for the pairs) a later export for viewers. Never a capture
  format.
- *The build order of phase 4:* (1) `io_uring` (LMCache); (2) mapping faults; (3) libaio's real
  completion; (4) the POSIX AIO uprobes; (5) the request heads by libssl uprobes, in place of
  the shim where it loads; (6) cuFile and libnfs last, needing hardware or a server not at
  hand. Each kind's records and their grammar come with its probes, and each is tested
  against the counts of the calls under it.

**Several nodes: no merged trace (the user, 2026-11-04).** A scale-out capture is one trace per
node, not per rank (the ranks on a node are lanes of its trace, §3.96), listed with each node's
source, losses, and clock offset in rank 0's `capture-set.json`. This side proposed, and the
user agreed: no merged capture format. A merged file breaks the one format (a thread id, a
descriptor, and a path are each one host's, so every line would need a host field strace never
prints); it writes one interleaving as fact where clock offsets are known only within half a
round trip; and it would copy 0.2 to 0.5 TiB of traces that stay on each node's disk by design.
So the set's directory is the unit a person keeps, archives, and sends (`tar --zstd` of it:
the many files a packaging matter, not a format's), and the readers (`metrics`, `digest`,
`compare`, `export`) take a set, find its nodes by `capture-set.json`, apply each node's offset,
and merge the lanes by time as they stream, memory per lane. Two events of different nodes
closer than the bound of their offsets are reported in *unknown order* (the user's choice over
picking an order and counting the cases). Grouping the nodes by like I/O (a symmetric job's
ranks against rank 0's checkpoint metadata and logs) is the digest's: a set digest that sorts
the nodes into classes by `compare`'s judgment, with counts, and is published in place of the
traces; the user's "curation" meant only keeping and preserving the trace, which the set's
directory serves. A merged timeline for a viewer is an export (`--jsonl`, Perfetto), where
JSON has room for the host and for an unknown order. Not implemented yet: a set read by any
reader, the set digest.

**Other programs on a node (the user, 2026-11-04: can their I/O be kept out of the trace).** It
is already, twice. Both sources trace the command's process tree alone: strace's `-f` attaches
to the command and its children only, and every eBPF probe returns at once for a process not in
`TREE` (`traced()`, the command added at its first call and each child at `sched_process_fork`,
before it runs); a daemon, a cron job, or an agent never enters either trace, whatever it
touches. And the readers count only calls under `--root`, so even a stray call to the root disk
is not the SUT's workload. `io_uring`'s `iou-wrk` and `iou-sqp` threads are threads of the
application's own process and so in the tree (src, rec, not tested here); kernel writeback,
readahead, and the NFS client's own RPCs are the storage's side, rightly outside the call
stream (mountstats has them). Found in the checking, two defects of the eBPF source: `TREE`
never removes an exited process, so an unrelated process given a recycled pid would be traced
(`pid_max` is 4,194,304 here, so rare, but a long capture of many short processes, a data
preparation's shell pipelines, makes it plausible); and `TREE` holds 65,536 processes, an
insert into a full one ignored, so a job starting more over its life had its later children
untraced in silence, against the rule that any loss is loud. *Decided (the user agreed to the
proposal, 2026-11-04):* a process leaves `TREE` when its thread group is dead (the last of its
threads exits: `signal_struct.live` at zero at `sched_process_exit`, read at offsets from the
kernel's BTF), not when its leader exits, which may be before its threads; a child `TREE` could
not hold is counted (`losses.untraced_processes`), makes the capture `complete: false`, and is
in the loss box; `capture.json` records the processes still in the tree at the end; and tests:
a process outside the command doing I/O under the same `--root` during a capture is in neither
source's trace, and no exited process stays in the tree. The opposite risk is the larger, and
no capture can detect it: the job's own I/O done by a process outside the tree (a server started
apart, Milvus or vLLM with LMCache, work handed to `systemd-run` or `docker exec`, a FUSE daemon,
a local cache daemon), so `CAPTURE.md` and the kits say to start everything that does the job's
I/O under the capture. Later, not now: a preflight report of other processes with files open
under `--root` (another user's need privilege to see) and a warning when `updatedb`'s pruned
filesystems do not cover the mount, since their I/O, though outside the trace, disturbs the
storage while it is taken.

*The tree's fix built (2026-11-04).* `capture-ebpf/src/main.rs`: `sched_process_exit` takes a
process out of `TREE` when `signal->live` is zero, and a failed insert counts `Loss.untraced`;
`ebpf.rs`: the offsets `task_struct.signal` and `signal_struct.live` from BTF,
`untraced_processes`, `incomplete`, the box, and `Probes::traced_at_end`; `capture.rs`:
`complete` and `processes_traced_at_end`; `aeiou-trace(1)` and `CAPTURE.md`. Tests: the unit
tests of the offsets, the count, and the box; `tests/capture.rs` and `tests/capture_ebpf.rs` run
an outsider writing under the same `--root` through a capture of a command that forks three
children, and find none of its calls in either source's trace, and under eBPF none of the four
processes left in the tree (the count read from the probes' own map). A full tree and a recycled
pid are not forced by a test (65,536 live processes; a chosen pid needs root): the first is
held by the unit test of its count, the second by the tree found empty.

**Phase 4 step 1, `io_uring` (proposed and agreed with the user 2026-11-04: "proceed with the
recommendations").** Settled before building, with this side's recommendation each time:
- *Pairing by the converter's number, not `user_data`* (a change to the decision above).
  Applications reuse `user_data`, 0 for requests nobody waits on is common, and the request's
  own address is unique in flight but is a kernel address, not to be published. So the
  converter numbers each request by its address (`id=`), the smallest number free, and frees it
  at the last completion (multishot: while `IORING_CQE_F_MORE`). `user_data` is printed.
- *Whose line.* A submission is the submitting thread's (under SQPOLL the kernel's thread, the
  issuer lost, said in `not_seen`); a completion is printed under its submission's thread,
  since it may be posted from an interrupt or an io-wq worker; no `<duration>`; an error as
  strace prints one.
- *The header* only in a trace that may hold the records (the probes attached); a plain strace
  is format 0, headerless.
- *Operations:* reads and writes (fixed, vectored, multishot), `FSYNC`, `OPENAT`, `OPENAT2`,
  `CLOSE` decoded, any other by its name and file.
- *Fixed files in step 1:* the descriptors `io_uring_register` registers copied, up to 1,024 a
  call, and a direct open's slot filled, so a registered file's index has its path.
- *Not attached* where the kernel lacks the tracepoints or its structures differ: the capture
  goes on, `capture.json` says why, and `not_seen` keeps strace's line on `io_uring`.
- *Readers:* a read or write at its completion, with the bytes it moved at its offset; one
  `io_uring_enter`'s a fan-out, exported as one `submit`; an open, a close, an fsync as calls;
  in-flight depth per ring later.
- *Tests first:* the readers skipping unknown records; each record's decoding; golden cases;
  a capture of this runner's own `--posix io_uring` engine against `--posix sync`.

*Built (2026-11-04).* `capture-ebpf/src/main.rs`: `io_uring_submit_req` (the request's opcode,
flags, descriptor, `user_data`, ring, and first 64 bytes, the operation's structure, decoded by
the converter at offsets from BTF the header carries; an open's or a statx's path from the
kernel's `struct filename`; the ring added to `RINGS`); `io_uring_complete` (traced by ring,
whatever thread or interrupt posts it, its record built on the program's stack so it cannot
corrupt the per-CPU buffer of a call's probe it interrupted); `uring_calls` on `sys_exit` (the
ring at `io_uring_setup`'s return, found through the descriptor table, and the files
`io_uring_register` registers). `ebpf.rs`: `uring_layout` from BTF (the tracepoints' prototypes
by their `btf_trace_` typedefs, the offsets, the operations' structures checked to fit 64
bytes), `untraced_rings` among the losses, `capture.json`'s `uring`. `ebpf_text.rs`: the
records, the header, the rings and their registered files. `strace.rs`: the header
(`trace_format_newer`), the records read as their calls, `io_uring_setup` no longer noted in a
trace that holds its ring's requests. Grammar: `runner/REFERENCE.md` §13, `aeiou-trace(1)`.

*Found in the building.* The verifier refused `sys_enter` with `io_uring_register`'s copy added
(a million instructions explored: the copy's branches multiplied by the six strings'), so the
copies went to a small program of their own and the call programs stayed as they were; it
refused a load from the tracepoint's arguments at an offset chosen at run time, into which the
compiler had folded "the request is argument 0 or 1"; and the kernel refuses to attach a raw
program that reads past its tracepoint's last argument (EINVAL), so each tracepoint form has its
own program, the loader attaching the one BTF gives (`io_uring_submit_req` with the request
alone, or after the ring before about 5.19, rec; `io_uring_complete` with the posted cqe, or
its fields one by one). On a real trace: the request's descriptor is the kernel's only for an
operation that takes a file (`needs_file`), otherwise a leftover (`fd=0</dev/null>` was printed
for every statx), so it is printed for those alone. *Beyond the agreed set:* `STATX` decoded
(directory, path, flags, mask), read as strace's `statx` is, an `fstat` on a descriptor, since
this runner's engine does every `fstat` so and the backends' traces disagreed without it; and an
operation on paths no record holds (`UNLINKAT`, `RENAMEAT`, `MKDIRAT`, ...) noted by name
always, since whether it is under a root cannot be judged. In the export, a request at offset
-1 (the file's position, which the kernel reads and moves) is the sequential `read` or `write`
it is, the position tracked, so a replay moves its own; a vectored request, whose length the
record does not hold, and a multishot's results are left out of the export and noted, as the
eBPF source's vectored AIO is. And the dry run counts the lengths an abstract asks, a read past
a file's end included (npy_per_sample's smoke set: 219,885,683 asked, 146,774,015 returned,
strace and eBPF agreeing to the byte), so a run's trace is held to the dry run by at most, never
equal. The `depth_not_computed` note is an `io_submit` trace's alone (`io_uring`'s rounds are
not chained).

*Tests.* Unit: the header and an unknown record skipped; every record form converted
(`ebpf_text.rs`) and read (`strace.rs`), a request at the position, a multishot, the export's
exclusions; the BTF layout and the programs chosen per kernel form (`ebpf.rs`). Golden: three
cases (every record form written out, and two captures of the runner's `io_uring` engine with
and without SQPOLL). Privileged (`tests/capture_ebpf.rs`): `aeiou run --posix io_uring` of
npy_per_sample's smoke set, and with `--sqpoll 10`, captured under eBPF, reads the same bytes of
the same files with the same opens, closes, reads, `fstat`s, `stat`s, request sizes, and
popularity as `--posix sync`, every completion matched; the seven earlier eBPF tests unchanged.
Not tested here: fixed files and direct opens from a real application (the runner registers
none; the converter's unit test holds them), an older kernel's tracepoint forms (CI's kernel may
differ from this box's), `io_uring` from an application other than this runner ~~(LMCache's is
the next to trace)~~ (LMCache's fs:// and local-disk backends make none: the note of 2026-11-05 below). Brief item 54.

*Risks.* A static function the compiler inlined fires no fentry (the tracepoints preferred);
field meanings drift between kernels (`io_kiocb.cmd` at offset 0 only since about 5.19, rec;
`len` counts iovecs for `READV`); `io_uring` doubles the records a request (submit and
complete), which makes the 256 MiB `--buffer` question (§3.96) sooner; aarch64 has no
`exceptions` tracepoints; nothing could be attached on this box without root. Brief item 54.

**Phase 4 step 2, faults on mappings (proposed, debated, and agreed with the user 2026-11-04:
"I agree, proceed").** This side proposed `fentry`/`fexit` on `handle_mm_fault`, the faults
folded into runs by the converter, the export left as it is, readahead left out. The debate
settled four points and a theme for judging such questions.
- *Runs end at a quiet time, not only at a break in the stride* (the user). Without a cap on
  the quiet time between two faults, a scan, a stretch of compute, and another scan that
  happens to continue at the next offset read as one extent: a different workload (two bursts
  with think time between them, storage idle in the gap). So a run ends when a fault starts
  more than `--fault-gap` (1 ms by default) after the last ended, measured from one fault's
  end to the next's start, so a fault's own time on the disk is not counted as quiet; the
  value is in the trace's header, since a run cannot be split again after the capture; and a
  run records its longest quiet time inside (`gap_max_ns`), so a reader tells a tight run from
  one that only just stayed under. Agreed on every point.
- *Major against minor is not the workload* (the user). A minor fault is a page already in the
  cache mapped (no I/O, about a µs); a major one is a page read from storage with the thread
  waiting, a synchronous read of the kernel's size, not the application's. Which a touch is
  depends on the capture host's memory, its other programs, and its readahead: the touch is
  the application's, its kind the host's. The user: "whether the application reads those
  pages is what matters"; a bad readahead saturates a system under test sooner, and that is
  the penalty, by its saturation point, never by a score. So one run holds both kinds, with a
  count of majors (kept as evidence of the capture's conditions: a capture said cold that shows
  no majors was warm), and each fault's time (a minor fault on a page still being read in is
  slow and is not marked major, rec).
- *Readahead is the system under test's, not the application's* (the user's statement of it,
  this side agreeing). Pages read ahead and never read are not the application's I/O, and are
  not monitored or scored as waste; pages read ahead and then read are, and the system under
  test that read them early is credited by the lower latency it delivers. So readahead never
  enters an abstract or the export. The kernel does tell the two apart (`readahead:page_cache_sync_ra`
  for a page asked for, its window beyond the page speculative; `page_cache_async_ra` all
  speculative; whether a speculative page was later read by `filemap:mm_filemap_add_to_page_cache`
  and `mm_filemap_delete_from_page_cache`; the tracepoints present in this kernel's BTF, box),
  which is diagnosis, for a later step: why a replay and its capture differ on one host.
- *The theme (the user, this side agreeing):* the abstract holds what the application demands;
  everything the storage stack does in response is the system under test's own behavior, judged
  only by the performance it delivers under that demand, never by how clever or wasteful it is.
  Its inverse, this side's point and the user's restatement: a capture measures the
  application's demand and think time, not the capture host's performance, so the time between
  an I/O's end and the next's start is wanted and the time an I/O took on the capture host is
  not. Consequences recorded: (1) the capture host is a system under test too, and its policy
  shapes the trace: readahead does not hide a touch (a page read ahead still faults, minor, at
  its first touch), but fault-around does (the cached pages beside a faulting one are mapped and
  never fault), so fault-around is to be one page in a capture; a large-memory host hides a
  re-touch that a smaller one would fault again (a blind spot); (2) the theme holds only if a
  replay goes through the application's path: a mapping's read-around is not `pread`'s
  readahead, and the export's one read of a mapping's whole range is another workload, kept only
  because it is what the `trace` node can run today (brief item 65); (3) readahead is in the
  client too (`read_ahead_kb`, NFS's `rsize` window), the solution's by the interposition test,
  so a run and a capture should record it, and `fault_around_bytes`, as host facts, whether a
  submitter may tune them the WG's rule (brief item 66); (4) the duration of an I/O on a
  capture host is still recorded, for three reasons: the quiet time needs the end; a gap is
  think time only where the thread waited on nothing else (a loader's consumer waits on its
  producers, a rank at a collective), so a gap is evidence of structure and the abstract's
  compute is its case's `step_time`, never a gap copied; and an application whose requests
  follow its latency (the AWS CRT's parallel ranged GETs sized by measured throughput, adaptive
  prefetch, timeouts) has a demand that is partly the capture host's speed, which only the
  durations show and the digest should flag (not built); the depth an asynchronous application
  reaches is likewise a lower bound of what it allows.
- *`fault_around_bytes` one page, required* (the user: "mandating"). A host-wide setting, in
  debugfs, root's alone. Found before building: debugfs is mode 0700 (box), so a capture under
  the file capabilities can neither read nor write it. So the requirement is a check,
  `fault-around`, of what it is for rather than of the setting: the capture maps a cached file
  of its own (unnamed, `O_TMPFILE`, under `-o`'s nearest directory), touches a page and the
  one beside it, and counts the second's faults with `getrusage`, which needs no privilege and
  ~~catches large folios too~~ (it did, so it refused where no setting helps: corrected below, "Found at the grant"); on this box at the default, page 1 does not fault and page 20 does
  (box). Root sets the value to one page, checks again, refuses if it still does not fault, and
  puts the value back when the command ends (and on a panic); anyone else is told the two lines
  to run as root, or to capture as root; `--accept fault-around` waives it (the capture's one
  way to waive anything, not a flag of its own), and the record, `not_seen`, and a line in the
  trace say the faults are the host's shape. The owner's cost: every program on the host faults
  more while it lasts.

*Built (2026-11-04).* `capture-ebpf/src/main.rs`: `fault_entry` on `handle_mm_fault`'s entry
keeps a fault of a traced process that an exception raised (`regs` set: a load, a store, or a
call's copy of a mapped buffer; a `get_user_pages`, an `O_DIRECT` buffer's pinning, passes
none), on a mapping of a file (`vm_file`) that is not code (`VM_EXEC`), reading the mapping's
start, `vm_pgoff`, and the file's inode and device at entry (a fault that drops the mmap lock
may return with the mapping gone) into `FAULTS` by thread; `fault_exit` writes the record with
the `VM_FAULT_` bits (the context's slot after the four arguments, so no helper newer than the
trampolines) and the end. `ebpf.rs`: `fault_layout` from BTF (`handle_mm_fault`'s parameters by
its `FUNC`'s prototype, the offsets, the `FAULT_FLAG_` and `VM_FAULT_` values from their enums,
read now by `Btf`), the programs loaded and attached apart, so a kernel without trampolines
keeps the capture going without faults, said why; `Layouts` in place of `io_uring`'s alone.
`ebpf_text.rs`: each process's mappings of files kept from its `mmap`s, `munmap`s, and
`mremap`s (shared by threads, copied at a fork, emptied at an `exec`), a fault's file the
mapping its address is in, its offset the kernel's; a retried fault (`VM_FAULT_RETRY`) summed
with its retry at the same address; the runs (one thread, one file and inode, one kind, read
or store, from an exception or in a call, one stride, no quieter than the gap), a run written
at the thread's next record of another kind (so a thread's lines keep their order), at a loss,
or once no fault could continue it (a thread gone quiet), stamped with its first fault's start,
so its line can follow lines of other threads stamped later; the header's records and the line
of a waived check. `strace.rs` and `metrics.rs`: `aeiou_fault` counted apart from the calls, the
document's `faults` (faults, majors, stores, in calls, runs, bytes, each run's span in bytes and
its faults, the strides by faults, the quiet time between a thread's runs on a file in µs) and
the note `faults_kernel_observed`; the export's `faults_not_replayed`. `capture.rs`: the check,
`FaultAround`, `--fault-gap`, `capture.json`'s `faults`, `NOT_SEEN_FAULTS`. Grammar:
`runner/REFERENCE.md` §13, `aeiou-trace(1)`.

*Found in the building.* `decode` indexed a record's fields past a short record's end (a panic
on a corrupt `syscalls.bin`; the unit test of an unknown kind used 9, now `FAULT`): every kind's
fixed part is checked first. The probes' filter is `regs`, not `FAULT_FLAG_USER` as the hooks'
table said: the flag is set only for a fault in user mode, which would drop a call's copy of a
mapped buffer (a `write` from a mapping), the application's touch; `in_call=1` marks those.

*Found at the grant: large folios (the user decided 2026-11-04, "option 2").* With
`fault_around_bytes` at one page (set by the user for the test), a file cached by one large
write still took one fault per 2 MiB (4,096 pages read, 8 faults), and the same file written a
page at a time one per page (4,096 faults; box, 6.18, ext4): a fault maps the whole folio its
page is in, and the page cache's folios are as large as the writes or the readahead that filled
them made them. So no setting makes every first touch fault: fault-around is one cause, with a
knob, and large folios another, the capture host's history. The check as first built wrote its
file in one write, so it measured both and refused every capture, as root too, with no remedy
but its waiver. This side offered three ways (the check measuring fault-around alone, the
folio granularity recorded as a limit; that and a later probe of the pages each fault maps;
the mandate dropped for host facts) and recommended the first; the user chose the second.
Built: the check's file written a page at a time, so it measures fault-around alone (the
mandate stands for the knob that has one); every capture that traces faults says in `not_seen`,
and the metrics' note, that a fault maps the whole folio its page is in, up to 2 MiB, so a run is
where touches began at the granularity the kernel mapped; the probe of the pages each fault
mapped queued (brief item 67). The test's program writes its file a page at a time, so the
check's verdict decides what it asserts.

*Found in CI 2026-11-05: the setting changed under a capture (the user chose "1 and 2
together").* `fault_around_bytes` is the host's, and a capture as root sets it to a page for its
command and puts it back when the command ends. Phase 4 step 4's tests are the first to capture
as root, and in CI one of them (`every_posix_aio_function_is_traced`) ran beside the faults test
under the file capabilities: the root capture's setting let the faults test's check pass, and its
end put 65,536 back while the faults test's command ran, so 128 pages faulted as 8 at a 64 KiB
stride and a capture recorded `first_touch_observed: true` of faults fault-around had shaped. A
real capture meets the same thing from another capture as root on the node, or an
administrator. Built: (1) the tests that change the setting or depend on it hold one lock (a
`flock` on a file of the temporary directory, across nextest's processes); (2) a capture whose
check passed at its start takes it again when the command ends, before root puts the setting
back, and where it fails says so on the terminal, records `faults.first_touch_at_end: false` and
`first_touch_observed: false`, puts the reason in `not_seen`, and the converter writes the
trace's line `a first touch of a page may not fault (... the host's fault-around changed during
the capture)` (noted `faults_first_touch_unseen`), which `--convert` writes again from the
record (the header of `syscalls.bin`, written at the start, cannot hold it). Two captures as root
on one host are left to (2) and the `alone` check, not shared: the one that loses the setting
says so. A change made and undone inside one capture is not seen (taken at the start and the
end alone).

*Risks.* `handle_mm_fault` inlined or renamed in a kernel (its parameters checked by name from
BTF, the capture going on without faults); `vm_flags` and the fault enums' values drift between
kernels (read from BTF, but `VM_EXEC` is a `#define`, its value 0x4 since 2.6); a fault on a
mapping the converter does not know (an inherited mapping it saw no `mmap` of: none, since the
trace starts at the command's `exec`) is counted, not printed; a run of two faults at a random
pair of offsets reads as a stride; faults on huge mappings of many files under one thread are
one record per run, the size risk §3.98 named.

**Phase 4 step 3, libaio's completions (proposed by this side and agreed by the user
2026-11-04: "Proceed").** What the eBPF source saw of libaio before: `io_submit`'s requests
copied at entry and `io_getevents`'s results at exit, the readers pairing a result with its
request by `(context, aio_data)` in submission order. Four gaps, this side's: (1) when a
request completed is unknown, only when the application collected its result, and buffered
kernel AIO mostly completes inside `io_submit` (the call blocks), so an application can look
asynchronous when it is not, demand-side information under the theme of step 2; (2) a result
read from the context's ring in user space, with no call (fio's `userspace_reap`), is not seen,
and its request was taken as complete with the bytes it asked; (3) `aio_data` is often reused
(0), so a short read's result could go to another request; (4) a vectored request has no
length in the record. Proposed and agreed: an `fentry` of `aio_complete(struct aio_kiocb *)`,
every request's completion as the kernel posts it to the context's ring (reads, writes, fsync,
poll), `aio_complete_rw(kiocb, res)` where the kernel's BTF has no `aio_complete` (inlined:
reads' and writes' alone); the context's `user_id`, the request's `ki_res.obj` (the
application's `struct iocb`), and the result read at offsets from BTF; ~~a completion traced by
its context, which an `io_submit` of a traced process adds to a map (a completion may run in an
interrupt), keyed by `user_id`, unique within a process only, a collision with an untraced
process's context at the same address accepted as rare (its completions match nothing and are
counted);~~ (superseded: "Found at the grant" below) `io_submit`'s probe copying each request's address with its structure (~~the key~~: corrected
below, "Found at the grant"); records `aeiou_aio_issue` and `aeiou_aio_complete` (the names §3.98 reserved, `aeiou_aio_issue` holding
the key alone, the request's fields staying in `io_submit`'s line); durations recorded, not
reported as metrics in this step.

*Built (2026-11-04).* `capture-ebpf/src/record.rs`: `AIO_ISSUE` and `AIO_DONE`, one record
`AioRec` (the kernel's request, never printed; the context's `user_id`; `aio_data`; the result),
`Config`'s `aio`, `ki_ctx_off`, `ki_res_off`, `ctx_user_id_off`. `capture-ebpf/src/main.rs`:
`fentry`s of `aio_prep_rw` (every read and write as `io_submit` prepares it, in its array's
order) and `aio_fsync` in the submitting thread, the request added to `AIO_REQS` (an LRU map of
65,536, so a request prepared and then refused, which never completes, cannot fill it), and of
`aio_complete` or `aio_complete_rw`, a completion recorded only for a request in the map, the
records built on the program's stack, as `io_uring_complete`'s. `ebpf.rs`: `aio_layout` (the
functions by their `FUNC`s' prototypes, `aio_kiocb` beginning with its `rw` and its `fsync`, the
offsets), `aio_programs`, `load_aio`, the attach (a kernel that refuses a program captures on
without them, said why), the header's `aio` (`complete`, `fsync`). `ebpf_text.rs`: a thread's
issues inside its `io_submit` kept until the call's exit, then, after its line, an
`aeiou_aio_issue(ctx=0xC, id=N, member=M) = 0` for each of the first `ret` that a completion
will come for, stamped with the call's start, `member` its place in the line's array (one the
line leaves out, vectored or past the 32 copied, waited for in silence); requests in flight by
the kernel's request, numbered from `io_uring`'s pool; a completion posted before its call
returned held until that call's line, then written `in_submit=1`; an `aeiou_aio_complete(ctx=0xC,
id=N, data=0xD, in_submit=0|1) = RES` under the submitting thread, `RES` an error as strace
prints one; the conversion's `aio_requests`, `aio_completions_unmatched`,
`aio_completed_in_submit`, `aio_in_flight_at_end`. `strace.rs`: a trace whose header lists
`aeiou_aio_complete` counts a request at its completion with the bytes it moved, paired through
its issue's number, `io_getevents` collecting only; the notes `aio_completed_in_submit` and
`aio_completed_never_reaped` (completions less the results calls collected), a request with no
completion `aio_results_assumed_complete` as before; the export's members get their
completions' results. `capture.rs`: `capture.json`'s `aio` (`traced`, `why`, `completions_at`),
the source line, `not_seen` (strace's line on libaio's completions, new, left out where they are
traced; the fallback's own). A trace without the records reads as before.

*Found at the grant: the application's `struct iocb`'s address is no key (the user agreed
the change 2026-11-04, "proceed").* As first built, the completion named its request by
`ki_res.obj` and the probe of `io_submit` copied each request's address; a context filter
(`AIO_CTXS`, filled by a program on `sys_enter` at each traced `io_submit`) chose the
completions. The capture of this runner's `--posix libaio --cache direct` matched 1,606
requests and left 440 completions unmatched: the kernel copies the structure at submission, so
an application may reuse its memory at once, and this runner's engine builds its requests in a
reused buffer, one address in flight for several requests (a decode of `syscalls.bin`, box).
Buffered, every completion was inside its own call and the reuse did not show. So requests are
paired by the kernel's own (`aio_kiocb`), recorded at `aio_prep_rw` and `aio_fsync` as the
call prepares them, the issue probe's map choosing the completions in place of the context
filter (gone, with its table and its loss box); a request's place in the array is the order of
its call's issues. Cost: `aio_poll` is inlined here, so a poll request is not traced (it was
never counted); a kernel that inlines `aio_prep_rw` traces no completion, said why. Found with
it: `aio_prep_rw` takes a third parameter, `rw_type`, on this kernel (box); the loader requires
the request first and no more. At the second grant, buffered: the kernel frees a request it
completed inside the call, and the call's next request takes its address (one address four
times in one `io_submit`; 1,124 of 1,606 completions unmatched), so the converter keeps each
of a call's issues with the completion posted after it, the latest unanswered issue of that
request answered, and writes a completion inside its call with its issue's number, never in
the requests in flight.

*Tests.* Unit: the layout from this kernel's BTF and every refusal of another's (`ebpf.rs`;
with it, tests of `io_uring`'s other tracepoint forms and of the source check's wording, which
had none); the records written (`ebpf_text.rs`: a completion inside its call, one after, an
error, a vectored request, the kernel's request reused inside one call and after, a completion
of no request and of one the call did not take, a lost exit, the fallback's fsync, no header);
the reader (`strace.rs`: counted at completion through the issue's number with `aio_data`
reused and completions out of order, the notes, the export's results, and the same calls
without the records misread by `aio_data` as before). Golden: two cases, the record forms by
hand and a capture of this runner's `--posix libaio --cache direct` (npy_per_sample at 20
samples, 322 requests). Privileged (`tests/capture_ebpf.rs`): `aeiou run --posix libaio` of
npy_per_sample's smoke set through the page cache and with `--cache direct`, captured under
eBPF, reads as `--posix sync` with the same cache; 1,606 requests each answered, none unmatched
or in flight; through the page cache all 1,606 posted inside their `io_submit`, direct none
(box, WSL2 6.18, ext4). Not tested here: a kernel without `aio_complete` or `aio_prep_rw` (the
layout's unit tests hold the refusals), an application that reads results from the ring in
user space (fio is not installed; the reader's unit test holds the note), poll requests.


*Found in the building.* `aio_complete` is called when the request's last reference drops:
`io_submit_one` holds one until it has submitted the request, the completion the other
(`iocb_put`, `fs/aio.c`, rec), so the record's time is when the result is posted, the later of
the completion and the submission's end, which is the moment the application could collect
it; `aio_complete_rw`, the fallback, is the completion itself. ~~A vectored request counted with
the bytes its completion moved~~ (proposed): not built, since the reader takes a request's
file, offset, and asked length from `io_submit`'s line, which leaves a vectored request out
(its iovec array is not copied, `NOT_SEEN_EBPF`); its completion is consumed and nothing is
written, as before. The `json!` of `capture.json` reached the macro's recursion limit with one
key more: `aio` is set after it.

**Phase 4 step 4, POSIX AIO (proposed by this side and agreed by the user 2026-11-05:
"proceed").** The experiment above, repeated with writes, an `O_APPEND` file, `aio_fsync`, and
`lio_listio` (glibc 2.39, box): the helper's calls are `pwrite64` at the `aiocb`'s offset (on
an `O_APPEND` file too, where the kernel ignores it), `pread64`, and `fdatasync` for
`O_DSYNC`; one descriptor's requests run one after another on one helper; an idle helper waits
about a second and exits. The storage's side is whole in either source; the application's (who
issued, when, how long the request waited in glibc's queue, how many were outstanding, when the
result was collected) is in neither, and both count the helper's call as a read of a thread
the application never made. Proposed and agreed:

- *Probes:* a uprobe and a uretprobe of each of `aio_read`, `aio_write`, `aio_fsync`,
  `lio_listio` (the issue; the `struct aiocb`'s descriptor, opcode, buffer, length, offset, and
  notification copied, a list's first 32), `aio_return` (the collection), `aio_suspend` (the
  wait, its list and timeout), and `aio_cancel`; not `aio_error`, which applications poll in a
  loop. On the exported functions: libc has no symbol table and no USDT note for AIO (box). The
  `*64` aliases share their addresses. Attached system-wide by file, filtered by `TREE`.
- *The library:* the `libc.so.6` the capture itself runs on (the system's), and `librt.so.1`
  beside it for glibc before 2.34; glibc's 64-bit `struct aiocb` (`aio_fildes` 0,
  `aio_lio_opcode` 4, `aio_reqprio` 8, `aio_buf` 16, `aio_nbytes` 24, `aio_sigevent` 32,
  `aio_offset` 128, box on x86_64; aarch64 the same by the ABI, rec). A program linked
  statically, or on another C library (a container's, musl's, whose `aiocb` and helpers
  differ), is not traced, and the converter names the C libraries the command mapped that the
  probes were not on. musl not built.
- *Records:* `aeiou_aio_read`, `aeiou_aio_write`, `aeiou_aio_fsync`, `aeiou_lio_listio` (the
  issue, named for the library's call, since no system call carries it as `io_submit` carries
  libaio's), `aeiou_aio_serve` (before the helper's call that serves a request: the helper's
  line itself unchanged, so the strace source's trace stays a subset), `aeiou_aio_suspend`,
  `aeiou_aio_return`, `aeiou_aio_cancel`; `aeiou_aio_issue` and `aeiou_aio_complete` stay
  libaio's.
- *Pairing, in the converter:* an issue to the helper's call by process, descriptor, buffer,
  length, and offset among the requests in flight (an fsync by descriptor and kind, the earliest
  first); `aio_return`, `aio_suspend`, and `aio_cancel` by the `aiocb`'s address, unique while
  its request is in flight (POSIX forbids reusing it before), so unlike step 3's the address is a
  key here, while the request is in flight only; a request whose `aiocb` is issued again before
  `aio_return` collected it is done uncollected; a helper's call that matches nothing is an
  ordinary call; one that matches several takes the earliest, and is counted. The numbers are
  the pool `io_uring` and libaio take theirs from.
- *Readers:* a request counted once, in the issuing thread, when the helper's call returns,
  with the bytes it moved; the helper's call not counted again; notes for the requests refused,
  canceled, never served, never collected; durations (glibc's queue, the call) recorded, not
  reported as metrics; each issue a `submit` of the export in the issuer's lane. A trace without
  the records reads as before.

The user's questions answered by agreement: whether `cap_perfmon` lets the capture attach a
uprobe (~~believed, Linux 5.8 and on; the first grant to tell~~ it does not: "Found at the grant"
below), `aio_error` left out (an
application that polls it and never calls `aio_return`, which POSIX requires, shows its
requests served and never collected), and musl and static programs said untraced only.

*Built (2026-11-05).* `capture-ebpf/src/record.rs`: `PAIO_ENTER` (an `Enter`, `nr` the
function), `PAIO_EXIT` (an `Exit`), the functions' numbers and names, `AIOCB_SIZE` (136, to
`aio_offset`'s end), `AIOCB_LIST`. `capture-ebpf/src/main.rs`: `paio_entry` (the `aiocb` of
`aio_read`'s and `aio_write`'s first argument and `aio_fsync`'s second, a `lio_listio`'s list
and its entries one after another, a null entry or one not read marked `LIO_NOP` so each keeps
its place, `aio_suspend`'s list and timeout), `paio_return` (the result, on the program's
stack), a uprobe and a uretprobe of each function by a macro. `ebpf.rs`: `paio_layout` (the C
library from `/proc/self/maps`, canonical; `librt.so.1` beside it if present), `paio_layout_of`
(each function from the first library that defines it), `elf_functions` (the `.dynsym`'s
functions of those names at file offsets, one per address, so `lio_listio`'s two, its version
before glibc 2.4 a function of its own, box), `load_paio`, the attach by absolute offset
(`UProbeScope::AllProcesses`), the header's and the source check's `posix_aio`. `ebpf_text.rs`:
`paio_enter` (an issue's line at the function's entry, since a helper may serve it before the
function returns; a `lio_listio`'s members in one line, a null or `LIO_NOP` entry left out),
`paio_serve` (at the helper's call's entry: a `pread64` or `pwrite64` by descriptor, buffer,
length, and offset; after `EINTR` the same call again, after `ESPIPE` a `read` or `write` with
no offset, glibc's fallback for a pipe or socket, rec; an `fsync` or `fdatasync` by descriptor),
`paio_exit` (`aio_return`'s, `aio_suspend`'s, and `aio_cancel`'s lines at their return, stamped
with their entry; a refused issue as `aeiou_aio_refused(id=N) = -1`, a fifth record kind found
needed in the building; `aio_cancel`'s requests of its descriptor or `aiocb` no helper had taken,
its return printed as strace prints a return it decodes, `0 (AIO_CANCELED)`, since the readers
take a return as a number), the conversion's `posix_aio_requests`, `_served`, `_ambiguous`,
`_refused`, `_canceled`, `_never_collected`, `_in_flight_at_end`, `_libraries_not_traced`.
`strace.rs`: an issue held by its number; `aeiou_aio_serve` marks the thread's next read,
write, or sync call as the request's, counted in the issuer (an `O_APPEND` write at the file's
known end; an fsync as the issuer's call), an interrupted one waiting for the next; the notes
`posix_aio_refused`, `_canceled`, `_never_served`, `_never_collected`; the fan-out of each
issue's data requests; the export's `submit` per issue, its members' results from the helper's
calls. Found in the building: the export wrote a request refused, canceled, or never served as
a member taken as complete, which a replay would do though the application's moved no bytes;
`export.rs` gained `member_dropped` (the first pass names the member, the second leaves it out,
and a `submit` with none left, the header's `lines` counting what is written), noted
`not_exported:posix_aio:refused`, `canceled`, `never served`. `capture.rs`: the header's `posix_aio`, the source line, `capture.json`'s `posix_aio`
(`traced`, `why`, `libraries`), `not_seen` (strace's line on POSIX AIO, new; eBPF's two,
`NOT_SEEN_PAIO`, in its place).

*Found at the grant: a uprobe needs `CAP_SYS_ADMIN` (decided with the user 2026-11-05, "1 now,
queue 2 on the brief").* Under `cap_bpf,cap_perfmon` the 14 programs passed the verifier and the
first attach failed in `perf_event_open`: Linux 6.18's `perf_uprobe_event_init` checks
`capable(CAP_SYS_ADMIN)` (`kernel/events/core.c`, src), where `perf_kprobe_event_init` moved to
`perfmon_capable()` in 5.8; this side's belief was wrong. A `uprobe_multi` link (Linux 6.6,
`BPF_LINK_CREATE`) needs only `CAP_BPF` and `CAP_PERFMON` (`bpf_uprobe_multi_link_attach`, src),
but its program must be loaded with the expected attach type `BPF_TRACE_UPROBE_MULTI`, which aya
0.14's `UProbe::load` never sets. Options put to the user: (1) POSIX AIO traced as root alone,
said why under file capabilities; (2) `uprobe_multi` links through a patch of aya, (1) below 6.6;
(3) `cap_sys_admin` in the setcap line, against the brief's choice of the least privilege. The
user chose (1), and (2) queued (brief item 68). Built: the probes are neither loaded nor
attached without `CAP_SYS_ADMIN` in the effective set, `posix_aio.why` saying so
(`PAIO_NEEDS_ROOT`), and the trace's POSIX AIO is then strace's (its `not_seen` line kept); an
attach error prints its whole chain of causes. The POSIX AIO tests capture as root by `sudo -n`
(the binary as built, no file capabilities), giving what it wrote back to the user after; the
file-capabilities test checks the reason.

*Tests.* Unit: the converter (`ebpf_text.rs`: each function's record, a request served, one
interrupted and served again, an fsync, a `lio_listio` with a null and a `LIO_NOP` entry, the
issuer's own call of the same buffer left alone, a refused, a canceled, and an uncollected
request, one in flight at the end, two a call could serve, a C library not probed, the records
without the header); the reader (`strace.rs`: counted in the issuer at the serve, the retry,
the fsync, the notes, the fan-out, the export's lanes and results, and the same trace without
the records read as strace's); the layout (`ebpf.rs`: this box's libc, each function found,
`lio_listio` at two offsets, the aliases, refusals of a file that is no ELF, cut short, or
without the functions, the first library defining each) and the source check's wording.
Privileged (`tests/capture_ebpf.rs`, the POSIX AIO captures as root): `aeiou run --posix
posix-aio` of npy_per_sample's smoke set reads as `--posix sync` does, every request served and collected, the data in one lane; a
Python program calling each function through ctypes (two writes waited for by `aio_suspend`, an
`O_APPEND` write, `aio_fsync(O_DSYNC)`, a `lio_listio` with a null entry, a read) gives
strace's metrics, its seven requests each served and collected.

**LMCache's `io_uring`, read off its source and an eBPF capture (2026-11-05).** "LMCache uses
`io_uring` for some operations" (how this section came up) holds for one backend only. In
LMCache 0.5.5 (src, the installed package of `workloads/kvcache/lmcache-local-disk/kit`) the
local-disk backend reads and writes on a thread pool (`local_disk_backend.py`: `write`,
`readinto`, `O_DIRECT` under `use_odirect`) and the `fs://` connector through `aiofiles`, whose
threads make the ordinary calls (`connector/fs_connector.py`); `io_uring` is the `io_engine` of
the Rust raw-block plugin (`plugins/rust_raw_block_backend.py`, fixed buffers and `uring_cmd`
optional), on an NVMe block device of its own, which no case here uses. So the strace captures
of both mechanisms missed no request of the store. The shared store's pair captured again under
each source the same day (user's choice; `workloads/kvcache/lmcache-shared-fs`, captures
`2026-11-05` and `2026-11-05-strace`, the model's replies byte for byte the same) confirms it:
the counts and bytes of the 2026-10-02 strace capture exactly, the sources differing only
in what the threads' timing orders, and the only rings libuv's in the API server (939
`IORING_OP_EPOLL_CTL`) and two caio's `linux_uring` makes at import to test the kernel and never
submits to (with one `io_setup` likewise, the note `aio_contexts_without_io_submit`). The five
rows the writer put outside for predating S-18 are within tolerance in both. The raw-block
plugin is the `io_uring` application still to trace, and only with a raw device and that scope
decided (not queued). `CAPTURE.md` named LMCache as an `io_uring` application, corrected.
The local-disk backend's captures taken again the same evening (§3.64's note of 2026-11-05),
ShareGPT, AgentX at both pools, and `O_DIRECT`, each under both sources, likewise show no
request of the store through a ring (libuv's `IORING_OP_EPOLL_CTL` alone in the eBPF traces).


### 3.99 The engine's KV pool as a computed cache: the proposal, what tested it, and the debate of 2026-11-06 (raised by the user, debated and measured the same morning; **proposed, not decided, nothing built**)

*Status.* The user prefers the pool to the drawn holds ("much more representative"), and has
decided nothing else here. This reopens `PROJECT_BRIEF.md` §6 item 24. The numbers are from scratch
experiments, not committed tools: a copy of `lmcache_local_disk.py` with a probe namespace that
writes each request's fields, so a dry run prints the abstract's real demand (its drawn holds
reproduce the dry run's reads exactly at every seed), and `agentx.simulate` over that demand.

**The proposal.** One pool per engine over its slots, replacing `keep`, `sys_held`, and `sub_held`.
The demand stays the application's (chains, side runs, lanes, growth, prefixes); what the engine
holds of a request becomes the pool's answer. `pool_tokens` tokens in `pool_block`-token blocks; a
prompt is its prefix's blocks, then its conversation's, then its reply's, keyed so two requests
share a block exactly when they share everything up to it, as vLLM's block hashes chain. A block
held is a hit and leaves the free queue; a new one evicts from its head; a finished request frees
its blocks tail first. Held is the run of hits from the prompt's start. Order: request r of each
slot in turn, then r + 1, each admitted and finished before the next, so no preemption and no
admission queue. Holds feed the existing rules, the ops unchanged. In the contract: a `pool`
declaration on an actor, a `held` expression naming the pool and the request's segments, one site
per pool. In the runner: a pass before the gate, one `u32` per request (the chat case 5.6 MB), so
the op multiset is fixed before the run and timing never enters. `pool_tokens` 0 keeps the drawn
holds and every fingerprint. `GRAMMAR_OPTIONS.md` §5.3 would allow a simulated cache within one
actor in canonical order. The size comes from the accelerator (HBM less weights and activations,
at the engine's memory fraction), its fragment's value (§3.81); the WG fixes it, a submitter never
chooses it.

*The draft's five questions, unanswered and on hold* (recommendation first): the order (one
request at a time, or each slot's next admitted together); when holds are computed (a pass before
the run, or lazily); the block (vLLM's 16 tokens, untested, the experiments ran at the corpus's
64; or LMCache's 256); the default (`pool_tokens` 0, or the pool everywhere); which cases move (all
four traced).

**The draft's evidence was the wrong cell (the user's doubt).** The user read "only if the
abstract's chains match the replay" as low confidence and asked whether trace and abstract differ.
They do, and the pool over the abstract's own demand had never been run. Chunk reads:

| source | 2 GiB pool | 6 GiB pool |
|---|---|---|
| the trace | 38,818 | 2,321 |
| the replay's demand, simulated pool | 39,881 | 2,865 |
| the abstract, drawn holds (median of 8 seeds) | 21,155 | 2,437 |
| the abstract's demand, simulated pool (median of 12) | 23,785 (16.2k to 33.9k) | 915 (352 to 4,895) |

The pool alone does not close the gap. "What the drawn holds lack is the pool"
(`replay-2gib-pool/trace.tolerances.json`) overclaims and the §3.64 note of 2026-11-06 leans the
same way; neither is corrected yet (the user's call). Split: the share of loadable chunks the
engine lost is 67.4 % in the replay and 69.2 % with the pool on the abstract at 2 GiB, 4.3 % and
2.4 % at 6 GiB; but the loadable chunks are 57,759 against 33,892 and the mean prompt 69.8k against
44.3k. On ShareGPT, whose demand is accepted, the pool gives a median of 788 reads against the
trace's 755 (drawn holds 721). So the pool's rate is good where the demand is, and the fault is mostly
the demand. **The drawn holds' 6 GiB fit is cancelling errors:** they lose 7.2 % where the replay
loses 4.3 %, on 59 % of its loadable chunks, and `keep` was read from that engine at that pool.

**How can an abstract generated from a trace be 2× off? (the user's question).** It is not a
compressed trace: independent distributions fitted to the request and engine logs, the storage
trace only judging the result. Each marginal matches (growth per turn 5.8k against 5.5k, first
prompt 21k both) while the joint does not (longest chain 19 turns against 36). Tested on a Python
form of the abstract's request rule over the replay's rows (2 GiB 39.6k, close; 6 GiB 4,413 against
2,865, since chains of a session sharing prefixes are not in it, so its 6 GiB column is a trend
only): the replay's own growth 39,555 / 4,413; growth shuffled 32,280 / 1,707; shuffled only within
long and short chains 30,182 / 2,296, so a two-class fix does not restore it; chain kind and first
prompt shuffled too 31,002 / 1,544; the abstract 23,785 / 915. Ten times the requests moves the mean
prompt only to 47k, so run length is not the cause. The last step, chains redrawn and laid on
slots, is the largest, by elimination, untested. Long sessions growing more slowly and being
compacted was already recorded (§3.59's corpus accounting: "a growth that depends on the chain's
length is not drawn here").

**Read count as a quality check (the user's point: it should be one).** `compare --judge` judges
shares, not counts (`mix: reads / data ops`), `--self` widens a row by the seeds' spread (about 17 %
to 44 % at 6 GiB, so the 6 GiB accept had little force), and nothing checks the steps between.
Proposed: at the fit, the abstract over many seeds against the replay on requests, mean prompt,
longest chain, loadable chunks, reads, and writes, failing loudly.

**The context limit (the user's questions).** 131,072 is `replay_agentx.py --max-context` and the
server's `MAX_MODEL_LEN`: the 8 GB GPU, not the workload. 990,016 is the corpus publisher's filter
on overcounted rows (no real prompt exceeds 1M; `in` overcounts by up to about 260k in the tail).
Both are one request's prompt tokens, and they are comparable. Corpus prompts: p50 142k; 53 % of
requests and 86.7 % of prompt tokens over 131k; 308 of 393 sessions peak over it, median peak 225k.
The replay is the first 300 of the 8 sessions' 1,892 requests (each session's share): 233 played,
67 skipped, 1,592 never reached. Every AgentX storage capture is under 131k, the low-context start
of sessions.

**Nothing real drops a request (the user).** The skip is the replay's artifact. vLLM refuses a
prompt and reply over its context; chunked prefill splits the compute below the limit, not the
requirement that a request's KV be in the pool. The client manages it: main-agent prompts that fall
under 60 % and stay there keep 34k of prefix and go on at 47k from 131k to 262k (65 cases), and
48k and 83k from 800k to 1M (32); drops below 131k keep nothing and are probably something else.
The abstract restarts from nothing: it has no compaction.

**The user's paraphrase, corrected.** Right: the corpus reaches about 1M, the replay ran it on an
8 GB GPU, those requests cannot run there, and HBM must enter the model. Corrected: the corpus was
recorded at a client-side proxy and holds no hardware, pool, or storage; prompts depend on the
conversation and the model's window, not the GPU; a small GPU refuses, it does not restructure (in
the vLLM traced; offload schemes exist); and the storage I/O is ours (vLLM and LMCache were chosen
here), nobody recorded the original's.

**Is pool > largest context enough? (the user's question).** Necessary, not sufficient. The pool
swept over the replay's requests, serial order, largest context 131k:

| pool (tokens) | 131,072 | 174,752 | 262,144 | 349,504 | 437,000 | 524,288 | 699,008 | 1,048,576 | 2,000,000 |
|---|---|---|---|---|---|---|---|---|---|
| chunk reads | 42,202 | 39,881 | 29,883 | 20,519 | 7,315 | 2,865 | 1,351 | 193 | 129 |

The sum of the sessions' current prompts is 425k at the median and 691k at peak. Three regimes:
below the largest context the request cannot run; between it and the sum of live contexts the
engine evicts and re-reads; above the sum almost nothing is read. The 6 GiB pool sits just past the
drop, which is why small demand errors move its reads so much. The ratio of pool to the sum of live
contexts is the quantity to report, and the slots matter as much as the HBM.

**How much larger was the real hardware? (the user's question).** No basis: the corpus names none,
the service is shared, and the only firm bound is the floor, one largest context. But the corpus's
`ttft` (56,432 requests, unused until now) shows behavior. Continuing requests with little new
input, median seconds, prompts of 20k to 100k and 500k to 1M: idle under 60 s, 2.05 and 3.18; 60 to
240 s, 1.95 and 3.22; 290 to 310 s, 4.55 and 11.35; over an hour, 2.69 and 13.36. Flat in prompt
size for about five minutes, then scaling with it: the 5-minute default of the provider's documented
prompt cache. 96 % of continuations arrive inside it, and at most about 9 % of warm requests at 500k
to 1M look lost. So these sessions' survival looks set by time, not capacity pressure, unlike the
LRU pool modelled here. It says nothing of the tier: a warm hit may come from HBM or a slower one.
The replay ran `--time-scale 0`, so idle time is in none of our captures. For scale only, and no
claim about this corpus: a 70B-class model at 320 KiB per token is about 305 GiB per 1M tokens, so
eight H200s leave room for about 2.7 such contexts, about 12 conversations at the corpus's mean of
219k; a busy engine is likely evicting.

**Where it stands (recommendations, none decided).** Fix the demand before the five questions. Pool
size and concurrency are stated parameters of a case, never inferred from a corpus; a case whose
pool is under its context limit is refused; a case reports its regime and runs more than one point
on the curve. An idle expiry beside the pool would model a provider's cache (an open vLLM and LMCache
deployment has no such timer). Real evidence above 131k needs a larger GPU, or sessions scaled
down by one factor, which assumes the storage behavior scales. The publisher's billed `cache_read`
and `cache_write` columns, not published, would replace the timing inference (the user may ask).

**What the limit's behavior adds: almost nothing (run at the user's request the same day).** Scratch
copies of the abstract, differing only at the limit, the pool simulated over each one's demand, the
median of 16 seeds; each copy's drawn holds reproduce its dry run's reads:

| at the limit | requests | mean prompt | live sum p50 | 2 GiB reads | 6 GiB reads |
|---|---|---|---|---|---|
| restart (today) | 232 | 44.6k | 355k | 23,785 | 880 |
| skip (the replay), 29 a slot | 224 | 43.0k | 338k | 22,080 | 867 |
| skip, 37 a slot (the replay's share) | 281 | 42.7k | 345k | 28,001 | 1,257 |
| compact, summary 14k | 232 | 43.9k | 349k | 22,688 | 738 |
| compact, summary 35k | 232 | 44.8k | 352k | 23,226 | 826 |
| no limit (diagnostic) | 232 | 48.7k | 388k | 29,205 | 2,154 |
| the replay, simulated pool | 233 | 69.8k | 425k | 39,881 | 2,865 |

So what the abstract does at the limit is not why its demand is short. Skip at 37 reads more only
because it serves 21 % more requests. Even with no limit the mean prompt reaches only 48.7k and the
longest chain 23 turns, so few chains reach 131k at all: the chain count rises by about two under
compaction, so it fires about twice a run, and this test says little of compaction at full session
length, where 308 of 393 sessions cross 131k. The no-limit run's 6 GiB median is no match: its seeds
range from 225 to 27,337, and its prompts exceed what the replay could send. Still open: `requests`
is 29 for every slot where the replay's sessions ran 19 to 37, so no chain of 32 or 36 turns can
exist; long chains open high in the replay (45k to 64k) where `first_in` is drawn apart from the
chain's length; and chains redrawn and laid on slots, the largest step above.

**Whole chains dealt to slots (run at the user's request the same day).** The replay's real requests,
each with its own blocks, only their order changed, the pool simulated, 24 draws. The first row
reproduces `holds --timeline serial` exactly, so the 6 GiB column here is sound, unlike the
ablation's.

| layout | mean prompt | live p50 | writes | 2 GiB reads | 6 GiB reads |
|---|---|---|---|---|---|
| the replay's order | 69.8k | 425k | 5,663 | 39,881 | 2,865 |
| sessions as slots, in rounds | 69.8k | 425k | 5,663 | 39,754 | 2,738 |
| sessions as slots, chains unbroken | 69.8k | 446k | 5,663 | 39,310 | 2 |
| all 42 chains dealt at random | 69.8k | 411k | 5,663 | 34,375 (18.6k to 37.9k) | 111 |
| the same, cut at 29 a slot | 64.4k | 484k | 5,524 | 31,050 (18.6k to 35.5k) | 110 |
| chains drawn with replacement, 233 requests | 67.8k | 340k | 6,355 | 29,934 (18.3k to 46.5k) | 66 |
| the same, cut at 29 a slot | 60.5k | 508k | 7,037 | 33,200 (24.1k to 40.5k) | 199 |
| whole sessions drawn with replacement | 64.2k | 412k | 5,529 | 32,031 (20.6k to 48.0k) | 2,585 (707 to 13,617) |

Keeping a chain whole is what restores the prompt: at the abstract's own budget (232 requests, 29 a
slot) the mean is 60.5k against 44.6k and the reads 33,200 against 23,785, so the 29-request cap is
not the cause, and the chain's joint values are. That recovers one to two thirds of the 2 GiB gap.
At 6 GiB nearly every read comes from a chain's suspension: unbroken, the same requests read 2
chunks. So the unit that gives both pools is the session, a main chain with its side runs. Two
limits. Drawn chains lose the prefixes a session's chains share, so writes rise 12 % to 24 %. And
the sample is small: resampling these same 8 sessions gives 20.6k to 48.0k and 707 to 13,617, so
the trace's 38,818 and 2,321 are one draw from a wide spread, and a tight match to them is not a
meaningful target at 233 requests. A table of sessions would be close to a listing of the replay
(against the abstracts being compact), and at the corpus's scale too large: how to keep the joint
structure compactly is open.

**Where the compared reads come from (asked by the user the same day).** "The trace" in this section
is a trace of our own lab replay, not of AgentX. The corpus holds what Claude Code sent through a
proxy and no storage I/O, so no real read count exists for the original sessions. The 38,818 and
2,321 are real measurements, the eBPF and strace captures of 2026-11-05 on this machine (at 2 GiB
122,110,869,504 bytes read, exactly 38,818 chunks of 3 MiB), of a system we chose: Qwen2.5-0.5B,
synthetic tokens from the hash ids, replies that are not what the next prompts contain, the first 37
or 38 requests of 8 sessions, all under 131k, `--time-scale 0`, 8 sessions at once, and pools of 2
and 6 GiB. The 39,881 and 2,865 are `agentx.simulate` over the replay's requests; every row with a
pool is computed.

So the comparison is fair for one question, whether the abstract reproduces the system we ran: both
sides have the same requests, pool, and concurrency, and the gap is real. It says nothing of the real
service's storage, where the same requests give 42,202 to 129 reads by the pool size alone. Even the
fair question has two limits: the abstract is fitted to that replay's logs and judged against the
same run's trace, a consistency check rather than a held-out test, and the target is one draw.
Proposed, not decided: call these numbers the lab replay from here on; aim at the shape of the pool
curve rather than two exact counts, since pool and concurrency would be stated parameters of a real
deployment; and fit on some sessions and judge on others.

**The replay at a 500k limit (2026-11-06 and 07, at the user's request; scratch, nothing in the
repository changed).** With no GPU but the laptop's, the user chose to raise the replay's limit.
The 131,072 was a setting matching the model's 128k configuration, not the GPU's memory: at 12,288
bytes a token the 6 GiB pool holds 524,288 tokens. `MAX_MODEL_LEN=500000`, YaRN factor 16,
`KV_BYTES` 6 GiB (vLLM needs a pool at least the limit, so 2 GiB cannot run), `replay_agentx.py
--requests 1000 --max-context 500000`: the first 125 requests of each of the 8 sessions, 700 in all
since three sessions are shorter, none over the limit (largest 445k). Under strace (the eBPF
binary's grant had lapsed); scratch in `~/MLPerfStorage/limit500k/`.

What it took. A cold 480k prompt prefills in 1,211 s, a cold 250k in about 7 minutes. The first run
hung after 431 requests: LMCache stages every load from disk through its CPU buffer
(`max_local_cpu_size`, 2 GB in the kit's `lmcache.agentx.yaml`, about 175k tokens here), and a load
of 191,040 tokens could never be allocated, so it retried forever. No replay under 131k could meet
this. Raised to 12 GB for the rerun, which played all 700 with no failed allocation. An FP8 KV cache
(half the bytes, so the same pool holds about 1M tokens) was tried at an 800k limit: it serves, on
the Triton backend at about a quarter of FlashAttention's speed, and past about 485k tokens its
prefill fell from 40 s to over 1,200 s per 2,048 tokens with the GPU nearly idle, cause unknown; a
790k prompt would have taken over 50 hours, so it was stopped, and its reload from LMCache never
tested. The spill into Windows' shared GPU memory was not tried.

| 6 GiB pool | requests | mean prompt | largest | chunk reads | chunk writes |
|---|---|---|---|---|---|
| lab replay at 131k (2026-11-05) | 233 | 69.8k | 131k | 2,321 | 5,663 |
| lab replay at 500k, first run, hung (partial) | 431 | 87k | 224k | about 63,800 | about 8,900 |
| lab replay at 500k | 700 | 134k | 445k | 184,972 | 11,910 |
| the pool, `holds --timeline serial` | 700 | | | 180,308 | 11,881 |
| the pool, `holds --timeline engine` | 700 | | | 99,405 | 11,881 |

Two findings. The 6 GiB count of the 131k replay was an artefact of its cut: three times the
requests at twice the prompt read 80 times as much, 264 chunks a request against 10, with the same
pool, so the 6 GiB target the abstract was judged against said little of the corpus. And the pool,
given its size alone and the requests one at a time as an abstract's slots take them, predicts the
measured reads within 2.5 % and the writes within 0.3 % at prompts near the corpus's median (142k),
the first test of it at realistic lengths; on the engine's own timeline (199 preemptions) it is 46 %
low. Still not ground truth for any real service: Qwen2.5-0.5B, a pool we chose, synthetic tokens,
125 requests a session, strace only.

**The abstract refitted to the 500k replay (2026-11-07, at the user's request).** `agentx.py log` and
`fit --replay ... --context 500000` with the kit README's `--set` values: 700 requests in 56 chains,
88 a slot, `context` 500,224. The committed abstract (`6fc276c`) at those parameters, seeds 1 to 8,
its drawn holds as it runs, and the same demand through the pool (the probe abstract and
`agentx.simulate` of the morning, pool 524,288 tokens):

| 6 GiB pool, 500k limit | mean prompt | chunk reads | chunk writes |
|---|---|---|---|
| lab replay (the trace) | 134k | 184,972 | 11,910 |
| abstract, drawn holds, median of 8 | 85.8k | 138,618 (91,708 to 155,788) | 13,371 |
| abstract's demand and the pool, median of 8 | 85.8k | 150,615 (62,854 to 187,170) | 13,371 |

Closer than at 131k (the drawn holds 25 % low against 45 %, the pool 19 % against 39 %), and the
same cause: the abstract's prompts are short, 64 % of the replay's, as its chains' joint values are
lost in independent draws (the whole-chain test above), while writes run 12 % high. The pool on the
replay's own requests is within 2.5 %, so what remains is the demand, not the pool. Seeds spread the
reads three to one, so one draw of 700 requests is still a loose target.

**Whole conversations in a scratch abstract (2026-11-07, at the user's request).** A copy of the
abstract (with the probe namespace) in which a new chain draws a row of a table of the 500k replay's
56 chains (21 main, 35 started in side runs), uniformly among its own lane's, and plays the row:
the first prompt, then each request's added tokens, reply, and trim, in order; the row's kind and
length with it. The draws of `turn_in`, `turn_out`, `trim`, `turns`, `kind`, `first_in` are gone;
the slots, side runs, and `keep` are the fit's. The table rebuilds all 700 of the replay's prompts
exactly (an entry is a balanced tree of comparisons on the row's index; the AST is 3 MB, against the
committed one's 150 KB: a listing, as expected, not the compact form). Seeds 1 to 8:

| 6 GiB pool, 500k limit | mean prompt | chunk reads | chunk writes |
|---|---|---|---|
| lab replay (the trace) | 134k | 184,972 | 11,910 |
| refitted abstract, drawn holds | 85.8k | 138,618 (91,708 to 155,788) | 13,371 |
| refitted abstract and the pool | 85.8k | 150,615 (62,854 to 187,170) | 13,371 |
| whole chains, drawn holds | 107.0k | 147,386 (73,579 to 215,154) | 15,161 |
| whole chains and the pool | 107.0k | 167,681 (38,593 to 285,632) | 15,161 |
| the same at 125 requests a slot (prompt only) | 114.3k | | |

Whole chains recover about half the prompt gap (86k to 107k of 134k) and with the pool bring the
reads to 9 % under the trace, against 19 %, but the spread doubles (38.6k to 285.6k), and writes
rise 27 %: chains drawn with replacement lose the prefixes a session's chains share, as `deal.py`
found. The prompt still falls short because a slot plays chains in a random order and is cut at 88
requests: a long chain that starts late loses its longest turns, where the replay's sessions open
with theirs, and the cut at 125 requests a slot recovers only 7k more. So the order of a session's
chains matters as well as each chain, and the session, its main chain with its side runs, remains
the unit, as the whole-session resample of the day before found. Scratch: `~/MLPerfStorage/limit500k/whole/`.

**Whole sessions in a scratch abstract (2026-11-07, at the user's request).** The table moves into
array parameters (`P.r_in[gi]`, an `elem`): per request of each of the replay's 8 sessions in its
order, the distance back to its chain's previous request (0 a new chain), its added tokens (a new
chain's first prompt), reply, trim, and kind. A slot plays a session from its start and draws
another, uniformly, when it ends; the distance is written `1 + (x − 1) mod (r + 1)` so rule V3 holds.
The side-run machinery and the per-request draws are gone: the AST is 103 KB, smaller than the
committed one's 150 KB, and the listing lives in the parameter file (700 entries in each of five
arrays). A check mode (slot i plays session i, then idles) replays the replay itself:

| 6 GiB pool, 500k limit | requests | mean prompt | chunk reads | chunk writes |
|---|---|---|---|---|
| lab replay (the trace) | 700 | 134k | 184,972 | 11,910 |
| the pool on the replay's own requests | 700 | | 180,308 | 11,881 |
| session table, slot i plays session i, and the pool | 700 | 131k | 174,976 (168,725 to 174,976) | 12,254 |
| session table, sessions drawn, 88 a slot, and the pool | 704 | 105.8k | 171,626 (133,935 to 244,944) | 13,389 |
| the same, drawn holds | 704 | 105.8k | 151,200 (127,863 to 195,550) | 13,389 |
| whole chains (above), and the pool | 704 | 107.0k | 167,681 (38,593 to 285,632) | 15,161 |

The check comes within 5 % of the trace and 3 % of writes: the representation holds a session's
storage (what is left is its prefixes, one size for every session, and a chain's base past them).
Drawn, the reads come within 7 % and the writes 12 % (a session drawn twice is two users, its
chunks written twice), the spread narrows to a factor of two against chains' seven, and the trace
lies inside it; the prompt is no longer than with whole chains, since every slot is cut at 88 where
four of the replay's sessions run 125 and three end by 35. That cut is the replay's shape, not the
workload's: a replay giving each session its own share ends unevenly, an abstract's slots are equal
and go on, so the drawn rows are compared with one transient. A fair target would be a replay that
deals sessions to slots as the abstract does. Scratch: `~/MLPerfStorage/limit500k/sess/`.

**The whole corpus as a session table, judged by the pool (2026-11-07, at the user's request; no
GPU).** All 393 sessions, 98,827 requests, as one engine sees them (one store, no limit past the
corpus's 990,016), in the session table's arrays (98,822 prompts rebuilt exactly, 5 off where a prompt
is shorter than its chain's base; `retain` unbounded, since a chain goes back up to 2,804 requests).
The reference is the corpus's real requests and blocks (`replay_agentx.py`'s keys) through
`agentx.simulate` and `lmcache_accesses`, in an arrangement both sides can make: 8 slots, slot i
playing sessions i, i + 8, ... for 2,000 requests (16,000 requests of 81 sessions, mean prompt 252.5k),
in rounds. The abstract's `keep` holds everything, so the pool alone decides the reads; the drawn
rows sample all 393 sessions with replacement, 8 seeds. Chunk reads by pool, in tokens:

| pool | reference | abstract, the same arrangement | abstract, drawn (median, range) |
|---|---|---|---|
| 1M | 12,971,964 | 12,242,918 (−6 %) | 9,010,264 (6,063,812 to 11,855,766) |
| 2M | 4,036,273 | 3,369,637 (−17 %) | 1,514,277 (198,229 to 3,370,399) |
| 4M | 45,622 | 37,405 (−18 %) | 19,281 (7,098 to 45,635) |
| 8M | 1,028 | 2,784 | 2,094 (402 to 12,457) |
| 16M | 196 | 243 | 243 (243 to 5,189) |
| writes | 229,129 | 230,880 (+1 %) | 242,585 |
| mean prompt | 252.5k | 243.8k | 207.0k (174.5k to 237.8k) |

Three things. The table holds at the corpus's scale: writes within 1 %, reads within 6 % where the
pool is far too small and 17 % to 18 % near the cliff, where 3 % shorter prompts (a chain's base past
its prefix is not in the table) move the reads most. The cliff is steep, as the 500k sweep showed: at
8 slots it falls from 4 million reads to 46 thousand between a pool of 2M and 4M tokens, where the
slots' live contexts add up. And near it the drawn runs spread seventeen to one (198k to 3.37M at 2M):
which sessions happen to be live together decides the reads, so a benchmark set near the cliff
would need far longer runs to be stable, or the WG's pool and slots (brief item 70) set away from it;
the drawn medians sit below the fixed arrangement's because its 81 sessions are the corpus's first,
longer than the mean (207k drawn against the corpus's 219k, the cut at 2,000 a slot shortening the
longest). The runner overflowed its stack at 2,000 requests a slot (`x @ (r − dd)` reaching back
thousands of requests); `RUST_MIN_STACK` of 1 GiB got past it, for the scratch only. Scratch:
`~/MLPerfStorage/limit500k/corpus/`.

**Where the benchmark should sit on the curve, and who turns which knob (2026-11-07, the user's
questions; proposed, nothing decided).** The cliff is not at a token count: it is where the pool
holds the slots' live contexts, so its axis is the ratio of pool tokens to users times live context
per user, and the 1M row at 8 slots is the same operating point as 8M at 64 users. Below the cliff
is where real agentic serving sits (an 8×H100 node serving a 70B GQA model has about 500 GB of HBM
past its weights, about 1.5M tokens at 327,680 bytes a token, 3M with FP8 KV, room for 6 to 12 live
contexts of AgentX's 250k mean; 64 users on it is a ratio near 0.1), and it is the stable side: the
engine holds little, reads follow the prompts, the fixed arrangement came within 6 % and the drawn
spread was two to one, against seventeen to one at the cliff. Above it storage sees writes. The
model matters through the ratio (its bytes a token), not little; what moved reads little was the
detail of the fitted session structure. Proposed: the WG fixes the ratio well below the cliff, and
the arrangement of sessions on slots (the fixed one), and the researcher keeps the draws.

The user's view: the submitter's knob is the number of users a system under test supports; GPU
count, DP, TP, and PP are the WG's (brief item 70), since a buyer compares vendors by users, not by
deployment tuning. Agreed, with one refinement: users added to one engine with a fixed pool change
the ratio and so every user's I/O, so results at different user counts would not be one workload;
proposed instead, the WG fixes one engine (model, deployment, pool, users per engine) and the
submitter scales engines, users being engines times users per engine, as training scales
accelerators at a fixed per-accelerator load; the pass criterion a latency budget for a KV load (as
AU), not yet defined.

The user's next question: users have queries of a similar structure and different data, and the data
does not matter to storage, so could a few session definitions, perturbed, stand for the corpus
(how much session information the abstract must carry)? Agreed in principle, if three things hold:
every user's blocks are its own (only the system prompt shared; the abstract already keys a
conversation's blocks by its instance), users are out of phase (not every slot at the same point of
the same session), and the archetypes span the corpus's lengths (mean 219k, a tail to 990k). The
user asked for the test: k = 1, 4, 16 archetypes from the 393 sessions, with and without perturbed
token counts, judged by the pool against the whole table at ratios below the cliff.

**Archetypes in place of the table (2026-11-07, the user's test; scratch, no GPU).** k sessions of
the 393 stand for all: sorted by mean prompt into k bins of equal count, from each the session
nearest the bin's medians of mean prompt, peak, and length, drawn uniformly (equal bins); `a1w` is
the one session nearest the request-weighted mean prompt (187k) and length; `jitter` scales every
request's new input and reply by a draw within ±20 %. The pool on the abstract's demand, 8 slots,
2,000 requests each; the whole table 16 seeds, the rest 8. Median reads (range), in thousands:

| set | mean prompt | writes | 0.5M | 1M | 2M | 4M | 8M |
|---|---|---|---|---|---|---|---|
| all 393 | 198k | 237.8k | 10,645 (9,241–13,359) | 7,992 (6,064–11,856) | 1,053 (198–4,088) | 23 (7–83) | 2 (0–12) |
| 64 | 255k | 234.7k | 13,964 | 12,863 | 2,916 | 8 | 1 |
| 16 | 256k | 254.0k | 13,887 | 12,662 | 3,585 | 5 | 2 |
| 4 | 187k | 184.6k | 9,968 (9,574–10,820) | 7,305 | 227 | 0 | 0 |
| 1 (`a1w`) | 254k | 232.3k | 13,113 | 11,586 | 10,129 | 3,679 | 175 |
| 1 (median) | 125k | 402.9k | 5,340 | 4,252 | 1,028 | 69 | 0 |

(jitter changed no row by more than a few percent; the rows without it shown.) What it says. Below
the cliff the reads are close to proportional to the mean prompt, about 54 reads a request per
thousand tokens in every set (0.5M: 53.8 all, 53.3 four, 54.2 sixteen, 51.6 one), so there the
number of archetypes does not matter and the request-weighted prompt does: 16 and 64 archetypes are
29 % long and 31 % high, four happen to match and are within 6 %; picking by session count misweights
long sessions, which hold a slot longer. One session repeated has no cliff at all (10.1M reads at 2M
against 1.05M, 3.7M at 4M against 23k): its eight copies peak together, and ±20 % of tokens does not
move them out of phase, since the length is the session's. So the abstract's session information is
small if the archetypes are chosen to match the request-weighted prompt distribution and the users
are put out of phase; jitter of tokens is not the lever. Scratch: `~/MLPerfStorage/limit500k/arch/`.

**Archetypes picked by time, users out of phase (2026-11-08, the user's go; scratch, no GPU).** The
picking corrected: sessions sorted by mean prompt into k bins of equal request count (the time a
slot spends), from each the session nearest the bin's request-weighted mean prompt (weighted 3),
chunks written a request, and length (0.5), drawn with weight bin requests over its own length, so
each holds slots for its bin's share of the time (`s_pick`, a distribution of session indices).
Users out of phase: slot i idles its first i × 1,000 / 8 requests (`stagger`; an idle request has no
I/O), and the first 1,000 rounds are warm, not counted. Every set under the same stagger and warm,
against the whole table drawn the same way (16 seeds; the rest 8). Median reads, change from the
whole table, and seed range, in thousands:

| set (table rows) | mean prompt | writes | 0.5M | 1M | 2M | 4M |
|---|---|---|---|---|---|---|
| all 393 (98,827) | 195k | 225.0k | 10,690 (7,148–14,195) | 8,207 (4,066–12,882) | 831 (110–3,557) | 29 (13–47) |
| 16 (13,197) | 206k | +3 % | +7 % (10,771–14,836) | +15 % | +118 % | +54 % |
| 8 (4,276) | 192k | −3 % | −2 % (9,371–11,365) | +1 % | +110 % | +34 % |
| 4 (2,626) | 205k | +4 % | +5 % (10,160–13,471) | +10 % | +84 % | +111 % |
| 2 (1,010) | 211k | −8 % | +6 % | +11 % | +8 % | −45 % |
| 1 (795) | 232k | −5 % | +17 % | +29 % | −12 % | −100 % |
| 1, no stagger | 240k | −2 % | +14 % | +27 % | +1,029 % | +12,588 % |
| all 393, no stagger | 191k | +0 % | −4 % | −6 % | +24 % | −6 % |

What it says. Below the cliff, 2 to 16 archetypes picked by time come within 2 % to 11 % of the
whole table's reads and 3 % to 8 % of its writes, eight within 3 % of both at 0.5M and 1M; that is
inside the whole table's own spread from seed to seed (7.1M to 14.2M at 0.5M), and the archetypes'
spread is narrower. At the cliff nothing matches, the whole table included (110k to 3.56M at 2M, a
factor of 32 between seeds), so there the medians compare nothing. The stagger matters only when
the sessions are few: one session without it has no cliff (9.4M reads at 2M), with it a cliff
again; with four or more drawn by weight the draws put the users out of phase themselves. So the
session information an abstract needs is a few thousand rows (eight sessions, 4,276 requests),
about 4 % of the table, if they are picked by the time they hold a slot. In-sample: the archetypes
were picked from the sessions they are judged against; the table itself was 6 % from the corpus's
real blocks at 1M. Scratch: `~/MLPerfStorage/limit500k/arch2/`.

**Archetypes held out (2026-11-08, the user's go; scratch, no GPU).** The sessions split in two,
by even and odd index and by a seeded random split; archetypes picked by time from one half are
judged against the other half's whole table (12 seeds), each way, beside eight picked from the judged
half itself (in-sample). Change from the judged half, reads at 0.5M and 1M:

| split, picked from | judged half's prompt | 4 | 8 | 16 | 8 in-sample |
|---|---|---|---|---|---|
| even, judged on odd | 191k | −3 %, −10 % | +8 %, +12 % | −4 %, −6 % | +4 %, +5 % |
| odd, judged on even | 214k | −4 %, −11 % | −7 %, −15 % | −5 %, −12 % | −4 %, −9 % |
| random A, judged on B | 180k | +50 %, +99 % | +35 %, +67 % | +47 %, +85 % | −9 %, −15 % |
| random B, judged on A | 225k | −21 %, −31 % | −29 %, −46 % | −26 %, −38 % | +5 %, +6 % |

Writes within 6 % in every row but one (+13 %). What it says. The error held out is the difference
between the halves, not the archetypes': the random halves' request-weighted prompts are 158k and
226k, and the reads follow the prompt error almost one for one (eight from A: prompt +29 %, reads
+35 %), as below the cliff they did in every test; the even and odd halves are alike and there four
to sixteen archetypes hold within 15 %. In-sample eight stay within 9 % at 0.5M. So the picking is
sound, and what limits fidelity is the corpus: half of it fixes its mean prompt only to within about
±20 %, all 393 sessions to perhaps half that, one publisher's agents of one period. The archetypes
should be picked from the whole corpus, and the mean prompt, which sets the reads below the cliff
nearly in proportion, is the number a reviewer should see stated beside them, a property of the
workload the WG chooses, not a claim about agentic serving in general. Scratch:
`~/MLPerfStorage/limit500k/arch3/`.

**Below ρ 0.34, and more users per engine (2026-11-08, the user's go; scratch, no GPU).** ρ = pool
tokens / (users × mean live context, about 200k). The whole table against eight and four archetypes
(the whole corpus's, picked by time), stagger and warm as before; 8 users 8 seeds, 32 and 64 users 6
(archetypes of four, 4). Median reads, millions, seed range as a share of the median:

| users | ρ | whole table | 8 archetypes | 4 archetypes | reads a write (whole) |
|---|---|---|---|---|---|
| 8 | 0.08 | 11.03 (58 %) | −1 % (18 %) | +5 % (27 %) | 48.4 |
| 8 | 0.17 | 10.89 (59 %) | −1 % | +5 % | 47.9 |
| 8 | 0.33 | 10.67 (66 %) | −2 % | +5 % | 46.4 |
| 32 | 0.12 | 33.14 (5 %) | +9 % (16 %) | 0 % (20 %) | 48.3 |
| 32 | 0.25 | 32.91 (5 %) | +10 % | 0 % | 48.0 |
| 32 | 0.49 | 32.62 (5 %) | +10 % | −1 % | 47.8 |
| 32 | 0.98 | 10.94 (35 %) | +65 % | +2 % | 15.7 |
| 64 | 0.12 | 44.66 (12 %) | +6 % (8 %) | +6 % (7 %) | 52.3 |
| 64 | 0.24 | 44.42 (12 %) | +6 % | +6 % | 52.1 |
| 64 | 0.48 | 44.35 (12 %) | +6 % | +6 % | 52.0 |
| 64 | 0.96 | 11.09 (176 %) | +107 % | +106 % | 12.8 |

What it says. Below ρ about 0.5 the pool's size does not matter: from 0.08 to 0.49 the reads move
4 % at most, at every user count, and nearly every stored chunk of a returning conversation is read
back (about 50 reads a write). The cliff sits near ρ 1 at 32 and at 64 users alike, so ρ is the
right axis. More users per engine narrow the spread between seeds (the whole table's range from 66 %
of the median at 8 users to 5 % and 12 % at 32 and 64), and archetypes of four or eight stay within
10 % below the cliff at every user count, writes within 8 %. A caution on what "the pool does not
matter" rests on: the timeline serves the users in strict rounds, which for an LRU pool smaller than
the live contexts is the worst case (each conversation is evicted just before it returns); arrivals
out of order, or idle time between a user's requests (the corpus's, never replayed), would give some
hits below the cliff, and that is the next thing to test before the operating point is decided.
Scratch: `~/MLPerfStorage/limit500k/rho/`.

**The order users arrive in (2026-11-08, the user's go; scratch, no GPU).** Every test above serves
the users in strict rounds, the worst case for an LRU pool smaller than the live contexts. The same
demand (one dry-run) through the pool in other orders: `exp`, each user's requests a draw of Exp(1)
apart, out of order with no idle; and the corpus's own times, each request at its session's `t`,
sessions back to back, users starting within the median session's span (2.8 h), with every gap
capped at 300 s (the provider's cache lifetime the corpus's `ttft` shows) or at 1,800 s, counted over
a time window in which every user is live. Uncapped the order cannot be run with a fixed count of
requests a user: 79 % of a session's time is gaps over 30 minutes (the user away; gaps p50 9.2 s,
p90 69 s, p99 1,413 s), so users' timelines end hours to days apart. Reads a request, the whole
table; users active at once counted as those without a gap over 300 s:

| users | order | ρ 0.12 | ρ 0.49 | ρ 0.98 | active |
|---|---|---|---|---|---|
| 32 | rounds | 718 | 710 | 278 | 32 |
| 32 | `exp` | 645 | 386 | 100 | 32 |
| 32 | corpus, gaps ≤ 300 s | 649 | 311 | 101 | 32 |
| 32 | corpus, gaps ≤ 1,800 s | 549 | 146 | 32 | 18.7 |
| 64 | rounds | 705 | 700 | 198 | 64 |
| 64 | `exp` | 611 | 357 | 76 | 64 |
| 64 | corpus, gaps ≤ 300 s | 546 | 226 | 68 | 64 |
| 64 | corpus, gaps ≤ 1,800 s | 480 | 102 | 20 | 39.5 |

Eight archetypes against the whole table, 32 users: rounds +1 %, +1 %, +20 %; `exp` 0 %, +3 %, +22 %;
corpus ≤ 300 s +13 %, +39 %, +37 %; corpus ≤ 1,800 s +26 %, +65 %, +127 % (writes within 4 % in all).

What it says. The flat region below ρ 0.5 of the note above is the rounds' artefact: in any other
order the reads fall steadily with ρ (at 0.49, a half to a fifth of the rounds'), so ρ sets the load
again, and there is no band where seeds spread thirty to one. Random arrival without idle comes near
the corpus's timing with gaps capped at five minutes (within 1 % at 0.12, 24 % at 0.49 at 32 users),
and how long users may stay away moves the reads as much as ρ does, through the users active at
once. The archetypes, picked on prompt, writes, and length, hold under random arrival and not under
the corpus's times, whose gaps they were not picked for. So the think time is a parameter of its own
the WG must set (and with it, what "users per engine" counts), and an abstract that draws it from a
distribution apart from the sessions keeps the archetypes valid. A user away for hours returns to an
engine that holds nothing of the conversation, so whether it is read back depends on how long the
storage tier keeps it: a dimension, the tier's capacity and retention, that no test here models.
Scratch: `~/MLPerfStorage/limit500k/arrive/` (two bugs found and fixed in the scratch's timing,
results above after the fix).

**A think time drawn apart from the sessions (2026-11-08, the user's go; scratch, no GPU).** The
test the note above proposes: each gap inside a session drawn from the corpus's gaps instead of
taken from its own row, capped as before, in two forms: pooled, every gap from all 98,434 (p50 9 s,
p90 69 s, 2.5 % over 300 s, 0.8 % over 1,800 s); and by kind, from the gaps before requests of the
same kind as the next one, since the corpus's gaps depend on it (main lane, kind 0: p50 17 s, 3.9 %
over 300 s, 90 % of its time in those; sub-agent, kind 1: p50 4 s, 0.1 %; kind 2: p50 4 s, 1.4 %).
Reads a request, medians of four seeds at 32 users and three at 64:

| users | think time | whole table, ρ 0.12 / 0.49 / 0.98 | eight archetypes against it |
|---|---|---|---|
| 32 | corpus, ≤ 300 s | 649 / 311 / 101 | +13 %, +39 %, +37 % |
| 32 | pooled, ≤ 300 s | 525 / 165 / 54 | −4 %, +4 %, +15 % |
| 32 | by kind, ≤ 300 s | 603 / 250 / 82 | +5 %, +20 %, +32 % |
| 32 | corpus, ≤ 1,800 s | 549 / 146 / 32 | +26 %, +65 %, +127 % |
| 32 | pooled, ≤ 1,800 s | 416 / 78 / 22 | +3 %, +15 %, +18 % |
| 32 | by kind, ≤ 1,800 s | 514 / 118 / 29 | +11 %, +34 %, +45 % |
| 64 | corpus, ≤ 300 s | 546 / 226 / 68 | |
| 64 | pooled, ≤ 300 s | 489 / 140 / 50 | +11 %, +30 %, +42 % |
| 64 | by kind, ≤ 300 s | 568 / 225 / 72 | +22 %, +45 %, +68 % |
| 64 | corpus, ≤ 1,800 s | 480 / 102 / 20 | |
| 64 | pooled, ≤ 1,800 s | 393 / 72 / 22 | +12 %, +28 %, +23 % |
| 64 | by kind, ≤ 1,800 s | 470 / 98 / 25 | +30 %, +68 %, +68 % |

Writes a request agree within 7 % everywhere. Users active at once are the same in every form
(18.7 to 19.0 of 32, 37.7 to 39.5 of 64 at 1,800 s).

What it says. A pooled think time is too gentle: it puts the long gaps before any request, most of
them sub-agent calls whose context is short-lived anyway, and halves the reads at ρ 0.49 against the
corpus's own times. Drawn by kind it comes back near them: within 7 %, 20 %, 19 % at 32 users, 4 %,
0 %, 6 % at 64 (at 1,800 s, −6 %, −19 %, −9 % and −2 %, −4 %, +25 %, the last on 20 reads). So a
think time a kind, three distributions, carries the timing that matters; a gap's dependence on
anything finer was not tested. The eight archetypes do not survive it: they overstate by 20 % to
68 % above ρ 0.12, more as the timing grows more realistic and more at 64 users than at 32. Their
draw holds more main-lane requests (54 % against 50 %) and fewer sub-agent calls (10 % against
15 %), and main-lane requests are the ones that wait long and come back to a context gone from the
pool. They were picked on prompt, writes, and length, under rounds, where none of that showed. Open:
archetypes picked also on the kind mix (or on the reads under timed arrival), or the sessions as
distributions; and the 64-user pass is three seeds. Scratch: `arrive/abt.py` (orders `drawN`,
`kindN`), `grid4.sh`, `grid5.sh`, `summ.py`, `summ5.txt`.

**Utilization, not users, as what a provider fixes (2026-11-09, the user's question; debate, nothing
decided).** The user's challenge: a provider keeps its GPUs busy, so a long think time adds users to
an engine instead of idling it, and a model with a fixed count of users an engine, a quarter of them
idle, is not how inference farms run; the provider balances utilization against the user's wait. The
corpus's gaps also measure start to start, the previous reply's generation included, so every timed
order above treated service time as idle. Proposed in the debate, for the user's decision:

- *The busy hour.* The engine is always fed: the WG fixes the requests in flight an engine (as vLLM
  admits to a full pool) and the think time a request kind; users an engine follow (Little's law)
  and are not a knob. A storage system sized for the average fails at the peak, so the peak is what
  the benchmark models. The pool is full of running requests by construction, so a returning main
  lane after a human gap reads from storage: the low-ρ plateau, the most stable region measured.
- *One axis.* Storage changing users an engine would make the op multiset depend on the system under
  test (the invariant: no work distributed by timing), so it stays fixed; the schedule is computed
  before the run, positional, and storage's speed shows only in latencies. The run passes on the
  stall storage causes (as training's AU) or a KV-load latency budget; the score is engines, or in-
  flight slots if engines are too coarse (the user's concern, from training rounds), with a load
  curve beside it (engines against p99 load latency, SPECstorage's way) and capacity for the idle
  contexts as a requirement.
- *The engine is one TP × PP replica* (GPUs multiply, DP × TP × PP; data-parallel replicas of a dense
  model share no pool and are separate engines; wide expert parallelism couples them).

Primary sources, searched at the user's request (Gemini's answer the user pasted cited vendor blogs
and journalism, none primary; no source gives a frontier lab's GPU utilization):
Mooncake (Moonshot AI, Kimi, arXiv 2407.00079; FAST '25 release) "consistently faces overload" and
rejects early; Splitwise (Azure, arXiv 2311.18677) sizes to a peak under a power cap, SLOs as slowdown
multiples (TTFT 2/3/6×); ServeGen (Alibaba Bailian, NSDI '26, arXiv 2505.09999) bursty arrivals (CV >
1), diurnal, multi-turn reasoning gaps "around 100 seconds, with an extremely long tail", 3.5 turns;
DistServe (arXiv 2401.09670) goodput under SLO; KVCache Cache in the Wild (Alibaba, ATC '25, arXiv
2506.02634) reuse within 10 minutes for 80 % in chat and 10 s in API traffic, reuse time exponential
within a request category (type and turn), a cache of 2 to 4 times HBM enough for its short contexts
(mean 973 tokens single turn, 5,953 multi-turn); Continuum (arXiv 2511.02230) pins a tool call's KV
in HBM for a TTL. Retention, from the providers' own documents: Anthropic's prompt cache lives 5
minutes, or 1 hour at a price, refreshed on each hit; OpenAI's 5 to 10 minutes idle, at most an hour
in memory, or up to 24 hours extended (the default for several newer models), which a provider can
only hold below HBM. No source characterizes agent traffic at production scale: both Alibaba papers
leave it to future work. What it says for the draft: the busy hour is how operators size; a think
time a request kind is supported, possibly an exponential a kind; retention can be set at published
product tiers (5 minutes, 1 hour, 24 hours) rather than invented, the capacity to hold following from
it; and a utilization figure is not needed, since the pass criterion is the stall storage causes.

Public traces fetched (the user's go; Apache 2.0, kept outside the repository in
`~/MLPerfStorage/public-traces/`): Mooncake's FAST '25 release, conversation (12,031 requests, 1 h,
mean input 12,035) and tool-and-agent (23,608, 1 h, mean input 8,596; the arXiv trace again, its
rows the same within a few tokens), and synthetic, 512-token block hashes, no session ids; and
Alibaba's Qwen-Bailian sample (`alibaba-edu/qwen-bailian-usagetraces-anon`, hashes checked against
the LFS pointers), 2 h each, 16-token block hashes with chat and parent ids and turn: to-C (43,058,
6/s), to-B (172,800, 24/s, single turn), thinking (10,812), coder (43,011, 39 % past turn 1, up to 54
turns, mean input 5,748). Their contexts are a tenth to a fortieth of AgentX's.

**Think time measured where a trace links a request to its parent (2026-11-09, the user's go;
scratch).** In the Qwen-Bailian traces a request names its parent, so the user's own time is the
child's arrival less the parent's arrival and the parent's generation; the generation is not in the
trace, so it is estimated, at a TPOT of 20 ms and prefill at 10,000 tokens/s (a fast engine), and at
50 ms and 5,000 (a slow one; it leaves 18 % of the coder's and 69 % of the thinking trace's gaps
negative, so the service was faster than that). Seconds, the fast setting:

| trace | returns | p10 | p50 | p90 | p99 | mean | CV |
|---|---|---|---|---|---|---|---|
| coder | 16,605 | 2.8 | 98 | 924 | 3,470 | 341 | 1.9 |
| to-C chat | 19,957 | 21 | 100 | 710 | 2,194 | 266 | 1.6 |
| thinking | 1,200 | 1.3 | 8.8 | 213 | 1,518 | 92 | 3.9 |

Turn changes it little (chat p50 90 to 120 s across turns 2 to 5+; coder 120 s at turn 2, 58 s past
turn 4). The share back within each published cache lifetime: 5 minutes, 74 % (coder), 76 % (chat),
94 % (thinking); 1 hour, 99 %, 100 %, 100 %. The traces span 2 hours, so a return after their end is
not seen and every share is an upper bound.

What it says. Human think time between turns is about a hundred seconds at the median, a mean of
four to six minutes, and a tail to an hour; generation is a small part of the gap for chat and code
and most of it for reasoning (long outputs, p50 1,665 tokens), so a start-to-start gap, the AgentX
corpus's only measure, must have the service taken out before it is a think time. The distribution is
not exponential: CV 1.6 to 3.9 against 1, a median below and a p90 above the exponential's of the
same mean, so the per-category exponential of the ATC '25 paper (which models a block's reuse
probability, not the gap between turns) is not a fit for think time; a fitted empirical or
heavy-tailed distribution a kind is. For the busy-hour engine: at a mean of about 300 s against
seconds to tens of seconds of service, a user holds a slot a few percent of the time, so tens of users
stand behind each in-flight request. For retention: a 5-minute tier misses a quarter of human returns
and a 1-hour tier nearly none in these traces; the AgentX corpus, whose gaps run to hours (79 % of its
time in gaps over 30 minutes), is where a longer tier would matter, and it is the agentic one.
Scratch: `~/MLPerfStorage/limit500k/think/` (`think.py`, `think.txt`, `ttl.py`, `ttl.txt`).

**The busy-hour engine, closed loop (2026-11-09, the user's go; scratch, no GPU).** C requests in
flight, N users behind them, a free slot taking the user ready longest; service the abstract's compute
model (40 µs a new prompt token, 12 ms an output token); think time the corpus's start-to-start gap
less that service, floored at 0 (56 % floored: the corpus's agent loops fire faster than the model
serves), uncapped; users start mid-stream; 1 h warm, 2 h counted; the whole table, two seeds.
Medians, reads and writes in 256-token chunks (3 MiB each at the scratch's model):

| C | pool | N | busy | wait mean s | reads/req | reads/s | reads ≤ 5 min / ≤ 1 h | tokens held 5 min / 1 h / 24 h |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.5M | 32 | 0.88 | 4.2 | 329 | 163 | 0.92 / 0.99 | 4.0M / 10.7M / 16.4M |
| 8 | 3M | 16 | 0.45 | 0 | 6.9 | 1.9 | 0.01 / 0.72 | 1.8M / 5.7M / 10.9M |
| 8 | 3M | 24 | 0.80 | 0.5 | 25 | 11 | 0.46 / 0.96 | 3.3M / 9.4M / 14.9M |
| 8 | 3M | 32 | 0.88 | 4.2 | 114 | 58 | 0.68 / 0.96 | |
| 8 | 3M | 48 | 1.00 | 36 | 581 | 272 | 0.94 / 1.00 | 7.5M / 16.5M / 22.8M |
| 8 | 3M | 128 | 1.00 | 226 | 720 | 294 | 0.83 / 0.99 | 20.4M / 33.9M / 40.0M |
| 8 | 6M | 32 | 0.88 | 4.2 | 8.1 | 3.9 | 0.00 / 0.80 | |
| 8 | 6M | 64 | 1.00 | 70 | 394 | 184 | 0.82 / 0.98 | 9.6M / 19.8M / 26.7M |
| 8 | 6M | 128 | 1.00 | 226 | 716 | 293 | 0.83 / 0.99 | |
| 16 | 3M | 64 | 0.98 | 5.2 | 434 | 426 | 0.96 / 1.00 | 9.7M / 25.7M / 38.6M |
| 16 | 6M | 64 | 0.98 | 5.2 | 164 | 159 | 0.88 / 0.99 | |
| 16 | 12M | 64 | 0.98 | 5.2 | 14 | 14 | 0.00 / 0.87 | |
| 16 | 6M | 128 | 1.00 | 77 | 669 | 586 | 0.93 / 0.99 | 21.2M / 39.6M / 52.8M |

Writes 12 to 14 chunks a request everywhere; preemption only where the pool is under about twice the
in-flight contexts (1.5M at 8 slots, 3M at 16). Full table: `~/MLPerfStorage/limit500k/closed/summ.txt`.

What it says. The engine fills at about four users a slot (88 % busy at 32 users on 8 slots, 98 % at
64 on 16, the queue a few seconds); past that the closed loop only lengthens the queue (minutes at
eight users a slot), which a provider would refuse (Mooncake's early rejection). Two regimes:

- *Overload* (queue of a minute or more): reads converge to 600 to 720 a request whatever the pool,
  from 1.5M to 12M tokens, the queue itself stretching every reuse past the pool: the stable plateau,
  and Mooncake's operating point, but with waits no interactive service keeps.
- *The busy hour* (busy 0.9 or more, queue of seconds): the reads still depend strongly on the pool a
  slot (at 8 slots and 32 users, 329, 114, 8 a request at 1.5M, 3M, 6M; at 16 and 64, 434, 164, 14
  at 3M, 6M, 12M), so the accelerator's memory past the weights, a WG parameter (item 3), sets the
  load at the point the benchmark would model; the cliff of the earlier notes is still there, now in
  pool a slot rather than pool a user.

Retention at the busy hour, the user's point borne out: 68 % to 96 % of the reads are of contexts
touched within 5 minutes and 96 % to 100 % within an hour, so the long-idle users who hold most of
the idle time contribute almost no busy-hour reads; a 1-hour tier serves the busy hour, and the
capacity it holds is 10.7M tokens an 8-slot engine and 25.7M a 16-slot one (about 125 and 300 GiB at
12 KiB a token), against 1.5 to 2.5 times that for 24 hours. Reads outweigh writes 10 to 30 to one.
Open: the think time here is the corpus's gap less a modelled service; the corpus records each
request's own `api_time`, from which the session's idle time is measured directly (77 % of returns
within 10 s, 96.6 % within 5 minutes, 99.3 % within an hour; 30 % of requests overlap another of their
session, sub-agents in parallel: `think/idle.py`), to be used in a rerun; and the users a slot at the
knee depend on it.

**The value of an old context, surges, and the morning ramp (2026-11-09, debate with the user;
nothing decided).** The user's points, and where the debate went:

- *Every context read, however old, takes compute off the busy hour.* The engine has no spare
  capacity then, so a recompute delays someone else: the value is the user's wait as well as the
  GPU's cost. Measured, a tier of 24 hours over one of 1 hour adds 1 % to 4 % of the busy hour's
  reads (in chunks, so bytes) for 1.5 to 2.5 times the capacity, a trade a buyer prices with their
  own costs. Proposed: the benchmark does no economics (MLPerf publishes no prices) but reports per
  tier the capacity held, the reads served, and the prefill compute they avoided; the tier is the
  WG's, a parameter of the workload, since a submitter choosing it would change the op set.
- *Coordinated demand* (the user's example: water and sewer at a football game's half-time).
  Candidates in agentic traffic: the start of a working day in each time zone (yesterday's sessions
  resumed within the same half hour, each a long-gap return reading its whole context), the end of
  lunch, a provider's recovery from an outage (stalled agents retrying at once), agents scheduled
  on the hour. These are where a long tier pays, the steady busy hour needing only an hour.
- *A resumed request is not read-only* (the user's correction of "little is generated"): it reads
  its whole context, about 780 chunks at 200k tokens, and writes its new input and reply, 12 to 14
  chunks on average (2 %); the agent loop it starts then runs from the pool for seconds at a time,
  reading little and writing 12 to 14 chunks a request, so over a loop of 20 the writes are about a
  quarter of the traffic. The steady busy hour measured 3 % to 11 % writes. Reads are on the
  latency path (the first token waits for them); writes are appends that may leave it (LMCache's
  asynchronous store, to be checked in its source).
- *The morning ramp* (the user's proposal, offered as an example to think through, not an answer): a
  run that models the busy hour from its start, users arriving on an S-curve over two hours, each
  arrival a burst of whole-context reads, then the steady mix. Read bursts follow the arrival rate,
  steepest at the curve's midpoint, while the steady load follows the count arrived, so storage's
  hardest moment is mid-ramp and the GPU's is the plateau, which a steady-state benchmark cannot
  show. It stays CLOSED (arrival times positional, the schedule computed before the run); the
  resumed contexts are a dataset written before the run (an `input` namespace, about 2.4 GB a
  200k-token context at 12 KiB a token, 150 GB an engine of 64 users, so datagen time enters the
  tuning cycle); it needs a long tier for the first reads; its result is a latency curve over time,
  the pass rule the KV-load p99 within budget in every window, the worst window (likely mid-ramp)
  deciding; and its length is real time, two hours a run. The WG would choose, without data (no
  source has time of day), the curve's shape (ServeGen's diurnal rates can inform it), the share of
  arrivals that resume rather than start, and the users at the plateau (the knee).

**The morning ramp, simulated (2026-11-09, the user's go; scratch, no GPU).** The closed loop with
the session's measured idle time as think time (each table row's start less the latest end, by the
corpus's own `api_time`, of every earlier request of its session: p50 2.1 s, p90 37 s, 31 % zero
where sub-agents overlap; `arrive/r_idle.json`). The knee did not move: steady, the engine is 86 %
busy at four users a slot on 8 slots (queue 3.5 s), 95 % on 16 (3.2 s), 100 % at six (36 s); reads
a request at the knee 113 and 9.8 at 3M and 6M on 8 slots, 175 and 13 at 6M and 12M on 16. Below
the knee, with the pool large enough, writes are most of the traffic (50 % to 89 %).

The ramp: users arriving on a logistic curve (centre 50 minutes, 5 % to 95 % in 79), each resuming
mid-session and reading its whole stored context, counted in 5-minute windows over 2 hours, engines
summed as one storage would see them. On 64 slots and a 24M pool (`closed/rsumm.txt`):

| users | window | busy | queue mean | reads, MiB/s | arrivals' first reads | writes |
|---|---|---|---|---|---|---|
| 192 (3 a slot) | 20 min | 0.15 | 0 | 197 | 85 % | 22 % |
| | 45 min (peak arrivals) | 0.65 | 0 | 455 | 61 % | 32 % |
| | 80 min | 0.93 | 0.4 s | 1,931 | 4 % | 10 % |
| | 110 min | 0.84 | 0 | 1,900 | 2 % | 11 % |
| 256 (4 a slot) | 45 min (peak arrivals) | 0.86 | 0.1 s | 650 | 36 % | 30 % |
| | 70 min | 1.00 | 11 s | 6,212 | 3 % | 5 % |
| | 95 min | 1.00 | 14 s | 9,257 | 0 % | 3 % |

The 8- and 16-slot ramps agree (16 slots, 64 users: 346 MiB/s at peak arrivals, 2,000 to 3,100 on
the plateau, writes 22 % then 4 % to 6 %).

What it says, against what the debate expected. Storage's hardest moment is not mid-ramp: the
arrivals' whole-context reads, at the arrival peak, are a fraction of what follows once the engine
nears saturation, when the pool can no longer hold even the contexts of users back within seconds
and every return is read again: four to five times the mid-ramp's reads at three users a slot,
fourteen times at four. The ramp's start is write-heavy, not read-heavy (20 % to 45 % writes, the
arrivals' loops appending while the pool still holds them); the user's 90/10 appears at the plateau
(3 % to 15 %). Arrivals' first reads, the ones a long tier serves, are 6 % to 18 % of the run's reads.
And the read rate near the knee is steep in the users a slot (1.9 against 6 to 9 GB/s for 192
against 256 users on 64 slots), as steep as the pool's cliff was, so the operating point, users a
slot and pool a slot, decides the load more than the shape of the morning; it is better fixed by a
queue-wait target (the latency a provider keeps, and Mooncake's rejection at overload) than by a count
of users. Caveats: the closed loop lets the queue grow where a provider would shed load, and the
queue's wait lengthens every gap, which feeds the churn; one think-time model, one corpus.

**Sessions as trees of concurrent contexts (2026-11-09, the user's go; scratch, no GPU).** The busy-
hour engine on the corpus's own sessions rather than the scratch abstract (which serializes
sub-agents): each request ready by its real trigger, the end of the latest earlier request of its
session that had ended, plus the real idle, or, where it ran beside every earlier one, the start of
the latest started plus the real offset; 16 slots, the service model of the closed loop, the pool
model and LMCache's accesses over the schedule, 1 h warm and 2 h counted, two seeds, the same
sessions in every order (`closed/tree.py`, `tsumm.txt`). Orders: `serial` (a session one request at a
time), `tree` (the corpus's own concurrency, depth 1), `tree2` (a what-if: each sub-agent with
probability 0.37 spawns, at its middle request, 1 to 8 children copied from the corpus's sub-agents,
and waits for them). At a 6M pool (375k tokens a slot):

| sessions | order | busy | queue mean s | reads/req | reads/s | writes % | contexts a session, p99 |
|---|---|---|---|---|---|---|---|
| 64 | serial | 0.77 | 0.2 | 55 | 54 | 19 | 6 |
| 64 | tree | 0.79 | 1.2 | 45 | 46 | 23 | 4 |
| 64 | tree2 | 0.94 | 8.5 | 116 | 154 | 13 | 8 |
| 96 | serial | 1.00 | 12 | 283 | 355 | 5 | 6 |
| 96 | tree | 1.00 | 22 | 425 | 539 | 3 | 5 |
| 96 | tree2 | 1.00 | 58 | 393 | 590 | 4 | 12 |

At 12M the same shape lower (64 sessions: 9.1, 8.2, 9.2 reads a request). What it says: at the depth
the corpus has, its concurrency changes little below the knee (the reads within 20 %; the queue
burstier, 1.2 s against 0.2); the knee in sessions is between 64 and 96 on 16 slots, four to six a
slot, agreeing with the abstract's four users a slot. One level more, the what-if, moves it a
quarter lower (busy 0.94 at 64) and triples the reads a second at 64 sessions, so the depth and
fan-out of item 10's tree are high-leverage parameters for the WG, and the measured depth 1 is the
default only because nothing deeper is measured. The contexts a session holds were sampled once a
minute, which misses short peaks.

**The morning ramp with people reviving several sessions (2026-11-09, the user's go; scratch).** Item
6's revision tested: the tree of the note above, people arriving on a logistic curve (5 % to 95 % in
30 minutes, centre 25), each reviving K sessions spread over 5 minutes, each session resuming
mid-way and reading its stored context; the same sessions in every K, only their grouping changed;
16 slots, 6M; 0 to 90 minutes in 5-minute windows, three engines summed (`closed/clustered.py`,
`csumm.txt`). Reads in thousands of chunks a window:

| sessions | K | saturated from | most reads before 35 min | worst window | its reads |
|---|---|---|---|---|---|
| 64 | 1 | 30 min | 78 | 40 min | 250 |
| 64 | 3 | 30 min | 81 | 45 min | 200 |
| 64 | 8 | 25 min | 96 | 45 min | 322 |
| 80 | 1 | 30 min | 116 | 45 min | 468 |
| 80 | 3 | 25 min | 183 | 55 min | 467 |
| 80 | 8 | 25 min | 174 | 45 min | 507 |

What it says. The user's expectation holds in direction: grouping steepens the ramp, the engines
saturating about 5 minutes sooner and the reads before saturation 20 % to 58 % higher. The worst
window is still past saturation (40 to 55 minutes), the churn of a full pool, and rises less, 8 % to
29 % at K = 8 (with K = 3 at 64 sessions lower than K = 1, within the noise of three engines). The
arrivals' own first reads stay small (at most 27 thousand a window); the clustered returns matter by
filling the pool sooner. So K belongs in the workload, and its default (the user's 3 to 8) moves the
ramp more than the plateau.

*Corrections made during the debate:* "the replay skips over-limit requests" was first presented as
the main cut, when the share cut far more; "long chains grow slowly" was then called a side effect
of the cut, too strongly, since the corpus accounting already records it. That a long chain reaches
the limit near turn 17 to 19 and restarts, offered as the cause of the short chains, is refuted by
the table above.

*Kept outside the repository:* the draft, the scripts, and the dataset card, in
`~/MLPerfStorage/pool-experiments-2026-11-06/`.

**The structure proposed for the KV-cache workloads (draft for the user's review, 2026-11-08;
nothing decided, nothing built).** (Superseded by the consolidated draft of 2026-11-09 below; kept for
the reasoning it records.) What the notes above add up to, as proposals the user decides one
by one; the numbers are the scratch tests' (AgentX, LMCache local disk, the pool model), and what WG
process means here is §8's.

*1. What a KV workload is judged against.* No measurement of a real service's storage exists; the
corpus has no storage I/O, and a GPU large enough to replay it is not to hand. The reference is the
pool model (`agentx.simulate` with `lmcache_accesses`) on the corpus's requests and blocks: on the one
capture with no length cut (500k, 700 requests) its serial timeline came within 2.5 % of the traced
reads, and the engine timeline 46 % under, so the serial one. A GPU capture is a check of the calls
and of the pool model at a point, not the target. Consequence: the committed AgentX target (2,321
reads at 6 GiB) is an artefact of the 131k cut (184,972 without it), so the case's tolerances and
the §3.64 note of 2026-11-06 ("the 6 GiB pairs accepted again", `6fc276c`) are corrected by this
section, and the case is retargeted against the pool model once the rest is decided.

*2. One abstract per storage mechanism, the model a parameter.* As brief item 70 proposes:
`lmcache-local-disk` and `lmcache-shared-fs` stay one abstract each, the attention family a parameter
(GQA's chunk split across tensor-parallel ranks, a file each; MLA's whole chunk from the first rank),
not an abstract per family. Open there: a TP 2 capture, to see whether the ranks' calls need a
structure values cannot express.

*3. The deployment is the WG's, as parameter files.* KV bytes a token (the model and its KV dtype),
pool tokens per engine (the accelerator's memory past the weights), tensor and pipeline parallelism,
users per engine, the CPU tier (0), all fixed by a case derived from a model, an accelerator, and a
parallelism; a submitter chooses none of them (the user's view, agreed: a buyer compares systems by
the users they serve, not by deployment tuning).

*4. The operating point, a ratio set well below the cliff.* The reads are set by ρ = pool tokens /
(users per engine × mean live context). Measured at 8 users and a mean prompt near 195k: ρ ≈ 0.34
(0.5M) stable, the whole table's seeds within two to one, eight archetypes within 3 %; ρ ≈ 0.67 (1M)
already three to one between seeds; ρ ≈ 1.3 (2M) the cliff, 32 to one; ρ ≈ 2.7 (4M) and above,
almost no reads. A real engine sits lower: 64 users on 1.5M tokens is about 0.12. Proposed: the WG
fixes ρ at or below about 0.3, users per engine large enough to be realistic. (Revised after the check
below 0.34, the note above: from ρ 0.08 to 0.49 the reads move 4 % at most, so ρ need only be set
well under the cliff near 1, about 0.25 for margin, and the accelerator's exact memory does not
change the I/O; 32 users or more per engine, where the seeds spread 5 % to 12 % against 66 % at 8;
open, whether the rounds' worst case for LRU holds under out-of-order arrivals and idle time.) (Revised again after the arrival test, the note above: the flat region was
the rounds' artefact; under random or the corpus's arrival the reads fall steadily with ρ, so the WG's
ρ sets the load and must be chosen for it, and is counted on the users active at once, which the
think time decides. Proposed: a think-time distribution fitted to the corpus's gaps, capped where the
WG decides a user has left (five minutes, the provider's cache lifetime, is one choice), drawn apart
from the sessions; the storage tier's retention of the contexts of users who left, a separate
question, recorded under *Not settled*.) (Tested, the note on a drawn think time: drawn from all
gaps pooled it halves the reads at ρ 0.49; drawn from the gaps of the next request's kind it comes
within 20 % of the corpus's times at 32 users and 6 % at 64. Proposed: three think-time
distributions, one a request kind.)

*5. The submitter's knob: engines.* Users added to a fixed engine move ρ and with it every user's
I/O, so the WG fixes users per engine and the submitter scales the number of engines; the result is
the users served, engines times users per engine, as training scales accelerators at a fixed load
each. What decides that a run passed is a latency budget for a KV load (as training's accelerator
utilization), not yet defined. The request stream's draws are keyed by engine (item 70).

*6. Sessions as archetypes.* Eight sessions of the corpus, picked by the time they hold a slot (bins
of equal request count by mean prompt, from each the session nearest the bin's request-weighted
mean prompt, writes a request, and length) and drawn with weights, a slot playing one session after
another, users out of phase by a stagger: 4,276 rows of five integers, about 4 % of the corpus, came
within 3 % of the whole table's reads and writes below the cliff, and held out within 15 % where the
halves agree. Picked from the whole corpus, with the mean prompt stated beside them, since it sets
the reads nearly in proportion and the corpus itself fixes it only to perhaps ±10 %. Perturbing
tokens added nothing. Open: whether a table of rows (array parameters, built and run in the scratch
abstract) is compact enough for the rule that an abstract is never a trace listing, or the eight
sessions should be fitted as distributions (untested). (Revised after the drawn think time, the
note above: with a think time drawn a kind, which the corpus's timing needs, the eight overstate the
reads by 20 % to 68 % above ρ 0.12, their draw rich in main-lane requests; the 3 % holds only
under rounds and random arrival. The pick has to match the kind mix as well, or be tested against
timed arrival, before this item stands.)

*7. Reproducibility of the stream.* Every draw is positional, so a seed gives the same ops; between
seeds the reads of eight archetypes at ρ ≈ 0.34 spread about ±10 %. Proposed: the WG fixes the seed
of the request stream, as it fixes the dataset seed, so that every submission runs one op set;
larger runs (many engines) narrow the spread for anyone who varies it.

*8. Tolerances.* Writes held tightly (they matched within 1 % to 6 % in every test); reads judged at
the WG's operating point only, against the pool model, with a tolerance from the spread between
seeds there; no target set at the cliff.

*9. The runner.* An `x @ (r − dd)` reaching back thousands of requests overflowed the runner's stack
at 2,000 requests a slot (worked around with `RUST_MIN_STACK`); a valid abstract must not need that,
so a brief item to make that evaluation iterative or its depth bounded.

*Not settled by any of this:* the think time and what a user away for long means for the storage
tier's capacity and retention (the arrival note); the pool model's eviction is checked at one point, one model, one
GPU; requests over 485k tokens (FP8 at 800k slowed past use); idle time between requests (the
corpus's `ttft` shows a provider's five-minute cache; the replays ran without pauses); compaction of
long sessions; the publisher's cache columns (the user's call); one corpus, one publisher's agents.

*Order proposed:* the user decides 1 to 9; then the check below 0.34; then brief items for 6 (the
session table in the shipped abstract, or distributions), 9 (the runner), and the retargeted case
with its tolerances (8), item 70 carrying 2, 3, and 5; then commit and push together.

**The session forms tested against the whole table (run at the user's word, 2026-11-09, for item 10
of the draft below; nothing decided).** Three forms through the closed-loop tree of the notes above
(16 slots, measured idle, seeds 1 to 6; steady at 32, 64, 96 sessions, pools 6M and 12M tokens; the
ramp with people reviving 3 sessions each at 64 and 80, pool 6M): the whole table, eight archetypes
picked again to match the main-agent share (0.576 against 0.575, 6,214 rows), and a generator of two
levels, a session class of 8 by mean prompt and, within it, turns, chain steps, fan-out, and timing
from quantile tables. The generator took about 4,250 numbers, not the few dozen estimated, and three
additions before its prompts matched: a side query's return to the large context, a session's own
share of deep cuts, and a factor per class on growth (1.0 to 2.2, and 4.1 for the largest-prompt
class), fitted in-sample. Against the line proposed beforehand (reads within about 20 %, writes
within about 7 %) neither form passes. The archetypes fail by far: their load is higher (busy 0.64
against 0.32 at 32 sessions, the cause not looked into), the knee comes at about half the sessions,
and reads are +62 % to +1,673 % wherever that matters. The generator's reads are within 20 % in 3 of
the 8 cases (−55 % to +84 % in all) but inside the whole table's own range over seeds in 7 of 8; its
load is 10 % to 15 % light (busy 0.65 against 0.77 at 64); its writes are +56 % to +93 % in all 8. The
cause of the writes is measured (`sess10/wr.py`, chunks a session stores for the first time, a
request): 14.4 in the corpus, 20.9 in the generator uncalibrated, 29.5 calibrated. The calibration
doubles the extending steps' writes (main agent 9 a step, 22 calibrated), so growth is the wrong lever
for prompt size, and what makes long sessions large is something else, not yet identified; uncalibrated,
the excess is sub-agents' cuts (218 a step against 30) and first prompts (61 against 31). The whole
table's own reads spread widely between seeds at one engine (13 to 46 reads a second at 64 sessions,
112 to 569 at 96, pool 6M), which bears on item 12: a band of two standard deviations at the knee, for
one engine, is wide. Scratch: `~/MLPerfStorage/limit500k/sess10/` (`summ.txt`).

**The slot limit, and a second generator (run at the user's word, 2026-11-09; nothing decided).** The
whole table with no slot limit against 16 slots (item 13's question), seeds 1 to 6 (5 at 96 sessions,
one killed for memory): at 64 sessions steady nothing moves (reads +1 % at pool 6M, −2 % at 12M, writes
−3 %); once queues form it does (96 sessions, reads −57 % and −29 %, writes +13 %; the ramp at 64, reads
−26 %, peak window −37 %, writes +12 %; at 80, reads −46 %, peak −31 %, writes +36 %). The ramp's worst
windows come after saturation, so the slot queue stays in the plan, in the same pass over the merged
timeline as the pool (proposed). A second generator (`sess10/gen2.py`, about 5,000 numbers, in-sample)
with no factor per class: the first's errors were structural. A cut re-sends content already stored
(a sub-agent's deep cut, 104k of 117k tokens), and is fitted now by what it stores; the session's real
start is its second prompt, a table per class; growth is front-loaded (the first tenth of a chain 1.56
times the mean) and drawn by the steps a context took; a compaction comes at a size (per class, a table
of sizes and the share of tall sessions that never compact), and the largest effect, a side prompt
leaves a context's main chain where it was. Static fit: mean prompt 209k against 219k, each class −19 %
to +2 %; chunks a session stores for the first time, a request, 17.2 against 14.4. Through the same grid,
against the whole table: reads within 20 % in 4 of the 8 cases (−3, +3, −8, −10; the others −28, −29,
−39, and +70 at a rate of about one read a second), inside the whole table's range over seeds in all 8;
writes +13 % to +28 %, still outside 7 %, inside the table's range over seeds in 5 of 8; load near
(busy 0.70 against 0.77 at 64 sessions). The seeds' spread (the generator's standard deviation 13 % to
53 % of its mean at one engine) means 6 seeds cannot resolve a 20 % difference in reads. The remaining
writes are measured (`wr2.py`): fresh prompts (the main agent's 117 chunks a step against 68, a sub-
agent's 34 against 21) and a sub-agent's first prompt (55 against 31), written as new where the corpus
re-sends stored content, the error cuts had. Scratch: `summ2.txt`, `ssumm.txt`.

**The third generator, twelve seeds, and held out (run at the user's word, 2026-11-09; nothing
decided).** `sess10/gen3.py`, a fresh prompt and a sub-agent's first prompt without the shared prefix
now fitted by what they store, about 5,070 numbers. Against the whole table, 12 seeds each, the
standard errors of the difference given: writes within 7 % in 6 of the 8 cases (+3, +3, −2, −2, +2, +2;
+18 ± 3 at 96 sessions in both pools). Reads within 20 % in 3 of 8 (−8 ± 19, −13 ± 25, +1 ± 12), and
low where they matter: −36 ± 22 at 64 sessions and pool 6M, −55 ± 8 and −29 ± 10 at 96, −30 ± 18 and −51 ±
8 on the ramp at 64 and 80, its peak window −22 ± 24 and −35 ± 8. Twelve seeds resolve what six did not:
the low reads are a bias. Load does not explain it alone: busy 0.64 against 0.77 at 64 sessions, but
0.73 against 0.75 and 0.78 against 0.78 on the ramp, where reads are as low. The prompts in the
simulated window are 9 % to 16 % smaller than the table's though the static mean is within 5 %. The
cause is not established. Held out (fit on the even or the odd sessions, 64 sessions at 6M and 12M and
the ramp at 64, 8 seeds, against each half's whole table): the reads held out (−25 ± 25, −81 ± 5, +8 ± 17,
−41 ± 9, −36 ± 10, −77 ± 5) are no worse than in sample against the fitted half (−57, −67, −19, −22, −66,
−56, each ± 7 to 15), so the error is bias, not overfitting, and the halves differ among themselves;
writes held out −2 ± 10, −19 ± 6, −2 ± 10, −19 ± 6, −3 ± 6, −9 ± 5. The shrinking of the tables was not
done, the form having not yet earned it. Scratch: `summ3.txt`.

**The low reads traced to the idle gaps, and a fourth generator (run at the user's word, 2026-11-10;
nothing decided).** The whole table was played with its own trigger gaps disturbed and nothing else
(`sim4.py`, 6 seeds, pool 6M). Permuted within each session, main and sub-agent gaps apart, reads fall
30 % at 64 sessions, 39 % at 96, 28 % on the ramp at 80. Drawn independently from the whole corpus's
gaps, as `gen3` draws them, they fall 46 %, 58 %, 57 %, and on the ramp the table so played reads what
`gen3` reads (92 against 92 a second) at the same busy fraction (0.78). So the bias was the gaps: which
requests carry them, not their distribution. The corpus's main lines cluster them: a gap of a minute or
more is 14.9 % of gaps but 38.1 % after such a gap, and a session's own share of them runs from 4.8 % at
the 10th percentile to 32.4 % at the 90th. Prompt size barely moves them. `sess10/gen4.py` draws a
session's share of long gaps (by quartile of its main-line length, weighted by gaps), a persistence
(0.185 main, 0.198 sub-agents), and separate tables for the short and the long; about 5,340 numbers. Its
gaps: long 16.6 % and 38.7 % after a long one; the mean main-line gap 497 s against 395 s. Against the
whole table, 12 seeds, the difference and its standard error: reads within 20 % in 7 of 8 cases and
within two standard errors in all 8 (−6 ± 21, +15 ± 33, +2 ± 22, +12 ± 14, +18 ± 18, +23 ± 26, ramp +17 ±
33, +8 ± 17; the ramp's peak window +35 ± 36, +22 ± 12). The read bias is gone. Writes are not: +36 ± 13
at 32 sessions, +34 ± 9 at 64, +24 ± 3 at 96, +6 ± 5 and +1 ± 4 on the ramp, 2 of 8 within 7 %. Requests a
second against writes per request show why, and correct the previous note: `gen3`'s writes matched by
cancellation, its rate 14 to 19 % low at 32 and 64 sessions and its writes per request 20 % to 21 % high
(15.8 against 13.2); `gen4` has the rate back (+10 %, +5 %, 0) and keeps the excess per request (16.2 to
17.0 against 13.1 to 13.8; on the ramp +12 % and +8 %). The excess per request is the open fault, in the
prompt structure and not the gaps; not diagnosed. Held out not repeated for `gen4`, nor the tables shrunk.
Seven of the first 72 jobs failed on a request of no new tokens and no output, its end sorted before its
start, exposed only by permuted gaps; `sim4.py` gives it a nanosecond and they were run again. Scratch:
`summ4.txt`, `summ5.txt`, `idle1.txt`, `wr5.py`.

**The writes per request, a fifth generator (run at the user's word, 2026-11-10; nothing decided).**
New chunks per request by the fit's own step kinds, no simulation (`sess10/wr7.py`), and each generated
step against the kind the classifier sees (`xt.py`), located the excess of `gen4` (16.5 against 14.4). A
deep cut was drawn at any size, where the corpus's are of tips small against the session's largest (of a
large one it is a compaction); trims and deep cuts shared one table of new tokens though they write 4 and
159 chunks; a fresh prompt, a sub-agent's first without the shared prefix, and the session's second
prompt took a stored share independent of their size, which is biased when the share and the size go
together. `sess10/gen5.py`, about 5,840 numbers: a deep cut only of a small tip, its own table for trims,
and those three prompts fitted by the tokens they write. Static writes per request +7 % (15.4 against
14.4; main lines 16.4 against 14.9, sub-agents 13.8 against 13.8). Busy now agrees (0.36, 0.77, 1.00,
0.75, 0.78 against 0.33, 0.77, 0.99, 0.75, 0.78). Against the whole table, 12 seeds: writes +20 ± 10 at 32
sessions, +16 ± 8 at 64, +12 ± 3 at 96, −5 ± 4 and −6 ± 3 on the ramp at 64 and 80, so 2 of 8 within 7 % (the
pools share a run's writes, so 5 distinct cases). Reads −15 ± 18, −14 ± 15, −25 ± 20, −10 ± 14, −2 ± 21, +2 ± 27,
ramp −33 ± 16, +1 ± 16, its peak window −19 ± 25, +11 ± 11; 6 of 8 within 20 %, 7 of 8 within two standard
errors. Half of `gen4`'s write excess is gone and the ramp holds; the steady cases do not. The steady
window's writes per request are 15.4 to 16.0 for `gen5` and 13.1 to 13.8 for the table, whose own static
figure is 14.4: the table's window runs below its static mean and the generator's does not. Why is not
diagnosed. Held out not repeated, nor the tables shrunk. The round was to stop here, the steady writes not
closing. Scratch: `summ6.txt`, `wr7.txt`, `wr8.txt`.

**The fifth generator held out (run at the user's word, 2026-11-10; the user chose the generator over the
whole table as a data file, this test being its condition).** Fit on the even or the odd sessions, 8
seeds, 64 sessions at 6M and 12M and the ramp at 64, each against each half's whole table. Writes held
out: +24 ± 7 and +4 ± 7 steady, −1 ± 4 and +1 ± 7 on the ramp; in sample +35 ± 8, −4 ± 6, 0 ± 5, 0 ± 7. The ramp's
writes hold for sessions the fit never saw; the steady ones depend on the half. Reads held out: +33 ± 36,
−82 ± 4 (6M); +18 ± 17, −52 ± 8 (12M); +41 ± 46, −75 ± 5 (ramp). In sample −23 ± 22, −69 ± 7; −11 ± 14, −36 ± 10; −25 ±
26, −52 ± 7. Reads do not generalize, and the two fits differ in themselves: fit on the even half,
reads 43.4 a second and busy 0.90; on the odd, 10.2 and 0.68, a factor of four in reads for the same
sessions' structure at half the data. The fit on all the sessions is within 20 % of its table in 6 of 8
cases; a fit on half can miss its own half by 69 %. The halves' own whole tables differ too: 56.7
against 32.7 reads a second at 64 sessions and 6M (1.7 times), 94.1 against 50.1 on the ramp (1.9 times), 7.4
against 5.6 at 12M (1.3 times). So near the knee the corpus's own sampling error in the reads is far
larger than the ±20 % item 12 carries for it, at least at 196 sessions a half. What the generator can claim
is therefore narrow: it reproduces the pool model on this corpus's fit, not reads for sessions outside it.
Scratch: `summ7.txt`.

**The closed loop with load shed (run at the user's word, 2026-11-10; ~~nothing decided~~ decided below).** The whole
table (`sess10/sim6.py`, 6 seeds, the same seeds as the queueing loop) with a stand-in provider: a request
found past a queue-wait limit when a slot comes free is refused, retried after 30 s, 60 s and 120 s, and
on a fourth refusal its session is abandoned, the stream going on to its next session. Limits 10 s, 30
s and 120 s, the queue-wait target being the WG's. The served plateau hardly moves: at 96 sessions 1.219
requests a second queueing, 1.194 at 30 s, 1.177 at 10 s (−3 %), busy 0.98 to 0.97; the ramp at 80 the same
(0.940 to 0.950). Refused requests come back, so the GPU stays as full; the closed loop's plateau is not
overstated in what is served. Writes do not move (within 2 % steady, +9 ± 5 on the ramp at 10 s). Reads do,
past the knee: at 96 sessions −26 ± 25 at 30 s and −74 ± 7 at 10 s (6M), −23 ± 21 and −55 ± 11 (12M); on the ramp
−18 ± 21 and −59 ± 9, its peak window −14 ± 19 and −62 ± 8. Why is not diagnosed. At 120 s nothing changes, no
wait reaching it (the p99 is 55 s to 58 s). At 64 sessions, below the knee, nothing changes at 30 s or 120
s and reads move −1 % to −3 % at 10 s. What the user sees: at 96 sessions and 10 s the mean delay falls from
23.3 s to 13.4 s and the p99 rises from 58 s to 141 s with the backoffs, 718 refusals and 12.5 abandoned
sessions an hour; at 30 s, 232 and 2.2. So the provider's queue policy is a parameter of the op set past
the knee, as the pool size is: the WG fixes it, or the operating point sits below the knee, where it does
not matter. The retry and abandonment rules are a stand-in, no trace giving them. Scratch: `summ8.txt`.
(Decided by the user, 2026-11-10: the operating point sits below the knee, where the queue policy does
not move the op set and the reads are best behaved; no queue policy is fixed in the workload, since one
would need retry and abandonment rules no trace gives. Item 4's queue-wait target is set with this as
its bound.)

**The generator in the abstract, a shared tier, and events between runners (the user's questions,
2026-11-10; decided by the user the same day).** Two things were called "the generator" loosely. The
*session generator* (`gen5` in the scratch) draws a session: its tree of contexts, each request's time
and what it sends (which earlier prompt it extends or cuts, the new blocks). The *plan pass* takes an
engine's sessions in planned order and runs the slot queue, the pool, and eviction over them, which
decides each request's reads, writes, and unlinks: the op set. Training has only the first kind of
step (the Feistel shuffle and the position formula); the second is new, a request's ops depending on
what the other sessions sharing its pool did before it. Both are deterministic and computed from the
plan, never from the run: the pool is filled by planned requests, not by the I/O a runner issued
(item 13's rule (1)), so a host plans alone for what it owns. The cost differs from the shuffle's: LRU
contents have no closed form, so position *i* needs the planned history before it. A runner streams
the plan: each session's generator yields its next request on demand, a heap merges the sessions in
planned order, and the pass consumes them, so a host holds each session's state and the pool, never
the whole timeline. Sessions an engine are about four a slot at item 4's point, not one.

Decided:
(1) *Approach (a).* The session generator and the plan pass are expressed in the abstract language,
the runner kept agnostic to the workload, rather than (b), a plan node with `gen5`'s logic in Rust.
First, a port test: `gen5` written as an abstract with today's language and the fewest additions,
held to the Python program's output (as the `aeiou-trace` port was held to its goldens); what the
port cannot do without is the list of constructs the language needs. Expected: a prompt as parent,
kept length, and new blocks; a choice among earlier prompts; `x @ i` without the stack (brief item
73's); and for the engine a timeline merging its lanes in planned order, a queue with a fixed number
of servers (the GPU slots), the pool, and watermark eviction. The timeline is the largest change: the
language's actors are independent today, and a timeline couples an engine's lanes, before the run.
New constructs are marked provisional in the contract until a second workload uses them; the user
expects the language to grow quickly while workloads are explored and slowly after.
(2) *The cache construct designed from two systems.* LMCache writes every new chunk to its store when
made (writes follow new tokens); Dynamo's block manager writes a block to the tier below when it is
evicted from the one above (writes follow evictions, cascading). The construct is a stack of tiers,
each with a capacity, an eviction policy, and a write policy, the stack with a rule for what a hit
promotes, named for the mechanism; LMCache's settings built first, Dynamo's source read before the
shape is fixed. Where a tier puts a block (a file a chunk, or a slot on a device) is a parameter. A VDB
caching hierarchy is expected to need it too; if a later system does not fit, a new construct.
(3) *Toward `lmcache-shared-fs`, in phases.* Phase 1: each session's contexts on its own engine, the
storage tier one shared namespace, the plan pass per engine. Phase 2: contexts routed across engines
by a WG-fixed rule (prefix hash and planned load, as llm-d, Dynamo, and OpenAI's routing document; no
trace shows a router in use, an assumption stated). Measured in the scratch before phase 2 is built:
the reads that would cross engines under prefix routing on the corpus, and the plan pass's cost at
the domain sizes proposed. Placement is a pure function every host evaluates; session ids are global
and positional, keyed by (seed, session), never generated by a runner, so no runner is told a session
moved: each issues what the plan assigns its engines.
(4) *Domains.* A domain is a sharing group: the engines sharing one storage tier (in practice the
replicas serving one model, whose KV no other model can use). Sessions route only within their
domain and no state crosses domains, so a host replays only its domains' plans, the state it holds
bounded by the domain, and adding a domain leaves the others' ops unchanged (item 9's property, at
domain granularity). A domain may span hosts (the user's preference over one host a domain), and is a
barrier and event scope within the run's one coordinator, which keeps one identity check, one `t0`,
and one merged report.
(5) *Events between lanes.* The user's design: an op at the head of a causal or validity edge signals
an event; the op at its tail waits for it, consuming it if it came early. Refined: the abstract names
engines and lanes, never runners or ranks (which engine a rank runs is the GPU-range formula, so any
host computes where engine *e* is, and the coordinator, which every host connects to, knows each
rank's address); event ids are computed from the edge's keys, never counted; a causal event carries
its sender's lag L, a validity event only completion; one event a request's stores, not one a chunk.
Delivered in-process when both ends are on one host, by the coordinator otherwise, two messages
through it an event (in, then out), so 1,000 events a second are 2,000 messages. The time an event
spends between hosts is the benchmark's, never charged to the system under test: the coordinator's
one-way latency is measured and reported with the edge waits, and a run whose messaging p99 is a
meaningful share of the latency budget is invalid. TCP behind the `Coordinator` trait, so an event is
lost only with a host, which fails the run; a wait has a timeout, a failed run rather than a hang.
Edges point forward in planned time, so no deadlock. The same primitives serve any cross-actor
dependency (a checkpoint's manifest after its shards), point-to-point beside the collective barriers.
They change when ops issue, never which, so the fingerprint and CLOSED are untouched. The coordinator
stays in rank 0's process; a standalone one comes on evidence that rank 0's own load delays events,
the messaging measurement being that evidence. A launcher written in something other than shell could
stay alive after starting the ranks and take the coordinator's role (the user's note).
(6) *State bounded by capacity.* The rule against per-file structures means state that scales with the
dataset; a cache's state scales with its WG-fixed capacity, holding only what the cache holds, and is
allowed. A host replicating a domain's plan holds the domain's tier, about ~~100,000~~ 40,000 to 58,000
files an engine at an hour's retention (corrected the same day, the note "LMCache's multi-process
storage read, and the pre-fill's datagen time"), so the domain's size bounds it.
(7) *The runner is the only reproducer.* No one needs the op set without the runner; who wants it
exports the run's trace and analyses that. The cache's rules (LRU ties, the order within a planned
instant) are definitions the runner fixes in `runner/REFERENCE.md`, a change to one changing every KV
fingerprint. The Python pool model stays the reference for item 12's tolerances.
(8) *A rule: never learn from the run.* Which ops a run issues, and their names, sizes, and offsets,
are a function of the abstract, its parameters, the seeds, and the host and engine counts; nothing
observed during the run changes them, and only when an op issues may follow the run. It names what
the fixed op multiset, "no work distributed based on timing", and item 13's "track the slowness,
neither control it nor respond to it" already enforce. It forbids modelling adaptive behaviour (a
router balancing on real queue depth, a store dropped when staging fills, client timeouts); those
become gate failures or reported numbers. Added to `CLAUDE.md`'s invariants with the commit.
(9) *Dynamo's G3.5, deferred.* The user's understanding, not yet checked against the source: G1 HBM, G2
host DRAM, G3 node-local NVMe, G3.5 an NVMe-oF block layer reached from user space, G4 an archive by an
object API. If so, supporting it needs block-device ops in the language, since a shim cannot turn file
or object calls into block calls, and the extent index it keeps must be computed (an allocator's state,
in the plan) and, if persisted, its writes modelled. A kernel NVMe-oF device would instead be POSIX at
offsets. Queued; the call path and the index read from the source first.

**Demand, the application's caches, layout: one model for every family (the user's question,
2026-11-10; accepted by the user the same day).** The user's concern: the tool's definitions began
myopic toward training, KV has pulled them toward generality, and the KV work, without a high-level
model like training's (pick a file, read it whole, repeat), was arduous and may have overcorrected
toward KV. Reviewed against brief items 70 to 76 as drafted: none harms training or checkpointing as
written, but three could if built the obvious way, so each carries a guardrail. Item 70's keys are
the abstract's choice (GPU id for every abstract outside `kvcache/`), the invariant's wording changed
for the KV abstracts only; item 72's timeline is opt-in, an abstract without one running as today
with no per-op cost, and D and the lag are a case's results through `result.` constraints, never
runner-wide; item 73 changes evaluation, never values. Each is accepted only when every committed
abstract outside `kvcache/` keeps its fingerprint and measured shape (the catalog check) and a
training run is no slower. `until_eof` was checked against the rule "never learn from the run": it
stops at the computed size, so training already obeys it. For vector search the items help more than
they hurt: the cache construct fits DiskANN's node cache (a static tier, modelled by hand today) and
tiered segment caches over an object store; the timeline and the latency gate fit query serving. The
graph walk, a read depending on the neighbours just read, none of them addresses. A trap named: KV's
chained prefix keys belong to the KV generator, the cache's keys opaque.

The cause of the wilderness, as read here: training's demand was trivial, so its architecture was
almost all layout; KV's difficulty was almost all demand and caching, discovered from a corpus before
any file op existed. One model covers every family, three layers, each abstract placing its
structure in one:

| Family | Demand | The application's caches | Layout |
|---|---|---|---|
| Training | a steady step loop, reads a Feistel draw | none, or an epoch cache; every read reaches storage | formats and framing |
| Checkpointing | writes on a schedule, then barriers | none | shards and a manifest |
| KV cache | arrivals, sessions, think times | the pool and the storage tiers | a file a chunk, or slots on a device |
| Vector search | query arrivals and the graph walk | node and segment caches | sectors and segments |

Training is the simple case, steady demand and no cache. Items 70 to 76 sit as demand (71, 72),
caches (75), and layout (76); a construct that fits no single layer is a sign it is in the wrong
place. The joint review of the KV structure checks it against this frame.

**The joint review of the KV structure (started by the user 2026-11-10; decided as below the same
day).** The structure laid out on the frame above: demand (the ramp and returns, the sessions of item
71, the engine's service model and the operating point below the knee, the keys); the application's
caches (the GPU pool, the staging of stores, LMCache's L2 tier and its eviction, the construct of
item 75); layout (a file a chunk named from its key, one flat directory, `rename` and `unlink`; the
two mechanisms; the block tier later); and across them the plan, events, D, the score, and the
tolerances. Issues found and decided:
1. *The pre-filled tier is not an `input` namespace* (read-only, V14): the run's evictions unlink its
   files and its stores add files beside them, so it is brief item 59's state, made before the run
   and modified by it. Item 59's reset already recreates what the run removed. Agreed.
2. *The eviction model mixed two LMCache adapters*: the calls modelled (flat directory, `rename`, 64
   unlinks at once) are the filesystem adapter's, which declares no capacity and is skipped by the
   capacity-triggered controller (its deletes come by a per-user quota path); capacity eviction is
   `fs_native`'s. Agreed: read which adapter the multi-process connector uses for a local and a
   shared path, and model that one whole, its trigger included.
3. *The mechanisms predate the connector choice*: both committed abstracts are captures under the
   in-process connector. Agreed: the two mechanisms are redefined as the multi-process connector's
   adapters on a local and on a shared path, confirmed by item 74's capture.
4. *Capacity in a domain.* No host learns which engine holds the oldest chunk: every host of a domain
   replays the domain's plan, so each computes the same LRU order and the same victims at the same
   planned instants. An unlink is an op of the plan like any write, issued at its planned time on its
   lane; the capacity is the application's WG-fixed setting (what LMCache is told), never asked of the
   storage, which needs only the free space the runner's limit checks confirm before the gate. An
   unlink that must follow another engine's read of the chunk is a validity edge. Settled in
   principle (the user, 2026-11-10); which lane issues an unlink (the engine that wrote the chunk, or
   one eviction lane a domain) follows LMCache: a capacity a server evicting what it wrote, or one
   controller over the shared usage, read with issue 2.
5. *The ramp's results came from past the knee.* Terms, the user's question: the plateau is the
   arrival curve's flat part, sessions no longer growing, not a flat spot in storage performance;
   saturation in the simulations was the emulated GPUs' (every slot busy, a queue forming, the
   knee), never storage's, whose slowness shows only as lag and D since the op set is fixed. With the
   session plateau below the knee, the steady state may never fill the slots, and the heaviest window
   may be clumped returns during the ramp. Agreed: the ramp rerun at the chosen point with clustered
   returns (k 1, 3, 8, queued in item 6) before item 74.
6. *Staging under the multi-process connector is unread*: its size, its drop policy, and where it
   lives. Agreed: read with issue 2.
7. *Reset and datagen time.* The user's refinement of §3.94's reset: a path of the computed write set
   that exists at the right size is left alone (one `stat`, never a walk), on the condition that the
   run never wrote into it while it existed, since a size check cannot see an in-place overwrite of
   the same length (a vector index); the computed write set says which paths were only created or
   unlinked. Regenerating what was written stays the rule, since datagen's bytes carry the controlled
   dedupe and compression ratios. LMCache's chunks never change in place, so the KV reset is the diff:
   unlink the chunks the run made, recreate the pre-filled ones it evicted. The diff is large: the tier
   holds about an hour of writes and a 90-minute run writes more, so most pre-filled chunks are evicted
   by its end. Agreed: the datagen time estimated at Rust datagen's measured speed before item 72 fixes
   the pre-fill.
8. *The frame refined*: the slot queue and the service model are neither cache nor layout but the
   application's compute pacing, as training's step time is, and sit in demand. Agreed.

*Chunk content* (the user's question; agreed). The dedupe and compression ratios meant are the
system under test's, done transparently to the application. A chunk's bytes are a function of its
key, so the same key gives the same bytes and the duplication a real deployment has (a system
prompt's chunks on every engine's disk) appears without a ratio invented for it. The compression
ratio is a case parameter measured from real chunk files in the GPU session (model, KV dtype, and
LMCache's encoding decide it); until then the default is close to incompressible.

*A per-run nonce in written content* (the user's question; agreed). Namespace content is a function
of (namespace seed, path, offset), all known from the published case, so anyone can compute every
byte a run will write and stash a copy outside the benchmark's directories for the system's dedupe to
find; a system that keeps dedupe fingerprints of deleted data gains the same from a prior run. Every
run draws a nonce from the operating system's randomness, never from anything published; rank 0
draws it and sends it with `t0`, so that the same chunk key gives the same bytes on every host and
duplication within the run stays real. It is mixed into the seed of every written namespace's
content, never a dataset's: content written before the run (datagen's, a pre-filled tier) is
unscored and must come back exactly at reset, so it keeps the dataset seed alone. The nonce is
recorded in item 59's dirty marker while the state is dirty (reset unlinks the run's files and
regenerates what it wrote or removed with datagen's content, so after reset no file carries it), in
the namespace manifest of an output namespace, and in the run report. Ops, paths, sizes, the
fingerprint, and CLOSED are unchanged; the compression ratio too. Cost: a pre-filled chunk evicted and
stored again would really match its earlier copy and here does not, a small understatement of dedupe.
A change to the runner's payload definition for namespaces, a recorded decision: brief item 77,
general, not KV's.

**LMCache's multi-process storage read, and the pre-fill's datagen time (item 72's first steps,
2026-11-10, at the user's word; nothing decided).** LMCache 0.5.5 as installed (`venvllm`), read
without running it; paths under `lmcache/v1/`.
*Adapters.* Each L2 tier is one `--l2-adapter` JSON naming its `type`; none is a default (the list is
empty by default). For a directory: `fs` (Python, aiofiles; `distributed/l2_adapters/fs_l2_adapter.py`)
and `fs_native` (a C++ extension, the same naming, which must match, and the same temporary file and
rename, read from its imported symbols and strings, the C++ source not installed); `raw_block` takes
one device or pre-sized file, `O_DIRECT` by default; `nixl_store` (POSIX or GDS) preallocates
`num_files` files and writes fixed pages at offsets in them. `"shared": true` labels events and
capacity reports and changes no I/O.
*The calls of `fs`.* One flat directory, one `.data` file a chunk and object group, raw tensor bytes,
no header, named `<model>@0x<kv_rank>@<group>@<chunk hash>[@salt].data`. A store: `stat` of the final
name, skipped if it exists (written by this or another server); open `<name>.tmp` (or under
`relative_tmp_dir`), one `write` of the whole buffer (`O_DIRECT` optional, buffered when the size is
not a multiple of `f_bsize`), `rename` onto the final name; no `fsync` anywhere. A lookup: one `stat`
a key, deliberately, for shared filesystems. A read: open, `readinto` until full (an optional
read-ahead first read). A delete: `stat` for the size, then `unlink`, 64 at once. `fs_native` makes the
same calls in batches on `num_workers` threads (4 by default), with `open`, `read`, `write`,
`rename`, `remove`, `status`, and no `pread`, `pwrite`, `fsync`, or `io_uring`. Against the in-process
local-disk backend the committed abstracts were captured from (`<key>.pt`, written in place, lookups
from an in-memory dict, evicted inline): a `stat` before every store and every lookup, a temporary
file and `rename`, and one file a chunk *and object group*, so the metadata load per request is larger.
*Eviction (issue 2).* A controller exists only if the adapter's JSON has `"eviction"`: `LRU` or `noop`
need a declared capacity, so `fs`, which declares none, is never evicted by usage, and without
eviction its files are never deleted; `fs_native` is evicted when `max_capacity_gb > 0`. Every second,
at usage ≥ the watermark (0.8), 20 % of the tracked keys are deleted. `IsolatedLRU` evicts by cache
salt against quotas set over HTTP (a salt without a quota losing all its keys each cycle), and an
optional fleet coordinator sends deletes over HTTP (watermark 1.0, every 5 s). So the combination
modelled since 2026-11-09 (the `fs` calls under capacity eviction) does not exist; the configuration
that has both is `fs_native` with `max_capacity_gb`, the same calls but deletes on 4 worker threads,
not 64 at once.
*Several servers on one directory (issue 4).* Accounting is per process: a server's bytes and LRU
hold only the keys it stored, an access reordering only keys already its own; a store finding the
file written by another server is skipped and not counted. No lock, no shared index, no listing at
start; a server finds others' chunks by `stat` at lookup; deletes are not coordinated ("a delete
racing a load turns that load into a miss"). So capacity is per engine even on a shared directory,
and the chunk's writer unlinks it (the first option of issue 4), by an LRU that only the writer's own
accesses order: a chunk another engine reads often can be evicted by its writer. Global accounting
exists only through the optional coordinator.
*Staging (issue 6).* Stores are staged in L1, pinned memory in the LMCache server, `--l1-size-gb`
required, with its own eviction (watermark 0.8, 20 %). Full, a store is dropped, never blocked
(`reserve_write` returns the keys it could place; no option like `force_store_wait`), which matches
item 13's rule (4), a would-be drop a gate failure. One store-loop thread copies L1 to L2, every key
to every adapter, no cap on tasks in flight.
*What follows, to decide:* ~~the mechanism is `fs_native` with a capacity (or `fs` without eviction, a
tier that only grows);~~ (decided by the user, 2026-11-10: `fs_native` with a capacity); local and shared paths make the same calls, differing in whether other
engines' chunks are found and skipped, so whether they stay two mechanisms or become one with the
sharing a parameter; the stats per lookup and per store, and the object groups a chunk, added to
the abstract; the capture of item 74 to confirm `fs_native`'s calls, which were inferred.
*Datagen time.* Rust datagen generates about 22 GB/s at 16 writers (§3.70's measurements, to tmpfs), so the
system's write bandwidth sets the pre-fill's time, not the generator. Item 7's tier of an hour was
stated as 100,000 files an engine; at 11 to 16 chunk writes a second an engine (the simulations, 16
slots) it is 40,000 to 58,000 files of 3 MiB, 116 to 170 GiB. At 10 engines, 1.2 to 1.7 TB: two to
three minutes at 10 GB/s, 20 to 28 at 1 GB/s; 400,000 to 580,000 creates in one directory, seconds to
a minute at the 0.5 to 0.7 million small files a second of §3.33's estimate. The reset of a full run (90 minutes, more than an
hour of writes) recreates most of the pre-fill, LRU taking the oldest first, about the pre-fill's own
time, plus the unlinks of what the run made; after the tuning form (30 minutes) about half. The
pre-filled chunks a run's arrivals resume are read, so they are not the ones evicted.

**The structure proposed for the KV-cache workloads, consolidated (draft for the user's review,
2026-11-09; nothing decided, nothing built).** The draft above, revised by the notes since it: the
arrival orders, the drawn and measured think times, the user's challenge that a provider fixes
utilization rather than users, the primary sources, the closed loop, and the morning ramp. Each item
is a proposal the user decides; numbers are the scratch tests' (AgentX corpus, LMCache local disk,
the pool model), WG process is §8's, and what the earlier draft held that no longer stands is listed
at the end.

*1. The reference is the pool model* (unchanged). `agentx.simulate` with `lmcache_accesses` on the
corpus's requests and blocks, within 2.5 % of the traced reads on the one uncut capture; a GPU capture
checks the calls and the model at a point. The committed AgentX target (2,321 reads at 6 GiB) is the
131k cut's artefact, so the case and its tolerances, and the §3.64 note of 2026-11-06 (`6fc276c`),
are retargeted against the model once the rest is decided. (Accepted by the user, 2026-11-09, with
the GPU checks named: the model already runs on the whole corpus, so what more GPU data would buy is
confidence in the model, first with requests in flight together, several slots at a pool near the
knee, the behaviour every busy-hour result rests on and the one the 2.5 % match (serial) did not test
(the engine timeline came 46 % under); then contexts past 485k tokens, where most reads are.)

*2. One abstract per storage mechanism* (unchanged). `lmcache-local-disk` and `lmcache-shared-fs`,
the attention family a parameter (brief item 70); open, a TP 2 capture. (Accepted by the user,
2026-11-09, the TP 2 capture its condition: a call structure across ranks that values cannot express
reopens it; the capture can share a GPU session with item 1's checks. The user's note: a person runs
a use case (`--case`, `aeiou cases`), which brings in its abstract by the lock's path, so whether one
abstract or several sit under the cases is the authors' choice and is hidden from whoever runs them;
the choice here is made for maintenance and comparability, not for the user interface.)

*3. The engine is the WG's.* A case fixes: the model and its KV bytes a token; the accelerator and
its service model (prefill and decode time a token); the engine as one tensor × pipeline parallel
replica (GPUs multiply, DP × TP × PP; data-parallel replicas of a dense model share no pool and are
separate engines; wide expert parallelism couples them); the requests in flight an engine (slots);
the pool a slot (the accelerator's memory past the weights); the CPU tier (0). The submitter chooses
none. Cases multiply by accelerator, as training's do. The pool a slot sets the busy-hour load
steeply (at four users a slot, 113 and 10 reads a request at 3M and 6M on 8 slots), which is why the
WG must fix it rather than leave it to a deployment. (Accepted by the user, 2026-11-09, with the slots
derived from the pool and the mean live context, as vLLM bounds them, not chosen beside the pool, so
pool and slots are one setting; the service model's numbers measured once on the accelerator and
published with the case, as training's step time is; a wide expert-parallel engine declared by its
case when one is proposed.)

*4. The operating point is the busy hour, fixed by a queue-wait target.* The engine is modelled when
it is full, since storage sized for the average fails at the peak and a load from storage is worth
most when the GPUs have no spare capacity (the user's framing). Operators size for peak under a
latency target and shed load past it (Mooncake, Splitwise, ServeGen, DistServe; no source gives a
frontier lab's utilization). Users an engine follow from the think time (Little's law) and are not a
knob: the WG sets a queue-wait target (a few seconds at p99, say), and the users a slot that meet it
at the WG's service model become a constant of the case (about four a slot in the corpus: 86 % to
95 % busy, queue about 3 s). Not a count of users chosen freely: near the knee the reads are as steep
in users a slot as they were in pool (1.9 against 6 to 9 GB/s at three against four a slot on 64
slots). Replaces the ρ target of the earlier item 4. (Accepted by the user, 2026-11-09: the target's
value is the WG's, made knowingly since it moves the load; the queue wait is computed at the WG's
service model, so slow storage shows only in the measured latencies, never in the users; the
load-shedding variant of the closed loop stays a pending check of the knee.) (Bounded by the user,
2026-11-10, after the load-shed test: the target puts the operating point below the knee, where a
provider's queue policy does not move the reads; the note "The closed loop with load shed".)

*The connector the workload models (decided by the user, 2026-11-09, from a reading of the source
the same day).* vLLM's connector contract lets a load run asynchronously
(`get_num_new_matched_tokens` returns `load_async`; the request waits in `WAITING_FOR_REMOTE_KVS`
while the scheduler serves others). `LMCacheConnectorV1`, the in-process connector every capture so
far used (`kit/serve.sh`), returns `False` and loads in `start_load_kv` before the step's forward, so
a slow load holds the whole batch (its layerwise mode overlaps a request's load with its own layers
only). LMCache's multi-process connector (`LMCacheMPConnector`, `lmcache_mp_connector.py`) returns
`need_to_load > 0` as `load_async`: the load runs in the LMCache server process while vLLM computes
other users' work. The user's choice: the benchmark models the overlapped connector, which has time to
mature before the benchmark is released. Consequences: item 8's budget becomes the user-facing
latency, not the recompute time (with overlap a load slower than recompute still pays, the GPU
serving others meanwhile; its costs are the waiting user's first token and the pool blocks held while
it waits); and the abstract must be captured again under the multi-process connector, whose I/O a
separate server process issues, before it can claim to represent it, in the GPU session with items
1 and 2's checks.

*5. Think time, measured from the corpus.* A session's idle time before a request, its start less
the latest end of every earlier request of the session by the corpus's own `api_time` (p50 2.1 s,
p90 37 s, 31 % zero where sub-agents overlap; 77 % of returns within 10 s, 99.3 % within an hour).
Human turns elsewhere (Qwen-Bailian, parent-linked): p50 about 100 s, mean 4 to 6 minutes, CV 1.6
to 3.9, so not exponential. If the sessions stay a table (item 10), each row carries its own idle
time; if they become distributions, a fitted heavy-tailed distribution a request kind (pooled gaps
halved the reads; drawn a kind they came within 20 %). Time of day appears in no source, so the
busy hour's think time is the all-day one with the busy hour's users, an assumption stated.
(Accepted by the user, 2026-11-09, revised: think time is a context's, not a session's. The user's
question, whether sub-agents spawn sub-agents and whether we model the parallel contexts a user
holds, measured (`think/width.py`, `think/lane.py`): the corpus has no tree (1,697 sub-agents, all one
level under the main agent), but sessions are wide: contexts live at once p50 1, p90 5, p99 11;
requests in flight at once p50 2, p90 6, p99 16, max 23, 95 % of sessions above one; sub-agents a
session p90 8, p99 75, max 153. Within one context the think time is the main agent's p50 5.1 s, p90
140 s, 72 % within 10 s, and a sub-agent's p50 1.6 s, p90 6.3 s, 93 % within 10 s: drawn per context,
the main agent's and the sub-agents' apart. The corpus's own `think_time` field agrees with the
session idle measured (p50 2.3 s, p90 37.5 s against 2.1 and 37) and may be used directly. The
committed abstract (side lanes run in turn, the main lane suspended, `6fc276c`) and the scratch closed
loop (a user's requests one at a time) both serialize what runs in parallel, so the knee of about four
users a slot is an upper bound and the busy hour's reads likely understated; the test of the tree,
2026-11-09, below, measures by how much.)

*6. The run's shape: a ramp into a plateau.* Users arrive on an S-curve (the user's morning ramp),
each resuming a session and reading its stored context, then the steady mix; the schedule computed
before the run from positional draws, so the op set is fixed and slow storage shows only as latency.
Simulated, the ramp's start is write-heavy (20 % to 45 %), the plateau 3 % to 15 % writes, and the
heaviest reads come at saturation, not at the arrival peak (four to fourteen times its reads), so a
short ramp and a longer plateau. The WG chooses, without data: the curve (ServeGen's diurnal rates can
inform it), the share of arrivals that resume rather than start, the run's length (real time; a
shorter form for tuning, the full one for submission). (Accepted by the user, 2026-11-09, revised.
Defaults proposed: a ramp of 30 minutes from 5 % to 95 % of arrivals, a plateau of 60 minutes at item
4's operating point, a tuning form a third as long from the same generator, every 5-minute window
scored, the ramp's included. The user's revision: arrivals are people, not sessions. A person back
from lunch checks every agent that stopped while they were away and answers what each asked, so a
return revives k sessions (the user's estimate 3 to 8, the default; no trace measures it), spread
over the minutes it takes to read and answer each (1 to 5, a default), each revived session's first
request reading its whole stored context and its loops then running concurrently as item 10's tree
generates them. With the plateau fixed in sessions the total of resumes is unchanged; what the
grouping adds is clumping, higher peaks in short windows and a brief surge of a person's requests in
flight, which may make the arrival peak, not saturation, the worst window; a ramp test with clustered
arrivals, k 1, 3, and 8, is queued behind the test of the tree to measure it.)

*7. Retention, fixed by the WG.* Two parts. The contexts the arrivals resume are written before the
run, a dataset (~~an `input` namespace~~ brief item 59's state, made before the run and modified by
it, the joint review's issue 1 of 2026-11-10; about 2.4 GB a 200k-token context at 12 KiB a token, 150 GB
an engine of 64 users, so datagen time is part of the tuning cycle); their first reads are 6 % to 18 %
of a run's reads. Within the run, a tier of 1 hour serves 94 % to 100 % of the busy hour's reads (5
minutes, 68 % to 96 %), against 1.5 to 2.5 times the capacity for 24 hours that adds 1 % to 4 %. The
tier is the workload's, not the submitter's (a submitter's choice would change the op set); published
product tiers (Anthropic 5 minutes or 1 hour; OpenAI 5 to 10 minutes in memory, up to 24 hours
extended) are the reference points. An unlimited tier, if wanted, is a second workload. (Added
2026-11-09, the user's point: a person holds several sessions at once, a window each, attending to
one at a time, so each window's human gaps lengthen while its agent loops run; the windows add idle
sessions, raising the capacity held and the long-gap returns, more than the load in flight.)
(Added 2026-11-09, from item 6's revision: a lunch of 60 to 90 minutes sits at the edge of a 1-hour
tier, and a person's return revives several sessions at once, so a tier that expires before the
return makes every one of them a recompute together, the costliest miss. Inside the benchmark the
pre-written dataset covers the away time; the report says that a deployment's own tier must cover its
surges' away times (lunch, overnight), so that no one sizes one from the in-run tier alone.)
(Accepted by the user, 2026-11-09, revised after a reading of LMCache's source the same day, `dev`,
the multi-process architecture item 4 chose. LMCache has no time-based expiry (the one "ttl" in its
tree is a lock's): eviction is by capacity. An L2 adapter declares a capacity; a background thread
checks usage every second and, above a watermark (default 0.8), its policy (LRU, ARC, or LRU per
cache salt) picks keys making 20 % of the used capacity (`eviction_ratio`) and the adapter deletes
them (`docs/design/v1/distributed/l2_adapters/l2_eviction.md`, `storage_controllers/
eviction_controller.py`). The filesystem adapter (`l2_adapters/fs_l2_adapter.py`) keeps one file a
chunk, named from its key, flat in one base directory; stores through a temporary file and `rename`;
deletes by `unlink`, 64 at once (`_DELETE_CONCURRENCY`). It declares no capacity itself, so the
capacity-triggered controller skips it (its deletes come by the per-user quota path); `fs_native`
takes `max_capacity_gb` and is evicted by capacity. Deleting a user's data is the application's, never
the storage system's (the user's principle), so the workload emulates LMCache's policy: (1) the
retention tier is a WG-fixed L2 capacity an engine, sized for the retention wanted (about an hour of
the busy hour's writes), not a time; (2) eviction as LMCache's, LRU, watermark 0.8, 20 % of the used
capacity a trigger, checked every second; (3) the deletes are the application's unlinks, in the
first version (reversing the deferral first proposed), the keys and their place in the schedule
computed before the run from its access order, as the reads are, so the op set stays fixed; (4) the
pre-written dataset fills the tier to its steady level, just under the watermark, so evictions run
during the plateau (the dataset's size then follows the capacity, and with it datagen time). Scale
at 16 slots: writes about 11 chunk files a second an engine, a tier of an hour about ~~100,000~~ 40,000
(corrected 2026-11-10: 11 a second is 40,000 an hour, 58,000 at the 16 a second the simulations reach)
files
of 3 MiB, so a trigger unlinks about 20,000 files, 64 at a time, every half hour or so an engine,
interleaved across engines with the steady creates, renames, and reads. The user's assessment: 64
concurrent deletes in one very large directory is near the worst case for a POSIX filesystem, and
the renames load the same directory's locks, which is the metadata stress the benchmark should
carry. The rule that truncate and punch-hole are a deferred space-reclaim workload is unaffected;
these unlinks are the application's own. To check before the abstract claims it: a capture under the
multi-process connector's filesystem adapter (item 4's), its calls (temporary file, `rename`, flat
directory, 64 unlinks at once) against what the abstract emits.)

*8. Pass and report.* A run passes when the KV-load p99 is within budget in every window, the worst
window deciding; the budget proposed is the time the engine would take to recompute the same prefix
at its service model (a load slower than recompute saves nothing), or a WG number. Reported beside
the result: the load curve (engines against p99 load latency), read and write bandwidth over the
run, capacity held, and per retention tier the reads served and the prefill compute they avoided, so
a buyer can price the trade; no economics in the benchmark (MLPerf publishes no prices), and power
left out. (Accepted by the user, 2026-11-09, revised by item 4's choice of the overlapped connector,
under which a load slower than recompute still pays (the user's example: 111 ms to load against 110
to recompute, the GPU serving others meanwhile), so recompute time is not the failure line. The load
latency is measured from the moment the schedule makes a request ready to the moment its context is
in the pool, so queueing in the storage client counts. The budget is the case's time-to-first-token
target, set by the WG, less the prefill of the tokens not loaded at the case's service model
(references: Splitwise's 2, 3, and 6 times the unloaded TTFT at p50, p90, p99; Mooncake's 30 s
ceiling on real traffic; a 200k-token context recomputes in about 8 s at the scratch's model). Judged
at p99 in every 5-minute window, the ramp's included; p50 reported beside it. Reported as well:
creates, renames, and unlinks a second and the eviction bursts' size and timing (item 7), and the
margin of the p99 under the recompute time, as information. How the runner emulates the engine
without a GPU (compute a request, slots, loads overlapped) is item 13's.) (Revised by item 13's
debate, the user's agreement of 2026-11-09: what the budget judges is D, a request's own storage wait,
its load time and any edge wait, the time a user experiences between submitting a query and the
response, and no longer the time from the scheduled-ready moment, under which a lane once behind would
charge every later request its inherited lag. p99 of D per 5-minute window stands, windows assigned by
planned time so that each window's request count, and so the violations it can tolerate, are known
before the run; a run stops at the violation that makes failure certain. A single-response ceiling is
added (30 s, Mooncake's, as the example; the WG's to set), since one maximum over tens of thousands of
requests is decided by a hiccup and does not repeat, which is why a single response over the budget is
not itself a failure. Reported beside it: each main-line session's accumulated lag L, as the sign of
a system nearing saturation, and the sum of D over all requests, as `top` reports 1600 % on a saturated
16-core system: the total delay the system under test caused, so three sub-agents' delays all counting
is not triple counting.)

*9. The submitter's knob: engines, one axis.* The score is the engines (or, if engines are too coarse
for small systems, the slots, the last engine partly filled) served at a pass, users served being
engines times the case's users an engine. Storage's speed never changes the users an engine (that
would make the op set depend on the system under test); faster storage shows as margin on the load
curve. (Revised 2026-11-09, the user's point that a user fans out to sessions: the unit counted is the
concurrent session, not the user. No trace measures sessions a user (the corpus's records are
sessions, with no user identity), and the number will drift as agents spread, so a score in users
would hide an unmeasured multiplier; users may be reported beside it as sessions divided by a stated
sessions-a-user assumption.) (Accepted by the user, 2026-11-09, with whole engines the headline: an
operator builds out identical replicas (a data-parallel replica is one configuration copied; racks
and nodes come in fixed units, often several engines each), so 5.3 engines guides a build-out less
than 5, and the run's precision (reads spread about ±10 % between seeds, a p99 over 5-minute windows
noisy) does not carry a decimal. The headline is the largest whole number of engines that passes,
as concurrent sessions served; the count by slots (the last engine partly filled) is reported beside
it, labelled supplementary and never ranked on, and is the useful resolution only for systems of one
or two engines. Every draw keyed by engine id, so adding an engine leaves the others' ops unchanged;
engines ramp together by default (one time zone), staggering a WG what-if; the submitter finds the
count by raising engines until a run fails (by hand or the WG's driver), one run at the count
submitted, the load curve optional.) (Amended by the user's decisions of 2026-11-10, the note "The
generator in the abstract, a shared tier, and events between runners": with a storage tier shared by
engines, adding an engine changes the others' hits; the property holds by domain, every draw keyed by
the seed, and sessions by (seed, session), placed on engines by the domain's routing rule.)

*10. Sessions: table, archetypes, or distributions* (~~open~~ settled 2026-11-10, below). Eight archetypes held within 3 % under
rounds and random arrival but overstate by 20 % to 68 % under timed arrival, their draw rich in
main-lane requests; picked again to match the kind mix and checked under the measured think time and
the ramp, or replaced by distributions (untested). Whether a table of rows is compact enough for the
rule that an abstract is never a trace listing is the user's call. (Requirement added 2026-11-09,
the user's: whatever form the sessions take, a session is generated as a tree of concurrent
contexts, the main agent and its sub-agents running in parallel, with fan-out (sub-agents a context
spawns), concurrency (how many run at once), and each context's turns and think time fitted at depth
1, where the corpus measures them. The user's example of depth: an orchestrator spawning an agent a
project, which may spawn its own; a child that starts a fresh context is the same thing to storage
whether called a sub-agent or a new session, so one generator covers both. Depth and the
orchestrator's fan-out are WG what-if parameters, 1 by default, documented as supported by no
trace.) (The user chose, 2026-11-10, the fitted generator `gen5` over the whole table as a data file,
on condition of a held-out test, which ran the same day: the note "The fifth generator held out".
~~Not yet accepted: what that result means for the read claim, and, by item 12's rule, the steady
writes outside ±7 %, are back with the user.~~ Accepted by the user, 2026-11-10, with a narrow claim
for reads: the generator reproduces the pool model on this corpus and says nothing of reads for
other sessions, its reads not generalizing between halves of the corpus; writes as item 12 now has
them.)

*11. The stream's seed* (unchanged). Fixed by the WG as the dataset seed is; many engines narrow the
spread for anyone who varies it. (Accepted by the user, 2026-11-09: every draw keyed by the seed and
the engine id, item 9's keying; a CLOSED run needs the same op set for every system under test, so a
submitter never chooses it, and a researcher who varies it reports the spread between seeds beside
the result.)

*12. Tolerances.* Writes tight (1 % to 7 % in every test); reads judged at the WG's operating point
against the pool model, with a tolerance from the spread between seeds there. (Accepted by the user,
2026-11-09, with three things fixed: (1) what is compared is the abstract's dry-run counts against
the pool model at the operating point on the whole session table, the lab replay (233 requests, its
own resampling 20.6k to 48.0k reads) remaining a consistency check at its own point with the band of
that range; (2) the read band is measured, about two standard deviations of the pool model's reads
over at least 8 seeds at the WG's operating point, and never carried over from a neighbouring point,
reads being steep in load near the knee (the 5 % to 12 % seen at 32 and 64 users came from another
arrival order); (3) the tolerance file says that the band claims the abstract reproduces the pool
model and nothing about how near the pool model is to a real service, the corpus's sampling limit
~~of about ±20 %~~ being separate. ~~Writes stay ±7 %, pending item 10's result; if the chosen form cannot
hold it, that comes back to the user rather than the band widening.~~ Revised and accepted by the
user, 2026-11-10, on the held-out result (the note "The fifth generator held out"): the corpus's
sampling limit is the measured difference between halves of the sessions, 1.3 to 1.9 times in reads
near the knee, at 196 sessions a half; the write bar of ±7 % is held on the ramp, whose writes
generalize (−1 ± 4 and +1 ± 7 held out), and a steady-state run is for exploration, not scored on
writes.)

*13. The runner.* The stack overflow on a deep `x @ (r − dd)` (a brief item); and two new needs: an
abstract whose requests carry a schedule (arrival times and think times, computed before the run,
issued at them; the scratch abstract's `think` is const 0 today), and the resumed contexts as a
dataset the abstract reads (a namespace written by a datagen run). Both are contract work for brief
item 70's successor. (Added 2026-11-09: lanes that spawn concurrent lanes, nested, for item 10's
tree of contexts; today's side lanes run in turn, one level deep.) (Accepted by the user, 2026-11-09,
after a debate that replaced the proposal first made here, a contract construct of lanes served by C
servers. The user's design: a lane is a context (the main line or one sub-agent), a session a group of
lanes, and a global timeline per engine on which each op launches at its time. The rules:
(1) The op set is absolutely fixed: every pool hit and miss computed before the run from the plan's
global order, absolute times, and pool size, a pass over the merged timeline, one ~~engine~~ domain
at a time since ~~engines~~ domains share no pool (amended 2026-11-10, the note "The generator in the
abstract, a shared tier, and events between runners"; a domain of one engine is the case first built). The status of another op never changes this op; "the pool's contents
would have shifted" is timing changing the ops, excluded by definition. The principle, the user's:
track the slowness, neither control it nor respond to it.
(2) A new main line starts on absolute time, without lag: the user arrives when they arrive. Every
other op launches at its planned time plus the lag it inherits, `launch = plan + L`, the user's form:
pure absolute times would compress the gaps to zero (think time gone, a burst of catch-up load), and
this way a gap is never shorter than planned and nothing runs early. L grows by each load's time
(the plan's allowance for a load is zero, so no parameter) and never shrinks. An oversaturated system
accumulates lag, as it should; offered load falls a little as storage slows, small while a system
passes (a session requests about once a minute), so the gate judges D and never op latency alone.
(3) Edges of two kinds, all pointing forward in planned time, so no deadlock. A causal edge (the next
turn after a response, a sub-agent's start after its parent's request, a main line waiting for its
sub-agents) carries the planned gap and the lag, the maximum over the ops that cause this one. A
validity edge only forces order, and exists only if reversing the two ops would change an op's result:
write then read, read then unlink, create after unlink of a name, never read against read. That rule
is the middle between ignoring the other ops and a global order that would benchmark our own
synchronization. An edge names its op by lane and index, with a completed-ops counter per lane, not a
table of ops in flight (a lagging lane may not have issued the op yet), so no per-file state. Edges
never fire for a system that keeps up. When one does, the wait is real delay the system under test
caused and goes into the waiter's D, and the count and total of edge waits are reported; a forced
wait stands in for what the application would have done (miss and recompute), so it is a bias, not
part of the error bars, which stay the spread between seeds.
(4) Stores (LMCache 0.5.5's local-disk path, read 2026-11-09; the multi-process connector's L2 store
path not traced, item 4's capture to confirm): the forward step waits only for the copy to host staging
memory; the write is queued (4 threads by default), the chunk pinned in staging until it finishes and
visible to a lookup only then, so a turn arriving early misses and loses every later chunk of its
chained keys. Staging full, a store is dropped by default (`force_store_wait` off), silently, and a
load, drawing on the same pool, spins until memory frees. Stores run on their own lanes, a fixed
worker count draining a queue, with no causal edge to the response. A write is too slow if it has not
finished when its first planned reader launches (the validity edge, the wait in the reader's D) or
if the bytes issued and unfinished, loads' included, exceed the WG-fixed staging size, a would-be drop.
The second is a gate failure, the user's: the plan assumes every store lands, so a drop ends its
validity, and the true cost (recomputing the block and its children) could be charged only by letting
timing change the ops. The pool integrates over minutes (12 GiB about 6 minutes at 33 MiB/s), so a
hiccup cannot trip it; the staging size is then a sensitive WG parameter (2 GiB in the kit, 12 in the
scratch, neither chosen), the case states the default store policy, and allocator fragmentation, which
would fill it sooner, is a known optimism. Write latency, backlog peak, and would-be drops are reported.
(5) Unlinks are their own lanes, no user waiting on them, their lag reported and not charged to users;
each eviction batch has a window ending at the next planned trigger, inside it slow but fast enough
and its duration reported, past it a failure. A create waiting for an unlink goes into D.
(6) Gate failures leave the run valid but not passed, like any constraint after a run. A tuning run goes
to completion, under a lag ceiling, and reports when and how much; a submittable run stops at certain
failure, under an option named for the mechanism. An op issued after its time with nothing
outstanding is the benchmark host's lateness, not the system's: detected and reported as a validity
check. GPU idle time, unlike D, needs the slot model and is reported as modelled, computed after the
run from the measured loads; it is first-order, holding the op stream fixed. The GPU slot queue stays
in the plan, in the same pass over the merged timeline as the pool (the slot-limit test's result,
accepted by the user 2026-11-09): dropping it changes nothing at 64 sessions, but reads by −26 % to
−57 % once queues form, and the ramp's worst windows come after saturation.)

*No longer held from the 2026-11-08 draft:* ρ as the operating point and a target near 0.25 (the
flat region was the rounds' artefact, and the busy hour fixes the point by queue wait); users per
engine as a free WG count (they follow from the think time and the target); think time capped at a
provider's cache lifetime (it is measured, and retention is its own item); eight archetypes as
settled (item 10); mid-ramp as storage's hardest moment (the debate's expectation; saturation is).

*Not settled:* time of day in any source; one corpus and one publisher's agents; the closed loop
lets queues grow where a provider sheds load~~, so its plateau overstates~~ (tested 2026-11-10, the
note "The closed loop with load shed": the served plateau moves 3 % at most, but past the knee the
reads depend on the queue policy, which ~~the WG fixes or~~ the operating point avoids, the user's
decision of 2026-11-10; why the reads fall under shedding, not diagnosed); ~~whether LMCache's store is
off the latency path (its source, to check)~~ (off it, 0.5.5's local-disk path read, item 13's rule 4); the pool model checked at one point, one model, one
GPU; requests over 485k tokens; compaction of long sessions; a TP 2 capture; the publisher's cache
columns (the user's call); a GPU check of the pool model with several slots in flight at a pool near
the knee, first, and past 485k tokens, second (item 1); a capture under LMCache's multi-process
connector, the overlapped one the workload models (item 4's note); the queue-wait target, the latency budget, the ramp's shape and resumed
share, and the retention tier, all the WG's to set.

*Order proposed:* the user decides 1 to 13; then the tests the decisions need (archetypes matched on
kind mix under the measured think time and the ramp, or distributions; a load-shedding variant of the
closed loop); then brief items: the sessions (10), the schedule and resumed-context dataset in the
contract (13), the runner's stack (13), the retargeted case and tolerances (1, 12), item 70 carrying
2, 3, and 9; then commit and push together. (Extended 2026-11-10 by the note "The generator in the
abstract, a shared tier, and events between runners": the brief items also carry the cache
construct, domains and events, and Dynamo's G3.5; the first work, once the brief items are accepted,
is the port test of `gen5` as an abstract and the scratch measurement of cross-engine reads.)

## 4. Plan changes

- Paper abstracts first, derived from `strace` of real loaders. Added a fourth: checkpoint
  restore, the one workload where many ranks read the same files, so delegations and page cache
  matter. Drafted 2026-09-29 (`ABSTRACTS.md`, §3.15); traces still to be captured.
- Then: fix the determinism items in the docs (done), build the VM with `--dry-run` and the
  fingerprint against ext4 and loopback NFS, then Spike 1 on the real target with the thread-pool
  backend first.
- 2026-09-30: constructs accepted, Option D chosen, AST schema v0.1 drafted (§3.18). ~~Next in
  order: the builder package with the eight abstracts as its first tests, the hermetic build
  harness and build-twice CI, then the VM against the schema.~~ Builder, abstracts, harness,
  and CI done the same day (§3.19). ~~Next: the VM with `--dry-run` and the fingerprint against
  the nine committed ASTs,~~ Done the same day (§3.22, `runner/`). ~~Next: the `sync` backend
  and `aeiou run` against ext4 and loopback NFS,~~ `aeiou run` with `sync` and `sync-direct`,
  `aeiou datagen`, and the manifest check done the same day and run against ext4 (§3.23);
  loopback NFS still to run. Input namespaces, the namespace manifest, and rank rotation
  done the same day (§3.24), so every committed abstract now runs. ~~Next: the loopback NFS
  run, the TCP coordinator (which makes `--ranks` and `--rank-rotate` real),~~ The
  coordinator done the same day (§3.25) and run as two processes on `localhost`; ~~loopback
  NFS still to run.~~ The loopback NFS run done the same day (§3.26): every abstract and the
  two-rank tests on the mount, with the NFS client's RPC counts per backend. ~~Next: the
  format-class reader protocols and the parameter-file split,~~ The parameter-file split done
  the same day (§3.27, `schema/README.md` §8, `aeiou-params`). ~~Next: the format-class reader
  protocols,~~ Four format classes, contract 0.2, `aeiou-datagen`, and three container
  abstracts done the same day (§3.28). ~~Next: the resumable VM and `io_uring`, the per-actor
  sub-actor pool, `mountstats` and `--metrics`;~~ The resumable VM and the `io_uring` backends
  done 2026-10-01 (§3.29), the host counters the same day (§3.30); `--drop-caches` decided
  and the object-backend opinion recorded (§3.31, §3.32, brief §6 items 16–17). ~~Next: the
  `io_uring` knobs,~~ The ring and io-wq knobs done the same day (§3.34). ~~Next:
  `--drop-caches` with the residency check and the mount options in the
  counters,~~ `--drop-caches`, the residency check, and the mount options done the same day
  (§3.31, Built). ~~Next: `libaio`/`posix-aio`/`mmap`,~~ `posix-aio`, `libaio`, and `mmap`
  done the same day (§3.35). ~~Next: the per-actor sub-actor pool,~~ The sub-actor pool done
  the same day (§3.36). ~~Next: `--metrics`,~~ `--metrics` built the same day, its
  definitions decided (§3.39). ~~Next: the `RLIMIT`
  checks,~~ The limit checks built the same day, their choices decided (§3.40). ~~Next: the JSON report,~~ The JSON report built the same day, its choices decided (§3.41). ~~Next: the trace-side metrics tool;~~ `aeiou-trace` built the same day, its choices decided (§3.42). ~~Next: a trace of a real application through it (the capture plan of `ABSTRACTS.md` §11),~~ Rows 1 to 4 of the capture plan traced the same day (§3.43, §3.44, §3.45, §3.47); CLOSED defined as the same operation sequence, the backend declared by the abstract (contract 0.3), and the restore's buffer chain, the same day (§3.48). Next: row 6 (FAISS IVF), then the heavier rows (DiskANN, vLLM + LMCache), `gds`/`nixl-posix`/`libnfs`, the object backends; the
  remaining classes (Arrow IPC, MDS, Megatron) and the tenth abstract when their readers
  can be traced. Row 6 (FAISS IVF) traced the same day (§3.49); ~~rows 5, 7, and 8 remain.~~ rows 5 and 7 (DiskANN search and build) traced 2026-10-02 (§3.50); ~~row 8 (vLLM + LMCache) remains.~~ row 8 (vLLM + LMCache) traced the same day (§3.51), and its shared-store pair (a writer, and the cold reader decided in §3.51) traced and built the same day (§3.52). Every row of the capture plan has a trace; open: ~~a chat replay for the KV distributions,~~ (run 2026-10-02, §3.56, decided) ~~the tolerances,~~ (built and decided 2026-10-02, §3.55) ~~the `trace` node~~ (designed and built 2026-10-02, §3.58; traces under the event-loop backends remain), a GPU engine's touch pattern for `model_load`.

## 5. Things reviewed and left as-is

- Feistel + cycle-walking over the smallest 2^k ≥ N. Correct and O(1). Small test domains
  (N ≈ 1000) give a weak permutation at 4 rounds but still a bijection, which is all tests need.
- The bitmap probe count N·H(N) and the DRAM table are correct.
- Op chaining off by default. A short read cancels the rest of an `IOSQE_IO_LINK` chain, and the
  round trip is µs against 100s of µs per NFS RPC.
- ~~MPI as the multi-host mechanism.~~ Superseded on 2026-09-28 by the TCP coordinator; see
  §3.7.
- Timer wheel for compute sleeps rather than one `IORING_OP_TIMEOUT` per actor.
