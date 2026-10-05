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

`builder/aeiou` implements Option D (`builder/README.md`). Every `ABSTRACTS.md`
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

`runner/` holds the Rust crate `aeiou` (`runner/README.md`): the AST loader, the canonical
hash, the validator, the positional VM, and `aeiou dry-run`, which walks every actor
instance of an AST without I/O and prints op counts, bytes, and the fingerprint. All nine
committed ASTs run; `aeiou check` prints byte-identical output to `schema/check.py` over
them and CI diffs the two. Building it forced the choices the schema had left to the runner
(`schema/README.md` §5, "its exact definition belongs to the runner"). Each is listed in
`runner/README.md` §2 and pinned by a golden test; the reasoning behind the ones that were
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

`runner/README.md` §4–§5 state what exists; this is why it is shaped that way.

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
(`runner/aeiou/src/coord.rs`, `runner/README.md` §6) changed four details and settled four
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
`runner/README.md` §7). It was a correctness exercise: the same kernel on both sides, tmpfs
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
  `mkdir` fail with `EEXIST`. Two practices are recorded in `runner/README.md` §7: generate
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
the mount after writing behind the server, as `runner/README.md` §7 prescribes):

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

`runner/README.md` §8 states what exists; this is why it is shaped that way.

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
  `runner/README.md` §4). The price is that one instance's sub-actors share one core; a
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
  faster still but used as much CPU as the threads (`runner/README.md` §8). On the loopback
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
totals (`runner/README.md` §4). Three sources, each optional where the kernel lacks it: a
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
(`runner/README.md` §8). That is the kind of finding the fields exist for: without them the
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

**Built 2026-10-01** (`runner/aeiou/src/cold.rs`, `runner/README.md` §4), as decided, with
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
(`runner/README.md` §8, `RunOpts::uring`, `tests/uring_knobs.rs`).

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

**What the first runs showed (loopback, `runner/README.md` §8 table).** Capping the workers
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
`libaio`, `libaio-direct`, `mmap` (`runner/README.md` §9, `backend.rs`, `aio.rs`,
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

**What the loopback showed** (`runner/README.md` §9 has the tables): `posix-aio` sends the
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

**What the loopback showed after the change** (`runner/README.md` §9): warm, `mmap` with
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
the same day (user, as root; `runner/README.md` §9): at 1024 the cold fault path sent
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

`runner/README.md` §4 states what exists; this is why it is shaped that way. It was the
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

Built as the next item in the recorded order (`runner/README.md` §10, `metrics.rs`,
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
the open-file high-water mark. Built as `limits.rs` (`runner/README.md` §11). The choices
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
`aeiou run --report-json FILE` (`runner/README.md` §12, format `aeiou_report: 1`). The
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
Built as `aeiou-trace` (`builder/aeiou/trace.py`, `builder/README.md` §7): `metrics` turns
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
  would be a second front end to the same `Metrics`; none is written.
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
  (a restart, or a second engine), where they are read by every new conversation.
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
  and the loop indices (`runner/README.md` §2), so two abstracts draw the same values when
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
of verdicts: `builder/README.md` §7.

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
documents (`runner/README.md` §13, `builder/README.md` §7, schema README).

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
| chunks stored | 14.41 | 15.6, 15.4 (the 108 % above) |
| chunks hit | 840 | 784, 850 (loads 713, 773 at 0.91) |
| prompt tokens, mean | 218,922 | — |

The hit count depends on how finely `turns` resolves its tail (`--turn-bins`, 100 by
default): with 50 shares the abstract hits 890 to 920, with 200 it hits 715. The cause is
the context: the abstract draws every turn's growth independently of the session's length,
and the corpus's long sessions grow more slowly per turn (and are compacted), so the
abstract's longest conversations reach the 990k-token cap and are cut. A growth that
depends on the chain's length is not drawn here. The fit at 100 shares is within 10 % on
hits and 9 % on stores; `tests/test_trace.py` repeats it on 10,000 requests.

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
`runner/README.md` §14 describes it; `PROJECT_BRIEF.md` §8 records the `AEIOU_` prefix in
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
(`runner/README.md` §15). The wording is the parsers' own (`the following required
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
