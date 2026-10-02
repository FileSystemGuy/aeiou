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
- *Deferred, marked as such in the schema:* the `replay` node's trace format, the container
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
runner's `--params FILE`, and the `aeiou-params` helper (`schema/README.md` §8). Decisions
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
the `replay` node.

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
  `--max-distance` is there for when they exist.
- **The format gains `source`.** `"dry-run"` or `"strace"`; the `total` is the same shape,
  so the format number stays 1. A trace document has no templates and no fingerprint.

Checked: the runner under `strace` (`sync`, one GPU) on `train_small_files`,
`kv_cache_serving`, and `vdb_search_diskann` gives the dry run's request sizes, run
lengths, popularity, first touches, and data op counts exactly; reuse distance differs by
the order (CDF distance 0.07 to 0.20 at these sizes). No trace of a real application has
been taken; that, and the tolerances, are what remains of item 14 besides `replay`.

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

Row 2 of the capture plan: `np.load(path, allow_pickle=True)["x"]`, the call in DLIO's
`npz_reader.py`, through a `DataLoader` on loopback NFS, over archives written as DLIO's
`npz_generator.py` writes them (`np.savez(x=volume, y=labels)`). Kit in
`builder/traces/train_large_samples`, findings in `ABSTRACTS.md` §2. Unlike §3.43, where
the draft was one call short, here the draft was wrong in shape. **The choices below were
made while building. The user confirmed the explicit seeks (decided 2026-10-01); the
others are not yet confirmed.**

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
- **Not modeled.** The `glob` that lists the corpus (two directories). DLIO lists files
  itself and differently; the walk is not part of this abstract.
- **Seen and not explained.** About one GETATTR per READ on the wire. Recorded in
  `ABSTRACTS.md` §2; whether it is the client revalidating during a long buffered read
  without a delegation is a question for a run with `rpcdebug`, not for the abstract.

The lesson for the remaining six rows: the drafts marked **[verify]** are hypotheses. One
was nearly right and one was not.

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
  checks,~~ The limit checks built the same day, their choices decided (§3.40). ~~Next: the JSON report,~~ The JSON report built the same day, its choices decided (§3.41). ~~Next: the trace-side metrics tool;~~ `aeiou-trace` built the same day, its choices decided (§3.42). Next: a trace of a real application through it (the capture plan of `ABSTRACTS.md` §11), `gds`/`nixl-posix`/`libnfs`, the object backends; the
  remaining classes (Arrow IPC, MDS, Megatron) and the tenth abstract when their readers
  can be traced.

## 5. Things reviewed and left as-is

- Feistel + cycle-walking over the smallest 2^k ≥ N. Correct and O(1). Small test domains
  (N ≈ 1000) give a weak permutation at 4 rounds but still a bijection, which is all tests need.
- The bitmap probe count N·H(N) and the DRAM table are correct.
- Op chaining off by default. A short read cancels the rest of an `IOSQE_IO_LINK` chain, and the
  round trip is µs against 100s of µs per NFS RPC.
- ~~MPI as the multi-host mechanism.~~ Superseded on 2026-09-28 by the TCP coordinator; see
  §3.7.
- Timer wheel for compute sleeps rather than one `IORING_OP_TIMEOUT` per actor.
