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
  and CI done the same day (§3.19). Next: the VM with `--dry-run` and the fingerprint against
  the nine committed ASTs, then the format-class reader protocols and the parameter-file split.

## 5. Things reviewed and left as-is

- Feistel + cycle-walking over the smallest 2^k ≥ N. Correct and O(1). Small test domains
  (N ≈ 1000) give a weak permutation at 4 rounds but still a bijection, which is all tests need.
- The bitmap probe count N·H(N) and the DRAM table are correct.
- Op chaining off by default. A short read cancels the rest of an `IOSQE_IO_LINK` chain, and the
  round trip is µs against 100s of µs per NFS RPC.
- ~~MPI as the multi-host mechanism.~~ Superseded on 2026-09-28 by the TCP coordinator; see
  §3.7.
- Timer wheel for compute sleeps rather than one `IORING_OP_TIMEOUT` per actor.
