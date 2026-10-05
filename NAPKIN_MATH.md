# Abstract-Driven I/O Benchmark Runner — Napkin Math & Risk Assessment

Status: pre-implementation analysis (2026-09-24, revised same day with user decisions — see §8;
revised 2026-09-25 after design review — see `DESIGN_REVIEW.md`).
Nothing here has been measured yet.
Numbers marked **[verify]** are from general experience/literature, not from this project,
and should be confirmed by the spikes listed at the end before design decisions depend on them.

---

## 1. What we are building (restated)

A Rust program that:

1. Reads an **abstract** (a compact, nested, regex-like description of a workload) from a file.
2. Takes `--seed` and `--gpus` (plus other run parameters) on the CLI.
3. Instantiates one **actor** per simulated GPU (and sub-actors for data-loader workers), each
   running the abstract as a small state machine / interpreter.
4. Issues the resulting POSIX-equivalent operations via io_uring, closed-loop, and measures
   whether the storage system keeps the simulated GPUs busy.

Key principle: **the dependency graph is never materialized.** It is implied by the abstract
(a tiny AST) plus per-actor interpreter state (a few hundred bytes to a few KB). A run with
50M files and 10,000 GPUs has the same abstract as a run with 1,000 files and 1 GPU.

---

## 2. DRAM budget

### 2.1 The "consumer" (sample-without-replacement over N files)

| Approach | 50M files | 100M files | Cost per draw | Epoch reset | Notes |
|---|---|---|---|---|---|
| Bitmap + rejection sampling | 6.25 MB | 12.5 MB | avg ≈ ln N ≈ 18–19 probes, but **last draw ≈ N probes** | memset, trivial | Total probes/epoch ≈ N·H(N): **0.92 B** (50M), **1.9 B** (100M). Aggregate is tolerable; the tail is not (final draws take 10s of ms – 1 s). |
| Bitmap + rank/select index (Fenwick tree over popcounts) | ~7–8 MB | ~14–16 MB | O(log N), ~27 steps | rebuild, fast | Fixes the tail, more code. |
| Pre-shuffled `Vec<u32>` (Fisher–Yates up front), pop from end | **200 MB** | **400 MB** | O(1), 1 read | full reshuffle: a few seconds (random-access bound) **[verify]** | u32 is enough up to 4.29 B files. `Vec<u64>` doubles it. Double-buffer to hide reshuffle → 2×. |
| Incremental Fisher–Yates (`Vec<u32>`, swap one element per draw) | 200 MB | 400 MB | O(1), ~1 cache miss (~100 ns) | re-init to identity (fast, sequential) | Same memory as pre-shuffled, but no epoch-boundary stall. |
| **Keyed pseudo-random permutation (Feistel + cycle-walking)** | **~0** | **~0** | O(1): ~4–6 hash rounds × expected ≤1.34–2 walks ≈ 20–60 ns | change key (`hash(seed, epoch)`) | **Recommended.** Draw *i* of the epoch is `perm_key(i)`. Stateless, trivially shardable, perfectly repeatable. |

For comparison, real PyTorch `DistributedSampler` materializes `torch.randperm(N)` as int64 on
**every rank** — 400 MB per rank at 50M. So even the 200 MB list is "cheaper than reality";
the permutation approach is simply better.

**Why a Feistel permutation and not an affine/LCG permutation (`(a·i + b) mod N`):** affine
maps produce constant strides between consecutive draws. A storage system (or its prefetcher,
or directory-layout locality) could accidentally benefit. A ≥4-round Feistel network with a
decent 64-bit mixing round function has no exploitable structure for this purpose.

Domain handling: Feistel works on a power-of-two domain; use the smallest 2^k ≥ N
(2^26 = 67.1M for 50M → 1.34 walks avg; 2^27 = 134M for 100M → 1.34 walks avg) with an
unbalanced split when k is odd, and cycle-walk (re-encrypt until result < N).

Caveat (applies equally to Fisher–Yates): a 64-bit seed can only reach 2^64 of the N!
possible orderings. This is irrelevant for I/O fidelity.

### 2.2 Everything else the runner holds

| Item | Sizing rule | Example | Memory |
|---|---|---|---|
| Abstract AST | size of the abstract file | — | KB |
| File names | **computed** from pattern + file id (`train/{id/1e6:03}/{id/1e3%1e3:03}/img_{id:09}.jpg`), ~50–100 ns to format | — | 0 |
| File sizes | constant, or `dist.sample(hash(dataset_seed, file_id))`. The dataset seed lives in the dataset definition and the `datagen` manifest, **not** in `--seed`, so a different run seed never changes what the corpus is expected to look like. | — | 0 |
| File sizes from a manifest (only if real, irregular datasets must be replayed) | 4 B × N | 50M / 100M | 200 MB / 400 MB |
| *Materialized path strings (anti-pattern, for reference)* | ~64 B path + 24 B `String` + allocator overhead | 50M / 100M | **~5 GB / ~10 GB** |
| Actor state (GPU + W workers + reorder window + interpreter stack) | ~256 B + 64 B × W + ~16 B × stack depth | W=8 → ~1 KB; 50K GPUs | ~50 MB |
| In-flight op records (slab indexed by `user_data`) | ~64 B × ops in flight | 1M in flight | 64 MB |
| io_uring rings | SQE 64 B, CQE 16 B; 4K SQ + 8K CQ ≈ 400 KB per ring | 64 threads | ~26 MB |
| Read buffers — per-thread **ring of sink buffers, sized larger than L3** (data is discarded, but the copy must miss cache the way a real loader's does; a hot 2 MiB region would hide the memory-bandwidth cost and understate client CPU per op, which is the key A/B metric) | threads × ~64 MiB | 64 × 64 MiB | 4 GB (tunable; 16 MiB/thread is the floor) |
| *Read buffers — one per in-flight op (anti-pattern)* | in-flight × transfer | 344K × 128 KiB | **~44 GB** |
| Write source buffers (must be non-dedupable/non-compressible, see §5) | pool, or regenerate with fast PRNG | 64 threads × 16 MiB | ~1 GB |
| Latency stats: HDR histograms per (thread × op type × step bucket) + 1 s time series | ~30–200 KB each | 64 × 8 × 10 | ~500 MB worst case; use sparse buckets |
| Per-step stall time per GPU (time from end of `compute` to the next `take` returning) | 4 B × steps × GPUs on host | 500 × 2,500 | 5 MB. **Not** per-op logging; this is the most useful single output and shows the cache-fill transition of §8.C directly. |
| *Per-op latency log (anti-pattern)* | ~24 B × ops | 50M files × 4 ops per epoch | **~5 GB / epoch** — stream to disk if ever needed |

**Runner total, recommended design, 100M files, ~50K simulated GPUs, 1M ops in flight:
well under 1 GB of state, plus a tunable 1–4 GB of sink buffers.** DRAM is not the constraint on
the runner. It *is* a constraint on the kernel
of the client host (next section).

### 2.3 Hidden memory: the client kernel

Not our process, but it will shape results:

- **dentry + NFS inode cache:** roughly 1–1.2 KB per file (nfs_inode ≈ 1 KB + dentry ≈ 200 B)
  **[verify]**. Touching all 50M files → **~50–60 GB** of slab on each client host if nothing is
  evicted; 100M → ~100–120 GB. The kernel will reclaim under pressure, but lookup behavior
  (and so the op mix actually sent to the server) changes over the run as the caches fill.
- **Page cache:** any client with RAM comparable to its share of the dataset will serve
  re-reads from cache. Dataset must be ≫ aggregate client RAM, or use O_DIRECT, or drop
  caches between epochs.

---

## 3. IOPS / throughput math

### 3.1 What the workload demands

For a training abstract with batch size B, per-step compute time T (the emulated GPU time),
G GPUs, file size S, and k ops per file:

```
files/s  = G · B / T
bytes/s  = G · B · S / T
ops/s    = G · B · k / T
```

k for "real" PyTorch + Python `open()` on a small file is not 3. A typical trace is
`openat, fstat, ioctl(TCGETS), lseek, read, read(→0 EOF), close` — ~7 syscalls, of which on
NFS maybe 2–3 go over the wire (OPEN, READ, CLOSE; GETATTR/LOOKUP depending on caching).
**Abstracts should be derived from `strace` of real loaders**, not from intuition.

| Scenario | G | B | T | S | files/s | bandwidth | wire ops/s (k≈3) |
|---|---|---|---|---|---|---|---|
| Small files | 100 | 7 | 0.3 s | 128 KiB | 2,333 | 0.3 GB/s | 7K |
| Small files, saturating | 43,000 | 7 | 0.3 s | 128 KiB | 1.0 M | 134 GB/s | 3 M |
| Large samples (1 MiB reads) | 100 | 7 | 0.3 s | 140 MB | 2,333 | 327 GB/s | 327K reads/s |

Epoch duration at 50M files, B=7, T=0.3 s: **60 h** @ 10 GPUs, **6 h** @ 100, **36 min** @ 1,000.
Runs will need to be bounded by step count or time rather than full epochs at low G.

Dataset size sanity check: 50M × 128 KiB ≈ 6.4 TB (reasonable); 50M × 140 MB ≈ 7 PB (not).
**A 50–100M-file corpus implies small files, which implies a metadata-dominated op mix.**
That's where both the storage system and the benchmark client are weakest.

### 3.2 What one client host can issue **[verify all]**

| Path | Rough per-core rate | Notes |
|---|---|---|
| Runner's own state machine (advance actor, build SQE, reap CQE, update histogram) | 3–10 M ops/s | Never the bottleneck if sharded per thread with no shared locks. |
| io_uring, local NVMe, 4K random reads, inline (non-punted) completion | 1–3 M IOPS typical; >10 M with extreme tuning (polling, registered files/buffers) | Irrelevant for NFS targets, but a useful reference for the harness itself. |
| io_uring ops **punted to io-wq** (kernel worker thread performs the blocking call) | ~100–300 K ops/s | Context switch + wakeup per op. See Risk R1. |
| NFS client, per mount | ~100 K to a few 100 K small ops/s | Bounded by NFSv4.1 session slots (server-negotiated), RPC transport slots, `nconnect` (≤16), per-inode locking. |

**Conclusion:** one client host at 3 M wire ops/s is not plausible. Assume **10–50 client hosts**
for a small-file saturation test of a serious system. The design must be **multi-host from
day one** (see §4.3), even if the PoC runs on one.

### 3.3 Concurrency per simulated GPU — closed loop matters

In a real PyTorch `DataLoader`, each of W worker processes builds *whole batches* and reads its
files **sequentially with blocking syscalls**; prefetch depth is `prefetch_factor` batches per
worker, and the main process consumes batches **in order** (head-of-line blocking if worker j is
slow). So the queue depth offered by one GPU is ≈ W, not B.

Total outstanding ops ≈ G × W. Example: 43K GPUs × 8 workers ≈ 344K outstanding ops, split
across hosts. The runner can hold that easily (§2.2). The important part is **not** to
"optimize" it into a high queue-depth blaster. The benchmark's value is that it offers a
realistic, latency-sensitive, closed-loop load, and scales it up only by adding actors.

### 3.4 Speed of light of the benchmark code (2026-10-01, from the host counters)

§3.2 guessed; the host counters in the report (`runner/REFERENCE.md` §4) measure. On the
development box (WSL2, 20 cores), from the page cache so that no storage is in the path:

| run | ops | bytes | process CPU | derived |
|---|---|---|---|---|
| `train_small_files`, 64 GPUs, `io_uring --threads 4` | 180k | 3.2 GB | 0.94 s | ~2 µs per op after the copy |
| same, `io_uring` with 20 loops | 180k | 3.2 GB | 3.3 s | contention: 41 io-wq workers at peak |
| `train_large_samples` (1 MiB reads), 8 GPUs, `io_uring --threads 4` | 101k | 81 GB | 16 s | 5 GB/s per core, all kernel copy |
| dry run, the VM alone (`runner/REFERENCE.md` §3) | | | | 150–350 ns per op |

The benchmark code costs about **2 µs of CPU per op and nothing measurable per byte**; the
rest is the Linux NFS client. The estimate below is therefore "runner plus kernel client
against a server that keeps up", for one node of **96 cores with two 400 GbE ports**
(~92 GB/s of NFS payload on the wire; buffered reads move three times that through DRAM,
DMA in plus the copy out; roughly 25–40 µs of client CPU per small RPC **[verify on a real
client]**, plus the per-connection and per-mount serialization in sunrpc):

| workload | one node | ten nodes | what binds |
|---|---|---|---|
| 1 MiB reads, buffered | 60–90 GB/s | 0.6–0.9 TB/s | the wire, given 30-plus TCP flows or RDMA; otherwise per-flow and per-mount limits, and ~280 GB/s of DRAM traffic |
| 1 MiB reads, `O_DIRECT` | 85–92 GB/s | ~0.9 TB/s | the wire; CPU halves without the copy |
| 128 KiB files, open-read-close | 0.5–0.7M files/s (65–90 GB/s) | 5–7M files/s | wire and RPC rate meet here: ~100 µs of client CPU per file, 3 RPCs per file |
| 4 KiB `O_DIRECT` random reads | 1–3M IOPS | 10–30M IOPS | RPC CPU and per-mount RPC rate; several mounts needed |
| metadata-only RPCs | 1–3M RPC/s | 10–30M RPC/s | same; OPEN is the expensive one |
| checkpoint writes | 80–90 GB/s | ~0.9 TB/s | the wire; COMMIT and close-to-open semantics |

The runner's share is under 1 % of the CPU at the bandwidth ceilings and 5–10 % at the RPC
ceilings. Ten nodes are linear because the coordinator acts only at phase barriers. Three
caveats: the per-RPC CPU and the connection scaling are guesses until a real client runs
against a real target (the loopback cannot test them); a 96-core node is two sockets, and
buffered reads at 280 GB/s of DRAM traffic need NUMA placement of loops and buffers; and the
20-loop row says the `io_uring` lever on a big node is io-wq, not loop count:
`IORING_REGISTER_IOWQ_MAX_WORKERS` and `--threads` at about a quarter of the cores are the
first knobs to build (§8.5). (Built 2026-10-01 as `--iowq-max-workers`, per loop; on the
loopback mount a cap of one worker per loop lost nothing: `runner/REFERENCE.md` §8.) Supersedes the per-core guesses in §3.2 where they differ.

---

## 4. Architecture implications from the numbers

### 4.1 Determinism: what's repeatable, what isn't

- **Repeatable:** the exact sequence of operations *each actor* issues, given (seed, actor id, epoch).
- **Not repeatable (and shouldn't be):** the global interleaving and timing across actors.
  That's set by storage latencies, which are what we're measuring.

Requirements that follow:

1. **Positional, counter-based RNG**, keyed on `(seed, actor_id, site_id, [enclosing loop
   indices])` (e.g. SplitMix/wyhash/Philox-style hash). Never a shared, stateful RNG: its output
   would depend on thread count and completion order. Also **no per-actor draw counter**: the
   loop indices are the counter, so the op at (GPU g, step s) can be computed without simulating
   steps `0..s`. See `GRAMMAR_OPTIONS.md` §2.
2. **Static sharding of the consumer by formula**: GPU g, batch b, item j takes permutation
   position `g + G·(b·B + j)`, and batch b is built by worker `b mod W` (this is
   `DistributedSampler` + round-robin `BatchSampler` + round-robin worker assignment). A shared
   dynamic "next file" pool, or a per-GPU counter shared by its W workers, would make file→worker
   assignment timing-dependent. Epoch wrap is `drop_last`: an epoch is `floor(N / (G·B))` steps,
   and all GPUs switch keys at the same step.
3. **Finite producers.** A loader is declared with its total batch count (`steps`) and dispatches
   exactly that many, so workers never prefetch past the last step and no in-flight op has to be
   cancelled at the end. The op multiset is fixed before the run starts.

### 4.2 Threading model

- One OS thread per core, one io_uring per thread, actors partitioned across threads.
  No shared mutable state on the hot path.
- Event loop: advance runnable actors → queue SQEs → `submit_and_wait` → drain CQEs →
  dispatch to actor via `user_data` → repeat.
- Emulated compute (`sleep`): thread-local **timer wheel** (tens of thousands of concurrent
  timers), not one `IORING_OP_TIMEOUT` per actor. Timescale is ~100 ms, so accuracy is not a
  concern. **Gotcha:** with only timers pending and no I/O outstanding, a plain
  `submit_and_wait` never wakes. Use the submit variant that takes a timeout
  (`IORING_ENTER_EXT_ARG`, kernel ≥ 5.11) set to the next timer-wheel expiry.
- I/O backend behind a trait. **The blocking thread pool (one OS thread per worker actor,
  blocking `pread`) is the fidelity reference, not a fallback**: it is literally what PyTorch
  does. io_uring is the scaling lever. Build the pool backend first; it is smaller and gives the
  baseline for the A/B in §8.5.
- The trait has two halves: `issue(op, buffer) -> token` and a **completion source** the event
  loop can wait on. Blocking APIs (`sync`, `sync-direct`, `posix-aio`, `mmap`, `libnfs` sync
  calls, `cuFileRead`) complete on the thread pool, which writes the per-thread `eventfd`.
  Submit-and-poll APIs (`libaio`, cuFile batch, NIXL) get a poller thread per event-loop thread
  that feeds the same `eventfd`. (As built 2026-10-01, `libaio` needs no poller: the loop
  waits in `io_getevents` and the eventfd is an `IOCB_CMD_POLL` in the same context;
  `DESIGN_REVIEW.md` §3.35.) io_uring is the only backend whose completions arrive natively.
  The full backend list and its three axes (initiation, completion, memory target) are in
  `PROJECT_BRIEF.md` §4.
- Buffers: per-thread ring larger than L3, so read copies miss cache the way a real loader's do
  (§2.2).
- Barrier: per-host atomic counter + cross-host coordinator. At one barrier per 500 steps
  (~150 s at T=0.3 s), even a 10 ms network barrier costs nothing.

**Built 2026-10-01** (`runner/aeiou/src/uring.rs`, `runner/REFERENCE.md` §8, `DESIGN_REVIEW.md`
§3.29): as above, with a heap of deadlines in the timer wheel's role (and `IORING_ENTER_EXT_ARG`
carrying the nearest expiry, as foreseen), one ring per event-loop thread, instances
round-robin over the loops with their sub-actors beside them, the VM a resumable state
machine parked at its current event, one op in flight per actor, and every read and write
issued at its effective offset after Linux 6.18 was seen not to advance the position of an
`O_DIRECT` file for a −1 read through the ring. The blocking pool remains the fidelity
reference; the two put identical RPCs on the loopback NFS wire.

### 4.3 Multi-host

Because every actor's behavior depends only on (seed, actor id), hosts can be given actor ranges
(`--gpus 10000 --rank 3 --ranks 20 --coordinator host:port`) and need to communicate only for
a start gate, barriers, one stats/fingerprint reduction at the end, and stop. **Decision
(2026-09-28): a small pure-Rust TCP coordinator, not MPI** — see §8.A for the protocol and how
barrier release integrates with the io_uring event loop. "Rank" in this document means the host
index that selects a GPU id range, nothing more.

### 4.4 The abstract language needs more than the regex sketch

The proposed grammar (operation / sequence / selection / consumer) is purely sequential. The
training workload as described also needs:

| Needed construct | Why |
|---|---|
| **Variable binding** (`f = consume(files)`) and references (`open(f)`, `read(f, …)`) | The selection picks *which file*, and the following fixed sequence must act on it. |
| **Replication / actors** (`per gpu`, `workers W`) | Scaling is by instantiating the abstract G×W times. |
| **Explicit loop indices** (`for step in $steps`) | The RNG key, file names such as `ckpt/{step:06}`, and `every N` all refer to them. No actor-local counters. |
| **`parallel(W)` + bounded `channel(capacity, ordered/unordered)`** as primitives | Fork/join, bounded prefetch, and in-order (head-of-line) or unordered delivery all fall out of these two. PyTorch's `loader` is sugar; tf.data-style unordered interleave, DALI, and checkpoint writer pools need nothing new. |
| **Finite producers** (`loader … batches = $steps`) | Workers must not prefetch past the last step, or the op multiset depends on when the run stops. |
| **Barrier(scope)** | Every-500-steps all-reduce; checkpoint phases. |
| **Distributions as parameters** (sizes, counts, sleep times, repeat counts) | Already in the sketch; must accept CLI overrides. |
| **Per-epoch reset** of consumers | Reshuffle = new key. |

Without these, the "dependency graph" can't be expressed. This is the biggest design risk
because it is easy to either under-build (can't express DataLoader) or over-build (a general
programming language). Suggested approach: compile the abstract to a small bytecode that
per-actor interpreters execute (a regex-VM-style design: `(pc, repeat-count)` stack frames),
with fork/join/barrier as VM instructions.

### 4.5 Op chaining in io_uring

io_uring can express `open → read → read → close` as an `IOSQE_IO_LINK` chain using *direct
descriptors* (fixed-file slots; kernel ≥5.15, auto-alloc ≥5.19), saving a user-space round trip
per op. **Recommendation: don't link by default.** Issue the next op of a sequence on completion
of the previous one.
- That matches real applications (and the round trip is µs versus 100s of µs for NFS RPCs).
- A **short read breaks a link chain** and cancels the rest (`-ECANCELED`). EOF reads,
  unknown/variable file sizes, and the "read until 0" idiom all hit this.
- Keep linking as an optional "aggressive" mode.

---

## 5. Risk register

| # | Risk | Impact | Likelihood | Mitigation / early test |
|---|---|---|---|---|
| **R1** | **io_uring on NFS runs largely through io-wq worker threads.** `openat`/`close`/`statx` usually punt, and buffered reads that miss the page cache likely punt on NFS (async buffered read support is filesystem-dependent) **[verify]**. In that case io_uring behaves like a kernel thread pool, so its advantage over a user-space blocking thread pool may be small for this workload. **Hypothesis to test:** NFS direct I/O is AIO-capable (the client issues the RPCs and returns `-EIOCBQUEUED` for a non-synchronous kiocb), so an O_DIRECT read should *release* its io-wq worker immediately, while a buffered cache-missing read *holds* one for the whole RPC round trip. If so, O_DIRECT keeps `iou-wrk` count small at tens of thousands of outstanding reads and only opens/closes cost a worker each **[verify]**. | CPU per op much higher than expected; per-host op rate lower; "use io_uring" stops being the main performance lever. | High | **Spike 1** (below), measuring `iou-wrk` count separately for open-heavy and read-heavy phases. I/O backend behind a trait; the blocking thread pool is the fidelity reference and is built first (§4.2). Tune `IORING_REGISTER_IOWQ_MAX_WORKERS`; consider `IORING_SETUP_ATTACH_WQ` to share one io-wq pool across rings. **Loopback observation (2026-10-01, `runner/REFERENCE.md` §8):** 20 `iou-wrk` threads at peak, ~~the core count,~~ for buffered and `O_DIRECT` reads alike, because every `openat` punts; the real-target measurement must separate open-heavy from read-heavy phases. **Corrected the same day with `--iowq-max-workers`:** 20 was the actor count; the kernel's cap is `min(SQ entries, 4 × cores)` = 80 per loop; 72 actors peak at 30 to 46 workers, and four workers in all (a cap of 1 per loop) carry the same run in the same time. |
| **R2** | **The client is the bottleneck, not the storage.** Per-mount NFS slot/session limits, RPC processing, per-host NIC. | Can't saturate the SUT from one host; the results measure the client. | High for small-file workloads | Multi-host from the start (§4.3). Report per-host client CPU and NFS RPC stats (`/proc/self/mountstats`) with every run. `nconnect`, multiple mounts. |
| **R3** | **Client caching distorts the workload.** Page cache, dentry/inode cache (~50–60 GB for 50M files), attribute cache, NFSv4 delegations (can turn OPEN/CLOSE into local ops). | Server sees a different (lighter) op mix than intended; results drift across a run as caches warm. | High | Options for O_DIRECT, ~~drop caches per epoch~~ `--drop-caches` at the start gate only, never inside a run (decided 2026-10-01, `DESIGN_REVIEW.md` §3.31), recommended mount options (`actimeo`, `lookupcache`), and a check that dataset ≫ aggregate client RAM. Record server-side op counts to validate. |
| **R4** | **Abstract language under- or over-designed** (§4.4). | Can't express the target workloads, or the PoC turns into a compiler project. | Medium–High | Write the 3–4 target abstracts (training small-file, training large-sample, checkpoint write burst, checkpoint restore) *by hand, on paper* before writing the parser. **Done 2026-09-29/30:** eight abstracts on paper (`ABSTRACTS.md`), nine constructs added and accepted, AST schema v0.1 drafted (`schema/`). The node set is now fixed for the VM; growth from here needs a §9-style entry and a schema version bump. |
| **R5** | **Determinism is broken by accident.** Shared RNG, dynamic work distribution, `HashMap` iteration order, thread-count-dependent sharding, per-actor counters shared by concurrent workers, producers that run past the last step, an order-dependent fingerprint. | "Same seed, same workload" stops being true, and runs are no longer comparable. | Medium | Positional RNG keyed on loop indices, no draw counters; sharding by formula; finite producers; order-independent fingerprint (§8.D); dataset seed separate from run seed; a `--dry-run` mode that prints any actor's op stream for any step range (`--dry-run --gpu 17 --steps 300..302`) so two runs can be diffed and golden-tested in CI. |
| **R6** | **Dataset generation cost.** 50M creates at 5–50K creates/s = **17 min – 2.8 h**; directory fan-out affects both creation and lookup performance. | Slow iteration; layout mismatch between generator and abstract. | High (certain to be slow) | `datagen` mode in the same binary, driven by the *same* filename pattern and size distribution as the abstract. Hierarchical layout (≤~10K entries/dir). Resumable. |
| **R7** | **Write data is dedupable/compressible** (shared buffer reused for every write). | Inflated write and checkpoint numbers on systems with data reduction. | Medium | Per-write content generated positionally from `(dataset seed, file id, block index)` with **controlled** dedupe and compression ratios (dgen-py or a block-wise wrapper around it, `PROJECT_BRIEF.md` §5). ~~With a per-block header carrying the same tuple; the header doubles as the verification target (R18).~~ Revised 2026-09-29: no headers, since a unique header per block forces a dedupe ratio of one; the generator's determinism is what makes the data verifiable. dgen-data investigated the same day (`DESIGN_REVIEW.md` §3.17): positional at 1 MiB granularity by seed arithmetic; corpus-wide dedupe is our layer, not the library's. |
| **R8** | **io_uring disabled by the OS, or resource limits too low.** Containers are out of scope (runner runs on bare Linux), so the remaining exposure is the `kernel.io_uring_disabled` sysctl (6.6+) set by some hardened/enterprise distros **[verify target OS]**, `RLIMIT_MEMLOCK` on kernels <5.12, and **`RLIMIT_NOFILE`**: 2,000 GPUs per host × W=8 is 16K files open at once per host, plus a fixed-file table of the same size, plus hundreds of thousands of NFSv4 open stateids server-side across the cluster. | Runner won't start on some lab hosts, or fails mid-run with `EMFILE`. | Low–Medium | Startup probe that computes the needed fd count from G, W, and the abstract, raises the soft limit, and fails early with a clear message naming the sysctl or limit. Report open-file high-water mark with the run. **Built 2026-10-01** (`runner/REFERENCE.md` §11): the count is estimated from a bounded walk of the abstract, not from G and W alone; threads and `vm.max_map_count` are checked with it; `RLIMIT_MEMLOCK` is not. |
| **R9** | **Measurement volume.** Per-op logging = GBs per epoch. | Measurement perturbs the run or fills disks. | Medium | HDR histograms per thread/op type, bucketed by step range so steady state can be selected after the run; 1 s time series; per-step stall time per GPU (§2.2); merge at end. Optional sampled tracing. |
| **R10** | **Emulating PyTorch too literally or not literally enough** (Python `open()` side syscalls, in-order batch delivery, worker sequential reads). | Results don't match real training runs. | Medium | Validate against real `strace`/NFS server op-mix captures of a small real training run; compare op mix and per-file latency distribution. |
| **R20** | **"Random" used as a stand-in for data-dependent access** (VDB search, KV cache). Storage systems exploit reuse, sequential runs, popularity and size skew, dependency shape, and write-then-read lag; uniform random has none of these, so the SUT is denied benefits it would get in production and the benchmark under-reports. SSD firmware and server readahead are known to detect such structure. | Results for VDB/KV-cache classes do not reflect real behavior; vendors dispute them. | High for those classes | Random is the null model only. Add distributions over ids/offsets, recency references, and positional write-then-read names to the model (`GRAMMAR_OPTIONS.md` §5). Accept an abstract for a workload class only when its locality metrics (from `--dry-run --metrics`) match the real trace's. Validate end to end with the `trace` node (`replay` until 2026-10-02) at small scale. Parameters (Zipf exponents, reuse distances, cluster sizes, hop counts) are measured from real indexes on real data, as compute time is. |
| **R21** | **Cross-actor read-after-write is timing-dependent** (one GPU's prefill writes a KV block another GPU later reads). | Either the fingerprint breaks or the workload cannot be expressed. | Medium | Restrict to barrier-separated phases, or model hit ratio and lag statistically with fail-soft miss reads. Cache capacity and eviction are workload inputs, not emergent. Documented per workload as a fidelity loss. |
| **R11** | **Long runtimes at realistic scale** (6 h/epoch at 100 GPUs on 50M files). | Painful testing; tempting to cut corners on fidelity. | Certain | Step/time-bounded runs; "accelerator utilization ≥ X%" as the pass metric (MLPerf Storage style), which converges long before an epoch ends. |
| **R12** | **Dev environment is WSL2.** io_uring works on WSL2's 6.x kernel, but there's no realistic NFS target, and `/mnt/c` (9p/drvfs) behaves nothing like production. | Wrong conclusions from local testing. | Medium | Develop logic locally against ext4/tmpfs **and a loopback NFS mount** (`nfs-kernel-server` exporting a tmpfs directory, mounted from `localhost`; needs systemd enabled in WSL2). No performance meaning, but it exercises the real NFS client paths (io-wq punting, O_DIRECT on NFS, attribute caches), which ext4 does not. Run all performance work on real Linux clients against real NFS. |
| **R14** | **`mmap`-based loaders** (safetensors, Arrow-backed Hugging Face datasets, `np.load(mmap_mode=…)`). | Page faults are not submittable through io_uring, and fault-around/readahead, not the abstract, decide the RPC sizes. | Medium | **In scope as a backend (revised 2026-09-28):** runs on the blocking thread pool; `read(f, off, len)` maps to `MADV_POPULATE_READ` (5.14+) or touching the pages, with `MADV_WILLNEED` as a prefetch variant. The abstract is unchanged; the RPC pattern is reported, not assumed. **Built 2026-10-01** (`runner/REFERENCE.md` §9): on the loopback mount the fault path sent 39 % more READs than `read(2)` for the same bytes and a GETATTR per mapping; `MADV_WILLNEED` restored one READ per file. A read touches one byte per page and copies nothing (`DESIGN_REVIEW.md` §3.35, Revised); warm, the map/unmap/fault overhead still cost 20 to 35 % more CPU than `sync` on small files, and on 140 MiB files `mmap` used a tenth of `sync`'s CPU; cold on those files the fault path sent six times as many READs of a sixth the size (109 KiB against 660 KiB), which the mount's `read_ahead_kb` (128 by default) sets: at 1024 the READ count fell to `read(2)`'s and cold `mmap` became the fastest row (`runner/REFERENCE.md` §9). |
| **R15** | **GDS silently in compat mode.** cuFile falls back to a POSIX bounce buffer when the filesystem, driver, or MOFED stack lacks true GDS support, which on NFS is the common case. | A "GDS" result that measured a bounce-buffer path. | High on NFS | The `gds` backend reads `cufile_stats`/the cuFile JSON config and the per-handle compat flag at startup and after the run, prints it in the report, and fails if `--require-gds` is set. |
| **R16** | **GPU-memory backends need a CUDA device on every client.** `gds` and `nixl-posix` cannot run on CPU-only client nodes, and the GPU is only a sink. | Those rows of the A/B matrix run on fewer hosts than the rest. | Certain | Record it in the matrix; keep the abstract and fingerprint identical so the rows are still like-for-like on the hosts that can run them. |
| **R18** | **The corpus is not what the abstract thinks it is.** ~~A shim or client below the interposition line returns wrong or stale data fast.~~ Reframed 2026-09-29: the realistic case is a wrong corpus (older generator version, partial regeneration after a crash, wrong dataset seed, wrong dedupe/compression setting), which hands the SUT's data reduction a free lunch. A fabricating shim is fraud, handled by rules and review. A warm cache from a previous run has *correct* content and is invisible to any content check; it is handled by fresh seeds (R19) and dataset sizing (§8.C). | An invalid result that looks fast. | Medium | Manifest check at startup (pattern, count, sizes, seed, generator version and parameters). Runner structural checks: byte counts, short reads, index bytes equal to the computed layout. Content check by the separate Python `verify` tool that regenerates expected bytes from the same generator, after `datagen` and on reviewer request; never in a scored run. ~~Per-block headers and sampled header checks on reads.~~ |
| **R19** | **The solution learns the file order.** With a known seed and abstract, a shim or client could prefetch the next file and defeat the closed loop. | Results that do not reflect the real workload, where the sampler is inside the application. | Low–Medium | Seed privacy (`PROJECT_BRIEF.md` §5): seed, abstract, and file order are application-private; reviews re-run with a fresh seed and expect the same result within noise. |
| **R17** | **Backend is part of the application.** Two runs with different backends are two different applications, not two storage solutions. Mixing the two comparisons attributes client-stack differences to storage. | Wrong conclusions from the matrix. | Medium | Comparison policy in `PROJECT_BRIEF.md` §5 (decided 2026-09-28): CLOSED uses the framework's real API (`sync` for PyTorch); other backends measure the SUT's speed of light and advise implementors; fix the backend when comparing storage; fix the storage when comparing backends. RPC counts, client CPU, and client DRAM are audit data. |
| **R22** | **Container formats modeled with the wrong shuffle** (added 2026-09-29). Random sample access inside Parquet or TFRecord would emulate a pattern no production reader produces (they shuffle shards and stream row groups); conversely, streaming a format that real map-style loaders access randomly (HDF5, Arrow, MDS) understates seek load. | The container rows of the benchmark measure a fiction, in either direction. | Medium–High | Access mode is declared per dataset (`map` / `stream`) and the format class refuses modes the real format does not support. Format classes are derived from and validated against traces of the named reader library. Loader-side choices (interleave, prefetch, projection) stay in the abstract, never in the class. `GRAMMAR_OPTIONS.md` §6. |
| **R13** | ~~liburing vs. Rust crate~~ **Decided:** use the pure-Rust `io-uring` crate. | — | — | — |

---

## 6. Spikes to run before committing to the design

1. **io_uring vs. blocking thread pool on the real NFS target** (addresses R1, R2, R8).
   Microbenchmark `open → read(S) → close` on random files at QD 1…4096, S ∈ {4K, 128K, 1M},
   buffered and O_DIRECT. Measure ops/s, client CPU per op, and how many io-wq workers are
   spawned (`ps -eLf | grep iou-wrk`), **separately for an open-heavy phase and a read-heavy
   phase**. Hypothesis (R1): O_DIRECT reads release their worker immediately because NFS direct
   I/O completes asynchronously; buffered cache-missing reads hold one per outstanding RPC. This
   decides whether io_uring is the main lever or just a convenience, and whether O_DIRECT is
   required for scale as well as for reclaim.
2. **Feistel permutation**: correctness (a bijection over [0, N) for N = 50M/100M, checked
   with a bitmap), throughput (target <50 ns/draw), and a quick check that there's no locality
   (distribution of |Δ| between consecutive draws). Correctness half done 2026-09-30: the
   runner's `Perm` (`runner/aeiou/src/rng.rs`) is bitmap-checked at N = 50M in its tests;
   throughput and the |Δ| distribution are still to measure.
3. **Client cache behavior at scale**: `stat`/`open` 50M files from one host, watching slab
   growth (`slabtop`) and the NFS op mix (`nfsstat -c`, `mountstats`).
4. **Paper abstracts** for the target workloads (R4), derived from `strace` of the real
   applications, then decide the grammar: small-file training, large-sample training, checkpoint
   write, **checkpoint restore** (the one workload where many ranks read the same files, so
   delegations and page cache matter), and the data-dependent classes: **DiskANN-style search,
   IVF search, index build, KV-cache serving** (R20; sketches in `GRAMMAR_OPTIONS.md` §5.5).
5. **Locality metrics of a real VDB or KV-cache trace** (R20): capture one with `strace`/eBPF,
   compute reuse-distance, run-length, popularity, size, and dependency-depth distributions, and
   check that the model of `GRAMMAR_OPTIONS.md` §5.2 can reproduce each one. This is the test of
   whether the architecture's shape is right before code exists.

Order of work after the abstracts: build the VM with `--dry-run` and the fingerprint against
ext4 and loopback NFS (R12), then Spike 1 on the real target with the thread-pool backend first.
(`aeiou dry-run` and the fingerprint exist since 2026-09-30, `DESIGN_REVIEW.md` §3.22;
`aeiou run` with the `sync` and `sync-direct` backends since the same day, §3.23, run against
ext4; the loopback NFS run and Spike 1 on the real target are next.)

---

## 7. Bottom line

- **DRAM is a non-issue for the runner** if the consumer is a keyed permutation and filenames
  and sizes are computed, not stored: under 1 GB of state even at 100M files and ~50K simulated
  GPUs, plus a few GB of sink buffers chosen deliberately to defeat the cache.
  The 200/400 MB shuffled-list approach is also perfectly viable. The bitmap's slowdown is
  real (N·ln N total probes, near-N probes for the last draws) but it's a tail-latency problem,
  not a throughput one.
- **Client-side DRAM (kernel caches) is an issue**, mostly as a fidelity problem.
- **IOPS ceiling is set by the NFS client path and io-wq punting, not by the runner's state
  machine.** Plan for many client hosts and validate io_uring's benefit on NFS early.
  Measured 2026-10-01 (§3.4): ~2 µs of CPU per op in the runner, nothing per byte; one
  96-core, 2×400 GbE node is wire-bound near 90 GB/s and RPC-bound near 1–3M RPC/s.
- **The biggest design risk is the abstract language.** It needs binding, explicit loop
  indices, `parallel` + `channel`, finite producers, and barriers beyond the regex-style sketch.
- **Exactness is a property of the whole design, not just the RNG.** Positional randomness, a
  sharding formula instead of counters, finite producers, a dataset seed separate from the run
  seed, and an order-independent fingerprint are each necessary (§4.1, §8.D).

---

## 8. Round 2: user decisions and follow-up analysis (2026-09-24)

User inputs recorded:
- **Multi-host with one shared seed.** Each rank takes its own subset of the work with no
  negotiation over the network. Originally MPI; **revised 2026-09-28 to a pure-Rust TCP
  coordinator** (see §8.A) to avoid installing and ABI-matching OpenMPI/MPICH plus libclang on
  every bare-Linux client and build host.
- **Run length.** Runs are bounded by a step count (currently 500, adjustable).
- **Dataset sizing.** The dataset must be ≥ 5× the total DRAM of all client nodes.
- **Why the page cache matters.** Finding pages to evict gets non-linearly slower as memory
  fills, even when the pages are clean. This is the main motivation for io_uring and/or O_DIRECT.
- **Reproducibility.** Exact run-to-run reproducibility is required; the across-run randomness of
  real training is not.
- **Deployment and tooling.** Bare Linux, no containers. Use the `io-uring` crate.
- **Grammar.** Extensions are agreed; see `GRAMMAR_OPTIONS.md`.
- **Backends.** A/B testing of I/O backends is agreed; see §8.5.

### 8.A Same seed + rank-based sharding — yes, with two refinements

1. **Shard by global GPU id, not by rank.** Rank r owns GPU ids `[r·G/R, (r+1)·G/R)`.
   GPU g consumes permutation positions `g, g+G, g+2G, …`, so the work per GPU does not depend on
   R. A 1,000-GPU run on 10 nodes and on 20 nodes issues exactly the same operations. The only
   change is which node sends them.
2. **Apply the modulo to positions in the permutation, not to file ids.** With `file_id % R`, each
   node owns a fixed slice of the namespace, and that slice is the same in every epoch.
   `perm_{seed,epoch}(g + k·G)` gives each GPU a random subset instead, and a different one per
   epoch, which is closer to `DistributedSampler`. The node-to-node communication cost is still
   zero.

**Cross-host coordination: a pure-Rust TCP coordinator (decided 2026-09-28, replaces MPI).**
The runner needs only a start gate, a barrier every few hundred steps, one reduction of stats
and the fingerprint at the end, and stop. That is a few hundred lines over TCP, and it removes
the OpenMPI/MPICH + libclang dependency from the build host and the MPI runtime (and its ABI
matching) from every client node. The result is one static binary copied with `scp`.

- **Topology:** star. Rank 0 (or a separate `runner coordinate` process) listens. Every rank
  connects once at startup and keeps the socket for the whole run.
- **Transport:** blocking `std::net` on a dedicated coordinator thread. No tokio, no tonic/gRPC:
  HTTP/2, protobuf codegen, and a second async runtime buy nothing for one barrier per ~150 s.
- **Wire format:** length-prefixed structs serialized with `postcard` (or `bincode`):
  `Hello { rank, ranks, config_fingerprint }`, `Ready`, `Start { t0 }`, `Arrive { barrier_id }`,
  `Release { barrier_id }`, `Stats { bytes }`, `Stop { reason }`, plus a periodic `Heartbeat`.
- **Config check:** `Hello` carries a hash of the abstract, parameters, seed, G, and the dataset
  manifest. The coordinator refuses any rank whose hash differs, before any I/O starts. This
  replaces the protection `mpirun`'s single command line used to give.
- **Start gate:** all ranks send `Ready` after rings, buffers, and the manifest check are done;
  the coordinator replies `Start` with a common `t0` so AU windows and step buckets align.
- **Barrier:** the last event-loop thread on a host to arrive (atomic counter) tells the
  coordinator thread, which sends `Arrive`. When all ranks have arrived the coordinator sends
  `Release` to every rank; the coordinator thread writes the local `eventfd`, and each event-loop
  thread keeps an io_uring read posted on that eventfd, so the release shows up as an ordinary
  completion. Fan-out to 20 hosts is well under 1 ms against a 300 ms step.
- **Reduction:** at the end each rank sends its histograms (a few MB) and its fingerprint partial
  sum; the coordinator merges histograms and adds fingerprints modulo 2^64.
- **Faults:** a missed heartbeat or a closed socket aborts the run on every rank, which is what
  MPI would have done.
- **Launch:** `pdsh` or an ssh loop starting `runner --rank r --ranks R --coordinator host:port`
  on each host, replacing `mpirun`.

**Coordinator trait.** The TCP coordinator sits behind a `Coordinator` trait (`start`, `barrier`,
`reduce`, `stop`). A single-host run uses an in-process implementation and opens no sockets, which
keeps WSL2 development and CI simple. There is no MPI build dependency in any configuration.

**Built 2026-09-30** (`runner/aeiou/src/coord.rs`, `runner/REFERENCE.md` §6, `DESIGN_REVIEW.md`
§3.25), as above with four changes: JSON frames instead of `postcard`; a reader thread per
socket instead of one event loop, with rank 0 serving in-process and connecting as a client;
the release wakes a condvar ~~, the eventfd waits for the `io_uring` backend~~ (and, since
2026-10-01, writes every eventfd an `io_uring` loop subscribed, so the read posted on it is
the completion this section planned); and `Leave` beside
`Arrive`, so a host whose instances have all finished stops being expected. The trait is
`barrier`/`leave`/`stop`; the start gate (`ready`) and the reduction (`report`/`result`) are
the client's own methods, which `aeiou run` calls around the run. `runner/aeiou-launch` is
the ssh loop.

### 8.B Does O_DIRECT keep the NFS client out of the page cache? Yes, for data.

With O_DIRECT, Linux NFS reads and writes skip the page cache. User pages are pinned and the RPC
data goes into or out of them directly (one copy from the socket over TCP; direct placement over
RDMA). As a result:
- no page-cache insertion and no eviction/reclaim;
- no LRU pressure, so the non-linear reclaim slowdown does not occur;
- no readahead.

This is the most direct fix for the problem described.

Consequences and caveats **[verify on the target kernel]**:
- **No kernel readahead.** Each `read()` becomes READ RPCs split at `rsize`, and small
  application reads become small RPCs. Turn this into a feature: the abstract's `readahead` /
  `xfer` parameter becomes the application transfer size. The wire pattern is then set
  explicitly by the benchmark and deterministic, instead of depending on the client's
  readahead heuristics.
- **Alignment.** NFS O_DIRECT has historically not needed block-aligned buffers or offsets,
  unlike local block devices. Page-aligned buffers are still cheaper.
- **Writes.** O_DIRECT NFS writes wait for the data to be stable on the server (a stable write,
  or UNSTABLE + COMMIT) before returning. This matters for checkpoint abstracts.
- **Metadata is unchanged.** OPEN, CLOSE, GETATTR, and LOOKUP cost the same, and the
  dentry/inode slab (§2.3) still grows. O_DIRECT fixes the data path only. For the small-file,
  metadata-heavy regime, the per-host op ceiling in §3.2 still applies.
- **io_uring + O_DIRECT on NFS may still go through io-wq for submission**, because the NFS
  client does not advertise non-blocking submission (`FMODE_NOWAIT`). The expected shape is one
  worker *dispatch* per read (a context switch), but the worker is released as soon as the RPCs
  are queued, because NFS direct I/O completes asynchronously through the kiocb. Buffered reads
  that miss the cache, by contrast, hold a worker for the whole round trip. Spike 1 measures
  exactly this (R1). If confirmed, O_DIRECT is what keeps the worker count bounded at scale.
- **Fidelity.** Real PyTorch uses buffered I/O. Accepting O_DIRECT is a benchmark-policy
  decision; with the explicit readahead-size emulation above, the load the server sees can be
  kept representative.

**Observed on the loopback NFS v4.2 mount, 2026-09-30** (`runner/REFERENCE.md` §7,
`DESIGN_REVIEW.md` §3.26; the WSL2 6.18 client against a server on the same kernel, so the
target kernels are still to verify): O_DIRECT reads and writes bypass the client cache (the
mount's `directreadbytes` counter carries them all, and a same-host checkpoint restore that
issues zero READ RPCs buffered issues 30 direct); there is no readahead under O_DIRECT, one
RPC per aligned application read, with an unaligned read rounded out to 4 KiB; buffered reads
are merged into fewer, larger RPCs; O_DIRECT writes go out one RPC per application write
where buffered ones coalesce to `wsize`; metadata is unchanged, a GETATTR per open even when
every byte is cached. The io-wq question (the last bullet) waits for the `io_uring` backend.

Other options to A/B for the reclaim problem:
- **`RWF_DONTCACHE` / uncached buffered I/O.** Merged around Linux 6.14. Buffered I/O that drops
  the pages after use, so readahead is kept and reclaim is avoided. Filesystem opt-in; NFS client
  support arrived later or may still be pending **[verify]**.
- **`posix_fadvise(POSIX_FADV_DONTNEED)` after reading each file.** Costs one extra syscall per
  file.

Other levers for more client throughput per host:
- `nconnect` (up to 16);
- NFS over RDMA, which saves the most CPU;
- `rsize`/`wsize` of 1 MiB;
- multiple server IPs or mounts;
- pNFS/FlexFiles, so data goes straight to the data servers;
- io_uring registered (fixed) buffers, which may avoid pinning pages on every O_DIRECT op
  **[verify]**.

### 8.C 500-step runs: implications

- **Files read per run:** 500 · B · G. At B=7, a 50M-file corpus is fully read once only at
  G ≥ 14,300. Below that, **no file is ever read twice**, epochs never end, and the consumer
  only walks a prefix of the permutation. Page-cache *hits* are therefore impossible by
  construction. The 5× rule works as a capacity/scale rule, not as a cache-hit guard.
- **What does matter with buffered I/O is how soon client DRAM fills up:** fill time ≈
  node DRAM ÷ node read rate. Example: 1 TiB at 25 GB/s → ~44 s, against a 150 s run
  (500 × 0.3 s). The first ~30% of the run is in the "free memory" regime and the rest is in the
  reclaim regime, so **results depend on run length**. O_DIRECT (or DONTCACHE) removes this
  dependence. Otherwise, report only the steady state after memory is full.
- `steps` and `sync_every` are separate parameters. With `steps = sync_every = 500`, there is one
  barrier, at the end.
- **`--dry-run` prints total ops and bytes per host and compares bytes to the host's DRAM**, so the
  regime the run will be in (free memory, reclaim, or both) is known before it starts.

### 8.D Is the workload exactly reproducible? Yes: per-GPU op streams are identical; timing and interleaving are not.

With these rules, the **exact multiset of operations and each GPU's exact op order** are
identical from run to run:
- a positional RNG keyed on `(seed, gpu_id, site, [enclosing loop indices])`, with no draw
  counters;
- Feistel permutation positions sharded by global GPU id by formula, `g + G·(b·B + j)`, with
  `drop_last` epochs;
- finite producers (the loader dispatches exactly `steps` batches);
- step-bounded runs;
- no conditionals on timing or completion order;
- file sizes known from the dataset definition.

A given operation always means the same file, offset, and length. This holds regardless of node
count, thread count, and I/O backend.

What is **not** identical, and can't be for any benchmark against real storage, is the global
interleaving and timing across GPUs. Those are the storage system's response, which is what is
being measured.

The workload changes **by design** when any of these change: the abstract, its parameters,
`--seed`, G, or N. Under the 5× rule, N scales with client DRAM, so a different cluster size
gives a different workload. That's expected, but the report should record it.

Things that would break this, and are treated as run failures rather than silent deviations:
- short reads, missing files, or I/O errors (for example, if the dataset doesn't match its
  definition);
- a "shared stateful RNG where each rank skips to every R-th value". That approach *is*
  reproducible, but each rank has to generate the whole stream. Counter-based RNG is O(1) instead.

**Workload fingerprint (order-independent):**
- Issue order within a GPU actor is *not* deterministic: its W workers run concurrently, so a
  rolling hash "in issue order" would change with storage timing. The fingerprint is therefore a
  **sum modulo 2^64 of per-op hashes** (xxh3) of
  `(actor_id, [enclosing loop indices], op, file_id, offset, len)`. Including the position makes
  the sum sensitive to *which* op happened *where*, so it is as strong as an ordered hash for this
  purpose while being independent of interleaving.
- Per-thread partial sums are added on the host and summed across ranks by the coordinator at
  the end of the run, then printed with the results.
- `--dry-run` computes the fingerprint without doing any I/O, in parallel and in any order, which
  gives a golden value for CI. `--dry-run --gpu g --steps a..b` prints one actor's op stream for a
  step range for diffing.
- Two runs with the same fingerprint executed the same workload. This also proves that A/B
  backend comparisons (§8.5) are like-for-like.
- Preconditions for exactness: finite producers (the loader dispatches exactly `steps` batches,
  §4.1), a dataset seed separate from `--seed`, and a manifest check that the corpus matches its
  definition.

### 8.5 A/B test matrix (all rows run the same abstract, seed, and G, so the fingerprints are equal)

| Dimension | Variants |
|---|---|
| I/O backend (`--io-backend`) | `sync` · `sync-direct` · `posix-aio` · `libaio` · `io_uring` · `mmap` (touch / `MADV_POPULATE_READ` / `MADV_WILLNEED`) · `gds` (sync / batch / stream) · `nixl-posix` · `libnfs`; `s3` deferred. See `PROJECT_BRIEF.md` §4. |
| Memory target (`--buffer`) | pageable host · pinned host · GPU (the last two only where the backend supports them) |
| Cache mode (`--cache`) | buffered · O_DIRECT · buffered + FADV_DONTNEED · RWF_DONTCACHE (if supported); a validity table rejects impossible pairs |
| io_uring features | fixed files (direct descriptors) on/off · fixed buffers on/off · `SINGLE_ISSUER`+`DEFER_TASKRUN` · SQPOLL · io-wq max workers · op linking on/off. Built 2026-10-01: `--defer-taskrun`, `--coop-taskrun`, `--sqpoll IDLE_MS` (`--sqpoll-shared` for one poll thread), `--iowq-max-workers N` (per loop); fixed files/buffers and linking not (`runner/REFERENCE.md` §8) |
| NFS client | nconnect 1/4/16 · rsize 256K/1M · TCP vs RDMA · v3 / v4.1 / v4.2 / pNFS |

Measured for each run:
- ops/s and bytes/s;
- **client CPU per op** (the number that decides how many client hosts are needed);
- peak `iou-wrk` and `iou-sqp` thread counts (in the report since 2026-10-01, with the task peak, CPU, RSS,
  and the `mountstats` deltas: `runner/REFERENCE.md` §4) and open-file high-water mark;
- backend-specific counters: libaio context depth, cuFile compat-mode flag and `cufile_stats`,
  NIXL plugin selected, `mmap` fault and populate counts (the `libaio:` and `mmap:` lines
  and the host line's fault counts since 2026-10-01, `runner/REFERENCE.md` §9; the mount's
  `read_ahead_kb` beside its options the same day);
- accelerator utilization (AU%), overall and per step bucket;
- **per-step stall time per GPU** (end of `compute` to next `take` returning);
- per-op latency histograms, bucketed by step range;
- `/proc/self/mountstats` RPC counts, kernel version, mount options, and the fingerprint.

Which rows may be compared with which is set by the comparison policy in `PROJECT_BRIEF.md`
§5 (R17): CLOSED results come only from ~~the `sync` row~~ the row of the backend the abstract declares (2026-10-01; `sync` unless the traced application uses another API, as `model_load` does `mmap`); the other rows are speed-of-light and
advisory data. Before trusting any backend's numbers, run fio with the matching engine (`psync`,
`posixaio`, `libaio`, `io_uring`, `mmap`, `libcufile`, `nfs`) against the same files as a
cross-check.

### 8.E Deployment

Bare Linux only, so R8 drops to *Low*: only the `kernel.io_uring_disabled` sysctl and memlock
limits on old kernels are left to check at startup.
