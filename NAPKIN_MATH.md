# Abstract-Driven I/O Benchmark Runner — Napkin Math & Risk Assessment

Status: pre-implementation analysis (2026-09-24, revised same day with user decisions — see §8).
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
| File sizes | constant, or `dist.sample(hash(seed, file_id))` | — | 0 |
| File sizes from a manifest (only if real, irregular datasets must be replayed) | 4 B × N | 50M / 100M | 200 MB / 400 MB |
| *Materialized path strings (anti-pattern, for reference)* | ~64 B path + 24 B `String` + allocator overhead | 50M / 100M | **~5 GB / ~10 GB** |
| Actor state (GPU + W workers + reorder window + interpreter stack) | ~256 B + 64 B × W + ~16 B × stack depth | W=8 → ~1 KB; 50K GPUs | ~50 MB |
| In-flight op records (slab indexed by `user_data`) | ~64 B × ops in flight | 1M in flight | 64 MB |
| io_uring rings | SQE 64 B, CQE 16 B; 4K SQ + 8K CQ ≈ 400 KB per ring | 64 threads | ~26 MB |
| Read buffers — **shared "sink" buffers** (data is discarded, concurrent reads may target the same buffer) | threads × 2 × max transfer | 64 × 2 × 1 MiB | 128 MB |
| *Read buffers — one per in-flight op (anti-pattern)* | in-flight × transfer | 344K × 128 KiB | **~44 GB** |
| Write source buffers (must be non-dedupable/non-compressible, see §5) | pool, or regenerate with fast PRNG | 64 threads × 16 MiB | ~1 GB |
| Latency stats: HDR histograms per (thread × op type) + 1 s time series | ~30–200 KB each | 64 × 8 | ~50 MB |
| *Per-op latency log (anti-pattern)* | ~24 B × ops | 50M files × 4 ops per epoch | **~5 GB / epoch** — stream to disk if ever needed |

**Runner total, recommended design, 100M files, ~50K simulated GPUs, 1M ops in flight:
well under 1 GB.** DRAM is not the constraint on the runner. It *is* a constraint on the kernel
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

---

## 4. Architecture implications from the numbers

### 4.1 Determinism: what's repeatable, what isn't

- **Repeatable:** the exact sequence of operations *each actor* issues, given (seed, actor id, epoch).
- **Not repeatable (and shouldn't be):** the global interleaving and timing across actors.
  That's set by storage latencies, which are what we're measuring.

Requirements that follow:

1. **Counter-based RNG**, keyed on `(seed, actor_id, epoch, step, draw#)` (e.g. SplitMix/
   wyhash/Philox-style hash). Never a shared, stateful RNG: its output would depend on
   thread count and completion order.
2. **Static sharding of the consumer**: GPU g takes permutation indices `g, g+G, g+2G, …`
   (like `DistributedSampler`). A shared dynamic "next file" pool would make file→GPU
   assignment timing-dependent and would also need cross-thread/cross-host synchronization.

### 4.2 Threading model

- One OS thread per core, one io_uring per thread, actors partitioned across threads.
  No shared mutable state on the hot path.
- Event loop: advance runnable actors → queue SQEs → `submit_and_wait` → drain CQEs →
  dispatch to actor via `user_data` → repeat.
- Emulated compute (`sleep`): thread-local **timer wheel** (tens of thousands of concurrent
  timers), not one `IORING_OP_TIMEOUT` per actor. Timescale is ~100 ms, so accuracy is not a
  concern.
- Barrier: per-host atomic counter + cross-host coordinator. At one barrier per 500 steps
  (~150 s at T=0.3 s), even a 10 ms network barrier costs nothing.

### 4.3 Multi-host

Because every actor's behavior depends only on (seed, actor id), hosts can be given actor ranges
(`--gpus 10000 --host-rank 3 --host-count 20`) and need to communicate only for barriers,
start/stop, and stats aggregation. **Decision: use MPI** (`mpi` crate / rsmpi) — see §8.A for
how barriers integrate with the io_uring event loop.

### 4.4 The abstract language needs more than the regex sketch

The proposed grammar (operation / sequence / selection / consumer) is purely sequential. The
training workload as described also needs:

| Needed construct | Why |
|---|---|
| **Variable binding** (`f = consume(files)`) and references (`open(f)`, `read(f, …)`) | The selection picks *which file*, and the following fixed sequence must act on it. |
| **Replication / actors** (`per gpu`, `workers W`) | Scaling is by instantiating the abstract G×W times. |
| **Fork/join** (`parallel`, `join`) | Batch = W concurrent readers; the step waits for all of them. |
| **Pipelining / bounded prefetch** (`prefetch N`) | The next batch loads while the GPU computes. |
| **In-order delivery** | Faithful DataLoader head-of-line behavior. |
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
| **R1** | **io_uring on NFS runs largely through io-wq worker threads.** `openat`/`close`/`statx` usually punt, and buffered reads that miss the page cache likely punt on NFS (async buffered read support is filesystem-dependent) **[verify]**. In that case io_uring behaves like a kernel thread pool, so its advantage over a user-space blocking thread pool may be small for this workload. | CPU per op much higher than expected; per-host op rate lower; "use io_uring" stops being the main performance lever. | High | **Spike 1** (below). Design the I/O backend behind a trait so a blocking-thread-pool backend is a drop-in fallback/baseline. Tune `IORING_REGISTER_IOWQ_MAX_WORKERS`. |
| **R2** | **The client is the bottleneck, not the storage.** Per-mount NFS slot/session limits, RPC processing, per-host NIC. | Can't saturate the SUT from one host; the results measure the client. | High for small-file workloads | Multi-host from the start (§4.3). Report per-host client CPU and NFS RPC stats (`/proc/self/mountstats`) with every run. `nconnect`, multiple mounts. |
| **R3** | **Client caching distorts the workload.** Page cache, dentry/inode cache (~50–60 GB for 50M files), attribute cache, NFSv4 delegations (can turn OPEN/CLOSE into local ops). | Server sees a different (lighter) op mix than intended; results drift across a run as caches warm. | High | Options for O_DIRECT, drop caches per epoch, recommended mount options (`actimeo`, `lookupcache`), and a check that dataset ≫ aggregate client RAM. Record server-side op counts to validate. |
| **R4** | **Abstract language under- or over-designed** (§4.4). | Can't express the target workloads, or the PoC turns into a compiler project. | Medium–High | Write the 3–4 target abstracts (training small-file, training large-sample, checkpoint write burst, checkpoint restore) *by hand, on paper* before writing the parser. |
| **R5** | **Determinism is broken by accident.** Shared RNG, dynamic work distribution, `HashMap` iteration order, thread-count-dependent sharding. | "Same seed, same workload" stops being true, and runs are no longer comparable. | Medium | Counter-based RNG; static sharding; a `--dry-run` mode that prints each actor's op stream so two runs can be diffed (and golden-tested in CI). |
| **R6** | **Dataset generation cost.** 50M creates at 5–50K creates/s = **17 min – 2.8 h**; directory fan-out affects both creation and lookup performance. | Slow iteration; layout mismatch between generator and abstract. | High (certain to be slow) | `datagen` mode in the same binary, driven by the *same* filename pattern and size distribution as the abstract. Hierarchical layout (≤~10K entries/dir). Resumable. |
| **R7** | **Write data is dedupable/compressible** (shared buffer reused for every write). | Inflated write and checkpoint numbers on systems with data reduction. | Medium | Per-write unique content: fast PRNG fill (≥10 GB/s/core with SIMD-friendly generators) or a large random pool with unique per-block headers. Same rule for `datagen`. |
| **R8** | **io_uring disabled by the OS.** Containers are out of scope (runner runs on bare Linux), so the remaining exposure is the `kernel.io_uring_disabled` sysctl (6.6+) set by some hardened/enterprise distros **[verify target OS]**, and `RLIMIT_MEMLOCK` on kernels <5.12. | Runner won't start on some lab hosts. | Low | Startup probe with a clear error message naming the sysctl; blocking-thread-pool backend as fallback. |
| **R9** | **Measurement volume.** Per-op logging = GBs per epoch. | Measurement perturbs the run or fills disks. | Medium | HDR histograms per thread/op type + 1 s time series; merge at end. Optional sampled tracing. |
| **R10** | **Emulating PyTorch too literally or not literally enough** (Python `open()` side syscalls, in-order batch delivery, worker sequential reads). | Results don't match real training runs. | Medium | Validate against real `strace`/NFS server op-mix captures of a small real training run; compare op mix and per-file latency distribution. |
| **R11** | **Long runtimes at realistic scale** (6 h/epoch at 100 GPUs on 50M files). | Painful testing; tempting to cut corners on fidelity. | Certain | Step/time-bounded runs; "accelerator utilization ≥ X%" as the pass metric (MLPerf Storage style), which converges long before an epoch ends. |
| **R12** | **Dev environment is WSL2.** io_uring works on WSL2's 6.x kernel, but there's no realistic NFS target, and `/mnt/c` (9p/drvfs) behaves nothing like production. | Wrong conclusions from local testing. | Medium | Develop logic locally against ext4/tmpfs; run all performance work on real Linux clients against real NFS. |
| **R13** | ~~liburing vs. Rust crate~~ **Decided:** use the pure-Rust `io-uring` crate. | — | — | — |

---

## 6. Spikes to run before committing to the design

1. **io_uring vs. blocking thread pool on the real NFS target** (addresses R1, R2, R8).
   Microbenchmark `open → read(S) → close` on random files at QD 1…4096, S ∈ {4K, 128K, 1M},
   buffered and O_DIRECT. Measure ops/s, client CPU per op, and how many io-wq workers are
   spawned (`ps -eLf | grep iou-wrk`). This decides whether io_uring is the main lever or
   just a convenience.
2. **Feistel permutation**: correctness (a bijection over [0, N) for N = 50M/100M, checked
   with a bitmap), throughput (target <50 ns/draw), and a quick check that there's no locality
   (distribution of |Δ| between consecutive draws).
3. **Client cache behavior at scale**: `stat`/`open` 50M files from one host, watching slab
   growth (`slabtop`) and the NFS op mix (`nfsstat -c`, `mountstats`).
4. **Paper abstracts** for the 3–4 target workloads (R4), then decide the grammar.

---

## 7. Bottom line

- **DRAM is a non-issue for the runner** if the consumer is a keyed permutation and filenames
  and sizes are computed, not stored: under 1 GB even at 100M files and ~50K simulated GPUs.
  The 200/400 MB shuffled-list approach is also perfectly viable. The bitmap's slowdown is
  real (N·ln N total probes, near-N probes for the last draws) but it's a tail-latency problem,
  not a throughput one.
- **Client-side DRAM (kernel caches) is an issue**, mostly as a fidelity problem.
- **IOPS ceiling is set by the NFS client path and io-wq punting, not by the runner's state
  machine.** Plan for many client hosts and validate io_uring's benefit on NFS early.
- **The biggest design risk is the abstract language.** It needs binding, fork/join,
  prefetch, and barriers beyond the regex-style sketch.

---

## 8. Round 2: user decisions and follow-up analysis (2026-09-24)

User inputs recorded:
- **Multi-host via MPI.** All ranks use the same seed, and each rank takes its own subset of the
  work with no negotiation over the network.
- **Run length.** Runs are bounded by a step count (currently 500, adjustable).
- **Dataset sizing.** The dataset must be ≥ 5× the total DRAM of all client nodes.
- **Why the page cache matters.** Finding pages to evict gets non-linearly slower as memory
  fills, even when the pages are clean. This is the main motivation for io_uring and/or O_DIRECT.
- **Reproducibility.** Exact run-to-run reproducibility is required; the across-run randomness of
  real training is not.
- **Deployment and tooling.** Bare Linux, no containers. Use the `io-uring` crate.
- **Grammar.** Extensions are agreed; see `GRAMMAR_OPTIONS.md`.
- **Backends.** A/B testing of I/O backends is agreed; see §8.5.

### 8.A MPI + same seed + rank-based sharding — yes, with two refinements

1. **Shard by global GPU id, not by MPI rank.** Rank r owns GPU ids `[r·G/R, (r+1)·G/R)`.
   GPU g consumes permutation positions `g, g+G, g+2G, …`, so the work per GPU does not depend on
   R. A 1,000-GPU run on 10 nodes and on 20 nodes issues exactly the same operations. The only
   change is which node sends them.
2. **Apply the modulo to positions in the permutation, not to file ids.** With `file_id % R`, each
   node owns a fixed slice of the namespace, and that slice is the same in every epoch.
   `perm_{seed,epoch}(g + k·G)` gives each GPU a random subset instead, and a different one per
   epoch, which is closer to `DistributedSampler`. The node-to-node communication cost is still
   zero.

**Integrating the barrier with io_uring:** a dedicated MPI thread (`MPI_THREAD_FUNNELED`)
does the blocking `MPI_Barrier`. When it returns, the thread writes to an `eventfd`. Each
event-loop thread keeps an io_uring read posted on that eventfd, so the release shows up as an
ordinary completion.

Within a node, the threads first gather through an atomic counter. The last thread to arrive
asks the MPI thread to enter the barrier. At one barrier per ~150 s the cost does not matter.

**Build note:** the `mpi` crate (rsmpi) needs an MPI installation (Open MPI or MPICH) and
libclang at build time.

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
- **io_uring + O_DIRECT on NFS may still go through io-wq.** Whether NFS direct I/O supports
  non-blocking submission is exactly what Spike 1 must measure.
- **Fidelity.** Real PyTorch uses buffered I/O. Accepting O_DIRECT is a benchmark-policy
  decision; with the explicit readahead-size emulation above, the load the server sees can be
  kept representative.

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

### 8.D Is the workload exactly reproducible? Yes: per-GPU op streams are identical; timing and interleaving are not.

With these rules, the **exact multiset of operations and each GPU's exact op order** are
identical from run to run:
- a counter-based RNG keyed on `(seed, gpu_id, epoch, step, site, draw)`;
- Feistel permutation positions sharded by global GPU id;
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

**Workload fingerprint:**
- Each actor keeps a rolling hash (xxh3) of `(op, file_id, offset, len)` in issue order.
- These are combined in actor-id order through `MPI_Reduce` and printed with the results.
- `--dry-run` computes the fingerprint without doing any I/O, which gives a golden value for CI.
- Two runs with the same fingerprint executed the same workload. This also proves that A/B
  backend comparisons (§8.5) are like-for-like.

### 8.5 A/B test matrix (all rows run the same abstract, seed, and G, so the fingerprints are equal)

| Dimension | Variants |
|---|---|
| Submission backend | io_uring · blocking thread pool (pread/pwrite) |
| Cache mode | buffered · O_DIRECT · buffered + FADV_DONTNEED · RWF_DONTCACHE (if supported) |
| io_uring features | fixed files (direct descriptors) on/off · fixed buffers on/off · `SINGLE_ISSUER`+`DEFER_TASKRUN` · SQPOLL · io-wq max workers · op linking on/off |
| NFS client | nconnect 1/4/16 · rsize 256K/1M · TCP vs RDMA · v3 / v4.1 / v4.2 / pNFS |

Measured for each run:
- ops/s and bytes/s;
- **client CPU per op** (the number that decides how many client hosts are needed);
- peak `iou-wrk` thread count;
- accelerator utilization (AU%);
- per-op latency histograms;
- `/proc/self/mountstats` RPC counts.

### 8.E Deployment

Bare Linux only, so R8 drops to *Low*: only the `kernel.io_uring_disabled` sysctl and memlock
limits on old kernels are left to check at startup.
