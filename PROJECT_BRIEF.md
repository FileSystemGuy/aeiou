# Project Brief — Abstract-Driven I/O Benchmark Runner

Captured 2026-09-24 from the design conversation. It records the original requirements, the
decisions made so far, and what is still open. Details are in:
- `NAPKIN_MATH.md`: DRAM and IOPS estimates, risk register, spikes, and the round-2 decisions (§8)
- `GRAMMAR_OPTIONS.md`: options for extending the abstract language, with a recommendation

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

Agreed additions (see `GRAMMAR_OPTIONS.md`): binding, per-GPU replication, fork/join, a loader
with prefetch and in-order delivery, `take`, `barrier(scope)`, `every N`, and named parameters
that can be overridden from the CLI.

## 4. Requirements

- A Rust **benchmark runner**, highly parallel, using io_uring through the **`io-uring` crate**
  (decided; not C liburing).
- Inputs: an abstract file, plus CLI arguments `--seed`, `--gpus`, and per-run parameters
  (batch size, readahead/transfer size, workers, prefetch, compute time, steps, …).
- It must handle corpora of **50M–100M files** without one data structure per file. Filenames
  follow patterns (e.g. `data_NNNN_of_MMMM`, or hierarchical) and are computed, not stored.
- Tracking "consumed" files: the user asked about a bitmap vs. a fully populated shuffled list.
  The **recommendation is a keyed Feistel permutation**, which needs no memory and has O(1),
  constant-cost draws.
- The target storage is **NFS** (the user is at Hammerspace; pNFS/FlexFiles is relevant).

## 5. Decisions so far

| Topic | Decision |
|---|---|
| Multi-host | MPI. Same seed on every rank; each rank takes its own subset with no negotiation. Work is split **by global GPU id** over **positions in the Feistel shuffle**. |
| Run length | Step-bounded, currently 500 steps (adjustable). `steps` and `sync_every` are separate parameters. |
| Dataset sizing rule | Dataset capacity ≥ **5× the total DRAM of all client nodes**. |
| Page cache | The main concern: Linux gets non-linearly slower at finding pages to evict as memory fills, even with clean pages. This motivates O_DIRECT / io_uring / DONTCACHE. |
| Reproducibility | **Exact** run-to-run reproducibility of each GPU's op stream is required; training-style randomness across runs is not needed. Plan: counter-based RNG, static sharding, and a workload fingerprint (`--dry-run` computes it). |
| Deployment | Bare Linux on the client nodes, **no containers**. |
| I/O crate | `io-uring` (Rust). |
| A/B testing | Agreed. Backend × cache mode × io_uring features × NFS mount options (`NAPKIN_MATH.md` §8.5). The key metric is client CPU per op. |
| Grammar | Options written up. Recommended: extended regex-style DSL, with a serde AST (YAML/JSON) as the canonical form. **User has not yet chosen.** |

## 6. Open items / next steps

1. **User to choose a grammar option** (A/B/C in `GRAMMAR_OPTIONS.md`).
2. **Spike 1:** io_uring vs. a blocking thread pool, buffered vs. O_DIRECT, for
   `open → read → close` against the real NFS target. It decides the I/O backend and the cache
   strategy.
3. Spike 2: correctness and speed of the Feistel permutation at 50M/100M.
4. Spike 3: client dentry/inode slab growth and NFS op mix when touching 50M files.
5. Write paper abstracts for the target workloads: small-file training, large-sample training,
   checkpoint write, checkpoint restore.
6. Check `RWF_DONTCACHE` support in the NFS client on the target kernels.
7. Build `datagen` mode, driven by the same filename pattern and size distribution as the
   abstract, writing non-dedupable data.

## 7. Environment

The development machine is WSL2 (kernel 6.18, 20 cores, 31 GB RAM), with no realistic NFS
target. All performance work must run on real Linux client nodes.
