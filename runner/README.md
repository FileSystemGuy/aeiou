# `aeiou`: the runner

Layer 3 of `GRAMMAR_OPTIONS.md` Option D: the Rust program that loads the AST contract
(`schema/`), validates it, and executes it. This directory is a Cargo workspace; the crate
`aeiou` builds the `aeiou` binary (`PROJECT_BRIEF.md` §8 naming). Nothing here depends on
Python.

```
cd runner
cargo build --release
cargo test --release
./target/release/aeiou check ../schema/examples/*.ast.json
./target/release/aeiou dry-run ../schema/examples/train_small_files.ast.json --gpus 8 --seed 1
./target/release/aeiou dry-run ../schema/examples/kv_cache_serving.ast.json --gpus 1 \
    --param concurrency=2 --param requests=20 --gpu 0 --steps 1..2 --limit 50
./target/release/aeiou datagen ../schema/examples/train_small_files.ast.json --root /mnt/sut --param files=4000
./target/release/aeiou run ../schema/examples/train_small_files.ast.json --root /mnt/sut --gpus 8 --seed 1 \
    --param files=4000 --param steps=50 --io-backend sync
./target/release/aeiou dry-run ../schema/examples/model_load.ast.json --gpus 8 \
    --params ../schema/examples/params/model_load.synthetic.params.json      # a parameter file (§4)
```

## 1. What exists (2026-09-30)

| Command | What it does |
|---|---|
| `aeiou check FILES…` | Loads each AST, validates it (structure plus rules V1–V13), prints its canonical SHA-256 and op-kind counts in the same format as `schema/check.py`. CI diffs the two outputs. |
| `aeiou dry-run AST --gpus G [--seed S] [--params FILE]… [--param k=v]…` | Walks every actor instance without I/O: op counts by kind and phase, bytes read and written, emulated compute, barriers, and the **workload fingerprint**. `--ranks R` adds bytes per host against this host's DRAM. `--gpu g [--steps a..b] [--limit n]` prints one instance's op stream. |
| `aeiou datagen AST --root DIR [--params FILE]… [--param k=v]… [--dedupe D] [--compress C] [--threads N] [--dataset NAME]…` | Writes every `files` and `regions` dataset the abstract declares under `DIR`, names, sizes, and chunks from the definition and the dataset seed, content per §5, in parallel by id, then the manifest `.aeiou-dataset.json` at each dataset root. Refuses a non-empty root (datasets are read-only, V12). Prints each dataset's id. |
| `aeiou run AST --gpus G --root DIR [--seed S] [--params FILE]… [--param k=v]… [--io-backend sync\|sync-direct\|io_uring\|io_uring-direct\|posix-aio\|posix-aio-direct\|libaio\|libaio-direct\|mmap] [--threads N] [--aio-depth N] [--mmap-mode fault\|populate\|willneed] [--mmap-consume touch\|copy] [--iowq-max-workers N] [--sqpoll IDLE_MS [--sqpoll-shared]] [--defer-taskrun] [--coop-taskrun] [--time-scale X] [--buffer-mib N] [--write-compress C] [--clean-namespaces] [--expect-fingerprint HEX] [--expect-dataset-id SHA]… [--ranks R --rank r --coordinator HOST:PORT] [--rank-rotate k] [--max-gap SECS] [--require-cold] [--drop-caches]` | Executes the abstract against `DIR` on one host, or on several with the coordinator (§4, §6): checks every dataset against its manifest and every input namespace against the manifest of the run that wrote it, requires empty output namespace roots, runs one OS thread per actor and sub-actor with blocking POSIX calls (`sync`) or multiplexes them over one `io_uring` per event-loop thread (`io_uring`, §8), checks every result structurally, prints latency histograms, per-phase totals, per-step stall and busy fraction, and the fingerprint, and leaves `.aeiou-namespace.json` at every namespace root it wrote. |
| `aeiou-launch [-p PORT] HOST… -- aeiou run ARGS…` | Starts rank *i* on the *i*-th host over ssh with `--ranks`, `--rank`, and `--coordinator HOST0:PORT` appended (§6). |

Not yet: the other asynchronous backends (~~`libaio`, `posix-aio`, `mmap`,~~ **built
2026-10-01**, §9; `gds`, `nixl-posix`, `libnfs` not; ~~`io_uring`~~
**built 2026-10-01**, §8), ~~the per-backend counters, `mountstats`~~ (the host counters,
**built 2026-10-01**, §4: task and io-wq worker peaks, CPU, RSS, the mount's NFS RPCs;
backend-specific counters beyond those come with each backend), `--drop-caches` at the
start gate with the residency check and the mount options line (decided 2026-10-01,
`PROJECT_BRIEF.md` §6 item 16), `RLIMIT` startup checks, a JSON report, `--metrics` (`PROJECT_BRIEF.md` §6 item 14), and the `replay` node. ~~`stream`
access, container layouts beyond `samples_per_file`~~ (contract 0.2, 2026-09-30: `eval.rs`
computes every offset of a framed container from `format.layout`, `consume` under `stream`
shuffles shards, `fadvise` is the eighteenth op; `tests/layout.rs`). Datagen for format
classes is the Python side (`aeiou-datagen`, `builder/README.md`); `aeiou datagen` refuses a
dataset that has a format class.

```
aeiou/src/
  ast.rs       the contract as serde types (externally tagged, deny_unknown_fields)
  canon.rs     canonical form and SHA-256, byte-identical to check.py::canonical
  validate.rs  rules V1–V13 and the schema's structural constraints, ported from check.py
  sites.rs     draw sites: the JSON pointer of every draw / pick / choose, hashed
  pattern.rs   path patterns: {id div 1300:05}, {conv:016x}, {name}
  rng.rs       positional keys, SplitMix64 words, the 4-round Feistel permutation
  eval.rs      the resolved model: parameters, datasets (sizes, layouts, names), namespaces, distributions
  vm.rs        expression evaluation and the walk as a resumable state machine: `next` yields the
               next op, control, or fork event and the VM can be parked between any two; `drive`
               feeds a Sink, with the fork protocol a Sink uses to run `parallel` / `loader` sub-actors
  dryrun.rs    the dry-run Sink, parallel over actor instances, and the report
  backend.rs   the backend kinds, the Backend trait (blocking form), `sync` / `sync-direct`
  run.rs       `aeiou run`: per-actor state shared by both drivers (files, structural checks,
               recording, namespace bookkeeping), the thread-per-actor Sink, startup checks, report
  uring.rs     the `io_uring` backends: one event loop per thread, a parked VM per task, the buffer
               pool, loop-local channels, barriers through the coordinator's eventfd (§8)
  aio.rs       the `libaio` backends: the kernel AIO context as a second engine of that loop (§9)
  coord.rs     the Coordinator trait, the in-process implementation, the TCP client and the server for several hosts
  cold.rs      the cold start before the gate: `--drop-caches`, the kernel's cache sizes around it, the
               `mincore` residency sample of every dataset, the `--require-cold` refusal
  counters.rs  the host counters around a run: task and io-wq worker peaks sampled from `/proc`,
               `getrusage`, and the `mountstats` delta of the mount `--root` is on (§4)
  payload.rs   positional content (dgen-data behind the `aeiou-positional/1` wrapper), the manifest
  datagen.rs   `aeiou datagen`
  main.rs      the CLI
aeiou/tests/golden.rs   hash parity with check.py, golden fingerprints, semantics tests, parameter files
aeiou/tests/layout.rs   contract 0.2: framed layouts by hand, unit/column handles, stream consume, fadvise
aeiou/tests/run.rs      datagen + run round trips on a temporary directory, refusals, loader order;
                        the `io_uring` backends over the same abstracts, on one loop and on several;
                        `posix-aio`, `libaio`, and `mmap` over the same (§9)
aeiou/tests/uring_knobs.rs  the ring and io-wq knobs (§8): same fingerprint, the io-wq cap holds, the SQPOLL threads are counted
aeiou/tests/coord.rs    barriers across hosts, the configuration check, two-rank runs as threads and as processes
aeiou-launch            the ssh loop: one rank per host
```

## 2. Definitions the runner fixes

`schema/README.md` §5 states the positional semantics and leaves the exact hash functions to
the runner. These are the runner's choices; changing any of them changes every fingerprint, so
each is a recorded decision and the golden tests pin them.

- **Key of a positional draw.** `xxh3_64(actor ‖ site ‖ i₁ ‖ … ‖ iₙ, seed = --seed)` over
  little-endian 64-bit words, where `site = xxh3_64(JSON pointer of the node)` and `i₁…iₙ`
  are the enclosing loop, `parallel`, and `loader` indices, outermost first. A `choose` uses
  the statement's pointer; a `pick` uses the handle's.
- **Words of a key.** The SplitMix64 sequence started at the key. A draw takes as many words
  as it needs, in order (a mixture arm then its value; two for Box–Muller).
- **Distributions.** `uniform`: `lo + word mod (hi − lo)`. `uniform64`: the word. `normal`
  and `lognormal`: Box–Muller, clamped to `[min, max]`, rounded to the nearest integer (V10).
  `empirical`: cumulative weights. `zipf {s}` over N ids: the inverse CDF of the continuous
  envelope `r^-s` on `[1, N+1)` gives a rank in O(1) with no table; `hotset`: a rank in the
  first `ceil(fraction·N)` with probability `weight`, else in the rest. Ranks map to ids
  through `Perm(N, key(dataset seed, "rank"))`, so the popular ids are fixed by the dataset
  seed, not by the run seed. A `uniform` under `pick` is an id directly.
- **Dataset draws** (sizes, region sizes) use `xxh3_64(id, seed = dataset seed)`; no site.
- **Permutation.** `Perm(N, key)`: a 4-round Feistel network on the smallest power of two
  ≥ N (unbalanced halves when the bit count is odd), round function SplitMix64's mixer over
  `(key, round, half)`, cycle-walked until the image is below N. Tested as a bijection at
  N = 50M.
- **`consume` position.** With the nearest enclosing `loader` (else the innermost loop) as
  the batch frame: `b` is the row-major ordinal of the frames down to and including it, `j`
  and `B` the ordinal and the product of the iteration counts of the frames inside it. An
  epoch is `N div (G·B)` batches (`drop_last`); the sample is
  `Perm(N, key(seed, "consume", dataset, epoch))(g + G·((b mod epoch_len)·B + j))`. For the
  `loader` sugar this is exactly `g + G·(b·B + j)` of `NAPKIN_MATH.md` §8.A.
- **`x @ i`.** The binding's definition is re-evaluated with its loop's frame set to `i`;
  sibling `let`s of the same body are re-evaluated at that index and memoized for the
  duration; draws inside use the shifted index in their key, so `conv @ (r − d)` is the value
  request `r − d` computed. Below the loop's `from`, the enclosing `cond` expression takes
  its other arm (the "fresh draw" of `ABSTRACTS.md` §9.5).
- **Container layouts** (0.2). `size` and `offset` of a unit or column handle, `units`, and
  `unit_index` follow the formulas of `schema/README.md` §2 (*Container layout*) exactly, in
  integer arithmetic except a column's share `floor(size × weight)` in IEEE doubles, which
  Python's writer computes identically. The unit lengths of a file are computed once per
  file and held in a bounded per-actor cache (`GRAMMAR_OPTIONS.md` §6.3). Under `stream`,
  `consume` and `pick` draw file ids from `Perm(files, …)` with the same position formula;
  `offset(s)` of a sample in a one-column container is the start of its payload.
- **File positions.** `open` sets the position to 0 (`APPEND`: the known size); a `read` or
  `write` without `offset` uses and advances it; `lseek` moves it. A read's expected count is
  `clamp(size − offset, 0, len)` when the size is known. `until_eof` issues reads of `len`
  until the known size is exhausted, then one more (returning 0). An `as_written` object's
  size is the sum of the write lengths this actor issued to it through the same path at this
  position (rule V4 guarantees a reader that uses `until_eof` on one is the writer).
- **`readdir`** is one op in the stream; how many `getdents64` calls it takes is a backend
  matter, like the RPC count of a read.
- **Fingerprint.** Sum modulo 2⁶⁴ over ops of `xxh3_64(kind ‖ actor ‖ n ‖ i₁…iₙ ‖ offset ‖
  len ‖ aux ‖ path [‖ 0 ‖ path₂])`, where `offset` is the effective offset of a read or write,
  `len` the requested length, `aux` the open flags and mode, `lseek` whence, `ioctl` request,
  or `mkdir` mode, and `path₂` is `rename`'s destination. `phase`, `expect`, results, and
  issue order are not hashed. Two runs with the same fingerprint executed the same multiset
  of positioned ops.

## 3. Costs

Dry-run walks an actor instance at 150–350 ns per op on one core (one thread per instance,
so `--gpus 8` uses eight; re-measured 2026-10-01 after the walk became a resumable state
machine, §8: `vdb_build_diskann` 203M ops in 36 s against 32 s before, +12 %, the explicit
stack and the pending op slot standing in for the recursion's stack frames). The committed abstracts at their defaults: training shapes in
under a second; `kv_cache_serving` 72M ops per instance in 24 s; `vdb_search_diskann` 68M
ops in 28 s; `vdb_build_diskann` 203M ops in 33 s. Memory is a few MB: no per-file
structure exists, and an actor's state is its frames, bindings, open files, and the as-written
sums of the objects it created. The obvious speed-up, caching a bound handle's resolved path
instead of re-formatting the pattern on every op, is not done yet.

## 4. `aeiou run` (2026-09-30)

What runs where, and what is checked. The design reasoning is `DESIGN_REVIEW.md` §3.23.

- **Threads.** One OS thread per actor instance. A `loader` spawns `workers` threads that
  live until the actor ends; a `parallel` ~~spawns `width` threads and joins them before the
  node returns~~ runs its `width` sub-actors on threads the forking actor keeps (the
  sub-actor pool, 2026-10-01, `DESIGN_REVIEW.md` §3.36) and returns when all have ended;
  either may nest. Sub-actor `k` of a fork always runs on pool thread `k`, the pool grows to
  the widest fork its actor has issued, and its threads idle between forks and end with the
  actor. A pool thread keeps its two buffer rings and, when its sub-actors fork in turn, a
  pool of its own; the backend and the file table are new for every sub-actor. The report's
  `threads` is therefore the number of threads the run created, not the number of
  sub-actors it ran. Every sub-actor starts from a snapshot of its parent's
  position and bindings (`vm::Snapshot`) and sees the files the parent had open at the fork;
  what it opens itself is its own. The VM's walk runs on the thread, so blocking calls
  are simply blocking: this is the `sync` fidelity reference of `PROJECT_BRIEF.md` §5~~, and
  the asynchronous backends will need a resumable VM instead~~. The VM is resumable since
  2026-10-01 (`vm.rs`: `next` yields one event at a time), and the `io_uring` backends park
  it on an event loop instead of a thread (§8).
- **Loader.** An ordered channel named after the loader with `workers × prefetch` slots
  bounds batches *started*: worker `w` builds batches `w, w + W, …` and may start batch `b`
  only once fewer than `workers × prefetch` batches are started-but-untaken, which is
  PyTorch's index dispatch. `take` blocks until the next batch in order is complete. A `take`
  after the last batch, or an actor that ends with batches untaken, is an error: producers
  are finite and the multiset is fixed (`NAPKIN_MATH.md` §4.1).
- **`channel` / `put` / `take`** are the general form: `put` blocks while `capacity` items are
  delivered and untaken; `take` returns the next in `seq` order, or any item if unordered.
- **Barrier.** The participants of `barrier {scope}` are the instances of every template
  whose body contains it outside any `parallel` or `loader` (a barrier inside a sub-actor is
  refused). An instance that finishes leaves its barriers; a generation completed by
  departures is released and reported as a warning, since it means the instances did not all
  hit the barrier the same number of times. Across hosts the barrier is two-level (§6),
  behind the same `Coordinator` trait.
- **`compute`** sleeps for `ns × --time-scale` and is recorded unscaled. `--time-scale 0`
  runs the I/O back to back.
- **Buffers.** Each thread has a 4 KiB-aligned read ring and a write ring of `--buffer-mib`
  (default 8), allocated on first use and grown to the largest op, so successive copies land
  in successive slices and the aggregate across threads exceeds L3 (`NAPKIN_MATH.md` §2.2).
- **Writes** carry the positional content of §5 for the object at that offset
  (`--write-compress` sets the ratio; no dedupe control on writes yet).
- **`sync-direct`** (and `io_uring-direct`, §8) adds `O_DIRECT` to every regular-file open. An unaligned read (the
  `until_eof` idiom's last read starts at EOF, which is rarely aligned) is rounded out to
  4 KiB and the requested part is counted, as an `O_DIRECT` shim under a buffered application
  has to do; an unaligned write is refused.
- **Structural checks, every op.** A read must return the computed count; a write its
  length; `readdir` the computed entry count for a one-sample-per-file, unchunked dataset
  directory (`.aeiou*` entries not counted); a failing op's errno must be in the statement's
  `expect` list. Anything else aborts the run with the actor, position, and op. The runner
  never looks at the bytes (`PROJECT_BRIEF.md` §5, *Data verification*).
- **Parameter files** (2026-09-30, `schema/README.md` §8). `--params FILE` applies a
  `.params.json` set over the defaults; several apply in order and `--param` applies last.
  The file must name this abstract (and this AST's hash, if it names one); every value must
  be a declared parameter of the same kind as its default (`--param` is held to the kind too).
  The header line prints each file with its SHA-256, and the datagen manifest records them as
  provenance; the identity of a run is still the resolved values.
- **Startup.** Every dataset's manifest must match the resolved definition (§5); the ids are
  printed and may be pinned with `--expect-dataset-id`. Namespace roots are created and must
  be empty; `--clean-namespaces` empties them, leaving a dataset root that lies inside one
  alone.
- **Ranks.** `--rank r --ranks R` give this host its GPU id range, contiguous blocks of
  `ceil(G / R)`; `--rank-rotate k` makes it run rank `(r + k) mod R`'s range instead. Nothing
  in the op stream depends on the host, so rotating a read run against the write run's host
  list makes every host read what another host wrote (`DESIGN_REVIEW.md` §3.24). `--ranks`
  above 1 needs `--coordinator` (§6).
- **Input namespaces** (`input: true`, V14). The run requires `.aeiou-namespace.json` at the
  root (`schema/README.md` §6), refuses a differing definition, never empties the root,
  prints who wrote it and how long ago (`--max-gap` makes a longer gap an error), and counts
  the opens of recorded input objects that hit the host that wrote them; one or more is a
  warning, or an error under `--require-cold`. At the end of a successful run the manifest
  is written for every root this run created objects in (open with `CREAT`, `mkdir`, the
  destination of a `rename`; unlinked paths and rename sources dropped).
- **Measurement.** Per-op latency histograms by kind (four buckets per octave; mean, p50,
  p99, max), per-phase ops, bytes, and I/O time, barrier count and wait, and per `take` the
  stall (time blocked) and the compute issued before the next take, kept per instance in
  take order and reported in ten step buckets with the busy fraction
  `compute / (compute + stall)`, so steady state is selected after the run. The fingerprint is
  summed over the ops actually issued; `--expect-fingerprint` (from `dry-run`) makes a
  mismatch an error.
- **Cold start** (`cold.rs`, added 2026-10-01, `DESIGN_REVIEW.md` §3.31). Between the
  startup checks (and rank 0's namespace preparation) and the start gate, every host does
  two things, both outside `elapsed` by construction and both in the report
  (`Report::cold`, one entry per host):
  - with `--drop-caches`: `sync`, then `3` into `/proc/sys/vm/drop_caches`. The file is
    opened before the `sync`, so a host without root (or `CAP_SYS_ADMIN`) refuses at once,
    and through the coordinator that stops every host before the gate. The report gives
    the two durations and ~~`Cached`~~ the file cache (`Cached − Shmem` of `/proc/meminfo`
    since 2026-10-01: `Cached` counts tmpfs, which a drop cannot free), the dentry count,
    and the inode count before and after. The flag is part of the configuration the coordinator compares: a run is cold on
    every host or on none. Never inside a run.
  - with `--drop-caches` or `--require-cold` (~~always~~; opt-in since later the same day,
    `DESIGN_REVIEW.md` §3.31: a plain run does not pay the sample's opens), the residency
    sample: for every dataset, at most 256 files at evenly spaced
    file ids (`⌊k·files/256⌋`, the same on every host; chunk `id mod chunks` of a chunked
    file), each mapped and passed to `mincore`: whole up to 256 MiB, 64 evenly spaced
    4 MiB windows beyond. No data is read and no root is needed. The line is
    `256 of 16000 files sampled, 280 of 7959 pages resident (3.5 %)`; `--require-cold`
    refuses when any page is resident, as it already did for input namespaces written on
    the reading host, and `--drop-caches` alone warns (pages that survived the drop). The sample is a floor, not a
    proof: it sees data pages of the sampled files only, and its own opens leave those few
    hundred files' dentries and attributes (on NFSv4 possibly delegations) on the client.
  A tmpfs is its own page cache: nothing drops and every page is resident, which the check
  reports truthfully. What a drop cannot reach (`fscache`, the NFSv4 client state) needs a
  remount, which is the launcher's job, not the runner's.
- **Host counters** (`counters.rs`, added 2026-10-01): what the client did meanwhile, as
  distinct from what the abstract did, printed after the totals and carried in the report
  (`Report::counters`, summed over hosts by the coordinator). A sampler thread reads
  `/proc/self/status` every 10 ms for the peak task count of the process and, at every
  change, counts the `iou-wrk-*` threads in `/proc/self/task` for the io-wq worker peak
  (and the `iou-sqp-*` threads, printed as `sqpoll threads peak` when there are any)
  (workers linger idle for seconds, so the peak is not missed); `getrusage` before and after
  gives user and system CPU, the peak RSS, and (since later on 2026-10-01, §9) the minor and major page faults; `/proc/self/mountstats` before and after gives
  the mount `--root` is on (longest mount point that is a prefix of the canonical root: its
  device and type on any filesystem, and its options as a `mount opts` line: the `opts:`
  line of `mountstats` on NFS, with `vers`, `rsize`, `acregmin`…, and `lookupcache` and
  `nconnect` when set, the options field of `/proc/self/mounts` elsewhere; and, since
  2026-10-01 (§9), a `mount read_ahead_kb` line: the readahead window of the mount's
  backing device info, `/sys/class/bdi/MAJOR:MINOR/read_ahead_kb` for the device `--root`
  is on, a partition's being its disk's; absent on tmpfs, which has none) and, on NFS, the deltas of the client's byte counters
  and of the per-procedure RPC statistics, printed as counts with the mean round trip, and
  transmissions, timeouts, and errors when they differ from the count. Two caveats on the
  line itself: `mountstats` is per mount, not per process, so every process on the host
  using that mount is in the delta; and the NFS client counts buffered bytes as returned but
  `O_DIRECT` bytes as requested (an aligned 1 MiB read of a 120 KiB file counts 1 MiB).
  Output:

  ```
  host: tasks peak 26  io-wq workers peak 20  cpu user 0.039 s sys 0.509 s  maxrss 15.67 MiB  faults minor 2589 major 0
  mount /mnt/aeiou-nfs (nfs4, localhost:/srv/aeiou-export): server read 191.40 MiB wrote 0 B; buffered read 191.40 MiB wrote 0 B; O_DIRECT requested read 0 B wrote 0 B
  rpcs 3242: READ=1600 (399µs rtt)  OPEN=1222 (385µs rtt)  OPEN_NOATTR=378 (349µs rtt)  GETATTR=15 (67µs rtt)  READDIR=14 (500µs rtt)  ACCESS=8 (0ns rtt)  LOOKUP=5 (200µs rtt)
    (the mount's counters over the run, every process on this host included)
  ```

Observed on the WSL2 ext4 disk (2026-09-30, `runner/aeiou/tests/run.rs` and the smoke runs):
every committed abstract runs to the dry-run fingerprint under `sync` (`ckpt_restore`
against the namespace a `ckpt_write_dcp` run left, through its manifest), and
`train_small_files` also under `sync-direct`; a page-cache read of 1 MiB costs
10 µs, an `O_DIRECT` one 258 µs; `vdb_search_diskann` ~~spawns a thread per beam per hop
(923 threads for 924 reads), the cost of forking `parallel` afresh each time, and a per-actor
sub-actor pool is the planned fix~~ spawned a thread per beam per hop (923 threads for 924
reads) until the sub-actor pool (2026-10-01). With it, on the same disk (`nodes=200000
threads=8 queries=3000`, one instance, 507,608 `O_DIRECT` reads of 4 KiB, the same
fingerprint before and after):

| | threads created | elapsed | read mean | cpu user | cpu sys | minor faults | maxrss |
|---|---|---|---|---|---|---|---|
| `sync`, a thread per sub-actor | 507,617 | 41.5 s | 611 µs | 19.5 s | 135 s | 1,017,433 | 7.8 MiB |
| `sync`, pool | 41 | 6.3 s | 197 µs | 13.3 s | 36.2 s | 66,559 | 264 MiB |
| `mmap`, a thread per sub-actor | 507,617 | 38.0 s | 420 µs | 16.4 s | 130 s | 1,019,792 | 21 MiB |
| `mmap`, pool | 41 | 12.0 s | 198 µs | 10.0 s | 48.1 s | 508,669 | 31 MiB |

The resident set under `sync` is the buffer rings: a thread that lived for one read only
ever touched the first slice of its ring, and a pool thread walks the whole of it, as
`NAPKIN_MATH.md` §2.2 intends. Under `mmap` every beam sub-actor still maps the index
file for its one read (a sub-actor inherits descriptors, not mappings), which is the
remaining half-million faults and the reason `mmap` is the slower row here; open in
`DESIGN_REVIEW.md` §3.36. The same runs on the loopback NFS mount are §7.

## 5. `aeiou datagen`, the payload, and the manifest

- **Payload** (`payload.rs`). Content is cut into 1 MiB blocks; block `b` of unit `u` under
  seed `s` is the prefix of a 1 MiB `dgen-data` 0.3.0 stream seeded
  `labeled_key(s, "payload", [u, b])`, with dgen's compression layout (the last `(C−1)/C` of
  each block zero-filled for ratio `C`) and no dgen dedupe. The prefix of a block is
  independent of how it is read out, so the verifier can regenerate any 4 KiB piece with
  dgen-py's public API (`DESIGN_REVIEW.md` §3.17). Dedupe is the wrapper's, by seed reuse
  (`aeiou-positional/1`): a `files` dataset uses `u = id mod ceil(files / D)` and `b` the
  block of the logical offset (a chunked file is the logical file cut at `chunk`), so file
  `id` and file `id + files/D` carry the same bytes; a `regions` dataset (one file) uses
  `u = 0` and `b mod ceil(blocks / D)`; a namespace object written by `run` uses
  `s = labeled_key(namespace seed, "object", [xxh3(path)])`, `u = 0`.
- **Manifest** (`schema/README.md` §6). `dataset` is the abstract's `datasets` entry with
  every `{"param": x}` replaced by the value in effect (`gpus` included) and `doc` removed,
  in canonical key order; `payload` is the block above (generator, version, wrapper, block
  size, dedupe, compress); `manifest_version` is 1; `provenance` records the abstract's name
  and hash, all parameter values, the datagen version, host, start and end times, and the
  files and bytes written. The id is the SHA-256 of the canonical normative part. `run`
  compares `dataset` field by field against what it resolves and prints the id; the payload
  block is recorded and printed, not derivable from the abstract, so a published id is what
  pins it (`--expect-dataset-id`).
- **Containers** are written by the Python side: `aeiou-datagen` (`builder/README.md` §6)
  writes every dataset that has a format class, with the same names, sizes, payload, and
  manifest form (its `format` block records the class and writer library); `aeiou datagen`
  refuses such a dataset. A corpus with both kinds is written by both tools, each dataset by
  one of them.
- Because the corpus is sized per submission (`PROJECT_BRIEF.md` §5, dataset sizing rule),
  the count of every committed corpus is a parameter (`files`, `nodes`, `lists`,
  `sys_prompts`, `shards`, `n`), and so are the size parameters a small test corpus needs
  (`sample_mean`, `sample_sd`, `sys_tokens`); the resolved definition carries the values.

## 6. Several hosts: the coordinator (2026-09-30)

`aeiou run … --ranks R --rank r --coordinator HOST:PORT` on each of R hosts, rank 0 first
(`aeiou-launch` does it over ssh). The design is `NAPKIN_MATH.md` §8.A; what changed in
building it is `DESIGN_REVIEW.md` §3.25.

- **Topology and wire.** Rank 0 runs the coordinator in-process (`coord::Server`) and
  connects to it like every other host (`coord::Tcp`). Blocking `std::net`, one reader
  thread per socket, frames of a big-endian `u32` length and a JSON body. Hosts may take up
  to 120 s to connect; a peer silent for 30 s is dead (heartbeats every 5 s while idle) and
  the run aborts on every host.
- **Configuration check.** `Hello` carries a SHA-256 over the abstract's hash, seed, G, the
  resolved parameters, the dataset ids, the backend, the rotation, the time scale, and the
  write compression, plus this host's barrier scopes with their instance counts. A host whose
  hash differs from the first host's is refused with the reason, and the run stops on every
  host before any I/O.
- **Startup order.** Each host checks its datasets, connects, and checks its input
  namespaces; rank 0 alone prepares the output namespace roots (`--root` is the storage under
  test, shared by every host, so only one host may empty anything); every host then sends
  `Ready`, and the server answers `Start` with a common `t0` once all have. A host that fails
  a startup check after connecting tells the others; one that fails before connecting leaves
  them to the connect window.
- **Barriers** are two-level: a host's instances arrive in-process as on one host; the last
  arrival sends `Arrive`; the server sends `Release` with the generation once every host
  with participants in the scope has arrived. A host whose participants have all left sends
  `Leave`; a generation completed by a host leaving is a host-level departure release,
  reported with the in-host ones.
- **Reduction.** Each host sends its report (stats, histograms, per-take records, created
  and removed objects, its rank record) and prints its own; the server merges them
  (`Report::merge_all`: fingerprint modulo 2^64, histograms bucket-wise, created objects net
  of every host's removals, elapsed the longest host's). Rank 0 prints the merged report,
  writes the namespace manifests with every rank's host and GPU range, applies
  `--expect-fingerprint` to the merged fingerprint, and sends the verdict (`Result`) to every
  host, which exits with it. A host's own fingerprint line is its partial sum; the "all
  hosts" line is the fingerprint.
- **Faults.** An actor failing on any host, a refused configuration, a dropped socket, or a
  missed heartbeat sends `Stop` to every host; actors see it at their next op or wait, and
  the run fails on every host naming the originating rank and reason.
- **`aeiou-launch`** (`runner/aeiou-launch`, POSIX sh): `aeiou-launch [-p PORT] HOST… --
  aeiou run ARGS…` starts rank *i* on the *i*-th host over ssh with `--ranks`, `--rank`, and
  `--coordinator HOST0:PORT` (port 7311 by default) appended, prefixes each host's output
  with its rank and host, and exits non-zero if any rank did. `AEIOU_RSH` replaces `ssh`.

Observed on WSL2 (2026-09-30, `runner/aeiou/tests/coord.rs` and `aeiou-launch` with a local
shim for ssh): two processes on `localhost` run `train_small_files` to the dry-run
fingerprint; `ckpt_write_dcp` on two ranks leaves a manifest with both rank records, and
`ckpt_restore` on two ranks with `--rank-rotate 1` reads it, each rank running the other's
GPU range; a differing seed on one rank is refused on both before any I/O; a failing
`--expect-fingerprint` fails both ranks and the launcher. Real hosts have not been tried.

## 7. Loopback NFS on WSL2 (2026-09-30)

The development box has no NFS target (`PROJECT_BRIEF.md` §7). A loopback mount exercises
the real Linux NFS client code paths for correctness, not for performance: the server is
`nfs-kernel-server` exporting a tmpfs on the same kernel (6.18, WSL2), so every number below
is RPC overhead over loopback with memory behind it. The sequence that set it up (root
needed; the tmpfs and the mount do not survive a WSL restart, `/etc/exports` does):

```
sudo apt install nfs-kernel-server
sudo mkdir -p /srv/aeiou-export /mnt/aeiou-nfs
sudo mount -t tmpfs -o size=8g tmpfs /srv/aeiou-export
sudo chown $USER /srv/aeiou-export && sudo chmod 1777 /srv/aeiou-export
echo '/srv/aeiou-export localhost(rw,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports
sudo exportfs -ra
sudo systemctl start nfs-server          # or: sudo service nfs-kernel-server start
sudo mount -t nfs4 localhost:/srv/aeiou-export /mnt/aeiou-nfs
```

The mount comes up as NFS v4.2 over TCP, `rsize`/`wsize` 1 MiB, `hard`, with the default
attribute cache (`acregmin=3,acregmax=60,acdirmin=30,acdirmax=60`). Two practices follow
from what the first attempts showed:

- **Generate behind the server, run through the mount.** `aeiou datagen --root
  /srv/aeiou-export/…` then `aeiou run --root /mnt/aeiou-nfs/…`: the client's page cache
  starts cold, so the first buffered run shows what reaches the server. Datagen through the
  mount works (600 files, 69 MiB, in 145 ms against 39 ms on the tmpfs) but leaves every
  byte in the client's cache, and the first `sync` run then issues no READ RPC at all.
- **Use a directory name the client has never looked up.** Removing a tree through the
  mount and recreating it behind the server makes it invisible for up to `acdirmax` (60 s):
  the client caches the negative lookup, `aeiou run` reports no manifest, and `mkdir` of the
  namespace root fails with `EEXIST` because the server has it and the client does not. A
  fresh name per generation (`smoke-<unix time>` in the runs below) avoids it; so does
  waiting, or `echo 3 > /proc/sys/vm/drop_caches` as root.

**Observed (2026-09-30).** With `TMPDIR` on the mount, `tests/run.rs` and `tests/coord.rs`
pass (13 tests: every round trip, the write-then-restore handoff, and the two-rank runs as
threads and as processes). Every committed abstract then ran on the mount with small
parameters, each three times: `sync` with a cold client cache, `sync` again, and
`sync-direct`; all 27 runs reproduced their dry-run fingerprint. The NFS client's counters
(`/proc/self/mountstats`, per run) show what each backend put on the wire:

| abstract (G) | application ops | `sync`, cold: RPCs | `sync`, warm: RPCs | `sync-direct`: RPCs |
|---|---|---|---|---|
| train_small_files (2) | 678; 98 opens, 192 reads, 10.6 MiB | 97 READ (10.5 MiB), 53 OPEN, 2 READDIR | 0 READ, 99 GETATTR | 288 READ (10.7 MiB) |
| train_large_samples (2) | 412; 244 reads, 159 MiB | 129 READ (80.8 MiB from the server, the rest page-cache hits within the run) | 0 READ, 19 GETATTR | 464 READ (159 MiB) |
| kv_cache_serving (1) | 1,144; 162 reads, 115 writes, 34 mkdirs | 14 READ, 116 WRITE (28.8 MiB), 117 OPEN, 34 CREATE | 0 READ, 116 WRITE | 162 READ, 116 WRITE, 120 CLOSE |
| model_load (2) | 88; 76 reads, 4.1 MiB | 53 READ | 0 READ, 5 GETATTR | 76 READ (one per read) |
| vdb_search_diskann (1) | 244; 240 reads of 4 KiB | 241 READ | 240 READ | 240 READ |
| vdb_search_ivf (1) | 36; 32 reads, 1.8 MiB | 28 READ | 0 READ | 32 READ (one per read) |
| vdb_build_diskann (1) | 386; 106 reads, 262 writes, 7 MiB | 139 READ, 8 WRITE, 1 COMMIT | 0 READ, 8 WRITE, 1 COMMIT | 106 READ, 263 WRITE, 0 COMMIT |
| ckpt_write_dcp (2) | 70; 34 writes, 21.1 MiB | 27 WRITE, 4 COMMIT, 3 RENAME | same | 39 WRITE (21.1 MiB direct), 4 COMMIT |
| ckpt_restore (2), after the write on this client | 36; 20 reads, 18.1 MiB | **0 READ**: all 18.1 MiB from the client's page cache | 0 READ | 30 READ (18.2 MiB from the server) |

What the table says, with the reasoning in `DESIGN_REVIEW.md` §3.26:

- **A warm restore reaches the server not at all.** Reading a checkpoint on the client that
  wrote it is served entirely from the page cache; `sync-direct` is the only single-host way
  to make the restore touch storage, and other hosts (`--rank-rotate`, `--require-cold`) are
  the real answer (`DESIGN_REVIEW.md` §3.24).
- **Buffered reads are merged by readahead; O_DIRECT reads go out one RPC per application
  read** (model_load 76 of 76, IVF 32 of 32; the training abstracts issue more because an
  unaligned tail read is rounded out and every `until_eof` probe is an RPC returning EOF).
  This is `NAPKIN_MATH.md` §8.B's expectation that the abstract's `xfer` becomes the wire
  size, seen on a real NFS client.
- **The abstract's own `O_DIRECT` is honoured whatever the backend.** DiskANN opens its index
  direct, and all three runs show 240 READ RPCs of 4 KiB; the backend adds nothing.
- **Buffered writes coalesce to `wsize`; O_DIRECT writes go out as issued** (vdb_build: 8
  WRITE RPCs for 262 application writes buffered, 263 direct). The checkpoint's `fsync`
  becomes COMMIT either way.
- **Close-to-open costs a GETATTR per open even when everything is cached** (99 for 98 opens
  in the warm small-file run): the per-file metadata floor of `NAPKIN_MATH.md` §3.2.

Latencies on the loopback: open 40–250 µs (the first open of a name 130–340 µs, later ones
40–50 µs); buffered read 100–370 µs from the server, 13–18 µs from the cache for a small
file and ~140 µs for 1 MiB; `O_DIRECT` read 150–550 µs; write 400–650 µs buffered,
600–800 µs direct; `mkdir` and `rename` 300 µs–2 ms. Against ext4 (§4: 10 µs cached, 258 µs
direct) the pattern is the same with an RPC round trip added. None of this is throughput:
the server is memory on the same kernel.

## 8. The `io_uring` backends (2026-10-01)

`--io-backend io_uring` and `io_uring-direct` run the same abstract, the same op stream,
and the same fingerprint on an event loop instead of a thread per actor. The reasoning,
and the kernel behaviour found on the way, is `DESIGN_REVIEW.md` §3.29.

- **The VM is resumable.** The walk is an explicit state machine (`vm.rs`): `next()` yields
  the next op, control, or fork event, and between two calls the VM holds no borrowed
  state, so a driver may park it at an event for as long as the event takes. `dry-run` and
  the `sync` sink call `drive`, which feeds the events to a `Sink` as before; the fork
  protocol (snapshot, `start_sub`) is unchanged, and no golden fingerprint moved.
- **Tasks and loops.** `--threads N` event-loop threads (default: one per core, at most one
  per actor instance); instances go round-robin over the loops, and an instance's
  sub-actors (`parallel` sub-actors, loader workers) run on its loop, so channels are
  loop-local and lock-free. A task is a VM plus its `ActorState` (files, payload filler,
  statistics, created and removed paths), the type the thread-per-actor sink uses too, so
  the structural checks, `expect`, recording, and the namespace bookkeeping are one
  implementation under every backend.
- **One ring per loop; one op in flight per task.** An actor is a sequential program (a
  PyTorch worker blocks in `read`), so a task issues one SQE and parks; concurrency is the
  number of tasks on the loop. The loop steps every runnable task, submits, waits in
  `io_uring_enter` for a completion or the nearest `compute` timer (`IORING_ENTER_EXT_ARG`,
  kernel 5.11+), drains the CQ, and dispatches by `user_data`. Opcodes used: `openat`,
  `read`, `write`, `fsync`/`fdatasync`, `statx` (stat, fstat), `unlinkat` (unlink, rmdir),
  `mkdirat`, `renameat`, `fallocate`, `fadvise`, `ftruncate` (6.9+). What has no opcode
  (`lseek`, `ioctl`, `readdir`/`getdents64`), or what the ring's probe says the kernel lacks,
  runs inline on the loop thread through the blocking backend, as an `io_uring` application
  has to do; `close` drops the actor's reference as under `sync`.
- **Every read and write is issued at its effective offset**, the one the VM computed and
  the fingerprint hashes, never at the kernel's file position (`offset = -1`). Observed on
  Linux 6.18: a `-1` read of an `O_DIRECT` file through the ring returns data but does not
  advance the position (a buffered one does, and `pread` is right in both modes), so the
  `until_eof` idiom read the first megabyte again and again. The op stream is unchanged;
  turning `read` at a known position into `pread` is what a shim may do under the
  interposition test (`PROJECT_BRIEF.md` §5). `lseek` is still issued (inline) and recorded.
- **Buffers** are a pool of 4 KiB-aligned chunks per loop, one per op in flight, returned to
  the back of a FIFO and reused only once the pool holds `--buffer-mib`, so successive
  copies land in different memory (`NAPKIN_MATH.md` §2.2). Memory is (ops in flight) ×
  (op size), what any `io_uring` loader needs.
- **Control.** `compute` is a timer (a heap of deadlines; the enter timeout is the nearest);
  `take`, `put`, and a loader's slot park the task on the loop-local channel, and every
  change to the channel wakes its waiters to retry; a `parallel` parent parks until its
  sub-actors end; a main line that has ended parks until its loader workers end, then
  checks that every batch was taken. Barriers use the coordinator's non-blocking half
  (`arrive`, `released`): the loop keeps a read posted on an eventfd that every release
  writes, local or from the TCP coordinator, so a release arrives as a completion
  (`NAPKIN_MATH.md` §8.A as designed). A loop with nothing in flight, no timer, and no
  barrier is a deadlock and is reported with the parked actors.
- **Stall and latency** are measured as under `sync`: an op from SQE push to CQE (queueing
  in the SQ included, as the application sees it), a `take` from park to wake.
- ~~**Not done**, deliberately, as the A/B knobs of `NAPKIN_MATH.md` §8.5: fixed files and
  buffers, `SQPOLL`, `SINGLE_ISSUER`/`DEFER_TASKRUN`, `IORING_SETUP_ATTACH_WQ`,
  `IORING_REGISTER_IOWQ_MAX_WORKERS`, op linking. And the per-backend counters (io-wq
  workers, `mountstats` deltas), which are still shell scripts around the runner.~~ The
  counters are report fields since 2026-10-01 (§4, host counters); the ring knobs follow.
- **Knobs** (`RunOpts::uring`, added 2026-10-01, `DESIGN_REVIEW.md` §3.34). Options of the
  two `io_uring` backends, never backends of their own; none changes the op stream or the
  fingerprint, and the `sync` backends refuse them rather than ignore them, so a row of the
  A/B matrix cannot be mislabeled. Like `--threads` they are a host's tuning, not part of
  the configuration the coordinator compares; the report prints them (`io_uring:` line,
  rank 0's once merged).

  | flag | kernel | what it does |
  |---|---|---|
  | `--iowq-max-workers N` | `IORING_REGISTER_IOWQ_MAX_WORKERS` (5.15) | caps the bounded io-wq workers. io-wq belongs to the task that owns the ring, so the cap is **per loop**: `N × --threads` workers on the host. The same call returns the kernel's caps before ours, which the report prints with or without the flag (`min(SQ entries, 4 × cores)` bounded, `RLIMIT_NPROC` unbounded: 80 and 127,569 here). |
  | `--sqpoll IDLE_MS` | `IORING_SETUP_SQPOLL` (unprivileged since 5.13) | a kernel thread per loop takes SQEs off the ring and sleeps after `IDLE_MS` idle. Punted ops then run on the poll thread's io-wq, and the cap above applies to that one. |
  | `--sqpoll-shared` | `IORING_SETUP_ATTACH_WQ` | every loop attaches to the first loop's poll thread: one `iou-sqp` thread and one io-wq for the host, so the cap is a host total. Since 5.12 this flag shares the poll thread only; without `SQPOLL` there is nothing for it to share, hence `requires --sqpoll`. |
  | `--defer-taskrun` | `IORING_SETUP_SINGLE_ISSUER` + `IORING_SETUP_DEFER_TASKRUN` (6.1) | completions are processed only inside the loop's own `io_uring_enter`. The loop already is the single issuer and always enters with `GETEVENTS`. Not with `--sqpoll` (the kernel refuses the pair). |
  | `--coop-taskrun` | `IORING_SETUP_COOP_TASKRUN` (5.19) | no interrupt of the loop thread to run completions. |

  Each ring is built on its own loop thread (`SINGLE_ISSUER` binds a ring to the task that
  built it); under `--sqpoll-shared` loops 1.. wait for loop 0's ring. **Still not done:**
  fixed files (direct descriptors: `openat` into a slot table, a second descriptor table
  beside the one sub-actors inherit), fixed buffers (the pool would have to be one
  registered slab carved by size), and op linking (left off by decision, `DESIGN_REVIEW.md`
  §5).

**Observed (2026-10-01, WSL2).** Every committed abstract reproduces its dry-run fingerprint
under `io_uring` (`tests/run.rs`: five abstracts with nested `parallel`, loaders, barriers
across loops, namespace write-then-read, on one loop and on three;
`builder/tests/test_formats.py`: the three container abstracts; the other five by hand), and
under `io_uring-direct` wherever `sync-direct` runs. On ext4 from the page cache,
`train_small_files` with 64 GPUs × (1 + 4 workers) = 320 actors, 92,672 ops, second run:

| backend | threads | elapsed | client CPU (user + sys) | read mean / p99 | RSS |
|---|---|---|---|---|---|
| `sync` | 320 | 0.23 s | 1.5 s | 632 µs / 16.8 ms | 210 MB |
| `io_uring` | 20 loops | 0.18 s | 1.7 s | 450 µs / 6.3 ms | 87 MB |
| `io_uring --threads 4` | 4 loops | 0.22 s | 0.58 s | 606 µs / 2.1 ms | 91 MB |
| `sync-direct` | 320 | 0.94 s | 12.9 s | 6.0 ms / 50 ms | 1.85 GB |
| `io_uring-direct` | 20 loops | 0.93 s | 11.5 s | 5.3 ms / 50 ms | 538 MB |

The direct rows read the virtual disk and say nothing about storage; the buffered rows are
page-cache copies, where client CPU per op is the whole cost: four loops move the same
bytes in the same time as 320 threads for a third of the CPU. On the loopback NFS mount
(§7), `train_small_files` with 4 GPUs (11,260 ops: 3,200 reads, 1,620 opens): `sync` and
`io_uring` put identical RPCs on the wire (1,601 READ, 1,225 OPEN), `io_uring-direct`
4,801 READ and 1,600 CLOSE like `sync-direct`; the peak count of `iou-wrk` threads was 20,
~~the core count (io-wq's bounded-worker cap),~~ for buffered and direct reads alike, because
every `openat` punts whatever the reads do. (Corrected 2026-10-01 with the knobs: 20 was the
actor count, 4 GPUs × 5 actors with one op in flight each, which happens to equal the cores
of this box. The kernel's cap is 80 per loop; see the knob table below.) Spike 1's question (does `O_DIRECT` keep the
worker count bounded on a real NFS client) needs open-heavy and read-heavy phases measured
separately on the real target with `IORING_REGISTER_IOWQ_MAX_WORKERS` in hand; the backend
exists to ask it.

**Observed again with the host counters in the report (2026-10-01, later).** The same
`train_small_files` with 4 GPUs on the loopback mount, `--time-scale 0`, each backend on its
own freshly generated directory, cold then warm, as the report now prints them:

| backend, pass | tasks peak | io-wq peak | cpu user + sys | server read | RPCs |
|---|---|---|---|---|---|
| `sync`, cold | 22 (20 actors, main, sampler) | 0 | 0.34 s | 191.4 MiB | 3,241: READ 1,600, OPEN 1,225, OPEN_NOATTR 375, GETATTR 15, READDIR 14 |
| `sync`, warm | 22 | 0 | 0.28 s | 0 | 1,620: GETATTR 1,620 |
| `io_uring`, cold | 26 (4 loops + 20 workers) | 20 | 0.53 s | 191.4 MiB | 3,242: READ 1,600, OPEN 1,223, OPEN_NOATTR 377, … |
| `io_uring`, warm | 18 | 12 | 0.20 s | 0 | 20: GETATTR 20 |
| `io_uring-direct`, cold | 25 | 19 | 0.51 s | 194.5 MiB | 8,042: READ 4,800, CLOSE 1,600, OPEN 1,223, OPEN_NOATTR 377, … |
| `io_uring-direct`, warm | 20 | 14 | 0.38 s | 194.5 MiB | 8,020: READ 4,800, CLOSE 1,600, OPEN_NOATTR 1,600, GETATTR 20 |

The warm `GETATTR` difference (1,620 under `sync`, 20 under `io_uring`) is not the backend:
run each backend warm on the other's directory and both send 20, and `sync` sends 1,620
only on a directory it has not touched for a few seconds. It is the attribute cache
(`acregmin=3`): the first open of each file after the timeout revalidates with a `GETATTR`,
and the next run within the (now longer) timeout does not. So warm comparisons across
backends need the same gap since the last touch, which the counters now make visible. The
`O_DIRECT requested read` figure is 3.13 GiB for 194.5 MiB from the server: 3,200 reads of
a 1 MiB aligned buffer each, counted as requested.

**Observed with the knobs (2026-10-01, later still).** `train_small_files` with 8 GPUs
(72 actors), 16,000 files, 896,000 ops of which 256,000 reads and 88,000 opens, on the
loopback mount, `--time-scale 0`, `--threads 4`, data from the client's page cache in every
row (one cold `sync` pass first), so each open still costs its `OPEN` and `DELEGRETURN`:

| row | elapsed | io-wq peak | cpu user + sys | RPCs |
|---|---|---|---|---|
| `sync` (72 threads) | 5.2 s | 0 | 10.2 s | 181k |
| `io_uring` | 6.9 s | 30 (46 on another pass) | 9.6 s | 177k |
| `io_uring --iowq-max-workers 2` | 7.2 s | 8 | 10.0 s | 177k |
| `io_uring --iowq-max-workers 1` | 6.4 s | 4 | 8.6 s | 176k |
| `io_uring --defer-taskrun --coop-taskrun` | 7.0 s | 28 | 8.8 s | 177k |
| `io_uring --sqpoll 200` | 7.3 s | 54, and 4 `iou-sqp` | 13.0 s | 176k |
| `io_uring --sqpoll 200 --sqpoll-shared` | 13.6 s | 6, and 1 `iou-sqp` | 9.0 s | 177k |
| `io_uring-direct` | 21.8 s | 23 | 24.7 s | 560k (READ 384k) |
| `io_uring-direct --iowq-max-workers 2` | 22.6 s | 8 | 24.6 s | 560k |

What the rows say, with the loopback's usual caveat that the server shares the client's
cores: (1) the worker peak follows the actors with an op punted at that moment, far below
the kernel's 4 × 80, and is not the core count; (2) capping it changes almost nothing: four
workers in all carry the run that 30 to 46 carried, with the same RPCs and slightly less
CPU, so on this mix the punted opens are short and the workers were mostly waking and
sleeping; (3) `SQPOLL` buys nothing here and costs CPU (four spinning threads), and one
shared poll thread halves the rate, since it serializes four rings' submissions and their
punts; (4) `DEFER_TASKRUN` with `COOP_TASKRUN` is neutral. None of this is a number for a
real target. It says which rows Spike 1 should run there: the cap at 1, 2, and default,
separately for open-heavy and read-heavy phases, buffered and direct.

## 9. The `posix-aio`, `libaio`, and `mmap` backends (2026-10-01)

Five more values of `--io-backend`, same abstract, same op stream, same fingerprint. Two
are blocking backends on the thread per actor of `sync`, and one is a second engine for
the event loop of §8. The choices and what they leave open are `DESIGN_REVIEW.md` §3.35.

- **`posix-aio`, `posix-aio-direct`** (`backend.rs`). Every read, write, `fsync`, and
  `fdatasync` is glibc's `aio_read`/`aio_write`/`aio_fsync` followed by `aio_suspend` until
  it is done; the other ops are the `sync` calls. glibc serves requests from a user-space
  pool (20 threads by default, and requests on one descriptor one at a time), so an op is
  a hand-off to another thread around the same `pread`. An actor has one op in flight, so
  there is no list for `lio_listio`. Its counter block is the host line's task peak (the
  pool's threads appear there).
- **`libaio`, `libaio-direct`** (`aio.rs`). The kernel AIO system calls (`io_setup`,
  `io_submit`, `io_getevents`, issued directly: the libaio library is a thin wrapper and is
  not linked) as the engine of the §8 loop: `--threads N` loops, one AIO context per loop,
  one op in flight per task. The loop of `uring.rs` is generic over an `Engine` (issue,
  complete, wait, and a wake-up on the coordinator's eventfd); the ring is one engine and
  the AIO context the other, so tasks, channels, timers, and barriers are one
  implementation. The interface carries `pread`, `pwrite`, `fsync`, `fdatasync`, and a
  poll (`IOCB_CMD_POLL` on the eventfd, Linux 4.18: how a barrier release wakes the
  loop). **Every other op is issued inline and blocks the loop**: `open`, `close`, `stat`
  have no AIO form, which is what an AIO application lives with. Requests queue while the
  loop advances its tasks and go to the kernel in one `io_submit` per turn. Reads and writes
  go at their effective offset, as under `io_uring`. `--aio-depth N` is `io_setup`'s
  `nr_events` per loop (default 256; the host's total is bounded by `fs.aio-max-nr`); a
  full context returns `EAGAIN`, and the loop then reaps a completion and submits again.
  The interface is asynchronous only with `O_DIRECT`: a buffered read is carried out
  inside `io_submit`. The report's `libaio:` line says how long the loops spent there.
- **`mmap`** (`backend.rs`). The first read of a descriptor maps the whole file
  (`PROT_READ`, `MAP_SHARED`, the size from an `fstat` the backend issues itself), a read
  makes its range of the mapping resident, and `close` unmaps: the shape of a safetensors
  or Arrow load. A read at or past the end returns what `pread` would; a read past the
  mapped length asks `fstat` again and remaps if the file has grown. Writes and every
  other op are the `sync` calls. Two options, both part of the configuration the
  coordinator compares (`--aio-depth`, like `--threads`, is a host's tuning and is not):
  - `--mmap-consume touch` (default) reads one byte of every page of the range and copies
    nothing; `copy` copies the range into the actor's buffer (a loader that copies out of
    the mapping). Either way the read returns only when every page has arrived, because a
    fault does not return before its page is read. The default charges `mmap` what the API
    costs and no more: under `read(2)` the kernel's copy out of the page cache is part of
    the call, under `O_DIRECT` and `mmap` there is none, and what the application then
    does with the bytes (the copy to the GPU) is charged to no backend
    (`DESIGN_REVIEW.md` §3.35, revised).
  - `--mmap-mode` is the prefetch before that: `fault` (default: none, the touch faults the
    pages in), `populate` (`MADV_POPULATE_READ` over the range, Linux 5.14; nothing is
    touched afterwards, the call returns when the range is resident and mapped),
    `willneed` (`MADV_WILLNEED` starts readahead and the touch waits for it).

  A fault blocks the faulting thread only, and the backend runs one thread per actor, so
  a page being read stalls the one actor that asked for it, as a blocking `pread` does.
  Four things to know when reading its numbers: an I/O error or a truncation under
  `fault` and `willneed` is a `SIGBUS`, as for the applications it models; the map, the
  unmap, the translation flush each unmap causes, and the faults are what the technique
  costs and are counted against it (decided 2026-10-01); because every actor thread maps
  into the runner's one address space, those costs also couple the actors (one
  `mmap_lock`, flushes to every core running an actor) more than they couple PyTorch
  workers, which are processes; and the NFS client's `buffered read` byte counter counts
  `read(2)` only, so it stays at zero and `server read` is the figure.
- **Report.** `libaio: loops … context depth … in-flight peak … io_submit N calls, M
  requests, T inside  io_getevents … context full …` and `mmap: mode … consume … mappings N
  (bytes)  madvise calls …  pages touched …  copied out …` (`Report::aio`, `Report::mmap`, summed over hosts). The
  host line gained `faults minor … major …` (`getrusage`) under every backend; under
  `mmap` the major faults are the reads that reached storage.
- **Refusals.** `--aio-depth` without a `libaio` backend and `--mmap-mode` or
  `--mmap-consume` without `mmap` are refused, like the ring knobs under `sync`. There is no `mmap-direct`.
- **Tests** (`tests/run.rs`). `train_small_files` under all nine backends; the
  `kv_cache_serving`, `vdb_search_diskann`, and `ckpt_write_dcp` runs of the `io_uring` test
  under the five new ones (`libaio` on one loop and on three, barriers through the poll,
  write-then-read-back through a mapping made after the write), and the three `mmap`
  modes under `touch` and under `copy` with their counters (pages touched equals bytes
  over 4 KiB, none under `populate`, bytes copied only under `copy`); `builder/tests/test_formats.py` runs the three container
  abstracts under `posix-aio`, `libaio`, and `mmap` as well. Not tested: a context that fills (the kernel rounds
  `nr_events` up to a few per CPU, so a small test never gets `EAGAIN`).

**Observed (2026-10-01, loopback NFS of §7, `--time-scale 0`).** `train_small_files` with
4 GPUs (36 actors), 1,600 files, `steps=12`: 10,752 ops, of which 3,072 reads and 1,536
opens, 179 MiB. Each row on its own freshly generated directory, cold; all rows reproduce
the fingerprint.

| backend | elapsed | tasks peak | cpu user + sys | read mean | RPCs, cold | backend's own line |
|---|---|---|---|---|---|---|
| `sync` | 0.16 s | 38 | 0.27 s | 0.52 ms | 4,618: READ 1,536, OPEN 1,536, CLOSE 1,536 | |
| `posix-aio` | 0.23 s | 58 | 0.68 s | 1.2 ms | 4,616: the same | 20 pool threads in the task peak |
| `posix-aio-direct` | 0.31 s | 58 | 0.77 s | 1.6 ms | 7,688: READ 4,608 | |
| `io_uring --threads 4` | 0.24 s | 41 | 0.48 s | 0.77 ms | 4,632 | io-wq peak 35 |
| `io_uring-direct --threads 4` | 0.34 s | 38 | 0.54 s | 1.4 ms | 7,693: READ 4,608 | io-wq peak 32 |
| `libaio --threads 4` | 0.28 s | 6 | 0.17 s | 1.3 ms | 4,614: as `sync` | 512 `io_submit` calls, 0.36 s inside |
| `libaio-direct --threads 4` | 0.37 s | 6 | 0.28 s | 1.7 ms | 7,686: READ 4,608 | 1,192 calls, 0.17 s inside |
| `mmap` (touch) | 0.29 s | 38 | 0.48 s | 1.3 ms | 6,756: READ 2,137, GETATTR 1,536, OPEN, CLOSE | major faults 1,536; 46,651 pages touched |
| `mmap --mmap-mode populate` | 0.24 s | 38 | 0.46 s | 1.1 ms | 6,755: READ 2,137, GETATTR 1,536 | major faults 3,087; nothing touched |
| `mmap --mmap-mode willneed` | 0.24 s | 38 | 0.45 s | 1.0 ms | 6,148: READ 1,536, GETATTR 1,536 | major faults 0 |
| `mmap --mmap-consume copy`, the three modes | 0.25, 0.25, 0.22 s | 38 | 0.50, 0.50, 0.46 s | 1.2, 1.2, 1.0 ms | as the rows above | RSS 33 MiB against 8 MiB |

A second pass on the same directory sends 3,072 RPCs under every buffered backend
(`OPEN_NOATTR` and `CLOSE`, no READ) and the cold count again under the direct ones.

Then the 8-GPU run of §8's knob table (72 actors, 16,000 files, 896,000 ops of which
256,000 reads and 128,000 opens), data from the client's page cache, the server handing
out read delegations (176k RPCs: `OPEN_NOATTR` and `DELEGRETURN`) in every buffered row:

| row | elapsed | tasks peak | cpu user + sys | read mean | backend's own line |
|---|---|---|---|---|---|
| `sync` (72 threads) | 5.5 s, 6.1 s | 74 | 10.3 s, 11.1 s | 13 µs, 18 µs | |
| `io_uring --threads 4` | 7.7 s | 43 | 10.8 s | 0.72 ms | io-wq peak 37 |
| `posix-aio` | 10.0 s | 94 | 31.1 s | 2.3 ms | |
| `libaio --threads 4` | 7.9 s | 6 | 8.8 s | 0.99 ms | in-flight peak 16; 16,128 `io_submit` calls, 2.4 s inside |
| `libaio` (8 loops) | 6.8 s | 10 | 10.8 s | 0.78 ms | in-flight peak 8; 3.0 s inside |
| `libaio-direct --threads 4` | 23.8 s | 6 | 22.1 s | 4.2 ms | 595k RPCs (READ 384k); 12.9 s inside |
| `libaio-direct` (8 loops) | 22.2 s | 10 | 26.4 s | 3.8 ms | 600k RPCs; 14.7 s inside |
| `mmap` (touch) | 6.3 s | 74 | 15.1 s | 29 µs | 128,000 mappings; 3.9M pages touched; 330k minor faults; RSS 11 MiB |
| `mmap --mmap-consume copy` | 7.2 s | 74 | 18.7 s | 39 µs | 393k minor faults; RSS 122 MiB |
| `mmap --mmap-mode populate --mmap-consume copy` | 7.2 s | 74 | 21.9 s | 0.26 ms | 128,000 `madvise` calls |
| `mmap --mmap-mode willneed --mmap-consume copy` | 7.2 s, 7.5 s | 74 | 23.1 s, 23.4 s | 0.48 ms | 128,000 `madvise` calls |
| `sync-direct` (72 threads) | 20.8 s | 74 | 32.3 s | 2.9 ms | 595k RPCs (READ 384k), 15 GiB from the server |

The same run in the other regime the server was in for part of the series (no
delegations: 256k RPCs, an `OPEN_NOATTR` and a `CLOSE` per open), where the three prefetch
modes under `touch` were measured:

| row | elapsed | cpu user + sys | read mean |
|---|---|---|---|
| `sync` | 8.4 s | 13.4 s | 13 µs |
| `mmap` (touch) | 8.6 s | 16.2 s | 13 µs |
| `mmap --mmap-mode populate` (nothing touched) | 8.9 s | 17.9 s | 43 µs |
| `mmap --mmap-mode willneed` | 9.1 s | 18.8 s | 61 µs |
| `mmap --mmap-consume copy` | 9.2 s | 19.9 s | 23 µs |

What the rows say, with the loopback's caveat that the server shares the client's cores
and that none of this is a number for a real target:

1. **`posix-aio` tracks `sync` on the wire and not in CPU.** The RPCs are identical; the
   hand-off to glibc's pool and back costs 2.5 to 3 times the CPU and turns a 13 µs
   page-cache read into 2.3 ms, because 72 actors queue for 20 pool threads. The brief's
   "expect it to track `sync`" holds for what reaches the server only.
2. **`libaio` buffered is a small synchronous thread pool, and here a cheap one.** Four
   loops carry what 72 threads carry with the least CPU of any row and 6 tasks, because
   nothing is handed to another thread: opens, closes, and (inside `io_submit`) the reads
   all run on the loop. The price is that each loop does one thing at a time, which a
   server with real latency would expose and this one does not.
3. **`libaio-direct` is asynchronous for the reads only.** Its RPCs are those of the other
   direct backends (one READ per megabyte asked, 3 per file here). A seventh of the loops'
   time in the 16,000-file run was still spent inside `io_submit` (12.9 s of 4 × 23.8 s: what
   the NFS client does to set a direct read going), and every open and close blocks the
   loop besides. Buffered and cold, in the first table, it was a third (0.36 s of 4 × 0.28 s).
4. **`mmap` puts different RPCs on the wire for the same reads.** Cold, the fault path
   sent 2,137 READs where `read(2)` sent 1,536 (readahead for faults uses its own window),
   plus one GETATTR per mapping that `read(2)` does not send; `MADV_WILLNEED` brought it
   back to one READ per file with no major fault, and `MADV_POPULATE_READ` changed nothing
   on the wire. `touch` and `copy` send the same RPCs.
5. **Warm, `mmap` with `touch` costs 20 to 35 % more CPU than `sync`, and the copy another
   quarter on top.** 16.2 s against 13.4 s without delegations, 15.1 s against 11.2 s with
   them: that difference is the map, the unmap with its flush, and about three minor
   faults per file, on one address space, with no copy at all against `sync`'s copy of
   every byte. It is the price of the technique and stays in the row. The copy adds
   3.6 s for 14.8 GiB (user time 3.8 s to 6.1 s) and 110 MiB of resident buffers; it was
   the default for a few hours and overstated `mmap` by that much. A touched read takes
   as long as a `read(2)` from the page cache (13 µs mean). The two advising modes cost
   more than plain faulting when the pages are already cached: an `madvise` per read for
   nothing. Against `sync-direct`, which fetched all 15 GiB again, any page-cache row
   wins on this warm run; the comparison that matters is a cold one on a real target.
6. **The first `sync` passes of the series ran without delegations** (256k RPCs, a `CLOSE`
   per open, 8.6 s and 9.6 s) and the later ones with them (176k, 5.5 s). The server
   decides that, not the backend; rows are comparable only within one regime, which the RPC
   line shows. It changed back and forth during the `touch` series as well, which is why
   that series is given in both regimes.

**Observed on large files (2026-10-01, same mount).** `train_large_samples` with 4 GPUs
(20 actors), `files=28`, `steps=2`: 28 files of about 140 MiB, 8,220 reads of 1 MiB,
7.83 GiB read, every file read twice (3.85 GiB from the server on a cold pass). This is
the shape `mmap` loaders are used for. Cold rows each on a freshly generated directory:

| row | elapsed | cpu user + sys | READ RPCs (mean size) | faults |
|---|---|---|---|---|
| `sync`, cold | 1.17 s | 4.8 s | 6,087 (about 660 KiB) | |
| `mmap` (touch), cold | 1.93 s | 3.6 s | 36,981 (about 109 KiB) | 3,573 major |
| `mmap --mmap-mode populate`, cold | 2.25 s | 5.0 s | 38,255 | 47,244 major |
| `mmap --mmap-mode willneed`, cold | 1.74 s | 4.7 s | 18,503 (about 218 KiB) | 4,031 major |
| `sync-direct`, cold | 1.58 s | 0.55 s | 16,384, 7.86 GiB from the server | |
| `sync`, warm | 0.46 to 0.50 s | 3.1 to 3.4 s | 0 | |
| `mmap` (touch), warm | 0.12 to 0.14 s | 0.32 to 0.40 s | 0 | 75k minor |
| `mmap --mmap-mode populate`, warm | 0.14 s | 0.24 s | 0 | |
| `mmap --mmap-mode willneed`, warm | 0.13 s | 0.32 s | 0 | |
| `mmap --mmap-consume copy`, warm | 0.47 s | 3.1 s (2.5 s of it user) | 0 | |

1. **Warm, on large files, `mmap` with `touch` uses a tenth of the CPU of `sync`** and
   finishes in under a third of the time: 56 mappings carry 7.8 GiB, so the fixed cost
   per mapping that dominated the small-file run is spread over 140 MiB each, and there
   is no copy. With `copy` the row is `sync`'s again, the copy moved from system to user
   time. This is the advantage the default now lets `mmap` show.
2. **Cold, the fault path sends six times as many READs, a sixth the size.** About
   109 KiB per READ against about 660 KiB under `read(2)`, on a mount with `rsize` 1 MiB.
   That is consistent with the mount's `read_ahead_kb` of 128 bounding the readahead
   window of a fault where a 1 MiB `read(2)` asks for its whole length; it has not been
   confirmed by changing the setting (root). `MADV_WILLNEED` halves the count, and
   `MADV_POPULATE_READ` does not help. On this loopback `mmap` was the slowest way to
   read cold and nearly the cheapest in CPU after `sync-direct`; what small READs cost
   on a real server is the question for the real target, and `read_ahead_kb` is a
   solution-side setting ~~the report does not yet record~~ the host counters record
   since later the same day (`mount read_ahead_kb 128` here, 8192 on this box's ext4),
   recorded and never set by the runner.
   **Confirmed later the same day (user, as root):** with
   `echo 1024 > /sys/class/bdi/0:78/read_ahead_kb` and nothing else changed, the same
   cold rows on fresh directories:

   | row, cold | `read_ahead_kb` 128 | `read_ahead_kb` 1024 |
   |---|---|---|
   | `mmap` (touch) | 1.93 s, 3.6 s CPU, 36,981 READ, 3,573 major faults | 0.85 s and 1.04 s, 1.6 s and 2.9 s CPU, 5,376 and 6,587 READ, 249 and 476 major faults |
   | `mmap --mmap-mode populate` | 2.25 s, 5.0 s CPU, 38,255 READ | 0.91 s, 2.2 s CPU, 6,073 READ |
   | `mmap --mmap-mode willneed` | 1.74 s, 4.7 s CPU, 18,503 READ | 1.23 s, 3.0 s CPU, 11,318 READ |
   | `sync` | 1.17 s, 4.8 s CPU, 6,087 READ | 1.38 s, 5.0 s CPU, 6,120 READ |

   The window is what sets the size of a fault's READ: about 109 KiB became 600 to
   750 KiB, the READ count fell to that of `read(2)`, and `read(2)` itself did not change
   (a 1 MiB read already asked for its whole length). With the larger window `mmap` is the
   fastest cold row here and uses a third to a half of `sync`'s CPU, where with the default
   it was the slowest; `willneed` went from the best `mmap` mode to the worst. So on NFS
   the cold result of an `mmap` loader is decided by a client setting that defaults to
   128 KiB whatever `rsize` is, which is why the report records it.
3. **RSS under `mmap` counts the mapped file pages** (about 1 GiB here against 70 MiB):
   they are page cache, shared and reclaimable, not buffers the runner allocated.
