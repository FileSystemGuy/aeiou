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
```

## 1. What exists (2026-09-30)

| Command | What it does |
|---|---|
| `aeiou check FILES…` | Loads each AST, validates it (structure plus rules V1–V13), prints its canonical SHA-256 and op-kind counts in the same format as `schema/check.py`. CI diffs the two outputs. |
| `aeiou dry-run AST --gpus G [--seed S] [--param k=v]…` | Walks every actor instance without I/O: op counts by kind and phase, bytes read and written, emulated compute, barriers, and the **workload fingerprint**. `--ranks R` adds bytes per host against this host's DRAM. `--gpu g [--steps a..b] [--limit n]` prints one instance's op stream. |
| `aeiou datagen AST --root DIR [--param k=v]… [--dedupe D] [--compress C] [--threads N] [--dataset NAME]…` | Writes every `files` and `regions` dataset the abstract declares under `DIR`, names, sizes, and chunks from the definition and the dataset seed, content per §5, in parallel by id, then the manifest `.aeiou-dataset.json` at each dataset root. Refuses a non-empty root (datasets are read-only, V12). Prints each dataset's id. |
| `aeiou run AST --gpus G --root DIR [--seed S] [--param k=v]… [--io-backend sync\|sync-direct] [--time-scale X] [--buffer-mib N] [--write-compress C] [--clean-namespaces] [--expect-fingerprint HEX] [--expect-dataset-id SHA]… [--rank r --ranks R --rank-rotate k] [--max-gap SECS] [--require-cold]` | Executes the abstract against `DIR` on one host (§4): checks every dataset against its manifest and every input namespace against the manifest of the run that wrote it, requires empty output namespace roots, runs one OS thread per actor and sub-actor with blocking POSIX calls, checks every result structurally, prints latency histograms, per-phase totals, per-step stall and busy fraction, and the fingerprint, and leaves `.aeiou-namespace.json` at every namespace root it wrote. |

Not yet: the TCP coordinator (several hosts), the asynchronous backends (`io_uring`,
`libaio`, `mmap`, …) and their counters, `mountstats`, `RLIMIT` startup checks, a JSON
report, `--metrics` (`PROJECT_BRIEF.md` §6 item 14), `stream` access, the `replay` node,
container layouts beyond `samples_per_file`, and datagen for format classes (that is the
Python side, `PROJECT_BRIEF.md` §6 item 8).

```
aeiou/src/
  ast.rs       the contract as serde types (externally tagged, deny_unknown_fields)
  canon.rs     canonical form and SHA-256, byte-identical to check.py::canonical
  validate.rs  rules V1–V13 and the schema's structural constraints, ported from check.py
  sites.rs     draw sites: the JSON pointer of every draw / pick / choose, hashed
  pattern.rs   path patterns: {id div 1300:05}, {conv:016x}, {name}
  rng.rs       positional keys, SplitMix64 words, the 4-round Feistel permutation
  eval.rs      the resolved model: parameters, datasets (sizes, layouts, names), namespaces, distributions
  vm.rs        expression evaluation and the tree walk that emits ops and control to a Sink;
               the fork protocol a Sink uses to run `parallel` and `loader` sub-actors itself
  dryrun.rs    the dry-run Sink, parallel over actor instances, and the report
  backend.rs   the Backend trait (blocking form) and `sync` / `sync-direct`
  run.rs       the run Sink: threads, channels, barriers, buffers, structural checks, report
  coord.rs     the Coordinator trait and the in-process implementation
  payload.rs   positional content (dgen-data behind the `aeiou-positional/1` wrapper), the manifest
  datagen.rs   `aeiou datagen`
  main.rs      the CLI
aeiou/tests/golden.rs   hash parity with check.py, golden fingerprints, semantics tests
aeiou/tests/run.rs      datagen + run round trips on a temporary directory, refusals, loader order
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
so `--gpus 8` uses eight). The committed abstracts at their defaults: training shapes in
under a second; `kv_cache_serving` 72M ops per instance in 24 s; `vdb_search_diskann` 68M
ops in 28 s; `vdb_build_diskann` 203M ops in 33 s. Memory is a few MB: no per-file
structure exists, and an actor's state is its frames, bindings, open files, and the as-written
sums of the objects it created. The obvious speed-up, caching a bound handle's resolved path
instead of re-formatting the pattern on every op, is not done yet.

## 4. `aeiou run` (2026-09-30)

What runs where, and what is checked. The design reasoning is `DESIGN_REVIEW.md` §3.23.

- **Threads.** One OS thread per actor instance. A `loader` spawns `workers` threads that
  live until the actor ends; a `parallel` spawns `width` threads and joins them before the
  node returns; either may nest. Every sub-actor starts from a snapshot of its parent's
  position and bindings (`vm::Snapshot`) and sees the files the parent had open at the fork;
  what it opens itself is its own. The VM's tree walk runs on the thread, so blocking calls
  are simply blocking: this is the `sync` fidelity reference of `PROJECT_BRIEF.md` §5, and
  the asynchronous backends will need a resumable VM instead.
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
  hit the barrier the same number of times. Single host only: `coord::Local` behind the
  `Coordinator` trait; the TCP implementation of `NAPKIN_MATH.md` §8.A is next.
- **`compute`** sleeps for `ns × --time-scale` and is recorded unscaled. `--time-scale 0`
  runs the I/O back to back.
- **Buffers.** Each thread has a 4 KiB-aligned read ring and a write ring of `--buffer-mib`
  (default 8), allocated on first use and grown to the largest op, so successive copies land
  in successive slices and the aggregate across threads exceeds L3 (`NAPKIN_MATH.md` §2.2).
- **Writes** carry the positional content of §5 for the object at that offset
  (`--write-compress` sets the ratio; no dedupe control on writes yet).
- **`sync-direct`** adds `O_DIRECT` to every regular-file open. An unaligned read (the
  `until_eof` idiom's last read starts at EOF, which is rarely aligned) is rounded out to
  4 KiB and the requested part is counted, as an `O_DIRECT` shim under a buffered application
  has to do; an unaligned write is refused.
- **Structural checks, every op.** A read must return the computed count; a write its
  length; `readdir` the computed entry count for a one-sample-per-file, unchunked dataset
  directory (`.aeiou*` entries not counted); a failing op's errno must be in the statement's
  `expect` list. Anything else aborts the run with the actor, position, and op. The runner
  never looks at the bytes (`PROJECT_BRIEF.md` §5, *Data verification*).
- **Startup.** Every dataset's manifest must match the resolved definition (§5); the ids are
  printed and may be pinned with `--expect-dataset-id`. Namespace roots are created and must
  be empty; `--clean-namespaces` empties them, leaving a dataset root that lies inside one
  alone.
- **Ranks.** `--rank r --ranks R` give this host its GPU id range, contiguous blocks of
  `ceil(G / R)`; `--rank-rotate k` makes it run rank `(r + k) mod R`'s range instead. Nothing
  in the op stream depends on the host, so rotating a read run against the write run's host
  list makes every host read what another host wrote (`DESIGN_REVIEW.md` §3.24). `--ranks`
  above 1 is refused until the coordinator exists; the arithmetic and the manifest records
  are in place.
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

Observed on the WSL2 ext4 disk (2026-09-30, `runner/aeiou/tests/run.rs` and the smoke runs):
every committed abstract runs to the dry-run fingerprint under `sync` (`ckpt_restore`
against the namespace a `ckpt_write_dcp` run left, through its manifest), and
`train_small_files` also under `sync-direct`; a page-cache read of 1 MiB costs
10 µs, an `O_DIRECT` one 258 µs; `vdb_search_diskann` spawns a thread per beam per hop
(923 threads for 924 reads), the cost of forking `parallel` afresh each time, and a per-actor
sub-actor pool is the planned fix. The loopback NFS mount of `PROJECT_BRIEF.md` §7 has not
been run yet (it needs root on the development box).

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
- Because the corpus is sized per submission (`PROJECT_BRIEF.md` §5, dataset sizing rule),
  the count of every committed corpus is a parameter (`files`, `nodes`, `lists`,
  `sys_prompts`, `shards`, `n`), and so are the size parameters a small test corpus needs
  (`sample_mean`, `sample_sd`, `sys_tokens`); the resolved definition carries the values.
