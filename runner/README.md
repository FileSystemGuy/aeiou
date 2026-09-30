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
```

## 1. What exists (2026-09-30)

| Command | What it does |
|---|---|
| `aeiou check FILES…` | Loads each AST, validates it (structure plus rules V1–V13), prints its canonical SHA-256 and op-kind counts in the same format as `schema/check.py`. CI diffs the two outputs. |
| `aeiou dry-run AST --gpus G [--seed S] [--param k=v]…` | Walks every actor instance without I/O: op counts by kind and phase, bytes read and written, emulated compute, barriers, and the **workload fingerprint**. `--ranks R` adds bytes per host against this host's DRAM. `--gpu g [--steps a..b] [--limit n]` prints one instance's op stream. |

Not yet: `aeiou run` (the I/O backends), `aeiou datagen`, the coordinator, `--metrics`
(`PROJECT_BRIEF.md` §6 item 14), `stream` access, the `replay` node, and container layouts
beyond `samples_per_file` (which is implemented but untested against a format class).

```
aeiou/src/
  ast.rs       the contract as serde types (externally tagged, deny_unknown_fields)
  canon.rs     canonical form and SHA-256, byte-identical to check.py::canonical
  validate.rs  rules V1–V13 and the schema's structural constraints, ported from check.py
  sites.rs     draw sites: the JSON pointer of every draw / pick / choose, hashed
  pattern.rs   path patterns: {id div 1300:05}, {conv:016x}, {name}
  rng.rs       positional keys, SplitMix64 words, the 4-round Feistel permutation
  eval.rs      the resolved model: parameters, datasets (sizes, layouts, names), namespaces, distributions
  vm.rs        expression evaluation and the tree walk that emits ops to a Sink
  dryrun.rs    the dry-run Sink, parallel over actor instances, and the report
  main.rs      the CLI
aeiou/tests/golden.rs   hash parity with check.py, golden fingerprints, semantics tests
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
