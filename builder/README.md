# `aeiou`: the Python builder

Layer 1 of `GRAMMAR_OPTIONS.md` Option D (chosen 2026-09-30). The package is named for the
repository, not for MLPerf Storage: nothing in it is specific to the working group's process
(`PROJECT_BRIEF.md` §8). An authoring script constructs
the AST of `schema/abstract-ast.schema.json` by running once; the calls that look like I/O are
declarations. The AST is the only thing the Rust runner executes and the only thing an author
publishes with a hash. Nothing here runs on a client node.

```
builder/
  aeiou/     the package: nodes, dists, builder (Workload, Cursor), validate, emit, hermetic, cli,
             params (parameter files, `aeiou-params`), formats (the format classes), datagen
             (`aeiou-datagen`), trace (`aeiou-trace`, §7), and the ports of the runner's definitions
             it needs: rng, pattern, layout
  abstracts/         the ABSTRACTS.md workloads as authoring scripts (§1–§8; §4 is two scripts), and the
                     container workloads train_stream_shards.py (TFRecord and Parquet) and train_map_hdf5.py
  tests/             pytest: every abstract builds, matches its committed AST, the discipline holds;
                     the parameter files fit their abstracts; the format classes write files the
                     libraries read back and the runner executes to the dry-run fingerprint; the
                     trace metrics against hand-written traces and against the runner under `strace`
  pyproject.toml     dep: jsonschema; extras `test` and `formats` (pyarrow, h5py, numpy, crc32c, dgen-py,
                     xxhash); `aeiou-build`, `aeiou-params`, `aeiou-datagen`, `aeiou-trace` entry points
```

```
cd builder && uv sync --extra test --extra formats     # or: pip install -e '.[test,formats]'
uv run aeiou-build --hermetic --twice -o ../schema/examples abstracts/*.py
uv run pytest
uv run aeiou-build --check -o ../schema/examples abstracts/*.py     # drift check (CI)
uv run aeiou-params check ../schema/examples/model_load.ast.json ../schema/examples/params/model_load.*.params.json
```

## 1. Writing an abstract

```python
from aeiou import *

w = Workload("train_small_files", doc="…")
P = w.P
w.param("batch", 32, unit="count")
w.param("step_time", 105 * ms, unit="ns", doc="[measure]")
w.param("reuse", mixture((0.55, none), (0.45, lognormal(median=40, sigma=1.2, min=1))))

train = w.dataset("train", pattern="train/{id div 1300:05}/img_{id:09}.jpg",
                  count=50_000_000, size=lognormal(median=110 * KiB, sigma=0.45),
                  seed=0x5eed_da7a, access="map")
ckpt = w.namespace("ckpt", pattern="ckpt/step_{step:06}/__{rank}_0.distcp",
                   fields={"step": int, "rank": int}, size="as_written", seed=0x5eed_da82)

with w.actor("gpu") as gpu:                              # count defaults to the reserved `gpus`
    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch, batches=P.steps) as worker:
        with worker.loop("j", P.batch):
            f = worker.let("f", train.consume())         # position = gpu + G·(b·batch + j)
            worker.open(f, "RDONLY|CLOEXEC")
            worker.read(f, P.hdr_read, repeat="until_eof")
            worker.close(f)
    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")
        with gpu.when(gpu_id == 0):
            c = gpu.let("c", ckpt.object(step=step, rank=gpu_id))
            …

if __name__ == "__main__":
    w.write()                                            # <name>.ast.json, validated, with provenance
```

**Declarations** on the `Workload`: `param(name, default, unit=, doc=, cli=)` or
`params(**defaults)`; `dataset(name, pattern=, count=, size=, seed=, access=, samples_per_file=,
chunk=, format=)`; `regions(name, file=, count=, slot=, size=, seed=)`; `namespace(name,
pattern=, fields=, size=, seed=, input=)`; `actor(name, count=)`. `w.P.x` is the parameter reference;
`P.xs[t]`, `P.xs.len`, `P.xs.sum` index and reduce a parameter array (a table is several
parallel arrays).

**Handles**: `ds.file(id)`, `ds.dir(id)`, `ds.consume()`, `ds.pick(dist=None)`, `ds.count`,
`ds.dirs`; a `regions` dataset's single file is `ds.file()`; `h.size`, `h.offset`, `h.unit`,
`h.chunks`, `h.chunk(k)`, `h.container`; `ns.object(**fields)`. Bind a handle with
`cursor.let(name, h)` before using it in ops when the same sample must flow through several
ops (a `let` is the way to use one positional draw twice).

**Statements** on a cursor: `let`, `draw(name, dist)`, `ref(name)` (a binding defined later in
the loop, for `x @ i` chains), `loop(index, to, start=, step=)` yielding the index,
`parallel(index, width)` and `loader(name, workers=, prefetch=, batches=, ordered=)` yielding a
sub-cursor with `.index`, `channel`/`put`/`take`, `barrier(scope)`, `compute(ns)`, `when(test)`
/ `otherwise()`, `every(n)`, `choose()` with `.arm(weight)`, `phase(name)`, `replay`, and the
seventeen ops: `open(f, "RDONLY|CLOEXEC", mode=, expect=)`, `close fstat stat fsync fdatasync
unlink`, `read(f, len, offset=, repeat=, expect=)`, `write`, `lseek(f, off, "SET|CUR|END")`,
`ioctl(f, "TCGETS")`, `ftruncate`, `fallocate`, `mkdir`, `rmdir`, `rename`, `readdir`.

**Expressions**: node arithmetic (`+ - * // %`, `ceil_div`, `min_`, `max_`, `-x`), comparisons
(`== != < <= > >=`), `&`, `|`, `~`, `when(test, a, b)` (arms may be expressions, distributions,
or handles), `draw(dist)`, `x.at(i)` for `x @ i`, `gpu_id`, `actor_count`. `/` is refused: say
`//` or `ceil_div`.

**Distributions**: `const uniform uniform64 normal lognormal empirical zipf hotset mixture`;
`none` is the null arm of a mixture.

**Units**: `KiB MiB GiB TiB KB MB GB`, `ns us ms s minute`; plain ints, so `105 * ms` is an int.

## 2. The one discipline

Python control flow runs at build time and generates *structure*; cursor control flow runs at
execution time and generates *iterations*. A symbolic value has no truth value, cannot be
iterated, and cannot be converted to an int: `for j in P.batch`, `if step % 500 == 0`, and
`range(P.steps)` raise a `TypeError` naming `cursor.loop` / `cursor.when`. Python loops are
still the right tool for generating four similar actors, a `choose` from a table, or the seven
fixed syscalls of a Python `open()`. The lint refuses three or more identical statements in a
row (a hand-unrolled iteration); `Workload(..., lint=False)` allows it.

Sugar that never reaches the AST: `every(n)` (a `cond` on `mod`), `repeat=n` on `read`/`write`
(a `loop` with a generated index `rep`, so a drawn length is re-drawn per repetition),
parameter tables (parallel arrays), `draw(name, dist)`, unit multipliers, and any helper
function an author writes over nodes (`chunks(f)` in `abstracts/train_large_samples.py`).
`x @ (i − d)` is spelled `cursor.ref("x").at(i - d)`; `recent` is not a construct.

Namespaces are always explicit (`w.namespace(...)` then `ns.object(...)`); the paper form's
`file("literal/pattern")` is not offered, since a namespace needs a seed and a size rule and
those belong in one place.

## 3. What fails where

| At construction (Python traceback) | At `build()` / `write()` (schema + `schema/check.py`) |
|---|---|
| Python control flow over a node; `/`; callables or foreign objects in a node | every rule in `schema/README.md` §4, on the emitted AST, with JSON-pointer paths |
| unknown flag, errno, ioctl, whence, unit; duplicate flag; `gpus` as a param | unresolved names (a `ref` never bound, an unknown dataset) |
| `x @ e` on the binding being defined, or one defined later, unless `e` is `i - d` with `d` provably ≥ 1 (V3) | `consume` outside any loop (V6) |
| `until_eof` on an `as_written` object through a different binding (V4) | AST size |
| index shadowing; `every` outside a loop; channel not declared; namespace fields vs pattern | |
| three identical sibling statements (lint) | |
| a write, truncate, unlink, rename, or write-mode `open` on a dataset handle (V12); a pattern component beginning with `.aeiou`, two datasets sharing a root, a namespace inside a dataset root (V13) | the same rules, on the emitted AST |

The builder keeps no second copy of the rules: `validate()` loads `schema/abstract-ast.schema.json`
and `schema/check.py` from the repository (or `$AEIOU_SCHEMA_DIR`) and runs them on the emitted
dict. The Rust validator must reject everything they reject.

## 4. Reproducible builds

`aeiou-build` implements techniques 4–6 of `GRAMMAR_OPTIONS.md` Option D:

- **Canonical hash.** `sha256` of the canonical JSON (sorted keys, no whitespace, floats in
  `repr`, `provenance` removed); the same function as `schema/check.py`. The on-disk file is
  the same JSON pretty-printed, and its `provenance` block carries the hash. JSON is the only
  format on the contract (`schema/README.md` §1).
- **Provenance.** `generator: {script, sha256, git, builder_version}`, `python`, `lock`
  (`uv.lock` and its hash, when present), `built_twice_identical`, `ast_sha256`. `git` is the
  short HEAD with `-dirty` when the script has uncommitted changes; the script's own `sha256`
  is the exact identity.
- **`--hermetic`.** The script runs in a child `python -s -B` with a scrubbed environment
  (`PATH`, `PYTHONHASHSEED`, `LANG` only) and PEP 578 audit hooks that deny sockets,
  subprocesses, every write, and reads outside the interpreter, its packages, the schema, and
  the script's directory. Clocks (`time.*`, `datetime.now`), entropy (`os.urandom`, `uuid`),
  and the unseeded `random` / `numpy.random` entry points are replaced with functions that
  raise `HermeticViolation`. The child returns the AST on stdout; the parent validates it again
  and writes the file.
- **`--twice`.** Two children with different `PYTHONHASHSEED` values; the hashes must agree.
  This catches set-iteration order and `id()`-based ordering without anticipating them
  (`tests/test_builder.py::test_build_twice_detects_hash_order_dependence`).
- **`--check`.** Rebuild and compare with the committed file's hash; CI runs it on every push
  so `schema/examples/*.ast.json` cannot drift from `abstracts/*.py`.

Three don'ts for authors: iterate sets or dicts keyed by nodes, use `id()`/`hash()` for
ordering, read the environment or the clock. The harness turns each into an immediate error.

## 5. Parameter files (2026-09-30)

A script's `param` defaults are placeholders or one reference set; the fitted values of a
shape live in a **parameter file** (`schema/README.md` §8, `schema/params.schema.json`), so
the shape is published once and its parameter sets separately. `aeiou-params` is the helper:

```
uv run aeiou-params defaults ../schema/examples/train_small_files.ast.json -o tsf.params.json --pin
uv run aeiou-params check ../schema/examples/train_small_files.ast.json tsf.params.json
uv run aeiou-params safetensors ../schema/examples/model_load.ast.json model-0000?-of-00004.safetensors -o llama.params.json --tp 8
uv run aeiou-params npz ../schema/examples/train_large_samples.ast.json corpus/train/00000/sample_00000000?.npz [-o corpus.params.json]
```

`npz` (2026-10-02) reads `framing` and `cd_len`, the two archive numbers `train_large_samples`
takes as parameters, from real `.npz` files with the standard library alone, and compares
them with the abstract's defaults: it prints both, exits 2 when they differ and no `-o` was
given, and writes them as a parameter file with `-o`. It refuses archives that disagree with
each other, a compressed member, a member that is not first, and a zip64 end record, since
the abstract reads none of those shapes. Run it on a few files of any corpus before fitting
an abstract to a trace over it. `tests/test_formats.py` runs the same comparison against an
archive written in memory by the installed NumPy (`DESIGN_REVIEW.md` §3.53, §3.54).

`defaults` writes every default as a set to edit or fit; `check` applies the runner's rules
(declared names, no `gpus`, same kind as the default: scalar, array of the same element kind,
or distribution); `safetensors` reads real shard headers and emits `model_load`'s tensor
table (`shard`, `off`, `bytes`, `split`, `rows`, `row_bytes`, `hdr_len`, `shards`,
`shard_bytes`, `tp`), deciding column- or row-parallel by tensor name (`--column`, `--row`,
Llama-shaped defaults) and marking the rest replicated. The runner takes the file with
`aeiou run --params FILE` (repeatable; `--param` still wins). `python -m aeiou.params` is the
same program.

## 6. Format classes and `aeiou-datagen` (2026-09-30)

A dataset stored in a container format declares its class (`aeiou.formats`, `GRAMMAR_OPTIONS.md`
§6.4, `DESIGN_REVIEW.md` §3.28), and the class supplies the three things the two-contract
split assigns to the format: the **layout** (`schema/README.md` §2 *Container layout*; the
runner computes every offset from it and stays format-ignorant), the reader library's fixed
**protocol** as ordinary ops emitted on a cursor, and the **writer** that `aeiou-datagen`
uses. The loader's choices (access mode, how many shards a worker streams, projection, batch
composition) stay in the abstract.

```python
from aeiou.formats import tfrecord, parquet, hdf5, webdataset

shards = w.dataset("shards", pattern="train/shard-{id:05}.parquet", count=P.samples,
                   samples_per_file=P.per_shard, size=lognormal(median=P.sample_median, sigma=0.45),
                   seed=0x5eed_da90, access="stream",
                   format=parquet(rows_per_group=64, columns=(("image", "binary"), ("label", "int64"))))
with gpu.loader("shards", workers=P.cycle, prefetch=1, batches=ceil_div(P.steps * P.batch, P.per_shard)) as worker:
    s = worker.let("s", shards.consume())       # under `stream`, consume draws a shard
    shards.format.open_reads(worker, s)         # two fstats, the 64 KiB footer read (and the rest if larger)
    shards.format.read_all(worker, s)           # WILLNEED + one pread per coalesced run of row groups
    shards.format.close(worker, s)
```

| Class | Reader, access | Protocol (traced 2026-09-30 unless marked) | Layout constants |
|---|---|---|---|
| `tfrecord(xfer=256 KiB)` | `tf.data.TFRecordDataset`, `stream` | `stream(cur, f)`: open, positioned reads of `xfer` to EOF [verify: from source] | 12-byte row header, 4-byte row footer |
| `parquet(rows_per_group, columns)` | `pyarrow.parquet.ParquetFile`, `stream` | `open_reads`, `read_all(columns=None)`, `close`: fstat ×2, 64 KiB footer read, `fadvise(WILLNEED)` then one `pread` per range; adjacent row groups coalesce to ≤ 32 MiB; a projection reads each run of adjacent projected chunks per group | 4-byte magic; page header 16 + 3 varints (probed); footer declared with a margin and padded exactly; one binary column carries the sample |
| `hdf5(dataset, shape, dtype)` | h5py `f[name][i]`, `map` | `open_reads`, `read_sample`, `close`: fstat, eight metadata reads, one `pread` per row through the 64 KiB sieve | data offset probed with the `core` driver; samples are fixed-size rows (`size` must be `const`) |
| `webdataset(members, xfer)` | CPython `tarfile` streaming, `stream` | `stream(cur, f)`: open, fstat, `ioctl`, `lseek`, `read(xfer)` to EOF | 512-byte member headers and alignment, 1024 footer, 10240 file alignment |

`spec(access, size, params)` is what `w.dataset(format=…)` calls; it refuses an access mode
the class does not support (V9) and records `class`, `reader`, `version` (the installed
library's, when a constant was probed from it), and `layout` with `writer` settings. The
probes write a one-row file in memory with the library; the libraries are the `formats`
extra and are pre-imported by the hermetic child. Arrow IPC, MDS, and Megatron are not
written yet (`DESIGN_REVIEW.md` §3.28 says why).

**`aeiou-datagen AST --root DIR [--params FILE]… [--param k=v]… [--gpus G] [--dedupe D]
[--compress C] [--threads N] [--dataset NAME]…** writes every dataset that has a format class:
names from the pattern, sizes from the dataset seed (`rng.py` is the runner's sampler),
bytes from `dgen-py` 0.3.0 under the `aeiou-positional/1` wrapper (bit-identical to the Rust
writer's), each file checked against the geometry its layout predicts (`layout.py`), then the
manifest `aeiou run` compares (`schema/README.md` §6; its `format` block names the writer
library). Datasets without a class are left to `aeiou datagen`, which in turn refuses the
ones with a class. `tests/test_formats.py` reads the files back with pyarrow and h5py and,
when the runner is built, runs the three container abstracts over them under the `sync`
and `io_uring` backends.

## 7. `aeiou-trace`: the metrics of a real trace (2026-10-01)

The trace side of the locality-metrics check (`GRAMMAR_OPTIONS.md` §5.4, `PROJECT_BRIEF.md`
§6 item 14). `aeiou dry-run --metrics-json` gives an abstract's numbers; this gives the same
numbers, by the definitions of `runner/README.md` §10 and in the same document
(`aeiou_metrics: 1`, with `"source": "strace"`), from an `strace` of the application the
abstract models, and compares two such documents. Standard library only (`aeiou/trace.py`);
the reasoning is in `DESIGN_REVIEW.md` §3.42. **The choices below were made while building
and confirmed by the user the same day (decided 2026-10-01).**

```
strace -f -ttt -T -yy -e trace=%file,%desc,%process -o trace.txt <command>
aeiou-trace metrics trace.txt --root /mnt/data [--exclude GLOB]… [--block BYTES] [--sample N]
            [--instance-root PID]… [--chain-gap-us US] [--cwd DIR] -o trace.metrics.json
aeiou dry-run x.ast.json --gpus 1 --params fitted.params.json --metrics-json abstract.metrics.json
aeiou-trace compare trace.metrics.json abstract.metrics.json [--template-b NAME] [--only PREFIX]… [--max-distance X]
```

What a trace needs: `-f` (threads and children), `-yy` (the path behind every descriptor,
which is how a call is known to be on the storage under test), `%process` (the `clone`s,
so descriptor tables and process trees are followed; without it a thread that uses a
descriptor it did not open is attached to the table that has that descriptor and path, and
the document notes `threads_without_clone`). `-ttt -T` are needed only for `--chain-gap-us`.

- **Which calls count.** Calls on paths under `--root`, less `--exclude` (a glob on the
  part below the root). Library loading, `/proc`, pipes, and sockets are not the workload.
- **Calls to ops.** `open`/`openat`/`creat` → `open`; `read`, `pread64`, `readv`, `preadv`
  → `read` and the same for `write`; `fstat`, and `statx`/`newfstatat` on a descriptor →
  `fstat`, on a path → `stat`; `getdents64` → `readdir`, once per listing (the calls that
  continue a listing and the empty one that ends it are the same op, as the abstract's
  `readdir` is one listing); `unlinkat` with `AT_REMOVEDIR` → `rmdir`; `lseek`, `ioctl`,
  `fsync`, `fdatasync`, `ftruncate`, `fallocate`, `mkdir`, `rename`, `fadvise64`, `close`
  by name. A failed call counts as its op, as an op the abstract expects to fail does.
- **Offsets.** `pread`/`pwrite` carry theirs. `read`/`write` use the position of the open
  file description, tracked from `open`, `lseek`, and the bytes each call moved, shared by
  `dup`ed descriptors, by threads, and across `fork`. An `O_APPEND` write to a file not
  created in the trace has an unknown base (noted; counted from 0).
- **Order.** The order in which calls returned (the order of the trace's lines). A trace
  has a real order; the dry run has the round-robin of §10. Only reuse distance depends on it.
- **Instance.** The whole trace is one instance; `--instance-root PID` (repeatable) makes
  each named process tree one, and leaves out what is under none. Reuse distance is per
  instance, as on the dry-run side; popularity is over all of them.
- **Sequential context.** A thread. A run is the consecutive ops of one kind by one thread
  on one path, ended by a gap, the thread's `close` of the path, or the thread's exit.
- **Fan-out and depth.** Only where the trace shows a fork: the requests of one
  `io_submit` (on paths under the root) are a fan-out; their sizes and offsets are in the
  call, their results in the `io_getevents` that reaps them (matched by context and
  `aio_data`; a request never seen reaped is taken as complete and noted). Consecutive
  `io_submit`s of a thread are a chain until the thread issues another counted call or
  more than `--chain-gap-us` passes between a round's end and the next submit. **There is
  no default gap**: without the flag no depth is reported, because the think time that
  separates one search from the next is a property of the application. A beam search done
  by a thread pool of `pread`s has no fork in the trace: its fan-out and depth are not
  recoverable, and that row of the comparison says `none`.
- **Not visible to `strace`:** `io_uring` submissions and page faults on a mapping. Not
  counted: `sendfile`, `splice`, `copy_file_range`.
- **`compare`.** One row per metric with both values and a distance in [0, 1]: the
  difference of two shares (mix, ops by kind, first touches, bytes in multi-op runs,
  popularity of the top 0.1/1/10 %), the largest difference between two histograms'
  cumulative shares over the buckets of both (request size, run length, reuse distance),
  the total-variation distance of two exact distributions (fan-out, depth).
  `--max-distance X` exits 1 when a row exceeds it; there is no default, since the
  tolerances for accepting an abstract are not set (`PROJECT_BRIEF.md` §6 item 14).

**Checked against the runner (2026-10-01).** The runner under `strace` is an application
whose abstract is known, so the trace's numbers must be the dry run's wherever order does
not enter. `tests/test_trace.py` runs `train_small_files`, `kv_cache_serving`, and
`vdb_search_diskann` (one GPU, `sync` backend) that way: read and write counts, bytes,
request sizes, run lengths, block and object popularity, and first touches are equal;
reuse distances differ by the order alone (CDF distance 0.07 to 0.20 on these small
configurations). Fan-out and depth are `none` on the trace side, as said above: the
runner's sub-actors are threads.

**Checked against a real application (2026-10-01).** `traces/train_small_files` holds the
first row of the capture plan (`ABSTRACTS.md` §11): PyTorch `DataLoader` + `ImageFolder`
on NFS, the script that was traced, the corpus writer, the fitted parameter file, and the
trace's metrics document. The trace corrected the abstract (a second `lseek` per file,
`stat` and `fstat` per directory in the walk); `tests/test_trace.py` now holds the abstract
at the fitted parameters to that document. Findings in `ABSTRACTS.md` §1, reasoning in
`DESIGN_REVIEW.md` §3.43 (decided 2026-10-01). `traces/train_large_samples` is row 2,
`np.load(...)["x"]` as upstream DLIO and any NumPy user issues it: that abstract was rewritten from its trace
(`ABSTRACTS.md` §2, `DESIGN_REVIEW.md` §3.44; explicit seeks decided 2026-10-01; reviewed 2026-10-02, §3.53 and §3.54, all choices decided: the script's `glob` is the phase `enumerate`, on by default, and `tests/test_formats.py` checks `framing` against the installed NumPy).
`traces/ckpt_write_dcp` is row 3, `torch.distributed.checkpoint.save` on two ranks
(`ABSTRACTS.md` §3, `DESIGN_REVIEW.md` §3.45, decided 2026-10-02).
`traces/ckpt_restore` and `traces/model_load` are row 4: `torch.distributed.checkpoint.load`
on two ranks, whose trace rewrote the restore's item loop, and safetensors `from_pretrained`,
which maps the shards and issues no `read` (`ABSTRACTS.md` §4, `DESIGN_REVIEW.md` §3.47, not
yet confirmed). A third `dcp.load` trace (`name-order`) showed that the reader takes the
items in the sorted order of their names and that an item's cost depends on where the
buffer was left; the restore carries the buffer's start as a chain since (§3.48, decided
2026-10-01), and `traces/ckpt_restore/params.py` makes the parameter file from a real
checkpoint's `.metadata`.
`traces/vdb_search_ivf` is row 6: FAISS `IndexIVFPQ` over `OnDiskInvertedLists`, which maps
the lists file and issues no call on it; its `faults.py` measures with `mincore` what the
trace cannot show (`ABSTRACTS.md` §6, `DESIGN_REVIEW.md` §3.49, decided 2026-10-01).
`traces/vdb_build_diskann` and `traces/vdb_search_diskann` are rows 7 and 5: DiskANN through
`diskannpy`, one index built in 13 shards and then searched; `hops.py` reads the beam search
(rounds, batch sizes, sector spread) off the `io_submit` lines (`ABSTRACTS.md` §5 and §7,
`DESIGN_REVIEW.md` §3.50, decided 2026-10-02).
`traces/kv_cache_serving` is row 8: vLLM with LMCache's local-disk backend on a GPU, with a
synthetic chat load (`chat.py`); it fixes the call sequence, not the distributions
(`ABSTRACTS.md` §8, `DESIGN_REVIEW.md` §3.51, decided 2026-10-02).
`traces/kv_cache_shared` is the same load on LMCache's `fs://` backend, twice: an engine on
an empty store, then a restarted engine on the filled one, sent the same requests
(`chat.py --save`, `--replay`). `abstracts/kv_cache_shared.py` emits both abstracts,
`kv_cache_shared` and `kv_cache_shared_reader`; the reader takes the writer's namespace as
`input` and `same_run` (contract 0.4: `w.namespace(..., input=True, same_run=True)`), so a run
without the writer's `--seed`, `--gpus`, and parameters is refused before the gate
(`DESIGN_REVIEW.md` §3.52, decided 2026-10-02).

**The application's API is declared in the script** (contract 0.3, 2026-10-01):
`Workload("model_load", backend="mmap")`. Leave it out for an application that calls `read`
and `write`. The runner uses it as the default backend (`runner/README.md` §4).

## 8. Not yet

- The `replay` trace format (schema §6); `cursor.replay` emits the node only.
- Format classes for Arrow IPC, MDS, and Megatron `.bin`/`.idx`; the Parquet→Arrow conversion
  abstract (`GRAMMAR_OPTIONS.md` §6.5).
- The `strace` → parameter fitting tool (`aeiou-fit`, `ABSTRACTS.md` §11), which will write
  parameter files (`aeiou-trace` of §7 reads the trace and measures; it fits nothing); the offline content verifier (`aeiou-verify`), which `rng.py` and
  `layout.py` now make a short tool.
