# `aeiou`: the Python builder

Layer 1 of `GRAMMAR_OPTIONS.md` Option D (chosen 2026-09-30). The package is named for the
repository, not for MLPerf Storage: nothing in it is specific to the working group's process
(`PROJECT_BRIEF.md` §8). An authoring script constructs
the AST of `schema/abstract-ast.schema.json` by running once; the calls that look like I/O are
declarations. The AST is the only thing the Rust runner executes and the only thing an author
publishes with a hash. Nothing here runs on a client node.

```
builder/
  aeiou/     the package: nodes, dists, units, builder (Workload, Cursor), validate, emit, hermetic,
             cli, params (parameter files, `aeiou-params`), formats (the format classes), datagen
             (`aeiou-datagen`, deprecated, §6) and coord (its coordinator protocol), header
             (`aeiou-header`, §6), trace (`aeiou-trace`, §7), catalog (`aeiou-catalog`, §8) and
             describe (its descriptions), options and usage (the runner's option layers and
             usage frame), and the ports of the runner's definitions it needs: rng, pattern, framing
  tests/             pytest: every abstract builds, matches its committed AST, the discipline holds;
                     the two checkers refuse alike; the parameter files fit their abstracts; the
                     format classes write files the libraries read back and the runner executes to
                     the dry-run fingerprint, and `aeiou datagen` writes the same bytes; the trace
                     metrics against hand-written traces and against the runner under `strace`; the
                     kit scripts; the catalog; the manual pages against `--help`
  pyproject.toml     dep: jsonschema; extras `test`, `formats` (pyarrow, h5py, numpy, crc32c,
                     xxhash), and `s3` (obstore); `aeiou-build`, `aeiou-params`, `aeiou-datagen`,
                     `aeiou-trace`, `aeiou-catalog`, `aeiou-header` entry points
```

```
./dev.sh setup && cd builder     # the venv (and the runner) at the checkout's pins (DESIGN_REVIEW.md §3.82, §3.83)
.venv/bin/aeiou-build --hermetic --twice $(find ../workloads -name '*.py' -not -path '*/kit/*')   # each AST beside its script
.venv/bin/pytest
.venv/bin/aeiou-build --check $(find ../workloads -name '*.py' -not -path '*/kit/*')     # drift check (CI)
.venv/bin/aeiou-params check ../workloads/model-loading/safetensors-load/safetensors_load.ast.json ../workloads/model-loading/safetensors-load/params/synthetic.params.json
.venv/bin/aeiou-catalog check --workloads ../workloads --runner ../runner/target/release/aeiou   # the catalog (CI)
```

The authoring scripts and their ASTs live under `workloads/` (`DESIGN_REVIEW.md` §3.66), one
directory per I/O mechanism.

## 1. Writing an abstract

```python
from aeiou import *

w = Workload("file_per_sample", doc="…")
P = w.P
w.param("batch", kind=int, unit="count")                 # required: each use case gives it
w.param("step_time", kind=int, unit="ns", doc="[measure]")
w.param("samples", 50_000_000, unit="count", scale=True)   # sized per site by a submission
w.param("hdr_read", st_blksize(of="train", fallback=1 * MiB), unit="bytes")   # what the storage reports
w.param("reuse", mixture((0.55, none), (0.45, lognormal(median=40, sigma=1.2, min=1))))

train = w.dataset("train", pattern="train/{id div 1300:05}/img_{id:09}.jpg", layout="whole", access="map",
                  count=50_000_000, size=lognormal(median=110 * KiB, sigma=0.45), seed=0x5eed_da7a)
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

A parameter (contract 0.11, `schema/README.md` §1, `DESIGN_REVIEW.md` §3.78) has a default,
the value of a real fact of the layer that sets it (a library's chunk size), or is
**required**: `kind=int`, `float`, `bool`, `str`, `Dist`, or `list[int]` and the like, and no
default, for a value that varies with every use case (the job's batch and step time), so a run
that forgets it is refused instead of running another job. `scale=True` marks what a
submission sizes per site. `st_blksize(of=NAME, fallback=BYTES)` is a default the storage
reports: `aeiou run` reads it by `stat` of the dataset's first file or (contract 0.12) of a
namespace's, an input one's first object or a probe file in an output one's root (a shard's
`BufferedWriter`, `dcp_save`'s `buf`), `aeiou dry-run` takes it with `--st-blksize`,
`fallback` where the storage reports none. A parameter's name never
has a dot (an identifier), so it never takes a runner fact's or result's name.

**Declarations** on the `Workload`: `param(name, default, unit=, doc=, cli=)` or
`params(**defaults)`; `dataset(name, pattern=, layout=, access=, count=, size=, seed=,
samples_per_file=, chunk=, slot=, format=, inventory=)` (contract 0.8, `DESIGN_REVIEW.md` §3.72: `layout`
`whole`, `chunked` with `chunk`, `compound` with `samples_per_file` and optionally `format`,
`array` with `slot` and a pattern without fields; `access` `map`, `stream`, or `indexed`; the
builder refuses what V20 and V21 refuse, and `consume()` on an `indexed` dataset; `regions()`
is gone, an array being `dataset(…, layout="array", access="indexed", slot=…)`; contract 0.9,
§3.73: `inventory={"name": "inventory.txt", "entry": P.inv_entry}` declares a consumed dataset's
file list, refused as V22 refuses; contract 0.10, §3.75: a pattern's divisor may name a
parameter declared before it with an integer default, `{id div files_per_directory:05}`, refused
as V23 refuses; the training abstracts' layout parameters are `samples`, `samples_per_file`, and
`files_per_directory`, `schema/README.md` §2; also `protocol=` and `doc=`); `namespace(name,
pattern=, fields=, size=, seed=, input=, same_run=, protocol=, doc=)`; `actor(name, count=, doc=)`. `w.P.x` is the parameter reference;
`P.xs[t]`, `P.xs.len`, `P.xs.sum` index and reduce a parameter array (a table is several
parallel arrays).

**Handles**: `ds.file(id)`, `ds.dir(id)`, `ds.inventory()` (the dataset's inventory, contract 0.9), `ds.consume()`, `ds.pick(dist=None)`, `ds.count`,
`ds.dirs`; an `array` dataset's single file is `ds.file()`; `h.size`, `h.offset`, `h.unit`,
`h.chunks`, `h.chunk(k)`, `h.container`; `ns.object(**fields)`. Bind a handle with
`cursor.let(name, h)` before using it in ops when the same sample must flow through several
ops (a `let` is the way to use one positional draw twice).

**Statements** on a cursor: `let`, `draw(name, dist)`, `ref(name)` (a binding defined later in
the loop, for `x @ i` chains), `loop(index, to, start=, step=)` yielding the index,
`parallel(index, width)` and `loader(name, workers=, prefetch=, batches=, ordered=)` yielding a
sub-cursor with `.index`, `channel`/`put`/`take`, `barrier(scope)`, `compute(ns)`, `when(test)`
/ `otherwise()`, `every(n)`, `choose()` with `.arm(weight)`, `phase(name)`, `trace(file, sha256)`, and the
eighteen ops: `open(f, "RDONLY|CLOEXEC", mode=, expect=)`, `close fstat stat fsync fdatasync
unlink`, `read(f, len, offset=, repeat=, expect=)`, `write`, `lseek(f, off, "SET|CUR|END")`,
`ioctl(f, "TCGETS")`, `fadvise(f, "WILLNEED", offset=, len=)`, `ftruncate`, `fallocate`,
`mkdir`, `rmdir`, `rename`, `readdir`.

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
function an author writes over nodes (`chunks(f)` in `workloads/input-pipeline/map/npy-per-sample/npy_per_sample.py`).
`x @ (i − d)` is spelled `cursor.ref("x").at(i - d)`; `recent` is not a construct.

Namespaces are always explicit (`w.namespace(...)` then `ns.object(...)`); the paper form's
`file("literal/pattern")` is not offered, since a namespace needs a seed and a size rule and
those belong in one place.

## 3. What fails where

| At construction (Python traceback) | At `build()` / `write()` (schema + `schema/check.py`) |
|---|---|
| Python control flow over a node; `/`; callables or foreign objects in a node | every rule in `schema/README.md` §4, on the emitted AST, with JSON-pointer paths |
| unknown flag, errno, ioctl, whence, unit; duplicate flag; `gpus` as a param | unresolved names (a `ref` never bound, an unknown dataset) |
| `x @ e` on a binding of the same loop body (itself included) unless `e` is `i - d` with `d` provably ≥ 1: a literal, a draw with that minimum, or `add` of such a term and one provably ≥ 0 (a `mod`, a loop index, a draw with minimum ≥ 0; contract 0.5; a required parameter passes and its value is judged when given, 0.13) (V3) | `consume` outside any loop, a second `consume` of one dataset, or a `consume` under a loop, `parallel`, or `loader` whose count varies with an enclosing loop (V6) |
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
  (`PATH`, `PYTHONHASHSEED`, and `LANG` and `LC_ALL` set to `C.UTF-8`, only) and PEP 578 audit hooks that deny sockets,
  subprocesses, every write, and reads outside the interpreter, its packages, the schema, and
  the script's directory. Clocks (every `time.*` clock, `clock_gettime` included, `os.times`,
  `resource.getrusage`, `datetime.now`), entropy (`os.urandom`, `uuid`),
  and the unseeded `random` / `numpy.random` entry points are replaced with functions that
  raise `HermeticViolation`. The child returns the AST on stdout; the parent validates it again
  and writes the file.
- **`--twice`.** Two children with different `PYTHONHASHSEED` values; the hashes must agree.
  This catches set-iteration order, `id()`-based ordering, and the pid (which `--hermetic`
  leaves readable) without anticipating them
  (`tests/test_builder.py::test_build_twice_detects_hash_order_dependence`).
- **`--check`.** Rebuild and compare with the committed file's hash; CI runs it on every push
  so no committed `*.ast.json` under `workloads/` can drift from the script beside it.

Three don'ts for authors: iterate sets or dicts keyed by nodes, use `id()`/`hash()` for
ordering, read the environment or the clock. The harness turns each into an immediate error.

## 5. Parameter files (2026-09-30)

A script's `param` defaults are placeholders or one reference set; the fitted values of a
shape live in a **parameter file** (`schema/README.md` §8, `schema/params.schema.json`), so
the shape is published once and its parameter sets separately. `aeiou-params` is the helper:

```
.venv/bin/aeiou-params defaults ../workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json -o tsf.params.json --pin
.venv/bin/aeiou-params check ../workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json tsf.params.json
.venv/bin/aeiou-params safetensors ../workloads/model-loading/safetensors-load/safetensors_load.ast.json model-0000?-of-00004.safetensors -o llama.params.json --tp 8
.venv/bin/aeiou-params npz ../workloads/input-pipeline/map/npy-per-sample/npy_per_sample.ast.json corpus/train/00000/sample_00000000?.npz [-o corpus.params.json]
```

`npz` (2026-10-02) reads `framing` and `cd_len`, the two archive numbers `npy_per_sample`
takes as parameters, from real `.npz` files with the standard library alone, and compares
them with the abstract's defaults: it prints both, ~~exits 2~~ exits 1 (2026-10-04; 2 is the
suite's usage-error status, `runner/REFERENCE.md` §15) when they differ and no `-o` was given, and writes them as a parameter file with `-o`. It refuses archives that disagree with
each other, a compressed member, a member that is not first, and a zip64 end record, since
the abstract reads none of those shapes. Run it on a few files of any corpus before fitting
an abstract to a trace over it. `tests/test_formats.py` runs the same comparison against an
archive written in memory by the installed NumPy (`DESIGN_REVIEW.md` §3.53, §3.54).

`defaults` writes every default as a set to edit or fit; `check` applies the runner's rules
(declared names, no `gpus`, same kind as the default: scalar, array of the same element kind,
or distribution); `safetensors` reads real shard headers and emits `safetensors_load`'s tensor
table (`shard`, `off`, `bytes`, `split`, `rows`, `row_bytes`, `hdr_len`, `shards`,
`shard_bytes`, `tp`), deciding column- or row-parallel by tensor name (`--column`, `--row`,
Llama-shaped defaults) and marking the rest replicated. The runner takes the file with
`aeiou run --params-file FILE` (repeatable; `--param` still wins). `python -m aeiou.params` is the
same program.

**`params_version` 2 and the lock** (contract 0.11, `DESIGN_REVIEW.md` §3.78, 2026-10-22).
Every entry of a file carries its state, `{"default": v}`, `{"locked": v}`, or `{"required":
true}`, and a file may hold a `run` table of the runner's identity options; `values_of(pset)`
is a set's values, `entry(e)` one entry's state and value. A **use case** is a case directory
whose `case.toml` lists, for each `[[run]]`, the abstract (`ast`) and its fragments (`params`,
a file or a list of them); `aeiou-params lock CASE [--rules FILE]` (`aeiou/lock.py`)
resolves it into `CASE/case.lock.json`:

```
.venv/bin/aeiou-params lock ../workloads/input-pipeline/map/npy-per-sample/cases/npy-8mib
.venv/bin/aeiou-params lock ../workloads/…/cases/unet3d-h100 --rules bench.rules.json -o bench-unet3d-h100.lock.json
```

The abstract gives each parameter its first state (its default, or required); the fragments
of one run combine by agreement, in any order (two that set a name to different entries are
refused, both files named); a rules file freezes by state (`defaults: locked`, `scale:
required`, `identity_options: locked`) and its constraints, infix text over `fact.` and
`result.` names (`lock.parse_constraint`), are copied in with their expression form. The lock
pins each run's abstract by its path below `workloads/` and its `ast_sha256`, keeps every
value's source, and records the fragments and the rules file with their hashes; it is
rendered one value to a line so its diff is a line per change, and it never carries a
version of the tool, so a release does not make every lock stale. `lock.stale(case)` says why
a committed lock is not the one its inputs make now; `aeiou-build` re-locks the committed cases
of every abstract it writes (`cli._relock`), and leaves a case without a lock to its author.
The runner computes a lock's workload id (`runner/REFERENCE.md`); the Python side never does,
so the id has one definition.

## 6. Format classes and `aeiou-datagen` (2026-09-30)

A dataset stored in a container format declares its class (`aeiou.formats`, `GRAMMAR_OPTIONS.md`
§6.4, `DESIGN_REVIEW.md` §3.28), and the class supplies the three things the two-contract
split assigns to the format: the **framing** (`schema/README.md` §2 *Container framing*, `layout` until contract 0.8; the
runner computes every offset from it and stays format-ignorant), the reader library's fixed
**protocol** as ordinary ops emitted on a cursor, and the **writer** that `aeiou-datagen`
uses. The loader's choices (access mode, how many shards a worker streams, projection, batch
composition) stay in the abstract.

```python
from aeiou.formats import tfrecord, parquet, hdf5, webdataset

shards = w.dataset("shards", pattern="train/shard-{id:05}.parquet", layout="compound", access="stream",
                   count=P.samples, samples_per_file=P.samples_per_file,
                   size=lognormal(median=P.sample_median, sigma=0.45), seed=0x5eed_da90,
                   format=parquet(rows_per_group=64, columns=(("image", "binary"), ("label", "int64"))))
with gpu.loader("shards", workers=P.cycle, prefetch=1, batches=ceil_div(P.steps * P.batch, P.samples_per_file)) as worker:
    s = worker.let("s", shards.consume())       # under `stream`, consume draws a shard
    shards.format.open_reads(worker, s)         # two fstats, the 64 KiB footer read (and the rest if larger)
    shards.format.read_all(worker, s)           # WILLNEED + one pread per coalesced run of row groups
    shards.format.close(worker, s)
```

| Class | Reader, access | Protocol (traced 2026-09-30 unless marked) | Layout constants |
|---|---|---|---|
| `tfrecord(xfer=256 KiB)` | `tf.data.TFRecordDataset`, `stream` | `stream(cur, f)`: open, positioned reads of `xfer` to EOF [verify: from source] | 12-byte row header, 4-byte row footer |
| `parquet(rows_per_group, columns, band_sample_bytes=None)` | `pyarrow.parquet.ParquetFile`, `stream` | `open_reads`, `read_all(columns=None)`, `close`: fstat ×2, 64 KiB footer read, `fadvise(WILLNEED)` then one `pread` per range; adjacent row groups coalesce to ≤ 32 MiB; a projection reads each run of adjacent projected chunks per group | 4-byte magic; page header 16 + 3 varints (probed), their widths the varint band of the chunk the mean sample size gives, or `band_sample_bytes` when the size names a required parameter (no mean at build time; a chunk outside the band is refused when written); footer declared with a margin and padded exactly; one binary column carries the sample |
| `hdf5(dataset, shape, dtype)` | h5py `f[name][i]`, `map` | `open_reads`, `read_sample`, `close`: fstat, eight metadata reads, one `pread` per row through the 64 KiB sieve | data offset probed with the `core` driver, the header from a sparse scratch file at write time; samples are fixed-size rows (`size` must be `const`) |
| `webdataset(members, xfer)` | CPython `tarfile` streaming, `stream` | `stream(cur, f)`: open, fstat, `ioctl`, `lseek`, `read(xfer)` to EOF | 512-byte member headers and alignment, 1024 footer, 10240 file alignment |

`spec(access, size, params)` is what `w.dataset(format=…)` calls; it refuses an access mode
the class does not support (V9) and records `class`, `reader`, `version` (the installed
library's, when a constant was probed from it), and `framing` with `writer` settings. The
probes write a one-row file in memory with the library; the libraries are the `formats`
extra and are pre-imported by the hermetic child. Arrow IPC, MDS, and Megatron are not
written yet (`DESIGN_REVIEW.md` §3.28 says why).

**Deprecated (2026-10-14, `PROJECT_BRIEF.md` §6 item 32).** `aeiou datagen`, built with the
cargo feature `formats` as the release is, writes every format class (TFRecord, WebDataset,
HDF5, and Parquet), to directories and object stores, faster, and alone has `--resume`,
`--grow`, `--shrink`, and `--settle`. `aeiou-datagen` gets no new features and stays, until it
is retired, the byte and behaviour reference the tests compare the Rust writer with.

**`aeiou-datagen AST --root DIR [--endpoint NAME=DIR]… [--params-file FILE]… [--param k=v]… [--gpus G] [--dedupe D]
[--compress C] [--writers N] [--dataset NAME]… [--ranks R --rank r --coordinator HOST:PORT] [--config FILE]** writes every dataset that has a format class:
names from the pattern, sizes from the dataset seed (`rng.py` is the runner's sampler),
bytes under `aeiou-positional/4` (numpy's `SFC64` in 4 KiB pieces, `payload_block`; each
sample's bytes the unit's at the sample's file offset, the unit `dedupe_unit`;
bit-identical to the Rust writer's, `tests/test_payload.py`), each file checked against the geometry its framing predicts (`framing.py`), then the
manifest `aeiou run` compares (`schema/README.md` §6; its `format` block names the writer
library). Datasets without a class are left to `aeiou datagen`, which in turn refuses the
ones with a class (since 2026-10-12, built with the cargo feature `formats`, it writes TFRecord,
WebDataset, HDF5, and Parquet itself, byte for byte this writer's files, `tests/test_formats.py` and
`tests/test_object.py` comparing them; `runner/REFERENCE.md` §5). For HDF5 it runs
**`aeiou-header --format JSON --rows N…`** (`aeiou/header.py`, `man/aeiou-header.1.md`),
`Hdf5.header` for each row count, the headers back to back on stdout and nothing else there,
refused when the installed h5py and libhdf5 differ from the format entry's `version`; the
release ships it as one PyInstaller file beside `aeiou`. `tests/test_formats.py` reads the files back with pyarrow and h5py and,
when the runner is built, runs the three container abstracts over them under the `sync`
and `io_uring` backends. `--root`, `--endpoint`, `--writers`, `--rank`, `--ranks`, and
`--coordinator` come through the runner's option layers
(`aeiou/options.py` mirrors `runner/aeiou/src/options.rs`, `runner/REFERENCE.md` §14: the
command line, else `AEIOU_<FLAG>`, else the `[datagen]` table of the TOML file
`--config` or `AEIOU_CONFIG` names, else the default), the rest is the command line's alone,
and the block of what was resolved, with each value's source, is printed as `aeiou datagen`
prints it (`tests/test_options.py`; 2026-10-04). The writers (`--writers`, files written at once;
the name since 2026-10-11, `--threads` before) are processes, not threads (2026-10-11,
`DESIGN_REVIEW.md` §3.69): a `forkserver` that imported the libraries once
forks them, each rebuilds its `Job` (and opens its own object store) from what pickles, and
`map` hands them file ids; one writer, or one file, runs in the process itself.

**On several hosts (2026-10-11, `DESIGN_REVIEW.md` §3.69).** `--ranks R --rank r
--coordinator HOST:PORT`, as `aeiou datagen` has them (`runner/REFERENCE.md` §5, §6), and
started the same way (`aeiou-launch … -- aeiou-datagen …`). `aeiou/coord.py` speaks the
runner's coordinator protocol: a frame is a big-endian `u32` length and a JSON body, the
messages serde's externally tagged form of `coord.rs`'s `Msg`, of which a datagen uses
`Hello`, `Welcome`, `Ready`, `Start`, `Done`, `Result`, `Stop`, and `Heartbeat`, with the
runner's windows (120 s to connect, a heartbeat after 5 s idle, a peer silent for 30 s
gone). Rank 0 runs the `Server` on threads and connects to it as every rank does. The
identity is the abstract's hash, `--gpus`, the parameters, `--dedupe`, `--compress`, and
`--dataset`, compared by its canonical hash and, when it differs, field by field. The order
is the runner's: every rank checks its roots are empty and makes them, the gate opens once
every rank is ready, rank *r* writes ids `[r·⌈N/R⌉, (r+1)·⌈N/R⌉)` of every dataset with
its writers, `syncfs`es the file systems it wrote, and sends its counts (`Done`); rank 0
gathers them, writes the manifests with `ranks` and `hosts` (each rank's host, files,
bytes, start, and end) in the provenance, and sends the verdict every rank exits with. A
failure on any host stops every host. On one host the provenance has `ranks: 1` and one
entry in `hosts`, and the manifests are written after every dataset, not after each.
`tests/test_coord.py`: four ranks over six files (one slice empty) write the one host's
tree byte for byte with the same id; a rank with another `--dedupe` fails both before a
byte is written; the server's refusals in process; and a Python rank's `Hello` read and
refused by `aeiou datagen`'s coordinator, naming the difference, so the frames are the
runner's. `tests/test_object.py`: two ranks into one bucket prefix write the objects
of one host, with the same id. A Python rank cannot join a Rust corpus: the identities differ (the runner's
parameter document is not the Python's), and the two tools write different datasets.

**To an object store (2026-10-11, `DESIGN_REVIEW.md` §3.69).** A writer writes into a byte
stream, never a path: `write_file(out, first_id, sizes, payload, geo)` takes a `Sink` (it
counts the bytes and answers `tell`, for `tarfile` and pyarrow) over an open file or an
upload, and returns what `check_file(name, written, sizes, geo, info)` needs beyond the
count (Parquet: the footer's metadata as the writer wrote it, where the check read the file
back before). For a dataset declared `protocol: s3`, `--endpoint NAME=s3://BUCKET[/PREFIX]`
places its root at a prefix as the runner places it (`datagen.placement` mirrors
`endpoint.rs`, its refusals word for word), the store from the runner's `AWS_*` variables
through `obstore` (the `s3` extra; the Python bindings of the runner's `object_store`), and
each file goes out as one `PUT` or a multipart upload in 8 MiB parts, each sent before the
next is filled, as `aeiou datagen` sends them. The check runs before the close, so a file
that fails it, like any exception, aborts its upload and leaves no object; the prefix must
hold no key, and the manifest is one `PUT` at it. Every writer streams, so a thread holds a
piece of its file, never the file (`DESIGN_REVIEW.md` §3.69, revised the same day). TFRecord
and tar append. Parquet writes one row group at a time, twice: the first pass, of zero bytes
in the samples' sizes into a counting `Sink`, measures the natural footer (its bytes do not
depend on the samples' bytes, since no statistics are written) so the second can pad it to
the declared size; pyarrow holds about three copies of a row group while it writes it (one
page per column chunk). HDF5 cannot be written into a stream by libhdf5 (it seeks back over
its metadata), and h5py's Python file-object driver lays the file out otherwise (the default
shape's data at 1400, not 2048), so `Hdf5.header(n)` takes the file's first `data_offset`
bytes from libhdf5: the same dataset of `n` rows created in a sparse scratch file (in
`/dev/shm` where there is one), one row stored, the metadata read back, the file removed,
cached per row count (a dataset has at most two). A contiguous dataset with no filter and
no fill has metadata independent of the rows' bytes, so the header followed by the rows is
byte for byte the file libhdf5 writes with every row (an 18 GiB file costs 260 KiB of
scratch). `tests/test_formats.py` compares both writers with their library handed every row
at once and checks that writing two 256 MiB files raises the peak memory by far less than
either. `tests/test_object.py` (MinIO from `TEST_MINIO_BIN`, skipped without it) writes each
class into a bucket, compares every object with the directory writer's file of the same
name and the manifests' ids, runs the abstracts over the bucket to the dry run's fingerprint
under `--s3 blocking` and `async` when the runner is built with `object`, and checks a
three-part upload, an aborted one, and the placement refusals.

**Usage errors (2026-10-04).** The Python tools report a wrong command line in the
suite's one frame (`aeiou/usage.py`, the mirror of `runner/aeiou/src/usage.rs`;
`runner/REFERENCE.md` §15): the command as typed, the message, the usage line, the pointer to
the help, exit status 2; every argument the command still lacks is listed at once, from any
layer, and a builder script or a missing file given as the abstract says what an abstract is.
`usage.Parser` is argparse in the runner's look, so `--help` reads the same as `aeiou
run --help`. `tests/test_usage.py` covers the frame of every tool and, when the runner is
built, that `aeiou datagen` and `aeiou-datagen` print the same bytes for the same mistake.

## 7. `aeiou-trace`: the metrics of a real trace (2026-10-01)

The trace side of the locality-metrics check (`GRAMMAR_OPTIONS.md` §5.4, `PROJECT_BRIEF.md`
§6 item 14). `aeiou dry-run --metrics-json` gives an abstract's numbers; this gives the same
numbers, by the definitions of `runner/REFERENCE.md` §10 and in the same document
(~~`aeiou_metrics: 1`~~ `aeiou_metrics: 2` since 2026-10-22, reuse and popularity over extents
and, under `--metrics-block`, blocks, `DESIGN_REVIEW.md` §3.79; with `"source": "strace"`), from an `strace` of the application the
abstract models, and compares two such documents. Standard library only (`aeiou/trace.py`);
the reasoning is in `DESIGN_REVIEW.md` §3.42. **The choices below were made while building
and confirmed by the user the same day (decided 2026-10-01).**

```
strace -f -ttt -T -yy -v -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt <command>
aeiou-trace metrics trace.txt --root /mnt/data [--exclude GLOB]… [--metrics-block BYTES]
            [--instance-root PID]… [--chain-gap-us US] [--cwd DIR] -o trace.metrics.json
aeiou dry-run x.ast.json --gpus 1 --params-file fitted.params.json --metrics-json abstract.metrics.json
aeiou-trace compare trace.metrics.json abstract.metrics.json [--template-b NAME] [--only PREFIX]… [--max-distance X]
aeiou-trace compare trace.metrics.json abstract.metrics.json --judge [--self other-seed.metrics.json]… [--tolerances FILE]
```

What a trace needs: `-f` (threads and children), `-yy` (the path behind every descriptor,
which is how a call is known to be on the storage under test), `%process` (the `clone`s,
so descriptor tables and process trees are followed; without it a thread that uses a
descriptor it did not open is attached to the table that has that descriptor and path, and
the document notes `threads_without_clone`). `-ttt -T` are needed only for `--chain-gap-us`.
The AIO calls are named (`io_setup`, `io_submit`, `io_getevents`, `io_pgetevents`,
`io_destroy`): `strace` classes none of them in `%desc`, and a trace without them loses
every libaio read and write with nothing to show for it. A context the trace shows
`io_setup` make and no `io_submit` name is noted (`aio_contexts_without_io_submit`): the
application never used it, or the trace left `io_submit` out (fixed 2026-10-14). `-v`,
because without it `strace` prints only the first 32 elements of an array: the requests of
an `io_submit` past the 32nd, the events of an `io_getevents`, and the iovecs of a `readv`
or `writev` are cut off (`...]`). A cut array is noted, and `arrays_truncated` says to trace
again: `aio_requests_truncated` counts the requests the return value says were submitted
and the trace does not show (not counted: their file, offset, and size are not in the
trace, and the fan-out is short by them), `aio_events_truncated` the results it does not
show (their requests are taken as complete at the end), and `iov_arrays_truncated` the
vector calls whose length is taken as the larger of the shown iovecs' sum and the bytes
the call moved (fixed 2026-10-14).

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
  `dup`ed descriptors, by threads, and across `fork`. An `O_APPEND` write goes to the end
  of the file wherever the trace knows the file's size: from an open that truncated or
  exclusively created it, or a `creat`, kept through `ftruncate`, `truncate`, `fallocate`,
  every write that extends it, and a `rename`, and lost at an `unlink` (2026-10-14, RC1
  review). An `O_APPEND` write to a file whose size the trace cannot know has an unknown
  base (`append_to_unknown_size`, per open; counted from 0).
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
  `--max-distance X` exits 1 when a row exceeds it; ~~there is no default, since the
  tolerances for accepting an abstract are not set (`PROJECT_BRIEF.md` §6 item 14).~~ it is
  one number for every row; the rule by class is `--judge`, below (2026-10-02).

**Tolerances: `compare --judge` (added 2026-10-02; `DESIGN_REVIEW.md` §3.55;
decided 2026-10-02: the class values and the two thin margins are fine for now).** A is the trace, B the abstract at the fitted parameters. Each row gets a
verdict, and the comparison exits 0 only when at least one row was judged and none is outside.

- **Allowed distance** = the tolerance of the row's class + the row's largest distance
  between B and each `--self` document. A `--self` document is the same abstract, same
  parameters and `--gpus`, at another seed (`aeiou dry-run --seed N --metrics-json`): an
  abstract is not asked to be nearer the trace than it is to itself. Three other seeds is
  what the tests use. The tool checks the abstract's hash, `--gpus`, ~~block, and sample~~ and block of a
  `--self` document; it cannot check the parameters, which the document does not carry.
- **Which rows (2026-10-22, `DESIGN_REVIEW.md` §3.79).** The extent rows when both documents
  have extents; the block rows when both have blocks, which a capture taken before then
  (`aeiou_metrics: 1`) has alone, so its abstract's dry run is taken with `--metrics-block`
  at the capture's size; block sizes that differ are refused where only the blocks are in
  common, and a pair with neither in common is refused.

  | Class (prefix of the row's name) | Tolerance |
  |---|---|
  | `mix`, `ops` (shares of the op mix) | 0.05 |
  | `request size` | 0.05 |
  | `run length` (the two histograms and the share of bytes in multi-op runs) | 0.10 |
  | `reuse` (first touches, and what a block's previous access was) | 0.10 |
  | `reuse distance` | 0.10 |
  | `extent reuse` (first touches, what a byte's previous access was, the share of partial re-reads; 2026-10-22) | 0.10 |
  | `extent reuse distance` (2026-10-22) | 0.10 |
  | `popularity` | 0.05 |
  | `fan-out`, `depth` | 0.10 |

- **Not judged:** a row with nothing on either side; a popularity row whose top fraction
  is fewer than 10 units on either side (the top 10 % of 16 files says nothing); `depth`
  when the trace was read without `--chain-gap-us`.
- **The tolerance file** (`--tolerances FILE`, `aeiou_tolerances: 1`), kept beside a kit's
  trace document as `<trace>.tolerances.json`: `tolerances` replaces class values;
  `unseen` lists rows the trace cannot show (a prefix of the row's name and the reason;
  not judged); `outside` records rows known to be outside with the reason (judged, still
  outside, the reason printed). An entry that matches no row, or records as outside a
  row that is within, is reported.
- **These are the repository's defaults.** Whoever accepts abstracts for a workload class
  can publish a file with other values.

The verdicts of the committed pairs (`tests/test_trace.py` reads them from the capture
records, `capture.toml`, §8; the reasons are in each pair's tolerance file):

| Pair | Verdict | Outside or unseen |
|---|---|---|
| `file_per_sample` | accepted | |
| `npy_per_sample`, both corpora | accepted | reuse distance 0.36 on 16 files, where the abstract differs from itself by 0.31 |
| `dcp_save` | accepted | |
| `dcp_load`, three traces | accepted | |
| `diskann_search`, with and without the node cache | accepted | depth not judged (no `--chain-gap-us`) |
| `diskann_build` | not accepted, 9 rows | run-length histograms (unmodeled small files and headers), write-after-write (a header puts the stream's writes off block boundaries), two reuse distances (cause not isolated) |
| `lmcache_local_disk` | accepted | fan-out unseen (a thread pool); the reuse distance of the reads was outside until `keep` was drawn once per round of conversations (`DESIGN_REVIEW.md` §3.57; 1 row before, 6 before the ShareGPT replay, §3.56) |
| `lmcache_local_disk`, AgentX replay at 2 GiB of engine KV | not accepted, 4 rows | each with its reason in the capture's tolerance file (`DESIGN_REVIEW.md` §3.63) |
| `lmcache_local_disk`, AgentX replay at 6 GiB | accepted | (§3.64) |
| `lmcache_local_disk`, AgentX replay with LMCache's `use_odirect` | not accepted | the read-after-write distance (the server's admission order) |
| `lmcache_shared_fs`, writer | not accepted, 1 row | the distance from a file's header write to its chunk write (two store threads); run length and fan-out unseen (a thread pool); 2 rows before §3.57, 7 before the replay |
| `lmcache_shared_fs`, reader | accepted | run length and fan-out unseen; 1 row before §3.57, 4 before the replay |
| `safetensors_load`, `faiss_ivf_ondisk` | nothing judged | the application reads through a mapping: every row is unseen. The exact call counts of their tests and `faults.py` are the evidence for these two. |

**Checked against the runner (2026-10-01).** The runner under `strace` is an application
whose abstract is known, so the trace's numbers must be the dry run's wherever order does
not enter. `tests/test_trace.py` runs `file_per_sample`, `lmcache_local_disk`, and
`diskann_search` (one GPU, `sync` backend) that way: read and write counts, bytes,
request sizes, run lengths, block and object popularity, and first touches are equal;
reuse distances differ by the order alone (CDF distance 0.07 to 0.20 on these small
configurations). Fan-out and depth are `none` on the trace side, as said above: the
runner's sub-actors are threads.

**Checked against a real application (2026-10-01).** A traced application's kit is its
mechanism's `kit/` under `workloads/`, and each capture's metrics document, tolerance file,
and fitted parameters are under the mechanism's `cases/<case>/captures/` (`traces/<name>/`
held both until 2026-10-10, `DESIGN_REVIEW.md` §3.66). `file_per_sample`'s kit holds the
first row of the capture plan (`ABSTRACTS.md` §11): PyTorch `DataLoader` + `ImageFolder`
on NFS, the script that was traced and the corpus writer, and its capture the fitted
parameter file and the trace's metrics document. The trace corrected the abstract (a second `lseek` per file,
`stat` and `fstat` per directory in the walk); `tests/test_trace.py` now holds the abstract
at the fitted parameters to that document. Findings in `ABSTRACTS.md` §1, reasoning in
`DESIGN_REVIEW.md` §3.43 (decided 2026-10-01). `npy_per_sample`'s kit is row 2,
`np.load(...)["x"]` as upstream DLIO and any NumPy user issues it: that abstract was rewritten from its trace
(`ABSTRACTS.md` §2, `DESIGN_REVIEW.md` §3.44; explicit seeks decided 2026-10-01; reviewed 2026-10-02, §3.53 and §3.54, all choices decided: the script's `glob` is the phase `enumerate`, on by default, and `tests/test_formats.py` checks `framing` against the installed NumPy).
`dcp_save`'s kit is row 3, `torch.distributed.checkpoint.save` on two ranks
(`ABSTRACTS.md` §3, `DESIGN_REVIEW.md` §3.45, decided 2026-10-02).
`dcp_load`'s kit and `safetensors_load`'s kit are row 4: `torch.distributed.checkpoint.load`
on two ranks, whose trace rewrote the restore's item loop, and safetensors `from_pretrained`,
which maps the shards and issues no `read` (`ABSTRACTS.md` §4, `DESIGN_REVIEW.md` §3.47, not
yet confirmed). A third `dcp.load` trace (`name-order`) showed that the reader takes the
items in the sorted order of their names and that an item's cost depends on where the
buffer was left; the restore carries the buffer's start as a chain since (§3.48, decided
2026-10-01), and `dcp_load`'s `kit/params.py` makes the parameter file from a real
checkpoint's `.metadata`.
`faiss_ivf_ondisk`'s kit is row 6: FAISS `IndexIVFPQ` over `OnDiskInvertedLists`, which maps
the lists file and issues no call on it; its `faults.py` measures with `mincore` what the
trace cannot show (`ABSTRACTS.md` §6, `DESIGN_REVIEW.md` §3.49, decided 2026-10-01).
`diskann_build`'s kit and `diskann_search`'s kit are rows 7 and 5: DiskANN through
`diskannpy`, one index built in 13 shards and then searched; `hops.py` reads the beam search
(rounds, batch sizes, sector spread) off the `io_submit` lines (`ABSTRACTS.md` §5 and §7,
`DESIGN_REVIEW.md` §3.50, decided 2026-10-02).
`lmcache_local_disk`'s kit is row 8: vLLM with LMCache's local-disk backend on a GPU. A
synthetic chat load (`chat.py`) fixed the call sequence (`ABSTRACTS.md` §8,
`DESIGN_REVIEW.md` §3.51, decided 2026-10-02); the committed trace is a replay of ShareGPT
(`replay.py`: the dataset's turns and reply lengths, eight conversations open and served
in turn), and `fit.py` writes `fitted.params.json` from the two logs of that run, the
load's and LMCache's, both in the kit (§3.56, decided 2026-10-02). It is a kit
script and not `aeiou-fit`: it reads no `strace`. `agentx.py` is a second fit of the same
abstracts from a public corpus of agentic coding sessions (SemiAnalysis's InferenceX AgentX
traces on HuggingFace: timestamps, think times, and the prompts' KV blocks as hash ids),
writing `fitted.agentx.params.json`, and its `reference` is the corpus's own chunk
accounting (`agentx.reference.json`), which `tests/test_trace.py` checks a dry run against;
`replay_agentx.py` replays the corpus through a server for the trace and `keep` the corpus
cannot give (`DESIGN_REVIEW.md` §3.59, 2026-10-02, ~~not yet run on a GPU~~ run 2026-10-07
and 2026-10-08, §3.63, at 2 GiB and 6 GiB of engine KV, §3.64). A fit of what an engine
holds records the engine it was fitted at (pool, model, CPU tier, slots) as its parameter
file's `provenance`; `agentx.py holds` checks a replay's holds against a simulation of the
engine's prefix cache, and `agentx.py log` writes the reduced server log the kit keeps.
`lmcache_shared_fs`'s kit is the same load on LMCache's `fs://` backend, twice: an engine on
an empty store, then a restarted engine on the filled one, sent the same requests
(`lmcache_local_disk`'s `kit/replay.py --save`, `--replay`). `workloads/kvcache/lmcache-shared-fs/lmcache_shared_fs.py` emits both abstracts,
`lmcache_shared_fs` and `lmcache_shared_fs_reader`; the reader takes the writer's namespace as
`input` and `same_run` (contract 0.4: `w.namespace(..., input=True, same_run=True)`), so a run
without the writer's `--seed`, `--gpus`, and parameters is refused before the gate
(`DESIGN_REVIEW.md` §3.52, decided 2026-10-02).

**Where a dataset or namespace lives is declared in the script** (contract 0.6,
2026-10-09; `s3` was `object` until contract 0.7): `w.dataset(..., protocol="s3")`,
`w.namespace(..., protocol=...)` for one the application
reaches through an S3 store;
leave it out for a file system. Namespaces sharing a root declare one protocol (V17). A run
places each with `--endpoint` (`runner/REFERENCE.md` §4), and `aeiou-datagen` takes the same
flag; the protocol is not part of a dataset id. ~~The runner refuses `object` until its object
engine is built~~ The runner reads an `s3` dataset through its object engine and writes one
with `aeiou datagen` (`runner/REFERENCE.md` §16, 2026-10-09); ~~an `object` namespace waits for
the engine's writes,~~ it writes and reads an `s3` namespace too (the same night), where
V18 refuses `open` with `APPEND`, `RDWR`, or `EXCL`, `ftruncate`, and `fallocate` on its
objects (`namespace(..., protocol="s3")` raises at the op); ~~`aeiou-datagen` writes
container datasets to directories only (`DESIGN_REVIEW.md` §3.65).~~ `aeiou-datagen` writes
container datasets to object stores too, streaming each file into an upload (§6,
`DESIGN_REVIEW.md` §3.69, 2026-10-11).

**The application's APIs are declared in the script** (contract 0.3, 2026-10-01; `api` and
`cache` since contract 0.6, `posix`, `cache`, and `s3` since 0.7, 2026-10-09):
`Workload("safetensors_load", posix="mmap")`. Leave `posix` out for an application that calls
`read` and `write`. `cache="direct"` declares an application that opens every regular file
`O_DIRECT` (refused under `mmap`); leave it out when the open flags of the ops say what it
does. `s3="async"` declares an application whose S3 client keeps many requests in flight
per thread; leave it out for one that blocks on each. V19 refuses an API for a protocol the
workload has no names on, and, with both, APIs of different kinds. The runner uses them as
the default backend (`runner/REFERENCE.md` §4).

**`aeiou-trace export`: the trace file of the `trace` node (added 2026-10-02;
`DESIGN_REVIEW.md` §3.58, `runner/REFERENCE.md` §13).** The same `strace`, through the same
resolver (descriptor tables, `clone` and `fork`, `dup`, `cwd`, the paths under the root),
written as the file the runner executes literally: JSON Lines, a header, one op per line in
the order the calls returned, a lane per traced task, opens by id, paths relative to the one
`--root`.

```
aeiou-trace export trace.txt --root /mnt/data [--exclude GLOB]… [--cwd DIR] -o app.jsonl
aeiou-trace metrics app.jsonl -o app.metrics.json      # an exported file measures too
```

`export` prints the file's sha256, which the abstract's `w.trace("app.jsonl", sha256)`
names; the runner refuses the file when it differs. What differs from `metrics` on the same
strace, each counted in the header's `notes`: a sequential read or write on an open that
more than one lane used is written **positioned** at the offset the trace shows it used
(`shared_positions_resolved`; the lanes' interleaving decided it, and the runner's lanes
are not that interleaving), as is every read and write on an open whose `O_APPEND` write
went to a known end of the file (`append_positions_resolved`, the writes; that offset is
the file's size, not the open's position),
an `mmap` of a file range is one positioned `read` of the range
(`mmap_as_read`; `metrics` does not count a mapping), and a call the schema has no op for
(`truncate` by path, `sendfile`, an `ioctl` or `fadvise` the schema does not name) is
dropped by name. `metrics` reads an exported file as it reads an strace (the header is
sniffed), with a lane as the context and the file as one instance, and **gives the same
document as the strace it came from** (`tests/test_trace.py`), which is also what `aeiou
dry-run --metrics` gives for an abstract that is that one node. Under `io_submit` the
members are placed at the submission (the runner issues them there) where `metrics` counts
them at the reap, so the reuse-distance histogram of that pair differs within a round.

**`aeiou-trace digest`: the structural summary a fit is made from (added 2026-10-29;
`DESIGN_REVIEW.md` §3.87, `PROJECT_BRIEF.md` item 47).** The same calls `export` writes,
through the same resolver, or an exported trace file, reduced to a document that can leave
the trace's owner (`aeiou_digest: 1`; no raw path unless `--paths`, no data bytes ever):

```
aeiou-trace digest trace.txt --root /mnt/data [--exclude GLOB]… [--cwd DIR] [--epochs N] [--candidates N] [--paths] [--text] [-o trace.digest.json]
aeiou-trace digest app.jsonl                     # an exported file carries its root
```

`templates`: every run of digits in a path a field (`train/{a:05}/sample_{b:09}.npz`; sixteen
or more hex digits with a letter, `{a:x32}`), with each field's value count and range, the
op mix, bytes, request sizes, and per open how it moved through its file (`whole`,
`sequential`, `positioned`, `none`). `lanes`: per traced task its `phases`, runs of units
(the calls on one open, or one path call, or one `io_submit`) of one motif on one template,
the motif the op sequence with repeating cycles collapsed (`read+`, `(lseek read)+`), with
repeat counts, bytes, and think time. `sites`: every template opened at least twice, with
three verdicts, each either decided or `undecided` with `settles`, the evidence that would
decide it: `selection` between `consume` (a permutation pass, each path once per pass) and
`choose` (a draw with replacement), from the passes the path sequence shows (greedy: a pass
ends at the first path it already holds) and, with no repeat, from the birthday bound at 5 %
against `--candidates`, the corpus size; `order` within a pass, `sequential`, `permuted`, or
`other`; `size` per open, `constant`, `per-path`, or `drawn`. `open`: every undecided verdict
as a question in the application's terms, its candidates, what settles it, and the
candidates' divergence at `--epochs` passes (the share of the set a draw never reads, e^-N,
against zero for a pass), ranked. As built, the digest asks nothing; designed 2026-10-29
(`DESIGN_REVIEW.md` §3.88) and next to build: it asks these questions on a terminal, facts first,
checks each answer against the trace, and records answers under `asserted` with provenance,
never as measurements, in the document and in an answers file beside it that `--answers FILE`
replays without a terminal. `tests/test_digest.py` holds the verdicts to constructed sequences
and the command to its refusals.

## 8. `aeiou-catalog`: the workload catalog (2026-10-10)

`workloads/` holds one directory per I/O mechanism and, under `cases/`, one per case: an AST
and a parameter set, the unit a user asks for and the catalog's row (`DESIGN_REVIEW.md`
§3.66). Building is a compiler's job, a script to an AST; the catalog is data management:
curated records, the vocabulary they are written in, and their consistency with the
abstracts. `aeiou/catalog.py` is that tool, no extra needed (`tomllib` for the records, the schema's `jsonschema` for the ASTs), never writing
an AST or a parameter file.

**The records.** TOML, written by a person:

| File | Holds |
|---|---|
| `<category>/[<class>/]<mechanism>/mechanism.toml` | `category`, `class` (where the category has classes), `application`, `format`, `phase` and `domain` (one value or a list); a `[defaults]` table (`ast`, `gpus`, `length`) while the mechanism has no case, so its row is the abstract at its defaults |
| `cases/<case>/case.toml` | `kind`, `modality`, `model-family`; `gpus` (the counts its parameters hold at; the shape uses the first); `length` (the parameter that only lengthens the run: `requests`, `queries`, `calls`); ~~`metrics_block` (a KV chunk for the KV cases, so every access is one block)~~ (gone 2026-10-22 with blocks, `DESIGN_REVIEW.md` §3.79); one `[[run]]` (`ast`, `params`, one fragment or a list) per AST |
| `captures/<capture>/capture.toml` | `raw_trace` (a pointer, or `"not retained"`); one `[[pair]]` per trace (`trace` stem, `ast`, `params` when the capture's fit differs, `verdict`) |

`test_trace.py` reads its pairs from the capture records (`catalog.capture_pairs`), so a
verdict lives beside the trace it is about; the list kept in the test is gone.

**The dictionary.** `workloads/attributes.toml`: per attribute its meaning, source
(`curated`, `derived`, `measured`), level (`mechanism`, `case`), and values, or for a measured
one its unit, tolerance, and classes (first match, bounds inclusive). A record may state only
curated attributes of its level, with defined values, and the tool refuses the rest: an
unknown key or value, a derived or measured attribute stated by hand, a missing required one
(`required = true`, or `required = { kind = "use" }`: required when the record's `kind` is `use`),
a list where the dictionary does not mark the attribute `multiple`, a category or class
other than the directory's, a directory of the first level that is not a category (or a
category without its directory, or a class directory its category does not define), a run
of an AST the mechanism does not have, a parameter file no run names, a trace no pair judges,
a tolerance file without its trace, two ASTs of one name, a mechanism with neither a case nor
`[defaults]`.

**The tree.** The first level is the dictionary's `category` values and nothing else, the
second level (where a category has one) its `class` values (`DESIGN_REVIEW.md` §3.68). Every
category and class has its directory from the start: `write` makes a missing one and writes
its `README.md` from the dictionary (the meaning, the words it is searched for as, the
classes, the mechanisms there now, or where a first one goes), and `check` holds those READMEs
to what it renders, as it does `CATALOG.md`. `CATALOG.md` opens with the coverage: every
category and class with its mechanism and case counts, empty ones included.

**Derived.** `api` (the AST's `posix` key, `sync` without one), `writes` (a namespace not
`input`), `inputs` (an `input` namespace), `status` (the best verdict of the captures of that
run: accepted, then not accepted, then nothing judged; untraced without one).

**Measured: the shape.** From `aeiou dry-run --metrics-json` of each run, its `total`:

| Attribute | From the metrics |
|---|---|
| `run_length_read`, `_write` | byte-weighted median of `run_length.<kind>.bytes` |
| `request_size_read`, `_write` | byte-weighted median of `request_size.<kind>` (as asked) |
| `read_share` | `bytes_read / (bytes_read + bytes_written)` |
| `metadata` | open, close, stat, fstat, readdir, mkdir, rmdir, unlink, rename, ftruncate, fallocate per read and write (lseek, ioctl, fsync, fadvise are neither) |
| `reads_own_writes` | 1 when `read_after_write.n` > 0 |
| `reread` | `reuse_extents.read_after_read.n` over the bytes read (first touches and both reuse kinds), in bytes; ~~over the block reads~~ until 2026-10-22 (`DESIGN_REVIEW.md` §3.79) |
| `skew` | `popularity_bytes.top_1_pct`; ~~`popularity_blocks.top_1_pct`~~ until 2026-10-22 |

A byte-weighted median is the bucket at or above which half the bytes lie, so a size is a
quarter-octave bucket's lower bound; a count median is wrong here (the header reads of
`np.load` would make 140 MiB samples 112-byte runs).

The shape is measured **at the case's own parameters** and first GPU count, seeds 1, 2, and
3, the median of the three recorded with its spread and the measures on which the seeds
differ by more than the tolerance (named in the catalog: the case's numbers carry that
error). The user's rule (2026-10-10) was the smallest length at which the numbers are
stable; building it showed that a shorter run is not a cheaper measurement of the same
numbers. Shortening `steps` drops a training job's second epoch (ImageFolder's `reread` 0.00
at 14 of 400 steps, 0.50 at 400); on a fixed index a search's reuse climbs with the run
(DiskANN over SIFT1M re-reads 0.2 % of its block reads at 8 queries, 24 % at 500, 59 % at
2,000, 97 % at 32,000, and its 2 GiB PQ load dominates the read request sizes until about
8,000 queries), and a test of agreement between successive doublings accepts such a slow
trend at the short end, where it barely moves. So the length is the case's, and the rule
serves the cost instead: for a case with a `length`, `write` ~~halves it (down to 1/64) while
the medians stay the case's within their tolerances and records the smallest such length~~
finds the smallest of its halves (down to 1/64) whose medians are the case's within their
tolerances (2026-10-23, below) and records it as
`length.check`, and `check` re-measures there. The full AgentX corpus costs its 98,827
requests three times, once a seed, when its shape is written; CI re-measures it at a fraction
of that, and runs it once more at its full length, the first seed's, for the counts (below).

`shape.json` (`shape.defaults.json` for a defaults row): per run the AST and parameter
file with their SHA-256, the GPU count, ~~the block,~~ the seeds, the length (`param`, `case`,
`check`), `stable` and `moving`, one seed's op count, its counts by kind and bytes read and
written, each dataset's projection as `dry-run --metrics-json` gives it (`datasets`), and per
attribute `value`, `spread`, `class`. `shape_version` 2 since 2026-10-14 (the counts, bytes, and
projections); `check` reports a shape of another version as stale.

**The descriptions** (2026-10-14, `DESIGN_REVIEW.md` §3.76, `PROJECT_BRIEF.md` §6 item 27).
`aeiou/describe.py` walks an AST at a parameter set and puts it in English: a paragraph (the
AST's `doc`, the records' context, each dataset, each actor's ops in order with the parameters'
values in place and the branches they switch off left out, the shape) and an account (the
parameters with values, defaults, units, and `doc`; the datasets and namespaces; every
statement in order, every op named, every branch marked taken or not; the shape in full).
Expressions over parameters and literals are worked out and shown with their formula, others
stay formulas; distributions are given with percentiles computed from their parameters. `write`
writes each case's `DESCRIPTION.md` (`DESCRIPTION.defaults.md` for a defaults row) and the
paragraphs into `CATALOG.md`; `check` holds them to what it renders. `aeiou-catalog describe AST
[--params-file FILE] [--gpus N]` prints one for any abstract, its shape measured by dry run
unless `--no-shapes`. `test_catalog.py` holds the describer's kinds to the schema's.

**The commands.** `check` validates the records, re-measures every shape at its check
length against the recorded numbers (their tolerances), holds the recorded op count, counts
by kind, bytes, and `datasets` exactly to the first seed's dry run at the case's own length
(the run they came from; one more dry run when the check length is shorter, started first
beside the rest: the descriptions and `CATALOG.md` print these numbers, 2026-10-14), renders
`CATALOG.md`, and fails on any difference; with `--no-shapes` (the builder's tests, no runner) it checks instead that
each shape is there and was measured on the AST, parameter file, GPU count, and length
parameter committed now. `write` measures and writes the shapes, the descriptions, and
`CATALOG.md`. CI runs
`check` with the runner in the job that builds it.

*2026-10-23 (the user's request, after a rewrite took 36 minutes on two or three of twenty
cores; goal: the least wait, no CPU wasted):* `write` measures only the shapes that are stale
by that same check (missing, of another `shape_version`, or measured on another AST, parameter
file, GPU count, or length parameter) and keeps the rest as committed, since their numbers
holding is `check`'s to say; `--all-shapes` measures every one, for a change to the runner's
metrics, which no recorded hash shows. Both commands run their dry runs on one scheduler of
`--jobs N` at once (default the CPUs the process may run on, `sched_getaffinity`: 20 on the
development machine, 4 on CI's runners), from one queue ordered by the op count each case's
shape recorded, the largest first (a case never measured first of all), so the longest case
starts first instead of ending the write alone; every case's first dry runs are queued before
any starts. The check length is searched, not halved in turn: the full length's three seeds and
the first level's run at once; each level is decided on its three seeds against the full run's
medians (never against a neighbouring level, as above), and the next level is queued only once
it is decided: down while a level agrees, up while it differs. The search starts at the check
length the shape recorded before (at half the case's when that was the case's own), and a new
case at the smallest level, since the committed shapes hold at 1/32 (the LMCache AgentX and
chat cases) or only at the full length (every other). A case whose shape changes not at all
thus costs its full length and two levels. Nothing runs speculatively, so no dry run is
wasted; speculating on the level after next would shorten the tail and was left out. The dry
runs are deterministic (positional randomness, no timing) and each search depends on its own
case's numbers alone, so the files are the same at any `N`; memory, not the CPUs, may bound
`N`: a large case's dry run under `--metrics-json` takes gigabytes.

*2026-10-23 (item 40, `DESIGN_REVIEW.md` §3.80):* `write` also writes `workloads/catalog.json`,
the catalog for a program (`render_index`, `INDEX_VERSION` 1), and `check` holds it to the
records as it holds `CATALOG.md`: the vocabulary as a list in the dictionary's order (a measured
attribute's values its classes), and every case a lock names, in the tree's order, with its
curated attributes, GPU counts, lock, description, and per run what it does (the AST's doc),
what it is (its parameter files' docs), and its derived attributes and measured classes. The
runner's `aeiou cases` reads it and never the records (`runner/REFERENCE.md` §18), so the
derivations have this one author.

*Choices made in the build, open to revision:* the shape at the case's own length, re-checked
at a shorter one (above); KV cases measured in chunk-sized blocks (the shares are those at
4 KiB, the work far less); a training job's `steps` is never a `length`; a tolerance may give
both a relative and an absolute bound, either enough (`metadata`: 5 % or 0.005, since a ratio
near zero moves by more than 5 % with no class seeing it); a case with two runs is two rows;
the derived attributes per run; ~~`metrics_block`,~~ `gpus` and the length in `case.toml`
rather than the dictionary, being what the case runs rather than what it is.

## 9. Not yet

- Format classes for Arrow IPC, MDS, and Megatron `.bin`/`.idx`; the Parquet→Arrow conversion
  abstract (`GRAMMAR_OPTIONS.md` §6.5).
- The `strace` → parameter fitting tool (`aeiou-fit`, `ABSTRACTS.md` §11), which will write
  parameter files (`aeiou-trace` of §7 reads the trace and measures; it fits nothing); the offline content verifier (`aeiou-verify`), which `rng.py` and
  `framing.py` now make a short tool.
