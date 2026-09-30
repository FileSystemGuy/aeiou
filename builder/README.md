# `mlps_abstract`: the Python builder

Layer 1 of `GRAMMAR_OPTIONS.md` Option D (chosen 2026-09-30). An authoring script constructs
the AST of `schema/abstract-ast.schema.json` by running once; the calls that look like I/O are
declarations. The AST is the only thing the Rust runner executes and the only thing the
working group publishes with a hash. Nothing here runs on a client node.

```
builder/
  mlps_abstract/     the package: nodes, dists, builder (Workload, Cursor), validate, emit, hermetic, cli
  abstracts/         the ABSTRACTS.md workloads as authoring scripts (§1–§8; §4 is two scripts)
  tests/             pytest: every abstract builds, matches its committed AST, the discipline holds
  pyproject.toml     deps: pyyaml, jsonschema; `abstract-build` entry point
```

```
cd builder && uv sync --extra test              # or: pip install -e '.[test]'
uv run abstract-build --hermetic --twice -o ../schema/examples abstracts/*.py
uv run pytest
uv run abstract-build --check -o ../schema/examples abstracts/*.py     # drift check (CI)
```

## 1. Writing an abstract

```python
from mlps_abstract import *

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
    w.write()                                            # <name>.ast.yaml, validated, with provenance
```

**Declarations** on the `Workload`: `param(name, default, unit=, doc=, cli=)` or
`params(**defaults)`; `dataset(name, pattern=, count=, size=, seed=, access=, samples_per_file=,
chunk=, format=)`; `regions(name, file=, count=, slot=, size=, seed=)`; `namespace(name,
pattern=, fields=, size=, seed=)`; `actor(name, count=)`. `w.P.x` is the parameter reference;
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
function an author writes over nodes (`member_off(f, m)` in `abstracts/train_large_samples.py`).
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

The builder keeps no second copy of the rules: `validate()` loads `schema/abstract-ast.schema.json`
and `schema/check.py` from the repository (or `$MLPS_SCHEMA_DIR`) and runs them on the emitted
dict. The Rust validator must reject everything they reject.

## 4. Reproducible builds

`abstract-build` implements techniques 4–6 of `GRAMMAR_OPTIONS.md` Option D:

- **Canonical hash.** `sha256` of the canonical JSON (sorted keys, no whitespace, floats in
  `repr`, `provenance` removed); the same function as `schema/check.py`. The YAML header and
  the `provenance` block carry it.
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
  so `schema/examples/*.ast.yaml` cannot drift from `abstracts/*.py`.

Three don'ts for authors: iterate sets or dicts keyed by nodes, use `id()`/`hash()` for
ordering, read the environment or the clock. The harness turns each into an immediate error.

## 5. Not yet

- The `replay` trace format (schema §6); `cursor.replay` emits the node only.
- Format classes (`GRAMMAR_OPTIONS.md` §6.4): the reader protocols that emit POSIX nodes for
  Parquet, TFRecord, HDF5, Arrow, WebDataset, MDS, Megatron. `dataset(format={...})` records
  the class for the manifest; nothing is generated from it yet.
- A parameter file that fills named distribution slots of a published shape (Option D, "three
  sources"); today the defaults live in the script.
- The `strace` → parameter fitting tools (`ABSTRACTS.md` §11).
