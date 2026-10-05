# The builder

The Python half of aeiou: the package `aeiou`, with which a workload is *authored*, and its
four tools: `aeiou-build` (scripts to abstracts), `aeiou-params` (parameter files),
`aeiou-datagen` (container datasets), and `aeiou-trace` (the metrics of a real
application's trace, their comparison with an abstract's, and the trace file of the
runner's `trace` node). Nothing here runs on a client node during a benchmark. The manual
pages are in [`../man/`](../man/README.md); the reference, the authoring API and every
choice made in building it, is [REFERENCE.md](REFERENCE.md).

```
uv sync --extra test --extra formats          # or: pip install -e '.[test,formats]'
uv run aeiou-build --hermetic --twice -o ../schema/examples abstracts/*.py
uv run pytest
```

## What it is

An abstract is authored as a Python script and published as a JSON document. The package
is the vocabulary of that script: `Workload`, its `param`, `dataset`, `namespace`, and
`actor` declarations, and cursors on which the body is written, loops, bindings, forks,
channels, barriers, compute, and the eighteen operations. Running the script once
constructs an abstract syntax tree; `aeiou-build` validates it against the published
schema and the semantic rules, and writes it with its provenance. The document's canonical
hash is the workload's identity, and the Rust runner (`../runner/`) executes the document
and nothing else.

## Theory of operation

**Two kinds of control flow.** Python control flow runs at build time and generates
*structure*; cursor control flow (`loop`, `when`, `choose`, `parallel`, `loader`) is
recorded in the tree and generates *iterations* at run time. A symbolic value has no truth
value, cannot be iterated, and cannot be converted to an int, so `for j in P.batch` or
`if step % 500 == 0` is a `TypeError` naming the cursor form to use. Python loops remain
the right tool for generating four similar actors, a `choose` from a table, or the seven
fixed system calls of a Python `open()`; a lint refuses three identical statements in a row.

**Nodes, not values.** Every quantity in the tree is symbolic (`nodes.py`): a parameter
reference, a loop index, a draw from a distribution (`dists.py`), arithmetic over these.
Sizes are bytes and durations nanoseconds, so `105 * ms` is an int. The builder keeps no
second copy of the validation rules: `validate.py` loads `../schema/abstract-ast.schema.json`
and `../schema/check.py` and runs them on the emitted dictionary (`emit.py`), which is
written in the builder's key order with the hash in its `provenance` block.

**Reproducible builds.** `aeiou-build --hermetic` runs each script in a child interpreter
with a scrubbed environment and PEP 578 audit hooks that deny sockets, subprocesses, every
write, and reads outside the interpreter, the schema, and the script's directory; clocks and
entropy raise. `--twice` builds in two children with different hash seeds and refuses to
write unless the hashes agree, which catches set-iteration order without anticipating it.
`--check` is CI's drift test: the committed abstracts in `../schema/examples/` must
regenerate hash-identically from `abstracts/*.py`.

**Shape and parameters.** A script's defaults are placeholders or one reference set; the
measured values of a shape live in a parameter file, so one published shape carries several
sets. `aeiou-params` writes a shape's defaults as a set, checks sets with the runner's
rules (a value keeps the kind of its default), and builds sets from real files: the tensor
table of a safetensors checkpoint, the archive framing of a NumPy corpus (`params.py`).

**Format classes.** A dataset in a container format declares its class (`formats.py`:
`tfrecord`, `parquet`, `hdf5`, `webdataset`), and the class supplies what the two-contract
split assigns to the format: the *layout*, a generic framing formula the runner computes
every offset from; the reader library's *protocol*, traced once and emitted as ordinary
operations on a cursor; and the *writer* `aeiou-datagen` uses to produce real files with
the same positional payload as the Rust writer (`datagen.py`, `layout.py`, `rng.py` are the
ports of the runner's definitions it needs). Layout constants are probed from the installed
library at build time and recorded in the document.

**Calibration.** `aeiou-trace` (`trace.py`, standard library only) resolves an `strace`
of the real application, descriptor tables through `clone`, `fork`, and `dup`, positions
through `lseek` and the bytes moved, paths under the storage root, and computes the same
locality metrics the runner's dry run computes for the abstract, in the same document
format. `compare` holds the two to a tolerance per class of metric, raised by what the
abstract differs from itself by at another seed. `export` writes the same resolved trace as
the JSON Lines file a `trace` node executes literally. `traces/` holds the capture kit of
every real application traced so far: the script that was traced, the corpus writer, the
fitted parameter file, the trace's metrics document, and its tolerance file.

**One error frame.** The four tools report a wrong command line as the runner does
(`usage.py`): the command as typed, every missing argument at once, the usage line, the
pointer to the help, exit status 2. `options.py` is the Python mirror of the runner's
option layers, used by `aeiou-datagen`.

## Technologies

Python 3.12 or later. The package depends on `jsonschema` alone; `hatchling` builds it,
`uv` locks it (`uv.lock`), `pytest` with `pytest-xdist` tests it. The `formats` extra
installs the libraries of the format classes: `pyarrow` (Parquet), `h5py` (HDF5), `numpy`,
`crc32c` (TFRecord framing), `xxhash`, and `dgen-py` 0.3.0, the wheel that reproduces the
Rust payload generator's bytes. PEP 578 (`sys.addaudithook`) for the hermetic build;
`tarfile`, `zipfile`, and `re` from the standard library for WebDataset, `.npz`, and the
strace parser.

## Layout

```
aeiou/           the package: nodes, dists, builder (Workload, Cursor), validate, emit, hermetic, cli,
                 params, formats, layout, rng, pattern, datagen, trace, options, usage
abstracts/       the workloads as authoring scripts: small-file and large-sample training, checkpoint
                 write and restore, model load, DiskANN search and build, IVF search, KV-cache serving
                 and the shared store (writer and reader), streaming training over TFRecord and
                 Parquet shards, map-style training over HDF5
traces/          one directory per real application traced: the kit, the fitted parameters, the
                 trace's metrics and tolerances
tests/           pytest: every abstract builds and matches its committed AST; the discipline holds;
                 parameter files fit; the format classes write files the libraries read back and
                 the runner executes to the dry-run fingerprint; the trace metrics against
                 hand-written traces and against the runner under strace; the option layers; the
                 usage frame; the manual pages against --help
pyproject.toml   the package, its extras, and the four entry points
REFERENCE.md     the reference: writing an abstract (§1), the discipline (§2), what fails where (§3),
                 reproducible builds (§4), parameter files (§5), format classes and aeiou-datagen (§6),
                 aeiou-trace (§7), not yet (§8)
```

Tests that drive the runner use `../runner/target/release/aeiou` (or `$AEIOU_RUNNER`) and
skip when it is not built; rebuild it after runner changes.

## Pointers

- `../man/aeiou-build.1.md`, `aeiou-params.1.md`, `aeiou-datagen.1.md`, `aeiou-trace.1.md`,
  `aeiou-abstract.7.md`: the manual.
- `REFERENCE.md` §1: the authoring API in one page.
- `../schema/README.md`: the contract the builder emits.
- `../ABSTRACTS.md`: the workloads on paper, and what each trace corrected.
- `../DESIGN_REVIEW.md` §3.19, §3.27, §3.28, §3.42 onward: why the builder, the parameter
  split, the format classes, and the trace tooling are the way they are.
