# The builder

The Python half of aeiou: the package `aeiou`, with which a workload is *authored*, and its
six tools: `aeiou-build` (scripts to abstracts), `aeiou-params` (parameter files),
`aeiou-trace` (the comparison of two metrics documents and the digest a fit is made from;
the tool itself is the runner's binary since 2026-10-29, which runs these two from here until
they are ported, `DESIGN_REVIEW.md` §3.89), `aeiou-catalog` (the
workload catalog under `../workloads/`), `aeiou-header` (an HDF5 file's header from
libhdf5, which `aeiou datagen` runs), and `aeiou-datagen` (container datasets; deprecated,
since `aeiou datagen` writes every format class). Nothing here runs on a client node
during a benchmark. The manual
pages are in [`../man/`](../man/README.md); the reference, the authoring API and every
choice made in building it, is [REFERENCE.md](REFERENCE.md).

```
../dev.sh setup          # the venv (and the runner) at the checkout's pins (DESIGN_REVIEW.md §3.82, §3.83)
.venv/bin/aeiou-build --hermetic --twice $(find ../workloads -name '*.py' -not -path '*/kit/*')   # each AST beside its script
.venv/bin/pytest
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
`--check` is CI's drift test: every committed abstract under `../workloads/` must
regenerate hash-identically from the script beside it.

**Shape and parameters.** A script's defaults are placeholders or one reference set; the
measured values of a shape live in a parameter file, so one published shape carries several
sets. `aeiou-params` writes a shape's defaults as a set, checks sets with the runner's
rules (a value keeps the kind of its default), and builds sets from real files: the tensor
table of a safetensors checkpoint, the archive framing of a NumPy corpus (`params.py`).

**Format classes.** A dataset in a container format declares its class (`formats.py`:
`tfrecord`, `parquet`, `hdf5`, `webdataset`), and the class supplies what the two-contract
split assigns to the format: the *framing*, a generic formula the runner computes every
offset from; the reader library's *protocol*, traced once and emitted as ordinary
operations on a cursor; and the *writer* `aeiou-datagen` uses to produce real files with
the same positional payload as the Rust writer (`rng.py`, `pattern.py`, `framing.py` are the
ports of the runner's definitions it needs). Framing constants are probed from the installed
library at build time and recorded in the document. `aeiou datagen`, built with the cargo
feature `formats`, writes the same files byte for byte; the Python writer stays as the
reference the tests compare it with until it is retired, and gets no new features. For
HDF5, whose header only libhdf5 can lay out, `aeiou datagen` runs `aeiou-header`
(`header.py`) for the header bytes and writes the rows itself.

**Calibration.** `aeiou-trace` (~~`trace.py`, standard library only~~ the runner's binary since
2026-10-29, `runner/aeiou/src/strace.rs`; `trace.py` keeps `compare` and `digest.py` the digest) resolves an `strace`
of the real application, descriptor tables through `clone`, `fork`, and `dup`, positions
through `lseek` and the bytes moved, paths under the storage root, and computes the same
locality metrics the runner's dry run computes for the abstract, in the same document
format. `compare` holds the two to a tolerance per class of metric, raised by what the
abstract differs from itself by at another seed. `export` writes the same resolved trace as
the JSON Lines file a `trace` node executes literally. The capture kit of every real
application traced so far lives with its mechanism under `../workloads/`: the traced
script, the corpus writer, and the fitting scripts in `kit/`, and each capture's metrics
document, tolerance file, logs, and fitted parameters under `cases/<case>/captures/`.

**The catalog.** `aeiou-catalog` (`catalog.py`, no extra needed) holds the records of
`../workloads/` (`mechanism.toml`, `case.toml`, `capture.toml`) to their dictionary,
`attributes.toml`; measures each case's shape with the runner's dry run; and renders
`CATALOG.md`, the category and class READMEs, and each case's `DESCRIPTION.md`, the abstract
put in English by a deterministic walk of the AST at the case's parameters (`describe.py`).
`check` fails on any drift, as `aeiou-build --check` does for the abstracts.

**One error frame.** The six tools report a wrong command line as the runner does
(`usage.py`): the command as typed, every missing argument at once, the usage line, the
pointer to the help, exit status 2. `options.py` is the Python mirror of the runner's
option layers, and `coord.py` of its coordinator protocol, both used by `aeiou-datagen`.

## Technologies

Python 3.12 or later. The package depends on `jsonschema` alone; `hatchling` builds it,
`uv` locks it (`uv.lock`), `pytest` with `pytest-xdist` tests it. The `formats` extra
installs the libraries of the format classes: `pyarrow` (Parquet), `h5py` (HDF5), `numpy`,
`crc32c` (TFRecord framing), and `xxhash`; numpy's `SFC64` makes the payload, the Rust
writer's bytes. The `s3` extra installs `obstore`, the Python bindings of the runner's
`object_store`, for `aeiou-datagen`'s uploads. PEP 578 (`sys.addaudithook`) for the hermetic
build; `tarfile`, `zipfile`, `re`, and `tomllib` from the standard library for WebDataset,
`.npz`, the strace parser, and the catalog's records. The release ships `aeiou-header` as one
PyInstaller file beside the `aeiou` binary.

## Layout

```
aeiou/           the package: nodes, dists, units, builder (Workload, Cursor), validate, emit,
                 hermetic, cli, params, formats, framing, rng, pattern, datagen, header, coord,
                 trace, catalog, describe, options, usage
tests/           pytest: every abstract builds and matches its committed AST; the discipline holds;
                 the two checkers refuse alike; parameter files fit; the format classes write files
                 the libraries read back and the runner executes to the dry-run fingerprint, and
                 `aeiou datagen` writes the same bytes; the payload; the trace metrics against
                 hand-written traces and against the runner under strace; the kit scripts; the
                 catalog and the descriptions; several-host datagen; object stores (MinIO);
                 aeiou-launch; the option layers; the usage frame; the manual pages against --help
pyproject.toml   the package, its extras (test, formats, s3), and the six entry points
REFERENCE.md     the reference: writing an abstract (§1), the discipline (§2), what fails where (§3),
                 reproducible builds (§4), parameter files (§5), format classes and aeiou-datagen (§6),
                 aeiou-trace (§7), aeiou-catalog (§8), not yet (§9)
```

The workloads themselves, the authoring scripts and the ASTs generated beside them, live in
[`../workloads/`](../workloads/README.md), one directory per I/O mechanism.

Tests that drive the runner use `../runner/target/release/aeiou` (or `$AEIOU_RUNNER`) and
skip when it is not built; rebuild it after runner changes. `AEIOU_REQUIRE` names, comma-separated,
what a job provides (`runner`, `formats`, `object`, `strace`, `header`, `minio`, and the extras `py-formats`,
`py-s3`), and a test skipped
for lacking one of them fails (`tests/conftest.py`); CI sets it, so a test that needs what a job
built cannot be skipped there unseen, and every skip in the suite must name one
(`tests/test_require.py`).

## Pointers

- `../man/aeiou-build.1.md`, `aeiou-params.1.md`, `aeiou-trace.1.md`, `aeiou-catalog.1.md`,
  `aeiou-header.1.md`, `aeiou-datagen.1.md`, `aeiou-abstract.7.md`: the manual.
- `REFERENCE.md` §1: the authoring API in one page.
- `../schema/README.md`: the contract the builder emits.
- `../ABSTRACTS.md`: the workloads on paper, and what each trace corrected.
- `../DESIGN_REVIEW.md` §3.19, §3.27, §3.28, §3.42 onward: why the builder, the parameter
  split, the format classes, and the trace tooling are the way they are; §3.66 and §3.76, the
  catalog and the descriptions.
