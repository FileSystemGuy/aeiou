# aeiou-abstract(7)

## NAME

aeiou-abstract - the workload abstract: its model, its JSON form, its parameter files, and the manifests a run compares

## DESCRIPTION

An **abstract** is a workload written as a program over POSIX-shaped I/O operations: the
operation sequence a real application issues, as its author understands it, with the
quantities the application draws at random expressed as distributions. It is authored as a
Python script with **aeiou-build**(1), compiled to one JSON document (`<name>.ast.json`,
contract version 0.5), and executed by **aeiou**(1). The document is the only thing the
runner executes and the only artifact an author publishes with a hash.

The model has three parts:

- **Datasets** are sample spaces that exist before the run: `files` named by a pattern with
  sizes drawn from a dataset seed, or `regions` inside one file. They are read-only. A
  dataset stored in a container format (Parquet, HDF5, TFRecord, tar) declares its format
  class and a *layout*, a generic framing formula from which the runner computes every
  offset without reading the container's own metadata.
- **Namespaces** are the objects a run creates, named by a pattern over declared fields, with
  sizes computed from the writes that create them, never observed. A namespace declared
  `input` holds the objects a previous run wrote (a checkpoint restore reads what the write
  left), and comes with that run's manifest.
- **Actors** are the concurrent programs: a template with a count (the reserved parameter
  `gpus` by default) whose instances have global ids. An actor's body is statements:
  loops, bindings, conditionals, weighted choices, phases, forks into sub-actors
  (`parallel`, and the PyTorch-shaped `loader`), channels, barriers, emulated compute, and
  the eighteen operations.

**Determinism.** Randomness is *positional*: a draw at a site (the JSON pointer of its node)
inside loops with indices `(i1 ... in)` on actor `a` is a pure function of `(seed, a, site,
i1 ... in)`. There are no draw counters, no shared random state, and no inputs but the seed,
the actor id, the site, the indices, the parameters, and dataset metadata: no clock, no
completion order. Every loop bound, width, and condition is an expression over these, so the
multiset of operations is fixed before the run starts and every instance can be computed
alone. `consume` draws a sample without replacement through a keyed Feistel permutation of
the dataset at position `g + G * (b * B + j)`, which splits an epoch over the instances with
no coordination; `x @ i` reads a binding as it was at an earlier index of its loop, so a
dependency on recent history is a formula, not a stored list.

**Identity.** The abstract's identity is the SHA-256 of its canonical form: the JSON with
keys sorted, no whitespace, ASCII escapes, floats in shortest round-trip form, and the
`provenance` block removed. A result is the abstract's hash, the parameters in effect, and
the dataset ids. The run's **fingerprint** is the sum modulo 2^64 over operations of a hash
of the operation's kind, actor, indices, effective offset, length, flags, and path; issue
order, phases, `expect` lists, and results are not hashed. Two runs with the same
fingerprint executed the same multiset of positioned operations.

**Backend.** The optional root key `backend` names the API the traced application issues
its I/O through (`sync`, `mmap`, ...); absent means `sync`. The runner uses it by default; the
operation stream and the fingerprint are the same under every backend.

## FORM

Every node, expression, distribution, and handle is a JSON object with exactly one key naming
its kind: `{"read": {...}}`, `{"add": [a, b]}`, `{"zipf": {"s": 1.1}}`, `{"ref": "f"}`.
Sizes are byte integers, durations nanosecond integers; floats appear only as
probabilities, exponents, and distribution parameters. Identifiers are
`[a-z_][a-z0-9_]*`. The top-level keys, in the builder's order: `ast` (the contract
version), `name`, `doc`, `backend`, `params`, `datasets`, `namespaces`, `actors`,
`provenance`.

The on-disk form is pretty-printed JSON, two-space indentation, suffix `.ast.json`. JSON
is the only format on the contract; nobody hand-writes an abstract.

## STATEMENTS

A body is an array of statements: thirteen control forms and eighteen operations.

| Kind | Semantics |
|---|---|
| `let {name, value}` | Bind a name to a positional value or a handle, visible to later siblings and child bodies, not to sibling bodies. A `let` is the way to use one draw twice. |
| `loop {index, from, to, step, body}` | `for index in [from, to) step step`. The index joins every random key inside. |
| `parallel {index, width, body}` | `width` concurrent sub-actors, index `0..width`, joined at the end. |
| `channel {name, capacity, ordered}` | A bounded channel in the enclosing actor. |
| `put {channel, seq}`, `take {channel}` | Deliver item `seq` (blocks while full); wait for the next item (in `seq` order if ordered). |
| `loader {name, index, workers, prefetch, batches, ordered, body}` | PyTorch's DataLoader: `workers` sub-actors, worker `w` builds batches `w, w+W, ...`, an ordered channel of `workers x prefetch` slots bounding batches started, exactly `batches` batches. |
| `barrier {scope}` | `global`, `host`, or a named group. |
| `compute {ns}` | Emulated compute, scaled by the runner's `--time-scale`. |
| `cond {if, then, else}` | A conditional over indices, parameters, bindings, and the actor id. |
| `choose {arms: [{weight, body}]}` | Run one arm, chosen positionally by weight. |
| `phase {name, body}` | A statistics label; no effect on execution or the fingerprint. |
| `trace {file, sha256}` | A captured trace of a real application, executed literally, one lane per traced task; for calibration, never comparable across systems as a scored workload. The file is **aeiou-trace**(1) `export`'s. |
| `open {file, flags, mode, expect}` | Flags from the schema's enum (`RDONLY`, `WRONLY`, `RDWR`, `CREAT`, `TRUNC`, `APPEND`, `CLOEXEC`, `DIRECTORY`, ...). |
| `close`, `fstat`, `stat`, `fsync`, `fdatasync`, `unlink` `{file, expect}` | |
| `read {file, len, offset, repeat, expect}` | With `offset`: positioned, the file position unchanged. Without: sequential from the current position. `repeat: until_eof` issues reads of `len` until the known size is exhausted, then the terminating short or zero-length read. |
| `write {file, len, offset, repeat, expect}` | As `read`, without `until_eof`. The bytes are positional content keyed by the namespace seed, the path, and the offset. |
| `lseek {file, offset, whence}` | `SET`, `CUR`, `END`. |
| `ioctl {file, request, expect}` | `TCGETS`, `FIONREAD`, `BLKGETSIZE64`. |
| `fadvise {file, advice, offset, len, expect}` | `posix_fadvise`: `NORMAL RANDOM SEQUENTIAL WILLNEED DONTNEED NOREUSE`. |
| `ftruncate {file, len}`, `fallocate {file, offset, len}` | |
| `mkdir {dir, mode, expect}`, `rmdir {dir}`, `rename {from, to}` | |
| `readdir {dir, repeat: until_end}` | One listing, `getdents64` until exhausted. |

`expect` on an operation lists the errno names that count as success (`ENOENT` on a `stat`
that probes for a file); any other failure aborts the run.

## EXPRESSIONS, HANDLES, DISTRIBUTIONS

**Expressions** are scalar integer arithmetic with floor division: literals; `param`,
`index`, `ref`, `at {ref, index}` (the binding as it was at another index of its loop),
`actor: id|count`, `draw dist`; `elem {array, index}`, `len`, `sum` over parameter arrays;
`size`, `offset`, `unit_index`, `units`, `chunks` of a handle, `count` and `dirs` of a
dataset; `add sub mul div mod ceil_div min max neg`; `eq ne lt le gt ge and or not`; and
`cond {if, then, else}` whose arms may be expressions, distributions, or handles.

**Handles** name what an operation acts on: `ref` (a binding); `file {dataset, id}`; `dir
{dataset, id}`; `object {namespace, fields}`; `consume dataset` (the next sample without
replacement); `pick {dataset, dist}` (a sample with replacement, uniform or by a
distribution over ids); `unit {of, index}` and `column {of, index}` of a container; `file
{of, chunk}` (the container file, or chunk object `k`, of a sample handle). Under `stream`
access `consume` and `pick` return file handles: the shuffle runs over shards.

**Distributions**: `const`, `uniform {lo, hi}` (integers in `[lo, hi)`), `uniform64`,
`normal {mean, sd, min, max}`, `lognormal {median, sigma, min, max}`, `empirical {values,
weights}`, `zipf {s}`, `hotset {fraction, weight}`, `mixture [{weight, dist|null}]`.
Continuous draws round to the nearest integer wherever an integer is consumed. The popular
ids of `zipf` and `hotset` are fixed by the dataset seed, not the run seed.

## DATASETS, NAMESPACES, ACTORS

**Datasets**: `files {pattern, count, size, seed, access, samples_per_file, chunk, format}`
or `regions {file, count, slot, size, seed}`. `count` is samples. Patterns use
`{field[:format]}` with the field `id` (and `k` for chunk objects) and the forms `{id:09}`,
`{id div 1300:05}`, `{id mod 16}`, `{conv:016x}`; a dataset's directories are the distinct
prefixes up to the last `/`, enumerated in id order. `access` is `map` (the shuffle runs
over samples) or `stream` (over files). A `format {class, reader, version, layout}` block
declares the container layout: a file is `file_header || units || file_footer`, a unit is
`unit_header || column chunks || unit_footer`, a column chunk is `header || rows`, a row is
`row_header || fixed || its share of the sample || row_footer`, each with its alignment, and
every field an expression over parameters.

**Namespaces**: `{pattern, fields, size, seed, input, same_run}`. `size` is an expression or
`as_written`, the sum of the writes that created the object. `input` marks objects a previous
run wrote, read-only here; `same_run` marks a namespace read under names the writing run
drew, so a run without the writer's seed, instance count, and common parameters is refused.

**Actors**: `{count, body}`; `count` defaults to `{param: gpus}`. Instances have global ids
`0..count`.

**Parameters**: `params` maps each name to a default (a scalar, a distribution, or an
array), with `unit`, `doc`, and `cli` (whether `--param` may override it). `gpus` is
reserved and set by `--gpus`.

## SEMANTIC RULES

The schema fixes the structure; the validators (`schema/check.py` and `aeiou check`) add
these, and the builder enforces most of them at construction:

- **V1** Every name resolves. **V2** `gpus` is not declared. **V3** an `at` on a binding of
  the same loop body uses an index provably below the current one (`i - e` with `e >= 1`).
- **V4** Sizes are computed, never observed: `until_eof` on an `as_written` object only
  through the handle its writes used; `fstat` results are never consumed.
- **V5** Finite and timing-free by construction. **V6** `consume` sits inside a loop.
- **V7** Object fields match the namespace's. **V8** `empirical` weights match its values.
- **V9** The access mode is one the format class supports. **V10** Continuous draws round.
- **V11** A `let` is visible to later siblings and child bodies only.
- **V12** Datasets are read-only: no write, truncate, allocate, unlink, or rename on a
  dataset handle, no write-mode `open`.
- **V13** No path component begins with `.aeiou`; dataset roots are distinct and not nested.
- **V14** Input namespaces are read-only and come with the writer's manifest.
- **V15** `same_run` needs `input`.
- **V16** (runner) A `trace` that creates files runs in one instance only.

## PARAMETER FILES

A parameter file (`<abstract>.<set>.params.json`, schema `schema/params.schema.json`)
fills the shape's slots: `{"params_version": 1, "abstract", "ast_sha256"?, "doc"?,
"params": {name: value}, "provenance"?}`. A value keeps the kind of the default it
replaces. `aeiou run AST --params-file A --params-file B --param k=v` applies the defaults,
then *A*, then *B*, then `--param`. **aeiou-params**(1) writes, checks, and builds them.

## MANIFESTS

**`.aeiou-dataset.json`**, at each dataset root, written by `aeiou datagen` or
**aeiou-datagen**(1), last and atomically. Its normative part, compared by `aeiou run` for
exact equality with what it resolves: `dataset` (the entry with every parameter reference
substituted, in canonical form), `payload` (generator, version, dedupe and compression
ratios, block size, wrapper version), `format` when the dataset has a format class, and
`manifest_version`. Its SHA-256 is the **dataset id**, which `--expect-dataset-id` pins.
Provenance (the abstract's name and hash, every parameter value, host, times, files
written) is recorded, never compared. No relaxation: a larger dataset is not accepted for a
smaller declared count.

**`.aeiou-namespace.json`**, at every namespace root a run wrote: the resolved definition
of each namespace there, the abstract's name and hash, the seed, `gpus`, the parameters,
each rank's host and GPU range, start and finish times, and the objects created with the
GPU id that created each. A run declaring the namespace `input` requires it, refuses on a
differing definition (and, for `same_run`, on a differing seed, instance count, or common
parameter), reports the write-to-read gap (`--max-gap`), and counts the input objects it
opens on the host that wrote them (`--require-cold`; `--rank-rotate` avoids them).

A run refuses a non-empty output namespace root unless `--clean-namespaces` is given, and
a namespace root may not lie inside a dataset root.

## FILES

- `schema/abstract-ast.schema.json`

  The JSON Schema of the document (draft 2020-12), structural rules only.
- `schema/check.py`

  The reference validator: schema, semantic rules, canonical hash, operation counts.
  `aeiou check` must reject everything it rejects, and CI diffs their outputs.
- `schema/params.schema.json`

  The schema of a parameter file.
- `schema/examples/*.ast.json`, `schema/examples/params/`

  The committed abstracts, generated from `builder/abstracts/*.py` and never edited by
  hand, and their example parameter sets.

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-datagen**(1),
**aeiou-trace**(1).

The reference: `schema/README.md` (the contract: form, vocabulary, rules, positional
semantics, manifests, parameter files, version history); `runner/REFERENCE.md` §2 (the
hash functions, the permutation, the `consume` position, `x @ i`, the fingerprint);
`builder/REFERENCE.md` §1 (the authoring API); `ABSTRACTS.md` (the workloads as written and
the constructs they needed); `GRAMMAR_OPTIONS.md` (the semantic model and why).
