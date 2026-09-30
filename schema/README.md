# The AST contract

Version `0.1`, drafted 2026-09-30. This directory is layer 2 of the three-layer design in
`GRAMMAR_OPTIONS.md` Option D: the Python builder (layer 1) emits an AST; the Rust runner
(layer 3) loads, validates, and executes it. The AST is the only thing the runner executes and
the only artifact the WG publishes with a hash. Nothing here depends on Python at run time.

| File | What it is |
|---|---|
| `abstract-ast.schema.json` | JSON Schema (draft 2020-12) for the AST. Structural rules only. |
| `check.py` | Reference validator: schema plus the semantic rules of §4, canonical hash, op counts. The Rust validator must reject everything it rejects. |
| `examples/*.ast.yaml` | Abstracts from `ABSTRACTS.md` in AST form, hand-written to test the schema; the builder will regenerate them. |

```
python3 schema/check.py                      # all examples
python3 schema/check.py path/to/x.ast.yaml   # one file
```

## 1. Form

- **Externally tagged.** Every node, expression, distribution, and handle is a JSON object with
  exactly one key naming its kind: `{read: {…}}`, `{add: [a, b]}`, `{zipf: {s: 1.1}}`,
  `{ref: f}`. This is serde's default enum encoding and the form of the Option B sketch.
- **Canonical units.** Sizes are byte integers, durations are nanosecond integers. `1MiB` and
  `105ms` are builder spellings. Floats appear only as probabilities, exponents, and
  distribution parameters.
- **Identifiers** are `[a-z_][a-z0-9_]*`. `gpus` is a reserved parameter set by `--gpus`.
- **Canonical form and hash.** The on-disk form is YAML for readability. The identity of an AST
  is the SHA-256 of its canonical JSON: keys sorted, no whitespace, ASCII escapes, floats in
  Python `repr` (shortest round-trip), the `provenance` block removed. `check.py` prints it;
  the runner prints it with every result.
- **No site ids.** A draw's site is its structural path in the tree (the JSON pointer of the
  node). Two builds of the same source therefore agree on every site without coordination, and
  reordering two independent statements changes the sites, which is intended: they are
  different abstracts.

## 2. Vocabulary

**Top level:** `ast` (version), `name`, `doc`, `params`, `datasets`, `namespaces`, `actors`,
`provenance`.

**Statements** (a `body` is an array of these). Thirteen control forms, seventeen ops.

| Kind | Semantics |
|---|---|
| `let {name, value}` | Bind a name to a positional value or a handle. Visible for the rest of the enclosing body and in child bodies; not in sibling bodies. |
| `loop {index, from, to, step, body}` | `for index in [from, to) step step`. The index joins every RNG key inside. |
| `parallel {index, width, body}` | `width` concurrent sub-actors, index `0..width`, joined at the end. |
| `channel {name, capacity, ordered}` | Declare a bounded channel in the enclosing actor. |
| `put {channel, seq}` / `take {channel}` | Deliver item `seq` (blocks while full) / wait for the next item (in `seq` order if ordered). |
| `loader {name, index, workers, prefetch, batches, ordered, body}` | PyTorch sugar the runner keeps: `workers` sub-actors, worker `w` builds batches `w, w+W, …`; an ordered channel `name` of `workers × prefetch` slots; exactly `batches` batches. |
| `barrier {scope}` | `global`, `host`, or a named group. |
| `compute {ns}` | Emulated compute; scaled by `--time-scale`. |
| `cond {if, then, else}` | Statement conditional over indices, parameters, bindings, and the actor id. |
| `choose {arms: [{weight, body}]}` | Run one arm, chosen positionally by weight (the original *selection*). |
| `phase {name, body}` | Statistics label; no effect on execution or the fingerprint. `name` may be an expression. |
| `replay {trace, sha256}` | Literal captured sequence for calibration; never CLOSED. Trace format deferred. |
| `open {file, flags, mode, expect}` | Flags from the enum in the schema. `DIRECTORY` for `readdir` handles. |
| `close`, `fstat`, `stat`, `fsync`, `fdatasync`, `unlink` `{file, expect}` | |
| `read {file, len, offset, repeat, expect}` | With `offset`: positioned (`pread`), file position unchanged. Without: sequential from the current position. `repeat: until_eof` issues reads of `len` until the known size is exhausted, including the terminating short or zero-length read; each returned count is checked against the computed size. |
| `write {file, len, offset, repeat, expect}` | As `read`, without `until_eof`. |
| `lseek {file, offset, whence}` | `SET`, `CUR`, `END`. |
| `ioctl {file, request, expect}` | `TCGETS` (Python's isatty check), `FIONREAD`, `BLKGETSIZE64`. |
| `ftruncate {file, len}`, `fallocate {file, offset, len}` | |
| `mkdir {dir, mode, expect}`, `rmdir {dir}`, `rename {from, to}` | |
| `readdir {dir, repeat: until_end}` | `getdents64` until exhausted. |

**Expressions** (scalar, positional, integer arithmetic with floor `div`):
literals (`null` is `none`); `param`, `index`, `ref`, `at {ref, index}`, `actor: id|count`,
`draw dist`; `elem {array, index}`, `len`, `sum` over parameter arrays; `size`, `offset`, `unit`,
`chunks` of a handle, `count` and `dirs` of a dataset; `add sub mul div mod ceil_div min max
neg`; `eq ne lt le gt ge and or not`; `cond {if, then, else}` whose arms may be expressions,
distributions, or handles (this is how a distribution or a phase name is selected by index).

**Handles:** `ref` (a binding); `file {dataset, id}` (a dataset file by id) or
`file {of, chunk}` (the container file, or chunk object `k`, of a sample handle); `dir {dataset,
id}`; `object {namespace, fields}`; `consume dataset` (next sample without replacement);
`pick {dataset, dist}` (a sample with replacement, uniform or by a distribution over ids). A
sample handle used where a file is expected means its container file.

**Distributions:** `const`, `uniform {lo, hi}` (integer, `[lo, hi)`), `uniform64`, `normal
{mean, sd, min, max}`, `lognormal {median, sigma, min, max}`, `empirical {values, weights}`,
`zipf {s}`, `hotset {fraction, weight}`, `mixture [{weight, dist|null}]`. Wherever the schema
says `distref`, a `param` or a `cond` that evaluates to a distribution is also accepted.

**Datasets:** `files {pattern, count, size, seed, access, samples_per_file, chunk, format}` or
`regions {file, count, slot, size, seed}`. `count` is samples. Sizes are drawn from the dataset
seed, never from `--seed`. Patterns use `{field[:format]}` with fields `id` (and `k` for chunk
objects) and the forms `{id:09}`, `{id div 1300:05}`, `{id mod 16}`, `{conv:016x}`; a
dataset's directories are the distinct prefixes up to the last `/`, enumerated in id order.

**Namespaces:** `{pattern, fields, size, seed}` for objects the run creates. `size` is an
expression or `as_written` (§4, rule V4).

**Actors:** `{count, body}`; `count` defaults to `{param: gpus}`. Instances have global ids
`0..count`; `{actor: id}` reads it.

## 3. Where the `ABSTRACTS.md` §9 constructs went

| §9 | AST form |
|---|---|
| 9.1 actor id in conditions and names | `{actor: id}` in any expression; `cond` on it |
| 9.2 `until_eof` semantics | `read.repeat: until_eof` |
| 9.3 `expect` | `expect: [errno…]` on any op |
| 9.4 offsets and counts as expressions | `read.offset`, `loop.to`, `write.len` are expressions over `size`, `offset`, `elem`, indices; `member_off`-style helpers are expanded by the builder |
| 9.5 `x @ i` and chains | `{at: {ref: x, index: e}}`, rule V3 |
| 9.6 `regions` | the `regions` dataset kind |
| 9.7 `namespace`, `chunk` | `namespaces`, `object` handles, `files.chunk`, `chunks(h)`, `file {of, chunk}` |
| 9.8 `phase`, `dirs`, `readdir` | `phase`, `{dirs: ds}`, `dir` handles, `readdir` |
| 9.9 `when` selecting a distribution | expression `cond` with distribution arms |

**Builder sugar that never reaches the AST:** `every N` (a `cond` on `mod`), integer
`op[n]` repeats (a `loop` with a generated index), `recent(x, d)` (`x @ (i − d)`), parameter
tables (parallel parameter arrays), `file("literal/pattern")` (a namespace plus an `object`
handle), unit suffixes, and the format-class protocols (`GRAMMAR_OPTIONS.md` §6.4), which
serialize as ordinary ops.

## 4. Semantic rules the validator adds

The schema cannot express these; `check.py` does, and the Rust validator must.

- **V1 Names resolve.** Every `param`, dataset, namespace, channel, index, and `ref` is
  declared in scope. `at` may name a binding defined later in the same loop body.
- **V2 `gpus` is reserved.** It may not appear in `params`.
- **V3 `at` decreases.** A self-reference (`at` on the binding being defined) or a forward
  reference must use an index of the form `{sub: [{index: i}, e]}` where `i` is the innermost
  loop index and `e` is provably `≥ 1`: a positive literal, or a draw from a distribution with
  `min ≥ 1` (`uniform` with `lo ≥ 1`, `empirical` with all values `≥ 1`, a `mixture` whose
  non-null arms all qualify). The runtime guard for a null arm is the author's `cond`. An `at`
  whose index falls below the loop's `from` evaluates the binding's non-recursive arm (the
  "fresh draw" of `ABSTRACTS.md` §9.5).
- **V4 Sizes are computed, never observed.** For a namespace with `size: as_written`, an
  object's size is the sum of the write lengths in the sequence that created it. `until_eof`
  on such an object is allowed only through the handle binding those writes used, which pins
  it to the creator's position. Elsewhere a reader states its length. `fstat` results are
  never consumed by the abstract.
- **V5 Finite and timing-free by construction.** Every loop bound, width, repeat, and
  condition is an expression, and expressions have no inputs except the seed, the actor id,
  the site, indices, parameters, and dataset metadata. There is no clock, no completion order,
  and no counter to read. The op multiset is therefore fixed before the run.
- **V6 `consume` needs a position.** It must sit inside at least one loop (normally the
  loader's batch loop and an item loop); its position is `actor + count × (b × B + j)` over
  the enclosing loader/loop indices, epochs are `drop_last`.
- **V7 Object fields match** the namespace's declared fields exactly.
- **V8 `empirical`** has as many weights as values.
- **V9 Access mode.** `map` over a container the format class cannot address randomly is
  refused by the builder; the runner checks the dataset against the manifest at startup.
- **V10 Continuous draws round** to the nearest integer wherever an integer is consumed
  (sizes, counts, indices, nanoseconds).
- **V11 Scope.** A `let` is visible to later siblings and to child bodies, not to sibling
  bodies (bind a handle above two phases that share it).

## 5. Positional semantics, stated once

- A draw at site `s` inside loops with indices `(i₁ … iₙ)` on actor `a` is
  `hash(seed, a, s, i₁ … iₙ)`. The same site at the same position yields the same value, so a
  `let` is the way to use one draw twice, and `at` is the way to use a draw made at another
  index.
- Dataset sizes and layouts are functions of `(dataset seed, id)`; namespace content is a
  function of `(namespace seed, hash(path), offset)`.
- The fingerprint is the sum over ops of `hash(op kind, path, offset, len, flags)`, summed by
  the coordinator across actors; `phase`, `expect`, and results are excluded; issue order is
  never hashed. Its exact definition belongs to the runner and is not fixed by this schema.

## 6. Deferred (each is a version bump)

- The `replay` trace file format.
- Runner-facing container layout fields beyond `samples_per_file` (per-unit headers and
  footers, sample overhead), which the format classes will need for `map` access over HDF5,
  Arrow, and MDS.
- A records-per-batch knob on `loader` for `stream` access (`GRAMMAR_OPTIONS.md` §6.2).
- Wall-clock-bounded phases (the opt-in that turns off the fingerprint guarantee).
