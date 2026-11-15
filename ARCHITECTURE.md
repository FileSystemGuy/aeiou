# ARCHITECTURE.md

**Status: outline only (2026-11-14).** No section is written yet. This file holds the outline the
user approved on 2026-11-14 and the review it came from, so neither is lost. Decided the same day:
one document, the proposals inside it (each marked as a proposal until decided), at the top level
under this name. Not yet done: the sections themselves, the entry in `CLAUDE.md`'s reading list,
the dated note in `DESIGN_REVIEW.md`, and the brief items for the proposals.

This document is a map and a rulebook, not a second reference. Where a definition already lives
elsewhere (`schema/README.md`, `runner/REFERENCE.md`, `builder/REFERENCE.md`), it points there and
does not restate it.

Each section below is tagged:

- **exists**: points to current text.
- **new**: writes down what is true today but unwritten.
- **proposal**: not built, not decided; needs the user's decision before it is stated as fact.

## The model in one table

The framing chosen 2026-11-14: a compiler and a CPU, with the runner's two halves as the
microarchitecture under the instruction set.

| Compiler and CPU | aeiou |
|---|---|
| Source language, compiler, standard library | The Python builder and its library functions (`lru()`, the format classes, `opened()`) |
| Instruction set, the published contract | The AST (contract 0.21): versioned, hashed, the only thing the runner executes |
| Architectural state | Positional values, a lane's `prev`, and timeline state (entries, tiers, resources) |
| Architectural result | The op multiset and its planned times, which is what the fingerprint names |
| Microarchitecture, front end | The planner: decodes the AST into a stream of planned ops |
| Micro-ops | The planned-op stream, the trace file `dry-run --emit-trace` writes |
| Microarchitecture, back end | The executor and the I/O backends: issue against real time |
| Timing state, invisible to the program | Lag, staging, channel occupancy, D |

The closest relatives are eBPF (a small restricted instruction set, a verifier, growth by map
types and helpers) and SQL (a planner and an executor, `EXPLAIN`), not Java or Python.

## Outline

### Part 0: The model

- **0.1 The three layers** (new). The language and its library (the Python builder), the
  instruction set (the AST contract), the microarchitecture (the runner's planner, executor, and
  backends). The table above.
- **0.2 Where the analogy bends** (new). Timing is the product, not incidental; the executor has
  no branches and no data-dependent behaviour; the planned-op stream is public where a CPU's
  micro-ops are hidden; decode is lazy, a bounded horizon ahead of the executor; the builder lowers
  and never rewrites, since the abstract is the application's call stream.
- **0.3 Why the language is restricted** (new). Not general-purpose, as SQL and eBPF are not,
  because describing, bounding, and proving an abstract is the point. A general VM (WebAssembly,
  Lua) would end runner changes for generator logic and make the abstract opaque: no description
  in prose, no proof it is finite, no op counts or write sets without running it, no `x @ i`.
  Points to `GRAMMAR_OPTIONS.md` Option C.
- **0.4 What each kind of change costs** (new). Library: nothing downstream. Instruction set:
  every AST regenerated, every case relocked. Microarchitecture: no fingerprint moves and no
  contract version.

### Part I: The instruction set

- **1. Architectural state** (new). Four classes and what orders each: positional values (a pure
  function of seed, actor, site, indices); a lane's carried values (`prev`); timeline state in
  planned order (the published log, tiers, resources); and nothing else. The fingerprint as the
  architectural result.
- **2. The frozen core** (new, mostly exists). `let`, `loop`, `cond`, expressions, draws, handles,
  the ops, `compute`, `parallel`. Points to `schema/README.md`.
- **3. Maps** (new framing; one proposal). The state structures a timeline may declare: the tier,
  the published log, the counted resource, the `array`. For each: its state, its order, what
  bounds its memory.
  - *Proposal:* the tier becomes a general keyed table (review suggestions 1 and 2 below).
- **4. Helpers** (new). `probe`, `touch`, `store`, `lookup`, `acquire`, `release`, `publish`,
  `submit`, `span`, each with a declared effect signature.
- **5. The verifier** (new; one proposal). The effect classes: issues an op, moves the planned
  clock, waits in planned time, reads or writes a map, spawns lanes, run-side synchronisation. A
  table of which effects each context allows (a timeline's lane, a `span`, a `submit` body, an
  `on_evict` body, outside a timeline).
  - *Proposal:* rules V30 to V36 restated as derived from that table. Whether the two checkers'
    code follows is a separate decision.
- **6. Instruction, intrinsic, library** (new). The test: if the builder can lower it to existing
  instructions with the same op set, it is library. An intrinsic is what could be library in
  principle and the runner carries for a global order or efficient state (the tier's sorted victim
  index). The current intrinsics, and which are candidates to leave (`loader`, `lookup`).
- **7. The growth policy** (proposal). The core is frozen. Capability grows by adding a map type
  or a helper with a declared signature. An addition never moves an existing fingerprint. A new
  piece is provisional until a second workload uses it.
- **8. The definitions fingerprints depend on** (exists). Points to `runner/REFERENCE.md` §2.
- **9. The conformance kit** (proposal). A published set of small ASTs with their fingerprints and
  the shared refusal cases, named as the definition of a correct runner. Largely the golden tests
  and the two checkers' shared cases today.

### Part II: The microarchitecture

- **10. The planner** (exists). The walk and its bounded horizon. Points to `runner/REFERENCE.md`
  §2 and §4.
- **11. The planned-op stream** (exists; one decision). The trace file (§13). *Decision needed:*
  how stable this lower interface is.
- **12. The executor** (exists). The issue rule (`max(plan + L, previous completion, edges)`),
  lanes, lag, spans.
- **13. Run-side synchronisation** (exists; one proposal). Channels, barriers, staging, validity
  edges.
  - *Proposal:* events (brief item 72 (6)) as the one base, the others lowered onto it inside the
    runner (review suggestion 3).
- **14. Backends and the coordinator** (exists). Points to `runner/REFERENCE.md` §6, §8, §9, §16.
- **15. The wall** (new). Timing state never reaches architectural state: "never learn from the
  run" (`DESIGN_REVIEW.md` §3.99) in this document's terms. What is reported, what is a gate
  failure, what is never modelled.

### Part III: The library

- **16. Lowering** (new, short). What a library function may and may not do.
- **17. The idiom catalogue** (proposal, to be costed). Each library function carries its lowering
  and its trace signature, so the digest names the idiom and the builder emits it. First entries:
  Python's `open()` around one read or write (in eight abstracts today), the format classes'
  readers. To be queued as a brief item; not designed here.

### Part IV: The fit specification (a trace to an abstract)

- **18. Why this is decompilation** (new). One trace, many abstracts; judgment and outside sources
  are needed; an agent does the work. So the specification is of the acceptance test, not of the
  agent's process (translation validation).
- **19. The artefacts in order** (exists, scattered). Trace, digest, interview answers, authoring
  script, parameters, case, capture. Points to `builder/REFERENCE.md` §7 and §8.
- **20. The acceptance test** (new, the core of this part). An abstract fits a trace when its own
  emitted trace (`dry-run --emit-trace`), digested, compares (`aeiou-trace compare`) with the
  original's digest within the capture's tolerances. What is compared; what a tolerance may and
  may not excuse.
- **21. Provenance** (exists as practice). Every claim tagged with its source and confidence;
  primary sources before backfitting.
- **22. What the agent may not do** (exists as rules). No emulator as a source, no trace listing
  as an abstract, no dropped truncates, nothing learned from a run.

### Appendix

- Glossary: our term beside the familiar one (the describer and a disassembler, `dry-run` and
  `EXPLAIN`, the validator and a bytecode verifier).
- Open decisions, listed once.

## Writing order

The new sections first (0, 1, 5, 6, 15, 20), since the rest is pointers.

## The review behind this outline (2026-11-14)

The user asked for a critical look at the DSL's primitives after the tier's policies moved into
AST expressions (contract 0.18). Counted against the 14 committed ASTs:
109 primitives (43 statements, 48 expressions, 9 distributions, 9 handles).

| Group | Primitives | Abstracts using it |
|---|---|---|
| Core: `let`, `loop`, `cond`, arithmetic, `open`/`read`/`close`, `lseek`, `fstat`, `compute`, `phase` | about 35 | 10 to 14 |
| Real-time concurrency: `parallel`, `barrier`, `loader`, `channel`, `take`, `consume` | 6 | 5 to 8 |
| Positional structure: `at`, `pick`, the `object` handle, `write` | 4 | 3 to 6 |
| Timeline family: `timeline`, `publish`, `array`, `lookup`, `probe`, `touch`, `store`, `acquire`, `release`, `submit`, `span`, and 12 expressions (`prev`, `entry`, `published`, `fresh`, `shared`, `prefix`, `cat`, `seq_len`, `chunk_key`, `cached`, `tier`, `victim`) | 23 | 1 (`lmcache_local_disk_tiered`) |
| Container and listing queries: `unit`, `units`, `inventory`, `dir`, `dirs`, `readdir` | 6 | 1 to 2 |
| Unused by any committed abstract: `choose`, `put`, `trace`, `fallocate`, `fdatasync`, `rmdir`, `truncate`, `ftruncate`, `normal`, `lognormal`, `hotset`, `mixture`, `column`, `sum`, `unit_index`, `chunks`, `count`, `now` | 18 | 0 |

The finding: the DSL is RISC at its edges (ops, expressions) and CISC in the middle (the
concurrency constructs and the cache verbs). The session generator, the most complex thing
written, needed no generator-specific runner primitive.

Six suggestions, none decided:

1. **Cache verbs take keys, not chunks of a sequence** (instruction set). `lookup`, `probe`,
   `touch`, `store` take `(seq, per, from, count)` today; brief item 75 said keys are opaque to
   the cache. A key range (an index, a count, a key expression), `chunk_key(seq, j, per)` one such
   expression. With `hold` and `full: wait` on `store`, `lookup` becomes a builder function. Gives
   the cache its second user (DiskANN's node cache, an epoch cache, a local disk cache over an
   object store).
2. **An eviction body may act on another tier; entries carry values** (instruction set).
   `on_evict` bodies are barred from every cache statement today, so "evicted from G2, written to
   G3" cannot be said: Dynamo's cascade (item 76), G3 with G4 (item 78), S3-FIFO. Entries carrying
   values make the tier a keyed table (routing, allocation). A trigger on tier state running a
   body might express a vector database's compaction; unchecked against any source.
3. **One run-side synchronisation primitive** (microarchitecture first). Four today (`channel`
   with `put`/`take`, `barrier`, `workers` with staging, validity edges), `loader` a composite of
   the first, and item 72 (6)'s events a fifth. Build events as the base and lower the others onto
   it. `loader` is exactly PyTorch; multi-stage pipelines (DALI, `tf.data`, Ray Data) need the raw
   pieces. The catch: `consume`'s position formula and the step times are defined through
   `loader`.
4. **`span` everywhere, merged with `phase`** (instruction set). A measured region is
   timeline-only; time to load a model, time to save a checkpoint, and query latency are those
   workloads' headline numbers.
5. **A quantile-table distribution** (instruction set). The tiered script builds inverse CDFs by
   hand in fixed point, two draws and about eight nodes a sample at about 40 sites. A
   piecewise-linear quantile table is what a fit produces and is integer-only; it would supplant
   `normal`, `lognormal`, `hotset`, `mixture`. Untested risk: `rng.rs` computes normals with `ln`,
   `sqrt`, `cos` in floating point, and releases are x86_64 and aarch64.
6. **Ops as table rows** (the runner's internals). `fadvise` is named in 21 source files, about
   half capture-side; item 58 adds about 25 calls, item 61 object requests, item 76 possibly block
   commands. Not costed against the dispatch code.

Considered and not recommended now: functions in the AST (the tiered abstract's 380 lines of
Python expand to a 1.6 MB AST, but Python functions already give the builder the reuse).

Coverage for the next year or two, as judged in the review: covered today, training pipelines,
checkpoints, model loading, one-engine KV with G1 to G3; needing 1 and 2, vector search with
caches, G3 with G4, Dynamo's cascade; needing 3 and item 72 (5), several engines sharing a tier;
needing a new op family (6), object-native requests and block or NVMe-KV devices; excluded by
design, anything adaptive to the run.

Three things taken from pushing the analogy further (the user's "yes", 2026-11-14): the fit
specification as translation validation (Part IV), the eBPF growth policy (section 7), and the
idiom catalogue (section 17).
