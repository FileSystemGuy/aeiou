# ARCHITECTURE.md

**Status: first sections drafted (2026-11-14).** Written: Part 0, and sections 1, 5, 6, 15, and
20. Every other section is still its outline entry, tagged as below. Decided by the user on
2026-11-14: one document, the proposals inside it (each marked a proposal until decided), at the
top level under this name. The draft sections are for the user's review and are not yet held to
the code by any test, but for section 5, whose two tables `builder/tests/test_architecture.py`
holds to both checkers (2026-11-14). The reasoning is `DESIGN_REVIEW.md`
§3.100; what remains is brief item 79.

This document is a map and a rulebook, not a second reference. Where a definition already lives
elsewhere (`schema/README.md`, `runner/REFERENCE.md`, `builder/REFERENCE.md`), it points there and
does not restate it; where the two disagree, the reference is right and this document is the bug.

Each section is tagged:

- **exists**: points to current text.
- **new**: writes down what is true today but was unwritten.
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

## Part 0: The model

### 0.1 The three layers (new)

An abstract passes through three layers, each with its own specification and its own rate of
change.

1. **The language and its library** is the Python builder (`builder/REFERENCE.md`). A person or
   an agent writes an authoring script; running it once emits an AST. Python functions are the
   language's only means of abstraction: `lru()`, the format classes, a script's own helpers.
   Python never runs the workload and never reaches a client node.
2. **The instruction set** is the AST contract (`schema/README.md`, version 0.22). It is the
   published artefact: hashed, pinned by a case's lock, the only thing the runner executes. Its
   meaning is its architectural result, the op multiset and each op's planned time where the
   abstract plans time. The definitions that fix that result bit for bit are
   `runner/REFERENCE.md` §2.
3. **The microarchitecture** is the runner's inside: a planner that walks the AST and produces
   ops, an executor that issues them against real time, the I/O backends, and the coordinator
   between hosts. It may change freely so long as the architectural result does not.

A fourth activity sits beside these and is not a layer: getting from a trace of a real
application to an authoring script. That is Part IV.

### 0.2 Where the analogy bends (new)

- **Timing is the product.** A processor exists for its results and its timing is incidental.
  Here the results are fixed before the run and the timing of the system under test is what is
  measured.
- **The executor is less than a processor.** The stream it issues has no branches and no
  behaviour that depends on data. It is closer to a score being played than to a program being
  run. This is deliberate (section 15).
- **The micro-ops are public.** A processor hides its micro-ops. The planned-op stream is the
  trace file that `dry-run --emit-trace` writes and the `trace` node runs, so it is a second,
  lower interface, whose stability is an open decision (section 11).
- **Decode is lazy.** The planner never holds the whole stream, only a bounded horizon ahead of
  the executor, since nothing per file or per op may be materialised.
- **The compiler never optimises.** The abstract is the application's call stream, local calls
  included, so the builder lowers and never rewrites.

### 0.3 Why the language is restricted (new)

The instruction set is not general-purpose, in the way SQL and eBPF are not. Loops are loop nodes
with bounds that are expressions, draws are nodes keyed by position, parameters are references.
That restriction is what lets tools that never run the workload say what it will do: the
validator proves an abstract finite and its self-references well-founded; `dry-run` projects op
counts, bytes, and write sets; the describer writes the abstract in prose; the lock narrows
parameters; `x @ i` recomputes a value at another index because the value is a pure function of
its position.

Embedding a general virtual machine (WebAssembly, Lua) would end changes to the runner for
generator logic, and every one of those properties would go with it. This was Option C of
`GRAMMAR_OPTIONS.md` §3, and the reason it was not taken still holds.

### 0.4 What each kind of change costs (new)

| Change to | What it moves | What it costs |
|---|---|---|
| The library | Nothing, if the emitted AST is the same | A builder release |
| One abstract's script | That abstract's AST hash, and its fingerprint if its ops changed | Its cases relocked, its shapes re-measured |
| The instruction set, by addition | The contract version; no existing fingerprint; today every AST hash and so every workload id (section 7.5) | Every AST regenerated at the new version, every case relocked, every shape re-measured |
| A definition a fingerprint depends on | Every fingerprint or every dataset id | A recorded decision (`CLAUDE.md`); to be avoided |
| The microarchitecture | Nothing published | The runner's own tests |

The aim of the growth policy (section 7) is that a new development in the AI stack lands in the
first three rows and never in the fourth.

## Part I: The instruction set

### 1. Architectural state (new)

Architectural state is everything the op set is computed from. There are three classes, and the
property that matters about each is what orders it.

**Positional values.** A draw at site `s`, inside loops with indices `(i1 ... in)`, on actor `a`,
is `hash(seed, a, s, i1 ... in)` (`schema/README.md` §5). Everything built from draws, indices,
parameters, and the actor id by expressions is positional: a pure function of where it stands.
Dataset layouts are functions of `(dataset seed, id)`; a namespace's object sizes are computed
from the writes' expressions. Positional state needs no order at all, which is why `x @ i` can
recompute it anywhere, why actors may run in any interleaving, and why most abstracts need
nothing else. Thirteen of the fourteen committed abstracts are purely positional.

**A lane's carried values.** `prev` reads the value a `let` took in the loop's previous
iteration, rather than recomputing it. It exists for values that depend on timeline state and so
are not positional (V31). It is ordered by the lane's own sequence, and no other lane sees it.

**Timeline state.** A `timeline` declares state its lanes share, and gives it the one thing
shared state needs, a total order that does not come from the run: planned time, a tie going to
the smaller lane key, with a tier's timed eviction before any lane's step at its instant and the
worker lanes after every other lane at theirs. The state is:

| State | Written by | Read by | Memory bounded by |
|---|---|---|---|
| Each lane's planned clock | `compute`; a wait for a resource, a `full: wait` tier, or a join | `now` | The live lanes |
| The published log, by group | `publish` | `published`, `entry` | The entries published (a known gap: nothing bounds it but the abstract) |
| A tier's entries and their attributes (`stored`, `touched`, `freed`, `hits`, `pos`, `size`), and which a lane holds | `store`, `touch`, `release`, an eviction | `probe`, the tier's `order` (through `cached`), its `evict` (through `tier`) | The tier's capacity, a parameter |
| A resource's free units and its queue of waiting lanes | `acquire`, `release` | `acquire` | The live lanes |
| An `array`, and a `store`'s `taken` | Where it stands, once | `elem`, `len` | Its count |

**There is nothing else.** Outside a timeline, actors and the sub-actors of a `parallel` share no
architectural state, because nothing orders them but the run. That is the reason a cache needs a
timeline, and it is the test for any future shared state: name its order, and the order may not
come from the run.

**The architectural result** is the op multiset, which the fingerprint names (the sum of per-op
hashes, order-independent), and, in a timeline, each op's planned time and lane. Issue order
within a GPU outside a timeline is not architectural and is never hashed.

### 2. The frozen core (new, mostly exists; not yet written)

`let`, `loop`, `cond`, expressions, draws, handles, the ops, `compute`, `parallel`. Points to
`schema/README.md` §2.

### 3. Maps (new framing; one proposal; not yet written)

The state structures a timeline may declare: the tier, the published log, the counted resource,
the `array`. For each: its state, its order, what bounds its memory (the table in section 1 is
the start).

- *Proposal:* the tier becomes a general keyed table (review suggestions 1 and 2 below).

### 4. Helpers (new; not yet written)

`probe`, `touch`, `store`, `acquire`, `release`, `publish`, `submit`, `span`, each with
a declared effect signature in section 5's terms.

### 5. The verifier (new; one proposal)

The validator's rules for timelines (V30 to V36) are written as lists: each context names the
statements it may not hold. The lists are long, each new statement must be added to several, and
the reason behind them is the same few properties each time. This section states the properties.

**The effects.** Every statement has a set of effects; a body's effects are those of everything
in it.

| Effect | Meaning | Statements |
|---|---|---|
| `op` | Issues an op to the system under test | The POSIX ops (`open` to `readdir`) |
| `time` | Moves the lane's clock (planned, in a timeline; a real delay outside one) | `compute` |
| `wait` | May wait in planned time for another lane | `acquire`; a store of `present: touch` (on a `full: wait` tier; ~~`lookup`~~, a builder function over it since contract 0.23) |
| `read` | Reads timeline state | `probe`; the expressions `published` and `entry` |
| `write` | Changes timeline state | `publish`, `touch`, `store`, `release` |
| `fork` | Starts lanes that plan their own time | `parallel` (joined, or with `join: false`) |
| `hand` | Hands ops to a worker lane | `submit` |
| `measure` | Marks a measured region | `span` |
| `sync` | Waits on another actor in real time | `channel`, `put`, `take`, `barrier`, `loader` |
| `nest` | Opens a scope of its own | `timeline`, `trace` |

`let`, `loop`, `cond`, `phase`, and `array` have no effect of their own. Two expressions
carry an effect, and give it to the statement they stand in: `published` and `entry` read the
timeline's log (contract 0.22; see "What the check found" below). `now` carries none: it is the
lane's own clock, accepted wherever a timeline encloses it.

**The contexts.** Each context allows a set of effects. The reason is in the last column.

| Context | Allows | Why |
|---|---|---|
| An actor's body outside a timeline | `op`, `time`, `fork` (joined), `sync`, `nest`, `measure` | No shared order exists, so no timeline state; `measure` since contract 0.24 |
| A `span`'s body outside a timeline | `op`, `fork` (joined) | Contract 0.24: its D is the elapsed time of its own ops and a joined `parallel`'s, so no emulated compute and no wait on another actor in it |
| A timeline's lane | `op`, `time`, `wait`, `read`, `write`, `fork` (joined or not), `hand`, `measure` | Lanes wait on one another only in planned time, so no `sync`; one planned order, so no `nest` |
| A `span`'s body | `op`, `read`, `write`, `hand` | Its D is a storage wait, so nothing in it may move or wait on the planned clock |
| A `submit`'s body | `op` | A worker's ops stand at the submitting lane's instant: it plans no time, and neither reads nor changes the timeline's state |
| A tier's `on_evict` body | `op` | The same, at the eviction's instant |

Read against `schema/README.md` §4, this table gives V30's list (a timeline's lane lacks `sync`
and `nest`), V35's (an eviction body allows `op` alone), V36's for `submit` (the same) and for
`span` (no `time`, `wait`, or `fork`, and no `measure`, so no span in a span), and V34's rule that
`acquire` and `release` stand only in a timeline.

**Held to the checker (2026-11-14).** Each of 26 statements (every control statement, one op,
`parallel` joined and not, and every timeline statement) was placed in each of the five contexts,
130 abstracts, and each was given to the Python checker (`schema/check.py`), which holds the same
lists as the runner's (`NOT_IN_TIMELINE`, `EVICT_BANS`, `SPAN_BANS`; `validate.rs`'s `evict_bans`
and `span_bans`). The table's verdict and the checker's agree in all 130. The shared refusal
cases (`runner/aeiou/tests/refusals.json`, 236) hold 21 placement refusals, every one of which
the table gives. Since the same day a test keeps them together: `builder/tests/test_architecture.py` reads the
two tables from this section, places every statement of the schema (43, and `parallel` not
joined) in every context, 220 abstracts, and holds `Check`, and `aeiou check` when the runner is
built, to refuse exactly the placements the tables refuse (68 of them; since contract 0.22
also `published` and `entry`, each in an op of every context, 230 abstracts and 74 refusals). A statement the schema
gains fails it until this section classifies it, and so does a ban list changed in a checker
alone. So the wording of the two tables is read by a test: an effect is a backticked word in
the first column, a statement a backticked name in the third, and a context is known by the
words of its first column.

**What the check found.** The first draft of this table gave the expressions `published`,
`entry`, and `now` the effect `read`, with `probe`. That is not what the checkers do: both accept
those expressions in a `submit` body and in an `on_evict` body, where `probe` is refused, and
`aeiou dry-run` walks them there. So a statement's placement and an expression's were two rules,
and the table was the statements' alone. Two questions followed, both settled the same day
(below):

- `probe` is refused in those two bodies though it changes nothing, and the reason the rules
  give for the bodies ("it plans no time and changes no cache or timeline") does not cover it.
  Either the ban on `probe` there is stricter than it needs to be, or the reason should say that
  such a body makes no statement about the timeline at all.
- What `now`, `published`, and `entry` evaluate to inside an `on_evict` body (the tick's
  instant or the evicting store's, and the log as of when) is stated nowhere found in
  `schema/README.md` or `runner/REFERENCE.md`, and no test of `tests/evict.rs` reads one there.
  In a `submit` body the answer is given: the submitting lane's instant and bindings.

**Settled (2026-11-14, contract 0.22; the user's decision).** Both questions have one answer. A
`submit` body and an eviction body each run as a lane of their own, and that lane takes its step
only after the lane that submitted or evicted has run on to its next `compute` (a worker lane
after every other lane at its instant as well). So anything such a body read of the timeline,
a tier by `probe` or the log by `published` or `entry`, would be the state as the planner's
order left it, not a state the abstract names: with `publish`, a `store` that evicts, `publish`,
`compute`, the eviction body saw both publishes. That is the reason for the ban on `probe`, and
it covers the two expressions, which both checkers refuse there since contract 0.22. The rule
is one sentence: a body that only issues ops reads no state of its timeline. What it needs is
given to it: `victim`, the bindings it starts from (a `let` before a `submit` holds a
`published` as it stood there), and `now`, the planned instant the lane was made at, which
nothing in the body can move. So `published` and `entry` have the effect `read` in the table
above, and it is again one rule for statements and expressions.

What the table does not replace: the rules about names and declarations (a cache named once, a
`release` naming a hold some `acquire` takes), about what an expression may read (a tier's
`order` reads only `cached` and parameters), about `@` and `prev` not reaching a value that is
not positional, and the refusals made only when walked (a lane ending holding, a deadlock).

- *Proposal:* restate V30, V35, V36, and the placement part of V34 in `schema/README.md` as this
  table, each statement's effects beside its definition, so that a new statement declares its
  effects once and every context's rule follows. Whether the two checkers' code is rewritten to
  match is a separate decision; their messages and the shared refusal cases would not change.
- *What the table exposes.* Review suggestion 2 (an eviction body acting on another tier) is the
  `on_evict` context gaining `write`, and nothing else; suggestion 4 (`span` outside a timeline)
  is the first context gaining `measure`.

### 6. Instruction, intrinsic, library (new)

Three places a capability can live, and the test for which.

- **Library.** If a Python function over existing instructions emits the same ops, it is library.
  It costs nothing downstream and needs no decision. Today: `lru()` (an `order` expression),
  `timed()`, `quantile()` (a draw from a fit's quantile table: two draws and an interpolation), the format classes' readers (plain POSIX nodes), every script's helpers such as the
  six calls of Python's `open()`.
- **Intrinsic.** A capability the runner carries because it needs an order across lanes or
  state kept efficiently, neither of which expressions can give. Today: the timeline and its
  planned order; the tier (a sorted victim index) with `probe`, `touch`, `store`; the resource
  with `acquire` and `release`; the published log; `workers` and `submit`; `span`; the Feistel
  shuffle behind `consume`; `channel`, `put`, `take`, and `barrier`.
- **Instruction.** The core that neither of the above can be reduced to: binding, loops,
  conditions, expressions, positional draws, the ops, `compute`, `parallel`.

The contract also carries composites, statements that could be lowered to others and are in the
runner for historical reasons. None of these lowerings has been tested for an equal fingerprint;
each is a candidate, not a finding.

| Composite | Would lower to | What stands in the way |
|---|---|---|
| `loader` | `parallel`, `channel`, `put` | `consume`'s position formula and a run's step times are defined through the enclosing `loader` |
| ~~`lookup`~~ | ~~`probe`, `touch`, `store`~~ One `store` of `present: touch` with `found` | ~~`hold` and waiting on a `full: wait` tier exist only on `lookup`~~ Lowered and removed in contract 0.23 (2026-11-22): `store` took `hold`, the wait, and the hit count before the store (`found`), since a `probe` before a store that waits would bind a count gone stale; shown equal by the planned trace of the tiered case, line for line |
| `choose` | A draw and `cond` | ~~Nothing known; no committed abstract uses it~~ Nothing: the builder emits the lowering since 2026-11-22 (`choose(name)`, a draw of the arm's index over an `empirical` and a `cond` an arm), and the node left the contract in 0.23, the same day |
| ~~`mixture`, `hotset`~~ `mixture` | A draw and a `cond` between distributions | ~~Nothing known; no committed abstract uses either~~ Lowers in a `draw` and in a `pick`'s `dist`; not in a dataset's `size`, which is one distribution resolved once and drawn per id from the dataset seed, with no site a `cond` could be evaluated at, and `empirical` holds literals alone, so two `lognormal` modes of file sizes have no other form. It stays |

*Corrected 2026-11-22 (read from `eval.rs`, `vm.rs`, and `schema/README.md`; no lowering was
run for `mixture` or `hotset`):* `hotset` was listed here as a composite and is not one. It
draws a rank and maps it through the dataset's rank order (`rank_perm`, keyed by the dataset
seed), which no expression exposes; lowered over `uniform` its hot ids would be `0` to
`hot − 1`, neighbours in directories and container files, another workload to the storage. It
is a primitive beside `zipf`.

The rule this section proposes for the future: a composite is never added to the instruction
set. A new intrinsic needs a reason from the definition above (an order, or state), stated where
it is defined.

### 7. The growth policy (drafted and decided 2026-11-14; 7.5's version range ~~not built yet~~ built 2026-11-22)

The question this section answers is the one the review began with: how does the suite take a
new development in the AI stack without a change to the runner each time, and when a change to
the runner is right, what must it be held to. The model is eBPF's: a small instruction set that
does not grow, and capability added as **maps** (state structures with a declared order and
bound, section 3) and **helpers** (statements over them with declared effects, section 4).

**7.1 The ladder.** A new requirement is tried at each rung in turn and stops at the first that
holds it. Each rung costs more than the one before (section 0.4).

| Rung | The requirement is met by | Who decides | Example |
|---|---|---|---|
| 1 | Parameters of a committed abstract: a new case | The case's author | A second accelerator for a training job |
| 2 | The abstract's own script, or a library function, over existing instructions | The script's author | `lru()`, the session generator of the tiered KV abstract |
| 3 | A new helper on an existing map | A recorded decision | `probe`, `touch`, `store` over a range (contract 0.19) |
| 4 | A new map type, with its helpers | A recorded decision | The tier (0.16, 0.18), the counted resource (0.18) |
| 5 | A new op, or a new handle or expression of the core | A recorded decision | `truncate` (0.14) |
| 6 | A change to a definition fingerprints depend on | To be avoided (`CLAUDE.md`) | None since the release |

The test between rungs 2 and 3 is section 6's: a capability is an intrinsic only if it needs an
order across lanes or state kept efficiently, and the reason is written where it is defined. A
composite, something a library function over existing instructions would emit the same ops
for, is never added.

**7.2 What an addition declares.** A new helper or map type comes with these, in
`schema/README.md` beside its definition, before it is built:

1. **Its effects** in section 5's terms, so that every context's rule follows without a new
   list (`builder/tests/test_architecture.py` fails until the section classifies it).
2. **Its state** (a map): what it holds, the order it is read and changed in (positional, a
   lane's own, or planned order; section 1), and what bounds its memory. Never a structure per
   file.
3. **Its side of the wall** (section 15): which of its inputs are the abstract's, and that
   none is the run's. What a run observes of it has one of the four fates.
4. **Why it is not library**: the order or the state it needs.
5. **Its refusals**: the shared refusal cases both checkers are held to.

**7.3 What an addition never does.**

- It never changes the op multiset of a document that does not use it, so no existing
  fingerprint and no dataset id moves. The history in `schema/README.md` §9 records "no
  fingerprint and no dataset id changed" for every version from 0.15 to 0.22.
- It never makes the plan a function of the run (section 15).
- It never gives a committed abstract a second meaning: a document valid before and after
  means the same.

**7.4 Provisional and settled.** A new piece is **provisional** until a second mechanism uses
it; a settled piece is frozen. This is practice already: the history of `schema/README.md` §9
marks 0.15's, 0.16's, and 0.18's pieces "provisional until a second workload uses them". What
the policy adds is what the word permits: while provisional a piece may be narrowed, reshaped,
or removed with a contract version. Twice so far: contract 0.18 rewrote 0.16's `evict` enum
as `order` expressions and changed what a `lookup` touches, and contract 0.22 refused
`published` and `entry` in two bodies where 0.21 accepted them; each was possible because one
abstract, regenerated the same day, was the only user. The whole timeline family (23
primitives, one abstract; "The review behind this outline") is provisional today; 0.19's,
0.20's, and 0.21's pieces are not marked so in the history and would be. The 16 primitives no committed abstract uses (~~18~~, corrected 2026-11-14: `normal` and `lognormal` size five
mechanisms' datasets and are settled) are provisional
too, and a candidate for removal is any of them a library function could emit (section 6's
composites).

**7.5 The version, and what a bump moves today.** *Found while drafting, 2026-11-14.* The
runner accepts one contract version exactly (`ast.rs`: "ast version `X` is not `Y`"), the
version is a key of the document, and the AST hash covers it (`schema/check.py`'s `canonical`
removes only `provenance`). A case's workload id, what results are compared by, covers the AST
hash (`runner/REFERENCE.md` §17). So every addition to the contract, though it moves no
fingerprint, changes every AST hash, every lock, and every workload id: eight versions in three
days (0.15 to 0.22) each regenerated 14 ASTs and relocked 25 cases, and the shapes take about
36 minutes to re-measure. Before anything is published this costs time only. After a working
group publishes reference hashes, an addition made for one new workload would split the
results of every other workload from those before it.

- *Proposal (a):* the runner accepts every contract version from a stated floor up to its own,
  and an abstract carries the lowest version that holds everything it uses. An addition then
  touches no committed AST, lock, or workload id; eBPF's rule that a program loads on every
  later kernel. It rests on 7.3: a document valid at version N is valid at N+1 and means the
  same. The builder would emit the lowest sufficient version, which needs each primitive's
  version recorded (the schema's descriptions carry it in prose today).
  *Built 2026-11-22 (brief item 80):* the floor is 0.23, the schema's `ast` lists the versions read, the table `SINCE` in `schema/check.py` and `validate.rs` records what each version past the floor added as a test of a document, both checkers refuse a document that does not carry the lowest version that holds what it uses, and the builder stamps it. The floor is 0.23 and not lower because 0.23 removed two nodes: a 0.22 document may hold one.
- *Proposal (b):* a change that 7.3 does not hold for (a narrowing such as 0.22, a reshaping
  such as 0.8's one dataset shape) raises the floor, or is confined to provisional pieces whose
  abstracts are regenerated alone. Which of the two is a decision; the second keeps settled
  abstracts' hashes through every change. *2026-11-22:* built with the first as its default, the floor raised; ~~the second is not built and the decision is still the user's~~ decided by the user 2026-11-22 ("I'm ok with 'raise the floor' at this stage of the project"): the floor is raised; to be done better once the project has dozens of users or more, when regenerating every abstract is no longer cheap. The second is not built.
- *Not proposed:* taking the version out of the hash. The hash would then no longer say which
  rules the document was checked under.
- *The record, for scale:* of the eight versions 0.15 to 0.22, six only added (the history says
  of each that the document before it is one of the new version "with the version changed"),
  and two changed a provisional piece (0.18, 0.22). Earlier versions that changed existing keys
  or rules (0.5, 0.6, 0.8) were not reread for this draft.

**7.6 What the policy would have said of the last eight versions.** A check of the policy
against the record, not a criticism of it: the pieces were built in the order one abstract
needed them.

| Version | What it added | Rung | Under this policy |
|---|---|---|---|
| 0.15 | `timeline`, `publish`, `published`, `entry`, `prev`, `array` | 4 | A map (the published log) and the planned order; `prev` and `array` are core (rung 5) |
| 0.16 | `seq` values, a tier with `lookup` | 4 | A map; `lookup` is a composite (section 6) and would have been `probe`, `touch`, `store` first |
| 0.17 | Groups of entries | 3 | A key on an existing map |
| 0.18 | Tier `order`, holds, `full`; resources; `now` | 3 and 4 | The tier's policy moved from an enum to expressions: a provisional piece reshaped (7.4) and made general, the pattern to repeat |
| 0.19 | Timed eviction, `on_evict`, `probe`, `touch`, `store` | 3 | Helpers; the eviction body is a new context (section 5) |
| 0.20 | `workers`, `submit`, `span` | 4 and 3 | A map (staging) and two helpers; a new context |
| 0.21 | A span's `staging` | 3 | A key on an existing helper |
| 0.22 | A refusal | none | A provisional piece narrowed (7.4) |
| 0.24 | A `span` outside a timeline | 3 | A helper allowed in a second context (section 5); the first version under 7.5's range, so no committed AST, lock, or workload id moved |
| 0.23 | Key ranges on `probe`, `touch`, `store`; `store`'s `present: touch`, `hold`, `found`; `lookup` and `choose` removed | none | A provisional piece reshaped (7.4) and two composites lowered to the library (7.1); added 2026-11-22, after the two observations below were written |

Two things the table shows. Contexts (a body with its own allowed effects) were added three
times and are as much a part of the growth as maps and helpers, so 7.2's first item matters
most. And 0.18's move, a fixed list of policies replaced by expressions over declared
attributes, is the one that removed a future runner change for every new eviction policy; the
six suggestions of the review are mostly the same move applied elsewhere (keys for chunks,
values on entries, a quantile table for four distributions).

**7.7 Decided (the user, 2026-11-14; `DESIGN_REVIEW.md` §3.100, the note "The growth policy
decided").**

1. The ladder and 7.2's five declarations are the rule for additions to the contract.
2. "Provisional until a second mechanism uses it" is the rule, and it permits narrowing,
   reshaping, and removal. The timeline family is declared provisional in `schema/README.md`
   §9.
3. Proposal 7.5 (a) is accepted: the runner is to accept every contract version from a floor
   up to its own, an abstract carrying the lowest version that holds what it uses. ~~**Not built
   yet** (brief item 80); until it is, the runner accepts its own version alone and every
   version regenerates every AST.~~ Built 2026-11-22 (7.5). The form of (b) is settled when it is built: built with the floor raised, which the user decided the same day for this stage of the project.
4. Removing the unused composites (`choose`, `mixture`, `hotset`) waits for the decision on
   review suggestion 5, the quantile table, which would replace two of them. *Settled
   2026-11-15:* suggestion 5 is library, `quantile()`, and no distribution was added; whether
   a library function can emit each of the three is checked before any is removed. *Checked
   and decided 2026-11-22 (the user's "proceed as recommended"):* `choose` can, and the builder
   emits the lowering now; its node leaves the contract with the next contract version, not
   in a version of its own, since every version regenerates every AST until item 80 is built
   (it left in 0.23, the same day, with suggestion 1).
   `mixture` cannot where a dataset's `size` stands, and `hotset` cannot at all; both stay
   (section 6's table, corrected).

### 8. The definitions fingerprints depend on (exists; not yet written)

Points to `runner/REFERENCE.md` §2.

### 9. The conformance kit (proposal; not yet written)

A published set of small ASTs with their fingerprints and the shared refusal cases, named as the
definition of a correct runner. Largely the golden tests and the two checkers' shared cases today.

## Part II: The microarchitecture

### 10. The planner (exists; not yet written)

The walk and its bounded horizon. Points to `runner/REFERENCE.md` §2 and §4.

### 11. The planned-op stream (exists; one decision; not yet written)

The trace file (`runner/REFERENCE.md` §13). *Decision needed:* how stable this lower interface is.

### 12. The executor (exists; not yet written)

The issue rule (`max(plan + L, previous completion, edges)`), lanes, lag, spans.

### 13. Run-side synchronisation (exists; one proposal; not yet written)

Channels, barriers, staging, validity edges.

- *Proposal:* events (brief item 72 (6)) as the one base, the others lowered onto it inside the
  runner (review suggestion 3).

### 14. Backends and the coordinator (exists; not yet written)

Points to `runner/REFERENCE.md` §6, §8, §9, §16.

### 15. The wall (new)

One rule separates the two halves of the runner: **timing state never reaches architectural
state.** It is the rule "never learn from the run" (decided 2026-11-10, `DESIGN_REVIEW.md` §3.99)
stated for this document. A processor's pipeline may reorder and stall, and the program's result
is the same; two systems under test may be fast and slow, and the op set they receive is the
same.

**What crosses from plan to run.** Each op with its lane; in a timeline its planned time, the
marks of its spans, and the edges it must wait on (an earlier op on the same path, a worker
lane's turn); the sizes the run is held to (a `workers`' staging).

**What crosses from run to plan.** Nothing.

**What the run alone decides** is when an op issues, never whether or what:

- Outside a timeline, an actor issues its next op when the previous one completes; channels and
  barriers hold it for another actor. The order of ops across actors follows the run, which is
  why issue order is never hashed.
- In a timeline, a lane's op issues at the latest of its planned time plus the lane's lag, the
  completion of the lane's previous op, and its edges. Lag never shrinks.

**What the run does with what it observes.** Each observation has exactly one of four fates, and
none of them is to adapt.

| Fate | Meaning | Examples |
|---|---|---|
| A reported number | The measurement the run exists for | Throughput, a span's D, each lane's lag, edge waits, a workers' backlog peak, an eviction batch's overage |
| An invalid run (exit 1) | The run did not impose the workload, so its numbers mean nothing | A would-be drop (staging past its size); the benchmark host's own lateness past a p99 of 10 ms |
| A valid run that did not pass (exit 3) | The workload was imposed and the system missed a case's constraint | A `result.` constraint not met; the lag ceiling crossed |
| Never modelled | What a real application would have done differently | A router balancing on real queue depth, a store dropped because staging filled, a client timeout and retry |

**The one exit.** A run may stop early at a certain failure (`--on-certain-failure stop`, the
default): it leaves the planner, issues nothing more, and lets what is in flight complete. A
stopped run has issued fewer ops than its abstract's set, so its fingerprint is not judged. This
is the only way the run changes which ops issue, and it does so only by ending.

**Why the wall is absolute.** If an observation changed the op set, a slower system would
receive a different, usually easier, workload; the fingerprint would no longer name the
workload; and no dry run could say what a run will do. The cost is accepted: real adaptive
behaviour is outside the model, and a system that cannot keep up with the plan gets an invalid
or failing run rather than a gentler one (revisited and kept by the user on 2026-11-14 for the
would-be drop).

## Part III: The library

### 16. Lowering (new, short; not yet written)

What a library function may and may not do.

### 17. The idiom catalogue (proposal, to be costed; not yet written)

Each library function carries its lowering and its trace signature, so the digest names the idiom
and the builder emits it. First entries: Python's `open()` around one read or write (in eight
abstracts today), the format classes' readers. Queued in brief item 79; not designed here.

## Part IV: The fit specification (a trace to an abstract)

### 18. Why this is decompilation (new; not yet written)

One trace, many abstracts; judgment and outside sources are needed; an agent does the work. So
the specification is of the acceptance test, not of the agent's process.

### 19. The artefacts in order (exists, scattered; not yet written)

Trace, digest, interview answers, authoring script, parameters, case, capture. Points to
`builder/REFERENCE.md` §7 and §8.

### 20. The acceptance test (new)

A compiler is trusted because it was verified once. The route from a trace to an abstract cannot
be: it takes judgment, it reads sources outside the trace, and an agent walks it. So each output
is checked instead, and what must be specified is the check. The route may be anything; the
check is deterministic and runs without the agent.

**What exists today** is two checks of different kinds, and an abstract may need either or both.

*The trace check: are these the application's calls?* The abstract's locality metrics at the
fitted parameters (`aeiou dry-run --metrics-json`) are compared with the trace's
(`aeiou-trace compare --judge`, `builder/REFERENCE.md` §7):

- Each row (the op mix, request sizes, run lengths, reuse and its distance, popularity, fan-out
  and depth) gets a verdict. Its allowed distance is its class's tolerance plus the abstract's
  own distance from itself at other seeds (`--self`): an abstract is not asked to be nearer the
  trace than it is to itself.
- The pair is accepted only when at least one row was judged and none is outside.
- A capture's tolerance file may do three things, each with a written reason: replace a class's
  tolerance; list a row as `unseen`, one the trace cannot show, which is then not judged; record
  a row as `outside`, which is printed with its reason and is still outside.
- The verdict is recorded with the capture (its record `capture.toml` and its tolerance file,
  `builder/REFERENCE.md` §7 and §8).

*The policy check: does the plan decide as the application decides?* A trace shows what one run
did, not the rule that produced it, so a stateful abstract (a cache, a queue) is held to the
application's own policy code instead: the abstract's walk and the reference are driven by the
same planned events and compared decision by decision. `lmcache_local_disk_tiered` was accepted
this way, equal request by request and unlink by unlink to LMCache 0.5.5's own
`LRUEvictionPolicy` and to the kit's simulation of vLLM's pool (`DESIGN_REVIEW.md` §3.99).

**What a tolerance may not excuse.** A row outside stays outside whatever the reason; an
`unseen` row is one the trace could not show, never one the abstract gets wrong; and nothing
excuses a call the application makes and the abstract leaves out (a truncate is never dropped).

**What is not specified yet** (each an open decision, listed in the appendix):

- The structural comparison of two digests (`compare` on two `aeiou_digest: 2` documents, the run
  side's from `dry-run --emit-trace`) exists and is not part of acceptance. Whether it should be
  is undecided.
- The policy check has no tool and no recorded form: each has been a script in a kit or the
  scratch. A named, repeatable form for it is the largest gap in this part.
- Nothing judges time: arrival, think time, and service time are fitted and never compared.
- What an agent must hand over for a fit to be reviewable (section 21) is practice, not a
  checked format.

### 21. Provenance (exists as practice; not yet written)

Every claim tagged with its source and confidence; primary sources before backfitting.

### 22. What the agent may not do (exists as rules; not yet written)

No emulator as a source, no trace listing as an abstract, no dropped truncates, nothing learned
from a run.

## Appendix (not yet written)

- Glossary: our term beside the familiar one (the describer and a disassembler, `dry-run` and
  `EXPLAIN`, the validator and a bytecode verifier).
- Open decisions, listed once. So far: the stability of the planned-op stream (11); whether the
  checkers' code follows section 5's table; whether `probe` stays refused in a `submit` and an
  `on_evict` body, and what the timeline's expressions mean in an `on_evict` body (5); whether the structural comparison is part of
  acceptance (20); a form for the policy check (20); whether time is judged (20); the six review
  suggestions below.

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
| Size distributions of dataset definitions: `lognormal`, `normal` | 2 | 5 and 1 |
| Unused by any committed abstract: `choose`, `put`, `trace`, `fallocate`, `fdatasync`, `rmdir`, `truncate`, `ftruncate`, `hotset`, `mixture`, `column`, `sum`, `unit_index`, `chunks`, `count`, `now` | 16 | 0 |

*Corrected 2026-11-14:* the table as first written listed `normal` and `lognormal` among 18
unused primitives. The count had missed dataset definitions: five committed ASTs draw a
dataset's sizes from `lognormal` (`file_per_sample`, `tfrecord_stream`, `parquet_stream`, and
both `lmcache_shared_fs` documents) and one from `normal` (`npy_per_sample`). The two are
settled, not provisional, and those sizes are part of each dataset's id.

The finding: the DSL is RISC at its edges (ops, expressions) and CISC in the middle (the
concurrency constructs and the cache verbs). The session generator, the most complex thing
written, needed no generator-specific runner primitive.

Six suggestions (decided 2026-11-14, after the list; ~~none built~~ 5 built 2026-11-15 as a library function, 1 built 2026-11-22 as contract 0.23, 4 the same day as contract 0.24):

1. **Cache verbs take keys, not chunks of a sequence** (instruction set). `lookup`, `probe`,
   `touch`, `store` take `(seq, per, from, count)` today; brief item 75 said keys are opaque to
   the cache. A key range (an index, a count, a key expression), `chunk_key(seq, j, per)` one such
   expression. With `hold` and `full: wait` on `store`, `lookup` becomes a builder function. Gives
   the cache its second user (DiskANN's node cache, an epoch cache, a local disk cache over an
   object store). *Built 2026-11-22, contract 0.23:* `keys {index, from, count, key}` on `probe`,
   `touch`, and `store`; `store` with `present: touch`, `hold`, and `found` in place of the
   `lookup` node, the builder's `lookup()` over it; the range's cut at a sequence's end and the
   chunks of an array moved to the builder's `chunk_keys`. No second user is written yet.
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
4. **`span` everywhere, ~~merged with `phase`~~** (instruction set; *built 2026-11-22 as contract 0.24, a separate statement as decided: outside a timeline a span is the elapsed time of its body, ops and a joined `parallel`, no compute and no wait on another actor; no committed abstract uses one yet*). A measured region is
   timeline-only; time to load a model, time to save a checkpoint, and query latency are those
   workloads' headline numbers.
5. **A quantile-table distribution** (instruction set). The tiered script builds inverse CDFs by
   hand in fixed point, two draws and about eight nodes a sample at about 40 sites. A
   piecewise-linear quantile table is what a fit produces and is integer-only; it would supplant
   ~~`normal`, `lognormal`,~~ `hotset`, `mixture`. ~~Untested risk:~~ `rng.rs` computes normals with `ln`,
   `sqrt`, `cos` in floating point, and releases are x86_64 and aarch64. *Corrected
   2026-11-14:* `lognormal` and `normal` size the datasets of six committed ASTs, so the table
   cannot supplant them without changing those datasets' ids (rung 6 of section 7), and the
   floating-point risk is not the table's but today's: those sizes go through libm (`ln`,
   `sqrt`, `cos`, `exp`), `DESIGN_REVIEW.md` records it as a caveat "rather than solved", and
   the aarch64 release leg smoke-tests only, so the golden hashes are not known to hold there.
6. **Ops as table rows** (the runner's internals). `fadvise` is named in 21 source files, about
   half capture-side; item 58 adds about 25 calls, item 61 object requests, item 76 possibly block
   commands. Not costed against the dispatch code.

**Decided (the user, 2026-11-14, "agreed on all"; `DESIGN_REVIEW.md` §3.100, the note "The six
suggestions decided"; none built).** In this order: the golden tests run on aarch64, to learn
whether the drawn sizes differ today (run 2026-11-14 by the release workflow, all 36 passing
on both architectures), then ~~the quantile table added as a new distribution
(rung 5, no fingerprint moved), committed datasets leaving `lognormal` only if that run shows
a difference, and~~ the quantile table as a library function, `quantile()` in the builder,
and no distribution added to the contract (the user, 2026-11-15: the tiered script's `inv()`
already was one over `empirical`, `uniform`, and an array parameter, so 7.1 makes it rung 2;
moved to the library with the tiered abstract's AST hash unchanged; a contract distribution
waits for a dataset size that needs a fitted table), `normal` and `lognormal` kept; 1, the cache verbs over keys, a provisional
piece reshaped (7.4), `lookup` becoming a builder function, ~~accepted when the tiered
abstract's fingerprint is unchanged~~ built 2026-11-22 and accepted by the user on other evidence, the planned trace of the tiered case equal line for line between the old node on the old runner and the new store on the new one, the fingerprint itself having moved for a `let` the new form needs (a statement added moves every later draw site of its body); 4 in part, `span` outside a timeline, `phase` left as it
is and not merged; 2 in part, an eviction body acting on a strictly lower tier (so a cascade
ends), built with brief item 78 after 1, and entries carrying values and a trigger on tier
state not built for want of a committed user. Deferred: 3, recorded as a constraint on brief
item 72 (6) (events designed as the base, nothing existing lowered onto them until then, since
`consume`'s position formula is defined through `loader`), and 6, costed when item 58 starts.

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
