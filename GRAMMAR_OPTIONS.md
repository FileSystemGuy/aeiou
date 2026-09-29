# Abstract Grammar — Options for the Missing Constructs

Status: design options (2026-09-24). Companion to `NAPKIN_MATH.md` §4.4.

## 1. The problem

The original sketch has four constructs:

- **operation**: one syscall, e.g. `open`, `read`, `write`, `lseek`, `close`, `sleep`
- **sequence**: an ordered list of elements with a repeat-count distribution, e.g.
  `(open, (read)[1..10], close)[1..8192]`
- **selection**: pick one of several sequences by probability, e.g. `read[80%] | write[20%]`
- **consumer**: a selection that draws without replacement until exhausted

These describe one sequential stream of operations. The PyTorch training workload also needs:

| # | Construct | Needed for |
|---|---|---|
| 1 | **Binding**: `let f = consume(train)`, then `open(f)`, `read(f)` | The file a consumer picks must flow into the ops that follow. |
| 2 | **Replication** (`per gpu`, workers) | Scaling by `--gpus`. |
| 3 | **Fork/join** | W workers reading at the same time; a batch is done when all of its files are read. |
| 4 | **Producer/consumer with bounded prefetch and in-order delivery** | DataLoader keeps `workers × prefetch` batches in flight while the GPU computes, and hands them over in order. |
| 5 | **Barrier(scope)** | Periodic all-GPU sync (all-reduce); checkpoints. |
| 6 | **Named parameters with CLI override** | `batch`, `workers`, `xfer`, `compute`, `steps`, … set per run. |
| 7 | **Data-dependent repeat** (`read(f, xfer)[until_eof]`, `[ceil(size(f)/xfer)]`) | The read count depends on the file's size. The file sizes come from the dataset definition, so the count is still known ahead of time. |
| 8 | **Explicit loop indices and conditionals on them** (`for step in $steps`, `every 500`, `if step % n == 0`) | Periodic barriers and checkpoints; file names such as `ckpt/{step:06}`; the RNG key (see §2). |

The semantic model below is the same for all three syntax options. Only the surface
syntax differs.

## 2. Semantic model (common to all options)

- A **workload** declares `params`, `datasets`, and one or more **actor templates** (e.g. `gpu`).
- `--gpus G` instantiates G actors of the `gpu` template, with **global** ids `0..G`. Hosts
  (ranks) own contiguous id ranges. The ids do not depend on the number of ranks, so the same run can be
  spread over 10 or 20 nodes and produce the same workload.
- An actor runs a **program** (a tree of elements). An element can suspend on:
  an I/O completion, a timer, a join, a channel `take`, or a barrier.
- **Every repeat binds an explicit index variable** (`for step in $steps { … }`,
  `for j in $batch { … }`). A bare `( … )[n]` is sugar for a loop with an anonymous index. The
  vector of enclosing loop indices is the actor's *position* in the program's iteration space.
  Everything random or name-like (draws, file names such as `ckpt/{step:06}`, `every`) is a
  function of that position, never of a counter that the actor increments.
- **Randomness** at any point is `hash(seed, actor_id, site_id, [enclosing loop indices])`.
  `site_id` is a stable id for the place in the abstract where the draw happens. There is no
  `draw#` counter and nothing to reset at epoch boundaries. Consequence: "what does GPU 17 issue
  at step 300" is computable without simulating steps 0–299, which makes `--dry-run` fast and
  parallel and lets a single actor be debugged in isolation.
- **Primitives for concurrency are `parallel(W) { … }` and a bounded
  `channel(capacity, ordered | unordered)`** with `put` and `take`. `take` on an ordered channel
  waits for the *next item in sequence number order* (head-of-line blocking); on an unordered
  channel it takes whatever is ready. The VM needs channels anyway, so making them primitives
  costs nothing and keeps tf.data-style unordered interleave, DALI, or a pool of checkpoint
  writers expressible without new VM instructions.
- **`loader`** is syntactic sugar over those primitives: W worker sub-actors plus an ordered
  channel of `W × prefetch` slots. Worker w builds batches `w, w+W, w+2W, …`, matching PyTorch's
  round-robin worker assignment. **A loader is finite:** it is declared with its total batch count
  (`batches = $steps`), like a sampler of finite length, and dispatches exactly that many. Workers
  never prefetch past the last step, so the op multiset does not depend on when the run stops and
  nothing has to be cancelled at the end.
- **`consume(ds)`** is `perm_{dataset,seed,epoch}(position)`. The position is a **formula, not a
  counter**: GPU g, batch b, item j uses position `g + G·(b·B + j)`, and batch b is built by
  worker `b mod W`. This is `DistributedSampler` (rank takes positions `rank, rank+R, …`) followed
  by the round-robin `BatchSampler` and the DataLoader's round-robin worker assignment. No
  per-actor or per-GPU counter exists, so the file a worker draws cannot depend on which worker
  finished first.
- **Epoch wrap uses `drop_last` semantics.** An epoch is `floor(N / (G·B))` steps long; every GPU
  switches to the next epoch's permutation key at the same step. The `N mod (G·B)` leftover
  positions of each epoch are never drawn. Epoch = `step div epoch_len`, position within the
  epoch uses `b = step mod epoch_len`.
- **Datasets carry their own seed.** File sizes are `size_dist.sample(hash(dataset_seed,
  file_id))`, independent of `--seed`, so changing the run seed never changes what the dataset is
  expected to look like. `datagen` writes a manifest at the corpus root (pattern, count, size
  distribution, dataset seed, generator version); the runner validates the abstract's dataset
  declaration against it before starting.
- **The implementation compiles the tree to bytecode** that runs on a per-actor VM, in the style
  of a regex VM: a stack of `(pc, loop index, loop bound)` frames plus a few registers for
  bindings. Because there is no hidden state, the VM is a pure step function
  `next_op(state) -> (op, state')`, which is what `--dry-run` iterates. `parallel`/`loader`
  spawn child VMs, and `join`/`take` are wait instructions.

## 3. Syntax options

### Option A — Extended regex-style DSL (recommended until 2026-09-28; see Option D and §4)

This keeps your notation as the core and adds a few keywords. Existing sequence expressions stay valid.

```
workload unet3d_train {
  param batch      = 7
  param workers    = 4
  param prefetch   = 2
  param steps      = 500
  param sync_every = 500
  param xfer       = 1MiB
  param compute    = 323ms            # or a distribution: normal(323ms, 10ms)

  dataset train = files("train/{id/10000:05}/sample_{id:09}.npz",
                        count = 50_000_000,
                        size  = const(140MiB),
                        seed  = 0x5eed_da7a)      # dataset seed, independent of --seed

  per gpu {
    # finite: exactly $steps batches, so workers never prefetch past the last step
    loader batches(workers = $workers, prefetch = $prefetch, order = in_order,
                   batches = $steps) {
      # one worker builds one batch; the loader binds `b` (batch index), `j` is the item
      for j in $batch {
        let f = consume(train)          # position = gpu + G·(b·$batch + j)
        open(f, RDONLY), fstat(f),
        read(f, $xfer)[until_eof],
        close(f)
      }
    }

    for step in $steps {
      take(batches),
      compute($compute),
      every $sync_every { barrier(global) }   # tests `step`
    }
  }
}
```

The same loader written out in the primitives it desugars to, to show there is nothing hidden:

```
    channel batches(capacity = $workers * $prefetch, ordered)
    parallel w in $workers {
      for b in $steps step $workers from w {      # b = w, w+W, w+2W, …
        for j in $batch { … }
        put(batches, seq = b)
      }
    }
```

Checkpoint fragment, as an example of what writes look like:

```
    every 100 {                        # inside `for step in $steps`, so `step` is bound
      barrier(global),
      ( let c = file("ckpt/{step:06}/rank_{gpu:05}.pt"),
        open(c, WRONLY|CREAT|TRUNC),
        write(c, 64MiB)[ckpt_bytes / 64MiB],
        fsync(c), close(c) ),
      barrier(global)
    }
```

Selection and consumer, as in your sketch:

```
    choose { 80%: (read(f, 4KiB))[1..10], 20%: write(f, 4KiB) }
    let g = consume(train)         # without replacement
    let h = pick(train)            # with replacement
```

- **Pros:** compact; reads like your mental model; one abstract fits on a page. A 50M-file
  workload is still ~30 lines.
- **Cons:** we write a parser (≈500–800 lines with `pest` or `winnow`/`chumsky`) and good
  error messages.

### Option B — Structured data (YAML/TOML/JSON → serde)

The same tree, written as data:

```yaml
params: { batch: 7, workers: 4, prefetch: 2, steps: 500, xfer: 1MiB, compute: 323ms }
datasets:
  train: { pattern: "train/{id/10000:05}/sample_{id:09}.npz", count: 50000000, size: 140MiB }
actors:
  gpu:
    - loader:
        name: batches
        workers: $workers
        prefetch: $prefetch
        batches: $steps
        body:
          - for: { j: $batch }
            do:
              - let: { f: { consume: train } }
              - open: { file: f, flags: RDONLY }
              - fstat: f
              - repeat: until_eof
                do: [ { read: { file: f, size: $xfer } } ]
              - close: f
    - for: { step: $steps }
      do:
        - take: batches
        - compute: $compute
        - every: { n: 500, do: [ { barrier: global } ] }
```

- **Pros:** almost no parser work (serde derives it); schema validation; easy to generate from
  tools, for example a converter from `strace` traces.
- **Cons:** deeply nested sequences are hard to read and edit by hand; roughly 3× longer.

### Option C — Embedded scripting language (Rhai or Lua via `mlua`), actors as coroutines

```lua
function gpu(ctx)
  local batches = ctx:loader{workers=4, prefetch=2, body=function(w)
    for i = 1, P.batch do
      local f = w:consume("train")
      w:open(f); w:fstat(f)
      repeat local n = w:read(f, P.xfer) until n == 0
      w:close(f)
    end
  end}
  for step = 1, P.steps do
    ctx:take(batches); ctx:compute(P.compute)
    if step % 500 == 0 then ctx:barrier("global") end
  end
end
```

- **Pros:** unlimited expressiveness; no grammar design.
- **Cons:** per-op interpreter cost of ~0.5–1 µs and a coroutine per actor. It also makes it easy
  to write non-deterministic abstracts (anything that reads the clock or branches on completion
  order). The workload is no longer data, so it's hard to validate, diff, or reason about. It
  undercuts the "abstract" idea.

### Option D — Full Python as the authoring language, building Option B's AST (added 2026-09-28)

**Three layers, and what this option changes.** The language does three jobs: (1) *authoring*,
how a human or a converter writes a workload; (2) *the contract*, the artifact that is
fingerprinted, archived with results, and validated for CLOSED; (3) *execution*, what runs inside
each actor state machine. Option D changes layer 1 only. Layer 2 stays the serde AST of Option B
and layer 3 stays the Rust VM. Every guarantee in §2 and §4 lives in layers 2 and 3, so Python at
layer 1 cannot weaken execution determinism or performance.

**Roles.** The MLPerf Storage WG authors workloads in Python and publishes the resulting AST and
its hash. Submitters run layers 2 and 3 only, with the published AST. Other users of the tool
author their own ASTs the same way; the runner treats every AST identically and none is
"official" except by the WG's published hash.

**The AST is the artifact.** The only thing a nondeterministic script can do is produce a
*different but valid* AST on a second run. That is a provenance problem (can this AST be
regenerated from this source?), not a benchmark problem, because submitters run the hash. The
techniques below make source→AST reproducible and auditable; they are the standard
reproducible-builds toolkit.

#### Builder sketch

```python
from mlps_abstract import Workload, const, normal, MiB, ms

w = Workload("unet3d_train")

# Parameters are symbolic references, not ints. P.steps is a ParamRef node; arithmetic on it
# produces an Expr node. This is what stops Python from expanding the workload.
P = w.params(batch=7, workers=4, prefetch=2, steps=500, sync_every=500,
             xfer=1 * MiB, compute=323 * ms)        # or compute=normal(323 * ms, 10 * ms)

train = w.dataset("train",
                  pattern="train/{id/10000:05}/sample_{id:09}.npz",
                  count=50_000_000,
                  size=const(140 * MiB),
                  seed=0x5eed_da7a)                  # dataset seed, independent of --seed

with w.actor("gpu") as gpu:
    # finite: exactly P.steps batches; the loader binds `b` (batch index)
    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch,
                    batches=P.steps, ordered=True) as worker:
        with worker.loop("j", P.batch):
            f = worker.consume(train)               # position = gpu + G·(b·batch + j)
            worker.open(f, "RDONLY")
            worker.fstat(f)
            worker.read(f, P.xfer, repeat="until_eof")
            worker.close(f)

    with gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.compute)
        with gpu.every(P.sync_every):               # tests `step`
            gpu.barrier("global")

w.write("unet3d_train.ast.yaml")   # canonical form + provenance block; prints the AST hash
```

Checkpoint fragment, showing symbolic arithmetic and loop-index references:

```python
        with gpu.every(100):
            gpu.barrier("global")
            c = gpu.file("ckpt/{step:06}/rank_{gpu:05}.pt")   # `step` is bound by the loop
            gpu.open(c, "WRONLY|CREAT|TRUNC")
            gpu.write(c, 64 * MiB, repeat=P.ckpt_bytes // (64 * MiB))   # Expr node
            gpu.fsync(c)
            gpu.close(c)
            gpu.barrier("global")
```

The loader desugared to the primitives, as in Option A:

```python
    ch = gpu.channel("batches", capacity=P.workers * P.prefetch, ordered=True)
    with gpu.parallel("w", P.workers) as worker:
        with worker.loop("b", P.steps, step=P.workers, start=worker.index):   # b = w, w+W, …
            with worker.loop("j", P.batch):
                ...
            worker.put(ch, seq="b")
```

What the builder refuses or the lint flags:

```python
for j in range(7):                  # unrolled: 7 sibling subtrees instead of a loop node.
    worker.consume(train)           # Lint: "run of identical siblings; use worker.loop".

n = random.randint(1, 10)           # Hermetic mode: `random` unseeded → raises.
worker.read(f, n * MiB)             # (Distributions belong in the AST: worker.read(f, uniform(1*MiB, 10*MiB)).)

worker.read(f, callback=lambda r: …)   # TypeError at construction: nodes hold no callables.
```

#### How the AST is built (staging), and the three routes to an AST

**The Python never runs the workload.** It is the same kind of thing the DSL of Option A was: a
description of what the execution layer should do. The DSL was *parsed* into the tree; the Python
*constructs* the tree by running once, and the calls that look like I/O are declarations. This
technique is called staging and is how JAX tracing, PyTorch FX, TensorFlow graph mode, and Halide
work.

Mechanically:
- `worker.read(f, P.xfer)` reads nothing; it appends a `Read` node at the builder's current
  cursor.
- `with gpu.loop("step", P.steps) as step:` pushes a `Loop` node, makes it the cursor, and hands
  back `step` as a symbolic `LoopIndex`. Leaving the block pops the cursor.
- Parameters are `ParamRef` objects. Symbolic values overload operators, so
  `step % P.sync_every == 0` does not evaluate to a boolean; it returns an `Expr` node holding
  the formula, and `gpu.when(...)` / `gpu.every(...)` wrap it in a `Cond` node.
- `w.write()` walks the finished tree, validates it against the schema, and emits canonical
  YAML. Nothing is extracted from Python source; Python's own syntax tree is never parsed.
  About 500 lines of Python with pydantic.

The unet3d step loop becomes:

```yaml
- loop: { index: step, count: { param: steps } }
  body:
    - take: batches
    - compute: { param: compute }
    - cond:
        expr: { eq: [ { mod: [ { index: step }, { param: sync_every } ] }, 0 ] }
        then: [ { barrier: global } ]
```

**The one discipline.** Python control flow runs at build time; builder control flow runs at
execution time. A Python `for` over a symbolic `P.steps` cannot iterate at all, which is the
same error JAX raises for `if` on a traced value, and it tells the author to use `gpu.loop`.
Python loops remain useful for generating *structure*: four similar actor templates, a `choose`
whose weights come from a table, the seven fixed syscalls of a Python `open()`. Rule of thumb:
**Python loops generate structure; builder loops generate iterations.** The identical-siblings
lint catches hand-unrolled iterations.

**Authoring is abstraction, not emulation.** The author reproduces the real application's I/O
skeleton: the order and nesting of I/O calls, which run in parallel, what waits on what, what
repeats and how often. The application's logic, data structures, and data-dependent branching
are either gone or reduced to a `compute` node or a distribution slot. The skeleton comes from
knowing how the real system is built (the DataLoader shape came from PyTorch's design, not from
a trace), and the same is true for a VDB or a KV cache. The author also decides the *cuts*: the
behaviors that cannot be carried into the shape without breaking determinism (cross-actor
reuse, cache eviction, §5.3) and are replaced by statistical inputs. Cuts are listed in the
workload's documentation.

**Three sources fill the AST's slots**, and it helps to keep shape and fitted parameters as
separate artifacts (the AST with named distribution slots, plus a parameter file that fills
them; the WG can publish one shape with several parameter sets):
1. *Configuration* of the real system: batch size, workers, block size, beam width, nprobe.
2. *Measurement* of the real system's compute: the `compute` distributions.
3. *An I/O trace* of the real system, for what only the data determines: sizes, offsets, reuse
   distances, hop counts, popularity skew, hit lengths. The trace never enters the AST; it
   informs the parameter file, validates the result (§5.4), and is then set aside.

**Three routes to an AST**, answering different questions:
1. **Builder** (this option). A human writes the shape and names the slots. Exact by
   construction.
2. **Trace and fit.** Capture a trace of the real application (`strace`, eBPF, an `LD_PRELOAD`
   shim) and fit the distributions in the slots. A trace is one *linearization* of the workload:
   it yields distributions but not the dependency graph, because the graph is exactly what the
   linearization threw away. Fitting therefore fills in the numbers of a shape a human already
   wrote; it does not discover the shape. Automated shape inference from traces is a research
   problem and is not promised. The trace also *tests* the shape: a locality metric (§5.4) that
   no choice of parameters can match is the signal that the shape is missing a construct, for
   example a reuse-distance curve with no `recent` reference to produce it. The loop is: write
   the shape, capture, fit, compare, revise the shape, repeat.
3. **Replay.** The trace itself as a literal `replay` node, bounded to small-scale calibration
   (§5.4). Never a CLOSED workload.

Running a *Python model* that performs fake I/O and recording it would be route 2 applied to a
model instead of the real application. It needs a Python execution engine with a simulated
scheduler to produce an interleaving (Option C's problems, in Python) and it still loses the
graph. Route 1 already has the graph and route 2 already has real numbers, so the model adds
nothing.

**Variant D2, for later if wanted.** A decorator could translate a restricted Python subset by
reading its syntax tree, so `for step in range(P.steps):` becomes a loop node and
`if step % 500 == 0:` a cond (the Numba/Triton/Taichi approach). It reads more naturally but
needs a translator, a precise definition of the allowed subset, and harder error messages.
Start with the builder; add the decorator only if authors find the `with` blocks tedious.

#### Techniques that make source→AST reproducible, in leverage order

1. **Serialization is the firewall.** The builder emits YAML/JSON and the runner reads only that.
   A lambda, a callback, an open file, or an array of already-drawn values cannot be serialized
   and therefore cannot reach layer 2. Structural, not a lint.
2. **Runner-side validation is the enforcement point.** On load the runner checks the AST against
   a published JSON Schema plus the semantic rules of §4 (finite loops, conditionals only on loop
   indices and parameters, no wall-clock phases in CLOSED, dataset matches manifest, op count
   computable). A misbehaving generator produces a rejected AST, never a drifting execution.
   Everything below this line is hygiene on top of the guarantee.
3. **The AST stays symbolic.** Python computes parameters; it never expands the workload.
   Distributions, consumers, and random offsets are nodes evaluated by the positional RNG at run
   time, so a script has no reason to draw random numbers. Loops stay loop nodes. An AST size
   limit and a lint for runs of identical siblings catch accidental unrolling. This also keeps the
   AST readable and diffable for WG review.
4. **Content-addressed AST with provenance.** The AST's identity is the hash of its canonical
   form: sorted keys, fixed float formatting, site ids derived from structural position rather
   than from Python object ids or creation order. A provenance block records the script hash, its
   git commit, the Python version, and the lockfile hash. The runner prints the AST hash with
   every result.

   ```yaml
   provenance:
     generator: { script: unet3d_train.py, sha256: 3f9c…, git: a1b2c3d }
     python: "3.12.6"
     lock: { file: uv.lock, sha256: 77e1… }
     built_twice_identical: true
   ```
5. **Build twice, compare, in CI.** Generate the AST in two fresh processes with different
   `PYTHONHASHSEED` values, ideally on two machines, and fail if the hashes differ. This catches
   set-iteration order, hash-dependent ordering, timestamps, and unpinned dependency drift without
   anticipating them (the Debian/Nix reproducible-builds test).
6. **Hermetic generation environment.** `abstract-build --hermetic script.py` runs the script with
   a pinned interpreter and hash-locked dependencies (`uv run --locked`), in isolated mode
   (`python -I -E -s`) so user site-packages and environment variables cannot leak in, with no
   network. PEP 578 audit hooks deny socket connections, subprocesses, and file opens outside the
   declared inputs. Audit hooks do not see clock reads, so the harness also stubs `time.time`,
   `datetime.now`, `os.urandom`, `uuid.uuid4`, and unseeded `random`/`numpy.random` to raise.
   Item 5 is what proves this worked.
7. **Builder API hygiene.** Typed, immutable node constructors validated with pydantic against
   the same schema the runner uses, so most mistakes fail at construction with a Python
   traceback. Three documented don'ts: iterate sets, use `id()`/`hash()` for identity, read the
   environment.

**What this does not cover, accepted:** a script can be deliberately nondeterministic in a way
that survives item 5 by chance, or can embed a value from a declared input file. Neither affects
submitters, who run the published hash. For the WG, the published AST is the normative artifact;
the script and provenance are attached for audit.

- **Pros:** well-known syntax; the whole Python ecosystem for authoring (numpy for distributions,
  pandas for trace analysis, an `strace`→AST converter in the same package); no parser or grammar
  to design or document; PyTorch implementors can read it; Python lives only on the authoring
  station, never on the bare-Linux client nodes; layers 2 and 3 are untouched.
- **Cons:** a Python toolchain and lockfile discipline on the authoring side; reproducibility of
  *authoring* is CI-enforced rather than by construction (Starlark would give it by construction
  but loses numpy and familiarity); two representations to keep readable (Python source and
  YAML AST), though only the AST is normative.

**Relation to the other options.** D subsumes A (a Python builder is a better front end than a
bespoke parser, and A's syntax would become a third representation to maintain), keeps B as the
contract, and differs from C in kind: C puts an interpreter in layer 3; D puts one in layer 1.

## 4. Recommendation

**Revised 2026-09-28.** Option B's tree remains the canonical AST and the only thing the runner
executes. For the authoring language the leading candidate is now **Option D**: a Python builder
package that emits the AST, with the reproducibility techniques above enforced by CI and by the
runner's validator. Option A is dropped from the recommendation rather than kept as a third
representation. The choice is still the user's.

The earlier recommendation (Option A for the language people write, Option B as the canonical
serialization, `--dump-ast` to emit it) is retained here for the record; its reasoning about a
single semantic model and a machine-friendly format for generated abstracts applies unchanged
to D.

Guardrails to keep the language from growing into a general-purpose language:

- No unbounded loops. Every repeat count is a constant, a distribution, a parameter, or a
  function of dataset metadata (`size(f)`), and `until_eof` is resolved from the known file size.
  This means the total op count of a run can be computed before it starts.
- Conditionals are allowed only on loop indices, the epoch, and parameters, never on timing or
  completion order. This is what makes the workload fingerprint (see `NAPKIN_MATH.md` §8) exact.
- No actor-local mutable counters. Every draw, name, and condition is a function of the actor id
  and the enclosing loop indices (§2). This is what makes the op stream randomly accessible.
- Producers are finite. A `loader` or `parallel` block declares how many items it produces, so
  the run's op multiset is fixed before it starts.
- Wall-clock-bounded phases (`for 60s`) are allowed only as an opt-in that turns off the exact
  fingerprint guarantee.
- The abstract is POSIX-shaped and has no `mmap` op. `mmap`-based loaders are handled by the
  `mmap` backend, which maps `read(f, off, len)` onto populating that range of a mapping
  (`PROJECT_BRIEF.md` §4). The op stream and fingerprint are therefore the same for every backend.

## 5. Expressiveness beyond training: data-dependent workloads (added 2026-09-28)

Training and checkpointing are not the hard cases. Vector-database (VDB) and KV-cache workloads
are, because their access pattern is determined by the data. The question examined here is
whether "random" is an acceptable stand-in for data-dependent sequences. **Answer: only as a null
model.** Random is equivalent to a data-dependent sequence exactly along the dimensions a storage
system cannot see, and a storage system (server readahead, client cache, SSD firmware) can see
and exploit a short, well-known list of dimensions. The abstract must reproduce every one of
those that the real workload has, and none that it does not.

### 5.1 What storage can exploit

| Property | Real examples | What uniform random gets wrong |
|---|---|---|
| **Reuse** (temporal locality) | HNSW entry points and upper layers on every query; DiskANN medoid and first hops; KV-cache system prompts and conversations that return within minutes | none, so caches are penalized unfairly |
| **Sequential runs and strides** | posting lists, KV block chains, interleaved per-thread streams that firmware de-interleaves | no runs, so readahead/prefetch are denied a real benefit |
| **Size and popularity skew** | IVF cluster sizes vary by orders of magnitude; queries hit popular clusters because they come from the same distribution as the data | uniform sizes and ids |
| **Dependency shape** | beam search: a burst of parallel reads, then a dependent hop | independent draws at a fixed queue depth |
| **Write-then-read with a lag** | KV blocks written at prefill, read at a later prefix hit | no relation between writes and reads |

A synthetic stream that matches a real trace on these dimensions is indistinguishable to any
cache or prefetcher that works on them. Firmware that identified an application by a signature
outside them would be fooled by every synthetic benchmark (fio and DLIO included); the answer to
that is end-to-end validation against the real application (R10, and the replay mode in §5.4).

The asymmetry: the Feistel shuffle deliberately removes *accidental* structure so storage cannot
get an unfair benefit. Real structure must then be deliberately *added back*, or the benchmark
under-reports what the storage would do in production.

### 5.2 Additions to the semantic model

Three additions. None touches layers 2 or 3 beyond new node types; every draw remains a pure
function of `(seed, actor, site, loop indices)`.

1. **Distributions over ids and offsets.**
   - `pick(ds, dist = zipf(s))`, `pick(ds, dist = hotset(fraction, weight))`, and mixtures.
   - `read(f, offset = draw(dist), len)`: random offsets within a file. Today's model only has
     uniform consumers and sequential offsets; this is the main gap.
2. **Recency references** (the stack-distance model of temporal locality).
   `recent(site, d)` refers to the object drawn at `site` `d` iterations ago, with `d` drawn from
   a reuse-distance distribution. Because every draw is positional, the draw at step `s − d` is
   recomputable, so this needs **no stored history**. It is what makes KV-cache workloads
   expressible. *Refinement proposed 2026-09-29 (`ABSTRACTS.md` §9.5): the primitive should be
   `x @ i`, a binding evaluated at another index of its loop, with self-reference at a strictly
   smaller index allowed, so that a conversation's identity chains back to its origin; `recent`
   becomes sugar. Not yet decided.*
3. **Positional names for write-then-read.** A writer names its object from its position
   (`file("kv/{prefix_id:016x}/blk_{k:04}")` with `prefix_id` drawn positionally); a later reader
   recomputes the same name. Within one actor this is exact.

Already expressible with the primitives of §2: dependent chains with fan-out
(`for hop in H { parallel(beam) { … } }`), hop counts and burst widths from distributions,
periodic compaction/rebuild phases, LSM-style `choose { hit%: …, miss%: … }` lookups.

### 5.3 The limit: cross-actor read-after-write

A KV block written by one GPU's prefill and read by another GPU's decode is a hit only because it
exists, and existence is timing. The model cannot make one actor wait on another actor's
unrelated write without becoming timing-dependent, which would break the fingerprint. Therefore:

- cross-actor reuse lives in **phases separated by barriers** (write phase, `barrier`, read
  phase), or
- it is **modeled statistically**: a hit ratio and a lag distribution are inputs, and misses are
  emitted as reads that fail-soft (`ENOENT` counted, not fatal, in a declared "miss-tolerant" op).

Cache capacity and eviction are treated the same way: an input property of the workload, not
something that emerges from a simulated cache. For a storage benchmark this is the right cut,
but it is a stated fidelity loss and belongs in the workload's documentation.

### 5.4 Method: locality-metrics check and replay mode

- **Locality-metrics check.** From a real trace, compute: reuse-distance distribution,
  sequential run-length distribution, popularity skew (rank–frequency), request-size
  distribution, dependency depth and fan-out, read/write mix. From `--dry-run`, compute the same
  metrics on the abstract's op stream. An abstract is accepted for a workload class only when the
  two match within stated tolerances. This sits next to the fingerprint: the fingerprint proves
  *which* stream ran, the metrics prove it is the *right* stream.
- **Replay mode.** The AST may be a literal captured sequence (a `replay` node holding ops with
  their dependencies). It is bounded to small-scale calibration runs and exists so each workload
  class's abstract can be validated end to end against the real application on the same storage.
  It is never a CLOSED workload.

### 5.5 Sketches

Superseded in detail by the full drafts in `ABSTRACTS.md` (2026-09-29); kept as the short form.

**DiskANN-style search** (per query): hop count `H ~ dist`, beam width `beam`; the first hops from
a small hot set, later hops at random 4 KiB offsets in the index file.

```
for q in $queries {
  for hop in draw(hops) {
    parallel(beam) {
      read(index, offset = if hop < 2 { draw(hot_offsets) } else { draw(uniform_offsets) }, 4KiB)
    }
  }
  compute($rerank)
}
```

**IVF search**: `nprobe` clusters from a popularity-skewed draw, each read sequentially with its
own (skewed) size.

```
for q in $queries {
  parallel(nprobe) { let c = pick(clusters, dist = zipf($s)); read(c, size(c)) }
  compute($distance)
}
```

**Index build**: training-shaped: large sequential reads, emulated compute, large sequential
writes. Data-dependent only in compute time.

**KV cache** (per request): prefix identity from a Zipf-with-recency draw, hit length from a
distribution, sequential reads of the hit blocks, writes of the new blocks with positional names.

```
for r in $requests {
  let p = choose { $reuse%: recent(prefix_site, draw(reuse_dist)), else: draw(new_prefix) }
  let hit = draw(hit_len), let total = draw(prompt_len)
  for k in hit   { read(file("kv/{p:016x}/blk_{k:04}"), $blk) }        # sequential chain
  compute($decode)
  for k in hit..total { write(file("kv/{p:016x}/blk_{k:04}"), $blk) }  # new blocks
}
```

Every one of these has a shape we can write; every one has parameters (`hops`, `zipf` exponent,
`reuse_dist`, `hit_len`, cluster sizes) that must be **measured from a real index on real data**,
exactly as compute time is measured.

## 6. Multi-sample containers and format classes (decided 2026-09-29)

Training data is more often packed into containers (Parquet, HDF5, TFRecord, Arrow IPC,
WebDataset tar shards, Mosaic MDS shards, Megatron `.bin`/`.idx` token files) than stored one
sample per file. This section records how the model covers them. Reasoning in
`DESIGN_REVIEW.md` §3.16; risk R22.

### 6.1 Datasets are sample spaces

A dataset declares `count` **samples**, not files. `consume(ds)` returns a sample handle `s`
with `file(s)`, `offset(s)`, `size(s)`, and `unit(s)` (the row group, chunk, or shard that
contains it). One sample per file is the special case `samples_per_file = 1`, and the
`regions` dataset of `ABSTRACTS.md` §9.6 is the special case of one container file. Epochs are
`drop_last` over samples; the position formula `g + G·(b·B + j)` is unchanged.

### 6.2 Two access modes, declared per dataset

| Mode | Shuffle | Within a container | Real loaders | Formats that support it |
|---|---|---|---|---|
| `map` | Feistel over **sample ids** | seek to `offset(s)`, read `size(s)` (or the unit that holds it) | map-style PyTorch `Dataset`, HF memory-mapped Arrow, DLIO HDF5 | HDF5 (contiguous or one sample per chunk), Arrow IPC, MDS, Megatron token files, LMDB, `regions` |
| `stream` | Feistel over **shard ids**, same sharding formula | sequential reads in `xfer` chunks; an in-memory shuffle buffer that produces no I/O | tf.data `TFRecordDataset` + `interleave` + `shuffle`, WebDataset, HF `streaming=True`, Ray Data, DALI | TFRecord, Parquet (by row group), tar, and any of the above |

The format class (§6.4) lists the modes it supports and the validator refuses the rest:
`map` over TFRecord is impossible (no index) and `map` over Parquet is refused because no
production reader does it (a row costs its whole row group). tf.data's `interleave(cycle_length
= C)` is `parallel(C)` inside a worker over an unordered channel; the shuffle buffer is
application memory. The `loader` sugar gains one knob: items per batch are decoupled from reads
per item, since a batch is `B` records out of a stream rather than `B` files.

### 6.3 Container layout by formula

Samples per file is fixed (the normal case for shards; the last file may be short). Sample sizes
come from the dataset seed as today. A sample's offset inside its file is a prefix sum over
that file's samples, computed once per `open` and held while the file is open. This is
O(open files), not O(files), so the "never materialize per-file structures" invariant holds.
When samples must sit at fixed strides, the padded-slot layout of `ABSTRACTS.md` §9.6 applies.
Because the layout is a formula, `--dry-run` and the fingerprint never need the files.

### 6.4 Format classes: two contracts

A format is a built-in library in the Python builder. It has exactly two halves, and the line
between them is the fidelity guard.

**The format contract** carries what a trace of the named reader library shows regardless of
which application drives it:
- a **layout writer** for `datagen` (real files in the real format, uncompressed, PLAIN
  encoding, payload from the generator), deterministic from the dataset seed and the layout
  parameters (rows per group, page size, columns, samples per file);
- **locate formulas**: sample → (file, unit, offset, length) and unit → (offset, length);
- the reader's **fixed protocol** as ordinary POSIX nodes: on open (Parquet: 8 bytes at EOF−8,
  then the footer; HDF5: superblock, object header, chunk index; TFRecord: nothing), per unit
  (Parquet: the column chunks of the projected columns), per sample (TFRecord: 12-byte header,
  payload, in the buffered reader's request size);
- the **access modes** it supports (§6.2);
- **compute slots** for decode and index parsing, filled by measurement.

**The loader contract** stays in the abstract: access mode, interleave width, prefetch depth,
column projection, batch composition. Column projection is the clearest case: reading 2 of 50
columns turns one sequential row-group read into many small column-chunk reads, and only the
loader knows the projection.

Consequences:
- A class is tied to a reader library and version (`parquet(reader = "pyarrow")`) and is
  derived from and validated against a trace of that library, exactly as an abstract is.
- **The Rust runner stays POSIX-only and format-ignorant.** Nothing about Parquet exists in the
  runner. Index bytes are read because the real reader reads them, and compared to the expected
  layout as drift detection, but never parsed.
- The benchmark never interprets sample data. Content-dependent control flow in real readers
  (TFRecord length prefixes, Parquet page headers) is recomputed from the layout, decode CPU is
  a `compute` slot, and under the `mmap` backend every page of a sample is touched so the fault
  happens.
- An author writes `dataset train = parquet(pattern, count, rows_per_group = …, columns = …,
  seed = …)` plus a loader shape; the class supplies the rest.

### 6.5 Derived workloads to add

- Streaming training over TFRecord or Parquet shards (the MLPerf Storage ResNet50 and CosmoFlow
  shape).
- The HF non-streaming path: a one-time sequential conversion of Parquet to Arrow cache files
  (both sides on the SUT when `HF_HOME` is the shared filesystem), then map-style training from
  the memory-mapped cache. Cache placement (local disk vs. SUT) is a stated parameter.
