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
| 8 | **Step counter and conditionals** (`every 500`, `if step % n == 0`) | Periodic barriers and checkpoints. |

The semantic model below is the same for all three syntax options. Only the surface
syntax differs.

## 2. Semantic model (common to all options)

- A **workload** declares `params`, `datasets`, and one or more **actor templates** (e.g. `gpu`).
- `--gpus G` instantiates G actors of the `gpu` template, with **global** ids `0..G`. MPI ranks
  own contiguous id ranges. The ids do not depend on the number of ranks, so the same run can be
  spread over 10 or 20 nodes and produce the same workload.
- An actor runs a **program** (a tree of elements). An element can suspend on:
  an I/O completion, a timer, a join, a channel `take`, or a barrier.
- **`loader`** is a built-in pattern that expands to W worker sub-actors plus an in-order ring
  buffer of `W × prefetch` batch slots. Worker w builds batches `w, w+W, w+2W, …`, matching
  PyTorch's worker assignment. `take(loader)` waits for the *next batch in order*.
- **`consume(ds)`** is `perm_{seed,epoch}(position)`, where positions are sharded by global actor
  id (`gpu_id + k·G`). Position counters are per actor, so no shared state is needed.
- **Randomness** at any point is `hash(seed, actor_id, epoch, step, site_id, draw#)`.
  `site_id` is a stable id for the place in the abstract where the draw happens.
- **The implementation compiles the tree to bytecode** that runs on a per-actor VM, in the style
  of a regex VM: a stack of `(pc, remaining_count)` frames plus a few registers for bindings.
  `parallel`/`loader` spawn child VMs, and `join`/`take` are wait instructions.

## 3. Syntax options

### Option A — Extended regex-style DSL (recommended)

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
                        size  = const(140MiB))

  per gpu {
    loader batches(workers = $workers, prefetch = $prefetch, order = in_order) {
      # one worker builds one batch
      ( let f = consume(train),
        open(f, RDONLY), fstat(f),
        read(f, $xfer)[until_eof],
        close(f)
      )[$batch]
    }

    ( take(batches),
      compute($compute),
      every $sync_every { barrier(global) }
    )[$steps]
  }
}
```

Checkpoint fragment, as an example of what writes look like:

```
    every 100 {
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
        body:
          - repeat: $batch
            do:
              - let: { f: { consume: train } }
              - open: { file: f, flags: RDONLY }
              - fstat: f
              - repeat: until_eof
                do: [ { read: { file: f, size: $xfer } } ]
              - close: f
    - repeat: $steps
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

## 4. Recommendation

**Option A for the language people write, with Option B's tree as the canonical AST
serialization.** The parser produces a serde AST. `--dump-ast` emits YAML/JSON, and the runner
also accepts that AST directly. We get a readable language and still have a machine-friendly
format for generated abstracts (e.g. from traces), with a single semantic model.

Guardrails to keep the language from growing into a general-purpose language:

- No unbounded loops. Every repeat count is a constant, a distribution, a parameter, or a
  function of dataset metadata (`size(f)`), and `until_eof` is resolved from the known file size.
  This means the total op count of a run can be computed before it starts.
- Conditionals are allowed only on the step/epoch counters and parameters, never on timing or
  completion order. This is what makes the workload fingerprint (see `NAPKIN_MATH.md` §8) exact.
- Wall-clock-bounded phases (`for 60s`) are allowed only as an opt-in that turns off the exact
  fingerprint guarantee.
