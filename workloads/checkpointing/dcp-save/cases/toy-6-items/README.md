# `toy-6-items`: an example case of `dcp_save`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (dcp_save): Fitted to the 2026-10-01 trace of torch.distributed.checkpoint.save on 2 ranks: 6 DTensor items, 2 checkpoints.

Its parameters hold at 2 GPUs, the count it was traced at. Captures:

- `captures/2026-10-01/`: `trace.metrics.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../dcp_save.ast.json --gpus 2 --params-file params.json
```
