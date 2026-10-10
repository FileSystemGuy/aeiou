# `name-order-17-items`: an example case of `dcp_load`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (dcp_load): Fitted to the 2026-10-01 trace of torch.distributed.checkpoint.load on 2 ranks: 17 items, runs of small ones longer than the buffer, read in name order.

Its parameters hold at 2 GPUs, the count it was traced at. Captures:

- `captures/2026-10-01/`: `trace.metrics.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../dcp_load.ast.json --gpus 2 --params-file params.json
```
