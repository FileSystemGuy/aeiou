# `gpt2-random`: an example case of `safetensors_load`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (safetensors_load): Fitted to the 2026-10-01 trace of from_pretrained: the tensor table of the kit's five shards (GPT-2 shape, 6 layers, random weights); tp 1, every tensor whole.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-01/`: `trace.metrics.json`, `trace.tolerances.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../safetensors_load.ast.json --gpus 1 --params-file params.json
```
