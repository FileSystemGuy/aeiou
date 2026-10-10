# `sharegpt-chat`: an example case of `lmcache_local_disk`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (lmcache_local_disk): Fitted by fit.py to the 2026-10-02 replay of ShareGPT through vLLM 0.30 + LMCache 0.5.5 (local disk): Qwen2.5-0.5B (12,288 bytes per token, context 4,096), 300 requests, 8 conversations open and served in turn by one client, 2 system prompts of 466 tokens, 96 MiB of GPU KV memory.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-02/`: `lmcache.log`, `replay.log`, `trace.metrics.json`, `trace.tolerances.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../lmcache_local_disk.ast.json --gpus 1 --params-file params.json
```
