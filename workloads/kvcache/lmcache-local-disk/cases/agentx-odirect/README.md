# `agentx-odirect`: an example case of `lmcache_local_disk`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (lmcache_local_disk): Fitted by agentx.py to traces.jsonl (8 sessions): 223 requests in 45 chains; 1.1% of continuing requests cut their chain (trim) and the chain grows by 101% of the corpus's prompt growth (the reply counted whole where the model kept less of it); sys_tokens the median prefix a sub-agent's first prompt shares with its session; the requests replay_agentx.py sent (8 sessions as 8 slots, 28 requests each, back to back), and keep from the server's log: of 178 continuing requests the engine held the whole kept prefix in 6.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/replay-2gib-pool/`: `lmcache.log`, `replay.log`, `trace.metrics.json`, `trace.tolerances.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../lmcache_local_disk.ast.json --gpus 1 --params-file params.json
```
