# `agentx`: an example case of `lmcache_local_disk`

Its kind is not settled. `params.json` is the fit to the whole AgentX corpus (393 sessions,
98,827 requests, 32 MiB chunks), the request stream of a real job, which makes it the first
candidate for a **use case**; but `keep` is not measured in it (the corpus saw no engine) and
it was never traced at that scale. The captures are of 8-session replays on one small GPU,
each with the parameters fitted to it (`fitted.params.json`): example-case evidence
(`DESIGN_REVIEW.md` §3.66).

`params.json` (lmcache_local_disk): Fitted by agentx.py to traces.jsonl (393 sessions): 98827 requests in 10391 chains; 3.7% of continuing requests cut their chain (trim) and the chain grows by 108% of the corpus's prompt growth (the reply counted whole where the model kept less of it); sys_tokens the main agent's prefix (the prefix its cuts most often keep), sub_tokens the median prefix a sub-agent's first prompt shares with its session, kind the chains' first prompts by the prefix they open with; keep not measured (the proxy saw no engine).

One engine (`--gpus 1`). Captures:

- `captures/replay-2gib-pool/`: `fitted.params.json`, `lmcache.log`, `replay.log`, `trace.metrics.json`, `trace.tolerances.json`
- `captures/replay-6gib-pool/`: `fitted.params.json`, `lmcache.log`, `replay.log`, `trace.metrics.json`, `trace.tolerances.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../lmcache_local_disk.ast.json --gpus 1 --params-file params.json
```
