# `chat-64-slots`: a case of `lmcache_local_disk`, not measured

Its kind is not settled. `params.json` holds the values `lmcache_local_disk`'s defaults held
until 2026-10-23, when the model's values on its GPU, the deployment's, and the load's became
required (`DESIGN_REVIEW.md` §3.78 points 1 and 4). They are placeholders, not measured: no
job stands behind them, and there is no capture. The abstract's calls were traced on the
example cases `sharegpt-chat` and `agentx-odirect`. This case keeps the old configuration in
the catalog and gives it a lock to run until a use case, a real model served on an
accelerator, replaces it.

One GPU (`--gpus 1`): one engine, its 64 request slots each serving 40 open conversations in
turn. `lmcache_shared_fs`'s case of the same name holds the same values, so `--case` names this
one `lmcache-local-disk/chat-64-slots`. The mechanism's README (`../../README.md`) says what
the abstract models.
