# `chat-64-slots`: a case of `lmcache_shared_fs` and its reader, not measured

Its kind is not settled. `writer.params.json` and `reader.params.json` hold the same values,
the ones `lmcache_shared_fs`'s defaults held until 2026-10-23, when the model's values on its
GPU, the deployment's, and the load's became required (`DESIGN_REVIEW.md` §3.78 points 1 and
4): those of `../../../lmcache-local-disk/cases/chat-64-slots`, the same request stream. They
are placeholders, not measured: no job stands behind them, and there is no capture. The
abstracts' calls were traced on the example case `sharegpt-chat`. The chunk file's buffer is
the store's `st_blksize` (contract 0.12), not a value of the case.

Two runs (`--case-run`): `lmcache_shared_fs` fills an empty store, and
`lmcache_shared_fs_reader`, a cold engine, reads what it left; the reader is run with the
writer's `--seed`, `--gpus`, and parameters. One GPU (`--gpus 1`). The mechanism's README
(`../../README.md`) says what the abstracts model.
