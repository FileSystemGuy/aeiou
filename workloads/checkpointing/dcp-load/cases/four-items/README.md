# `four-items`: a case of `dcp_load`, not measured

Its kind is not settled. `params.json` holds the values `dcp_load`'s defaults held until
2026-10-23, when the job's values and the model's became required (`DESIGN_REVIEW.md` §3.78
points 1 and 4). They are placeholders, not measured: no job stands behind them, and there is no
capture. They are the checkpoint `dcp_save`'s case of the same name writes at step 100
(`../../../dcp-save/cases/four-items`), so a run of that case leaves what this one reads. The
abstract's calls were traced on the three example cases. This case keeps the old configuration in
the catalog and gives it a lock to run until a use case, a real model's restore on an
accelerator, replaces it.

Two GPUs (`--gpus 2`): `item_bytes` is each rank's share of a tensor, every rank its own shard
(`replicas` 1). Its name is the save case's too, so `--case` names it `dcp-load/four-items`. The
mechanism's README (`../../README.md`) says what the abstract models.
