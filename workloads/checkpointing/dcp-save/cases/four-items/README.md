# `four-items`: a case of `dcp_save`, not measured

Its kind is not settled. `params.json` holds the values `dcp_save`'s defaults held until
2026-10-23, when the job's values and the model's on a GPU became required (`DESIGN_REVIEW.md`
§3.78 points 1 and 4). They are placeholders, not measured: no job stands behind them, and there
is no capture. The abstract's calls were traced on the example case `toy-6-items`, six tensors
saved on two ranks. This case keeps the old configuration in the catalog and gives it a lock to
run until a use case, a real model's checkpoint on an accelerator, replaces it.

Two GPUs (`--gpus 2`): `item_bytes` is each rank's share of a tensor. The mechanism's README
(`../../README.md`) says what the abstract models.
