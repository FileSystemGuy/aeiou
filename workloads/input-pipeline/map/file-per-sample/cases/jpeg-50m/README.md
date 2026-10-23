# `jpeg-50m`: a case of `file_per_sample`, not measured

Its kind is not settled. `params.json` holds the values `file_per_sample`'s defaults held until
2026-10-22, when the job's values and the model's on a GPU became required (`DESIGN_REVIEW.md`
§3.78 points 1 and 4). They are placeholders, not measured: no job stands behind them, and there
is no capture. The abstract's calls were traced on the example case `imagefolder-jpeg`, a
3,200-file stand-in. This case keeps the old configuration in the catalog and gives it a lock
to run until a use case, a real job on an accelerator, replaces it.

One GPU (`--gpus 1`). The mechanism's README (`../../README.md`) says what the abstract models.
