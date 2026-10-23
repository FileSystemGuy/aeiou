# `five-tensors`: a case of `safetensors_load`, not measured

Its kind is not settled. `params.json` holds the values `safetensors_load`'s defaults held until
2026-10-23, when the model's shards and tensor table and the deployment's `tp` became required
(`DESIGN_REVIEW.md` §3.78 points 1 and 4). They are placeholders, not measured: a five-tensor
stand-in in two 5 GiB shards, no model's table behind it, and there is no capture. The
abstract's calls were traced on the example case `gpt2-random`. This case keeps the old
configuration in the catalog and gives it a lock to run until a use case, a real model's table
made by `aeiou-params safetensors` from its shards and served at its `tp`, replaces it.

Two GPUs (`--gpus 2`) at `tp` 8: ranks 0 and 1 take their slices. The mechanism's README
(`../../README.md`) says what the abstract models.
