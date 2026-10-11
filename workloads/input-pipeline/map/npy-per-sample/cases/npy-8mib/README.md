# `npy-8mib`: an example case of `npy_per_sample`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (npy_per_sample): Fitted to the 2026-10-01 trace of np.load(...)["x"] through a DataLoader: 256 .npz files of about 8 MiB, batch 4, 2 workers, 2 epochs.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-01/`: `trace.metrics.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../npy_per_sample.ast.json --gpus 1 --params-file params.json
```
