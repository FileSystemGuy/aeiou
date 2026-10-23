# `imagefolder-jpeg`: an example case of `file_per_sample`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (file_per_sample): Fitted to the 2026-10-01 trace of DataLoader + ImageFolder: 3200 JPEGs, batch 16, 2 workers, 2 epochs; sample_median is the corpus kit/mkcorpus.py writes. step_time is the abstract's default until 2026-10-22, 105 ms, not fitted to the trace.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-01/`: `trace.metrics.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../file_per_sample.ast.json --gpus 1 --params-file params.json
```
