# `sift1m-13-shards`: an example case of `diskann_build`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (diskann_build): Fitted to the 2026-10-02 trace of DiskANN build_disk_index (diskannpy 0.7.0): SIFT1M, degree 64, build list 100, 13 shards (build memory 0.25 GB), 20 threads.
Its compute is not fitted: PQ training 59 s and the partition 91 s as traced, and 20 s per
shard, an upper bound from the traced build's 407 s (`../../README.md`).

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-02/`: `trace.metrics.json`, `trace.tolerances.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../diskann_build.ast.json --gpus 1 --params-file params.json
```
