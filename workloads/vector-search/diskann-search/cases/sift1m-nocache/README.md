# `sift1m-nocache`: an example case of `diskann_search`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (diskann_search): Fitted to the 2026-10-02 trace of DiskANN PQFlashIndex (diskannpy 0.7.0): SIFT1M, degree 64, search list 100, beam 4, no node cache, 1,000 queries over 2 threads.
Its CPU (`rerank`) is not fitted per case: 0.5 ms a query, the traces' user time.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-02/`: `trace.metrics.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../diskann_search.ast.json --gpus 1 --params-file params.json
```
