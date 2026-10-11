# `sift1m-ivf1024-pq32`: an example case of `faiss_ivf_ondisk`

An **example case**: a configuration small enough to trace, chosen to show that the abstract
issues the application's calls; not a job anyone runs as a benchmark (`DESIGN_REVIEW.md` §3.66).

`params.json` (faiss_ivf_ondisk): Fitted to the 2026-10-01 trace of IndexIVFPQ.search over OnDiskInvertedLists: SIFT1M, IVF1024,PQ32, nprobe 64, 20 calls of one query, one OpenMP thread.

Its parameters hold at 1 GPU, the count it was traced at. Captures:

- `captures/2026-10-01/`: `trace.metrics.json`, `trace.tolerances.json`

The mechanism's README (`../../README.md`) has the commands that made them and what was read
off them.

```
aeiou dry-run ../../faiss_ivf_ondisk.ast.json --gpus 1 --params-file params.json
```
