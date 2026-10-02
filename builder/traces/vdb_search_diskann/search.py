"""Search a DiskANN disk index: the process to trace for `vdb_search_diskann`.

    python search.py INDEX_DIR QUERIES.fbin [--queries 200] [--threads 2] [--beam 4]
                     [--complexity 100] [--cache 0] [--cache-mechanism 0] [--k 10] [--batch]

Queries go one `search` call at a time from one thread, or with `--batch` as one
`batch_search` over `--threads` threads. `--cache` is `num_nodes_to_cache`, and
`--cache-mechanism` chooses how they are found: 0 none, 1 by searching the build's sample
of the base (diskannpy's default, about 100,000 searches before the first query), 2 by
breadth-first levels from the medoids (what DiskANN's own `search_disk_index` does).
"""
import argparse, os
import numpy as np
import diskannpy

MARK = "/aeiou-query-mark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("queries_file")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--complexity", type=int, default=100)
    ap.add_argument("--cache", type=int, default=0)
    ap.add_argument("--cache-mechanism", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    xq = diskannpy.vectors_from_file(a.queries_file, np.float32)[:a.queries]
    index = diskannpy.StaticDiskIndex(index_directory=a.index, num_threads=a.threads,
                                      num_nodes_to_cache=a.cache, cache_mechanism=a.cache_mechanism, distance_metric="l2",
                                      vector_dtype=np.float32, dimensions=xq.shape[1])
    if a.batch:
        index.batch_search(xq, a.k, a.complexity, a.threads, a.beam)
    else:
        for q in xq:
            try:
                os.stat(MARK)                   # a line in the trace between two queries, for hops.py
            except OSError:
                pass
            index.search(q, a.k, a.complexity, a.beam)


if __name__ == "__main__":
    main()
