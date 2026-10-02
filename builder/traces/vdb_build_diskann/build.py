"""Build a DiskANN disk index: the process to trace for `vdb_build_diskann`.

    python build.py BASE.fbin INDEX_DIR [--degree 64] [--complexity 100] [--search-gb 0.03]
                    [--build-gb 0.25] [--threads N]

`--build-gb` below the size of the graph in memory makes the builder partition the base
into shards, build each, and merge them (the path a corpus larger than DRAM takes).
`python build.py --fbin SIFT.fvecs BASE.fbin` converts an fvecs file first.
"""
import argparse, os, sys
import numpy as np
import diskannpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--fbin", action="store_true", help="convert: base is an .fvecs file, out the .fbin to write")
    ap.add_argument("--degree", type=int, default=64)
    ap.add_argument("--complexity", type=int, default=100)
    ap.add_argument("--search-gb", type=float, default=0.03)
    ap.add_argument("--build-gb", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.fbin:
        v = np.fromfile(a.base, dtype="int32")
        d = int(v[0])
        diskannpy.vectors_to_file(a.out, v.reshape(-1, d + 1)[:, 1:].copy().view("float32"))
        return
    os.makedirs(a.out, exist_ok=True)
    diskannpy.build_disk_index(data=a.base, distance_metric="l2", index_directory=a.out,
                               complexity=a.complexity, graph_degree=a.degree,
                               search_memory_maximum=a.search_gb, build_memory_maximum=a.build_gb,
                               num_threads=a.threads, vector_dtype=np.float32)


if __name__ == "__main__":
    main()
