"""Search a FAISS IVF index with on-disk inverted lists: the process to trace.

    python search.py OUT QUERIES.fvecs [--nprobe 64] [--batch 1] [--queries 200]
                     [--threads N] [--k 10]

`--batch` is the number of queries per `index.search` call (the application's choice);
`--threads` sets OpenMP's thread count.
"""
import argparse, os
import numpy as np
import faiss


def fvecs(path):
    a = np.fromfile(path, dtype="int32")
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("queries_file")
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    if a.threads:
        faiss.omp_set_num_threads(a.threads)
    xq = fvecs(a.queries_file)[:a.queries]
    os.chdir(a.out)                      # the index names its lists file relative to itself
    index = faiss.read_index("populated.index")
    index.nprobe = a.nprobe
    for i in range(0, len(xq), a.batch):
        index.search(xq[i:i + a.batch], a.k)


if __name__ == "__main__":
    main()
