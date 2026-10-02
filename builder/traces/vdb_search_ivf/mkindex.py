"""Build a FAISS IVF index whose inverted lists live in one file on storage.

The documented route (faiss/contrib/ondisk.py, demos/demo_ondisk_ivf.py): train, fill
shard indexes in memory, write them, then `merge_ondisk` packs their lists back to back
into `<out>/merged_index.ivfdata` and `<out>/populated.index` refers to that file.

    python mkindex.py SIFT_DIR OUT [--factory IVF1024,PQ32] [--shards 4]

SIFT_DIR holds sift_base.fvecs and sift_learn.fvecs (ftp.irisa.fr/local/texmex/corpus).
"""
import argparse, os, sys
import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk


def fvecs(path):
    a = np.fromfile(path, dtype="int32")
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sift")
    ap.add_argument("out")
    ap.add_argument("--factory", default="IVF1024,PQ32")
    ap.add_argument("--shards", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    xt = fvecs(os.path.join(a.sift, "sift_learn.fvecs"))
    xb = fvecs(os.path.join(a.sift, "sift_base.fvecs"))
    index = faiss.index_factory(xb.shape[1], a.factory)
    index.train(xt)
    trained = os.path.join(a.out, "trained.index")
    faiss.write_index(index, trained)
    names = []
    n = len(xb)
    for s in range(a.shards):
        i0, i1 = s * n // a.shards, (s + 1) * n // a.shards
        sub = faiss.read_index(trained)
        sub.add_with_ids(xb[i0:i1], np.arange(i0, i1))
        names.append(os.path.join(a.out, "block_%d.index" % s))
        faiss.write_index(sub, names[-1])
    index = faiss.read_index(trained)
    merge_ondisk(index, names, os.path.join(a.out, "merged_index.ivfdata"))
    faiss.write_index(index, os.path.join(a.out, "populated.index"))
    for f in names + [trained]:
        os.unlink(f)
    print("%d vectors, %d lists, code_size %d" % (index.ntotal, index.nlist, index.code_size),
          file=sys.stderr)


if __name__ == "__main__":
    main()
