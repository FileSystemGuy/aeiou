"""TEST FAKE of `faiss` (see `standins/standin.py`): an `IVF<nlist>,PQ<m>` index whose vector `i` is
in list `i mod nlist` and whose coarse quantizer probes lists `7q, 7q + 1, …` for query `q`.
`write_index` writes the index's fields as JSON; `read_index` opens the lists file under the name
`merge_ondisk` stored, relative to the working directory, as `OnDiskInvertedLists` does when not
given `IO_FLAG_ONDISK_SAME_DIR`."""
import json
import os

import numpy as np
from standin import record


class _List:
    def __init__(self, size, offset):
        self.size, self.offset = size, offset


class _Lists:
    def __init__(self, items):
        self.items = items

    def at(self, i):
        return self.items[i]


class _OnDiskInvertedLists:
    prefetch_nthread = 32                     # faiss's default

    def __init__(self, counts, code_size):
        self.nlist, self.code_size = len(counts), code_size
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) * (code_size + 8)
        self.lists = _Lists([_List(int(c), int(o)) for c, o in zip(counts, offsets)])


class _Quantizer:
    def __init__(self, nlist):
        self.nlist = nlist

    def search(self, x, k):
        record("quantizer.search", n=len(x), k=k)
        return None, (np.arange(len(x))[:, None] * 7 + np.arange(k)[None, :]) % self.nlist


class Index:
    def __init__(self, d, nlist, code_size, counts=None, ivfdata=None):
        self.d, self.nlist, self.code_size, self.ivfdata = d, nlist, code_size, ivfdata
        self.counts = counts or [0] * nlist
        self.quantizer = _Quantizer(nlist)
        self.nprobe = 1

    @property
    def ntotal(self):
        return sum(self.counts)

    @property
    def invlists(self):
        return _OnDiskInvertedLists(self.counts, self.code_size)

    def train(self, x):
        record("Index.train", shape=list(x.shape), dtype=str(x.dtype))

    def add_with_ids(self, x, ids):
        record("Index.add_with_ids", n=len(x), ids=[int(ids[0]), int(ids[-1])])
        for i in ids:
            self.counts[i % self.nlist] += 1

    def search(self, x, k):
        record("Index.search", n=len(x), k=k, nprobe=self.nprobe)


def index_factory(d, description):
    record("index_factory", d=d, description=description)
    ivf, pq = description.split(",")
    return Index(d, int(ivf[3:]), int(pq[2:]))


def write_index(index, fname):
    record("write_index", path=fname)
    with open(fname, "w") as f:
        json.dump({"d": index.d, "nlist": index.nlist, "code_size": index.code_size, "counts": index.counts,
                   "ivfdata": index.ivfdata}, f)


def read_index(fname):
    record("read_index", path=fname, cwd=os.getcwd())
    with open(fname) as f:
        index = Index(**json.load(f))
    if index.ivfdata is not None:
        open(index.ivfdata, "rb").close()
    return index


def downcast_InvertedLists(il):
    return il


def omp_set_num_threads(n):
    record("omp_set_num_threads", n=n)
