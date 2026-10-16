"""TEST FAKE of `diskannpy` (see `standins/standin.py`): `.fbin` files as diskannpy writes them
(uint32 count and dimension, then the float32 rows); the index calls only record."""
import struct

import numpy as np
from standin import record


def vectors_to_file(vector_file, vectors):
    record("vectors_to_file", path=vector_file, shape=list(vectors.shape), dtype=str(vectors.dtype))
    with open(vector_file, "wb") as f:
        f.write(struct.pack("<II", *vectors.shape) + vectors.tobytes())


def vectors_from_file(vector_file, dtype):
    record("vectors_from_file", path=vector_file, dtype=dtype)
    with open(vector_file, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(), dtype=dtype).reshape(n, d)


def build_disk_index(**kw):
    record("build_disk_index", **kw)


class StaticDiskIndex:
    def __init__(self, **kw):
        record("StaticDiskIndex", **kw)

    def search(self, query, k_neighbors, complexity, beam_width):
        record("search", dim=len(query), k=k_neighbors, complexity=complexity, beam=beam_width)

    def batch_search(self, queries, k_neighbors, complexity, num_threads, beam_width):
        record("batch_search", shape=list(queries.shape), k=k_neighbors, complexity=complexity, threads=num_threads,
               beam=beam_width)
