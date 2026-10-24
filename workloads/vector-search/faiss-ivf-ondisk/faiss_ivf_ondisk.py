"""ABSTRACTS.md §6: vector-database search, IVF (inverted lists on storage, FAISS
OnDiskInvertedLists shape).

The calls are the trace of `faiss.read_index` and `IndexIVFPQ.search` over an index made by
`merge_ondisk` (2026-10-01, faiss 1.15.1, workloads/vector-search/faiss-ivf-ondisk). `read_index` reads
the index file (quantizer, codebooks, the table of list offsets) through stdio and maps the
lists file whole; a search issues no call on it at all. `search` cuts its batch into one
slice per OpenMP thread, and each slice starts `prefetch` threads
(`OnDiskInvertedLists::prefetch_nthread`, 32) that touch every byte of every list its
queries probe while the slice's own thread scans them. The reads below are what the prefetch
threads touch through the mapping, and `--posix mmap` is the application's own API here.
Popularity and the list-size spread are measured on one corpus (SIFT1M, IVF1024,PQ32).
The index, the search's configuration, its load, and its CPU are required (DESIGN_REVIEW.md
§3.78, 2026-10-23): the traced example sift1m-ivf1024-pq32 gives them, and the million-list
stand-in the defaults held until then is no case. The prefetch threads are derived, as FAISS
starts them.
"""
from aeiou import *

w = Workload("faiss_ivf_ondisk", posix="mmap",   # FAISS maps the lists file and never calls read on it
             doc="Per search call and OpenMP slice, prefetch threads touch every inverted list the slice's queries probe; lists chosen by popularity.")
P = w.P
w.param("threads", kind=int, unit="count", doc="[config] OpenMP threads: a search call of n queries runs min(threads, n) slices")
w.param("calls", kind=int, unit="count", doc="search calls")
w.param("batch", kind=int, unit="count", doc="[config] queries per slice per call (the call's batch is threads * batch)")
w.param("nprobe", kind=int, unit="count", doc="[config] lists probed per query")
w.param("prefetch_nthread", 32, unit="count",
        doc="the library's prefetch threads per slice at most (`OnDiskInvertedLists::prefetch_nthread`, traced)")
w.param("list_pop", kind=Dist, doc="list popularity over the queries (params.py fits it; 0.20-0.25 on SIFT1M with its own query set)")
w.param("distance", kind=int, unit="ns", doc="[measure] scan time per query (no trace measures it: the traced example takes 0.4 ms, the value the defaults held, not fitted)")
# the index, as mkindex.py builds it and params.py reads it
w.param("lists", kind=int, unit="count", doc="[config] inverted lists")
w.param("list_bytes", kind=Dist,
        doc="[config] codes + ids of one list: vectors per list * (code size + 8), a distribution over the lists (params.py fits it); its max no larger than the slot")
w.param("code_size", kind=int, unit="bytes", doc="[config] bytes of one vector's code (32 for PQ32); its id is 8 more")
w.param("index_bytes", kind=int, unit="bytes",
        doc="[config] the index file: centroids (lists * dimension * 4 bytes), codebooks, 16 bytes per list")
w.param("index_reads", kind=list[int], unit="bytes",
        doc="[config] read lengths on the index file: stdio fills 8,192 bytes, and reads the bulk of each large array (centroids, codebooks, list table) directly (params.py reads them from the trace)")

# FAISS starts min(prefetch_nthread, batch * nprobe) prefetch threads per slice; batch * nprobe is modeled as a multiple of it
prefetch = min_(P.prefetch_nthread, P.batch * P.nprobe)

lists = w.dataset("lists", pattern="ivf/merged_index.ivfdata", layout="array", access="indexed", count=P.lists,
                  slot=256 * KiB,                                        # ≥ the largest list
                  size=P.list_bytes, seed=0x5eed_da7e)
index = w.dataset("index", pattern="index/populated-{id}.index", layout="whole", access="indexed", count=1, size=const(P.index_bytes),
                  seed=0x5eed_da7f)    # beside the lists file in the real layout; a dataset has its root to itself here (V13)

with w.actor("gpu") as gpu:                                              # one searching process
    i = gpu.let("i", index.file(0))                                      # faiss.read_index
    gpu.open(i, "RDONLY")
    gpu.fstat(i)
    with gpu.loop("r", P.index_reads.len) as r:
        gpu.read(i, P.index_reads[r])                                    # the last is short
    f = gpu.let("f", lists.file())
    gpu.open(f, "RDONLY")                                                # traced O_RDWR and a shared writable mapping (no IO_FLAG_READ_ONLY); the application closes f here: its mapping stays, ours needs the handle
    gpu.close(i)
    with gpu.loop("q", P.calls) as q:
        with gpu.parallel("th", P.threads) as thread:                    # a slice of the batch
            with thread.parallel("p", prefetch) as pf:
                with pf.loop("k", P.batch * P.nprobe // prefetch) as k:
                    c = pf.let("c", lists.pick(P.list_pop))
                    ids = pf.let("ids", c.size * 8 // (P.code_size + 8))
                    pf.read(f, ids, offset=c.offset + c.size - ids)      # a list is its codes, then its ids; the prefetch
                    pf.read(f, c.size - ids, offset=c.offset)            # sums the ids first, then the codes
            thread.compute(P.distance * P.batch)                         # the scan, over pages the prefetch brought in
    gpu.close(f)

if __name__ == "__main__":
    w.write()
