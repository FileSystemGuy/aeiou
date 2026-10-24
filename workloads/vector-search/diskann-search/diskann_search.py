"""ABSTRACTS.md §5: vector-database search, DiskANN-style (graph index on storage).

The calls are the trace of DiskANN's `PQFlashIndex` under `diskannpy.StaticDiskIndex`
(2026-10-02, diskannpy 0.7.0, SIFT1M, workloads/vector-search/diskann-search). Loading reads the
PQ-compressed vectors whole through a stream and the first bytes of the index, then opens
the index again with `O_DIRECT`; a node cache, when configured, is filled with sector reads
submitted eight at a time. A query is a beam search: each round submits up to `beam` sector
reads in one `io_submit` and waits for all of them. Without a node cache the first round is
one read, the sector of an entry point (a medoid), and the second round lands on the entry
points' neighbours; every later round is spread over the whole index. A node cache of 1 % of
the nodes holds exactly those two rounds. The index (named as `diskann_build` names it, its
sectors derived), the search's configuration, its load, and its CPU are required
(DESIGN_REVIEW.md §3.78, 2026-10-23): the traced examples sift1m-nocache and sift1m-cache give
them, and the billion-point stand-in the defaults held until then is no case.
"""
from aeiou import *

w = Workload("diskann_search", posix="libaio",   # DiskANN on Linux: io_submit on an O_DIRECT descriptor
             doc="Beam search over a sector-packed graph index: dependent rounds of `beam` concurrent 4 KiB reads per query.")
P = w.P
w.param("threads", kind=int, unit="count", doc="search threads per node (an actor is a search node)")
w.param("queries", kind=int, unit="count", doc="per thread; finite")
w.param("beam", kind=int, unit="count", doc="[config] the beam width: sector reads per round")
w.param("hops", kind=Dist, unit="count",
        doc="rounds of a query after the two near the entry points: the index's at its search list and beam (hops.py fits it)")
w.param("entry", kind=Dist,
        doc="the entry points' sectors, a hot set: medoids / sectors (one medoid per build shard; hops.py fits it)")
w.param("near", kind=Dist,
        doc="the sectors of the entry points' neighbours, a hot set: about 0.8 * medoids * degree / sectors (661 distinct sectors for 13 medoids of degree 64, measured)")
w.param("node_cache", kind=int, unit="count",
        doc="[config] sector reads that fill the node cache at load (about one per cached node), 0 for none; with a cache, queries skip the two rounds near the entry points")
w.param("cache_batch", 8, unit="count", doc="requests per submit while the node cache loads (traced)")
w.param("rerank", kind=int, unit="ns", doc="[measure] CPU of one query (0.5 ms of user time over 28 rounds measured on SIFT1M, one host)")
w.param("sector", 4 * KiB, unit="bytes")
# the index, as diskann_build names it
w.param("n", kind=int, unit="count", doc="the index's points")
w.param("dim", kind=int, unit="count", doc="the points' dimensions (fp32)")
w.param("degree", kind=int, unit="count", doc="[config] the graph's degree, as the disk layout read it from the merged graph")
w.param("pq_code", kind=int, unit="bytes", doc="[config] bytes of one PQ-compressed vector, held in memory")
w.param("stream", 8191, unit="bytes", doc="the first read of a C++ stream on a file (traced)")
w.param("pq_read", 0x7ffff000, unit="bytes", doc="the stream reads the rest of the PQ file in one call; Linux returns at most this much per read")

node = P.dim * 4 + 4 + P.degree * 4                    # a node in the disk index: its vector, its degree, its neighbours
# DiskANN's layout: a header sector, then whole nodes to a sector, or a node bigger than a sector in whole sectors
sectors = 1 + ceil_div(P.n, max_(1, P.sector // node)) * ceil_div(node, P.sector)

index = w.dataset("index", pattern="diskann/index.bin", layout="array", access="indexed", count=sectors, slot=P.sector,
                  size=const(P.sector), seed=0x5eed_da7d)
pq = w.dataset("pq", pattern="pq/pq_compressed-{id}.bin", layout="whole", access="indexed", count=1,
               size=const(P.n * P.pq_code + 8),
               seed=0x5eed_da80)   # beside the index in the real layout; a dataset has its root to itself here (V13)

with w.actor("gpu") as gpu:
    c = gpu.let("c", pq.file(0))                                         # the PQ-compressed vectors, into memory
    gpu.open(c, "RDONLY")
    gpu.lseek(c, 0, "END")
    gpu.lseek(c, 0, "SET")
    gpu.lseek(c, 0, "SET")
    gpu.read(c, P.stream)
    gpu.read(c, P.pq_read, repeat=ceil_div(c.size - P.stream, P.pq_read))
    gpu.close(c)
    f = gpu.let("f", index.file())
    gpu.open(f, "RDONLY")                                                # the index's first sector, through a stream
    gpu.read(f, P.stream)
    gpu.close(f)
    gpu.open(f, "RDONLY|DIRECT")                                         # one descriptor for every search thread
    with gpu.loop("cl", P.node_cache // P.cache_batch) as cl:            # the node cache's load
        with gpu.parallel("cb", P.cache_batch) as cb:
            n = cb.let("n", index.pick())
            cb.read(f, P.sector, offset=n.offset)
    with gpu.parallel("th", P.threads) as thread:
        with thread.loop("q", P.queries) as q:
            with thread.when(P.node_cache == 0):
                with thread.parallel("en", 1) as en:                     # a submit of one: the entry point's sector
                    e = en.let("e", index.pick(P.entry))
                    en.read(f, P.sector, offset=e.offset)
                with thread.parallel("nb", P.beam) as nb:                # its neighbours
                    s = nb.let("s", index.pick(P.near))
                    nb.read(f, P.sector, offset=s.offset)
            with thread.loop("hop", draw(P.hops)) as hop:
                with thread.parallel("bm", P.beam) as beam:
                    s = beam.let("s", index.pick())
                    beam.read(f, P.sector, offset=s.offset)
            thread.compute(P.rerank)                                     # the query's CPU, in fact spread over its rounds
    gpu.close(f)

if __name__ == "__main__":
    w.write()
