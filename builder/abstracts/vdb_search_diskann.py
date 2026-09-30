"""ABSTRACTS.md §5: vector-database search, DiskANN-style (graph index on storage).

Per query, `hops` dependent rounds of `beam` concurrent 4 KiB reads at sector offsets; the
first storage round is biased to hub nodes (§9.9). `--io-backend libaio` is the fidelity
reference; `io_uring` and `sync-direct` are the comparison rows.
"""
from mlps_abstract import *

w = Workload("vdb_search_diskann",
             doc="Beam search over a sector-packed graph index: `hops` dependent rounds of `beam` concurrent 4 KiB reads per query.")
P = w.P
w.param("threads", 32, unit="count", doc="search threads per node (an actor is a search node)")
w.param("queries", 100_000, unit="count", doc="per thread; finite")
w.param("beam", 4, unit="count")
w.param("cached_hops", 2, unit="count", doc="rounds served from the node cache [measure]; documentation only")
w.param("hops", empirical([3, 4, 5, 6, 7, 8], [5, 20, 35, 25, 10, 5]), unit="count",
        doc="[measure] rounds that reach storage: the trace's rounds minus cached_hops")
w.param("hot", hotset(fraction=0.01, weight=0.30), doc="hub nodes [measure from trace]")
w.param("rerank", 120 * us, unit="ns", doc="[measure]")
w.param("sector", 4 * KiB, unit="bytes")

index = w.regions("index", file="diskann/index.bin", count=200_000_000, slot=P.sector,
                  size=const(P.sector), seed=0x5eed_da7d)

with w.actor("gpu") as gpu:
    with gpu.parallel("th", P.threads) as thread:
        f = thread.let("f", index.file())
        thread.open(f, "RDONLY|DIRECT")
        with thread.loop("q", P.queries) as q:
            with thread.loop("hop", draw(P.hops)) as hop:
                with thread.parallel("bm", P.beam) as beam:
                    s = beam.let("s", index.pick(when(hop < 1, P.hot, uniform(0, index.count))))   # §9.9
                    beam.read(f, P.sector, offset=s.offset)
            thread.compute(P.rerank)
        thread.close(f)

if __name__ == "__main__":
    w.write()
