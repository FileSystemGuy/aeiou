"""ABSTRACTS.md §6: vector-database search, IVF (inverted lists on storage, FAISS
OnDiskInvertedLists shape). A query reads `nprobe` lists, each one contiguous region, with
Zipf popularity; no dependency between the reads. `--io-backend mmap` is FAISS's real path.
"""
from mlps_abstract import *

w = Workload("vdb_search_ivf",
             doc="Per query, nprobe concurrent reads of whole inverted lists chosen by Zipf popularity.")
P = w.P
w.param("threads", 32, unit="count")
w.param("queries", 100_000, unit="count")
w.param("nprobe", 64, unit="count")
w.param("list_pop", zipf(s=0.8), doc="list popularity [measure]")
w.param("distance", 400 * us, unit="ns", doc="[measure]")

lists = w.regions("lists", file="ivf/invlists.bin", count=1_000_000,
                  slot=256 * KiB,                                        # ≥ p99 list size
                  size=lognormal(median=40 * KiB, sigma=0.9),            # codes+ids [measure]
                  seed=0x5eed_da7e)

with w.actor("gpu") as gpu:
    with gpu.parallel("th", P.threads) as thread:
        f = thread.let("f", lists.file())
        thread.open(f, "RDONLY")
        with thread.loop("q", P.queries) as q:
            with thread.parallel("p", P.nprobe) as probe:
                c = probe.let("c", lists.pick(P.list_pop))
                probe.read(f, c.size, offset=c.offset)                   # one sequential run per list
            thread.compute(P.distance)
        thread.close(f)

if __name__ == "__main__":
    w.write()
