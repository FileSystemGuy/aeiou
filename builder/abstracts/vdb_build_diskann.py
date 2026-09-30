"""ABSTRACTS.md §7: index build (DiskANN). One builder (`--gpus 1`); phases: PQ sample
(small random reads), per-shard sequential read + long compute + sequential write, merge, and
the sector-by-sector disk layout write. `--time-scale` shrinks the compute.
"""
from aeiou import *

w = Workload("vdb_build_diskann",
             doc="Single builder: PQ sample, per-shard build, merge, disk layout write.")
P = w.P
w.param("dim", 128, unit="count")
w.param("n", 1_000_000_000, unit="count")
w.param("shards", 40, unit="count")
w.param("sample", 256_000, unit="count", doc="PQ training rows [config]")
w.param("build_time", normal(1800 * s, 60 * s, min=1 * s), unit="ns", doc="per shard [measure]; scaled by --time-scale")
w.param("xfer", 1 * MiB, unit="bytes")
w.param("shard_index_bytes", 12 * GiB, unit="bytes", doc="[config]")
w.param("index_bytes", 480 * GiB, unit="bytes", doc="[config]")
w.param("sectors", 200_000_000, unit="count", doc="[config]")
w.param("sector", 4 * KiB, unit="bytes")

base = w.regions("base", file="base.fbin", count=P.n, slot=P.dim * 4, size=const(P.dim * 4), seed=0x5eed_da7f)
shard_ns = w.namespace("shard", pattern="diskann/shard_{k:03}.index", fields={"k": int}, size="as_written",
                       seed=0x5eed_da83)
out = w.namespace("out", pattern="diskann/{name}", fields={"name": str}, size="as_written", seed=0x5eed_da83)

shard_bytes = (P.n // P.shards) * P.dim * 4          # bytes of base vectors per shard

with w.actor("gpu", count=1) as b:                    # one builder; --gpus 1
    base_f = b.let("base_f", base.file())
    b.open(base_f, "RDONLY")
    with b.phase("pq_sample"), b.loop("i", P.sample) as i:
        r = b.let("r", base.pick())
        b.read(base_f, r.size, offset=r.offset)
    with b.phase("shard_build"), b.loop("k", P.shards) as k:
        b.lseek(base_f, k * shard_bytes, "SET")
        b.read(base_f, P.xfer, repeat=ceil_div(shard_bytes, P.xfer))
        b.compute(P.build_time)
        sh = b.let("sh", shard_ns.object(k=k))
        b.open(sh, "WRONLY|CREAT|TRUNC")
        b.write(sh, P.xfer, repeat=ceil_div(P.shard_index_bytes, P.xfer))
        b.close(sh)
    with b.phase("merge"):
        with b.loop("k", P.shards) as k:
            sh = b.let("sh", shard_ns.object(k=k))
            b.open(sh, "RDONLY")
            b.read(sh, P.xfer, repeat=ceil_div(P.shard_index_bytes, P.xfer))   # size known from the writes (V4)
            b.close(sh)
        g = b.let("g", out.object(name="merged.index"))
        b.open(g, "WRONLY|CREAT|TRUNC")
        b.write(g, P.xfer, repeat=ceil_div(P.index_bytes, P.xfer))
        b.close(g)
    with b.phase("layout"):
        b.lseek(base_f, 0, "SET")
        b.read(base_f, P.xfer, repeat="until_eof")                              # base re-read for full vectors
        d = b.let("d", out.object(name="index.bin"))
        b.open(d, "WRONLY|CREAT|TRUNC")
        b.write(d, P.sector, repeat=P.sectors)
        b.fsync(d)
        b.close(d)
    b.close(base_f)

if __name__ == "__main__":
    w.write()
