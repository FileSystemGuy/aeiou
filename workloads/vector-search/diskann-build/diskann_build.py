"""ABSTRACTS.md §7: index build (DiskANN), one builder (`--gpus 1`).

The calls are the trace of DiskANN's `build_disk_index` under diskannpy (2026-10-02, SIFT1M,
13 shards, workloads/vector-search/diskann-build). The build reads the base file whole twenty
times at 13 shards (five block passes, and a load per shard and two more), in two ways: through a 64 MiB
block reader whose last partial block goes row by row through an 8,191-byte stream (the
sampling passes and the layout), and as one load into memory (PQ compression, and once per
shard to pick its rows). Each shard's rows are copied to a file, read back, built, and saved
as a graph and a second copy of the rows; the merge reads the graphs back with one `read`
each and writes the merged graph with one `write`; the layout writes the index in 64 MiB
pieces. Nothing is synced. `--time-scale` shrinks the compute. The base's shape, the build's
configuration, the graphs it makes, and its compute are required (DESIGN_REVIEW.md §3.78,
2026-10-23): the traced example sift1m-13-shards gives them, and the billion-point stand-in
the defaults held until then is no case (its metrics exceed the dry run's limit).
"""
from aeiou import *

w = Workload("diskann_build",
             doc="Single builder: PQ training and compression, partition into shards, per-shard build, merge, disk layout, sample.")
P = w.P
w.param("dim", kind=int, unit="count", doc="the base's dimensions (fp32)")
w.param("n", kind=int, unit="count", doc="the base's points")
w.param("shards", kind=int, unit="count", doc="[config] from the build's memory budget")
w.param("overlap", 2, unit="count", doc="shards a point is put in (traced)")
w.param("cache", 64 * MiB, unit="bytes", doc="the block reader's and the layout writer's block (traced)")
w.param("stream", 8191, unit="bytes", doc="the read length of a C++ stream on a file (traced)")
w.param("chunk", 8192, unit="bytes", doc="the write length of a C++ stream (traced)")
w.param("big", 0x7ffff000, unit="bytes", doc="one read or write call for a whole array; Linux transfers at most this much per call")
w.param("pq_code", kind=int, unit="bytes", doc="[config] bytes of one PQ-compressed vector")
w.param("degree", kind=int, unit="count", doc="[config] the graph's degree, as the disk layout reads it from the merged graph")
w.param("shard_index_bytes", kind=int, unit="bytes", doc="a shard's graph (measured: about 132 bytes per point of the shard at degree 64)")
w.param("index_bytes", kind=int, unit="bytes", doc="the merged graph (measured: about 205 bytes per point at degree 64)")
w.param("sector", 4 * KiB, unit="bytes")
w.param("sample_rows", kind=int, unit="count", doc="rows of the sample of the base written for the search's warm-up (100,131 of a million traced)")
w.param("pq_train", kind=int, unit="ns", doc="[measure] PQ training: the build's on its hosts")
w.param("partition_time", kind=int, unit="ns", doc="[measure] the partition into shards")
w.param("build_time", kind=int, unit="ns", doc="[measure] per shard; scaled by --time-scale")

row = P.dim * 4
base_bytes = P.n * row + 8
shard_rows = P.overlap * P.n // P.shards               # modeled equal; traced 52 to 111 MB of rows per shard, mean 79
shard_bytes = shard_rows * row + 8
ids_bytes = shard_rows * 4 + 8
node = row + 4 + P.degree * 4                          # a node in the disk index: its vector, its degree, its neighbours
# DiskANN's layout: a header sector, then whole nodes to a sector, or a node bigger than a sector in whole sectors
sectors = 1 + ceil_div(P.n, max_(1, P.sector // node)) * ceil_div(node, P.sector)

base = w.dataset("base", pattern="base/base.fbin", layout="array", access="indexed", count=P.n, slot=row,
                 size=const(row), seed=0x5eed_da7f)
tmp = w.namespace("tmp", pattern="diskann/tmp_subshard-{k}_{part}", fields={"k": int, "part": str}, size="as_written",
                  seed=0x5eed_da83)
out = w.namespace("out", pattern="diskann/{name}", fields={"name": str}, size="as_written", seed=0x5eed_da83)

uid = iter(range(1000))


def put(b, f, total, chunk):
    """Write `total` bytes in calls of `chunk`, the last one shorter."""
    b.write(f, chunk, repeat=total // chunk)
    with b.when(total % chunk > 0):
        b.write(f, total % chunk)


def block_pass(b, f):
    """The base file through the block reader: whole blocks, each a stream fill and the rest of the
    block; then the last partial block row by row through the stream, the position asked twice a row."""
    u = next(uid)
    b.open(f, "RDONLY")
    b.lseek(f, 0, "END")
    b.lseek(f, 0, "SET")
    with b.loop(f"blk{u}", base_bytes // P.cache):
        b.read(f, P.stream)
        b.read(f, P.cache - P.stream)
    with b.loop(f"fill{u}", ceil_div(base_bytes % P.cache, P.stream)):
        b.read(f, P.stream)
        with b.loop(f"row{u}", ceil_div(P.stream, row)):
            b.lseek(f, 0, "CUR")
            b.lseek(f, 0, "CUR")
    b.close(f)


def load_pass(b, f):
    """The base file loaded into memory: one block, then everything else in one read."""
    b.open(f, "RDONLY")
    b.lseek(f, 0, "END")
    b.lseek(f, 0, "CUR")
    b.lseek(f, 0, "SET")
    b.read(f, P.cache)
    b.lseek(f, 0, "CUR")
    b.read(f, P.big, repeat=ceil_div(base_bytes - P.cache, P.big))
    b.lseek(f, 0, "CUR")
    b.close(f)


with w.actor("gpu", count=1) as b:                    # one builder; --gpus 1
    base_f = b.let("base_f", base.file())
    with b.phase("pq"):
        block_pass(b, base_f)                          # the training sample
        b.compute(P.pq_train)
        load_pass(b, base_f)                           # compress every vector
        c = b.let("c", out.object(name="pq_compressed.bin"))
        b.open(c, "WRONLY|CREAT|TRUNC")
        put(b, c, P.n * P.pq_code + 8, P.big)          # one write
        b.close(c)
    with b.phase("partition"):
        block_pass(b, base_f)                          # a sample for the shard centres
        block_pass(b, base_f)                          # every point to its `overlap` nearest centres
        b.compute(P.partition_time)
        load_pass(b, base_f)                           # the whole base again, before the ids are written
        with b.loop("k", P.shards) as k:               # (the application has every ids file open at once)
            i = b.let("i", tmp.object(k=k, part="ids"))
            b.open(i, "WRONLY|CREAT|TRUNC")
            put(b, i, ids_bytes, P.chunk)
            b.close(i)
    with b.phase("shard_build"), b.loop("k", P.shards) as k:
        d = b.let("d", tmp.object(k=k, part="rows"))
        i = b.let("i", tmp.object(k=k, part="ids"))
        b.open(d, "WRONLY|CREAT|TRUNC")
        b.open(i, "RDONLY")
        b.read(i, P.stream)
        b.read(i, P.big, repeat=ceil_div(ids_bytes - P.stream, P.big))
        b.close(i)
        load_pass(b, base_f)                           # the whole base, for this shard's rows
        put(b, d, shard_bytes, P.chunk)
        b.close(d)
        b.open(d, "RDONLY")
        b.read(d, P.stream, repeat=ceil_div(shard_bytes, P.stream))
        b.close(d)
        b.compute(P.build_time)
        g = b.let("g", tmp.object(k=k, part="graph"))
        b.open(g, "WRONLY|CREAT|TRUNC")
        put(b, g, P.shard_index_bytes, P.chunk)
        b.close(g)
        r = b.let("r", tmp.object(k=k, part="graph.data"))
        b.open(r, "WRONLY|CREAT|TRUNC")                # the rows again, beside the graph
        put(b, r, shard_bytes, P.chunk)
        b.close(r)
        b.unlink(d)
    with b.phase("merge"):
        with b.loop("k", P.shards) as k:
            i = b.let("i", tmp.object(k=k, part="ids"))
            b.open(i, "RDONLY")
            b.read(i, P.stream)
            b.read(i, P.big, repeat=ceil_div(ids_bytes - P.stream, P.big))
            b.close(i)
            g = b.let("g", tmp.object(k=k, part="graph"))
            b.open(g, "RDONLY")
            b.read(g, P.big, repeat=ceil_div(P.shard_index_bytes, P.big))       # one read
            b.close(g)
        with b.loop("k", P.shards) as k:
            b.unlink(tmp.object(k=k, part="ids"))
            b.unlink(tmp.object(k=k, part="graph"))
            b.unlink(tmp.object(k=k, part="graph.data"))
        m = b.let("m", out.object(name="mem.index"))
        b.open(m, "WRONLY|CREAT|TRUNC")
        put(b, m, P.index_bytes, P.big)                # one write
        b.close(m)
    with b.phase("layout"):
        block_pass(b, base_f)                          # the full vectors
        m = b.let("m", out.object(name="mem.index"))
        b.open(m, "RDONLY")
        b.read(m, P.stream, repeat=ceil_div(P.index_bytes, P.stream))
        b.close(m)
        x = b.let("x", out.object(name="disk.index"))
        b.open(x, "RDWR|CREAT|TRUNC")
        put(b, x, sectors * P.sector, P.cache + P.sector)
        b.close(x)
    with b.phase("sample"):
        block_pass(b, base_f)
        sd = b.let("sd", out.object(name="sample_data.bin"))
        b.open(sd, "WRONLY|CREAT|TRUNC")
        put(b, sd, P.sample_rows * row + 8, P.chunk)
        b.close(sd)
        si = b.let("si", out.object(name="sample_ids.bin"))
        b.open(si, "WRONLY|CREAT|TRUNC")
        put(b, si, P.sample_rows * 4 + 8, P.chunk)
        b.close(si)

if __name__ == "__main__":
    w.write()
