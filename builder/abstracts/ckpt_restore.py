"""ABSTRACTS.md §4a: checkpoint restore into the same topology (DCP load).

Every rank reads the same small .metadata, then its own shard file; with `replicas` ranks
sharing a shard (replicated state saved once), those ranks read the same file (fan-in).
The namespaces are inputs: a previous `ckpt_write_dcp` run wrote them and left its
`.aeiou-namespace.json`; this run may not modify them (schema/README.md V14). Item offsets are a parameter
table: prefix sums of §3's item sizes plus the per-item header and trailer.

The item loop is the trace of `torch.distributed.checkpoint.load` (2026-10-01, torch 2.14,
builder/traces/ckpt_restore): one `torch.load` per item on a view of the shard file, through one
Python BufferedReader of `buf` bytes. Every read is a buffer fill of `buf` at the last seek target
(short at EOF) except the storage, whose part past the buffer is one direct read; every `tell` is an
`lseek(0, CUR)`; a seek inside the buffer costs nothing, which is why an item smaller than the
buffer is one fill, and none when the item before it was small too. The last item of a file is
the exception: a fill that reaches EOF is used up by the read that caused it, so the zip reader's
steps backwards from the end are fills of their own.
"""
from aeiou import *

w = Workload("ckpt_restore",
             doc="DCP load: every rank reads .metadata, then the items of its shard file (fan-in `replicas`).")
P = w.P
w.param("items", 4, unit="count", doc="read items per rank [config]; 900 for an 8B model")
w.param("item_bytes", [16 * MiB, 32 * MiB, 16 * MiB, 8 * MiB], unit="bytes", doc="[config]")
w.param("item_off", [0, 16 * MiB + 1577, 48 * MiB + 2 * 1577, 64 * MiB + 3 * 1577], unit="bytes",
        doc="[config] prefix sums of hdr + item_bytes + trailer, as §3 writes them")
w.param("replicas", 1, unit="count", doc="ranks that read the same shard file (replicated state saved once); 1 when fully sharded")
w.param("meta_bytes", 2 * MiB, unit="bytes", doc="[measure] .metadata size; traced 2,606 bytes for 6 items on 2 ranks")
w.param("hdr", 704, unit="bytes", doc="per item, before the storage, as §3 writes it (traced 2026-10-01, torch 2.14)")
w.param("trailer", 873, unit="bytes", doc="per item, after the storage, as §3 writes it")
w.param("buf", 1 * MiB, unit="bytes", doc="Python BufferedReader size = st_blksize; every buffer fill asks for this much")
w.param("rec2", 236, unit="bytes", doc="offset in an item of its second zip record, where the reader goes after the central directory (traced)")
w.param("eocd_scan", 4096, unit="bytes", doc="the zip reader looks for the end record in the last 4096 bytes of an item")
w.param("tail_back", [22, 42, 98, 561, 727, 857], unit="bytes",
        doc="the last item of a file only: the fill at the scan offset stops at EOF, so each later step of the zip reader "
            "(end record, zip64 locator and record, central directory, the two records after the storage) is a fill of its "
            "own, this many bytes before the end (traced)")
w.param("restore_step", 100, unit="count")

SEED = 0x5eed_da82   # the same namespace seed §3 wrote with, so the bytes agree
ckpt = w.namespace("ckpt", pattern="ckpt/step_{step:06}/__{rank}_0.distcp", fields={"step": int, "rank": int},
                   size=P.item_off[P.items - 1] + P.hdr + P.item_bytes[P.items - 1] + P.trailer, seed=SEED, input=True)
ckpt_meta = w.namespace("ckpt_meta", pattern="ckpt/step_{step:06}/{name}", fields={"step": int, "name": str},
                        size=P.meta_bytes, seed=SEED, input=True)

with w.actor("gpu") as gpu:
    with gpu.phase("restore"):
        m = gpu.let("m", ckpt_meta.object(step=P.restore_step, name=".metadata"))
        gpu.open(m, "RDONLY|CLOEXEC")
        gpu.fstat(m)
        gpu.ioctl(m, "TCGETS", expect=["ENOTTY"])
        gpu.lseek(m, 0, "CUR")
        gpu.read(m, P.buf, repeat=ceil_div(P.meta_bytes, P.buf))                    # pickle.load: fills, the last short; no EOF read
        gpu.close(m)
        c = gpu.let("c", ckpt.object(step=P.restore_step, rank=gpu_id // P.replicas))   # fan-in
        gpu.open(c, "RDONLY|CLOEXEC")
        gpu.fstat(c)
        gpu.ioctl(c, "TCGETS", expect=["ENOTTY"])
        gpu.lseek(c, 0, "CUR")
        with gpu.loop("t", P.items) as t:                                           # one torch.load per item
            o = gpu.let("o", P.item_off[t])
            n = gpu.let("n", P.hdr + P.item_bytes[t] + P.trailer)                    # the item: a zip archive of one storage
            held = gpu.let("held", (t > 0) & (P.hdr + P.item_bytes[max_(t - 1, 0)] + P.trailer < P.buf))   # the fill of a small item before holds this head
            with gpu.when(~held):
                gpu.lseek(c, o, "SET")
            gpu.lseek(c, 0, "CUR")
            with gpu.when(~held):
                gpu.lseek(c, o, "SET")
            with gpu.loop("k", 2):
                gpu.lseek(c, 0, "CUR")
            with gpu.when(~held):
                gpu.read(c, P.buf)                                                  # the magic: a fill at the item
            with gpu.loop("k", 2):
                gpu.lseek(c, 0, "CUR")
            far = gpu.let("far", n >= P.buf)                                        # the item's end is outside the buffer
            with gpu.when(far | (t == P.items - 1)):
                with gpu.when(far):
                    gpu.lseek(c, o + n, "SET")                                      # its length; the buffer is dropped
                gpu.lseek(c, 0, "CUR")
                gpu.lseek(c, o, "SET")
                gpu.lseek(c, o, "SET")
                gpu.lseek(c, 0, "CUR")
                gpu.read(c, P.buf)                                                  # the magic again
                with gpu.when(far):
                    gpu.lseek(c, o + n - P.eocd_scan, "SET")
                gpu.lseek(c, 0, "CUR")
                with gpu.when(far):
                    gpu.read(c, P.buf)                                              # end record and central directory, and the next item's head
                with gpu.when(t == P.items - 1), gpu.loop("b", P.tail_back.len) as b:   # at EOF a fill holds nothing afterwards
                    gpu.lseek(c, o + n - P.tail_back[b], "SET")
                    gpu.read(c, P.buf)
                with gpu.loop("k", 8):
                    gpu.lseek(c, 0, "CUR")
                gpu.lseek(c, o + P.rec2, "SET")
                gpu.lseek(c, 0, "CUR")
                gpu.read(c, P.buf)                                                  # the records before the storage
                with gpu.loop("k", 5):
                    gpu.lseek(c, 0, "CUR")
                gpu.lseek(c, o, "SET")
                gpu.lseek(c, 0, "CUR")
                gpu.read(c, P.buf)                                                  # data.pkl, and the storage's first part
                with gpu.loop("k", 3):
                    gpu.lseek(c, 0, "CUR")
                with gpu.when(P.hdr + P.item_bytes[t] > P.buf):
                    gpu.read(c, P.hdr + P.item_bytes[t] - P.buf)                    # the rest of the storage: one read past the buffer
            with gpu.otherwise(), gpu.loop("k", 21):                                # the whole item is in the buffer: only the tells
                gpu.lseek(c, 0, "CUR")
        gpu.close(c)
        gpu.barrier("global")

if __name__ == "__main__":
    w.write()
