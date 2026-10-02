"""ABSTRACTS.md §4a: checkpoint restore into the same topology (DCP load).

Every rank reads the same small .metadata, then its own shard file; with `replicas` ranks
sharing a shard (replicated state saved once), those ranks read the same file (fan-in).
The namespaces are inputs: a previous `ckpt_write_dcp` run wrote them and left its
`.aeiou-namespace.json`; this run may not modify them (schema/README.md V14). Item offsets are a parameter
table: prefix sums of §3's item sizes plus the per-item header and trailer.

The item loop is the trace of `torch.distributed.checkpoint.load` (2026-10-01, torch 2.14,
builder/traces/ckpt_restore): one `torch.load` per item on a view of the shard file, through one
Python BufferedReader of `buf` bytes, the items taken in the sorted order of their names
(`read_order`), which is not the order they lie in the file. Every `tell` is an `lseek(0, CUR)`,
26 per item. A seek to a target inside the buffer costs nothing; one outside it is an `lseek` and
empties the buffer, and the next read fills it with `buf` bytes from there (short at EOF). So what
an item costs depends on where the buffer was left: `bst` carries its start from item to item
(`bst @ k-1`), and the three tests on it (`head_in`, `end_in`, `far`) decide which seeks and fills
are issued. The storage's part past the buffer is one direct read, which leaves the buffer empty.
"""
from aeiou import *

w = Workload("ckpt_restore",
             doc="DCP load: every rank reads .metadata, then the items of its shard file (fan-in `replicas`).")
P = w.P
w.param("items", 4, unit="count", doc="read items per rank [config]; 900 for an 8B model")
w.param("item_bytes", [16 * MiB, 32 * MiB, 16 * MiB, 8 * MiB], unit="bytes", doc="[config] storage bytes of the item at each place in the file")
w.param("item_off", [0, 16 * MiB + 1577, 48 * MiB + 2 * 1577, 64 * MiB + 3 * 1577], unit="bytes",
        doc="[config] the item's offset in the file: prefix sums of hdr + item_bytes + trailer - item_bytes mod 64, as §3 writes them")
w.param("read_order", [0, 1, 2, 3], unit="count",
        doc="[config] the place in the file of the k-th item read: the items sorted by name (layer.10 before layer.2), "
            "against the writer's state-dict order")
w.param("replicas", 1, unit="count", doc="ranks that read the same shard file (replicated state saved once); 1 when fully sharded")
w.param("meta_bytes", 2 * MiB, unit="bytes", doc="[measure] .metadata size; traced 2,606 bytes for 6 items on 2 ranks")
w.param("hdr", 704, unit="bytes", doc="per item, before the storage, as §3 writes it (traced 2026-10-01, torch 2.14)")
w.param("trailer", 873, unit="bytes", doc="per item, after the storage, less the storage's length mod 64 (the next record is aligned), as §3 writes it")
w.param("buf", 1 * MiB, unit="bytes", doc="Python BufferedReader size = st_blksize; every buffer fill asks for this much")
w.param("rec2", 236, unit="bytes", doc="offset in an item of its second zip record, where the reader goes after the central directory (traced)")
w.param("eocd_scan", 4096, unit="bytes", doc="the zip reader looks for the end record in the last 4096 bytes of an item")
w.param("tail_back", [22, 42, 98, 561, 727, 857], unit="bytes",
        doc="the item at the end of the file only: a fill there stops at EOF and is used up by the read that caused it, so each "
            "later step of the zip reader (end record, zip64 locator and record, central directory, the two records after the "
            "storage) is a fill of its own, this many bytes before the end (traced)")
w.param("restore_step", 100, unit="count")

SEED = 0x5eed_da82   # the same namespace seed §3 wrote with, so the bytes agree
LAST = P.items - 1                                   # the item at the end of the file
FSIZE = P.item_off[LAST] + P.hdr + P.item_bytes[LAST] + P.trailer - P.item_bytes[LAST] % 64
ckpt = w.namespace("ckpt", pattern="ckpt/step_{step:06}/__{rank}_0.distcp", fields={"step": int, "rank": int},
                   size=FSIZE, seed=SEED, input=True)
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
        with gpu.loop("k", P.items) as k:                                           # one torch.load per item, in name order
            t = gpu.let("t", P.read_order[k])
            o = gpu.let("o", P.item_off[t])
            n = gpu.let("n", P.hdr + P.item_bytes[t] + P.trailer - P.item_bytes[t] % 64)   # the item: a zip archive of one storage
            pb = gpu.let("pb", when(k == 0, -1, gpu.ref("bst").at(k - 1)))          # where the buffer starts; -1: it is empty
            head_in = gpu.let("head_in", (pb >= 0) & (pb <= o) & (o < min_(pb + P.buf, FSIZE)))
            be = gpu.let("be", min_(when(head_in, pb, o) + P.buf, FSIZE))           # the buffer's end once the head is in it
            end_in = gpu.let("end_in", o + n < be)                                  # the whole item is in the buffer
            far = gpu.let("far", n > P.buf)                                         # no fill at the item holds its end record
            last = gpu.let("last", o + n == FSIZE)
            gpu.let("bst", when(~end_in & far & (P.hdr + P.item_bytes[t] > P.buf), -1, when(head_in & end_in, pb, o)))
            with gpu.when(~head_in):
                gpu.lseek(c, o, "SET")
            gpu.lseek(c, 0, "CUR")
            with gpu.when(~head_in):
                gpu.lseek(c, o, "SET")
            with gpu.loop("i", 2):
                gpu.lseek(c, 0, "CUR")
            with gpu.when(~head_in):
                gpu.read(c, P.buf)                                                  # the magic: a fill at the item
            with gpu.loop("i", 2):
                gpu.lseek(c, 0, "CUR")
            with gpu.when(~end_in):                                                 # the view's length: a seek to its end
                with gpu.when(o + n > be):
                    gpu.lseek(c, o + n, "SET")                                      # (free when the end is the buffer's own, at EOF)
                gpu.lseek(c, 0, "CUR")
                gpu.lseek(c, o, "SET")                                              # the buffer is gone
                gpu.lseek(c, o, "SET")
                gpu.lseek(c, 0, "CUR")
                gpu.read(c, P.buf)                                                  # the magic again
                with gpu.when(far):
                    gpu.lseek(c, o + n - P.eocd_scan, "SET")
                gpu.lseek(c, 0, "CUR")
                with gpu.when(far):
                    gpu.read(c, P.buf)                                              # end record and central directory, and what follows the item
                with gpu.when(last), gpu.loop("b", P.tail_back.len) as b:           # at EOF each step back is a fill
                    gpu.lseek(c, o + n - P.tail_back[b], "SET")
                    gpu.read(c, P.buf)
                with gpu.loop("i", 8):
                    gpu.lseek(c, 0, "CUR")
                with gpu.when(far | last):
                    gpu.lseek(c, o + P.rec2, "SET")
                gpu.lseek(c, 0, "CUR")
                with gpu.when(far | last):
                    gpu.read(c, P.buf)                                              # the records before the storage
                with gpu.loop("i", 5):
                    gpu.lseek(c, 0, "CUR")
                with gpu.when(far | last):
                    gpu.lseek(c, o, "SET")
                gpu.lseek(c, 0, "CUR")
                with gpu.when(far | last):
                    gpu.read(c, P.buf)                                              # data.pkl, and the storage's first part
                with gpu.loop("i", 3):
                    gpu.lseek(c, 0, "CUR")
                with gpu.when(far & (P.hdr + P.item_bytes[t] > P.buf)):
                    gpu.read(c, P.hdr + P.item_bytes[t] - P.buf)                    # the rest of the storage: one read past the buffer
            with gpu.otherwise(), gpu.loop("i", 21):                                # every seek lands in the buffer: only the tells
                gpu.lseek(c, 0, "CUR")
        gpu.close(c)
        gpu.barrier("global")

if __name__ == "__main__":
    w.write()
