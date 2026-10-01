"""ABSTRACTS.md §4a: checkpoint restore into the same topology (DCP load).

Every rank reads the same small .metadata, then its own shard file; with `replicas` ranks
sharing a shard (replicated state saved once), those ranks read the same file (fan-in).
The namespaces are inputs: a previous `ckpt_write_dcp` run wrote them and left its
`.aeiou-namespace.json`; this run may not modify them (schema/README.md V14). Item offsets are a parameter
table: prefix sums of §3's item sizes plus the per-item header.
"""
from aeiou import *

w = Workload("ckpt_restore",
             doc="DCP load: every rank reads .metadata, then the items of its shard file (fan-in `replicas`).")
P = w.P
w.param("items", 4, unit="count", doc="read items per rank [config]; 900 for an 8B model")
w.param("item_bytes", [16 * MiB, 32 * MiB, 16 * MiB, 8 * MiB], unit="bytes", doc="[config]")
w.param("item_off", [0, 16 * MiB + 64 * KiB, 48 * MiB + 128 * KiB, 64 * MiB + 192 * KiB], unit="bytes",
        doc="[config] prefix sums of item_bytes + tail, as §3 wrote them")
w.param("replicas", 1, unit="count", doc="ranks that read the same shard file (replicated state saved once); 1 when fully sharded")
w.param("meta_bytes", 2 * MiB, unit="bytes", doc="[measure]")
w.param("hdr_read", 1 * MiB, unit="bytes", doc="[verify] st_blksize")
w.param("lh_len", 40, unit="bytes", doc="[verify] zip local header of a torch.save item")
w.param("restore_step", 100, unit="count")

SEED = 0x5eed_da82   # the same namespace seed §3 wrote with, so the bytes agree
ckpt = w.namespace("ckpt", pattern="ckpt/step_{step:06}/__{rank}_0.distcp", fields={"step": int, "rank": int},
                   size=P.item_off[P.items - 1] + P.lh_len + P.item_bytes[P.items - 1], seed=SEED, input=True)
ckpt_meta = w.namespace("ckpt_meta", pattern="ckpt/step_{step:06}/{name}", fields={"step": int, "name": str},
                        size=P.meta_bytes, seed=SEED, input=True)

with w.actor("gpu") as gpu:
    with gpu.phase("restore"):
        m = gpu.let("m", ckpt_meta.object(step=P.restore_step, name=".metadata"))
        gpu.open(m, "RDONLY")
        gpu.fstat(m)
        gpu.read(m, P.hdr_read, repeat="until_eof")
        gpu.close(m)
        c = gpu.let("c", ckpt.object(step=P.restore_step, rank=gpu_id // P.replicas))   # fan-in
        gpu.open(c, "RDONLY")
        gpu.fstat(c)
        gpu.ioctl(c, "TCGETS", expect=["ENOTTY"])
        gpu.lseek(c, 0, "CUR")
        with gpu.loop("t", P.items) as t:
            gpu.read(c, P.hdr_read, offset=P.item_off[t])                            # zip tail of the item
            gpu.read(c, P.item_bytes[t], offset=P.item_off[t] + P.lh_len)            # storage: one large read
        gpu.close(c)
        gpu.barrier("global")

if __name__ == "__main__":
    w.write()
