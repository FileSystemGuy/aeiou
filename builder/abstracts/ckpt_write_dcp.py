"""ABSTRACTS.md §3: checkpoint write (torch.distributed.checkpoint shape) inside a minimal
training loop, with the read-back phase.

Exercises: a namespace with size as_written and the same-handle until_eof rule (§9.7, the
optional readback),
parameter arrays, mkdir/rename/fsync, expect on mkdir, a conditional on the actor id (§9.1).
"""
from aeiou import *

w = Workload("ckpt_write_dcp",
             doc="Every rank writes its shard file per checkpoint step; rank 0 writes .metadata through a temp file and rename.")
P = w.P
w.param("steps", 500, unit="count")
w.param("step_time", 323 * ms, unit="ns")
w.param("ckpt_every", 100, unit="count")
w.param("items", 4, unit="count",
        doc="write items per rank [config]; 900 for an 8B model, 4 here to keep the example short")
w.param("item_bytes", [16 * MiB, 32 * MiB, 16 * MiB, 8 * MiB], unit="bytes",
        doc="per-item bytes, 1/G of each tensor [config]")
w.param("hdr", 704, unit="bytes", doc="per item, before the storage: zip local headers, data.pkl, padding to 64 (traced 2026-10-01, torch 2.14)")
w.param("trailer", 873, unit="bytes", doc="per item, after the storage: version, byteorder, central directory, EOCD (traced 2026-10-01)")
w.param("buf", 1 * MiB, unit="bytes", doc="Python BufferedWriter size = st_blksize; an item above it is written directly, one at or below it is coalesced with its header and trailer")
w.param("meta_bytes", 2 * MiB, unit="bytes", doc="[measure] .metadata size; traced 2,602 bytes for 6 items on 2 ranks, about 217 per item and rank")
w.param("xfer", 1 * MiB, unit="bytes")
w.param("readback", False, doc="read each shard back on the writing node after the final barrier; off by default, "
                                "the restore is the separate ckpt_restore run on other nodes (PROJECT_BRIEF.md §8)")

SEED = 0x5eed_da82
ckpt_dir = w.namespace("ckpt_dir", pattern="ckpt/step_{step:06}", fields={"step": int}, size=0, seed=SEED)
ckpt = w.namespace("ckpt", pattern="ckpt/step_{step:06}/__{rank}_0.distcp", fields={"step": int, "rank": int},
                   size="as_written", seed=SEED)          # Σ write lengths of the creating sequence (§9.7)
ckpt_meta = w.namespace("ckpt_meta", pattern="ckpt/step_{step:06}/{name}", fields={"step": int, "name": str},
                        size=P.meta_bytes, seed=SEED)

with w.actor("gpu") as gpu:
    with gpu.loop("step", P.steps) as step:
        gpu.compute(P.step_time)
        with gpu.every(P.ckpt_every):
            gpu.barrier("global")                                   # collective before save
            c = gpu.let("c", ckpt.object(step=step, rank=gpu_id))   # bound above both phases
            with gpu.phase("ckpt_write"):
                gpu.mkdir(ckpt_dir.object(step=step), expect=["EEXIST"])
                gpu.stat(ckpt_meta.object(step=step, name=".metadata"), expect=["ENOENT"])   # is there a checkpoint here already
                gpu.open(c, "WRONLY|CREAT|TRUNC|CLOEXEC")
                gpu.fstat(c)
                gpu.ioctl(c, "TCGETS", expect=["ENOTTY"])
                gpu.lseek(c, 0, "CUR")
                with gpu.loop("t", P.items) as t:                   # one torch.save stream per write item
                    gpu.lseek(c, 0, "CUR")                          # tell: the item's offset
                    with gpu.when(P.item_bytes[t] > P.buf):
                        gpu.write(c, P.hdr)
                        gpu.write(c, P.item_bytes[t])               # the storage, past the buffer in one call
                        gpu.write(c, P.trailer)
                    with gpu.otherwise():
                        gpu.write(c, P.hdr + P.item_bytes[t] + P.trailer)
                    gpu.lseek(c, 0, "CUR")                          # tell: the item's length
                gpu.fsync(c)
                gpu.close(c)
                gpu.barrier("global")                               # gather WriteResults
                with gpu.when(gpu_id == 0):
                    m = gpu.let("m", ckpt_meta.object(step=step, name=".metadata.tmp"))
                    gpu.open(m, "WRONLY|CREAT|TRUNC|CLOEXEC")
                    gpu.fstat(m)
                    gpu.ioctl(m, "TCGETS", expect=["ENOTTY"])
                    gpu.lseek(m, 0, "CUR")
                    gpu.write(m, P.meta_bytes)
                    gpu.fsync(m)
                    gpu.close(m)
                    gpu.stat(ckpt_meta.object(step=step, name=".metadata"), expect=["ENOENT"])
                    gpu.rename(m, ckpt_meta.object(step=step, name=".metadata"))
                gpu.barrier("global")
            with gpu.when(P.readback), gpu.phase("ckpt_readback"):  # same position, same handle; a correctness run only
                gpu.open(c, "RDONLY")
                gpu.read(c, P.xfer, repeat="until_eof")
                gpu.close(c)
                gpu.barrier("global")

if __name__ == "__main__":
    w.write()
