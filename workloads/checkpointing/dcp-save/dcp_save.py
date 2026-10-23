"""ABSTRACTS.md §3: checkpoint write (torch.distributed.checkpoint shape) inside a minimal
training loop, with the read-back phase.

Exercises: a namespace with size as_written and the same-handle until_eof rule (§9.7, the
optional readback),
parameter arrays, mkdir/rename/fsync, expect on mkdir, a conditional on the actor id (§9.1).
"""
from aeiou import *

w = Workload("dcp_save",
             doc="Every rank writes its shard file per checkpoint step; rank 0 writes .metadata through a temp file and rename.")
P = w.P
# the job's and the model's on its accelerator: a case gives them (DESIGN_REVIEW.md §3.78 points 1, 4)
w.param("steps", kind=int, unit="count")
w.param("step_time", kind=int, unit="ns", doc="compute per step: the model's on its accelerator")
w.param("ckpt_every", kind=int, unit="count")
w.param("items", kind=int, unit="count",
        doc="write items per rank: the model's parameter table (about 900 for an 8B model under FSDP2)")
w.param("item_bytes", kind=list[int], unit="bytes",
        doc="per-item bytes on each rank, 1/G of each tensor: a case holds at the GPU counts it declares")
w.param("meta_bytes", kind=int, unit="bytes",
        doc=".metadata size, measured per job: the global plan's pickle, which grows with the items, ranks, and names "
            "(traced 2,602 bytes for 6 items on 2 ranks)")
# the library's (torch 2.14's torch.save framing)
w.param("hdr", 704, unit="bytes", doc="per item, before the storage: zip local headers, data.pkl, padding to 64 (traced 2026-10-01, torch 2.14)")
w.param("trailer", 873, unit="bytes", doc="per item, after the storage: version, byteorder, central directory, EOCD (traced 2026-10-01); less the storage's length mod 64, since the record after it is aligned")
# the storage's, as the application reads it (point 11)
w.param("buf", st_blksize(of="ckpt", fallback=1 * MiB), unit="bytes",
        doc="Python BufferedWriter size = st_blksize of the shard file; an item above it is written directly, one at or below it is coalesced with its header and trailer")
w.param("xfer", st_blksize(of="ckpt", fallback=1 * MiB), unit="bytes", doc="the readback's BufferedReader request size")
w.param("readback", False, doc="read each shard back on the writing node after the final barrier; off by default, "
                                "the restore is the separate dcp_load run on other nodes (PROJECT_BRIEF.md §8)")

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
                        gpu.write(c, P.trailer - P.item_bytes[t] % 64)
                    with gpu.otherwise():
                        gpu.write(c, P.hdr + P.item_bytes[t] + P.trailer - P.item_bytes[t] % 64)
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
