"""ABSTRACTS.md §3: checkpoint write (torch.distributed.checkpoint shape) inside a minimal
training loop, with the read-back phase.

Exercises: a namespace with size as_written and the same-handle until_eof rule (§9.7),
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
w.param("tail", 64 * KiB, unit="bytes", doc="[verify] coalesced small records per item")
w.param("meta_bytes", 2 * MiB, unit="bytes", doc="[measure] .metadata size")
w.param("xfer", 1 * MiB, unit="bytes")

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
                gpu.open(c, "WRONLY|CREAT|TRUNC|CLOEXEC")
                gpu.fstat(c)
                gpu.ioctl(c, "TCGETS", expect=["ENOTTY"])
                gpu.lseek(c, 0, "CUR")
                with gpu.loop("t", P.items) as t:
                    gpu.write(c, P.item_bytes[t])
                    gpu.write(c, P.tail)
                gpu.fsync(c)
                gpu.close(c)
                gpu.barrier("global")                               # gather WriteResults
                with gpu.when(gpu_id == 0):
                    m = gpu.let("m", ckpt_meta.object(step=step, name=".metadata.tmp"))
                    gpu.open(m, "WRONLY|CREAT|TRUNC")
                    gpu.write(m, P.meta_bytes)
                    gpu.fsync(m)
                    gpu.close(m)
                    gpu.rename(m, ckpt_meta.object(step=step, name=".metadata"))
                gpu.barrier("global")
            with gpu.phase("ckpt_readback"):                        # the restore shape, same position, same handle
                gpu.open(c, "RDONLY")
                gpu.read(c, P.xfer, repeat="until_eof")
                gpu.close(c)
                gpu.barrier("global")

if __name__ == "__main__":
    w.write()
