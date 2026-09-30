"""ABSTRACTS.md §1: small-file training (one sample per file).

PyTorch DataLoader over an ImageFolder-style corpus; W workers, B files per batch, in-order
delivery; an optional startup directory walk (§9.8).
"""
from aeiou import *

w = Workload("train_small_files",
             doc="PyTorch DataLoader over an ImageFolder-style corpus; W workers, B files per batch, in-order delivery.")

P = w.P
w.param("batch", 32, unit="count")
w.param("workers", 8, unit="count")
w.param("prefetch", 2, unit="count")
w.param("steps", 500, unit="count")
w.param("sync_every", 500, unit="count", doc="1 for DDP; 500 keeps the brief's reference workload")
w.param("step_time", 105 * ms, unit="ns", doc="[measure] GPU step time on the target accelerator")
w.param("hdr_read", 1 * MiB, unit="bytes", doc="[verify] Python BufferedReader request = st_blksize")
w.param("enumerate", False, doc="include the startup directory walk (ABSTRACTS.md §9.8)")

train = w.dataset("train", pattern="train/{id div 1300:05}/img_{id:09}.jpg",
                  count=50_000_000,
                  size=lognormal(median=110 * KiB, sigma=0.45),   # [measure] corpus
                  seed=0x5eed_da7a, access="map")

with w.actor("gpu") as gpu:
    with gpu.when(P.enumerate), gpu.phase("enumerate"):
        with gpu.loop("d", train.dirs) as d:
            dh = gpu.let("dh", train.dir(d))
            gpu.open(dh, "RDONLY|DIRECTORY")
            gpu.readdir(dh)
            gpu.close(dh)

    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch, batches=P.steps) as worker:
        with worker.loop("j", P.batch):
            f = worker.let("f", train.consume())          # position = gpu + G·(b·batch + j)
            worker.open(f, "RDONLY|CLOEXEC")
            worker.fstat(f)
            worker.ioctl(f, "TCGETS", expect=["ENOTTY"])
            worker.lseek(f, 0, "CUR")
            worker.read(f, P.hdr_read, repeat="until_eof")
            worker.close(f)

    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")

if __name__ == "__main__":
    w.write()
