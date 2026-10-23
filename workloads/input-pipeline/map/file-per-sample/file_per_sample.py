"""ABSTRACTS.md §1: small-file training (one sample per file).

PyTorch DataLoader over an ImageFolder-style corpus; W workers, B files per batch, in-order
delivery; an optional startup directory walk (§9.8), or in its place a read of the corpus's
inventory (DESIGN_REVIEW.md §3.73: no traced job reads one yet, so its shape is `[measure]`).
"""
from aeiou import *

w = Workload("file_per_sample",
             doc="PyTorch DataLoader over an ImageFolder-style corpus; W workers, B files per batch, in-order delivery.")

P = w.P
# the job and the model on a GPU are a use case's (DESIGN_REVIEW.md §3.78 point 1): required here, the
# placeholders the defaults held until 2026-10-22 in the unsettled case cases/jpeg-50m
w.param("batch", kind=int, unit="count", doc="samples per step: the model's on its accelerator, whose memory sets it")
w.param("workers", kind=int, unit="count", doc="DataLoader worker processes per GPU: the job's")
w.param("prefetch", kind=int, unit="count", doc="batches each worker keeps ahead (prefetch_factor): the job's")
w.param("steps", kind=int, unit="count", doc="steps of the run: the job's")
w.param("sync_every", 1, unit="count", doc="synchronous data parallel (DDP, FSDP): every step waits for the slowest GPU (DESIGN_REVIEW.md §3.71 point 4; 500, the brief's reference workload, until 2026-10-14)")
w.param("step_time", kind=int, unit="ns", doc="compute per step: the model's on its accelerator")
w.param("hdr_read", st_blksize(of="train", fallback=1 * MiB), unit="bytes", doc="Python BufferedReader request = st_blksize; 1 MiB on NFS with rsize 1 MiB (traced 2026-10-01)")
w.param("walk", True, doc="the startup directory walk, as ImageFolder does it in every process (ABSTRACTS.md §9.8); on by default since 2026-10-02 (`enumerate` until 2026-10-13)")
w.param("inventory", False, doc="the startup read of the corpus's inventory, every process, in place of the walk at scale (DESIGN_REVIEW.md §3.73)")
w.param("inv_entry", 24, unit="bytes", doc="[measure] inventory bytes per file: the name below the root (`00000/img_000000000.jpg`) and a newline")
w.param("inv_read", st_blksize(of="train", fallback=1 * MiB), unit="bytes", doc="[measure] the inventory reader's request; Python's BufferedReader asks st_blksize")
w.param("samples", kind=int, scale=True, unit="count", doc="corpus size: the submission's, sized to the dataset rule (PROJECT_BRIEF.md §5)")
w.param("files_per_directory", 1_300, unit="count", doc="[config] files a directory holds, {id div files_per_directory} (contract 0.10); 1,300 is ImageNet-1k's most images in a class, as kit/mkcorpus.py writes them")
w.param("sample_median", kind=int, unit="bytes", doc="median sample file size: the job's corpus")

train = w.dataset("train", pattern="train/{id div files_per_directory:05}/img_{id:09}.jpg", layout="whole", access="map",
                  count=P.samples,
                  size=lognormal(median=P.sample_median, sigma=0.45),   # [measure] corpus; sigma is a literal in the contract
                  seed=0x5eed_da7a,
                  inventory={"name": "inventory.txt", "entry": P.inv_entry})

with w.actor("gpu") as gpu:
    with gpu.when(P.walk), gpu.phase("walk"):
        with gpu.loop("d", train.dirs) as d:
            dh = gpu.let("dh", train.dir(d))
            gpu.stat(dh)                                  # make_dataset: os.path.isdir(class dir)
            gpu.open(dh, "RDONLY|CLOEXEC|DIRECTORY")         # the real call adds O_NONBLOCK, which a directory ignores
            gpu.fstat(dh)                                 # os.scandir
            gpu.readdir(dh)
            gpu.close(dh)

    with gpu.when(P.inventory), gpu.phase("inventory"):           # [measure] a Python open and read to EOF
        inv = gpu.let("inv", train.inventory())
        gpu.open(inv, "RDONLY|CLOEXEC")
        gpu.fstat(inv)
        gpu.ioctl(inv, "TCGETS", expect=["ENOTTY"])
        gpu.lseek(inv, 0, "CUR")
        gpu.read(inv, P.inv_read, repeat="until_eof")
        gpu.close(inv)

    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch, batches=P.steps) as worker:
        with worker.loop("j", P.batch):
            f = worker.let("f", train.consume())          # position = gpu + G·(b·batch + j)
            worker.open(f, "RDONLY|CLOEXEC")
            worker.fstat(f)
            worker.ioctl(f, "TCGETS", expect=["ENOTTY"])
            worker.lseek(f, 0, "CUR")
            worker.lseek(f, 0, "SET")                     # PIL Image.open: fp.seek(0)
            worker.read(f, P.hdr_read, repeat="until_eof")
            worker.close(f)

    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")

if __name__ == "__main__":
    w.write()
