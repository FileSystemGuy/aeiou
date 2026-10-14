"""ABSTRACTS.md §2: large-sample training (one large .npz file per sample, unet3d shape).

`np.load(path)["x"]` as upstream DLIO's reader (argonne-lcf) issues it (traced 2026-10-01, `workloads/input-pipeline/map/npy-per-sample`):
a buffer fill at offset 0 for the magic, the zip tail (EOCD, the zip64 locator probe, the central
directory), then the archive front to back in buffer fills, with one `tell` per NumPy chunk. The
`x` member is the whole archive but for a few hundred bytes of `y` and directory at the end.
"""
from aeiou import *

w = Workload("npy_per_sample",
             doc="np.load of one ~140 MiB .npz per sample through a PyTorch DataLoader; magic, zip tail, then the archive front to back.")
P = w.P
w.param("batch", 7, unit="count")
w.param("workers", 4, unit="count")
w.param("prefetch", 2, unit="count")
w.param("steps", 500, unit="count")
w.param("sync_every", 500, unit="count")
w.param("step_time", 323 * ms, unit="ns", doc="[measure]")
w.param("hdr_read", 1 * MiB, unit="bytes", doc="Python BufferedReader request = st_blksize; 1 MiB on NFS with rsize 1 MiB (traced 2026-10-01)")
w.param("xfer", 1 * MiB, unit="bytes", doc="buffer fill under the member read: the same BufferedReader, so st_blksize (traced 2026-10-01)")
w.param("np_chunk", 256 * KiB, unit="bytes", doc="NumPy's read chunk from a zip member (numpy.lib.format BUFFER_SIZE); one tell per chunk")
w.param("cd_len", 102, unit="bytes", doc="central directory bytes: two entries, \"x.npy\" and \"y.npy\" (traced 2026-10-01)")
w.param("framing", 498, unit="bytes", doc="archive bytes that are not x's data: x's local header and npy header (183), the y member (191), central directory (102), EOCD (22); np.savez(x=, y=[0]) (traced 2026-10-01)")
w.param("walk", True, doc="the startup listing: the training script's glob over the corpus, one scandir per directory (traced 2026-10-02); on by default (`enumerate` until 2026-10-13)")
w.param("inventory", False, doc="the startup read of the corpus's inventory, every process, in place of the glob at scale (DESIGN_REVIEW.md §3.73)")
w.param("inv_entry", 27, unit="bytes", doc="[measure] inventory bytes per file: the name below the root (`00000/sample_000000000.npz`) and a newline")
w.param("inv_read", 1 * MiB, unit="bytes", doc="[measure] the inventory reader's request; Python's BufferedReader asks st_blksize")
w.param("samples", 50_000, unit="count", doc="[config] corpus size; sized to the dataset rule (PROJECT_BRIEF.md §5)")
w.param("files_per_directory", 10_000, unit="count", doc="[config] files a directory holds, {id div files_per_directory} (contract 0.10); 10,000 as kit/mkcorpus.py writes them")
w.param("sample_mean", 140 * MiB, unit="bytes", doc="[measure]")
w.param("sample_sd", 4 * MiB, unit="bytes", doc="[measure]")
EOCD = 22      # end-of-central-directory record
LOCATOR = 20   # zip64 end-of-central-directory locator, probed just before the EOCD

train = w.dataset("train", pattern="train/{id div files_per_directory:05}/sample_{id:09}.npz", layout="whole", access="map",
                  count=P.samples, size=normal(P.sample_mean, P.sample_sd, min=1 * MiB),
                  seed=0x5eed_da7b,
                  inventory={"name": "inventory.txt", "entry": P.inv_entry})

per_fill = P.xfer // P.np_chunk


def chunks(f):
    """NumPy chunks in the `x` member: its data is the archive less the framing."""
    return ceil_div(f.size - P.framing, P.np_chunk)


with w.actor("gpu") as gpu:
    with gpu.when(P.walk), gpu.phase("walk"):               # glob("…/*/*.npz"): no stat, unlike ImageFolder's walk
        with gpu.loop("d", train.dirs) as d:
            dh = gpu.let("dh", train.dir(d))
            gpu.open(dh, "RDONLY|CLOEXEC|DIRECTORY")                     # the real call adds O_NONBLOCK, which a directory ignores
            gpu.fstat(dh)                                                # os.scandir
            gpu.readdir(dh)
            gpu.close(dh)

    with gpu.when(P.inventory), gpu.phase("inventory"):                # [measure] a Python open and read to EOF
        inv = gpu.let("inv", train.inventory())
        gpu.open(inv, "RDONLY|CLOEXEC")
        gpu.fstat(inv)
        gpu.ioctl(inv, "TCGETS", expect=["ENOTTY"])
        gpu.lseek(inv, 0, "CUR")
        gpu.read(inv, P.inv_read, repeat="until_eof")
        gpu.close(inv)

    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch, batches=P.steps) as worker:
        with worker.loop("j", P.batch):
            f = worker.let("f", train.consume())
            worker.open(f, "RDONLY|CLOEXEC")
            worker.fstat(f)
            worker.ioctl(f, "TCGETS", expect=["ENOTTY"])
            worker.lseek(f, 0, "CUR")
            worker.read(f, P.hdr_read)                                   # np.load: the magic, one buffer fill at 0
            worker.lseek(f, 0, "END")                                    # zipfile._EndRecData
            worker.lseek(f, 0, "CUR")
            worker.lseek(f, -EOCD, "END")
            worker.read(f, P.hdr_read)                                   # EOCD record: 22 bytes, short
            worker.lseek(f, f.size - EOCD - LOCATOR, "SET")
            worker.read(f, P.hdr_read)                                   # zip64 locator probe: 42 bytes, short
            worker.lseek(f, f.size - EOCD - P.cd_len, "SET")
            worker.read(f, P.hdr_read)                                   # central directory to EOF, short
            worker.lseek(f, 0, "SET")                                    # local header of "x", then its data
            with worker.loop("k", chunks(f) // per_fill):                # buffer fills that NumPy drains whole
                worker.read(f, P.xfer)
                with worker.loop("t", per_fill):
                    worker.lseek(f, 0, "CUR")                            # one tell per NumPy chunk
            worker.read(f, P.xfer, repeat=ceil_div(f.size, P.xfer) - chunks(f) // per_fill)   # the rest, to EOF; the last is short
            with worker.loop("t", chunks(f) % per_fill + 4):             # its chunks, and four tells of the header parse
                worker.lseek(f, 0, "CUR")
            worker.close(f)

    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")

if __name__ == "__main__":
    w.write()
