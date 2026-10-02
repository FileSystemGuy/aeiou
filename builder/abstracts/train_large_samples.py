"""ABSTRACTS.md §2: large-sample training (one large .npz file per sample, unet3d shape).

`np.load(path)["x"]` as upstream DLIO's reader (argonne-lcf) issues it (traced 2026-10-01, `builder/traces/train_large_samples`):
a buffer fill at offset 0 for the magic, the zip tail (EOCD, the zip64 locator probe, the central
directory), then the archive front to back in buffer fills, with one `tell` per NumPy chunk. The
`x` member is the whole archive but for a few hundred bytes of `y` and directory at the end.
"""
from aeiou import *

w = Workload("train_large_samples",
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
w.param("files", 50_000, unit="count", doc="[config] corpus size; sized to the dataset rule (PROJECT_BRIEF.md §5)")
w.param("sample_mean", 140 * MiB, unit="bytes", doc="[measure]")
w.param("sample_sd", 4 * MiB, unit="bytes", doc="[measure]")
EOCD = 22      # end-of-central-directory record
LOCATOR = 20   # zip64 end-of-central-directory locator, probed just before the EOCD

train = w.dataset("train", pattern="train/{id div 10000:05}/sample_{id:09}.npz",
                  count=P.files, size=normal(P.sample_mean, P.sample_sd, min=1 * MiB),
                  seed=0x5eed_da7b, access="map")

per_fill = P.xfer // P.np_chunk


def chunks(f):
    """NumPy chunks in the `x` member: its data is the archive less the framing."""
    return ceil_div(f.size - P.framing, P.np_chunk)


with w.actor("gpu") as gpu:
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
