"""ABSTRACTS.md §2: large-sample training (one large .npz file per sample, unet3d shape).

The read pattern is not sequential from offset 0: the zip tail (EOCD) first, then the central
directory, then each member's local header and its data in fixed chunks. The archive layout is
fixed by the generator, so member offsets are formulas over size(f) (§9.4); `member_off` and
`member_len` below are the builder-side helpers the paper form names.
"""
from mlps_abstract import *

w = Workload("train_large_samples",
             doc="np.load of one ~140 MiB .npz per sample through a PyTorch DataLoader; tail read, central directory, then members.")
P = w.P
w.param("batch", 7, unit="count")
w.param("workers", 4, unit="count")
w.param("prefetch", 2, unit="count")
w.param("steps", 500, unit="count")
w.param("sync_every", 500, unit="count")
w.param("step_time", 323 * ms, unit="ns", doc="[measure]")
w.param("hdr_read", 1 * MiB, unit="bytes", doc="[verify] st_blksize")
w.param("xfer", 1 * MiB, unit="bytes", doc="[verify] member read chunk (numpy 256 KiB, BufferedReader max(256 KiB, st_blksize))")
w.param("members", 2, unit="count", doc="[verify] \"x\" and \"y\" in the npz")
w.param("cd_len", 200, unit="bytes", doc="[verify] central directory bytes")
w.param("lh_len", 40, unit="bytes", doc="[verify] local header + name bytes")
EOCD = 22   # end-of-central-directory record

train = w.dataset("train", pattern="train/{id div 10000:05}/sample_{id:09}.npz",
                  count=50_000, size=normal(140 * MiB, 4 * MiB, min=1 * MiB),   # [measure]
                  seed=0x5eed_da7b, access="map")


def member_len(f):
    """Bytes of each member: the archive minus tail, central directory, and local headers,
    split evenly (the generator writes members of equal size)."""
    return (f.size - EOCD - P.cd_len - P.members * P.lh_len) // P.members


def member_off(f, m):
    """Offset of member m's local header."""
    return m * (P.lh_len + member_len(f))


with w.actor("gpu") as gpu:
    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch, batches=P.steps) as worker:
        with worker.loop("j", P.batch):
            f = worker.let("f", train.consume())
            worker.open(f, "RDONLY|CLOEXEC")
            worker.fstat(f)
            worker.ioctl(f, "TCGETS", expect=["ENOTTY"])
            worker.lseek(f, 0, "CUR")
            worker.read(f, P.hdr_read, offset=f.size - EOCD, repeat="until_eof")        # tail: EOCD record
            worker.read(f, P.hdr_read, offset=f.size - P.cd_len - EOCD)                # central directory
            with worker.loop("m", P.members) as m:
                worker.read(f, P.hdr_read, offset=member_off(f, m))                      # local header (§9.4)
                worker.lseek(f, member_off(f, m) + P.lh_len, "SET")
                worker.read(f, P.xfer, repeat=ceil_div(member_len(f), P.xfer))           # member data, sequential
            worker.close(f)

    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")

if __name__ == "__main__":
    w.write()
