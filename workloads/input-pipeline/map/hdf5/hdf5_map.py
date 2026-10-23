"""Map-style training over HDF5 containers: a PyTorch DataLoader whose `Dataset.__getitem__`
opens the container holding sample `i` with h5py, reads row `i` of its one contiguous dataset,
and closes it (the plain map-style HDF5 dataset that survives `fork`; an open-file cache is
the cut). DLIO's HDF5 reader issues the same per-row read. The format class carries what a
trace of h5py shows: the metadata reads at open and libhdf5's sieve buffer on the row read.
"""
from aeiou import *
from aeiou.formats import hdf5

SHAPE = (256, 256, 3)                       # one sample: 196,608 bytes, larger than the 64 KiB sieve

w = Workload("hdf5_map",
             doc="DataLoader over HDF5 containers of fixed-size rows: open, metadata reads, one positioned row read, close per sample.")
P = w.P
# the job and the model on a GPU are a case's (DESIGN_REVIEW.md §3.78 point 1): required here; the
# placeholders the defaults held until 2026-10-22, never measured, are cases/rgb256-rows (untraced)
w.param("batch", kind=int, unit="count", doc="samples per step: the model's on its accelerator, whose memory sets it")
w.param("workers", kind=int, unit="count", doc="DataLoader worker processes per GPU: the job's")
w.param("prefetch", kind=int, unit="count", doc="batches each worker keeps ahead (prefetch_factor): the job's")
w.param("steps", kind=int, unit="count", doc="steps of the run: the job's")
w.param("sync_every", 1, unit="count", doc="synchronous data parallel (DDP, FSDP): every step waits for the slowest GPU (DESIGN_REVIEW.md §3.71 point 4; 500, the brief's reference workload, until 2026-10-14)")
w.param("step_time", kind=int, unit="ns", doc="compute per step: the model's on its accelerator")
w.param("samples_per_file", kind=int, unit="count", doc="samples per container: the job's corpus")
w.param("samples", kind=int, scale=True, unit="count", doc="corpus size: the submission's, sized to the dataset rule (PROJECT_BRIEF.md §5)")

train = w.dataset("train", pattern="train/samples-{id:05}.h5", layout="compound", access="map", count=P.samples, samples_per_file=P.samples_per_file,
                  size=const(SHAPE[0] * SHAPE[1] * SHAPE[2]), seed=0x5eed_da91,
                  format=hdf5(dataset="records", shape=SHAPE, dtype="u1"))

with w.actor("gpu") as gpu:
    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch, batches=P.steps) as worker:
        with worker.loop("j", P.batch):
            s = worker.let("s", train.consume())          # position = gpu + G·(b·batch + j)
            f = worker.let("f", s.container)
            train.format.open_reads(worker, f)
            train.format.read_sample(worker, f, s)
            train.format.close(worker, f)

    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")

if __name__ == "__main__":
    w.write()
