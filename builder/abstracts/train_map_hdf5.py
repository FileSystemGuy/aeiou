"""Map-style training over HDF5 containers: a PyTorch DataLoader whose `Dataset.__getitem__`
opens the container holding sample `i` with h5py, reads row `i` of its one contiguous dataset,
and closes it (the plain map-style HDF5 dataset that survives `fork`; an open-file cache is
the cut). DLIO's HDF5 reader issues the same per-row read. The format class carries what a
trace of h5py shows: the metadata reads at open and libhdf5's sieve buffer on the row read.
"""
from aeiou import *
from aeiou.formats import hdf5

SHAPE = (256, 256, 3)                       # one sample: 196,608 bytes, larger than the 64 KiB sieve

w = Workload("train_map_hdf5",
             doc="DataLoader over HDF5 containers of fixed-size rows: open, metadata reads, one positioned row read, close per sample.")
P = w.P
w.param("batch", 32, unit="count")
w.param("workers", 8, unit="count")
w.param("prefetch", 2, unit="count")
w.param("steps", 500, unit="count")
w.param("sync_every", 500, unit="count", doc="1 for DDP; 500 keeps the brief's reference workload")
w.param("step_time", 105 * ms, unit="ns", doc="[measure] GPU step time on the target accelerator")
w.param("per_file", 512, unit="count", doc="[config] samples per container (96 MiB each)")
w.param("samples", 1_000_000, unit="count", doc="[config] corpus size; sized to the dataset rule (PROJECT_BRIEF.md §5)")

train = w.dataset("train", pattern="train/samples-{id:05}.h5", count=P.samples, samples_per_file=P.per_file,
                  size=const(SHAPE[0] * SHAPE[1] * SHAPE[2]), seed=0x5eed_da91, access="map",
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
