"""Streaming training over sharded containers: the tf.data shape (`GRAMMAR_OPTIONS.md` §6.5,
`PROJECT_BRIEF.md` §6 item 15; the ResNet50 / CosmoFlow pattern MLPerf Storage runs, and what
HF `streaming=True`, Ray Data, and DALI do over shards).

The input pipeline lists the shards, shuffles the list, and interleaves `cycle` shards at a
time, each streamed sequentially by its reader library; records are batched from the
interleaved stream and an in-memory shuffle buffer produces no I/O. In the model the unit of
work a loader worker does is therefore a **shard**, not a batch: `consume` over a `stream`
dataset draws shard ids from the epoch's permutation, the worker runs the format class's
reader protocol over the whole shard, and the training loop takes a shard every
`per_shard / batch` steps. Two workloads from one shape: TFRecord through
`tf.data.TFRecordDataset` and Parquet through `pyarrow.parquet.ParquetFile` (full
projection; the format class's `read_all(columns=[…])` is the projected variant).

Cuts: decode CPU is one `compute` per shard (`decode`, 0 by default); the shuffle buffer and
the batch boundary inside a shard are application memory; the first shard of a worker is
read in full before the first step runs, as tf.data's prefetch would.
"""
from aeiou import *
from aeiou.formats import parquet, tfrecord


def shape(name, pattern, make_format, doc):
    w = Workload(name, doc=doc)
    P = w.P
    w.param("batch", 256, unit="count", doc="[config] samples per step")
    w.param("per_shard", 1024, unit="count", doc="[config] samples per shard; a multiple of batch")
    w.param("cycle", 4, unit="count", doc="[config] interleave cycle_length: shards one GPU's pipeline streams concurrently")
    w.param("prefetch", 1, unit="count", doc="[config] shards finished ahead of the training loop, per cycle slot")
    w.param("steps", 500, unit="count")
    w.param("sync_every", 500, unit="count", doc="1 for DDP; 500 keeps the brief's reference workload")
    w.param("step_time", 105 * ms, unit="ns", doc="[measure] GPU step time on the target accelerator")
    w.param("decode", 0, unit="ns", doc="[measure] decode compute per shard in the input pipeline; 0 = I/O bound")
    w.param("xfer", 256 * KiB, unit="bytes", doc="[verify] the reader library's read buffer (TFRecord)")
    w.param("samples", 1_281_167, unit="count", doc="[config] corpus size; sized to the dataset rule (PROJECT_BRIEF.md §5)")
    w.param("sample_median", 110 * KiB, unit="bytes", doc="[measure] corpus")
    shards = w.dataset("shards", pattern=pattern, count=P.samples, samples_per_file=P.per_shard,
                       size=lognormal(median=P.sample_median, sigma=0.45), seed=0x5eed_da90,
                       access="stream", format=make_format(P))

    with w.actor("gpu") as gpu:
        with gpu.loader("shards", workers=P.cycle, prefetch=P.prefetch,
                        batches=ceil_div(P.steps * P.batch, P.per_shard)) as worker:
            s = worker.let("s", shards.consume())                    # a shard, by position in the epoch's permutation
            yield w, worker, shards, s
            worker.compute(P.decode)

        with gpu.phase("train"), gpu.loop("step", P.steps) as step:
            with gpu.every(P.per_shard // P.batch):
                gpu.take("shards")
            gpu.compute(P.step_time)
            with gpu.every(P.sync_every):
                gpu.barrier("global")


for w, worker, shards, s in shape("train_stream_tfrecord", "train/shard-{id:05}.tfrecord",
                                  lambda P: tfrecord(xfer=P.xfer),
                                  "tf.data over TFRecord shards: cycle shards streamed concurrently, positioned reads of the buffer size to EOF."):
    shards.format.stream(worker, s)

for w, worker, shards, s in shape("train_stream_parquet", "train/shard-{id:05}.parquet",
                                  lambda P: parquet(rows_per_group=64, columns=(("image", "binary"), ("label", "int64"))),
                                  "pyarrow over Parquet shards: footer read, then WILLNEED and one pread per coalesced run of row groups."):
    shards.format.open_reads(worker, s)
    shards.format.read_all(worker, s)
    shards.format.close(worker, s)

if __name__ == "__main__":
    for wl in Workload._registry:
        wl.write()
