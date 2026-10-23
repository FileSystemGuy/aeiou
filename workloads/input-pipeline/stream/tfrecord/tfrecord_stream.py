"""Streaming training over sharded containers: the tf.data shape (`GRAMMAR_OPTIONS.md` §6.5,
`PROJECT_BRIEF.md` §6 item 15; the ResNet50 / CosmoFlow pattern MLPerf Storage runs, and what
HF `streaming=True`, Ray Data, and DALI do over shards).

The input pipeline lists the shards, shuffles the list, and interleaves `cycle` shards at a
time, each streamed sequentially by its reader library; records are batched from the
interleaved stream and an in-memory shuffle buffer produces no I/O. In the model the unit of
work a loader worker does is therefore a **shard**, not a batch: `consume` over a `stream`
dataset draws shard ids from the epoch's permutation, the worker runs the format class's
reader protocol over the whole shard, and the training loop takes a shard at the step whose
batch holds the shard's first sample (every `samples_per_file / batch` steps when batch divides
it, several in one step when `batch` exceeds `samples_per_file`). This one: TFRecord through
`tf.data.TFRecordDataset`.
`../parquet/parquet_stream.py` is the same shape over Parquet; the two repeat the
shape on purpose, one mechanism per directory (`DESIGN_REVIEW.md` §3.66).

Cuts: decode CPU is one `compute` per shard (`decode`, 0 by default); the shuffle buffer and
the batch boundary inside a shard are application memory; the first shard of a worker is
read in full before the first step runs, as tf.data's prefetch would.
"""
from aeiou import *
from aeiou.formats import tfrecord


def shape(name, pattern, make_format, doc):
    w = Workload(name, doc=doc)
    P = w.P
    # the job and the model on a GPU are a case's (DESIGN_REVIEW.md §3.78 point 1): required here; the
    # placeholders the defaults held until 2026-10-22, never measured, are cases/imagenet-records (untraced)
    w.param("batch", kind=int, unit="count", doc="samples per step: the model's on its accelerator, whose memory sets it")
    w.param("samples_per_file", kind=int, unit="count", doc="samples per shard: the job's corpus")
    w.param("cycle", kind=int, unit="count", doc="interleave cycle_length: shards one GPU's pipeline streams concurrently; the job's")
    w.param("prefetch", kind=int, unit="count", doc="shards finished ahead of the training loop, per cycle slot; the job's")
    w.param("steps", kind=int, unit="count", doc="steps of the run: the job's")
    w.param("sync_every", 1, unit="count", doc="synchronous data parallel (DDP, FSDP): every step waits for the slowest GPU (DESIGN_REVIEW.md §3.71 point 4; 500, the brief's reference workload, until 2026-10-14)")
    w.param("step_time", kind=int, unit="ns", doc="compute per step: the model's on its accelerator")
    w.param("decode", kind=int, unit="ns", doc="decode compute per shard in the input pipeline, the job's; 0 = I/O bound")
    w.param("xfer", 256 * KiB, unit="bytes", doc="[verify] the reader library's read buffer (TFRecord)")
    w.param("samples", kind=int, scale=True, unit="count", doc="corpus size: the submission's, sized to the dataset rule (PROJECT_BRIEF.md §5)")
    w.param("sample_median", kind=int, unit="bytes", doc="median sample size: the job's corpus")
    shards = w.dataset("shards", pattern=pattern, layout="compound", access="stream", count=P.samples, samples_per_file=P.samples_per_file,
                       size=lognormal(median=P.sample_median, sigma=0.45), seed=0x5eed_da90,
                       format=make_format(P))

    with w.actor("gpu") as gpu:
        with gpu.loader("shards", workers=P.cycle, prefetch=P.prefetch,
                        batches=ceil_div(P.steps * P.batch, P.samples_per_file)) as worker:
            s = worker.let("s", shards.consume())                    # a shard, by position in the epoch's permutation
            yield w, worker, shards, s
            worker.compute(P.decode)

        with gpu.phase("train"), gpu.loop("step", P.steps) as step:
            # the shards whose first sample is in this step's batch: one every samples_per_file / batch
            # steps when batch divides it, several in a step when a batch spans shards; the loader's
            # ceil_div(steps · batch, samples_per_file) in all, for any two positive values
            with gpu.loop("k", ceil_div((step + 1) * P.batch, P.samples_per_file) - ceil_div(step * P.batch, P.samples_per_file)):
                gpu.take("shards")
            gpu.compute(P.step_time)
            with gpu.every(P.sync_every):
                gpu.barrier("global")


for w, worker, shards, s in shape("tfrecord_stream", "train/shard-{id:05}.tfrecord",
                                  lambda P: tfrecord(xfer=P.xfer),
                                  "tf.data over TFRecord shards: cycle shards streamed concurrently, positioned reads of the buffer size to EOF."):
    shards.format.stream(worker, s)

if __name__ == "__main__":
    for wl in Workload._registry:
        wl.write()
