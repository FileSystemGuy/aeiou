"""ABSTRACTS.md §4b: model load for serving (safetensors shards, vLLM/TGI shape).

Every process reads every shard: an 8-byte header length, the JSON header, then the byte range
of each tensor it needs. Tensor-parallel rank `gpu mod tp` takes a 1/TP slice: contiguous for
column-parallel weights, one piece per row for row-parallel ones (a page-fault storm under
`--io-backend mmap`). The tensor table is a set of parallel parameter arrays indexed by t;
`tensors_in(s)` of the paper form is a loop over all tensors guarded by `shard[t] == s`.
"""
from mlps_abstract import *

w = Workload("model_load",
             doc="Every process reads every safetensors shard, touching its tensor-parallel slice of each tensor (fan-in G).")
P = w.P
w.param("shards", 2, unit="count", doc="[config]; 4 here would be a 20 GiB model, 2 keeps the example short")
w.param("shard_bytes", 5 * GiB, unit="bytes", doc="[config]")
w.param("hdr_len", [96 * KiB, 96 * KiB], unit="bytes", doc="[config] JSON header per shard")
w.param("tp", 8, unit="count")
# the tensor table [config], one column per parameter array; index t
w.param("shard", [0, 0, 1, 1], unit="count", doc="which shard holds tensor t")
w.param("off", [96 * KiB + 8, 96 * KiB + 8 + 512 * MiB, 96 * KiB + 8, 96 * KiB + 8 + 64 * MiB], unit="bytes")
w.param("bytes", [512 * MiB, 128 * MiB, 64 * MiB, 1 * GiB], unit="bytes")
w.param("split", ["column", "row", "column", "row"], doc="column: contiguous 1/tp slice; row: one piece per row")
w.param("rows", [1, 4096, 1, 8192], unit="count")
w.param("row_bytes", [512 * MiB, 32 * KiB, 64 * MiB, 128 * KiB], unit="bytes")

model = w.dataset("model", pattern="model-{id:05}-of-00002.safetensors", count=P.shards,
                  size=const(P.shard_bytes), seed=0x5eed_da7c)

with w.actor("gpu") as gpu:
    rank = gpu.let("rank", gpu_id % P.tp)
    with gpu.phase("load"):
        with gpu.loop("s", P.shards) as s:
            f = gpu.let("f", model.file(s))
            gpu.open(f, "RDONLY|CLOEXEC")
            gpu.fstat(f)
            gpu.read(f, 8)
            gpu.read(f, P.hdr_len[s])
            with gpu.loop("t", P.shard.len) as t:
                with gpu.when(P.shard[t] == s):
                    with gpu.when(P.split[t] == "column"):
                        gpu.read(f, P.bytes[t] // P.tp, offset=P.off[t] + rank * (P.bytes[t] // P.tp))
                    with gpu.otherwise():
                        with gpu.loop("r", P.rows[t]) as r:              # strided: one piece per row
                            gpu.read(f, P.row_bytes[t] // P.tp,
                                     offset=P.off[t] + r * P.row_bytes[t] + rank * (P.row_bytes[t] // P.tp))
            gpu.close(f)
        gpu.barrier("global")

if __name__ == "__main__":
    w.write()
