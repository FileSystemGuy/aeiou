"""ABSTRACTS.md §4b: model load for serving (safetensors shards, vLLM/TGI shape).

Every process reads the model's small JSON files, then every shard: an 8-byte header length,
the JSON header, then the byte range of each tensor it needs. The calls around those touches
are the trace of `safetensors.safe_open(framework="pt")` under `from_pretrained` (2026-10-01,
safetensors 0.8, builder/traces/model_load): the library maps each shard twice, once to parse
the header and once as the tensors' storage, advises the kernel that access is sequential, and
closes both descriptors before any tensor byte is touched. It issues no `read`: the reads
below are what the mappings touch, and `--io-api mmap` is the application's own API here.
Which bytes of a tensor a process touches is still the draft's [verify] (a serving engine with
GPUs was not traced): the traced loader touches none until the tensor is used. Tensor-parallel rank `gpu mod tp` takes a 1/TP slice: contiguous for
column-parallel weights, one piece per row for row-parallel ones (a page-fault storm under
`--io-api mmap`), and the whole tensor for replicated ones (norms, biases). The tensor
table is a set of parallel parameter arrays indexed by t; `tensors_in(s)` of the paper form is
a loop over all tensors guarded by `shard[t] == s`. The defaults are a five-tensor stand-in; a
real model's table is a parameter file built by `aeiou-params safetensors` from its shards
(schema/README.md §8), the shape/parameters split this abstract motivated (DESIGN_REVIEW.md
§3.19, §3.27).
"""
from aeiou import *

w = Workload("model_load", api="mmap",     # safetensors maps the shards and never calls read
             doc="Every process reads every safetensors shard, touching its tensor-parallel slice of each tensor (fan-in G).")
P = w.P
w.param("shards", 2, unit="count", doc="[config]; 4 here would be a 20 GiB model, 2 keeps the example short")
w.param("shard_bytes", 5 * GiB, unit="bytes", doc="[config] every shard is modeled at the largest shard's size")
w.param("hdr_len", [96 * KiB, 96 * KiB], unit="bytes", doc="[config] JSON header per shard")
w.param("tp", 8, unit="count")
w.param("configs", 3, unit="count", doc="small JSON files read before the shards: config.json, model.safetensors.index.json, generation_config.json (traced)")
w.param("config_bytes", 4 * KiB, unit="bytes", doc="[config] modeled at one size; traced 832, 5,701 and 203 bytes")
# the tensor table [config], one column per parameter array; index t (a real model's table comes from
# `aeiou-params safetensors`, schema/examples/params/model_load.synthetic.params.json is one)
w.param("shard", [0, 0, 1, 1, 1], unit="count", doc="which shard holds tensor t")
w.param("off", [96 * KiB + 8, 96 * KiB + 8 + 512 * MiB, 96 * KiB + 8, 96 * KiB + 8 + 64 * MiB, 96 * KiB + 8 + 64 * MiB + 1 * GiB],
        unit="bytes")
w.param("bytes", [512 * MiB, 128 * MiB, 64 * MiB, 1 * GiB, 16 * KiB], unit="bytes")
w.param("split", ["column", "row", "column", "row", "full"],
        doc="column: contiguous 1/tp slice; row: one piece per row; full: replicated, read whole")
w.param("rows", [1, 4096, 1, 8192, 1], unit="count")
w.param("row_bytes", [512 * MiB, 32 * KiB, 64 * MiB, 128 * KiB, 16 * KiB], unit="bytes")

model = w.dataset("model", pattern="model/model-{id:05}.safetensors", count=P.shards,
                  size=const(P.shard_bytes), seed=0x5eed_da7c)
config = w.dataset("config", pattern="config/config-{id}.json", count=P.configs, size=const(P.config_bytes),
                   seed=0x5eed_da7d)   # beside the shards in the real layout; a dataset has its root to itself here (V13)

with w.actor("gpu") as gpu:
    rank = gpu.let("rank", gpu_id % P.tp)
    with gpu.phase("load"):
        with gpu.loop("i", P.configs) as i:                                   # json.load(open(...)): Python reads size + 1, then EOF
            j = gpu.let("j", config.file(i))
            gpu.stat(j)
            gpu.open(j, "RDONLY|CLOEXEC")
            gpu.fstat(j)
            gpu.ioctl(j, "TCGETS", expect=["ENOTTY"])
            gpu.lseek(j, 0, "CUR")
            gpu.lseek(j, 0, "CUR")
            gpu.fstat(j)
            gpu.read(j, j.size + 1, repeat="until_eof")
            gpu.close(j)
        with gpu.loop("s", P.shards) as s:
            f = gpu.let("f", model.file(s))
            gpu.stat(f)
            gpu.open(f, "RDONLY|CLOEXEC")                                     # the mapping the header is parsed from
            gpu.fstat(f)
            gpu.read(f, 8)
            gpu.read(f, P.hdr_len[s])
            gpu.close(f)                                                      # (the application closes it after the second open)
            gpu.open(f, "RDONLY")                                             # the mapping the tensors are views of
            gpu.fstat(f)
            gpu.fadvise(f, "SEQUENTIAL", offset=0, len=f.size)                # the application closes f here: its mapping stays, ours needs the handle
            with gpu.loop("t", P.shard.len) as t:
                with gpu.when(P.shard[t] == s):
                    with gpu.when(P.split[t] == "column"):
                        gpu.read(f, P.bytes[t] // P.tp, offset=P.off[t] + rank * (P.bytes[t] // P.tp))
                    with gpu.otherwise():
                        with gpu.when(P.split[t] == "row"):
                            with gpu.loop("r", P.rows[t]) as r:              # strided: one piece per row
                                gpu.read(f, P.row_bytes[t] // P.tp,
                                         offset=P.off[t] + r * P.row_bytes[t] + rank * (P.row_bytes[t] // P.tp))
                        with gpu.otherwise():                                 # replicated: every rank reads it whole
                            gpu.read(f, P.bytes[t], offset=P.off[t])
            gpu.close(f)
        gpu.barrier("global")

if __name__ == "__main__":
    w.write()
