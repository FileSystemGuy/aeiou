"""ABSTRACTS.md §4b: model load for serving (safetensors shards, vLLM/TGI shape).

Every process reads the model's small JSON files, then every shard: an 8-byte header length,
the JSON header, then the byte range of each tensor it needs. The calls around those touches
are the trace of `safetensors.safe_open(framework="pt")` under `from_pretrained` (2026-10-01,
safetensors 0.8, workloads/model-loading/safetensors-load): the library maps each shard twice, once to parse
the header and once as the tensors' storage, advises the kernel that access is sequential, and
closes both descriptors before any tensor byte is touched. It issues no `read`: the reads
below are what the mappings touch, and `--posix mmap` is the application's own API here.
Which bytes of a tensor a process touches is still the draft's [verify] (a serving engine with
GPUs was not traced): the traced loader touches none until the tensor is used. Tensor-parallel rank `gpu mod tp` takes a 1/TP slice: contiguous for
column-parallel weights, one piece per row for row-parallel ones (a page-fault storm under
`--posix mmap`), and the whole tensor for replicated ones (norms, biases). The tensor
table is a set of parallel parameter arrays indexed by t; `tensors_in(s)` of the paper form is
a loop over all tensors guarded by `shard[t] == s`. The model's shards and tensor table and the
deployment's `tp` are required (DESIGN_REVIEW.md §3.78, 2026-10-23): a real model's table is a
parameter file built by `aeiou-params safetensors` from its shards (schema/README.md §8), the
shape/parameters split this abstract motivated (DESIGN_REVIEW.md §3.19, §3.27), and the
five-tensor stand-in the defaults held until then is the case five-tensors, not measured.
"""
from aeiou import *

w = Workload("safetensors_load", posix="mmap",     # safetensors maps the shards and never calls read
             doc="Every process reads every safetensors shard, touching its tensor-parallel slice of each tensor (fan-in G).")
P = w.P
w.param("shards", kind=int, unit="count", doc="the model's shard files")
w.param("shard_bytes", kind=int, unit="bytes", doc="every shard is modeled at the largest shard's size")
w.param("hdr_len", kind=list[int], unit="bytes", doc="JSON header per shard")
w.param("tp", kind=int, unit="count", doc="the deployment's tensor-parallel degree: rank gpu mod tp takes a 1/tp slice")
w.param("configs", 3, unit="count", doc="small JSON files read before the shards: config.json, model.safetensors.index.json, generation_config.json (traced)")
w.param("config_bytes", 4 * KiB, unit="bytes", doc="[config] modeled at one size; traced 832, 5,701 and 203 bytes")
# the tensor table, one column per parameter array; index t (a real model's table comes from
# `aeiou-params safetensors`, workloads/model-loading/safetensors-load/params/synthetic.params.json is one)
w.param("shard", kind=list[int], unit="count", doc="which shard holds tensor t")
w.param("off", kind=list[int], unit="bytes", doc="tensor t's offset in its shard")
w.param("bytes", kind=list[int], unit="bytes", doc="tensor t's length")
w.param("split", kind=list[str],
        doc="column: contiguous 1/tp slice; row: one piece per row; full: replicated, read whole")
w.param("rows", kind=list[int], unit="count", doc="tensor t's rows (1 unless its split is row)")
w.param("row_bytes", kind=list[int], unit="bytes", doc="the bytes of one of tensor t's rows")

model = w.dataset("model", pattern="model/model-{id:05}.safetensors", layout="whole", access="indexed", count=P.shards,
                  size=const(P.shard_bytes), seed=0x5eed_da7c)
config = w.dataset("config", pattern="config/config-{id}.json", layout="whole", access="indexed", count=P.configs, size=const(P.config_bytes),
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
