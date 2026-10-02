"""ABSTRACTS.md §11 row 4b: the library alone. `safetensors.safe_open(framework="pt")` on every shard,
then a copy of every tensor in file order, which is what faults the mapping in. ONE=1 copies with one
thread (torch's default is a parallel copy, several faulting threads per tensor)."""
import glob, json, os, struct, sys
import torch
from safetensors import safe_open
if os.environ.get("ONE"):
    torch.set_num_threads(1)
for p in sorted(glob.glob(sys.argv[1] + "/*.safetensors")):
    with open(p, "rb") as b:
        h = json.loads(b.read(struct.unpack("<Q", b.read(8))[0]))
    ks = sorted((k for k in h if k != "__metadata__"), key=lambda k: h[k]["data_offsets"][0])
    with safe_open(p, framework="pt") as f:
        for k in ks:
            f.get_tensor(k).clone()
