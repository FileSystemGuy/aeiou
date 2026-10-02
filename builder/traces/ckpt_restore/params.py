"""The `ckpt_restore` parameters of a real DCP checkpoint, from its `.metadata`: the items of rank 0's
shard in file order (storage bytes, offsets), and the order `dcp.load` reads them in (sorted names)."""
import json, math, os, pickle, sys
import torch.distributed.checkpoint  # noqa: F401  (the classes the pickle names)

ckpt, step, doc = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(os.path.join(ckpt, ".metadata"), "rb") as f:
    m = pickle.load(f)
items = sorted((v.offset, k.fqn, tuple(k.offset)) for k, v in m.storage_data.items() if v.relative_path == "__0_0.distcp")
def nbytes(fqn, off):
    t = m.state_dict_metadata[fqn]
    c = next(c for c in t.chunks if tuple(c.offsets) == off)
    return math.prod(c.sizes) * t.properties.dtype.itemsize
by_name = sorted(range(len(items)), key=lambda i: items[i][1])
print(json.dumps({"params_version": 1, "abstract": "ckpt_restore", "doc": doc, "params": {
    "items": len(items), "item_bytes": [nbytes(f, o) for _, f, o in items], "item_off": [o for o, _, _ in items],
    "read_order": by_name, "meta_bytes": os.path.getsize(os.path.join(ckpt, ".metadata")), "restore_step": step}}))
