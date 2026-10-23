"""The `diskann_build` parameters of a traced build: the base file's shape, the shard
count and the mean shard graph size from the merge's reads in the trace, the graph's degree
from the disk index's header, and the sizes of the files the build left. The compute
(`pq_train`, `partition_time`, `build_time`) is the build's on its hosts, which no trace
measures: `--set` gives it (a JSON value each).

    python params.py TRACE BASE.fbin INDEX_DIR DOC [--prefix ann] [--set name=value]...
"""
import argparse, json, os, re, struct, sys

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("base"); ap.add_argument("index"); ap.add_argument("doc")
ap.add_argument("--prefix", default="ann")
ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
a = ap.parse_args()
with open(a.base, "rb") as f:
    n, dim = struct.unpack("<II", f.read(8))
graph = re.compile(r" read\(\d+<[^>]*subshard-(\d+)_mem\.index>.*\) = (\d+)")
merged = re.compile(r" writev\(\d+<[^>]*" + re.escape(a.prefix) + r"_mem\.index>.*\) = (\d+)")
shard, index_bytes = {}, 0
for line in open(a.trace, errors="replace"):
    m = graph.search(line)
    if m:
        shard[int(m.group(1))] = shard.get(int(m.group(1)), 0) + int(m.group(2))
    m = merged.search(line)
    if m:
        index_bytes += int(m.group(1))
if not shard:
    sys.exit("params.py: the trace reads no shard graph (subshard-N_mem.index): "
             "diskann_build models a build that partitions its base (lower --build-gb)")
size = lambda name: os.path.getsize(os.path.join(a.index, a.prefix + name))
# the disk index's header is DiskANN's save_bin of u64s: points, dimensions, medoid, a node's bytes (its vector,
# its neighbour count, and `degree` neighbours, as the layout read it from the merged graph), ...
with open(os.path.join(a.index, a.prefix + "_disk.index"), "rb") as f:
    points, _, _, node = struct.unpack("<8x4Q", f.read(40))
sectors = 1 + -(-points // max(1, 4096 // node)) * -(-node // 4096)
if size("_disk.index") // 4096 != sectors:
    sys.exit(f"params.py: {a.prefix}_disk.index is {size('_disk.index') // 4096} sectors where its header's "
             f"{points:,} points of {node} bytes take {sectors}")
params = {"dim": dim, "n": n, "shards": len(shard), "pq_code": (size("_pq_compressed.bin") - 8) // n,
    "degree": (node - dim * 4 - 4) // 4,
    "shard_index_bytes": sum(shard.values()) // len(shard), "index_bytes": index_bytes + 8,
    "sample_rows": (size("_sample_ids.bin") - 8) // 4}
for kv in a.set:
    k, v = kv.split("=", 1)
    params[k] = json.loads(v)
print(json.dumps({"params_version": 2, "abstract": "diskann_build", "doc": a.doc, "params": {k: {"default": v} for k, v in params.items()}}))
