"""The `vdb_build_diskann` parameters of a traced build: the base file's shape, the shard
count and the mean shard graph size from the merge's reads in the trace, and the sizes of
the files the build left.

    python params.py TRACE BASE.fbin INDEX_DIR DOC [--prefix ann]
"""
import argparse, json, os, re, struct

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("base"); ap.add_argument("index"); ap.add_argument("doc")
ap.add_argument("--prefix", default="ann")
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
size = lambda name: os.path.getsize(os.path.join(a.index, a.prefix + name))
print(json.dumps({"params_version": 1, "abstract": "vdb_build_diskann", "doc": a.doc, "params": {
    "dim": dim, "n": n, "shards": len(shard), "pq_code": (size("_pq_compressed.bin") - 8) // n,
    "shard_index_bytes": sum(shard.values()) // len(shard), "index_bytes": index_bytes + 8,
    "sectors": size("_disk.index") // 4096, "sample_rows": (size("_sample_ids.bin") - 8) // 4}}))
