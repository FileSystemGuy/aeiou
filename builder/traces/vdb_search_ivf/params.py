"""The `vdb_search_ivf` parameters of a real index: list count, the list-size distribution
and the popularity of lists over a query set (both fitted), and the read lengths
`read_index` issued on the index file, from an `strace` of the search.

    python params.py OUT QUERIES.fvecs TRACE DOC [--nprobe 64] [--calls 20] [--threads 1] [--batch 1]
"""
import argparse, json, os, re
import numpy as np
import faiss

from search import fvecs

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("queries_file"); ap.add_argument("trace"); ap.add_argument("doc")
ap.add_argument("--nprobe", type=int, default=64)
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--batch", type=int, default=1)
a = ap.parse_args()

xq = fvecs(a.queries_file)
os.chdir(a.out)
index = faiss.read_index("populated.index")
od = faiss.downcast_InvertedLists(index.invlists)
size = np.array([od.lists.at(i).size for i in range(od.nlist)]) * (od.code_size + 8)
ln = np.log(size[size > 0])
_, probes = index.quantizer.search(xq, a.nprobe)
count = np.sort(np.bincount(probes.ravel(), minlength=od.nlist))[::-1].astype(float)
rank = np.arange(1, od.nlist + 1)
top = slice(0, od.nlist // 2)                       # the tail of rarely probed lists falls off faster than any Zipf
s = -np.polyfit(np.log(rank[top]), np.log(count[top]), 1)[0]
reads = [int(m.group(1)) for m in map(re.compile(r" read\(\d+<[^>]*/populated\.index>.*, (\d+)\) = \d+").search, open(a.trace)) if m]
print(json.dumps({"params_version": 1, "abstract": "vdb_search_ivf", "doc": a.doc, "params": {
    "threads": a.threads, "calls": a.calls, "batch": a.batch, "nprobe": a.nprobe,
    "prefetch": min(od.prefetch_nthread, a.batch * a.nprobe),
    "lists": od.nlist, "code_size": od.code_size,
    "list_bytes": {"lognormal": {"median": round(float(np.exp(ln.mean()))), "sigma": round(float(ln.std()), 2),
                                 "min": int(size.min()), "max": int(size.max())}},
    "list_pop": {"zipf": {"s": round(float(s), 2)}},
    "index_bytes": os.path.getsize("populated.index"), "index_reads": reads}}))
