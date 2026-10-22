"""The `faiss_ivf_ondisk` parameters of a real index: list count, the list-size distribution
and the popularity of lists over a query set (both fitted), and the read lengths
`read_index` issued on the index file, from an `strace` of the search.

    python params.py OUT QUERIES.fvecs TRACE DOC [--nprobe 64] [--calls 20] [--threads 1] [--batch 1]
"""
import argparse, json, os, re
import numpy as np

from search import fvecs

READ = re.compile(r" read\(\d+<[^>]*/populated\.index>.*, (\d+)\) = \d+")


def params(a, size, probes, code_size, prefetch_nthread, trace, index_bytes):
    """The parameter file from the lists' sizes in bytes (in list order), the lists the coarse
    quantizer probed per query, the index's on-disk lists, and the lines of the trace."""
    nlist = len(size)
    ln = np.log(size[size > 0])
    count = np.sort(np.bincount(probes.ravel(), minlength=nlist))[::-1].astype(float)
    rank = np.arange(1, nlist + 1)
    top = slice(0, nlist // 2)                       # the tail of rarely probed lists falls off faster than any Zipf
    s = -np.polyfit(np.log(rank[top]), np.log(count[top]), 1)[0]
    reads = [int(m.group(1)) for m in map(READ.search, trace) if m]
    return {"params_version": 2, "abstract": "faiss_ivf_ondisk", "doc": a.doc, "params": {k: {"default": v} for k, v in {
        "threads": a.threads, "calls": a.calls, "batch": a.batch, "nprobe": a.nprobe,
        "prefetch": min(prefetch_nthread, a.batch * a.nprobe),
        "lists": nlist, "code_size": code_size,
        "list_bytes": {"lognormal": {"median": round(float(np.exp(ln.mean()))), "sigma": round(float(ln.std()), 2),
                                     "min": int(size.min()), "max": int(size.max())}},
        "list_pop": {"zipf": {"s": round(float(s), 2)}},
        "index_bytes": index_bytes, "index_reads": reads}.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("queries_file"); ap.add_argument("trace"); ap.add_argument("doc")
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    import faiss                                     # here, so that `params` needs numpy alone (tests)

    xq = fvecs(a.queries_file)
    with open(a.trace) as f:                         # before the chdir: a relative path is the caller's (until 2026-10-15 OUT's)
        trace = f.readlines()
    os.chdir(a.out)
    index = faiss.read_index("populated.index")
    od = faiss.downcast_InvertedLists(index.invlists)
    size = np.array([od.lists.at(i).size for i in range(od.nlist)]) * (od.code_size + 8)
    _, probes = index.quantizer.search(xq, a.nprobe)
    print(json.dumps(params(a, size, probes, od.code_size, od.prefetch_nthread, trace, os.path.getsize("populated.index"))))


if __name__ == "__main__":
    main()
