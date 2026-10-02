"""Read the beam search off an `strace` of a DiskANN search: per thread, `io_submit` batch
sizes; per query (a query starts at search.py's mark, or without marks at the submit that reads a
medoid's sector, which only an index without a node cache issues), rounds and
sector reads; how the sector reads spread over the file; and the submits of more than 32
requests (the node cache's load), which `strace` abbreviates and which are only counted.

    python hops.py TRACE INDEX_DIR [--prefix ann] [--tail N] [--params DOC]

`--tail N` keeps only the last N queries of each thread (the queries after a warm-up).
`--params DOC` prints the `vdb_search_diskann` parameter file fitted to the trace instead.
"""
import argparse, collections, json, os, re, struct, sys
import numpy as np

SUBMIT = re.compile(r"^(\d+) +([\d.]+) io_submit\(\w+, (\d+), ")
OFF = re.compile(r"aio_nbytes=(\d+), aio_offset=(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("index")
    ap.add_argument("--prefix", default="ann")
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--params", metavar="DOC", help="print the abstract's parameter file, with this description")
    a = ap.parse_args()
    disk = os.path.join(a.index, a.prefix + "_disk.index")
    with open(disk, "rb") as f:
        hdr = f.read(4096)
    npts, ndims, medoid, max_node_len, per_sector = struct.unpack_from("<QQQQQ", hdr, 8)
    med = np.fromfile(os.path.join(a.index, a.prefix + "_disk.index_medoids.bin"), dtype="uint32")[2:]
    med_off = {(int(m) // per_sector + 1) * 4096 for m in med}
    sectors = os.path.getsize(disk) // 4096
    q = collections.defaultdict(list)                 # tid -> [[batch sizes of one query], ...]
    offs = collections.defaultdict(list)              # tid -> [[offsets], ...]
    sizes = collections.Counter()
    bulk = []
    marked, new = False, False
    for line in open(a.trace, errors="replace"):
        m = SUBMIT.match(line)
        if not m:
            if "/aeiou-query-mark" in line:      # search.py without --batch marks every query
                marked = new = True
            continue
        tid, n = int(m.group(1)), int(m.group(3))
        io = [(int(b), int(o)) for b, o in OFF.findall(line)]
        if len(io) != n:                              # strace prints 32 elements: a bulk submit, the node cache's load
            bulk.append(n)
            continue
        sizes.update(b for b, _ in io)
        if (new if marked else n == 1 and io[0][1] in med_off) or not q[tid]:
            new = False
            q[tid].append([]); offs[tid].append([])
        q[tid][-1].append(n); offs[tid][-1].extend(o for _, o in io)
    if marked:                                        # what precedes the first mark is the load
        load = {t: v[0] for t, v in q.items()}
        q = {t: v[1:] for t, v in q.items()}; offs = {t: v[1:] for t, v in offs.items()}
        loadinfo = {"submits": sum(len(v) for v in load.values()), "requests": sum(sum(v) for v in load.values())}
    else:
        loadinfo = None
    if a.tail:
        q = {t: v[-a.tail:] for t, v in q.items()}; offs = {t: v[-a.tail:] for t, v in offs.items()}
    queries = [x for v in q.values() for x in v]
    rounds = collections.Counter(len(x) for x in queries)
    batch = collections.Counter(n for x in queries for n in x)
    first = collections.Counter(x[0] for x in queries)
    reads = np.array([sum(x) for x in queries])
    alloff = np.array([o for v in offs.values() for x in v for o in x]) // 4096
    cnt = np.sort(np.bincount(alloff, minlength=sectors))[::-1]
    top = max(1, sectors // 100)
    cached = queries and first.get(1, 0) < len(queries)     # with a node cache no query starts at a medoid
    second = [o // 4096 for t in q for b, x in zip(q[t], offs[t]) if len(b) > 1 for o in x[b[0]:b[0] + b[1]]]
    if a.params:
        k = 0 if cached else 2
        hops = collections.Counter(len(x) - k for x in queries)
        load = sum(bulk) + (loadinfo["requests"] if loadinfo else 0)
        print(json.dumps({"params_version": 1, "abstract": "vdb_search_diskann", "doc": a.params, "params": {
            "threads": len(q), "queries": len(queries) // len(q), "beam": max(batch),
            "hops": {"empirical": {"values": sorted(hops), "weights": [hops[h] for h in sorted(hops)]}},
            "entry": {"hotset": {"fraction": (len(med_off) - 0.5) / sectors, "weight": 1.0}},
            **({} if cached else {"near": {"hotset": {"fraction": (len(set(second)) - 0.5) / sectors, "weight": 1.0}}}),
            "node_cache": load if cached else 0,
            "nodes": sectors, "per_sector": per_sector}}))
        return
    out = {
        "second_round": {"reads": len(second), "distinct_sectors": len(set(second))},
        "nodes": npts, "nodes_per_sector": per_sector, "sectors": sectors, "medoids": len(med),
        "bulk_submits": len(bulk), "bulk_requests": sum(bulk), "bulk_largest": max(bulk, default=0),
        "before_first_query": loadinfo,
        "threads": len(q), "queries": len(queries), "request_bytes": dict(sizes),
        "rounds_per_query": dict(sorted(rounds.items())), "batch_sizes": dict(sorted(batch.items())),
        "first_batch": dict(first), "reads_per_query": {"mean": round(float(reads.mean()), 1), "min": int(reads.min()),
                                                        "p50": int(np.median(reads)), "max": int(reads.max())},
        "sector_reads": int(len(alloff)), "distinct_sectors": int((cnt > 0).sum()),
        "share_of_reads_on_top_1pct_sectors": round(float(cnt[:top].sum() / cnt.sum()), 3),
        "share_of_reads_on_medoid_sectors": round(float(sum((alloff * 4096 == m).sum() for m in med_off) / len(alloff)), 4),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
