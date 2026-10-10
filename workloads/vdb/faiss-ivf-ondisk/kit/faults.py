"""What a search touches in the lists file, which `strace` cannot see (the file is mapped).

    python faults.py OUT QUERIES.fvecs [--nprobe 64] [--batch 1] [--queries 20] [--threads N]

Drops the lists file from the page cache, searches, then compares the file's resident pages
(`mincore` on a second mapping) with the pages of the lists the coarse quantizer selected
(offsets and sizes from the index), and prints the NFS READ delta of the mount.
"""
import argparse, ctypes, mmap, os
import numpy as np
import faiss

from search import fvecs

PAGE = mmap.PAGESIZE
libc = ctypes.CDLL(None, use_errno=True)


def resident(path):
    fd = os.open(path, os.O_RDONLY)
    n = os.fstat(fd).st_size
    m = mmap.mmap(fd, n, prot=mmap.PROT_READ)
    pages = (n + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * pages)()
    arr = np.frombuffer(m, dtype=np.uint8)
    if libc.mincore(ctypes.c_void_p(arr.ctypes.data), ctypes.c_size_t(n), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    out = np.frombuffer(vec, dtype=np.uint8) & 1
    out = out.astype(bool).copy()
    del arr
    m.close()
    os.close(fd)
    return out


def nfs_reads(path):
    """(READ ops, bytes read over the wire) of the NFS mount holding `path`, or None."""
    best, cur, ops, wire = "", None, None, None
    for line in open("/proc/self/mountstats"):
        w = line.split()
        if line.startswith("device "):
            cur = w[4] if len(w) > 7 and w[7].startswith("nfs") else None
            if cur and not (path.startswith(cur.rstrip("/") + "/") and len(cur) > len(best)):
                cur = None
            if cur:
                best = cur
        elif cur and w and w[0] == "bytes:":
            wire = int(w[5])                      # serverbytesread
        elif cur and w and w[0] == "READ:":
            ops = int(w[1])
    return None if ops is None else (ops, wire)


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("queries_file")
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    if a.threads:
        faiss.omp_set_num_threads(a.threads)
    xq = fvecs(a.queries_file)[:a.queries]
    data = os.path.join(os.path.abspath(a.out), "merged_index.ivfdata")
    os.chdir(a.out)
    index = faiss.read_index("populated.index")
    index.nprobe = a.nprobe
    od = faiss.downcast_InvertedLists(index.invlists)
    rec = od.code_size + 8
    _, probes = index.quantizer.search(xq, a.nprobe)
    drop(data)
    cold = resident(data)
    r0 = nfs_reads(data)
    for i in range(0, len(xq), a.batch):
        index.search(xq[i:i + a.batch], a.k)
    r1 = nfs_reads(data)
    got = resident(data)
    want = np.zeros(len(got), dtype=bool)
    nbytes = 0
    for c in np.unique(probes):
        l = od.lists.at(int(c))
        if l.size:
            want[l.offset // PAGE:(l.offset + l.size * rec - 1) // PAGE + 1] = True
            nbytes += l.size * rec
    print("probes %d, distinct lists %d of %d, their bytes %d"
          % (probes.size, len(np.unique(probes)), od.nlist, nbytes))
    print("pages: resident before %d, of the probed lists %d, resident after %d, "
          "probed and not resident %d, resident and not probed %d"
          % (cold.sum(), want.sum(), got.sum(), (want & ~got).sum(), (got & ~want).sum()))
    if r0:
        print("NFS READ: %d ops, %d bytes" % (r1[0] - r0[0], r1[1] - r0[1]))


if __name__ == "__main__":
    main()
