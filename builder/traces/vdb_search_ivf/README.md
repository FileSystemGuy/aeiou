# Trace of the real search behind `vdb_search_ivf` (2026-10-01)

`ABSTRACTS.md` §11, row 6: FAISS `IndexIVFPQ` with `OnDiskInvertedLists`, `nprobe = 64`, over
SIFT1M (`IVF1024,PQ32`: 1,024 lists, 40 bytes per vector, a 40 MB lists file) on the loopback
NFS mount of `runner/README.md` §7. What was read off the trace is in `ABSTRACTS.md` §6 and
`DESIGN_REVIEW.md` §3.49.

Versions: Python 3.12.3, faiss-cpu 1.15.1, numpy 2.5.3, Linux 6.18. The corpus is
`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (168 MB).

```
python mkindex.py sift /mnt/nfs/out                       # train, four shard indexes, merge_ondisk
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt \
    python search.py /mnt/nfs/out sift/sift_query.fvecs --queries 20 --threads 1
aeiou-trace metrics trace.txt --root /mnt/nfs -o trace.metrics.json
python params.py /mnt/nfs/out sift/sift_query.fvecs trace.txt "…" > fitted.params.json
aeiou dry-run ../../../schema/examples/vdb_search_ivf.ast.json --gpus 1 \
    --params-file fitted.params.json --metrics-json abstract.metrics.json
python faults.py /mnt/nfs/out sift/sift_query.fvecs --queries 1 --threads 1
```

FAISS issues no call on the lists file after `mmap`: `strace` sees the seven `read`s of the
index file, the two opens, and the threads (32 per search slice), and nothing of the lists.
`faults.py` supplies what the trace cannot: it drops the lists file from the page cache
(`fsync` and `POSIX_FADV_DONTNEED`, no root), searches, and compares the file's resident
pages (`mincore`) with the pages of the lists the coarse quantizer selected, whose offsets
and sizes it reads from the index; it also prints the mount's NFS READ delta. Every page of
every probed list is resident afterwards, with the kernel's read-around on top.

`params.py` fits the parameter file from the index (list count, code size, the log-normal
fit of list bytes, the Zipf fit of list popularity over the 10,000 SIFT queries) and takes
the index file's read lengths from the trace.

`tests/test_trace.py` repeats the dry run against the committed metrics.
