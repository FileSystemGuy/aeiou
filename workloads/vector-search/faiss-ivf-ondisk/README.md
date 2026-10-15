# Trace of the real search behind `faiss_ivf_ondisk` (2026-10-01)

`ABSTRACTS.md` §11, row 6: FAISS `IndexIVFPQ` with `OnDiskInvertedLists`, `nprobe = 64`, over
SIFT1M (`IVF1024,PQ32`: 1,024 lists, 40 bytes per vector, a 40 MB lists file) on the loopback
NFS mount of `runner/REFERENCE.md` §7. What was read off the trace is in `ABSTRACTS.md` §6 and
`DESIGN_REVIEW.md` §3.49.

Versions: Python 3.12.3, faiss-cpu 1.15.1, numpy 2.5.3, Linux 6.18. The corpus is
`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (168 MB).

```
C=cases/sift1m-ivf1024-pq32                               # from this directory
python kit/mkindex.py sift /mnt/nfs/out                   # train, four shard indexes, merge_ondisk
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt \
    python kit/search.py /mnt/nfs/out sift/sift_query.fvecs --queries 20 --threads 1
aeiou-trace metrics trace.txt --root /mnt/nfs -o $C/captures/2026-10-01/trace.metrics.json
python kit/params.py /mnt/nfs/out sift/sift_query.fvecs trace.txt "…" > $C/params.json
aeiou dry-run faiss_ivf_ondisk.ast.json --gpus 1 \
    --params-file $C/params.json --metrics-json abstract.metrics.json
python kit/faults.py /mnt/nfs/out sift/sift_query.fvecs --queries 1 --threads 1
```

FAISS issues no call on the lists file after `mmap`: `strace` sees the seven `read`s of the
index file, the two opens, and the threads (32 per search slice), and nothing of the lists.
`kit/faults.py` supplies what the trace cannot: it drops the lists file from the page cache
(`fsync` and `POSIX_FADV_DONTNEED`, no root), searches, and compares the file's resident
pages (`mincore`) with the pages of the lists the coarse quantizer selected, whose offsets
and sizes it reads from the index; it also prints the mount's NFS READ delta. Every page of
every probed list is resident afterwards, with the kernel's read-around on top.

`kit/params.py` fits the parameter file from the index (list count, code size, the log-normal
fit of list bytes, the Zipf fit of list popularity over the 10,000 SIFT queries) and takes
the index file's read lengths from the trace.

`builder/tests/test_trace.py` repeats the dry run against the committed metrics.
