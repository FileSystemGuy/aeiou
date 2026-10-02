# Trace of the real build behind `vdb_build_diskann` (2026-10-02)

`ABSTRACTS.md` §11, row 7: DiskANN's `build_disk_index` through diskannpy, over SIFT1M
(graph degree 64, build list 100), with a build memory budget of 0.25 GB so that the builder
partitions the base into shards (13 here), builds each, and merges them: the path a corpus
larger than memory takes. On the loopback NFS mount of `runner/README.md` §7. What was read
off the trace is in `ABSTRACTS.md` §7 and `DESIGN_REVIEW.md` §3.50. The index it leaves is
the one `../vdb_search_diskann` searches.

Versions: Python 3.11, diskannpy 0.7.0 (no wheel for 3.12), numpy 1.25.0, Linux 6.18. The
corpus is `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`.

```
python build.py --fbin sift/sift_base.fvecs /mnt/nfs/base/base.fbin
strace -f --seccomp-bpf -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt \
    python build.py /mnt/nfs/base/base.fbin /mnt/nfs/index
aeiou-trace metrics trace.txt --root /mnt/nfs -o trace.metrics.json
python params.py trace.txt /mnt/nfs/base/base.fbin /mnt/nfs/index "…" > fitted.params.json
aeiou dry-run ../../../schema/examples/vdb_build_diskann.ast.json --gpus 1 \
    --params fitted.params.json --metrics-json abstract.metrics.json
```

`--seccomp-bpf` is not optional here. Without it `strace` stops the build's 20 threads at
every system call, traced or not, and the build's k-means spends its time there: the first
attempt had finished two of 32 PQ chunks after 20 minutes; with the filter the whole build
took 407 s. The trace is 1.3 million lines, of which 825,000 are `lseek`s that move nothing
(a C++ stream asked for its position twice per row).

`tests/test_trace.py` repeats the dry run against the committed metrics.
