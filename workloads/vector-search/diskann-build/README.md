# Trace of the real build behind `diskann_build` (2026-10-02)

`ABSTRACTS.md` §11, row 7: DiskANN's `build_disk_index` through diskannpy, over SIFT1M
(graph degree 64, build list 100), with a build memory budget of 0.25 GB so that the builder
partitions the base into shards (13 here), builds each, and merges them: the path a corpus
larger than memory takes. On the loopback NFS mount of `runner/REFERENCE.md` §7. What was read
off the trace is in `ABSTRACTS.md` §7 and `DESIGN_REVIEW.md` §3.50. The index it leaves is
the one `../diskann-search` searches.

Versions: Python 3.11, diskannpy 0.7.0 (no wheel for 3.12), numpy 1.25.0, Linux 6.18. The
corpus is `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`.

```
C=cases/sift1m-13-shards                              # from this directory
python kit/build.py --fbin sift/sift_base.fvecs /mnt/nfs/base/base.fbin
strace -f --seccomp-bpf -ttt -T -yy -v -e trace=%file,%desc,%network,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt \
    python kit/build.py /mnt/nfs/base/base.fbin /mnt/nfs/index
aeiou-trace metrics trace.txt --root /mnt/nfs -o $C/captures/2026-10-02/trace.metrics.json
python kit/params.py trace.txt /mnt/nfs/base/base.fbin /mnt/nfs/index "…" --set pq_train=59000000000 \
    --set partition_time=91000000000 --set build_time=20000000000 > $C/params.json   # the compute: not fitted
aeiou dry-run diskann_build.ast.json --gpus 1 \
    --params-file $C/params.json --metrics-json abstract.metrics.json
```

`--seccomp-bpf` is not optional here. Without it `strace` stops the build's 20 threads at
every system call, traced or not, and the build's k-means spends its time there: the first
attempt had finished two of 32 PQ chunks after 20 minutes; with the filter the whole build
took 407 s. The trace is 1.3 million lines, of which 825,000 are `lseek`s that move nothing
(a C++ stream asked for its position twice per row).

The compute is not in the metrics a trace gives. The `--set` values are the traced run's: PQ
training 59 s and the partition 91 s, and 20 s per shard, the 407 s of the build less those two
over its 13 shards, an upper bound with the merge, the layout, and the I/O in it. `degree` is
read from the disk index's header, a node's bytes as the layout wrote them.

`builder/tests/test_trace.py` repeats the dry run against the committed metrics.
