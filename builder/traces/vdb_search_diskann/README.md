# Trace of the real search behind `vdb_search_diskann` (2026-10-02)

`ABSTRACTS.md` §11, row 5: DiskANN's `PQFlashIndex` through `diskannpy.StaticDiskIndex`, over
SIFT1M (graph degree 64, 13 build shards, 200,001 sectors of 4 KiB, five nodes to a sector),
search list 100, beam 4, on the loopback NFS mount of `runner/README.md` §7. The index is the
one `../vdb_build_diskann/build.py` wrote. What was read off the traces is in `ABSTRACTS.md`
§5 and `DESIGN_REVIEW.md` §3.50.

Versions: Python 3.11, diskannpy 0.7.0 (no wheel for 3.12), numpy 1.25.0, Linux 6.18.

```
T="strace -f --seccomp-bpf -ttt -T -yy -o"
E="-e trace=%file,%desc,%network,%process,io_setup,io_submit,io_getevents,io_destroy"
$T trace.nocache.txt $E python search.py /mnt/nfs/index query.fbin --queries 1000 --threads 2 --batch
$T trace.cache.txt $E python search.py /mnt/nfs/index query.fbin --queries 1000 --threads 2 \
    --cache 10000 --cache-mechanism 2
aeiou-trace metrics trace.nocache.txt --root /mnt/nfs -o trace.nocache.metrics.json
python hops.py trace.nocache.txt /mnt/nfs/index                  # rounds, batch sizes, sector spread
python hops.py trace.nocache.txt /mnt/nfs/index --params "…" > fitted.nocache.params.json
aeiou dry-run ../../../schema/examples/vdb_search_diskann.ast.json --gpus 1 \
    --params fitted.nocache.params.json --metrics-json abstract.metrics.json
```

Three things about the capture:

- `%desc` does not include the AIO calls; they are named. `--seccomp-bpf` lets the calls
  that are not traced run at full speed, which a search with OpenMP threads needs.
- `strace` prints the first 32 requests of an `io_submit`. The searches submit at most
  `beam`; the node cache's load submits more, and `hops.py` only counts those.
- **The trace has no query boundaries.** Without a node cache a query starts at the submit
  of one read of a medoid's sector, and `hops.py` cuts there. With a cache nothing marks
  it, so `search.py` without `--batch` issues a `stat` of `/aeiou-query-mark` between
  queries. `aeiou-trace`'s `--chain-gap-us` found no usable gap under `strace`: the think
  time between rounds and between queries overlap.

`--cache-mechanism 1`, diskannpy's default, searches the build's sample of the base (100,131
queries here, 2.7 million sector reads) before the first query, whatever `--cache` is.
DiskANN's own `search_disk_index` uses breadth-first levels, mechanism 2.

`tests/test_trace.py` repeats the two dry runs against the committed metrics.
