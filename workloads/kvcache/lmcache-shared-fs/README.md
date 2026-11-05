# Traces behind `lmcache_shared_fs` and `lmcache_shared_fs_reader` (2026-10-02; captured again 2026-11-05)

vLLM with LMCache's `fs://` remote backend on the loopback NFS mount of `runner/REFERENCE.md`
§7: one engine that fills an empty store, then a restarted engine in front of the filled
store, sent the same requests. What was read off the traces is in `ABSTRACTS.md` §8
("The shared store") and `DESIGN_REVIEW.md` §3.52; the committed traces are of the ShareGPT
replay of `../lmcache-local-disk` (§3.56), 300 requests with 8 conversations open: the case
`sharegpt-chat`, whose captures hold both traces' metrics, tolerance files, and capture records:
`captures/2026-11-05` under the eBPF source, `captures/2026-11-05-strace` under strace, the same
load (the model's replies byte for byte the same) on the loopback mount of
`runner/REFERENCE.md` §7.

Versions, model, `serve.sh`, `replay.py`, and `fit.py` are those of `../lmcache-local-disk/kit/`;
only `kit/lmcache.yaml` differs (`remote_url` in place of `local_disk`).

```
export PYTHONHASHSEED=0       # LMCache's chunk hash is Python's: two processes agree on names only with one seed
export KV_BYTES=100663296 LMCACHE_CONFIG_FILE=kit/lmcache.yaml   # from this directory; the store /mnt/aeiou-nfs/kv
K=../lmcache-local-disk/kit
S=ebpf; C=cases/sharegpt-chat/captures/2026-11-05      # or S=strace; C=cases/sharegpt-chat/captures/2026-11-05-strace
# --accept fault-around: LMCache maps no file of the store (CAPTURE.md, "The eBPF source")
CAPTURE="aeiou-trace capture --source $S --accept fault-around --root /mnt/aeiou-nfs/kv"
# the writer: an empty store
mkdir /mnt/aeiou-nfs/kv
$CAPTURE -o /local/writer -- sh $K/serve.sh > serve.log 2>&1 &
python $K/replay.py ShareGPT_V3_unfiltered_cleaned_split.json --save replies.json > replay.log   # once /health answers
kill -TERM %1; wait            # the capture's SIGTERM reaches the server; the record says it was interrupted
# evict the store from this client's page cache (or drop caches as root, or use another client)
python -c 'import os,sys
for f in os.scandir(sys.argv[1]):
    fd = os.open(f.path, os.O_RDONLY); os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' /mnt/aeiou-nfs/kv
# the reader: a restarted engine, the same requests
$CAPTURE -o /local/reader -- sh $K/serve.sh > serve.log 2>&1 &
python $K/replay.py ShareGPT_V3_unfiltered_cleaned_split.json --replay replies.json
kill -TERM %1; wait
for r in writer reader; do
    aeiou-trace metrics /local/$r/trace.txt --root /mnt/aeiou-nfs --metrics-block 4096 -o $C/$r.trace.metrics.json
    cp /local/$r/capture.json $C/$r.capture.json
done
```

- `--replay` puts the first server's replies into the history in place of the second
  server's own. Without it the two diverge at the second turn: a model whose KV cache was
  loaded gives other replies than one that computed it, and then the prompts hash to chunks
  the store does not have (27 of 47 were written again in an unreplayed run of the
  synthetic load).
- `PYTHONHASHSEED`: without it the restarted engine names every chunk differently and
  finds nothing. LMCache logs `Using hash algorithm: builtin` and warns about it.
- The local-disk backend (`../lmcache-local-disk`) cannot be the reader's backend. Its index
  is in memory: restarted on a filled directory, with the same names, it truncated and
  rewrote all 47 chunks and read none it had not written itself.
- The capture of 2026-11-05 under each source gives the 2026-10-02 strace capture's counts
  and bytes exactly (writer 9,664 calls on the store, 2,375,045,780 bytes read, 1,163,929,720
  written; reader 9,433, 3,535,829,744 read); the two sources differ only in what the threads'
  timing orders (reuse distances, read runs). The eBPF source sees no request of the store
  through `io_uring`: LMCache 0.5.5's `fs://` connector reads and writes through `aiofiles`'
  threads. The only rings are libuv's in the API server (939 `IORING_OP_EPOLL_CTL`) and two
  that caio's `linux_uring` makes and never submits to, testing the kernel when it is imported
  (`DESIGN_REVIEW.md` §3.98, note of 2026-11-05). The 2026-10-02 metrics predated
  `FS_API_SWEEP.md` S-18 and its raw trace was not retained; the five rows that put outside
  are within tolerance in both new captures.
- LMCache's log gives per request what the engine held, what the store had, and what was
  loaded (`Inference Engine computed tokens`, `LMCache hit tokens`, `Retrieved`); in both
  runs the chunks loaded are the hit chunks less the whole chunks the engine held.

The reader's namespace is declared `same_run` (contract 0.4): `aeiou run` refuses the
reader unless its `--seed`, `--gpus`, and parameters are the writer's.

The case's `writer.params.json` and `reader.params.json` hold the same values, the ones of
`../lmcache-local-disk/cases/sharegpt-chat/params.json` plus `meta_bytes` (the buffer is the
store's `st_blksize` since 2026-10-23, contract 0.12, 1 MiB on the traced NFS mount): `fit.py` on the
writer's two logs gives the serving run's values exactly (the same requests, and the engine
held the same of each).

The wire counts in `ABSTRACTS.md` §8 are from the synthetic load (`chat.py`, 40 requests),
a repeat of both runs without `strace`: `-yy` adds GETATTRs and WRITEs of its own
(`DESIGN_REVIEW.md` §3.53). They were not taken again under the replay.

`builder/tests/test_trace.py` repeats the dry runs against the committed metrics.
