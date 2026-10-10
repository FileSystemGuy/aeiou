# Traces behind `lmcache_shared_fs` and `lmcache_shared_fs_reader` (2026-10-02)

vLLM with LMCache's `fs://` remote backend on the loopback NFS mount of `runner/REFERENCE.md`
§7: one engine that fills an empty store, then a restarted engine in front of the filled
store, sent the same requests. What was read off the traces is in `ABSTRACTS.md` §8
("The shared store") and `DESIGN_REVIEW.md` §3.52; the committed traces are of the ShareGPT
replay of `../lmcache-local-disk` (§3.56), 300 requests with 8 conversations open: the case
`sharegpt-chat`, whose capture holds both traces' metrics and tolerance files.

Versions, model, `serve.sh`, `replay.py`, and `fit.py` are those of `../lmcache-local-disk/kit/`;
only `kit/lmcache.yaml` differs (`remote_url` in place of `local_disk`).

```
export PYTHONHASHSEED=0       # LMCache's chunk hash is Python's: two processes agree on names only with one seed
K=../lmcache-local-disk/kit                          # from this directory
C=cases/sharegpt-chat/captures/2026-10-02
TRACE='strace -f --seccomp-bpf -ttt -T -yy -e trace=%file,%desc,%process'
# the writer: an empty store
KV_BYTES=100663296 LMCACHE_CONFIG_FILE=kit/lmcache.yaml $TRACE -o writer.txt sh $K/serve.sh &
python $K/replay.py ShareGPT_V3_unfiltered_cleaned_split.json --save replies.json > replay.log   # once /health answers; then stop the server
# evict the store from this client's page cache (or drop caches as root, or use another client)
python -c 'import os,sys
for f in os.scandir(sys.argv[1]):
    fd = os.open(f.path, os.O_RDONLY); os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' /mnt/nfs/kv
# the reader: a restarted engine, the same requests
KV_BYTES=100663296 LMCACHE_CONFIG_FILE=kit/lmcache.yaml $TRACE -o reader.txt sh $K/serve.sh &
python $K/replay.py ShareGPT_V3_unfiltered_cleaned_split.json --replay replies.json
aeiou-trace metrics writer.txt --root /mnt/nfs -o $C/writer.trace.metrics.json
aeiou-trace metrics reader.txt --root /mnt/nfs -o $C/reader.trace.metrics.json
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
- LMCache's log gives per request what the engine held, what the store had, and what was
  loaded (`Inference Engine computed tokens`, `LMCache hit tokens`, `Retrieved`); in both
  runs the chunks loaded are the hit chunks less the whole chunks the engine held.

The reader's namespace is declared `same_run` (contract 0.4): `aeiou run` refuses the
reader unless its `--seed`, `--gpus`, and parameters are the writer's.

The case's `writer.params.json` and `reader.params.json` hold the same values, the ones of
`../lmcache-local-disk/cases/sharegpt-chat/params.json` plus `meta_bytes` and `buf`: `fit.py` on the
writer's two logs gives the serving run's values exactly (the same requests, and the engine
held the same of each).

The wire counts in `ABSTRACTS.md` §8 are from the synthetic load (`chat.py`, 40 requests),
a repeat of both runs without `strace`: `-yy` adds GETATTRs and WRITEs of its own
(`DESIGN_REVIEW.md` §3.53). They were not taken again under the replay.

`builder/tests/test_trace.py` repeats the dry runs against the committed metrics.
