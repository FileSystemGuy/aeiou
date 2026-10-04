# Trace of the real loader behind `train_large_samples` (2026-10-01)

`ABSTRACTS.md` §11, row 2: `np.load(path, allow_pickle=True)["x"]` per sample, as upstream
DLIO's `npz_reader.py` (argonne-lcf) issues it and as any NumPy user does, through a PyTorch `DataLoader` with two workers, on the loopback
NFS mount of `runner/README.md` §7 (v4.2, rsize 1 MiB). The MLCommons fork of DLIO reads `.npz` differently
(`DESIGN_REVIEW.md` §3.46) and is not what this models. The corpus is what DLIO's
`npz_generator.py` writes: `np.savez(path, x=<uint8 volume>, y=<labels>)`. What was read off
the trace is in `ABSTRACTS.md` §2 and `DESIGN_REVIEW.md` §3.44.

Versions: Python 3.12.3, NumPy 2.5.2, torch 2.14.1+cpu, Linux 6.18.

| File | What |
|---|---|
| `mkcorpus.py` | writes N archives at `train/{id div 10000:05}/sample_{id:09}.npz`, sizes near normal(mean, sd) |
| `train.py` | the traced application |
| `fitted.140MiB.params.json`, `trace.140MiB.metrics.json` | 16 files of about 140 MiB, batch 2, 2 epochs: the reference shape |
| `fitted.8MiB.params.json`, `trace.8MiB.metrics.json` | 256 files of about 8 MiB, batch 4, 2 epochs: enough files for the reuse distance to mean something |

```
python mkcorpus.py /srv/export/corpus 16                      # or: … 256 8388608 262144
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt \
    python train.py /mnt/nfs/corpus/train 2 2 2 x             # or: … 4 2 2 x
aeiou-trace metrics trace.txt --root /mnt/nfs/corpus -o trace.metrics.json
aeiou dry-run ../../../schema/examples/train_large_samples.ast.json --gpus 1 \
    --params-file fitted.140MiB.params.json --metrics-json abstract.metrics.json
aeiou-trace compare trace.metrics.json abstract.metrics.json
```

`tests/test_trace.py` repeats the last two steps against the committed metrics.

The script's `glob` (one `scandir` per directory) is the abstract's optional `enumerate`
phase since 2026-10-02. Wire counts are taken from a run without `strace`: `-yy` adds about
one GETATTR per READ (`DESIGN_REVIEW.md` §3.53).
