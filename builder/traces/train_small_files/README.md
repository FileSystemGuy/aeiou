# Trace of the real loader behind `train_small_files` (2026-10-01)

`ABSTRACTS.md` §11, row 1: PyTorch `DataLoader` over `torchvision.datasets.ImageFolder`,
two workers, taken on the loopback NFS mount of `runner/README.md` §7 (v4.2, rsize 1 MiB).
What was read off it is in `ABSTRACTS.md` §1 and `DESIGN_REVIEW.md` §3.43.

Versions: Python 3.12.3, torch 2.14.1+cpu, torchvision 0.29.1+cpu, Pillow 12.3.0, Linux 6.18.

| File | What |
|---|---|
| `mkcorpus.py` | writes 3,200 real JPEGs at `train/{id div 1300:05}/img_{id:09}.jpg`, sizes near lognormal(110 KiB, 0.45) |
| `train.py` | the traced application: `ImageFolder` + `DataLoader(shuffle, drop_last)`, batch 16, 2 workers, 2 epochs = 400 steps |
| `fitted.params.json` | the abstract's parameters for that run |
| `trace.metrics.json` | `aeiou-trace metrics` of the trace (the trace itself is 34 MB and is not kept) |

```
python mkcorpus.py /srv/export/corpus 3200            # behind the server
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt \
    python train.py /mnt/nfs/corpus/train 16 2 2      # through the mount
aeiou-trace metrics trace.txt --root /mnt/nfs/corpus -o trace.metrics.json
aeiou dry-run ../../../schema/examples/train_small_files.ast.json --gpus 1 \
    --params fitted.params.json --metrics-json abstract.metrics.json
aeiou-trace compare trace.metrics.json abstract.metrics.json
```

`tests/test_trace.py` repeats the last two steps against the committed metrics, so a change
to the abstract that moves it away from the application fails a test.
