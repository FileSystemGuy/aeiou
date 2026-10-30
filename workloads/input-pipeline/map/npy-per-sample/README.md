# Trace of the real loader behind `npy_per_sample` (2026-10-01)

`ABSTRACTS.md` §11, row 2: `np.load(path, allow_pickle=True)["x"]` per sample, as upstream
DLIO's `npz_reader.py` (argonne-lcf) issues it and as any NumPy user does, through a PyTorch `DataLoader` with two workers, on the loopback
NFS mount of `runner/REFERENCE.md` §7 (v4.2, rsize 1 MiB). The MLCommons fork of DLIO reads `.npz` differently
(one whole-file read per sample, nothing decoded, and its worker pre-warm re-reads the corpus 1.3–2.5× per run;
measured against this loader 2026-10-28, `DESIGN_REVIEW.md` §3.46) and is not what this models. The corpus is what DLIO's
`npz_generator.py` writes: `np.savez(path, x=<uint8 volume>, y=<labels>)`. What was read off
the trace is in `ABSTRACTS.md` §2 and `DESIGN_REVIEW.md` §3.44.

Versions: Python 3.12.3, NumPy 2.5.2, torch 2.14.1+cpu, Linux 6.18.

| File | What |
|---|---|
| `kit/mkcorpus.py` | writes N archives at `train/{id div 10000:05}/sample_{id:09}.npz`, sizes near normal(mean, sd) |
| `kit/train.py` | the traced application |
| `cases/npy-140mib/`: `params.json`, `captures/2026-10-01/trace.metrics.json` | 16 files of about 140 MiB, batch 2, 2 epochs: the reference shape |
| `cases/npy-8mib/`: `params.json`, `captures/2026-10-01/trace.metrics.json` | 256 files of about 8 MiB, batch 4, 2 epochs: enough files for the reuse distance to mean something |
| `jobs/unet3d.params.json` | the Unet3D job, every accelerator's: the WG's sample sizes (provisional), workers, prefetch, steps, and the corpus size (`DESIGN_REVIEW.md` §3.81) |
| `cases/unet3d-a100/`, `cases/unet3d-h100/`, `cases/unet3d-b200/`: `<accelerator>.params.json` | use cases: the Unet3D job on each accelerator, its batch and step time the WG's (B200's the H100's halved, not measured); `aeiou run --case unet3d-h100` |

```
C=cases/npy-140mib                                            # or: cases/npy-8mib; from this directory
python kit/mkcorpus.py /srv/export/corpus 16                  # or: … 256 8388608 262144
strace -f -ttt -T -yy -v -e trace=%file,%desc,%network,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt \
    python kit/train.py /mnt/nfs/corpus/train 2 2 2 x         # or: … 4 2 2 x
aeiou-trace metrics trace.txt --root /mnt/nfs/corpus -o $C/captures/2026-10-01/trace.metrics.json
aeiou dry-run npy_per_sample.ast.json --gpus 1 \
    --params-file $C/params.json --metrics-json abstract.metrics.json
aeiou-trace compare $C/captures/2026-10-01/trace.metrics.json abstract.metrics.json
```

`builder/tests/test_trace.py` repeats the last two steps against the committed metrics.

The trace above is of a 1 MiB buffer, where NumPy's 256 KiB chunks are drained from whole buffer
fills. On storage whose `st_blksize` is smaller than a chunk (4096 on ext4, XFS, and tmpfs),
CPython's `BufferedReader` reads the whole blocks of each chunk past its buffer and then refills
it: per chunk a read of 252 KiB (248 KiB for the first), a 4 KiB refill, and the tell; the npy
header takes a second 4 KiB fill, since zipfile reads at least 4096 bytes. Traced 2026-10-30
with `np.load` alone (NumPy 2.5.4, Python 3.12.13, no DataLoader) over three archives on a local
file system and held to that trace by `runner/aeiou/tests/golden.rs`
(`npy_per_sample_under_a_small_buffer`); the abstract divided by zero there until then. A
buffer between 4 KiB and 256 KiB is not traced (the second header fill would not happen).

The script's `glob` (one `scandir` per directory) is the abstract's optional `walk`
phase since 2026-10-02 (`enumerate` until 2026-10-13). At scale a job lists its corpus from an
inventory instead: the optional `inventory` phase reads `train/inventory.txt`, whose shape is
`[measure]` until a job that reads one is traced (`DESIGN_REVIEW.md` §3.73). Wire counts are taken from a run without `strace`: `-yy` adds about
one GETATTR per READ (`DESIGN_REVIEW.md` §3.53).
