# Trace of the real reader behind `dcp_load` (2026-10-01)

`ABSTRACTS.md` §11, row 4a: `torch.distributed.checkpoint.load` on two ranks (gloo, CPU) into
a state dict of `DTensor`s sharded over the ranks, each rank reading its own shard file,
through the loopback NFS mount of `runner/REFERENCE.md` §7. Three checkpoints were read, one
case each (`cases/<case>/`: `params.json` and `captures/2026-10-01/trace.metrics.json`):

- `small-last-6-items`: the one `../dcp-save/kit/save.py` writes (six items, the last two below the
  1 MiB buffer, so the file ends in a small item);
- `mixed-9-items`: nine items with small ones between large ones and a large one last
  (`kit/load.py --save`), which is where the buffer effects of `ABSTRACTS.md` §4 were first read off;
- `name-order-17-items`: seventeen items, `32,0.5,0.5,0.5,0.5,0.5,0.5,0.004,0.004,0.004,0.75,0.75,0.75,0.75,1.9,1.9,16`,
  enough names for their sorted order (`layer1`, `layer10`, …, `layer16`, `layer2`, …) to differ
  from the order in the file, with runs of small items longer than the buffer.

What was read off the traces is in `ABSTRACTS.md` §4 and `DESIGN_REVIEW.md` §3.47.

Versions: Python 3.12.3, torch 2.14.1+cpu, Linux 6.18.

```
C=cases/mixed-9-items                                 # from this directory
mkdir /mnt/nfs/out
python kit/load.py --save /mnt/nfs/out 2 200 32,1,64,0.004,0.004,1,32,0.004,16   # ranks, step, items in MiB
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process,io_setup,io_submit,io_getevents,io_pgetevents,io_destroy -o trace.txt \
    python kit/load.py /mnt/nfs/out 2 200 32,1,64,0.004,0.004,1,32,0.004,16
aeiou-trace metrics trace.txt --root /mnt/nfs/out \
    --instance-root <pid of rank 0> --instance-root <pid of rank 1> -o $C/captures/2026-10-01/trace.metrics.json
python kit/params.py /mnt/nfs/out/ckpt/step_000200 200 "what this is" > $C/params.json
aeiou dry-run dcp_load.ast.json --gpus 2 \
    --params-file $C/params.json --metrics-json abstract.metrics.json
aeiou-trace compare $C/captures/2026-10-01/trace.metrics.json abstract.metrics.json
```

For `small-last-6-items`: `python ../dcp-save/kit/save.py /mnt/nfs/out 2 1`, then
`kit/load.py /mnt/nfs/out 2 100 32,64,32,16,1,0.004`.

Each rank is one instance (`--instance-root`), as in `../dcp-save`. The rank pids are the
two processes that open `__0_0.distcp` and `__1_0.distcp`.

`kit/params.py` reads the checkpoint's `.metadata` for the items of rank 0's shard in file order
(storage bytes, offsets) and the order `dcp.load` takes them in.

A cold read on the wire needs the checkpoint out of the client's page cache; without root,
`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on each file after an `fsync` does it.

`builder/tests/test_trace.py` repeats the last two steps against the committed metrics.
