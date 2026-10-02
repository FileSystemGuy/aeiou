# Trace of the real reader behind `ckpt_restore` (2026-10-01)

`ABSTRACTS.md` §11, row 4a: `torch.distributed.checkpoint.load` on two ranks (gloo, CPU) into
a state dict of `DTensor`s sharded over the ranks, each rank reading its own shard file,
through the loopback NFS mount of `runner/README.md` §7. Three checkpoints were read:

- `small-last`: the one `../ckpt_write_dcp/save.py` writes (six items, the last two below the
  1 MiB buffer, so the file ends in a small item);
- `mixed`: nine items with small ones between large ones and a large one last
  (`load.py --save`), which is where the buffer effects of `ABSTRACTS.md` §4 were first read off;
- `name-order`: seventeen items, `32,0.5,0.5,0.5,0.5,0.5,0.5,0.004,0.004,0.004,0.75,0.75,0.75,0.75,1.9,1.9,16`,
  enough names for their sorted order (`layer1`, `layer10`, …, `layer16`, `layer2`, …) to differ
  from the order in the file, with runs of small items longer than the buffer.

What was read off the traces is in `ABSTRACTS.md` §4 and `DESIGN_REVIEW.md` §3.47.

Versions: Python 3.12.3, torch 2.14.1+cpu, Linux 6.18.

```
mkdir /mnt/nfs/out
python load.py --save /mnt/nfs/out 2 200 32,1,64,0.004,0.004,1,32,0.004,16   # ranks, step, items in MiB
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.mixed.txt \
    python load.py /mnt/nfs/out 2 200 32,1,64,0.004,0.004,1,32,0.004,16
aeiou-trace metrics trace.mixed.txt --root /mnt/nfs/out \
    --instance-root <pid of rank 0> --instance-root <pid of rank 1> -o trace.mixed.metrics.json
python params.py /mnt/nfs/out/ckpt/step_000200 200 "what this is" > fitted.mixed.params.json
aeiou dry-run ../../../schema/examples/ckpt_restore.ast.json --gpus 2 \
    --params fitted.mixed.params.json --metrics-json abstract.metrics.json
aeiou-trace compare trace.mixed.metrics.json abstract.metrics.json
```

For `small-last`: `python ../ckpt_write_dcp/save.py /mnt/nfs/out 2 1`, then
`load.py /mnt/nfs/out 2 100 32,64,32,16,1,0.004`.

Each rank is one instance (`--instance-root`), as in `../ckpt_write_dcp`. The rank pids are the
two processes that open `__0_0.distcp` and `__1_0.distcp`.

`params.py` reads the checkpoint's `.metadata` for the items of rank 0's shard in file order
(storage bytes, offsets) and the order `dcp.load` takes them in.

A cold read on the wire needs the checkpoint out of the client's page cache; without root,
`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on each file after an `fsync` does it.

`tests/test_trace.py` repeats the last two steps against the committed metrics.
