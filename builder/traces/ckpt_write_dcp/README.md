# Trace of the real writer behind `ckpt_write_dcp` (2026-10-01)

`ABSTRACTS.md` §11, row 3: `torch.distributed.checkpoint.save` on two ranks (gloo, CPU), a
state dict of six `DTensor`s sharded over the ranks (the FSDP2 shape: one write item per
parameter, half of it on each rank), two checkpoints, written through the loopback NFS mount
of `runner/README.md` §7. `save.py --torch-save` writes the same tensors with `torch.save`
on one rank, the variant of `ABSTRACTS.md` §3. What was read off the traces is in
`ABSTRACTS.md` §3 and `DESIGN_REVIEW.md` §3.45.

Versions: Python 3.12.3, torch 2.14.1+cpu, Linux 6.18.

```
mkdir /mnt/nfs/out
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt \
    python save.py /mnt/nfs/out 2 2                   # ranks, checkpoints
aeiou-trace metrics trace.txt --root /mnt/nfs/out \
    --instance-root <pid of rank 0> --instance-root <pid of rank 1> -o trace.metrics.json
aeiou dry-run ../../../schema/examples/ckpt_write_dcp.ast.json --gpus 2 \
    --params fitted.params.json --metrics-json abstract.metrics.json
aeiou-trace compare trace.metrics.json abstract.metrics.json
```

Each rank is one instance (`--instance-root`): the abstract's reuse distance is per GPU, and
without it the other rank's writes fall between a rank's own. The rank pids are the two
children of the launcher; they are the processes that open `__0_0.distcp` and `__1_0.distcp`.

`tests/test_trace.py` repeats the last two steps against the committed metrics.
