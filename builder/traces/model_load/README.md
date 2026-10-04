# Trace of the real loader behind `model_load` (2026-10-01)

`ABSTRACTS.md` §11, row 4b: `AutoModelForCausalLM.from_pretrained` on a local directory of
five safetensors shards (a GPT-2-shaped model with random weights, 437 MB, made by
`mkmodel.py`; nothing is downloaded), on the loopback NFS mount of `runner/README.md` §7.
What was read off the trace is in `ABSTRACTS.md` §4 and `DESIGN_REVIEW.md` §3.47.

Versions: Python 3.12.3, torch 2.14.1+cpu, safetensors 0.8.0, transformers 5.18.0, Linux 6.18.

```
mkdir /mnt/nfs/out
HF_HUB_OFFLINE=1 python mkmodel.py /mnt/nfs/out/model
HF_HUB_OFFLINE=1 strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt \
    python loadmodel.py --notouch /mnt/nfs/out/model
aeiou-trace metrics trace.txt --root /mnt/nfs/out -o trace.metrics.json
aeiou-params safetensors ../../../schema/examples/model_load.ast.json \
    /mnt/nfs/out/model/model-0000?-of-00005.safetensors --tp 1 -o fitted.params.json
aeiou dry-run ../../../schema/examples/model_load.ast.json --gpus 1 \
    --params-file fitted.params.json --metrics-json abstract.metrics.json
```

The library issues no `read` on a shard: it maps the file, and `strace` does not see page
faults. The trace therefore fixes the calls around the mapping (two opens per shard, the
`fadvise`, the small JSON files) and says nothing about which tensor bytes are touched; the
abstract's header and tensor reads have no counterpart in `trace.metrics.json`.

What the mapping costs on the wire was measured with `/proc/self/mountstats` deltas, cold
(without root: `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on each file after an `fsync`):
`loadmodel.py --notouch`, `loadmodel.py --touch`, `touch.py` with and without `ONE=1`, and
`aeiou run … --params-file fitted.params.json --io-backend mmap` over a dataset `aeiou datagen`
wrote behind the server. The numbers are in `DESIGN_REVIEW.md` §3.47.

`tests/test_trace.py` repeats the dry run against the committed metrics.
