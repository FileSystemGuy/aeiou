# Trace of the real server behind `kv_cache_serving` (2026-10-02)

`ABSTRACTS.md` §11, row 8: vLLM with LMCache's local-disk backend on the loopback NFS mount
of `runner/README.md` §7, a multi-turn chat load from `chat.py`. What was read off the trace
is in `ABSTRACTS.md` §8 and `DESIGN_REVIEW.md` §3.51.

Versions: Python 3.12, vLLM 0.30.0, LMCache 0.5.5, torch 2.13.0; one 8 GB GPU. The model is
`Qwen/Qwen2.5-0.5B-Instruct` (24 layers, 2 KV heads of 64, bf16: 12,288 bytes of KV per
token, so a 256-token chunk is 3,145,728 bytes).

```
# lmcache.yaml: local_disk is the directory under test
KV_BYTES=100663296 strace -f --seccomp-bpf -ttt -T -yy \
    -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_destroy,io_uring_setup,io_uring_enter \
    -o trace.txt sh serve.sh &
python chat.py --conversations 8 --turns 5 --system-prompts 2      # once /health answers
aeiou-trace metrics trace.txt --root /mnt/nfs -o trace.metrics.json
aeiou dry-run ../../../schema/examples/kv_cache_serving.ast.json --gpus 1 \
    --params fitted.params.json --metrics-json abstract.metrics.json
```

- `KV_BYTES` limits vLLM's own KV memory (96 MiB here, 7,800 tokens). Without a limit the
  engine keeps every conversation of a small load in GPU memory and LMCache is never read.
- `serve.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0`: that sampler compiles its kernels at
  first use and needs the CUDA compiler, which a machine with only the driver lacks.
- `%network` is left out: the server's HTTP and IPC traffic would be most of the trace.
- LMCache's log (`Stored … tokens`, `Retrieved … tokens`, `LMCache hit tokens: …, need to
  load: …`) gives per request what the trace gives per file.

`fitted.params.json` is written by hand from the load's shape (system prompts of 464
tokens, 197 tokens per user turn, 100 generated, every conversation returning 8 requests
later). The token counts are the server's `usage` figures that `chat.py` prints.

The wire counts in `ABSTRACTS.md` §8 are from a repeat without `strace`, the mount's block
of `/proc/self/mountstats` before and after `chat.py`, against the `rpcs` line of `aeiou run
--params fitted.params.json` on the same mount (datagen first, for the system prompts).

Not captured: a public chat replay (ShareGPT) for the reuse and length distributions, which
stay **[measure]** in the abstract; `chat.py` is a synthetic load for the call sequence.

`tests/test_trace.py` repeats the dry run against the committed metrics.
