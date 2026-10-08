# Trace of the real server behind `kv_cache_serving` (2026-10-02)

`ABSTRACTS.md` §11, row 8: vLLM with LMCache's local-disk backend on the loopback NFS mount
of `runner/REFERENCE.md` §7, under a replay of public chat conversations (`replay.py`, ShareGPT).
What was read off the trace is in `ABSTRACTS.md` §8 and `DESIGN_REVIEW.md` §3.51 (the call
sequence, from the synthetic load of `chat.py`) and §3.56 (the replay and the distributions).

Versions: Python 3.12, vLLM 0.30.0, LMCache 0.5.5, torch 2.13.0; one 8 GB GPU. The model is
`Qwen/Qwen2.5-0.5B-Instruct` (24 layers, 2 KV heads of 64, bf16: 12,288 bytes of KV per
token, so a 256-token chunk is 3,145,728 bytes).

```
# lmcache.yaml: local_disk is the directory under test
KV_BYTES=100663296 strace -f --seccomp-bpf -ttt -T -yy \
    -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_destroy,io_uring_setup,io_uring_enter \
    -o trace.txt sh serve.sh &
python replay.py ShareGPT_V3_unfiltered_cleaned_split.json --requests 300 --active 8 \
    > replay.log                                                   # once /health answers
aeiou-trace metrics trace.txt --root /mnt/nfs -o trace.metrics.json
grep "Inference Engine computed" serve.log > lmcache.log           # the server's output, colors stripped
python fit.py replay.log lmcache.log --set chunk_bytes=3145728 -o fitted.params.json
aeiou dry-run ../../../schema/examples/kv_cache_serving.ast.json --gpus 1 \
    --params-file fitted.params.json --metrics-json abstract.metrics.json
```

- `replay.py` sends the users' turns of ShareGPT (the file vLLM's own benchmarks use,
  672,837,942 bytes, SHA-256 `35f0e213…f6479ba4`; its parts joined back into 50,142 whole
  conversations) and asks for replies of the dataset's lengths. The dataset has no
  timestamps: 8 conversations are open and are served in turn, a load parameter. Its
  docstring has the rest. `chat.py` is the synthetic load the call sequence was first
  traced with; `replay.py` takes its system prompts from it.

- `KV_BYTES` limits vLLM's own KV memory (96 MiB here, 7,800 tokens). Without a limit the
  engine keeps every conversation of a small load in GPU memory and LMCache is never read.
- `serve.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0`: that sampler compiles its kernels at
  first use and needs the CUDA compiler, which a machine with only the driver lacks.
- `%network` is left out: the server's HTTP and IPC traffic would be most of the trace.
- LMCache's log (`Stored … tokens`, `Retrieved … tokens`, `LMCache hit tokens: …, need to
  load: …`) gives per request what the trace gives per file.

`fitted.params.json` is `fit.py` on the two committed logs: `replay.log` (the load's line
per request, with the server's `usage` figures) and `lmcache.log` (LMCache's line per
request, which has what the engine still held). The lengths and the share of new
conversations come from the first, `keep` from the second. `tests/test_trace.py` repeats
the fit.

The wire counts in `ABSTRACTS.md` §8 are from the synthetic load (`chat.py`, 40 requests):
a repeat without `strace`, the mount's block of `/proc/self/mountstats` before and after,
against the `rpcs` line of `aeiou run` on the same mount (datagen first, for the system
prompts). They were not taken again under the replay.

Not captured: when users send their turns (no public dataset found with both the turns and
their times), so the number of open conversations is a load parameter; and system prompts,
which ShareGPT does not have.

`tests/test_trace.py` repeats the dry run and the judgement against the committed metrics.

## The agentic load: AgentX (2026-10-02)

A second fit of the same abstracts, from a public corpus and not a replay: SemiAnalysis's
InferenceX AgentX traces (`semianalysisai/cc-traces-weka-062126` on HuggingFace, Apache-2.0,
1.8 GB, not committed), 393 Claude Code sessions whose requests carry a timestamp, a think
time, the reply's length, and the prompt's 64-token KV blocks as hash ids. `agentx.py`'s
docstring says how each parameter is read off it; `DESIGN_REVIEW.md` §3.59 what changed in
the abstracts because of it (`turns`, `prefill_step`, `trim`, `think`).

```
curl -L -o traces.jsonl https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
python agentx.py fit traces.jsonl --set chunk_bytes=33554432 -o fitted.agentx.params.json
python agentx.py reference traces.jsonl -o agentx.reference.json      # the corpus's chunk accounting
aeiou dry-run ../../../schema/examples/kv_cache_serving.ast.json --gpus 1 --seed 1 --params-file fitted.agentx.params.json
```

The dry run's `write` count against the reference's `chunks_stored` and its `read` count
against 0.91 × `chunks_hit` (the default `keep`) is the check `tests/test_trace.py` makes.
What the corpus lacks is `keep` and the call sequence under this load: `replay_agentx.py`
sends the corpus's requests to a vLLM server (a block's tokens generated from its hash id, so
the prefix structure is the corpus's) for the `strace` and LMCache's log, as `replay.py` does
for ShareGPT. ~~Written for the trace box; not run yet.~~ Run 2026-10-07 (below).

## The AgentX replay on a GPU (2026-10-07)

`DESIGN_REVIEW.md` §3.63. Qwen2.5-0.5B at 128k context (YaRN), 2 GiB of engine KV, the first
8 sessions at once, 300 requests (225 sent; 75 over the context are skipped), back to back.
LMCache's store must not evict (the corpus reference assumes it): 18 GB of chunks, so the
export is a directory on disk, not the 8 GB tmpfs of `runner/REFERENCE.md` §7.

```
export PYTHONHASHSEED=0 KV_BYTES=2147483648 MAX_MODEL_LEN=131072 LMCACHE_CONFIG_FILE=lmcache.agentx.yaml
export VLLM_ARGS='--hf-overrides {"max_position_embeddings":131072,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000.0}}'
strace -f --seccomp-bpf -ttt -T -yy \
    -e trace=%file,%desc,%process,io_setup,io_submit,io_getevents,io_destroy,io_uring_setup,io_uring_enter \
    -o trace.txt sh serve.sh > serve.log 2>&1 &
python replay_agentx.py traces.jsonl --sessions 8 --requests 300 --max-context 131072 > agentx.replay.log   # once /health answers
aeiou-trace metrics trace.txt --root /mnt/nfs -o agentx.trace.metrics.json
python agentx.py fit traces.jsonl --replay agentx.replay.log serve.log --context 131072 \
    --set chunk_bytes=3145728 --set prefill_step=2048 --set sys_local=false --set sys_per_slot=true \
    -o fitted.agentx-replay.params.json
```

- vLLM 0.30 reads `rope_parameters` (Transformers 5) and wants `max_position_embeddings`
  already scaled for YaRN; the text the model generates past 32k does not matter here (the
  prompts are tokens drawn from hash ids, the replies `ignore_eos`).
- LMCache logs a lookup at every step a request waits for KV memory, its held prefix falling
  as the running requests evict it. `agentx.py fit --replay` takes the last lookup before the
  request's first load or store. `agentx.lmcache.log` keeps just those two lines per request
  (441 lines of the server's 32 MB; `agentx.replayed` parses both the same).
- The kit's pair (`agentx.trace`, `fitted.agentx-replay.params.json`) is judged in
  `tests/test_trace.py` and accepted; the corpus is not committed, so the fit is repeated by
  the command above, not by the tests.
- `lmcache.agentx.odirect.yaml` is the same with LMCache's `use_odirect`, the second run
  (`agentx-odirect.*`, fitted with `--set direct=true` added): its calls are the abstract's
  `direct` path. That pair is not accepted, one row recorded outside with its reason in
  `agentx-odirect.trace.tolerances.json` (the server's admission order, §3.63).

