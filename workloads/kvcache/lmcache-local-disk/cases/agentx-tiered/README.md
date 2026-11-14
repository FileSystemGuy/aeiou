# `agentx-tiered`: an unsettled case of `lmcache_local_disk_tiered`

The AgentX session generator (`kit/agentx.py sessions`, fitted to the whole corpus:
`kit/sessions.fit.json`) on one engine with the whole cache hierarchy modelled, G1 to G3 at the
WG's example sizes (`DESIGN_REVIEW.md` §3.99, note "Tier names"): G1 is 180 GiB of HBM less
Llama 3.1 8B's bf16 weights (21,125 blocks of 64 tokens), G2 490 GiB of host DRAM (15,680
chunks), G3 a 7.68 TB drive (228,881 chunks), at the model's 128 KiB of KV a token.

Its kind is not settled: `aeiou run` refuses a timeline until brief item 72 (7) is built, so the
case is checked and dry-run only; 40 sessions over an hour on 16 slots is the scratch port's load,
not the WG's operating point (item 72 (1), the ramp, and item 74's retargeting decide it); and the
calls on a chunk file are the in-process connector's, until item 74 captures the multi-process
`fs_native` adapter. At these sizes G3 never reaches its watermark within the run, so no file is
unlinked; G1 and G2 hold far less than the sessions' contexts, so most reuse is read from G3.

No capture: the tiers' behaviour is held to LMCache 0.5.5's own `LRUEvictionPolicy` request by
request (`DESIGN_REVIEW.md` §3.99, addendum "Built"), not to a trace.
