# `agentx-tiered-g4`: an unsettled case of `lmcache_local_disk_tiered_g4`

A what-if of `agentx-tiered` (brief item 78): the same sessions on the same engine, G1, and G2,
with LMCache's two L2 adapters active together, G3 on a node-local drive and G4 on shared storage,
each a directory under test of its own (namespaces `kv` and `kv4`; `--endpoint kv4=DIR` places
G4's on the shared system, `--root` or `--endpoint kv=DIR` G3's on the node's drive).

G3 is a 960 GB drive (28,610 chunks) and G4 7.68 TB (228,881 chunks). These two sizes are this
project's, chosen so that both tiers are exercised within the run, not the WG's: at them G3
reaches its watermark and unlinks most of what it took, G4 never reaches its own, and so G4 serves
the chunks G3 dropped. At seed 1 and one GPU: 116,456 files written to each tier, 205,774 chunks read
from G3 and 152,601 from G4, 96,453 files unlinked in G3 and none in G4 (2026-11-23, the chunk-file
calls changed; before, another sample: 88,707, 83,517, 35,080, and 69,133). With the adapters'
stats, the same day and another sample again: 120,589 files written to each tier, 332,164 chunks
read from G3 and 161,455 from G4, 101,298 files unlinked in G3, and 5,740,435 stats in each tier,
48 for every file written where the lab's capture has 7 or 8: most lookups stat one to three keys
and a tenth more than 600, a prompt whose chunks G2 has dropped, G2 here holding 15,680 chunks
against prompts of up to some thousands. That count follows from the lookup's rule as read from
the source (every key past G2's hit, in every adapter) and is not itself held to a capture.

What LMCache does with two adapters is read from its source (0.5.5, `v1/distributed/
storage_controllers/`) and, since 2026-11-22, seen in a capture of its multi-process server
(`kit/captures/agentx-mp/`, the mechanism's README): every chunk G2 newly takes is stored in each
adapter that lacks it; a lookup past
G2's hit asks both, the hit is the leading run some adapter holds, and each chunk of it is loaded
by the first adapter that holds it (G3 before G4), which alone is touched for it; each adapter
evicts by its own rule, so a chunk G3 dropped is still found in G4. Two things are not settled:

- ~~The calls on a chunk file are the in-process connector's, as in `agentx-tiered`, while two L2
  adapters exist only in LMCache's multi-process server, whose `fs` adapter makes other calls (a
  `stat` before every store and every lookup, a temporary file and a `rename`); no capture of it
  exists (brief item 74).~~ Since 2026-11-23 the calls on a chunk file are the multi-process
  server's `fs_native` adapter's, as captured: a temporary file and a `rename` for a store, open,
  read, close for a load, a chunk the adapter holds never rewritten; and the adapter's `stat`
  of every key looked up, in both tiers, and before every store; and its four workers carry
  its lookups and loads as well as its stores, the request's span waiting for them (contract
  0.26). Not so: its unlinks, on eviction lanes of their own. The adapter's `O_DIRECT` option was not traced, and the
  abstract has no `direct` parameter.
- Each adapter has workers of its own (`store` and `store4`, `store_lanes` each) for its
  lookups, loads, and stores, so a slow tier does not hold the other's writes; each pool's unfinished bytes are bounded
  by G2's size separately, where LMCache has one G2.

Its sample is not `agentx-tiered`'s: an abstract's draws are keyed by where they stand in the
document, and this one has statements the other lacks, so the same seed gives other sessions.

No capture of its own. The planned trace at seed 1 was checked for consistency (no read of a file not yet
written or already unlinked, in planned order); the probe over both tiers is held by the runner's
tests (`evict.rs`).
