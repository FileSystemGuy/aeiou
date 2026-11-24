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

Since 2026-11-23 (later the same day; another sample again) every chunk of the hit that G2 lacks is
loaded, the ones the GPU already holds too, as the multi-process server does (read from its
source, not traced; `DESIGN_REVIEW.md` §3.100, the note "What the multi-process server loads"):
88,707 files written to each tier, 320,958 chunks read from G3 and 734,220 from G4, 69,133 files
unlinked in G3, 2,190,010 stats in each tier, 32.2 TiB read ~~in the hour~~ over the whole plan (see the correction below). That is 5.9 chunks read
for every file written where the lab's capture has 0.86, at other sizes and sessions; the count
is the rule's consequence here and is not held to a capture.

And the load is all or nothing (contract 0.27, the same day, the same source): the server takes a
G2 buffer for every chunk it would load, and when G2 has not the room nothing is loaded, the hit
is G2's own, and the engine computes and stores the rest again. Another sample: 88,866 files
written to each tier, 348,900 chunks read from G3 and 655,228 from G4, 69,278 files unlinked in
G3, 2,096,859 stats in each tier, 30.6 TiB read ~~in the hour~~ over the whole plan, which is not an hour: the sessions arrive over the hour `window_s` and run to their ends, the last at 51.9 hours of planned time at seed 1 (corrected 2026-11-23, found by running it; the first planned hour holds 46,217 of the 1,004,128 reads and 72,052 of the 177,732 writes, about 1.4 TiB read and 2.2 TiB written). At these sizes the rule seldom
applies (the largest load of one request is 2,068 chunks, G2 holding 15,680); G2's room is the
plan's at the request's instant, the bytes of loads in flight during a run not in it.

Since contract 0.28 (2026-11-23, the same day; another sample again, the statements having
changed) the lookup is written from sets: what each of G2, G3, and G4 holds of the chunks past
G2's hit, the hit the leading run of their union, G3 loading the chunks of it G2 lacks and G4
those neither has. The rules are the ones above, now lines of the abstract; the probe form,
given the same number of statements, plans the same ops to the last one (fingerprint
`8edc49c45a97b621` both ways). At seed 1: 89,958 files written to each tier, 984,977 chunks
read from the two, 69,000 files unlinked in G3, 2,161,935 stats in each tier, 30.1 TiB read
over the plan's 51.9 hours.

What LMCache does with two adapters is read from its source (0.5.5, `v1/distributed/
storage_controllers/`) and, since 2026-11-22, seen in a capture of its multi-process server
(`kit/captures/agentx-mp/`, the mechanism's README): every chunk G2 newly takes is stored in each
adapter that lacks it; a lookup past
G2's hit asks both, the hit is the leading run G2 or some adapter holds, and each chunk of it
G2 lacks is loaded, whatever the GPU holds, by the first adapter that holds it (G3 before G4),
which alone is touched for it, all of them when G2 has room and none when it has not; each adapter
evicts by its own rule, so a chunk G3 dropped is still found in G4. Two things are not settled:

- ~~The calls on a chunk file are the in-process connector's, as in `agentx-tiered`, while two L2
  adapters exist only in LMCache's multi-process server, whose `fs` adapter makes other calls (a
  `stat` before every store and every lookup, a temporary file and a `rename`); no capture of it
  exists (brief item 74).~~ Since 2026-11-23 the calls on a chunk file are the multi-process
  server's `fs_native` adapter's, as captured: a temporary file and a `rename` for a store, open,
  read, close for a load, a chunk the adapter holds never rewritten; and the adapter's `stat`
  of every key looked up, in both tiers, and before every store; and its four workers carry
  its lookups and loads as well as its stores, the request's spans waiting for them, `lookup` and then `load`, the loads submitted only once
  every adapter's lookup has answered (contract
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
