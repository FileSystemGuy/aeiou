"""ABSTRACTS.md §8: KV-cache serving (prefix cache offloaded to storage).

The calls are the trace of vLLM with LMCache's local-disk backend (2026-10-02, vLLM 0.30,
LMCache 0.5.5, workloads/kvcache/lmcache-local-disk). LMCache keeps its index in memory: a lookup
touches no storage. One file holds one chunk of `chunk_tokens` tokens, in one flat directory,
named by the hash of the token prefix it ends; it is written with one `write` and read with
one `read`, each inside the six calls of a Python `open` (with `direct`, LMCache's `O_DIRECT`
path: `os.open` and the data call, and an `fstat` before a read, §3.63). Only whole chunks of the prompt are
stored, when the request is prefilled; nothing is written during decode. A returning
conversation's chunks are read, all at once from a small thread pool, only when the engine
no longer holds them in GPU memory; it loses a conversation from its end, so what is read is
the chunks past the part it kept.

The lengths are those of a public chat replay (ShareGPT through the same server, 2026-10-02,
workloads/kvcache/lmcache-local-disk/kit/replay.py). A slot serves its open conversations in turn:
`reuse` has one distance, so a request has at most one continuation. A slot that is one agent
(`reuse` 1) may run side runs: after a request of its main lane, `side_lanes` chains in turn, the
first of `side` requests, then the main lane goes on (an agent waiting on its sub-agents, the
AgentX replay, DESIGN_REVIEW.md §3.64 note of 2026-11-06); what a request does at a side run is
decided by the bindings of the request before it (`x @ (r − 1)`), all of them after the draws of
the main lane, so a file without side runs keeps its fingerprints.

Exercises: `at` chains (§9.5), an `at` offset that is an expression (`kp @ (r − (r mod d + 1))`,
rule V3 as widened in contract 0.5), a mixture with a none arm, expression-selected phase names, a
chunk-realized dataset, a namespace with a size expression.
"""
from aeiou import *

w = Workload("lmcache_local_disk",
             doc="One engine per actor; `concurrency` request slots; each request continues an earlier conversation or starts one.")
P = w.P
# A case gives the values with no default (DESIGN_REVIEW.md §3.78 points 1, 4): the model's on its GPU
# (chunk_bytes, context, keep, prefill_per_token, decode_per_token, prefill_step), the deployment's (retain),
# and the load's (concurrency, warm, requests, reuse, turn_in, turn_out, sys_prompts, sys_tokens, sys_pop).
# The defaults are the library's (chunk_tokens, direct) or a load's feature, off unless a case turns it on.
w.param("concurrency", kind=int, unit="count", doc="request slots per engine [load]")
w.param("warm", kind=int, unit="count", doc="requests per slot before measurement (§9.5)")
w.param("requests", kind=int, unit="count", doc="the measured requests per slot [load]: the case's length")
w.param("chunk_tokens", 256, unit="tokens")
w.param("chunk_bytes", kind=int, unit="bytes", doc="the model's KV bytes per token × chunk_tokens (128 KiB × 256 for an 8B model); a file is exactly this, no header (traced)")
w.param("sys_prompts", kind=int, unit="count", doc="system prompts [load]")
w.param("sys_tokens", kind=int, unit="tokens", doc="median system prompt length [load, measure]")
w.param("sys_tokens_min", 1, unit="tokens", doc="bounds of the system prompt length; equal bounds give every prompt that length")
w.param("sys_tokens_max", 100_000, unit="tokens")
w.param("sys_pop", kind=Dist, doc="the system prompts' popularity, by index [load, measure]")
w.param("sys_local", True, doc="[config] the engine holds the system prompts in GPU memory and never reads their chunks (traced); false: a system prompt is a prefix like the conversation's own, and its chunks are read when the engine holds none of the conversation (a new one, or one whose `keep` draw is under a chunk: measured in the AgentX replay, where the engine held the system prompt and some of the conversation or neither, §3.63)")
w.param("direct", False, doc="[config] LMCache's `use_odirect`: every chunk opened with `O_DIRECT` by `os.open` and read or written "
        "in one call (traced 2026-10-07, §3.63); LMCache takes this path only when chunk_bytes is a multiple of the file system's "
        "block size, and is buffered otherwise, so set it only then")
w.param("sys_per_slot", False, doc="[config: load] a slot's conversations all start with the system prompt of its own index (sessions of an agent, each with its own, §3.63); false picks one per conversation by `sys_pop`")
w.param("sub_prompts", 50, unit="count", doc="sub-agent prompts: a sub-agent's prompt is its own system prompt and tools, not the main agent's (Claude Code's documentation, §3.63)")
w.param("sub_tokens", 1500, unit="tokens", doc="median sub-agent prefix length [measure]")
w.param("sub_tokens_min", 1, unit="tokens", doc="bounds of the sub-agent prefix length; equal bounds give every prefix that length")
w.param("sub_tokens_max", 100_000, unit="tokens")
w.param("sub_pop", zipf(s=1.1), doc="[measure]")
w.param("sub_per_slot", False, doc="[config: load] a slot's sub-agent conversations all start with the sub-agent prefix of its own index (one session's sub-agents); false picks one per conversation by `sub_pop`")
w.param("kind", const(0), doc="[config: load] the prefix a new conversation opens with: 0 the system prompt (the main agent's tools and system prompt), 1 a sub-agent's prefix, 2 none (measured in the AgentX corpus: a prompt is its tools, then its system prompt, then its messages, and a sub-agent's are its own, §3.63)")
w.param("first_in", const(-1), unit="tokens", doc="the tokens of a new conversation's first prompt past its prefix; -1 draws them from `turn_in` as for any request (measured in the AgentX corpus: a chain's first prompt is not a turn's growth, §3.63)")
w.param("sys_held", const(0), unit="tokens", doc="[config: GPU KV memory against the prefixes] the tokens of the system prompt the engine holds when a request arrives, apart from its conversation: a prefix many requests share stays in the engine's own cache (measured in the AgentX replay: the whole main-agent prefix in 89 of 103 requests, §3.63)")
w.param("sub_held", const(0), unit="tokens", doc="[config: GPU KV memory against the prefixes] the same for a sub-agent prefix")
w.param("reuse", kind=Dist,
        doc="none for a new conversation (0.148 of ShareGPT's requests are first turns), or requests ago: the conversations "
            "a slot has open and serves in turn [config: load]. One distance: with several, two requests can continue the same one")
w.param("keep", kind=Dist, unit="tokens",
        doc="[config: GPU KV memory against the open conversations] the tokens of a returning conversation the engine still holds, "
            "counted from its start: all of it when the draw exceeds its length, else the chunks past them are read. Drawn once per "
            "round of the open conversations (the GPU's memory is one state for all of them), so a round's loads come together")
w.param("retain", kind=int, unit="count", doc="older chunks are evicted [config: capacity]")
w.param("context", kind=int, unit="tokens", doc="[config: model] a conversation whose next prompt and reply would not fit starts anew")
w.param("turns", const(0), unit="count",
        doc="[config: load] the requests a conversation has, drawn when it starts; 0 for no limit, the conversation then ends by "
            "`reuse`'s none arm or the context. A session of many turns and a side call of one are the same slot's conversations "
            "(measured in the AgentX corpus, §3.59)")
w.param("trim", const(0), unit="tokens",
        doc="[config: load] tokens the client drops from the end of its conversation before a request (an agent editing its context; "
            "measured in the AgentX corpus, §3.59); the chunks past the kept prefix are stored again under their names, where the "
            "real store writes new files")
w.param("turn_in", kind=Dist, unit="tokens",
        doc="the tokens a request adds to its conversation's prompt [load] (ShareGPT's: user turns, twenty equal shares, each its mean)")
w.param("turn_out", kind=Dist, unit="tokens",
        doc="generated tokens [load] (ShareGPT's replies, the same way)")
w.param("prefill_per_token", kind=int, unit="ns", doc="prefill compute per prompt token: the model's on its accelerator")
w.param("decode_per_token", kind=int, unit="ns", doc="decode compute per generated token: the model's on its accelerator")
w.param("prefill_step", kind=int, unit="tokens",
        doc="[config: engine] tokens per prefill step (vLLM's max_num_batched_tokens: 8192 for an API server on a large GPU, 2048 "
            "otherwise); the chunks a step completes are stored after it, so a long prompt's writes come in bursts of prefill_step / chunk_tokens")
w.param("side", const(0), unit="count",
        doc="[config: load] after each request of a slot's main lane, the requests of the leading chain of a side run, drawn at that "
            "request; 0 none. The run serves `side_lanes` chains in turn for `side_lanes` x `side` requests, its leading chain in its "
            "first lane, then the main lane goes on (an agent waiting on its sub-agents and side calls, a run lasting about as long as "
            "its longest sub-agent: measured in the AgentX corpus and replay, where a chain resumed after them is read back after "
            "their traffic, §3.64 note of 2026-11-06). For a slot of one agent: a request whose `reuse` draw is not 1 is never in a side run")
w.param("side_lanes", const(1), unit="count",
        doc="[config: load] the chains a side run serves in turn, drawn at the main-lane request before it: each goes on `side_lanes` "
            "requests later (sub-agents beside one another), a chain that ends is followed in its lane by a new one of `side_turns`, "
            "and the run's chains end with it")
w.param("side_kind", const(1), doc="[config: load] the prefix a chain started in a side run opens with, as `kind` (a sub-agent's)")
w.param("side_turns", const(0), unit="count", doc="[config: load] the requests a chain of a side run has, as `turns`, but its leading chain's (`side`)")
w.param("think", const(0), unit="ns",
        doc="[config: load] the client's delay before a request (its user's or its agent's think time); zero sends a slot's requests back to back")

token_bytes = P.chunk_bytes // P.chunk_tokens
sysp = w.dataset("sysp", pattern="kv/sys/{id:04}/blk_{k:04}", layout="chunked", access="indexed", count=P.sys_prompts,
                 size=lognormal(median=P.sys_tokens * token_bytes, sigma=0.3,
                                min=P.sys_tokens_min * token_bytes, max=P.sys_tokens_max * token_bytes),
                 chunk=P.chunk_bytes,                                  # realized as ceil(size / chunk) block objects {k}; the last, partial one is never read
                 seed=0x5eed_da80)
subp = w.dataset("subp", pattern="kv/sub/{id:04}/blk_{k:04}", layout="chunked", access="indexed", count=P.sub_prompts,
                 size=lognormal(median=P.sub_tokens * token_bytes, sigma=0.3,
                                min=P.sub_tokens_min * token_bytes, max=P.sub_tokens_max * token_bytes),
                 chunk=P.chunk_bytes,                                  # realized as ceil(size / chunk) block objects {k}; the last, partial one is never read
                 seed=0x5eed_da84)
kv = w.namespace("kv", pattern="kv/{conv:016x}-{k:04}.pt", fields={"conv": int, "k": int},
                 size=P.chunk_bytes, seed=0x5eed_da81)                 # one flat directory (traced)


def opened(a, b, flags):
    """Python's open(): the six calls around one chunk, less the data call."""
    a.open(b, flags)
    a.fstat(b)
    a.ioctl(b, "TCGETS", expect=["ENOTTY"])
    a.lseek(b, 0, "CUR")


def chunk_read(a, b):
    """One chunk read whole: Python's open() around it, or LMCache's `O_DIRECT` path (`direct`):
    `os.open`, then `os.fdopen(fd, buffering=0).readinto`, which checks the file with `fstat`."""
    with a.when(P.direct == False):                                  # noqa: E712
        opened(a, b, "RDONLY|CLOEXEC")
        a.read(b, P.chunk_bytes)
        a.close(b)
    with a.when(P.direct == True):                                   # noqa: E712
        a.open(b, "RDONLY|DIRECT|CLOEXEC")
        a.fstat(b)
        a.read(b, P.chunk_bytes)
        a.close(b)


def chunk_write(a, b):
    """One chunk written whole: Python's open() around it, or `os.open`, `os.write`, `os.close`."""
    with a.when(P.direct == False):                                  # noqa: E712
        opened(a, b, "WRONLY|CREAT|TRUNC|CLOEXEC")
        a.write(b, P.chunk_bytes)
        a.close(b)
    with a.when(P.direct == True):                                   # noqa: E712
        a.open(b, "WRONLY|CREAT|DIRECT|CLOEXEC")
        a.write(b, P.chunk_bytes)
        a.close(b)


with w.actor("gpu") as gpu:
    with gpu.parallel("slot", P.concurrency) as slot:
        with slot.loop("r", P.warm + P.requests) as r:                # one index space (§9.5)
            d = slot.draw("d", P.reuse)
            inn = slot.draw("inn", P.turn_in)
            out = slot.draw("out", P.turn_out)
            kp = slot.draw("kp", P.keep)
            tr = slot.draw("tr", P.trim)
            tn = slot.draw("tn", P.turns)
            # side runs, as the request before left them for this one (its bindings at the loop's end): whether this request
            # is in one, the distance past d to the request it goes on from (`side_lanes` − 1 in a side run, else the side
            # requests since the main lane's last), and whether its lane has a chain to go on yet
            # only where d is 1, a slot of one agent: elsewhere nothing reads the state, and a miss in the VM's cache of
            # `x @ i` would walk it back a request at a time to the slot's start (chat-64-slots at reuse 40 overflowed the stack)
            ex = (r > 0) & (d == 1) & slot.ref("nex").at(r - 1)
            # `x mod (r + 1)` is x for 0 <= x <= r: the form rule V3 proves
            dd = d + (when(r > 0, slot.ref("nxd").at(r - 1), 0) % (r + 1))
            back = slot.let("back", (d != None) & (dd <= r) & when(r > 0, slot.ref("ngo").at(r - 1), True))   # noqa: E711
            # the conversation at r − dd goes on when it has turns left (or no limit) and the context holds this request
            more = slot.let("more", back & ((slot.ref("turns").at(r - dd) == 0) | (slot.ref("turn").at(r - dd) + 1 < slot.ref("turns").at(r - dd))))
            # the conversation's own tokens when its last request ended: that prompt and its reply
            prior = slot.let("prior", when(back, max_(slot.ref("ptoks").at(r - dd) + slot.ref("out").at(r - dd) - tr, 0), 0))
            cont = slot.let("cont", more & (when(back, slot.ref("sysblk").at(r - dd), 0) * P.chunk_tokens + prior + inn + out <= P.context))
            # the chain: conv @ r = conv @ (r − dd) or fresh
            conv = slot.let("conv", when(cont, slot.ref("conv").at(r - dd), draw(uniform64())))
            lead = when(r > 0, slot.ref("npos").at(r - 1), 0) == 0          # the side run's first request starts its leading chain
            turns = slot.let("turns", when(cont, slot.ref("turns").at(r - dd),
                                           when(ex, when(lead, slot.ref("sd").at(r - 1), slot.ref("stn").at(r - 1)), tn)))
            turn = slot.let("turn", when(cont, slot.ref("turn").at(r - dd) + 1, 0))
            sp = slot.let("sp", when(cont, slot.ref("sp").at(r - dd),
                                     when(P.sys_per_slot == True, sysp.file(slot.index % P.sys_prompts), sysp.pick(P.sys_pop))))   # noqa: E712
            kd = slot.draw("kd", P.kind)                       # last of the draws: their sites (JSON pointers) stay put
            fi = slot.draw("fi", P.first_in)
            sh = slot.draw("sh", P.sys_held)
            bh = slot.draw("bh", P.sub_held)
            kind = slot.let("kind", when(cont, slot.ref("kind").at(r - dd), when(ex, slot.ref("skd").at(r - 1), kd)))
            sb = slot.let("sb", when(cont, slot.ref("sb").at(r - dd),
                                     when(P.sub_per_slot == True, subp.file(slot.index % P.sub_prompts), subp.pick(P.sub_pop))))   # noqa: E712
            psize = slot.let("psize", when(kind == 0, sp.size, when(kind == 1, sb.size, 0)))   # the prefix the conversation opens with
            sysblk = slot.let("sysblk", psize // P.chunk_bytes)    # whole chunks inside its prefix, shared by the conversations that open with it
            pheld = slot.let("pheld", min_(when(kind == 0, sh, when(kind == 1, bh, 0)) // P.chunk_tokens, sysblk))   # whole chunks of its prefix the engine holds
            # the conversation's own tokens in this request's prompt; a new conversation starts with its prefix's tokens past its last whole chunk
            ptoks = slot.let("ptoks", when(cont, prior, (psize % P.chunk_bytes) // token_bytes) + when(cont | (fi < 0), inn, fi))
            stored = slot.let("stored", ptoks // P.chunk_tokens)       # whole chunks only
            had = slot.let("had", when(cont & (dd <= P.retain), min_(slot.ref("stored").at(r - dd), prior // P.chunk_tokens), 0))   # the store's chunks of the kept prefix
            # whole chunks the engine kept, from the start: one `keep` draw per round of the d open conversations, the one
            # made at the last request of the previous round, so a round's loads come together (§3.57)
            held = slot.let("held", when(cont, min_(prior, slot.ref("kp").at(r - (r % d + 1))) // P.chunk_tokens, 0))
            load = slot.let("load", when(had > held, had - held, 0))

            with slot.phase(when(r < P.warm, "warm", "serve")):
                slot.compute(P.think)
                with slot.when((P.sys_local == False) & (held == 0) & (kind == 0)):   # noqa: E712
                    with slot.parallel("sk", sysblk - pheld) as rd:
                        chunk_read(rd, rd.let("b", sp.chunk(pheld + rd.index)))
                with slot.when((P.sys_local == False) & (held == 0) & (kind == 1)):   # noqa: E712
                    with slot.parallel("ak", sysblk - pheld) as rd:
                        chunk_read(rd, rd.let("b", sb.chunk(pheld + rd.index)))
                with slot.parallel("lk", load) as rd:                  # the cached chunks the engine lost, all at once
                    chunk_read(rd, rd.let("b", kv.object(conv=conv, k=held + rd.index)))
                cps = P.prefill_step // P.chunk_tokens                 # chunks a prefill step completes
                with slot.loop("s", ceil_div(stored - had, cps)) as s:   # the prefill, a step at a time; its chunks stored after each
                    slot.compute(P.prefill_per_token * min_((stored - had - s * cps) * P.chunk_tokens, P.prefill_step))
                    with slot.loop("k", min_(stored, had + (s + 1) * cps), start=had + s * cps) as k:   # the step's new whole chunks
                        chunk_write(slot, slot.let("b", kv.object(conv=conv, k=k)))
                slot.compute(P.decode_per_token * out)
            # the side runs' draws and the next request's state, after everything above so its sites stay put
            sd = slot.draw("sd", P.side)
            sl = slot.draw("sl", P.side_lanes)
            slot.draw("skd", P.side_kind)
            slot.draw("stn", P.side_turns)
            pos = when(r > 0, slot.ref("npos").at(r - 1), 0)               # this request's place in its side run
            lanes = when(r > 0, slot.ref("nlanes").at(r - 1), 1)
            left = slot.let("left", when(ex, slot.ref("left").at(r - 1) - 1, sd * sl))   # side requests still to come after this one
            nex = slot.let("nex", left > 0)
            npos = slot.let("npos", when(nex, when(ex, pos + 1, 0), 0))
            nlanes = slot.let("nlanes", when(nex, when(ex, lanes, sl), 1))
            ngap = slot.let("ngap", when(ex, when(r > 0, slot.ref("ngap").at(r - 1), 0) + 1, 0))   # side requests since the main lane's last
            slot.let("nxd", when(nex, nlanes - 1, ngap))
            slot.let("ngo", (nex == False) | (npos >= nlanes))             # noqa: E712

if __name__ == "__main__":
    w.write()
