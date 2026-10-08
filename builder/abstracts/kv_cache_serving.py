"""ABSTRACTS.md §8: KV-cache serving (prefix cache offloaded to storage).

The calls are the trace of vLLM with LMCache's local-disk backend (2026-10-02, vLLM 0.30,
LMCache 0.5.5, builder/traces/kv_cache_serving). LMCache keeps its index in memory: a lookup
touches no storage. One file holds one chunk of `chunk_tokens` tokens, in one flat directory,
named by the hash of the token prefix it ends; it is written with one `write` and read with
one `read`, each inside the six calls of a Python `open`. Only whole chunks of the prompt are
stored, when the request is prefilled; nothing is written during decode. A returning
conversation's chunks are read, all at once from a small thread pool, only when the engine
no longer holds them in GPU memory; it loses a conversation from its end, so what is read is
the chunks past the part it kept.

The lengths are those of a public chat replay (ShareGPT through the same server, 2026-10-02,
builder/traces/kv_cache_serving/replay.py). A slot serves its open conversations in turn:
`reuse` has one distance, so a request has at most one continuation.

Exercises: `at` chains (§9.5), an `at` offset that is an expression (`kp @ (r − (r mod d + 1))`,
rule V3 as widened in contract 0.5), a mixture with a none arm, expression-selected phase names, a
chunk-realized dataset, a namespace with a size expression.
"""
from aeiou import *

w = Workload("kv_cache_serving",
             doc="One engine per actor; `concurrency` request slots; each request continues an earlier conversation or starts one.")
P = w.P
w.param("concurrency", 64, unit="count")
w.param("warm", 2000, unit="count", doc="requests per slot before measurement (§9.5)")
w.param("requests", 20000, unit="count")
w.param("chunk_tokens", 256, unit="tokens")
w.param("chunk_bytes", 32 * MiB, unit="bytes", doc="128 KiB/token × 256 [config]; a file is exactly this, no header (traced)")
w.param("sys_prompts", 50, unit="count")
w.param("sys_tokens", 1500, unit="tokens", doc="median system prompt length [measure]")
w.param("sys_tokens_min", 1, unit="tokens", doc="bounds of the system prompt length; equal bounds give every prompt that length")
w.param("sys_tokens_max", 100_000, unit="tokens")
w.param("sys_pop", zipf(s=1.1), doc="[measure]")
w.param("sys_local", True, doc="[config] the engine holds the system prompts in GPU memory and never reads their chunks (traced); false: a system prompt is a prefix like the conversation's own, and its chunks are read when the engine holds none of the conversation (a new one, or one whose `keep` draw is under a chunk: measured in the AgentX replay, where the engine held the system prompt and some of the conversation or neither, §3.63)")
w.param("sys_per_slot", False, doc="[config: load] a slot's conversations all start with the system prompt of its own index (sessions of an agent, each with its own, §3.63); false picks one per conversation by `sys_pop`")
w.param("reuse", mixture((0.15, none), (0.85, const(40))),
        doc="none for a new conversation (measured: 0.148 of ShareGPT's requests are first turns), or requests ago: the conversations "
            "a slot has open and serves in turn [config: load]. One distance: with several, two requests can continue the same one")
w.param("keep", empirical({0: 91, 1_000_000: 9}), unit="tokens",
        doc="[config: GPU KV memory against the open conversations] the tokens of a returning conversation the engine still holds, "
            "counted from its start: all of it when the draw exceeds its length, else the chunks past them are read. Drawn once per "
            "round of the open conversations (the GPU's memory is one state for all of them), so a round's loads come together")
w.param("retain", 5000, unit="count", doc="older chunks are evicted [config: capacity]")
w.param("context", 8192, unit="tokens", doc="[config: model] a conversation whose next prompt and reply would not fit starts anew")
w.param("turns", const(0), unit="count",
        doc="[config: load] the requests a conversation has, drawn when it starts; 0 for no limit, the conversation then ends by "
            "`reuse`'s none arm or the context. A session of many turns and a side call of one are the same slot's conversations "
            "(measured in the AgentX corpus, §3.59)")
w.param("trim", const(0), unit="tokens",
        doc="[config: load] tokens the client drops from the end of its conversation before a request (an agent editing its context; "
            "measured in the AgentX corpus, §3.59); the chunks past the kept prefix are stored again under their names, where the "
            "real store writes new files")
w.param("turn_in", empirical([1, 4, 6, 7, 9, 10, 11, 13, 15, 17, 19, 22, 26, 31, 39, 50, 68, 107, 214, 944]), unit="tokens",
        doc="the tokens a request adds to its conversation's prompt (measured: ShareGPT user turns, twenty equal shares, each its mean)")
w.param("turn_out", empirical([12, 34, 62, 92, 119, 148, 175, 201, 225, 248, 271, 294, 317, 342, 370, 407, 454, 519, 615, 785]), unit="tokens",
        doc="generated tokens (measured: ShareGPT replies, the same way)")
w.param("prefill_per_token", 40 * us, unit="ns")
w.param("decode_per_token", 12 * ms, unit="ns")
w.param("prefill_step", 8192, unit="tokens",
        doc="[config: engine] tokens per prefill step (vLLM's max_num_batched_tokens: 8192 for an API server on a large GPU, 2048 "
            "otherwise); the chunks a step completes are stored after it, so a long prompt's writes come in bursts of prefill_step / chunk_tokens")
w.param("think", const(0), unit="ns",
        doc="[config: load] the client's delay before a request (its user's or its agent's think time); zero sends a slot's requests back to back")

token_bytes = P.chunk_bytes // P.chunk_tokens
sysp = w.dataset("sysp", pattern="kv/sys/{id:04}/blk_{k:04}", count=P.sys_prompts,
                 size=lognormal(median=P.sys_tokens * token_bytes, sigma=0.3,
                                min=P.sys_tokens_min * token_bytes, max=P.sys_tokens_max * token_bytes),
                 chunk=P.chunk_bytes,                                  # realized as ceil(size / chunk) block objects {k}; the last, partial one is never read
                 seed=0x5eed_da80)
kv = w.namespace("kv", pattern="kv/{conv:016x}-{k:04}.pt", fields={"conv": int, "k": int},
                 size=P.chunk_bytes, seed=0x5eed_da81)                 # one flat directory (traced)


def opened(a, b, flags):
    """Python's open(): the six calls around one chunk, less the data call."""
    a.open(b, flags)
    a.fstat(b)
    a.ioctl(b, "TCGETS", expect=["ENOTTY"])
    a.lseek(b, 0, "CUR")


with w.actor("gpu") as gpu:
    with gpu.parallel("slot", P.concurrency) as slot:
        with slot.loop("r", P.warm + P.requests) as r:                # one index space (§9.5)
            d = slot.draw("d", P.reuse)
            inn = slot.draw("inn", P.turn_in)
            out = slot.draw("out", P.turn_out)
            kp = slot.draw("kp", P.keep)
            tr = slot.draw("tr", P.trim)
            tn = slot.draw("tn", P.turns)
            back = slot.let("back", (d != None) & (d <= r))            # noqa: E711
            # the conversation at r − d goes on when it has turns left (or no limit) and the context holds this request
            more = slot.let("more", back & ((slot.ref("turns").at(r - d) == 0) | (slot.ref("turn").at(r - d) + 1 < slot.ref("turns").at(r - d))))
            # the conversation's own tokens when its last request ended: that prompt and its reply
            prior = slot.let("prior", when(back, max_(slot.ref("ptoks").at(r - d) + slot.ref("out").at(r - d) - tr, 0), 0))
            cont = slot.let("cont", more & (when(back, slot.ref("sysblk").at(r - d), 0) * P.chunk_tokens + prior + inn + out <= P.context))
            # the chain: conv @ r = conv @ (r − d) or fresh
            conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), draw(uniform64())))
            turns = slot.let("turns", when(cont, slot.ref("turns").at(r - d), tn))
            turn = slot.let("turn", when(cont, slot.ref("turn").at(r - d) + 1, 0))
            sp = slot.let("sp", when(cont, slot.ref("sp").at(r - d),
                                     when(P.sys_per_slot == True, sysp.file(slot.index % P.sys_prompts), sysp.pick(P.sys_pop))))   # noqa: E712
            sysblk = slot.let("sysblk", sp.size // P.chunk_bytes)      # whole chunks inside the system prompt, shared by its conversations
            # the conversation's own tokens in this request's prompt; a new conversation starts with the system prompt's tokens past its last whole chunk
            ptoks = slot.let("ptoks", when(cont, prior, (sp.size % P.chunk_bytes) // token_bytes) + inn)
            stored = slot.let("stored", ptoks // P.chunk_tokens)       # whole chunks only
            had = slot.let("had", when(cont & (d <= P.retain), min_(slot.ref("stored").at(r - d), prior // P.chunk_tokens), 0))   # the store's chunks of the kept prefix
            # whole chunks the engine kept, from the start: one `keep` draw per round of the d open conversations, the one
            # made at the last request of the previous round, so a round's loads come together (§3.57)
            held = slot.let("held", when(cont, min_(prior, slot.ref("kp").at(r - (r % d + 1))) // P.chunk_tokens, 0))
            load = slot.let("load", when(had > held, had - held, 0))

            with slot.phase(when(r < P.warm, "warm", "serve")):
                slot.compute(P.think)
                with slot.when((P.sys_local == False) & (held == 0)):                 # noqa: E712
                    with slot.parallel("sk", sysblk) as rd:
                        b = rd.let("b", sp.chunk(rd.index))
                        opened(rd, b, "RDONLY|CLOEXEC")
                        rd.read(b, P.chunk_bytes)
                        rd.close(b)
                with slot.parallel("lk", load) as rd:                  # the cached chunks the engine lost, all at once
                    b = rd.let("b", kv.object(conv=conv, k=held + rd.index))
                    opened(rd, b, "RDONLY|CLOEXEC")
                    rd.read(b, P.chunk_bytes)
                    rd.close(b)
                cps = P.prefill_step // P.chunk_tokens                 # chunks a prefill step completes
                with slot.loop("s", ceil_div(stored - had, cps)) as s:   # the prefill, a step at a time; its chunks stored after each
                    slot.compute(P.prefill_per_token * min_((stored - had - s * cps) * P.chunk_tokens, P.prefill_step))
                    with slot.loop("k", min_(stored, had + (s + 1) * cps), start=had + s * cps) as k:   # the step's new whole chunks
                        b = slot.let("b", kv.object(conv=conv, k=k))
                        opened(slot, b, "WRONLY|CREAT|TRUNC|CLOEXEC")
                        slot.write(b, P.chunk_bytes)
                        slot.close(b)
                slot.compute(P.decode_per_token * out)

if __name__ == "__main__":
    w.write()
