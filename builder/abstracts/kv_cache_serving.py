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

Exercises: `at` chains (§9.5), a mixture with a none arm, expression-selected phase names, a
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
w.param("sys_local", True, doc="[config] the engine holds the system prompts in GPU memory and never reads their chunks (traced); false for an engine that starts cold beside a filled cache")
w.param("reuse", mixture((0.15, none), (0.85, const(40))),
        doc="none for a new conversation (measured: 0.148 of ShareGPT's requests are first turns), or requests ago: the conversations "
            "a slot has open and serves in turn [config: load]. One distance: with several, two requests can continue the same one")
w.param("keep", empirical({0: 91, 1_000_000: 9}), unit="tokens",
        doc="[config: GPU KV memory against the open conversations] the tokens of a returning conversation the engine still holds, "
            "counted from its start: all of it when the draw exceeds its length, else the chunks past them are read")
w.param("retain", 5000, unit="count", doc="older chunks are evicted [config: capacity]")
w.param("context", 8192, unit="tokens", doc="[config: model] a conversation whose next prompt and reply would not fit starts anew")
w.param("turn_in", empirical([1, 4, 6, 7, 9, 10, 11, 13, 15, 17, 19, 22, 26, 31, 39, 50, 68, 107, 214, 944]), unit="tokens",
        doc="the tokens a request adds to its conversation's prompt (measured: ShareGPT user turns, twenty equal shares, each its mean)")
w.param("turn_out", empirical([12, 34, 62, 92, 119, 148, 175, 201, 225, 248, 271, 294, 317, 342, 370, 407, 454, 519, 615, 785]), unit="tokens",
        doc="generated tokens (measured: ShareGPT replies, the same way)")
w.param("prefill_per_token", 40 * us, unit="ns")
w.param("decode_per_token", 12 * ms, unit="ns")

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
            back = slot.let("back", (d != None) & (d <= r))            # noqa: E711
            # the conversation's own tokens when its last request ended: that prompt and its reply
            prior = slot.let("prior", when(back, slot.ref("ptoks").at(r - d) + slot.ref("out").at(r - d), 0))
            cont = slot.let("cont", back & (when(back, slot.ref("sysblk").at(r - d), 0) * P.chunk_tokens + prior + inn + out <= P.context))
            # the chain: conv @ r = conv @ (r − d) or fresh
            conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), draw(uniform64())))
            sp = slot.let("sp", when(cont, slot.ref("sp").at(r - d), sysp.pick(P.sys_pop)))
            sysblk = slot.let("sysblk", sp.size // P.chunk_bytes)      # whole chunks inside the system prompt, shared by its conversations
            # the conversation's own tokens in this request's prompt; a new conversation starts with the system prompt's tokens past its last whole chunk
            ptoks = slot.let("ptoks", when(cont, prior, (sp.size % P.chunk_bytes) // token_bytes) + inn)
            stored = slot.let("stored", ptoks // P.chunk_tokens)       # whole chunks only
            had = slot.let("had", when(cont & (d <= P.retain), slot.ref("stored").at(r - d), 0))
            held = slot.let("held", when(cont, min_(prior, kp) // P.chunk_tokens, 0))   # whole chunks the engine kept, from the start
            load = slot.let("load", when(had > held, had - held, 0))

            with slot.phase(when(r < P.warm, "warm", "serve")):
                with slot.when(P.sys_local == False):                 # noqa: E712
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
                slot.compute(P.prefill_per_token * ((stored - had) * P.chunk_tokens))
                with slot.loop("k", stored, start=had) as k:           # the prompt's new whole chunks
                    b = slot.let("b", kv.object(conv=conv, k=k))
                    opened(slot, b, "WRONLY|CREAT|TRUNC|CLOEXEC")
                    slot.write(b, P.chunk_bytes)
                    slot.close(b)
                slot.compute(P.decode_per_token * out)

if __name__ == "__main__":
    w.write()
