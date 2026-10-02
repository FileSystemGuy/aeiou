"""ABSTRACTS.md §8: KV-cache serving (prefix cache offloaded to storage).

The calls are the trace of vLLM with LMCache's local-disk backend (2026-10-02, vLLM 0.30,
LMCache 0.5.5, builder/traces/kv_cache_serving). LMCache keeps its index in memory: a lookup
touches no storage. One file holds one chunk of `chunk_tokens` tokens, in one flat directory,
named by the hash of the token prefix it ends; it is written with one `write` and read with
one `read`, each inside the six calls of a Python `open`. Only whole chunks of the prompt are
stored, when the request is prefilled; nothing is written during decode. A returning
conversation's chunks are read, all at once from a small thread pool, only when the engine
no longer holds them in GPU memory.

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
w.param("reuse", mixture((0.55, none), (0.45, lognormal(median=40, sigma=1.2, min=1))),
        doc="requests ago, or none for a new conversation [measure]")
w.param("local", 8, unit="count", doc="[config: GPU KV memory] a conversation returning within this many requests is still in the engine and reads nothing")
w.param("retain", 5000, unit="count", doc="older chunks are evicted [config: capacity]")
w.param("turn_in", lognormal(median=300, sigma=0.8), unit="tokens")
w.param("turn_out", lognormal(median=250, sigma=0.6), unit="tokens")
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
            cont = slot.let("cont", (d != None) & (d <= r))            # noqa: E711
            # the chain: conv @ r = conv @ (r − d) or fresh
            conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), draw(uniform64())))
            sp = slot.let("sp", when(cont, slot.ref("sp").at(r - d), sysp.pick(P.sys_pop)))
            inn = slot.draw("inn", P.turn_in)
            out = slot.draw("out", P.turn_out)
            sysblk = slot.let("sysblk", sp.size // P.chunk_bytes)      # whole chunks inside the system prompt, shared by its conversations
            # the conversation's own tokens in this request's prompt: what the last turn had and generated, plus this input;
            # a new conversation starts with the system prompt's tokens past its last whole chunk
            ptoks = slot.let("ptoks", when(cont, slot.ref("ptoks").at(r - d) + slot.ref("out").at(r - d),
                                           (sp.size % P.chunk_bytes) // token_bytes) + inn)
            stored = slot.let("stored", ptoks // P.chunk_tokens)       # whole chunks only
            had = slot.let("had", when(cont & (d <= P.retain), slot.ref("stored").at(r - d), 0))
            load = slot.let("load", when(cont & (d > P.local), had, 0))

            with slot.phase(when(r < P.warm, "warm", "serve")):
                with slot.when(P.sys_local == False):                 # noqa: E712
                    with slot.parallel("sk", sysblk) as rd:
                        b = rd.let("b", sp.chunk(rd.index))
                        opened(rd, b, "RDONLY|CLOEXEC")
                        rd.read(b, P.chunk_bytes)
                        rd.close(b)
                with slot.parallel("lk", load) as rd:                  # the cached prefix the engine lost, all chunks at once
                    b = rd.let("b", kv.object(conv=conv, k=rd.index))
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
