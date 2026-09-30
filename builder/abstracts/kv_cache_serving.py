"""ABSTRACTS.md §8: KV-cache serving (prefix cache offloaded to storage).

Exercises: `at` chains (§9.5), a mixture with a none arm, expression-selected phase names, a
chunk-realized dataset, a namespace with a size expression, stat with expect.
"""
from aeiou import *

w = Workload("kv_cache_serving",
             doc="One engine per actor; `concurrency` request slots; each request continues an earlier conversation or starts one.")
P = w.P
w.param("concurrency", 64, unit="count")
w.param("warm", 2000, unit="count", doc="requests per slot before measurement (§9.5)")
w.param("requests", 20000, unit="count")
w.param("chunk_tokens", 256, unit="tokens")
w.param("chunk_bytes", 32 * MiB, unit="bytes", doc="128 KiB/token × 256 [config]")
w.param("sys_prompts", 50, unit="count")
w.param("sys_tokens", 1500, unit="tokens", doc="median system prompt length [measure]")
w.param("sys_pop", zipf(s=1.1), doc="[measure]")
w.param("reuse", mixture((0.55, none), (0.45, lognormal(median=40, sigma=1.2, min=1))),
        doc="requests ago, or none for a new conversation [measure]")
w.param("retain", 5000, unit="count", doc="older chunks are evicted [config: capacity]")
w.param("turn_in", lognormal(median=300, sigma=0.8), unit="tokens")
w.param("turn_out", lognormal(median=250, sigma=0.6), unit="tokens")
w.param("prefill_per_token", 40 * us, unit="ns")
w.param("decode_per_token", 12 * ms, unit="ns")

sysp = w.dataset("sysp", pattern="kv/sys/{id:04}/blk_{k:04}", count=P.sys_prompts,
                 size=lognormal(median=P.sys_tokens * 128 * KiB, sigma=0.3),   # 128 KiB per token
                 chunk=P.chunk_bytes,                                  # realized as ceil(size / chunk) block objects {k}
                 seed=0x5eed_da80)
kv = w.namespace("kv", pattern="kv/{conv:016x}/blk_{k:04}", fields={"conv": int, "k": int},
                 size=P.chunk_bytes, seed=0x5eed_da81)
kv_dir = w.namespace("kv_dir", pattern="kv/{conv:016x}", fields={"conv": int}, size=0, seed=0x5eed_da81)

with w.actor("gpu") as gpu:
    with gpu.parallel("slot", P.concurrency) as slot:
        with slot.loop("r", P.warm + P.requests) as r:                # one index space (§9.5)
            d = slot.draw("d", P.reuse)
            cont = slot.let("cont", (d != None) & (d <= r))            # noqa: E711
            # the chain: conv @ r = conv @ (r − d) or fresh
            conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), draw(uniform64())))
            sp = slot.let("sp", when(cont, slot.ref("sp").at(r - d), sysp.pick(P.sys_pop)))
            prev = slot.let("prev", when(cont, slot.ref("total").at(r - d), 0))     # blocks the conversation had
            hit = slot.let("hit", when(cont & (d <= P.retain), prev, 0))
            inn = slot.draw("inn", P.turn_in)
            out = slot.draw("out", P.turn_out)
            total = slot.let("total", prev + ceil_div(inn + out, P.chunk_tokens))
            sysblk = slot.let("sysblk", sp.chunks)

            with slot.phase(when(r < P.warm, "warm", "serve")):
                with slot.loop("k", sysblk) as k:                     # lookups: system prompt blocks
                    slot.stat(sp.chunk(k))
                with slot.loop("k", total) as k:                      # lookups: conversation blocks
                    slot.stat(kv.object(conv=conv, k=k), expect=["ENOENT"])
                with slot.loop("k", sysblk) as k:                     # read the system prompt
                    b = slot.let("b", sp.chunk(k))
                    slot.open(b, "RDONLY")
                    slot.read(b, P.chunk_bytes)
                    slot.close(b)
                with slot.loop("k", hit) as k:                        # read the cached prefix
                    b = slot.let("b", kv.object(conv=conv, k=k))
                    slot.open(b, "RDONLY")
                    slot.read(b, P.chunk_bytes)
                    slot.close(b)
                slot.compute(P.prefill_per_token * ((total - hit) * P.chunk_tokens))
                with slot.when(hit < total):                          # the conversation directory
                    slot.mkdir(kv_dir.object(conv=conv), expect=["EEXIST"])
                with slot.loop("k", total, start=hit) as k:           # write the new blocks
                    b = slot.let("b", kv.object(conv=conv, k=k))
                    slot.open(b, "WRONLY|CREAT|TRUNC")
                    slot.write(b, P.chunk_bytes)
                    slot.close(b)
                slot.compute(P.decode_per_token * out)

if __name__ == "__main__":
    w.write()
