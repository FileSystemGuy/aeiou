"""ABSTRACTS.md §8, the shared store: an engine in front of a KV-cache store that other
engines read, and a cold engine in front of a store that is already filled.

The calls are the traces of vLLM with LMCache's `fs://` remote backend (2026-10-02, vLLM 0.30,
LMCache 0.5.5, builder/traces/kv_cache_shared). Unlike the local-disk backend of
`kv_cache_serving`, this one keeps no index: the store is the directory, so a restarted or a
second engine finds what the first one wrote. A lookup is a `stat` per whole chunk of the
prompt, from the first chunk until one is missing. A chunk is a file of `meta_bytes` of
header and `chunk_bytes`, written under a temporary name and renamed, and read through a
Python buffered reader: one `read` of the buffer, one of the rest. An engine loads the
chunks the store has, less the whole chunks it still holds in GPU memory.

Two workloads from one script, because they are the same request stream on two states of
the store:

  kv_cache_shared         the store starts empty; the engine stores every whole chunk of a
                          prompt that is not there yet and loads only what its GPU lost
  kv_cache_shared_reader  the store is the namespace a `kv_cache_shared` run left (`input`,
                          V14); every lookup hits, nothing is written, and the prompt's new
                          chunks are loaded where the first engine computed them

The reader must be run with the writer's `--seed`, `--gpus` and parameters: the conversation
ids are positional draws, and both workloads place them at the same sites.

Exercises: `at` chains (§9.5), an input namespace read by a second abstract, `rename`,
expected errors, a chunk-realized dataset.
"""
from aeiou import *


def shape(name, reader, doc):
    w = Workload(name, doc=doc)
    P = w.P
    w.param("concurrency", 64, unit="count")
    w.param("warm", 2000, unit="count", doc="requests per slot before measurement (§9.5)")
    w.param("requests", 20000, unit="count")
    w.param("chunk_tokens", 256, unit="tokens")
    w.param("chunk_bytes", 32 * MiB, unit="bytes", doc="128 KiB/token × 256 [config]")
    w.param("meta_bytes", 28, unit="bytes", doc="the header in front of a chunk (traced: seven 32-bit integers)")
    w.param("buf", 1 * MiB, unit="bytes", doc="Python BufferedReader size = st_blksize; the first read of a chunk file is this long")
    w.param("sys_prompts", 50, unit="count")
    w.param("sys_tokens", 1500, unit="tokens", doc="median system prompt length [measure]")
    w.param("sys_tokens_min", 1, unit="tokens", doc="bounds of the system prompt length; equal bounds give every prompt that length")
    w.param("sys_tokens_max", 100_000, unit="tokens")
    w.param("sys_pop", zipf(s=1.1), doc="[measure]")
    w.param("sys_local", True, doc="[config] the engine holds the system prompts in GPU memory and never loads their chunks; false loads them for every new conversation")
    w.param("reuse", mixture((0.55, none), (0.45, lognormal(median=40, sigma=1.2, min=1))),
            doc="requests ago, or none for a new conversation [measure]")
    w.param("local", 8, unit="count", doc="[config: GPU KV memory] a conversation returning within this many requests is still in the engine")
    w.param("retain", 5000, unit="count", doc="older chunks are evicted [config: capacity]")
    w.param("turn_in", lognormal(median=300, sigma=0.8), unit="tokens")
    w.param("turn_out", lognormal(median=250, sigma=0.6), unit="tokens")
    w.param("prefill_per_token", 40 * us, unit="ns")
    w.param("decode_per_token", 12 * ms, unit="ns")

    token_bytes = P.chunk_bytes // P.chunk_tokens
    fsize = P.meta_bytes + P.chunk_bytes                               # a chunk file
    sysp = w.dataset("sysp", pattern="kv/sys/{id:04}/blk_{k:04}", count=P.sys_prompts,
                     size=lognormal(median=P.sys_tokens * token_bytes, sigma=0.3,
                                    min=P.sys_tokens_min * token_bytes, max=P.sys_tokens_max * token_bytes),
                     chunk=fsize,                                      # ceil(size / chunk) block objects {k}; the last, partial one is never read
                     seed=0x5eed_da80)
    kv = w.namespace("kv", pattern="kv/{conv:016x}-{k:04}.{ext}", fields={"conv": int, "k": int, "ext": str},
                     size=fsize, seed=0x5eed_da83, input=reader, same_run=reader)   # one flat directory (traced); the reader's names are the writer's draws (V15)

    def load(a, b):
        """Python's open() and a buffered reader: the header's read fills the buffer, the chunk's read takes the rest."""
        a.open(b, "RDONLY|CLOEXEC")
        a.fstat(b)
        a.ioctl(b, "TCGETS", expect=["ENOTTY"])
        a.lseek(b, 0, "CUR")
        a.read(b, P.buf)
        with a.when(fsize > P.buf):
            a.read(b, fsize - P.buf)
        a.close(b)

    with w.actor("gpu") as gpu:
        with gpu.parallel("slot", P.concurrency) as slot:
            with slot.loop("r", P.warm + P.requests) as r:            # one index space (§9.5)
                # the same statements, in the same places, in both workloads: the draws are keyed by where they stand
                d = slot.draw("d", P.reuse)
                cont = slot.let("cont", (d != None) & (d <= r))        # noqa: E711
                conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), draw(uniform64())))
                sp = slot.let("sp", when(cont, slot.ref("sp").at(r - d), sysp.pick(P.sys_pop)))
                inn = slot.draw("inn", P.turn_in)
                out = slot.draw("out", P.turn_out)
                sysblk = slot.let("sysblk", sp.size // fsize)          # whole chunks inside the system prompt, shared by its conversations
                ptoks = slot.let("ptoks", when(cont, slot.ref("ptoks").at(r - d) + slot.ref("out").at(r - d),
                                               (sp.size % fsize) // token_bytes) + inn)
                stored = slot.let("stored", ptoks // P.chunk_tokens)   # whole chunks only
                had = slot.let("had", when(cont & (d <= P.retain), slot.ref("stored").at(r - d), 0))
                # what the engine itself still holds of this conversation: its last prompt and reply, in whole chunks
                held = slot.let("held", when(cont & (d <= P.local), (ptoks - inn) // P.chunk_tokens, 0))
                hit = slot.let("hit", stored if reader else had)       # the chunks the store has for this prompt
                nload = slot.let("nload", when(hit > held, hit - held, 0))

                with slot.phase(when(r < P.warm, "warm", "serve")):
                    with slot.loop("q", sysblk) as q:                  # the lookup: one stat per whole chunk, in order
                        slot.stat(sp.chunk(q))
                    with slot.loop("q", hit) as q:
                        slot.stat(kv.object(conv=conv, k=q, ext="data"))
                    if not reader:
                        with slot.when(stored > had):                  # the lookup ends at the first chunk that is not there
                            slot.stat(kv.object(conv=conv, k=had, ext="data"), expect=["ENOENT"])
                    with slot.when(P.sys_local == False):             # noqa: E712
                        with slot.parallel("sk", sysblk) as rd:
                            load(rd, rd.let("b", sp.chunk(rd.index)))
                    with slot.parallel("lk", nload) as rd:             # the chunks the engine does not hold, all at once
                        load(rd, rd.let("b", kv.object(conv=conv, k=held + rd.index, ext="data")))
                    if not reader:
                        slot.compute(P.prefill_per_token * ((stored - had) * P.chunk_tokens))
                        with slot.loop("k", stored, start=had) as k:   # the prompt's new whole chunks
                            t = slot.let("t", kv.object(conv=conv, k=k, ext="tmp"))
                            slot.open(t, "WRONLY|CREAT|TRUNC|CLOEXEC")
                            slot.fstat(t)
                            slot.ioctl(t, "TCGETS", expect=["ENOTTY"])
                            slot.lseek(t, 0, "CUR")
                            slot.write(t, P.meta_bytes)
                            slot.write(t, P.chunk_bytes)
                            slot.close(t)
                            slot.rename(t, kv.object(conv=conv, k=k, ext="data"))
                    slot.compute(P.decode_per_token * out)
    return w


shape("kv_cache_shared", False,
      "One engine per actor in front of a store with no index (LMCache fs://): stat lookups, chunks written to a temporary name and renamed.")
shape("kv_cache_shared_reader", True,
      "A cold engine in front of the store a kv_cache_shared run filled: every lookup hits, every chunk it does not hold is read, nothing is written.")

if __name__ == "__main__":
    for wl in Workload._registry:
        wl.write()
