"""ABSTRACTS.md §8, the shared store: an engine in front of a KV-cache store that other
engines read, and a cold engine in front of a store that is already filled.

The calls are the traces of vLLM with LMCache's `fs://` remote backend (2026-10-02, vLLM 0.30,
LMCache 0.5.5, builder/traces/kv_cache_shared). Unlike the local-disk backend of
`kv_cache_serving`, this one keeps no index: the store is the directory, so a restarted or a
second engine finds what the first one wrote. A lookup is a `stat` per whole chunk of the
prompt, from the first chunk until one is missing. A chunk is a file of `meta_bytes` of
header and `chunk_bytes`, written under a temporary name and renamed, and read through a
Python buffered reader: one `read` of the buffer, one of the rest. An engine loads the
chunks the store has, less the whole chunks it still holds in GPU memory. The lengths and
the request stream are `kv_cache_serving`'s (a ShareGPT replay; one `reuse` distance).

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
    w.param("sys_local", True, doc="[config] the engine holds the system prompts in GPU memory and never reads their chunks (traced); false: a system prompt is a prefix like the conversation's own, and its chunks are read when the engine holds none of the conversation (a new one, or one whose `keep` draw is under a chunk: measured in the AgentX replay, where the engine held the system prompt and some of the conversation or neither, §3.63)")
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
    w.param("reuse", mixture((0.15, none), (0.85, const(40))),
            doc="none for a new conversation (measured: 0.148 of ShareGPT's requests are first turns), or requests ago: the conversations "
                "a slot has open and serves in turn [config: load]. One distance: with several, two requests can continue the same one")
    w.param("keep", empirical({0: 91, 1_000_000: 9}), unit="tokens",
            doc="[config: GPU KV memory against the open conversations] the tokens of a returning conversation the engine still holds, "
                "counted from its start: all of it when the draw exceeds its length, else the chunks past them are loaded. Drawn once "
                "per round of the open conversations (the GPU's memory is one state for all of them), so a round's loads come together")
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
    fsize = P.meta_bytes + P.chunk_bytes                               # a chunk file
    sysp = w.dataset("sysp", pattern="kv/sys/{id:04}/blk_{k:04}", count=P.sys_prompts,
                     size=lognormal(median=P.sys_tokens * token_bytes, sigma=0.3,
                                    min=P.sys_tokens_min * token_bytes, max=P.sys_tokens_max * token_bytes),
                     chunk=fsize,                                      # ceil(size / chunk) block objects {k}; the last, partial one is never read
                     seed=0x5eed_da80)
    subp = w.dataset("subp", pattern="kv/sub/{id:04}/blk_{k:04}", count=P.sub_prompts,
                     size=lognormal(median=P.sub_tokens * token_bytes, sigma=0.3,
                                    min=P.sub_tokens_min * token_bytes, max=P.sub_tokens_max * token_bytes),
                     chunk=fsize,                                      # ceil(size / chunk) block objects {k}; the last, partial one is never read
                     seed=0x5eed_da84)
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
                inn = slot.draw("inn", P.turn_in)
                out = slot.draw("out", P.turn_out)
                kp = slot.draw("kp", P.keep)
                tr = slot.draw("tr", P.trim)
                tn = slot.draw("tn", P.turns)
                back = slot.let("back", (d != None) & (d <= r))        # noqa: E711
                # the conversation at r − d goes on when it has turns left (or no limit) and the context holds this request
                more = slot.let("more", back & ((slot.ref("turns").at(r - d) == 0) | (slot.ref("turn").at(r - d) + 1 < slot.ref("turns").at(r - d))))
                # the conversation's own tokens when its last request ended: that prompt and its reply
                prior = slot.let("prior", when(back, max_(slot.ref("ptoks").at(r - d) + slot.ref("out").at(r - d) - tr, 0), 0))
                cont = slot.let("cont", more & (when(back, slot.ref("sysblk").at(r - d), 0) * P.chunk_tokens + prior + inn + out <= P.context))
                conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), draw(uniform64())))
                turns = slot.let("turns", when(cont, slot.ref("turns").at(r - d), tn))
                turn = slot.let("turn", when(cont, slot.ref("turn").at(r - d) + 1, 0))
                sp = slot.let("sp", when(cont, slot.ref("sp").at(r - d),
                                         when(P.sys_per_slot == True, sysp.file(slot.index % P.sys_prompts), sysp.pick(P.sys_pop))))   # noqa: E712
                kd = slot.draw("kd", P.kind)                       # last of the draws: their sites (JSON pointers) stay put
                fi = slot.draw("fi", P.first_in)
                sh = slot.draw("sh", P.sys_held)
                bh = slot.draw("bh", P.sub_held)
                kind = slot.let("kind", when(cont, slot.ref("kind").at(r - d), kd))
                sb = slot.let("sb", when(cont, slot.ref("sb").at(r - d),
                                         when(P.sub_per_slot == True, subp.file(slot.index % P.sub_prompts), subp.pick(P.sub_pop))))   # noqa: E712
                psize = slot.let("psize", when(kind == 0, sp.size, when(kind == 1, sb.size, 0)))   # the prefix the conversation opens with
                sysblk = slot.let("sysblk", psize // fsize)    # whole chunks inside its prefix, shared by the conversations that open with it
                pheld = slot.let("pheld", min_(when(kind == 0, sh, when(kind == 1, bh, 0)) // P.chunk_tokens, sysblk))   # whole chunks of its prefix the engine holds
                ptoks = slot.let("ptoks", when(cont, prior, (psize % fsize) // token_bytes) + when(cont | (fi < 0), inn, fi))
                stored = slot.let("stored", ptoks // P.chunk_tokens)   # whole chunks only
                had = slot.let("had", when(cont & (d <= P.retain), min_(slot.ref("stored").at(r - d), prior // P.chunk_tokens), 0))   # the store's chunks of the kept prefix
                # what the engine itself still holds of this conversation: what it kept of its last prompt and reply, in whole
                # chunks; one `keep` draw per round of the d open conversations, made at the last request of the previous round (§3.57)
                held = slot.let("held", when(cont, min_(prior, slot.ref("kp").at(r - (r % d + 1))) // P.chunk_tokens, 0))
                hit = slot.let("hit", stored if reader else had)       # the chunks the store has for this prompt
                nload = slot.let("nload", when(hit > held, hit - held, 0))

                with slot.phase(when(r < P.warm, "warm", "serve")):
                    slot.compute(P.think)
                    with slot.when(kind == 0):
                        with slot.loop("q", sysblk) as q:              # the lookup: one stat per whole chunk, in order
                            slot.stat(sp.chunk(q))
                    with slot.when(kind == 1):
                        with slot.loop("q", sysblk) as q:
                            slot.stat(sb.chunk(q))
                    with slot.loop("q", hit) as q:
                        slot.stat(kv.object(conv=conv, k=q, ext="data"))
                    if not reader:
                        with slot.when(stored > had):                  # the lookup ends at the first chunk that is not there
                            slot.stat(kv.object(conv=conv, k=had, ext="data"), expect=["ENOENT"])
                    with slot.when((P.sys_local == False) & (held == 0) & (kind == 0)):   # noqa: E712
                        with slot.parallel("sk", sysblk - pheld) as rd:
                            load(rd, rd.let("b", sp.chunk(pheld + rd.index)))
                    with slot.when((P.sys_local == False) & (held == 0) & (kind == 1)):   # noqa: E712
                        with slot.parallel("ak", sysblk - pheld) as rd:
                            load(rd, rd.let("b", sb.chunk(pheld + rd.index)))
                    with slot.parallel("lk", nload) as rd:             # the chunks the engine does not hold, all at once
                        load(rd, rd.let("b", kv.object(conv=conv, k=held + rd.index, ext="data")))
                    if not reader:
                        cps = P.prefill_step // P.chunk_tokens         # chunks a prefill step completes
                        with slot.loop("s", ceil_div(stored - had, cps)) as s:   # the prefill, a step at a time; its chunks stored after each
                            slot.compute(P.prefill_per_token * min_((stored - had - s * cps) * P.chunk_tokens, P.prefill_step))
                            with slot.loop("k", min_(stored, had + (s + 1) * cps), start=had + s * cps) as k:   # the step's new whole chunks
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
