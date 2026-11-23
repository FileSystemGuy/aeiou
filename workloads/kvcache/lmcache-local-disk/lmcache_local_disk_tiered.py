"""ABSTRACTS.md §8: KV-cache serving, the cache hierarchy modelled (contract 0.20; brief items 71, 72,
75; DESIGN_REVIEW.md §3.99, the notes "Tier names", "Contract 0.19 proposed" and its addendum "Built",
"The run side of timelines").

One engine (vLLM with LMCache 0.5.5) serving agentic-coding sessions, each a timeline lane of its
main agent and its sub-agents, from the session generator fitted to the AgentX corpus
(`kit/agentx.py sessions`, `kit/sessions.fit.json`, the scratch's `gen6`: planned-time order).
Every request takes one of `slots` (first come first, waiting in planned time), then holds its
prompt and its reply's blocks in G1, vLLM's KV pool in HBM (a held tier, freed least recently first
and a request's tail first, `full: wait`). LMCache then looks the prompt up in G2 (host DRAM) from
its start and in G3 (node-local NVMe, the directory under test) past G2's hit; it reads G3's hits
from the chunk holding the tokens the GPU held, stores the chunks past both hits in G2 (a planned
drop when G2 is full), and in G3 the chunks G2 newly took, writing a file for each (rewriting one G3
already holds) on one of `store_lanes` store workers, never on the request's lane (contract 0.20's
`submit`; the unfinished writes' bytes held in G2, the workers' staging); a request's G3 reads are
its span `load`, the storage wait a run judges it by; at a request's end every chunk of its prompt
is touched in G2. G2 and G3 evict as
LMCache 0.5.5 does: every second, at 0.8 of the capacity, 20 % of the keys (at least one), least
recently touched first and a request's tail first; a G3 victim's file is unlinked on `evict_lanes`
eviction lanes (the connector's delete workers). Held request by request and unlink by unlink to
LMCache's own `LRUEvictionPolicy` driven by the walk's planned events (DESIGN_REVIEW.md §3.99,
addendum "Built", seven configurations).

The calls on a chunk file are those traced under LMCache's in-process connector (2026-10-02, the
slot-model abstract `lmcache_local_disk`, retired 2026-11-14, its captures in `kit/captures/`): Python's `open` around one `read` or one `write`,
or the `O_DIRECT` path; an eviction is one `unlink`. The second abstract, `_g4`, has the calls
of LMCache's multi-process `fs_native` adapter instead, the mechanism's target (brief item 70),
captured 2026-11-22 (item 74, `kit/captures/agentx-mp/`): a store is `NAME.tmp` opened
`O_WRONLY|O_CREAT|O_TRUNC`, one write, close, and a `rename` to `NAME.data`, never of a chunk the
adapter holds; a load is open, one read, close; a lookup is a `stat` of every key past G2's hit
in each adapter, in the request's span `load`, and a store stats each chunk first, on the
adapter's store workers (since 2026-11-23). The lookup's stats and the loads are on the
adapter's workers too, the span waiting for them (contract 0.26's joined `submit`), so an
adapter's `store_lanes` lanes carry its lookups, loads, and stores; its unlinks stay on eviction
lanes of their own, where a run times each batch. This abstract, with one adapter, keeps the
in-process calls and its reads on the request's lane.

`aeiou run` runs it since brief item 72 (7) (2026-11-13/14); its case is `unsettled`. Not built yet:
sessions arrive uniformly over `window_s`
(the WG's ramp, item 72 (1), is not yet written); a session's service time is the corpus's
measured `api_time`, not a model of prefill and decode on an accelerator (item 70).
"""
import json
import pathlib

from aeiou import *
from aeiou.builder import ArrayRef

HERE = pathlib.Path(__file__).resolve().parent
FIT = json.loads((HERE / "kit" / "sessions.fit.json").read_text())
G, CL = FIT['glob'], FIT['classes']
NC, NG = len(CL), len(G['grid'])
S = 1_000_000                                   # fixed point
BL = 64                                         # the corpus's block of tokens
FR = 1 << 16                                    # the uniform point inside a segment
widths = [G['grid'][i + 1] - G['grid'][i] for i in range(NG - 1)]
E, O, F, T, Dk, C, M, FS, FB, NONE = range(10)  # step kinds (agentx.py sessions), then after a fresh prompt (smaller, bigger)
KC = dict(E=E, O=O, F=F, T=T, D=Dk, C=C, M=M)

tab, off = [], {}
def put(name, rows):
    off[name] = len(tab)
    for r in rows:
        assert len(r) == NG
        tab.extend(round(x * S) for x in r)
for k, v in G.items():
    if isinstance(v, list) and len(v) == NG and k != 'grid':
        put('g.' + k, [v])
for j, v in enumerate(G['plong_main']):
    put('g.plong_main%d' % j, [v])
for k in ('n_main', 'n_batch', 'sp', 't2', 'dfrom'):
    put('c.' + k, [c[k] for c in CL])
for who in ('main', 'sub'):
    for k, v in CL[0][who].items():
        if isinstance(v, list):
            put('%s.%s' % (who, k), [c[who][k] for c in CL])

def shape(name, doc, g4):
    w = Workload(name, doc=doc)
    P = w.P
    # A case gives the values with no default (DESIGN_REVIEW.md §3.78): the model's (chunk_bytes), the
    # WG's tier sizes (g1_bytes, g2_bytes, g3_bytes; §3.99 note "Tier names"), and the load's (sessions,
    # window_s, slots). The fit's tables are the generator's defaults (agentx.py sessions on the AgentX corpus).
    w.param("sessions", kind=int, unit="count", doc="sessions served by the engine [load]")
    w.param("window_s", kind=int, unit="count", doc="seconds: sessions arrive uniformly over this window [load]")
    w.param("slots", kind=int, unit="count", doc="the engine's request slots (vLLM's max_num_seqs) [config: engine]")
    w.param("chunk_bytes", kind=int, unit="bytes",
            doc="the model's KV bytes per token × 256, LMCache's chunk; a file is exactly this, no header (traced)")
    w.param("per", 4, unit="count", doc="the corpus's 64-token blocks a chunk (LMCache's chunk_size 256)")
    w.param("g1_bytes", kind=int, unit="bytes", doc="G1: vLLM's KV pool in HBM, the GPU's memory less the model's weights [WG]")
    w.param("g2_bytes", kind=int, unit="bytes", doc="G2: LMCache's L1 in host DRAM (--l1-size-gb) [WG]")
    w.param("g3_bytes", kind=int, unit="bytes", doc="G3: LMCache's L2 on node-local NVMe, the directory under test [WG]")
    if g4:
        w.param("g4_bytes", kind=int, unit="bytes", doc="G4: LMCache's second L2, on shared storage, a directory under test of its own [WG]")
    w.param("evict_lanes", 4, unit="count", doc="LMCache's delete workers (the connector's num_workers, 4 in 0.5.5)")
    w.param("store_lanes", 4, unit="count", doc="LMCache's store workers: a chunk's write queued to them, the forward step waiting only "
            "for its copy to host memory (0.5.5's local-disk path, read 2026-11-09, DESIGN_REVIEW.md §3.99 item 13 (4))")
    if not g4:                                      # fs_native's own `odirect` path is not traced, so the G4 abstract has none
        w.param("direct", False, doc="[config] LMCache's `use_odirect`: every chunk opened with `O_DIRECT` by `os.open` and read or "
                "written in one call (traced 2026-10-07, §3.63); LMCache takes this path only when chunk_bytes is a multiple of the "
                "file system's block size")
    w.param("tab", tab, doc="[load, fitted] the fit's quantile tables, fixed point × 1e6, one row a table on the fit's grid")
    w.param("prof", [round(x * S) for x in G['prof']], doc="[load, fitted] the main agent's growth by tenth of its session")
    w.param("p_sub", [round(c['p_sub'] * S) for c in CL], doc="[load, fitted] a session has sub-agents, by class")
    w.param("p_nodeep", [round(c['p_nodeep'] * S) for c in CL], doc="[load, fitted] a tall session never compacts, by class")
    w.param("p_d_small", [round(c['main']['p_D_small'] * S) for c in CL], doc="[load, fitted] a deep cut of a small tip, by class")
    for k in ('p_has_sp', 'lam_main', 'lam_sub', 'p_ext_full', 'p_sync'):
        w.param(k, round(G[k] * S), doc="[load, fitted]")
    w.param("cap", G['cap'], unit="tokens", doc="[load, fitted] the longest prompt: a context at it compacts")
    w.param("mb", G['mb'], unit="count", doc="[load, fitted] the main agent's shared prefix in blocks")
    w.param("cls", empirical(list(range(NC)), [c['weight'] for c in CL]), doc="[load, fitted] a session's class")
    for who in ('main', 'sub'):
        for i, c in enumerate(CL):
            st = {k: v for k, v in c[who]['steps'].items() if not (who == 'main' and k == 'D')}
            w.param("k_%s%d" % (who, i), empirical([KC[k] for k in st], list(st.values())), doc="[load, fitted] step kinds")
    for k, v in G['afterF'].items():
        w.param("k_" + k.lower(), empirical([KC[x] for x in v], list(v.values())), doc="[load, fitted] step kinds after a fresh prompt")

    # the capacities, in the tiers' units: G1 in blocks, G2 and G3 in chunks
    G1_BLOCKS = P.g1_bytes // (P.chunk_bytes // P.per)
    G2_CHUNKS = P.g2_bytes // P.chunk_bytes
    G3_CHUNKS = P.g3_bytes // P.chunk_bytes

    if not g4:
        kv = w.namespace("kv", pattern="kv/{key:016x}.pt", fields={"key": int}, size=P.chunk_bytes, seed=0x5eed_da85)   # one flat directory (traced)
        final = lambda ns, key: ns.object(key=key)                     # noqa: E731
    else:                                           # fs_native: `NAME.tmp` renamed to `NAME.data`; G4's directory a namespace of its own, so `--endpoint kv4=DIR` places it on the shared system
        G4_CHUNKS = P.g4_bytes // P.chunk_bytes
        kv = w.namespace("kv", pattern="kv/{key:016x}.{ext}", fields={"key": int, "ext": str}, size=P.chunk_bytes, seed=0x5eed_da85)
        kv4 = w.namespace("kv4", pattern="kv4/{key:016x}.{ext}", fields={"key": int, "ext": str}, size=P.chunk_bytes, seed=0x5eed_da86)
        final = lambda ns, key: ns.object(key=key, ext="data")         # noqa: E731


    def opened(a, b, flags):
        """Python's open(): the six calls around one chunk, less the data call."""
        a.open(b, flags)
        a.fstat(b)
        a.ioctl(b, "TCGETS", expect=["ENOTTY"])
        a.lseek(b, 0, "CUR")


    def chunk_read(a, b):
        """One chunk read whole: Python's open() around it, or LMCache's `O_DIRECT` path (`direct`)."""
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


    _n = [0]
    def inv(c, table, cls=None):
        """The fit's inverse CDF, × 1e6: the library's `quantile` over the table's rows in `tab`, a
        segment by the grid's widths, then a point in it."""
        _n[0] += 1
        b = off[table] + (0 if cls is None else cls * NG)
        return quantile(c, "q%d" % _n[0], P.tab, widths, at=b, points=FR)

    def inv_at(seg, frac, table):
        """inv() as one expression of a segment and a fraction drawn apart (an array's element)."""
        return quantile_at(seg, frac, P.tab, at=off[table], points=FR)

    def blocks(x):                                  # a fixed-point token count in whole blocks
        return x // (S * BL)

    def chance(c, p):                               # true with probability p, p × 1e6
        _n[0] += 1
        return c.draw("u%d" % _n[0], uniform(0, S)) < p

    def pick(c, who, last, cls):
        ks = getattr(P, "k_%s0" % who)
        for i in range(1, NC):
            ks = when(cls == i, getattr(P, "k_%s%d" % (who, i)), ks)
        return draw(when(last == FS, getattr(P, "k_%sfs" % who), when(last == FB, getattr(P, "k_%sfb" % who), ks)))


    def reused(c, tl, tag, n, nw):
        """A fresh prompt of n blocks writing nw: the longest of 8 prompts drawn from the session's pool
        (the prompts already sent, the timeline group's entries), its start re-sent. Returns (len, seq)."""
        pub = c.let(tag + 'pub', tl.published)
        nn = c.let(tag + 'nn', max_(n, nw))
        best_len, best_i = None, None
        for t in range(8):
            u = c.draw(tag + 'pu%d' % t, uniform(0, max_(pub, 1)))
            ln = c.let(tag + 'pl%d' % t, when(pub > 0, tl.entry(u, 'len'), 0))
            if best_len is None:
                best_len, best_i = ln, u
            else:
                better = ln > best_len
                best_len = c.let(tag + 'bl%d' % t, when(better, ln, best_len))
                best_i = c.let(tag + 'bi%d' % t, when(better, u, best_i))
        hit = c.let(tag + 'hit', (nn - nw > 0) & (pub > 0))
        keep = c.let(tag + 'keep', when(hit, min_(nn - nw, best_len), 0))
        ln = c.let(tag + 'len', when(hit, keep + max_(1, nw), max_(1, when(pub > 0, nw, nn))))
        sq = c.let(tag + 'seq', when(hit, cat(prefix(tl.entry(best_i, 'seq'), keep), fresh(ln - keep)), fresh(ln)))
        return ln, sq


    def chain(c, tl, i, who, n, first, cls, extra):
        """One context's prompt at index i of its loop; `first` = (len, seq) of prompt 0. The lane's state
        (its current prompt, last kind, largest prompt, compaction size, long-gap state) is carried by
        `prev`. Returns (len, out tokens, service ms, idle ms, seq)."""
        W = who
        N = lambda x: W[0] + '_' + x
        prev = lambda name, d: c.ref(N(name)).prev(d)
        rate = extra['rate']
        plen = prev('plen', 0)
        pseq = prev('pseq', shared(9, 0))
        last = prev('last', NONE)
        rm = prev('rm', first[0] * BL)
        atsz = prev('atsz', extra['atsz0'])
        k0 = c.let(N('k0'), pick(c, W, last, cls))
        if W == 'main':
            dsmall = chance(c, P.p_d_small[cls])
            k0 = c.let(N('k0b'), when((last != FS) & (last != FB) & (k0 != F) & (plen * BL * 2 < rm) & dsmall, Dk, k0))
        comp = c.let(N('comp'), (plen * BL * S >= atsz) & (k0 != F))
        k1 = c.let(N('k1'), when(comp, C, k0))
        c.let(N('atsz'), when(comp, inv(c, 'c.dfrom', cls), atsz))
        k = c.let(N('k'), when((k1 != F) & (plen * BL >= P.cap) & ((k1 == E) | (k1 == O)), C if W == 'main' else Dk, k1))
        # F: a fresh prompt, its start perhaps an earlier prompt's
        nf = blocks(inv(c, W + '.fresh', cls))
        nw = c.let(N('nw'), blocks(inv(c, 'g.fnew_' + W)))
        flen, fseq = reused(c, tl, W[0] + '_f', nf, nw)
        # E / O: extend the tip
        pf = (P.prof[(i * 10) // n] if W == 'main' else S)
        d = max_(BL, rate * inv(c, 'g.z_' + W) // S * pf // S // S)
        full = chance(c, P.p_ext_full)
        # T, D, C, M: a cut, then new blocks
        kf = when(k == C, inv(c, 'main.keepC', cls) if W == 'main' else 0,
                  when(k == T, inv(c, W + '.keepT', cls), inv(c, W + '.keepD', cls)))
        cn = when(k == C, inv(c, 'main.cutnewC', cls) if W == 'main' else 0,
                  when(k == T, inv(c, W + '.cutnewT', cls), inv(c, W + '.cutnew', cls)))
        ck = min_(when(k == M, P.mb, max_(1, plen * kf // S)), plen)
        keep = when(k == F, 0, when((k == E) | (k == O), when(full, plen, plen - 1), ck))
        ln = when(k == F, flen, when((k == E) | (k == O), when(full, plen, plen - 1) + max_(1, d // BL), ck + max_(1, blocks(cn))))
        sq = when(k == F, fseq, cat(prefix(pseq, keep), fresh(ln - keep)))
        if W == 'main':                             # the second prompt, the session's real start
            new2 = c.let(N('new2'), blocks(inv(c, 'g.t2_new')))
            n2 = max_(blocks(inv(c, 'c.t2', cls)), new2)
            keep2 = min_(n2 - new2, first[0])
            ln = when(i == 1, keep2 + max_(1, new2), ln)
            sq = when(i == 1, cat(prefix(pseq, keep2), fresh(max_(1, new2))), sq)
        ln = c.let(N('len'), when(i == 0, first[0], ln))
        sq = c.let(N('seq'), when(i == 0, first[1], sq))
        kind = c.let(N('kind'), when(i == 0, NONE, when((W == 'main') & (i == 1), FB, when(k == F, when(plen > ln, FS, FB), k))))
        c.let(N('last'), kind)
        adv = (i <= 1) | (k == E) | (k == O) | (k == T) | (k == C) | (ln * 2 >= plen)
        c.let(N('plen'), when(adv, ln, plen))
        c.let(N('pseq'), when(adv, sq, pseq))
        c.let(N('rm'), max_(rm, ln * BL) if W == 'main' else rm)
        # sent: the pool holds every prompt but a context's first, at its planned time
        with c.when(i >= 1):
            c.publish(tl, len=ln, seq=sq)
        # times: the long-gap state, then out, api, idle
        p, lam = (extra['pl'], getattr(P, 'lam_' + W))
        c.let(N('lg0'), chance(c, p))
        lp = prev('lg', c.ref(N('lg0')))
        c.let(N('lg'), chance(c, when(lp, p + lam * (S - p) // S, p * (S - lam) // S)))
        out = c.let(N('out'), max_(1, inv(c, 'g.out_' + W) // S))
        api = c.let(N('api'), inv(c, 'g.api_' + W) // 1000)                 # milliseconds
        idle = c.let(N('idle'), when(c.ref(N('lg')), inv(c, 'g.idle_l_' + W), inv(c, 'g.idle_s_' + W)) // 1000)
        return ln, out, api, idle, sq


    def admit(c, tag, sq, out):
        """A slot, then the prompt and its reply's blocks held in G1; the blocks G1 already held."""
        c.acquire("slots", 1, "slot")
        gs = c.let(tag + "gs", cat(sq, fresh(ceil_div(out, BL))))
        return c.lookup("gpu", "g1", chunk_keys(gs, 1), tag + "gh", hold="kv")


    def finish(c, sq):
        c.touch("lm", "g2", chunk_keys(sq, P.per))                      # LMCache touches every chunk at a request's end
        c.release("kv")
        c.release("slot")


    def chunk_ops(c, tag, sq, gh):
        """LMCache 0.5.5 over G2 and G3: the leading chunks in G2, then those in G3 past them; G3's hits read
        (loaded, and touched in G3) from the chunk holding the tokens the GPU held; the chunks past both hits
        stored in G2, and those G2 newly took stored in G3, each file G3 took written."""
        n = c.let(tag + "nch", seq_len(sq) // P.per)
        h2 = c.probe("lm", "g2", chunk_keys(sq, P.per), tag + "h2")
        if g4:
            return chunk_ops_g4(c, tag, sq, gh, n, h2)
        h3 = c.probe("lm", "g3", chunk_keys(sq, P.per, h2), tag + "h3")
        hit = c.let(tag + "hit", h2 + h3)
        first = c.let(tag + "first", max_(h2, min_(gh * BL, seq_len(sq) * BL - 1) // (BL * P.per)))
        # D: the request's own storage wait; its reads load into G2, the pool the pending stores
        # hold (item 13 (4)), so they draw on the store workers' staging (contract 0.21)
        with c.span("load", staging="store"):
            with c.loop(tag + "r", hit, start=first) as j:
                chunk_read(c, c.let(tag + "rf", kv.object(key=chunk_key(sq, j, P.per))))
        c.touch("lm", "g3", chunk_keys(sq, P.per, first, hit - first))
        t2 = c.store("lm", "g2", chunk_keys(sq, P.per, hit, n - hit), taken=tag + "t2")   # what G2 newly took: chunk indices
        t3 = c.store("lm", "g3", chunk_keys(sq, P.per, chunks=t2), present="renew", taken=tag + "t3")   # indices into t2
        with c.loop(tag + "w", t3.len) as j:
            with c.submit("store"):                                     # on a store worker, never the request's lane
                chunk_write(c, c.let(tag + "wf", kv.object(key=chunk_key(sq, t2[t3[j]], P.per))))


    def chunk_ops_g4(c, tag, sq, gh, n, h2):
        """The same with two L2 adapters, G3 and G4 (LMCache 0.5.5's `distributed` controllers, read
        2026-11-22: `store_policy.py`, `prefetch_policy.py`, `prefetch_controller.py`): the chunks past
        G2's hit are looked up in both, the leading run some adapter holds is the hit, and each of its
        chunks is loaded by the first adapter that holds it, G3 before G4, which alone is touched for
        it; every chunk G2 newly took is stored in each adapter that lacks it, each adapter's write on
        its own workers. The calls on a chunk file are `fs_native`'s, traced 2026-11-22 (brief item 74)."""
        h3 = c.probe("lm", ["g3", "g4"], chunk_keys(sq, P.per, h2), tag + "h3")
        hit = c.let(tag + "hit", h2 + h3)
        first = c.let(tag + "first", max_(h2, min_(gh * BL, seq_len(sq) * BL - 1) // (BL * P.per)))
        _, a3, a4 = c.probe("lm", ["g3", "g4"], chunk_keys(sq, P.per, first, hit - first), tag + "hs", held=(tag + "a3", tag + "a4"))
        # the request waits for its lookup and its loads, which each adapter's workers carry (contract
        # 0.26; traced 2026-11-22: an adapter's four threads issue all its calls): worker q of an
        # adapter takes every `store_lanes`-th key from the q-th, behind what the lane already holds
        with c.span("load", staging="store"):
            # the lookup: every key past G2's hit statted in each adapter, whatever the other holds
            # (0.5.5's storage_manager.py), a key it lacks ENOENT
            for tg, ns, wk in (("q3", kv, "store"), ("q4", kv4, "store4")):
                with c.loop(tag + tg, min_(P.store_lanes, n - h2)) as q:
                    with c.submit(wk, join=True):
                        with c.loop(tag + tg + "i", ceil_div(n - h2 - q, P.store_lanes)) as i:
                            c.stat(final(ns, chunk_key(sq, h2 + q + i * P.store_lanes, P.per)), expect=["ENOENT"])
            for tg, ns, arr, wk in (("r3", kv, a3, "store"), ("r4", kv4, a4, "store4")):
                with c.loop(tag + tg, min_(P.store_lanes, arr.len)) as q:
                    with c.submit(wk, join=True):
                        with c.loop(tag + tg + "i", ceil_div(arr.len - q, P.store_lanes)) as i:
                            native_read(c, c.let(tag + tg + "f", final(ns, chunk_key(sq, arr[q + i * P.store_lanes], P.per))))
        c.touch("lm", "g3", chunk_keys(sq, P.per, chunks=a3))
        c.touch("lm", "g4", chunk_keys(sq, P.per, chunks=a4))
        t2 = c.store("lm", "g2", chunk_keys(sq, P.per, hit, n - hit), taken=tag + "t2")
        for tr, ns, wk in (("g3", kv, "store"), ("g4", kv4, "store4")):
            with c.loop(tag + "c" + tr, t2.len) as j:                   # a store's check: each chunk statted first, on the adapter's workers
                with c.submit(wk):
                    c.stat(final(ns, chunk_key(sq, t2[j], P.per)), expect=["ENOENT"])
            tk = c.store("lm", tr, chunk_keys(sq, P.per, chunks=t2), taken=tag + "t" + tr)   # indices into t2; a chunk the adapter holds is left, never rewritten (traced)
            with c.loop(tag + "w" + tr, tk.len) as j:
                with c.submit(wk):
                    native_write(c, tag + "wf" + tr, ns, chunk_key(sq, t2[tk[j]], P.per))


    def native_read(a, b):
        """One chunk loaded by LMCache's `fs_native` adapter (the multi-process server, traced 2026-11-22)."""
        a.open(b, "RDONLY")
        a.read(b, P.chunk_bytes)
        a.close(b)


    def native_write(a, tag, ns, key):
        """One chunk stored by `fs_native`: written whole to `NAME.tmp`, renamed into place (traced 2026-11-22)."""
        t = a.let(tag, ns.object(key=key, ext="tmp"))
        a.open(t, "WRONLY|CREAT|TRUNC", mode=0o644)
        a.write(t, P.chunk_bytes)
        a.close(t)
        a.rename(t, final(ns, key))


    def sub_agents(bt, tl, s, cls, spl, psub, join):
        """A batch's members, each a lane: its start offset, first prompt, and chain."""
        with bt.parallel("k", s.bsize[bt.index], join=join) as sa:
            st = sa.let("st", sa.draw("stu", uniform(0, 1_000_000)))       # the main request's end + U(0, 1) s, in us
            sa.compute(st * 1000)
            hsp = sa.let("hsp", chance(sa, P.p_has_sp))
            spf = sa.let("spf", blocks(inv(sa, 'g.sub_past')))
            rlen, rseq = reused(sa, tl, 'sf_', blocks(inv(sa, 'g.sub_first')), blocks(inv(sa, 'g.first_new')))
            first = (sa.let("fl", when(hsp, spl + max_(1, spf), rlen)),
                     sa.let("fsq", when(hsp, cat(shared(1, spl), fresh(max_(1, spf))), rseq)))
            ns = sa.let("ns", max_(1, (inv(sa, 'sub.turns', cls) + S // 2) // S))
            rate_s = sa.let("rate_s", inv(sa, 'sub.rate', cls))
            with sa.loop("j", ns) as j:
                ln2, out2, api2, idle2, sq2 = chain(sa, tl, j, 'sub', ns, first, cls, dict(rate=rate_s, atsz0=1 << 62, pl=psub))
                gh2 = admit(sa, "s_", sq2, out2)
                chunk_ops(sa, "s_", sq2, gh2)
                sa.compute(api2 * 1_000_000)
                finish(sa, sq2)
                sa.compute(idle2 * 1_000_000)


    with w.actor("gpu", count=1) as gpu:
        vllm = [cached("freed"), -cached("pos")]                       # vLLM's free queue: freed least recently first, a tail first
        lmc_order = [cached("touched"), -cached("pos")]                # LMCache's LRU, a batch inserted in reverse
        def lmc_rule(cap):                                             # every second, at 0.8 of the capacity, 20 % of the keys
            return timed(1_000_000_000, tier_attr("used") * 10 >= cap * 8, max_(1, tier_attr("entries") * 2 // 10))
        g2 = tier("g2", lmc_order, capacity=G2_CHUNKS, evict=lmc_rule(G2_CHUNKS))
        g3 = tier("g3", lmc_order, capacity=G3_CHUNKS, evict=lmc_rule(G3_CHUNKS),
                  on_evict=(P.evict_lanes, lambda b: b.unlink(final(kv, victim()))))
        lm, workers = [g2, g3], {"store": (P.store_lanes, P.g2_bytes)}
        if g4:                                                         # each adapter evicts by its own rule and has its own workers
            lm.append(tier("g4", lmc_order, capacity=G4_CHUNKS, evict=lmc_rule(G4_CHUNKS),
                           on_evict=(P.evict_lanes, lambda b: b.unlink(final(kv4, victim())))))
            workers["store4"] = (P.store_lanes, P.g2_bytes)
        with gpu.timeline("tl", caches={"lm": lm, "gpu": [tier("g1", vllm, capacity=G1_BLOCKS, full="wait")]},
                          resources={"slots": P.slots},
                          # a chunk whose write has not landed stays pinned in LMCache's L1 (G2), so G2's bytes are
                          # what a run's unfinished stores may hold before one would have been dropped
                          workers=workers) as tl_all:
            with gpu.parallel("sess", P.sessions) as s:
                s.compute(s.draw("arrive", uniform(0, P.window_s * 1000)) * 1_000_000)
                cls = s.let("cls", draw(P.cls))
                nm = s.let("nm", max_(1, (inv(s, 'c.n_main', cls) + S // 2) // S))
                e = G['n_edges']
                bk = when(nm >= e[2], 3, when(nm >= e[1], 2, when(nm >= e[0], 1, 0)))
                plm = when(bk == 3, inv(s, 'g.plong_main3'), when(bk == 2, inv(s, 'g.plong_main2'),
                           when(bk == 1, inv(s, 'g.plong_main1'), inv(s, 'g.plong_main0'))))
                pmain = s.let("pmain", min_(S, max_(0, plm)))
                psub = s.let("psub", min_(S, max_(0, inv(s, 'g.plong_sub'))))
                rate_m = s.let("rate_m", inv(s, 'main.rate', cls))
                deep = s.let("deep", chance(s, P.p_nodeep[cls]) == False)              # noqa: E712
                atsz0 = s.let("atsz0", when(deep, inv(s, 'c.dfrom', cls), 1 << 62))
                has_sub = s.let("has_sub", chance(s, P.p_sub[cls]))
                spl = s.let("spl", when(has_sub, max_(1, (inv(s, 'c.sp', cls) + S // 2) // S), 0))
                nb = s.let("nb", when(has_sub, max_(1, (inv(s, 'c.n_batch', cls) + S // 2) // S), 0))
                f0 = s.let("f0", max_(1, blocks(inv(s, 'g.first_main'))))
                # a batch's draws, once a session: segment and fraction apart, so each is one draw
                seg = list(range(NG - 1))
                s.array("sp_s", nb, "bb", draw(empirical(seg, widths)))
                s.array("sp_f", nb, "bb", draw(uniform(0, FR)))
                s.array("bz_s", nb, "bb", draw(empirical(seg, widths)))
                s.array("bz_f", nb, "bb", draw(uniform(0, FR)))
                s.array("bsy_u", nb, "bb", draw(uniform(0, S)))
                A = lambda n: ArrayRef(n)
                s.bspawn = s.array("bspawn", nb, "bb", lambda bb: inv_at(A("sp_s")[bb], A("sp_f")[bb], 'g.spawn'))
                s.bsize = s.array("bsize", nb, "bb", lambda bb: max_(1, (inv_at(A("bz_s")[bb], A("bz_f")[bb], 'g.batch') + S // 2) // S))
                s.bsync = s.array("bsync", nb, "bb", lambda bb: A("bsy_u")[bb] < P.p_sync)
                f0s = s.let("f0s", cat(shared(0, min_(f0, P.mb)), fresh(f0 - min_(f0, P.mb))))
                tl = tl_all.group(s.index)
                with s.loop("i", nm) as i:
                    ln, out, api, idle, sq = chain(s, tl, i, 'main', nm, (f0, f0s), cls, dict(rate=rate_m, atsz0=atsz0, pl=pmain))
                    gh = admit(s, "m_", sq, out)
                    chunk_ops(s, "m_", sq, gh)
                    s.compute(api * 1_000_000)
                    finish(s, sq)
                    # the batches spawned after main's request i: a synchronous one holds main until its members end
                    with s.parallel("b", nb) as bt:
                        pos = bt.let("pos", min_(nm - 1, s.bspawn[bt.index] * nm // S))
                        with bt.when((pos == i) & s.bsync[bt.index]):
                            sub_agents(bt, tl, s, cls, spl, psub, True)
                        with bt.when((pos == i) & ~s.bsync[bt.index]):
                            sub_agents(bt, tl, s, cls, spl, psub, False)
                    s.compute(idle * 1_000_000)
    return w


shape("lmcache_local_disk_tiered",
      "One engine's sessions on a timeline: slots, G1 (vLLM's pool), G2 and G3 (LMCache's), G3 the directory under test.", False)
shape("lmcache_local_disk_tiered_g4",
      "One engine's sessions on a timeline: slots, G1 (vLLM's pool), G2, and two L2 adapters, G3 and G4, each a directory under test.", True)

if __name__ == "__main__":
    for wl in Workload._registry:
        wl.write()
