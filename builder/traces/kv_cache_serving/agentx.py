"""Parameters of the KV-cache abstracts from an agentic-coding trace corpus: SemiAnalysis's
InferenceX AgentX sessions (Claude Code through a proxy, published on HuggingFace as
`semianalysisai/cc-traces-weka-062126` and its 256k-context variant, Apache-2.0).

    curl -L -o traces.jsonl https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
    python agentx.py fit traces.jsonl [--traces N] [--bins 20] [--turn-bins 100] [--set name=value]... [--doc TEXT] -o fitted.agentx.params.json
    python agentx.py fit traces.jsonl --replay replay.log serve.log [--context TOKENS] [--set name=value]... -o fitted.PARAMS.json
    python agentx.py reference traces.jsonl [--traces N] [--context TOKENS] -o agentx.reference.json
    python agentx.py log serve.log -o agentx.lmcache.log
    python agentx.py holds traces.jsonl --replay replay.log serve.log [--pool TOKENS] [--clockless [--service C_IN C_OUT]] [-o FILE]

One line of the file is one session: `requests` in time order, each with `t` (seconds), `in`
(prompt tokens, a count of 64-token KV blocks), `out` (generated tokens), `hash_ids` (the
prompt's blocks, in order; ids are local to the session), `think_time` (the client's delay
before the request), `api_time`, and a `model`; or a sub-agent group, whose `requests` are
the same, run beside the main agent while it waits.

What the corpus gives, and how the abstracts take it:

  A chain is built by continuity within one agent (the main agent, or a sub-agent group)
  and one model: a request joins the open chain whose last prompt it keeps the most of,
  past the session's sub-agents' shared prefix (their tools and system prompt ~~which every
  agent of the session begins with~~; the main agent's prompts begin with their own,
  2026-10-08: the abstract's two prefixes, never written by a run),
  and starts a chain when it keeps no more than that of any (a session's first request, a
  sub-agent, a side call of a few hundred tokens between an agent's turns, a restart after
  the context was replaced). A request extends its chain when its blocks begin with the
  chain's last prompt, and rewrites it when it keeps a shorter prefix.

  reuse      1, with no none arm: a slot is one agent and serves its own chain, and a
             conversation ends by its length. Sub-agents are chains on other slots, and the
             parent's think time spans their run
  turns      the requests of a chain, in equal shares: thousands of chains are one side
             call, and the sessions of hundreds of turns hold nearly all the prefix hits
  trim       0 for a request that extends its chain; for one that keeps a shorter prefix,
             the tokens of the chain's last prompt and reply past the kept prefix, as a
             mixture with the rewrites' share
  turn_in    the tokens a continuing request adds beyond the previous reply: in − previous
             in − previous out, at least 0 ~~, and for a chain's first request in −
             sys_tokens~~ (2026-10-08: that is `first_in`).
             `out` counts tokens the model generated and did not keep (its thinking), so in
             a fifth of the turns the prompt grew by less than the reply and the clipped
             sample overstates growth; the doc of the fitted file says by how much
  first_in   a chain's first prompt past the prefix it opens with: in less its kind's
             prefix or what its session had stored of it, the longer, at least 0 (a first
             prompt is not a turn's growth; a chain of neither kind often opens with
             another sub-agent type's tools, which the session had stored)
  turn_out   the generated tokens
  think      the client's delay before a request, in nanoseconds
  sys_tokens the main agent's prefix, its tools and system prompt: the length a main-agent
             prompt most often keeps when it cuts the previous one (`main_blocks`); ~~the
             prefix a sub-agent's first prompt shares with what its session already stored
             (the system prompt and the tool definitions), its median; every chain starts
             past it~~ (2026-10-08: that measure is the sub-agents' prefix, `sub_tokens`)
  sub_tokens the prefix a sub-agent's first prompt shares with what its session already
             stored (its tools and system prompt, which are not the main agent's), its median
  kind       the chains' first prompts by the prefix they open with: the session's
             main-agent prefix (0), its sub-agents' (1), or neither (2: a tool set changed
             mid-session, or a session without sub-agents). From Anthropic's documentation:
             a prompt is its tools, then its system prompt, then its messages, and a
             sub-agent's tools and system prompt are its own (DESIGN_REVIEW.md §3.63)
  context    the corpus's cap on a prompt (the 990,016-token filter of the 062126 build;
             262,144 in the 256k variant), rounded up to the next chunk
  requests   the chains' requests, all of them; concurrency 1 (one slot), warm 0

  keep       is not in the corpus: what an engine holds is the engine's memory against its
             open sessions, and the proxy saw none. The abstract's default stands in a fit
             of the corpus. `--replay` fits what `replay_agentx.py` sent instead (its log:
             the requests, the first `sessions` of the file as as many slots) and takes
             keep from the server's, as `fit.py` does: per continuing request, the tokens
             the engine held of the chain's kept prefix past the system prompt's whole
             chunks, a prefix held whole being a lower bound. The engine can never hold
             the reply (its blocks are not the next prompt's), so the bound is the kept
             prompt, not the prompt and the reply the abstract's `prior` counts. And
             `sys_held` and `sub_held` the same way: of the requests whose engine held none
             of the conversation, the tokens of their prefix it held (a prefix the sessions
             share stays in the engine's own cache).

A fit with `--replay` records the engine it was fitted at as the file's provenance
(`fit.conditions`: the model, its context, the KV pool, LMCache's CPU tier, the slots, and the
replay's seconds per token), since the holds stand for that pool (DESIGN_REVIEW.md §3.64).
`log` reduces the server's log to what `fit --replay` and `holds` read; `holds` simulates
vLLM's prefix cache under the replay and compares its holds with the measured ones.

`reference` is the corpus's own chunk accounting, the thing a dry run at the fitted
parameters is compared with: per request, the whole 256-token chunks its prompt holds, how
many of them the session had stored before (the prefix the store would hit), and how many
are new (stored now), with totals, means, and equal shares.
"""
import argparse, collections, datetime, json, re, statistics

from fit import conditions, kept, write

BLOCK = 64                                            # tokens per hash id


def shares(values, n):
    v = sorted(values)
    n = min(n, len(v))
    cut = [round(i * len(v) / n) for i in range(n + 1)]
    return {"empirical": {"values": [round(sum(v[a:b]) / (b - a)) for a, b in zip(cut, cut[1:])], "weights": [1] * n}}


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def flatten(requests, out, own=-1, counter=None):
    """Every model request of a session, main agent and sub-agents, in time order, each with
    the id of the sub-agent group it belongs to (−1 for the main agent; a group nested in a
    group is a group of its own)."""
    counter = counter if counter is not None else [0]
    for r in requests:
        if r.get("type") == "subagent":
            g = counter[0]
            counter[0] += 1
            flatten(r["requests"], out, g, counter)
        else:
            out.append((own, r))
    if own == -1:
        out.sort(key=lambda gr: gr[1]["t"])


def shared_prefix(flat, per, model=None):
    """The hash ids of a session's shared prefix (the system prompt and the tool definitions):
    the longest prefix a sub-agent group's first prompt found stored, by `walk`'s rule over
    all the session's requests; empty for a session without sub-agents. With `model`, of
    that model's sub-agents only (another model's tokens are other tokens)."""
    store, seen, best = set(), set(), []
    for g, r in flat:
        h = r["hash_ids"]
        whole = h[:-1]
        ch = [tuple(whole[i:i + per]) for i in range(0, len(whole) - per + 1, per)]
        hit = 0
        for c in ch:
            if c in store:
                hit += 1
            else:
                break
        if g >= 0 and g not in seen and (model is None or r["model"] == model):
            seen.add(g)
            if hit * per > len(best):
                best = h[:hit * per]
        store.update(ch)
    return best


def main_blocks(path):
    """The main agent's prefix, in blocks: the length a main-agent prompt most often keeps of
    the previous one of its model when it cuts it (a restart keeps its tools and system prompt;
    511 blocks, 32,704 tokens, in 134 of the 062126 build's cuts, no other length above 19)."""
    kept = collections.Counter()
    with open(path) as f:
        for line in f:
            flat = []
            flatten(json.loads(line)["requests"], flat)
            last = {}
            for g, r in flat:
                if g >= 0:
                    continue
                h, p = r["hash_ids"], last.get(r["model"])
                if p is not None and 0 < common(p, h) < len(p) - 1:
                    kept[common(p, h)] += 1
                last[r["model"]] = h
    return kept.most_common(1)[0][0] if kept else 0


def main_prefix(flat, blocks, model=None):
    """The hash ids of a session's main-agent prefix: the first `blocks` of its first main-agent
    prompt that long (a prompt is its tools, then its system prompt, then its messages); with
    `model`, of that model's prompts only."""
    for g, r in flat:
        if g < 0 and len(r["hash_ids"]) >= blocks > 0 and (model is None or r["model"] == model):
            return r["hash_ids"][:blocks]
    return []


def prefixes(flat, per, blocks):
    """A session's prefixes by (kind, model): 0 the main agent's, 1 its sub-agents'."""
    models = {r["model"] for _, r in flat}
    return {(k, m): (main_prefix(flat, blocks, m) if k == 0 else shared_prefix(flat, per, m)) for k in (0, 1) for m in models}


def walk(path, traces, chunk_tokens, context, played=None, main=0):
    """Per request: its session and its index among the session's requests (`ti`, `fi`), its chain, whether it starts one, in, out, the blocks it keeps of the chain's
    last prompt, think time, and its chunks stored and hit under a store that never evicts;
    and the prefix of each sub-agent group's first prompt that the session had stored.

    Chains as the module doc says. `in` is a count of 64-token blocks and the last block of
    a prompt is partial as often as not, so its hash differs once the prompt has grown: a
    request that keeps all but that block extends the chain, and the block is left out of
    the chunk accounting. With `played` (a set of (session, index) pairs, `replay_agentx.py`'s),
    only the requests a replay sent: the store and the chains are the server's. A chain's
    `kind` is the prefix its first prompt opens with: 0 the session's main-agent prefix (the
    first `main` blocks, `main_prefix`), 1 its sub-agents' (`shared_prefix`), 2 neither."""
    per = max(1, chunk_tokens // BLOCK)                # hash ids per chunk
    rows, shared = [], []
    nchain = 0
    with open(path) as f:
        for ti, line in enumerate(f):
            if ti >= traces:
                break
            store = set()
            flat = []
            flatten(json.loads(line)["requests"], flat)
            sys_blocks = 0                             # the session's shared prefix, in blocks: the longest a sub-agent's first prompt found stored
            lead = prefixes(flat, per, main)
            kinds = {}                                 # chain -> its kind
            last = {}                                  # chain -> its last request; chains are keyed by (group, model) and continuity
            seen_group = set()
            for fi, (g, r) in enumerate(flat):
                if r["in"] > context or played is not None and (ti, fi) not in played:
                    continue
                h = r["hash_ids"]
                whole = h[:-1]
                ch = [tuple(whole[i:i + per]) for i in range(0, len(whole) - per + 1, per)]
                hit = 0
                for c in ch:
                    if c in store:
                        hit += 1
                    else:
                        break
                if g >= 0 and g not in seen_group:    # a sub-agent group's first prompt: what of it the session had stored
                    seen_group.add(g)
                    if hit:                            # a group that found nothing has no prefix to measure
                        shared.append(hit * chunk_tokens)
                    sys_blocks = max(sys_blocks, hit * per)
                # the agent's open chain of this model whose last prompt it keeps the most of, past the shared prefix
                best, kept = None, sys_blocks
                for c, prev in last.items():
                    if c[0] == (g, r["model"]):
                        k = common(prev["hash_ids"], h)
                        if k > kept:
                            best, kept = c, k
                if best is None:
                    best = ((g, r["model"]), nchain)
                    nchain += 1
                    prev, kept = None, 0
                    p = lead[(0 if g < 0 else 1, r["model"])]
                    kinds[best] = (0 if g < 0 else 1) if p and common(p, h) >= len(p) else 2
                else:
                    prev = last[best]
                    if kept >= len(prev["hash_ids"]) - 1:
                        kept = len(prev["hash_ids"])
                key = best
                rows.append(dict(ti=ti, fi=fi, chain=key[1], first=prev is None, kind=kinds[key], inp=r["in"], out=r["out"], kept=kept * BLOCK,
                                 prev_in=prev["in"] if prev else 0, prev_out=prev["out"] if prev else 0,
                                 think=r.get("think_time"), stored=len(ch) - hit, hit=hit))
                store.update(ch)
                last[key] = r
    return rows, shared


def replayed(log, server_log):
    """What `replay_agentx.py` sent: its sessions, and per request sent (session, index) ->
    (prompt tokens, the tokens the engine held of it when it was admitted). LMCache logs a
    lookup (`Reqid: ID, Total tokens T, Inference Engine computed tokens: H`) at every step a
    request waits for KV memory, H falling as the running requests evict its prefix, so a
    request's H is its last lookup before its first load or store (`[req_id=ID] Retrieved`,
    `Stored`). LMCache logs in schedule order and the replay in completion order: a request
    is matched by its prompt length, in order of first lookup among equals."""
    lines = open(log).read().splitlines()
    sessions = int(re.match(r"sessions (\d+)", lines[0])[1])
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    total, held, admitted = {}, {}, set()
    for l in open(server_log, errors="replace"):
        l = ansi.sub("", l)
        m = re.search(r"Reqid: (\S+), Total tokens (\d+), Inference Engine computed tokens: (\d+)", l)
        if m:
            total.setdefault(m[1], int(m[2]))
            if m[1] not in admitted:
                held[m[1]] = int(m[3])
            continue
        m = re.search(r"\[req_id=(\S+)\] (Retrieved|Stored)", l)
        if m:
            admitted.add(m[1])
    by_prompt = collections.defaultdict(collections.deque)
    for rid, t in total.items():                   # dicts keep first-lookup order
        by_prompt[t].append(held[rid])
    sent = {}
    for l in lines[1:]:
        m = re.match(r"req \d+ session (\d+) index (\d+) prompt (\d+)", l)
        if m:
            prompt = int(m[3])
            assert by_prompt[prompt], "the server's log has no request of %d prompt tokens: the two logs are not of one run" % prompt
            sent[(int(m[1]), int(m[2]))] = (prompt, by_prompt[prompt].popleft())
    assert not any(by_prompt.values()), "the server's log has requests the replay's does not"
    return sessions, sent

SENT = re.compile(r"req \d+ session (\d+) index (\d+) prompt (\d+) cached \S+ completion (\d+) ([\d.]+)s")
LOOKUP = re.compile(r"Reqid: (\S+), Total tokens (\d+), Inference Engine computed tokens: (\d+)")
ADMIT = re.compile(r"\[req_id=(\S+)\] (Retrieved|Stored)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def service(log):
    """A request's seconds in the replay as c_in × prompt tokens + c_out × reply tokens, least
    squares over the requests the replay's log has (the engine's speed at that load, queueing
    included)."""
    xs = [(int(m[3]), int(m[4]), float(m[5])) for m in map(SENT.match, open(log)) if m]
    sxx, syy, sxy = sum(p * p for p, _, _ in xs), sum(c * c for _, c, _ in xs), sum(p * c for p, c, _ in xs)
    bx, by = sum(p * t for p, _, t in xs), sum(c * t for _, c, t in xs)
    d = sxx * syy - sxy * sxy
    return (bx * syy - by * sxy) / d, (by * sxx - bx * sxy) / d


def trim(server_log, out):
    """The server's log as the kit keeps it: the lines that give the fit's conditions (vLLM's
    arguments and KV size, LMCache's configuration), and per request its first lookup, its
    last before its first load or store (what `replayed` takes), and that load or store
    (when the engine admitted it, what `holds` takes), ANSI codes removed."""
    lines = [ANSI.sub("", l).rstrip("\n") for l in open(server_log, errors="replace")]
    keep, first, last, admitted = set(), set(), {}, set()
    for i, l in enumerate(lines):
        if "non-default args" in l or "GPU KV cache size" in l or "Creating LMCacheEngine with config" in l:
            keep.add(i)
            continue
        m = LOOKUP.search(l)
        if m and m[1] not in admitted:
            if m[1] not in first:
                first.add(m[1])
                keep.add(i)
            last[m[1]] = i
            continue
        m = ADMIT.search(l)
        if m and m[1] not in admitted:
            admitted.add(m[1])
            keep.update((i, last[m[1]]) if m[1] in last else (i,))
    with open(out, "w") as f:
        f.writelines(lines[i] + "\n" for i in sorted(keep))


def stamp(line):
    m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\]", line)
    return datetime.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp() + int(m[2]) / 1000


def engine_times(log, server_log):
    """Per request sent (session, index): when the engine first looked it up and when it
    admitted it (its first load or store), in seconds, matched as `replayed` matches."""
    first, admit, total = {}, {}, {}
    for l in open(server_log, errors="replace"):
        l = ANSI.sub("", l)
        m = LOOKUP.search(l)
        if m:
            first.setdefault(m[1], stamp(l))
            total.setdefault(m[1], int(m[2]))
            continue
        m = ADMIT.search(l)
        if m:
            admit.setdefault(m[1], stamp(l))
    by_prompt = collections.defaultdict(collections.deque)
    for rid in first:
        by_prompt[total[rid]].append(rid)
    times = {}
    for m in map(SENT.match, open(log)):
        if m:
            rid = by_prompt[int(m[3])].popleft()
            times[(int(m[1]), int(m[2]))] = (first[rid], admit.get(rid, first[rid]))
    return times


def simulate(events, blocks, replies, cap):
    """vLLM's prefix cache over the corpus's blocks, `cap` of them: a block in use is counted
    by the requests using it; a freed one joins the LRU queue, a request's freed tail first
    (`free_blocks(reversed(...))`, so its head is evicted last) and leaves it on a hit; a
    request takes its prompt's and its reply's blocks when admitted, evicting from the queue's
    head, and when the queue is empty the last admitted running request is preempted, its
    blocks freed (the FCFS scheduler's `running[-1]`). `events` are (time, 1 admitted or 0
    finished, request) in time order; returns per request the blocks of its prompt held when
    admitted, and the number of preemptions. A preempted request is not readmitted here."""
    ref, queue, running, held = collections.Counter(), collections.OrderedDict(), collections.OrderedDict(), {}
    preempted = 0
    serial = 0

    def free(mine):
        for b in reversed(mine):
            ref[b] -= 1
            if not ref[b]:
                del ref[b]
                queue[b] = True

    for _, admitted, k in events:
        if not admitted:
            free(running.pop(k))
            continue
        h = 0
        for b in blocks[k]:
            if b in queue or b in ref:
                h += 1
            else:
                break
        held[k] = h
        mine = list(dict.fromkeys(blocks[k])) + [("reply", serial + j) for j in range(-(-replies[k] // BLOCK))]
        serial += len(mine)
        new = 0
        for b in mine:
            if b in queue:
                del queue[b]
            elif b not in ref:
                new += 1
        need = len(ref) + len(queue) + new - cap
        while need > 0:
            if queue:
                queue.popitem(last=False)
                need -= 1
                continue
            victims = [v for v, m in running.items() if m]
            if not victims:
                break
            free(running[victims[-1]])
            running[victims[-1]] = []
            preempted += 1
        for b in mine:
            ref[b] += 1
        running[k] = mine
    return held, preempted


def holds(a):
    """Whether the engine's pool explains the holds a replay measured (`DESIGN_REVIEW.md`
    §3.64): `simulate` on the replay's requests, with the block keys `replay_agentx.py` made,
    at `--pool` tokens (the server's own by default), on the engine's timeline (each request
    admitted when LMCache first loaded or stored for it, finished when its session's next
    arrived) or, `--clockless`, on one an abstract could compute (each session back to back,
    a request taking c_in × prompt + c_out × reply seconds, `service`'s fit or `--service`)."""
    log, server_log = a.replay
    pool = a.pool or conditions(server_log, 0)["engine_pool_tokens"]
    nsess, sent = replayed(log, server_log)
    mb, per = main_blocks(a.corpus), max(1, a.chunk_tokens // BLOCK)
    keys = {}
    with open(a.corpus) as f:
        for ti, line in enumerate(f):
            if ti == nsess:
                break
            flat = []
            flatten(json.loads(line)["requests"], flat)
            at = {b: "sub %d" % p for p, b in enumerate(shared_prefix(flat, per))}
            at.update({b: "main %d" % p for p, b in enumerate(main_prefix(flat, mb))})
            for fi, (_, r) in enumerate(flat):
                keys[(ti, fi)] = [at.get(b, "%d %d" % (ti, b)) for b in r["hash_ids"]]
    reqs = {(int(m[1]), int(m[2])): (int(m[3]), int(m[4]), float(m[5])) for m in map(SENT.match, open(log)) if m}
    order = {s: sorted(i for ss, i in reqs if ss == s) for s in range(nsess)}
    events = []
    if a.clockless:
        c_in, c_out = a.service or service(log)
        for s, idx in order.items():
            t = 0.0
            for i in idx:
                p, c, _ = reqs[(s, i)]
                events += [(t, 1, (s, i)), (t + c_in * p + c_out * c, 0, (s, i))]
                t += c_in * p + c_out * c
    else:
        times = engine_times(log, server_log)
        for s, idx in order.items():
            for j, i in enumerate(idx):
                arrived, admitted = times[(s, i)]
                done = times[(s, idx[j + 1])][0] if j + 1 < len(idx) else arrived + reqs[(s, i)][2]
                events += [(admitted, 1, (s, i)), (done, 0, (s, i))]
    events.sort(key=lambda e: (e[0], e[1]))         # at one time, finishes first
    held, preempted = simulate(events, keys, {k: v[1] for k, v in reqs.items()}, pool // BLOCK)
    rows = [(k, p, h, min(held[k] * BLOCK, p - 1)) for k, (p, h) in sorted(sent.items())]
    meas, pred = sum(r[2] for r in rows), sum(r[3] for r in rows)
    near = sum(1 for r in rows if abs(r[2] - r[3]) <= a.chunk_tokens)
    print("pool %d tokens, %s timeline: %d requests, %d preemptions; held tokens measured %d, predicted %d (%.2f); within a chunk %d (%.0f%%); "
          "nothing held (measured, predicted) %d, %d" % (pool, "clockless" if a.clockless else "the engine's", len(rows), preempted, meas, pred,
                                                        pred / max(1, meas), near, 100 * near / len(rows),
                                                        sum(1 for r in rows if r[2] < a.chunk_tokens), sum(1 for r in rows if r[3] < a.chunk_tokens)))
    if a.out:
        with open(a.out, "w") as f:
            f.writelines("%d %d %d %d %d\n" % (k[0], k[1], p, m, q) for k, p, m, q in rows)

def fit(a):
    sent = None
    if a.replay:
        a.traces, sent = replayed(*a.replay)
    mb = main_blocks(a.corpus)                     # over the whole corpus: a property of the application
    rows, shared = walk(a.corpus, a.traces, a.chunk_tokens, a.context, None if sent is None else set(sent), mb)
    sys_tokens = mb * BLOCK                        # the main agent's prefix
    sub_tokens = int(statistics.median(shared)) if shared else 0
    prefix = {0: sys_tokens, 1: sub_tokens, 2: 0}  # by kind
    whole = {k: v // a.chunk_tokens * a.chunk_tokens for k, v in prefix.items()}
    inn, first, out, think, growth, trim = [], [], [], [], [], []
    starts = 0
    kinds = collections.Counter()
    for r in rows:
        out.append(r["out"])
        if r["think"] is not None:
            think.append(round(r["think"] * 1e9))
        if r["first"]:
            starts += 1
            kinds[r["kind"]] += 1
            first.append(max(0, r["inp"] - max(prefix[r["kind"]], r["hit"] * a.chunk_tokens)))   # past its prefix, or what its session had stored
        else:
            if r["kept"] >= r["prev_in"]:                                   # extends: grew by the kept reply and the new input
                inn.append(max(0, r["inp"] - r["prev_in"] - r["prev_out"]))
                growth.append((r["inp"] - r["prev_in"], r["prev_out"] + inn[-1]))
            else:                                                           # a rewrite: the chain is cut to the kept prefix, then grows by what the request added past it
                trim.append(r["prev_in"] + r["prev_out"] - r["kept"])
                inn.append(max(0, r["inp"] - r["kept"]))
                growth.append((r["inp"] - r["kept"], inn[-1]))
    cont = n_cont = sum(1 for r in rows if not r["first"])
    rewrite = len(trim) / max(1, n_cont)
    n = len(rows)
    lengths = collections.Counter(r["chain"] for r in rows)
    context = -(-a.context // a.chunk_tokens) * a.chunk_tokens
    sessions = a.traces if a.traces < 10 ** 9 else sum(1 for _ in open(a.corpus))
    params = {"concurrency": 1, "warm": 0, "requests": n, "chunk_tokens": a.chunk_tokens, "context": context,
              "sys_prompts": sessions,
              "sys_tokens": sys_tokens, "sys_tokens_min": sys_tokens, "sys_tokens_max": sys_tokens, "sys_local": True,
              "sub_prompts": sessions, "sub_tokens": sub_tokens, "sub_tokens_min": sub_tokens, "sub_tokens_max": sub_tokens,
              "kind": {"empirical": {"values": [0, 1, 2], "weights": [kinds[0], kinds[1], kinds[2]]}},
              "reuse": {"mixture": [{"weight": 0, "dist": None}, {"weight": 1, "dist": {"const": 1}}]},
              "turns": shares(list(lengths.values()), a.turn_bins), "retain": n,
              "turn_in": shares(inn, a.bins), "first_in": shares(first, a.bins), "turn_out": shares(out, a.bins), "think": shares(think, a.bins),
              "trim": {"mixture": [{"weight": round(1 - rewrite, 4), "dist": {"const": 0}}, {"weight": round(rewrite, 4), "dist": shares(trim, a.bins)}]}}
    if sent is not None:                           # the replay's sessions are its slots, served back to back; keep is the engine's
        keep, pheld = [], {0: [], 1: []}
        for r in rows:
            prompt, held = sent[(r["ti"], r["fi"])]
            if r["kind"] < 2 and held < whole[r["kind"]] + a.chunk_tokens:   # no whole chunk of the conversation held: what of its prefix the engine held
                pheld[r["kind"]].append(min(held, prefix[r["kind"]]))
            if not r["first"]:
                prompt, held = sent[(r["ti"], r["fi"])]
                avail = min(r["kept"], prompt)         # the engine can hold the kept prefix only: the reply's blocks are not the next prompt's
                got = min(held, avail)
                keep.append((max(0, got - whole[r["kind"]]), got + a.block > avail))   # the engine counts whole blocks
        held_of = lambda v: shares(v, a.bins) if v else {"const": 0}
        params.update(concurrency=a.traces, requests=round(n / a.traces), keep=kept(keep, a.bins, context),
                      sys_held=held_of(pheld[0]), sub_held=held_of(pheld[1]),
                      sys_prompts=1, sub_prompts=1,       # the replay's sessions share both prefixes (replay_agentx.py)
                      think={"const": 0} if not a.time_scale else shares([t * a.time_scale for t in think], a.bins))
    for s in a.set:
        k, v = s.split("=", 1)
        params[k] = json.loads(v)
    over = sum(b for _, b in growth) / max(1, sum(g for g, _ in growth))
    doc = a.doc or ("Fitted by agentx.py to %s (%d sessions): %d requests in %d chains; %.1f%% of continuing requests cut their chain "
                    "(trim) and the chain grows by %.0f%% of the corpus's prompt growth (the reply counted whole where the model kept "
                    "less of it); sys_tokens the main agent's prefix (the prefix its cuts most often keep), sub_tokens the median prefix a "
                    "sub-agent's first prompt shares with its session, kind the chains' first prompts by the prefix they open with; keep not measured "
                    "(the proxy saw no engine)." % (
                        a.corpus.rsplit("/", 1)[-1], sessions, n, len({r["chain"] for r in rows}), 100 * rewrite, 100 * over))
    if sent is not None and not a.doc:
        doc = doc.replace("keep not measured (the proxy saw no engine).", "the requests replay_agentx.py sent (%d sessions as %d slots, %d "
                          "requests each, back to back), and keep from the server's log: of %d continuing requests the engine held the "
                          "whole kept prefix in %d." % (a.traces, a.traces, params["requests"], len(keep), sum(w for _, w in keep)))
    conds = None
    if sent is not None:                           # what the holds stand for: the engine, its pool, and its load (§3.64)
        conds = conditions(a.replay[1], a.traces, a.condition)
        c_in, c_out = service(a.replay[0])
        conds.setdefault("latency_s_per_prompt_token", float("%.3g" % c_in))
        conds.setdefault("latency_s_per_reply_token", float("%.3g" % c_out))
    write(a.out, a.abstract, doc, params, conds)
    print("requests %d, chains %d (one request: %d, longest %d), rewrites %.4f of continuations (trim mean %.0f), sys_tokens %d, tokens in %d "
          "(mean %.1f) out %d (mean %.1f), think mean %.1f s, chain growth / corpus growth %.3f" % (
              n, len(lengths), sum(1 for v in lengths.values() if v == 1), max(lengths.values()), rewrite, sum(trim) / max(1, len(trim)),
              sys_tokens, sum(inn), sum(inn) / n,
              sum(out), sum(out) / n, sum(think) / 1e9 / max(1, len(think)), over))


def reference(a):
    rows, shared = walk(a.corpus, a.traces, a.chunk_tokens, a.context)
    n = len(rows)
    stored, hit, inp = [r["stored"] for r in rows], [r["hit"] for r in rows], [r["inp"] for r in rows]
    ref = {"corpus": a.corpus.rsplit("/", 1)[-1], "traces": a.traces if a.traces < 10 ** 9 else sum(1 for _ in open(a.corpus)),
           "chunk_tokens": a.chunk_tokens, "context": a.context, "requests": n, "chains": len({r["chain"] for r in rows}),
           "first": sum(1 for r in rows if r["first"]),
           "chunks_stored": {"total": sum(stored), "mean": round(sum(stored) / n, 2), "shares": shares(stored, a.bins)["empirical"]["values"]},
           "chunks_hit": {"total": sum(hit), "mean": round(sum(hit) / n, 2), "shares": shares(hit, a.bins)["empirical"]["values"]},
           "prompt_tokens": {"mean": round(sum(inp) / n), "shares": shares(inp, a.bins)["empirical"]["values"]},
           "out_tokens": {"total": sum(r["out"] for r in rows), "mean": round(sum(r["out"] for r in rows) / n, 1)},
           "subagent_shared_prefix_tokens": {"median": int(statistics.median(shared)) if shared else 0, "n": len(shared)}}
    with open(a.out, "w") as f:
        json.dump(ref, f, indent=1)
        f.write("\n")
    print("requests %d, chains %d, chunks stored %d (%.2f per request), hit %d (%.1f per request), prompt tokens mean %d" % (
        n, ref["chains"], sum(stored), sum(stored) / n, sum(hit), sum(hit) / n, sum(inp) / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fit", "reference", "holds", "log"])
    ap.add_argument("corpus")
    ap.add_argument("--abstract", default="kv_cache_serving")
    ap.add_argument("--traces", type=int, default=10 ** 9, help="use the first N sessions")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--turn-bins", type=int, default=100, help="shares of `turns`: the long sessions hold the prefix hits, so its tail needs resolving")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=990_016, help="requests with longer prompts are skipped (the corpus's own cap)")
    ap.add_argument("--replay", nargs=2, metavar=("LOG", "SERVER_LOG"),
                    help="fit: the requests replay_agentx.py sent (its log), and keep from the server's (LMCache's line per request)")
    ap.add_argument("--time-scale", type=float, default=0.0, help="--replay: the replay's own (0: think 0)")
    ap.add_argument("--block", type=int, default=16, help="--replay: tokens in a block of the engine's own cache")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--doc", default="")
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=VALUE", help="fit --replay: a condition the server's log lacks")
    ap.add_argument("--pool", type=int, default=0, help="holds: the engine's KV pool in tokens (default: the server's)")
    ap.add_argument("--clockless", action="store_true", help="holds: sessions back to back at --service's cost, not the engine's timeline")
    ap.add_argument("--service", type=float, nargs=2, metavar=("C_IN", "C_OUT"), help="holds --clockless: seconds per prompt and per reply token")
    ap.add_argument("-o", "--out", help="fit, reference: the file; holds: per request, session index prompt measured predicted")
    a = ap.parse_args()
    if a.what != "holds" and not a.out or a.what == "holds" and not a.replay:
        ap.error("fit, reference, and log need -o; holds needs --replay")
    if a.what == "log":                            # its one argument is the server's log
        return trim(a.corpus, a.out)
    {"fit": fit, "reference": reference, "holds": holds}[a.what](a)


if __name__ == "__main__":
    main()
