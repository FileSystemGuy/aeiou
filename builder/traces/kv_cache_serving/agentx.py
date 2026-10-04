"""Parameters of the KV-cache abstracts from an agentic-coding trace corpus: SemiAnalysis's
InferenceX AgentX sessions (Claude Code through a proxy, published on HuggingFace as
`semianalysisai/cc-traces-weka-062126` and its 256k-context variant, Apache-2.0).

    curl -L -o traces.jsonl https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
    python agentx.py fit traces.jsonl [--traces N] [--bins 20] [--turn-bins 100] [--set name=value]... [--doc TEXT] -o fitted.agentx.params.json
    python agentx.py reference traces.jsonl [--traces N] [--context TOKENS] -o agentx.reference.json

One line of the file is one session: `requests` in time order, each with `t` (seconds), `in`
(prompt tokens, a count of 64-token KV blocks), `out` (generated tokens), `hash_ids` (the
prompt's blocks, in order; ids are local to the session), `think_time` (the client's delay
before the request), `api_time`, and a `model`; or a sub-agent group, whose `requests` are
the same, run beside the main agent while it waits.

What the corpus gives, and how the abstracts take it:

  A chain is built by continuity within one agent (the main agent, or a sub-agent group)
  and one model: a request joins the open chain whose last prompt it keeps the most of,
  past the session's shared prefix (the system prompt and the tool definitions, which every
  agent of the session begins with: the abstract's system prompt, never written by a run),
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
  turn_in    the tokens a request adds beyond the previous reply: in − previous in −
             previous out, at least 0, and for a chain's first request in − sys_tokens.
             `out` counts tokens the model generated and did not keep (its thinking), so in
             a fifth of the turns the prompt grew by less than the reply and the clipped
             sample overstates growth; the doc of the fitted file says by how much
  turn_out   the generated tokens
  think      the client's delay before a request, in nanoseconds
  sys_tokens the prefix a sub-agent's first prompt shares with what its session already
             stored (the system prompt and the tool definitions), its median; every chain
             starts past it
  context    the corpus's cap on a prompt (the 990,016-token filter of the 062126 build;
             262,144 in the 256k variant), rounded up to the next chunk
  requests   the chains' requests, all of them; concurrency 1 (one slot), warm 0

  keep       is not in the corpus: what an engine holds is the engine's memory against its
             open sessions, and the proxy saw none. The abstract's default stands until a
             replay on a GPU measures it (`replay.py`).

`reference` is the corpus's own chunk accounting, the thing a dry run at the fitted
parameters is compared with: per request, the whole 256-token chunks its prompt holds, how
many of them the session had stored before (the prefix the store would hit), and how many
are new (stored now), with totals, means, and equal shares.
"""
import argparse, collections, json, statistics

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


def walk(path, traces, chunk_tokens, context):
    """Per request: its chain, whether it starts one, in, out, the blocks it keeps of the chain's
    last prompt, think time, and its chunks stored and hit under a store that never evicts;
    and the prefix of each sub-agent group's first prompt that the session had stored.

    Chains as the module doc says. `in` is a count of 64-token blocks and the last block of
    a prompt is partial as often as not, so its hash differs once the prompt has grown: a
    request that keeps all but that block extends the chain, and the block is left out of
    the chunk accounting."""
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
            last = {}                                  # chain -> its last request; chains are keyed by (group, model) and continuity
            seen_group = set()
            for g, r in flat:
                if r["in"] > context:
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
                else:
                    prev = last[best]
                    if kept >= len(prev["hash_ids"]) - 1:
                        kept = len(prev["hash_ids"])
                key = best
                rows.append(dict(chain=key[1], first=prev is None, inp=r["in"], out=r["out"], kept=kept * BLOCK,
                                 prev_in=prev["in"] if prev else 0, prev_out=prev["out"] if prev else 0,
                                 think=r.get("think_time"), stored=len(ch) - hit, hit=hit))
                store.update(ch)
                last[key] = r
    return rows, shared


def fit(a):
    rows, shared = walk(a.corpus, a.traces, a.chunk_tokens, a.context)
    sys_tokens = int(statistics.median(shared)) if shared else 0
    sys_whole = sys_tokens // a.chunk_tokens * a.chunk_tokens
    inn, out, think, growth, trim = [], [], [], [], []
    starts = 0
    for r in rows:
        out.append(r["out"])
        if r["think"] is not None:
            think.append(round(r["think"] * 1e9))
        if r["first"]:
            starts += 1
            inn.append(max(0, r["inp"] - sys_tokens))
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
    params = {"concurrency": 1, "warm": 0, "requests": n, "chunk_tokens": a.chunk_tokens, "context": context,
              "sys_prompts": a.traces if a.traces < 10 ** 9 else sum(1 for _ in open(a.corpus)),
              "sys_tokens": sys_tokens, "sys_tokens_min": sys_tokens, "sys_tokens_max": sys_tokens, "sys_local": True,
              "reuse": {"mixture": [{"weight": 0, "dist": None}, {"weight": 1, "dist": {"const": 1}}]},
              "turns": shares(list(lengths.values()), a.turn_bins), "retain": n,
              "turn_in": shares(inn, a.bins), "turn_out": shares(out, a.bins), "think": shares(think, a.bins),
              "trim": {"mixture": [{"weight": round(1 - rewrite, 4), "dist": {"const": 0}}, {"weight": round(rewrite, 4), "dist": shares(trim, a.bins)}]}}
    for s in a.set:
        k, v = s.split("=", 1)
        params[k] = json.loads(v)
    over = sum(b for _, b in growth) / max(1, sum(g for g, _ in growth))
    doc = a.doc or ("Fitted by agentx.py to %s (%d sessions): %d requests in %d chains; %.1f%% of continuing requests cut their chain "
                    "(trim) and the chain grows by %.0f%% of the corpus's prompt growth (the reply counted whole where the model kept "
                    "less of it); sys_tokens the median prefix a sub-agent's first prompt shares with its session; keep not measured "
                    "(the proxy saw no engine)." % (
                        a.corpus.rsplit("/", 1)[-1], params["sys_prompts"], n, len({r["chain"] for r in rows}), 100 * rewrite, 100 * over))
    with open(a.out, "w") as f:                    # a parameter to a line
        f.write('{"params_version": 1, "abstract": %s,\n "doc": %s,\n "params": {\n%s}}\n' % (
            json.dumps(a.abstract), json.dumps(doc), ",\n".join("  %s: %s" % (json.dumps(k), json.dumps(v)) for k, v in params.items())))
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
    ap.add_argument("what", choices=["fit", "reference"])
    ap.add_argument("corpus")
    ap.add_argument("--abstract", default="kv_cache_serving")
    ap.add_argument("--traces", type=int, default=10 ** 9, help="use the first N sessions")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--turn-bins", type=int, default=100, help="shares of `turns`: the long sessions hold the prefix hits, so its tail needs resolving")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=990_016, help="requests with longer prompts are skipped (the corpus's own cap)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--doc", default="")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    (fit if a.what == "fit" else reference)(a)


if __name__ == "__main__":
    main()
