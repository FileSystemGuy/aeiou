"""Parameters of the KV-cache abstracts from a replay.py run: its log and the server's.

    python fit.py replay.log serve.log [--abstract kv_cache_serving] [--bins 20]
                  [--set name=value]... [--condition name=value]... [--doc TEXT] -o fitted.params.json

From the load's log (every request's prompt and completion tokens, and how many requests ago
its conversation was last served):

  requests, context, sys_tokens   as logged
  reuse      none with the share of requests that began a conversation because the last one
             in their place had ended (not counting the first round, nor conversations cut
             by the context, which the abstract cuts itself); else the one distance of the
             run, the number of open conversations
  turn_in    the prompt tokens a request adds to its conversation (the user's turn and the
             chat template's), in `--bins` equal shares of the sorted sample, each share's mean
  turn_out   the completion tokens, the same way

From the server's log (LMCache's line per request: `Inference Engine computed tokens`, what
the engine still held of the prompt; `grep "Inference Engine computed"` of it is enough):

  keep       the tokens of a returning conversation (its last prompt and reply, less the
             system prompt's whole chunks) that the engine held, in equal shares like the
             lengths. A conversation held whole says only that the engine would have kept
             at least that many: the distribution is the product-limit estimate over such
             censored observations, and the share beyond the longest is given the context.

`--set` adds what belongs to the model and the backend (chunk_bytes, meta_bytes, buf).

The fit's conditions go into the file's `provenance` (never compared): what an engine holds
is its KV pool against the load, so `keep` stands for the pool it was fitted at
(`DESIGN_REVIEW.md` §3.64). They come from the server's log where it has them, else from
`--condition NAME=VALUE`, which also overrides.
"""
import argparse, json, re


def conditions(server_log, slots, given=()):
    """The conditions a fit of the engine's holds is valid at: the model and its context, the
    engine's KV pool (vLLM's `GPU KV cache size`, and the `--kv-cache-memory-bytes` that set
    it), LMCache's CPU tier in bytes (0 with `local_cpu` off; its sizes are GiB), and the
    slots one engine serves. Taken from the server's log, then `given` (NAME=VALUE, JSON)."""
    found = {}
    pats = [("model", r"'model': '([^']+)'", str), ("max_model_len", r"'max_model_len': (\d+)", int),
            ("kv_cache_memory_bytes", r"'kv_cache_memory_bytes': (\d+)", int),
            ("engine_pool_tokens", r"GPU KV cache size: ([\d,]+) tokens", lambda s: int(s.replace(",", ""))),
            ("local_cpu", r"'local_cpu': (True|False)", lambda s: s == "True"), ("max_local_cpu_size", r"'max_local_cpu_size': ([\d.]+)", float)]
    for l in open(server_log, errors="replace"):
        for k, p, f in pats:
            if k not in found:
                m = re.search(p, l)
                if m:
                    found[k] = f(m[1])
    c = {k: found[k] for k in ("model", "max_model_len", "kv_cache_memory_bytes", "engine_pool_tokens") if k in found}
    if "local_cpu" in found:
        c["cpu_tier_bytes"] = int(found.get("max_local_cpu_size", 0) * 1024 ** 3) if found["local_cpu"] else 0
    c["slots_per_engine"] = slots
    for s in given:
        k, v = s.split("=", 1)
        c[k] = json.loads(v)
    order = ["model", "max_model_len", "kv_cache_memory_bytes", "engine_pool_tokens", "cpu_tier_bytes", "slots_per_engine"]
    return dict(sorted(c.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))


def write(path, abstract, doc, params, conds=None):
    """A parameter to a line; the fit's conditions, when known, as the file's provenance."""
    with open(path, "w") as f:
        f.write('{"params_version": 1, "abstract": %s,\n "doc": %s,\n "params": {\n%s}%s}\n' % (
            json.dumps(abstract), json.dumps(doc), ",\n".join("  %s: %s" % (json.dumps(k), json.dumps(v)) for k, v in params.items()),
            ',\n "provenance": {"conditions": %s}' % json.dumps(conds) if conds else ""))


def shares(values, n):
    v = sorted(values)
    n = min(n, len(v))
    cut = [round(i * len(v) / n) for i in range(n + 1)]
    return {"empirical": {"values": [round(sum(v[a:b]) / (b - a)) for a, b in zip(cut, cut[1:])], "weights": [1] * n}}


def kept(samples, n, most):
    """`n` equal shares of the tokens an engine keeps, from (tokens held, held whole) pairs:
    Kaplan-Meier, a conversation held whole being a lower bound."""
    alive, at_risk, steps = 1.0, len(samples), []      # steps: (tokens, probability of exactly that many)
    for t, whole in sorted(samples, key=lambda s: (s[0], s[1])):
        if not whole:
            steps.append((t, alive / at_risk))
            alive -= alive / at_risk
        at_risk -= 1
    steps.append((most, alive))
    values, i, left = [], 0, steps[0][1]
    for _ in range(n):                                 # each share is 1/n of the probability; its value is its mean
        need, total = 1 / n, 0.0
        while need > 1e-12:
            take = min(need, left)
            total += take * steps[i][0]
            need -= take
            left -= take
            if left <= 1e-12 and i + 1 < len(steps):
                i += 1
                left = steps[i][1]
            elif left <= 1e-12:
                total += need * steps[i][0]
                need = 0
        values.append(round(total * n))
    return {"empirical": {"values": values, "weights": [1] * n}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("server_log")
    ap.add_argument("--abstract", default="kv_cache_serving")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--block", type=int, default=16, help="tokens in a block of the engine's own cache")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--doc", default="")
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=VALUE", help="a condition of the fit the server's log lacks")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    engine = [(int(m[1]), int(m[2])) for m in (re.search(r"Total tokens (\d+), Inference Engine computed tokens: (\d+)", ansi.sub("", l))
                                               for l in open(a.server_log, errors="replace")) if m]
    lines = open(a.log).read().splitlines()
    context = int(re.search(r"context (\d+)", lines[0])[1])
    sys_tokens = json.loads(re.search(r"system prompts (\[.*\])", lines[0])[1])
    assert len(set(sys_tokens)) == 1, "system prompts of several lengths: sys_tokens_min/max need a look"
    sys_whole = sys_tokens[0] // a.chunk_tokens * a.chunk_tokens
    last, ago, inn, out, keep = {}, set(), [], [], []
    ended = cut = False
    began = 0                                          # conversations begun where another had ended
    for l in lines[1:]:
        m = re.match(r"conv \d+ (ended|cut)", l)
        if m:
            ended, cut = ended or m[1] == "ended", cut or m[1] == "cut"
            continue
        m = re.match(r"req (\d+) conv (\d+) turn \d+ ago (\S+) prompt (\d+) cached \S+ completion (\d+)", l)
        if not m:
            continue
        conv, d, prompt, completion = int(m[2]), m[3], int(m[4]), int(m[5])
        total, held = engine[len(out)]
        assert total == prompt, "the two logs are not of one run: request %s" % m[1]
        if d == "-":
            began += ended and not cut
            inn.append(prompt - sys_tokens[0])
        else:
            ago.add(int(d))
            inn.append(prompt - last[conv])
            keep.append((max(0, held - sys_whole), held + a.block > last[conv]))   # the engine counts whole blocks
        out.append(completion)
        last[conv] = prompt + completion
        ended = cut = False
    n = len(out)
    assert len(engine) == n, "the server's log has %d requests, the load's %d" % (len(engine), n)
    assert len(ago) == 1, "conversations returned after %s requests: the abstracts take one distance" % sorted(ago)
    active = ago.pop()
    none = began / (n - active)
    params = {"concurrency": 1, "warm": 0, "requests": n, "chunk_tokens": a.chunk_tokens, "context": context,
              "sys_prompts": len(sys_tokens), "sys_tokens": sys_tokens[0], "sys_tokens_min": sys_tokens[0], "sys_tokens_max": sys_tokens[0],
              "sys_local": True,
              "reuse": {"mixture": [{"weight": round(none, 4), "dist": None}, {"weight": round(1 - none, 4), "dist": {"const": active}}]},
              "keep": kept(keep, a.bins, context), "retain": n,
              "turn_in": shares(inn, a.bins), "turn_out": shares(out, a.bins)}
    for s in a.set:
        k, v = s.split("=", 1)
        params[k] = json.loads(v)
    write(a.out, a.abstract, a.doc, params, conditions(a.server_log, active, a.condition))
    print("requests %d, open conversations %d, begun after an ended one %d, tokens in %d out %d (mean %.1f, %.1f), returning conversations held whole %d of %d" % (
        n, active, began, sum(inn), sum(out), sum(inn) / n, sum(out) / n, sum(w for _, w in keep), len(keep)))


if __name__ == "__main__":
    main()
