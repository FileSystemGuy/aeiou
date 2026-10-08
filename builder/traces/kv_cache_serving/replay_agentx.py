"""A replay of the AgentX corpus (`agentx.py`) through a vLLM server, for a trace of the
server under an agentic load and for `keep` (what the engine still holds, from LMCache's
log), which the corpus cannot give.

    python replay_agentx.py traces.jsonl [--url http://127.0.0.1:8000] [--model NAME]
                            [--sessions 8] [--skip 0] [--requests 300] [--max-context 131072]
                            [--max-reply 4096] [--time-scale 0.0] [--chunk-tokens 256] [--vocab 151643]

The corpus has no text: a request is its prompt's 64-token blocks as hash ids, local to
the session (every session numbers its blocks from 0). A block's tokens are generated from
its session and its id (64 ids drawn from the vocabulary by a generator seeded with both),
so two requests of a session that share a prefix of blocks send the same prefix of tokens,
and the server's prefix cache and LMCache see the corpus's reuse structure. A block of the
session's shared prefix (`agentx.shared_prefix`: the system prompt and the tool
definitions) is generated from its position in that prefix instead, so the sessions share
one system prompt, as far as the shorter of two prefixes goes: the corpus cannot say what
its sessions share, and a public corpus with the text can (`sammshen/lmcache-agentic-traces`:
three system prompts serve 756 of its 767 sessions). The replay of 2026-10-07 seeded by the
id alone, and sessions whose ids happened to line up shared whole conversations
(`DESIGN_REVIEW.md` §3.63). The
reply a request got is not what the next prompt holds (its blocks have their own ids), so
the engine misses on the previous reply's blocks and recomputes them; AIPerf's replay of
the same corpus has the same property.

`--sessions` sessions are played at once, each on a thread, from `--skip` into the file;
a session's requests go in time order, the main agent's and its sub-agents' alike, with
the corpus's `think_time` before each scaled by `--time-scale` (0: back to back). A
request whose prompt exceeds `--max-context` is skipped. A reply is asked for with
`max_tokens` = the corpus's `out` (at most `--max-reply`) and `ignore_eos`.

Prints one line per request: request, session, the request's index in it, prompt tokens,
cached tokens when the server reports them, completion tokens, seconds. Run 2026-10-07 and, corrected,
2026-10-08 on an 8 GB GPU at 128k context (the kit's README, `DESIGN_REVIEW.md` §3.63).
"""
import argparse, json, random, threading, time, urllib.request

from agentx import BLOCK, flatten, shared_prefix


def tokens_of(key, vocab, lo=1000):
    rng = random.Random(key)                       # a str seed: the same tokens in every process
    return [rng.randrange(lo, vocab) for _ in range(64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--requests", type=int, default=300, help="in all, over the sessions")
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--max-reply", type=int, default=4096)
    ap.add_argument("--time-scale", type=float, default=0.0)
    ap.add_argument("--chunk-tokens", type=int, default=256, help="LMCache's chunk, for the session's shared prefix (agentx.shared_prefix)")
    ap.add_argument("--vocab", type=int, default=151643, help="token ids are drawn below it (Qwen2.5's; the server refuses ids past its vocabulary)")
    a = ap.parse_args()

    def post(path, body):
        req = urllib.request.Request(a.url + path, json.dumps(dict(body, model=a.model)).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=3600))

    limit = post("/tokenize", {"prompt": "x"})["max_model_len"]
    vocab = a.vocab
    sessions = []
    with open(a.corpus) as f:
        for i, line in enumerate(f):
            if i < a.skip:
                continue
            if len(sessions) == a.sessions:
                break
            flat = []
            flatten(json.loads(line)["requests"], flat)
            sys_at = {b: p for p, b in enumerate(shared_prefix(flat, max(1, a.chunk_tokens // BLOCK)))}
            sessions.append(([r for _, r in flat], i, sys_at))
    print("sessions %d context %d (server %d) vocab %d shared prefix %s" % (
        len(sessions), a.max_context, limit, vocab, ",".join(str(len(s[2]) * BLOCK) for s in sessions)), flush=True)

    lock = threading.Lock()
    count = [0]

    def play(si, session):
        reqs, ti, sys_at = session
        for ri, r in enumerate(reqs):
            with lock:
                if count[0] >= a.requests:
                    return
                n = count[0]
                count[0] += 1
            if r["in"] > min(a.max_context, limit):
                continue
            if a.time_scale and r.get("think_time"):
                time.sleep(r["think_time"] * a.time_scale)
            ids = [t for b in r["hash_ids"] for t in tokens_of("sys %d" % sys_at[b] if b in sys_at else "%d %d" % (ti, b), vocab)]
            want = max(1, min(a.max_reply, r["out"], limit - len(ids)))
            t0 = time.monotonic()
            out = post("/v1/completions", {"prompt": ids, "max_tokens": want, "ignore_eos": True, "temperature": 0})
            u = out["usage"]
            print("req %d session %d index %d prompt %d cached %s completion %d %.2fs" % (
                n, si, ri, u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens"), u["completion_tokens"],
                time.monotonic() - t0), flush=True)

    threads = [threading.Thread(target=play, args=(si, session)) for si, session in enumerate(sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
