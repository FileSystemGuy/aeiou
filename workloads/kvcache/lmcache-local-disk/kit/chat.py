"""A multi-turn chat load for a vLLM server: `--conversations` conversations taken round-robin
for `--turns` turns each, every conversation starting from one of `--system-prompts` shared
system prompts. Text is generated from a seed, so two runs send the same requests.

    python chat.py [--url http://127.0.0.1:8000] [--model NAME] [--conversations 8] [--turns 5]
                   [--system-prompts 2] [--system-words 450] [--turn-words 180] [--max-tokens 100]
                   [--concurrency 1] [--seed 1] [--save FILE | --replay FILE]

`--save` writes the replies; `--replay` puts those into the history in place of this server's
own, so that a second server is sent the first one's requests byte for byte (a model's
replies differ from run to run once its KV cache is loaded instead of computed).

Prints one line per request: conversation, turn, prompt tokens, cached tokens when the server
reports them, completion tokens, seconds.
"""
import argparse, concurrent.futures, json, random, time, urllib.request

WORDS = ("storage file block read write cache token model layer tensor graph index query vector "
         "shard batch epoch sample loader buffer page fault mount server client lock lease open close "
         "stream chunk prefix suffix hash table queue thread process kernel driver device network").split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--conversations", type=int, default=8)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--system-prompts", type=int, default=2)
    ap.add_argument("--system-words", type=int, default=450)
    ap.add_argument("--turn-words", type=int, default=180)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save")
    ap.add_argument("--replay")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    systems = ["You are an assistant for a storage team. Notes: " + text(rng, a.system_words) for _ in range(a.system_prompts)]
    convs = [[{"role": "system", "content": systems[c % a.system_prompts]}] for c in range(a.conversations)]
    turns = [[text(random.Random(a.seed * 1000003 + c * 1009 + t), a.turn_words) for t in range(a.turns)] for c in range(a.conversations)]

    said = json.load(open(a.replay)) if a.replay else {}
    replies = {}

    def one(c, t):
        convs[c].append({"role": "user", "content": "Summarize in one paragraph: " + turns[c][t]})
        body = json.dumps({"model": a.model, "messages": convs[c], "max_tokens": a.max_tokens, "temperature": 0}).encode()
        t0 = time.time()
        req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        reply = r["choices"][0]["message"]["content"]
        replies["%d %d" % (c, t)] = reply
        convs[c].append({"role": "assistant", "content": said.get("%d %d" % (c, t), reply)})
        u = r["usage"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        return "conv %d turn %d prompt %d cached %s completion %d %.2fs" % (c, t, u["prompt_tokens"], cached, u["completion_tokens"], time.time() - t0)

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        for t in range(a.turns):                       # a conversation's turns are in order; conversations interleave
            for line in ex.map(lambda c: one(c, t), range(a.conversations)):
                print(line, flush=True)
    if a.save:
        json.dump(replies, open(a.save, "w"))


if __name__ == "__main__":
    main()
