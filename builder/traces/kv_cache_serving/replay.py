"""A replay of public chat conversations (ShareGPT) through a vLLM server: the users' turns
are the dataset's, a reply is generated to the length of the dataset's reply, and
`--active` conversations are open at a time, served in turn.

    python replay.py ShareGPT_V3_unfiltered_cleaned_split.json [--url http://127.0.0.1:8000]
                     [--model NAME] [--requests 300] [--active 8] [--system-prompts 2]
                     [--system-words 450] [--max-reply 1024] [--seed 1]
                     [--save FILE | --replay FILE]

The file is the one vLLM's own benchmarks use (huggingface.co/datasets/anon8231489123/
ShareGPT_Vicuna_unfiltered). It holds long conversations in parts that overlap by one
message; they are joined again, and a conversation is used if it alternates human and gpt
from a human turn. The order of conversations is a shuffle by `--seed`.

What the dataset gives: the tokens of every user turn and of every reply, and the turns of
a conversation. What it does not give: when the turns were sent (it has no timestamps), so
how conversations interleave is this load's choice: `--active` conversations are open and
are served in turn, one request each; a conversation that ends is replaced, in its place in
the round, by the next of the dataset. A conversation therefore returns `--active` requests
after its last turn. Nor system prompts, which are synthetic here as in chat.py.

A reply is asked for with `max_tokens` = the tokens of the dataset's reply (at most
`--max-reply`) and `ignore_eos`, so reply lengths are the dataset's whatever the model would
have said, and with `skip_special_tokens` off, so that the reply put into the history is,
token for token, what the engine generated and still holds (an end-of-turn token the model
emitted on the way is dropped from the text otherwise, and the next prompt is then shorter
than the engine's copy of the conversation). A conversation ends early when its next prompt and reply would not fit the
server's context (`cut` in the log).

`--save` and `--replay` are chat.py's: the replies of one run put into the history of
another, so that a second server is sent the first one's requests byte for byte.

Prints the system prompts' token counts, then one line per request: request, conversation,
turn, requests since the conversation's last turn (`-` for its first), prompt tokens, cached
tokens when the server reports them, completion tokens, seconds.
"""
import argparse, collections, json, random, time, urllib.request

from chat import text


def conversations(path):
    """The dataset's whole conversations as lists of (user, reply), in id order."""
    parts = collections.defaultdict(list)
    for c in json.load(open(path)):
        base, at = c["id"].rsplit("_", 1)
        parts[base].append((int(at), c["conversations"]))
    out = []
    for base in sorted(parts):
        m = []
        for at, msgs in sorted(parts[base], key=lambda p: p[0]):       # a part's suffix is the index of its first message
            m = m[:at] + msgs
        if len(m) >= 2 and len(m) % 2 == 0 and all(x["from"] == ("human", "gpt")[i % 2] for i, x in enumerate(m)):
            out.append([(m[i]["value"], m[i + 1]["value"]) for i in range(0, len(m), 2)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--active", type=int, default=8)
    ap.add_argument("--system-prompts", type=int, default=2)
    ap.add_argument("--system-words", type=int, default=450)
    ap.add_argument("--max-reply", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save")
    ap.add_argument("--replay")
    a = ap.parse_args()

    def post(path, body):
        req = urllib.request.Request(a.url + path, json.dumps(dict(body, model=a.model)).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))

    rng = random.Random(a.seed)
    systems = ["You are an assistant for a storage team. Notes: " + text(rng, a.system_words) for _ in range(a.system_prompts)]
    data = conversations(a.data)
    rng.shuffle(data)
    limit = post("/tokenize", {"prompt": "x"})["max_model_len"]
    print("conversations %d context %d system prompts %s" % (
        len(data), limit, [post("/tokenize", {"messages": [{"role": "system", "content": s}], "add_generation_prompt": False})["count"] for s in systems]), flush=True)

    said = json.load(open(a.replay)) if a.replay else {}
    replies = {}
    taken = 0                                          # conversations taken from the dataset so far

    def start():
        nonlocal taken
        c = {"id": taken, "turns": data[taken], "t": 0, "last": None,
             "msgs": [{"role": "system", "content": systems[taken % a.system_prompts]}]}
        taken += 1
        return c

    def ready(c):
        """The next request of a conversation, or None when it has ended or no longer fits the context."""
        while True:
            if c["t"] == len(c["turns"]):
                return None
            user, reply = c["turns"][c["t"]]
            msgs = c["msgs"] + [{"role": "user", "content": user}]
            want = max(1, min(a.max_reply, post("/tokenize", {"prompt": reply, "add_special_tokens": False})["count"]))
            if post("/tokenize", {"messages": msgs, "add_generation_prompt": True})["count"] + want <= limit:
                return msgs, want
            if c["t"]:
                return None                            # cut: the context is full
            c.update(start())                          # a first turn that never fits: take another conversation

    active = [start() for _ in range(a.active)]
    for r in range(a.requests):
        i = r % a.active                               # the open conversations in turn
        while True:
            c = active[i]
            nxt = ready(c)
            if nxt is not None:
                break
            print("conv %d %s after %d turns" % (c["id"], "ended" if c["t"] == len(c["turns"]) else "cut", c["t"]), flush=True)
            active[i] = start()
        msgs, want = nxt
        t0 = time.monotonic()
        out = post("/v1/chat/completions", {"messages": msgs, "max_tokens": want, "ignore_eos": True, "skip_special_tokens": False, "temperature": 0})
        key = "%d %d" % (c["id"], c["t"])
        reply = out["choices"][0]["message"]["content"]
        replies[key] = reply
        c["msgs"] = msgs + [{"role": "assistant", "content": said.get(key, reply)}]
        u = out["usage"]
        print("req %d conv %d turn %d ago %s prompt %d cached %s completion %d %.2fs" % (
            r, c["id"], c["t"], "-" if c["last"] is None else r - c["last"], u["prompt_tokens"],
            (u.get("prompt_tokens_details") or {}).get("cached_tokens"), u["completion_tokens"], time.monotonic() - t0), flush=True)
        c["t"] += 1
        c["last"] = r
    if a.save:
        json.dump(replies, open(a.save, "w"))


if __name__ == "__main__":
    main()
