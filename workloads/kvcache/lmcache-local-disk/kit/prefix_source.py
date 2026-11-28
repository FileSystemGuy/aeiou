"""Where the prefix a prompt shares with an earlier prompt of its session comes from, over the
AgentX corpus (`agentx.py`'s docstring has the file's format; DESIGN_REVIEW.md §3.100, the note
"The corpus measured", 2026-11-27).

    python prefix_source.py traces.jsonl [SESSIONS]

Each request's blocks are walked down a prefix tree of the earlier requests of its session and
model, and what it shares is credited in layers, each class only what it adds past those before:
its own agent; an ancestor (a request of an enclosing agent sent before this branch was spawned);
a branch that had finished (before this branch began, or, for the main agent, before the request);
a branch running beside it, an enclosing agent's later requests and a nested group among them.
Printed: the counters, by main agent and sub-agents, in blocks and in requests, and the size of
what a concurrent branch adds. About two minutes for the corpus's 393 sessions."""
import collections
import json
import statistics
import sys

Row = collections.namedtuple("Row", "group t blocks first own ancestor finished concurrent")


def flatten(requests, out, own, counter, parent):
    """Every model request with its group (−1 the main agent), and each group's parent."""
    for r in requests:
        if r.get("type") == "subagent":
            counter[0] += 1
            g = counter[0]
            parent[g] = own
            flatten(r["requests"], out, g, counter, parent)
        else:
            out.append((own, r))


def session(requests, T):
    """One session's rows, in time order."""
    flat, par = [], {}
    flatten(requests, flat, -1, [-1], par)
    flat.sort(key=lambda gr: gr[1]["t"])
    start, end = {}, {}
    for g, r in flat:
        e = r["t"] + (r.get("api_time") or 0)
        while g != -1:                               # a group's interval covers its nested groups
            start[g] = min(start.get(g, r["t"]), r["t"])
            end[g] = max(end.get(g, e), e)
            g = par[g]
    T["sessions"] += 1
    T["groups"] += len(start)
    T["sessions_with_subagents"] += bool(start)
    mains = [r["t"] for g, r in flat if g == -1]
    for g in start:
        if par[g] == -1:
            T["top_groups"] += 1
            T["top_groups_main_sent_during"] += any(start[g] < t < end[g] for t in mains)
    kids = {}                                        # (model, node, hash id) -> node; node 0 the root
    who = [{}]                                       # node -> {group: the earliest t of its requests through it}
    firsts, rows = set(), []
    for g, r in flat:
        m, hs, t = r["model"], r["hash_ids"], r["t"]
        first = g not in firsts
        firsts.add(g)
        anc, h = {}, g                               # an enclosing agent -> when this branch left it
        while h != -1:
            anc[par[h]] = start[h]
            h = par[h]
        node, path = 0, []
        for x in hs:
            node = kids.get((m, node, x))
            if node is None:
                break
            path.append(node)
        depth, seen = {}, set()
        for d in range(len(path), 0, -1):            # deepest first; nothing shallower adds once its own is found
            for gc, tc in who[path[d - 1]].items():
                if gc in seen:
                    continue
                seen.add(gc)
                if gc == g:
                    c = "own"
                elif gc in anc:
                    c = "ancestor" if tc < anc[gc] else "concurrent"
                else:
                    c = "finished" if end[gc] <= (start[g] if g != -1 else t) else "concurrent"
                depth.setdefault(c, d)
            if "own" in depth:
                break
        m1 = depth.get("own", 0)
        m2 = max(m1, depth.get("ancestor", 0))
        m3 = max(m2, depth.get("finished", 0))
        m4 = max(m3, depth.get("concurrent", 0))
        rows.append(Row(g, t, len(hs), first, m1, m2 - m1, m3 - m2, m4 - m3))
        node = 0
        for x in hs:
            nx = kids.get((m, node, x))
            if nx is None:
                nx = len(who)
                who.append({})
                kids[(m, node, x)] = nx
            node = nx
            who[node].setdefault(g, t)
    return rows


def measure(path, limit=10 ** 9):
    """Every request's row and the counters, over the first `limit` sessions of the file."""
    T, rows = collections.Counter(), []
    with open(path) as f:
        for n, line in enumerate(f):
            if n >= limit:
                break
            rows += session(json.loads(line)["requests"], T)
    for r in rows:
        k = "main" if r.group == -1 else "sub"
        T[k, "requests"] += 1
        T[k, "blocks"] += r.blocks
        for c in ("own", "ancestor", "finished", "concurrent"):
            T[k, c] += getattr(r, c)
            T[k, "req_" + c] += getattr(r, c) > 0
        T[k, "req_concurrent_only"] += r.concurrent > 0 and r.own + r.ancestor + r.finished == 0
        T[k, "req_nothing_shared"] += r.own + r.ancestor + r.finished + r.concurrent == 0
    return rows, T


def report(rows, T):
    lines = ["%s %s" % (k, T[k]) for k in sorted(T, key=str)]
    for k in ("main", "sub"):
        for first in (True, False):
            xs = [r for r in rows if r.concurrent and (r.group == -1) == (k == "main") and r.first == first]
            if xs:
                ex = sorted(r.concurrent for r in xs)
                q = lambda p: ex[min(len(ex) - 1, int(p * len(ex)))]      # noqa: E731
                lines.append("%s %s n %d blocks added %d p10/50/90/max %d %d %d %d median shared before %s median prompt %s" % (
                    k, "first-of-group" if first else "later", len(xs), sum(ex), q(.1), q(.5), q(.9), ex[-1],
                    statistics.median(r.own + r.ancestor + r.finished for r in xs), statistics.median(r.blocks for r in xs)))
    return lines


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: prefix_source.py traces.jsonl [SESSIONS]", file=sys.stderr)
        sys.exit(2)
    print("\n".join(report(*measure(sys.argv[1], *map(int, sys.argv[2:])))))


if __name__ == "__main__":
    main()
