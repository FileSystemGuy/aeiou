"""The describer's distribution prose (builder/aeiou/describe.py)."""

from aeiou import describe


def test_a_clamp_past_both_percentiles_moves_both():
    # The runner clamps every draw to [min, max] (eval.rs `clamp`), so a bound beyond both
    # percentiles moves both: the range is never printed inverted.
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    assert d.dist({"normal": {"mean": 1000, "sd": 10, "max": 500}}, "bytes") == \
        "normal, mean 1,000 bytes, standard deviation 10 bytes (5th and 95th percentiles both 500 bytes), clamped to at most 500 bytes"
    assert d.dist({"lognormal": {"median": 100, "sigma": 1, "min": 1000}}, "bytes") == \
        "lognormal, median 100 bytes, σ 1 (5th and 95th percentiles both 1,000 bytes), clamped to at least 1,000 bytes"
    # A bound between the percentiles moves only the one beyond it.
    assert d.dist({"normal": {"mean": 1000, "sd": 10, "min": 995}}, "bytes") == \
        "normal, mean 1,000 bytes, standard deviation 10 bytes (5th–95th percentile 995 bytes to 1,016 bytes), clamped to at least 995 bytes"
    # min above max: the runner raises to min, then lowers to max.
    assert d.dist({"normal": {"mean": 1000, "sd": 10, "min": 2000, "max": 1500}}, "bytes").startswith(
        "normal, mean 1,000 bytes, standard deviation 10 bytes (5th and 95th percentiles both 1.5 KiB)")


def test_the_seq_kinds_a_cache_and_a_lookup_are_described():
    """Contract 0.16: a timeline's caches, a `lookup`, and the `seq` expressions in words."""
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    s = {"cat": [{"prefix": [{"shared": {"id": 1, "n": 4}}, 2]}, {"fresh": 4}]}
    tl = {"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": [
        {"name": "disk", "order": [{"cached": "touched"}], "capacity": 64, "unit": "bytes"}, {"name": "all"}]}],
        "body": [{"store": {"cache": "lm", "tier": "disk", "keys": {"index": "j", "count": 2, "key": {"chunk_key": {"seq": s, "chunk": {"index": "j"}, "per": 4}}},
                            "present": "touch", "found": "h", "size": 2}}]}}
    lines = d.body([tl], 0)
    assert "tier `disk`, evicting the entry used longest ago past 64 bytes; tier `all`, keeping every entry." in lines[0], lines
    assert lines[1] == ("  - Stores, of the 2 keys `chunk_key(cat(prefix(shared(1, 4), 2), fresh(4)), j, 4)`, `j` from `0`, in tier `disk` of "
                        "cache `lm`, those it lacks, those held touched first, as one event; `h` is the leading ones held before it."), lines
    assert d.sym({"chunk_key": {"seq": {"ref": "s"}, "chunk": 0, "per": 4}}) == "chunk_key(s, 0, 4)"
    assert d.sym({"seq_len": {"ref": "s"}}) == "seq_len(s)"
    assert d.gist([tl]) == ""                       # a lookup alone does no I/O
    lru = {"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": [{"name": "d", "order": [{"cached": "touched"}], "capacity": 3}]}], "body": []}}
    assert "tier `d`, evicting the entry used longest ago past 3 entries." in d.body([lru], 0)[0]


def test_orders_holds_resources_and_now_are_described():
    """Contract 0.18: a tier's order and `full`, a held lookup, `acquire`, `release`, `now`, `cached`."""
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    tiers = [{"name": "hbm", "order": [{"cached": "freed"}, {"neg": {"cached": "pos"}}], "capacity": 8, "full": "wait"},
             {"name": "pin", "capacity": 4, "full": "fail"}]
    tl = {"timeline": {"name": "tl", "caches": [{"name": "g", "tiers": tiers}], "resources": [{"name": "slots", "capacity": 2}],
          "body": [{"acquire": {"resource": "slots", "n": 1, "hold": "s"}},
                   {"store": {"cache": "g", "tier": "hbm", "keys": {"index": "j", "count": 4, "key": {"index": "j"}}, "present": "touch", "found": "h", "hold": "kv"}},
                   {"release": {"hold": "kv"}}, {"release": {"hold": "s"}}]}}
    lines = d.body([tl], 0)
    assert ("tier `hbm`, evicting past 8 entries the entry least by (cached.freed, −cached.pos); a store that finds no room "
            "waits for a release; tier `pin`, keeping every entry up to 4 entries; a store that finds no room fails.") in lines[0], lines
    assert "Its resource `slots` of 2 units." in lines[0]
    assert lines[1] == "  - Takes 1 units of `slots` as `s`, waiting in planned time while they are not free, first come first."
    assert "The lane holds them under `kv` until it releases it" in lines[2]
    assert lines[3:] == ["  - Releases `kv`.", "  - Releases `s`."]
    assert d.sym({"now": {}}) == "now"


def test_entries_in_groups_are_described():
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    assert d.sym({"published": {"timeline": "tl", "group": {"index": "s"}}}) == "published(tl, group s)"
    assert d.sym({"entry": {"timeline": "tl", "group": 2, "index": 0, "key": "v"}}) == "tl[0 of group 2].v"
    assert d.body([{"publish": {"timeline": "tl", "group": 2, "values": {"v": 1}}}], 0) == ["- Publishes to `tl`, group `2`: v = `1`."]


def test_timed_eviction_eviction_bodies_and_ranges_are_described():
    """Contract 0.19: a tier's `evict` and `on_evict`, `probe`, `touch`, `store`, `tier`, `victim`."""
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    g3 = {"name": "g3", "order": [{"cached": "touched"}, {"neg": {"cached": "pos"}}], "capacity": 10,
          "evict": {"every": 1000, "when": {"ge": [{"tier": "used"}, 8]}, "count": 2},
          "on_evict": {"lanes": 4, "body": [{"unlink": {"file": {"file": {"dataset": "d", "id": {"victim": {}}}}}}]}}
    s = {"fresh": 4}
    tl = {"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": [g3]}], "body": [
        {"probe": {"cache": "lm", "tier": "g3", "keys": {"index": "j", "from": 0, "count": 4, "key": {"chunk_key": {"seq": s, "chunk": {"index": "j"}, "per": 1}}}, "bind": "h"}},
        {"touch": {"cache": "lm", "tier": "g3", "keys": {"index": "j", "from": 0, "count": {"ref": "h"}, "key": {"chunk_key": {"seq": s, "chunk": {"index": "j"}, "per": 1}}}}},
        {"store": {"cache": "lm", "tier": "g3", "keys": {"index": "j", "from": {"ref": "h"}, "count": 4, "key": {"chunk_key": {"seq": s, "chunk": {"index": "j"}, "per": 1}}}, "bind": "n"}},
        {"store": {"cache": "lm", "tier": "g3", "keys": {"index": "j", "count": 1, "key": 7}}}]}}
    lines = d.body([tl], 0)
    assert "`tier.used >= 8`" in lines[0] or "tier.used" in lines[0], lines
    assert "a store past it is dropped" in lines[0], lines
    assert "Each victim of tier `g3`, on 4 eviction lanes" in lines[1], lines
    assert any("Probes the 4 keys `chunk_key(fresh(4), j, 1)`, `j` from `0`, in tier `g3`" in l and "nothing changed" in l for l in lines), lines
    assert any(l.strip().startswith("- Touches those of the `h` keys `chunk_key(fresh(4), j, 1)`, `j` from `0`, that are held") for l in lines), lines
    assert any("`j` from `h`," in l and "`n` is the leading ones held after it" in l for l in lines), lines
    assert any("Stores, of the 1 keys `7`, `j` from `0`, in tier `g3` of cache `lm`, those it lacks, as one event." in l for l in lines), lines
    assert d.sym({"victim": {}}) == "victim" and d.sym({"tier": "entries"}) == "tier.entries"
    tl = {"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": [g3]}], "body": [
        {"store": {"cache": "lm", "tier": "g3", "keys": {"index": "j", "from": 0, "count": 4, "key": {"chunk_key": {"seq": s, "chunk": {"index": "j"}, "per": 1}}}, "taken": "t2"}},
        {"store": {"cache": "lm", "tier": "g3", "keys": {"index": "j", "count": {"len": "t2"}, "key": {"elem": {"array": "t2", "index": {"index": "j"}}}},
                   "present": "renew", "taken": "t3"}}]}}
    lines = d.body([tl], 0)
    assert any("`t2` is the indices it stored" in l for l in lines), lines
    assert any("those held stored again" in l and "`t3` is the indices it stored" in l for l in lines), lines


def test_workers_submit_and_span_are_described():
    """Contract 0.20: a timeline's `workers`, a `submit`'s body on a worker lane, a `span`."""
    d = describe.Describer({"name": "t", "params": {}, "actors": {}}, {}, 1)
    st = {"stat": {"file": {"file": {"dataset": "d", "id": 1}}}}
    tl = {"timeline": {"name": "tl", "workers": [{"name": "store", "lanes": 4, "staging": 2147483648}], "body": [
        {"span": {"name": "load", "body": [st]}}, {"submit": {"workers": "store", "body": [st]}}]}}
    lines = d.body([tl], 0)
    assert "Its workers `store`: 4 worker lanes, staging" in lines[0], lines
    assert lines[1].strip() == "- Measured as span `load`, a request's own storage wait:", lines
    assert lines[3].strip() == "- On a worker lane of `store`, at this lane's planned time:", lines
    assert d.gist([{"submit": {"workers": "store", "body": [st]}}]).startswith("on a worker lane, "), d.gist([{"submit": {"workers": "store", "body": [st]}}])
    # contract 0.21: a span naming the staging its loads draw on
    lines = d.body([{"timeline": {"name": "tl", "workers": [{"name": "store", "lanes": 4, "staging": 2147483648}], "body": [
        {"span": {"name": "load", "staging": "store", "body": [st]}}]}}], 0)
    assert lines[1].strip() == "- Measured as span `load`, a request's own storage wait, its reads drawing on the staging of `store`:", lines
