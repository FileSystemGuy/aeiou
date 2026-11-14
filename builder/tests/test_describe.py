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
        "body": [{"lookup": {"cache": "lm", "tier": "disk", "seq": s, "per": 4, "bind": "h", "size": 2}}]}}
    lines = d.body([tl], 0)
    assert "tier `disk`, evicting the entry used longest ago past 64 bytes; tier `all`, keeping every entry." in lines[0], lines
    assert "Looks up `cat(prefix(shared(1, 4), 2), fresh(4))` in tier `disk` of cache `lm`" in lines[1], lines
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
                   {"lookup": {"cache": "g", "tier": "hbm", "seq": {"fresh": 4}, "per": 1, "bind": "h", "hold": "kv"}},
                   {"release": {"hold": "kv"}}, {"release": {"hold": "s"}}]}}
    lines = d.body([tl], 0)
    assert ("tier `hbm`, evicting past 8 entries the entry least by (cached.freed, −cached.pos); a lookup that finds no room "
            "waits for a release; tier `pin`, keeping every entry up to 4 entries; a lookup that finds no room fails.") in lines[0], lines
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
        {"probe": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "from": 0, "bind": "h"}},
        {"touch": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "from": 0, "count": {"ref": "h"}}},
        {"store": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "from": {"ref": "h"}, "count": 4, "bind": "n"}},
        {"store": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "from": 0, "count": 1}}]}}
    lines = d.body([tl], 0)
    assert "`tier.used >= 8`" in lines[0] or "tier.used" in lines[0], lines
    assert "a store past it is dropped" in lines[0], lines
    assert "Each victim of tier `g3`, on 4 eviction lanes" in lines[1], lines
    assert any("Probes `fresh(4)` from chunk `0` in tier `g3`" in l and "nothing changed" in l for l in lines), lines
    assert any(l.strip().startswith("- Touches `fresh(4)`'s chunks `0` to `0` + `h`") for l in lines), lines
    assert any("`n` is the leading chunks held after it" in l for l in lines), lines
    assert d.sym({"victim": {}}) == "victim" and d.sym({"tier": "entries"}) == "tier.entries"
    tl = {"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": [g3]}], "body": [
        {"store": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "from": 0, "count": 4, "taken": "t2"}},
        {"store": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "chunks": "t2", "present": "renew", "taken": "t3"}},
        {"touch": {"cache": "lm", "tier": "g3", "seq": s, "per": 1, "chunks": "t3"}}]}}
    lines = d.body([tl], 0)
    assert any("`t2` is the indices of the chunks it stored" in l for l in lines), lines
    assert any("Stores `fresh(4)`'s chunks in `t2` in tier `g3`" in l and "those held stored again" in l for l in lines), lines
    assert any("Touches `fresh(4)`'s chunks in `t3`" in l for l in lines), lines


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
