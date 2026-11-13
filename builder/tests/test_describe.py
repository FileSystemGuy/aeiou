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
        {"name": "disk", "evict": "lru", "capacity": 64, "unit": "bytes"}, {"name": "all", "evict": "none"}]}],
        "body": [{"lookup": {"cache": "lm", "tier": "disk", "seq": s, "per": 4, "bind": "h", "size": 2}}]}}
    lines = d.body([tl], 0)
    assert "tier `disk`, evicting the entry used longest ago past 64 bytes; tier `all`, keeping every entry." in lines[0], lines
    assert "Looks up `cat(prefix(shared(1, 4), 2), fresh(4))` in tier `disk` of cache `lm`" in lines[1], lines
    assert d.sym({"chunk_key": {"seq": {"ref": "s"}, "chunk": 0, "per": 4}}) == "chunk_key(s, 0, 4)"
    assert d.sym({"seq_len": {"ref": "s"}}) == "seq_len(s)"
    assert d.gist([tl]) == ""                       # a lookup alone does no I/O
    lru = {"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": [{"name": "d", "evict": "lru", "capacity": 3}]}], "body": []}}
    assert "tier `d`, evicting the entry used longest ago past 3 entries." in d.body([lru], 0)[0]
