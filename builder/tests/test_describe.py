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
