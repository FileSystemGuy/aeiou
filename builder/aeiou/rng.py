"""The runner's positional definitions, ported exactly (`runner/REFERENCE.md` §2,
`runner/aeiou/src/rng.rs`, `eval.rs`): keys, the SplitMix64 word sequence, the dataset size
draw, and the payload block seed. The Python side needs them to write a container dataset
(`aeiou-datagen`) and to verify one; everything here is a pure function of its arguments.

Floating-point draws (`normal`, `lognormal`) use the same IEEE double operations in the same
order as the Rust code and the platform libm for `exp`, `log`, `cos`, `sqrt`; on one host
class the results are bit-identical, across hosts a one-ulp libm difference could in principle
move a value across a rounding boundary. Constant sizes have no such exposure.
"""
from __future__ import annotations

import math
import struct

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def _xxh3(data: bytes, seed: int) -> int:
    import xxhash
    return xxhash.xxh3_64_intdigest(data, seed=seed)


def labeled_key(seed: int, label: str, parts) -> int:
    """`xxh3_64(label ‖ 0 ‖ parts as LE u64, seed = seed)`."""
    return _xxh3(label.encode() + b"\0" + b"".join(struct.pack("<Q", p & MASK) for p in parts), seed)


def dataset_key(dataset_seed: int, id: int) -> int:
    """The key of a per-id dataset draw: `xxh3_64(id as LE i64, seed = dataset seed)`."""
    return _xxh3(struct.pack("<q", id), dataset_seed)


def block_seed(seed: int, unit: int, block: int) -> int:
    """The seed of 1 MiB block `block` of payload unit `unit` (`aeiou-positional/1`)."""
    return labeled_key(seed, "payload", [unit, block])


def object_seed(namespace_seed: int, path: str) -> int:
    return labeled_key(namespace_seed, "object", [_xxh3(path.encode(), 0)])


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


class Words:
    """The SplitMix64 sequence started at a key."""
    __slots__ = ("s",)

    def __init__(self, key: int):
        self.s = key & MASK

    def next_u64(self) -> int:
        self.s = (self.s + GOLDEN) & MASK
        return mix64(self.s)

    def next_f64(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def next_f64_open(self) -> float:
        return ((self.next_u64() >> 11) + 1) * (1.0 / (1 << 53))

    def next_range(self, lo: int, hi: int) -> int:
        span = (hi - lo) & MASK
        return lo + (self.next_u64() % span)

    def next_normal(self) -> float:
        u1 = self.next_f64_open()
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def rust_round(x: float) -> int:
    """`f64::round`: half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _clamp(x, lo, hi):
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


def sample(dist: dict, w: Words):
    """Draw one value from a resolved distribution (arguments are numbers; `eval::sample`).
    `zipf` and `hotset` (ranks over a dataset) are not needed here."""
    kind, a = next(iter(dist.items()))
    if kind == "const":
        return a
    if kind == "uniform":
        lo, hi = int(a["lo"]), int(a["hi"])
        if hi <= lo:
            raise ValueError(f"uniform: empty range [{lo}, {hi})")
        return w.next_range(lo, hi)
    if kind == "uniform64":
        v = w.next_u64()
        return v - (1 << 64) if v >= (1 << 63) else v
    if kind == "normal":
        x = float(a["mean"]) + float(a["sd"]) * w.next_normal()
        return rust_round(_clamp(x, a.get("min"), a.get("max")))
    if kind == "lognormal":
        x = float(a["median"]) * math.exp(float(a["sigma"]) * w.next_normal())
        return rust_round(_clamp(x, a.get("min"), a.get("max")))
    if kind == "empirical":
        weights = [float(x) for x in a["weights"]]
        cum, acc = [], 0.0
        for x in weights:
            acc += x
            cum.append(acc)
        total = cum[-1] if cum else 0.0
        if total <= 0.0:
            raise ValueError("empirical: total weight is zero")
        u = w.next_f64() * total
        i = 0
        while i < len(cum) and cum[i] <= u:
            i += 1
        return a["values"][min(i, len(a["values"]) - 1)]
    if kind == "mixture":
        total = sum(float(arm["weight"]) for arm in a)
        if total <= 0.0:
            raise ValueError("mixture: total weight is zero")
        u = w.next_f64() * total
        acc, chosen = 0.0, len(a) - 1
        for i, arm in enumerate(a):
            acc += float(arm["weight"])
            if u < acc:
                chosen = i
                break
        d = a[chosen]["dist"]
        return None if d is None else sample(d, w)
    raise ValueError(f"cannot draw a dataset size from `{kind}`")


def sample_size(dist: dict, dataset_seed: int, id: int) -> int:
    """Bytes of sample `id`: `sample(size, Words(dataset_key(seed, id)))`, rounded (V10)."""
    v = sample(dist, Words(dataset_key(dataset_seed, id)))
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"sample size drew {v!r}")
    return rust_round(float(v)) if isinstance(v, float) else int(v)
