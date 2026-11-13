"""Distribution constructors. Each returns a `Dist` node; nothing is drawn here."""
from __future__ import annotations

import math

from .nodes import BuildError, Dist, Draw, Cond, Expr, SeqOp, ast_of, lift, distref

none = None   # the `none` outcome of a mixture arm


def _literal(x):
    """The value of a literal number, or None for a node (a parameter or formula, which the
    runner checks when it resolves the draw)."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _weight(w, what: str) -> float:
    """A literal weight: a finite, non-negative number."""
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or w < 0:
        raise BuildError(f"{what} must be a finite non-negative number, got {w!r}")
    return w


def _bounds(kind: str, a: dict) -> None:
    """Literal `min` above literal `max` clamps every draw to a contradiction."""
    lo, hi = _literal(a.get("min")), _literal(a.get("max"))
    if lo is not None and hi is not None and lo > hi:
        raise BuildError(f"{kind}: min {lo} is above max {hi}")


def const(v) -> Dist:
    return Dist("const", lift(v, "const"))


def uniform(lo, hi) -> Dist:
    """Integer in [lo, hi)."""
    lo, hi = lift(lo, "uniform lo"), lift(hi, "uniform hi")
    if _literal(lo) is not None and _literal(hi) is not None and lo >= hi:
        raise BuildError(f"uniform: empty range [{lo}, {hi}): hi must be above lo")
    return Dist("uniform", {"lo": lo, "hi": hi})


def uniform64() -> Dist:
    """A fresh 64-bit identifier."""
    return Dist("uniform64", {})


def normal(mean, sd, *, min=None, max=None) -> Dist:
    a = {"mean": lift(mean, "normal mean"), "sd": lift(sd, "normal sd")}
    if min is not None:
        a["min"] = lift(min, "normal min")
    if max is not None:
        a["max"] = lift(max, "normal max")
    if _literal(a["sd"]) is not None and a["sd"] < 0:
        raise BuildError(f"normal: sd {a['sd']} is negative")
    _bounds("normal", a)
    return Dist("normal", a)


def lognormal(median, sigma, *, min=None, max=None) -> Dist:
    if _literal(sigma) is None or not math.isfinite(sigma) or sigma <= 0:
        raise BuildError(f"lognormal sigma must be a finite positive number, got {sigma!r}")
    a = {"median": lift(median, "lognormal median"), "sigma": float(sigma)}
    if _literal(a["median"]) is not None and a["median"] <= 0:
        raise BuildError(f"lognormal: median {a['median']} is not positive")
    if min is not None:
        a["min"] = lift(min, "lognormal min")
    if max is not None:
        a["max"] = lift(max, "lognormal max")
    _bounds("lognormal", a)
    return Dist("lognormal", a)


def empirical(values, weights=None) -> Dist:
    """Values with relative weights (equal if omitted). `values` may also be a dict
    value -> weight."""
    if isinstance(values, dict):
        values, weights = list(values), list(values.values())
    values = [lift(v, "empirical value") for v in values]
    if any(isinstance(v, Dist) for v in values) or not values:
        raise BuildError("empirical values must be non-empty scalars")
    weights = [1] * len(values) if weights is None else [lift(w, "empirical weight") for w in weights]
    if len(weights) != len(values):
        raise BuildError("empirical: values and weights differ in length")
    if sum(_weight(w, "empirical weight") for w in weights) <= 0:
        raise BuildError("empirical: total weight is zero")
    return Dist("empirical", {"values": values, "weights": weights})


def zipf(s) -> Dist:
    if _literal(s) is None or not math.isfinite(s) or s <= 0:
        raise BuildError(f"zipf s must be a finite positive number, got {s!r}")
    return Dist("zipf", {"s": float(s)})


def hotset(fraction, weight) -> Dist:
    if not (0 < fraction <= 1) or not (0 <= weight <= 1):
        raise BuildError("hotset needs 0 < fraction <= 1 and 0 <= weight <= 1")
    return Dist("hotset", {"fraction": float(fraction), "weight": float(weight)})


def mixture(*arms) -> Dist:
    """mixture((0.55, none), (0.45, lognormal(...)))  or  mixture({0.55: none, 0.45: d})."""
    if len(arms) == 1 and isinstance(arms[0], dict):
        arms = tuple(arms[0].items())
    out = []
    for arm in arms:
        if not (isinstance(arm, (tuple, list)) and len(arm) == 2):
            raise BuildError("each mixture arm is (weight, dist-or-none)")
        w, d = arm
        _weight(w, "mixture weight")
        out.append({"weight": w, "dist": None if d is None else distref(d)})
    if not out:
        raise BuildError("mixture needs at least one arm")
    if sum(a["weight"] for a in out) <= 0:
        raise BuildError("mixture: total weight is zero")
    return Dist("mixture", out)


def draw(dist) -> Draw:
    """A positional draw from a distribution, a parameter holding one, or a `when` that
    selects one."""
    return Draw(dist)


def when(test, then, otherwise) -> Cond:
    """Expression conditional. The arms may be expressions, distributions, or handles."""
    return Cond(test, then, otherwise)


def fresh(n) -> Expr:
    """A `seq` of `n` new blocks, new at every position and the same when `@` recomputes them
    (contract 0.16)."""
    return SeqOp("fresh", lift(n, "fresh"))


def shared(id, n) -> Expr:
    """A `seq` of the first `n` blocks of shared prefix `id`: the same in every lane, actor, and
    run, as a system prompt is every session's (contract 0.16)."""
    return SeqOp("shared", {"id": lift(id, "shared id"), "n": lift(n, "shared n")})


def prefix(seq, k) -> Expr:
    """The first `k` blocks of a `seq` (contract 0.16)."""
    return SeqOp("prefix", (lift(seq, "prefix"), lift(k, "prefix k")))


def cat(a, b) -> Expr:
    """One `seq` after another (contract 0.16)."""
    return SeqOp("cat", (lift(a, "cat"), lift(b, "cat")))


def seq_len(seq) -> Expr:
    """The blocks of a `seq` (contract 0.16)."""
    return SeqOp("seq_len", lift(seq, "seq_len"))


def chunk_key(seq, chunk, per) -> Expr:
    """The key of whole chunk `chunk` of `per` blocks of a `seq`: the chained hash of every
    block to its end, equal for two sequences exactly when they hold the same blocks up to it
    (contract 0.16)."""
    return SeqOp("chunk_key", {"seq": lift(seq, "chunk_key"), "chunk": lift(chunk, "chunk_key chunk"), "per": lift(per, "chunk_key per")})


CACHED = ("stored", "touched", "freed", "hits", "pos", "size")


def cached(attr: str) -> Expr:
    """An attribute of the entry a tier's `order` is computed for (contract 0.18, rule V33):
    `stored`, `touched`, `freed` (the tier's event counts when it was stored, last looked up,
    and last released), `hits`, `pos` (its index in the sequence of the lookup that last found
    or stored it), or `size`."""
    from .nodes import Plain
    if attr not in CACHED:
        raise BuildError(f"cached: one of {', '.join(CACHED)}, got {attr!r}")
    return Plain({"cached": attr})


def now() -> Expr:
    """This lane's planned time in nanoseconds (contract 0.18, rule V30): only in a timeline,
    and not positional, so `@` and `prev` may not reach it."""
    from .nodes import Plain
    return Plain({"now": {}})


TIER_ATTRS = ("used", "entries")


def tier_attr(attr: str) -> Expr:
    """What a tier holds, read by its timed eviction's `when` and `count` (contract 0.19, rule
    V33): `used` (its entries' size in the tier's unit) or `entries` (how many it holds)."""
    from .nodes import Plain
    if attr not in TIER_ATTRS:
        raise BuildError(f"tier_attr: one of {', '.join(TIER_ATTRS)}, got {attr!r}")
    return Plain({"tier": attr})


def victim() -> Expr:
    """The key of the entry an `on_evict` body runs for (contract 0.19, rule V35), as
    `chunk_key` gives it."""
    from .nodes import Plain
    return Plain({"victim": {}})


def timed(every, when, count) -> dict:
    """A tier's timed eviction (contract 0.19, rule V33), for `tier(evict=...)`: at every
    planned instant k × `every` ns, if `when` holds, the `count` entries least in the tier's
    order that no lane holds are evicted, never at store. `when` and `count` read parameters,
    literals, and `tier_attr`; `every` parameters and literals."""
    return {"every": ast_of(lift(every, "evict every")), "when": ast_of(lift(when, "evict when")),
            "count": ast_of(lift(count, "evict count"))}


def lru() -> list:
    """The order that evicts the entry looked up longest ago first: `[cached("touched")]`."""
    return [cached("touched")]


def tier(name: str, order=None, capacity=None, unit: str = "entries", full: str = "skip", evict: dict | None = None,
         on_evict=None) -> dict:
    """A tier of a timeline's cache (contracts 0.16 and 0.18, rule V33): at most `capacity`
    (in entries or bytes) of entries no lane holds evicted least `order` first (a list of
    integer expressions over `cached` and parameters, compared in turn; `lru()` for the entry
    looked up longest ago), never one a lane holds. With no `order`, nothing is evicted. `full`,
    what a store finding no room and no victim does: `skip` it, `wait` (the lane waits for a
    release, first come first), or `fail`. `evict` (contract 0.19), `timed(...)`: victims taken
    at planned ticks instead of at store. `on_evict` (contract 0.19, rule V35), `(lanes, fn)`:
    each victim runs the statements `fn(b)` writes on the builder `b` it is given, with
    `victim()` its key, victim i of a batch on eviction lane i mod `lanes`."""
    if unit not in ("entries", "bytes"):
        raise BuildError(f"tier {name!r}: unit is `entries` or `bytes`, got {unit!r}")
    if full not in ("skip", "wait", "fail"):
        raise BuildError(f"tier {name!r}: full is `skip`, `wait`, or `fail`, got {full!r}")
    if capacity is None and (order is not None or full != "skip"):
        raise BuildError(f"tier {name!r}: an `order` or a `full` other than `skip` needs a capacity (V33)")
    if order is not None and not order:
        raise BuildError(f"tier {name!r}: an empty `order` (V33)")
    t = {"name": name}
    if order is not None:
        t["order"] = [ast_of(lift(e, f"tier {name} order")) for e in order]
    if capacity is not None:
        t["capacity"] = ast_of(lift(capacity, f"tier {name} capacity"))
    if unit != "entries":
        t["unit"] = unit
    if full != "skip":
        t["full"] = full
    if evict is not None:
        if capacity is None or order is None:
            raise BuildError(f"tier {name!r}: `evict` needs a capacity and an order (V33)")
        t["evict"] = dict(evict)
    if on_evict is not None:
        lanes, fn = on_evict
        if order is None:
            raise BuildError(f"tier {name!r}: `on_evict` with no `order`, so nothing is evicted (V35)")
        if not callable(fn):
            raise BuildError(f"tier {name!r}: `on_evict` is (lanes, fn), fn writing the body")
        t["on_evict"] = {"lanes": ast_of(lift(lanes, f"tier {name} on_evict lanes")), "body": fn}
    return t


def ceil_div(a, b):
    from .nodes import Op
    return Op("ceil_div", (lift(a, "ceil_div"), lift(b, "ceil_div")))


def min_(a, b):
    from .nodes import Op
    return Op("min", (lift(a, "min"), lift(b, "min")))


def max_(a, b):
    from .nodes import Op
    return Op("max", (lift(a, "max"), lift(b, "max")))
