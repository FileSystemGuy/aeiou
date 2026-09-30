"""Distribution constructors. Each returns a `Dist` node; nothing is drawn here."""
from __future__ import annotations

from .nodes import BuildError, Dist, Draw, Cond, lift, distref

none = None   # the `none` outcome of a mixture arm


def const(v) -> Dist:
    return Dist("const", lift(v, "const"))


def uniform(lo, hi) -> Dist:
    """Integer in [lo, hi)."""
    return Dist("uniform", {"lo": lift(lo, "uniform lo"), "hi": lift(hi, "uniform hi")})


def uniform64() -> Dist:
    """A fresh 64-bit identifier."""
    return Dist("uniform64", {})


def normal(mean, sd, *, min=None, max=None) -> Dist:
    a = {"mean": lift(mean, "normal mean"), "sd": lift(sd, "normal sd")}
    if min is not None:
        a["min"] = lift(min, "normal min")
    if max is not None:
        a["max"] = lift(max, "normal max")
    return Dist("normal", a)


def lognormal(median, sigma, *, min=None, max=None) -> Dist:
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise BuildError("lognormal sigma must be a positive number")
    a = {"median": lift(median, "lognormal median"), "sigma": float(sigma)}
    if min is not None:
        a["min"] = lift(min, "lognormal min")
    if max is not None:
        a["max"] = lift(max, "lognormal max")
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
    return Dist("empirical", {"values": values, "weights": weights})


def zipf(s) -> Dist:
    if not isinstance(s, (int, float)) or s <= 0:
        raise BuildError("zipf s must be a positive number")
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
        if not isinstance(w, (int, float)) or w < 0:
            raise BuildError("mixture weight must be a non-negative number")
        out.append({"weight": w, "dist": None if d is None else distref(d)})
    if not out:
        raise BuildError("mixture needs at least one arm")
    return Dist("mixture", out)


def draw(dist) -> Draw:
    """A positional draw from a distribution, a parameter holding one, or a `when` that
    selects one."""
    return Draw(dist)


def when(test, then, otherwise) -> Cond:
    """Expression conditional. The arms may be expressions, distributions, or handles."""
    return Cond(test, then, otherwise)


def ceil_div(a, b):
    from .nodes import Op
    return Op("ceil_div", (lift(a, "ceil_div"), lift(b, "ceil_div")))


def min_(a, b):
    from .nodes import Op
    return Op("min", (lift(a, "min"), lift(b, "min")))


def max_(a, b):
    from .nodes import Op
    return Op("max", (lift(a, "max"), lift(b, "max")))
