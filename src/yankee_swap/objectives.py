"""Gain functions for the justice criteria supported by General Yankee Swap.

A *gain function* ``phi(utilities, i)`` receives the current utility vector
(a tuple of ints; ``utilities[k] = v_k(X_k) = |X_k|``) and an agent index and
returns a comparable value (a number or a tuple, compared lexicographically).
In every iteration General Yankee Swap lets the active agent with the largest
gain move; ties are broken in favour of the lowest index.

For General Yankee Swap to maximize a justice criterion ``Psi`` the gain
function must satisfy *Consistency (G1)* and *Generalized Concavity (G2)*
(Section 3.1 of the paper) and ``Psi`` must respect Pareto dominance (C1).
The gain functions below are taken from Table 1 / Section 6 of the paper:

=====================  ========================================================
objective              gain ``phi(X, i)``
=====================  ========================================================
leximin (unweighted)   ``-v_i``, ties by index  (Theorem 6.2)
weighted leximin       ``(-v_i / w_i, -w_i)``  (Theorem 6.3)
maximin share          ``(-v_i / MMS_i, -MMS_i)`` if ``MMS_i > 0`` else ``-inf``
                       (Equation (4), Theorem 6.4, Corollary 6.5)
weighted Nash welfare  ``(1 + 1/v_i) ** w_i`` if ``v_i > 0`` else ``+inf``
                       (Equation (5), Theorem 6.6)
=====================  ========================================================

"Infinite" gains are encoded with a leading tuple component so all
comparisons stay exact.
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GainFunction = Callable[[Sequence[int], int], Any]

__all__ = [
    "GainFunction",
    "leximin_gain",
    "weighted_leximin_gain",
    "maximin_share_gain",
    "weighted_nash_gain",
    "resolve_objective",
    "OBJECTIVE_ALIASES",
]


def _check_weights(weights: Optional[Sequence[float]], n: int) -> List[float]:
    if weights is None:
        return [1] * n
    w = list(weights)
    if len(w) != n:
        raise ValueError("expected %d weights, got %d" % (n, len(w)))
    for x in w:
        if not (x > 0) or (isinstance(x, float) and not math.isfinite(x)):
            raise ValueError("weights must be positive and finite, got %r" % (x,))
    return w


def _to_fraction(x: Any) -> Fraction:
    # Fraction(float) is exact, so comparisons below never suffer from rounding.
    return x if isinstance(x, Fraction) else Fraction(x)


def leximin_gain() -> GainFunction:
    """Unweighted leximin: ``phi(X, i) = -v_i(X_i)`` (Theorem 6.2).

    The agent with the lowest utility goes first; ties are broken by the
    lowest index. This computes a leximin (equivalently, prioritized Lorenz
    dominating, with priority given by the agent index) allocation.
    """

    def gain(utilities: Sequence[int], i: int) -> int:
        return -utilities[i]

    gain.__name__ = "leximin"
    return gain


def weighted_leximin_gain(weights: Sequence[float]) -> GainFunction:
    """``phi(X, i) = (-v_i(X_i) / w_i, -w_i)`` (Theorem 6.3).

    The agent with the lowest weighted utility goes first; ties are broken in
    favour of the *lower* weight (then the lower index).
    """
    ws = [_to_fraction(w) for w in weights]

    def gain(utilities: Sequence[int], i: int) -> Tuple[Fraction, Fraction]:
        return (-utilities[i] / ws[i], -ws[i])

    gain.__name__ = "weighted_leximin"
    return gain


def maximin_share_gain(shares: Sequence[float]) -> GainFunction:
    """Equation (4): maximize the sorted vector of fair-share fractions ``v_i / c_i``.

    ``phi(X, i) = (-v_i / c_i, -c_i)`` if ``c_i > 0`` and ``-infinity`` otherwise.
    With ``c_i = MMS_i`` this computes a maximin share allocation (Cor. 6.5).
    """
    cs = [_to_fraction(c) for c in shares]
    for c in cs:
        if c < 0:
            raise ValueError("maximin shares must be non-negative")

    def gain(utilities: Sequence[int], i: int) -> Tuple[int, Fraction, Fraction]:
        c = cs[i]
        if c > 0:
            return (1, -utilities[i] / c, -c)
        return (0, Fraction(0), Fraction(0))  # "-M": lowest possible priority

    gain.__name__ = "maximin_share"
    return gain


def weighted_nash_gain(weights: Sequence[float]) -> GainFunction:
    """Equation (5): ``phi(X, i) = (1 + 1/v_i) ** w_i`` if ``v_i > 0`` else a large ``M``.

    We compare ``w_i * log(1 + 1/v_i)`` (a monotone transform), which avoids
    overflow for large weights.
    """
    ws = [float(w) for w in weights]

    def gain(utilities: Sequence[int], i: int) -> Tuple[int, float]:
        v = utilities[i]
        if v == 0:
            return (1, 0.0)
        return (0, ws[i] * math.log1p(1.0 / v))

    gain.__name__ = "weighted_nash_welfare"
    return gain


OBJECTIVE_ALIASES: Dict[str, str] = {
    # weighted Nash welfare
    "weighted_nash_welfare": "weighted_nash_welfare",
    "weighted_nash": "weighted_nash_welfare",
    "nash_welfare": "weighted_nash_welfare",
    "nash": "weighted_nash_welfare",
    "mwnw": "weighted_nash_welfare",
    "max_weighted_nash_welfare": "weighted_nash_welfare",
    # weighted leximin
    "weighted_leximin": "weighted_leximin",
    "wleximin": "weighted_leximin",
    # unweighted leximin
    "leximin": "leximin",
    "unweighted_leximin": "leximin",
    # maximin share
    "maximin_share": "maximin_share",
    "maximin_shares": "maximin_share",
    "mms": "maximin_share",
}


def _canonical(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key not in OBJECTIVE_ALIASES:
        raise ValueError(
            "unknown objective %r; choose one of 'weighted_nash_welfare', "
            "'weighted_leximin', 'maximin_share', 'leximin'" % (name,)
        )
    return OBJECTIVE_ALIASES[key]


def resolve_objective(
    objective: str,
    n_agents: int,
    n_items: int,
    valuations: Sequence[Callable],
    *,
    weights: Optional[Sequence[float]] = None,
    mms_values: Optional[Sequence[float]] = None,
) -> Tuple[str, GainFunction, Dict[str, Any]]:
    """Map an objective name to ``(canonical_name, gain_function, extra_info)``."""
    name = _canonical(objective)
    extra: Dict[str, Any] = {}
    if name == "weighted_leximin":
        return name, weighted_leximin_gain(_check_weights(weights, n_agents)), extra
    if name == "leximin":
        if weights is not None:
            raise ValueError("'leximin' is unweighted; use 'weighted_leximin' to pass weights")
        return name, leximin_gain(), extra
    if name == "weighted_nash_welfare":
        return name, weighted_nash_gain(_check_weights(weights, n_agents)), extra
    if name == "maximin_share":
        if weights is not None:
            raise ValueError("the maximin share objective does not take weights")
        if mms_values is None:
            from .mms import maximin_shares  # local import avoids a cycle

            t0 = time.perf_counter()
            mms_values = maximin_shares(n_agents, n_items, valuations)
            extra["mms_seconds"] = time.perf_counter() - t0
        elif len(mms_values) != n_agents:
            raise ValueError("expected %d MMS values, got %d" % (n_agents, len(mms_values)))
        extra["mms_values"] = list(mms_values)
        return name, maximin_share_gain(mms_values), extra
    raise AssertionError("unreachable")  # pragma: no cover
