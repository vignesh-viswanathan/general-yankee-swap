"""Exhaustive-search reference implementations (for verification on tiny instances).

These enumerate all ``(n + 1) ** m`` allocations (every item goes to one of the
``n`` agents or stays unallocated), so only use them for very small instances.
They are used by the test-suite to verify General Yankee Swap, and are exposed
so that users can sanity-check their own oracles / gain functions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "value_tables",
    "achievable_utility_vectors",
    "weighted_leximin_key",
    "maximin_share_key",
    "nash_key",
    "optimal_vectors",
    "brute_force_mms",
    "brute_force_optimum",
]

Vector = Tuple[int, ...]


def value_tables(n_items: int, valuations: Sequence[Callable]) -> List[List[int]]:
    """``tables[i][mask] = v_i(items in mask)`` for every subset ``mask``."""
    tables = []
    for v in valuations:
        tables.append(
            [v(frozenset(g for g in range(n_items) if mask >> g & 1)) for mask in range(1 << n_items)]
        )
    return tables


def achievable_utility_vectors(n_agents: int, n_items: int, valuations: Sequence[Callable]) -> Set[Vector]:
    """All utility vectors induced by some allocation (unallocated items allowed)."""
    tables = value_tables(n_items, valuations)
    vectors: Set[Vector] = set()
    for assignment in product(range(n_agents + 1), repeat=n_items):  # n_agents == unallocated
        masks = [0] * (n_agents + 1)
        for g, owner in enumerate(assignment):
            masks[owner] |= 1 << g
        vectors.add(tuple(tables[i][masks[i]] for i in range(n_agents)))
    return vectors


def brute_force_mms(n_agents: int, n_items: int, valuation: Callable) -> int:
    """``max over partitions into n_agents bundles of the minimum bundle value``."""
    table = value_tables(n_items, [valuation])[0]
    best = -1
    for assignment in product(range(n_agents), repeat=n_items):
        masks = [0] * n_agents
        for g, b in enumerate(assignment):
            masks[b] |= 1 << g
        best = max(best, min(table[mask] for mask in masks))
    if n_items == 0:
        best = table[0]
    return best


# --------------------------------------------------------------------- keys
def weighted_leximin_key(u: Vector, weights: Sequence[float]) -> Tuple[Fraction, ...]:
    """Sorted weighted utility vector (compare lexicographically, larger is better)."""
    return tuple(sorted(Fraction(x) / Fraction(w) for x, w in zip(u, weights)))


def maximin_share_key(u: Vector, shares: Sequence[float]) -> Tuple[Fraction, ...]:
    """Sorted vector of ``v_i / c_i`` over agents with ``c_i > 0`` (Theorem 6.4)."""
    return tuple(sorted(Fraction(x) / Fraction(c) for x, c in zip(u, shares) if c > 0))


def nash_key(u: Vector, weights: Sequence[float]) -> Tuple[int, float]:
    """(#agents with positive utility, sum of w_i log v_i over them)."""
    pos = [(x, w) for x, w in zip(u, weights) if x > 0]
    return (len(pos), math.fsum(w * math.log(x) for x, w in pos))


def optimal_vectors(vectors: Iterable[Vector], key: Callable[[Vector], tuple], rel_tol: float = 1e-9) -> List[Vector]:
    """Vectors maximizing ``key``. If the key's last entry is a float it is compared with tolerance."""
    vectors = list(vectors)
    keys = [key(u) for u in vectors]
    best = max(keys)

    def equal(a: tuple, b: tuple) -> bool:
        if a and isinstance(a[-1], float):
            return a[:-1] == b[:-1] and math.isclose(a[-1], b[-1], rel_tol=rel_tol, abs_tol=1e-12)
        return a == b

    return [u for u, k in zip(vectors, keys) if equal(k, best)]


def brute_force_optimum(
    n_agents: int,
    n_items: int,
    valuations: Sequence[Callable],
    objective: str,
    weights: Optional[Sequence[float]] = None,
    shares: Optional[Sequence[float]] = None,
    vectors: Optional[Set[Vector]] = None,
) -> Dict[str, object]:
    """Optimal utility vectors for ``objective`` found by exhaustive search.

    Returns a dict with ``"optimal"`` (all optimal utility vectors), ``"lex_max"``
    (the lexicographically largest optimal vector -- what General Yankee Swap
    returns according to Theorem 3.4), ``"max_usw"`` and ``"key"``.
    """
    if vectors is None:
        vectors = achievable_utility_vectors(n_agents, n_items, valuations)
    w = list(weights) if weights is not None else [1] * n_agents
    if objective in ("weighted_leximin", "leximin"):
        key = lambda u: weighted_leximin_key(u, w)  # noqa: E731
    elif objective == "maximin_share":
        if shares is None:
            shares = [brute_force_mms(n_agents, n_items, v) for v in valuations]
        key = lambda u: maximin_share_key(u, shares)  # noqa: E731
    elif objective == "weighted_nash_welfare":
        key = lambda u: nash_key(u, w)  # noqa: E731
    else:
        raise ValueError("unknown objective %r" % (objective,))
    optimal = optimal_vectors(vectors, key)
    return {
        "optimal": optimal,
        "lex_max": max(optimal),
        "max_usw": max(sum(u) for u in vectors),
        "key": key,
        "shares": shares,
    }
