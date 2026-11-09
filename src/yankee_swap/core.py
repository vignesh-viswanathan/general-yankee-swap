"""Core implementation of General Yankee Swap.

Reference
---------
V. Viswanathan and Y. Zick. *A General Framework for Fair Allocation under
Matroid Rank Valuations*. ACM Transactions on Economics and Computation 13(3),
Article 14, 2025. https://doi.org/10.1145/3728367

Overview
--------
All items start unallocated (in the pool ``X_0``). In every iteration the
algorithm picks the still-playing agent ``i`` with the highest *gain*
``phi(X, i)`` (ties broken in favour of the lowest index) and looks for a
shortest path in the *item exchange graph* from ``F_i(X)`` -- the items that
give ``i`` a marginal gain of 1 -- to the pool of unallocated items. If such a
path exists, the items are shifted along it (``i`` "steals" an item, the victim
steals another one, and so on until somebody takes an unallocated item); every
agent on the path keeps its utility and ``i`` gains one unit. Otherwise ``i``
leaves the game. (Algorithm 1 in the paper.)

Shortest paths are found with the breadth-first search of Algorithm 3
(``Get-Distances``), which never builds the exchange graph explicitly: the
out-neighbours of a node are discovered one at a time by the binary search of
Algorithm 2 (``Find-Desired``). This keeps the number of valuation queries at
``O(m log m)`` per iteration.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Union

from .allocation import Allocation
from .objectives import GainFunction, resolve_objective

Bundle = FrozenSet[int]
ValuationOracle = Callable[[Bundle], float]
Valuations = Union[Sequence[ValuationOracle], Callable[[int, Bundle], float]]

__all__ = ["general_yankee_swap", "yankee_swap", "NonMatroidValuationError"]


class NonMatroidValuationError(ValueError):
    """Raised (with ``check=True``) when an oracle is detectably not a matroid rank function."""


class _CountingOracle:
    """Wraps a valuation oracle and counts how often it is queried."""

    __slots__ = ("func", "calls")

    def __init__(self, func: ValuationOracle):
        self.func = func
        self.calls = 0

    def __call__(self, bundle: Bundle) -> float:
        self.calls += 1
        return self.func(bundle)


def _normalize_valuations(valuations: Valuations, n_agents: int) -> List[ValuationOracle]:
    """Accept either a list of per-agent oracles or one oracle ``f(agent, bundle)``."""
    if callable(valuations) and not isinstance(valuations, (list, tuple)):
        joint = valuations
        return [(lambda bundle, _i=i: joint(_i, bundle)) for i in range(n_agents)]
    oracles = list(valuations)
    if len(oracles) != n_agents:
        raise ValueError(
            "expected %d valuation oracles (one per agent), got %d" % (n_agents, len(oracles))
        )
    for i, f in enumerate(oracles):
        if not callable(f):
            raise TypeError("valuation oracle for agent %d is not callable" % i)
    return oracles


def _find_desired(oracle: ValuationOracle, base: Bundle, base_value: float, candidates: List[int]) -> Optional[int]:
    """Algorithm 2 (Find-Desired).

    Return an item ``g`` in ``candidates`` with ``v(base + g) = v(base) + 1``, or
    ``None`` if there is no such item. Uses ``O(log |candidates|)`` queries.
    """
    if not candidates:
        return None
    if oracle(base.union(candidates)) <= base_value:
        return None
    while len(candidates) > 1:
        half = len(candidates) // 2
        first = candidates[:half]
        if oracle(base.union(first)) > base_value:
            candidates = first
        else:
            candidates = candidates[half:]
    return candidates[0]


class _State:
    """Mutable allocation maintained by the algorithm (always nonredundant)."""

    __slots__ = ("n", "m", "oracles", "bundles", "owner", "unallocated", "sizes")

    UNALLOCATED = -1

    def __init__(self, n: int, m: int, oracles: List[ValuationOracle]):
        self.n = n
        self.m = m
        self.oracles = oracles
        self.bundles: List[Set[int]] = [set() for _ in range(n)]
        self.owner: List[int] = [self.UNALLOCATED] * m
        # dict used as an insertion-ordered set => deterministic behaviour
        self.unallocated: Dict[int, None] = dict.fromkeys(range(m))
        self.sizes: List[int] = [0] * n

    # ------------------------------------------------------------------ paths
    def shortest_path(self, i: int) -> Optional[List[int]]:
        """Shortest path in the exchange graph from ``F_i(X)`` to ``X_0`` (Algorithm 3).

        Returns the path as a list of items ``[g_1, ..., g_t]`` with ``g_1 in F_i(X)``
        and ``g_t`` unallocated, or ``None`` if no such path exists.
        """
        owner = self.owner
        oracle = self.oracles[i]
        bundle_i = frozenset(self.bundles[i])
        value_i = self.sizes[i]

        # Fast path: an unallocated item that i likes is a path of length 0.
        g = _find_desired(oracle, bundle_i, value_i, list(self.unallocated))
        if g is not None:
            return [g]

        # F_i(X) restricted to allocated items (none of the unallocated ones qualify).
        pool = [h for h in range(self.m) if owner[h] != self.UNALLOCATED and owner[h] != i]
        frontier: List[int] = []
        while True:
            g = _find_desired(oracle, bundle_i, value_i, pool)
            if g is None:
                break
            frontier.append(g)
            pool.remove(g)
        if not frontier:
            return None

        # Breadth-first search. `undiscovered` lists unallocated items first so
        # that Find-Desired tends to hit a path end as early as possible.
        prev: Dict[int, Optional[int]] = {g: None for g in frontier}
        undiscovered: Dict[int, None] = dict.fromkeys(self.unallocated)
        for h in range(self.m):
            if h not in prev and owner[h] != self.UNALLOCATED:
                undiscovered[h] = None
        queue = deque(frontier)
        while queue:
            a = queue.popleft()
            j = owner[a]  # always an agent: unallocated items are never enqueued
            base = frozenset(self.bundles[j])
            base = base.difference((a,))
            base_value = self.sizes[j] - 1
            oracle_j = self.oracles[j]
            while undiscovered:
                b = _find_desired(oracle_j, base, base_value, list(undiscovered))
                if b is None:
                    break
                del undiscovered[b]
                prev[b] = a
                if owner[b] == self.UNALLOCATED:
                    path = [b]
                    node = a
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    path.reverse()
                    return path
                queue.append(b)
        return None

    def augment(self, i: int, path: List[int]) -> List[int]:
        """Transfer items along ``path`` and give ``path[0]`` to agent ``i``.

        Returns the list of agents whose bundles changed.
        """
        owners = [self.owner[g] for g in path]
        assert owners[-1] == self.UNALLOCATED
        for k in range(len(path) - 1):
            self.bundles[owners[k]].remove(path[k])
        for k in range(len(path) - 1):
            self.bundles[owners[k]].add(path[k + 1])
            self.owner[path[k + 1]] = owners[k]
        del self.unallocated[path[-1]]
        self.bundles[i].add(path[0])
        self.owner[path[0]] = i
        self.sizes[i] += 1
        changed = set(owners[:-1])
        changed.add(i)
        return sorted(changed)


def general_yankee_swap(
    n_agents: int,
    n_items: int,
    valuations: Valuations,
    objective: Union[str, GainFunction],
    *,
    weights: Optional[Sequence[float]] = None,
    mms_values: Optional[Sequence[float]] = None,
    check: bool = False,
) -> Allocation:
    """Compute an allocation with General Yankee Swap.

    Parameters
    ----------
    n_agents:
        Number of agents. Agents are indexed ``0, ..., n_agents - 1``; the index
        is used to break ties (lower index first), exactly as in the paper.
    n_items:
        Number of (indivisible) items, indexed ``0, ..., n_items - 1``.
    valuations:
        Value oracles. Either a sequence of ``n_agents`` callables
        ``v_i(bundle) -> number`` or a single callable ``v(agent, bundle)``.
        ``bundle`` is a ``frozenset`` of item indices. Every ``v_i`` must be a
        *matroid rank function* (a.k.a. binary submodular valuation): ``v(∅)=0``,
        each marginal gain is 0 or 1, and marginal gains are non-increasing.
    objective:
        ``"weighted_nash_welfare"``, ``"weighted_leximin"``, ``"leximin"`` (unweighted) or
        ``"maximin_share"`` (aliases such as ``"weighted nash welfare"``,
        ``"nash"``, ``"mms"`` are accepted). Alternatively pass your own gain function
        ``phi(utilities, i) -> comparable`` (see :mod:`yankee_swap.objectives`).
    weights:
        Positive agent weights (entitlements) for the weighted objectives.
        Defaults to all ones.
    mms_values:
        Precomputed maximin shares (or any other fair shares) for the
        ``"maximin_share"`` objective. If omitted they are computed exactly with
        :func:`yankee_swap.maximin_shares`.
    check:
        If true, verify after every transfer that the allocation stays
        nonredundant (``v_i(X_i) = |X_i|``). This costs extra queries but
        catches oracles that are not matroid rank functions.

    Returns
    -------
    Allocation
        The (nonredundant) allocation. ``allocation.bundles[i]`` is agent ``i``'s
        bundle; ``allocation.unallocated`` holds items nobody can use.
    """
    if n_agents < 0 or n_items < 0:
        raise ValueError("n_agents and n_items must be non-negative")
    n_agents = int(n_agents)
    n_items = int(n_items)
    raw = _normalize_valuations(valuations, n_agents)
    counters = [_CountingOracle(f) for f in raw]

    mms_time = 0.0
    if isinstance(objective, str):
        name, gain, extra = resolve_objective(
            objective, n_agents, n_items, raw, weights=weights, mms_values=mms_values
        )
        mms_time = extra.get("mms_seconds", 0.0)
    elif callable(objective):
        name, gain, extra = getattr(objective, "__name__", "custom"), objective, {}
    else:
        raise TypeError("objective must be a string or a gain function")

    start = time.perf_counter()  # excludes the (optional) MMS precomputation
    state = _State(n_agents, n_items, counters)
    active = list(range(n_agents))  # kept sorted => ties go to the lowest index
    iterations = 0
    transfers = 0
    path_lengths: List[int] = []

    while active:
        iterations += 1
        utilities = tuple(state.sizes)
        best_agent = active[0]
        best_gain = gain(utilities, best_agent)
        for k in active[1:]:
            g = gain(utilities, k)
            if g > best_gain:
                best_agent, best_gain = k, g
        path = state.shortest_path(best_agent)
        if path is None:
            active.remove(best_agent)
            continue
        changed = state.augment(best_agent, path)
        transfers += 1
        path_lengths.append(len(path))
        if check:
            for j in changed:
                value = raw[j](frozenset(state.bundles[j]))
                if value != len(state.bundles[j]):
                    raise NonMatroidValuationError(
                        "agent %d's valuation is not a matroid rank function: after a "
                        "transfer its bundle of size %d has value %r"
                        % (j, len(state.bundles[j]), value)
                    )

    elapsed = time.perf_counter() - start
    stats = {
        "iterations": iterations,
        "transfers": transfers,
        "valuation_queries": sum(c.calls for c in counters),
        "max_path_length": max(path_lengths) if path_lengths else 0,
        "mean_path_length": (sum(path_lengths) / len(path_lengths)) if path_lengths else 0.0,
        "runtime_seconds": elapsed,
        "mms_seconds": mms_time,
    }
    stats.update({k: v for k, v in extra.items() if k not in stats})
    return Allocation(
        bundles=tuple(frozenset(b) for b in state.bundles),
        unallocated=frozenset(state.unallocated),
        utilities=tuple(state.sizes),
        objective=name,
        stats=stats,
    )


#: Short alias.
yankee_swap = general_yankee_swap
