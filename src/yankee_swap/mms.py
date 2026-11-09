"""Exact maximin shares for matroid rank valuations.

``MMS_i = max over partitions (P_1, ..., P_n) of the items of min_k v_i(P_k)``.

With matroid rank valuations this can be computed in polynomial time with
General Yankee Swap itself: give ``n`` identical copies of ``v_i`` to ``n``
clone agents and compute a leximin allocation (Theorem 6.2). A leximin
allocation maximizes the minimum utility, and since valuations are monotone
the unallocated items can be added to any bundle without lowering it, so the
minimum utility of that allocation is exactly ``MMS_i``.
"""

from __future__ import annotations

from typing import Callable, FrozenSet, List, Sequence

__all__ = ["maximin_share", "maximin_shares"]


def maximin_share(n_agents: int, n_items: int, valuation: Callable[[FrozenSet[int]], float]) -> int:
    """Maximin share of an agent with matroid rank valuation ``valuation``."""
    from .core import general_yankee_swap
    from .valuations import Truncated

    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    # Upper bound: every bundle of a partition needs MMS_i items that are
    # individually valuable, so MMS_i <= floor(#{g : v({g}) = 1} / n); also
    # MMS_i <= v(G). (Note: v(G) / n is NOT a bound -- valuations are
    # submodular, so bundle values can sum to more than v(G).) Truncating the
    # clones at this bound does not change the answer (min(v, k) has maximin
    # share min(MMS, k)) but makes the leximin run cheaper, and lets us skip it
    # entirely when the bound is 0.
    useful = sum(1 for g in range(n_items) if valuation(frozenset((g,))) > 0)
    bound = min(int(valuation(frozenset(range(n_items)))), useful // n_agents)
    if bound == 0:
        return 0
    clones = [Truncated(valuation, bound)] * n_agents
    alloc = general_yankee_swap(n_agents, n_items, clones, objective="leximin")
    return min(alloc.utilities)


def maximin_shares(
    n_agents: int, n_items: int, valuations: Sequence[Callable[[FrozenSet[int]], float]]
) -> List[int]:
    """Maximin shares of all agents (one leximin computation per agent)."""
    return [maximin_share(n_agents, n_items, v) for v in valuations]
