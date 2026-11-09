"""Ready-made matroid rank valuation oracles and a brute-force validator.

Every class here is a callable ``v(bundle) -> int`` that can be passed directly
to :func:`yankee_swap.general_yankee_swap`. They are all matroid rank
functions (binary submodular valuations). You can of course write your own
oracle -- any function of a ``frozenset`` of item indices works -- and use
:func:`check_matroid_rank` to sanity-check it on small item sets.
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "BinaryAdditive",
    "CappedAdditive",
    "PartitionMatroid",
    "TransversalValuation",
    "GraphicValuation",
    "Truncated",
    "Cached",
    "check_matroid_rank",
]


class BinaryAdditive:
    """``v(S) = |S ∩ liked|`` -- the agent wants every item in ``liked``."""

    def __init__(self, liked: Iterable[int]):
        self.liked = frozenset(liked)

    def __call__(self, bundle: FrozenSet[int]) -> int:
        return len(self.liked.intersection(bundle))

    def __repr__(self) -> str:
        return "BinaryAdditive(liked=%s)" % sorted(self.liked)


class CappedAdditive:
    """``v(S) = min(|S ∩ liked|, cap)`` -- e.g. "I want at most ``cap`` courses"."""

    def __init__(self, liked: Iterable[int], cap: int):
        if cap < 0:
            raise ValueError("cap must be non-negative")
        self.liked = frozenset(liked)
        self.cap = int(cap)

    def __call__(self, bundle: FrozenSet[int]) -> int:
        return min(len(self.liked.intersection(bundle)), self.cap)

    def __repr__(self) -> str:
        return "CappedAdditive(liked=%s, cap=%d)" % (sorted(self.liked), self.cap)


class PartitionMatroid:
    """Liked items grouped into categories with per-category and overall caps.

    ``v(S) = min(total_cap, sum_c min(caps[c], |S ∩ liked ∩ category c|))``.

    This is the "course allocation" valuation from the paper's introduction:
    a student wants at most ``caps[c]`` courses from time slot / category ``c``
    and at most ``total_cap`` courses in total. (A truncated partition
    matroid, hence a matroid rank function.)
    """

    def __init__(
        self,
        liked: Iterable[int],
        category_of: Mapping[int, Hashable],
        caps: Mapping[Hashable, int],
        total_cap: Optional[int] = None,
    ):
        self.liked = frozenset(liked)
        self.category_of = dict(category_of)
        missing = [g for g in self.liked if g not in self.category_of]
        if missing:
            raise ValueError("liked items without a category: %s" % sorted(missing))
        self.caps = dict(caps)
        self.total_cap = total_cap

    def __call__(self, bundle: FrozenSet[int]) -> int:
        counts: Dict[Hashable, int] = {}
        for g in self.liked.intersection(bundle):
            c = self.category_of[g]
            counts[c] = counts.get(c, 0) + 1
        value = sum(min(k, self.caps.get(c, 0)) for c, k in counts.items())
        if self.total_cap is not None:
            value = min(value, self.total_cap)
        return value

    def __repr__(self) -> str:
        return "PartitionMatroid(|liked|=%d, caps=%s, total_cap=%s)" % (
            len(self.liked),
            self.caps,
            self.total_cap,
        )


class TransversalValuation:
    """Matching-based (OXS / transversal matroid) valuation.

    ``compatible[g]`` lists the "slots" item ``g`` can fill; ``v(S)`` is the size
    of a maximum matching between ``S`` and the slots. E.g. slots are a
    student's free time slots and items are course sections.
    """

    def __init__(self, compatible: Mapping[int, Iterable[Hashable]]):
        self.compatible = {g: tuple(slots) for g, slots in compatible.items()}

    def __call__(self, bundle: FrozenSet[int]) -> int:
        match_of_slot: Dict[Hashable, int] = {}
        items = [g for g in bundle if self.compatible.get(g)]

        def try_assign(g: int, seen: set) -> bool:
            for s in self.compatible[g]:
                if s in seen:
                    continue
                seen.add(s)
                if s not in match_of_slot or try_assign(match_of_slot[s], seen):
                    match_of_slot[s] = g
                    return True
            return False

        return sum(1 for g in items if try_assign(g, set()))

    def __repr__(self) -> str:
        return "TransversalValuation(%d items)" % len(self.compatible)


class GraphicValuation:
    """Graphic matroid: item ``g`` is the edge ``edges[g] = (u, v)``.

    ``v(S)`` is the size of a spanning forest of the edges in ``S`` (items not
    in ``edges`` and self-loops are worthless).
    """

    def __init__(self, edges: Mapping[int, Tuple[Hashable, Hashable]]):
        self.edges = dict(edges)

    def __call__(self, bundle: FrozenSet[int]) -> int:
        parent: Dict[Hashable, Hashable] = {}

        def find(x: Hashable) -> Hashable:
            root = x
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(x, x) != root:
                parent[x], x = root, parent[x]
            return root

        rank = 0
        for g in bundle:
            e = self.edges.get(g)
            if e is None:
                continue
            ru, rv = find(e[0]), find(e[1])
            if ru != rv:
                parent[ru] = rv
                rank += 1
        return rank

    def __repr__(self) -> str:
        return "GraphicValuation(%d edges)" % len(self.edges)


class Truncated:
    """``min(f(S), k)`` -- the truncation of a matroid rank function is one too."""

    def __init__(self, func: Callable[[FrozenSet[int]], int], k: int):
        self.func = func
        self.k = int(k)

    def __call__(self, bundle: FrozenSet[int]) -> int:
        return min(self.func(bundle), self.k)

    def __repr__(self) -> str:
        return "Truncated(%r, k=%d)" % (self.func, self.k)


class Cached:
    """Memoizes an expensive oracle (useful when queries are costly)."""

    def __init__(self, func: Callable[[FrozenSet[int]], int], maxsize: Optional[int] = None):
        self.func = func
        self.maxsize = maxsize
        self._cache: Dict[FrozenSet[int], int] = {}

    def __call__(self, bundle: FrozenSet[int]) -> int:
        bundle = frozenset(bundle)
        try:
            return self._cache[bundle]
        except KeyError:
            value = self.func(bundle)
            if self.maxsize is None or len(self._cache) < self.maxsize:
                self._cache[bundle] = value
            return value

    def __repr__(self) -> str:
        return "Cached(%r)" % (self.func,)


def check_matroid_rank(
    func: Callable[[FrozenSet[int]], float], items: Sequence[int], raise_error: bool = False
) -> bool:
    """Brute-force check that ``func`` restricted to ``items`` is a matroid rank function.

    Checks ``v(∅) = 0``, that every marginal gain is 0 or 1, and submodularity
    (``Δ(S, g) >= Δ(S + h, g)``). Runs in ``O(2^k k^2)`` for ``k = len(items)``,
    so only use it for small item sets (say ``k <= 14``).
    """
    items = list(items)
    k = len(items)
    if k > 20:
        raise ValueError("check_matroid_rank is exponential; use at most 20 items")
    value: Dict[int, float] = {}
    for mask in range(1 << k):
        value[mask] = func(frozenset(items[t] for t in range(k) if mask >> t & 1))

    def fail(msg: str) -> bool:
        if raise_error:
            raise ValueError(msg)
        return False

    if value[0] != 0:
        return fail("v(empty set) = %r != 0" % (value[0],))
    for mask in range(1 << k):
        for t in range(k):
            if mask >> t & 1:
                continue
            gain = value[mask | 1 << t] - value[mask]
            if gain not in (0, 1):
                return fail("marginal gain %r of item %r is not 0/1" % (gain, items[t]))
            for u in range(k):
                if u == t or mask >> u & 1:
                    continue
                bigger = mask | 1 << u
                if value[bigger | 1 << t] - value[bigger] > gain:
                    return fail("not submodular at item %r" % (items[t],))
    return True
