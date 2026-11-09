"""Random instance generators (matroid rank valuations) for tests and experiments."""

from __future__ import annotations

import random
from typing import Callable, List, Optional

from .valuations import (
    BinaryAdditive,
    CappedAdditive,
    GraphicValuation,
    PartitionMatroid,
    TransversalValuation,
)

__all__ = [
    "random_valuation",
    "random_instance",
    "random_weights",
    "course_allocation_instance",
    "VALUATION_KINDS",
]

VALUATION_KINDS = ("additive", "capped", "partition", "transversal", "graphic")


def _liked(rng: random.Random, n_items: int, density: float) -> List[int]:
    return [g for g in range(n_items) if rng.random() < density]


def random_valuation(
    kind: str, n_items: int, rng: random.Random, density: float = 0.5
) -> Callable:
    """One random matroid rank valuation of the given ``kind``."""
    if kind == "mixed":
        kind = rng.choice(VALUATION_KINDS)
    if kind == "additive":
        return BinaryAdditive(_liked(rng, n_items, density))
    if kind == "capped":
        liked = _liked(rng, n_items, density)
        return CappedAdditive(liked, rng.randint(1, max(1, len(liked))))
    if kind == "partition":
        n_cats = max(1, rng.randint(1, max(1, n_items // 3)))
        category_of = {g: rng.randrange(n_cats) for g in range(n_items)}
        caps = {c: rng.randint(1, 3) for c in range(n_cats)}
        liked = _liked(rng, n_items, density)
        total = rng.randint(1, max(1, len(liked))) if rng.random() < 0.5 else None
        return PartitionMatroid(liked, category_of, caps, total)
    if kind == "transversal":
        n_slots = rng.randint(1, max(1, n_items // 2))
        compatible = {}
        for g in _liked(rng, n_items, density):
            compatible[g] = [s for s in range(n_slots) if rng.random() < 0.4] or [rng.randrange(n_slots)]
        return TransversalValuation(compatible)
    if kind == "graphic":
        n_vertices = rng.randint(2, max(2, n_items // 2 + 1))
        edges = {}
        for g in _liked(rng, n_items, density):
            edges[g] = (rng.randrange(n_vertices), rng.randrange(n_vertices))
        return GraphicValuation(edges)
    raise ValueError("unknown valuation kind %r" % (kind,))


def random_weights(n_agents: int, rng: random.Random, integer: bool = True) -> List[float]:
    if integer:
        return [rng.randint(1, 5) for _ in range(n_agents)]
    return [round(rng.uniform(0.2, 5.0), 3) for _ in range(n_agents)]


def random_instance(
    n_agents: int,
    n_items: int,
    kind: str = "mixed",
    seed: Optional[int] = None,
    density: float = 0.5,
) -> List[Callable]:
    """A list of ``n_agents`` random matroid rank valuations over ``n_items`` items."""
    rng = random.Random(seed)
    return [random_valuation(kind, n_items, rng, density) for _ in range(n_agents)]


def course_allocation_instance(
    n_agents: int,
    n_items: int,
    seed: Optional[int] = None,
    n_slots: Optional[int] = None,
    max_courses: Optional[int] = None,
    n_liked: Optional[int] = None,
) -> List[Callable]:
    """Course-allocation style random instance.

    Items are course seats grouped into ``n_slots`` time slots. Seats have a
    skewed (Zipf-like) popularity, so popular seats are contested. Each
    student likes ``n_liked`` seats sampled by popularity, can take at most one
    course per time slot and at most ``max_courses`` courses overall (by
    default about twice their fair share ``m / n``, so demand exceeds supply).
    """
    rng = random.Random(seed)
    if n_slots is None:
        n_slots = max(2, n_items // 8)
    if max_courses is None:
        max_courses = max(2, (2 * n_items) // max(1, n_agents))
    if n_liked is None:
        n_liked = min(n_items, 3 * max_courses)
    slot_of = {g: rng.randrange(n_slots) for g in range(n_items)}
    caps = {s: 1 for s in range(n_slots)}
    popularity = [1.0 / (rank + 1) ** 0.8 for rank in range(n_items)]
    rng.shuffle(popularity)
    vals = []
    for _ in range(n_agents):
        liked = set()
        while len(liked) < n_liked:
            liked.update(rng.choices(range(n_items), weights=popularity, k=n_liked - len(liked)))
        vals.append(PartitionMatroid(liked, slot_of, caps, total_cap=rng.randint(1, max_courses)))
    return vals
