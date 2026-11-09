"""The :class:`Allocation` result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class Allocation:
    """Result of :func:`yankee_swap.general_yankee_swap`.

    Attributes
    ----------
    bundles:
        ``bundles[i]`` is the frozenset of items given to agent ``i``.
    unallocated:
        Items that are not given to anybody (no agent has a positive marginal
        value for any of them at the end of the algorithm).
    utilities:
        ``utilities[i] = v_i(bundles[i])``. The allocation is nonredundant,
        so this also equals ``len(bundles[i])``.
    objective:
        Name of the objective / gain function that was used.
    stats:
        Run statistics (iterations, number of valuation queries, runtime, ...).
    """

    bundles: Tuple[FrozenSet[int], ...]
    unallocated: FrozenSet[int]
    utilities: Tuple[int, ...]
    objective: str = ""
    stats: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def n_agents(self) -> int:
        return len(self.bundles)

    @property
    def utilitarian_welfare(self) -> int:
        """Sum of utilities (General Yankee Swap always maximizes this)."""
        return sum(self.utilities)

    def owner_of(self, item: int) -> Optional[int]:
        """Agent holding ``item`` or ``None`` if it is unallocated."""
        for i, bundle in enumerate(self.bundles):
            if item in bundle:
                return i
        return None

    def assignment(self, n_items: Optional[int] = None) -> List[Optional[int]]:
        """List mapping every item to its owner (``None`` = unallocated)."""
        if n_items is None:
            items = set(self.unallocated)
            for b in self.bundles:
                items |= b
            n_items = (max(items) + 1) if items else 0
        result: List[Optional[int]] = [None] * n_items
        for i, bundle in enumerate(self.bundles):
            for g in bundle:
                result[g] = i
        return result

    def as_lists(self) -> List[List[int]]:
        """Bundles as sorted lists (handy for printing / JSON)."""
        return [sorted(b) for b in self.bundles]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "bundles": self.as_lists(),
            "unallocated": sorted(self.unallocated),
            "utilities": list(self.utilities),
            "stats": dict(self.stats),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = ["Allocation (%s)" % (self.objective or "custom")]
        for i, b in enumerate(self.bundles):
            lines.append("  agent %d: utility %d, items %s" % (i, self.utilities[i], sorted(b)))
        lines.append("  unallocated: %s" % sorted(self.unallocated))
        return "\n".join(lines)
