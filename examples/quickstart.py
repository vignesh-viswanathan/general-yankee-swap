"""Quick start for the general-yankee-swap package.

Run with:  python examples/quickstart.py

This works whether or not the package is installed: if it isn't, the
repository's ``src/`` folder is added to the import path.
"""

import importlib.util
import os
import sys

if importlib.util.find_spec("yankee_swap") is None:  # not installed: use the checkout
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from yankee_swap import (  # noqa: E402
    BinaryAdditive,
    CappedAdditive,
    PartitionMatroid,
    general_yankee_swap,
    maximin_shares,
)

# ---------------------------------------------------------------------------
# 1. Example 1.1 from the paper: two agents with weights 2 and 8, seven goods
#    that both agents like. Weighted leximin and weighted Nash welfare differ.
# ---------------------------------------------------------------------------
vals = [BinaryAdditive(range(7)), BinaryAdditive(range(7))]
for objective in ("weighted leximin", "weighted nash welfare"):
    alloc = general_yankee_swap(2, 7, vals, objective, weights=[2, 8])
    print("%-22s -> utilities %s" % (objective, alloc.utilities))

# ---------------------------------------------------------------------------
# 2. A small course allocation instance.
#    Items 0-5 are course seats; seats 0-2 are in the morning, 3-5 in the
#    afternoon. A student takes at most one course per time slot.
# ---------------------------------------------------------------------------
slot = {0: "am", 1: "am", 2: "am", 3: "pm", 4: "pm", 5: "pm"}
students = [
    PartitionMatroid(liked=[0, 1, 3], category_of=slot, caps={"am": 1, "pm": 1}),
    PartitionMatroid(liked=[0, 3, 4], category_of=slot, caps={"am": 1, "pm": 1}),
    PartitionMatroid(liked=[1, 2, 5], category_of=slot, caps={"am": 1, "pm": 1}, total_cap=1),
]
print()
print("MMS values:", maximin_shares(3, 6, students))
alloc = general_yankee_swap(3, 6, students, "maximin share")
print(alloc)

# ---------------------------------------------------------------------------
# 3. Any Python function of a frozenset of item indices can be an oracle,
#    as long as it is a matroid rank function. Here: "at most 2 of my items".
# ---------------------------------------------------------------------------
def my_valuation(bundle):
    return min(len(bundle & {0, 1, 2, 3}), 2)


alloc = general_yankee_swap(
    3, 6, [my_valuation, CappedAdditive([2, 3, 4], 3), BinaryAdditive([0, 5])],
    "weighted_nash_welfare", weights=[1, 1, 2],
)
print()
print(alloc)
print("stats:", {k: alloc.stats[k] for k in ("iterations", "transfers", "valuation_queries")})
