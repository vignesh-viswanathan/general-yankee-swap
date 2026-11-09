"""General Yankee Swap: fair allocation under matroid rank valuations.

Implementation of the General Yankee Swap framework of Viswanathan and Zick,
*A General Framework for Fair Allocation under Matroid Rank Valuations*,
ACM Transactions on Economics and Computation 13(3), 2025.

Quick start::

    from yankee_swap import general_yankee_swap, CappedAdditive

    valuations = [CappedAdditive({0, 1, 2}, cap=2), CappedAdditive({1, 2, 3}, cap=2)]
    alloc = general_yankee_swap(2, 4, valuations, "weighted_nash_welfare", weights=[1, 2])
    print(alloc.bundles, alloc.utilities)
"""

import sys

if sys.version_info < (3, 8):
    raise ImportError(
        "yankee_swap requires Python 3.8 or newer; this interpreter is Python %d.%d (%s)"
        % (sys.version_info[0], sys.version_info[1], sys.executable)
    )

from .allocation import Allocation  # noqa: E402
from .core import NonMatroidValuationError, general_yankee_swap, yankee_swap
from .mms import maximin_share, maximin_shares
from .objectives import (
    leximin_gain,
    maximin_share_gain,
    weighted_leximin_gain,
    weighted_nash_gain,
)
from .valuations import (
    BinaryAdditive,
    Cached,
    CappedAdditive,
    GraphicValuation,
    PartitionMatroid,
    TransversalValuation,
    Truncated,
    check_matroid_rank,
)

__version__ = "0.1.0"

__all__ = [
    "Allocation",
    "general_yankee_swap",
    "yankee_swap",
    "NonMatroidValuationError",
    "maximin_share",
    "maximin_shares",
    "leximin_gain",
    "weighted_leximin_gain",
    "maximin_share_gain",
    "weighted_nash_gain",
    "BinaryAdditive",
    "CappedAdditive",
    "PartitionMatroid",
    "TransversalValuation",
    "GraphicValuation",
    "Truncated",
    "Cached",
    "check_matroid_rank",
    "__version__",
]
