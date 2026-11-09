"""Verify General Yankee Swap against exhaustive search on small random instances.

For every instance (n <= 4 agents, m <= 7 items, all valuation families) and
every objective we check that the output of General Yankee Swap

1. is a valid, nonredundant allocation (``v_i(X_i) = |X_i|``),
2. maximizes utilitarian social welfare (Proposition 3.6),
3. is optimal for the objective (Theorems 6.2, 6.3, 6.4, 6.6), and
4. is the lexicographically largest optimal utility vector (Theorem 3.4),
   which pins down the exact output, including the tie-breaking.
"""

import random

import pytest

from yankee_swap import general_yankee_swap, maximin_share
from yankee_swap.bruteforce import achievable_utility_vectors, brute_force_mms, brute_force_optimum
from yankee_swap.generators import VALUATION_KINDS, random_instance, random_weights

KINDS = list(VALUATION_KINDS) + ["mixed"]


def _instances():
    """(n, m, kind, seed) grid of small instances (~500 instances in total)."""
    cases = []
    sizes = [(1, 4), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (5, 4), (2, 7), (3, 7), (4, 6), (5, 5)]
    for n, m in sizes:
        for kind in KINDS:
            n_seeds = 3 if (n + 1) ** m > 10000 else 8
            for seed in range(n_seeds):
                cases.append((n, m, kind, 1000 * n + 100 * m + seed + 7 * KINDS.index(kind)))
    return cases


INSTANCES = _instances()


def _check_valid(alloc, n, m, vals):
    seen = set(alloc.unallocated)
    for i, bundle in enumerate(alloc.bundles):
        assert not (seen & bundle), "bundles overlap"
        seen |= bundle
        assert vals[i](bundle) == len(bundle) == alloc.utilities[i], "allocation is redundant"
    assert seen == set(range(m)), "items lost"


def _run_and_check(n, m, vals, objective, vectors, **kw):
    alloc = general_yankee_swap(n, m, vals, objective, check=True, **kw)
    _check_valid(alloc, n, m, vals)
    shares = kw.get("mms_values")
    ref = brute_force_optimum(
        n, m, vals, objective, weights=kw.get("weights"), shares=shares, vectors=vectors
    )
    u = alloc.utilities
    assert sum(u) == ref["max_usw"], "not utilitarian optimal"
    assert u in ref["optimal"], "%s: GYS output %s is not optimal; optima %s" % (objective, u, ref["optimal"][:5])
    return alloc, ref


@pytest.mark.parametrize("n,m,kind,seed", INSTANCES)
def test_against_brute_force(n, m, kind, seed):
    rng = random.Random(seed)
    vals = random_instance(n, m, kind=kind, seed=seed)
    vectors = achievable_utility_vectors(n, m, vals)

    # --- weighted leximin (unit, integer and real weights)
    for weights in ([1] * n, random_weights(n, rng, integer=True), random_weights(n, rng, integer=False)):
        alloc, ref = _run_and_check(n, m, vals, "weighted_leximin", vectors, weights=weights)
        assert alloc.utilities == ref["lex_max"]

    # --- unweighted leximin
    alloc, ref = _run_and_check(n, m, vals, "leximin", vectors)
    assert alloc.utilities == ref["lex_max"]

    # --- weighted Nash welfare
    for weights in (random_weights(n, rng, integer=True), random_weights(n, rng, integer=False)):
        alloc, ref = _run_and_check(n, m, vals, "weighted_nash_welfare", vectors, weights=weights)
        assert alloc.utilities == ref["lex_max"]

    # --- maximin share: our MMS values must match brute force, everybody gets their MMS
    mms_brute = [brute_force_mms(n, m, v) for v in vals]
    mms_ours = [maximin_share(n, m, v) for v in vals]
    assert mms_ours == mms_brute
    alloc, ref = _run_and_check(n, m, vals, "maximin_share", vectors, mms_values=mms_brute)
    assert all(alloc.utilities[i] >= mms_brute[i] for i in range(n)), "MMS not guaranteed"
    alloc_auto = general_yankee_swap(n, m, vals, "mms")  # computes the MMS values itself
    assert alloc_auto.utilities == alloc.utilities
    assert alloc_auto.stats["mms_values"] == mms_brute


def test_number_of_brute_force_instances():
    assert len(INSTANCES) >= 150
