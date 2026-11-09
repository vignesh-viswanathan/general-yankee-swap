"""Unit tests: paper examples, API behaviour, valuation helpers."""

import random

import pytest

from yankee_swap import (
    BinaryAdditive,
    Cached,
    CappedAdditive,
    GraphicValuation,
    NonMatroidValuationError,
    PartitionMatroid,
    TransversalValuation,
    Truncated,
    check_matroid_rank,
    general_yankee_swap,
    leximin_gain,
    weighted_leximin_gain,
    maximin_shares,
)
from yankee_swap.generators import VALUATION_KINDS, course_allocation_instance, random_instance


# ----------------------------------------------------------- paper examples
def test_example_1_1_weighted_leximin_vs_nash():
    """Example 1.1: w = (2, 8), seven goods, additive valuations."""
    vals = [BinaryAdditive(range(7))] * 2
    lex = general_yankee_swap(2, 7, vals, "weighted leximin", weights=[2, 8])
    nash = general_yankee_swap(2, 7, vals, "weighted nash welfare", weights=[2, 8])
    assert lex.utilities == (2, 5)
    assert nash.utilities == (1, 6)


def test_single_item_weighted_leximin_goes_to_lower_weight():
    """Section 6.2: one item, w = (1, 2) -> giving it to agent 1 (weight 1) is leximin."""
    vals = [BinaryAdditive([0])] * 2
    alloc = general_yankee_swap(2, 1, vals, "weighted_leximin", weights=[1, 2])
    assert alloc.utilities == (1, 0)
    alloc = general_yankee_swap(2, 1, vals, "weighted_leximin", weights=[2, 1])
    assert alloc.utilities == (0, 1)


def test_example_8_1_max_usw():
    """Example 8.1: v_i(S) = min(|S|, 2), four goods -> every agent gets two goods."""
    vals = [CappedAdditive(range(4), 2)] * 2
    for obj in ("weighted_leximin", "weighted_nash_welfare", "maximin_share"):
        kw = {} if obj == "maximin_share" else {"weights": [10, 1]}
        alloc = general_yankee_swap(2, 4, vals, obj, **kw)
        assert alloc.utilities == (2, 2)


def test_stealing_path_is_used():
    """Agent 0 only likes item 0, agent 1 likes items 0 and 1 (but takes item 0 first).

    Agent 1 moves first (lower utility tie -> index 0 is agent 0 actually), so we
    force the interesting order with a custom gain: agent 1 always first.
    """
    vals = [BinaryAdditive([0]), CappedAdditive([0, 1], 1)]

    def agent1_first(u, i):
        return (1 if i == 1 else 0, -u[i])

    alloc = general_yankee_swap(2, 2, vals, agent1_first)
    # Agent 1 grabs item 0 or 1; when agent 0 plays it must be able to get item 0
    # (possibly by stealing it and letting agent 1 take item 1).
    assert alloc.utilities == (1, 1)
    assert alloc.bundles[0] == frozenset({0}) and alloc.bundles[1] == frozenset({1})


def test_long_transfer_path():
    """A chain where the only way to serve the last agent is a long stealing path."""
    n = 6
    # agent k likes items k and k+1 but wants only one of them; agent n-1 likes only item n-1
    vals = [CappedAdditive([k, k + 1], 1) for k in range(n - 1)] + [BinaryAdditive([0])]
    # items 0..n-1; agents 0..n-2 grab item k first (lowest index in binary search),
    # the last agent needs item 0 -> path 0 -> 1 -> ... -> n-1
    alloc = general_yankee_swap(n, n, vals, "weighted_leximin", check=True)
    assert alloc.utilities == tuple([1] * n)
    assert alloc.bundles[n - 1] == frozenset({0})
    assert alloc.stats["max_path_length"] >= 2


# ----------------------------------------------------------------- API
def test_joint_oracle_and_aliases():
    liked = [{0, 1}, {1, 2}, {0, 2}]
    alloc_a = general_yankee_swap(3, 3, lambda i, S: len(S & liked[i]), "Weighted Leximin")
    alloc_b = general_yankee_swap(3, 3, [BinaryAdditive(s) for s in liked], "wleximin")
    assert alloc_a.utilities == alloc_b.utilities == (1, 1, 1)
    # two liked items cannot fill three bundles, so every MMS is 0
    alloc_c = general_yankee_swap(3, 3, lambda i, S: len(S & liked[i]), "maximin-share")
    assert alloc_c.stats["mms_values"] == [0, 0, 0]
    assert alloc_c.utilitarian_welfare == 3


def test_bad_inputs():
    vals = [BinaryAdditive([0])]
    with pytest.raises(ValueError):
        general_yankee_swap(1, 1, vals, "utilitarian-ish")
    with pytest.raises(ValueError):
        general_yankee_swap(1, 1, vals, "weighted_leximin", weights=[0])
    with pytest.raises(ValueError):
        general_yankee_swap(1, 1, vals, "weighted_leximin", weights=[1, 2])
    with pytest.raises(ValueError):
        general_yankee_swap(2, 1, vals, "weighted_leximin")
    with pytest.raises(ValueError):
        general_yankee_swap(1, 1, vals, "weighted_p_mean")
    with pytest.raises(ValueError):
        general_yankee_swap(1, 1, vals, "leximin", weights=[2])


def test_check_detects_non_matroid_oracle():
    # supermodular "pairs only" valuation: worthless alone, 2 for the pair
    def pairs(S):
        return 2 * (len(S & {0, 1}) // 2)

    assert not check_matroid_rank(pairs, range(2))
    with pytest.raises(NonMatroidValuationError):
        general_yankee_swap(1, 2, [pairs], "weighted_leximin", check=True)


def test_empty_instances():
    assert general_yankee_swap(0, 3, [], "weighted_leximin").unallocated == frozenset({0, 1, 2})
    alloc = general_yankee_swap(2, 0, [BinaryAdditive([])] * 2, "nash")
    assert alloc.utilities == (0, 0)


def test_allocation_helpers():
    vals = [BinaryAdditive([0, 1]), BinaryAdditive([2])]
    alloc = general_yankee_swap(2, 4, vals, "weighted_leximin")
    assert alloc.assignment(4) == [0, 0, 1, None]
    assert alloc.owner_of(2) == 1 and alloc.owner_of(3) is None
    assert alloc.utilitarian_welfare == 3
    d = alloc.to_dict()
    assert d["bundles"] == [[0, 1], [2]] and d["unallocated"] == [3]
    assert alloc.stats["valuation_queries"] > 0


def test_custom_gain_function_matches_builtin():
    vals = random_instance(4, 10, "mixed", seed=3)
    a = general_yankee_swap(4, 10, vals, weighted_leximin_gain([1, 2, 3, 1]))
    b = general_yankee_swap(4, 10, vals, "weighted_leximin", weights=[1, 2, 3, 1])
    assert a.utilities == b.utilities and a.bundles == b.bundles


def test_leximin_gain_and_alias():
    vals = random_instance(4, 10, "mixed", seed=5)
    a = general_yankee_swap(4, 10, vals, leximin_gain())
    b = general_yankee_swap(4, 10, vals, "leximin")
    c = general_yankee_swap(4, 10, vals, "weighted_leximin")  # unit weights
    assert a.bundles == b.bundles and b.utilities == c.utilities
    assert b.objective == "leximin"


def test_mms_on_identical_additive_agents():
    vals = [BinaryAdditive(range(7))] * 3
    assert maximin_shares(3, 7, vals) == [2, 2, 2]
    alloc = general_yankee_swap(3, 7, vals, "mms")
    assert sorted(alloc.utilities) == [2, 2, 3]


# ------------------------------------------------------------ valuations
@pytest.mark.parametrize("kind", VALUATION_KINDS)
def test_generated_valuations_are_matroid_rank(kind):
    rng = random.Random(kind)
    for seed in range(15):
        for v in random_instance(3, 8, kind=kind, seed=rng.randrange(10 ** 6)):
            assert check_matroid_rank(v, range(8), raise_error=True)


def test_valuation_classes():
    assert check_matroid_rank(PartitionMatroid([0, 1, 2, 3], {0: "a", 1: "a", 2: "b", 3: "b"}, {"a": 1, "b": 2}, 2), range(5))
    assert check_matroid_rank(TransversalValuation({0: ["x"], 1: ["x", "y"], 2: ["y"]}), range(3))
    g = GraphicValuation({0: (0, 1), 1: (1, 2), 2: (0, 2), 3: (3, 3)})
    assert g(frozenset({0, 1, 2, 3})) == 2 and check_matroid_rank(g, range(4))
    t = Truncated(BinaryAdditive(range(5)), 3)
    assert t(frozenset(range(5))) == 3 and check_matroid_rank(t, range(5))
    c = Cached(BinaryAdditive([1]))
    assert c(frozenset({1})) == 1 and c(frozenset({1})) == 1
    for course_v in course_allocation_instance(3, 12, seed=1):
        assert check_matroid_rank(course_v, range(12))


def test_deterministic():
    vals = random_instance(5, 30, "mixed", seed=11)
    a = general_yankee_swap(5, 30, vals, "weighted_nash_welfare", weights=[1, 2, 3, 4, 5])
    b = general_yankee_swap(5, 30, vals, "weighted_nash_welfare", weights=[1, 2, 3, 4, 5])
    assert a.bundles == b.bundles
