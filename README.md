# General Yankee Swap

[![tests](https://github.com/vignesh-viswanathan/general-yankee-swap/actions/workflows/tests.yml/badge.svg)](https://github.com/vignesh-viswanathan/general-yankee-swap/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A small, dependency-free Python package implementing **General Yankee Swap**,
the algorithmic framework from

> Vignesh Viswanathan and Yair Zick. **A General Framework for Fair Allocation
> under Matroid Rank Valuations.** *ACM Transactions on Economics and
> Computation* 13(3), Article 14, 2025. [doi:10.1145/3728367](https://doi.org/10.1145/3728367)

Given agents with **matroid rank valuations** (binary submodular valuations:
every item adds 0 or 1 to a bundle and marginal gains are non-increasing),
General Yankee Swap computes an allocation of indivisible goods that maximizes
a chosen justice criterion. It also maximizes utilitarian social welfare, is
strategyproof (Section 4 of the paper; for maximin share this holds only for
fixed MMS values, since computing them is not strategyproof), and uses a
polynomial number of valuation queries.

Supported objectives:

| objective (string) | justice criterion | gain function φ(X, i) | paper |
|---|---|---|---|
| `"weighted_nash_welfare"` | max weighted Nash welfare: first minimize the number of zero-utility agents, then maximize ∏ v_i(X_i)^{w_i} | (1 + 1/v_i)^{w_i} if v_i > 0, otherwise a large M | Thm 6.6 |
| `"weighted_leximin"` | lexicographically maximize the sorted vector of v_i(X_i)/w_i | (−v_i/w_i, −w_i) | Thm 6.3 |
| `"leximin"` | unweighted leximin: lexicographically maximize the sorted utility vector | −v_i (ties: lowest index) | Thm 6.2 |
| `"maximin_share"` | lexicographically maximize the sorted vector of v_i(X_i)/MMS_i, so every agent gets their maximin share | (−v_i/MMS_i, −MMS_i) if MMS_i > 0, otherwise −∞ | Thm 6.4, Cor 6.5 |

You can also pass **your own gain function** (any criterion that satisfies the
paper's conditions C1/C2).

## Installation

```bash
pip install git+https://github.com/vignesh-viswanathan/general-yankee-swap.git
```

or, from a local clone (the `[test]` extra installs `pytest`):

```bash
pip install -e ".[test]"
```

The package has no runtime dependencies and supports Python 3.8+. Editable
installs (`-e`) need pip 21.3 or newer, so run `python -m pip install --upgrade pip` first if
pip complains about a missing `setup.py`. The example scripts also run straight
from a clone without installing anything.

## Quick start

```python
from yankee_swap import general_yankee_swap, BinaryAdditive

# Example 1.1 from the paper: 2 agents with weights 2 and 8, 7 goods both like.
valuations = [BinaryAdditive(range(7)), BinaryAdditive(range(7))]

alloc = general_yankee_swap(
    n_agents=2,
    n_items=7,
    valuations=valuations,              # one value oracle per agent
    objective="weighted nash welfare",  # or "weighted leximin", "maximin share"
    weights=[2, 8],
)
print(alloc.utilities)   # (1, 6)
print(alloc.bundles)     # (frozenset({...}), frozenset({...}))
print(alloc.unallocated) # items nobody can use
print(alloc.stats)       # iterations, valuation queries, runtime, ...
```

With `objective="weighted leximin"` the same instance gives `(2, 5)`.
For the maximin share objective the MMS values are computed exactly unless you
pass them yourself:

```python
alloc = general_yankee_swap(n, m, valuations, "maximin share")
alloc.stats["mms_values"]          # the computed maximin shares
alloc = general_yankee_swap(n, m, valuations, "mms", mms_values=[2, 1, 3])
```

See [`examples/quickstart.py`](examples/quickstart.py) for more.

## Writing valuation oracles

A valuation oracle is any callable that takes a `frozenset` of item indices
(`0 … n_items-1`) and returns the bundle's value. You can pass either a list
with one oracle per agent, or a single function `v(agent, bundle)`:

```python
from yankee_swap import general_yankee_swap

liked = [{0, 1, 2}, {1, 2, 3}]
def v(agent, bundle):                       # "I want at most 2 of my liked items"
    return min(len(bundle & liked[agent]), 2)

alloc = general_yankee_swap(2, 4, v, "weighted_leximin", weights=[1, 3])
```

**The oracle must be a matroid rank function**; the guarantees do not hold
otherwise. `check_matroid_rank(v, items)` checks this by brute force on small
item sets, and `general_yankee_swap(..., check=True)` checks after every step
that the allocation stays nonredundant (v_i(X_i) = |X_i|). Ready-made oracles:

| class | valuation |
|---|---|
| `BinaryAdditive(liked)` | v(S) = \|S ∩ liked\| |
| `CappedAdditive(liked, cap)` | min(\|S ∩ liked\|, cap) |
| `PartitionMatroid(liked, category_of, caps, total_cap=None)` | at most `caps[c]` liked items per category, at most `total_cap` overall (e.g. course allocation with time slots) |
| `TransversalValuation(compatible)` | maximum matching between items and "slots" (OXS valuations) |
| `GraphicValuation(edges)` | items are edges of a graph; value = size of a spanning forest |
| `Truncated(v, k)` | min(v(S), k) |
| `Cached(v)` | memoizes an expensive oracle |

## Custom gain functions

A gain function maps `(utilities, i)` to a comparable value (a number or a
tuple, compared lexicographically). The active agent with the largest gain
moves next; ties go to the lowest index. For example, to maximize
v_1 + √(3·v_2) (an asymmetric objective in the spirit of Example 3.2 in the
paper), use the marginal gain of each agent's term:

```python
import math
from yankee_swap import general_yankee_swap, BinaryAdditive

def phi(u, i):
    if i == 0:
        return 1.0                                          # d/dv of v_1
    return math.sqrt(3 * (u[1] + 1)) - math.sqrt(3 * u[1])  # gain of sqrt(3 v_2)

valuations = [BinaryAdditive(range(8)), BinaryAdditive(range(8))]
alloc = general_yankee_swap(2, 8, valuations, phi)
print(alloc.utilities)   # (7, 1)
```

General Yankee Swap is guaranteed to be optimal only if the criterion respects
Pareto dominance (C1) and φ satisfies Consistency (G1) and Generalized
Concavity (G2); see Section 3.1 of the paper.

## How it works

The implementation follows the paper closely ([`src/yankee_swap/core.py`](src/yankee_swap/core.py)):

* **Algorithm 1 (General Yankee Swap).** All items start unallocated. In each
  iteration the active agent `i` with the largest gain φ(X, i) is chosen (ties
  go to the lowest index). If there is a path in the exchange graph from
  F_i(X) (items worth 1 to `i`) to an unallocated item, items are moved along
  a **shortest** such path: `i` takes an item, its previous owner takes the
  next item on the path, and so on. Everyone keeps their utility and `i` gains
  one. Otherwise `i` leaves the game.
* **Algorithms 2 and 3 (Find-Desired, Get-Distances).** The exchange graph is
  never built. Shortest paths come from a breadth-first search whose edges are
  found by binary search over item sets, which uses O(m log m) value queries
  per iteration.
* **Maximin shares** ([`src/yankee_swap/mms.py`](src/yankee_swap/mms.py)) are
  computed exactly with General Yankee Swap itself: MMS_i is the minimum
  utility in a leximin allocation among `n` copies of agent `i`.

## Verifying correctness

[`tests/test_bruteforce.py`](tests/test_bruteforce.py) runs the algorithm on
about 500 small random instances (1–5 agents, 3–7 items) drawn from five
families of matroid rank valuations (additive, capped, partition, transversal,
graphic) and their mixtures. It compares the output with **exhaustive search
over all (n+1)^m allocations**. For each instance and objective it checks that
the output

1. is a valid, nonredundant allocation;
2. maximizes utilitarian social welfare (Proposition 3.6);
3. is optimal for the objective (Theorems 6.2, 6.3, 6.4 and 6.6), with unit, integer and real weights;
4. is the **lexicographically largest** optimal utility vector (Theorem 3.4), which also pins down the exact tie-breaking;
5. gives every agent their maximin share, and the computed MMS values match brute force.

[`tests/test_api.py`](tests/test_api.py) covers the examples from the paper,
long transfer paths, input validation, and the valuation classes.

```bash
pytest          # 633 tests, about 7 seconds
```

The brute-force reference is also available as `yankee_swap.bruteforce`, so you
can check your own oracles and gain functions.

## Repository layout

```
src/yankee_swap/
    core.py          General Yankee Swap (Algorithms 1-3)
    objectives.py    gain functions (Table 1 / Section 6)
    mms.py           exact maximin shares
    valuations.py    matroid rank valuation oracles + validator
    generators.py    random instance generators
    bruteforce.py    exhaustive-search reference implementation
    allocation.py    Allocation result type
tests/               brute-force verification and unit tests
examples/            usage examples
```

## Citation

If you use this package, please cite the paper:

```bibtex
@article{viswanathan2025general,
  title   = {A General Framework for Fair Allocation under Matroid Rank Valuations},
  author  = {Viswanathan, Vignesh and Zick, Yair},
  journal = {ACM Transactions on Economics and Computation},
  volume  = {13},
  number  = {3},
  pages   = {14:1--14:32},
  year    = {2025},
  doi     = {10.1145/3728367}
}
```

## License

MIT, see [LICENSE](LICENSE).
