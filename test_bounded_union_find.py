"""Test for bounded_union_find.py, verifiable by hand.

Graph (a chain):             0--[0.9]-- 1 --[0.8]-- 2 --[0.3]-- 3      4
Degrees (# edges touching): 0:1        1:2         2:2         3:1     4:0

Usage: uv run python test_bounded_union_find.py
"""

import numpy as np
import pandas as pd

from bounded_union_find import bounded_union_find

EDGES = pd.DataFrame({"left": [0, 1, 2], "right": [1, 2, 3], "score": [0.9, 0.8, 0.3]})
DEGREE = np.array([1, 2, 2, 1, 0])


def groups(comp):
    """Turn comp into a set of groups"""
    result = {}
    for node, root in enumerate(comp):
        result.setdefault(root, []).append(node)
    return {tuple(group) for group in result.values()}


def test_no_budget():
    """No budget: everything connected ends up together (connected components)"""
    comp = bounded_union_find(5, EDGES, DEGREE, np.inf)
    assert groups(comp) == {(0, 1, 2, 3), (4,)}


def test_budget_5():
    """Block going beyond the bounded budget of 5 total degrees in a group"""
    comp = bounded_union_find(5, EDGES, DEGREE, 5)
    assert groups(comp) == {(0, 1, 2), (3,), (4,)}


def test_budget_3():
    """Block going beyond the bounded budget of 3 total degrees in a group"""
    comp = bounded_union_find(5, EDGES, DEGREE, 3)
    assert groups(comp) == {(0, 1), (2, 3), (4,)}


def test_edge_order():
    """Highest score goes first, regardless of order"""
    backward = EDGES.sort_values("score")
    comp = bounded_union_find(5, backward, DEGREE, 5)
    assert groups(comp) == {(0, 1, 2), (3,), (4,)}


def main():
    test_no_budget()
    test_budget_5()
    test_budget_3()
    test_edge_order()
    print("All bounded union-find tests passed")


if __name__ == "__main__":
    main()
