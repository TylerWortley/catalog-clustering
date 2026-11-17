"""Eighth step in the cross-store product clustering pipeline

Bounded union-find.

Typical union-find groups connected nodes into chains where each points towards
a single leader, and the leader points to itself. The bounded aspect goes through
edges from highest to lowest score and only merges groups if their combined cost
(sum of node degrees) <= budget.

Verify with: uv run python test_bounded_union_find.py
"""

import numpy as np


def find(parent, x):
    """Follow parent pointers from x until root and return the root"""
    while parent[x] != x:
        x = parent[x]
    return x


def bounded_union_find(n_nodes, edges, degree, budget):
    """
    n_nodes: number of nodes, named 0 .. n_nodes-1
    edges:   DataFrame with columns left, right, score
    degree:  degree[i] = number of edges touching node i
    budget:  max sum of degrees allowed in one group (np.inf = no limit)

    Returns an array comp where comp[i] = root of node i's group.
    """
    parent = np.arange(n_nodes)
    used = degree.astype(np.int64).copy()

    edges = edges.sort_values("score", ascending=False)

    for a, b in zip(edges["left"].values, edges["right"].values):
        root_a = find(parent, a)
        root_b = find(parent, b)

        # see if both already belong to same group or if merging would go over budget
        if root_a == root_b:
            continue
        if used[root_a] + used[root_b] > budget:
            continue

        # merge: point b's root at a's root and accumulate cost
        parent[root_b] = root_a
        used[root_a] += used[root_b]

    return np.array([find(parent, i) for i in range(n_nodes)])
