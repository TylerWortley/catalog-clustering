"""Ninth step in the cross-store product clustering pipeline

Turn scored pairs into product clusters

    1. Pairs with scores >= EDGE_THRESHOLD become edges in a graph
    2. Bounded union-find cuts graph into small work units
    3. Within each unit, score every pair ANN didn't retrieve (cheap)
    4. Hierarchical agglomerative clustering (HAC) within each unit decides final groups

Output: data/predictions_<split>.parquet

Usage: uv run python cluster.py
"""

import pickle

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from bounded_union_find import bounded_union_find
from features import make_features

OFFERS_PATH = "data/offers.parquet"
EMBEDDINGS_PATH = "data/embeddings.npy"
SCORED_PATH = "data/scored_pairs.parquet"
L1_PATH = "models/l1.pkl"

SPLIT = "test"
SCORE = "score"  # score: L1+L2, l1_score: L1
BASELINE_THRESHOLD = 0.5
EDGE_THRESHOLD = 0.3
BUDGET = 3000  # max sum of degrees in one work unit
MERGE_THRESHOLD = 0.3  # cutoff for HAC merging


def make_work_units(rows, units):
    """Group offers by union-find root"""
    return [np.sort(group.values) for _, group in pd.Series(rows).groupby(units[rows])]


def fill_in_scores(unit_members, pair_scores, offers, embeddings):
    """Use L1 to score every pair inside a work unit that ANN never retrieved"""
    missing = []
    for members in unit_members:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if (members[i], members[j]) not in pair_scores:
                    missing.append((members[i], members[j]))

    missing = pd.DataFrame(missing, columns=["left", "right"])

    with open(L1_PATH, "rb") as f:
        l1 = pickle.load(f)

    features = make_features(missing, offers, embeddings)
    missing["score"] = l1.predict_proba(features)[:, 1]
    pair_scores.update(zip(zip(missing["left"], missing["right"]), missing["score"]))
    print(f"Filled in {len(missing):,} pairs inside work units")


def cluster_unit(members, pair_scores):
    """Average-linkage HAC inside one work unit. Returns a label per member"""
    k = len(members)
    if k == 1:
        return np.array([1])

    # distance between two offers
    dist = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            dist[i, j] = dist[j, i] = 1 - pair_scores[(members[i], members[j])]

    # continue merging two closest clusters until theyre too far apart
    tree = linkage(squareform(dist), method="average")
    return fcluster(tree, t=1 - MERGE_THRESHOLD, criterion="distance")


def main():
    offers = pd.read_parquet(OFFERS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    pairs = pd.read_parquet(SCORED_PATH)
    pairs = pairs[pairs["split"] == SPLIT].copy()
    pairs["score"] = pairs[SCORE]
    edges = pairs[pairs["score"] >= EDGE_THRESHOLD]

    print(f"{SPLIT}: {len(pairs):,} scored pairs, {len(edges):,} edges")

    # degree = number of edges touching each node
    n_nodes = len(offers)
    degree = np.bincount(
        np.concatenate([edges["left"], edges["right"]]), minlength=n_nodes
    )

    # baseline: everything above threshold connected (no budget)
    baseline_edges = pairs[pairs["score"] >= BASELINE_THRESHOLD]
    baseline = bounded_union_find(n_nodes, baseline_edges, degree, np.inf)

    # work units (budget)
    units = bounded_union_find(n_nodes, edges, degree, BUDGET)
    rows = np.where(offers["split"] == SPLIT)[0]
    unit_members = make_work_units(rows, units)
    print(
        f"{len(unit_members):,} work units, largest has {max(len(m) for m in unit_members)} offers"
    )

    # ensure pairs skipped by ANN have a score (L1)
    pair_scores = dict(zip(zip(pairs["left"], pairs["right"]), pairs["score"]))
    fill_in_scores(unit_members, pair_scores, offers, embeddings)

    # Hierarchical Agglomerative Clustering (HAC) within each work unit
    cluster_ids = {}
    next_id = 0
    for members in unit_members:
        labels = cluster_unit(members, pair_scores)
        for member, label in zip(members, labels):
            cluster_ids[member] = next_id + label
        next_id += labels.max() + 1

    predictions = pd.DataFrame(
        {
            "row": rows,
            "true": offers["cluster_id"].values[rows],
            "baseline": baseline[rows],
            "bounded_hac": [cluster_ids[r] for r in rows],
        }
    )

    predictions.to_parquet(f"data/predictions_{SPLIT}.parquet", index=False)

    print(
        f"baseline: largest cluster has {predictions.groupby('baseline').size().max():,} offers"
    )
    print(
        f"bounded hac: largest has {predictions.groupby('bounded_hac').size().max():,} offers"
    )
    print(f"Saved data/predictions_{SPLIT}.parquet")


if __name__ == "__main__":
    main()
