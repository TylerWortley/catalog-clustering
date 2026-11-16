"""Third step in the cross-store product clustering pipeline

Candidate generation with approximate nearest neighbours (ANN) search.

For every offer, find its K most similar offers in the same split.
Only these pairs will be scored later - all pairs would be impossible due to scale.

Output: data/candidates.parquet
    left, right -> row numbers in offers.parquet
    cosine -> embedding cosine similarity (1 = identical direction)
    rank -> neighbour position (1 = closest)
    label -> 1 if same product (cluster_id), otherwise 0
    split -> train / val / test

Usage: uv run python candidates.py
"""

import faiss
import numpy as np
import pandas as pd

OFFERS_PATH = "data/offers.parquet"
EMBEDDINGS_PATH = "data/embeddings.npy"
OUT_PATH = "data/candidates.parquet"
K = 20  # neighbours per offer


def find_neighbours(vectors):
    """Find K most similar vectors (plus itself) for each vector"""
    # HNSW - graph based ANN commonly used in vector databases and high-dimensional search applications
    index = faiss.IndexHNSWFlat(
        vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT
    )  # 32 = links per node
    index.hnsw.efSearch = 64  # how hard search looks
    index.add(vectors)

    return index.search(vectors, K + 1)


def to_pairs(rows, scores, neighbours):
    """Flatten (offers * K+1) search result into one row per unique pair"""
    flat = neighbours.ravel()
    found = flat >= 0  # -1 returned for no neighbour found
    a = np.repeat(rows, K + 1)[found]
    b = rows[flat[found]]

    pairs = pd.DataFrame(
        {
            "left": np.minimum(a, b),
            "right": np.maximum(a, b),
            "cosine": scores.ravel()[found],
            "rank": np.tile(np.arange(K + 1), len(rows))[found],
        }
    )

    pairs = pairs[pairs["left"] != pairs["right"]]
    return pairs.sort_values("rank").drop_duplicates(["left", "right"])


def print_recall(pairs, offers, rows, split):
    """Number of true matches search found"""
    group_sizes = offers.loc[rows, "cluster_id"].value_counts()
    n_true_pairs = (group_sizes * (group_sizes - 1) // 2).sum()
    print(
        f"\n{split}: {len(rows):,} offers -> {len(pairs):,} candidate pairs "
        f"({pairs['label'].mean():.1%} are true matches)"
    )

    # Recall@k for various k
    for k in [1, 5, 10, 20]:
        recall = pairs.loc[pairs["rank"] <= k, "label"].sum() / n_true_pairs
        print(f"  recall@{k:<2}: {recall:.1%}")

    # pair recall alternative: of offers with a match, how many found at least one in their top K
    group_size = offers.loc[rows, "cluster_id"].map(group_sizes).values
    offers_match = rows[group_size > 1]
    true_pairs = pairs[pairs["label"] == 1]
    found_match = np.union1d(true_pairs["left"], true_pairs["right"])
    hit_rate = np.isin(offers_match, found_match).mean()

    print(f"  Offers that found >= 1 true match: {hit_rate:.1%}")


def main():
    offers = pd.read_parquet(OFFERS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    cluster = offers["cluster_id"].values

    all_pairs = []

    for split in ["train", "val", "test"]:
        rows = np.where(offers["split"] == split)[0]
        scores, neighbours = find_neighbours(embeddings[rows])
        pairs = to_pairs(rows, scores, neighbours)
        pairs["label"] = (cluster[pairs["left"]] == cluster[pairs["right"]]).astype(int)
        pairs["split"] = split
        print_recall(pairs, offers, rows, split)
        all_pairs.append(pairs)

    pd.concat(all_pairs).to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
