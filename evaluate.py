"""Tenth step in the cross-store product clustering pipeline

Evaluate the quality of the clusters.

Pairwise: look at every pair of offers that share a predicted cluster
    precision = share of pairs that really are the same product tp/(tp+fp)
    recall = share of all true same-product pairs that were actually clustered

B-cubed: score each offer, then average over offers
    precision(offer) = share of its predicted cluster that is truly the product
    recall(offer) = share of its true product group that landed in its predicted cluster

Usage: uv run python evaluate.py
"""

import pandas as pd

SPLIT = "test"
METHODS = ["baseline", "bounded_hac"]


def count_pairs(sizes):
    """cluster with n offers contains n*(n-1) / 2 pairs, sum over clusters"""
    return int((sizes * (sizes - 1) // 2).sum())


def f1_score(precision, recall):
    """compute f1 score (balance of precision and recall)"""
    return 2 * precision * recall / (precision + recall)


def pairwise_scores(true, pred):
    """compute pairwise metrics"""
    df = pd.DataFrame({"true": true, "pred": pred})

    correct_pairs = count_pairs(df.groupby(["true", "pred"]).size())
    pred_pairs = count_pairs(df.groupby("pred").size())
    true_pairs = count_pairs(df.groupby("true").size())

    precision = correct_pairs / pred_pairs
    recall = correct_pairs / true_pairs
    f1 = f1_score(precision, recall)
    wrong_pairs = pred_pairs - correct_pairs

    return precision, recall, f1, wrong_pairs


def bcubed_scores(true, pred):
    """compute b-cubed metrics"""
    df = pd.DataFrame({"true": true, "pred": pred})

    overlap = df.groupby(["true", "pred"])["true"].transform("size")
    pred_size = df.groupby("pred")["pred"].transform("size")
    true_size = df.groupby("true")["true"].transform("size")

    precision = (overlap / pred_size).mean()
    recall = (overlap / true_size).mean()
    f1 = f1_score(precision, recall)

    return precision, recall, f1


def main():
    predictions = pd.read_parquet(f"data/predictions_{SPLIT}.parquet")

    results = []
    for method in METHODS:
        p, r, f1, wrong = pairwise_scores(predictions["true"], predictions[method])
        bp, br, bf1 = bcubed_scores(predictions["true"], predictions[method])
        results.append(
            {
                "method": method,
                "pair_precision": p,
                "pair_recall": r,
                "pair_f1": f1,
                "wrong_pairs": wrong,
                "bcubed_precision": bp,
                "bcubed_recall": br,
                "bcubed_f1": bf1,
                "largest_cluster": predictions.groupby(method).size().max(),
            }
        )
    print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
