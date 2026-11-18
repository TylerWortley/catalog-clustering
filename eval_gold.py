"""Final step in the cross-store product clustering pipeline

Evaluate on Web Data Commons (WDC) Products Dataset gold standard.

This dataset consists of 4,400 pairs checked by humans, with many
incorrect matches differing by a specific aspect (e.g. colour, year, type).
All data in this evaluation was held out of the training data.

Usage: uv run python eval_gold.py
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from features import make_features
from score_l2 import HIGH, LOW, load_l2
from train_l2 import OFFERS_PATH, offer_texts, pick_device, score_pairs

EMBEDDINGS_PATH = "data/embeddings.npy"
GOLD_PATH = "data/raw/all_gs.json.gz"
L1_PATH = "models/l1.pkl"


def load_gold_pairs(offers):
    """gold pairs as row numbers in offers.parquet plus human label"""
    gold = pd.read_json(GOLD_PATH, lines=True)
    row_of_id = pd.Series(offers.index, index=offers["id"])
    gold = gold[
        gold["id_left"].isin(row_of_id.index) & gold["id_right"].isin(row_of_id.index)
    ]
    a = row_of_id[gold["id_left"]].values
    b = row_of_id[gold["id_right"]].values

    return pd.DataFrame(
        {
            "left": np.minimum(a, b),
            "right": np.maximum(a, b),
            "label": gold["label"].values,
        }
    )


def best_cosine_threshold(pairs):
    """give best possible threshold"""
    best_threshold, best_f1 = 0, 0
    for threshold in np.arange(0.8, 1.0, 0.01):
        f1 = precision_recall_fscore_support(
            pairs["label"], pairs["cosine"] >= threshold, average="binary"
        )[2]

        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1

    return best_threshold


def main():
    device = pick_device()
    offers = pd.read_parquet(OFFERS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    pairs = load_gold_pairs(offers)

    print(f"{len(pairs):,} gold pairs ({pairs['label'].mean():.0%} matches)")

    # score with every model
    features = make_features(pairs, offers, embeddings)
    pairs["cosine"] = features["cosine"]

    with open(L1_PATH, "rb") as file:
        l1 = pickle.load(file)

    pairs["l1"] = l1.predict_proba(features)[:, 1]

    model, tokenizer = load_l2(device)
    pairs["l2"] = score_pairs(
        model,
        tokenizer,
        offer_texts(offers),
        pairs["left"].values,
        pairs["right"].values,
        device,
    )

    unsure = (pairs["l1"] > LOW) & (pairs["l1"] < HIGH)
    pairs["final"] = np.where(unsure, pairs["l2"], pairs["l1"])

    thresholds = {
        "cosine": best_cosine_threshold(pairs),
        "l1": 0.5,
        "l2": 0.5,
        "final": 0.5,
    }
    results = []
    for model_name, threshold in thresholds.items():
        predicted = pairs[model_name] >= threshold
        p, r, f1, _ = precision_recall_fscore_support(
            pairs["label"], predicted, average="binary"
        )
        results.append(
            {
                "model": model_name,
                "threshold": threshold,
                "precision": p,
                "recall": r,
                "f1": f1,
            }
        )

    print(pd.DataFrame(results).round(3).to_string(index=False))
    print(f"\nL2 used on {unsure.mean():.0%} of gold pairs")


if __name__ == "__main__":
    main()
