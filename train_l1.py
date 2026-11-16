"""Fifth step in the cross-store product clustering pipeline

Train the L1 Gradient Boosted Decision Tree on cheap pair features.

L1 is an inexpensive first pass that is fast enough to score every candidate pair:
- discard obvious non-matches
- accept only confident matches
- send uncertain pairs to slower, smarter downstream L2 model

Output: models/l1.pkl and data/scored_pairs.parquet (val + test pairs with an l1_score)

Usage: uv run python train_l1.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from features import make_features

OFFERS_PATH = "data/offers.parquet"
EMBEDDINGS_PATH = "data/embeddings.npy"
CANDIDATES_PATH = "data/candidates.parquet"
MODEL_PATH = "models/l1.pkl"
OUT_PATH = "data/scored_pairs.parquet"

N_TRAINED_PAIRS = 2_000_000
SEED = 42


def train_model(train, offers, embeddings):
    """Fit GBDT that predicts same product from the features"""
    print(f"Computing features for {len(train):,} training pairs")
    X = make_features(train, offers, embeddings)
    model = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
    model.fit(X, train["label"])
    return model


def add_l1_scores(model, pairs, offers, embeddings):
    """Add l1_score column (model's prediction)"""
    print(f"Computing features for {len(pairs):,} val/test pairs")
    X = make_features(pairs, offers, embeddings)
    pairs["l1_score"] = model.predict_proba(X)[:, 1]
    return pairs


def print_average_precision_score(scored):
    for split in ["val", "test"]:
        part = scored[scored["split"] == split]
        ap = average_precision_score(part["label"], part["l1_score"])
        ap_cosine = average_precision_score(part["label"], part["cosine"])
        print(
            f"{split}: average precision L1 = {ap:.3f} (cosine alone = {ap_cosine:.3f})"
        )


def main():
    offers = pd.read_parquet(OFFERS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    pairs = pd.read_parquet(CANDIDATES_PATH)

    train = pairs[pairs["split"] == "train"]
    train = train.sample(min(N_TRAINED_PAIRS, len(train)), random_state=SEED)
    model = train_model(train, offers, embeddings)

    scored = add_l1_scores(
        model, pairs[pairs["split"] != "train"].copy(), offers, embeddings
    )
    print_average_precision_score(scored)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    scored.to_parquet(OUT_PATH, index=False)
    print(f"Saved {MODEL_PATH} and {OUT_PATH}")


if __name__ == "__main__":
    main()
