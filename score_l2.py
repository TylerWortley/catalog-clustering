"""Seventh step in the cross-store product clustering pipeline

Use L2 to re-score the pairs that L1 was unsure about.

Adds two columns to data/scored_pairs.parquet:
    1. l2_score -> L2's match probability
    2. score -> final score (L2 if we compute it, otherwise L1)

Usage: uv run python score_l2.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_l2 import (
    MODEL_DIR,
    OFFERS_PATH,
    SCORED_PATH,
    offer_texts,
    pick_device,
    score_pairs,
)

LOW, HIGH = 0.05, 0.95  # L2 uses L1 scores within this range


def load_l2(device):
    """Load fine-tuned cross-encoded saved by train_l2.py"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    return model, tokenizer


def print_average_precision(pairs):
    """Compare final L1/L2 vs L1"""
    for split in ["val", "test"]:
        part = pairs[pairs["split"] == split]
        ap_l1 = average_precision_score(part["label"], part["l1_score"])
        ap_final = average_precision_score(part["label"], part["score"])
        print(f"{split}: average precision L1 = {ap_l1:.3f}, final = {ap_final:.3f}")


def main():
    device = pick_device()
    print(f"Using: {device} as device")

    offers = pd.read_parquet(OFFERS_PATH)
    pairs = pd.read_parquet(SCORED_PATH)
    texts = offer_texts(offers)

    unsure = (pairs["l1_score"] > LOW) & (pairs["l1_score"] < HIGH)
    print(
        f"{unsure.sum():,} of {len(pairs):,} pairs ({unsure.mean():.0%} are sent to L2)"
    )

    model, tokenizer = load_l2(device)
    pairs["l2_score"] = np.nan
    pairs.loc[unsure, "l2_score"] = score_pairs(
        model,
        tokenizer,
        texts,
        pairs.loc[unsure, "left"].values,
        pairs.loc[unsure, "right"].values,
        device,
    )
    pairs["score"] = pairs["l2_score"].fillna(pairs["l1_score"])
    print_average_precision(pairs)

    pairs.to_parquet(SCORED_PATH, index=False)
    print(f"Saved {SCORED_PATH}")


if __name__ == "__main__":
    main()
