"""Fourth step in the cross-store product clustering pipeline

Inexpensive features for a pair of offers.

Each features is a number that can be used to indicate if two products are the same.
The L1 model in the following step will learn how to combine them.

Usage: uv run python features.py
"""

import re
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def title_overlap(title_a, title_b):
    """Share of words two titles have in common from 0 to 1"""

    a_words = set(title_a.lower().split())
    b_words = set(title_b.lower().split())

    if not a_words or not b_words:
        return 0.0

    shared = a_words & b_words
    combined = a_words | b_words

    return len(shared) / len(combined)


def model_number(title):
    """Retrieve words that look like a model number (contain both numbers and letters)"""

    codes = set()
    for word in title.lower().split():
        word = re.sub(r"[^a-z0-9]", "", word)  # drop punctuation
        has_letter = any(c.isalpha() for c in word)
        has_number = any(c.isdigit() for c in word)
        if has_letter and has_number:
            codes.add(word)

    return codes


def model_number_match(title_a, title_b):
    """Find model match in titles. 1 if model numbers match, zero if both have model number but no match, -1 if either has none"""
    codes_a = model_number(title_a)
    codes_b = model_number(title_b)

    if not codes_a or not codes_b:
        return -1
    if codes_a & codes_b:
        return 1
    else:
        return 0


def brand_match(brand_a, brand_b):
    """Check brand match. 1 if brands are the same, 0 if different, -1 if either missing"""
    if not brand_a or not brand_b:
        return -1
    if brand_a.lower().strip() == brand_b.lower().strip():
        return 1
    else:
        return 0


def length_ratio(title_a, title_b):
    """Ratio of title lengths from 0 to 1"""
    return min(len(title_a), len(title_b)) / max(len(title_a), len(title_b))


def number_overlap(title_a, title_b):
    """Jaccard overlap of plain numbers in each title, -1 if either has none"""
    nums_a = set(re.findall(r"\b\d+\b", title_a))
    nums_b = set(re.findall(r"\b\d+\b", title_b))

    if not nums_a or not nums_b:
        return -1

    return len(nums_a & nums_b) / len(nums_a | nums_b)


def make_features(pairs, offers, embeddings):
    """One row of features per pair"""

    titles = offers["title"].values
    brands = offers["brand"].values
    shops = offers["shop"].values
    categories = offers["category"].values

    rows = []

    for a, b in zip(pairs["left"].values, pairs["right"].values):
        rows.append(
            {
                "title_overlap": title_overlap(titles[a], titles[b]),
                "model_number_match": model_number_match(titles[a], titles[b]),
                "brand_match": brand_match(brands[a], brands[b]),
                "same_shop": int(shops[a] == shops[b]),
                "same_category": int(categories[a] == categories[b]),
                "length_ratio": length_ratio(titles[a], titles[b]),
                "fuzzy": fuzz.token_set_ratio(titles[a], titles[b]) / 100,
                "number_overlap": number_overlap(titles[a], titles[b]),
            }
        )

    features = pd.DataFrame(rows)

    left_vectors = embeddings[pairs["left"].values]
    right_vectors = embeddings[pairs["right"].values]
    features["cosine"] = np.sum(left_vectors * right_vectors, axis=1)

    return features


def check_example():
    """Hand-checkable pairs: does each function say what you'd expect?"""
    examples = [
        # (title A, title B, brand A, brand B)
        ("Apple iPhone 13 128GB Blue", "iPhone 13 (128 GB) - Blue", "Apple", "apple"),
        ("Apple iPhone 13 128GB Blue", "Apple iPhone 13 256GB Blue", "Apple", "Apple"),
        (
            "Raymond Weil Freelancer 2740-ST-20021",
            "Raymond Weil Freelancer 2740-STP-65021",
            "Raymond Weil",
            "",
        ),
        (
            "WD Red 4TB NAS Hard Drive WD40EFRX",
            "Western Digital WD40EFRX 4TB Red",
            "WD",
            "Western Digital",
        ),
    ]
    for title_a, title_b, brand_a, brand_b in examples:
        print(f"{title_a}  vs  {title_b}")
        print(
            f"  title_overlap={title_overlap(title_a, title_b):.2f}"
            f"  model_number_match={model_number_match(title_a, title_b)}"
            f"  brand_match={brand_match(brand_a, brand_b)}"
        )


def check_speed(n_pairs=100_000):
    """Time make_features on real pairs. Step 5 runs it on ~6.6M pairs."""
    offers = pd.read_parquet("data/offers.parquet")
    embeddings = np.load(
        "data/embeddings.npy", mmap_mode="r"
    )  # read rows from disk as needed
    pairs = pd.read_parquet("data/candidates.parquet").sample(n_pairs, random_state=0)

    start = time.time()
    features = make_features(pairs, offers, embeddings)
    seconds = time.time() - start
    print(
        f"\n{n_pairs:,} pairs in {seconds:.1f}s -> about {seconds * 6.6e6 / n_pairs / 60:.0f} min in step 5"
    )

    # A useful feature has a clearly different average for matches (1) vs non-matches (0)
    print("\nAverage feature value by label:")
    print(features.groupby(pairs["label"].values).mean().round(2).T.to_string())


def main():
    check_example()
    check_speed()


if __name__ == "__main__":
    main()
