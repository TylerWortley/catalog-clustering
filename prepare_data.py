"""First step in the cross-store product clustering pipeline

Reads raw WDC files and write a single table: data/offers.parquet
Each row is one offer (a shop's listing of a product).

    1. Load every offer in English (only the needed columns)
    2. Look up which shop each offer came from
    3. Keep product groups sold by 2 or more different shops (cross-store matches)
    4. Clean messy titles and brand text
    5. Split product groups into train / val /test sets

    Usage: uv run python prepare_data.py
"""

import gzip
import json
import re

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/offers.parquet"

MAX_GROUP_SIZE = 50
N_MATCH_OFFERS = 800_000
N_SINGLE_OFFERS = 200_000
SEED = 42

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2


def clean_text(text):
    """Clean leftover artifacts of WDC dataset"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'"@[a-zA-Z-]+', " ", text)  # replace language tags with spaces
    text = text.replace('"', " ")
    text = re.sub(r"\s+", " ", text)  # collapse repeated spaces and/or newlines

    return text.strip(" ,")


def load_offers():
    """Load all English offers, keeping only the columns are needed"""

    print("Loading offers")
    ids, group_ids, titles, brands, categories = [], [], [], [], []

    # extract relevant data for each offer
    with gzip.open(
        f"{RAW_DIR}/offers_corpus_english_v2_non_norm.json.gz", "rt"
    ) as file:
        for line in file:
            offer = json.loads(line)
            ids.append(offer["id"])
            group_ids.append(offer["cluster_id"])
            titles.append(offer["title"])
            brands.append(offer["brand"])
            categories.append(offer["category"])

    # convert to pandas dataframe
    offers = pd.DataFrame(
        {
            "id": ids,
            "cluster_id": group_ids,
            "title": titles,
            "brand": brands,
            "category": categories,
        }
    )

    offers = offers[offers["title"].notna()]
    print(
        f"{len(offers):,} offers in {offers['cluster_id'].nunique():,} product groups"
    )

    return offers


def add_shops(offers):
    """Add a column corresponding to the shop the offer came from"""

    print("Loading offer URLs")
    urls = pd.read_csv(f"{RAW_DIR}/id_url_mapping.csv.gz")
    urls = urls[urls["id"].isin(offers["id"])]

    # condense full link into just {website}.com
    urls["shop"] = urls["url"].str.split("/").str[2].str.removeprefix("www.")
    offers = offers.merge(urls[["id", "shop"]], on="id", how="left")
    print(f"Offers come from {offers['shop'].nunique():,} shops")

    return offers


def load_gold_groups():
    """Obtain ground-truth labels from the hand labelled WDC pairs"""
    gold = pd.read_json(f"{RAW_DIR}/all_gs.json.gz", lines=True)
    return set(gold["cluster_id_left"]) | set(gold["cluster_id_right"])


def choose_groups(offers, gold_groups):
    """Keep cross store groups, single offers, and gold groups"""
    group_stats = offers.groupby("cluster_id").agg(
        n_offers=("id", "size"), n_shops=("shop", "nunique")
    )

    # cross store groups
    match = group_stats[
        (group_stats["n_shops"] >= 2) & (group_stats["n_offers"] <= MAX_GROUP_SIZE)
    ]
    match = match.sample(frac=1, random_state=SEED)  # shuffle
    match = match[match["n_offers"].cumsum() <= N_MATCH_OFFERS]

    singles = group_stats[group_stats["n_offers"] == 1].sample(
        N_SINGLE_OFFERS, random_state=SEED
    )

    keep = set(match.index) | set(singles.index) | gold_groups

    return offers[offers["cluster_id"].isin(keep)].copy()


def clean_offers(offers):
    """Clean title and brand text, drop offers with no title"""
    offers["title"] = offers["title"].map(clean_text)
    offers["brand"] = offers["brand"].map(clean_text)
    offers["shop"] = offers["shop"].fillna("unknown")

    return offers[offers["title"] != ""]


def split_by_group(offers, gold_groups):
    """Give each product group one split - test set should only have unseen products"""
    rng = np.random.default_rng(SEED)
    groups = offers["cluster_id"].unique()
    group_split = rng.choice(
        ["train", "val", "test"],
        size=len(groups),
        p=[TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
    )
    offers["split"] = offers["cluster_id"].map(dict(zip(groups, group_split)))

    # hold out ground truth labels from training
    offers.loc[offers["cluster_id"].isin(gold_groups), "split"] = "test"
    return offers.reset_index(drop=True)


def print_summary(offers):
    print(f"Product groups: {offers['cluster_id'].nunique():,}")
    print(f"Shops: {offers['shop'].nunique():,}")
    print(
        offers.groupby("split").agg(
            offers=("id", "size"), groups=("cluster_id", "nunique")
        )
    )
    print("\nExample rows:")
    print(offers[["cluster_id", "shop", "title"]].head(10).to_string())


def main():
    offers = load_offers()
    offers = add_shops(offers)
    gold_groups = load_gold_groups()
    offers = choose_groups(offers, gold_groups)
    offers = clean_offers(offers)
    offers = split_by_group(offers, gold_groups)

    offers.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(offers):,} offers to {OUT_PATH}")

    print_summary(offers)


if __name__ == "__main__":
    main()
