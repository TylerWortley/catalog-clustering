"""Second step in the cross-store product clustering pipeline

Turn every offer into an embedding. 

Offers for the same product should have similar vector alignment in embedding space.
Using multilingual E5 (same model used in Shopify's cross store product clustering blog article).
Runs on best available hardware: NVIDIA GPU ("cuda"), Apple GPU ("mps"), or CPU.

Usage: uv run python embed.py
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OFFERS_PATH = "data/offers.parquet"
OUT_PATH = "data/embeddings.npy"
MODEL_NAME = "intfloat/multilingual-e5-small"


def embed(texts):
    """Turn list of texts into normalized 384-num vectors"""
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding on:", model.device)

    model.max_seq_length = 64  # titles should be short
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )

    return embeddings.astype(np.float32)


def main():
    offers = pd.read_parquet(OFFERS_PATH)

    # info model reads; E5 expects "query: " prefix
    texts = ("query: " + offers["brand"] + " " + offers["title"]).tolist()
    embeddings = embed(texts)

    np.save(OUT_PATH, embeddings)
    print("Saved embeddings with shape", embeddings.shape)


if __name__ == "__main__":
    main()
