"""Sixth step in the cross-store product clustering pipeline

Fine-tune multilingual E5 cross-encoder.

L1 only sees a few numbers about a pair, a cross encoder reads both listings together
word by word. This helps catch more subtle differences, but is much slower than L1 so
it only scores pairs L1 is unsure about.

Output: models/l2/ (fine-tuned model)

Usage: uv run python train_l2.py
"""

import time

import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

OFFERS_PATH = "data/offers.parquet"
CANDIDATES_PATH = "data/candidates.parquet"
SCORED_PATH = "data/scored_pairs.parquet"
MODEL_DIR = "models/l2"

MODEL_NAME = "intfloat/multilingual-e5-small"
N_TRAIN_PAIRS = 400_000
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
MAX_LENGTH = 128  # max tokens for two listings
SEED = 42


def pick_device():
    """Select best available hardware"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def offer_texts(offers):
    """Input for the model: brand title | category | shop"""
    return (
        offers["brand"]
        + " "
        + offers["title"]
        + " | "
        + offers["category"].str.replace("_", " ")
        + " | "
        + offers["shop"]
    ).values


def tokenize(tokenizer, texts, left, right, device):
    """Turn pairs of offers into model inputs: both texts joined into one sequence"""
    return tokenizer(
        list(texts[left]),
        list(texts[right]),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)


def train_model(train, texts, device):
    """Fine-tune E5 model with one pass over training pairs"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    start = time.time()
    n_batches = len(train) // BATCH_SIZE

    for batch_num in range(n_batches):
        batch = train.iloc[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        inputs = tokenize(
            tokenizer, texts, batch["left"].values, batch["right"].values, device
        )
        labels = torch.tensor(batch["label"].values, dtype=torch.float32, device=device)

        logits = model(**inputs).logits.squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_num % 500 == 0:
            minutes = (time.time() - start) / 60
            print(
                f"  batch {batch_num:,}/{n_batches:,}  loss {loss.item():.3f}  ({minutes:.1f} min)",
                flush=True,
            )
        if batch_num % 2000 == 0:
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

    return model, tokenizer


def score_pairs(model, tokenizer, texts, left, right, device, batch_size=256):
    """Return the match probability (0 to 1) for each pair of offer rows"""
    model.eval()
    scores = []

    with torch.no_grad():
        for i in range(0, len(left), batch_size):
            inputs = tokenize(
                tokenizer,
                texts,
                left[i : i + batch_size],
                right[i : i + batch_size],
                device,
            )
            logits = model(**inputs).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())

    return scores


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"Using: {device} as device")

    offers = pd.read_parquet(OFFERS_PATH)
    pairs = pd.read_parquet(CANDIDATES_PATH)
    texts = offer_texts(offers)

    # random sample of training candidate pairs from ANN search
    train = pairs[pairs["split"] == "train"].sample(N_TRAIN_PAIRS, random_state=SEED)
    print(f"Training on {len(train):,} pairs ({train['label'].mean():.1%} matches)")
    model, tokenizer = train_model(train, texts, device)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved {MODEL_DIR}")

    # check L1 vs L2 on 20k validation pairs
    val = pd.read_parquet(SCORED_PATH)
    val = val[val["split"] == "val"].sample(20_000, random_state=SEED)
    val["l2_score"] = score_pairs(
        model, tokenizer, texts, val["left"].values, val["right"].values, device
    )
    print(
        f"val average precision:  L1 = {average_precision_score(val['label'], val['l1_score']):.3f}"
        f"   L2 = {average_precision_score(val['label'], val['l2_score']):.3f}"
    )


if __name__ == "__main__":
    main()
