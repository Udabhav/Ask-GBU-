"""Intent classifier: decide which topic a question is about.

Two models are trained on the same data so we can compare them:
1. Baseline: sentence embeddings + logistic regression (simple, fast).
2. DistilBERT: a small transformer fine-tuned on our questions (the deep learning model).

Run from the project root:
    python -m src.intent_model
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ROOT
from src.embedder import encode
from src.intent import LABELS, load_dataset, train_baseline

MODEL_DIR = ROOT / "models" / "intent_distilbert"
# Multilingual model: the dataset has English, Hinglish, and Devanagari Hindi.
# The English-only "uncased" model strips Hindi vowel signs, so it cannot read Hindi properly.
BERT_NAME = "distilbert-base-multilingual-cased"


def baseline_predictions(train_texts, train_labels, test_texts):
    """Embeddings + logistic regression. Returns predicted label ids for the test set."""
    clf = train_baseline(train_texts, train_labels)
    return clf.predict(encode(test_texts))


def train_bert(train_texts, train_labels, test_texts, epochs=8, lr=3e-5, batch_size=16):
    """Fine-tune DistilBERT. Returns predicted label ids for the test set."""
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_NAME, num_labels=len(LABELS)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_labels = np.array(train_labels)
    n = len(train_texts)

    model.train()
    for epoch in range(epochs):
        order = np.random.default_rng(epoch).permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = tokenizer(
                [train_texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            labels = torch.tensor(train_labels[idx])
            loss = model(**batch, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{epochs}  loss {total_loss / n:.4f}")

    model.eval()
    with torch.no_grad():
        batch = tokenizer(
            test_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        preds = model(**batch).logits.argmax(dim=-1).numpy()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    return preds


def report(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(
        classification_report(
            y_true, y_pred, target_names=LABELS, zero_division=0, digits=3
        )
    )


if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_dataset()
    print(f"Train: {len(train_texts)} examples, Test: {len(test_texts)} examples")

    baseline_preds = baseline_predictions(train_texts, train_labels, test_texts)
    report("Baseline: embeddings + logistic regression", test_labels, baseline_preds)

    print("\nTraining DistilBERT...")
    bert_preds = train_bert(train_texts, train_labels, test_texts)
    report("DistilBERT (fine-tuned)", test_labels, bert_preds)
    print(f"\nSaved DistilBERT model to {MODEL_DIR}")
