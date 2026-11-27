"""Intent classifier used by the chatbot at runtime.

It predicts the topic of a question (fees, hostel, admissions, contacts,
scholarships, or other) so the retriever can search only the matching records.

Model: sentence embeddings + logistic regression (the baseline from
src/intent_model.py, which scored best on the test set). It trains in a few
seconds on the "train" split of data/intent_dataset.csv when the app starts.
"""

import csv

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import EMBEDDING_MODEL, ROOT
from src.embedder import encode

DATA_PATH = ROOT / "data" / "intent_dataset.csv"

LABELS = ["admissions", "contacts", "fees", "hostel", "other", "scholarships"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load_dataset():
    """Return (train_texts, train_labels, test_texts, test_labels). Labels are integers."""
    splits = {"train": ([], []), "test": ([], [])}
    with open(DATA_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            texts, labels = splits[row["split"]]
            texts.append(row["text"].strip())
            labels.append(LABEL2ID[row["label"]])
    return (*splits["train"], *splits["test"])


def train_baseline(train_texts, train_labels, model_name: str = EMBEDDING_MODEL) -> LogisticRegression:
    """Fit logistic regression on sentence embeddings."""
    clf = LogisticRegression(max_iter=2000, C=5.0)
    clf.fit(encode(train_texts, model_name), train_labels)
    return clf


class IntentClassifier:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # The classifier reads vectors, so it must use the same embedding model as the retriever.
        self.model_name = model_name
        # Train only on the "train" split, so the test split stays unseen for evaluation.
        train_texts, train_labels, _, _ = load_dataset()
        self.clf = train_baseline(train_texts, train_labels, model_name)

    def predict_vector(self, vector: np.ndarray) -> tuple[str, float]:
        """Predict from an already-encoded question. Returns (topic, confidence 0 to 1)."""
        probs = self.clf.predict_proba(vector.reshape(1, -1))[0]
        best = int(probs.argmax())
        return LABELS[int(self.clf.classes_[best])], float(probs[best])

    def predict(self, text: str) -> tuple[str, float]:
        return self.predict_vector(encode([text], self.model_name)[0])
