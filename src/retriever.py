"""Find the best knowledge base record for a user's question.

How it works:
1. Every question and paraphrase in the knowledge base is turned into a vector.
2. A user question is turned into a vector too.
3. We compare the user vector with all stored vectors (cosine similarity).
4. The best-scoring record wins. If its score is below the threshold, we return the fallback.

Run a quick test from the project root:
    python -m src.retriever "hostel ki fee kitni hai"
"""

import sys

import numpy as np

from src.config import FALLBACK_MESSAGE, SIMILARITY_THRESHOLD, TOP_K
from src.data_loader import load_knowledge_base, validate_records
from src.embedder import encode


class Retriever:
    def __init__(self, records: list[dict]):
        self.records = records

        # Each text (question or paraphrase) gets its own vector.
        # owners[i] tells us which record vector i belongs to.
        texts, owners = [], []
        for record_index, record in enumerate(records):
            for text in [record["question"], *record["paraphrases"]]:
                texts.append(text)
                owners.append(record_index)

        self.owners = np.array(owners)
        self.vectors = encode(texts)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict:
        query_vector = encode([query])[0]
        scores = self.vectors @ query_vector  # cosine similarity, one score per text

        # Keep only the best score for each record, in order of best first.
        best_per_record: dict[int, float] = {}
        for text_index in np.argsort(-scores):
            record_index = int(self.owners[text_index])
            if record_index not in best_per_record:
                best_per_record[record_index] = float(scores[text_index])
            if len(best_per_record) >= top_k:
                break

        ranked = sorted(best_per_record.items(), key=lambda item: -item[1])
        matches = [
            {
                "id": self.records[record_index]["id"],
                "question": self.records[record_index]["question"],
                "score": round(score, 3),
            }
            for record_index, score in ranked
        ]

        best_index, best_score = ranked[0]
        if best_score < threshold:
            return {
                "answered": False,
                "answer": FALLBACK_MESSAGE,
                "score": round(best_score, 3),
                "matches": matches,
            }

        record = self.records[best_index]
        return {
            "answered": True,
            "answer": record["answer"],
            "source_url": record["source_url"],
            "academic_year": record["academic_year"],
            "matched_id": record["id"],
            "matched_question": record["question"],
            "score": round(best_score, 3),
            "matches": matches,
        }


def build_retriever() -> Retriever:
    """Load and validate the knowledge base, then build the search index."""
    kb = load_knowledge_base()
    errors = validate_records(kb["records"])
    if errors:
        raise ValueError("Knowledge base has errors:\n" + "\n".join(errors))
    return Retriever(kb["records"])


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Ask a question: ")
    result = build_retriever().search(question)

    print(f"\nQ: {question}")
    print(f"Answered: {result['answered']} (best score {result['score']})")
    print(f"A: {result['answer']}")
    if result["answered"]:
        print(f"Source: {result['source_url']}")
    print("Top matches:")
    for match in result["matches"]:
        print(f"  {match['score']:.3f}  {match['id']}  |  {match['question']}")
