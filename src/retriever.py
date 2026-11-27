"""Find the best knowledge base record for a user's question.

How it works:
1. Every question and paraphrase in the knowledge base is turned into a vector.
2. A user question is turned into a vector too.
3. The intent classifier predicts the topic (fees, hostel, ...).
   - If it is sure the question is "other" (not about GBU), we return the fallback.
   - If it is sure about a GBU topic, we search only records of that topic.
   - If it is not sure, we search all records.
4. We compare the user vector with the stored vectors (cosine similarity).
5. The best-scoring record wins. If its score is below the threshold, we return the fallback.

Run a quick test from the project root:
    python -m src.retriever "hostel ki fee kitni hai"
"""

import sys

import numpy as np

from src.config import (
    EMBEDDING_MODEL,
    FALLBACK_MESSAGE,
    INTENT_CONFIDENCE,
    OTHER_CONFIDENCE,
    SIMILARITY_THRESHOLD,
    TOP_K,
    USE_INTENT_FILTER,
)
from src.data_loader import load_knowledge_base, validate_records
from src.embedder import encode
from src.intent import IntentClassifier


class Retriever:
    def __init__(
        self,
        records: list[dict],
        intent_classifier: IntentClassifier | None = None,
        model_name: str = EMBEDDING_MODEL,
    ):
        if intent_classifier is not None and intent_classifier.model_name != model_name:
            raise ValueError("The intent classifier and the retriever must use the same embedding model.")
        self.records = records
        self.intent_classifier = intent_classifier
        self.model_name = model_name

        # Each text (question or paraphrase) gets its own vector.
        # owners[i] tells us which record vector i belongs to.
        texts, owners = [], []
        for record_index, record in enumerate(records):
            for text in [record["question"], *record["paraphrases"]]:
                texts.append(text)
                owners.append(record_index)

        self.owners = np.array(owners)
        self.vectors = encode(texts, model_name)
        # Topics of each stored text, used to search only inside one topic.
        # A record can belong to extra topics through "also_topics"
        # (for example, the hostel office contact is in both "contacts" and "hostel").
        self.text_topics = [
            {records[i]["topic"], *records[i].get("also_topics", [])} for i in owners
        ]

    def _rank(self, scores: np.ndarray, allowed: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Best score per record, best first, looking only at allowed texts."""
        candidates = np.where(allowed)[0]
        best_per_record: dict[int, float] = {}
        for text_index in candidates[np.argsort(-scores[candidates])]:
            record_index = int(self.owners[text_index])
            if record_index not in best_per_record:
                best_per_record[record_index] = float(scores[text_index])
            if len(best_per_record) >= top_k:
                break
        return sorted(best_per_record.items(), key=lambda item: -item[1])

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        use_intent: bool = USE_INTENT_FILTER,
    ) -> dict:
        query_vector = encode([query], self.model_name)[0]
        scores = self.vectors @ query_vector  # cosine similarity, one score per text

        intent, confidence = None, None
        if use_intent and self.intent_classifier is not None:
            intent, confidence = self.intent_classifier.predict_vector(query_vector)

        # Decide which stored texts we are allowed to match.
        allowed = np.ones(len(scores), dtype=bool)
        topic_filter = None
        if intent and intent != "other" and confidence >= INTENT_CONFIDENCE:
            in_topic = np.array([intent in topics for topics in self.text_topics])
            if in_topic.any():
                allowed = in_topic
                topic_filter = intent

        ranked = self._rank(scores, allowed, top_k)
        matches = [
            {
                "id": self.records[record_index]["id"],
                "topics": [
                    self.records[record_index]["topic"],
                    *self.records[record_index].get("also_topics", []),
                ],
                "question": self.records[record_index]["question"],
                "score": score,
            }
            for record_index, score in ranked
        ]
        best_index, best_score = ranked[0]

        result = {
            "intent": intent,
            "intent_confidence": None if confidence is None else round(confidence, 3),
            "topic_filter": topic_filter,
            # Not rounded: the threshold check below and the evaluation must see the same value.
            "score": best_score,
            "matches": matches,
        }

        out_of_scope = intent == "other" and confidence >= OTHER_CONFIDENCE
        if out_of_scope or best_score < threshold:
            return {
                **result,
                "answered": False,
                "answer": FALLBACK_MESSAGE,
                "reason": "out_of_scope" if out_of_scope else "low_score",
            }

        record = self.records[best_index]
        return {
            **result,
            "answered": True,
            "answer": record["answer"],
            "source_url": record["source_url"],
            "academic_year": record["academic_year"],
            "matched_id": record["id"],
            "matched_question": record["question"],
        }


def build_retriever(
    use_intent: bool = USE_INTENT_FILTER, model_name: str = EMBEDDING_MODEL
) -> Retriever:
    """Load and validate the knowledge base, then build the search index."""
    kb = load_knowledge_base()
    errors = validate_records(kb["records"])
    if errors:
        raise ValueError("Knowledge base has errors:\n" + "\n".join(errors))
    classifier = IntentClassifier(model_name) if use_intent else None
    return Retriever(kb["records"], intent_classifier=classifier, model_name=model_name)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Ask a question: ")
    result = build_retriever().search(question)

    print(f"\nQ: {question}")
    if result["intent"]:
        print(
            f"Intent: {result['intent']} (confidence {result['intent_confidence']}), "
            f"searched in: {result['topic_filter'] or 'all topics'}"
        )
    print(f"Answered: {result['answered']} (best score {result['score']:.3f})")
    print(f"A: {result['answer']}")
    if result["answered"]:
        print(f"Source: {result['source_url']}")
    print("Top matches:")
    for match in result["matches"]:
        print(f"  {match['score']:.3f}  {match['id']}  |  {match['question']}")
