"""Load and validate the Ask GBU knowledge base.

Run from the project root:
    python src/data_loader.py
"""

import json
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

REQUIRED_FIELDS = [
    "id",
    "topic",
    "question",
    "paraphrases",
    "answer",
    "source_url",
    "academic_year",
    "last_checked",
    "verification",
]

ALLOWED_TOPICS = {
    "fees",
    "hostel",
    "admissions",
    "contacts",
    "scholarships",
    "courses",
    "exams",
    "campus",
}


def load_knowledge_base(path: Path = KB_PATH) -> dict:
    """Read the JSON file and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_records(records: list) -> list:
    """Return a list of error messages. An empty list means all records are valid."""
    errors = []
    seen_ids = set()

    for index, record in enumerate(records):
        label = record.get("id", f"record #{index + 1}")

        for field in REQUIRED_FIELDS:
            if field not in record or record[field] in ("", [], None):
                errors.append(f"{label}: missing or empty field '{field}'")

        record_id = record.get("id")
        if record_id in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(record_id)

        topic = record.get("topic")
        if topic and topic not in ALLOWED_TOPICS:
            errors.append(f"{label}: unknown topic '{topic}'")

        url = record.get("source_url", "")
        if url and not url.startswith("https://"):
            errors.append(f"{label}: source_url must start with https://")

    return errors


def main() -> None:
    kb = load_knowledge_base()
    records = kb["records"]
    errors = validate_records(records)

    print(f"Loaded {len(records)} records from {KB_PATH.name}")
    if errors:
        print(f"Found {len(errors)} problem(s):")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("All records are valid.")


if __name__ == "__main__":
    main()
