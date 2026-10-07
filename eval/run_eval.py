"""Evaluate the advisor bot against eval/questions.csv.

questions.csv columns:
  question         - what a student might ask
  expected_source  - part of the file name that should be retrieved; several
                     acceptable files can be separated with "|"
  expected_answer  - the correct answer, for grading by hand (may be blank)

Always reports the retrieval hit rate: how often an expected file is among
the TOP_K chunks the LLM would see. With an API key in .env (and without
--retrieval-only), it also generates an answer for every question and saves
them next to the expected answers in eval/results.csv for you to grade.

Run from the project folder:
  python eval/run_eval.py                    (retrieval + answers)
  python eval/run_eval.py --retrieval-only   (free; no API calls)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import rag" works
import config  # noqa: E402
import rag  # noqa: E402

QUESTIONS = config.EVAL_DIR / "questions.csv"
RESULTS = config.EVAL_DIR / "results.csv"


def found_at(hits: list[dict], expected: str) -> int | None:
    """Rank (1 = best) of the first hit from an expected file, or None."""
    wanted = [w.strip().lower() for w in expected.split("|") if w.strip()]
    for rank, hit in enumerate(hits, start=1):
        source = unquote(hit["metadata"]["source"]).lower()
        if any(w in source for w in wanted):
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generating answers")
    args = parser.parse_args()

    rows = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    with_answers = not args.retrieval_only and bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    if not args.retrieval_only and not with_answers:
        print("No ANTHROPIC_API_KEY in .env, so only retrieval is checked.\n")

    results, hits_in_top = [], 0
    for n, row in enumerate(rows, start=1):
        hits = rag.retrieve(row["question"])
        rank = found_at(hits, row["expected_source"])
        hits_in_top += rank is not None
        if rank is None:
            top = rag.describe(hits[0]["metadata"]) if hits else "nothing"
            print(f"  MISS  {row['question']}\n        expected {row['expected_source']!r}, top was {top}")

        generated = ""
        if with_answers:
            print(f"  answering {n}/{len(rows)}...", end="\r")
            try:
                generated = rag.answer(row["question"])["answer"]
            except RuntimeError as err:
                generated = f"ERROR: {err}"
        results.append({
            "question": row["question"],
            "expected_source": row["expected_source"],
            "retrieved_rank": rank or "",
            "top_sources": " ; ".join(rag.describe(h["metadata"]) for h in hits),
            "expected_answer": row.get("expected_answer", ""),
            "generated_answer": generated,
            "grade": "",  # fill in by hand: correct / partly / wrong
        })

    rate = hits_in_top / len(rows)
    print(f"\nRetrieval hit rate: {hits_in_top}/{len(rows)} = {rate:.0%} "
          f"(expected source in the top {config.TOP_K}; the target is 85%)")

    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved details to {RESULTS.relative_to(config.PROJECT_DIR)}"
          + ("; grade the generated answers in the 'grade' column." if with_answers else "."))


if __name__ == "__main__":
    main()
