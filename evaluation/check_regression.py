
import json
import sys
from pathlib import Path

RESULTS = Path("evaluation/ragas_scores.json")

THRESHOLDS = {
    "faithfulness": 0.95,
    "answer_relevancy": 0.90,
    "context_precision": 0.90,
    "context_recall": 0.80,
}


def main():
    if not RESULTS.exists():
        print(f"ERROR: Missing results file: {RESULTS}")
        return 1

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    total = data.get("total_samples", 0)
    failed = []

    print("\nRAG QUALITY REGRESSION CHECK")
    print("=" * 45)

    for name, threshold in THRESHOLDS.items():
        metric = data.get("metrics", {}).get(name, {})
        score = metric.get("average")
        successful = metric.get("successful", 0)
        status = metric.get("status")

        passed = (
            status == "completed"
            and total > 0
            and successful == total
            and isinstance(score, (int, float))
            and score >= threshold
        )

        print(
            f"{name}: score={score}, minimum={threshold:.2f}, "
            f"status={'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            failed.append(name)

    print("=" * 45)

    if failed:
        print("REGRESSION CHECK FAILED:", ", ".join(failed))
        return 1

    print("All RAG quality thresholds passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
