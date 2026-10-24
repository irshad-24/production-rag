import os
import json
import math
import time

from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import llm_factory

from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall

from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
DATA_PATH = "evaluation/evaluation_data.json"
OUTPUT_PATH = "evaluation/ragas_scores.json"

api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")

if not api_key:
    raise ValueError(
        "Groq API key missing. Check GROQ_API_KEY or groq_api_key in .env"
    )


# ============================================================
# LOAD DATA
# ============================================================

with open(DATA_PATH, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)

if not evaluation_data:
    raise ValueError("Evaluation dataset is empty.")

required_columns = {
    "user_input",
    "response",
    "retrieved_contexts",
    "reference",
}

for index, sample in enumerate(evaluation_data):
    missing = required_columns - set(sample)
    if missing:
        raise ValueError(
            f"Sample {index} is missing fields: {sorted(missing)}"
        )

print(f"Loaded {len(evaluation_data)} samples.")


# ============================================================
# GROQ + RAGAS LLM
# ============================================================

groq_client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

ragas_llm = llm_factory(
    model=MODEL_NAME,
    provider="openai",
    client=groq_client,
    adapter="instructor",
    max_tokens=4096,
)

print(f"Evaluation model: {MODEL_NAME}")


# ============================================================
# LOCAL EMBEDDINGS
# ============================================================

ragas_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# METRICS
# ============================================================

faithfulness.llm = ragas_llm

answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
answer_relevancy.strictness = 1

context_precision.llm = ragas_llm
context_recall.llm = ragas_llm

metrics = [
    ("faithfulness", faithfulness),
    ("answer_relevancy", answer_relevancy),
    ("context_precision", context_precision),
    ("context_recall", context_recall),
]


# ============================================================
# RESUMABLE RESULTS
# ============================================================

def new_results():
    return {
        "model": MODEL_NAME,
        "total_samples": len(evaluation_data),
        "metrics": {},
    }


# Reuse saved results only when model and sample count match.
# This preserves the scores already saved in ragas_scores.json.
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
            previous = json.load(file)

        if (
            previous.get("model") == MODEL_NAME
            and previous.get("total_samples") == len(evaluation_data)
        ):
            results = previous
            print("Resuming existing Groq evaluation.")
        else:
            results = new_results()
            print("Starting fresh results for this model/dataset.")
    except (json.JSONDecodeError, OSError):
        results = new_results()
        print("Existing results unreadable; starting fresh.")
else:
    results = new_results()


def save_results():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    temp_path = OUTPUT_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    os.replace(temp_path, OUTPUT_PATH)


def valid_score(value):
    try:
        score = float(value)
        return score if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def update_metric_summary(metric_name):
    metric_result = results["metrics"][metric_name]
    scores = metric_result["scores"]

    metric_result["successful"] = len(scores)
    metric_result["failed"] = len(evaluation_data) - len(scores)
    metric_result["average"] = (
        round(sum(item["score"] for item in scores) / len(scores), 4)
        if scores
        else None
    )

    metric_result["status"] = (
        "completed"
        if len(scores) == len(evaluation_data)
        else "incomplete"
    )


def is_daily_token_limit(error_text):
    """Return True for Groq daily token quota errors (TPD)."""
    text = error_text.lower()
    return (
        "rate_limit_exceeded" in text
        and (
            "tokens per day" in text
            or "tpd" in text
            or "limit 200000" in text
        )
    )


# ============================================================
# EVALUATE ONE SAMPLE AT A TIME
# ============================================================

print("\n" + "=" * 55)
print("STARTING RESUMABLE GROQ RAGAS EVALUATION")
print("=" * 55)

stop_due_to_quota = False

for metric_name, metric in metrics:
    metric_result = results["metrics"].setdefault(
        metric_name,
        {
            "status": "incomplete",
            "successful": 0,
            "failed": len(evaluation_data),
            "average": None,
            "scores": [],
            "failed_samples": [],
        },
    )

    # Keep valid scores and retry only indices that have no valid score.
    score_by_index = {
        item["sample_index"]: item
        for item in metric_result["scores"]
        if (
            isinstance(item, dict)
            and isinstance(item.get("sample_index"), int)
            and valid_score(item.get("score")) is not None
            and 0 <= item["sample_index"] < len(evaluation_data)
        )
    }

    metric_result["scores"] = [
        {"sample_index": i, "score": valid_score(score_by_index[i]["score"])}
        for i in sorted(score_by_index)
    ]

    print(f"\n{'=' * 45}")
    print(f"Metric: {metric_name}")
    print(f"Already completed: {len(score_by_index)}/{len(evaluation_data)}")

    # If this metric is already complete, don't make any API calls for it.
    if len(score_by_index) == len(evaluation_data):
        metric_result["failed_samples"] = []
        update_metric_summary(metric_name)
        save_results()
        print(f"{metric_name} already complete. Skipping API calls.")
        continue

    for index, sample in enumerate(evaluation_data):
        if index in score_by_index:
            print(f"[{index + 1}/{len(evaluation_data)}] Skipping saved score.")
            continue

        print(f"[{index + 1}/{len(evaluation_data)}] Evaluating sample...")

        last_error = None
        score = None
        daily_quota_hit = False

        # Retry ordinary failures, but stop immediately on daily token quota.
        for attempt in range(3):
            try:
                one_sample_dataset = Dataset.from_list([sample])

                result = evaluate(
                    dataset=one_sample_dataset,
                    metrics=[metric],
                    run_config=RunConfig(
                        max_workers=1,
                        max_retries=1,
                        max_wait=15,
                        timeout=120,
                    ),
                    batch_size=1,
                    raise_exceptions=True,
                    show_progress=False,
                )

                score_df = result.to_pandas()

                if metric_name not in score_df.columns:
                    raise RuntimeError(
                        f"Missing metric column: {metric_name}"
                    )

                score = valid_score(score_df.iloc[0][metric_name])

                if score is None:
                    raise RuntimeError("Metric returned no valid score.")

                break

            except Exception as exc:
                last_error = str(exc)

                if is_daily_token_limit(last_error):
                    daily_quota_hit = True
                    stop_due_to_quota = True
                    print(
                        "Groq daily token quota exhausted. "
                        "Stopping evaluation now; saved scores will be preserved."
                    )
                    break

                if attempt < 2:
                    wait_seconds = 5 * (2 ** attempt)
                    print(
                        f"Attempt {attempt + 1} failed. "
                        f"Retrying in {wait_seconds}s: {last_error[:200]}"
                    )
                    time.sleep(wait_seconds)

        if score is not None:
            score_by_index[index] = {
                "sample_index": index,
                "score": score,
            }

            metric_result["scores"] = [
                score_by_index[i] for i in sorted(score_by_index)
            ]

            # Remove any old failure entry for this sample after success.
            metric_result["failed_samples"] = [
                item
                for item in metric_result.get("failed_samples", [])
                if item.get("sample_index") != index
            ]

            update_metric_summary(metric_name)
            save_results()
            print(f"Score: {score:.4f} | Saved.")

        elif not daily_quota_hit:
            failures = {
                item["sample_index"]: item
                for item in metric_result.get("failed_samples", [])
                if isinstance(item, dict) and "sample_index" in item
            }

            failures[index] = {
                "sample_index": index,
                "user_input": sample["user_input"],
                "error": last_error or "Unknown evaluation error",
            }

            metric_result["failed_samples"] = [
                failures[i] for i in sorted(failures)
            ]

            print(f"Failed after retries: {last_error}")
            update_metric_summary(metric_name)
            save_results()

        # Don't keep making calls once the daily token quota is exhausted.
        if stop_due_to_quota:
            break

        time.sleep(1)

    # Update the summary even if this metric was interrupted by a quota limit.
    update_metric_summary(metric_name)
    save_results()

    print(
        f"{metric_name}: {metric_result['successful']}/"
        f"{len(evaluation_data)} successful; "
        f"average={metric_result['average']}"
    )

    # Stop the outer metric loop too, so no other metric consumes API tokens.
    if stop_due_to_quota:
        print(
            "Evaluation paused because of the Groq daily token quota. "
            "Run this script again after quota resets to resume missing samples."
        )
        break


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("FINAL RAGAS RESULTS")
print("=" * 55)

for metric_name, _ in metrics:
    metric_result = results["metrics"].get(metric_name)

    if metric_result is None:
        print(f"{metric_name}: not started")
        continue

    print(
        f"{metric_name}: "
        f"{metric_result['successful']}/{len(evaluation_data)} successful | "
        f"average={metric_result['average']} | "
        f"status={metric_result['status']}"
    )

print(f"\nResults saved to: {OUTPUT_PATH}")
