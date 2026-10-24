
import hashlib
import json
import sys
from pathlib import Path

# Allow imports from app/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from chunk_store import load_unified_chunks
from vector_store import load_vector_store
from bm25 import create_bm25_index
from reranker import load_reranker, rerank_documents
from retrieval import hybrid_search
from query_decomposer import decompose_query
from rag import generate_answer


DATASET_PATH = ROOT / "evaluation" / "dataset.json"
OUTPUT_PATH = ROOT / "evaluation" / "evaluation_data.json"
STATE_PATH = ROOT / "evaluation" / "evaluation_state.json"

PIPELINE_VERSION = "context-selection-v2"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def atomic_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temp_path.replace(path)


def dataset_fingerprint(dataset):
    content = json.dumps(
        dataset, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_progress(dataset):
    fingerprint = dataset_fingerprint(dataset)

    try:
        state = load_json(STATE_PATH)
        records = load_json(OUTPUT_PATH)

        valid_state = (
            state.get("fingerprint") == fingerprint
            and state.get("pipeline_version") == PIPELINE_VERSION
            and isinstance(state.get("completed_indices"), list)
            and isinstance(records, list)
            and len(records) == len(state["completed_indices"])
        )

        if valid_state:
            indices = state["completed_indices"]

            if (
                len(set(indices)) == len(indices)
                and all(
                    isinstance(i, int) and 0 <= i < len(dataset)
                    for i in indices
                )
                and all(
                    isinstance(record, dict)
                    and record.get("user_input") == dataset[i]["question"]
                    and record.get("reference") == dataset[i]["ground_truth"]
                    and isinstance(record.get("response"), str)
                    and isinstance(record.get("retrieved_contexts"), list)
                    for i, record in zip(indices, records)
                )
            ):
                print(f"Resuming: {len(records)}/{len(dataset)} completed.")
                return dict(zip(indices, records)), fingerprint

    except (OSError, ValueError, TypeError, AttributeError):
        pass

    print("No compatible saved progress. Starting fresh.")
    atomic_save(OUTPUT_PATH, [])
    atomic_save(STATE_PATH, {
        "fingerprint": fingerprint,
        "pipeline_version": PIPELINE_VERSION,
        "completed_indices": [],
    })
    return {}, fingerprint


def save_progress(completed, fingerprint):
    indices = sorted(completed)

    # Keep the standard RAGAS input schema.
    records = [completed[i] for i in indices]

    atomic_save(OUTPUT_PATH, records)
    atomic_save(STATE_PATH, {
        "fingerprint": fingerprint,
        "pipeline_version": PIPELINE_VERSION,
        "completed_indices": indices,
    })


def run_rag(query, chunks, vector_store, bm25, reranker):
    sub_questions = decompose_query(query)
    selected_chunks = {}

    for sub_question in sub_questions:
        hybrid_results, _, _, _ = hybrid_search(
            vector_store=vector_store,
            bm25=bm25,
            chunks=chunks,
            query=sub_question,
            vector_k=10,
            bm25_k=10,
            final_k=10,
        )

        if not hybrid_results:
            continue

        reranked_results = rerank_documents(
            reranker=reranker,
            query=sub_question,
            documents=hybrid_results,
            top_k=3,
        )

        for chunk, score in reranked_results:
            chunk_id = chunk["metadata"]["chunk_id"]
            selected_chunks[chunk_id] = chunk

    class SimpleDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    documents = [
        SimpleDocument(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"],
        )
        for chunk in selected_chunks.values()
    ]

    if documents:
        answer = generate_answer(
            query=query,
            retrieved_documents=documents,
        )
    else:
        answer = "I could not find the answer in the provided document."

    contexts = [doc.page_content for doc in documents]
    return answer, contexts


def main():
    print("=" * 40)
    print("RESUMABLE RAG EVALUATION DATA GENERATION")
    print("=" * 40)

    dataset = load_json(DATASET_PATH)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    for i, item in enumerate(dataset):
        if not all(k in item for k in ("question", "ground_truth")):
            raise ValueError(f"Invalid dataset item at index {i}.")

    completed, fingerprint = load_progress(dataset)

    if len(completed) == len(dataset):
        print("All questions are already complete. Nothing to do.")
        print(f"Output: {OUTPUT_PATH}")
        return

    print("Loading unified chunks...")
    chunks = load_unified_chunks()

    print("Loading vector store...")
    vector_store = load_vector_store()

    print("Creating BM25 index...")
    bm25 = create_bm25_index(chunks)

    print("Loading reranker...")
    reranker = load_reranker()

    for index, item in enumerate(dataset):
        if index in completed:
            print(f"\n[{index + 1}/{len(dataset)}] Already saved. Skipping.")
            continue

        question = item["question"]
        print(f"\n[{index + 1}/{len(dataset)}] {question}")

        try:
            answer, contexts = run_rag(
                query=question,
                chunks=chunks,
                vector_store=vector_store,
                bm25=bm25,
                reranker=reranker,
            )

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Generated answer is empty.")

            record = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }

            completed[index] = record
            save_progress(completed, fingerprint)

            print(f"Answer: {answer}")
            print(f"Retrieved contexts: {len(contexts)}")
            print("Progress saved successfully.")

        except Exception as exc:
            print(f"Question failed: {type(exc).__name__}: {exc}")
            print("Previous completed results are preserved.")
            print("Rerun this script to retry incomplete questions.")

            error_text = str(exc).lower()
            if any(term in error_text for term in (
                "rate_limit_exceeded",
                "tokens per day",
                "tokens per minute",
                "insufficient_quota",
            )):
                print("Possible API quota/rate limit detected; stopping.")
                break

    print("\n" + "=" * 40)
    print(f"Completed: {len(completed)}/{len(dataset)}")
    print(f"Saved data: {OUTPUT_PATH}")
    print(f"Saved state: {STATE_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()
