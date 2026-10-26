
import sys
from pathlib import Path

# Allow imports from app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from chunk_store import load_unified_chunks
from vector_store import load_vector_store
from bm25 import create_bm25_index
from reranker import load_reranker, rerank_documents
from retrieval import hybrid_search, expand_hybrid_context


import json

DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    QUESTIONS = json.load(f)

def show_chunk(label, chunk, score=None):
    metadata = chunk["metadata"]

    print(f"\n{label}")
    print(f"  Chunk ID: {metadata.get('chunk_id')}")
    print(f"  Page: {metadata.get('page')}")
    print(f"  Source type: {metadata.get('source_type')}")

    if score is not None:
        print(f"  Score: {float(score):.6f}")

    content = chunk["page_content"].replace("\n", " ")
    print(f"  Content: {content[:450]}")



def main():
    print("=" * 65)
    print("RETRIEVAL DIAGNOSTIC — ALL EVALUATION QUESTIONS")
    print("=" * 65)

    chunks = load_unified_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    vector_store = load_vector_store()
    bm25 = create_bm25_index(chunks)
    reranker = load_reranker()

    for index, sample in enumerate(QUESTIONS, start=1):
        query = sample["question"]

        print("\n" + "=" * 65)
        print(f"QUESTION {index}/{len(QUESTIONS)}: {query}")
        print("=" * 65)

        hybrid_results, _, _, rrf_scores = hybrid_search(
            vector_store=vector_store,
            bm25=bm25,
            chunks=chunks,
            query=query,
            vector_k=10,
            bm25_k=10,
            final_k=10,
        )

        reranked_results = rerank_documents(
            reranker=reranker,
            query=query,
            documents=hybrid_results,
            top_k=3,
        )

        reranked_chunks = [
            chunk for chunk, score in reranked_results
        ]

        print("\n--- RERANKED TOP 3 ---")
        for rank, (chunk, score) in enumerate(
            reranked_results, start=1
        ):
            show_chunk(f"Rank {rank}", chunk, score)

        expanded_chunks = expand_hybrid_context(
            chunks=chunks,
            hybrid_results=reranked_chunks,
            neighbor_count=1,
        )

        reranked_ids = {
            chunk["metadata"]["chunk_id"]
            for chunk in reranked_chunks
        }

        print("\n--- EXPANDED CONTEXT ---")
        print("Reranked IDs:", sorted(map(str, reranked_ids)))

        for chunk in expanded_chunks:
            chunk_id = chunk["metadata"]["chunk_id"]
            label = (
                "SELECTED BY RERANKER"
                if chunk_id in reranked_ids
                else "ADDED AS NEIGHBOR"
            )
            show_chunk(label, chunk)

    print("\nDiagnostic complete. No evaluation scores were modified.")


if __name__ == "__main__":
    main()


