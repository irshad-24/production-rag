
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


QUERY = "What are the objectives of communication?"


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
    print("RETRIEVAL DIAGNOSTIC")
    print("=" * 65)
    print(f"Query: {QUERY}")

    print("\n[1/4] Loading chunks...")
    chunks = load_unified_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("\n[2/4] Loading vector store and BM25...")
    vector_store = load_vector_store()
    bm25 = create_bm25_index(chunks)

    print("\n[3/4] Running hybrid retrieval...")
    hybrid_results, vector_results, bm25_results, rrf_scores = (
        hybrid_search(
            vector_store=vector_store,
            bm25=bm25,
            chunks=chunks,
            query=QUERY,
            vector_k=10,
            bm25_k=10,
            final_k=10,
        )
    )

    print("\n--- HYBRID TOP 10 ---")
    for rank, chunk in enumerate(hybrid_results, start=1):
        chunk_id = chunk["metadata"]["chunk_id"]
        show_chunk(
            f"Rank {rank}",
            chunk,
            rrf_scores.get(chunk_id),
        )

    print("\n[4/4] Running reranker...")
    reranker = load_reranker()

    reranked_results = rerank_documents(
        reranker=reranker,
        query=QUERY,
        documents=hybrid_results,
        top_k=3,
    )

    reranked_chunks = [
        chunk for chunk, score in reranked_results
    ]

    print("\n--- RERANKER TOP 3 ---")
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

    print("\n--- CONTEXT EXPANSION ---")
    print(f"Reranked chunks: {len(reranked_ids)}")
    print(f"Expanded context chunks: {len(expanded_chunks)}")

    for chunk in expanded_chunks:
        chunk_id = chunk["metadata"]["chunk_id"]

        if chunk_id in reranked_ids:
            label = "SELECTED BY RERANKER"
        else:
            label = "ADDED AS NEIGHBOR"

        show_chunk(label, chunk)

    print("\n--- SUMMARY ---")
    print("Reranked IDs:", sorted(map(str, reranked_ids)))
    print(
        "Expanded IDs:",
        [
            str(chunk["metadata"]["chunk_id"])
            for chunk in expanded_chunks
        ],
    )
    print(
        "New neighbor IDs:",
        [
            str(chunk["metadata"]["chunk_id"])
            for chunk in expanded_chunks
            if chunk["metadata"]["chunk_id"] not in reranked_ids
        ],
    )

    print("\nDiagnostic complete. No evaluation scores were modified.")


if __name__ == "__main__":
    main()
