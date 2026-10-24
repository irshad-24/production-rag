
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from chunk_store import load_unified_chunks
from vector_store import load_vector_store
from bm25 import create_bm25_index
from reranker import load_reranker, rerank_documents
from retrieval import hybrid_search, expand_hybrid_context

QUERY = "What are the objectives of communication?"


def chunk_id(chunk):
    return chunk["metadata"]["chunk_id"]


def show_configuration(title, chunks):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"Context count: {len(chunks)}")
    print("=" * 70)

    for chunk in chunks:
        metadata = chunk["metadata"]
        content = chunk["page_content"].replace("\n", " ")

        print(
            f"\nChunk {metadata.get('chunk_id')} | "
            f"Page {metadata.get('page')} | "
            f"Type {metadata.get('source_type')}"
        )
        print(content[:350])


def main():
    print("Loading chunks and retrieval components...")
    chunks = load_unified_chunks()
    vector_store = load_vector_store()
    bm25 = create_bm25_index(chunks)
    reranker = load_reranker()

    print(f"\nQuery: {QUERY}")

    hybrid_results, _, _, _ = hybrid_search(
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        query=QUERY,
        vector_k=10,
        bm25_k=10,
        final_k=10,
    )

    reranked = rerank_documents(
        reranker=reranker,
        query=QUERY,
        documents=hybrid_results,
        top_k=3,
    )

    top3 = [chunk for chunk, score in reranked]

    # Configuration 1: Reranked chunks only
    config1 = top3

    # Configuration 2: Current expansion behavior
    config2 = expand_hybrid_context(
        chunks=chunks,
        hybrid_results=top3,
        neighbor_count=1,
    )

    # Configuration 3: Keep top 3, but only add the next
    # chunk when a selected chunk contains the objectives heading.
    config3_by_id = {chunk_id(c): c for c in top3}

    chunk_lookup = {chunk_id(c): c for c in chunks}
    numeric_ids = sorted(
        cid for cid in chunk_lookup
        if isinstance(cid, int)
    )

    for chunk in top3:
        text = chunk["page_content"].lower()

        if (
            "objectives & purpose of communication" in text
            or "objectives and purpose of communication" in text
        ):
            current_id = chunk_id(chunk)

            try:
                position = numeric_ids.index(current_id)
            except ValueError:
                continue

            next_position = position + 1

            if next_position < len(numeric_ids):
                next_id = numeric_ids[next_position]
                config3_by_id[next_id] = chunk_lookup[next_id]

    config3 = list(config3_by_id.values())

    show_configuration("CONFIG 1 — RERANKED TOP 3 ONLY", config1)
    show_configuration("CONFIG 2 — CURRENT NEIGHBOR EXPANSION", config2)
    show_configuration("CONFIG 3 — SELECTIVE OBJECTIVES EXPANSION", config3)

    print("\n" + "=" * 70)
    print("CHUNK ID COMPARISON")
    print("=" * 70)

    for name, result in [
        ("Config 1", config1),
        ("Config 2", config2),
        ("Config 3", config3),
    ]:
        print(f"{name}: {[chunk_id(c) for c in result]}")

    print("\nComparison complete. No evaluation files or scores were changed.")


if __name__ == "__main__":
    main()
