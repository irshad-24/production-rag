
from app.chunk_store import load_unified_chunks
from app.vector_store import load_vector_store
from app.bm25 import create_bm25_index
from app.retrieval import hybrid_search


chunks = load_unified_chunks()
vector_store = load_vector_store()
bm25 = create_bm25_index(chunks)

queries = [
    ("Who has Python skill?", "csv"),
    ("What is Example Domain used for?", "url"),
]

for query, expected_source in queries:
    results, vector_results, bm25_results, scores = hybrid_search(
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        query=query,
        vector_k=10,
        bm25_k=10,
        final_k=5,
    )

    print(f"\nQuery: {query}")
    print("Expected source:", expected_source)
    print("Vector source types:",
          sorted({d.metadata.get("source_type", "unknown")
                  for d, _ in vector_results}))
    print("BM25 source types:",
          sorted({c["metadata"].get("source_type", "unknown")
                  for c, _ in bm25_results}))
    print("Hybrid results:")

    for chunk in results:
        metadata = chunk["metadata"]
        print(
            metadata.get("source_type"),
            metadata.get("chunk_id"),
            chunk["page_content"][:120],
        )

    assert any(
        chunk["metadata"].get("source_type") == expected_source
        for chunk in results
    ), f"{expected_source} chunk missing from hybrid results"

print("\nBoth multi-source hybrid retrieval tests passed!")
