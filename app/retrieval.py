from bm25 import search_bm25
from reranker import rerank_documents


def retrieve_with_context(
    vector_store,
    query,
    chunks,
    k=3,
    neighbor_count=1
):

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    # Build explicit lookup using chunk_id
    chunk_lookup = {
        chunk["metadata"]["chunk_id"]: chunk
        for chunk in chunks
    }

    retrieved_chunks = {}

    # Only numeric text chunks participate
    # in neighboring context expansion.
    text_chunk_ids = sorted(
        chunk_id
        for chunk_id in chunk_lookup
        if isinstance(chunk_id, int)
    )

    for document, score in results:

        chunk_id = document.metadata["chunk_id"]

        # ------------------------------------------
        # VISUAL / NON-TEXT CHUNK
        # ------------------------------------------

        if not isinstance(chunk_id, int):

            if chunk_id in chunk_lookup:
                retrieved_chunks[chunk_id] = (
                    chunk_lookup[chunk_id]
                )

            continue

        # ------------------------------------------
        # TEXT CHUNK
        # ------------------------------------------

        try:
            position = text_chunk_ids.index(chunk_id)
        except ValueError:
            continue

        start = max(
            0,
            position - neighbor_count
        )

        end = min(
            len(text_chunk_ids),
            position + neighbor_count + 1
        )

        for neighbor_position in range(
            start,
            end
        ):

            neighbor_id = text_chunk_ids[
                neighbor_position
            ]

            retrieved_chunks[
                neighbor_id
            ] = chunk_lookup[
                neighbor_id
            ]

    return (
        list(retrieved_chunks.values()),
        results
    )


def hybrid_search(
    vector_store,
    bm25,
    chunks,
    query,
    vector_k=5,
    bm25_k=5,
    final_k=3,
    rrf_k=60
):

    # 1. Semantic search using Qdrant

    vector_results = (
        vector_store
        .similarity_search_with_score(
            query,
            k=vector_k
        )
    )

    # 2. Keyword search using BM25

    bm25_results = search_bm25(
        bm25,
        chunks,
        query,
        k=bm25_k
    )

    # Stores combined RRF scores

    rrf_scores = {}

    # 3. Add Qdrant rankings

    for rank, (
        document,
        score
    ) in enumerate(
        vector_results,
        start=1
    ):

        chunk_id = document.metadata[
            "chunk_id"
        ]

        if chunk_id not in rrf_scores:

            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += (
            1 / (rrf_k + rank)
        )

    # 4. Add BM25 rankings

    for rank, (
        chunk,
        score
    ) in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = chunk["metadata"][
            "chunk_id"
        ]

        if chunk_id not in rrf_scores:

            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += (
            1 / (rrf_k + rank)
        )

    # 5. Sort chunks by combined RRF score

    ranked_chunk_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    top_chunk_ids = ranked_chunk_ids[
        :final_k
    ]

    # 6. Get actual chunks

    chunk_lookup = {
        chunk["metadata"]["chunk_id"]: chunk
        for chunk in chunks
    }

    hybrid_results = [
        chunk_lookup[chunk_id]
        for chunk_id in top_chunk_ids
    ]

    return (
        hybrid_results,
        vector_results,
        bm25_results,
        rrf_scores
    )


def expand_hybrid_context(
    chunks,
    hybrid_results,
    neighbor_count=1
):

    expanded_chunks = {}

    # Lookup chunks directly by chunk_id
    chunk_lookup = {
        chunk["metadata"]["chunk_id"]: chunk
        for chunk in chunks
    }

    # Get only numeric text chunk IDs
    text_chunk_ids = sorted(
        chunk_id
        for chunk_id in chunk_lookup
        if isinstance(chunk_id, int)
    )

    for chunk in hybrid_results:

        chunk_id = chunk["metadata"]["chunk_id"]

        # ------------------------------------------
        # VISUAL / NON-TEXT CHUNK
        # ------------------------------------------

        if not isinstance(chunk_id, int):

            if chunk_id in chunk_lookup:
                expanded_chunks[
                    chunk_id
                ] = chunk_lookup[
                    chunk_id
                ]

            continue

        # ------------------------------------------
        # TEXT CHUNK
        # ------------------------------------------

        try:
            position = text_chunk_ids.index(chunk_id)
        except ValueError:
            continue

        start = max(
            0,
            position - neighbor_count
        )

        end = min(
            len(text_chunk_ids),
            position + neighbor_count + 1
        )

        for neighbor_position in range(
            start,
            end
        ):

            neighbor_id = text_chunk_ids[
                neighbor_position
            ]

            expanded_chunks[
                neighbor_id
            ] = chunk_lookup[
                neighbor_id
            ]

    return list(
        expanded_chunks.values()
    )

def hybrid_search_with_reranking(
    vector_store,
    bm25,
    reranker,
    chunks,
    query,
    vector_k=5,
    bm25_k=5,
    candidate_k=5,
    final_k=3
):

    # Step 1: Hybrid retrieval

    (
        hybrid_results,
        vector_results,
        bm25_results,
        rrf_scores
    ) = hybrid_search(
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        query=query,
        vector_k=vector_k,
        bm25_k=bm25_k,
        final_k=candidate_k
    )

    # Step 2: Rerank hybrid candidates

    reranked_results = rerank_documents(
        reranker=reranker,
        query=query,
        documents=hybrid_results,
        top_k=final_k
    )

    return (
        reranked_results,
        vector_results,
        bm25_results,
        rrf_scores
    )