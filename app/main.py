from chunk_store import load_unified_chunks
from vector_store import load_vector_store
from bm25 import create_bm25_index
from reranker import load_reranker, rerank_documents
from retrieval import hybrid_search, expand_hybrid_context
from query_decomposer import decompose_query
from rag import generate_answer


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading unified chunks...")

chunks = load_unified_chunks()

print(
    f"Loaded {len(chunks)} unified chunks."
)


print("Loading vector store...")

vector_store = load_vector_store()

print(
    "Vector store loaded successfully!"
)


print("Creating BM25 index...")

bm25 = create_bm25_index(
    chunks
)

print(
    "BM25 index created successfully!"
)


print("Loading reranker...")

reranker = load_reranker()

print(
    "Reranker loaded successfully!"
)


# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

query = input(
    "\nEnter your question: "
)


# --------------------------------------------------
# STEP 1: QUERY DECOMPOSITION
# --------------------------------------------------

sub_questions = decompose_query(
    query
)


print(
    "\n=============================="
)

print(
    "DECOMPOSED QUESTIONS"
)

print(
    "=============================="
)


for index, sub_question in enumerate(
    sub_questions,
    start=1
):

    print(
        f"{index}. {sub_question}"
    )


# --------------------------------------------------
# PROCESS EACH SUB-QUESTION
# --------------------------------------------------

all_expanded_chunks = {}


for sub_question in sub_questions:

    print(
        f"\nProcessing sub-question: "
        f"{sub_question}"
    )


    # --------------------------------------------------
    # STEP 2: HYBRID SEARCH
    # --------------------------------------------------

    hybrid_results, vector_results, bm25_results, rrf_scores = hybrid_search(
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        query=sub_question,
        vector_k=10,
        bm25_k=10,
        final_k=10
    )
    
    # --------------------------------------------------
    # STEP 3: RERANKING
    # --------------------------------------------------

    reranked_results = rerank_documents(
        reranker=reranker,
        query=sub_question,
        documents=hybrid_results,
        top_k=3
    )


    # --------------------------------------------------
    # STEP 4: CONTEXT EXPANSION
    # --------------------------------------------------

    reranked_chunks = [
        chunk
        for chunk, score in reranked_results
    ]


    expanded_chunks_for_query = (
        expand_hybrid_context(
            chunks=chunks,
            hybrid_results=reranked_chunks,
            neighbor_count=1
        )
    )


    # --------------------------------------------------
    # STORE UNIQUE CONTEXT
    # --------------------------------------------------

    for chunk in expanded_chunks_for_query:

        chunk_id = chunk["metadata"]["chunk_id"]

        all_expanded_chunks[
            chunk_id
        ] = chunk


# --------------------------------------------------
# FINAL COMBINED CONTEXT
# --------------------------------------------------

expanded_chunks = list(
    all_expanded_chunks.values()
)


print(
    "\n=============================="
)

print(
    "COMBINED CONTEXT"
)

print(
    "=============================="
)


for chunk in expanded_chunks:

    metadata = chunk["metadata"]

    print(
        f"Chunk {metadata.get('chunk_id')} "
        f"| Page {metadata.get('page')} "
        f"| Source {metadata.get('source_type')}"
    )


print(
    f"\nTotal context chunks: "
    f"{len(expanded_chunks)}"
)


# --------------------------------------------------
# STEP 5: CONVERT TO DOCUMENT OBJECTS
# --------------------------------------------------

class SimpleDocument:

    def __init__(
        self,
        page_content,
        metadata
    ):

        self.page_content = page_content

        self.metadata = metadata


documents = [

    SimpleDocument(
        page_content=
            chunk["page_content"],

        metadata=
            chunk["metadata"]
    )

    for chunk in expanded_chunks
]


# --------------------------------------------------
# STEP 6: GENERATE FINAL ANSWER
# --------------------------------------------------

print(
    "\nGenerating answer..."
)


answer = generate_answer(
    query=query,
    retrieved_documents=documents
)


# --------------------------------------------------
# FINAL ANSWER
# --------------------------------------------------

print(
    "\n=============================="
)

print(
    "FINAL ANSWER"
)

print(
    "==============================\n"
)


print(answer)