from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


def load_reranker():
    reranker = CrossEncoder(MODEL_NAME)

    return reranker


def rerank_documents(
    reranker,
    query,
    documents,
    top_k=3
):
    pairs = [
        (query, document["page_content"])
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_documents[:top_k]