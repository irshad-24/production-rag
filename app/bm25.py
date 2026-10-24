import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def create_bm25_index(chunks):
    tokenized_corpus = [
        tokenize(chunk["page_content"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def search_bm25(bm25, chunks, query, k=5):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:k]

    results = []

    for index in ranked_indices:
        results.append(
            (
                chunks[index],
                scores[index]
            )
        )

    return results

