from groq import RateLimitError

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

# Existing modules use imports relative to the app directory.
APP_DIR = str(Path(__file__).resolve().parent)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from chunk_store import load_unified_chunks
from vector_store import load_vector_store
from bm25 import create_bm25_index
from reranker import load_reranker, rerank_documents
from retrieval import hybrid_search, expand_hybrid_context
from query_decomposer import decompose_query
from rag import generate_answer


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    chunk_id: str
    page: int | None = None
    source_type: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared components once when the API starts.
    try:
        app.state.chunks = load_unified_chunks()
        app.state.vector_store = load_vector_store()
        app.state.bm25 = create_bm25_index(app.state.chunks)
        app.state.reranker = load_reranker()
        app.state.ready = True
    except Exception:
        app.state.ready = False
        raise

    yield


app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    ready = getattr(app.state, "ready", False)

    return {
        "status": "ok" if ready else "not_ready",
        "ready": ready,
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank.",
        )

    started = time.perf_counter()

    try:
        chunks = app.state.chunks
        all_context = {}

        sub_questions = decompose_query(question)

        for sub_question in sub_questions:
            hybrid_results, _, _, _ = hybrid_search(
                vector_store=app.state.vector_store,
                bm25=app.state.bm25,
                chunks=chunks,
                query=sub_question,
                vector_k=10,
                bm25_k=10,
                final_k=10,
            )

            reranked = rerank_documents(
                reranker=app.state.reranker,
                query=sub_question,
                documents=hybrid_results,
                top_k=2,
            )

            reranked_chunks = [
                chunk for chunk, score in reranked
            ]

            expanded = expand_hybrid_context(
                chunks=chunks,
                hybrid_results=reranked_chunks,
                neighbor_count=1,
            )

            for chunk in expanded:
                chunk_id = chunk["metadata"]["chunk_id"]
                all_context[chunk_id] = chunk

        if not all_context:
            raise HTTPException(
                status_code=404,
                detail="No relevant context was retrieved.",
            )

        class SimpleDocument:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        context_documents = [
            SimpleDocument(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"],
            )
            for chunk in all_context.values()
        ]

        answer = generate_answer(
            query=question,
            retrieved_documents=context_documents,
        )

        sources = [
            Source(
                chunk_id=str(doc["metadata"]["chunk_id"]),
                page=doc["metadata"].get("page"),
                source_type=doc["metadata"].get("source_type"),
            )
            for doc in all_context.values()
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            latency_seconds=round(
                time.perf_counter() - started, 3
            ),
        )

    
    except RateLimitError:
        raise HTTPException(
            status_code=503,
            detail="Groq quota exhausted. Try again after the quota resets.",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question.",
        )

