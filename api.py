"""
api.py

FastAPI REST server for the RAG engine.

Endpoints:
    POST /query       — ask a question, get a RAG-grounded answer
    GET  /health      — liveness check
    GET  /stats       — index statistics

Usage:
    pip install fastapi uvicorn
    uvicorn api:app --reload

    Then POST to http://localhost:8000/query:
    {
        "question": "What caused the collapse of Atlantic cod?",
        "top_k": 3
    }
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import (
    load_documents,
    create_chunks,
    embed_chunks,
    retrieve,
    generate_answer,
)
import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
MODEL_NAME   = os.getenv("EMBED_MODEL",  "text-embedding-3-small") 
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE",  "120"))
OVERLAP      = int(os.getenv("OVERLAP",     "30"))
DEFAULT_TOP_K = int(os.getenv("TOP_K",      "3"))

# ---------------------------------------------------------------------------
# Index — built once at startup, shared across all requests
# ---------------------------------------------------------------------------

index = {}

def ensure_index_loaded():
    if index:
        return

    documents = load_documents(DOCS_FOLDER)
    chunks = create_chunks(documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    model, embeddings = embed_chunks(chunks, model_name=MODEL_NAME)

    index["model"] = model
    index["embeddings"] = embeddings
    index["chunks"] = chunks
    index["doc_count"] = len(documents)
    index["chunk_count"] = len(chunks)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Engine API",
    description=(
        "A retrieval-augmented generation API that answers questions "
        "over a private document corpus using dense vector retrieval "
        "and GPT-4o-mini generation."
    ),
    version="1.0.0",

)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = DEFAULT_TOP_K
    use_llm: Optional[bool] = True

class SourceResult(BaseModel):
    filename: str
    chunk_id: int
    score: float
    excerpt: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResult]
    top_k: int
    llm_used: bool

class HealthResponse(BaseModel):
    status: str

class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    embedding_dimensions: int
    embedding_model: str
    docs_folder: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse, tags=["System"])
def stats():
    """Returns index statistics: document count, chunk count, embedding dimensions."""
    
    ensure_index_loaded()
    return {
        "document_count":      index["doc_count"],
        "chunk_count":         index["chunk_count"],
        "embedding_dimensions": index["embeddings"].shape[1],
        "embedding_model":     MODEL_NAME,
        "docs_folder":         DOCS_FOLDER,
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Ask a question. Returns a grounded answer with cited sources.

    - **question**: natural language question about the document corpus
    - **top_k**: number of chunks to retrieve (default 3, max 10)
    - **use_llm**: if false, returns raw retrieved evidence without LLM generation
    """
    ensure_index_loaded()

    top_k = min(max(1, request.top_k or DEFAULT_TOP_K), 10)

    retrieved = retrieve(
        request.question,
        index["model"],
        index["embeddings"],
        index["chunks"],
        top_k=top_k,
    )

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    answer = generate_answer(
        request.question,
        retrieved,
        use_llm=request.use_llm,
        openai_model=OPENAI_MODEL,
    )

    sources = [
        SourceResult(
            filename=c["filename"],
            chunk_id=c["chunk_id"],
            score=round(c["score"], 4),
            excerpt=c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
        )
        for c in retrieved
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        top_k=top_k,
        llm_used=request.use_llm,
    )
