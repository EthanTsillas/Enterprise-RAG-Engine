"""
rag_engine.py

Core retrieval-augmented generation engine.

Implements document loading, chunking, embedding, cosine similarity
retrieval, and LLM-augmented answer generation — all from scratch
with no vector database dependencies.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(folder_path: str) -> List[Dict]:
    """
    Load all .txt files from a folder.

    Returns a list of dicts: {"filename": str, "text": str}
    """
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"filename": filename, "text": text})
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:       Input text string.
        chunk_size: Number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def create_chunks(
    documents: List[Dict],
    chunk_size: int = 120,
    overlap: int = 30
) -> List[Dict]:
    """
    Chunk all documents.

    Returns a list of dicts: {"filename": str, "chunk_id": int, "text": str}
    """
    all_chunks = []
    for doc in documents:
        for i, chunk_str in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk_str,
            })
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[Dict],
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Embed all chunks using a SentenceTransformer model.

    Returns:
        model:      The loaded SentenceTransformer.
        embeddings: NumPy array of shape (n_chunks, embedding_dim).
    """
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return model, embeddings


# ---------------------------------------------------------------------------
# Retrieval — cosine similarity from scratch (no sklearn, no faiss)
# ---------------------------------------------------------------------------

def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix.

    Args:
        query_vector: Shape (embedding_dim,)
        matrix:       Shape (n_chunks, embedding_dim)

    Returns:
        similarities: Shape (n_chunks,)
    """
    dot_products = np.dot(matrix, query_vector)
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    return dot_products / (matrix_norms * query_norm + 1e-10)


def retrieve(
    question: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """
    Retrieve the top_k most relevant chunks for a question.

    Returns chunks sorted by cosine similarity, each with an added "score" field.
    """
    question_embedding = model.encode(question, convert_to_numpy=True)
    scores = cosine_similarity(question_embedding, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = chunks[idx].copy()
        result["score"] = float(scores[idx])
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Construct a RAG prompt from the question and retrieved context chunks.
    """
    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    return f"""You are a precise and helpful AI assistant.

Answer the question using ONLY the context provided below.
If the answer is not present in the context, respond with:
"I do not have enough information in the provided documents to answer this."

Context:
{context}

Question:
{question}

Answer:"""


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    use_llm: bool = True,
    openai_model: str = "gpt-4o-mini"
) -> str:
    """
    Generate an answer using retrieved context.

    Args:
        question:        The user's question.
        retrieved_chunks: Chunks returned by retrieve().
        use_llm:         If True, call the OpenAI API. If False, return raw chunks.
        openai_model:    Which OpenAI model to use.

    Returns:
        Answer string.
    """
    prompt = build_prompt(question, retrieved_chunks)

    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer questions precisely using only retrieved context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            print("[Warning] openai package not installed. Falling back to retrieval-only mode.")
        except Exception as e:
            print(f"[Warning] OpenAI API error: {e}. Falling back to retrieval-only mode.")

    # Fallback: surface retrieved evidence directly
    lines = ["Retrieved evidence:\n"]
    for i, chunk in enumerate(retrieved_chunks, start=1):
        lines.append(f"[{i}] {chunk['filename']} (chunk {chunk['chunk_id']}, score: {chunk['score']:.4f})")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A/B Comparison — RAG vs base LLM (no context)
# ---------------------------------------------------------------------------

def generate_answer_no_context(
    question: str,
    openai_model: str = "gpt-4o-mini"
) -> str:
    """
    Generate an answer WITHOUT retrieval context, for A/B comparison.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {e}"


def compare_rag_vs_base(
    question: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: List[Dict],
    top_k: int = 3,
    openai_model: str = "gpt-4o-mini"
) -> Dict:
    """
    Run the same question through both the RAG pipeline and the base LLM,
    returning both answers and the sources used for the RAG answer.

    Useful for validating that retrieval meaningfully improves answer quality.
    """
    retrieved = retrieve(question, model, embeddings, chunks, top_k=top_k)
    rag_answer = generate_answer(question, retrieved, use_llm=True, openai_model=openai_model)
    base_answer = generate_answer_no_context(question, openai_model=openai_model)

    return {
        "question": question,
        "rag_answer": rag_answer,
        "base_answer": base_answer,
        "sources": [
            {"filename": c["filename"], "chunk_id": c["chunk_id"], "score": c["score"]}
            for c in retrieved
        ]
    }
