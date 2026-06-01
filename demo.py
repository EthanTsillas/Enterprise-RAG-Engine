"""
demo.py

Demonstrates the RAG engine end-to-end:
  1. Load documents from docs/
  2. Chunk and embed
  3. Answer questions with retrieval augmentation
  4. A/B compare RAG vs base LLM (no context)

Usage:
    # With OpenAI (recommended):
    Add openai api key in a .env file
    python demo.py

    # Without OpenAI (retrieval-only mode):
    python demo.py --no-llm
"""

import argparse
from rag_engine import (
    load_documents,
    create_chunks,
    embed_chunks,
    retrieve,
    generate_answer,
    compare_rag_vs_base,
)
import os
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = "docs"
TOP_K = 3
MODEL_NAME = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4o-mini"

DEMO_QUESTIONS = [
    "How much did Atlantic cod populations decline in the Grand Banks?",
    "What percentage of the Mediterranean Sea is overfished?",
    "How does Global Fishing Watch detect illegal fishing vessels?",
]


def print_separator(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("\n" + "=" * width)


def run_demo(use_llm: bool = True) -> None:
    # -- Build the index --------------------------------------------------
    print_separator("Loading Documents")
    documents = load_documents(DOCS_FOLDER)
    print(f"Loaded {len(documents)} documents from '{DOCS_FOLDER}/'")

    print_separator("Chunking")
    chunks = create_chunks(documents, chunk_size=120, overlap=30)
    print(f"Created {len(chunks)} chunks  (chunk_size=120, overlap=30)")

    print_separator("Embedding")
    print(f"Encoding with '{MODEL_NAME}' ...")
    model, embeddings = embed_chunks(chunks, model_name=MODEL_NAME)
    print(f"Embedding matrix: {embeddings.shape}  "
          f"({embeddings.shape[0]} chunks x {embeddings.shape[1]} dims)")

    # -- Answer questions --------------------------------------------------
    print_separator("RAG Question Answering")
    for question in DEMO_QUESTIONS:
        print(f"\nQ: {question}")
        retrieved = retrieve(question, model, embeddings, chunks, top_k=TOP_K)

        print("  Sources:")
        for i, chunk in enumerate(retrieved, start=1):
            print(f"    [{i}] {chunk['filename']}  "
                  f"chunk {chunk['chunk_id']}  "
                  f"score={chunk['score']:.4f}")

        answer = generate_answer(
            question, retrieved,
            use_llm=use_llm,
            openai_model=OPENAI_MODEL
        )
        print(f"\n  Answer:\n  {answer}\n")

    # -- A/B comparison ---------------------------------------------------
    if use_llm:
        print_separator("A/B Comparison: RAG vs Base LLM")
        ab_question = "What caused the collapse of Atlantic cod in the Grand Banks?"
        print(f"\nQuestion: {ab_question}\n")
        result = compare_rag_vs_base(
            ab_question, model, embeddings, chunks,
            top_k=TOP_K, openai_model=OPENAI_MODEL
        )
        print("--- RAG Answer (with retrieved context) ---")
        print(result["rag_answer"])
        print("\n--- Base LLM Answer (no context) ---")
        print(result["base_answer"])
        print("\n--- Sources used for RAG ---")
        for s in result["sources"]:
            print(f"  {s['filename']}  chunk {s['chunk_id']}  score={s['score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine Demo")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in retrieval-only mode without calling the OpenAI API"
    )
    args = parser.parse_args()
    run_demo(use_llm=not args.no_llm)
