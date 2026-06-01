# Enterprise RAG Engine

A from-scratch Retrieval-Augmented Generation (RAG) system built in Python.

Answers questions over a private document corpus using dense vector retrieval and GPT-4o-mini generation — with **no vector database**, no LangChain, and no heavyweight dependencies. Every component from chunking to cosine similarity is implemented directly.

---

## What It Does

Standard LLMs hallucinate when asked about private or domain-specific knowledge. RAG solves this by retrieving relevant documents at query time and grounding the model's answer in real evidence.

```
User Question
     │
     ▼
Embed question with all-MiniLM-L6-v2
     │
     ▼
Cosine similarity search over pre-embedded document chunks (NumPy)
     │
     ▼
Top-k chunks injected into prompt context
     │
     ▼
GPT-4o-mini generates a grounded answer
     │
     ▼
Answer + sources returned to user
```

---

## Features

- **Document loading** — ingests any folder of `.txt` files
- **Overlapping chunking** — configurable chunk size and overlap to preserve context at boundaries
- **Dense retrieval** — cosine similarity computed from scratch over NumPy arrays, no FAISS or Pinecone required
- **OpenAI generation** — GPT-4o-mini generates grounded answers using retrieved context
- **Retrieval-only fallback** — runs without an OpenAI key, surfacing raw retrieved evidence
- **A/B comparison** — built-in method to run the same question through RAG vs base LLM and compare factual grounding
- **Modular architecture** — every component (loader, chunker, embedder, retriever, generator) is a standalone function; swap any piece independently

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/EthanTsillas/Enterprise-RAG-Engine.git
cd Enterprise-RAG-Engine
pip install -r requirements.txt
```

### 2. Add your documents

Place `.txt` files in the `docs/` folder. The engine will load all of them automatically.

```
docs/
├── report_q1.txt
├── product_spec.txt
├── meeting_notes.txt
└── ...
```

### 3. Set your OpenAI API key

```bash
inside .env file you create add:
OPENAI_API_KEY="your_api_key_here"
```

### 4. Run the demo

```bash
python demo.py
```

To run in retrieval-only mode without calling the OpenAI API:

```bash
python demo.py --no-llm
```

---

## Usage in Code

```python
from rag_engine import (
    load_documents,
    create_chunks,
    embed_chunks,
    retrieve,
    generate_answer,
    compare_rag_vs_base,
)

# Build the index
documents = load_documents("docs")
chunks = create_chunks(documents, chunk_size=120, overlap=30)
model, embeddings = embed_chunks(chunks)

# Ask a question
question = "What were the Q1 revenue figures?"
retrieved = retrieve(question, model, embeddings, chunks, top_k=3)
answer = generate_answer(question, retrieved)
print(answer)

# A/B: RAG vs base LLM
result = compare_rag_vs_base(question, model, embeddings, chunks)
print("RAG answer:", result["rag_answer"])
print("Base answer:", result["base_answer"])
```

---

## Project Structure

```
/
├── rag_engine.py      # Core engine — all RAG logic
├── demo.py            # End-to-end demo with A/B comparison
├── requirements.txt
├── docs/              # Your document corpus goes here
│   └── *.txt
├── .env               # used to keep Openai api key safe
├── .gitignore         # Ignore .env
└── README.md
```

---

## Design Decisions

**Why no vector database?**
FAISS, Pinecone, and Chroma are the right tool at scale. At 50-500 documents, a NumPy cosine similarity scan is faster to set up, has zero infrastructure overhead, and is easier to understand. Building without a vector DB also forces a deeper understanding of what retrieval is actually doing.

**Why overlapping chunks?**
Fixed-size chunks without overlap lose information at boundaries — a key sentence may be split across two chunks, degrading retrieval quality. Overlapping by 30 words ensures every sentence appears in full in at least one chunk.

**Why all-MiniLM-L6-v2?**
It is the best trade-off between embedding quality and inference speed for a local retrieval system. 384-dimensional embeddings, ~22M parameters, runs in milliseconds on CPU.

**Why GPT-4o-mini for generation?**
Strong instruction following, low latency, and low cost per query. The retrieval layer handles factual grounding; the generation model just needs to be a reliable summarizer.

---

## A/B Comparison — RAG vs Base LLM

The `compare_rag_vs_base()` function runs any question through both the RAG pipeline and the base LLM with no context, then returns both answers side by side.

This is useful for validating that retrieval is actually improving answer quality on your specific corpus. On domain-specific questions the RAG answer consistently includes specific facts from the documents that the base model cannot know.

---

## Built With

- [sentence-transformers](https://www.sbert.net/) — dense text embeddings
- [NumPy](https://numpy.org/) — cosine similarity and vector operations
- [OpenAI Python SDK](https://github.com/openai/openai-python) — GPT-4o-mini generation

---

## Author

**Ethan Tsillas**
github.com/EthanTsillas
linkedin.com/in/ethan-tsillas
