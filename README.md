# Enterprise RAG Engine

website link: https://enterprise-rag-engine-zhef.onrender.com/docs

A lightweight Retrieval-Augmented Generation (RAG) system built with Python, NumPy, OpenAI embeddings, and FastAPI.

The project answers questions over a private `.txt` document corpus by loading documents, splitting them into overlapping chunks, embedding those chunks with OpenAI `text-embedding-3-small`, retrieving the most relevant chunks with cosine similarity, and generating grounded answers with `gpt-4o-mini`.

It does **not** require LangChain, Pinecone, Chroma, FAISS, or a separate vector database.

---

## What It Does

Standard LLMs can hallucinate when asked about private or domain-specific knowledge. This engine reduces that problem by retrieving relevant document chunks first, then forcing the model to answer using only that retrieved context.

```text
User Question
     │
     ▼
Load private .txt documents from docs/
     │
     ▼
Split documents into overlapping chunks
     │
     ▼
Embed chunks with OpenAI text-embedding-3-small
     │
     ▼
Compute cosine similarity with NumPy
     │
     ▼
Retrieve the top-k most relevant chunks
     │
     ▼
Inject retrieved chunks into a grounded prompt
     │
     ▼
gpt-4o-mini generates an answer
     │
     ▼
Return answer + source filenames, chunk IDs, scores, and excerpts
```

---

## Features

- **Private document Q&A** — ask questions over local `.txt` files in the `docs/` folder.
- **OpenAI embeddings** — uses `text-embedding-3-small` for lightweight hosted embeddings.
- **Grounded answer generation** — uses `gpt-4o-mini` with retrieved context only.
- **FastAPI REST API** — includes `/query`, `/health`, and `/stats` endpoints.
- **Retrieval-only fallback** — set `use_llm=false` or run `demo.py --no-llm` to return raw retrieved evidence instead of an LLM-written answer.
- **Source citations** — API responses include source filename, chunk ID, similarity score, and an excerpt.
- **Configurable retrieval** — change docs folder, embedding model, OpenAI model, chunk size, overlap, and top-k through environment variables.
- **No vector database** — cosine similarity search is implemented directly with NumPy.
- **A/B comparison utility** — `compare_rag_vs_base()` compares a RAG-grounded answer against a base LLM answer with no retrieved context.

---

## Project Structure

```text
Enterprise-RAG-Engine/
├── api.py              # FastAPI server for RAG queries
├── rag_engine.py       # Core document loading, chunking, embedding, retrieval, and generation logic
├── demo.py             # Command-line demo and RAG vs base LLM comparison
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime version for deployment platforms
├── docs/               # Private text document corpus
│   └── *.txt
├── .gitignore          # Ignores .env and __pycache__
└── README.md
```

---

## Requirements

- Python 3.11+
- OpenAI API key
- `.txt` documents inside the `docs/` folder

Install dependencies:

```bash
pip install -r requirements.txt
```

Current dependencies:

```text
fastapi
uvicorn
pydantic
python-dotenv
numpy>=1.26.0
openai>=1.30.0
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your_api_key_here"
```

Optional configuration variables used by `api.py`:

```env
DOCS_FOLDER="docs"
EMBED_MODEL="text-embedding-3-small"
OPENAI_MODEL="gpt-4o-mini"
CHUNK_SIZE="120"
OVERLAP="30"
TOP_K="3"
```

Defaults are already set in `api.py`, so only `OPENAI_API_KEY` is required for normal usage.

---

## Quickstart: Run the API

### 1. Clone the repository

```bash
git clone https://github.com/EthanTsillas/Enterprise-RAG-Engine.git
cd Enterprise-RAG-Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.env` file:

```env
OPENAI_API_KEY="your_api_key_here"
```

### 4. Add documents

Place `.txt` files inside the `docs/` folder:

```text
docs/
├── report_q1.txt
├── product_spec.txt
├── meeting_notes.txt
└── ...
```

The included sample corpus contains text files about fisheries, overfishing, marine ecosystems, and seafood policy.

### 5. Start the FastAPI server

```bash
uvicorn api:app --reload
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### `GET /health`

Simple liveness check.

Example response:

```json
{
  "status": "ok"
}
```

---

### `GET /stats`

Builds the index if needed and returns document/index statistics.

Example response:

```json
{
  "document_count": 50,
  "chunk_count": 50,
  "embedding_dimensions": 1536,
  "embedding_model": "text-embedding-3-small",
  "docs_folder": "docs"
}
```

---

### `POST /query`

Ask a question about the document corpus.

Request body:

```json
{
  "question": "What caused the collapse of Atlantic cod?",
  "top_k": 3,
  "use_llm": true
}
```

Fields:

- `question` — natural language question about the document corpus.
- `top_k` — number of chunks to retrieve. Defaults to `3`; API clamps values between `1` and `10`.
- `use_llm` — when `true`, generates a final answer with OpenAI chat completions. When `false`, returns retrieved evidence directly.

Example response shape:

```json
{
  "question": "What caused the collapse of Atlantic cod?",
  "answer": "...",
  "sources": [
    {
      "filename": "doc_02_atlantic_cod_collapse.txt",
      "chunk_id": 0,
      "score": 0.8123,
      "excerpt": "..."
    }
  ],
  "top_k": 3,
  "llm_used": true
}
```

---

## Example API Calls

Using `curl`:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What percentage of the Mediterranean Sea is overfished?",
    "top_k": 3,
    "use_llm": true
  }'
```

Retrieval-only mode through the API:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does Global Fishing Watch detect illegal fishing vessels?",
    "top_k": 3,
    "use_llm": false
  }'
```

---

## Run the Command-Line Demo

The demo loads the documents, chunks them, embeds them, retrieves sources, answers sample questions, and optionally compares RAG output against a base LLM answer.

```bash
python demo.py
```

Retrieval-only demo mode:

```bash
python demo.py --no-llm
```

The included demo questions are based on the sample fisheries documents in `docs/`.

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
model, embeddings = embed_chunks(chunks, model_name="text-embedding-3-small")

# Ask a question
question = "What caused the collapse of Atlantic cod in the Grand Banks?"
retrieved = retrieve(question, model, embeddings, chunks, top_k=3)
answer = generate_answer(question, retrieved, use_llm=True, openai_model="gpt-4o-mini")

print(answer)

# A/B comparison: RAG vs base LLM
result = compare_rag_vs_base(question, model, embeddings, chunks)
print("RAG answer:", result["rag_answer"])
print("Base answer:", result["base_answer"])
print("Sources:", result["sources"])
```

---

## How the Engine Works

### 1. Document loading

`load_documents(folder_path)` reads every `.txt` file from the selected folder and returns a list of dictionaries:

```python
{"filename": "doc_01.txt", "text": "..."}
```

### 2. Chunking

`chunk_text()` splits each document into overlapping word chunks. By default, chunks are `120` words with `30` words of overlap.

This helps preserve context when important information appears near a chunk boundary.

### 3. Embedding

`embed_chunks()` sends all chunk text to OpenAI's embeddings API using `text-embedding-3-small` by default.

The result is stored as a NumPy matrix.

### 4. Retrieval

`retrieve()` embeds the user question with the same OpenAI embedding model, computes cosine similarity against the chunk matrix, sorts by score, and returns the top-k chunks.

### 5. Prompting

`build_prompt()` creates a grounded prompt that instructs the model to answer using only the retrieved context.

If the answer is not present in the context, the model is instructed to say:

```text
I do not have enough information in the provided documents to answer this.
```

### 6. Answer generation

`generate_answer()` calls OpenAI chat completions with `gpt-4o-mini` by default.

If `use_llm=False`, or if the OpenAI call fails, it returns retrieved evidence directly.

---

## Design Decisions

### Why no vector database?

For small and medium document sets, a NumPy cosine similarity scan is simple, transparent, and deployment-friendly. It avoids extra infrastructure while still demonstrating the core retrieval mechanics behind RAG.

### Why OpenAI embeddings?

The current code uses `text-embedding-3-small` to keep deployment lightweight. This avoids bundling local embedding models, which can be too large for small free-tier deployments.

### Why chunk overlap?

Overlapping chunks reduce the chance that an important sentence gets split across two chunks and lost during retrieval.

### Why FastAPI?

FastAPI makes the RAG engine easy to test locally through `/docs`, easy to call from a frontend, and easy to deploy as a REST API.

### Why retrieval-only mode?

Retrieval-only mode is useful for debugging. It shows exactly which chunks were selected before any LLM rewriting happens.

---

## Deployment Notes

This project includes `runtime.txt` for platforms that read Python runtime versions.

For Render or similar hosts, a typical start command is:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Make sure the deployment environment has this variable set:

```env
OPENAI_API_KEY="your_api_key_here"
```

Optional production variables:

```env
DOCS_FOLDER="docs"
EMBED_MODEL="text-embedding-3-small"
OPENAI_MODEL="gpt-4o-mini"
CHUNK_SIZE="120"
OVERLAP="30"
TOP_K="3"
```

---

## Limitations

- Only `.txt` files are loaded by default.
- The index is built in memory when first needed and is not persisted to disk.
- Large corpora may require batching, caching, persistence, or a real vector database.
- Embedding calls require an OpenAI API key.
- The API currently rebuilds the index only when the server process restarts.

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [Uvicorn](https://www.uvicorn.org/) — ASGI server
- [OpenAI Python SDK](https://github.com/openai/openai-python) — embeddings and chat completions
- [NumPy](https://numpy.org/) — cosine similarity and vector operations
- [python-dotenv](https://pypi.org/project/python-dotenv/) — local environment variable loading
- [Pydantic](https://docs.pydantic.dev/) — request and response schemas

---

## Author

**Ethan Tsillas**

- GitHub: github.com/EthanTsillas
- LinkedIn: linkedin.com/in/ethan-tsillas
