# Enterprise RAG Engine

A deployed Retrieval-Augmented Generation (RAG) API built with **Python, FastAPI, OpenAI embeddings, GPT-4o-mini, and NumPy**.

This project answers questions over a private 50-document corpus using dense vector retrieval and source-cited LLM generation. The retrieval layer is implemented from scratch with overlapping word-level chunking and raw NumPy cosine similarity — no vector database required.

## Live API

The project is deployed on Render with public Swagger/OpenAPI documentation:

```text
https://enterprise-rag-engine-zhef.onrender.com/docs
```

Main endpoint:

```text
POST /query
```

Example request:

```json
{
  "question": "What caused the collapse of Atlantic cod?",
  "top_k": 3,
  "use_llm": true
}
```

Example response shape:

```json
{
  "question": "What caused the collapse of Atlantic cod?",
  "answer": "...",
  "sources": [
    {
      "filename": "doc_02_atlantic_cod_collapse.txt",
      "chunk_id": 0,
      "score": 0.8421,
      "excerpt": "..."
    }
  ],
  "top_k": 3,
  "llm_used": true
}
```

---

## What It Does

Standard LLMs can hallucinate when asked about private or domain-specific information. This RAG system reduces that problem by retrieving relevant document chunks before generating an answer.

```text
User Question
     │
     ▼
Embed question with OpenAI text-embedding-3-small
     │
     ▼
Cosine similarity search over pre-embedded document chunks using NumPy
     │
     ▼
Select top-k most relevant chunks
     │
     ▼
Inject retrieved context into a GPT-4o-mini prompt
     │
     ▼
Return grounded answer + cited source chunks
```

The API can also run in **retrieval-only mode** by setting `use_llm` to `false`, returning raw retrieved evidence without calling the LLM.

---

## Features

- **FastAPI backend** with public Swagger/OpenAPI documentation
- **50-document demo corpus** focused on fisheries and ocean sustainability topics
- **Source-cited JSON responses** containing answer text, retrieved files, chunk IDs, similarity scores, and excerpts
- **OpenAI `text-embedding-3-small` embeddings** for lightweight cloud deployment
- **GPT-4o-mini generation** for grounded answer synthesis
- **No vector database dependency** — cosine similarity is implemented directly with NumPy
- **Overlapping word-level chunking** with configurable `chunk_size` and `overlap`
- **Retrieval-only fallback mode** using `use_llm=false`
- **Pydantic request/response schemas** for typed API inputs and outputs
- **Health and stats endpoints** for basic production observability
- **Environment-variable configuration** for embedding model, LLM model, chunk size, overlap, top-k, and docs folder
- **A/B comparison harness** to compare RAG-grounded answers against base GPT-4o-mini answers without retrieved context

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns API liveness status |
| `GET` | `/stats` | Returns document count, chunk count, embedding dimensions, embedding model, and docs folder |
| `POST` | `/query` | Retrieves relevant chunks and returns a grounded answer with sources |

### Query Parameters

`POST /query` accepts:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `question` | string | required | Natural-language question about the document corpus |
| `top_k` | integer | `3` | Number of chunks to retrieve, clamped between 1 and 10 |
| `use_llm` | boolean | `true` | If `false`, returns retrieved evidence without LLM generation |

---

## Core Pipeline

### 1. Document Loading

The engine loads every `.txt` file from the configured `docs/` folder.

```python
load_documents("docs")
```

Each document is stored with its filename and text content.

### 2. Chunking

Documents are split into overlapping word-level chunks.

```python
create_chunks(documents, chunk_size=120, overlap=30)
```

The overlap helps preserve context across chunk boundaries so important sentences are less likely to be split away from their surrounding meaning.

### 3. Embedding

Chunks are embedded with OpenAI `text-embedding-3-small`.

```python
embed_chunks(chunks, model_name="text-embedding-3-small")
```

The deployed version uses API-based embeddings instead of a local embedding model to keep memory usage low on Render.

### 4. Retrieval

The question is embedded with the same embedding model, then compared against the chunk embedding matrix using raw NumPy cosine similarity.

```python
retrieve(question, model, embeddings, chunks, top_k=3)
```

No FAISS, Pinecone, Chroma, or LangChain vector store is required.

### 5. Generation

The top retrieved chunks are injected into a prompt and passed to GPT-4o-mini.

```python
generate_answer(question, retrieved_chunks, use_llm=True)
```

The prompt instructs the model to answer only from the retrieved context and to say when the documents do not contain enough information.

---

## Project Structure

```text
Enterprise-RAG-Engine/
├── api.py              # FastAPI REST API with /query, /health, and /stats
├── rag_engine.py       # Core RAG logic: loading, chunking, embeddings, retrieval, generation
├── demo.py             # CLI demo with RAG answers and A/B comparison
├── requirements.txt
├── runtime.txt
├── docs/               # 50-document text corpus
│   └── *.txt
├── .env                # Local environment variables, not committed
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/EthanTsillas/Enterprise-RAG-Engine.git
cd Enterprise-RAG-Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
EMBED_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-4o-mini
DOCS_FOLDER=docs
CHUNK_SIZE=120
OVERLAP=30
TOP_K=3
```

Only `OPENAI_API_KEY` is required for LLM generation. The other values have defaults in the code.

---

## Run Locally

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

Then open the Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Run a local query:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What caused the collapse of Atlantic cod?","top_k":3,"use_llm":true}'
```

Run in retrieval-only mode:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What caused the collapse of Atlantic cod?","top_k":3,"use_llm":false}'
```

---

## CLI Demo

The repo also includes a command-line demo.

```bash
python demo.py
```

Run without LLM generation:

```bash
python demo.py --no-llm
```

The demo loads the documents, chunks them, embeds them, retrieves relevant sources, generates RAG-grounded answers, and then runs an A/B comparison between the RAG answer and a base GPT-4o-mini answer with no retrieved context.

---

## Design Decisions

### Why OpenAI embeddings?

The deployed API uses `text-embedding-3-small` because it keeps the Render deployment lightweight. Instead of loading a local sentence-transformer model into memory, the API calls OpenAI for embedding generation and stores the resulting vectors in a NumPy matrix.

### Why no vector database?

For a small-to-medium document corpus, a NumPy cosine similarity scan is simple, transparent, and easy to deploy. Vector databases like FAISS, Pinecone, and Chroma are useful at larger scale, but this project intentionally implements retrieval directly to show the mechanics behind RAG.

### Why overlapping chunks?

Fixed-size chunks can split important information across boundaries. A 30-word overlap helps preserve context so retrieval has a better chance of selecting complete evidence.

### Why retrieval-only mode?

The API can return raw retrieved evidence without LLM generation. This is useful for debugging retrieval quality, reducing cost, and keeping the system functional when LLM generation is disabled.

### Why A/B comparison?

The `compare_rag_vs_base()` function runs the same question through the RAG pipeline and a base GPT-4o-mini call with no context. This helps validate whether retrieval is actually improving factual grounding on the document corpus.

---

## Resume Summary

Built and deployed a FastAPI RAG service on Render that answers questions over a 50-document corpus with source-cited JSON responses. Implemented overlapping chunking, OpenAI `text-embedding-3-small` embeddings, raw NumPy cosine similarity retrieval, GPT-4o-mini answer generation, typed Pydantic API schemas, retrieval-only fallback mode, `/health` and `/stats` endpoints, and an A/B evaluation harness comparing RAG-grounded answers against base LLM answers.

---

## Built With

- Python
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- OpenAI Python SDK
- GPT-4o-mini
- OpenAI `text-embedding-3-small`
- Render

---

## Author

**Ethan Tsillas**  
GitHub: [github.com/EthanTsillas](https://github.com/EthanTsillas)  
LinkedIn: [linkedin.com/in/ethan-tsillas](https://www.linkedin.com/in/ethan-tsillas)
