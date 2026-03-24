# RAG Study Assistant

A multi-book Retrieval-Augmented Generation (RAG) system that answers questions grounded in your personal PDF library using semantic search and OpenAI GPT-4o mini.

## Project Overview

This project implements a complete RAG pipeline that:
1. Ingests PDF books via a CLI tool — extract, clean, chunk, embed, index
2. Chunks text using a global token-stream sliding window with overlap
3. Generates embeddings using `BAAI/bge-base-en-v1.5` (BGE asymmetric encoding)
4. Indexes vectors per-book with FAISS `IndexIVFFlat` for fast similarity search
5. Retrieves across multiple books using Reciprocal Rank Fusion (RRF)
6. Generates grounded answers with source citations via OpenAI GPT-4o mini
7. Serves a polished React chat interface with Markdown rendering and a book selector

**Key Design**: One FAISS index per book in `store/{slug}/`. Per-book indices are exact, composable, and avoid the over-retrieval required by post-filtering on a merged index.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

1. INDEXING PHASE (Offline — ingest.py)
   ┌──────────────┐      ┌──────────────┐      ┌────────────────────┐
   │  PDF Extract │ ───> │ Token-stream │ ───> │  BGE Embedding     │
   │  (PyMuPDF)   │      │   Chunking   │      │ (bge-base-en-v1.5) │
   └──────────────┘      └──────────────┘      └────────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  FAISS Index     │
                                                │ (IndexIVFFlat)   │
                                                │  store/{slug}/   │
                                                └──────────────────┘

2. RETRIEVAL PHASE (Online — retriever.py)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
   │  User Query  │ ───> │ BGE Query    │ ───> │  Per-book FAISS  │
   │              │      │  Prefix      │      │  Search (Top-K)  │
   └──────────────┘      └──────────────┘      └──────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  RRF Merge       │
                                                │  (multi-book)    │
                                                └──────────────────┘

3. GENERATION PHASE (Online — llm.py)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
   │ Build Prompt │ <─── │ Deduplicate  │ <─── │ Top-K Chunks     │
   │ (with source │      │   Chunks     │      │ (book + page)    │
   │  citations)  │      └──────────────┘      └──────────────────┘
   └──────────────┘
          │
          ▼
   ┌──────────────┐      ┌──────────────┐
   │ GPT-4o mini  │ ───> │ Cited Answer │
   │  (OpenAI)    │      │  Response    │
   └──────────────┘      └──────────────┘
```

---

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI**: Async REST API with CORS support
- **SentenceTransformers**: `BAAI/bge-base-en-v1.5` for embeddings (768-dim, asymmetric)
- **FAISS**: Per-book `IndexIVFFlat` indices with serialized `nprobe`
- **PyMuPDF (fitz)**: PDF text extraction
- **OpenAI**: `gpt-4o-mini` for grounded answer generation
- **python-dotenv**: Environment variable management

### Frontend
- **React 19** + **Vite**: Fast modern UI
- **react-markdown** + **remark-gfm**: Full Markdown rendering with GFM support
- **Custom CSS**: GitHub-inspired dark theme with IBM Plex / Space Mono fonts

### Evaluation
- **Custom IR metrics**: Precision@K, Mean Reciprocal Rank (MRR)
- **Keyword-based relevance**: Ground truth matching via stem-truncated keywords

---

## Evaluation Results

Metrics use keyword-based relevance matching (chunk must contain all annotated keywords). Test set: 20 questions. Full analysis in `experiments/EXPERIMENTS.md`.

### Experiment 1 — Chunk Size Comparison
*Model: `all-mpnet-base-v2`, overlap: 50 tokens*

| Chunk Size | P@1  | P@3  | P@5  | MRR  |
|-----------|------|------|------|------|
| 128       | 0.20 | 0.25 | 0.35 | 0.25 |
| 256       | 0.20 | 0.40 | 0.40 | 0.29 |
| **384**   | **0.40** | **0.55** | **0.55** | **0.45** |

**Finding**: 384-token chunks are 2× better on P@1 and 80% better on MRR than 128-token chunks.

---

### Experiment 2 — Overlap Sensitivity
*Model: `all-mpnet-base-v2`, chunk size: 384 tokens*

| Overlap | # Chunks | P@1  | P@3  | P@5  | MRR  |
|---------|----------|------|------|------|------|
| 0       | 312      | 0.35 | **0.60** | **0.60** | **0.47** |
| 25      | 334      | 0.35 | 0.55 | 0.55 | 0.44 |
| **50**  | **359**  | **0.40** | 0.55 | 0.55 | 0.45 |
| 100     | 422      | 0.30 | 0.55 | 0.55 | 0.42 |

**Finding**: No overlap wins on P@3/P@5/MRR (cleaner top-K diversity); 50-token overlap wins on P@1 (best single result). P@1 prioritised — **50 tokens selected**. 100-token overlap strictly worse across all metrics.

---

### Architecture Upgrades Applied After Baseline

| Change | Old | New | Rationale |
|--------|-----|-----|-----------|
| Embedding model | `all-mpnet-base-v2` | `BAAI/bge-base-en-v1.5` | MTEB SOTA for retrieval; asymmetric query/passage encoding |
| Chunk size | 384 tokens | 508 tokens | BGE full context window (512 − 4 special tokens) |
| FAISS index | `IndexFlatIP` (single merged) | `IndexIVFFlat` (per-book) | Scales to multi-book; <5% accuracy loss with tuned `nprobe` |
| Retrieval | Single index top-K | Multi-book RRF fusion | Composable, exact per-book filtering |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- OpenAI API key

### 1. Clone Repository
```bash
git clone <repository-url>
cd rag_assistant
```

### 2. Backend Setup

#### Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

#### Configure Environment
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

#### Ingest a Book
```bash
python ingest.py \
  --pdf "data/Data Science from Scratch by Joel Grus.pdf" \
  --name "Data Science from Scratch" \
  --author "Joel Grus" \
  --start-page 11 \
  --end-page 321
```

This will:
- Extract and clean text from the specified page range
- Chunk via a global token-stream sliding window
- Embed with `BAAI/bge-base-en-v1.5`
- Build a FAISS `IndexIVFFlat` (or `IndexFlatIP` for small corpora)
- Save to `store/{slug}/` and update `store/registry.json`

Repeat for each book you want to index. Each book gets its own isolated index.

#### Start Backend Server
```bash
uvicorn api:app --reload --port 8000
```

API available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd rag-frontend
npm install
npm run dev
```

Frontend available at `http://localhost:5173`

---

## Project Structure

```
rag_assistant/
├── ingest.py               # PDF ingestion CLI (extract → chunk → embed → index)
├── retriever.py            # Multi-book FAISS search + RRF fusion
├── llm.py                  # Prompt building and OpenAI generation
├── api.py                  # FastAPI server (/ask, /books)
├── evaluate.py             # IR evaluation framework (P@K, MRR)
├── run_overlap_experiments.py  # Chunk overlap sensitivity experiments
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── data/                   # PDF source documents (gitignored)
│
├── store/                  # Generated indices (gitignored)
│   ├── registry.json       # Book registry (slug, title, author)
│   └── {slug}/
│       ├── faiss.index     # Per-book IVFFlat index (nprobe serialized)
│       ├── chunks.json     # Chunk dicts (text, book, slug, page, chunk_id)
│       └── meta.json       # Book metadata
│
├── experiments/
│   ├── EXPERIMENTS.md      # Experiment log and analysis
│   └── results/
│
└── rag-frontend/           # React frontend
    ├── src/
    │   ├── App.jsx         # Chat UI (Markdown, BookSelector, ChunkDrawer)
    │   └── main.jsx        # React entry point
    ├── package.json
    └── vite.config.js
```

---

## API Reference

### `POST /ask`

Retrieve relevant chunks and generate a grounded answer.

**Request**
```json
{
  "question": "What is gradient descent?",
  "sources": ["data-science-from-scratch"]
}
```
`sources` is optional — omit or set to `null` to search all indexed books.

**Response**
```json
{
  "answer": "Gradient descent is an optimization algorithm...\n\n*Source: Data Science from Scratch — Page 94*",
  "chunks": [
    { "text": "...", "book": "Data Science from Scratch", "page": 94, "chunk_id": "data-science-from-scratch:312" }
  ]
}
```

**Error codes**: `400` bad input / unknown source slug · `500` retrieval failure · `502` OpenAI API error

### `GET /books`

Returns metadata for all indexed books.

```json
[
  { "title": "Data Science from Scratch", "author": "Joel Grus", "slug": "data-science-from-scratch", "total_chunks": 1048 }
]
```

---

## Usage Examples

### Via Web Interface
1. Open `http://localhost:5173`
2. Select source books via the chip selector (or leave **All** active)
3. Type a question: *"How does gradient descent minimize a function?"*
4. Answers render with full Markdown — headers, code blocks, bold, lists
5. Click **View N source chunks →** to inspect retrieved passages with book and page metadata

### Via REST API
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?"}'
```

### Via Python
```python
from sentence_transformers import SentenceTransformer
from retriever import load_all_indices, retrieve
from llm import ask_llm

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
indices = load_all_indices("store")

chunks = retrieve("What is linear regression?", model, indices, top_k=5)
answer = ask_llm("What is linear regression?", chunks)
print(answer)
```

---

## Running Evaluation

```bash
python evaluate.py
```

Results are printed to stdout and optionally saved to `experiments/results/`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| One FAISS index per book | Exact per-book filtering; no over-retrieval or post-filtering needed |
| `IndexIVFFlat` over `IndexFlatIP` | 10–100x faster at scale (~75k vectors), <5% accuracy loss with tuned `nprobe` |
| `nprobe` set at build time | Serialized by `faiss.write_index`; overriding on load would silently break calibration |
| BGE asymmetric query prefix | Required by `bge-base-en-v1.5`; omitting it causes silent retrieval degradation |
| Namespaced `chunk_id` (`{slug}:{n}`) | Prevents RRF deduplication collisions across books (bare counters restart at 0) |
| RRF fusion (`k=60`) | Standard default; composes per-book rankings without score normalization |
| Global token-stream chunking | Preserves semantic overlap across page boundaries |

---

## Frontend Features

- **Markdown rendering**: Full GFM support via `react-markdown` + `remark-gfm` — code blocks, tables, bold, lists
- **Book selector**: Pill chips to filter queries to specific books; "All" queries everything
- **Chunk drawer**: Slide-in panel showing retrieved passages with book title and page number
- **Typing indicator**: Animated dots while awaiting response
- **Keyboard shortcuts**: Enter to send, Shift+Enter for newline

---

## Performance Characteristics

- **Index build time**: ~3-5 minutes per 300-page PDF (embedding-dominated)
- **Query latency**:
  - Embedding: ~50ms
  - FAISS search (IVFFlat): ~5ms per book
  - LLM generation: ~1-2s (GPT-4o mini)
  - **Total**: ~1.5-2.5s per query
- **Memory footprint**: ~500MB (BGE model + indices)

---

## Known Limitations

1. **No conversation history**: Each query is independent (planned: multi-turn + SQLite)
2. **Naive deduplication**: Prefix matching (planned: embedding-based)
3. **No re-ranking**: Simple RRF top-K (planned: cross-encoder)
4. **Keyword-based evaluation**: Ground truth relies on manual keyword annotation
5. **No streaming**: Full response before any text is shown (planned)

---

## Roadmap

- [ ] Multi-turn conversation memory with SQLite
- [ ] Streaming LLM responses
- [ ] Hybrid search (BM25 + semantic)
- [ ] Cross-encoder re-ranking
- [ ] `/health` endpoint
- [ ] Docker containerization

---

## Experiment Log

See `experiments/EXPERIMENTS.md` for analysis of chunk size, overlap sensitivity, and failure cases.

---

## License

MIT License — free to use as a learning resource or starting point for your own RAG projects.

---

## Acknowledgments

- **BAAI** for `bge-base-en-v1.5`
- **SentenceTransformers** for the embedding library
- **FAISS** team at Meta AI for fast vector search
- **OpenAI** for GPT-4o mini

---

**Built as a demonstration of RAG architecture and ML engineering best practices.**
