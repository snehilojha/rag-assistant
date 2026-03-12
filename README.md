# RAG Chatbot for "Data Science from Scratch"

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about Joel Grus's "Data Science from Scratch" book using semantic search and LLM generation.

## Project Overview

This project implements a complete RAG pipeline that:
1. Extracts and processes text from a PDF textbook
2. Chunks text using token-based splitting with overlap
3. Generates embeddings using SentenceTransformers
4. Indexes vectors with FAISS for fast similarity search
5. Retrieves relevant passages for user queries
6. Generates grounded answers using Google Gemini LLM
7. Provides a polished web interface for interaction

**Key Feature**: The system includes a comprehensive evaluation framework that compares different chunk sizes using standard IR metrics (P@1, P@3, P@5, MRR).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

1. INDEXING PHASE (Offline)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │  PDF Extract │ ───> │ Text Chunking│ ───> │  Embedding   │
   │  (PyMuPDF)   │      │ (Token-based)│      │ (MPNet-v2)   │
   └──────────────┘      └──────────────┘      └──────────────┘
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │ FAISS Index  │
                                                │ (IndexFlatIP)│
                                                └──────────────┘

2. RETRIEVAL PHASE (Online)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │ User Query   │ ───> │   Embed      │ ───> │   Search     │
   │              │      │   Query      │      │   Top-K      │
   └──────────────┘      └──────────────┘      └──────────────┘
                                                        │
                                                        ▼
3. GENERATION PHASE                             ┌──────────────┐
   ┌──────────────┐      ┌──────────────┐      │  Retrieved   │
   │ Build Prompt │ <─── │ Deduplicate  │ <─── │   Chunks     │
   │              │      │   Chunks     │      │              │
   └──────────────┘      └──────────────┘      └──────────────┘
          │
          ▼
   ┌──────────────┐      ┌──────────────┐
   │ Gemini LLM   │ ───> │   Answer     │
   │  Generation  │      │   Response   │
   └──────────────┘      └──────────────┘
```

---

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI**: Async REST API with CORS support
- **SentenceTransformers**: `all-mpnet-base-v2` for embeddings (768-dim)
- **FAISS**: Facebook AI Similarity Search for vector indexing
- **PyMuPDF (fitz)**: PDF text extraction
- **Google Gemini**: `gemini-2.5-flash` for answer generation

### Frontend
- **React 19**: Modern UI with hooks
- **Vite**: Fast build tool
- **Custom CSS**: GitHub-inspired dark theme

### Evaluation
- **Custom metrics**: Precision@K, Mean Reciprocal Rank
- **Keyword-based relevance**: Ground truth matching

---

## Evaluation Results

The system was evaluated on a test set of questions using keyword-based relevance matching across three chunk sizes:

| Chunk Size | P@1  | P@3  | P@5  | MRR  |
|-----------|------|------|------|------|
| 128       | 0.20 | 0.25 | 0.35 | 0.25 |
| 256       | 0.20 | 0.40 | 0.40 | 0.29 |
| **384**   | **0.40** | **0.55** | **0.55** | **0.45** |

**Key Findings**:
- **Larger chunks (384 tokens) perform best** across all metrics
- **2x improvement in P@1** for 384-token chunks vs smaller sizes
- **80% improvement in MRR** from 128 to 384 tokens (0.25 → 0.45)
- Trade-off: Larger chunks provide more context but may include irrelevant information

**Selected Configuration**: 384 tokens with 50-token overlap for optimal retrieval quality.

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ (for frontend)
- Google Gemini API key

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
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Build Indices
Place your PDF in `data/Data Science from Scratch by Joel Grus.pdf`, then:
```bash
python build_index.py
```

This will:
- Extract text from pages 12-322
- Create chunks of sizes 128, 256, and 384 tokens
- Generate embeddings and FAISS indices
- Save to `store/` directory

Expected output:
```
Built index for chunk size 128: 2847 chunks
Built index for chunk size 256: 1512 chunks
Built index for chunk size 384: 1048 chunks
```

#### Start Backend Server
```bash
uvicorn api:app --reload --port 8000
```

API will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd rag-frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`

---

## Project Structure

```
rag_assistant/
├── build_index.py          # PDF processing, chunking, and FAISS indexing
├── retriever.py            # Vector search and chunk retrieval
├── llm.py                  # LLM prompt building and generation
├── api.py                  # FastAPI backend server
├── evaluate.py             # Evaluation framework with IR metrics
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .gitignore             # Git ignore rules
│
├── data/                   # PDF source documents
│   └── Data Science from Scratch by Joel Grus.pdf
│
├── store/                  # Generated indices and chunks
│   ├── faiss_128.index
│   ├── faiss_256.index
│   ├── faiss_384.index
│   ├── chunks_128.json
│   ├── chunks_256.json
│   └── chunks_384.json
│
├── eval/                   # Evaluation data and results
│   ├── questions.json      # Test questions with keywords
│   └── results/
│       └── metrics.json    # Evaluation metrics output
│
└── rag-frontend/           # React frontend application
    ├── src/
    │   ├── App.jsx         # Main chat interface
    │   └── main.jsx        # React entry point
    ├── package.json
    └── vite.config.js
```

---

## Usage Examples

### Via Web Interface
1. Open `http://localhost:5173`
2. Type a question: *"How does gradient descent minimize a function?"*
3. View the generated answer grounded in retrieved passages
4. Click "View source chunks" to inspect retrieved context

### Via Python API
```python
from sentence_transformers import SentenceTransformer
from retriever import retrieve
from llm import ask_gemini

model = SentenceTransformer("all-mpnet-base-v2")
query = "What is linear regression?"

# Retrieve relevant chunks
chunks = retrieve(query, chunk_size=384, model=model, k=5)

# Generate answer
answer = ask_gemini(query, chunks)
print(answer)
```

### Via REST API
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?"}'
```

Response:
```json
{
  "answer": "Gradient descent is an optimization algorithm...",
  "chunks": ["...", "...", "..."]
}
```

---

## Running Evaluation

To evaluate retrieval quality across different chunk sizes:

```bash
python evaluate.py
```

Output:
```
Chunk Size  P@1     P@3     P@5     MRR
--------------------------------------------
128         0.20    0.25    0.35    0.25
256         0.20    0.40    0.40    0.29
384         0.40    0.55    0.55    0.45
```

Results are saved to `eval/results/metrics.json`

---

## Frontend Features

- **Real-time chat interface** with typing indicators
- **Chunk visualization drawer** to inspect retrieved passages
- **Responsive design** with smooth animations
- **GitHub-inspired dark theme** for developer aesthetics
- **Keyboard shortcuts**: Enter to send, Shift+Enter for newline

---

## Configuration

### Chunk Size Selection
Edit `build_index.py`:
```python
CHUNK_SIZES = [128, 256, 384]  # Token counts
OVERLAP = 50                    # Token overlap between chunks
```

### Retrieval Parameters
Edit `api.py`:
```python
chunks = retrieve(question.question, 384, model, k=5)  # Top-5 chunks
```

### LLM Model
Edit `llm.py`:
```python
MODEL = "gemini-2.5-flash"  # Or gemini-pro, gemini-2.0-flash-exp
```

---

## Performance Characteristics

- **Index build time**: ~2-3 minutes for 310-page PDF
- **Query latency**: 
  - Embedding: ~50ms
  - FAISS search: ~5ms
  - LLM generation: ~1-2s (depends on Gemini API)
  - **Total**: ~1.5-2.5s per query
- **Memory footprint**: ~500MB (model + indices)

---

## Known Limitations

1. **Single document only**: Currently hardcoded to one PDF
2. **No conversation history**: Each query is independent
3. **Naive deduplication**: Uses first 50 characters only
4. **No re-ranking**: Simple top-K retrieval without reordering
5. **Keyword-based evaluation**: Ground truth relies on manual keyword annotation

---

## Future Improvements

### High Priority
- [ ] Multi-document support with metadata filtering
- [ ] Conversation history and context management
- [ ] Hybrid search (BM25 + semantic)
- [ ] Re-ranking with cross-encoder
- [ ] Better deduplication using semantic similarity

### Medium Priority
- [ ] Query expansion and reformulation
- [ ] Streaming LLM responses
- [ ] Citation tracking (which chunk answered which part)
- [ ] User feedback loop for continuous improvement
- [ ] Docker containerization

### Low Priority
- [ ] Multi-modal support (images, tables from PDF)
- [ ] Fine-tuned embedding model on domain data
- [ ] Experiment tracking with MLflow
- [ ] A/B testing framework

---

## Experiment Log

See `experiments/EXPERIMENTS.md` for detailed analysis of:
- Chunk size comparison (128 vs 256 vs 384)
- Overlap sensitivity analysis
- Embedding model comparison
- Failure case analysis

---

## Contributing

This is a portfolio project, but suggestions are welcome! Please open an issue to discuss proposed changes.

---

## License

MIT License - feel free to use this as a learning resource or starting point for your own RAG projects.

---

## Acknowledgments

- **Joel Grus** for "Data Science from Scratch"
- **SentenceTransformers** for excellent embedding models
- **FAISS** team at Meta AI for fast vector search
- **Google** for Gemini API access

---

## Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**Built as a demonstration of RAG architecture and ML engineering best practices.**
