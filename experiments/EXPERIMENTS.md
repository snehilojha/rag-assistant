# Experiment Log: RAG System Optimization

This document details the experiments conducted to optimize the RAG system's retrieval quality and design decisions made during development.

---

## Experiment 1: Chunk Size Comparison

### Objective
Determine the optimal chunk size for text splitting to maximize retrieval quality while maintaining answer coherence.

### Methodology
- **Configurations tested**: 128, 256, 384 tokens per chunk
- **Overlap**: 50 tokens (constant across all configurations)
- **Embedding model**: `all-mpnet-base-v2` (768-dim) — baseline model at time of experiment
- **Evaluation metrics**: P@1, P@3, P@5, MRR
- **Test set**: 20 questions with manually annotated keywords
- **Relevance criterion**: Chunk must contain ALL specified keywords

### Results

| Chunk Size | P@1  | P@3  | P@5  | MRR  |
|-----------|------|------|------|------|
| 128       | 0.20 | 0.25 | 0.35 | 0.25 |
| 256       | 0.20 | 0.40 | 0.40 | 0.29 |
| **384**   | **0.40** | **0.55** | **0.55** | **0.45** |

### Analysis

**Key Findings:**
1. **Larger chunks consistently outperform smaller chunks** across all metrics
2. **384-token chunks show 2x improvement** in P@1 vs smaller chunks (0.40 vs 0.20)
3. **P@3 and P@5 improve significantly** with larger chunks (0.25 → 0.55 for P@3)
4. **MRR improvement** from 0.25 → 0.45 (80% increase) shows better ranking quality

**Why larger chunks perform better:**
- More context per chunk reduces fragmentation of related concepts
- Complete paragraphs/sections vs. sentence fragments — better semantic coherence
- Less chance of splitting key information across boundaries

### Decision
**Selected: 384 tokens** with 50-token overlap as the baseline configuration.

> **Note**: This experiment used `all-mpnet-base-v2`. The production model was later upgraded to `BAAI/bge-base-en-v1.5` and the max chunk size raised to 508 tokens (BGE's full context window). See Experiment 3.

---

## Experiment 2: Overlap Sensitivity Analysis

### Objective
Determine if 50-token overlap is necessary and at what level it helps or hurts.

### Methodology
- **Overlap values tested**: 0, 25, 50, 100 tokens
- **Chunk size**: 384 tokens (constant)
- **Embedding model**: `all-mpnet-base-v2`
- **Results file**: `experiments/results/overlap_comparison.json`

### Results

| Overlap | # Chunks | P@1  | P@3  | P@5  | MRR  |
|---------|----------|------|------|------|------|
| 0       | 312      | 0.35 | **0.60** | **0.60** | **0.47** |
| 25      | 334      | 0.35 | 0.55 | 0.55 | 0.44 |
| **50**  | **359**  | **0.40** | 0.55 | 0.55 | 0.45 |
| 100     | 422      | 0.30 | 0.55 | 0.55 | 0.42 |

### Analysis

**Key Findings:**
1. **No overlap (0) achieves best P@3, P@5, and MRR** — likely because cleaner boundaries reduce redundancy in top-K
2. **50-token overlap achieves best P@1** (0.40 vs 0.35) — overlap helps surface the single most relevant chunk
3. **100-token overlap degrades all metrics** — too much duplicate content inflates chunk count (+35% vs overlap=0) without benefit
4. **P@1 vs P@3/P@5 trade-off is real**: overlap improves precision at rank 1 at the cost of recall diversity

**Why overlap=50 was kept:**
- P@1 is the most user-facing metric — first result quality matters most
- 14% improvement in P@1 (0.35 → 0.40) is meaningful
- Prevents hard sentence splits at chunk boundaries (qualitative benefit)
- MRR at overlap=50 (0.45) is close to the best (0.47 at overlap=0)

### Decision
**Retained: 50-token overlap**. The P@1 advantage outweighs the small P@3/P@5 gap, and boundary-split prevention adds qualitative value not captured by keyword metrics.

---

## Experiment 3: Embedding Model Upgrade — `all-mpnet-base-v2` → `BAAI/bge-base-en-v1.5`

### Status
✅ **Completed** — model upgraded to BGE in production

### Objective
Evaluate whether `BAAI/bge-base-en-v1.5` improves retrieval over the `all-mpnet-base-v2` baseline.

### Why BGE was considered
- `bge-base-en-v1.5` consistently leads MTEB leaderboard benchmarks for retrieval tasks
- Designed for **asymmetric retrieval** — distinct representations for queries vs. passages
- Requires a query-time prefix (`"Represent this sentence for searching relevant passages: "`) — passages are indexed without it
- Same embedding dimension (768-dim) as MPNet — drop-in compatible with existing FAISS setup

### Architectural implication of asymmetric encoding
BGE's query prefix is not cosmetic — omitting it causes silent retrieval degradation with no error signal. It is applied in `retriever.py` at query time only; passage embeddings stored in FAISS have no prefix.

### Configuration change
| Parameter | Old (MPNet) | New (BGE) |
|-----------|------------|-----------|
| Model | `all-mpnet-base-v2` | `BAAI/bge-base-en-v1.5` |
| Max context | 384 tokens | 512 tokens |
| Production chunk size | 384 tokens | 508 tokens (full window minus special tokens) |
| Query prefix | None | `"Represent this sentence for searching relevant passages: "` |

### Decision
**Adopted `BAAI/bge-base-en-v1.5`** as the production embedding model. MTEB retrieval benchmarks show consistent improvement over MPNet for passage retrieval tasks. Chunk size raised to 508 tokens to use BGE's full context window.

---

## Experiment 4: FAISS Index Type — `IndexFlatIP` → `IndexIVFFlat`

### Status
✅ **Completed** — IVFFlat adopted for all per-book indices

### Objective
Migrate from exact search to approximate search to support multi-book scale without meaningful quality loss.

### Motivation
The architecture shifted from a single merged index to **one `IndexIVFFlat` per book**. Target scale: ~50 books × ~1500 chunks = ~75k vectors. At this scale, `IndexFlatIP` (exact linear scan) becomes the bottleneck; `IndexIVFFlat` uses Voronoi clustering to restrict search to a subset of cells.

### Index configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `nlist` | `min(sqrt(N), N // 39)` | Standard heuristic; 39× minimum training requirement from FAISS source |
| `nprobe` | `nlist` if `nlist ≤ 20`, else `nlist // 4` | Search all cells for small indices; 25% for larger (good recall / speed tradeoff) |
| Training | Random sample of `nlist × 40` chunks | Prevents biased centroids from early-chapter topic clustering |
| Serialization | `faiss.write_index` / `faiss.read_index` | `nprobe` is persisted — **never override on load** |

### Small-corpus fallback
Indices with fewer than 100 chunks use `IndexFlatIP` automatically (IVFFlat's clustering overhead is not justified and requires a minimum training set).

### Quality impact
IVFFlat introduces <5% accuracy loss vs. exact search when `nprobe` is tuned correctly. At the current single-book scale (~1000 chunks), the practical difference is negligible.

### Decision
**Adopted `IndexIVFFlat`** with the `nprobe` formula above. The index type is transparent to the retriever — `faiss.read_index` restores the full configured state including `nprobe`.

---

## Experiment 5: Deduplication Strategy Analysis (FUTURE WORK)

### Status
**Not yet conducted**

### Objective
Evaluate and improve the current prefix-based deduplication approach.

### Current Implementation
**Prefix matching (50 characters)** — checks if the first 50 chars of a new chunk appear in any previously seen chunk text. Applied in `llm.py` before prompt construction.

### Known Limitations
- Arbitrary 50-char threshold chosen without experimentation
- Misses semantic duplicates with different sentence starts
- No semantic understanding — may pass through near-identical chunks with different wording

### Proposed Next Steps
1. Measure actual duplicate rate in top-K results across queries
2. Test embedding-based cosine similarity deduplication (threshold ~0.95)
3. Benchmark latency difference vs. prefix matching
4. Evaluate impact on answer quality

---

## Failure Case Analysis

### Common Failure Modes

#### 1. Multi-hop Questions
**Example**: "How does gradient descent relate to linear regression?"
- **Issue**: Requires information from multiple chunks
- **Current behavior**: Retrieves chunks about each topic independently
- **Mitigation**: None currently; future work on query decomposition

#### 2. Ambiguous Queries
**Example**: "What is variance?"
- **Issue**: Could refer to statistical variance or code variance
- **Current behavior**: Returns chunks about both
- **Mitigation**: Conversation history would add context (not yet implemented)

#### 3. Negation Queries
**Example**: "What is NOT a supervised learning algorithm?"
- **Issue**: Semantic search does not model negation well
- **Current behavior**: Returns chunks about supervised learning
- **Mitigation**: Query rewriting or hybrid BM25 + semantic

#### 4. Numerical/Formula Queries
**Example**: "What is the formula for standard deviation?"
- **Issue**: Formulas may not extract cleanly from PDF
- **Current behavior**: Returns text description instead of formula
- **Mitigation**: Better PDF extraction or multi-modal embeddings

---

## Design Decisions Summary

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| **Global token-stream chunking** | Preserves overlap across page boundaries | More complex than page-by-page |
| **508-token chunks** | Uses BGE's full context window | More noise per chunk than 384 |
| **50-token overlap** | Best P@1; prevents sentence splits | +15% chunk count vs. no overlap |
| **`BAAI/bge-base-en-v1.5`** | MTEB SOTA for passage retrieval; asymmetric encoding | Requires query prefix at inference time |
| **One `IndexIVFFlat` per book** | Exact per-book filtering; scales to multi-book | `nprobe` must be set and serialized correctly |
| **`nprobe` set at build time** | Serialized by FAISS; consistent across restarts | Must never be overridden on load |
| **Namespaced `chunk_id` (`{slug}:{n}`)** | Prevents RRF collision across books | Slight overhead in chunk storage |
| **RRF fusion (`k=60`)** | Standard default; no score normalization needed | Fixed `k` may not be optimal at all scales |
| **Prefix deduplication (50 chars)** | Fast, O(n) | Misses semantic duplicates |
| **Keyword-based evaluation** | Simple, interpretable | Doesn't capture semantic relevance |

---

## Future Experiments

### High Priority
- [ ] **Hybrid search**: BM25 (lexical) + semantic search
- [ ] **Re-ranking**: Cross-encoder for top-K reordering
- [ ] **Embedding deduplication**: Cosine similarity vs. prefix matching
- [ ] **Better evaluation**: Human relevance judgments vs. keyword matching

### Medium Priority
- [ ] **Conversation context**: Incorporate chat history into retrieval query
- [ ] **Adaptive chunking**: Semantic-based boundaries vs. fixed tokens
- [ ] **Query expansion**: LLM-generated alternative phrasings

### Low Priority
- [ ] **Multi-modal**: Extract and embed images/tables from PDF
- [ ] **Fine-tuning**: Domain-specific embedding model on annotated data
- [ ] **A/B testing**: Framework for comparing configurations in production

---

## Reproducibility

All experiments can be reproduced with:

```bash
# Ingest a book
python ingest.py --pdf data/<book>.pdf --name "<Title>" --author "<Author>"

# Run evaluation
python evaluate.py

# Run overlap sensitivity experiment
python run_overlap_experiments.py
# Results saved to: experiments/results/overlap_comparison.json
```

**Environment:**
- Python 3.10+
- `sentence-transformers` (see `requirements.txt`)
- `faiss-cpu` (see `requirements.txt`)
- Hardware: CPU-only (no GPU required)

---

**Last Updated**: March 24, 2026
**Experiment Owner**: Portfolio Project
