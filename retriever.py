import os
import json
from typing import Dict, List, Optional
import faiss
from sentence_transformers import SentenceTransformer


# ----------------------------
# Load all indices
# ----------------------------

def load_all_indices(store_dir: str) -> Dict[str, Dict]:
    """
    Loads all per-book FAISS indices from store_dir.

    nprobe is serialized by faiss.write_index and correctly restored by
    faiss.read_index — do NOT override it here. It was set at build time
    in ingest.py with the formula:
        nlist if nlist <= 20 else max(1, nlist // 4)

    Returns:
    {
      slug: {
        "index": faiss.Index,
        "chunks": List[dict],
        "meta": dict
      }
    }
    """
    indices = {}

    registry_path = os.path.join(store_dir, "registry.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError("registry.json not found")

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    for entry in registry:
        slug = entry["slug"]
        base = os.path.join(store_dir, slug)

        index_path = os.path.join(base, "faiss.index")
        chunks_path = os.path.join(base, "chunks.json")
        meta_path = os.path.join(base, "meta.json")

        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            print(f"  Warning: missing files for slug '{slug}', skipping.")
            continue

        index = faiss.read_index(index_path)
        # nprobe is persisted — no override needed or wanted here.

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        indices[slug] = {
            "index": index,
            "chunks": chunks,
            "meta": meta
        }

    return indices


# ----------------------------
# Retrieve with RRF
# ----------------------------

def retrieve(
    query: str,
    model: SentenceTransformer,
    indices: Dict[str, Dict],
    sources: Optional[List[str]] = None,
    top_k: int = 5,
    k: int = 60
) -> List[dict]:
    """
    Multi-index retrieval with Reciprocal Rank Fusion (RRF).

    If sources is None, queries all loaded indices and merges results with RRF.
    If sources is a list of slugs, queries only those indices.

    RRF formula: score += 1.0 / (k + rank)  [k=60 is standard default]

    Deduplication key is chunk["chunk_id"], which is namespaced as "{slug}:{n}"
    in ingest.py — guaranteed unique across all books. A bare integer counter
    would restart at 0 per book and cause silent chunk collisions here.

    Returns top_k chunk dicts ordered by fused RRF score.
    """
    if not indices:
        return []

    # Validate and select target indices
    if sources:
        selected = {}
        for s in sources:
            if s not in indices:
                raise ValueError(f"Unknown source: '{s}'. Available: {list(indices.keys())}")
            selected[s] = indices[s]
    else:
        selected = indices

    # BGE asymmetric query prefix — passages are indexed WITHOUT this prefix.
    # Omitting it causes silent retrieval degradation (no error, just bad results).
    query_text = f"Represent this sentence for searching relevant passages: {query}"

    query_vec = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    per_index_results = []

    # Per-index FAISS search
    for slug, data in selected.items():
        index = data["index"]
        chunks = data["chunks"]

        # Fetch more than top_k so RRF has meaningful signal to rerank.
        # Capped at ntotal to avoid FAISS errors on small indices.
        fetch_k = min(top_k * 3, index.ntotal)

        faiss_scores, ids = index.search(query_vec, fetch_k)

        results = []
        for rank, idx in enumerate(ids[0], start=1):
            if idx == -1:
                continue
            results.append({
                "chunk": chunks[idx],
                "rank": rank
            })

        per_index_results.append(results)

    # RRF merge across all indices
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, dict] = {}

    for results in per_index_results:
        for item in results:
            chunk = item["chunk"]
            rank = item["rank"]
            cid = chunk["chunk_id"]  # "{slug}:{n}" — unique across all books

            if cid not in scores:
                scores[cid] = 0.0
                chunk_map[cid] = chunk

            scores[cid] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked[:top_k]]
