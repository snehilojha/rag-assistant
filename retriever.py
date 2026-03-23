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
    Loads all per-book indices and sets nprobe properly.

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
            continue

        index = faiss.read_index(index_path)

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
    Multi-index retrieval + RRF fusion.
    Returns full chunk dicts.
    """

    if not indices:
        return []

    # validate sources
    if sources:
        selected = {}
        for s in sources:
            if s not in indices:
                raise ValueError(f"Unknown source: {s}")
            selected[s] = indices[s]
    else:
        selected = indices

    # BGE requires query prefix
    query_text = f"Represent this sentence for searching relevant passages: {query}"

    query_vec = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    per_index_results = []

    # ----------------------------
    # Per-index retrieval
    # ----------------------------
    for slug, data in selected.items():
        index = data["index"]
        chunks = data["chunks"]
        fetch_k = min(top_k * 3, index.ntotal)

        faiss_scores, ids = index.search(query_vec, fetch_k)

        results = []
        for rank, idx in enumerate(ids[0], start=1):
            if idx == -1:
                continue

            chunk = chunks[idx]

            # attach rank for RRF (optional but explicit)
            results.append({
                "chunk": chunk,
                "rank": rank
            })

        per_index_results.append(results)

    # ----------------------------
    # RRF merge
    # ----------------------------
    scores = {}
    meta = {}

    for results in per_index_results:
        for item in results:
            chunk = item["chunk"]
            rank = item["rank"]

            cid = chunk["chunk_id"]

            if cid not in scores:
                scores[cid] = 0.0
                meta[cid] = chunk

            scores[cid] += 1.0 / (k + rank)

    # sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [meta[cid] for cid, _ in ranked[:top_k]]