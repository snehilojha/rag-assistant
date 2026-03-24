import os
import re
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer


# ----------------------------
# Types
# ----------------------------

Page = Tuple[int, str]
Chunk = Dict[str, object]


# ----------------------------
# Utils
# ----------------------------

def generate_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


# ----------------------------
# Step 1 — Extract
# ----------------------------

def load_and_extract(pdf_path: str, start: int = None, end: int = None) -> List[Page]:
    """
    Extract raw text per page from PDF.
    Page indices are 0-based (fitz convention).
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        if start is not None and i < start:
            continue
        if end is not None and i > end:
            break
        text = page.get_text()
        pages.append((i, text))

    print(f"  Extracted {len(pages)} pages.")
    return pages


# ----------------------------
# Step 2 — Clean
# ----------------------------

def clean_text(pages: List[Page]) -> List[Page]:
    """
    Clean raw page text:
    - collapse line breaks
    - fix soft hyphens / hyphenation artifacts
    - collapse whitespace
    """
    cleaned = []

    for page_num, text in pages:
        text = re.sub(r'www\.it-ebooks\.info', '', text)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\x0c', '', text)
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        cleaned.append((page_num, text))

    return cleaned


# ----------------------------
# Step 3 — Chunk (global token stream)
# ----------------------------

def chunk_text(
    pages: List[Page],
    book: str,
    slug: str,
    tokenizer,
    chunk_size: int = 508,
    overlap: int = 50,
    min_chunk_tokens: int = 20
) -> List[Chunk]:
    """
    Tokenize all pages into a single global token stream,
    then slide a window across it. This preserves overlap
    across page boundaries. Each chunk records the page
    it started on.

    chunk_id is namespaced as "{slug}:{n}" to guarantee
    global uniqueness across all books. This is critical
    for correct RRF deduplication in multi-book retrieval —
    a bare integer counter restarts at 0 for every book,
    causing silent chunk collisions in the RRF merge step.
    """
    stride = chunk_size - overlap
    chunks = []
    local_id = 0

    # Build global token stream + page map
    all_tokens = []
    token_page_map = []

    for page_num, text in pages:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        token_page_map.extend([page_num] * len(tokens))

    for i in range(0, len(all_tokens), stride):
        chunk_tokens = all_tokens[i:i + chunk_size]

        # drop near-empty tail chunks
        if len(chunk_tokens) < min_chunk_tokens:
            continue

        chunk_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        start_page = token_page_map[i]

        chunks.append({
            "text": chunk_str,
            "book": book,
            "slug": slug,
            "page": start_page,
            "chunk_id": f"{slug}:{local_id}"  # globally unique across all books
        })

        local_id += 1

    print(f"  Created {len(chunks)} chunks.")
    return chunks


# ----------------------------
# Step 4 — Embed + Build Index
# ----------------------------

def build_index(
    chunks: List[Chunk],
    model: SentenceTransformer,
    batch_size: int = 32
) -> faiss.Index:
    """
    Embed all chunks and build a FAISS index.

    Index type selection:
    - IndexFlatIP for small datasets (< 100 chunks): exact search, no training needed
    - IndexIVFFlat for larger datasets: Voronoi clustering, 10-100x faster at scale,
      <5% accuracy loss when nprobe is tuned correctly

    nprobe controls how many Voronoi clusters are searched at query time.
    Formula: search all clusters if nlist <= 20 (small index, no speed penalty),
    otherwise nlist // 4 (good recall / speed tradeoff for larger indices).

    nprobe IS serialized by faiss.write_index / faiss.read_index, so this
    value is set once here at build time and correctly restored on load.
    Do NOT override it in retriever.py.

    Two-pass build (memory safety):
    - Pass 1: encode a random sample for training only, then free those embeddings
    - Pass 2: stream all chunks in batches of batch_size, add to trained index
    Random sampling for training prevents biased centroids from early-chapter
    topic clustering (first N chunks are all from the same section of the book).
    """
    np.random.seed(42)
    total = len(chunks)
    all_texts = [c["text"] for c in chunks]

    # Get actual embedding dim from model — never hardcode
    dim = model.get_sentence_embedding_dimension()

    print(f"  Embedding dimension: {dim}")
    print(f"  Total chunks: {total}")

    # --- Small dataset fallback ---
    if total < 100:
        print("  Small dataset — using IndexFlatIP.")
        index = faiss.IndexFlatIP(dim)

        embeddings = model.encode(
            all_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        index.add(embeddings)
        return index

    # --- IVFFlat for larger datasets ---
    # nlist formula:
    # - sqrt(total): standard heuristic for cluster count
    # - total // 39: FAISS hard requirement — training needs >= 39 * nlist vectors
    #   (39 from FAISS source code, not the commonly cited 30)
    nlist = max(1, min(int(np.sqrt(total)), total // 39))
    print(f"  nlist: {nlist}")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Pass 1 — random sample for training
    sample_size = min(total, nlist * 40)
    sample_indices = np.random.choice(total, size=sample_size, replace=False)
    sample_texts = [all_texts[i] for i in sample_indices]

    print(f"  Encoding training sample ({sample_size} chunks)...")
    train_embeddings = model.encode(
        sample_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("  Training index...")
    index.train(train_embeddings)
    del train_embeddings  # free before streaming pass

    # Pass 2 — stream all chunks into trained index
    print("  Adding vectors (streaming)...")
    for i in range(0, total, batch_size):
        batch_texts = all_texts[i:i + batch_size]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        index.add(batch_embeddings)

        if i % (batch_size * 50) == 0:
            print(f"    {min(i + batch_size, total)}/{total} chunks added...")

    # nprobe: search all clusters for small indices (no speed cost),
    # nlist // 4 for larger ones (good recall / speed tradeoff).
    # This value IS persisted by faiss.write_index — set once here, never override on load.
    index.nprobe = nlist if nlist <= 20 else max(1, nlist // 4)

    print(f"  nprobe set to: {index.nprobe}")
    print("  Index built.")

    return index


# ----------------------------
# Step 5 — Save
# ----------------------------

def save_book(
    chunks: List[Chunk],
    index: faiss.Index,
    slug: str,
    book: str,
    author: str
) -> Dict:
    """
    Write to store/{slug}/:
    - faiss.index
    - chunks.json
    - meta.json
    Returns meta dict (passed to update_registry).
    """
    base_path = os.path.join("store", slug)
    os.makedirs(base_path, exist_ok=True)

    faiss.write_index(index, os.path.join(base_path, "faiss.index"))

    with open(os.path.join(base_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    meta = {
        "title": book,
        "author": author,
        "slug": slug,
        "total_chunks": len(chunks),
        "added_date": datetime.utcnow().isoformat()
    }

    with open(os.path.join(base_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to store/{slug}/")
    return meta


# ----------------------------
# Step 6 — Registry
# ----------------------------

def update_registry(meta: Dict) -> None:
    """
    Upsert entry in store/registry.json.
    If slug already exists, replaces it (re-ingestion safe).
    """
    path = os.path.join("store", "registry.json")
    os.makedirs("store", exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = []

    registry = [r for r in registry if r["slug"] != meta["slug"]]

    registry.append({
        "slug": meta["slug"],
        "title": meta["title"],
        "author": meta["author"],
        "added": meta["added_date"]
    })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"  Registry updated. Total books: {len(registry)}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF book into the RAG store.")
    parser.add_argument("--pdf",        required=True,          help="Path to PDF file")
    parser.add_argument("--name",       required=True,          help="Book title")
    parser.add_argument("--author",     required=True,          help="Author name")
    parser.add_argument("--start-page", type=int, default=None, help="First page to index (0-based, optional)")
    parser.add_argument("--end-page",   type=int, default=None, help="Last page to index (0-based, inclusive, optional)")

    args = parser.parse_args()

    slug = generate_slug(args.name)
    print(f"\nIngesting: '{args.name}' by {args.author}")
    print(f"Slug: {slug}\n")

    print("Loading model...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tokenizer = model.tokenizer

    print("Extracting PDF...")
    raw_pages = load_and_extract(args.pdf, args.start_page, args.end_page)

    print("Cleaning text...")
    cleaned_pages = clean_text(raw_pages)

    print("Chunking...")
    chunks = chunk_text(cleaned_pages, args.name, slug, tokenizer)

    print("Building index...")
    index = build_index(chunks, model)

    print("Saving...")
    meta = save_book(chunks, index, slug, args.name, args.author)

    print("Updating registry...")
    update_registry(meta)

    print(f"\nDone. '{args.name}' is ready to query.\n")


if __name__ == "__main__":
    main()
