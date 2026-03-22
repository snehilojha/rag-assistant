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
    page indices are 0-based (fitz convention).
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
        # rejoin hyphenated line-broken words: "compu-\nter" → "computer"
        text = re.sub(r'www\.it-ebooks\.info', '', text)
        # Fix hyphenated line breaks that split words
        text = re.sub(r'-\n', '', text)
        # Remove form feed characters (page breaks)
        text = re.sub(r'\x0c', '', text)
        # Keep only printable ASCII + newlines and tabs
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        # Normalize excessive newlines to double newline (paragraph breaks)
        text = re.sub(r'\n{3,}','\n\n', text)
        # Collapse multiple spaces/tabs to single space
        text = re.sub(r'[ \t]{2,}',' ', text)
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
    """
    stride = chunk_size - overlap
    chunks = []
    chunk_id = 0

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
            "chunk_id": chunk_id
        })

        chunk_id += 1

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
    - Uses IndexIVFFlat for large datasets (>= 100 chunks)
    - Falls back to IndexFlatIP for small datasets
    - nprobe set to max(1, nlist // 10) for reasonable recall
    - Training uses a random sample encoded in one pass (not all vectors)
    - Adding vectors is streamed batch-by-batch to avoid OOM at scale
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
    nlist = max(1, min(int(np.sqrt(total)), total // 39))
    print(f"  nlist: {nlist}")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Pass 1 — encode a random sample for training only
    # Must happen before index.add(). FAISS needs ~39x nlist vectors minimum.
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
    del train_embeddings  # free memory before streaming pass

    # Pass 2 — stream all chunks in batches, add to trained index
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

    # nprobe: how many clusters searched at query time
    # Default of 1 gives terrible recall. nlist // 10 is a good tradeoff.
    index.nprobe = max(1, nlist // 10)

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

    # FAISS index
    faiss.write_index(index, os.path.join(base_path, "faiss.index"))

    # Chunks
    with open(os.path.join(base_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Meta
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

    # Remove existing entry for this slug (deduplication)
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

    # Load model once — tokenizer is attached as model.tokenizer
    print("Loading model...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tokenizer = model.tokenizer  # no separate AutoTokenizer load needed

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
