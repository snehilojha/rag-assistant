"""Build FAISS indices from PDF document with multiple chunk sizes.

This module handles the offline indexing phase of the RAG pipeline:
1. Extract text from PDF using PyMuPDF
2. Clean and normalize text (remove artifacts, fix spacing)
3. Chunk text using token-based splitting with overlap
4. Generate embeddings using SentenceTransformers
5. Build FAISS indices for fast similarity search
6. Save indices and chunks to disk for retrieval phase

The module creates multiple indices with different chunk sizes to enable
comparative evaluation of retrieval quality.
"""

import json
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import re

# Configuration: Multiple chunk sizes for comparative evaluation
CHUNK_SIZES = [128, 256, 384]  # Token counts for different granularities
OVERLAP = 50  # Token overlap to preserve context across chunk boundaries
PDF_PATH = "data/Data Science from Scratch by Joel Grus.pdf"
STORE_PATH = "store/"

def extract_text_from_pdf(pdf_path: str, start_page: int, end_page: int) -> str:
    """Extract raw text from PDF pages.
    
    Uses PyMuPDF (fitz) to extract text content from specified page range.
    Pages are 0-indexed in PyMuPDF but we use human-readable numbering.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: First page to extract (inclusive, 0-indexed)
        end_page: Last page to extract (exclusive, 0-indexed)
    
    Returns:
        Single concatenated string of all extracted text, pages separated by newlines
    
    Note:
        Pages 12-322 are used to skip front matter and index sections
    """
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc.pages(start_page, end_page))

def clean_text(text: str) -> str:
    """Clean and normalize extracted PDF text.
    
    Removes common PDF artifacts and normalizes whitespace to improve
    chunking quality and embedding generation.
    
    Args:
        text: Raw text extracted from PDF
    
    Returns:
        Cleaned text with normalized whitespace
    
    Cleaning steps:
        1. Remove 'www.it-ebooks.info' watermarks
        2. Fix hyphenated line breaks (e.g., 'exam-\nple' -> 'example')
        3. Remove form feed characters (\x0c)
        4. Remove non-printable characters (keep only ASCII 32-126, \n, \t)
        5. Collapse excessive newlines (3+ -> 2)
        6. Collapse multiple spaces/tabs to single space
    """
    # Remove common PDF watermarks
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
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, tokenizer) -> list[str]:
    """Split text into overlapping chunks using token-based splitting.
    
    Uses the embedding model's tokenizer to ensure chunks align with how
    the model will process them. Token-based splitting is superior to
    character or word-based splitting because:
    - Ensures chunks don't exceed model's max sequence length
    - Aligns with how embeddings are actually generated
    - Handles subword tokenization correctly
    
    Args:
        text: Cleaned text to chunk
        chunk_size: Target number of tokens per chunk
        overlap: Number of tokens to overlap between consecutive chunks
        tokenizer: Tokenizer from the embedding model (ensures consistency)
    
    Returns:
        List of text chunks, each approximately chunk_size tokens
    
    Note:
        Overlap preserves context across boundaries, preventing information
        loss when a relevant passage spans chunk boundaries.
    """
    # Tokenize entire text without special tokens or truncation
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, max_length=None)
    
    start = 0
    chunks = []
    while start < len(token_ids):
        # Extract chunk_size tokens starting from current position
        end = start + chunk_size
        chunk = token_ids[start:end]
        # Decode back to text for storage and embedding
        decoded_chunk = tokenizer.decode(chunk)
        chunks.append(decoded_chunk)
        # Move forward by (chunk_size - overlap) to create overlap
        start = end - overlap
    return chunks

def build_and_save_index(chunks: list[str], chunk_size: int, model):
    """Generate embeddings and build FAISS index for fast similarity search.
    
    Creates a FAISS IndexFlatIP (inner product) index for exact nearest neighbor
    search. Inner product is used because SentenceTransformers embeddings are
    normalized, making inner product equivalent to cosine similarity but faster.
    
    Args:
        chunks: List of text chunks to index
        chunk_size: Chunk size identifier for file naming
        model: SentenceTransformer model for embedding generation
    
    Saves:
        - store/faiss_{chunk_size}.index: FAISS index file
        - store/chunks_{chunk_size}.json: Original text chunks for retrieval
    
    Note:
        IndexFlatIP provides exact search (no approximation) which is acceptable
        for datasets of this size (~1000-3000 chunks). For larger datasets,
        consider IndexIVFFlat or IndexHNSW for approximate search.
    """
    # Generate embeddings for all chunks (batch processing for efficiency)
    embeddings = model.encode(chunks)
    # FAISS requires float32 (SentenceTransformers returns float32 by default)
    embeddings = embeddings.astype('float32')
    # Get embedding dimension (768 for all-mpnet-base-v2)
    dimension = embeddings.shape[1]
    # Create flat index with inner product similarity
    index = faiss.IndexFlatIP(dimension)
    # Add all embeddings to the index
    index.add(embeddings)
    # Persist index to disk
    faiss.write_index(index, f"{STORE_PATH}faiss_{chunk_size}.index")
    # Save original chunks for retrieval (indices map back to these)
    with open(f"{STORE_PATH}chunks_{chunk_size}.json", "w") as f:
        json.dump(chunks, f)

if __name__ == "__main__":
    # Load embedding model (downloads on first run, ~420MB)
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Extract text from main content pages (skip front matter and index)
    text = extract_text_from_pdf(PDF_PATH, 12, 322)
    text = clean_text(text)
    
    # Build indices for each chunk size to enable comparative evaluation
    for size in CHUNK_SIZES:
        chunks = chunk_text(text, chunk_size=size, overlap=OVERLAP, tokenizer=model.tokenizer)
        build_and_save_index(chunks, chunk_size=size, model=model)
        print(f"Built index for chunk size {size}: {len(chunks)} chunks")