"""Retrieve relevant text chunks using FAISS vector similarity search.

This module handles the online retrieval phase of the RAG pipeline:
1. Load pre-built FAISS index and chunk data from disk
2. Embed user query using the same model as indexing
3. Perform similarity search to find top-K most relevant chunks
4. Return original text chunks for LLM context
"""

import faiss
import json
import numpy as np

def retrieve(query: str, chunk_size: int, model, k: int = 5) -> list[str]:
    """Retrieve top-K most relevant text chunks for a given query.
    
    Uses FAISS inner product search to find chunks with highest semantic similarity
    to the query. The same embedding model must be used for both indexing and retrieval
    to ensure embedding space consistency.
    
    Args:
        query: User's natural language question
        chunk_size: Which chunk size index to use (128, 256, or 384)
        model: SentenceTransformer model (must match the one used in indexing)
        k: Number of top chunks to retrieve (default: 5)
    
    Returns:
        List of k text chunks, ordered by relevance (highest similarity first)
    
    Note:
        FAISS returns indices into the original chunk array. We use these indices
        to retrieve the actual text chunks from the JSON file.
    """
    # Load pre-built FAISS index from disk
    faiss_index = faiss.read_index(f"store/faiss_{chunk_size}.index")
    # Load corresponding text chunks
    with open(f"store/chunks_{chunk_size}.json", "r") as f:
        chunk_data = json.load(f)
    # Embed the query using the same model as indexing
    embedded_query = model.encode([query])
    # Ensure float32 for FAISS compatibility
    embedded_query = embedded_query.astype(np.float32)
    # Search for k nearest neighbors (highest inner product = most similar)
    distances, indices = faiss_index.search(embedded_query, k)
    # Return the actual text chunks (indices[0] because we only have 1 query)
    return [chunk_data[i] for i in indices[0]]

if __name__ == "__main__":
    # Test retrieval with a sample query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    # Retrieve top-5 chunks for a test question
    results = retrieve("what is gradient descent?", 128, model)
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)