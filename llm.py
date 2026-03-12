"""LLM integration for answer generation using Google Gemini.

This module handles the generation phase of the RAG pipeline:
1. Deduplicate retrieved chunks to avoid redundancy
2. Build a grounded prompt with retrieved context
3. Generate answer using Gemini LLM
4. Return answer text to user

The prompt is designed to ensure answers are grounded in retrieved passages only,
reducing hallucination and improving factual accuracy.
"""

from google import genai
import os
from dotenv import load_dotenv

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# Initialize Gemini client with API key from environment
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# Use Flash model for fast, cost-effective generation
MODEL = "gemini-2.5-flash"

def build_prompt(query: str, chunks: list[str]) -> str:
    """Build a grounded prompt for the LLM with retrieved context.
    
    Creates a prompt that instructs the LLM to answer based ONLY on the
    provided passages, reducing hallucination and ensuring factual accuracy.
    
    Args:
        query: User's question
        chunks: Retrieved text passages (already deduplicated)
    
    Returns:
        Formatted prompt string for LLM generation
    
    Note:
        The prompt explicitly instructs the model to refuse answering if
        the context is insufficient, preventing fabricated answers.
    """
    # Concatenate all chunks into a single context block
    context = "\n".join(chunks)
    return f"""You are a helpful tutor for the book "Data Science from Scratch" by Joel Grus.
Use ONLY the following retrieved passages to answer the question.
If the answer cannot be found in the passages, say "I don't have enough context to answer this."
    
Passages: {context}
Question: {query}

Answer:"""

def deduplicate(chunks: list[str], threshold: int = 50) -> list[str]:
    """Remove duplicate or highly similar chunks from retrieval results.
    
    Uses a simple prefix-matching heuristic to detect duplicates. This is necessary
    because overlapping chunks can result in near-identical passages being retrieved.
    
    Args:
        chunks: List of retrieved text chunks
        threshold: Number of characters to compare for similarity (default: 50)
    
    Returns:
        Deduplicated list of chunks, preserving order
    
    Note:
        This is a naive approach using first N characters. A better approach would
        use semantic similarity (cosine distance between embeddings), but this is
        faster and works reasonably well for the current use case.
    """
    seen = []
    for chunk in chunks:
        # Check if first 'threshold' chars of this chunk appear in any seen chunk
        if not any(chunk[:threshold] in seen_chunk for seen_chunk in seen):
            seen.append(chunk)
    return seen

def ask_gemini(query: str, chunks: list[str]) -> str:
    """Generate an answer to the query using Gemini LLM with retrieved context.
    
    Main entry point for answer generation. Combines deduplication, prompt building,
    and LLM generation into a single function.
    
    Args:
        query: User's question
        chunks: Retrieved text chunks from FAISS search
    
    Returns:
        Generated answer text from Gemini
    
    Raises:
        May raise API errors if Gemini service is unavailable or API key is invalid
    """
    # Remove duplicate chunks to reduce prompt length and redundancy
    chunks = deduplicate(chunks)
    # Build grounded prompt with context
    prompt = build_prompt(query, chunks)
    # Generate answer using Gemini
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    # Test end-to-end RAG pipeline
    from sentence_transformers import SentenceTransformer
    from retriever import retrieve
    
    model = SentenceTransformer("all-mpnet-base-v2")
    query = "How does gradient descent minimize a function?"
    # Retrieve relevant chunks
    chunks = retrieve(query, chunk_size=384, model=model)
    # Generate grounded answer
    answer = ask_gemini(query, chunks)
    print(answer)
