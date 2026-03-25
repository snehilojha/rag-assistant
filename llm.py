"""LLM integration for answer generation using OpenAI GPT-4o mini.

RAG pipeline:
1. Deduplicate retrieved chunks
2. Build grounded prompt with retrieved context
3. Generate answer via OpenAI
4. Return answer text
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
MODEL = "gpt-4o-mini"


def deduplicate(chunks: list[dict], threshold: int = 50) -> list[dict]:
    """
    Remove near-duplicate chunks using prefix matching.
    Naive but fast — good enough until embedding-based dedup is implemented.
    """
    seen = []
    unique = []
    for chunk in chunks:
        text = chunk["text"]
        if not any(text[:threshold] in s for s in seen):
            seen.append(text)
            unique.append(chunk)
    return unique


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a grounded prompt from retrieved chunk dicts.
    Each passage is labelled with its source book and page.
    """
    passages = []
    for c in chunks:
        label = f"[{c['book']} — Page {c['page']}]"
        passages.append(f"{label}\n{c['text']}")

    context = "\n\n".join(passages)

    return f"""You are a helpful study assistant. You have access to passages from the user's book library.
Answer the question using ONLY the passages below.
If the answer cannot be found in the passages, say "I don't have enough context to answer this."
Always cite which book and page your answer comes from.

Passages:
{context}

Question: {query}

Answer:"""


def ask_llm(query: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer using GPT-4o mini.

    Args:
        query: User's question
        chunks: Retrieved chunk dicts from retriever

    Returns:
        Generated answer string

    Raises:
        ValueError: if chunks or query are empty
        openai.APIConnectionError: network failure
        openai.APIStatusError: upstream API error
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not chunks:
        raise ValueError("No chunks retrieved — cannot generate answer")

    chunks = deduplicate(chunks)
    prompt = build_prompt(query, chunks)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    # Build the client at call time so a fresh server restart picks up the
    # current .env value instead of a stale import-time value.
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # low temp — factual grounded answers, not creative
    )

    return response.choices[0].message.content
