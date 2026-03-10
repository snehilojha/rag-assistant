from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

def build_prompt(query: str, chunks: list[str]) -> str:
    context = "\n".join(chunks)
    return f"""You are a helpful tutor for the book "Data Science from Scratch" by Joel Grus.
Use ONLY the following retrieved passages to answer the question.
If the answer cannot be found in the passages, say "I don't have enough context to answer this."
    
Passages: {context}
Question: {query}

Answer:"""

def deduplicate(chunks: list[str], threshold: int = 50) -> list[str]:
    seen = []
    for chunk in chunks:
        if not any(chunk[:threshold] in seen_chunk for seen_chunk in seen):
            seen.append(chunk)
    return seen

def ask_gemini(query: str, chunks: list[str]) -> str:
    chunks = deduplicate(chunks)
    prompt = build_prompt(query, chunks)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from retriever import retrieve
    
    model = SentenceTransformer("all-mpnet-base-v2")
    query = "How does gradient descent minimize a function?"
    chunks = retrieve(query, chunk_size=384, model=model)
    answer = ask_gemini(query, chunks)
    print(answer)
