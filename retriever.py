import faiss
import json
import numpy as np

def retrieve(query : str, chunk_size: int, model, k: int = 5) -> list[str]:
    faiss_index = faiss.read_index(f"store/faiss_{chunk_size}.index")
    with open(f"store/chunks_{chunk_size}.json", "r") as f:
        chunk_data = json.load(f)
    embedded_query = model.encode([query])
    embedded_query = embedded_query.astype(np.float32)
    distances, indices = faiss_index.search(embedded_query, k)
    return [chunk_data[i] for i in indices[0]]

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    results = retrieve("what is gradient descent?", 128, model)
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)