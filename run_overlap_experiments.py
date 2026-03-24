"""Run experiments with different overlap values to compare retrieval quality."""

import json
import os
from sentence_transformers import SentenceTransformer
from build_index import extract_text_from_pdf, clean_text, chunk_text, build_and_save_index
from evaluate import load_questions

# Configuration
PDF_PATH = "data/Data Science from Scratch by Joel Grus.pdf"
CHUNK_SIZE = 384  # Use best performing chunk size
OVERLAP_VALUES = [0, 25, 50, 100]
STORE_PATH = "store/"

def run_overlap_experiment(overlap: int, model, questions):
    """Run experiment for a specific overlap value."""
    print(f"\n{'='*60}")
    print(f"Testing overlap={overlap} tokens")
    print(f"{'='*60}")
    
    # Extract and clean text
    text = extract_text_from_pdf(PDF_PATH, 12, 322)
    text = clean_text(text)
    
    # Create chunks with this overlap
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=overlap, tokenizer=model.tokenizer)
    print(f"Created {len(chunks)} chunks")
    
    # Build and save index with overlap suffix
    embeddings = model.encode(chunks)
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    
    import faiss
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save with overlap suffix
    index_path = f"{STORE_PATH}faiss_{CHUNK_SIZE}_overlap{overlap}.index"
    chunks_path = f"{STORE_PATH}chunks_{CHUNK_SIZE}_overlap{overlap}.json"
    
    faiss.write_index(index, index_path)
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)
    
    # Evaluate this configuration
    # Temporarily modify retriever to use our custom index
    from retriever import retrieve
    import faiss as faiss_module
    
    # Monkey patch retrieve function for this experiment
    def retrieve_custom(query: str, chunk_size: int, model, k: int = 5):
        faiss_index = faiss_module.read_index(index_path)
        with open(chunks_path, "r") as f:
            chunk_data = json.load(f)
        embedded_query = model.encode([query])
        import numpy as np
        embedded_query = embedded_query.astype(np.float32)
        distances, indices = faiss_index.search(embedded_query, k)
        return [chunk_data[i] for i in indices[0]]
    
    # Run evaluation
    p1_scores = []
    p3_scores = []
    p5_scores = []
    rr_scores = []
    
    def _matches(text: str, keywords: list) -> bool:
        """Check if all keywords appear in text (old-schema: chunks are plain strings)."""
        t = text.lower()
        return all(kw.lower() in t for kw in keywords)

    def _rr(results: list, keywords: list) -> float:
        """Reciprocal rank over plain-string chunks."""
        for i, text in enumerate(results):
            if _matches(text, keywords):
                return 1.0 / (i + 1)
        return 0.0

    for question in questions:
        query = question["question"]
        keywords = question["keywords"]
        results = retrieve_custom(query, CHUNK_SIZE, model)
        p1_scores.append(1.0 if results and _matches(results[0], keywords) else 0.0)
        p3_scores.append(1.0 if any(_matches(r, keywords) for r in results[:3]) else 0.0)
        p5_scores.append(1.0 if any(_matches(r, keywords) for r in results[:5]) else 0.0)
        rr_scores.append(_rr(results, keywords))
    
    result = {
        'overlap': overlap,
        'chunk_size': CHUNK_SIZE,
        'num_chunks': len(chunks),
        "p1": sum(p1_scores) / len(p1_scores),
        "p3": sum(p3_scores) / len(p3_scores),
        "p5": sum(p5_scores) / len(p5_scores),
        "mrr": sum(rr_scores) / len(rr_scores)
    }
    
    print(f"Results: P@1={result['p1']:.2f}, P@3={result['p3']:.2f}, P@5={result['p5']:.2f}, MRR={result['mrr']:.2f}")
    
    return result

if __name__ == "__main__":
    # Load model and questions
    print("Loading model and evaluation questions...")
    model = SentenceTransformer("all-mpnet-base-v2")
    questions = load_questions("eval/questions.json")
    
    # Run experiments for each overlap value
    all_results = []
    for overlap in OVERLAP_VALUES:
        result = run_overlap_experiment(overlap, model, questions)
        all_results.append(result)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("OVERLAP EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"{'Overlap':<10}{'Chunks':<10}{'P@1':<8}{'P@3':<8}{'P@5':<8}{'MRR':<8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['overlap']:<10}{r['num_chunks']:<10}{r['p1']:<8.2f}{r['p3']:<8.2f}{r['p5']:<8.2f}{r['mrr']:<8.2f}")
    
    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/overlap_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/overlap_comparison.json")
