"""Evaluation framework for RAG retrieval quality assessment.

This module implements standard Information Retrieval metrics to evaluate
the quality of chunk retrieval across different configurations:
- Precision@K: Proportion of queries with relevant chunk in top-K
- Mean Reciprocal Rank (MRR): Average of reciprocal ranks of first relevant chunk

Evaluation uses keyword-based relevance matching as ground truth.
"""

import json
from sentence_transformers import SentenceTransformer
from retriever import retrieve

def load_questions(path: str) -> list[dict]:
    """Load evaluation questions from JSON file.
    
    Args:
        path: Path to questions JSON file
    
    Returns:
        List of question dictionaries with 'question' and 'keywords' fields
    
    Expected format:
        [{"question": "What is X?", "keywords": ["term1", "term2"]}, ...]
    """
    with open(path, "r") as f:
        return json.load(f)
    
def chunk_matches(chunk: str, keywords: list[str]) -> bool:
    """Check if a chunk is relevant based on keyword presence.
    
    A chunk is considered relevant if it contains ALL specified keywords.
    This is a simple but effective relevance heuristic for evaluation.
    
    Args:
        chunk: Text chunk to check
        keywords: List of keywords that must all be present
    
    Returns:
        True if chunk contains all keywords, False otherwise
    
    Note:
        Case-sensitive matching. Keywords should be chosen carefully to
        represent the core concepts needed to answer the question.
    """
    return all(keyword in chunk for keyword in keywords)

def reciprocal_rank(results: list[str], keywords: list[str]) -> float:
    """Calculate reciprocal rank of first relevant chunk.
    
    Reciprocal rank is 1/rank where rank is the position (1-indexed) of the
    first relevant result. If no relevant result is found, returns 0.
    
    Args:
        results: List of retrieved chunks (ordered by relevance)
        keywords: Keywords defining relevance
    
    Returns:
        1/(position of first match) or 0 if no match
    
    Example:
        - First result matches: RR = 1/1 = 1.0
        - Third result matches: RR = 1/3 = 0.33
        - No match: RR = 0.0
    """
    for i, result in enumerate(results):
        if chunk_matches(result, keywords):
            return 1.0 / (i + 1)  # i+1 because rank is 1-indexed
    return 0.0

def evaluate(chunk_size: int, model, questions: list[dict]) -> dict:
    """Evaluate retrieval quality for a specific chunk size configuration.
    
    Computes standard IR metrics across all evaluation questions:
    - P@1: Precision at rank 1 (top result is relevant)
    - P@3: Precision at rank 3 (relevant result in top 3)
    - P@5: Precision at rank 5 (relevant result in top 5)
    - MRR: Mean Reciprocal Rank (average position of first relevant result)
    
    Args:
        chunk_size: Chunk size configuration to evaluate (128, 256, or 384)
        model: SentenceTransformer model for embedding queries
        questions: List of evaluation questions with ground truth keywords
    
    Returns:
        Dictionary with chunk_size and metric scores (all between 0 and 1)
    
    Note:
        Higher scores are better. P@K measures recall (did we find it?),
        while MRR measures ranking quality (how high did we rank it?).
    """
    p1_scores = []
    p3_scores = []
    p5_scores = []
    rr_scores = []
    
    for question in questions:
        query = question["question"]
        keywords = question["keywords"]
        # Retrieve top-5 chunks for this query
        results = retrieve(query, chunk_size, model)
        # P@1: Is the top result relevant?
        p1_scores.append(1.0 if chunk_matches(results[0], keywords) else 0.0)
        # P@3: Is any of the top-3 results relevant?
        p3_scores.append(1.0 if any(chunk_matches(result, keywords) for result in results[:3]) else 0.0)
        # P@5: Is any of the top-5 results relevant?
        p5_scores.append(1.0 if any(chunk_matches(result, keywords) for result in results[:5]) else 0.0)
        # MRR: What's the reciprocal rank of the first relevant result?
        rr_scores.append(reciprocal_rank(results, keywords))
    
    # Return average scores across all questions
    return {
        'chunk_size': chunk_size,
        "p1": sum(p1_scores) / len(p1_scores),
        "p3": sum(p3_scores) / len(p3_scores),
        "p5": sum(p5_scores) / len(p5_scores),
        "rr": sum(rr_scores) / len(rr_scores)
    }

if __name__ == "__main__":
    # Run comparative evaluation across all chunk sizes
    model = SentenceTransformer("all-mpnet-base-v2")
    questions = load_questions("eval/questions.json")
    all_results = []
    
    # Evaluate each chunk size configuration
    for chunk_size in [128, 256, 384]:
        result = evaluate(chunk_size, model, questions)
        all_results.append(result)

    # Print results table to console
    print(f"{'Chunk Size':<12}{'P@1':<8}{'P@3':<8}{'P@5':<8}{'MRR':<8}")
    print("-" * 44)
    for r in all_results:
        print(f"{r['chunk_size']:<12}{r['p1']:<8.2f}{r['p3']:<8.2f}{r['p5']:<8.2f}{r['rr']:<8.2f}")

    # Save metrics to JSON for later analysis
    with open("eval/results/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)