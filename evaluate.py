"""Evaluation framework for RAG retrieval quality assessment.

Metrics:
- Precision@K (P@1, P@3, P@5): proportion of queries with a relevant chunk in top-K
- Mean Reciprocal Rank (MRR): mean of 1/rank of first relevant chunk

Relevance is determined by keyword presence in chunk text.

Matching strategy:
- Case-insensitive substring match on the keyword root (first 5+ chars),
  which handles common morphological variants:
    "generalizes" → root "gener" matches "generalize", "generalization", "general"
    "branching"   → root "branch" matches "branch", "branches", "branched"
- A chunk is considered relevant if it matches >= MATCH_THRESHOLD fraction
  of the keywords (default: 0.67, i.e. 2 out of 3).
  ALL-keyword AND logic is too brittle — one rare or inflected keyword
  causes false negatives on otherwise correct retrievals.

Usage:
    python evaluate.py --questions eval/questions.json
    python evaluate.py --questions eval/questions.json --source data-science-from-scratch
    python evaluate.py --questions eval/questions.json --baseline eval/results/metrics.json
    python evaluate.py --questions eval/questions.json --output eval/results/new_metrics.json
    python evaluate.py --questions eval/questions.json --debug   # show per-question results
"""

import argparse
import json
import os
from sentence_transformers import SentenceTransformer
from retriever import retrieve, load_all_indices

# Fraction of keywords that must match for a chunk to be considered relevant.
# 0.67 = at least 2 out of 3 keywords (rounds up via ceil in chunk_matches).
MATCH_THRESHOLD = 0.67

# Minimum keyword root length used for stem-aware matching.
# 5 chars catches most useful roots without over-truncating short words.
MIN_ROOT_LEN = 5


def load_questions(path: str) -> list[dict]:
    """Load evaluation questions from JSON file.

    Expected format:
        [{"question": "What is X?", "keywords": ["term1", "term2"]}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_root(kw: str) -> str:
    """
    Return a truncated root for stem-aware matching.

    Takes the first MIN_ROOT_LEN characters of the keyword (lowercased).
    This is a deliberately simple approach — no NLTK dependency, no stemmer
    install — that handles the most common failure mode: morphological variants
    like "generalizes" vs "generalization", "branching" vs "branches".

    Examples:
        "generalizes" → "gener"
        "branching"   → "branc"  (matches "branch", "branches", "branched")
        "shortest"    → "short"  (matches "short", "shorter", "shortest")
        "critical"    → "criti"  (matches "critical", "criticism")
        "bayes"       → "bayes"  (short word — taken in full if < MIN_ROOT_LEN)
    """
    return kw.lower()[:min(MIN_ROOT_LEN, len(kw))]


def chunk_matches(chunk: dict, keywords: list[str], threshold: float = MATCH_THRESHOLD) -> bool:
    """
    Check if a chunk is relevant based on keyword root presence.

    A chunk is relevant if at least `threshold` fraction of keyword roots
    appear in the chunk text (case-insensitive).

    Args:
        chunk:     Chunk dict with at minimum a 'text' field
        keywords:  Keywords defining relevance for this question
        threshold: Minimum fraction of keywords that must match (default 0.67)

    Returns:
        True if enough keyword roots match, False otherwise
    """
    text = chunk["text"].lower()
    required = max(1, round(len(keywords) * threshold))
    matched = sum(1 for kw in keywords if keyword_root(kw) in text)
    return matched >= required


def reciprocal_rank(chunks: list[dict], keywords: list[str]) -> float:
    """
    Calculate reciprocal rank of the first relevant chunk.

    Returns:
        1 / (1-based position of first match), or 0.0 if no match found
    """
    for i, chunk in enumerate(chunks):
        if chunk_matches(chunk, keywords):
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    model,
    indices: dict,
    questions: list[dict],
    sources: list[str] | None = None,
    debug: bool = False,
) -> dict:
    """
    Evaluate retrieval quality across all questions.

    Args:
        model:     SentenceTransformer model for query embedding
        indices:   Loaded index dict from load_all_indices()
        questions: List of question dicts with 'question' and 'keywords' fields
        sources:   Optional list of book slugs to restrict retrieval to
        debug:     If True, print per-question hit/miss details

    Returns:
        Dict with p1, p3, p5, rr keys (floats between 0 and 1)
    """
    p1_scores = []
    p3_scores = []
    p5_scores = []
    rr_scores = []

    if debug:
        print(f"\n{'#':<4} {'Hit@1':<7} {'Hit@5':<7} {'RR':<6} Question")
        print("-" * 70)

    for i, question in enumerate(questions):
        query = question["question"]
        keywords = question["keywords"]

        chunks = retrieve(query, model, indices=indices, sources=sources, top_k=5)

        hit1 = chunks and chunk_matches(chunks[0], keywords)
        hit3 = any(chunk_matches(c, keywords) for c in chunks[:3])
        hit5 = any(chunk_matches(c, keywords) for c in chunks[:5])
        rr   = reciprocal_rank(chunks, keywords)

        p1_scores.append(1.0 if hit1 else 0.0)
        p3_scores.append(1.0 if hit3 else 0.0)
        p5_scores.append(1.0 if hit5 else 0.0)
        rr_scores.append(rr)

        if debug:
            mark1 = "✓" if hit1 else "✗"
            mark5 = "✓" if hit5 else "✗"
            print(f"{i+1:<4} {mark1:<7} {mark5:<7} {rr:<6.3f} {query[:55]}")

            # On a miss, show what keywords failed so you can diagnose
            if not hit5:
                text_sample = chunks[0]["text"].lower()[:200] if chunks else "(no chunks)"
                roots = [keyword_root(kw) for kw in keywords]
                matched = [r for r in roots if r in text_sample]
                missed  = [r for r in roots if r not in text_sample]
                print(f"     keywords: {keywords}")
                print(f"     roots:    matched={matched}  missed={missed}")
                print(f"     top chunk preview: ...{text_sample[:120]}...")

    n = len(p1_scores)
    return {
        "p1": sum(p1_scores) / n,
        "p3": sum(p3_scores) / n,
        "p5": sum(p5_scores) / n,
        "rr": sum(rr_scores) / n,
    }


def print_comparison(new: dict, baseline_path: str | None) -> None:
    """Print results and optional side-by-side comparison with old baseline."""
    print("\n── New Architecture ──────────────────────")
    print(f"{'Metric':<8}{'Score':<10}")
    print("-" * 20)
    for metric in ("p1", "p3", "p5", "rr"):
        label = "MRR" if metric == "rr" else f"P@{metric[1]}"
        print(f"{label:<8}{new[metric]:<10.3f}")

    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

        # Old architecture stored one entry per chunk_size; best was chunk_size=384
        old = next((r for r in baseline_results if r.get("chunk_size") == 384), None)
        if old:
            print("\n── vs Old Baseline (chunk_size=384, all-mpnet-base-v2) ────")
            print(f"{'Metric':<8}{'Old':<10}{'New':<10}{'Delta':<10}")
            print("-" * 38)
            for metric in ("p1", "p3", "p5", "rr"):
                label = "MRR" if metric == "rr" else f"P@{metric[1]}"
                delta = new[metric] - old[metric]
                sign = "+" if delta >= 0 else ""
                print(f"{label:<8}{old[metric]:<10.3f}{new[metric]:<10.3f}{sign}{delta:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("--questions", required=True,                          help="Path to questions JSON file")
    parser.add_argument("--source",    default=None,                           help="Book slug to restrict retrieval to (default: all books)")
    parser.add_argument("--baseline",  default="eval/results/metrics.json",   help="Path to old metrics.json for comparison")
    parser.add_argument("--store",     default="store",                        help="Path to index store directory")
    parser.add_argument("--output",    default=None,                           help="Optional path to save new metrics as JSON")
    parser.add_argument("--debug",     action="store_true",                    help="Print per-question hit/miss breakdown")
    args = parser.parse_args()

    print("Loading model and indices...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    indices = load_all_indices(args.store)
    print(f"Loaded {len(indices)} book(s): {list(indices.keys())}")

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    sources = [args.source] if args.source else None
    if sources:
        print(f"Restricting retrieval to: {sources}")

    result = evaluate(model, indices, questions, sources=sources, debug=args.debug)

    print_comparison(result, args.baseline)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
