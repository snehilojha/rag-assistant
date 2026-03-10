import json
from sentence_transformers import SentenceTransformer
from retriever import retrieve


def load_questions(path : str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)
    
def chunk_matches(chunk :str, keywords: list[str]) -> bool:
    return all(keyword in chunk for keyword in keywords)

def reciprocal_rank(results: list[str], keywords: list[str]) -> float:
    for i, result in enumerate(results):
        if chunk_matches(result, keywords):
            return 1.0 / (i + 1)
    return 0.0

def evaluate(chunk_size: int, model, questions: list[dict]) -> dict:
    p1_scores = []
    p3_scores = []
    p5_scores = []
    rr_scores = []
    
    for question in questions:
        query = question["question"]
        keywords = question["keywords"]
        results = retrieve(query, chunk_size, model)
        p1_scores.append(1.0 if chunk_matches(results[0], keywords) else 0.0)
        p3_scores.append(1.0 if any(chunk_matches(result, keywords) for result in results[:3]) else 0.0)
        p5_scores.append(1.0 if any(chunk_matches(result, keywords) for result in results[:5]) else 0.0)
        rr_scores.append(reciprocal_rank(results, keywords))
    
    return {
        'chunk_size': chunk_size,
        "p1": sum(p1_scores) / len(p1_scores),
        "p3": sum(p3_scores) / len(p3_scores),
        "p5": sum(p5_scores) / len(p5_scores),
        "rr": sum(rr_scores) / len(rr_scores)
    }

if __name__ == "__main__":
    model = SentenceTransformer("all-mpnet-base-v2")
    questions = load_questions("eval/questions.json")
    all_results = []
    for chunk_size in [128, 256, 384]:
        result = evaluate(chunk_size, model, questions)
        all_results.append(result)

    # print table
    print(f"{'Chunk Size':<12}{'P@1':<8}{'P@3':<8}{'P@5':<8}{'MRR':<8}")
    print("-" * 44)
    for r in all_results:
        print(f"{r['chunk_size']:<12}{r['p1']:<8.2f}{r['p3']:<8.2f}{r['p5']:<8.2f}{r['rr']:<8.2f}")

    # save metrics.json
    with open("eval/results/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    