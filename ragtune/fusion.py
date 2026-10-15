"""Combine two rankings of the same candidates. Pure functions, easy to test."""


def rrf(rank_lists, k=60):
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank). Returns [(doc_id, score)] best first."""
    scores = {}
    for ranking in rank_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def minmax(scores):
    """Scale to 0..1. If every score is equal the list carries no signal, so every doc gets 0."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {doc: 0.0 for doc in scores}
    return {doc: (value - low) / (high - low) for doc, value in scores.items()}


def weighted(a_scores, b_scores, alpha=0.5):
    """alpha * norm(a) + (1 - alpha) * norm(b). A doc missing from one side counts as that side's minimum (0)."""
    a, b = minmax(a_scores), minmax(b_scores)
    docs = set(a) | set(b)
    fused = {doc: alpha * a.get(doc, 0.0) + (1 - alpha) * b.get(doc, 0.0) for doc in docs}
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
