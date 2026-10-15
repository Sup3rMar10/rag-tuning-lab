"""Retrieval metrics, written out in plain Python so every number can be checked by hand.

ndcg@k, recall@k and mrr@k follow the BEIR definitions: recall@k is the share of a question's relevant
passages found in the top k, and ranks are 1-based.
"""

import math
import random


def ndcg_at_k(ranked, relevant, k=10):
    if not relevant:
        return 0.0
    dcg = sum(relevant.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, relevant, k=100):
    if not relevant:
        return 0.0
    return len({d for d in ranked[:k] if d in relevant}) / len(relevant)


def mrr_at_k(ranked, relevant, k=10):
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def per_query(run, qrels, ks=(10, 100)):
    """Per-question scores for every question that has relevance judgments. Questions missing from run score 0."""
    out = {}
    for query_id, relevant in qrels.items():
        ranked = run.get(query_id, [])
        scores = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks}
        scores["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
        scores["mrr@10"] = mrr_at_k(ranked, relevant, 10)
        out[query_id] = scores
    return out


def evaluate(run, qrels, ks=(10, 100)):
    """Mean of each metric over the judged questions."""
    scores = per_query(run, qrels, ks)
    if not scores:
        return {}
    names = next(iter(scores.values())).keys()
    return {name: sum(s[name] for s in scores.values()) / len(scores) for name in names}


def bootstrap_ci(values, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap confidence interval for the mean."""
    values = list(values)
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(n))
    return means[int(n * alpha / 2)], means[min(n - 1, int(n * (1 - alpha / 2)))]


def paired_bootstrap(a, b, n=2000, seed=0, alpha=0.05):
    """Confidence interval for mean(b) - mean(a) over the same questions, plus the share of resamples where b > a."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"diff": 0.0, "ci": (0.0, 0.0), "p_better": 0.0}
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(seed)
    size = len(diffs)
    means = sorted(sum(diffs[rng.randrange(size)] for _ in range(size)) / size for _ in range(n))
    return {
        "diff": sum(diffs) / size,
        "ci": (means[int(n * alpha / 2)], means[min(n - 1, int(n * (1 - alpha / 2)))]),
        "p_better": sum(1 for m in means if m > 0) / n,
    }
