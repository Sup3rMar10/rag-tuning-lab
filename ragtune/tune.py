"""Stage-wise tuning. Winners are chosen on the tuning questions only; the held-out questions are scored once, at the end.

Stage 1 tunes the BM25 first stage (stopwords, query-term pruning, title weight).
Stage 2 tunes the dense re-ranking on top of it (embedding input length, re-rank depth, fusion method and weight).
Stage 3 scores the baseline and both tuned systems on the held-out questions, with confidence intervals.

Rule for ties: among settings within `EPSILON` nDCG@10 of the best, the cheapest wins. Quality is not worth
paying for when it is indistinguishable.
"""

import json
import os
import time
from dataclasses import replace

from . import metrics
from .dense import EmbeddingCache
from .pipeline import Config, Retriever, measure_latency, run_queries

EPSILON = 0.003  # 0.3 points of nDCG@10

DEFAULT_GRIDS = {
    "bm25": [("stopwords", [False, True]), ("max_terms", [None, 8, 5, 3]), ("w_title", [0.0, 0.5, 1.0, 2.0, 4.0])],
    "seq_lengths": [64, 128],  # 256 tokens runs at about 26 passages/s on a laptop CPU, too slow to be worth sweeping
    "depths": [20, 50, 100],
    "alphas": [0.3, 0.5, 0.7, 0.9],
    "rrf_ks": [10, 60],
}


class TrialLog:
    """Append-only JSONL of every configuration tried, so the search is reproducible and auditable."""

    def __init__(self, path):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, record):
        if self.path:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def choose(scored, epsilon=EPSILON):
    """scored: list of (ndcg, cost, payload). Best quality wins; within epsilon of it, the lowest cost wins; ties go to the earlier entry."""
    top = max(s[0] for s in scored)
    near = [(i, s) for i, s in enumerate(scored) if s[0] >= top - epsilon]
    return min(near, key=lambda item: (item[1][1], item[0]))[1][2]


def score(retriever, queries, qrels, cfg):
    """Run and evaluate one configuration over the given questions only."""
    started = time.perf_counter()
    run = run_queries(retriever, queries, cfg)
    elapsed = time.perf_counter() - started
    judged = {q: qrels[q] for q in queries if q in qrels}
    return metrics.evaluate(run, judged), elapsed


def mean_first_stage_ms(retriever, queries, cfg):
    times = [retriever.first_stage(text, cfg)[1] for text in queries.values()]
    return 1000 * sum(times) / len(times) if times else 0.0


def tune_bm25(retriever, queries, qrels, log, grids=None, progress=None):
    grids = grids or DEFAULT_GRIDS
    best = Config(rerank="none")
    trials = []
    base_metrics, _ = score(retriever, queries, qrels, best)
    trials.append(_record("bm25", "baseline", best, base_metrics, mean_first_stage_ms(retriever, queries, best), len(queries)))
    log.add(trials[-1])
    for field, values in grids["bm25"]:
        scored = []
        for value in values:
            cfg = replace(best, **{field: value})
            result, _ = score(retriever, queries, qrels, cfg)
            ms = mean_first_stage_ms(retriever, queries, cfg)
            record = _record("bm25", f"{field}={value}", cfg, result, ms, len(queries))
            trials.append(record)
            log.add(record)
            scored.append((result["ndcg@10"], ms, cfg))
            if progress:
                progress(record)
        best = choose(scored)
    return best, trials


def tune_rerank(index, embedder_factory, cache_path, queries, qrels, log, base_cfg, grids=None, progress=None):
    """embedder_factory(seq_length) -> embedder. One embedder per input length; the cache is shared and keyed by model name."""
    grids = grids or DEFAULT_GRIDS
    deepest = max(grids["depths"])
    trials, scored = [], []
    retriever = Retriever(index)  # shared across input lengths, so the slow BM25 pass over the questions happens once
    for seq_length in grids["seq_lengths"]:
        embedder = embedder_factory(seq_length)
        cache = EmbeddingCache(cache_path)
        retriever.embedder, retriever.cache = embedder, cache
        pool = replace(base_cfg, rerank="dense", candidates=deepest, rerank_depth=deepest)
        started = time.perf_counter()
        retriever.prefetch(list(queries.values()), pool, progress=None)
        embed_seconds = time.perf_counter() - started
        log.add({"stage": "rerank", "name": f"prefetch seq={seq_length}", "seconds": round(embed_seconds, 1), "n_queries": len(queries), "embedder": embedder.name})
        variants = []
        for depth in grids["depths"]:
            variants.append(("dense", {"rerank": "dense", "rerank_depth": depth}))
            for k in grids["rrf_ks"]:
                variants.append((f"rrf k={k}", {"rerank": "rrf", "rerank_depth": depth, "rrf_k": k}))
            for alpha in grids["alphas"]:
                variants.append((f"weighted a={alpha}", {"rerank": "weighted", "rerank_depth": depth, "alpha": alpha}))
        for label, changes in variants:
            cfg = replace(pool, **changes)
            result, _ = score(retriever, queries, qrels, cfg)
            record = _record("rerank", f"seq={seq_length} depth={cfg.rerank_depth} {label}", cfg, result, None, len(queries))
            record["seq_length"] = seq_length
            trials.append(record)
            log.add(record)
            scored.append((result["ndcg@10"], cfg.rerank_depth * seq_length, (seq_length, cfg)))
            if progress:
                progress(record)
        cache.close()
    seq_length, cfg = choose(scored)
    return seq_length, cfg, trials


def final_eval(index, embedder_factory, cache_path, queries, qrels, systems, seq_length, latency_queries=40, seed=0):
    """Score every named system on the held-out questions: metrics, confidence intervals and cold latency.

    systems: {"name": Config}. The first one is the baseline that the others are compared with.
    """
    out, per_query_scores = {}, {}
    embedder = embedder_factory(seq_length)
    cache = EmbeddingCache(cache_path)
    retriever = Retriever(index, embedder, cache)
    for name, cfg in systems.items():
        retriever.prefetch(list(queries.values()), cfg)
        run = run_queries(retriever, queries, cfg)
        judged = {q: qrels[q] for q in queries if q in qrels}
        per_query = metrics.per_query(run, judged)
        per_query_scores[name] = per_query
        summary = {m: sum(s[m] for s in per_query.values()) / len(per_query) for m in next(iter(per_query.values()))}
        summary["ci_ndcg@10"] = metrics.bootstrap_ci([s["ndcg@10"] for s in per_query.values()], seed=seed)
        summary["latency"] = measure_latency(index, embedder, cfg, queries, n=latency_queries)
        summary["config"] = cfg.as_dict()
        out[name] = summary
    cache.close()
    names = list(systems)
    for name in names[1:]:
        base = {q: s["ndcg@10"] for q, s in per_query_scores[names[0]].items()}
        other = {q: s["ndcg@10"] for q, s in per_query_scores[name].items()}
        out[name]["vs_baseline_ndcg@10"] = metrics.paired_bootstrap(base, other, seed=seed)
    out["n_questions"] = len(queries)
    return out


def _record(stage, name, cfg, result, first_stage_ms, n_queries):
    return {
        "stage": stage, "name": name, "config": cfg.as_dict(), "n_queries": n_queries,
        "metrics": {k: round(v, 5) for k, v in result.items()},
        "first_stage_ms": None if first_stage_ms is None else round(first_stage_ms, 2),
    }
