"""BM25 first stage, optional dense re-ranking, and fusion, driven by one Config so a sweep is just a list of Configs."""

import time
from dataclasses import asdict, dataclass

from . import fusion
from .dense import embed_items, query_key

RERANK_MODES = ("none", "dense", "rrf", "weighted")


@dataclass(frozen=True)
class Config:
    # first stage (BM25)
    w_title: float = 1.0
    w_text: float = 1.0
    stopwords: bool = True
    max_terms: int | None = None
    candidates: int = 100
    # second stage
    rerank: str = "none"        # none | dense (cosine only) | rrf | weighted
    rerank_depth: int = 100     # only the top N BM25 candidates are re-ranked
    alpha: float = 0.5          # weight on BM25 in "weighted"
    rrf_k: int = 60
    final_k: int = 100

    def __post_init__(self):
        if self.rerank not in RERANK_MODES:
            raise ValueError(f"rerank must be one of {RERANK_MODES}")
        if self.rerank_depth > self.candidates:
            raise ValueError("rerank_depth cannot exceed candidates")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

    def first_stage_key(self):
        return (self.w_title, self.w_text, self.stopwords, self.max_terms, self.candidates)

    def as_dict(self):
        return asdict(self)


def passage_text(hit):
    return f"{hit['title']}. {hit['text']}" if hit["title"] else hit["text"]


def apply_rerank(cfg, hits, dense_scores):
    """Return hits re-ordered. Candidates beyond rerank_depth keep their BM25 order, after the re-ranked ones."""
    depth = min(cfg.rerank_depth, len(hits))
    head, tail = hits[:depth], hits[depth:]
    if cfg.rerank == "none" or depth == 0:
        return hits[:cfg.final_k]
    by_id = {h["doc_id"]: h for h in head}
    bm25 = {h["doc_id"]: h["score"] for h in head}
    dense = {h["doc_id"]: float(s) for h, s in zip(head, dense_scores)}
    if cfg.rerank == "dense":
        order = [d for d, _ in sorted(dense.items(), key=lambda item: (-item[1], item[0]))]
    elif cfg.rerank == "rrf":
        bm25_order = [h["doc_id"] for h in head]
        dense_order = [d for d, _ in sorted(dense.items(), key=lambda item: (-item[1], item[0]))]
        order = [d for d, _ in fusion.rrf([bm25_order, dense_order], k=cfg.rrf_k)]
    else:
        order = [d for d, _ in fusion.weighted(bm25, dense, alpha=cfg.alpha)]
    return ([by_id[d] for d in order] + tail)[:cfg.final_k]


class Retriever:
    def __init__(self, index, embedder=None, cache=None, batch_size=64):
        self.index, self.embedder, self.cache, self.batch_size = index, embedder, cache, batch_size
        self.embed_stats = {}
        self._first_stage = {}

    def first_stage(self, query, cfg):
        """BM25 candidates, memoised so a sweep over second-stage settings does not repeat the slow part."""
        key = (cfg.first_stage_key(), query)
        if key not in self._first_stage:
            started = time.perf_counter()
            hits = self.index.search(query, k=cfg.candidates, w_title=cfg.w_title, w_text=cfg.w_text, stopwords=cfg.stopwords, max_terms=cfg.max_terms)
            self._first_stage[key] = (hits, time.perf_counter() - started)
        return self._first_stage[key]

    def prefetch(self, queries, cfg, progress=None):
        """Embed every candidate for every question in large batches (much faster than question by question)."""
        if cfg.rerank == "none":
            return
        items = []
        for query in queries:
            hits, _ = self.first_stage(query, cfg)
            items.append((query_key(query), query))
            items.extend((h["doc_id"], passage_text(h)) for h in hits[:cfg.rerank_depth])
        embed_items(self.embedder, self.cache, items, self.batch_size, self.embed_stats, progress)

    def dense_scores(self, query, hits):
        items = [(query_key(query), query)] + [(h["doc_id"], passage_text(h)) for h in hits]
        vectors = embed_items(self.embedder, self.cache, items, self.batch_size, self.embed_stats)
        return vectors[1:] @ vectors[0]

    def retrieve(self, query, cfg):
        """Return (hits, timings). Timings are seconds: bm25 (first call only), dense, fuse."""
        hits, bm25_seconds = self.first_stage(query, cfg)
        timings = {"bm25": bm25_seconds, "dense": 0.0, "fuse": 0.0}
        if cfg.rerank == "none" or not hits:
            return hits[:cfg.final_k], timings
        started = time.perf_counter()
        head = hits[:cfg.rerank_depth]
        scores = self.dense_scores(query, head)
        timings["dense"] = time.perf_counter() - started
        started = time.perf_counter()
        ranked = apply_rerank(cfg, hits, scores)
        timings["fuse"] = time.perf_counter() - started
        return ranked, timings


def run_queries(retriever, queries, cfg, progress=None):
    """Run every question. Returns {query_id: [doc_ids]}."""
    run = {}
    for n, (query_id, text) in enumerate(queries.items(), start=1):
        hits, _ = retriever.retrieve(text, cfg)
        run[query_id] = [h["doc_id"] for h in hits]
        if progress:
            progress(n, len(queries))
    return run


def measure_latency(index, embedder, cfg, queries, n=40):
    """Online, cold-cache per-question latency in milliseconds: BM25 plus embedding, nothing memoised or cached."""
    retriever = Retriever(index, embedder, cache=None)
    totals = []
    for text in list(queries.values())[:n]:  # queries is {question_id: text}
        started = time.perf_counter()
        hits = index.search(text, k=cfg.candidates, w_title=cfg.w_title, w_text=cfg.w_text, stopwords=cfg.stopwords, max_terms=cfg.max_terms)
        if cfg.rerank != "none" and hits:
            scores = retriever.dense_scores(text, hits[:cfg.rerank_depth])
            apply_rerank(cfg, hits, scores)
        totals.append((time.perf_counter() - started) * 1000)
    totals.sort()
    return {"n": len(totals), "p50_ms": totals[len(totals) // 2], "p95_ms": totals[min(len(totals) - 1, int(len(totals) * 0.95))], "mean_ms": sum(totals) / len(totals)}
