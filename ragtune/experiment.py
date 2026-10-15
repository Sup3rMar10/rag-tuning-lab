"""Run the three tuning stages on the real corpus. Each stage saves its result, so a long run can be resumed.

    python -m ragtune tune --stage 1     # BM25 settings
    python -m ragtune tune --stage 2     # dense re-ranking settings
    python -m ragtune tune --stage 3     # held-out evaluation, then summary.json
"""

import json
import os
import platform
import random
import time
from dataclasses import replace

from . import data
from .dense import SentenceTransformerEmbedder
from .index import Bm25Index
from .pipeline import Config, Retriever
from .tune import TrialLog, final_eval, tune_bm25, tune_rerank

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
INDEX_PATH = os.path.join(DATA_DIR, "index_porter.db")
CACHE_PATH = os.path.join(DATA_DIR, "embeddings.db")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# What an out-of-the-box FTS5 search does: OR together every word of the question. Tuning is judged against this.
BASELINE = Config(stopwords=False, rerank="none")


def embedder_factory(seq_length):
    return SentenceTransformerEmbedder(MODEL, max_seq_length=seq_length)


def peak_memory_mb():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 1e6)
    except ImportError:
        return None


def _path(name):
    return os.path.join(RESULTS_DIR, name)


def _save(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(_path(name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load(name):
    with open(_path(name), encoding="utf-8") as handle:
        return json.load(handle)


def sample_ids(ids, n, seed):
    ids = sorted(ids)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids)))) if n else ids


def load_questions(tune_n, holdout_n, seed=13):
    root = os.path.join(DATA_DIR, "nq")
    queries = data.load_queries(os.path.join(root, "queries.jsonl"))
    qrels = data.load_qrels(os.path.join(root, "qrels", "test.tsv"))
    tune_half, held_half = data.split_queries(qrels, seed=seed)
    tune_ids = sample_ids(tune_half, tune_n, seed)
    held_ids = sample_ids(held_half, holdout_n, seed + 1)
    pick = lambda ids: ({q: queries[q] for q in ids}, {q: qrels[q] for q in ids})  # noqa: E731
    return pick(tune_ids), pick(held_ids), {"judged_questions": len(qrels), "tune_half": len(tune_half), "held_out_half": len(held_half)}


def _say(record):
    m = record["metrics"]
    ms = record.get("first_stage_ms")
    print(f"  {record['name']:46s} ndcg@10 {m['ndcg@10']:.4f}  recall@100 {m['recall@100']:.4f}" + (f"  bm25 {ms:7.1f} ms" if ms is not None else ""), flush=True)


def run(stage, tune_n=300, holdout_n=500, fresh=False):
    (tune_q, tune_qrels), (held_q, held_qrels), split = load_questions(tune_n, holdout_n)
    index = Bm25Index(INDEX_PATH)
    log_path = _path("trials.jsonl")
    if fresh and stage == 1 and os.path.exists(log_path):
        os.remove(log_path)
    log = TrialLog(log_path)
    started = time.perf_counter()

    if stage == 1:
        print(f"stage 1: BM25 settings on {len(tune_q)} tuning questions", flush=True)
        best, trials = tune_bm25(Retriever(index), tune_q, tune_qrels, log, progress=_say)
        _save("stage1.json", {"best": best.as_dict(), "trials": trials, "n_questions": len(tune_q), "seconds": round(time.perf_counter() - started)})
        print("best BM25 config:", best.as_dict())

    elif stage == 2:
        base = Config(**_load("stage1.json")["best"])
        print(f"stage 2: dense re-ranking on {len(tune_q)} tuning questions, base {base.as_dict()}", flush=True)
        seq_length, cfg, trials = tune_rerank(index, embedder_factory, CACHE_PATH, tune_q, tune_qrels, log, base, progress=_say)
        _save("stage2.json", {"seq_length": seq_length, "best": cfg.as_dict(), "trials": trials, "n_questions": len(tune_q), "seconds": round(time.perf_counter() - started), "model": MODEL})
        print(f"best re-rank config (seq_length={seq_length}):", cfg.as_dict())

    elif stage == 3:
        stage1, stage2 = _load("stage1.json"), _load("stage2.json")
        systems = {
            "baseline (naive BM25)": BASELINE,
            "tuned BM25": Config(**stage1["best"]),
            "tuned BM25 + dense re-rank": Config(**stage2["best"]),
        }
        print(f"stage 3: held-out evaluation on {len(held_q)} questions the tuning never saw", flush=True)
        result = final_eval(index, embedder_factory, CACHE_PATH, held_q, held_qrels, systems, stage2["seq_length"])
        for name, summary in result.items():
            if isinstance(summary, dict) and "ndcg@10" in summary:
                print(f"  {name:30s} ndcg@10 {summary['ndcg@10']:.4f}  p50 {summary['latency']['p50_ms']:.0f} ms", flush=True)
        extras = {}
        for name in ("build_porter_full.json", "bench_build.json"):
            if os.path.exists(_path(name)):
                extras[name.replace(".json", "")] = _load(name)
        _save("summary.json", {
            "environment": {"python": platform.python_version(), "machine": platform.processor(), "os": platform.platform(), "embedding_model": MODEL},
            "corpus": {"passages": index.n_docs, **split, "tuning_questions_used": len(tune_q), "held_out_questions_used": len(held_q)},
            "index_bytes": os.path.getsize(INDEX_PATH), "peak_memory_mb": peak_memory_mb(),
            "stage1": {"best": stage1["best"], "n_questions": stage1["n_questions"]},
            "stage2": {"best": stage2["best"], "seq_length": stage2["seq_length"], "n_questions": stage2["n_questions"]},
            "final": result, **extras,
        })
        print("wrote results/summary.json", flush=True)
    index.close()
    print(f"stage {stage} finished in {time.perf_counter() - started:.0f}s; peak memory {peak_memory_mb()} MB", flush=True)
