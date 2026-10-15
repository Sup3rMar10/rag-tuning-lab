"""Command line:  python -m ragtune <command>

  download      fetch the 1.5 GB corpus (BEIR Natural Questions) into data/
  build-index   build the full BM25 index
  bench-build   time the naive and optimized index builds on the same slice
  tune          run a tuning stage on the real corpus (--stage 1, 2 or 3)
  report        write the results tables and charts from results/summary.json
  ask           answer a question from the real index
  demo          run the whole pipeline on a tiny built-in corpus (no download)
"""

import argparse
import json
import os
import sys
import time

from . import data
from .index import build_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
CORPUS = os.path.join(DATA_DIR, "nq", "corpus.jsonl")


def _write_result(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def cmd_download(args):
    zip_path = os.path.join(DATA_DIR, "nq.zip")
    started = time.perf_counter()
    data.download(zip_path, progress=lambda n: print(f"\r  {n / 1e6:7.0f} MB", end="", flush=True) if n % (1 << 25) < (1 << 20) else None)
    print(f"\ndownloaded in {time.perf_counter() - started:.0f}s")
    print("extracted to", data.extract(zip_path, DATA_DIR))
    return 0


def cmd_build_index(args):
    out = args.out or os.path.join(DATA_DIR, f"index_{args.tokenizer}.db")
    print(f"building {out} (tokenizer={args.tokenizer}, detail={args.detail}, limit={args.limit})", flush=True)
    stats = build_index(
        out, data.iter_corpus(CORPUS), tokenizer=args.tokenizer, detail=args.detail, optimized=True, limit=args.limit,
        progress=lambda n: print(f"  {n:>9,} passages", flush=True),
    )
    name = f"build_{args.tokenizer}_{args.detail}{'' if args.limit is None else '_slice'}.json"
    path = _write_result(name, {
        "rows": stats.rows, "seconds": round(stats.seconds, 1), "rows_per_second": round(stats.rows_per_second),
        "db_bytes": stats.db_bytes, "tokenizer": args.tokenizer, "detail": args.detail, "mode": stats.mode,
    })
    print(f"done: {stats.rows:,} passages in {stats.seconds:.0f}s, {stats.db_bytes / 1e9:.2f} GB -> {path}")
    return 0


def cmd_bench_build(args):
    """Same slice, naive vs optimized, so the speedup is measured on identical work."""
    result = {"slice_passages": args.rows}
    for mode in ("naive", "optimized"):
        path = os.path.join(DATA_DIR, f"bench_{mode}.db")
        stats = build_index(path, data.iter_corpus(CORPUS), tokenizer="porter", optimized=(mode == "optimized"), limit=args.rows)
        result[mode] = {"seconds": round(stats.seconds, 2), "rows_per_second": round(stats.rows_per_second), "db_bytes": stats.db_bytes}
        print(f"{mode:>9}: {stats.seconds:8.1f}s  {stats.rows_per_second:9,.0f} rows/s", flush=True)
        os.remove(path)
    result["speedup"] = round(result["naive"]["seconds"] / result["optimized"]["seconds"], 1)
    print(f"speedup: {result['speedup']}x ->", _write_result("bench_build.json", result))
    return 0


def cmd_tune(args):
    from . import experiment

    experiment.run(args.stage, tune_n=args.tune_questions, holdout_n=args.holdout_questions, fresh=args.fresh)
    return 0


def cmd_report(args):
    from . import report

    for path in report.build(write_readme=args.write_readme):
        print("wrote", path)
    print("wrote", os.path.join(RESULTS_DIR, "REPORT.md") + (" and updated README.md" if args.write_readme else ""))
    return 0


def cmd_ask(args):
    """Answer from the real index with BM25 only (the cheap, fast path). Add a model with the Python API, see the README."""
    from . import experiment, rag
    from .index import Bm25Index
    from .pipeline import Config, Retriever

    index = Bm25Index(experiment.INDEX_PATH)
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    cfg = Config(rerank="none")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            tuned = json.load(handle)["stage1"]["best"]
        cfg = Config(**{**tuned, "rerank": "none"})
    hits, timings = Retriever(index).retrieve(args.question, cfg)
    result = rag.answer(args.question, hits, k=args.k)
    print(result["answer"])
    for n, hit in zip(result["citations"], result["sources"]):
        print(f"  [{n}] {hit['title']} (passage {hit['doc_id']})")
    print(f"  retrieved from {index.n_docs:,} passages in {timings['bm25'] * 1000:.0f} ms (BM25, {'tuned' if os.path.exists(summary_path) else 'default'} settings)")
    return 0


def cmd_demo(args):
    """The whole pipeline on a tiny corpus, offline. A fake embedder stands in for the real model."""
    import tempfile

    from . import metrics
    from .dense import HashEmbedder
    from .index import Bm25Index
    from .pipeline import Config, Retriever, run_queries
    from .synthetic import make_synthetic

    passages, queries, qrels = make_synthetic()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        path = os.path.join(work, "demo.db")
        stats = build_index(path, passages)
        print(f"indexed {stats.rows} passages ({stats.db_bytes / 1e3:.0f} KB) in {stats.seconds * 1000:.0f} ms")
        index = Bm25Index(path)
        retriever = Retriever(index, HashEmbedder(128))
        for name, cfg in [
            ("BM25, every word", Config(stopwords=False)),
            ("BM25, stopwords removed", Config()),
            ("BM25 + dense rerank (fake embedder)", Config(rerank="weighted", alpha=0.5)),
        ]:
            run = run_queries(retriever, queries, cfg)
            scores = metrics.evaluate(run, qrels)
            print(f"  {name:38s} ndcg@10 {scores['ndcg@10']:.3f}  recall@100 {scores['recall@100']:.3f}")
        index.close()
    print("This is a toy corpus. Run `python -m ragtune download` for the real 1.5 GB benchmark.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ragtune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download").set_defaults(func=cmd_download)
    build = sub.add_parser("build-index")
    build.add_argument("--tokenizer", choices=["porter", "plain"], default="porter")
    build.add_argument("--detail", choices=["full", "column", "none"], default="full")
    build.add_argument("--limit", type=int, default=None, help="index only the first N passages (for experiments)")
    build.add_argument("--out", default=None)
    build.set_defaults(func=cmd_build_index)
    bench = sub.add_parser("bench-build")
    bench.add_argument("--rows", type=int, default=20000)
    bench.set_defaults(func=cmd_bench_build)
    tune = sub.add_parser("tune")
    tune.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    tune.add_argument("--tune-questions", type=int, default=300, help="tuning questions to use (from the tuning half)")
    tune.add_argument("--holdout-questions", type=int, default=500, help="held-out questions to use (from the held-out half)")
    tune.add_argument("--fresh", action="store_true", help="stage 1 only: start trials.jsonl over")
    tune.set_defaults(func=cmd_tune)
    rep_cmd = sub.add_parser("report")
    rep_cmd.add_argument("--write-readme", action="store_true", help="also fill the results section of README.md")
    rep_cmd.set_defaults(func=cmd_report)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=5)
    ask.set_defaults(func=cmd_ask)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
