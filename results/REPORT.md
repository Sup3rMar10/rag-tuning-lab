![Quality against latency](results/quality_vs_latency.png)

![Re-rank tuning](results/rerank_tuning.png)

### Setup

| | |
|---|---|
| Corpus | 2,681,468 Wikipedia passages (BEIR Natural Questions, 1.5 GB of text) |
| Index | 2.48 GB on disk, built in 128 s (20,977 passages/s) |
| Questions | 3,452 judged. Split once, seeded: 1,726 for tuning, 1,726 held out |
| Used here | 300 tuning questions (to choose settings), 500 held-out questions (scored once, at the end) |
| Hardware | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, CPU only, no GPU |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Peak memory | 858 MB (whole process, including the model) |

### Held-out results

Scored once on 500 questions that tuning never saw. Latency is cold and per question: BM25 plus embedding, nothing cached, median and 95th percentile over 40 questions per system. All figures here were measured on this machine and are read from `results/summary.json`.

| System | nDCG@10 (95% CI) | Recall@100 | MRR@10 | p50 latency | p95 latency | nDCG@10 vs baseline (95% CI) |
|---|---|---|---|---|---|---|
| baseline (naive BM25) | 0.269 (0.239 to 0.300) | 0.743 | 0.233 | 4,213 ms | 7,986 ms | - |
| tuned BM25 | 0.270 (0.239 to 0.299) | 0.766 | 0.227 | 2,561 ms | 3,561 ms | +0.000 (-0.012 to +0.011) |
| tuned BM25 + dense re-rank | 0.384 (0.351 to 0.416) | 0.766 | 0.330 | 3,671 ms | 4,789 ms | +0.114 (+0.091 to +0.137) |

### What each optimization bought

Measured on this machine, from `results/bench_build.json` (build) and `results/trials.jsonl` (stopwords, on the tuning questions).

| Change | Before | After | Effect |
|---|---|---|---|
| Index build: one transaction and tuned SQLite settings instead of a transaction per row (20,000-passage slice) | 195.0 s | 0.7 s | 300.1x faster |
| Remove stopwords from the query | 3,387 ms/question, nDCG@10 0.286 | 402 ms/question, nDCG@10 0.273 | 8.4x faster, -0.012 nDCG@10 |

### Stage 1: BM25 settings (tuning questions)

One setting changes at a time; the winner of each step is kept. Ties within 0.003 nDCG@10 go to the faster setting.

| Setting | nDCG@10 | Recall@100 | BM25 time per question |
|---|---|---|---|
| baseline | 0.273 | 0.754 | 402 ms |
| stopwords=False | 0.286 | 0.764 | 3,387 ms |
| stopwords=True | 0.273 | 0.754 | 402 ms |
| max_terms=None | 0.286 | 0.764 | 3,387 ms |
| max_terms=8 | 0.283 | 0.767 | 2,198 ms |
| max_terms=5 | 0.260 | 0.744 | 416 ms |
| max_terms=3 | 0.207 | 0.654 | 119 ms |
| w_title=0.0 | 0.221 | 0.632 | 2,012 ms |
| w_title=0.5 | 0.258 | 0.703 | 1,993 ms |
| w_title=1.0 | 0.283 | 0.767 | 2,198 ms |
| w_title=2.0 | 0.299 | 0.777 | 2,080 ms |
| w_title=4.0 | 0.295 | 0.783 | 2,085 ms |

Chosen: stopwords kept, max query terms 8, title weight 2.0.

### Stage 2: dense re-ranking (tuning questions)

The ten best of the settings tried. Re-ranking depth is how many BM25 candidates get embedded, which is where the time goes.

| Embedding input length | Re-rank depth | Method | nDCG@10 | Recall@10 |
|---|---|---|---|---|
| 64 tokens | 100 | weighted (BM25 weight 0.3) | 0.419 | 0.634 |
| 128 tokens | 100 | weighted (BM25 weight 0.3) | 0.413 | 0.638 |
| 128 tokens | 100 | dense only | 0.413 | 0.605 |
| 64 tokens | 50 | weighted (BM25 weight 0.3) | 0.412 | 0.622 |
| 128 tokens | 50 | dense only | 0.412 | 0.608 |
| 64 tokens | 100 | dense only | 0.407 | 0.627 |
| 64 tokens | 50 | dense only | 0.405 | 0.623 |
| 128 tokens | 50 | weighted (BM25 weight 0.3) | 0.404 | 0.619 |
| 64 tokens | 100 | RRF (k=10) | 0.402 | 0.629 |
| 128 tokens | 100 | RRF (k=10) | 0.401 | 0.636 |

Chosen: input length 64, depth 100, method weighted, BM25 weight 0.3.
