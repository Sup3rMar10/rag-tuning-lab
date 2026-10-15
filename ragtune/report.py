"""Turn results/*.json into tables and charts. Every number comes from the result files; none is typed by hand."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# Categorical slots 1-3, validated for all-pairs use (light surface). Aqua is under 3:1 contrast on light, so every
# mark is directly labeled and the same numbers are in a table, as the palette's relief rule requires.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as handle:
        return json.load(handle)


def _systems(final):
    return [(name, value) for name, value in final.items() if isinstance(value, dict) and "ndcg@10" in value]


def _fmt(x, digits=3):
    return f"{x:.{digits}f}"


def _ci(pair):
    return f"{pair[0]:.3f} to {pair[1]:.3f}"


def _trial_ms(trial):
    return trial["first_stage_ms"]


def tables(summary, stage1, stage2, bench):
    final = summary["final"]
    corpus = summary["corpus"]
    lines = []
    out = lines.append

    out("### Setup")
    out("")
    out("| | |")
    out("|---|---|")
    out(f"| Corpus | {corpus['passages']:,} Wikipedia passages (BEIR Natural Questions, 1.5 GB of text) |")
    out(f"| Index | {summary['index_bytes'] / 1e9:.2f} GB on disk, built in {summary['build_porter_full']['seconds']:.0f} s ({summary['build_porter_full']['rows_per_second']:,} passages/s) |")
    out(f"| Questions | {corpus['judged_questions']:,} judged. Split once, seeded: {corpus['tune_half']:,} for tuning, {corpus['held_out_half']:,} held out |")
    out(f"| Used here | {corpus['tuning_questions_used']} tuning questions (to choose settings), {corpus['held_out_questions_used']} held-out questions (scored once, at the end) |")
    out(f"| Hardware | {summary['environment']['machine']}, CPU only, no GPU |")
    out(f"| Embedding model | `{summary['environment']['embedding_model']}` |")
    if summary.get("peak_memory_mb"):
        out(f"| Peak memory | {summary['peak_memory_mb']:,} MB (whole process, including the model) |")
    out("")

    out("### Held-out results")
    out("")
    out(f"Scored once on {final['n_questions']} questions that tuning never saw. Latency is cold and per question: BM25 plus embedding, nothing cached, median and 95th percentile over 40 questions per system. All figures here were measured on this machine and are read from `results/summary.json`.")
    out("")
    out("| System | nDCG@10 (95% CI) | Recall@100 | MRR@10 | p50 latency | p95 latency | nDCG@10 vs baseline (95% CI) |")
    out("|---|---|---|---|---|---|---|")
    systems = _systems(final)
    for name, s in systems:
        vs = s.get("vs_baseline_ndcg@10")
        gain = "-" if vs is None else f"{vs['diff']:+.3f} ({vs['ci'][0]:+.3f} to {vs['ci'][1]:+.3f})"
        out(f"| {name} | {_fmt(s['ndcg@10'])} ({_ci(s['ci_ndcg@10'])}) | {_fmt(s['recall@100'])} | {_fmt(s['mrr@10'])} | {s['latency']['p50_ms']:,.0f} ms | {s['latency']['p95_ms']:,.0f} ms | {gain} |")
    out("")

    out("### What each optimization bought")
    out("")
    out("Measured on this machine, from `results/bench_build.json` (build) and `results/trials.jsonl` (stopwords, on the tuning questions).")
    out("")
    out("| Change | Before | After | Effect |")
    out("|---|---|---|---|")
    if bench:
        out(f"| Index build: one transaction and tuned SQLite settings instead of a transaction per row ({bench['slice_passages']:,}-passage slice) | {bench['naive']['seconds']:.1f} s | {bench['optimized']['seconds']:.1f} s | {bench['speedup']}x faster |")
    trials = {t["name"]: t for t in stage1["trials"]}
    base = trials.get("stopwords=False")  # the first setting tried; "baseline" is the starting config, which already removes stopwords
    stop = trials.get("stopwords=True")
    if base and stop:
        out(f"| Remove stopwords from the query | {_trial_ms(base):,.0f} ms/question, nDCG@10 {_fmt(base['metrics']['ndcg@10'])} | {_trial_ms(stop):,.0f} ms/question, nDCG@10 {_fmt(stop['metrics']['ndcg@10'])} | {_trial_ms(base) / _trial_ms(stop):.1f}x faster, {stop['metrics']['ndcg@10'] - base['metrics']['ndcg@10']:+.3f} nDCG@10 |")
    out("")

    out("### Stage 1: BM25 settings (tuning questions)")
    out("")
    out("One setting changes at a time; the winner of each step is kept. Ties within 0.003 nDCG@10 go to the faster setting.")
    out("")
    out("| Setting | nDCG@10 | Recall@100 | BM25 time per question |")
    out("|---|---|---|---|")
    for t in stage1["trials"]:
        out(f"| {t['name']} | {_fmt(t['metrics']['ndcg@10'])} | {_fmt(t['metrics']['recall@100'])} | {_trial_ms(t):,.0f} ms |")
    best = stage1["best"]
    out("")
    out(f"Chosen: stopwords {'removed' if best['stopwords'] else 'kept'}, max query terms {best['max_terms']}, title weight {best['w_title']}.")
    out("")

    out("### Stage 2: dense re-ranking (tuning questions)")
    out("")
    ranked = sorted(stage2["trials"], key=lambda t: -t["metrics"]["ndcg@10"])
    out("The ten best of the settings tried. Re-ranking depth is how many BM25 candidates get embedded, which is where the time goes.")
    out("")
    out("| Embedding input length | Re-rank depth | Method | nDCG@10 | Recall@10 |")
    out("|---|---|---|---|---|")
    for t in ranked[:10]:
        c = t["config"]
        method = {"dense": "dense only", "rrf": f"RRF (k={c['rrf_k']})", "weighted": f"weighted (BM25 weight {c['alpha']})"}[c["rerank"]]
        out(f"| {t['seq_length']} tokens | {c['rerank_depth']} | {method} | {_fmt(t['metrics']['ndcg@10'])} | {_fmt(t['metrics']['recall@10'])} |")
    chosen = stage2["best"]
    out("")
    out(f"Chosen: input length {stage2['seq_length']}, depth {chosen['rerank_depth']}, method {chosen['rerank']}" + (f", BM25 weight {chosen['alpha']}" if chosen["rerank"] == "weighted" else "") + ".")
    return "\n".join(lines)


def charts(summary, stage2, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color="#e7e6e1", linewidth=0.8)
        ax.set_axisbelow(True)

    paths = []

    # 1. Quality against latency for the three systems: the headline trade-off.
    systems = _systems(summary["final"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160, facecolor=SURFACE)
    style(ax)
    for (name, s), color, marker in zip(systems, (GRAY, BLUE, ORANGE), ("s", "o", "D")):
        low, high = s["ci_ndcg@10"]
        x = s["latency"]["p50_ms"]
        ax.errorbar(x, s["ndcg@10"], yerr=[[s["ndcg@10"] - low], [high - s["ndcg@10"]]], fmt=marker, color=color, ecolor=color, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=4, linewidth=1.6)
        ax.annotate(f"{name}\n{s['ndcg@10']:.3f} nDCG@10, {x:,.0f} ms", (x, s["ndcg@10"]), textcoords="offset points", xytext=(12, -4), fontsize=9, color=INK, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per question, cold (ms, log scale)", color=MUTED, fontsize=10)
    ax.set_ylabel("nDCG@10 on held-out questions", color=MUTED, fontsize=10)
    ax.set_title("Better answers cost time. Tuning moves the whole trade-off.", color=INK, fontsize=11, loc="left")
    xs = [s["latency"]["p50_ms"] for _, s in systems]
    ax.set_xlim(min(xs) / 2, max(xs) * 12)
    fig.tight_layout()
    path = os.path.join(out_dir, "quality_vs_latency.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    paths.append(path)

    # 2. Re-rank depth against quality, one line per embedding input length (best fusion method at each point).
    best = {}
    for t in stage2["trials"]:
        key = (t["seq_length"], t["config"]["rerank_depth"])
        best[key] = max(best.get(key, 0.0), t["metrics"]["ndcg@10"])
    seqs = sorted({k[0] for k in best})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160, facecolor=SURFACE)
    style(ax)
    for seq, color, marker in zip(seqs, (BLUE, ORANGE, AQUA), ("o", "D", "s")):
        depths = sorted(d for s, d in best if s == seq)
        values = [best[(seq, d)] for d in depths]
        ax.plot(depths, values, color=color, marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, linewidth=2)
        ax.annotate(f"{seq} tokens", (depths[-1], values[-1]), textcoords="offset points", xytext=(8, 0), fontsize=9, color=INK, va="center")
    ax.set_xticks(sorted({d for _, d in best}))
    ax.set_xlabel("Re-rank depth (BM25 candidates embedded)", color=MUTED, fontsize=10)
    ax.set_ylabel("nDCG@10 on tuning questions", color=MUTED, fontsize=10)
    ax.set_title("Re-rank depth and input length, best fusion method at each point", color=INK, fontsize=11, loc="left")
    ax.set_xlim(min(d for _, d in best) - 5, max(d for _, d in best) + 22)
    fig.tight_layout()
    path = os.path.join(out_dir, "rerank_tuning.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    paths.append(path)
    return paths


def build(write_readme=False):
    summary = _load("summary.json")
    stage1, stage2 = _load("stage1.json"), _load("stage2.json")
    bench = _load("bench_build.json") if os.path.exists(os.path.join(RESULTS, "bench_build.json")) else None
    body = tables(summary, stage1, stage2, bench)
    chart_paths = charts(summary, stage2, RESULTS)
    markdown = (
        "![Quality against latency](results/quality_vs_latency.png)\n\n![Re-rank tuning](results/rerank_tuning.png)\n\n" + body + "\n"
    )
    with open(os.path.join(RESULTS, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)
    if write_readme:
        readme = os.path.join(ROOT, "README.md")
        with open(readme, encoding="utf-8") as handle:
            text = handle.read()
        if START not in text or END not in text:
            raise RuntimeError(f"README.md needs the markers {START} and {END}")
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        with open(readme, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{head}{START}\n\n{markdown}\n{END}{tail}")
    return chart_paths
