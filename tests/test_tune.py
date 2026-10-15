"""Tests for the tuning harness, using the synthetic corpus and a fake embedder so they run in seconds."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtune import __main__ as cli  # noqa: E402
from ragtune import experiment, tune  # noqa: E402
from ragtune.dense import HashEmbedder  # noqa: E402
from ragtune.index import Bm25Index, build_index  # noqa: E402
from ragtune.pipeline import Config, Retriever  # noqa: E402
from ragtune.synthetic import make_synthetic  # noqa: E402

SMALL_GRIDS = {
    "bm25": [("stopwords", [False, True]), ("max_terms", [None, 4]), ("w_title", [0.5, 2.0])],
    "seq_lengths": [32, 64],
    "depths": [10, 30],
    "alphas": [0.3, 0.7],
    "rrf_ks": [60],
}


class ChooseTests(unittest.TestCase):
    def test_clear_winner(self):
        self.assertEqual(tune.choose([(0.30, 5, "a"), (0.40, 9, "b"), (0.35, 1, "c")]), "b")

    def test_within_epsilon_the_cheapest_wins(self):
        scored = [(0.400, 100, "slow"), (0.399, 10, "fast"), (0.380, 1, "too-weak")]
        self.assertEqual(tune.choose(scored, epsilon=0.003), "fast")

    def test_exact_ties_go_to_the_earlier_entry(self):
        self.assertEqual(tune.choose([(0.4, 5, "first"), (0.4, 5, "second")]), "first")

    def test_a_single_candidate(self):
        self.assertEqual(tune.choose([(0.1, 1, "only")]), "only")


class TrialLogTests(unittest.TestCase):
    def test_appends_one_json_line_per_trial_and_creates_folders(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = os.path.join(d, "deep", "trials.jsonl")
            log = tune.TrialLog(path)
            log.add({"name": "a", "metrics": {"ndcg@10": 0.5}})
            log.add({"name": "b"})
            rows = [json.loads(line) for line in open(path, encoding="utf-8")]
            self.assertEqual([r["name"] for r in rows], ["a", "b"])
            tune.TrialLog(None).add({"ignored": True})  # a log with no path is a no-op, not an error


class Harness(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        passages, queries, qrels = make_synthetic(n_docs=240, n_queries=48)
        path = os.path.join(self.dir.name, "i.db")
        build_index(path, passages)
        self.index = Bm25Index(path)
        self.addCleanup(self.index.close)
        ids = sorted(queries)
        self.tune_q = {q: queries[q] for q in ids[:24]}
        self.hold_q = {q: queries[q] for q in ids[24:]}
        self.qrels = qrels
        self.cache = os.path.join(self.dir.name, "emb.db")
        self.log = tune.TrialLog(os.path.join(self.dir.name, "trials.jsonl"))

    def factory(self, seq_length):
        return HashEmbedder(dim=seq_length)


class TuneBm25Tests(Harness):
    def test_baseline_first_then_every_grid_value_and_a_valid_best(self):
        best, trials = tune.tune_bm25(Retriever(self.index), self.tune_q, self.qrels, self.log, SMALL_GRIDS)
        self.assertEqual(trials[0]["name"], "baseline")
        self.assertEqual(len(trials), 1 + 2 + 2 + 2)
        self.assertEqual(best.rerank, "none")
        self.assertIn(best.w_title, (0.5, 2.0))
        rows = [json.loads(line) for line in open(self.log.path, encoding="utf-8")]
        self.assertEqual(len(rows), len(trials))  # everything tried is on disk
        self.assertTrue(all(0.0 <= t["metrics"]["ndcg@10"] <= 1.0 for t in trials))

    def test_each_stage_changes_one_setting_at_a_time(self):
        _, trials = tune.tune_bm25(Retriever(self.index), self.tune_q, self.qrels, self.log, SMALL_GRIDS)
        stopword_trials = [t for t in trials if t["name"].startswith("stopwords=")]
        self.assertEqual({t["config"]["max_terms"] for t in stopword_trials}, {None})
        self.assertEqual({t["config"]["w_title"] for t in stopword_trials}, {1.0})

    def test_the_tuned_config_is_at_least_as_good_as_the_baseline_on_the_tuning_questions(self):
        best, trials = tune.tune_bm25(Retriever(self.index), self.tune_q, self.qrels, self.log, SMALL_GRIDS)
        chosen = tune.score(Retriever(self.index), self.tune_q, self.qrels, best)[0]["ndcg@10"]
        self.assertGreaterEqual(chosen + tune.EPSILON, trials[0]["metrics"]["ndcg@10"])


class TuneSelectionTests(unittest.TestCase):
    """Controlled scores, so the test knows exactly which value should win at every step."""

    def run_with(self, ndcg, cost):
        def fake_score(retriever, queries, qrels, cfg):
            return {"ndcg@10": ndcg(cfg), "recall@100": 0.5}, 0.0

        with unittest.mock.patch.object(tune, "score", fake_score), unittest.mock.patch.object(tune, "mean_first_stage_ms", lambda r, q, c: cost(c)):
            return tune.tune_bm25(object(), {"q": "text"}, {}, tune.TrialLog(None), SMALL_GRIDS)

    def test_each_step_keeps_the_best_value_not_the_first_one(self):
        # In every grid the best value is the last one, so picking the first value (or the baseline) would fail.
        best, _ = self.run_with(lambda c: 0.10 + (0.20 if c.stopwords else 0) + (0.05 if c.max_terms == 4 else 0) + (0.10 if c.w_title == 2.0 else 0), lambda c: 1.0)
        self.assertEqual((best.stopwords, best.max_terms, best.w_title), (True, 4, 2.0))

    def test_equal_quality_goes_to_the_cheaper_setting(self):
        best, _ = self.run_with(lambda c: 0.3, lambda c: 5.0 if c.stopwords else 50.0)
        self.assertTrue(best.stopwords)
        best, _ = self.run_with(lambda c: 0.3, lambda c: 5.0 if not c.stopwords else 50.0)
        self.assertFalse(best.stopwords)

    def test_a_clearly_better_slow_setting_beats_a_cheap_one(self):
        best, _ = self.run_with(lambda c: 0.5 if not c.stopwords else 0.3, lambda c: 5.0 if c.stopwords else 500.0)
        self.assertFalse(best.stopwords)  # quality wins unless the difference is within the tolerance


class TuneRerankTests(Harness):
    def test_runs_every_variant_and_returns_a_valid_choice(self):
        seq, cfg, trials = tune.tune_rerank(self.index, self.factory, self.cache, self.tune_q, self.qrels, self.log, Config(), SMALL_GRIDS)
        per_depth = 1 + len(SMALL_GRIDS["rrf_ks"]) + len(SMALL_GRIDS["alphas"])  # dense + rrf + weighted
        self.assertEqual(len(trials), len(SMALL_GRIDS["seq_lengths"]) * len(SMALL_GRIDS["depths"]) * per_depth)
        self.assertIn(seq, SMALL_GRIDS["seq_lengths"])
        self.assertIn(cfg.rerank, ("dense", "rrf", "weighted"))
        self.assertLessEqual(cfg.rerank_depth, cfg.candidates)

    def test_the_embedding_pool_is_built_once_per_input_length(self):
        tune.tune_rerank(self.index, self.factory, self.cache, self.tune_q, self.qrels, self.log, Config(), SMALL_GRIDS)
        rows = [json.loads(line) for line in open(self.log.path, encoding="utf-8")]
        self.assertEqual(len([r for r in rows if r["name"].startswith("prefetch")]), len(SMALL_GRIDS["seq_lengths"]))


class FinalEvalTests(Harness):
    def systems(self):
        return {"baseline": Config(stopwords=False), "tuned": Config(), "hybrid": Config(rerank="weighted", alpha=0.5, candidates=30, rerank_depth=30)}

    def test_reports_metrics_intervals_latency_and_paired_comparisons(self):
        out = tune.final_eval(self.index, self.factory, self.cache, self.hold_q, self.qrels, self.systems(), 64, latency_queries=5)
        self.assertEqual(out["n_questions"], len(self.hold_q))
        for name in ("baseline", "tuned", "hybrid"):
            low, high = out[name]["ci_ndcg@10"]
            self.assertLessEqual(low, out[name]["ndcg@10"] + 1e-9)
            self.assertGreaterEqual(high, out[name]["ndcg@10"] - 1e-9)
            self.assertLessEqual(out[name]["latency"]["p50_ms"], out[name]["latency"]["p95_ms"])
        self.assertNotIn("vs_baseline_ndcg@10", out["baseline"])
        self.assertIn("diff", out["tuned"]["vs_baseline_ndcg@10"])

    def test_selection_ignores_the_held_out_questions(self):
        """Poison the held-out judgments. What gets chosen must not change; only the reported score may."""
        def choose_everything(qrels):
            best, _ = tune.tune_bm25(Retriever(self.index), self.tune_q, qrels, tune.TrialLog(None), SMALL_GRIDS)
            _, rerank_cfg, _ = tune.tune_rerank(self.index, self.factory, self.cache, self.tune_q, qrels, tune.TrialLog(None), best, SMALL_GRIDS)
            return best, rerank_cfg

        # On this tiny corpus many settings score identically, and ties are broken by measured BM25 time. Real timings
        # are noise, so use a fixed cost; otherwise this test would compare timing luck and not the selection logic.
        fixed_cost = lambda retriever, queries, cfg: float(len(repr(cfg.first_stage_key())))  # noqa: E731
        with unittest.mock.patch.object(tune, "mean_first_stage_ms", fixed_cost):
            honest = choose_everything(self.qrels)
            poisoned = dict(self.qrels)
            for q in self.hold_q:
                poisoned[q] = {"not-a-real-doc": 1}  # no run can score on these
            self.assertEqual(honest, choose_everything(poisoned))
        good = tune.final_eval(self.index, self.factory, self.cache, self.hold_q, self.qrels, {"b": Config()}, 64, latency_queries=3)["b"]["ndcg@10"]
        bad = tune.final_eval(self.index, self.factory, self.cache, self.hold_q, poisoned, {"b": Config()}, 64, latency_queries=3)["b"]["ndcg@10"]
        self.assertGreater(good, 0.0)
        self.assertEqual(bad, 0.0)


class ExperimentHelperTests(unittest.TestCase):
    def test_sampling_is_deterministic_sorted_and_bounded(self):
        ids = [f"q{i}" for i in range(100)]
        a = experiment.sample_ids(ids, 10, seed=1)
        self.assertEqual(a, experiment.sample_ids(list(reversed(ids)), 10, seed=1))
        self.assertEqual((len(a), a == sorted(a)), (10, True))
        self.assertNotEqual(a, experiment.sample_ids(ids, 10, seed=2))
        self.assertEqual(len(experiment.sample_ids(ids, 1000, seed=1)), 100)
        self.assertEqual(experiment.sample_ids(ids, 0, seed=1), sorted(ids))

    def test_the_baseline_is_the_naive_every_word_search(self):
        self.assertFalse(experiment.BASELINE.stopwords)
        self.assertEqual((experiment.BASELINE.rerank, experiment.BASELINE.max_terms, experiment.BASELINE.w_title), ("none", None, 1.0))

    def test_peak_memory_is_a_positive_number(self):
        peak = experiment.peak_memory_mb()
        self.assertTrue(peak is None or peak > 0)


class DemoTests(unittest.TestCase):
    def test_demo_runs_offline_and_reports_three_systems(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["demo"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        for label in ("BM25, every word", "BM25, stopwords removed", "BM25 + dense rerank"):
            self.assertIn(label, text)
        self.assertIn("ndcg@10", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
