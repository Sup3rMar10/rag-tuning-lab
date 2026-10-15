"""Tests for embeddings, the cache, the pipeline and the generation stage. They use a deterministic fake embedder."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtune import metrics, rag  # noqa: E402
from ragtune.dense import EmbeddingCache, HashEmbedder, embed_items, query_key  # noqa: E402
from ragtune.index import Bm25Index, build_index  # noqa: E402
from ragtune.pipeline import Config, Retriever, apply_rerank, measure_latency, run_queries  # noqa: E402
from ragtune.synthetic import make_synthetic  # noqa: E402


def tmpdir():
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__(64)
        self.encoded = 0

    def encode(self, texts, batch_size=64):
        self.encoded += len(texts)
        return super().encode(texts, batch_size)


class DenseTests(unittest.TestCase):
    def test_hash_embedder_is_normalized_deterministic_and_meaningful(self):
        e = HashEmbedder(128)
        a, b, c = e.encode(["volcano lava ash", "volcano lava ash", "violin cello oboe"])
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(float(a @ b), float(a @ c))
        self.assertEqual(float(np.linalg.norm(e.encode([""])[0])), 0.0)  # empty text does not divide by zero

    def test_cache_only_embeds_misses_and_hit_equals_miss(self):
        e = CountingEmbedder()
        with tmpdir() as d:
            cache = EmbeddingCache(os.path.join(d, "c.db"))
            items = [("a", "alpha beta"), ("b", "gamma delta")]
            stats = {}
            first = embed_items(e, cache, items, stats=stats)
            self.assertEqual((e.encoded, stats["hits"], stats["misses"]), (2, 0, 2))
            stats2 = {}
            second = embed_items(e, cache, items + [("c", "epsilon")], stats=stats2)
            self.assertEqual((e.encoded, stats2["hits"], stats2["misses"]), (3, 2, 1))  # only "c" was embedded
            self.assertTrue(np.array_equal(first, second[:2]))  # a cached vector is bit-identical to the fresh one
            self.assertEqual(cache.count(e.name), 3)
            cache.close()

    def test_cache_is_keyed_by_model(self):
        with tmpdir() as d:
            cache = EmbeddingCache(os.path.join(d, "c.db"))
            embed_items(HashEmbedder(64), cache, [("a", "x")])
            embed_items(HashEmbedder(32), cache, [("a", "x")])
            self.assertEqual((cache.count("hash64"), cache.count("hash32")), (1, 1))
            cache.close()

    def test_duplicate_keys_in_one_call_are_embedded_once(self):
        e = CountingEmbedder()
        out = embed_items(e, None, [("a", "x y"), ("a", "x y"), ("b", "z")])
        self.assertEqual((e.encoded, out.shape[0]), (2, 3))
        self.assertTrue(np.array_equal(out[0], out[1]))

    def test_query_keys_never_collide_with_doc_ids(self):
        self.assertTrue(query_key("anything").startswith("q:"))
        self.assertEqual(query_key("same"), query_key("same"))
        self.assertNotEqual(query_key("one"), query_key("two"))


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            Config(rerank="magic")
        with self.assertRaises(ValueError):
            Config(candidates=10, rerank_depth=50)
        with self.assertRaises(ValueError):
            Config(alpha=1.5)
        self.assertEqual(Config().as_dict()["candidates"], 100)

    def test_first_stage_key_ignores_second_stage_settings(self):
        self.assertEqual(Config(rerank="rrf", alpha=0.3).first_stage_key(), Config(rerank="none").first_stage_key())
        self.assertNotEqual(Config(w_title=2.0).first_stage_key(), Config().first_stage_key())


def hit(doc, score=1.0, title="t", text="x"):
    return {"doc_id": doc, "title": title, "text": text, "score": score}


class ApplyRerankTests(unittest.TestCase):
    HITS = [hit("a", 9.0), hit("b", 8.0), hit("c", 7.0), hit("d", 1.0)]

    def order(self, cfg, dense):
        return [h["doc_id"] for h in apply_rerank(cfg, self.HITS, dense)]

    def test_none_keeps_bm25_order_and_respects_final_k(self):
        self.assertEqual(self.order(Config(rerank="none", final_k=3), []), ["a", "b", "c"])

    def test_dense_orders_by_cosine_and_keeps_the_tail_in_bm25_order(self):
        cfg = Config(rerank="dense", candidates=4, rerank_depth=3)
        self.assertEqual(self.order(cfg, [0.1, 0.9, 0.5]), ["b", "c", "a", "d"])  # d was beyond the depth: stays last

    def test_weighted_extremes(self):
        dense = [0.1, 0.9, 0.5]
        self.assertEqual(self.order(Config(rerank="weighted", candidates=4, rerank_depth=3, alpha=1.0), dense), ["a", "b", "c", "d"])
        self.assertEqual(self.order(Config(rerank="weighted", candidates=4, rerank_depth=3, alpha=0.0), dense), ["b", "c", "a", "d"])

    def test_rrf_blends_both_rankings(self):
        # bm25 order is a,b,c and dense order is c,a,b. With k=60 the scores are a: 1/61+1/62, c: 1/63+1/61, b: 1/62+1/63,
        # so the blend (a, c, b) matches neither input ranking.
        cfg = Config(rerank="rrf", candidates=4, rerank_depth=3, rrf_k=60)
        self.assertEqual(self.order(cfg, [0.5, 0.1, 0.9]), ["a", "c", "b", "d"])

    def test_empty_hits(self):
        self.assertEqual(apply_rerank(Config(rerank="dense"), [], []), [])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir()
        passages, self.queries, self.qrels = make_synthetic()
        path = os.path.join(self.dir.name, "i.db")
        build_index(path, passages)
        self.index = Bm25Index(path)
        self.embedder = CountingEmbedder()
        self.cache = EmbeddingCache(os.path.join(self.dir.name, "c.db"))
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.index.close)

    def test_run_scores_well_and_dense_rerank_does_not_lose_candidates(self):
        retriever = Retriever(self.index, self.embedder, self.cache)
        base = run_queries(retriever, self.queries, Config())
        reranked = run_queries(retriever, self.queries, Config(rerank="weighted", alpha=0.5))
        self.assertGreater(metrics.evaluate(base, self.qrels)["recall@100"], 0.9)
        for qid in self.queries:
            self.assertEqual(sorted(base[qid]), sorted(reranked[qid]))  # re-ranking only reorders

    def test_first_stage_is_computed_once_per_question_and_setting(self):
        retriever = Retriever(self.index, self.embedder, self.cache)
        calls = []
        real = self.index.search
        self.index.search = lambda *a, **k: calls.append(1) or real(*a, **k)
        run_queries(retriever, self.queries, Config(rerank="none"))
        run_queries(retriever, self.queries, Config(rerank="rrf"))
        run_queries(retriever, self.queries, Config(rerank="weighted", alpha=0.2))
        self.assertEqual(len(calls), len(self.queries))

    def test_prefetch_embeds_everything_once_and_sweeps_then_cost_nothing(self):
        retriever = Retriever(self.index, self.embedder, self.cache)
        cfg = Config(rerank="weighted", candidates=50, rerank_depth=50)
        retriever.prefetch(self.queries.values(), cfg)
        embedded_after_prefetch = self.embedder.encoded
        for alpha in (0.0, 0.3, 0.7, 1.0):
            run_queries(retriever, self.queries, Config(rerank="weighted", candidates=50, rerank_depth=50, alpha=alpha))
        self.assertEqual(self.embedder.encoded, embedded_after_prefetch)

    def test_a_deeper_rerank_depth_needs_more_embeddings(self):
        retriever = Retriever(self.index, self.embedder, self.cache)
        retriever.prefetch(self.queries.values(), Config(rerank="dense", candidates=20, rerank_depth=10))
        shallow = self.embedder.encoded
        retriever.prefetch(self.queries.values(), Config(rerank="dense", candidates=20, rerank_depth=20))
        self.assertGreater(self.embedder.encoded, shallow)

    def test_latency_times_real_searches_of_the_question_text(self):
        """Regression: it once looped over question IDs, searched for strings that match nothing, and reported ~0 ms."""
        searched = []
        real_search = self.index.search

        def spy(query, **kwargs):
            hits = real_search(query, **kwargs)
            searched.append((query, len(hits)))
            return hits

        self.index.search = spy
        self.assertTrue(set(self.queries) & set(self.queries.values()) == set())  # ids and texts differ, so a mix-up is visible
        measure_latency(self.index, self.embedder, Config(rerank="none"), self.queries, n=5)
        self.assertEqual([q for q, _ in searched], list(self.queries.values())[:5])  # the question text, in order
        self.assertTrue(all(n > 0 for _, n in searched))  # each search found something, so real work was timed

    def test_latency_measure_is_cold_and_reports_percentiles(self):
        stats = measure_latency(self.index, self.embedder, Config(rerank="dense", candidates=30, rerank_depth=30), self.queries, n=10)
        self.assertEqual(stats["n"], 10)
        self.assertLessEqual(stats["p50_ms"], stats["p95_ms"])
        self.assertGreater(stats["mean_ms"], 0)
        before = self.embedder.encoded
        measure_latency(self.index, self.embedder, Config(rerank="none"), self.queries, n=5)
        self.assertEqual(self.embedder.encoded, before)  # no embedding when there is no re-ranking


class RagTests(unittest.TestCase):
    HITS = [
        hit("1", title="Vesuvius", text="Mount Vesuvius is a volcano in Italy. It erupted in 79 AD and buried Pompeii."),
        hit("2", title="Etna", text="Etna is a volcano in Sicily. It is the tallest active volcano in Europe."),
        hit("3", title="Cake", text="A chocolate dessert baked in a hot oven."),
    ]

    def test_prompt_numbers_passages_and_states_the_rules(self):
        prompt, used = rag.build_prompt("what buried Pompeii?", self.HITS)
        self.assertEqual(len(used), 3)
        self.assertIn("[1] Vesuvius:", prompt)
        self.assertIn("[3] Cake:", prompt)
        self.assertIn("using only the numbered passages", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_prompt_respects_the_token_budget_but_always_keeps_the_best_passage(self):
        _, used = rag.build_prompt("q?", self.HITS, token_budget=60)
        self.assertLess(len(used), 3)
        _, used = rag.build_prompt("q?", self.HITS, token_budget=1)
        self.assertEqual([h["doc_id"] for h in used], ["1"])
        long_hit = hit("x", text="word " * 500)
        prompt, used = rag.build_prompt("q?", [long_hit], token_budget=100)
        self.assertEqual(len(used), 1)
        self.assertLess(rag.estimate_tokens(prompt), 200)  # the lone passage was trimmed, not dropped

    def test_extractive_answer_picks_the_best_matching_sentence_with_its_source(self):
        result = rag.extractive_answer("when did Vesuvius erupt and bury Pompeii", self.HITS)
        self.assertIn("79 AD", result["answer"])
        self.assertEqual(result["citations"], [1])

    def test_stemmer_matches_word_forms(self):
        for a, b in [("erupt", "erupted"), ("bury", "buried"), ("volcano", "volcanoes"), ("bake", "baking"), ("run", "runs")]:
            self.assertEqual(rag.stem(a), rag.stem(b), (a, b))
        self.assertNotEqual(rag.stem("cat"), rag.stem("dog"))
        self.assertEqual(rag.stem("is"), "is")  # short words are left alone

    def test_extractive_answer_admits_when_nothing_matches(self):
        result = rag.extractive_answer("zebra migration patterns", self.HITS)
        self.assertEqual(result["citations"], [])
        self.assertIn("do not know", result["answer"])

    def test_citation_check_separates_real_from_invented_sources(self):
        self.assertEqual(rag.check_citations("It erupted [1], as Etna did [2] and also [7] and [0].", 3), ([1, 2], [0, 7]))
        self.assertEqual(rag.check_citations("no citations here", 3), ([], []))

    def test_answer_with_a_model_uses_the_prompt_and_flags_invented_citations(self):
        seen = {}

        def fake_llm(prompt):
            seen["prompt"] = prompt
            return "  It buried Pompeii in 79 AD [1] [9].  "

        result = rag.answer("what buried Pompeii?", self.HITS, llm=fake_llm)
        self.assertEqual((result["method"], result["citations"], result["invalid_citations"]), ("llm", [1], [9]))
        self.assertEqual(result["sources"][0]["doc_id"], "1")
        self.assertIn("Question: what buried Pompeii?", seen["prompt"])
        self.assertEqual(result["answer"], "It buried Pompeii in 79 AD [1] [9].")

    def test_answer_without_a_model_is_extractive_and_k_limits_the_passages(self):
        result = rag.answer("what buried Pompeii?", self.HITS)
        self.assertEqual(result["method"], "extractive")
        self.assertEqual(result["sources"][0]["doc_id"], "1")
        prompts = []
        rag.answer("q?", self.HITS, llm=lambda p: prompts.append(p) or "x", k=1)
        self.assertNotIn("[2] Etna", prompts[0])  # the instructions mention "[2]" too, so look for the passage itself


if __name__ == "__main__":
    unittest.main(verbosity=1)
