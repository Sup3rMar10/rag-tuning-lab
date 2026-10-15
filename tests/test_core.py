"""Tests for data loading, metrics, fusion and the BM25 index. None of them needs the 1.5 GB corpus."""

import math
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtune import data, fusion, metrics  # noqa: E402
from ragtune.index import Bm25Index, build_index  # noqa: E402
from ragtune.synthetic import make_synthetic  # noqa: E402


def tmpdir():
    d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    return d


class MetricsTests(unittest.TestCase):
    def test_ndcg_matches_a_hand_computed_value(self):
        # relevant b and d at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3)
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics.ndcg_at_k(["a", "b", "c", "d"], {"b": 1, "d": 1}, 10), expected)
        self.assertAlmostEqual(expected, 0.65092, places=4)

    def test_graded_gains_and_cutoff(self):
        # a is worth 3, b is worth 1. Ranking [b, a]: DCG = 1/1 + 3/log2(3); ideal = 3/1 + 1/log2(3)
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics.ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 10), expected)
        self.assertAlmostEqual(metrics.ndcg_at_k(["x", "y", "a"], {"a": 1}, 2), 0.0)  # relevant doc is below the cutoff

    def test_perfect_and_empty_rankings(self):
        self.assertAlmostEqual(metrics.ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 10), 1.0)
        self.assertEqual(metrics.ndcg_at_k([], {"a": 1}, 10), 0.0)
        self.assertEqual(metrics.ndcg_at_k(["a"], {}, 10), 0.0)

    def test_recall_and_mrr(self):
        self.assertEqual(metrics.recall_at_k(["a", "b", "c"], {"b": 1, "z": 1}, 2), 0.5)
        self.assertEqual(metrics.recall_at_k(["a", "b", "c"], {"b": 1, "z": 1}, 1), 0.0)
        self.assertEqual(metrics.mrr_at_k(["a", "b", "c"], {"c": 1, "b": 1}, 10), 0.5)
        self.assertEqual(metrics.mrr_at_k(["a", "b", "c"], {"c": 1}, 2), 0.0)  # beyond the cutoff

    def test_duplicates_in_a_ranking_cannot_inflate_recall(self):
        self.assertEqual(metrics.recall_at_k(["a", "a", "a"], {"a": 1, "b": 1}, 10), 0.5)

    def test_evaluate_scores_missing_questions_as_zero(self):
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        result = metrics.evaluate({"q1": ["a"]}, qrels, ks=(10,))
        self.assertAlmostEqual(result["mrr@10"], 0.5)  # q1 scores 1, q2 was not answered so scores 0
        self.assertAlmostEqual(result["recall@10"], 0.5)

    def test_bootstrap_is_deterministic_and_brackets_the_mean(self):
        values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        low, high = metrics.bootstrap_ci(values, seed=3)
        self.assertEqual((low, high), metrics.bootstrap_ci(values, seed=3))
        self.assertLess(low, 0.625)
        self.assertGreater(high, 0.625)
        self.assertEqual(metrics.bootstrap_ci([]), (0.0, 0.0))

    def test_paired_bootstrap_detects_a_consistent_gain_and_no_gain(self):
        a = {f"q{i}": 0.2 for i in range(50)}
        better = {f"q{i}": 0.5 for i in range(50)}
        result = metrics.paired_bootstrap(a, better)
        self.assertAlmostEqual(result["diff"], 0.3)
        self.assertGreater(result["ci"][0], 0)
        self.assertEqual(result["p_better"], 1.0)
        same = metrics.paired_bootstrap(a, dict(a))
        self.assertEqual((same["diff"], same["ci"]), (0.0, (0.0, 0.0)))


class FusionTests(unittest.TestCase):
    def test_rrf_uses_the_documented_formula_and_breaks_ties_by_id(self):
        fused = dict(fusion.rrf([["a", "b", "c"], ["c", "b", "a"]], k=60))
        self.assertAlmostEqual(fused["b"], 2 / 62)
        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 63)
        order = [doc for doc, _ in fusion.rrf([["a", "b", "c"], ["c", "b", "a"]], k=60)]
        self.assertEqual(order, ["a", "c", "b"])  # a and c tie exactly, so id order decides

    def test_rrf_handles_docs_missing_from_one_list(self):
        fused = dict(fusion.rrf([["a", "b"], ["c"]], k=10))
        self.assertAlmostEqual(fused["c"], 1 / 11)
        self.assertAlmostEqual(fused["a"], 1 / 11)

    def test_minmax_and_a_list_with_no_signal(self):
        self.assertEqual(fusion.minmax({"a": 2, "b": 4, "c": 3}), {"a": 0.0, "b": 1.0, "c": 0.5})
        self.assertEqual(fusion.minmax({"a": 5, "b": 5}), {"a": 0.0, "b": 0.0})
        self.assertEqual(fusion.minmax({}), {})

    def test_weighted_extremes_follow_one_side(self):
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"x": 3.0, "y": 2.0, "z": 1.0}
        self.assertEqual([d for d, _ in fusion.weighted(a, b, alpha=1.0)], ["z", "y", "x"])
        self.assertEqual([d for d, _ in fusion.weighted(a, b, alpha=0.0)], ["x", "y", "z"])
        mid = dict(fusion.weighted(a, b, alpha=0.5))
        self.assertAlmostEqual(mid["y"], 0.5)

    def test_weighted_treats_a_missing_doc_as_the_minimum(self):
        fused = dict(fusion.weighted({"x": 1.0, "y": 2.0}, {"x": 5.0}, alpha=0.5))
        self.assertAlmostEqual(fused["y"], 0.5)  # y: 0.5 * 1.0 (best on a) + 0.5 * 0 (missing from b)


def tiny_corpus():
    return [
        {"doc_id": "volc", "title": "Mount Vesuvius", "text": "The volcano erupted and buried the city under ash."},
        {"doc_id": "rail", "title": "Railway", "text": "The locomotive pulled carriages along the viaduct."},
        {"doc_id": "bake", "title": "Volcano cake", "text": "A chocolate dessert baked in a hot oven."},
        {"doc_id": "zyz", "title": "Zyzzyva", "text": "A genus of weevils native to the tropics, common common common."},
    ]


class IndexTests(unittest.TestCase):
    def build(self, directory, passages=None, **kwargs):
        path = os.path.join(directory, "idx.db")
        stats = build_index(path, passages if passages is not None else tiny_corpus(), **kwargs)
        return path, stats

    def test_search_finds_the_matching_passage_first(self):
        with tmpdir() as d:
            path, stats = self.build(d)
            index = Bm25Index(path)
            hits = index.search("what erupted and buried the city")
            self.assertEqual(hits[0]["doc_id"], "volc")
            self.assertTrue(all(h["score"] > 0 for h in hits))
            self.assertEqual((stats.rows, index.n_docs), (4, 4))
            index.close()

    def test_porter_stemming_matches_word_forms_and_plain_does_not(self):
        with tmpdir() as d:
            stemmed = Bm25Index(self.build(os.path.join(d), tokenizer="porter")[0])
            self.assertIn("volc", [h["doc_id"] for h in stemmed.search("volcanoes erupting")])
            stemmed.close()
        with tmpdir() as d:
            plain = Bm25Index(self.build(d, tokenizer="plain")[0])
            self.assertEqual(plain.search("volcanoes erupting"), [])
            plain.close()

    def test_column_weights_change_the_ranking(self):
        with tmpdir() as d:
            index = Bm25Index(self.build(d)[0])
            title_heavy = index.search("volcano", w_title=10.0, w_text=0.1)
            text_heavy = index.search("volcano", w_title=0.1, w_text=10.0)
            self.assertEqual(title_heavy[0]["doc_id"], "bake")  # "volcano" is in its title, and only in its title
            self.assertEqual(text_heavy[0]["doc_id"], "volc")   # its text mentions the volcano
            index.close()

    def test_stopwords_are_dropped_unless_that_leaves_nothing(self):
        with tmpdir() as d:
            index = Bm25Index(self.build(d)[0])
            self.assertEqual(index.query_terms("what is the volcano"), ["volcano"])
            self.assertEqual(index.query_terms("what is the"), ["what", "is", "the"])
            self.assertEqual(index.query_terms("what is the volcano", stopwords=False), ["what", "is", "the", "volcano"])
            index.close()

    def test_max_terms_keeps_the_most_informative_terms(self):
        with tmpdir() as d:
            index = Bm25Index(self.build(d)[0])
            # "common" is in 1 passage, "volcano" in 2 (title and text), "city" in 1: rarest first, ties by name
            kept = index.query_terms("volcano city zyzzyva common", max_terms=2)
            self.assertEqual(len(kept), 2)
            self.assertNotIn("volcano", kept)
            index.close()

    def test_hostile_query_text_cannot_break_the_search_syntax(self):
        with tmpdir() as d:
            index = Bm25Index(self.build(d)[0])
            for query in ['"volcano" OR (', "NEAR(volcano city)", "volcano*", "title:volcano AND NOT", "'; DROP TABLE passages;--", "", "   "]:
                index.search(query)  # must not raise
            self.assertEqual(index.search(""), [])
            index.close()

    def test_naive_and_optimized_builds_give_identical_results(self):
        passages, queries, _ = make_synthetic(n_docs=80, n_queries=8)
        with tmpdir() as d:
            fast_path, fast = self.build(os.path.join(d), passages, optimized=True)
            fast_hits = {q: [h["doc_id"] for h in Bm25Index(fast_path).search(text, k=20)] for q, text in queries.items()}
        with tmpdir() as d:
            slow_path, slow = self.build(d, passages, optimized=False)
            slow_hits = {q: [h["doc_id"] for h in Bm25Index(slow_path).search(text, k=20)] for q, text in queries.items()}
        self.assertEqual(fast_hits, slow_hits)
        self.assertEqual((fast.mode, slow.mode), ("optimized", "naive"))

    def test_detail_modes_rank_the_same(self):
        passages, queries, _ = make_synthetic(n_docs=80, n_queries=8)
        results = {}
        for detail in ("full", "column", "none"):
            with tmpdir() as d:
                path, _ = self.build(d, passages, detail=detail)
                index = Bm25Index(path)
                results[detail] = {q: [h["doc_id"] for h in index.search(t, k=20)] for q, t in queries.items()}
                index.close()
        self.assertEqual(results["full"], results["column"])
        self.assertEqual(results["full"], results["none"])

    def test_limit_and_progress_and_validation(self):
        seen = []
        with tmpdir() as d:
            path, stats = self.build(d, make_synthetic(60, 4)[0], limit=25, batch=10, progress=seen.append)
            self.assertEqual(stats.rows, 25)
            self.assertEqual(seen, [10, 20])
        with tmpdir() as d:
            with self.assertRaises(ValueError):
                self.build(d, tokenizer="nope")
            with self.assertRaises(ValueError):
                self.build(d, detail="nope")

    def test_the_index_is_opened_read_only(self):
        with tmpdir() as d:
            index = Bm25Index(self.build(d)[0])
            with self.assertRaises(sqlite3.OperationalError):
                index.con.execute("INSERT INTO passages(doc_id, title, text) VALUES ('x','y','z')")
            index.close()

    def test_search_on_the_synthetic_corpus_has_real_recall(self):
        passages, queries, qrels = make_synthetic()
        with tmpdir() as d:
            index = Bm25Index(self.build(d, passages)[0])
            run = {q: [h["doc_id"] for h in index.search(text, k=100)] for q, text in queries.items()}
            scores = metrics.evaluate(run, qrels)
            self.assertGreater(scores["recall@100"], 0.9)
            index.close()


class DataTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self):
        ids = [f"q{i}" for i in range(101)]
        tune, held = data.split_queries(ids, seed=5)
        self.assertEqual((tune, held), data.split_queries(list(reversed(ids)), seed=5))  # input order does not matter
        self.assertEqual(set(tune) | set(held), set(ids))
        self.assertFalse(set(tune) & set(held))
        self.assertEqual(len(tune), 50)
        self.assertNotEqual(tune, data.split_queries(ids, seed=6)[0])

    def test_loaders(self):
        with tmpdir() as d:
            corpus = os.path.join(d, "corpus.jsonl")
            with open(corpus, "w", encoding="utf-8") as f:
                f.write('{"_id": "1", "title": "A", "text": "alpha"}\n{"_id": 2, "text": "beta"}\n{"_id": "3", "title": "C", "text": "gamma"}\n')
            self.assertEqual([p["doc_id"] for p in data.iter_corpus(corpus)], ["1", "2", "3"])
            self.assertEqual(next(data.iter_corpus(corpus))["title"], "A")
            self.assertEqual(len(list(data.iter_corpus(corpus, limit=2))), 2)
            self.assertEqual(list(data.iter_corpus(corpus, limit=2))[1]["title"], "")  # missing title becomes empty
            queries = os.path.join(d, "queries.jsonl")
            with open(queries, "w", encoding="utf-8") as f:
                f.write('{"_id": "q1", "text": "who?"}\n')
            self.assertEqual(data.load_queries(queries), {"q1": "who?"})

    def test_qrels_keep_only_positive_judgments_and_check_the_header(self):
        with tmpdir() as d:
            path = os.path.join(d, "test.tsv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\nq2\td3\t2\n")
            self.assertEqual(data.load_qrels(path), {"q1": {"d1": 1}, "q2": {"d3": 2}})
            bad = os.path.join(d, "bad.tsv")
            with open(bad, "w", encoding="utf-8") as f:
                f.write("a\tb\tc\nq1\td1\t1\n")
            with self.assertRaises(ValueError):
                data.load_qrels(bad)

    def test_extract_refuses_paths_that_escape(self):
        with tmpdir() as d:
            archive = os.path.join(d, "evil.zip")
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escaped.txt", "x")
            with self.assertRaises(RuntimeError):
                data.extract(archive, os.path.join(d, "out"))
            self.assertFalse(os.path.exists(os.path.join(d, "escaped.txt")))
            good = os.path.join(d, "good.zip")
            with zipfile.ZipFile(good, "w") as z:
                z.writestr("nq/queries.jsonl", "{}")
            self.assertTrue(os.path.exists(os.path.join(data.extract(good, os.path.join(d, "out2")), "queries.jsonl")))


PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


class _RangeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        header = self.headers.get("Range")
        start = int(header.split("=")[1].split("-")[0]) if header else 0
        body = PAYLOAD[start:]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def serve(self):
        server = HTTPServer(("127.0.0.1", 0), _RangeHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/f.zip"

    def test_full_download_and_skip_when_complete(self):
        url = self.serve()
        with tmpdir() as d:
            dest = os.path.join(d, "f.zip")
            data.download(dest, url=url, expected_bytes=len(PAYLOAD))
            self.assertEqual(open(dest, "rb").read(), PAYLOAD)
            calls = []
            data.download(dest, url="http://127.0.0.1:1/never", expected_bytes=len(PAYLOAD), progress=calls.append)
            self.assertEqual(calls, [])  # already complete, so it did not touch the network

    def test_an_interrupted_download_resumes_instead_of_restarting(self):
        url = self.serve()
        with tmpdir() as d:
            dest = os.path.join(d, "f.zip")
            with open(dest, "wb") as f:
                f.write(PAYLOAD[:30000])
            seen = []
            data.download(dest, url=url, expected_bytes=len(PAYLOAD), progress=seen.append)
            self.assertEqual(open(dest, "rb").read(), PAYLOAD)
            self.assertEqual(seen[0], 30000 + min(1 << 20, len(PAYLOAD) - 30000))  # it continued from byte 30000

    def test_a_wrong_size_is_reported(self):
        url = self.serve()
        with tmpdir() as d:
            with self.assertRaisesRegex(RuntimeError, "expected 5"):
                data.download(os.path.join(d, "f.zip"), url=url, expected_bytes=5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
