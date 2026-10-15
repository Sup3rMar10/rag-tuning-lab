"""Embeddings for re-ranking the candidates BM25 retrieves.

Embedding all 2.68M passages on a CPU would take days, so the dense model only ever sees the few
hundred candidates per question that BM25 returns. Vectors are cached on disk, so a sweep that changes
only the fusion settings costs no embedding time at all.
"""

import hashlib
import re
import sqlite3
import time

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic stand-in for tests: hashed bag-of-words, so similar texts really do score higher."""

    def __init__(self, dim=128):
        self.dim = dim
        self.name = f"hash{dim}"

    def encode(self, texts, batch_size=64):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0 if digest[4] % 2 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEmbedder:
    """A real model on CPU. `max_seq_length` trades accuracy for speed: longer inputs cost more."""

    def __init__(self, model_name, max_seq_length=128, query_prefix="", passage_prefix="", device="cpu"):
        from sentence_transformers import SentenceTransformer  # imported here so the core works without it

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.name = f"{model_name}@{max_seq_length}"

    def encode(self, texts, batch_size=64):
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)


def _round_trip(vectors):
    """Cached vectors are stored as float16. Rounding fresh ones the same way keeps hit and miss paths identical."""
    return vectors.astype(np.float16).astype(np.float32)


class EmbeddingCache:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=OFF")
        self.con.execute("CREATE TABLE IF NOT EXISTS emb(model TEXT, key TEXT, vec BLOB, PRIMARY KEY(model, key)) WITHOUT ROWID")

    def close(self):
        self.con.commit()
        self.con.close()

    def get_many(self, model, keys):
        found = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for key, blob in self.con.execute(f"SELECT key, vec FROM emb WHERE model=? AND key IN ({marks})", [model, *chunk]):
                found[key] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        return found

    def put_many(self, model, items):
        self.con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", [(model, k, v.astype(np.float16).tobytes()) for k, v in items.items()])
        self.con.commit()

    def count(self, model=None):
        if model is None:
            return self.con.execute("SELECT count(*) FROM emb").fetchone()[0]
        return self.con.execute("SELECT count(*) FROM emb WHERE model=?", (model,)).fetchone()[0]


def query_key(text):
    return "q:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def embed_items(embedder, cache, items, batch_size=64, stats=None, progress=None):
    """items: list of (key, text). Returns a float32 matrix in the same order. Only cache misses are embedded."""
    stats = stats if stats is not None else {}
    keys = [k for k, _ in items]
    have = cache.get_many(embedder.name, keys) if cache is not None else {}
    stats["hits"] = stats.get("hits", 0) + len(have)
    todo = [(k, t) for k, t in dict(items).items() if k not in have]
    stats["misses"] = stats.get("misses", 0) + len(todo)
    started = time.perf_counter()
    for start in range(0, len(todo), batch_size * 8):
        part = todo[start:start + batch_size * 8]
        vectors = _round_trip(embedder.encode([t for _, t in part], batch_size=batch_size))
        fresh = {k: v for (k, _), v in zip(part, vectors)}
        have.update(fresh)
        if cache is not None:
            cache.put_many(embedder.name, fresh)
        if progress:
            progress(min(start + len(part), len(todo)), len(todo))
    stats["embed_seconds"] = stats.get("embed_seconds", 0.0) + time.perf_counter() - started
    return np.stack([have[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
