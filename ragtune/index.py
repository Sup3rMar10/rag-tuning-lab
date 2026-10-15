"""BM25 retrieval over the whole corpus with SQLite FTS5 (built into Python, no extra dependency).

Two build paths are kept on purpose: a naive one (one transaction per row, default settings) and an
optimized one, so the speedup is measured rather than claimed.
"""

import math
import os
import pathlib
import re
import sqlite3
import time
from dataclasses import dataclass

TOKENIZERS = {
    "porter": "porter unicode61 remove_diacritics 2",  # stems words: "running" matches "run"
    "plain": "unicode61 remove_diacritics 2",          # no stemming
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being below between both but by can
    could did do does doing down during each few for from further had has have having he her here hers him his how i if in into
    is it its me more most my no nor not of off on once only or other our out over own same she should so some such than that the
    their them then there these they this those through to too under until up very was we were what when where which while who whom
    why will with would you your""".split()
)


@dataclass
class BuildStats:
    rows: int
    seconds: float
    db_bytes: int
    mode: str
    tokenizer: str

    @property
    def rows_per_second(self):
        return self.rows / self.seconds if self.seconds else 0.0


def tokenize_query(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_index(db_path, passages, tokenizer="porter", detail="full", optimized=True, batch=20000, limit=None, progress=None):
    """Build the index from an iterable of {"doc_id", "title", "text"}. Replaces any existing file."""
    if tokenizer not in TOKENIZERS:
        raise ValueError(f"tokenizer must be one of {sorted(TOKENIZERS)}")
    if detail not in ("full", "column", "none"):
        raise ValueError("detail must be full, column or none")
    if os.path.exists(db_path):
        os.remove(db_path)
    started = time.perf_counter()
    con = sqlite3.connect(db_path, isolation_level=None)  # autocommit; transactions are explicit below
    if optimized:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA locking_mode=EXCLUSIVE")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-262144")  # 256 MB page cache
    con.execute(
        f"CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, title, text, "
        f"tokenize='{TOKENIZERS[tokenizer]}', detail={detail})"
    )
    con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    insert = "INSERT INTO passages(doc_id, title, text) VALUES (?, ?, ?)"
    rows = 0
    if optimized:
        con.execute("BEGIN")
        pending = []
        for passage in passages:
            if limit is not None and rows >= limit:
                break
            pending.append((passage["doc_id"], passage["title"], passage["text"]))
            rows += 1
            if len(pending) >= batch:
                con.executemany(insert, pending)
                pending.clear()
                if progress:
                    progress(rows)
        if pending:
            con.executemany(insert, pending)
        con.execute("COMMIT")
        con.execute("INSERT INTO passages(passages) VALUES('optimize')")  # merge segments for faster queries
    else:
        for passage in passages:
            if limit is not None and rows >= limit:
                break
            con.execute(insert, (passage["doc_id"], passage["title"], passage["text"]))  # its own transaction
            rows += 1
            if progress and rows % batch == 0:
                progress(rows)
    for key, value in (("rows", str(rows)), ("tokenizer", tokenizer), ("detail", detail), ("optimized_build", str(optimized))):
        con.execute("INSERT INTO meta VALUES (?, ?)", (key, value))
    con.close()
    return BuildStats(rows, time.perf_counter() - started, os.path.getsize(db_path), "optimized" if optimized else "naive", tokenizer)


class Bm25Index:
    """Read-only BM25 search. `search` returns hits best first, with positive scores (higher is better)."""

    def __init__(self, db_path, cache_mb=256):
        uri = pathlib.Path(db_path).resolve().as_uri() + "?mode=ro"
        self.con = sqlite3.connect(uri, uri=True)
        self.con.execute(f"PRAGMA cache_size=-{int(cache_mb) * 1024}")
        self.meta = dict(self.con.execute("SELECT key, value FROM meta"))
        self.n_docs = int(self.meta["rows"])
        self._df = {}

    def close(self):
        self.con.close()

    def doc_freq(self, term):
        """How many passages contain the term (memoised, because common terms repeat across questions)."""
        if term not in self._df:
            self._df[term] = self.con.execute("SELECT count(*) FROM passages WHERE passages MATCH ?", (f'"{term}"',)).fetchone()[0]
        return self._df[term]

    def idf(self, term):
        df = self.doc_freq(term)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def query_terms(self, query, stopwords=True, max_terms=None):
        terms = list(dict.fromkeys(tokenize_query(query)))
        if stopwords:
            terms = [t for t in terms if t not in STOPWORDS] or terms
        if max_terms and len(terms) > max_terms:
            keep = set(sorted(terms, key=lambda t: (-self.idf(t), t))[:max_terms])  # keep the most informative terms
            terms = [t for t in terms if t in keep]
        return terms

    def search(self, query, k=100, w_title=1.0, w_text=1.0, stopwords=True, max_terms=None):
        terms = self.query_terms(query, stopwords, max_terms)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.con.execute(
            "SELECT doc_id, title, text, bm25(passages, 0.0, ?, ?) AS s FROM passages WHERE passages MATCH ? ORDER BY s LIMIT ?",
            (w_title, w_text, match, k),
        ).fetchall()
        return [{"doc_id": d, "title": t, "text": x, "score": -s} for d, t, x, s in rows]
