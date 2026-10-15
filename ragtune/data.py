"""Load the BEIR Natural Questions corpus (2.68M Wikipedia passages, 1.5 GB) and its labeled test questions.

Format (BEIR): corpus.jsonl has {"_id", "title", "text"}, queries.jsonl has {"_id", "text"}, and
qrels/test.tsv has a header then "query-id <tab> corpus-id <tab> score".
"""

import json
import os
import random
import urllib.request
import zipfile

NQ_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
NQ_ZIP_BYTES = 498307926  # as served when this project was written; used only as a sanity check


def download(dest_zip, url=NQ_URL, expected_bytes=NQ_ZIP_BYTES, progress=None):
    """Download with resume. Skips work if the file is already complete."""
    have = os.path.getsize(dest_zip) if os.path.exists(dest_zip) else 0
    if expected_bytes and have == expected_bytes:
        return dest_zip
    os.makedirs(os.path.dirname(os.path.abspath(dest_zip)), exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request, timeout=60) as response, open(dest_zip, "ab" if have else "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            have += len(chunk)
            if progress:
                progress(have)
    if expected_bytes and os.path.getsize(dest_zip) != expected_bytes:
        raise RuntimeError(f"Download is {os.path.getsize(dest_zip)} bytes, expected {expected_bytes}. Delete it and retry.")
    return dest_zip


def extract(zip_path, data_dir):
    """Unzip, refusing paths that would escape data_dir."""
    root = os.path.abspath(data_dir)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            target = os.path.abspath(os.path.join(root, name))
            if not (target == root or target.startswith(root + os.sep)):
                raise RuntimeError(f"Refusing to extract {name!r}: it would write outside {data_dir}")
        archive.extractall(root)
    return os.path.join(root, "nq")


def iter_corpus(path, limit=None):
    """Yield {"doc_id", "title", "text"} one at a time, so the 1.5 GB file is never held in memory."""
    with open(path, encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if limit is not None and n >= limit:
                return
            record = json.loads(line)
            yield {"doc_id": str(record["_id"]), "title": record.get("title") or "", "text": record.get("text") or ""}


def load_queries(path):
    with open(path, encoding="utf-8") as handle:
        return {str(r["_id"]): r["text"] for r in map(json.loads, handle)}


def load_qrels(path):
    """Return {query_id: {doc_id: gain}} for positive judgments only."""
    qrels = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:2] != ["query-id", "corpus-id"]:
            raise ValueError(f"Unexpected qrels header: {header!r}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, gain = parts[0], parts[1], int(float(parts[2]))
            if gain > 0:
                qrels.setdefault(query_id, {})[doc_id] = gain
    return qrels


def split_queries(query_ids, seed=13, tune_fraction=0.5):
    """Split once, deterministically, into a tuning half and a held-out half.

    All sweeps choose their winner on the tuning half only. The held-out half is touched once, at the end,
    so the reported numbers are not flattered by the choice of settings.
    """
    ids = sorted(query_ids)
    random.Random(seed).shuffle(ids)
    cut = round(len(ids) * tune_fraction)
    return sorted(ids[:cut]), sorted(ids[cut:])
