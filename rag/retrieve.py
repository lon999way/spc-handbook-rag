"""Hybrid retrieval: BM25 keyword search + dense embeddings, merged with reciprocal rank fusion.

Embeddings: BAAI/bge-small-en-v1.5 via fastembed (runs locally on CPU, MIT license).
Usage: python -m rag.retrieve "how do I compute control limits for an X-bar chart"
       (first run builds data/embeddings.npy; it is rebuilt automatically when chunks change)
"""
import hashlib
import json
import re
import sys
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import DATA, OUT as CHUNKS

EMB = DATA / "embeddings.npy"
EMB_SHA = DATA / "embeddings.sha256"  # hash of chunks.jsonl the cache was built from
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval convention
RRF_K = 60


NORMALIZE_VERSION = "2"  # bump when normalize() changes so the embedding cache is rebuilt


def normalize(text):
    """Formulas are MathJax LaTeX: turn \\hat{C}_{pk} into "Cpk" so keyword and embedding search both match."""
    text = re.sub(r"\\(?:hat|bar|tilde|mbox|mathrm|text)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"_\{(\w+)\}|_(\w)", r"\1\2", text)
    text = re.sub(r"\$\$|\\[()\[\]]", " ", text)
    return re.sub(r"\s+", " ", text)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize(text).lower())


@lru_cache(maxsize=1)
def load():
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    docs = [normalize(f"{c['title']}\n{c['text']}") for c in chunks]
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    digest = hashlib.sha256(CHUNKS.read_bytes() + NORMALIZE_VERSION.encode()).hexdigest()
    if EMB.exists() and EMB_SHA.exists() and EMB_SHA.read_text() == digest:
        vecs = np.load(EMB)
    else:  # first run, or chunks changed since the cache was built
        vecs = np.array(list(_embedder().embed(docs)), dtype=np.float32)
        np.save(EMB, vecs)
        EMB_SHA.write_text(digest)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, bm25, vecs


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL)


def search(query, k=5, pool=30):
    """Return the top-k chunks with their fused score and the best dense cosine similarity."""
    chunks, bm25, vecs = load()
    kw = np.argsort(-bm25.get_scores(tokenize(query)))[:pool]
    q = np.array(next(iter(_embedder().embed([QUERY_PREFIX + normalize(query)]))), dtype=np.float32)
    cos = vecs @ (q / np.linalg.norm(q))
    dense = np.argsort(-cos)[:pool]
    fused = {}
    for ranking in (kw, dense):
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**chunks[i], "score": round(fused[i], 4), "cosine": round(float(cos[i]), 3)} for i in top]


if __name__ == "__main__":
    for hit in search(" ".join(sys.argv[1:]) or "what is Cpk"):
        print(f"{hit['id']}  cos={hit['cosine']}  {hit['title']}\n    {hit['url']}\n    {hit['text'][:160]}...")
