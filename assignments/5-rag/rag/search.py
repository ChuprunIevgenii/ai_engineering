"""Hybrid search: BM25 + dense + reciprocal rank fusion (ready — the retrieval from homework 4)."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from . import config
from .ingest import Chunk

TOKEN_RE = re.compile(r"[a-zа-яіїєґ0-9ʼ']+(?:[-.@][a-zа-яіїєґ0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; words longer than 6 letters are cut to a 6-letter prefix (crude Ukrainian stemmer).

    Tokens with digits (ПЗ-204, 3.101, 1150) stay whole, so exact codes still match exactly in BM25.
    """
    toks = TOKEN_RE.findall(text.lower())
    return [t if any(ch.isdigit() for ch in t) or len(t) <= 6 else t[:6] for t in toks]


class Index:
    """Search over all chunks. `dense=False` skips the embedding model (BM25 only) for fast offline checks."""

    def __init__(self, chunks: list[Chunk], dense: bool = True):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        texts = [f"{c.title}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.encoder = None
        self.emb = None
        if dense:
            self.encoder, self.emb = _embed(texts)

    def search(self, query: str, k: int = config.TOP_K,
               where: Callable[[Chunk], bool] | None = None) -> list[Chunk]:
        """Top-k chunks for `query`; `where` filters chunks before ranking (kind, validity, ...)."""
        allowed = [where is None or where(c) for c in self.chunks]
        bm25 = self.bm25.get_scores(tokenize(query))
        # A chunk with no query term has BM25 score 0 and no rank: ranking zeros in arbitrary order would
        # give dense-only hits a second RRF bonus and push the exact match out of the top-k.
        rankings = [(bm25, bm25 > 0)]
        if self.encoder is not None:
            dense = self.emb @ self.encoder.encode([query], normalize_embeddings=True)[0]
            rankings.append((dense, np.ones(len(dense), dtype=bool)))
        scores: dict[int, float] = defaultdict(float)
        for arr, ranked in rankings:
            order = [i for i in np.argsort(-arr) if allowed[i] and ranked[i]][: config.RRF_DEPTH]
            for rank, i in enumerate(order):
                scores[i] += 1 / (config.RRF_K + rank)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.chunks[i] for i in best]


def _embed(texts: list[str]):
    """Encode chunk texts once; cached in .cache/ keyed by model and texts."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(config.ENCODER)
    key = hashlib.sha256((config.ENCODER + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = config.CACHE_DIR / f"emb_{key}.npy"
    if path.exists():
        return encoder, np.load(path)
    emb = encoder.encode(texts, normalize_embeddings=True, batch_size=64)
    config.CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, emb)
    return encoder, emb
