"""
Hybrid Retriever with RAG enhancement techniques:

1. DENSE RETRIEVAL   — Multilingual FAISS embeddings (semantic similarity, cross-lingual)
2. SPARSE RETRIEVAL  — BM25 (keyword overlap, good for plant names, scientific terms)
3. FUSION            — Reciprocal Rank Fusion (RRF) to merge dense + sparse rankings
4. RERANKING         — Cross-encoder reranker (ms-marco-MiniLM) for final scoring

Memory optimizations for Render free tier (512MB):
- L12 → L6 embedding model (~180MB saved)
- torch single-threaded to reduce thread stack overhead
- CrossEncoder max_length 512 → 256 (halves reranker peak RAM)
- TOKENIZERS_PARALLELISM disabled to prevent HF tokenizer forking
- Lazy loading: models load on first request, not at startup
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Must be set BEFORE torch/transformers are imported ──────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import get_settings
from app.core.logger import get_logger

# Cap PyTorch threads after import
torch.set_num_threads(1)

logger = get_logger(__name__)
settings = get_settings()


class HybridRetriever:
    def __init__(self):
        self._embedding_model: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None
        self._index = None
        self._chunks: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        vdir = Path(settings.VECTORSTORE_DIR)
        logger.info("Loading vector store and models...")

        index_path = vdir / "index.faiss"
        chunks_path = vdir / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Vector store not found in '{vdir}'. Run: python scripts/build_index.py"
            )

        # Load FAISS index
        self._index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded: dim={self._index.d}, vectors={self._index.ntotal}")

        # Load chunks
        with open(chunks_path, encoding="utf-8") as f:
            self._chunks = json.load(f)
        logger.info(f"Chunks loaded: {len(self._chunks)}")

        # Load embedding model (L6 — ~170MB vs L12 ~350MB, still multilingual)
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self._embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu",
        )

        # Build BM25 index on tokenized chunk texts
        tokenized = [str(c.get("text", "")).lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

        # Cross-encoder reranker — max_length 256 halves peak RAM vs 512
        logger.info("Loading cross-encoder reranker...")
        self._reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=256,
            device="cpu",
        )

        self._loaded = True
        logger.info(
            f"Retriever ready: {len(self._chunks)} chunks, "
            f"FAISS dim={self._index.d}, "
            f"embedding={settings.EMBEDDING_MODEL}"
        )

    # ------------------------------------------------------------------
    # DENSE RETRIEVAL
    # ------------------------------------------------------------------
    def _dense_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return (chunk_idx, score) pairs from FAISS."""
        assert self._embedding_model is not None
        assert self._index is not None

        q_emb = (
            self._embedding_model
            .encode(
                [query],
                normalize_embeddings=True,
                batch_size=1,          # single query — no batching overhead
                show_progress_bar=False,
            )
            .astype("float32")
        )
        scores, indices = self._index.search(q_emb, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if int(idx) >= 0
        ]

    # ------------------------------------------------------------------
    # SPARSE RETRIEVAL (BM25)
    # ------------------------------------------------------------------
    def _bm25_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return (chunk_idx, score) pairs from BM25."""
        assert self._bm25 is not None

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if float(scores[i]) > 0.0]

    # ------------------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # ------------------------------------------------------------------
    @staticmethod
    def _rrf_merge(
        ranked_lists: List[List[Tuple[int, float]]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """Merge multiple ranked lists using RRF formula: 1 / (k + rank)."""
        scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # CROSS-ENCODER RERANKING
    # ------------------------------------------------------------------
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Score (query, chunk) pairs with cross-encoder, return top_k."""
        assert self._reranker is not None

        if not candidates:
            return []

        pairs = [(query, c.get("text", "")) for c in candidates]
        ce_scores = self._reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        ranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # PUBLIC RETRIEVE METHOD
    # ------------------------------------------------------------------
    def retrieve(self, query: str, lang: str = "en") -> List[Dict[str, Any]]:
        """
        Full hybrid retrieve pipeline.
        For Urdu queries: relies on cross-lingual multilingual embeddings for dense retrieval.
        Lazy-loads models on first call.
        """
        self.load()

        dense_results = self._dense_retrieve(query, settings.TOP_K_DENSE)
        bm25_results = self._bm25_retrieve(query, settings.TOP_K_BM25)

        # RRF merge
        merged = self._rrf_merge([dense_results, bm25_results])

        # Deduplicate and fetch chunk objects
        seen: set = set()
        candidates: List[Dict[str, Any]] = []
        for idx, _ in merged:
            if idx not in seen and 0 <= idx < len(self._chunks):
                seen.add(idx)
                candidates.append(self._chunks[idx])

        # Rerank with cross-encoder
        final_chunks = self._rerank(query, candidates, settings.TOP_K_RERANK)

        logger.info(
            f"Retrieved {len(final_chunks)} chunks "
            f"(dense={len(dense_results)}, bm25={len(bm25_results)}) "
            f"lang={lang}"
        )
        return final_chunks


# ---------------------------------------------------------------------------
# Singleton — lazy: instantiated on first get_retriever() call, not at import
# ---------------------------------------------------------------------------
_retriever_instance: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
        _retriever_instance.load()
    return _retriever_instance
