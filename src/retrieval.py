"""
Lloyd Onny — 10211100341

Hybrid retrieval: dense FAISS (SentenceTransformers) + sparse TF-IDF keyword search,
fused with Reciprocal Rank Fusion (RRF). Query expansion merges multi-query dense scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from hybrid_fusion import reciprocal_rank_fusion
from pipeline_log import clear_run_memory, log_stage
from query_expansion import build_retrieval_variants

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = DATA_DIR / "faiss_index.bin"
META_FILE = DATA_DIR / "chunk_metadata.csv"
TFIDF_INDEX_FILE = DATA_DIR / "tfidf_keyword_index.joblib"

RRF_K = 60
RANK_POOL = 200


class RAGRetriever:
    """
    Hybrid search: vector (L2 on embeddings) + keyword (TF-IDF cosine).
    Final ranking = RRF over dense-order and sparse-order candidate lists.
    """

    def __init__(self, k: int = 5):
        self.k = k
        log_stage("init_start", k=k)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        if not INDEX_FILE.is_file():
            raise FileNotFoundError(
                f"Missing FAISS index at {INDEX_FILE}. From project root run: python src/embedding_index.py"
            )
        self.index = faiss.read_index(str(INDEX_FILE))
        self.chunks = pd.read_csv(META_FILE)["chunk"].tolist()

        self._tfidf_vec = None
        self._tfidf_mat = None
        self._hybrid_enabled = False
        if TFIDF_INDEX_FILE.is_file():
            blob = joblib.load(TFIDF_INDEX_FILE)
            self._tfidf_vec = blob["vectorizer"]
            self._tfidf_mat = blob["matrix"]
            self._hybrid_enabled = True
            log_stage(
                "init_keyword_index_loaded",
                path=str(TFIDF_INDEX_FILE),
                sparse_shape=getattr(self._tfidf_mat, "shape", None),
            )
        else:
            log_stage(
                "init_keyword_index_missing",
                warning="Run embedding_index.py to enable hybrid TF-IDF search.",
            )

        log_stage("init_done", num_chunks=len(self.chunks), hybrid=self._hybrid_enabled)

    def retrieve(self, query: str):
        clear_run_memory()
        log_stage("query_received", query=query[:500])

        variants, expansion_detail = build_retrieval_variants(query.strip())
        log_stage(
            "query_expansion",
            num_variants=len(variants),
            variants=variants,
            matched_phrases=expansion_detail.get("matched_phrases", []),
        )

        per_search = self.k if len(variants) == 1 else min(self.k * 2, len(self.chunks))

        best_dist: dict[int, float] = {}
        for variant in variants:
            if not variant:
                continue
            query_emb = self.model.encode([variant], convert_to_numpy=True)
            distances, indices = self.index.search(query_emb, per_search)
            row_d, row_i = distances[0], indices[0]
            for col, chunk_idx in enumerate(row_i):
                if chunk_idx < 0:
                    continue
                d = float(row_d[col])
                chunk_idx = int(chunk_idx)
                if chunk_idx not in best_dist or d < best_dist[chunk_idx]:
                    best_dist[chunk_idx] = d

        dense_ordered = sorted(best_dist.keys(), key=lambda i: best_dist[i])[:RANK_POOL]
        log_stage(
            "dense_retrieval",
            candidates=len(best_dist),
            pool_preview=dense_ordered[:15],
            best_l2=min(best_dist.values()) if best_dist else None,
        )

        if not self._hybrid_enabled:
            sorted_ids = dense_ordered[: self.k]
            results = [(self.chunks[i], best_dist[i]) for i in sorted_ids]
            expansion_detail["retrieved_chunk_ids"] = [int(i) for i in sorted_ids]
            expansion_detail["hybrid_mode"] = False
            expansion_detail["score_interpretation"] = (
                "Dense-only fallback: score is L2 distance (lower is closer)."
            )
            expansion_detail["retrieval_variants_used"] = variants
            expansion_detail["merge_strategy"] = (
                "Dense-only: minimum L2 distance per chunk across expanded query embeddings. "
                "Rebuild with python src/embedding_index.py to enable TF-IDF hybrid fusion."
            )
            log_stage(
                "fusion_skipped",
                reason="tfidf_keyword_index_missing",
                top_ids=sorted_ids,
            )
            log_stage("retrieve_done", num_results=len(results), mode="dense_only")
            return results, expansion_detail

        best_cos: dict[int, float] = {}
        for variant in variants:
            if not variant:
                continue
            qv = self._tfidf_vec.transform([variant])
            sims = cosine_similarity(qv, self._tfidf_mat)[0]
            for i, s in enumerate(sims):
                s = float(s)
                if i not in best_cos or s > best_cos[i]:
                    best_cos[i] = s

        sparse_ordered = sorted(best_cos.keys(), key=lambda i: -best_cos[i])[:RANK_POOL]
        log_stage(
            "sparse_retrieval",
            candidates=len(best_cos),
            pool_preview=sparse_ordered[:15],
            best_cos=max(best_cos.values()) if best_cos else None,
        )

        fused = reciprocal_rank_fusion([dense_ordered, sparse_ordered], rrf_k=RRF_K)
        top_ids = sorted(fused.keys(), key=lambda i: -fused[i])[: self.k]

        expansion_detail["hybrid_mode"] = True
        expansion_detail["fusion"] = "Reciprocal Rank Fusion (dense L2 ranking + sparse TF-IDF cosine)"
        expansion_detail["score_interpretation"] = (
            "Hybrid RRF score — higher is better (combined dense + keyword rank lists)."
        )
        expansion_detail["rrf_k"] = RRF_K

        breakdown = []
        results = []
        for cid in top_ids:
            breakdown.append(
                {
                    "chunk_index": cid,
                    "rrf_score": round(fused[cid], 6),
                    "l2_distance": round(best_dist.get(cid, float("nan")), 4)
                    if cid in best_dist
                    else None,
                    "tfidf_cosine": round(best_cos.get(cid, float("nan")), 4)
                    if cid in best_cos
                    else None,
                }
            )
            results.append((self.chunks[cid], fused[cid]))

        expansion_detail["hybrid_breakdown"] = breakdown
        expansion_detail["retrieved_chunk_ids"] = [int(i) for i in top_ids]
        expansion_detail["retrieval_variants_used"] = variants
        expansion_detail["merge_strategy"] = (
            "Hybrid: Reciprocal Rank Fusion of dense-vector ranking (FAISS L2) "
            "and sparse keyword ranking (TF-IDF cosine); per-chunk best dense/sparse scores "
            "across query expansion variants."
        )
        log_stage(
            "fusion_rrf",
            top_chunk_ids=top_ids,
            rrf_scores=[round(fused[i], 5) for i in top_ids],
        )
        log_stage("retrieve_done", num_results=len(results), mode="hybrid_rrf")
        return results, expansion_detail


if __name__ == "__main__":
    os.chdir(ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    q = " ".join(sys.argv[1:]).strip() or input("Enter your query: ").strip()
    retriever = RAGRetriever(k=5)
    results, detail = retriever.retrieve(q)
    print("Variants used:", detail.get("variants"))
    print("Hybrid:", detail.get("hybrid_mode"))
    print("Top results:")
    for i, (chunk, score) in enumerate(results):
        print(f"[{i+1}] score={score:.5f}\n{chunk[:400]}...\n---")
