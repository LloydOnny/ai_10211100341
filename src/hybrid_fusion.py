"""
Lloyd Onny — 10211100341

Reciprocal Rank Fusion (RRF) for merging dense-vector and sparse keyword rankings.
Manual implementation — no retrieval frameworks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def reciprocal_rank_fusion(
    rankings: List[List[int]],
    *,
    rrf_k: int = 60,
) -> Dict[int, float]:
    """
    Combine ordered rankings (best chunk index first per list).

    Score(chunk) = sum_i  1 / (rrf_k + rank_i(chunk))
    If a chunk does not appear in ranking i, it contributes 0 from that ranker.

    Reference: Cormack, Clarke, Büttacher — RRF is commonly used for hybrid retrieval.
    """
    scores: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        if not ranking:
            continue
        for rank, chunk_idx in enumerate(ranking, start=1):
            scores[int(chunk_idx)] += 1.0 / (rrf_k + rank)
    return dict(scores)
