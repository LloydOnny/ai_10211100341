"""
Lloyd Onny — 10211100341

Innovation: domain-aware query expansion for Ghana election + fiscal/budget RAG.
Expands short user queries with controlled synonyms / related phrases before retrieval
to improve recall without high-level RAG frameworks.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# Multi-word phrases (longest-first match) → related phrasing for retrieval diversity
_PHRASE_EXPANSIONS: Sequence[Tuple[str, Sequence[str]]] = (
    (
        "election results",
        ("poll outcomes", "voting results", "electoral tallies", "ballot outcomes"),
    ),
    (
        "general election",
        ("national polls", "parliamentary election", "presidential election"),
    ),
    (
        "vote share",
        ("percentage of votes", "share of ballots", "electoral proportion"),
    ),
    (
        "budget statement",
        ("fiscal policy", "budget speech", "national budget"),
    ),
    (
        "fiscal deficit",
        ("budget shortfall", "fiscal gap", "deficit financing"),
    ),
    (
        "tax revenue",
        ("tax receipts", "revenue mobilization", "tax collection"),
    ),
    (
        "gdp growth",
        ("economic growth", "output growth", "growth rate"),
    ),
    (
        "public debt",
        ("government debt", "national debt", "debt stock"),
    ),
)

# Single-token expansions (Ghana elections + macro/fiscal vocabulary)
_WORD_EXPANSIONS: dict[str, Tuple[str, ...]] = {
    "election": ("poll", "ballot", "electoral", "vote", "polling"),
    "elections": ("polls", "ballots", "votes"),
    "vote": ("ballot", "polling", "cast"),
    "votes": ("ballots", "polls"),
    "party": ("political party", "partisan"),
    "candidate": ("aspirant", "contestant"),
    "constituency": ("electoral area", "district seat"),
    "region": ("administrative region", "area"),
    "budget": ("fiscal", "appropriation", "expenditure framework"),
    "revenue": ("receipts", "inflows", "income"),
    "expenditure": ("spending", "outlay", "costs"),
    "minister": ("ministry", "cabinet"),
    "economy": ("economic", "macroeconomic"),
    "inflation": ("price increases", "cpi"),
    "tax": ("taxation", "levy"),
    "ghana": ("republic", "country"),
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def _phrase_variants(query: str, limit: int) -> List[str]:
    """Add query strings with phrase-level synonym substitution."""
    out: List[str] = []
    q_lower = query.lower()
    for phrase, alts in _PHRASE_EXPANSIONS:
        if phrase in q_lower:
            for alt in alts:
                if len(out) >= limit:
                    return out
                replaced = re.compile(re.escape(phrase), re.IGNORECASE).sub(alt, query, count=1)
                if replaced.strip() and replaced.strip() != query.strip():
                    out.append(replaced.strip())
    return out


def _augment_with_word_synonyms(query: str) -> str | None:
    """Append a small set of missing related terms (same sentence embedding space)."""
    q_lower = query.lower()
    extra: List[str] = []
    seen = set()
    for tok in _tokenize(query):
        if tok in _WORD_EXPANSIONS:
            for syn in _WORD_EXPANSIONS[tok]:
                s = syn.lower()
                if s not in q_lower and s not in seen:
                    seen.add(s)
                    extra.append(syn)
        if len(extra) >= 8:
            break
    if not extra:
        return None
    return f"{query.strip()} {' '.join(extra[:6])}"


def build_retrieval_variants(
    query: str,
    *,
    max_variants: int = 5,
    include_augmented: bool = True,
) -> Tuple[List[str], dict]:
    """
    Produce distinct query strings used for multi-query retrieval.

    Returns:
        variants: non-empty list of queries (always includes original first).
        detail: metadata for logging / UI (innovation transparency).
    """
    original = query.strip()
    if not original:
        return [original], {"original_query": original, "variants": [], "synonyms_used": []}

    q_lower = original.lower()
    matched_phrases = [phrase for phrase, _ in _PHRASE_EXPANSIONS if phrase in q_lower]

    variants: List[str] = [original]
    synonyms_used: List[str] = []

    # Phrase substitutions (different surface forms → broader recall)
    for v in _phrase_variants(original, limit=max(0, max_variants - len(variants))):
        if v not in variants:
            variants.append(v)
        if len(variants) >= max_variants:
            return variants[:max_variants], _detail(original, variants, synonyms_used, matched_phrases)

    # Single augmented query with extra domain terms
    if include_augmented and len(variants) < max_variants:
        aug = _augment_with_word_synonyms(original)
        if aug and aug not in variants:
            variants.append(aug)
            synonyms_used = [t for t in aug[len(original) :].strip().split() if t]

    return variants[:max_variants], _detail(original, variants[:max_variants], synonyms_used, matched_phrases)


def _detail(
    original: str,
    variants: List[str],
    synonyms_used: List[str],
    matched_phrases: List[str],
) -> dict:
    return {
        "original_query": original,
        "variants": variants,
        "synonyms_used": synonyms_used,
        "matched_phrases": matched_phrases,
        "expansion_note": (
            "Innovation — domain-aware query expansion: "
            "multiple embeddings merged by best vector distance per chunk."
        ),
    }
