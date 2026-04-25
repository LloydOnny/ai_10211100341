"""
Lloyd Onny — 10211100341

Evaluation helpers: pure-LLM baseline, duplicate-run consistency, lightweight grounding label.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from llm_utils import build_pure_llm_prompt, call_openai


def _normalize_chunk(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\xa0", " ").replace("\u200b", "").strip()


def _normalize_for_compare(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def consistency_with_duplicate(rag_prompt: str, first_answer: str, api_key: str) -> Dict[str, Any]:
    """Re-send the same RAG prompt at temperature 0; compare to first answer."""
    second = call_openai(rag_prompt, api_key=api_key, temperature=0.0)
    a, b = _normalize_for_compare(first_answer), _normalize_for_compare(second)
    return {
        "duplicate_rag_answer": second,
        "exact_match": a == b,
        "char_length_first": len(first_answer),
        "char_length_second": len(second),
    }


def grounding_label(chunks: Sequence[str], user_query: str, answer: str, api_key: str) -> str:
    """Single-token-ish classifier from the model (exam-friendly qualitative metric)."""
    joined = "\n\n---\n\n".join(_normalize_chunk(c) for c in chunks[:5])[:6000]
    judge_prompt = f"""You grade whether an ANSWER stays faithful to SOURCE excerpts for a QUESTION.

Reply with exactly one word on the first line (choose one): GROUNDED, PARTIAL, or UNSUPPORTED
- GROUNDED: every factual claim in the ANSWER is directly supported by SOURCE (or the answer says it does not know).
- PARTIAL: some claims supported, others missing support or vague.
- UNSUPPORTED: important factual claims contradict or go beyond SOURCE.

SOURCE:
{joined}

QUESTION:
{user_query}

ANSWER:
{answer}

First-line label only, then one short sentence of justification."""
    return call_openai(judge_prompt, api_key=api_key, temperature=0.0)


def pure_llm_baseline(user_query: str, api_key: str) -> str:
    return call_openai(build_pure_llm_prompt(user_query), api_key=api_key, temperature=0.0)


def run_evaluation_bundle(
    chunks: List[str],
    user_query: str,
    rag_answer: str,
    rag_prompt: str,
    api_key: str,
) -> Dict[str, Any]:
    """Run baseline + consistency + grounding label (multiple API calls)."""
    pure = pure_llm_baseline(user_query, api_key)
    consistency = consistency_with_duplicate(rag_prompt, rag_answer, api_key)
    judge = grounding_label(chunks, user_query, rag_answer, api_key)
    first_line = (judge.splitlines()[0] if judge else "").strip()
    return {
        "pure_llm_answer": pure,
        "consistency": consistency,
        "grounding_judge_raw": judge,
        "grounding_label_line": first_line,
    }
