"""
Lloyd Onny — 10211100341

Prompt construction (multiple templates) and OpenAI chat call for the RAG pipeline.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import openai

from pipeline_log import log_stage

DEFAULT_MAX_CONTEXT_CHUNKS = 5
MAX_PROMPT_CHARS = 14_000

# Multiple prompt templates selectable from the UI.
PROMPT_TEMPLATE_IDS = ("grounded", "structured", "concise")
PROMPT_TEMPLATE_LABELS: Dict[str, str] = {
    "grounded": "Grounded — default anti-hallucination rules",
    "structured": "Structured — sections: Summary / Key facts / Gaps",
    "concise": "Concise — short bullets, tight length",
}


def _normalize_chunk(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
        .strip()
    )


def _format_conversation_history(history: Optional[Sequence[Dict[str, str]]], max_messages: int = 8) -> str:
    if not history:
        return ""
    tail = list(history)[-max_messages:]
    lines = []
    for msg in tail:
        role = (msg.get("role") or "user").strip()
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > 900:
            content = content[:897] + "..."
        lines.append(f"{role.upper()}: {content}")
    if not lines:
        return ""
    return (
        "Conversation memory (for follow-up wording and pronouns only; "
        "do not treat past turns as factual sources):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def _build_context_block(context_chunks, max_chunks: int) -> Tuple[str, int]:
    chunks = [_normalize_chunk(c) for c in context_chunks[:max_chunks]]
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        if total + len(chunk) > MAX_PROMPT_CHARS:
            break
        parts.append(f"Chunk {i + 1}:\n{chunk}")
        total += len(chunk)
    context = "\n\n---\n\n".join(parts)
    return context, len(parts)


def build_pure_llm_prompt(user_query: str) -> str:
    """Baseline prompt without retrieved documents (general-knowledge answer)."""
    return (
        "Answer using your general knowledge only. "
        "Do not pretend you were given private documents.\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )


def build_prompt(
    context_chunks,
    user_query,
    *,
    max_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    template_id: str = "grounded",
    conversation_history: Optional[Sequence[Dict[str, str]]] = None,
):
    """
    Build the user message for the chat model. template_id selects prompt style.
    conversation_history: optional list of {"role": "user"|"assistant", "content": "..."} for multi-turn context.
    """
    if template_id not in PROMPT_TEMPLATE_IDS:
        template_id = "grounded"

    context, n_parts = _build_context_block(context_chunks, max_chunks)
    hist = _format_conversation_history(conversation_history)

    if template_id == "structured":
        rules = """RULES:
- Use ONLY information from the retrieved chunks below. If the chunks do not contain enough information to answer, say exactly: I don't know based on the retrieved documents.
- Organize your answer under these headings (use the headings verbatim): Summary | Key facts | Gaps/uncertainties
- Under Key facts, use short bullets. Copy numbers (votes, percentages, years) exactly from the chunks.
- Under Gaps/uncertainties, state what the chunks do not cover."""
    elif template_id == "concise":
        rules = """RULES:
- Use ONLY information from the retrieved chunks below. If the chunks do not contain enough information, reply in one sentence: I don't know based on the retrieved documents.
- At most 5 short bullet points total; no paragraph prose unless a single bullet needs two lines.
- Copy numbers exactly from the chunks; no tables unless the user asked for a table."""
    else:
        rules = """RULES:
- Use ONLY information from the retrieved chunks below. If the chunks do not contain enough information to answer, say exactly: I don't know based on the retrieved documents.
- Write your answer in plain English sentences or short bullet points. Do NOT dump Python dict representations, JSON, or raw record strings unless the user explicitly asks for raw data.
- When citing numbers (votes, percentages, years), copy them exactly from the chunks.
- Briefly tie facts to their source where helpful (e.g., year and region)."""

    prompt = f"""You are an assistant for Academic City helping users understand Ghana election data and budget documents.

{hist}{rules}

Retrieved chunks:
{context}

User question: {user_query}

Answer:"""
    log_stage(
        "prompt_built",
        template_id=template_id,
        chunks_in_prompt=n_parts,
        prompt_chars=len(prompt),
        user_query_preview=user_query[:240],
        memory_turns=len(conversation_history or []),
    )
    return prompt


def call_openai(prompt, api_key=None, model="gpt-3.5-turbo", temperature: float = 0.1):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not set.")
    log_stage("llm_request_start", model=model, prompt_chars=len(prompt), temperature=temperature)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=temperature,
    )
    text = response.choices[0].message.content.strip()
    log_stage(
        "llm_response_done",
        response_chars=len(text),
        finish_reason=getattr(response.choices[0], "finish_reason", None),
    )
    return text
