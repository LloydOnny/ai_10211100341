"""
Lloyd Onny — 10211100341

Structured pipeline logging for the RAG flow (exam requirement: log every stage).
"""

from __future__ import annotations

import logging
from typing import Any, List

_LOGGER = logging.getLogger("rag_pipeline")

# Copy of recent log lines for Streamlit / debugging (cleared each retrieve).
_memory: List[str] = []


def configure_logging(level: int = logging.INFO) -> None:
    """Console handler once; safe to call multiple times."""
    if _LOGGER.handlers:
        _LOGGER.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | rag_pipeline | %(message)s")
    )
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(level)


def clear_run_memory() -> None:
    global _memory
    _memory = []


def log_stage(stage: str, **fields: Any) -> None:
    configure_logging()
    msg = " | ".join([f"stage={stage}"] + [f"{k}={_short(v)}" for k, v in fields.items()])
    _LOGGER.info(msg)
    _memory.append(msg)


def get_run_memory() -> List[str]:
    return list(_memory)


def _short(v: Any, max_len: int = 220) -> str:
    s = repr(v)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."
