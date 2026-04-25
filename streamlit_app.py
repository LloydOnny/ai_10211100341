"""
Entry point for Streamlit Cloud.
Streamlit Cloud automatically runs this file.
"""

import sys
from pathlib import Path

# Ensure src is in path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

# Import all app components
import os
import streamlit as st
from evaluation_utils import run_evaluation_bundle
from llm_utils import (
    PROMPT_TEMPLATE_IDS,
    PROMPT_TEMPLATE_LABELS,
    build_prompt,
    call_openai,
)
from pipeline_log import configure_logging, get_run_memory
from retrieval import RAGRetriever

configure_logging()

st.set_page_config(page_title="Academic City RAG Chatbot", layout="wide")
st.title("Academic City RAG Chatbot (Manual RAG + Hybrid Search)")

@st.cache_resource
def load_retriever():
    """Cache the retriever to avoid reloading FAISS index on every interaction."""
    try:
        return RAGRetriever(k=5)
    except FileNotFoundError as e:
        st.error(f"❌ Missing index files: {e}")
        st.info("**To deploy on Streamlit Cloud:**\n1. Run `python src/embedding_index.py` locally\n2. Commit the indices to git\n3. Redeploy")
        return None

try:
    retriever = load_retriever()
    if retriever is None:
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading retriever: {e}")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Get API key from secrets or user input
api_key = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    value=st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
)
template_id = st.sidebar.selectbox(
    "Response style",
    options=list(PROMPT_TEMPLATE_IDS),
    format_func=lambda x: PROMPT_TEMPLATE_LABELS.get(x, x),
    index=0,
)
use_memory = st.sidebar.checkbox(
    "Remember this conversation",
    value=False,
    help="Includes recent turns in the prompt for follow-up questions. Factual claims must still follow the retrieved chunks.",
)
if st.sidebar.button("Clear conversation history"):
    st.session_state["chat_history"] = []

query = st.text_input("Enter your query:")

if use_memory and st.session_state["chat_history"]:
    with st.expander("Conversation history (included in the next reply)", expanded=False):
        for msg in st.session_state["chat_history"][-8:]:
            st.markdown(f"**{msg.get('role', '?')}:** {msg.get('content', '')[:1200]}")

if query:
    with st.spinner("Retrieving (dense + keyword + fusion)..."):
        results, expansion_detail = retriever.retrieve(query)

    with st.expander("Query expansion & retrieval variants", expanded=False):
        st.markdown(expansion_detail.get("expansion_note", ""))
        if expansion_detail.get("matched_phrases"):
            st.caption("Matched domain phrases: " + ", ".join(expansion_detail["matched_phrases"]))
        st.write("**Retrieval variants used:**")
        for i, v in enumerate(expansion_detail.get("variants", []), start=1):
            st.code(v, language="text")
        if expansion_detail.get("synonyms_used"):
            st.caption(
                "Extra terms appended in augmented variant: "
                + ", ".join(expansion_detail["synonyms_used"])
            )

    hybrid = expansion_detail.get("hybrid_mode", False)
    st.subheader("Top retrieved chunks")
    score_help = expansion_detail.get(
        "score_interpretation",
        "Score explains retrieval ranking for this mode.",
    )
    st.caption(score_help)
    if hybrid:
        st.caption(
            "Fusion: dense FAISS (sentence embeddings) + TF-IDF keyword cosine → RRF merged ranking."
        )
    st.caption(
        "The LLM sees the same ranked chunks below (in order), constrained by prompt size limits."
    )

    id_list = expansion_detail.get("retrieved_chunk_ids") or []
    for i, (chunk, score) in enumerate(results):
        display_chunk = chunk.replace("\xa0", " ")
        label = "RRF score" if hybrid else "L2 distance"
        cid = id_list[i] if i < len(id_list) else None
        id_line = f"Chunk ID **{cid}** · " if cid is not None else ""
        st.markdown(f"**Rank {i + 1}** · {id_line}{label}: **{score:.6f}**\n\n{display_chunk}")

    if hybrid and expansion_detail.get("hybrid_breakdown"):
        with st.expander("Hybrid score breakdown (per chunk)", expanded=False):
            st.json(expansion_detail["hybrid_breakdown"])

    chunks = [c for c, _ in results]

    with st.expander("Pipeline log (structured stages)", expanded=False):
        st.code("\n".join(get_run_memory()), language="text")

    if api_key:
        hist = st.session_state["chat_history"] if use_memory else None
        with st.spinner("Generating answer with LLM..."):
            prompt = build_prompt(
                chunks,
                query,
                max_chunks=len(chunks),
                template_id=template_id,
                conversation_history=hist,
            )
            try:
                answer = call_openai(prompt, api_key=api_key)
                st.subheader("LLM response")
                st.caption(f"Response style: **{PROMPT_TEMPLATE_LABELS.get(template_id, template_id)}**")
                st.write(answer)
                st.markdown("---")
                st.caption(
                    "Part D evidence: open the three expanders below and capture them in one scroll "
                    "(or one tall screenshot)."
                )
                passage_json = []
                for i, (chunk, score) in enumerate(results):
                    cid = id_list[i] if i < len(id_list) else None
                    passage_json.append(
                        {
                            "rank": i + 1,
                            "chunk_id": cid,
                            "score": float(score),
                            "passage": chunk[:2500],
                        }
                    )
                with st.expander("Source passages & scores", expanded=False):
                    st.json(passage_json)
                with st.expander("Full prompt", expanded=False):
                    st.code(prompt)
                with st.expander("Run trace (JSON)", expanded=False):
                    st.json({"pipeline": get_run_memory()})

