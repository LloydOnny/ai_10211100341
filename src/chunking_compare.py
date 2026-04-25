"""
Lloyd Onny — 10211100341

Compare two PDF chunking configs (size/overlap) on retrieval overlap and chunk stats.
Does not modify production index files; runs in-memory FAISS for the comparison only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from pdf_chunking import chunk_text, extract_text_from_pdf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
CLEAN_CSV = DATA_DIR / "Ghana_Election_Result_clean.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
REPORT_DIR = ROOT / "reports"


def load_election_chunks(csv_file: Path) -> List[str]:
    df = pd.read_csv(csv_file)
    return [str(row) for row in df.to_dict(orient="records")]


def token_set(text: str) -> Set[str]:
    return {t.lower() for t in text.split() if len(t) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def build_index(chunks: Sequence[str], model: SentenceTransformer):
    emb = model.encode(list(chunks), show_progress_bar=False, convert_to_numpy=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index, emb


def search(index, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    distances, indices = index.search(query_emb, k)
    return distances[0], indices[0]


def main():
    parser = argparse.ArgumentParser(description="Compare two chunking strategies on the budget PDF.")
    parser.add_argument("--a-size", type=int, default=500)
    parser.add_argument("--a-overlap", type=int, default=50)
    parser.add_argument("--b-size", type=int, default=800)
    parser.add_argument("--b-overlap", type=int, default=120)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Missing PDF: {PDF_PATH}")
    if not CLEAN_CSV.is_file():
        raise FileNotFoundError(f"Missing cleaned CSV: {CLEAN_CSV}")

    text = extract_text_from_pdf(str(PDF_PATH))
    chunks_a_pdf = chunk_text(text, chunk_size=args.a_size, overlap=args.a_overlap)
    chunks_b_pdf = chunk_text(text, chunk_size=args.b_size, overlap=args.b_overlap)
    election = load_election_chunks(CLEAN_CSV)

    all_a = chunks_a_pdf + election
    all_b = chunks_b_pdf + election

    lens_a = [len(c) for c in chunks_a_pdf]
    lens_b = [len(c) for c in chunks_b_pdf]

    print("=== Chunking statistics (budget PDF only) ===")
    print(f"Config A (size={args.a_size}, overlap={args.a_overlap}): n={len(chunks_a_pdf)} chunks, "
          f"mean_len={np.mean(lens_a):.1f}, std_len={np.std(lens_a):.1f}")
    print(f"Config B (size={args.b_size}, overlap={args.b_overlap}): n={len(chunks_b_pdf)} chunks, "
          f"mean_len={np.mean(lens_b):.1f}, std_len={np.std(lens_b):.1f}")
    print(f"Election rows appended to each index: {len(election)} (same for both)\n")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index_a, _ = build_index(all_a, model)
    index_b, _ = build_index(all_b, model)

    default_queries = [
        "What does the 2025 budget say about fiscal deficit and debt?",
        "tax revenue mobilization Ghana 2025",
        "NPP votes Ashanti Region 2020 election results",
        "GDP growth macroeconomic outlook",
        "capital expenditure infrastructure budget",
    ]

    print("=== Per-query retrieval comparison (dense L2, same embedding model) ===")
    rows = []
    for q in default_queries:
        qv = model.encode([q], convert_to_numpy=True)
        da, ia = search(index_a, qv, args.k)
        db, ib = search(index_b, qv, args.k)
        top_a = all_a[int(ia[0])] if ia[0] >= 0 else ""
        top_b = all_b[int(ib[0])] if ib[0] >= 0 else ""
        jac = jaccard(token_set(top_a), token_set(top_b))
        row = {
            "query": q,
            "l2_top1_A": float(da[0]) if len(da) else None,
            "l2_top1_B": float(db[0]) if len(db) else None,
            "top1_len_A": len(top_a),
            "top1_len_B": len(top_b),
            "token_jaccard_top1": round(jac, 4),
        }
        rows.append(row)
        print(f"Q: {q[:70]}...")
        print(f"  L2@1  A={da[0]:.4f}  B={db[0]:.4f}  |  top1 token Jaccard={jac:.3f}")
        print(f"  top1[A] preview: {top_a[:160].replace(chr(10), ' ')}...")
        print(f"  top1[B] preview: {top_b[:160].replace(chr(10), ' ')}...")
        print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "chunking_compare_summary.txt"
    lines = [
        "Chunking comparison (Lloyd Onny — 10211100341)",
        f"A: chunk_size={args.a_size}, overlap={args.a_overlap}",
        f"B: chunk_size={args.b_size}, overlap={args.b_overlap}",
        "",
    ]
    for r in rows:
        lines.append(str(r))
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {out}")


if __name__ == "__main__":
    main()
