"""
Lloyd Onny — 10211100341

Builds FAISS dense index + TF-IDF sparse index for hybrid retrieval,
and chunk metadata from cleaned CSV + PDF-derived chunks.
"""

from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_FILE = DATA_DIR / "budget_chunks.txt"
CLEAN_CSV = DATA_DIR / "Ghana_Election_Result_clean.csv"
INDEX_FILE = DATA_DIR / "faiss_index.bin"
META_FILE = DATA_DIR / "chunk_metadata.csv"
TFIDF_INDEX_FILE = DATA_DIR / "tfidf_keyword_index.joblib"


def load_chunks(chunk_file: Path):
    with open(chunk_file, "r", encoding="utf-8") as f:
        raw = f.read()
    return [c.strip() for c in raw.split("\n---\n") if c.strip()]


def load_election_chunks(csv_file: Path):
    df = pd.read_csv(csv_file)
    return [str(row) for row in df.to_dict(orient="records")]


if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL)
    budget_chunks = load_chunks(CHUNK_FILE)
    election_chunks = load_election_chunks(CLEAN_CSV)
    all_chunks = budget_chunks + election_chunks

    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    pd.DataFrame({"chunk": all_chunks}).to_csv(META_FILE, index=False)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=40_000,
        min_df=2,
        dtype=np.float64,
    )
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    joblib.dump(
        {"vectorizer": vectorizer, "matrix": tfidf_matrix},
        TFIDF_INDEX_FILE,
    )

    print(f"Embedded {len(all_chunks)} chunks and wrote index to {INDEX_FILE}.")
    print(f"TF-IDF keyword index saved to {TFIDF_INDEX_FILE} (shape={tfidf_matrix.shape}).")
