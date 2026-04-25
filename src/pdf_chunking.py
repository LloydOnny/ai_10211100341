"""
Lloyd Onny — 10211100341

Extract text from the 2025 budget PDF and sentence-aware chunking with overlap.
"""

import argparse
from pathlib import Path

import PyPDF2


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    import re

    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-(overlap // len(sent) + 1) :] if current else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def write_budget_chunks(chunks, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    default_pdf = root / "data" / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    default_out = root / "data" / "budget_chunks.txt"

    p = argparse.ArgumentParser(description="Chunk budget PDF for RAG indexing.")
    p.add_argument("--pdf", type=Path, default=default_pdf)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--output", type=Path, default=default_out)
    args = p.parse_args()

    text = extract_text_from_pdf(str(args.pdf))
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    write_budget_chunks(chunks, args.output)
    print(
        f"Extracted and chunked PDF into {len(chunks)} chunks "
        f"(chunk_size={args.chunk_size}, overlap={args.overlap}) -> {args.output}"
    )
