"""
ingest.py — PDF → ChromaDB pipeline
Run once (or re-run) to populate the local ChromaDB collection.

Usage:
    python ingest.py
"""

import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
PDF_DIR = Path(__file__).parent / "data" / "pdfs"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma"
COLLECTION_NAME = "clin_lit_corpus"
CHUNK_TOKENS = 500      # approximate tokens per chunk
OVERLAP_TOKENS = 50     # overlap between chunks
WORDS_PER_TOKEN = 0.75  # rough words-per-token estimate for splitting
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_WORDS = int(CHUNK_TOKENS * WORDS_PER_TOKEN)
OVERLAP_WORDS = int(OVERLAP_TOKENS * WORDS_PER_TOKEN)


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return list of {text, page} dicts for every page in the PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"text": text, "page": i + 1})
    return pages


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


def ingest_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Drop your seed PDFs there and re-run.")
        return

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete and recreate collection so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        pages = extract_pages(pdf_path)
        print(f"  Extracted {len(pages)} pages")

        doc_chunks = []
        metadatas = []
        ids = []

        for page_data in pages:
            chunks = chunk_text(page_data["text"], CHUNK_WORDS, OVERLAP_WORDS)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{pdf_path.stem}_p{page_data['page']}_c{chunk_idx}"
                doc_chunks.append(chunk)
                metadatas.append({
                    "source": pdf_path.name,
                    "page": page_data["page"],
                    "title": pdf_path.stem.replace("_", " ").replace("-", " "),
                })
                ids.append(chunk_id)

        if not doc_chunks:
            print(f"  Warning: no text extracted from {pdf_path.name}")
            continue

        print(f"  Embedding {len(doc_chunks)} chunks...")
        embeddings = model.encode(doc_chunks, show_progress_bar=True).tolist()

        # Upsert in batches to avoid memory spikes
        batch_size = 100
        for i in range(0, len(doc_chunks), batch_size):
            collection.add(
                documents=doc_chunks[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

        total_chunks += len(doc_chunks)
        print(f"  Stored {len(doc_chunks)} chunks from {pdf_path.name}")

    print(f"\nIngestion complete. {total_chunks} total chunks across {len(pdf_files)} PDFs.")
    print(f"ChromaDB stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_pdfs()
