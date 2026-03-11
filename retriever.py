"""
retriever.py — Query embedding + ChromaDB search
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "data" / "chroma"
COLLECTION_NAME = "clin_lit_corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._model = None
        self._collection = None

    def _load(self):
        """Lazy-load model and ChromaDB collection on first use."""
        if self._model is None:
            self._model = SentenceTransformer(EMBED_MODEL)

        if self._collection is None:
            client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_collection(COLLECTION_NAME)

    def query(self, question: str) -> list[dict]:
        """
        Embed question, retrieve top-k chunks.

        Returns list of dicts:
            {
                "text": str,
                "source": str,   # filename
                "page": int,
                "title": str,
                "distance": float,
            }
        """
        self._load()

        embedding = self._model.encode([question]).tolist()
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "title": meta.get("title", ""),
                "distance": round(dist, 4),
            })

        return chunks

    @property
    def corpus_size(self) -> int:
        """Total chunks stored in the collection."""
        try:
            self._load()
            return self._collection.count()
        except Exception:
            return 0
