from __future__ import annotations
from pathlib import Path
from langchain_chroma import Chroma
from src.rag.ingest import COLLECTION, get_embeddings
from src.config import VECTOR_DB_PATH

def retrieve_policy(query: str, k: int = 4) -> list[dict]:
    if not query.strip():
        return []
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError("Chroma index missing. Run: python -m src.rag.ingest")
    store = Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(), persist_directory=str(VECTOR_DB_PATH))
    try:
        docs = store.similarity_search(query, k=k)
    except Exception as exc:
        raise RuntimeError(f"Policy retrieval failed. Re-run ingestion and verify Ollama embeddings: {exc}") from exc
    results=[]
    for index, doc in enumerate(docs, start=1):
        results.append({
            "citation_id": f"POL-{index}",
            "title": doc.metadata.get("title", "Untitled document"),
            "source_url": doc.metadata.get("source_url", "unknown"),
            "source_type": doc.metadata.get("source_type", "unknown"),
            "effective_date": doc.metadata.get("effective_date", "unknown"),
            "filename": doc.metadata.get("filename", "unknown"),
            "content": doc.page_content,
        })
    return results
