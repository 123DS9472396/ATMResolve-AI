from __future__ import annotations
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.config import ROOT, VECTOR_DB_PATH, EMBEDDING_MODEL, OLLAMA_BASE_URL

DOCS_DIR = ROOT / "data" / "documents"
COLLECTION = "atmresolve_policy"

def parse_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    metadata = {"title": path.stem.replace("_", " "), "source_url": f"local://{path.name}", "source_type": "local_document", "effective_date": "unknown", "filename": path.name}
    if text.startswith("---\n"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            front = parts[1]
            text = parts[2].lstrip()
            for line in front.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"\'')
    metadata["filename"] = path.name
    return Document(page_content=text, metadata=metadata)

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

def ingest_documents(docs_dir: str | Path = DOCS_DIR) -> int:
    docs_dir = Path(docs_dir)
    paths = sorted([*docs_dir.glob("*.md"), *docs_dir.glob("*.txt")])
    if not paths:
        raise FileNotFoundError(f"No .md or .txt documents found in {docs_dir}")
    documents = [parse_document(p) for p in paths]
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, add_start_index=True)
    chunks = splitter.split_documents(documents)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    store = Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(), persist_directory=str(VECTOR_DB_PATH))
    ids = []
    for i, doc in enumerate(chunks):
        raw = f"{doc.metadata.get('filename')}:{doc.metadata.get('start_index',i)}:{doc.page_content}"
        ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    store.add_documents(chunks, ids=ids)
    return len(chunks)

def main() -> None:
    total = ingest_documents()
    print(f"Indexed {total} chunks into {VECTOR_DB_PATH} / collection={COLLECTION}")
    print("Documents are indexed locally. Inspect source_type/source_url metadata in results.")

if __name__ == "__main__":
    main()
