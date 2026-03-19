"""
RAG Ingestor — handles PDF ingestion pipeline.

RAG Enhancement Techniques used here:
- Multilingual embedding model (paraphrase-multilingual-mpnet-base-v2) for cross-lingual retrieval
- Recursive character text splitting with sentence-boundary awareness
- Chunk metadata enrichment (source filename, page number, chunk index)
- FAISS IndexFlatIP (inner product = cosine similarity after L2-norm) for fast dense retrieval
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def load_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    """Load all PDFs from directory, return list of {text, metadata} dicts."""
    docs: List[Dict[str, Any]] = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        logger.info(f"Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            docs.append(
                {
                    "text": page.page_content,
                    "metadata": {"source": pdf_path.name, "page": page.metadata.get("page", 0)},
                }
            )
    logger.info(f"Loaded {len(docs)} pages from {pdf_dir}")
    return docs


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split pages into overlapping chunks.
    Uses RecursiveCharacterTextSplitter with sentence-boundary separators.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        splits = splitter.split_text(doc["text"] or "")
        for i, split in enumerate(splits):
            if len(split.strip()) < 30:  # skip trivially short chunks
                continue
            chunks.append({"text": split.strip(), "metadata": {**doc["metadata"], "chunk_index": i}})
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def build_index(chunks: List[Dict[str, Any]], vectorstore_dir: str):
    """
    Embed chunks with multilingual SentenceTransformer.
    Build FAISS index (cosine similarity via normalized inner product).
    Persist index + metadata to disk.
    """
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks with {settings.EMBEDDING_MODEL}...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized = cosine similarity
    index.add(embeddings)

    os.makedirs(vectorstore_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(vectorstore_dir, "index.faiss"))

    with open(os.path.join(vectorstore_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved FAISS index + {len(chunks)} chunks to '{vectorstore_dir}/'")


def run_ingestion():
    docs = load_pdfs(settings.PDF_DIR)
    chunks = chunk_documents(docs)
    build_index(chunks, settings.VECTORSTORE_DIR)


if __name__ == "__main__":
    run_ingestion()

