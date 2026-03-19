"""
Run this ONCE (or whenever PDFs change) to build the FAISS vector index.
Usage: python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingestor import run_ingestion  # noqa: E402


if __name__ == "__main__":
    print("Starting ingestion pipeline...")
    run_ingestion()
    print("Done! Vector index saved to vectorstore/")

