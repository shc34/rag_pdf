# -*- coding: utf-8 -*-
"""
Ingestion script.
Run once to parse, chunk and index the PDF into ChromaDB + BM25.
"""

from pathlib import Path

from src.core.config import RAW_DIR
from src.core.logger import get_logger
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import index_documents

logger = get_logger(__name__)


def run(pdf_path: Path) -> None:
    logger.info(f"Starting ingestion for: {pdf_path.name}")

    documents = parse_pdf(pdf_path)
    chunks = chunk_documents(documents)
    index_documents(chunks)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    pdf_path = RAW_DIR / "kalman_filter.pdf"
    run(pdf_path)
