# -*- coding: utf-8 -*-
"""
Ingestion script.
Run once to parse, chunk and index the PDF into ChromaDB + BM25.
"""

import argparse
from pathlib import Path

from src.core.config import RAW_DIR
from src.core.logger import get_logger
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import index_documents

logger = get_logger(__name__)


def run(pdf_path: Path, corpus: str) -> None:
    logger.info(f"Starting ingestion for: {pdf_path.name} (corpus='{corpus}')")

    documents = parse_pdf(pdf_path)
    chunks = chunk_documents(documents)
    index_documents(chunks, corpus=corpus)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into a named corpus.")
    parser.add_argument("--corpus", required=True, help="Corpus name (e.g. 'zola', 'balzac')")
    parser.add_argument("--file", type=Path, default=None, help="Single PDF to ingest")
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
        else:
            run(args.file, corpus=args.corpus)
    else:
        corpus_dir = RAW_DIR / args.corpus
        if not corpus_dir.exists():
            logger.error(f"Corpus directory not found: {corpus_dir}")
        else:
            pdf_files = sorted(corpus_dir.glob("*.pdf"))
            if not pdf_files:
                logger.error(f"No PDF files found in {corpus_dir}")
            for pdf_path in pdf_files:
                run(pdf_path, corpus=args.corpus)
