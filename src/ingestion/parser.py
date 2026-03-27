# -*- coding: utf-8 -*-
"""
PDF parser module.
Extracts text from PDF files page by page using PyMuPDF.
"""

from pathlib import Path

import fitz  # pymupdf
from langchain_core.documents import Document

from src.core.logger import get_logger

logger = get_logger(__name__)


def parse_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Parse a PDF file and return a list of LangChain Documents.

    Each document corresponds to one page, with metadata:
    - source: file path
    - page: page number (1-indexed)
    - total_pages: total number of pages

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF has no extractable text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Parsing PDF: {pdf_path.name}")

    documents: list[Document] = []

    with fitz.open(str(pdf_path)) as pdf:
        total_pages = len(pdf)

        if total_pages == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                logger.debug(f"Page {page_num} has no extractable text, skipping.")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": page_num,
                        "total_pages": total_pages,
                    },
                )
            )

    logger.info(f"Parsed {len(documents)} pages from {pdf_path.name}")
    return documents
