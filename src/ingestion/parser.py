# -*- coding: utf-8 -*-
"""
PDF parser module.
Extracts text from PDF files page by page using PyMuPDF.
Enriches each page with document-level structure (title, chapters).
"""

from pathlib import Path

import fitz  # pymupdf
from langchain_core.documents import Document

from src.core.logger import get_logger

logger = get_logger(__name__)


def _extract_structure(pdf: fitz.Document, filename: str) -> dict:
    """
    Extract document-level structure: title and top-level chapters.

    Strategy:
    - Title: first non-empty line of page 1 (fallback: filename)
    - Chapters: from PDF TOC if available, else empty

    Args:
        pdf: Open fitz.Document instance.
        filename: PDF filename used as title fallback.

    Returns:
        Dict with keys: title (str), chapter_count (int), chapters (str).
    """
    # --- Title ---
    title = filename  # fallback
    first_page_text = pdf[0].get_text("text").strip() if len(pdf) > 0 else ""
    if first_page_text:
        first_line = first_page_text.split("\n")[0].strip()
        if first_line:
            title = first_line

    # --- Chapters from TOC ---
    toc = pdf.get_toc()  # [[level, title, page], ...]
    chapters = [
        {"level": item[0], "title": item[1], "page": item[2]}
        for item in toc
        if item[0] == 1  # top-level only
    ]

    logger.debug(f"Document structure — title: '{title}', chapters: {len(chapters)}")

    return {
        "title": title,
        "chapter_count": len(chapters),
        # ChromaDB metadata values must be scalar: serialize as string
        "chapters": "; ".join(c["title"] for c in chapters) if chapters else "",
    }


def parse_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Parse a PDF file and return a list of LangChain Documents.

    Each document corresponds to one page, with metadata:
    - source: absolute file path
    - filename: PDF file name
    - page: page number (1-indexed)
    - total_pages: total number of pages in the document
    - doc_title: inferred document title (from first line or filename)
    - chapter_count: number of top-level chapters (from TOC)
    - chapters: semicolon-separated chapter titles (from TOC)

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects with page content and enriched metadata.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF has no pages.
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

        structure = _extract_structure(pdf, filename=pdf_path.name)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                logger.debug(f"Page {page_num}: no extractable text, skipping.")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": page_num,
                        "total_pages": total_pages,
                        **structure,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No extractable text found in: {pdf_path.name}")

    logger.info(
        f"Parsed {len(documents)}/{total_pages} pages from '{pdf_path.name}' "
        f"— title: '{structure['title']}', chapters: {structure['chapter_count']}"
    )

    return documents
