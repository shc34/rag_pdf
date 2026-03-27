# -*- coding: utf-8 -*-
"""
Text chunking module.
Splits parsed Documents into smaller chunks for indexing and retrieval.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import config
from src.core.logger import get_logger

logger = get_logger(__name__)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Chunk size and overlap are driven by config (CHUNK_SIZE, CHUNK_OVERLAP).
    Source metadata (page, filename, source) is preserved on each chunk.

    Args:
        documents: List of parsed Documents (one per page typically).

    Returns:
        List of chunked Documents with preserved metadata.
    """
    if not documents:
        logger.warning("No documents to chunk.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk.chunk_size,
        chunk_overlap=config.chunk.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        f"Chunked {len(documents)} pages into {len(chunks)} chunks "
        f"(size={config.chunk.chunk_size}, overlap={config.chunk.chunk_overlap})"
    )

    return chunks
