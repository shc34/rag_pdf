# -*- coding: utf-8 -*-
"""
Indexer: stores document chunks into ChromaDB (dense) and BM25 (sparse),
partitioned by corpus name.

Usage:
    from src.ingestion.indexer import index_documents
    index_documents(chunks, corpus="zola")

This creates:
    - ChromaDB collection: "{base_collection}_zola"
    - BM25 pickle: "data/bm25/bm25_zola.pkl"
"""

import hashlib
import pickle
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from src.core.config import config, CHROMA_DIR, BM25_DIR
from src.core.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_embedder() -> OllamaEmbeddings:
    """Instantiate the Ollama embedding model from config."""
    emb = config.embedding

    if not emb.local:
        raise NotImplementedError(
            "Remote embedding provider not yet supported. "
            "Set EMBEDDING_LOCAL=true to use Ollama."
        )

    return OllamaEmbeddings(
        model=emb.model,
        base_url=emb.base_url,
    )


def _generate_chunk_id(doc: Document, index: int) -> str:
    """Generate a stable unique ID for a chunk based on source, page, and index."""
    key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', 0)}_{index}"
    return hashlib.md5(key.encode()).hexdigest()


def _collection_name(corpus: str) -> str:
    """Build the ChromaDB collection name for a given corpus."""
    return f"{config.chroma_collection}_{corpus}"


def _bm25_path(corpus: str) -> Path:
    """Build the BM25 pickle path for a given corpus."""
    return BM25_DIR / f"bm25_{corpus}.pkl"


# ---------------------------------------------------------------------------
# ChromaDB (dense)
# ---------------------------------------------------------------------------

def _index_chroma(chunks: list[Document], corpus: str) -> None:
    """Embed and upsert chunks into a corpus-specific ChromaDB collection."""
    collection_name = _collection_name(corpus)
    logger.info(
        f"Indexing {len(chunks)} chunks into ChromaDB "
        f"(collection='{collection_name}')..."
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = _build_embedder()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [_generate_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]

    # Embed in batches
    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // BATCH_SIZE)  # ceil division

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(embeddings)
        logger.debug(f"Embedded batch {i // BATCH_SIZE + 1}/{total_batches}")

    # Upsert in batches (ChromaDB can choke on large single upserts)
    for i in range(0, len(texts), BATCH_SIZE):
        end = i + BATCH_SIZE
        collection.upsert(
            documents=texts[i:end],
            embeddings=all_embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )

    logger.info(
        f"ChromaDB [{collection_name}]: {collection.count()} total chunks indexed."
    )


# ---------------------------------------------------------------------------
# BM25 (sparse)
# ---------------------------------------------------------------------------

def _index_bm25(chunks: list[Document], corpus: str) -> None:
    """Build and persist a BM25 index for a given corpus.

    If a BM25 index already exists for this corpus, existing chunks are
    loaded and merged with the new ones before rebuilding.
    """
    bm25_file = _bm25_path(corpus)
    logger.info(f"Building BM25 index for corpus='{corpus}'...")

    BM25_DIR.mkdir(parents=True, exist_ok=True)

    # Merge with existing index if present
    existing_chunks: list[dict] = []
    if bm25_file.exists():
        with open(bm25_file, "rb") as f:
            existing = pickle.load(f)
            existing_chunks = existing.get("chunks", [])
        logger.info(
            f"Loaded {len(existing_chunks)} existing chunks from {bm25_file.name}"
        )

    new_entries = [
        {"text": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]

    # Deduplicate by text content
    seen_texts = {entry["text"] for entry in existing_chunks}
    merged = existing_chunks + [
        entry for entry in new_entries if entry["text"] not in seen_texts
    ]

    # Rebuild BM25 from merged corpus
    tokenized_corpus = [entry["text"].lower().split() for entry in merged]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunks": merged,
    }

    with open(bm25_file, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        f"BM25 index saved to {bm25_file} "
        f"({len(merged)} total chunks, {len(new_entries)} new)."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_documents(chunks: list[Document], corpus: str) -> None:
    """Index chunks into both ChromaDB and BM25 for a given corpus.

    Args:
        chunks:  List of chunked Documents to index.
        corpus:  Corpus name (e.g. 'zola', 'balzac'). Used to partition
                 both the ChromaDB collection and the BM25 pickle file.
    """
    if not chunks:
        logger.warning("No chunks to index.")
        return

    _index_chroma(chunks, corpus)
    _index_bm25(chunks, corpus)

    logger.info(
        f"Indexing complete for corpus='{corpus}' ({len(chunks)} chunks)."
    )
