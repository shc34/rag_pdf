# -*- coding: utf-8 -*-
"""
Indexer module.
Indexes document chunks into ChromaDB (dense) and BM25 (sparse).
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

BM25_INDEX_FILE = BM25_DIR / "bm25_index.pkl"


def _generate_chunk_id(doc: Document, index: int) -> str:
    """Generate a stable unique ID for a chunk based on its content and metadata."""
    key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', 0)}_{index}"
    return hashlib.md5(key.encode()).hexdigest()


def _get_chroma_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    return client.get_or_create_collection(
        name=config.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _build_embedder() -> OllamaEmbeddings:
    """Instantiate the embedding model from config."""
    emb = config.embedding

    if not emb.local:
        raise NotImplementedError(
            "Remote embedding provider not yet supported in indexer. "
            "Set EMBEDDING_LOCAL=true to use Ollama."
        )

    return OllamaEmbeddings(
        model=emb.model,
        base_url=emb.base_url,
    )


def index_documents(chunks: list[Document]) -> None:
    """
    Index document chunks into ChromaDB and BM25.

    - ChromaDB: dense vector index using configured embeddings.
    - BM25: sparse index persisted as a pickle file.

    Args:
        chunks: List of chunked Documents to index.
    """
    if not chunks:
        logger.warning("No chunks to index.")
        return

    _index_chroma(chunks)
    _index_bm25(chunks)


def _index_chroma(chunks: list[Document]) -> None:
    """Index chunks into ChromaDB with embeddings."""
    logger.info(f"Indexing {len(chunks)} chunks into ChromaDB...")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = _get_chroma_collection(client)
    embedder = _build_embedder()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [_generate_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]

    batch_size = 32
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(embeddings)
        logger.debug(f"Embedded batch {i // batch_size + 1}/{-(-len(texts) // batch_size)}")

    collection.upsert(
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"ChromaDB: {collection.count()} total chunks indexed.")


def _index_bm25(chunks: list[Document]) -> None:
    """Build and persist a BM25 index from chunks."""
    logger.info(f"Building BM25 index for {len(chunks)} chunks...")

    BM25_DIR.mkdir(parents=True, exist_ok=True)

    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunks": [
            {"text": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ],
    }

    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"BM25 index saved to {BM25_INDEX_FILE}")
