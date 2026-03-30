# -*- coding: utf-8 -*-
"""
Hybrid retriever: ChromaDB (semantic) + BM25 (sparse), merged via RRF,
then reranked with a cross-encoder.
"""

import pickle

import chromadb
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.core.config import config, CHROMA_DIR, BM25_DIR
from src.core.logger import get_logger
from src.ingestion.indexer import _build_embedder, _get_chroma_collection

logger = get_logger(__name__)

BM25_INDEX_FILE = BM25_DIR / "bm25_index.pkl"

# Lazy-loaded singleton to avoid reloading on every call
_reranker: CrossEncoder | None = None

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def retrieve(query: str, top_k: int = 5, use_reranker: bool = True) -> list[Document]:
    """
    Hybrid retrieval: semantic + BM25, fused with RRF, optionally reranked.

    Args:
        query: User query string.
        top_k: Number of final documents to return.
        use_reranker: Whether to apply cross-encoder reranking.

    Returns:
        List of Documents ranked by relevance.
    """
    # Fetch more candidates when reranking
    candidate_k = top_k * 3 if use_reranker else top_k

    semantic_results = _retrieve_chroma(query, top_k=candidate_k)
    bm25_results = _retrieve_bm25(query, top_k=candidate_k)

    fused = _reciprocal_rank_fusion(
        result_lists=[semantic_results, bm25_results],
        top_k=candidate_k,
    )

    if use_reranker and fused:
        fused = _rerank(query, fused, top_k=top_k)

    logger.info(
        f"Hybrid retrieval: {len(semantic_results)} semantic + "
        f"{len(bm25_results)} BM25 → {len(fused)} final results "
        f"(reranker={'on' if use_reranker else 'off'})"
    )
    return fused


def _rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Rerank documents using a cross-encoder model."""
    reranker = _get_reranker()

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["score_reranker"] = float(score)
        results.append(doc)

    logger.debug(
        f"Reranker scores: best={scored_docs[0][0]:.3f}, "
        f"worst={scored_docs[-1][0]:.3f}"
    )
    return results


def _retrieve_chroma(query: str, top_k: int) -> list[Document]:
    """Dense retrieval via ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = _get_chroma_collection(client)
    embedder = _build_embedder()

    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        doc = Document(
            page_content=text,
            metadata={**metadata, "score_semantic": 1 - distance},
        )
        docs.append(doc)

    return docs


def _retrieve_bm25(query: str, top_k: int) -> list[Document]:
    """Sparse retrieval via BM25."""
    if not BM25_INDEX_FILE.exists():
        logger.warning("BM25 index not found, skipping sparse retrieval.")
        return []

    with open(BM25_INDEX_FILE, "rb") as f:
        payload = pickle.load(f)

    bm25 = payload["bm25"]
    chunks = payload["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    docs = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue
        chunk = chunks[idx]
        doc = Document(
            page_content=chunk["text"],
            metadata={**chunk["metadata"], "score_bm25": float(scores[idx])},
        )
        docs.append(doc)

    return docs


def _reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    top_k: int,
    k: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across all lists.
    k=60 is the standard constant from the original paper.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    return [doc_map[key] for key in ranked_keys]
