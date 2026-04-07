# -*- coding: utf-8 -*-
"""
Hybrid retriever: ChromaDB (semantic) + BM25 (sparse), merged via RRF,
optionally reranked with a cross-encoder. Partitioned by corpus.
"""

import pickle

import chromadb
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.core.config import CHROMA_DIR
from src.core.logger import get_logger
from src.ingestion.indexer import _build_embedder, _collection_name, _bm25_path

logger = get_logger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    corpus: str,
    top_k: int = 5,
    use_reranker: bool = True,
) -> list[Document]:
    """
    Hybrid retrieval over a specific corpus.

    Args:
        query: User query string.
        corpus: Name of the corpus to search.
        top_k: Number of final documents to return.
        use_reranker: Whether to apply cross-encoder reranking.

    Returns:
        List of Documents ranked by relevance.
    """
    candidate_k = top_k * 3 if use_reranker else top_k

    semantic_results = _retrieve_chroma(query, corpus, top_k=candidate_k)
    bm25_results = _retrieve_bm25(query, corpus, top_k=candidate_k)

    fused = _reciprocal_rank_fusion(
        result_lists=[semantic_results, bm25_results],
        top_k=candidate_k,
    )

    if use_reranker and fused:
        fused = _rerank(query, fused, top_k=top_k)

    logger.info(
        f"Hybrid retrieval (corpus='{corpus}'): "
        f"{len(semantic_results)} semantic + {len(bm25_results)} BM25 "
        f"→ {len(fused)} final (reranker={'on' if use_reranker else 'off'})"
    )
    return fused


# ---------------------------------------------------------------------------
# Semantic (ChromaDB)
# ---------------------------------------------------------------------------

def _retrieve_chroma(query: str, corpus: str, top_k: int) -> list[Document]:
    """Dense retrieval via ChromaDB for a given corpus."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col_name = _collection_name(corpus)
    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
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
        docs.append(Document(
            page_content=text,
            metadata={**metadata, "score_semantic": 1 - distance},
        ))
    return docs


# ---------------------------------------------------------------------------
# Sparse (BM25)
# ---------------------------------------------------------------------------

def _retrieve_bm25(query: str, corpus: str, top_k: int) -> list[Document]:
    """Sparse retrieval via BM25 for a given corpus."""
    index_file = _bm25_path(corpus)
    if not index_file.exists():
        logger.warning(f"BM25 index not found for corpus='{corpus}', skipping.")
        return []

    with open(index_file, "rb") as f:
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
        docs.append(Document(
            page_content=chunk["text"],
            metadata={**chunk["metadata"], "score_bm25": float(scores[idx])},
        ))
    return docs


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    top_k: int,
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using RRF (k=60 per original paper)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    return [doc_map[key] for key in ranked_keys]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Rerank documents using a cross-encoder model."""
    reranker = _get_reranker()

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["score_reranker"] = float(score)
        results.append(doc)

    logger.debug(
        f"Reranker: best={scored_docs[0][0]:.3f}, "
        f"worst={scored_docs[-1][0]:.3f}"
    )
    return results
