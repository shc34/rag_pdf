# -*- coding: utf-8 -*-
"""
Tests for RAG components: retriever.py, graph.py
"""

import importlib
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def clean_graph_module():
    """Ensure src.rag.graph is freshly imported for each test."""
    sys.modules.pop("src.rag.graph", None)
    yield
    sys.modules.pop("src.rag.graph", None)


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Tesla stock rose sharply after earnings.",
            metadata={"source": "cnbc", "url": "http://cnbc.com", "date": "2024-04-10", "title": "Tesla Q1"},
        ),
        Document(
            page_content="Climate summit reached a historic agreement.",
            metadata={"source": "bbc", "url": "http://bbc.com", "date": "2024-11-20", "title": "Climate Deal"},
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrieve — retrieve() public function
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieve:

    @patch("src.rag.retriever._retrieve_bm25")
    @patch("src.rag.retriever._retrieve_chroma")
    def test_returns_list_of_documents(self, mock_chroma, mock_bm25, sample_documents):
        mock_chroma.return_value = sample_documents
        mock_bm25.return_value = sample_documents

        with patch("src.rag.retriever._rerank", side_effect=lambda q, docs, top_k: docs[:top_k]):
            from src.rag.retriever import retrieve
            results = retrieve("tesla stock", top_k=2, use_reranker=True)

        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    @patch("src.rag.retriever._retrieve_bm25")
    @patch("src.rag.retriever._retrieve_chroma")
    def test_without_reranker(self, mock_chroma, mock_bm25, sample_documents):
        mock_chroma.return_value = sample_documents
        mock_bm25.return_value = []

        from src.rag.retriever import retrieve
        results = retrieve("tesla", top_k=2, use_reranker=False)

        assert isinstance(results, list)

    @patch("src.rag.retriever._retrieve_bm25")
    @patch("src.rag.retriever._retrieve_chroma")
    def test_empty_results_do_not_crash(self, mock_chroma, mock_bm25):
        mock_chroma.return_value = []
        mock_bm25.return_value = []

        from src.rag.retriever import retrieve
        results = retrieve("nothing", top_k=5, use_reranker=False)

        assert results == []

    @patch("src.rag.retriever._retrieve_bm25")
    @patch("src.rag.retriever._retrieve_chroma")
    def test_reranker_not_called_when_disabled(self, mock_chroma, mock_bm25, sample_documents):
        mock_chroma.return_value = sample_documents
        mock_bm25.return_value = []

        with patch("src.rag.retriever._rerank") as mock_rerank:
            from src.rag.retriever import retrieve
            retrieve("tesla", top_k=2, use_reranker=False)
            mock_rerank.assert_not_called()

    @patch("src.rag.retriever._retrieve_bm25")
    @patch("src.rag.retriever._retrieve_chroma")
    def test_reranker_called_when_enabled(self, mock_chroma, mock_bm25, sample_documents):
        mock_chroma.return_value = sample_documents
        mock_bm25.return_value = sample_documents

        with patch("src.rag.retriever._rerank", return_value=sample_documents[:1]) as mock_rerank:
            from src.rag.retriever import retrieve
            retrieve("tesla", top_k=1, use_reranker=True)
            mock_rerank.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrieveChroma — _retrieve_chroma()
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieveChroma:

    def _make_chroma_result(self, texts, metadatas, distances):
        return {
            "documents": [texts],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    @patch("src.rag.retriever._build_embedder")
    @patch("src.rag.retriever._get_chroma_collection")
    @patch("src.rag.retriever.chromadb.PersistentClient")
    def test_returns_documents(self, mock_client, mock_collection_fn, mock_embedder):
        mock_embedder.return_value.embed_query.return_value = [0.1] * 384

        mock_collection = MagicMock()
        mock_collection.query.return_value = self._make_chroma_result(
            texts=["Tesla rose sharply."],
            metadatas=[{"source": "cnbc", "date": "2024-04-10"}],
            distances=[0.15],
        )
        mock_collection_fn.return_value = mock_collection

        from src.rag.retriever import _retrieve_chroma
        results = _retrieve_chroma("tesla stock", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert "score_semantic" in results[0].metadata

    @patch("src.rag.retriever._build_embedder")
    @patch("src.rag.retriever._get_chroma_collection")
    @patch("src.rag.retriever.chromadb.PersistentClient")
    def test_score_semantic_is_one_minus_distance(self, mock_client, mock_collection_fn, mock_embedder):
        mock_embedder.return_value.embed_query.return_value = [0.0] * 384

        mock_collection = MagicMock()
        mock_collection.query.return_value = self._make_chroma_result(
            texts=["Some article."],
            metadatas=[{"source": "bbc"}],
            distances=[0.2],
        )
        mock_collection_fn.return_value = mock_collection

        from src.rag.retriever import _retrieve_chroma
        results = _retrieve_chroma("query", top_k=1)

        assert pytest.approx(results[0].metadata["score_semantic"], abs=1e-6) == 0.8


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrieveBM25 — _retrieve_bm25()
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieveBM25:

    def test_returns_documents_from_index(self, tmp_path):
        from rank_bm25 import BM25Okapi
        import src.rag.retriever as retriever_module

        # Corpus suffisamment large pour que BM25Okapi produise des scores > 0
        chunks = [
            {"text": "tesla stock rose sharply", "metadata": {"source": "cnbc", "date": "2024-04-10"}},
            {"text": "climate summit agreement reached", "metadata": {"source": "bbc", "date": "2024-11-20"}},
            {"text": "apple earnings beat expectations", "metadata": {"source": "reuters", "date": "2024-05-01"}},
            {"text": "federal reserve holds interest rates", "metadata": {"source": "ft", "date": "2024-06-15"}},
            {"text": "tesla launches new model in europe", "metadata": {"source": "cnbc", "date": "2024-07-22"}},
        ]
        corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)

        index_file = tmp_path / "bm25_index.pkl"
        with open(index_file, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)

        original = retriever_module.BM25_INDEX_FILE
        retriever_module.BM25_INDEX_FILE = index_file
        try:
            results = retriever_module._retrieve_bm25("tesla stock", top_k=5)
        finally:
            retriever_module.BM25_INDEX_FILE = original

        assert len(results) >= 1
        assert isinstance(results[0], Document)
        assert "score_bm25" in results[0].metadata
        assert results[0].metadata["score_bm25"] > 0
        assert "tesla" in results[0].page_content

    def test_returns_empty_when_index_missing(self, tmp_path):
        import src.rag.retriever as retriever_module

        original = retriever_module.BM25_INDEX_FILE
        retriever_module.BM25_INDEX_FILE = tmp_path / "nonexistent.pkl"
        try:
            results = retriever_module._retrieve_bm25("anything", top_k=5)
        finally:
            retriever_module.BM25_INDEX_FILE = original

        assert results == []

    def test_zero_score_documents_excluded(self, tmp_path):
        from rank_bm25 import BM25Okapi
        import src.rag.retriever as retriever_module

        chunks = [
            {"text": "tesla stock rose sharply", "metadata": {"source": "cnbc", "date": "2024-04-10"}},
            {"text": "unrelated content about cooking", "metadata": {"source": "bbc", "date": "2024-11-20"}},
        ]
        corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)

        index_file = tmp_path / "bm25_index.pkl"
        with open(index_file, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)

        original = retriever_module.BM25_INDEX_FILE
        retriever_module.BM25_INDEX_FILE = index_file
        try:
            results = retriever_module._retrieve_bm25("tesla stock", top_k=5)
        finally:
            retriever_module.BM25_INDEX_FILE = original

        assert all(d.metadata["score_bm25"] > 0 for d in results)


# ─────────────────────────────────────────────────────────────────────────────
# TestRRF — _reciprocal_rank_fusion()
# ─────────────────────────────────────────────────────────────────────────────


class TestRRF:

    def _make_doc(self, content: str) -> Document:
        return Document(page_content=content, metadata={})

    def test_merges_two_lists(self):
        from src.rag.retriever import _reciprocal_rank_fusion

        list_a = [self._make_doc("doc A"), self._make_doc("doc B")]
        list_b = [self._make_doc("doc B"), self._make_doc("doc C")]

        results = _reciprocal_rank_fusion([list_a, list_b], top_k=3)
        contents = [d.page_content for d in results]

        # doc B appears in both lists → should rank highest
        assert contents[0] == "doc B"
        assert len(results) <= 3

    def test_respects_top_k(self):
        from src.rag.retriever import _reciprocal_rank_fusion

        docs = [self._make_doc(f"doc {i}") for i in range(10)]
        results = _reciprocal_rank_fusion([docs], top_k=3)

        assert len(results) == 3

    def test_empty_lists_return_empty(self):
        from src.rag.retriever import _reciprocal_rank_fusion

        results = _reciprocal_rank_fusion([[], []], top_k=5)
        assert results == []

    def test_single_list_preserves_order(self):
        from src.rag.retriever import _reciprocal_rank_fusion

        docs = [self._make_doc(f"doc {i}") for i in range(4)]
        results = _reciprocal_rank_fusion([docs], top_k=4)

        # First doc in input has best rank → best RRF score
        assert results[0].page_content == "doc 0"


# ─────────────────────────────────────────────────────────────────────────────
# TestRerank — _rerank()
# ─────────────────────────────────────────────────────────────────────────────


class TestRerank:

    @patch("src.rag.retriever._get_reranker")
    def test_returns_top_k_documents(self, mock_get_reranker):
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.3, 0.6]
        mock_get_reranker.return_value = mock_reranker

        docs = [
            Document(page_content=f"doc {i}", metadata={}) for i in range(3)
        ]

        from src.rag.retriever import _rerank
        results = _rerank("query", docs, top_k=2)

        assert len(results) == 2

    @patch("src.rag.retriever._get_reranker")
    def test_scores_attached_to_metadata(self, mock_get_reranker):
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.8, 0.2]
        mock_get_reranker.return_value = mock_reranker

        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(2)]

        from src.rag.retriever import _rerank
        results = _rerank("query", docs, top_k=2)

        assert all("score_reranker" in d.metadata for d in results)

    @patch("src.rag.retriever._get_reranker")
    def test_sorted_by_score_descending(self, mock_get_reranker):
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.1, 0.95, 0.5]
        mock_get_reranker.return_value = mock_reranker

        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(3)]

        from src.rag.retriever import _rerank
        results = _rerank("query", docs, top_k=3)

        scores = [d.metadata["score_reranker"] for d in results]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# TestRAGGraph — graph.py nodes and compilation
# ─────────────────────────────────────────────────────────────────────────────


class TestRAGGraph:

    def test_graph_compiles(self, clean_graph_module):
        with patch("src.rag.retriever.retrieve", return_value=[]),              patch("src.rag.llm.get_llm", return_value=MagicMock()):
            from src.rag.graph import build_rag_graph
            graph = build_rag_graph()
            assert graph is not None

    def test_retrieve_node_calls_retrieve(self, clean_graph_module):
        mock_docs = [Document(page_content="Tesla rose.", metadata={})]

        with patch("src.rag.graph.retrieve", return_value=mock_docs) as mock_retrieve:
            from src.rag.graph import retrieve_node
            result = retrieve_node({"query": "tesla", "documents": [], "answer": "", "sources": []})

        mock_retrieve.assert_called_once_with("tesla")
        assert result["documents"] == mock_docs

    def test_generate_node_calls_llm(self, clean_graph_module):
        mock_response = MagicMock()
        mock_response.content = "Tesla did well."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.rag.graph.get_llm", return_value=mock_llm):
            from src.rag.graph import generate_node
            state = {
                "query": "How did Tesla do?",
                "documents": [
                    Document(
                        page_content="Tesla stock rose sharply.",
                        metadata={"title": "Tesla Q1", "url": "http://cnbc.com", "date": "2024-04-10"},
                    )
                ],
                "answer": "",
                "sources": [],
            }
            result = generate_node(state)

        assert result["answer"] == "Tesla did well."
        mock_llm.invoke.assert_called_once()

    def test_generate_node_sources_have_expected_keys(self, clean_graph_module):
        mock_response = MagicMock()
        mock_response.content = "Answer."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.rag.graph.get_llm", return_value=mock_llm):
            from src.rag.graph import generate_node
            state = {
                "query": "query",
                "documents": [
                    Document(
                        page_content="content",
                        metadata={
                            "filename": "book.pdf",
                            "page": 3,
                            "score_reranker": 0.95,
                        },
                    )
                ],
                "answer": "",
                "sources": [],
            }
            result = generate_node(state)

        assert len(result["sources"]) == 1
        source = result["sources"][0]
        assert "filename" in source
        assert "page" in source
        assert "score" in source


    def test_generate_node_missing_metadata_defaults_to_empty_string(self, clean_graph_module):
        mock_response = MagicMock()
        mock_response.content = "Answer."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.rag.graph.get_llm", return_value=mock_llm):
            from src.rag.graph import generate_node
            state = {
                "query": "query",
                "documents": [Document(page_content="content", metadata={})],
                "answer": "",
                "sources": [],
            }
            result = generate_node(state)

        source = result["sources"][0]
        assert source["filename"] == "unknown"
        assert source["page"] == "?"
        assert source["score"] == 0.0

    def test_rag_graph_singleton_exists(self, clean_graph_module):
        with patch("src.rag.graph.retrieve", return_value=[]),              patch("src.rag.graph.get_llm", return_value=MagicMock()):
            from src.rag.graph import rag_graph
            assert rag_graph is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestGetLLM
# ─────────────────────────────────────────────────────────────────────────────


class TestGetLLM:
    def _mock_config(self, local: bool):
        cfg = MagicMock()
        cfg.llm.local = local
        cfg.llm.model = "test-model"
        cfg.llm.base_url = "http://localhost:11434"
        cfg.llm.api_key = "test-key"
        cfg.llm.temperature = 0.0
        return cfg

    def test_returns_ollama_when_local_true(self, monkeypatch):
        from langchain_ollama import ChatOllama
        import src.rag.llm as llm_module

        monkeypatch.setattr(llm_module, "config", self._mock_config(local=True))
        result = llm_module.get_llm()
        assert isinstance(result, ChatOllama)

    def test_returns_openai_when_provider_set(self, monkeypatch):
        from langchain_openai import ChatOpenAI
        import src.rag.llm as llm_module

        monkeypatch.setattr(llm_module, "config", self._mock_config(local=False))
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        result = llm_module.get_llm()
        assert isinstance(result, ChatOpenAI)

    def test_returns_anthropic_when_provider_set(self, monkeypatch):
        from langchain_anthropic import ChatAnthropic
        import src.rag.llm as llm_module

        monkeypatch.setattr(llm_module, "config", self._mock_config(local=False))
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        result = llm_module.get_llm()
        assert isinstance(result, ChatAnthropic)

    def test_raises_on_unsupported_provider(self, monkeypatch):
        import src.rag.llm as llm_module

        monkeypatch.setattr(llm_module, "config", self._mock_config(local=False))
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm_module.get_llm()


# ─────────────────────────────────────────────────────────────────────────────
# TestGetReranker
# ─────────────────────────────────────────────────────────────────────────────


class TestGetReranker:
    def test_loads_reranker_on_first_call(self, monkeypatch):
        from sentence_transformers import CrossEncoder
        import src.rag.retriever as retriever_module

        mock_encoder = MagicMock(spec=CrossEncoder)
        monkeypatch.setattr(retriever_module, "_reranker", None)
        monkeypatch.setattr(retriever_module, "CrossEncoder", lambda model: mock_encoder)

        result = retriever_module._get_reranker()
        assert result is mock_encoder

    def test_returns_cached_reranker_on_second_call(self, monkeypatch):
        from sentence_transformers import CrossEncoder
        import src.rag.retriever as retriever_module

        existing = MagicMock(spec=CrossEncoder)
        monkeypatch.setattr(retriever_module, "_reranker", existing)

        result = retriever_module._get_reranker()
        assert result is existing  # singleton réutilisé, pas rechargé
