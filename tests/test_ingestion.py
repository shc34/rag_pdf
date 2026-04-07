# tests/test_ingestion.py
# -*- coding: utf-8 -*-
"""
Tests for parser, chunker, indexer and run_ingestion.
"""

import pickle
import logging

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
from langchain_core.documents import Document

from src.ingestion.parser import parse_pdf, _extract_structure
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import (
    index_documents,
    _generate_chunk_id,
    _index_chroma,
    _index_bm25,
    _collection_name,
    _bm25_path,
)
from src.ingestion.run_ingestion import run

CORPUS = "zola"


# ===================================================================== #
#                          FIXTURES                                      #
# ===================================================================== #

@pytest.fixture()
def tmp_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF with two pages of text."""
    pdf_path = tmp_path / "test_book.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Chapitre I\n\nCeci est la première page du livre.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Chapitre II\n\nVoici la deuxième page du livre.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture()
def empty_pdf(tmp_path: Path) -> Path:
    """Create a PDF with one blank page (no text)."""
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture()
def sample_documents() -> list[Document]:
    """Fake parsed documents (output of parser)."""
    return [
        Document(
            page_content="Chapitre I\n\nCeci est la première page du livre.",
            metadata={
                "source": "/fake/path.pdf",
                "filename": "path.pdf",
                "page": 1,
                "total_pages": 2,
                "title": "path",
                "chapter_count": 2,
                "chapters": ["Chapitre I", "Chapitre II"],
            },
        ),
        Document(
            page_content="Chapitre II\n\nVoici la deuxième page du livre.",
            metadata={
                "source": "/fake/path.pdf",
                "filename": "path.pdf",
                "page": 2,
                "total_pages": 2,
                "title": "path",
                "chapter_count": 2,
                "chapters": ["Chapitre I", "Chapitre II"],
            },
        ),
    ]


@pytest.fixture()
def sample_chunks() -> list[Document]:
    """Fake chunks (output of chunker)."""
    return [
        Document(
            page_content="Ceci est la première page du livre.",
            metadata={
                "source": "/fake/path.pdf",
                "filename": "path.pdf",
                "page": 1,
                "total_pages": 2,
                "title": "path",
                "chapter_count": 2,
                "chapters": ["Chapitre I", "Chapitre II"],
                "chunk_index": 0,
                "corpus": CORPUS,
            },
        ),
        Document(
            page_content="Voici la deuxième page du livre.",
            metadata={
                "source": "/fake/path.pdf",
                "filename": "path.pdf",
                "page": 2,
                "total_pages": 2,
                "title": "path",
                "chapter_count": 2,
                "chapters": ["Chapitre I", "Chapitre II"],
                "chunk_index": 1,
                "corpus": CORPUS,
            },
        ),
    ]


# ===================================================================== #
#                          parser.py                                     #
# ===================================================================== #


class TestParsePDF:
    """Tests for src.ingestion.parser.parse_pdf."""

    def test_returns_documents(self, tmp_pdf: Path):
        docs = parse_pdf(tmp_pdf)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_keys_present(self, tmp_pdf: Path):
        docs = parse_pdf(tmp_pdf)
        required = {
            "source", "filename", "page",
            "total_pages", "title", "chapter_count", "chapters",
        }
        for doc in docs:
            assert required.issubset(doc.metadata.keys())

    def test_pages_are_one_indexed(self, tmp_pdf: Path):
        docs = parse_pdf(tmp_pdf)
        pages = [d.metadata["page"] for d in docs]
        assert min(pages) >= 1

    def test_accepts_string_path(self, tmp_pdf: Path):
        docs = parse_pdf(str(tmp_pdf))
        assert len(docs) >= 1

    def test_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_pdf(tmp_path / "ghost.pdf")

    def test_raises_on_empty_pdf(self, empty_pdf: Path):
        with pytest.raises(ValueError, match="(?i)no.*extrac"):
            parse_pdf(empty_pdf)


# ===================================================================== #
#                          chunker.py                                    #
# ===================================================================== #


class TestChunkDocuments:
    """Tests for src.ingestion.chunker.chunk_documents."""

    def test_returns_documents(self, sample_documents: list[Document]):
        chunks = chunk_documents(sample_documents)
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_metadata_preserved(self, sample_documents: list[Document]):
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata

    def test_no_empty_chunks(self, sample_documents: list[Document]):
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert chunk.page_content.strip()

    def test_empty_input(self):
        assert chunk_documents([]) == []


# ===================================================================== #
#                          indexer.py – helpers                          #
# ===================================================================== #


class TestGenerateChunkId:
    """Tests for _generate_chunk_id."""

    def test_deterministic(self, sample_chunks: list[Document]):
        a = _generate_chunk_id(sample_chunks[0], 0)
        b = _generate_chunk_id(sample_chunks[0], 0)
        assert a == b

    def test_different_index_different_id(self, sample_chunks: list[Document]):
        a = _generate_chunk_id(sample_chunks[0], 0)
        b = _generate_chunk_id(sample_chunks[0], 1)
        assert a != b

    def test_returns_hex_string(self, sample_chunks: list[Document]):
        result = _generate_chunk_id(sample_chunks[0], 0)
        assert isinstance(result, str)
        int(result, 16)  # valid hex


class TestCollectionName:
    """Tests for _collection_name."""

    def test_contains_corpus(self):
        name = _collection_name("balzac")
        assert "balzac" in name

    def test_different_corpora_different_names(self):
        assert _collection_name("zola") != _collection_name("balzac")


class TestBm25Path:
    """Tests for _bm25_path."""

    def test_contains_corpus(self):
        p = _bm25_path("zola")
        assert "zola" in p.name

    def test_is_pkl(self):
        assert _bm25_path("zola").suffix == ".pkl"


# ===================================================================== #
#                          indexer.py – ChromaDB                         #
# ===================================================================== #


class TestIndexChroma:
    """Tests for _index_chroma (dense indexing)."""

    @patch("src.ingestion.indexer._build_embedder")
    @patch("src.ingestion.indexer.chromadb.PersistentClient")
    def test_upsert_called_with_correct_count(
        self,
        mock_client_cls: MagicMock,
        mock_embedder: MagicMock,
        sample_chunks: list[Document],
        tmp_path: Path,
    ):
        # -- mock ChromaDB --
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(sample_chunks)
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        # -- mock embedder --
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 768] * len(sample_chunks)
        mock_embedder.return_value = mock_emb

        with patch("src.ingestion.indexer.CHROMA_DIR", tmp_path / "chroma"):
            _index_chroma(sample_chunks, corpus=CORPUS)

        # Collect all ids across batched upserts
        total_ids: list[str] = []
        for call in mock_collection.upsert.call_args_list:
            total_ids.extend(call.kwargs.get("ids", call[1].get("ids", [])))

        assert len(total_ids) == len(sample_chunks)

    @patch("src.ingestion.indexer._build_embedder")
    @patch("src.ingestion.indexer.chromadb.PersistentClient")
    def test_collection_name_uses_corpus(
        self,
        mock_client_cls: MagicMock,
        mock_embedder: MagicMock,
        sample_chunks: list[Document],
        tmp_path: Path,
    ):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 768] * len(sample_chunks)
        mock_embedder.return_value = mock_emb

        with patch("src.ingestion.indexer.CHROMA_DIR", tmp_path / "chroma"):
            _index_chroma(sample_chunks, corpus=CORPUS)

        call_kwargs = mock_client.get_or_create_collection.call_args
        used_name = call_kwargs.kwargs.get("name") or call_kwargs[1].get("name") or call_kwargs[0][0]
        assert CORPUS in used_name


# ===================================================================== #
#                          indexer.py – BM25                             #
# ===================================================================== #


class TestIndexBm25:
    """Tests for _index_bm25 (sparse indexing)."""

    def test_creates_pickle(self, sample_chunks: list[Document], tmp_path: Path):
        with patch("src.ingestion.indexer.BM25_DIR", tmp_path):
            _index_bm25(sample_chunks, corpus=CORPUS)

        bm25_file = tmp_path / f"bm25_{CORPUS}.pkl"
        assert bm25_file.exists()

    def test_pickle_structure(self, sample_chunks: list[Document], tmp_path: Path):
        with patch("src.ingestion.indexer.BM25_DIR", tmp_path):
            _index_bm25(sample_chunks, corpus=CORPUS)

        bm25_file = tmp_path / f"bm25_{CORPUS}.pkl"
        with open(bm25_file, "rb") as f:
            payload = pickle.load(f)

        assert "bm25" in payload
        assert "chunks" in payload
        assert len(payload["chunks"]) == len(sample_chunks)

    def test_merge_deduplicates(self, sample_chunks: list[Document], tmp_path: Path):
        """Indexing the same chunks twice should not duplicate entries."""
        with patch("src.ingestion.indexer.BM25_DIR", tmp_path):
            _index_bm25(sample_chunks, corpus=CORPUS)
            _index_bm25(sample_chunks, corpus=CORPUS)

        bm25_file = tmp_path / f"bm25_{CORPUS}.pkl"
        with open(bm25_file, "rb") as f:
            payload = pickle.load(f)

        assert len(payload["chunks"]) == len(sample_chunks)

    def test_merge_adds_new_chunks(self, sample_chunks: list[Document], tmp_path: Path):
        """New chunks should be appended to existing index."""
        extra = Document(
            page_content="Un troisième paragraphe totalement inédit.",
            metadata={"source": "extra.pdf", "page": 1},
        )
        with patch("src.ingestion.indexer.BM25_DIR", tmp_path):
            _index_bm25(sample_chunks, corpus=CORPUS)
            _index_bm25([extra], corpus=CORPUS)

        bm25_file = tmp_path / f"bm25_{CORPUS}.pkl"
        with open(bm25_file, "rb") as f:
            payload = pickle.load(f)

        assert len(payload["chunks"]) == len(sample_chunks) + 1


# ===================================================================== #
#                          indexer.py – public API                       #
# ===================================================================== #


class TestIndexDocuments:
    """Tests for index_documents (public entry point)."""

    def test_skips_empty_chunks(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            index_documents([], corpus=CORPUS)
        assert "No chunks to index" in caplog.text

    @patch("src.ingestion.indexer._index_bm25")
    @patch("src.ingestion.indexer._index_chroma")
    def test_delegates_to_both_indexers(
        self,
        mock_chroma: MagicMock,
        mock_bm25: MagicMock,
        sample_chunks: list[Document],
    ):
        index_documents(sample_chunks, corpus=CORPUS)
        mock_chroma.assert_called_once_with(sample_chunks, CORPUS)
        mock_bm25.assert_called_once_with(sample_chunks, CORPUS)


# ===================================================================== #
#                          run_ingestion.py                              #
# ===================================================================== #


class TestRunIngestion:
    """Tests for the pipeline orchestrator."""

    @patch("src.ingestion.run_ingestion.index_documents")
    @patch("src.ingestion.run_ingestion.chunk_documents")
    @patch("src.ingestion.run_ingestion.parse_pdf")
    def test_pipeline_order(
        self,
        mock_parse: MagicMock,
        mock_chunk: MagicMock,
        mock_index: MagicMock,
        tmp_pdf: Path,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ):
        mock_parse.return_value = sample_documents
        mock_chunk.return_value = sample_chunks

        run(tmp_pdf, corpus=CORPUS)

        mock_parse.assert_called_once_with(tmp_pdf)
        mock_chunk.assert_called_once_with(sample_documents)
        mock_index.assert_called_once_with(sample_chunks, corpus=CORPUS)

    @patch("src.ingestion.run_ingestion.index_documents")
    @patch("src.ingestion.run_ingestion.chunk_documents")
    @patch("src.ingestion.run_ingestion.parse_pdf")
    def test_completes_without_error(
        self,
        mock_parse: MagicMock,
        mock_chunk: MagicMock,
        mock_index: MagicMock,
        tmp_pdf: Path,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ):
        mock_parse.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        run(tmp_pdf, corpus=CORPUS)  # should not raise
