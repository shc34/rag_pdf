# -*- coding: utf-8 -*-
"""
Tests for parser, chunker, indexer and run_ingestion.
"""

import pickle
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.ingestion.parser import parse_pdf, _extract_structure
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import index_documents, _generate_chunk_id, BM25_INDEX_FILE
from src.ingestion.run_ingestion import run


# ---------------------------------------------------------------------------
# parser.py
# ---------------------------------------------------------------------------


class TestParsePDF:
    def test_parses_valid_pdf(self, tmp_pdf):
        docs = parse_pdf(tmp_pdf)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_is_complete(self, tmp_pdf):
        docs = parse_pdf(tmp_pdf)
        for doc in docs:
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "page" in doc.metadata
            assert "total_pages" in doc.metadata
            assert "title" in doc.metadata
            assert "chapter_count" in doc.metadata
            assert "chapters" in doc.metadata

    def test_page_numbers_are_one_indexed(self, tmp_pdf):
        docs = parse_pdf(tmp_pdf)
        pages = [d.metadata["page"] for d in docs]
        assert min(pages) >= 1

    def test_raises_if_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_pdf(tmp_path / "ghost.pdf")

    def test_accepts_string_path(self, tmp_pdf):
        docs = parse_pdf(str(tmp_pdf))
        assert len(docs) >= 1

    def test_empty_pdf_raises_value_error(self, tmp_path):
        """A PDF with pages but no extractable text raises ValueError."""
        import fitz
        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # page exists but has no text
        doc.save(str(pdf_path))
        doc.close()

        with pytest.raises(ValueError, match="No extractable text found in"):
            parse_pdf(pdf_path)

    def test_all_docs_share_same_structure_metadata(self, tmp_pdf):
        """All pages from the same PDF share title/chapters metadata."""
        import fitz
        pdf_path = tmp_pdf.parent / "multi.pdf"
        doc = fitz.open()
        for text in ["First page content", "Second page content"]:
            page = doc.new_page()
            page.insert_text((50, 100), text)
        doc.save(str(pdf_path))
        doc.close()

        docs = parse_pdf(pdf_path)
        assert len(docs) == 2
        titles = {d.metadata["title"] for d in docs}
        assert len(titles) == 1  # same title across all pages


class TestExtractStructure:
    def _make_pdf(self, tmp_path, pages: list[str]) -> "fitz.Document":
        import fitz
        pdf_path = tmp_path / "struct.pdf"
        doc = fitz.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((50, 100), text)
        doc.save(str(pdf_path))
        doc.close()
        # Re-open to simulate production usage
        return fitz.open(str(pdf_path))

    def test_title_from_first_line(self, tmp_path):
        import fitz
        pdf = self._make_pdf(tmp_path, ["My Book Title\nSome body text"])
        structure = _extract_structure(pdf, filename="fallback.pdf")
        pdf.close()
        assert structure["title"] == "My Book Title"

    def test_title_fallback_to_filename(self, tmp_path):
        import fitz
        pdf = self._make_pdf(tmp_path, [""])  # blank page
        structure = _extract_structure(pdf, filename="fallback.pdf")
        pdf.close()
        assert structure["title"] == "fallback.pdf"

    def test_no_toc_gives_zero_chapters(self, tmp_path):
        import fitz
        pdf = self._make_pdf(tmp_path, ["Content without TOC"])
        structure = _extract_structure(pdf, filename="book.pdf")
        pdf.close()
        assert structure["chapter_count"] == 0
        assert structure["chapters"] == ""

    def test_chapters_is_string(self, tmp_path):
        import fitz
        pdf = self._make_pdf(tmp_path, ["Content"])
        structure = _extract_structure(pdf, filename="book.pdf")
        pdf.close()
        assert isinstance(structure["chapters"], str)

    def test_structure_keys_present(self, tmp_path):
        import fitz
        pdf = self._make_pdf(tmp_path, ["Content"])
        structure = _extract_structure(pdf, filename="book.pdf")
        pdf.close()
        assert {"title", "chapter_count", "chapters"} == set(structure.keys())


# ---------------------------------------------------------------------------
# chunker.py
# ---------------------------------------------------------------------------


class TestChunkDocuments:
    def test_returns_chunks(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        assert len(chunks) >= len(sample_documents)
        assert all(isinstance(c, Document) for c in chunks)

    def test_metadata_preserved(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata

    def test_empty_input_returns_empty(self):
        result = chunk_documents([])
        assert result == []

    def test_chunk_content_is_non_empty(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert chunk.page_content.strip() != ""


# ---------------------------------------------------------------------------
# indexer.py
# ---------------------------------------------------------------------------


class TestGenerateChunkId:
    def test_is_deterministic(self, sample_chunks):
        id1 = _generate_chunk_id(sample_chunks[0], 0)
        id2 = _generate_chunk_id(sample_chunks[0], 0)
        assert id1 == id2

    def test_different_index_gives_different_id(self, sample_chunks):
        id1 = _generate_chunk_id(sample_chunks[0], 0)
        id2 = _generate_chunk_id(sample_chunks[0], 1)
        assert id1 != id2

    def test_returns_string(self, sample_chunks):
        result = _generate_chunk_id(sample_chunks[0], 0)
        assert isinstance(result, str)


class TestIndexDocuments:
    def test_skips_empty_chunks(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            index_documents([])
        assert "No chunks to index" in caplog.text

    @patch("src.ingestion.indexer._index_chroma")
    @patch("src.ingestion.indexer._index_bm25")
    def test_calls_both_indexers(self, mock_bm25, mock_chroma, sample_chunks):
        index_documents(sample_chunks)
        mock_chroma.assert_called_once_with(sample_chunks)
        mock_bm25.assert_called_once_with(sample_chunks)

    @patch("src.ingestion.indexer._build_embedder")
    @patch("src.ingestion.indexer.chromadb.PersistentClient")
    def test_chroma_upsert_called(self, mock_client_cls, mock_embedder, sample_chunks, tmp_path):
        """Verify ChromaDB upsert is called with correct number of items."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(sample_chunks)
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 768] * len(sample_chunks)
        mock_embedder.return_value = mock_emb

        with patch("src.ingestion.indexer.CHROMA_DIR", tmp_path / "chroma"):
            from src.ingestion.indexer import _index_chroma
            _index_chroma(sample_chunks)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert len(call_kwargs["ids"]) == len(sample_chunks)

    def test_bm25_index_persisted(self, sample_chunks, tmp_path):
        bm25_file = tmp_path / "bm25_index.pkl"
        with patch("src.ingestion.indexer.BM25_DIR", tmp_path), \
             patch("src.ingestion.indexer.BM25_INDEX_FILE", bm25_file):
            from src.ingestion.indexer import _index_bm25
            _index_bm25(sample_chunks)

        assert bm25_file.exists()
        with open(bm25_file, "rb") as f:
            payload = pickle.load(f)

        assert "bm25" in payload
        assert "chunks" in payload
        assert len(payload["chunks"]) == len(sample_chunks)


# ---------------------------------------------------------------------------
# run_ingestion.py
# ---------------------------------------------------------------------------


class TestRunIngestion:
    @patch("src.ingestion.run_ingestion.index_documents")
    @patch("src.ingestion.run_ingestion.chunk_documents")
    @patch("src.ingestion.run_ingestion.parse_pdf")
    def test_pipeline_called_in_order(self, mock_parse, mock_chunk, mock_index, tmp_pdf, sample_documents, sample_chunks):
        mock_parse.return_value = sample_documents
        mock_chunk.return_value = sample_chunks

        run(tmp_pdf)

        mock_parse.assert_called_once_with(tmp_pdf)
        mock_chunk.assert_called_once_with(sample_documents)
        mock_index.assert_called_once_with(sample_chunks)

    @patch("src.ingestion.run_ingestion.index_documents")
    @patch("src.ingestion.run_ingestion.chunk_documents")
    @patch("src.ingestion.run_ingestion.parse_pdf")
    def test_run_completes_without_error(self, mock_parse, mock_chunk, mock_index, tmp_pdf, sample_documents, sample_chunks):
        mock_parse.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        run(tmp_pdf)  # should not raise
