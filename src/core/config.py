# -*- coding: utf-8 -*-
"""Centralized configuration for the application."""

from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_DIR = DATA_DIR / "bm25_index"


def _env_bool(key: str, default: str = "true") -> bool:
    """Parse a boolean from environment variable."""
    return os.getenv(key, default).lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings.

    If local=True: uses Ollama at base_url.
    If local=False: uses remote API (OpenAI-compatible) with api_key.
    """

    local: bool = _env_bool("LLM_LOCAL", "true")
    model: str = os.getenv("LLM_MODEL", "llama3")
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    def __post_init__(self) -> None:
        if not self.local and not self.api_key:
            raise ValueError("LLM_API_KEY is required when LLM_LOCAL=false")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model settings.

    Same local/remote logic as LLM.
    """

    local: bool = _env_bool("EMBEDDING_LOCAL", "true")
    model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    base_url: str = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
    api_key: str = os.getenv("EMBEDDING_API_KEY", "")

    def __post_init__(self) -> None:
        if not self.local and not self.api_key:
            raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_LOCAL=false")


@dataclass(frozen=True)
class ChunkConfig:
    """Text chunking parameters."""

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1250"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))


@dataclass(frozen=True)
class AppConfig:
    """Top-level application config."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documents")


# Singleton instance
config = AppConfig()
