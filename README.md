# 📚 RAG Littérature — Zola & Balzac

Système de Question-Answering basé sur le RAG (Retrieval-Augmented Generation) permettant d'interroger les œuvres complètes d'Émile Zola et d'Honoré de Balzac.

## 🎯 Objectif

Poser des questions en langage naturel sur les œuvres de Zola ou Balzac et obtenir des réponses sourcées, générées par un LLM à partir des textes originaux.

## 🏗️ Architecture

Textes (Gutenberg) → Découpage en chunks → Embeddings → ChromaDB
                                                            ↓
                                              LangGraph RAG → Réponse + Sources


## 🚀 Quickstart

### Prérequis
- check le pyproject.toml
- [uv](https://docs.astral.sh/uv/)
- Clé API OpenAI (ou autre LLM)

### Installation

```bash
git clone git@github.com:shc34/rag_livres.git
cd rag_livre
uv sync --extra dev --group dev

Configuration

cp .env.example .env
# Ajouter votre clé API : OPENAI_API_KEY=sk-...

Utilisation

# Ingestion des textes de Zola
uv run python -m src.ingestion --author zola

# Ingestion des textes de Balzac
uv run python -m src.ingestion --author balzac

# Lancer le RAG
uv run python -m src.main

# Lancer l'api sur le port 8001
uvicorn src.api.endpoints:app --reload --port 8001

Entrée — POST /api/rag/chat
{
  "message": "Quel est le thème principal du livre ?",
  "corpus": "nom_du_corpus"
}
Sortie
{
  "answer": "Le thème principal est ...",
  "sources": [
    {
      "filename": "chapitre1.pdf",
      "page": "12",
      "score": 0.87
    }
  ]
}

📁 Structure

projet_rag_livre/
├── src/
│   ├── ingestion/      # Chargement et découpage des textes
│   ├── rag/            # Graph LangGraph, LLM, retriever
│   ├── config/         # Settings
│   └── main.py         # Point d'entrée
├── data/               # Textes sources
├── tests/
├── pyproject.toml
└── README.md

🔑 Fonctionnalités

    Multi-auteurs : collections séparées par auteur (Zola, Balzac)
    RAG avec sources : chaque réponse cite les passages utilisés
    Vectorisation : ChromaDB avec embeddings OpenAI
    Orchestration : LangGraph pour le flux de raisonnement

🛠️ Tech Stack

Python · LangGraph · LangChain · ChromaDB · OpenAI

Amélioration: ragas, langfuse (overkill?)

📄 License

MIT
