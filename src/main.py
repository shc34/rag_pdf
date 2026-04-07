# -*- coding: utf-8 -*-
# src/main.py
"""Interactive CLI entrypoint for the RAG chatbot."""

from src.core.logger import get_logger
from src.rag.graph import build_rag_graph

logger = get_logger(__name__)

CORPUS_OPTIONS = {
    "1": ("zola", "Émile Zola"),
    "2": ("balzac", "Honoré de Balzac"),
}


def _select_corpus() -> str:
    """Prompt user to select a corpus."""
    print("Choisissez un auteur :")
    for key, (_, label) in CORPUS_OPTIONS.items():
        print(f"  {key}. {label}")
    while True:
        choice = input("Votre choix : ").strip()
        if choice in CORPUS_OPTIONS:
            corpus, label = CORPUS_OPTIONS[choice]
            print(f"\n✅ Corpus sélectionné : {label}\n")
            return corpus
        print("Choix invalide, réessayez.")


def _format_sources(sources: list[dict]) -> str:
    """Format source list for display."""
    if not sources:
        return ""

    lines = ["📎 Sources :"]
    for src in sources:
        filename = src.get("filename", "inconnu")
        page = src.get("page", "?")
        score = src.get("score")
        score_str = f" (score: {score})" if score else ""
        lines.append(f"  - {filename} — page {page}{score_str}")

    return "\n".join(lines)


def run_chat() -> None:
    """Start an interactive CLI chat session using the RAG graph."""
    print("\n📚 RAG Livre — Assistant interactif")
    print("Tapez 'exit' ou 'quit' pour quitter.\n")

    corpus = _select_corpus()
    graph = build_rag_graph()

    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Au revoir !")
            break

        try:
            result = graph.invoke({"query": question, "corpus": corpus})

            answer = result.get("answer", "Aucune réponse générée.")
            sources = result.get("sources", [])

            print(f"\n🤖 Assistant : {answer}\n")

            formatted = _format_sources(sources)
            if formatted:
                print(formatted)

            print()

        except Exception as e:
            logger.error(f"Erreur lors de l'invocation du graph : {e}", exc_info=True)
            print(f"❌ Erreur : {e}\n")


if __name__ == "__main__":
    run_chat()
