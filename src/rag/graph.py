# -*- coding: utf-8 -*-
"""RAG graph orchestration with LangGraph."""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from src.rag.llm import get_llm
from src.rag.retriever import retrieve
from src.rag.prompts import RAG_PROMPT


class RAGState(TypedDict):
    query: str
    documents: list[Document]
    answer: str
    sources: list[dict]


def retrieve_node(state: RAGState) -> dict:
    """Retrieve relevant documents."""
    docs = retrieve(state["query"])
    return {"documents": docs}


def generate_node(state: RAGState) -> dict:
    """Generate answer from retrieved context."""
    llm = get_llm()

    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    prompt = RAG_PROMPT.format(context=context, query=state["query"])
    response = llm.invoke(prompt)

    sources = [
        {
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(doc.metadata.get("score_reranker", 0.0), 3),
        }
        for doc in state["documents"]
    ]

    return {"answer": response.content, "sources": sources}


def build_rag_graph():
    """Build and compile the RAG graph."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_graph = build_rag_graph()
