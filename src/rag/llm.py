# -*- coding: utf-8 -*-
"""LLM client factory."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.core.config import config


def get_llm() -> BaseChatModel:
    """Return a configured LLM based on config."""
    llm_cfg = config.llm

    if llm_cfg.local:
        return ChatOllama(
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            temperature=llm_cfg.temperature,
        )

    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        return ChatAnthropic(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            temperature=llm_cfg.temperature,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            temperature=llm_cfg.temperature,
        )

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'openai' or 'anthropic'.")
