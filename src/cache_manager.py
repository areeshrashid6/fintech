"""
cache_manager.py
Registers a global LangChain LLM cache (InMemoryCache or SQLiteCache).
set_llm_cache(...) is called once; LangChain then checks the cache before
each model call, so repeating the exact same prompt is faster and makes
no new API request.
"""

import os
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", ".finwise_cache.db")


def setup_cache(cache_type: str) -> str:
    """
    cache_type: one of "No cache", "In-memory cache", "SQLite cache (persistent)"
    Returns a short status string for display in the UI.
    """
    if cache_type == "In-memory cache":
        set_llm_cache(InMemoryCache())
        return "In-memory cache active (cleared when the app restarts)."
    if cache_type == "SQLite cache (persistent)":
        set_llm_cache(SQLiteCache(database_path=CACHE_DB_PATH))
        return f"SQLite cache active — persisted to {os.path.basename(CACHE_DB_PATH)}."

    set_llm_cache(None)
    return "Caching disabled — every submission calls the model fresh."
