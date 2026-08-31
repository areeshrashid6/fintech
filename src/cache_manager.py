"""
cache_manager.py
----------------
Handles LangChain LLM caching.

Supports:
- In-memory cache
- SQLite cache
"""

from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache


def setup_cache(cache_type="In-memory"):
    """
    Configure the global LangChain cache.

    Parameters
    ----------
    cache_type : str
        Either "In-memory" or "SQLite".
    """

    if cache_type == "SQLite":
        cache = SQLiteCache(
            database_path=".finwise_cache.db"
        )
    else:
        cache = InMemoryCache()

    set_llm_cache(cache)


# Keep this alias so either function name works.
def set_cache(cache_type="In-memory"):
    setup_cache(cache_type)
