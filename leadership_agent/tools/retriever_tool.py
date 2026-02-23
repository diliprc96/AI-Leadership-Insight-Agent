"""
retriever_tool.py — LangChain tool for semantic search against Qdrant.

Retrieves top-5 relevant chunks from the leadership_reports collection
based on the user query, using Titan Embed v2 for query embedding.
"""

import json
import logging
import time
from typing import Any

from langchain_core.tools import tool

from leadership_agent.embeddings.embedder import TitanEmbedder
from leadership_agent.vectorstore.qdrant_store import QdrantStore
from leadership_agent.config import QDRANT_TOP_K

logger = logging.getLogger(__name__)

# Lazy singletons — initialised on first use
_embedder: TitanEmbedder | None = None
_store: QdrantStore | None = None


def _get_embedder() -> TitanEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TitanEmbedder()
    return _embedder


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


@tool
def retriever_tool(query: str) -> str:
    """
    Search the vector database for relevant 10-K report passages.

    Use this tool for narrative or qualitative questions such as:
    - "What are the key risks?"
    - "Describe Microsoft's cloud strategy."
    - "What did leadership say about AI?"

    Args:
        query: The user's natural language question.

    Returns:
        JSON string with top retrieved chunks, metadata, and similarity scores.
    """
    logger.info("RetrieverTool invoked — query: %r", query[:120])
    t0 = time.perf_counter()

    try:
        embedder = _get_embedder()
        store = _get_store()

        # Embed the query
        query_vector = embedder.embed_query(query)
        logger.debug("Query vector generated — dim=%d", len(query_vector))

        # Search Qdrant
        results = store.search(query_vector, top_k=QDRANT_TOP_K)

        if not results:
            logger.warning("RetrieverTool: no results found for query: %r", query)
            return json.dumps(
                {"status": "empty", "message": "No relevant documents found.", "chunks": []}
            )

        # Log retrieved doc IDs and scores
        logger.info(
            "Retrieved %d chunks — IDs: %s | Scores: %s",
            len(results),
            [r["id"][:8] for r in results],
            [r["score"] for r in results],
        )

        elapsed = time.perf_counter() - t0
        logger.info("RetrieverTool completed in %.3fs", elapsed)

        output = {
            "status": "ok",
            "query": query,
            "retrieval_latency_s": round(elapsed, 3),
            "chunk_count": len(results),
            "chunks": [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                }
                for r in results
            ],
        }
        return json.dumps(output, ensure_ascii=False)

    except Exception as exc:
        logger.error("RetrieverTool error: %s", exc, exc_info=True)
        return json.dumps(
            {"status": "error", "message": str(exc), "chunks": []}
        )
