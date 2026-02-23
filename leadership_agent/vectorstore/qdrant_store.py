"""
qdrant_store.py — Local persistent Qdrant vector database wrapper.

Collection: leadership_reports
Distance:   Cosine
Dimension:  1024
Storage:    ./qdrant_storage (configurable via config.py)
"""

import atexit
import logging
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from leadership_agent.config import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    QDRANT_PATH,
    QDRANT_TOP_K,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Local Qdrant vector store for leadership report embeddings.

    Usage:
        store = QdrantStore()
        store.create_collection()           # idempotent
        store.upsert(chunks, embeddings)    # list of chunks + vectors
        results = store.search(query_vec)   # returns scored hits
    """

    def __init__(self, path: str = QDRANT_PATH, collection: str = COLLECTION_NAME) -> None:
        self.path = path
        self.collection = collection
        self._client = QdrantClient(path=path)
        # Register cleanup BEFORE Python tears down sys.modules — prevents the
        # Windows msvcrt/portalocker ModuleNotFoundError on interpreter exit.
        atexit.register(self._cleanup)
        logger.info(
            "QdrantStore initialised — path=%s, collection=%s",
            path, collection,
        )

    def _cleanup(self) -> None:
        """Gracefully close the Qdrant client on interpreter exit."""
        try:
            self._client.close()
        except Exception:
            pass  # Silently ignore any errors during shutdown

    # ── Collection Management ──────────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection if it does not exist.

        Args:
            recreate: If True, deletes and recreates the collection.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if self.collection in existing:
            if recreate:
                logger.warning("Recreating collection '%s'...", self.collection)
                self._client.delete_collection(self.collection)
            else:
                logger.info("Collection '%s' already exists — skipping creation.", self.collection)
                return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Collection '%s' created — dim=%d, distance=Cosine",
            self.collection, EMBEDDING_DIMENSION,
        )

    def collection_exists(self) -> bool:
        existing = [c.name for c in self._client.get_collections().collections]
        return self.collection in existing

    def count(self) -> int:
        """Return number of points in the collection."""
        if not self.collection_exists():
            return 0
        result = self._client.count(collection_name=self.collection)
        return result.count

    # ── Upsert ─────────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        """
        Upsert chunks with their embeddings into Qdrant.

        Args:
            chunks:     List of {"text": str, "metadata": dict} dicts.
            embeddings: Corresponding list of float vectors.

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not self.collection_exists():
            self.create_collection()

        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {**chunk["metadata"], "text": chunk["text"]}
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            points.append(point)

        t0 = time.perf_counter()
        self._client.upsert(collection_name=self.collection, points=points)
        elapsed = time.perf_counter() - t0

        total = self.count()
        logger.info(
            "Upserted %d points in %.2fs — collection total: %d",
            len(points), elapsed, total,
        )
        return len(points)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = QDRANT_TOP_K,
        filter_dict: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for nearest neighbours.

        Uses query_points() (qdrant-client >= 1.12).
        Falls back to legacy search() for older versions.

        Args:
            query_vector: Embedded query vector.
            top_k:        Number of results to return.
            filter_dict:  Optional equality filters on payload fields,
                          e.g. {"company": "Microsoft", "year": "2023"}.

        Returns:
            List of dicts with keys: id, score, text, metadata.
        """
        qdrant_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            qdrant_filter = Filter(must=conditions)

        t0 = time.perf_counter()

        # qdrant-client >= 1.12: use query_points()
        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            # Fallback for qdrant-client < 1.12
            logger.debug("query_points() not available — falling back to search()")
            hits = self._client.search(  # type: ignore[attr-defined]
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        elapsed = time.perf_counter() - t0

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "id": str(hit.id),
                    "score": round(float(hit.score), 4),
                    "text": payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in payload.items() if k != "text"
                    },
                }
            )

        logger.info(
            "Search complete — top_k=%d, hits=%d, elapsed=%.3fs",
            top_k, len(results), elapsed,
        )
        for r in results:
            logger.debug("  Hit id=%s score=%.4f section=%s",
                         r["id"], r["score"], r["metadata"].get("section", "?"))

        return results
