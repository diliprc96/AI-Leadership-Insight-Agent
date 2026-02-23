"""
embedder.py — Amazon Titan Text Embed v2 wrapper.

Embeds texts in batches of 32 using the Bedrock Runtime API.
Output dimension: 1024.
"""

import json
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from leadership_agent.config import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_ID,
)

logger = logging.getLogger(__name__)


class TitanEmbedder:
    """
    Wraps Amazon Titan Text Embed v2 for batch text embedding.

    Example:
        embedder = TitanEmbedder()
        vectors = embedder.embed_texts(["hello world", "foo bar"])
        # vectors is a list of 1024-dimensional float lists
    """

    def __init__(self) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
        )
        self.model_id = EMBEDDING_MODEL_ID
        self.dimension = EMBEDDING_DIMENSION
        self.batch_size = EMBEDDING_BATCH_SIZE
        logger.info(
            "TitanEmbedder initialised — model=%s, dim=%d, batch_size=%d",
            self.model_id, self.dimension, self.batch_size,
        )

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text string and return its vector."""
        body = json.dumps({"inputText": text})
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        vector = result["embedding"]
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in batches.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.

        Raises:
            ValueError: If any returned vector has wrong dimension.
            ClientError: On Bedrock API failure.
        """
        if not texts:
            logger.warning("embed_texts called with empty list — returning []")
            return []

        all_vectors: list[list[float]] = []
        total = len(texts)
        n_batches = (total + self.batch_size - 1) // self.batch_size

        logger.info("Embedding %d texts in %d batch(es)...", total, n_batches)
        t_start = time.perf_counter()

        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, total)
            batch = texts[batch_start:batch_end]

            batch_vectors: list[list[float]] = []
            for text in batch:
                try:
                    vec = self._embed_single(text)
                    if len(vec) != self.dimension:
                        raise ValueError(
                            f"Expected dim {self.dimension}, got {len(vec)}"
                        )
                    batch_vectors.append(vec)
                except ClientError as exc:
                    logger.error(
                        "Bedrock ClientError embedding text (batch %d): %s",
                        batch_idx, exc, exc_info=True,
                    )
                    raise
                except Exception as exc:
                    logger.error(
                        "Unexpected error embedding text (batch %d): %s",
                        batch_idx, exc, exc_info=True,
                    )
                    raise

            all_vectors.extend(batch_vectors)
            logger.debug(
                "Batch %d/%d done — %d vectors so far",
                batch_idx + 1, n_batches, len(all_vectors),
            )

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Embedding complete — %d vectors, dim=%d, elapsed=%.2fs",
            len(all_vectors), self.dimension, elapsed,
        )
        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Convenience method for single-query embedding (used by retriever)."""
        logger.debug("Embedding query: %r", query[:100])
        return self.embed_texts([query])[0]


# ─── Module self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from leadership_agent.logging_config import setup_logging

    setup_logging("DEBUG")
    embedder = TitanEmbedder()
    test_texts = [
        "Microsoft reported strong revenue growth in fiscal year 2024.",
        "Key risk factors include cybersecurity threats and regulatory changes.",
    ]
    vectors = embedder.embed_texts(test_texts)
    for i, vec in enumerate(vectors):
        print(f"  Text {i+1}: dim={len(vec)}, first5={vec[:5]}")
    print(f"\n✅ Embedding self-test passed — {len(vectors)} vectors, dim={len(vectors[0])}")
