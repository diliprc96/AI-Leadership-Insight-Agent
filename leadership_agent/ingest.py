"""
ingest.py — Full ingestion pipeline runner.

Parses all DOCX files in data/raw/, generates Titan embeddings,
and stores vectors in the local Qdrant collection.

Run once before using the agent:
    python -m leadership_agent.ingest
    # or
    python leadership_agent/ingest.py
"""

import argparse
import logging
import sys
import time

from leadership_agent.logging_config import setup_logging
setup_logging("INFO")

from leadership_agent.config import DATA_RAW_DIR
from leadership_agent.ingestion.pdf_parser import ingest_all
from leadership_agent.embeddings.embedder import TitanEmbedder
from leadership_agent.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def run_ingestion(recreate: bool = False) -> None:
    """
    Full pipeline: parse → embed → store.

    Args:
        recreate: If True, wipe and rebuild the Qdrant collection.
    """
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE START")
    logger.info("=" * 60)
    t_total = time.perf_counter()

    # ── Step 1: Parse all documents ────────────────────────────────────────────
    logger.info("Step 1/3: Parsing documents from %s", DATA_RAW_DIR)
    chunks = ingest_all(DATA_RAW_DIR)

    if not chunks:
        logger.error("No chunks produced. Ensure DOCX or PDF files are in data/raw/.")
        sys.exit(1)

    logger.info("Step 1/3 complete — %d chunks produced", len(chunks))

    # ── Step 2: Generate embeddings ────────────────────────────────────────────
    logger.info("Step 2/3: Generating embeddings for %d chunks...", len(chunks))
    embedder = TitanEmbedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    logger.info("Step 2/3 complete — %d embeddings, dim=%d", len(embeddings), len(embeddings[0]))

    # ── Step 3: Store in Qdrant ────────────────────────────────────────────────
    logger.info("Step 3/3: Storing vectors in Qdrant...")
    store = QdrantStore()
    store.create_collection(recreate=recreate)
    n_stored = store.upsert(chunks, embeddings)

    total_elapsed = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("  Documents parsed : %d files", len(set(c["metadata"]["source_file"] for c in chunks)))
    logger.info("  Chunks created   : %d", len(chunks))
    logger.info("  Vectors stored   : %d", n_stored)
    logger.info("  Total time       : %.2fs", total_elapsed)
    logger.info("  Collection total : %d", store.count())
    logger.info("=" * 60)

    print(f"\n✅ Ingestion complete!")
    print(f"   Chunks: {len(chunks)}  |  Vectors stored: {n_stored}  |  Time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before ingesting",
    )
    args = parser.parse_args()
    run_ingestion(recreate=args.recreate)
