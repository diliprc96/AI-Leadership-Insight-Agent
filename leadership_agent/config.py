"""
config.py — Centralized configuration for the Leadership Agent.

All parameters, model IDs, and paths are defined here.
Import this module everywhere instead of hard-coding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# ─── AWS / Bedrock ────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


# ─── PDF Ingestion ────────────────────────────────────────────────────────────
# Set PDF_OCR_ENABLED=true in .env only for scanned/image PDFs.
# Digitally-born SEC 10-K PDFs do NOT need OCR (and it would be slow).
PDF_OCR_ENABLED: bool = os.getenv("PDF_OCR_ENABLED", "false").lower() == "true"


# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION: int = 1024
CHUNK_SIZE: int = 1200          # characters
CHUNK_OVERLAP: int = 200        # characters
EMBEDDING_BATCH_SIZE: int = 32


# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_MODEL_ID: str = "amazon.nova-pro-v1:0"
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 1024
LLM_TOP_P: float = 0.9


# ─── Vector DB ────────────────────────────────────────────────────────────────
QDRANT_PATH: str = str(_PROJECT_ROOT / "qdrant_storage")
COLLECTION_NAME: str = "leadership_reports"
QDRANT_TOP_K: int = 5


# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR: Path = _PROJECT_ROOT / "data" / "raw"
DATA_STRUCTURED_DIR: Path = _PROJECT_ROOT / "data" / "structured"
STATIC_DIR: Path = _PROJECT_ROOT / "static"
LOGS_DIR: Path = _PROJECT_ROOT / "logs"
PLOT_OUTPUT_PATH: Path = STATIC_DIR / "trend.png"
METRICS_FILE: Path = LOGS_DIR / "metrics.jsonl"
LOG_FILE: Path = LOGS_DIR / "agent.log"
EVAL_RESULTS_FILE: Path = LOGS_DIR / "eval_results.jsonl"

# Ensure directories exist at import time
for _dir in [DATA_RAW_DIR, DATA_STRUCTURED_DIR, STATIC_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Document Metadata Inference ─────────────────────────────────────────────
# Pattern: MSFT_FY23Q4_10K.docx  →  company=Microsoft, year=2023, doc_type=10K
COMPANY_MAP: dict = {
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "GOOGL": "Google",
}
