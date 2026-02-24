"""
pdf_parser.py — Document ingestion using Docling.

Parses DOCX/PDF 10-K reports, extracts:
  - Narrative text (chunked at 1200 chars with 200 overlap)
  - Tables (saved as CSV under data/structured/)

Each chunk carries metadata: company, year, document_type, section.

Usage as script:
    python -m leadership_agent.ingestion.pdf_parser
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# Project imports
from leadership_agent.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_RAW_DIR,
    DATA_STRUCTURED_DIR,
    COMPANY_MAP,
    PDF_OCR_ENABLED,
)

logger = logging.getLogger(__name__)


# ─── Converter Factory ───────────────────────────────────────────────────────

def _build_converter(file_path: Path) -> DocumentConverter:
    """
    Return a Docling DocumentConverter configured for the given file type.

    - DOCX : plain converter (Docling handles DOCX natively, no PDF options needed)
    - PDF  : explicit PdfFormatOption with OCR on/off via PDF_OCR_ENABLED config.
              Default is False — digitally-born SEC 10-K PDFs don't need OCR.
              Set PDF_OCR_ENABLED=true in .env for scanned/image-based PDFs.
    """
    if file_path.suffix.lower() == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=PDF_OCR_ENABLED)
        logger.debug(
            "PDF converter — OCR enabled: %s", PDF_OCR_ENABLED
        )
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
    logger.debug("DOCX converter selected for: %s", file_path.name)
    return DocumentConverter()


# ─── Metadata Inference ────────────────────────────────────────────────────────

def _infer_metadata(filename: str) -> dict[str, str]:
    """
    Infer metadata from filename convention: MSFT_FY23Q4_10K.docx
    Returns: {"company": "Microsoft", "year": "2023", "document_type": "10K"}
    """
    stem = Path(filename).stem.upper()  # e.g. "MSFT_FY23Q4_10K"
    parts = stem.split("_")

    # Company
    ticker = parts[0] if parts else "UNKNOWN"
    company = COMPANY_MAP.get(ticker, ticker)

    # Year — look for FYxx pattern
    year = "UNKNOWN"
    for part in parts:
        match = re.search(r"FY(\d{2,4})", part)
        if match:
            y = match.group(1)
            year = f"20{y}" if len(y) == 2 else y
            break

    # Document type
    doc_type = "10K" if "10K" in stem else "UNKNOWN"

    return {"company": company, "year": year, "document_type": doc_type}


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ─── Table Extraction ─────────────────────────────────────────────────────────

def _save_tables(doc: Any, meta: dict[str, str], structured_dir: Path) -> int:
    """
    Extract tables from a Docling Document and save each as CSV.
    Returns the number of tables saved.
    """
    table_count = 0
    try:
        for idx, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe()
                if df is not None and not df.empty:
                    fname = f"{meta['company']}_{meta['year']}_table_{idx + 1}.csv"
                    fpath = structured_dir / fname
                    df.to_csv(fpath, index=False)
                    table_count += 1
                    logger.info(
                        "Table saved: %s (%d rows × %d cols)",
                        fname, df.shape[0], df.shape[1],
                    )
            except Exception as table_err:
                logger.warning("Could not export table %d: %s", idx, table_err)
    except AttributeError:
        logger.debug("No .tables attribute on document object.")
    return table_count


# ─── Section Tagging ──────────────────────────────────────────────────────────

_SECTION_KEYWORDS: list[tuple[str, str]] = [
    ("risk factor", "Risk Factors"),
    ("management's discussion", "MD&A"),
    ("financial statement", "Financial Statements"),
    ("revenue", "Revenue"),
    ("operating income", "Operating Results"),
    ("segment", "Segment Information"),
    ("business overview", "Business Overview"),
    ("quantitative and qualitative", "Market Risk"),
    ("legal proceeding", "Legal Proceedings"),
    ("note to", "Notes to Financial Statements"),
]


def _infer_section(text: str) -> str:
    """Heuristically tag a chunk with a section name."""
    lower = text[:400].lower()
    for keyword, section in _SECTION_KEYWORDS:
        if keyword in lower:
            return section
    return "General"


# ─── Core Parser ──────────────────────────────────────────────────────────────

def parse_document(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a single DOCX or PDF file using Docling and return a list of chunks.

    Each chunk is a dict:
        {
            "text": str,
            "metadata": {
                "company": str,
                "year": str,
                "document_type": str,
                "section": str,
                "source_file": str,
                "chunk_index": int,
            }
        }
    """
    file_path = Path(file_path)
    logger.info("Starting ingestion for: %s", file_path.name)
    t0 = time.perf_counter()

    # ── Build metadata from filename ──────────────────────────────────────────
    meta = _infer_metadata(file_path.name)
    logger.info("Inferred metadata: %s", meta)

    # ── Convert document ──────────────────────────────────────────────────────
    try:
        converter = _build_converter(file_path)
        logger.info("Converter type: %s for file: %s", type(converter).__name__, file_path.name)
        result = converter.convert(str(file_path))
        doc = result.document
    except Exception as exc:
        logger.error("Docling conversion failed for %s: %s", file_path.name, exc, exc_info=True)
        return []

    # ── Export narrative text ─────────────────────────────────────────────────
    try:
        full_text = doc.export_to_markdown()
    except Exception:
        try:
            full_text = doc.export_to_text()
        except Exception as exp:
            logger.error("Text export failed: %s", exp, exc_info=True)
            return []

    if not full_text.strip():
        logger.warning("No text extracted from %s — skipping.", file_path.name)
        return []

    # ── Chunk text ────────────────────────────────────────────────────────────
    raw_chunks = _chunk_text(full_text)
    logger.info("Text chunks created: %d", len(raw_chunks))

    # ── Save tables ───────────────────────────────────────────────────────────
    table_count = _save_tables(doc, meta, DATA_STRUCTURED_DIR)
    logger.info("Tables saved to data/structured/: %d", table_count)

    # ── Build chunk objects ──────────────────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(raw_chunks):
        section = _infer_section(chunk_text)
        chunk = {
            "text": chunk_text,
            "metadata": {
                "company": meta["company"],
                "year": meta["year"],
                "document_type": meta["document_type"],
                "section": section,
                "source_file": file_path.name,
                "chunk_index": idx,
            },
        }
        chunks.append(chunk)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ingestion complete: %s — %d chunks in %.2fs",
        file_path.name, len(chunks), elapsed,
    )
    return chunks


# ─── Batch Ingestion ──────────────────────────────────────────────────────────

def ingest_all(raw_dir: Path = DATA_RAW_DIR) -> list[dict[str, Any]]:
    """
    Parse all DOCX/PDF files in raw_dir and return combined list of chunks.
    """
    supported = {".docx", ".pdf"}
    files = [f for f in raw_dir.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.warning("No DOCX/PDF files found in %s", raw_dir)
        return []

    logger.info("Found %d document(s) to ingest: %s", len(files), [f.name for f in files])
    all_chunks: list[dict[str, Any]] = []

    for file_path in sorted(files):
        chunks = parse_document(file_path)
        all_chunks.extend(chunks)
        logger.info("Cumulative chunk total: %d", len(all_chunks))

    logger.info("All documents ingested. Total chunks: %d", len(all_chunks))
    return all_chunks


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    from leadership_agent.logging_config import setup_logging

    setup_logging("INFO")
    chunks = ingest_all()
    print(f"\n✅ Total chunks created: {len(chunks)}")
    if chunks:
        print(f"   Sample chunk metadata: {chunks[0]['metadata']}")
        print(f"   Sample text preview: {chunks[0]['text'][:200]!r}")
