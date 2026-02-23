"""
financial_tool.py — LangChain tool for financial trend analysis.

Loads CSVs from data/structured/, detects revenue/income columns,
computes year-over-year growth, and returns structured JSON.
"""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from leadership_agent.config import DATA_STRUCTURED_DIR

logger = logging.getLogger(__name__)


# ─── Column Detection ─────────────────────────────────────────────────────────

_REVENUE_KEYWORDS = [
    "revenue", "net revenue", "total revenue", "sales", "net sales",
]
_INCOME_KEYWORDS = [
    "operating income", "income from operations", "net income",
    "operating profit", "gross profit", "gross margin",
]


def _find_columns(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    """Return column names that match any keyword (case-insensitive)."""
    matches = []
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                matches.append(col)
                break
    return matches


# ─── CSV Loading ──────────────────────────────────────────────────────────────

def _load_all_csvs(structured_dir: Path = DATA_STRUCTURED_DIR) -> pd.DataFrame:
    """
    Load all CSVs from structured_dir, annotate with source filename,
    and concatenate into one DataFrame.
    """
    csv_files = list(structured_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {structured_dir}. Run ingestion first."
        )

    frames = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f, dtype=str)   # keep everything as str for safety
            df["_source_file"] = f.name
            frames.append(df)
            logger.info("CSV loaded: %s (%d rows × %d cols)", f.name, df.shape[0], df.shape[1])
            logger.debug("  Columns: %s", list(df.columns))
        except Exception as exc:
            logger.warning("Could not load %s: %s", f.name, exc)

    if not frames:
        raise ValueError("All CSV files failed to load.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined DataFrame: %d rows × %d columns", combined.shape[0], combined.shape[1])
    return combined


# ─── Year-over-Year Calculation ───────────────────────────────────────────────

def _extract_year_from_filename(filename: str) -> str | None:
    """Extract fiscal year from filename like MSFT_FY23Q4_... → '2023'."""
    match = re.search(r"FY(\d{2,4})", filename, re.IGNORECASE)
    if match:
        y = match.group(1)
        return f"20{y}" if len(y) == 2 else y
    return None


def _parse_numeric(value: str) -> float | None:
    """Parse a possibly formatted number string like '211,915' or '$123.4M'."""
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(value))
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _compute_yoy(values_by_year: dict[str, float]) -> dict[str, float | None]:
    """Compute year-over-year growth percentages."""
    yoy: dict[str, float | None] = {}
    years = sorted(values_by_year.keys())
    for i, year in enumerate(years):
        if i == 0:
            yoy[year] = None  # No prior year
        else:
            prev_year = years[i - 1]
            prev_val = values_by_year.get(prev_year)
            curr_val = values_by_year.get(year)
            if prev_val and curr_val and prev_val != 0:
                pct = round(((curr_val - prev_val) / abs(prev_val)) * 100, 2)
                yoy[year] = pct
            else:
                yoy[year] = None
    return yoy


# ─── Core Analysis ────────────────────────────────────────────────────────────

def _run_financial_analysis(query: str) -> dict:
    """Run the financial analysis and return structured dict."""
    df = _load_all_csvs()

    # Determine which metric to look for
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["income", "profit", "operating"]):
        keywords = _INCOME_KEYWORDS
        metric_label = "Operating Income"
    else:
        keywords = _REVENUE_KEYWORDS
        metric_label = "Revenue"

    matching_cols = _find_columns(df, keywords)
    logger.info("Metric='%s', matching columns: %s", metric_label, matching_cols)

    if not matching_cols:
        return {
            "status": "no_data",
            "message": f"No columns matching '{metric_label}' found in CSVs.",
            "available_columns": list(df.columns)[:20],
        }

    # Attempt to extract values per year from source files
    values_by_year: dict[str, float] = {}
    for _, row in df.iterrows():
        year = _extract_year_from_filename(str(row.get("_source_file", "")))
        if not year:
            continue
        for col in matching_cols:
            val = _parse_numeric(str(row.get(col, "")))
            if val is not None and val > 0:
                # Keep the max value (handles repeated rows with sub-totals)
                if year not in values_by_year or val > values_by_year[year]:
                    values_by_year[year] = val
                    logger.debug("  Year=%s col=%s val=%.2f", year, col, val)

    if not values_by_year:
        return {
            "status": "no_numeric_data",
            "message": "Could not parse numeric values from matching columns.",
            "matching_columns": matching_cols,
        }

    yoy = _compute_yoy(values_by_year)
    logger.info(
        "Financial analysis done — metric=%s, years=%s, yoy=%s",
        metric_label, sorted(values_by_year.keys()), yoy,
    )

    return {
        "status": "ok",
        "company": "Microsoft",
        "metric": metric_label,
        "values_by_year": {k: round(v, 2) for k, v in sorted(values_by_year.items())},
        "yoy_growth_pct": {k: v for k, v in sorted(yoy.items())},
        "columns_used": matching_cols,
    }


# ─── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def financial_tool(query: str) -> str:
    """
    Analyse financial trends from extracted 10-K CSV data.

    Use this tool for quantitative or trend questions such as:
    - "How has revenue changed from 2023 to 2025?"
    - "Compare operating income over 3 years."
    - "What is the year-over-year revenue growth?"

    Args:
        query: The financial question to analyse.

    Returns:
        JSON string with metric values by year and YoY growth percentages.
    """
    logger.info("FinancialTool invoked — query: %r", query[:120])
    t0 = time.perf_counter()

    try:
        result = _run_financial_analysis(query)
        elapsed = time.perf_counter() - t0
        result["analysis_latency_s"] = round(elapsed, 3)
        logger.info("FinancialTool completed in %.3fs — status=%s", elapsed, result.get("status"))
        return json.dumps(result, ensure_ascii=False)

    except FileNotFoundError as exc:
        logger.warning("FinancialTool: %s", exc)
        return json.dumps({"status": "error", "message": str(exc)})
    except Exception as exc:
        logger.error("FinancialTool unexpected error: %s", exc, exc_info=True)
        return json.dumps({"status": "error", "message": str(exc)})
