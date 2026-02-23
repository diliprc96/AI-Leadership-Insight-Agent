"""
plot_tool.py — LangChain tool for financial trend visualization.

Reads structured CSV data, generates a matplotlib bar/line chart,
and saves it to static/trend.png.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool

from leadership_agent.config import DATA_STRUCTURED_DIR, PLOT_OUTPUT_PATH, STATIC_DIR

logger = logging.getLogger(__name__)


# ─── Data Extraction (reuses logic from financial_tool) ───────────────────────

_REVENUE_KEYWORDS = ["revenue", "net revenue", "total revenue", "sales", "net sales"]
_INCOME_KEYWORDS  = ["operating income", "income from operations", "net income"]


def _find_columns(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    return [
        col for col in df.columns
        if any(kw.lower() in col.lower() for kw in keywords)
    ]


def _extract_year(filename: str) -> str | None:
    m = re.search(r"FY(\d{2,4})", filename, re.IGNORECASE)
    if m:
        y = m.group(1)
        return f"20{y}" if len(y) == 2 else y
    return None


def _parse_numeric(val: str) -> float | None:
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(val))
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _collect_values(query: str) -> tuple[str, dict[str, float]]:
    """
    Load CSVs and collect metric values by year.

    Returns:
        (metric_label, values_by_year)
    """
    csv_files = list(DATA_STRUCTURED_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files in {DATA_STRUCTURED_DIR}. Run ingestion first."
        )

    query_lower = query.lower()
    if any(kw in query_lower for kw in ["income", "profit", "operating"]):
        keywords = _INCOME_KEYWORDS
        metric_label = "Operating Income"
    else:
        keywords = _REVENUE_KEYWORDS
        metric_label = "Revenue"

    values_by_year: dict[str, float] = {}

    for fpath in sorted(csv_files):
        year = _extract_year(fpath.name)
        if not year:
            continue
        try:
            df = pd.read_csv(fpath, dtype=str)
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath.name, exc)
            continue

        cols = _find_columns(df, keywords)
        logger.debug("File=%s year=%s cols=%s", fpath.name, year, cols)

        for col in cols:
            for val_str in df[col].dropna():
                val = _parse_numeric(val_str)
                if val and val > 0:
                    if year not in values_by_year or val > values_by_year[year]:
                        values_by_year[year] = val

    logger.info("PlotTool collected: metric=%s values=%s", metric_label, values_by_year)
    return metric_label, values_by_year


# ─── Chart Generation ─────────────────────────────────────────────────────────

def _generate_chart(metric_label: str, values_by_year: dict[str, float]) -> str:
    """
    Generate and save matplotlib bar chart.

    Returns:
        Absolute path to saved image.
    """
    years = sorted(values_by_year.keys())
    values = [values_by_year[y] for y in years]

    logger.info(
        "Generating chart — metric=%s, years=%s, values=%s",
        metric_label, years, values,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    # Bar chart
    bars = ax.bar(years, values)

    # Labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add trend line
    if len(years) > 1:
        ax.plot(years, values, marker="o", linestyle="--", linewidth=1.5, zorder=5)

    ax.set_title(f"Microsoft {metric_label} Trend (FY 2023–2025)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel(f"{metric_label} (USD millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    # Ensure static dir exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    out_path = str(PLOT_OUTPUT_PATH)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Plot saved to: %s", out_path)
    return out_path


# ─── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def plot_tool(query: str) -> str:
    """
    Generate a financial trend chart from 10-K extracted data.

    Use this tool when the user asks for a graph, chart, or visual:
    - "Show revenue trend graph."
    - "Plot operating income over 3 years."
    - "Create a chart of revenue 2023–2025."

    Args:
        query: The visualization request.

    Returns:
        JSON string with image_path on success, or error message.
    """
    logger.info("PlotTool invoked — query: %r", query[:120])
    t0 = time.perf_counter()

    try:
        metric_label, values_by_year = _collect_values(query)

        if not values_by_year:
            return json.dumps({
                "status": "no_data",
                "message": "No numeric data found in structured CSVs to plot.",
            })

        out_path = _generate_chart(metric_label, values_by_year)
        elapsed = time.perf_counter() - t0
        logger.info("PlotTool completed in %.3fs", elapsed)

        return json.dumps({
            "status": "ok",
            "metric": metric_label,
            "years_plotted": sorted(values_by_year.keys()),
            "image_path": out_path,
            "plot_latency_s": round(elapsed, 3),
        })

    except FileNotFoundError as exc:
        logger.warning("PlotTool: %s", exc)
        return json.dumps({"status": "error", "message": str(exc), "image_path": None})
    except Exception as exc:
        logger.error("PlotTool unexpected error: %s", exc, exc_info=True)
        return json.dumps({"status": "error", "message": str(exc), "image_path": None})
