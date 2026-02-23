"""
planner.py — Query classification / routing node.

Deterministic keyword routing first; LLM-assisted disambiguation fallback.

Routing rules:
  - Contains "plot" / "chart" / "graph" / "trend graph" → plot
  - Contains "revenue" / "growth" / "compare" / "trend" / "income" / "profit" → financial
  - Everything else → retriever
"""

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

from leadership_agent.config import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from leadership_agent.agent.state import AgentState

logger = logging.getLogger(__name__)

# ─── Keyword routing sets ──────────────────────────────────────────────────────
_PLOT_KEYWORDS     = {"plot", "chart", "graph", "visuali", "show trend", "bar chart", "line chart"}
_FINANCIAL_KEYWORDS = {
    "revenue", "growth", "compare", "comparison", "trend", "income",
    "profit", "operating", "year over year", "yoy", "fiscal", "earnings",
    "sales", "margin",
}


def _keyword_route(query: str) -> str | None:
    """
    Fast keyword-based routing. Returns tool name or None if ambiguous.
    """
    lower = query.lower()
    if any(kw in lower for kw in _PLOT_KEYWORDS):
        return "plot"
    if any(kw in lower for kw in _FINANCIAL_KEYWORDS):
        return "financial"
    return None


def _llm_route(query: str, bedrock_client) -> tuple[str, str]:
    """
    Fallback: ask Nova Pro to classify the query.
    Returns (tool_name, reasoning).
    """
    system_prompt = (
        "You are a routing agent. Given a user query about Microsoft 10-K reports, "
        "classify it into exactly one of these categories:\n"
        "  - 'retriever' : narrative, qualitative, or risk questions\n"
        "  - 'financial' : quantitative trend or number analysis\n"
        "  - 'plot'      : requests for a chart, graph, or visualization\n\n"
        "Reply ONLY with valid JSON: {\"tool\": \"<category>\", \"reason\": \"<one sentence>\"}"
    )

    try:
        response = bedrock_client.converse(
            modelId=LLM_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": query}]}],
            inferenceConfig={
                "maxTokens": 80,
                "temperature": 0.0,   # Deterministic for routing
                "topP": 1.0,
            },
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        parsed = json.loads(text)
        tool = parsed.get("tool", "retriever").lower()
        reason = parsed.get("reason", "LLM classification")
        if tool not in {"retriever", "financial", "plot"}:
            tool = "retriever"
        return tool, reason
    except Exception as exc:
        logger.warning("LLM routing failed (%s) — defaulting to 'retriever'", exc)
        return "retriever", "Fallback due to LLM routing error"


# ─── Planner Node ─────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """
    LangGraph node: classify the query and decide which tool to invoke.

    Mutates state keys: plan, plan_reasoning, tools_used, metrics.
    """
    query = state["query"]
    logger.info("PlannerNode: classifying query — %r", query[:120])
    t0 = time.perf_counter()

    # 1. Try fast keyword routing
    tool_name = _keyword_route(query)
    reasoning = "Keyword-based routing"

    # 2. Fallback to LLM if keyword routing is inconclusive
    if tool_name is None:
        logger.debug("No keyword match — falling back to LLM routing")
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
        )
        tool_name, reasoning = _llm_route(query, bedrock)

    elapsed = time.perf_counter() - t0
    logger.info(
        "PlannerNode: selected tool='%s', reasoning='%s', elapsed=%.3fs",
        tool_name, reasoning, elapsed,
    )

    return {
        **state,
        "plan": tool_name,
        "plan_reasoning": reasoning,
        "tools_used": [],
        "sources": [],
        "image_path": None,
        "error": None,
        "metrics": {
            **state.get("metrics", {}),
            "planner_latency_s": round(elapsed, 3),
        },
    }
