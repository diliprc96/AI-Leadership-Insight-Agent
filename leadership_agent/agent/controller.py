"""
controller.py — LangGraph graph definition and execution controller.

Graph topology:
    START → planner → tool_executor → synthesizer → END

v1.0 (Stable): Narrative Q&A via RetrieverTool only.
    FinancialTool / PlotTool are implemented but disabled — see Phase 2.
"""

import json
import logging
import time
from typing import Any

import boto3
from langgraph.graph import StateGraph, END

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
from leadership_agent.agent.planner import planner_node
from leadership_agent.tools.retriever_tool import retriever_tool
from leadership_agent.tools.financial_tool import financial_tool
from leadership_agent.tools.plot_tool import plot_tool

logger = logging.getLogger(__name__)


# ─── Bedrock client (lazy singleton) ──────────────────────────────────────────
_bedrock_client = None


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
        )
    return _bedrock_client


# ─── Tool Executor Node ────────────────────────────────────────────────────────

def tool_executor_node(state: AgentState) -> AgentState:
    """
    Dispatches to the appropriate tool based on the planner's decision.

    Tools:
      - retriever  : Semantic search over 10-K narrative text (Qdrant)
      - financial  : Year-over-year analysis from structured CSV tables
      - plot       : Matplotlib trend chart saved to static/trend.png
    """
    plan = state["plan"]
    query = state["query"]
    t0 = time.perf_counter()

    tool_output = ""
    sources: list[dict[str, Any]] = []
    image_path: str | None = None
    error = None

    logger.info("ToolExecutorNode: invoking tool='%s'", plan)

    try:
        if plan == "retriever":
            raw = retriever_tool.invoke({"query": query})
            tool_output = raw
            parsed = json.loads(raw)
            if parsed.get("status") == "ok":
                sources = [
                    {
                        "id":    c["id"],
                        "score": c["score"],
                        "text":  c["text"],
                        **c["metadata"],
                    }
                    for c in parsed.get("chunks", [])
                ]
            elif parsed.get("status") == "empty":
                logger.warning("RetrieverTool: no results for query=%r", query[:80])

        elif plan == "financial":
            raw = financial_tool.invoke({"query": query})
            tool_output = raw
            logger.info("FinancialTool output: %s", raw[:200])

        elif plan == "plot":
            raw = plot_tool.invoke({"query": query})
            tool_output = raw
            parsed_plot = json.loads(raw)
            if parsed_plot.get("status") == "ok":
                image_path = parsed_plot.get("image_path")
                logger.info("PlotTool: chart saved to %s", image_path)

        else:
            # Unknown plan — fall back to retriever
            logger.warning("Unknown plan '%s' — falling back to retriever", plan)
            raw = retriever_tool.invoke({"query": query})
            tool_output = raw

    except Exception as exc:
        logger.error("ToolExecutorNode error: %s", exc, exc_info=True)
        error = str(exc)
        tool_output = json.dumps({"status": "error", "message": error})

    elapsed = time.perf_counter() - t0
    logger.info(
        "ToolExecutorNode complete — elapsed=%.3fs, sources=%d, image=%s",
        elapsed, len(sources), image_path,
    )

    return {
        **state,
        "tool_outputs": tool_output,
        "tools_used": [plan],
        "sources": sources,
        "image_path": image_path,
        "error": error,
        "metrics": {
            **state.get("metrics", {}),
            "tool_latency_s": round(elapsed, 3),
        },
    }


# ─── Synthesizer Node ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a financial intelligence assistant specialising in Microsoft 10-K annual reports "
    "(FY2023–FY2025). Use the provided tool output to give a clear, concise, factual answer. "
    "If the tool output mentions a phase2_note, include it politely in your answer. "
    "If numeric data is present, highlight key figures. "
    "Keep the answer under 300 words."
)


def synthesizer_node(state: AgentState) -> AgentState:
    """Call Nova Pro to compose a final answer from the tool output."""
    query = state["query"]
    tool_outputs = state.get("tool_outputs", "{}")
    logger.info("SynthesizerNode: composing answer for query=%r", query[:80])
    t0 = time.perf_counter()

    user_msg = (
        f"User Question: {query}\n\n"
        f"Tool Output (JSON):\n{tool_outputs}\n\n"
        f"Please provide a clear, factual answer based on the tool output."
    )

    final_answer = ""
    error = state.get("error")

    if error:
        final_answer = (
            f"I encountered an error while processing your request: {error}. "
            "Please ensure the documents have been ingested and try again."
        )
    else:
        try:
            bedrock = _get_bedrock()
            response = bedrock.converse(
                modelId=LLM_MODEL_ID,
                system=[{"text": _SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                inferenceConfig={
                    "maxTokens": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                    "topP": LLM_TOP_P,
                },
            )
            final_answer = response["output"]["message"]["content"][0]["text"]
            usage = response.get("usage", {})
            logger.info(
                "SynthesizerNode: answer_len=%d, tokens_in=%s, tokens_out=%s",
                len(final_answer),
                usage.get("inputTokens"),
                usage.get("outputTokens"),
            )
        except Exception as exc:
            logger.error("SynthesizerNode LLM error: %s", exc, exc_info=True)
            final_answer = (
                "I was unable to generate a response due to an LLM error. "
                f"Raw tool output: {str(tool_outputs)[:500]}"
            )

    elapsed = time.perf_counter() - t0
    logger.info("SynthesizerNode complete in %.3fs", elapsed)

    return {
        **state,
        "final_answer": final_answer,
        "metrics": {
            **state.get("metrics", {}),
            "llm_latency_s": round(elapsed, 3),
        },
    }


# ─── Graph Construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner",       planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("synthesizer",   synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",       "tool_executor")
    graph.add_edge("tool_executor", "synthesizer")
    graph.add_edge("synthesizer",   END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Building LangGraph StateGraph...")
        _compiled_graph = build_graph()
        logger.info("Graph compiled successfully.")
    return _compiled_graph


def run_agent(query: str) -> AgentState:
    """
    Run the full agent pipeline for a query.

    Args:
        query: User's natural language question.

    Returns:
        Final AgentState with all fields populated.
    """
    logger.info("AgentController: start — query=%r", query[:120])
    t_total = time.perf_counter()

    initial_state: AgentState = {
        "query": query,
        "plan": "",
        "plan_reasoning": "",
        "tool_outputs": "",
        "final_answer": "",
        "tools_used": [],
        "sources": [],
        "image_path": None,
        "error": None,
        "metrics": {"request_start_ts": t_total},
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    total_elapsed = time.perf_counter() - t_total
    final_state["metrics"]["total_latency_s"] = round(total_elapsed, 3)

    logger.info(
        "AgentController: done — tool=%s, answer_len=%d, total=%.3fs",
        final_state.get("plan"),
        len(final_state.get("final_answer", "")),
        total_elapsed,
    )
    return final_state
