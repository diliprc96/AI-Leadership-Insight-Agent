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

# ── Phase 2 tools (implemented, not yet stable — disabled for v1.0) ────────────
# from leadership_agent.tools.financial_tool import financial_tool
# from leadership_agent.tools.plot_tool import plot_tool

logger = logging.getLogger(__name__)

_PHASE2_NOTE = (
    "Financial trend analysis and chart generation are planned for Phase 2. "
    "Searching the narrative report text instead."
)


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
    v1.0: Only the RetrieverTool is active.

    If the planner routes to 'financial' or 'plot', we redirect to the
    retriever and append a Phase 2 note.  Tool files remain in the repo
    for Phase 2 activation.
    """
    plan = state["plan"]
    query = state["query"]
    t0 = time.perf_counter()

    tool_output = ""
    sources: list[dict[str, Any]] = []
    error = None
    phase2_redirected = False

    # ── Phase 2 redirect ─────────────────────────────────────────────────────
    if plan in ("financial", "plot"):
        logger.info(
            "ToolExecutorNode: '%s' is a Phase 2 feature — redirecting to retriever. query=%r",
            plan, query[:80],
        )
        phase2_redirected = True
        plan = "retriever"

    logger.info("ToolExecutorNode: invoking tool='retriever'")

    try:
        raw = retriever_tool.invoke({"query": query})
        tool_output = raw
        parsed = json.loads(raw)
        if parsed.get("status") == "ok":
            sources = [
                {
                    "id":   c["id"],
                    "score": c["score"],
                    "text": c["text"],          # ← included for RAGAS context extraction
                    **c["metadata"],
                }
                for c in parsed.get("chunks", [])
            ]
        elif parsed.get("status") == "empty":
            logger.warning("RetrieverTool: no results for query=%r", query[:80])

    except Exception as exc:
        logger.error("ToolExecutorNode error: %s", exc, exc_info=True)
        error = str(exc)
        tool_output = json.dumps({"status": "error", "message": error})

    # Inject phase2 note so synthesizer can mention it gracefully
    if phase2_redirected:
        try:
            parsed_out = json.loads(tool_output)
            parsed_out["phase2_note"] = _PHASE2_NOTE
            tool_output = json.dumps(parsed_out)
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    logger.info(
        "ToolExecutorNode complete — elapsed=%.3fs, sources=%d",
        elapsed, len(sources),
    )

    return {
        **state,
        "tool_outputs": tool_output,
        "tools_used": ["retriever"],
        "sources": sources,
        "image_path": None,
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
