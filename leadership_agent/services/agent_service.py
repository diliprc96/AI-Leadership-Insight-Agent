"""
agent_service.py — Orchestration layer between entry points and the agent.

Provides a single run() method that:
  1. Calls the LangGraph agent controller
  2. Tracks per-request metrics
  3. Persists metrics to logs/metrics.jsonl
  4. Returns a clean response dict for CLI/API consumers
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from leadership_agent.agent.controller import run_agent
from leadership_agent.config import METRICS_FILE

logger = logging.getLogger(__name__)


class AgentService:
    """Entry-point facade for running the Leadership Agent."""

    def __init__(self) -> None:
        logger.info("AgentService initialised")

    def run(self, query: str) -> dict[str, Any]:
        """
        Execute the agent pipeline for a user query.

        Args:
            query: The user's natural language question.

        Returns:
            dict with keys:
                answer     : str   — synthesized LLM answer
                tools_used : list  — tool names invoked
                sources    : list  — source metadata dicts
                image_path : str | None — chart path if generated
                metrics    : dict  — latency statistics
                error      : str | None — error if present
        """
        logger.info("AgentService.run() — query=%r", query[:120])
        t0 = time.perf_counter()

        try:
            state = run_agent(query)

            response = {
                "answer":     state.get("final_answer", ""),
                "tools_used": state.get("tools_used", []),
                "sources":    state.get("sources", []),
                "image_path": state.get("image_path"),
                "metrics":    state.get("metrics", {}),
                "error":      state.get("error"),
            }

        except Exception as exc:
            logger.error("AgentService.run() fatal error: %s", exc, exc_info=True)
            response = {
                "answer":     f"Agent encountered an unrecoverable error: {exc}",
                "tools_used": [],
                "sources":    [],
                "image_path": None,
                "metrics":    {},
                "error":      str(exc),
            }

        elapsed = time.perf_counter() - t0
        response["metrics"]["total_service_latency_s"] = round(elapsed, 3)

        # Persist metrics
        self._save_metrics(query, response)

        logger.info(
            "AgentService.run() complete — total=%.3fs, tools=%s, answer_len=%d",
            elapsed,
            response["tools_used"],
            len(response["answer"]),
        )
        return response

    def _save_metrics(self, query: str, response: dict[str, Any]) -> None:
        """Append a metrics record to logs/metrics.jsonl."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query":     query[:100],
            "tools":     response.get("tools_used", []),
            "image":     response.get("image_path") is not None,
            "error":     response.get("error") is not None,
            **response.get("metrics", {}),
        }
        try:
            with open(METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug("Metrics saved to %s", METRICS_FILE)
        except Exception as exc:
            logger.warning("Could not save metrics: %s", exc)
