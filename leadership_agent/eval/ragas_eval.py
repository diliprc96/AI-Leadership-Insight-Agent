"""
ragas_eval.py — RAGAS-compatible evaluation harness for the Leadership Agent.

Uses Amazon Nova Pro (AWS Bedrock) as the LLM judge — no OpenAI dependency.

Metrics implemented:
  1. Faithfulness      — Is the answer grounded in the retrieved context?  (0.0–1.0)
  2. Answer Relevancy  — Does the answer address the question?             (0.0–1.0)
  3. Context Recall    — Fraction of retrieved chunks with score >= threshold (heuristic, no LLM cost)

Usage:
    from leadership_agent.eval.ragas_eval import RAGASEvaluator
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate_sample(query, answer, contexts)
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import boto3

from leadership_agent.config import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    LLM_MODEL_ID,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# Minimum cosine similarity score to count a chunk as "recalled"
CONTEXT_RECALL_THRESHOLD: float = 0.70


# ─── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    num_chunks: int = 0
    latency_s: float = 0.0
    error: str | None = None

    @property
    def mean_score(self) -> float:
        return round((self.faithfulness + self.answer_relevancy + self.context_recall) / 3, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer_preview": self.answer[:200],
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "mean_score": self.mean_score,
            "num_chunks": self.num_chunks,
            "latency_s": self.latency_s,
            "error": self.error,
        }


# ─── Bedrock LLM Judge ────────────────────────────────────────────────────────

class BedrockJudge:
    """
    Wraps Amazon Nova Pro to score RAGAS metrics.

    Sends a prompt to the model and parses a float score from the response.
    Prompts are deliberately simple: ask for a single JSON object {"score": <float>}.
    """

    def __init__(self) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
        )
        self._model_id = LLM_MODEL_ID
        logger.info("BedrockJudge initialised — model: %s", self._model_id)

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Call Nova Pro via Converse API and return raw text response."""
        try:
            response = self._client.converse(
                modelId=self._model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={
                    "maxTokens": 128,   # scores only — very short response
                    "temperature": 0.0,  # deterministic scoring
                },
            )
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as exc:
            logger.error("BedrockJudge._call failed: %s", exc, exc_info=True)
            return '{"score": 0.0}'

    def _parse_score(self, raw: str) -> float:
        """Extract score float from model output. Falls back to 0.0 on parse failure."""
        # Try JSON first
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            pass
        # Regex fallback — find first float in response
        match = re.search(r"\b([01](?:\.\d+)?)\b", raw)
        if match:
            return float(match.group(1))
        logger.warning("Could not parse score from: %r", raw[:200])
        return 0.0

    # ── Metric: Faithfulness ──────────────────────────────────────────────────

    def score_faithfulness(self, query: str, answer: str, context: str) -> float:
        """
        Ask: Is every claim in the answer directly supported by the context?
        Score 1.0 = fully grounded, 0.0 = hallucinated / unsupported.
        """
        system = (
            "You are an evaluation judge. Given a QUESTION, an ANSWER, and CONTEXT passages, "
            "score how faithful the answer is to the context. "
            "A faithful answer contains only information that can be inferred from the context. "
            "Respond ONLY with JSON: {\"score\": <float between 0.0 and 1.0>}. No other text."
        )
        user = (
            f"QUESTION: {query}\n\n"
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"ANSWER: {answer[:1000]}\n\n"
            "Score faithfulness (0.0 = not grounded, 1.0 = fully grounded)."
        )
        raw = self._call(system, user)
        score = self._parse_score(raw)
        logger.debug("Faithfulness score: %.3f | raw: %r", score, raw[:100])
        return score

    # ── Metric: Answer Relevancy ──────────────────────────────────────────────

    def score_answer_relevancy(self, query: str, answer: str) -> float:
        """
        Ask: Does the answer actually address the question asked?
        Score 1.0 = directly answers the question, 0.0 = off-topic.
        """
        system = (
            "You are an evaluation judge. Given a QUESTION and an ANSWER, "
            "score how relevant the answer is to the question. "
            "A relevant answer directly addresses what was asked without unnecessary information. "
            "Respond ONLY with JSON: {\"score\": <float between 0.0 and 1.0>}. No other text."
        )
        user = (
            f"QUESTION: {query}\n\n"
            f"ANSWER: {answer[:1000]}\n\n"
            "Score answer relevancy (0.0 = off-topic, 1.0 = perfectly addresses the question)."
        )
        raw = self._call(system, user)
        score = self._parse_score(raw)
        logger.debug("Answer relevancy score: %.3f | raw: %r", score, raw[:100])
        return score


# ─── Heuristic: Context Recall ────────────────────────────────────────────────

def score_context_recall(chunks: list[dict[str, Any]], threshold: float = CONTEXT_RECALL_THRESHOLD) -> float:
    """
    Heuristic context recall: fraction of retrieved chunks with similarity
    score >= threshold. No LLM cost — uses the cosine scores from Qdrant.

    Args:
        chunks  : list of chunk dicts with a 'score' key (from RetrieverTool output)
        threshold: minimum score to count as "recalled" (default 0.70)

    Returns:
        float in [0.0, 1.0]
    """
    if not chunks:
        return 0.0
    recalled = sum(1 for c in chunks if c.get("score", 0.0) >= threshold)
    return round(recalled / len(chunks), 3)


# ─── Main Evaluator ───────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    RAGAS-compatible evaluator using Nova Pro as the LLM judge.

    Example usage:
        evaluator = RAGASEvaluator()
        result = evaluator.evaluate_sample(query, answer, contexts, chunks)
    """

    def __init__(self) -> None:
        self._judge = BedrockJudge()

    def evaluate_sample(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        chunks: list[dict[str, Any]] | None = None,
    ) -> EvalResult:
        """
        Run all three metrics for a single (query, answer, contexts) triple.

        Args:
            query    : The original NL question
            answer   : The agent's synthesized answer
            contexts : List of retrieved passage texts (from RetrieverTool chunks)
            chunks   : Raw chunk dicts with 'score' keys (for context recall heuristic)

        Returns:
            EvalResult with all metric scores
        """
        t0 = time.perf_counter()
        result = EvalResult(query=query, answer=answer, num_chunks=len(contexts))

        try:
            combined_context = "\n---\n".join(contexts)

            # Metric 1: Faithfulness (LLM judge)
            result.faithfulness = self._judge.score_faithfulness(query, answer, combined_context)

            # Metric 2: Answer Relevancy (LLM judge)
            result.answer_relevancy = self._judge.score_answer_relevancy(query, answer)

            # Metric 3: Context Recall (heuristic)
            result.context_recall = score_context_recall(chunks or [])

        except Exception as exc:
            logger.error("RAGASEvaluator.evaluate_sample failed: %s", exc, exc_info=True)
            result.error = str(exc)

        result.latency_s = round(time.perf_counter() - t0, 3)
        logger.info(
            "RAGAS | query=%r | faith=%.3f | relev=%.3f | recall=%.3f | mean=%.3f",
            query[:60], result.faithfulness, result.answer_relevancy,
            result.context_recall, result.mean_score,
        )
        return result
