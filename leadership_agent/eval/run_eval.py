"""
run_eval.py — CLI entry point for the RAGAS evaluation harness.

Runs the validation query set through the full agent pipeline,
scores each result, prints a summary table, and saves JSONL output.

Usage:
    python -m leadership_agent.eval.run_eval
    python -m leadership_agent.eval.run_eval --samples 2
    python -m leadership_agent.eval.run_eval --output logs/my_eval.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from leadership_agent.logging_config import setup_logging
setup_logging("INFO")

from leadership_agent.config import EVAL_RESULTS_FILE
from leadership_agent.services.agent_service import AgentService
from leadership_agent.eval.ragas_eval import RAGASEvaluator
from leadership_agent.eval.validation_set import VALIDATION_SET

logger = logging.getLogger(__name__)


def _extract_contexts_and_chunks(response: dict) -> tuple[list[str], list[dict]]:
    """
    Pull the retrieved text passages and raw chunk dicts from an AgentService response.

    The 'sources' field contains metadata dicts; we need the 'text' field for RAGAS context.
    """
    sources = response.get("sources", [])
    contexts = [s.get("text", "") for s in sources if s.get("text")]
    return contexts, sources


def run_evaluation(
    num_samples: int | None = None,
    output_path: Path = EVAL_RESULTS_FILE,
) -> list[dict]:
    """
    Run RAGAS evaluation over the validation set.

    Args:
        num_samples : Limit evaluation to first N samples (None = all 10)
        output_path : JSONL file to write results to

    Returns:
        List of result dicts
    """
    samples = VALIDATION_SET[:num_samples] if num_samples else VALIDATION_SET
    logger.info("Starting RAGAS evaluation — %d samples", len(samples))

    service = AgentService()
    evaluator = RAGASEvaluator()
    all_results = []

    for i, sample in enumerate(samples, 1):
        logger.info("[%d/%d] Evaluating: %r", i, len(samples), sample.query)
        print(f"\n[{i}/{len(samples)}] Running agent for: {sample.query!r}")

        # Run the agent
        agent_response = service.run(sample.query)

        answer = agent_response.get("answer", "")
        contexts, chunks = _extract_contexts_and_chunks(agent_response)

        if not answer:
            print(f"  ⚠️  No answer returned — skipping RAGAS scoring.")
            result_dict = {
                "query": sample.query,
                "answer_preview": "",
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "mean_score": 0.0,
                "num_chunks": 0,
                "latency_s": 0.0,
                "error": "No answer from agent",
            }
            all_results.append(result_dict)
            continue

        # Score with RAGAS
        eval_result = evaluator.evaluate_sample(
            query=sample.query,
            answer=answer,
            contexts=contexts,
            chunks=chunks,
        )
        result_dict = eval_result.to_dict()
        all_results.append(result_dict)

        print(
            f"  ✅ Faithfulness={eval_result.faithfulness:.2f} | "
            f"Relevancy={eval_result.answer_relevancy:.2f} | "
            f"CtxRecall={eval_result.context_recall:.2f} | "
            f"Mean={eval_result.mean_score:.2f}"
        )

    # ── Save JSONL ────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    logger.info("Results saved to %s", output_path)

    # ── Print summary table ──────────────────────────────────────────────────
    _print_summary(all_results)

    return all_results


def _print_summary(results: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    if not results:
        print("\nNo results to summarise.")
        return

    try:
        from tabulate import tabulate
        rows = [
            [
                r["query"][:55] + "…" if len(r["query"]) > 55 else r["query"],
                f"{r['faithfulness']:.2f}",
                f"{r['answer_relevancy']:.2f}",
                f"{r['context_recall']:.2f}",
                f"{r['mean_score']:.2f}",
            ]
            for r in results
        ]
        headers = ["Query", "Faithfulness", "Relevancy", "Ctx Recall", "Mean"]
        print("\n" + "=" * 80)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 80)
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

        # Aggregate means
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_relev = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_recall = sum(r["context_recall"] for r in results) / len(results)
        avg_mean = sum(r["mean_score"] for r in results) / len(results)
        print(f"\n  AVERAGES — Faithfulness: {avg_faith:.3f} | Relevancy: {avg_relev:.3f} | "
              f"CtxRecall: {avg_recall:.3f} | Overall Mean: {avg_mean:.3f}")
        print("=" * 80)

    except ImportError:
        # tabulate not installed — plain fallback
        print("\n" + "=" * 70)
        print("RAGAS RESULTS SUMMARY")
        print("=" * 70)
        for r in results:
            print(
                f"  Q: {r['query'][:50]!r}\n"
                f"     Faith={r['faithfulness']:.2f} | Relev={r['answer_relevancy']:.2f} | "
                f"Recall={r['context_recall']:.2f} | Mean={r['mean_score']:.2f}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the validation set")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of validation samples to run (default: all 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_RESULTS_FILE),
        help="Path to JSONL output file (default: logs/eval_results.jsonl)",
    )
    args = parser.parse_args()

    run_evaluation(num_samples=args.samples, output_path=Path(args.output))
