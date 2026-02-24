"""
cli.py — Command-line interface for the Leadership Agent.

Usage:
    # Single query
    python -m leadership_agent.cli --query "What are the key risks in 2024?"

    # Single query with inline RAGAS evaluation
    python -m leadership_agent.cli --query "What are the key risks in 2024?" --eval

    # Interactive mode (with optional eval scoring per answer)
    python -m leadership_agent.cli --eval
"""

import argparse
import sys
import textwrap
from pathlib import Path

# ── Ensure project root is on sys.path (works for both invocation styles) ─────
# When run as `python leadership_agent/cli.py`, the parent dir may not be on path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Bootstrap logging before any other project imports ────────────────────────
from leadership_agent.logging_config import setup_logging
setup_logging("INFO")

from leadership_agent.services.agent_service import AgentService

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║   AI Leadership Insight Agent  (Microsoft 10-K FY23-25) ║
║   Type your question and press Enter. Ctrl+C to exit.   ║
╚══════════════════════════════════════════════════════════╝
"""

_DIVIDER = "─" * 60


def print_response(response: dict) -> None:
    """Pretty-print the agent response to stdout."""
    print(f"\n{_DIVIDER}")

    # Answer
    answer = response.get("answer", "(no answer)")
    print("\n📝  ANSWER:")
    for line in textwrap.wrap(answer, width=78):
        print(f"    {line}")

    # Tools used
    tools = response.get("tools_used", [])
    print(f"\n🔧  TOOLS USED:  {', '.join(tools) if tools else 'none'}")

    # Sources
    sources = response.get("sources", [])
    if sources:
        print(f"\n📚  SOURCES ({len(sources)}):")
        for s in sources[:5]:
            company = s.get("company", "?")
            year    = s.get("year", "?")
            section = s.get("section", "?")
            score   = s.get("score", 0.0)
            print(f"    [{score:.3f}] {company} {year} — {section}")

    # Image
    image_path = response.get("image_path")
    if image_path:
        print(f"\n📊  CHART SAVED:  {image_path}")

    # Error
    error = response.get("error")
    if error:
        print(f"\n⚠️   ERROR:  {error}")

    # Metrics
    metrics = response.get("metrics", {})
    if metrics:
        parts = []
        for k in ("planner_latency_s", "tool_latency_s", "llm_latency_s", "total_latency_s"):
            if k in metrics:
                label = k.replace("_latency_s", "").replace("_", " ").title()
                parts.append(f"{label}: {metrics[k]:.2f}s")
        if parts:
            print(f"\n⏱️   TIMING:  {' | '.join(parts)}")

    print(f"\n{_DIVIDER}\n")


def print_eval_result(eval_result) -> None:
    """Print inline RAGAS evaluation scores after an answer."""
    print(f"\n{'─' * 60}")
    print("📊  RAGAS EVALUATION")
    print(f"    Faithfulness    : {eval_result.faithfulness:.2f}  (answer grounded in context?)")
    print(f"    Answer Relevancy: {eval_result.answer_relevancy:.2f}  (answer addresses the question?)")
    print(f"    Context Recall  : {eval_result.context_recall:.2f}  (chunks above similarity threshold?)")
    mean = eval_result.mean_score
    bar = "█" * int(mean * 10) + "░" * (10 - int(mean * 10))
    print(f"    Overall Mean    : {mean:.2f}  [{bar}]")
    if eval_result.error:
        print(f"    ⚠️  Eval error: {eval_result.error}")
    print(f"    Eval latency    : {eval_result.latency_s:.2f}s")
    print(f"{'─' * 60}\n")


def run_eval_on_response(query: str, response: dict) -> None:
    """Run RAGAS scoring on a completed agent response and print results."""
    from leadership_agent.eval.ragas_eval import RAGASEvaluator
    sources = response.get("sources", [])
    contexts = [s.get("text", "") for s in sources if s.get("text")]
    answer = response.get("answer", "")
    if not answer:
        print("\n⚠️  No answer to evaluate.")
        return
    print("\n⏳  Running RAGAS evaluation (2 LLM judge calls)...")
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate_sample(
        query=query,
        answer=answer,
        contexts=contexts,
        chunks=sources,
    )
    print_eval_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Leadership Insight Agent — Microsoft 10-K Q&A"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query string. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--eval", "-e",
        action="store_true",
        default=False,
        help="Run inline RAGAS evaluation after each answer (2 extra LLM calls per query).",
    )
    args = parser.parse_args()

    # Re-init logging if the user changed the level
    if args.log_level != "INFO":
        setup_logging(args.log_level)

    service = AgentService()

    if args.eval:
        print("[RAGAS eval mode ON — 2 extra LLM calls per query]")

    if args.query:
        # ─── Single-shot mode ──────────────────────────────────────────────────
        response = service.run(args.query)
        print_response(response)
        if args.eval:
            run_eval_on_response(args.query, response)
    else:
        # ─── Interactive mode ──────────────────────────────────────────────────
        print(_BANNER)
        while True:
            try:
                query = input("❓ Your question: ").strip()
                if not query:
                    continue
                if query.lower() in {"exit", "quit", "q"}:
                    print("Goodbye!")
                    break
                response = service.run(query)
                print_response(response)
                if args.eval:
                    run_eval_on_response(query, response)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"\n❌  Unexpected error: {exc}\n")


if __name__ == "__main__":
    main()
