"""
cli.py — Command-line interface for the Leadership Agent.

Usage:
    # Recommended
    python -m leadership_agent.cli --query "What are the key risks in 2024?"

    # Also works from project root
    python leadership_agent/cli.py --query "..."
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
    args = parser.parse_args()

    # Re-init logging if the user changed the level
    if args.log_level != "INFO":
        setup_logging(args.log_level)

    service = AgentService()

    if args.query:
        # ─── Single-shot mode ──────────────────────────────────────────────────
        response = service.run(args.query)
        print_response(response)
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
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"\n❌  Unexpected error: {exc}\n")


if __name__ == "__main__":
    main()
