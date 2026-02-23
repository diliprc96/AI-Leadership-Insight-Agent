"""
state.py — LangGraph AgentState definition.

The state dict flows through every node in the graph,
accumulating results as each node executes.
"""

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state for the Leadership Agent LangGraph graph.

    Fields:
        query:          The original user query string.
        plan:           Planner decision — "retriever" | "financial" | "plot"
        plan_reasoning: Short explanation of why this tool was chosen.
        tool_outputs:   Raw JSON string returned by the selected tool.
        final_answer:   Synthesized answer from the LLM synthesizer node.
        tools_used:     List of tool names that were invoked.
        sources:        List of source metadata dicts from retrieval.
        image_path:     Path to generated plot, or None.
        error:          Error message if any node failed, else None.
        metrics:        Dict of timing & count statistics.
    """
    query: str
    plan: str                           # "retriever" | "financial" | "plot"
    plan_reasoning: str
    tool_outputs: str                   # Raw JSON string from tool
    final_answer: str
    tools_used: list[str]
    sources: list[dict[str, Any]]
    image_path: str | None
    error: str | None
    metrics: dict[str, Any]
