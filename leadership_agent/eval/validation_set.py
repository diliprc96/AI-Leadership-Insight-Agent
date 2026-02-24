"""
validation_set.py — Hardcoded RAGAS validation query set.

10 questions derived from the Adobe assignment NL input examples
and standard 10-K analysis queries. Each sample includes:
  - query        : the NL question
  - expected_themes : keywords a good answer should touch (used for context recall hint)
"""

from dataclasses import dataclass, field


@dataclass
class ValidationSample:
    query: str
    expected_themes: list[str] = field(default_factory=list)


VALIDATION_SET: list[ValidationSample] = [
    ValidationSample(
        query="What are the key risks Microsoft faces in FY2024?",
        expected_themes=["cybersecurity", "regulation", "competition", "supply chain", "AI"],
    ),
    ValidationSample(
        query="What is Microsoft's cloud strategy?",
        expected_themes=["Azure", "cloud", "AI", "infrastructure", "data centre"],
    ),
    ValidationSample(
        query="How does Microsoft describe its AI investments?",
        expected_themes=["AI", "Copilot", "OpenAI", "Azure AI", "investment"],
    ),
    ValidationSample(
        query="What are the main competition risks Microsoft faces?",
        expected_themes=["Google", "Amazon", "Meta", "competition", "market share"],
    ),
    ValidationSample(
        query="How has Microsoft's revenue changed between FY2023 and FY2025?",
        expected_themes=["revenue", "growth", "cloud", "Intelligent Cloud", "Productivity"],
    ),
    ValidationSample(
        query="What does Microsoft say about cybersecurity risks?",
        expected_themes=["cybersecurity", "nation-state", "ransomware", "data breach", "security"],
    ),
    ValidationSample(
        query="What is Microsoft's strategy for generative AI?",
        expected_themes=["generative AI", "Copilot", "Azure OpenAI", "foundation models", "integration"],
    ),
    ValidationSample(
        query="What regulatory risks does Microsoft highlight?",
        expected_themes=["regulation", "EU AI Act", "antitrust", "GDPR", "compliance"],
    ),
    ValidationSample(
        query="How does Microsoft describe its gaming business?",
        expected_themes=["Xbox", "gaming", "Activision", "Game Pass", "console"],
    ),
    ValidationSample(
        query="What does Microsoft say about its sustainability and ESG commitments?",
        expected_themes=["carbon", "sustainability", "ESG", "emissions", "renewable"],
    ),
]
