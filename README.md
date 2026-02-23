# AI Leadership Insight & Decision Agent

A modular, locally-runnable **RAG + Tool-Augmented AI Agent** that ingests Microsoft 10-K annual reports (FY2023–FY2025) and answers narrative, financial, and visualization queries.

**Stack**: Amazon Bedrock (Titan Embed v2 + Nova Pro) · LangGraph · Qdrant · Docling · FastAPI

---

## Project Structure

```
leadership_agent/
├── app.py              # FastAPI entry point
├── cli.py              # CLI entry point
├── ingest.py           # Ingestion pipeline runner
├── config.py           # Centralized configuration
├── logging_config.py   # Structured logging setup
├── ingestion/
│   └── pdf_parser.py   # Docling DOCX parser + chunker
├── embeddings/
│   └── embedder.py     # Amazon Titan Embed v2 wrapper
├── vectorstore/
│   └── qdrant_store.py # Local Qdrant vector DB
├── tools/
│   ├── retriever_tool.py   # Semantic search tool
│   ├── financial_tool.py   # Financial trend analysis tool
│   └── plot_tool.py        # Matplotlib chart tool
├── agent/
│   ├── state.py        # LangGraph AgentState
│   ├── planner.py      # Query routing (keyword + LLM)
│   └── controller.py   # LangGraph graph controller
└── services/
    └── agent_service.py # Orchestration + metrics
```

---

## Prerequisites

- Python 3.11+
- AWS credentials with Bedrock access (Titan Embed v2 + Nova Pro)
- Documents in `data/raw/` (DOCX or PDF)

---

## Setup

```powershell
# 1. Activate virtual environment
.\agent_venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
# Edit .env with your AWS keys (already present)
```

---

## Quickstart

### Step 1 — Ingest Documents (run once)

```powershell
python leadership_agent/ingest.py
# Add --recreate to wipe and rebuild the vector store
python leadership_agent/ingest.py --recreate
```

### Step 2 — Use the CLI

```powershell
# Interactive mode
python leadership_agent/cli.py

# Single query
python leadership_agent/cli.py --query "What are the key risks in 2024?"
python leadership_agent/cli.py --query "How has revenue changed from 2023 to 2025?"
python leadership_agent/cli.py --query "Show revenue trend graph."
python leadership_agent/cli.py --query "Compare operating income over 3 years."
```

### Step 3 — Use the API

```powershell
# Start server
uvicorn leadership_agent.app:app --reload --port 8000

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key risks in 2024?"}'
```

**API Response schema:**
```json
{
  "answer": "...",
  "tools_used": ["retriever"],
  "sources": [{"company": "Microsoft", "year": "2024", "section": "Risk Factors", ...}],
  "image_path": null,
  "metrics": {"planner_latency_s": 0.001, "tool_latency_s": 1.2, "llm_latency_s": 2.1}
}
```

---

## How It Works

```
User Query
    ↓
Planner (keyword → LLM fallback routing)
    ├── "risks / strategy / leadership" → RetrieverTool (Qdrant semantic search)
    ├── "revenue / trend / growth"      → FinancialTool (pandas CSV analysis)
    └── "chart / graph / plot"          → PlotTool (matplotlib → static/trend.png)
    ↓
Tool Output (JSON)
    ↓
Synthesizer (Amazon Nova Pro via Converse API)
    ↓
Final Answer + Sources + Image Path
```

---

## Configuration

All settings in `leadership_agent/config.py`:

| Parameter | Value |
|-----------|-------|
| Embedding model | `amazon.titan-embed-text-v2:0` |
| Embedding dimension | 1024 |
| Chunk size | 1200 chars |
| Chunk overlap | 200 chars |
| Batch size | 32 |
| LLM | `amazon.nova-pro-v1:0` |
| Temperature | 0.2 |
| Qdrant collection | `leadership_reports` |
| Distance metric | Cosine |

---

## Logs & Observability

- **Console + rotating file**: `logs/agent.log` (10 MB × 5 backups)
- **Per-request metrics**: `logs/metrics.jsonl`
- **Generated charts**: `static/trend.png`
- **Extracted tables**: `data/structured/*.csv`

Logging covers: ingestion chunks, embedding batches, Qdrant ops, planner decisions, tool outputs, LLM latency, and API request timing.
