# AI Leadership Insight Agent — v1.0

> **Phase 1 (Stable): Narrative RAG Q&A over Microsoft 10-K reports (FY2023–FY2025)**

---

## Problem Understanding

Corporate 10-K filings contain hundreds of pages of dense financial narrative, risk disclosures, and MD&A commentary. Leadership teams and analysts need to ask focused questions—about risks, strategy shifts, or competitive outlook—without manually trawling through hundreds of pages.

This agent solves that by:
1. **Ingesting** all three Microsoft 10-K DOCX reports using Docling
2. **Embedding** the narrative chunks with Amazon Titan Embed v2 (1024-dim) and storing them locally in Qdrant
3. **Routing** queries through a two-stage planner (keyword → LLM fallback)
4. **Retrieving** the most relevant passages semantically and **synthesizing** a factual answer via Amazon Nova Pro

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION  (run once)                           │
│  data/raw/*.docx  →  Docling  →  Chunk+Tag  →  Titan Embed v2          │
│                                               →  Qdrant (local)        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT  (LangGraph)                              │
│                                                                         │
│  User Query                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  [ Planner ]  ── keyword match  ──►  "retriever"                        │
│      │         └─ LLM fallback                                          │
│      ▼                                                                  │
│  [ Tool Executor ]  ──►  RetrieverTool                                  │
│                            │  Titan Embed query → Qdrant top-5          │
│                            └──► scored chunks + metadata                │
│      ▼                                                                  │
│  [ Synthesizer ]  ──►  Amazon Nova Pro (Converse API)                   │
│                          │  Compose factual answer from chunks          │
│                          └──► Final answer + sources + metrics          │
│                                                                         │
│  CLI / FastAPI  ◄────────────────────────────────────────────────────── │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `leadership_agent/config.py` | Centralized settings, loads `.env` |
| Logging | `leadership_agent/logging_config.py` | Rotating file + console |
| Parser | `ingestion/pdf_parser.py` | Docling DOCX → chunks + table CSVs |
| Embedder | `embeddings/embedder.py` | Titan Embed v2 batchprocessor |
| Vector Store | `vectorstore/qdrant_store.py` | Local Qdrant, Cosine similarity |
| Retriever | `tools/retriever_tool.py` | Semantic search → top-5 scored chunks |
| Planner | `agent/planner.py` | Keyword routing + Nova Pro fallback |
| Controller | `agent/controller.py` | LangGraph StateGraph orchestrator |
| Service | `services/agent_service.py` | Unified `run()` + metrics JSONL |
| CLI | `leadership_agent/cli.py` | Interactive + single-query interface |
| API | `leadership_agent/app.py` | FastAPI `POST /query` endpoint |
| Ingestion | `leadership_agent/ingest.py` | Pipeline runner (parse→embed→store) |

---

## Assumptions

- **Documents:** Microsoft 10-K DOCX **or PDF** files (FY2023, FY2024, FY2025) in `data/raw/`
- **PDF type:** Digitally-born (SEC-filed) PDFs work out of the box. Scanned/image PDFs require `PDF_OCR_ENABLED=true` in `.env` (slower, requires `easyocr`/`tesseract`)
- **AWS credentials** configured in `.env` with Bedrock access to `us-east-1`
- **Models available:** `amazon.titan-embed-text-v2:0` and `amazon.nova-pro-v1:0`
- **Run order:** `ingest.py` must be run once before any agent query
- **Local execution**: Qdrant runs as a local embedded store (no Docker required)
- **Filename convention:** `<Company>_<Year>_<doctype>.(docx|pdf)` for metadata inference

---

## Setup

### Prerequisites
- Python 3.11+
- AWS account with Bedrock access in `us-east-1`
- Virtual environment (`agent_venv/`)

### Installation

```powershell
# 1. Activate virtual environment
.\agent_venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env   # then fill in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

### `.env` file

```env
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1

# Optional — set true only for scanned/image PDFs (requires OCR)
PDF_OCR_ENABLED=false
```

---

## How to Run

### Step 1 — Ingest Documents (one-time, ~10–15 min)

```powershell
python leadership_agent/ingest.py
```

Output:
```
✅ Ingestion complete!
   Documents: 3 | Chunks: ~3000 | Time: ~12 min
```

To rebuild from scratch:
```powershell
python leadership_agent/ingest.py --recreate
```

### Step 2a — CLI

```powershell
# Single query
python -m leadership_agent.cli --query "What are the key risks in 2024?"

# Single query + inline RAGAS evaluation (2 extra LLM judge calls)
python -m leadership_agent.cli --query "What are the key risks in 2024?" --eval

# Interactive mode
python -m leadership_agent.cli

# Interactive mode with RAGAS scoring after every answer
python -m leadership_agent.cli --eval
```

### Step 2b — FastAPI Server

```powershell
uvicorn leadership_agent.app:app --reload --port 8000
```

Then POST:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key risks Microsoft faces in FY2024?"}'
```

---

## Sample Queries

| Query | Type | What it tests |
|-------|------|---------------|
| `"What are the key risks Microsoft faces in 2024?"` | Narrative Q&A | Risk Factors section retrieval |
| `"What is Microsoft's cloud strategy?"` | Narrative Q&A | MD&A + Strategy section |
| `"How does Microsoft describe its AI investments?"` | Narrative Q&A | Cross-year AI narrative |
| `"What are the main competition risks?"` | Narrative Q&A | Competition section |
| `"What happened to revenue between 2023 and 2025?"` | Financial *(Phase 2)* | Redirects to narrative |

---

## Sample Output

```
────────────────────────────────────────────────

📝  ANSWER:
    Microsoft identifies several key risks in its FY2024 10-K filing.
    Cybersecurity threats, including nation-state actors, remain a top
    concern... [abbreviated]

🔧  TOOLS USED:  retriever

📚  SOURCES (5):
    [0.842] Microsoft 2024 — Risk Factors
    [0.831] Microsoft 2024 — Risk Factors
    [0.819] Microsoft 2023 — Risk Factors
    [0.801] Microsoft 2025 — Risk Factors
    [0.788] Microsoft 2024 — MD&A

⏱️   TIMING:  Planner: 0.00s | Tool: 1.24s | Llm: 3.17s | Total: 4.43s

────────────────────────────────────────────────
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Local Qdrant** | No cloud dependency; reproducible local execution; data stays private |
| **Amazon Titan Embed v2** | Native Bedrock integration; 1024-dim provides rich semantic space |
| **Amazon Nova Pro** | Cost-efficient LLM via Converse API; strong instruction-following |
| **Two-stage planner** | Keyword match = 0ms cost for obvious queries; LLM fallback for ambiguous cases |
| **Docling for parsing** | Handles complex DOCX structure, table extraction, and layout preservation |
| **Character-level chunking** (1200/200) | Balances context window vs. retrieval precision for long financial docs |
| **Section tagging** | Heuristic section detection (Risk Factors, MD&A, etc.) improves source attribution |
| **LangGraph** | Explicit node-edge graph makes the agent pipeline auditable and extensible |
| **Modular package structure** | Each component (ingestion, embedding, vectorstore, tools, agent) is independently testable |

---

## Observability

| Output | Location | Format |
|--------|----------|--------|
| Structured logs | `logs/agent.log` | Rotating (10MB × 5 files) |
| Per-request metrics | `logs/metrics.jsonl` | JSON lines: latency, tool, query |
| Extracted tables | `data/structured/` | CSV per table per document |
| API timing header | HTTP response | `X-Process-Time: 4.432` |

---

## Evaluation (RAGAS)

A lightweight RAGAS-compatible evaluation harness is included, using **Amazon Nova Pro as the LLM judge** — no OpenAI dependency.

### Metrics

| Metric | Method | LLM Cost |
|--------|--------|----------|
| **Faithfulness** | Is the answer grounded in retrieved context? | Nova Pro judge |
| **Answer Relevancy** | Does the answer address the question? | Nova Pro judge |
| **Context Recall** | Fraction of chunks with cosine score ≥ 0.70 | Heuristic (free) |

### Running the Eval

```powershell
# Run all 10 validation samples
python -m leadership_agent.eval.run_eval

# Quick smoke test (2 samples)
python -m leadership_agent.eval.run_eval --samples 2

# Custom output path
python -m leadership_agent.eval.run_eval --output logs/my_eval.jsonl
```

Outputs:
- **Console:** Formatted results table with per-query scores + aggregate averages
- **File:** `logs/eval_results.jsonl` — one JSON record per query

### Validation Query Set

10 NL questions in `leadership_agent/eval/validation_set.py` covering:
Key risks · Cloud strategy · AI investments · Competition · Revenue trends ·
Cybersecurity · Generative AI · Regulation · Gaming · ESG/Sustainability

---

## Project Structure

```
AI-Leadership-Insight-Agent/
├── data/
│   ├── raw/                        # 10-K DOCX source files (not committed)
│   └── structured/                 # Auto-generated table CSVs
├── leadership_agent/
│   ├── config.py                   # Centralized config + .env loader
│   ├── logging_config.py           # Structured logging setup
│   ├── ingest.py                   # Ingestion pipeline runner
│   ├── cli.py                      # CLI interface
│   ├── app.py                      # FastAPI application
│   ├── ingestion/pdf_parser.py     # Docling DOCX parser + chunker
│   ├── embeddings/embedder.py      # Titan Embed v2 batch embedder
│   ├── vectorstore/qdrant_store.py # Local Qdrant wrapper
│   ├── tools/
│   │   ├── retriever_tool.py       # ✅ Stable: semantic search
│   │   ├── financial_tool.py       # 🔜 Phase 2: CSV financial analysis
│   │   └── plot_tool.py            # 🔜 Phase 2: trend visualization
│   ├── agent/
│   │   ├── state.py                # LangGraph AgentState TypedDict
│   │   ├── planner.py              # Tool routing (keyword + LLM)
│   │   └── controller.py           # LangGraph StateGraph
│   └── services/agent_service.py   # Orchestration + metrics
├── logs/                           # Auto-created on first run
├── qdrant_storage/                 # Auto-created by Qdrant
├── static/                         # Auto-created (plot output)
├── requirements.txt
└── README.md
```

---

## Future Improvements (Phase 2)

| Feature | Status | Description |
|---------|--------|-------------|
| **FinancialTool** | 🔜 Planned | Pandas-based CSV analysis, YoY revenue/income growth |
| **PlotTool** | 🔜 Planned | matplotlib bar + trend charts saved to `static/` |
| **Multi-doc comparison** | 🔜 Planned | Cross-year semantic comparison with metadata filters |
| **Table embeddings** | 🔜 Planned | Embed table content in Qdrant for numeric RAG |
| **Evaluation harness** | 🔜 Planned | Automated RAGAS-style faithfulness + relevancy scoring |
| **Streaming API** | 🔜 Planned | SSE endpoint for real-time token streaming |
| **Web UI** | 🔜 Planned | Minimal React interface for interactive Q&A |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'leadership_agent'` | Activate venv: `.\agent_venv\Scripts\Activate.ps1` |
| `Empty results` from retriever | Run ingestion first: `python leadership_agent/ingest.py` |
| `ResourceNotFoundException` (Bedrock) | Check `.env` credentials and ensure `us-east-1` region |
| `msvcrt` error at shutdown | Benign Windows artifact — safe to ignore (suppressed in v1.0 via `atexit`) |
