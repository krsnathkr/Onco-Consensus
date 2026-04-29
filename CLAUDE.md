# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Onco-Consensus** is a Multi-Agent System (MAS) for breast cancer pathology diagnostic reasoning. Three AI agents independently analyze FNA biopsy data, debate over 3 rounds, and a meta-synthesizer produces a final consensus diagnosis.

Deliverable: a fully functional Jupyter Notebook (`onco_consensus.ipynb`).

## Architecture

1. **Agent Layer** (`src/agents.py`) — Three agents, all using `openai` SDK via OpenRouter:
   - `GeminiAgent` → `google/gemini-2.0-flash-001`
   - `GroqAgent` → `meta-llama/llama-3.3-70b-instruct`
   - `OllamaAgent` → local Ollama `llama3:8b` (uses `requests` directly)

2. **Consensus Layer** (`src/orchestrator.py`) — `DebateOrchestrator` runs N rounds; `MetaSynthesizer` uses Gemini as chief pathologist to produce a `ConsensusReport`.

3. **Analytics Layer** (`src/analytics.py`) — Four matplotlib/seaborn visualizations.

4. **Data Layer** (`src/data_layer.py`) — UCI Wisconsin FNA dataset via sklearn; `case_to_narrative()` converts features to clinical text.

5. **Models** (`src/models.py`) — Pydantic v2: `PathologyCase`, `DiagnosisOutput`, `DebateTranscript`, `ConsensusReport`.

## API / Credentials

All cloud agents route through **OpenRouter** (`https://openrouter.ai/api/v1`) using the `openai` SDK. One key covers everything.

`.env` (never committed):
```
OPENROUTER_API_KEY=sk-or-v1-...
GEMINI_MODEL=google/gemini-2.0-flash-001
GROQ_MODEL=meta-llama/llama-3.3-70b-instruct
```

## Development Setup

```bash
pip install openai python-dotenv pydantic scikit-learn matplotlib seaborn pandas requests jupyter
# for Ollama: brew install ollama && ollama pull llama3:8b
jupyter notebook
```

## Running Tests

```bash
pytest tests/ -v   # 25 tests across 4 files
```

## Key Design Constraints

- All agents receive identical input in Round 0; no agent sees another's answer before forming its initial opinion.
- The consensus mechanism must be explicit and traceable.
- Use Pydantic models for all agent I/O so the consensus layer can parse reliably.
- Never commit `.env` or API keys.
