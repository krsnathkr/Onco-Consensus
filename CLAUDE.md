# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Onco-Consensus** is a Multi-Agent System (MAS) for breast cancer pathology diagnostic reasoning. Multiple AI agents — each backed by a different LLM — independently analyze pathology data, debate findings, and converge on a consensus diagnosis.

Deliverable: a fully functional Jupyter Notebook (`.ipynb`).

## Architecture

The system has three conceptual layers:

1. **Agent Layer** — Individual diagnostic agents, each wrapping a different LLM (Claude, Gemini, GPT, and at least one of Grok/Llama). Each agent receives the same patient/pathology input and returns a structured diagnosis with reasoning.

2. **Consensus Layer** — An orchestrator that collects agent outputs, runs a debate/deliberation loop (agents can challenge each other), and produces a final agreed-upon diagnosis with confidence score.

3. **Analytics Layer** — Visualization and reporting of individual agent opinions, disagreement patterns, and the consensus result.

## Required Integrations (at least 3 of these)

- Anthropic Claude (`anthropic` SDK)
- Google Gemini (`google-generativeai` SDK)
- OpenAI ChatGPT (`openai` SDK)
- xAI Grok (OpenAI-compatible API)
- Meta Llama (via LlamaIndex or `ollama`)

**Free-first approach (preferred):**
- **Google Gemini Flash** (`google-generativeai`) — free tier via [AI Studio](https://aistudio.google.com), use `gemini-2.0-flash`
- **Groq** (`groq` SDK, OpenAI-compatible) — free tier at [console.groq.com](https://console.groq.com), runs Llama 3 / Mixtral
- **Ollama** (local) — run `ollama pull llama3` for a fully offline, zero-cost agent

If a 4th agent is needed: **Claude Haiku** (`anthropic` SDK) is the cheapest paid option (~$0.25/M tokens).

## Development Setup

```bash
pip install google-generativeai groq anthropic python-dotenv jupyter
# for Ollama: brew install ollama && ollama pull llama3
jupyter notebook
```

API keys go in a `.env` file (never committed); load with `python-dotenv`.

## Key Design Constraints

- All agents must receive identical input; no agent sees another's answer before forming its own initial opinion.
- The consensus mechanism must be explicit and traceable — show which agents agreed/disagreed and how the final call was reached.
- Use structured outputs (JSON schemas or Pydantic models) for agent responses so the consensus layer can parse and compare them reliably.
