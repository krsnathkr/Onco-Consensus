# Onco-Consensus

**CS505 — Spring 2026**

A Multi-Agent System (MAS) for breast cancer pathology diagnostic reasoning. Three LLM agents independently analyze FNA biopsy data, debate over 3 rounds, and a Gemini meta-synthesizer produces a final consensus diagnosis.

---

## How It Works

1. **Data** — UCI Wisconsin Breast Cancer dataset (sklearn). Each case's 30 nuclear morphology features are converted into a clinical pathology narrative.
2. **Round 0 (blind)** — All three agents independently analyze the same biopsy report with no knowledge of each other's opinions.
3. **Rounds 1–2 (debate)** — Each agent sees the other agents' prior diagnoses and reasoning, then updates its own assessment.
4. **Consensus** — A Gemini meta-synthesizer acts as "chief pathologist," synthesizing the full debate transcript into a final diagnosis with confidence score and dissent notes.

## Agents

| Agent | Model | Provider |
|---|---|---|
| GeminiAgent | `google/gemini-2.0-flash-001` | OpenRouter |
| GroqAgent | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| OllamaAgent | `llama3:8b` | Local Ollama |
| MetaSynthesizer | `google/gemini-2.0-flash-001` | OpenRouter |

## Project Structure

```
onco_consensus.ipynb   # Main deliverable — run this
src/
  agents.py            # GeminiAgent, GroqAgent, OllamaAgent
  orchestrator.py      # DebateOrchestrator, MetaSynthesizer
  models.py            # Pydantic models for all I/O
  data_layer.py        # UCI dataset loading + narrative generation
  analytics.py         # 4 matplotlib/seaborn visualizations
tests/                 # 25 pytest unit tests (all mocked, no API calls)
```

## Setup

**1. Install dependencies**
```bash
pip install openai python-dotenv pydantic scikit-learn matplotlib seaborn pandas requests jupyter
```

**2. Install and start Ollama**
```bash
brew install ollama
ollama pull llama3:8b
ollama serve   # runs at http://localhost:11434
```

**3. Create `.env`**
```
OPENROUTER_API_KEY=sk-or-v1-...
GEMINI_MODEL=google/gemini-2.0-flash-001
GROQ_MODEL=meta-llama/llama-3.3-70b-instruct
```

**4. Run the notebook**
```bash
jupyter notebook onco_consensus.ipynb
```
Run all cells from top to bottom (Kernel → Restart & Run All).

## Running Tests

```bash
pytest tests/ -v
```

All 25 tests run without any API calls (fully mocked).

## Output

The notebook produces:
- Per-case debate transcript showing each agent's diagnosis and confidence per round
- Consensus report with final diagnosis, confidence score, and dissent notes
- 4 analytics plots: opinion evolution, agreement heatmap, confidence comparison, consensus summary cards
- Results table showing all agents across all rounds
