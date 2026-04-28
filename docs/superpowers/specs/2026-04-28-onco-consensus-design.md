# Onco-Consensus: Design Spec
**Date:** 2026-04-28  
**Project:** CS505 — Multi-Agent Diagnostic Reasoning for Breast Cancer Pathology  
**Deliverable:** Fully functional Jupyter Notebook (`.ipynb`)

---

## 1. Overview

A Multi-Agent System (MAS) where three LLM-backed diagnostic agents independently analyze breast cancer biopsy data, engage in a structured multi-round debate, and converge on a final consensus diagnosis via a meta-synthesizer. The system is traceable: every agent opinion, round transition, and consensus rationale is captured in structured Pydantic models and visualized in an analytics section.

---

## 2. Dataset

**Source:** UCI Wisconsin Breast Cancer Diagnostic dataset, loaded via `sklearn.datasets.load_breast_cancer()`. No download or registration required.

**Size:** 569 samples with 30 FNA nuclear morphology features (mean, standard error, and worst value of: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension). Ground-truth labels: Malignant / Benign.

**Case selection:** 5 cases selected per run — a mix of malignant and benign — to keep token costs low while producing interesting debate dynamics.

**Narrative generation:** A `case_to_narrative(idx: int) -> str` function converts a row's numeric features into a clinical pathology paragraph (e.g., *"Fine needle aspirate biopsy of a breast mass. Nuclear morphology: mean radius 17.99 μm (worst 25.38)…"*). Ground-truth labels are hidden from agents during the debate.

**Pydantic model:**
```python
class PathologyCase(BaseModel):
    case_id: str
    narrative: str
    ground_truth: str   # hidden from agents until after debate
```

---

## 3. Agent Layer

### Base class

```python
class DiagnosticAgent(ABC):
    def analyze(
        self,
        case: PathologyCase,
        prior_opinions: list[DiagnosisOutput] | None
    ) -> DiagnosisOutput: ...
```

`prior_opinions=None` signals Round 0 (blind analysis). Populated in Rounds 1 and 2.

### Subclasses

| Class | SDK | Model |
|-------|-----|-------|
| `GeminiAgent` | `google-generativeai` | `gemini-2.0-flash` |
| `GroqAgent` | `groq` (OpenAI-compatible; base URL configurable via `GROQ_BASE_URL` env var for xAI Grok swap-in) | `llama3-70b-8192` |
| `OllamaAgent` | `requests` (HTTP to localhost:11434) | `llama3` |

All three receive **identical** system and user prompts in Round 0. No agent sees another's output before forming its initial position.

**System prompt (all agents):**  
*"You are an expert breast pathologist. Analyze the following biopsy report and return a structured JSON diagnosis."*

**User prompt (Round 0):**  
The pathology narrative only.

**User prompt (Rounds 1–2):**  
The pathology narrative plus the other two agents' `DiagnosisOutput` from the prior round, formatted as readable JSON.

### Structured output model

```python
class DiagnosisOutput(BaseModel):
    agent_name: str
    round: int
    diagnosis: Literal["Malignant", "Benign"]
    confidence: float           # 0.0–1.0
    key_findings: list[str]     # bullet points from biopsy report
    reasoning: str              # free-text explanation
    changed_opinion: bool       # True if diagnosis differs from prior round
```

---

## 4. Debate Orchestrator

`DebateOrchestrator` runs 3 rounds per case:

```
Round 0  →  all 3 agents analyze independently (prior_opinions=None)
Round 1  →  all 3 agents see the other two's Round 0 outputs
Round 2  →  all 3 agents see Round 1 outputs, lock in final position
```

All outputs are collected into:

```python
class DebateTranscript(BaseModel):
    case_id: str
    rounds: list[list[DiagnosisOutput]]   # rounds[round_idx][agent_idx]
    ground_truth: str                      # revealed after debate completes
```

---

## 5. Meta-Synthesizer

Gemini is reused in a "chief pathologist / tumor board chair" role. It receives the full `DebateTranscript` and returns:

```python
class ConsensusReport(BaseModel):
    case_id: str
    final_diagnosis: Literal["Malignant", "Benign"]
    confidence_score: float        # 0.0–1.0
    rationale: str                 # synthesis of agreed points
    dissent_notes: str | None      # minority agent position, if any
    correct: bool                  # compared against ground_truth
```

**System prompt:**  
*"You are the chief pathologist chairing a tumor board. Synthesize the following multi-round diagnostic debate into a final consensus diagnosis with rationale."*

---

## 6. Analytics Layer

Four inline `matplotlib`/`seaborn` visualizations:

1. **Opinion evolution chart** — per case, per agent: diagnosis at each round (color-coded Malignant/Benign). Shows opinion flips.
2. **Agreement heatmap** — agents × rounds matrix per case. Shows when consensus formed.
3. **Confidence bar chart** — each agent's confidence at Round 0 vs Round 2. Shows whether debate increased or decreased certainty.
4. **Consensus summary card** — final diagnosis, confidence score, correct/incorrect vs ground truth, dissent notes.

---

## 7. Notebook Structure

```
§0  Setup & Installs       pip install google-generativeai groq anthropic python-dotenv; imports; load .env
§1  Data Layer             load_breast_cancer(), case_to_narrative(), PathologyCase model, select 5 cases
§2  Pydantic Models        DiagnosisOutput, DebateTranscript, ConsensusReport
§3  Agent Definitions      DiagnosticAgent base class + GeminiAgent, GroqAgent, OllamaAgent
§4  Orchestrator           DebateOrchestrator (3 rounds) + MetaSynthesizer (Gemini judge)
§5  Run Pipeline           instantiate agents, run debate for each of 5 cases, collect transcripts
§6  Analytics              4 visualizations
§7  Results Summary        markdown table: case × agent opinions × consensus × correct?
```

---

## 8. Configuration & Security

- API keys stored in `.env` (never committed): `GEMINI_API_KEY`, `GROQ_API_KEY`, optionally `GROQ_BASE_URL` (set to xAI endpoint to swap GroqAgent to xAI Grok without code changes)
- Loaded via `python-dotenv` in §0
- Ollama runs locally on `http://localhost:11434` — no key required
- `.env` listed in `.gitignore`

---

## 9. Key Design Constraints (from assignment)

- All agents receive identical input in Round 0; no agent sees another's answer before forming its initial opinion.
- The consensus mechanism is explicit and traceable — `DebateTranscript` records every round.
- Structured outputs (Pydantic models) throughout so the consensus layer can parse and compare reliably.
- At least 3 LLM platforms integrated: Google Gemini, Groq (Meta Llama), Ollama (Meta Llama local). ✓

---

## 10. Dependencies

```
google-generativeai
groq
python-dotenv
pydantic
scikit-learn
matplotlib
seaborn
jupyter
requests   # for Ollama HTTP calls
```
