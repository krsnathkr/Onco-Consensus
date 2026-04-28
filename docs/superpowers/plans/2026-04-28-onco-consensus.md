# Onco-Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Jupyter Notebook MAS where three LLM agents (Gemini Flash, Groq/Llama3, Ollama/Llama3) independently analyze FNA breast biopsy cases, debate over 3 rounds, and a Gemini meta-synthesizer produces a final consensus diagnosis.

**Architecture:** Core logic lives in `src/` Python modules (testable with pytest); `onco_consensus.ipynb` imports from `src/` and orchestrates the full pipeline end-to-end. Agents accept injected dependencies so unit tests can mock API calls without hitting real endpoints.

**Tech Stack:** `google-generativeai`, `groq`, `pydantic`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `requests`, `python-dotenv`, `pytest`

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/models.py` | `PathologyCase`, `DiagnosisOutput`, `DebateTranscript`, `ConsensusReport` |
| `src/data_layer.py` | `case_to_narrative()`, `select_cases()` |
| `src/agents.py` | `DiagnosticAgent` base, `GeminiAgent`, `GroqAgent`, `OllamaAgent` |
| `src/orchestrator.py` | `DebateOrchestrator`, `MetaSynthesizer` |
| `src/analytics.py` | `plot_opinion_evolution`, `plot_agreement_heatmap`, `plot_confidence_comparison`, `plot_consensus_summary` |
| `tests/test_models.py` | Pydantic model validation tests |
| `tests/test_data_layer.py` | Narrative generation + case selection tests |
| `tests/test_agents.py` | Agent tests with mocked API clients |
| `tests/test_orchestrator.py` | Orchestrator + synthesizer tests with mocked agents |
| `onco_consensus.ipynb` | Notebook deliverable — imports src/, runs full pipeline |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 2: Create `.env.example`**

```
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
# Optional: set to xAI base URL to swap GroqAgent to xAI Grok
# GROQ_BASE_URL=https://api.x.ai/v1
```

- [ ] **Step 3: Create `requirements.txt`**

```
google-generativeai>=0.8.0
groq>=0.11.0
python-dotenv>=1.0.0
pydantic>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.0.0
requests>=2.31.0
jupyter>=1.0.0
nbformat>=5.9.0
pytest>=8.0.0
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 5: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files are empty. Run:
```bash
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore .env.example requirements.txt pyproject.toml src/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `tests/test_models.py`
- Create: `src/models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
from pydantic import ValidationError
import pytest
from src.models import PathologyCase, DiagnosisOutput, DebateTranscript, ConsensusReport


def test_pathology_case_valid():
    case = PathologyCase(case_id="CASE-001", narrative="Test narrative.", ground_truth="Malignant")
    assert case.case_id == "CASE-001"
    assert case.ground_truth == "Malignant"


def test_diagnosis_output_valid():
    out = DiagnosisOutput(
        agent_name="GeminiAgent",
        round=0,
        diagnosis="Malignant",
        confidence=0.85,
        key_findings=["high concavity", "large radius"],
        reasoning="Nuclear features are concerning.",
        changed_opinion=False,
    )
    assert out.diagnosis == "Malignant"
    assert out.confidence == 0.85
    assert out.changed_opinion is False


def test_diagnosis_output_rejects_invalid_diagnosis():
    with pytest.raises(ValidationError):
        DiagnosisOutput(
            agent_name="GeminiAgent", round=0, diagnosis="Unknown",
            confidence=0.5, key_findings=[], reasoning="", changed_opinion=False,
        )


def test_diagnosis_output_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        DiagnosisOutput(
            agent_name="GeminiAgent", round=0, diagnosis="Malignant",
            confidence=1.5, key_findings=[], reasoning="", changed_opinion=False,
        )


def test_debate_transcript_valid():
    out = DiagnosisOutput(
        agent_name="GeminiAgent", round=0, diagnosis="Benign",
        confidence=0.7, key_findings=[], reasoning="Looks benign", changed_opinion=False,
    )
    t = DebateTranscript(case_id="CASE-001", rounds=[[out]], ground_truth="Benign")
    assert t.case_id == "CASE-001"
    assert len(t.rounds) == 1
    assert len(t.rounds[0]) == 1


def test_consensus_report_correct_flag():
    r = ConsensusReport(
        case_id="CASE-001",
        final_diagnosis="Malignant",
        confidence_score=0.9,
        rationale="Strong agreement.",
        dissent_notes=None,
        correct=True,
    )
    assert r.correct is True
    assert r.dissent_notes is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Implement `src/models.py`**

```python
from typing import Literal
from pydantic import BaseModel, field_validator


class PathologyCase(BaseModel):
    case_id: str
    narrative: str
    ground_truth: str


class DiagnosisOutput(BaseModel):
    agent_name: str
    round: int
    diagnosis: Literal["Malignant", "Benign"]
    confidence: float
    key_findings: list[str]
    reasoning: str
    changed_opinion: bool

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be 0.0–1.0, got {v}")
        return v


class DebateTranscript(BaseModel):
    case_id: str
    rounds: list[list[DiagnosisOutput]]
    ground_truth: str


class ConsensusReport(BaseModel):
    case_id: str
    final_diagnosis: Literal["Malignant", "Benign"]
    confidence_score: float
    rationale: str
    dissent_notes: str | None
    correct: bool
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add Pydantic models"
```

---

## Task 3: Data Layer

**Files:**
- Create: `tests/test_data_layer.py`
- Create: `src/data_layer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_layer.py`:

```python
import pytest
from src.data_layer import case_to_narrative, select_cases
from src.models import PathologyCase


def test_case_to_narrative_returns_string():
    narrative = case_to_narrative(0)
    assert isinstance(narrative, str)
    assert len(narrative) > 100
    assert "μm" in narrative
    assert "biopsy" in narrative.lower()


def test_case_to_narrative_contains_key_features():
    narrative = case_to_narrative(0)
    assert "radius" in narrative
    assert "texture" in narrative
    assert "concavity" in narrative


def test_select_cases_returns_correct_count():
    cases = select_cases(n_malignant=2, n_benign=1)
    assert len(cases) == 3


def test_select_cases_returns_pathology_cases():
    cases = select_cases(n_malignant=1, n_benign=1)
    for case in cases:
        assert isinstance(case, PathologyCase)
        assert case.case_id.startswith("CASE-")
        assert case.ground_truth in ("Malignant", "Benign")
        assert len(case.narrative) > 50


def test_select_cases_has_correct_ground_truth_mix():
    cases = select_cases(n_malignant=3, n_benign=2)
    malignant = [c for c in cases if c.ground_truth == "Malignant"]
    benign = [c for c in cases if c.ground_truth == "Benign"]
    assert len(malignant) == 3
    assert len(benign) == 2


def test_select_cases_unique_ids():
    cases = select_cases(n_malignant=3, n_benign=2)
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data_layer.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data_layer'`

- [ ] **Step 3: Implement `src/data_layer.py`**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer

from src.models import PathologyCase


def case_to_narrative(idx: int) -> str:
    bc = load_breast_cancer()
    fd = dict(zip(bc.feature_names, bc.data[idx]))
    return (
        f"Fine needle aspirate (FNA) biopsy of a breast mass. "
        f"Nuclear morphology: mean radius {fd['mean radius']:.2f} μm "
        f"(worst {fd['worst radius']:.2f} μm), "
        f"mean texture {fd['mean texture']:.2f} (worst {fd['worst texture']:.2f}), "
        f"mean perimeter {fd['mean perimeter']:.2f} μm "
        f"(worst {fd['worst perimeter']:.2f} μm), "
        f"mean area {fd['mean area']:.2f} μm² (worst {fd['worst area']:.2f} μm²). "
        f"Surface characteristics: smoothness {fd['mean smoothness']:.4f} "
        f"(worst {fd['worst smoothness']:.4f}), "
        f"compactness {fd['mean compactness']:.4f} (worst {fd['worst compactness']:.4f}). "
        f"Structural features: concavity {fd['mean concavity']:.4f} "
        f"(worst {fd['worst concavity']:.4f}), "
        f"concave points {fd['mean concave points']:.4f} "
        f"(worst {fd['worst concave points']:.4f}). "
        f"Nuclear symmetry: {fd['mean symmetry']:.4f} (worst {fd['worst symmetry']:.4f}). "
        f"Fractal dimension: {fd['mean fractal dimension']:.4f} "
        f"(worst {fd['worst fractal dimension']:.4f})."
    )


def select_cases(n_malignant: int = 3, n_benign: int = 2) -> list[PathologyCase]:
    bc = load_breast_cancer()
    malignant_idx = np.where(bc.target == 0)[0][:n_malignant]
    benign_idx = np.where(bc.target == 1)[0][:n_benign]

    cases = []
    for idx in list(malignant_idx) + list(benign_idx):
        gt = "Malignant" if bc.target[idx] == 0 else "Benign"
        cases.append(PathologyCase(
            case_id=f"CASE-{idx:03d}",
            narrative=case_to_narrative(idx),
            ground_truth=gt,
        ))
    return cases
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data_layer.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/data_layer.py tests/test_data_layer.py
git commit -m "feat: add data layer with FNA narrative generation"
```

---

## Task 4: Agent Layer

**Files:**
- Create: `tests/test_agents.py`
- Create: `src/agents.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agents.py`:

```python
from unittest.mock import MagicMock
import pytest
from src.models import DiagnosisOutput, PathologyCase
from src.agents import GeminiAgent, GroqAgent, OllamaAgent


MOCK_CASE = PathologyCase(
    case_id="TEST-001",
    narrative="Fine needle aspirate biopsy. Nuclear radius 17.99 μm, high concavity.",
    ground_truth="Malignant",
)

R0_MALIGNANT_JSON = (
    '{"diagnosis": "Malignant", "confidence": 0.85, '
    '"key_findings": ["high concavity"], "reasoning": "Concerning morphology"}'
)
R0_BENIGN_JSON = (
    '{"diagnosis": "Benign", "confidence": 0.72, '
    '"key_findings": ["low radius"], "reasoning": "Benign features"}'
)


# ── GeminiAgent ──────────────────────────────────────────────────────────────

def _mock_gemini(response_text: str) -> GeminiAgent:
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text=response_text)
    return GeminiAgent(model=mock_model)


def test_gemini_round0_returns_diagnosis_output():
    agent = _mock_gemini(R0_MALIGNANT_JSON)
    out = agent.analyze(MOCK_CASE, prior_opinions=None)
    assert isinstance(out, DiagnosisOutput)
    assert out.agent_name == "GeminiAgent"
    assert out.round == 0
    assert out.diagnosis == "Malignant"
    assert out.confidence == 0.85
    assert out.changed_opinion is False


def test_gemini_round1_increments_round_number():
    prior = [
        DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Malignant",
                        confidence=0.85, key_findings=[], reasoning="r0", changed_opinion=False),
        DiagnosisOutput(agent_name="GroqAgent", round=0, diagnosis="Benign",
                        confidence=0.7, key_findings=[], reasoning="r0", changed_opinion=False),
        DiagnosisOutput(agent_name="OllamaAgent", round=0, diagnosis="Malignant",
                        confidence=0.6, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    agent = _mock_gemini(R0_MALIGNANT_JSON)
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.round == 1


def test_gemini_round1_detects_unchanged_opinion():
    prior = [
        DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Malignant",
                        confidence=0.85, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    agent = _mock_gemini(R0_MALIGNANT_JSON)
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.changed_opinion is False


def test_gemini_round1_detects_changed_opinion():
    prior = [
        DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Benign",
                        confidence=0.6, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    # R0 was Benign, now returning Malignant → changed
    agent = _mock_gemini(R0_MALIGNANT_JSON)
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.changed_opinion is True


# ── GroqAgent ─────────────────────────────────────────────────────────────────

def _mock_groq(response_text: str) -> GroqAgent:
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = response_text
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return GroqAgent(client=mock_client)


def test_groq_round0_returns_diagnosis_output():
    agent = _mock_groq(R0_BENIGN_JSON)
    out = agent.analyze(MOCK_CASE, prior_opinions=None)
    assert isinstance(out, DiagnosisOutput)
    assert out.agent_name == "GroqAgent"
    assert out.round == 0
    assert out.diagnosis == "Benign"
    assert out.confidence == 0.72


def test_groq_round1_increments_round():
    prior = [
        DiagnosisOutput(agent_name="GroqAgent", round=0, diagnosis="Benign",
                        confidence=0.7, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    agent = _mock_groq(R0_BENIGN_JSON)
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.round == 1


# ── OllamaAgent ───────────────────────────────────────────────────────────────

def _mock_ollama(response_text: str) -> tuple[OllamaAgent, MagicMock]:
    from unittest.mock import patch
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": response_text}}
    mock_resp.raise_for_status = MagicMock()
    return OllamaAgent(), mock_resp


def test_ollama_round0_returns_diagnosis_output():
    from unittest.mock import patch
    agent = OllamaAgent()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": R0_MALIGNANT_JSON}}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.agents.requests.post", return_value=mock_resp):
        out = agent.analyze(MOCK_CASE, prior_opinions=None)

    assert isinstance(out, DiagnosisOutput)
    assert out.agent_name == "OllamaAgent"
    assert out.round == 0
    assert out.diagnosis == "Malignant"


def test_ollama_round1_increments_round():
    from unittest.mock import patch
    prior = [
        DiagnosisOutput(agent_name="OllamaAgent", round=0, diagnosis="Malignant",
                        confidence=0.8, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    agent = OllamaAgent()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": R0_MALIGNANT_JSON}}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.agents.requests.post", return_value=mock_resp):
        out = agent.analyze(MOCK_CASE, prior_opinions=prior)

    assert out.round == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agents.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.agents'`

- [ ] **Step 3: Implement `src/agents.py`**

```python
import json
import os
import re
from abc import ABC, abstractmethod

import requests

from src.models import DiagnosisOutput, PathologyCase

SYSTEM_PROMPT = (
    "You are an expert breast pathologist. Analyze the following biopsy report "
    "and return a structured JSON diagnosis."
)
JSON_SCHEMA_HINT = (
    "\n\nReturn ONLY valid JSON with no additional text, matching exactly this schema:\n"
    '{"diagnosis": "Malignant" or "Benign", "confidence": 0.0 to 1.0, '
    '"key_findings": ["finding1", "finding2"], "reasoning": "your explanation"}'
)


def _build_user_prompt(case: PathologyCase, others: list[DiagnosisOutput]) -> str:
    prompt = f"Biopsy Report:\n{case.narrative}"
    if others:
        prompt += "\n\nOther pathologists' opinions from the previous round:\n"
        for op in others:
            prompt += (
                f"\n{op.agent_name}: {op.diagnosis} (confidence: {op.confidence:.2f})\n"
                f"Reasoning: {op.reasoning}\n"
            )
        prompt += "\nConsidering these opinions, provide your updated diagnosis."
    prompt += JSON_SCHEMA_HINT
    return prompt


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw[:300]}")
    return json.loads(match.group())


def _make_output(
    agent_name: str,
    data: dict,
    round_num: int,
    self_prior: DiagnosisOutput | None,
) -> DiagnosisOutput:
    return DiagnosisOutput(
        agent_name=agent_name,
        round=round_num,
        diagnosis=data["diagnosis"],
        confidence=float(data["confidence"]),
        key_findings=data.get("key_findings", []),
        reasoning=data.get("reasoning", ""),
        changed_opinion=(self_prior.diagnosis != data["diagnosis"]) if self_prior else False,
    )


class DiagnosticAgent(ABC):
    name: str

    @abstractmethod
    def analyze(
        self,
        case: PathologyCase,
        prior_opinions: list[DiagnosisOutput] | None,
    ) -> DiagnosisOutput: ...


def _split_prior(name: str, prior_opinions: list[DiagnosisOutput] | None):
    if not prior_opinions:
        return [], None, 0
    others = [op for op in prior_opinions if op.agent_name != name]
    self_prior = next((op for op in prior_opinions if op.agent_name == name), None)
    round_num = prior_opinions[0].round + 1
    return others, self_prior, round_num


class GeminiAgent(DiagnosticAgent):
    name = "GeminiAgent"

    def __init__(self, model=None):
        if model is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-2.0-flash")
        self.model = model

    def analyze(
        self, case: PathologyCase, prior_opinions: list[DiagnosisOutput] | None
    ) -> DiagnosisOutput:
        others, self_prior, round_num = _split_prior(self.name, prior_opinions)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(case, others)}"
        response = self.model.generate_content(full_prompt)
        return _make_output(self.name, _parse_json(response.text), round_num, self_prior)


class GroqAgent(DiagnosticAgent):
    name = "GroqAgent"

    def __init__(self, client=None):
        if client is None:
            from groq import Groq
            kwargs = {}
            if base_url := os.environ.get("GROQ_BASE_URL"):
                kwargs["base_url"] = base_url
            client = Groq(api_key=os.environ["GROQ_API_KEY"], **kwargs)
        self.client = client
        self.model_id = "llama3-70b-8192"

    def analyze(
        self, case: PathologyCase, prior_opinions: list[DiagnosisOutput] | None
    ) -> DiagnosisOutput:
        others, self_prior, round_num = _split_prior(self.name, prior_opinions)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(case, others)},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content
        return _make_output(self.name, _parse_json(raw), round_num, self_prior)


class OllamaAgent(DiagnosticAgent):
    name = "OllamaAgent"

    def __init__(self, base_url: str = "http://localhost:11434", model_id: str = "llama3"):
        self.base_url = base_url
        self.model_id = model_id

    def analyze(
        self, case: PathologyCase, prior_opinions: list[DiagnosisOutput] | None
    ) -> DiagnosisOutput:
        others, self_prior, round_num = _split_prior(self.name, prior_opinions)
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(case, others)},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        return _make_output(self.name, _parse_json(raw), round_num, self_prior)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agents.py -v
```
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents.py tests/test_agents.py
git commit -m "feat: add agent layer with GeminiAgent, GroqAgent, OllamaAgent"
```

---

## Task 5: Orchestrator + MetaSynthesizer

**Files:**
- Create: `tests/test_orchestrator.py`
- Create: `src/orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator.py`:

```python
from unittest.mock import MagicMock
import pytest
from src.models import ConsensusReport, DebateTranscript, DiagnosisOutput, PathologyCase
from src.orchestrator import DebateOrchestrator, MetaSynthesizer


MOCK_CASE = PathologyCase(
    case_id="TEST-001", narrative="Test narrative.", ground_truth="Malignant"
)


def _make_mock_agent(name: str, diagnoses: list[str]) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    call_count = {"n": 0}

    def analyze(case, prior_opinions):
        idx = call_count["n"]
        call_count["n"] += 1
        round_num = 0 if not prior_opinions else (prior_opinions[0].round + 1)
        return DiagnosisOutput(
            agent_name=name,
            round=round_num,
            diagnosis=diagnoses[idx % len(diagnoses)],
            confidence=0.8,
            key_findings=[],
            reasoning="mock",
            changed_opinion=False,
        )

    agent.analyze = analyze
    return agent


def test_orchestrator_runs_three_rounds():
    agents = [
        _make_mock_agent("A1", ["Malignant"]),
        _make_mock_agent("A2", ["Benign"]),
        _make_mock_agent("A3", ["Malignant"]),
    ]
    orch = DebateOrchestrator(agents)
    transcript = orch.run(MOCK_CASE, num_rounds=3)

    assert isinstance(transcript, DebateTranscript)
    assert len(transcript.rounds) == 3
    assert len(transcript.rounds[0]) == 3
    assert transcript.case_id == "TEST-001"
    assert transcript.ground_truth == "Malignant"


def test_orchestrator_round0_receives_none_prior():
    received = []

    def analyze(case, prior_opinions):
        received.append(prior_opinions)
        return DiagnosisOutput(
            agent_name="A1", round=0 if prior_opinions is None else 1,
            diagnosis="Malignant", confidence=0.8,
            key_findings=[], reasoning="", changed_opinion=False,
        )

    agent = MagicMock()
    agent.name = "A1"
    agent.analyze = analyze

    orch = DebateOrchestrator([agent])
    orch.run(MOCK_CASE, num_rounds=2)

    assert received[0] is None
    assert received[1] is not None


def test_orchestrator_passes_previous_round_to_next():
    calls = []

    def analyze(case, prior_opinions):
        calls.append(prior_opinions)
        round_num = 0 if not prior_opinions else prior_opinions[0].round + 1
        return DiagnosisOutput(
            agent_name="A1", round=round_num, diagnosis="Malignant",
            confidence=0.8, key_findings=[], reasoning="", changed_opinion=False,
        )

    agent = MagicMock()
    agent.name = "A1"
    agent.analyze = analyze

    orch = DebateOrchestrator([agent])
    orch.run(MOCK_CASE, num_rounds=2)

    assert calls[1] is not None
    assert calls[1][0].agent_name == "A1"
    assert calls[1][0].round == 0


def test_meta_synthesizer_returns_consensus_report():
    transcript = DebateTranscript(
        case_id="TEST-001",
        rounds=[[
            DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Malignant",
                            confidence=0.9, key_findings=[], reasoning="Concerning", changed_opinion=False),
            DiagnosisOutput(agent_name="GroqAgent", round=0, diagnosis="Malignant",
                            confidence=0.8, key_findings=[], reasoning="Malignant", changed_opinion=False),
            DiagnosisOutput(agent_name="OllamaAgent", round=0, diagnosis="Benign",
                            confidence=0.6, key_findings=[], reasoning="Benign", changed_opinion=False),
        ]],
        ground_truth="Malignant",
    )
    mock_response = MagicMock()
    mock_response.text = (
        '{"final_diagnosis": "Malignant", "confidence_score": 0.87, '
        '"rationale": "2 of 3 agents agree on malignancy.", '
        '"dissent_notes": "OllamaAgent disagreed, citing benign features."}'
    )
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    synth = MetaSynthesizer(model=mock_model)
    report = synth.synthesize(transcript)

    assert isinstance(report, ConsensusReport)
    assert report.case_id == "TEST-001"
    assert report.final_diagnosis == "Malignant"
    assert report.confidence_score == 0.87
    assert report.correct is True
    assert "OllamaAgent" in report.dissent_notes


def test_meta_synthesizer_correct_flag_false_when_wrong():
    transcript = DebateTranscript(
        case_id="TEST-002",
        rounds=[[
            DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Benign",
                            confidence=0.8, key_findings=[], reasoning="Benign", changed_opinion=False),
        ]],
        ground_truth="Malignant",
    )
    mock_response = MagicMock()
    mock_response.text = (
        '{"final_diagnosis": "Benign", "confidence_score": 0.8, '
        '"rationale": "Agents agree benign.", "dissent_notes": null}'
    )
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    synth = MetaSynthesizer(model=mock_model)
    report = synth.synthesize(transcript)

    assert report.correct is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_orchestrator.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.orchestrator'`

- [ ] **Step 3: Implement `src/orchestrator.py`**

```python
import json
import os
import re

from src.agents import DiagnosticAgent
from src.models import ConsensusReport, DebateTranscript, DiagnosisOutput, PathologyCase

SYNTHESIZER_SYSTEM = (
    "You are the chief pathologist chairing a tumor board. "
    "Synthesize the following multi-round diagnostic debate into a final consensus diagnosis with rationale."
)
SYNTHESIZER_JSON_HINT = (
    "\n\nReturn ONLY valid JSON:\n"
    '{"final_diagnosis": "Malignant" or "Benign", "confidence_score": 0.0 to 1.0, '
    '"rationale": "synthesis of agreed points", "dissent_notes": "minority view or null"}'
)


class DebateOrchestrator:
    def __init__(self, agents: list[DiagnosticAgent]):
        self.agents = agents

    def run(self, case: PathologyCase, num_rounds: int = 3) -> DebateTranscript:
        rounds: list[list[DiagnosisOutput]] = []
        prior_outputs: list[DiagnosisOutput] | None = None

        for _ in range(num_rounds):
            round_outputs = [agent.analyze(case, prior_outputs) for agent in self.agents]
            rounds.append(round_outputs)
            prior_outputs = round_outputs

        return DebateTranscript(
            case_id=case.case_id,
            rounds=rounds,
            ground_truth=case.ground_truth,
        )


class MetaSynthesizer:
    def __init__(self, model=None):
        if model is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-2.0-flash")
        self.model = model

    def synthesize(self, transcript: DebateTranscript) -> ConsensusReport:
        debate_text = f"Case ID: {transcript.case_id}\n\n"
        for r_idx, round_ops in enumerate(transcript.rounds):
            debate_text += f"--- Round {r_idx} ---\n"
            for op in round_ops:
                debate_text += (
                    f"{op.agent_name}: {op.diagnosis} (confidence: {op.confidence:.2f})\n"
                    f"  Key findings: {', '.join(op.key_findings) or 'none listed'}\n"
                    f"  Reasoning: {op.reasoning}\n\n"
                )

        full_prompt = f"{SYNTHESIZER_SYSTEM}\n\n{debate_text}{SYNTHESIZER_JSON_HINT}"
        response = self.model.generate_content(full_prompt)

        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON in synthesizer response: {response.text[:300]}")
        data = json.loads(match.group())

        final_dx = data["final_diagnosis"]
        return ConsensusReport(
            case_id=transcript.case_id,
            final_diagnosis=final_dx,
            confidence_score=float(data["confidence_score"]),
            rationale=data["rationale"],
            dissent_notes=data.get("dissent_notes"),
            correct=(final_dx == transcript.ground_truth),
        )
```

- [ ] **Step 4: Run all tests**

```bash
pytest -v
```
Expected: `22 passed` (all test files)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add DebateOrchestrator and MetaSynthesizer"
```

---

## Task 6: Analytics

**Files:**
- Create: `src/analytics.py`

No pytest unit tests for visualization — verified visually when the notebook runs. Implementation only.

- [ ] **Step 1: Implement `src/analytics.py`**

```python
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.models import ConsensusReport, DebateTranscript

_COLOR = {"Malignant": "#e74c3c", "Benign": "#2ecc71"}
_BG = {"Malignant": "#fadbd8", "Benign": "#d5f5e3"}


def plot_opinion_evolution(transcripts: list[DebateTranscript]) -> None:
    n_cases = len(transcripts)
    n_rounds = len(transcripts[0].rounds)
    agent_names = [op.agent_name for op in transcripts[0].rounds[0]]

    fig, axes = plt.subplots(1, n_cases, figsize=(4 * n_cases, 4), sharey=True)
    if n_cases == 1:
        axes = [axes]

    for ax, t in zip(axes, transcripts):
        for a_idx, aname in enumerate(agent_names):
            for r_idx in range(n_rounds):
                op = t.rounds[r_idx][a_idx]
                ax.scatter(r_idx, a_idx, color=_COLOR[op.diagnosis], s=260, zorder=5)
                ax.text(r_idx, a_idx + 0.2, f"{op.confidence:.2f}", ha="center", fontsize=8)
                if op.changed_opinion:
                    ax.scatter(r_idx, a_idx, color="gold", s=500, marker="*", zorder=4)
            ax.plot(range(n_rounds), [a_idx] * n_rounds, "k--", alpha=0.15)

        ax.set_title(
            f"{t.case_id}\n(GT: {t.ground_truth})",
            fontsize=10, color=_COLOR[t.ground_truth], fontweight="bold",
        )
        ax.set_xlabel("Round")
        ax.set_xticks(range(n_rounds))
        ax.set_xticklabels([f"R{i}" for i in range(n_rounds)])
        ax.set_yticks(range(len(agent_names)))
        ax.set_yticklabels([n.replace("Agent", "") for n in agent_names])

    handles = [
        mpatches.Patch(color=_COLOR["Malignant"], label="Malignant"),
        mpatches.Patch(color=_COLOR["Benign"], label="Benign"),
        plt.scatter([], [], color="gold", marker="*", s=120, label="Opinion flip"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Agent Opinion Evolution Across Debate Rounds", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.show()


def plot_agreement_heatmap(transcripts: list[DebateTranscript]) -> None:
    n_rounds = len(transcripts[0].rounds)
    n_agents = len(transcripts[0].rounds[0])
    case_ids = [t.case_id for t in transcripts]

    data = np.array([
        [sum(op.diagnosis == "Malignant" for op in t.rounds[r]) / n_agents
         for t in transcripts]
        for r in range(n_rounds)
    ])

    fig, ax = plt.subplots(figsize=(max(6, len(transcripts) + 2), n_rounds + 1))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
        xticklabels=case_ids,
        yticklabels=[f"Round {r}" for r in range(n_rounds)],
        ax=ax, linewidths=0.5,
        cbar_kws={"label": "Fraction diagnosing Malignant"},
    )
    ax.set_title("Agent Agreement: Fraction Diagnosing Malignant per Round", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_confidence_comparison(transcripts: list[DebateTranscript]) -> None:
    agent_names = [op.agent_name for op in transcripts[0].rounds[0]]
    case_ids = [t.case_id for t in transcripts]
    x = np.arange(len(case_ids))
    width = 0.35

    fig, axes = plt.subplots(1, len(agent_names), figsize=(5 * len(agent_names), 5), sharey=True)
    if len(agent_names) == 1:
        axes = [axes]

    for a_idx, (ax, aname) in enumerate(zip(axes, agent_names)):
        r0 = [t.rounds[0][a_idx].confidence for t in transcripts]
        r2 = [t.rounds[-1][a_idx].confidence for t in transcripts]
        ax.bar(x - width / 2, r0, width, label="Round 0", color="#3498db", alpha=0.85)
        ax.bar(x + width / 2, r2, width, label="Round 2", color="#e67e22", alpha=0.85)
        ax.set_title(aname.replace("Agent", ""), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(case_ids, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Confidence")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("Agent Confidence: Round 0 vs Final Round", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_consensus_summary(
    reports: list[ConsensusReport], transcripts: list[DebateTranscript]
) -> None:
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, report, t in zip(axes, reports, transcripts):
        ax.axis("off")
        bg = _BG[report.final_diagnosis]
        correct_sym = "✓ CORRECT" if report.correct else "✗ INCORRECT"
        correct_color = "#1e8449" if report.correct else "#c0392b"

        rationale = (report.rationale[:130] + "…") if len(report.rationale) > 130 else report.rationale
        dissent = ""
        if report.dissent_notes:
            d = (report.dissent_notes[:90] + "…") if len(report.dissent_notes) > 90 else report.dissent_notes
            dissent = f"\n\nDissent:\n{d}"

        body = f"Diagnosis: {report.final_diagnosis}\nConfidence: {report.confidence_score:.0%}\n\nRationale:\n{rationale}{dissent}"

        ax.text(0.5, 0.95, report.case_id, transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="center", va="top")
        ax.text(0.5, 0.87, f"GT: {t.ground_truth}", transform=ax.transAxes,
                fontsize=9, ha="center", va="top", color="#555")
        ax.text(0.5, 0.65, body, transform=ax.transAxes, fontsize=8.5,
                va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=bg, edgecolor="#bbb"))
        ax.text(0.5, 0.05, correct_sym, transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center", va="bottom", color=correct_color)

    fig.suptitle("Consensus Diagnosis — Tumor Board Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 2: Quick smoke test**

Run in a Python shell to verify imports:
```bash
python -c "from src.analytics import plot_opinion_evolution, plot_agreement_heatmap, plot_confidence_comparison, plot_consensus_summary; print('analytics OK')"
```
Expected: `analytics OK`

- [ ] **Step 3: Commit**

```bash
git add src/analytics.py
git commit -m "feat: add analytics visualization functions"
```

---

## Task 7: Notebook Assembly

**Files:**
- Create: `onco_consensus.ipynb` (via `build_notebook.py`)
- Create: `build_notebook.py` (helper script, deletable after use)

- [ ] **Step 1: Create `build_notebook.py`**

```python
"""Run once: python build_notebook.py → produces onco_consensus.ipynb"""
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

def md(src): return nbformat.v4.new_markdown_cell(src)
def code(src): return nbformat.v4.new_code_cell(src)

nb.cells = [
    # ── §0 Setup ─────────────────────────────────────────────────────────────
    md("# Onco-Consensus: Multi-Agent Diagnostic Reasoning for Breast Cancer Pathology\n\n"
       "**CS505 — Spring 2026**\n\n"
       "Three LLM agents (Gemini Flash, Groq/Llama3, Ollama/Llama3) independently analyze "
       "FNA biopsy data, debate over 3 rounds, and a Gemini meta-synthesizer produces the "
       "final consensus diagnosis.\n\n---\n\n## §0 Setup & Installs"),

    code(
        "import subprocess\n"
        "subprocess.run(\n"
        '    ["pip", "install", "-q",\n'
        '     "google-generativeai", "groq", "python-dotenv",\n'
        '     "pydantic", "scikit-learn", "matplotlib", "seaborn", "pandas", "requests"],\n'
        "    check=True,\n"
        ")\n"
        'print("Installation complete")'
    ),

    code(
        "import os\n"
        "import sys\n"
        "sys.path.insert(0, os.getcwd())  # ensure src/ is importable\n\n"
        "from dotenv import load_dotenv\n"
        "load_dotenv()\n\n"
        "gemini_ok = bool(os.environ.get('GEMINI_API_KEY'))\n"
        "groq_ok   = bool(os.environ.get('GROQ_API_KEY'))\n"
        "print(f'GEMINI_API_KEY: {\"✓ loaded\" if gemini_ok else \"✗ missing — add to .env\"}')\n"
        "print(f'GROQ_API_KEY:   {\"✓ loaded\" if groq_ok   else \"✗ missing — add to .env\"}')\n"
        "print('Ollama:         local (no key needed)')"
    ),

    # ── §1 Data Layer ─────────────────────────────────────────────────────────
    md("---\n\n## §1 Data Layer\n\n"
       "UCI Wisconsin Breast Cancer dataset loaded via scikit-learn. "
       "A `case_to_narrative()` function converts 30 FNA nuclear morphology features "
       "into a clinical pathology paragraph. Ground-truth labels are hidden from agents."),

    code(
        "from src.data_layer import select_cases\n\n"
        "cases = select_cases(n_malignant=3, n_benign=2)\n"
        "print(f'Selected {len(cases)} cases\\n')\n"
        "for c in cases:\n"
        "    print(f'--- {c.case_id}  (GT: {c.ground_truth}) ---')\n"
        "    print(c.narrative[:220] + '...\\n')"
    ),

    # ── §2 Pydantic Models ────────────────────────────────────────────────────
    md("---\n\n## §2 Structured Data Models\n\n"
       "All agent outputs and orchestrator outputs use Pydantic models for reliable parsing."),

    code(
        "from src.models import DiagnosisOutput, DebateTranscript, ConsensusReport\n"
        "print('DiagnosisOutput fields:', list(DiagnosisOutput.model_fields.keys()))\n"
        "print('DebateTranscript fields:', list(DebateTranscript.model_fields.keys()))\n"
        "print('ConsensusReport fields:', list(ConsensusReport.model_fields.keys()))"
    ),

    # ── §3 Agent Definitions ──────────────────────────────────────────────────
    md("---\n\n## §3 Diagnostic Agents\n\n"
       "Three agents, each backed by a different LLM. All receive **identical** input in "
       "Round 0. In Rounds 1–2 they see the other agents' prior reasoning."),

    code(
        "from src.agents import GeminiAgent, GroqAgent, OllamaAgent\n\n"
        "gemini_agent = GeminiAgent()   # google-generativeai, gemini-2.0-flash\n"
        "groq_agent   = GroqAgent()     # groq SDK, llama3-70b-8192\n"
        "ollama_agent = OllamaAgent()   # local Ollama, llama3\n"
        "agents = [gemini_agent, groq_agent, ollama_agent]\n"
        "print('Agents initialized:')\n"
        "for a in agents:\n"
        "    print(f'  • {a.name}')"
    ),

    # ── §4 Orchestrator ───────────────────────────────────────────────────────
    md("---\n\n## §4 Debate Orchestrator + Meta-Synthesizer\n\n"
       "- `DebateOrchestrator`: runs 3 rounds of debate, collecting all `DiagnosisOutput` "
       "objects into a `DebateTranscript`.\n"
       "- `MetaSynthesizer`: Gemini in 'chief pathologist' mode synthesizes the full "
       "transcript into a `ConsensusReport`."),

    code(
        "from src.orchestrator import DebateOrchestrator, MetaSynthesizer\n\n"
        "orchestrator = DebateOrchestrator(agents)\n"
        "synthesizer  = MetaSynthesizer()\n"
        "print('DebateOrchestrator: 3-round structured debate')\n"
        "print('MetaSynthesizer:    Gemini in chief-pathologist role')"
    ),

    # ── §5 Run Pipeline ───────────────────────────────────────────────────────
    md("---\n\n## §5 Run Full Pipeline\n\n"
       "For each of 5 selected cases: 3-round debate → meta-synthesis → consensus report."),

    code(
        "transcripts = []\n"
        "reports     = []\n\n"
        "for case in cases:\n"
        "    print(f'\\n{'='*55}')\n"
        "    print(f'Processing {case.case_id}  (hidden GT: {case.ground_truth})')\n"
        "    print('='*55)\n\n"
        "    transcript = orchestrator.run(case, num_rounds=3)\n"
        "    transcripts.append(transcript)\n\n"
        "    for r_idx, round_ops in enumerate(transcript.rounds):\n"
        "        row = '  |  '.join(f'{op.agent_name.replace(\"Agent\",\"\")}: {op.diagnosis[0]} {op.confidence:.2f}'\n"
        "                           for op in round_ops)\n"
        "        flips = sum(op.changed_opinion for op in round_ops)\n"
        "        print(f'  Round {r_idx}: {row}  [{flips} flip(s)]')\n\n"
        "    report = synthesizer.synthesize(transcript)\n"
        "    reports.append(report)\n"
        "    verdict = '✓ CORRECT' if report.correct else '✗ WRONG'\n"
        "    print(f'  Consensus: {report.final_diagnosis} (conf: {report.confidence_score:.2f}) — {verdict}')\n\n"
        "accuracy = sum(r.correct for r in reports) / len(reports)\n"
        "print(f'\\nOverall accuracy: {accuracy:.0%} ({sum(r.correct for r in reports)}/{len(reports)})')"
    ),

    # ── §6 Analytics ──────────────────────────────────────────────────────────
    md("---\n\n## §6 Analytics\n\n"
       "Four visualizations:\n"
       "1. Opinion evolution (did any agent flip?)\n"
       "2. Agreement heatmap (consensus formation per round)\n"
       "3. Confidence bar chart (Round 0 vs Round 2)\n"
       "4. Consensus summary cards"),

    code(
        "%matplotlib inline\n"
        "import matplotlib\n"
        "matplotlib.rcParams['figure.dpi'] = 100\n\n"
        "from src.analytics import (\n"
        "    plot_opinion_evolution,\n"
        "    plot_agreement_heatmap,\n"
        "    plot_confidence_comparison,\n"
        "    plot_consensus_summary,\n"
        ")\n\n"
        "plot_opinion_evolution(transcripts)"
    ),

    code("plot_agreement_heatmap(transcripts)"),
    code("plot_confidence_comparison(transcripts)"),
    code("plot_consensus_summary(reports, transcripts)"),

    # ── §7 Results Summary ────────────────────────────────────────────────────
    md("---\n\n## §7 Results Summary"),

    code(
        "import pandas as pd\n"
        "from IPython.display import display\n\n"
        "rows = []\n"
        "for report, t in zip(reports, transcripts):\n"
        "    row = {'Case': report.case_id, 'Ground Truth': t.ground_truth}\n"
        "    for a_idx in range(len(t.rounds[0])):\n"
        "        aname = t.rounds[0][a_idx].agent_name.replace('Agent', '')\n"
        "        for r_idx in range(len(t.rounds)):\n"
        "            op = t.rounds[r_idx][a_idx]\n"
        "            row[f'{aname} R{r_idx}'] = f\"{op.diagnosis[0]} {op.confidence:.2f}\"\n"
        "    row['Consensus']  = report.final_diagnosis\n"
        "    row['Confidence'] = f'{report.confidence_score:.2f}'\n"
        "    row['Correct']    = '✓' if report.correct else '✗'\n"
        "    rows.append(row)\n\n"
        "df = pd.DataFrame(rows).set_index('Case')\n"
        "display(df)\n"
        "print(f'\\nAccuracy: {sum(r.correct for r in reports)}/{len(reports)}')"
    ),
]

with open("onco_consensus.ipynb", "w") as f:
    nbformat.write(nb, f)

print("Created: onco_consensus.ipynb")
```

- [ ] **Step 2: Run the build script**

```bash
pip install -q nbformat
python build_notebook.py
```
Expected: `Created: onco_consensus.ipynb`

- [ ] **Step 3: Verify notebook structure**

```bash
python -c "
import json
nb = json.load(open('onco_consensus.ipynb'))
cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    preview = (c['source'][:60] if isinstance(c['source'], str) else ''.join(c['source'])[:60]).replace('\n',' ')
    print(f'  [{i}] {c[\"cell_type\"]}: {preview}')
"
```
Expected: `Total cells: 17` (8 markdown + 9 code cells), with section headers visible.

- [ ] **Step 4: Commit**

```bash
git add onco_consensus.ipynb build_notebook.py
git commit -m "feat: assemble Jupyter notebook deliverable"
```

---

## Task 8: End-to-End Validation

**Files:** None created — validation only.

> **Prerequisites:** `.env` file present with `GEMINI_API_KEY` and `GROQ_API_KEY`. Ollama running locally with `llama3` pulled (`ollama pull llama3`).

- [ ] **Step 1: Run all unit tests one final time**

```bash
pytest -v
```
Expected: `22 passed, 0 failed`

- [ ] **Step 2: Verify Ollama is running**

```bash
curl -s http://localhost:11434/api/tags | python -c "import json,sys; models=[m['name'] for m in json.load(sys.stdin)['models']]; print('Ollama models:', models)"
```
Expected: output includes `llama3` in the models list.

- [ ] **Step 3: Execute the notebook non-interactively**

```bash
pip install -q jupyter nbconvert
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 onco_consensus.ipynb --output onco_consensus_executed.ipynb
```
Expected: exits with code 0 and produces `onco_consensus_executed.ipynb`.

- [ ] **Step 4: Verify all cells executed without errors**

```bash
python -c "
import json
nb = json.load(open('onco_consensus_executed.ipynb'))
errors = []
for i, cell in enumerate(nb['cells']):
    for output in cell.get('outputs', []):
        if output.get('output_type') == 'error':
            errors.append(f'Cell {i}: {output[\"ename\"]}: {output[\"evalue\"]}')
if errors:
    print('ERRORS FOUND:')
    for e in errors: print(' ', e)
else:
    print(f'All {len(nb[\"cells\"])} cells executed successfully — no errors.')
"
```
Expected: `All 17 cells executed successfully — no errors.`

- [ ] **Step 5: Final commit**

```bash
git add onco_consensus_executed.ipynb
git commit -m "feat: add executed notebook with full pipeline results"
```

---

## Self-Review

**Spec coverage:**
- ✓ §2 Dataset — `select_cases()` uses sklearn, `case_to_narrative()` generates clinical text
- ✓ §3 Agent Layer — `DiagnosticAgent` base, `GeminiAgent`, `GroqAgent`, `OllamaAgent` with identical Round-0 inputs
- ✓ §4 Debate Orchestrator — `DebateOrchestrator` runs 3 rounds, collects `DebateTranscript`
- ✓ §5 Meta-Synthesizer — `MetaSynthesizer` (Gemini judge) returns `ConsensusReport` with `correct` flag
- ✓ §6 Analytics — all 4 visualizations implemented in `src/analytics.py`
- ✓ §7 Notebook Structure — all 8 sections present in `onco_consensus.ipynb`
- ✓ §8 Config/Security — `.env` + `.gitignore`, Ollama needs no key, `GROQ_BASE_URL` optional
- ✓ §9 Design constraints — identical Round-0 inputs, traceable `DebateTranscript`, Pydantic throughout

**Type consistency check:**
- `_split_prior()` returns `(others, self_prior, round_num)` — used consistently across all 3 agent classes ✓
- `DebateOrchestrator.run()` returns `DebateTranscript` — consumed by `MetaSynthesizer.synthesize()` ✓
- `MetaSynthesizer.synthesize()` accepts `DebateTranscript`, returns `ConsensusReport` ✓
- `plot_consensus_summary(reports, transcripts)` — both args are lists of matching type ✓
- `rounds[-1]` in `plot_confidence_comparison` — correct for final round regardless of `num_rounds` ✓

**Placeholder scan:** None found. All code blocks are complete.
