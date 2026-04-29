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


def _mock_openai_client(response_text: str) -> MagicMock:
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = response_text
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


# ── GeminiAgent ───────────────────────────────────────────────────────────────

def test_gemini_round0_returns_diagnosis_output():
    agent = GeminiAgent(client=_mock_openai_client(R0_MALIGNANT_JSON))
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
    agent = GeminiAgent(client=_mock_openai_client(R0_MALIGNANT_JSON))
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.round == 1


def test_gemini_round1_detects_unchanged_opinion():
    prior = [
        DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Malignant",
                        confidence=0.85, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    agent = GeminiAgent(client=_mock_openai_client(R0_MALIGNANT_JSON))
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.changed_opinion is False


def test_gemini_round1_detects_changed_opinion():
    prior = [
        DiagnosisOutput(agent_name="GeminiAgent", round=0, diagnosis="Benign",
                        confidence=0.6, key_findings=[], reasoning="r0", changed_opinion=False),
    ]
    agent = GeminiAgent(client=_mock_openai_client(R0_MALIGNANT_JSON))
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.changed_opinion is True


# ── GroqAgent ─────────────────────────────────────────────────────────────────

def test_groq_round0_returns_diagnosis_output():
    agent = GroqAgent(client=_mock_openai_client(R0_BENIGN_JSON))
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
    agent = GroqAgent(client=_mock_openai_client(R0_BENIGN_JSON))
    out = agent.analyze(MOCK_CASE, prior_opinions=prior)
    assert out.round == 1


# ── OllamaAgent ───────────────────────────────────────────────────────────────

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
