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


def _mock_openai_client(response_text: str) -> MagicMock:
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = response_text
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


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
    synth = MetaSynthesizer(client=_mock_openai_client(
        '{"final_diagnosis": "Malignant", "confidence_score": 0.87, '
        '"rationale": "2 of 3 agents agree on malignancy.", '
        '"dissent_notes": "OllamaAgent disagreed, citing benign features."}'
    ))
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
    synth = MetaSynthesizer(client=_mock_openai_client(
        '{"final_diagnosis": "Benign", "confidence_score": 0.8, '
        '"rationale": "Agents agree benign.", "dissent_notes": null}'
    ))
    report = synth.synthesize(transcript)

    assert report.correct is False
