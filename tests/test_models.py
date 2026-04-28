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
