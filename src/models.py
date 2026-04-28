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
    dissent_notes: str | None = None
    correct: bool

    @field_validator("confidence_score")
    @classmethod
    def confidence_score_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence_score must be 0.0–1.0, got {v}")
        return v
