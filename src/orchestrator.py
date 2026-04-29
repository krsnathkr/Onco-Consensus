import json
import os
import re

from src.agents import DiagnosticAgent, _openrouter_client, _chat
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
    def __init__(self, client=None):
        self.client = client or _openrouter_client()
        self.model_id = os.environ.get("GEMINI_MODEL", "google/gemini-flash-1.5")

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

        raw = _chat(
            self.client,
            self.model_id,
            SYNTHESIZER_SYSTEM,
            debate_text + SYNTHESIZER_JSON_HINT,
        )

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON in synthesizer response: {raw[:300]}")
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
