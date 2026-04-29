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
        self.model_id = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

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

    def __init__(self, base_url: str = "http://localhost:11434", model_id: str = "llama3:8b"):
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
