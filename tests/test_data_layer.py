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
