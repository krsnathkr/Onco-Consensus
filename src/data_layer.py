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
