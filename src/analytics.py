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
