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
        "    print(f'\\n' + '='*55)\n"
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
        "            row[f'{aname} R{r_idx}'] = f\\\"{op.diagnosis[0]} {op.confidence:.2f}\\\"\n"
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
