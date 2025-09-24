## Agentic NL2SQL (LangGraph pipeline)

End-to-end, multi-stage NL→SQL system with critique-driven prompt refinement and partial reruns. The core orchestration lives in `agentic_nl2sql_graph.py` and uses LangGraph to coordinate stages, analysis, critique, and targeted refinements.

### Architecture Overview
- **Stage 1 (Relevant attributes)**: select relevant tables/columns for the question.
- **Stage 2 (Value instances)**: extract likely literal values for predicates.
- **Stage 3 (SQL synthesis)**: generate the final SQL using outputs of Stage 1 and 2.
- **Analyzer**: executes/inspects SQL against SQLite; reports syntax, exec status, sample rows, and semantic hints.
- **Critic**: LLM evaluates the output and analysis; returns a structured critique with `likely_stage` to improve.
- **Refiner**: updates the prompt for the chosen stage and records the change.
- **Refinement Router**: restarts the graph from `stage1`, `stage2`, or `stage3` depending on which stage was refined (not always from stage1).

### Key Behaviors
- **Partial reruns from refined stage**: After a refinement, the graph resumes at the refined stage:
  - `stage1` refinement → rerun `stage1_rerun → stage2_rerun → stage3_rerun → analyzer_rerun`.
  - `stage2` refinement → rerun `stage2_rerun → stage3_rerun → analyzer_rerun`.
  - `stage3` refinement → rerun `stage3_rerun → analyzer_rerun`.
- **Early stop / retry policy**: A router decides to stop or continue refining based on analysis improvements and a max-refinements cap.
- **History logging**: Prompt changes are appended to timestamped files under `history/`.

### Repository Layout (selected)
- `agentic_nl2sql_graph.py`: LangGraph pipeline orchestration and entry point.
- `agents/`
  - `stage1.py`, `stage2.py`, `stage3.py`: stage agents powered by prompts.
  - `analyzer.py`: SQLite-based analysis and semantic hints.
  - `critic.py`: LLM-based critique returning JSON with `likely_stage`.
  - `refiner.py`: updates prompts and logs changes.
- `core/prompt_manager.py`: loads prompts from `prompts/*.yaml`, supports both single `prompt` or split `system/user` forms.
- `prompts/`: stage prompts and the `critic`/`refiner` templates.
- `f1.sqlite`: example SQLite database (or use your own via env var).

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` in the repo root. The graph expects a single SQLite file path.

```ini
# SQLite database to analyze queries against
SQLITE_DB_FILE=/absolute/path/to/your.db

# Optional: cap the number of refinement loops (default 3)
MAX_REFINEMENTS=3

# Azure OpenAI settings used by agents (names match your deployments)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
# Example deployment names (adjust to your setup)
AZURE_OPENAI_DEPLOYMENT=lunar-gpt-4o
```

### Running the LangGraph Pipeline
You can execute the pipeline directly via the module’s `__main__` block, which demonstrates the flow on a sample question and writes a prompt-evolution log.

```bash
source venv/bin/activate
python agentic_nl2sql_graph.py
```

What happens:
- Loads `.env` and prompts from `prompts/` via `PromptManager`.
- Builds an inline schema string from `SQLITE_DB_FILE` for stage prompts.
- Runs `stage1 → stage2 → stage3 → analyzer → critic → refiner`.
- Uses the refinement router to restart from the refined stage (if any), then `analyzer_rerun` decides to stop or continue based on improvement and `MAX_REFINEMENTS`.
- Prints the final SQL, analysis, critique, and refinement summary.

### Prompts
- Stored in `prompts/*.yaml`.
- File keys supported:
  - Single-key form: `prompt: "..."` for stage prompts.
  - Dual-key form: `system: "..."` and `user: "..."` for `critic`/`refiner` templates.
- The `RefinerAgent` updates the in-memory prompt set and logs diffs to `history/prompt_evolution_*.log`.

### Analyzer Outputs
The `ExperimentAnalyzer` returns a JSON-like dict including:
- `syntax_ok`: boolean
- `exec_ok`: boolean
- `row_count_sample`: integer sample size
- `rows_sample`: small sample of result rows
- `columns`: column names
- `error`: syntax/exec error if any
- `semantic_hints`: lightweight hints to guide the critic

### Critique and Refinement
- `CriticAgent` produces JSON with at least `likely_stage` and issue notes.
- `RefinerAgent` consumes the critique, refines the prompt for that stage, and records a log entry. It also returns `{"stage": "stage1|stage2|stage3", ...}` used by the router.

### Rerun and Acceptance Logic
- After a refinement, the router selects the appropriate rerun entry (`stage1_rerun | stage2_rerun | stage3_rerun`).
- The rerun chain always proceeds forward to `analyzer_rerun`.
- A decision node compares the latest analysis to the previous analysis to either stop or loop back for another critique/refinement, honoring `MAX_REFINEMENTS`.

### Example: Custom Invocation
Programmatically invoke the compiled graph with a custom question and DB path.

```python
from agentic_nl2sql_graph import graph, PromptManager, get_schema_string
import os

pm = PromptManager(prompt_dir="prompts")
question = "Which teams scored the most points in 2010?"
schema = get_schema_string(os.getenv("SQLITE_DB_FILE"))

state = {
    "prompt_manager": pm,
    "question": question,
    "schema": schema,
    "config": {
        "db_path": os.getenv("SQLITE_DB_FILE"),
        "history_file": "history/prompt_evolution_custom.log",
    },
    "refinement_count": 0,
}

final_state = graph.invoke(state)
print(final_state.get("sql"))
```

### CLI/Batch Alternative
For batch prediction over a questions file with schema-thinning and basic error-correction, see `agentic_nl2sql.py`. It builds an in-memory DB (from a single file, a directory of SQLite files, or CSVs), generates SQL, validates/corrects, and writes one SQL per line to an output file.

### Logging & History
- Prompt refinements are appended to `history/prompt_evolution_*.log` with stage, issues, notes, explanation, and before/after prompt content.
- The graph prints a concise summary of the final artifacts to stdout.

### Requirements
- Python 3.10+
- Valid Azure OpenAI Chat Completions deployment(s)

### Troubleshooting
- Ensure `SQLITE_DB_FILE` points to a readable SQLite file; analysis uses that DB directly.
- If LLM requests fail, verify Azure environment variables and deployment names.
- If the rerun always starts from stage1, confirm the critic returns a valid `likely_stage` and that prompts for `critic/refiner` are present.

### License
MIT or project’s default; update as needed.
