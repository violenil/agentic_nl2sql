## Agentic NL2SQL

Lightweight pipeline to convert natural language questions to SQL against a SQLite database, validate executions, and write predictions for downstream evaluation.

### Features
- In-memory SQLite built from either:
  - A directory of `.sqlite/.db` files (merged into one connection), or
  - CSVs listed in a JSON table map.
- Agentic SQL generation with Azure OpenAI + LangChain.
- Schema-thinning to control tokens/latency (filter/cap tables, optional sample rows).
- Validation and one-pass auto-correction on execution error.
- Deterministic question sampling (first N) and progress logging.

### Requirements
- Python 3.10+
- An Azure OpenAI deployment for Chat Completions (e.g., GPT-4o).

### Install
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Environment
Create a `.env` in the repo root. Choose ONE database input method.

```ini
# --- Azure OpenAI ---
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
# Optional; defaults used if omitted
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=lunar-chatgpt-4o

# --- Database Source (choose one) ---
# Option A: Directory of SQLite files (merged into an in-memory DB)
SQLITE_DB_DIR=/absolute/path/to/sqlite_dir

# Option B: CSVs listed in a JSON file mapping table name -> csv path
# TABLE_DICT=/absolute/path/to/table_map.json

# --- Questions & Outputs ---
QUESTION_FILE_PATH=/absolute/path/to/questions.txt
PREDICT_FILE_PATH=/absolute/path/to/predictions/pred.txt

# --- Optional controls ---
# Take first N questions deterministically (omit or 0 = all)
SAMPLE_SIZE=50

# Schema thinning: include only matching prefixes (comma-separated)
# Example: academic__,college__
DB_INCLUDE_TABLE_PREFIX=

# Cap number of tables included (omit to disable)
DB_MAX_TABLES=15

# Number of sample rows per table to include in schema context (0 = none)
DB_SAMPLE_ROWS_IN_TABLE_INFO=0

# Treat existing per-question artifacts as ignorable; this script no longer writes them
FORCE=false
```

Example `TABLE_DICT` (if using CSVs):
```json
{
  "students": "/abs/path/to/students.csv",
  "courses": "/abs/path/to/courses.csv"
}
```

### Run
```bash
source venv/bin/activate
python agent_langchain.py
```

### What it does
1. Loads env and builds an in-memory SQLite DB:
   - If `SQLITE_DB_DIR` is set, each file is attached, tables are copied into `main` as `filename__table`, then detached (no 10-DB limit).
   - Else it loads CSVs from `TABLE_DICT` into tables.
2. Prepares a schema-limited `SQLDatabase`:
   - `DB_INCLUDE_TABLE_PREFIX`, `DB_MAX_TABLES` control table visibility.
   - `DB_SAMPLE_ROWS_IN_TABLE_INFO` controls example rows in schema (0 by default).
3. For the first `SAMPLE_SIZE` questions (or all):
   - Generates SQL with a strict system hint (explicit JOIN ... ON; semicolon).
   - Retries on 429 with backoff.
   - Validates via `pd.read_sql`; on error, requests one correction and re-validates.
   - Appends the final SQL (single line) to `PREDICT_FILE_PATH`.

### Outputs
- Predictions: one SQL per line in `PREDICT_FILE_PATH`.
- Gold: this script does not write gold. Provide your own gold file to your evaluation pipeline.

### Logging & Progress
- Timestamps for: DB build, source DB attaches, table copies, schema summary, per-question steps, retries, validation, and completion.
- Progress prefix `[i/N]` is printed for each question.

### Tuning Tips
- If requests are slow or rate-limited:
  - Reduce `DB_MAX_TABLES`, add prefixes in `DB_INCLUDE_TABLE_PREFIX`.
  - Keep `DB_SAMPLE_ROWS_IN_TABLE_INFO=0` or at most 1–3 to reduce tokens.
  - Lower `SAMPLE_SIZE`.
- If joins are incorrect:
  - Slightly increase `DB_SAMPLE_ROWS_IN_TABLE_INFO` (1–2) to reveal key columns.
  - Tighten prefixes so only relevant tables are visible.

### Troubleshooting
- 429 Rate limit: the script retries with exponential backoff automatically. If persistent, lower table count/sample rows or increase Azure quota.
- “too many attached databases”: Avoided by copying tables one source DB at a time, then detaching.
- Evaluator KeyError ",": Ensure generated SQL uses explicit `JOIN ... ON`, not comma-separated tables in FROM (already enforced by the system hint).

### License
MIT or project’s default; update as needed.


