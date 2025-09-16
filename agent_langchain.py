import os
import json
import re
import sqlite3
import datetime
import time
import random
from io import StringIO
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

# LangChain + OpenAI Azure
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType


def log(message: str):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}", flush=True)


def read_questions_from_file(question_file_path: str) -> List[str]:
    with open(question_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return re.findall(r"'(.*?)'", content, re.DOTALL)


def build_sqlite_in_memory_from_csvs(dict_path_csv: Dict[str, str]) -> sqlite3.Connection:
    log("Starting CSV -> SQLite in-memory build")
    conn = sqlite3.connect(':memory:')
    for table_name, csv_path in dict_path_csv.items():
        log(f"Loading CSV for table '{table_name}' from: {csv_path}")
        df = pd.read_csv("../nl2sql/"+csv_path, encoding='utf8')
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        try:
            log(f"Loaded {len(df)} rows into table '{table_name}'")
        except Exception:
            pass
    log("Completed CSV -> SQLite in-memory build")
    return conn


def build_sqlite_in_memory_from_sqlite_dir(sqlite_dir: str) -> sqlite3.Connection:
    log(f"Starting SQLite merge from directory: {sqlite_dir}")
    conn = sqlite3.connect(':memory:')
    base_dir = os.path.abspath(os.path.expanduser(sqlite_dir))
    if not os.path.isdir(base_dir):
        # Try to recover from a missing leading slash like 'Users/...' -> '/Users/...'
        alt = os.path.sep + sqlite_dir if not sqlite_dir.startswith(os.path.sep) else sqlite_dir
        alt = os.path.abspath(os.path.expanduser(alt))
        if os.path.isdir(alt):
            base_dir = alt
        else:
            raise RuntimeError(f"SQLITE_DB_DIR not found: {sqlite_dir}")

    # Collect .db / .sqlite / .sqlite3 files
    candidates = []
    for name in os.listdir(base_dir):
        lower = name.lower()
        if lower.endswith('.db') or lower.endswith('.sqlite') or lower.endswith('.sqlite3'):
            candidates.append(os.path.join(base_dir, name))

    log(f"Found {len(candidates)} sqlite files to merge")
    if not candidates:
        raise RuntimeError(f"No sqlite files found in directory: {base_dir}")

    created_table_names: set = set()

    def make_prefix(file_path: str) -> str:
        base = os.path.splitext(os.path.basename(file_path))[0]
        return re.sub(r"[^A-Za-z0-9_]", "_", base) or "db"

    total_copied = 0
    for db_path in sorted(candidates):
        src_path = os.path.abspath(db_path)
        log(f"Attaching source DB: {src_path}")
        # Attach one-by-one to avoid exceeding SQLITE_MAX_ATTACHED
        conn.execute("ATTACH DATABASE ? AS src", (src_path,))
        # Read tables from attached db
        cursor = conn.execute("SELECT name FROM src.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        prefix = make_prefix(db_path)
        log(f"Copying {len(tables)} tables from {src_path} with prefix '{prefix}__'")
        for tbl in tables:
            # sanitize and ensure unique target name
            base_name = re.sub(r"[^A-Za-z0-9_]", "_", str(tbl)) or "table"
            target = f"{prefix}__{base_name}"
            if target in created_table_names:
                i = 2
                while f"{target}_{i}" in created_table_names:
                    i += 1
                target = f"{target}_{i}"
            # Create table in main by copying data
            log(f"Creating table '{target}' from 'src.{tbl}'")
            conn.execute(f"CREATE TABLE \"{target}\" AS SELECT * FROM src.\"{tbl}\"")
            created_table_names.add(target)
            total_copied += 1
        conn.execute("DETACH DATABASE src")
        log(f"Detached source DB: {src_path}")

    log(f"Completed merge. Total tables copied: {total_copied}")
    return conn


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)


def now_log_prefix() -> str:
    directory = 'logs'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return datetime.datetime.now().strftime('logs/%Y-%m-%d_%H:%M:%S')


def setup_azure_llm() -> AzureChatOpenAI:
    log("Setting up Azure LLM client")
    # LangChain AzureChatOpenAI expects the OpenAI SDK envs
    # Required envs:
    #   AZURE_OPENAI_API_KEY
    #   AZURE_OPENAI_ENDPOINT
    #   AZURE_OPENAI_API_VERSION (defaults provided below if missing)
    #   AZURE_OPENAI_DEPLOYMENT (your chat model deployment name)
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'lunar-chatgpt-4o')

    if not api_key or not endpoint:
        raise RuntimeError('Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment')

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
        temperature=0.0,
    )


def retry_with_backoff(callable_fn, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    attempt = 0
    while True:
        try:
            return callable_fn()
        except Exception as e:
            message = str(e)
            is_rate_limited = '429' in message or 'rate limit' in message.lower()
            if not is_rate_limited or attempt >= max_retries:
                raise
            sleep_s = min(max_delay, base_delay * (2 ** attempt))
            log(f"Rate limited (attempt {attempt + 1}/{max_retries}). Sleeping ~{sleep_s:.1f}s before retry")
            jitter = sleep_s * (0.5 + random.random() * 0.5)
            time.sleep(jitter)
            attempt += 1


def make_sql_agent(conn: sqlite3.Connection) -> any:
    # Wrap existing sqlite3 connection in LangChain SQLDatabase
    # Use a custom driver string for in-memory connection
    log("Preparing schema context for agent")

    # Read all table names from the in-memory DB
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = sorted([row[0] for row in cur.fetchall()])

    # Env-driven filtering/capping
    include_prefixes_env = os.getenv('DB_INCLUDE_TABLE_PREFIX', '').strip()
    include_prefixes = [p.strip() for p in include_prefixes_env.split(',') if p.strip()] if include_prefixes_env else []
    max_tables_env = os.getenv('DB_MAX_TABLES', '').strip()
    try:
        max_tables = int(max_tables_env) if max_tables_env else None
    except Exception:
        max_tables = None
    sample_rows_env = os.getenv('DB_SAMPLE_ROWS_IN_TABLE_INFO', '').strip()
    try:
        sample_rows = int(sample_rows_env) if sample_rows_env else 0
    except Exception:
        sample_rows = 0

    # Apply prefix filtering
    if include_prefixes:
        filtered = [t for t in all_tables if any(t.startswith(pref) for pref in include_prefixes)]
    else:
        filtered = list(all_tables)

    # Apply cap
    if max_tables is not None and len(filtered) > max_tables:
        filtered = filtered[:max_tables]

    log(f"Schema tables: total={len(all_tables)}, included={len(filtered)}, sample_rows={sample_rows}")

    # Construct SQLDatabase with constraints
    db = SQLDatabase.from_uri(
        'sqlite://',
        engine_args={'creator': lambda: conn},
        include_tables=filtered if filtered else None,
        sample_rows_in_table_info=sample_rows,
    )
    llm = setup_azure_llm()
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
    )
    log("Agent ready")
    return agent


def main():
    load_dotenv(override=True)
    log("Environment loaded")

    table_dict_path = os.getenv('TABLE_DICT')
    sqlite_db_dir = os.getenv('SQLITE_DB_DIR')
    question_file_path = os.getenv('QUESTION_FILE_PATH')
    predict_file_path = os.getenv('PREDICT_FILE_PATH')
    force = os.getenv('FORCE', 'False')
    force = force.lower() in ('1', 'true', 'yes')

    if not question_file_path or not predict_file_path:
        raise RuntimeError('Expected env vars QUESTION_FILE_PATH and PREDICT_FILE_PATH')

    dict_path_csv = None
    if not sqlite_db_dir:
        if not table_dict_path:
            raise RuntimeError('Provide either SQLITE_DB_DIR or TABLE_DICT')
        with open(table_dict_path, 'r') as f:
            dict_path_csv = json.load(f)
        log(f"Loaded TABLE_DICT with {len(dict_path_csv)} entries")

    # Disable question_logs/question_last outputs

    # Build DB in memory and create agent
    if sqlite_db_dir:
        print("Building DB from sqlite directory")
        conn = build_sqlite_in_memory_from_sqlite_dir(sqlite_db_dir)
    else:
        print("Build DB from csv files")
        conn = build_sqlite_in_memory_from_csvs(dict_path_csv)
    agent = make_sql_agent(conn)
    print("Done building DB")

    questions = read_questions_from_file(question_file_path)
    log(f"Loaded {len(questions)} questions from file")

    # Optional sampling (deterministic: first N)
    sample_size_env = os.getenv('SAMPLE_SIZE', '').strip()
    try:
        sample_size = int(sample_size_env) if sample_size_env else 0
    except Exception:
        sample_size = 0

    if sample_size and sample_size > 0:
        n = min(sample_size, len(questions))
        questions_sampled = questions[:n]
        log(f"Selecting first {n} questions deterministically")
    else:
        questions_sampled = questions

    # Prepare gold and predictions files
    pred_dir = os.path.dirname(predict_file_path)
    pred_base = os.path.splitext(os.path.basename(predict_file_path))[0]
    # Reset predictions file only (no gold file writes here)
    open(predict_file_path, 'w').close()
    log(f"Initialized predictions at {predict_file_path}")

    total = len(questions_sampled)
    for idx, raw_q in enumerate(questions_sampled, start=1):
        log(f"[{idx}/{total}] Starting new question")
        nl_query = raw_q[0].upper() + raw_q[1:] if raw_q else raw_q
        # No skip based on prior logs; always process sampled questions

        # Ask the agent to produce SQL only
        system_hint = (
            "You are a text-to-SQL assistant. Return ONLY the final SQL query, "
            "no markdown fences, no explanation. Use explicit JOIN ... ON syntax "
            "(never comma-separated tables in FROM). Always terminate with a semicolon."
        )
        try:
            # LangChain agent interface: call with a prompt that nudges SQL-only output.
            log(f"[{idx}/{total}] Invoking agent for SQL generation")
            result = retry_with_backoff(lambda: agent.invoke({
                'input': f"{system_hint}\nQuestion: {nl_query}",
            }))
            # result may be dict or string depending on LC version
            sql_text = result.get('output') if isinstance(result, dict) else str(result)
        except Exception as e:
            sql_text = f"-- agent_error: {e}"
            log(f"Agent invocation failed: {e}")

        # Try to extract the first terminated SQL statement
        try:
            query = re.findall(r"^(?!.*\\bsql\\b)[\\s\\S]+?;", sql_text.replace("```", "").strip() + ";", re.MULTILINE)[0]
        except Exception:
            query = sql_text.strip()
            if not query.endswith(';'):
                query += ';'

        # Execute on in-memory DB to validate quickly
        exec_err = None
        log(f"[{idx}/{total}] Validating generated SQL against in-memory DB")
        try:
            pd.read_sql(query, conn)
        except Exception as e:
            exec_err = str(e)
            log(f"Validation error: {exec_err}")

        # If failed, try a simple correction pass via LLM
        if exec_err and not sql_text.startswith('-- agent_error'):
            llm = setup_azure_llm()
            correction_prompt = f"Correct this SQL for SQLite. Return only SQL.\nError: {exec_err}\nSQL:\n{query}"
            try:
                log(f"[{idx}/{total}] Attempting correction via LLM")
                fixed = retry_with_backoff(lambda: llm.invoke(correction_prompt))
                corrected = fixed.content if hasattr(fixed, 'content') else str(fixed)
                try:
                    query = re.findall(r"^(?!.*\\bsql\\b)[\\s\\S]+?;", corrected.replace("```", "").strip() + ";", re.MULTILINE)[0]
                except Exception:
                    query = corrected.strip()
                    if not query.endswith(';'):
                        query += ';'
                # re-try execution silently
                try:
                    log(f"[{idx}/{total}] Re-validating corrected SQL")
                    pd.read_sql(query, conn)
                except Exception:
                    pass
            except Exception:
                log("Correction step failed; proceeding with original query")
                pass

        with open(predict_file_path, 'a') as f:
            f.write(query.replace('\n', ' ') + "\n")
        log(f"[{idx}/{total}] Appended query to predictions file")

    conn.close()
    log(f"Completed processing {total} question(s). Closed in-memory DB connection")


if __name__ == '__main__':
    main()


