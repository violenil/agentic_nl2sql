import os
import json
import re
import sqlite3
import datetime
from io import StringIO
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

# LangChain + OpenAI Azure
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType


def read_questions_from_file(question_file_path: str) -> List[str]:
    with open(question_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return re.findall(r"'(.*?)'", content, re.DOTALL)


def build_sqlite_in_memory_from_csvs(dict_path_csv: Dict[str, str]) -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    for table_name, csv_path in dict_path_csv.items():
        df = pd.read_csv("../nl2sql/"+csv_path, encoding='utf8')
        df.to_sql(table_name, conn, index=False, if_exists='replace')
    return conn


def build_sqlite_in_memory_from_sqlite_dir(sqlite_dir: str) -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    if not os.path.isdir(sqlite_dir):
        raise RuntimeError(f"SQLITE_DB_DIR not found: {sqlite_dir}")

    # Collect .db / .sqlite / .sqlite3 files
    candidates = []
    for name in os.listdir(sqlite_dir):
        lower = name.lower()
        if lower.endswith('.db') or lower.endswith('.sqlite') or lower.endswith('.sqlite3'):
            candidates.append(os.path.join(sqlite_dir, name))

    if not candidates:
        raise RuntimeError(f"No sqlite files found in directory: {sqlite_dir}")

    used_names = set()

    def make_attach_name(file_path: str) -> str:
        base = os.path.splitext(os.path.basename(file_path))[0]
        # sanitize to sqlite identifier
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", base)
        if not sanitized:
            sanitized = "db"
        name = sanitized
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{sanitized}_{suffix}"
        used_names.add(name)
        return name

    for db_path in sorted(candidates):
        attach_name = make_attach_name(db_path)
        conn.execute(f"ATTACH DATABASE ? AS {attach_name}", (os.path.abspath(db_path),))

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


def make_sql_agent(conn: sqlite3.Connection) -> any:
    # Wrap existing sqlite3 connection in LangChain SQLDatabase
    # Use a custom driver string for in-memory connection
    db = SQLDatabase.from_uri('sqlite://', engine_args={'creator': lambda: conn})
    llm = setup_azure_llm()
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent


def main():
    load_dotenv(override=True)

    table_dict_path = os.getenv('TABLE_DICT')
    sqlite_db_dir = os.getenv('SQLITE_DB_DIR')
    question_file_path = os.getenv('QUESTION_FILE_PATH')
    question_log_dir = os.getenv('QUESTION_LOG_FILE_PATH')
    predict_file_path = os.getenv('PREDICT_FILE_PATH')
    force = os.getenv('FORCE', 'False')
    force = force.lower() in ('1', 'true', 'yes')

    if not question_file_path or not question_log_dir or not predict_file_path:
        raise RuntimeError('Expected env vars QUESTION_FILE_PATH, QUESTION_LOG_FILE_PATH, PREDICT_FILE_PATH')

    dict_path_csv = None
    if not sqlite_db_dir:
        if not table_dict_path:
            raise RuntimeError('Provide either SQLITE_DB_DIR or TABLE_DICT')
        with open(table_dict_path, 'r') as f:
            dict_path_csv = json.load(f)

    ensure_dir(question_log_dir)
    log_prefix = now_log_prefix()
    json_log_path = f'{log_prefix}.json'
    txt_log_path = f'{log_prefix}.txt'
    with open(json_log_path, 'w') as f:
        json.dump({'iterations': []}, f)
    open(txt_log_path, 'w').close()

    def write_log(input_text: str, output_text: str):
        with open(json_log_path, 'r') as f:
            data = json.load(f)
        data['iterations'].append({'input': input_text})
        data['iterations'].append({'output': output_text})
        with open(json_log_path, 'w') as f:
            json.dump(data, f)
        with open(txt_log_path, 'a') as f:
            f.write(f"\ninput:\n{input_text}\n---------------------------------\n")
        with open(txt_log_path, 'a') as f:
            f.write(f"output:\n{output_text}\n---------------------------------\n")

    # Build DB in memory and create agent
    if sqlite_db_dir:
        conn = build_sqlite_in_memory_from_sqlite_dir(sqlite_db_dir)
    else:
        conn = build_sqlite_in_memory_from_csvs(dict_path_csv)
    agent = make_sql_agent(conn)

    questions = read_questions_from_file(question_file_path)

    for raw_q in questions:
        nl_query = raw_q[0].upper() + raw_q[1:] if raw_q else raw_q
        q_log_path = os.path.join(question_log_dir, f"{nl_query}.txt")
        if os.path.isfile(q_log_path) and not force:
            continue

        # Ask the agent to produce SQL only
        system_hint = (
            "You are a text-to-SQL assistant. Return ONLY the final SQL query, "
            "no markdown fences, no explanation."
        )
        try:
            # LangChain agent interface: call with a prompt that nudges SQL-only output.
            result = agent.invoke({
                'input': f"{system_hint}\nQuestion: {nl_query}",
            })
            # result may be dict or string depending on LC version
            sql_text = result.get('output') if isinstance(result, dict) else str(result)
        except Exception as e:
            sql_text = f"-- agent_error: {e}"

        write_log(nl_query, sql_text)

        with open(q_log_path, 'w') as f:
            f.write(f"{nl_query}\n\n")
            f.write("-" * 50 + "\n")
            f.write(sql_text)

        # Try to extract the first terminated SQL statement
        try:
            query = re.findall(r"^(?!.*\\bsql\\b)[\\s\\S]+?;", sql_text.replace("```", "").strip() + ";", re.MULTILINE)[0]
        except Exception:
            query = sql_text.strip()
            if not query.endswith(';'):
                query += ';'

        # Execute on in-memory DB to validate quickly
        exec_err = None
        try:
            pd.read_sql(query, conn)
        except Exception as e:
            exec_err = str(e)

        # If failed, try a simple correction pass via LLM
        if exec_err and not sql_text.startswith('-- agent_error'):
            llm = setup_azure_llm()
            correction_prompt = f"Correct this SQL for SQLite. Return only SQL.\nError: {exec_err}\nSQL:\n{query}"
            try:
                fixed = llm.invoke(correction_prompt)
                corrected = fixed.content if hasattr(fixed, 'content') else str(fixed)
                write_log('correction', corrected)
                try:
                    query = re.findall(r"^(?!.*\\bsql\\b)[\\s\\S]+?;", corrected.replace("```", "").strip() + ";", re.MULTILINE)[0]
                except Exception:
                    query = corrected.strip()
                    if not query.endswith(';'):
                        query += ';'
                # re-try execution silently
                try:
                    pd.read_sql(query, conn)
                except Exception:
                    pass
            except Exception:
                pass

        with open(predict_file_path, 'a') as f:
            f.write(query.replace('\n', ' ') + "\n")

        # Also write quick result/error to the per-question last dir for parity
        with open(os.path.join('question_last', f"{nl_query}.txt"), 'w') as f:
            f.write(f"{nl_query}\n\n")
            f.write("-" * 50 + "\n")
            f.write(query)

        break  # keep one question like original main.py

    conn.close()


if __name__ == '__main__':
    main()


