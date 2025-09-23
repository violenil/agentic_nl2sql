import os
import sqlite3
from typing import Any, Dict, List, Optional


def _get_db_path(explicit_path: Optional[str] = None) -> str:
    """
    Resolve the SQLite DB path either from an explicit argument or the SQLITE_DB_FILE env var.
    """
    db_path = explicit_path or os.getenv("SQLITE_DB_FILE")
    if not db_path or not os.path.exists(db_path):
        raise FileNotFoundError(
            "SQLite DB not found. Pass db_path explicitly or set env var SQLITE_DB_FILE."
        )
    return db_path


class ExperimentAnalyzer:
    """
    Runs a series of checks on a generated SQL query against a SQLite DB:
    - Syntax check via EXPLAIN
    - Execution check (with fetch sample)
    - Lightweight semantic hints (pattern-based)
    """

    def __init__(self, db_path: Optional[str] = None, sample_rows: int = 25, timeout_sec: int = 5):
        self.db_path = _get_db_path(db_path)
        self.sample_rows = int(sample_rows)
        self.timeout_sec = int(timeout_sec)

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None keeps autocommit semantics; set timeout to avoid long locks
        return sqlite3.connect(self.db_path, timeout=self.timeout_sec)

    def _syntax_ok(self, conn: sqlite3.Connection, sql: str, report: Dict[str, Any]) -> bool:
        try:
            # SQLite doesn't have a "parse only" mode; EXPLAIN is a good proxy for syntax validity.
            conn.execute("EXPLAIN " + sql)
            report["syntax_ok"] = True
            return True
        except Exception as e:
            report["syntax_ok"] = False
            report["error"] = f"Syntax error: {e}"
            return False

    def _execute(self, conn: sqlite3.Connection, sql: str, report: Dict[str, Any]) -> bool:
        try:
            cur = conn.cursor()
            cur.execute(sql)
            # Safely fetch a sample to avoid huge materializations
            rows = cur.fetchmany(self.sample_rows)
            report["exec_ok"] = True
            report["row_count_sample"] = len(rows)
            report["columns"] = [d[0] for d in cur.description] if cur.description else []
            # Convert rows to plain python types (lists) for JSON-serializability
            report["rows_sample"] = [list(r) for r in rows]
            return True
        except Exception as e:
            report["exec_ok"] = False
            report["error"] = f"Execution error: {e}"
            return False

    def _semantic_hints(self, question: str, sql: str) -> Dict[str, Any]:
        """
        Very lightweight heuristics to help the Critic.
        Add more rules as needed (e.g., date ranges, GROUP BY when 'per', etc.).
        """
        q_lower = question.lower()
        sql_lower = sql.lower()

        hints: List[str] = []
        if "how many" in q_lower and "count(" not in sql_lower:
            hints.append("Question suggests COUNT but SQL lacks COUNT().")
        if "between" in q_lower and (" between " not in sql_lower and ">" not in sql_lower and "<" not in sql_lower):
            hints.append("Question suggests range filter, but SQL has no BETWEEN/>/<.")
        if "join" in q_lower and " join " not in sql_lower:
            hints.append("Question implies combining entities; SQL has no JOIN.")
        if "list" in q_lower and "select" not in sql_lower:
            hints.append("Question suggests listing values; SQL may be missing SELECT columns.")

        return {
            "semantic_hint": hints[0] if hints else None,
            "semantic_hints": hints,
        }

    def analyze(self, question: str, sql: str) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "syntax_ok": False,
            "exec_ok": False,
            "row_count_sample": 0,
            "rows_sample": [],
            "columns": [],
            "error": None,
        }

        conn = self._connect()
        try:
            if not self._syntax_ok(conn, sql, report):
                # If syntax fails, we still add semantic hints for better critique
                report.update(self._semantic_hints(question, sql))
                return report

            # Syntax fine → try execution
            _ = self._execute(conn, sql, report)
            report.update(self._semantic_hints(question, sql))
            return report
        finally:
            conn.close()
