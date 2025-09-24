import os
import sqlite3


def get_schema_from_sqlite(db_path: str) -> str:
    """Introspect SQLite DB and return schema as a formatted string."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_parts = []
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_defs = [f"{col[1]} {col[2]}" for col in columns]  # name + type
        schema_parts.append(f"{table}({', '.join(col_defs)})")

        # Foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        for fk in fks:
            schema_parts.append(f"FOREIGN KEY: {table}.{fk[3]} → {fk[2]}.{fk[4]}")

    conn.close()
    return "\n".join(schema_parts)


def load_schema() -> str:
    """
    Load schema string for NL2SQL pipeline.
    Priority:
    1. DB_SCHEMA_FILE env var → read text
    2. SQLITE_DB_FILE env var → introspect DB dynamically
    """
    schema_file = os.getenv("DB_SCHEMA_FILE")
    sqlite_db_file = os.getenv("SQLITE_DB_FILE")

    if schema_file and os.path.exists(schema_file):
        with open(schema_file, "r") as f:
            return f.read()

    if sqlite_db_file and os.path.exists(sqlite_db_file):
        return get_schema_from_sqlite(sqlite_db_file)

    raise FileNotFoundError(
        "No schema source found. Please set DB_SCHEMA_FILE or SQLITE_DB_FILE."
    )
