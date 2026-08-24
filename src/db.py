import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def get_config():
    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "loan_database"),
        "table": os.getenv("MYSQL_TABLE", "loan_sample"),
    }

def get_engine():
    c = get_config()
    if not c["password"]:
        raise ValueError("MYSQL_PASSWORD is empty. Copy .env.example to .env and enter your MySQL password.")
    url = f"mysql+pymysql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"
    return create_engine(url, pool_pre_ping=True)

def quote_identifier(name):
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return f"`{name}`"

def load_loans(columns=None, limit=None):
    c = get_config()
    cols = "*"
    if columns:
        cols = ", ".join(quote_identifier(x) for x in columns)
    lim = f" LIMIT {int(limit)}" if limit else ""
    query = f"SELECT {cols} FROM {quote_identifier(c['table'])}{lim}"
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)

def table_count():
    c = get_config()
    with get_engine().connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {quote_identifier(c['table'])}")).scalar())

def table_columns():
    c = get_config()
    with get_engine().connect() as conn:
        rows = conn.execute(text(f"SHOW COLUMNS FROM {quote_identifier(c['table'])}")).fetchall()
    return [r[0] for r in rows]
