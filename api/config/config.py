import sqlite3
from pathlib import Path

API_DB_PATH = Path("./mnemis_api.db")


API_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id              TEXT PRIMARY KEY,
        email           TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        display_name    TEXT DEFAULT '',
        created_at      TEXT NOT NULL,
        is_active       INTEGER DEFAULT 1,
        daily_token_budget INTEGER DEFAULT 100000
    );

    CREATE TABLE IF NOT EXISTS token_usage (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id),
        date        TEXT NOT NULL,                      -- YYYY-MM-DD
        input_tokens  INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        cost_usd    REAL DEFAULT 0.0,
        UNIQUE(user_id, date)
    );

    CREATE INDEX IF NOT EXISTS idx_token_usage_user_date
        ON token_usage(user_id, date);
"""


def init_api_db():
    with sqlite3.connect(API_DB_PATH) as conn:
        conn.executescript(API_SCHEMA)
