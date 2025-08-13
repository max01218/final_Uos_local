import sqlite3
import time
from typing import Optional
from app.core.settings import settings


def init_session_db():
    try:
        conn = sqlite3.connect(settings.session_db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_summary (
                session_id TEXT PRIMARY KEY,
                summary TEXT,
                updated_at REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id TEXT,
                ts REAL,
                weekly_goal TEXT,
                feasibility REAL,
                anxiety_level REAL
            );
            """
        )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def save_session_summary(session_id: str, summary: str):
    conn = sqlite3.connect(settings.session_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "REPLACE INTO session_summary(session_id, summary, updated_at) VALUES(?,?,?)",
            (session_id, summary, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def load_session_summary(session_id: str) -> str:
    conn = sqlite3.connect(settings.session_db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT summary FROM session_summary WHERE session_id=?", (session_id,))
        row = cur.fetchone()
        return row[0] if row else ""
    finally:
        conn.close()


def append_session_metrics(session_id: str, weekly_goal: Optional[str], feasibility: Optional[float], anxiety_level: Optional[float]):
    if weekly_goal is None and feasibility is None and anxiety_level is None:
        return
    conn = sqlite3.connect(settings.session_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO session_metrics(session_id, ts, weekly_goal, feasibility, anxiety_level) VALUES(?,?,?,?,?)",
            (session_id, time.time(), weekly_goal, feasibility, anxiety_level),
        )
        conn.commit()
    finally:
        conn.close()


