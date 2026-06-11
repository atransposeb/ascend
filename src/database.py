"""
database.py – Lightweight SQLite storage for sessions and SAM feedback.

Uses aiosqlite for async FastAPI compatibility.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sessions.db"


def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        conn.execute("SELECT start_valence FROM sessions LIMIT 1")
    except sqlite3.OperationalError:
        print("Schema mismatch detected, resetting database tables...")
        conn.execute("DROP TABLE IF EXISTS feedback")
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS user_preferences")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id                TEXT PRIMARY KEY,
            email             TEXT NOT NULL DEFAULT '',
            provider          TEXT NOT NULL,
            provider_user_id  TEXT NOT NULL,
            name              TEXT NOT NULL DEFAULT '',
            avatar_url        TEXT NOT NULL DEFAULT '',
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider, provider_user_id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id        TEXT PRIMARY KEY,
            start_valence     REAL NOT NULL,
            start_arousal     REAL NOT NULL,
            target_valence    REAL NOT NULL,
            target_arousal    REAL NOT NULL,
            user_id           TEXT NOT NULL,
            path_type         TEXT NOT NULL,
            playlist          TEXT NOT NULL,
            metrics           TEXT,
            rejected_track_ids TEXT DEFAULT '[]',
            original_waypoints TEXT DEFAULT NULL,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL REFERENCES sessions(session_id),
            pre_valence   INTEGER NOT NULL,
            pre_arousal   INTEGER NOT NULL,
            post_valence  INTEGER NOT NULL,
            post_arousal  INTEGER NOT NULL,
            completed     BOOLEAN DEFAULT 1,
            skipped_tracks TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id        TEXT PRIMARY KEY,
            valence_bias   REAL DEFAULT 0.0,
            arousal_bias   REAL DEFAULT 0.0,
            feedback_count INTEGER DEFAULT 0
        );


        CREATE TABLE IF NOT EXISTS saved_journeys (
            id                TEXT PRIMARY KEY,
            user_id           TEXT NOT NULL,
            name              TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            start_valence     REAL NOT NULL,
            start_arousal     REAL NOT NULL,
            target_valence    REAL NOT NULL,
            target_arousal    REAL NOT NULL,
            path_type         TEXT NOT NULL,
            mood_start_json   TEXT NOT NULL,
            mood_target_json  TEXT NOT NULL,
            waypoints_json    TEXT NOT NULL,
            playlist_json     TEXT NOT NULL,
            metrics_json      TEXT,
            track_count       INTEGER NOT NULL,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    for col, default in [
        ("rejected_track_ids", "'[]'"),
        ("original_waypoints", "NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT {default}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()
    print(f"Database initialised at {DB_PATH}")


# ── User CRUD ──────────────────────────────────────────────────────────────


def create_user(
    provider: str,
    provider_user_id: str,
    email: str = "",
    name: str = "",
    avatar_url: str = "",
) -> dict:
    """Insert or return existing user by (provider, provider_user_id)."""
    import uuid
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE provider = ? AND provider_user_id = ?",
        (provider, provider_user_id),
    ).fetchone()
    if row:
        conn.close()
        return dict(row)

    user_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO users (id, email, provider, provider_user_id, name, avatar_url) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, email, provider, provider_user_id, name, avatar_url),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row)


def get_user_by_provider_id(user_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_sessions(user_id: str, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT session_id, start_valence, start_arousal, target_valence, target_arousal, "
        "       path_type, created_at "
        "FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_preferences(user_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row is None:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO user_preferences (user_id, valence_bias, arousal_bias, feedback_count) "
            "VALUES (?, 0.0, 0.0, 0)",
            (user_id,),
        )
        conn.commit()
        conn.close()
        return {"user_id": user_id, "valence_bias": 0.0, "arousal_bias": 0.0, "feedback_count": 0}
    return dict(row)


def update_user_preferences(user_id: str, valence_bias: float, arousal_bias: float) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE user_preferences "
        "SET valence_bias = ?, arousal_bias = ?, feedback_count = feedback_count + 1 "
        "WHERE user_id = ?",
        (valence_bias, arousal_bias, user_id),
    )
    conn.commit()
    conn.close()


def save_session(
    session_id: str,
    start_valence: float,
    start_arousal: float,
    target_valence: float,
    target_arousal: float,
    user_id: str,
    path_type: str,
    playlist_json: str,
    metrics_json: str,
    original_waypoints_json: Optional[str] = None,
) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions "
        "(session_id, start_valence, start_arousal, target_valence, target_arousal, "
        " user_id, path_type, playlist, metrics, rejected_track_ids, original_waypoints) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?)",
        (session_id, start_valence, start_arousal, target_valence, target_arousal,
         user_id, path_type, playlist_json, metrics_json, original_waypoints_json),
    )
    conn.commit()
    conn.close()


def add_rejected_track(session_id: str, track_id: str) -> None:
    """Append a track_id to the session's rejected_track_ids exclusion list."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT rejected_track_ids FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row:
        existing: list = json.loads(row["rejected_track_ids"] or "[]")
        if track_id not in existing:
            existing.append(track_id)
        conn.execute(
            "UPDATE sessions SET rejected_track_ids = ? WHERE session_id = ?",
            (json.dumps(existing), session_id),
        )
        conn.commit()
    conn.close()


def update_session_playlist(session_id: str, playlist_json: str, metrics_json: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET playlist = ?, metrics = ? WHERE session_id = ?",
        (playlist_json, metrics_json, session_id),
    )
    conn.commit()
    conn.close()


def get_session(session_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def save_feedback(
    session_id: str,
    pre_valence: int,
    pre_arousal: int,
    post_valence: int,
    post_arousal: int,
    completed: bool,
    skipped_tracks: list,
    user_id: str = "default_user",
) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO feedback "
        "(session_id, pre_valence, pre_arousal, post_valence, post_arousal, completed, skipped_tracks) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, pre_valence, pre_arousal, post_valence, post_arousal,
         completed, json.dumps(skipped_tracks)),
    )
    conn.commit()
    conn.close()

    # --- Online adaptation loop ---
    # Retrieve session details
    session = get_session(session_id)
    if not session:
        return

    # Convert 1-9 SAM self-reports to 0-1 continuous scale
    v_reported_pre = (pre_valence - 1) / 8.0
    v_reported_post = (post_valence - 1) / 8.0
    a_reported_pre = (pre_arousal - 1) / 8.0
    a_reported_post = (post_arousal - 1) / 8.0

    delta_v_reported = v_reported_post - v_reported_pre
    delta_a_reported = a_reported_post - a_reported_pre

    # Expected trajectory shift from the plan
    delta_v_target = session["target_valence"] - session["start_valence"]
    delta_a_target = session["target_arousal"] - session["start_arousal"]

    # Calculate adjustment errors
    error_v = delta_v_target - delta_v_reported
    error_a = delta_a_target - delta_a_reported

    # Fetch existing preferences
    prefs = get_user_preferences(user_id)
    learning_rate = 0.05
    
    # Update biases based on the error
    new_v_bias = prefs["valence_bias"] + learning_rate * error_v
    new_a_bias = prefs["arousal_bias"] + learning_rate * error_a

    # Clamp biases to sensible range [-0.2, 0.2] to prevent radical runaway shifts
    new_v_bias = max(-0.2, min(0.2, new_v_bias))
    new_a_bias = max(-0.2, min(0.2, new_a_bias))

    update_user_preferences(user_id, new_v_bias, new_a_bias)
    print(f"Updated preferences for '{user_id}': valence_bias={new_v_bias:.3f}, arousal_bias={new_a_bias:.3f}")


# ── Saved Journeys CRUD ──────────────────────────────────────────────────────


def save_journey(
    journey_id: str,
    user_id: str,
    name: str,
    session_id: str,
    start_valence: float,
    start_arousal: float,
    target_valence: float,
    target_arousal: float,
    path_type: str,
    mood_start_json: str,
    mood_target_json: str,
    waypoints_json: str,
    playlist_json: str,
    metrics_json: str | None,
    track_count: int,
) -> dict:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO saved_journeys "
        "(id, user_id, name, session_id, start_valence, start_arousal, "
        " target_valence, target_arousal, path_type, mood_start_json, "
        " mood_target_json, waypoints_json, playlist_json, metrics_json, track_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (journey_id, user_id, name, session_id, start_valence, start_arousal,
         target_valence, target_arousal, path_type, mood_start_json,
         mood_target_json, waypoints_json, playlist_json, metrics_json, track_count),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM saved_journeys WHERE id = ?", (journey_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def list_journeys(user_id: str, limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, session_id, start_valence, start_arousal, "
        "       target_valence, target_arousal, path_type, track_count, created_at "
        "FROM saved_journeys WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_journey(journey_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM saved_journeys WHERE id = ?", (journey_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_journey(journey_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM saved_journeys WHERE id = ?", (journey_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
