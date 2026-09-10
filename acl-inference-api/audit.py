"""
SQLite Audit Trail & Clinical Telemetry Logger.
Maintains persistent record of all inference invocations for compliance and auditability.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

DB_PATH = Path(__file__).parent / "audit_logs.db"


def init_db(db_path: Path = DB_PATH):
    """Initializes the SQLite audit database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            exam_id TEXT,
            model TEXT NOT NULL,
            probability REAL NOT NULL,
            predicted_label TEXT NOT NULL,
            plane_logits_json TEXT NOT NULL,
            execution_time_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(
    model: str,
    probability: float,
    predicted_label: str,
    plane_logits: Dict[str, float],
    execution_time_ms: float,
    exam_id: Optional[str] = None,
    db_path: Path = DB_PATH
) -> int:
    """Logs a completed prediction event to SQLite."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audit_logs (
            timestamp, exam_id, model, probability, predicted_label, plane_logits_json, execution_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_iso,
            exam_id,
            model,
            float(probability),
            predicted_label,
            json.dumps(plane_logits),
            float(execution_time_ms)
        )
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_recent_logs(limit: int = 50, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent audit logs sorted chronologically descending."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, exam_id, model, probability, predicted_label, plane_logits_json, execution_time_ms
        FROM audit_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "exam_id": r["exam_id"],
            "model": r["model"],
            "probability": r["probability"],
            "predicted_label": r["predicted_label"],
            "plane_logits": json.loads(r["plane_logits_json"]),
            "execution_time_ms": r["execution_time_ms"]
        })
    return results
