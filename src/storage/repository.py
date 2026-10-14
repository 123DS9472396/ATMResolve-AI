from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.config import APPLICATION_DB_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@contextmanager
def get_connection():
    APPLICATION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APPLICATION_DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    complaint_text_redacted TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    status TEXT NOT NULL,
    intent_label TEXT,
    intent_confidence REAL,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    amount_inr REAL NOT NULL,
    transaction_time TEXT NOT NULL,
    account_debited INTEGER NOT NULL,
    cash_dispensed INTEGER NOT NULL,
    reversal_status TEXT NOT NULL,
    demo_note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS atm_events (
    event_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_time TEXT NOT NULL,
    event_reference TEXT NOT NULL,
    FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
);
CREATE TABLE IF NOT EXISTS case_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(case_id) REFERENCES cases(case_id)
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer_note TEXT NOT NULL DEFAULT '',
    edited_recommendation TEXT,
    decided_at TEXT NOT NULL,
    FOREIGN KEY(case_id) REFERENCES cases(case_id)
);
"""

SYNTHETIC_TRANSACTIONS = [
    ("TXN-0001", 5000, "2026-10-10T10:15:00+05:30", 1, 0, "pending", "Demo: debit recorded, dispense failed, reversal pending"),
    ("TXN-0002", 2000, "2026-10-10T11:20:00+05:30", 1, 1, "not_applicable", "Demo: successful dispense event exists"),
    ("TXN-0003", 1000, "2026-10-09T16:30:00+05:30", 1, 0, "completed", "Demo: dispense failed, reversal posted"),
    ("TXN-0004", 3000, "2026-10-08T09:45:00+05:30", 1, 0, "unknown", "Demo: only debit event exists; ATM log incomplete"),
    ("TXN-0005", 5000, "2026-10-07T18:05:00+05:30", 1, 0, "pending", "Demo: deliberately contradictory ATM events"),
    ("TXN-0006", 500, "2026-10-06T12:00:00+05:30", 1, 1, "not_applicable", "Demo: cash dispensed and no reversal expected"),
    ("TXN-0007", 2500, "2026-10-05T14:10:00+05:30", 1, 0, "pending", "Demo: failed dispense with incomplete reversal status"),
    ("TXN-0008", 1500, "2026-10-04T08:20:00+05:30", 1, 0, "completed", "Demo: failed dispense and completed reversal"),
]

def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"]
        if count == 0:
            for txn in SYNTHETIC_TRANSACTIONS:
                conn.execute(
                    "INSERT INTO transactions(transaction_id,amount_inr,transaction_time,account_debited,cash_dispensed,reversal_status,demo_note) VALUES(?,?,?,?,?,?,?)",
                    txn,
                )
            events = [
                ("EV-001-A", "TXN-0001", "debit_posted", "2026-10-10T10:15:00+05:30", "DEMO-JOURNAL-001"),
                ("EV-001-B", "TXN-0001", "dispense_failed", "2026-10-10T10:15:04+05:30", "DEMO-JOURNAL-001"),
                ("EV-002-A", "TXN-0002", "debit_posted", "2026-10-10T11:20:00+05:30", "DEMO-JOURNAL-002"),
                ("EV-002-B", "TXN-0002", "cash_dispensed", "2026-10-10T11:20:03+05:30", "DEMO-JOURNAL-002"),
                ("EV-003-A", "TXN-0003", "debit_posted", "2026-10-09T16:30:00+05:30", "DEMO-JOURNAL-003"),
                ("EV-003-B", "TXN-0003", "dispense_failed", "2026-10-09T16:30:05+05:30", "DEMO-JOURNAL-003"),
                ("EV-003-C", "TXN-0003", "reversal_posted", "2026-10-09T18:30:00+05:30", "DEMO-JOURNAL-003"),
                ("EV-004-A", "TXN-0004", "debit_posted", "2026-10-08T09:45:00+05:30", "DEMO-JOURNAL-004"),
                ("EV-005-A", "TXN-0005", "debit_posted", "2026-10-07T18:05:00+05:30", "DEMO-JOURNAL-005"),
                ("EV-005-B", "TXN-0005", "cash_dispensed", "2026-10-07T18:05:03+05:30", "DEMO-JOURNAL-005"),
                ("EV-005-C", "TXN-0005", "dispense_failed", "2026-10-07T18:05:04+05:30", "DEMO-JOURNAL-005"),
                ("EV-006-A", "TXN-0006", "debit_posted", "2026-10-06T12:00:00+05:30", "DEMO-JOURNAL-006"),
                ("EV-006-B", "TXN-0006", "cash_dispensed", "2026-10-06T12:00:02+05:30", "DEMO-JOURNAL-006"),
                ("EV-007-A", "TXN-0007", "debit_posted", "2026-10-05T14:10:00+05:30", "DEMO-JOURNAL-007"),
                ("EV-007-B", "TXN-0007", "dispense_failed", "2026-10-05T14:10:04+05:30", "DEMO-JOURNAL-007"),
                ("EV-008-A", "TXN-0008", "debit_posted", "2026-10-04T08:20:00+05:30", "DEMO-JOURNAL-008"),
                ("EV-008-B", "TXN-0008", "dispense_failed", "2026-10-04T08:20:03+05:30", "DEMO-JOURNAL-008"),
                ("EV-008-C", "TXN-0008", "reversal_posted", "2026-10-04T10:20:00+05:30", "DEMO-JOURNAL-008"),
            ]
            conn.executemany("INSERT INTO atm_events(event_id,transaction_id,event_type,event_time,event_reference) VALUES(?,?,?,?,?)", events)

def create_case(case_id: str, complaint_text: str, transaction_id: str) -> dict[str, Any]:
    init_db()
    timestamp = now_iso()
    with get_connection() as conn:
        conn.execute("INSERT INTO cases(case_id,complaint_text_redacted,transaction_id,status,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                     (case_id, complaint_text, transaction_id, "investigating", "{}", timestamp, timestamp))
    log_case_event(case_id, "intake", "case_created", {"transaction_id": transaction_id})
    return get_case(case_id) or {}

def update_case(case_id: str, **fields: Any) -> None:
    allowed = {"status", "intent_label", "intent_confidence", "result_json"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "result_json" in updates and not isinstance(updates["result_json"], str):
        updates["result_json"] = json.dumps(updates["result_json"], ensure_ascii=False)
    if not updates:
        return
    updates["updated_at"] = now_iso()
    assignments = ", ".join(f"{col}=?" for col in updates)
    with get_connection() as conn:
        conn.execute(f"UPDATE cases SET {assignments} WHERE case_id=?", [*updates.values(), case_id])

def get_case(case_id: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            result["result"] = json.loads(result.pop("result_json") or "{}")
        except json.JSONDecodeError:
            result["result"] = {}
            result.pop("result_json", None)
        return result

def list_cases(limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        try:
            item["result"] = json.loads(item.pop("result_json") or "{}")
        except json.JSONDecodeError:
            item["result"] = {}
            item.pop("result_json", None)
        results.append(item)
    return results

def get_transaction(transaction_id: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM transactions WHERE transaction_id=?", (transaction_id,)).fetchone()
    return dict(row) if row else None

def list_transactions() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT transaction_id, amount_inr, transaction_time, reversal_status, demo_note FROM transactions ORDER BY transaction_id").fetchall()]

def get_atm_events(transaction_id: str) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT event_id, transaction_id, event_type, event_time, event_reference FROM atm_events WHERE transaction_id=? ORDER BY event_time", (transaction_id,)).fetchall()
    return [dict(r) for r in rows]

def log_case_event(case_id: str, stage: str, event_type: str, details: dict[str, Any] | None = None) -> None:
    with get_connection() as conn:
        conn.execute("INSERT INTO case_events(case_id,stage,event_type,details_json,created_at) VALUES(?,?,?,?,?)",
                     (case_id, stage, event_type, json.dumps(details or {}, ensure_ascii=False), now_iso()))

def list_case_events(case_id: str) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT event_id,case_id,stage,event_type,details_json,created_at FROM case_events WHERE case_id=? ORDER BY event_id", (case_id,)).fetchall()
    out=[]
    for row in rows:
        item=dict(row)
        item["details"]=json.loads(item.pop("details_json") or "{}")
        out.append(item)
    return out

def save_approval(case_id: str, decision: str, reviewer_note: str = "", edited_recommendation: str | None = None) -> None:
    with get_connection() as conn:
        conn.execute("INSERT INTO approvals(case_id,decision,reviewer_note,edited_recommendation,decided_at) VALUES(?,?,?,?,?)",
                     (case_id, decision, reviewer_note, edited_recommendation, now_iso()))

def list_approvals(case_id: str) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM approvals WHERE case_id=? ORDER BY approval_id", (case_id,)).fetchall()]

if __name__ == "__main__":
    init_db()
    print(f"Initialized SQLite at {APPLICATION_DB_PATH}")
    print(f"Synthetic transactions available: {len(list_transactions())}")
