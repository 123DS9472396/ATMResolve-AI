import sqlite3
from src.storage import repository


def test_seeded_transactions_exist(isolated_db):
    repository.init_db()
    txns = repository.list_transactions()
    assert any(t["transaction_id"] == "TXN-0001" for t in txns)


def test_demo_txn_conflicting_events(isolated_db):
    repository.init_db()
    events = repository.get_atm_events("TXN-0005")
    types = {e["event_type"] for e in events}
    assert "cash_dispensed" in types and "dispense_failed" in types


def test_wal_journal_mode_is_enabled(isolated_db):
    """WAL mode must be active after get_connection() is called.

    This prevents writer-lock contention under concurrent requests.
    The PRAGMA is idempotent; applying it every open is safe.
    """
    with repository.get_connection() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal", f"Expected journal_mode=wal, got {mode!r}"
