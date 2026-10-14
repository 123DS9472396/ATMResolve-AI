from src.storage import repository

def test_seeded_transactions_exist():
    repository.init_db()
    txns = repository.list_transactions()
    assert any(t["transaction_id"] == "TXN-0001" for t in txns)

def test_demo_txn_conflicting_events():
    repository.init_db()
    events = repository.get_atm_events("TXN-0005")
    types = {e["event_type"] for e in events}
    assert "cash_dispensed" in types and "dispense_failed" in types
