from src.rules.dispute_rules import investigate_evidence

def test_missing_transaction_requires_review():
    findings, review = investigate_evidence(None, [], 0.9, 0.35)
    assert review is True
    assert any("No synthetic transaction" in f for f in findings)

def test_contradictory_atm_events_require_review():
    transaction = {"account_debited": 1, "cash_dispensed": 0, "reversal_status": "pending"}
    events = [{"event_type": "cash_dispensed"}, {"event_type": "dispense_failed"}]
    findings, review = investigate_evidence(transaction, events, 0.9, 0.35)
    assert review is True
    assert any("contradictory" in f.lower() for f in findings)

def test_low_confidence_requires_review():
    transaction = {"account_debited": 1, "cash_dispensed": 0, "reversal_status": "pending"}
    findings, review = investigate_evidence(transaction, [{"event_type": "dispense_failed"}], 0.1, 0.35)
    assert review is True
    assert any("below the configured review threshold" in f for f in findings)
