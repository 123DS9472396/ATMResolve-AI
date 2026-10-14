from src.rules.dispute_rules import investigate_evidence

def test_missing_transaction_cannot_be_auto_resolved():
    findings, review = investigate_evidence(None, [], 0.95, 0.35)
    assert review is True
    assert any("Do not infer" in item for item in findings)

def test_every_recommendation_requires_human_review():
    txn = {"account_debited": 1, "cash_dispensed": 0, "reversal_status": "completed"}
    events = [{"event_type": "debit_posted"}, {"event_type": "dispense_failed"}, {"event_type": "reversal_posted"}]
    _, review = investigate_evidence(txn, events, 0.99, 0.35)
    assert review is True
