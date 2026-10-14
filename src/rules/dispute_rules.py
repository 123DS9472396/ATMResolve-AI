from __future__ import annotations
from typing import Any

def investigate_evidence(transaction: dict[str, Any] | None, events: list[dict[str, Any]], confidence: float, threshold: float) -> tuple[list[str], bool]:
    findings: list[str] = []
    requires_review = False
    if confidence < threshold:
        findings.append("Intent model score is below the configured review threshold; treat classification as uncertain.")
        requires_review = True
    if transaction is None:
        findings.append("No synthetic transaction record matches the supplied reference. Do not infer that a debit occurred.")
        requires_review = True
        return findings, requires_review
    if int(transaction.get("account_debited", 0)) == 1:
        findings.append("Synthetic transaction record indicates an account debit.")
    else:
        findings.append("Synthetic transaction record does not indicate an account debit.")
    if int(transaction.get("cash_dispensed", 0)) == 1:
        findings.append("Synthetic transaction record indicates cash was dispensed.")
    else:
        findings.append("Synthetic transaction record indicates cash was not dispensed.")
    if not events:
        findings.append("ATM event log is missing; the evidence is incomplete.")
        requires_review = True
    event_types = [str(e.get("event_type", "")) for e in events]
    if "cash_dispensed" in event_types and "dispense_failed" in event_types:
        findings.append("ATM event log is contradictory: it contains both cash_dispensed and dispense_failed.")
        requires_review = True
    if transaction.get("reversal_status") in {"pending", "unknown"}:
        findings.append(f"Reversal status is {transaction.get('reversal_status')}; verify the current state before recommending closure.")
        requires_review = True
    elif transaction.get("reversal_status") == "completed":
        findings.append("Synthetic transaction record indicates the reversal is completed; verify the corresponding event before closure.")
    # Every proposed action requires a human reviewer in this portfolio prototype.
    return findings, True
