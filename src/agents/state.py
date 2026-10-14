from __future__ import annotations
from typing import Any, TypedDict

class InvestigationState(TypedDict, total=False):
    case_id: str
    complaint_text: str
    transaction_id: str
    intent_label: str
    intent_confidence: float
    transaction_record: dict[str, Any] | None
    atm_events: list[dict[str, Any]]
    retrieved_policy: list[dict[str, Any]]
    findings: list[str]
    errors: list[str]
    recommendation: dict[str, Any]
    review_decision: dict[str, Any]
    case_status: str
    requires_human_review: bool
