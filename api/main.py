from __future__ import annotations
import uuid
from fastapi import FastAPI, HTTPException
from api.models import CaseCreate, ReviewRequest, PolicySearchRequest
from src.storage.repository import init_db, create_case, get_case, list_cases, list_case_events, list_approvals, get_transaction, list_transactions
from src.agents.graph import start_investigation, resume_investigation
from src.rag.retrieve import retrieve_policy

app = FastAPI(title="ATMResolve AI API", version="0.1.0", description="Local synthetic ATM dispute investigation prototype; not a real banking system.")

@app.on_event("startup")
def startup() -> None:
    init_db()

@app.get("/health")
def health():
    return {"status": "ok", "mode": "local_synthetic_demo"}

@app.get("/transactions")
def transactions():
    return list_transactions()

@app.post("/cases")
def submit_case(payload: CaseCreate):
    txn_id = payload.transaction_id.strip().upper()
    # A missing synthetic reference is allowed so the app can demonstrate a safe missing-evidence route.
    case_id = "CASE-" + uuid.uuid4().hex[:10].upper()
    try:
        create_case(case_id, payload.complaint_text.strip(), txn_id)
        start_investigation(case_id, payload.complaint_text.strip(), txn_id)
        case = get_case(case_id)
        return {"case": case, "events": list_case_events(case_id), "approvals": list_approvals(case_id)}
    except Exception as exc:
        current = get_case(case_id)
        if current:
            raise HTTPException(status_code=500, detail=f"Investigation failed after case creation: {type(exc).__name__}: {exc}") from exc
        raise HTTPException(status_code=500, detail=f"Could not create case: {type(exc).__name__}: {exc}") from exc

@app.get("/cases")
def cases():
    return list_cases()

@app.get("/cases/{case_id}")
def case_detail(case_id: str):
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"case": case, "events": list_case_events(case_id), "approvals": list_approvals(case_id)}

@app.post("/cases/{case_id}/approval")
def approve_case(case_id: str, payload: ReviewRequest):
    if payload.decision not in {"approve", "reject", "edit", "escalate"}:
        raise HTTPException(status_code=422, detail="decision must be approve, reject, edit or escalate")
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if case.get("status") != "awaiting_review":
        raise HTTPException(status_code=409, detail=f"Case status is {case.get('status')}; only awaiting_review cases can be resumed")
    if payload.decision == "edit" and not (payload.edited_recommendation or "").strip():
        raise HTTPException(status_code=422, detail="edited_recommendation is required when decision=edit")
    decision = payload.model_dump()
    try:
        resume_investigation(case_id, decision)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not resume workflow: {type(exc).__name__}: {exc}") from exc
    return {"case": get_case(case_id), "events": list_case_events(case_id), "approvals": list_approvals(case_id)}

@app.post("/policy-search")
def policy_search(payload: PolicySearchRequest):
    try:
        return {"query": payload.query, "results": retrieve_policy(payload.query, k=4)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Policy search unavailable: {type(exc).__name__}: {exc}") from exc
