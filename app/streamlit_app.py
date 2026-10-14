from __future__ import annotations
import json
from pathlib import Path
import httpx
import streamlit as st
from src.config import API_BASE_URL, ROOT

st.set_page_config(page_title="ATMResolve AI", page_icon="🏦", layout="wide")
st.title("🏦 ATMResolve AI")
st.caption("Synthetic ATM complaint investigation copilot · local-first · human review required")
API = API_BASE_URL.rstrip("/")
TIMEOUT = httpx.Timeout(240.0, connect=5.0)

def api_get(path: str):
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(API + path)
        response.raise_for_status()
        return response.json()

def api_post(path: str, data: dict):
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(API + path, json=data)
        response.raise_for_status()
        return response.json()

def show_case(case: dict):
    result = case.get("result") or {}
    c1,c2,c3 = st.columns(3)
    c1.metric("Case status", case.get("status", "unknown"))
    c2.metric("Predicted intent", result.get("intent_label", case.get("intent_label") or "not available"))
    score = result.get("intent_confidence", case.get("intent_confidence"))
    c3.metric("Model score (not calibrated)", f"{score:.3f}" if isinstance(score,(float,int)) else "not available")
    st.write("**Complaint:**", case.get("complaint_text_redacted", ""))
    st.write("**Transaction reference:**", case.get("transaction_id", ""))
    t1,t2,t3,t4 = st.tabs(["Evidence & findings", "RAG citations", "LLM recommendation", "Audit trail"])
    with t1:
        st.subheader("Transaction record")
        st.json(result.get("transaction_record") or {"note":"No matching synthetic transaction was found."})
        st.subheader("ATM events")
        st.dataframe(result.get("atm_events", []), use_container_width=True, hide_index=True)
        st.subheader("Deterministic findings")
        for f in result.get("findings", []): st.warning(f)
        for e in result.get("errors", []): st.error(e)
    with t2:
        docs=result.get("retrieved_policy", [])
        if not docs: st.info("No policy evidence retrieved.")
        for doc in docs:
            st.markdown(f"**{doc.get('citation_id')} · {doc.get('title')}**")
            st.caption(f"Source type: {doc.get('source_type')} · Effective date: {doc.get('effective_date')}")
            st.write(doc.get("content", ""))
            st.markdown(f"Source: [{doc.get('source_url')}]({doc.get('source_url')})")
            st.divider()
    with t3:
        recommendation=result.get("recommendation", {})
        if recommendation:
            st.write("**Summary**")
            st.write(recommendation.get("case_summary", ""))
            st.write("**Proposed next step**")
            st.info(recommendation.get("recommended_next_step", ""))
            st.write("**Citations returned by the LLM**", recommendation.get("evidence_citations", []))
            st.write("**Generation mode**", recommendation.get("generation_mode", "unknown"))
            st.write("**Missing information**", recommendation.get("missing_information", []))
            st.json(recommendation)
        else: st.info("Recommendation is not available yet.")
    with t4:
        st.dataframe(st.session_state.get("current_events", []), use_container_width=True, hide_index=True)
        st.dataframe(st.session_state.get("current_approvals", []), use_container_width=True, hide_index=True)

tab_dash, tab_new, tab_review, tab_policy, tab_metrics = st.tabs(["Overview", "New investigation", "Review cases", "Policy search", "ML evaluation"])
with tab_dash:
    st.subheader("Case overview")
    try:
        cases=api_get("/cases")
        cols=st.columns(4)
        statuses=[c.get("status") for c in cases]
        cols[0].metric("Total cases", len(cases))
        cols[1].metric("Awaiting review", statuses.count("awaiting_review"))
        cols[2].metric("Approved", statuses.count("approved")+statuses.count("approved_edited"))
        cols[3].metric("Escalated", statuses.count("escalated"))
        st.dataframe([{k:c.get(k) for k in ("case_id","transaction_id","status","intent_label","intent_confidence","created_at")} for c in cases], use_container_width=True, hide_index=True)
        if not cases: st.info("No cases yet. Open New investigation and submit a synthetic complaint.")
    except Exception as exc:
        st.error(f"Could not connect to the API at {API}. Start it with `uvicorn api.main:app --reload`. Details: {exc}")
with tab_new:
    st.subheader("Submit a synthetic ATM complaint")
    st.warning("Use fictional information only. All transaction references below are seeded demo records.")
    try:
        txns=api_get("/transactions")
        txn_options=[t["transaction_id"] for t in txns] + ["TXN-MISSING-DEMO"]
        txn_map={t["transaction_id"]:t for t in txns}
    except Exception as exc:
        txn_options=["TXN-0001","TXN-MISSING-DEMO"]
        txn_map={}
        st.error(f"API not available: {exc}")
    with st.form("new_case_form"):
        txn_id=st.selectbox("Synthetic transaction reference", txn_options, format_func=lambda x: f"{x} — ₹{txn_map[x]['amount_inr']:,.0f} · {txn_map[x]['reversal_status']}" if x in txn_map else x)
        sample= "₹5,000 was debited but the ATM did not dispense cash. Please investigate the transaction and explain the next step."
        complaint=st.text_area("Complaint", value=sample, max_chars=2000, height=120)
        submitted=st.form_submit_button("Investigate complaint", type="primary")
    if submitted:
        if len(complaint.strip()) < 8: st.error("Please enter a more detailed complaint.")
        else:
            with st.spinner("Running classifier, local RAG, mock evidence lookups and local LLM. CPU inference may take a while..."):
                try:
                    response=api_post("/cases", {"complaint_text":complaint, "transaction_id":txn_id})
                    st.session_state["selected_case_id"]=response["case"]["case_id"]
                    st.session_state["current_events"]=response.get("events",[])
                    st.session_state["current_approvals"]=response.get("approvals",[])
                    st.success(f"Case created: {response['case']['case_id']} — status: {response['case']['status']}")
                    st.info("Open Review cases to inspect the evidence and make a human decision.")
                except Exception as exc: st.error(f"Investigation failed: {exc}")
with tab_review:
    st.subheader("Inspect and review an investigation")
    try:
        cases=api_get("/cases")
        if cases:
            ids=[c["case_id"] for c in cases]
            default_index=ids.index(st.session_state["selected_case_id"]) if st.session_state.get("selected_case_id") in ids else 0
            selected=st.selectbox("Case", ids, index=default_index)
            detail=api_get(f"/cases/{selected}")
            st.session_state["current_events"]=detail.get("events",[])
            st.session_state["current_approvals"]=detail.get("approvals",[])
            case=detail["case"]
            show_case(case)
            if case.get("status")=="awaiting_review":
                st.divider(); st.subheader("Human approval")
                decision=st.radio("Decision", ["approve","edit","reject","escalate"], horizontal=True)
                note=st.text_area("Reviewer note", key=f"note-{selected}")
                edited=""
                if decision=="edit": edited=st.text_area("Edited proposed next step", key=f"edit-{selected}")
                if st.button("Submit review decision", type="primary", key=f"review-{selected}"):
                    try:
                        result=api_post(f"/cases/{selected}/approval", {"decision":decision,"reviewer_note":note,"edited_recommendation":edited or None})
                        st.success(f"Recorded decision. New status: {result['case']['status']}")
                        st.rerun()
                    except Exception as exc: st.error(f"Could not save review: {exc}")
            else: st.info(f"This case is not awaiting review. Current status: {case.get('status')}")
        else: st.info("No cases available yet.")
    except Exception as exc: st.error(f"Unable to load review cases: {exc}")
with tab_policy:
    st.subheader("Search the local policy knowledge base")
    st.caption("The default index includes synthetic demo policy and a locally authored summary linked to RBI guidance. Verify official source text independently.")
    with st.form("policy_search_form"):
        query=st.text_input("Question", value="What evidence should be checked if an ATM debits an account but dispenses no cash?")
        go=st.form_submit_button("Search policies")
    if go:
        with st.spinner("Searching Chroma with local embeddings..."):
            try:
                results=api_post("/policy-search", {"query":query})["results"]
                for doc in results:
                    st.markdown(f"**{doc['citation_id']} · {doc['title']}**")
                    st.caption(f"{doc['source_type']} · {doc['effective_date']}")
                    st.write(doc["content"])
                    st.markdown(f"[Open source]({doc['source_url']})")
                    st.divider()
                if not results: st.info("No relevant passages found.")
            except Exception as exc: st.error(f"Policy search unavailable: {exc}. Run `python -m src.rag.ingest` first.")
with tab_metrics:
    st.subheader("Classifier evaluation")
    metrics_path=ROOT / "artifacts/ml/metrics.json"
    if metrics_path.exists():
        metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
        a,b,c=st.columns(3)
        a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        b.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
        c.metric("Test rows", metrics["test_rows"])
        st.json(metrics)
        st.caption("Values are read from the actual training run. BANKING77 scores do not guarantee performance on Indian ATM complaint text.")
    else:
        st.info("No trained metrics yet. Run `python scripts/download_banking77.py` and `python -m src.ml.train`.")
