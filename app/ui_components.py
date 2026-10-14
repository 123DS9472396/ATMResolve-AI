"""UI components and renderers for ATMResolve AI Streamlit application.

Provides unified, robust rendering for:
- System health monitoring (FastAPI, SQLite, Ollama, Chroma)
- Overview dashboard with dispute metrics, workflow diagram, and transaction guide
- New complaint investigation intake form
- Case inspection with evidence comparison, deterministic findings, RAG citations,
  and local LLM synthesis
- Human review action form with unselected decision radio and confirmed updates
- Chronological investigation timeline from persistent backend events
- Policy RAG search and ML evaluation benchmarks
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import httpx
import pandas as pd
import streamlit as st
from src.config import API_BASE_URL, ROOT

API = API_BASE_URL.rstrip("/")
TIMEOUT = httpx.Timeout(240.0, connect=5.0)
HEALTH_TIMEOUT = httpx.Timeout(5.0, connect=3.0)


# ---------------------------------------------------------------------------
# API Client Helpers
# ---------------------------------------------------------------------------

def api_get(path: str) -> Any:
    """Make a GET request to the backend API."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{API}{path}")
        response.raise_for_status()
        return response.json()


def api_post(path: str, data: dict[str, Any]) -> Any:
    """Make a POST request to the backend API."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(f"{API}{path}", json=data)
        response.raise_for_status()
        return response.json()


def fetch_health() -> dict[str, Any]:
    """Probe the backend health endpoint and return structured status."""
    try:
        with httpx.Client(timeout=HEALTH_TIMEOUT) as client:
            resp = client.get(f"{API}/health")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "reachable": True,
                    "status": data.get("status", "unknown"),
                    "mode": data.get("mode", "local_synthetic_demo"),
                    "components": data.get("components", {}),
                    "error": None,
                }
            return {
                "reachable": True,
                "status": "degraded",
                "mode": "unknown",
                "components": {},
                "error": f"HTTP {resp.status_code}",
            }
    except Exception as exc:
        return {
            "reachable": False,
            "status": "unreachable",
            "mode": "offline",
            "components": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def format_iso_timestamp(iso_str: str | None) -> str:
    """Format an ISO timestamp into a human-readable string."""
    if not iso_str:
        return "—"
    try:
        clean_str = str(iso_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(iso_str)


# ---------------------------------------------------------------------------
# Sidebar System Health Component
# ---------------------------------------------------------------------------

def render_sidebar_health() -> dict[str, Any]:
    """Render the persistent system health status card in the sidebar."""
    st.sidebar.markdown("### 🏥 System Status")
    health = fetch_health()

    if not health["reachable"]:
        st.sidebar.error("🔴 **Backend API Offline**")
        st.sidebar.caption(
            f"Cannot connect to `{API}`.\n\n"
            "Start the API server in a terminal:\n"
            "```powershell\n"
            "uvicorn api.main:app --reload\n"
            "```"
        )
    elif health["status"] == "ok":
        st.sidebar.success("🟢 **All Services Operational**")
    else:
        st.sidebar.warning("🟡 **System Degraded**")

    with st.sidebar.expander("Component Diagnostics", expanded=(health["status"] != "ok")):
        if health["reachable"]:
            comps = health.get("components", {})
            db_status = comps.get("database", {}).get("status", "unknown")
            ollama_status = comps.get("ollama", {}).get("status", "unknown")
            chroma_status = comps.get("chroma", {}).get("status", "unknown")

            db_icon = "🟢" if db_status == "ok" else "🔴"
            ollama_icon = "🟢" if ollama_status == "ok" else "🟡"
            chroma_icon = "🟢" if chroma_status == "ok" else "🟡"

            st.markdown(f"{db_icon} **SQLite DB:** `{db_status}`")
            st.markdown(f"{ollama_icon} **Ollama LLM:** `{ollama_status}`")
            if ollama_status != "ok":
                st.caption("Tip: Start Ollama with `ollama run qwen3.5:2b`")
            st.markdown(f"{chroma_icon} **Chroma RAG:** `{chroma_status}`")
            if chroma_status != "ok":
                st.caption("Tip: Run `python -m src.rag.ingest`")
        else:
            st.caption(f"Error: {health.get('error')}")

    if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

    st.sidebar.divider()
    return health


# ---------------------------------------------------------------------------
# Page 1: Overview & Workflow Dashboard
# ---------------------------------------------------------------------------

def render_overview() -> None:
    """Render the main Overview dashboard for first-time and returning visitors."""
    st.title("🏦 ATMResolve AI — Dispute Investigation Dashboard")
    st.caption("Local-first ATM dispute reconciliation copilot · Powered by LangGraph, Chroma RAG & Local Qwen LLM")

    # Top Metrics
    try:
        cases = api_get("/cases")
    except Exception as exc:
        st.error(f"Could not load cases from backend API at `{API}`. Please verify the server is running. Details: {exc}")
        cases = []

    statuses = [c.get("status") for c in cases]
    total_count = len(cases)
    awaiting_count = statuses.count("awaiting_review")
    approved_count = statuses.count("approved") + statuses.count("approved_edited")
    escalated_count = statuses.count("escalated")
    rejected_count = statuses.count("rejected")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Cases", total_count)
    m2.metric("Awaiting Review", awaiting_count, delta="Action Required" if awaiting_count > 0 else None, delta_color="inverse")
    m3.metric("Approved", approved_count)
    m4.metric("Escalated", escalated_count)
    m5.metric("Rejected", rejected_count)

    st.divider()

    # Workflow Architecture Section
    st.subheader("🧭 ATM Dispute Resolution Architecture")
    st.markdown(
        """
        ATMResolve AI orchestrates an end-to-end investigation workflow where **deterministic financial rules take precedence over AI**,
        and every customer-impacting recommendation **strictly requires authorized human approval**.
        """
    )

    steps = [
        ("1. Intake & Validation", "Customer complaint text and synthetic transaction ID are verified and registered in SQLite.", "📥"),
        ("2. Intent Classification", "PolyAI BANKING77 / India ATM TF-IDF classifier predicts intent and outputs confidence.", "🏷️"),
        ("3. Policy Retrieval (RAG)", "Chroma vector store retrieves RBI Master Directions and bank procedural rules with traceable citations.", "📚"),
        ("4. Evidence Verification", "Queries core banking debit records and ATM terminal Electronic Journal (EJ) dispense logs.", "💳"),
        ("5. Deterministic Rules", "Evaluates debit status, dispense failures, contradictory events, and reversal deadlines.", "⚖️"),
        ("6. Local LLM Synthesis", "Local Qwen model synthesizes evidence into a structured case summary with source citations.", "🤖"),
        ("7. Human Review Gate", "LangGraph interrupts execution, awaiting human review to Approve, Edit, Reject, or Escalate.", "👤"),
        ("8. Audit Trail & Closure", "All stage events, timestamps, and human signatures are immutably logged to SQLite.", "📜"),
    ]

    cols = st.columns(4)
    for idx, (title, desc, icon) in enumerate(steps[:4]):
        with cols[idx]:
            st.info(f"**{icon} {title}**\n\n{desc}")

    cols2 = st.columns(4)
    for idx, (title, desc, icon) in enumerate(steps[4:]):
        with cols2[idx]:
            st.info(f"**{icon} {title}**\n\n{desc}")

    st.divider()

    # Seeded Test Scenarios Reference Guide
    with st.expander("📖 Seeded Demo Transactions & Test Scenarios Reference", expanded=False):
        st.markdown(
            """
            The local database seeds 8 realistic synthetic transaction scenarios to test deterministic dispute rules:
            """
        )
        scenarios = [
            {"Transaction": "TXN-0001", "Amount": "₹5,000", "Outcome": "Debit recorded, dispense failed, reversal pending", "Tests": "Auto-reversal delay & RBI T+5 compensation checks"},
            {"Transaction": "TXN-0002", "Amount": "₹2,000", "Outcome": "Debit recorded, cash dispensed successfully", "Tests": "Valid dispense verification (fraud/false claim detection)"},
            {"Transaction": "TXN-0003", "Amount": "₹1,000", "Outcome": "Debit recorded, dispense failed, reversal posted", "Tests": "Reversal completed confirmation"},
            {"Transaction": "TXN-0004", "Amount": "₹3,000", "Outcome": "Debit recorded, ATM journal events missing", "Tests": "Incomplete evidence handling (mandatory escalation)"},
            {"Transaction": "TXN-0005", "Amount": "₹5,000", "Outcome": "Both cash_dispensed and dispense_failed logged", "Tests": "Contradictory event detection rule"},
            {"Transaction": "TXN-0006", "Amount": "₹500", "Outcome": "Completed dispense, no reversal expected", "Tests": "Normal withdrawal verification"},
            {"Transaction": "TXN-0007", "Amount": "₹2,500", "Outcome": "Failed dispense, reversal status pending", "Tests": "Pending reversal tracking"},
            {"Transaction": "TXN-0008", "Amount": "₹1,500", "Outcome": "Failed dispense, reversal completed", "Tests": "Completed reversal resolution"},
            {"Transaction": "TXN-MISSING-DEMO", "Amount": "N/A", "Outcome": "Non-existent transaction reference", "Tests": "Missing reference rule (prevents unverified debits)"},
        ]
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

    # Recent Cases Table
    st.subheader("📋 Registered Investigation Cases")
    if cases:
        table_data = []
        for c in cases:
            table_data.append({
                "Case ID": c.get("case_id"),
                "Transaction Reference": c.get("transaction_id"),
                "Status": c.get("status"),
                "Predicted Intent": c.get("intent_label") or "—",
                "Confidence": f"{c.get('intent_confidence', 0):.3f}" if c.get("intent_confidence") is not None else "—",
                "Created At": format_iso_timestamp(c.get("created_at")),
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.info("No cases registered yet. Switch to **New Investigation** to submit your first synthetic complaint.")


# ---------------------------------------------------------------------------
# Page 2: New Investigation Form
# ---------------------------------------------------------------------------

def render_new_investigation() -> None:
    """Render the complaint submission form."""
    st.title("➕ Submit a Synthetic ATM Complaint")
    st.caption("Initiates an automated investigation across ML classification, RAG retrieval, evidence matching, deterministic rules, and LLM drafting.")

    st.warning("⚠️ **Synthetic Portfolio Prototype:** Use fictional transaction data only. Never input real customer PII or actual banking credentials.")

    try:
        txns = api_get("/transactions")
        txn_options = [t["transaction_id"] for t in txns] + ["TXN-MISSING-DEMO"]
        txn_map = {t["transaction_id"]: t for t in txns}
    except Exception as exc:
        st.error(f"Could not load transaction reference list from API: {exc}")
        txn_options = ["TXN-0001", "TXN-0002", "TXN-0003", "TXN-0004", "TXN-0005", "TXN-MISSING-DEMO"]
        txn_map = {}

    with st.form("new_investigation_form"):
        st.markdown("#### 1. Transaction Reference")
        selected_txn = st.selectbox(
            "Select Synthetic Transaction",
            txn_options,
            format_func=lambda x: f"{x} — ₹{txn_map[x]['amount_inr']:,.0f} ({txn_map[x]['reversal_status']}) · {txn_map[x]['demo_note']}" if x in txn_map else f"{x} (Simulate missing CBS record)",
            help="Choose a pre-seeded synthetic transaction scenario to investigate."
        )

        st.markdown("#### 2. Customer Complaint Narrative")
        sample_complaint = (
            "₹5,000 was debited from my account at the ATM, but the machine made a grinding noise "
            "and dispensed no cash. The screen showed transaction failed. Please investigate and refund my money."
        )
        complaint_text = st.text_area(
            "Customer Statement",
            value=sample_complaint,
            height=130,
            max_chars=2000,
            help="Enter the simulated customer complaint narrative (at least 8 characters)."
        )

        submitted = st.form_submit_button("🚀 Run Full Investigation", type="primary", use_container_width=True)

    if submitted:
        if len(complaint_text.strip()) < 8:
            st.error("Please enter a more detailed complaint narrative (minimum 8 characters).")
        else:
            with st.spinner("🤖 Running LangGraph pipeline: Classifying Intent → Querying Chroma RAG → Verifying Evidence → Evaluating Rules → Drafting LLM Synthesis..."):
                try:
                    payload = {"complaint_text": complaint_text.strip(), "transaction_id": selected_txn}
                    response = api_post("/cases", payload)
                    created_case = response.get("case", {})
                    case_id = created_case.get("case_id")
                    status = created_case.get("status")

                    st.session_state["selected_case_id"] = case_id
                    st.success(f"✅ Case `{case_id}` created successfully with status: **{status}**")
                    st.info("The investigation is paused at the human review gate. Navigate to **Case Review & Actions** to inspect the evidence and record your decision.")
                except Exception as exc:
                    st.error(f"Investigation execution failed: {exc}")


# ---------------------------------------------------------------------------
# Page 3: Case Review & Decisions (with complete Persistent Evidence Display)
# ---------------------------------------------------------------------------

def render_case_review(target_case_id: str | None = None) -> None:
    """Render the case review interface with complete persistent backend data."""
    st.title("📋 Case Review & Human Decision Gate")
    st.caption("Inspect grounded evidence, deterministic findings, RAG citations, and authorize human disposition.")

    try:
        cases = api_get("/cases")
    except Exception as exc:
        st.error(f"Could not load cases from API: {exc}")
        return

    if not cases:
        st.info("No cases exist yet. Please submit an investigation first via **New Investigation**.")
        return

    case_ids = [c["case_id"] for c in cases]
    default_id = target_case_id or st.session_state.get("selected_case_id")
    default_idx = case_ids.index(default_id) if default_id in case_ids else 0

    c_select, c_refresh = st.columns([4, 1])
    with c_select:
        selected_case_id = st.selectbox("Select Case to Inspect", case_ids, index=default_idx)
    with c_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Case", use_container_width=True):
            st.rerun()

    # Always fetch full case details directly from the persistent backend API!
    try:
        detail = api_get(f"/cases/{selected_case_id}")
    except Exception as exc:
        st.error(f"Could not fetch case `{selected_case_id}`: {exc}")
        return

    case = detail.get("case", {})
    events = detail.get("events", [])
    approvals = detail.get("approvals", [])
    result = case.get("result", {})

    status = case.get("status", "unknown")

    # Header Card with Status Indicator
    st.markdown(
        f"""
        ### Case `{case.get('case_id')}`
        **Status:** `{status.upper()}` &nbsp;|&nbsp;
        **Transaction Reference:** `{case.get('transaction_id')}` &nbsp;|&nbsp;
        **Created:** {format_iso_timestamp(case.get('created_at'))}
        """
    )

    # Customer Statement
    with st.container():
        st.markdown("**Customer Complaint Statement:**")
        st.info(f"🗣️ *\"{case.get('complaint_text_redacted', '')}\"*")

    # Investigation Tabs
    t_findings, t_rag, t_llm, t_timeline, t_technical = st.tabs([
        "🔎 Evidence & Rules",
        "📚 Policy Citations (RAG)",
        "🤖 AI Synthesis & Next Step",
        "📜 Audit Timeline",
        "🛠️ Technical Details",
    ])

    # Tab 1: Evidence & Rules
    with t_findings:
        st.markdown("#### 1. Machine Learning Intent Classification")
        ml_col1, ml_col2, ml_col3 = st.columns(3)
        intent_label = result.get("intent_label") or case.get("intent_label") or "unavailable"
        confidence = result.get("intent_confidence") or case.get("intent_confidence") or 0.0

        ml_col1.metric("Predicted Intent", intent_label)
        ml_col2.metric("Confidence Score", f"{confidence:.3f}")
        is_calibrated = "Yes (Sigmoid calibrated)" if result.get("confidence_is_calibrated") else "No (BANKING77 raw score)"
        ml_col3.metric("Calibration Status", is_calibrated)

        st.markdown("#### 2. Evidence Verification: CBS Record vs ATM Journal")
        ev_col1, ev_col2 = st.columns(2)

        with ev_col1:
            st.markdown("**Core Banking System (CBS) Record:**")
            txn = result.get("transaction_record")
            if txn:
                st.markdown(f"- **Amount:** ₹{txn.get('amount_inr', 0):,.2f}")
                st.markdown(f"- **Timestamp:** {format_iso_timestamp(txn.get('transaction_time'))}")
                st.markdown(f"- **Account Debited:** {'✅ Yes' if txn.get('account_debited') == 1 else '❌ No'}")
                st.markdown(f"- **Cash Dispensed:** {'✅ Yes' if txn.get('cash_dispensed') == 1 else '❌ No'}")
                st.markdown(f"- **Reversal Status:** `{txn.get('reversal_status', 'none')}`")
                st.caption(f"Note: {txn.get('demo_note', '')}")
            else:
                st.error("❌ No matching transaction was found in the core banking system database.")

        with ev_col2:
            st.markdown("**ATM Terminal Electronic Journal (EJ) Events:**")
            atm_events = result.get("atm_events", [])
            if atm_events:
                event_df = pd.DataFrame([{
                    "Event Type": e.get("event_type"),
                    "Event Time": format_iso_timestamp(e.get("event_time")),
                    "Journal Ref": e.get("event_reference"),
                } for e in atm_events])
                st.dataframe(event_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No electronic journal events recorded for this transaction reference.")

        st.markdown("#### 3. Deterministic Rules Engine Findings")
        findings = result.get("findings", [])
        errors = result.get("errors", [])

        if findings:
            for f in findings:
                st.warning(f"⚖️ {f}")
        else:
            st.info("No deterministic rule violations detected.")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")

    # Tab 2: RAG Policy Citations
    with t_rag:
        st.markdown("#### Retrieved Official & Internal Policy Citations")
        docs = result.get("retrieved_policy", [])
        if not docs:
            st.info("No policy documents retrieved for this investigation.")
        else:
            for d in docs:
                with st.container():
                    st.markdown(f"##### 📖 {d.get('citation_id', 'POL')} · {d.get('title', 'Policy Passage')}")
                    st.caption(f"**Source Type:** `{d.get('source_type')}` | **Effective Date:** `{d.get('effective_date')}`")
                    st.markdown(f"> {d.get('content', '')}")
                    url = d.get("source_url", "#")
                    st.markdown(f"🔗 Source Reference: [{url}]({url})")
                    st.divider()

    # Tab 3: Local LLM Synthesis
    with t_llm:
        st.markdown("#### Local Qwen LLM Investigation Synthesis")
        rec = result.get("recommendation", {})
        if rec:
            st.markdown("##### 📝 Case Summary")
            st.write(rec.get("case_summary", "No summary generated."))

            st.markdown("##### 💡 Recommended Next Step")
            st.success(f"**Proposed Disposition:** {rec.get('recommended_next_step', 'Manual investigation required.')}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Evidence Citations Referenced:**")
                citations = rec.get("evidence_citations", [])
                if citations:
                    st.write(", ".join(f"`{c}`" for c in citations))
                else:
                    st.write("None cited.")

            with c2:
                st.markdown("**Generation Mode:**")
                st.code(rec.get("generation_mode", "unknown"))

            missing = rec.get("missing_information", [])
            if missing:
                st.markdown("**Identified Missing Information:**")
                for m in missing:
                    st.markdown(f"- ❓ {m}")
        else:
            st.info("No LLM recommendation has been drafted yet.")

    # Tab 4: Audit Timeline
    with t_timeline:
        st.markdown("#### Investigation Event Timeline")
        render_timeline_view(events, approvals)

    # Tab 5: Technical Details (Collapsed raw JSON)
    with t_technical:
        st.markdown("#### Raw Case Payload")
        st.caption("Complete technical state stored in SQLite database.")
        st.json(case)

    # -----------------------------------------------------------------------
    # Human Review Action Form (Requires Explicit Selection)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("👤 Human Review & Decision Gate")

    if status == "awaiting_review":
        st.info("This investigation is paused at the LangGraph interrupt node. Review the evidence above and record your decision.")

        with st.form(f"review_form_{selected_case_id}"):
            # Notice index=None: NO decision is pre-selected!
            decision = st.radio(
                "Authorized Review Decision (Mandatory)",
                ["approve", "edit", "reject", "escalate"],
                index=None,
                horizontal=True,
                format_func=lambda x: {
                    "approve": "✅ Approve (Accept proposed recommendation)",
                    "edit": "✏️ Edit (Modify proposed next step before approval)",
                    "reject": "❌ Reject (Decline customer claim)",
                    "escalate": "🚨 Escalate (Transfer to Senior Dispute Team)",
                }.get(x, x),
                help="You must explicitly select one decision option."
            )

            reviewer_note = st.text_area(
                "Reviewer Note / Operational Rationale",
                placeholder="Enter justification for the audit trail (e.g., 'Dispense failure confirmed by journal EV-001-B; proceeding with auto-credit').",
                max_chars=1000,
            )

            current_rec = rec.get("recommended_next_step", "")
            edited_rec = st.text_area(
                "Edited Proposed Next Step (Required only if decision is 'Edit')",
                value=current_rec,
                placeholder="Enter the revised recommendation text if editing...",
                max_chars=2000,
            )

            submit_review = st.form_submit_button("💾 Commit Authorized Decision", type="primary", use_container_width=True)

        if submit_review:
            if not decision:
                st.error("⚠️ Please select a decision option (Approve, Edit, Reject, or Escalate) before submitting.")
            elif decision == "edit" and not (edited_rec or "").strip():
                st.error("⚠️ An edited recommendation text is required when selecting 'Edit'.")
            else:
                try:
                    payload = {
                        "decision": decision,
                        "reviewer_note": reviewer_note.strip(),
                        "edited_recommendation": edited_rec.strip() if decision == "edit" else None,
                    }
                    res = api_post(f"/cases/{selected_case_id}/approval", payload)
                    updated_status = res.get("case", {}).get("status", "updated")
                    st.success(f"✅ Decision `{decision}` recorded successfully! New case status: **{updated_status}**")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to record review decision: {exc}")
    else:
        st.success(f"This case has been resolved with status: **{status.upper()}**.")
        if approvals:
            st.markdown("**Recorded Human Decisions:**")
            for a in approvals:
                st.markdown(
                    f"- **Decision:** `{a.get('decision')}` | **By:** Reviewer | **At:** {format_iso_timestamp(a.get('decided_at'))}\n"
                    f"  - *Note:* {a.get('reviewer_note') or 'None'}\n"
                    + (f"  - *Edited Recommendation:* {a.get('edited_recommendation')}\n" if a.get('edited_recommendation') else "")
                )


# ---------------------------------------------------------------------------
# Page 4: Audit Trail & Timeline (Dedicated Component)
# ---------------------------------------------------------------------------

def render_timeline_view(events: list[dict[str, Any]], approvals: list[dict[str, Any]]) -> None:
    """Render a clean, chronological timeline of case events and human decisions."""
    if not events:
        st.info("No recorded events for this case.")
        return

    stage_icons = {
        "intake": "📥",
        "validation": "✅",
        "classification": "🏷️",
        "rag": "📚",
        "evidence": "💳",
        "rules": "⚖️",
        "llm": "🤖",
        "review": "👤",
    }

    st.markdown("##### Chronological Workflow Execution Log")
    for idx, ev in enumerate(events, start=1):
        stage = ev.get("stage", "general")
        icon = stage_icons.get(stage, "🔹")
        event_type = ev.get("event_type", "event")
        ts = format_iso_timestamp(ev.get("created_at"))
        details = ev.get("details", {})

        with st.container():
            c_icon, c_info = st.columns([1, 11])
            with c_icon:
                st.markdown(f"### {icon}")
            with c_info:
                st.markdown(f"**Step {idx}: `{event_type}`** &nbsp;·&nbsp; Stage: `{stage}` &nbsp;·&nbsp; `{ts}`")
                if details:
                    details_str = ", ".join(f"`{k}`: {v}" for k, v in details.items() if k != "sources")
                    st.caption(details_str)
                    if "sources" in details and details["sources"]:
                        st.caption(f"Sources cited: {', '.join(str(s) for s in details['sources'])}")
            st.divider()

    if approvals:
        st.markdown("##### Human Review Sign-Offs")
        for app in approvals:
            st.success(
                f"👤 **Decision: `{app.get('decision')}`** at `{format_iso_timestamp(app.get('decided_at'))}`\n\n"
                f"**Reviewer Note:** {app.get('reviewer_note') or '—'}\n\n"
                + (f"**Edited Next Step:** {app.get('edited_recommendation')}" if app.get('edited_recommendation') else "")
            )


def render_audit_log_page() -> None:
    """Render the standalone Audit Log page."""
    st.title("📜 Audit Trail & Investigation History")
    st.caption("Inspect immutable workflow execution events, model predictions, rule evaluations, and human sign-offs.")

    try:
        cases = api_get("/cases")
    except Exception as exc:
        st.error(f"Could not load cases from API: {exc}")
        return

    if not cases:
        st.info("No cases registered yet.")
        return

    case_ids = [c["case_id"] for c in cases]
    selected = st.selectbox("Select Case", case_ids)

    try:
        detail = api_get(f"/cases/{selected}")
    except Exception as exc:
        st.error(f"Could not load audit data for `{selected}`: {exc}")
        return

    case = detail.get("case", {})
    events = detail.get("events", [])
    approvals = detail.get("approvals", [])

    st.markdown(f"### Case `{selected}` (Status: **{case.get('status')}**)")
    st.markdown(f"**Customer Statement:** *\"{case.get('complaint_text_redacted')}\"*")

    render_timeline_view(events, approvals)


# ---------------------------------------------------------------------------
# Page 5: Policy Search (RAG)
# ---------------------------------------------------------------------------

def render_policy_search() -> None:
    """Render the policy knowledge base semantic search."""
    st.title("🔍 RBI & Internal Bank Policy Search")
    st.caption("Semantic retrieval over official RBI Master Directions and synthetic bank operating procedures using local embeddings.")

    with st.form("policy_search_form"):
        query = st.text_input(
            "Enter policy inquiry or complaint phrase:",
            value="What is the compensation and turnaround time if an ATM debits an account but dispenses no cash?",
            help="Queries the local Chroma vector database using nomic-embed-text."
        )
        submitted = st.form_submit_button("🔎 Search Knowledge Base", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Please enter a query to search.")
        else:
            with st.spinner("Searching Chroma vector store with local embeddings..."):
                try:
                    res = api_post("/policy-search", {"query": query.strip()})
                    results = res.get("results", [])
                    st.markdown(f"Found **{len(results)}** relevant policy passage(s):")

                    for doc in results:
                        with st.container():
                            st.markdown(f"##### 📖 {doc.get('citation_id', 'POL')} · {doc.get('title', 'Policy')}")
                            st.caption(f"**Type:** `{doc.get('source_type')}` | **Effective Date:** `{doc.get('effective_date')}`")
                            st.markdown(f"> {doc.get('content', '')}")
                            url = doc.get("source_url", "#")
                            st.markdown(f"🔗 [Open Reference Document]({url})")
                            st.divider()

                    if not results:
                        st.info("No relevant passages found. Check that `python -m src.rag.ingest` has been run.")
                except Exception as exc:
                    st.error(f"Policy search failed: {exc}. Ensure the backend API and Ollama embedding model are operational.")


# ---------------------------------------------------------------------------
# Page 6: ML Evaluation & Benchmark
# ---------------------------------------------------------------------------

def render_evaluation() -> None:
    """Render machine learning evaluation metrics, reports, and confusion matrix."""
    st.title("📈 ML Complaint Intent Classifier Evaluation")
    st.caption("Evaluation benchmarks on PolyAI/BANKING77 and supplementary Indian complaint datasets.")

    metrics_path = ROOT / "artifacts" / "ml" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.markdown("### Official BANKING77 Held-out Test Split")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        c2.metric("Macro-F1", f"{metrics.get('macro_f1', 0):.3f}")
        c3.metric("Test Rows", metrics.get("test_rows", "—"))
        c4.metric("Classes", metrics.get("n_train_labels", "—"))

        st.caption(
            "⚠️ **Evaluation Limitation Notice:** Values are strictly measured on the external PolyAI/BANKING77 dataset. "
            "High benchmark performance on general English banking does not guarantee equivalent accuracy on real Indian ATM dispute narratives."
        )

        with st.expander("Detailed Training & Evaluation Report", expanded=False):
            st.json(metrics)

        # Confusion Matrix if exists
        cm_path = ROOT / "artifacts" / "ml" / "confusion_matrix.csv"
        if cm_path.exists():
            with st.expander("Confusion Matrix (Sample)", expanded=False):
                cm_df = pd.read_csv(cm_path, index_col=0)
                st.dataframe(cm_df.iloc[:15, :15], use_container_width=True)
    else:
        st.info("No recorded classifier metrics found. Run `python scripts/download_banking77.py` and `python -m src.ml.train` to generate benchmark artifacts.")

    # Check for calibrated benchmark v2
    v2_path = ROOT / "artifacts" / "ml" / "benchmark_v2_metrics.json"
    if v2_path.exists():
        st.divider()
        st.markdown("### Calibrated Classifier Benchmark (v2)")
        v2_metrics = json.loads(v2_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Validation-Selected Model", v2_metrics.get("selected_model", "—"))
        c2.metric("Brier Score (Calibration)", f"{v2_metrics.get('multiclass_brier_score', 0):.4f}")
        c3.metric("Macro-F1", f"{v2_metrics.get('macro_f1', 0):.3f}")
        with st.expander("v2 Benchmark Details", expanded=False):
            st.json(v2_metrics)
