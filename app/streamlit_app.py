"""ATMResolve AI — Main Streamlit Application.

Single unified navigation interface for ATM dispute investigation,
evidence verification, human review gate, policy search, and ML metrics.
"""
from __future__ import annotations

import streamlit as st
from app.ui_components import (
    render_sidebar_health,
    render_overview,
    render_new_investigation,
    render_case_review,
    render_audit_log_page,
    render_policy_search,
    render_evaluation,
)

st.set_page_config(
    page_title="ATMResolve AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render persistent sidebar system health monitor
render_sidebar_health()

# Unified multi-page navigation replacing duplicate navigation
pages = {
    "Operations": [
        st.Page(render_overview, title="Overview & Workflow", icon="📊", default=True),
        st.Page(render_new_investigation, title="New Investigation", icon="➕"),
        st.Page(render_case_review, title="Case Review & Actions", icon="📋"),
        st.Page(render_audit_log_page, title="Audit Trail & Timeline", icon="📜"),
    ],
    "Intelligence & Reference": [
        st.Page(render_policy_search, title="Policy Search (RAG)", icon="🔍"),
        st.Page(render_evaluation, title="ML Evaluation", icon="📈"),
    ],
}

pg = st.navigation(pages)
pg.run()
