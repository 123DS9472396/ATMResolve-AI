"""Audit trail and investigation timeline page for ATMResolve AI."""
from __future__ import annotations

import streamlit as st
from app.ui_components import render_sidebar_health, render_audit_log_page

st.set_page_config(page_title="Audit Trail · ATMResolve AI", page_icon="📜", layout="wide")
render_sidebar_health()
render_audit_log_page()
