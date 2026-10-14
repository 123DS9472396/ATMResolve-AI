"""Case review page for ATMResolve AI."""
from __future__ import annotations

import streamlit as st
from app.ui_components import render_sidebar_health, render_case_review

st.set_page_config(page_title="Case Review · ATMResolve AI", page_icon="📋", layout="wide")
render_sidebar_health()
render_case_review()
