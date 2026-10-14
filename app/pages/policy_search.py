"""Policy search page for ATMResolve AI."""
from __future__ import annotations

import streamlit as st
from app.ui_components import render_sidebar_health, render_policy_search

st.set_page_config(page_title="Policy Search · ATMResolve AI", page_icon="🔍", layout="wide")
render_sidebar_health()
render_policy_search()
