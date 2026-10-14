from __future__ import annotations
import httpx
import streamlit as st
from src.config import API_BASE_URL

st.set_page_config(page_title="Audit Log · ATMResolve AI", layout="wide")
st.title("Audit log")
base = API_BASE_URL.rstrip("/")
try:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{base}/cases")
        response.raise_for_status()
        cases = response.json()
    if not cases:
        st.info("No case events yet.")
    else:
        ids = [c["case_id"] for c in cases]
        selected = st.selectbox("Case", ids)
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{base}/cases/{selected}")
            response.raise_for_status()
            detail = response.json()
        st.subheader("Workflow events")
        st.dataframe(detail.get("events", []), use_container_width=True, hide_index=True)
        st.subheader("Human decisions")
        st.dataframe(detail.get("approvals", []), use_container_width=True, hide_index=True)
except Exception as exc:
    st.error(f"Could not load audit data from {base}. Start the API first. Details: {exc}")
