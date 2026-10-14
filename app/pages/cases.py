from __future__ import annotations
import httpx
import streamlit as st
from src.config import API_BASE_URL

st.set_page_config(page_title="Cases · ATMResolve AI", layout="wide")
st.title("Case review")
base = API_BASE_URL.rstrip("/")
try:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{base}/cases")
        response.raise_for_status()
        cases = response.json()
    if not cases:
        st.info("No cases yet. Start the API and create a synthetic case from the main dashboard.")
    else:
        ids = [c["case_id"] for c in cases]
        selected = st.selectbox("Case ID", ids)
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{base}/cases/{selected}")
            response.raise_for_status()
            detail = response.json()
        case = detail["case"]
        st.write("**Status:**", case.get("status"))
        st.write("**Complaint:**", case.get("complaint_text_redacted"))
        st.json(case.get("result", {}))
        st.subheader("Investigation events")
        st.dataframe(detail.get("events", []), use_container_width=True, hide_index=True)
        st.subheader("Approvals")
        st.dataframe(detail.get("approvals", []), use_container_width=True, hide_index=True)
except Exception as exc:
    st.error(f"Could not reach API at {base}. Start it with `uvicorn api.main:app --reload`. Details: {exc}")
