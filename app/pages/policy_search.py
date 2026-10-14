from __future__ import annotations
import httpx
import streamlit as st
from src.config import API_BASE_URL

st.set_page_config(page_title="Policy Search · ATMResolve AI", layout="wide")
st.title("Policy search")
st.caption("Search the local RAG index. Results are evidence snippets, not automatic policy decisions.")
base = API_BASE_URL.rstrip("/")
with st.form("policy_search"):
    query = st.text_input("Question", "What should be checked when an ATM debits an account but dispenses no cash?")
    submitted = st.form_submit_button("Search", type="primary")
if submitted:
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{base}/policy-search", json={"query": query})
            response.raise_for_status()
            data = response.json()
        if not data.get("results"):
            st.warning("No passages were retrieved. Check that ingestion has run and the query is supported by the documents.")
        for doc in data.get("results", []):
            st.markdown(f"**{doc.get('citation_id')} · {doc.get('title')}**")
            st.caption(f"{doc.get('source_type')} · {doc.get('effective_date')}")
            st.write(doc.get("content", ""))
            st.markdown(f"[Open source]({doc.get('source_url', '#')})")
            st.divider()
    except Exception as exc:
        st.error(f"Policy search failed. Check the API, Ollama and Chroma index. Details: {exc}")
