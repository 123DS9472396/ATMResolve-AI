from __future__ import annotations
import json
import streamlit as st
from src.config import ROOT

st.set_page_config(page_title="ML Evaluation · ATMResolve AI", layout="wide")
st.title("ML evaluation")
path = ROOT / "artifacts" / "ml" / "metrics.json"
if not path.exists():
    st.info("No measured metrics found yet. Run `python scripts/download_banking77.py` and then `python -m src.ml.train`.")
else:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    b.metric("Macro-F1", f"{metrics.get('macro_f1', 0):.3f}")
    c.metric("Test rows", str(metrics.get("test_rows", "—")))
    st.json(metrics)
    st.caption("These values come from the recorded run on BANKING77, not a guarantee of performance on Indian ATM complaint text.")
