"""Download and normalize the supplementary Indian-style bank complaint dataset.

Source: https://huggingface.co/datasets/KunalEsM/bank_complaint_classifier
The dataset should not be assumed to be real production-customer data; inspect its
provenance, duplicates and labels before using it for evaluation or deployment.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URLS = [
    "https://huggingface.co/datasets/KunalEsM/bank_complaint_classifier/resolve/main/combined_data.csv?download=true",
    "https://hf-mirror.com/datasets/KunalEsM/bank_complaint_classifier/resolve/main/combined_data.csv",
]
OUTPUT = ROOT / "data" / "indian_bank_complaints.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().casefold().replace(" ", "_"): column for column in df.columns}
    text_candidates = ("text", "complaint", "complaint_text", "query", "sentence", "message")
    label_candidates = ("label", "category", "intent", "class", "complaint_type")
    text_col = next((lookup[name] for name in text_candidates if name in lookup), None)
    label_col = next((lookup[name] for name in label_candidates if name in lookup), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not identify text and label columns in {list(df.columns)}. "
            "Inspect the downloaded CSV manually; do not guess which columns contain labels."
        )
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out["text"] = out["text"].astype("string").str.strip()
    out["label"] = out["label"].astype("string").str.strip()
    out = out.dropna(subset=["text", "label"])
    out = out[(out["text"] != "") & (out["label"] != "")]
    return out.reset_index(drop=True)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ATMResolveAI-dataset-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            df = pd.read_csv(BytesIO(payload))
            normalized = _normalize_columns(df)
            if len(normalized) < 100:
                raise ValueError(f"Downloaded only {len(normalized)} valid rows; refusing to save an unexpected dataset.")
            normalized.to_csv(OUTPUT, index=False, encoding="utf-8")
            print(f"Saved {len(normalized):,} rows to {OUTPUT}")
            print(f"Text rows: {normalized['text'].nunique():,} unique / {len(normalized):,} total")
            print(f"Labels: {normalized['label'].nunique():,}")
            print("Next: python scripts/inspect_indian_bank_complaints.py")
            print("Note: this is supplementary data, not a verified real-customer ATM-dispute benchmark.")
            return 0
        except Exception as exc:  # show alternate URL error if needed
            last_error = exc
    print(f"Dataset download failed: {last_error}", file=sys.stderr)
    print("Open the dataset page manually and download combined_data.csv:", file=sys.stderr)
    print("https://huggingface.co/datasets/KunalEsM/bank_complaint_classifier/tree/main", file=sys.stderr)
    print(f"Save it as: {OUTPUT}", file=sys.stderr)
    print("Then normalize it to two columns named text,label before running the inspection script.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
