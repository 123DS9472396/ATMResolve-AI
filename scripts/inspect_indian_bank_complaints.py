"""Audit the Indian-style bank complaint data before any training."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "indian_bank_complaints.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run: python scripts/download_indian_bank_complaints.py")
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    print("=== Dataset audit ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique exact texts: {df['text'].nunique():,}")
    print(f"Exact duplicate rows: {df.duplicated(subset=['text', 'label']).sum():,}")
    print(f"Same text with conflicting labels: {df.groupby('text')['label'].nunique().gt(1).sum():,}")
    print(f"Number of labels: {df['label'].nunique():,}\n")
    print("=== Class balance ===")
    counts = df["label"].value_counts(dropna=False)
    for label, count in counts.items():
        print(f"{count:5d}  {label}")
    print("\n=== Sample ATM / withdrawal / cash-related rows ===")
    mask = df["text"].str.contains(r"atm|withdraw|cash|debit|dispens|reversal|revers|rupee|inr|rs\.?", case=False, regex=True, na=False)
    sample = df.loc[mask, ["label", "text"]].head(25)
    if sample.empty:
        print("No keyword matches found. Review the data manually.")
    else:
        for index, row in sample.iterrows():
            print(f"[{row['label']}] {row['text']}")
    print("\nIMPORTANT: Don't map a broad 'charges deducted' label to a narrow ATM-failure label.")
    print("The label and text must support the exact intent before a row is used in the ATM model.")


if __name__ == "__main__":
    main()
