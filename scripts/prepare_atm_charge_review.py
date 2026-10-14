"""Create a human-review queue for ATM-related charge complaints.

This script DOES NOT create training labels automatically. Keyword flags are
only suggestions. A human must review text/label alignment before examples are
approved for the fine-grained ATM intent dataset.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "indian_bank_complaints.csv"
OUTPUT = ROOT / "data" / "atm_charge_review_queue.csv"
SUMMARY = ROOT / "data" / "atm_charge_review_summary.json"
ATM_LABEL_PATTERN = re.compile(r"ATM RELATED.*Dispute in ATM AMC Charges", re.IGNORECASE)

PATTERNS = {
    "amc_annual_maintenance": re.compile(
        r"\batm\s*amc\b|\bamc\b|annual\s+maintenance|atm\s+maintenance|maintenance\s+charge",
        re.IGNORECASE,
    ),
    "card_renewal_replacement": re.compile(
        r"card\s+renewal|renewal\s+(?:fee|charge|of\s+(?:my\s+)?(?:atm\s+)?card)|replacement\s+(?:card|fee|charge)|new\s+card\s+(?:fee|charge)",
        re.IGNORECASE,
    ),
    "atm_withdrawal_or_usage_fee": re.compile(
        r"atm\s+withdrawal\s+fee|withdrawal\s+(?:fee|fees|charge|charges)|atm\s+usage\s+fee|balance\s+inquir(?:y|ies)\s+(?:fee|charge)",
        re.IGNORECASE,
    ),
    "failed_transaction_or_debit": re.compile(
        r"failed\s+transaction|cash\s+(?:was\s+)?not\s+dispensed|did\s+not\s+dispense|account\s+was\s+debited|account\s+debited|withdrawal\s+i\s+did\s+not\s+make|unauthori[sz]ed\s+withdrawal",
        re.IGNORECASE,
    ),
}


def norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def main() -> int:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run: python scripts/download_indian_bank_complaints.py"
        )
    df = pd.read_csv(INPUT)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    mask = df["label"].map(lambda value: bool(ATM_LABEL_PATTERN.search(value)))
    subset = df.loc[mask, ["text", "label"]].copy()
    if subset.empty:
        raise ValueError("No ATM AMC-labelled rows found. Inspect the dataset labels before continuing.")

    subset.insert(0, "source_row_number", subset.index + 2)  # CSV header is row 1.
    subset["normalized_text_for_duplicate_check"] = subset["text"].map(norm)
    label_counts = subset.groupby("normalized_text_for_duplicate_check")["label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    subset["same_text_conflicting_source_labels"] = subset["normalized_text_for_duplicate_check"].isin(conflicts)
    subset["exact_duplicate_count_in_atm_subset"] = subset.groupby("normalized_text_for_duplicate_check")["text"].transform("size")

    for name, pattern in PATTERNS.items():
        subset[f"keyword_flag_{name}"] = subset["text"].map(lambda value: bool(pattern.search(value)))

    def subtype(row: pd.Series) -> str:
        # Multiple keyword flags may be true. This priority is for queue sorting only.
        if row["keyword_flag_failed_transaction_or_debit"]:
            return "possible_failed_transaction_or_withdrawal_dispute_review"
        if row["keyword_flag_card_renewal_replacement"]:
            return "card_renewal_or_replacement_fee_review"
        if row["keyword_flag_atm_withdrawal_or_usage_fee"]:
            return "atm_withdrawal_or_usage_fee_review"
        if row["keyword_flag_amc_annual_maintenance"]:
            return "atm_amc_or_annual_maintenance_review"
        return "manual_subtype_review"

    subset["suggested_subtype_for_review"] = subset.apply(subtype, axis=1)
    subset["candidate_broad_intent"] = "atm_fee_dispute_candidate_only"
    subset["review_status"] = "PENDING_HUMAN_REVIEW"
    subset["approved_training_label"] = ""
    subset["reviewer_notes"] = ""
    subset["training_eligible_after_review"] = False

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep the original text and source label, but omit the helper normalization column from the CSV.
    subset.drop(columns=["normalized_text_for_duplicate_check"]).to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )
    summary = {
        "input_file": str(INPUT),
        "output_file": str(OUTPUT),
        "atm_amc_label_rows": int(len(subset)),
        "unique_normalized_texts": int(subset["exact_duplicate_count_in_atm_subset"].eq(1).sum()
                                         + subset.loc[subset["exact_duplicate_count_in_atm_subset"] > 1,
                                                       "normalized_text_for_duplicate_check"].nunique()),
        "rows_with_exact_duplicate_text": int(subset["exact_duplicate_count_in_atm_subset"].gt(1).sum()),
        "unique_texts_with_conflicting_source_labels": int(len(conflicts)),
        "keyword_flag_counts_not_mutually_exclusive": {
            name: int(subset[f"keyword_flag_{name}"].sum()) for name in PATTERNS
        },
        "suggested_subtype_counts_not_ground_truth": {
            str(key): int(value) for key, value in subset["suggested_subtype_for_review"].value_counts().items()
        },
        "safety_note": (
            "No examples were approved or relabelled automatically. Keyword flags are a manual-review aid, "
            "not model training labels. Do not map a broad charge category to cash-not-dispensed or another narrow intent."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nReview queue created. Open data/atm_charge_review_queue.csv in Excel.")
    print("Only after human review should approved rows be copied into data/atm_intents_india.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
