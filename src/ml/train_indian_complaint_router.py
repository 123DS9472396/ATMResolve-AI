"""Train a separate broad Indian banking-grievance router.

This model is NOT the fine-grained ATM intent classifier. It uses the 12 broad
labels from the supplementary complaint dataset as supplied. Exact duplicates
are de-duplicated, and normalized texts with conflicting labels are excluded.
The dataset looks templated; its held-out metrics measure performance on this
source distribution and must not be presented as real-customer accuracy.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "indian_bank_complaints.csv"
OUT_DIR = ROOT / "artifacts" / "ml"
RANDOM_STATE = 42


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def make_model(c: float) -> Pipeline:
    combined_features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            sublinear_tf=True, strip_accents="unicode", max_features=120_000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            sublinear_tf=True, max_features=150_000,
        )),
    ])
    return Pipeline([
        ("features", combined_features),
        ("classifier", LogisticRegression(
            C=c, max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE,
        )),
    ])


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def multiclass_brier(y_true: list[str], probabilities: np.ndarray, classes: np.ndarray) -> float:
    class_index = {str(label): i for i, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        one_hot[row, class_index[str(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def train() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/indian_bank_complaints.csv. Run scripts/download_indian_bank_complaints.py first."
        )
    data = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Expected columns named text and label in data/indian_bank_complaints.csv")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["label"] != "")].copy()
    data["norm_text"] = data["text"].map(normalize_text)
    data = data[data["norm_text"] != ""].copy()

    labels_by_text = data.groupby("norm_text")["label"].nunique()
    conflicting_texts = set(labels_by_text[labels_by_text > 1].index)
    conflict_count = len(conflicting_texts)
    data = data[~data["norm_text"].isin(conflicting_texts)].copy()
    rows_after_conflict_exclusion = int(len(data))
    before_dedupe = int(len(data))
    data = data.drop_duplicates(subset=["norm_text"], keep="first").copy()
    duplicates_removed = before_dedupe - int(len(data))
    if data["label"].nunique() < 2:
        raise ValueError("Need at least two classes after duplicate/conflict cleaning.")
    label_counts = data["label"].value_counts()
    if int(label_counts.min()) < 10:
        raise ValueError("At least 10 non-conflicting unique texts per label are required for this 3-way split.")

    # Separate final test before any tuning.
    train_val_df, test_df = train_test_split(
        data, test_size=0.20, random_state=RANDOM_STATE, stratify=data["label"]
    )
    train_df, validation_df = train_test_split(
        train_val_df, test_size=0.20, random_state=RANDOM_STATE, stratify=train_val_df["label"]
    )

    candidates = []
    best_model = None
    best_c = None
    best_macro_f1 = -1.0
    for c in (1.0, 2.0, 4.0, 8.0):
        print(f"Validation trial: word+character TF-IDF + LogisticRegression C={c}", flush=True)
        model = make_model(c)
        model.fit(train_df["text"].tolist(), train_df["label"].tolist())
        pred = model.predict(validation_df["text"].tolist()).astype(str).tolist()
        metrics = {"C": c, **score(validation_df["label"].astype(str).tolist(), pred)}
        candidates.append(metrics)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_c = c
            best_model = model

    # Refit the selected model on train+validation, then calibrate probabilities
    # with CV entirely inside train+validation. The final test remains untouched.
    fit_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_model = CalibratedClassifierCV(estimator=make_model(float(best_c)), method="sigmoid", cv=3)
    final_model.fit(fit_df["text"].tolist(), fit_df["label"].tolist())
    final_model._atmresolve_calibrated = True
    y_true = test_df["label"].astype(str).tolist()
    y_pred = final_model.predict(test_df["text"].tolist()).astype(str).tolist()
    proba = final_model.predict_proba(test_df["text"].tolist())
    classes = np.asarray(final_model.classes_)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "indian_complaint_router.joblib"
    joblib.dump(final_model, model_path)
    final_metrics = {
        "task": "broad 12-label Indian banking grievance routing (not fine-grained ATM classification)",
        "source": "KunalEsM/bank_complaint_classifier combined_data.csv",
        "original_rows_after_basic_validation": int(len(pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]))),
        "conflicting_normalized_text_groups_excluded": conflict_count,
        "rows_after_conflict_exclusion_before_deduplication": rows_after_conflict_exclusion,
        "duplicate_rows_removed_after_conflict_exclusion": int(duplicates_removed),
        "clean_unique_rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "label_count": int(data["label"].nunique()),
        "labels": sorted(data["label"].unique().astype(str).tolist()),
        "validation_candidates": candidates,
        "selected_C_by_validation_macro_f1": float(best_c),
        "test_metrics": score(y_true, y_pred),
        "test_multiclass_brier_score": multiclass_brier(y_true, proba, classes),
        "calibration": "sigmoid CalibratedClassifierCV, cv=3, fitted only on train+validation split",
        "model_path": str(model_path),
        "critical_limitations": [
            "The dataset contains templated/synthetic-looking text; provenance does not establish real customer representativeness.",
            "Exact duplicate and conflicting-label leakage is addressed, but near-duplicate template leakage may remain.",
            "Test metrics estimate performance on this dataset distribution only; do not claim them as real Indian customer or ATM failure accuracy.",
            "This model's 12 broad complaint labels must not replace the 77-label BANKING77 model or the future fine-grained ATM model.",
        ],
    }
    (OUT_DIR / "indian_complaint_router_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    report = classification_report(y_true, y_pred, labels=sorted(data["label"].unique()), output_dict=True, zero_division=0)
    (OUT_DIR / "indian_complaint_router_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    labels = sorted(data["label"].unique())
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(
        OUT_DIR / "indian_complaint_router_confusion_matrix.csv", encoding="utf-8-sig"
    )
    print(json.dumps(final_metrics, indent=2))
    print("\nSaved a separate broad-grievance model. The active ATMResolve classifier was not overwritten.")
    return final_metrics


def main() -> int:
    try:
        train()
        return 0
    except Exception as exc:
        print(f"Indian complaint router training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
