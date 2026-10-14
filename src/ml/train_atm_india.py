"""Train a fine-grained India-focused ATM intent model from manually reviewed data.

Input schema in data/atm_intents_india_template.csv:
    text,label,split,source
Allowed split values: train, validation, test.
Do not put near-duplicate paraphrases of the same template into different splits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "atm_intents_india.csv"
OUTPUT_DIR = ROOT / "artifacts" / "ml"
REQUIRED_COLUMNS = {"text", "label", "split"}
VALID_SPLITS = {"train", "validation", "test"}


def make_model(c: float = 4.0) -> Pipeline:
    features = FeatureUnion([
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
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2500, C=c, solver="lbfgs", class_weight="balanced", random_state=42,
        )),
    ])


def _normalize_for_leak_check(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def _score(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Create data/atm_intents_india.csv from data/atm_intents_india_template.csv, "
            "then add reviewed, labelled examples and explicit train/validation/test split values."
        )
    df = pd.read_csv(DATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}. Required: text,label,split; source is recommended.")
    df = df.dropna(subset=["text", "label", "split"]).copy()
    for col in ("text", "label", "split"):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "") & (df["split"] != "")]
    df["split"] = df["split"].str.casefold()
    unknown_splits = sorted(set(df["split"]) - VALID_SPLITS)
    if unknown_splits:
        raise ValueError(f"Invalid split value(s): {unknown_splits}. Use train, validation, or test.")
    if set(df["split"]) != VALID_SPLITS:
        raise ValueError("Dataset must contain all three explicit splits: train, validation, test.")
    df["normalized_for_check"] = df["text"].map(_normalize_for_leak_check)
    duplicated = df.groupby("normalized_for_check")["split"].nunique()
    leakage = duplicated[duplicated > 1]
    if not leakage.empty:
        examples = df[df["normalized_for_check"].isin(leakage.index)][["text", "label", "split"]].head(8)
        raise ValueError(
            "Exact/whitespace/case-only duplicates cross dataset splits. Remove or reassign them before training.\n"
            + examples.to_string(index=False)
        )
    if df.duplicated(subset=["normalized_for_check", "label"]).any():
        print("WARNING: repeated text exists inside at least one split; review duplicates before trusting metrics.", file=sys.stderr)

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df["label"].nunique() < 2:
        raise ValueError("Training split must contain at least two classes.")
    train_labels = set(train_df["label"])
    if not set(valid_df["label"]).issubset(train_labels) or not set(test_df["label"]).issubset(train_labels):
        raise ValueError("Validation/test contains a label not seen in train. Add train examples for every label.")
    absent_validation = sorted(train_labels - set(valid_df["label"]))
    absent_test = sorted(train_labels - set(test_df["label"]))
    if absent_validation or absent_test:
        raise ValueError(
            f"Every class needs validation and test examples. Missing in validation={absent_validation}; test={absent_test}"
        )

    train_counts = train_df["label"].value_counts().sort_index()
    low = train_counts[train_counts < 20]
    if not low.empty:
        print("WARNING: fewer than 20 training examples for some classes; accuracy/confidence may be unstable:")
        print(low.to_string())

    best_c, best_f1 = None, -1.0
    validation_results = []
    for c in (1.0, 2.0, 4.0, 8.0):
        print(f"Training C={c} on labelled India-specific ATM data...", flush=True)
        candidate = make_model(c)
        candidate.fit(train_df["text"].tolist(), train_df["label"].tolist())
        pred = candidate.predict(valid_df["text"].tolist()).astype(str).tolist()
        metrics = {"C": c, **_score(valid_df["label"].astype(str).tolist(), pred)}
        validation_results.append(metrics)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_c = metrics["macro_f1"], c

    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    final_base = make_model(float(best_c))
    class_counts = combined_df["label"].value_counts()
    # Calibrate only when each class has enough examples for three-fold calibration.
    if len(class_counts) > 1 and int(class_counts.min()) >= 15:
        final_model = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=3)
        final_model.fit(combined_df["text"].tolist(), combined_df["label"].tolist())
        final_model._atmresolve_calibrated = True
        calibration_note = "sigmoid calibration using 3-fold CV on train+validation; still requires real-world validation"
    else:
        final_model = final_base
        final_model.fit(combined_df["text"].tolist(), combined_df["label"].tolist())
        final_model._atmresolve_calibrated = False
        calibration_note = "not calibrated: at least 15 train+validation examples per class are required by this script"

    y_true = test_df["label"].astype(str).tolist()
    y_pred = final_model.predict(test_df["text"].tolist()).astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "atm_india_classifier.joblib"
    joblib.dump(final_model, model_path)
    metrics = {
        "task": "fine-grained India-focused ATM complaint intent classification",
        "rows_total": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "classes": sorted(train_labels),
        "train_examples_per_class": {str(k): int(v) for k, v in train_counts.items()},
        "selected_C_by_validation_macro_f1": float(best_c),
        "validation_results": validation_results,
        "test_metrics": _score(y_true, y_pred),
        "calibration": calibration_note,
        "confidence_note": "A confidence score is not a guarantee of correctness; validate separately on representative Indian customer language.",
        "dataset_note": "Human-review all labels. Synthetic examples are useful for prototyping but cannot establish real-customer performance.",
        "model_path": str(model_path),
    }
    (OUTPUT_DIR / "atm_india_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    (OUTPUT_DIR / "atm_india_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "atm_india_confusion_matrix.csv"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    try:
        train()
        return 0
    except Exception as exc:
        print(f"ATM-India training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
