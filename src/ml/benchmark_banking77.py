"""Compare two TF-IDF baselines on validation, then calibrate the winner.

This improves the *general English banking baseline*. It does NOT make the model
India-specific by itself. The official BANKING77 test split is used only to report
metrics for the validation-selected approach, not to choose between approaches.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

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
TRAIN_PATH = ROOT / "data" / "banking77_train.csv"
TEST_PATH = ROOT / "data" / "banking77_test.csv"
OUTPUT_DIR = ROOT / "artifacts" / "ml"


def make_word_model() -> Pipeline:
    return Pipeline([
        ("features", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, strip_accents="unicode", max_features=150_000,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000, C=4.0, solver="lbfgs", random_state=42,
        )),
    ])


def make_word_char_model() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, strip_accents="unicode", max_features=150_000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=200_000,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000, C=4.0, solver="lbfgs", random_state=42,
        )),
    ])


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run: python scripts/download_banking77.py")
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{path.name} must contain text,label columns")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    return df.reset_index(drop=True)


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def benchmark() -> dict:
    train_df = _read_csv(TRAIN_PATH)
    test_df = _read_csv(TEST_PATH)

    # Select the feature design on a validation split from the official training data.
    fit_df, valid_df = train_test_split(
        train_df, test_size=0.20, random_state=42, stratify=train_df["label"],
    )
    candidates = {
        "word_tfidf_logistic_regression": make_word_model,
        "word_plus_character_tfidf_logistic_regression": make_word_char_model,
    }
    validation_results: dict[str, dict] = {}
    for name, factory in candidates.items():
        print(f"Training validation candidate: {name}", flush=True)
        model = factory()
        model.fit(fit_df["text"].tolist(), fit_df["label"].tolist())
        predictions = model.predict(valid_df["text"].tolist()).astype(str).tolist()
        validation_results[name] = _metrics(valid_df["label"].astype(str).tolist(), predictions)
        print(json.dumps({"model": name, **validation_results[name]}, indent=2), flush=True)

    selected_name = max(validation_results, key=lambda name: validation_results[name]["macro_f1"])
    factory = candidates[selected_name]
    print(f"Selected by validation macro-F1: {selected_name}", flush=True)

    # Fit a calibrated version on only the official training split.
    # Three-fold sigmoid calibration uses training data, not the official test split.
    calibrated = CalibratedClassifierCV(estimator=factory(), method="sigmoid", cv=3)
    calibrated.fit(train_df["text"].tolist(), train_df["label"].tolist())
    calibrated._atmresolve_calibrated = True

    y_true = test_df["label"].astype(str).tolist()
    y_pred = calibrated.predict(test_df["text"].tolist()).astype(str).tolist()
    proba = calibrated.predict_proba(test_df["text"].tolist())
    classes = [str(value) for value in calibrated.classes_]
    one_hot = np.zeros_like(proba)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    for row_idx, label in enumerate(y_true):
        if label in class_to_index:
            one_hot[row_idx, class_to_index[label]] = 1.0
    multiclass_brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    labels = sorted(set(y_true) | set(y_pred))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "intent_classifier_v2_calibrated.joblib"
    joblib.dump(calibrated, model_path)
    final_metrics = {
        "training_data": "PolyAI/BANKING77 official training split",
        "test_data": "PolyAI/BANKING77 official test split",
        "model_selected_using": "20% stratified validation split from official training split",
        "validation_results": validation_results,
        "selected_model": selected_name,
        "calibration": "CalibratedClassifierCV(method='sigmoid', cv=3) fit on official training split",
        "test_rows": int(len(test_df)),
        "test_classes": int(test_df["label"].nunique()),
        **_metrics(y_true, y_pred),
        "multiclass_brier_score": multiclass_brier,
        "model_path": str(model_path),
        "important_limit": (
            "These are English BANKING77 metrics. Calibration is measured for this distribution; "
            "it does not establish calibrated confidence on Indian-English, Hinglish, Marathi, "
            "or ATM-specific customer complaints."
        ),
    }
    (OUTPUT_DIR / "benchmark_v2_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    (OUTPUT_DIR / "benchmark_v2_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUTPUT_DIR / "benchmark_v2_confusion_matrix.csv")
    print("\nFINAL TEST METRICS (reported after selection on validation only):")
    print(json.dumps(final_metrics, indent=2))
    return final_metrics


def main() -> int:
    try:
        benchmark()
        return 0
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
