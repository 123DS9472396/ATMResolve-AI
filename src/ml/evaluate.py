"""Evaluation helpers for the complaint-intent classifier."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def _load_test_csv(test_csv: str | Path) -> tuple[list[str], list[str]]:
    path = Path(test_csv)
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")
    frame = pd.read_csv(path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Test CSV must contain text,label columns.")
    frame = frame.dropna(subset=["text", "label"])
    return frame["text"].astype(str).tolist(), frame["label"].astype(str).tolist()

def evaluate_model(model, test_csv: str | Path) -> dict:
    """Return metrics for a trained model on a labelled CSV split."""
    texts, labels = _load_test_csv(test_csv)
    predictions = [str(value) for value in model.predict(texts)]
    class_names = sorted(set(labels) | set(predictions))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_names).tolist(),
        "labels": class_names,
        "classification_report": classification_report(labels, predictions, labels=class_names, zero_division=0, output_dict=True),
        "examples": len(labels),
    }
