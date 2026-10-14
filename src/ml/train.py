from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.config import ROOT, ML_MODEL_PATH


def train_model(train_path: str | Path, test_path: str | Path, output_dir: str | Path) -> dict:
    train_path, test_path, output_dir = Path(train_path), Path(test_path), Path(output_dir)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("BANKING77 CSV files missing. Run: python scripts/download_banking77.py")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(train_df.columns) or not required_columns.issubset(test_df.columns):
        raise ValueError("Each CSV must contain text,label columns. Re-run scripts/download_banking77.py.")
    train_df = train_df.dropna(subset=["text", "label"])
    test_df = test_df.dropna(subset=["text", "label"])
    if train_df.empty or test_df.empty:
        raise ValueError("Training and test CSVs must each contain at least one labelled example.")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1200, C=4.0, solver="lbfgs", random_state=42)),
    ])
    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(str))
    pred = pipeline.predict(test_df["text"].astype(str))
    labels = sorted(set(test_df["label"].astype(str)) | set(pred.astype(str)))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "intent_classifier.joblib"
    joblib.dump(pipeline, model_path)
    metrics = {
        "dataset": "PolyAI/BANKING77",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_train_labels": int(train_df["label"].nunique()),
        "accuracy": float(accuracy_score(test_df["label"], pred)),
        "macro_f1": float(f1_score(test_df["label"], pred, average="macro", zero_division=0)),
        "model": "TF-IDF(1,2) + LogisticRegression",
        "model_path": str(model_path),
        "note": "Metrics measured on official dataset test split. Not a guarantee of performance on Indian ATM complaint data.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(test_df["label"].astype(str), pred.astype(str), output_dict=True, zero_division=0)
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm = confusion_matrix(test_df["label"].astype(str), pred.astype(str), labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    return metrics


def main() -> None:
    result = train_model(ROOT / "data/banking77_train.csv", ROOT / "data/banking77_test.csv", ML_MODEL_PATH.parent)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
