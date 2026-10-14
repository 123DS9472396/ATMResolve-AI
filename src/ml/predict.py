from __future__ import annotations
from pathlib import Path
import joblib
from src.config import ML_MODEL_PATH

_cached_model = None
_cached_path: Path | None = None


def load_model(path: str | Path | None = None):
    global _cached_model, _cached_path
    resolved = Path(path) if path is not None else ML_MODEL_PATH
    if not resolved.is_absolute():
        # Relative explicit paths are interpreted from the project root.
        from src.config import ROOT
        resolved = ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Classifier not found at {resolved}. Train/download the relevant dataset first.")
    if _cached_model is None or _cached_path != resolved:
        _cached_model = joblib.load(resolved)
        _cached_path = resolved
    return _cached_model


def predict_intent(text: str, path: str | Path | None = None) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Complaint text cannot be empty")
    model = load_model(path)
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return {
        "intent_label": str(model.classes_[index]),
        "confidence_score": float(probabilities[index]),
        "confidence_is_calibrated": bool(getattr(model, "_atmresolve_calibrated", False)),
    }
