from __future__ import annotations
import pandas as pd
from src.ml.evaluate import evaluate_model
from src.ml.predict import load_model, predict_intent
from src.ml.train import train_model

def test_train_predict_evaluate_round_trip(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    rows = [
        {"text": "cash withdrawal failed at the ATM", "label": "cash_withdrawal"},
        {"text": "cash withdrawal did not work at ATM", "label": "cash_withdrawal"},
        {"text": "bank transfer was declined", "label": "transfer_failed"},
        {"text": "my bank transfer did not go through", "label": "transfer_failed"},
        {"text": "cash withdrawal failed at ATM again", "label": "cash_withdrawal"},
        {"text": "transfer payment is being declining", "label": "transfer_failed"},
    ]
    pd.DataFrame(rows).to_csv(train_path, index=False)
    pd.DataFrame(rows).to_csv(test_path, index=False)
    out_dir = tmp_path / "artifacts"
    train_model(train_path, test_path, out_dir)
    model = load_model(out_dir / "intent_classifier.joblib")
    result = predict_intent("cash withdrawal failed at the ATM", out_dir / "intent_classifier.joblib")
    metrics = evaluate_model(model, test_path)
    assert result["intent_label"] == "cash_withdrawal"
    assert 0 <= result["confidence_score"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert metrics["examples"] == len(rows)
    assert (out_dir / "metrics.json").exists()

def test_training_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame([{"message": "missing schema"}]).to_csv(path, index=False)
    try:
        train_model(path, path, tmp_path / "artifacts")
    except ValueError as error:
        assert "text,label" in str(error)
    else:
        raise AssertionError("Expected invalid dataset schema to fail")


def test_predict_intent_returns_calibrated_flag(tmp_path):
    """predict_intent() must always return a 'confidence_is_calibrated' key.

    The graph reads this key to propagate the flag into case_event logs;
    it must be present and be a bool on any trained model — calibrated or not.
    """
    train_path = tmp_path / "train.csv"
    rows = [
        {"text": "cash not dispensed but account debited", "label": "cash_not_dispensed"},
        {"text": "ATM swallowed my card", "label": "card_retained"},
        {"text": "cash not dispensed and no receipt", "label": "cash_not_dispensed"},
        {"text": "card was kept by ATM machine", "label": "card_retained"},
    ]
    pd.DataFrame(rows).to_csv(train_path, index=False)
    out_dir = tmp_path / "artifacts"
    # train_model produces an uncalibrated pipeline (no CalibratedClassifierCV)
    train_model(train_path, train_path, out_dir)
    result = predict_intent("ATM did not give cash", out_dir / "intent_classifier.joblib")
    assert "confidence_is_calibrated" in result, (
        "predict_intent() must return 'confidence_is_calibrated' for graph logging"
    )
    assert isinstance(result["confidence_is_calibrated"], bool)
    # A plain train_model() pipeline has no _atmresolve_calibrated attr,
    # so the flag should be False.
    assert result["confidence_is_calibrated"] is False
