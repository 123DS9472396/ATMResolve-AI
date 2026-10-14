from pathlib import Path
import pytest
from src.ml.train import train_model

def test_train_raises_helpful_error_if_dataset_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="BANKING77 CSV files missing"):
        train_model(tmp_path / "missing_train.csv", tmp_path / "missing_test.csv", tmp_path / "out")
