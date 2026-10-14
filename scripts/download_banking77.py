
from pathlib import Path

import pandas as pd

from src.config import ROOT


TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/train.csv"
)

TEST_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/test.csv"
)


def download_split(url: str, output_path: Path) -> int:
    """Download a BANKING77 CSV and normalize its column names."""
    print(f"Downloading {output_path.name}...")

    df = pd.read_csv(url)

    required_columns = {"text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Unexpected dataset columns: {list(df.columns)}"
        )

    df = df[["text", "category"]].rename(
        columns={"category": "label"}
    )

    df = df.dropna(subset=["text", "label"])

    if df.empty:
        raise ValueError(f"No valid rows downloaded from {url}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(df):,} rows to {output_path}")
    return len(df)


def main() -> None:
    data_dir = ROOT / "data"

    train_count = download_split(
        TRAIN_URL,
        data_dir / "banking77_train.csv",
    )

    test_count = download_split(
        TEST_URL,
        data_dir / "banking77_test.csv",
    )

    print("\nBANKING77 download complete.")
    print(f"Training examples: {train_count:,}")
    print(f"Test examples:     {test_count:,}")
    print("Columns: text, label")
    print("Labels are readable intent names.")


if __name__ == "__main__":
    main()
