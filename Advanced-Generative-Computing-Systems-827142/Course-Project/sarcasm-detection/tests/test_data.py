from pathlib import Path

import pandas as pd

from sarcasm_judge.data import normalize_label, prepare_csv_dataset
from sarcasm_judge.templates import format_judge_text


def test_normalize_label_aliases():
    assert normalize_label("ironic") == 1
    assert normalize_label("regular") == 0
    assert normalize_label(True) == 1


def test_prepare_csv_dataset_writes_splits(tmp_path: Path):
    rows = []
    for idx in range(20):
        rows.append({"review": f"text {idx}", "sarcasm": idx % 2, "pair": f"p{idx}"})
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    output_dir = tmp_path / "processed"
    prepare_csv_dataset(
        csv_path,
        output_dir,
        text_column="review",
        label_column="sarcasm",
        group_column="pair",
        seed=7,
    )

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "metadata.json").exists()


def test_format_judge_text_requires_placeholder():
    assert "hello" in format_judge_text("hello", "Review: {text}")
