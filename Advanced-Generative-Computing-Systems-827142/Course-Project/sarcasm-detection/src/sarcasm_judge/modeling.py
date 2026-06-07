from __future__ import annotations

from pathlib import Path


def load_tokenizer_and_model(model_name_or_path: str | Path, num_labels: int = 2):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers and torch before training or inference."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_name_or_path),
        num_labels=num_labels,
        id2label={0: "non_sarcastic", 1: "sarcastic"},
        label2id={"non_sarcastic": 0, "sarcastic": 1},
    )
    return tokenizer, model
