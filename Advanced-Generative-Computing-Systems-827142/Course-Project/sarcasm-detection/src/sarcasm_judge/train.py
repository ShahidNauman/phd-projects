from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import compute_classification_metrics, trainer_metrics
from .modeling import load_tokenizer_and_model
from .templates import DEFAULT_PROMPT_TEMPLATE, format_judge_text


def train_model(config: dict) -> None:
    try:
        from datasets import Dataset, DatasetDict
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "Install training dependencies from requirements.txt before training."
        ) from exc

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    dataset_dir = Path(data_config["dataset_dir"])
    max_length = int(data_config.get("max_length", 192))
    prompt_template = data_config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    tokenizer, model = load_tokenizer_and_model(
        model_config["name_or_path"], int(model_config.get("num_labels", 2))
    )

    dataset = DatasetDict(
        {
            split: Dataset.from_pandas(
                _load_split(dataset_dir, split), preserve_index=False
            )
            for split in ("train", "validation", "test")
        }
    )

    def tokenize(batch):
        texts = [format_judge_text(text, prompt_template) for text in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    keep_columns = {"input_ids", "attention_mask", "labels"}
    drop_columns = [
        column
        for column in tokenized["train"].column_names
        if column not in keep_columns
    ]
    tokenized = tokenized.remove_columns(drop_columns)

    args = TrainingArguments(
        output_dir=training_config["output_dir"],
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(
            training_config.get("per_device_train_batch_size", 16)
        ),
        per_device_eval_batch_size=int(
            training_config.get("per_device_eval_batch_size", 32)
        ),
        num_train_epochs=float(training_config.get("num_train_epochs", 4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.06)),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        label_smoothing_factor=float(
            training_config.get("label_smoothing_factor", 0.1)
        ),
        metric_for_best_model=training_config.get("metric_for_best_model", "macro_f1"),
        greater_is_better=bool(training_config.get("greater_is_better", True)),
        eval_strategy=training_config.get("eval_strategy")
        or training_config.get("evaluation_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        logging_steps=int(training_config.get("logging_steps", 50)),
        load_best_model_at_end=bool(
            training_config.get("load_best_model_at_end", True)
        ),
        report_to=training_config.get("report_to", "none"),
        seed=seed,
    )

    import inspect

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "compute_metrics": trainer_metrics,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    best_dir = Path(training_config["output_dir"]) / "best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    (Path(training_config["output_dir"]) / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )


def evaluate_model(
    model_path: Path,
    dataset_dir: Path,
    split: str = "test",
    max_length: int = 192,
    prompt_template: str | None = None,
) -> dict[str, object]:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding
    except ImportError as exc:
        raise RuntimeError(
            "Install inference dependencies from requirements.txt before evaluation."
        ) from exc

    tokenizer, model = load_tokenizer_and_model(model_path)
    dataset = Dataset.from_pandas(_load_split(dataset_dir, split), preserve_index=False)

    def tokenize(batch):
        texts = [format_judge_text(text, prompt_template) for text in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    labels = tokenized["label"]
    tokenized = tokenized.remove_columns(
        [
            column
            for column in tokenized.column_names
            if column not in {"input_ids", "attention_mask"}
        ]
    )
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(tokenized, batch_size=32, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(
                np.argmax(logits.detach().cpu().numpy(), axis=-1).tolist()
            )
    return compute_classification_metrics(labels, predictions)


def _load_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_json(path, lines=True)
