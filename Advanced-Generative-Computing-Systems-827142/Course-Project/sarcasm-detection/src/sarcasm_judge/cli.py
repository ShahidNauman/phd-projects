from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import prepare_amazon_corpus, prepare_csv_dataset, prepare_hf_dataset
from .inference import predict_texts
from .train import evaluate_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="sarcasm_judge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    csv_parser = subparsers.add_parser(
        "prepare-csv", help="Normalize and split a CSV dataset."
    )
    csv_parser.add_argument("csv_path", type=Path)
    csv_parser.add_argument("output_dir", type=Path)
    csv_parser.add_argument("--text-column", default="text")
    csv_parser.add_argument("--label-column", default="label")
    csv_parser.add_argument("--group-column", default=None)
    csv_parser.add_argument("--seed", type=int, default=42)

    amazon_parser = subparsers.add_parser(
        "prepare-amazon", help="Normalize and split the Amazon sarcasm corpus."
    )
    amazon_parser.add_argument("corpus_path", type=Path)
    amazon_parser.add_argument("output_dir", type=Path)
    amazon_parser.add_argument("--seed", type=int, default=42)

    hf_parser = subparsers.add_parser(
        "prepare-hf", help="Normalize and split a Hugging Face dataset."
    )
    hf_parser.add_argument("dataset_name")
    hf_parser.add_argument("output_dir", type=Path)
    hf_parser.add_argument("--config-name", default=None)
    hf_parser.add_argument("--split", default="train")
    hf_parser.add_argument("--text-column", default="headline")
    hf_parser.add_argument("--label-column", default="is_sarcastic")
    hf_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Fine-tune a sarcasm judge.")
    train_parser.add_argument("--config", type=Path, required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a fine-tuned judge.")
    eval_parser.add_argument("--model-path", type=Path, required=True)
    eval_parser.add_argument("--dataset-dir", type=Path, required=True)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--max-length", type=int, default=192)
    eval_parser.add_argument("--prompt-template", default=None)
    eval_parser.add_argument("--output-json", type=Path, default=None)

    predict_parser = subparsers.add_parser(
        "predict", help="Predict sarcasm for one or more texts."
    )
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--text", action="append", required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--max-length", type=int, default=192)
    predict_parser.add_argument("--prompt-template", default=None)

    args = parser.parse_args()

    if args.command == "prepare-csv":
        prepare_csv_dataset(
            args.csv_path,
            args.output_dir,
            text_column=args.text_column,
            label_column=args.label_column,
            group_column=args.group_column,
            seed=args.seed,
        )
        return

    if args.command == "prepare-amazon":
        prepare_amazon_corpus(args.corpus_path, args.output_dir, seed=args.seed)
        return

    if args.command == "prepare-hf":
        prepare_hf_dataset(
            args.dataset_name,
            args.output_dir,
            config_name=args.config_name,
            split=args.split,
            text_column=args.text_column,
            label_column=args.label_column,
            seed=args.seed,
        )
        return

    if args.command == "train":
        train_model(load_config(args.config))
        return

    if args.command == "evaluate":
        metrics = evaluate_model(
            args.model_path,
            args.dataset_dir,
            args.split,
            args.max_length,
            args.prompt_template,
        )
        print(json.dumps(metrics, indent=2))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    if args.command == "predict":
        predictions = predict_texts(
            args.model_path,
            args.text,
            args.threshold,
            args.max_length,
            args.prompt_template,
        )
        print(json.dumps(predictions, indent=2))
