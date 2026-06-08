# Sarcasm Detection Judge Fine-Tuning

Implementation of the methodology from _LLM-as-a-judge for sarcasm detection using supervised fine-tuning_.

The project builds a binary sarcasm judge from review/headline datasets, creates grouped train/validation/test splits, fine-tunes a Hugging Face sequence-classification model, and reports paper-style metrics.

## Quick Start

Create and activate the virtual environment: _(Optional)_

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Prepare a real CSV dataset with at least `text` and `label` columns. The path below is only an example; the file must exist before you run the command.

```bash
mkdir data/raw
# Put your CSV at data/raw/reviews.csv first.
python -m sarcasm_judge prepare-csv data/raw/reviews.csv data/processed/reviews
```

Example CSV format:

```csv
text,label,group_id
"Great, another update that broke everything.",1,review-001
"Wonderful, the charger died on day two.",1,review-002
"Exactly what I wanted: a manual with no instructions.",1,review-003
"The battery lasts a long time.",0,review-004
"The packaging was secure and delivery was fast.",0,review-005
"The headphones connect quickly and sound clear.",0,review-006
```

For actual fine-tuning, use many more rows. The two labels must both have enough examples to create train, validation, and test splits.

Fine-tune:

```bash
python -m sarcasm_judge train --config configs/reviews_distilbert.yaml
```

Evaluate:

```bash
python -m sarcasm_judge evaluate --model-path runs/reviews-distilbert/best --dataset-dir data/processed/reviews --split test
```

Predict:

```bash
python -m sarcasm_judge predict --model-path runs/reviews-distilbert/best --text "Great, another update that broke everything."
```

## Methodology Coverage

- Binary sarcasm judge labels: `0 = non-sarcastic`, `1 = sarcastic`
- Dataset normalization for Amazon reviews, CSV input, and optional Hugging Face datasets
- Group-aware train/validation/test split to keep paired reviews out of multiple splits
- 192-token truncation by default, matching the paper setup
- Supervised fine-tuning with Hugging Face `Trainer`
- Label smoothing, weight decay, cosine scheduler, warmup ratio, and macro-F1 model selection
- Accuracy, precision, recall, macro-F1, weighted-F1, confusion matrix, and JSON reports
- Probability-based inference with configurable decision threshold

## Local Amazon Corpus

The paper uses the Amazon sarcasm review corpus. If you use `SarcasmAmazonReviewsCorpus-master.zip`, extract it first. The archive contains nested `.rar` files; Python can read them only if `rarfile` and an `unrar`/`bsdtar` backend are installed.

Prepare dataset:

```bash
python -m sarcasm_judge prepare-amazon data/raw/amazon data/processed/amazon
```

Fine-tune:

```bash
python -m sarcasm_judge train --config configs/amazon_distilbert.yaml
```

Evaluate:

```bash
python -m sarcasm_judge evaluate --model-path runs/amazon-distilbert/best --dataset-dir data/processed/amazon --split test
```

Predict:

```bash
python -m sarcasm_judge predict --model-path runs/amazon-distilbert/best --text "Great, another update that broke everything."
```

## Configuration

Edit files in `configs/`. The default model is `distilbert-base-uncased-finetuned-sst-2-english` because it is practical for coursework hardware. The same pipeline supports larger judge backbones such as DeBERTa, Llama, or Mistral if the local environment has the required memory and dependencies.
